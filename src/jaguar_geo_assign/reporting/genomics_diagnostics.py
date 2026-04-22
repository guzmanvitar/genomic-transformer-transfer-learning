"""Genomics corpus diagnostics and quality-assurance reporting.

Provides single-pass streaming summarisation of consensus and baseline genomic
window records, near-duplicate detection with bounded heap sampling, positional
missingness heatmaps, and split-leakage auditing.  All public entry-points
accept lazy iterables so that corpora larger than RAM can be processed without
materialising the full record set.

Typical usage::

    summary = summarize_corpus_records(records)
    comparison = compare_reference_baseline(consensus, baseline)
    write_eda_payload_json(consensus, baseline, "diagnostics.json")
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from hashlib import sha256
from heapq import heapreplace, heappush, nsmallest
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

# The four standard DNA nucleotide bases used to identify canonical positions.
CANONICAL_BASES = frozenset({"A", "C", "G", "T"})
# Bases representing missing or undetermined calls in the sequence.
MISSING_BASES = frozenset({"N"})
# Minimum field set required for per-sample summarisation (no locus metadata).
SAMPLE_REQUIRED_FIELDS = frozenset(
    {
        "sample_id",
        "sequence",
        "reference_sequence",
        "variant_count",
        "callable_bases",
        "unique_masked_bases",
        "filtered_bases",
        "no_call_bases",
        "token_count",
    }
)
# Extended field set for corpus-level summarisation; adds locus, split, and source.
CORPUS_REQUIRED_FIELDS = SAMPLE_REQUIRED_FIELDS | frozenset({"locus_id", "split", "source"})
# Upper bound on sampled records for the O(S×L) near-duplicate signature expansion.
DEFAULT_NEAR_DUPLICATE_SAMPLE_LIMIT = 512
# Maximum number of per-sample previews retained in the EDA payload.
DEFAULT_CONSENSUS_SAMPLE_LIMIT = 128
# Default number of equal-width bins for fraction-value histograms.
DEFAULT_DISTRIBUTION_BINS = 10
# Derived metric names computed by _summarize_record and aggregated corpus-wide.
FRACTION_FIELD_NAMES = (
    "gc_fraction",
    "ambiguity_fraction",
    "callable_fraction",
    "filtered_fraction",
    "no_call_fraction",
    "other_masked_fraction",
    "variant_fraction",
    "fraction_identical_to_reference",
    "token_to_base_ratio",
)


def summarize_sample_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return per-record observability metrics for consensus or baseline sequences.

    Each input record is passed through ``_summarize_record`` with the
    ``SAMPLE_REQUIRED_FIELDS`` contract, enriching it with derived fractions
    (GC content, ambiguity, callable coverage, etc.) and validating that
    ``sequence`` and ``reference_sequence`` are aligned to the same length.

    Args:
        records: Iterable of raw sample-level mappings.  Each mapping must
            contain at least the keys in ``SAMPLE_REQUIRED_FIELDS``.

    Returns:
        A list of enriched record dicts, one per input record, augmented with
        computed fraction fields and normalised sequences.

    Raises:
        ValueError: If any record is missing required fields, contains
            negative counts, or has misaligned sequence lengths.
    """
    return [_summarize_record(record, SAMPLE_REQUIRED_FIELDS) for record in records]


def summarize_corpus_records(
    records: Iterable[Mapping[str, object]],
    *,
    near_duplicate_sample_limit: int | None = DEFAULT_NEAR_DUPLICATE_SAMPLE_LIMIT,
) -> dict[str, object]:
    """Summarize corpus-level diversity, missingness, and duplication diagnostics.

    Delegates to ``_stream_corpus_summary`` for single-pass streaming
    aggregation, returning only the summary dict and discarding per-sample
    previews.

    Args:
        records: Lazy iterable of raw corpus records.  Each mapping must
            contain at least the keys in ``CORPUS_REQUIRED_FIELDS``.
        near_duplicate_sample_limit: Maximum number of records fed to the
            O(S×L) near-duplicate signature expansion.  ``None`` runs an
            exact (unbounded) pass over the full corpus.

    Returns:
        A single dict of corpus-wide aggregate metrics including mean
        fractions, duplication counts, length distribution, missingness
        heatmap, and near-duplicate analysis.
    """
    summary, _, _ = _stream_corpus_summary(
        records,
        near_duplicate_sample_limit=near_duplicate_sample_limit,
    )
    return summary


def compare_reference_baseline(
    consensus_records: Iterable[Mapping[str, object]],
    baseline_records: Iterable[Mapping[str, object]],
    *,
    near_duplicate_sample_limit: int | None = DEFAULT_NEAR_DUPLICATE_SAMPLE_LIMIT,
) -> dict[str, object]:
    """Compare consensus-derived corpus diagnostics against a reference-only baseline.

    Summarises both the consensus and baseline corpora independently, then
    computes signed deltas for key quality indicators (ambiguity, variant
    density, duplication, tokenisation ratio).

    Args:
        consensus_records: Lazy iterable of consensus-derived corpus records.
        baseline_records: Lazy iterable of reference-only corpus records.
        near_duplicate_sample_limit: Bound on the near-duplicate sample size
            applied identically to both corpora.

    Returns:
        A dict with keys ``"consensus"``, ``"baseline"``, and ``"deltas"``
        produced by ``_build_reference_baseline_comparison``.
    """
    consensus_summary = summarize_corpus_records(
        consensus_records,
        near_duplicate_sample_limit=near_duplicate_sample_limit,
    )
    baseline_summary = summarize_corpus_records(
        baseline_records,
        near_duplicate_sample_limit=near_duplicate_sample_limit,
    )

    return _build_reference_baseline_comparison(consensus_summary, baseline_summary)


def write_eda_payload_json(
    consensus_records: Iterable[Mapping[str, object]],
    baseline_records: Iterable[Mapping[str, object]],
    output_path: str | Path,
    *,
    near_duplicate_sample_limit: int | None = DEFAULT_NEAR_DUPLICATE_SAMPLE_LIMIT,
    consensus_sample_limit: int | None = DEFAULT_CONSENSUS_SAMPLE_LIMIT,
) -> Path:
    """Persist a notebook-friendly diagnostics payload as JSON.

    Builds the full EDA payload via ``build_eda_payload`` and writes it to
    *output_path*, creating parent directories as needed.

    Args:
        consensus_records: Lazy iterable of consensus-derived corpus records.
        baseline_records: Lazy iterable of reference-only corpus records.
        output_path: Filesystem destination for the JSON file.
        near_duplicate_sample_limit: Bound on the near-duplicate sample size.
        consensus_sample_limit: Maximum number of per-sample previews
            included in the payload.

    Returns:
        The resolved ``Path`` to the written JSON file.
    """
    payload = build_eda_payload(
        consensus_records,
        baseline_records,
        near_duplicate_sample_limit=near_duplicate_sample_limit,
        consensus_sample_limit=consensus_sample_limit,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def _build_reference_baseline_comparison(
    consensus_summary: Mapping[str, object],
    baseline_summary: Mapping[str, object],
) -> dict[str, object]:
    """Assemble a side-by-side comparison of consensus vs. baseline summaries.

    Produces a dict containing the full consensus summary, the full baseline
    summary, and a ``"deltas"`` sub-dict with signed differences for the
    headline quality indicators (ambiguity, variant density, duplication,
    near-duplicate counts, and tokenisation ratio).

    Args:
        consensus_summary: Corpus-level summary dict for the consensus corpus.
        baseline_summary: Corpus-level summary dict for the baseline corpus.

    Returns:
        A dict with ``"consensus"``, ``"baseline"``, and ``"deltas"`` keys.
    """
    return {
        "consensus": dict(consensus_summary),
        "baseline": dict(baseline_summary),
        "deltas": {
            "retained_window_count": (
                consensus_summary["retained_window_count"] - baseline_summary["retained_window_count"]
            ),
            "ambiguity_fraction": round(
                consensus_summary["mean_ambiguity_fraction"]
                - baseline_summary["mean_ambiguity_fraction"],
                6,
            ),
            "variant_fraction": round(
                consensus_summary["mean_variant_fraction"]
                - baseline_summary["mean_variant_fraction"],
                6,
            ),
            "duplicate_window_fraction": round(
                consensus_summary["duplicate_window_fraction"]
                - baseline_summary["duplicate_window_fraction"],
                6,
            ),
            "near_duplicate_pair_count": (
                consensus_summary["near_duplicate_pair_count"]
                - baseline_summary["near_duplicate_pair_count"]
            ),
            "near_duplicate_pair_fraction": round(
                consensus_summary["near_duplicate_pair_fraction"]
                - baseline_summary["near_duplicate_pair_fraction"],
                6,
            ),
            "token_to_base_ratio": round(
                consensus_summary["mean_token_to_base_ratio"]
                - baseline_summary["mean_token_to_base_ratio"],
                6,
            ),
        },
    }


def build_missingness_heatmap(
    records: Iterable[Mapping[str, object]], bins: int = 8
) -> list[float]:
    """Aggregate N-base missingness by relative position across multiple sequences.

    Divides each sequence into *bins* equal-width positional buckets and
    computes the fraction of ``N`` bases in each bucket, pooled across all
    input sequences.

    Args:
        records: Iterable of mappings each containing a ``"sequence"`` key.
        bins: Number of equal-width positional buckets.  Must be positive.

    Returns:
        A list of *bins* floats in ``[0.0, 1.0]`` representing the pooled
        N-base fraction per positional bucket.

    Raises:
        ValueError: If *bins* is not positive, or a sequence is empty.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")

    missing_counts = [0] * bins
    total_counts = [0] * bins
    for record in records:
        sequence = _normalize_sequence(str(record["sequence"]))
        for index, base in enumerate(sequence):
            bucket = min((index * bins) // len(sequence), bins - 1)
            total_counts[bucket] += 1
            if base in MISSING_BASES:
                missing_counts[bucket] += 1

    return [
        round(_safe_fraction(missing_counts[index], total_counts[index]), 6)
        for index in range(bins)
    ]


def audit_corpus_integrity(
    records: Iterable[Mapping[str, object]], *, summarized: bool = False
) -> dict[str, list[dict[str, object]]]:
    """Report split-leakage and basic shape-contract issues for corpus records.

    Detects two classes of integrity violations:

    * **Split conflicts** — a single ``locus_id`` appearing in more than one
      data split, indicating potential train/test leakage.
    * **Shape issues** — records where
      ``callable_bases + unique_masked_bases != len(sequence)`` (the
      ``_has_unique_coverage_mismatch`` invariant), or where ``token_count``
      is non-positive.

    Args:
        records: Iterable of corpus-level record mappings.
        summarized: If ``True``, *records* are assumed to have already been
            enriched by ``_summarize_record``; otherwise each record is
            summarised on the fly with ``CORPUS_REQUIRED_FIELDS``.

    Returns:
        A dict with ``"split_conflicts"`` (list of locus/split pairs) and
        ``"shape_issues"`` (list of per-record violation dicts).
    """
    summarized_records = list(records) if summarized else [
        _summarize_record(record, CORPUS_REQUIRED_FIELDS) for record in records
    ]
    locus_splits: dict[str, set[str]] = defaultdict(set)
    shape_issues: list[dict[str, object]] = []

    for record in summarized_records:
        locus_splits[str(record["locus_id"])].add(str(record["split"]))
        sequence = _normalize_sequence(str(record["sequence"]))
        if _has_unique_coverage_mismatch(
            sequence_length=len(sequence),
            callable_bases=int(record["callable_bases"]),
            unique_masked_bases=int(record["unique_masked_bases"]),
        ):
            shape_issues.append(
                {
                    "sample_id": record["sample_id"],
                    "locus_id": record["locus_id"],
                    "issue": "status_counts_do_not_match_sequence_length",
                }
            )
        if int(record["token_count"]) <= 0:
            shape_issues.append(
                {
                    "sample_id": record["sample_id"],
                    "locus_id": record["locus_id"],
                    "issue": "token_count_must_be_positive",
                }
            )

    split_conflicts = [
        {"locus_id": locus_id, "splits": sorted(splits)}
        for locus_id, splits in sorted(locus_splits.items())
        if len(splits) > 1
    ]
    return {"split_conflicts": split_conflicts, "shape_issues": shape_issues}


def build_eda_payload(
    consensus_records: Iterable[Mapping[str, object]],
    baseline_records: Iterable[Mapping[str, object]],
    *,
    near_duplicate_sample_limit: int | None = DEFAULT_NEAR_DUPLICATE_SAMPLE_LIMIT,
    consensus_sample_limit: int | None = DEFAULT_CONSENSUS_SAMPLE_LIMIT,
) -> dict[str, object]:
    """Build a notebook-friendly payload with per-sample and corpus-level summaries.

    Combines per-sample previews (capped to *consensus_sample_limit*),
    corpus-level summaries for both consensus and baseline, and a
    side-by-side baseline comparison into one JSON-serialisable dict.

    Args:
        consensus_records: Lazy iterable of consensus-derived corpus records.
        baseline_records: Lazy iterable of reference-only corpus records.
        near_duplicate_sample_limit: Bound on the near-duplicate sample size.
        consensus_sample_limit: Maximum number of per-sample previews
            included in the ``"consensus_samples"`` key.

    Returns:
        A dict with keys ``"consensus_samples"``,
        ``"consensus_sample_overview"``, ``"consensus_corpus"``,
        ``"baseline_corpus"``, and ``"baseline_comparison"``.
    """
    consensus_summary, consensus_preview, consensus_total_count = _stream_corpus_summary(
        consensus_records,
        near_duplicate_sample_limit=near_duplicate_sample_limit,
        sample_preview_limit=consensus_sample_limit,
    )
    baseline_summary = summarize_corpus_records(
        baseline_records,
        near_duplicate_sample_limit=near_duplicate_sample_limit,
    )
    return {
        "consensus_samples": consensus_preview,
        "consensus_sample_overview": {
            "total_record_count": consensus_total_count,
            "returned_record_count": len(consensus_preview),
            "sample_limit": consensus_sample_limit,
            "truncated": consensus_total_count > len(consensus_preview),
        },
        "consensus_corpus": consensus_summary,
        "baseline_corpus": baseline_summary,
        "baseline_comparison": _build_reference_baseline_comparison(
            consensus_summary,
            baseline_summary,
        ),
    }


def _summarize_record(
    record: Mapping[str, object], required_fields: frozenset[str]
) -> dict[str, object]:
    """Enrich a raw record with normalised sequences and derived fraction metrics.

    **Alignment contract (FRAGILITY FLAG):** ``sequence`` and
    ``reference_sequence`` must be the same length after normalisation
    (upper-cased, whitespace-stripped).  A length mismatch raises
    ``ValueError`` because all downstream per-position comparisons
    (identity-to-reference, missingness heatmaps) assume position-wise
    alignment between the two strings.

    The returned dict is a shallow copy of the original record with
    overwritten normalised sequences and additional keys for every entry
    in ``FRACTION_FIELD_NAMES``.

    Args:
        record: Raw record mapping containing at least *required_fields*.
        required_fields: Set of keys the record must contain.  Validated
            via ``_validate_required_fields`` before any computation.

    Returns:
        A new dict merging the original record with normalised sequences,
        coerced integer counts, and computed fraction fields (all rounded
        to six decimal places).

    Raises:
        ValueError: If required fields are missing, integer counts are
            negative, or the two sequences differ in length.
    """
    _validate_required_fields(record, required_fields)
    sequence = _normalize_sequence(str(record["sequence"]))
    reference_sequence = _normalize_sequence(str(record["reference_sequence"]))
    if len(sequence) != len(reference_sequence):
        raise ValueError("sequence and reference_sequence must have identical lengths")

    variant_count = _coerce_nonnegative_int(record, "variant_count")
    callable_bases = _coerce_nonnegative_int(record, "callable_bases")
    unique_masked_bases = _coerce_nonnegative_int(record, "unique_masked_bases")
    filtered_bases = _coerce_nonnegative_int(record, "filtered_bases")
    no_call_bases = _coerce_nonnegative_int(record, "no_call_bases")
    other_masked_bases = _coerce_optional_nonnegative_int(record, "other_masked_bases")
    token_count = _coerce_nonnegative_int(record, "token_count")
    masked_base_counts = _coerce_masked_base_counts(
        record,
        filtered_bases=filtered_bases,
        no_call_bases=no_call_bases,
        other_masked_bases=other_masked_bases,
    )
    canonical_bases = sum(base in CANONICAL_BASES for base in sequence)

    return {
        **dict(record),
        "sequence": sequence,
        "reference_sequence": reference_sequence,
        "unique_masked_bases": unique_masked_bases,
        "masked_base_counts": masked_base_counts,
        "gc_fraction": round(
            _safe_fraction(sum(base in {"G", "C"} for base in sequence), canonical_bases), 6
        ),
        "ambiguity_fraction": round(_safe_fraction(len(sequence) - canonical_bases, len(sequence)), 6),
        "callable_fraction": round(_safe_fraction(callable_bases, len(sequence)), 6),
        "filtered_fraction": round(_safe_fraction(filtered_bases, len(sequence)), 6),
        "no_call_fraction": round(_safe_fraction(no_call_bases, len(sequence)), 6),
        "other_masked_fraction": round(_safe_fraction(other_masked_bases, len(sequence)), 6),
        "variant_fraction": round(_safe_fraction(variant_count, len(sequence)), 6),
        "fraction_identical_to_reference": round(
            _safe_fraction(
                sum(base == reference for base, reference in zip(sequence, reference_sequence)),
                len(sequence),
            ),
            6,
        ),
        "token_to_base_ratio": round(_safe_fraction(token_count, len(sequence)), 6),
    }


def _count_near_duplicate_pairs(sequences: list[str]) -> int:
    """Count one-mismatch pairs across the analyzed sequence subset.

    The signature expansion below is intentionally exact, but its work scales with the
    analyzed sample size (`S`) and sequence length (`L`). Callers keep `S` bounded via
    `near_duplicate_sample_limit` for corpus-scale diagnostics unless an exact full-corpus
    pass is explicitly requested with `None`.
    """
    signatures: dict[tuple[int, str], set[int]] = defaultdict(set)
    for index, sequence in enumerate(sequences):
        for position in range(len(sequence)):
            signature = sequence[:position] + "*" + sequence[position + 1 :]
            signatures[(len(sequence), signature)].add(index)

    candidate_pairs = {
        pair
        for indexes in signatures.values()
        if len(indexes) > 1
        for pair in combinations(sorted(indexes), 2)
    }
    return sum(1 for left, right in candidate_pairs if sequences[left] != sequences[right])


def _summarize_near_duplicates(
    records: list[dict[str, object]],
    *,
    sample_limit: int | None,
    total_sequence_count: int | None = None,
) -> dict[str, object]:
    """Summarize near-duplicate pairs for either the full set or a bounded sample.

    When the corpus was streamed via ``_stream_corpus_summary``, *records*
    is the pre-sampled heap output and *total_sequence_count* reflects the
    true corpus size.  Otherwise this function performs its own
    ``nsmallest`` sampling when *sample_limit* is exceeded.

    Args:
        records: List of enriched record dicts to analyse.
        sample_limit: Maximum records for the O(S×L) near-duplicate check.
            ``None`` means analyse all *records* exactly.
        total_sequence_count: True corpus size when *records* is already
            a pre-sampled subset.  ``None`` means ``len(records)`` is the
            true count.

    Returns:
        A dict with ``"pair_count"``, ``"pair_fraction"``, and
        ``"analysis"`` (mode, counts, limit metadata).

    Raises:
        ValueError: If *sample_limit* is provided but non-positive.
    """
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("near_duplicate_sample_limit must be positive when provided")

    analyzed_records = records
    total_count = len(records) if total_sequence_count is None else total_sequence_count
    analysis_mode = "exact"
    if sample_limit is not None and total_count > len(records):
        # `_stream_corpus_summary()` already handed us a deterministic bounded sample.
        analysis_mode = "sampled"
    elif sample_limit is not None and len(records) > sample_limit:
        analyzed_records = list(
            nsmallest(sample_limit, records, key=_near_duplicate_sampling_key)
        )
        analysis_mode = "sampled"

    analyzed_sequences = [str(record["sequence"]) for record in analyzed_records]
    pair_count = _count_near_duplicate_pairs(analyzed_sequences)
    analyzed_pair_total = len(analyzed_sequences) * (len(analyzed_sequences) - 1) // 2
    return {
        "pair_count": pair_count,
        "pair_fraction": round(_safe_fraction(pair_count, analyzed_pair_total), 6),
        "analysis": {
            "mode": analysis_mode,
            "total_sequence_count": total_count,
            "analyzed_sequence_count": len(analyzed_sequences),
            "sample_limit": sample_limit,
        },
    }


def _near_duplicate_sampling_key(record: Mapping[str, object]) -> str:
    """Compute a deterministic SHA-256 hex digest for near-duplicate sampling.

    The key is derived from the pipe-delimited concatenation of
    ``sample_id``, ``locus_id``, ``source``, and ``sequence``, ensuring
    that sampling order is reproducible and independent of input order.

    Args:
        record: An enriched record mapping containing the four key fields.

    Returns:
        A 64-character lowercase hexadecimal digest string.
    """
    payload = "|".join(
        str(record[field]) for field in ("sample_id", "locus_id", "source", "sequence")
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _build_length_distribution(records: list[dict[str, object]]) -> list[dict[str, int]]:
    """Build a sorted histogram of sequence lengths across all records.

    Args:
        records: List of enriched record dicts with a ``"sequence"`` key.

    Returns:
        A list of ``{"length": int, "count": int}`` dicts sorted by length.
    """
    return [
        {"length": length, "count": count}
        for length, count in sorted(Counter(len(str(record["sequence"])) for record in records).items())
    ]


def _build_fraction_distribution(
    records: list[dict[str, object]],
    *,
    field: str,
    bins: int = DEFAULT_DISTRIBUTION_BINS,
) -> list[dict[str, float | int]]:
    """Build an equal-width histogram for a ``[0, 1]``-valued fraction field.

    Values are clamped to ``[0.0, 1.0]`` before binning.  The last bin is
    inclusive of 1.0 (``min(int(value * bins), bins - 1)``).

    Args:
        records: List of enriched record dicts.
        field: Name of the fraction field to histogram.
        bins: Number of equal-width bins.  Must be positive.

    Returns:
        A list of *bins* dicts with ``"start"``, ``"end"``, and ``"count"``.

    Raises:
        ValueError: If *bins* is not positive.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")

    counts = [0] * bins
    for record in records:
        value = min(max(float(record[field]), 0.0), 1.0)
        bucket = min(int(value * bins), bins - 1)
        counts[bucket] += 1

    return [
        {
            "start": round(index / bins, 6),
            "end": round((index + 1) / bins, 6),
            "count": counts[index],
        }
        for index in range(bins)
    ]


def _validate_required_fields(record: Mapping[str, object], required_fields: frozenset[str]) -> None:
    """Raise ``ValueError`` listing any keys from *required_fields* absent in *record*.

    Args:
        record: The mapping to validate.
        required_fields: Keys that must be present.

    Raises:
        ValueError: With a sorted comma-separated list of missing field names.
    """
    missing = sorted(field for field in required_fields if field not in record)
    if missing:
        raise ValueError(f"record is missing required fields: {', '.join(missing)}")


def _coerce_nonnegative_int(record: Mapping[str, object], field: str) -> int:
    """Extract *field* from *record* and validate it as a non-negative ``int``.

    Args:
        record: The source mapping.
        field: Key to extract.

    Returns:
        The validated non-negative integer value.

    Raises:
        ValueError: If the value is not an ``int`` or is negative.
    """
    value = record[field]
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _coerce_optional_nonnegative_int(record: Mapping[str, object], field: str) -> int:
    """Extract *field* as a non-negative ``int``, defaulting to ``0`` if absent.

    Args:
        record: The source mapping.
        field: Key to extract.

    Returns:
        The validated non-negative integer, or ``0`` when *field* is missing.
    """
    if field not in record:
        return 0
    return _coerce_nonnegative_int(record, field)


def _coerce_masked_base_counts(
    record: Mapping[str, object],
    *,
    filtered_bases: int,
    no_call_bases: int,
    other_masked_bases: int,
) -> dict[str, int]:
    """Validate and normalise the ``masked_base_counts`` mapping from a record.

    If the record already contains a ``masked_base_counts`` mapping, its
    keys and values are validated (strings and non-negative ints).  Missing
    standard categories (``"filtered"``, ``"no_call"``, ``"other_masked"``)
    are back-filled from the corresponding scalar arguments when non-zero.

    Args:
        record: The source mapping, optionally containing
            ``"masked_base_counts"``.
        filtered_bases: Scalar count of filtered bases for back-fill.
        no_call_bases: Scalar count of no-call bases for back-fill.
        other_masked_bases: Scalar count of other masked bases for back-fill.

    Returns:
        A sorted dict mapping masking categories to their non-negative counts.

    Raises:
        ValueError: If the existing mapping has non-string keys or negative
            integer values.
    """
    raw_value = record.get("masked_base_counts")
    counts: dict[str, int] = {}
    if raw_value is not None:
        if not hasattr(raw_value, "items"):
            raise ValueError("masked_base_counts must be a mapping when provided")
        for category, count in raw_value.items():
            if not isinstance(category, str):
                raise ValueError("masked_base_counts keys must be strings")
            if not isinstance(count, int) or count < 0:
                raise ValueError("masked_base_counts values must be non-negative integers")
            counts[category] = count
    if filtered_bases and "filtered" not in counts:
        counts["filtered"] = filtered_bases
    if no_call_bases and "no_call" not in counts:
        counts["no_call"] = no_call_bases
    if other_masked_bases and not any(category not in {"filtered", "no_call"} for category in counts):
        counts["other_masked"] = other_masked_bases
    return {category: count for category, count in sorted(counts.items())}


def _has_unique_coverage_mismatch(
    *,
    sequence_length: int,
    callable_bases: int,
    unique_masked_bases: int,
) -> bool:
    """Check whether the coverage bookkeeping invariant is violated.

    **Invariant (FRAGILITY FLAG):** every base in the sequence must be
    accounted for as either callable or uniquely masked, so
    ``callable_bases + unique_masked_bases`` must equal ``sequence_length``.
    A mismatch indicates corrupt or inconsistent upstream masking metadata
    and is surfaced as a shape issue by ``audit_corpus_integrity``.

    Args:
        sequence_length: Total number of bases in the normalised sequence.
        callable_bases: Count of bases classified as callable.
        unique_masked_bases: Count of bases classified as uniquely masked.

    Returns:
        ``True`` if the invariant is violated (i.e. counts do not sum to
        the sequence length).
    """
    return callable_bases + unique_masked_bases != sequence_length


def _stream_corpus_summary(
    records: Iterable[Mapping[str, object]],
    *,
    near_duplicate_sample_limit: int | None,
    sample_preview_limit: int | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], int]:
    """Compute corpus-level diagnostics in a single streaming pass.

    **Single-pass streaming (FRAGILITY FLAG):** the input *records*
    iterable is consumed exactly once with bounded auxiliary memory.
    Running accumulators (metric sums, counters, sets) are updated
    per-record.  The near-duplicate sample is maintained as a bounded
    max-heap of size *near_duplicate_sample_limit* via
    ``_update_sampled_near_duplicates``, so worst-case heap memory is
    ``O(near_duplicate_sample_limit)`` regardless of corpus size.

    When *near_duplicate_sample_limit* is ``None``, all records are
    retained for an exact (unbounded) near-duplicate pass.

    Args:
        records: Lazy iterable of raw corpus records consumed once.
        near_duplicate_sample_limit: Upper bound on the near-duplicate
            sample heap.  ``None`` means unbounded (exact analysis).
        sample_preview_limit: Maximum number of per-sample previews
            retained.  ``None`` retains all.

    Returns:
        A 3-tuple of ``(summary_dict, sample_preview_list,
        retained_window_count)``.

    Raises:
        ValueError: If *sample_preview_limit* is provided but non-positive.
    """
    if sample_preview_limit is not None and sample_preview_limit <= 0:
        raise ValueError("consensus_sample_limit must be positive when provided")

    metric_sums = {field: 0.0 for field in FRACTION_FIELD_NAMES}
    sample_preview: list[dict[str, object]] = []
    unique_samples: set[str] = set()
    unique_loci: set[str] = set()
    source_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    masked_category_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    gc_distribution_counts = [0] * DEFAULT_DISTRIBUTION_BINS
    missing_counts = [0] * 8
    total_counts = [0] * 8
    locus_splits: dict[str, set[str]] = defaultdict(set)
    shape_issue_count = 0
    total_base_count = 0
    retained_window_count = 0
    sampled_records: list[dict[str, object]] = []
    sampled_heap: list[tuple[int, int, dict[str, object]]] = []

    for record_index, raw_record in enumerate(records):
        record = _summarize_record(raw_record, CORPUS_REQUIRED_FIELDS)
        retained_window_count += 1
        if sample_preview_limit is None or len(sample_preview) < sample_preview_limit:
            sample_preview.append(record)

        unique_samples.add(str(record["sample_id"]))
        unique_loci.add(str(record["locus_id"]))
        source_counts[str(record["source"])] += 1
        sequence = str(record["sequence"])
        total_base_count += len(sequence)
        duplicate_counts[sequence] += 1
        masked_category_counts.update(dict(record["masked_base_counts"]))
        length_counts[len(sequence)] += 1
        _update_fraction_distribution_counts(
            gc_distribution_counts,
            float(record["gc_fraction"]),
            bins=DEFAULT_DISTRIBUTION_BINS,
        )
        _update_missingness_counts(sequence, missing_counts, total_counts)
        locus_splits[str(record["locus_id"])].add(str(record["split"]))

        if _has_unique_coverage_mismatch(
            sequence_length=len(sequence),
            callable_bases=int(record["callable_bases"]),
            unique_masked_bases=int(record["unique_masked_bases"]),
        ):
            shape_issue_count += 1
        if int(record["token_count"]) <= 0:
            shape_issue_count += 1

        for field in FRACTION_FIELD_NAMES:
            metric_sums[field] += float(record[field])

        if near_duplicate_sample_limit is None:
            sampled_records.append(record)
        else:
            # Keep the exact near-duplicate check bounded to a deterministic sample rather
            # than the full corpus; this avoids an unbounded O(S × L) signature expansion.
            _update_sampled_near_duplicates(
                record, near_duplicate_sample_limit, sampled_heap, record_index
            )

    duplicate_window_count = sum(count - 1 for count in duplicate_counts.values() if count > 1)
    near_duplicate_records = sampled_records if near_duplicate_sample_limit is None else [
        item[2] for item in sorted(sampled_heap, key=lambda item: (item[0], item[1]))
    ]
    near_duplicate_summary = _summarize_near_duplicates(
        near_duplicate_records,
        sample_limit=near_duplicate_sample_limit,
        total_sequence_count=retained_window_count,
    )

    summary = {
        "retained_window_count": retained_window_count,
        "unique_sample_count": len(unique_samples),
        "unique_locus_count": len(unique_loci),
        "mean_gc_fraction": _safe_mean(metric_sums["gc_fraction"], retained_window_count),
        "mean_ambiguity_fraction": _safe_mean(metric_sums["ambiguity_fraction"], retained_window_count),
        "mean_callable_fraction": _safe_mean(metric_sums["callable_fraction"], retained_window_count),
        "mean_filtered_fraction": _safe_mean(metric_sums["filtered_fraction"], retained_window_count),
        "mean_no_call_fraction": _safe_mean(metric_sums["no_call_fraction"], retained_window_count),
        "mean_other_masked_fraction": _safe_mean(
            metric_sums["other_masked_fraction"], retained_window_count
        ),
        "mean_variant_fraction": _safe_mean(metric_sums["variant_fraction"], retained_window_count),
        "mean_fraction_identical_to_reference": _safe_mean(
            metric_sums["fraction_identical_to_reference"], retained_window_count
        ),
        "mean_token_to_base_ratio": _safe_mean(metric_sums["token_to_base_ratio"], retained_window_count),
        "length_distribution": [
            {"length": length, "count": count} for length, count in sorted(length_counts.items())
        ],
        "gc_fraction_distribution": _finalize_fraction_distribution(gc_distribution_counts),
        "duplicate_window_count": duplicate_window_count,
        "duplicate_window_fraction": _safe_fraction(duplicate_window_count, retained_window_count),
        "near_duplicate_pair_count": near_duplicate_summary["pair_count"],
        "near_duplicate_pair_fraction": near_duplicate_summary["pair_fraction"],
        "near_duplicate_analysis": near_duplicate_summary["analysis"],
        "masked_category_base_counts": {
            category: count for category, count in sorted(masked_category_counts.items())
        },
        "masked_category_base_fractions": {
            category: round(_safe_fraction(count, total_base_count), 6)
            for category, count in sorted(masked_category_counts.items())
        },
        "missingness_heatmap": [
            round(_safe_fraction(missing_counts[index], total_counts[index]), 6) for index in range(8)
        ],
        "split_conflict_count": sum(1 for splits in locus_splits.values() if len(splits) > 1),
        "shape_issue_count": shape_issue_count,
        "source_counts": dict(source_counts),
    }
    return summary, sample_preview, retained_window_count


def _update_fraction_distribution_counts(counts: list[int], value: float, *, bins: int) -> None:
    """Increment the appropriate bin in *counts* for a single fraction *value*.

    The value is clamped to ``[0.0, 1.0]`` before binning.  Mutates
    *counts* in place for streaming accumulation.

    Args:
        counts: Mutable list of per-bin counts (length == *bins*).
        value: A fraction value to bin.
        bins: Total number of equal-width bins.
    """
    clamped_value = min(max(value, 0.0), 1.0)
    bucket = min(int(clamped_value * bins), bins - 1)
    counts[bucket] += 1


def _finalize_fraction_distribution(counts: list[int]) -> list[dict[str, float | int]]:
    """Convert raw bin counts into a list of ``start/end/count`` histogram dicts.

    Args:
        counts: List of per-bin counts accumulated during streaming.

    Returns:
        A list of dicts with ``"start"``, ``"end"`` (both rounded to six
        decimal places), and ``"count"`` for each bin.
    """
    bins = len(counts)
    return [
        {
            "start": round(index / bins, 6),
            "end": round((index + 1) / bins, 6),
            "count": counts[index],
        }
        for index in range(bins)
    ]


def _update_missingness_counts(sequence: str, missing_counts: list[int], total_counts: list[int]) -> None:
    """Accumulate per-position N-base missingness for a single sequence.

    Each base is assigned to a positional bucket and the corresponding
    *total_counts* entry is incremented.  If the base is in
    ``MISSING_BASES``, the matching *missing_counts* entry is also
    incremented.  Both lists are mutated in place for streaming use.

    Args:
        sequence: Normalised DNA sequence string.
        missing_counts: Mutable per-bucket missing-base accumulator.
        total_counts: Mutable per-bucket total-base accumulator.
    """
    for index, base in enumerate(sequence):
        bucket = min((index * len(missing_counts)) // len(sequence), len(missing_counts) - 1)
        total_counts[bucket] += 1
        if base in MISSING_BASES:
            missing_counts[bucket] += 1


def _update_sampled_near_duplicates(
    record: dict[str, object],
    sample_limit: int,
    heap: list[tuple[int, int, dict[str, object]]],
    record_index: int,
) -> None:
    """Maintain a deterministic hash-ordered sample for bounded near-duplicate analysis.

    **Bounded near-duplicate heap (FRAGILITY FLAG):** uses a max-heap
    (negated keys) of size *sample_limit* so that only the
    *sample_limit* records with the smallest SHA-256 sampling keys are
    retained.  The heap is keyed by ``(-sampling_key, record_index)``
    to break ties deterministically by insertion order.  When the heap
    is full, a new record replaces the current maximum (largest negated
    key, i.e. smallest original key that is still the ``heap[0]``
    sentinel) only if its key is strictly smaller.

    Args:
        record: Enriched record dict to consider for the sample.
        sample_limit: Maximum heap size.
        heap: Mutable min-heap of ``(-sampling_key, record_index, record)``
            tuples, maintained across calls.
        record_index: Monotonically increasing index for tie-breaking.
    """
    sampling_key = int(_near_duplicate_sampling_key(record), 16)
    entry = (-sampling_key, record_index, record)
    if len(heap) < sample_limit:
        heappush(heap, entry)
        return
    if sampling_key < -heap[0][0]:
        heapreplace(heap, entry)


def _safe_mean(total: float, count: int) -> float:
    """Return ``total / count`` rounded to six places, or ``0.0`` if *count* is zero.

    Args:
        total: Accumulated sum of values.
        count: Number of values summed.

    Returns:
        The rounded mean, or ``0.0`` for an empty population.
    """
    return 0.0 if count == 0 else round(total / count, 6)


def _normalize_sequence(sequence: str) -> str:
    """Upper-case and strip all whitespace from a DNA sequence string.

    Args:
        sequence: Raw sequence string, possibly with mixed case or spaces.

    Returns:
        A contiguous upper-cased string with no whitespace.

    Raises:
        ValueError: If the result is empty after normalisation.
    """
    normalized = "".join(sequence.upper().split())
    if not normalized:
        raise ValueError("sequence must not be empty")
    return normalized


def _safe_fraction(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator``, or ``0.0`` if *denominator* is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor (may be zero).

    Returns:
        The unrounded quotient, or ``0.0`` to avoid ``ZeroDivisionError``.
    """
    return 0.0 if denominator == 0 else numerator / denominator