"""Helper-backed genomics corpus diagnostics and QA."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from hashlib import sha256
from heapq import nsmallest
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping

CANONICAL_BASES = frozenset({"A", "C", "G", "T"})
MISSING_BASES = frozenset({"N"})
SAMPLE_REQUIRED_FIELDS = frozenset(
    {
        "sample_id",
        "sequence",
        "reference_sequence",
        "variant_count",
        "callable_bases",
        "filtered_bases",
        "no_call_bases",
        "token_count",
    }
)
CORPUS_REQUIRED_FIELDS = SAMPLE_REQUIRED_FIELDS | frozenset({"locus_id", "split", "source"})
DEFAULT_NEAR_DUPLICATE_SAMPLE_LIMIT = 512
DEFAULT_DISTRIBUTION_BINS = 10


def summarize_sample_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return per-record observability metrics for consensus or baseline sequences."""
    return [_summarize_record(record, SAMPLE_REQUIRED_FIELDS) for record in records]


def summarize_corpus_records(
    records: Iterable[Mapping[str, object]],
    *,
    near_duplicate_sample_limit: int | None = DEFAULT_NEAR_DUPLICATE_SAMPLE_LIMIT,
) -> dict[str, object]:
    """Summarize corpus-level diversity, missingness, and duplication diagnostics."""
    materialized = [_summarize_record(record, CORPUS_REQUIRED_FIELDS) for record in records]
    integrity = audit_corpus_integrity(materialized, summarized=True)
    duplicate_window_count = sum(
        count - 1 for count in Counter(str(record["sequence"]) for record in materialized).values() if count > 1
    )
    near_duplicate_summary = _summarize_near_duplicates(
        materialized,
        sample_limit=near_duplicate_sample_limit,
    )

    return {
        "retained_window_count": len(materialized),
        "unique_sample_count": len({record["sample_id"] for record in materialized}),
        "unique_locus_count": len({record["locus_id"] for record in materialized}),
        "mean_gc_fraction": _mean_metric(materialized, "gc_fraction"),
        "mean_ambiguity_fraction": _mean_metric(materialized, "ambiguity_fraction"),
        "mean_callable_fraction": _mean_metric(materialized, "callable_fraction"),
        "mean_filtered_fraction": _mean_metric(materialized, "filtered_fraction"),
        "mean_no_call_fraction": _mean_metric(materialized, "no_call_fraction"),
        "mean_variant_fraction": _mean_metric(materialized, "variant_fraction"),
        "mean_fraction_identical_to_reference": _mean_metric(
            materialized, "fraction_identical_to_reference"
        ),
        "mean_token_to_base_ratio": _mean_metric(materialized, "token_to_base_ratio"),
        "length_distribution": _build_length_distribution(materialized),
        "gc_fraction_distribution": _build_fraction_distribution(
            materialized,
            field="gc_fraction",
        ),
        "duplicate_window_count": duplicate_window_count,
        "duplicate_window_fraction": _safe_fraction(duplicate_window_count, len(materialized)),
        "near_duplicate_pair_count": near_duplicate_summary["pair_count"],
        "near_duplicate_pair_fraction": near_duplicate_summary["pair_fraction"],
        "near_duplicate_analysis": near_duplicate_summary["analysis"],
        "missingness_heatmap": build_missingness_heatmap(materialized),
        "split_conflict_count": len(integrity["split_conflicts"]),
        "shape_issue_count": len(integrity["shape_issues"]),
        "source_counts": dict(Counter(record["source"] for record in materialized)),
    }


def compare_reference_baseline(
    consensus_records: Iterable[Mapping[str, object]],
    baseline_records: Iterable[Mapping[str, object]],
    *,
    near_duplicate_sample_limit: int | None = DEFAULT_NEAR_DUPLICATE_SAMPLE_LIMIT,
) -> dict[str, object]:
    """Compare consensus-derived corpus diagnostics against a reference-only baseline."""
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
) -> Path:
    """Persist a notebook-friendly diagnostics payload as JSON."""
    payload = build_eda_payload(
        consensus_records,
        baseline_records,
        near_duplicate_sample_limit=near_duplicate_sample_limit,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def _build_reference_baseline_comparison(
    consensus_summary: Mapping[str, object],
    baseline_summary: Mapping[str, object],
) -> dict[str, object]:
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
    """Aggregate N-base missingness by relative position across multiple sequences."""
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
    """Report split-leakage and basic shape-contract issues for corpus records."""
    summarized_records = list(records) if summarized else [
        _summarize_record(record, CORPUS_REQUIRED_FIELDS) for record in records
    ]
    locus_splits: dict[str, set[str]] = defaultdict(set)
    shape_issues: list[dict[str, object]] = []

    for record in summarized_records:
        locus_splits[str(record["locus_id"])].add(str(record["split"]))
        observed_bases = (
            int(record["callable_bases"])
            + int(record["filtered_bases"])
            + int(record["no_call_bases"])
        )
        if observed_bases != len(str(record["sequence"])):
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
) -> dict[str, object]:
    """Build a notebook-friendly payload with per-sample and corpus-level summaries."""
    consensus_materialized = list(consensus_records)
    baseline_materialized = list(baseline_records)
    consensus_summary = summarize_corpus_records(
        consensus_materialized,
        near_duplicate_sample_limit=near_duplicate_sample_limit,
    )
    baseline_summary = summarize_corpus_records(
        baseline_materialized,
        near_duplicate_sample_limit=near_duplicate_sample_limit,
    )
    return {
        "consensus_samples": summarize_sample_records(consensus_materialized),
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
    _validate_required_fields(record, required_fields)
    sequence = _normalize_sequence(str(record["sequence"]))
    reference_sequence = _normalize_sequence(str(record["reference_sequence"]))
    if len(sequence) != len(reference_sequence):
        raise ValueError("sequence and reference_sequence must have identical lengths")

    variant_count = _coerce_nonnegative_int(record, "variant_count")
    callable_bases = _coerce_nonnegative_int(record, "callable_bases")
    filtered_bases = _coerce_nonnegative_int(record, "filtered_bases")
    no_call_bases = _coerce_nonnegative_int(record, "no_call_bases")
    token_count = _coerce_nonnegative_int(record, "token_count")
    canonical_bases = sum(base in CANONICAL_BASES for base in sequence)

    return {
        **dict(record),
        "sequence": sequence,
        "reference_sequence": reference_sequence,
        "gc_fraction": round(
            _safe_fraction(sum(base in {"G", "C"} for base in sequence), canonical_bases), 6
        ),
        "ambiguity_fraction": round(_safe_fraction(len(sequence) - canonical_bases, len(sequence)), 6),
        "callable_fraction": round(_safe_fraction(callable_bases, len(sequence)), 6),
        "filtered_fraction": round(_safe_fraction(filtered_bases, len(sequence)), 6),
        "no_call_fraction": round(_safe_fraction(no_call_bases, len(sequence)), 6),
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
    records: list[dict[str, object]], *, sample_limit: int | None
) -> dict[str, object]:
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("near_duplicate_sample_limit must be positive when provided")

    analyzed_records = records
    analysis_mode = "exact"
    if sample_limit is not None and len(records) > sample_limit:
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
            "total_sequence_count": len(records),
            "analyzed_sequence_count": len(analyzed_sequences),
            "sample_limit": sample_limit,
        },
    }


def _near_duplicate_sampling_key(record: Mapping[str, object]) -> str:
    payload = "|".join(
        str(record[field]) for field in ("sample_id", "locus_id", "source", "sequence")
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _build_length_distribution(records: list[dict[str, object]]) -> list[dict[str, int]]:
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
    missing = sorted(field for field in required_fields if field not in record)
    if missing:
        raise ValueError(f"record is missing required fields: {', '.join(missing)}")


def _coerce_nonnegative_int(record: Mapping[str, object], field: str) -> int:
    value = record[field]
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _mean_metric(records: list[dict[str, object]], field: str) -> float:
    return round(mean(float(record[field]) for record in records), 6) if records else 0.0


def _normalize_sequence(sequence: str) -> str:
    normalized = "".join(sequence.upper().split())
    if not normalized:
        raise ValueError("sequence must not be empty")
    return normalized


def _safe_fraction(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator