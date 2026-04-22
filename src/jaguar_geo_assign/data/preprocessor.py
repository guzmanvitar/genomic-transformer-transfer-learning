"""Preprocessing, split-safety, tokenization, and export helpers for felid corpora.

This module implements the full data-preparation pipeline for the feline
genomics transformer project, covering four stages:

1. **Sequence normalisation** — IUPAC ambiguity resolution and alphabet
   validation against the post-consensus contract (``A/C/G/T/N``).
2. **Sliding-window extraction** — locus-block-aligned windowing with
   deterministic SHA-256-based split assignment that prevents genomic
   leakage across train/validation folds.
3. **Tokenisation** — DNABERT-2 BPE tokenisation with contract-enforced
   provenance pinning (model ID, revision hash, ``trust_remote_code``).
4. **Export** — Parquet and WebDataset serialisation with an auditable
   ``ExportContract`` that guarantees coordinate and hash preservation.
   ``TokenizedCorpusWriter`` is the streaming/append Parquet writer that
   backs the multi-species felid foundation corpus: callers feed one
   batch (typically one species's tokenized windows) at a time so peak
   RAM is bounded by the largest single batch rather than the full
   corpus. The legacy ``write_tokenized_dataset`` function is preserved
   as a thin one-batch shim so the consensus pretrain pipeline keeps
   working unchanged.

Contract change — within-split Parquet ordering
-----------------------------------------------
Prior to the streaming writer, a single ``write_tokenized_dataset``
call sorted the entire corpus and emitted Parquet files in globally
sorted ``(split, contig, block_start, window_start, ...)`` order within
each Hive ``split=`` partition. With the streaming writer, sorting is
**per-batch by ``locus_id``** only; multiple Parquet files may coexist
under a single ``split=.../contig=.../block_id=.../`` directory, each
internally sorted, but there is no global order across files within a
split. Downstream consumers must not assume a row-level sort across
the Parquet dataset; use ``split_manifest`` in ``metadata.json`` to
recover the split assignment of any locus.

Fragility flags
---------------
* ``assign_split`` uses SHA-256 deterministic bucketing: changing the
  ``split_seed`` or the locus-ID format silently reshuffles every split.
* ``window_sequences`` relies on locus-block alignment so that every
  window in a block inherits a single split; misaligned blocks break the
  leakage guarantee.
* ``_WindowMaskCounter`` is a sweep-line algorithm that assumes
  ``mask_spans`` are sorted by start position and that ``summarize`` is
  called with monotonically non-decreasing ``window_start``.
* ``SplitLeakageError`` is the hard guardrail: it fires from both
  ``build_split_manifest`` and ``assert_split_safety`` when a locus
  appears in more than one fold.
* ``TokenizerProvenance.__post_init__`` enforces that
  ``trust_remote_code`` is an actual ``bool`` (not a truthy int) via
  ``_require_boolean_trust_remote_code``.
* ``ExportContract`` mandates that at least one of raw windows or
  immutable sequence hashes is preserved, and locks the hash algorithm
  to SHA-256 for downstream auditability.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tarfile
from typing import Any, Protocol

from .pipeline_contract import (
    DNABERT2_TOKENIZER_ID,
    DNABERT2_TOKENIZER_REVISION as DNABERT2_TOKENIZER_REVISION_HASH,
    DNABERT2_TRUST_REMOTE_CODE,
    POST_CONSENSUS_ALLOWED_ALPHABET,
)

DNABERT2_TOKENIZER_NAME = DNABERT2_TOKENIZER_ID
DNABERT2_TOKENIZER_REVISION = DNABERT2_TOKENIZER_REVISION_HASH
DNABERT2_MAX_POSITION_EMBEDDINGS = 512
ALLOWED_DNA_ALPHABET = POST_CONSENSUS_ALLOWED_ALPHABET
IUPAC_AMBIGUITY_CODES = frozenset("RYSWKMBDHV")
DEFAULT_UNSUPPORTED_SYMBOL_POLICY = "reject"
DEFAULT_EXPORT_FORMAT = "parquet"
DEFAULT_ROW_GROUP_SIZE = 4096
DEFAULT_EXPORT_PARTITION_KEYS = ("split", "contig", "block_id")
DEFAULT_EXPORT_ACCESS_PATTERN = "offline_window_materialization"
DEFAULT_SPLITS = (("train", 0.8), ("validation", 0.2))


class TokenizerLike(Protocol):
    """Structural typing protocol for any callable that behaves like a HuggingFace tokenizer.

    Implementations must accept a DNA sequence string and return a dict
    containing at least ``input_ids`` and optionally ``attention_mask``.
    """

    def __call__(self, sequence: str, **kwargs: Any) -> dict[str, Any]: ...


class PreprocessingError(ValueError):
    """Raised when sequence normalization or preprocessing validation fails."""


class SplitLeakageError(ValueError):
    """Raised when genomic windows would leak across train/validation splits.

    This is the hard guardrail for split safety.  It fires in two places:
    ``build_split_manifest`` (when a locus ID maps to multiple splits) and
    ``assert_split_safety`` (when overlapping windows on the same contig
    belong to different folds).  Any occurrence indicates a bug in the
    locus-block alignment or split-assignment logic.
    """


class TokenizerContractError(ValueError):
    """Raised when tokenizer output violates the pipeline contract.

    Common triggers include mismatched ``trust_remote_code`` policy,
    token counts exceeding ``max_position_embeddings``, or missing
    ``input_ids`` / ``attention_mask`` fields in the tokenizer output.
    """


class ExportContractError(ValueError):
    """Raised when export settings violate the approved artifact contract.

    Guards include: unsupported format, non-positive ``row_group_size``,
    disabling *both* raw-window and sequence-hash preservation (which
    would break downstream reproducibility auditing), and hash algorithm
    changes away from SHA-256.
    """


def _require_boolean_trust_remote_code(
    value: object,
    *,
    field_name: str,
    error_type: type[ValueError] = ValueError,
) -> bool:
    """Validate that *value* is an actual ``bool``, not a truthy integer.

    DNABERT-2's HuggingFace loader interprets ``trust_remote_code=1`` as
    truthy, which would bypass the explicit opt-in safety gate.  This
    guard rejects anything that is not ``True`` or ``False`` by identity,
    preventing accidental coercion.

    Args:
        value: The candidate value to check.
        field_name: Human-readable label used in error messages.
        error_type: Exception class to raise on failure.

    Returns:
        The validated boolean.

    Raises:
        error_type: If ``type(value)`` is not exactly ``bool``.
    """
    if type(value) is not bool:
        raise error_type(f"{field_name} must be an actual boolean, got {value!r} ({type(value).__name__})")
    return value


@dataclass(frozen=True)
class TokenizerProvenance:
    """Immutable provenance record pinning the exact DNABERT-2 tokenizer version.

    Every field is validated in ``__post_init__`` against the pipeline
    contract constants.  In particular, ``trust_remote_code`` is checked
    via ``_require_boolean_trust_remote_code`` to reject truthy-but-not-bool
    values (e.g. ``1``), since HuggingFace's ``AutoTokenizer`` would silently
    accept them, bypassing the explicit security opt-in.

    Fragility: changing any default breaks all downstream provenance
    assertions; update ``pipeline_contract.py`` first.

    Attributes:
        identifier: HuggingFace model ID; must equal ``DNABERT2_TOKENIZER_NAME``.
        revision: Immutable Git SHA for reproducible loading.
        max_position_embeddings: Maximum token-sequence length the model supports.
        allowed_alphabet: Post-consensus nucleotide alphabet (A/C/G/T/N).
        unsupported_symbol_policy: ``"reject"`` or ``"normalize_to_n"``.
        trust_remote_code: Must be an actual ``bool`` matching
            ``DNABERT2_TRUST_REMOTE_CODE``.
    """

    identifier: str = DNABERT2_TOKENIZER_NAME
    revision: str = DNABERT2_TOKENIZER_REVISION
    max_position_embeddings: int = DNABERT2_MAX_POSITION_EMBEDDINGS
    allowed_alphabet: tuple[str, ...] = ALLOWED_DNA_ALPHABET
    unsupported_symbol_policy: str = DEFAULT_UNSUPPORTED_SYMBOL_POLICY
    trust_remote_code: bool = DNABERT2_TRUST_REMOTE_CODE

    def __post_init__(self) -> None:
        if self.identifier != DNABERT2_TOKENIZER_NAME:
            raise ValueError("Tokenizer provenance must pin the approved DNABERT-2 identifier")
        if self.revision != DNABERT2_TOKENIZER_REVISION:
            raise ValueError("Tokenizer provenance must pin the approved DNABERT-2 revision")
        if tuple(self.allowed_alphabet) != ALLOWED_DNA_ALPHABET:
            raise ValueError("Tokenizer provenance allowed_alphabet must match A/C/G/T/N")
        if self.unsupported_symbol_policy not in {"reject", "normalize_to_n"}:
            raise ValueError("unsupported_symbol_policy must be reject or normalize_to_n")
        _require_boolean_trust_remote_code(
            self.trust_remote_code,
            field_name="Tokenizer provenance trust_remote_code",
        )


DNABERT2_TOKENIZER_PROVENANCE = TokenizerProvenance()


@dataclass(frozen=True)
class PreprocessingConfig:
    """Validated configuration for the sequence-preparation and windowing stages.

    All invariants are enforced in ``__post_init__``.  Notably,
    ``locus_block_size`` must be ``>= window_size`` to guarantee that
    every sliding window fits within a single locus block and therefore
    inherits exactly one split assignment.

    Attributes:
        min_sequence_length: Sequences shorter than this (after normalisation)
            are filtered as ``"short_sequence"``.
        max_ambiguity_fraction: Maximum fraction of ``N`` bases allowed.
        window_size: Length of each sliding window in base pairs.
        window_stride: Step size between consecutive windows.
        locus_block_size: Size of the genomic locus block used for
            deterministic split assignment; must be ``>= window_size``.
        ambiguity_mode: ``"mask"`` replaces ambiguous bases with ``N``;
            ``"reject"`` raises on any ambiguous base.
        split_weights: Named fold proportions, e.g.
            ``(("train", 0.8), ("validation", 0.2))``.
        split_seed: Salt string for the SHA-256 split hash; changing this
            silently reshuffles every split assignment.
        allowed_alphabet: Post-consensus nucleotide alphabet.
        export_format: ``"parquet"`` or ``"webdataset"``.
        records_per_shard: Number of records per export shard / row group.
    """

    min_sequence_length: int
    max_ambiguity_fraction: float
    window_size: int
    window_stride: int
    locus_block_size: int
    ambiguity_mode: str = "mask"
    split_weights: tuple[tuple[str, float], ...] = DEFAULT_SPLITS
    split_seed: str = "feline-locus-split-v1"
    allowed_alphabet: tuple[str, ...] = ALLOWED_DNA_ALPHABET
    export_format: str = DEFAULT_EXPORT_FORMAT
    records_per_shard: int = DEFAULT_ROW_GROUP_SIZE

    def __post_init__(self) -> None:
        if self.min_sequence_length <= 0:
            raise ValueError("min_sequence_length must be positive")
        if not 0 <= self.max_ambiguity_fraction <= 1:
            raise ValueError("max_ambiguity_fraction must be between 0 and 1")
        if self.window_size <= 0 or self.window_stride <= 0:
            raise ValueError("window_size and window_stride must be positive")
        if self.locus_block_size < self.window_size:
            raise ValueError("locus_block_size must be >= window_size")
        if self.ambiguity_mode not in {"mask", "reject"}:
            raise ValueError("ambiguity_mode must be 'mask' or 'reject'")
        if set(self.allowed_alphabet) != set(ALLOWED_DNA_ALPHABET):
            raise ValueError("allowed_alphabet must exactly match A/C/G/T/N")
        if self.export_format not in {"parquet", "webdataset"}:
            raise ValueError("export_format must currently be 'parquet' or 'webdataset'")
        if self.records_per_shard <= 0:
            raise ValueError("records_per_shard must be positive")
        total = sum(weight for _, weight in self.split_weights)
        if total <= 0:
            raise ValueError("split_weights must sum to a positive value")
        if any(weight <= 0 for _, weight in self.split_weights):
            raise ValueError("split_weights must be positive")


@dataclass(frozen=True)
class SequenceRecord:
    """Raw consensus sequence before normalisation and quality filtering.

    Attributes:
        sample_id: Unique sample identifier from the manifest.
        individual_id: Biological individual (may share across samples).
        contig: Chromosome or scaffold name.
        sequence: Raw nucleotide string (may contain IUPAC codes).
        source: Provenance label (default ``"consensus"``).
        sequence_start: 0-based genomic start coordinate.
        mask_spans: Sorted ``(start, end, category)`` tuples for masked
            regions propagated from consensus calling.
    """

    sample_id: str
    individual_id: str
    contig: str
    sequence: str
    source: str = "consensus"
    sequence_start: int = 0
    mask_spans: tuple[tuple[int, int, str], ...] = ()

    @property
    def sequence_end(self) -> int:
        """Return the exclusive genomic end coordinate after normalisation."""
        return self.sequence_start + len(normalize_sequence(self.sequence, ambiguity_mode="mask"))


@dataclass(frozen=True)
class PreparedSequence:
    """Normalised and quality-checked sequence ready for windowing.

    Produced by ``prepare_sequences`` after passing the minimum-length
    and maximum-ambiguity filters.

    Attributes:
        sample_id: Unique sample identifier.
        individual_id: Biological individual identifier.
        contig: Chromosome or scaffold name.
        source: Provenance label (e.g. ``"consensus"``).
        sequence_start: 0-based genomic start coordinate.
        sequence: Normalised nucleotide string (alphabet-validated).
        gc_fraction: Fraction of G+C among canonical bases.
        ambiguity_fraction: Fraction of ``N`` bases in the sequence.
        mask_spans: Sorted ``(start, end, category)`` tuples inherited
            from the input ``SequenceRecord``.
    """

    sample_id: str
    individual_id: str
    contig: str
    source: str
    sequence_start: int
    sequence: str
    gc_fraction: float
    ambiguity_fraction: float
    mask_spans: tuple[tuple[int, int, str], ...] = ()

    @property
    def sequence_end(self) -> int:
        """Return the exclusive genomic end coordinate."""
        return self.sequence_start + len(self.sequence)


@dataclass(frozen=True)
class FilteredSequence:
    """Metadata for a sequence that was rejected during preparation.

    Attributes:
        sample_id: Unique sample identifier.
        individual_id: Biological individual identifier.
        contig: Chromosome or scaffold name.
        source: Provenance label.
        reason: Machine-readable rejection reason (``"short_sequence"``
            or ``"high_ambiguity"``).
        sequence_length: Length of the normalised sequence.
        ambiguity_fraction: Fraction of ``N`` bases.
    """

    sample_id: str
    individual_id: str
    contig: str
    source: str
    reason: str
    sequence_length: int
    ambiguity_fraction: float


@dataclass(frozen=True)
class PreprocessingReport:
    """Summary of the sequence-preparation stage.

    Attributes:
        retained: Sequences that passed all quality filters.
        filtered: Metadata for sequences that were rejected.
        mean_gc_fraction: Average GC content across retained sequences.
        mean_ambiguity_fraction: Average ambiguity across retained sequences.
    """

    retained: tuple[PreparedSequence, ...]
    filtered: tuple[FilteredSequence, ...]
    mean_gc_fraction: float
    mean_ambiguity_fraction: float


@dataclass(frozen=True)
class WindowRecord:
    """A single sliding window extracted from a prepared sequence.

    Each window belongs to exactly one locus block and inherits that
    block's deterministic split assignment.  The ``sequence_hash``
    (SHA-256 of the window nucleotide string) provides an immutable
    fingerprint for downstream reproducibility auditing.

    Attributes:
        sample_id: Unique sample identifier.
        individual_id: Biological individual identifier.
        contig: Chromosome or scaffold name.
        source: Provenance label.
        split: Fold name inherited from the parent locus block.
        locus_id: ``"{contig}:{block_start}-{block_end}"`` identifier.
        block_start: Genomic start of the enclosing locus block.
        block_end: Genomic end of the enclosing locus block.
        window_start: Genomic start coordinate of this window.
        window_end: Genomic end coordinate of this window.
        sequence: Normalised nucleotide string for this window.
        gc_fraction: GC content among canonical bases.
        ambiguity_fraction: Fraction of ``N`` bases.
        sequence_hash: SHA-256 hex digest of ``sequence``.
        unique_masked_bases: Reported exclusive masked-base count with
            source-aware fallback: for ``source == "consensus"`` this is
            the de-duplicated span-derived count verbatim (no fallback, so
            provenance gaps remain auditable); for ``source == "reference"``
            windows (the only approved non-consensus emitted source label)
            it falls back to realized ``N`` coverage when no mask spans are
            declared.
        filtered_bases: Bases masked with the ``"filtered"`` category.
        no_call_bases: Bases masked with the ``"no_call"`` category.
        other_masked_bases: Bases in mask categories other than
            ``"filtered"`` and ``"no_call"``.
        masked_base_counts: Sorted ``(category, count)`` tuples for all
            mask categories overlapping this window.
    """

    sample_id: str
    individual_id: str
    contig: str
    source: str
    split: str
    locus_id: str
    block_start: int
    block_end: int
    window_start: int
    window_end: int
    sequence: str
    gc_fraction: float
    ambiguity_fraction: float
    sequence_hash: str
    unique_masked_bases: int = 0
    filtered_bases: int = 0
    no_call_bases: int = 0
    other_masked_bases: int = 0
    masked_base_counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class SplitManifestEntry:
    """One row of the split manifest mapping a locus block to a fold.

    Attributes:
        locus_id: ``"{contig}:{block_start}-{block_end}"`` identifier.
        contig: Chromosome or scaffold name.
        block_start: Genomic start of the locus block.
        block_end: Genomic end of the locus block.
        split: Assigned fold name.
    """

    locus_id: str
    contig: str
    block_start: int
    block_end: int
    split: str


@dataclass(frozen=True)
class TokenizedWindow:
    """A genomic window after DNABERT-2 BPE tokenisation.

    Attributes:
        window: The source ``WindowRecord``.
        input_ids: Integer token IDs produced by the tokenizer.
        attention_mask: Binary mask (``1`` = real token, ``0`` = padding).
        token_count: Number of tokens (including special tokens).
        token_to_base_ratio: ``token_count / len(window.sequence)``.
        tokenizer: Provenance record of the tokenizer used.
    """

    window: WindowRecord
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_count: int
    token_to_base_ratio: float
    tokenizer: TokenizerProvenance


@dataclass(frozen=True)
class ExportContract:
    """Immutable contract governing how tokenised windows are serialised.

    The ``__post_init__`` validates that the export configuration is
    auditable: at least one of ``preserve_raw_windows`` or
    ``preserve_sequence_hashes`` must be ``True``, and the hash algorithm
    is locked to SHA-256 so downstream consumers can verify window
    integrity without access to the original FASTA.

    Fragility: disabling both preservation flags raises
    ``ExportContractError``; changing ``sequence_hash_algorithm`` away
    from ``"sha256"`` is also rejected.

    Attributes:
        format: ``"parquet"`` or ``"webdataset"``.
        access_pattern: Intended read pattern label.
        row_group_size: Records per Parquet row group.
        deterministic_partition_keys: Hive-style partition columns.
        preserve_raw_windows: Whether to include raw nucleotide strings.
        preserve_sequence_hashes: Whether to include SHA-256 hashes.
        preserve_coordinates: Whether to include genomic coordinates.
        sequence_hash_algorithm: Must remain ``"sha256"``.
    """

    format: str = DEFAULT_EXPORT_FORMAT
    access_pattern: str = DEFAULT_EXPORT_ACCESS_PATTERN
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE
    deterministic_partition_keys: tuple[str, ...] = DEFAULT_EXPORT_PARTITION_KEYS
    preserve_raw_windows: bool = False
    preserve_sequence_hashes: bool = True
    preserve_coordinates: bool = True
    sequence_hash_algorithm: str = "sha256"

    def __post_init__(self) -> None:
        if self.format not in {"parquet", "webdataset"}:
            raise ExportContractError("export format must be parquet or webdataset")
        if self.row_group_size <= 0:
            raise ExportContractError("row_group_size must be positive")
        if not self.preserve_raw_windows and not self.preserve_sequence_hashes:
            raise ExportContractError("export must preserve raw windows or immutable sequence hashes")
        if self.sequence_hash_algorithm != "sha256":
            raise ExportContractError("sequence_hash_algorithm must remain sha256")


DEFAULT_PARQUET_EXPORT_CONTRACT = ExportContract()


def normalize_sequence(
    sequence: str,
    *,
    ambiguity_mode: str = "mask",
    allowed_alphabet: tuple[str, ...] = ALLOWED_DNA_ALPHABET,
) -> str:
    """Normalise genomic sequence characters deterministically before tokenisation.

    Upper-cases the input, strips whitespace, maps IUPAC ambiguity codes
    (and gap/unknown symbols ``-``, ``?``, ``.``) to ``N`` when
    *ambiguity_mode* is ``"mask"``, and rejects any character outside the
    allowed alphabet when *ambiguity_mode* is ``"reject"``.

    Args:
        sequence: Raw nucleotide string (may contain mixed case, whitespace,
            IUPAC codes).
        ambiguity_mode: ``"mask"`` converts ambiguous symbols to ``N``;
            ``"reject"`` raises on any out-of-alphabet character.
        allowed_alphabet: Post-consensus alphabet to validate against.

    Returns:
        A normalised string containing only characters in *allowed_alphabet*.

    Raises:
        ValueError: If *ambiguity_mode* is not ``"mask"`` or ``"reject"``.
        PreprocessingError: If an unsupported base is encountered in
            ``"reject"`` mode.
    """

    if ambiguity_mode not in {"mask", "reject"}:
        raise ValueError("ambiguity_mode must be 'mask' or 'reject'")

    allowed = set(allowed_alphabet)
    normalized: list[str] = []
    for raw_base in sequence.upper():
        if raw_base.isspace():
            continue
        if raw_base in allowed:
            normalized.append(raw_base)
            continue
        if ambiguity_mode == "mask" and (
            raw_base in IUPAC_AMBIGUITY_CODES or raw_base.isalpha() or raw_base in {"-", "?", "."}
        ):
            normalized.append("N")
            continue
        raise PreprocessingError(f"Unsupported base '{raw_base}' for alphabet {sorted(allowed)}")
    return "".join(normalized)


def gc_fraction(sequence: str) -> float:
    """Compute the GC fraction among canonical (A/C/G/T) bases.

    ``N`` bases are excluded from both numerator and denominator,
    so the result reflects GC content of the informative portion only.

    Args:
        sequence: Normalised nucleotide string.

    Returns:
        GC fraction in ``[0.0, 1.0]``, or ``0.0`` if no canonical bases.
    """
    canonical_bases = sum(1 for base in sequence if base in {"A", "C", "G", "T"})
    if canonical_bases == 0:
        return 0.0
    gc_bases = sum(1 for base in sequence if base in {"G", "C"})
    return gc_bases / canonical_bases


def ambiguity_fraction(sequence: str) -> float:
    """Compute the fraction of ambiguous (``N``) bases in *sequence*.

    Args:
        sequence: Normalised nucleotide string.

    Returns:
        Fraction in ``[0.0, 1.0]``, or ``0.0`` for an empty string.
    """
    if not sequence:
        return 0.0
    return sequence.count("N") / len(sequence)

def prepare_sequences(
    records: list[SequenceRecord],
    config: PreprocessingConfig,
) -> PreprocessingReport:
    """Normalise, filter, and compute quality metrics for raw sequence records.

    Each record is normalised via ``normalize_sequence``, then checked
    against ``config.min_sequence_length`` and
    ``config.max_ambiguity_fraction``.  Records that fail either filter
    are captured in ``PreprocessingReport.filtered`` with a machine-readable
    ``reason``.

    Producer ``source`` labels are validated *before* any filter runs, so
    malformed provenance (e.g. typos such as ``"consensus_typo"``) cannot
    be silently swallowed by the short-sequence or high-ambiguity filters
    and leak into downstream analysis as a missing-record statistic.

    Args:
        records: Raw consensus sequences to process.
        config: Preprocessing thresholds and alphabet settings.

    Returns:
        A ``PreprocessingReport`` with retained and filtered sequences
        plus aggregate GC and ambiguity statistics.

    Raises:
        PreprocessingError: If any record carries a ``source`` label outside
            the approved producer set ``{"consensus", "reference"}``.
    """
    retained: list[PreparedSequence] = []
    filtered: list[FilteredSequence] = []

    for record in records:
        _require_approved_source_label(record.source)
        normalized = normalize_sequence(
            record.sequence,
            ambiguity_mode=config.ambiguity_mode,
            allowed_alphabet=config.allowed_alphabet,
        )
        ambiguity = ambiguity_fraction(normalized)
        if len(normalized) < config.min_sequence_length:
            filtered.append(
                FilteredSequence(
                    sample_id=record.sample_id,
                    individual_id=record.individual_id,
                    contig=record.contig,
                    source=record.source,
                    reason="short_sequence",
                    sequence_length=len(normalized),
                    ambiguity_fraction=ambiguity,
                )
            )
            continue
        if ambiguity > config.max_ambiguity_fraction:
            filtered.append(
                FilteredSequence(
                    sample_id=record.sample_id,
                    individual_id=record.individual_id,
                    contig=record.contig,
                    source=record.source,
                    reason="high_ambiguity",
                    sequence_length=len(normalized),
                    ambiguity_fraction=ambiguity,
                )
            )
            continue
        retained.append(
            PreparedSequence(
                sample_id=record.sample_id,
                individual_id=record.individual_id,
                contig=record.contig,
                source=record.source,
                sequence_start=record.sequence_start,
                sequence=normalized,
                gc_fraction=gc_fraction(normalized),
                ambiguity_fraction=ambiguity,
                mask_spans=tuple(sorted(record.mask_spans)),
            )
        )

    mean_gc = sum(item.gc_fraction for item in retained) / len(retained) if retained else 0.0
    mean_ambiguity = (
        sum(item.ambiguity_fraction for item in retained) / len(retained) if retained else 0.0
    )
    return PreprocessingReport(
        retained=tuple(retained),
        filtered=tuple(filtered),
        mean_gc_fraction=mean_gc,
        mean_ambiguity_fraction=mean_ambiguity,
    )


def assign_split(locus_id: str, split_weights: tuple[tuple[str, float], ...], split_seed: str) -> str:
    """Deterministically assign a locus block to a fold using SHA-256 hashing.

    The assignment is computed as
    ``SHA256("{split_seed}:{locus_id}")[:16]`` → 64-bit bucket →
    normalised position in ``[0, 1)`` → cumulative-weight lookup.

    Fragility: changing *split_seed* or the locus-ID format
    (``"{contig}:{block_start}-{block_end}"``) silently reshuffles
    every split assignment across the entire corpus.

    Args:
        locus_id: Unique block identifier used as hash input.
        split_weights: ``(name, weight)`` tuples; weights are normalised
            to sum to 1.
        split_seed: Salt string prepended to the hash input.

    Returns:
        The name of the assigned fold.
    """
    total = sum(weight for _, weight in split_weights)
    bucket = int(sha256(f"{split_seed}:{locus_id}".encode("utf-8")).hexdigest()[:16], 16)
    position = bucket / float(16**16)
    cumulative = 0.0
    for split_name, weight in split_weights:
        cumulative += weight / total
        if position < cumulative:
            return split_name
    return split_weights[-1][0]


@dataclass
class _WindowMaskCounter:
    """Sweep-line accumulator for mask-span overlap counts across sliding windows.

    Designed to be called once per window with monotonically non-decreasing
    ``window_start`` values.  It maintains an ``active_spans`` list that
    is incrementally pruned (spans whose end ≤ current ``window_start``
    are dropped) and extended (spans whose start < ``window_end`` are
    absorbed), amortising the cost over the full sweep.

    Fragility: calling ``summarize`` with non-monotonic ``window_start``
    values will silently produce incorrect counts because already-discarded
    spans cannot be recovered.

    Attributes:
        mask_spans: Sorted ``(start, end, category)`` tuples from the
            parent ``PreparedSequence``.
        next_index: Cursor into ``mask_spans`` tracking the next span
            to absorb.
        active_spans: Spans currently overlapping the sweep frontier.
    """

    mask_spans: tuple[tuple[int, int, str], ...]
    next_index: int = 0
    active_spans: list[tuple[int, int, str]] | None = None

    def __post_init__(self) -> None:
        """Initialise ``active_spans`` to an empty list."""
        self.active_spans = []

    def summarize(self, *, window_start: int, window_end: int) -> tuple[Counter[str], int]:
        """Count masked bases overlapping the window ``[window_start, window_end)``.

        Args:
            window_start: Inclusive genomic start of the window.
            window_end: Exclusive genomic end of the window.

        Returns:
            A ``(category_counts, unique_masked_bases)`` tuple where
            *category_counts* maps each mask category to the number of
            overlapping bases, and *unique_masked_bases* is the
            de-duplicated total across all categories.
        """
        assert self.active_spans is not None
        while self.next_index < len(self.mask_spans) and self.mask_spans[self.next_index][0] < window_end:
            self.active_spans.append(self.mask_spans[self.next_index])
            self.next_index += 1
        self.active_spans = [span for span in self.active_spans if span[1] > window_start]

        counts: Counter[str] = Counter()
        overlapping_ranges: list[tuple[int, int]] = []
        for span_start, span_end, category in self.active_spans:
            overlap = min(window_end, span_end) - max(window_start, span_start)
            if overlap > 0:
                counts[category] += overlap
                overlapping_ranges.append((max(window_start, span_start), min(window_end, span_end)))
        return counts, _count_unique_masked_bases(overlapping_ranges)


def _count_unique_masked_bases(overlapping_ranges: list[tuple[int, int]]) -> int:
    """Merge overlapping genomic ranges and return the total covered length.

    Uses a sort-and-sweep merge to de-duplicate base counts when multiple
    mask categories overlap the same genomic interval.

    Args:
        overlapping_ranges: ``(start, end)`` pairs clipped to the current
            window boundaries.

    Returns:
        Total number of unique bases covered by at least one range.
    """
    if not overlapping_ranges:
        return 0

    merged_ranges = sorted(overlapping_ranges)
    merged_total = 0
    current_start, current_end = merged_ranges[0]
    for span_start, span_end in merged_ranges[1:]:
        if span_start > current_end:
            merged_total += current_end - current_start
            current_start, current_end = span_start, span_end
            continue
        current_end = max(current_end, span_end)

    return merged_total + (current_end - current_start)


CONSENSUS_SOURCE_LABEL = "consensus"
REFERENCE_SOURCE_LABEL = "reference"
APPROVED_SOURCE_LABELS = frozenset({CONSENSUS_SOURCE_LABEL, REFERENCE_SOURCE_LABEL})


def _require_approved_source_label(source: str) -> None:
    """Fail loudly on any producer ``source`` label outside the approved set.

    Intent: the approved emitted producer source set is exactly
    ``{"consensus", "reference"}``. Any other label must raise before
    downstream filtering or fallback logic can silently swallow it
    (e.g. a short-sequence or high-ambiguity filter would otherwise drop
    malformed records with typos and hide the provenance defect).

    Args:
        source: Candidate producer source label.

    Raises:
        PreprocessingError: If ``source`` is not in ``APPROVED_SOURCE_LABELS``.
    """
    if source not in APPROVED_SOURCE_LABELS:
        raise PreprocessingError(
            f"Unknown source label '{source}': only {sorted(APPROVED_SOURCE_LABELS)} are approved. "
            "Update the centralized source contract if a new label is required."
        )


def _count_realized_unique_masked_bases(
    sequence: str,
    span_unique_masked_bases: int,
    *,
    source: str,
) -> int:
    """Return the single reported exclusive masked-base count for a window.

    The fallback behaviour is strictly source-aware so the producer never
    silently reconciles a provenance gap for consensus-derived windows. For
    consensus windows the span-derived count is reported verbatim so
    downstream diagnostics can detect any ``N`` base that is not accounted
    for by an explicit mask span. For approved reference windows (which
    legitimately carry intrinsic ``N`` bases but never declare mask spans)
    the realized ``N`` coverage is used as a fallback so valid windows are
    not false-flagged by the coverage invariant.

    Args:
        sequence: The window nucleotide string.
        span_unique_masked_bases: De-duplicated base count from mask spans.
        source: Window provenance label. Must be one of the approved labels
            in ``APPROVED_SOURCE_LABELS``. ``"consensus"`` is provenance-strict;
            ``"reference"`` is the only approved non-consensus label allowed to
            fall back to the realized ``N`` coverage of *sequence*.

    Returns:
        The reported ``unique_masked_bases`` value for the window.

    Raises:
        PreprocessingError: If ``source`` is not in ``APPROVED_SOURCE_LABELS``.
    """
    _require_approved_source_label(source)
    if source == CONSENSUS_SOURCE_LABEL:
        return span_unique_masked_bases
    return max(span_unique_masked_bases, sequence.count("N"))


def window_sequences(
    sequences: list[PreparedSequence],
    config: PreprocessingConfig,
) -> tuple[WindowRecord, ...]:
    """Extract locus-block-aligned sliding windows with deterministic split labels.

    For each prepared sequence the function:

    1. Computes the enclosing locus blocks (aligned to
       ``config.locus_block_size``).
    2. Assigns each block to a fold via ``assign_split``.
    3. Slides a window of ``config.window_size`` with stride
       ``config.window_stride`` across the block–sequence overlap,
       skipping windows that exceed ``config.max_ambiguity_fraction``.
    4. Calls ``assert_split_safety`` at the end to guarantee that no
       overlapping windows on the same contig belong to different folds.

    Fragility: the locus-block alignment guarantees split safety *only*
    when ``config.locus_block_size >= config.window_size``.  If that
    invariant is violated, windows may span block boundaries and the
    split-safety assertion will fire.

    Args:
        sequences: Normalised and filtered sequences from
            ``prepare_sequences``.
        config: Windowing and split parameters.

    Returns:
        Tuple of ``WindowRecord`` instances, sorted by extraction order.

    Raises:
        PreprocessingError: If any ``PreparedSequence.source`` lies outside
            the approved producer set ``{"consensus", "reference"}``. The
            check runs *before* any per-block overlap/length filter or
            per-window ambiguity filter so direct callers that bypass
            ``prepare_sequences`` cannot let a malformed label disappear
            into an empty-windows result.
        SplitLeakageError: If ``assert_split_safety`` detects cross-fold
            overlap.
    """
    windows: list[WindowRecord] = []
    for sequence in sequences:
        _require_approved_source_label(sequence.source)
        mask_counter = _WindowMaskCounter(sequence.mask_spans)
        genomic_start = sequence.sequence_start
        genomic_end = sequence.sequence_end
        first_block_start = (genomic_start // config.locus_block_size) * config.locus_block_size

        for block_start in range(first_block_start, genomic_end, config.locus_block_size):
            block_end = block_start + config.locus_block_size
            overlap_start = max(block_start, genomic_start)
            overlap_end = min(block_end, genomic_end)
            if overlap_end - overlap_start < config.window_size:
                continue

            block_offset_start = overlap_start - genomic_start
            block_offset_end = overlap_end - genomic_start
            block_sequence = sequence.sequence[block_offset_start:block_offset_end]
            locus_id = f"{sequence.contig}:{block_start}-{block_end}"
            split = assign_split(locus_id, config.split_weights, config.split_seed)

            for offset in range(0, len(block_sequence) - config.window_size + 1, config.window_stride):
                window_start = overlap_start + offset
                window_end = window_start + config.window_size
                if window_end > overlap_end:
                    break
                window_sequence = block_sequence[offset : offset + config.window_size]
                window_ambiguity = ambiguity_fraction(window_sequence)
                if window_ambiguity > config.max_ambiguity_fraction:
                    continue
                mask_counts, span_unique_masked_bases = mask_counter.summarize(
                    window_start=window_start,
                    window_end=window_end,
                )
                unique_masked_bases = _count_realized_unique_masked_bases(
                    window_sequence,
                    span_unique_masked_bases,
                    source=sequence.source,
                )
                filtered_bases = mask_counts.get("filtered", 0)
                no_call_bases = mask_counts.get("no_call", 0)
                other_masked_bases = sum(
                    count for category, count in mask_counts.items() if category not in {"filtered", "no_call"}
                )
                windows.append(
                    WindowRecord(
                        sample_id=sequence.sample_id,
                        individual_id=sequence.individual_id,
                        contig=sequence.contig,
                        source=sequence.source,
                        split=split,
                        locus_id=locus_id,
                        block_start=block_start,
                        block_end=block_end,
                        window_start=window_start,
                        window_end=window_end,
                        sequence=window_sequence,
                        gc_fraction=gc_fraction(window_sequence),
                        ambiguity_fraction=window_ambiguity,
                        sequence_hash=sha256(window_sequence.encode("utf-8")).hexdigest(),
                        unique_masked_bases=unique_masked_bases,
                        filtered_bases=filtered_bases,
                        no_call_bases=no_call_bases,
                        other_masked_bases=other_masked_bases,
                        masked_base_counts=tuple(sorted(mask_counts.items())),
                    )
                )
    assert_split_safety(tuple(windows))
    return tuple(windows)


def build_split_manifest(windows: tuple[WindowRecord, ...]) -> tuple[SplitManifestEntry, ...]:
    """Build a deduplicated manifest mapping each locus block to its fold.

    If a locus ID appears with conflicting split labels, a
    ``SplitLeakageError`` is raised immediately.

    Args:
        windows: Window records (typically from ``window_sequences``).

    Returns:
        Sorted tuple of ``SplitManifestEntry`` instances, one per unique
        locus block, ordered by ``(contig, block_start, split)``.

    Raises:
        SplitLeakageError: If any locus ID is assigned to more than one
            fold.
    """
    manifest: dict[str, SplitManifestEntry] = {}
    for window in windows:
        current = manifest.get(window.locus_id)
        entry = SplitManifestEntry(
            locus_id=window.locus_id,
            contig=window.contig,
            block_start=window.block_start,
            block_end=window.block_end,
            split=window.split,
        )
        if current is not None and current.split != entry.split:
            raise SplitLeakageError(f"Locus {window.locus_id} is assigned to multiple splits")
        manifest[window.locus_id] = entry
    return tuple(sorted(manifest.values(), key=lambda item: (item.contig, item.block_start, item.split)))


def assert_split_safety(windows: tuple[WindowRecord, ...]) -> None:
    """Verify that no genomic leakage exists across train/validation folds.

    Performs two checks:

    1. **Locus-level**: every locus ID maps to exactly one split.
    2. **Overlap-level**: on each contig, no pair of overlapping windows
       belongs to different splits.  This uses a sweep-line algorithm
       that sorts windows by ``(window_start, window_end)`` and
       maintains an active set of unexpired windows.

    Args:
        windows: Window records to validate.

    Raises:
        SplitLeakageError: If either check fails.
    """
    per_locus_split: dict[str, str] = {}
    for window in windows:
        existing = per_locus_split.setdefault(window.locus_id, window.split)
        if existing != window.split:
            raise SplitLeakageError(f"Locus {window.locus_id} leaked across splits")

    by_contig: dict[str, list[WindowRecord]] = defaultdict(list)
    for window in windows:
        by_contig[window.contig].append(window)

    for contig, contig_windows in by_contig.items():
        active: list[WindowRecord] = []
        for window in sorted(contig_windows, key=lambda item: (item.window_start, item.window_end)):
            active = [candidate for candidate in active if candidate.window_end > window.window_start]
            for candidate in active:
                overlaps = candidate.window_start < window.window_end and window.window_start < candidate.window_end
                if overlaps and candidate.split != window.split:
                    raise SplitLeakageError(
                        "Overlapping windows across splits detected on "
                        f"{contig}: {candidate.window_start}-{candidate.window_end} ({candidate.split}) vs "
                        f"{window.window_start}-{window.window_end} ({window.split})"
                    )
            active.append(window)


def load_dnabert2_tokenizer(
    provenance: TokenizerProvenance = DNABERT2_TOKENIZER_PROVENANCE,
) -> tuple[TokenizerLike, TokenizerProvenance]:
    """Load the pinned DNABERT-2 tokenizer from HuggingFace.

    Before loading, the ``trust_remote_code`` flag is validated against
    the pipeline contract via ``_assert_approved_dnabert2_trust_policy``.

    Args:
        provenance: Tokenizer pinning record; defaults to the module-level
            ``DNABERT2_TOKENIZER_PROVENANCE``.

    Returns:
        ``(tokenizer, provenance)`` tuple.

    Raises:
        TokenizerContractError: If ``trust_remote_code`` does not match the
            approved pipeline contract value.
        RuntimeError: If the ``transformers`` package is not installed.
    """
    _assert_approved_dnabert2_trust_policy(provenance.trust_remote_code)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised through integration, not unit logic
        raise RuntimeError(
            "DNABERT-2 tokenization requires transformers. Install with: "
            "uv add \"transformers>=4.28,<5\""
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        provenance.identifier,
        revision=provenance.revision,
        trust_remote_code=provenance.trust_remote_code,
    )
    return tokenizer, provenance


def _assert_approved_dnabert2_trust_policy(trust_remote_code: bool) -> None:
    """Guard that *trust_remote_code* is a real bool matching the contract.

    Combines the ``_require_boolean_trust_remote_code`` type check with
    an identity comparison against ``DNABERT2_TRUST_REMOTE_CODE``.

    Args:
        trust_remote_code: Value to validate.

    Raises:
        TokenizerContractError: On type mismatch or value mismatch.
    """
    trust_remote_code = _require_boolean_trust_remote_code(
        trust_remote_code,
        field_name="DNABERT-2 trust_remote_code policy",
        error_type=TokenizerContractError,
    )
    if trust_remote_code is not DNABERT2_TRUST_REMOTE_CODE:
        raise TokenizerContractError(
            "DNABERT-2 trust_remote_code policy mismatch: "
            f"expected {DNABERT2_TRUST_REMOTE_CODE}, got {trust_remote_code}. "
            "Use the approved pipeline contract value explicitly."
        )


def _resolve_export_tokenizer_provenance(
    tokenized_windows: tuple[TokenizedWindow, ...],
    *,
    provenance: TokenizerProvenance | None,
) -> TokenizerProvenance:
    """Resolve and validate the tokenizer provenance for an export batch.

    If *tokenized_windows* is non-empty, the provenance is extracted from
    the first record and verified to be uniform across all records.  An
    explicit *provenance* argument, if given, must match.  If
    *tokenized_windows* is empty, *provenance* must be provided.

    Args:
        tokenized_windows: Records to export (may be empty).
        provenance: Optional explicit provenance override.

    Returns:
        The validated ``TokenizerProvenance``.

    Raises:
        ExportContractError: On provenance mismatch or missing provenance.
    """
    if tokenized_windows:
        resolved = tokenized_windows[0].tokenizer
        _assert_approved_dnabert2_trust_policy(resolved.trust_remote_code)
        if any(record.tokenizer != resolved for record in tokenized_windows[1:]):
            raise ExportContractError("All tokenized windows must share identical tokenizer provenance")
        if provenance is not None and provenance != resolved:
            raise ExportContractError(
                "Explicit tokenizer provenance does not match the tokenized window metadata"
            )
        return resolved
    if provenance is None:
        raise ExportContractError(
            "Tokenized export metadata requires explicit tokenizer provenance when no tokenized windows are available"
        )
    _assert_approved_dnabert2_trust_policy(provenance.trust_remote_code)
    return provenance


def tokenize_windows(
    windows: tuple[WindowRecord, ...],
    tokenizer: TokenizerLike,
    provenance: TokenizerProvenance = DNABERT2_TOKENIZER_PROVENANCE,
) -> tuple[TokenizedWindow, ...]:
    """Tokenise genomic windows with a DNABERT-2-compatible tokenizer.

    Each window is first re-normalised against the provenance alphabet
    via ``_prepare_window_for_tokenization``, then tokenised.  The
    function enforces that token counts do not exceed
    ``provenance.max_position_embeddings`` and that ``input_ids`` and
    ``attention_mask`` are aligned.

    Args:
        windows: Genomic windows to tokenise.
        tokenizer: A callable conforming to ``TokenizerLike``.
        provenance: Tokenizer provenance for contract validation.

    Returns:
        Tuple of ``TokenizedWindow`` instances.

    Raises:
        TokenizerContractError: On missing fields, length mismatches,
            or token counts exceeding ``max_position_embeddings``.
    """
    tokenized: list[TokenizedWindow] = []
    for window in windows:
        tokenization_window = _prepare_window_for_tokenization(window, provenance)
        encoding = tokenizer(tokenization_window.sequence, add_special_tokens=True, truncation=False)
        input_ids = _coerce_int_tuple(encoding.get("input_ids"), field_name="input_ids")
        attention_mask_raw = encoding.get("attention_mask", [1] * len(input_ids))
        attention_mask = _coerce_int_tuple(attention_mask_raw, field_name="attention_mask")
        if len(input_ids) != len(attention_mask):
            raise TokenizerContractError("attention_mask must align with input_ids length")
        if len(input_ids) > provenance.max_position_embeddings:
            raise TokenizerContractError(
                "Retained genomic window tokenized beyond max_position_embeddings: "
                f"observed token_count={len(input_ids)}, "
                f"max_position_embeddings={provenance.max_position_embeddings}, "
                f"{_format_window_context(tokenization_window)}"
            )
        tokenized.append(
            TokenizedWindow(
                window=tokenization_window,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_count=len(input_ids),
                token_to_base_ratio=len(input_ids) / len(tokenization_window.sequence),
                tokenizer=provenance,
            )
        )
    return tuple(tokenized)


class TokenizedCorpusWriter:
    """Streaming/append Parquet writer for multi-batch tokenized corpora.

    Intent — why this class exists:
        The felid foundation pretraining corpus is assembled from six
        multi-gigabase reference assemblies. Full materialisation of all
        tokenized windows in one process would OOM even a large VM. This
        writer lets callers feed one batch at a time (typically one
        species per batch) so peak RAM stays bounded by the **largest
        single batch**, not the full corpus. The legacy single-shot
        ``write_tokenized_dataset`` is preserved as a thin one-batch
        shim around this class so the consensus pretrain pipeline
        continues to run unchanged.

    Contract change (vs. legacy single-shot writer):
        The legacy writer globally sorted all records within each
        ``split=`` Hive partition before writing, producing a single
        totally-ordered sequence of Parquet files per split. The
        streaming writer sorts **per batch** by ``locus_id``, then
        partitions by ``(split, contig, block_id)`` as before. Across
        multiple ``write_batch`` calls, a single
        ``split=.../contig=.../block_id=.../`` directory may therefore
        contain several internally-sorted Parquet files with **no
        global order across files**. Downstream consumers that relied
        on within-split row ordering must now read the whole partition
        and re-sort if they need a total order (see the module
        docstring for the full rationale).

    Grep evidence supporting the contract change (captured during the
    refactor so we don't accidentally reintroduce an invariant nobody
    enforces): the only hit for ``sort_values.*locus`` / ``locus_id.*sort``
    / ``sorted.*locus_id`` across ``src/`` and ``tests/`` was in
    ``reporting/genomics_diagnostics.py`` (sorting diagnostic dict
    output by locus_id), which does not read the Parquet dataset and
    does not assume row order within any Parquet file.

    Usage:
        >>> with TokenizedCorpusWriter(output_dir, contract=contract,
        ...                            provenance=provenance) as writer:
        ...     for batch in batches:
        ...         writer.write_batch(batch)
        >>> writer.split_paths  # {"train": [...], "validation": [...]}

    Lifecycle guarantees:
        - Per-split Parquet writers are created lazily: no
          ``split=validation/`` file tree is produced if no batch ever
          contains a validation window.
        - ``__exit__`` writes the corpus ``metadata.json`` sidecar on
          clean exit. On an exception bubbling out of the ``with``
          block, any Parquet files already written are deleted and no
          manifest is emitted, so the output directory never contains
          half-written artifacts that would corrupt a downstream train
          loader.
        - ``row_group_size`` from the ``ExportContract`` is honoured
          **per batch**: each ``write_batch`` call partitions its own
          records into row-group-sized chunks. The cumulative row
          count in a single Parquet file never exceeds
          ``contract.row_group_size``.

    Thread-safety:
        Not thread-safe. A single writer instance must be driven by a
        single producer. Multi-process writers are out of scope.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        contract: ExportContract = DEFAULT_PARQUET_EXPORT_CONTRACT,
        provenance: TokenizerProvenance | None = None,
    ) -> None:
        """Initialise a streaming Parquet writer (no I/O until ``__enter__``).

        Args:
            output_dir: Root directory for the Parquet dataset. Created on
                ``__enter__`` if it does not already exist.
            contract: Export settings governing format, partitioning, and
                preservation flags.
            provenance: Optional explicit tokenizer provenance. Required if
                the corpus ends up empty (no batches written); otherwise
                inferred from the first non-empty batch and validated for
                consistency across subsequent batches.
        """
        self._output_path = Path(output_dir)
        self._contract = contract
        self._explicit_provenance = provenance
        self._resolved_provenance: TokenizerProvenance | None = None
        self._batch_index = 0
        self._written_files: list[Path] = []
        self._split_paths: dict[str, list[Path]] = defaultdict(list)
        self._split_record_counts: dict[str, int] = defaultdict(int)
        self._sqlite_path: Path | None = None
        self._sqlite_conn: sqlite3.Connection | None = None
        self._parquet_backend: tuple[Any, Any] | None = None
        self._opened = False
        self._closed = False

    def __enter__(self) -> "TokenizedCorpusWriter":
        """Validate the contract, create the output directory, open the writer.

        The locus manifest is stored in a SQLite sidecar at
        ``{output_dir}/.locus_manifest.sqlite`` rather than an in-memory
        dict. Intent: prevent the previous O(total-windows) Python heap
        pressure (~2–4 GB on the full felid corpus) that contradicted
        the "peak RAM ≈ O(largest single species)" spec claim. The
        sidecar is scratch: it is created here and removed unconditionally
        by ``__exit__`` so it never appears in the deliverable tree.
        """
        if self._contract.format != "parquet":
            raise ExportContractError(
                "TokenizedCorpusWriter only supports parquet contracts; "
                "use write_webdataset_shards for webdataset exports"
            )
        self._output_path.mkdir(parents=True, exist_ok=True)
        sidecar_path = self._output_path / ".locus_manifest.sqlite"
        if sidecar_path.exists():
            sidecar_path.unlink()
        self._sqlite_path = sidecar_path
        self._sqlite_conn = sqlite3.connect(sidecar_path)
        self._sqlite_conn.execute(
            "CREATE TABLE locus_entries ("
            "locus_id TEXT PRIMARY KEY, "
            "contig TEXT NOT NULL, "
            "block_start INTEGER NOT NULL, "
            "block_end INTEGER NOT NULL, "
            "split TEXT NOT NULL)"
        )
        self._sqlite_conn.commit()
        self._opened = True
        return self

    def write_batch(self, tokenized_windows: Any) -> None:
        """Append one batch of tokenized windows to the corpus.

        The batch is sorted by ``locus_id`` (then by the stable tie-breaker
        ``(contig, block_start, window_start, sample_id, source)``),
        partitioned by ``(split, contig, block_id)``, and chunked into
        row-groups of at most ``contract.row_group_size`` records. Each
        chunk is written to its own Parquet file named
        ``part-{batch_index:05d}-{chunk_index:05d}.parquet`` under the
        partition directory.

        Empty batches are accepted (they just advance the batch counter)
        so a pipeline that iterates over species can safely write an
        empty result for a species that yielded no tokens without
        special-casing at the call site.

        Args:
            tokenized_windows: Records to append. An empty sequence is a
                no-op beyond advancing the batch counter.

        Raises:
            ExportContractError: If the writer is not inside a
                ``with`` block, if a later batch carries a different
                tokenizer provenance than the first non-empty batch, or
                if PyArrow is not installed.
            SplitLeakageError: If a ``locus_id`` already seen in a prior
                batch is now assigned to a different split.
        """
        if not self._opened or self._closed:
            raise ExportContractError(
                "TokenizedCorpusWriter.write_batch must be called inside a with-block"
            )
        batch = tuple(tokenized_windows)
        if not batch:
            self._batch_index += 1
            return

        batch_provenance = _resolve_export_tokenizer_provenance(
            batch,
            provenance=self._explicit_provenance,
        )
        if self._resolved_provenance is None:
            self._resolved_provenance = batch_provenance
        elif batch_provenance != self._resolved_provenance:
            raise ExportContractError(
                "TokenizedCorpusWriter batches must share identical tokenizer provenance"
            )

        sorted_batch = sorted(
            batch,
            key=lambda item: (
                item.window.locus_id,
                item.window.contig,
                item.window.block_start,
                item.window.window_start,
                item.window.sample_id,
                item.window.source,
            ),
        )

        batch_entries: dict[str, tuple[str, int, int, str]] = {}
        for record in sorted_batch:
            window = record.window
            row = (window.contig, window.block_start, window.block_end, window.split)
            existing = batch_entries.get(window.locus_id)
            if existing is not None and existing[3] != window.split:
                raise SplitLeakageError(
                    f"Locus {window.locus_id} is assigned to multiple splits"
                )
            batch_entries[window.locus_id] = row

        assert self._sqlite_conn is not None
        conn = self._sqlite_conn
        locus_ids = list(batch_entries.keys())
        chunk_size = 500
        for start in range(0, len(locus_ids), chunk_size):
            chunk = locus_ids[start : start + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cursor = conn.execute(
                f"SELECT locus_id, split FROM locus_entries WHERE locus_id IN ({placeholders})",
                chunk,
            )
            for locus_id, prior_split in cursor.fetchall():
                batch_split = batch_entries[locus_id][3]
                if prior_split != batch_split:
                    raise SplitLeakageError(
                        f"Locus {locus_id} is assigned to multiple splits"
                    )
        conn.executemany(
            "INSERT OR IGNORE INTO locus_entries "
            "(locus_id, contig, block_start, block_end, split) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (locus_id, contig, block_start, block_end, split)
                for locus_id, (contig, block_start, block_end, split) in batch_entries.items()
            ],
        )
        conn.commit()

        partitioned: dict[tuple[str, str, str], list[TokenizedWindow]] = defaultdict(list)
        for record in sorted_batch:
            partitioned[_partition_tuple(record.window)].append(record)

        if self._parquet_backend is None:
            self._parquet_backend = _load_pyarrow_parquet()
        pyarrow, pyarrow_parquet = self._parquet_backend

        for partition_key in sorted(partitioned):
            split, contig, block_id = partition_key
            partition_dir = (
                self._output_path
                / f"split={split}"
                / f"contig={contig}"
                / f"block_id={block_id}"
            )
            partition_dir.mkdir(parents=True, exist_ok=True)
            partition_records = partitioned[partition_key]
            row_group_size = self._contract.row_group_size
            for chunk_index, chunk_start in enumerate(
                range(0, len(partition_records), row_group_size)
            ):
                chunk = partition_records[chunk_start : chunk_start + row_group_size]
                file_path = (
                    partition_dir
                    / f"part-{self._batch_index:05d}-{chunk_index:05d}.parquet"
                )
                table = pyarrow.Table.from_pylist(
                    [_export_record(record, contract=self._contract) for record in chunk]
                )
                pyarrow_parquet.write_table(
                    table,
                    file_path,
                    row_group_size=row_group_size,
                )
                self._written_files.append(file_path)
                self._split_paths[split].append(file_path)
                self._split_record_counts[split] += len(chunk)

        self._batch_index += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Flush and close the writer.

        On a clean close (no exception in the ``with`` block), emit the
        corpus ``metadata.json`` sidecar by streaming ``split_manifest``
        rows from the SQLite sidecar ordered by
        ``(contig, block_start, split)``. On an exception, delete any
        Parquet files that were already written during this session so
        the output directory never contains half-written partitions
        that would corrupt a downstream train loader. Empty partition
        directories and the output root are left in place; only the
        Parquet artifacts are removed.

        The SQLite sidecar is unconditionally closed and deleted on
        every exit path (success, internal failure, caller exception)
        so it never leaks into the deliverable tree.

        Args:
            exc_type: Type of the propagating exception, or ``None``.
            exc_val: Value of the propagating exception, or ``None``.
            exc_tb: Traceback of the propagating exception, or ``None``.
        """
        self._closed = True
        try:
            if exc_type is not None:
                for path in self._written_files:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                return None
            self._write_metadata_json()
            return None
        finally:
            self._teardown_sqlite_sidecar()

    def _teardown_sqlite_sidecar(self) -> None:
        """Close the SQLite connection and remove the sidecar file.

        Intent: ensure the scratch database never leaks into the
        output tree regardless of which exit path triggered teardown,
        so downstream packagers and auditors see only the canonical
        Parquet + ``metadata.json`` deliverable.
        """
        if self._sqlite_conn is not None:
            try:
                self._sqlite_conn.close()
            except sqlite3.Error:
                pass
            self._sqlite_conn = None
        if self._sqlite_path is not None:
            try:
                self._sqlite_path.unlink()
            except FileNotFoundError:
                pass
            self._sqlite_path = None

    def _write_metadata_json(self) -> None:
        """Write ``metadata.json`` with a streamed ``split_manifest`` array.

        Intent: avoid materialising the full manifest as a Python list
        (~O(total-windows)) to keep peak RAM at close near the SQLite
        row-factory buffer rather than the full 7M-locus dict. The
        non-manifest head is serialised with ``json.dumps(sort_keys=True,
        indent=2)`` so top-level key order is byte-identical to the prior
        implementation; ``split_manifest`` is spliced in immediately
        before ``"splits"`` (its lexicographic successor) by streaming
        rows from the SQLite sidecar in
        ``(contig, block_start, split)`` order.
        """
        metadata_provenance = self._resolve_final_provenance()
        metadata_head = {
            "access_pattern": self._contract.access_pattern,
            "deterministic_partition_keys": list(
                self._contract.deterministic_partition_keys
            ),
            "export_format": self._contract.format,
            "preserve_coordinates": self._contract.preserve_coordinates,
            "preserve_raw_windows": self._contract.preserve_raw_windows,
            "preserve_sequence_hashes": self._contract.preserve_sequence_hashes,
            "row_group_size": self._contract.row_group_size,
            "sequence_hash_algorithm": self._contract.sequence_hash_algorithm,
            "tokenizer": asdict(metadata_provenance),
            "splits": {
                split: {
                    "record_count": self._split_record_counts[split],
                    "files": sorted(
                        str(path.relative_to(self._output_path)) for path in paths
                    ),
                }
                for split, paths in sorted(self._split_paths.items())
            },
        }
        head_serialized = json.dumps(metadata_head, indent=2, sort_keys=True)
        splice_marker = '\n  "splits":'
        splice_index = head_serialized.index(splice_marker)
        prefix = head_serialized[:splice_index]
        suffix = head_serialized[splice_index:]

        assert self._sqlite_conn is not None
        cursor = self._sqlite_conn.execute(
            "SELECT locus_id, contig, block_start, block_end, split "
            "FROM locus_entries "
            "ORDER BY contig, block_start, split"
        )
        count_cursor = self._sqlite_conn.execute(
            "SELECT COUNT(*) FROM locus_entries"
        )
        total_rows = count_cursor.fetchone()[0]

        metadata_path = self._output_path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            handle.write(prefix)
            handle.write("\n")
            if total_rows == 0:
                handle.write('  "split_manifest": []')
            else:
                handle.write('  "split_manifest": [\n')
                fetch_chunk = 500
                emitted = 0
                while True:
                    rows = cursor.fetchmany(fetch_chunk)
                    if not rows:
                        break
                    for locus_id, contig, block_start, block_end, split in rows:
                        entry = {
                            "block_end": block_end,
                            "block_start": block_start,
                            "contig": contig,
                            "locus_id": locus_id,
                            "split": split,
                        }
                        body = json.dumps(entry, indent=2, sort_keys=True)
                        indented = "\n".join("    " + line for line in body.splitlines())
                        handle.write(indented)
                        emitted += 1
                        if emitted < total_rows:
                            handle.write(",\n")
                        else:
                            handle.write("\n")
                handle.write("  ]")
            handle.write(",")
            handle.write(suffix)

    @property
    def split_paths(self) -> dict[str, list[Path]]:
        """Return a copy of the split-to-Parquet-paths mapping written so far.

        Intent: surface the per-split file list to the shim so the
        public ``write_tokenized_dataset`` return value shape remains
        unchanged for legacy callers.
        """
        return {split: list(paths) for split, paths in self._split_paths.items()}

    def _resolve_final_provenance(self) -> TokenizerProvenance:
        """Resolve the tokenizer provenance to embed in the corpus manifest.

        If at least one non-empty batch was written, the provenance
        validated at that point is returned. For a writer that never
        received a non-empty batch, ``provenance`` must have been
        supplied at construction time, matching the legacy contract
        where an empty export requires an explicit provenance.
        """
        if self._resolved_provenance is not None:
            return self._resolved_provenance
        if self._explicit_provenance is None:
            raise ExportContractError(
                "Tokenized export metadata requires explicit tokenizer provenance "
                "when no tokenized windows are available"
            )
        _assert_approved_dnabert2_trust_policy(
            self._explicit_provenance.trust_remote_code
        )
        return self._explicit_provenance


def write_tokenized_dataset(
    tokenized_windows: tuple[TokenizedWindow, ...],
    output_dir: str | Path,
    *,
    contract: ExportContract = DEFAULT_PARQUET_EXPORT_CONTRACT,
    provenance: TokenizerProvenance | None = None,
) -> dict[str, list[Path]]:
    """Write tokenised windows to a Hive-partitioned Parquet dataset (one-batch shim).

    Intent: preserve the legacy single-shot API for the consensus
    pretrain pipeline. This function is a thin wrapper that opens a
    :class:`TokenizedCorpusWriter`, calls ``write_batch`` once with the
    supplied windows, and closes. All on-disk artifacts (Hive partition
    layout, ``metadata.json`` schema, return-value shape) are produced
    by the underlying writer; this function adds no behaviour of its
    own beyond the single-batch call.

    Args:
        tokenized_windows: Records to serialise in a single batch.
        output_dir: Root directory for the Parquet dataset.
        contract: Export settings governing format, partitioning, and
            preservation flags.
        provenance: Optional explicit tokenizer provenance; inferred
            from *tokenized_windows* if ``None``.

    Returns:
        Mapping from split name to the list of written Parquet file paths.

    Raises:
        ExportContractError: If ``contract.format`` is not ``"parquet"``
            or provenance validation fails.
    """
    with TokenizedCorpusWriter(
        output_dir,
        contract=contract,
        provenance=provenance,
    ) as writer:
        writer.write_batch(tokenized_windows)
    return writer.split_paths


def write_webdataset_shards(
    tokenized_windows: tuple[TokenizedWindow, ...],
    output_dir: str | Path,
    *,
    records_per_shard: int | None = None,
    provenance: TokenizerProvenance | None = None,
) -> dict[str, list[Path]]:
    """Write tokenised windows as WebDataset ``.tar`` shards.

    Each shard contains JSON records with deterministic tar metadata
    (``mtime=0``, ``uid=0``, ``gid=0``) for reproducibility.  A
    ``metadata.json`` sidecar is written alongside the shards.

    Args:
        tokenized_windows: Records to serialise.
        output_dir: Root directory for the shard files.
        records_per_shard: Maximum records per ``.tar`` file; if ``None``,
            all records for a split are written to a single shard.
        provenance: Optional explicit tokenizer provenance.

    Returns:
        Mapping from split name to the list of written shard paths.

    Raises:
        ValueError: If *records_per_shard* is non-positive.
        ExportContractError: On provenance validation failure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    requested_records_per_shard = records_per_shard
    if requested_records_per_shard is not None and requested_records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")

    sorted_records = sorted(
        tokenized_windows,
        key=lambda item: (
            item.window.split,
            item.window.contig,
            item.window.block_start,
            item.window.window_start,
            item.window.sample_id,
            item.window.source,
        ),
    )
    metadata_provenance = _resolve_export_tokenizer_provenance(
        tuple(sorted_records),
        provenance=provenance,
    )

    shard_paths: dict[str, list[Path]] = defaultdict(list)
    split_records: dict[str, list[TokenizedWindow]] = defaultdict(list)
    split_counts: dict[str, int] = defaultdict(int)
    manifest = build_split_manifest(tuple(item.window for item in sorted_records))
    for record in sorted_records:
        split_records[record.window.split].append(record)

    for split in sorted(split_records):
        records = split_records[split]
        split_shard_size = requested_records_per_shard or max(1, len(records))
        for index in range(0, len(records), split_shard_size):
            shard_records = records[index : index + split_shard_size]
            shard_index = len(shard_paths[split])
            shard_path = output_path / f"{split}-{shard_index:05d}.tar"
            with tarfile.open(shard_path, mode="w") as archive:
                for record in shard_records:
                    split_counts[split] += 1
                    sample_key = f"{split_counts[split] - 1:08d}"
                    payload = json.dumps(
                        _export_record(
                            record,
                            contract=ExportContract(
                                format="webdataset",
                                row_group_size=split_shard_size,
                                preserve_raw_windows=True,
                                preserve_sequence_hashes=True,
                                preserve_coordinates=True,
                            ),
                        ),
                        sort_keys=True,
                    ).encode("utf-8")
                    tar_info = tarfile.TarInfo(name=f"{sample_key}.json")
                    tar_info.size = len(payload)
                    tar_info.mtime = 0
                    tar_info.uid = 0
                    tar_info.gid = 0
                    tar_info.uname = ""
                    tar_info.gname = ""
                    archive.addfile(tar_info, BytesIO(payload))
            shard_paths[split].append(shard_path)

    metadata = {
        "export_format": "webdataset",
        "records_per_shard": requested_records_per_shard,
        "tokenizer": asdict(metadata_provenance),
        "splits": {
            split: {
                "record_count": len(records),
                "shards": [path.name for path in paths],
            }
            for split, records in sorted(split_records.items())
            for paths in [shard_paths[split]]
        },
        "split_manifest": [asdict(entry) for entry in manifest],
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {split: paths for split, paths in shard_paths.items()}


def _coerce_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    """Coerce tokenizer output to a flat ``tuple[int, ...]``.

    Handles NumPy arrays (via ``.tolist()``), nested single-element
    batches (``[[1, 2, 3]]``), and plain lists.

    Args:
        value: Raw tokenizer output for a single field.
        field_name: Human-readable label for error messages.

    Returns:
        A flat tuple of integers.

    Raises:
        TokenizerContractError: If *value* is ``None``, not coercible
            to a flat int list, or contains non-integer elements.
    """
    if value is None:
        raise TokenizerContractError(f"Tokenizer output is missing {field_name}")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise TokenizerContractError(f"{field_name} must be a flat list of integers")
    return tuple(value)


def _prepare_window_for_tokenization(
    window: WindowRecord,
    provenance: TokenizerProvenance,
) -> WindowRecord:
    """Re-normalise a window's sequence against the tokenizer's alphabet.

    If the normalised sequence differs from the original, a new
    ``WindowRecord`` is returned with updated sequence, GC fraction,
    ambiguity fraction, and sequence hash.

    Args:
        window: The window to prepare.
        provenance: Tokenizer provenance controlling the alphabet and
            unsupported-symbol policy.

    Returns:
        The original *window* if no changes were needed, or a new
        ``WindowRecord`` with the re-normalised sequence.
    """
    normalized_sequence = _normalize_tokenizer_sequence(window.sequence, provenance)
    if normalized_sequence == window.sequence:
        return window
    return WindowRecord(
        sample_id=window.sample_id,
        individual_id=window.individual_id,
        contig=window.contig,
        source=window.source,
        split=window.split,
        locus_id=window.locus_id,
        block_start=window.block_start,
        block_end=window.block_end,
        window_start=window.window_start,
        window_end=window.window_end,
        sequence=normalized_sequence,
        gc_fraction=gc_fraction(normalized_sequence),
        ambiguity_fraction=ambiguity_fraction(normalized_sequence),
        sequence_hash=sha256(normalized_sequence.encode("utf-8")).hexdigest(),
        unique_masked_bases=window.unique_masked_bases,
        filtered_bases=window.filtered_bases,
        no_call_bases=window.no_call_bases,
        other_masked_bases=window.other_masked_bases,
        masked_base_counts=window.masked_base_counts,
    )


def _format_window_context(window: WindowRecord) -> str:
    """Format window metadata as a human-readable string for error messages.

    Args:
        window: The window to describe.

    Returns:
        Comma-separated key=value summary.
    """
    return (
        f"sample_id={window.sample_id}, "
        f"individual_id={window.individual_id}, "
        f"source={window.source}, "
        f"contig={window.contig}, "
        f"locus_id={window.locus_id}, "
        f"window={window.window_start}-{window.window_end}"
    )


def _normalize_tokenizer_sequence(sequence: str, provenance: TokenizerProvenance) -> str:
    """Re-normalise *sequence* using the tokenizer's alphabet policy.

    Translates ``provenance.unsupported_symbol_policy`` into the
    ``ambiguity_mode`` parameter expected by ``normalize_sequence``,
    and wraps ``PreprocessingError`` in ``TokenizerContractError``.

    Args:
        sequence: Nucleotide string to normalise.
        provenance: Tokenizer provenance supplying the alphabet and policy.

    Returns:
        Normalised sequence.

    Raises:
        TokenizerContractError: If the sequence contains unsupported bases.
    """
    ambiguity_mode = "reject" if provenance.unsupported_symbol_policy == "reject" else "mask"
    try:
        return normalize_sequence(
            sequence,
            ambiguity_mode=ambiguity_mode,
            allowed_alphabet=provenance.allowed_alphabet,
        )
    except PreprocessingError as exc:
        raise TokenizerContractError(str(exc)) from exc


def _partition_tuple(window: WindowRecord) -> tuple[str, str, str]:
    """Extract the Hive-partition key ``(split, contig, block_id)`` for a window.

    Args:
        window: The window record.

    Returns:
        Three-element tuple used as a dictionary key for partitioned export.
    """
    return window.split, window.contig, f"{window.block_start}-{window.block_end}"


def _load_pyarrow_parquet() -> tuple[Any, Any]:
    """Lazily import ``pyarrow`` and ``pyarrow.parquet``.

    Returns:
        ``(pyarrow, pyarrow.parquet)`` module tuple.

    Raises:
        ExportContractError: If PyArrow is not installed.
    """
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError as exc:
        raise ExportContractError(
            "Parquet export requires pyarrow. Install with: uv add pyarrow"
        ) from exc
    return pyarrow, pyarrow.parquet


def _export_record(record: TokenizedWindow, *, contract: ExportContract) -> dict[str, Any]:
    """Serialise a ``TokenizedWindow`` into a flat export dictionary.

    Args:
        record: The tokenised window to export.
        contract: Export settings controlling which fields are preserved.

    Returns:
        Dictionary suitable for Parquet row or WebDataset JSON payload.
    """
    return {
        "attention_mask": list(record.attention_mask),
        "input_ids": list(record.input_ids),
        "token_count": record.token_count,
        "token_to_base_ratio": record.token_to_base_ratio,
        "tokenizer": asdict(record.tokenizer),
        "window": _export_window(record.window, contract=contract),
    }


def _export_window(window: WindowRecord, *, contract: ExportContract) -> dict[str, Any]:
    """Serialise a ``WindowRecord`` into a dictionary respecting *contract*.

    Conditionally includes coordinates, raw sequence, and sequence hash
    based on the ``ExportContract`` preservation flags.

    Args:
        window: The window to serialise.
        contract: Export settings.

    Returns:
        Dictionary of window metadata and (optionally) sequence data.
    """
    payload: dict[str, Any] = {
        "ambiguity_fraction": window.ambiguity_fraction,
        "gc_fraction": window.gc_fraction,
        "individual_id": window.individual_id,
        "sample_id": window.sample_id,
        "split": window.split,
        "source": window.source,
    }
    if contract.preserve_coordinates:
        payload.update(
            {
                "block_end": window.block_end,
                "block_start": window.block_start,
                "contig": window.contig,
                "locus_id": window.locus_id,
                "window_end": window.window_end,
                "window_start": window.window_start,
            }
        )
    if contract.preserve_raw_windows:
        payload["sequence"] = window.sequence
    if contract.preserve_sequence_hashes:
        payload["sequence_hash"] = window.sequence_hash
    return payload