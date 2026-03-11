"""Preprocessing, split-safety, tokenization, and export helpers for felid corpora."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile
from typing import Any, Protocol

from .pipeline_contract import (
    DNABERT2_TOKENIZER_ID,
    DNABERT2_TOKENIZER_REVISION as DNABERT2_TOKENIZER_REVISION_HASH,
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
    def __call__(self, sequence: str, **kwargs: Any) -> dict[str, Any]: ...


class PreprocessingError(ValueError):
    """Raised when sequence normalization or preprocessing validation fails."""


class SplitLeakageError(ValueError):
    """Raised when genomic windows would leak across splits."""


class TokenizerContractError(ValueError):
    """Raised when tokenizer output violates the export contract."""


class ExportContractError(ValueError):
    """Raised when export settings violate the approved artifact contract."""


@dataclass(frozen=True)
class TokenizerProvenance:
    identifier: str = DNABERT2_TOKENIZER_NAME
    revision: str = DNABERT2_TOKENIZER_REVISION
    max_position_embeddings: int = DNABERT2_MAX_POSITION_EMBEDDINGS
    allowed_alphabet: tuple[str, ...] = ALLOWED_DNA_ALPHABET
    unsupported_symbol_policy: str = DEFAULT_UNSUPPORTED_SYMBOL_POLICY
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        if self.identifier != DNABERT2_TOKENIZER_NAME:
            raise ValueError("Tokenizer provenance must pin the approved DNABERT-2 identifier")
        if self.revision != DNABERT2_TOKENIZER_REVISION:
            raise ValueError("Tokenizer provenance must pin the approved DNABERT-2 revision")
        if tuple(self.allowed_alphabet) != ALLOWED_DNA_ALPHABET:
            raise ValueError("Tokenizer provenance allowed_alphabet must match A/C/G/T/N")
        if self.unsupported_symbol_policy not in {"reject", "normalize_to_n"}:
            raise ValueError("unsupported_symbol_policy must be reject or normalize_to_n")


DNABERT2_TOKENIZER_PROVENANCE = TokenizerProvenance()


@dataclass(frozen=True)
class PreprocessingConfig:
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
    sample_id: str
    individual_id: str
    contig: str
    sequence: str
    source: str = "consensus"
    sequence_start: int = 0
    mask_spans: tuple[tuple[int, int, str], ...] = ()

    @property
    def sequence_end(self) -> int:
        return self.sequence_start + len(normalize_sequence(self.sequence, ambiguity_mode="mask"))


@dataclass(frozen=True)
class PreparedSequence:
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
        return self.sequence_start + len(self.sequence)


@dataclass(frozen=True)
class FilteredSequence:
    sample_id: str
    individual_id: str
    contig: str
    source: str
    reason: str
    sequence_length: int
    ambiguity_fraction: float


@dataclass(frozen=True)
class PreprocessingReport:
    retained: tuple[PreparedSequence, ...]
    filtered: tuple[FilteredSequence, ...]
    mean_gc_fraction: float
    mean_ambiguity_fraction: float


@dataclass(frozen=True)
class WindowRecord:
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
    filtered_bases: int = 0
    no_call_bases: int = 0
    other_masked_bases: int = 0
    masked_base_counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class SplitManifestEntry:
    locus_id: str
    contig: str
    block_start: int
    block_end: int
    split: str


@dataclass(frozen=True)
class TokenizedWindow:
    window: WindowRecord
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_count: int
    token_to_base_ratio: float
    tokenizer: TokenizerProvenance


@dataclass(frozen=True)
class ExportContract:
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
    """Normalize genomic sequence characters deterministically before tokenization."""

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
    canonical_bases = sum(1 for base in sequence if base in {"A", "C", "G", "T"})
    if canonical_bases == 0:
        return 0.0
    gc_bases = sum(1 for base in sequence if base in {"G", "C"})
    return gc_bases / canonical_bases


def ambiguity_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sequence.count("N") / len(sequence)


@dataclass
class _WindowMaskCounter:
    mask_spans: tuple[tuple[int, int, str], ...]
    next_index: int = 0
    active_spans: list[tuple[int, int, str]] | None = None

    def __post_init__(self) -> None:
        self.active_spans = []

    def count(self, *, window_start: int, window_end: int) -> Counter[str]:
        assert self.active_spans is not None
        while self.next_index < len(self.mask_spans) and self.mask_spans[self.next_index][0] < window_end:
            self.active_spans.append(self.mask_spans[self.next_index])
            self.next_index += 1
        self.active_spans = [span for span in self.active_spans if span[1] > window_start]

        counts: Counter[str] = Counter()
        for span_start, span_end, category in self.active_spans:
            overlap = min(window_end, span_end) - max(window_start, span_start)
            if overlap > 0:
                counts[category] += overlap
        return counts


def prepare_sequences(
    records: list[SequenceRecord],
    config: PreprocessingConfig,
) -> PreprocessingReport:
    retained: list[PreparedSequence] = []
    filtered: list[FilteredSequence] = []

    for record in records:
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
    total = sum(weight for _, weight in split_weights)
    bucket = int(sha256(f"{split_seed}:{locus_id}".encode("utf-8")).hexdigest()[:16], 16)
    position = bucket / float(16**16)
    cumulative = 0.0
    for split_name, weight in split_weights:
        cumulative += weight / total
        if position < cumulative:
            return split_name
    return split_weights[-1][0]


def window_sequences(
    sequences: list[PreparedSequence],
    config: PreprocessingConfig,
) -> tuple[WindowRecord, ...]:
    windows: list[WindowRecord] = []
    for sequence in sequences:
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
                mask_counts = mask_counter.count(window_start=window_start, window_end=window_end)
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
                        filtered_bases=filtered_bases,
                        no_call_bases=no_call_bases,
                        other_masked_bases=other_masked_bases,
                        masked_base_counts=tuple(sorted(mask_counts.items())),
                    )
                )
    assert_split_safety(tuple(windows))
    return tuple(windows)


def build_split_manifest(windows: tuple[WindowRecord, ...]) -> tuple[SplitManifestEntry, ...]:
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


def load_dnabert2_tokenizer() -> tuple[TokenizerLike, TokenizerProvenance]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised through integration, not unit logic
        raise RuntimeError(
            "DNABERT-2 tokenization requires transformers. Install with: "
            "uv add \"transformers>=4.28,<5\""
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        DNABERT2_TOKENIZER_PROVENANCE.identifier,
        revision=DNABERT2_TOKENIZER_PROVENANCE.revision,
        trust_remote_code=DNABERT2_TOKENIZER_PROVENANCE.trust_remote_code,
    )
    return tokenizer, DNABERT2_TOKENIZER_PROVENANCE


def tokenize_windows(
    windows: tuple[WindowRecord, ...],
    tokenizer: TokenizerLike,
    provenance: TokenizerProvenance = DNABERT2_TOKENIZER_PROVENANCE,
) -> tuple[TokenizedWindow, ...]:
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
                f"Tokenized length {len(input_ids)} exceeds max_position_embeddings "
                f"{provenance.max_position_embeddings} for {window.locus_id}"
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


def write_tokenized_dataset(
    tokenized_windows: tuple[TokenizedWindow, ...],
    output_dir: str | Path,
    *,
    contract: ExportContract = DEFAULT_PARQUET_EXPORT_CONTRACT,
) -> dict[str, list[Path]]:
    if contract.format != "parquet":
        raise ExportContractError(
            "write_tokenized_dataset only supports parquet contracts; "
            "use write_webdataset_shards for webdataset exports"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
    manifest = build_split_manifest(tuple(item.window for item in sorted_records))

    split_paths: dict[str, list[Path]] = defaultdict(list)
    partitioned_records: dict[tuple[str, str, str], list[TokenizedWindow]] = defaultdict(list)
    for record in sorted_records:
        partitioned_records[_partition_tuple(record.window)].append(record)

    parquet_backend: tuple[Any, Any] | None = _load_pyarrow_parquet() if sorted_records else None

    for partition_key in sorted(partitioned_records):
        split, contig, block_id = partition_key
        partition_path = output_path / f"split={split}" / f"contig={contig}" / f"block_id={block_id}"
        partition_path.mkdir(parents=True, exist_ok=True)
        records = partitioned_records[partition_key]
        for index in range(0, len(records), contract.row_group_size):
            row_group = records[index : index + contract.row_group_size]
            file_path = partition_path / f"part-{index // contract.row_group_size:05d}.parquet"
            pyarrow, pyarrow_parquet = parquet_backend
            table = pyarrow.Table.from_pylist(
                [_export_record(record, contract=contract) for record in row_group]
            )
            pyarrow_parquet.write_table(table, file_path, row_group_size=contract.row_group_size)
            split_paths[split].append(file_path)

    metadata = {
        "access_pattern": contract.access_pattern,
        "deterministic_partition_keys": list(contract.deterministic_partition_keys),
        "export_format": contract.format,
        "preserve_coordinates": contract.preserve_coordinates,
        "preserve_raw_windows": contract.preserve_raw_windows,
        "preserve_sequence_hashes": contract.preserve_sequence_hashes,
        "row_group_size": contract.row_group_size,
        "sequence_hash_algorithm": contract.sequence_hash_algorithm,
        "tokenizer": asdict(sorted_records[0].tokenizer) if sorted_records else asdict(DNABERT2_TOKENIZER_PROVENANCE),
        "splits": {
            split: {
                "record_count": len([item for item in sorted_records if item.window.split == split]),
                "files": [str(path.relative_to(output_path)) for path in paths],
            }
            for split, paths in sorted(split_paths.items())
        },
        "split_manifest": [asdict(entry) for entry in manifest],
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {split: paths for split, paths in split_paths.items()}


def write_webdataset_shards(
    tokenized_windows: tuple[TokenizedWindow, ...],
    output_dir: str | Path,
    *,
    records_per_shard: int | None = None,
) -> dict[str, list[Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if records_per_shard is None:
        records_per_shard = max(1, len(tokenized_windows) or 1)
    if records_per_shard <= 0:
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

    shard_paths: dict[str, list[Path]] = defaultdict(list)
    split_counts: dict[str, int] = defaultdict(int)
    manifest = build_split_manifest(tuple(item.window for item in sorted_records))

    for index in range(0, len(sorted_records), records_per_shard):
        shard_records = sorted_records[index : index + records_per_shard]
        split = shard_records[0].window.split if shard_records else "empty"
        shard_index = len(shard_paths[split])
        shard_path = output_path / f"{split}-{shard_index:05d}.tar"
        with tarfile.open(shard_path, mode="w") as archive:
            for record_offset, record in enumerate(shard_records):
                split_counts[split] += 1
                sample_key = f"{split_counts[split] - 1:08d}"
                payload = json.dumps(
                    _export_record(
                        record,
                        contract=ExportContract(
                            format="webdataset",
                            row_group_size=records_per_shard,
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
        "records_per_shard": records_per_shard,
        "tokenizer": asdict(DNABERT2_TOKENIZER_PROVENANCE),
        "splits": {
            split: {
                "record_count": len([item for item in sorted_records if item.window.split == split]),
                "shards": [path.name for path in paths],
            }
            for split, paths in sorted(shard_paths.items())
        },
        "split_manifest": [asdict(entry) for entry in manifest],
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {split: paths for split, paths in shard_paths.items()}


def _coerce_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
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
        filtered_bases=window.filtered_bases,
        no_call_bases=window.no_call_bases,
        other_masked_bases=window.other_masked_bases,
        masked_base_counts=window.masked_base_counts,
    )


def _normalize_tokenizer_sequence(sequence: str, provenance: TokenizerProvenance) -> str:
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
    return window.split, window.contig, f"{window.block_start}-{window.block_end}"


def _load_pyarrow_parquet() -> tuple[Any, Any]:
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError as exc:
        raise ExportContractError(
            "Parquet export requires pyarrow. Install with: uv add pyarrow"
        ) from exc
    return pyarrow, pyarrow.parquet


def _export_record(record: TokenizedWindow, *, contract: ExportContract) -> dict[str, Any]:
    return {
        "attention_mask": list(record.attention_mask),
        "input_ids": list(record.input_ids),
        "token_count": record.token_count,
        "token_to_base_ratio": record.token_to_base_ratio,
        "tokenizer": asdict(record.tokenizer),
        "window": _export_window(record.window, contract=contract),
    }


def _export_window(window: WindowRecord, *, contract: ExportContract) -> dict[str, Any]:
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