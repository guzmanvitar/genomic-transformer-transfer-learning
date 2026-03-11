"""Runtime wiring for the fixture-backed feline pretraining data pipeline."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
from typing import Callable, Iterable

from ..config import FelinePipelineConfig, load_feline_pipeline_config
from ..data.acquisition import ConsensusResult, generate_consensus_fastas
from ..data.pipeline_contract import REQUIRED_SAMPLE_MANIFEST_FIELDS
from ..data.preprocessor import (
    ExportContract,
    PreprocessingConfig,
    SequenceRecord,
    TokenizedWindow,
    TokenizerProvenance,
    load_dnabert2_tokenizer,
    prepare_sequences,
    tokenize_windows,
    write_tokenized_dataset,
    window_sequences,
)
from ..reporting import build_eda_payload

TokenizerLoader = Callable[[], tuple[object, TokenizerProvenance]]
ExportWriter = Callable[[tuple[TokenizedWindow, ...], Path, FelinePipelineConfig], Path]
DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT = 128


@dataclass(frozen=True)
class FelineSampleManifestEntry:
    sample_id: str
    individual_id: str
    vcf_path: Path


@dataclass(frozen=True)
class FelinePretrainArtifacts:
    consensus_dir: Path
    consensus_export: Path
    baseline_export: Path
    diagnostics_path: Path
    summary_path: Path


@dataclass(frozen=True)
class FelinePretrainRunResult:
    config_name: str
    sample_count: int
    consensus_window_count: int
    baseline_window_count: int
    consensus_fastas: tuple[Path, ...]
    artifacts: FelinePretrainArtifacts


def run_feline_pretrain_pipeline(
    config_path: str | Path,
    *,
    bcftools_executable: str = "bcftools",
    tokenizer_loader: TokenizerLoader | None = None,
    export_writer: ExportWriter | None = None,
) -> FelinePretrainRunResult:
    tokenizer_loader = tokenizer_loader or load_dnabert2_tokenizer
    export_writer = export_writer or write_tokenized_corpus

    config_file = Path(config_path).resolve()
    config = load_feline_pipeline_config(config_file)
    config_root = config_file.parent

    reference_fasta = _resolve_path(config_root, config.paths.reference_fasta, prefer_cwd=True)
    manifest_path = _resolve_path(config_root, config.paths.sample_manifest, prefer_cwd=True)
    processed_dir = _resolve_path(config_root, config.paths.processed_dir, prefer_cwd=True)
    baseline_dir = _resolve_path(config_root, config.paths.baseline_dir, prefer_cwd=True)
    artifact_dir = _resolve_path(config_root, config.paths.artifact_dir, prefer_cwd=True)
    report_dir = _resolve_path(config_root, config.paths.report_dir, prefer_cwd=True)

    _require_existing_file(reference_fasta, "reference FASTA")
    manifest_entries = load_feline_sample_manifest(manifest_path)
    for entry in manifest_entries:
        _require_existing_file(entry.vcf_path, f"VCF for sample '{entry.sample_id}'")

    consensus_dir = processed_dir / "consensus_fastas"
    consensus_results = generate_consensus_fastas(
        reference_fasta=reference_fasta,
        sample_vcfs={entry.sample_id: entry.vcf_path for entry in manifest_entries},
        output_dir=consensus_dir,
        bcftools_executable=bcftools_executable,
    )

    reference_sequences = _load_fasta_sequences(reference_fasta)
    preprocessing_config = _build_preprocessing_config(config)

    consensus_records = _build_consensus_sequence_records(consensus_results, manifest_entries)
    baseline_records = _build_reference_sequence_records(reference_sequences, manifest_entries)

    prepared_consensus = prepare_sequences(consensus_records, preprocessing_config)
    prepared_baseline = prepare_sequences(baseline_records, preprocessing_config)
    consensus_windows = window_sequences(list(prepared_consensus.retained), preprocessing_config)
    baseline_windows = window_sequences(list(prepared_baseline.retained), preprocessing_config)
    if not consensus_windows:
        raise RuntimeError("No consensus windows survived preprocessing; check sequence length and ambiguity filters")
    if not baseline_windows:
        raise RuntimeError("No baseline windows survived preprocessing; check the reference FASTA and window settings")

    tokenizer, provenance = tokenizer_loader()
    _assert_tokenizer_matches_config(config, provenance)
    tokenized_consensus = tokenize_windows(consensus_windows, tokenizer, provenance=provenance)
    tokenized_baseline = tokenize_windows(baseline_windows, tokenizer, provenance=provenance)

    consensus_export = export_writer(tokenized_consensus, processed_dir / "consensus_tokens", config)
    baseline_export = export_writer(tokenized_baseline, baseline_dir / "reference_tokens", config)

    diagnostics_payload = _build_diagnostics_payload(
        tokenized_consensus=tokenized_consensus,
        tokenized_baseline=tokenized_baseline,
        consensus_results=consensus_results,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = report_dir / "eda_payload.json"
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2, sort_keys=True), encoding="utf-8")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "pretrain_run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config_name": config.name,
                "sample_count": len(manifest_entries),
                "consensus_window_count": len(tokenized_consensus),
                "baseline_window_count": len(tokenized_baseline),
                "consensus_fastas": [
                    str(consensus_results[entry.sample_id].output_fasta) for entry in manifest_entries
                ],
                "consensus_export": str(consensus_export),
                "baseline_export": str(baseline_export),
                "diagnostics_path": str(diagnostics_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return FelinePretrainRunResult(
        config_name=config.name,
        sample_count=len(manifest_entries),
        consensus_window_count=len(tokenized_consensus),
        baseline_window_count=len(tokenized_baseline),
        consensus_fastas=tuple(consensus_results[entry.sample_id].output_fasta for entry in manifest_entries),
        artifacts=FelinePretrainArtifacts(
            consensus_dir=consensus_dir,
            consensus_export=consensus_export,
            baseline_export=baseline_export,
            diagnostics_path=diagnostics_path,
            summary_path=summary_path,
        ),
    )


def format_feline_pretrain_result(result: FelinePretrainRunResult) -> str:
    return "\n".join(
        [
            f"Feline pretrain artifact generation finished for '{result.config_name}'.",
            f"Samples: {result.sample_count}",
            f"Consensus windows: {result.consensus_window_count}",
            f"Baseline windows: {result.baseline_window_count}",
            f"Consensus export: {result.artifacts.consensus_export}",
            f"Baseline export: {result.artifacts.baseline_export}",
            f"Diagnostics: {result.artifacts.diagnostics_path}",
        ]
    )


def load_feline_sample_manifest(path: str | Path) -> tuple[FelineSampleManifestEntry, ...]:
    manifest_path = Path(path)
    _require_existing_file(manifest_path, "sample manifest")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Sample manifest {manifest_path} must include a header row")
        missing_fields = [field for field in REQUIRED_SAMPLE_MANIFEST_FIELDS if field not in reader.fieldnames]
        if missing_fields:
            raise ValueError(
                "Sample manifest must include columns: "
                + ", ".join(REQUIRED_SAMPLE_MANIFEST_FIELDS)
                + f" (missing: {', '.join(missing_fields)})"
            )

        entries: list[FelineSampleManifestEntry] = []
        seen_sample_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            individual_id = (row.get("individual_id") or "").strip()
            vcf_path_raw = (row.get("vcf_path") or "").strip()
            if not sample_id or not individual_id or not vcf_path_raw:
                raise ValueError(
                    f"Sample manifest row {row_number} must define sample_id, individual_id, and vcf_path"
                )
            if sample_id in seen_sample_ids:
                raise ValueError(f"Sample manifest contains duplicate sample_id '{sample_id}'")
            seen_sample_ids.add(sample_id)
            entries.append(
                FelineSampleManifestEntry(
                    sample_id=sample_id,
                    individual_id=individual_id,
                    vcf_path=_resolve_path(manifest_path.parent, Path(vcf_path_raw)),
                )
            )
    if not entries:
        raise ValueError(f"Sample manifest {manifest_path} must contain at least one sample row")
    return tuple(entries)


def write_tokenized_corpus(
    tokenized_windows: tuple[TokenizedWindow, ...],
    output_path: str | Path,
    config: FelinePipelineConfig,
) -> Path:
    destination = Path(output_path)
    write_tokenized_dataset(
        tokenized_windows,
        destination,
        contract=_build_export_contract(config),
    )
    return destination


def _build_preprocessing_config(config: FelinePipelineConfig) -> PreprocessingConfig:
    return PreprocessingConfig(
        min_sequence_length=config.windowing.context_window if config.windowing.drop_short_sequences else 1,
        max_ambiguity_fraction=config.windowing.max_ambiguous_fraction,
        window_size=config.windowing.context_window,
        window_stride=config.windowing.context_window - config.windowing.window_overlap,
        locus_block_size=config.split.locus_block_size,
        ambiguity_mode=(
            "reject" if config.tokenizer.unsupported_symbol_policy == "reject" else "mask"
        ),
    )


def _assert_tokenizer_matches_config(
    config: FelinePipelineConfig, provenance: TokenizerProvenance
) -> None:
    if provenance.identifier != config.tokenizer.identifier:
        raise RuntimeError(
            f"Tokenizer loader returned {provenance.identifier}, expected {config.tokenizer.identifier}"
        )
    if provenance.revision != config.tokenizer.revision:
        raise RuntimeError(
            f"Tokenizer loader returned revision {provenance.revision}, expected {config.tokenizer.revision}"
        )
    if tuple(provenance.allowed_alphabet) != config.tokenizer.allowed_alphabet:
        raise RuntimeError("Tokenizer loader returned an alphabet that does not match the config contract")
    if provenance.max_position_embeddings != config.tokenizer.max_position_embeddings:
        raise RuntimeError(
            "Tokenizer loader max_position_embeddings does not match the approved config"
        )
    if provenance.unsupported_symbol_policy != config.tokenizer.unsupported_symbol_policy:
        raise RuntimeError(
            "Tokenizer loader unsupported_symbol_policy does not match the approved config"
        )


def _build_export_contract(config: FelinePipelineConfig) -> ExportContract:
    return ExportContract(
        format=config.export.format,
        access_pattern=config.export.access_pattern,
        row_group_size=config.export.row_group_size,
        deterministic_partition_keys=config.export.deterministic_partition_keys,
        preserve_raw_windows=config.export.preserve_raw_windows,
        preserve_sequence_hashes=config.export.preserve_sequence_hashes,
        preserve_coordinates=config.export.preserve_coordinates,
        sequence_hash_algorithm=config.export.sequence_hash_algorithm,
    )


def _build_consensus_sequence_records(
    consensus_results: dict[str, ConsensusResult],
    manifest_entries: tuple[FelineSampleManifestEntry, ...],
) -> list[SequenceRecord]:
    individual_by_sample = {entry.sample_id: entry.individual_id for entry in manifest_entries}
    records: list[SequenceRecord] = []
    for sample_id, result in sorted(consensus_results.items()):
        mask_spans_by_contig: dict[str, list[tuple[int, int, str]]] = {}
        for span in result.mask_spans:
            mask_spans_by_contig.setdefault(span.contig, []).append((span.start, span.end, span.category))
        for contig, sequence in _load_fasta_sequences(result.output_fasta).items():
            records.append(
                SequenceRecord(
                    sample_id=sample_id,
                    individual_id=individual_by_sample[sample_id],
                    contig=contig,
                    sequence=sequence,
                    source="consensus",
                    sequence_start=0,
                    mask_spans=tuple(sorted(mask_spans_by_contig.get(contig, []))),
                )
            )
    return records


def _build_reference_sequence_records(
    reference_sequences: dict[str, str],
    manifest_entries: tuple[FelineSampleManifestEntry, ...],
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for entry in manifest_entries:
        for contig, sequence in reference_sequences.items():
            records.append(
                SequenceRecord(
                    sample_id=f"reference-{entry.sample_id}",
                    individual_id=entry.individual_id,
                    contig=contig,
                    sequence=sequence,
                    source="reference",
                    sequence_start=0,
                )
            )
    return records


def _build_diagnostics_payload(
    *,
    tokenized_consensus: tuple[TokenizedWindow, ...],
    tokenized_baseline: tuple[TokenizedWindow, ...],
    consensus_results: dict[str, ConsensusResult],
) -> dict[str, object]:
    reference_lookup = {
        _tokenized_window_lookup_key(record): record.window.sequence
        for record in tokenized_baseline
    }
    unmatched_consensus_window_count = sum(
        1 for record in tokenized_consensus if _tokenized_window_lookup_key(record) not in reference_lookup
    )
    return {
        "consensus_generation": {
            sample_id: asdict(result.diagnostics) for sample_id, result in sorted(consensus_results.items())
        },
        "baseline_window_alignment": {
            "matched_consensus_window_count": len(tokenized_consensus) - unmatched_consensus_window_count,
            "unmatched_consensus_window_count": unmatched_consensus_window_count,
        },
        **build_eda_payload(
            _tokenized_windows_to_diagnostics_records(
                tokenized_consensus,
                reference_lookup,
                allow_unmatched_reference=True,
            ),
            _tokenized_windows_to_diagnostics_records(tokenized_baseline, reference_lookup),
            consensus_sample_limit=DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT,
        ),
    }


def _tokenized_window_lookup_key(record: TokenizedWindow) -> tuple[str, int, int]:
    return record.window.contig, record.window.window_start, record.window.window_end


def _tokenized_windows_to_diagnostics_records(
    tokenized_windows: tuple[TokenizedWindow, ...],
    reference_lookup: dict[tuple[str, int, int], str],
    *,
    allow_unmatched_reference: bool = False,
) -> Iterable[dict[str, object]]:
    for record in tokenized_windows:
        key = _tokenized_window_lookup_key(record)
        reference_sequence = reference_lookup.get(key)
        reference_window_matched = reference_sequence is not None
        if reference_sequence is None:
            if not allow_unmatched_reference:
                raise KeyError(key)
            reference_sequence = record.window.sequence
        filtered_bases = record.window.filtered_bases
        no_call_bases = record.window.no_call_bases
        other_masked_bases = record.window.other_masked_bases
        variant_count = 0
        if reference_window_matched:
            variant_count = sum(
                1
                for base, reference_base in zip(record.window.sequence, reference_sequence)
                if base != "N" and base != reference_base
            )
        yield {
            "sample_id": record.window.sample_id,
            "locus_id": record.window.locus_id,
            "split": record.window.split,
            "source": record.window.source,
            "sequence": record.window.sequence,
            "reference_sequence": reference_sequence,
            "variant_count": variant_count,
            "callable_bases": len(record.window.sequence)
            - filtered_bases
            - no_call_bases
            - other_masked_bases,
            "filtered_bases": filtered_bases,
            "no_call_bases": no_call_bases,
            "other_masked_bases": other_masked_bases,
            "masked_base_counts": dict(record.window.masked_base_counts),
            "token_count": record.token_count,
            "reference_window_matched": reference_window_matched,
        }


def _load_fasta_sequences(path: str | Path) -> dict[str, str]:
    fasta_path = Path(path)
    sequences: dict[str, list[str]] = {}
    current_name: str | None = None
    with _open_maybe_gzip(fasta_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_name = line[1:].split()[0]
                sequences.setdefault(current_name, [])
                continue
            if current_name is None:
                raise ValueError(f"FASTA {fasta_path} contains sequence data before the first header")
            sequences[current_name].append(line)
    if not sequences:
        raise ValueError(f"FASTA {fasta_path} did not contain any sequences")
    return {name: "".join(parts) for name, parts in sequences.items()}


def _open_maybe_gzip(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _resolve_path(base_dir: Path, path: str | Path, *, prefer_cwd: bool = False) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = (Path.cwd() / candidate).resolve()
    base_candidate = (base_dir / candidate).resolve()
    if prefer_cwd:
        return cwd_candidate if cwd_candidate.exists() or not base_candidate.exists() else base_candidate
    return base_candidate if base_candidate.exists() else cwd_candidate


def _require_existing_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
