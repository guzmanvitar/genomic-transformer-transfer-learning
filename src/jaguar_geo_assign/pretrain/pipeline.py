"""Runtime wiring for the fixture-backed feline pretraining data pipeline.

Orchestrates the end-to-end feline genome pretraining pipeline: loading a
YAML config, resolving paths, parsing the sample manifest, generating
consensus FASTAs via bcftools, tokenizing both consensus and reference
sequences, exporting tokenized corpora, and producing diagnostics/EDA
reports.

The module exposes two dependency-injection hooks—``TokenizerLoader`` and
``ExportWriter``—so callers can swap tokeniser back-ends or serialisation
formats without touching pipeline logic.

Key fragility flags
-------------------
* ``_assert_tokenizer_matches_config`` uses an identity check (``is not``)
  on ``trust_remote_code`` to reject non-boolean values that would
  pass an equality test (e.g. ``1 == True``).
* ``load_feline_sample_manifest`` rejects duplicate ``sample_id`` values
  to prevent silent overwrites in downstream dict-keyed lookups.
* ``_resolve_path`` implements a *prefer_cwd* strategy where, for
  relative paths, the current working directory is tried first when the
  flag is set, falling back to the config-relative base directory only
  when the CWD candidate does not exist.
"""

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

TokenizerLoader = Callable[[TokenizerProvenance], tuple[object, TokenizerProvenance]]
"""Dependency-injection hook for tokenizer loading.

A callable that receives a ``TokenizerProvenance`` describing the expected
tokenizer identity and returns ``(tokenizer_object, actual_provenance)``.
The default implementation is ``load_dnabert2_tokenizer``.  Swapping this
callable lets tests or alternative pipelines provide mock or non-HuggingFace
tokenizers without modifying orchestration logic.
"""

ExportWriter = Callable[[tuple[TokenizedWindow, ...], Path, FelinePipelineConfig, TokenizerProvenance], Path]
"""Dependency-injection hook for corpus serialisation.

A callable that persists a tuple of ``TokenizedWindow`` objects to disk and
returns the resolved output ``Path``.  The default implementation is
``write_tokenized_corpus``, which delegates to ``write_tokenized_dataset``.
Replace this to change the on-disk format (e.g. TFRecord, HDF5) without
altering pipeline orchestration.
"""

DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT = 128


@dataclass(frozen=True)
class FelineSampleManifestEntry:
    """Single row from the feline sample manifest TSV.

    Each entry maps a unique ``sample_id`` to the biological individual and
    the filesystem path of its VCF file.  The ``vcf_path`` is always
    resolved to an absolute path at parse time via ``_resolve_path``.

    Attributes:
        sample_id: Unique identifier for the sample; duplicates are rejected
            during manifest loading.
        individual_id: Biological individual this sample belongs to.
        vcf_path: Absolute, resolved path to the sample's VCF file.
    """

    sample_id: str
    individual_id: str
    vcf_path: Path


@dataclass(frozen=True)
class FelinePretrainArtifacts:
    """Filesystem locations of all artifacts produced by a pretrain run.

    Attributes:
        consensus_dir: Directory containing per-sample consensus FASTA files.
        consensus_export: Path to the exported tokenized consensus corpus.
        baseline_export: Path to the exported tokenized reference corpus.
        diagnostics_path: Path to the JSON EDA/diagnostics payload.
        summary_path: Path to the JSON run summary.
    """

    consensus_dir: Path
    consensus_export: Path
    baseline_export: Path
    diagnostics_path: Path
    summary_path: Path


@dataclass(frozen=True)
class FelinePretrainRunResult:
    """Immutable summary returned by ``run_feline_pretrain_pipeline``.

    Attributes:
        config_name: Human-readable pipeline config name (from YAML).
        sample_count: Number of samples processed.
        consensus_window_count: Total tokenized windows from consensus
            sequences.
        baseline_window_count: Total tokenized windows from the reference
            genome.
        consensus_fastas: Paths to per-sample consensus FASTA files, in
            manifest order.
        artifacts: Filesystem artifact locations.
    """

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
    """Orchestrate the full feline pretraining data pipeline.

    Loads the pipeline YAML config, resolves all filesystem paths using the
    ``prefer_cwd`` strategy, parses the sample manifest, generates
    consensus FASTAs via *bcftools*, tokenizes both consensus and reference
    sequences, writes tokenized corpora, and emits diagnostics and a JSON
    run summary.

    Args:
        config_path: Path to the YAML pipeline configuration file.
        bcftools_executable: Name or path of the ``bcftools`` binary used
            to generate consensus FASTAs.
        tokenizer_loader: Optional override for the tokenizer loading
            callable (see ``TokenizerLoader``).  Defaults to
            ``load_dnabert2_tokenizer``.
        export_writer: Optional override for the corpus serialisation
            callable (see ``ExportWriter``).  Defaults to
            ``write_tokenized_corpus``.

    Returns:
        A ``FelinePretrainRunResult`` summarising sample counts, window
        counts, and artifact paths.

    Raises:
        RuntimeError: If required input files are missing, the loaded
            tokenizer does not match the config contract, or no windows
            survive preprocessing.
        ValueError: If the sample manifest is malformed.
    """
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

    preprocessing_config = _build_preprocessing_config(config)
    expected_provenance = _build_tokenizer_provenance(config)
    tokenizer, provenance = tokenizer_loader(expected_provenance)
    _assert_tokenizer_matches_config(config, provenance)

    tokenized_consensus = _tokenize_sequence_records(
        _iter_consensus_sequence_records(consensus_results, manifest_entries),
        preprocessing_config,
        tokenizer,
        provenance,
    )
    tokenized_baseline = _tokenize_sequence_records(
        _iter_reference_sequence_records(reference_fasta),
        preprocessing_config,
        tokenizer,
        provenance,
    )
    if not tokenized_consensus:
        raise RuntimeError("No consensus windows survived preprocessing; check sequence length and ambiguity filters")
    if not tokenized_baseline:
        raise RuntimeError("No baseline windows survived preprocessing; check the reference FASTA and window settings")

    consensus_export = export_writer(tokenized_consensus, processed_dir / "consensus_tokens", config, provenance)
    baseline_export = export_writer(tokenized_baseline, baseline_dir / "reference_tokens", config, provenance)

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
    """Format a ``FelinePretrainRunResult`` as a human-readable multi-line string.

    Args:
        result: The run result to format.

    Returns:
        A newline-joined summary suitable for logging or CLI output.
    """
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
    """Parse a TSV sample manifest into validated manifest entries.

    Reads a tab-delimited file whose header must contain all columns
    defined in ``REQUIRED_SAMPLE_MANIFEST_FIELDS``.  Each row is
    validated for completeness, and **duplicate ``sample_id`` values are
    rejected** to prevent silent overwrites when downstream code keys
    dictionaries by sample ID.

    VCF paths are resolved relative to the manifest's parent directory
    via ``_resolve_path``.

    Args:
        path: Filesystem path to the TSV manifest file.

    Returns:
        An ordered tuple of ``FelineSampleManifestEntry`` instances.

    Raises:
        ValueError: If the header is missing required columns, any row is
            incomplete, duplicate ``sample_id`` values are found, or the
            manifest is empty.
        RuntimeError: If the manifest file does not exist.
    """
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
    provenance: TokenizerProvenance,
) -> Path:
    """Default ``ExportWriter`` implementation: persist tokenized windows.

    Delegates to ``write_tokenized_dataset`` after building an
    ``ExportContract`` from the pipeline config.

    Args:
        tokenized_windows: Windows to serialise.
        output_path: Target directory or file path for the export.
        config: Pipeline config used to derive the export contract.
        provenance: Tokenizer provenance metadata embedded in the export.

    Returns:
        The resolved ``Path`` where the corpus was written.
    """
    destination = Path(output_path)
    write_tokenized_dataset(
        tokenized_windows,
        destination,
        contract=_build_export_contract(config),
        provenance=provenance,
    )
    return destination


def _require_runtime_boolean(value: object, *, field_name: str) -> bool:
    """Assert that *value* is a genuine ``bool``, not a truthy surrogate.

    Uses ``type(value) is not bool`` (identity check) so that ``int``
    values like ``1`` or ``0`` are rejected even though they pass
    ``== True`` / ``== False``.  This prevents subtle security
    misconfigurations when deserialised config values are integers.

    Args:
        value: The value to check.
        field_name: Human-readable name included in the error message.

    Returns:
        The validated boolean.

    Raises:
        RuntimeError: If *value* is not exactly ``True`` or ``False``.
    """
    if type(value) is not bool:
        raise RuntimeError(f"{field_name} must be an actual boolean, got {value!r} ({type(value).__name__})")
    return value


def _build_preprocessing_config(config: FelinePipelineConfig) -> PreprocessingConfig:
    """Derive a ``PreprocessingConfig`` from the pipeline config.

    Maps windowing and split settings to the preprocessor's expected
    parameter structure, computing ``window_stride`` from the context
    window minus the overlap.

    Args:
        config: The loaded pipeline configuration.

    Returns:
        A ``PreprocessingConfig`` ready for sequence preparation.
    """
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


def _build_tokenizer_provenance(config: FelinePipelineConfig) -> TokenizerProvenance:
    """Build a ``TokenizerProvenance`` from the pipeline config's tokenizer section.

    Args:
        config: The loaded pipeline configuration.

    Returns:
        A ``TokenizerProvenance`` capturing the expected tokenizer identity.
    """
    return TokenizerProvenance(
        identifier=config.tokenizer.identifier,
        revision=config.tokenizer.revision,
        max_position_embeddings=config.tokenizer.max_position_embeddings,
        allowed_alphabet=config.tokenizer.allowed_alphabet,
        unsupported_symbol_policy=config.tokenizer.unsupported_symbol_policy,
        trust_remote_code=config.tokenizer.trust_remote_code,
    )


def _assert_tokenizer_matches_config(
    config: FelinePipelineConfig, provenance: TokenizerProvenance
) -> None:
    """Validate that the loaded tokenizer's provenance matches the config.

    Checks identifier, revision, alphabet, max_position_embeddings, and
    unsupported_symbol_policy via equality.  The ``trust_remote_code``
    field is checked via **identity** (``is not``) after passing through
    ``_require_runtime_boolean``, which rejects non-bool truthy values
    (e.g. ``1``).  This is intentional: a deserialised integer ``1``
    equals ``True`` but is not ``True``, and accepting it could silently
    enable remote code execution.

    Args:
        config: Pipeline configuration containing the expected tokenizer
            contract.
        provenance: Provenance returned by the tokenizer loader.

    Raises:
        RuntimeError: If any provenance field does not match the config.
    """
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
    expected_trust_remote_code = _require_runtime_boolean(
        config.tokenizer.trust_remote_code,
        field_name="Config tokenizer trust_remote_code",
    )
    actual_trust_remote_code = _require_runtime_boolean(
        provenance.trust_remote_code,
        field_name="Tokenizer loader trust_remote_code",
    )
    if actual_trust_remote_code is not expected_trust_remote_code:
        raise RuntimeError(
            "Tokenizer loader trust_remote_code does not match the approved config"
        )


def _build_export_contract(config: FelinePipelineConfig) -> ExportContract:
    """Derive an ``ExportContract`` from the pipeline config's export section.

    Args:
        config: The loaded pipeline configuration.

    Returns:
        An ``ExportContract`` controlling serialisation behaviour.
    """
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
    """Eagerly materialise consensus ``SequenceRecord`` objects into a list.

    Convenience wrapper around ``_iter_consensus_sequence_records``.

    Args:
        consensus_results: Mapping of sample ID to ``ConsensusResult``.
        manifest_entries: Ordered manifest entries for individual-ID lookup.

    Returns:
        A list of ``SequenceRecord`` instances for all consensus contigs.
    """
    return list(_iter_consensus_sequence_records(consensus_results, manifest_entries))


def _iter_consensus_sequence_records(
    consensus_results: dict[str, ConsensusResult],
    manifest_entries: tuple[FelineSampleManifestEntry, ...],
) -> Iterable[SequenceRecord]:
    """Lazily yield ``SequenceRecord`` objects from consensus FASTAs.

    Iterates over consensus results in sorted sample-ID order, reads
    each output FASTA, and attaches per-contig mask spans from the
    consensus diagnostics.

    Args:
        consensus_results: Mapping of sample ID to ``ConsensusResult``.
        manifest_entries: Ordered manifest entries for individual-ID lookup.

    Yields:
        One ``SequenceRecord`` per contig per sample.
    """
    individual_by_sample = {entry.sample_id: entry.individual_id for entry in manifest_entries}
    for sample_id, result in sorted(consensus_results.items()):
        mask_spans_by_contig: dict[str, list[tuple[int, int, str]]] = {}
        for span in result.mask_spans:
            mask_spans_by_contig.setdefault(span.contig, []).append((span.start, span.end, span.category))
        for contig, sequence in _iter_fasta_sequences(result.output_fasta):
            yield SequenceRecord(
                sample_id=sample_id,
                individual_id=individual_by_sample[sample_id],
                contig=contig,
                sequence=sequence,
                source="consensus",
                sequence_start=0,
                mask_spans=tuple(sorted(mask_spans_by_contig.get(contig, []))),
            )


def _build_reference_sequence_records(
    reference_sequences: dict[str, str],
    manifest_entries: tuple[FelineSampleManifestEntry, ...],
) -> list[SequenceRecord]:
    """Build reference ``SequenceRecord`` objects from pre-loaded sequences.

    Returns an empty list when *manifest_entries* is empty (no samples
    to compare against).

    Args:
        reference_sequences: Mapping of contig name to nucleotide string.
        manifest_entries: Manifest entries; used only to gate the
            early-return for the empty-manifest case.

    Returns:
        A list of reference ``SequenceRecord`` instances.
    """
    if not manifest_entries:
        return []

    return [
        SequenceRecord(
            sample_id=f"reference-{contig}",
            individual_id="reference",
            contig=contig,
            sequence=sequence,
            source="reference",
            sequence_start=0,
        )
        for contig, sequence in reference_sequences.items()
    ]


def _iter_reference_sequence_records(reference_fasta: str | Path) -> Iterable[SequenceRecord]:
    """Lazily yield ``SequenceRecord`` objects from the reference FASTA.

    Each contig is emitted as a separate record with ``source="reference"``
    and a synthetic ``sample_id`` of ``"reference-{contig}"``.

    Args:
        reference_fasta: Path to the reference genome FASTA (plain or
            gzip-compressed).

    Yields:
        One ``SequenceRecord`` per contig in the reference FASTA.
    """
    for contig, sequence in _iter_fasta_sequences(reference_fasta):
        yield SequenceRecord(
            sample_id=f"reference-{contig}",
            individual_id="reference",
            contig=contig,
            sequence=sequence,
            source="reference",
            sequence_start=0,
        )


def _tokenize_sequence_records(
    records: Iterable[SequenceRecord],
    preprocessing_config: PreprocessingConfig,
    tokenizer: object,
    provenance: TokenizerProvenance,
) -> tuple[TokenizedWindow, ...]:
    """Prepare, window, and tokenize an iterable of sequence records.

    Processes each record individually through the prepare → window →
    tokenize pipeline.  Records or windows that fail quality filters
    (ambiguity, minimum length) are silently dropped.

    Args:
        records: Iterable of ``SequenceRecord`` to process.
        preprocessing_config: Controls windowing size, stride, and
            quality thresholds.
        tokenizer: The tokenizer object (opaque; passed to
            ``tokenize_windows``).
        provenance: Tokenizer provenance attached to each output window.

    Returns:
        A tuple of ``TokenizedWindow`` instances that survived all
        preprocessing filters.
    """
    tokenized_windows: list[TokenizedWindow] = []
    for record in records:
        prepared = prepare_sequences([record], preprocessing_config)
        if not prepared.retained:
            continue
        windows = window_sequences(list(prepared.retained), preprocessing_config)
        if not windows:
            continue
        tokenized_windows.extend(tokenize_windows(windows, tokenizer, provenance=provenance))
    return tuple(tokenized_windows)


def _build_diagnostics_payload(
    *,
    tokenized_consensus: tuple[TokenizedWindow, ...],
    tokenized_baseline: tuple[TokenizedWindow, ...],
    consensus_results: dict[str, ConsensusResult],
) -> dict[str, object]:
    """Assemble the JSON-serialisable EDA / diagnostics payload.

    Combines per-sample consensus generation diagnostics, a
    baseline-window alignment summary, and the full EDA payload produced
    by ``build_eda_payload``.

    Args:
        tokenized_consensus: Tokenized windows from consensus sequences.
        tokenized_baseline: Tokenized windows from the reference genome.
        consensus_results: Per-sample consensus generation results.

    Returns:
        A nested dict suitable for ``json.dumps``; contains keys
        ``consensus_generation``, ``baseline_window_alignment``, and
        all keys contributed by ``build_eda_payload``.
    """
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
    """Return a hashable ``(contig, window_start, window_end)`` key for window lookup.

    Args:
        record: A tokenized window record.

    Returns:
        A 3-tuple suitable for use as a dictionary key.
    """
    return record.window.contig, record.window.window_start, record.window.window_end


def _tokenized_windows_to_diagnostics_records(
    tokenized_windows: tuple[TokenizedWindow, ...],
    reference_lookup: dict[tuple[str, int, int], str],
    *,
    allow_unmatched_reference: bool = False,
) -> Iterable[dict[str, object]]:
    """Yield per-window diagnostics dicts for the EDA payload.

    Each dict contains sample metadata, sequence and reference text,
    variant counts, masking statistics, and token count.  When a
    matching reference window exists, variant count is computed as the
    number of non-N positions that differ from the reference.

    Args:
        tokenized_windows: Windows to report on.
        reference_lookup: Mapping from ``(contig, start, end)`` to
            reference sequence strings.
        allow_unmatched_reference: If ``True``, windows without a
            reference match use their own sequence as the reference
            (for consensus diagnostics).  If ``False``, a ``KeyError``
            is raised on mismatch.

    Yields:
        One dict per window with keys required by ``build_eda_payload``.

    Raises:
        KeyError: If *allow_unmatched_reference* is ``False`` and a
            window has no matching entry in *reference_lookup*.
    """
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
            "callable_bases": _count_callable_bases(record.window.sequence),
            "unique_masked_bases": record.window.unique_masked_bases,
            "filtered_bases": filtered_bases,
            "no_call_bases": no_call_bases,
            "other_masked_bases": other_masked_bases,
            "masked_base_counts": dict(record.window.masked_base_counts),
            "token_count": record.token_count,
            "reference_window_matched": reference_window_matched,
        }


def _count_callable_bases(sequence: str) -> int:
    """Count bases in *sequence* that are not ``'N'``.

    Args:
        sequence: A nucleotide string.

    Returns:
        The number of non-N characters.
    """
    return sum(base != "N" for base in sequence)


def _load_fasta_sequences(path: str | Path) -> dict[str, str]:
    """Eagerly load all sequences from a FASTA file into a dict.

    Args:
        path: Path to a FASTA file (plain or gzip-compressed).

    Returns:
        A mapping of contig name to nucleotide string.
    """
    return dict(_iter_fasta_sequences(path))


def _iter_fasta_sequences(path: str | Path) -> Iterable[tuple[str, str]]:
    """Lazily parse a FASTA file, yielding ``(contig_name, sequence)`` pairs.

    Supports both plain-text and gzip-compressed (``.gz``) FASTA files.
    Contig names are extracted from the first whitespace-delimited token
    after the ``>`` header marker.  Empty lines are skipped.

    Args:
        path: Path to the FASTA file.

    Yields:
        Tuples of ``(contig_name, full_sequence_string)``.

    Raises:
        ValueError: If sequence data appears before the first header, or
            the file contains no sequences.
    """
    fasta_path = Path(path)
    current_name: str | None = None
    current_parts: list[str] = []
    with _open_maybe_gzip(fasta_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    yield current_name, "".join(current_parts)
                current_name = line[1:].split()[0]
                current_parts = []
                continue
            if current_name is None:
                raise ValueError(f"FASTA {fasta_path} contains sequence data before the first header")
            current_parts.append(line)
    if current_name is None:
        raise ValueError(f"FASTA {fasta_path} did not contain any sequences")
    yield current_name, "".join(current_parts)


def _open_maybe_gzip(path: Path):
    """Open a file for text reading, auto-detecting gzip by ``.gz`` suffix.

    Args:
        path: Filesystem path to open.

    Returns:
        A text-mode file handle (UTF-8).
    """
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _resolve_path(base_dir: Path, path: str | Path, *, prefer_cwd: bool = False) -> Path:
    """Resolve a potentially relative path against a base directory.

    Absolute paths are returned unchanged.  For relative paths, two
    candidates are formed: one relative to the current working directory
    and one relative to *base_dir*.

    **Fragility flag — ``prefer_cwd`` strategy:**
    When ``prefer_cwd=True``, the CWD candidate is returned if it exists
    **or** if neither candidate exists (defaulting to CWD).  When
    ``prefer_cwd=False`` (default), the *base_dir* candidate wins if it
    exists, otherwise the CWD candidate is returned.  This asymmetry
    means that pipeline behaviour depends on the caller's working
    directory, which can cause hard-to-reproduce path resolution in CI
    vs. local runs.

    Args:
        base_dir: Directory to resolve relative paths against (typically
            the config file's parent).
        path: The path to resolve (absolute or relative).
        prefer_cwd: If ``True``, prefer the CWD-relative candidate.

    Returns:
        A resolved absolute ``Path``.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = (Path.cwd() / candidate).resolve()
    base_candidate = (base_dir / candidate).resolve()
    if prefer_cwd:
        return cwd_candidate if cwd_candidate.exists() or not base_candidate.exists() else base_candidate
    return base_candidate if base_candidate.exists() else cwd_candidate


def _require_existing_file(path: Path, label: str) -> None:
    """Raise ``RuntimeError`` if *path* does not point to an existing file.

    Args:
        path: Filesystem path to check.
        label: Human-readable label included in the error message.

    Raises:
        RuntimeError: If the path does not exist or is not a regular file.
    """
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
