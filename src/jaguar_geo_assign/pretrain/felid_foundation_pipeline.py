"""Runtime wiring for the multi-species felid foundation pretraining pipeline.

Orchestrates the end-to-end FASTA-only pretraining corpus across the six
approved felid reference assemblies. Every species is processed
**independently and end-to-end** (FASTA parse → prepare → window →
tokenize → Parquet ``write_batch``) before the next species begins. The
:class:`TokenizedCorpusWriter` from
:mod:`jaguar_geo_assign.data.preprocessor` is opened once at the start of
the run and closed at the end, so peak heap usage is bounded by the
largest single assembly rather than by the full six-species corpus. For
each species the pipeline holds at most one species' windows in memory,
calls ``write_batch``, and releases; tests explicitly verify that the
tokenizer fake never observes more than one species' records concurrently.
"""

from __future__ import annotations

import json
import logging
import resource
import sys
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..config import FelidSpeciesEntry, load_felid_foundation_pipeline_config
from ..data.preprocessor import (
    SequenceRecord,
    TokenizedCorpusWriter,
    TokenizerProvenance,
    load_dnabert2_tokenizer,
    prepare_sequences,
    tokenize_windows,
    window_sequences,
)
from ._shared import (
    _assert_tokenizer_matches_config,
    _build_export_contract,
    _build_preprocessing_config,
    _build_tokenizer_provenance,
    _iter_fasta_sequences,
    _resolve_path,
    normalize_ru_maxrss_to_bytes,
)

_LOGGER = logging.getLogger(__name__)


class MissingFelidReferenceError(RuntimeError):
    """Raised when an expected per-species FASTA is not on disk.

    The error message intentionally includes the ``acquire`` CLI
    invocation so operators can recover without having to consult the
    docs. This is the only expected "operator-fixable" error path out of
    :func:`run_felid_foundation_pretrain`.
    """


TokenizerLoader = Callable[[TokenizerProvenance], tuple[object, TokenizerProvenance]]
"""Dependency-injection hook for tokenizer loading; mirrors the feline hook."""


class _WriterContextManager(AbstractContextManager, Protocol):
    """Structural type for the writer returned by an :data:`ExportWriter`.

    The felid pipeline only uses the streaming ``write_batch`` entry point
    and the context-manager protocol; anything satisfying this structural
    type can be injected in tests. The default implementation is
    :class:`TokenizedCorpusWriter`.
    """

    def write_batch(self, tokenized_windows: Any) -> None: ...  # pragma: no cover


ExportWriter = Callable[..., _WriterContextManager]
"""Factory callable producing a streaming tokenized-corpus writer.

Signature: ``(output_dir, *, contract, provenance) -> writer``. The
returned writer must implement ``write_batch(tokens)`` and the context
manager protocol. Defaults to :class:`TokenizedCorpusWriter`.
"""


@dataclass(frozen=True)
class FelidFoundationPretrainArtifacts:
    """Filesystem locations of all artifacts produced by a felid-foundation run.

    Attributes:
        corpus_dir: Root directory of the streaming Parquet corpus.
        summary_path: Path to the run-summary JSON.
    """

    corpus_dir: Path
    summary_path: Path


@dataclass(frozen=True)
class FelidSpeciesPretrainStats:
    """Per-species summary mirroring the run-summary JSON ``per_species`` entry.

    Every field here maps 1:1 to a key in the pinned run-summary
    JSON schema for a single species. Storing the structured values on a
    typed dataclass lets downstream callers (CLI, tests) introspect
    results without re-parsing the JSON, while the ``_dump`` serialiser
    in :func:`run_felid_foundation_pretrain` is the single source of
    truth for the on-disk key set.

    Attributes:
        species_slug: Canonical slug (e.g. ``"panthera_leo"``). Used as
            ``individual_id`` on every emitted window.
        identifier: RefSeq identifier for this species' assembly.
        assembly_name: RefSeq assembly name (e.g. ``"Felis_catus_9.0"``).
        contig_count: Number of FASTA contigs parsed for this species.
        retained_sequence_count: Number of ``PreparedSequence`` that
            survived the length/ambiguity filters.
        filtered_short_count: Sequences rejected as ``short_sequence``.
        filtered_high_ambiguity_count: Sequences rejected as
            ``high_ambiguity``.
        window_counts_by_split: Per-split window counts. Always has
            exactly ``{"train", "validation"}`` keys (zero-filled when a
            split was not represented).
        peak_window_count_in_memory: Maximum number of tokenized windows
            held in memory concurrently for this species (reflects the
            streaming-writer model: the species batch is held once before
            :meth:`TokenizedCorpusWriter.write_batch`).
        peak_rss_bytes: Normalised peak RSS observed at species_end.
        bytes_tokenized: Sum of raw normalised-sequence byte counts
            passed into :func:`tokenize_windows` for this species.
        export_path: Root directory of the shared streaming corpus.
    """

    species_slug: str
    identifier: str
    assembly_name: str
    contig_count: int
    retained_sequence_count: int
    filtered_short_count: int
    filtered_high_ambiguity_count: int
    window_counts_by_split: dict[str, int]
    peak_window_count_in_memory: int
    peak_rss_bytes: int
    bytes_tokenized: int
    export_path: str


@dataclass(frozen=True)
class FelidFoundationPretrainRunResult:
    """Immutable summary returned by :func:`run_felid_foundation_pretrain`.

    This dataclass mirrors the top-level run-summary JSON schema
    so the CLI and tests can introspect corpus-wide statistics without
    re-parsing the JSON. The schema keys are pinned; any change requires
    a corresponding update in the schema-equality test.

    Attributes:
        config_name: Human-readable pipeline config name.
        tokenizer_identifier: HuggingFace identifier of the tokenizer
            that validated successfully against the config contract.
        tokenizer_revision: Immutable revision SHA of the loaded tokenizer.
        species: Tuple of ``(species_slug, identifier, assembly_name)`` in
            registry order, matching the ``species`` JSON list.
        per_species_stats: Per-species stats keyed by species slug (as
            an ordered mapping; emitted as the JSON ``per_species`` map).
        totals: Corpus-wide split totals with exactly ``{"train",
            "validation"}`` keys.
        artifacts: Filesystem artifact locations.
    """

    config_name: str
    tokenizer_identifier: str
    tokenizer_revision: str
    species: tuple[tuple[str, str, str], ...]
    per_species_stats: tuple[FelidSpeciesPretrainStats, ...]
    totals: dict[str, int]
    artifacts: FelidFoundationPretrainArtifacts


def _resolve_fasta_path(reference_dir: Path, entry: FelidSpeciesEntry) -> Path:
    """Derive the canonical ``<identifier>.fna.gz`` path for a species.

    The per-species filename is deterministic from the registry
    so :func:`acquire_felid_foundation_assemblies` and
    :func:`run_felid_foundation_pretrain` resolve the same path without
    sharing state. Keeping the derivation in a single helper means any
    future change to the filename convention is a one-line edit.
    """
    return reference_dir / f"{entry.identifier}.fna.gz"


def _iter_species_sequence_records(
    fasta_path: Path,
    species_slug: str,
) -> Iterable[SequenceRecord]:
    """Yield one ``SequenceRecord`` per contig in a species FASTA.

    The felid foundation pipeline encodes species identity as
    ``individual_id=<species_slug>`` so downstream consumers can group
    emitted windows by species without re-deriving the slug. The
    ``sample_id`` is ``f"{species_slug}-{contig}"`` so every
    ``SequenceRecord`` from a single species still has a unique sample
    identifier, preserving the per-record uniqueness invariant that the
    feline consensus path relies on while keeping species provenance
    inspectable. ``source="reference"`` is mandatory for this pipeline
    (the foundation corpus never consumes consensus sequences) and
    matches the approved producer set enforced by
    :func:`prepare_sequences`.

    Args:
        fasta_path: Path to the per-species ``<ACC>_<ASM>.fna.gz`` file.
        species_slug: Canonical species slug used for individual ID and
            as the prefix of the per-contig sample ID.

    Yields:
        One ``SequenceRecord`` per contig, with ``source="reference"``.
    """
    for contig, sequence in _iter_fasta_sequences(fasta_path):
        yield SequenceRecord(
            sample_id=f"{species_slug}-{contig}",
            individual_id=species_slug,
            contig=contig,
            sequence=sequence,
            source="reference",
            sequence_start=0,
        )


def _read_peak_rss_bytes() -> int:
    """Return the current process's peak RSS in normalised bytes.

    Isolating the ``resource.getrusage`` syscall behind a helper
    makes it trivial to stub out in unit tests without monkeypatching the
    stdlib. The unit-conversion responsibility lives in
    :func:`normalize_ru_maxrss_to_bytes`, which is platform-aware.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return normalize_ru_maxrss_to_bytes(raw, sys.platform)


def _run_single_species(
    *,
    entry: FelidSpeciesEntry,
    fasta_path: Path,
    preprocessing_config: Any,
    tokenizer: object,
    provenance: TokenizerProvenance,
    writer: TokenizedCorpusWriter,
    contig_owner: dict[str, str],
    corpus_dir: Path,
) -> FelidSpeciesPretrainStats:
    """Run prepare → window → tokenize → write for one species.

    This helper owns the per-species streaming-writer contract.
    It reads the species FASTA lazily, guards cross-species contig
    collisions on first sighting, decomposes the prepare/window/tokenize
    cascade so the intermediate retained/filtered counts required by the
    pinned run-summary schema are observable, calls
    :meth:`TokenizedCorpusWriter.write_batch` exactly once with the
    species batch, and then lets the batch fall out of scope so the
    caller's next-species iteration starts from the writer's bounded
    memory floor rather than the previous species's batch. Structured
    log events (``species_start``, ``fasta_parsed``, ``sequences_prepared``,
    ``windows_generated``, ``windows_tokenized``, ``species_end``) fire
    at ``INFO`` so a production operator can reconstruct the run from
    logs alone; per-filter-reason counts fire at ``DEBUG``.

    Args:
        entry: Validated species entry (identifier, assembly, slug).
        fasta_path: Per-species FASTA path resolved from the config.
        preprocessing_config: Shared preprocessing thresholds.
        tokenizer: Loaded tokenizer object (opaque).
        provenance: Validated tokenizer provenance.
        writer: Shared streaming :class:`TokenizedCorpusWriter`.
        contig_owner: Mutable map ``{contig -> owning species_slug}``
            shared across species for collision detection.
        corpus_dir: Root of the streaming corpus; reported in the
            species's ``export_path`` field.

    Returns:
        A :class:`FelidSpeciesPretrainStats` mirroring the
        ``per_species`` JSON entry for this species.

    Raises:
        RuntimeError: If a contig declared by this species was already
            declared by a different species earlier in the run.
    """
    _LOGGER.info(
        "species_start species=%s identifier=%s fasta=%s",
        entry.species_slug,
        entry.identifier,
        fasta_path,
    )

    species_records: list[SequenceRecord] = []
    contig_count = 0
    for record in _iter_species_sequence_records(fasta_path, entry.species_slug):
        prior = contig_owner.get(record.contig)
        if prior is not None and prior != entry.species_slug:
            raise RuntimeError(
                "Cross-species contig-name collision detected: "
                f"contig {record.contig!r} is declared by both "
                f"{prior!r} and {entry.species_slug!r}; aborting "
                "before windowing to avoid silent locus_id aliasing"
            )
        contig_owner[record.contig] = entry.species_slug
        species_records.append(record)
        contig_count += 1
    _LOGGER.info(
        "fasta_parsed species=%s identifier=%s contigs=%d",
        entry.species_slug,
        entry.identifier,
        contig_count,
    )

    report = prepare_sequences(species_records, preprocessing_config)
    filter_reason_counts: dict[str, int] = {}
    for filtered in report.filtered:
        filter_reason_counts[filtered.reason] = filter_reason_counts.get(filtered.reason, 0) + 1
    for reason, count in sorted(filter_reason_counts.items()):
        _LOGGER.debug(
            "filter_reason species=%s reason=%s count=%d",
            entry.species_slug,
            reason,
            count,
        )
    _LOGGER.info(
        "sequences_prepared species=%s identifier=%s retained=%d filtered=%d",
        entry.species_slug,
        entry.identifier,
        len(report.retained),
        len(report.filtered),
    )

    # TRADE-OFF: counts characters not bytes; all inputs are ASCII DNA so char == byte in practice.
    bytes_tokenized = sum(len(prepared.sequence) for prepared in report.retained)
    windows = (
        window_sequences(list(report.retained), preprocessing_config) if report.retained else ()
    )
    _LOGGER.info(
        "windows_generated species=%s identifier=%s windows=%d",
        entry.species_slug,
        entry.identifier,
        len(windows),
    )

    species_windows = tokenize_windows(windows, tokenizer, provenance=provenance) if windows else ()
    species_windows = tuple(species_windows)
    _LOGGER.info(
        "windows_tokenized species=%s identifier=%s tokens=%d",
        entry.species_slug,
        entry.identifier,
        len(species_windows),
    )

    writer.write_batch(species_windows)

    window_counts_by_split: dict[str, int] = {"train": 0, "validation": 0}
    for window in species_windows:
        split = window.window.split
        window_counts_by_split[split] = window_counts_by_split.get(split, 0) + 1

    peak_rss_bytes = _read_peak_rss_bytes()
    _LOGGER.info(
        "species_end species=%s identifier=%s windows=%d peak_rss_bytes=%d",
        entry.species_slug,
        entry.identifier,
        len(species_windows),
        peak_rss_bytes,
    )

    return FelidSpeciesPretrainStats(
        species_slug=entry.species_slug,
        identifier=entry.identifier,
        assembly_name=entry.assembly_name,
        contig_count=contig_count,
        retained_sequence_count=len(report.retained),
        filtered_short_count=filter_reason_counts.get("short_sequence", 0),
        filtered_high_ambiguity_count=filter_reason_counts.get("high_ambiguity", 0),
        window_counts_by_split=window_counts_by_split,
        peak_window_count_in_memory=len(species_windows),
        peak_rss_bytes=peak_rss_bytes,
        bytes_tokenized=bytes_tokenized,
        export_path=str(corpus_dir),
    )


def run_felid_foundation_pretrain(
    config_path: str | Path,
    *,
    tokenizer_loader: TokenizerLoader | None = None,
    export_writer: ExportWriter | None = None,
) -> FelidFoundationPretrainRunResult:
    """Orchestrate the multi-species felid foundation pretraining run.

    Species are processed sequentially. For each species the pipeline:

    1. Resolves the per-species FASTA path via :func:`_resolve_fasta_path`.
    2. Iterates contigs lazily, yielding one ``SequenceRecord`` per contig
       with ``source="reference"``, ``individual_id=<species_slug>``,
       and ``sample_id=f"{species_slug}-{contig}"``.
    3. Runs prepare → window → tokenize per contig, accumulating the
       species batch and tracking intermediate filter counts for the
       run-summary schema.
    4. Calls :meth:`TokenizedCorpusWriter.write_batch` with the species
       batch and then releases the batch before the next species begins,
       so peak memory is bounded by the single largest species rather
       than the full corpus.

    Cross-species contig-name collisions are detected inline while
    records are being emitted for the second-and-later species: any
    contig already owned by a different species aborts the run before
    windowing, preserving the single-namespace invariant of the shared
    ``locus_id = f"{contig}:{block_start}-{block_end}"`` Parquet key.

    Args:
        config_path: Path to a TOML felid-foundation pipeline config.
        tokenizer_loader: Optional override for the tokenizer loader
            (see :data:`TokenizerLoader`). Defaults to
            :func:`load_dnabert2_tokenizer`.
        export_writer: Optional factory producing the streaming writer
            used to append per-species tokenized batches (see
            :data:`ExportWriter`). Defaults to
            :class:`TokenizedCorpusWriter`.

    Returns:
        A :class:`FelidFoundationPretrainRunResult` mirroring the pinned
        run-summary JSON schema.

    Raises:
        MissingFelidReferenceError: If any approved species FASTA is not
            present on disk under the configured ``reference_dir``.
        RuntimeError: On contig-name collisions between species, on a
            tokenizer contract mismatch, or if no species emits any
            windows (an empty corpus is always a configuration error).
    """
    tokenizer_loader = tokenizer_loader or load_dnabert2_tokenizer
    export_writer = export_writer or TokenizedCorpusWriter

    config_file = Path(config_path).resolve()
    config = load_felid_foundation_pipeline_config(config_file)
    config_root = config_file.parent

    reference_dir = _resolve_path(config_root, config.paths.reference_dir, prefer_cwd=True)
    processed_dir = _resolve_path(config_root, config.paths.processed_dir, prefer_cwd=True)
    artifact_dir = _resolve_path(config_root, config.paths.artifact_dir, prefer_cwd=True)

    species_paths: list[tuple[FelidSpeciesEntry, Path]] = []
    for entry in config.species:
        fasta_path = _resolve_fasta_path(reference_dir, entry)
        if not fasta_path.exists() or not fasta_path.is_file():
            raise MissingFelidReferenceError(
                f"Missing reference FASTA for {entry.species} "
                f"({entry.identifier}) at {fasta_path}. Run: "
                "uv run python -m jaguar_geo_assign.cli "
                f"acquire-felid-foundation-assemblies {config_path}"
            )
        species_paths.append((entry, fasta_path))

    preprocessing_config = _build_preprocessing_config(config)
    expected_provenance = _build_tokenizer_provenance(config)
    tokenizer, provenance = tokenizer_loader(expected_provenance)
    _assert_tokenizer_matches_config(config, provenance)

    export_contract = _build_export_contract(config)
    corpus_dir = processed_dir / "felid_foundation_tokens"

    contig_owner: dict[str, str] = {}
    per_species_stats: list[FelidSpeciesPretrainStats] = []
    totals: dict[str, int] = {"train": 0, "validation": 0}

    with export_writer(
        corpus_dir,
        contract=export_contract,
        provenance=provenance,
    ) as writer:
        for entry, fasta_path in species_paths:
            stats = _run_single_species(
                entry=entry,
                fasta_path=fasta_path,
                preprocessing_config=preprocessing_config,
                tokenizer=tokenizer,
                provenance=provenance,
                writer=writer,
                contig_owner=contig_owner,
                corpus_dir=corpus_dir,
            )
            per_species_stats.append(stats)
            for split, count in stats.window_counts_by_split.items():
                totals[split] = totals.get(split, 0) + count

        total_window_count = sum(totals.values())
        if total_window_count == 0:
            raise RuntimeError(
                "Felid foundation pretrain produced zero tokenized windows across all "
                "species; check windowing/ambiguity filters and per-species FASTA contents"
            )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "felid_foundation_pretrain_run_summary.json"
    summary_payload = {
        "config_name": config.name,
        "tokenizer_identifier": provenance.identifier,
        "tokenizer_revision": provenance.revision,
        "species": [
            {
                "species_slug": stats.species_slug,
                "identifier": stats.identifier,
                "assembly_name": stats.assembly_name,
            }
            for stats in per_species_stats
        ],
        "per_species": {
            stats.species_slug: {
                "identifier": stats.identifier,
                "assembly_name": stats.assembly_name,
                "contig_count": stats.contig_count,
                "retained_sequence_count": stats.retained_sequence_count,
                "filtered_short_count": stats.filtered_short_count,
                "filtered_high_ambiguity_count": stats.filtered_high_ambiguity_count,
                "window_counts_by_split": dict(stats.window_counts_by_split),
                "peak_window_count_in_memory": stats.peak_window_count_in_memory,
                "peak_rss_bytes": stats.peak_rss_bytes,
                "bytes_tokenized": stats.bytes_tokenized,
                "export_path": stats.export_path,
            }
            for stats in per_species_stats
        },
        "totals": dict(totals),
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    species_tuple = tuple(
        (stats.species_slug, stats.identifier, stats.assembly_name) for stats in per_species_stats
    )
    return FelidFoundationPretrainRunResult(
        config_name=config.name,
        tokenizer_identifier=provenance.identifier,
        tokenizer_revision=provenance.revision,
        species=species_tuple,
        per_species_stats=tuple(per_species_stats),
        totals=dict(totals),
        artifacts=FelidFoundationPretrainArtifacts(
            corpus_dir=corpus_dir,
            summary_path=summary_path,
        ),
    )


def format_felid_foundation_pretrain_result(
    result: FelidFoundationPretrainRunResult,
) -> str:
    """Format a :class:`FelidFoundationPretrainRunResult` as a multi-line string.

    Args:
        result: The run result to format.

    Returns:
        A newline-joined summary suitable for logging or CLI output.
    """
    total_windows = sum(result.totals.values())
    peak_rss_bytes = max(
        (stats.peak_rss_bytes for stats in result.per_species_stats),
        default=0,
    )
    lines = [
        f"Felid foundation pretrain finished for '{result.config_name}'.",
        f"Tokenizer: {result.tokenizer_identifier}@{result.tokenizer_revision}",
        f"Species: {len(result.per_species_stats)}",
        f"Total windows: {total_windows}",
        f"Totals by split: {dict(result.totals)}",
        f"Peak RSS (bytes): {peak_rss_bytes}",
        f"Corpus directory: {result.artifacts.corpus_dir}",
        f"Summary: {result.artifacts.summary_path}",
    ]
    for stats in result.per_species_stats:
        lines.append(
            f"  - {stats.species_slug} ({stats.identifier} / {stats.assembly_name}): "
            f"contigs={stats.contig_count} retained={stats.retained_sequence_count} "
            f"windows={stats.peak_window_count_in_memory} "
            f"splits={dict(stats.window_counts_by_split)} "
            f"bytes={stats.bytes_tokenized}"
        )
    return "\n".join(lines)
