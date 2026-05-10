"""Runtime wiring for the multi-species felid foundation pretraining pipeline.

Orchestrates the end-to-end FASTA-only pretraining corpus across the six
approved felid reference assemblies. Every species is processed
**independently and end-to-end** (FASTA parse → prepare → window →
tokenize → Parquet ``write_batch``) before the next species begins. The
:class:`TokenizedCorpusWriter` from
:mod:`jaguar_geo_assign.data.preprocessor` is opened once at the start of
the run and closed at the end, so peak heap usage is bounded by the
largest single assembly rather than by the full six-species corpus. For
each species the pipeline streams tokenized windows to the writer in
fixed-size chunks, releasing each chunk immediately after ``write_batch``;
tests explicitly verify that the tokenizer fake never observes more than
one species' records concurrently.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import pickle
import queue as queue_module
import re
import resource
import sys
import tempfile
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..config import FelidSpeciesEntry, load_felid_foundation_pipeline_config
from ..data.preprocessor import (
    SequenceRecord,
    TokenizedCorpusWriter,
    TokenizedWindow,
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
    _open_maybe_gzip,
    _resolve_path,
    normalize_ru_maxrss_to_bytes,
)

_LOGGER = logging.getLogger(__name__)

_DEFAULT_TOKENIZED_WINDOW_CHUNK_SIZE = 10_000
_CHECKPOINT_SCHEMA_VERSION = "1"
_PART_FILE_NAME_PATTERN = re.compile(r"^part-(\d+)-\d+\.parquet$")
_DEFAULT_QUEUE_MAXSIZE_FACTOR = 2
_DEFAULT_WORKER_SIGTERM_TIMEOUT_SECONDS = 30.0


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
            held in memory concurrently for this species while the
            streaming writer is flushing one tokenized chunk at a time.
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


@dataclass(frozen=True)
class CheckpointState:
    """Resume state persisted to ``checkpoint.json`` for successful species only.

    The checkpoint exists to make restarts idempotent at the species level.
    Each completed species carries its final stats so a resumed run can rebuild
    the final summary without recomputing already-finished FASTAs.
    """

    schema_version: str
    config_path: str
    config_name: str
    updated_at: str
    completed_species: tuple[str, ...]
    per_species_stats: tuple[FelidSpeciesPretrainStats, ...]

    @classmethod
    def empty(cls, *, config_file: Path, config_name: str) -> CheckpointState:
        """Return an empty checkpoint for a fresh run of the current config."""
        return cls(
            schema_version=_CHECKPOINT_SCHEMA_VERSION,
            config_path=str(config_file.resolve()),
            config_name=config_name,
            updated_at="",
            completed_species=(),
            per_species_stats=(),
        )

    def validate(self, *, config_file: Path, config_name: str) -> None:
        """Raise if the checkpoint is incompatible with the current config."""
        if self.schema_version != _CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported felid-foundation checkpoint schema_version "
                f"{self.schema_version!r}; delete checkpoint.json to start fresh"
            )
        expected_path = str(config_file.resolve())
        if self.config_path != expected_path:
            raise RuntimeError(
                f"Checkpoint config_path {self.config_path!r} does not match "
                f"current config {expected_path!r}; delete checkpoint.json to start fresh"
            )
        if self.config_name != config_name:
            raise RuntimeError(
                f"Checkpoint config_name {self.config_name!r} does not match "
                f"current config name {config_name!r}; delete checkpoint.json to start fresh"
            )


@dataclass(frozen=True)
class _ChunkMessage:
    """One tokenized chunk emitted by a producer worker."""

    species_slug: str
    chunk: tuple[TokenizedWindow, ...]


@dataclass(frozen=True)
class _DoneMessage:
    """Final successful completion signal for one species worker."""

    species_slug: str
    stats: FelidSpeciesPretrainStats


@dataclass(frozen=True)
class _ErrorMessage:
    """Structured worker failure forwarded to the single consumer."""

    species_slug: str
    error_type: str
    error_message: str
    traceback_str: str


class _WorkerShutdownRequestedError(RuntimeError):
    """Internal signal used to stop workers promptly after consumer failure.

    Workers can spend most of their time blocked in ``queue.put`` when the
    consumer stops draining. Raising a dedicated internal exception lets the
    worker unwind without emitting a misleading ``_ErrorMessage`` for a
    shutdown that was initiated by the parent process.
    """


class _ZeroTokenizedWindowsError(RuntimeError):
    """Internal marker for the empty-corpus guard that must trip inside writer cleanup."""


def _put_queue_message(
    *,
    ipc_queue: Any,
    message: object,
    shutdown_event: Any,
    put_timeout_seconds: float,
) -> bool:
    """Retry ``queue.put`` until it succeeds or the parent requests shutdown."""
    while True:
        try:
            ipc_queue.put(message, timeout=put_timeout_seconds)
            return True
        except queue_module.Full:
            if shutdown_event.is_set():
                return False


class _QueueBatchWriter:
    """Queue-backed ``write_batch`` adapter used inside worker processes.

    The real ``TokenizedCorpusWriter`` remains owned by the main process; workers
    only forward already-tokenized batches over IPC so all filesystem writes stay
    single-consumer.
    """

    def __init__(
        self,
        *,
        species_slug: str,
        queue: Any,
        shutdown_event: Any,
        put_timeout_seconds: float,
    ) -> None:
        self._species_slug = species_slug
        self._queue = queue
        self._shutdown_event = shutdown_event
        self._put_timeout_seconds = put_timeout_seconds

    def write_batch(self, tokenized_windows: Any) -> None:
        """Emit a non-empty tokenized batch to the consumer queue."""
        chunk = tuple(tokenized_windows)
        if not chunk:
            return
        if not _put_queue_message(
            ipc_queue=self._queue,
            message=_ChunkMessage(species_slug=self._species_slug, chunk=chunk),
            shutdown_event=self._shutdown_event,
            put_timeout_seconds=self._put_timeout_seconds,
        ):
            raise _WorkerShutdownRequestedError(
                f"Shutdown requested while writing queued chunk for {self._species_slug!r}"
            )


def _resolve_fasta_path(reference_dir: Path, entry: FelidSpeciesEntry) -> Path:
    """Derive the canonical ``<identifier>.fna.gz`` path for a species.

    The per-species filename is deterministic from the registry
    so :func:`acquire_felid_foundation_assemblies` and
    :func:`run_felid_foundation_pretrain` resolve the same path without
    sharing state. Keeping the derivation in a single helper means any
    future change to the filename convention is a one-line edit.
    """
    return reference_dir / f"{entry.identifier}.fna.gz"


def _iter_fasta_contig_names(fasta_path: Path) -> Iterator[str]:
    """Yield FASTA contig names without materialising any nucleotide sequence data.

    Resume mode only needs the contig namespace of checkpointed species so it can
    keep enforcing the cross-species collision guard after skipped species are
    elided from the main processing loop.
    """
    with _open_maybe_gzip(fasta_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">"):
                yield line[1:].split()[0]


def _preflight_contig_check(species_paths: list[tuple[FelidSpeciesEntry, Path]]) -> None:
    """Reject cross-species contig collisions before any worker is launched."""
    contig_owner: dict[str, str] = {}
    for entry, fasta_path in species_paths:
        for contig_name in _iter_fasta_contig_names(fasta_path):
            prior = contig_owner.get(contig_name)
            if prior is not None and prior != entry.species_slug:
                raise RuntimeError(
                    "Cross-species contig-name collision detected: "
                    f"contig {contig_name!r} is declared by both {prior!r} and "
                    f"{entry.species_slug!r}; aborting before windowing to avoid "
                    "silent locus_id aliasing"
                )
            contig_owner[contig_name] = entry.species_slug


def _can_multiprocess_tokenizer_loader(tokenizer_loader: TokenizerLoader) -> bool:
    """Return whether the loader can cross a ``spawn`` multiprocessing boundary.

    Nested test doubles are intentionally not pickleable. Falling back to the
    sequential path keeps injection-based tests working while production still
    uses the parallel producer/consumer implementation.
    """
    try:
        pickle.dumps(tokenizer_loader)
    except Exception:
        return False
    return True


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


def _serialize_checkpoint(checkpoint: CheckpointState) -> dict[str, Any]:
    """Convert checkpoint state into the stable JSON payload written on disk."""
    return {
        "schema_version": checkpoint.schema_version,
        "config_path": checkpoint.config_path,
        "config_name": checkpoint.config_name,
        "updated_at": checkpoint.updated_at,
        "completed_species": list(checkpoint.completed_species),
        "per_species_stats": {
            stats.species_slug: asdict(stats) for stats in checkpoint.per_species_stats
        },
    }


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    config_file: Path,
    config_name: str,
) -> CheckpointState:
    """Load checkpoint state for the current config, or return an empty state."""
    if not checkpoint_path.exists():
        return CheckpointState.empty(config_file=config_file, config_name=config_name)

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed_species = tuple(payload.get("completed_species", ()))
    per_species_payload = payload.get("per_species_stats", {})
    restored_stats: list[FelidSpeciesPretrainStats] = []
    for species_slug in completed_species:
        species_payload = per_species_payload.get(species_slug)
        if species_payload is None:
            raise RuntimeError(
                f"Checkpoint is missing per-species stats for {species_slug!r}; "
                "delete checkpoint.json to start fresh"
            )
        restored_stats.append(FelidSpeciesPretrainStats(**species_payload))

    checkpoint = CheckpointState(
        schema_version=str(payload.get("schema_version", "")),
        config_path=str(payload.get("config_path", "")),
        config_name=str(payload.get("config_name", "")),
        updated_at=str(payload.get("updated_at", "")),
        completed_species=completed_species,
        per_species_stats=tuple(restored_stats),
    )
    checkpoint.validate(config_file=config_file, config_name=config_name)
    return checkpoint


def _build_checkpoint(
    *,
    config_file: Path,
    config_name: str,
    completed_species: list[str],
    stats_by_slug: dict[str, FelidSpeciesPretrainStats],
) -> CheckpointState:
    """Create a fresh checkpoint snapshot from the current completed species set."""
    return CheckpointState(
        schema_version=_CHECKPOINT_SCHEMA_VERSION,
        config_path=str(config_file.resolve()),
        config_name=config_name,
        updated_at=datetime.now(UTC).isoformat(),
        completed_species=tuple(completed_species),
        per_species_stats=tuple(stats_by_slug[species_slug] for species_slug in completed_species),
    )


def _write_checkpoint(checkpoint_path: Path, checkpoint: CheckpointState) -> None:
    """Atomically persist checkpoint state in the artifact directory."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.stem}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(_serialize_checkpoint(checkpoint), indent=2, sort_keys=True))
        temp_path.replace(checkpoint_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _terminate_workers(
    live_processes: dict[str, multiprocessing.process.BaseProcess],
    *,
    sigterm_timeout: float = _DEFAULT_WORKER_SIGTERM_TIMEOUT_SECONDS,
) -> None:
    """Stop all workers, escalating from SIGTERM to SIGKILL if required."""
    for process in live_processes.values():
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + sigterm_timeout
    for process in live_processes.values():
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
        if process.is_alive():
            process.kill()
            process.join()


def _species_worker(
    *,
    entry: FelidSpeciesEntry,
    fasta_path: Path,
    preprocessing_config: Any,
    provenance: TokenizerProvenance,
    tokenizer_loader: TokenizerLoader,
    queue: Any,
    corpus_dir: Path,
    chunk_size: int,
    shutdown_event: Any,
    put_timeout_seconds: float,
) -> None:
    """Run one species in a worker and stream tokenized chunks over the queue."""
    try:
        tokenizer, loaded_provenance = tokenizer_loader(provenance)
        if loaded_provenance != provenance:
            raise RuntimeError(
                "Worker tokenizer provenance does not match the approved config contract"
            )
        stats = _run_single_species(
            entry=entry,
            fasta_path=fasta_path,
            preprocessing_config=preprocessing_config,
            tokenizer=tokenizer,
            provenance=loaded_provenance,
            writer=_QueueBatchWriter(
                species_slug=entry.species_slug,
                queue=queue,
                shutdown_event=shutdown_event,
                put_timeout_seconds=put_timeout_seconds,
            ),
            contig_owner={},
            corpus_dir=corpus_dir,
            chunk_size=chunk_size,
        )
        _put_queue_message(
            ipc_queue=queue,
            message=_DoneMessage(species_slug=entry.species_slug, stats=stats),
            shutdown_event=shutdown_event,
            put_timeout_seconds=put_timeout_seconds,
        )
    except BaseException as exc:
        if isinstance(exc, _WorkerShutdownRequestedError):
            return
        try:
            _put_queue_message(
                ipc_queue=queue,
                message=_ErrorMessage(
                    species_slug=entry.species_slug,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback_str=traceback.format_exc(),
                ),
                shutdown_event=shutdown_event,
                put_timeout_seconds=put_timeout_seconds,
            )
        except Exception:
            _LOGGER.exception(
                "species_worker_error_report_failed species=%s pid=%d",
                entry.species_slug,
                os.getpid(),
            )


def _consume_queue(
    *,
    queue: Any,
    writer: _WriterContextManager,
    expected_species: int,
    checkpoint_path: Path,
    config_file: Path,
    config_name: str,
    checkpointed_species: tuple[str, ...],
    checkpointed_stats_by_slug: dict[str, FelidSpeciesPretrainStats],
    on_species_done: Callable[[], None],
    on_queue_idle: Callable[[], None] | None = None,
    queue_get_timeout_seconds: float | None = None,
) -> list[FelidSpeciesPretrainStats]:
    """Drain queue messages until every expected species finishes or one errors.

    The consumer intentionally polls ``queue.get`` with a timeout when the
    parallel path is active. Catastrophic worker exits (for example ``os._exit``,
    SIGKILL, or OOM termination) bypass the worker's structured ``_ErrorMessage``
    path entirely; timed polling lets the parent inspect worker liveness and fail
    fast instead of hanging forever on an empty queue.
    """
    done_count = 0
    total_windows = 0
    completed_stats: list[FelidSpeciesPretrainStats] = []
    completed_species_snapshot = list(checkpointed_species)
    stats_by_slug_snapshot = dict(checkpointed_stats_by_slug)

    while done_count < expected_species:
        try:
            if queue_get_timeout_seconds is None:
                message = queue.get()
            else:
                message = queue.get(timeout=queue_get_timeout_seconds)
        except queue_module.Empty:
            if on_queue_idle is not None:
                on_queue_idle()
            continue
        if isinstance(message, _ChunkMessage):
            writer.write_batch(message.chunk)
            total_windows += len(message.chunk)
            continue
        if isinstance(message, _DoneMessage):
            stats_by_slug_snapshot[message.species_slug] = message.stats
            completed_species_snapshot.append(message.species_slug)
            _write_checkpoint(
                checkpoint_path,
                _build_checkpoint(
                    config_file=config_file,
                    config_name=config_name,
                    completed_species=completed_species_snapshot,
                    stats_by_slug=stats_by_slug_snapshot,
                ),
            )
            completed_stats.append(message.stats)
            done_count += 1
            on_species_done()
            continue
        if isinstance(message, _ErrorMessage):
            raise RuntimeError(
                f"Worker for {message.species_slug!r} failed "
                f"({message.error_type}): {message.error_message}\n"
                f"{message.traceback_str}"
            )
        raise RuntimeError(
            f"Unknown message type from queue: {type(message).__name__!r}. "
            "This indicates a version mismatch or queue corruption. Aborting."
        )

    if total_windows == 0:
        raise _ZeroTokenizedWindowsError(
            "Felid foundation pretrain produced zero tokenized windows across all "
            "species; check windowing/ambiguity filters and per-species FASTA contents"
        )

    return completed_stats


def _run_parallel_pipeline(
    species_paths: list[tuple[FelidSpeciesEntry, Path]],
    *,
    preprocessing_config: Any,
    provenance: TokenizerProvenance,
    tokenizer_loader: TokenizerLoader,
    writer: _WriterContextManager,
    checkpoint_path: Path,
    config_file: Path,
    config_name: str,
    checkpointed_species: tuple[str, ...],
    checkpointed_stats_by_slug: dict[str, FelidSpeciesPretrainStats],
    corpus_dir: Path,
    num_workers: int,
    chunk_size: int,
    queue_maxsize_factor: int = _DEFAULT_QUEUE_MAXSIZE_FACTOR,
    sigterm_timeout: float = _DEFAULT_WORKER_SIGTERM_TIMEOUT_SECONDS,
) -> list[FelidSpeciesPretrainStats]:
    """Run spawn-based producer workers and a single main-process consumer."""
    completed_species_set = set(checkpointed_species)
    already_completed = [
        entry.species_slug
        for entry, _fasta_path in species_paths
        if entry.species_slug in completed_species_set
    ]
    if already_completed:
        raise ValueError(
            "Parallel species dispatch received checkpointed species unexpectedly: "
            + ", ".join(sorted(already_completed))
        )
    if not species_paths:
        return []

    ctx = multiprocessing.get_context("spawn")
    shutdown_event = ctx.Event()
    queue = ctx.Queue(maxsize=max(1, queue_maxsize_factor * num_workers))
    pending: deque[tuple[FelidSpeciesEntry, Path]] = deque(species_paths[num_workers:])
    live_processes: dict[str, multiprocessing.process.BaseProcess] = {}
    put_timeout_seconds = max(0.1, min(1.0, sigterm_timeout / 4.0))
    shared_kwargs = {
        "preprocessing_config": preprocessing_config,
        "provenance": provenance,
        "tokenizer_loader": tokenizer_loader,
        "queue": queue,
        "corpus_dir": corpus_dir,
        "chunk_size": chunk_size,
        "shutdown_event": shutdown_event,
        "put_timeout_seconds": put_timeout_seconds,
    }

    def _start_worker(entry: FelidSpeciesEntry, fasta_path: Path) -> None:
        """Launch a producer for exactly one species."""
        process = ctx.Process(
            target=_species_worker,
            kwargs={"entry": entry, "fasta_path": fasta_path, **shared_kwargs},
            daemon=True,
        )
        process.start()
        live_processes[entry.species_slug] = process

    def _dispatch_next() -> None:
        """Start the next queued species after one finishes."""
        if pending:
            _start_worker(*pending.popleft())

    def _raise_if_worker_exit_bypassed_queue() -> None:
        """Abort if an idle consumer observes workers that can no longer emit messages.

        The worker normally reports failures by enqueueing ``_ErrorMessage``.
        Abrupt exits such as ``os._exit`` or external termination can prevent that
        final enqueue, so the consumer must inspect process exit codes whenever the
        queue stays empty for longer than the polling interval.
        """
        for species_slug, process in live_processes.items():
            exitcode = process.exitcode
            if exitcode is None or exitcode == 0:
                continue
            termination_reason = f"signal {-exitcode}" if exitcode < 0 else f"exit code {exitcode}"
            raise RuntimeError(
                f"Worker for {species_slug!r} exited unexpectedly with {termination_reason} "
                "before reporting completion"
            )
        if pending:
            return
        if live_processes and all(process.exitcode == 0 for process in live_processes.values()):
            raise RuntimeError(
                "All felid-foundation workers exited before the consumer observed every "
                "completion message; aborting instead of waiting forever on an empty queue"
            )

    for entry, fasta_path in species_paths[:num_workers]:
        _start_worker(entry, fasta_path)

    try:
        completed_stats = _consume_queue(
            queue=queue,
            writer=writer,
            expected_species=len(species_paths),
            checkpoint_path=checkpoint_path,
            config_file=config_file,
            config_name=config_name,
            checkpointed_species=checkpointed_species,
            checkpointed_stats_by_slug=checkpointed_stats_by_slug,
            on_species_done=_dispatch_next,
            on_queue_idle=_raise_if_worker_exit_bypassed_queue,
            queue_get_timeout_seconds=put_timeout_seconds,
        )
        shutdown_event.set()
        for process in live_processes.values():
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
        return completed_stats
    except BaseException:
        shutdown_event.set()
        queue.cancel_join_thread()
        _terminate_workers(live_processes, sigterm_timeout=sigterm_timeout)
        raise


def _restore_completed_contig_owners(
    *,
    completed_species: Iterable[str],
    species_paths: list[tuple[FelidSpeciesEntry, Path]],
    contig_owner: dict[str, str],
) -> None:
    """Rebuild contig ownership for skipped species before resuming later ones."""
    species_lookup = {
        entry.species_slug: (entry, fasta_path) for entry, fasta_path in species_paths
    }
    for species_slug in completed_species:
        entry, fasta_path = species_lookup[species_slug]
        for contig_name in _iter_fasta_contig_names(fasta_path):
            prior = contig_owner.get(contig_name)
            if prior is not None and prior != entry.species_slug:
                raise RuntimeError(
                    "Cross-species contig-name collision detected while restoring "
                    f"checkpointed species: contig {contig_name!r} is declared by both "
                    f"{prior!r} and {entry.species_slug!r}; aborting resume"
                )
            contig_owner[contig_name] = entry.species_slug


def _resume_writer_from_metadata(
    *,
    writer: TokenizedCorpusWriter,
    corpus_dir: Path,
    provenance: TokenizerProvenance,
) -> None:
    """Seed a reopened writer with prior metadata so resumed batches append safely.

    Resume relies on the previous clean-close metadata sidecar because the writer
    stores both the split file registry and the SQLite-backed locus manifest in
    process-local state rather than discovering them automatically on reopen.

    The direct writes to ``TokenizedCorpusWriter`` private attributes below are
    intentional and narrowly scoped to resume hydration. The writer exposes no
    public API for restoring ``metadata.json`` + SQLite sidecar state, but resume
    must rehydrate exactly these fields so subsequent ``write_batch`` calls append
    with the original split registry, locus manifest, tokenizer provenance, and
    batch index. Keeping the private mutation isolated to this helper makes that
    contract explicit until the writer grows a first-class resume hook.
    """
    metadata_path = corpus_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            f"Cannot resume felid-foundation run: {metadata_path} is missing. "
            "Delete checkpoint.json to start fresh if the previous run was terminated uncleanly."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_tokenizer = json.loads(json.dumps(asdict(provenance), sort_keys=True))
    if metadata.get("tokenizer") != expected_tokenizer:
        raise RuntimeError(
            f"Existing corpus metadata at {metadata_path} does not match the current "
            "tokenizer provenance; delete the corpus and checkpoint to start fresh"
        )

    max_batch_index = -1
    for split, split_payload in metadata.get("splits", {}).items():
        relative_files = split_payload.get("files", [])
        absolute_files = [corpus_dir / relative_path for relative_path in relative_files]
        # NOTE: resume must hydrate the writer's internal registries before any new
        # batch is appended; there is intentionally no public setter for this state.
        writer._split_paths[split] = absolute_files
        writer._split_record_counts[split] = int(split_payload.get("record_count", 0))
        for path in absolute_files:
            match = _PART_FILE_NAME_PATTERN.match(path.name)
            if match is not None:
                max_batch_index = max(max_batch_index, int(match.group(1)))

    split_manifest = metadata.get("split_manifest", [])
    if split_manifest:
        if writer._sqlite_conn is None:
            raise RuntimeError("TokenizedCorpusWriter resume requires an open SQLite sidecar")
        writer._sqlite_conn.executemany(
            "INSERT INTO locus_entries (locus_id, contig, block_start, block_end, split) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    str(entry["locus_id"]),
                    str(entry["contig"]),
                    int(entry["block_start"]),
                    int(entry["block_end"]),
                    str(entry["split"]),
                )
                for entry in split_manifest
            ],
        )
        writer._sqlite_conn.commit()

    # These fields must match the pre-existing corpus so future write_batch calls
    # continue numbering files monotonically and preserve the validated provenance.
    writer._resolved_provenance = provenance
    writer._batch_index = max_batch_index + 1


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
    chunk_size: int = _DEFAULT_TOKENIZED_WINDOW_CHUNK_SIZE,
) -> FelidSpeciesPretrainStats:
    """Run prepare → window → tokenize → write for one species.

    This helper owns the per-species streaming-writer contract.
    It reads the species FASTA lazily, guards cross-species contig
    collisions on first sighting, decomposes the prepare/window/tokenize
    cascade so the intermediate retained/filtered counts required by the
    pinned run-summary schema are observable, and writes tokenized windows
    to :meth:`TokenizedCorpusWriter.write_batch` in fixed-size chunks so
    the full species batch is never materialized in tokenized form.
    Structured
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
        "species_start species=%s identifier=%s fasta=%s pid=%d",
        entry.species_slug,
        entry.identifier,
        fasta_path,
        os.getpid(),
    )

    contig_count = 0
    for contig_name in _iter_fasta_contig_names(fasta_path):
        prior = contig_owner.get(contig_name)
        if prior is not None and prior != entry.species_slug:
            raise RuntimeError(
                "Cross-species contig-name collision detected: "
                f"contig {contig_name!r} is declared by both "
                f"{prior!r} and {entry.species_slug!r}; aborting "
                "before windowing to avoid silent locus_id aliasing"
            )
        contig_owner[contig_name] = entry.species_slug
        contig_count += 1
    _LOGGER.info(
        "fasta_parsed species=%s identifier=%s contigs=%d",
        entry.species_slug,
        entry.identifier,
        contig_count,
    )

    filter_reason_counts: dict[str, int] = {}
    retained_sequence_count = 0
    filtered_sequence_count = 0
    bytes_tokenized = 0
    windows_generated_count = 0
    windows_tokenized_count = 0
    window_counts_by_split: dict[str, int] = {"train": 0, "validation": 0}
    peak_window_count_in_memory = 0

    def _iter_species_tokenized_chunks() -> Iterator[tuple[TokenizedWindow, ...]]:
        """Yield fixed-size tokenized chunks while preserving collision semantics.

        A second FASTA pass is intentional: it preserves the original
        contract that cross-species contig collisions abort before any
        tokenized output is written, while still keeping tokenized memory
        bounded by a single chunk.
        """

        nonlocal retained_sequence_count
        nonlocal filtered_sequence_count
        nonlocal bytes_tokenized
        nonlocal windows_generated_count
        nonlocal windows_tokenized_count
        nonlocal peak_window_count_in_memory

        for record in _iter_species_sequence_records(fasta_path, entry.species_slug):
            report = prepare_sequences([record], preprocessing_config)
            retained_sequence_count += len(report.retained)
            filtered_sequence_count += len(report.filtered)
            for filtered in report.filtered:
                filter_reason_counts[filtered.reason] = (
                    filter_reason_counts.get(filtered.reason, 0) + 1
                )
            if not report.retained:
                continue

            windows = window_sequences(list(report.retained), preprocessing_config)
            windows_generated_count += len(windows)
            for start in range(0, len(windows), chunk_size):
                chunk_windows = windows[start : start + chunk_size]
                if not chunk_windows:
                    continue
                bytes_tokenized += sum(len(window.sequence) for window in chunk_windows)
                tokenized_chunk = tokenize_windows(
                    chunk_windows,
                    tokenizer,
                    provenance=provenance,
                )
                peak_window_count_in_memory = max(
                    peak_window_count_in_memory,
                    len(tokenized_chunk),
                )
                windows_tokenized_count += len(tokenized_chunk)
                for window in tokenized_chunk:
                    split = window.window.split
                    window_counts_by_split[split] = window_counts_by_split.get(split, 0) + 1
                yield tokenized_chunk

    for tokenized_chunk in _iter_species_tokenized_chunks():
        writer.write_batch(tokenized_chunk)

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
        retained_sequence_count,
        filtered_sequence_count,
    )
    _LOGGER.info(
        "windows_generated species=%s identifier=%s windows=%d",
        entry.species_slug,
        entry.identifier,
        windows_generated_count,
    )
    _LOGGER.info(
        "windows_tokenized species=%s identifier=%s tokens=%d",
        entry.species_slug,
        entry.identifier,
        windows_tokenized_count,
    )

    peak_rss_bytes = _read_peak_rss_bytes()
    _LOGGER.info(
        "species_end species=%s identifier=%s windows=%d peak_rss_bytes=%d pid=%d",
        entry.species_slug,
        entry.identifier,
        windows_tokenized_count,
        peak_rss_bytes,
        os.getpid(),
    )

    return FelidSpeciesPretrainStats(
        species_slug=entry.species_slug,
        identifier=entry.identifier,
        assembly_name=entry.assembly_name,
        contig_count=contig_count,
        retained_sequence_count=retained_sequence_count,
        filtered_short_count=filter_reason_counts.get("short_sequence", 0),
        filtered_high_ambiguity_count=filter_reason_counts.get("high_ambiguity", 0),
        window_counts_by_split=window_counts_by_split,
        peak_window_count_in_memory=peak_window_count_in_memory,
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
    3. Runs prepare → window → tokenize per contig, yielding tokenized
       windows as fixed-size chunks while tracking the intermediate
       filter counts required by the run-summary schema.
    4. Calls :meth:`TokenizedCorpusWriter.write_batch` once per chunk and
       releases each chunk before the next chunk is produced, so peak
       memory is bounded by the single largest chunk rather than the full
       corpus.

    The run also maintains ``{artifact_dir}/checkpoint.json``. Each species is
    added to the checkpoint only after it finishes successfully, so a restart can
    skip already-completed species and rebuild the final summary from persisted
    stats.

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
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "checkpoint.json"

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
    export_contract = _build_export_contract(config)
    corpus_dir = processed_dir / "felid_foundation_tokens"
    checkpoint = _load_checkpoint(
        checkpoint_path,
        config_file=config_file,
        config_name=config.name,
    )

    contig_owner: dict[str, str] = {}
    completed_species = list(checkpoint.completed_species)
    stats_by_slug = {stats.species_slug: stats for stats in checkpoint.per_species_stats}

    completed_species_set = set(completed_species)
    remaining_species_paths = [
        (entry, fasta_path)
        for entry, fasta_path in species_paths
        if entry.species_slug not in completed_species_set
    ]
    for entry, _fasta_path in species_paths:
        if entry.species_slug in completed_species_set:
            _LOGGER.info(
                "species_skip_checkpoint species=%s checkpoint=%s",
                entry.species_slug,
                checkpoint_path,
            )

    available_cpu_count = max(1, os.cpu_count() or 1)
    parallel_worker_count = min(
        len(remaining_species_paths),
        available_cpu_count,
        config.pipeline.num_workers,
    )
    use_parallel_pipeline = (
        len(remaining_species_paths) > 0
        and parallel_worker_count > 1
        and _can_multiprocess_tokenizer_loader(tokenizer_loader)
    )
    if use_parallel_pipeline:
        provenance = expected_provenance
        _preflight_contig_check(species_paths)
    else:
        tokenizer, provenance = tokenizer_loader(expected_provenance)
        _assert_tokenizer_matches_config(config, provenance)
        _restore_completed_contig_owners(
            completed_species=completed_species,
            species_paths=species_paths,
            contig_owner=contig_owner,
        )

    run_failure: BaseException | None = None
    run_failure_tb = None
    if remaining_species_paths:
        with export_writer(
            corpus_dir,
            contract=export_contract,
            provenance=provenance,
        ) as writer:
            if checkpoint.completed_species and isinstance(writer, TokenizedCorpusWriter):
                _resume_writer_from_metadata(
                    writer=writer,
                    corpus_dir=corpus_dir,
                    provenance=provenance,
                )
            try:
                if use_parallel_pipeline:
                    completed_parallel_stats = _run_parallel_pipeline(
                        remaining_species_paths,
                        preprocessing_config=preprocessing_config,
                        provenance=provenance,
                        tokenizer_loader=tokenizer_loader,
                        writer=writer,
                        checkpoint_path=checkpoint_path,
                        config_file=config_file,
                        config_name=config.name,
                        checkpointed_species=tuple(completed_species),
                        checkpointed_stats_by_slug=dict(stats_by_slug),
                        corpus_dir=corpus_dir,
                        num_workers=parallel_worker_count,
                        chunk_size=config.pipeline.chunk_size,
                        queue_maxsize_factor=config.pipeline.queue_maxsize_factor,
                        sigterm_timeout=config.pipeline.sigterm_timeout,
                    )
                    for stats in completed_parallel_stats:
                        stats_by_slug[stats.species_slug] = stats
                        completed_species.append(stats.species_slug)
                        completed_species_set.add(stats.species_slug)
                else:
                    for entry, fasta_path in remaining_species_paths:
                        stats = _run_single_species(
                            entry=entry,
                            fasta_path=fasta_path,
                            preprocessing_config=preprocessing_config,
                            tokenizer=tokenizer,
                            provenance=provenance,
                            writer=writer,
                            contig_owner=contig_owner,
                            corpus_dir=corpus_dir,
                            chunk_size=config.pipeline.chunk_size,
                        )
                        stats_by_slug[stats.species_slug] = stats
                        completed_species.append(stats.species_slug)
                        completed_species_set.add(stats.species_slug)
                        _write_checkpoint(
                            checkpoint_path,
                            _build_checkpoint(
                                config_file=config_file,
                                config_name=config.name,
                                completed_species=completed_species,
                                stats_by_slug=stats_by_slug,
                            ),
                        )
            except BaseException as exc:
                if isinstance(exc, _ZeroTokenizedWindowsError):
                    raise RuntimeError(str(exc)) from exc
                run_failure = exc
                run_failure_tb = exc.__traceback__
            if run_failure is None:
                completed_total_windows = sum(
                    sum(stats.window_counts_by_split.values()) for stats in stats_by_slug.values()
                )
                if completed_total_windows == 0:
                    raise RuntimeError(
                        "Felid foundation pretrain produced zero tokenized windows across all "
                        "species; check windowing/ambiguity filters and per-species FASTA contents"
                    )
            if run_failure is not None:
                _LOGGER.warning(
                    "felid_foundation_interrupted completed_species=%d remaining_species=%d",
                    len(completed_species),
                    len(species_paths) - len(completed_species),
                )

    if run_failure is not None:
        raise run_failure.with_traceback(run_failure_tb)

    per_species_stats = [
        stats_by_slug[entry.species_slug]
        for entry, _fasta_path in species_paths
        if entry.species_slug in stats_by_slug
    ]
    totals: dict[str, int] = {"train": 0, "validation": 0}
    for stats in per_species_stats:
        for split, count in stats.window_counts_by_split.items():
            totals[split] = totals.get(split, 0) + count

    total_window_count = sum(totals.values())
    if total_window_count == 0:
        raise RuntimeError(
            "Felid foundation pretrain produced zero tokenized windows across all "
            "species; check windowing/ambiguity filters and per-species FASTA contents"
        )

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
