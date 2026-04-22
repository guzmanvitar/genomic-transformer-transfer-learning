"""Cross-pipeline helpers shared by the consensus and felid-foundation pretraining paths.

The consensus (feline) pretraining pipeline and the felid-foundation
pretraining pipeline both load a typed pipeline config, map it to
preprocessor/tokenizer contracts, resolve filesystem paths, and run
the prepare → window → tokenize cascade. Before this module existed
those helpers lived exclusively in ``pretrain/pipeline.py`` and the
new felid pipeline would have had to either duplicate them (drift
risk) or import private symbols from a sibling pipeline module
(tight coupling).

Moving the truly shared helpers here gives both pipelines a single
source of truth. ``pretrain/pipeline.py`` re-imports every moved
symbol at its own module scope so that existing tests which patch
``jaguar_geo_assign.pretrain.pipeline.<helper>`` via
``monkeypatch.setattr`` continue to work unchanged — the re-import
is the explicit test-seam preservation called out in the task DoD.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Iterable

from ..data.preprocessor import (
    ExportContract,
    PreprocessingConfig,
    SequenceRecord,
    TokenizedWindow,
    TokenizerProvenance,
    prepare_sequences,
    tokenize_windows,
    window_sequences,
)


def normalize_ru_maxrss_to_bytes(raw: int, platform: str) -> int:
    """Normalise a raw ``ru_maxrss`` reading to bytes across platforms.

    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` reports peak
    resident-set-size in **kilobytes** on Linux and in **bytes** on
    macOS. A peak-RSS telemetry field that silently mixes units would
    make the run-summary JSON useless for cross-platform comparison
    (and would understate macOS runs by 1000x on a Linux reader's
    mental model).

    Args:
        raw: The raw ``ru_maxrss`` integer returned by
            :func:`resource.getrusage`. Caller is responsible for the
            actual syscall; this helper performs no I/O.
        platform: Platform string returned by :data:`sys.platform`. The
            ``linux`` family (including ``linux2``) is treated as KB;
            ``darwin`` is treated as bytes.

    Returns:
        Normalised peak RSS expressed in bytes.

    Raises:
        ValueError: If *platform* is not a recognised normalisation
            target. Unknown platforms fail loudly rather than silently
            picking a default.
    """
    if platform.startswith("linux"):
        return int(raw) * 1024
    if platform == "darwin":
        return int(raw)
    raise ValueError(
        f"Unsupported platform {platform!r} for ru_maxrss normalisation; "
        "expected 'linux*' (KB) or 'darwin' (bytes)"
    )


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


def _resolve_path(base_dir: Path, path: str | Path, *, prefer_cwd: bool = False) -> Path:
    """Resolve a potentially relative path against a base directory.

    Absolute paths are returned unchanged. For relative paths, two
    candidates are formed: one relative to the current working directory
    and one relative to *base_dir*.

    Fragility flag — ``prefer_cwd`` strategy:
        When ``prefer_cwd=True``, the CWD candidate is returned if it
        exists or if neither candidate exists (defaulting to CWD). When
        ``prefer_cwd=False`` (default), the *base_dir* candidate wins if
        it exists, otherwise the CWD candidate is returned. This asymmetry
        means pipeline behaviour depends on the caller's working
        directory, which can cause hard-to-reproduce path resolution in
        CI vs. local runs.

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


def _open_maybe_gzip(path: Path):
    """Open a file for text reading, auto-detecting gzip by ``.gz`` suffix.

    Args:
        path: Filesystem path to open.

    Returns:
        A text-mode file handle (UTF-8).
    """
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _iter_fasta_sequences(path: str | Path) -> Iterable[tuple[str, str]]:
    """Lazily parse a FASTA file, yielding ``(contig_name, sequence)`` pairs.

    Supports both plain-text and gzip-compressed (``.gz``) FASTA files.
    Contig names are extracted from the first whitespace-delimited token
    after the ``>`` header marker. Empty lines are skipped.

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


def _require_runtime_boolean(value: object, *, field_name: str) -> bool:
    """Assert that *value* is a genuine ``bool``, not a truthy surrogate.

    Uses ``type(value) is not bool`` (identity check) so that ``int``
    values like ``1`` or ``0`` are rejected even though they pass
    ``== True`` / ``== False``. This prevents subtle security
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


def _build_preprocessing_config(config: Any) -> PreprocessingConfig:
    """Derive a ``PreprocessingConfig`` from any pipeline config with windowing/split/tokenizer sections.

    The consensus and felid-foundation configs share the same
    windowing/split/tokenizer shape even though their top-level dataclass
    types differ. Duck-typing on the three required attribute paths
    lets one helper serve both pipelines without leaking cross-pipeline
    coupling into the config layer.

    Args:
        config: Any object exposing ``config.windowing.context_window``,
            ``config.windowing.window_overlap``,
            ``config.windowing.max_ambiguous_fraction``,
            ``config.windowing.drop_short_sequences``,
            ``config.split.locus_block_size``, and
            ``config.tokenizer.unsupported_symbol_policy``.

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


def _build_tokenizer_provenance(config: Any) -> TokenizerProvenance:
    """Build a ``TokenizerProvenance`` from any pipeline config's tokenizer section.

    Args:
        config: Any object exposing a ``config.tokenizer`` section with the
            standard DNABERT-2 fields.

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


def _assert_tokenizer_matches_config(config: Any, provenance: TokenizerProvenance) -> None:
    """Validate that the loaded tokenizer's provenance matches the config.

    Checks identifier, revision, alphabet, max_position_embeddings, and
    unsupported_symbol_policy via equality. The ``trust_remote_code``
    field is checked via **identity** (``is not``) after passing through
    ``_require_runtime_boolean``, which rejects non-bool truthy values
    (e.g. ``1``). This is intentional: a deserialised integer ``1``
    equals ``True`` but is not ``True``, and accepting it could silently
    enable remote code execution.

    Args:
        config: Pipeline configuration exposing a ``config.tokenizer``
            section with the expected tokenizer contract.
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


def _build_export_contract(config: Any) -> ExportContract:
    """Derive an ``ExportContract`` from any pipeline config's export section.

    Args:
        config: Any object exposing a ``config.export`` section with the
            standard parquet-export fields.

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


def _tokenize_sequence_records(
    records: Iterable[SequenceRecord],
    preprocessing_config: PreprocessingConfig,
    tokenizer: object,
    provenance: TokenizerProvenance,
) -> tuple[TokenizedWindow, ...]:
    """Prepare, window, and tokenize an iterable of sequence records.

    Processes each record individually through the prepare → window →
    tokenize pipeline. Records or windows that fail quality filters
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

