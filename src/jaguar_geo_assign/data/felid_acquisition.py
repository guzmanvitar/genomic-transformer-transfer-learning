"""Operator-facing acquisition runtime for the six approved felid reference FASTAs.

This module is the single entry point for downloading the pinned
multi-species felid foundation corpus. It wraps
:func:`jaguar_geo_assign.data.acquisition.download_with_retry` with two
felid-specific concerns that the generic primitive does not express:

1. **Explicit MD5-mismatch observability.** The generic retry loop skips
   when the destination checksum already matches and silently deletes +
   re-downloads on mismatch. Foundation operators need the mismatch event
   (actual hash vs. expected hash, plus the accession) surfaced in logs;
   otherwise a silently-corrected corruption bug would be undetectable.
2. **Root-cause-preserving error contract.** :class:`FelidAcquisitionError`
   surfaces the underlying exception class name and message, plus the
   failing accession and pinned checksum, so a ``ConnectionResetError``
   vs. ``HTTPError`` vs. checksum mismatch is distinguishable without
   parsing tracebacks.

This module is deliberately decoupled from the pretraining runtime:
``run_felid_foundation_pretrain`` imports neither this module nor
``acquisition.py`` directly — the acquire step is a separate CLI verb so
the pretrain entry point never implicitly reaches the network.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import OpenerDirector, build_opener

from ..config import FelidFoundationPipelineConfig, FelidSpeciesEntry
from .acquisition import AcquisitionError, DownloadAsset, DownloadResult, download_with_retry
from .felid_assemblies import APPROVED_FELID_ASSEMBLIES, build_felid_reference_manifest

_LOGGER = logging.getLogger(__name__)


class FelidAcquisitionError(RuntimeError):
    """Raised when an approved felid reference FASTA cannot be acquired or verified.

    The CLI operator needs the root cause (``ConnectionResetError``,
    ``HTTPError``, ``ChecksumMismatch``) surfaced directly in the error
    message so they can distinguish transient network faults from
    deterministic contract violations without digging through traceback
    chains. The message therefore includes the failing accession, the
    pinned expected MD5, and the class name plus string form of the
    underlying exception.
    """


@dataclass(frozen=True)
class FelidAcquisitionSummary:
    """Typed summary of a felid-foundation acquisition run.

    The only value returned to the CLI so operators can inspect
    which files were downloaded, skipped, or redownloaded without
    re-parsing logs. Every field mirrors a metric emitted by the
    download loop.

    Attributes:
        per_species: Tuple of :class:`DownloadResult` in accession-sorted
            registry order, one per approved species.
        total_bytes_written: Sum of bytes written across all downloads
            (zero when every file was skipped because it matched the
            pinned checksum).
        skipped_count: Number of FASTAs that already existed with the
            correct MD5 and were therefore not re-downloaded.
        redownloaded_count: Number of FASTAs that existed but had the
            wrong MD5 and were deleted + redownloaded.
    """

    per_species: tuple[DownloadResult, ...]
    total_bytes_written: int
    skipped_count: int
    redownloaded_count: int


def _compute_md5(path: Path) -> str:
    """Return the lowercase hex MD5 of *path* using streaming reads.

    Felid assemblies are multi-hundred-MB ``.fna.gz`` blobs.
    Loading the whole file into memory to hash it would make the acquire
    step a memory hog on operators' laptops; a 1 MiB streaming chunk
    keeps the hash cost bounded and matches the chunk size already used
    by :func:`download_with_retry`.
    """
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _zip_entries_with_assets(
    species: tuple[FelidSpeciesEntry, ...],
    assets: tuple[DownloadAsset, ...],
) -> tuple[tuple[FelidSpeciesEntry, DownloadAsset], ...]:
    """Pair each configured species with its matching :class:`DownloadAsset`.

    :class:`DownloadAsset` is intentionally accession-agnostic so
    the acquisition layer stays generic. For felid-specific logging we
    need the species slug and Latin binomial alongside the asset. The
    manifest is sorted by accession; we index assets by their
    ``destination`` stem (``<ACC>_<ASM>``) so the pairing works even when
    the config's species list is in a different order.
    """
    by_stem = {asset.destination.name.split(".fna.gz")[0]: asset for asset in assets}
    pairs: list[tuple[FelidSpeciesEntry, DownloadAsset]] = []
    for entry in species:
        stem = f"{entry.accession}_{entry.assembly_name}"
        asset = by_stem.get(stem)
        if asset is None:
            raise FelidAcquisitionError(
                f"Config species {entry.species} ({entry.accession}) has no "
                "matching asset in the felid reference manifest; the species "
                "list must be a subset of APPROVED_FELID_ASSEMBLIES"
            )
        pairs.append((entry, asset))
    return tuple(pairs)


def acquire_felid_foundation_assemblies(
    config: FelidFoundationPipelineConfig,
    *,
    opener: OpenerDirector | None = None,
    output_dir: str | Path | None = None,
    retries: int = 3,
    sleep: Callable[[float], None] | None = None,
) -> FelidAcquisitionSummary:
    """Download every configured felid reference FASTA with integrity checks.

    The operator-facing acquisition entry point. It resolves
    destination paths via :func:`build_felid_reference_manifest`,
    verifies existing files by MD5, deletes and re-downloads on
    mismatch, and short-circuits on match. The download is idempotent:
    invoking this function twice in a row (barring disk corruption)
    writes zero bytes on the second call and returns ``skipped_count``
    equal to the species count. Structured ``INFO``-level logging
    records every ``start``, ``skip``, ``verify_mismatch_redownload``,
    ``download_finish``, and ``failure`` event so operators can audit
    the acquisition without inspecting the file system.

    Args:
        config: Validated felid-foundation pipeline config. The
            ``paths.reference_dir`` field is the default download root.
        opener: Optional :class:`~urllib.request.OpenerDirector` for
            dependency injection (unit tests supply a fake that returns
            canned FASTA bytes).
        output_dir: Optional explicit download root that overrides
            ``config.paths.reference_dir`` (used by integration tests).
        retries: Retry budget forwarded to :func:`download_with_retry`.
        sleep: Optional sleep callable forwarded to
            :func:`download_with_retry`; defaults to ``time.sleep``.

    Returns:
        A :class:`FelidAcquisitionSummary` describing the outcome across
        every species present in ``config.species``.

    Raises:
        FelidAcquisitionError: If any download fails after all retries,
            with the root-cause exception class name and message
            preserved in the error text.
    """
    opener = opener or build_opener()
    reference_dir = Path(output_dir) if output_dir is not None else config.paths.reference_dir
    reference_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_felid_reference_manifest(reference_dir, assemblies=APPROVED_FELID_ASSEMBLIES)
    # Rebase destinations so files land directly in ``reference_dir`` rather than
    # the ``reference/`` subdirectory :func:`build_felid_reference_manifest`
    # creates. This keeps the acquire path aligned with
    # ``_resolve_fasta_path`` in the pipeline module.
    rebased = tuple(
        DownloadAsset(
            url=asset.url,
            destination=reference_dir / asset.destination.name,
            checksum=asset.checksum,
            checksum_name=asset.checksum_name,
            sample_id=asset.sample_id,
            kind=asset.kind,
        )
        for asset in manifest
    )
    pairs = _zip_entries_with_assets(config.species, rebased)

    results: list[DownloadResult] = []
    total_bytes_written = 0
    skipped_count = 0
    redownloaded_count = 0

    for entry, asset in pairs:
        _LOGGER.info(
            "start species=%s accession=%s destination=%s",
            entry.species_slug,
            entry.accession,
            asset.destination,
        )

        redownloaded = False
        if asset.destination.exists():
            actual_md5 = _compute_md5(asset.destination)
            if actual_md5 == asset.checksum:
                _LOGGER.info(
                    'skip species=%s accession=%s reason="checksum match" destination=%s',
                    entry.species_slug,
                    entry.accession,
                    asset.destination,
                )
                results.append(
                    DownloadResult(
                        path=asset.destination,
                        attempts=0,
                        resumed=False,
                        skipped_existing=True,
                        bytes_written=0,
                    )
                )
                skipped_count += 1
                continue
            _LOGGER.info(
                "verify_mismatch_redownload species=%s accession=%s "
                'reason="checksum mismatch; hash=%s expected=%s" destination=%s',
                entry.species_slug,
                entry.accession,
                actual_md5,
                asset.checksum,
                asset.destination,
            )
            asset.destination.unlink()
            redownloaded = True

        sleep_fn = sleep if sleep is not None else time.sleep
        try:
            result = download_with_retry(
                asset,
                opener=opener,
                retries=retries,
                sleep=sleep_fn,
            )
        except AcquisitionError as exc:
            root_cause = exc.__cause__ or exc
            _LOGGER.error(
                "failure species=%s accession=%s checksum=%s root_cause=%s: %s",
                entry.species_slug,
                entry.accession,
                asset.checksum,
                type(root_cause).__name__,
                root_cause,
            )
            raise FelidAcquisitionError(
                f"Failed to acquire felid reference for {entry.species} "
                f"(accession={entry.accession}, expected md5={asset.checksum}): "
                f"{type(root_cause).__name__}: {root_cause}"
            ) from exc

        if result.resumed:
            _LOGGER.info(
                "resume species=%s accession=%s attempts=%d",
                entry.species_slug,
                entry.accession,
                result.attempts,
            )
        _LOGGER.info(
            "download_finish species=%s accession=%s attempts=%d bytes=%d",
            entry.species_slug,
            entry.accession,
            result.attempts,
            result.bytes_written,
        )
        _LOGGER.info(
            "verify_ok species=%s accession=%s checksum=%s",
            entry.species_slug,
            entry.accession,
            asset.checksum,
        )
        results.append(result)
        total_bytes_written += result.bytes_written
        if redownloaded:
            redownloaded_count += 1

    return FelidAcquisitionSummary(
        per_species=tuple(results),
        total_bytes_written=total_bytes_written,
        skipped_count=skipped_count,
        redownloaded_count=redownloaded_count,
    )
