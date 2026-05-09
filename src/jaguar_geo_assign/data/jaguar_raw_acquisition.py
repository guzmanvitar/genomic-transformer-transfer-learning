"""Operator-facing acquisition runtime for the two jaguar raw input files.

This module is the single entry point for downloading the pinned jaguar VCF
and location CSV. It wraps
:func:`jaguar_geo_assign.data.acquisition.download_with_retry` and is
intentionally simpler than the felid-foundation equivalent: there is no
per-species config, no registry filtering, and no config dependency — just a
destination directory and the two canonical assets.

The download is fully idempotent: invoking this function twice in a row
(barring disk corruption) writes zero bytes on the second call and returns
``skipped_count == 2``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import OpenerDirector, build_opener

from .acquisition import AcquisitionError, DownloadResult, download_with_retry
from .jaguar_raw_data import JAGUAR_RAW_ASSETS, JaguarRawAsset, build_jaguar_raw_manifest

_LOGGER = logging.getLogger(__name__)


class JaguarRawAcquisitionError(RuntimeError):
    """Raised when a jaguar raw file cannot be acquired or verified.

    Preserves the root-cause exception class name and message alongside the
    failing asset name so operators can distinguish transient network faults
    from deterministic integrity failures.
    """


@dataclass(frozen=True)
class JaguarRawAcquisitionSummary:
    """Typed summary of a jaguar raw data acquisition run.

    Attributes:
        per_file: One :class:`DownloadResult` per asset in
            :data:`JAGUAR_RAW_ASSETS` order.
        total_bytes_written: Sum of bytes written across all downloads
            (zero when every file was already present with the correct
            checksum).
        skipped_count: Number of files that already existed on disk with a
            matching SHA-256 and were therefore not re-downloaded.
    """

    per_file: tuple[DownloadResult, ...]
    total_bytes_written: int
    skipped_count: int


def acquire_jaguar_raw_data(
    output_dir: str | Path,
    *,
    opener: OpenerDirector | None = None,
    retries: int = 3,
    sleep: Callable[[float], None] | None = None,
) -> JaguarRawAcquisitionSummary:
    """Download the jaguar VCF and location CSV with integrity checks.

    Files already present on disk with a matching SHA-256 are skipped without
    a network request. Files with a mismatched checksum are deleted and
    re-downloaded. Structured ``INFO``-level logging records every ``start``,
    ``skip``, and ``download_finish`` event.

    Args:
        output_dir: Destination directory. Created (including parents) if it
            does not exist.
        opener: Optional :class:`~urllib.request.OpenerDirector` for
            dependency injection in unit tests.
        retries: Retry budget forwarded to
            :func:`~jaguar_geo_assign.data.acquisition.download_with_retry`.
        sleep: Optional sleep callable forwarded to
            :func:`~jaguar_geo_assign.data.acquisition.download_with_retry`.

    Returns:
        A :class:`JaguarRawAcquisitionSummary` describing the outcome.

    Raises:
        JaguarRawAcquisitionError: If any download fails after all retries,
            with the root-cause exception class name and message preserved.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    opener = opener or build_opener()
    sleep_fn = sleep if sleep is not None else time.sleep
    manifest = build_jaguar_raw_manifest(dest)

    results: list[DownloadResult] = []
    total_bytes_written = 0
    skipped_count = 0

    for asset_meta, asset in zip(JAGUAR_RAW_ASSETS, manifest, strict=False):
        _log_start(asset_meta)

        if asset.destination.exists() and _checksum_matches(asset.destination, asset):
            _LOGGER.info(
                'skip name=%s reason="checksum match" destination=%s',
                asset_meta.name,
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

        if asset.destination.exists():
            _LOGGER.info(
                "verify_mismatch_redownload name=%s destination=%s",
                asset_meta.name,
                asset.destination,
            )
            asset.destination.unlink()

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
                "failure name=%s root_cause=%s: %s",
                asset_meta.name,
                type(root_cause).__name__,
                root_cause,
            )
            raise JaguarRawAcquisitionError(
                f"Failed to acquire jaguar raw file '{asset_meta.name}': "
                f"{type(root_cause).__name__}: {root_cause}"
            ) from exc

        _LOGGER.info(
            "download_finish name=%s attempts=%d bytes=%d",
            asset_meta.name,
            result.attempts,
            result.bytes_written,
        )
        results.append(result)
        total_bytes_written += result.bytes_written

    return JaguarRawAcquisitionSummary(
        per_file=tuple(results),
        total_bytes_written=total_bytes_written,
        skipped_count=skipped_count,
    )


def _log_start(asset: JaguarRawAsset) -> None:
    _LOGGER.info("start name=%s url=%s", asset.name, asset.url)


def _checksum_matches(path: Path, asset: object) -> bool:
    import hashlib

    hasher = hashlib.new(asset.checksum_name)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == asset.checksum
