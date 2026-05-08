"""Download helpers for remote genomic assets.

The active code path retains only the resumable, checksum-aware transfer
primitives used by felid-foundation assembly acquisition.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import OpenerDirector, Request, build_opener

_LOGGER = logging.getLogger(__name__)


class AcquisitionError(RuntimeError):
    """Base error for all download and transfer failures in this module."""


@dataclass(frozen=True)
class DownloadAsset:
    """Immutable descriptor for a single file to download.

    Attributes:
        url: Fully-qualified HTTP/HTTPS URL for the remote resource.
        destination: Local filesystem path where the file will be written.
        checksum: Expected hex-encoded digest, or ``None`` to skip
            integrity verification.
        checksum_name: Hash algorithm name accepted by :func:`hashlib.new`
            (default ``"sha256"``).
        sample_id: Optional sample identifier when the asset is a
            per-sample VCF.
        kind: Free-form tag distinguishing asset types
            (``"reference"``, ``"vcf"``, ``"generic"``).
    """

    url: str
    destination: Path
    checksum: str | None = None
    checksum_name: str = "sha256"
    sample_id: str | None = None
    kind: str = "generic"
    mirror_url: str | None = None
    expected_size: int | None = None


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a single :func:`download_with_retry` invocation.

    Attributes:
        path: Final filesystem path of the successfully downloaded file.
        attempts: Number of HTTP attempts consumed (``0`` when *skipped_existing*).
        resumed: ``True`` if the successful attempt resumed a partial
            ``.part`` file via an HTTP ``Range`` header.
        skipped_existing: ``True`` if the file already existed on disk and
            its checksum matched — no network request was made.
        bytes_written: Total bytes written to disk (``0`` when skipped).
    """

    path: Path
    attempts: int
    resumed: bool
    skipped_existing: bool
    bytes_written: int


def download_with_retry(
    asset: DownloadAsset,
    *,
    retries: int = 3,
    timeout_seconds: float = 30.0,
    chunk_size: int = 1024 * 1024,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    opener: OpenerDirector | None = None,
) -> DownloadResult:
    """Download a single asset with idempotency, resume, and retry support.

    Implements a three-layer resilience strategy:

    1. **Skip** — If the destination file already exists and its checksum
       matches, the download is skipped entirely (zero network I/O).
    2. **Resume** — If a ``.part`` file exists from a previous interrupted
       attempt, an HTTP ``Range`` header is sent to resume from the last
       byte.  If the server responds with ``200`` instead of ``206``
       (i.e. it does not support range requests), the partial file is
       deleted and the download restarts from scratch.
    3. **Retry** — Transient failures trigger exponential back-off
       (``backoff_seconds * 2^(attempt-1)``) up to *retries* attempts.
       :class:`AcquisitionError` (e.g. checksum mismatch) is **not**
       retried — it is raised immediately.

    .. warning::

       Checksum verification happens **after** the full file is written
       to the ``.part`` path.  If the checksum fails, the partial file is
       deleted and the error is raised immediately (no retry), because a
       checksum mismatch indicates data corruption rather than a transient
       network fault.

    Args:
        asset: Typed download descriptor specifying URL, destination path,
            and optional checksum.
        retries: Maximum number of download attempts.
        timeout_seconds: Per-request socket timeout in seconds.
        chunk_size: Bytes per read chunk (default 1 MiB).
        backoff_seconds: Base delay for exponential back-off between retries.
        sleep: Callable used for back-off delays; injectable for tests.
        opener: Optional :class:`~urllib.request.OpenerDirector` for
            dependency injection.

    Returns:
        A :class:`DownloadResult` describing the outcome.

    Raises:
        AcquisitionError: On checksum mismatch (immediate, not retried) or
            after all retry attempts are exhausted.
    """
    opener = opener or build_opener()

    if asset.expected_size is not None:
        try:
            head_req = Request(asset.url, method="HEAD")
            with opener.open(head_req, timeout=timeout_seconds) as head_resp:
                size_header = head_resp.headers.get("Content-Length") or head_resp.headers.get(
                    "x-linked-size"
                )
                if size_header is not None:
                    observed_size = int(size_header)
                    if observed_size != asset.expected_size:
                        raise AcquisitionError(
                            f"Size mismatch during HEAD pre-flight for {asset.url}: "
                            f"expected {asset.expected_size}, got {observed_size}"
                        )
        except AcquisitionError:
            raise
        except Exception as exc:
            _LOGGER.warning(
                "HEAD pre-flight failed for %s, falling through to download: %s", asset.url, exc
            )

    destination = asset.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.part")

    if destination.exists() and _checksum_matches(destination, asset.checksum, asset.checksum_name):
        return DownloadResult(
            destination, attempts=0, resumed=False, skipped_existing=True, bytes_written=0
        )
    if destination.exists():
        destination.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resumed = partial_path.exists() and partial_path.stat().st_size > 0
            bytes_written = _download_once(
                opener=opener,
                url=asset.url,
                partial_path=partial_path,
                timeout_seconds=timeout_seconds,
                chunk_size=chunk_size,
            )
            if asset.checksum and not _checksum_matches(
                partial_path, asset.checksum, asset.checksum_name
            ):
                observed_checksum = _get_digest(partial_path, asset.checksum_name)
                partial_path.unlink(missing_ok=True)

                if asset.mirror_url:
                    _LOGGER.warning(
                        "Checksum mismatch for primary %s "
                        "(expected %s, got %s); "
                        "falling back to mirror %s",
                        asset.url,
                        asset.checksum,
                        observed_checksum,
                        asset.mirror_url,
                    )
                    bytes_written = _download_once(
                        opener=opener,
                        url=asset.mirror_url,
                        partial_path=partial_path,
                        timeout_seconds=timeout_seconds,
                        chunk_size=chunk_size,
                    )
                    if not _checksum_matches(partial_path, asset.checksum, asset.checksum_name):
                        mirror_observed = _get_digest(partial_path, asset.checksum_name)
                        partial_path.unlink(missing_ok=True)
                        raise AcquisitionError(
                            f"Checksum mismatch for both primary and mirror; "
                            f"primary: {asset.url} (got {observed_checksum}), "
                            f"mirror: {asset.mirror_url} (got {mirror_observed}); "
                            f"expected {asset.checksum_name}={asset.checksum}"
                        )
                else:
                    raise AcquisitionError(
                        f"Checksum mismatch for {asset.url}; "
                        f"expected {asset.checksum_name}={asset.checksum} (got {observed_checksum})"
                    )
            partial_path.replace(destination)
            return DownloadResult(
                path=destination,
                attempts=attempt,
                resumed=resumed,
                skipped_existing=False,
                bytes_written=bytes_written,
            )
        except AcquisitionError:
            raise
        except Exception as exc:  # pragma: no cover - exercised through tests
            last_error = exc
            if attempt >= retries:
                break
            sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise AcquisitionError(
        f"Failed to download {asset.url} after {retries} attempts"
    ) from last_error


def _download_once(
    *,
    opener: OpenerDirector,
    url: str,
    partial_path: Path,
    timeout_seconds: float,
    chunk_size: int,
) -> int:
    """Execute a single HTTP download attempt with optional resume.

    If a partial file exists, sends an HTTP ``Range`` header to resume.
    If the server responds with ``200`` instead of ``206`` (range not
    supported), the partial file is deleted and this function recurses
    once to restart from scratch.

    Args:
        opener: HTTP opener to use for the request.
        url: URL to download.
        partial_path: Local path for the in-progress ``.part`` file.
        timeout_seconds: Socket timeout for the HTTP request.
        chunk_size: Bytes per read chunk.

    Returns:
        Total file size in bytes after the download completes.
    """
    partial_size = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={partial_size}-"} if partial_size else {}
    request = Request(url, headers=headers)
    response = opener.open(request, timeout=timeout_seconds)
    status = getattr(response, "status", getattr(response, "code", 200))
    if partial_size and status != 206:
        partial_path.unlink(missing_ok=True)
        return _download_once(
            opener=opener,
            url=url,
            partial_path=partial_path,
            timeout_seconds=timeout_seconds,
            chunk_size=chunk_size,
        )
    mode = "ab" if partial_size and status == 206 else "wb"
    with partial_path.open(mode) as handle:
        for chunk in iter(lambda: response.read(chunk_size), b""):
            handle.write(chunk)
    return partial_path.stat().st_size


def _get_digest(path: Path, checksum_name: str) -> str:
    """Compute the streaming hex digest of path using hashlib.new(checksum_name)."""
    digest = hashlib.new(checksum_name)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_matches(path: Path, checksum: str | None, checksum_name: str) -> bool:
    """Check whether a file's digest matches an expected checksum.

    When *checksum* is ``None``, simply returns whether the file exists
    (i.e. no integrity verification is performed).

    Args:
        path: Path to the file to verify.
        checksum: Expected hex-encoded digest, or ``None`` to skip.
        checksum_name: Hash algorithm name (e.g. ``"sha256"``).

    Returns:
        ``True`` if the checksum matches or was not requested (file
        exists), ``False`` otherwise.
    """
    if checksum is None:
        return path.exists()
    return _get_digest(path, checksum_name) == checksum


def _sample_destination_name(sample_id: str, url: str) -> str:
    """Derive a local filename for a sample VCF from its ID and URL.

    Extracts the file extension(s) from the URL path (e.g. ``.vcf.gz``)
    and prepends the sample ID.  Falls back to ``.vcf`` if the URL has
    no recognisable suffix.

    Args:
        sample_id: Sample identifier used as the filename stem.
        url: Remote URL from which the suffix is extracted.

    Returns:
        A filename string like ``"sample_01.vcf.gz"``.
    """
    suffix = "".join(Path(urlparse(url).path).suffixes) or ".vcf"
    return f"{sample_id}{suffix}"
