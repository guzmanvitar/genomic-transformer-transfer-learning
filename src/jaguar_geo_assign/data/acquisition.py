"""Feline data acquisition helpers: BioProject discovery and download primitives.

This module implements the discovery and acquisition stages for obtaining
feline genomic data from NCBI:

1. **Discovery** — :func:`fetch_bioproject_summary` queries the NCBI
   Entrez E-utilities API to retrieve and validate BioProject metadata for
   the approved 99 Lives Cat Genome Project.
2. **Acquisition** — :func:`build_feline_acquisition_manifest` constructs a
   typed download plan, and :func:`download_with_retry` executes idempotent,
   resumable, checksum-verified file transfers with exponential back-off.

Consensus-sequence construction (VCF → FASTA) lives in
:mod:`jaguar_geo_assign.data.consensus`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import OpenerDirector, Request, build_opener

DEFAULT_BIOPROJECT_ACCESSION = "PRJNA308208"
DEFAULT_REFERENCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/181/335/"
    "GCF_000181335.3_Felis_catus_9.0/GCF_000181335.3_Felis_catus_9.0_genomic.fna.gz"
)
BIOPROJECT_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    "esearch.fcgi?db=bioproject&retmode=json&term={accession}"
)
BIOPROJECT_SUMMARY_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    "esummary.fcgi?db=bioproject&retmode=json&id={project_id}"
)


class AcquisitionError(RuntimeError):
    """Base error for all feline data-acquisition failures.

    Serves as the base class for both download/discovery errors raised in
    this module and consensus-specific subclasses defined in
    :mod:`jaguar_geo_assign.data.consensus` (reference mismatch, contig
    mismatch, missing tool, malformed genotype), so callers can catch
    broad acquisition failures at this level while still distinguishing
    root causes when needed.
    """


@dataclass(frozen=True)
class BioProjectSummary:
    """Immutable snapshot of an NCBI BioProject retrieved via E-utilities.

    Attributes:
        accession: NCBI BioProject accession (e.g. ``"PRJNA308208"``).
        project_id: Numeric NCBI internal project identifier.
        title: Human-readable project title returned by the API.
        description: Extended project description (may be empty).
        submitter: Submitting organisation name (may be empty).
    """

    accession: str
    project_id: str
    title: str
    description: str
    submitter: str


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


@dataclass(frozen=True)
class AcquisitionManifest:
    """Complete typed download plan for a feline acquisition run.

    Groups the validated BioProject metadata together with the reference
    genome asset and all per-sample VCF assets into a single immutable
    manifest that :func:`download_with_retry` can execute.

    Attributes:
        project: Validated BioProject summary from NCBI.
        reference: Download descriptor for the reference genome FASTA.
        sample_vcfs: Ordered tuple of per-sample VCF download descriptors.
    """

    project: BioProjectSummary
    reference: DownloadAsset
    sample_vcfs: tuple[DownloadAsset, ...]


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


def fetch_bioproject_summary(
    accession: str = DEFAULT_BIOPROJECT_ACCESSION,
    opener: OpenerDirector | None = None,
) -> BioProjectSummary:
    """Query NCBI E-utilities for a BioProject and return a typed summary.

    Performs two sequential HTTP requests: an ``esearch`` lookup to resolve
    the accession to a numeric project ID, followed by an ``esummary`` call
    to retrieve the project metadata.

    Args:
        accession: NCBI BioProject accession string (e.g.
            ``"PRJNA308208"``).
        opener: Optional :class:`~urllib.request.OpenerDirector` for
            dependency injection in tests.  A default opener is built
            when ``None``.

    Returns:
        A :class:`BioProjectSummary` populated from the NCBI response.

    Raises:
        AcquisitionError: If the accession does not resolve to exactly
            one project.
    """
    opener = opener or build_opener()
    search_payload = _load_json(opener, BIOPROJECT_SEARCH_URL.format(accession=accession))
    id_list = search_payload["esearchresult"]["idlist"]
    if len(id_list) != 1:
        raise AcquisitionError(
            f"Expected exactly one BioProject for {accession}, found {len(id_list)}"
        )
    project_id = id_list[0]
    summary_payload = _load_json(opener, BIOPROJECT_SUMMARY_URL.format(project_id=project_id))
    record = summary_payload["result"][project_id]
    return BioProjectSummary(
        accession=record["project_acc"],
        project_id=project_id,
        title=record["project_title"],
        description=record.get("project_description", ""),
        submitter=record.get("submitter_organization", ""),
    )


def build_feline_acquisition_manifest(
    output_dir: str | Path,
    sample_vcf_urls: Mapping[str, str],
    *,
    sample_checksums: Mapping[str, str] | None = None,
    reference_url: str = DEFAULT_REFERENCE_URL,
    reference_checksum: str | None = None,
    project_accession: str = DEFAULT_BIOPROJECT_ACCESSION,
    opener: OpenerDirector | None = None,
) -> AcquisitionManifest:
    """Build a validated, typed download plan for feline genomic data.

    Validates the BioProject accession against the NCBI API, confirms the
    project title contains ``"99 Lives"``, and constructs
    :class:`DownloadAsset` descriptors for the reference genome and every
    per-sample VCF.  Sample VCFs are sorted by sample ID for deterministic
    ordering.

    Args:
        output_dir: Root directory under which ``reference/`` and ``vcf/``
            subdirectories will be created.
        sample_vcf_urls: Mapping of sample ID → VCF download URL.  Must
            contain at least one entry.
        sample_checksums: Optional mapping of sample ID → expected hex
            digest for integrity verification.
        reference_url: URL of the reference genome FASTA (gzipped).
        reference_checksum: Optional hex digest for the reference file.
        project_accession: NCBI BioProject accession to validate.
        opener: Optional :class:`~urllib.request.OpenerDirector` for
            dependency injection.

    Returns:
        An :class:`AcquisitionManifest` ready for download execution.

    Raises:
        ValueError: If *sample_vcf_urls* is empty.
        AcquisitionError: If the BioProject title does not match the
            expected 99 Lives project.
    """
    if not sample_vcf_urls:
        raise ValueError("sample_vcf_urls must contain at least one sample-specific VCF URL")
    project = fetch_bioproject_summary(project_accession, opener=opener)
    if "99 Lives" not in project.title:
        raise AcquisitionError(
            f"BioProject {project.accession} does not look like the approved 99 Lives target: "
            f"{project.title}"
        )

    output_root = Path(output_dir)
    reference_name = Path(urlparse(reference_url).path).name
    sample_checksums = sample_checksums or {}
    sample_vcfs = tuple(
        DownloadAsset(
            url=url,
            destination=output_root / "vcf" / _sample_destination_name(sample_id, url),
            checksum=sample_checksums.get(sample_id),
            sample_id=sample_id,
            kind="vcf",
        )
        for sample_id, url in sorted(sample_vcf_urls.items())
    )
    return AcquisitionManifest(
        project=project,
        reference=DownloadAsset(
            url=reference_url,
            destination=output_root / "reference" / reference_name,
            checksum=reference_checksum,
            kind="reference",
        ),
        sample_vcfs=sample_vcfs,
    )


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
                partial_path.unlink(missing_ok=True)
                raise AcquisitionError(
                    f"Checksum mismatch for {asset.url}; "
                    f"expected {asset.checksum_name}={asset.checksum}"
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
    digest = hashlib.new(checksum_name)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == checksum


def _load_json(opener: OpenerDirector, url: str) -> dict[str, object]:
    """Fetch a URL and parse the response body as JSON.

    Args:
        opener: HTTP opener to use for the request.
        url: URL to fetch.

    Returns:
        Parsed JSON payload as a dict.
    """
    with opener.open(url, timeout=30.0) as response:
        return json.loads(response.read().decode("utf-8"))


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
