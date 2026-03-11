"""Feline acquisition and consensus-generation helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import OpenerDirector, Request, build_opener

DEFAULT_BIOPROJECT_ACCESSION = "PRJNA308208"
DEFAULT_REFERENCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/181/335/"
    "GCF_000181335.3_Felis_catus_9.0/GCF_000181335.3_Felis_catus_9.0_genomic.fna.gz"
)
EXPECTED_REFERENCE_TOKENS = ("GCF_000181335.3", "Felis_catus_9.0")
PASSING_FILTER_VALUES = frozenset({"PASS", "."})
BIOPROJECT_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    "esearch.fcgi?db=bioproject&retmode=json&term={accession}"
)
BIOPROJECT_SUMMARY_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    "esummary.fcgi?db=bioproject&retmode=json&id={project_id}"
)


class AcquisitionError(RuntimeError):
    """Base error for feline acquisition failures."""


class ReferenceMismatchError(AcquisitionError):
    """Raised when the FASTA and VCF do not agree on reference/build."""


class ContigMismatchError(AcquisitionError):
    """Raised when VCF contigs do not exist in the target FASTA."""


class MissingToolError(AcquisitionError):
    """Raised when a required external executable is unavailable."""


@dataclass(frozen=True)
class BioProjectSummary:
    accession: str
    project_id: str
    title: str
    description: str
    submitter: str


@dataclass(frozen=True)
class DownloadAsset:
    url: str
    destination: Path
    checksum: str | None = None
    checksum_name: str = "sha256"
    sample_id: str | None = None
    kind: str = "generic"


@dataclass(frozen=True)
class AcquisitionManifest:
    project: BioProjectSummary
    reference: DownloadAsset
    sample_vcfs: tuple[DownloadAsset, ...]


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    attempts: int
    resumed: bool
    skipped_existing: bool
    bytes_written: int


@dataclass(frozen=True)
class ConsensusDecision:
    action: str
    category: str
    replacement: str | None


@dataclass(frozen=True)
class ConsensusDiagnostics:
    sample_id: str
    total_records: int
    callable_records: int
    applied_variant_count: int
    masked_site_count: int
    filtered_or_nocall_count: int
    indel_count: int
    identical_to_reference_calls: int
    callable_fraction: float
    fraction_identical_to_reference_calls: float


@dataclass(frozen=True)
class ConsensusResult:
    sample_id: str
    output_fasta: Path
    diagnostics: ConsensusDiagnostics


@dataclass(frozen=True)
class _PreparedConsensus:
    sample_id: str
    filtered_vcf: Path
    mask_bed: Path | None
    diagnostics: ConsensusDiagnostics


def fetch_bioproject_summary(
    accession: str = DEFAULT_BIOPROJECT_ACCESSION,
    opener: OpenerDirector | None = None,
) -> BioProjectSummary:
    opener = opener or build_opener()
    search_payload = _load_json(opener, BIOPROJECT_SEARCH_URL.format(accession=accession))
    id_list = search_payload["esearchresult"]["idlist"]
    if len(id_list) != 1:
        raise AcquisitionError(f"Expected exactly one BioProject for {accession}, found {len(id_list)}")
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
    if not sample_vcf_urls:
        raise ValueError("sample_vcf_urls must contain at least one sample-specific VCF URL")
    project = fetch_bioproject_summary(project_accession, opener=opener)
    if "99 Lives" not in project.title:
        raise AcquisitionError(
            f"BioProject {project.accession} does not look like the approved 99 Lives target: {project.title}"
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
    opener = opener or build_opener()
    destination = asset.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.part")

    if destination.exists() and _checksum_matches(destination, asset.checksum, asset.checksum_name):
        return DownloadResult(destination, attempts=0, resumed=False, skipped_existing=True, bytes_written=0)
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
            if asset.checksum and not _checksum_matches(partial_path, asset.checksum, asset.checksum_name):
                partial_path.unlink(missing_ok=True)
                raise AcquisitionError(
                    f"Checksum mismatch for {asset.url}; expected {asset.checksum_name}={asset.checksum}"
                )
            partial_path.replace(destination)
            return DownloadResult(
                path=destination,
                attempts=attempt,
                resumed=resumed,
                skipped_existing=False,
                bytes_written=bytes_written,
            )
        except Exception as exc:  # pragma: no cover - exercised through tests
            last_error = exc
            if attempt >= retries:
                break
            sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise AcquisitionError(f"Failed to download {asset.url} after {retries} attempts") from last_error


def classify_consensus_site(
    ref: str,
    alts: Sequence[str],
    genotype: str | None,
    *,
    filter_value: str = "PASS",
) -> ConsensusDecision:
    if filter_value not in PASSING_FILTER_VALUES:
        return ConsensusDecision(action="mask", category="filtered", replacement=None)
    if not genotype or "." in genotype:
        return ConsensusDecision(action="mask", category="no_call", replacement=None)

    separator = "/" if "/" in genotype else "|"
    allele_tokens = genotype.split(separator)
    if not allele_tokens or any(token == "." for token in allele_tokens):
        return ConsensusDecision(action="mask", category="no_call", replacement=None)
    if len(set(allele_tokens)) != 1:
        category = "multiallelic_heterozygous" if len(set(allele_tokens)) > 1 and len(alts) > 1 else "heterozygous"
        return ConsensusDecision(action="mask", category=category, replacement=None)

    allele_index = int(allele_tokens[0])
    if allele_index == 0:
        return ConsensusDecision(action="reference", category="homozygous_reference", replacement=ref)
    if allele_index > len(alts):
        return ConsensusDecision(action="mask", category="invalid_alt_index", replacement=None)

    replacement = alts[allele_index - 1]
    category = "homozygous_alternate_indel" if len(replacement) != len(ref) else "homozygous_alternate"
    return ConsensusDecision(action="apply_alt", category=category, replacement=replacement)


def ensure_bcftools_available(executable: str = "bcftools") -> str:
    resolved = shutil.which(executable)
    if not resolved:
        raise MissingToolError(
            f"Required executable '{executable}' was not found on PATH; install bcftools before consensus generation."
        )
    return resolved


def generate_consensus_fasta(
    *,
    sample_id: str,
    reference_fasta: str | Path,
    sample_vcf: str | Path,
    output_fasta: str | Path,
    bcftools_executable: str = "bcftools",
    expected_reference_tokens: Sequence[str] = EXPECTED_REFERENCE_TOKENS,
) -> ConsensusResult:
    bcftools_path = ensure_bcftools_available(bcftools_executable)
    reference_path = Path(reference_fasta)
    vcf_path = Path(sample_vcf)
    output_path = Path(output_fasta)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"consensus-{sample_id}-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        prepared = _prepare_consensus(
            sample_id=sample_id,
            reference_fasta=reference_path,
            sample_vcf=vcf_path,
            work_dir=temp_dir,
            expected_reference_tokens=expected_reference_tokens,
        )
        command = [bcftools_path, "consensus", "-s", sample_id]
        if prepared.mask_bed is not None:
            command.extend(["-m", str(prepared.mask_bed)])
        command.append(str(prepared.filtered_vcf))
        with _open_maybe_gzip(reference_path) as reference_handle, output_path.open(
            "w", encoding="utf-8"
        ) as output_handle:
            completed = subprocess.run(
                command,
                stdin=reference_handle,
                stdout=output_handle,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
    if completed.returncode != 0:
        raise AcquisitionError(f"bcftools consensus failed for {sample_id}: {completed.stderr.strip()}")
    return ConsensusResult(sample_id=sample_id, output_fasta=output_path, diagnostics=prepared.diagnostics)


def generate_consensus_fastas(
    *,
    reference_fasta: str | Path,
    sample_vcfs: Mapping[str, str | Path],
    output_dir: str | Path,
    max_workers: int = 4,
    bcftools_executable: str = "bcftools",
    expected_reference_tokens: Sequence[str] = EXPECTED_REFERENCE_TOKENS,
) -> dict[str, ConsensusResult]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def _worker(sample_and_vcf: tuple[str, str | Path]) -> tuple[str, ConsensusResult]:
        sample_id, vcf_path = sample_and_vcf
        result = generate_consensus_fasta(
            sample_id=sample_id,
            reference_fasta=reference_fasta,
            sample_vcf=vcf_path,
            output_fasta=output_root / f"{sample_id}.fa",
            bcftools_executable=bcftools_executable,
            expected_reference_tokens=expected_reference_tokens,
        )
        return sample_id, result

    worker_count = max(1, min(max_workers, len(sample_vcfs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return dict(executor.map(_worker, sample_vcfs.items()))


def _prepare_consensus(
    *,
    sample_id: str,
    reference_fasta: Path,
    sample_vcf: Path,
    work_dir: Path,
    expected_reference_tokens: Sequence[str],
) -> _PreparedConsensus:
    contig_headers = _read_fasta_headers(reference_fasta)
    if not any(token in " ".join(contig_headers.values()) or token in reference_fasta.name for token in expected_reference_tokens):
        raise ReferenceMismatchError(
            f"Reference FASTA {reference_fasta} does not advertise any expected build token: {expected_reference_tokens}"
        )

    filtered_vcf = work_dir / f"{sample_id}.prepared.vcf"
    mask_bed = work_dir / f"{sample_id}.mask.bed"
    mask_ranges: list[tuple[str, int, int]] = []
    total_records = callable_records = applied_variant_count = 0
    filtered_or_nocall_count = indel_count = identical_to_reference_calls = 0

    with _open_maybe_gzip(sample_vcf) as source, filtered_vcf.open("w", encoding="utf-8") as sink:
        sample_index: int | None = None
        header_contigs: set[str] = set()
        vcf_reference = ""
        for line in source:
            if line.startswith("##"):
                if line.startswith("##contig=<ID="):
                    header_contigs.add(line.split("ID=", 1)[1].split(",", 1)[0].rstrip(">\n"))
                if line.startswith("##reference="):
                    vcf_reference = line.split("=", 1)[1].strip()
                sink.write(line)
                continue
            if line.startswith("#CHROM"):
                columns = line.rstrip("\n").split("\t")
                if sample_id not in columns[9:]:
                    raise AcquisitionError(f"Sample '{sample_id}' not found in VCF {sample_vcf}")
                sample_index = columns.index(sample_id)
                sink.write(line)
                continue

            if sample_index is None:
                raise AcquisitionError(f"VCF {sample_vcf} is missing a #CHROM header row")
            if vcf_reference and not any(token in vcf_reference for token in expected_reference_tokens):
                raise ReferenceMismatchError(
                    f"VCF {sample_vcf} declares reference '{vcf_reference}', expected one of {expected_reference_tokens}"
                )
            if header_contigs and not header_contigs.issubset(contig_headers.keys()):
                missing_contigs = sorted(header_contigs.difference(contig_headers.keys()))
                raise ContigMismatchError(
                    f"VCF {sample_vcf} references contigs absent from {reference_fasta}: {missing_contigs[:5]}"
                )

            total_records += 1
            fields = line.rstrip("\n").split("\t")
            chrom, pos_str, _, ref, alt_field, _, filter_value, _, format_field = fields[:9]
            if chrom not in contig_headers:
                raise ContigMismatchError(f"Contig '{chrom}' from {sample_vcf} is absent from {reference_fasta}")
            alts = alt_field.split(",") if alt_field else []
            sample_format = dict(zip(format_field.split(":"), fields[sample_index].split(":"), strict=False))
            decision = classify_consensus_site(ref, alts, sample_format.get("GT"), filter_value=filter_value)
            if decision.category in {"filtered", "no_call"}:
                filtered_or_nocall_count += 1
            if decision.action != "mask":
                callable_records += 1
            if decision.action == "reference":
                identical_to_reference_calls += 1
            elif decision.action == "apply_alt":
                applied_variant_count += 1
                if decision.category.endswith("indel"):
                    indel_count += 1
                sink.write(line)
                continue
            start = int(pos_str) - 1
            mask_ranges.append((chrom, start, start + len(ref)))
        if not total_records:
            raise AcquisitionError(f"VCF {sample_vcf} does not contain any records for sample {sample_id}")

    if mask_ranges:
        with mask_bed.open("w", encoding="utf-8") as handle:
            for chrom, start, end in mask_ranges:
                handle.write(f"{chrom}\t{start}\t{end}\n")
        mask_path: Path | None = mask_bed
    else:
        mask_path = None

    masked_site_count = len(mask_ranges)
    diagnostics = ConsensusDiagnostics(
        sample_id=sample_id,
        total_records=total_records,
        callable_records=callable_records,
        applied_variant_count=applied_variant_count,
        masked_site_count=masked_site_count,
        filtered_or_nocall_count=filtered_or_nocall_count,
        indel_count=indel_count,
        identical_to_reference_calls=identical_to_reference_calls,
        callable_fraction=callable_records / total_records,
        fraction_identical_to_reference_calls=(
            identical_to_reference_calls / total_records if total_records else 0.0
        ),
    )
    return _PreparedConsensus(
        sample_id=sample_id,
        filtered_vcf=filtered_vcf,
        mask_bed=mask_path,
        diagnostics=diagnostics,
    )


def _download_once(
    *,
    opener: OpenerDirector,
    url: str,
    partial_path: Path,
    timeout_seconds: float,
    chunk_size: int,
) -> int:
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
    if checksum is None:
        return path.exists()
    digest = hashlib.new(checksum_name)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == checksum


def _load_json(opener: OpenerDirector, url: str) -> dict[str, object]:
    with opener.open(url, timeout=30.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_fasta_headers(reference_fasta: Path) -> dict[str, str]:
    headers: dict[str, str] = {}
    with _open_maybe_gzip(reference_fasta) as handle:
        for line in handle:
            if line.startswith(">"):
                full_header = line[1:].strip()
                headers[full_header.split()[0]] = full_header
    if not headers:
        raise AcquisitionError(f"Reference FASTA {reference_fasta} did not contain any contig headers")
    return headers


def _open_maybe_gzip(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _sample_destination_name(sample_id: str, url: str) -> str:
    suffix = "".join(Path(urlparse(url).path).suffixes) or ".vcf"
    return f"{sample_id}{suffix}"