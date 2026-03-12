"""Feline acquisition and consensus-generation helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
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


class MalformedGenotypeError(AcquisitionError):
    """Raised when a VCF GT field contains malformed non-numeric allele tokens."""


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
class ConsensusMaskSpan:
    contig: str
    start: int
    end: int
    category: str


@dataclass(frozen=True)
class ConsensusResult:
    sample_id: str
    output_fasta: Path
    diagnostics: ConsensusDiagnostics
    mask_spans: tuple[ConsensusMaskSpan, ...] = ()


@dataclass(frozen=True)
class _PreparedConsensus:
    sample_id: str
    filtered_vcf: Path
    mask_bed: Path | None
    diagnostics: ConsensusDiagnostics
    mask_spans: tuple[ConsensusMaskSpan, ...]


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
        except AcquisitionError:
            raise
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
    sample_id: str | None = None,
    contig: str | None = None,
    position: int | None = None,
    vcf_path: str | Path | None = None,
) -> ConsensusDecision:
    normalized_alts = _normalize_alt_alleles(alts)
    allele_tokens = _validated_gt_tokens(
        genotype,
        sample_id=sample_id,
        contig=contig,
        position=position,
        vcf_path=vcf_path,
    )
    if filter_value not in PASSING_FILTER_VALUES:
        return ConsensusDecision(action="mask", category="filtered", replacement=None)
    if allele_tokens is None:
        return ConsensusDecision(action="mask", category="no_call", replacement=None)
    if len(set(allele_tokens)) != 1:
        category = "multiallelic" if len(normalized_alts) > 1 else "heterozygous"
        return ConsensusDecision(action="mask", category=category, replacement=None)

    allele_index = int(allele_tokens[0])
    if allele_index == 0:
        return ConsensusDecision(action="reference", category="homozygous_reference", replacement=ref)
    if allele_index > len(normalized_alts):
        return ConsensusDecision(action="mask", category="invalid_alt_index", replacement=None)
    if len(normalized_alts) > 1:
        return ConsensusDecision(action="mask", category="multiallelic", replacement=None)

    replacement = normalized_alts[allele_index - 1]
    if len(replacement) != len(ref):
        return ConsensusDecision(action="mask", category="indel", replacement=None)
    category = "homozygous_alternate"
    return ConsensusDecision(action="apply_alt", category=category, replacement=replacement)


def _normalize_alt_alleles(alts: Sequence[str]) -> tuple[str, ...]:
    if len(alts) == 1 and alts[0] == ".":
        return ()
    return tuple(alts)


def _validated_gt_tokens(
    genotype: str | None,
    *,
    sample_id: str | None,
    contig: str | None,
    position: int | None,
    vcf_path: str | Path | None,
) -> list[str] | None:
    if not genotype:
        return None
    separator = "/" if "/" in genotype else "|"
    allele_tokens = genotype.split(separator)
    if not allele_tokens or any(token == "." for token in allele_tokens):
        return None

    malformed_tokens = sorted({token for token in allele_tokens if not token.isdigit()})
    if malformed_tokens:
        sample_fragment = f" for sample '{sample_id}'" if sample_id else ""
        locus_fragment = f" at {contig}:{position}" if contig and position is not None else ""
        vcf_fragment = f" in VCF {vcf_path}" if vcf_path is not None else ""
        raise MalformedGenotypeError(
            "Malformed non-numeric GT token(s) "
            f"{malformed_tokens} in GT='{genotype}'{sample_fragment}{locus_fragment}{vcf_fragment}; "
            "expected numeric allele indices or '.' no-call markers."
        )
    return allele_tokens


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
            with subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=output_handle,
                stderr=subprocess.PIPE,
                text=True,
            ) as completed:
                stdin_handle = completed.stdin
                if stdin_handle is None:
                    raise AcquisitionError(f"bcftools consensus did not expose stdin for {sample_id}")

                def _write_reference_to_stdin() -> None:
                    try:
                        shutil.copyfileobj(reference_handle, stdin_handle)
                    except BrokenPipeError:
                        pass
                    finally:
                        try:
                            stdin_handle.close()
                        except BrokenPipeError:
                            pass

                with ThreadPoolExecutor(max_workers=1) as stdin_writer:
                    stdin_future = stdin_writer.submit(_write_reference_to_stdin)
                    stderr_text = completed.stderr.read() if completed.stderr is not None else ""
                    return_code = completed.wait()
                    stdin_future.result()
    if return_code != 0:
        raise AcquisitionError(f"bcftools consensus failed for {sample_id}: {stderr_text.strip()}")
    return ConsensusResult(
        sample_id=sample_id,
        output_fasta=output_path,
        diagnostics=prepared.diagnostics,
        mask_spans=prepared.mask_spans,
    )


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
    reference_evidence = " ".join((reference_fasta.name, *contig_headers.values()))
    if not _matches_expected_reference_build(reference_evidence, expected_reference_tokens):
        raise ReferenceMismatchError(
            "Reference FASTA "
            f"{reference_fasta} does not canonically match expected build evidence {expected_reference_tokens}"
        )

    filtered_vcf = work_dir / f"{sample_id}.prepared.vcf"
    mask_bed = work_dir / f"{sample_id}.mask.bed"
    mask_spans: list[ConsensusMaskSpan] = []
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
                if not vcf_reference:
                    raise ReferenceMismatchError(
                        f"VCF {sample_vcf} is missing explicit reference/build metadata in a ##reference header"
                    )
                if not _matches_expected_reference_build(vcf_reference, expected_reference_tokens):
                    raise ReferenceMismatchError(
                        "VCF "
                        f"{sample_vcf} declares reference '{vcf_reference}', which does not canonically match "
                        f"expected build evidence {expected_reference_tokens}"
                    )
                if header_contigs and not header_contigs.issubset(contig_headers.keys()):
                    missing_contigs = sorted(header_contigs.difference(contig_headers.keys()))
                    raise ContigMismatchError(
                        f"VCF {sample_vcf} references contigs absent from {reference_fasta}: {missing_contigs[:5]}"
                    )
                sample_index = columns.index(sample_id)
                sink.write(line)
                continue

            if sample_index is None:
                raise AcquisitionError(f"VCF {sample_vcf} is missing a #CHROM header row")

            total_records += 1
            fields = line.rstrip("\n").split("\t")
            chrom, pos_str, _, ref, alt_field, _, filter_value, _, format_field = fields[:9]
            if chrom not in contig_headers:
                raise ContigMismatchError(f"Contig '{chrom}' from {sample_vcf} is absent from {reference_fasta}")
            alts = _normalize_alt_alleles(alt_field.split(",") if alt_field else [])
            sample_format = dict(zip(format_field.split(":"), fields[sample_index].split(":"), strict=False))
            decision = classify_consensus_site(
                ref,
                alts,
                sample_format.get("GT"),
                filter_value=filter_value,
                sample_id=sample_id,
                contig=chrom,
                position=int(pos_str),
                vcf_path=sample_vcf,
            )
            if decision.category in {"filtered", "no_call"}:
                filtered_or_nocall_count += 1
            if decision.category == "indel":
                indel_count += 1
            if decision.action != "mask":
                callable_records += 1
            if decision.action == "reference":
                identical_to_reference_calls += 1
                continue
            if decision.action == "apply_alt":
                applied_variant_count += 1
                sink.write(line)
                continue
            start = int(pos_str) - 1
            mask_spans.append(
                ConsensusMaskSpan(
                    contig=chrom,
                    start=start,
                    end=start + len(ref),
                    category=decision.category,
                )
            )
        if not total_records:
            raise AcquisitionError(f"VCF {sample_vcf} does not contain any records for sample {sample_id}")

    if mask_spans:
        with mask_bed.open("w", encoding="utf-8") as handle:
            for span in mask_spans:
                handle.write(f"{span.contig}\t{span.start}\t{span.end}\n")
        mask_path: Path | None = mask_bed
    else:
        mask_path = None

    masked_site_count = len(mask_spans)
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
        mask_spans=tuple(mask_spans),
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


def _matches_expected_reference_build(evidence: str, expected_reference_tokens: Sequence[str]) -> bool:
    canonical_evidence = _canonicalize_reference_evidence(evidence)
    return all(
        _canonicalize_reference_evidence(token) in canonical_evidence for token in expected_reference_tokens
    )


def _canonicalize_reference_evidence(evidence: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", evidence.lower()).strip("_")


def _open_maybe_gzip(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _sample_destination_name(sample_id: str, url: str) -> str:
    suffix = "".join(Path(urlparse(url).path).suffixes) or ".vcf"
    return f"{sample_id}{suffix}"