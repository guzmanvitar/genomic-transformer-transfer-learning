"""VCF-to-consensus-sequence construction for felid genomes.

Retained for downstream jaguar-assignment workflows that require per-sample
consensus calling; not wired into the current felid foundation pretraining
path, which operates on reference FASTAs directly.

This module implements the consensus stage of the feline data workflow:
:func:`classify_consensus_site` encodes a deterministic per-site decision
tree, and :func:`generate_consensus_fasta` /
:func:`generate_consensus_fastas` orchestrate ``bcftools consensus`` to
produce per-sample FASTA files from filtered VCF records and masked
intervals.

.. warning::

   The ordering of branches inside :func:`classify_consensus_site` is load-
   bearing: filter status is evaluated before genotype parsing, and
   multi-allelic checks precede allele-index lookup.  Reordering will
   silently change which sites are masked vs. applied.
"""

from __future__ import annotations

import gzip
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .acquisition import AcquisitionError

EXPECTED_REFERENCE_TOKENS = ("GCF_000181335.3", "Felis_catus_9.0")
PASSING_FILTER_VALUES = frozenset({"PASS", "."})


class ReferenceMismatchError(AcquisitionError):
    """The FASTA reference and VCF ``##reference`` header disagree on build.

    Raised during consensus preparation when either the reference FASTA file
    name or the VCF ``##reference`` metadata line does not contain the
    expected canonical build tokens (e.g. ``GCF_000181335.3``,
    ``Felis_catus_9.0``).  This prevents silent genome-build mismatches from
    corrupting consensus sequences.
    """


class ContigMismatchError(AcquisitionError):
    """A VCF contig identifier is absent from the reference FASTA.

    Raised when a ``##contig=<ID=…>`` header or a data-record chromosome
    field references a contig that does not appear in the reference FASTA
    header lines.  This typically indicates that the VCF was called against
    a different assembly or that contig naming conventions differ.
    """


class MissingToolError(AcquisitionError):
    """A required external CLI tool (e.g. ``bcftools``) is not on ``$PATH``.

    Raised by :func:`ensure_bcftools_available` before any subprocess
    invocation so that the user receives a clear diagnostic instead of an
    opaque ``FileNotFoundError`` from the operating system.
    """


class MalformedGenotypeError(AcquisitionError):
    """A VCF ``GT`` field contains non-numeric, non-missing allele tokens.

    Raised by :func:`_validated_gt_tokens` when allele tokens (after
    splitting on ``/`` or ``|``) are neither valid digit strings nor the
    VCF-standard ``.`` no-call marker.  The error message includes the
    sample ID, locus, and VCF path for rapid triage.
    """


@dataclass(frozen=True)
class ConsensusDecision:
    """Result of the deterministic per-site consensus decision tree.

    Produced by :func:`classify_consensus_site` for every VCF data record.

    Attributes:
        action: One of ``"reference"`` (keep ref), ``"apply_alt"`` (emit
            alternate allele), or ``"mask"`` (mask the site with ``N``).
        category: Fine-grained reason string describing why this action
            was chosen (e.g. ``"homozygous_alternate"``, ``"filtered"``,
            ``"heterozygous"``, ``"indel"``, ``"multiallelic"``).
        replacement: The allele string to emit, or ``None`` when the site
            is masked.
    """

    action: str
    category: str
    replacement: str | None


@dataclass(frozen=True)
class ConsensusDiagnostics:
    """Aggregate quality metrics for a single sample's consensus run.

    Attributes:
        sample_id: Identifier of the sample these diagnostics describe.
        total_records: Number of VCF data records processed.
        callable_records: Records that were *not* masked (action ≠ "mask").
        applied_variant_count: Records where an alternate allele was applied.
        masked_site_count: Records where the site was masked with ``N``.
        filtered_or_nocall_count: Records masked because of a non-PASS
            FILTER value or a missing/no-call genotype.
        indel_count: Records masked because REF and ALT lengths differ.
        identical_to_reference_calls: Homozygous-reference records
            (action = "reference").
        callable_fraction: ``callable_records / total_records``.
        fraction_identical_to_reference_calls: ``identical_to_reference_calls
            / total_records`` (``0.0`` when *total_records* is zero).
    """

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
    """A single masked genomic interval written to the BED mask file.

    Attributes:
        contig: Chromosome / contig name (BED column 0).
        start: Zero-based inclusive start coordinate (BED column 1).
        end: Zero-based exclusive end coordinate (BED column 2).
        category: The :attr:`ConsensusDecision.category` that triggered
            masking (e.g. ``"heterozygous"``, ``"indel"``).
    """

    contig: str
    start: int
    end: int
    category: str


@dataclass(frozen=True)
class ConsensusResult:
    """Final output of :func:`generate_consensus_fasta` for one sample.

    Attributes:
        sample_id: Identifier of the processed sample.
        output_fasta: Path to the written consensus FASTA file.
        diagnostics: Aggregate quality metrics for the run.
        mask_spans: Genomic intervals that were masked (may be empty).
    """

    sample_id: str
    output_fasta: Path
    diagnostics: ConsensusDiagnostics
    mask_spans: tuple[ConsensusMaskSpan, ...] = ()


@dataclass(frozen=True)
class _PreparedConsensus:
    """Internal intermediate result from VCF filtering before ``bcftools``.

    Holds the paths to the filtered VCF and optional mask BED file together
    with the diagnostics computed during the filtering pass, ready for
    ``bcftools consensus`` to consume.

    Attributes:
        sample_id: Identifier of the sample being processed.
        filtered_vcf: Path to the VCF containing only ``apply_alt`` records.
        mask_bed: Path to the BED file of masked intervals, or ``None``
            if no sites were masked.
        diagnostics: Aggregate quality metrics from the filtering pass.
        mask_spans: Individual masked intervals for downstream reporting.
    """

    sample_id: str
    filtered_vcf: Path
    mask_bed: Path | None
    diagnostics: ConsensusDiagnostics
    mask_spans: tuple[ConsensusMaskSpan, ...]


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
    """Classify a single VCF site into a deterministic consensus action.

    Implements a strict decision tree whose **branch ordering is
    load-bearing** — reordering will silently change which sites are
    masked vs. applied.  The evaluation order is:

    1. **Filter gate** — if *filter_value* is not in
       :data:`PASSING_FILTER_VALUES` (``{"PASS", "."}``), the site is
       masked as ``"filtered"`` regardless of genotype content.
    2. **No-call gate** — if the genotype is missing or contains any
       ``.`` no-call token, the site is masked as ``"no_call"``.
    3. **Heterozygosity / multi-allelic gate** — if the genotype allele
       tokens are not all identical, the site is masked.  The category is
       ``"multiallelic"`` when >1 ALT allele exists, otherwise
       ``"heterozygous"``.
    4. **Homozygous reference** — allele index ``0`` ⇒ action
       ``"reference"``, replacement is the REF allele.
    5. **Invalid allele index** — index exceeds the ALT list ⇒ masked as
       ``"invalid_alt_index"``.
    6. **Multi-allelic with single homozygous ALT** — even if the
       genotype is homozygous for one ALT, ≥2 ALT alleles triggers
       masking as ``"multiallelic"`` for safety.
    7. **Indel** — ``len(ALT) != len(REF)`` ⇒ masked as ``"indel"``.
    8. **Homozygous alternate** — action ``"apply_alt"``, replacement is
       the selected ALT allele.

    Args:
        ref: VCF ``REF`` column value.
        alts: VCF ``ALT`` column values (pre-split on ``,``).  A
            single-element list of ``"."`` represents a monomorphic
            reference site and is normalised to an empty tuple by
            :func:`_normalize_alt_alleles`.
        genotype: The sample's ``GT`` format field (e.g. ``"0/1"``,
            ``"1|1"``), or ``None`` if absent.
        filter_value: VCF ``FILTER`` column value for this record.
        sample_id: Optional sample identifier for error messages.
        contig: Optional contig name for error messages.
        position: Optional 1-based position for error messages.
        vcf_path: Optional VCF file path for error messages.

    Returns:
        A :class:`ConsensusDecision` encoding the action, category, and
        replacement allele (or ``None`` for masked sites).

    Raises:
        MalformedGenotypeError: If the genotype contains non-numeric,
            non-missing allele tokens (propagated from
            :func:`_validated_gt_tokens`).
    """
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
        return ConsensusDecision(
            action="reference", category="homozygous_reference", replacement=ref
        )
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
    """Normalise VCF ALT alleles, collapsing the monomorphic ``"."`` sentinel.

    The VCF specification uses a single ``"."`` in the ALT column to
    indicate a monomorphic reference site (no alternate alleles).  This
    function converts that sentinel into an empty tuple so that downstream
    length checks (e.g. ``len(normalized_alts) > 1`` for multi-allelic
    detection) work uniformly without special-casing.

    .. warning::

       Only a **single-element** list containing exactly ``"."`` is
       collapsed.  A multi-element list like ``["A", "."]`` is passed
       through unchanged — the ``"."`` would be treated as a literal
       allele string, which is intentional: such records are malformed
       and will be caught by downstream allele-index validation.

    Args:
        alts: Raw ALT column values, already split on ``,``.

    Returns:
        An empty tuple for monomorphic sites, or the input converted to a
        tuple otherwise.
    """
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
    """Parse and validate a VCF ``GT`` field into numeric allele tokens.

    Splits the genotype string on ``/`` (unphased) or ``|`` (phased).
    The separator detection is order-dependent: ``/`` is checked first,
    so a genotype like ``"0/1"`` is always treated as unphased even if
    ``|`` appears elsewhere.  This matches the VCF 4.x specification
    where ``/`` and ``|`` are mutually exclusive separators within a
    single GT field.

    The function distinguishes three outcomes:

    * **None** — returned when the genotype is empty/falsy, or contains
      any ``.`` no-call token (VCF-standard missing allele marker).
    * **list[str]** — returned when all tokens are valid digit strings.
    * **MalformedGenotypeError** — raised when tokens are non-numeric
      *and* non-missing, which indicates a corrupted or non-standard VCF.

    Args:
        genotype: Raw GT field value (e.g. ``"0/1"``, ``"1|1"``,
            ``"./."``) or ``None``.
        sample_id: Optional sample ID for diagnostic error messages.
        contig: Optional contig name for diagnostic error messages.
        position: Optional 1-based position for diagnostic error messages.
        vcf_path: Optional VCF path for diagnostic error messages.

    Returns:
        A list of digit-string allele tokens, or ``None`` for missing /
        no-call genotypes.

    Raises:
        MalformedGenotypeError: If any allele token is neither a digit
            string nor the ``"."`` no-call marker.
    """
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
            f"{malformed_tokens} in GT='{genotype}'"
            f"{sample_fragment}{locus_fragment}{vcf_fragment}; "
            "expected numeric allele indices or '.' no-call markers."
        )
    return allele_tokens


def _raise_malformed_vcf_record(
    *,
    sample_vcf: Path,
    sample_id: str,
    line_number: int,
    raw_record: str,
    observed_columns: int,
    expected_min_columns: int,
) -> None:
    """Raise a diagnostic :class:`AcquisitionError` for a malformed VCF record.

    Constructs a human-readable error message that includes the file path,
    sample ID, line number, expected vs. observed column counts, and a
    truncated preview of the offending record (capped at 200 characters).

    This is extracted as a helper to keep the main VCF parsing loop in
    :func:`_prepare_consensus` readable.

    Args:
        sample_vcf: Path to the VCF file being parsed.
        sample_id: Sample identifier for the error message.
        line_number: 1-based line number of the malformed record.
        raw_record: The raw tab-delimited line (stripped of newline).
        observed_columns: Actual number of tab-delimited fields found.
        expected_min_columns: Minimum columns required (at least 9, or
            the sample column index + 1).

    Raises:
        AcquisitionError: Always raised with a descriptive message.
    """
    record_preview = raw_record if raw_record else "<blank line>"
    if len(record_preview) > 200:
        record_preview = f"{record_preview[:197]}..."
    raise AcquisitionError(
        "Malformed VCF record "
        f"in {sample_vcf} for sample '{sample_id}' at line {line_number}: expected at least "
        f"{expected_min_columns} tab-delimited columns, found {observed_columns}; "
        f"record={record_preview!r}"
    )


def ensure_bcftools_available(executable: str = "bcftools") -> str:
    """Verify that ``bcftools`` (or a named executable) is on ``$PATH``.

    Args:
        executable: Name of the CLI tool to locate (default ``"bcftools"``).

    Returns:
        The resolved absolute path to the executable.

    Raises:
        MissingToolError: If :func:`shutil.which` cannot locate the tool.
    """
    resolved = shutil.which(executable)
    if not resolved:
        raise MissingToolError(
            f"Required executable '{executable}' was not found on PATH; "
            "install bcftools before consensus generation."
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
    """Generate a consensus FASTA for one sample via ``bcftools consensus``.

    Orchestrates the full consensus pipeline for a single sample:

    1. Validates that ``bcftools`` is available on ``$PATH``.
    2. Calls :func:`_prepare_consensus` to filter the VCF (keeping only
       ``apply_alt`` records) and write a BED mask file for sites that
       should be replaced with ``N``.
    3. Launches ``bcftools consensus`` as a subprocess, piping the
       reference FASTA into stdin and capturing the consensus output.

    .. warning::

       The reference FASTA is streamed into ``bcftools`` stdin via a
       :class:`~concurrent.futures.ThreadPoolExecutor` with
       ``max_workers=1`` to prevent deadlock.  Without the background
       thread, the main thread would block writing to stdin while
       ``bcftools`` blocks writing to stdout — a classic pipe deadlock.
       The writer thread catches :class:`BrokenPipeError` silently
       (``bcftools`` may close stdin early on error) and always closes
       the handle in a ``finally`` block to avoid resource leaks.  The
       main thread reads stderr and waits for the process *before*
       calling ``stdin_future.result()`` to propagate any writer
       exceptions.

    Args:
        sample_id: Identifier of the sample to extract from the VCF.
        reference_fasta: Path to the reference genome FASTA (plain or
            gzipped).
        sample_vcf: Path to the input VCF file (plain or gzipped).
        output_fasta: Path where the consensus FASTA will be written.
        bcftools_executable: Name or path of the ``bcftools`` binary.
        expected_reference_tokens: Canonical build tokens that must
            appear in both the FASTA filename and VCF ``##reference``
            header.

    Returns:
        A :class:`ConsensusResult` containing the output path and
        diagnostics.

    Raises:
        MissingToolError: If ``bcftools`` is not found on ``$PATH``.
        ReferenceMismatchError: If reference build validation fails.
        ContigMismatchError: If VCF contigs are absent from the FASTA.
        AcquisitionError: If ``bcftools consensus`` exits with a
            non-zero return code.
    """
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
        with (
            _open_maybe_gzip(reference_path) as reference_handle,
            output_path.open("w", encoding="utf-8") as output_handle,
        ):
            with subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=output_handle,
                stderr=subprocess.PIPE,
                text=True,
            ) as completed:
                stdin_handle = completed.stdin
                if stdin_handle is None:
                    raise AcquisitionError(
                        f"bcftools consensus did not expose stdin for {sample_id}"
                    )

                def _write_reference_to_stdin() -> None:
                    """Stream the reference FASTA into bcftools stdin.

                    Runs in a background thread to avoid pipe deadlock.
                    Catches :class:`BrokenPipeError` silently because
                    ``bcftools`` may close stdin early on error.  Always
                    closes the stdin handle in ``finally`` to prevent
                    resource leaks.
                    """
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
    """Generate consensus FASTAs for multiple samples in parallel.

    Wraps :func:`generate_consensus_fasta` with a
    :class:`~concurrent.futures.ThreadPoolExecutor` to process samples
    concurrently.  The worker count is clamped to
    ``max(1, min(max_workers, len(sample_vcfs)))`` so that small batches
    do not spawn unnecessary threads.

    Each worker invocation calls :func:`generate_consensus_fasta`, which
    itself spawns an internal single-thread executor for stdin piping.
    This creates a nested ``ThreadPoolExecutor`` pattern, but deadlock is
    avoided because the inner executor has ``max_workers=1`` and the
    outer executor uses :meth:`~concurrent.futures.Executor.map` (which
    consumes results lazily and propagates exceptions promptly).

    Args:
        reference_fasta: Path to the shared reference genome FASTA.
        sample_vcfs: Mapping of sample ID → VCF file path.
        output_dir: Directory where ``{sample_id}.fa`` files are written.
        max_workers: Maximum number of concurrent worker threads.
        bcftools_executable: Name or path of the ``bcftools`` binary.
        expected_reference_tokens: Canonical build tokens for reference
            validation.

    Returns:
        A dict mapping sample ID → :class:`ConsensusResult`.

    Raises:
        AcquisitionError: If any individual sample consensus fails
            (propagated from the worker thread).
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def _worker(sample_and_vcf: tuple[str, str | Path]) -> tuple[str, ConsensusResult]:
        """Process a single (sample_id, vcf_path) pair via :func:`generate_consensus_fasta`."""
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
    """Filter a VCF and build mask BED for ``bcftools consensus``.

    Performs a single streaming pass over the input VCF to:

    * Validate that the ``##reference`` header and contig declarations
      match the reference FASTA.
    * Locate the target sample column in the ``#CHROM`` header.
    * Classify every data record via :func:`classify_consensus_site`.
    * Write only ``apply_alt`` records to a filtered VCF.
    * Accumulate masked intervals into a BED file.
    * Compute :class:`ConsensusDiagnostics` aggregate metrics.

    Args:
        sample_id: Identifier of the sample to extract.
        reference_fasta: Path to the reference genome FASTA.
        sample_vcf: Path to the input VCF (plain or gzipped).
        work_dir: Temporary directory for intermediate files.
        expected_reference_tokens: Canonical build tokens for
            reference validation.

    Returns:
        A :class:`_PreparedConsensus` containing paths to the filtered
        VCF, optional mask BED, diagnostics, and mask spans.

    Raises:
        AcquisitionError: If the sample is not found in the VCF, the
            VCF lacks a ``#CHROM`` header, or contains no data records.
        ReferenceMismatchError: If reference build validation fails.
        ContigMismatchError: If VCF contigs are absent from the FASTA.
    """
    contig_headers = _read_fasta_headers(reference_fasta)
    reference_evidence = " ".join((reference_fasta.name, *contig_headers.values()))
    if not _matches_expected_reference_build(reference_evidence, expected_reference_tokens):
        raise ReferenceMismatchError(
            "Reference FASTA "
            f"{reference_fasta} does not canonically match expected build evidence "
            f"{expected_reference_tokens}"
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
        for line_number, line in enumerate(source, start=1):
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
                        f"VCF {sample_vcf} is missing explicit reference/build metadata "
                        "in a ##reference header"
                    )
                if not _matches_expected_reference_build(vcf_reference, expected_reference_tokens):
                    raise ReferenceMismatchError(
                        f"VCF {sample_vcf} declares reference '{vcf_reference}', "
                        "which does not canonically match "
                        f"expected build evidence {expected_reference_tokens}"
                    )
                if header_contigs and not header_contigs.issubset(contig_headers.keys()):
                    missing_contigs = sorted(header_contigs.difference(contig_headers.keys()))
                    raise ContigMismatchError(
                        f"VCF {sample_vcf} references contigs absent from "
                        f"{reference_fasta}: {missing_contigs[:5]}"
                    )
                sample_index = columns.index(sample_id)
                sink.write(line)
                continue

            if sample_index is None:
                raise AcquisitionError(f"VCF {sample_vcf} is missing a #CHROM header row")

            total_records += 1
            raw_record = line.rstrip("\n")
            fields = raw_record.split("\t")
            expected_min_columns = max(9, sample_index + 1)
            if len(fields) < expected_min_columns:
                _raise_malformed_vcf_record(
                    sample_vcf=sample_vcf,
                    sample_id=sample_id,
                    line_number=line_number,
                    raw_record=raw_record,
                    observed_columns=len(fields),
                    expected_min_columns=expected_min_columns,
                )
            chrom, pos_str, _, ref, alt_field, _, filter_value, _, format_field = fields[:9]
            if chrom not in contig_headers:
                raise ContigMismatchError(
                    f"Contig '{chrom}' from {sample_vcf} is absent from {reference_fasta}"
                )
            alts = _normalize_alt_alleles(alt_field.split(",") if alt_field else [])
            sample_format = dict(
                zip(format_field.split(":"), fields[sample_index].split(":"), strict=False)
            )
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
            raise AcquisitionError(
                f"VCF {sample_vcf} does not contain any records for sample {sample_id}"
            )

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


def _read_fasta_headers(reference_fasta: Path) -> dict[str, str]:
    """Extract contig names and full header lines from a FASTA file.

    Reads only the ``>``-prefixed header lines (not sequence data).
    The first whitespace-delimited token of each header is used as the
    contig key.

    Args:
        reference_fasta: Path to the FASTA file (plain or gzipped).

    Returns:
        Dict mapping contig name → full header string (without the
        leading ``>``).

    Raises:
        AcquisitionError: If the file contains no contig headers.
    """
    headers: dict[str, str] = {}
    with _open_maybe_gzip(reference_fasta) as handle:
        for line in handle:
            if line.startswith(">"):
                full_header = line[1:].strip()
                headers[full_header.split()[0]] = full_header
    if not headers:
        raise AcquisitionError(
            f"Reference FASTA {reference_fasta} did not contain any contig headers"
        )
    return headers


def _matches_expected_reference_build(
    evidence: str, expected_reference_tokens: Sequence[str]
) -> bool:
    """Check whether *evidence* contains all expected reference build tokens.

    Both the evidence string and each token are canonicalized (lowercased,
    non-alphanumeric characters replaced with ``_``) before substring
    matching.

    Args:
        evidence: Free-form string to search (e.g. a FASTA filename
            concatenated with contig headers).
        expected_reference_tokens: Canonical token strings that must all
            appear in *evidence*.

    Returns:
        ``True`` if every token is found in the canonicalized evidence.
    """
    canonical_evidence = _canonicalize_reference_evidence(evidence)
    return all(
        _canonicalize_reference_evidence(token) in canonical_evidence
        for token in expected_reference_tokens
    )


def _canonicalize_reference_evidence(evidence: str) -> str:
    """Lowercase and normalise a string for case-insensitive token matching.

    Replaces all non-alphanumeric character runs with ``_`` and strips
    leading/trailing underscores.

    Args:
        evidence: Raw string to canonicalize.

    Returns:
        Normalised string suitable for substring containment checks.
    """
    return re.sub(r"[^a-z0-9]+", "_", evidence.lower()).strip("_")


def _open_maybe_gzip(path: Path):
    """Open a file for text reading, decompressing if the suffix is ``.gz``.

    Args:
        path: Path to the file.

    Returns:
        A text-mode file handle (UTF-8).
    """
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8")
    )
