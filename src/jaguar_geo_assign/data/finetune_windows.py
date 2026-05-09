"""Locus-centered 512bp window extraction for fine-tuning DNABERT-2 on jaguar variants.

Where ``consensus.py`` masks heterozygous and ambiguous sites to emit a single
per-sample FASTA, this module emits **per-locus windows** suited to supervised
fine-tuning. The behavioral split is intentional: heterozygotes are doubled
into two allele-specific windows (one per observed allele) so that the
classifier sees both haplotype contributions instead of losing the locus to
masking. Reference-only homozygotes carry no signal vs. the reference and are
dropped to keep the training corpus informative.

VCF parsing helpers (filter gating, allele normalization, GT validation,
malformed-record diagnostics) and reference-build validation primitives are
imported from ``consensus.py`` to keep the two pipelines' interpretation of
"valid record" in lockstep; diverging here would silently let one pipeline
accept records the other rejects.
"""

from __future__ import annotations

import csv
import json
import logging
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .acquisition import AcquisitionError
from .consensus import (
    PASSING_FILTER_VALUES,
    ContigMismatchError,
    MalformedGenotypeError,
    ReferenceMismatchError,
    _matches_expected_reference_build,
    _normalize_alt_alleles,
    _open_maybe_gzip,
    _raise_malformed_vcf_record,
    _validated_gt_tokens,
    canonicalize_reference_evidence,
)

_LOGGER = logging.getLogger(__name__)

WINDOW_SIZE = 512
UPSTREAM_BASES = 256
DOWNSTREAM_BASES = 255

POSITIVE_REFERENCE_TOKENS = ("HiC_scaffold_1", "Panthera_onca_HiC")
"""Canonical build tokens for the DNA Zoo jaguar fine-tuning reference.

The fine-tuning VCF was generated using the DNA Zoo submission, which retains
the original contig names (unlike NCBI's RefSeq curation which renamed them).
All of these tokens must be present in the loaded FASTA headers/filename.
"""

NEGATIVE_REFERENCE_TOKENS = ("NC_083295.1", "GCF_028533385.1")
"""NCBI-specific tokens that indicate a RefSeq/GenBank reference distribution.

If any of these tokens appear in the FASTA, we raise immediately; they indicate
the user downloaded an NCBI curated repackaging which altered contig names
and will break the VCF's positional coordinate mapping.
"""

ALLOWED_NUCLEOTIDES = frozenset("ACGTN")
"""Whitelist of nucleotide tokens DNABERT-2 can ingest without surprises.

Intentionally narrower than the full IUPAC ambiguity alphabet: any REF/ALT
allele outside this set (e.g. ``*`` spanning-deletion sentinels, IUPAC codes
like ``Y``/``R``, or stray symbols) is rejected before a window is emitted so
the model never sees a token its tokenizer cannot map.
"""


class ReferenceBaseMismatchError(AcquisitionError):
    """The FASTA base at a VCF locus disagrees with the VCF ``REF`` allele.

    Raised before a window is emitted whenever the reference sequence at the
    1-based ``locus_pos`` does not match (case-insensitive) the ``REF``
    column of the VCF record. This typically indicates that the FASTA and
    VCF were derived from different assemblies even when build-token
    validation passes (e.g. same identifier, different patch level).
    """


class PloidyError(AcquisitionError):
    """A VCF ``GT`` field has unexpected ploidy for the diploid jaguar pipeline.

    Raised when the genotype contains anything other than exactly two allele
    tokens. The fine-tuning heterozygote-doubling logic is mathematically
    defined only for diploid calls, so haploid (e.g. ``"1"``), triploid
    (e.g. ``"0/1/1"``) or higher-ploidy records must fail loudly rather than
    be reinterpreted via ``set(tokens)`` collapsing.
    """


class InvalidAlleleAlphabetError(AcquisitionError):
    """A VCF ``REF`` or ``ALT`` allele uses characters outside :data:`ALLOWED_NUCLEOTIDES`.

    Raised eagerly so a single ``*`` spanning-deletion or stray IUPAC
    ambiguity code never reaches :func:`extract_fasta_window` (where it
    would otherwise be silently substituted into a training window the
    tokenizer cannot represent).
    """


@dataclass(frozen=True)
class FinetuneWindow:
    """A 512bp genomic window for fine-tuning, anchored on a single SNP locus.

    Coordinates are stored in two complementary conventions to avoid the
    classic VCF/0-based off-by-one bug downstream: ``locus_pos`` keeps the
    1-based VCF position (so it round-trips exactly to the source record),
    while ``window_start`` / ``window_end`` use 0-based half-open BED-style
    coordinates (so ``sequence == reference[window_start:window_end]`` modulo
    the substituted allele).

    Attributes:
        sample_id: Sample identifier the window was extracted for.
        contig: Chromosome / contig name (matches the VCF CHROM field).
        locus_pos: 1-based VCF position of the SNP at the window center.
        window_start: 0-based inclusive start of the 512bp window.
        window_end: 0-based exclusive end of the 512bp window.
        sequence: The 512-character genomic sequence (uppercase A/C/G/T/N).
        ref_allele: Reference allele at ``locus_pos`` (single base).
        alt_allele: The base actually placed at ``locus_pos`` in this window;
            equals ``ref_allele`` for the reference-allele copy of a
            heterozygote, and the VCF ALT for homozygous-alternate or the
            alternate-allele copy of a heterozygote.
        is_heterozygous: True iff the sample is heterozygous at this locus.
        genotype: Raw VCF GT field (e.g. ``"0/1"``, ``"1|1"``).
        filter_status: VCF FILTER column value (``"PASS"`` or ``"."`` here).
    """

    sample_id: str
    contig: str
    locus_pos: int
    window_start: int
    window_end: int
    sequence: str
    ref_allele: str
    alt_allele: str
    is_heterozygous: bool
    genotype: str
    filter_status: str


@dataclass(frozen=True)
class ReferenceIndex:
    """Cached, validated FASTA reference loaded once for many-sample reuse.

    Bundles the per-contig sequences with the headers and source path used
    to perform reference-build validation. Construct via
    :func:`load_reference_index` so the (one-time) build-token check runs
    exactly once per object; callers can then thread the same index through
    many :func:`iter_locus_windows_from_vcf` calls without re-reading the
    multi-gigabyte FASTA per sample.

    Attributes:
        fasta_path: Path the FASTA was loaded from. Retained for
            diagnostic error messages emitted by per-locus validators.
        contig_sequences: Mapping of contig name → full reference sequence
            (case preserved as on disk; callers uppercase on read).
        contig_headers: Mapping of contig name → full FASTA header line
            (without the leading ``>``). Used by the build-token check
            and surfaced in error messages.
        validated_alphabet: Whether the per-contig sequences have been scanned
            for invalid IUPAC/non-ACGTN characters. Defaults to ``False`` so
            ad hoc construction cannot claim the alphabet contract without an
            explicit scan; :func:`load_reference_index` and
            :func:`extract_locus_windows_from_vcf` set it to ``True`` only
            after validating their inputs.
    """

    fasta_path: Path
    contig_sequences: Mapping[str, str]
    contig_headers: Mapping[str, str]
    validated_alphabet: bool = False


def extract_fasta_window(
    *,
    contig_sequence: str,
    locus_pos: int,
    allele: str,
    upstream: int = UPSTREAM_BASES,
    downstream: int = DOWNSTREAM_BASES,
) -> tuple[str, int, int] | None:
    """Build a window centered on a 1-based locus with ``allele`` substituted at the center.

    Returns ``None`` when the requested window would extend past either contig
    boundary; padding with ``N`` is intentionally avoided so that downstream
    training does not silently learn from synthetic flanks. The center base is
    always replaced even when ``allele`` already equals the reference base, so
    that callers can build "ref-allele" windows for heterozygotes without
    branching.

    Args:
        contig_sequence: Full reference sequence for the contig containing
            the locus (any case; output is uppercased).
        locus_pos: 1-based VCF position of the SNP.
        allele: Single-base allele to place at the locus position. Must be
            in :data:`ALLOWED_NUCLEOTIDES` once uppercased.
        upstream: Number of bases to include before the locus (default 256).
        downstream: Number of bases to include after the locus (default 255).

    Returns:
        Tuple ``(sequence, window_start, window_end)`` with 0-based half-open
        coordinates, or ``None`` if the window extends beyond the contig.

    Raises:
        ValueError: If ``allele`` is not exactly one character.
        InvalidAlleleAlphabetError: If ``allele`` is single-character but
            outside :data:`ALLOWED_NUCLEOTIDES` (case-insensitive); guards
            spanning-deletion ``*`` sentinels and IUPAC ambiguity codes
            from leaking into a training window.
    """
    if len(allele) != 1:
        raise ValueError(f"extract_fasta_window only supports single-base alleles; got {allele!r}")
    upper_allele = allele.upper()
    if upper_allele not in ALLOWED_NUCLEOTIDES:
        raise InvalidAlleleAlphabetError(
            f"Allele {allele!r} is outside the allowed nucleotide alphabet "
            f"{sorted(ALLOWED_NUCLEOTIDES)}; refusing to write it into a training window."
        )
    locus_idx = locus_pos - 1
    window_start = locus_idx - upstream
    window_end = locus_idx + 1 + downstream
    if window_start < 0 or window_end > len(contig_sequence):
        return None
    upstream_seq = contig_sequence[window_start:locus_idx].upper()
    downstream_seq = contig_sequence[locus_idx + 1 : window_end].upper()
    sequence = f"{upstream_seq}{upper_allele}{downstream_seq}"
    return sequence, window_start, window_end


def _read_fasta_sequences(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load every contig's full sequence and header line into memory.

    Trades memory for O(1) random access at every locus; the alternative
    (re-streaming the FASTA per locus) would dominate runtime for VCFs with
    tens of thousands of records. Both the sequence map and the header map
    are returned so callers can do build-token validation without a second
    pass over the file.

    Returns:
        Pair ``(contig_sequences, contig_headers)`` keyed by the first
        whitespace token of each FASTA header.

    Raises:
        AcquisitionError: If the file contains no ``>`` headers.
    """
    sequences: dict[str, list[str]] = {}
    headers: dict[str, str] = {}
    current: str | None = None
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if line.startswith(">"):
                full_header = line[1:].strip()
                current = full_header.split()[0]
                sequences[current] = []
                headers[current] = full_header
            elif current is not None:
                sequences[current].append(line.strip())
    if not sequences:
        raise AcquisitionError(f"Reference FASTA {path} did not contain any contig headers")
    return {name: "".join(parts) for name, parts in sequences.items()}, headers


def _validate_finetune_reference_evidence(
    evidence: str,
    *,
    positive_tokens: Sequence[str] = POSITIVE_REFERENCE_TOKENS,
    negative_tokens: Sequence[str] = NEGATIVE_REFERENCE_TOKENS,
) -> None:
    """Validate that the reference evidence satisfies fine-tuning constraints."""
    if not _matches_expected_reference_build(evidence, positive_tokens):
        missing = [
            t
            for t in positive_tokens
            if canonicalize_reference_evidence(t) not in canonicalize_reference_evidence(evidence)
        ]
        canon = canonicalize_reference_evidence(evidence)
        trunc = canon[:200] + "..." if len(canon) > 200 else canon
        raise ReferenceMismatchError(
            f"Reference evidence missing expected positive tokens {missing}. "
            f"Canonicalized evidence (truncated): {trunc}"
        )

    canonical_evidence = canonicalize_reference_evidence(evidence)
    for token in negative_tokens:
        canon_token = canonicalize_reference_evidence(token)
        if canon_token in canonical_evidence:
            raise ReferenceMismatchError(
                f"Negative reference token {token!r} found in evidence. "
                f"This token is forbidden in the fine-tuning pipeline."
            )


def load_reference_index(
    reference_fasta: str | Path,
    *,
    positive_reference_tokens: Sequence[str] = POSITIVE_REFERENCE_TOKENS,
    negative_reference_tokens: Sequence[str] = NEGATIVE_REFERENCE_TOKENS,
) -> ReferenceIndex:
    """Load and validate a reference FASTA exactly once for many-sample reuse.

    Performs the same build-token sanity check that ``consensus.py``
    enforces (filename + header lines must jointly contain every expected
    token), then returns a :class:`ReferenceIndex` callers thread through
    every per-sample :func:`iter_locus_windows_from_vcf` call. This is the
    seam that prevents the production 57-sample workflow from re-reading
    the ~2.5 GB FASTA per sample.

    Args:
        reference_fasta: Path to the reference FASTA (plain or gzipped).
        positive_reference_tokens: Canonical build tokens that must
            appear in either the filename or any header line.
        negative_reference_tokens: Tokens that must NOT appear.

    Returns:
        A :class:`ReferenceIndex` ready for streaming window extraction.

    Raises:
        ReferenceMismatchError: If the FASTA filename + header evidence
            violates the positive/negative token constraints.
        AcquisitionError: Propagated from :func:`_read_fasta_sequences`
            if the FASTA contains no contig headers.
        InvalidAlleleAlphabetError: If any contig contains characters outside
            the allowed nucleotide alphabet.
    """
    fasta_path = Path(reference_fasta)
    contig_sequences, contig_headers = _read_fasta_sequences(fasta_path)
    reference_evidence = " ".join((fasta_path.name, *contig_headers.values()))

    _validate_finetune_reference_evidence(
        reference_evidence,
        positive_tokens=positive_reference_tokens,
        negative_tokens=negative_reference_tokens,
    )

    for contig_name, sequence in contig_sequences.items():
        offending = set(sequence.upper()) - ALLOWED_NUCLEOTIDES
        if offending:
            raise InvalidAlleleAlphabetError(
                f"Contig {contig_name!r} contains invalid characters {sorted(offending)} "
                f"in FASTA {fasta_path}. Allowed alphabet is {sorted(ALLOWED_NUCLEOTIDES)}."
            )

    return ReferenceIndex(
        fasta_path=fasta_path,
        contig_sequences=contig_sequences,
        contig_headers=contig_headers,
        validated_alphabet=True,
    )


def _validate_diploid_tokens(
    tokens: list[str],
    *,
    sample_id: str,
    contig: str,
    locus_pos: int,
    genotype: str,
    vcf_path: Path,
) -> None:
    """Reject any genotype whose ploidy is not exactly two.

    The heterozygote-doubling contract (one ref-allele copy plus one
    alt-allele copy) is mathematically defined only for diploid calls;
    haploid or polyploid records would silently collapse via
    ``set(tokens)`` and emit the wrong number of windows.
    """
    if len(tokens) != 2:
        raise PloidyError(
            f"Non-diploid genotype GT='{genotype}' (ploidy={len(tokens)}) for sample "
            f"'{sample_id}' at {contig}:{locus_pos} in VCF {vcf_path}; the fine-tuning "
            "pipeline requires exactly diploid calls."
        )


def _validate_allele_alphabet(
    *,
    ref: str,
    alt: str,
    sample_id: str,
    contig: str,
    locus_pos: int,
    vcf_path: Path,
) -> None:
    """Reject REF/ALT alleles outside :data:`ALLOWED_NUCLEOTIDES`.

    Invoked before window extraction so that ``*`` spanning-deletion
    sentinels and IUPAC ambiguity codes raise with full locus context
    rather than triggering a later, opaque
    :class:`InvalidAlleleAlphabetError` inside :func:`extract_fasta_window`.
    """
    for label, allele in (("REF", ref), ("ALT", alt)):
        if allele.upper() not in ALLOWED_NUCLEOTIDES:
            raise InvalidAlleleAlphabetError(
                f"{label} allele {allele!r} for sample '{sample_id}' at "
                f"{contig}:{locus_pos} in VCF {vcf_path} is outside the allowed alphabet "
                f"{sorted(ALLOWED_NUCLEOTIDES)}."
            )


def _validate_reference_base(
    *,
    reference: ReferenceIndex,
    contig: str,
    locus_pos: int,
    ref_allele: str,
    sample_id: str,
    vcf_path: Path,
) -> None:
    """Assert the FASTA base at ``locus_pos`` matches the VCF ``REF`` allele.

    Case-insensitive: real FASTAs ship soft-masked (lowercase) flanks, so
    the comparison must canonicalise both sides. Without this guard, a
    silently mis-matched FASTA (e.g. wrong patch level even though build
    tokens agree) would still produce plausible-looking 512bp windows
    with the wrong genomic context.
    """
    sequence = reference.contig_sequences[contig]
    if not 1 <= locus_pos <= len(sequence):
        raise ReferenceBaseMismatchError(
            f"Locus position {locus_pos} for sample '{sample_id}' is outside contig "
            f"{contig!r} (length {len(sequence)}) in reference {reference.fasta_path}; "
            f"VCF {vcf_path} disagrees with the reference."
        )
    fasta_base = sequence[locus_pos - 1].upper()
    if fasta_base != ref_allele.upper():
        raise ReferenceBaseMismatchError(
            f"REF allele mismatch for sample '{sample_id}' at {contig}:{locus_pos}: "
            f"VCF REF={ref_allele!r} but reference FASTA base={fasta_base!r}; "
            f"FASTA={reference.fasta_path}, VCF={vcf_path}."
        )


def iter_locus_windows_from_vcf(
    *,
    sample_id: str,
    sample_vcf: str | Path,
    reference: ReferenceIndex,
    positive_reference_tokens: Sequence[str] = POSITIVE_REFERENCE_TOKENS,
    negative_reference_tokens: Sequence[str] = NEGATIVE_REFERENCE_TOKENS,
) -> Iterator[FinetuneWindow]:
    """Stream windows for one sample without materialising the full list.

    The production-scaling entry point: callers iterate the generator and
    write each window to disk (e.g. via :func:`write_locus_windows_jsonl`)
    so peak memory stays bounded by a single record regardless of how many
    loci the VCF contains.

    Validation contract (each guard prevents a silent-failure mode):
        * ``##reference`` header must be present and match the expected
          build tokens (lockstep with ``consensus.py``).
        * Every ``##contig=<ID=...>`` declaration must exist in the
          reference FASTA; raises :class:`ContigMismatchError` otherwise.
        * Every per-record ``CHROM`` must exist in the reference; raises
          :class:`ContigMismatchError` instead of silently skipping.
        * Truncated rows raise :class:`AcquisitionError` with line
          context (no silent drop).
        * ``GT`` must appear in the per-record ``FORMAT`` schema.
        * Genotype ploidy must be exactly two.
        * REF/ALT alleles must be in :data:`ALLOWED_NUCLEOTIDES`.
        * The FASTA base at every emitted locus must equal the VCF
          ``REF`` (case-insensitive).

    Emission policy (records that pass validation but yield no window):
        * Non-PASS ``FILTER`` values, multi-allelic ALT fields, indels
          (``len(REF) != 1`` or ``len(ALT) != 1``), and reference-only
          homozygotes (``0/0``) are dropped silently because they carry
          no informative signal vs. the reference.
        * Heterozygotes whose unique genotype indices are not exactly
          ``{0, 1}`` (e.g. ``1/2`` against a multi-allelic REF/ALT site
          that survived earlier filtering) are also skipped, since the
          ref/alt window-pair contract requires a biallelic 0/1 layout.
          A ``DEBUG``-level log is emitted at each such skip for
          observability.

    Args:
        sample_id: VCF column to read genotypes from.
        sample_vcf: Path to the input VCF (plain or gzipped).
        reference: Pre-loaded :class:`ReferenceIndex` (call
            :func:`load_reference_index` once, reuse across samples).
        positive_reference_tokens: Build tokens enforced on the VCF
            ``##reference`` header (must match those used to load the
            reference index).
        negative_reference_tokens: Build tokens that must NOT appear in the
            VCF ``##reference`` header.

    Yields:
        :class:`FinetuneWindow` instances in VCF record order, with
        heterozygote pairs emitted consecutively (ref-allele copy first).
    """
    vcf_path = Path(sample_vcf)
    contig_sequences = reference.contig_sequences
    fasta_contigs = set(contig_sequences.keys())
    with _open_maybe_gzip(vcf_path) as source:
        sample_index: int | None = None
        header_contigs: set[str] = set()
        vcf_reference = ""
        for line_number, line in enumerate(source, start=1):
            if line.startswith("##"):
                if line.startswith("##contig=<ID="):
                    header_contigs.add(line.split("ID=", 1)[1].split(",", 1)[0].rstrip(">\n"))
                if line.startswith("##reference="):
                    vcf_reference = line.split("=", 1)[1].strip()
                continue
            if line.startswith("#CHROM"):
                columns = line.rstrip("\n").split("\t")
                if sample_id not in columns[9:]:
                    raise AcquisitionError(f"Sample '{sample_id}' not found in VCF {vcf_path}")
                if not vcf_reference:
                    raise ReferenceMismatchError(
                        f"VCF {vcf_path} is missing explicit reference/build metadata "
                        "in a ##reference header"
                    )
                try:
                    _validate_finetune_reference_evidence(
                        vcf_reference,
                        positive_tokens=positive_reference_tokens,
                        negative_tokens=negative_reference_tokens,
                    )
                except ReferenceMismatchError as e:
                    raise ReferenceMismatchError(
                        f"VCF {vcf_path} declares reference '{vcf_reference}', "
                        f"which failed build evidence validation: {e}"
                    ) from e
                if header_contigs and not header_contigs.issubset(fasta_contigs):
                    missing_contigs = sorted(header_contigs.difference(fasta_contigs))
                    raise ContigMismatchError(
                        f"VCF {vcf_path} references contigs absent from "
                        f"{reference.fasta_path}: {missing_contigs[:5]}"
                    )
                sample_index = columns.index(sample_id)
                continue
            if sample_index is None:
                raise AcquisitionError(f"VCF {vcf_path} is missing a #CHROM header row")
            yield from _parse_vcf_record_to_windows(
                raw_line=line,
                line_number=line_number,
                fields_count_min=max(9, sample_index + 1),
                sample_id=sample_id,
                sample_index=sample_index,
                reference=reference,
                fasta_contigs=fasta_contigs,
                vcf_path=vcf_path,
            )


def _parse_vcf_record_to_windows(
    *,
    raw_line: str,
    line_number: int,
    fields_count_min: int,
    sample_id: str,
    sample_index: int,
    reference: ReferenceIndex,
    fasta_contigs: set[str],
    vcf_path: Path,
) -> Iterator[FinetuneWindow]:
    """Parse one VCF data row and yield 0, 1, or 2 windows from it.

    Split out of :func:`iter_locus_windows_from_vcf` to keep the streaming
    loop readable: the validation contract documented on the parent
    function is enforced *here* per record, while the parent only handles
    header bookkeeping.
    """
    raw_record = raw_line.rstrip("\n")
    fields = raw_record.split("\t")
    if len(fields) < fields_count_min:
        _raise_malformed_vcf_record(
            sample_vcf=vcf_path,
            sample_id=sample_id,
            line_number=line_number,
            raw_record=raw_record,
            observed_columns=len(fields),
            expected_min_columns=fields_count_min,
        )
    chrom, pos_str, _, ref, alt_field, _, filter_value, _, format_field = fields[:9]

    if chrom not in fasta_contigs:
        raise ContigMismatchError(
            f"Contig '{chrom}' from {vcf_path} is absent from {reference.fasta_path}"
        )
    if filter_value not in PASSING_FILTER_VALUES:
        return

    alts = _normalize_alt_alleles(alt_field.split(",") if alt_field else [])
    if len(alts) != 1:
        return
    alt = alts[0]
    if len(ref) != 1 or len(alt) != 1:
        return

    locus_pos = int(pos_str)
    _validate_allele_alphabet(
        ref=ref,
        alt=alt,
        sample_id=sample_id,
        contig=chrom,
        locus_pos=locus_pos,
        vcf_path=vcf_path,
    )

    format_keys = format_field.split(":")
    if "GT" not in format_keys:
        raise AcquisitionError(
            f"VCF {vcf_path} record at {chrom}:{locus_pos} for sample '{sample_id}' "
            f"is missing 'GT' in FORMAT schema {format_keys!r}; the fine-tuning "
            "pipeline cannot derive zygosity without it."
        )
    sample_format = dict(zip(format_keys, fields[sample_index].split(":"), strict=False))
    genotype_raw = sample_format.get("GT")
    tokens = _validated_gt_tokens(
        genotype_raw,
        sample_id=sample_id,
        contig=chrom,
        position=locus_pos,
        vcf_path=vcf_path,
    )
    if tokens is None:
        return
    _validate_diploid_tokens(
        tokens,
        sample_id=sample_id,
        contig=chrom,
        locus_pos=locus_pos,
        genotype=genotype_raw or "",
        vcf_path=vcf_path,
    )

    unique_indices = set(tokens)
    is_heterozygous = len(unique_indices) != 1
    if is_heterozygous:
        if unique_indices != {"0", "1"}:
            _LOGGER.debug(
                "Skipping non-{0,1} heterozygote at %s:%d for sample %s "
                "(genotype=%r, vcf=%s); only biallelic 0/1 hets are emitted as window pairs.",
                chrom,
                locus_pos,
                sample_id,
                genotype_raw,
                vcf_path,
            )
            return
        alleles_to_emit: tuple[str, ...] = (ref, alt)
    else:
        allele_index = int(tokens[0])
        if allele_index != 1:
            return
        alleles_to_emit = (alt,)

    _validate_reference_base(
        reference=reference,
        contig=chrom,
        locus_pos=locus_pos,
        ref_allele=ref,
        sample_id=sample_id,
        vcf_path=vcf_path,
    )
    contig_seq = reference.contig_sequences[chrom]
    for emitted_allele in alleles_to_emit:
        window = extract_fasta_window(
            contig_sequence=contig_seq,
            locus_pos=locus_pos,
            allele=emitted_allele,
        )
        if window is None:
            continue
        sequence, window_start, window_end = window
        yield FinetuneWindow(
            sample_id=sample_id,
            contig=chrom,
            locus_pos=locus_pos,
            window_start=window_start,
            window_end=window_end,
            sequence=sequence,
            ref_allele=ref,
            alt_allele=emitted_allele,
            is_heterozygous=is_heterozygous,
            genotype=genotype_raw or "",
            filter_status=filter_value,
        )


def write_locus_windows_jsonl(
    windows: Iterable[FinetuneWindow],
    output_jsonl: str | Path,
) -> int:
    """Stream windows to a JSONL file without buffering them in memory.

    Pairs with :func:`iter_locus_windows_from_vcf` to give the production
    pipeline an O(1)-memory write path: parent directories are created
    once, then each window is serialised and flushed in record order.

    Args:
        windows: Iterable of :class:`FinetuneWindow` (typically the
            generator returned by :func:`iter_locus_windows_from_vcf`).
        output_jsonl: Destination path; parent directories are created.

    Returns:
        Number of records written (useful for diagnostics / smoke tests).
    """
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for window in windows:
            handle.write(json.dumps(asdict(window)) + "\n")
            written += 1
    return written


def extract_locus_windows_from_vcf(
    *,
    sample_id: str,
    sample_vcf: str | Path,
    contig_sequences: Mapping[str, str],
    reference_path: str | Path | None = None,
    positive_reference_tokens: Sequence[str] = POSITIVE_REFERENCE_TOKENS,
    negative_reference_tokens: Sequence[str] = NEGATIVE_REFERENCE_TOKENS,
) -> list[FinetuneWindow]:
    """Materialise all windows for one sample as a list (test-scale convenience).

    Wraps :func:`iter_locus_windows_from_vcf` for callers that already
    hold a ``contig_sequences`` mapping (e.g. unit tests reusing a
    cached reference). The in-memory sequences are alphabet-scanned up
    front so this convenience path preserves the same invalid-base
    contract as :func:`load_reference_index`. Production-scale callers should prefer
    :func:`iter_locus_windows_from_vcf` directly so peak memory stays
    bounded by a single window.

    Args:
        sample_id: VCF column to read genotypes from.
        sample_vcf: Path to the input VCF (plain or gzipped).
        contig_sequences: Pre-loaded contig name to sequence mapping.
        reference_path: Optional path used in error messages; defaults
            to ``"<in-memory>"`` when sequences are synthesised.
        positive_reference_tokens: Build tokens enforced on the VCF
            ``##reference`` header.
        negative_reference_tokens: Build tokens that must NOT appear in the
            VCF ``##reference`` header.

    Returns:
        Windows in VCF record order; heterozygote pairs are consecutive.

    Raises:
        InvalidAlleleAlphabetError: If any supplied contig sequence contains
            characters outside :data:`ALLOWED_NUCLEOTIDES`.
    """
    fasta_path = Path(reference_path) if reference_path is not None else Path("<in-memory>")
    for contig_name, sequence in contig_sequences.items():
        offending = set(sequence.upper()) - ALLOWED_NUCLEOTIDES
        if offending:
            raise InvalidAlleleAlphabetError(
                f"Contig {contig_name!r} contains invalid characters {sorted(offending)} "
                f"in reference {fasta_path}. Allowed alphabet is {sorted(ALLOWED_NUCLEOTIDES)}."
            )
    reference = ReferenceIndex(
        fasta_path=fasta_path,
        contig_sequences=contig_sequences,
        contig_headers={name: name for name in contig_sequences},
        validated_alphabet=True,
    )
    return list(
        iter_locus_windows_from_vcf(
            sample_id=sample_id,
            sample_vcf=sample_vcf,
            reference=reference,
            positive_reference_tokens=positive_reference_tokens,
            negative_reference_tokens=negative_reference_tokens,
        )
    )


def extract_fasta_windows_for_sample(
    *,
    sample_id: str,
    reference_fasta: str | Path,
    sample_vcf: str | Path,
    output_jsonl: str | Path | None = None,
    positive_reference_tokens: Sequence[str] = POSITIVE_REFERENCE_TOKENS,
    negative_reference_tokens: Sequence[str] = NEGATIVE_REFERENCE_TOKENS,
) -> list[FinetuneWindow]:
    """Single-sample convenience wrapper. **Test/small-workload use only.**

    .. warning::

       This wrapper re-loads the entire reference FASTA on every call.
       For the production 57-sample jaguar workflow, call
       :func:`load_reference_index` **once**, then thread the resulting
       :class:`ReferenceIndex` through :func:`iter_locus_windows_from_vcf`
       and :func:`write_locus_windows_jsonl` per sample. Doing so cuts
       peak memory and I/O by a factor of ``num_samples``.

    Args:
        sample_id: VCF column to extract genotypes for.
        reference_fasta: Path to the reference FASTA (plain or gzipped).
        sample_vcf: Path to the input VCF (plain or gzipped).
        output_jsonl: Optional path to write one JSON record per window.
        positive_reference_tokens: Build tokens enforced on both the
            FASTA evidence and the VCF ``##reference`` header.
        negative_reference_tokens: Build tokens that must NOT appear in the
            VCF ``##reference`` header.

    Returns:
        All windows extracted for ``sample_id``, in VCF record order.
    """
    warnings.warn(
        "extract_fasta_windows_for_sample() reloads the full reference FASTA "
        "(~2.5 GB for the jaguar build) on every call and is intended for tests "
        "and small workloads only. For multi-sample production runs, call "
        "load_reference_index() once and thread the resulting ReferenceIndex "
        "through iter_locus_windows_from_vcf() per sample.",
        stacklevel=2,
    )
    reference = load_reference_index(
        reference_fasta,
        positive_reference_tokens=positive_reference_tokens,
        negative_reference_tokens=negative_reference_tokens,
    )
    windows = list(
        iter_locus_windows_from_vcf(
            sample_id=sample_id,
            sample_vcf=Path(sample_vcf),
            reference=reference,
            positive_reference_tokens=positive_reference_tokens,
            negative_reference_tokens=negative_reference_tokens,
        )
    )
    if output_jsonl is not None:
        write_locus_windows_jsonl(windows, output_jsonl)
    return windows


@dataclass(frozen=True)
class WindowExtractionResult:
    """Summary returned by :func:`extract_windows_for_samples`."""

    total_windows: int
    samples_processed: int
    samples_skipped: int
    output_path: Path


def extract_windows_for_samples(
    *,
    reference_fasta: str | Path,
    vcf: str | Path,
    metadata_csv: str | Path,
    output_jsonl: str | Path,
    positive_reference_tokens: Sequence[str] = POSITIVE_REFERENCE_TOKENS,
    negative_reference_tokens: Sequence[str] = NEGATIVE_REFERENCE_TOKENS,
) -> WindowExtractionResult:
    """Orchestrate multi-sample window extraction from a single VCF.

    Loads the reference FASTA exactly once, reads sample IDs from the
    metadata CSV, and streams windows for every sample into a combined
    JSONL file.

    Args:
        reference_fasta: Path to the DNA Zoo Panthera onca HiC reference.
        vcf: Path to the (possibly multi-sample) VCF.
        metadata_csv: CSV with at least a ``sample_id`` column.
        output_jsonl: Destination JSONL path.
        positive_reference_tokens: Build tokens for the FASTA.
        negative_reference_tokens: Forbidden build tokens.

    Returns:
        A :class:`WindowExtractionResult` summarising the run.

    Raises:
        ValueError: If the metadata CSV is missing a ``sample_id`` column
            or contains no sample rows.
    """
    csv_path = Path(metadata_csv)
    sample_ids: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"Metadata CSV {csv_path} is missing a 'sample_id' column.")
        for row in reader:
            sid = row.get("sample_id", "").strip()
            if sid:
                sample_ids.append(sid)
    if not sample_ids:
        raise ValueError(f"Metadata CSV {csv_path} contains no rows with a non-empty sample_id.")

    reference = load_reference_index(
        reference_fasta,
        positive_reference_tokens=positive_reference_tokens,
        negative_reference_tokens=negative_reference_tokens,
    )

    vcf_path = Path(vcf)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    samples_processed = 0
    samples_skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            try:
                for window in iter_locus_windows_from_vcf(
                    sample_id=sample_id,
                    sample_vcf=vcf_path,
                    reference=reference,
                    positive_reference_tokens=positive_reference_tokens,
                    negative_reference_tokens=negative_reference_tokens,
                ):
                    handle.write(json.dumps(asdict(window)) + "\n")
                    total_windows += 1
                samples_processed += 1
            except (
                ContigMismatchError,
                ReferenceMismatchError,
                ReferenceBaseMismatchError,
                PloidyError,
                InvalidAlleleAlphabetError,
                MalformedGenotypeError,
            ):
                raise
            except AcquisitionError:
                _LOGGER.warning(
                    "Skipping sample %r: not found in VCF %s",
                    sample_id,
                    vcf_path,
                )
                samples_skipped += 1

    if samples_processed == 0:
        _LOGGER.warning(
            "No samples were successfully processed from %s. "
            "Check that sample IDs in %s match the VCF sample columns.",
            vcf_path,
            csv_path,
        )

    return WindowExtractionResult(
        total_windows=total_windows,
        samples_processed=samples_processed,
        samples_skipped=samples_skipped,
        output_path=output_path,
    )
