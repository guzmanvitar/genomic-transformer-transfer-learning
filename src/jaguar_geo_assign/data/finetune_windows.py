"""Locus-centered 512bp window extraction for fine-tuning DNABERT-2 on jaguar variants.

Where ``consensus.py`` masks heterozygous and ambiguous sites to emit a single
per-sample FASTA, this module emits **per-locus windows** suited to supervised
fine-tuning. The behavioral split is intentional: heterozygotes are doubled
into two allele-specific windows (one per observed allele) so that the
classifier sees both haplotype contributions instead of losing the locus to
masking. Reference-only homozygotes carry no signal vs. the reference and are
dropped to keep the training corpus informative.

VCF parsing helpers (filter gating, allele normalization, GT validation) are
imported from ``consensus.py`` to keep the two pipelines' interpretation of
"valid record" in lockstep; diverging here would silently let one pipeline
accept records the other rejects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .acquisition import AcquisitionError
from .consensus import (
    PASSING_FILTER_VALUES,
    _normalize_alt_alleles,
    _open_maybe_gzip,
    _validated_gt_tokens,
)

WINDOW_SIZE = 512
UPSTREAM_BASES = 256
DOWNSTREAM_BASES = 255


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
        allele: Single-base allele to place at the locus position.
        upstream: Number of bases to include before the locus (default 256).
        downstream: Number of bases to include after the locus (default 255).

    Returns:
        Tuple ``(sequence, window_start, window_end)`` with 0-based half-open
        coordinates, or ``None`` if the window extends beyond the contig.

    Raises:
        ValueError: If ``allele`` is not a single base (multi-base variants
            cannot be substituted into a fixed-width window without shifting
            the flank coordinates).
    """
    if len(allele) != 1:
        raise ValueError(f"extract_fasta_window only supports single-base alleles; got {allele!r}")
    locus_idx = locus_pos - 1
    window_start = locus_idx - upstream
    window_end = locus_idx + 1 + downstream
    if window_start < 0 or window_end > len(contig_sequence):
        return None
    upstream_seq = contig_sequence[window_start:locus_idx].upper()
    downstream_seq = contig_sequence[locus_idx + 1 : window_end].upper()
    sequence = f"{upstream_seq}{allele.upper()}{downstream_seq}"
    return sequence, window_start, window_end


def _read_fasta_sequences(path: Path) -> dict[str, str]:
    """Load every contig's full sequence into memory keyed by contig name.

    Trades memory for O(1) random access at every locus; the alternative
    (re-streaming the FASTA per locus) would dominate runtime for VCFs with
    tens of thousands of records. For the jaguar reference (~2.5GB) this fits
    in RAM on the target machines; if that ever changes, swap this for an
    indexed FASTA reader without touching ``extract_fasta_window``.
    """
    sequences: dict[str, list[str]] = {}
    current: str | None = None
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if line.startswith(">"):
                current = line[1:].split()[0]
                sequences[current] = []
            elif current is not None:
                sequences[current].append(line.strip())
    if not sequences:
        raise AcquisitionError(f"Reference FASTA {path} did not contain any contig headers")
    return {name: "".join(parts) for name, parts in sequences.items()}


def extract_locus_windows_from_vcf(
    *,
    sample_id: str,
    sample_vcf: str | Path,
    contig_sequences: Mapping[str, str],
) -> list[FinetuneWindow]:
    """Stream a VCF and emit one ``FinetuneWindow`` per (locus, observed allele) pair.

    The decision tree is intentionally narrower than ``classify_consensus_site``:
    only single-base biallelic PASS records produce output, because anything
    else (indels, multi-allelics, filtered, no-call) cannot be represented as
    a clean single-base substitution into a fixed-width window. Heterozygous
    biallelic records produce **two** windows (ref-allele then alt-allele
    copy) so that both haplotypes contribute training signal.
    Homozygous-reference records are dropped because they carry no signal
    beyond the reference.

    Args:
        sample_id: VCF column to read genotypes from. Must appear in the
            ``#CHROM`` header row.
        sample_vcf: Path to the input VCF (plain or gzipped).
        contig_sequences: Mapping of contig name → full reference sequence.
            Records on contigs absent from this mapping are skipped silently;
            this mirrors how the integration test will subset to a few
            contigs without having to filter the VCF first.

    Returns:
        Windows in VCF record order. Heterozygote pairs are emitted
        consecutively (ref-allele copy first, then alt-allele copy).

    Raises:
        AcquisitionError: If ``sample_id`` is not a column in the VCF or the
            VCF lacks a ``#CHROM`` header before the first data record.
        MalformedGenotypeError: Propagated from ``_validated_gt_tokens`` when
            a GT field contains non-numeric, non-missing tokens.
    """
    vcf_path = Path(sample_vcf)
    windows: list[FinetuneWindow] = []
    with _open_maybe_gzip(vcf_path) as source:
        sample_index: int | None = None
        for line in source:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                columns = line.rstrip("\n").split("\t")
                if sample_id not in columns[9:]:
                    raise AcquisitionError(f"Sample '{sample_id}' not found in VCF {vcf_path}")
                sample_index = columns.index(sample_id)
                continue
            if sample_index is None:
                raise AcquisitionError(f"VCF {vcf_path} is missing a #CHROM header row")

            fields = line.rstrip("\n").split("\t")
            if len(fields) < max(9, sample_index + 1):
                continue
            chrom, pos_str, _, ref, alt_field, _, filter_value, _, format_field = fields[:9]

            if filter_value not in PASSING_FILTER_VALUES:
                continue
            if chrom not in contig_sequences:
                continue

            alts = _normalize_alt_alleles(alt_field.split(",") if alt_field else [])
            if len(alts) != 1:
                continue
            alt = alts[0]
            if len(ref) != 1 or len(alt) != 1:
                continue

            locus_pos = int(pos_str)
            sample_format = dict(
                zip(format_field.split(":"), fields[sample_index].split(":"), strict=False)
            )
            genotype_raw = sample_format.get("GT")
            tokens = _validated_gt_tokens(
                genotype_raw,
                sample_id=sample_id,
                contig=chrom,
                position=locus_pos,
                vcf_path=vcf_path,
            )
            if tokens is None:
                continue

            unique_indices = set(tokens)
            is_heterozygous = len(unique_indices) != 1
            if is_heterozygous:
                if unique_indices != {"0", "1"}:
                    continue
                alleles_to_emit: tuple[str, ...] = (ref, alt)
            else:
                allele_index = int(tokens[0])
                if allele_index == 0:
                    continue
                if allele_index != 1:
                    continue
                alleles_to_emit = (alt,)

            contig_seq = contig_sequences[chrom]
            for emitted_allele in alleles_to_emit:
                window = extract_fasta_window(
                    contig_sequence=contig_seq,
                    locus_pos=locus_pos,
                    allele=emitted_allele,
                )
                if window is None:
                    continue
                sequence, window_start, window_end = window
                windows.append(
                    FinetuneWindow(
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
                )
    return windows


def extract_fasta_windows_for_sample(
    *,
    sample_id: str,
    reference_fasta: str | Path,
    sample_vcf: str | Path,
    output_jsonl: str | Path | None = None,
) -> list[FinetuneWindow]:
    """Orchestrate FASTA loading and VCF extraction for a single sample.

    Optionally serializes windows to a newline-delimited JSON file. Keeping
    the serialization here (rather than in the caller) prevents drift between
    the dataclass schema and the on-disk format consumed by the training
    loader.

    Args:
        sample_id: VCF column to extract genotypes for.
        reference_fasta: Path to the reference FASTA (plain or gzipped).
        sample_vcf: Path to the input VCF (plain or gzipped).
        output_jsonl: Optional path to write one JSON record per window.

    Returns:
        All windows extracted for ``sample_id``, in VCF record order.
    """
    contig_sequences = _read_fasta_sequences(Path(reference_fasta))
    windows = extract_locus_windows_from_vcf(
        sample_id=sample_id,
        sample_vcf=Path(sample_vcf),
        contig_sequences=contig_sequences,
    )
    if output_jsonl is not None:
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for window in windows:
                handle.write(json.dumps(asdict(window)) + "\n")
    return windows
