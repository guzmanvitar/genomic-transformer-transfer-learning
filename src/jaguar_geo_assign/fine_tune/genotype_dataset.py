# ruff: noqa: F722  # jaxtyping shape annotations use string-based dimensions
"""Genotype matrix construction from jaguar VCF for geographic assignment.

This module constructs a dense genotype matrix (individuals x loci) directly
from VCF genotype calls. Unlike the DNABERT-2 embedding pathway, this
representation preserves all allelic states including homozygous reference
(0/0), which is critical for population-level geographic assignment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from beartype import beartype
from jaxtyping import Int8, jaxtyped
from torch import Tensor

from jaguar_geo_assign.data.consensus import (
    PASSING_FILTER_VALUES,
    _normalize_alt_alleles,
    _open_maybe_gzip,
    _validated_gt_tokens,
)
from jaguar_geo_assign.data.finetune_windows import ALLOWED_NUCLEOTIDES
from jaguar_geo_assign.fine_tune.dataset import _load_metadata_csv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocusInfo:
    """Metadata for a single SNP locus in the genotype matrix.

    Attributes:
        contig: Chromosome / contig name from the VCF CHROM field.
        pos: 1-based VCF position of the SNP.
        ref: Reference allele (single base).
        alt: Alternate allele (single base).
        locus_key: Unique identifier string formatted as ``"contig:pos"``.
    """

    contig: str
    pos: int
    ref: str
    alt: str
    locus_key: str


@dataclass(frozen=True)
class GenotypeMatrixResult:
    """Output of genotype matrix construction.

    Attributes:
        genotypes: Dense genotype tensor with shape
            ``(n_individuals, n_loci)``. Values are 0 (hom-ref),
            1 (het), 2 (hom-alt), or -1 (missing).
        locus_info: Per-column locus metadata, one entry per SNP.
        individual_ids: Per-row individual identifiers from the metadata CSV.
        sample_ids: Per-row sample identifiers from the VCF header.
        latitudes: Per-row decimal latitude (WGS-84).
        longitudes: Per-row decimal longitude (WGS-84).
        biome_labels: Per-row biome population labels.
    """

    genotypes: Tensor
    locus_info: list[LocusInfo]
    individual_ids: list[str]
    sample_ids: list[str]
    latitudes: list[float]
    longitudes: list[float]
    biome_labels: list[str]


def _parse_vcf_header_samples(vcf_path: Path) -> list[str]:
    """Extract sample IDs from the VCF ``#CHROM`` header line.

    Args:
        vcf_path: Path to the VCF file (plain or gzipped).

    Returns:
        List of sample identifiers from columns 9 onward.

    Raises:
        ValueError: If the VCF lacks a ``#CHROM`` header line.
    """
    with _open_maybe_gzip(vcf_path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                columns = line.rstrip("\n").split("\t")
                return columns[9:]
    raise ValueError(f"VCF {vcf_path} is missing a #CHROM header line")


def _encode_diploid_gt(gt_tokens: list[str] | None) -> int:
    """Encode validated diploid GT allele tokens as an integer genotype.

    Args:
        gt_tokens: Two-element list of digit-string allele indices from
            :func:`_validated_gt_tokens`, or ``None`` for missing calls.

    Returns:
        Integer genotype value: 0 (hom-ref ``0/0``), 1 (het ``0/1``),
        2 (hom-alt ``1/1``), or -1 (missing).

    Raises:
        ValueError: If *gt_tokens* does not contain exactly 2 elements.
    """
    if gt_tokens is None:
        return -1
    if len(gt_tokens) != 2:
        raise ValueError(
            f"Expected exactly 2 diploid allele tokens, got {len(gt_tokens)}: {gt_tokens}"
        )
    a, b = int(gt_tokens[0]), int(gt_tokens[1])
    return a + b


def build_genotype_matrix(
    vcf_path: str | Path,
    metadata_csv: str | Path,
) -> GenotypeMatrixResult:
    """Parse VCF into dense genotype matrix joined with metadata.

    The VCF is read in two conceptual passes implemented as a single streaming
    pass that accumulates per-locus genotype columns into a list-of-lists and
    stacks the result into a tensor at the end. Only samples present in both
    the VCF header and the metadata CSV are included (inner join); unmatched
    samples are logged as warnings.

    Filtering criteria (matching the existing pipeline):
        - FILTER must be in :data:`PASSING_FILTER_VALUES` (``PASS`` or ``.``).
        - REF and ALT must each be exactly 1 character in
          :data:`ALLOWED_NUCLEOTIDES`.
        - Multi-allelic sites (ALT contains ``,``) are skipped.
        - GT must be diploid (exactly 2 allele tokens).

    Args:
        vcf_path: Path to the multi-sample VCF file (plain or gzipped).
        metadata_csv: Path to the metadata CSV with columns ``sample_id``,
            ``individual_id``, ``biome_population_label``, ``latitude``,
            ``longitude``.

    Returns:
        A :class:`GenotypeMatrixResult` containing the dense genotype tensor
        and associated metadata vectors.

    Raises:
        ValueError: If no samples survive the inner join between VCF and
            metadata, or if the VCF is missing a ``#CHROM`` header.
    """
    vcf = Path(vcf_path)
    metadata = _load_metadata_csv(Path(metadata_csv))

    vcf_samples = _parse_vcf_header_samples(vcf)

    matched_vcf_indices: list[int] = []
    matched_sample_ids: list[str] = []
    matched_individual_ids: list[str] = []
    matched_latitudes: list[float] = []
    matched_longitudes: list[float] = []
    matched_biome_labels: list[str] = []

    vcf_only: list[str] = []
    for i, sample_id in enumerate(vcf_samples):
        if sample_id in metadata:
            meta = metadata[sample_id]
            matched_vcf_indices.append(i)
            matched_sample_ids.append(sample_id)
            matched_individual_ids.append(str(meta["individual_id"]))
            matched_latitudes.append(float(meta["latitude"]))
            matched_longitudes.append(float(meta["longitude"]))
            matched_biome_labels.append(str(meta["biome_population_label"]))
        else:
            vcf_only.append(sample_id)

    if vcf_only:
        logger.warning(
            "Dropping %d VCF sample(s) without metadata: %s",
            len(vcf_only),
            vcf_only[:10],
        )

    metadata_only = set(metadata.keys()) - set(vcf_samples)
    if metadata_only:
        logger.warning(
            "Metadata CSV contains %d sample(s) absent from VCF: %s",
            len(metadata_only),
            sorted(metadata_only)[:10],
        )

    if not matched_sample_ids:
        raise ValueError(
            f"No samples survived the inner join between VCF {vcf} and "
            f"metadata {metadata_csv}. Check that sample_id values match."
        )

    n_individuals = len(matched_sample_ids)
    logger.info("Matched %d individuals between VCF and metadata", n_individuals)

    locus_info_list: list[LocusInfo] = []
    genotype_columns: list[list[int]] = []

    with _open_maybe_gzip(vcf) as handle:
        for line in handle:
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            chrom = fields[0]
            pos_str = fields[1]
            ref = fields[3]
            alt_field = fields[4]
            filter_value = fields[6]
            format_field = fields[8]

            if filter_value not in PASSING_FILTER_VALUES:
                continue

            if "," in alt_field:
                continue

            alts = _normalize_alt_alleles(alt_field.split(",") if alt_field else [])
            if len(alts) != 1:
                continue
            alt = alts[0]

            if len(ref) != 1 or len(alt) != 1:
                continue
            if ref.upper() not in ALLOWED_NUCLEOTIDES or alt.upper() not in ALLOWED_NUCLEOTIDES:
                continue

            format_keys = format_field.split(":")
            if "GT" not in format_keys:
                continue
            gt_index = format_keys.index("GT")

            pos = int(pos_str)
            locus_key = f"{chrom}:{pos}"

            column: list[int] = []
            for vcf_col_offset in matched_vcf_indices:
                sample_field = fields[9 + vcf_col_offset]
                sample_values = sample_field.split(":")
                gt_raw = sample_values[gt_index] if gt_index < len(sample_values) else None

                tokens = _validated_gt_tokens(
                    gt_raw,
                    sample_id=matched_sample_ids[len(column)],
                    contig=chrom,
                    position=pos,
                    vcf_path=vcf,
                )

                if tokens is not None and len(tokens) != 2:
                    logger.debug(
                        "Non-diploid genotype at %s for sample %s — encoding as missing",
                        locus_key,
                        matched_sample_ids[len(column)],
                    )
                    tokens = None

                column.append(_encode_diploid_gt(tokens))

            locus_info_list.append(
                LocusInfo(
                    contig=chrom,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    locus_key=locus_key,
                )
            )
            genotype_columns.append(column)

    n_loci = len(locus_info_list)
    logger.info("Parsed %d biallelic SNP loci from VCF", n_loci)

    if n_loci == 0:
        genotypes = torch.zeros(n_individuals, 0, dtype=torch.int8)
    else:
        genotypes = torch.tensor(list(zip(*genotype_columns, strict=True)), dtype=torch.int8)

    return GenotypeMatrixResult(
        genotypes=genotypes,
        locus_info=locus_info_list,
        individual_ids=matched_individual_ids,
        sample_ids=matched_sample_ids,
        latitudes=matched_latitudes,
        longitudes=matched_longitudes,
        biome_labels=matched_biome_labels,
    )


def save_genotype_matrix(result: GenotypeMatrixResult, output_dir: str | Path) -> None:
    """Save genotype matrix and metadata to disk.

    Writes three files into *output_dir*:

    - ``genotypes.pt``: the dense ``Int8`` genotype tensor.
    - ``locus_metadata.json``: per-column locus information.
    - ``individual_order.json``: per-row sample/individual metadata and
      coordinates.

    Args:
        result: A :class:`GenotypeMatrixResult` to persist.
        output_dir: Directory to write into; created if it does not exist.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(result.genotypes, out / "genotypes.pt")

    locus_records = [
        {
            "contig": locus.contig,
            "pos": locus.pos,
            "ref": locus.ref,
            "alt": locus.alt,
            "locus_key": locus.locus_key,
        }
        for locus in result.locus_info
    ]
    (out / "locus_metadata.json").write_text(json.dumps(locus_records, indent=2), encoding="utf-8")

    individual_records = {
        "sample_ids": result.sample_ids,
        "individual_ids": result.individual_ids,
        "latitudes": result.latitudes,
        "longitudes": result.longitudes,
        "biome_labels": result.biome_labels,
    }
    (out / "individual_order.json").write_text(
        json.dumps(individual_records, indent=2), encoding="utf-8"
    )

    logger.info(
        "Saved genotype matrix (%d x %d) to %s",
        result.genotypes.shape[0],
        result.genotypes.shape[1],
        out,
    )


def load_genotype_matrix(output_dir: str | Path) -> GenotypeMatrixResult:
    """Load a previously saved genotype matrix from disk.

    Reads the three files written by :func:`save_genotype_matrix` and
    reconstructs a :class:`GenotypeMatrixResult`.

    Args:
        output_dir: Directory containing ``genotypes.pt``,
            ``locus_metadata.json``, and ``individual_order.json``.

    Returns:
        The reconstructed :class:`GenotypeMatrixResult`.

    Raises:
        FileNotFoundError: If any of the three expected files is missing.
    """
    out = Path(output_dir)

    genotypes = torch.load(out / "genotypes.pt", weights_only=True)

    locus_records = json.loads((out / "locus_metadata.json").read_text(encoding="utf-8"))
    locus_info = [
        LocusInfo(
            contig=rec["contig"],
            pos=rec["pos"],
            ref=rec["ref"],
            alt=rec["alt"],
            locus_key=rec["locus_key"],
        )
        for rec in locus_records
    ]

    individual_data = json.loads((out / "individual_order.json").read_text(encoding="utf-8"))

    return GenotypeMatrixResult(
        genotypes=genotypes,
        locus_info=locus_info,
        individual_ids=individual_data["individual_ids"],
        sample_ids=individual_data["sample_ids"],
        latitudes=individual_data["latitudes"],
        longitudes=individual_data["longitudes"],
        biome_labels=individual_data["biome_labels"],
    )


@jaxtyped(typechecker=beartype)
def impute_missing_genotypes(
    genotypes: Int8[Tensor, "n_individuals n_loci"],
    train_indices: list[int],
    *,
    seed: int = 42,
) -> Int8[Tensor, "n_individuals n_loci"]:
    """Impute missing genotypes (-1) using binomial draws at training allele frequencies.

    For each locus, the alternate allele frequency is computed from training
    individuals only (excluding missing values coded as -1). Each missing
    entry is then replaced with the sum of two independent Bernoulli draws
    at that frequency, producing values in {0, 1, 2}.

    When a locus has no non-missing training observations, the allele
    frequency defaults to 0.0 (imputed as homozygous reference).

    Args:
        genotypes: Dense genotype matrix with shape
            ``(n_individuals, n_loci)``. Values in {-1, 0, 1, 2}.
        train_indices: Row indices of training individuals used to
            estimate per-locus allele frequencies.
        seed: Random seed for reproducible imputation.

    Returns:
        A new tensor of the same shape with all -1 entries replaced by
        imputed values in {0, 1, 2}. Non-missing entries are unchanged.
    """
    result = genotypes.clone()
    n_individuals, n_loci = result.shape

    train_mask = torch.zeros(n_individuals, dtype=torch.bool)
    train_mask[train_indices] = True

    train_genotypes = result[train_mask]

    generator = torch.Generator()
    generator.manual_seed(seed)

    for locus_idx in range(n_loci):
        train_col = train_genotypes[:, locus_idx]
        valid_mask = train_col >= 0
        valid_calls = train_col[valid_mask]

        if valid_calls.numel() == 0:
            alt_freq = 0.0
        else:
            # Each genotype is the sum of two allele indicators (0/1),
            # so dividing by 2*N gives the per-allele frequency.
            alt_freq = valid_calls.float().sum().item() / (2.0 * valid_calls.numel())

        col = result[:, locus_idx]
        missing_mask = col < 0
        n_missing = missing_mask.sum().item()

        if n_missing > 0:
            draws = torch.bernoulli(
                torch.full((int(n_missing), 2), alt_freq),
                generator=generator,
            )
            imputed = draws.sum(dim=1).to(torch.int8)
            col[missing_mask] = imputed

    return result


__all__ = [
    "GenotypeMatrixResult",
    "LocusInfo",
    "build_genotype_matrix",
    "impute_missing_genotypes",
    "load_genotype_matrix",
    "save_genotype_matrix",
]
