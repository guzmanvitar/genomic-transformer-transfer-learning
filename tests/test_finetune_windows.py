"""Unit tests for ``jaguar_geo_assign.data.finetune_windows``.

These tests pin down the behavioral contract that distinguishes this pipeline
from ``consensus.py``: heterozygotes are doubled into two allele-specific
windows (instead of being masked), homozygous-reference loci are dropped
(instead of being preserved as reference), and edge cases that cannot be
represented as a clean single-base substitution into a fixed-width window
(boundary loci, indels, multi-allelics, filtered, no-call) are skipped rather
than being silently truncated or padded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaguar_geo_assign.data.acquisition import AcquisitionError
from jaguar_geo_assign.data.consensus import MalformedGenotypeError
from jaguar_geo_assign.data.finetune_windows import (
    DOWNSTREAM_BASES,
    UPSTREAM_BASES,
    WINDOW_SIZE,
    FinetuneWindow,
    extract_fasta_window,
    extract_fasta_windows_for_sample,
    extract_locus_windows_from_vcf,
)

# A reference long enough that a 1-based locus at the center has 256bp of
# upstream and 255bp of downstream context within bounds, plus extra margin
# so we can also test off-center loci near the boundary.
_CONTIG_LENGTH = 1024


def _make_contig_sequence(length: int = _CONTIG_LENGTH, fill: str = "A") -> str:
    """Build a deterministic contig sequence for window-extraction tests.

    The sequence uses a single repeated base so that a flank of all ``fill``
    characters is unambiguously distinguishable from the substituted center
    base, which makes assertions on substitution position trivial.
    """
    return fill * length


def test_window_size_constants_sum_to_total_window():
    """Guard the load-bearing arithmetic: 256 + 1 + 255 must equal 512."""
    assert UPSTREAM_BASES + 1 + DOWNSTREAM_BASES == WINDOW_SIZE


def test_extract_fasta_window_produces_512bp_centered_window():
    """A locus comfortably away from boundaries yields exactly 512 bp with the allele at center."""
    sequence = _make_contig_sequence(fill="A")
    locus_pos = 300  # 1-based; idx 299 is well within bounds for a 1024-base contig
    result = extract_fasta_window(contig_sequence=sequence, locus_pos=locus_pos, allele="T")
    assert result is not None
    window, window_start, window_end = result
    assert len(window) == WINDOW_SIZE
    assert window_end - window_start == WINDOW_SIZE
    # Locus base sits at position UPSTREAM_BASES (0-based) within the window.
    assert window[UPSTREAM_BASES] == "T"
    # All flanks come from the homogeneous reference fill.
    assert set(window[:UPSTREAM_BASES]) == {"A"}
    assert set(window[UPSTREAM_BASES + 1 :]) == {"A"}
    # Coordinates round-trip to the VCF locus position.
    assert window_start == (locus_pos - 1) - UPSTREAM_BASES
    assert window_end == locus_pos + DOWNSTREAM_BASES


def test_extract_fasta_window_uppercases_soft_masked_reference():
    """Soft-masked (lowercase) reference flanks must be normalized so DNABERT-2 sees A/C/G/T."""
    sequence = "acgt" * 256  # length 1024, all lowercase
    result = extract_fasta_window(contig_sequence=sequence, locus_pos=300, allele="g")
    assert result is not None
    window, _, _ = result
    assert window.isupper()
    assert window[UPSTREAM_BASES] == "G"


@pytest.mark.parametrize("locus_pos", [1, UPSTREAM_BASES, _CONTIG_LENGTH - DOWNSTREAM_BASES + 1])
def test_extract_fasta_window_returns_none_when_window_extends_past_boundary(locus_pos):
    """Boundary-adjacent loci must return None instead of being silently padded with N."""
    sequence = _make_contig_sequence()
    assert extract_fasta_window(contig_sequence=sequence, locus_pos=locus_pos, allele="T") is None


def test_extract_fasta_window_rejects_multi_base_alleles():
    """Indels would shift flank coordinates; the function must refuse them loudly."""
    sequence = _make_contig_sequence()
    with pytest.raises(ValueError, match="single-base"):
        extract_fasta_window(contig_sequence=sequence, locus_pos=300, allele="AT")


def _build_vcf(records: list[str], sample_id: str = "cat_1") -> str:
    """Return a minimal VCF text with the supplied data records.

    The header is intentionally minimal: ``finetune_windows`` does not perform
    reference-build validation (that is consensus.py's job), so we only need
    a syntactically valid ``#CHROM`` row. Avoiding ``textwrap.dedent`` here
    keeps multi-record bodies aligned at column zero regardless of caller
    indentation.
    """
    header_lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=chr1,length={_CONTIG_LENGTH}>",
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_id}",
    ]
    return "\n".join([*header_lines, *records, ""])


def _write_fixture(tmp_path: Path, vcf_text: str, sequence: str | None = None) -> tuple[Path, Path]:
    """Materialize VCF + FASTA fixture files and return their paths."""
    vcf = tmp_path / "sample.vcf"
    vcf.write_text(vcf_text, encoding="utf-8")
    fasta = tmp_path / "ref.fa"
    fasta.write_text(f">chr1\n{sequence or _make_contig_sequence()}\n", encoding="utf-8")
    return fasta, vcf


def test_homozygous_alternate_produces_one_window_with_alt_substituted(tmp_path: Path):
    """1/1 locus emits exactly one window with ALT placed at the center."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    windows = extract_fasta_windows_for_sample(
        sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf
    )
    assert len(windows) == 1
    only = windows[0]
    assert only.is_heterozygous is False
    assert only.alt_allele == "T"
    assert only.ref_allele == "A"
    assert only.sequence[UPSTREAM_BASES] == "T"
    assert len(only.sequence) == WINDOW_SIZE


def test_heterozygous_locus_produces_two_windows_one_per_allele(tmp_path: Path):
    """0/1 locus emits ref-allele copy first, then alt-allele copy; both flagged heterozygous."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0/1"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    windows = extract_fasta_windows_for_sample(
        sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf
    )
    assert len(windows) == 2
    ref_window, alt_window = windows
    assert ref_window.is_heterozygous is True
    assert alt_window.is_heterozygous is True
    assert ref_window.alt_allele == "A" and ref_window.sequence[UPSTREAM_BASES] == "A"
    assert alt_window.alt_allele == "T" and alt_window.sequence[UPSTREAM_BASES] == "T"
    # Flanks must be identical between the two copies of a het pair.
    assert ref_window.sequence[:UPSTREAM_BASES] == alt_window.sequence[:UPSTREAM_BASES]
    assert ref_window.sequence[UPSTREAM_BASES + 1 :] == alt_window.sequence[UPSTREAM_BASES + 1 :]
    assert ref_window.genotype == "0/1" and alt_window.genotype == "0/1"


def test_phased_heterozygote_is_handled_identically_to_unphased(tmp_path: Path):
    """Phased GT (``0|1``) must produce the same two-window output as ``0/1``."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0|1"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    windows = extract_fasta_windows_for_sample(
        sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf
    )
    assert [w.alt_allele for w in windows] == ["A", "T"]
    assert all(w.is_heterozygous for w in windows)


@pytest.mark.parametrize(
    ("record", "skip_reason"),
    [
        ("chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0/0", "homozygous_reference"),
        ("chr1\t300\t.\tA\tAT\t.\tPASS\t.\tGT\t1/1", "indel_insertion"),
        ("chr1\t300\t.\tAT\tA\t.\tPASS\t.\tGT\t1/1", "indel_deletion"),
        ("chr1\t300\t.\tA\tT,G\t.\tPASS\t.\tGT\t1/2", "multiallelic_het"),
        ("chr1\t300\t.\tA\tT,G\t.\tPASS\t.\tGT\t1/1", "multiallelic_homo"),
        ("chr1\t300\t.\tA\tT\t.\tLowQual\t.\tGT\t1/1", "filtered"),
        ("chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t./.", "no_call"),
        ("chr1\t300\t.\tA\t.\t.\tPASS\t.\tGT\t1/1", "monomorphic_alt_dot"),
        ("chr1\t1\t.\tA\tT\t.\tPASS\t.\tGT\t1/1", "boundary_start"),
        (f"chr1\t{_CONTIG_LENGTH}\t.\tA\tT\t.\tPASS\t.\tGT\t1/1", "boundary_end"),
    ],
)
def test_records_that_cannot_form_clean_single_base_window_are_skipped(
    tmp_path: Path, record: str, skip_reason: str
):
    """Each record represents a category that must not produce a window.

    The ``skip_reason`` parameter is used purely to label parametrize cases in
    pytest output so failures point directly at the offending category.
    """
    del skip_reason
    vcf_text = _build_vcf([record])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    windows = extract_fasta_windows_for_sample(
        sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf
    )
    assert windows == []


def test_record_on_unknown_contig_is_skipped_without_error(tmp_path: Path):
    """Records on contigs absent from the reference are silently skipped."""
    vcf_text = _build_vcf(["chrZ\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    fasta.write_text(f">chr1\n{_make_contig_sequence()}\n", encoding="utf-8")
    windows = extract_fasta_windows_for_sample(
        sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf
    )
    assert windows == []


def test_missing_sample_raises_acquisition_error(tmp_path: Path):
    """An unknown sample column must fail loudly rather than silently returning no windows."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"], sample_id="cat_1")
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(AcquisitionError, match="not found in VCF"):
        extract_fasta_windows_for_sample(sample_id="cat_99", reference_fasta=fasta, sample_vcf=vcf)


def test_malformed_genotype_propagates_from_consensus_validator(tmp_path: Path):
    """Reusing consensus.py's GT validator means malformed tokens raise the same error type."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/?"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(MalformedGenotypeError):
        extract_fasta_windows_for_sample(sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf)


def test_mixed_record_set_emits_expected_window_counts_in_vcf_order(tmp_path: Path):
    """A realistic VCF with one het, one hom-alt, one hom-ref must emit 2 + 1 + 0 = 3 windows."""
    vcf_text = _build_vcf(
        [
            "chr1\t260\t.\tA\tT\t.\tPASS\t.\tGT\t0/1",  # het → 2 windows
            "chr1\t400\t.\tA\tG\t.\tPASS\t.\tGT\t1/1",  # hom-alt → 1 window
            "chr1\t500\t.\tA\tC\t.\tPASS\t.\tGT\t0/0",  # hom-ref → 0 windows
        ]
    )
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    windows = extract_fasta_windows_for_sample(
        sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf
    )
    assert [(w.locus_pos, w.alt_allele, w.is_heterozygous) for w in windows] == [
        (260, "A", True),
        (260, "T", True),
        (400, "G", False),
    ]
    assert all(len(w.sequence) == WINDOW_SIZE for w in windows)


def test_extract_locus_windows_uses_provided_contig_sequences_directly(tmp_path: Path):
    """Bypassing FASTA loading lets callers reuse a cached reference across many samples."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"])
    _, vcf = _write_fixture(tmp_path, vcf_text)
    contig_sequences = {"chr1": _make_contig_sequence(fill="C")}
    windows = extract_locus_windows_from_vcf(
        sample_id="cat_1", sample_vcf=vcf, contig_sequences=contig_sequences
    )
    assert len(windows) == 1
    # Flanks pick up the C-fill we passed, not the default A-fill from disk.
    assert set(windows[0].sequence[:UPSTREAM_BASES]) == {"C"}
    assert windows[0].sequence[UPSTREAM_BASES] == "T"


def test_output_jsonl_round_trips_window_records(tmp_path: Path):
    """JSONL output must serialize every dataclass field so the training loader can reconstruct
    windows."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0/1"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    output = tmp_path / "windows.jsonl"
    windows = extract_fasta_windows_for_sample(
        sample_id="cat_1",
        reference_fasta=fasta,
        sample_vcf=vcf,
        output_jsonl=output,
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(windows) == 2
    parsed = [json.loads(line) for line in lines]
    expected_keys = {f for f in FinetuneWindow.__dataclass_fields__}
    for record, window in zip(parsed, windows, strict=True):
        assert set(record.keys()) == expected_keys
        assert record["sequence"] == window.sequence
        assert record["locus_pos"] == window.locus_pos
        assert record["is_heterozygous"] == window.is_heterozygous
