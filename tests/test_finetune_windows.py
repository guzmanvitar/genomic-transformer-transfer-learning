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
from jaguar_geo_assign.data.consensus import (
    ContigMismatchError,
    MalformedGenotypeError,
    ReferenceMismatchError,
)
from jaguar_geo_assign.data.finetune_windows import (
    DOWNSTREAM_BASES,
    UPSTREAM_BASES,
    WINDOW_SIZE,
    FinetuneWindow,
    InvalidAlleleAlphabetError,
    PloidyError,
    ReferenceBaseMismatchError,
    extract_fasta_window,
    extract_fasta_windows_for_sample,
    extract_locus_windows_from_vcf,
    load_reference_index,
)

# Build-token override used throughout the unit-test suite. The strict
# data-contract guards in finetune_windows.py require both the FASTA
# evidence and the VCF ##reference header to mention every expected
# token; we therefore embed this synthetic token in both the test FASTA
# filename and the VCF ``##reference=`` line so production validation
# runs (instead of being bypassed) but does so against a deterministic,
# locally-controlled identifier rather than the live jaguar build name.
_TEST_BUILD_TOKENS = ("TEST_BUILD_v1",)

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


def _build_vcf(
    records: list[str],
    sample_id: str = "cat_1",
    *,
    contigs: tuple[str, ...] = ("chr1",),
    reference_token: str | None = _TEST_BUILD_TOKENS[0],
) -> str:
    """Return a minimal VCF text with the supplied data records.

    Header injection contract: the production guards require both a
    ``##reference=`` line containing every expected build token and at
    least one ``##contig=<ID=...>`` per declared contig. ``reference_token``
    is parameterised so individual tests can simulate a missing or
    mismatched header by passing ``None`` or an unrelated string.
    """
    header_lines = ["##fileformat=VCFv4.2"]
    if reference_token is not None:
        header_lines.append(f"##reference={reference_token}")
    for contig in contigs:
        header_lines.append(f"##contig=<ID={contig},length={_CONTIG_LENGTH}>")
    header_lines.append(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_id}")
    return "\n".join([*header_lines, *records, ""])


def _write_fixture(
    tmp_path: Path,
    vcf_text: str,
    sequence: str | None = None,
    *,
    contig_name: str = "chr1",
    fasta_filename: str = f"ref.{_TEST_BUILD_TOKENS[0]}.fa",
) -> tuple[Path, Path]:
    """Materialize VCF + FASTA fixture files and return their paths.

    The default ``fasta_filename`` embeds the test build token so the
    production reference-validation guard finds matching evidence in the
    filename (consensus.py's check inspects both filename and headers).
    """
    vcf = tmp_path / "sample.vcf"
    vcf.write_text(vcf_text, encoding="utf-8")
    fasta = tmp_path / fasta_filename
    fasta.write_text(f">{contig_name}\n{sequence or _make_contig_sequence()}\n", encoding="utf-8")
    return fasta, vcf


def _extract(fasta: Path, vcf: Path, *, sample_id: str = "cat_1", **kwargs):
    """Thin wrapper threading ``_TEST_BUILD_TOKENS`` through every call.

    Centralised so individual tests do not have to repeat the kwargs
    dance for every invocation; tests that want to *exercise* the
    build-token guard call :func:`extract_fasta_windows_for_sample`
    directly without this helper.
    """
    return extract_fasta_windows_for_sample(
        sample_id=sample_id,
        reference_fasta=fasta,
        sample_vcf=vcf,
        positive_reference_tokens=_TEST_BUILD_TOKENS,
        **kwargs,
    )


def test_homozygous_alternate_produces_one_window_with_alt_substituted(tmp_path: Path):
    """1/1 locus emits exactly one window with ALT placed at the center."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    windows = _extract(fasta, vcf)
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
    windows = _extract(fasta, vcf)
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
    windows = _extract(fasta, vcf)
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
    windows = _extract(fasta, vcf)
    assert windows == []


def test_record_on_unknown_contig_raises_contig_mismatch(tmp_path: Path):
    """Per the strict data contract, unknown record-level CHROM must raise.

    Earlier versions silently dropped such records, hiding mis-aligned
    VCF/FASTA pairs. The fail-fast guard makes that drift visible at
    the first offending record.
    """
    vcf_text = _build_vcf(["chrZ\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"], contigs=("chr1",))
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(ContigMismatchError, match="chrZ"):
        _extract(fasta, vcf)


def test_missing_sample_raises_acquisition_error(tmp_path: Path):
    """An unknown sample column must fail loudly rather than silently returning no windows."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"], sample_id="cat_1")
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(AcquisitionError, match="not found in VCF"):
        _extract(fasta, vcf, sample_id="cat_99")


def test_malformed_genotype_propagates_from_consensus_validator(tmp_path: Path):
    """Reusing consensus.py's GT validator means malformed tokens raise the same error type."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/?"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(MalformedGenotypeError):
        _extract(fasta, vcf)


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
    windows = _extract(fasta, vcf)
    assert [(w.locus_pos, w.alt_allele, w.is_heterozygous) for w in windows] == [
        (260, "A", True),
        (260, "T", True),
        (400, "G", False),
    ]
    assert all(len(w.sequence) == WINDOW_SIZE for w in windows)


def test_extract_locus_windows_uses_provided_contig_sequences_directly(tmp_path: Path):
    """Bypassing FASTA loading lets callers reuse a cached reference across many samples.

    The in-memory contig is built from a homogeneous ``C``-fill so the
    flanks are unambiguously distinguishable from any default disk fill.
    The VCF ``REF`` matches the fill base because the new
    REF-vs-reference guard raises on disagreement.
    """
    vcf_text = _build_vcf(["chr1\t300\t.\tC\tT\t.\tPASS\t.\tGT\t1/1"])
    _, vcf = _write_fixture(tmp_path, vcf_text)
    contig_sequences = {"chr1": _make_contig_sequence(fill="C")}
    windows = extract_locus_windows_from_vcf(
        sample_id="cat_1",
        sample_vcf=vcf,
        contig_sequences=contig_sequences,
        positive_reference_tokens=_TEST_BUILD_TOKENS,
    )
    assert len(windows) == 1
    # Flanks pick up the C-fill we passed in memory.
    assert set(windows[0].sequence[:UPSTREAM_BASES]) == {"C"}
    assert windows[0].sequence[UPSTREAM_BASES] == "T"


def test_extract_locus_windows_rejects_invalid_in_memory_reference_alphabet(tmp_path: Path):
    """The in-memory reference path must enforce the same alphabet contract as FASTA loading."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"])
    _, vcf = _write_fixture(tmp_path, vcf_text)
    contig_sequences = {"chr1": f"{'A' * 100}R{'A' * (_CONTIG_LENGTH - 101)}"}
    with pytest.raises(InvalidAlleleAlphabetError, match="invalid characters.*R"):
        extract_locus_windows_from_vcf(
            sample_id="cat_1",
            sample_vcf=vcf,
            contig_sequences=contig_sequences,
            positive_reference_tokens=_TEST_BUILD_TOKENS,
        )


def test_output_jsonl_round_trips_window_records(tmp_path: Path):
    """JSONL output must serialize every dataclass field so the training loader can reconstruct
    windows."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0/1"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    output = tmp_path / "windows.jsonl"
    windows = _extract(fasta, vcf, output_jsonl=output)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(windows) == 2
    parsed = [json.loads(line) for line in lines]
    expected_keys = {f for f in FinetuneWindow.__dataclass_fields__}
    for record, window in zip(parsed, windows, strict=True):
        assert set(record.keys()) == expected_keys
        assert record["sequence"] == window.sequence
        assert record["locus_pos"] == window.locus_pos
        assert record["is_heterozygous"] == window.is_heterozygous


def test_fasta_without_build_evidence_raises_reference_mismatch(tmp_path: Path):
    """A FASTA whose filename + headers lack every expected build token must fail loudly.

    Guards against a stale or wrong-build reference being silently
    accepted; mismatched genome builds would otherwise produce
    plausible-looking but biologically incorrect windows.
    """
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text, fasta_filename="anonymous_reference.fa")
    with pytest.raises(ReferenceMismatchError, match="missing expected positive tokens"):
        _extract(fasta, vcf)


def test_vcf_missing_reference_header_raises(tmp_path: Path):
    """A VCF without a ``##reference`` header must fail rather than be inferred.

    The fine-tuning pipeline cannot verify build alignment without an
    explicit declaration, so a missing header is a hard error.
    """
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"], reference_token=None)
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(ReferenceMismatchError, match="missing explicit reference"):
        _extract(fasta, vcf)


def test_vcf_reference_header_with_mismatched_token_raises(tmp_path: Path):
    """A ``##reference`` value that omits any expected token must raise.

    Catches the case where a VCF was called against a different build
    than the FASTA the pipeline is loading.
    """
    vcf_text = _build_vcf(
        ["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"], reference_token="OTHER_BUILD_v9"
    )
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(ReferenceMismatchError, match="failed build evidence validation"):
        _extract(fasta, vcf)


def test_header_contig_absent_from_reference_raises(tmp_path: Path):
    """A ``##contig`` declaration absent from the FASTA must fail at header time.

    Header-level mismatch is caught before any record is read so the
    failure surface is one error instead of one error per record.
    """
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"], contigs=("chr1", "chrX"))
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(ContigMismatchError, match="contigs absent"):
        _extract(fasta, vcf)


def test_reference_base_mismatch_against_fasta_raises(tmp_path: Path):
    """If the FASTA base at the locus disagrees with the VCF ``REF``, raise.

    Same identifier + different patch level is the typical real-world
    cause; the per-locus guard catches it even when build-token
    validation passes.
    """
    sequence = _make_contig_sequence(fill="A")
    vcf_text = _build_vcf(["chr1\t300\t.\tG\tT\t.\tPASS\t.\tGT\t1/1"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text, sequence=sequence)
    with pytest.raises(ReferenceBaseMismatchError, match="REF allele mismatch"):
        _extract(fasta, vcf)


@pytest.mark.parametrize(
    ("genotype", "ploidy_label"),
    [("1", "haploid"), ("0/1/1", "triploid"), ("1/1/1/1", "tetraploid")],
)
def test_non_diploid_genotype_raises_ploidy_error(tmp_path: Path, genotype: str, ploidy_label: str):
    """Any non-diploid GT must raise; the doubling logic is undefined otherwise."""
    del ploidy_label
    record = f"chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t{genotype}"
    vcf_text = _build_vcf([record])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(PloidyError, match="Non-diploid genotype"):
        _extract(fasta, vcf)


def test_missing_gt_in_format_schema_raises(tmp_path: Path):
    """A FORMAT field without ``GT`` cannot yield zygosity and must fail loudly."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS\t.\tDP\t10"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(AcquisitionError, match="missing 'GT' in FORMAT"):
        _extract(fasta, vcf)


def test_truncated_vcf_record_raises_instead_of_silent_skip(tmp_path: Path):
    """Truncated rows (fewer columns than the schema) must fail with line context."""
    vcf_text = _build_vcf(["chr1\t300\t.\tA\tT\t.\tPASS"])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(AcquisitionError):
        _extract(fasta, vcf)


@pytest.mark.parametrize(
    ("ref", "alt", "alphabet_label"),
    [
        ("A", "*", "spanning_deletion_sentinel"),
        ("A", "Y", "iupac_pyrimidine"),
        ("A", "y", "lowercase_iupac"),
        ("R", "T", "iupac_in_ref"),
        ("A", "?", "stray_symbol"),
    ],
)
def test_disallowed_allele_alphabet_raises(tmp_path: Path, ref: str, alt: str, alphabet_label: str):
    """Spanning deletions, IUPAC codes, and stray symbols must not reach the model."""
    del alphabet_label
    record = f"chr1\t300\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t1/1"
    vcf_text = _build_vcf([record])
    fasta, vcf = _write_fixture(tmp_path, vcf_text)
    with pytest.raises(InvalidAlleleAlphabetError, match="outside the allowed alphabet"):
        _extract(fasta, vcf)


def test_extract_fasta_window_rejects_disallowed_allele():
    """Direct calls with disallowed alleles must also raise (defensive guard)."""
    sequence = _make_contig_sequence()
    with pytest.raises(InvalidAlleleAlphabetError):
        extract_fasta_window(contig_sequence=sequence, locus_pos=300, allele="*")


def test_r2_positive_pass_succeeds(tmp_path: Path):
    """R2 positive-pass: load a fixture FASTA containing positive tokens -> succeeds."""
    vcf_text = _build_vcf(
        ["HiC_scaffold_1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"],
        contigs=("HiC_scaffold_1",),
        reference_token="Panthera_onca_HiC HiC_scaffold_1",
    )
    fasta, vcf = _write_fixture(
        tmp_path, vcf_text, fasta_filename="Panthera_onca_HiC.fa", contig_name="HiC_scaffold_1"
    )
    windows = extract_fasta_windows_for_sample(
        sample_id="cat_1",
        reference_fasta=fasta,
        sample_vcf=vcf,
    )
    assert len(windows) == 1


def test_r2_negative_rejection_ncbi_shape(tmp_path: Path):
    """R2 negative-rejection (NCBI shape): NC_083295.1 causes loud failure."""
    vcf_text = _build_vcf(
        ["HiC_scaffold_1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"],
        contigs=("HiC_scaffold_1",),
        reference_token="Panthera_onca_HiC HiC_scaffold_1",
    )
    fasta, vcf = _write_fixture(
        tmp_path,
        vcf_text,
        fasta_filename="Panthera_onca_HiC.fa",
        contig_name="NC_083295.1 HiC_scaffold_1",
    )
    with pytest.raises(ReferenceMismatchError, match="[Nn]egative reference token.*NC_083295.1"):
        extract_fasta_windows_for_sample(sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf)


def test_r2_negative_rejection_legacy_ncbi_accession(tmp_path: Path):
    """R2 negative-rejection (legacy NCBI accession): GCF_028533385.1 causes failure."""
    vcf_text = _build_vcf(
        ["HiC_scaffold_1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"],
        contigs=("HiC_scaffold_1",),
        reference_token="Panthera_onca_HiC HiC_scaffold_1",
    )
    fasta, vcf = _write_fixture(
        tmp_path,
        vcf_text,
        fasta_filename="GCF_028533385.1_Panthera_onca_HiC.fa",
        contig_name="HiC_scaffold_1",
    )
    with pytest.raises(ReferenceMismatchError, match="[Nn]egative reference token"):
        extract_fasta_windows_for_sample(sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf)


def test_r2_missing_positive(tmp_path: Path):
    """R2 missing-positive: missing both positive tokens -> fails."""
    vcf_text = _build_vcf(
        ["chr1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"], reference_token="some_other_build"
    )
    fasta, vcf = _write_fixture(
        tmp_path, vcf_text, fasta_filename="some_other_build.fa", contig_name="chr1"
    )
    with pytest.raises(ReferenceMismatchError, match="missing expected positive tokens"):
        extract_fasta_windows_for_sample(sample_id="cat_1", reference_fasta=fasta, sample_vcf=vcf)


def test_r4_iupac_rejection(tmp_path: Path):
    """R4 IUPAC rejection: Y or R raises at load_reference_index before iter_locus."""
    vcf_text = _build_vcf(
        ["HiC_scaffold_1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"],
        contigs=("HiC_scaffold_1",),
        reference_token="Panthera_onca_HiC HiC_scaffold_1",
    )
    fasta, vcf = _write_fixture(
        tmp_path,
        vcf_text,
        fasta_filename="Panthera_onca_HiC.fa",
        contig_name="HiC_scaffold_1",
        sequence="ACGTNRACGT",
    )
    with pytest.raises(InvalidAlleleAlphabetError, match="invalid characters.*'R'"):
        load_reference_index(fasta)


def test_r4_valid_passes(tmp_path: Path):
    """R4 valid passes: contig with only ACGTN loads fine and has validated_alphabet=True."""
    vcf_text = _build_vcf(
        ["HiC_scaffold_1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"],
        contigs=("HiC_scaffold_1",),
        reference_token="Panthera_onca_HiC HiC_scaffold_1",
    )
    fasta, _ = _write_fixture(
        tmp_path,
        vcf_text,
        fasta_filename="Panthera_onca_HiC.fa",
        contig_name="HiC_scaffold_1",
        sequence="ACGTN",
    )
    index = load_reference_index(fasta)
    assert getattr(index, "validated_alphabet", False) is True


def test_iter_locus_windows_from_vcf_signature_and_rejection(tmp_path: Path):
    import inspect

    from jaguar_geo_assign.data.finetune_windows import (
        iter_locus_windows_from_vcf,
        load_reference_index,
    )

    sig = inspect.signature(iter_locus_windows_from_vcf)
    assert "positive_reference_tokens" in sig.parameters
    assert "negative_reference_tokens" in sig.parameters

    vcf_text = _build_vcf(
        ["HiC_scaffold_1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t1/1"],
        contigs=("HiC_scaffold_1",),
        reference_token="Panthera_onca_HiC HiC_scaffold_1 NC_083295.1",
    )
    fasta, vcf = _write_fixture(
        tmp_path,
        vcf_text,
        fasta_filename="Panthera_onca_HiC.fa",
        contig_name="HiC_scaffold_1",
    )

    reference = load_reference_index(
        fasta,
        positive_reference_tokens=["Panthera_onca_HiC"],
        negative_reference_tokens=["NC_fake"],
    )

    with pytest.raises(ReferenceMismatchError, match="failed build evidence validation"):
        list(
            iter_locus_windows_from_vcf(
                sample_id="cat_1",
                sample_vcf=vcf,
                reference=reference,
                positive_reference_tokens=["Panthera_onca_HiC"],
                negative_reference_tokens=["NC_083295.1"],
            )
        )
