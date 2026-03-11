from __future__ import annotations

from pathlib import Path
import stat
import textwrap

import pytest

from jaguar_geo_assign.data.acquisition import (
    ContigMismatchError,
    MalformedGenotypeError,
    MissingToolError,
    ReferenceMismatchError,
    classify_consensus_site,
    ensure_bcftools_available,
    generate_consensus_fasta,
)


@pytest.mark.parametrize(
    ("ref", "alts", "genotype", "filter_value", "expected_action", "expected_category"),
    [
        ("A", ["T"], "0/0", "PASS", "reference", "homozygous_reference"),
        ("A", ["T"], "1/1", "PASS", "apply_alt", "homozygous_alternate"),
        ("A", ["AT"], "1/1", "PASS", "mask", "indel"),
        ("A", ["T"], "0/1", "PASS", "mask", "heterozygous"),
        ("A", ["T", "G"], "1/2", "PASS", "mask", "multiallelic"),
        ("A", ["T", "G"], "1/1", "PASS", "mask", "multiallelic"),
        ("A", ["T"], "./.", "PASS", "mask", "no_call"),
        ("A", ["T"], "1/1", "LowQual", "mask", "filtered"),
    ],
)
def test_classify_consensus_site_covers_explicit_contract(
    ref: str,
    alts: list[str],
    genotype: str,
    filter_value: str,
    expected_action: str,
    expected_category: str,
) -> None:
    decision = classify_consensus_site(ref, alts, genotype, filter_value=filter_value)

    assert decision.action == expected_action
    assert decision.category == expected_category


@pytest.mark.parametrize(
    ("genotype", "filter_value"),
    [("*/*", "PASS"), ("?/?", "LowQual"), ("1/?", "PASS")],
)
def test_classify_consensus_site_raises_actionable_error_on_malformed_gt(
    genotype: str,
    filter_value: str,
) -> None:
    with pytest.raises(MalformedGenotypeError, match=r"GT='.*'.*sample 'cat_1'.*chr1:4"):
        classify_consensus_site(
            "A",
            ["T"],
            genotype,
            filter_value=filter_value,
            sample_id="cat_1",
            contig="chr1",
            position=4,
            vcf_path=Path("fixture.vcf"),
        )


def test_generate_consensus_fasta_preserves_reference_and_masks_multiallelic_indel_and_ambiguous_calls(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(
        ">chr1 GCF_000181335.3 Felis_catus_9.0\nAACCGGAA\n",
        encoding="utf-8",
    )
    vcf = tmp_path / "cat_1.vcf"
    vcf.write_text(
        textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##reference=GCF_000181335.3_Felis_catus_9.0
            ##contig=<ID=chr1,length=8>
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_1
            chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t1/1
            chr1\t4\t.\tC\tG\t.\tPASS\t.\tGT\t0/0
            chr1\t5\t.\tG\tA,C\t.\tPASS\t.\tGT\t1/1
            chr1\t6\t.\tG\tGA\t.\tPASS\t.\tGT\t1/1
            chr1\t8\t.\tA\tG\t.\tPASS\t.\tGT\t./.
            """
        ),
        encoding="utf-8",
    )
    fake_bcftools = _write_fake_bcftools(tmp_path)

    result = generate_consensus_fasta(
        sample_id="cat_1",
        reference_fasta=reference,
        sample_vcf=vcf,
        output_fasta=tmp_path / "cat_1.fa",
        bcftools_executable=str(fake_bcftools),
    )

    assert result.output_fasta.read_text(encoding="utf-8") == ">chr1\nATCCNNAN\n"
    assert result.diagnostics.applied_variant_count == 1
    assert result.diagnostics.masked_site_count == 3
    assert result.diagnostics.filtered_or_nocall_count == 1
    assert result.diagnostics.callable_records == 2
    assert result.diagnostics.identical_to_reference_calls == 1
    assert result.diagnostics.indel_count == 1
    assert [(span.start, span.end, span.category) for span in result.mask_spans] == [
        (4, 5, "multiallelic"),
        (5, 6, "indel"),
        (7, 8, "no_call"),
    ]


@pytest.mark.parametrize("genotype", ["*/*", "?/?"])
def test_generate_consensus_fasta_fails_fast_on_malformed_gt_tokens(tmp_path: Path, genotype: str) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(
        ">chr1 GCF_000181335.3 Felis_catus_9.0\nAACCAA\n",
        encoding="utf-8",
    )
    vcf = tmp_path / "cat_1.vcf"
    vcf.write_text(
        textwrap.dedent(
            f"""\
            ##fileformat=VCFv4.2
            ##reference=GCF_000181335.3_Felis_catus_9.0
            ##contig=<ID=chr1,length=6>
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_1
            chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t{genotype}
            """
        ),
        encoding="utf-8",
    )
    fake_bcftools = _write_fake_bcftools(tmp_path)

    with pytest.raises(MalformedGenotypeError, match=r"GT='.*'.*sample 'cat_1'.*chr1:2"):
        generate_consensus_fasta(
            sample_id="cat_1",
            reference_fasta=reference,
            sample_vcf=vcf,
            output_fasta=tmp_path / "cat_1.fa",
            bcftools_executable=str(fake_bcftools),
        )


@pytest.mark.parametrize(
    ("reference_header", "match"),
    [
        (None, "missing explicit reference/build metadata"),
        ("##reference=GCF_000181335.3", "does not canonically match expected build evidence"),
        ("##reference=Felis_catus_9.0", "does not canonically match expected build evidence"),
        ("##reference=GCF_000181335.3_GRCh38", "does not canonically match expected build evidence"),
        ("##reference=GRCh38", "does not canonically match expected build evidence"),
    ],
)
def test_generate_consensus_fasta_requires_explicit_matching_reference_metadata(
    tmp_path: Path,
    reference_header: str | None,
    match: str,
) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(
        ">chr1 GCF_000181335.3 Felis_catus_9.0\nAACCAA\n",
        encoding="utf-8",
    )
    vcf = tmp_path / "cat_1.vcf"
    header_lines = [
        "##fileformat=VCFv4.2",
        *( [reference_header] if reference_header is not None else [] ),
        "##contig=<ID=chr1,length=6>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_1",
        "chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t1/1",
    ]
    vcf.write_text("\n".join(header_lines) + "\n", encoding="utf-8")
    fake_bcftools = _write_fake_bcftools(tmp_path)

    with pytest.raises(ReferenceMismatchError, match=match):
        generate_consensus_fasta(
            sample_id="cat_1",
            reference_fasta=reference,
            sample_vcf=vcf,
            output_fasta=tmp_path / "cat_1.fa",
            bcftools_executable=str(fake_bcftools),
        )


@pytest.mark.parametrize(
    ("reference_name", "reference_header"),
    [
        ("GCF_000181335.3_only.fa", ">chr1\nunrelated\nAACCAA\n"),
        ("reference.fa", ">chr1 Felis_catus_9.0\nAACCAA\n"),
        ("reference.fa", ">chr1 GCF_000181335.3 Felis_catus_8.0\nAACCAA\n"),
    ],
)
def test_generate_consensus_fasta_rejects_partial_or_inconsistent_fasta_build_evidence(
    tmp_path: Path,
    reference_name: str,
    reference_header: str,
) -> None:
    reference = tmp_path / reference_name
    reference.write_text(reference_header, encoding="utf-8")
    vcf = tmp_path / "cat_1.vcf"
    vcf.write_text(
        textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##reference=GCF_000181335.3_Felis_catus_9.0
            ##contig=<ID=chr1,length=6>
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_1
            chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t1/1
            """
        ),
        encoding="utf-8",
    )
    fake_bcftools = _write_fake_bcftools(tmp_path)

    with pytest.raises(ReferenceMismatchError, match="does not canonically match expected build evidence"):
        generate_consensus_fasta(
            sample_id="cat_1",
            reference_fasta=reference,
            sample_vcf=vcf,
            output_fasta=tmp_path / "cat_1.fa",
            bcftools_executable=str(fake_bcftools),
        )


def test_generate_consensus_fasta_fails_fast_on_contig_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(
        ">chr1 GCF_000181335.3 Felis_catus_9.0\nAACCAA\n",
        encoding="utf-8",
    )
    vcf = tmp_path / "cat_1.vcf"
    vcf.write_text(
        textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##reference=GCF_000181335.3_Felis_catus_9.0
            ##contig=<ID=chr2,length=6>
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_1
            chr2\t2\t.\tA\tT\t.\tPASS\t.\tGT\t1/1
            """
        ),
        encoding="utf-8",
    )
    fake_bcftools = _write_fake_bcftools(tmp_path)

    with pytest.raises(ContigMismatchError, match="contigs absent"):
        generate_consensus_fasta(
            sample_id="cat_1",
            reference_fasta=reference,
            sample_vcf=vcf,
            output_fasta=tmp_path / "cat_1.fa",
            bcftools_executable=str(fake_bcftools),
        )


def test_ensure_bcftools_available_raises_actionable_error() -> None:
    with pytest.raises(MissingToolError, match="install bcftools"):
        ensure_bcftools_available("bcftools-does-not-exist")


def _write_fake_bcftools(tmp_path: Path) -> Path:
    script = tmp_path / "fake_bcftools.py"
    script.write_text(
        (
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "def read_fasta(text: str):\n"
            "    name = None\n"
            "    seq = []\n"
            "    out = {}\n"
            "    for line in text.splitlines():\n"
            "        if line.startswith('>'):\n"
            "            if name is not None:\n"
            "                out[name] = list(''.join(seq))\n"
            "            name = line[1:].split()[0]\n"
            "            seq = []\n"
            "        else:\n"
            "            seq.append(line.strip())\n"
            "    if name is not None:\n"
            "        out[name] = list(''.join(seq))\n"
            "    return out\n\n"
            "args = sys.argv[1:]\n"
            "if not args or args[0] != 'consensus':\n"
            "    sys.stderr.write('expected consensus command')\n"
            "    sys.exit(2)\n"
            "sample = None\n"
            "mask_path = None\n"
            "header = None\n"
            "index = 1\n"
            "while index < len(args) - 1:\n"
            "    if args[index] == '-s':\n"
            "        sample = args[index + 1]\n"
            "        index += 2\n"
            "        continue\n"
            "    if args[index] == '-m':\n"
            "        mask_path = Path(args[index + 1])\n"
            "        index += 2\n"
            "        continue\n"
            "    index += 1\n"
            "vcf_path = Path(args[-1])\n"
            "sequences = read_fasta(sys.stdin.read())\n"
            "if mask_path and mask_path.exists():\n"
            "    for line in mask_path.read_text(encoding='utf-8').splitlines():\n"
            "        chrom, start, end = line.split('\\t')\n"
            "        for position in range(int(start), int(end)):\n"
            "            sequences[chrom][position] = 'N'\n"
            "for line in vcf_path.read_text(encoding='utf-8').splitlines():\n"
            "    if line.startswith('#'):\n"
            "        header = line.split('\\t') if line.startswith('#CHROM') else header\n"
            "        continue\n"
            "    fields = line.split('\\t')\n"
            "    chrom, pos, _id, ref, alt_field, _qual, filt, _info, fmt = fields[:9]\n"
            "    sample_fields = dict(zip(fmt.split(':'), fields[header.index(sample)].split(':'), strict=False))\n"
            "    gt = sample_fields.get('GT')\n"
            "    if filt not in {'PASS', '.'} or gt is None:\n"
            "        continue\n"
            "    alleles = gt.replace('|', '/').split('/')\n"
            "    if len(set(alleles)) != 1 or alleles[0] in {'0', '.'}:\n"
            "        continue\n"
            "    alt = alt_field.split(',')[int(alleles[0]) - 1]\n"
            "    position = int(pos) - 1\n"
            "    sequences[chrom][position : position + len(ref)] = list(alt)\n"
            "for chrom, seq in sequences.items():\n"
            "    sys.stdout.write('>' + chrom + '\\n' + ''.join(seq) + '\\n')\n"
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script