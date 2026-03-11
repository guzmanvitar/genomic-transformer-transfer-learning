from pathlib import Path
import json
import stat
import textwrap

import pytest

from jaguar_geo_assign.cli import main
from jaguar_geo_assign.data.acquisition import ConsensusDiagnostics, ConsensusResult
from jaguar_geo_assign.data import preprocessor as preprocessor_module
from jaguar_geo_assign.data.preprocessor import ExportContractError, TokenizedWindow, TokenizerProvenance, WindowRecord
from jaguar_geo_assign.pretrain import pipeline as pretrain_pipeline


def test_validate_config_reports_success(capsys) -> None:
    exit_code = main(["validate-config", "configs/examples/fine_tune.toml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "is valid" in captured.out


def test_describe_experiment_reports_deferred_baseline(capsys) -> None:
    exit_code = main(["describe-experiment", "configs/examples/regression_transfer.toml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Deferred baseline: baseline_evaluate -> deferred_legacy_group_model" in captured.out


def test_stage_entry_points_accept_optional_config(capsys) -> None:
    config_path = Path("configs/examples/regression_transfer.toml")

    exit_code = main(["baseline-evaluate", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "baseline-evaluate entry point scaffold is available" in captured.out
    assert "Deferred baseline stage is reserved for baseline_evaluate" in captured.out
    assert "Loaded config: regression_transfer_bootstrap" in captured.out


def test_validate_feline_config_reports_success(capsys) -> None:
    exit_code = main(["validate-feline-config", "configs/examples/feline_pretrain.toml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "matches the approved contract" in captured.out


def test_describe_feline_config_reports_split_contract(capsys) -> None:
    exit_code = main(["describe-feline-config", "configs/examples/feline_pretrain.toml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Split contract: global_locus_block via contig, block_id" in captured.out
    assert "Tokenizer: zhihan1996/DNABERT-2-117M@7bce263b15377fc15361f52cfab88f8b586abda0" in captured.out


def test_check_feline_runtime_reports_missing_tool(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)

    exit_code = main(["check-feline-runtime", "configs/examples/feline_pretrain.toml"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Missing required external tools" in captured.out


def test_pretrain_cli_smoke_path_runs_fixture_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(
        ">chr1 GCF_000181335.3 Felis_catus_9.0\nAACCAA\n",
        encoding="utf-8",
    )
    sample_vcf = tmp_path / "cat_1.vcf"
    sample_vcf.write_text(
        textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##reference=GCF_000181335.3_Felis_catus_9.0
            ##contig=<ID=chr1,length=6>
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_1
            chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t1/1
            chr1\t4\t.\tC\tG\t.\tPASS\t.\tGT\t0/1
            chr1\t6\t.\tA\tG\t.\tPASS\t.\tGT\t./.
            """
        ),
        encoding="utf-8",
    )
    sample_manifest = tmp_path / "sample_manifest.tsv"
    sample_manifest.write_text(
        f"sample_id\tindividual_id\tvcf_path\ncat_1\tcat-1\t{sample_vcf}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "feline_smoke.toml"
    config_path.write_text(
        textwrap.dedent(
            f"""\
            [pipeline]
            name = "feline_smoke"
            description = "Fixture-backed feline pipeline smoke test."
            project_accession = "PRJNA308208"

            [paths]
            reference_fasta = "{reference}"
            sample_manifest = "{sample_manifest}"
            source_vcf = "{sample_vcf}"
            raw_dir = "{tmp_path / 'raw'}"
            processed_dir = "{tmp_path / 'processed'}"
            baseline_dir = "{tmp_path / 'baseline'}"
            artifact_dir = "{tmp_path / 'artifacts'}"
            report_dir = "{tmp_path / 'reports'}"

            [consensus]
            assembly = "Felis_catus_9.0"
            require_assembly_match = true
            require_contig_match = true
            mask_symbol = "N"
            homozygous_reference = "emit_reference_if_callable"
            homozygous_alternate = "apply_alternate_allele"
            heterozygous = "mask_and_report"
            multiallelic = "mask_and_report"
            filtered = "mask_and_report"
            missing = "mask_and_report"
            indel = "mask_and_report"

            [windowing]
            context_window = 6
            window_overlap = 0
            max_ambiguous_fraction = 0.5
            drop_short_sequences = true

            [split]
            strategy = "global_locus_block"
            locus_key_fields = ["contig", "block_id"]
            locus_block_size = 6
            assignment_stage = "before_windowing"
            evaluation_target = "unseen_loci"
            baseline_policy = "reuse_locus_assignments"

            [tokenizer]
            identifier = "zhihan1996/DNABERT-2-117M"
            revision = "7bce263b15377fc15361f52cfab88f8b586abda0"
            allowed_alphabet = ["A", "C", "G", "T", "N"]
            unsupported_symbol_policy = "reject"
            max_position_embeddings = 8

            [export]
            format = "parquet"
            access_pattern = "offline_window_materialization"
            row_group_size = 2
            deterministic_partition_keys = ["split", "contig", "block_id"]
            preserve_raw_windows = false
            preserve_sequence_hashes = true
            preserve_coordinates = true
            sequence_hash_algorithm = "sha256"

            [runtime]
            external_tools = ["bcftools"]
            """
        ),
        encoding="utf-8",
    )

    class FakeTokenizer:
        def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
            return {
                "input_ids": [101, *range(200, 200 + len(sequence)), 102],
                "attention_mask": [1] * (len(sequence) + 2),
            }

    def fake_tokenizer_loader() -> tuple[object, TokenizerProvenance]:
        return (
            FakeTokenizer(),
            TokenizerProvenance(max_position_embeddings=8),
        )

    monkeypatch.setattr(pretrain_pipeline, "load_dnabert2_tokenizer", fake_tokenizer_loader)
    monkeypatch.setattr(preprocessor_module, "_load_pyarrow_parquet", _fake_pyarrow_parquet_backend)

    fake_bcftools = _write_fake_bcftools(tmp_path)
    exit_code = main(
        [
            "pretrain",
            "--config",
            str(config_path),
            "--bcftools-executable",
            str(fake_bcftools),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Feline pretrain artifact generation finished for 'feline_smoke'." in captured.out
    consensus_fasta = tmp_path / "processed" / "consensus_fastas" / "cat_1.fa"
    assert consensus_fasta.read_text(encoding="utf-8") == ">chr1\nATCNAN\n"
    consensus_metadata = json.loads(
        (tmp_path / "processed" / "consensus_tokens" / "metadata.json").read_text(encoding="utf-8")
    )
    baseline_metadata = json.loads(
        (tmp_path / "baseline" / "reference_tokens" / "metadata.json").read_text(encoding="utf-8")
    )
    assert consensus_metadata["export_format"] == "parquet"
    assert baseline_metadata["export_format"] == "parquet"
    consensus_files = [
        tmp_path / "processed" / "consensus_tokens" / relative_path
        for split in consensus_metadata["splits"].values()
        for relative_path in split["files"]
    ]
    baseline_files = [
        tmp_path / "baseline" / "reference_tokens" / relative_path
        for split in baseline_metadata["splits"].values()
        for relative_path in split["files"]
    ]
    assert consensus_files and all(path.suffix == ".parquet" and path.exists() for path in consensus_files)
    assert baseline_files and all(path.suffix == ".parquet" and path.exists() for path in baseline_files)
    exported_consensus_rows = json.loads(consensus_files[0].read_text(encoding="utf-8"))["rows"]
    assert exported_consensus_rows[0]["window"]["sequence_hash"]
    assert "sequence" not in exported_consensus_rows[0]["window"]
    diagnostics = json.loads((tmp_path / "reports" / "eda_payload.json").read_text(encoding="utf-8"))
    assert diagnostics["consensus_generation"]["cat_1"]["applied_variant_count"] == 1
    assert diagnostics["consensus_sample_overview"] == {
        "total_record_count": 1,
        "returned_record_count": 1,
        "sample_limit": pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT,
        "truncated": False,
    }
    assert diagnostics["consensus_samples"][0]["filtered_bases"] == 0
    assert diagnostics["consensus_samples"][0]["no_call_bases"] == 1
    assert diagnostics["consensus_samples"][0]["other_masked_bases"] == 1
    assert diagnostics["consensus_samples"][0]["masked_base_counts"] == {
        "heterozygous": 1,
        "no_call": 1,
    }
    assert diagnostics["baseline_comparison"]["deltas"]["retained_window_count"] == 0


def test_runtime_diagnostics_payload_is_bounded_and_provenance_faithful() -> None:
    consensus_windows = tuple(_synthetic_tokenized_window(index=index, source="consensus") for index in range(192))
    baseline_windows = tuple(_synthetic_tokenized_window(index=index, source="reference") for index in range(192))

    payload = pretrain_pipeline._build_diagnostics_payload(
        tokenized_consensus=consensus_windows,
        tokenized_baseline=baseline_windows,
        consensus_results={
            "cat-1": ConsensusResult(
                sample_id="cat-1",
                output_fasta=Path("cat-1.fa"),
                diagnostics=ConsensusDiagnostics(
                    sample_id="cat-1",
                    total_records=192,
                    callable_records=96,
                    applied_variant_count=48,
                    masked_site_count=96,
                    filtered_or_nocall_count=48,
                    indel_count=0,
                    identical_to_reference_calls=48,
                    callable_fraction=0.5,
                    fraction_identical_to_reference_calls=0.25,
                ),
            )
        },
    )

    assert payload["consensus_corpus"]["retained_window_count"] == 192
    assert payload["consensus_corpus"]["source_counts"] == {"consensus": 192}
    assert len(payload["consensus_samples"]) == pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT
    assert payload["consensus_sample_overview"] == {
        "total_record_count": 192,
        "returned_record_count": pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT,
        "sample_limit": pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT,
        "truncated": True,
    }
    assert payload["consensus_samples"][1]["sequence"] == "ANCCAA"
    assert payload["consensus_samples"][1]["filtered_bases"] == 1
    assert payload["consensus_samples"][1]["no_call_bases"] == 0
    assert payload["consensus_samples"][1]["other_masked_bases"] == 0
    assert payload["consensus_samples"][1]["masked_base_counts"] == {"filtered": 1}
    assert payload["consensus_samples"][3]["sequence"] == "ANCNAA"
    assert payload["consensus_samples"][3]["filtered_bases"] == 0
    assert payload["consensus_samples"][3]["no_call_bases"] == 1
    assert payload["consensus_samples"][3]["other_masked_bases"] == 1
    assert payload["consensus_samples"][3]["masked_base_counts"] == {
        "heterozygous": 1,
        "no_call": 1,
    }


def test_pretrain_cli_reports_actionable_config_error(capsys) -> None:
    exit_code = main(["pretrain", "--config", "configs/examples/fine_tune.toml"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing required sections" in captured.out


def test_pretrain_cli_reports_actionable_parquet_dependency_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1 GCF_000181335.3 Felis_catus_9.0\nAACCAA\n", encoding="utf-8")
    sample_vcf = tmp_path / "cat_1.vcf"
    sample_vcf.write_text(
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
    sample_manifest = tmp_path / "sample_manifest.tsv"
    sample_manifest.write_text(
        f"sample_id\tindividual_id\tvcf_path\ncat_1\tcat-1\t{sample_vcf}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "feline_smoke.toml"
    config_path.write_text(
        textwrap.dedent(
            f"""\
            [pipeline]
            name = "feline_smoke"
            description = "Fixture-backed feline pipeline smoke test."
            project_accession = "PRJNA308208"

            [paths]
            reference_fasta = "{reference}"
            sample_manifest = "{sample_manifest}"
            source_vcf = "{sample_vcf}"
            raw_dir = "{tmp_path / 'raw'}"
            processed_dir = "{tmp_path / 'processed'}"
            baseline_dir = "{tmp_path / 'baseline'}"
            artifact_dir = "{tmp_path / 'artifacts'}"
            report_dir = "{tmp_path / 'reports'}"

            [consensus]
            assembly = "Felis_catus_9.0"
            require_assembly_match = true
            require_contig_match = true
            mask_symbol = "N"
            homozygous_reference = "emit_reference_if_callable"
            homozygous_alternate = "apply_alternate_allele"
            heterozygous = "mask_and_report"
            multiallelic = "mask_and_report"
            filtered = "mask_and_report"
            missing = "mask_and_report"
            indel = "mask_and_report"

            [windowing]
            context_window = 6
            window_overlap = 0
            max_ambiguous_fraction = 0.5
            drop_short_sequences = true

            [split]
            strategy = "global_locus_block"
            locus_key_fields = ["contig", "block_id"]
            locus_block_size = 6
            assignment_stage = "before_windowing"
            evaluation_target = "unseen_loci"
            baseline_policy = "reuse_locus_assignments"

            [tokenizer]
            identifier = "zhihan1996/DNABERT-2-117M"
            revision = "7bce263b15377fc15361f52cfab88f8b586abda0"
            allowed_alphabet = ["A", "C", "G", "T", "N"]
            unsupported_symbol_policy = "reject"
            max_position_embeddings = 8

            [export]
            format = "parquet"
            access_pattern = "offline_window_materialization"
            row_group_size = 2
            deterministic_partition_keys = ["split", "contig", "block_id"]
            preserve_raw_windows = false
            preserve_sequence_hashes = true
            preserve_coordinates = true
            sequence_hash_algorithm = "sha256"

            [runtime]
            external_tools = ["bcftools"]
            """
        ),
        encoding="utf-8",
    )

    class FakeTokenizer:
        def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
            return {
                "input_ids": [101, *range(200, 200 + len(sequence)), 102],
                "attention_mask": [1] * (len(sequence) + 2),
            }

    def fake_tokenizer_loader() -> tuple[object, TokenizerProvenance]:
        return (FakeTokenizer(), TokenizerProvenance(max_position_embeddings=8))

    monkeypatch.setattr(pretrain_pipeline, "load_dnabert2_tokenizer", fake_tokenizer_loader)
    monkeypatch.setattr(preprocessor_module, "_load_pyarrow_parquet", _raise_missing_pyarrow)

    exit_code = main(
        [
            "pretrain",
            "--config",
            str(config_path),
            "--bcftools-executable",
            str(_write_fake_bcftools(tmp_path)),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Parquet export requires pyarrow. Install with: uv add pyarrow" in captured.out


def _fake_pyarrow_parquet_backend():
    class FakeTableModule:
        @staticmethod
        def from_pylist(rows: list[dict[str, object]]) -> dict[str, object]:
            return {"rows": rows}

    class FakePyArrow:
        Table = FakeTableModule

    class FakeParquet:
        @staticmethod
        def write_table(table: dict[str, object], file_path: Path, row_group_size: int) -> None:
            Path(file_path).write_text(
                json.dumps(
                    {
                        "row_group_size": row_group_size,
                        "rows": table["rows"],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    return FakePyArrow, FakeParquet


def _raise_missing_pyarrow():
    raise ExportContractError("Parquet export requires pyarrow. Install with: uv add pyarrow")


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


def _synthetic_tokenized_window(*, index: int, source: str) -> TokenizedWindow:
    reference_sequence = "AACCAA"
    sequence = reference_sequence
    filtered_bases = 0
    no_call_bases = 0
    other_masked_bases = 0
    masked_base_counts: tuple[tuple[str, int], ...] = ()
    if source == "consensus":
        pattern = index % 4
        if pattern == 1:
            sequence = "ANCCAA"
            filtered_bases = 1
            masked_base_counts = (("filtered", 1),)
        elif pattern == 2:
            sequence = "AACNAA"
            no_call_bases = 1
            masked_base_counts = (("no_call", 1),)
        elif pattern == 3:
            sequence = "ANCNAA"
            no_call_bases = 1
            other_masked_bases = 1
            masked_base_counts = (("heterozygous", 1), ("no_call", 1))

    window = WindowRecord(
        sample_id="cat-1" if source == "consensus" else "ref-1",
        individual_id="cat-1" if source == "consensus" else "reference",
        contig="chr1",
        source=source,
        split="train" if index % 5 else "validation",
        locus_id=f"chr1:block-{index}",
        block_start=index * 6,
        block_end=(index + 1) * 6,
        window_start=index * 6,
        window_end=(index + 1) * 6,
        sequence=sequence,
        gc_fraction=0.5,
        ambiguity_fraction=sequence.count("N") / len(sequence),
        sequence_hash=f"hash-{source}-{index}",
        filtered_bases=filtered_bases,
        no_call_bases=no_call_bases,
        other_masked_bases=other_masked_bases,
        masked_base_counts=masked_base_counts,
    )
    return TokenizedWindow(
        window=window,
        input_ids=(101, 201, 202, 203, 204, 205, 206, 102),
        attention_mask=(1, 1, 1, 1, 1, 1, 1, 1),
        token_count=6,
        token_to_base_ratio=1.0,
        tokenizer=TokenizerProvenance(max_position_embeddings=8),
    )