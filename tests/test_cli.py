from dataclasses import replace
import gzip
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


@pytest.mark.parametrize(
    "command",
    ["validate-feline-config", "describe-feline-config", "check-feline-runtime"],
)
def test_feline_cli_inspection_commands_report_actionable_config_errors(
    command: str, capsys
) -> None:
    exit_code = main([command, "configs/examples/fine_tune.toml"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Feline pipeline config is missing required sections" in captured.out
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


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


def test_build_reference_sequence_records_deduplicates_identical_baseline_sequences() -> None:
    records = pretrain_pipeline._build_reference_sequence_records(
        {"chr1": "AACCAA", "chr2": "TTGGCC"},
        (
            pretrain_pipeline.FelineSampleManifestEntry(
                sample_id="cat_1",
                individual_id="cat-1",
                vcf_path=Path("cat_1.vcf"),
            ),
            pretrain_pipeline.FelineSampleManifestEntry(
                sample_id="cat_2",
                individual_id="cat-2",
                vcf_path=Path("cat_2.vcf"),
            ),
        ),
    )

    assert [
        (record.sample_id, record.individual_id, record.contig, record.sequence, record.source)
        for record in records
    ] == [
        ("reference-chr1", "reference", "chr1", "AACCAA", "reference"),
        ("reference-chr2", "reference", "chr2", "TTGGCC", "reference"),
    ]


def test_pretrain_pipeline_does_not_inflate_identical_reference_baseline_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(
        ">chr1 GCF_000181335.3 Felis_catus_9.0\nAACCAA\n",
        encoding="utf-8",
    )
    sample_vcf_1 = tmp_path / "cat_1.vcf"
    sample_vcf_1.write_text(
        textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##reference=GCF_000181335.3_Felis_catus_9.0
            ##contig=<ID=chr1,length=6>
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_1
            chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t0/0
            """
        ),
        encoding="utf-8",
    )
    sample_vcf_2 = tmp_path / "cat_2.vcf"
    sample_vcf_2.write_text(
        textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##reference=GCF_000181335.3_Felis_catus_9.0
            ##contig=<ID=chr1,length=6>
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_2
            chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t0/0
            """
        ),
        encoding="utf-8",
    )
    sample_manifest = tmp_path / "sample_manifest.tsv"
    sample_manifest.write_text(
        (
            "sample_id\tindividual_id\tvcf_path\n"
            f"cat_1\tcat-1\t{sample_vcf_1}\n"
            f"cat_2\tcat-2\t{sample_vcf_2}\n"
        ),
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
            source_vcf = "{sample_vcf_1}"
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

    tokenizer_calls: list[str] = []

    class FakeTokenizer:
        def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
            tokenizer_calls.append(sequence)
            return {
                "input_ids": [101, *range(200, 200 + len(sequence)), 102],
                "attention_mask": [1] * (len(sequence) + 2),
            }

    def fake_tokenizer_loader() -> tuple[object, TokenizerProvenance]:
        return (FakeTokenizer(), TokenizerProvenance(max_position_embeddings=8))

    monkeypatch.setattr(pretrain_pipeline, "load_dnabert2_tokenizer", fake_tokenizer_loader)
    monkeypatch.setattr(preprocessor_module, "_load_pyarrow_parquet", _fake_pyarrow_parquet_backend)

    result = pretrain_pipeline.run_feline_pretrain_pipeline(
        config_path,
        bcftools_executable=str(_write_fake_bcftools(tmp_path)),
    )

    baseline_metadata = json.loads(
        (tmp_path / "baseline" / "reference_tokens" / "metadata.json").read_text(encoding="utf-8")
    )
    baseline_files = [
        tmp_path / "baseline" / "reference_tokens" / relative_path
        for split in baseline_metadata["splits"].values()
        for relative_path in split["files"]
    ]
    baseline_rows = json.loads(baseline_files[0].read_text(encoding="utf-8"))["rows"]
    diagnostics = json.loads((tmp_path / "reports" / "eda_payload.json").read_text(encoding="utf-8"))

    assert result.sample_count == 2
    assert result.consensus_window_count == 2
    assert result.baseline_window_count == 1
    assert len(tokenizer_calls) == 3
    assert len(baseline_rows) == 1
    assert diagnostics["baseline_window_alignment"] == {
        "matched_consensus_window_count": 2,
        "unmatched_consensus_window_count": 0,
    }
    assert diagnostics["baseline_corpus"]["retained_window_count"] == 1
    assert diagnostics["baseline_comparison"]["deltas"]["retained_window_count"] == 1


def test_pretrain_pipeline_streams_fasta_records_and_prunes_only_non_emittable_baseline_contigs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(
        ">chr1 GCF_000181335.3 Felis_catus_9.0\nAACCAA\n"
        ">chr2 GCF_000181335.3 Felis_catus_9.0\nTTGGCC\n"
        ">chr3 GCF_000181335.3 Felis_catus_9.0\nAAAA\n",
        encoding="utf-8",
    )
    sample_vcf_1 = tmp_path / "cat_1.vcf"
    sample_vcf_1.write_text(
        textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##reference=GCF_000181335.3_Felis_catus_9.0
            ##contig=<ID=chr1,length=6>
            ##contig=<ID=chr2,length=6>
            ##contig=<ID=chr3,length=4>
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_1
            chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t0/0
            """
        ),
        encoding="utf-8",
    )
    sample_vcf_2 = tmp_path / "cat_2.vcf"
    sample_vcf_2.write_text(
        textwrap.dedent(
            """\
            ##fileformat=VCFv4.2
            ##reference=GCF_000181335.3_Felis_catus_9.0
            ##contig=<ID=chr1,length=6>
            ##contig=<ID=chr2,length=6>
            ##contig=<ID=chr3,length=4>
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tcat_2
            chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t0/0
            """
        ),
        encoding="utf-8",
    )
    sample_manifest = tmp_path / "sample_manifest.tsv"
    sample_manifest.write_text(
        (
            "sample_id\tindividual_id\tvcf_path\n"
            f"cat_1\tcat-1\t{sample_vcf_1}\n"
            f"cat_2\tcat-2\t{sample_vcf_2}\n"
        ),
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
            source_vcf = "{sample_vcf_1}"
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

    prepare_batch_sizes: list[int] = []
    original_prepare_sequences = pretrain_pipeline.prepare_sequences

    def recording_prepare_sequences(records, config):
        prepare_batch_sizes.append(len(records))
        return original_prepare_sequences(records, config)

    class FakeTokenizer:
        def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
            return {
                "input_ids": [101, *range(200, 200 + len(sequence)), 102],
                "attention_mask": [1] * (len(sequence) + 2),
            }

    def fake_tokenizer_loader() -> tuple[object, TokenizerProvenance]:
        return (FakeTokenizer(), TokenizerProvenance(max_position_embeddings=8))

    monkeypatch.setattr(pretrain_pipeline, "load_dnabert2_tokenizer", fake_tokenizer_loader)
    monkeypatch.setattr(pretrain_pipeline, "prepare_sequences", recording_prepare_sequences)
    monkeypatch.setattr(
        pretrain_pipeline,
        "_load_fasta_sequences",
        lambda *_args, **_kwargs: pytest.fail("run_feline_pretrain_pipeline should stream FASTA records"),
    )
    monkeypatch.setattr(preprocessor_module, "_load_pyarrow_parquet", _fake_pyarrow_parquet_backend)

    result = pretrain_pipeline.run_feline_pretrain_pipeline(
        config_path,
        bcftools_executable=str(_write_fake_bcftools(tmp_path)),
    )

    baseline_metadata = json.loads(
        (tmp_path / "baseline" / "reference_tokens" / "metadata.json").read_text(encoding="utf-8")
    )
    baseline_files = [
        tmp_path / "baseline" / "reference_tokens" / relative_path
        for split in baseline_metadata["splits"].values()
        for relative_path in split["files"]
    ]
    baseline_rows = [
        row
        for file_path in baseline_files
        for row in json.loads(file_path.read_text(encoding="utf-8"))["rows"]
    ]
    diagnostics = json.loads((tmp_path / "reports" / "eda_payload.json").read_text(encoding="utf-8"))

    assert result.sample_count == 2
    assert result.consensus_window_count == 4
    assert result.baseline_window_count == 2
    assert max(prepare_batch_sizes) == 1
    assert sorted(row["window"]["contig"] for row in baseline_rows) == ["chr1", "chr2"]
    assert diagnostics["baseline_window_alignment"] == {
        "matched_consensus_window_count": 4,
        "unmatched_consensus_window_count": 0,
    }


def test_pretrain_cli_smoke_path_accepts_gzip_fasta_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    reference = tmp_path / "reference.fasta.gz"
    with gzip.open(reference, "wt", encoding="utf-8") as handle:
        handle.write(">chr1 GCF_000181335.3 Felis_catus_9.0\nAACCAA\n")

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
        return (FakeTokenizer(), TokenizerProvenance(max_position_embeddings=8))

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
    diagnostics = json.loads((tmp_path / "reports" / "eda_payload.json").read_text(encoding="utf-8"))
    assert diagnostics["consensus_generation"]["cat_1"]["applied_variant_count"] == 1
    assert diagnostics["consensus_samples"][0]["no_call_bases"] == 1
    assert diagnostics["consensus_samples"][0]["other_masked_bases"] == 1
    assert diagnostics["consensus_samples"][0]["sequence"] == "ATCNAN"
    assert diagnostics["consensus_samples"][0]["reference_sequence"] == "AACCAA"


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
    assert payload["consensus_corpus"]["masked_category_base_counts"] == {
        "filtered": 64,
        "heterozygous": 22,
        "indel": 21,
        "multiallelic": 21,
        "no_call": 64,
    }
    assert payload["consensus_corpus"]["masked_category_base_fractions"] == {
        "filtered": round(64 / 1152, 6),
        "heterozygous": round(22 / 1152, 6),
        "indel": round(21 / 1152, 6),
        "multiallelic": round(21 / 1152, 6),
        "no_call": round(64 / 1152, 6),
    }
    assert len(payload["consensus_samples"]) == pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT
    assert payload["consensus_sample_overview"] == {
        "total_record_count": 192,
        "returned_record_count": pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT,
        "sample_limit": pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT,
        "truncated": True,
    }
    preview_categories = {
        category
        for sample in payload["consensus_samples"]
        for category in sample["masked_base_counts"]
    }
    assert preview_categories == {"filtered", "no_call"}
    assert payload["consensus_samples"][0]["sequence"] == "ANCCAA"
    assert payload["consensus_samples"][0]["masked_base_counts"] == {"filtered": 1}
    assert payload["consensus_samples"][1]["sequence"] == "AACNAA"
    assert payload["consensus_samples"][1]["masked_base_counts"] == {"no_call": 1}


def test_runtime_diagnostics_passes_streaming_records_into_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_types: dict[str, str] = {}

    def fake_build_eda_payload(consensus_records, baseline_records, **_: object) -> dict[str, object]:
        captured_types["consensus"] = type(consensus_records).__name__
        captured_types["baseline"] = type(baseline_records).__name__
        assert next(iter(consensus_records))["sample_id"] == "cat-1"
        assert next(iter(baseline_records))["sample_id"] == "ref-1"
        return {
            "consensus_samples": [],
            "consensus_sample_overview": {
                "total_record_count": 0,
                "returned_record_count": 0,
                "sample_limit": pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT,
                "truncated": False,
            },
            "consensus_corpus": {},
            "baseline_corpus": {},
            "baseline_comparison": {},
        }

    monkeypatch.setattr(pretrain_pipeline, "build_eda_payload", fake_build_eda_payload)

    payload = pretrain_pipeline._build_diagnostics_payload(
        tokenized_consensus=tuple(_synthetic_tokenized_window(index=index, source="consensus") for index in range(4)),
        tokenized_baseline=tuple(_synthetic_tokenized_window(index=index, source="reference") for index in range(4)),
        consensus_results={},
    )

    assert captured_types == {"consensus": "generator", "baseline": "generator"}
    assert payload["consensus_generation"] == {}


def test_runtime_diagnostics_records_unmatched_consensus_windows_without_crashing() -> None:
    unmatched_consensus = replace(
        _synthetic_tokenized_window(index=2, source="consensus"),
        window=replace(
            _synthetic_tokenized_window(index=2, source="consensus").window,
            window_start=999,
            window_end=1005,
        ),
    )
    payload = pretrain_pipeline._build_diagnostics_payload(
        tokenized_consensus=(
            _synthetic_tokenized_window(index=0, source="consensus"),
            _synthetic_tokenized_window(index=1, source="consensus"),
            unmatched_consensus,
        ),
        tokenized_baseline=tuple(_synthetic_tokenized_window(index=index, source="reference") for index in range(2)),
        consensus_results={},
    )

    assert payload["baseline_window_alignment"] == {
        "matched_consensus_window_count": 2,
        "unmatched_consensus_window_count": 1,
    }
    assert payload["consensus_corpus"]["retained_window_count"] == 3
    unmatched_sample = payload["consensus_samples"][2]
    assert unmatched_sample["reference_window_matched"] is False
    assert unmatched_sample["reference_sequence"] == unmatched_sample["sequence"]
    assert unmatched_sample["variant_count"] == 0


def test_runtime_diagnostics_handles_overlapping_mask_categories_without_callable_underflow() -> None:
    overlapping_mask_window = replace(
        _synthetic_tokenized_window(index=0, source="consensus"),
        window=replace(
            _synthetic_tokenized_window(index=0, source="consensus").window,
            sequence="NNNNAA",
            ambiguity_fraction=4 / 6,
            filtered_bases=4,
            no_call_bases=4,
            other_masked_bases=0,
            masked_base_counts=(("filtered", 4), ("no_call", 4)),
        ),
    )

    payload = pretrain_pipeline._build_diagnostics_payload(
        tokenized_consensus=(overlapping_mask_window,),
        tokenized_baseline=(_synthetic_tokenized_window(index=0, source="reference"),),
        consensus_results={},
    )

    sample = payload["consensus_samples"][0]
    assert sample["callable_bases"] == 2
    assert sample["filtered_bases"] == 4
    assert sample["no_call_bases"] == 4
    assert sample["masked_base_counts"] == {"filtered": 4, "no_call": 4}
    assert payload["consensus_corpus"]["shape_issue_count"] == 0
    assert payload["consensus_corpus"]["masked_category_base_counts"] == {"filtered": 4, "no_call": 4}


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
        if index < pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT:
            if index % 2 == 0:
                sequence = "ANCCAA"
                filtered_bases = 1
                masked_base_counts = (("filtered", 1),)
            else:
                sequence = "AACNAA"
                no_call_bases = 1
                masked_base_counts = (("no_call", 1),)
        else:
            pattern = (index - pretrain_pipeline.DEFAULT_RUNTIME_DIAGNOSTIC_SAMPLE_LIMIT) % 3
            if pattern == 0:
                sequence = "ANCNAA"
                other_masked_bases = 1
                masked_base_counts = (("heterozygous", 1),)
            elif pattern == 1:
                sequence = "AANCNA"
                other_masked_bases = 1
                masked_base_counts = (("multiallelic", 1),)
            else:
                sequence = "AACCAN"
                other_masked_bases = 1
                masked_base_counts = (("indel", 1),)

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