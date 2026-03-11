from pathlib import Path

import pytest

from jaguar_geo_assign.cli import main


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