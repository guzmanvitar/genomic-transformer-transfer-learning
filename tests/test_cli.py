from pathlib import Path

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