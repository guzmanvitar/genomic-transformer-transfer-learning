"""Tests for the active ``jaguar_geo_assign`` CLI surface.

These tests intentionally focus on commands that still exist after the legacy
feline cleanup: bootstrap config inspection, felid foundation pretraining, and
felid foundation continued pre-training dispatch. The goal is to keep the
user-facing contract covered without retaining assertions for removed commands.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jaguar_geo_assign.cli import main


def test_validate_config_reports_success(capsys: pytest.CaptureFixture[str]) -> None:
    """Bootstrap config validation should succeed on the shipped example."""
    exit_code = main(["validate-config", "configs/examples/fine_tune.toml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "is valid for the bootstrap scaffold" in captured.out


def test_describe_experiment_reports_deferred_baseline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Experiment description should expose the deferred baseline wiring."""
    exit_code = main(["describe-experiment", "configs/examples/regression_transfer.toml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Deferred baseline: baseline_evaluate -> deferred_legacy_group_model" in captured.out


@pytest.mark.parametrize("command", ["fine-tune", "evaluate", "report"])
def test_stage_scaffolds_echo_loaded_bootstrap_config(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Deferred stage entry points should stay callable while the CLI is scaffold-only."""
    exit_code = main([command, "--config", "configs/examples/regression_transfer.toml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"{command} entry point scaffold is available." in captured.out
    assert "Loaded config: regression_transfer_bootstrap" in captured.out


def test_baseline_evaluate_scaffold_reports_reserved_stage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The baseline scaffold should explain why execution is still deferred."""
    exit_code = main(["baseline-evaluate", "--config", "configs/examples/regression_transfer.toml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "baseline-evaluate entry point scaffold is available" in captured.out
    assert "Deferred baseline stage is reserved for baseline_evaluate" in captured.out


def test_validate_felid_foundation_config_reports_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The shipped felid foundation config should validate cleanly."""
    exit_code = main(
        ["validate-felid-foundation-config", "configs/examples/felid_foundation_pretrain.toml"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "matches the approved contract" in captured.out


def test_validate_felid_foundation_config_rejects_bootstrap_toml(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Foundation validation should fail loudly on a non-foundation config."""
    exit_code = main(["validate-felid-foundation-config", "configs/examples/fine_tune.toml"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing required sections" in captured.out
    assert "Traceback" not in captured.out


def test_describe_felid_foundation_config_reports_species(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The describe command should summarize the six-species roster."""
    exit_code = main(
        ["describe-felid-foundation-config", "configs/examples/felid_foundation_pretrain.toml"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Felis catus" in captured.out
    assert "Panthera onca" in captured.out


def test_check_felid_foundation_runtime_reports_no_external_tools(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime checks should explain that the active foundation path needs no external CLI tools."""
    exit_code = main(
        ["check-felid-foundation-runtime", "configs/examples/felid_foundation_pretrain.toml"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "no external tools required" in captured.out


def test_felid_foundation_pretrain_dispatches_runner_and_formatter(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI dispatch should pass the config path into the active felid pretrain runner."""
    calls: list[Path] = []
    fake_result = SimpleNamespace()

    def fake_runner(config_path: Path) -> object:
        calls.append(config_path)
        return fake_result

    monkeypatch.setattr("jaguar_geo_assign.cli.run_felid_foundation_pretrain", fake_runner)
    monkeypatch.setattr(
        "jaguar_geo_assign.cli.format_felid_foundation_pretrain_result",
        lambda result: "formatted felid run" if result is fake_result else "unexpected",
    )

    exit_code = main(
        ["felid-foundation-pretrain", "configs/examples/felid_foundation_pretrain.toml"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [Path("configs/examples/felid_foundation_pretrain.toml")]
    assert "formatted felid run" in captured.out


def test_acquire_felid_foundation_assemblies_reports_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Acquisition dispatch should print the structured checksum summary."""
    fake_config = object()
    fake_summary = SimpleNamespace(total_bytes_written=123, skipped_count=5, redownloaded_count=1)

    monkeypatch.setattr(
        "jaguar_geo_assign.cli.load_felid_foundation_pipeline_config", lambda _: fake_config
    )
    monkeypatch.setattr(
        "jaguar_geo_assign.cli.acquire_felid_foundation_assemblies",
        lambda config: fake_summary if config is fake_config else None,
    )

    exit_code = main(
        ["acquire-felid-foundation-assemblies", "configs/examples/felid_foundation_pretrain.toml"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Felid foundation assembly acquisition summary:" in captured.out
    assert "Total bytes written: 123" in captured.out
    assert "Skipped (checksum match): 5" in captured.out
    assert "Redownloaded (checksum mismatch): 1" in captured.out


def test_train_felid_foundation_with_config_dispatches_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``train-felid-foundation`` should call the continued pre-training entry point."""
    called: list[Path] = []

    monkeypatch.setattr(
        "jaguar_geo_assign.pretrain.foundation_training.run_felid_foundation_training",
        lambda config_path: called.append(config_path),
    )

    exit_code = main(
        ["train-felid-foundation", "--config", "configs/examples/felid_foundation_train.toml"]
    )

    assert exit_code == 0
    assert called == [Path("configs/examples/felid_foundation_train.toml")]


def test_train_felid_foundation_integration_flag_uses_tiny_mode() -> None:
    """The integration flag should request the cheap local smoke path, not the Hub-backed one."""
    with patch(
        "jaguar_geo_assign.pretrain.foundation_training.integration_test"
    ) as mock_integration:
        with patch("jaguar_geo_assign.pretrain.foundation_training.run_felid_foundation_training"):
            exit_code = main(
                [
                    "train-felid-foundation",
                    "--config",
                    "configs/examples/felid_foundation_train.toml",
                    "--integration-test",
                ]
            )

    assert exit_code == 0
    mock_integration.assert_called_once_with(use_real_model=False)
