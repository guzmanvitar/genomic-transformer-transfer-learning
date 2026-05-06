"""Tests for MtlFinetuneConfig and jaguar multi-task fine-tuning data path.

Covers config loading contracts, CoordStats JSON round-trip, dataset tensor
contracts, and basic behaviour of build_fold_dataloaders (join, weighting,
logging of dropped windows).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from jaguar_geo_assign.config import MtlFinetuneConfig, load_mtl_finetune_config
from jaguar_geo_assign.fine_tune.dataset import (
    BIOME_CLASSES,
    CoordStats,
    JaguarMTLDataset,
    build_fold_dataloaders,
)


class DummyTokenizer:
    """Minimal tokenizer that emits fixed-length tensors for testing."""

    def __call__(
        self, text: str, *, max_length: int, padding: str, truncation: bool, return_tensors: str
    ):
        assert padding == "max_length" and truncation and return_tensors == "pt"
        input_ids = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_load_mtl_finetune_config_happy_path(tmp_path: Path) -> None:
    """Loader must populate defaults and enforce basic contracts on happy path."""

    config_file = tmp_path / "mtl.toml"
    config_file.write_text(
        """
[training]
backbone_path = "models/foundation_felid/best/hf_model"
windows_jsonl = "windows.jsonl"
metadata_csv = "metadata.csv"
output_dir = "artifacts/mtl"
""",
        encoding="utf-8",
    )
    config = load_mtl_finetune_config(config_file)
    assert isinstance(config, MtlFinetuneConfig)
    assert config.pooling_strategy in {"cls", "mean"}
    assert config.n_biomes == 5
    assert 0 <= config.fold_index < config.n_folds
    assert config.phase1_steps > 0 and config.phase2_steps > 0


def test_load_mtl_finetune_config_rejects_bad_pooling(tmp_path: Path) -> None:
    """Invalid pooling_strategy must raise a descriptive ValueError."""

    config_file = tmp_path / "mtl_bad_pooling.toml"
    config_file.write_text(
        """
[training]
backbone_path = "b"
windows_jsonl = "w"
metadata_csv = "m"
output_dir = "o"
pooling_strategy = "invalid"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc_info:
        load_mtl_finetune_config(config_file)
    assert "pooling_strategy" in str(exc_info.value)


def test_coord_stats_json_round_trip(tmp_path: Path) -> None:
    """CoordStats.to_json and from_json must round-trip and clamp std devs."""

    stats = CoordStats(lat_mean=1.0, lat_std=0.0, lon_mean=-2.0, lon_std=0.0)
    path = tmp_path / "coord_stats.json"
    stats.to_json(path)
    loaded = CoordStats.from_json(path)
    assert loaded.lat_mean == pytest.approx(1.0)
    assert loaded.lon_mean == pytest.approx(-2.0)
    assert loaded.lat_std >= 1e-6 and loaded.lon_std >= 1e-6


def test_jaguar_mtl_dataset_emits_expected_tensors() -> None:
    """Dataset __getitem__ must return correctly shaped tensors and z-scored coords."""

    record = {
        "sequence": "ACGT" * 10,
        "biome_population_label": BIOME_CLASSES[0],
        "latitude": "10.0",
        "longitude": "20.0",
    }
    stats = CoordStats(lat_mean=10.0, lat_std=2.0, lon_mean=10.0, lon_std=5.0)
    dataset = JaguarMTLDataset([record], DummyTokenizer(), stats, max_length=8)
    item = dataset[0]
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["biome_label"].shape == ()
    lat_z, lon_z = item["coord_target"].tolist()
    assert lat_z == pytest.approx(0.0)
    assert lon_z == pytest.approx(2.0)


def test_build_fold_dataloaders_join_and_weights(tmp_path: Path) -> None:
    """build_fold_dataloaders must join metadata, fit CoordStats, and build a weighted sampler."""

    windows_path = tmp_path / "windows.jsonl"
    metadata_path = tmp_path / "metadata.csv"

    windows = [
        {"sample_id": "s1", "sequence": "A" * 16},
        {"sample_id": "s1", "sequence": "C" * 16},
        {"sample_id": "s2", "sequence": "G" * 16},
    ]
    windows_path.write_text("\n".join(json.dumps(w) for w in windows) + "\n", encoding="utf-8")

    metadata_path.write_text(
        "sample_id,individual_id,biome_population_label,latitude,longitude\n"
        "s1,ind-1,Amazon,0.0,0.0\n"
        "s2,ind-2,Amazon,10.0,20.0\n",
        encoding="utf-8",
    )

    config = MtlFinetuneConfig(
        backbone_path=tmp_path / "backbone",
        windows_jsonl=windows_path,
        metadata_csv=metadata_path,
        output_dir=tmp_path / "out",
        n_folds=2,
        fold_index=0,
    )

    train_loader, eval_loader, coord_stats = build_fold_dataloaders(config, DummyTokenizer())
    assert coord_stats.lat_std >= 1e-6 and coord_stats.lon_std >= 1e-6

    train_records = train_loader.dataset._records  # type: ignore[attr-defined]
    assert train_records[0]["individual_id"] in {
        "ind-1",
        "ind-2",
    }  # individual_id sourced from metadata CSV row, not FinetuneWindow

    # Check that per-individual window counts are equalised by the sampler.
    from collections import Counter

    counts = Counter(str(r["individual_id"]) for r in train_records)
    expected_weights = [1.0 / counts[str(r["individual_id"])] for r in train_records]
    sampler_weights = list(map(float, train_loader.sampler.weights))  # type: ignore[arg-type]
    assert sampler_weights == pytest.approx(expected_weights)


def test_build_fold_dataloaders_logs_dropped_windows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Mismatched sample IDs must produce a WARNING log with the dropped-count summary."""

    windows_path = tmp_path / "windows.jsonl"
    metadata_path = tmp_path / "metadata.csv"

    windows_path.write_text(
        json.dumps({"sample_id": "missing", "sequence": "A" * 16}) + "\n", encoding="utf-8"
    )
    metadata_path.write_text(
        "sample_id,individual_id,biome_population_label,latitude,longitude\n",
        encoding="utf-8",
    )

    config = MtlFinetuneConfig(
        backbone_path=tmp_path / "backbone",
        windows_jsonl=windows_path,
        metadata_csv=metadata_path,
        output_dir=tmp_path / "out",
        n_folds=2,
        fold_index=0,
    )

    logger_name = "jaguar_geo_assign.fine_tune.dataset"
    caplog.set_level(logging.WARNING, logger=logger_name)
    # A join that drops every window must surface as a ValueError so callers
    # never proceed with an empty dataset silently.
    with pytest.raises(ValueError):
        build_fold_dataloaders(config, DummyTokenizer())
    assert any("Dropped 1 windows" in rec.getMessage() for rec in caplog.records)
