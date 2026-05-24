"""Unit tests for MIL bag datasets and fold-aware dataloader construction.

These tests keep the embedding tensors tiny so they validate dataset semantics,
manifest parsing, and single-bag collation without paying the cost of the full
integration-scale 84k-locus smoke test.
"""

from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from jaguar_geo_assign.config import MILFinetuneConfig
from jaguar_geo_assign.fine_tune.dataset import BIOME_CLASSES, CoordStats
from jaguar_geo_assign.fine_tune.mil_dataset import (
    MILBagDataset,
    build_mil_fold_dataloaders,
    mil_collate_fn,
)
from jaguar_geo_assign.fine_tune.trainer import _compute_baselines


def _write_metadata_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the minimal metadata CSV required by the MIL split helpers."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "individual_id",
                "biome_population_label",
                "latitude",
                "longitude",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _append_manifest_record(
    manifest_path: Path,
    *,
    shard_path: Path,
    individual_id: str,
    sample_id: str,
    biome_population_label: str,
    latitude: float,
    longitude: float,
    n_windows: int,
) -> None:
    """Append one per-individual manifest record for a synthetic MIL shard."""

    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "individual_id": individual_id,
                    "shard_path": str(shard_path),
                    "n_windows": n_windows,
                    "sample_id": sample_id,
                    "latitude": latitude,
                    "longitude": longitude,
                    "biome_population_label": biome_population_label,
                }
            )
            + "\n"
        )


def test_mil_bag_dataset_loads_bag_and_normalizes_targets(tmp_path: Path) -> None:
    """MILBagDataset should return one full bag with z-scored coordinates."""

    shard_path = tmp_path / "ind-0.pt"
    torch.save(
        {
            "embeddings": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "bp_positions": torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
            "contigs": ["chr1", "chr1", "chr1"],
        },
        shard_path,
    )
    manifest_path = tmp_path / "manifest.jsonl"
    _append_manifest_record(
        manifest_path,
        shard_path=shard_path,
        individual_id="ind-0",
        sample_id="sample-0",
        biome_population_label=BIOME_CLASSES[0],
        latitude=12.0,
        longitude=28.0,
        n_windows=3,
    )

    dataset = MILBagDataset(
        manifest_path=manifest_path,
        individual_ids=["ind-0"],
        coord_stats=CoordStats(lat_mean=10.0, lat_std=2.0, lon_mean=20.0, lon_std=4.0),
        biome_to_idx={BIOME_CLASSES[0]: 0},
    )
    sample = dataset[0]

    assert sample["embeddings"].shape == (3, 4)
    assert sample["bp_positions"].shape == (3,)
    assert sample["coord_target"].shape == (2,)
    assert torch.allclose(sample["coord_target"], torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert sample["biome_label"].ndim == 0
    assert int(sample["biome_label"].item()) == 0


def test_mil_bag_dataset_iter_raw_targets_avoids_shard_loading(tmp_path: Path) -> None:
    """Raw target iteration should stay on manifest metadata for baseline setup.

    This guards the scalability contract for MIL baselines: collecting eval
    targets must not read per-individual embedding shards from disk.
    """

    manifest_path = tmp_path / "manifest.jsonl"
    _append_manifest_record(
        manifest_path,
        shard_path=tmp_path / "unused-a.pt",
        individual_id="ind-a",
        sample_id="sample-a",
        biome_population_label=BIOME_CLASSES[0],
        latitude=12.0,
        longitude=28.0,
        n_windows=3,
    )
    _append_manifest_record(
        manifest_path,
        shard_path=tmp_path / "unused-b.pt",
        individual_id="ind-b",
        sample_id="sample-b",
        biome_population_label=BIOME_CLASSES[1],
        latitude=8.0,
        longitude=12.0,
        n_windows=5,
    )

    dataset = MILBagDataset(
        manifest_path=manifest_path,
        individual_ids=["ind-a", "ind-b"],
        coord_stats=CoordStats(lat_mean=10.0, lat_std=2.0, lon_mean=20.0, lon_std=4.0),
        biome_to_idx={BIOME_CLASSES[0]: 0, BIOME_CLASSES[1]: 1},
    )

    with patch("jaguar_geo_assign.fine_tune.mil_dataset.torch.load") as load_mock:
        raw_targets = list(dataset.iter_raw_targets())

    assert raw_targets == [(0, 1.0, 2.0), (1, -1.0, -2.0)]
    assert load_mock.call_count == 0


def test_mil_bag_dataset_rejects_empty_shard(tmp_path: Path) -> None:
    """Empty bags must fail loudly before softmax sees a zero-length bag."""

    shard_path = tmp_path / "empty.pt"
    torch.save(
        {
            "embeddings": torch.empty((0, 4), dtype=torch.float32),
            "bp_positions": torch.empty((0,), dtype=torch.float32),
            "contigs": [],
        },
        shard_path,
    )
    manifest_path = tmp_path / "manifest.jsonl"
    _append_manifest_record(
        manifest_path,
        shard_path=shard_path,
        individual_id="ind-empty",
        sample_id="sample-empty",
        biome_population_label=BIOME_CLASSES[0],
        latitude=0.0,
        longitude=0.0,
        n_windows=0,
    )

    dataset = MILBagDataset(
        manifest_path=manifest_path,
        individual_ids=["ind-empty"],
        coord_stats=CoordStats(lat_mean=0.0, lat_std=1.0, lon_mean=0.0, lon_std=1.0),
        biome_to_idx={BIOME_CLASSES[0]: 0},
    )
    with pytest.raises(ValueError, match="Empty bag"):
        _ = dataset[0]


def test_mil_collate_fn_rejects_multi_item_batches() -> None:
    """The MIL collate helper must guard against accidental batch_size > 1."""

    with pytest.raises(ValueError, match="batch_size=1"):
        mil_collate_fn([{"x": torch.tensor(1)}, {"x": torch.tensor(2)}])


def test_compute_baselines_uses_iter_raw_targets_for_mil_datasets(tmp_path: Path) -> None:
    """Baseline setup must not call ``MILBagDataset.__getitem__`` for eval bags.

    A regression here would reload every embedding shard just to recover labels
    and coordinates, turning baseline computation into avoidable O(dataset size)
    disk IO before training even starts.
    """

    manifest_path = tmp_path / "manifest.jsonl"
    _append_manifest_record(
        manifest_path,
        shard_path=tmp_path / "unused-train-0.pt",
        individual_id="train-0",
        sample_id="sample-train-0",
        biome_population_label=BIOME_CLASSES[0],
        latitude=10.0,
        longitude=20.0,
        n_windows=3,
    )
    _append_manifest_record(
        manifest_path,
        shard_path=tmp_path / "unused-train-1.pt",
        individual_id="train-1",
        sample_id="sample-train-1",
        biome_population_label=BIOME_CLASSES[0],
        latitude=11.0,
        longitude=24.0,
        n_windows=4,
    )
    _append_manifest_record(
        manifest_path,
        shard_path=tmp_path / "unused-eval-0.pt",
        individual_id="eval-0",
        sample_id="sample-eval-0",
        biome_population_label=BIOME_CLASSES[1],
        latitude=12.0,
        longitude=28.0,
        n_windows=5,
    )

    coord_stats = CoordStats(lat_mean=10.0, lat_std=2.0, lon_mean=20.0, lon_std=4.0)
    biome_to_idx = {BIOME_CLASSES[0]: 0, BIOME_CLASSES[1]: 1}
    train_source = SimpleNamespace(
        dataset=MILBagDataset(
            manifest_path=manifest_path,
            individual_ids=["train-0", "train-1"],
            coord_stats=coord_stats,
            biome_to_idx=biome_to_idx,
        )
    )
    eval_source = SimpleNamespace(
        dataset=MILBagDataset(
            manifest_path=manifest_path,
            individual_ids=["eval-0"],
            coord_stats=coord_stats,
            biome_to_idx=biome_to_idx,
        )
    )

    with patch.object(
        MILBagDataset,
        "__getitem__",
        autospec=True,
        side_effect=AssertionError("_compute_baselines should not load MIL shards"),
    ) as getitem_mock:
        metrics = _compute_baselines(
            train_loader=train_source,
            eval_loader=eval_source,
            coord_stats=coord_stats,
            n_biomes=2,
        )

    assert getitem_mock.call_count == 0
    assert isfinite(metrics["macro_f1"])
    assert isfinite(metrics["haversine_km_mean"])
    assert isfinite(metrics["haversine_km_median"])


def test_build_mil_fold_dataloaders_yields_single_full_bag_batches(tmp_path: Path) -> None:
    """Fold builders should emit per-individual batches with no outer batch axis."""

    manifest_path = tmp_path / "manifest.jsonl"
    metadata_path = tmp_path / "metadata.csv"
    metadata_rows: list[dict[str, object]] = []

    for biome_index, biome in enumerate(BIOME_CLASSES):
        for sample_offset in range(2):
            individual_id = f"{biome_index}-{sample_offset}"
            sample_id = f"sample-{individual_id}"
            shard_path = tmp_path / f"{individual_id}.pt"
            torch.save(
                {
                    "embeddings": torch.full(
                        (2, 4), float(biome_index + sample_offset), dtype=torch.float32
                    ),
                    "bp_positions": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "contigs": ["chr1", "chr1"],
                },
                shard_path,
            )
            latitude = float(biome_index + sample_offset)
            longitude = float(10 + biome_index + sample_offset)
            _append_manifest_record(
                manifest_path,
                shard_path=shard_path,
                individual_id=individual_id,
                sample_id=sample_id,
                biome_population_label=biome,
                latitude=latitude,
                longitude=longitude,
                n_windows=2,
            )
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "individual_id": individual_id,
                    "biome_population_label": biome,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )

    _write_metadata_csv(metadata_path, metadata_rows)
    config = MILFinetuneConfig(
        embeddings_dir=tmp_path,
        metadata_csv=metadata_path,
        output_dir=tmp_path / "out",
        embedding_dim=4,
        hidden_dim=2,
        n_folds=2,
        fold_index=0,
        n_biomes=5,
        num_workers=0,
        mil_steps=1,
    )

    train_loader, eval_loader, coord_stats = build_mil_fold_dataloaders(config)
    sample = next(iter(train_loader))

    assert len(train_loader.dataset) + len(eval_loader.dataset) == 10
    assert sample["embeddings"].shape == (2, 4)
    assert sample["bp_positions"].shape == (2,)
    assert sample["coord_target"].shape == (2,)
    assert sample["biome_label"].ndim == 0
    assert coord_stats.lat_std > 0.0
    assert coord_stats.lon_std > 0.0


def test_build_mil_fold_dataloaders_reuses_eval_workers_when_enabled(tmp_path: Path) -> None:
    """Eval workers should persist across repeated iterations when num_workers > 0.

    This regression guards against PyTorch respawning worker processes on every
    eval pass, which adds avoidable startup overhead for lazy MIL shard loads.
    """

    manifest_path = tmp_path / "manifest.jsonl"
    metadata_path = tmp_path / "metadata.csv"
    metadata_rows: list[dict[str, object]] = []

    for biome_index, biome in enumerate(BIOME_CLASSES):
        for sample_offset in range(2):
            individual_id = f"persist-{biome_index}-{sample_offset}"
            sample_id = f"sample-{individual_id}"
            shard_path = tmp_path / f"{individual_id}.pt"
            torch.save(
                {
                    "embeddings": torch.full((2, 4), float(biome_index), dtype=torch.float32),
                    "bp_positions": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "contigs": ["chr1", "chr1"],
                },
                shard_path,
            )
            latitude = float(biome_index + sample_offset)
            longitude = float(10 + biome_index + sample_offset)
            _append_manifest_record(
                manifest_path,
                shard_path=shard_path,
                individual_id=individual_id,
                sample_id=sample_id,
                biome_population_label=biome,
                latitude=latitude,
                longitude=longitude,
                n_windows=2,
            )
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "individual_id": individual_id,
                    "biome_population_label": biome,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )

    _write_metadata_csv(metadata_path, metadata_rows)
    config = MILFinetuneConfig(
        embeddings_dir=tmp_path,
        metadata_csv=metadata_path,
        output_dir=tmp_path / "out",
        embedding_dim=4,
        hidden_dim=2,
        n_folds=2,
        fold_index=0,
        n_biomes=5,
        num_workers=2,
        mil_steps=1,
    )

    _train_loader, eval_loader, _coord_stats = build_mil_fold_dataloaders(config)
    try:
        assert eval_loader.persistent_workers is True

        first_iter = iter(eval_loader)
        list(first_iter)
        first_worker_pids = {worker.pid for worker in eval_loader._iterator._workers}

        second_iter = iter(eval_loader)
        list(second_iter)
        second_worker_pids = {worker.pid for worker in eval_loader._iterator._workers}

        assert len(first_worker_pids) == config.num_workers
        assert len(second_worker_pids) == config.num_workers
        assert second_worker_pids == first_worker_pids
    finally:
        if eval_loader._iterator is not None:
            eval_loader._iterator._shutdown_workers()
