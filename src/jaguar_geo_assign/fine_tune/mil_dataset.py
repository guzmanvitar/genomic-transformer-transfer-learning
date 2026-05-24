# ruff: noqa: F722  # jaxtyping shape annotations use string-based dimensions
"""Data loading helpers for full-bag MIL training on offline jaguar embeddings.

The MIL path consumes per-individual embedding shards written by the offline
extraction stage. Each dataset item therefore represents one whole jaguar bag
instead of one genomic window, which keeps the training loop aligned with the
multi-locus problem structure.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from beartype import beartype
from jaxtyping import jaxtyped
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from jaguar_geo_assign.config import MILFinetuneConfig
from jaguar_geo_assign.fine_tune.dataset import (
    BIOME_CLASSES,
    CoordStats,
    _fit_coord_stats,
    _load_metadata_csv,
)

Tensor = torch.Tensor


def _load_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load the extraction manifest and coerce paths and scalars to stable types."""

    manifest_path = Path(path)
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            shard_path = Path(record["shard_path"])
            if not shard_path.is_absolute():
                shard_path = (manifest_path.parent / shard_path).resolve()
            records.append(
                {
                    "individual_id": str(record["individual_id"]),
                    "sample_id": str(record["sample_id"]),
                    "latitude": float(record["latitude"]),
                    "longitude": float(record["longitude"]),
                    "biome_population_label": str(record["biome_population_label"]),
                    "n_windows": int(record["n_windows"]),
                    "shard_path": shard_path,
                }
            )
    if not records:
        raise ValueError(f"Manifest {manifest_path} contained no records")
    return records


def _load_split_records(
    manifest_path: str | Path,
    metadata_csv: str | Path,
) -> list[dict[str, Any]]:
    """Join the manifest with metadata to validate MIL split inputs eagerly."""

    manifest_records = _load_manifest_jsonl(manifest_path)
    metadata_by_sample = _load_metadata_csv(Path(metadata_csv))
    split_records: list[dict[str, Any]] = []
    for record in manifest_records:
        sample_id = str(record["sample_id"])
        if sample_id not in metadata_by_sample:
            raise ValueError(
                f"Manifest sample_id {sample_id!r} is missing from metadata_csv {metadata_csv}"
            )
        metadata = metadata_by_sample[sample_id]
        if str(metadata["individual_id"]) != record["individual_id"]:
            raise ValueError(
                f"Manifest individual_id {record['individual_id']!r} does not match metadata "
                f"individual_id {metadata['individual_id']!r} for sample_id {sample_id!r}"
            )
        split_records.append(
            {
                **record,
                "latitude": float(metadata["latitude"]),
                "longitude": float(metadata["longitude"]),
                "biome_population_label": str(metadata["biome_population_label"]),
            }
        )
    return split_records


class MILBagDataset(Dataset):
    """Per-individual bag dataset backed by pre-extracted embedding shards.

    Each ``__getitem__`` call loads exactly one shard lazily from disk so the
    training loop never materializes the entire dataset in memory. Returned
    coordinates are z-scored using the supplied ``CoordStats`` instance so the
    MIL path reuses the existing loss and metric helpers unchanged.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        individual_ids: list[str],
        coord_stats: CoordStats,
        biome_to_idx: dict[str, int] | None = None,
    ) -> None:
        self._coord_stats = coord_stats
        self._biome_to_idx = biome_to_idx
        self._individual_ids = list(individual_ids)
        self._records_by_individual = {
            record["individual_id"]: record for record in _load_manifest_jsonl(manifest_path)
        }
        missing = [item for item in self._individual_ids if item not in self._records_by_individual]
        if missing:
            raise ValueError(
                f"MILBagDataset missing manifest entries for individual_ids: {missing}"
            )

    def __len__(self) -> int:
        """Return the number of jaguar bags in this split."""

        return len(self._individual_ids)

    def iter_raw_targets(self) -> Iterator[tuple[int, float, float]]:
        """Yield normalized targets from manifest metadata without loading shards.

        Baseline computation only needs the per-individual biome label and
        normalized coordinates, so this iterator stays on the manifest-backed
        record cache instead of touching ``__getitem__`` and incurring full shard
        reads for every evaluation individual.
        """

        for individual_id in self._individual_ids:
            record = self._records_by_individual[individual_id]
            biome_idx = -1
            if self._biome_to_idx is not None:
                biome_label = str(record["biome_population_label"])
                if biome_label not in self._biome_to_idx:
                    raise ValueError(
                        f"Unknown biome_population_label {biome_label!r}; expected one of "
                        f"{sorted(self._biome_to_idx)}"
                    )
                biome_idx = self._biome_to_idx[biome_label]
            lat_z = (
                float(record["latitude"]) - self._coord_stats.lat_mean
            ) / self._coord_stats.lat_std
            lon_z = (
                float(record["longitude"]) - self._coord_stats.lon_mean
            ) / self._coord_stats.lon_std
            yield biome_idx, lat_z, lon_z

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Load one shard and return bag tensors for the requested individual."""

        individual_id = self._individual_ids[idx]
        record = self._records_by_individual[individual_id]
        shard_path = Path(record["shard_path"])
        if not shard_path.exists():
            raise FileNotFoundError(
                f"Embedding shard for {individual_id!r} is missing: {shard_path}"
            )

        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "torch.load(weights_only=True) is required by the MIL shard-loading contract"
            ) from exc

        embeddings = torch.as_tensor(shard["embeddings"], dtype=torch.float32)
        bp_positions = torch.as_tensor(shard["bp_positions"], dtype=torch.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding shard for "
                f"{individual_id!r} must be rank-2; got shape {tuple(embeddings.shape)}"
            )
        if bp_positions.ndim != 1:
            raise ValueError(
                "bp_positions for "
                f"{individual_id!r} must be rank-1; got shape {tuple(bp_positions.shape)}"
            )
        if embeddings.shape[0] == 0:
            raise ValueError(
                f"Empty bag for {individual_id}: shard has 0 windows. "
                "Verify extraction pipeline for dropped metadata joins."
            )
        if embeddings.shape[0] != bp_positions.shape[0]:
            raise ValueError(
                f"Shard for {individual_id!r} has mismatched embeddings/bp_positions lengths: "
                f"{embeddings.shape[0]} vs {bp_positions.shape[0]}"
            )

        coord_target = torch.tensor(
            [
                (float(record["latitude"]) - self._coord_stats.lat_mean)
                / self._coord_stats.lat_std,
                (float(record["longitude"]) - self._coord_stats.lon_mean)
                / self._coord_stats.lon_std,
            ],
            dtype=torch.float32,
        )

        sample: dict[str, Tensor] = {
            "embeddings": embeddings,
            "bp_positions": bp_positions,
            "coord_target": coord_target,
        }
        if self._biome_to_idx is not None:
            biome_label = str(record["biome_population_label"])
            if biome_label not in self._biome_to_idx:
                raise ValueError(
                    f"Unknown biome_population_label {biome_label!r}; expected one of "
                    f"{sorted(self._biome_to_idx)}"
                )
            sample["biome_label"] = torch.tensor(self._biome_to_idx[biome_label], dtype=torch.long)

        if os.getenv("JAGUAR_DEBUG_TENSORS") == "1":
            for key in ("embeddings", "bp_positions", "coord_target"):
                value = sample[key]
                if not torch.isfinite(value).all():
                    raise RuntimeError(
                        f"Shard for {individual_id!r} produced non-finite tensor {key}"
                    )
        return sample


def mil_collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Unwrap the single-item DataLoader batch used by full-bag MIL training."""

    if len(batch) != 1:
        raise ValueError(
            f"MIL DataLoader requires batch_size=1; got {len(batch)} items. "
            "Ensure DataLoader is constructed with batch_size=1."
        )
    return batch[0]


def build_mil_fold_dataloaders(
    config: MILFinetuneConfig,
) -> tuple[DataLoader, DataLoader, CoordStats]:
    """Build train/eval loaders over full embedding bags for one MIL fold."""

    manifest_path = Path(config.embeddings_dir) / "manifest.jsonl"
    split_records = _load_split_records(manifest_path, Path(config.metadata_csv))

    allowed_biomes = BIOME_CLASSES[: config.n_biomes]
    individuals_per_biome: dict[str, set[str]] = {biome: set() for biome in allowed_biomes}
    for record in split_records:
        biome = str(record["biome_population_label"])
        if biome not in individuals_per_biome:
            raise ValueError(
                f"Manifest biome_population_label {biome!r} is outside the configured label set "
                f"{allowed_biomes}"
            )
        individuals_per_biome[biome].add(str(record["individual_id"]))
    scarce = {
        biome: len(individual_ids)
        for biome, individual_ids in individuals_per_biome.items()
        if len(individual_ids) < config.n_folds
    }
    if scarce:
        raise ValueError(
            "Each biome class must have at least n_folds unique individuals for MIL splitting; "
            f"got {scarce} with n_folds={config.n_folds}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.seed,
    )
    record_indices = list(range(len(split_records)))
    labels = [str(record["biome_population_label"]) for record in split_records]
    groups = [str(record["individual_id"]) for record in split_records]
    splits = list(splitter.split(record_indices, labels, groups))
    train_idx, eval_idx = splits[config.fold_index]

    train_records = [split_records[index] for index in train_idx]
    eval_records = [split_records[index] for index in eval_idx]
    coord_stats = _fit_coord_stats(train_records)
    biome_to_idx = {name: idx for idx, name in enumerate(BIOME_CLASSES[: config.n_biomes])}

    train_dataset = MILBagDataset(
        manifest_path=manifest_path,
        individual_ids=[str(record["individual_id"]) for record in train_records],
        coord_stats=coord_stats,
        biome_to_idx=biome_to_idx,
    )
    eval_dataset = MILBagDataset(
        manifest_path=manifest_path,
        individual_ids=[str(record["individual_id"]) for record in eval_records],
        coord_stats=coord_stats,
        biome_to_idx=biome_to_idx,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        collate_fn=mil_collate_fn,
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        collate_fn=mil_collate_fn,
    )
    return train_loader, eval_loader, coord_stats


__all__ = ["MILBagDataset", "build_mil_fold_dataloaders", "mil_collate_fn"]
