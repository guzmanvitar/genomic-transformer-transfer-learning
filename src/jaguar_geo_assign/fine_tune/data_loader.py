"""Data loading pipeline for MTL fine-tuning datasets.

Orchestrates loading FinetuneWindow sequences, metadata CSV files, applying
stratified group k-fold splits, standardizing coordinates, and building
PyTorch DataLoaders for training and validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from jaguar_geo_assign.data.finetune_windows import FinetuneWindow
from jaguar_geo_assign.fine_tune.mtl_dataset import JaguarGeoDataset, NormalizationArtifact
from jaguar_geo_assign.fine_tune.split import assign_fold_indices, create_stratified_group_kfold

_LOGGER = logging.getLogger(__name__)


def collate_mtl_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for MTL batches: pad sequences and tensorize labels.

    Args:
        batch: List of sample dictionaries from JaguarGeoDataset.

    Returns:
        Dictionary with input_ids, attention_mask, biome_labels, coordinate_labels.
    """
    max_seq_len = max(len(sample["input_ids"]) for sample in batch)

    padded_input_ids = []
    padded_masks = []
    biome_labels = []
    coord_labels = []

    for sample in batch:
        input_ids = sample["input_ids"]
        pad_len = max_seq_len - len(input_ids)
        padded_input_ids.append(input_ids + [0] * pad_len)
        padded_masks.append(sample["attention_mask"] + [0] * pad_len)

        if "biome_label" in sample:
            biome_labels.append(sample["biome_label"])
        if "coordinate_labels" in sample:
            coord_labels.append(sample["coordinate_labels"])

    result = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
    }

    if biome_labels:
        result["biome_labels"] = torch.tensor(biome_labels, dtype=torch.long)
    if coord_labels:
        result["coordinate_labels"] = torch.stack(
            [torch.tensor(c, dtype=torch.float32) for c in coord_labels]
        )

    return result


def load_metadata_csv(metadata_path: str | Path) -> pd.DataFrame:
    """Load and validate jaguar metadata CSV."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    df = pd.read_csv(path)
    required_cols = {
        "sample_id",
        "individual_id",
        "biome_population_label",
        "latitude",
        "longitude",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.set_index("sample_id")
    _LOGGER.info(f"Loaded metadata for {len(df)} samples from {metadata_path}")
    return df


def load_finetune_windows_jsonl(jsonl_path: str | Path) -> list[FinetuneWindow]:
    """Load FinetuneWindow objects from JSONL file."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Windows JSONL not found: {jsonl_path}")

    windows = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                window = FinetuneWindow(**record)
                windows.append(window)
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"Error parsing line {line_num}: {e}") from e

    _LOGGER.info(f"Loaded {len(windows)} FinetuneWindow records from {jsonl_path}")
    return windows


def standardize_coordinates(
    metadata_df: pd.DataFrame,
) -> tuple[StandardScaler, NormalizationArtifact]:
    """Fit standardizer on coordinate columns and return artifacts."""
    coords = metadata_df[["latitude", "longitude"]].values
    scaler = StandardScaler()
    scaler.fit(coords)

    artifact = NormalizationArtifact(
        feature_names=["latitude", "longitude"],
        means=scaler.mean_.tolist(),
        stds=scaler.scale_.tolist(),
    )

    _LOGGER.info(f"Fitted coordinate standardizer: means={artifact.means}, stds={artifact.stds}")
    return scaler, artifact


def build_dataloaders(
    windows: list[FinetuneWindow],
    metadata_df: pd.DataFrame,
    current_fold: int = 0,
    n_splits: int = 5,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, NormalizationArtifact]:
    """Build train and validation dataloaders with k-fold split.

    Returns:
        Tuple of (train_loader, val_loader, normalization_artifact).
    """
    folds = create_stratified_group_kfold(metadata_df, n_splits, random_state)
    scaler, artifact = standardize_coordinates(metadata_df)
    fold_assignments = assign_fold_indices(metadata_df, folds, current_fold)

    train_dataset = JaguarGeoDataset(
        windows, metadata_df, scaler, fold_assignments, current_fold, "train"
    )
    val_dataset = JaguarGeoDataset(
        windows, metadata_df, scaler, fold_assignments, current_fold, "val"
    )

    _LOGGER.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=RandomSampler(train_dataset),
        collate_fn=collate_mtl_batch,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=collate_mtl_batch,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, artifact
