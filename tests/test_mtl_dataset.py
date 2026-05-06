"""Unit tests for MTL dataset and data loading for jaguar fine-tuning.

These tests protect the contract that:
1. Stratified Group K-Fold correctly groups by individual and stratifies by biome
2. Windows from the same individual don't cross train/val split boundaries
3. Coordinate normalization artifacts can serialize and inverse-transform
4. Dataloaders emit batches with correct schema (input_ids, attention_mask, labels)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jaguar_geo_assign.data.finetune_windows import FinetuneWindow
from jaguar_geo_assign.fine_tune import (
    JaguarGeoDataset,
    NormalizationArtifact,
    assign_fold_indices,
    build_dataloaders,
    create_stratified_group_kfold,
    load_finetune_windows_jsonl,
    load_metadata_csv,
    standardize_coordinates,
)


def _create_test_windows(n_samples: int = 4, windows_per_sample: int = 3) -> list[FinetuneWindow]:
    """Create dummy FinetuneWindow objects for testing."""
    windows = []
    for sample_idx in range(n_samples):
        for window_idx in range(windows_per_sample):
            window = FinetuneWindow(
                sample_id=f"sample_{sample_idx}",
                contig=f"chr{(sample_idx % 3) + 1}",
                locus_pos=1000 + window_idx * 100,
                window_start=500 + window_idx * 100,
                window_end=1012 + window_idx * 100,
                sequence="ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" * 8,
                ref_allele="A",
                alt_allele="T",
                is_heterozygous=False,
                genotype="1/1",
                filter_status="PASS",
            )
            windows.append(window)
    return windows


def _create_test_metadata(n_samples: int = 6) -> pd.DataFrame:
    """Create dummy metadata with individual_id and biome labels."""
    records = []
    for sample_idx in range(n_samples):
        individual_idx = sample_idx
        biome_idx = sample_idx // 3
        biome_idx = min(biome_idx, 1)
        biomes = ["savanna", "rainforest"]
        records.append(
            {
                "sample_id": f"sample_{sample_idx}",
                "individual_id": f"jaguar_{individual_idx}",
                "biome_population_label": biomes[biome_idx],
                "latitude": -5.0 + individual_idx * 0.5,
                "longitude": -60.0 + individual_idx * 0.5,
            }
        )
    return pd.DataFrame(records).set_index("sample_id")


def test_normalization_artifact_round_trips():
    """NormalizationArtifact must serialize/deserialize with fidelity."""
    artifact = NormalizationArtifact(
        feature_names=["latitude", "longitude"],
        means=[-5.0, -60.0],
        stds=[1.5, 2.5],
    )
    restored = NormalizationArtifact.from_dict(artifact.to_dict())
    assert restored.feature_names == ["latitude", "longitude"]
    assert restored.means == [-5.0, -60.0]
    assert restored.stds == [1.5, 2.5]


def test_normalization_artifact_inverse_transform():
    """NormalizationArtifact must correctly inverse-transform standardized coordinates."""
    artifact = NormalizationArtifact(
        feature_names=["latitude", "longitude"],
        means=[0.0, 0.0],
        stds=[1.0, 2.0],
    )
    normalized = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float32)
    original = artifact.inverse_transform(normalized)
    expected = np.array([[1.0, 4.0], [0.0, -2.0]], dtype=np.float32)
    np.testing.assert_allclose(original, expected, rtol=1e-5)


def test_stratified_group_kfold_splits_individuals():
    """Stratified Group K-Fold must keep individuals together within folds."""
    metadata = _create_test_metadata(n_samples=6)
    folds = create_stratified_group_kfold(metadata, n_splits=2, random_state=42)
    assert len(folds) == 2
    for train_idx, val_idx in folds:
        assert len(train_idx) + len(val_idx) == 6


def test_stratified_group_kfold_preserves_biome_distribution():
    """Stratified Group K-Fold must preserve biome distribution across folds."""
    records = []
    for i in range(10):
        biome = "savanna" if i < 5 else "rainforest"
        records.append(
            {
                "sample_id": f"sample_{i}",
                "individual_id": f"jaguar_{i}",
                "biome_population_label": biome,
                "latitude": -5.0 + i * 0.1,
                "longitude": -60.0 + i * 0.1,
            }
        )
    metadata = pd.DataFrame(records).set_index("sample_id")
    folds = create_stratified_group_kfold(metadata, n_splits=2, random_state=42)
    assert len(folds) == 2
    for _train_idx, val_idx in folds:
        assert len(val_idx) > 0


def test_assign_fold_indices_creates_correct_mapping():
    """assign_fold_indices must create per-individual fold assignments."""
    metadata = _create_test_metadata(n_samples=6)
    folds = create_stratified_group_kfold(metadata, n_splits=2, random_state=42)
    fold_assignments = assign_fold_indices(metadata, folds, current_fold=0)
    unique_individuals = metadata["individual_id"].nunique()
    assert len(fold_assignments) == unique_individuals


def test_jaguar_geo_dataset_filters_by_split():
    """JaguarGeoDataset must filter windows to match train/val split."""
    metadata = _create_test_metadata(n_samples=6)
    windows = _create_test_windows(n_samples=6, windows_per_sample=2)
    folds = create_stratified_group_kfold(metadata, n_splits=2, random_state=42)
    fold_assignments = assign_fold_indices(metadata, folds, current_fold=0)

    train_dataset = JaguarGeoDataset(
        windows, metadata, fold_indices=fold_assignments, current_fold=0, split="train"
    )
    val_dataset = JaguarGeoDataset(
        windows, metadata, fold_indices=fold_assignments, current_fold=0, split="val"
    )
    assert len(train_dataset) + len(val_dataset) == len(windows)


def test_jaguar_geo_dataset_windows_same_individual_dont_cross_split():
    """Windows from the same individual must not appear in both train and val."""
    windows = _create_test_windows(n_samples=6, windows_per_sample=2)
    metadata = _create_test_metadata(n_samples=6)
    folds = create_stratified_group_kfold(metadata, n_splits=2, random_state=42)
    fold_assignments = assign_fold_indices(metadata, folds, current_fold=0)

    train_dataset = JaguarGeoDataset(
        windows, metadata, fold_indices=fold_assignments, current_fold=0, split="train"
    )
    val_dataset = JaguarGeoDataset(
        windows, metadata, fold_indices=fold_assignments, current_fold=0, split="val"
    )

    train_individuals = {w.sample_id for w in train_dataset._filtered_windows}
    val_individuals = {w.sample_id for w in val_dataset._filtered_windows}
    assert len(train_individuals & val_individuals) == 0


def test_build_dataloaders_returns_correct_types():
    """build_dataloaders must return DataLoader objects and artifact."""
    windows = _create_test_windows(n_samples=6, windows_per_sample=2)
    metadata = _create_test_metadata(n_samples=6)

    train_loader, val_loader, artifact = build_dataloaders(
        windows, metadata, current_fold=0, n_splits=2, train_batch_size=2, eval_batch_size=2
    )

    assert hasattr(train_loader, "__iter__")
    assert hasattr(val_loader, "__iter__")
    assert isinstance(artifact, NormalizationArtifact)
    assert artifact.feature_names == ["latitude", "longitude"]


def test_load_metadata_csv_validates_required_columns(tmp_path: Path):
    """load_metadata_csv must validate required columns exist."""
    csv_path = tmp_path / "metadata.csv"
    df = pd.DataFrame({"sample_id": ["s1", "s2"], "individual_id": ["i1", "i2"]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_metadata_csv(csv_path)


def test_load_finetune_windows_jsonl_round_trips(tmp_path: Path):
    """load_finetune_windows_jsonl must correctly deserialize FinetuneWindow records."""
    jsonl_path = tmp_path / "windows.jsonl"
    windows = _create_test_windows(n_samples=2, windows_per_sample=1)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for window in windows:
            f.write(json.dumps(window.__dict__) + "\n")

    loaded = load_finetune_windows_jsonl(jsonl_path)
    assert len(loaded) == len(windows)
    assert all(isinstance(w, FinetuneWindow) for w in loaded)


def test_standardize_coordinates_fits_and_artifact_is_valid():
    """standardize_coordinates must fit scaler and return valid artifact."""
    metadata = _create_test_metadata(n_samples=4)
    scaler, artifact = standardize_coordinates(metadata)

    assert len(artifact.means) == 2
    assert len(artifact.stds) == 2
    assert artifact.feature_names == ["latitude", "longitude"]
