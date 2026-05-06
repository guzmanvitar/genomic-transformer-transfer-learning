"""Stratified Group K-Fold split for preventing data leakage in fine-tuning.

Implements 5-Fold Stratified Group K-Fold split that groups samples by
individual_id (to prevent individual leakage across folds) and stratifies
by biome_population_label (to preserve class distribution across folds).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

_LOGGER = logging.getLogger(__name__)


def create_stratified_group_kfold(
    metadata_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified group k-fold splits grouped by individual and stratified by biome.

    Groups are based on individual_id (to prevent individual leakage across folds) and
    stratification preserves biome distribution. Pre-validates that each biome has at
    least n_splits individuals to ensure valid stratified splits.

    Args:
        metadata_df: DataFrame indexed by sample_id with columns:
            - individual_id: Biological individual identifier
            - biome_population_label: Biome/population label for stratification
        n_splits: Number of folds (default 5).
        random_state: Seed for reproducibility.

    Returns:
        List of (train_indices, val_indices) tuples for each fold.

    Raises:
        ValueError: If any biome has fewer than n_splits individuals (required for stratification).
    """
    if "individual_id" not in metadata_df.columns:
        raise ValueError("metadata_df must have 'individual_id' column")
    if "biome_population_label" not in metadata_df.columns:
        raise ValueError("metadata_df must have 'biome_population_label' column")

    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    # Get unique individuals with their representative biome/metadata
    individual_metadata = metadata_df.groupby("individual_id").first().reset_index()

    _LOGGER.info(f"Creating {n_splits}-fold splits for {len(individual_metadata)} individuals")

    # Validate minimum individuals per biome before attempting stratified split
    unique_biomes = individual_metadata["biome_population_label"].unique()
    for biome in unique_biomes:
        n_individuals_in_biome = (individual_metadata["biome_population_label"] == biome).sum()
        if n_individuals_in_biome < n_splits:
            raise ValueError(
                f"Biome '{biome}' has only {n_individuals_in_biome} individuals but "
                f"n_splits={n_splits} requires at least {n_splits} per biome for stratification."
            )

    # Encode biome labels for stratification
    biome_to_id = {biome: idx for idx, biome in enumerate(unique_biomes)}
    biome_encoded = individual_metadata["biome_population_label"].map(biome_to_id).values

    _LOGGER.info(f"Biomes: {list(unique_biomes)}")

    # Apply StratifiedGroupKFold with actual individual_id as groups
    # This ensures samples from the same individual never cross train/val boundaries
    group_ids = individual_metadata["individual_id"].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(
        sgkf.split(
            individual_metadata,
            y=biome_encoded,
            groups=group_ids,
        )
    )

    _LOGGER.info(f"Created {len(folds)} folds")
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        _LOGGER.info(f"Fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")

    return folds


def assign_fold_indices(
    metadata_df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    current_fold: int,
) -> np.ndarray:
    """Create fold assignment array for all samples.

    Args:
        metadata_df: DataFrame indexed by sample_id with 'individual_id' column.
        folds: List of (train_indices, val_indices) from k-fold split.
        current_fold: Which fold to use (0-indexed).

    Returns:
        Array where each element is fold assignment (current_fold for validation).
    """
    individual_metadata = metadata_df.groupby("individual_id").first().reset_index()
    unique_individuals = individual_metadata["individual_id"].values

    train_idx, val_idx = folds[current_fold]
    val_individuals = set(unique_individuals[val_idx])

    # Create fold assignment
    fold_assignments = np.zeros(len(unique_individuals), dtype=np.int32)
    for i, ind_id in enumerate(unique_individuals):
        if ind_id in val_individuals:
            fold_assignments[i] = current_fold
        else:
            fold_assignments[i] = -1

    return fold_assignments
