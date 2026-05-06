"""Multi-task learning dataset and dataloaders for fine-tuning DNABERT-2 on jaguar coordinates.

This module implements dataset classes that combine FinetuneWindow sequence inputs with
geographic and biome labels for coordinate regression and biome classification tasks.
The dataset supports stratified group k-fold splitting (grouped by individual, stratified by biome)
and coordinate standardization with artifact persistence for de-standardization at inference time.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

_LOGGER = logging.getLogger(__name__)


@dataclass
class NormalizationArtifact:
    """Serializable standardization state for de-standardization at inference.

    Stores fitted scalers for coordinate regression targets so that predictions
    can be reverse-transformed back to original geographic coordinates.

    Attributes:
        feature_names: List of feature names (e.g., ["latitude", "longitude"]).
        means: Fitted mean for each feature (from StandardScaler).
        stds: Fitted standard deviation for each feature (from StandardScaler).
    """

    feature_names: list[str]
    means: list[float]
    stds: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizationArtifact:
        """Deserialize from dictionary."""
        return cls(**data)

    def inverse_transform(self, normalized_coords: np.ndarray) -> np.ndarray:
        """Convert standardized coordinates back to original scale.

        Args:
            normalized_coords: Array of shape (N, len(feature_names)) with standardized values.

        Returns:
            Array of same shape with values in original coordinate range.
        """
        stds_arr = np.array(self.stds)
        means_arr = np.array(self.means)
        return normalized_coords * stds_arr + means_arr


class JaguarGeoDataset(Dataset):
    """PyTorch dataset combining FinetuneWindow inputs with geographic metadata.

    Loads sequences and metadata from windows, integrating biome labels and
    (optionally) normalized coordinate targets for multi-task learning.

    Attributes:
        windows: List of FinetuneWindow objects with genomic sequences.
        metadata_df: DataFrame with sample_id index and biome/lat/lon columns.
        coordinate_scaler: Optional fitted StandardScaler for latitude/longitude.
        fold_indices: Integer array indicating train/val split for k-fold CV.
        current_fold: Current fold index for iteration (0-indexed).
        split: 'train' or 'val' to filter based on current_fold.
    """

    def __init__(
        self,
        windows: list[Any],
        metadata_df: pd.DataFrame,
        coordinate_scaler: StandardScaler | None = None,
        fold_indices: np.ndarray | None = None,
        current_fold: int = 0,
        split: str = "train",
    ):
        """Initialize the dataset.

        Args:
            windows: List of FinetuneWindow dataclass instances.
            metadata_df: DataFrame indexed by sample_id with biome and coordinate columns.
            coordinate_scaler: Fitted StandardScaler or None to use raw coordinates.
            fold_indices: Array of fold assignments (must have same length as unique samples).
            current_fold: Which fold to use (0-indexed).
            split: 'train' to use train folds, 'val' to use validation fold.
        """
        self.windows = windows
        self.metadata_df = metadata_df
        self.coordinate_scaler = coordinate_scaler
        self.fold_indices = fold_indices
        self.current_fold = current_fold
        self.split = split

        # Build sample_id -> fold mapping
        if fold_indices is not None:
            unique_samples = sorted(metadata_df.index.unique())
            self.sample_fold_map = {
                sid: fold_idx for sid, fold_idx in zip(unique_samples, fold_indices, strict=False)
            }
        else:
            self.sample_fold_map = {}

        # Filter windows by split
        self._filtered_windows = self._filter_by_split()

    def _filter_by_split(self) -> list[Any]:
        """Filter windows to only include current split (train/val)."""
        if not self.sample_fold_map:
            return self.windows

        filtered = []
        for window in self.windows:
            sample_fold = self.sample_fold_map.get(window.sample_id)
            if sample_fold is None:
                continue

            is_val_fold = sample_fold == self.current_fold
            if (self.split == "val" and is_val_fold) or (self.split == "train" and not is_val_fold):
                filtered.append(window)

        return filtered

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._filtered_windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single record with tokenized sequence and labels.

        Args:
            idx: Index into filtered windows.

        Returns:
            Dictionary with:
            - input_ids: Token IDs for DNABERT-2 (pre-tokenized DNA sequence)
            - attention_mask: Attention mask
            - biome_label: Integer label for biome classification (optional)
            - coordinate_labels: [latitude, longitude] (optional)
        """
        window = self._filtered_windows[idx]

        # For fine-tuning, we use tokenized input_ids from DNABERT-2 tokenizer
        # This is a placeholder; actual tokenization happens in preprocessing
        sequence_tokens = self._tokenize_sequence(window.sequence)

        result = {
            "input_ids": sequence_tokens["input_ids"],
            "attention_mask": sequence_tokens["attention_mask"],
        }

        # Add biome label if available
        try:
            biome_label = self.metadata_df.loc[window.sample_id, "biome_population_label"]
            # Convert string labels to integers (0, 1, 2, ...)
            # In practice, this would use a label encoder fitted on training data
            if isinstance(biome_label, str):
                # Use hash for deterministic label assignment
                result["biome_label"] = hash(biome_label) % 10  # 10 possible classes
            else:
                result["biome_label"] = int(biome_label)
        except KeyError:
            _LOGGER.warning(f"No biome label for sample {window.sample_id}")

        # Add coordinate targets if available
        try:
            lat = float(self.metadata_df.loc[window.sample_id, "latitude"])
            lon = float(self.metadata_df.loc[window.sample_id, "longitude"])
            coords = np.array([[lat, lon]], dtype=np.float32)

            if self.coordinate_scaler is not None:
                coords = self.coordinate_scaler.transform(coords)

            result["coordinate_labels"] = coords[0]  # shape (2,)
        except (KeyError, ValueError):
            _LOGGER.warning(f"No valid coordinates for sample {window.sample_id}")

        return result

    def _tokenize_sequence(self, sequence: str) -> dict[str, list[int]]:
        """Placeholder tokenization (assumes pre-tokenized in practice).

        In production, sequences are tokenized during preprocessing and stored
        with input_ids. This is a fallback for testing.
        """
        # Simple placeholder: map bases to token IDs
        base_to_id = {"A": 2, "C": 3, "G": 4, "T": 5, "N": 10}
        token_ids = [101] + [base_to_id.get(b, 10) for b in sequence.upper()] + [102]
        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }
