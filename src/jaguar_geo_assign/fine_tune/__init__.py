"""Fine-tuning stage for DNABERT-2 on jaguar geographic assignment.

Implements multi-task learning (MTL) fine-tuning combining coordinate regression
and biome classification. Includes:

- mtl_model: GeographicAssignmentMTL with classification + regression heads
- mtl_dataset: JaguarGeoDataset for combining sequence inputs with geographic labels
- split: Stratified Group K-Fold splitting (grouped by individual, stratified by biome)
- data_loader: Pipeline for loading windows, metadata, standardizing coordinates,
  and building PyTorch DataLoaders
- mtl_training: Full training loop with accelerate, mixed-precision, and TensorBoard

The module ensures data leakage prevention: all windows from the same individual
are assigned to the same fold (train or validation) to prevent individual leakage
that would artificially inflate metrics.
"""

from jaguar_geo_assign.fine_tune.data_loader import (
    build_dataloaders,
    collate_mtl_batch,
    compute_sample_weights,
    load_finetune_windows_jsonl,
    load_metadata_csv,
    standardize_coordinates,
)
from jaguar_geo_assign.fine_tune.mtl_dataset import JaguarGeoDataset, NormalizationArtifact
from jaguar_geo_assign.fine_tune.split import assign_fold_indices, create_stratified_group_kfold

__all__ = [
    "JaguarGeoDataset",
    "NormalizationArtifact",
    "create_stratified_group_kfold",
    "assign_fold_indices",
    "load_metadata_csv",
    "load_finetune_windows_jsonl",
    "standardize_coordinates",
    "build_dataloaders",
    "collate_mtl_batch",
    "compute_sample_weights",
]
