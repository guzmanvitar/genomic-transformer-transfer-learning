"""Dataset and dataloader helpers for jaguar DNABERT-2 multi-task fine-tuning.

This module owns the data-layer contract for the downstream fine-tuning path:

* BIOME label vocabulary (:data:`BIOME_CLASSES`).
* Coordinate normalisation statistics (:class:`CoordStats`).
* A minimal PyTorch :class:`~torch.utils.data.Dataset` that wraps raw locus
  windows and jaguar metadata (:class:`JaguarMTLDataset`).
* A fold-aware dataloader builder that performs inner-joining, stratified
  group k-fold splitting, and per-individual window equalisation via
  :class:`~torch.utils.data.WeightedRandomSampler`.

All scientific and fairness-related contracts around folds, labels, and
coordinate normalisation are enforced here so that the training loop can
assume well-formed tensors.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from jaguar_geo_assign.config import MtlFinetuneConfig

logger = logging.getLogger(__name__)


BIOME_CLASSES: tuple[str, ...] = (
    "Amazon",
    "Atlantic Forest",
    "Caatinga",
    "Cerrado",
    "Pantanal",
)
"""Approved biome-population labels for jaguar fine-tuning.

The labels are alphabetically sorted for determinism. Any metadata row whose
``biome_population_label`` is not in this tuple is rejected with ``ValueError``;
there is intentionally no "unknown" catch-all class.
"""


@dataclass(frozen=True)
class CoordStats:
    """Simple latitude/longitude normalisation statistics.

    ``lat_std`` and ``lon_std`` are clamped to a minimum of ``1e-6`` during
    construction to guard against division-by-zero when computing z-scores.
    The class provides JSON serialisation helpers so that inference code can
    reload the exact training-time normalisation parameters.
    """

    lat_mean: float
    lat_std: float
    lon_mean: float
    lon_std: float

    def __post_init__(self) -> None:
        # Frozen dataclass; use object.__setattr__ for clamping.
        object.__setattr__(self, "lat_std", max(self.lat_std, 1e-6))
        object.__setattr__(self, "lon_std", max(self.lon_std, 1e-6))

    def to_json(self, path: Path) -> None:
        """Serialise the stats to *path* as a small JSON object.

        The output schema is intentionally minimal and stable so that
        downstream consumers (inference, diagnostics) can reload it
        without importing this module if necessary.
        """

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lat_mean": self.lat_mean,
            "lat_std": self.lat_std,
            "lon_mean": self.lon_mean,
            "lon_std": self.lon_std,
        }
        target.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> CoordStats:
        """Load coordinate statistics from *path* written by :meth:`to_json`."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            lat_mean=float(payload["lat_mean"]),
            lat_std=float(payload["lat_std"]),
            lon_mean=float(payload["lon_mean"]),
            lon_std=float(payload["lon_std"]),
        )


JAGUAR_FINETUNE_METADATA_FIELDS: tuple[str, ...] = (
    "sample_id",  # join key
    "individual_id",  # GroupKFold grouping; sourced from CSV only
    "biome_population_label",  # classification target; validated against BIOME_CLASSES
    "latitude",  # decimal degrees WGS-84
    "longitude",  # decimal degrees WGS-84
)
"""Required metadata CSV fields for the fine-tuning path.

Note that this omits ``locality_id`` from the bootstrap
``JAGUAR_METADATA_FIELDS`` contract: the fine-tuning pipeline does not group,
normalise, or predict on locality IDs, so requiring that column here would
reject otherwise-valid CSVs.
"""


class JaguarMTLDataset(Dataset):
    """PyTorch dataset wrapping joined FinetuneWindow + metadata records.

    Each item yields a dict compatible with the multi-task training loop:

    * ``input_ids``: ``LongTensor[max_length]``
    * ``attention_mask``: ``LongTensor[max_length]``
    * ``biome_label``: scalar ``LongTensor`` (class index into
      :data:`BIOME_CLASSES`)
    * ``coord_target``: ``FloatTensor[2]`` with z-scored (lat, lon)
    """

    def __init__(
        self,
        records: list[Mapping[str, Any]],
        tokenizer: Any,
        coord_stats: CoordStats,
        max_length: int = 512,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self._records: list[Mapping[str, Any]] = list(records)
        self._tokenizer = tokenizer
        self._coord_stats = coord_stats
        self._max_length = max_length
        self._biome_to_idx = {name: idx for idx, name in enumerate(BIOME_CLASSES)}

        # Eagerly validate biome labels so mislabelled rows fail fast.
        for record in self._records:
            label = record.get("biome_population_label")
            if label not in self._biome_to_idx:
                raise ValueError(
                    f"Unknown biome_population_label {label!r}; expected one of {BIOME_CLASSES}"
                )

    def __len__(self) -> int:  # noqa: D401
        """Return the number of joined window records."""

        return len(self._records)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:  # noqa: D401
        """Return the tokenised inputs and targets for a single window."""

        record = self._records[index]
        sequence = str(record["sequence"])

        encoded = self._tokenizer(  # type: ignore[call-arg]
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0).to(dtype=torch.long)
        attention_mask = encoded["attention_mask"].squeeze(0).to(dtype=torch.long)

        biome = record["biome_population_label"]
        biome_idx = self._biome_to_idx[biome]
        biome_label = torch.tensor(biome_idx, dtype=torch.long)

        lat = float(record["latitude"])
        lon = float(record["longitude"])
        lat_z = (lat - self._coord_stats.lat_mean) / self._coord_stats.lat_std
        lon_z = (lon - self._coord_stats.lon_mean) / self._coord_stats.lon_std
        coord_target = torch.tensor([lat_z, lon_z], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "biome_label": biome_label,
            "coord_target": coord_target,
        }


def _load_windows_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load FinetuneWindow records from a JSONL file into a list of dicts."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_metadata_csv(path: Path) -> dict[str, dict[str, Any]]:
    """Load jaguar metadata CSV and index rows by ``sample_id``.

    Validates that all columns in :data:`JAGUAR_FINETUNE_METADATA_FIELDS` are
    present in the header.
    """

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in JAGUAR_FINETUNE_METADATA_FIELDS if column not in header]
        if missing:
            raise ValueError(
                "metadata_csv is missing required columns: "
                f"{missing}. Required by JAGUAR_FINETUNE_METADATA_FIELDS."
            )

        by_sample: dict[str, dict[str, Any]] = {}
        for row in reader:
            sample_id = row.get("sample_id")
            if not sample_id:
                continue
            # Parse coordinates eagerly so downstream code can treat them as floats.
            row["latitude"] = float(row["latitude"])
            row["longitude"] = float(row["longitude"])
            by_sample[sample_id] = row
    return by_sample


def build_fold_dataloaders(
    config: MtlFinetuneConfig,
    tokenizer: Any,
) -> tuple[DataLoader, DataLoader, CoordStats]:
    """Build train and evaluation dataloaders for a single cross-validation fold.

    The procedure is:

    1. Load locus windows from ``config.windows_jsonl``.
    2. Load jaguar metadata from ``config.metadata_csv`` and validate the
       header against :data:`JAGUAR_FINETUNE_METADATA_FIELDS`.
    3. Inner join windows on ``sample_id``; windows whose sample lacks a
       metadata row are dropped and counted. A WARNING log records the
       dropped-window count.
    4. Validate that each biome class has at least ``config.n_folds`` unique
       individuals; otherwise raise ``ValueError``.
    5. Run :class:`StratifiedGroupKFold` over biome labels (stratification)
       and ``individual_id`` (grouping) and select ``config.fold_index``.
    6. Fit :class:`CoordStats` on training indices only.
    7. Build :class:`JaguarMTLDataset` instances for train/eval.
    8. Construct a :class:`WeightedRandomSampler` that equalises window
       contribution per individual in the training split.
    9. Return ``(train_loader, eval_loader, coord_stats)``.
    """

    windows = _load_windows_jsonl(Path(config.windows_jsonl))
    metadata_by_sample = _load_metadata_csv(Path(config.metadata_csv))

    # Inner join on sample_id, sourcing individual_id exclusively from metadata.
    joined: list[dict[str, Any]] = []
    dropped = 0
    for window in windows:
        sample_id = window.get("sample_id")
        if not sample_id or sample_id not in metadata_by_sample:
            dropped += 1
            continue
        meta_row = metadata_by_sample[sample_id]
        record = {**window, **meta_row}
        joined.append(record)

    if dropped:
        logger.warning(
            "Dropped %d windows with missing metadata rows during join (sample_id mismatch)",
            dropped,
        )

    if not joined:
        raise ValueError("No joined records after inner join; cannot build dataloaders")

    biome_to_idx = {name: idx for idx, name in enumerate(BIOME_CLASSES)}

    # Pre-CV validation: unique individuals per biome.
    individuals_per_biome: dict[str, set[str]] = {name: set() for name in BIOME_CLASSES}
    for record in joined:
        biome = record.get("biome_population_label")
        if biome not in biome_to_idx:
            raise ValueError(
                f"Unknown biome_population_label {biome!r}; expected one of {BIOME_CLASSES}"
            )
        individual_id = record[
            "individual_id"
        ]  # individual_id sourced from metadata CSV row, not FinetuneWindow
        individuals_per_biome[biome].add(str(individual_id))

    for biome, individuals in individuals_per_biome.items():
        count = len(individuals)
        if count < config.n_folds:
            raise ValueError(
                f"Biome {biome} has only {count} individuals — cannot create "
                f"{config.n_folds} stratified folds. Minimum required: {config.n_folds}."
            )

    groups = [str(rec["individual_id"]) for rec in joined]
    stratify = [biome_to_idx[rec["biome_population_label"]] for rec in joined]

    cv = StratifiedGroupKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.seed,
    )
    indices = list(range(len(joined)))

    train_indices: Iterable[int] | None = None
    eval_indices: Iterable[int] | None = None
    for fold_idx, (train_idx, eval_idx) in enumerate(cv.split(indices, stratify, groups)):
        if fold_idx == config.fold_index:
            train_indices, eval_indices = train_idx, eval_idx
            break

    if train_indices is None or eval_indices is None:
        raise ValueError(
            f"fold_index {config.fold_index} is out of range for {config.n_folds} folds"
        )

    train_indices = list(train_indices)
    eval_indices = list(eval_indices)

    # Fit CoordStats on train split only.
    train_lats = [float(joined[i]["latitude"]) for i in train_indices]
    train_lons = [float(joined[i]["longitude"]) for i in train_indices]
    if not train_lats or not train_lons:
        raise ValueError("Training split has no records; cannot fit CoordStats")

    lat_mean = sum(train_lats) / len(train_lats)
    lon_mean = sum(train_lons) / len(train_lons)
    lat_var = sum((x - lat_mean) ** 2 for x in train_lats) / len(train_lats)
    lon_var = sum((x - lon_mean) ** 2 for x in train_lons) / len(train_lons)
    coord_stats = CoordStats(
        lat_mean=lat_mean,
        lat_std=math.sqrt(lat_var),
        lon_mean=lon_mean,
        lon_std=math.sqrt(lon_var),
    )

    train_records = [joined[i] for i in train_indices]
    eval_records = [joined[i] for i in eval_indices]

    train_dataset = JaguarMTLDataset(train_records, tokenizer, coord_stats)
    eval_dataset = JaguarMTLDataset(eval_records, tokenizer, coord_stats)

    # Window-count equalisation per individual for training sampler.
    window_count_for_individual: dict[str, int] = {}
    for rec in train_records:
        individual_id = str(rec["individual_id"])
        window_count_for_individual[individual_id] = (
            window_count_for_individual.get(individual_id, 0) + 1
        )

    sample_weights = [
        1.0 / window_count_for_individual[str(rec["individual_id"])] for rec in train_records
    ]
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, eval_loader, coord_stats
