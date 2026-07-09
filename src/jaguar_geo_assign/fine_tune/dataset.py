"""Shared data helpers for jaguar fine-tuning.

This module owns:

* BIOME label vocabulary (:data:`BIOME_CLASSES`).
* Coordinate normalisation statistics (:class:`CoordStats`).
* Metadata CSV loader (:func:`_load_metadata_csv`).
* Per-individual coordinate stats fitting (:func:`_fit_coord_stats`).
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
        """Clamp extremely small standard deviations and log when this occurs.

        The clamping guard ensures downstream z-score computations never
        divide by zero. A WARNING log is emitted when clamping occurs so
        that degenerate coordinate distributions are visible in experiment
        logs rather than silently hidden in the normalisation step.
        """

        # Frozen dataclass; use object.__setattr__ for clamping.
        original_lat_std = self.lat_std
        original_lon_std = self.lon_std
        lat_std = max(original_lat_std, 1e-6)
        lon_std = max(original_lon_std, 1e-6)
        object.__setattr__(self, "lat_std", lat_std)
        object.__setattr__(self, "lon_std", lon_std)
        if lat_std != original_lat_std or lon_std != original_lon_std:
            logger.warning(
                "CoordStats std devs clamped to minimum 1e-6 "
                "(lat_std=%g, lon_std=%g; original lat_std=%g, lon_std=%g)",
                lat_std,
                lon_std,
                original_lat_std,
                original_lon_std,
            )

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
"""Required metadata CSV fields for the fine-tuning path."""


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


def _fit_coord_stats(records: Iterable[Mapping[str, Any]]) -> CoordStats:
    """Fit coordinate normalisation stats with equal per-individual weighting.

    Training uses a sampler that equalises each individual's contribution
    regardless of how many windows that individual produced. ``CoordStats``
    must mirror that contract; otherwise individuals with many windows skew the
    z-score centering/scaling seen by the regression head. To remove that bias,
    this helper first collapses records to one mean coordinate pair per
    ``individual_id`` and then computes the usual unbiased sample standard
    deviation across those per-individual coordinates.
    """

    latitudes_by_individual: dict[str, list[float]] = {}
    longitudes_by_individual: dict[str, list[float]] = {}
    for record in records:
        individual_id = str(record["individual_id"])
        latitudes_by_individual.setdefault(individual_id, []).append(float(record["latitude"]))
        longitudes_by_individual.setdefault(individual_id, []).append(float(record["longitude"]))

    if not latitudes_by_individual or not longitudes_by_individual:
        raise ValueError("Training split has no records; cannot fit CoordStats")

    train_lats = [sum(values) / len(values) for values in latitudes_by_individual.values()]
    train_lons = [sum(values) / len(values) for values in longitudes_by_individual.values()]

    lat_mean = sum(train_lats) / len(train_lats)
    lon_mean = sum(train_lons) / len(train_lons)
    if len(train_lats) > 1:
        lat_var = sum((x - lat_mean) ** 2 for x in train_lats) / (len(train_lats) - 1)
    else:
        lat_var = 0.0
    if len(train_lons) > 1:
        lon_var = sum((x - lon_mean) ** 2 for x in train_lons) / (len(train_lons) - 1)
    else:
        lon_var = 0.0
    return CoordStats(
        lat_mean=lat_mean,
        lat_std=math.sqrt(max(lat_var, 0.0)),
        lon_mean=lon_mean,
        lon_std=math.sqrt(max(lon_var, 0.0)),
    )
