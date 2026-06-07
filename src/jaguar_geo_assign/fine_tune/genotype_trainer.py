"""Genotype MLP training with LOOCV and optional Optuna optimization.

This module implements the full training pipeline for the genotype
matrix + VES-based transfer learning jaguar geographic assignment.

The pipeline uses Optuna to search the hyperparameter space by running full
LOOCV for each trial, minimizing median haversine distance. Loci are
optionally weighted or selected using pre-computed Variant Effect Scores
from the felid-pretrained DNABERT-2 backbone.

Each LOOCV fold trains on N-1 individuals as a single batch (N~57 individuals
fit trivially in memory) and predicts the held-out individual. This avoids
data leakage in coordinate normalization and missing-data imputation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import set_seed
from torch import Tensor
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from jaguar_geo_assign.config import load_genotype_finetune_config
from jaguar_geo_assign.fine_tune.dataset import BIOME_CLASSES, CoordStats, _fit_coord_stats
from jaguar_geo_assign.fine_tune.genotype_dataset import (
    GenotypeMatrixResult,
    build_genotype_matrix,
    impute_missing_genotypes,
    load_genotype_matrix,
    save_genotype_matrix,
)
from jaguar_geo_assign.fine_tune.genotype_model import (
    GenotypeMLPConfig,
    JaguarGenotypeMLP,
    apply_ves_selection,
    apply_ves_weighting,
    compute_genotype_loss,
    compute_ves_gate_init,
)
from jaguar_geo_assign.fine_tune.trainer import compute_eval_metrics, haversine_distance_km
from jaguar_geo_assign.fine_tune.variant_scoring import (
    compute_variant_effect_scores,
    load_ves_scores,
    save_ves_scores,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LOOCVPrediction:
    """Prediction for one held-out individual in LOOCV.

    Attributes:
        individual_id: Unique individual identifier from the metadata.
        sample_id: VCF sample identifier for the individual.
        true_lat: Ground-truth latitude in decimal degrees (WGS-84).
        true_lon: Ground-truth longitude in decimal degrees (WGS-84).
        pred_lat: Predicted latitude in decimal degrees.
        pred_lon: Predicted longitude in decimal degrees.
        haversine_km: Great-circle distance between predicted and true
            coordinates in kilometres.
        true_biome: Ground-truth biome population label.
        pred_biome: Predicted biome label (argmax of biome logits).
        biome_correct: Whether the predicted biome matches the true biome.
        biome_logits: Raw logits for all biome classes, ordered to match
            :data:`BIOME_CLASSES`.
    """

    individual_id: str
    sample_id: str
    true_lat: float
    true_lon: float
    pred_lat: float
    pred_lon: float
    haversine_km: float
    true_biome: str
    pred_biome: str
    biome_correct: bool
    biome_logits: list[float]


@dataclass(frozen=True)
class GenotypeTrainResult:
    """Summary of a completed genotype MLP training run.

    Attributes:
        predictions: Full list of per-individual LOOCV predictions.
        haversine_km_median: Median haversine distance across all folds.
        haversine_km_mean: Mean haversine distance across all folds.
        accuracy: Biome classification accuracy across all folds.
        macro_f1: Macro-averaged F1 score across all biome classes.
        per_class_f1: Per-biome F1 scores keyed by biome name.
        per_biome_haversine: Median haversine per true biome.
        hyperparams: Hyperparameters used for this run.
        output_dir: Directory where results are saved.
    """

    predictions: list[LOOCVPrediction]
    haversine_km_median: float
    haversine_km_mean: float
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    per_biome_haversine: dict[str, float]
    hyperparams: dict[str, Any]
    output_dir: Path


def _train_single_fold(
    *,
    genotypes_f32: Tensor,
    latitudes: list[float],
    longitudes: list[float],
    biome_indices: Tensor,
    biome_labels: list[str],
    individual_ids: list[str],
    sample_ids: list[str],
    train_idx: list[int],
    eval_idx: int,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    ves_init_logits: Tensor | None = None,
) -> LOOCVPrediction:
    """Train on N-1 individuals, predict the held-out one.

    This is the inner loop of LOOCV. The function:
    1. Fits ``CoordStats`` on training individuals only.
    2. Prepares degree-space coordinate targets for haversine loss.
    3. Builds the MLP model, optimizer, and cosine LR scheduler.
    4. Trains for ``max_epochs`` on the full training batch.
    5. Predicts the held-out individual in eval mode.
    6. Denormalizes predictions and computes haversine distance.

    Args:
        genotypes_f32: Float32 genotype matrix of shape
            ``(n_individuals, n_features)`` after VES integration and
            imputation.
        latitudes: Per-individual latitudes in decimal degrees.
        longitudes: Per-individual longitudes in decimal degrees.
        biome_indices: Integer biome class indices of shape
            ``(n_individuals,)``.
        biome_labels: Per-individual biome population label strings.
        individual_ids: Per-individual identifiers.
        sample_ids: Per-individual VCF sample identifiers.
        train_idx: Row indices of training individuals.
        eval_idx: Row index of the single held-out individual.
        config: Hyperparameter dictionary with keys ``n_hidden_layers``,
            ``hidden_dim``, ``dropout``, ``learning_rate``, ``weight_decay``,
            ``coord_loss_weight``, ``cls_loss_weight``, ``max_epochs``.
        seed: Random seed for reproducibility within this fold.
        device: Target device for tensors and model parameters.
        ves_init_logits: Optional gate initialization logits from
            :func:`compute_ves_gate_init`.  When provided, the model
            uses a learnable per-locus gate initialized from VES scores.

    Returns:
        A :class:`LOOCVPrediction` for the held-out individual.
    """
    set_seed(seed)

    # Step 1: Fit CoordStats from training individuals only.
    records = [
        {
            "individual_id": individual_ids[i],
            "latitude": latitudes[i],
            "longitude": longitudes[i],
        }
        for i in train_idx
    ]
    coord_stats = _fit_coord_stats(records)

    # Step 2: Prepare degree-space coordinate targets for haversine loss.
    train_coords_deg = torch.tensor(
        [[latitudes[i], longitudes[i]] for i in train_idx],
        dtype=torch.float32,
    ).to(device)
    train_biomes = biome_indices[train_idx].to(device)
    train_geno = genotypes_f32[train_idx].to(device)

    n_features = train_geno.shape[1]
    n_biomes = len(BIOME_CLASSES)

    # Step 3: Build model, optimizer, scheduler.
    mlp_config = GenotypeMLPConfig(
        n_input_features=n_features,
        n_biomes=n_biomes,
        n_hidden_layers=int(config.get("n_hidden_layers", 2)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.2)),
    )
    model = JaguarGenotypeMLP(mlp_config, ves_init_logits=ves_init_logits).to(device)

    max_epochs = int(config.get("max_epochs", 500))
    learning_rate = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = max(1, int(max_epochs * 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_epochs,
    )

    coord_loss_weight = float(config.get("coord_loss_weight", 1.0))
    cls_loss_weight = float(config.get("cls_loss_weight", 1.0))

    # Step 4: Train for max_epochs.
    model.train()
    for _ in range(max_epochs):
        outputs = model(train_geno)
        total_loss, *_ = compute_genotype_loss(
            outputs,
            train_coords_deg,
            train_biomes,
            coord_stats=coord_stats,
            coord_loss_weight=coord_loss_weight,
            cls_loss_weight=cls_loss_weight,
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    # Step 5: Predict held-out individual.
    model.eval()
    eval_geno = genotypes_f32[eval_idx : eval_idx + 1].to(device)

    with torch.no_grad():
        eval_outputs = model(eval_geno)

    # Step 6: Denormalize predictions and compute haversine.
    pred_lat_z = eval_outputs.coordinate[0, 0].item()
    pred_lon_z = eval_outputs.coordinate[0, 1].item()
    pred_lat = pred_lat_z * coord_stats.lat_std + coord_stats.lat_mean
    pred_lon = pred_lon_z * coord_stats.lon_std + coord_stats.lon_mean

    true_lat = latitudes[eval_idx]
    true_lon = longitudes[eval_idx]

    pred_deg = torch.tensor([[pred_lat, pred_lon]], dtype=torch.float32)
    target_deg = torch.tensor([[true_lat, true_lon]], dtype=torch.float32)
    hav_km = float(haversine_distance_km(pred_deg, target_deg).item())

    # Biome prediction.
    biome_logits = eval_outputs.biome_logits[0].detach().cpu().tolist()
    pred_biome_idx = int(torch.tensor(biome_logits).argmax().item())
    pred_biome = BIOME_CLASSES[pred_biome_idx]
    true_biome = biome_labels[eval_idx]
    biome_correct = pred_biome == true_biome

    return LOOCVPrediction(
        individual_id=individual_ids[eval_idx],
        sample_id=sample_ids[eval_idx],
        true_lat=true_lat,
        true_lon=true_lon,
        pred_lat=pred_lat,
        pred_lon=pred_lon,
        haversine_km=hav_km,
        true_biome=true_biome,
        pred_biome=pred_biome,
        biome_correct=biome_correct,
        biome_logits=biome_logits,
    )


def run_loocv(
    *,
    geno_result: GenotypeMatrixResult,
    ves_scores: Tensor | None,
    ves_mode: str,
    ves_top_k: int | None,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> GenotypeTrainResult:
    """Run full N-fold LOOCV with the given hyperparameters.

    Steps:
    1. Apply VES integration (weighting, selection, or none) to the
       genotype matrix.
    2. For each individual *i* in ``0..N-1``:
       a. Impute missing genotypes using training allele frequencies.
       b. Hold out individual *i*.
       c. Train on the remaining N-1 individuals.
       d. Predict individual *i*.
    3. Aggregate predictions into summary metrics.
    4. Save results to ``output_dir``.

    Args:
        geno_result: Dense genotype matrix with per-individual metadata.
        ves_scores: Pre-computed VES tensor of shape ``(n_loci,)``, or
            ``None`` when VES integration is disabled.
        ves_mode: One of ``"weighted"``, ``"selection"``, or ``"none"``.
        ves_top_k: Number of top loci for selection mode.
        config: Hyperparameter dictionary forwarded to
            :func:`_train_single_fold`.
        seed: Base random seed. Each fold uses ``seed + fold_index``.
        device: Target device for training.
        output_dir: Directory to write per-individual predictions and
            aggregate metrics.
    Returns:
        A :class:`GenotypeTrainResult` summarizing the LOOCV run.

    Raises:
        ValueError: If ``ves_mode`` is not one of the recognized values, or
            if required scores are not provided.
    """
    n_individuals = geno_result.genotypes.shape[0]
    n_loci = geno_result.genotypes.shape[1]
    logger.info(
        "Starting LOOCV: %d individuals, %d loci, ves_mode=%s",
        n_individuals,
        n_loci,
        ves_mode,
    )

    # Step 1: Validate VES mode and resolve effective_top_k.
    # The actual VES integration (weighting or selection) is applied per-fold
    # after imputation to avoid data leakage.
    effective_top_k: int | None = None
    ves_init_logits: Tensor | None = None
    if ves_mode == "weighted":
        if ves_scores is None:
            raise ValueError("ves_mode='weighted' requires ves_scores to be provided")
        logger.info("VES mode: weighted (all %d loci, scaled by |VES|).", n_loci)
    elif ves_mode == "selection":
        if ves_scores is None:
            raise ValueError("ves_mode='selection' requires ves_scores to be provided")
        effective_top_k = ves_top_k if ves_top_k is not None else n_loci
        logger.info("VES mode: selection (top %d of %d loci by |VES|).", effective_top_k, n_loci)
    elif ves_mode == "learnable":
        if ves_scores is None:
            raise ValueError("ves_mode='learnable' requires ves_scores to be provided")
        ves_init_logits = compute_ves_gate_init(ves_scores)
        logger.info("VES mode: learnable gates (all %d loci, gate init from VES).", n_loci)
    elif ves_mode == "none":
        logger.info("VES integration disabled; using raw genotypes.")
    else:
        raise ValueError(
            f"Unrecognized ves_mode={ves_mode!r}; "
            "expected 'weighted', 'selection', 'learnable', or 'none'"
        )

    # Build biome index tensor.
    biome_to_idx = {biome: idx for idx, biome in enumerate(BIOME_CLASSES)}
    biome_indices = torch.tensor(
        [biome_to_idx[b] for b in geno_result.biome_labels],
        dtype=torch.long,
    )

    # Step 2: LOOCV loop.
    predictions: list[LOOCVPrediction] = []
    for fold_idx in range(n_individuals):
        train_idx = [j for j in range(n_individuals) if j != fold_idx]

        # Per-fold imputation: compute allele frequencies from training
        # individuals only, then impute missing values for all individuals.
        # The int8 genotypes must be used as source for imputation since
        # VES-weighted values are continuous and not suitable for allele
        # frequency estimation.
        if ves_mode == "weighted":
            imputed_raw = impute_missing_genotypes(
                geno_result.genotypes,
                train_idx,
                seed=seed + fold_idx,
            )
            imputed_f32 = imputed_raw.float()
            fold_genotypes = apply_ves_weighting(imputed_f32, ves_scores)
        elif ves_mode == "selection":
            imputed_raw = impute_missing_genotypes(
                geno_result.genotypes,
                train_idx,
                seed=seed + fold_idx,
            )
            imputed_f32 = imputed_raw.float()
            fold_genotypes, _ = apply_ves_selection(
                imputed_f32,
                ves_scores,
                effective_top_k,
            )
        elif ves_mode == "learnable":
            imputed_raw = impute_missing_genotypes(
                geno_result.genotypes,
                train_idx,
                seed=seed + fold_idx,
            )
            fold_genotypes = imputed_raw.float()
        else:
            imputed_raw = impute_missing_genotypes(
                geno_result.genotypes,
                train_idx,
                seed=seed + fold_idx,
            )
            fold_genotypes = imputed_raw.float()

        pred = _train_single_fold(
            genotypes_f32=fold_genotypes,
            latitudes=geno_result.latitudes,
            longitudes=geno_result.longitudes,
            biome_indices=biome_indices,
            biome_labels=geno_result.biome_labels,
            individual_ids=geno_result.individual_ids,
            sample_ids=geno_result.sample_ids,
            train_idx=train_idx,
            eval_idx=fold_idx,
            config=config,
            seed=seed + fold_idx,
            device=device,
            ves_init_logits=ves_init_logits,
        )
        predictions.append(pred)

        if (fold_idx + 1) % 10 == 0 or fold_idx == n_individuals - 1:
            logger.info(
                "LOOCV progress: %d / %d folds completed (last haversine=%.1f km)",
                fold_idx + 1,
                n_individuals,
                pred.haversine_km,
            )

    # Step 3: Aggregate predictions into metrics.
    all_haversines = torch.tensor([p.haversine_km for p in predictions])
    haversine_median = float(all_haversines.median().item())
    haversine_mean = float(all_haversines.mean().item())
    accuracy = sum(1 for p in predictions if p.biome_correct) / len(predictions)

    # Compute macro F1 and per-class F1 using the existing compute_eval_metrics.
    n_biomes = len(BIOME_CLASSES)
    all_cls_logits = torch.stack([torch.tensor(p.biome_logits) for p in predictions])
    all_coord_pred = torch.stack([torch.tensor([p.pred_lat, p.pred_lon]) for p in predictions])
    all_biome_label = torch.tensor(
        [biome_to_idx[p.true_biome] for p in predictions],
        dtype=torch.long,
    )
    all_coord_target = torch.stack([torch.tensor([p.true_lat, p.true_lon]) for p in predictions])
    # Use identity CoordStats since predictions are already in degree space.
    identity_stats = CoordStats(lat_mean=0.0, lat_std=1.0, lon_mean=0.0, lon_std=1.0)
    eval_metrics = compute_eval_metrics(
        all_cls_logits,
        all_coord_pred,
        all_biome_label,
        all_coord_target,
        identity_stats,
        n_biomes=n_biomes,
    )
    macro_f1 = eval_metrics["macro_f1"]

    per_class_f1: dict[str, float] = {}
    for biome_name in BIOME_CLASSES:
        key = f"per_class_f1_{biome_name}"
        if key in eval_metrics:
            per_class_f1[biome_name] = eval_metrics[key]

    # Per-biome haversine: group by true biome, compute median AND mean.
    per_biome_haversine: dict[str, float] = {}
    per_biome_haversine_mean: dict[str, float] = {}
    per_biome_accuracy: dict[str, float] = {}
    biome_groups: dict[str, list[float]] = defaultdict(list)
    biome_correct_counts: dict[str, int] = defaultdict(int)
    biome_total_counts: dict[str, int] = defaultdict(int)
    for p in predictions:
        biome_groups[p.true_biome].append(p.haversine_km)
        biome_total_counts[p.true_biome] += 1
        if p.biome_correct:
            biome_correct_counts[p.true_biome] += 1
    for biome_name, havs in biome_groups.items():
        havs_t = torch.tensor(havs)
        per_biome_haversine[biome_name] = float(havs_t.median().item())
        per_biome_haversine_mean[biome_name] = float(havs_t.mean().item())
        per_biome_accuracy[biome_name] = (
            biome_correct_counts[biome_name] / biome_total_counts[biome_name]
        )

    # Distance threshold metrics (aligned with Zenato Lazzari 2025).
    pct_within_250 = sum(1 for p in predictions if p.haversine_km <= 250) / len(predictions)
    pct_within_500 = sum(1 for p in predictions if p.haversine_km <= 500) / len(predictions)
    pct_within_1000 = sum(1 for p in predictions if p.haversine_km <= 1000) / len(predictions)

    # Log diagnostics.
    pred_biome_counts: dict[str, int] = defaultdict(int)
    for p in predictions:
        pred_biome_counts[p.pred_biome] += 1
    logger.info("LOOCV per-class prediction counts: %s", dict(pred_biome_counts))
    logger.info(
        "LOOCV aggregate: haversine_median=%.1f km, haversine_mean=%.1f km, "
        "accuracy=%.4f, macro_f1=%.4f",
        haversine_median,
        haversine_mean,
        accuracy,
        macro_f1,
    )
    logger.info(
        "LOOCV distance thresholds: within_250km=%.1f%%, within_500km=%.1f%%, within_1000km=%.1f%%",
        pct_within_250 * 100,
        pct_within_500 * 100,
        pct_within_1000 * 100,
    )
    logger.info("LOOCV per-biome haversine (median): %s", per_biome_haversine)
    logger.info("LOOCV per-biome haversine (mean): %s", per_biome_haversine_mean)
    logger.info("LOOCV per-biome accuracy: %s", per_biome_accuracy)
    logger.info("LOOCV per-class F1: %s", per_class_f1)

    # Step 4: Save results.
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_json = [asdict(p) for p in predictions]
    predictions_path = output_dir / "loocv_predictions.json"
    predictions_path.write_text(
        json.dumps(predictions_json, indent=2),
        encoding="utf-8",
    )

    summary = {
        "haversine_km_median": haversine_median,
        "haversine_km_mean": haversine_mean,
        "pct_within_250km": pct_within_250,
        "pct_within_500km": pct_within_500,
        "pct_within_1000km": pct_within_1000,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_biome_haversine_median": per_biome_haversine,
        "per_biome_haversine_mean": per_biome_haversine_mean,
        "per_biome_accuracy": per_biome_accuracy,
        "n_individuals": n_individuals,
        "n_features": int(fold_genotypes.shape[1]),
        "ves_mode": ves_mode,
        "hyperparams": config,
    }
    summary_path = output_dir / "loocv_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved LOOCV results to %s", output_dir)

    return GenotypeTrainResult(
        predictions=predictions,
        haversine_km_median=haversine_median,
        haversine_km_mean=haversine_mean,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        per_biome_haversine=per_biome_haversine,
        hyperparams=config,
        output_dir=output_dir,
    )


def _build_ensemble_predictions(
    *,
    trial_dirs: list[Path],
    trial_values: list[float],
    output_dir: Path,
) -> GenotypeTrainResult | None:
    """Average coordinate predictions from the top-K LOOCV trial results.

    For each held-out individual, the ensemble averages predicted lat/lon
    across trials and takes the majority-vote biome prediction. This reduces
    variance from stochastic training without retraining.

    Args:
        trial_dirs: Ordered list of trial output directories (best first).
        trial_values: Corresponding Optuna objective values for logging.
        output_dir: Directory to write ensemble results.

    Returns:
        A :class:`GenotypeTrainResult` for the ensemble, or ``None`` if
        trial prediction files cannot be loaded.
    """
    all_trial_preds: list[list[dict[str, Any]]] = []
    for trial_dir in trial_dirs:
        pred_path = trial_dir / "loocv_predictions.json"
        if not pred_path.exists():
            logger.warning("Ensemble: missing predictions at %s, skipping.", pred_path)
            continue
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        all_trial_preds.append(preds)

    if len(all_trial_preds) < 2:
        logger.warning("Ensemble requires at least 2 trials with predictions; skipping.")
        return None

    n_individuals = len(all_trial_preds[0])
    for trial_preds in all_trial_preds:
        if len(trial_preds) != n_individuals:
            logger.warning("Ensemble: mismatched prediction counts across trials; skipping.")
            return None

    ensemble_predictions: list[LOOCVPrediction] = []
    for i in range(n_individuals):
        first = all_trial_preds[0][i]

        avg_lat = sum(tp[i]["pred_lat"] for tp in all_trial_preds) / len(all_trial_preds)
        avg_lon = sum(tp[i]["pred_lon"] for tp in all_trial_preds) / len(all_trial_preds)

        biome_votes: dict[str, int] = defaultdict(int)
        for tp in all_trial_preds:
            biome_votes[tp[i]["pred_biome"]] += 1
        pred_biome = max(biome_votes, key=biome_votes.get)  # type: ignore[arg-type]

        pred_deg = torch.tensor([[avg_lat, avg_lon]], dtype=torch.float32)
        target_deg = torch.tensor([[first["true_lat"], first["true_lon"]]], dtype=torch.float32)
        hav_km = float(haversine_distance_km(pred_deg, target_deg).item())

        avg_logits = [
            sum(tp[i]["biome_logits"][c] for tp in all_trial_preds) / len(all_trial_preds)
            for c in range(len(first["biome_logits"]))
        ]

        ensemble_predictions.append(
            LOOCVPrediction(
                individual_id=first["individual_id"],
                sample_id=first["sample_id"],
                true_lat=first["true_lat"],
                true_lon=first["true_lon"],
                pred_lat=avg_lat,
                pred_lon=avg_lon,
                haversine_km=hav_km,
                true_biome=first["true_biome"],
                pred_biome=pred_biome,
                biome_correct=pred_biome == first["true_biome"],
                biome_logits=avg_logits,
            )
        )

    all_haversines = torch.tensor([p.haversine_km for p in ensemble_predictions])
    haversine_median = float(all_haversines.median().item())
    haversine_mean = float(all_haversines.mean().item())
    accuracy = sum(1 for p in ensemble_predictions if p.biome_correct) / len(ensemble_predictions)

    n_biomes = len(BIOME_CLASSES)
    biome_to_idx = {biome: idx for idx, biome in enumerate(BIOME_CLASSES)}
    all_cls_logits = torch.stack([torch.tensor(p.biome_logits) for p in ensemble_predictions])
    all_coord_pred = torch.stack(
        [torch.tensor([p.pred_lat, p.pred_lon]) for p in ensemble_predictions]
    )
    all_biome_label = torch.tensor(
        [biome_to_idx[p.true_biome] for p in ensemble_predictions], dtype=torch.long
    )
    all_coord_target = torch.stack(
        [torch.tensor([p.true_lat, p.true_lon]) for p in ensemble_predictions]
    )
    identity_stats = CoordStats(lat_mean=0.0, lat_std=1.0, lon_mean=0.0, lon_std=1.0)
    eval_metrics = compute_eval_metrics(
        all_cls_logits,
        all_coord_pred,
        all_biome_label,
        all_coord_target,
        identity_stats,
        n_biomes=n_biomes,
    )

    per_class_f1: dict[str, float] = {}
    for biome_name in BIOME_CLASSES:
        key = f"per_class_f1_{biome_name}"
        if key in eval_metrics:
            per_class_f1[biome_name] = eval_metrics[key]

    per_biome_haversine: dict[str, float] = {}
    biome_groups: dict[str, list[float]] = defaultdict(list)
    for p in ensemble_predictions:
        biome_groups[p.true_biome].append(p.haversine_km)
    for biome_name, havs in biome_groups.items():
        per_biome_haversine[biome_name] = float(torch.tensor(havs).median().item())

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_json = [asdict(p) for p in ensemble_predictions]
    (output_dir / "loocv_predictions.json").write_text(
        json.dumps(predictions_json, indent=2), encoding="utf-8"
    )
    summary = {
        "haversine_km_median": haversine_median,
        "haversine_km_mean": haversine_mean,
        "accuracy": accuracy,
        "macro_f1": eval_metrics["macro_f1"],
        "per_class_f1": per_class_f1,
        "per_biome_haversine": per_biome_haversine,
        "ensemble_k": len(all_trial_preds),
        "trial_values": trial_values[: len(all_trial_preds)],
    }
    (output_dir / "loocv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved ensemble results to %s", output_dir)

    return GenotypeTrainResult(
        predictions=ensemble_predictions,
        haversine_km_median=haversine_median,
        haversine_km_mean=haversine_mean,
        accuracy=accuracy,
        macro_f1=eval_metrics["macro_f1"],
        per_class_f1=per_class_f1,
        per_biome_haversine=per_biome_haversine,
        hyperparams={"ensemble_k": len(all_trial_preds)},
        output_dir=output_dir,
    )


def run_genotype_training(config_path: str | Path) -> GenotypeTrainResult:
    """Main entry point for genotype MLP training.

    Orchestrates the full pipeline:
    1. Load and validate configuration via :func:`load_genotype_finetune_config`.
    2. Build or load the genotype matrix.
    3. Compute or load VES scores.
    4. Run Optuna hyperparameter optimization where each trial executes
       a full LOOCV run.
    5. Save final results.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        A :class:`GenotypeTrainResult` from the best LOOCV run.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required config fields are missing or invalid.
    """

    config = load_genotype_finetune_config(config_path)
    logger.info("Loaded genotype training config from %s", config_path)

    # Resolve device.
    device_name = config.device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device(device_name)
    logger.info("Using device: %s", device)

    seed = config.seed
    set_seed(seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Build or load genotype matrix.
    genotype_cache_dir = config.genotype_cache_dir
    if genotype_cache_dir is not None and Path(genotype_cache_dir).exists():
        logger.info("Loading cached genotype matrix from %s", genotype_cache_dir)
        geno_result = load_genotype_matrix(genotype_cache_dir)
    else:
        logger.info("Building genotype matrix from VCF: %s", config.vcf_path)
        geno_result = build_genotype_matrix(config.vcf_path, config.metadata_csv)
        if genotype_cache_dir is not None:
            save_genotype_matrix(geno_result, genotype_cache_dir)
            logger.info("Cached genotype matrix to %s", genotype_cache_dir)

    n_individuals = geno_result.genotypes.shape[0]
    n_loci = geno_result.genotypes.shape[1]
    logger.info("Genotype matrix: %d individuals x %d loci", n_individuals, n_loci)

    # Step 3: Compute or load VES scores.
    ves_mode = config.ves_mode
    ves_scores: Tensor | None = None
    ves_top_k = config.ves_top_k

    if ves_mode != "none":
        ves_scores_path = (
            Path(config.genotype_cache_dir) / "ves_scores"
            if config.genotype_cache_dir is not None
            else None
        )
        if ves_scores_path is not None and ves_scores_path.with_suffix(".pt").exists():
            logger.info("Loading cached VES scores from %s", ves_scores_path)
            ves_result = load_ves_scores(ves_scores_path)
            ves_scores = ves_result.scores
        else:
            if not config.compute_ves:
                raise ValueError(
                    f"ves_mode={ves_mode!r} requires VES scores but compute_ves=false "
                    "and no cached scores found."
                )
            logger.info("Computing VES scores from backbone: %s", config.backbone_path)
            ves_result = compute_variant_effect_scores(
                geno_result.locus_info,
                config.reference_fasta,
                config.backbone_path,
                batch_size=config.ves_batch_size,
                device=device_name,
            )
            ves_scores = ves_result.scores
            if ves_scores_path is not None:
                save_ves_scores(ves_result, ves_scores_path)
                logger.info("Cached VES scores to %s", ves_scores_path)

    # Step 4: Optuna LOOCV optimization.
    import optuna

    optuna_n_trials = config.optuna_n_trials
    optuna_study_name = config.optuna_study_name
    logger.info(
        "Starting Optuna hyperparameter optimization: %d trials",
        optuna_n_trials,
    )

    best_result: GenotypeTrainResult | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_result

        trial_config: dict[str, Any] = {
            "n_hidden_layers": trial.suggest_int("n_hidden_layers", 3, 4),
            "hidden_dim": trial.suggest_int("hidden_dim", 200, 512, log=True),
            "dropout": trial.suggest_float("dropout", 0.03, 0.25),
            "learning_rate": trial.suggest_float("learning_rate", 1.5e-3, 1.5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True),
            "coord_loss_weight": trial.suggest_float("coord_loss_weight", 0.5, 6.0, log=True),
            "cls_loss_weight": (
                0.0
                if config.cls_loss_weight == 0.0
                else trial.suggest_float("cls_loss_weight", 0.1, 3.0, log=True)
            ),
            "max_epochs": trial.suggest_int("max_epochs", 200, 800),
        }
        if ves_mode == "selection":
            trial_config["ves_top_k"] = trial.suggest_int("ves_top_k", 2000, 5000)
            trial_ves_top_k = trial_config["ves_top_k"]
        else:
            trial_ves_top_k = ves_top_k

        trial_dir = output_dir / f"trial_{trial.number:04d}"
        result = run_loocv(
            geno_result=geno_result,
            ves_scores=ves_scores,
            ves_mode=ves_mode,
            ves_top_k=trial_ves_top_k,
            config=trial_config,
            seed=seed,
            device=device,
            output_dir=trial_dir,
        )

        logger.info(
            "Optuna trial %d: haversine_median=%.1f km, accuracy=%.4f, macro_f1=%.4f",
            trial.number,
            result.haversine_km_median,
            result.accuracy,
            result.macro_f1,
        )

        if best_result is None or result.haversine_km_median < best_result.haversine_km_median:
            best_result = result

        return result.haversine_km_median

    study = optuna.create_study(
        direction="minimize",
        study_name=optuna_study_name,
    )
    study.optimize(objective, n_trials=optuna_n_trials)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError(
            f"All {len(study.trials)} Optuna trials failed. "
            "Check logs for per-trial errors (CUDA OOM, imputation failures, etc.)."
        )

    logger.info(
        "Optuna optimization complete. Best trial: %d, "
        "best haversine_median=%.1f km, best params=%s",
        study.best_trial.number,
        study.best_trial.value,
        study.best_trial.params,
    )

    # Save Optuna study summary.
    optuna_summary = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
    }
    optuna_summary_path = output_dir / "optuna_summary.json"
    optuna_summary_path.write_text(
        json.dumps(optuna_summary, indent=2),
        encoding="utf-8",
    )

    # Ensemble: average predictions from the top-K trials.
    completed_trials = [t for t in study.trials if t.value is not None]
    ensemble_k = min(5, len(completed_trials))
    top_trials = sorted(completed_trials, key=lambda t: t.value)[:ensemble_k]
    top_trial_dirs = [output_dir / f"trial_{t.number:04d}" for t in top_trials]

    ensemble_result = _build_ensemble_predictions(
        trial_dirs=top_trial_dirs,
        trial_values=[t.value for t in top_trials],
        output_dir=output_dir / "ensemble",
    )
    if ensemble_result is not None:
        logger.info(
            "Ensemble (top-%d trials): haversine_median=%.1f km, accuracy=%.4f, macro_f1=%.4f",
            ensemble_k,
            ensemble_result.haversine_km_median,
            ensemble_result.accuracy,
            ensemble_result.macro_f1,
        )
        if ensemble_result.haversine_km_median < best_result.haversine_km_median:
            logger.info(
                "Ensemble improves over best single trial (%.1f -> %.1f km).",
                best_result.haversine_km_median,
                ensemble_result.haversine_km_median,
            )
            best_result = ensemble_result

    assert best_result is not None
    return best_result


def format_genotype_train_result(result: GenotypeTrainResult) -> str:
    """Format a :class:`GenotypeTrainResult` for human-readable CLI output."""

    pct_250 = sum(1 for p in result.predictions if p.haversine_km <= 250) / max(
        len(result.predictions), 1
    )
    pct_500 = sum(1 for p in result.predictions if p.haversine_km <= 500) / max(
        len(result.predictions), 1
    )
    pct_1000 = sum(1 for p in result.predictions if p.haversine_km <= 1000) / max(
        len(result.predictions), 1
    )

    lines = [
        "Jaguar genotype MLP training completed.",
        "",
        "  Geographic assignment:",
        f"    Haversine median:    {result.haversine_km_median:.1f} km",
        f"    Haversine mean:      {result.haversine_km_mean:.1f} km",
        f"    Within 250 km:       {pct_250 * 100:.1f}%",
        f"    Within 500 km:       {pct_500 * 100:.1f}%  (paper: 65-69%)",
        f"    Within 1000 km:      {pct_1000 * 100:.1f}%",
        "",
        "  Biome classification:",
        f"    Accuracy:            {result.accuracy * 100:.1f}%  (paper: 98%)",
        f"    Macro F1:            {result.macro_f1:.4f}",
        "",
        f"  Output: {result.output_dir}",
    ]
    if result.per_biome_haversine:
        lines.append("")
        lines.append("  Per-biome haversine (median / mean km)          Paper mean (km)")
        paper_mean = {
            "Amazon": 707.7,
            "Atlantic Forest": 125.4,
            "Caatinga": 80.5,
            "Cerrado": 492.6,
            "Pantanal": 195.6,
        }
        biome_groups: dict[str, list[float]] = defaultdict(list)
        for p in result.predictions:
            biome_groups[p.true_biome].append(p.haversine_km)
        for biome in sorted(result.per_biome_haversine):
            median_val = result.per_biome_haversine[biome]
            havs = biome_groups.get(biome, [])
            mean_val = sum(havs) / max(len(havs), 1) if havs else 0.0
            paper_val = paper_mean.get(biome, 0.0)
            lines.append(
                f"    {biome:<20s} {median_val:>7.1f} / {mean_val:>7.1f}"
                f"                    {paper_val:>7.1f}"
            )
    if result.per_class_f1:
        lines.append("")
        lines.append("  Per-biome F1:")
        for biome in sorted(result.per_class_f1):
            lines.append(f"    {biome:<20s} {result.per_class_f1[biome]:.4f}")
    return "\n".join(lines)


__all__ = [
    "GenotypeTrainResult",
    "LOOCVPrediction",
    "format_genotype_train_result",
    "run_genotype_training",
    "run_loocv",
]
