# ruff: noqa: F722  # jaxtyping shape annotations use string-based dimensions
"""Single-phase MIL fine-tuning on offline jaguar embedding bags.

This module keeps the original per-window MTL trainer intact as a rollback path
while introducing a dedicated full-bag trainer that consumes the offline
embedding shards and the positional MIL model.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from jaguar_geo_assign.config import MILFinetuneConfig, load_mil_finetune_config
from jaguar_geo_assign.fine_tune.dataset import BIOME_CLASSES, CoordStats
from jaguar_geo_assign.fine_tune.mil_dataset import (
    MILBagDataset,
    build_mil_fold_dataloaders,
    mil_collate_fn,
)
from jaguar_geo_assign.fine_tune.model import JaguarMTLOutput
from jaguar_geo_assign.fine_tune.positional_mil import JaguarPositionalMILNetwork
from jaguar_geo_assign.fine_tune.trainer import (
    _compute_baselines,
    _compute_grad_norm,
    _compute_mtl_loss,
    compute_eval_metrics,
)
from jaguar_geo_assign.pretrain.foundation_training import _save_json_atomically, atomic_dir_replace

logger = logging.getLogger(__name__)
Tensor = torch.Tensor


@dataclass(frozen=True)
class MILTrainResult:
    """Summary of one completed jaguar MIL fine-tuning run."""

    fold_index: int
    steps_completed: int
    best_eval_haversine_km: float | None
    best_eval_macro_f1: float | None
    output_dir: Path
    coord_stats: CoordStats


def _build_mil_optimizer_and_scheduler(
    model: nn.Module,
    *,
    lr_mil: float,
    weight_decay: float,
    warmup_fraction: float,
    total_steps: int,
) -> tuple[AdamW, Any]:
    """Construct AdamW plus cosine warmup for the MIL training phase."""

    no_decay = {"bias", "LayerNorm.weight"}
    decay_params: list[torch.nn.Parameter] = []
    nodecay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = nodecay_params if any(marker in name for marker in no_decay) else decay_params
        target.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr_mil,
    )
    warmup_steps = max(0, int(math.floor(warmup_fraction * total_steps)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _batch_mil_outputs(
    outputs: JaguarMTLOutput,
    batch: dict[str, Tensor],
) -> tuple[JaguarMTLOutput, dict[str, Tensor]]:
    """Unsqueeze MIL outputs and targets to match the shared MTL loss contract."""

    batched_outputs = JaguarMTLOutput(
        coordinate=outputs.coordinate.unsqueeze(0),
        biome_logits=(
            outputs.biome_logits.unsqueeze(0) if outputs.biome_logits is not None else None
        ),
    )
    batched_batch = dict(batch)
    batched_batch["coord_target"] = batch["coord_target"].unsqueeze(0)
    if "biome_label" in batch:
        batched_batch["biome_label"] = batch["biome_label"].reshape(1)
    return batched_outputs, batched_batch


def _gather_for_metrics(accelerator: Accelerator, tensor: Tensor) -> Tensor:
    """Gather metrics tensors when supported, otherwise fall back to identity."""

    gather = getattr(accelerator, "gather_for_metrics", None)
    if callable(gather):
        return gather(tensor)
    return tensor


def _serialize_mil_config(config: MILFinetuneConfig) -> dict[str, Any]:
    """Convert ``MILFinetuneConfig`` into JSON-serializable primitives."""

    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _run_mil_evaluation(
    *,
    model: nn.Module,
    eval_loader: Any,
    accelerator: Accelerator,
    coord_stats: CoordStats,
    config: MILFinetuneConfig,
    cls_loss_weight: float,
    reg_loss_weight: float,
    huber_delta: float,
    global_step: int,
    best_eval_haversine_km: float | None,
    best_eval_macro_f1: float | None,
    output_dir: Path,
    eval_max_steps: int | None = None,
) -> tuple[float, float | None, float | None, dict[str, float]]:
    """Run MIL evaluation and update the best checkpoint when metrics improve."""

    model.eval()
    cls_list: list[Tensor] = []
    coord_pred_list: list[Tensor] = []
    biome_list: list[Tensor] = []
    coord_tgt_list: list[Tensor] = []
    eval_total_loss = 0.0
    eval_steps = 0

    with torch.no_grad():
        eval_batch_count = 0
        for batch in eval_loader:
            if eval_max_steps is not None and eval_batch_count >= eval_max_steps:
                break
            eval_batch_count += 1

            batch = {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}
            outputs = model(
                embeddings=batch["embeddings"],
                bp_positions=batch["bp_positions"],
            )
            batched_outputs, batched_batch = _batch_mil_outputs(outputs, batch)
            eval_loss, _, _ = _compute_mtl_loss(
                batched_outputs,
                batched_batch,
                cls_loss_weight=cls_loss_weight,
                reg_loss_weight=reg_loss_weight,
                huber_delta=huber_delta,
            )
            eval_loss_detached = eval_loss.detach().float()
            if torch.isfinite(eval_loss_detached).all():
                eval_total_loss += float(eval_loss_detached.mean().item())
                eval_steps += 1
                coord_pred_list.append(outputs.coordinate.unsqueeze(0).detach().cpu())
                coord_tgt_list.append(batch["coord_target"].unsqueeze(0).detach().cpu())
                if "biome_label" in batch:
                    biome_list.append(batch["biome_label"].reshape(1).detach().cpu())
                if outputs.biome_logits is not None:
                    cls_list.append(outputs.biome_logits.unsqueeze(0).detach().cpu())

    if not coord_pred_list or eval_steps == 0:
        mean_eval_loss = eval_total_loss / max(eval_steps, 1) if eval_steps > 0 else float("nan")
        model.train()
        return mean_eval_loss, best_eval_haversine_km, best_eval_macro_f1, {}

    all_coord_pred = _gather_for_metrics(accelerator, torch.cat(coord_pred_list))
    all_coord_tgt = _gather_for_metrics(accelerator, torch.cat(coord_tgt_list))
    if biome_list:
        all_biome = _gather_for_metrics(accelerator, torch.cat(biome_list))
    else:
        all_biome = torch.empty(all_coord_pred.shape[0], dtype=torch.long)
    if cls_list:
        all_cls = _gather_for_metrics(accelerator, torch.cat(cls_list))
    else:
        all_cls = torch.empty(all_coord_pred.shape[0], 0, dtype=torch.float32)

    metrics = compute_eval_metrics(
        all_cls,
        all_coord_pred,
        all_biome,
        all_coord_tgt,
        coord_stats,
        n_biomes=config.n_biomes,
    )
    mean_eval_loss = eval_total_loss / max(eval_steps, 1) if eval_steps > 0 else float("nan")
    accelerator.log(
        {
            "eval/total_loss": mean_eval_loss,
            "eval/accuracy": metrics["accuracy"],
            "eval/macro_f1": metrics["macro_f1"],
            "eval/mae_lat_deg": metrics["mae_lat_deg"],
            "eval/mae_lon_deg": metrics["mae_lon_deg"],
            "eval/haversine_km_mean": metrics["haversine_km_mean"],
            "eval/haversine_km_median": metrics["haversine_km_median"],
        },
        step=global_step,
    )
    if accelerator.is_main_process:
        logger.info(
            "[Eval @ step %d] haversine_km_median=%.2f haversine_km_mean=%.2f "
            "macro_f1=%.4f eval_loss=%.4f",
            global_step,
            metrics["haversine_km_median"],
            metrics["haversine_km_mean"],
            metrics["macro_f1"],
            mean_eval_loss,
        )

    current_hav = metrics["haversine_km_median"]
    current_f1 = metrics["macro_f1"]
    is_better = False
    if math.isfinite(current_hav):
        if best_eval_haversine_km is None or current_hav < best_eval_haversine_km:
            is_better = True
        elif (
            best_eval_haversine_km is not None
            and math.isclose(current_hav, best_eval_haversine_km)
            and current_f1 > (best_eval_macro_f1 or -1.0)
        ):
            is_better = True

    if is_better:
        best_eval_haversine_km = float(current_hav)
        best_eval_macro_f1 = float(current_f1)
        if accelerator.is_main_process:
            logger.info(
                "New best checkpoint at step %d: haversine_km_median=%.2f macro_f1=%.4f",
                global_step,
                best_eval_haversine_km,
                best_eval_macro_f1,
            )
            best_dir = output_dir / "best"
            unwrapped = accelerator.unwrap_model(model)
            with atomic_dir_replace(best_dir) as tmp_best:
                torch.save(unwrapped.state_dict(), tmp_best / "mil_model.pt")
                _save_json_atomically(
                    tmp_best / "coord_norm.json",
                    {
                        "lat_mean": float(coord_stats.lat_mean),
                        "lat_std": float(coord_stats.lat_std),
                        "lon_mean": float(coord_stats.lon_mean),
                        "lon_std": float(coord_stats.lon_std),
                    },
                )
                _save_json_atomically(
                    tmp_best / "metrics.json",
                    {
                        "haversine_km_median": current_hav,
                        "macro_f1": current_f1,
                        "fold_index": int(config.fold_index),
                        "step": int(global_step),
                    },
                )
                _save_json_atomically(tmp_best / "config.json", _serialize_mil_config(config))

    model.train()
    return mean_eval_loss, best_eval_haversine_km, best_eval_macro_f1, metrics


def run_jaguar_mil_training(config_path: str | Path) -> MILTrainResult:
    """Run single-phase jaguar MIL training over offline embedding bags."""

    config = load_mil_finetune_config(config_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_state_path = output_dir / "latest" / "train_state.json"
    if train_state_path.exists():
        raise RuntimeError(
            "run_jaguar_mil_training: resume from existing checkpoint is not implemented yet; "
            "remove the 'latest' directory or implement the full resume protocol before re-running."
        )

    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Building MIL dataloaders...")
    train_loader, eval_loader, coord_stats = build_mil_fold_dataloaders(config)
    logger.info("Dataloaders ready.")
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(output_dir / config.tensorboard_subdir),
    )
    model = JaguarPositionalMILNetwork(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_biomes=config.n_biomes,
        dropout_prob=config.dropout,
        locus_dropout=config.locus_dropout,
        genome_scale=config.genome_scale,
    )
    optimizer, scheduler = _build_mil_optimizer_and_scheduler(
        model,
        lr_mil=config.lr_mil,
        weight_decay=config.weight_decay,
        warmup_fraction=config.warmup_fraction,
        total_steps=config.mil_steps,
    )
    accelerator.init_trackers("jaguar_mil_training")

    logger.info("Computing baselines (iterating train+eval sets)...")
    baseline_metrics = _compute_baselines(
        train_loader=train_loader,
        eval_loader=eval_loader,
        coord_stats=coord_stats,
        n_biomes=config.n_biomes,
    )
    logger.info(
        "Baselines: haversine_km_median=%.2f macro_f1=%.4f",
        baseline_metrics["haversine_km_median"],
        baseline_metrics["macro_f1"],
    )
    accelerator.log(
        {
            "baseline/macro_f1": baseline_metrics["macro_f1"],
            "baseline/haversine_km_median": baseline_metrics["haversine_km_median"],
            "baseline/haversine_km_mean": baseline_metrics["haversine_km_mean"],
        },
        step=0,
    )

    model, optimizer, scheduler, train_loader, eval_loader = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        train_loader,
        eval_loader,
    )

    mil_steps_completed = 0
    best_eval_haversine_km: float | None = None
    best_eval_macro_f1: float | None = None
    patience_counter = 0
    early_stopped = False
    nan_attention_steps = 0
    skipped_steps = 0
    train_loss_sum = 0.0
    train_cls_loss_sum = 0.0
    train_reg_loss_sum = 0.0
    train_loss_count = 0
    last_eval_haversine: float | None = None
    plateau_window: deque[float] = deque(maxlen=1000)

    logger.info(
        "Starting MIL training: %d steps, log_every=%d, eval_every=%d",
        config.mil_steps,
        config.log_every,
        config.eval_every,
    )
    model.train()
    while mil_steps_completed < config.mil_steps and not early_stopped:
        for batch in train_loader:
            if mil_steps_completed >= config.mil_steps or early_stopped:
                break

            batch = {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}
            if batch["embeddings"].shape[-1] != config.embedding_dim:
                raise ValueError(
                    "MIL embedding_dim mismatch: config expects "
                    f"{config.embedding_dim}, batch provides {batch['embeddings'].shape[-1]}"
                )

            with accelerator.accumulate(model):
                outputs = model(
                    embeddings=batch["embeddings"],
                    bp_positions=batch["bp_positions"],
                )
                batched_outputs, batched_batch = _batch_mil_outputs(outputs, batch)
                total_loss, cls_loss, reg_loss = _compute_mtl_loss(
                    batched_outputs,
                    batched_batch,
                    cls_loss_weight=config.cls_loss_weight,
                    reg_loss_weight=config.reg_loss_weight,
                    huber_delta=config.huber_delta,
                )
                loss_detached = total_loss.detach().float()
                if not torch.isfinite(loss_detached).all():
                    nan_attention_steps += 1
                    skipped_steps += 1
                    continue

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    if config.gradient_clip > 0.0:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), config.gradient_clip
                        )
                    else:
                        grad_norm = _compute_grad_norm(model)
                    skip_step = not torch.isfinite(grad_norm.detach()).all()
                    if skip_step:
                        skipped_steps += 1
                    else:
                        optimizer.step()
                        scheduler.step()
                        mil_steps_completed += 1
                        train_loss_sum += float(loss_detached.item())
                        train_cls_loss_sum += float(cls_loss.detach().item())
                        train_reg_loss_sum += float(reg_loss.detach().item())
                        train_loss_count += 1

                        if mil_steps_completed % config.log_every == 0:
                            denom = max(train_loss_count, 1)
                            grad_norm_value = float(grad_norm.detach().float().item())
                            accelerator.log(
                                {
                                    "mil/train_loss": train_loss_sum / denom,
                                    "mil/cls_loss": train_cls_loss_sum / denom,
                                    "mil/reg_loss": train_reg_loss_sum / denom,
                                    "mil/grad_norm": grad_norm_value,
                                    "mil/learning_rate": scheduler.get_last_lr()[0],
                                    "mil/nan_attention_steps": float(nan_attention_steps),
                                },
                                step=mil_steps_completed,
                            )
                            if accelerator.is_main_process:
                                logger.info(
                                    "[Step %d/%d] loss=%.4f cls_loss=%.4f reg_loss=%.4f "
                                    "grad_norm=%.4f lr=%.2e skipped=%d",
                                    mil_steps_completed,
                                    config.mil_steps,
                                    train_loss_sum / denom,
                                    train_cls_loss_sum / denom,
                                    train_reg_loss_sum / denom,
                                    grad_norm_value,
                                    scheduler.get_last_lr()[0],
                                    skipped_steps,
                                )
                            train_loss_sum = 0.0
                            train_cls_loss_sum = 0.0
                            train_reg_loss_sum = 0.0
                            train_loss_count = 0

                        if mil_steps_completed % config.eval_every == 0:
                            prev_best_hav = best_eval_haversine_km
                            (
                                mean_eval_loss,
                                best_eval_haversine_km,
                                best_eval_macro_f1,
                                eval_metrics,
                            ) = _run_mil_evaluation(
                                model=model,
                                eval_loader=eval_loader,
                                accelerator=accelerator,
                                coord_stats=coord_stats,
                                config=config,
                                cls_loss_weight=config.cls_loss_weight,
                                reg_loss_weight=config.reg_loss_weight,
                                huber_delta=config.huber_delta,
                                global_step=mil_steps_completed,
                                best_eval_haversine_km=best_eval_haversine_km,
                                best_eval_macro_f1=best_eval_macro_f1,
                                output_dir=output_dir,
                                eval_max_steps=config.eval_max_steps,
                            )
                            del mean_eval_loss
                            improved = best_eval_haversine_km != prev_best_hav
                            if eval_metrics and improved:
                                patience_counter = 0
                            elif eval_metrics:
                                patience_counter += 1
                                if accelerator.is_main_process:
                                    logger.info(
                                        "No improvement at step %d (patience %d/%s).",
                                        mil_steps_completed,
                                        patience_counter,
                                        config.patience if config.patience is not None else "inf",
                                    )

                            current_hav = eval_metrics.get("haversine_km_median", float("nan"))
                            plateau_detected = 0.0
                            if math.isfinite(current_hav):
                                last_eval_haversine = float(current_hav)
                                plateau_window.append(last_eval_haversine)
                                if len(plateau_window) == plateau_window.maxlen:
                                    window_min = min(plateau_window)
                                    window_max = max(plateau_window)
                                    if (
                                        window_min > 0.0
                                        and ((window_max - window_min) / window_min) < 0.05
                                    ):
                                        plateau_detected = 1.0
                            accelerator.log(
                                {"mil/plateau_detected": plateau_detected},
                                step=mil_steps_completed,
                            )

                            if config.patience is not None and patience_counter >= config.patience:
                                if accelerator.is_main_process:
                                    logger.info(
                                        "Early stopping triggered at step %d after %d eval cycles "
                                        "without improvement.",
                                        mil_steps_completed,
                                        patience_counter,
                                    )
                                early_stopped = True
                                break

                        if (
                            mil_steps_completed % config.save_every == 0
                            and accelerator.is_main_process
                        ):
                            _save_json_atomically(
                                train_state_path,
                                {
                                    "step": mil_steps_completed,
                                    "best_eval_haversine_km": best_eval_haversine_km,
                                    "best_eval_macro_f1": best_eval_macro_f1,
                                },
                            )

                    optimizer.zero_grad()
            if early_stopped:
                break

    logger.info(
        "MIL training complete: steps=%d best_haversine_km=%.2f best_macro_f1=%.4f",
        mil_steps_completed,
        best_eval_haversine_km if best_eval_haversine_km is not None else float("nan"),
        best_eval_macro_f1 if best_eval_macro_f1 is not None else float("nan"),
    )
    accelerator.end_training()
    return MILTrainResult(
        fold_index=int(config.fold_index),
        steps_completed=mil_steps_completed,
        best_eval_haversine_km=best_eval_haversine_km,
        best_eval_macro_f1=best_eval_macro_f1,
        output_dir=output_dir,
        coord_stats=coord_stats,
    )


def integration_test(
    *,
    bag_size: int = 84_000,
    embedding_dim: int = 32,
    hidden_dim: int = 16,
    mixed_precision: str = "bf16",
) -> None:
    """Run a synthetic MIL smoke test on a full-size bag without the real backbone.

    The bag length matches the production-scale contract (~84k loci) while the
    embedding width remains intentionally small so CPU-only CI can validate the
    O(N) training path without incurring the full DNABERT-2 hidden-size cost.
    """

    if bag_size <= 0:
        raise ValueError("bag_size must be positive for integration_test")
    if embedding_dim <= 0 or hidden_dim <= 0:
        raise ValueError("embedding_dim and hidden_dim must be positive for integration_test")

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_path = Path(tmp_root)
        output_dir = tmp_path / "mil_integration_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = tmp_path / "individual-0.pt"
        torch.save(
            {
                "embeddings": torch.randn(bag_size, embedding_dim, dtype=torch.float32),
                "bp_positions": torch.arange(1, bag_size + 1, dtype=torch.float32),
                "contigs": ["chr1"] * bag_size,
            },
            shard_path,
        )
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "individual_id": "individual-0",
                    "shard_path": str(shard_path),
                    "n_windows": bag_size,
                    "sample_id": "sample-0",
                    "latitude": 1.0,
                    "longitude": 2.0,
                    "biome_population_label": BIOME_CLASSES[0],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        coord_stats = CoordStats(lat_mean=0.0, lat_std=1.0, lon_mean=0.0, lon_std=1.0)
        dataset = MILBagDataset(
            manifest_path=manifest_path,
            individual_ids=["individual-0"],
            coord_stats=coord_stats,
            biome_to_idx={BIOME_CLASSES[0]: 0, BIOME_CLASSES[1]: 1},
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)
        batch = next(iter(loader))

        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=1,
            log_with="tensorboard",
            project_dir=str(output_dir / "tensorboard"),
        )
        model = JaguarPositionalMILNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_biomes=2,
            dropout_prob=0.0,
            locus_dropout=0.1,
            genome_scale=1.0,
        )
        optimizer = AdamW(model.parameters(), lr=1e-3)
        accelerator.init_trackers("jaguar_mil_integration_test")
        model, optimizer = accelerator.prepare(model, optimizer)
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        outputs = model(embeddings=batch["embeddings"], bp_positions=batch["bp_positions"])
        batched_outputs, batched_batch = _batch_mil_outputs(outputs, batch)
        total_loss, cls_loss, reg_loss = _compute_mtl_loss(
            batched_outputs,
            batched_batch,
            cls_loss_weight=1.0,
            reg_loss_weight=0.5,
            huber_delta=1.0,
        )
        assert torch.isfinite(total_loss.detach()).all(), "MIL integration loss must be finite"
        assert torch.isfinite(cls_loss.detach()).all(), "MIL integration cls_loss must be finite"
        assert torch.isfinite(reg_loss.detach()).all(), "MIL integration reg_loss must be finite"

        with accelerator.accumulate(model):
            accelerator.backward(total_loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            assert torch.isfinite(grad_norm.detach()).all(), (
                "MIL integration grad_norm must be finite"
            )
            optimizer.step()
            optimizer.zero_grad()

        config = MILFinetuneConfig(
            embeddings_dir=tmp_path,
            metadata_csv=tmp_path / "<unused_metadata>",
            output_dir=output_dir,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_folds=2,
            n_biomes=2,
            mil_steps=1,
            mixed_precision=mixed_precision,
            eval_every=1,
            save_every=1,
        )
        mean_eval_loss, best_eval_hav, best_eval_f1, _ = _run_mil_evaluation(
            model=model,
            eval_loader=[batch],
            accelerator=accelerator,
            coord_stats=coord_stats,
            config=config,
            cls_loss_weight=1.0,
            reg_loss_weight=0.5,
            huber_delta=1.0,
            global_step=1,
            best_eval_haversine_km=None,
            best_eval_macro_f1=None,
            output_dir=output_dir,
        )
        assert math.isfinite(mean_eval_loss), "MIL integration eval loss must be finite"
        assert best_eval_hav is not None, "MIL integration eval should set best haversine"
        assert best_eval_f1 is not None, "MIL integration eval should set best macro_f1"
        best_dir = output_dir / "best"
        assert (best_dir / "mil_model.pt").exists(), "MIL best checkpoint must include mil_model.pt"
        assert (best_dir / "coord_norm.json").exists(), (
            "MIL best checkpoint must include coord_norm.json"
        )
        assert (best_dir / "metrics.json").exists(), "MIL best checkpoint must include metrics.json"
        assert (best_dir / "config.json").exists(), "MIL best checkpoint must include config.json"


__all__ = [
    "MILTrainResult",
    "integration_test",
    "run_jaguar_mil_training",
]
