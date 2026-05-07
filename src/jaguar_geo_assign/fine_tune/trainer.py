# ruff: noqa: F722  # jaxtyping shape annotations use string-based dimensions
"""Two-phase DNABERT-2 jaguar multi-task fine-tuning with Accelerate.

This module implements the high-level training loop for the jaguar multi-task
learning (MTL) model.  It wires together the DNABERT-2 backbone, the
coordinate-regression and biome-classification heads, fold-aware dataloaders,
and :mod:`accelerate` to run a two-phase schedule:

* Phase 1 – backbone frozen, heads-only warm-up.
* Phase 2 – last ``unfreeze_layers`` transformer blocks unfrozen with a lower
  learning rate than the heads.

The design mirrors :mod:`jaguar_geo_assign.pretrain.foundation_training` where
possible (bf16 mixed precision, gradient accumulation, atomic best-checkpoint
writes) while remaining small enough for fast CPU-based tests.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from jaguar_geo_assign.config import MtlFinetuneConfig, load_mtl_finetune_config
from jaguar_geo_assign.fine_tune.dataset import BIOME_CLASSES, CoordStats, build_fold_dataloaders
from jaguar_geo_assign.fine_tune.model import JaguarMTLModel
from jaguar_geo_assign.pretrain.foundation_training import _save_json_atomically, atomic_dir_replace

# Alias used only in jaxtyping shape annotations; kept out of runtime logic.
batch = "batch"  # noqa: F841

logger = logging.getLogger(__name__)

Tensor = torch.Tensor


def _load_tokenizer(config: MtlFinetuneConfig, backbone: Any | None = None) -> AutoTokenizer:
    """Load a DNABERT-2-compatible tokenizer from a local backbone path.

    The loader never consults the Hugging Face Hub at runtime; it reads the
    tokenizer configuration from the on-disk ``backbone_path`` directory only.

    To guarantee that ``pad_token_id`` is defined, a three-step fallback is
    applied if the pretrained tokenizer is missing a pad token:

    1. Reuse ``eos_token`` as pad when present.
    2. Otherwise reuse ``unk_token``.
    3. Otherwise add a new ``[PAD]`` token.  When a backbone is supplied, its
       embedding matrix is resized so that model and tokenizer remain aligned.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.backbone_path),
        trust_remote_code=True,
    )

    pad_added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            pad_added = True

    if tokenizer.pad_token is None:
        raise RuntimeError(
            "_load_tokenizer: failed to set pad_token; tokenizer has no eos/unk "
            "token and add_special_tokens did not register a pad token."
        )

    if pad_added and backbone is not None:
        backbone.resize_token_embeddings(len(tokenizer))

    return tokenizer


@jaxtyped(typechecker=beartype)
def haversine_distance_km(
    pred_deg: Float[Tensor, "batch 2"],
    target_deg: Float[Tensor, "batch 2"],
    *,
    radius_km: float = 6371.0,
    epsilon: float = 1e-7,
) -> Float[Tensor, batch]:
    """Compute great-circle distance between coordinate pairs in kilometres.

    The implementation follows the standard Haversine formula with explicit
    numerical guards suitable for mixed-precision training:

    * Inputs are promoted to ``float32`` before trigonometric operations.
    * The intermediate ``a`` term is clamped to ``[epsilon, 1-epsilon]`` to
      avoid invalid values inside :func:`torch.asin` due to rounding.
    * The square root is applied *before* ``asin`` (``asin(sqrt(a))``); omitting
      the square root would systematically over-estimate distances.
    """

    if pred_deg.ndim != 2 or target_deg.ndim != 2:
        raise ValueError("haversine_distance_km expects 2D tensors of shape [B, 2]")
    if pred_deg.shape[-1] != 2 or target_deg.shape[-1] != 2:
        raise ValueError("haversine_distance_km last dimension must be of size 2 (lat, lon)")

    pred = pred_deg.to(dtype=torch.float32)
    target = target_deg.to(dtype=torch.float32)

    lat1, lon1 = torch.unbind(pred, dim=-1)
    lat2, lon2 = torch.unbind(target, dim=-1)

    # Convert degrees to radians explicitly.
    deg2rad = math.pi / 180.0
    lat1 = lat1 * deg2rad
    lon1 = lon1 * deg2rad
    lat2 = lat2 * deg2rad
    lon2 = lon2 * deg2rad

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = torch.sin(dlat * 0.5)
    sin_dlon = torch.sin(dlon * 0.5)
    a = sin_dlat.square() + torch.cos(lat1) * torch.cos(lat2) * sin_dlon.square()
    a_clamped = a.clamp(min=epsilon, max=1.0 - epsilon)
    c = 2.0 * torch.asin(a_clamped.sqrt())
    return radius_km * c


@jaxtyped(typechecker=beartype)
def compute_eval_metrics(
    cls_logits: Float[Tensor, "batch n_biomes"],
    coord_pred: Float[Tensor, "batch 2"],
    biome_label: Int[Tensor, batch],
    coord_target: Float[Tensor, "batch 2"],
    coord_stats: CoordStats,
    n_biomes: int = 5,
) -> dict[str, float]:
    """Compute classification and regression metrics on the full eval set.

    Callers are expected to concatenate and gather outputs across all eval
    batches and DDP ranks *before* invoking this function. Metrics are
    computed on CPU using numerically stable Torch operations only.

    The implementation explicitly supports coordinate-only heads by allowing
    ``cls_logits`` to be empty. In that case, classification metrics are
    reported as ``NaN`` while coordinate metrics remain fully defined.
    """

    with torch.no_grad():
        # ---------------------- classification metrics ----------------------
        per_class_f1: list[float] = []
        if cls_logits.numel() == 0:
            acc = float("nan")
            macro_f1 = float("nan")
        else:
            preds = cls_logits.argmax(dim=1).to("cpu")
            labels = biome_label.to("cpu")

            correct = (preds == labels).sum().item()
            acc = correct / max(int(labels.numel()), 1)

            for k in range(n_biomes):
                true_k = labels == k
                pred_k = preds == k
                tp = (true_k & pred_k).sum().item()
                fp = ((~true_k) & pred_k).sum().item()
                fn = (true_k & (~pred_k)).sum().item()
                if tp == 0 and (fp > 0 or fn > 0):
                    per_class_f1.append(0.0)
                    continue
                denom_p = tp + fp
                denom_r = tp + fn
                if denom_p == 0 or denom_r == 0:
                    per_class_f1.append(0.0)
                    continue
                precision = tp / denom_p
                recall = tp / denom_r
                if precision + recall == 0.0:
                    per_class_f1.append(0.0)
                else:
                    per_class_f1.append(2.0 * precision * recall / (precision + recall))

            if per_class_f1:
                macro_f1 = float(sum(per_class_f1) / len(per_class_f1))
            else:
                macro_f1 = float("nan")

        # --------------------- coordinate metrics (degrees) -----------------
        if coord_pred.numel() == 0 or coord_target.numel() == 0:
            mae_lat = float("nan")
            mae_lon = float("nan")
            hav_mean = float("nan")
            hav_median = float("nan")
        else:
            coord_pred_cpu = coord_pred.to("cpu", dtype=torch.float32)
            coord_target_cpu = coord_target.to("cpu", dtype=torch.float32)

            lat_pred = coord_pred_cpu[:, 0] * float(coord_stats.lat_std) + float(
                coord_stats.lat_mean
            )
            lon_pred = coord_pred_cpu[:, 1] * float(coord_stats.lon_std) + float(
                coord_stats.lon_mean
            )
            lat_true = coord_target_cpu[:, 0] * float(coord_stats.lat_std) + float(
                coord_stats.lat_mean
            )
            lon_true = coord_target_cpu[:, 1] * float(coord_stats.lon_std) + float(
                coord_stats.lon_mean
            )

            mae_lat = (lat_pred - lat_true).abs().mean().item()
            mae_lon = (lon_pred - lon_true).abs().mean().item()

            pred_deg = torch.stack((lat_pred, lon_pred), dim=-1)
            target_deg = torch.stack((lat_true, lon_true), dim=-1)
            hav = haversine_distance_km(pred_deg, target_deg)
            hav_mean = float(hav.mean().item())
            hav_median = float(hav.median().item())

    metrics: dict[str, float] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "mae_lat_deg": float(mae_lat),
        "mae_lon_deg": float(mae_lon),
        "haversine_km_mean": float(hav_mean),
        "haversine_km_median": float(hav_median),
    }

    for idx, biome in enumerate(BIOME_CLASSES[:n_biomes]):
        if idx < len(per_class_f1):
            metrics[f"per_class_f1_{biome}"] = float(per_class_f1[idx])

    return metrics


@dataclass(frozen=True)
class MTLTrainResult:
    """Summary of a jaguar MTL fine-tuning run for a single fold.

    The result object is intentionally lightweight and focused on values needed
    by downstream evaluation or reporting stages rather than exposing internal
    optimiser or scheduler state.
    """

    fold_index: int
    phase1_steps_completed: int
    phase2_steps_completed: int
    best_eval_haversine_km: float | None
    best_eval_macro_f1: float | None
    output_dir: Path
    coord_stats: CoordStats


def _freeze_backbone(model: JaguarMTLModel) -> None:
    """Freeze backbone parameters while keeping heads trainable.

    This helper operates purely on ``requires_grad`` flags and does not touch
    optimizer state; it is safe to call before :func:`accelerator.prepare`.
    """

    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            param.requires_grad = False
        else:
            param.requires_grad = True


def _unfreeze_last_n_layers(model: JaguarMTLModel, n_layers: int) -> None:
    """Unfreeze the last *n_layers* transformer blocks of the backbone.

    The implementation assumes a BERT-style backbone exposing
    ``backbone.encoder.layer`` as a sequence of blocks, which matches
    DNABERT-2.  All earlier layers remain frozen.  Heads are always left
    trainable regardless of *n_layers*.
    """

    if n_layers <= 0:
        return

    backbone = model.backbone
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None) if encoder is not None else None
    if layers is None or not hasattr(layers, "__len__"):
        raise RuntimeError("Backbone does not expose encoder.layer for unfreezing.")

    total_layers = len(layers)
    if n_layers > total_layers:
        raise ValueError("unfreeze_layers exceeds number of transformer blocks in backbone")

    # Freeze all backbone params first.
    for param in backbone.parameters():
        param.requires_grad = False

    # Unfreeze the last n_layers blocks and pooler (if present).
    for layer in layers[total_layers - n_layers :]:
        for param in layer.parameters():
            param.requires_grad = True

    pooler = getattr(backbone, "pooler", None)
    if pooler is not None:
        for param in pooler.parameters():
            param.requires_grad = True

    # Heads stay trainable.
    for param in model.coordinate_head.parameters():
        param.requires_grad = True
    if model.biome_head is not None:
        for param in model.biome_head.parameters():
            param.requires_grad = True


def _build_phase1_optimizer_and_scheduler(
    model: nn.Module,
    *,
    lr_heads: float,
    weight_decay: float,
    warmup_fraction: float,
    total_steps: int,
) -> tuple[AdamW, Any]:
    """Construct Phase 1 AdamW and cosine LR scheduler.

    Only parameters with ``requires_grad=True`` are optimised; callers are
    expected to freeze the backbone via :func:`_freeze_backbone` first.
    Bias and LayerNorm parameters are excluded from weight decay.
    """

    no_decay = {"bias", "LayerNorm.weight"}
    decay_params: list[torch.nn.Parameter] = []
    nodecay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = nodecay_params if any(nd in name for nd in no_decay) else decay_params
        target.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr_heads)

    warmup_steps = max(0, int(math.floor(warmup_fraction * total_steps)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _build_phase2_optimizer_and_scheduler(
    model: nn.Module,
    *,
    lr_backbone: float,
    lr_heads: float,
    weight_decay: float,
    warmup_fraction: float,
    total_steps: int,
) -> tuple[AdamW, Any]:
    """Construct Phase 2 AdamW with differential LRs for backbone and heads."""

    no_decay = {"bias", "LayerNorm.weight"}
    backbone_decay: list[torch.nn.Parameter] = []
    backbone_nodecay: list[torch.nn.Parameter] = []
    head_decay: list[torch.nn.Parameter] = []
    head_nodecay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        target_decay = backbone_decay if is_backbone else head_decay
        target_nodecay = backbone_nodecay if is_backbone else head_nodecay
        if any(nd in name for nd in no_decay):
            target_nodecay.append(param)
        else:
            target_decay.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_decay, "weight_decay": weight_decay, "lr": lr_backbone},
        {"params": backbone_nodecay, "weight_decay": 0.0, "lr": lr_backbone},
        {"params": head_decay, "weight_decay": weight_decay, "lr": lr_heads},
        {"params": head_nodecay, "weight_decay": 0.0, "lr": lr_heads},
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    warmup_steps = max(0, int(math.floor(warmup_fraction * total_steps)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _compute_mtl_loss(
    outputs: Any,
    batch: dict[str, Tensor],
    *,
    cls_loss_weight: float,
    reg_loss_weight: float,
    huber_delta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute weighted multi-task loss components.

    Returns ``(total_loss, cls_loss, reg_loss)`` with all three tensors living
    on the same device as the coordinate predictions.
    """

    coord_pred = outputs.coordinate
    coord_target = batch["coord_target"]

    # For numerical stability and broader device support (e.g. CPU
    # backends without full bf16/fp16 coverage), always compute the
    # regression loss in float32 even when the model runs under
    # mixed-precision. Gradients still back-propagate correctly.
    if coord_pred.dtype in (torch.float16, torch.bfloat16):
        coord_pred_reg = coord_pred.float()
        coord_target_reg = coord_target.to(device=coord_pred.device, dtype=torch.float32)
    else:
        coord_pred_reg = coord_pred
        coord_target_reg = coord_target.to(device=coord_pred.device, dtype=coord_pred.dtype)

    reg_loss_fn = nn.SmoothL1Loss(beta=huber_delta)
    reg_loss = reg_loss_fn(coord_pred_reg, coord_target_reg)

    cls_logits = getattr(outputs, "biome_logits", None)
    if cls_logits is not None and cls_loss_weight != 0.0:
        cls_loss = nn.functional.cross_entropy(cls_logits, batch["biome_label"])
    else:
        cls_loss = coord_pred.new_zeros(())

    total = cls_loss_weight * cls_loss + reg_loss_weight * reg_loss
    return total, cls_loss, reg_loss


def _run_evaluation(
    *,
    model: nn.Module,
    eval_loader: Any,
    accelerator: Accelerator,
    coord_stats: CoordStats,
    config: MtlFinetuneConfig,
    cls_loss_weight: float,
    reg_loss_weight: float,
    huber_delta: float,
    global_step: int,
    best_eval_haversine_km: float | None,
    best_eval_macro_f1: float | None,
    output_dir: Path,
) -> tuple[
    float,
    float | None,
    float | None,
]:
    """Run evaluation over ``eval_loader`` and update best checkpoint if needed."""

    model.eval()
    cls_list: list[Tensor] = []
    coord_pred_list: list[Tensor] = []
    biome_list: list[Tensor] = []
    coord_tgt_list: list[Tensor] = []
    eval_total_loss = 0.0
    eval_steps = 0

    with torch.no_grad():
        for eval_batch in eval_loader:
            eval_batch = {
                k: v.to(accelerator.device, non_blocking=True) for k, v in eval_batch.items()
            }
            outputs = model(
                input_ids=eval_batch["input_ids"],
                attention_mask=eval_batch.get("attention_mask"),
            )
            eval_loss, _, _ = _compute_mtl_loss(
                outputs,
                eval_batch,
                cls_loss_weight=cls_loss_weight,
                reg_loss_weight=reg_loss_weight,
                huber_delta=huber_delta,
            )
            eval_loss_detached = eval_loss.detach().float()
            if torch.isfinite(eval_loss_detached).all():
                eval_total_loss += float(eval_loss_detached.mean().item())
                eval_steps += 1

            if getattr(outputs, "biome_logits", None) is not None:
                cls_list.append(outputs.biome_logits.detach())
            coord_pred_list.append(outputs.coordinate.detach())
            biome_list.append(eval_batch["biome_label"].detach())
            coord_tgt_list.append(eval_batch["coord_target"].detach())

        # If no finite-loss eval steps were observed, skip metric computation to
        # avoid bogus aggregates from all-NaN batches.
        if not coord_pred_list or eval_steps == 0:
            mean_eval_loss = (
                eval_total_loss / max(eval_steps, 1) if eval_steps > 0 else float("nan")
            )
            model.train()
            return mean_eval_loss, best_eval_haversine_km, best_eval_macro_f1

        all_coord_pred = accelerator.gather_for_metrics(torch.cat(coord_pred_list))
        all_biome = accelerator.gather_for_metrics(torch.cat(biome_list))
        all_coord_tgt = accelerator.gather_for_metrics(torch.cat(coord_tgt_list))
        if cls_list:
            all_cls = accelerator.gather_for_metrics(torch.cat(cls_list))
        else:
            # Coordinate-only configuration: supply an empty logits tensor with
            # zero biome dimension so ``compute_eval_metrics`` computes
            # coordinate metrics while returning NaN classification metrics.
            all_cls = torch.empty(
                all_coord_pred.shape[0],
                0,
                dtype=all_coord_pred.dtype,
                device=all_coord_pred.device,
            )

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

    if is_better and accelerator.is_main_process:
        best_eval_haversine_km = float(current_hav)
        best_eval_macro_f1 = float(current_f1)
        best_dir = output_dir / "best"
        unwrapped = accelerator.unwrap_model(model)
        with atomic_dir_replace(best_dir) as tmp_best:
            # Save backbone in HF format
            hf_dir = tmp_best / "hf_model"
            unwrapped.backbone.save_pretrained(str(hf_dir), safe_serialization=True)

            # Save heads
            heads_path = tmp_best / "heads.pt"
            torch.save(
                {
                    "coordinate_head": unwrapped.coordinate_head.state_dict(),
                    "biome_head": (
                        None if unwrapped.biome_head is None else unwrapped.biome_head.state_dict()
                    ),
                },
                heads_path,
            )

            # Save coordinate normalisation stats and best metrics sidecars.
            coord_norm_path = tmp_best / "coord_norm.json"
            _save_json_atomically(
                coord_norm_path,
                {
                    "lat_mean": float(coord_stats.lat_mean),
                    "lat_std": float(coord_stats.lat_std),
                    "lon_mean": float(coord_stats.lon_mean),
                    "lon_std": float(coord_stats.lon_std),
                },
            )
            metrics_path = tmp_best / "metrics.json"
            _save_json_atomically(
                metrics_path,
                {
                    "haversine_km_median": current_hav,
                    "macro_f1": current_f1,
                    "fold_index": int(config.fold_index),
                    "step": int(global_step),
                },
            )

    model.train()
    return mean_eval_loss, best_eval_haversine_km, best_eval_macro_f1


def run_jaguar_mtl_training(config_path: str | Path) -> MTLTrainResult:
    """Run two-phase jaguar MTL fine-tuning for a single cross-validation fold.

    The implementation is intentionally conservative with respect to checkpoint
    resume: it writes best-model snapshots and a rolling ``latest/train_state``
    JSON sidecar but will raise :class:`RuntimeError` if a previous run left a
    ``latest/train_state.json`` file in place.  This prevents silent clobbering
    of existing artefacts and keeps the future resume protocol explicit.
    """

    config = load_mtl_finetune_config(config_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guard against accidentally overwriting an existing run; proper resume
    # semantics (including Accelerate state) are deferred to a follow-up task.
    train_state_path = output_dir / "latest" / "train_state.json"
    if train_state_path.exists():
        raise RuntimeError(
            "run_jaguar_mtl_training: resume from existing checkpoint is not "
            "implemented yet; remove the 'latest' directory or implement the "
            "full resume protocol before re-running."
        )

    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    backbone = AutoModel.from_pretrained(
        str(config.backbone_path),
        trust_remote_code=True,
    )
    tokenizer = _load_tokenizer(config, backbone=backbone)

    train_loader, eval_loader, coord_stats = build_fold_dataloaders(config, tokenizer)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(output_dir / config.tensorboard_subdir),
    )

    model = JaguarMTLModel(
        backbone,
        num_biomes=config.n_biomes,
        dropout_prob=config.dropout,
    )

    accelerator.init_trackers("jaguar_mtl_training")

    # ---------------------------- Phase 1 ---------------------------------
    _freeze_backbone(model)
    phase1_optimizer, phase1_scheduler = _build_phase1_optimizer_and_scheduler(
        model,
        lr_heads=config.lr_heads_phase1,
        weight_decay=config.weight_decay,
        warmup_fraction=config.warmup_fraction,
        total_steps=config.phase1_steps,
    )

    (
        model,
        phase1_optimizer,
        phase1_scheduler,
        train_loader,
        eval_loader,
    ) = accelerator.prepare(
        model,
        phase1_optimizer,
        phase1_scheduler,
        train_loader,
        eval_loader,
    )

    global_step = 0
    phase1_steps_completed = 0
    phase2_steps_completed = 0
    best_eval_haversine_km: float | None = None
    best_eval_macro_f1: float | None = None

    nan_steps = 0
    skipped_steps = 0

    train_loss_sum = 0.0
    train_cls_loss_sum = 0.0
    train_reg_loss_sum = 0.0
    train_loss_count = 0

    model.train()

    while phase1_steps_completed < config.phase1_steps:
        for batch in train_loader:
            if phase1_steps_completed >= config.phase1_steps:
                break

            batch = {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                total_loss, cls_loss, reg_loss = _compute_mtl_loss(
                    outputs,
                    batch,
                    cls_loss_weight=config.cls_loss_weight,
                    reg_loss_weight=config.reg_loss_weight,
                    huber_delta=config.huber_delta,
                )
                loss_detached = total_loss.detach().float()

                if not torch.isfinite(loss_detached).all():
                    nan_steps += 1

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                    skip_step = (not torch.isfinite(loss_detached).all()) or (
                        not torch.isfinite(grad_norm.detach()).all()
                    )
                    if skip_step:
                        skipped_steps += 1
                    else:
                        phase1_optimizer.step()
                        phase1_scheduler.step()
                        phase1_steps_completed += 1
                        global_step += 1

                        train_loss_sum += float(loss_detached.item())
                        train_cls_loss_sum += float(cls_loss.detach().item())
                        train_reg_loss_sum += float(reg_loss.detach().item())
                        train_loss_count += 1

                        if global_step % config.log_every == 0:
                            denom = max(train_loss_count, 1)
                            logs = {
                                "train/total_loss": train_loss_sum / denom,
                                "train/cls_loss": train_cls_loss_sum / denom,
                                "train/reg_loss": train_reg_loss_sum / denom,
                                "train/nan_steps": float(nan_steps),
                                "train/skipped_steps": float(skipped_steps),
                                "train/lr_backbone": 0.0,
                                "train/lr_heads": phase1_scheduler.get_last_lr()[0],
                                "train/phase": 1.0,
                            }
                            accelerator.log(logs, step=global_step)
                            train_loss_sum = train_cls_loss_sum = train_reg_loss_sum = 0.0
                            train_loss_count = 0
                            nan_steps = 0
                            skipped_steps = 0

                        if eval_loader is not None and global_step % config.eval_every == 0:
                            (
                                mean_eval_loss,
                                best_eval_haversine_km,
                                best_eval_macro_f1,
                            ) = _run_evaluation(
                                model=model,
                                eval_loader=eval_loader,
                                accelerator=accelerator,
                                coord_stats=coord_stats,
                                config=config,
                                cls_loss_weight=config.cls_loss_weight,
                                reg_loss_weight=config.reg_loss_weight,
                                huber_delta=config.huber_delta,
                                global_step=global_step,
                                best_eval_haversine_km=best_eval_haversine_km,
                                best_eval_macro_f1=best_eval_macro_f1,
                                output_dir=output_dir,
                            )
                            logger.info(
                                "phase1_eval",
                                extra={"step": global_step, "loss": mean_eval_loss},
                            )

                        if global_step % config.save_every == 0 and accelerator.is_main_process:
                            _save_json_atomically(
                                train_state_path,
                                {
                                    "step": global_step,
                                    "phase": 1,
                                    "best_eval_haversine_km": best_eval_haversine_km,
                                    "best_eval_macro_f1": best_eval_macro_f1,
                                },
                            )

                phase1_optimizer.zero_grad()

    # ---------------------------- Phase 2 ---------------------------------
    inner_model = accelerator.unwrap_model(model)
    before_trainable = sum(1 for p in inner_model.parameters() if p.requires_grad)
    _unfreeze_last_n_layers(inner_model, config.unfreeze_layers)
    after_trainable = sum(1 for p in inner_model.parameters() if p.requires_grad)
    if after_trainable <= before_trainable:
        raise RuntimeError("unfreeze_last_n_layers did not increase trainable parameter count")

    phase2_optimizer, phase2_scheduler = _build_phase2_optimizer_and_scheduler(
        inner_model,
        lr_backbone=config.lr_backbone_phase2,
        lr_heads=config.lr_heads_phase2,
        weight_decay=config.weight_decay,
        warmup_fraction=config.warmup_fraction,
        total_steps=config.phase2_steps,
    )

    # SAFE PROTOCOL: model is already wrapped; only the new optimizer and
    # scheduler go through ``accelerator.prepare``.
    phase2_optimizer, phase2_scheduler = accelerator.prepare(phase2_optimizer, phase2_scheduler)

    model.train()
    train_loss_sum = train_cls_loss_sum = train_reg_loss_sum = 0.0
    train_loss_count = 0

    while phase2_steps_completed < config.phase2_steps:
        for batch in train_loader:
            if phase2_steps_completed >= config.phase2_steps:
                break

            batch = {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                total_loss, cls_loss, reg_loss = _compute_mtl_loss(
                    outputs,
                    batch,
                    cls_loss_weight=config.cls_loss_weight,
                    reg_loss_weight=config.reg_loss_weight,
                    huber_delta=config.huber_delta,
                )
                loss_detached = total_loss.detach().float()

                if not torch.isfinite(loss_detached).all():
                    nan_steps += 1

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                    skip_step = (not torch.isfinite(loss_detached).all()) or (
                        not torch.isfinite(grad_norm.detach()).all()
                    )
                    if skip_step:
                        skipped_steps += 1
                    else:
                        phase2_optimizer.step()
                        phase2_scheduler.step()
                        phase2_steps_completed += 1
                        global_step += 1

                        train_loss_sum += float(loss_detached.item())
                        train_cls_loss_sum += float(cls_loss.detach().item())
                        train_reg_loss_sum += float(reg_loss.detach().item())
                        train_loss_count += 1

                        if global_step % config.log_every == 0:
                            denom = max(train_loss_count, 1)
                            logs = {
                                "train/total_loss": train_loss_sum / denom,
                                "train/cls_loss": train_cls_loss_sum / denom,
                                "train/reg_loss": train_reg_loss_sum / denom,
                                "train/nan_steps": float(nan_steps),
                                "train/skipped_steps": float(skipped_steps),
                                "train/lr_backbone": phase2_scheduler.get_last_lr()[0],
                                "train/lr_heads": phase2_scheduler.get_last_lr()[-1],
                                "train/phase": 2.0,
                            }
                            accelerator.log(logs, step=global_step)
                            train_loss_sum = train_cls_loss_sum = train_reg_loss_sum = 0.0
                            train_loss_count = 0
                            nan_steps = 0
                            skipped_steps = 0

                        if eval_loader is not None and global_step % config.eval_every == 0:
                            (
                                mean_eval_loss,
                                best_eval_haversine_km,
                                best_eval_macro_f1,
                            ) = _run_evaluation(
                                model=model,
                                eval_loader=eval_loader,
                                accelerator=accelerator,
                                coord_stats=coord_stats,
                                config=config,
                                cls_loss_weight=config.cls_loss_weight,
                                reg_loss_weight=config.reg_loss_weight,
                                huber_delta=config.huber_delta,
                                global_step=global_step,
                                best_eval_haversine_km=best_eval_haversine_km,
                                best_eval_macro_f1=best_eval_macro_f1,
                                output_dir=output_dir,
                            )
                            logger.info(
                                "phase2_eval",
                                extra={"step": global_step, "loss": mean_eval_loss},
                            )

                        if global_step % config.save_every == 0 and accelerator.is_main_process:
                            _save_json_atomically(
                                train_state_path,
                                {
                                    "step": global_step,
                                    "phase": 2,
                                    "best_eval_haversine_km": best_eval_haversine_km,
                                    "best_eval_macro_f1": best_eval_macro_f1,
                                },
                            )

                phase2_optimizer.zero_grad()

    accelerator.end_training()

    return MTLTrainResult(
        fold_index=int(config.fold_index),
        phase1_steps_completed=phase1_steps_completed,
        phase2_steps_completed=phase2_steps_completed,
        best_eval_haversine_km=best_eval_haversine_km,
        best_eval_macro_f1=best_eval_macro_f1,
        output_dir=output_dir,
        coord_stats=coord_stats,
    )


def integration_test(
    *,
    n_individuals: int = 10,
    windows_per_individual: int = 5,
    use_real_model: bool = False,
) -> None:
    """Run a synthetic end-to-end check of the jaguar MTL training helpers.

    This integration harness mirrors the spirit of
    :func:`jaguar_geo_assign.pretrain.foundation_training.integration_test`
    but operates on the multi-task fine-tuning stack. It exercises the
    following contracts on a small synthetic workload:

    1. Forward pass through :class:`JaguarMTLModel` produces finite total,
       classification, and regression losses on a mixed biome cohort.
    2. A single optimiser step updates at least one model parameter
       (L2-difference > 0) under :class:`~accelerate.Accelerator` control.
    3. :func:`_run_evaluation` logs metrics and writes a complete "best"
       checkpoint directory (HF backbone, heads, normalisation stats,
       metrics).
    4. The evaluation path correctly handles coordinate-only configurations
       where the biome head is absent (no early return, finite coordinate
       metrics, checkpoint updates driven by geodesic error).
    5. All assertions run on a tiny BERT backbone by default; when
       ``use_real_model=True`` the real DNABERT-2 backbone from the
       Hugging Face Hub is exercised instead. The default CPU-only path
       avoids any network calls or large models.

    Args:
        n_individuals: Number of synthetic individuals **per biome**.
            The total synthetic cohort size is ``n_individuals *
            len(BIOME_CLASSES)``; larger values stress-test metric
            aggregation without altering the semantics of the assertions.
        windows_per_individual: Number of synthetic windows per
            individual. This parameter controls the batch size used in the
            synthetic training/eval batches.
        use_real_model: If ``True``, load the real DNABERT-2 backbone from
            the Hugging Face Hub ("zhihan1996/DNABERT-2-117M"). When
            ``False`` (the default), use a tiny randomly initialised BERT
            backbone that runs quickly on CPU.

    Raises:
        AssertionError: If any of the integration invariants fail.
        RuntimeError: If model loading or checkpoint I/O fails.
    """

    import json
    import tempfile

    from transformers import BertConfig

    if n_individuals <= 0:
        raise ValueError("n_individuals must be positive for integration_test")
    if windows_per_individual <= 0:
        raise ValueError("windows_per_individual must be positive for integration_test")

    # Synthetic cohort: all biomes represented with n_individuals per biome.
    n_biomes = len(BIOME_CLASSES)
    total_individuals = n_biomes * n_individuals
    batch_size = total_individuals * windows_per_individual
    seq_len = 16

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_path = Path(tmp_root)
        output_dir = tmp_path / "mtl_integration_out"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------- backbone + model ------------------------
        if use_real_model:
            backbone = AutoModel.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                trust_remote_code=True,
            )
        else:
            config = BertConfig(
                num_hidden_layers=2,
                num_attention_heads=2,
                hidden_size=32,
                intermediate_size=64,
                vocab_size=128,
            )
            backbone = AutoModel.from_config(config)

        model = JaguarMTLModel(
            backbone,
            num_biomes=n_biomes,
            dropout_prob=0.1,
        )
        coord_stats = CoordStats(
            lat_mean=0.0,
            lat_std=1.0,
            lon_mean=0.0,
            lon_std=1.0,
        )

        accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=1,
            log_with="tensorboard",
            project_dir=str(output_dir / "tensorboard"),
        )
        model = accelerator.prepare(model)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        optimizer = accelerator.prepare(optimizer)
        accelerator.init_trackers("jaguar_mtl_integration_test")

        vocab_size = int(getattr(backbone.config, "vocab_size", 128))
        device = accelerator.device
        input_ids = torch.randint(
            low=0,
            high=max(vocab_size, 2),
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        biome_labels = (torch.arange(batch_size, device=device) % n_biomes).to(torch.long)
        coord_target = torch.randn(
            batch_size,
            2,
            device=device,
            dtype=torch.float32,
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "biome_label": biome_labels,
            "coord_target": coord_target,
        }

        # ---------------- Assertion 1 & 2: forward + optimiser -------------
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        total_loss, cls_loss, reg_loss = _compute_mtl_loss(
            outputs,
            batch,
            cls_loss_weight=1.0,
            reg_loss_weight=0.5,
            huber_delta=1.0,
        )
        loss_detached = total_loss.detach().float()
        assert torch.isfinite(loss_detached).all(), (
            f"Non-finite total_loss in integration test: {loss_detached}"
        )
        assert torch.isfinite(cls_loss.detach()).all(), "Non-finite cls_loss in integration test"
        assert torch.isfinite(reg_loss.detach()).all(), "Non-finite reg_loss in integration test"

        unwrapped = accelerator.unwrap_model(model)
        pre_params = {name: param.detach().clone() for name, param in unwrapped.named_parameters()}

        with accelerator.accumulate(model):
            accelerator.backward(total_loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            assert torch.isfinite(grad_norm.detach()).all(), (
                "Non-finite grad_norm in integration test"
            )
            optimizer.step()
            optimizer.zero_grad()

        unwrapped_after = accelerator.unwrap_model(model)
        l2_diffs = [
            (unwrapped_after.state_dict()[name] - before).norm().item()
            for name, before in pre_params.items()
        ]
        assert any(diff > 0 for diff in l2_diffs), "Optimiser step did not change any parameters"

        # ---------------- Assertion 3: evaluation + checkpoints ------------
        eval_loader = [batch]
        config = MtlFinetuneConfig(
            backbone_path=output_dir / "<unused_backbone>",
            windows_jsonl=output_dir / "<unused_windows>",
            metadata_csv=output_dir / "<unused_metadata>",
            output_dir=output_dir,
        )
        best_eval_haversine_km: float | None = None
        best_eval_macro_f1: float | None = None
        mean_eval_loss, best_eval_haversine_km, best_eval_macro_f1 = _run_evaluation(
            model=model,
            eval_loader=eval_loader,
            accelerator=accelerator,
            coord_stats=coord_stats,
            config=config,
            cls_loss_weight=1.0,
            reg_loss_weight=0.5,
            huber_delta=1.0,
            global_step=1,
            best_eval_haversine_km=best_eval_haversine_km,
            best_eval_macro_f1=best_eval_macro_f1,
            output_dir=output_dir,
        )
        assert math.isfinite(mean_eval_loss), "Mean eval loss should be finite in integration test"
        assert best_eval_haversine_km is not None, (
            "Best eval haversine should be set after evaluation"
        )
        assert best_eval_macro_f1 is not None, "Best eval macro_f1 should be set after evaluation"

        best_dir = output_dir / "best"
        hf_dir = best_dir / "hf_model"
        assert (hf_dir / "config.json").exists(), (
            "HF config.json was not written in best checkpoint"
        )
        assert any(p.suffix == ".safetensors" for p in hf_dir.glob("*")), (
            "No safetensors file in hf_model dir"
        )
        assert (best_dir / "heads.pt").exists(), "Heads checkpoint not written"
        assert (best_dir / "coord_norm.json").exists(), "coord_norm.json sidecar not written"
        metrics_path = best_dir / "metrics.json"
        assert metrics_path.exists(), "metrics.json sidecar not written"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert "haversine_km_median" in metrics, "metrics.json missing haversine_km_median"
        assert "macro_f1" in metrics, "metrics.json missing macro_f1"

        # ---------------- Assertion 4: coordinate-only eval path -----------
        coord_only_model = JaguarMTLModel(
            unwrapped_after.backbone,
            num_biomes=None,
            dropout_prob=0.1,
        )
        coord_only_model = accelerator.prepare(coord_only_model)
        outputs_coord_only = coord_only_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        total_loss2, cls_loss2, reg_loss2 = _compute_mtl_loss(
            outputs_coord_only,
            batch,
            cls_loss_weight=1.0,
            reg_loss_weight=0.5,
            huber_delta=1.0,
        )
        assert torch.isfinite(total_loss2.detach()).all(), (
            "Non-finite total_loss in coord-only integration path"
        )
        assert cls_loss2.detach().item() == 0.0, "Coordinate-only head should yield zero cls_loss"
        assert torch.isfinite(reg_loss2.detach()).all(), (
            "Non-finite reg_loss in coord-only integration path"
        )

        mean_eval_loss2, best_eval_haversine_km2, best_eval_macro_f12 = _run_evaluation(
            model=coord_only_model,
            eval_loader=eval_loader,
            accelerator=accelerator,
            coord_stats=coord_stats,
            config=config,
            cls_loss_weight=1.0,
            reg_loss_weight=0.5,
            huber_delta=1.0,
            global_step=2,
            best_eval_haversine_km=best_eval_haversine_km,
            best_eval_macro_f1=best_eval_macro_f1,
            output_dir=output_dir,
        )
        assert math.isfinite(mean_eval_loss2), "Coordinate-only eval produced non-finite mean loss"
        assert best_eval_haversine_km2 is not None, (
            "Coordinate-only eval did not update best haversine"
        )
        # macro_f1 is expected to be NaN in coordinate-only mode; we
        # intentionally do not assert on its value here.


__all__ = [
    "BIOME_CLASSES",
    "MTLTrainResult",
    "compute_eval_metrics",
    "haversine_distance_km",
    "integration_test",
    "run_jaguar_mtl_training",
]
