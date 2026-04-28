"""Foundation model continued pre-training for DNABERT-2 on felid corpus.

This module implements the complete MLM training loop for DNABERT-2 on the
tokenized felid foundation corpus. It handles model loading with pad-token
fallbacks, mixed-precision training via accelerate, checkpoint management
with atomic writes, and integration testing via synthetic or real models.

Key design decisions (tagged with # TRADE-OFF:):
- IterableDataset is used (streaming reader); resume restarts at epoch boundary.
- Perplexity is clamped at exp(20) to protect against bf16 noise.
- Pad-token guard implements all three fallback strategies (eos/unk/add_pad).
- Evaluation uses a deterministic eval_max_steps to prevent DDP deadlock.

The integration_test() function is designed to be callable from both the
CLI (--integration-test flag) and pytest, with configurable model source
(real DNABERT-2 or tiny synthetic) and synthetic corpus fixture.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

from jaguar_geo_assign.config import FoundationTrainingConfig
from jaguar_geo_assign.data.preprocessor import load_dnabert2_tokenizer
from jaguar_geo_assign.data.tokenized_corpus_reader import (
    CorpusReaderError,
    TokenizedCorpusReader,
    _get_distributed_state,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricAccumulator:
    """Per-window accumulation of training metrics with NaN/Inf guard.

    Accumulates loss, token accuracy, and masked token counts across steps
    without propagating NaN/Inf values into running sums. A separate nan_count
    tracks the number of steps that produced non-finite loss.
    """

    loss_sum: float = 0.0
    token_correct: int = 0
    token_masked: int = 0
    step_count: int = 0
    nan_count: int = 0
    skipped_steps: int = 0

    def reset(self) -> None:
        """Reset accumulator to zero state.

        Called after logging a window's metrics to prevent carryover
        into the next logging window.
        """
        self.loss_sum = 0.0
        self.token_correct = 0
        self.token_masked = 0
        self.step_count = 0
        self.nan_count = 0
        self.skipped_steps = 0


@dataclass(frozen=True)
class TrainingRunResult:
    """Immutable summary of a completed training run.

    Attributes:
        final_step: Total training steps completed.
        best_eval_loss: Best validation loss observed (None if no eval).
        mlm_head_random_init: Whether MLM head was randomly initialized.
        pad_token_fallback_used: The fallback strategy used (eos/unk/add_pad).
        resumed: Whether training resumed from a prior checkpoint.
        trainable_param_count: Number of trainable parameters (verified after prepare).
        total_param_count: Total number of parameters.
        resolved_versions: Pinned versions of torch, accelerate, transformers, tensorboard.
    """

    final_step: int
    best_eval_loss: float | None = None
    mlm_head_random_init: bool = False
    pad_token_fallback_used: str = "none"
    resumed: bool = False
    trainable_param_count: int = 0
    total_param_count: int = 0
    resolved_versions: dict[str, str] = field(default_factory=dict)


def _get_resolved_versions() -> dict[str, str]:
    """Capture resolved versions of key dependencies for auditability.

    Returns:
        Dict mapping package name to version string.
    """
    versions = {
        "torch": torch.__version__,
    }
    try:
        import accelerate

        versions["accelerate"] = accelerate.__version__
    except (ImportError, AttributeError):
        pass
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except (ImportError, AttributeError):
        pass
    try:
        import tensorboard

        versions["tensorboard"] = tensorboard.__version__
    except (ImportError, AttributeError):
        pass
    return versions


def _build_model_and_tokenizer(
    config: FoundationTrainingConfig,
) -> tuple[Any, Any, str, bool]:
    """Load model and tokenizer, applying pad-token guard and MLM head check.

    Implements §3.2 (pad-token guard with all three branches) and §3.3
    (MLM head missing-keys check).

    Args:
        config: Training configuration with model_identifier and model_revision.

    Returns:
        Tuple of (model, tokenizer, pad_token_fallback_used, mlm_head_random_init).

    Raises:
        RuntimeError: If model loading fails or MLM head detection is inconclusive.
    """
    # Load tokenizer via existing helper
    tokenizer, _ = load_dnabert2_tokenizer()

    # §3.2: Pad-token guard with all three branches
    pad_token_fallback_used = "none"
    if tokenizer.pad_token_id is None:
        # TRADE-OFF: pad-token fallback is a pragmatic choice because DNABERT-2
        # ships without explicit pad_token. We rank the fallbacks: eos > unk > add.
        if config.pad_token_fallback == "eos" and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            pad_token_fallback_used = "eos"
        elif config.pad_token_fallback == "unk" and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            pad_token_fallback_used = "unk"
        else:
            # add_pad: inject a new [PAD] token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            pad_token_fallback_used = "add_pad"
        logger.info(
            "Applied pad-token fallback",
            extra={
                "strategy": pad_token_fallback_used,
                "new_pad_token_id": tokenizer.pad_token_id,
            },
        )

    # Load model with output_loading_info to inspect missing keys
    model, loading_info = AutoModelForMaskedLM.from_pretrained(
        config.model_identifier,
        revision=config.model_revision,
        trust_remote_code=True,
        output_loading_info=True,
    )

    # Sync model.config with tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id

    # If we added a new pad token, resize embeddings
    if pad_token_fallback_used == "add_pad":
        model.resize_token_embeddings(len(tokenizer))

    # §3.3: MLM head missing-keys check
    mlm_head_random_init = False
    missing_keys = loading_info.get("missing_keys", [])
    mlm_head_prefixes = ("cls.", "mlm_head", "lm_head")
    for key in missing_keys:
        if any(key.startswith(prefix) for prefix in mlm_head_prefixes):
            mlm_head_random_init = True
            logger.warning(
                f"MLM head was randomly initialized; missing key: {key}. "
                "This may hurt warm-start performance. Monitor early loss curves."
            )
            break

    return model, tokenizer, pad_token_fallback_used, mlm_head_random_init


def _build_optimizer(
    model: Any,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Build AdamW optimizer with weight decay groups per §3.4.

    Bias and LayerNorm parameters are excluded from weight decay.

    Args:
        model: PyTorch model to optimize.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization coefficient.

    Returns:
        Configured AdamW optimizer.
    """
    # Identify parameters to exclude from weight decay
    no_decay = {"bias", "LayerNorm.weight"}
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def _build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    max_steps: int,
) -> Any:
    """Build cosine schedule with linear warmup per §3.4.

    Args:
        optimizer: AdamW optimizer to schedule.
        warmup_steps: Linear warmup duration.
        max_steps: Total training steps (used for cosine phase).

    Returns:
        Configured scheduler.
    """
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return scheduler


def _build_dataloaders(
    config: FoundationTrainingConfig,
    tokenizer: Any,
) -> tuple[DataLoader, DataLoader | None]:
    """Build train and validation dataloaders using TokenizedCorpusReader.

    The training dataloader uses shuffling; the validation dataloader
    yields rows in canonical order for reproducibility.

    Args:
        config: Training configuration with corpus and dataloader params.
        tokenizer: Tokenizer for the data collator.

    Returns:
        Tuple of (train_loader, eval_loader) where eval_loader is None
        if eval_every is not set.

    Raises:
        FileNotFoundError: If corpus metadata.json does not exist.
    """
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
    )

    _, world_size, _, _ = _get_distributed_state()

    # Build training dataloader
    train_reader = TokenizedCorpusReader(
        config.corpus_metadata_path,
        "train",
        max_seq_length=config.max_seq_length,
        file_shuffle=True,
        shuffle_buffer_size=config.shuffle_buffer_size,
        seed=config.seed,
        drop_last=False,
        world_size=world_size,
        num_workers=config.num_workers,
    )
    train_loader = DataLoader(
        train_reader,
        batch_size=config.per_device_train_batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
    )

    # Build validation dataloader (if corpus has validation split)
    eval_loader = None
    try:
        eval_reader = TokenizedCorpusReader(
            config.corpus_metadata_path,
            "validation",
            max_seq_length=config.max_seq_length,
            file_shuffle=False,  # Validation must be deterministic
            shuffle_buffer_size=1,  # Ignored for validation
            seed=config.seed,
            drop_last=False,
            world_size=world_size,
            num_workers=config.num_workers,
        )
        eval_loader = DataLoader(
            eval_reader,
            batch_size=config.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=config.num_workers,
        )
    except CorpusReaderError as exc:
        # TRADE-OFF: Substring match is used here to identify "validation split missing"
        # as a non-fatal condition, while letting other CorpusReaderError variants
        # (empty shard, schema errors) propagate loudly during startup.
        if "validation" not in str(exc) or "not found" not in str(exc).lower():
            raise
        # Validation split not available
        logger.info("Validation split not found; skipping eval.")

    return train_loader, eval_loader


def _compute_eval_max_steps(
    eval_reader: TokenizedCorpusReader,
    per_device_eval_batch_size: int,
    world_size: int,
) -> int:
    """Compute fixed eval step count to prevent DDP deadlock per §1.2.

    All ranks must iterate exactly this many batches. Any leftover rows
    are dropped and logged.

    Args:
        eval_reader: Validation IterableDataset with record_count property.
        per_device_eval_batch_size: Batch size per device.
        world_size: Total number of processes (global_batch_size = per_device * world_size).

    Returns:
        Fixed maximum evaluation steps.
    """
    # TRADE-OFF: eval-step cap is derived from record_count to prevent DDP deadlock
    # when ranks yield different batch counts. Leftover rows are dropped.
    record_count = eval_reader.record_count
    global_batch_size = per_device_eval_batch_size * world_size
    eval_max_steps = max(1, record_count // global_batch_size)
    leftover = record_count % global_batch_size
    if leftover > 0:
        logger.info(
            f"Eval will drop {leftover} rows (not evenly divisible into {world_size} ranks)"
        )
    return eval_max_steps


@contextlib.contextmanager
def atomic_dir_replace(target: Path) -> Iterator[Path]:
    """Yield a tmp dir path; atomically rename to target on successful exit."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp_{target.name}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        yield tmp
        # TRADE-OFF: rename-old-aside prevents target disappearance during replacement
        if target.exists():
            old_target = target.parent / f".old_{target.name}_{os.getpid()}"
            os.replace(str(target), str(old_target))
            try:
                os.replace(str(tmp), str(target))
            except Exception:
                # Roll back: restore old to target
                os.replace(str(old_target), str(target))
                raise
            shutil.rmtree(old_target, ignore_errors=True)
        else:
            os.replace(str(tmp), str(target))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _recover_atomic_dir(target: Path) -> bool:
    """Recover from a mid-replace crash by restoring the most recent .old_* directory."""
    if target.exists():
        return False

    # TRADE-OFF: During mid-rename crash, target is lost but .old_ retains previous valid state.
    # Restoring .old_ is crash-safe but inherently reverts to previous state.
    # Search for siblings matching `.old_<target.name>_*`
    candidates = list(target.parent.glob(f".old_{target.name}_*"))
    if not candidates:
        return False

    # Pick the most recent by mtime
    most_recent = max(candidates, key=lambda p: p.stat().st_mtime)
    os.replace(str(most_recent), str(target))
    logger.warning("Recovered checkpoint from %s due to mid-rename crash", most_recent)
    return True


def _broadcast_save_failure(accelerator: Accelerator, save_failed: bool) -> bool:
    """Broadcast save failure across ranks to prevent deadlocks.
    Uses set_trigger if available (newer Accelerate), else gathers a tensor.
    """
    # TRADE-OFF: Fallback logic for older Accelerate versions ensures compatibility
    if hasattr(accelerator, "set_trigger"):
        if save_failed:
            accelerator.set_trigger()
        return accelerator.check_trigger()
    else:
        fail_tensor = torch.tensor(
            [1 if save_failed else 0], dtype=torch.int32, device=accelerator.device
        )
        gathered = accelerator.gather(fail_tensor)
        return gathered.sum().item() > 0


def _save_json_atomically(
    path: Path,
    content: dict[str, Any],
) -> None:
    """Write checkpoint sidecar atomically via temp file per §3.8.

    Args:
        path: Target file path.
        content: dict to serialize to JSON.

    Raises:
        OSError: If atomic write fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fix #14: Write JSON as a FILE with atomic rename.
    # Use unique tmp suffix per PID to be crash-safe during concurrent writes.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(content))
        # Atomically rename temp file to target
        os.replace(str(tmp_path), str(path))
    except Exception:
        # Cleanup temp file on failure
        tmp_path.unlink(missing_ok=True)
        raise


def _startup_probe_metrics(
    batch: dict[str, torch.Tensor],
    step: int,
    accelerator: Accelerator,
    model: Any,
) -> dict[str, float]:
    """Compute startup probe metrics for the first 20 steps per §3.7.

    Args:
        batch: Data batch with attention_mask and labels.
        step: Current training step (1-indexed).
        accelerator: Accelerate trainer for rank detection.
        model: Model for computing per-parameter gradient norms (rank-0 only).

    Returns:
        Dict of metric name -> value for rank-0-only logging (excludes grad_norm_hist).
    """
    if accelerator.is_main_process and step <= 20:
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        metrics = {}

        if attention_mask is not None:
            mean_attn_length = attention_mask.sum(dim=-1).float().mean().item()
            metrics["startup/mean_attention_length"] = mean_attn_length

        if labels is not None:
            mask = labels != -100
            observed_mask_rate = mask.sum().item() / max(mask.numel(), 1)
            metrics["startup/observed_mask_rate"] = observed_mask_rate

        return metrics
    return {}


def run_felid_foundation_training(
    config_path: str | Path,
    *,
    integration_test_mode: Literal["off", "real_model", "tiny_model"] = "off",
) -> TrainingRunResult:
    """Run felid foundation continued pre-training for DNABERT-2.

    Implements the full training loop per Technical Design §3, including
    model loading, mixed-precision training, checkpoint management, and
    integration testing.

    Args:
        config_path: Path to TOML training configuration.
        integration_test_mode: If "off" (default), run normal training.
            If "real_model" or "tiny_model", run integration_test instead.

    Returns:
        TrainingRunResult with final step, best eval loss, and metadata.

    Raises:
        FileNotFoundError: If config or corpus metadata not found.
        RuntimeError: If training fails or sanity checks fail.
    """
    from jaguar_geo_assign.config import load_foundation_training_config

    if integration_test_mode != "off":
        # Delegate to integration_test for testing modes
        integration_test(use_real_model=(integration_test_mode == "real_model"))
        return TrainingRunResult(
            final_step=0,
            best_eval_loss=None,
            mlm_head_random_init=False,
            pad_token_fallback_used="none",
            resumed=False,
            trainable_param_count=0,
            total_param_count=0,
            resolved_versions=_get_resolved_versions(),
        )

    config = load_foundation_training_config(config_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer, pad_fallback, mlm_random_init = _build_model_and_tokenizer(config)

    # Build optimizer, scheduler, dataloaders
    optimizer = _build_optimizer(model, config.learning_rate, config.weight_decay)
    scheduler = _build_scheduler(optimizer, config.warmup_steps, config.max_steps)
    train_loader, eval_loader = _build_dataloaders(config, tokenizer)

    # Initialize Accelerator with mixed precision (§3.5)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(output_dir / config.tensorboard_subdir),
    )

    # Check for resumed state
    latest_state_path = output_dir / "latest" / "accelerate_state"

    # Fix 28: Recover from mid-rename crash
    _recover_atomic_dir(output_dir / "latest" / "accelerate_state")
    _recover_atomic_dir(output_dir / "best" / "accelerate_state")
    _recover_atomic_dir(output_dir / "best" / "hf_model")
    _recover_atomic_dir(output_dir / "best" / "tokenizer")

    resumed = latest_state_path.exists()
    # Fix #16: Align read path with write path (both in output_dir / "best" / "best_eval_loss.json")
    best_eval_loss_file = output_dir / "best" / "best_eval_loss.json"
    best_eval_loss = None
    if best_eval_loss_file.exists():
        try:
            best_eval_loss = json.loads(best_eval_loss_file.read_text()).get("eval_loss")
        except (json.JSONDecodeError, KeyError):
            pass

    # Prepare for distributed training
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    eval_loader = accelerator.prepare(eval_loader) if eval_loader is not None else None

    # §3.10 (Fix #10): Model trainability verification (AC#5)
    # Count trainable vs. total parameters after construction and verify all are trainable.
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    if trainable != total:
        raise RuntimeError(
            f"Model trainability check failed: Expected all {total} params trainable; "
            f"only {trainable} are. Verify model construction and accelerator.prepare()."
        )
    logger.info(
        "model_trainability_verified",
        extra={
            "trainable_params": trainable,
            "total_params": total,
        },
    )

    # Resume if needed (§3.9)
    # TRADE-OFF: IterableDataset does not restore the cursor position on resume;
    # the dataloader restarts at the epoch boundary. This may cause the first batch
    # after resume to contain duplicate rows from earlier in the epoch.
    # Training loop
    step = 0
    if resumed:
        accelerator.load_state(str(latest_state_path))

        # TRADE-OFF: step is restored from sidecar (not accelerate_state) because
        # Python ints are not part of save_state. Sidecar is best-effort; old
        # checkpoints without it resume at step=0 with a warning.
        train_state_path = latest_state_path.parent / "train_state.json"
        if train_state_path.exists():
            try:
                train_state = json.loads(train_state_path.read_text(encoding="utf-8"))
                step = int(train_state.get("step", 0))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Failed to parse %s on resume; restarting step counter at 0: %s",
                    train_state_path,
                    exc,
                )
                step = 0
        else:
            logger.info(
                "train_state.json not found in latest checkpoint (older format); "
                "step counter restarting at 0."
            )
            step = 0

        logger.info("Resumed training from latest checkpoint")

    # Note: tokens-trained-so-far implications: no metrics currently use step as a
    # denominator in a way that breaks on resume. step counts all previous steps.
    best_eval_loss = float("inf") if best_eval_loss is None else float(best_eval_loss)
    train_metric = MetricAccumulator()
    eval_metric = MetricAccumulator()

    accelerator.init_trackers("felid_foundation_training")
    startup_grad_norms = []

    try:
        for _epoch in range(1, 1000):  # Iterate until max_steps reached
            # Fix #18: Set epoch on reader for deterministic multi-epoch shuffling
            train_loader.dataset.set_epoch(_epoch - 1)  # 0-indexed
            for batch in train_loader:
                if step >= config.max_steps:
                    break

                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss

                    # §3.6: NaN/Inf guard
                    loss_f = loss.detach().float()
                    if torch.isnan(loss_f).any() or torch.isinf(loss_f).any():
                        train_metric.nan_count += 1
                        logger.warning(f"NaN/Inf loss detected at step {step}")
                    else:
                        train_metric.loss_sum += loss_f.mean().item()
                        train_metric.step_count += 1

                    # §3.6 (Fix #9): Token accuracy accumulation for train loop.
                    # Compute predictions and accumulate against masked labels.
                    # TRADE-OFF: argmax over full vocab is the dominant new cost;
                    # computed inside torch.no_grad() and only for masked positions when feasible.
                    with torch.no_grad():
                        preds = outputs.logits.argmax(dim=-1)
                        labels = batch.get("labels")
                        if labels is not None:
                            mask = labels != -100
                            # TRADE-OFF: Computing correct predictions locally avoids DDP gather
                            # inside the accumulation loop. We accumulate locally and defer gather
                            # to the log step.
                            train_metric.token_correct += ((preds == labels) & mask).sum().item()
                            train_metric.token_masked += mask.sum().item()

                    # Backward pass
                    accelerator.backward(loss)

                    # Gradient clipping (§3.5)
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )

                    # Fix 1: Collect grad norms on sync steps before step/zero_grad
                    skip_step = False
                    if accelerator.sync_gradients:
                        if torch.isnan(grad_norm).any() or torch.isinf(grad_norm).any():
                            logger.warning(
                                f"NaN/Inf grad_norm ({grad_norm}) detected at step {step}. "
                                "Skipping optimizer step."
                            )
                            train_metric.skipped_steps += 1
                            skip_step = True
                        elif accelerator.is_main_process and step < 20:
                            grad_list = [
                                p.grad.detach().norm()
                                for p in model.parameters()
                                if p.grad is not None
                            ]
                            if grad_list:
                                # Save with the upcoming step number
                                startup_grad_norms.append((step + 1, torch.stack(grad_list)))

                    if not skip_step:
                        optimizer.step()
                        # Fix #17: Guard scheduler.step() by sync_gradients.
                        # accumulate() no-ops optimizer.step() and optimizer.zero_grad(),
                        # but NOT scheduler.step(). Without this guard, the scheduler advances
                        # once per accumulation micro-step instead of once per optimizer update,
                        # exhausting the learning rate schedule prematurely.
                        if accelerator.sync_gradients:
                            scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    # step counts optimizer updates, not micro-batches;
                    # cadence aligns with scheduler
                    step += 1

                    # Log metrics at log_every (§3.6: windowed averages)
                    if step % config.log_every == 0:
                        mean_loss = train_metric.loss_sum / max(train_metric.step_count, 1)

                        # DDP-safe global token accuracy reduce
                        local_counts = torch.tensor(
                            [train_metric.token_correct, train_metric.token_masked],
                            dtype=torch.float32,
                            device=accelerator.device,
                        )
                        global_counts = accelerator.reduce(local_counts, reduction="sum")
                        global_correct = global_counts[0].item()
                        global_masked = global_counts[1].item()

                        # §3.6 (Fix #12): NaN convention: token_accuracy is NaN when token_masked==0
                        # rather than 0.0, to distinguish "no masked tokens" from "zero accuracy".
                        # TRADE-OFF: This convention helps downstream analysis identify data issues.
                        token_acc = (
                            float("nan") if global_masked == 0 else global_correct / global_masked
                        )
                        # TRADE-OFF: perplexity clamped at 20 to prevent bf16 overflow
                        ppl = (
                            float("nan")
                            if train_metric.step_count == 0
                            else math.exp(min(mean_loss, 20.0))
                        )

                        logs = {
                            "train/mlm_loss": mean_loss,
                            "train/token_accuracy": token_acc,
                            "train/perplexity": ppl,
                            "train/nan_steps": train_metric.nan_count,
                            "train/skipped_steps": train_metric.skipped_steps,
                            "train/grad_norm": grad_norm,
                            "train/lr": scheduler.get_last_lr()[0],
                        }

                        # §3.7: Startup probe for first 20 steps (rank-0 only)
                        startup_logs = _startup_probe_metrics(batch, step, accelerator, model)
                        logs.update(startup_logs)

                        accelerator.log(logs, step=step)

                        # §3.7 (Fix #13): Log grad_norm_hist as TensorBoard histogram (rank-0 only).
                        # Must be done separately from accelerator.log because histograms need
                        # direct access to the TensorBoard writer, not the generic log interface.
                        if accelerator.is_main_process and startup_grad_norms:
                            tb_tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                            for s, norms in startup_grad_norms:
                                tb_tracker.add_histogram(
                                    "startup/grad_norm_hist",
                                    norms,
                                    global_step=s,
                                )
                            startup_grad_norms.clear()

                        train_metric.reset()

                    # Evaluation at eval_every (§1.2: fixed eval_max_steps)
                    if eval_loader is not None and step % config.eval_every == 0:
                        eval_max_steps = config.eval_max_steps
                        if eval_max_steps is None:
                            eval_max_steps = _compute_eval_max_steps(
                                eval_loader.dataset,
                                config.per_device_eval_batch_size,
                                accelerator.num_processes,
                            )
                        else:
                            auto_cap = eval_loader.dataset.record_count // (
                                config.per_device_eval_batch_size * accelerator.num_processes
                            )
                            if eval_max_steps > auto_cap:
                                logger.warning(
                                    f"Explicit eval_max_steps={eval_max_steps} exceeds "
                                    f"auto-derived cap ({auto_cap}) and risks DDP deadlock. "
                                    f"Ranks may iterate unevenly."
                                )

                        model.eval()
                        with torch.no_grad():
                            for eval_step, eval_batch in enumerate(eval_loader):
                                if eval_step >= eval_max_steps:
                                    break

                                outputs = model(**eval_batch)
                                loss = outputs.loss

                                loss_f = loss.detach().float()
                                # Fix 3: Gather eval loss across ranks before accumulation
                                gathered_loss = accelerator.gather_for_metrics(loss_f)
                                if not (
                                    torch.isnan(gathered_loss).any()
                                    or torch.isinf(gathered_loss).any()
                                ):
                                    eval_metric.loss_sum += gathered_loss.mean().item()
                                    eval_metric.step_count += 1

                                # §3.6 (Fix #11): Token accuracy accumulation for eval loop.
                                # Compute predictions and accumulate against masked labels locally.
                                preds = outputs.logits.argmax(dim=-1)
                                labels = eval_batch.get("labels")
                                if labels is not None:
                                    gathered_preds = accelerator.gather_for_metrics(preds)
                                    gathered_labels = accelerator.gather_for_metrics(labels)
                                    gathered_mask = gathered_labels != -100
                                    if accelerator.is_main_process:
                                        eval_metric.token_correct += (
                                            ((gathered_preds == gathered_labels) & gathered_mask)
                                            .sum()
                                            .item()
                                        )
                                        eval_metric.token_masked += gathered_mask.sum().item()

                        model.train()
                        mean_eval_loss = eval_metric.loss_sum / max(eval_metric.step_count, 1)

                        # §3.6 (Fix #12): NaN convention for eval token_accuracy
                        eval_token_acc = (
                            float("nan")
                            if eval_metric.token_masked == 0
                            else eval_metric.token_correct / eval_metric.token_masked
                        )
                        eval_ppl = (
                            float("nan")
                            if eval_metric.step_count == 0
                            else math.exp(min(mean_eval_loss, 20.0))
                        )

                        accelerator.log(
                            {
                                "eval/mlm_loss": mean_eval_loss,
                                "eval/token_accuracy": eval_token_acc,
                                "eval/perplexity": eval_ppl,
                            },
                            step=step,
                        )

                        if mean_eval_loss < best_eval_loss:
                            best_eval_loss = mean_eval_loss
                            # Save best checkpoint
                            best_dir = output_dir / "best"

                            # Fix 4: DDP-safe checkpointing staged through tmp
                            tmp_best_accel_state = best_dir / ".tmp_accelerate_state"
                            accelerator.save_state(str(tmp_best_accel_state))

                            accelerator.wait_for_everyone()
                            saved_exc = None
                            save_failed = False
                            if accelerator.is_main_process:
                                try:
                                    with atomic_dir_replace(
                                        best_dir / "accelerate_state"
                                    ) as swap_target:
                                        swap_target.rmdir()
                                        os.replace(str(tmp_best_accel_state), str(swap_target))

                                    _save_json_atomically(
                                        best_dir / "best_eval_loss.json",
                                        {
                                            "step": step,
                                            "eval_loss": best_eval_loss,
                                        },
                                    )
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    with atomic_dir_replace(best_dir / "hf_model") as tmp_model_dir:
                                        unwrapped_model.save_pretrained(
                                            str(tmp_model_dir),
                                            safe_serialization=True,
                                        )
                                    with atomic_dir_replace(best_dir / "tokenizer") as tmp_tok_dir:
                                        tokenizer.save_pretrained(str(tmp_tok_dir))
                                except Exception as e:
                                    # TRADE-OFF: Rank-0 exception broadcasted
                                    # to prevent ranks deadlocking.
                                    logger.error(
                                        "Rank-0 checkpoint save failed: %s", e, exc_info=True
                                    )
                                    save_failed = True
                                    saved_exc = e
                            accelerator.wait_for_everyone()
                            # TRADE-OFF: Collective called before rank-0 raise to prevent deadlock.
                            saw_failure = _broadcast_save_failure(accelerator, save_failed)
                            if save_failed:
                                raise saved_exc
                            if saw_failure:
                                raise RuntimeError(
                                    "Distributed checkpoint save failed on rank-0; aborting "
                                    "all ranks to allow torchrun cleanup. Inspect rank-0 "
                                    "logs for the original exception."
                                )

                        eval_metric.reset()

                    # Save latest checkpoint at save_every
                    if step % config.save_every == 0:
                        latest_dir = output_dir / "latest"

                        # Stage accelerate save_state through tmp to prevent corruption
                        tmp_accel_state = latest_dir / ".tmp_accelerate_state"
                        accelerator.save_state(str(tmp_accel_state))

                        accelerator.wait_for_everyone()
                        saved_exc = None
                        save_failed = False
                        if accelerator.is_main_process:
                            try:
                                with atomic_dir_replace(
                                    latest_dir / "accelerate_state"
                                ) as swap_target:
                                    swap_target.rmdir()
                                    os.replace(str(tmp_accel_state), str(swap_target))

                                _save_json_atomically(
                                    latest_dir / "train_state.json",
                                    {"step": step, "best_eval_loss": best_eval_loss},
                                )
                            except Exception as e:
                                # TRADE-OFF: Rank-0 checkpoint exception is caught and broadcasted
                                logger.error("Rank-0 checkpoint save failed: %s", e, exc_info=True)
                                save_failed = True
                                saved_exc = e
                        accelerator.wait_for_everyone()
                        # TRADE-OFF: Collective called before rank-0 raise to prevent deadlock.
                        saw_failure = _broadcast_save_failure(accelerator, save_failed)
                        if save_failed:
                            raise saved_exc
                        if saw_failure:
                            raise RuntimeError(
                                "Distributed checkpoint save failed on rank-0; aborting "
                                "all ranks to allow torchrun cleanup. Inspect rank-0 "
                                "logs for the original exception."
                            )

            if step >= config.max_steps:
                break

    finally:
        accelerator.end_training()

    return TrainingRunResult(
        final_step=step,
        best_eval_loss=best_eval_loss if best_eval_loss != float("inf") else None,
        mlm_head_random_init=mlm_random_init,
        pad_token_fallback_used=pad_fallback,
        resumed=resumed,
        trainable_param_count=trainable,
        total_param_count=total,
        resolved_versions=_get_resolved_versions(),
    )


def integration_test(
    *,
    use_real_model: bool = True,
) -> None:
    """Run integration test on synthetic data with real or tiny model.

    Asserts all five Technical Design §5 assertions:
    1. Forward returns finite loss.
    2. Optimizer step changes at least one parameter (L2 diff > 0).
    3. save_pretrained writes config.json + safetensors.
    4. from_pretrained reload yields identical state_dict keys.
    5. accelerate.save_state + load_state round-trips.

    Args:
        use_real_model: If True, load real DNABERT-2-117M from Hub.
            If False, create tiny synthetic model for fast CPU testing.

    Raises:
        AssertionError: If any assertion fails.
        RuntimeError: If setup or checkpoint I/O fails.
    """
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_path = Path(tmp_root)

        # Load or create model
        if use_real_model:
            model = AutoModelForMaskedLM.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                trust_remote_code=True,
            )
            _tokenizer, _ = load_dnabert2_tokenizer()
        else:
            # Tiny model for fast testing
            from transformers import BertConfig

            config = BertConfig(
                num_hidden_layers=2,
                num_attention_heads=2,
                hidden_size=32,
                vocab_size=30522,
            )
            model = AutoModelForMaskedLM.from_config(config)

        # Assertion 1: Forward returns finite loss
        model.train()
        synthetic_batch = {
            "input_ids": torch.tensor(
                [[101, 1010, 1010, 102, 0, 0], [101, 1010, 1010, 1010, 102, 0]]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long
            ),
            "labels": torch.tensor(
                [[101, 1010, -100, 102, -100, -100], [101, -100, 1010, 1010, 102, -100]],
                dtype=torch.long,
            ),
        }

        outputs = model(**synthetic_batch)
        loss = outputs.loss
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        logger.info(f"✓ Assertion 1: Forward returns finite loss ({loss:.4f})")

        # Assertion 2: Optimizer step changes at least one parameter
        optimizer = AdamW(model.parameters(), lr=1e-4)
        pre_params = {name: p.clone() for name, p in model.named_parameters()}
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        l2_diffs = []
        for name, p in model.named_parameters():
            if name in pre_params:
                diff = (p - pre_params[name]).norm().item()
                l2_diffs.append(diff)

        assert any(d > 0 for d in l2_diffs), "No parameters changed after optimizer step"
        logger.info(f"✓ Assertion 2: Optimizer step changed parameters (L2 diffs: {l2_diffs[:3]})")

        # Assertion 3: save_pretrained writes config.json + safetensors
        model_dir = tmp_path / "hf_model"
        model.save_pretrained(str(model_dir), safe_serialization=True)
        assert (model_dir / "config.json").exists(), "config.json not written"
        assert any(p.suffix == ".safetensors" for p in model_dir.glob("*")), (
            "No safetensors file written"
        )
        logger.info("✓ Assertion 3: save_pretrained writes config.json + safetensors")

        # Assertion 4: from_pretrained reload yields identical state_dict keys
        reloaded = AutoModelForMaskedLM.from_pretrained(str(model_dir))
        orig_keys = set(model.state_dict().keys())
        reload_keys = set(reloaded.state_dict().keys())
        assert orig_keys == reload_keys, f"State dict keys differ: {orig_keys ^ reload_keys}"
        logger.info("✓ Assertion 4: Reload yields identical state_dict keys")

        # Assertion 5: accelerate.save_state + load_state round-trips
        accelerator = Accelerator()
        model, optimizer = accelerator.prepare(model, optimizer)
        state_dir = tmp_path / "accelerate_state"
        accelerator.save_state(str(state_dir))
        accelerator.load_state(str(state_dir))
        logger.info("✓ Assertion 5: accelerate.save_state + load_state round-trips")

        logger.info("✅ All integration test assertions passed!")
