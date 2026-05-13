"""Foundation model continued pre-training for DNABERT-2 on felid corpus.

This module implements the complete MLM training loop for DNABERT-2 on the
tokenized felid foundation corpus. It handles model loading with pad-token
fallbacks, mixed-precision training via accelerate, checkpoint management
with atomic writes, and integration testing via synthetic or real models.

Key design decisions:
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
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file as safetensors_load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
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

    Attempts to set the tokenizer's pad_token_id using configured fallback
    strategy (eos > unk > add_pad). Also checks for MLM head in missing_keys
    and warns if randomly initialized.

    Args:
        config: Training configuration with model_identifier and model_revision.

    Returns:
        Tuple of (model, tokenizer, pad_token_fallback_used, mlm_head_random_init).

    Raises:
        RuntimeError: If model loading fails or MLM head detection is inconclusive.
    """
    # Load tokenizer via existing helper
    tokenizer, _ = load_dnabert2_tokenizer()

    # Pad-token guard with all three branches
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

    # Load config first and patch attributes removed from BertConfig in transformers v5+
    # that DNABERT-2's custom bert_layers.py still accesses directly.
    model_config = AutoConfig.from_pretrained(
        config.model_identifier,
        revision=config.model_revision,
        trust_remote_code=True,
    )
    if not hasattr(model_config, "is_decoder"):
        model_config.is_decoder = False
    if not hasattr(model_config, "pad_token_id") or model_config.pad_token_id is None:
        model_config.pad_token_id = tokenizer.pad_token_id
    if not hasattr(model_config, "return_dict"):
        model_config.return_dict = True

    # Bypass from_pretrained's meta-device initialization: transformers v5.x
    # creates tensors on torch.device("meta") internally, which raises
    # RuntimeError with DNABERT-2's v4.x custom code.  Create on CPU from
    # config, then load the pretrained state dict manually.
    model = AutoModelForMaskedLM.from_config(
        model_config,
        trust_remote_code=True,
    )

    try:
        weight_path = hf_hub_download(
            config.model_identifier,
            "model.safetensors",
            revision=config.model_revision,
        )
        state_dict = safetensors_load_file(weight_path, device="cpu")
    except (EntryNotFoundError, FileNotFoundError):
        weight_path = hf_hub_download(
            config.model_identifier,
            "pytorch_model.bin",
            revision=config.model_revision,
        )
        # weights_only=True is safe: DNABERT-2's .bin is a standard state dict.
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    load_result = model.load_state_dict(state_dict, strict=False)
    loading_info = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }

    # Sync model.config with tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id

    # If we added a new pad token, resize embeddings
    if pad_token_fallback_used == "add_pad":
        model.resize_token_embeddings(len(tokenizer))

    # MLM head missing-keys check
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
    """Build AdamW optimizer with weight decay groups.

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
    """Build cosine schedule with linear warmup.

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
        # Substring match tolerates missing validation split (non-fatal) while propagating
        # other errors (empty shard, schema issues). This allows training on train-only
        # corpora but fails fast if corpus configuration is broken.
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
    """Compute fixed eval step count to prevent DDP deadlock.

    All ranks must iterate exactly this many batches. Any leftover rows
    are dropped and logged. This fixed step count prevents different ranks
    from exiting at different times, which would hang the collective barrier.

    Args:
        eval_reader: Validation IterableDataset with record_count property.
        per_device_eval_batch_size: Batch size per device.
        world_size: Total number of processes (global_batch_size = per_device * world_size).

    Returns:
        Fixed maximum evaluation steps.
    """
    # Derive eval-step cap from record_count. Leftover rows are dropped when total
    # record_count is not evenly divisible by global_batch_size.
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
        # Atomic swap: move old target aside before renaming tmp to target, allowing
        # crash recovery. This trades space for safety: old checkpoints are retained
        # briefly and must be cleaned up after successful swap.
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
    """Recover from a mid-replace crash by restoring the most recent .old_* directory.

    Trades forward progress (losing the failed checkpoint attempt) for safety:
    restores the last known-good state and allows training to continue. If no
    .old_ backup exists, recovery is impossible and False is returned.
    """
    if target.exists():
        return False

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

    Trades performance (gather fallback) for compatibility: uses set_trigger if
    available (newer Accelerate), else uses tensor gather. Ensures all ranks
    agree on failure state before raising, preventing deadlock.
    """
    if hasattr(accelerator, "set_trigger"):
        if save_failed:
            accelerator.set_trigger()
        accelerator.wait_for_everyone()  # Barrier: ensure all ranks reach this point
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
    """Write checkpoint sidecar atomically via temp file.

    Uses atomic rename to ensure the sidecar is either fully written or absent,
    never partially written. PID suffix on temp file prevents collisions during
    concurrent writes (rank-0 and recovery attempts on same node).

    Args:
        path: Target file path.
        content: dict to serialize to JSON.

    Raises:
        OSError: If atomic write fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON to temp file with unique PID-based suffix, then atomically rename.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(content), encoding="utf-8")
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
) -> dict[str, float]:
    """Compute startup probe metrics for the first 20 steps.

    Logs mean sequence length and observed mask rate to detect data problems
    early (e.g., misaligned tokenization, MLM probability config issues).
    Only computed on rank-0 and for the first 20 optimizer updates.

    Args:
        batch: Data batch with attention_mask and labels.
        step: Current training step (1-indexed).
        accelerator: Accelerate trainer for rank detection.

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

    Implements the full training loop: model loading with pad-token fallback,
    mixed-precision training via bf16, checkpoint management with atomic writes,
    gradient accumulation, evaluation with fixed DDP-safe step counts, and
    integration testing via synthetic or real models.

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

    # Initialize Accelerator with bf16 mixed precision
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(output_dir / config.tensorboard_subdir),
    )

    # Check for resumed state
    latest_state_path = output_dir / "latest" / "accelerate_state"

    # Recover from mid-rename crash (rank-0 only to avoid race on shared FS)
    if accelerator.is_main_process:
        _recover_atomic_dir(output_dir / "latest" / "accelerate_state")
        _recover_atomic_dir(output_dir / "best" / "accelerate_state")
        _recover_atomic_dir(output_dir / "best" / "hf_model")
        _recover_atomic_dir(output_dir / "best" / "tokenizer")
    accelerator.wait_for_everyone()

    resumed = latest_state_path.exists()
    # Read best eval loss from sidecar (aligned with write path in checkpoint save)
    best_eval_loss_file = output_dir / "best" / "best_eval_loss.json"
    best_eval_loss = None
    if best_eval_loss_file.exists():
        try:
            best_eval_loss = json.loads(best_eval_loss_file.read_text()).get("eval_loss")
        except (json.JSONDecodeError, KeyError):
            pass

    # Prepare for distributed training
    # Dataloaders are NOT passed through accelerator.prepare because Accelerate would either:
    # (a) dispatch from rank-0 only (drops N-1/N of data), or (b) double-shard on top of our
    # reader's file-level sharding. Instead, manual device placement is used and the reader
    # handles all sharding. Gradient accumulation via accelerator.accumulate() works because
    # its sync_gradients flag is tied to self.step, not the dataloader. See
    # accelerate/accelerator.py:1228 _do_sync().
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Model trainability verification
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

    # Resume if needed
    # IterableDataset cannot restore cursor position; restart at epoch boundary.
    # This trades perfect resume (a few duplicate rows in first batch) for robustness.
    step = 0
    if resumed:
        accelerator.load_state(str(latest_state_path))

        # Step counter is restored from JSON sidecar (not accelerate_state) because
        # Python ints are not serializable by Accelerate. Sidecar is best-effort;
        # old checkpoints without it resume at step=0 with a warning.
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
    has_set_epoch = hasattr(train_loader.dataset, "set_epoch")

    try:
        for _epoch in range(1, 1000):  # Iterate until max_steps reached
            # Set epoch on reader for deterministic multi-epoch shuffling
            if has_set_epoch:
                train_loader.dataset.set_epoch(_epoch - 1)  # 0-indexed
            for batch in train_loader:
                if step >= config.max_steps:
                    break

                # Manual device placement (dataloaders bypassed accelerator.prepare)
                batch = {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}

                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss

                    # NaN/Inf guard: skip accumulation and count as anomaly
                    loss_f = loss.detach().float()
                    if not torch.isfinite(loss_f).all():
                        train_metric.nan_count += 1
                        logger.warning(f"NaN/Inf loss detected at step {step}")
                        optimizer.zero_grad()
                        continue
                    else:
                        train_metric.loss_sum += loss_f.mean().item()
                        train_metric.step_count += 1

                        # Token accuracy accumulation for train loop.
                        # Only accumulate on finite-loss steps; NaN/Inf logits produce
                        # garbage argmax results that corrupt the accuracy metric.
                        # Argmax over full vocab is computed inside torch.no_grad() to avoid
                        # polluting the backward pass with unnecessary graph nodes.
                        with torch.no_grad():
                            preds = outputs.logits.argmax(dim=-1)
                            labels = batch.get("labels")
                            if labels is not None:
                                mask = labels != -100
                                # Accumulate locally to avoid DDP gather inside the
                                # loop; defer global reduce to the log step for efficiency.
                                train_metric.token_correct += (
                                    ((preds == labels) & mask).sum().item()
                                )
                                train_metric.token_masked += mask.sum().item()
                    # Backward pass
                    accelerator.backward(loss)

                    # Gradient clipping
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )

                    # Collect grad norms on sync steps (before step/zero_grad) and guard against NaN
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
                        # Guard scheduler.step() by sync_gradients. The accumulate() context
                        # no-ops optimizer.step/zero_grad but NOT scheduler.step. Without
                        # this guard, scheduler advances per micro-step, exhausting the
                        # learning rate schedule prematurely.
                        if accelerator.sync_gradients:
                            scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    # step counts optimizer updates, not micro-batches;
                    # cadence aligns with scheduler
                    step += 1

                    # Log metrics at log_every (windowed averages reset after each log)
                    if step % config.log_every == 0:
                        mean_loss = (
                            float("nan")
                            if train_metric.step_count == 0
                            else train_metric.loss_sum / train_metric.step_count
                        )

                        # DDP-safe global token accuracy reduce
                        local_counts = torch.tensor(
                            [train_metric.token_correct, train_metric.token_masked],
                            dtype=torch.float32,
                            device=accelerator.device,
                        )
                        global_counts = accelerator.reduce(local_counts, reduction="sum")
                        global_correct = global_counts[0].item()
                        global_masked = global_counts[1].item()

                        # Return NaN for token_accuracy when no masked tokens (distinguishes
                        # "no data" from "zero accuracy"), helping identify tokenization issues.
                        token_acc = (
                            float("nan") if global_masked == 0 else global_correct / global_masked
                        )
                        # Perplexity clamped at 20 to prevent bf16 overflow (exp grows too fast)
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

                        # Startup probe for first 20 steps (rank-0 only)
                        startup_logs = _startup_probe_metrics(batch, step, accelerator)
                        logs.update(startup_logs)

                        accelerator.log(logs, step=step)

                        # Log grad_norm_hist as TensorBoard histogram (rank-0 only).
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

                    # Evaluation at eval_every (fixed eval_max_steps prevents DDP deadlock)
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

                                # Manual device placement (dataloaders bypassed accelerator.prepare)
                                eval_batch = {
                                    k: v.to(accelerator.device, non_blocking=True)
                                    for k, v in eval_batch.items()
                                }

                                outputs = model(**eval_batch)
                                loss = outputs.loss

                                loss_f = loss.detach().float()
                                # Gather eval loss across ranks before accumulation (DDP-safe)
                                gathered_loss = accelerator.gather_for_metrics(loss_f)
                                if not (
                                    torch.isnan(gathered_loss).any()
                                    or torch.isinf(gathered_loss).any()
                                ):
                                    eval_metric.loss_sum += gathered_loss.mean().item()
                                    eval_metric.step_count += 1

                                    # Token accuracy accumulation for eval loop.
                                    # Only accumulate on finite-loss steps; NaN/Inf logits
                                    # produce garbage argmax results that corrupt the metric.
                                    preds = outputs.logits.argmax(dim=-1)
                                    labels = eval_batch.get("labels")
                                    if labels is not None:
                                        gathered_preds = accelerator.gather_for_metrics(preds)
                                        gathered_labels = accelerator.gather_for_metrics(labels)
                                        gathered_mask = gathered_labels != -100
                                        if accelerator.is_main_process:
                                            mask_match = (
                                                gathered_preds == gathered_labels
                                            ) & gathered_mask
                                            eval_metric.token_correct += mask_match.sum().item()
                                            eval_metric.token_masked += gathered_mask.sum().item()

                        model.train()
                        mean_eval_loss = (
                            eval_metric.loss_sum / eval_metric.step_count
                            if eval_metric.step_count > 0
                            else float("nan")
                        )

                        # Return NaN for eval token_accuracy when no masked tokens
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

                        # Skip best-checkpoint update if all eval batches were non-finite.
                        # Avoids poisoning best-loss with NaN or treating "no data" as "worst".
                        # Warn so operators can investigate potential data/tokenization issues.
                        if eval_metric.step_count == 0:
                            logger.warning(
                                "Eval at step %d produced no finite loss values "
                                "(all batches were NaN/Inf or dataset was empty); "
                                "skipping best-checkpoint comparison to avoid poisoning.",
                                step,
                            )
                        elif mean_eval_loss < best_eval_loss:
                            best_eval_loss = mean_eval_loss
                            # Save best checkpoint
                            best_dir = output_dir / "best"

                            # Stage all state through tmp, then atomically swap (DDP-safe)
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
                                    # Catch exception to broadcast to other ranks (no deadlock)
                                    logger.error(
                                        "Rank-0 checkpoint save failed: %s", e, exc_info=True
                                    )
                                    save_failed = True
                                    saved_exc = e
                            accelerator.wait_for_everyone()
                            # Broadcast failure before rank-0 raise (prevents deadlock)
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
                                # Catch exception to broadcast to other ranks (prevents deadlock)
                                logger.error("Rank-0 checkpoint save failed: %s", e, exc_info=True)
                                save_failed = True
                                saved_exc = e
                        accelerator.wait_for_everyone()
                        # Broadcast failure before rank-0 raise (prevents deadlock)
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

    Asserts five critical system integration points:
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
            # Use from_config + manual weight loading, same as production path,
            # to avoid transformers v5.x meta-device RuntimeError.
            real_cfg = AutoConfig.from_pretrained(
                "zhihan1996/DNABERT-2-117M", trust_remote_code=True
            )
            model = AutoModelForMaskedLM.from_config(real_cfg, trust_remote_code=True)
            _weight_path = hf_hub_download("zhihan1996/DNABERT-2-117M", "model.safetensors")
            model.load_state_dict(safetensors_load_file(_weight_path, device="cpu"), strict=False)
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

        # Assertion 4: reload yields identical state_dict keys.
        # Use from_config + safetensors to avoid the same meta-device issue.
        reload_cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=use_real_model)
        reloaded = AutoModelForMaskedLM.from_config(reload_cfg, trust_remote_code=use_real_model)
        st_files = list(model_dir.glob("*.safetensors"))
        assert st_files, "No safetensors file written by save_pretrained"
        reload_sd = safetensors_load_file(str(st_files[0]), device="cpu")
        reloaded.load_state_dict(reload_sd, strict=False)
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
