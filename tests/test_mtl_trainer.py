"""Unit tests for trainer NaN/Inf control-flow and MTL helper contracts.

These tests focus on small, deterministic trainer behaviors that are cheaper to
exercise than the synthetic end-to-end integration harness while still
protecting the non-finite-loss guards in both training entry points.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from jaguar_geo_assign.fine_tune.trainer import _compute_mtl_loss, run_jaguar_mtl_training
from jaguar_geo_assign.pretrain.foundation_training import run_felid_foundation_training


def test_compute_mtl_loss_uses_true_huber_delta_semantics() -> None:
    """Regression loss must follow ``nn.HuberLoss(delta=...)`` semantics.

    This guards the configuration contract for ``huber_delta``. For
    ``delta != 1``, ``SmoothL1Loss(beta=delta)`` produces a differently scaled
    value, so this test uses a non-unit threshold that would fail under the old
    implementation.
    """

    outputs = SimpleNamespace(coordinate=torch.tensor([[3.0, 0.0]], dtype=torch.float32))
    batch = {"coord_target": torch.zeros((1, 2), dtype=torch.float32)}
    huber_delta = 2.0
    reg_loss_weight = 0.5

    total_loss, cls_loss, reg_loss = _compute_mtl_loss(
        outputs,
        batch,
        cls_loss_weight=0.0,
        reg_loss_weight=reg_loss_weight,
        huber_delta=huber_delta,
    )

    expected_reg_loss = torch.nn.HuberLoss(delta=huber_delta)(
        outputs.coordinate,
        batch["coord_target"],
    )

    assert cls_loss.item() == pytest.approx(0.0)
    assert reg_loss.item() == pytest.approx(expected_reg_loss.item())
    assert total_loss.item() == pytest.approx(reg_loss_weight * expected_reg_loss.item())


class _FakeAccelerator:
    """Minimal accelerator stub for unit-testing MTL loop control flow.

    The test only needs to observe whether ``backward()`` is invoked for a
    given loss tensor, so the stub keeps the API surface intentionally narrow.
    """

    def __init__(self) -> None:
        """Initialize the stub with CPU defaults and call capture buffers."""
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self.is_main_process = False
        self.num_processes = 1
        self.backward_calls: list[torch.Tensor] = []
        self.logged: list[tuple[dict[str, float], int | None]] = []

    def init_trackers(self, *_args, **_kwargs) -> None:
        """Match the production API without creating external side effects."""

    def wait_for_everyone(self) -> None:
        """Mirror the barrier API without synchronizing real processes."""

    def log(self, values: dict[str, float], step: int | None = None) -> None:
        """Record logged scalars so the test can inspect NaN-step reporting."""
        self.logged.append((values, step))

    def prepare(self, *args):
        """Return inputs unchanged because the test runs on a single CPU process."""
        return args

    def accumulate(self, _model: nn.Module):
        """Provide a no-op accumulation context for the training loop."""
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        """Capture each backward call instead of building gradients."""
        self.backward_calls.append(loss)

    def clip_grad_norm_(self, _parameters, _max_norm: float) -> torch.Tensor:
        """Return a finite norm so only the loss guard controls skipping."""
        return torch.tensor(1.0)

    def reduce(self, values: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Behave like an identity reduction in the single-process test."""
        del reduction
        return values

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Expose the underlying model unchanged in the single-process test."""
        return model

    def end_training(self) -> None:
        """Match the production cleanup hook without doing any work."""


class _FakeMTLModel(nn.Module):
    """Tiny stand-in model with explicit frozen/unfrozen parameters.

    Separate parameters make it easy for the test to emulate the phase-1 freeze
    and phase-2 unfreeze contracts without constructing a full DNABERT stack.
    """

    def __init__(self) -> None:
        """Create one initially-trainable parameter and one frozen parameter."""
        super().__init__()
        self.phase1_param = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.phase2_param = nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Return placeholder outputs; loss computation is patched by the test."""
        del attention_mask
        batch_size = input_ids.shape[0]
        return SimpleNamespace(coordinate=torch.zeros((batch_size, 2), dtype=torch.float32))


class _FakeFoundationModel(nn.Module):
    """Tiny pretraining model that emits one NaN loss and then one finite loss."""

    def __init__(self) -> None:
        """Create one trainable parameter and a deterministic call counter."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self._call_count = 0

    def forward(self, **_batch):
        """Return a NaN loss once, then a finite loss with valid logits."""
        self._call_count += 1
        if self._call_count == 1:
            return SimpleNamespace(
                loss=torch.tensor(float("nan")),
                logits=torch.full((1, 2, 2), float("nan")),
            )
        return SimpleNamespace(
            loss=torch.tensor(1.25),
            logits=torch.tensor([[[0.1, 0.9], [0.9, 0.1]]], dtype=torch.float32),
        )


class _StaticFoundationLoader:
    """Minimal iterable loader with a dataset attribute for foundation training."""

    def __init__(self) -> None:
        """Initialize two fixed CPU batches and record-count metadata."""
        self.dataset = SimpleNamespace(record_count=2)
        self._batches = [
            {
                "input_ids": torch.tensor([[101, 102]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "labels": torch.tensor([[1, 0]], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([[101, 102]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "labels": torch.tensor([[1, 0]], dtype=torch.long),
            },
        ]

    def __iter__(self):
        """Yield the two fixed batches on every fresh iteration."""
        yield from self._batches


def test_run_jaguar_mtl_training_skips_backward_on_non_finite_loss(tmp_path: Path) -> None:
    """Injected NaN losses must not trigger ``accelerator.backward()``.

    This regression test drives both training phases with one NaN micro-batch
    followed by one finite micro-batch and asserts that backward executes only
    for the finite losses. It also verifies each phase clears gradients once for
    the skipped NaN micro-batch and once after the successful optimizer update.
    That guards the control-flow contract added to avoid contaminating
    accumulated gradients with non-finite values.
    """

    config = SimpleNamespace(
        output_dir=tmp_path / "out",
        seed=0,
        backbone_path=tmp_path / "backbone",
        gradient_accumulation_steps=1,
        tensorboard_subdir="tb",
        n_folds=2,
        n_biomes=2,
        dropout=0.1,
        lr_heads_phase1=1e-3,
        weight_decay=0.0,
        warmup_fraction=0.0,
        phase1_steps=1,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        huber_delta=1.0,
        gradient_clip=1.0,
        log_every=1,
        eval_every=1,
        save_every=100,
        unfreeze_layers=1,
        lr_backbone_phase2=1e-4,
        lr_heads_phase2=1e-3,
        phase2_steps=1,
        fold_index=0,
    )
    train_loader = [
        {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "coord_target": torch.zeros((1, 2), dtype=torch.float32),
            "biome_label": torch.zeros(1, dtype=torch.long),
        },
        {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "coord_target": torch.zeros((1, 2), dtype=torch.float32),
            "biome_label": torch.zeros(1, dtype=torch.long),
        },
    ]
    coord_stats = SimpleNamespace(lat_mean=0.0, lat_std=1.0, lon_mean=0.0, lon_std=1.0)
    fake_model = _FakeMTLModel()
    fake_accelerator = _FakeAccelerator()
    phase1_optimizer = MagicMock()
    phase1_scheduler = MagicMock()
    phase1_scheduler.get_last_lr.return_value = [1e-3]
    phase2_optimizer = MagicMock()
    phase2_scheduler = MagicMock()
    phase2_scheduler.get_last_lr.return_value = [1e-4, 1e-3]

    def _freeze_fake_backbone(model: _FakeMTLModel) -> None:
        """Emulate phase-1 freezing by disabling all fake parameters."""
        model.phase1_param.requires_grad = False
        model.phase2_param.requires_grad = False

    def _unfreeze_fake_backbone(model: _FakeMTLModel, _n_layers: int) -> None:
        """Emulate phase-2 unfreezing by re-enabling one fake parameter."""
        model.phase2_param.requires_grad = True

    loss_sequence = [
        (torch.tensor(float("nan")), torch.tensor(0.0), torch.tensor(0.0)),
        (torch.tensor(1.0), torch.tensor(0.25), torch.tensor(0.75)),
        (torch.tensor(float("nan")), torch.tensor(0.0), torch.tensor(0.0)),
        (torch.tensor(2.0), torch.tensor(0.5), torch.tensor(1.5)),
    ]

    with (
        patch("jaguar_geo_assign.fine_tune.trainer.load_mtl_finetune_config", return_value=config),
        patch(
            "jaguar_geo_assign.fine_tune.trainer.AutoModel.from_pretrained", return_value=object()
        ),
        patch("jaguar_geo_assign.fine_tune.trainer._load_tokenizer", return_value=object()),
        patch(
            "jaguar_geo_assign.fine_tune.trainer.build_fold_dataloaders",
            return_value=(train_loader, None, coord_stats),
        ),
        patch("jaguar_geo_assign.fine_tune.trainer.Accelerator", return_value=fake_accelerator),
        patch("jaguar_geo_assign.fine_tune.trainer.JaguarMTLModel", return_value=fake_model),
        patch(
            "jaguar_geo_assign.fine_tune.trainer._compute_baselines",
            return_value={"macro_f1": 0.0, "haversine_km_median": 0.0, "haversine_km_mean": 0.0},
        ),
        patch(
            "jaguar_geo_assign.fine_tune.trainer._build_phase1_optimizer_and_scheduler",
            return_value=(phase1_optimizer, phase1_scheduler),
        ),
        patch(
            "jaguar_geo_assign.fine_tune.trainer._build_phase2_optimizer_and_scheduler",
            return_value=(phase2_optimizer, phase2_scheduler),
        ),
        patch(
            "jaguar_geo_assign.fine_tune.trainer._freeze_backbone",
            side_effect=_freeze_fake_backbone,
        ),
        patch(
            "jaguar_geo_assign.fine_tune.trainer._unfreeze_last_n_layers",
            side_effect=_unfreeze_fake_backbone,
        ),
        patch("jaguar_geo_assign.fine_tune.trainer._compute_mtl_loss", side_effect=loss_sequence),
    ):
        result = run_jaguar_mtl_training(tmp_path / "config.toml")

    assert result.phase1_steps_completed == 1
    assert result.phase2_steps_completed == 1
    assert len(fake_accelerator.backward_calls) == 2
    assert [float(loss.item()) for loss in fake_accelerator.backward_calls] == [1.0, 2.0]
    assert phase1_optimizer.step.call_count == 1
    assert phase2_optimizer.step.call_count == 1
    assert phase1_optimizer.zero_grad.call_count == phase1_optimizer.step.call_count + 1
    assert phase2_optimizer.zero_grad.call_count == phase2_optimizer.step.call_count + 1

    train_logs = [
        values for values, _step in fake_accelerator.logged if "train/nan_steps" in values
    ]
    assert len(train_logs) == 2
    assert [logs["train/nan_steps"] for logs in train_logs] == [1.0, 1.0]


def test_run_foundation_training_skips_backward_on_non_finite_loss(tmp_path: Path) -> None:
    """Injected NaN losses must not trigger ``accelerator.backward()`` in pretraining.

    This mirrors the MTL regression test at the foundation-training entry point:
    the first micro-batch emits a NaN loss and must clear gradients and continue,
    while the following finite micro-batch is allowed to backpropagate and step.
    """

    config = SimpleNamespace(
        output_dir=tmp_path / "out",
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_steps=0,
        max_steps=1,
        gradient_accumulation_steps=1,
        tensorboard_subdir="tb",
        log_every=1,
        eval_every=1,
        eval_max_steps=None,
        save_every=100,
        gradient_clip=1.0,
        per_device_eval_batch_size=1,
    )
    fake_accelerator = _FakeAccelerator()
    fake_model = _FakeFoundationModel()
    optimizer = MagicMock()
    scheduler = MagicMock()
    scheduler.get_last_lr.return_value = [1e-4]

    with (
        patch("jaguar_geo_assign.config.load_foundation_training_config", return_value=config),
        patch(
            "jaguar_geo_assign.pretrain.foundation_training._build_model_and_tokenizer",
            return_value=(fake_model, object(), "none", False),
        ),
        patch(
            "jaguar_geo_assign.pretrain.foundation_training._build_optimizer",
            return_value=optimizer,
        ),
        patch(
            "jaguar_geo_assign.pretrain.foundation_training._build_scheduler",
            return_value=scheduler,
        ),
        patch(
            "jaguar_geo_assign.pretrain.foundation_training._build_dataloaders",
            return_value=(_StaticFoundationLoader(), None),
        ),
        patch(
            "jaguar_geo_assign.pretrain.foundation_training.Accelerator",
            return_value=fake_accelerator,
        ),
    ):
        result = run_felid_foundation_training(tmp_path / "config.toml")

    assert result.final_step == 1
    assert len(fake_accelerator.backward_calls) == 1
    assert float(fake_accelerator.backward_calls[0].item()) == pytest.approx(1.25)
    assert optimizer.step.call_count == 1
    assert scheduler.step.call_count == 1
    assert optimizer.zero_grad.call_count == optimizer.step.call_count + 1

    train_logs = [
        values for values, _step in fake_accelerator.logged if "train/nan_steps" in values
    ]
    assert len(train_logs) == 1
    assert train_logs[0]["train/nan_steps"] == 1
