"""Unit tests for MIL trainer control-flow and NaN handling.

These tests mirror the existing MTL trainer regressions but target the new
single-phase MIL loop so non-finite attention/loss events cannot silently slip
through optimizer stepping.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from jaguar_geo_assign.fine_tune.dataset import CoordStats
from jaguar_geo_assign.fine_tune.mil_dataset import MILBagDataset, mil_collate_fn
from jaguar_geo_assign.fine_tune.mil_trainer import _run_mil_evaluation, run_jaguar_mil_training
from jaguar_geo_assign.fine_tune.model import JaguarMTLOutput
from jaguar_geo_assign.fine_tune.positional_mil import JaguarPositionalMILNetwork


class _FakeAccelerator:
    """Small accelerator stub for deterministic MIL trainer unit tests."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self.is_main_process = False
        self.backward_calls: list[torch.Tensor] = []
        self.logged: list[tuple[dict[str, float], int | None]] = []

    def init_trackers(self, *_args, **_kwargs) -> None:
        """Match the production API without creating tracker files."""

    def log(self, values: dict[str, float], step: int | None = None) -> None:
        """Record logs so the test can inspect MIL NaN-step counters."""

        self.logged.append((values, step))

    def prepare(self, *args):
        """Return all inputs unchanged for the single-process CPU test path."""

        return args

    def accumulate(self, _model: nn.Module):
        """Provide a no-op accumulation context manager."""

        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        """Capture backward calls instead of building real gradients."""

        self.backward_calls.append(loss)

    def clip_grad_norm_(self, _parameters, _max_norm: float) -> torch.Tensor:
        """Return a finite norm so the loss guard controls the skip path."""

        return torch.tensor(1.0)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Return the model unchanged in the single-process test."""

        return model

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mirror a single-process gather operation."""

        return tensor

    def end_training(self) -> None:
        """Match the production cleanup hook without side effects."""


class _AccumulatingFakeAccelerator(_FakeAccelerator):
    """Accelerator stub that simulates alternating accumulation boundaries.

    The real MIL regression needs autograd-enabled backward calls so the test can
    observe whether a valid micro-step gradient survives a later NaN skip until
    the next optimizer boundary.
    """

    def __init__(self, sync_schedule: list[bool]) -> None:
        super().__init__()
        self._sync_schedule = sync_schedule
        self._accumulate_index = 0

    @contextmanager
    def accumulate(self, _model: nn.Module):
        """Apply a predefined ``sync_gradients`` value for each micro-step."""

        if self._accumulate_index >= len(self._sync_schedule):
            raise AssertionError("Accumulation schedule exhausted before training finished.")
        self.sync_gradients = self._sync_schedule[self._accumulate_index]
        self._accumulate_index += 1
        yield

    def backward(self, loss: torch.Tensor) -> None:
        """Run real autograd so optimizer updates encode surviving gradients."""

        self.backward_calls.append(loss.detach())
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm: float) -> torch.Tensor:
        """Delegate to PyTorch so clipping sees the real accumulated gradients."""

        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class _FakeMILModel(nn.Module):
    """Tiny MIL stand-in with one trainable parameter and fixed output shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, embeddings: torch.Tensor, bp_positions: torch.Tensor) -> JaguarMTLOutput:
        """Ignore the batch contents and return shape-compatible MIL outputs."""

        del embeddings, bp_positions
        return JaguarMTLOutput(
            coordinate=torch.zeros(2, dtype=torch.float32),
            biome_logits=torch.zeros(2, dtype=torch.float32),
        )


class _TrackingDeque:
    """Record plateau-window appends so trainer regressions stay observable in tests."""

    def __init__(self, maxlen: int | None = None) -> None:
        self.maxlen = maxlen
        self.items: list[float] = []

    def append(self, value: float) -> None:
        """Append one plateau value while honoring the configured max length."""

        self.items.append(value)
        if self.maxlen is not None and len(self.items) > self.maxlen:
            self.items.pop(0)

    def __len__(self) -> int:
        """Expose deque length for the regression assertion."""

        return len(self.items)


def _write_manifest_record(
    manifest_path: Path,
    *,
    shard_path: Path,
    individual_id: str,
    latitude: float,
    longitude: float,
) -> None:
    """Write one synthetic manifest record for a coordinate-only MIL test."""

    record = {
        "individual_id": individual_id,
        "sample_id": f"sample-{individual_id}",
        "shard_path": str(shard_path),
        "n_windows": 3,
        "latitude": latitude,
        "longitude": longitude,
        "biome_population_label": "unused-in-coordinate-only-mode",
    }
    manifest_path.write_text(f"{json.dumps(record)}\n", encoding="utf-8")


def test_run_jaguar_mil_training_skips_backward_on_non_finite_loss(tmp_path: Path) -> None:
    """MIL training must skip backward and continue when loss becomes non-finite."""

    config = SimpleNamespace(
        output_dir=tmp_path / "out",
        seed=0,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        tensorboard_subdir="tb",
        embedding_dim=4,
        hidden_dim=2,
        n_biomes=2,
        dropout=0.0,
        locus_dropout=0.1,
        genome_scale=1.0,
        lr_mil=1e-3,
        weight_decay=0.0,
        warmup_fraction=0.1,
        mil_steps=1,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        huber_delta=1.0,
        gradient_clip=1.0,
        log_every=1,
        eval_every=10,
        save_every=100,
        fold_index=0,
        patience=None,
        eval_max_steps=None,
    )
    train_loader = [
        {
            "embeddings": torch.ones((3, 4), dtype=torch.float32),
            "bp_positions": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "coord_target": torch.zeros(2, dtype=torch.float32),
            "biome_label": torch.tensor(0, dtype=torch.long),
        },
        {
            "embeddings": torch.ones((3, 4), dtype=torch.float32),
            "bp_positions": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "coord_target": torch.zeros(2, dtype=torch.float32),
            "biome_label": torch.tensor(0, dtype=torch.long),
        },
    ]
    coord_stats = SimpleNamespace(lat_mean=0.0, lat_std=1.0, lon_mean=0.0, lon_std=1.0)
    fake_model = _FakeMILModel()
    fake_accelerator = _FakeAccelerator()
    optimizer = MagicMock()
    scheduler = MagicMock()
    scheduler.get_last_lr.return_value = [1e-3]
    loss_sequence = [
        (torch.tensor(float("nan")), torch.tensor(0.0), torch.tensor(0.0)),
        (torch.tensor(1.5), torch.tensor(0.5), torch.tensor(1.0)),
    ]

    with (
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.load_mil_finetune_config", return_value=config
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.build_mil_fold_dataloaders",
            return_value=(train_loader, None, coord_stats),
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.Accelerator", return_value=fake_accelerator
        ) as accelerator_ctor,
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.JaguarPositionalMILNetwork",
            return_value=fake_model,
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._build_mil_optimizer_and_scheduler",
            return_value=(optimizer, scheduler),
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._compute_baselines",
            return_value={"macro_f1": 0.0, "haversine_km_median": 0.0, "haversine_km_mean": 0.0},
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._compute_mtl_loss", side_effect=loss_sequence
        ),
    ):
        result = run_jaguar_mil_training(tmp_path / "config.toml")

    assert result.steps_completed == 1
    accelerator_ctor.assert_called_once_with(
        mixed_precision="no",
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=str(config.output_dir / config.tensorboard_subdir),
    )
    assert len(fake_accelerator.backward_calls) == 1
    assert float(fake_accelerator.backward_calls[0].item()) == 1.5
    assert optimizer.step.call_count == 1
    assert optimizer.zero_grad.call_count == 1
    nan_logs = [
        values for values, _step in fake_accelerator.logged if "mil/nan_attention_steps" in values
    ]
    assert len(nan_logs) == 1
    assert nan_logs[0]["mil/nan_attention_steps"] == 1.0


def test_run_jaguar_mil_training_preserves_accumulated_gradients_across_nan_micro_step(
    tmp_path: Path,
) -> None:
    """A NaN sync micro-step must not erase a valid gradient from the prior micro-step.

    This regression drives ``gradient_accumulation_steps=2`` with one finite
    micro-step, a NaN on the next accumulation boundary, and a later finite sync
    boundary. The final optimizer update must still include the earlier valid
    gradient, proving the NaN skip path no longer clears accumulated state.
    """

    config = SimpleNamespace(
        output_dir=tmp_path / "out",
        seed=0,
        gradient_accumulation_steps=2,
        mixed_precision="no",
        tensorboard_subdir="tb",
        embedding_dim=4,
        hidden_dim=2,
        n_biomes=2,
        dropout=0.0,
        locus_dropout=0.1,
        genome_scale=1.0,
        lr_mil=1e-3,
        weight_decay=0.0,
        warmup_fraction=0.1,
        mil_steps=1,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        huber_delta=1.0,
        gradient_clip=1.0,
        log_every=1,
        eval_every=10,
        save_every=100,
        fold_index=0,
        patience=None,
        eval_max_steps=None,
    )
    train_loader = [
        {
            "embeddings": torch.ones((3, 4), dtype=torch.float32),
            "bp_positions": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "coord_target": torch.zeros(2, dtype=torch.float32),
            "biome_label": torch.tensor(0, dtype=torch.long),
        }
        for _ in range(4)
    ]
    coord_stats = SimpleNamespace(lat_mean=0.0, lat_std=1.0, lon_mean=0.0, lon_std=1.0)
    fake_model = _FakeMILModel()
    fake_accelerator = _AccumulatingFakeAccelerator(sync_schedule=[False, True, False, True])
    optimizer = torch.optim.SGD(fake_model.parameters(), lr=0.1)
    scheduler = MagicMock()
    scheduler.get_last_lr.return_value = [0.1]
    loss_plan = iter(
        [
            lambda: fake_model.weight,
            lambda: torch.tensor(float("nan")),
            lambda: fake_model.weight * 0.0,
            lambda: fake_model.weight * 0.0,
        ]
    )

    def _loss_side_effect(*_args, **_kwargs):
        """Return the scripted loss sequence while keeping finite losses on-graph."""

        total_loss = next(loss_plan)()
        zero_loss = fake_model.weight * 0.0
        return total_loss, zero_loss, zero_loss

    with (
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.load_mil_finetune_config", return_value=config
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.build_mil_fold_dataloaders",
            return_value=(train_loader, None, coord_stats),
        ),
        patch("jaguar_geo_assign.fine_tune.mil_trainer.Accelerator", return_value=fake_accelerator),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.JaguarPositionalMILNetwork",
            return_value=fake_model,
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._build_mil_optimizer_and_scheduler",
            return_value=(optimizer, scheduler),
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._compute_baselines",
            return_value={"macro_f1": 0.0, "haversine_km_median": 0.0, "haversine_km_mean": 0.0},
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._compute_mtl_loss",
            side_effect=_loss_side_effect,
        ),
    ):
        result = run_jaguar_mil_training(tmp_path / "config.toml")

    assert result.steps_completed == 1
    assert len(fake_accelerator.backward_calls) == 3
    assert torch.isclose(fake_model.weight.detach(), torch.tensor(0.9), atol=1e-6)


def test_run_jaguar_mil_training_appends_plateau_window_once_per_eval_step(
    tmp_path: Path,
) -> None:
    """Eval-boundary steps must not double-append plateau values for the same step."""

    config = SimpleNamespace(
        output_dir=tmp_path / "out",
        seed=0,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        tensorboard_subdir="tb",
        embedding_dim=4,
        hidden_dim=2,
        n_biomes=2,
        dropout=0.0,
        locus_dropout=0.1,
        genome_scale=1.0,
        lr_mil=1e-3,
        weight_decay=0.0,
        warmup_fraction=0.1,
        mil_steps=300,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        huber_delta=1.0,
        gradient_clip=1.0,
        log_every=100,
        eval_every=100,
        save_every=1000,
        fold_index=0,
        patience=None,
        eval_max_steps=None,
    )
    train_loader = [
        {
            "embeddings": torch.ones((3, 4), dtype=torch.float32),
            "bp_positions": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "coord_target": torch.zeros(2, dtype=torch.float32),
            "biome_label": torch.tensor(0, dtype=torch.long),
        }
        for _ in range(config.mil_steps)
    ]
    coord_stats = SimpleNamespace(lat_mean=0.0, lat_std=1.0, lon_mean=0.0, lon_std=1.0)
    fake_model = _FakeMILModel()
    fake_accelerator = _FakeAccelerator()
    optimizer = MagicMock()
    scheduler = MagicMock()
    scheduler.get_last_lr.return_value = [1e-3]
    tracked_windows: list[_TrackingDeque] = []
    eval_haversines = iter((11.0, 12.0, 13.0))

    def _make_tracking_deque(*, maxlen: int | None = None) -> _TrackingDeque:
        """Capture the plateau deque created by the trainer under test."""

        window = _TrackingDeque(maxlen=maxlen)
        tracked_windows.append(window)
        return window

    def _fake_run_mil_evaluation(**_kwargs) -> tuple[float, float, float, dict[str, float]]:
        """Return deterministic eval metrics at each configured eval boundary."""

        haversine = next(eval_haversines)
        return 0.0, haversine, 0.0, {"haversine_km_median": haversine, "macro_f1": 0.0}

    with (
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.load_mil_finetune_config", return_value=config
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.build_mil_fold_dataloaders",
            return_value=(train_loader, [], coord_stats),
        ),
        patch("jaguar_geo_assign.fine_tune.mil_trainer.Accelerator", return_value=fake_accelerator),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer.JaguarPositionalMILNetwork",
            return_value=fake_model,
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._build_mil_optimizer_and_scheduler",
            return_value=(optimizer, scheduler),
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._compute_baselines",
            return_value={"macro_f1": 0.0, "haversine_km_median": 0.0, "haversine_km_mean": 0.0},
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._compute_mtl_loss",
            return_value=(torch.tensor(1.5), torch.tensor(0.5), torch.tensor(1.0)),
        ),
        patch(
            "jaguar_geo_assign.fine_tune.mil_trainer._run_mil_evaluation",
            side_effect=_fake_run_mil_evaluation,
        ),
        patch("jaguar_geo_assign.fine_tune.mil_trainer.deque", side_effect=_make_tracking_deque),
    ):
        result = run_jaguar_mil_training(tmp_path / "config.toml")

    assert result.steps_completed == 300
    assert len(tracked_windows) == 1
    assert len(tracked_windows[0]) == 3
    assert tracked_windows[0].items == [11.0, 12.0, 13.0]


def test_run_mil_evaluation_handles_coordinate_only_batches(tmp_path: Path) -> None:
    """Coordinate-only MIL evaluation must tolerate missing biome labels."""

    shard_path = tmp_path / "ind-0.pt"
    torch.save(
        {
            "embeddings": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "bp_positions": torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
            "contigs": ["chr1", "chr1", "chr1"],
        },
        shard_path,
    )
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest_record(
        manifest_path,
        shard_path=shard_path,
        individual_id="ind-0",
        latitude=10.0,
        longitude=20.0,
    )

    coord_stats = CoordStats(lat_mean=10.0, lat_std=1.0, lon_mean=20.0, lon_std=1.0)
    dataset = MILBagDataset(
        manifest_path=manifest_path,
        individual_ids=["ind-0"],
        coord_stats=coord_stats,
        biome_to_idx=None,
    )
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)
    model = JaguarPositionalMILNetwork(
        embedding_dim=4,
        hidden_dim=2,
        num_biomes=None,
        dropout_prob=0.0,
        locus_dropout=0.0,
        genome_scale=1.0,
    )

    mean_eval_loss, best_eval_haversine_km, best_eval_macro_f1, metrics = _run_mil_evaluation(
        model=model,
        eval_loader=eval_loader,
        accelerator=_FakeAccelerator(),
        coord_stats=coord_stats,
        config=SimpleNamespace(n_biomes=5, fold_index=0),
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        huber_delta=1.0,
        global_step=1,
        best_eval_haversine_km=None,
        best_eval_macro_f1=None,
        output_dir=tmp_path / "out",
    )

    assert math.isfinite(mean_eval_loss)
    assert best_eval_haversine_km is None
    assert best_eval_macro_f1 is None
    assert math.isnan(metrics["accuracy"])
    assert math.isnan(metrics["macro_f1"])
    assert math.isfinite(metrics["haversine_km_median"])
