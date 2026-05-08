"""Unit tests for the jaguar multi-task fine-tuning trainer helpers.

These tests focus on small, deterministic trainer contracts that are cheaper to
exercise than the synthetic end-to-end integration harness.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from jaguar_geo_assign.fine_tune.trainer import _compute_mtl_loss


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
