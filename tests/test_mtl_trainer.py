"""Integration tests for the jaguar DNABERT-2 MTL training helpers.

These tests exercise the synthetic integration harness implemented in
:func:`jaguar_geo_assign.fine_tune.trainer.integration_test` to guard
against regressions in the multi-task training stack.
"""

from __future__ import annotations

import pytest

from jaguar_geo_assign.fine_tune.trainer import integration_test


def test_integration_test_default_smoke() -> None:
    """Smoke test: integration_test(use_real_model=False) runs on CPU.

    This mirrors the foundation-training smoke test and ensures the MTL
    integration harness remains lightweight and free of accidental
    network calls in the default pytest selection.
    """

    # Use a small synthetic cohort to keep runtime comfortably under CI
    # time budgets while still exercising all integration assertions.
    integration_test(n_individuals=2, windows_per_individual=2, use_real_model=False)


@pytest.mark.integration
def test_integration_test_real_model() -> None:
    """Integration test: exercise the real DNABERT-2 backbone path.

    This test is gated behind the ``integration`` marker so it only
    runs when explicitly requested (e.g. ``pytest -m integration``).
    It validates that the same synthetic harness can drive a real
    DNABERT-2 backbone loaded from the Hugging Face Hub.
    """

    integration_test(n_individuals=1, windows_per_individual=1, use_real_model=True)
