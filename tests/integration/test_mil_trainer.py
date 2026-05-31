"""Integration smoke tests for the MIL trainer path."""

from __future__ import annotations

from jaguar_geo_assign.fine_tune.mil_trainer import integration_test


def test_mil_integration_test_full_bag_smoke() -> None:
    """The MIL smoke path should handle a production-scale bag length on CPU.

    The embedding width stays intentionally small so CI validates the full-bag
    O(N) path without paying the cost of the real 768-wide DNABERT-2 features.
    This explicitly disables mixed precision so the smoke path remains valid on
    CPUs and GPUs that do not expose bfloat16 kernels.
    """

    integration_test(bag_size=84_000, embedding_dim=16, hidden_dim=8, mixed_precision="no")
