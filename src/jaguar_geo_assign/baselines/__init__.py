"""Baseline comparison stage — deferred integration points.

Placeholder constants that reserve the baseline-evaluation stage name, its
default provider identity, and the shared extension point used by downstream
metric-reporting logic.  These values are referenced during config validation
to ensure that experiment TOMLs declare a recognised baseline contract even
before the baseline models are implemented.
"""

# Stage name registered in the ``stages.order`` list of experiment configs.
BASELINE_EVALUATION_STAGE = "baseline_evaluate"

# Provider identifier for the legacy group-centroid baseline (not yet wired).
DEFERRED_BASELINE_PROVIDER = "deferred_legacy_group_model"

# Extension point that the reporting layer will query once baselines are active.
SHARED_BASELINE_EXTENSION_POINT = "shared_split_metric_report_contract"
