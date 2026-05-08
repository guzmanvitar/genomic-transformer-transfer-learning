"""Pretraining stage — felid foundation corpus construction and acquisition.

Public surface for the active pretraining workflow:

* **Felid foundation pipeline** (``run_felid_foundation_pretrain``):
  multi-species reference-FASTA-only pretraining corpus generation for the
  six approved felid assemblies.
* **Felid acquisition** (``acquire_felid_foundation_assemblies``):
  integrity-checked download helpers for the approved reference FASTAs.
"""

from ..data.felid_acquisition import (
    FelidAcquisitionError,
    FelidAcquisitionSummary,
    acquire_felid_foundation_assemblies,
)
from .felid_foundation_pipeline import (
    FelidFoundationPretrainRunResult,
    MissingFelidReferenceError,
    format_felid_foundation_pretrain_result,
    run_felid_foundation_pretrain,
)

__all__ = [
    # Felid foundation (reference) pipeline
    "FelidFoundationPretrainRunResult",
    "format_felid_foundation_pretrain_result",
    "run_felid_foundation_pretrain",
    "MissingFelidReferenceError",
    # Felid acquisition
    "FelidAcquisitionSummary",
    "FelidAcquisitionError",
    "acquire_felid_foundation_assemblies",
]
