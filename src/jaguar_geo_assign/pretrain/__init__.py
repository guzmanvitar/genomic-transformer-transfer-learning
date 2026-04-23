"""Pretraining stage — feline and felid foundation corpus construction.

Public surface for the implemented pretraining data pipelines:

* **Feline pipeline** (``run_feline_pretrain_pipeline``): VCF + consensus
  FASTA construction via ``bcftools`` for jaguar geographic assignment.
* **Felid foundation pipeline** (``run_felid_foundation_pretrain``):
  Multi-species reference-FASTA-only pretraining corpus for the six
  approved felid assemblies.

Both pipelines read validated TOML configs, preprocess and tokenize
sequences into DNABERT-2-ready windows, and write Parquet corpora and
run-summary artifacts.
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
from .pipeline import (
    FelinePretrainRunResult,
    format_feline_pretrain_result,
    run_feline_pretrain_pipeline,
)

__all__ = [
    # Feline (consensus) pipeline
    "FelinePretrainRunResult",
    "format_feline_pretrain_result",
    "run_feline_pretrain_pipeline",
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
