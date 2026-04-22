"""Pretraining stage — feline corpus construction and export pipeline.

Public surface for the implemented feline pretraining data pipeline.  The
pipeline reads a validated TOML config, generates per-sample consensus
FASTAs via ``bcftools``, preprocesses and tokenizes the resulting sequences
into DNABERT-2-ready windows, and writes diagnostics and run-summary
artifacts.
"""

from .pipeline import FelinePretrainRunResult, format_feline_pretrain_result, run_feline_pretrain_pipeline

__all__ = [
    "FelinePretrainRunResult",
    "format_feline_pretrain_result",
    "run_feline_pretrain_pipeline",
]
