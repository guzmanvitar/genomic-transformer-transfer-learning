"""Reporting layer — genomics diagnostics and EDA payload generation.

Exposes helpers that summarize tokenized corpus records, audit corpus
integrity, compare consensus-derived diagnostics against a reference-only
baseline, build missingness heatmaps, and serialise structured EDA payloads
consumed by the interactive ``notebooks/eda_genomics.py`` workflow.
"""

from .genomics_diagnostics import (
    audit_corpus_integrity,
    build_eda_payload,
    build_missingness_heatmap,
    compare_reference_baseline,
    summarize_corpus_records,
    summarize_sample_records,
    write_eda_payload_json,
)

__all__ = [
    "audit_corpus_integrity",
    "build_eda_payload",
    "build_missingness_heatmap",
    "compare_reference_baseline",
    "summarize_corpus_records",
    "summarize_sample_records",
    "write_eda_payload_json",
]
