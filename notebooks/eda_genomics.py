"""Canonical VS Code interactive entry point for genomics EDA diagnostics.

This module provides a VS Code Interactive Python (``#%%`` cells) workflow
for generating and inspecting synthetic genomic-record payloads.  It
delegates all analytical logic to :mod:`jaguar_geo_assign.reporting` and
focuses on constructing realistic sample records, running the helper-backed
EDA pipeline, and pretty-printing key diagnostic sections.
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

from jaguar_geo_assign.reporting import build_eda_payload, write_eda_payload_json


# %% [markdown]
# # Genomics EDA diagnostics
#
# This is the single maintained entry point for the helper-backed genomics EDA
# workflow. It intentionally reuses `jaguar_geo_assign.reporting` helpers instead
# of re-implementing diagnostics logic inline.


# %% Configuration
CONSENSUS_RECORD_TOTAL = 96
BASELINE_RECORD_TOTAL = 48
NEAR_DUPLICATE_SAMPLE_LIMIT = 64
REPORT_PATH = Path("reports/generated/feline_pretrain/diagnostics_sample_payload.json")


# %% Sample data builder
def build_realistic_records(*, total: int, source: str) -> list[dict[str, object]]:
    """Build a list of synthetic genomic records for EDA diagnostics.

    Each record mimics a real consensus/reference alignment row with
    deterministic sequence content so that downstream diagnostics
    (variant counts, callable-base ratios, near-duplicate detection)
    produce reproducible, interpretable results.

    For ``source="consensus"`` records, a simple rotation pattern
    introduces SNP-like and no-call mutations every four records,
    ensuring the payload exercises non-trivial variant-counting paths.

    Args:
        total: Number of records to generate.
        source: Label applied to every record (e.g. ``"consensus"`` or
            ``"reference"``).  Also controls whether deterministic
            mutations are injected into the sequences.

    Returns:
        A list of dictionaries, each representing one genomic record
        with keys such as ``sample_id``, ``locus_id``, ``sequence``,
        ``variant_count``, and ``callable_bases``.
    """
    records: list[dict[str, object]] = []
    for index in range(total):
        length = 32 if index % 2 == 0 else 36
        reference_sequence = ("ACGT" * ((length // 4) + 1))[:length]
        sequence = reference_sequence
        if source == "consensus":
            pattern = index % 4
            if pattern == 2:
                sequence = "T" + reference_sequence[1:]
            elif pattern == 3:
                sequence = reference_sequence[:-1] + "N"
        records.append(
            {
                "sample_id": f"{source}-{index}",
                "locus_id": f"chr{1 + (index % 3)}:block-{index}",
                "split": "train" if index % 5 else "validation",
                "source": source,
                "sequence": sequence,
                "reference_sequence": reference_sequence,
                "variant_count": 0 if sequence == reference_sequence else 1,
                "callable_bases": len(sequence) - sequence.count("N"),
                "unique_masked_bases": sequence.count("N"),
                "filtered_bases": 0,
                "no_call_bases": sequence.count("N"),
                "token_count": max(1, len(sequence) // 4),
            }
        )
    return records


# %% Helper-backed payload generation
def run_workflow() -> tuple[dict[str, object], Path]:
    consensus_records = build_realistic_records(total=CONSENSUS_RECORD_TOTAL, source="consensus")
    baseline_records = build_realistic_records(total=BASELINE_RECORD_TOTAL, source="reference")
    payload = build_eda_payload(
        consensus_records,
        baseline_records,
        near_duplicate_sample_limit=NEAR_DUPLICATE_SAMPLE_LIMIT,
    )
    report_path = write_eda_payload_json(
        consensus_records,
        baseline_records,
        REPORT_PATH,
        near_duplicate_sample_limit=NEAR_DUPLICATE_SAMPLE_LIMIT,
    )
    return payload, report_path


# %% Execute workflow
payload, report_path = run_workflow()


# %% Inspect key diagnostics
pprint(payload["consensus_sample_overview"])
pprint(payload["consensus_corpus"])
pprint(payload["baseline_comparison"])
print(f"Report written to {report_path}")