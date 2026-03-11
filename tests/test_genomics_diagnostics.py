import json

import pytest

from jaguar_geo_assign.reporting import (
    audit_corpus_integrity,
    build_eda_payload,
    build_missingness_heatmap,
    compare_reference_baseline,
    summarize_corpus_records,
    summarize_sample_records,
    write_eda_payload_json,
)


def _build_realistic_records(*, total: int, source: str) -> list[dict[str, object]]:
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
                "filtered_bases": 0,
                "no_call_bases": sequence.count("N"),
                "token_count": max(1, len(sequence) // 4),
            }
        )
    return records


@pytest.fixture
def consensus_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "cat-1",
            "locus_id": "chr1:block-1",
            "split": "train",
            "source": "consensus",
            "sequence": "ACGTNCGT",
            "reference_sequence": "ACGTACGT",
            "variant_count": 1,
            "callable_bases": 7,
            "filtered_bases": 0,
            "no_call_bases": 1,
            "token_count": 4,
        },
        {
            "sample_id": "cat-2",
            "locus_id": "chr1:block-2",
            "split": "train",
            "source": "consensus",
            "sequence": "ACGTTCGT",
            "reference_sequence": "ACGTACGT",
            "variant_count": 1,
            "callable_bases": 8,
            "filtered_bases": 0,
            "no_call_bases": 0,
            "token_count": 4,
        },
        {
            "sample_id": "cat-3",
            "locus_id": "chr1:block-3",
            "split": "validation",
            "source": "consensus",
            "sequence": "ACGTTCGT",
            "reference_sequence": "ACGTACGT",
            "variant_count": 1,
            "callable_bases": 8,
            "filtered_bases": 0,
            "no_call_bases": 0,
            "token_count": 4,
        },
    ]


@pytest.fixture
def baseline_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "ref-1",
            "locus_id": "chr1:block-1",
            "split": "train",
            "source": "reference",
            "sequence": "ACGTACGT",
            "reference_sequence": "ACGTACGT",
            "variant_count": 0,
            "callable_bases": 8,
            "filtered_bases": 0,
            "no_call_bases": 0,
            "token_count": 4,
        },
        {
            "sample_id": "ref-2",
            "locus_id": "chr1:block-2",
            "split": "train",
            "source": "reference",
            "sequence": "ACGTACGA",
            "reference_sequence": "ACGTACGA",
            "variant_count": 0,
            "callable_bases": 8,
            "filtered_bases": 0,
            "no_call_bases": 0,
            "token_count": 4,
        },
    ]


def test_summarize_sample_records_reports_observability_metrics(consensus_records) -> None:
    summary = summarize_sample_records(consensus_records[:1])[0]

    assert summary["sample_id"] == "cat-1"
    assert summary["gc_fraction"] == 0.571429
    assert summary["ambiguity_fraction"] == 0.125
    assert summary["callable_fraction"] == 0.875
    assert summary["no_call_fraction"] == 0.125
    assert summary["fraction_identical_to_reference"] == 0.875
    assert summary["token_to_base_ratio"] == 0.5


def test_missingness_heatmap_tracks_n_burden_by_relative_position(consensus_records) -> None:
    heatmap = build_missingness_heatmap(consensus_records[:1], bins=4)

    assert heatmap == [0.0, 0.0, 0.5, 0.0]


def test_summarize_corpus_records_tracks_duplicates_and_near_duplicates(consensus_records) -> None:
    summary = summarize_corpus_records(consensus_records)

    assert summary["retained_window_count"] == 3
    assert summary["length_distribution"] == [{"length": 8, "count": 3}]
    assert summary["duplicate_window_count"] == 1
    assert summary["duplicate_window_fraction"] == pytest.approx(1 / 3)
    assert summary["near_duplicate_pair_count"] == 2
    assert summary["near_duplicate_pair_fraction"] == pytest.approx(2 / 3)
    assert summary["near_duplicate_analysis"] == {
        "mode": "exact",
        "total_sequence_count": 3,
        "analyzed_sequence_count": 3,
        "sample_limit": 512,
    }
    assert sum(bucket["count"] for bucket in summary["gc_fraction_distribution"]) == 3
    assert summary["source_counts"] == {"consensus": 3}


def test_compare_reference_baseline_reports_corpus_level_deltas(
    consensus_records, baseline_records
) -> None:
    comparison = compare_reference_baseline(consensus_records, baseline_records)

    assert comparison["deltas"]["retained_window_count"] == 1
    assert comparison["deltas"]["ambiguity_fraction"] == 0.041667
    assert comparison["deltas"]["variant_fraction"] == 0.125
    assert comparison["deltas"]["duplicate_window_fraction"] == pytest.approx(1 / 3)
    assert comparison["deltas"]["near_duplicate_pair_fraction"] == pytest.approx(-1 / 3)
    assert comparison["deltas"]["token_to_base_ratio"] == 0.0


def test_audit_corpus_integrity_flags_split_conflicts_and_shape_issues(consensus_records) -> None:
    issues = audit_corpus_integrity(
        [
            consensus_records[0],
            {
                **consensus_records[1],
                "sample_id": "cat-4",
                "split": "validation",
                "locus_id": consensus_records[0]["locus_id"],
                "token_count": 0,
            },
        ]
    )

    assert issues["split_conflicts"] == [
        {"locus_id": "chr1:block-1", "splits": ["train", "validation"]}
    ]
    assert issues["shape_issues"] == [
        {
            "sample_id": "cat-4",
            "locus_id": "chr1:block-1",
            "issue": "token_count_must_be_positive",
        }
    ]


def test_build_eda_payload_collects_sample_and_baseline_views(
    consensus_records, baseline_records
) -> None:
    payload = build_eda_payload(consensus_records, baseline_records)

    assert len(payload["consensus_samples"]) == 3
    assert payload["consensus_corpus"]["retained_window_count"] == 3
    assert payload["consensus_corpus"]["near_duplicate_analysis"]["mode"] == "exact"
    assert payload["baseline_corpus"]["retained_window_count"] == 2
    assert payload["baseline_comparison"]["deltas"]["retained_window_count"] == 1


def test_summarize_sample_records_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="required fields"):
        summarize_sample_records([{"sample_id": "cat-1", "sequence": "ACGT"}])


def test_summarize_corpus_records_samples_near_duplicate_analysis_for_large_corpus() -> None:
    realistic_consensus = _build_realistic_records(total=640, source="consensus")

    summary = summarize_corpus_records(realistic_consensus, near_duplicate_sample_limit=64)

    assert summary["retained_window_count"] == 640
    assert summary["duplicate_window_count"] > 0
    assert summary["near_duplicate_pair_count"] > 0
    assert summary["near_duplicate_pair_fraction"] > 0.0
    assert summary["near_duplicate_analysis"] == {
        "mode": "sampled",
        "total_sequence_count": 640,
        "analyzed_sequence_count": 64,
        "sample_limit": 64,
    }
    assert summary["length_distribution"] == [
        {"length": 32, "count": 320},
        {"length": 36, "count": 320},
    ]
    assert sum(bucket["count"] for bucket in summary["gc_fraction_distribution"]) == 640


def test_write_eda_payload_json_persists_report_payload(tmp_path, consensus_records, baseline_records) -> None:
    output_path = tmp_path / "reports" / "diagnostics_payload.json"

    written_path = write_eda_payload_json(consensus_records, baseline_records, output_path)

    assert written_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["consensus_corpus"]["retained_window_count"] == 3
    assert payload["consensus_corpus"]["near_duplicate_analysis"]["mode"] == "exact"
    assert payload["baseline_comparison"]["deltas"]["near_duplicate_pair_fraction"] == pytest.approx(-1 / 3)