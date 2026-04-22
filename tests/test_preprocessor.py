"""Tests for sequence preprocessing, normalization, and window accounting.

These tests protect the contracts that short or overly ambiguous sequences
are filtered with auditable reasons, IUPAC ambiguity codes are deterministically
masked to ``N``, mask-span provenance (filtered / no_call / heterozygous)
is preserved and deduplicated across overlapping spans, and windowing
accurately re-accounts both declared masks and realized ``N`` coverage.
Together they prevent silent data contamination and maintain traceability
for downstream model-quality diagnostics.
"""

import pytest

from jaguar_geo_assign.data.preprocessor import (
    PreparedSequence,
    PreprocessingConfig,
    PreprocessingError,
    SequenceRecord,
    normalize_sequence,
    prepare_sequences,
    window_sequences,
)


def test_prepare_sequences_filters_short_and_ambiguous_records() -> None:
    """Records below length or above ambiguity thresholds are filtered with labeled reasons."""
    config = PreprocessingConfig(
        min_sequence_length=6,
        max_ambiguity_fraction=0.34,
        window_size=4,
        window_stride=2,
        locus_block_size=8,
    )
    report = prepare_sequences(
        [
            SequenceRecord("sample-1", "cat-1", "chr1", "acgtnn"),
            SequenceRecord("sample-2", "cat-2", "chr1", "ACG"),
            SequenceRecord("sample-3", "cat-3", "chr1", "NNNNTA"),
        ],
        config,
    )

    assert len(report.retained) == 1
    assert report.retained[0].sequence == "ACGTNN"
    assert report.retained[0].gc_fraction == 0.5
    assert report.retained[0].ambiguity_fraction == 2 / 6
    assert {item.reason for item in report.filtered} == {"short_sequence", "high_ambiguity"}


def test_normalize_sequence_masks_iupac_codes_deterministically() -> None:
    """IUPAC ambiguity codes and gaps collapse to ``N`` under the default masking policy."""
    assert normalize_sequence("acgtRysw?-") == "ACGTNNNNNN"


def test_normalize_sequence_rejects_unsupported_bases_when_requested() -> None:
    """In ``reject`` ambiguity mode, unsupported bases raise instead of being silently masked."""
    config = PreprocessingConfig(
        min_sequence_length=4,
        max_ambiguity_fraction=1.0,
        window_size=4,
        window_stride=4,
        locus_block_size=4,
        ambiguity_mode="reject",
    )

    try:
        prepare_sequences([SequenceRecord("sample-1", "cat-1", "chr1", "ACGR")], config)
    except ValueError as exc:
        assert "Unsupported base" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("expected preprocessing to reject unsupported bases")


def test_window_sequences_preserve_mask_provenance_counts() -> None:
    """Each window reports per-reason mask counts that match the originating mask spans."""
    config = PreprocessingConfig(
        min_sequence_length=8,
        max_ambiguity_fraction=1.0,
        window_size=4,
        window_stride=4,
        locus_block_size=8,
    )

    report = prepare_sequences(
        [
            SequenceRecord(
                "sample-1",
                "cat-1",
                "chr1",
                "ACGTNNNN",
                mask_spans=((4, 5, "filtered"), (5, 6, "no_call"), (6, 8, "heterozygous")),
            )
        ],
        config,
    )

    windows = window_sequences(list(report.retained), config)

    assert len(windows) == 2
    assert windows[0].unique_masked_bases == 0
    assert windows[0].filtered_bases == 0
    assert windows[0].no_call_bases == 0
    assert windows[0].other_masked_bases == 0
    assert windows[1].unique_masked_bases == 4
    assert windows[1].filtered_bases == 1
    assert windows[1].no_call_bases == 1
    assert windows[1].other_masked_bases == 2
    assert windows[1].masked_base_counts == (("filtered", 1), ("heterozygous", 2), ("no_call", 1))


def test_window_sequences_track_unique_masked_bases_across_overlapping_spans() -> None:
    """Overlapping spans with different reasons don't double-count unique masked positions."""
    config = PreprocessingConfig(
        min_sequence_length=8,
        max_ambiguity_fraction=1.0,
        window_size=4,
        window_stride=4,
        locus_block_size=8,
    )

    report = prepare_sequences(
        [
            SequenceRecord(
                "sample-1",
                "cat-1",
                "chr1",
                "ACGTNNNN",
                mask_spans=((4, 8, "filtered"), (4, 8, "no_call")),
            )
        ],
        config,
    )

    windows = window_sequences(list(report.retained), config)

    assert windows[1].unique_masked_bases == 4
    assert windows[1].filtered_bases == 4
    assert windows[1].no_call_bases == 4
    assert windows[1].masked_base_counts == (("filtered", 4), ("no_call", 4))


def test_window_sequences_count_realized_n_coverage_without_mask_spans() -> None:
    """Reference inputs without declared mask spans still report ``N`` bases as unique masked coverage."""
    config = PreprocessingConfig(
        min_sequence_length=6,
        max_ambiguity_fraction=1.0,
        window_size=6,
        window_stride=6,
        locus_block_size=6,
    )

    report = prepare_sequences(
        [SequenceRecord("ref-1", "reference", "chr1", "ACGNAA", source="reference")],
        config,
    )

    windows = window_sequences(list(report.retained), config)

    assert len(windows) == 1
    assert windows[0].sequence == "ACGNAA"
    assert windows[0].unique_masked_bases == 1
    assert windows[0].filtered_bases == 0
    assert windows[0].no_call_bases == 0
    assert windows[0].other_masked_bases == 0
    assert windows[0].masked_base_counts == ()


def test_window_sequences_preserve_consensus_provenance_without_sequence_fallback() -> None:
    """Consensus windows report span-derived masked coverage verbatim so silent ``N`` bases remain auditable.

    The ``source="reference"`` fallback that folds in ``sequence.count("N")``
    must not apply to consensus-sourced windows; otherwise a consensus
    record with an ``N`` base unaccounted for by any mask span would be
    silently reconciled and the downstream coverage invariant could no
    longer catch corrupt VCF→FASTA provenance.
    """
    config = PreprocessingConfig(
        min_sequence_length=6,
        max_ambiguity_fraction=1.0,
        window_size=6,
        window_stride=6,
        locus_block_size=6,
    )

    report = prepare_sequences(
        [SequenceRecord("cat-1", "cat-1", "chr1", "ACGNAA", source="consensus")],
        config,
    )

    windows = window_sequences(list(report.retained), config)

    assert len(windows) == 1
    assert windows[0].sequence == "ACGNAA"
    assert windows[0].source == "consensus"
    assert windows[0].unique_masked_bases == 0
    assert windows[0].masked_base_counts == ()


def test_window_sequences_report_consensus_span_coverage_when_spans_declared() -> None:
    """Consensus windows with faithful mask spans report the span-derived unique coverage."""
    config = PreprocessingConfig(
        min_sequence_length=6,
        max_ambiguity_fraction=1.0,
        window_size=6,
        window_stride=6,
        locus_block_size=6,
    )

    report = prepare_sequences(
        [
            SequenceRecord(
                "cat-1",
                "cat-1",
                "chr1",
                "ACGNAA",
                source="consensus",
                mask_spans=((3, 4, "no_call"),),
            )
        ],
        config,
    )

    windows = window_sequences(list(report.retained), config)

    assert len(windows) == 1
    assert windows[0].sequence == "ACGNAA"
    assert windows[0].unique_masked_bases == 1
    assert windows[0].no_call_bases == 1
    assert windows[0].masked_base_counts == (("no_call", 1),)


def test_window_sequences_drop_high_ambiguity_windows_from_retained_sequence() -> None:
    """A retained sequence can still shed individual windows that exceed the per-window ambiguity cap."""
    config = PreprocessingConfig(
        min_sequence_length=8,
        max_ambiguity_fraction=0.25,
        window_size=4,
        window_stride=4,
        locus_block_size=8,
    )

    report = prepare_sequences(
        [SequenceRecord("sample-1", "cat-1", "chr1", "ACGTNNAC")],
        config,
    )

    assert len(report.retained) == 1
    assert report.retained[0].ambiguity_fraction == 0.25

    windows = window_sequences(list(report.retained), config)

    assert len(windows) == 1
    assert windows[0].window_start == 0
    assert windows[0].window_end == 4
    assert windows[0].sequence == "ACGT"
    assert windows[0].ambiguity_fraction == 0.0


def test_window_sequences_reference_source_with_intrinsic_n_uses_realized_coverage() -> None:
    """Regression: approved reference windows with intrinsic ``N`` fall back to realized coverage.

    Wave 6b approval confirmed that ``source="reference"`` is the only
    approved non-consensus label allowed to use realized ``N`` fallback
    for windows lacking declared mask spans.
    """
    config = PreprocessingConfig(
        min_sequence_length=6,
        max_ambiguity_fraction=1.0,
        window_size=6,
        window_stride=6,
        locus_block_size=6,
    )

    report = prepare_sequences(
        [SequenceRecord("ref-1", "reference", "chr1", "ACNNAA", source="reference")],
        config,
    )

    windows = window_sequences(list(report.retained), config)

    assert len(windows) == 1
    assert windows[0].sequence == "ACNNAA"
    assert windows[0].source == "reference"
    assert windows[0].unique_masked_bases == 2
    assert windows[0].masked_base_counts == ()


def test_prepare_sequences_rejects_unknown_source_label_on_retained_record_shape() -> None:
    """Regression: retained-shape records with unknown source labels fail loudly at ingress.

    Wave 6b approval mandated that the approved producer-side source set
    is exactly ``{"consensus", "reference"}``; any other label must raise
    a fail-fast contract error before filtering or fallback logic can
    swallow it. Wave 6c moved the validation to ``prepare_sequences`` so
    even records that would otherwise be retained now fail at the earliest
    ingress point. This prevents silent data-quality degradation from
    typos or unapproved labels.
    """
    config = PreprocessingConfig(
        min_sequence_length=6,
        max_ambiguity_fraction=1.0,
        window_size=6,
        window_stride=6,
        locus_block_size=6,
    )

    with pytest.raises(PreprocessingError, match=r"Unknown source label 'consensus_typo'"):
        prepare_sequences(
            [SequenceRecord("cat-1", "cat-1", "chr1", "ACGNAA", source="consensus_typo")],
            config,
        )


def test_prepare_sequences_rejects_unknown_source_label_on_short_sequence_record() -> None:
    """Regression: malformed source labels raise even when the record would be short-filtered.

    A short-sequence record with an unapproved source must not be allowed
    to disappear into ``PreprocessingReport.filtered`` as a ``"short_sequence"``
    rejection; the provenance defect is the primary failure and must be
    surfaced by ``prepare_sequences`` before any length filter runs.
    """
    config = PreprocessingConfig(
        min_sequence_length=10,
        max_ambiguity_fraction=1.0,
        window_size=4,
        window_stride=4,
        locus_block_size=8,
    )

    with pytest.raises(PreprocessingError, match=r"Unknown source label 'consensus_typo'"):
        prepare_sequences(
            [SequenceRecord("cat-1", "cat-1", "chr1", "ACGT", source="consensus_typo")],
            config,
        )


def test_prepare_sequences_rejects_unknown_source_label_on_high_ambiguity_record() -> None:
    """Regression: malformed source labels raise even when the record would be ambiguity-filtered.

    A high-ambiguity record with an unapproved source must not be allowed
    to disappear into ``PreprocessingReport.filtered`` as a
    ``"high_ambiguity"`` rejection; the provenance defect is the primary
    failure and must be surfaced by ``prepare_sequences`` before the
    ambiguity filter runs.
    """
    config = PreprocessingConfig(
        min_sequence_length=4,
        max_ambiguity_fraction=0.1,
        window_size=4,
        window_stride=4,
        locus_block_size=8,
    )

    with pytest.raises(PreprocessingError, match=r"Unknown source label 'consensus_typo'"):
        prepare_sequences(
            [SequenceRecord("cat-1", "cat-1", "chr1", "NNNN", source="consensus_typo")],
            config,
        )


def test_window_sequences_rejects_unknown_source_label_on_direct_entry() -> None:
    """Regression: direct ``PreparedSequence`` callers cannot bypass source validation.

    Wave 6d closes the remaining direct-entry provenance bypass: callers
    that construct ``PreparedSequence`` values themselves (instead of going
    through ``prepare_sequences``) must still be rejected by
    ``window_sequences`` if the ``source`` label lies outside the approved
    producer set ``{"consensus", "reference"}``. The check runs before any
    per-block or per-window filter so the failure is loud even for inputs
    that would otherwise yield windows.
    """
    config = PreprocessingConfig(
        min_sequence_length=6,
        max_ambiguity_fraction=0.5,
        window_size=6,
        window_stride=6,
        locus_block_size=6,
    )
    prepared = [
        PreparedSequence("cat-1", "cat-1", "chr1", "consensus_typo", 0, "ACGTAC", 0.5, 0.0),
    ]

    with pytest.raises(PreprocessingError, match=r"Unknown source label 'consensus_typo'"):
        window_sequences(prepared, config)


def test_window_sequences_rejects_unknown_source_label_when_all_windows_would_be_length_filtered() -> None:
    """Regression: invalid direct-entry source raises even when the block overlap is too short for any window.

    Without the up-front source validation, a ``PreparedSequence`` shorter
    than ``window_size`` would cause ``window_sequences`` to silently
    return an empty tuple and the malformed provenance would disappear as
    a missing-record statistic. The contract requires the provenance
    defect to surface as the primary failure.
    """
    config = PreprocessingConfig(
        min_sequence_length=2,
        max_ambiguity_fraction=1.0,
        window_size=6,
        window_stride=6,
        locus_block_size=6,
    )
    prepared = [
        PreparedSequence("cat-1", "cat-1", "chr1", "consensus_typo", 0, "ACG", 0.5, 0.0),
    ]

    with pytest.raises(PreprocessingError, match=r"Unknown source label 'consensus_typo'"):
        window_sequences(prepared, config)


def test_window_sequences_rejects_unknown_source_label_when_all_windows_would_be_ambiguity_filtered() -> None:
    """Regression: invalid direct-entry source raises even when every candidate window exceeds the ambiguity cap.

    A high-``N`` direct-entry sequence whose only candidate window would
    be dropped by the per-window ambiguity filter must not let an
    unapproved ``source`` label slip through as an empty-windows result;
    the provenance defect must fail loudly before the filter runs.
    """
    config = PreprocessingConfig(
        min_sequence_length=6,
        max_ambiguity_fraction=0.25,
        window_size=6,
        window_stride=6,
        locus_block_size=6,
    )
    prepared = [
        PreparedSequence("cat-1", "cat-1", "chr1", "consensus_typo", 0, "NNNNNN", 0.0, 1.0),
    ]

    with pytest.raises(PreprocessingError, match=r"Unknown source label 'consensus_typo'"):
        window_sequences(prepared, config)
