from jaguar_geo_assign.data.preprocessor import (
    PreprocessingConfig,
    SequenceRecord,
    normalize_sequence,
    prepare_sequences,
    window_sequences,
)


def test_prepare_sequences_filters_short_and_ambiguous_records() -> None:
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
    assert normalize_sequence("acgtRysw?-") == "ACGTNNNNNN"


def test_normalize_sequence_rejects_unsupported_bases_when_requested() -> None:
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


def test_window_sequences_drop_high_ambiguity_windows_from_retained_sequence() -> None:
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