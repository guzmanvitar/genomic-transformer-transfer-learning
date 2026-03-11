from jaguar_geo_assign.data.preprocessor import (
    PreprocessingConfig,
    SequenceRecord,
    normalize_sequence,
    prepare_sequences,
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