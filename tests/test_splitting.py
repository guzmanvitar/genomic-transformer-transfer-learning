import pytest

from jaguar_geo_assign.data.preprocessor import (
    PreprocessingConfig,
    PreparedSequence,
    SplitLeakageError,
    WindowRecord,
    assert_split_safety,
    build_split_manifest,
    window_sequences,
)


def test_window_sequences_reuse_global_locus_split_across_individuals_and_baseline() -> None:
    config = PreprocessingConfig(
        min_sequence_length=8,
        max_ambiguity_fraction=0.5,
        window_size=4,
        window_stride=2,
        locus_block_size=8,
    )
    prepared = [
        PreparedSequence("sample-1", "cat-1", "chr1", "consensus", 0, "ACGTACGT", 0.5, 0.0),
        PreparedSequence("sample-2", "cat-2", "chr1", "reference", 0, "ACGTACGT", 0.5, 0.0),
    ]

    windows = window_sequences(prepared, config)
    manifest = build_split_manifest(windows)

    assert {window.locus_id for window in windows} == {"chr1:0-8"}
    assert len({window.split for window in windows}) == 1
    assert len(manifest) == 1
    assert manifest[0].split == windows[0].split


def test_window_sequences_apply_window_ambiguity_filter_to_consensus_and_baseline() -> None:
    config = PreprocessingConfig(
        min_sequence_length=8,
        max_ambiguity_fraction=0.25,
        window_size=4,
        window_stride=4,
        locus_block_size=8,
    )
    prepared = [
        PreparedSequence("sample-1", "cat-1", "chr1", "consensus", 0, "ACGTNNAC", 0.5, 0.25),
        PreparedSequence("sample-2", "cat-2", "chr1", "reference", 0, "ACGTNNAC", 0.5, 0.25),
    ]

    windows = window_sequences(prepared, config)

    assert len(windows) == 2
    assert {(window.source, window.window_start, window.window_end) for window in windows} == {
        ("consensus", 0, 4),
        ("reference", 0, 4),
    }
    assert len({window.split for window in windows}) == 1


def test_assert_split_safety_rejects_cross_split_overlap() -> None:
    train_window = WindowRecord(
        sample_id="sample-1",
        individual_id="cat-1",
        contig="chr1",
        source="consensus",
        split="train",
        locus_id="chr1:0-8",
        block_start=0,
        block_end=8,
        window_start=0,
        window_end=6,
        sequence="ACGTAC",
        gc_fraction=0.5,
        ambiguity_fraction=0.0,
        sequence_hash="train",
    )
    validation_window = WindowRecord(
        sample_id="sample-2",
        individual_id="cat-2",
        contig="chr1",
        source="reference",
        split="validation",
        locus_id="chr1:8-16",
        block_start=8,
        block_end=16,
        window_start=4,
        window_end=10,
        sequence="GTACGT",
        gc_fraction=0.5,
        ambiguity_fraction=0.0,
        sequence_hash="validation",
    )

    with pytest.raises(SplitLeakageError, match="Overlapping windows"):
        assert_split_safety((train_window, validation_window))