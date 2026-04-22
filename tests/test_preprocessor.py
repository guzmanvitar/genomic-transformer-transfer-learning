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




# ---------------------------------------------------------------------------
# Streaming tokenized-corpus writer tests
# ---------------------------------------------------------------------------
#
# These tests cover the refactor from full-materialisation to an
# append-capable :class:`TokenizedCorpusWriter`. Intent: guard the RAM
# bound (multi-species pretraining never materialises the whole corpus)
# while keeping the legacy single-shot ``write_tokenized_dataset`` API
# binary-compatible via a thin one-batch shim.

from dataclasses import replace as _dataclass_replace
import json as _json

from jaguar_geo_assign.data.preprocessor import (
    ExportContract,
    TokenizedCorpusWriter,
    TokenizedWindow,
    TokenizerProvenance,
    WindowRecord,
    write_tokenized_dataset,
)


def _streaming_window(
    *,
    sample_id: str = "sample-1",
    individual_id: str = "cat-1",
    contig: str = "chr1",
    split: str = "train",
    locus_id: str = "chr1:0-8",
    block_start: int = 0,
    block_end: int = 8,
    window_start: int = 0,
    window_end: int = 6,
    sequence: str = "ACGTNN",
    sequence_hash: str = "hash-default",
) -> WindowRecord:
    """Build a minimal ``WindowRecord`` for streaming-writer tests.

    Intent: centralise the fixture boilerplate so each test expresses
    only the dimensions it actually varies (split, locus, contig,
    etc.), making the test's failure mode obvious from its body.
    """
    return WindowRecord(
        sample_id=sample_id,
        individual_id=individual_id,
        contig=contig,
        source="consensus",
        split=split,
        locus_id=locus_id,
        block_start=block_start,
        block_end=block_end,
        window_start=window_start,
        window_end=window_end,
        sequence=sequence,
        gc_fraction=0.5,
        ambiguity_fraction=2 / 6,
        sequence_hash=sequence_hash,
    )


def _streaming_tokenized(**overrides: object) -> TokenizedWindow:
    """Wrap ``_streaming_window`` in a ``TokenizedWindow`` with fixed provenance.

    Intent: every test in this suite exercises the writer, not the
    tokenizer, so the token arrays are intentionally trivial and the
    provenance is the canonical pinned default.
    """
    return TokenizedWindow(
        window=_dataclass_replace(_streaming_window(), **overrides),
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        token_count=3,
        token_to_base_ratio=0.5,
        tokenizer=TokenizerProvenance(),
    )


def _read_parquet_records(output_path, split: str) -> list[dict]:
    """Read every Parquet row under ``split=<split>/`` for set-equality checks.

    Intent: the streaming writer no longer guarantees a global
    within-split sort order, so tests must compare record sets
    without relying on row ordering. This helper collapses all
    Parquet files under a split into a flat list of row dicts.
    """
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    rows: list[dict] = []
    split_dir = output_path / f"split={split}"
    for parquet_path in sorted(split_dir.rglob("*.parquet")):
        table = pyarrow_parquet.read_table(parquet_path)
        rows.extend(table.to_pylist())
    return rows


def _primary_key(row: dict) -> tuple:
    """Stable primary key over the fields that uniquely identify a window.

    Intent: set-equality across two Parquet datasets must not be
    confused by dict-ordering or list vs. tuple representation of
    ``input_ids`` / ``attention_mask``. Coercing to a tuple of
    scalar identifiers gives a hashable, comparable key.
    """
    window = row["window"]
    return (
        window["sample_id"],
        window["contig"],
        window["locus_id"],
        window["window_start"],
        window["window_end"],
        window["sequence_hash"],
        tuple(row["input_ids"]),
    )



def test_streaming_writer_round_trip_matches_single_shot(tmp_path) -> None:
    """Two-batch streaming write produces the same record set as a one-shot shim call.

    Intent: proving equivalence on set-of-primary-keys (order-insensitive)
    guards the invariant that streaming does not drop, duplicate, or
    mutate records relative to the legacy single-shot path. This is
    the RAM-for-correctness trade the refactor makes: downstream train
    loaders must see the exact same windows regardless of how many
    ``write_batch`` calls the producer used.
    """
    pytest.importorskip("pyarrow")
    batch_a = (
        _streaming_tokenized(sample_id="cat-a-1", sequence_hash="ha1"),
        _streaming_tokenized(
            sample_id="cat-a-2",
            split="validation",
            locus_id="chr1:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="ha2",
        ),
    )
    batch_b = (
        _streaming_tokenized(
            sample_id="cat-b-1",
            contig="chr2",
            locus_id="chr2:0-8",
            sequence_hash="hb1",
        ),
        _streaming_tokenized(
            sample_id="cat-b-2",
            contig="chr2",
            split="validation",
            locus_id="chr2:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="hb2",
        ),
    )

    streaming_dir = tmp_path / "streaming"
    with TokenizedCorpusWriter(streaming_dir) as writer:
        writer.write_batch(batch_a)
        writer.write_batch(batch_b)

    single_shot_dir = tmp_path / "single_shot"
    write_tokenized_dataset(batch_a + batch_b, single_shot_dir)

    for split in ("train", "validation"):
        streaming_rows = _read_parquet_records(streaming_dir, split)
        single_shot_rows = _read_parquet_records(single_shot_dir, split)
        assert {_primary_key(row) for row in streaming_rows} == {
            _primary_key(row) for row in single_shot_rows
        }
        assert len(streaming_rows) == len(single_shot_rows)


def test_streaming_writer_empty_validation_split_is_not_materialised(tmp_path) -> None:
    """Train-only batch must not create a zero-row ``split=validation/`` tree.

    Intent: downstream loaders globbing the Hive tree treat the absence
    of a ``split=validation`` directory as "no validation data" and
    skip it; a zero-row Parquet file would be loaded and then fail
    on empty-batch assumptions, wasting iteration on a corpus where
    that split genuinely has no data.
    """
    pytest.importorskip("pyarrow")
    tokenized = (
        _streaming_tokenized(sample_id="train-1", sequence_hash="t1"),
        _streaming_tokenized(
            sample_id="train-2",
            locus_id="chr1:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="t2",
        ),
    )

    with TokenizedCorpusWriter(tmp_path) as writer:
        writer.write_batch(tokenized)

    assert (tmp_path / "split=train").is_dir()
    assert not (tmp_path / "split=validation").exists()
    metadata = _json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert "validation" not in metadata["splits"]
    assert metadata["splits"]["train"]["record_count"] == 2


def test_streaming_writer_row_group_size_honoured_per_batch(tmp_path) -> None:
    """No Parquet row group emitted by ``write_batch`` exceeds ``row_group_size``.

    Intent: the contract promises a per-batch bound on row-group
    size so downstream predicate-pushdown readers can size buffers
    without scanning the whole corpus. Verifying via
    ``ParquetFile.metadata.row_group(i).num_rows`` checks the
    actual on-disk metadata, not just the writer's input handling.
    """
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    row_group_size = 2
    batch = tuple(
        _streaming_tokenized(
            sample_id=f"sample-{index}",
            locus_id=f"chr1:{index * 8}-{index * 8 + 8}",
            block_start=index * 8,
            block_end=index * 8 + 8,
            window_start=index * 8,
            window_end=index * 8 + 6,
            sequence_hash=f"h{index}",
        )
        for index in range(5)
    )
    contract = ExportContract(
        format="parquet",
        row_group_size=row_group_size,
        preserve_raw_windows=False,
        preserve_sequence_hashes=True,
        preserve_coordinates=True,
    )

    with TokenizedCorpusWriter(tmp_path, contract=contract) as writer:
        writer.write_batch(batch)

    parquet_files = list((tmp_path / "split=train").rglob("*.parquet"))
    assert parquet_files, "expected at least one Parquet file under split=train"
    for parquet_path in parquet_files:
        parquet_file = pyarrow_parquet.ParquetFile(parquet_path)
        for index in range(parquet_file.metadata.num_row_groups):
            assert parquet_file.metadata.row_group(index).num_rows <= row_group_size



def test_streaming_writer_cleans_partial_files_on_mid_stream_exception(tmp_path) -> None:
    """An exception after a successful batch must remove any half-written Parquet artifacts.

    Intent: downstream training jobs that autodiscover the corpus by
    globbing ``*.parquet`` must never pick up a file that was written
    before an error aborted the run. The cleanup contract is the
    difference between a failed pretraining run that can simply be
    retried and one that silently trains on a truncated prefix of
    the intended data.
    """
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    good_batch = (
        _streaming_tokenized(sample_id="ok-1", sequence_hash="ok1"),
        _streaming_tokenized(
            sample_id="ok-2",
            locus_id="chr1:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="ok2",
        ),
    )

    class _DeliberateFailure(RuntimeError):
        """Raised by the test to simulate a mid-stream producer failure."""

    with pytest.raises(_DeliberateFailure):
        with TokenizedCorpusWriter(tmp_path) as writer:
            writer.write_batch(good_batch)
            assert list(tmp_path.rglob("*.parquet")), (
                "precondition: first batch must have materialised at least one Parquet file"
            )
            raise _DeliberateFailure("simulated mid-stream producer error")

    remaining_parquet = list(tmp_path.rglob("*.parquet"))
    assert remaining_parquet == [], (
        f"expected no Parquet artifacts to survive exception cleanup, got {remaining_parquet}"
    )
    assert not (tmp_path / "metadata.json").exists(), (
        "metadata.json must not be written when the with-block raises"
    )
    for stray in tmp_path.rglob("*"):
        if stray.is_file() and stray.suffix == ".parquet":
            pyarrow_parquet.ParquetFile(stray)


def test_streaming_writer_shim_parity_identical_records_and_manifest(tmp_path) -> None:
    """Shim and direct one-batch streaming write produce identical records + manifests.

    Intent: the shim is the ONLY thing that stops the legacy consensus
    pretrain pipeline from needing edits. If the shim and the direct
    ``TokenizedCorpusWriter`` single-batch path ever diverge on
    record content or manifest JSON, downstream audits that diff
    corpus artifacts across pipeline versions would flag false
    positives. Comparing after stripping non-deterministic fields
    (there are none today, but keep the door open for pyarrow
    metadata timestamps) makes the parity contract explicit.
    """
    pytest.importorskip("pyarrow")
    batch = (
        _streaming_tokenized(sample_id="cat-1", sequence_hash="h1"),
        _streaming_tokenized(
            sample_id="cat-2",
            split="validation",
            locus_id="chr1:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="h2",
        ),
        _streaming_tokenized(
            sample_id="cat-3",
            contig="chr2",
            locus_id="chr2:0-8",
            sequence_hash="h3",
        ),
    )

    shim_dir = tmp_path / "shim"
    direct_dir = tmp_path / "direct"
    write_tokenized_dataset(batch, shim_dir)
    with TokenizedCorpusWriter(direct_dir) as writer:
        writer.write_batch(batch)

    for split in ("train", "validation"):
        shim_rows = _read_parquet_records(shim_dir, split)
        direct_rows = _read_parquet_records(direct_dir, split)
        assert {_primary_key(row) for row in shim_rows} == {
            _primary_key(row) for row in direct_rows
        }

    shim_manifest = _json.loads((shim_dir / "metadata.json").read_text(encoding="utf-8"))
    direct_manifest = _json.loads((direct_dir / "metadata.json").read_text(encoding="utf-8"))
    assert shim_manifest == direct_manifest


def test_tokenized_corpus_writer_bounded_manifest_memory(tmp_path) -> None:
    """Writer retains no in-memory manifest dict across batches.

    Intent: the full felid corpus has ~7M loci, so an in-memory
    ``{locus_id: SplitManifestEntry}`` dict at close would add
    ~2\u20134 GB of Python heap pressure and violate the spec's
    "peak RAM \u2248 O(largest single species)" claim. The refactor
    stores per-locus manifest rows in a SQLite sidecar and streams
    them into ``metadata.json`` at close. This test verifies two
    contracts:

    1. The offending attribute ``_manifest_entries`` no longer exists
       on the writer after the ``with`` block, so any accidental
       reintroduction during future edits trips this test.
    2. The streamed ``metadata.json`` still round-trips every locus
       across multiple batches in ``(contig, block_start, split)``
       order, so callers never see a truncated manifest.

    A regression in either clause breaks the memory-bound guarantee
    surfaced to multi-species callers.
    """
    pytest.importorskip("pyarrow")
    batches: list[tuple[TokenizedWindow, ...]] = []
    all_locus_ids: list[str] = []
    for batch_idx in range(3):
        batch_records: list[TokenizedWindow] = []
        for window_idx in range(5):
            start = batch_idx * 1000 + window_idx * 8
            end = start + 8
            locus_id = f"chr1:{start}-{end}"
            all_locus_ids.append(locus_id)
            batch_records.append(
                _streaming_tokenized(
                    sample_id=f"b{batch_idx}-w{window_idx}",
                    locus_id=locus_id,
                    block_start=start,
                    block_end=end,
                    window_start=start,
                    window_end=start + 6,
                    sequence_hash=f"h-{batch_idx}-{window_idx}",
                )
            )
        batches.append(tuple(batch_records))

    with TokenizedCorpusWriter(tmp_path) as writer:
        for batch in batches:
            writer.write_batch(batch)

    assert not hasattr(writer, "_manifest_entries"), (
        "writer must not retain an in-memory manifest dict after the SQLite refactor"
    )
    assert not (tmp_path / ".locus_manifest.sqlite").exists(), (
        "SQLite sidecar is scratch; it must be removed on clean exit"
    )

    metadata = _json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    manifest_locus_ids = [entry["locus_id"] for entry in metadata["split_manifest"]]
    assert sorted(manifest_locus_ids) == sorted(all_locus_ids)
    block_starts = [entry["block_start"] for entry in metadata["split_manifest"]]
    assert block_starts == sorted(block_starts), (
        "split_manifest must be streamed in (contig, block_start, split) order"
    )



def test_writer_cleans_up_parquet_when_metadata_write_fails(
    tmp_path, monkeypatch
) -> None:
    """Metadata-write failure must trigger the Parquet+sidecar cleanup path.

    Intent: Greptile #7 flagged that a raw ``_write_metadata_json`` call
    left a half-written tree (Parquet files on disk, no manifest, SQLite
    sidecar still present) when the metadata write itself failed mid-way.
    Inject an ``OSError`` at the ``metadata.json`` ``open`` call — and only
    there, leaving the SQLite sidecar and Parquet writes alone — then
    assert that (a) the error propagates to the caller and (b) neither
    the metadata file, the Parquet files, nor the scratch SQLite sidecar
    survive in the output tree.
    """
    pytest.importorskip("pyarrow")
    batch = (
        _streaming_tokenized(sample_id="meta-1", sequence_hash="m1"),
        _streaming_tokenized(
            sample_id="meta-2",
            locus_id="chr1:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="m2",
        ),
    )

    import pathlib as _pathlib

    real_open = _pathlib.Path.open

    def _open_failing_on_metadata(self, *args, **kwargs):
        """Raise only on the ``metadata.json`` write; pass through otherwise."""
        if self.name == "metadata.json":
            raise OSError("injected disk full")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(_pathlib.Path, "open", _open_failing_on_metadata)

    with pytest.raises(OSError, match="injected disk full"):
        with TokenizedCorpusWriter(tmp_path) as writer:
            writer.write_batch(batch)

    assert not (tmp_path / "metadata.json").exists(), (
        "metadata.json must not remain when its own write raised"
    )
    assert not list(tmp_path.rglob("*.parquet")), (
        "Parquet files must be cleaned up when metadata-write fails, just "
        "like when the caller raises inside the with-block"
    )
    assert not (tmp_path / ".locus_manifest.sqlite").exists(), (
        "SQLite sidecar must be torn down on every __exit__ path"
    )


def test_writer_metadata_json_tolerates_new_top_level_key(tmp_path) -> None:
    """A subclass inserting an extra head key must produce alphabetical JSON.

    Intent: Greptile #8 flagged that the prior ``str.index`` splice in
    ``_write_metadata_json`` hard-coded the position of ``split_manifest``
    between ``sequence_hash_algorithm`` and ``splits``. If a future key
    (e.g. ``split_registry``) were added to the head, the splice would
    either collide or place the key in the wrong slot. The refactor emits
    one top-level key at a time in ``sorted`` order, so any new head field
    lands in its alphabetical position for free. This test exercises that
    via the documented ``_build_metadata_head`` test seam.
    """
    pytest.importorskip("pyarrow")
    batch = (
        _streaming_tokenized(sample_id="reg-1", sequence_hash="r1"),
        _streaming_tokenized(
            sample_id="reg-2",
            locus_id="chr1:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="r2",
        ),
    )

    injected_registry = {"version": 1, "shards": ["train-0"]}

    class _WriterWithRegistry(TokenizedCorpusWriter):
        """Subclass that injects ``split_registry`` via the documented seam."""

        def _build_metadata_head(self) -> dict:
            head = super()._build_metadata_head()
            # ``split_registry`` sorts strictly between ``split_manifest``
            # (streamed) and ``splits`` (eagerly serialised), so its
            # presence proves alphabetical ordering is preserved across
            # the stream/eager boundary.
            head["split_registry"] = injected_registry
            return head

    with _WriterWithRegistry(tmp_path) as writer:
        writer.write_batch(batch)

    data = _json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert data["split_registry"] == injected_registry
    assert isinstance(data["split_manifest"], list)
    assert len(data["split_manifest"]) == len(batch)
    manifest_locus_ids = {entry["locus_id"] for entry in data["split_manifest"]}
    assert manifest_locus_ids == {"chr1:0-8", "chr1:8-16"}
    assert list(data.keys()) == sorted(data.keys()), (
        "top-level keys must be emitted in alphabetical order, including "
        "any subclass-injected field"
    )
    assert data["access_pattern"] == "offline_window_materialization"
    assert data["export_format"] == "parquet"
    assert "train" in data["splits"]
    assert data["splits"]["train"]["record_count"] == len(batch)


def test_transformers_import_error_surfaces_pinned_hint(monkeypatch):
    """The runtime remediation hint must match the project dep bound.

    Intent: Registry #13 — if ``transformers`` is missing, the
    ``RuntimeError`` raised from ``load_dnabert2_tokenizer`` must
    tell the user to install exactly the bound declared in
    ``pyproject.toml`` (no drift). Forces the ``ImportError`` path
    by hiding the real ``transformers`` module.
    """
    import builtins
    import sys
    from jaguar_geo_assign.data.preprocessor import (
        _SUPPORTED_TRANSFORMERS_VERSION_HINT,
        load_dnabert2_tokenizer,
    )

    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("simulated missing transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        load_dnabert2_tokenizer()

    # The message must carry the pinned remediation hint, and the hint must
    # itself reference the upper bound we enforce in pyproject.toml.
    assert _SUPPORTED_TRANSFORMERS_VERSION_HINT in str(exc_info.value)
    assert "<6" in _SUPPORTED_TRANSFORMERS_VERSION_HINT
    assert ">=5.6" in _SUPPORTED_TRANSFORMERS_VERSION_HINT
