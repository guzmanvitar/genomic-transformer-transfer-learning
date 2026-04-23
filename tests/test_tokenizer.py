"""Contract tests for windowed-sequence tokenization and export.

These tests enforce the boundary between raw genomic windows and the
tokenized/exported artifacts consumed by downstream training. They protect
the invariants that (a) the pinned DNABERT-2 tokenizer provenance travels
with every record, (b) over-length or unsupported inputs fail loudly instead
of being silently dropped, and (c) Parquet and WebDataset shards preserve
per-split purity, sequence hashes, and tokenizer metadata needed to
reproduce or re-tokenize a corpus.
"""

import json
import tarfile
from dataclasses import replace

import pytest

from jaguar_geo_assign.data.preprocessor import (
    DNABERT2_TOKENIZER_NAME,
    DNABERT2_TOKENIZER_PROVENANCE,
    DNABERT2_TOKENIZER_REVISION,
    ExportContract,
    ExportContractError,
    TokenizedWindow,
    TokenizerContractError,
    TokenizerProvenance,
    WindowRecord,
    tokenize_windows,
    write_tokenized_dataset,
    write_webdataset_shards,
)


class FakeTokenizer:
    def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
        return {
            "input_ids": [101, *range(200, 200 + len(sequence)), 102],
            "attention_mask": [1] * (len(sequence) + 2),
        }


class TooLongTokenizer:
    def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
        return {
            "input_ids": [1] * (len(sequence) + 600),
            "attention_mask": [1] * (len(sequence) + 600),
        }


class SelectiveTooLongTokenizer:
    def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
        token_count = len(sequence) + 2
        if sequence == "AACCGG":
            token_count += 600
        return {
            "input_ids": [1] * token_count,
            "attention_mask": [1] * token_count,
        }


class CapturingTokenizer:
    def __init__(self) -> None:
        self.sequences: list[str] = []

    def __call__(self, sequence: str, **_: object) -> dict[str, list[int]]:
        self.sequences.append(sequence)
        return {
            "input_ids": [101, *range(200, 200 + len(sequence)), 102],
            "attention_mask": [1] * (len(sequence) + 2),
        }


def _window(*, sequence: str = "ACGTNN", sequence_hash: str = "hash") -> WindowRecord:
    return WindowRecord(
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
        sequence=sequence,
        gc_fraction=0.5,
        ambiguity_fraction=2 / 6,
        sequence_hash=sequence_hash,
    )


def _tokenized_window(**window_overrides: object) -> TokenizedWindow:
    return TokenizedWindow(
        window=replace(_window(), **window_overrides),
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        token_count=3,
        token_to_base_ratio=0.5,
        tokenizer=TokenizerProvenance(),
    )


def _read_webdataset_records(shard_path) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    with tarfile.open(shard_path, mode="r") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            extracted = archive.extractfile(member)
            assert extracted is not None
            payloads.append(json.loads(extracted.read().decode("utf-8")))
    return payloads


def test_tokenize_windows_pins_tokenizer_provenance() -> None:
    """Ensure every tokenized window carries the exact DNABERT-2 identifier and revision hash."""
    window = _window()

    tokenized = tokenize_windows((window,), FakeTokenizer())

    assert tokenized[0].tokenizer == DNABERT2_TOKENIZER_PROVENANCE
    assert tokenized[0].tokenizer.identifier == DNABERT2_TOKENIZER_NAME
    assert tokenized[0].tokenizer.revision == DNABERT2_TOKENIZER_REVISION
    assert tokenized[0].token_count == 8
    assert tokenized[0].token_to_base_ratio == 8 / 6


def test_tokenize_windows_enforces_max_position_embeddings() -> None:
    """Reject windows that tokenize past ``max_position_embeddings`` with a diagnosable error."""
    window = _window()

    with pytest.raises(
        TokenizerContractError,
        match="Retained genomic window tokenized beyond max_position_embeddings",
    ) as exc_info:
        tokenize_windows((window,), TooLongTokenizer())

    message = str(exc_info.value)
    assert "observed token_count=606" in message
    assert "max_position_embeddings=512" in message
    assert "sample_id=sample-1" in message
    assert "individual_id=cat-1" in message
    assert "source=consensus" in message
    assert "contig=chr1" in message
    assert "locus_id=chr1:0-8" in message
    assert "window=0-6" in message


def test_tokenize_windows_fail_fast_instead_of_silently_skipping_over_length_window() -> None:
    """Guard that a later over-length window aborts the whole batch rather than being silently
    dropped."""
    retained_window = _window(sequence="ACGT", sequence_hash="retained")
    offending_window = replace(
        _window(sequence="AACCGG", sequence_hash="offending"),
        sample_id="sample-2",
        individual_id="cat-2",
        contig="chr2",
        locus_id="chr2:8-16",
        block_start=8,
        block_end=16,
        window_start=8,
        window_end=14,
    )

    with pytest.raises(TokenizerContractError, match="sample_id=sample-2"):
        tokenize_windows((retained_window, offending_window), SelectiveTooLongTokenizer())


def test_tokenize_windows_rejects_unsupported_symbols_at_tokenizer_boundary() -> None:
    """Block IUPAC/ambiguity symbols outside ``ACGTN`` before they reach the tokenizer."""
    with pytest.raises(TokenizerContractError, match="Unsupported base"):
        tokenize_windows((_window(sequence="ACGRTN"),), FakeTokenizer())


def test_tokenize_windows_can_normalize_unsupported_symbols_to_n() -> None:
    """Verify the opt-in ``normalize_to_n`` policy rewrites the sequence and refreshes its hash."""
    tokenizer = CapturingTokenizer()

    tokenized = tokenize_windows(
        (_window(sequence="ACGRTN"),),
        tokenizer,
        TokenizerProvenance(unsupported_symbol_policy="normalize_to_n"),
    )

    assert tokenizer.sequences == ["ACGNTN"]
    assert tokenized[0].window.sequence == "ACGNTN"
    assert tokenized[0].window.sequence_hash != "hash"


def test_write_tokenized_dataset_requires_pyarrow_for_parquet(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surface an actionable install hint when Parquet export is requested without pyarrow."""

    def missing_backend() -> None:
        raise ExportContractError("Parquet export requires pyarrow. Install with: uv add pyarrow")

    monkeypatch.setattr(
        "jaguar_geo_assign.data.preprocessor._load_pyarrow_parquet",
        missing_backend,
    )

    window = WindowRecord(
        sample_id="sample-1",
        individual_id="cat-1",
        contig="chr1",
        source="reference",
        split="validation",
        locus_id="chr1:8-16",
        block_start=8,
        block_end=16,
        window_start=8,
        window_end=14,
        sequence="AACCGT",
        gc_fraction=0.5,
        ambiguity_fraction=0.0,
        sequence_hash="abc123",
    )
    tokenized = TokenizedWindow(
        window=window,
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        token_count=3,
        token_to_base_ratio=0.5,
        tokenizer=TokenizerProvenance(),
    )

    with pytest.raises(ExportContractError, match="uv add pyarrow"):
        write_tokenized_dataset(
            (tokenized,),
            tmp_path,
            contract=ExportContract(
                format="parquet",
                row_group_size=1,
                preserve_raw_windows=False,
                preserve_sequence_hashes=True,
                preserve_coordinates=True,
            ),
        )


def test_write_tokenized_dataset_emits_real_parquet_artifact_and_schema(tmp_path) -> None:
    """Check that the written Parquet file has the expected schema and drops raw sequence bytes."""
    pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    window = WindowRecord(
        sample_id="sample-1",
        individual_id="cat-1",
        contig="chr1",
        source="reference",
        split="validation",
        locus_id="chr1:8-16",
        block_start=8,
        block_end=16,
        window_start=8,
        window_end=14,
        sequence="AACCGT",
        gc_fraction=0.5,
        ambiguity_fraction=0.0,
        sequence_hash="abc123",
    )
    tokenized = TokenizedWindow(
        window=window,
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        token_count=3,
        token_to_base_ratio=0.5,
        tokenizer=TokenizerProvenance(),
    )

    exported = write_tokenized_dataset(
        (tokenized,),
        tmp_path,
        contract=ExportContract(
            format="parquet",
            row_group_size=1,
            preserve_raw_windows=False,
            preserve_sequence_hashes=True,
            preserve_coordinates=True,
        ),
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["export_format"] == "parquet"
    assert metadata["preserve_raw_windows"] is False
    assert metadata["preserve_sequence_hashes"] is True
    export_path = exported["validation"][0]
    assert export_path.suffix == ".parquet"
    assert export_path.read_bytes()[:4] == b"PAR1"

    file_schema = pyarrow_parquet.read_schema(export_path)
    assert set(file_schema.names) == {
        "attention_mask",
        "input_ids",
        "token_count",
        "token_to_base_ratio",
        "tokenizer",
        "window",
    }
    assert "sequence_hash" in str(file_schema)
    assert "sequence:" not in str(file_schema)
    assert "split" in str(file_schema)

    standalone_shard = tmp_path / "standalone.parquet"
    standalone_shard.write_bytes(export_path.read_bytes())

    table = pyarrow_parquet.read_table(standalone_shard)
    payload = table.to_pylist()[0]
    assert "sequence" not in payload["window"]
    assert payload["window"]["sequence_hash"] == "abc123"
    assert payload["window"]["contig"] == "chr1"
    assert payload["window"]["split"] == "validation"
    assert payload["tokenizer"]["unsupported_symbol_policy"] == "reject"


def test_write_tokenized_dataset_requires_explicit_provenance_for_empty_export(tmp_path) -> None:
    """Reject empty exports that would otherwise produce metadata without a pinned tokenizer."""
    with pytest.raises(ExportContractError, match="explicit tokenizer provenance"):
        write_tokenized_dataset(
            (),
            tmp_path,
            contract=ExportContract(
                format="parquet",
                row_group_size=1,
                preserve_raw_windows=False,
                preserve_sequence_hashes=True,
                preserve_coordinates=True,
            ),
        )


def test_write_webdataset_shards_persists_retokenization_provenance(tmp_path) -> None:
    """Verify WebDataset shards retain sequence, hash, and tokenizer revision for retokenization."""
    tokenized = _tokenized_window(
        source="reference",
        split="validation",
        locus_id="chr1:8-16",
        block_start=8,
        block_end=16,
        window_start=8,
        window_end=14,
        sequence="AACCGT",
        ambiguity_fraction=0.0,
        sequence_hash="abc123",
    )

    shard_paths = write_webdataset_shards((tokenized,), tmp_path, records_per_shard=1)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["tokenizer"]["revision"] == DNABERT2_TOKENIZER_REVISION
    assert metadata["split_manifest"][0]["locus_id"] == "chr1:8-16"
    shard_path = shard_paths["validation"][0]
    with tarfile.open(shard_path, mode="r") as archive:
        member = archive.extractfile("00000000.json")
        assert member is not None
        payload = json.loads(member.read().decode("utf-8"))
    assert payload["window"]["sequence"] == "AACCGT"
    assert payload["window"]["sequence_hash"] == "abc123"


def test_write_webdataset_shards_default_sizing_stays_split_pure(tmp_path) -> None:
    """Guard that default shard sizing never co-mingles train and validation records in one tar."""
    tokenized = (
        _tokenized_window(
            sample_id="train-sample",
            individual_id="cat-train",
            split="train",
            sequence_hash="train-hash",
        ),
        _tokenized_window(
            sample_id="validation-sample",
            individual_id="cat-validation",
            split="validation",
            locus_id="chr1:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="validation-hash",
        ),
    )

    shard_paths = write_webdataset_shards(tokenized, tmp_path)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["records_per_shard"] is None
    assert metadata["splits"]["train"]["shards"] == ["train-00000.tar"]
    assert metadata["splits"]["validation"]["shards"] == ["validation-00000.tar"]
    for split, paths in shard_paths.items():
        assert len(paths) == 1
        records = _read_webdataset_records(paths[0])
        assert {record["window"]["split"] for record in records} == {split}


def test_write_webdataset_shards_oversize_shard_request_stays_within_split(tmp_path) -> None:
    """Check that a ``records_per_shard`` larger than a split still emits one shard per split."""
    tokenized = (
        _tokenized_window(
            sample_id="train-1", individual_id="cat-train-1", split="train", sequence_hash="train-1"
        ),
        _tokenized_window(
            sample_id="train-2",
            individual_id="cat-train-2",
            split="train",
            locus_id="chr1:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="train-2",
        ),
        _tokenized_window(
            sample_id="validation-1",
            individual_id="cat-validation-1",
            split="validation",
            contig="chr2",
            locus_id="chr2:0-8",
            sequence_hash="validation-1",
        ),
    )

    shard_paths = write_webdataset_shards(tokenized, tmp_path, records_per_shard=10)

    assert [path.name for path in shard_paths["train"]] == ["train-00000.tar"]
    assert [path.name for path in shard_paths["validation"]] == ["validation-00000.tar"]
    assert len(_read_webdataset_records(shard_paths["train"][0])) == 2
    assert len(_read_webdataset_records(shard_paths["validation"][0])) == 1


def test_write_webdataset_shards_does_not_bridge_train_validation_boundary(tmp_path) -> None:
    """Guard against a full train shard spilling validation records into the same tarball."""
    tokenized = (
        _tokenized_window(
            sample_id="train-1", individual_id="cat-train-1", split="train", sequence_hash="train-1"
        ),
        _tokenized_window(
            sample_id="validation-1",
            individual_id="cat-validation-1",
            split="validation",
            locus_id="chr1:8-16",
            block_start=8,
            block_end=16,
            window_start=8,
            window_end=14,
            sequence_hash="validation-1",
        ),
        _tokenized_window(
            sample_id="validation-2",
            individual_id="cat-validation-2",
            split="validation",
            contig="chr2",
            locus_id="chr2:0-8",
            sequence_hash="validation-2",
        ),
    )

    shard_paths = write_webdataset_shards(tokenized, tmp_path, records_per_shard=2)

    assert [path.name for path in shard_paths["train"]] == ["train-00000.tar"]
    assert [path.name for path in shard_paths["validation"]] == ["validation-00000.tar"]
    assert {
        record["window"]["split"] for record in _read_webdataset_records(shard_paths["train"][0])
    } == {"train"}
    assert {
        record["window"]["split"]
        for record in _read_webdataset_records(shard_paths["validation"][0])
    } == {"validation"}
