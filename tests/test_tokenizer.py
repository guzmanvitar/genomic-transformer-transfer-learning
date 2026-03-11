import json
import tarfile

import pytest

from jaguar_geo_assign.data.preprocessor import (
    DNABERT2_TOKENIZER_NAME,
    DNABERT2_TOKENIZER_PROVENANCE,
    DNABERT2_TOKENIZER_REVISION,
    ExportContract,
    ExportContractError,
    TokenizerContractError,
    TokenizerProvenance,
    TokenizedWindow,
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


def test_tokenize_windows_pins_tokenizer_provenance() -> None:
    window = _window()

    tokenized = tokenize_windows((window,), FakeTokenizer())

    assert tokenized[0].tokenizer == DNABERT2_TOKENIZER_PROVENANCE
    assert tokenized[0].tokenizer.identifier == DNABERT2_TOKENIZER_NAME
    assert tokenized[0].tokenizer.revision == DNABERT2_TOKENIZER_REVISION
    assert tokenized[0].token_count == 8
    assert tokenized[0].token_to_base_ratio == 8 / 6


def test_tokenize_windows_enforces_max_position_embeddings() -> None:
    window = _window()

    with pytest.raises(TokenizerContractError, match="max_position_embeddings"):
        tokenize_windows((window,), TooLongTokenizer())


def test_tokenize_windows_rejects_unsupported_symbols_at_tokenizer_boundary() -> None:
    with pytest.raises(TokenizerContractError, match="Unsupported base"):
        tokenize_windows((_window(sequence="ACGRTN"),), FakeTokenizer())


def test_tokenize_windows_can_normalize_unsupported_symbols_to_n() -> None:
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


def test_write_webdataset_shards_persists_retokenization_provenance(tmp_path) -> None:
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