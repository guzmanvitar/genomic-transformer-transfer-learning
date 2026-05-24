"""Tests for offline DNABERT-2 embedding extraction.

These tests cover the config loader contract, CLI-adjacent extraction behavior,
and the end-to-end small-subset materialization path required by the MIL spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jaguar_geo_assign.config import EmbeddingExtractionConfig, load_embedding_extraction_config
from jaguar_geo_assign.fine_tune.extract_embeddings import run_embedding_extraction


class DummyTokenizer:
    """Minimal tokenizer stub that emits deterministic token IDs for tests.

    The first nucleotide determines the non-padding token ID so the extraction
    test can assert on shard ordering without needing a real DNABERT tokenizer.
    """

    def __init__(self) -> None:
        """Initialize the stub with a pre-defined pad token."""

        self.pad_token = "[PAD]"
        self.eos_token = None
        self.unk_token = None

    def add_special_tokens(self, tokens: dict[str, str]) -> None:
        """Register a pad token when the production helper asks for one."""

        self.pad_token = tokens["pad_token"]

    def __len__(self) -> int:
        """Return a small fixed vocab size for resize_token_embeddings tests."""

        return 16

    def __call__(
        self,
        sequences: str | list[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize sequences into dense tensors with deterministic content."""

        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        if isinstance(sequences, str):
            batch = [sequences]
        else:
            batch = sequences
        token_map = {"A": 1, "C": 2, "G": 3, "T": 4}
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for sequence in batch:
            active_len = min(len(sequence), max_length)
            token_id = token_map.get(sequence[0], 5)
            input_ids.append([token_id] * active_len + [0] * (max_length - active_len))
            attention_masks.append([1] * active_len + [0] * (max_length - active_len))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


class DummyBackbone(nn.Module):
    """Tiny backbone stub that exposes ``config.hidden_size`` like transformers models."""

    def __init__(self, hidden_size: int = 8) -> None:
        """Initialize the stub with a 1→hidden linear projection."""

        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.projection = nn.Linear(1, hidden_size, bias=False)
        nn.init.constant_(self.projection.weight, 1.0)

    def resize_token_embeddings(self, size: int) -> int:
        """Mirror the HF API even though the stub's weights do not depend on vocab size."""

        return size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ):
        """Project token IDs into hidden states so pooling becomes easy to assert."""

        hidden = self.projection(input_ids.unsqueeze(-1).float())
        return SimpleNamespace(last_hidden_state=hidden)


def test_load_embedding_extraction_config_happy_path(tmp_path: Path) -> None:
    """Loader must populate defaults and return the typed extraction config."""

    config_path = tmp_path / "embedding_extraction.toml"
    config_path.write_text(
        "\n".join(
            [
                "[extraction]",
                'backbone_path = "models/foundation_felid/best/hf_model"',
                'windows_jsonl = "windows.jsonl"',
                'metadata_csv = "metadata.csv"',
                'output_dir = "artifacts/embeddings"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_embedding_extraction_config(config_path)

    assert isinstance(config, EmbeddingExtractionConfig)
    assert config.pooling_strategy == "cls"
    assert config.extraction_batch_size == 128
    assert config.device == "auto"
    assert config.dtype_str == "float32"


def test_load_embedding_extraction_config_rejects_bad_dtype(tmp_path: Path) -> None:
    """The extraction config must reject unsupported on-disk dtypes."""

    config_path = tmp_path / "embedding_extraction_bad_dtype.toml"
    config_path.write_text(
        "\n".join(
            [
                "[extraction]",
                'backbone_path = "backbone"',
                'windows_jsonl = "windows.jsonl"',
                'metadata_csv = "metadata.csv"',
                'output_dir = "out"',
                'dtype_str = "float16"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dtype_str"):
        load_embedding_extraction_config(config_path)


def test_run_embedding_extraction_writes_shards_manifest_and_contig_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Small-subset extraction must materialize tensors and metadata consistently.

    This regression test verifies the task's definition of done on a tiny local
    cohort: per-individual shards are written, the manifest is complete, the
    contig rank file is deterministic, and the emitted tensors use float32 with
    the expected ordering.
    """

    windows_path = tmp_path / "windows.jsonl"
    metadata_path = tmp_path / "metadata.csv"
    output_dir = tmp_path / "embeddings"
    backbone_dir = tmp_path / "backbone"
    backbone_dir.mkdir()

    windows = [
        {"sample_id": "s2", "contig": "chr2", "locus_pos": 30, "sequence": "AAAA"},
        {"sample_id": "s1", "contig": "chr10", "locus_pos": 20, "sequence": "CCCC"},
        {"sample_id": "s1", "contig": "chr1", "locus_pos": 10, "sequence": "GGGG"},
        {"sample_id": "missing", "contig": "chrX", "locus_pos": 99, "sequence": "TTTT"},
    ]
    windows_path.write_text(
        "\n".join(json.dumps(window) for window in windows) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        "\n".join(
            [
                "sample_id,individual_id,biome_population_label,latitude,longitude",
                "s1,ind-1,Amazon,1.5,2.5",
                "s2,ind-2,Cerrado,3.5,4.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "embedding_extraction.toml"
    config_path.write_text(
        "\n".join(
            [
                "[extraction]",
                f'backbone_path = "{backbone_dir}"',
                f'windows_jsonl = "{windows_path}"',
                f'metadata_csv = "{metadata_path}"',
                f'output_dir = "{output_dir}"',
                'pooling_strategy = "cls"',
                "extraction_batch_size = 2",
                'device = "cpu"',
                'dtype_str = "float32"',
            ]
        ),
        encoding="utf-8",
    )

    backbone = DummyBackbone(hidden_size=8)
    monkeypatch.setattr(
        "jaguar_geo_assign.fine_tune.extract_embeddings._ensure_custom_code",
        lambda _: None,
    )
    monkeypatch.setattr(
        "jaguar_geo_assign.fine_tune.extract_embeddings.AutoModel.from_pretrained",
        lambda *args, **kwargs: backbone,
    )
    monkeypatch.setattr(
        "jaguar_geo_assign.fine_tune.extract_embeddings.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    result = run_embedding_extraction(config_path)

    assert result.n_individuals == 2
    assert result.n_windows_extracted == 3
    assert result.n_windows_dropped == 1
    assert result.manifest_path == output_dir / "manifest.jsonl"
    assert backbone.training is False
    assert not any(parameter.requires_grad for parameter in backbone.parameters())

    contig_rank = json.loads((output_dir / "contig_rank.json").read_text(encoding="utf-8"))
    assert list(contig_rank) == ["chr1", "chr10", "chr2", "chrX"]
    assert contig_rank["chr10"] > contig_rank["chr1"]

    manifest_lines = (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    manifest_records = [json.loads(line) for line in manifest_lines]
    assert manifest_records == [
        {
            "individual_id": "ind-1",
            "shard_path": "ind-1.pt",
            "n_windows": 2,
            "sample_id": "s1",
            "latitude": 1.5,
            "longitude": 2.5,
            "biome_population_label": "Amazon",
        },
        {
            "individual_id": "ind-2",
            "shard_path": "ind-2.pt",
            "n_windows": 1,
            "sample_id": "s2",
            "latitude": 3.5,
            "longitude": 4.5,
            "biome_population_label": "Cerrado",
        },
    ]

    shard_ind1 = torch.load(output_dir / "ind-1.pt", map_location="cpu", weights_only=True)
    assert shard_ind1["embeddings"].shape == (2, 8)
    assert shard_ind1["embeddings"].dtype == torch.float32
    assert shard_ind1["bp_positions"].dtype == torch.float32
    assert shard_ind1["bp_positions"].tolist() == pytest.approx([10.0, 20.0])
    assert shard_ind1["contigs"] == ["chr1", "chr10"]
    assert shard_ind1["embeddings"][:, 0].tolist() == pytest.approx([3.0, 2.0])

    shard_ind2 = torch.load(output_dir / "ind-2.pt", map_location="cpu", weights_only=True)
    assert shard_ind2["embeddings"].shape == (1, 8)
    assert shard_ind2["bp_positions"].tolist() == pytest.approx([30.0])
    assert shard_ind2["contigs"] == ["chr2"]
    assert shard_ind2["embeddings"][0, 0].item() == pytest.approx(1.0)
