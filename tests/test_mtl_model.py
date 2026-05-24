"""Unit tests for the jaguar DNABERT-2 multi-task model architecture.

These tests construct a tiny BERT backbone from configuration so they exercise
the PyTorch wiring without requiring any network calls to the Hugging Face
Hub. The goal is to pin down tensor shapes and the optional classification-head
behaviour, not to validate training dynamics.
"""

from __future__ import annotations

import pytest
import torch
from transformers import BertConfig, BertModel

from jaguar_geo_assign.fine_tune.model import JaguarMTLModel


def _make_tiny_bert(hidden_size: int = 32) -> BertModel:
    """Return a small BERT backbone suitable for fast unit tests.

    The configuration keeps the model deliberately small so tests remain
    lightweight while still exercising the same code paths as a full
    DNABERT-2-style encoder.
    """

    config = BertConfig(
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=hidden_size * 2,
    )
    return BertModel(config)


def test_coordinate_only_head_emits_2d_predictions() -> None:
    """Model without a biome head returns (batch, 2) coordinate predictions.

    This guards the assumption that the coordinate-regression output is always
    a two-dimensional vector ``(latitude, longitude)`` regardless of batch
    size or sequence length.
    """

    backbone = _make_tiny_bert(hidden_size=16)
    model = JaguarMTLModel(backbone)

    batch_size, seq_len = 4, 8
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.coordinate.shape == (batch_size, 2)
    assert outputs.biome_logits is None


def test_biome_head_emits_logits_with_expected_shape() -> None:
    """Model configured with ``num_biomes`` returns classification logits.

    The logits tensor must have shape ``(batch_size, num_biomes)`` and should
    be returned alongside the coordinate predictions when a biome head is
    present.
    """

    num_biomes = 3
    backbone = _make_tiny_bert(hidden_size=24)
    model = JaguarMTLModel(backbone, num_biomes=num_biomes)

    batch_size, seq_len = 2, 5
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.coordinate.shape == (batch_size, 2)
    assert outputs.biome_logits is not None
    assert outputs.biome_logits.shape == (batch_size, num_biomes)


def test_rejects_invalid_num_biomes() -> None:
    """``num_biomes`` less than 2 is rejected to avoid degenerate heads."""

    backbone = _make_tiny_bert(hidden_size=8)
    with pytest.raises(ValueError):
        JaguarMTLModel(backbone, num_biomes=1)


def test_rejects_invalid_pooling_strategy() -> None:
    """Invalid pooling_strategy values are rejected early in the constructor.

    This ensures that callers cannot silently select an unsupported pooling
    mode and pins the contract to the set of strategies validated in
    :class:`MtlFinetuneConfig` ("cls" and "mean").
    """

    backbone = _make_tiny_bert(hidden_size=8)
    with pytest.raises(ValueError):
        JaguarMTLModel(backbone, pooling_strategy="invalid")


def test_cls_pooling_uses_first_token_hidden_state() -> None:
    """``pooling_strategy="cls"`` returns ``last_hidden_state[:, 0]``.

    The pooler_output is intentionally ignored because foundation MLM
    pre-training does not train the pooler weights, so using them would
    feed randomly-initialized features to the task heads.
    """

    backbone = _make_tiny_bert(hidden_size=4)
    model = JaguarMTLModel(backbone, pooling_strategy="cls")

    batch_size, seq_len, hidden = 2, 3, 4
    last_hidden_state = torch.randn(batch_size, seq_len, hidden, dtype=torch.float32)
    pooler_output = torch.randn(batch_size, hidden, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    pooled = model._pool(last_hidden_state, pooler_output, attention_mask)
    assert torch.allclose(pooled, last_hidden_state[:, 0])


def test_mean_pooling_respects_attention_mask() -> None:
    """Mean pooling should ignore padded positions using the attention mask.

    This guards against regressions where the mean includes padded tokens,
    which would bias the pooled representation away from content tokens
    for heavily padded sequences.
    """

    backbone = _make_tiny_bert(hidden_size=2)
    model = JaguarMTLModel(backbone, pooling_strategy="mean")

    # Single example with three tokens; the last token is masked out.
    last_hidden_state = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    pooled = model._pool(last_hidden_state, pooler_output=None, attention_mask=attention_mask)
    expected = last_hidden_state[:, :2, :].mean(dim=1)
    assert torch.allclose(pooled, expected)
