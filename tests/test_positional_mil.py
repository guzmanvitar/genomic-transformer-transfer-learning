"""Unit tests for positional MIL building blocks.

These tests exercise the fixed positional encoder, the gated-attention pooling
contract, and the top-level network wiring without relying on any extracted
embedding artefacts or training-loop infrastructure.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from jaguar_geo_assign.fine_tune.positional_mil import (
    ContinuousPositionalEmbedding1D,
    JaguarPositionalMILNetwork,
    PositionalGatedAttentionMIL,
)


class _FixedPositionalEncoding(nn.Module):
    """Return a deterministic positional tensor to isolate MIL aggregation tests."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, bp_positions: torch.Tensor) -> torch.Tensor:
        """Ignore positions and replay the fixed tensor registered at construction."""

        del bp_positions
        return self.value


def test_continuous_positional_embedding_repeats_when_positions_reset() -> None:
    """Same base-pair coordinate after a chromosome reset should map identically.

    Phase 1 intentionally encodes only raw base-pair positions, so loci at the
    same position on different chromosomes receive the same sinusoidal vector.
    This test pins that documented limitation so later Phase 2 work can change
    it explicitly rather than by accident.
    """

    encoder = ContinuousPositionalEmbedding1D(d_model=4, genome_scale=1.0)
    positions = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32, requires_grad=True)

    encoded = encoder(positions)

    assert encoded.shape == (3, 4)
    assert encoded.requires_grad is False
    assert torch.allclose(encoded[0], encoded[2])
    assert not torch.allclose(encoded[0], encoded[1])


def test_gated_attention_emits_normalized_weights_and_expected_shapes() -> None:
    """Gated MIL must return one pooled vector and one normalized weight per locus."""

    model = PositionalGatedAttentionMIL(
        embedding_dim=6,
        hidden_dim=4,
        locus_dropout=0.0,
        genome_scale=1.0,
    )
    model.eval()

    embeddings = torch.randn(5, 6, dtype=torch.float32)
    positions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

    pooled, alpha = model(embeddings, positions)

    assert pooled.shape == (6,)
    assert alpha.shape == (5,)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(alpha).all()
    assert alpha.sum().item() == pytest.approx(1.0)
    assert torch.all(alpha >= 0.0)


def test_gated_attention_aggregates_raw_embeddings_after_positional_gating() -> None:
    """Attention may depend on position, but pooling must use the raw embeddings.

    The spec requires positional information to influence only the gate. If the
    implementation accidentally pooled the position-conditioned tensor instead,
    the first feature below would become non-zero because only the stubbed
    positional encoding carries signal in that dimension.
    """

    model = PositionalGatedAttentionMIL(
        embedding_dim=2,
        hidden_dim=1,
        locus_dropout=0.0,
        genome_scale=1.0,
    )
    model.eval()
    model.pos_enc = _FixedPositionalEncoding(
        torch.tensor([[3.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    )

    with torch.no_grad():
        model.linear_tanh.weight.copy_(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        model.linear_tanh.bias.zero_()
        model.linear_sig.weight.copy_(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        model.linear_sig.bias.zero_()
        model.linear_attn.weight.copy_(torch.tensor([[1.0]], dtype=torch.float32))
        model.linear_attn.bias.zero_()

    embeddings = torch.tensor([[0.0, 2.0], [0.0, 4.0]], dtype=torch.float32)
    positions = torch.tensor([10.0, 20.0], dtype=torch.float32)

    pooled, alpha = model(embeddings, positions)

    gated_score = math.tanh(3.0) * torch.sigmoid(torch.tensor(3.0)).item()
    expected_alpha = torch.softmax(torch.tensor([gated_score, 0.0]), dim=0)
    expected_pooled = (expected_alpha.unsqueeze(-1) * embeddings).sum(dim=0)

    assert torch.allclose(alpha, expected_alpha)
    assert torch.allclose(pooled, expected_pooled)
    assert pooled[0].item() == pytest.approx(0.0)


def test_jaguar_positional_mil_network_matches_existing_head_output_shapes() -> None:
    """Top-level MIL network should emit JaguarMTL-compatible output shapes."""

    model = JaguarPositionalMILNetwork(
        embedding_dim=8,
        hidden_dim=4,
        num_biomes=3,
        dropout_prob=0.0,
        locus_dropout=0.0,
        genome_scale=1.0,
    )
    model.eval()

    embeddings = torch.randn(7, 8, dtype=torch.float32)
    positions = torch.arange(1, 8, dtype=torch.float32)

    outputs = model(embeddings, positions)

    assert outputs.coordinate.shape == (2,)
    assert outputs.biome_logits is not None
    assert outputs.biome_logits.shape == (3,)
