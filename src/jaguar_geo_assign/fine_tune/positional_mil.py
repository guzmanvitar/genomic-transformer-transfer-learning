# ruff: noqa: F722, F821, UP037  # jaxtyping shape annotations use string-based dimensions
"""Positional gated-attention MIL modules for jaguar geographic assignment.

This module implements the per-individual multi-instance learning (MIL)
architecture used after offline DNABERT-2 embedding extraction. Positional
conditioning is fixed and sinusoidal so the model cannot learn to erase genomic
location information, while the bag aggregator remains linear in the number of
loci.
"""

from __future__ import annotations

import logging

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import nn

from jaguar_geo_assign.fine_tune.model import (
    BiomeClassificationHead,
    CoordinateRegressionHead,
    JaguarMTLOutput,
)

logger = logging.getLogger(__name__)


class ContinuousPositionalEmbedding1D(nn.Module):
    """Fixed sinusoidal positional encoding for genomic base-pair coordinates.

    The encoding uses raw base-pair positions only, so identical positions on
    different chromosomes receive identical vectors after chromosome-boundary
    resets. This limitation is intentional for Phase 1: it preserves a simple,
    non-learned positional signal while deferring chromosome-specific identity
    to the later training-phase decision described in the project spec.
    """

    def __init__(self, d_model: int, genome_scale: float = 1e8) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal sin/cos pairs")
        if genome_scale <= 0.0:
            raise ValueError("genome_scale must be positive")

        self.d_model = d_model
        self.genome_scale = float(genome_scale)

        half_dim = d_model // 2
        exponent = torch.arange(half_dim, dtype=torch.float32) * (2.0 / float(d_model))
        omega = torch.pow(torch.tensor(10000.0, dtype=torch.float32), -exponent)
        self.register_buffer("omega", omega, persistent=False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        bp_positions: Float[torch.Tensor, "bag"],
    ) -> Float[torch.Tensor, "bag d_model"]:
        """Return detached sinusoidal encodings with no learnable parameters."""

        with torch.no_grad():
            pos_scaled = bp_positions.to(dtype=torch.float32) / self.genome_scale
            args = pos_scaled.unsqueeze(-1) * self.omega.unsqueeze(0)
            pe = torch.empty(
                (bp_positions.shape[0], self.d_model),
                device=bp_positions.device,
                dtype=torch.float32,
            )
            pe[:, 0::2] = torch.sin(args)
            pe[:, 1::2] = torch.cos(args)
        return pe.detach()


class PositionalGatedAttentionMIL(nn.Module):
    """Gated attention MIL aggregator with continuous positional conditioning.

    The attention gate follows Ilse et al. (2018): positional encodings are
    added only for gating, while the final pooled vector is the weighted sum of
    the raw embeddings so downstream heads operate on the original DNABERT-2
    representation space.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        locus_dropout: float,
        genome_scale: float = 1e8,
    ) -> None:
        super().__init__()
        if locus_dropout >= 1.0:
            raise ValueError("locus_dropout must be < 1.0")

        self.locus_dropout = float(locus_dropout)
        self.pos_enc = ContinuousPositionalEmbedding1D(embedding_dim, genome_scale=genome_scale)
        self.linear_tanh = nn.Linear(embedding_dim, hidden_dim)
        self.linear_sig = nn.Linear(embedding_dim, hidden_dim)
        self.linear_attn = nn.Linear(hidden_dim, 1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        embeddings: Float[torch.Tensor, "bag embed_dim"],
        bp_positions: Float[torch.Tensor, "bag"],
    ) -> tuple[Float[torch.Tensor, "embed_dim"], Float[torch.Tensor, "bag"]]:
        """Aggregate a variable-length bag into one embedding plus attention weights."""

        pe = self.pos_enc(bp_positions).to(device=embeddings.device, dtype=embeddings.dtype)
        conditioned = embeddings + pe.detach()
        h_tanh = torch.tanh(self.linear_tanh(conditioned))
        h_sig = torch.sigmoid(self.linear_sig(conditioned))
        h = h_tanh * h_sig
        attention_logits = self.linear_attn(h).squeeze(-1)

        if self.training and self.locus_dropout > 0.0:
            keep_mask = torch.bernoulli(
                torch.full(
                    (attention_logits.shape[0],),
                    1.0 - self.locus_dropout,
                    device=attention_logits.device,
                    dtype=torch.float32,
                )
            ).bool()
            attention_logits = attention_logits.masked_fill(~keep_mask, float("-inf"))

        alpha = torch.softmax(attention_logits, dim=0)
        if torch.isnan(alpha).any():
            bag_size = int(embeddings.shape[0])
            if self.training:
                logger.warning(
                    "NaN attention weights (bag=%d, locus_dropout=%.2f); propagating NaN to "
                    "force training-loop skip_step",
                    bag_size,
                    self.locus_dropout,
                )
                nan_z = torch.full(
                    (embeddings.shape[-1],),
                    float("nan"),
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
                return nan_z, alpha

            logger.warning(
                "NaN attention during eval (bag=%d, locus_dropout=%.2f); using uniform fallback",
                bag_size,
                self.locus_dropout,
            )
            alpha = torch.full(
                (bag_size,),
                1.0 / float(max(bag_size, 1)),
                device=attention_logits.device,
                dtype=embeddings.dtype,
            )

        pooled = (alpha.unsqueeze(-1).to(dtype=embeddings.dtype) * embeddings).sum(dim=0)
        return pooled, alpha


class JaguarPositionalMILNetwork(nn.Module):
    """Full MIL network: gated bag pooling plus the existing jaguar task heads.

    The backbone is intentionally absent from this module so the offline
    extraction contract remains explicit and accidental backbone fine-tuning is
    impossible at MIL-training time.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_biomes: int | None = None,
        dropout_prob: float = 0.1,
        locus_dropout: float = 0.1,
        genome_scale: float = 1e8,
    ) -> None:
        super().__init__()
        self.gated_mil = PositionalGatedAttentionMIL(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            locus_dropout=locus_dropout,
            genome_scale=genome_scale,
        )
        self.coordinate_head = CoordinateRegressionHead(embedding_dim, dropout_prob)
        self.biome_head: BiomeClassificationHead | None
        if num_biomes is None:
            self.biome_head = None
        else:
            self.biome_head = BiomeClassificationHead(embedding_dim, num_biomes, dropout_prob)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        embeddings: Float[torch.Tensor, "bag embed_dim"],
        bp_positions: Float[torch.Tensor, "bag"],
    ) -> JaguarMTLOutput:
        """Return per-individual coordinate predictions and optional biome logits."""

        pooled, _ = self.gated_mil(embeddings, bp_positions)
        pooled_batch = pooled.unsqueeze(0)
        coordinate = self.coordinate_head(pooled_batch).squeeze(0)
        biome_logits = None
        if self.biome_head is not None:
            biome_logits = self.biome_head(pooled_batch).squeeze(0)
        return JaguarMTLOutput(coordinate=coordinate, biome_logits=biome_logits)
