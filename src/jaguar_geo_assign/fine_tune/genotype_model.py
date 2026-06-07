# ruff: noqa: F722, F821, UP037  # jaxtyping shape annotations use string-based dimensions
"""Genotype MLP for jaguar geographic assignment.

Implements a Locator/GeoGenIE-style multilayer perceptron that predicts
geographic coordinates and biome class from genotype vectors. The architecture
follows GeoGenIE (Martin et al., 2025): batch normalization on input,
dynamically sized hidden layers with ELU activation, dropout regularization,
and dual-head output.

This module is intentionally simple — the complexity in the pipeline lives in
the VES scoring (transfer learning) and LOOCV training loop, not in the model
architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, nn

from jaguar_geo_assign.fine_tune.dataset import CoordStats
from jaguar_geo_assign.fine_tune.model import JaguarMTLOutput
from jaguar_geo_assign.fine_tune.trainer import haversine_distance_km

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenotypeMLPConfig:
    """Configuration for the genotype MLP.

    Attributes:
        n_input_features: Number of loci (after optional VES selection).
        n_biomes: Number of biome classes for the classification head.
        n_hidden_layers: Number of hidden layers in the MLP trunk.
        hidden_dim: Width of each hidden layer before overparameterization guard.
        dropout: Dropout probability applied after each hidden layer.
    """

    n_input_features: int
    n_biomes: int = 5
    n_hidden_layers: int = 2
    hidden_dim: int = 256
    dropout: float = 0.2


class JaguarGenotypeMLP(nn.Module):
    """MLP for geographic assignment from genotype vectors.

    Architecture:
        0. Optional learnable locus gate: x = genotypes * sigmoid(gate)
        1. BatchNorm1d(n_input_features) -- normalize allele counts
        2. For each hidden layer:
           Linear(in_dim, hidden_dim) -> ELU() -> Dropout(dropout)
        3. Coordinate head: Linear(hidden_dim, 2) -> (lat_z, lon_z)
        4. Biome head: Linear(hidden_dim, n_biomes) -> logits

    Overparameterization guard: if hidden_dim > n_input_features * 10,
    reduce by 20% recursively until compliant. Logs warning.
    """

    def __init__(
        self,
        config: GenotypeMLPConfig,
        ves_init_logits: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        if ves_init_logits is not None:
            self.locus_gate: nn.Parameter | None = nn.Parameter(ves_init_logits.clone())
        else:
            self.locus_gate = None

        hidden_dim = config.hidden_dim
        max_hidden = config.n_input_features * 10
        if hidden_dim > max_hidden:
            original = hidden_dim
            while hidden_dim > max_hidden:
                hidden_dim = int(hidden_dim * 0.8)
            hidden_dim = max(hidden_dim, 1)
            logger.warning(
                "Overparameterization guard: reduced hidden_dim from %d to %d "
                "(threshold: %d = n_input_features * 10).",
                original,
                hidden_dim,
                max_hidden,
            )
        self._effective_hidden_dim = hidden_dim

        self.input_bn = nn.BatchNorm1d(config.n_input_features)

        layers: list[nn.Module] = []
        in_dim = config.n_input_features
        for _ in range(config.n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.coordinate_head = nn.Linear(hidden_dim, 2)
        self.biome_head = nn.Linear(hidden_dim, config.n_biomes)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        genotypes: Float[Tensor, "batch n_features"],
    ) -> JaguarMTLOutput:
        """Forward pass returning coordinates and biome logits.

        Args:
            genotypes: Float tensor of shape (batch, n_features).
                Values are typically in {0.0, 1.0, 2.0} but may be
                continuous after VES weighting.

        Returns:
            JaguarMTLOutput with coordinate predictions and biome logits.
        """
        if self.locus_gate is not None:
            x = genotypes * torch.sigmoid(self.locus_gate)
        else:
            x = genotypes
        x = self.input_bn(x)
        x = self.trunk(x)
        coordinate = self.coordinate_head(x)
        biome_logits = self.biome_head(x)
        return JaguarMTLOutput(coordinate=coordinate, biome_logits=biome_logits)


@jaxtyped(typechecker=beartype)
def compute_genotype_loss(
    outputs: JaguarMTLOutput,
    coord_target_deg: Float[Tensor, "batch 2"],
    biome_label: Int[Tensor, "batch"],
    *,
    coord_stats: CoordStats,
    coord_loss_weight: float = 1.0,
    cls_loss_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute weighted multi-task loss: haversine(coords) + CE(biome).

    The coordinate loss denormalizes model predictions from Z-score space
    to degrees, then computes the mean great-circle distance (haversine)
    against degree-space targets.  This aligns the training objective with
    the evaluation metric.  The raw haversine (km) is divided by 1000 for
    gradient scaling.

    Args:
        outputs: Model outputs containing coordinate predictions and biome
            logits.  Coordinate predictions are in Z-score space.
        coord_target_deg: Ground-truth coordinates in decimal degrees,
            shape ``(batch, 2)`` as ``(latitude, longitude)``.
        biome_label: Ground-truth biome class indices of shape (batch,).
        coord_stats: Per-fold normalization statistics used to denormalize
            model predictions back to degree space.
        coord_loss_weight: Scalar weight for the coordinate regression loss.
        cls_loss_weight: Scalar weight for the biome classification loss.

    Returns:
        Tuple of (total_loss, coord_loss, cls_loss) where total_loss is the
        weighted sum of the individual components.
    """
    pred_z = outputs.coordinate.float()
    pred_deg = torch.stack(
        [
            pred_z[:, 0] * coord_stats.lat_std + coord_stats.lat_mean,
            pred_z[:, 1] * coord_stats.lon_std + coord_stats.lon_mean,
        ],
        dim=-1,
    )
    target_deg = coord_target_deg.to(device=pred_z.device, dtype=torch.float32)
    coord_loss = haversine_distance_km(pred_deg, target_deg).mean() / 1000.0

    biome_logits = outputs.biome_logits
    if biome_logits is not None and cls_loss_weight != 0.0:
        cls_loss = nn.functional.cross_entropy(
            biome_logits.float(),
            biome_label.to(device=biome_logits.device),
        )
    else:
        cls_loss = pred_z.new_zeros(())

    total_loss = coord_loss_weight * coord_loss + cls_loss_weight * cls_loss
    return total_loss, coord_loss, cls_loss


@jaxtyped(typechecker=beartype)
def apply_ves_weighting(
    genotypes: Float[Tensor, "batch n_loci"],
    ves_scores: Float[Tensor, "n_loci"],
) -> Float[Tensor, "batch n_loci"]:
    """Weight genotype values by absolute VES scores.

    Args:
        genotypes: Genotype matrix of shape (batch, n_loci).
        ves_scores: VES scores of shape (n_loci,).

    Returns:
        Element-wise product of genotypes and |ves_scores|.
    """
    return genotypes * ves_scores.abs()


@jaxtyped(typechecker=beartype)
def apply_ves_selection(
    genotypes: Float[Tensor, "batch n_loci"],
    ves_scores: Float[Tensor, "n_loci"],
    top_k: int,
) -> tuple[Float[Tensor, "batch top_k"], Int[Tensor, "top_k"]]:
    """Select top-K loci by absolute VES score.

    Args:
        genotypes: Genotype matrix of shape (batch, n_loci).
        ves_scores: VES scores of shape (n_loci,).
        top_k: Number of top loci to select. Clamped to the number of
            available loci if top_k exceeds n_loci.

    Returns:
        Tuple of (selected_genotypes, selected_indices) where
        selected_genotypes has shape (batch, top_k) and selected_indices
        has shape (top_k,).
    """
    top_k = min(top_k, ves_scores.shape[0])
    _, indices = torch.topk(ves_scores.abs(), k=top_k)
    indices_sorted, _ = torch.sort(indices)
    selected_genotypes = genotypes[:, indices_sorted]
    return selected_genotypes, indices_sorted


@jaxtyped(typechecker=beartype)
def compute_ves_gate_init(
    ves_scores: Float[Tensor, "n_loci"],
) -> Float[Tensor, "n_loci"]:
    """Compute gate initialization logits from VES scores.

    Maps ``|VES|`` into logit space via z-scored log-transform so that
    ``sigmoid(logit)`` spans approximately ``[0.02, 0.98]``.  High-|VES|
    loci (functionally constrained) start with gates near 1; low-|VES|
    loci start near 0.  The model refines these via backprop.

    Args:
        ves_scores: Per-locus VES scores of shape ``(n_loci,)``.

    Returns:
        Logit tensor of shape ``(n_loci,)`` suitable for
        ``nn.Parameter`` initialization.
    """
    log_abs = torch.log(ves_scores.abs().clamp(min=1e-8))
    mean = log_abs.mean()
    std = log_abs.std().clamp(min=1e-6)
    return (log_abs - mean) / std * 2.0


__all__ = [
    "GenotypeMLPConfig",
    "JaguarGenotypeMLP",
    "apply_ves_selection",
    "apply_ves_weighting",
    "compute_genotype_loss",
    "compute_ves_gate_init",
]
