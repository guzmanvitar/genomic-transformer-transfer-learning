# ruff: noqa: F722  # jaxtyping shape annotations use string-based dimensions
"""Multi-task DNABERT-2 model for downstream jaguar fine-tuning.

This module defines a lightweight PyTorch ``nn.Module`` that wraps a
transformer backbone (for this project, DNABERT-2) with task-specific heads
for coordinate regression (latitude/longitude) and optional
biome-population classification.

The design is deliberately minimal and architecture-focused:

* Callers supply an already-initialised backbone model; this module does not
  make any network calls or load weights from the Hugging Face Hub.
* The forward pass returns typed prediction tensors but does **not** compute
  losses. Training and evaluation code remain responsible for defining
  objectives (e.g. geodesic distance) and task weighting.

This separation keeps the model reusable across different training regimes
while pinning down the shared representation and head wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float, Int, jaxtyped
from torch import nn


class CoordinateRegressionHead(nn.Module):
    """Two-layer MLP head that predicts (latitude, longitude).

    The head remains deliberately small relative to the DNABERT-2 trunk but now
    uses a two-layer MLP with a GELU non-linearity. This matches the
    fine-tuning specification while keeping most capacity in the shared
    backbone.
    """

    def __init__(self, hidden_size: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return coordinate predictions with shape ``(batch_size, 2)``."""

        return self.mlp(pooled)


class BiomeClassificationHead(nn.Module):
    """Two-layer MLP classification head for biome-population labels.

    The head emits unnormalised logits so that callers can choose the exact
    loss function (e.g. cross-entropy with class weights) and calibration
    strategy. The number of output classes is supplied at construction time
    from dataset metadata rather than being baked into the model.
    """

    def __init__(
        self,
        hidden_size: int,
        num_biomes: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if num_biomes < 2:
            raise ValueError("num_biomes must be >= 2 for a meaningful classification head")
        self.num_biomes = num_biomes
        self.mlp = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_biomes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return logits with shape ``(batch_size, num_biomes)``."""

        return self.mlp(pooled)


@dataclass
class JaguarMTLOutput:
    """Container for jaguar multi-task model outputs.

    Attributes:
        coordinate: Tensor of shape ``(batch_size, 2)`` containing
            ``(latitude, longitude)`` predictions in the same units as the
            training targets.
        biome_logits: Optional tensor of shape ``(batch_size, num_biomes)``
            with unnormalised classification scores. ``None`` when the model
            was constructed without a biome head.
    """

    coordinate: torch.Tensor
    biome_logits: torch.Tensor | None = None


class JaguarMTLModel(nn.Module):
    """DNABERT-2-based multi-task model for jaguar downstream tasks.

    The model wraps a transformer backbone (e.g. DNABERT-2) and attaches two
    small heads:

    * a coordinate-regression head predicting latitude/longitude, and
    * an optional biome-population classification head.

    The backbone is treated as an opaque sequence encoder that returns a
    ``last_hidden_state`` tensor and, optionally, a ``pooler_output``. A
    single pooled embedding per sequence is extracted and fed into the heads.

    This class intentionally **does not** compute losses; callers are expected
    to implement task-specific objectives (for example, geodesic loss on
    coordinates and cross-entropy on biome labels) using the returned
    :class:`JaguarMTLOutput`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_biomes: int | None = None,
        dropout_prob: float = 0.1,
        pooling_strategy: str = "cls",
    ) -> None:
        super().__init__()

        # Infer hidden size from a transformers-style config object. The
        # backbone is expected to expose ``config.hidden_size``; failing that,
        # construction fails loudly so bugs surface during development rather
        # than at training time.
        config = getattr(backbone, "config", None)
        hidden_size = getattr(config, "hidden_size", None) if config is not None else None
        if hidden_size is None:
            raise ValueError("backbone must expose config.hidden_size for JaguarMTLModel")

        if num_biomes is not None and num_biomes < 2:
            raise ValueError("num_biomes must be >= 2 when provided")
        if pooling_strategy not in {"cls", "mean"}:
            msg = "pooling_strategy must be 'cls' or 'mean'"
            raise ValueError(msg)

        self.backbone = backbone
        self.pooling_strategy = pooling_strategy
        self.coordinate_head = CoordinateRegressionHead(hidden_size, dropout_prob)
        self.biome_head: BiomeClassificationHead | None
        if num_biomes is None:
            self.biome_head = None
        else:
            self.biome_head = BiomeClassificationHead(hidden_size, num_biomes, dropout_prob)

    @jaxtyped
    def _pool(
        self,
        last_hidden_state: Float[torch.Tensor, "batch seq hidden"],
        pooler_output: Float[torch.Tensor, "batch hidden"] | None,
        attention_mask: Int[torch.Tensor, "batch seq"] | None,
    ) -> Float[torch.Tensor, "batch hidden"]:
        """Return a single embedding per sequence from backbone outputs.

        When ``pooling_strategy`` is ``"cls"`` (the default), preference is
        given to ``pooler_output`` when available (common for BERT-style
        models). If absent, the method falls back to using the first token's
        hidden state (typically the ``[CLS]`` token) so that non-pooled
        backbones remain usable.

        When ``pooling_strategy`` is ``"mean"``, the method computes a masked
        mean over the sequence dimension using ``attention_mask`` when
        provided, and falls back to an unmasked mean otherwise. The denominator
        is clamped to at least one token to avoid division-by-zero NaNs in
        degenerate all-padding cases.
        """

        if self.pooling_strategy == "cls":
            if pooler_output is not None:
                return pooler_output
            return last_hidden_state[:, 0]

        # Mean pooling path.
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)

        mask = attention_mask.to(dtype=last_hidden_state.dtype)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        masked_sum = (last_hidden_state * mask.unsqueeze(-1)).sum(dim=1)
        return masked_sum / lengths

    @jaxtyped
    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"] | None = None,
        token_type_ids: Int[torch.Tensor, "batch seq"] | None = None,
        **kwargs: Any,
    ) -> JaguarMTLOutput:
        """Run the multi-task model and return coordinate and optional biome outputs.

        Args:
            input_ids: Token IDs of shape ``(batch_size, seq_len)``.
            attention_mask: Optional attention mask of the same shape.
            token_type_ids: Optional segment IDs (for models that use them).
            **kwargs: Forward-compatible keyword arguments passed through to
                the backbone model's ``forward`` method (for example,
                ``position_ids``).

        Returns:
            :class:`JaguarMTLOutput` containing coordinate predictions and
            optional biome logits.
        """

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            **kwargs,
        )
        pooled = self._pool(
            outputs.last_hidden_state,
            getattr(outputs, "pooler_output", None),
            attention_mask,
        )
        coordinate = self.coordinate_head(pooled)
        biome_logits = self.biome_head(pooled) if self.biome_head is not None else None
        return JaguarMTLOutput(coordinate=coordinate, biome_logits=biome_logits)
