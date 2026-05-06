"""Multi-Task Learning (MTL) model for geographic assignment using DNABERT-2 backbone.

Implements a dual-head architecture:
- Classification head: 5-class biome population prediction
- Regression head: Continuous lat/lon coordinate prediction

Two-phase unfreezing:
- Phase 1: Freeze backbone, train heads only
- Phase 2: Unfreeze final 2-3 transformer layers + heads
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class GeographicAssignmentMTL(nn.Module):
    """Multi-task learning model combining biome classification and coordinate regression.

    The model loads a DNABERT-2 backbone and attaches two task-specific heads:
    1. Classification head (Dense -> Softmax) for 5 biome classes
    2. Regression head (Dense -> Linear) for lat/lon prediction

    Implements two-phase training via freeze_backbone() and unfreeze_transformer_layers().
    """

    def __init__(
        self,
        model_name_or_path: str = "zhihan1996/DNABERT-2-117M",
        num_biome_classes: int = 5,
        hidden_size: int = 768,
        dropout_p: float = 0.1,
    ):
        """Initialize the MTL model with DNABERT-2 backbone and dual heads.

        Args:
            model_name_or_path: HuggingFace identifier or local path to DNABERT-2.
            num_biome_classes: Number of biome population classes (default 5).
            hidden_size: Backbone hidden dimension (typically 768 for DNABERT-2-117M).
            dropout_p: Dropout probability for head layers.
        """
        super().__init__()

        # Load DNABERT-2 backbone (removes MLM head, uses pooled output)
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.hidden_size = hidden_size

        # Classification head: [hidden_size] -> [num_biome_classes]
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, num_biome_classes),
        )

        # Regression head: [hidden_size] -> [2] for lat/lon
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning classification logits and regression predictions.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]

        Returns:
            Tuple of:
            - class_logits: [batch_size, num_biome_classes] unnormalized logits
            - coords_pred: [batch_size, 2] predicted lat/lon values
        """
        # Backbone forward pass; use pooler_output (CLS token representation)
        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = backbone_out.pooler_output  # [batch_size, hidden_size]

        # Task heads
        class_logits = self.classification_head(pooled)
        coords_pred = self.regression_head(pooled)

        return class_logits, coords_pred

    def freeze_backbone(self) -> None:
        """Phase 1: Freeze backbone, keep head parameters trainable.

        Sets requires_grad=False for all backbone parameters.
        Useful for fine-tuning with limited data or compute.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_transformer_layers(self, num_layers: int = 3) -> None:
        """Phase 2: Unfreeze final N transformer layers while keeping others frozen.

        Selectively unfreezes the last `num_layers` transformer blocks in the
        DNABERT-2 encoder, allowing fine-tuning of task-specific representations
        while retaining early-layer features learned during pretraining.

        Args:
            num_layers: Number of final transformer layers to unfreeze (default 3).
                For DNABERT-2-117M (12 layers total), 2-3 is typical.
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze final num_layers transformer blocks
        encoder = self.backbone.encoder
        total_layers = len(encoder.layer)
        start_idx = max(0, total_layers - num_layers)

        for idx in range(start_idx, total_layers):
            for param in encoder.layer[idx].parameters():
                param.requires_grad = True

        # Always unfreeze classification and regression heads
        for param in self.classification_head.parameters():
            param.requires_grad = True
        for param in self.regression_head.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, save_directory: str | None = None) -> None:
        """Save model state and heads to a directory.

        Saves both the backbone DNABERT-2 model and the custom MTL heads.
        In production, this integrates with the checkpoint management system.

        Args:
            save_directory: Directory to save model to.
        """
        if save_directory is None:
            raise ValueError("save_directory must be provided")

        import json
        from pathlib import Path

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save state dict
        state_dict = self.state_dict()
        torch.save(state_dict, save_path / "pytorch_model.bin")

        # Save config for later loading
        config_dict = {
            "num_biome_classes": self.classification_head[-1].out_features,
            "hidden_size": self.hidden_size,
            "model_class": "GeographicAssignmentMTL",
        }
        (save_path / "config.json").write_text(json.dumps(config_dict, indent=2))
