"""Unit tests for the Multi-Task Learning model architecture.

Tests verify:
- Forward pass returns correct output shapes (classification logits, regression preds)
- Parameter freezing and unfreezing mechanisms work correctly
- Gradients flow to correct parameters based on freeze state

Note: Uses a tiny BERT model for fast testing (2 layers, 2 attention heads, 32 hidden size).
Production would use DNABERT-2-117M backbone loaded via HuggingFace.
"""

from __future__ import annotations

import pytest
import torch
from transformers import BertConfig, BertModel

from jaguar_geo_assign.fine_tune.mtl_model import GeographicAssignmentMTL


@pytest.fixture
def tiny_bert_config() -> BertConfig:
    """Create a minimal BERT config for fast testing.

    Drastically reduces model size (32 hidden, 2 layers) compared to
    DNABERT-2-117M (768 hidden, 12 layers). Tests run in <1s.
    """
    return BertConfig(
        vocab_size=30522,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    )


@pytest.fixture
def mtl_model(tmp_path, tiny_bert_config: BertConfig) -> GeographicAssignmentMTL:
    """Create a fresh MTL model with tiny BERT backbone for testing.

    Saves the tiny BERT to a temp directory so AutoModel.from_pretrained
    can load it directly without hitting HuggingFace.
    """
    # Create and save tiny model
    tiny_bert = BertModel(tiny_bert_config)
    model_dir = tmp_path / "tiny_bert"
    model_dir.mkdir()
    tiny_bert.config.save_pretrained(str(model_dir))
    tiny_bert.save_pretrained(str(model_dir))

    # Create MTL model with tiny backbone
    return GeographicAssignmentMTL(
        model_name_or_path=str(model_dir),
        num_biome_classes=5,
        hidden_size=32,  # Match the tiny config
        dropout_p=0.1,
    )


@pytest.fixture
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a dummy batch of tokenized sequences for testing.

    Returns batch of 4 sequences, each of length 100 (reduced for speed).
    Mirrors the structure of real tokenized genomic windows but much shorter.
    """
    batch_size = 4
    seq_len = 100
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    return input_ids, attention_mask


def test_mtl_forward_pass_returns_correct_shapes(
    mtl_model: GeographicAssignmentMTL,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
):
    """Forward pass must return (class_logits, coords_pred) with correct shapes.

    Verifies:
    - class_logits shape: [batch_size, 5]
    - coords_pred shape: [batch_size, 2]
    """
    input_ids, attention_mask = dummy_batch
    batch_size = input_ids.shape[0]

    class_logits, coords_pred = mtl_model(input_ids, attention_mask)

    assert class_logits.shape == (batch_size, 5), (
        f"Expected class_logits [batch={batch_size}, classes=5], got {class_logits.shape}"
    )
    assert coords_pred.shape == (batch_size, 2), (
        f"Expected coords_pred [batch={batch_size}, dims=2], got {coords_pred.shape}"
    )


def test_mtl_forward_pass_is_differentiable(
    mtl_model: GeographicAssignmentMTL,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
):
    """Ensure gradients flow through the model during backward pass.

    Constructs a simple loss from both heads and verifies backprop works.
    """
    input_ids, attention_mask = dummy_batch
    mtl_model.train()

    class_logits, coords_pred = mtl_model(input_ids, attention_mask)

    # Create synthetic targets
    class_targets = torch.randint(0, 5, (input_ids.shape[0],))
    coords_targets = torch.randn(input_ids.shape[0], 2)

    # Compute loss
    classification_loss = torch.nn.functional.cross_entropy(class_logits, class_targets)
    regression_loss = torch.nn.functional.mse_loss(coords_pred, coords_targets)
    total_loss = classification_loss + regression_loss

    # Backward should not raise
    total_loss.backward()

    # Verify at least some parameters have gradients
    has_grads = any(p.grad is not None for p in mtl_model.parameters() if p.requires_grad)
    assert has_grads, "Backward pass did not produce any gradients"


def test_freeze_backbone_disables_backbone_gradients(
    mtl_model: GeographicAssignmentMTL,
    dummy_batch: tuple[torch.Tensor, torch.Tensor],
):
    """After freeze_backbone(), backbone params should not accumulate gradients.

    Verifies that:
    - backbone.requires_grad is False
    - head parameters still have requires_grad=True
    """
    mtl_model.freeze_backbone()

    # Check backbone is frozen
    for param in mtl_model.backbone.parameters():
        assert param.requires_grad is False, "Backbone should be frozen"

    # Check heads are trainable
    for param in mtl_model.classification_head.parameters():
        assert param.requires_grad is True, "Classification head should be trainable"
    for param in mtl_model.regression_head.parameters():
        assert param.requires_grad is True, "Regression head should be trainable"

    # Verify only head gradients are computed
    input_ids, attention_mask = dummy_batch
    mtl_model.train()
    class_logits, coords_pred = mtl_model(input_ids, attention_mask)
    loss = class_logits.sum() + coords_pred.sum()
    loss.backward()

    # Backbone params should NOT have gradients
    backbone_has_grads = any(p.grad is not None for p in mtl_model.backbone.parameters())
    assert not backbone_has_grads, "Frozen backbone should not have gradients"

    # Head params SHOULD have gradients
    head_has_grads = any(p.grad is not None for p in mtl_model.classification_head.parameters())
    assert head_has_grads, "Classification head should have gradients"


def test_unfreeze_transformer_layers_unfreezes_final_layers(
    mtl_model: GeographicAssignmentMTL,
):
    """unfreeze_transformer_layers(num_layers=3) unfreezes final 3 transformer blocks.

    For DNABERT-2-117M with 12 layers, unfreezing 3 means layers 9, 10, 11 are trainable.
    """
    num_to_unfreeze = 3
    mtl_model.unfreeze_transformer_layers(num_layers=num_to_unfreeze)

    # Get encoder
    encoder = mtl_model.backbone.encoder
    total_layers = len(encoder.layer)

    # Check early layers are frozen
    for idx in range(total_layers - num_to_unfreeze):
        layer_has_trainable = any(p.requires_grad for p in encoder.layer[idx].parameters())
        assert not layer_has_trainable, f"Layer {idx} should be frozen but has trainable params"

    # Check final layers are unfrozen
    for idx in range(total_layers - num_to_unfreeze, total_layers):
        layer_has_trainable = any(p.requires_grad for p in encoder.layer[idx].parameters())
        assert layer_has_trainable, f"Layer {idx} should be trainable but is frozen"


def test_unfreeze_always_trains_heads(
    mtl_model: GeographicAssignmentMTL,
):
    """unfreeze_transformer_layers() must always keep heads trainable.

    Even if num_layers=0, the classification and regression heads should
    have requires_grad=True so the model can learn task-specific weights.
    """
    mtl_model.unfreeze_transformer_layers(num_layers=0)

    # Heads must always be trainable
    for param in mtl_model.classification_head.parameters():
        assert param.requires_grad is True
    for param in mtl_model.regression_head.parameters():
        assert param.requires_grad is True


def test_trainable_parameters_count_reflects_freeze_state(
    mtl_model: GeographicAssignmentMTL,
):
    """get_trainable_parameters() must reflect current freeze state.

    Compares counts across three states: all-trainable, frozen backbone, unfrozen partial.
    """
    # State 1: All trainable (initial)
    all_trainable = mtl_model.get_trainable_parameters()
    assert all_trainable > 0

    # State 2: Frozen backbone (only heads trainable)
    mtl_model.freeze_backbone()
    frozen_count = mtl_model.get_trainable_parameters()
    assert frozen_count < all_trainable, "Frozen backbone should reduce trainable param count"

    # State 3: Unfreeze partial (backbone layers + heads)
    mtl_model.unfreeze_transformer_layers(num_layers=3)
    unfrozen_count = mtl_model.get_trainable_parameters()
    assert unfrozen_count > frozen_count, "Unfreezing layers should increase trainable param count"
    assert unfrozen_count < all_trainable, (
        "Partial unfreeze should still be less than all trainable"
    )
