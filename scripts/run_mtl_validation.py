#!/usr/bin/env python
"""MTL (Multi-Task Learning) validation pipeline with synthetic data.

Produces concrete empirical artifacts for jaguar MTL validation:
- Training run artifacts (logs, checkpoints, TensorBoard events)
- MTL training run summary JSON
- Metadata describing artifact locations

Usage:
    uv run python scripts/run_mtl_validation.py

Outputs canonical artifact locations:
- artifacts/mtl_validation/mtl_training_run_summary.json
- artifacts/mtl_validation/tensorboard/ (TensorBoard logs)
- artifacts/mtl_validation/checkpoints/ (model checkpoints)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from transformers import BertConfig, BertModel

from jaguar_geo_assign.data.contracts import BIOME_CLASSES
from jaguar_geo_assign.data.finetune_windows import FinetuneWindow
from jaguar_geo_assign.fine_tune.data_loader import build_dataloaders
from jaguar_geo_assign.fine_tune.mtl_model import GeographicAssignmentMTL
from jaguar_geo_assign.fine_tune.mtl_training import MTLTrainingConfig, run_mtl_fine_tune_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_windows(
    n_individuals: int = 10, windows_per_individual: int = 5
) -> list[FinetuneWindow]:
    """Create synthetic FinetuneWindow data for testing MTL pipeline.

    Args:
        n_individuals: Number of jaguar individuals.
        windows_per_individual: Windows per individual.

    Returns:
        List of FinetuneWindow objects with DNA sequences.
    """
    windows = []
    # Create shorter sequences to fit within DNABERT-2's max position embeddings (512 tokens)
    # After tokenization with [CLS] and [SEP], keep sequence to ~256 bp to stay well within limit
    sequence_template = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    for ind_idx in range(n_individuals):
        for win_idx in range(windows_per_individual):
            window = FinetuneWindow(
                sample_id=f"jaguar_{ind_idx:03d}_window_{win_idx}",
                contig=f"chr{(ind_idx % 22) + 1}",
                locus_pos=1000 + win_idx * 100,
                window_start=500 + win_idx * 100,
                window_end=1012 + win_idx * 100,
                sequence=sequence_template * 3,  # ~192 bp (well under 512 token limit)
                ref_allele="A",
                alt_allele="T",
                is_heterozygous=False,
                genotype="1/1",
                filter_status="PASS",
            )
            windows.append(window)
    return windows


def create_synthetic_metadata(n_individuals: int = 25) -> pd.DataFrame:
    """Create synthetic metadata for MTL training.

    Includes geographic coordinates (latitude/longitude) and biome labels.
    Ensures at least 5 individuals per biome for stratified k-fold splitting.
    Uses canonical BIOME_CLASSES from contracts.
    """
    records = []
    # Distribute individuals evenly across canonical biomes (5 each for n_splits=5)
    for ind_idx in range(n_individuals):
        biome_idx = ind_idx % len(BIOME_CLASSES)
        records.append(
            {
                "sample_id": f"jaguar_{ind_idx:03d}_window_0",  # Reference window
                "individual_id": f"jaguar_{ind_idx:03d}",
                "biome_population_label": BIOME_CLASSES[biome_idx],
                "latitude": -15.0 + ind_idx * 0.5,
                "longitude": -55.0 + ind_idx * 0.5,
            }
        )
    df = pd.DataFrame(records).set_index("sample_id")
    return df


def create_tiny_bert_model(tmp_dir: Path) -> str:
    """Create and save a minimal BERT model for fast testing.

    Returns path to the saved model.
    """
    config = BertConfig(
        vocab_size=30522,
        hidden_size=64,  # Tiny
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    model = BertModel(config)
    model_path = tmp_dir / "tiny_bert"
    model.save_pretrained(str(model_path))
    config.save_pretrained(str(model_path))
    logger.info(f"Saved tiny BERT to {model_path}")
    return str(model_path)


def main() -> None:
    """Run MTL validation with synthetic data."""
    artifact_dir = Path("artifacts/mtl_validation")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== MTL Validation Pipeline ===")
    logger.info(f"Artifacts root: {artifact_dir.resolve()}")

    # Create synthetic data
    logger.info("Creating synthetic windows and metadata...")
    windows = create_synthetic_windows(n_individuals=25, windows_per_individual=5)
    metadata_df = create_synthetic_metadata(n_individuals=25)
    logger.info(f"Created {len(windows)} windows from {len(metadata_df)} individuals")

    # Create tiny BERT for testing
    model_path = create_tiny_bert_model(artifact_dir)

    # Build dataloaders
    logger.info("Building dataloaders...")
    train_loader, val_loader, norm_artifact = build_dataloaders(
        windows,
        metadata_df,
        current_fold=0,
        n_splits=5,
        train_batch_size=4,
        eval_batch_size=8,
        num_workers=0,
    )

    # Initialize MTL model
    logger.info("Initializing GeographicAssignmentMTL model...")
    model = GeographicAssignmentMTL(
        model_name_or_path=model_path,
        num_biome_classes=5,
        hidden_size=64,
        dropout_p=0.1,
    )

    # Configure training
    config = MTLTrainingConfig(
        model_path=model_path,
        output_dir=artifact_dir / "checkpoints",
        max_steps=50,  # Minimal training
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_steps=5,
        eval_every=10,
        save_every=25,
        log_every=5,
        seed=42,
    )

    # Run training
    logger.info("Starting MTL fine-tuning training...")
    result = run_mtl_fine_tune_training(model, train_loader, val_loader, config)
    logger.info(
        f"Training complete: {result.final_step} steps, best_eval_loss={result.best_eval_loss}"
    )

    # Save run summary
    summary = {
        "pipeline": "mtl_fine_tune_validation",
        "timestamp": datetime.utcnow().isoformat(),
        "artifact_location": str(artifact_dir.resolve()),
        "tensorboard_location": str((artifact_dir / "checkpoints/tensorboard").resolve()),
        "checkpoint_location": str((artifact_dir / "checkpoints").resolve()),
        "training_result": {
            "final_step": result.final_step,
            "best_eval_loss": float(result.best_eval_loss)
            if result.best_eval_loss is not None
            else None,
            "trainable_params": result.trainable_param_count,
            "total_params": result.total_param_count,
        },
        "config": {
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "model_hidden_size": 64,
            "n_individuals": 25,
            "windows_per_individual": 5,
        },
    }

    summary_path = artifact_dir / "mtl_training_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved run summary to {summary_path}")

    # Print canonical artifact locations
    print("\n" + "=" * 70)
    print("MTL VALIDATION ARTIFACTS")
    print("=" * 70)
    print(f"Root:        {artifact_dir.resolve()}")
    print(f"Summary:     {summary_path.resolve()}")
    print(f"Checkpoints: {(artifact_dir / 'checkpoints').resolve()}")
    print(f"TensorBoard: {(artifact_dir / 'checkpoints/tensorboard').resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
