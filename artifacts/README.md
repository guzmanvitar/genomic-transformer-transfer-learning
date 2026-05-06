# Artifacts Directory

Model checkpoints, cached embeddings, and run outputs are written here locally and not committed.

## MTL Validation Artifacts

**Status**: ✅ **Available** (Generated 2026-05-06)

The MTL (Multi-Task Learning) validation pipeline produces empirical artifacts for jaguar geographic-assignment fine-tuning:

- **Run Summary**: `mtl_validation/mtl_training_run_summary.json`
  - Complete metadata: pipeline ID, timestamp, training results, configuration

- **Model Checkpoints**: `mtl_validation/checkpoints/`
  - `latest/model/`: Final trained model
  - `best/model/`: Best checkpoint by validation loss

- **TensorBoard Logs**: `mtl_validation/checkpoints/tensorboard/`
  - Training/validation metrics, learning rates, gradient norms
  - View: `tensorboard --logdir=artifacts/mtl_validation/checkpoints/tensorboard`

- **Complete Documentation**: See `MTL_VALIDATION_ARTIFACTS.md` for full schema and usage

## Running MTL Validation Pipeline

Generate new artifacts:
```bash
uv run python scripts/run_mtl_validation.py
```

Inspect artifacts:
```bash
uv run python scripts/inspect_mtl_artifacts.py
```
