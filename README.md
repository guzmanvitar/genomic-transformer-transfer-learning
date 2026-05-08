# jaguar-geo-assign

`jaguar-geo-assign` is a DNABERT-2 transfer-learning repository for jaguar geographic assignment. The codebase is organized around a two-stage architecture:

1. **Felid Foundation pre-training** builds a multi-species felid corpus from six approved reference assemblies and supports continued DNABERT-2 masked-language-model training.
2. **Jaguar multi-task fine-tuning** turns jaguar VCFs into 512 bp locus windows and trains a model that predicts coordinates and biome-population labels.

The legacy feline consensus-pretraining pipeline has been removed. What remains is the active foundation workflow plus the shared VCF/reference validation helpers still needed by jaguar fine-tuning.

## Repository status

### Implemented and active

- Felid foundation assembly acquisition.
- Felid foundation corpus construction (`felid-foundation-pretrain`).
- Felid foundation continued pre-training (`train-felid-foundation`).
- Jaguar fine-tuning data, model, and trainer modules under `src/jaguar_geo_assign/fine_tune/`.
- Jaguar locus-window extraction from VCFs under `src/jaguar_geo_assign/data/finetune_windows.py`.

### Present but still scaffolded in the CLI

- `fine-tune`
- `evaluate`
- `baseline-evaluate`
- `report`

Those stage names remain in the bootstrap CLI, but they are not yet wired to end-to-end training/evaluation commands.

## Installation

### Requirements

- Python `>=3.11,<3.12`
- `uv`

### Setup

1. Sync the environment:
   - `uv sync`
2. Inspect the CLI:
   - `uv run python -m jaguar_geo_assign.cli --help`

## Architecture overview

## Stage 1: Felid Foundation pre-training

This stage is FASTA-only. It does **not** use VCF consensus generation.

### Inputs

- Six approved felid reference assemblies declared in `configs/examples/felid_foundation_pretrain.toml`
- DNABERT-2 tokenizer contract pinned to `zhihan1996/DNABERT-2-117M`

### Pipeline flow

1. **Acquire assemblies**
   - `src/jaguar_geo_assign/data/felid_acquisition.py`
   - Downloads the six approved reference FASTAs with checksum validation.
2. **Build tokenized corpus**
   - `src/jaguar_geo_assign/pretrain/felid_foundation_pipeline.py`
   - Streams one species at a time, windows the sequence, tokenizes it, and writes Parquet shards.
3. **Run continued pre-training**
   - `src/jaguar_geo_assign/pretrain/foundation_training.py`
   - Loads the tokenized corpus and runs DNABERT-2 masked-language-model training.

### Key outputs

- `data/raw/felid_foundation/reference/` — downloaded FASTAs
- `data/processed/felid_foundation_pretrain/felid_foundation_tokens/` — tokenized Parquet corpus
- `artifacts/felid_foundation_pretrain/felid_foundation_pretrain_run_summary.json` — corpus summary
- `models/foundation_felid/` — training outputs such as `best/`, `latest/`, and TensorBoard logs

## Stage 2: Jaguar MTL fine-tuning

This stage uses jaguar VCFs plus metadata and fine-tunes DNABERT-2 for two tasks:

- **coordinate regression** (`latitude`, `longitude`)
- **biome-population classification**

### Core modules

- `src/jaguar_geo_assign/data/finetune_windows.py`
  - Validates reference/VCF compatibility.
  - Emits 512 bp locus-centered windows.
  - Duplicates heterozygous loci into ref/alt windows instead of masking them away.
- `src/jaguar_geo_assign/fine_tune/dataset.py`
  - Loads windows + metadata.
  - Builds fold-aware dataloaders with `StratifiedGroupKFold` and per-individual weighting.
- `src/jaguar_geo_assign/fine_tune/model.py`
  - Wraps a DNABERT-2 backbone with coordinate and biome heads.
- `src/jaguar_geo_assign/fine_tune/trainer.py`
  - Runs two-phase training: frozen-backbone warm-up, then partial unfreezing.

### Important note about configs

- `configs/examples/fine_tune.toml` is a **bootstrap experiment config**, not the trainer config consumed by `load_mtl_finetune_config`.
- The fine-tuning trainer schema currently lives in `src/jaguar_geo_assign/config.py` as `MtlFinetuneConfig`.
- The repository does not currently ship a committed example TOML for that trainer schema.

## CLI quick reference

### Bootstrap / project framing

- Validate bootstrap config:
  - `uv run python -m jaguar_geo_assign.cli validate-config configs/examples/fine_tune.toml`
- Describe bootstrap experiment:
  - `uv run python -m jaguar_geo_assign.cli describe-experiment configs/examples/regression_transfer.toml`

### Felid foundation corpus construction

- Validate foundation config:
  - `uv run python -m jaguar_geo_assign.cli validate-felid-foundation-config configs/examples/felid_foundation_pretrain.toml`
- Describe foundation config:
  - `uv run python -m jaguar_geo_assign.cli describe-felid-foundation-config configs/examples/felid_foundation_pretrain.toml`
- Check runtime contract:
  - `uv run python -m jaguar_geo_assign.cli check-felid-foundation-runtime configs/examples/felid_foundation_pretrain.toml`
- Download the six assemblies:
  - `uv run python -m jaguar_geo_assign.cli acquire-felid-foundation-assemblies configs/examples/felid_foundation_pretrain.toml`
- Build the corpus:
  - `uv run python -m jaguar_geo_assign.cli felid-foundation-pretrain configs/examples/felid_foundation_pretrain.toml`

### Felid foundation continued pre-training

- Single-process run:
  - `uv run python -m jaguar_geo_assign.cli train-felid-foundation --config configs/examples/felid_foundation_train.toml`
- Cheap local integration test:
  - `uv run python -m jaguar_geo_assign.cli train-felid-foundation --config configs/examples/felid_foundation_train.toml --integration-test`
- Multi-GPU launch example:
  - `uv run accelerate launch --multi_gpu --num_processes 8 -m jaguar_geo_assign.cli train-felid-foundation --config configs/examples/felid_foundation_train.toml`

## Repository layout

- `src/jaguar_geo_assign/cli.py` — top-level CLI
- `src/jaguar_geo_assign/config.py` — typed config loaders and contract enforcement
- `src/jaguar_geo_assign/pretrain/` — felid foundation corpus + continued pre-training
- `src/jaguar_geo_assign/data/felid_acquisition.py` — assembly downloads/checksums
- `src/jaguar_geo_assign/data/finetune_windows.py` — jaguar VCF → 512 bp windows
- `src/jaguar_geo_assign/fine_tune/` — dataset, model, and MTL trainer
- `configs/examples/` — shipped example configs
- `tests/` — unit and integration tests

## Development and verification

- Run the test suite:
  - `uv run pytest`

Pytest is configured with `-m 'not integration'` by default, so live/network integration tests are excluded unless explicitly requested.

## What was removed

The repository no longer documents or tests the legacy feline VCF-to-consensus pretraining pipeline, its reporting/EDA layer, or consensus FASTA orchestration. Shared pure-Python VCF validation helpers remain only where they are reused by the jaguar fine-tuning data path.
