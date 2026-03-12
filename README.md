# jaguar-geo-assign

`jaguar-geo-assign` is a transfer-learning repository with a long-term goal: build reproducible genomics workflows that can eventually support downstream jaguar geographic-assignment research.

## Current scope in this repository

The **implemented** work today is the feline corpus-construction/export/diagnostics pipeline used to prepare a DNABERT-2-ready pretraining dataset from 99 Lives feline reference-plus-variant inputs.

That current scope includes:

- validating and describing the feline pipeline config contract,
- checking runtime prerequisites for the feline pipeline,
- generating consensus FASTAs from feline VCF inputs,
- applying preprocessing, locus-safe splitting, tokenization, and export, and
- writing diagnostics and run-summary artifacts.

The repository does **not** currently claim any of the following:

- trained models,
- downstream evaluation benchmarks,
- a completed jaguar geographic-assignment system, or
- completed fine-tuning / evaluation / reporting stages.

## Capability boundary: implemented vs scaffold

### Implemented current entry points

These are real current CLI surfaces for the implemented feline pipeline:

- `pretrain`
- `validate-feline-config`
- `describe-feline-config`
- `check-feline-runtime`

### Scaffold or deferred-only entry points

These commands are present as scaffolds only and should **not** be interpreted as completed training/evaluation functionality:

- `fine-tune`
- `evaluate`
- `baseline-evaluate`
- `report`

There are also older bootstrap inspection commands (`validate-config`, `describe-experiment`) in the CLI, but the active production-facing contract in this branch is the feline pipeline config.

## Quickstart and smoke checks

Install the environment:

- `uv sync`

Inspect the CLI surface:

- `uv run python -m jaguar_geo_assign.cli --help`

Validate the example feline pipeline config:

- `uv run python -m jaguar_geo_assign.cli validate-feline-config configs/examples/feline_pretrain.toml`

Describe the example feline pipeline config:

- `uv run python -m jaguar_geo_assign.cli describe-feline-config configs/examples/feline_pretrain.toml`

Check runtime prerequisites for a local machine:

- `uv run python -m jaguar_geo_assign.cli check-feline-runtime configs/examples/feline_pretrain.toml`

Run the implemented feline pipeline once local inputs exist at the configured paths:

- `uv run python -m jaguar_geo_assign.cli pretrain --config configs/examples/feline_pretrain.toml`

`pretrain` expects local feline inputs such as the reference FASTA, sample manifest, source VCF, and external runtime tools (for example `bcftools`). The example config documents the contract and expected paths; it does not ship the large raw inputs.

## Example config and generated outputs

The main example contract lives at `configs/examples/feline_pretrain.toml`.

With that config, the implemented pipeline is designed to materialize outputs in locations such as:

- `data/processed/feline_pretrain/consensus_fastas/`
- `data/processed/feline_pretrain/consensus_tokens/`
- `data/processed/feline_reference_baseline/reference_tokens/`
- `reports/generated/feline_pretrain/eda_payload.json`
- `artifacts/feline_pretrain/pretrain_run_summary.json`

These artifacts describe corpus construction and diagnostics. They are **not** trained-model checkpoints or downstream jaguar assignment results.

## Diagnostics workflow entry point

The maintained exploratory diagnostics workflow lives at `notebooks/eda_genomics.py`.

It is the canonical VS Code interactive `#%%` entry point for inspecting helper-backed genomics EDA outputs derived from the reporting layer and generated payload artifacts.

## Repository layout

- `src/jaguar_geo_assign/`: package source, config loading, CLI, pipeline, tokenization, and reporting helpers
- `configs/examples/`: versioned example configs, including the feline pretraining contract
- `tests/`: regression coverage for CLI/config behavior, consensus semantics, preprocessing, tokenization, split safety, and diagnostics
- `notebooks/eda_genomics.py`: canonical interactive genomics EDA workflow entry point
- `data/`: ignored raw and processed data locations referenced by the pipeline config
- `artifacts/`: ignored run summaries and other generated artifacts
- `reports/generated/`: ignored generated diagnostics payloads and related report outputs
- `scripts/`: developer helper scripts

## Project framing

This repository is best understood as **pipeline-first groundwork** for a broader jaguar/feline transfer-learning program.

At the moment, the codebase demonstrates a reproducible feline data pipeline and its operator-facing contracts. It does **not** yet demonstrate trained transfer-learning performance or a finished geographic-assignment workflow.