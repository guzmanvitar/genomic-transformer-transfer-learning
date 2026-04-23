# jaguar-geo-assign

`jaguar-geo-assign` is a transfer-learning repository with a long-term goal: build reproducible genomics workflows that can eventually support downstream jaguar geographic-assignment research.

## Current scope in this repository

The **active pretraining contract** today is the felid foundation pipeline, which assembles a DNABERT-2-ready multi-species tokenized corpus from the six approved felid RefSeq reference assemblies. The feline consensus pipeline is retained alongside it as the consensus-FASTA workflow kept for downstream jaguar geographic-assignment workflows.

That current scope includes:

- acquiring the six approved felid reference FASTAs from RefSeq with pinned MD5 verify-before-skip semantics (idempotent, per-species integrity-checked),
- validating, describing, and runtime-checking the felid foundation pipeline TOML contract,
- running the felid foundation pretraining path end-to-end to produce a multi-species tokenized corpus via a streaming Parquet writer with a locus-safe within-assembly split, and
- retained support for the feline (consensus) pipeline — consensus FASTA generation from feline VCFs, preprocessing, locus-safe splitting, tokenization and export, plus diagnostics — retained for downstream jaguar geographic-assignment workflows.

The repository does **not** currently claim any of the following:

- trained models,
- downstream evaluation benchmarks,
- a completed jaguar geographic-assignment system, or
- completed fine-tuning / evaluation / reporting stages.

## Capability boundary: implemented vs scaffold

### Implemented current entry points

These are the real current CLI surfaces, grouped by pipeline.

**Felid foundation pretraining (active contract):**

- `felid-foundation-pretrain`
- `acquire-felid-foundation-assemblies`
- `validate-felid-foundation-config`
- `describe-felid-foundation-config`
- `check-felid-foundation-runtime`

**Feline consensus pretraining (retained for downstream jaguar-assignment workflows):**

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

### Felid foundation pretraining quickstart

The felid foundation path is the active pretraining contract. Operators run it as a **two-step flow**: first acquire the six approved felid reference FASTAs, then run pretraining against the downloaded assemblies.

Validate the example contract:

- `uv run python -m jaguar_geo_assign.cli validate-felid-foundation-config configs/examples/felid_foundation_pretrain.toml`

Describe the example contract:

- `uv run python -m jaguar_geo_assign.cli describe-felid-foundation-config configs/examples/felid_foundation_pretrain.toml`

Check runtime prerequisites:

- `uv run python -m jaguar_geo_assign.cli check-felid-foundation-runtime configs/examples/felid_foundation_pretrain.toml`

**Step 1 — acquire the six approved felid reference FASTAs** (MD5 verify-before-skip, idempotent, logs structured events):

- `uv run python -m jaguar_geo_assign.cli acquire-felid-foundation-assemblies configs/examples/felid_foundation_pretrain.toml`

**Step 2 — run pretraining** (fails loudly with a diagnostic pointing back to `acquire-felid-foundation-assemblies` if any expected FASTA is missing):

- `uv run python -m jaguar_geo_assign.cli felid-foundation-pretrain configs/examples/felid_foundation_pretrain.toml`

Peak RAM is bounded by the largest single species: the streaming Parquet writer holds at most one species' windows in memory before the next species begins.

### Feline consensus pretraining quickstart

The feline consensus pipeline is retained for downstream jaguar-assignment workflows.

Validate the example feline pipeline config:

- `uv run python -m jaguar_geo_assign.cli validate-feline-config configs/examples/feline_pretrain.toml`

Describe the example feline pipeline config:

- `uv run python -m jaguar_geo_assign.cli describe-feline-config configs/examples/feline_pretrain.toml`

Check runtime prerequisites for a local machine:

- `uv run python -m jaguar_geo_assign.cli check-feline-runtime configs/examples/feline_pretrain.toml`

Run the feline pipeline once local inputs exist at the configured paths:

- `uv run python -m jaguar_geo_assign.cli pretrain --config configs/examples/feline_pretrain.toml`

`pretrain` expects local feline inputs such as the reference FASTA, sample manifest, source VCF, and external runtime tools (for example `bcftools`). The example config documents the contract and expected paths; it does not ship the large raw inputs.

## Example config and generated outputs

### Felid foundation example contract

The active pretraining contract lives at `configs/examples/felid_foundation_pretrain.toml`. It pins the six approved RefSeq felid assemblies (species + accession) and the tokenizer/export contract used to assemble the multi-species corpus.

With that config, the felid foundation pipeline is designed to materialize outputs under the configured `paths.*` roots:

- `data/raw/felid_foundation/reference/` — per-species reference FASTAs (`<ACC>_<ASM>.fna.gz`) populated by `acquire-felid-foundation-assemblies`.
- `data/processed/felid_foundation_pretrain/felid_foundation_tokens/` — streaming Parquet corpus of tokenized windows written by `felid-foundation-pretrain`.
- `artifacts/felid_foundation_pretrain/felid_foundation_pretrain_run_summary.json` — pinned-schema run-summary JSON with per-species and corpus-wide statistics.

### Feline consensus example contract

The feline example contract lives at `configs/examples/feline_pretrain.toml`.

With that config, the feline pipeline is designed to materialize outputs in locations such as:

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
- `configs/examples/`: versioned example configs — now ships both contracts: `felid_foundation_pretrain.toml` (active pretraining contract) and `feline_pretrain.toml` (retained consensus pipeline)
- `tests/`: regression coverage for CLI/config behavior, consensus semantics, preprocessing, tokenization, split safety, and diagnostics
- `notebooks/eda_genomics.py`: canonical interactive genomics EDA workflow entry point
- `data/`: ignored raw and processed data locations referenced by the pipeline config
- `artifacts/`: ignored run summaries and other generated artifacts
- `reports/generated/`: ignored generated diagnostics payloads and related report outputs
- `scripts/`: developer helper scripts

## Project framing

This repository is best understood as **pipeline-first groundwork** for a broader jaguar/feline transfer-learning program, **now including a multi-species felid foundation pretraining corpus assembled from six RefSeq reference assemblies**, with the feline consensus path retained for downstream jaguar-assignment workflows.

At the moment, the codebase demonstrates reproducible felid foundation and feline consensus data pipelines and their operator-facing contracts. It does **not** yet demonstrate trained transfer-learning performance or a finished geographic-assignment workflow.
