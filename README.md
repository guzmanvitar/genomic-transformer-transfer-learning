# Jaguar Geographic Assignment via DNABERT-2 Transfer Learning

A transfer-learning pipeline for geographic assignment of jaguars (*Panthera onca*) from whole-genome sequencing data. The system pre-trains DNABERT-2 on multi-species felid reference genomes, uses the pre-trained model to score variant functional importance (Variant Effect Scoring), then trains a genotype-matrix-based MLP with learnable per-locus importance gates — initialized from VES scores and refined by backpropagation — that predicts geographic coordinates from allele counts using a haversine loss.

This work addresses a core challenge in conservation genomics: endangered species that most need precise forensic tools have the least genetic data available for model development. By using felid-pretrained DNABERT-2 to identify functionally constrained SNPs — a label-free alternative to FST-based marker selection — the pipeline transfers cross-species genomic knowledge into a geographic assignment model that works with just 55 jaguar samples.

---

## Table of Contents

1. [Research Context](#research-context)
2. [Methodology Overview](#methodology-overview)
   - [Stage 1: Felid Foundation Pre-training](#stage-1-felid-foundation-pre-training)
   - [Stage 2: Variant Effect Scoring](#stage-2-variant-effect-scoring)
   - [Stage 3: Genotype MLP Training](#stage-3-genotype-mlp-training)
3. [Data Decisions](#data-decisions)
   - [Pre-training Corpus: Species Selection and Rationale](#pre-training-corpus-species-selection-and-rationale)
   - [Sequence Windowing and Tokenization](#sequence-windowing-and-tokenization)
   - [Locus-Safe Train/Validation Splitting](#locus-safe-trainvalidation-splitting)
   - [Genotype Matrix Construction](#genotype-matrix-construction)
   - [Variant Effect Scoring](#variant-effect-scoring)
   - [Coordinate Normalization](#coordinate-normalization)
4. [Modeling Decisions](#modeling-decisions)
   - [Foundation Model Selection: DNABERT-2](#foundation-model-selection-dnabert-2)
   - [Pre-training Objective and Hyperparameters](#pre-training-objective-and-hyperparameters)
   - [Genotype MLP Architecture](#genotype-mlp-architecture)
   - [VES Integration Strategies](#ves-integration-strategies)
   - [Loss Functions and Task Weighting](#loss-functions-and-task-weighting)
   - [Evaluation Metrics](#evaluation-metrics)
   - [No-VES Tuned Baseline](#no-ves-tuned-baseline)
5. [Reproducibility and Integrity Guarantees](#reproducibility-and-integrity-guarantees)
6. [Installation](#installation)
7. [Running the Pipeline](#running-the-pipeline)
   - [Step 1: Acquire Felid Reference Assemblies](#step-1-acquire-felid-reference-assemblies)
   - [Step 2: Build the Tokenized Corpus](#step-2-build-the-tokenized-corpus)
   - [Step 3: Run Foundation Pre-training](#step-3-run-foundation-pre-training)
   - [Step 4: Acquire Jaguar Raw Data](#step-4-acquire-jaguar-raw-data)
   - [Step 5: Train the Genotype MLP](#step-5-train-the-genotype-mlp)
8. [Repository Layout](#repository-layout)
9. [Development](#development)

---

## Research Context

The jaguar occupies approximately 46% of its historical range, with an estimated 173,000 individuals remaining. Population status varies sharply across Brazilian biomes: the Amazon and Pantanal harbor the largest and most genetically diverse populations, while the Atlantic Forest and Caatinga populations are small, isolated, and genetically impoverished due to habitat fragmentation. A critical gap in conservation is the ability to assign poached individuals to their population of origin, which would allow authorities to identify and prioritize anti-poaching interventions.

Traditional approaches to geographic assignment (SCAT, SPASIBA, STRUCTURE, KLFDAPC) require substantial sample sizes per population and either produce only coarse population-level assignments or demand dense spatial sampling for continuous predictions. Machine learning methods such as Locator (Battey et al., 2020) achieve high-resolution continuous geographic assignment via deep neural networks, but still require large training datasets.

This project addresses a deeper challenge: not just data scarcity, but the feature-to-sample ratio. With 55 individuals and ~83,000 candidate SNPs, training on all loci produces ~1,500 features per individual — too sparse for any classifier. The standard alternative, FST-based locus selection, estimates allele frequency differences between populations — but with only 5–18 individuals per biome, these estimates have high sampling variance, and the selected loci may partly reflect noise rather than true geographic signal.

A DNABERT-2 genomic language model is pre-trained on six felid reference assemblies to learn the genomic grammar conserved across ~10 million years of felid evolution. This model is then used to score each of the 83k jaguar SNPs: positions where the alternate allele is biologically "surprising" in the felid context are more likely to be under functional constraint — and consequently more likely to carry biome-specific selective signal.

Because the constraint signal is derived from cross-species evolutionary history rather than from the small jaguar sample itself, VES-based selection does not degrade with small population sizes. Selecting the top ~3,000–3,500 loci by this score reduces the feature space to a size the available data can support. The resulting genotype matrix is then used to train a Locator/GeoGenIE-style MLP that predicts geographic coordinates and biome labels from 55 individuals across five Brazilian biomes.

Beyond jaguar-specific geographic assignment, the felid foundation model is a reusable asset: the same pretrained checkpoint can compute VES scores for any felid species with a VCF, enabling label-free locus selection for other small-sample conservation genomics studies across the Felidae family.

## Methodology Overview

### Stage 1: Felid Foundation Pre-training

The first stage builds a multi-species felid genomic corpus and trains DNABERT-2 via continued masked language modeling (MLM). This stage operates exclusively on reference assembly FASTA files.

**Pipeline flow:**

1. **Acquire assemblies** — Download six approved felid reference FASTAs with checksum validation.
2. **Build tokenized corpus** — Stream one species at a time, window sequences into 512 bp segments, tokenize with the DNABERT-2 BPE tokenizer, and write Parquet shards.
3. **Run continued pre-training** — Load the tokenized corpus and train DNABERT-2 with masked language modeling.

**Key outputs:**
- `models/foundation_felid/best/` — Best checkpoint (lowest validation loss)

### Stage 2: Variant Effect Scoring

The second stage uses the felid-pretrained DNABERT-2 to compute a functional importance score for each SNP in the jaguar VCF. This is the transfer learning mechanism — cross-species genomic context informs which variants are in constrained vs. neutral regions.

**Algorithm:** For each biallelic SNP locus:

1. Extract a 512 bp window centered on the locus from the reference FASTA
2. Tokenize and mask the center token (the variant position)
3. Forward pass through the frozen DNABERT-2 backbone (using `AutoModelForMaskedLM`)
4. Extract predicted probabilities for the reference and alternate alleles
5. Compute: `VES = log P(alt | context) - log P(ref | context)`

**Interpretation:**
- **VES ≈ 0:** Both alleles equally likely — unconstrained region, probably neutral
- **VES << 0:** Alternate allele is very surprising — constrained region, variant likely under purifying selection
- **VES >> 0:** Alternate is more expected than reference — possibly the reference carries the derived allele

**Key output:**
- `ves_scores.pt` — One scalar per SNP locus (~83k scores)

### Stage 3: Genotype MLP Training

The third stage trains a Locator/GeoGenIE-style MLP on the genotype matrix (individuals × loci, values 0/1/2), with VES-guided locus importance from the felid foundation model.

**Input representation:** A dense genotype matrix is constructed directly from the VCF. All genotypes are retained — including homozygous reference (0/0), which carries critical population-level information (the absence of a variant is as diagnostic as its presence).

**Architecture:** Optional learnable locus gate → BatchNorm → [Linear → ELU → Dropout] × L → coordinate head (2 outputs). When using the `"learnable"` VES mode, per-locus sigmoid gates initialized from VES scores modulate the genotype input before the MLP trunk — the foundation model provides a prior on locus importance that backpropagation refines during training.

**Loss function:** Differentiable haversine distance. The model predicts coordinates in Z-score normalized space; the loss denormalizes predictions to decimal degrees and computes mean great-circle distance against degree-space targets.

**Evaluation:** Leave-one-out cross-validation (55 folds) with optional Optuna Bayesian hyperparameter optimization (100 trials default). Fixed-hyperparameter mode (`optuna_n_trials = 0`) is available for controlled baseline comparisons.

**Key outputs:**
- `loocv_predictions.json` — Per-individual coordinate predictions with haversine errors and biome classifications
- `loocv_summary.json` — Aggregate metrics (haversine median/mean, distance thresholds, per-biome breakdown)
- `optuna_summary.json` — Best trial parameters (when Optuna is used)

---

## Data Decisions

### Pre-training Corpus: Species Selection and Rationale

The foundation corpus uses six felid reference assemblies spanning four genera across the Felidae family:

| Species | Common Name | Assembly | Source |
|---------|-------------|----------|--------|
| *Felis catus* | Domestic cat | Felis_catus_9.0 | NCBI RefSeq (GCF_000181335.3) |
| *Panthera leo* | Lion | P.leo_Ple1_pat1.1 | NCBI RefSeq (GCF_018350215.1) |
| *Panthera tigris* | Amur tiger | PanTig1.0 | NCBI RefSeq (GCF_000464555.1) |
| *Panthera onca* | Jaguar | Panthera_onca_HiC | DNA Zoo |
| *Puma concolor* | Puma | PumCon1.0 | NCBI RefSeq (GCF_003327715.1) |
| *Panthera pardus* | Leopard | PanPar1.0 | NCBI RefSeq (GCF_001857705.1) |

The species list is closed and pinned in code. Adding a species requires a code and test change.

### Sequence Windowing and Tokenization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Context window | 512 bp | Matches DNABERT-2's maximum position embedding length |
| Window overlap | 128 bp | Stride of 384 bp provides coverage of boundary regions while controlling corpus size |
| Max ambiguous fraction | 5% | Windows with >5% N bases carry minimal informative signal |
| Locus block size | 50 kb | Exceeds autocorrelation length of most local genomic features; ensures split safety |

Tokenization uses the DNABERT-2 BPE tokenizer (`zhihan1996/DNABERT-2-117M`), pinned to a specific Git revision for exact reproducibility. Allowed alphabet: {A, C, G, T, N}.

### Locus-Safe Train/Validation Splitting

Each 50 kb genomic block is assigned deterministically to train (80%) or validation (20%) via SHA-256 hash of its locus identifier. All windows within a block inherit its split assignment, preventing data leakage from overlapping windows.

### Genotype Matrix Construction

The genotype matrix is built directly from the VCF, representing each individual as a vector of allele counts at all biallelic SNP loci:

| Genotype | Encoding | Meaning |
|----------|----------|---------|
| 0/0 | 0 | Homozygous reference — retained (critical for population-level comparisons) |
| 0/1 | 1 | Heterozygous |
| 1/1 | 2 | Homozygous alternate |
| ./. | -1 | Missing data (imputed per-fold using training allele frequencies) |

**VCF filtering:** Only PASS or "." filter-status records are retained. Multi-allelic sites, indels, and spanning deletions are excluded. REF and ALT must each be a single nucleotide in {A, C, G, T}.

**Missing data imputation:** For each locus with missing data, the alternate allele frequency is computed from the training fold's non-missing individuals. Two Bernoulli draws at that frequency are summed to produce the imputed genotype (0, 1, or 2). This matches Locator's imputation contract and prevents data leakage across folds.

**Output:** `Int8Tensor` of shape `(n_individuals, n_loci)` — 55 rows × ~83,000 columns.

### Variant Effect Scoring

VES scores are computed once and cached. The computation requires:
- The felid-pretrained DNABERT-2 checkpoint (from Stage 1)
- The DNA Zoo jaguar reference FASTA
- The locus list from the genotype matrix (contig, position, ref/alt alleles)

The scoring uses `AutoModelForMaskedLM` (not `AutoModel`) to access the MLM logits head. The center token is identified via the tokenizer's offset mapping, masked, and the log-likelihood ratio between alternate and reference alleles is computed from the softmax output.

### Coordinate Normalization and Haversine Loss

The model predicts coordinates in Z-score normalized space: per LOOCV fold, latitude and longitude means and standard deviations are computed from the training individuals only (standard deviations clamped to 1e-6). This centers the model's output range around zero for optimizer-friendly gradients.

The loss function operates in degree space. During training, model predictions are denormalized back to decimal degrees via `pred_deg = pred_z × std + mean`, then the mean haversine (great-circle) distance to degree-space targets is computed. The raw haversine (km) is scaled by 1/1000 (converting to megameters) for gradient stability — this places the coordinate loss on a comparable scale to the cross-entropy classification loss.

The haversine function accounts for the cos(latitude) factor that makes 1° longitude vary from ~111 km at the equator to ~100 km at latitude -25°, which a Euclidean loss in Z-score space cannot capture.

---

## Modeling Decisions

### Foundation Model Selection: DNABERT-2

DNABERT-2 (Zhou et al., 2024) is a 117M-parameter transformer-based genomic language model. Key architectural features:

- **Byte-pair encoding (BPE) tokenization** over fixed k-mer approaches
- **Attention with Linear Biases (ALiBi)** for positional encoding
- **Flash Attention** for memory-efficient self-attention
- Pinned to HuggingFace revision `7bce263b15377fc15361f52cfab88f8b586abda0`

### Pre-training Objective and Hyperparameters

The foundation model is trained with **masked language modeling (MLM)**: 15% of tokens are randomly masked per batch, and the model learns to predict the original tokens from their surrounding context. This forces the transformer to build internal representations of nucleotide composition, sequence motifs, and longer-range genomic dependencies across felid species. Masking is dynamic — a new random mask is computed for every batch.

**Training hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max training steps | 50,000 | Safety ceiling only; training normally stops earlier via early stopping |
| Early stopping patience | 5 eval cycles | Halt if validation loss does not improve for 5 consecutive eval cycles |
| Learning rate | 5e-5 | Conservative LR for continued pre-training (lower than scratch training) |
| LR schedule | Cosine annealing with linear warmup | Linear ramp over 500 steps, then cosine decay to near-zero |
| Weight decay | 0.01 | Applied to all parameters except bias and LayerNorm weights |
| Gradient clipping | 1.0 | Max L2 norm to prevent gradient explosions |
| Per-device train batch size | 32 | Before gradient accumulation |
| Gradient accumulation steps | 2 | Effective batch = 32 × 2 × world_size |
| MLM probability | 0.15 | Standard 15% masking rate |
| Max sequence length | 512 tokens | Matches the context window size |
| Mixed precision | BF16 | Bfloat16 prevents gradient overflow while maintaining speed on Ampere/Ada GPUs |
| Evaluation frequency | Every 1,000 steps | Validation loss triggers best-checkpoint saves and early-stopping checks |
| Max eval steps | 500 | Caps each eval cycle duration; post-loop reduce safely handles uneven per-rank shards |

**Optimizer:** AdamW with decoupled weight decay. Bias and LayerNorm parameters are excluded from weight decay (weight decay = 0.0) following standard transformer training conventions.

**Pad token handling:** DNABERT-2 ships without an explicit pad token. A three-tier fallback assigns a pad token: (1) reuse `eos_token` if available (preferred), (2) reuse `unk_token`, (3) inject a new `[PAD]` token and resize embeddings.

### Genotype MLP Architecture

The geographic assignment model is a GeoGenIE-style MLP operating on genotype vectors:

```
Input: genotype vector (n_loci,) values in {0, 1, 2}
  → [Optional] Learnable locus gate: x = genotypes × sigmoid(gate)
  → BatchNorm1d
  → [Linear → ELU → Dropout] × L hidden layers
  → Coordinate head: Linear(hidden_dim, 2) → (lat_z, lon_z)
  → Biome head: Linear(hidden_dim, 5) → logits  [disabled when cls_loss_weight=0]
```

**Learnable locus gate:** When `ves_mode="learnable"`, a per-locus sigmoid gate is applied before BatchNorm. The gate parameter vector has shape `(n_loci,)` — one scalar per SNP — initialized from VES scores via z-scored log-transform: `logit = 2 × (log|VES| - mean) / std`. This maps `sigmoid(logit)` to approximately [0.02, 0.98], so high-|VES| loci (functionally constrained in the felid context) start with gates near 1, while low-|VES| loci start near 0. During training, backpropagation refines these gates — the foundation model provides a prior on locus importance, and the geographic labels provide the task-specific signal to adjust it. The gate adds ~83,000 learnable scalar parameters, negligible compared to the MLP trunk.

**Overparameterization guard (from GeoGenIE):** If `hidden_dim > n_input_features × 10`, the width is reduced by 20% recursively until compliant.

**Hyperparameter optimization:** Optuna (TPE sampler) tunes architecture and training hyperparameters. Each trial runs full LOOCV (55 folds), minimizing median haversine distance. Default budget: 100 trials.

### VES Integration Strategies

Four modes, controlled by config (`ves_mode`):

| Mode | Transform | Use case |
|------|-----------|----------|
| `"learnable"` | Per-locus sigmoid gates initialized from VES, refined by backprop | Foundation model provides prior on locus importance; training refines it |
| `"weighted"` | Multiply genotypes × \|VES\| | All loci retained, constrained loci contribute more to the input signal |
| `"selection"` | Keep top-K loci by \|VES\| | Hard feature selection — reduces dimensionality from ~83k to ~3k loci |
| `"none"` | Raw genotype vector | Baseline (no transfer learning from the felid foundation model) |

The `"learnable"` mode is recommended: it retains all loci, uses the foundation model's evolutionary signal as initialization, and allows the training objective to refine which loci matter for geographic assignment. The `"weighted"` mode achieved the best results in early experiments (146 km median haversine) before learnable gates were introduced. The `"none"` mode serves as the no-transfer-learning control.

### Loss Functions and Task Weighting

```
total_loss = coord_loss_weight × Haversine(denorm(pred_z), target_deg) / 1000
           + cls_loss_weight × CrossEntropy(biome_logits, biome_label)
```

| Component | Function | Default Weight | Notes |
|-----------|----------|----------------|-------|
| Coordinate regression | Mean haversine distance | 1.0 | Denormalizes Z-score predictions to degrees, computes great-circle distance (km), scales by 1/1000 |
| Biome classification | Cross-entropy | 0.0 | Set to 0 for coordinate-only mode; biome can be derived post-hoc from predicted coordinates |

The `/1000` scaling converts from km to megameters, keeping the loss magnitude comparable to the cross-entropy term (~0.2–0.8 range) for balanced gradient flow.

Setting `cls_loss_weight = 0.0` disables the biome classification head entirely, allowing the coordinate regression head to receive 100% of the gradient signal. Biome assignment can be recovered post-hoc from predicted coordinates using spatial polygon lookup (e.g., IBGE biome shapefiles), since the geographic predictions are precise enough to identify biome membership.

### Evaluation Metrics

**Primary metric:** Median haversine distance (km) across all LOOCV folds. This is the Optuna optimization target and the main comparison metric against Zenato Lazzari et al. (2025).

**Distance thresholds (aligned with Zenato Lazzari 2025):**
- Percentage of individuals assigned within 250 km of true origin
- Percentage within 500 km (the paper's primary threshold: 65–69%)
- Percentage within 1,000 km

**Classification:** Accuracy, per-class F1, macro F1 (when biome head is active).

**Per-biome breakdown:** Median and mean haversine distance per true biome, enabling direct comparison with the paper's per-biome SCAT results (Amazon: 708 km, Atlantic Forest: 125 km, Caatinga: 80 km, Cerrado: 493 km, Pantanal: 196 km).

**Diagnostics (logged per evaluation):**
- Per-class prediction counts (detects classification head collapse)
- Per-individual haversine error (full audit trail for outlier analysis)
- Per-biome accuracy (when biome head is active)

### No-VES Tuned Baseline

To isolate the transfer learning contribution, the pipeline includes a no-VES baseline that uses the same Optuna optimization budget (100 trials), the same MLP architecture search space, the same LOOCV protocol, and the same haversine loss — but operates on raw genotypes with `ves_mode="none"`. This ensures that any performance gap between the baseline and VES-guided models is attributable to the felid foundation model's locus importance signal, not to differences in hyperparameter tuning.

---

## Reproducibility and Integrity Guarantees

- **Immutable tokenizer pinning:** DNABERT-2 tokenizer locked to a specific Git commit hash.
- **Checksum-verified downloads:** All assemblies and jaguar raw data verified with pinned SHA-256 checksums.
- **Atomic checkpoint writes:** Temporary files with atomic rename prevent corruption from mid-write crashes.
- **Deterministic split assignment:** SHA-256 hashes of locus identifiers produce identical splits across runs.
- **Frozen configuration:** All config dataclasses are immutable after loading.
- **Sequence integrity:** SHA-256 hashes of processed windows stored for post-hoc auditing.

---

## Installation

### Requirements

- Python >=3.11, <3.12
- `uv` package manager
- Optional: `optuna` for hyperparameter optimization

### Setup

1. Sync the environment:
   ```bash
   uv sync
   ```

2. Verify the CLI:
   ```bash
   uv run python -m jaguar_geo_assign.cli --help
   ```

---

## Running the Pipeline

### Step 1: Acquire Felid Reference Assemblies

Download the six approved felid reference FASTAs with integrity verification:

```bash
uv run python -m jaguar_geo_assign.cli validate-felid-foundation-config \
  configs/examples/felid_foundation_pretrain.toml

uv run python -m jaguar_geo_assign.cli acquire-felid-foundation-assemblies \
  configs/examples/felid_foundation_pretrain.toml
```

### Step 2: Build the Tokenized Corpus

Construct the windowed, tokenized Parquet corpus from the reference assemblies:

```bash
uv run python -m jaguar_geo_assign.cli felid-foundation-pretrain \
  configs/examples/felid_foundation_pretrain.toml
```

### Step 3: Run Foundation Pre-training

Train DNABERT-2 with masked language modeling on the felid corpus:

```bash
# Single-GPU
uv run python -m jaguar_geo_assign.cli train-felid-foundation \
  --config configs/examples/felid_foundation_train.toml

# Multi-GPU (example: 8 GPUs)
uv run accelerate launch --multi_gpu --num_processes 8 \
  -m jaguar_geo_assign.cli train-felid-foundation \
  --config configs/examples/felid_foundation_train.toml
```

Best checkpoint is saved to `models/foundation_felid/best/hf_model/`.

### Step 4: Acquire Jaguar Raw Data

Download the jaguar VCF and location CSV:

```bash
uv run python -m jaguar_geo_assign.cli acquire-jaguar-raw-data
```

Downloads to `data/raw/`:
- `jaguar.57samples.allChr.snps.hardFilter.bi.maf.ld.masked.hwe.recode.vcf` (147 MB) — 57 jaguar samples, biallelic SNPs
- `jaguar_location.csv` — sample metadata (sample_id, individual_id, latitude, longitude, biome_population_label)

### Step 5: Train the Genotype MLP

This single command handles genotype matrix construction, VES scoring, and MLP training with LOOCV:

```bash
# Recommended: learnable VES gates + haversine loss
uv run python -m jaguar_geo_assign.cli genotype-finetune \
  --config configs/examples/genotype_finetune_learnable_haversine.toml

# No-VES tuned baseline (same Optuna budget, raw genotypes)
uv run python -m jaguar_geo_assign.cli genotype-finetune \
  --config configs/examples/genotype_finetune_no_ves_tuned.toml

# VES-weighted mode (all loci weighted by |VES|)
uv run python -m jaguar_geo_assign.cli genotype-finetune \
  --config configs/examples/genotype_finetune_weighted.toml
```

**What happens under the hood:**

1. **Genotype matrix** is built from the VCF (cached to `genotype_cache_dir` for reuse)
2. **VES scores** are computed using the felid-pretrained backbone (cached alongside the genotype matrix; skipped when `ves_mode="none"`)
3. **VES integration** is applied according to the configured mode: learnable gates, static weighting, hard selection, or none
4. **LOOCV training** — when `optuna_n_trials > 0`, Optuna runs N trials, each performing full 55-fold LOOCV minimizing median haversine distance, then averages predictions from the top-5 trials (ensemble). When `optuna_n_trials = 0`, a single LOOCV run uses the config-specified hyperparameters directly.
5. Final predictions, metrics, and hyperparameters are saved to `output_dir`

**Outputs:**
- `loocv_predictions.json` — all per-individual predictions with true/predicted coordinates, haversine errors, and biome classifications
- `loocv_summary.json` — aggregate metrics, per-biome breakdown, distance thresholds
- `optuna_summary.json` — best trial parameters (when Optuna is used)
- `tensorboard/` — training curves

---

## Repository Layout

```
src/jaguar_geo_assign/
├── cli.py                          # CLI entry points
├── config.py                       # Typed config loaders and contract enforcement
├── data/
│   ├── felid_assemblies.py         # Approved felid assembly registry (6 species)
│   ├── felid_acquisition.py        # Assembly download with checksum verification
│   ├── jaguar_raw_data.py          # Jaguar VCF + location CSV registry
│   ├── jaguar_raw_acquisition.py   # Jaguar raw data download
│   ├── finetune_windows.py         # Jaguar VCF → 512 bp locus-centered windows
│   ├── consensus.py                # VCF parsing helpers
│   ├── preprocessor.py             # Core preprocessing pipeline
│   ├── tokenized_corpus_reader.py  # Parquet corpus reader for training
│   ├── acquisition.py              # Download primitives with retry/checksum
│   ├── contracts.py                # Shared architecture constants
│   └── pipeline_contract.py        # Contract validation helpers
├── pretrain/
│   ├── felid_foundation_pipeline.py  # Multi-species corpus construction
│   ├── foundation_training.py        # DNABERT-2 continued MLM pre-training
│   └── _shared.py                    # Cross-pipeline helpers
├── fine_tune/
│   ├── genotype_dataset.py         # VCF → genotype matrix (0/1/2) construction
│   ├── variant_scoring.py          # DNABERT-2 masked prediction → VES scores
│   ├── genotype_model.py           # Genotype MLP architecture + VES helpers
│   ├── genotype_trainer.py         # LOOCV + Optuna training loop
│   ├── model.py                    # Shared coordinate/biome head definitions
│   ├── trainer.py                  # Shared loss/metric helpers
│   └── dataset.py                  # Shared data constants (BIOME_CLASSES, CoordStats)

configs/examples/
├── felid_foundation_pretrain.toml             # Corpus construction configuration
├── felid_foundation_train.toml                # Foundation training hyperparameters
├── genotype_finetune.toml                     # Genotype MLP + VES selection mode
├── genotype_finetune_weighted.toml            # VES-weighted mode (all loci × |VES|)
├── genotype_finetune_weighted_no_biome.toml   # Biome head ablation (weighted, no biome loss)
├── genotype_finetune_learnable_haversine.toml # Learnable VES gates + haversine loss
├── genotype_finetune_no_ves_tuned.toml         # No-VES tuned baseline (same Optuna budget)
└── fine_tune.toml                             # Fine-tuning experiment bootstrap config

design-logs/
└── option-a-ves-genotype-architecture.md  # Full architecture specification

dev_docs/
└── pipeline_diagnosis_and_plan.md  # Root-cause analysis of MIL pipeline failure
```

---

## Development

### Running Tests

```bash
# Unit tests only (default)
uv run pytest

# Include integration tests (requires network access)
uv run pytest -m integration
```

### Code Quality

The project uses `ruff` for linting and formatting (target: Python 3.11, line length: 100).

### Key Dependencies

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework |
| transformers | DNABERT-2 model and tokenizer loading |
| accelerate | Distributed training, mixed precision |
| pyarrow | Parquet corpus I/O |
| scikit-learn | StratifiedGroupKFold cross-validation |
| tensorboard | Training metrics visualization |
| beartype + jaxtyping | Runtime type and shape checking |
| optuna | Hyperparameter optimization (optional) |
