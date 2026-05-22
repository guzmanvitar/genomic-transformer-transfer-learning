# Jaguar Geographic Assignment via DNABERT-2 Transfer Learning

A two-stage transfer-learning pipeline for geographic assignment of jaguars (*Panthera onca*) from whole-genome sequencing data. The system pre-trains DNABERT-2 on multi-species felid reference genomes, then fine-tunes a multi-task model that jointly predicts geographic coordinates and biome-population labels from jaguar variant data.

This work addresses a core challenge in conservation genomics: endangered species that most need precise forensic tools have the least genetic data available for model development. By pre-training on abundant felid genomic resources and transferring that knowledge to jaguar-specific tasks, the pipeline can extract informative geographic signal from limited sample sizes across fragmented populations in Brazilian biomes.

---

## Table of Contents

1. [Research Context](#research-context)
2. [Methodology Overview](#methodology-overview)
   - [Stage 1: Felid Foundation Pre-training](#stage-1-felid-foundation-pre-training)
   - [Stage 2: Jaguar Multi-Task Fine-tuning](#stage-2-jaguar-multi-task-fine-tuning)
3. [Data Decisions](#data-decisions)
   - [Pre-training Corpus: Species Selection and Rationale](#pre-training-corpus-species-selection-and-rationale)
   - [Sequence Windowing and Tokenization](#sequence-windowing-and-tokenization)
   - [Locus-Safe Train/Validation Splitting](#locus-safe-trainvalidation-splitting)
   - [Jaguar Variant Processing](#jaguar-variant-processing)
   - [Coordinate Normalization and Sample Weighting](#coordinate-normalization-and-sample-weighting)
4. [Modeling Decisions](#modeling-decisions)
   - [Foundation Model Selection: DNABERT-2](#foundation-model-selection-dnabert-2)
   - [Pre-training Objective and Hyperparameters](#pre-training-objective-and-hyperparameters)
   - [Multi-Task Architecture](#multi-task-architecture)
   - [Two-Phase Fine-tuning Schedule](#two-phase-fine-tuning-schedule)
   - [Loss Functions and Task Weighting](#loss-functions-and-task-weighting)
   - [Evaluation Metrics](#evaluation-metrics)
5. [Reproducibility and Integrity Guarantees](#reproducibility-and-integrity-guarantees)
6. [Installation](#installation)
7. [Running the Pipeline](#running-the-pipeline)
   - [Step 1: Acquire Felid Reference Assemblies](#step-1-acquire-felid-reference-assemblies)
   - [Step 2: Build the Tokenized Corpus](#step-2-build-the-tokenized-corpus)
   - [Step 3: Run Foundation Pre-training](#step-3-run-foundation-pre-training)
   - [Step 4: Acquire Jaguar Raw Data](#step-4-acquire-jaguar-raw-data)
   - [Step 5: Prepare Jaguar Fine-tuning Data](#step-5-prepare-jaguar-fine-tuning-data)
   - [Step 6: Run Multi-Task Fine-tuning](#step-6-run-multi-task-fine-tuning)
8. [Repository Layout](#repository-layout)
9. [Development](#development)

---

## Research Context

The jaguar occupies approximately 46% of its historical range, with an estimated 173,000 individuals remaining. Population status varies sharply across Brazilian biomes: the Amazon and Pantanal harbor the largest and most genetically diverse populations, while the Atlantic Forest and Caatinga populations are small, isolated, and genetically impoverished due to habitat fragmentation. A critical gap in conservation is the ability to assign poached individuals to their population of origin, which would allow authorities to identify and prioritize anti-poaching interventions.

Traditional approaches to geographic assignment (SCAT, SPASIBA, STRUCTURE, KLFDAPC) require substantial sample sizes per population and either produce only coarse population-level assignments or demand dense spatial sampling for continuous predictions. Machine learning methods such as Locator (Battey et al., 2020) achieve high-resolution continuous geographic assignment via deep neural networks, but still require large training datasets.

This project applies transfer learning to overcome the data scarcity challenge. A DNABERT-2 genomic language model is pre-trained from scratch on six felid reference assemblies to learn the general "grammar" of felid DNA, then fine-tuned on jaguar whole-genome variants from five Brazilian biomes (Amazon, Atlantic Forest, Caatinga, Cerrado, Pantanal). The transfer-learning approach enables the extraction of informative patterns even from populations with very few sampled individuals.

## Methodology Overview

### Stage 1: Felid Foundation Pre-training

The first stage builds a multi-species felid genomic corpus and trains DNABERT-2 via continued masked language modeling (MLM). This stage operates exclusively on reference assembly FASTA files and does not involve any VCF processing.

**Pipeline flow:**

1. **Acquire assemblies** — Download six approved felid reference FASTAs with checksum validation.
2. **Build tokenized corpus** — Stream one species at a time, window sequences into 512 bp segments, tokenize with the DNABERT-2 BPE tokenizer, and write Parquet shards.
3. **Run continued pre-training** — Load the tokenized corpus and train DNABERT-2 with masked language modeling.

**Key outputs:**
- `data/raw/felid_foundation/reference/` — Downloaded FASTAs
- `data/processed/felid_foundation_pretrain/felid_foundation_tokens/` — Tokenized Parquet corpus
- `artifacts/felid_foundation_pretrain/felid_foundation_pretrain_run_summary.json` — Corpus summary
- `models/foundation_felid/best/` — Best checkpoint (lowest validation loss)

### Stage 2: Jaguar Multi-Task Fine-tuning

The second stage fine-tunes the pre-trained DNABERT-2 backbone on jaguar variant data for two simultaneous tasks:

- **Coordinate regression**: Predict latitude and longitude of geographic origin.
- **Biome-population classification**: Assign individuals to one of five Brazilian biomes.

This stage processes jaguar VCF files against the DNA Zoo *Panthera onca* reference to extract 512 bp locus-centered windows around variant sites, then trains a multi-task model with task-specific heads attached to the shared backbone.

**Key outputs:**
- JSONL of per-locus 512 bp windows with allele annotations
- `models/finetune/best/` — Best checkpoint, including backbone, task heads, and coordinate normalization parameters

---

## Data Decisions

### Pre-training Corpus: Species Selection and Rationale

The foundation corpus uses six felid reference assemblies spanning four genera across the Felidae family. The selection maximizes phylogenetic coverage within Felidae while focusing on species with high-quality publicly available reference genomes:

| Species | Common Name | Assembly | Source |
|---------|-------------|----------|--------|
| *Felis catus* | Domestic cat | Felis_catus_9.0 | NCBI RefSeq (GCF_000181335.3) |
| *Panthera leo* | Lion | P.leo_Ple1_pat1.1 | NCBI RefSeq (GCF_018350215.1) |
| *Panthera tigris* | Amur tiger | PanTig1.0 | NCBI RefSeq (GCF_000464555.1) |
| *Panthera onca* | Jaguar | Panthera_onca_HiC | DNA Zoo |
| *Puma concolor* | Puma | PumCon1.0 | NCBI RefSeq (GCF_003327715.1) |
| *Panthera pardus* | Leopard | PanPar1.0 | NCBI RefSeq (GCF_001857705.1) |

Including the jaguar reference itself in the pre-training corpus ensures the model sees the target species' genome-wide context before fine-tuning. The domestic cat (*Felis catus*) provides the best-annotated felid genome. The three other *Panthera* species (lion, tiger, leopard) represent the closest phylogenetic relatives to the jaguar. Puma adds an outgroup within the Felidae family.

The species list is closed and pinned in code. Adding a species requires a code and test change, preventing the foundation corpus from becoming an untracked mixture.

### Sequence Windowing and Tokenization

**Windowing parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Context window | 512 bp | Matches DNABERT-2's maximum position embedding length, maximizing the sequence context visible to the model per forward pass |
| Window overlap | 128 bp | Stride of 384 bp provides sufficient coverage of boundary regions while controlling corpus size |
| Max ambiguous fraction | 5% | Windows with more than 5% N bases are discarded; these carry minimal informative signal and introduce noise |
| Short sequence filter | Enabled | Sequences shorter than 512 bp are dropped to ensure uniform window dimensions |

**Locus block alignment:** Before windowing, each contig is partitioned into 50 kb (50,000 bp) blocks. Windows are then aligned within blocks, ensuring that no window ever spans a block boundary. This is critical for split safety — the train/validation split operates at the block level, so block alignment guarantees that overlapping windows from adjacent genomic positions land in the same split. The 50 kb block size is chosen to exceed the autocorrelation length of most local genomic features (GC content, repeat elements, recombination rate), ensuring that windows near the boundary of a training block and windows near the boundary of an adjacent validation block are separated by enough genomic distance to be statistically independent. A smaller block size would create frequent split boundaries where nearly identical genomic contexts appear in both splits, inflating validation performance.

**Tokenization:** Sequences are tokenized using the DNABERT-2 BPE tokenizer (`zhihan1996/DNABERT-2-117M`), pinned to a specific Git revision for exact reproducibility. The BPE scheme reduces redundancy compared to fixed k-mer tokenization, producing more compact representations while naturally handling variable motif lengths. The allowed alphabet is strictly {A, C, G, T, N}; sequences containing out-of-alphabet symbols are rejected.

### Locus-Safe Train/Validation Splitting

A locus-block-based splitting strategy prevents data leakage between train and validation sets. Because overlapping windows from nearby genomic positions share most of their sequence content, a naive random split would leak training signal into evaluation.

**Strategy:** Each 50 kb genomic block is assigned deterministically to either train (80%) or validation (20%) via a SHA-256 hash of its locus identifier (`contig:block_start-block_end`). All windows within a block inherit its split assignment. This ensures:

- No window in the validation set overlaps with any training window.
- The split is deterministic across runs (same locus identifiers always hash to the same split).
- The evaluation target is "unseen loci" — the validation set tests the model's ability to generalize to novel genomic regions, not memorize training regions.

### Jaguar Variant Processing

Jaguar fine-tuning data is derived from VCF files aligned to the DNA Zoo *Panthera onca* HiC assembly. The window extraction module produces 512 bp locus-centered windows around each variant site:

**Window geometry:**
- 256 bp upstream + 1 bp center locus + 255 bp downstream = 512 bp total
- Windows that would extend beyond contig boundaries are discarded (no padding)

**VCF filtering:**
- Only PASS or "." filter-status records are retained.
- Multi-allelic sites, indels, and spanning deletions are excluded; only biallelic single-nucleotide substitutions are processed.
- The nucleotide alphabet is restricted to {A, C, G, T, N} — IUPAC ambiguity codes are rejected.

**Genotype handling:**
- **Homozygous reference (0/0):** Dropped entirely. These loci carry no allelic signal relative to the reference and would dilute the training corpus.
- **Homozygous alternate (1/1):** One window emitted with the alternate allele placed at the center position.
- **Heterozygous (0/1):** Doubled into two windows — one with the reference allele and one with the alternate allele at the center. Both carry an `is_heterozygous` flag. This design preserves both haplotype contributions rather than losing the locus to ambiguity masking.

**Reference validation:** The pipeline enforces that the FASTA reference matches the DNA Zoo jaguar assembly by checking for positive contig tokens (`HiC_scaffold_1`, `Panthera_onca_HiC`) and rejecting NCBI-specific tokens (`NC_083295.1`, `GCF_028533385.1`) that indicate a RefSeq repackaging with altered contig names. Every VCF REF allele is verified against the actual FASTA base at that position.

### Coordinate Normalization and Sample Weighting

**Coordinate normalization:** Latitude and longitude targets are z-score normalized using per-individual mean and standard deviation computed from the training split. Normalization is computed at the individual level (not per-window) because the training sampler equalizes window contributions per individual — computing statistics per-window would bias the normalization toward individuals with more windows. Standard deviations are clamped to a minimum of 1e-6 to guard against division by zero. Normalization parameters are serialized to JSON alongside the best checkpoint for inference-time denormalization.

**Per-individual weighting:** Each training window receives a sampling weight inversely proportional to the number of windows from its individual: `weight = 1 / windows_per_individual`. A `WeightedRandomSampler` with replacement ensures that each individual contributes equally to each epoch regardless of how many variant sites they carry, preventing individuals with more variants from dominating training.

**Cross-validation:** `StratifiedGroupKFold` from scikit-learn stratifies on biome-population label and groups by individual identity. This guarantees:
- All biomes are represented in every fold.
- No individual appears in both training and evaluation within the same fold.
- Each biome must have at least `n_folds` unique individuals (validated at construction time).

---

## Modeling Decisions

### Foundation Model Selection: DNABERT-2

DNABERT-2 (Zhou et al., 2024) is a transformer-based genomic language model with 117M parameters, designed for multi-species genome understanding. It was selected for several architectural features:

- **Byte-pair encoding (BPE) tokenization** over traditional fixed k-mer approaches, reducing vocabulary redundancy and enabling more compact sequence representations.
- **Attention with Linear Biases (ALiBi)** for positional encoding, which allows the model to extrapolate to sequence lengths beyond those seen during training, unlike learned positional embeddings.
- **Flash Attention** integration for memory-efficient and computationally fast self-attention computation.
- **Multi-species pre-training capability**, making it suitable for learning generalizable genomic patterns from the felid foundation corpus.

The tokenizer and model are pinned to a specific HuggingFace revision (`7bce263b15377fc15361f52cfab88f8b586abda0`) to ensure exact reproducibility.

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

### Multi-Task Architecture

The fine-tuning model wraps the pre-trained DNABERT-2 backbone with two lightweight task-specific heads:

```
DNABERT-2 Backbone (shared encoder, 117M parameters)
  │
  ├── Pooling (CLS token or masked mean)
  │
  ├─→ Coordinate Regression Head
  │     Dropout(0.1) → Linear(hidden → hidden) → GELU → Dropout(0.1) → Linear(hidden → 2)
  │     Output: (latitude, longitude) predictions
  │
  └─→ Biome Classification Head
        Dropout(0.1) → Linear(hidden → hidden) → GELU → Dropout(0.1) → Linear(hidden → 5)
        Output: unnormalized logits over 5 biome classes
```

**Pooling strategy:** By default, CLS-token pooling is used (the backbone's `pooler_output` or the first-token hidden state). An alternative mean-pooling strategy computes a masked average over the sequence dimension using the attention mask, with denominators clamped to at least 1 to prevent division by zero.

**Head design:** Both heads use identical two-layer MLP architectures with GELU activation. This design keeps the majority of model capacity in the shared backbone while providing sufficient representational power for each task. The heads emit raw predictions: coordinate values in normalized space, and unnormalized logits for classification. Loss computation is delegated to the trainer, keeping the model reusable across different training regimes.

**Biome classes:** The five target biomes are alphabetically sorted for determinism: Amazon, Atlantic Forest, Caatinga, Cerrado, Pantanal. Any sample with an unrecognized biome label is rejected at dataset construction time.

### Two-Phase Fine-tuning Schedule

Fine-tuning follows a two-phase schedule designed to prevent catastrophic forgetting of pre-trained representations:

**Phase 1 — Heads-only warm-up (default: 1,000 steps):**
- The backbone is completely frozen (`requires_grad=False` on all backbone parameters).
- Only the coordinate regression and biome classification heads are trained.
- Learning rate for heads: 1e-4.
- This phase allows the task heads to calibrate to the backbone's representation space before any backbone parameters change.

**Phase 2 — Partial unfreezing (default: 3,000 steps):**
- The last 2 transformer blocks of the backbone are unfrozen (configurable: 2 or 3 blocks).
- The backbone's pooler layer is also unfrozen if present.
- Task heads continue training.
- **Differential learning rates:** The backbone uses a 10x lower learning rate (1e-5) than the heads (1e-4). This protects the deep pre-trained representations while allowing the top layers to adapt to the jaguar-specific task distribution.

Both phases use AdamW with cosine-annealing learning rate schedules and linear warmup over 10% of total phase steps.

### Loss Functions and Task Weighting

The total training loss is a weighted sum of classification and regression components:

```
total_loss = cls_loss_weight × CrossEntropy(biome_logits, biome_label)
           + reg_loss_weight × Huber(pred_coords, target_coords)
```

| Component | Function | Default Weight | Rationale |
|-----------|----------|----------------|-----------|
| Regression | Huber loss (delta=1.0) | 1.0 | The primary task and research contribution; more robust than MSE to outlier mispredictions — Huber transitions from L2 to L1 behavior beyond delta, limiting the influence of large geographic errors |
| Classification | Cross-entropy | 0.1 | Serves primarily as an auxiliary regularizer that encourages the shared backbone to learn population-structure-aware representations, indirectly benefiting regression |

The 1:10 weighting (regression-dominant) reflects two observations: (1) continuous geographic assignment is the hard task and the novel contribution of this work — it needs the majority of gradient budget from step 1; (2) biome classification saturates early, and a high classification weight would spend gradient budget sharpening already-correct logits rather than reducing geographic error. At 0.1 weight the classification head still converges to high accuracy (the task is easy enough), while the regression head receives first-class optimization throughout training. Both losses are computed in float32 regardless of mixed-precision settings to ensure numerical stability.

**Gradient management:**
- Gradient accumulation: 4 steps by default (effective batch size = per_device_batch × 4 × world_size).
- Gradient clipping: max L2 norm of 1.0.
- NaN/Inf guards: Steps with non-finite losses or gradients are skipped and counted as anomalies.

### Evaluation Metrics

**Classification metrics:**
- Accuracy: fraction of correctly predicted biome labels.
- Per-class F1 score for each of the five biomes.
- Macro F1: unweighted average of per-class F1 scores.

**Regression metrics:**
- Mean absolute error (MAE) in degrees for latitude and longitude separately.
- **Haversine distance (km):** Great-circle distance between predicted and true coordinates, computed via the standard Haversine formula with explicit float32 promotion and epsilon-clamping for numerical stability. Earth radius: 6,371 km.
- **Median Haversine distance:** Primary checkpoint selection metric. A lower median Haversine distance triggers a best-checkpoint save. When Haversine distance is tied, macro F1 serves as tie-breaker.

**Baseline comparisons:**
- Biome baseline: majority-class prediction from the training split.
- Coordinate baseline: zero vector in normalized space (the training mean), representing a naive "predict the centroid" strategy.

---

## Reproducibility and Integrity Guarantees

The pipeline enforces several reproducibility invariants:

- **Immutable tokenizer pinning:** The DNABERT-2 tokenizer is locked to a specific Git commit hash. Any model trained with this pipeline uses identical tokenization.
- **Checksum-verified downloads:** All reference assemblies and jaguar raw data files are verified with pinned SHA-256 checksums. Idempotent — second invocations skip already-verified files.
- **Atomic checkpoint writes:** All checkpoints use temporary files with atomic rename to prevent corruption from mid-write crashes. DDP-safe: rank-0 performs writes with failure broadcasting to prevent deadlocks.
- **Deterministic split assignment:** SHA-256 hashes of locus identifiers produce identical train/validation splits across runs.
- **Contig collision detection:** The corpus builder aborts if two species share contig names, preventing silent locus-identifier aliasing.
- **Frozen configuration:** All config dataclasses are immutable after loading. Validation contracts enforce parameter ranges and cross-field consistency at construction time.
- **Sequence integrity:** SHA-256 hashes of processed windows are stored alongside tokenized outputs in the Parquet corpus for post-hoc auditing.

---

## Installation

### Requirements

- Python >=3.11, <3.12
- `uv` package manager

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
# Validate the foundation config first
uv run python -m jaguar_geo_assign.cli validate-felid-foundation-config \
  configs/examples/felid_foundation_pretrain.toml

# Download assemblies (idempotent — skips already-verified files)
uv run python -m jaguar_geo_assign.cli acquire-felid-foundation-assemblies \
  configs/examples/felid_foundation_pretrain.toml
```

This downloads ~3 GB of compressed FASTA files to `data/raw/felid_foundation/reference/`. Each file is checksummed against pinned values on download.

### Step 2: Build the Tokenized Corpus

Construct the windowed, tokenized Parquet corpus from the reference assemblies:

```bash
# Optional: preview the configuration
uv run python -m jaguar_geo_assign.cli describe-felid-foundation-config \
  configs/examples/felid_foundation_pretrain.toml

# Build the corpus
uv run python -m jaguar_geo_assign.cli felid-foundation-pretrain \
  configs/examples/felid_foundation_pretrain.toml
```

Species are processed sequentially to keep peak memory bounded by the single largest assembly. Output is written to `data/processed/felid_foundation_pretrain/felid_foundation_tokens/` as Parquet files partitioned by split, contig, and block ID.

### Step 3: Run Foundation Pre-training

Train DNABERT-2 with masked language modeling on the felid corpus:

```bash
# Edit configs/examples/felid_foundation_train.toml to set corpus_metadata_path to the
# metadata.json written by Step 2. Its absolute path follows directly from processed_dir
# in felid_foundation_pretrain.toml:
#   corpus_metadata_path = "<repo_root>/data/processed/felid_foundation_pretrain/felid_foundation_tokens/metadata.json"

# Single-GPU training
uv run python -m jaguar_geo_assign.cli train-felid-foundation \
  --config configs/examples/felid_foundation_train.toml

# Multi-GPU training (example: 8 GPUs)
uv run accelerate launch --multi_gpu --num_processes 8 \
  -m jaguar_geo_assign.cli train-felid-foundation \
  --config configs/examples/felid_foundation_train.toml

# Quick integration test (verifies forward pass, optimizer step, checkpoint round-trip)
uv run python -m jaguar_geo_assign.cli train-felid-foundation \
  --config configs/examples/felid_foundation_train.toml --integration-test
```

Training outputs are saved to `models/foundation_felid/`. The best checkpoint (lowest validation MLM loss) is saved under `best/`, with the full HuggingFace model in `best/hf_model/` and tokenizer in `best/tokenizer/`. TensorBoard logs are written to the `tensorboard/` subdirectory.

Training resumes automatically from the latest checkpoint if one exists.

### Step 4: Acquire Jaguar Raw Data

Download the jaguar VCF and location CSV. Both files are hosted on HuggingFace as public datasets and can be fetched without credentials:

```bash
uv run python -m jaguar_geo_assign.cli acquire-jaguar-raw-data
```

This downloads to `data/raw/` by default:
- `jaguar.57samples.allChr.snps.hardFilter.bi.maf.ld.masked.hwe.recode.vcf` (147 MB) — hard-filtered, MAF/LD/HWE-cleaned SNPs for 57 jaguar samples; originally published on DataDryad (doi:10.5061/dryad.4tmpg4fkm, CC0)
- `jaguar_location.csv` — sample metadata with columns: `sample_id`, `individual_id`, `latitude`, `longitude`, `biome_population_label`

Both files are SHA-256 verified on download. Pass `--output-dir` to change the destination.

### Step 5: Prepare Jaguar Fine-tuning Data

Extract 512 bp locus-centered windows from jaguar VCF files. This step requires both files from Step 4 and the DNA Zoo *Panthera onca* HiC reference FASTA from Step 1:

```bash
uv run python -m jaguar_geo_assign.cli extract-finetune-windows \
  --reference-fasta data/raw/felid_foundation/reference/DNAZOO_Panthera_onca_HiC.fna.gz \
  --vcf data/raw/jaguar.57samples.allChr.snps.hardFilter.bi.maf.ld.masked.hwe.recode.vcf \
  --metadata-csv data/raw/jaguar_location.csv \
  --output-jsonl data/processed/finetune/windows.jsonl
```

The window extraction produces a JSONL file of `FinetuneWindow` records. Each record contains the 512 bp sequence, allele annotations, genotype, and genomic coordinates.

### Step 6: Run Multi-Task Fine-tuning

```bash
uv run python -m jaguar_geo_assign.cli fine-tune \
  --config configs/mtl_finetune.toml

# Quick smoke test with synthetic data (no pretrained backbone needed)
uv run python -m jaguar_geo_assign.cli fine-tune \
  --config configs/mtl_finetune.toml --integration-test
```

The fine-tuning trainer requires a `MtlFinetuneConfig` TOML with the following fields:

```toml
[training]
backbone_path = "models/foundation_felid/best/hf_model"
windows_jsonl  = "<path to extracted windows JSONL>"
metadata_csv   = "<path to jaguar metadata CSV>"
output_dir     = "models/finetune"
```

The fine-tuning trainer runs two phases:
1. **Heads-only warm-up** (1,000 steps): backbone frozen, only task heads trained.
2. **Joint training** (3,000 steps): last 2 transformer blocks and task heads trained with differential learning rates.

The best checkpoint is selected by median Haversine distance (lower is better) with macro F1 as tie-breaker.

---

## Repository Layout

```
src/jaguar_geo_assign/
├── cli.py                          # Top-level CLI entry points
├── config.py                       # Typed config loaders and contract enforcement
├── data/
│   ├── felid_assemblies.py         # Approved felid assembly registry (6 species)
│   ├── felid_acquisition.py        # Assembly download with checksum verification
│   ├── jaguar_raw_data.py          # Jaguar VCF + location CSV registry with pinned SHA-256
│   ├── jaguar_raw_acquisition.py   # Jaguar raw data download with checksum verification
│   ├── finetune_windows.py         # Jaguar VCF → 512 bp locus-centered windows
│   ├── consensus.py                # VCF parsing helpers shared with fine-tuning
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
│   ├── dataset.py                  # Fold-aware dataset with per-individual weighting
│   ├── model.py                    # DNABERT-2 backbone + MTL heads
│   └── trainer.py                  # Two-phase training with Accelerate
├── baselines/                      # Baseline evaluation constants
├── evaluation/                     # Evaluation module (scaffold)
└── reporting/                      # Report generation (scaffold)

configs/examples/
├── felid_foundation_pretrain.toml  # Corpus construction configuration
├── felid_foundation_train.toml     # Foundation training hyperparameters
├── fine_tune.toml                  # Fine-tuning experiment bootstrap config
└── regression_transfer.toml        # Full transfer-learning pipeline config

tests/
├── test_*.py                       # Unit tests for all modules
└── integration/                    # End-to-end integration tests
```

---

## Development

### Running Tests

```bash
# Unit tests only (default, excludes integration tests)
uv run pytest

# Include integration tests (requires network access to NCBI and HuggingFace)
uv run pytest -m integration
```

### Code Quality

The project uses `ruff` for linting and formatting (target: Python 3.11, line length: 100).

### Key Dependencies

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework |
| transformers | DNABERT-2 model and tokenizer loading |
| accelerate | Distributed training, mixed precision, gradient accumulation |
| pyarrow | Parquet corpus I/O |
| scikit-learn | StratifiedGroupKFold cross-validation |
| tensorboard | Training metrics visualization |
| beartype + jaxtyping | Runtime type and shape checking |
