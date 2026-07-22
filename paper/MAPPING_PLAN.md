# CONTENT-MAPPING PLAN — Proposal → *Ecology and Evolution* Manuscript

**Purpose.** This is the execution blueprint for converting the master's-thesis
*proposal* into a completed-work journal manuscript in *Ecology and Evolution*
(Wiley) format. It is a PLAN only — no `.tex` is written yet.

## THE HARD RULE (repeat, for the executor)
The final manuscript may contain ONLY:
- **(a)** text copied **byte-identical** from the four proposal `.tex` files or
  `main.tex` (the *source files*), or
- **(b)** clearly-marked **PLACEHOLDERS** — meta-instructions to a human, never
  prose disguised as paper text, or
- **(c)** pure LaTeX / format scaffolding and standard structural headings.

You may **discard** freely. You may **NOT** paraphrase, merge, fix tense, or
invent transitions. If proposal text is inaccurate for completed work, it is
**discarded and replaced by a PLACEHOLDER**, never rewritten. `README.md` and the
reference PDF are **context only** — their words never enter the paper; they only
inform what a human must write inside a PLACEHOLDER.

**Source files (prose may come only from here):**
- `thesis/main.tex` — title, English abstract, keywords, author/advisor names
- `thesis/chapter-introduction.tex`  (cited below as **INTRO** L#)
- `thesis/chapter-background.tex`     (cited below as **BG** L#)
- `thesis/chapter-related-work.tex`   (cited below as **RW** L#)
- `thesis/chapter-proposal.tex`       (cited below as **PROP** L#)
- `paper/bibliography.bib` — already copied; 23 entries; author–year keys.

---

## PART 1 — TARGET OUTLINE (E&E research article)

Legend: `[REUSE]` = byte-identical proposal text/title available (see Part 2);
`[PH]` = placeholder (Part 4); `[SCAF]` = format scaffolding / standard heading.

### FRONT MATTER
- **Title** `[REUSE]` — main.tex L31 (English title)
- **Author list** — "Guzman Vitar" `[REUSE main.tex L24]`; any co-authors + order `[PH]`
- **Affiliations** `[PH]`
- **Correspondence** `[PH]`
- **Funding statement** `[PH]`
- **Abstract** `[PH]` (future-tense proposal abstract is available verbatim as raw
  material — main.tex L134 — but predates results, so it becomes a placeholder)
- **Keywords** `[REUSE main.tex L133]`, re-joined pipe-separated `[SCAF]`

### BODY (Wiley numbered headings, e.g. `1 | INTRODUCTION`)

**`1 | INTRODUCTION`** `[SCAF heading]`
- 1.1 "Jaguar Conservation" `[REUSE title INTRO L4]`
- 1.2 "Conservation Genomics and Machine Learning" `[REUSE title INTRO L23]`
- 1.3 "The Challenge of Limited Data in Endangered Species" `[REUSE title INTRO L33]`
- 1.4 "Traditional Statistical Methods for Geographic Assignment" `[REUSE title BG L4]`
- 1.5 "Deep Learning for Geographical Assignment & Genetic Data Representation" `[REUSE title RW L12]`
- 1.6 "The Promise of Transfer Learning in Conservation Genomics" `[REUSE title INTRO L39]`
- 1.7 "OBJECTIVES" `[REUSE title PROP L8]` (aims; flag future tense)

> Note for executor: seven subsections is thesis-like and long for an E&E intro.
> That is acceptable under the rule — the human may **discard** whole subsections
> (esp. 1.4/1.5 method-by-method literature) for length. Discarding is allowed;
> rewriting to condense is not.

**`2 | MATERIALS AND METHODS`** `[REUSE title PROP L24]`
- 2.1 "Foundation Model: DNABERT-2" `[REUSE title+text PROP L42–46]` (flag future tense; architecture is accurate)
- 2.2 Felid foundation pre-training corpus `[PH title + PH content]`
- 2.3 Jaguar samples & study populations `[PH title + PH content]`
- 2.4 Variant Effect Scoring (VES) `[PH]`
- 2.5 Genotype matrix construction `[PH]`
- 2.6 Genotype MLP & learnable locus gates `[PH]` (partial reuse of PROP L62 regression framing)
- 2.7 Loss function & coordinate normalization `[PH]`
- 2.8 Cross-validation & hyperparameter optimization `[PH]` (partial reuse LOOCV mention PROP L88)
- 2.9 Baseline models `[PH]` (Locator baseline; no-VES tuned baseline)
- 2.10 Evaluation metrics `[PH]` (optional partial reuse PROP L70)
- 2.11 Implementation & reproducibility `[PH]` (optional reuse software stack PROP L82)

**`3 | RESULTS`** `[SCAF heading]` — **NO proposal source; entirely `[PH]`**
- 3.1 Overall geographic assignment accuracy `[PH title + PH]`
- 3.2 Comparison with published methods (Zenato Lazzari et al. 2025) `[PH title + PH]`
- 3.3 Contribution of Bayesian hyperparameter optimization `[PH title + PH]`
- 3.4 Contribution of VES transfer learning `[PH title + PH]`
- 3.5 Per-biome breakdown `[PH title + PH]`
- Results tables/figures `[PH]` (build from actual metrics; README is data source only)

**`4 | DISCUSSION`** `[SCAF heading]` — **NO proposal source; entirely `[PH]`**

### BACK MATTER
- **Author Contributions** `[PH]` (CRediT)
- **Data Availability Statement** `[PH]`
- **Acknowledgements** `[PH]` (proposal `agradecimentos` is empty — main.tex L119–122)
- **Conflict of Interest** `[PH]`
- **References** `[SCAF]` — `\bibliography{bibliography}` via natbib author–year
- **Supporting Information** `[PH]` (optional)

---

## PART 2 — REUSE MAP (ordered, byte-identical copy targets)

Each entry: `ID | source L# | first ~8 words | notes`. Copy exactly, including
`\cite{...}` and `\textit{...}`. `⚠FT` = future-tense passage; reuse verbatim now,
human does a later tense pass. `~` = the `\section`/`\subsection` wrapper is
discarded but the *title string* is reused as an E&E subsection heading.

### → SECTION 1 INTRODUCTION

**1.1 Jaguar Conservation** (all present/factual — clean reuse)
- R1  INTRO L4  — title "Jaguar Conservation" ~
- R2  INTRO L6  — "The jaguar (Panthera onca) is the largest neotropical felid…"
- R3  INTRO L8  — "Population status varies greatly across biomes. The Amazon…"
- R4  INTRO L10 — "Habitat loss and fragmentation remain the most severe…"
- R5  INTRO L12–17 — FIGURE block `images/image5.png` (jaguar distribution map, `\label{fig:jaguar_distribution}`) — **KEEP image5.png**
- R6  INTRO L19 — "Human-wildlife conflict and illegal trade compound these pressures…"
- R7  INTRO L21 — "A critical gap in conservation is the ability…" (core forensic motivation)

**1.2 Conservation Genomics and Machine Learning**
- R8  INTRO L23 — title "Conservation Genomics and Machine Learning" ~
- R9  INTRO L25 — "A wide range of disciplines is involved in biodiversity…"
- R10 INTRO L27 — "Among the greatest contributions of genomics to conservation…"
- R11 INTRO L29 — "Machine learning (ML) is increasingly being integrated…"
- R12 INTRO L31 — "In addition, ML methods can be used to detect…"

**1.3 The Challenge of Limited Data in Endangered Species**
- R13 INTRO L33 — title "The Challenge of Limited Data in Endangered Species" ~
- R14 INTRO L35 — "A critical challenge in conservation genomics is the limited…"
- R15 INTRO L37 — "This limitation is particularly acute for jaguars, where…"
- R16 PROP  L4  — "The core challenge addressed by this project is the fundamental…" (data-scarcity paradox framing; strong intro closer for 1.3)

**1.4 Traditional Statistical Methods for Geographic Assignment**
- R17 BG L4  — title "Traditional Statistical Methods for Geographic Assignment" ~
- R18 BG L8  — "The scientific challenge of assigning individuals to their geographic…"
- R19 BG L10–14 — SCAT: subsubsection title + "The development of SCAT by Wasser and colleagues (2004)…" + "The strength of SCAT lies in its ability…"
- R20 BG L16–20 — SPASIBA: title + "Building on the ideas of SCAT, Guillot and colleagues (2016)…" + "SPASIBA has been applied successfully to a range…"
- R21 BG L22–30 — STRUCTURE: title + "STRUCTURE implements a Bayesian clustering framework…" + "In practice, STRUCTURE has been widely applied…" + "Despite its impact, STRUCTURE has limitations…" + "Nevertheless, STRUCTURE remains a cornerstone…"
- R22 BG L32–36 — KLFDAPC: title + "A different branch of approaches focuses on discriminating…" + "Qin and colleagues (2022) addressed this limitation with KLFDAPC…"
- R23 BG L38–58 — TABLE `tab:traditional_methods` (Comparison of traditional methods) — reusable as-is
- R24 BG L60–68 — Jaguar geographical assignment (Zenato Lazzari): subsubsection title + "In 2025, Zenato Lazzari and colleagues applied various…" + "The dataset comprised 58 whole genomes…" + "To assess performance, the authors used principal component…" + "Overall, the Jaguar SNP panel demonstrates how…" (KEY comparison paper — also referenced by Results/Discussion placeholders)

**1.5 Deep Learning for Geographical Assignment & Genetic Data Representation**
- R25 RW L6–10 — "Geographic assignment of individuals based on genetic data is an…" + "Traditional approaches, such as PCA or Bayesian clustering…" + "In recent years, machine learning methods, ranging from…"
- R26 RW L12 — title "Deep Learning for Geographical Assignment & Genetic Data Representation" ~
- R27 RW L14–28 — Battey/Locator: title + "Battey and colleagues (2020) set out to address…" + full description through "Despite these advances, the study highlighted two key limitations:" (Locator is the paper's own baseline — high value)
- R28 RW L30–40 — Degen: title + "Degen and colleagues (2025) investigated whether modern…" + full description
- R29 RW L42–52 — Bayliss: title + "A recent preprint by Bayliss et al. (2023) introduces a…" + full description
- R30 RW L78–98 — TABLE `tab:ml_methods_comparison` — reuse rows 1–3 (Battey/Degen/Bayliss); **row 4 "Our Proposal … Expected to improve…" is future/expected → drop the row or replace it with a `[PH]` cell** (see D-list)

**1.6 The Promise of Transfer Learning in Conservation Genomics**
- R31 INTRO L39 — title "The Promise of Transfer Learning in Conservation Genomics" ~
- R32 INTRO L41 — "Recent advances in machine learning, particularly in the field…"
- R33 INTRO L43 — "In the context of genomics, foundation models like DNABERT-2…"
- R34 BG L187–189 — "Recent developments in genomic machine learning have introduced foundation models…" (foundation models overview; §title "Genomic Foundation Models and Transfer Learning" can be dropped since it duplicates 1.6)
- R35 BG L203–205 — "The results have been striking. Both DNABERT-2 and the Nucleotide…" (impact of foundation models)
- R36 BG L208–213 — FIGURE block `images/image1.png` (transfer-learning illustration, `\label{fig:illustration_transfer_learning_and_finetuning}`) — **KEEP image1.png**
- *(OPTIONAL, borderline tutorial — reuse only if the human wants depth):*
  - R37 BG L191–195 — "Why Transformers?" title + "The shift to transformer architectures is central…" + "Transformers, in contrast, use self-attention mechanisms…"
  - R38 BG L197–201 — "Pretraining Tasks and Representations" + "Foundation models in genomics rely on self-supervised…" + "The Nucleotide Transformer takes this further…"

**1.7 OBJECTIVES** (aims — `⚠FT` infinitive/future)
- R39 PROP L8  — title "OBJECTIVES" ~ (and/or "General Objective" PROP L10, "Specific Objectives" PROP L14)
- R40 PROP L12 — "Develop and evaluate a machine learning pipeline using transfer learning…" (General Objective) `⚠FT`
- R41 PROP L14–22 — Specific Objectives `enumerate` (5 items) `⚠FT`
- R42 PROP L6  — "The innovative approach proposed here leverages the power of transfer…" `⚠FT` (proposal-voiced; optional bridge into aims)

### → SECTION 2 MATERIALS AND METHODS
- R43 PROP L24 — title "MATERIALS AND METHODS" ~ `[SCAF]`
- R44 PROP L42 — subsection title "Foundation Model: DNABERT-2" ~
- R45 PROP L44 — "We will employ the DNABERT-2 architecture as our foundation model…" `⚠FT` (accurate)
- R46 PROP L46 — "The model incorporates several advanced architectural features…" `⚠FT` (ALiBi + Flash Attention description; accurate, matches actual work)
- R47 PROP L62 — "The geographic assignment task will be formulated as a regression problem…" `⚠FT` (regression/lat-lon/Locator-like framing is accurate; the surrounding sentences PROP L64–66 are vague/partly wrong → discard, see D-list) — reuse into 2.6
- R48 PROP L88 — "Given the limited sample sizes, we will implement a comprehensive…" `⚠FT` — **reuse ONLY the leave-one-out cross-validation clause** ("…we will utilize leave-one-out cross-validation.") into 2.8; discard the stratified-k-fold / bootstrap clauses (not what was done — see D-list)
- R49 PROP L70 — "We will evaluate the model's geographic prediction performance using metrics…" `⚠FT` — OPTIONAL partial reuse into 2.10 (Median Error Distance + comparison-with-traditional-methods are accurate; MAE/R² are not the actual primary metrics → prefer `[PH]`)
- R50 PROP L82 — "The computational pipeline will utilize a robust software stack anchored…" `⚠FT` — OPTIONAL reuse into 2.11 (Python/PyTorch/Transformers is accurate)

> **All of Section 2 beyond the reuse items above is `[PH]`.** The proposal's
> concrete data/pretraining/fine-tuning descriptions are inaccurate for the
> completed work (see Discards D5–D9, D11–D12) and must be replaced, not reused.

### → SECTIONS 3 & 4
No reuse. Entirely placeholders (Part 4).

---

## PART 3 — DISCARD LIST (with reason)

| ID | Source | Reason |
|----|--------|--------|
| D1 | BG L70–185 (entire "Machine Learning and Deep Learning" §: What is ML, ML vs Statistics, Types of ML, Deep Learning, How Models are Trained, CNNs/RNNs, Challenges & Limitations) | ML-101 tutorial content; inappropriate for a research article |
| D2 | Figures image2 (LSTM, BG L160–165), image3 (ML categories, BG L93–98), image4 (MLP, BG L122–127), image6 (CNN, BG L149–154) | Tutorial figures; not results |
| D3 | RW L54–75 ("Our proposal: Transfer Learning…", incl. "pretraining from scratch", 128 bp windows, classification/regression on sequences) | Inaccurate vs actual work (continued pre-training from released DNABERT-2, 512 bp, VES + genotype-MLP not sequence fine-tuning); future tense |
| D4 | RW table `tab:ml_methods_comparison` row 4 "Our Proposal … Expected to improve…" | Expected-outcome/future content; replace with `[PH]` or drop row |
| D5 | PROP L28–32 (Pre-training Dataset: Darwin's Ark) | Inaccurate: actual corpus = 6 felid reference assemblies (NCBI RefSeq + DNA Zoo), not Darwin's Ark → `[PH]` |
| D6 | PROP L34–38 (Fine-tuning Dataset: "at least 45 … 10/13/5/12/5 per biome") | Inaccurate sample counts (actual = 55 genomes; per-biome counts differ) → `[PH]` |
| D7 | PROP L52–54 (Stage 1 pre-training: Darwin's Ark, batch 4096, 128 bp) | Inaccurate hyperparameters/source → `[PH]` |
| D8 | PROP L56–58 (Stage 2: replace MLM head with classification head, catastrophic forgetting) | Inaccurate: no transformer fine-tuning head; actual = VES + genotype MLP → `[PH]` |
| D9 | PROP L64–66 (generic preprocessing / feature extraction / spatial k-fold blocking) | Vague and partly inaccurate (actual eval = LOOCV, not spatial blocking) → `[PH]` |
| D10 | PROP L74–78 (Computational Infrastructure → Hardware Requirements) | Hardware requirements inappropriate for a journal article |
| D11 | PROP L90–92 (Independent Test Set) | Inaccurate: actual protocol is LOOCV, no held-out test set → covered by `[PH]` 2.8 |
| D12 | PROP L94–96 (Comparison with Traditional Methods: STRUCTURE, ADMIXTURE, distance-based) | Inaccurate: not run; actual baselines = Locator + no-VES tuned + Zenato Lazzari comparison → `[PH]` 2.9 |
| D13 | PROP L98–147 (SCHEDULE §) + Figure image7 (timeline) | Schedule/Gantt inappropriate for a journal article |
| D14 | PROP L149–155 (BUDGET §) | Budget inappropriate for a journal article |
| D15 | PROP L157–170 (EXPECTED OUTCOMES §) | Expected-outcome/future content; superseded by Results + Discussion (use as *material* for Discussion `[PH]`, do not copy) |
| D16 | main.tex L30 (Portuguese title), L128–130 (Portuguese `resumo` + keywords) | E&E is English-only; Portuguese front matter not needed |
| D17 | PROP L88 stratified-k-fold + bootstrap-resampling clauses | Not what was done (actual = LOOCV only); keep only the LOOCV clause (R48) |
| D18 | BG L6 subsubsection "Traditional methods", and any `\subsection`/`\subsubsection` wrappers whose nesting doesn't fit E&E | Structural-only; titles collapsed into E&E heading levels |

---

## PART 4 — PLACEHOLDER LIST

Format: **(location)** → *what the human must write* → *README/source specifics to draw on
(context only — write fresh prose, do not copy README).* Use the `\PLACEHOLDER{...}`
macro (Part 6). Every entry below is meta-instruction, never paper prose.

### Front matter
- **PH-A (Abstract)** → Write a structured E&E abstract for COMPLETED work
  (background → aim → methods → key result → conclusion). Must state actual
  headline result. *Draw on:* README "Results" (149 km median haversine over
  55-genome LOOCV; ~2.4–2.7× better than SCAT/traditional; within-500 km 83.6%;
  Bayesian-optimization 199→174 km and VES transfer 174→149 km decomposition).
  Verbatim future-tense proposal abstract is available at main.tex L134 as raw
  material but must NOT be used as-is (it predates results and lists Darwin's Ark).
- **PH-B (Author list beyond first author)** → Confirm co-authors and order.
  *Available names:* "Guzman Vitar" (main.tex L24); advisor "Prof. Dr. Dalvan Jair
  Griebler" (main.tex L55); co-advisor "Prof. Dr. Eduardo Eizirik", PUCRS
  (main.tex L61). Human decides which become co-authors.
- **PH-C (Affiliations)** → Institutional affiliations with addresses. *Known:*
  PUCRS (Pontifícia Universidade Católica do Rio Grande do Sul), Porto Alegre, RS,
  Brazil; LBGM-PUCRS mentioned at PROP L36. Full addresses not in sources.
- **PH-D (Correspondence)** → Corresponding author name + email + postal address (not in sources).
- **PH-E (Funding statement)** → Grant numbers/agencies (budget section discarded; not in sources).
- **PH-F (Keywords)** → Confirm/adjust the 5 reused keywords; E&E wants pipe-separated (see Part 5).

### Section 2 (Materials and Methods) — gaps
- **PH-G (2.2 Felid pre-training corpus)** → Describe the actual corpus: six felid
  reference assemblies (*Felis catus*, *Panthera leo*, *P. tigris*, *P. onca*,
  *Puma concolor*, *P. pardus*) with accessions; 512 bp windows, 128 bp overlap,
  ≤5% N; DNABERT-2 BPE tokenizer (pinned revision); continued MLM pre-training
  (15% masking); training hyperparameters (LR 5e-5, cosine schedule, batch 32×2,
  BF16, early stopping); locus-safe 80/20 hash split. *Draw on:* README "Stage 1",
  "Pre-training Corpus", "Sequence Windowing and Tokenization", "Locus-Safe
  Train/Validation Splitting", "Pre-training Objective and Hyperparameters".
- **PH-H (2.3 Jaguar samples)** → Actual sample counts and provenance: 55 jaguar
  whole genomes (from a 57-sample VCF) across five Brazilian biomes with per-biome
  n (e.g., Caatinga n=5, Pantanal n=6; give all five); geographic coordinates;
  data source. *Draw on:* README "Research Context", "Genotype Matrix
  Construction", per-biome results table; Step 4 data acquisition.
- **PH-I (2.4 Variant Effect Scoring)** → Define VES: 512 bp window centered on
  each biallelic SNP, mask center token, frozen felid-pretrained DNABERT-2
  (`AutoModelForMaskedLM`), `VES = log P(alt|context) − log P(ref|context)`;
  ~83k scores; interpretation. *No proposal source.* *Draw on:* README "Stage 2",
  "Variant Effect Scoring".
- **PH-J (2.5 Genotype matrix)** → 0/1/2 allele-count encoding (0/0 retained),
  VCF filtering (PASS, biallelic SNPs, single-nt REF/ALT), per-fold
  allele-frequency imputation of missing genotypes; matrix 55 × ~83k. *Draw on:*
  README "Genotype Matrix Construction".
- **PH-K (2.6 Genotype MLP + learnable locus gates)** → GeoGenIE/Locator-style
  MLP: optional learnable per-locus sigmoid gate (initialized from VES, refined by
  backprop) → BatchNorm → [Linear→ELU→Dropout]×L → 2-output coordinate head;
  four VES modes (learnable/weighted/selection/none); overparameterization guard.
  Reuse R47 (PROP L62 regression framing) as the opening sentence if desired.
  *Draw on:* README "Genotype MLP Architecture", "VES Integration Strategies".
- **PH-L (2.7 Loss & coordinate normalization)** → Differentiable haversine loss
  in degree space; per-fold Z-score coordinate normalization; km→Mm scaling;
  optional biome cross-entropy head (weight 0 by default). *Draw on:* README
  "Coordinate Normalization and Haversine Loss", "Loss Functions and Task Weighting".
- **PH-M (2.8 Cross-validation & Optuna)** → 55-fold LOOCV; Optuna TPE
  hyperparameter search (100 trials) minimizing median haversine; top-5 ensemble;
  fixed-hyperparameter mode for baselines. Reuse R48 (LOOCV clause) as anchor.
  *Draw on:* README "Stage 3", "Genotype MLP Architecture" (Optuna), "Evaluation".
- **PH-N (2.9 Baselines)** → (i) Locator baseline reproducing Battey et al. 2020
  fixed defaults (10 layers, 256 units, Adam, dropout 0.25 after layer 5, LR 1e-3,
  5000 epochs); (ii) no-VES tuned baseline (same Optuna budget, raw genotypes).
  *Draw on:* README "No-VES Tuned Baseline", "Locator Baseline".
- **PH-O (2.10 Evaluation metrics)** → Primary = median haversine (km); within
  250/500/1000 km; per-biome median/mean; classification metrics when biome head
  active. *Draw on:* README "Evaluation Metrics". (Optional partial reuse R49.)
- **PH-P (2.11 Implementation & reproducibility)** → Software stack (Python,
  PyTorch, Transformers, Optuna, etc.); reproducibility guarantees (pinned
  tokenizer, checksum downloads, deterministic splits). Optional reuse R50.
  *Draw on:* README "Reproducibility and Integrity Guarantees", "Key Dependencies".

### Section 3 (Results) — entirely placeholder (no proposal source)
- **PH-Q (3.1 Overall accuracy)** → Report the full-pipeline result over 55-fold
  LOOCV (149 km median haversine; within-500 km). *Draw on:* README "Results" intro + summary table.
- **PH-R (3.2 Comparison with published methods)** → Table + prose comparing to
  Zenato Lazzari et al. 2025 / SCAT (median, within-500 km, per-biome). *Draw on:*
  README "Comparison with Published Methods" (incl. per-biome median table).
- **PH-S (3.3 Bayesian-optimization contribution)** → 199→174 km; converged
  architecture (4 layers, 511 units, 563 epochs). *Draw on:* README "Bayesian Optimization Contribution".
- **PH-T (3.4 VES transfer-learning contribution)** → 174→149 km; effect
  concentrated in smallest populations. *Draw on:* README "Transfer Learning Contribution".
- **PH-U (3.5 Per-biome breakdown)** → Per-biome median haversine vs. paper.
  *Draw on:* README per-biome tables.
- **PH-V (Results tables/figures)** → Construct the summary table (configuration →
  median km / within-500 km) and any error-distribution map from actual outputs
  (`loocv_summary.json`, `loocv_predictions.json`). *Draw on:* README "Results".

### Section 4 (Discussion) — entirely placeholder (no proposal source)
- **PH-W (Discussion)** → Interpret results; four contribution threads (Bayesian
  optimization unlocks Locator on small samples; felid foundation model as reusable
  asset; VES as label-free FST alternative; transfer learning helps most where data
  is scarcest); limitations; future work. *Draw on:* README "Key Contributions";
  proposal "EXPECTED OUTCOMES" (PROP L157–170) may be used as *idea material* only.

### Back matter — all placeholder
- **PH-X (Author Contributions)** → CRediT roles per author.
- **PH-Y (Data Availability Statement)** → Where VCF/location CSV, felid assemblies
  (NCBI accessions + DNA Zoo), and code live. *Draw on:* README Steps 1 & 4, repository.
- **PH-Z (Acknowledgements)** → Funders, LBGM-PUCRS, field collaborators (proposal
  `agradecimentos` block is empty).
- **PH-AA (Conflict of Interest)** → Standard declaration.
- **PH-AB (Supporting Information)** → Optional; list supplementary tables/figures if any.

---

## PART 5 — FRONT / BACK MATTER PLAN (exact strings)

### Title `[REUSE — main.tex L31, byte-identical]`
```
Development of Machine Learning Models for Precise Geographic Assignment of Jaguars (\textit{Panthera onca}) Using Complete Genomes and Transfer Learning
```

### Keywords `[REUSE — main.tex L133 terms; comma→pipe is format scaffolding]`
Source terms (byte-identical): `machine learning`, `conservation genomics`,
`jaguar`, `geographic assignment`, `transfer learning`. Rendered:
```
machine learning | conservation genomics | jaguar | geographic assignment | transfer learning
```

### Abstract `[PH-A]` — but the verbatim proposal abstract is available as raw material:
main.tex **L134** (first words: "This master's thesis proposal presents an
innovative approach to geographic assignment…"). ⚠ Future-tense, mentions Darwin's
Ark, predates results → **do not ship as-is; becomes PH-A**.

### Author / advisor names `[REUSE strings; assembly is PH-B]`
- Author: `Guzman Vitar` (main.tex L24)
- Advisor: `Prof. Dr. Dalvan Jair Griebler` (main.tex L55)
- Co-advisor: `Prof. Dr. Eduardo Eizirik` / `PUCRS` (main.tex L61)

### Placeholders in front/back matter
Affiliations (PH-C), Correspondence (PH-D), Funding (PH-E), Author Contributions
(PH-X), Data Availability (PH-Y), Acknowledgements (PH-Z), Conflict of Interest
(PH-AA), Supporting Information (PH-AB).

---

## PART 6 — SCAFFOLD PLAN (LaTeX; TinyTeX + Overleaf compatible)

**Do NOT depend on `pucrs-ppgcc.cls`.** Build a self-contained `article`-based
manuscript that mirrors E&E and compiles on both TinyTeX and Overleaf using only
standard/CTAN packages.

### Proposed file layout under `paper/`
```
paper/
├── main.tex            # standalone manuscript (to be written in step 2)
├── bibliography.bib    # already present (23 entries, author–year keys)
└── images/
    ├── image5.png      # KEEP — jaguar distribution map (fig:jaguar_distribution)
    ├── image1.png      # KEEP — transfer-learning illustration
    └── (image2,3,4,6,7.png present but UNUSED → not \includegraphics'd; may delete)
```

### Preamble (proposed)
- `\documentclass[11pt]{article}`
- Layout/fonts: `geometry` (e.g. A4/letter, 1in margins), `\usepackage{times}` or
  `newtxtext`+`newtxmath` (widely available on TinyTeX/Overleaf), `setspace`
  (double spacing, common E&E submission style), `microtype`.
- Graphics/tables: `graphicx` (with `\graphicspath{{images/}}`), `booktabs`,
  `multirow`, `array`, `float`.
- Refs/links: `natbib` (author–year: `\usepackage[round,authoryear]{natbib}`),
  `hyperref` (loaded last), `\bibliographystyle{apalike}` (author–year; ships with
  TeX Live/TinyTeX; no Wiley `.bst` dependency). Note: proposal's `ppgcc-num.bst`
  is numeric → discard; use `apalike` for E&E author–year style.
- Headings: `titlesec` to render Wiley-style numbered headings
  `1 | INTRODUCTION`. Proposed formatter:
  ```latex
  \usepackage{titlesec}
  \titleformat{\section}{\normalfont\large\bfseries\MakeUppercase}
    {\thesection\ \textbar\ }{0.5em}{}
  \titleformat{\subsection}{\normalfont\bfseries}{\thesubsection}{0.5em}{}
  ```
  (yields "1 | INTRODUCTION", "1.1 Jaguar Conservation", …)
- Author block: `authblk` (`\author[1]{...}` + `\affil[1]{...}`) for
  affiliation footnotes — standard, Overleaf/TinyTeX-safe. Keywords + funding +
  correspondence set as manual `\noindent\textbf{...}` blocks.

### Distinctive PLACEHOLDER macro (proposed)
```latex
\usepackage{xcolor}
% Block placeholder — impossible to mistake for paper prose.
\newcommand{\PLACEHOLDER}[1]{%
  \par\medskip\noindent
  \fcolorbox{red}{yellow!15}{%
    \parbox{0.95\linewidth}{\textbf{\textcolor{red}{[[PLACEHOLDER]]}}\ #1}}%
  \par\medskip}
% Inline variant for short slots (title cells, dates, etc.)
\newcommand{\PH}[1]{{\textbf{\textcolor{red}{[[PH: #1]]}}}}
```
Every human-writes-later slot uses one of these; nothing else red/boxed appears,
so placeholders are visually unambiguous and greppable (`\PLACEHOLDER`, `\PH`).

### Bibliography
- Reuse `paper/bibliography.bib` unchanged (already copied).
- `\bibliographystyle{apalike}` + `\bibliography{bibliography}` at the References
  slot. All in-text citations become `\citep{}` / `\citet{}` (natbib). The existing
  `\cite{}` calls copied from the proposal work under natbib but should ideally be
  reviewed to `\citep`/`\citet` — that is a mechanical citation-command change
  (format scaffolding), not prose editing.

### Figures kept vs discarded
- KEEP: `image5.png` (R5, jaguar distribution), `image1.png` (R36, transfer
  learning). Their `\caption{...}` text is reused byte-identical (INTRO L15, BG L211).
- DISCARD from the manuscript: `image2/3/4/6/7.png` (tutorial + timeline; D2, D13).

### Compile check (step-2 acceptance)
`pdflatex → bibtex → pdflatex → pdflatex` must succeed on TinyTeX and Overleaf
with no missing-package errors (all packages above are in the standard TeX Live /
TinyTeX collections; no PUCRS class, no custom `.bst`).

---

## OPEN QUESTIONS (for the requester)
1. **Intro length.** Section 1 as mapped has 7 subsections incl. method-by-method
   literature (1.4/1.5). Keep all (thesis-like) or discard some subsubsections for
   a tighter E&E intro? Both are rule-compliant (discarding is allowed).
2. **Optional foundation-model tutorial passages** (R37/R38, "Why Transformers",
   "Pretraining Tasks") — include or discard as too tutorial?
3. **Evaluation-metrics reuse** (R49, PROP L70): reuse verbatim (mentions MAE/R²
   which are not the actual primary metrics) or drop entirely in favour of PH-O?
4. **Portuguese abstract** — confirmed discarded (E&E English-only)? (Assumed yes.)
5. **Bib style** — `apalike` proposed for author–year; acceptable, or a specific
   Wiley/E&E `.bst` you want bundled?
