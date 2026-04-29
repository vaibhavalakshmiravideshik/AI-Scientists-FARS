# untitled

# Source-Referenced JS Coefficient Learning for LoRA Adapter Merging

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Parameter-efficient fine-tuning (PEFT) methods such as **LoRA** (Low-Rank Adaptation) are widely used to adapt a large pretrained model to many downstream tasks while training and storing only a small number of additional parameters. In practice, teams may accumulate dozens of task-specific LoRA adapters for the same base model (e.g., one per customer, domain, or capability).

When deployment constraints require *one* adapter (or one merged model) rather than many, **LoRA merging** tries to combine multiple task-specific adapters into a single multi-task adapter without further full fine-tuning. If reliable, merging can reduce storage and simplify serving.

### The Problem

The simplest merging approach is **task arithmetic**: treat each task’s adapter as a parameter delta from the base model and form a merged delta by taking a weighted sum (often uniform weights). However, recent work shows that merging LoRAs is substantially harder than merging full fine-tunes:

- **[DO-Merging](./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/meta/meta_info.txt)** argues that LoRA modules can have **large magnitude variance** across tasks, so naive averaging causes some tasks’ updates to dominate and harms multi-task performance.
- Geometry-based LoRA-merging methods attempt to reduce interference by manipulating adapter subspaces, e.g. **[KnOTS](./references/Model-merging-with-SVD-to-tie-the-Knots/meta/meta_info.txt)** (SVD alignment), **[Core Space](./references/Accurate-and-Efficient-Low-Rank-Model-Merging-in-Core-Space/meta/meta_info.txt)** (merge in a shared low-rank subspace), **[OSRM](./references/Unraveling-LoRA-Interference-Orthogonal-Subspaces-for-Robust-Model-Merging/meta/meta_info.txt)** (orthogonal subspaces), and modular decompositions such as **[LoRA-LEGO](./references/Merging-LoRAs-like-Playing-LEGO-Pushing-the-Modularity-of-LoRA-to-Extremes-Through-Rank-Wise-Clustering/meta/meta_info.txt)**.

A different family of methods avoids explicit subspace engineering by **learning merge coefficients** from data. For full fine-tuned models, this is effective:

- **[AdaMerging](./references/AdaMerging-Adaptive-Model-Merging-for-Multi-Task-Learning/meta/meta_info.txt)** learns task-/layer-wise coefficients by minimizing the **entropy of the merged model’s predictions** on unlabeled samples.
- **[DivMerge](./references/DivMerge-A-divergence-based-model-merging-method-for-multi-tasking/meta/meta_info.txt)** learns coefficients by minimizing **Jensen–Shannon (JS) divergence** between the merged model and each task model’s output distributions.

However, **DivMerge explicitly states LoRA as an untested limitation** (“in low rank adaptation (LoRA) … LoRA matrices … are responsible for significant performance loss when merging … we have not experimented within this constrained setup.”).

This creates an actionable open question for practitioners:

> When the “experts” to merge are **LoRA adapters** (not full fine-tunes), what coefficient-learning objective is robust enough to avoid collapse and actually improve merged performance?

This is not solvable by better prompting or inference-time scaling: the goal is to *compress* a set of existing LoRA adapters into one while preserving their fine-tuned behavior.

### Key Insight and Hypothesis

**Key insight**: In coefficient learning, the objective matters. **Entropy minimization (AdaMerging)** is *reference-free*: it encourages confident predictions but does not constrain *which* predictions the merged model should be confident about. Under LoRA interference, this can plausibly lead to **confidently wrong** predictions and coefficient collapse toward “easy-to-confident” tasks.

We propose to instead use a **source-referenced objective**: match the merged model’s output distribution to each task’s own fine-tuned LoRA model on that task’s inputs.

**Hypothesis**: For LoRA merging, optimizing merge coefficients with a **JS-to-sources objective** (teacher-anchored output agreement) will outperform entropy-minimization coefficient learning (AdaMerging-style) and uniform averaging, and will approach the performance of supervised coefficient tuning using a small labeled set.

Why this could be wrong:
- Task-specific LoRA models can disagree strongly; the objective may be underdetermined and lead to unstable coefficients.
- JS estimation could be noisy with small unlabeled sets.
- LoRA interference might be dominated by parameter-space geometry that coefficient tuning cannot fix, in which case geometry methods like DO-Merging should dominate.

---

## Proposed Approach

### Overview

We propose **SourceJS-LoRA**, a coefficient-learning method for merging K task-specific LoRA adapters into one multi-task adapter.

Given:
- a base model \(\theta_0\)
- K task adapters \(\Delta_1,\ldots,\Delta_K\) (LoRA deltas)
- unlabeled per-task samples \(X_1,\ldots,X_K\)

we learn coefficients \(\Gamma\) (task-wise or layer-wise) for the merged adapter:

\[
\Delta(\Gamma) = \sum_{k=1}^K \gamma_k \Delta_k \quad\text{(task-wise)}
\]

and minimize the **sum of per-task JS divergences** between each task model \(\theta_k = \theta_0 + \Delta_k\) and the merged model \(\theta(\Gamma)=\theta_0+\Delta(\Gamma)\) on that task’s inputs:

\[
\min_{\Gamma}\; \sum_{k=1}^K \mathbb{E}_{x\sim X_k}\left[ JS\big(p_{\theta_k}(\cdot\mid x)\;\|\;p_{\theta(\Gamma)}(\cdot\mid x)\big)\right].
\]

To keep the experiment compute-feasible and to reduce divergence sample complexity, we compute JS on a **label-space distribution** (small number of labels) rather than the full vocabulary distribution.

### Method Details

#### 1) Label-space distributions for GLUE-style classification

For classification tasks, we define a small label set \(\mathcal{Y}_k\) and verbalize each label as a short string (standard in T5-style evaluation). For an input \(x\), we compute

\[
q_{\theta}(y\mid x) \propto \exp(\log p_{\theta}(\text{verbalize}(y)\mid \text{prompt}(x))).
\]

This yields a \(|\mathcal{Y}_k|\)-way distribution, enabling stable JS computation with tens of samples.

#### 2) Optimization over coefficients only

We keep \(\theta_0\) and all \(\Delta_k\) fixed. Only \(\Gamma\) is trainable. This makes optimization cheap and avoids overfitting by introducing additional degrees of freedom.

We use Adam on \(\Gamma\) for a small number of steps (e.g., 300–1000), with optional regularization toward uniform coefficients:

\[
\mathcal{L}(\Gamma) = \sum_k \mathbb{E}_{x\sim X_k} JS(q_{\theta_k}(\cdot\mid x)\|q_{\theta(\Gamma)}(\cdot\mid x)) + \lambda\|\Gamma-\Gamma_0\|_2^2.
\]

We will start with **task-wise \(\gamma_k\)** as the main setting and treat **layer-wise \(\gamma_k^{(\ell)}\)** as a single ablation.

#### 3) Why JS-to-sources differs from AdaMerging entropy minimization

- **AdaMerging objective** (for task-wise coefficients) minimizes \(\sum_k\sum_{x\in B_k} H(q_{\theta(\Gamma)}(\cdot\mid x))\) on unlabeled samples.
- **SourceJS-LoRA** minimizes teacher-anchored divergence to \(q_{\theta_k}\) on the same \(x\).

Intuition: entropy minimization is “be confident”; SourceJS-LoRA is “agree with the task expert on its own inputs”. This is a stronger constraint in settings where the merged model might otherwise become confidently wrong.

### Key Innovations

1. **Source-referenced divergence objective for LoRA merging**: apply DivMerge-style JS minimization to LoRA adapters, but emphasize the **teacher-anchored** nature of the loss as a remedy for reference-free confidence collapse.
2. **Direct comparison of coefficient-learning objectives under LoRA interference**: isolate *objective choice* (JS-to-sources vs entropy-of-merged vs supervised accuracy tuning) while keeping the parameterization (merge coefficients) fixed.
3. **Label-space JS for sample-efficient coefficient learning**: make divergence estimation feasible with tens of unlabeled samples per task.

---

## Related Work

### Field Overview

Model merging aims to combine multiple task-specialized models into a single model without retraining. For foundation models, merging is attractive because retraining multi-task models can be expensive and data access may be restricted. A common abstraction is **task vectors** (parameter deltas from a shared base model) and arithmetic operations on these vectors.

For LoRA, merging is complicated by low-rank structure and task interference that can be worse than in full fine-tuning. Recent LoRA-merging work splits into (i) **geometry/subspace** methods (aligning or orthogonalizing subspaces), (ii) **parameter processing** methods (magnitude decoupling, pruning, masking), and (iii) **coefficient learning** methods (learn weights with unlabeled or labeled data).

This proposal focuses on (iii) but evaluates against strong (i)/(ii) baselines when feasible.

### Related Papers

- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**: Introduces LoRA, the PEFT method whose adapters we merge.
- **[Task Arithmetic](https://arxiv.org/abs/2212.04089)**: Introduces adding task vectors as a simple model composition baseline.
- **[Model Soups](https://arxiv.org/abs/2203.05482)**: Shows averaging multiple fine-tunes can improve robustness; foundational for merging.
- **[TIES-Merging](https://arxiv.org/abs/2306.01708)**: Resolves sign conflicts and trims parameters to reduce interference in merging.
- **[DARE](https://arxiv.org/abs/2311.03099)**: Drops and rescales deltas to improve merging robustness.
- **[Mergekit / Multi-SLERP](https://arxiv.org/abs/2403.13257)**: Practical toolkit and interpolation methods for LLM merging.
- **[AdaMerging](./references/AdaMerging-Adaptive-Model-Merging-for-Multi-Task-Learning/meta/meta_info.txt)**: Learns task-/layer-wise coefficients via entropy minimization on unlabeled samples.
- **[DivMerge](./references/DivMerge-A-divergence-based-model-merging-method-for-multi-tasking/meta/meta_info.txt)**: Learns coefficients by minimizing JS/KL divergence between merged and task models.
- **[DO-Merging](./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/meta/meta_info.txt)**: Identifies LoRA magnitude variance and proposes decoupling + data-free orthogonalization.
- **[KnOTS](./references/Model-merging-with-SVD-to-tie-the-Knots/meta/meta_info.txt)**: Aligns adapter subspaces via SVD for improved merging.
- **[Core Space](./references/Accurate-and-Efficient-Low-Rank-Model-Merging-in-Core-Space/meta/meta_info.txt)**: Merges in a low-rank shared subspace for accuracy/efficiency.
- **[OSRM](./references/Unraveling-LoRA-Interference-Orthogonal-Subspaces-for-Robust-Model-Merging/meta/meta_info.txt)**: Uses orthogonal subspaces to reduce LoRA interference.
- **[LoRA-LEGO](./references/Merging-LoRAs-like-Playing-LEGO-Pushing-the-Modularity-of-LoRA-to-Extremes-Through-Rank-Wise-Clustering/meta/meta_info.txt)**: Rank-wise clustering for modular adapter composition.
- **[HydraOpt](./references/HydraOpt-Navigating-the-Efficiency-Performance-Trade-off-of-Adapter-Merging/meta/meta_info.txt)**: Exploits LoRA asymmetry to trade off storage vs performance in data-free merging.
- **[IterIS](./references/IterIS-Iterative-Inference-Solving-Alignment-for-LoRA-Merging/meta/meta_info.txt)**: Iteratively refines feature alignment for LoRA merging with few samples.
- **[Recycling LoRAs](./references/The-Appeal-and-Reality-of-Recycling-LoRAs-with-Adaptive-Merging/meta/meta_info.txt)**: Large-scale evaluation of adaptive merging on real Hub LoRAs; highlights limited transfer in heterogeneous pools.
- **[Fisher Merging](https://arxiv.org/abs/2212.09993)**: Uses Fisher information to weight parameters when merging.
- **[RegMean](https://arxiv.org/abs/2212.09741)**: Regression-based merging using data-dependent constraints; often compared in adaptive merging.
- **[Model Breadcrumbs](https://arxiv.org/abs/2312.06795)**: Uses sparse masks to scale multi-task merging.
- **[Parameter Competition Balancing (PCB-Merging)](https://arxiv.org/abs/2410.02396)**: Balances parameter competition to mitigate interference.
- **[Task Singular Vectors (TSVM)](https://arxiv.org/abs/2412.00081)**: Uses singular vectors to reduce task interference.
- **[CoPA-Merging](https://arxiv.org/abs/2502.17159)**: Complementary parameter adaptation for multimodal merging.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Uniform task arithmetic | Add task deltas with fixed coefficients | Task Arithmetic, Model Soups | GLUE, vision multi-task suites | Sensitive to interference |
| Parameter pruning / sign resolution | Remove conflicts by trimming and sign voting | TIES, DARE | Vision + LLM merging | Still heuristic; not LoRA-specific |
| Coefficient learning (reference-free) | Optimize weights using confidence proxies on unlabeled data | AdaMerging | Vision MTL suites | Can be confidently wrong; may collapse |
| Coefficient learning (source-referenced) | Match merged outputs to per-task experts | DivMerge; **this proposal** | Full FT merging; (proposed) LoRA merging | Requires access to per-task data + expert outputs |
| Geometry/subspace methods | Align or orthogonalize adapter subspaces | KnOTS, Core Space, OSRM | Vision/LLM/LoRA merging | More implementation complexity; may require constraints |
| LoRA-specific magnitude handling | Decouple magnitude/direction; orthogonalize | DO-Merging | Vision+NLP+LLM LoRA merging | No released code; extra steps |

### Closest Prior Work

- **AdaMerging**: Uses task-/layer-wise coefficients and unlabeled samples, but optimizes **entropy of the merged model**. Our method uses the same coefficient parameterization but replaces the objective with **JS-to-sources**, anchoring to each task expert.
- **DivMerge**: Uses JS-to-sources for full fine-tunes and explicitly flags LoRA as untested; we test the same objective class in the LoRA setting and adapt it to label-space distributions.
- **DO-Merging**: Strong LoRA-specific baseline that targets magnitude variance and orthogonality in parameter space; our approach instead targets **behavioral agreement** and can be used when data-free geometry assumptions are insufficient.
- **IterIS**: Uses unlabeled data and iterative feature extraction to improve LoRA merging; unlike our approach, it solves an alignment regression objective rather than a divergence-to-teachers objective.
- **Recycling LoRAs**: Suggests many adaptive merging gains in heterogeneous pools come from regularization rather than transfer; our setting is controlled (in-house LoRAs) and tests whether teacher-anchored divergence can outperform reference-free objectives.

**Novelty Kill Search Summary:** Searched for the exact combination “JS/Jensen-Shannon divergence + LoRA/adapter merging + coefficient learning” (and variants like “DivMerge LoRA”, “divergence-based LoRA merging”, “logit divergence adapter merging”) and checked local KB for matches. No prior work directly applying DivMerge-style JS-to-sources coefficient learning to LoRA merging was found as of 2026-02-19. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| AdaMerging | Entropy-minimization objective to learn coefficients on unlabeled samples | Reference-free confidence can become confidently wrong under interference | Replace objective with JS-to-task-experts | Anchors each task to its expert distribution |
| DivMerge | JS-to-sources coefficient learning for full fine-tunes | Not tested for LoRA | Apply to LoRA + label-space distributions | Same objective but LoRA setting + sample-efficient divergence |
| DO-Merging | Decouple magnitude/direction + orthogonalize LoRAs data-free | Not an objective-learning method; extra parameter-space steps | Keep LoRAs fixed; learn only coefficients | Lower engineering; directly targets behavior |
| Task arithmetic | Uniform coefficients | Sensitive to interference | Learn coefficients from unlabeled outputs | Data-driven coefficients should reduce conflicts |

---

## Experiments

### Experimental Setup

**Primary benchmark (decisive):** LoRA merging for **T5-base** on **8 GLUE-style tasks** (CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, STS-B), following the task set used by **DO-Merging**.

- **Base model**: `t5-base` (220M parameters; HuggingFace).
- **Task adapters**: train one LoRA adapter per task using PEFT (rank `r=16`, `lora_alpha=32`, dropout `0.05`) applied to attention projection matrices (encoder+decoder `q` and `v`).
- **Data for coefficient learning**:
  - Unlabeled: sample `n_u=64` examples per task from the validation split (inputs only).
  - Labeled (for supervised coefficient baseline): sample `n_l=64` per task from validation with labels.

**Why n=64?** DivMerge reports strong coefficient learning with as few as 25 samples for full fine-tunes (classification pairs) (see `Dataset Size Influence..md`). We start slightly higher to reduce noise in the LoRA regime.

#### Methods compared

We will implement all methods in the same codebase with identical single-task LoRA adapters.

- **Pre-trained**: base model without any adapter.
- **Single-task LoRA**: per-task adapters (upper bound, not merged).
- **Uniform merge (Task arithmetic)**: \(\gamma_k=1/K\).
- **AdaMerging-style entropy coefficient learning**: optimize coefficients \(\Gamma\) by minimizing entropy of the merged model on unlabeled per-task samples (label-space entropy).
- **Supervised coefficient tuning**: optimize \(\Gamma\) on labeled samples to directly minimize cross-entropy on the label-space distribution.
- **DO-Merging**: implement decouple+orthogonalize as in the paper (data-free).
- **SourceJS-LoRA (ours)**: optimize \(\Gamma\) by minimizing JS divergence to the per-task expert predictions on unlabeled per-task samples.

**Baseline Ladder (REQUIRED):**
- **Level 0**: Pre-trained base model (no adapter).
- **Level 1**: Single-task LoRA (upper bound; shows what merging is trying to approximate).
- **Level 2**: Uniform task arithmetic merge.
- **Level 3**: Strong LoRA-specific baseline: DO-Merging.
- **Level 4**: Closest coefficient-learning baseline: AdaMerging-style entropy minimization.
- **Level 5**: Data-using baseline: supervised coefficient tuning (same coefficient parameterization).

(Inference-time scaling baselines are not applicable here because evaluation is deterministic classification/regression, not open-ended generation.)

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| GLUE (8-task subset) | Standard NLU suite covering grammatical acceptability, sentiment, paraphrase, entailment, and semantic similarity | CoLA: MCC (Matthews correlation coefficient; higher is better); MNLI/QNLI/RTE/SST-2: accuracy; QQP/MRPC: F1 (harmonic mean of precision/recall; higher is better); STS-B: Pearson/Spearman correlation (higher is better) | official train/val/test | https://huggingface.co/datasets/glue | HF `evaluate` / GLUE script + T5 prompting |

We will follow DO-Merging’s metric presentation (values scaled to 0–100) for direct comparability.

### Resource Estimate

- **Single-task LoRA training**: 8 tasks × (T5-base LoRA fine-tune). Expected to be feasible on 1×A100; budget **≤ 80 GPU-hours** total including 3 seeds and re-runs.
- **Coefficient optimization** (AdaMerging / supervised tuning / SourceJS-LoRA): tiny number of parameters; dominated by forward passes on `8 × 64 = 512` examples; budget **≤ 10 GPU-hours**.
- **DO-Merging**: parameter-only orthogonalization + decoupling; expected to be minor overhead relative to training.

Total expected budget: **≤ 120 GPU-hours**, well within the 768 GPU-hour constraint.

### Main Results

Published reference numbers from **DO-Merging Table 3 (T5-base)** (for context and sanity-checking reproduction):

| Method | Base Model | Benchmark | Avg (0–100) | Source | Notes |
|---|---|---|---:|---|---|
| Pre-trained | T5-base | 8-task GLUE | 75.7 | DO-Merging Table 3 (`./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/sections/4.2 Language Models.md`) | Reported single value (seed count not specified) |
| Task arithmetic | T5-base | 8-task GLUE | 77.4 | DO-Merging Table 3 (`./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/sections/4.2 Language Models.md`) | Reported single value (seed count not specified) |
| TIES | T5-base | 8-task GLUE | 77.5 | DO-Merging Table 3 (`./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/sections/4.2 Language Models.md`) | Reported single value (seed count not specified) |
| KnOTS | T5-base | 8-task GLUE | 78.4 | DO-Merging Table 3 (`./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/sections/4.2 Language Models.md`) | Reported single value (seed count not specified) |
| TSVM | T5-base | 8-task GLUE | 79.0 | DO-Merging Table 3 (`./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/sections/4.2 Language Models.md`) | Reported single value (seed count not specified) |
| CoPA-Merging | T5-base | 8-task GLUE | 78.5 | DO-Merging Table 3 (`./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/sections/4.2 Language Models.md`) | Reported single value (seed count not specified) |
| **DO-Merging** | T5-base | 8-task GLUE | **80.9** | DO-Merging Table 3 (`./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/sections/4.2 Language Models.md`) | Reported single value (seed count not specified) |
| AdaMerging-style entropy coeffs | T5-base | 8-task GLUE | **TBD** | - | **Needs re-run** (no published LoRA result) |
| Supervised coefficient tuning | T5-base | 8-task GLUE | **TBD** | - | **Needs re-run** |
| **SourceJS-LoRA (ours)** | T5-base | 8-task GLUE | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (task-wise \(\gamma_k\)) | One coefficient per task | Strong baseline for our method |
| Ours (layer-wise \(\gamma_k^{(\ell)}\)) | One coefficient per task per transformer block | If interference is depth-dependent, should improve over task-wise |

### Experimental Rigor

- **Seeds**: `seeds=[42, 123, 456]` for LoRA training and coefficient optimization; report mean ± std.
- **Controls / confounders**:
  - *Confound: gains from more data*: AdaMerging, supervised tuning, and SourceJS-LoRA use the **same** number of examples per task for coefficient learning.
  - *Confound: overfitting coefficient set*: use a held-out split of the validation set for coefficient learning vs evaluation (e.g., first 64 examples for tuning, rest for evaluation; or cross-validation).
  - *Confound: metric mismatch across tasks*: use standard GLUE evaluation for each task and average after scaling to 0–100.
- **Sanity checks**:
  - Uniform merge should match published baselines within reasonable tolerance.
  - Learned coefficients should not collapse to a single task unless that task alone dominates the average metric.
  - For AdaMerging-style entropy minimization, verify whether entropy decreases while accuracy does not improve (a “confidently wrong” signature).

---

## Success Criteria

**Hypothesis**: JS-to-sources coefficient learning is more stable than entropy minimization for LoRA merging and yields a better merged adapter.

**Decision Rule**:
- **Proceed** if SourceJS-LoRA outperforms the best unsupervised coefficient baseline (AdaMerging-style entropy) by a margin outside the std range **and** is within 1.0 average point of DO-Merging (or better) on T5-base 8-task GLUE.
- **Pivot** if SourceJS-LoRA beats entropy minimization but lags DO-Merging by >1.0 point: try adding a minimal magnitude normalization pre-step (layerwise Frobenius normalization) while keeping the objective fixed.
- **Refute** if SourceJS-LoRA does not beat uniform task arithmetic or if gains disappear when coefficient-learning examples are held out (indicating overfitting).

---

## Impact Statement

If SourceJS-LoRA works, practitioners who maintain many task-specific LoRA adapters can merge them into a single adapter using only unlabeled inputs (no access to original training labels) with an objective that is less prone to “confident-but-wrong” collapse than entropy minimization. This would provide a simpler alternative to parameter-geometry methods (SVD alignment, orthogonalization) when those are harder to implement or do not transfer across adapter types, and would clarify when coefficient learning is a viable approach for LoRA consolidation.

---

## References

- [DivMerge: A divergence-based model merging method for multi-tasking](./references/DivMerge-A-divergence-based-model-merging-method-for-multi-tasking/meta/meta_info.txt) - Touayouch et al., 2025
- [AdaMerging: Adaptive Model Merging for Multi-Task Learning](./references/AdaMerging-Adaptive-Model-Merging-for-Multi-Task-Learning/meta/meta_info.txt) - Yang et al., 2023
- [Decouple and Orthogonalize: A Data-Free Framework for LoRA Merging](./references/Decouple-and-Orthogonalize-A-Data-Free-Framework-for-LoRA-Merging/meta/meta_info.txt) - Zheng et al., 2025
- [Model merging with SVD to tie the Knots](./references/Model-merging-with-SVD-to-tie-the-Knots/meta/meta_info.txt) - Stoica et al., 2024
- [Accurate and Efficient Low-Rank Model Merging in Core Space](./references/Accurate-and-Efficient-Low-Rank-Model-Merging-in-Core-Space/meta/meta_info.txt) - 2025
- [Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging](./references/Unraveling-LoRA-Interference-Orthogonal-Subspaces-for-Robust-Model-Merging/meta/meta_info.txt) - 2025
- [Merging LoRAs like Playing LEGO: Rank-Wise Clustering](./references/Merging-LoRAs-like-Playing-LEGO-Pushing-the-Modularity-of-LoRA-to-Extremes-Through-Rank-Wise-Clustering/meta/meta_info.txt) - 2024
- [HydraOpt: Navigating the Efficiency-Performance Trade-off of Adapter Merging](./references/HydraOpt-Navigating-the-Efficiency-Performance-Trade-off-of-Adapter-Merging/meta/meta_info.txt) - 2025
- [IterIS: Iterative Inference-Solving Alignment for LoRA Merging](./references/IterIS-Iterative-Inference-Solving-Alignment-for-LoRA-Merging/meta/meta_info.txt) - 2024
- [The Appeal and Reality of Recycling LoRAs with Adaptive Merging](./references/The-Appeal-and-Reality-of-Recycling-LoRAs-with-Adaptive-Merging/meta/meta_info.txt) - Liu et al., 2026
- [LoRA](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [Task Arithmetic](https://arxiv.org/abs/2212.04089) - Ilharco et al., 2022
- [Model Soups](https://arxiv.org/abs/2203.05482) - Wortsman et al., 2022
- [TIES-Merging](https://arxiv.org/abs/2306.01708) - Yadav et al., 2023
- [DARE](https://arxiv.org/abs/2311.03099) - Yu et al., 2024
- [Mergekit](https://arxiv.org/abs/2403.13257) - Goddard et al., 2024
- [Fisher Merging](https://arxiv.org/abs/2212.09993) - Matena & Raffel, 2022
- [RegMean](https://arxiv.org/abs/2212.09741) - Jin et al., 2022
- [Model Breadcrumbs](https://arxiv.org/abs/2312.06795) - Davari & Belilovsky, 2023
- [PCB-Merging](https://arxiv.org/abs/2410.02396) - Du et al., 2024
- [Task Singular Vectors](https://arxiv.org/abs/2412.00081) - Gargiulo et al., 2024
- [CoPA-Merging](https://arxiv.org/abs/2502.17159) - Zeng et al., 2025
