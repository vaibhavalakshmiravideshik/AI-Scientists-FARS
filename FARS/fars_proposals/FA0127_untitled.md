# untitled

# Budget-Distilled ES-SSM: In-Place Cross-Budget Distillation for Elastic Spectral State Space Models

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR (or similar)
- **Core constraint**: fully automated training + evaluation (no human labeling/evaluation in the loop)
- **Verification budget**: ≤ 768 A100 GPU-hours

## Introduction

### Context and Motivation

Many sequence models must be deployed under variable latency / cost constraints (e.g., a single service serving heterogeneous hardware, or a mobile/edge setting where available compute fluctuates). A common solution is **budgeted inference**: the model exposes a runtime “budget knob” that trades quality for compute.

For long-context sequence tasks, **state space models (SSMs)** and related long-sequence architectures can be attractive because they avoid quadratic attention cost. However, most SSMs are trained at a fixed capacity and do not directly support post hoc truncation without retraining or accuracy collapse.

**Elastic Spectral State Space Models (ES-SSM)** by Song & Wang (2026) build on Hankel spectral filtering: each layer contains \(\bar{K}\) **spectral channels** (eigenmodes of a fixed Hankel matrix) ordered by decreasing eigenvalue, so truncating to the first \(K\) channels is a natural compute knob. At runtime the model sets a **spectral budget** \(K\le \bar{K}\) by computing per-channel gate logits and applying a **masked softmax** that normalizes over channels \(1..K\) and sets channels \(k>K\) to zero [Song & Wang 2026, §4.1](<./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/sections/4.1 Budgeted inference through Adaptive spectral gating..md>). On Long Range Arena (LRA) Text (IMDb sentiment classification with byte-level tokenization), accuracy (higher is better) drops from **93.91% at \(K=32\)** to **84.95% at \(K=2\)** ([Table 1](<./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/sections/5 Experiments.md>) and [Table 2](<./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/sections/5.2.2 Single-model budget sweeps and sweet spots.md>)) [Song & Wang 2026](./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/meta/meta_info.txt).

### The Problem

ES-SSM trains with **budget dropout**: each update samples \(K_{train}\) and trains only the truncated model at that budget (shared parameters across budgets) [Song & Wang 2026, §4.2](./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/sections/4.2%20Training%20ES-SSM%20with%20budget%20dropout.md). This gives the low-budget subnetworks direct supervision, but it does not explicitly enforce that **different budgets implement a consistent function**.

A plausible failure mode is **budget inconsistency**: for the same input \(x\), the distributions \(p_{K}(y\mid x)\) and \(p_{\bar{K}}(y\mid x)\) can drift because (i) different budgets activate different subsets of spectral channels, and (ii) small \(K\) updates are capacity-limited and may be noisier. In elastic CNN training, this exact setting (many subnetworks sharing weights) is commonly improved by **in-place knowledge distillation**, where the largest subnetwork acts as a teacher for smaller subnetworks during training (e.g., sandwich rule + in-place distillation in universally slimmable networks).

### Key Insight and Hypothesis

**Hypothesis:** During budget-dropout training of ES-SSM, adding an **in-place cross-budget distillation loss** from the model’s own full-budget prediction \(p_{\bar{K}}\) to the sampled-budget prediction \(p_{K_{train}}\) will reduce budget inconsistency and improve low-budget accuracy.

Mechanism intuition: the full-budget forward pass (with refinement channels available) provides a smoother target distribution than one-hot labels, and encourages the shared low-index channels to approximate the full model’s decision boundary when forced to operate at small \(K\). This is uncertain because the teacher and student share parameters; distillation may be redundant once we already include strong ground-truth supervision, or the small-budget subnetwork may be unable to match the full-budget behavior.

---

## Proposed Approach

### Overview

We propose **Budget-Distilled ES-SSM (BD-ES-SSM)**: train ES-SSM with budget dropout as usual, but for each minibatch compute both:
1) a full-budget forward pass at \(\bar{K}=32\)
2) a sampled-budget forward pass at \(K_{train}\in\{2,3,4,6,8,12,16,24,32\}\)

and add a KL distillation term aligning the sampled-budget logits to the full-budget logits.

### Method Details

Let \(z_{\bar{K}}(x)\) be the classifier logits from the ES-SSM run with \(K=\bar{K}=32\), and \(z_{K}(x)\) the logits when run with \(K=K_{train}\). Define
\[
q = \text{softmax}(z_{\bar{K}}/T),\quad p = \text{softmax}(z_{K}/T).
\]

**Baseline (compute-matched): anchored dual-CE**
\[
\mathcal{L}_{base} = \text{CE}(y, z_{\bar{K}}) + \text{CE}(y, z_{K}).
\]

**BD-ES-SSM (ours): anchored dual-CE + cross-budget KL**
\[
\mathcal{L}_{ours} = \mathcal{L}_{base} + \lambda\, T^2\, \mathrm{KL}(\;\text{stopgrad}(q)\;\|\;p\;).
\]

We pre-register **\(T=2\)** and **\(\lambda=0.5\)** (no tuning in the main experiment). The teacher distribution \(q\) is detached to avoid backpropagating distillation gradients into the teacher pass; full-budget quality is maintained by the explicit \(\text{CE}(y, z_{\bar{K}})\) anchor.

### Key Innovations

- Apply **in-place / cross-budget distillation** to **ordered spectral truncation** in ES-SSM, where budgets correspond to a prefix of spectral channels rather than arbitrary width choices.
- Use a **compute-matched baseline** (two forwards per step for both methods) to avoid confounding “more compute” with “better training objective”.
- Add an explicit mechanism diagnostic (“budget inconsistency”) measured as \(\mathbb{E}_x[\mathrm{KL}(p_{\bar{K}}(\cdot\mid x)\|p_K(\cdot\mid x))]\).

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) long-sequence modeling with state space models and efficient transformers, (ii) elastic / budgeted inference where one training run supports many runtime compute budgets, and (iii) knowledge distillation and self-distillation techniques that regularize families of subnetworks.

On the modeling side, long-range benchmarks like **Long Range Arena (LRA)** have been used to compare efficient transformers (sparse/linear attention) and SSMs. Separately, elastic model training in vision (slimmable networks, Once-for-All, BigNAS) has established training recipes (sandwich rule, in-place distillation) that often improve small subnetworks without retraining.

Our contribution is not a new architecture: it is a targeted training objective change for ES-SSM-style spectral budget knobs, designed to improve the most deployment-relevant regime (very small budgets).

### Related Papers

- **[Elastic Spectral State Space Models for Budgeted Inference](./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/meta/meta_info.txt)**: Introduces ES-SSM with adaptive spectral gating + budget dropout for truncation to any \(K\le\bar{K}\) (our direct base method).
- **[Once for All: Train One Network and Specialize it for Efficient Deployment](./references/Once-for-All-Train-One-Network-and-Specialize-it-for-Efficient-Deployment/meta/meta_info.txt)**: Trains a supernet for many deployment configurations using progressive shrinking and distillation (elastic training inspiration).
- **[Slimmable Neural Networks](./references/Slimmable-Neural-Networks/meta/meta_info.txt)**: Early width-elastic training showing shared-weight subnetworks can work well, often improved by distillation-style signals.
- **[Universally Slimmable Networks and Improved Training Techniques](https://arxiv.org/abs/1903.05134)**: Introduces the sandwich rule and in-place distillation for arbitrary-width subnetworks (closest analog of our training recipe).
- **[BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models](https://arxiv.org/abs/2003.11142)**: Extends sandwich-rule training to large supernets, emphasizing strong smallest-subnet performance.
- **[Inplace Knowledge Distillation with Teacher Assistant for Improved Flexible Deep Neural Networks](https://arxiv.org/abs/2105.08369)**: Adds teacher-assistant distillation within flexible models, supporting the idea that smaller subnetworks may benefit from more “relatable” teachers.
- **[Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393)**: Classic teacher-assistant distillation when teacher–student gaps are large; suggests why full-budget \(K=32\) might be too hard a teacher for \(K=2\).
- **[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)**: Foundational knowledge distillation framework.
- **[Born Again Neural Networks](https://arxiv.org/abs/1805.04770)**: Self-distillation improving accuracy without changing architecture; relevant precedent that self-teaching can help even with shared inductive biases.
- **[Spectral State Space Models](https://arxiv.org/abs/2312.06837)**: Introduces spectral SSM / STU building blocks that ES-SSM builds on.
- **[Flash STU: Fast Spectral Transform Units](https://arxiv.org/abs/2409.10489)**: Optimized STU implementations, relevant for practical ES-SSM training/inference cost.
- **[Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396)**: Popular SSM baseline on LRA; provides training infrastructure and context for long-sequence SSM evaluation.
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)**: Modern selective SSM architecture; ES-SSM compares against Mamba-family models at full capacity.
- **[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)**: Mamba-2 / SSM–Transformer duality framing; ES-SSM is part of the broader SSM resurgence.
- **[MEGA: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655)**: Efficient long-range sequence model baseline often evaluated on LRA.
- **[Performer: Rethinking Attention with Linear Complexity](https://arxiv.org/abs/2009.14794)**: Canonical linear-attention transformer baseline for long sequences.
- **[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)**: Low-rank attention baseline used in long-sequence benchmarking.
- **[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)**: LSH attention baseline used in LRA-era comparisons.
- **[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)**: Sparse attention baseline for long documents.
- **[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)**: Sparse attention baseline with theoretical guarantees.
- **[The Curious Case of In-Training Compression of State Space Models](./references/The-Curious-Case-of-In-Training-Compression-of-State-Space-Models/meta/meta_info.txt)**: Compresses SSM state during training; adjacent to “train once, deploy smaller” but not cross-budget distillation.
- **[SpectraLDS: Provable Distillation for Linear Dynamical Systems](./references/SpectraLDS-Provable-Distillation-for-Linear-Dynamical-Systems/meta/meta_info.txt)**: Distills spectral representations into simpler LDS; adjacent distillation in spectral/SSM setting.
- **[Retrieval-Aware Distillation for Transformer-SSM Hybrids](./references/Retrieval-Aware-Distillation-for-Transformer-SSM-Hybrids/meta/meta_info.txt)**: Distills Transformer–SSM hybrids with explicit retrieval heads; shows distillation is actively studied for SSM hybrids.
- **[Matryoshka Model Learning for Improved Elastic Student Models](https://arxiv.org/abs/2505.23337)**: Trains nested student/assistant models for elasticity; conceptually similar to “nested budgets” though not spectral truncation.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Elastic supernet training | One training run supports many subnetworks; often uses sandwich sampling + in-place distillation | [OFA](./references/Once-for-All-Train-One-Network-and-Specialize-it-for-Efficient-Deployment/meta/meta_info.txt), [US-Nets](https://arxiv.org/abs/1903.05134), [BigNAS](https://arxiv.org/abs/2003.11142) | ImageNet, downstream deployment metrics | Subnet interference; smallest subnet underperforms without extra regularization |
| Spectral / state-space long-sequence models | Replace attention with structured linear recurrences / spectral filters | [Spectral SSM](https://arxiv.org/abs/2312.06837), [S4](https://arxiv.org/abs/2111.00396), [Mamba](https://arxiv.org/abs/2312.00752) | LRA, LM, forecasting | Often trained at fixed capacity; truncation can be brittle |
| Budgeted inference knobs | Explicit runtime budget controls compute/quality | [ES-SSM](./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/meta/meta_info.txt), early-exit families (e.g., MSDNet) | LRA (multi-budget sweeps), latency–accuracy curves | Small-budget performance may lag; need training tricks to avoid collapse |
| Distillation / self-distillation | Match a stronger teacher distribution to regularize student training | [KD](https://arxiv.org/abs/1503.02531), [BAN](https://arxiv.org/abs/1805.04770), [TA-KD](https://arxiv.org/abs/1902.03393) | Many | Teacher–student mismatch; sometimes gains are small without careful controls |

### Closest Prior Work

- **ES-SSM** [Song & Wang 2026](./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/meta/meta_info.txt): Introduces spectral budget truncation and budget-dropout training, but does not use cross-budget distillation objectives between \(\bar{K}\) and \(K\).
- **Universally Slimmable Networks (US-Nets)** [Yu & Huang 2019](https://arxiv.org/abs/1903.05134): Uses sandwich rule + in-place distillation to improve small subnetworks, but studies CNN width elasticity (with BN issues) rather than ordered spectral-channel truncation in SSMs.
- **Once-for-All (OFA)** [Cai et al. 2019](./references/Once-for-All-Train-One-Network-and-Specialize-it-for-Efficient-Deployment/meta/meta_info.txt): Uses progressive shrinking and distillation to train elastic CNNs, but does not address spectral truncation or SSM gating/normalization.
- **IPKD-TA** [Ozerov & Duong 2021](https://arxiv.org/abs/2105.08369): Proposes teacher assistants for in-place distillation in flexible models; suggests a possible future extension if \(K=32\) is too strong a teacher.

**Novelty Kill Search Summary:** Searched for the exact combination of “ES-SSM / elastic spectral / budget dropout” with “distillation / KL / in-place distillation / cross-budget distillation”, including OpenReview and GitHub queries (full log in `notes.md`). No prior work explicitly adding cross-budget KL distillation to ES-SSM-style budget-dropout training was found as of 2026-02-17.

### Comparison Table

| Related work | What it does | Key limitation (for our question) | What we change | Why ours should win |
|---|---|---|---|---|
| [ES-SSM](./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/meta/meta_info.txt) | Budget dropout + masked softmax gate enables truncation to any \(K\) | No explicit objective tying \(p_K\) to \(p_{\bar{K}}\) | Add KL(\(p_{\bar{K}}\|p_K\)) during training | Should reduce budget inconsistency and help very small \(K\) |
| [US-Nets](https://arxiv.org/abs/1903.05134) | Sandwich sampling + in-place distillation across widths | Not spectral channels; BN/statistics issues dominate | Port idea to ordered spectral truncation | Distillation benefits should transfer to elastic SSM budgets |
| [OFA](./references/Once-for-All-Train-One-Network-and-Specialize-it-for-Efficient-Deployment/meta/meta_info.txt) | Progressive shrinking + distillation for many subnetworks | Focuses on CNN architecture dimensions | Use distillation but keep ES-SSM architecture fixed | Minimal change targets ES-SSM’s specific low-budget gap |
| [IPKD-TA](https://arxiv.org/abs/2105.08369) | Teacher assistants improve flexible-model distillation | Not applied to ES-SSM; more complex | Keep single teacher (\(\bar{K}\)) in main test; TA is fallback | If \(\bar{K}\rightarrow K\) gap is the issue, TA is next step |

---

## Experiments

### Experimental Setup

**Primary benchmark**: **Long Range Arena (LRA) Text (IMDb byte-level classification)**.
- Task: binary sentiment classification.
- Sequence length: 4096 (standard LRA setting).
- Metric: accuracy (%).
- Dataset source: Google Research LRA repository [long-range-arena](https://github.com/google-research/long-range-arena) (Apache-2.0 license).

**Budgets**:
- \(\bar{K}=32\), evaluate \(K\in\{2,3,4,6,8,12,16,24,32\}\).
- Training budgets sampled from the same discrete set (excluding 1 as in ES-SSM §5.1).

**Main conditions (2 total; compute-matched)**:
1) **Anchored dual-CE baseline**: \(\mathcal{L}_{base}=\text{CE}(y,z_{32})+\text{CE}(y,z_{K_{train}})\).
2) **BD-ES-SSM**: \(\mathcal{L}_{ours}=\mathcal{L}_{base}+\lambda T^2\,\mathrm{KL}(\text{stopgrad}(\text{softmax}(z_{32}/T))\|\text{softmax}(z_{K_{train}}/T))\), with \(T=2\), \(\lambda=0.5\).

**Mechanism diagnostic (same runs; no extra training)**:
- **Budget inconsistency**: for each \(K\in\{2,3,4,6,8\}\), compute \(\mathbb{E}_{x\sim\text{val}}[\mathrm{KL}(p_{32}(\cdot\mid x)\|p_{K}(\cdot\mid x))]\).

**Seeds / variance**:
- Train each condition with **3 seeds** (e.g., `seeds=[42,123,456]`), report mean±std.

**Implementation note (feasibility plan)**:
- ES-SSM’s public code is limited; verification can implement ES-SSM directly from the paper’s equations (§4.1–4.2) and reuse existing LRA dataloaders from `google-research/long-range-arena` or S4’s `HazyResearch/state-spaces` training harness.

**Resource Estimate** (conservative)
- Evidence for LRA IMDb training cost: DSS reports **20 minutes on a single A100** for LRA IMDB in its README table (not ES-SSM, but same benchmark family) [ag1988/dss README](https://github.com/ag1988/dss/blob/main/README.md).
- ES-SSM will be heavier than DSS; additionally, our compute-matched setup uses **2 forward passes per step**, roughly doubling training compute relative to a single-pass baseline.
- Conservative budget assumption: **5 GPU-hours / run** (single A100) × 2 conditions × 3 seeds = **30 GPU-hours** total, plus <5 GPU-hours for evaluation sweeps/diagnostics.
- This is well within the 768 GPU-hour cap.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| LRA Text (IMDb bytes) | Byte-level sentiment classification with long sequences | Accuracy (%) | standard LRA train/test | https://github.com/google-research/long-range-arena | Use LRA official scripts or S4 LRA pipeline (e.g., https://github.com/HazyResearch/state-spaces) |

### Main Results

**Published ES-SSM budget sweep (context):** from ES-SSM Table 2 (single \(\bar{K}=32\) training run, truncation at test time) [Song & Wang 2026, §5.2.2](./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/sections/5.2.2%20Single-model%20budget%20sweeps%20and%20sweet%20spots.md)
- Text accuracy (%): \(K=2\) 84.95, \(K=3\) 92.60, \(K=4\) 92.69, \(K=6\) 92.77, \(K=8\) 93.84, \(K=32\) 93.91.

**Primary metric for this proposal**:
- **AvgAcc_lowK** = mean accuracy across \(K\in\{2,3,4,6,8\}\).

#### Results Table

| Method | Base Model | Benchmark | AvgAcc_lowK (mean±std) | Acc@K=2 (mean±std) | Acc@K=32 (mean±std) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| ES-SSM (paper) | ES-SSM \(\bar{K}=32\) | LRA Text | 91.37 (derived, 1 run) | 84.95 (1 run) | 93.91 (1 run) | [ES-SSM Table 2](./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/sections/5.2.2%20Single-model%20budget%20sweeps%20and%20sweet%20spots.md) | Context only; different training protocol than ours |
| Anchored dual-CE (baseline) | ES-SSM \(\bar{K}=32\) | LRA Text | **TBD** | **TBD** | **TBD** | - | To be verified (3 seeds) |
| **BD-ES-SSM (ours)** | ES-SSM \(\bar{K}=32\) | LRA Text | **TBD** | **TBD** | **TBD** | - | To be verified (3 seeds) |

### Ablation Studies

We will not add extra training conditions beyond the two main runs unless the main result is inconclusive. If the main result is null but the budget-inconsistency diagnostic improves substantially, we allow one contingent follow-up run (not part of the main decisive test):

| Variant | What’s changed | Expected finding |
|---|---|---|
| BD-ES-SSM (λ=1.0) | Increase distillation weight, keep \(T=2\) | If effect is weight-limited, low-K accuracy may improve |

### Experimental Rigor

**Confounders and controls**:
- **Compute mismatch**: both methods use exactly two forward passes per step; the only difference is the KL term.
- **Different budget sampling**: use identical \(K_{train}\) sampling distribution and RNG seeds for both conditions.
- **Evaluation bugs across K**: add a sanity check that \(K=32\) evaluation equals the full-budget model’s standard evaluation, and that accuracy monotonically does not increase when truncating to extremely small \(K\) on a random-label control.

---

## Success Criteria

**Hypothesis** (directional): BD-ES-SSM improves low-budget accuracy by reducing budget inconsistency, while preserving full-budget accuracy.

**Decision Rule** (concrete):
- **Proceed** if BD-ES-SSM improves **AvgAcc_lowK** over anchored dual-CE by **≥ 0.8 percentage points** (or ≥ 1× pooled std across seeds) **and** \(|\Delta\text{Acc@32}| \le 0.5\) points.
- **Refute** if improvement in AvgAcc_lowK is **≤ 0.3 points** (within noise) or if Acc@32 drops by **> 0.5** points.
- **Contingent follow-up** (one additional run): if AvgAcc_lowK is null but the inconsistency diagnostic decreases by ≥5% relative on average across \(K\in\{2,3,4,6,8\}\), rerun with \(\lambda=1.0\).

---

## Impact Statement

If successful, this provides a simple, training-time-only recipe to improve the smallest-budget performance of ES-SSM-style models without training separate models per budget. Practitioners deploying a single long-sequence model under fluctuating compute constraints could obtain better quality at very low budgets (where the deployment pressure is highest) with minimal additional implementation complexity.

---

## References

- [Elastic Spectral State Space Models for Budgeted Inference](./references/Elastic-Spectral-State-Space-Models-for-Budgeted-Inference/meta/meta_info.txt) - Dachuan Song and Xuan Wang, 2026
- [Once for All: Train One Network and Specialize it for Efficient Deployment](./references/Once-for-All-Train-One-Network-and-Specialize-it-for-Efficient-Deployment/meta/meta_info.txt) - Han Cai, Chuang Gan, Song Han, 2019
- [Slimmable Neural Networks](./references/Slimmable-Neural-Networks/meta/meta_info.txt) - Jiahui Yu et al., 2018
- [SpectraLDS: Provable Distillation for Linear Dynamical Systems](./references/SpectraLDS-Provable-Distillation-for-Linear-Dynamical-Systems/meta/meta_info.txt) - Devan Shah et al., 2025
- [The Curious Case of In-Training Compression of State Space Models](./references/The-Curious-Case-of-In-Training-Compression-of-State-Space-Models/meta/meta_info.txt) - Makram Chahine et al., 2025
- [Retrieval-Aware Distillation for Transformer-SSM Hybrids](./references/Retrieval-Aware-Distillation-for-Transformer-SSM-Hybrids/meta/meta_info.txt) - Aviv Bick, Eric P. Xing, Albert Gu, 2026
- [Universally Slimmable Networks and Improved Training Techniques](https://arxiv.org/abs/1903.05134) - Jiahui Yu, Thomas Huang, 2019
- [BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models](https://arxiv.org/abs/2003.11142) - Jiahui Yu et al., 2020
- [Inplace Knowledge Distillation with Teacher Assistant for Improved Flexible Deep Neural Networks](https://arxiv.org/abs/2105.08369) - Alexey Ozerov, Ngoc Q. K. Duong, 2021
- [Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393) - Mehrdad Mirzadeh et al., 2019
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Geoffrey Hinton, Oriol Vinyals, Jeff Dean, 2015
- [Born Again Neural Networks](https://arxiv.org/abs/1805.04770) - Tommaso Furlanello et al., 2018
- [Spectral State Space Models](https://arxiv.org/abs/2312.06837) - Naman Agarwal et al., 2023
- [Flash STU: Fast Spectral Transform Units](https://arxiv.org/abs/2409.10489) - (see arXiv), 2024
- [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) - Albert Gu et al., 2021
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) - Albert Gu, Tri Dao, 2023
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) - Tri Dao, Albert Gu, 2024
- [MEGA: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655) - (see arXiv), 2022
- [Performer: Rethinking Attention with Linear Complexity](https://arxiv.org/abs/2009.14794) - Krzysztof Choromanski et al., 2020
- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) - Sinong Wang et al., 2020
- [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) - Nikita Kitaev et al., 2020
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Iz Beltagy et al., 2020
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) - Manzil Zaheer et al., 2020
- [Long Range Arena](https://arxiv.org/abs/2011.04006) - Yi Tay et al., 2020
