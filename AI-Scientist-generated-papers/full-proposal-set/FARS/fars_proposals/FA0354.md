# untitled

# Parameter-Group Learning Rates to Widen the Learning-Rate Window of Selective SSMs on Recall Benchmarks

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Efficient alternatives to softmax attention (which has \(O(n^2)\) time and memory in sequence length) are increasingly important for long-context language modeling. A prominent family of alternatives are **selective state space models (SSMs)** such as **Mamba**, which process sequences in linear time using a recurrent state update while retaining strong performance on many language modeling benchmarks.

However, recent work shows that comparing architectures by their best-reported accuracy can be misleading when a model family has **extreme optimization brittleness**. In particular, some modern recurrent/SSM models solve synthetic recall benchmarks only under a very narrow range of learning rates. This brittleness increases the cost of fair evaluation (dense hyperparameter sweeps) and harms reproducibility.

### The Problem

We focus on two fully-automated synthetic recall benchmarks:

1. **Multi-Query Associative Recall (MQAR)**: a synthetic key–value retrieval benchmark where tokens can act as keys or values and the model must output the correct value for multiple query keys that re-occur at different positions in the sequence.
2. **Copying**: a synthetic sequence copying task (e.g., input is `BOS x1 … xL COPY` and the model must output `x1 … xL`).

**Zoology** introduces MQAR to isolate long-range retrieval behavior and argues that associative recall failures explain a substantial portion of the perplexity gap between attention and efficient token mixers (**[Zoology](./references/Zoology-Measuring-and-Improving-Recall-in-Efficient-Language-Models/meta/meta_info.txt)**).

**Okpekpe & Orvieto (2025)** show a sharp optimization asymmetry: Transformers are relatively robust to learning-rate choice, while modern recurrent/SSM models (including Mamba-class) achieve near-perfect MQAR (and copying) accuracy only within an extremely narrow learning-rate window (**[Revisiting AR](./references/REVISITING-ASSOCIATIVE-RECALL-IN-MODERN-RECURRENT-MODELS/meta/meta_info.txt)**).

The open question is whether this brittleness is an inherent property of selective SSMs, or whether it can be substantially reduced by a simple optimizer-side intervention that does not change the model architecture or data.

### Key Insight and Hypothesis

**Key insight:** In selective SSM blocks (e.g., Mamba), a small subset of parameters controls the recurrence *dynamical system* (state transition and discretization). Concretely, Mamba uses parameters such as `A_log` (log-parameterization of the diagonal state matrix) and time-step parameters produced by `dt_proj` (passed through `softplus`). These parameters enter the recurrence through exponentials and long products (e.g., \(\exp(\Delta A)\)), so they can behave like **stiff dynamics parameters**: parameters whose stable step size is much smaller than that of surrounding projection and MLP weights.

**Hypothesis:** Using **AdamW with two parameter groups**, where the SSM-dynamics parameters use a smaller learning rate \(\eta_{\mathrm{ssm}} = r \cdot \eta\) (with fixed, pre-declared ratios \(r\in\{0.1, 0.03\}\)) while all other parameters use \(\eta\), will **widen the set of global learning rates \(\eta\)** that reach high accuracy within a fixed step budget on MQAR and Copying, compared to the standard single-learning-rate baseline, **without reducing best-of-grid accuracy**.

Why this could be wrong: (i) the narrow learning-rate window may be driven primarily by non-SSM components (e.g., projections/gating/norms) or by optimizer-state interactions in AdamW; (ii) AdamW’s per-parameter normalization may already compensate for some parameter-scale differences; (iii) reducing \(\eta_{\mathrm{ssm}}\) could stabilize training but also prevent learning, merely shifting the optimal range rather than widening it.

---

## Proposed Approach

### Overview

We propose a minimal optimizer modification for training selective SSMs:

- **Baseline**: AdamW with a single learning rate \(\eta\).
- **Proposed**: AdamW with two parameter groups:
  1) **SSM-dynamics group** with learning rate \(\eta_{\mathrm{ssm}} = r\cdot \eta\)
  2) **Everything else** with learning rate \(\eta\)

All other settings (model architecture, initialization, batch size, step budget, random seeds, learning-rate schedule, gradient clipping, and weight decay handling) are held constant.

### Mamba-Specific Mechanism Hypothesis (What Makes This Non-Trivial)

Classical SSM training recipes (e.g., S4/S5) often use smaller learning rates for SSM dynamics parameters. Selective SSMs differ in a way that may make the effect *stronger* and the failure mode *sharper*:

- In Mamba, the effective discretization \(\Delta_t\) is **input-dependent**, produced by a learned projection (e.g., `dt_proj`) and a nonlinearity (softplus). This means optimization can rapidly change not just a global time constant, but the **distribution of per-token time steps**.
- The recurrence uses exponentials like \(\exp(\Delta_t A)\). If updates push \(\Delta_t\) (or \(|A|\)) so that \(\Delta_t |A|\) becomes too large, the model can become overly forgetful (effective memory collapses). If it becomes too small, the dynamics can become poorly conditioned for learning long products over many steps.

**Mechanistic prediction:** When Mamba fails at high or low base learning rates, we expect to see an early collapse of learned time scales, measurable as a shift in the distribution of \(\Delta_t\) (or derived time constants such as \(\tau \approx 1/(\Delta_t\,\exp(A_{\log}))\)). Grouped learning rates should reduce the sensitivity of these time-scale statistics to the global \(\eta\), thereby widening the stable training window.

### Method Details

**Which parameters are in the SSM-dynamics group?**

We define the SSM-dynamics group operationally for Mamba(-like) implementations as all parameters that directly control the state transition and discretization, including:

- `A_log` (log-parameterization of the diagonal state transition)
- `D` (skip/diagonal residual parameter)
- `dt_proj.{weight,bias}` (time-step projection)
- (If present) other explicit discretization/time-step parameters that are part of the recurrence

We intentionally **do not** include general input/output projections or MLP weights in this group.

**Weight decay handling:** We keep weight decay identical across conditions by respecting standard Mamba conventions that mark some parameters as `_no_weight_decay` (e.g., `A_log`, `D`). To avoid a confound where AdamW’s shrinkage term scales with the per-group learning rate, we set **weight decay = 0** for the entire SSM-dynamics group (including `dt_proj.*`) in *all* conditions.

**Robustness metric:** We evaluate stability via a pre-declared learning-rate success-window metric:

- Fix a global learning-rate grid \(\eta \in \mathrm{logspace}(10^{-5}, 10^{-2}, K)\) (default \(K=16\)).
- For each condition and seed, define “success” as reaching **validation accuracy \(\ge 0.95\)** by step budget \(T\).
- Compute **contiguous LR-span** as \(\log_{10}(\eta_{\max}/\eta_{\min})\) for the *largest contiguous interval* of successful learning rates in the grid.

### Key Contributions

1. **A hypothesis-driven test** of whether reinstating S4/S5-style dynamics learning-rate separation reduces the extreme global LR brittleness reported for selective SSMs on recall tasks.
2. **Mechanism-oriented diagnostics** (time-scale and gradient statistics) that can distinguish “true widening” from “mere optimal-range shift” and connect outcomes to Mamba’s selective discretization parameters.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) efficient sequence model backbones beyond softmax attention, (ii) synthetic recall benchmarks that isolate retrieval-like behavior, and (iii) optimization/learnability analyses for recurrent models.

### Related Papers

- **[Zoology](./references/Zoology-Measuring-and-Improving-Recall-in-Efficient-Language-Models/meta/meta_info.txt)**: Introduces MQAR and argues associative recall explains a large portion of the perplexity gap between attention and efficient token mixers.
- **[Revisiting associative recall in modern recurrent models](./references/REVISITING-ASSOCIATIVE-RECALL-IN-MODERN-RECURRENT-MODELS/meta/meta_info.txt)**: Reports that recurrent/SSM models succeed on MQAR and copying only in a narrow learning-rate window.
- **[Mamba](./references/Mamba-Linear-Time-Sequence-Modeling-with-Selective-State-Spaces/meta/meta_info.txt)**: Introduces selective SSM blocks and documents special handling of dynamics parameters (e.g., `A_log` kept in fp32 and marked `_no_weight_decay`).
- **Hyena Hierarchy (arXiv:2302.10866)**: Long convolution architecture that is competitive on language modeling and is commonly compared with SSMs on long-range tasks.
- **H3: Hungry Hungry Hippos (arXiv:2212.14052)**: A hardware-efficient SSM-based language modeling approach that motivates recall benchmarks for efficient mixers.
- **RWKV (arXiv:2305.13048)**: A recurrent model mixing time-decay and attention-like updates, often used as a strong non-attention baseline in recall studies.
- **RetNet (arXiv:2307.08621)**: Retention-based sequence modeling that interpolates attention-like behavior with linear-time recurrence.
- **[Mimetic Initialization](./references/Mimetic-Initialization-Helps-State-Space-Models-Learn-to-Recall/meta/meta_info.txt)**: Improves Mamba’s recall/copying behavior via structured initialization; does not directly target learning-rate robustness.
- **[Repeat After Me](https://arxiv.org/abs/2402.01032)**: Formalizes the string-copying task and studies copying limits of state-space models.
- **Transformers are SSMs (arXiv:2405.21060)**: Explores formal connections between attention and (selective) SSMs, motivating mechanistic comparisons on recall tasks.
- **[A Theoretical Analysis of Mamba’s Training Dynamics](./references/A-THEORETICAL-ANALYSIS-OF-MAMBAS-TRAINING-DYNAMICS-FILTERING-RELEVANT-FEATURES-FOR-GENERALIZATION-IN-STATE-SPACE-MODELS/meta/meta_info.txt)**: Provides theory on how selective SSM dynamics interact with feature selection and generalization.
- **S4/S5 training recipes (arXiv:2208.04933)**: Classical SSM implementations often treat dynamics parameters (eigenvalues / time-scales) with different optimizer hyperparameters (e.g., smaller LR and/or no weight decay), suggesting optimizer-side stability levers.

**Novelty kill search summary:** We found S4/S5 precedent for “SSM dynamics get a smaller LR,” but we did not find work explicitly testing whether this simple optimizer change **widens the global LR success window** for **selective** SSMs (Mamba) on MQAR/copying.

---

## Experiments

### Experimental Setup

**Tasks**:

1. **MQAR (Zoology)**: synthetic multi-query key–value recall at varying positions, vocabulary size 8192.
2. **Copying (Repeat After Me-style)**: synthetic string copying with special tokens (e.g., `BOS`, `COPY`), evaluated by exact token accuracy on the copied segment.

**Data / generators**:

- MQAR: use the official MQAR generator in the Zoology codebase (synthetic, on-the-fly generation).
- Copying: implement the standard online generator used in prior work (sample random strings, concatenate `BOS x COPY`, train autoregressively).

**Models**:

- Primary: a small-from-scratch Mamba(-like) model (e.g., 2 layers, \(d_{model}\in\{256,512\}\), standard `d_state` for the implementation). Keep model size fixed across all conditions.
- Sanity baseline: a small Transformer (2 layers, RoPE or learned positional embeddings) trained on the same generators to verify the benchmark pipeline (Transformers are expected to be LR-robust on these tasks).

**Optimizer**: AdamW (same betas, eps, gradient clipping, scheduler across all conditions). Only difference is whether we use parameter-group learning rates and the fixed ratio \(r\).

**Main conditions (per task)**:

- **Closest baseline**: Mamba + AdamW single LR (\(r=1.0\))
- **Proposed**: Mamba + AdamW grouped LR (\(r=0.1\))
- **Robustness check**: Mamba + AdamW grouped LR (\(r=0.03\))

**Learning-rate grid (fixed a priori)**: \(\eta \in \mathrm{logspace}(10^{-5}, 10^{-2}, K)\), default \(K=16\).

**Step budget**: Train each run for a fixed number of optimizer steps \(T\) (default 20k), with early stopping for clearly failing runs (e.g., if validation accuracy remains <0.1 after 10% of steps).

**Seeds / variance plan**: 3 random seeds per condition: `seeds=[42,123,456]`.

**Primary metrics (per task)**:

1. **Contiguous LR-span** (largest contiguous successful LR interval where val acc \(\ge 0.95\) by step \(T\)).
2. **Best-of-grid validation accuracy**.

**Secondary diagnostics (automated logs)**:

- Fraction of parameters in the SSM-dynamics group.
- Gradient norm statistics for SSM-dynamics vs non-SSM parameters.
- Time-scale statistics: distribution of \(\Delta_t\) and derived time constants (e.g., quantiles of \(\tau\)).
- Divergence rate (NaNs / infs / loss blow-ups).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MQAR (Zoology) | Synthetic multi-query key–value recall (vocab=8192) | Query-token accuracy; contiguous LR-span; best acc | generated | https://github.com/HazyResearch/zoology | Zoology MQAR generator + eval |
| Copying (Repeat After Me) | Synthetic string copying with special separator token | Copy-segment token accuracy; contiguous LR-span; best acc | generated | arXiv:2402.01032 | Simple online generator + exact-match eval |

### Main Results

#### MQAR Results (planned)

| Method | Base Model | Benchmark | Metric (mean±std) | Source | Notes |
|---|---|---|---|---|---|
| AdamW (single LR, r=1.0) | Mamba-small | MQAR | LR-span=**TBD**, BestAcc=**TBD** | this work | baseline |
| AdamW (grouped LR, r=0.1) | Mamba-small | MQAR | LR-span=**TBD**, BestAcc=**TBD** | this work | proposed |
| AdamW (grouped LR, r=0.03) | Mamba-small | MQAR | LR-span=**TBD**, BestAcc=**TBD** | this work | robustness ratio |
| AdamW (single LR) | Transformer-small | MQAR | LR-span=**TBD**, BestAcc=**TBD** | this work | sanity check (expected LR-robust) |

#### Copying Results (planned)

| Method | Base Model | Benchmark | Metric (mean±std) | Source | Notes |
|---|---|---|---|---|---|
| AdamW (single LR, r=1.0) | Mamba-small | Copying | LR-span=**TBD**, BestAcc=**TBD** | this work | baseline |
| AdamW (grouped LR, r=0.1) | Mamba-small | Copying | LR-span=**TBD**, BestAcc=**TBD** | this work | proposed |
| AdamW (grouped LR, r=0.03) | Mamba-small | Copying | LR-span=**TBD**, BestAcc=**TBD** | this work | robustness ratio |
| AdamW (single LR) | Transformer-small | Copying | LR-span=**TBD**, BestAcc=**TBD** | this work | sanity check (expected LR-robust) |

### Ablation Studies

No additional ablations are required for the first verification pass to keep the decision test minimal. If results are positive, a follow-up can isolate which subset drives the effect (e.g., group only `dt_proj` vs only `A_log`).

### Experimental Rigor

- **Fairness:** All conditions use identical generators, model architecture, initialization, training budget, and LR schedule; only the parameter-group LR ratio differs.
- **Variance:** Report mean±std over 3 seeds.
- **Sanity checks:** (i) random guessing should be near chance on MQAR query tokens; (ii) a small Transformer should solve MQAR/copying across a wide LR range.
- **Shift vs widen:** Plot accuracy vs LR to distinguish widening from a pure shift of the optimal range.

### Resource Estimate

- MQAR runs: 3 conditions × 16 LRs × 3 seeds = 144 runs.
- Copying runs: 3 conditions × 16 LRs × 3 seeds = 144 runs.
- Total (Mamba): 288 runs, plus a small Transformer sanity sweep (≤16 runs).

For 2-layer small models on synthetic data, we expect ≤1 A100-hour per run (conservative). Target total budget: **≤350 A100 GPU-hours**.

---

## Success Criteria

**Hypothesis:** A fixed-ratio parameter-group LR for SSM-dynamics parameters widens the global LR success window of Mamba on recall benchmarks.

**Decision Rule**:

- **Proceed** if either grouped-LR setting (r=0.1 or r=0.03) improves **MQAR** contiguous LR-span by **≥0.5 decades** (≈3× wider) relative to baseline **and** achieves MQAR best-of-grid accuracy within **1 percentage point** of the baseline best-of-grid accuracy (mean across 3 seeds), **and** does not reduce Copying LR-span by more than 0.2 decades.
- **Pivot** if MQAR LR-span increases but MQAR best-of-grid accuracy drops by >1pp; next test is to modify the dynamics group definition (e.g., include/exclude `dt_proj` vs `A_log`).
- **Refute** if MQAR LR-span increases by <0.2 decades (≈1.6×) or if grouped-LR reduces MQAR best-of-grid accuracy by >1pp.

---

## Impact Statement

If successful, this work would provide a simple, implementation-light training default for selective SSMs that reduces expensive learning-rate sweeps and improves reproducibility of architecture comparisons on recall benchmarks. Because the intervention is optimizer-only, it has low adoption friction for existing Mamba training codebases.

---

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](./references/Mamba-Linear-Time-Sequence-Modeling-with-Selective-State-Spaces/meta/meta_info.txt) - Gu, Dao, 2023
- [Zoology: Measuring and Improving Recall in Efficient Language Models](./references/Zoology-Measuring-and-Improving-Recall-in-Efficient-Language-Models/meta/meta_info.txt) - Arora et al., 2023
- [Revisiting associative recall in modern recurrent models](./references/REVISITING-ASSOCIATIVE-RECALL-IN-MODERN-RECURRENT-MODELS/meta/meta_info.txt) - Okpekpe, Orvieto, 2025
- [Mimetic Initialization Helps State Space Models Learn to Recall](./references/Mimetic-Initialization-Helps-State-Space-Models-Learn-to-Recall/meta/meta_info.txt) - Trockman et al., 2024
- [A Theoretical Analysis of Mamba's Training Dynamics](./references/A-THEORETICAL-ANALYSIS-OF-MAMBAS-TRAINING-DYNAMICS-FILTERING-RELEVANT-FEATURES-FOR-GENERALIZATION-IN-STATE-SPACE-MODELS/meta/meta_info.txt) - Shandirasegaran et al., 2026
- [Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933) - Smith et al., 2022
- [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032) - Jelassi, Brandfonbrener, 2024
- [Hyena Hierarchy](https://arxiv.org/abs/2302.10866) - Poli et al., 2023
- [H3: Hungry Hungry Hippos](https://arxiv.org/abs/2212.14052) - Fu et al., 2023
- [RWKV](https://arxiv.org/abs/2305.13048) - Peng et al., 2023
- [RetNet](https://arxiv.org/abs/2307.08621) - Sun et al., 2023
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) - Dao, Gu, 2024
- [Adam](https://arxiv.org/abs/1412.6980) - Kingma, Ba, 2014
- [AdamW](https://arxiv.org/abs/1711.05101) - Loshchilov, Hutter, 2019
