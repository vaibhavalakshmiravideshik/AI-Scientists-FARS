# untitled

# Compute-Matched TA-GRPO: Do Semantic Transformations Outperform Longer GRPO Training Under a Fixed Rollout Budget?

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS (Conference on Neural Information Processing Systems), ICML (International Conference on Machine Learning), ICLR (International Conference on Learning Representations), ACL (Association for Computational Linguistics), EMNLP (Conference on Empirical Methods in Natural Language Processing), or similar venues

## Introduction

### Context and Motivation

Large language models (LLMs) can be improved on tasks such as competition mathematics and unit-tested code generation by training them with feedback from automated checkers. This setting is often called **reinforcement learning with verifiable rewards (RLVR)**: each model output receives an objective reward computed by a programmatic verifier (e.g., exact-match checking for a final numeric answer, or running unit tests on generated code).

A widely used RLVR algorithm is **Group Relative Policy Optimization (GRPO)**. GRPO is a policy-gradient method (it directly adjusts model parameters to increase the probability of higher-reward outputs) that avoids training a separate value function (critic) by sampling a *group* of multiple model outputs (often called **rollouts**) per prompt and estimating an **advantage** (a centered-and-scaled reward signal used for policy-gradient updates) by normalizing each rollout’s reward relative to the group’s mean and standard deviation.

Two GRPO failure modes are particularly relevant when practitioners care about *best-of-N* sampling at inference time (sampling multiple outputs for the same prompt):

1. **Zero-variance groups (gradient diminishing)**: if all rollouts for a prompt are correct (all rewards = 1) or all incorrect (all rewards = 0), the within-group reward standard deviation is 0, and advantage normalization yields little or no learning signal.
2. **Diversity collapse**: during reinforcement learning fine-tuning, probability mass can concentrate on a single dominant reasoning style, reducing the chance that multiple sampled attempts contain a correct solution.

A common metric for this setting is **Pass@k** (the probability that at least one of *k* independently sampled model outputs is correct; higher is better).

### The Problem

**Transform-Augmented GRPO (TA-GRPO)** proposes to address both zero-variance groups and diversity collapse by:

- generating **semantic transformations** of each training prompt (e.g., paraphrasing, variable renaming, and format changes that preserve the underlying math problem), and
- computing **pooled advantage normalization** across the original prompt and its transformed variants.

However, TA-GRPO’s reported gains are confounded with increased rollout compute. In the TA-GRPO paper’s main configuration:

- The baseline GRPO trains with **G = 8 rollouts per prompt**.
- TA-GRPO generates **N = 3 transformed variants** per prompt and samples **G = 8 rollouts per variant**, totaling **(N+1)×G = 32 rollouts per original prompt per optimizer step**.

TA-GRPO reports large improvements in Pass@32 (percentage; higher is better) on several benchmarks using Qwen3-1.7B:

- **AMC12** (American Mathematics Contest 12-style high-school competition math): 69.88 → 79.72
- **AIME24** (American Invitational Mathematics Examination 2024 problems): 41.31 → 50.00
- **GPQA-Diamond** (198 graduate-level science multiple-choice questions): 68.69 → 73.74

(These numbers are copied from TA-GRPO Table 1: `./references/Transform-Augmented-GRPO-Improves-Pass@k/sections/Summary.md`.)

This leaves a practitioner-facing question that the original paper does not answer:

**Under a fixed rollout budget, is it better to allocate rollouts to TA-GRPO (transforms + pooled advantage normalization) or to run standard GRPO for more optimizer steps (“train longer”)?**

### Key Insight and Hypothesis

**Thesis (one sentence)**: Under a fixed total rollout budget, TA-GRPO’s semantic transformations plus pooled advantage normalization yield higher Pass@k than compute-matched “train longer” GRPO.

**Hypothesis**: TA-GRPO outperforms compute-matched GRPO-long because transformations change per-prompt success probabilities away from extremes (near 0 or near 1), producing mixed reward outcomes within each pooled group; pooled normalization then converts this reward heterogeneity into a stronger and less wasteful learning signal.

**Most likely alternative explanation (confound)**: the transformations may systematically change prompt difficulty (e.g., make prompts consistently easier or harder on average), in which case gains would not support the claimed “variance rescue” mechanism. We therefore include a pre-registered diagnostic that measures how transformation types shift difficulty relative to the original prompts.

---

## Proposed Approach

### Overview

We propose a **compute-matched comparison** between TA-GRPO and a stronger, realistic baseline: standard GRPO with the same rollout group size per prompt, but trained for more optimizer steps so that the **total number of sampled rollouts** matches TA-GRPO.

Across conditions we keep the base model, training prompts, verifiable reward definition, decoding parameters, and evaluation protocol fixed.

### Method Details

We follow TA-GRPO’s reported training setup as closely as possible:

- **Base model**: Qwen3-1.7B (see Experiments)
- **Training set size**: ~7.5k math problems
- **Batch size**: **B = 128 original prompts per optimizer step**
- **Rollouts per prompt (GRPO group size)**: **G = 8**

Let **T** be TA-GRPO’s total number of optimizer steps.

- TA-GRPO reports training for 3 epochs over 7,498 questions with batch size 128 (`./references/Transform-Augmented-GRPO-Improves-Pass@k/sections/Experiments.md`).
- Steps per epoch ≈ ceil(7,498 / 128) = 59, so **T = 3 × 59 = 177** steps.

We run three training conditions:

**Condition A (GRPO-long, compute-matched baseline)**
- Standard GRPO on original prompts only.
- Sample **G = 8 rollouts per prompt** (so 8×B rollouts per step).
- Run **4T = 708** optimizer steps so total rollouts match TA-GRPO:
  - A total rollouts: B × 8 × (4T)
  - B/C total rollouts: B × 32 × T

**Condition B (TA-GRPO, pooled advantages)**
- For each training prompt, generate **N = 3 semantic transformations** (paraphrase, variable renaming, format change) using the prompts from TA-GRPO Appendix C.1 (`./references/Transform-Augmented-GRPO-Improves-Pass@k/sections/C.1. Transformation Generation Prompts.md`).
- Sample **G = 8 rollouts** for each variant (original + 3 transforms) → **(N+1)×G = 32 rollouts per original prompt per step**.
- Compute advantages by **pooling** the mean and standard deviation over all 32 binary rewards for that original prompt group.
- Run **T = 177** optimizer steps.

**Condition C (TA-GRPO without pooled normalization; ablation)**
- Same transformed variants and rollouts as Condition B.
- Compute advantage normalization **separately within each variant** (normalize within each 8-rollout subgroup) rather than pooling across all 32 rewards.
- Run **T = 177** optimizer steps.

**Pre-registered diagnostics (mechanism checks; not part of the primary decision rule)**

1. **Transformation difficulty profile**: On a held-out subset of 200 training prompts, estimate Pass@8 (probability at least one of 8 samples is correct; higher is better) for:
   - the original prompt, and
   - each transformation type (paraphrase / variable rename / format change).

   Report (i) the mean success-rate shift (in percentage points) per transformation type relative to the original prompts, and (ii) the per-prompt variance across the four variants. Interpretation commitment: if any transform type has a large mean shift (e.g., > ±10 percentage points), we treat “variance rescue” as confounded with a systematic prompt-distribution shift and report that explicitly.

2. **Base-model difficulty sanity check**: On the same 200 prompts, if >80% are all-fail (0/8) or all-pass (8/8) under the base model, switch to Qwen3-4B or filter the training set to an intermediate-difficulty subset to keep the RLVR signal non-degenerate.

3. **Seed sensitivity**: Run at least **2 random seeds** for Conditions A and B to estimate variance in the primary metric.

### Key Innovations

- A **compute-matched “train longer” GRPO baseline** that directly answers a realistic budget-allocation question for RLVR practitioners.
- A **pre-registered mechanism diagnostic** that tests whether transformations primarily (i) rescue reward variance within groups or (ii) simply change prompt difficulty.

---

## Related Work

### Field Overview

RLVR (reinforcement learning with verifiable rewards) uses programmatic verifiers instead of learned reward models, enabling scalable post-training when correctness can be checked automatically (e.g., by exact-match answers or executing code). GRPO is widely used in RLVR because it removes the need for a learned critic, but it can be sensitive to reward variance, group size, and diversity collapse. Recent GRPO-family work addresses these issues via modified advantage normalization, uncertainty-aware scaling, adaptive sampling / selective rollouts, diversity-aware reward shaping, and data augmentation.

TA-GRPO sits in the augmentation-and-normalization branch: it changes the training distribution via semantic transformations and modifies advantage normalization by pooling statistics across variants.

### Related Papers

- **[Transform-Augmented GRPO Improves Pass@k](./references/Transform-Augmented-GRPO-Improves-Pass@k/meta/meta_info.txt)**: Introduces TA-GRPO using semantic prompt transformations and pooled advantage normalization to improve Pass@k.
- **[No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](./references/No-Prompt-Left-Behind-Exploiting-Zero-Variance-Prompts-in-LLM-Reinforcement-Learning-via-Entropy-Guided-Advantage-Shaping/meta/meta_info.txt)**: Proposes RL-ZVP (reinforcement learning with zero-variance prompts) to extract learning signal when all rollouts in a group have identical rewards.
- **[Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives](./references/Reinforce-Ada-An-Adaptive-Sampling-Framework-for-Reinforce-Style-LLM-Training/meta/meta_info.txt)**: Allocates sampling effort adaptively to obtain informative reward mixtures before updating.
- **[MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning](./references/MC-GRPO-Median-Centered-Group-Relative-Policy-Optimization-for-Small-Rollout-Reinforcement-Learning/meta/meta_info.txt)**: Uses robust median/median absolute deviation (MAD) normalization to stabilize GRPO in small-rollout regimes.
- **[DeepSeekMath](https://arxiv.org/abs/2402.03300)**: Introduces GRPO-style RLVR for math reasoning at scale.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Demonstrates emergent reasoning improvements from GRPO-style RLVR and popularizes large-scale RLVR pipelines.
- **[Why GRPO Needs Normalization: A Local-Curvature Perspective](https://arxiv.org/abs/2601.23135)**: Analyzes why normalization is important for GRPO stability and learning dynamics.
- **[SEED-GRPO](https://arxiv.org/abs/2505.12346)**: Uses semantic entropy (an uncertainty proxy) to modulate GRPO updates.
- **[It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977)**: Connects GRPO to Direct Preference Optimization (DPO)-style objectives and studies group-size trade-offs.
- **[GRPO-MA: Multi-Answer Generation in GRPO for Stable and Efficient Chain-of-Thought Training](https://arxiv.org/abs/2509.24494)**: Samples multiple answers per reasoning trace to reduce advantage variance.
- **[DRA-GRPO: Exploring Diversity-Aware Reward Adjustment for R1-Zero-Like Training of Large Language Models](https://arxiv.org/abs/2505.09655)**: Uses diversity-aware reward shaping to mitigate diversity collapse.
- **[F-GRPO: Don’t Let Your Policy Learn the Obvious and Forget the Rare](https://arxiv.org/abs/2602.06717)**: Analyzes how group-relative updates can suppress rare-correct modes and proposes a focal-style weighting to preserve diversity.
- **[Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts](https://arxiv.org/abs/2506.02177)**: Filters uninformative prompts before rollout generation to reduce wasted sampling.
- **[DAPO](https://arxiv.org/abs/2503.14476)**: An open-source GRPO-family system with dynamic sampling and other stability improvements.
- **[OREO: Offline RL for Multi-Step Reasoning](https://arxiv.org/abs/2412.16145)**: Uses offline reinforcement learning and value functions for multi-step reasoning.
- **[Back to Basics: Revisiting REINFORCE Style Optimization](https://arxiv.org/abs/2402.14740)**: Argues that simpler REINFORCE-style methods can be strong baselines for alignment and post-training.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Motivates verifiable reward settings and process-level verification.
- **[GPQA](https://arxiv.org/abs/2311.12022)**: Introduces GPQA, a graduate-level science QA benchmark (including the 198-question Diamond subset).
- **[OlympiadBench](https://arxiv.org/abs/2402.14008)**: Introduces OlympiadBench, a challenging olympiad-level math and physics benchmark.
- **[What is the Alignment Objective of GRPO?](https://arxiv.org/abs/2502.18548)**: Studies objective interpretations and behaviors of GRPO-family methods.
- **[Hard Examples Are All You Need](https://arxiv.org/abs/2508.14094)**: Shows that selecting hard examples can substantially improve GRPO efficiency under limited budgets.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Prompt transformations + pooled normalization | Create semantic variants and pool reward statistics across variants | [TA-GRPO](./references/Transform-Augmented-GRPO-Improves-Pass@k/meta/meta_info.txt) | Pass@k on math and OOD tasks | Transform quality confounds; added preprocessing overhead |
| Learning from zero-variance groups | Extract learning signal even when all rewards in a group are identical | [RL-ZVP](./references/No-Prompt-Left-Behind-Exploiting-Zero-Variance-Prompts-in-LLM-Reinforcement-Learning-via-Entropy-Guided-Advantage-Shaping/meta/meta_info.txt) | Math RLVR | Advantage shaping may be sensitive to heuristics |
| Adaptive / selective sampling | Spend rollouts where reward variance exists | [Reinforce-Ada](./references/Reinforce-Ada-An-Adaptive-Sampling-Framework-for-Reinforce-Style-LLM-Training/meta/meta_info.txt), [GRESO](https://arxiv.org/abs/2506.02177), [DAPO](https://arxiv.org/abs/2503.14476) | Math RLVR | Added system complexity; adaptive policies introduce extra hyperparameters |
| Robust advantage estimation | Use robust statistics to stabilize group-relative baselines | [MC-GRPO](./references/MC-GRPO-Median-Centered-Group-Relative-Policy-Optimization-for-Small-Rollout-Reinforcement-Learning/meta/meta_info.txt) | GSM8K / math | Primarily targets small group sizes |
| Diversity-aware objectives | Encourage diverse correct modes rather than one dominant mode | [DRA-GRPO](https://arxiv.org/abs/2505.09655), [F-GRPO](https://arxiv.org/abs/2602.06717) | Pass@k | May require additional similarity models or careful weighting |

### Closest Prior Work

1) **TA-GRPO** (`./references/Transform-Augmented-GRPO-Improves-Pass@k/meta/meta_info.txt`) proposes semantic transformations plus pooled normalization and reports large Pass@k gains. However, its headline results do not control for rollout compute, because TA-GRPO uses 4× more rollouts per prompt per step than the GRPO baseline. Our proposal isolates the decision a practitioner faces under a fixed rollout budget by comparing against GRPO-long with matched total rollouts.

2) **RL-ZVP** (`./references/No-Prompt-Left-Behind-Exploiting-Zero-Variance-Prompts-in-LLM-Reinforcement-Learning-via-Entropy-Guided-Advantage-Shaping/meta/meta_info.txt`) also targets zero-variance groups, but uses entropy-guided advantage shaping rather than data augmentation and pooled statistics. Our proposal does not attempt to beat RL-ZVP; instead, it tests whether TA-GRPO’s specific transform+pooling mechanism provides value beyond simple compute reallocation.

3) **Reinforce-Ada / selective rollouts** (`./references/Reinforce-Ada-An-Adaptive-Sampling-Framework-for-Reinforce-Style-LLM-Training/meta/meta_info.txt`, https://arxiv.org/abs/2506.02177) allocate sampling adaptively to maintain reward variance. Our study is complementary: it compares a fixed-budget allocation choice (“more variants per step” vs “more steps”) under a controlled compute-matched design.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [TA-GRPO](./references/Transform-Augmented-GRPO-Improves-Pass@k/meta/meta_info.txt) | Prompt transforms + pooled advantage normalization | Not compute-matched vs “train longer” GRPO | Add GRPO-long compute-matched baseline; keep total rollouts fixed | Directly answers the rollout-budget allocation question |
| [RL-ZVP](./references/No-Prompt-Left-Behind-Exploiting-Zero-Variance-Prompts-in-LLM-Reinforcement-Learning-via-Entropy-Guided-Advantage-Shaping/meta/meta_info.txt) | Advantage shaping for zero-variance prompt groups | Different mechanism; not about transforms | Focus on transform+pooling vs compute allocation | Clarifies when transformations add value vs alternative fixes |
| [Reinforce-Ada](./references/Reinforce-Ada-An-Adaptive-Sampling-Framework-for-Reinforce-Style-LLM-Training/meta/meta_info.txt) | Adaptive sampling to obtain mixed rewards before updates | Extra adaptive machinery | Fixed-budget, fixed-protocol comparison | Minimal, decisive, reproducible test of a common confound |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen/Qwen3-1.7B-Base | 1.7B | https://huggingface.co/Qwen/Qwen3-1.7B-Base | Matches TA-GRPO small-scale setting |
| (fallback) Qwen/Qwen3-4B | 4B | https://huggingface.co/Qwen/Qwen3-4B | Use only if the difficulty sanity check indicates 1.7B is too easy/hard |

**Training Data (if applicable):**

We train on **PrimeIntellect/Hendrycks-Math**, a ~7.5k-problem version of the Hendrycks et al. MATH training set containing competition-level math word problems with short final answers (verifiable by exact match after answer extraction). As a larger fallback, we can use **DAPO-Math-17k**, a 17k-prompt math RLVR training set released with DAPO.

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| PrimeIntellect/Hendrycks-Math | RLVR training | ~7.5k prompts | https://huggingface.co/datasets/PrimeIntellect/Hendrycks-Math | apache-2.0 (derived from Hendrycks et al. MATH; verify dataset card) |
| (fallback) DAPO-Math-17k | RLVR training | 17k prompts | https://arxiv.org/abs/2503.14476 (project page provides dataset) | Check dataset card |

Notes:
- Some competition-math datasets have takedown/copyright restrictions on certain mirrors. The core comparison (A vs B) is internal and does not depend on reproducing TA-GRPO’s absolute numbers, but the verification run must use an accessible, license-compatible source.

**Other Resources (if applicable):**
- **Transformation generation prompts**: TA-GRPO Appendix C.1 (`./references/Transform-Augmented-GRPO-Improves-Pass@k/sections/C.1. Transformation Generation Prompts.md`).
- **Transformation generation model**: TA-GRPO used GPT-4. Our prompts are standard math questions and are unlikely to trigger Azure OpenAI content filters, but we can also generate transformations using non-OpenAI API models (e.g., Claude, Gemini) or an open-weight model to avoid any dependency.

**Training hyperparameters (from TA-GRPO Experiments section):**
- Group size **G = 8**, learning rate 1e-6, batch size 128 prompts, **KL penalty coefficient** 0.01 (Kullback–Leibler divergence regularization that discourages drifting too far from the **reference model**, i.e., a frozen copy of the pre-RL policy), and clip ratio [0.8, 1.2] (policy ratio clipping range; closer to 1 is more conservative) (`./references/Transform-Augmented-GRPO-Improves-Pass@k/sections/Experiments.md`).
- Train for 3 epochs for the TA-GRPO-equivalent schedule (**T ≈ 177** steps); GRPO-long runs **4×** as many steps (**≈ 708**).

**Resource Estimate**:

- **Compute budget (GPU-hours)**: expected **≤ 768 GPU-hours**.
  - TA-GRPO reports training Qwen3-1.7B for 3 epochs on **8× NVIDIA A100 (80GB)** GPUs (`./references/Transform-Augmented-GRPO-Improves-Pass@k/sections/Experiments.md`), and the arXiv paper notes this run is on the order of **~12 hours** on that hardware (arXiv:2601.22478, Appendix C; if unavailable in the local scrape, see https://arxiv.org/abs/2601.22478).
  - Using that as a coarse reference: one full training run (A or B or C) is ~8 GPUs × 12 h ≈ **96 GPU-hours**.
  - Planned runs: (A,B) × 2 seeds + C × 1 seed = 5 runs → ~**480 GPU-hours**.
  - Benchmark evaluation and diagnostics add additional inference cost but should remain within the remaining budget.

- **GPU memory**: Qwen3-1.7B fits on a single A100 80GB; multi-GPU is mainly for throughput during rollout generation and policy updates.

- **API usage**: one-time transformation generation for ~7.5k prompts × 3 transforms. Token cost depends on prompt length; a conservative order-of-magnitude is a few million tokens total.

### Benchmarks and Metrics

We evaluate using **Pass@k** for k ∈ {1, 8, 16, 32} (probability at least one of k sampled outputs is correct; higher is better), following TA-GRPO’s evaluation protocol (temperature 0.7, 32 samples per test question, exact-match after answer extraction).

**Primary decision metric**: mean Pass@32 across three accessible benchmarks.

- Default (no gated datasets): **AMC12**, **AIME24**, **Minerva-Math**.
- Optional out-of-distribution (OOD) add-on (if access is available): **GPQA-Diamond**.

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| AMC12 | American Mathematics Contest 12-style competition math problems | Pass@k | test | (e.g., Qwen2.5-Math eval data) https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation/data/amc12 | Use the Qwen2.5-Math evaluation harness where possible; otherwise custom answer extraction + exact match (numeric answer) |
| AIME24 | 30 AIME 2024 problems (integer answers 0–999) | Pass@k | test | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 | Custom answer extraction + exact match |
| AIME25 | 30 AIME 2025 problems (integer answers 0–999) | Pass@k | test | https://huggingface.co/datasets/math-ai/aime25 | Custom answer extraction + exact match |
| OlympiadBench (text-only) | A text-only subset of OlympiadBench, a large collection of olympiad-level math/physics problems | Pass@k | test | https://huggingface.co/datasets/Hothan/OlympiadBench | Prefer official repo eval; otherwise exact match where applicable |
| Minerva-Math | 272 applied math/science-context problems with short final answers | Pass@k | test | https://huggingface.co/datasets/svc-huggingface/minerva-math | Custom answer extraction + exact match |
| GPQA-Diamond (optional) | 198 graduate-level science multiple-choice questions (4 options; accuracy-style evaluation) | Pass@k | test | https://huggingface.co/datasets/Idavidrein/gpqa (gated) | lm-eval-harness GPQA task preferred |

### Main Results

#### Results Table

Published Pass@32 baselines from TA-GRPO Table 1 (Qwen3-1.7B; percentages; higher is better) are copied from `./references/Transform-Augmented-GRPO-Improves-Pass@k/sections/Summary.md`. (Pass@32 is the probability at least one of 32 sampled outputs is correct; higher is better.)

| Method | Rollouts per original prompt per step | Steps | AMC12 Pass@32 | AIME24 Pass@32 | AIME25 Pass@32 | OlympiadBench Pass@32 | Minerva Pass@32 | GPQA-Diamond Pass@32 | Source | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Base (no RL) | 0 | 0 | 65.06 | 30.00 | 30.00 | 60.09 | 48.53 | 57.58 | TA-GRPO Table 1 | Qwen3-1.7B |
| GRPO (paper) | 8 | ~177 | 69.88 | 41.31 | 30.00 | 66.62 | 50.37 | 68.69 | TA-GRPO Table 1 | Not compute-matched |
| TA-GRPO (paper) | 32 | ~177 | 79.72 | 50.00 | 33.33 | 68.84 | 52.94 | 73.74 | TA-GRPO Table 1 | Uses 4× rollouts per step |
| **A: GRPO-long (compute-matched)** | 8 | **708** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | - | 4× steps to match total rollouts |
| **B: TA-GRPO (compute-matched)** | 32 | **177** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | - | Same total rollouts as A |
| **C: TA-GRPO without pooling** | 32 | **177** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | - | Per-variant normalization |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| TA-GRPO without pooling (C) | Remove pooled normalization across variants | C < B if pooling remains essential under compute-matching |

### Analysis (Optional)

- **Transformation difficulty profile** and **base-model difficulty sanity check** (see Proposed Approach).
- **Zero-variance group rate during training**: fraction of prompt groups with std(reward) = 0.

---

## Success Criteria

**Criterion 1: Compute-matched comparison (primary)**
- Hypothesis: TA-GRPO (B) achieves higher mean Pass@32 than GRPO-long (A) under equal total rollouts.
- Validation / decision rule: Let Δ be the difference in mean Pass@32 between B and A averaged over the primary benchmark set.
  - If Δ is meaningfully positive (e.g., ≳ 2 percentage points), this supports using TA-GRPO under a rollout budget.
  - If Δ is meaningfully negative (e.g., ≲ −2 percentage points), this supports allocating the same budget to longer GRPO training.
  - If |Δ| is small (e.g., < 2 percentage points) or seed variance is comparable to Δ, treat the result as inconclusive for decision-making.

**Criterion 2: Pooling contributes under compute-matching (secondary)**
- Hypothesis: pooled normalization is an essential component of TA-GRPO.
- Validation: mean Pass@32 of C is lower than mean Pass@32 of B on the same benchmark set.

---

## Impact Statement

If TA-GRPO beats a compute-matched “train longer” GRPO baseline, it supports using semantic prompt transformations plus pooled normalization as a rollout-efficient way to improve best-of-N performance in RLVR. If it does not, the study provides an actionable negative result: when rollout compute is the binding constraint, allocating that budget to more GRPO steps is as good or better than adding transformation preprocessing and pooled normalization.

---

## References

- [Transform-Augmented GRPO Improves Pass@k](./references/Transform-Augmented-GRPO-Improves-Pass@k/meta/meta_info.txt) - Le et al., 2026
- [No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](./references/No-Prompt-Left-Behind-Exploiting-Zero-Variance-Prompts-in-LLM-Reinforcement-Learning-via-Entropy-Guided-Advantage-Shaping/meta/meta_info.txt) - Le et al., 2025
- [Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives](./references/Reinforce-Ada-An-Adaptive-Sampling-Framework-for-Reinforce-Style-LLM-Training/meta/meta_info.txt) - Xiong et al., 2025
- [MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning](./references/MC-GRPO-Median-Centered-Group-Relative-Policy-Optimization-for-Small-Rollout-Reinforcement-Learning/meta/meta_info.txt) - Kim, 2026
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - DeepSeek-AI et al., 2025
- [Why GRPO Needs Normalization: A Local-Curvature Perspective](https://arxiv.org/abs/2601.23135) - 2026
- [SEED-GRPO](https://arxiv.org/abs/2505.12346) - Chen et al., 2025
- [It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977) - Wu et al., 2024
- [GRPO-MA](https://arxiv.org/abs/2509.24494) - Wang et al., 2025
- [DRA-GRPO](https://arxiv.org/abs/2505.09655) - Chen et al., 2025
- [F-GRPO: Don’t Let Your Policy Learn the Obvious and Forget the Rare](https://arxiv.org/abs/2602.06717) - Plyusov et al., 2026
- [Act Only When It Pays](https://arxiv.org/abs/2506.02177) - Zheng et al., 2025
- [DAPO](https://arxiv.org/abs/2503.14476) - Yu et al., 2025
- [OREO](https://arxiv.org/abs/2412.16145) - Wang et al., 2024
- [Back to Basics: Revisiting REINFORCE Style Optimization](https://arxiv.org/abs/2402.14740) - Ahmadian et al., 2024
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [GPQA](https://arxiv.org/abs/2311.12022) - Rein et al., 2023
- [OlympiadBench](https://arxiv.org/abs/2402.14008) - He et al., 2024
- [What is the Alignment Objective of GRPO?](https://arxiv.org/abs/2502.18548) - 2025
- [Hard Examples Are All You Need](https://arxiv.org/abs/2508.14094) - 2025
