# untitled

# Does Autoregressive-Order RL Post-Training Reduce Order Robustness in Diffusion Language Models? A JustGRPO Mechanism Test

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Setting**: Inference-only evaluation of released checkpoints (no new training)
- **Automation**: Fully automated evaluation (exact-match arithmetic answers; string-parsed synthetic benchmark)
- **Compute budget**: <=768 A100-80GB GPU-hours (expected << 100 GPU-hours)

## Introduction

### Context and Motivation

Diffusion language models (dLLMs) are an alternative to autoregressive (AR) language models. Instead of generating tokens strictly left-to-right, dLLMs iteratively denoise a masked token canvas and can update many positions in parallel. This makes dLLMs attractive for settings where the desired output structure does not match the natural order of reasoning.

A common practical requirement is **output-order constraints**: the system must present the final answer before an explanation (e.g., user-interface requirements, structured templates, or instruction-following policies like "answer first, then explain"). In AR models, output-order constraints can force the model to commit to an answer before producing intermediate reasoning, which may reduce accuracy.

Recent work "Thinking Out of Order" (arXiv:2601.22035) reports that dLLMs trained from scratch (LLaDA-8B-Instruct) maintain accuracy under **Answer-First** prompts, while an AR baseline (Qwen2.5-7B-Instruct) suffers large relative drops. The same paper also reports that a diffusion model distilled from an AR model (Dream-7B) shows intermediate degradation, suggesting that **training methodology** may affect order robustness.

Separately, "The Flexibility Trap" (arXiv:2601.15165) argues that arbitrary-order generation can harm reasoning potential in dLLMs and proposes **JustGRPO**, a simple reinforcement learning post-training recipe that applies Group Relative Policy Optimization (GRPO) in **autoregressive order**. The public checkpoint `nzl-thu/LLaDA-Instruct-JustGRPO` reports substantial improvements on reasoning benchmarks, but it does not report performance under Answer-First output constraints.

### The Problem

These two lines of work create a decision-relevant tension:

- **Order robustness as an advantage**: dLLMs may be preferable when output order conflicts with reasoning order (2601.22035).
- **AR-order post-training for reasoning gains**: AR-order RL post-training may improve reasoning accuracy (2601.15165), but could also cause the model to internalize AR-style dependence on output order.

If AR-order RL post-training substantially reduces order robustness, then practitioners may face a concrete trade-off: improved standard (CoT-First) reasoning accuracy versus worse performance under Answer-First formatting constraints.

### Key Insight and Hypothesis

**Key insight.** Order robustness is best viewed as a stability property under a controlled distribution shift (the same task, but with the output fields permuted). Because JustGRPO changes the model via post-training, it can change this stability even if the inference algorithm remains diffusion-style.

**Hypothesis.** Compared to the base `GSAI-ML/LLaDA-8B-Instruct` checkpoint, the AR-order GRPO checkpoint `nzl-thu/LLaDA-Instruct-JustGRPO` will show a larger performance drop under Answer-First prompts, i.e., reduced order robustness.

**Why this could be wrong.** Order robustness may be primarily driven by diffusion inference dynamics (e.g., confidence-based remasking) rather than post-training. If so, JustGRPO could preserve (or even improve) Answer-First robustness while improving overall accuracy.

---

## Proposed Approach

### Overview

We propose a **mechanism test**: measure whether AR-order RL post-training changes a diffusion LM's sensitivity to output-order constraints.

We compare a base diffusion checkpoint to a JustGRPO diffusion checkpoint under two prompt formats:

- **CoT-First**: output `Reasoning:` then `Answer:`
- **Answer-First**: output `Answer:` then `Reasoning:`

We compute order robustness using a ratio metric that is less confounded by overall capability changes.

### Method Details

#### A. Models

Main models (frozen weights; no training):

- **LLaDA-8B-Instruct** (diffusion; trained from scratch): `GSAI-ML/LLaDA-8B-Instruct`
- **LLaDA-Instruct-JustGRPO** (diffusion; AR-order GRPO post-training): `nzl-thu/LLaDA-Instruct-JustGRPO`

Anchor baseline (AR model; context only):

- **Qwen2.5-7B-Instruct** (autoregressive): `Qwen/Qwen2.5-7B-Instruct`

#### B. Benchmarks

1. **ReasonOrderQA** (from arXiv:2601.22035): a synthetic benchmark of 1000 problems with a fixed structured output template with `Answer:` / `Reasoning:` / `Retrieval:` delimiters, designed to measure performance under output-order constraints.
   - If an official dataset release is unavailable, verification will reproduce the generator described in 2601.22035 (difficulty levels D1-D4 with arithmetic formulas and a noisy ~1000-token context containing secret-key statements).

2. **GSM8K** (https://huggingface.co/datasets/gsm8k): grade-school math word problems with numeric final answers.

#### C. Prompt templates and parsing

- For ReasonOrderQA, follow the structured template described in 2601.22035 and parse:
  - `Answer:`: integer answer
  - `Retrieval:`: extracted secret keys (for retrieval F1; optional analysis)
- For GSM8K, enforce a strict `Answer: <number>` line and evaluate exact match on the extracted number.

#### D. Metrics

Primary metric (order robustness ratio):

- **r = Acc(Answer-First) / Acc(CoT-First)**

Secondary metrics:

- Absolute gap: `Acc(CoT-First) - Acc(Answer-First)`
- Format compliance rate: fraction of outputs containing both `Answer:` and `Reasoning:` in the required order.

### Key Innovations

- **Order-robustness evaluation of RL post-training**: tests a practical, previously unreported trade-off for a widely shared post-training recipe (JustGRPO).
- **Ratio-based robustness metric**: uses a robustness ratio to avoid conflating robustness changes with overall accuracy changes.
- **Decision guidance**: produces an actionable recommendation about when to use AR-order RL post-training for diffusion LMs.

---

## Related Work

### Field Overview

This proposal relates to three areas: (i) diffusion language models and their decoding dynamics, (ii) output-order constraints and non-monotonic generation, and (iii) post-training and reinforcement learning for dLLMs. Recent work suggests dLLMs can support "compute order" that differs from "output order" (2601.22035), but other work argues that arbitrary-order generation can reduce reasoning potential and that AR-order training can improve it (2601.15165). The literature does not clearly report whether AR-order post-training preserves or harms the order-robustness advantage.

### Related Papers

- **[Thinking Out of Order](https://arxiv.org/abs/2601.22035)**: Defines order robustness and introduces ReasonOrderQA; reports strong robustness for LLaDA trained from scratch.
- **[The Flexibility Trap / JustGRPO](https://arxiv.org/abs/2601.15165)**: Argues arbitrary-order generation limits reasoning potential; proposes AR-order GRPO post-training and releases a checkpoint.
- **[Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992)**: Scales masked diffusion LMs to 8B and reports strong reasoning performance.
- **[Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)**: Diffusion LM distilled from AR weights; relevant for training-methodology effects on order robustness.
- **[Simple and Effective Masked Diffusion Language Models (MDLM)](https://arxiv.org/abs/2406.07524)**: Masked diffusion formulation and decoding heuristics.
- **[Loopholing Discrete Diffusion](https://arxiv.org/abs/2510.19304)**: Studies information loss across diffusion steps (sampling wall) and proposes a trained latent bypass.
- **[Diffusion-LM](https://arxiv.org/abs/2205.14217)**: Early continuous diffusion approach for text.
- **[D3PM](https://arxiv.org/abs/2107.03006)**: Foundational discrete diffusion with absorbing states.
- **[SEDD](https://arxiv.org/abs/2310.16834)**: Score-entropy discrete diffusion achieving competitive likelihoods.
- **[Diffusion of Thoughts](https://arxiv.org/abs/2402.07754)**: Uses diffusion-style reasoning trajectories.
- **[MDPO](https://arxiv.org/abs/2508.13148)**: Preference alignment for diffusion LMs.
- **[dUltra](https://arxiv.org/abs/2512.21446)**: GRPO-style alignment for parallel decoding.
- **[ESPO](https://arxiv.org/abs/2512.03759)**: RL for diffusion LLMs with principled objectives.
- **[IGPO](https://arxiv.org/abs/2509.17114)**: Inpainting-guided policy optimization for diffusion LLMs.
- **[Learning Unmasking Policies for Diffusion LMs](https://arxiv.org/abs/2512.09106)**: Trains auxiliary unmasking policies with GRPO; relevant for how training affects decoding behavior.
- **[Empirical Analysis of Decoding Biases in Masked Diffusion Models (UNCODE)](https://arxiv.org/abs/2508.13021)**: Shows decoding heuristics can strongly affect reasoning performance.
- **[Fast-Decoding Diffusion LMs via Progress-Aware Confidence Schedules (SchED)](https://arxiv.org/abs/2512.02892)**: Early-exit schedules for dLLMs; relevant for inference-time dynamics.
- **[Where-to-Unmask](https://arxiv.org/abs/2602.09501)**: Learning-to-rank unmasking order using a ground-truth-derived oracle.
- **[FOCUS](https://arxiv.org/abs/2601.23278)**: Inference efficiency system for dLLMs via attention-delta-based token eviction.
- **[Non-Monotonic Sequential Text Generation](https://arxiv.org/abs/1902.00536)**: Early analysis of generation orders beyond left-to-right.
- **[Insertion Transformer](https://arxiv.org/abs/1902.03268)**: Insertion-based non-autoregressive generation related to order flexibility.
- **[Levenshtein Transformer](https://arxiv.org/abs/1905.11006)**: Edit-based generation that decouples output structure from generation steps.
- **[MaskGIT](https://arxiv.org/abs/2202.04200)**: Masked iterative generation in vision; conceptual ancestor to masked diffusion decoding.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Typical eval | Known limitations |
|---|---|---|---|---|
| Diffusion LM backbones | Masked denoising generation over discrete tokens | LLaDA; Dream; MDLM; SEDD | LM + reasoning + code | Sensitive to decoding choices |
| Order robustness analyses | Measure sensitivity to output-order constraints | Thinking Out of Order; non-monotonic gen | GSM8K/Math + synthetic | Benchmarks may be synthetic |
| RL post-training for dLLMs | Update diffusion LM with RL (often GRPO variants) | JustGRPO; ESPO; MDPO; IGPO; dUltra | reasoning/code | Objectives may change decoding behavior |
| Non-monotonic / insertion generation | Decouple generation steps from output order | Insertion/Levenshtein; related NAR | translation/editing | Different model classes |

### Closest Prior Work

- **Thinking Out of Order (2601.22035)** measures order robustness for LLaDA and Dream, but does not evaluate AR-order RL post-training (JustGRPO).
- **The Flexibility Trap / JustGRPO (2601.15165)** proposes AR-order RL post-training and releases a checkpoint, but does not report Answer-First robustness.

**Novelty Kill Search Summary:** Searched for the exact combination of {"JustGRPO" AND "Answer-First"}, {"JustGRPO" AND "order robustness"}, {"autoregressive order" AND "diffusion" AND "order robustness"}, and scanned local proposal indexes for "JustGRPO" / "Answer-First" / "CoT-First". As of 2026-02-20, no prior work was found that reports order-robustness measurements for the released `LLaDA-Instruct-JustGRPO` checkpoint.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Thinking Out of Order (2601.22035) | Defines/measures order robustness; introduces ReasonOrderQA | No RL post-training checkpoints tested | Evaluate JustGRPO checkpoint under the same robustness protocol | Answers whether AR-order post-training preserves robustness |
| Flexibility Trap / JustGRPO (2601.15165) | AR-order GRPO post-training for dLLMs; large reasoning gains | No Answer-First evaluation | Add robustness evaluation + decision rule | Reveals a practical trade-off for deployment |
| LLaDA (2502.09992) | Base diffusion LM architecture + decoding | No output-order stress test | Connect LLaDA family to order robustness trade-offs | Guides checkpoint choice for structured outputs |

---

## Experiments

### Experimental Setup

**Decoding settings (diffusion models):** Follow 2601.22035 default: low-confidence remasking; generation length L=256; diffusion steps T=256. If the official implementation exposes additional parameters (temperature, block size), keep them fixed across models.

**Main conditions (3 models):**
1. LLaDA-8B-Instruct (diffusion)
2. LLaDA-Instruct-JustGRPO (diffusion)
3. Qwen2.5-7B-Instruct (AR anchor)

**Seeds / variance:** run 3 decoding seeds for diffusion models if any stochasticity is present (otherwise report deterministic results and confirm determinism). Use the same seeds across conditions.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| LLaDA-8B-Instruct | 8B | https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct | diffusion LM |
| LLaDA-Instruct-JustGRPO | 8B | https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO | GRPO post-trained |
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | AR anchor |

**Training Data:**
- No training data needed (inference-only evaluation).

**Resource Estimate (rough, to be refined by verification):**
- Total generations: 2 prompt orders * (1000 ReasonOrderQA + 1319 GSM8K) ~= 4638 per model.
- Expect total wall-clock on the order of hours to tens of hours on 1-4 A100s for diffusion models; fits within 768 GPU-hours.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ReasonOrderQA | Synthetic benchmark for order robustness with retrieval + arithmetic reasoning | reasoning accuracy; (optional) retrieval F1; format compliance; robustness ratio r | fixed 1000 | N/A (generate from paper) | Custom script (implement paper generator) |
| GSM8K | Grade-school math problems | exact-match accuracy of numeric answer; format compliance; robustness ratio r | test | https://huggingface.co/datasets/gsm8k | Standard GSM8K parsing |

### Main Results

#### Comparability Rules (CRITICAL)
All rows must use:
- Same datasets and splits
- Same prompt templates (CoT-First and Answer-First)
- Same generation length cap (L=256)
- Same diffusion steps (T=256) for diffusion models
- Same decoding temperature/sampling settings per model family

#### Results Table

| Model | Benchmark | Acc(CoT-First) (mean+-std) | Acc(Answer-First) (mean+-std) | r=AF/CoT | Format compliance (AF) | Source | Notes |
|---|---|---:|---:|---:|---:|---|---|
| LLaDA-8B-Instruct | ReasonOrderQA | TBD | TBD | TBD | TBD | - | to be measured |
| LLaDA-Instruct-JustGRPO | ReasonOrderQA | TBD | TBD | TBD | TBD | - | to be measured |
| Qwen2.5-7B-Instruct | ReasonOrderQA | TBD | TBD | TBD | TBD | - | anchor only |
| LLaDA-8B-Instruct | GSM8K | TBD | TBD | TBD | TBD | - | to be measured |
| LLaDA-Instruct-JustGRPO | GSM8K | TBD | TBD | TBD | TBD | - | to be measured |
| Qwen2.5-7B-Instruct | GSM8K | TBD | TBD | TBD | TBD | - | anchor only |

### Ablation Studies

| Variant | What is changed | Expected finding |
|---|---|---|
| Deterministic decoding | temp=0 / greedy updates | Confirms results are not due to sampling noise |
| Shorter L (64) on ReasonOrderQA | L=64, T=64 | Tests whether any degradation is amplified by long canvases (per 2601.22035 breakdown) |

### Experimental Rigor

- **Sanity checks**:
  - Reproduce one published number from 2601.22035 on ReasonOrderQA Table 3 (e.g., Qwen AR gap magnitude or diffusion strategy ordering) within tolerance.
  - Verify the Answer-First prompt template is actually enforced (measure format compliance).
- **Confounders**:
  - Overall capability change: primary metric is r=AF/CoT.
  - Parsing brittleness: report format compliance and treat non-compliant outputs as incorrect.

---

## Success Criteria

**Hypothesis (directional):** The JustGRPO checkpoint reduces order robustness (lower r) compared to the base LLaDA checkpoint.

**Decision Rule:**
- **Proceed / conclude trade-off** if on ReasonOrderQA: `r_JustGRPO <= r_LLaDA - 0.10` OR `Acc(CoT)-Acc(AF)` increases by >=5.0 absolute points.
- **Refute (robustness preserved)** if `|r_JustGRPO - r_LLaDA| <= 0.03`.
- **Pivot** if r changes are small but absolute accuracies change substantially; then report the practical trade-off curve (Acc(AF), Acc(CoT)) without claiming robustness erosion.

---

## Impact Statement

If AR-order RL post-training substantially reduces order robustness, practitioners can treat it as a concrete trade-off: use JustGRPO-style post-training for standard CoT-first reasoning, but avoid it when output-order constraints require answers before reasoning. If robustness is preserved, the result is also decision-relevant: AR-order RL can be applied without sacrificing one of diffusion decoding's distinctive deployment advantages.

---

## References

- Thinking Out of Order: When Output Order Stops Reflecting Reasoning Order in Diffusion Language Models (arXiv:2601.22035)
- The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models (arXiv:2601.15165)
- Large Language Diffusion Models (arXiv:2502.09992)
- Dream 7B: Diffusion Large Language Models (arXiv:2508.15487)
- Simple and Effective Masked Diffusion Language Models (arXiv:2406.07524)
- FOCUS: DLLMs Know How to Tame Their Compute Bound (arXiv:2601.23278)
- (Additional references are cited inline above.)
