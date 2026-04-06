# untitled

# Does Meta-Experience Learning Need Policy-Gradient RL? A Compute-Matched NLL-Only Ablation

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models can be improved at mathematical reasoning by training them with feedback from automated answer checkers. For example, given a math problem with a known final answer, we can verify whether a model’s generated solution is correct by extracting its final answer and comparing it to the ground truth using a symbolic verifier. This training paradigm—often called **reinforcement learning with verifiable rewards (RLVR)**—uses programmatic verification instead of human preference labels, and has become a common ingredient in recent reasoning-focused post-training systems.

A practical drawback of RLVR is that its rewards are typically **sparse and trajectory-level**: the verifier checks only the final answer, providing no direct supervision about which intermediate reasoning steps were wrong. This sparsity makes policy-gradient methods such as **Group Relative Policy Optimization (GRPO)** (a PPO-style method that normalizes rewards within each sampled group) slower or less stable in regimes where correct rollouts are rare.

**Meta-Experience Learning (MEL)** proposes to densify RLVR training by constructing **meta-experiences** from pairs of correct and incorrect rollouts: a localized bifurcation point where reasoning diverges, a critique that explains the failure, and an abstract heuristic intended to generalize. MEL replay-validates these meta-experiences by injecting them into the prompt and re-solving the same problem, then **internalizes** validated meta-experiences into model weights via a token-level negative log-likelihood (NLL) objective. MEL reports consistent gains over GRPO across several competition-math benchmarks and model scales **[MEL](./references/meta/meta_info.txt)**.

### The Problem

Despite MEL’s strong results, it is unclear **which part of MEL’s learning signal is actually responsible for the gains**. MEL optimizes a joint objective

\(J(\theta)=J_{\text{RLVR}}(\theta)+J_{\text{MEL}}(\theta)\),

where \(J_{\text{RLVR}}\) is the GRPO policy-gradient objective on verifiable outcome rewards, and \(J_{\text{MEL}}\) is an auxiliary NLL objective that predicts validated meta-experience text conditioned on a retrospective context \([I,x,y^+,y^-]\) **[MEL, Eq. 7–10](./references/sections/Joint%20Training%20Objective.md)**.

This creates an important ambiguity for practitioners:

- If MEL’s gains primarily require **policy-gradient RL updates**, then MEL should be treated as an RL method whose benefits may not be replicable with supervised learning.
- If MEL’s gains are mostly due to **extra supervised updates** on high-quality self-distilled heuristics (the NLL internalization term), then much of MEL’s benefit might be attainable without policy-gradient RL—potentially reducing training complexity and compute.

MEL does not report an ablation that removes the GRPO term while keeping the meta-experience pipeline intact.

### Key Insight and Hypothesis

**Key insight**: MEL’s internalization term is already a likelihood objective that can be interpreted as a dense, process-level learning signal. If this signal is the primary driver of MEL’s improvements, then removing the GRPO policy-gradient term should not remove most of the benefit.

**Hypothesis**: Under a compute-matched training loop that still performs rollouts, meta-experience construction, and replay validation, an **NLL-only MEL variant** (GRPO weight set to 0) will recover a large fraction of MEL’s improvement over GRPO-only.

Why this might fail: GRPO updates may be necessary to (i) discover enough correct trajectories for contrastive pairing, (ii) improve exploration quality in a way NLL-only updates cannot, or (iii) align the policy with the verifier signal beyond what meta-experience text prediction provides.

---

## Proposed Approach

### Overview

We propose a focused ablation study with three training conditions that differ only in which gradient terms update the policy.

To avoid ambiguity about the term “bifurcation point”: in MEL, a bifurcation point is the first reasoning step where a correct and incorrect solution trajectory diverge under the same prompt.

- **A: GRPO-only**: Standard RLVR training with GRPO using binary verifiable rewards.
- **B: MEL-full**: GRPO + meta-experience construction + replay validation + NLL internalization (as in MEL).
- **C: NLL-only MEL (ours)**: Same MEL pipeline and outer loop as (B), but the GRPO objective is removed and the policy is updated only via the NLL internalization loss on validated meta-experiences.

All three conditions use **identical rollout generation and verification settings** (same group size, temperature, batch size, and verifier). Conditions **B/C** additionally run meta-experience construction and replay validation, since these are required to define the internalization loss. The only change that affects parameter updates is which loss terms are active.

### Method Details

#### Base rollout and verification (shared across A/B/C)
Follow MEL’s reported setup **[MEL Experiments](./references/sections/Experiments.md)**:
- For each training prompt \(x\), sample a group of \(G=8\) reasoning trajectories at temperature 1.0.
- Use a rule-based verifier (**Math-Verify**, a symbolic answer extraction + equivalence checking library) to label each trajectory as correct/incorrect by comparing extracted final answer with ground truth.
- Partition the group into \(Y^+\) and \(Y^-\) for each prompt.

#### Meta-experience construction + replay validation (used in B/C)
Implement MEL’s construction pipeline **[MEL Meta-Experience Construction](./references/sections/Meta-Experience%20Construction.md)**:
- Construct contrastive pairs \((y^+,y^-)\) for prompts with non-empty \(Y^+\) and \(Y^-\).
- Prompt the policy to generate:
  - bifurcation point \(s^*\),
  - critique \(C\),
  - abstract heuristic \(H\),
  forming a meta-experience \(M=(s^*,C,H)\).
- Apply MEL’s strict generalization constraint to avoid copying instance-specific answers into the heuristic **[MEL Strict Generalization Constraint](./references/sections/Strict%20Generalization%20Constraint.md)**.
- Replay-validate each \(M\) by injecting it as a hint and re-solving the same problem; keep \(M\) only if the replay solution passes the verifier **[MEL Eq. 6](./references/sections/Meta-Experience%20Construction.md)**.

#### Loss functions (what differs between conditions)
- **A (GRPO-only)**: optimize \(J_{\text{RLVR}}\) (GRPO) and ignore \(J_{\text{MEL}}\).
- **B (MEL-full)**: optimize \(J_{\text{RLVR}} + J_{\text{MEL}}\) as in MEL **[MEL Joint Training Objective](./references/sections/Joint%20Training%20Objective.md)**.
- **C (NLL-only)**: optimize only \(J_{\text{MEL}}\) (token-averaged NLL on validated \(M\) conditioned on \([I,x,y^+,y^-]\)), with GRPO weight exactly 0.

#### Diagnostics (non-core; used to interpret results)
To detect whether outcomes are driven by “lack of training signal” rather than the loss term itself, log per condition. If NLL-only (C) produces substantially fewer validated meta-experiences than MEL-full (B), this would support the interpretation that policy-gradient updates help sustain exploration and/or generate contrastive pairs, explaining a negative result mechanistically.
- fraction of prompts with non-empty \(Y^+\) and \(Y^-\) per step,
- number of replay-validated meta-experiences per step,
- average meta-experience token length.

### Key Innovations

- A **compute-matched, end-to-end ablation** that directly tests whether MEL’s gains require policy-gradient RL updates, rather than only testing prompt-conditioning or negative-sample handling.
- A **ratio-based, CI-backed decision rule** that turns an ambiguous mechanism question (“is RL necessary?”) into a falsifiable claim.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) RLVR post-training for reasoning models, (ii) understanding when online RL is necessary versus likelihood-based objectives, and (iii) extracting denser learning signals from verifier feedback without training a learned reward model.

RLVR methods commonly use GRPO/PPO-style policy gradients with verifiable outcome rewards, and multiple recent works aim to improve sample efficiency or credit assignment using process supervision, experience replay, trajectory editing, or auxiliary supervised losses. Separately, theory and controlled experiments in RLHF suggest that the advantage of online RL over offline objectives can depend on a generation-versus-verification asymmetry, motivating careful ablations that isolate which components actually matter.

### Related Papers

- **[Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models](./references/meta/meta_info.txt)**: Introduces MEL (contrastive meta-experience construction + replay validation + NLL internalization) on top of GRPO in RLVR.
- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)**: Introduces GRPO as a memory-efficient PPO-style optimizer for reasoning RLVR.
- **[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)**: Large-scale RL for reasoning; motivates why RLVR/RLHF can change reasoning behaviors.
- **[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)**: Open RL system (VERL) and RL techniques for stabilizing long-CoT GRPO-style training.
- **[All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning](https://arxiv.org/abs/2503.01067)**: Argues RL’s advantage can be explained by generation–verification gap; motivates likelihood-vs-RL ablations.
- **[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)**: Canonical offline objective for preference optimization; relevant as an example of non-RL fine-tuning.
- **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)**: PPO foundation underlying many RL post-training methods.
- **[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)**: RLHF reference pipeline motivating online post-training.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Establishes process reward models for math reasoning; highlights benefits of step-level feedback.
- **[Process Reward Models that Think](https://arxiv.org/abs/2502.11517)**: PRM work emphasizing structured step-level supervision (and associated risks/costs).
- **[Math-Shepherd](https://arxiv.org/abs/2312.08935)**: Uses automated process supervision for math reasoning; relevant to densifying supervision without human labels.
- **[VinePPO: Refining Credit Assignment in RL Training of LLMs](https://arxiv.org/abs/2410.01679)**: Improves credit assignment via Monte Carlo step-level structure.
- **[GRPO is Secretly a Process Reward Model](https://arxiv.org/abs/2509.21154)**: Shows GRPO can induce implicit process-level rewards via prefix overlap.
- **[It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977)**: Connects GRPO to contrastive/offline objectives; relevant to “RL vs likelihood” framing.
- **[ExGRPO: Learning to Reason from Experience](https://arxiv.org/abs/2510.02245)**: Adds experience replay to GRPO-style RLVR for sample efficiency.
- **[Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback](https://arxiv.org/abs/2506.03106)**: Integrates textual critiques into GRPO training, combining language-level signals with RL.
- **[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)**: Uses natural-language feedback loops; relevant to “learning from experience” paradigms.
- **[Self-critiquing models for assisting human evaluators](https://arxiv.org/abs/2211.03555)**: Early work on self-critique as a learning signal.
- **[REINFORCE++](https://arxiv.org/abs/2501.03262)**: An RL optimizer used in some RLVR pipelines; relevant as an alternative to GRPO.
- **[RLOO: Reinforcement Learning from Optimized Outcomes](https://arxiv.org/abs/2402.14740)**: RL post-training variant emphasizing outcome-based optimization.
- **[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)**: Technical report for the base model family used in MEL and in this proposal.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| RLVR with outcome rewards | Policy-gradient training using programmatic verifier on final answer | GRPO (DeepSeekMath), DAPO, DeepSeek-R1 | GSM8K/MATH/MATH500/AIME, rule-based verifiers | Sparse rewards, weak credit assignment |
| Process / dense supervision | Provide step-level learning signal (learned or derived) | Let’s Verify Step by Step, PRMs that Think, VinePPO | Math reasoning benchmarks | PRM training cost, reward hacking risk |
| Experience learning loops | Use self-generated trajectories/critiques to learn | MEL, Critique-GRPO, ExGRPO | Math reasoning + OOD tasks | Mechanism ambiguity; extra pipeline cost |
| RL vs likelihood perspective | Explain when online RL is necessary | All Roads Lead to Likelihood, GRPO↔DPO analyses | Controlled RLHF/RLVR settings | Depends on task horizon / verifier properties |

### Closest Prior Work

- **MEL** **[MEL](./references/meta/meta_info.txt)**: Combines GRPO updates with NLL internalization of replay-validated meta-experiences. Our proposal isolates whether GRPO is necessary by removing the GRPO term while keeping the pipeline and compute structure otherwise unchanged.
- **All Roads Lead to Likelihood** **[Swamy et al.](https://arxiv.org/abs/2503.01067)**: Provides a general explanation for why online RLHF may outperform offline objectives. Our work is a concrete, mechanism-focused test of a similar hypothesis in the specific MEL/RLVR setting.
- **GRPO↔DPO / GRPO-as-PRM analyses** **[GRPO↔DPO](https://arxiv.org/abs/2510.00977)**, **[GRPO-as-PRM](https://arxiv.org/abs/2509.21154)**: Show that GRPO can resemble offline/step-level objectives under certain structures. Our work tests a stronger statement: can we eliminate the GRPO term entirely in an end-to-end MEL loop.

**Novelty Kill Search Summary:** Searched for “MEL without GRPO”, “NLL-only meta-experience learning”, “Internalizing Meta-Experience ablation GRPO”, and related variants; no prior work or concurrent proposals performing this exact ablation were found as of 2026-02-28 (query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MEL | GRPO + replay-validated meta-experiences + NLL internalization | Does not isolate whether GRPO is necessary | Remove GRPO term (C) under matched outer loop | Decisively attributes MEL’s gains to RL vs NLL |
| All Roads Lead to Likelihood | Explains RL benefit via generation–verification gap | Not MEL-specific; not an RLVR pipeline ablation | Test in MEL’s concrete setting | Converts theory into a decision for practitioners |
| GRPO↔DPO / GRPO-as-PRM | Reframes GRPO as contrastive / implicit process supervision | Still assumes GRPO objective | Drop GRPO entirely | Tests whether “RL is secretly likelihood” extends to MEL |

---

## Experiments

### Experimental Setup

**Core experiment (3 conditions):**
- **A: GRPO-only**
- **B: MEL-full**
- **C: NLL-only MEL (ours)**

**Baseline Ladder (REQUIRED):**
- Prompting baseline: base model Pass@1 on each benchmark (no training).
- Inference-time scaling baseline: base model Pass@8 (best-of-8) on each benchmark.
- Training baselines: GRPO-only (A) and MEL-full (B).
- Closest method: MEL (B); our contribution is the NLL-only ablation (C).

**Compute matching principle:**
- Match **outer-loop training steps** (same number of rollout iterations) and identical rollout generation settings across A/B/C.
- Because conditions B/C include meta-experience generation and replay validation (extra generation), we will track: total rollout tokens, total replay tokens, and total NLL-train tokens per condition, and report an approximate FLOPs estimate.
- Interpret results in light of any residual compute differences (if C matches B while using fewer FLOPs, that strengthens the “NLL-only is sufficient” conclusion).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3-4B-Base | 4B | https://huggingface.co/Qwen/Qwen3-4B-Base | Matches MEL’s smallest scale; lower cost than 8B/14B |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| DAPO-Math-17k | RLVR training prompts with ground-truth integer answers | ~17k prompts | https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k | (check HF card) |

**Other Resources (if applicable):**
- VERL RLHF framework (used by MEL): https://github.com/volcengine/verl
- Math-Verify (answer extraction + verification): https://github.com/huggingface/Math-Verify

**Resource Estimate**:
- **Compute budget**: Target **≤ 720 A100 GPU-hours** total.
  - Plan: 3 conditions × 3 seeds = 9 runs.
  - Each run: 8×A100 for ≤10 hours (initial budget). Use early-stop if MEL replication fails.
- **GPU memory**: Qwen3-4B fits on 80GB A100 with bf16; training via VERL with vLLM inference may use 4–8 GPUs per run.
- **API usage**: None (all local).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MATH-500 | 500 competition-math problems from the MATH benchmark (standard evaluation subset) | Pass@1 (T=0), Pass@8 (T=0.6) | test | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | Math-Verify-based harness |
| AIME24 | 30 AIME 2024 problems | Pass@1, Pass@8 | test | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 | Math-Verify-based harness |
| AIME25 | 30 AIME 2025 problems | Pass@1, Pass@8 | test | https://huggingface.co/datasets/MathArena/aime_2025 | Math-Verify-based harness |
| AMC23 | AMC 2023 problems | Pass@1, Pass@8 | test | https://huggingface.co/datasets/math-ai/amc23 | Math-Verify-based harness |
| OlympiadBench | Olympiad-level math benchmark (subset as in MEL) | Pass@1, Pass@8 | test | https://huggingface.co/datasets/math-ai/olympiadbench | Math-Verify-based harness |

**Evaluation Scripts:**
- Use Math-Verify extraction + verification. Follow MEL protocol: Pass@1 at temperature 0; Pass@8 and Avg@8 at temperature 0.6 where applicable.

### Main Results

#### Published anchor numbers (from MEL paper)
From MEL’s `Joint Training Objective` section (Qwen3-4B-Base; Pass@1/Avg@8/Pass@8): **[MEL Table](./references/sections/Joint%20Training%20Objective.md)**.

| Benchmark | Base | GRPO | MEL |
|---|---:|---:|---:|
| AIME24 | 13.33 / 9.90 / 30.00 | 13.33 / 18.33 / 30.00 | 20.00 / 20.83 / 33.00 |
| AIME25 | 10.00 / 6.56 / 23.33 | 6.67 / 17.50 / 30.00 | 16.67 / 18.33 / 33.00 |
| AMC23 | 45.00 / 42.73 / 72.50 | 57.50 / 58.13 / 85.00 | 60.00 / 60.31 / 87.50 |
| MATH500 | 74.20 / 65.74 / 89.60 | 81.80 / 82.20 / 93.00 | 82.20 / 82.30 / 93.80 |
| OlympiadBench | 39.17 / 35.37 / 60.38 | 48.51 / 48.46 / 67.21 | 48.51 / 49.48 / 69.73 |
| **Average** | **36.34 / 32.06 / 55.16** | **41.56 / 44.92 / 61.04** | **45.48 / 46.25 / 63.41** |

#### Results Table (to be verified)

| Method | Base Model | Benchmark | Pass@1 (mean±std) | Pass@8 (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Base (no training) | Qwen3-4B-Base | all above | TBD | TBD | - | prompting baseline |
| GRPO-only (A) | Qwen3-4B-Base | all above | TBD | TBD | - | training baseline |
| MEL-full (B) | Qwen3-4B-Base | all above | TBD | TBD | - | closest method |
| **NLL-only MEL (C)** | Qwen3-4B-Base | all above | TBD | TBD | - | removes GRPO term |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Yield logging (diagnostic) | Report #validated meta-experiences per step | Helps interpret failures without changing decision rule |

### Experimental Rigor

- **Seeds**: `seeds=[1,2,3]` for A/B/C.
- **Primary confounders**:
  1) **Compute mismatch** (B has additional backward pass for GRPO): match outer steps; report token/FLOPs; interpret accordingly.
  2) **Meta-experience yield differences**: log yields; treat as mechanism explanation, not as post-hoc excuse.
  3) **Replication failure**: if B does not outperform A, the study cannot attribute MEL’s gains and should stop.
- **Sanity checks**:
  - Base model Pass@1 and Pass@8 reproduce reasonable ranges for the chosen model family.
  - B matches (or is directionally consistent with) MEL’s published improvements over A on the macro-average.
- **Data leakage**:
  - Competition-math benchmarks may have partial pretraining contamination. Mitigation: (i) focus on *relative* differences between A/B/C under identical base model and protocol, and (ii) report per-benchmark deltas rather than only absolute accuracy.

### Analysis (Optional)

- Correlate per-benchmark improvement with validated meta-experience yield to test whether NLL-only failures are explained by “no usable meta-experience signal”.

---

## Success Criteria

**Hypothesis**: NLL-only internalization (C) recovers most of MEL’s gain over GRPO-only (A) under a matched outer loop.

**Decision Rule**:

Let \(S(\cdot)\) be the macro-average Pass@1 across the five benchmarks above, averaged across 3 seeds.
Define \(\Delta_B = S(B) - S(A)\), \(\Delta_C = S(C) - S(A)\), and \(r = \Delta_C/\Delta_B\).

Compute an 80% bootstrap confidence interval for \(r\) using 10k bootstrap replicates by resampling **problems** (with replacement) within each benchmark, first averaging accuracies across seeds.

- **Proceed (NLL-only sufficient)**: \(\Delta_B>0\) and the 80% bootstrap CI lower bound of \(r\) is \(\ge 0.7\).
- **Proceed (GRPO necessary)**: \(\Delta_B>0\) and the 80% bootstrap CI upper bound of \(r\) is \(\le 0.3\).
- **Pivot (ambiguous)**: \(\Delta_B>0\) and CI overlaps \([0.3,0.7]\). Optional pivot: add a frozen-rollout variant to test whether online rollouts are necessary.
- **Refute**: \(\Delta_B \le 0\) (failed MEL replication) or \(S(C) < S(A)\) by a margin outside the std range.

---

## Impact Statement

If NLL-only MEL recovers most of MEL’s gains, practitioners could reconsider whether policy-gradient RL updates are necessary for MEL-style experience internalization, potentially simplifying RLVR training pipelines and reducing post-training compute. If NLL-only fails, the result clarifies that MEL’s benefit is genuinely tied to policy-gradient RL dynamics rather than being explainable as auxiliary supervised fine-tuning, guiding future work toward better RL credit assignment rather than more heuristic distillation.

---

## References

- [Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models](./references/meta/meta_info.txt) - Huang et al., 2026
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - Guo et al., 2025
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) - Yu et al., 2025
- [All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning](https://arxiv.org/abs/2503.01067) - Swamy et al., 2025
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [Process Reward Models that Think](https://arxiv.org/abs/2502.11517) - Khalifa et al., 2025
- [Math-Shepherd](https://arxiv.org/abs/2312.08935) - Wang et al., 2023
- [VinePPO: Refining Credit Assignment in RL Training of LLMs](https://arxiv.org/abs/2410.01679) - Kazemnejad et al., 2024/2025
- [GRPO is Secretly a Process Reward Model](https://arxiv.org/abs/2509.21154) - Sullivan, 2025
- [It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977) - Wu et al., 2025
- [ExGRPO: Learning to Reason from Experience](https://arxiv.org/abs/2510.02245) - Zhan et al., 2025
- [Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback](https://arxiv.org/abs/2506.03106) - Zhang et al., 2025
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - Shinn et al., 2023
- [Self-critiquing models for assisting human evaluators](https://arxiv.org/abs/2211.03555) - Saunders et al., 2022
- [REINFORCE++](https://arxiv.org/abs/2501.03262) - 2025
- [RLOO: Reinforcement Learning from Optimized Outcomes](https://arxiv.org/abs/2402.14740) - 2024
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - Qwen Team, 2025
