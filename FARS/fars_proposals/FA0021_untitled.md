# untitled

# Does iGRPO Need a Good Draft? Best-vs-Worst Self-Conditioning Ablation for RLVR Math

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models can be improved on mathematical reasoning by training them with **verifiable rewards**. In this setting, a model generates a solution, a programmatic verifier extracts the final answer, and the model receives a binary reward for correctness. This approach is often called **reinforcement learning from verifiable rewards (RLVR)** and is a common alternative to human-preference-based reinforcement learning.

A widely used RLVR optimizer is **Group Relative Policy Optimization (GRPO)**, introduced in **[DeepSeekMath](./references/DeepSeekMath-Pushing-the-Limits-of-Mathematical-Reasoning-in-Open-Language-Models/meta/meta_info.txt)**. GRPO is a PPO-like method that avoids training a value function by sampling multiple rollouts per prompt and normalizing rewards within each group.

Recent work proposes **iGRPO**, a two-stage extension of GRPO that adds “draft then refine” self-conditioning during training: it samples several drafts, selects the best draft under the reward, appends it to the prompt, and then applies the GRPO update on refinements conditioned on that draft (**[iGRPO](./references/iGRPO-Self-Feedback-Driven-LLM-Reasoning/meta/meta_info.txt)**). iGRPO reports consistent gains on hard math benchmarks under matched rollout budgets.

### The Problem

iGRPO’s core design choice is **reward-based draft selection**: Stage 1 chooses the highest-reward draft to condition Stage 2. This selection step has two practical consequences:

1. **Extra compute and plumbing**: implementing selection requires evaluating multiple drafts per prompt (and often computing additional “format” rewards), plus maintaining a two-stage rollout pipeline.
2. **Mechanism ambiguity**: it is unclear whether iGRPO’s gains come from conditioning on a *high-quality* draft (selection matters), or whether conditioning on *any* draft already provides most of the benefit (selection may be unnecessary).

The iGRPO paper does not include a direct control that conditions Stage 2 on an intentionally **low-reward draft**. Without this, practitioners cannot tell whether iGRPO is (i) a fragile method that only works when the conditioning draft is good, or (ii) a robust “refinement interface” where draft quality is not essential.

### Key Insight and Hypothesis

**Key insight.** If iGRPO’s gains mainly come from conditioning Stage 2 on a *correct / high-reward* draft (a strong scaffold), then conditioning on a *wrong* but well-formatted draft should remove most of the benefit. Conversely, if the benefit mainly comes from exposing the model to an in-context “attempt” (regardless of correctness), then even a wrong draft should still improve over GRPO.

**Hypothesis.** Under a compute-matched RLVR setup, **iGRPO-best** (paper method) will outperform GRPO, but **iGRPO-worst-of-formatted** (conditioning on a low-reward draft that still follows the required output format) will recover only a small fraction of iGRPO-best’s gain on MATH500 Pass@1 (single-sample accuracy).

This could fail if (a) the main benefit of iGRPO is “train on draft-conditioned prompts” rather than “train on good drafts”, or (b) the model learns to ignore low-quality drafts due to the prompt instruction “do not repeat the draft verbatim”.

---

## Proposed Approach

### Overview

We propose a controlled 3-condition ablation to isolate the role of **draft quality** in iGRPO (a form of *two-stage self-conditioning* under RLVR):

- **A: GRPO** (baseline, one-stage)
- **B: iGRPO-best** (paper method, Stage-1 draft chosen by max reward)
- **C: iGRPO-worst-of-formatted** (new control, Stage-1 draft chosen to be low reward but still format-valid)

All conditions use the same base model, the same total number of sampled completions per prompt, the same reward functions, and the same training schedule. The only difference is how the Stage-1 draft (the conditioning context for Stage 2) is chosen.

### Method Details

#### Background: iGRPO two-stage sampling
Following **[iGRPO](./references/iGRPO-Self-Feedback-Driven-LLM-Reasoning/meta/meta_info.txt)**, each training iteration uses a frozen behavior policy \(\pi_{\theta_{old}}\):

1. **Stage 1** (drafts): sample \(N\) drafts \(d_1,\dots,d_N\sim \pi_{\theta_{old}}(\cdot\mid x)\).
2. **Select** one draft \(d\) according to a selection rule.
3. **Stage 2** (refinements): construct an augmented prompt \(x' = \text{Concat}(x, d)\), sample \(G\) refinements \(y_1,\dots,y_G\sim \pi_{\theta_{old}}(\cdot\mid x')\), compute group-normalized advantages, and apply a PPO/GRPO-style clipped update **only on Stage-2 tokens**.

We use the iGRPO paper’s default compute matching: total rollouts per prompt \(N+G = 8\), with \(N=4\) drafts and \(G=4\) refinements.

#### Rewards
We follow iGRPO’s training recipe (Table S.2 in `Prompt.md` of the iGRPO paper artifacts):

- **Accuracy reward** \(r_{acc}\in\{0,1\}\): extract the final answer from the completion and compare to the ground truth.
- **Format reward** \(r_{fmt}\in\{0,1\}\): 1 if the completion contains a parseable `<answer>...</answer>` field (and optionally `<think>...</think>`), else 0.
- Scalar reward: \(R = r_{acc} + r_{fmt}\) (weights 1.0, 1.0).

#### Draft selection rules
- **iGRPO-best**: select \(d = \arg\max_i R(d_i)\).
- **iGRPO-worst-of-formatted (ours)**:
  - Let \(\mathcal{I}=\{i: r_{fmt}(d_i)=1\}\).
  - If \(\mathcal{I}\neq\emptyset\), choose \(d=\arg\min_{i\in\mathcal{I}} R(d_i)\) (prefer wrong-but-formatted drafts when they exist).
  - Else (rare), fall back to \(d=\arg\min_i R(d_i)\).

This control avoids a degenerate “worst” draft that is merely unparseable noise.

#### Prompting
We use the iGRPO prompt instruction from `Prompt.md` (same across conditions). For Stage 2, we append the selected draft under a fixed delimiter (e.g., `\n\n[Prior draft]\n{draft}\n\n[Task]\n`), keeping the delimiter constant across conditions.

### Key Innovations

- A **best-vs-worst-of-formatted** self-conditioning control that directly tests whether iGRPO’s improvement depends on conditioning on a high-quality draft.
- A **pre-registered decision rule** based on a recovery-ratio statistic, so the result is interpretable even if absolute performance differs from the paper due to implementation details.

---

## Related Work

### Field Overview

RLVR for math reasoning typically combines (i) a strong pretrained or distilled base model, (ii) a verifiable outcome reward (final answer correctness), and (iii) a stable policy-optimization algorithm such as GRPO or its variants. A persistent challenge is to improve long-horizon reasoning without collapsing exploration or overfitting to narrow solution modes. Recent methods explore alternative advantage normalization, ratio stabilization, critique- or verifier-augmented training, and iterative/self-conditioning training loops.

iGRPO is part of a broader family of **self-improvement** methods, where a model learns from its own generated artifacts (drafts, critiques, verifications, or selected samples). Prior work has shown that self-refinement can help at inference time (e.g., Self-Refine, Reflexion), while RLVR research explores baking such behaviors into training.

### Related Papers

- **[iGRPO: Self-Feedback-Driven LLM Reasoning](./references/iGRPO-Self-Feedback-Driven-LLM-Reasoning/meta/meta_info.txt)**: Introduces two-stage draft→refine training under matched rollout budgets; this proposal ablates whether draft *quality* matters.
- **[DeepSeekMath](./references/DeepSeekMath-Pushing-the-Limits-of-Mathematical-Reasoning-in-Open-Language-Models/meta/meta_info.txt)**: Introduces GRPO for math reasoning; our baseline condition.
- **[Critique-GRPO](./references/Critique-GRPO-Advancing-LLM-Reasoning-with-Natural-Language-and-Numerical-Feedback/meta/meta_info.txt)**: Uses critique-conditioned refinements during online RL; another “refine after feedback” approach with different feedback source.
- **[Incentivizing LLMs to Self-Verify Their Answers](./references/Incentivizing-LLMs-to-Self-Verify-Their-Answers/meta/meta_info.txt)**: Trains generation+verification jointly; related self-improvement via verification signals.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Demonstrates large-scale RL for reasoning; sets the modern RLVR context.
- **[DAPO](https://arxiv.org/abs/2503.14476)**: A GRPO-family RLVR system emphasizing stability/engineering for reasoning training.
- **[GSPO](https://arxiv.org/abs/2507.18071)**: Group Sequence Policy Optimization uses length-normalized sequence-level importance ratios to improve stability vs token-level GRPO.
- **[GMPO](https://arxiv.org/abs/2507.20673)**: Geometric-Mean Policy Optimization replaces GRPO’s arithmetic mean with a geometric mean to reduce outlier sensitivity and stabilize updates.
- **[GEPO](https://arxiv.org/abs/2508.17850)**: Group Expectation Policy Optimization targets stability under high-latency/off-policy heterogeneity; highlights importance-weight variance as a core failure mode.
- **[SAPO](https://arxiv.org/abs/2511.20347)**: Soft Adaptive Policy Optimization replaces hard clipping with smooth gates for more stable GRPO/GSPO-style training.
- **[KPO](https://arxiv.org/abs/2602.10609)**: Kalman-based token-level ratio smoothing for GRPO-style RLVR stability; orthogonal to iGRPO’s draft-conditioning.
- **[Self-Refine](https://arxiv.org/abs/2303.17651)**: Inference-time iterative refinement with self-feedback; conceptually similar but not an RL training method.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Agentic self-reflection to improve performance over trials; related self-feedback paradigm.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Shows process verification improves reasoning; motivates RLVR and verifier-centric training.
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)**: Test-time sampling + majority vote for reasoning; a standard baseline for scaling inference compute.
- **[Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)**: Uses models as reward providers; contrasts with iGRPO’s use of external verifiable reward.
- **[ReST: Reinforced Self-Training](https://arxiv.org/abs/2308.08998)**: Iteratively generates and filters samples for self-training; another self-improvement loop.
- **[Latent Principle Discovery for Language Model Self-Improvement (STaPLe)](https://arxiv.org/abs/2505.16927)**: Discovers principles for self-improvement; related “learn from own artifacts” framing.
- **[Learning to Refine: Self-Refinement of Parallel Reasoning in LLMs](https://arxiv.org/abs/2509.00084)**: Trains a unified model to refine over multiple candidate solutions; relevant because it suggests even imperfect candidates can be useful scaffolds.
- **[StepCo](https://arxiv.org/abs/2410.12934)**: Verify-then-revise prompting for math reasoning; an inference-time counterpart to “draft then refine” loops.
- **[FastGRPO](https://arxiv.org/abs/2509.21792)**: Accelerates GRPO training via concurrency-aware speculative decoding; relevant for the compute cost of two-stage pipelines.
- **[TIC-GRPO](https://arxiv.org/abs/2508.02833)**: Trajectory-corrected GRPO variant with theoretical analysis; representative of 2025–2026 GRPO-family stabilization work.
- **[Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening](https://arxiv.org/abs/2506.02355)**: Analyzes GRPO’s tendency to sharpen distributions and proposes fixes; related to best-of selection dynamics.
- **[MC-GRPO](https://arxiv.org/abs/2601.22582)**: Improves GRPO advantage normalization for small rollout groups.
- **[PPO](https://arxiv.org/abs/1707.06347)**: Standard policy-optimization baseline that GRPO variants derive from.
- **[TRPO](https://arxiv.org/abs/1502.05477)**: Trust-region policy optimization; conceptual ancestor to PPO-style stabilization.
- **[DPO](https://arxiv.org/abs/2305.18290)**: Preference optimization alternative to RL; included for context on post-training objectives.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Group-based PPO / GRPO family | Value-free PPO-style RLVR with group-normalized rewards | DeepSeekMath (GRPO), DAPO, GSPO, GMPO, SAPO, GEPO | MATH500, AIME, AMC | Instability, mode collapse, ratio noise |
| Draft/refine self-conditioning | Condition training updates on model-generated drafts | iGRPO, Critique-GRPO | AIME, MATH500 | Mechanism ambiguity; may be brittle to draft quality |
| Verifier/self-verification | Train models to verify or critique solutions | Incentivizing Self-Verify, generative verifiers | MATH500, AIME | Requires careful calibration; risk of self-confirmation |
| Inference-time self-improvement | Improve via iterative refinement at inference time | Self-Refine, Reflexion, self-consistency | GSM8K, AIME | Extra inference cost; not directly trained end-to-end |

### Closest Prior Work

- **iGRPO**: Two-stage draft selection (best-of-N under reward) then refinement updates. Does not test whether conditioning on a low-quality draft removes the gain.
- **Critique-GRPO**: Also performs refinement, but the conditioning signal is a natural-language critique from an external model; does not isolate “quality of conditioning context” in a best-vs-worst sense.
- **Incentivizing LLMs to Self-Verify Their Answers**: Uses verification rather than drafting to improve robustness and test-time scaling; not a draft-conditioning method.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| iGRPO | Best-draft-conditioned RL updates | No ablation for draft-quality necessity | Add worst-of-formatted draft control | Directly identifies whether draft correctness/selection is the mechanism |
| Critique-GRPO | Critique-conditioned refinement + RL | Depends on critique model; different mechanism | Keep verifiable rewards; manipulate only draft quality | Cleaner isolation of conditioning-quality effect under RLVR |
| Self-Verify | Jointly train solving + verification | Different interface (verification tokens) | Keep draft/refine interface; vary draft quality | Diagnoses whether draft interface needs good exemplars or any exemplar |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 7B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | Matches iGRPO’s controlled study setting; fits in A100-80GB |

**Training Data (paper setting + verification setting):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| MATH (train split) | RLVR training prompts + ground-truth answers | 7,500 | https://huggingface.co/datasets/hendrycks/competition_math | MIT |
| AceReason-Math (paper uses) | Additional RLVR math prompts | 9,400 (as reported in iGRPO) | https://huggingface.co/datasets/nvidia/AceReason-Math | CC BY 4.0 |

*Note:* If reproducing the paper’s exact mixed-data setup is too costly, the minimum decisive experiment can run on **MATH-only** (7.5k) while keeping the rollout budget and hyperparameters identical across A/B/C.

**Other Resources (if applicable):**
- Evaluation harness: NeMo-Skills (https://github.com/NVIDIA-NeMo/Skills) or a minimal answer-extraction evaluator.

**Resource Estimate**:
- **Compute budget**: target ≤ **650 GPU-hours** total.
  - Evidence: iGRPO reports ~94.1 GPU-hours for one 7B training run under their setup (Appendix D.2).
  - Plan: 3 conditions × 2 random seeds × 100 GPU-hours/run (rounded up) ≈ 600 GPU-hours + evaluation overhead.
- **GPU memory**: iGRPO reports ~55 GB peak on A100-80GB for 7B training; expect ≤80 GB per GPU.
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MATH500 | 500-problem subset of MATH for competition-style reasoning | Pass@1 (single-sample accuracy; higher is better) | test | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | NeMo-Skills or custom answer extraction |
| AIME24 | 30 AIME 2024 problems (integer answers 0–999) | Pass@1 (single-sample accuracy) | test | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 | NeMo-Skills or custom |
| AIME25 | 30 AIME 2025 problems (integer answers 0–999) | Pass@1 (single-sample accuracy) | test | https://huggingface.co/datasets/math-ai/aime25 | NeMo-Skills or custom |
| AMC23 | 2023 AMC 12 problems (multiple-choice answers A–E) | Pass@1 (single-sample accuracy) | test | https://huggingface.co/datasets/math-ai/amc23 | NeMo-Skills or custom |
| GSM8K | Grade-school math word problems | Pass@1 (single-sample accuracy) | test | https://huggingface.co/datasets/openai/gsm8k | NeMo-Skills or custom |
| Minerva | STEM problem set (Minerva Math) | Pass@1 (single-sample accuracy) | test | https://huggingface.co/datasets/math-ai/minervamath | NeMo-Skills or custom |

### Main Results

#### Results Table

Published baselines (Pass@1, %) from **iGRPO Table 1** (extracted from the arXiv HTML because the PDF scrape stubbed the table) for DeepSeek-R1-Distill-Qwen-7B are included for reference. Here, **Avg** is the macro-average over {AIME25, AIME24, MATH500, AMC23, GSM8K, Minerva}. Verification should re-run A and B to confirm reproducibility before interpreting C.

| Method | AIME25 | AIME24 | MATH500 | AMC | GSM8K | Minerva | Avg | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepSeek-R1-Distill-Qwen-7B (base) | 38.60 | 54.40 | 92.80 | 90.00 | 92.00 | 39.10 | 61.93 | iGRPO Table 1 |
| + GRPO (A) | 38.90 | 55.00 | 93.25 | 90.00 | 92.12 | 40.44 | 68.29 | iGRPO Table 1 |
| + Self-Verification | 39.45 | 55.80 | 93.50 | 92.50 | 92.20 | 41.00 | 69.08 | iGRPO Table 1 |
| + Critique-GRPO | 39.60 | 55.65 | 93.45 | 92.80 | 92.25 | 41.10 | 69.14 | iGRPO Table 1 |
| + iGRPO-best (B) | 40.16 | 56.30 | 93.80 | 95.00 | 92.42 | 41.54 | 69.87 | iGRPO Table 1 |
| + **iGRPO-worst-of-formatted (C, ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | This work |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| iGRPO-best (B) | Stage-1 selects highest-reward draft | Strong performance (replicates paper) |
| iGRPO-worst-of-formatted (C) | Stage-1 selects low-reward but format-valid draft | If selection/correctness matters: C ≈ A; if not: C ≈ B |

### Analysis (Optional)

All analyses are computed from stored rollouts (no extra training conditions):

- **Selected-draft length distribution** (best vs worst): detect “length-as-confound”.
- **Stage-2 reward variance** within each prompt group: detect gradient-signal differences.
- **Draft/refinement copy rate**: n-gram overlap or edit distance; stratify by whether the draft was correct.
- **Fraction of prompts where best and worst differ**: if rare, the ablation is not informative.

---

## Success Criteria

**Criterion 1: Draft-quality necessity (primary)**
- Hypothesis: iGRPO-worst-of-formatted does not recover most of iGRPO-best’s gain over GRPO.
- Validation (primary benchmark): Using **MATH500 Pass@1**, require Score(iGRPO-best) − Score(GRPO) ≥ 3 points; then compute \(r_{worst}\). Success if \(r_{worst} \le 0.25\).
- Fallback if MATH500 is inconclusive: If the MATH500 gap is <3 points, compute the same recovery ratio using **AIME24/AIME25 macro-average** as the benchmark (noting higher variance), and report the MATH500 result as inconclusive.

**Criterion 2: Draft-quality non-necessity (alternative outcome, still publishable)**
- Hypothesis: conditioning on a draft is sufficient even if the draft is low reward.
- Validation: If \(r_{worst} \ge 0.75\) on the primary benchmark (or the AIME fallback when invoked), conclude draft correctness is not essential, implying iGRPO can likely be simplified to avoid reward-based selection.

---

## Impact Statement

If draft quality is essential, the result supports iGRPO’s core mechanism claim and indicates that practitioners should invest in reliable draft selection (and expect brittleness when drafts are poor). If draft quality is not essential, the result suggests a simpler and potentially cheaper variant of iGRPO that does not require reward-based best-draft selection, lowering implementation complexity for RLVR pipelines.

---

## References

- [iGRPO: Self-Feedback-Driven LLM Reasoning](./references/iGRPO-Self-Feedback-Driven-LLM-Reasoning/meta/meta_info.txt) - Hatamizadeh et al., 2026
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](./references/DeepSeekMath-Pushing-the-Limits-of-Mathematical-Reasoning-in-Open-Language-Models/meta/meta_info.txt) - Shao et al., 2024
- [Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback](./references/Critique-GRPO-Advancing-LLM-Reasoning-with-Natural-Language-and-Numerical-Feedback/meta/meta_info.txt) - Zhang et al., 2025
- [Incentivizing LLMs to Self-Verify Their Answers](./references/Incentivizing-LLMs-to-Self-Verify-Their-Answers/meta/meta_info.txt) - Zhang et al., 2025
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - Guo et al., 2025
- [DAPO](https://arxiv.org/abs/2503.14476) - Yu et al., 2025
- [KPO](https://arxiv.org/abs/2602.10609) - He et al., 2026
- [Self-Refine](https://arxiv.org/abs/2303.17651) - Madaan et al., 2023
- [Reflexion](https://arxiv.org/abs/2303.11366) - Shinn et al., 2023
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [Self-Consistency](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) - Yuan et al., 2024
- [ReST: Reinforced Self-Training](https://arxiv.org/abs/2308.08998) - Gulcehre et al., 2023
- [Latent Principle Discovery for Language Model Self-Improvement (STaPLe)](https://arxiv.org/abs/2505.16927) - 2025
- [Learning to Refine: Self-Refinement of Parallel Reasoning in LLMs](https://arxiv.org/abs/2509.00084) - Wang et al., 2025
- [StepCo](https://arxiv.org/abs/2410.12934) - Tan et al., 2024
- [FastGRPO](https://arxiv.org/abs/2509.21792) - Zhang et al., 2025
- [TIC-GRPO: A Trajectory-Corrected Approach](https://arxiv.org/abs/2508.02833) - Pang & Jin, 2025
- [Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening](https://arxiv.org/abs/2506.02355) - He et al., 2025
- [PPO](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [TRPO](https://arxiv.org/abs/1502.05477) - Schulman et al., 2015
- [DPO](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
