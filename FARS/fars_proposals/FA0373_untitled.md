# untitled

# Answer-Conditioned On-Policy Distillation for Verifiable Math: Does OPSD Need Full Reference Solutions?

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated evaluation (symbolic answer extraction + equivalence checking)
  - Open-weight base models and public datasets
  - Must fit within **≤768 A100 GPU-hours**

## Introduction

### Context and Motivation

Post-training has become a primary way to turn pretrained language models into strong verifiable reasoners for domains like mathematics and unit-tested code generation. Two widely used families of post-training methods are:

1. **Supervised distillation / supervised fine-tuning (SFT)** on teacher-produced chain-of-thought solutions (dense token-level supervision, but off-policy).
2. **Reinforcement learning with verifiable rewards (RLVR)**, where the model is trained from binary feedback from an automated checker (e.g., whether the final numeric answer matches) using algorithms such as **Group Relative Policy Optimization (GRPO)**.

A recent line of work shows that **on-policy distillation** can combine the main strengths of both: it trains on the student’s *own* trajectories (reducing distribution mismatch / exposure bias) while still using a dense learning signal from a teacher distribution (reducing the sparse credit-assignment problem of RLVR). In particular, **On-Policy Self-Distillation (OPSD)** (“Self-Distilled Reasoner”, arXiv:2601.18734) instantiates teacher and student from the *same* model by conditioning the teacher on privileged information (a reference solution) while the student sees only the problem.

However, OPSD as presented assumes that training data includes **full reference solutions** (often long chain-of-thought). In many verifiable math datasets and many practical settings, the cheapest reliable supervision is **only the final answer label**, not a full solution trace.

### The Problem

OPSD’s core promise is *token-efficient* reasoning post-training: it can match or beat GRPO with far fewer rollouts by replacing sparse rewards with dense token-level guidance. But it is unclear what privileged information is actually necessary for the teacher to provide useful guidance.

- **[Self-Distilled Reasoner / OPSD](https://arxiv.org/abs/2601.18734)** conditions the teacher on a full reference solution (often chain-of-thought), which may be expensive to obtain or curate.
- **RLVR/GRPO** does not require solutions, but needs multiple rollouts per prompt and provides the same scalar reward to all tokens in a trajectory (poor credit assignment).
- **Efficient SFT distillation** work suggests that early reasoning tokens carry most of the learning signal (e.g., “Distilling the Essence”, arXiv:2512.21002), but those results are for **off-policy SFT**, not on-policy teacher-guided training.

For practitioners choosing a post-training recipe, the decision-relevant question is:

> If we only have (problem, final answer) supervision (no solution traces), can we still get most of OPSD’s benefit over compute-matched GRPO?

### Key Insight and Hypothesis

**Key insight.** In OPSD, the teacher’s main role is to provide a sharper next-token distribution at the student’s “branch points” (early reasoning decisions) than a scalar verifiable reward can. A teacher conditioned on the **full reference solution** can produce a relatively concentrated distribution aligned with that solution. In contrast, a teacher conditioned on **the final answer only** must effectively “reverse engineer” a reasoning path to that answer; on hard problems this may yield a diffuse, high-entropy teacher distribution over early reasoning tokens, weakening the token-level credit assignment.

**Hypothesis.** On verifiable math (MATH-500), answer-conditioned OPSD will recover **less than 80%** of full-solution OPSD’s improvement over compute-matched GRPO on a pre-registered hard subset, indicating that full reference solutions contain useful information beyond the final answer label for token-level distillation.

**Why we could be wrong.** If rationalization given the final answer is sufficiently easy for the model, then answer-only conditioning may still produce a sharp teacher distribution and match full-solution OPSD. In that case, full solution traces are unnecessary for on-policy distillation, and answer-only labels suffice.

---

## Proposed Approach

### Overview

We propose **Answer-Conditioned OPSD (AC-OPSD)**: a minimal variant of OPSD where the teacher policy is conditioned only on the ground-truth final answer, not the full reference solution.

We compare three training conditions under a matched per-update compute proxy (matched generated tokens per prompt, counting OPSD’s extra teacher logprob pass):

- **A: Compute-matched GRPO (RLVR baseline)**
- **B: OPSD-full (existing method; teacher sees full reference solution)**
- **C: AC-OPSD (proposed; teacher sees final answer only)**

### Method Details

#### Common setup

- **Base model**: `Qwen/Qwen3-4B-Instruct` (open weights)
- **Training prompts**: math problems from OpenThoughts math subset (the same data source used by OPSD), with available reference solutions and final answers
- **Verifier**: Math-Verify (symbolic answer normalization + equivalence checking) for outcome rewards and evaluation

We use a common system prompt requiring a boxed final answer, e.g., “Please reason step by step and put your final answer within \boxed{...}.”

#### A) Compute-matched GRPO baseline

We train with GRPO using a binary outcome reward:
- Reward = 1 if extracted answer is equivalent to ground truth, else 0.
- Group size **G = 8** rollouts per prompt.

**Compute matching to OPSD.** OPSD requires an additional forward pass for teacher log-probabilities. We match a simple compute proxy: total forward tokens per prompt per update.
- OPSD uses 1 rollout of length `L_opd = 2048` tokens and a teacher logprob pass on the same tokens → ~`2 × 2048 = 4096` forward tokens.
- For GRPO, we set max generation length `L_grpo = 512` so `G × L_grpo = 8 × 512 = 4096` forward tokens per prompt.

This keeps the dominant cost (generation + forward passes) approximately matched per update.

#### B) OPSD-full (teacher conditioned on reference solution)

We follow OPSD’s formulation (arXiv:2601.18734):
- **Student policy** samples one on-policy solution `y ~ p_S(·|x)` with `max_new_tokens=2048`.
- **Teacher policy** is the same model parameters, but conditioned on privileged information `y*` (the reference solution) in addition to the problem: `p_T(·|x, y*)`.
- Train by minimizing a per-token divergence between teacher and student next-token distributions along the student’s sampled trajectory, backpropagating only through the student.

**Prompting format (teacher).** For each training example, teacher context is:
- Problem statement `x`
- Reference solution `y*` (as provided by dataset)
- Instruction: “After understanding the reference solution, solve the problem again in your own words.”

#### C) AC-OPSD (teacher conditioned on final answer only)

This is identical to B except the teacher is conditioned on the **final answer** `a*` only (no reference solution):
- Teacher context is:
  - Problem statement `x`
  - Ground-truth final answer `a*`
  - Instruction: “The correct final answer is \boxed{a*}. Please derive a solution that reaches this answer; do not simply restate the answer.”

This isolates whether full solution traces provide additional information beyond the answer label for producing useful token-level teacher guidance.

#### Distillation loss implementation (memory-bounded)

Full-vocabulary per-token divergences can be memory intensive. Following SDPO’s practical recipe (arXiv:2601.20802), we use **top-K KL distillation** with `K=100` to approximate token-level KL/JSD without storing full logits for both passes.

### Key Innovations

- A **minimal privileged-information ablation** for on-policy distillation: does OPSD need full reference solutions, or only final answers?
- A **compute-matched** comparison against GRPO that counts OPSD’s additional teacher-logprob computation.
- A pre-registered **hard-subset analysis** to test the regime where answer-only conditioning is most likely to fail.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) reasoning distillation (teacher traces → student), (ii) RLVR for verifiable reasoning (GRPO/PPO variants), and (iii) on-policy distillation / context distillation, which aims to reduce exposure bias while keeping dense learning signals.

A common empirical pattern across these areas is that dense guidance (full solutions, process rewards, token-level distillation) can be more sample-efficient than sparse outcome rewards, but may require more structured supervision. Recent work also emphasizes **compute accounting** (e.g., token budgets, rollout counts) because many reported gains are sensitive to generation length and number of samples.

### Related Papers

- **[Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs](https://arxiv.org/abs/2601.18734)**: Introduces OPSD and shows it can match/beat GRPO with 4–8× fewer generated tokens on competition math.
- **[Reinforcement Learning via Self-Distillation (SDPO)](https://arxiv.org/abs/2601.20802)**: Converts tokenized feedback into dense distillation targets; provides top-K distillation and compute-overhead discussion.
- **[Learning beyond Teacher: Generalized OPD with Reward Extrapolation](https://arxiv.org/abs/2602.12125)**: Connects OPD to KL-regularized RL and introduces reward scaling (ExOPD).
- **[On-Policy Context Distillation for Language Models (OPCD)](https://arxiv.org/abs/2602.12275)**: Distills context-conditioned behaviors on-policy; relevant alternative framing of teacher conditioning.
- **[MiniLLM: On-Policy Distillation of LLMs](https://arxiv.org/abs/2306.08543)**: Early OPD framework connecting reverse-KL distillation and RL-style objectives.
- **[GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300)**: Establishes GRPO-style RLVR for math with verifiable rewards.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Large-scale RLVR reasoning pipeline plus released distilled models; motivates RLVR vs distillation tradeoffs.
- **[DAPO](https://arxiv.org/abs/2503.14476)**: Open RL system for LLMs; representative of modern RLVR engineering.
- **[JustRL: Scaling a 1.5B LLM with a Simple RL Recipe](https://arxiv.org/abs/2501.17133)**: Shows stable small-model RLVR and highlights sensitivity to length/rollout choices.
- **[SASR: Step-wise Adaptive SFT+GRPO](https://arxiv.org/abs/2505.13026)**: Hybridizes SFT and GRPO adaptively; evidence that pure GRPO can collapse and hybrids can help.
- **[REDI: Reinforcement Distillation from Teacher Data](https://arxiv.org/abs/2505.24850)**: Uses positive and negative teacher traces for offline distillation; contrasts offline distillation vs online RL.
- **[RLKD: Distilling Implicit Multi-Branch Structure via RL](https://arxiv.org/abs/2505.16142)**: Structure-aware reward to distill reasoning; shows richer guidance than scalar rewards.
- **[Why Distillation can Outperform Zero-RL](https://arxiv.org/abs/2505.21067)**: Evidence that distillation can outperform zero-RL by transferring flexible reasoning behaviors.
- **[Curriculum Learning for Efficient CoT Distillation via Structure-Aware Masking and GRPO](https://arxiv.org/abs/2602.17686)**: Uses GRPO to compress reasoning traces; relevant for data/compute efficiency.
- **[Distilling the Essence: Efficient Reasoning Distillation via Sequence Truncation](https://arxiv.org/abs/2512.21002)**: Shows early tokens dominate in SFT distillation; motivates hard/prefix sensitivity.
- **[Which Reasoning Trajectories Teach Students to Reason Better? (RSR)](https://arxiv.org/abs/2601.14249)**: Analyzes which teacher traces are informative; relevant to “what information matters”.
- **[Privileged Information Distillation for Language Models (π-Distill)](https://arxiv.org/abs/2602.04942)**: Privileged information transfer for agents; similar teacher/student asymmetry but different tasks.
- **[GATES: Self-Distillation under Privileged Context with Consensus Gating](https://arxiv.org/abs/2602.20574)**: Uses privileged context + consensus to gate distillation; highlights reliability of privileged guidance.
- **[From Correction to Mastery: Reinforced Distillation](https://arxiv.org/abs/2509.14257)**: Combines teacher interventions with short-horizon RL; shows hybrid distillation/RL interactions.
- **[Lessons from Training Grounded LLMs with Verifiable Rewards](https://arxiv.org/abs/2506.15522)**: Reports task-dependent distillation vs RLVR behavior and ordering effects.
- **[A Practical Two-Stage Recipe for Mathematical LLMs](https://arxiv.org/abs/2507.08267)**: Demonstrates extended SFT + GRPO can trade accuracy vs token efficiency.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Typical eval | Limitation relevant here |
|---|---|---|---|---|
| Off-policy solution distillation | Train student on teacher solutions (SFT) | DeepSeek-R1-Distill, REDI, Distilling the Essence | GSM8K, MATH-500, AIME | Exposure bias; depends on solution traces |
| RLVR (outcome rewards) | Optimize with binary verifier rewards | DeepSeekMath/GRPO, DAPO, JustRL | MATH-500, AIME | Sparse credit assignment; multi-rollout cost |
| On-policy distillation | Train on student rollouts with teacher guidance | MiniLLM, OPSD, G-OPD | AIME, MATH | Teacher information requirements unclear |
| Privileged-context distillation | Teacher sees extra context not available at test | π-Distill, GATES, OPCD | agent/tool benchmarks, QA | Choosing minimal privileged info is open |

### Closest Prior Work

- **Self-Distilled Reasoner (OPSD)**: Closest method; conditions teacher on full reference solutions and compares to GRPO on competition math. It does not test whether the privileged information can be reduced to the final answer label.
- **Distilling the Essence (sequence truncation)**: Shows full solutions are not always necessary for *off-policy* SFT distillation and early tokens dominate, but does not study on-policy teacher guidance or answer-only conditioning.
- **MiniLLM / OPD**: Establishes on-policy distillation but relies on an external teacher (often larger) and does not study minimal privileged information for teacher conditioning.
- **SDPO**: Provides practical top-K distillation machinery and shows dense feedback can replace scalar rewards; its teacher signal comes from rich feedback rather than privileged answers.

**Novelty Kill Search Summary:** Searched for “answer-only on-policy distillation”, “answer-conditioned on-policy distillation”, “Self-Distilled Reasoner ablation answer only”, “teacher conditioned on ground-truth answer on-policy distillation”, and checked recent arXiv hits on OPD/OPSD/OPCD (2026). No paper was found that tests *answer-only vs full-solution privileged conditioning* for OPSD-style training on verifiable math as of 2026-02-28. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| OPSD (Self-Distilled Reasoner) | Self-teacher conditioned on full solution; token-level distillation on student rollouts | Requires full reference solutions | Replace privileged solution with answer-only conditioning | Directly tests minimal supervision needed for OPSD-style gains |
| GRPO (DeepSeekMath) | RLVR with binary outcome rewards and group-relative normalization | Sparse credit assignment; expensive rollouts | Compare to compute-matched GRPO | Tests whether dense answer-conditioned guidance is more compute-efficient |
| Distilling the Essence | Shows early tokens dominate in SFT distillation | Off-policy only; not teacher-guided | Apply “minimal information” question to on-policy distillation | Connects token-efficiency observations to on-policy training |
| SDPO | Dense distillation targets from feedback-conditioned self-teacher | Needs rich feedback tokens | Use privileged answer as “feedback” | Keeps teacherless dense signal but with minimal supervision |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3-4B-Instruct | 4B | https://huggingface.co/Qwen/Qwen3-4B-Instruct | Open weights; used because OPSD shows gains at ≥4B |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| OpenThoughts (math subset) | Training prompts + reference solutions | up to 30k | https://huggingface.co/datasets/OpenThoughts/OpenThoughts-114k (subset) | (per dataset card; verifier should confirm) |

**Other Resources (if applicable):**
- Math-Verify answer equivalence checker: https://github.com/huggingface/math-verify (or equivalent) for parsing and equivalence

**Resource Estimate** (upper bound; conservative):
- OPSD reports training on **8×A100** GPUs with LoRA for math reasoning (arXiv:2601.18734, Sec. 4.1). Wall-clock time is not reported; we therefore propose a bounded verification budget.
- Plan: run each training condition on **4×A100** with LoRA/QLoRA, capped at **≤16 hours per run**.
  - Training runs: 3 conditions × 3 seeds × (4 GPUs × 16h) = **576 A100 GPU-hours**.
  - Evaluation (MATH-500 with up to 8 samples, plus logging diagnostics): ≤ **120 A100 GPU-hours**.
  - Total ≤ **696 A100 GPU-hours**.
- **API usage**: None.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MATH-500 | 500 competition math problems with verifiable final answers | Pass@1, Pass@8, mean output tokens | test | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | Math-Verify based evaluator |
| GSM8K (secondary) | grade-school math word problems with final numeric answers | Pass@1, Pass@8 | test | https://huggingface.co/datasets/openai/gsm8k | Math-Verify based evaluator |

**Hard subset (pre-registered):** MATH-500 problems with `level ≥ 4` (if `level` metadata exists; otherwise define hard subset as the top-25% longest ground-truth solutions by token length).

### Main Results

| Method | Base Model | Benchmark | Pass@1 (mean±std) | Pass@8 (mean±std) | Mean output tokens | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| Prompting (zero-shot) | Qwen3-4B-Instruct | MATH-500 | TBD | TBD | TBD | This work | No training |
| Prompting (best-of-8) | Qwen3-4B-Instruct | MATH-500 | TBD | TBD | TBD | This work | Inference scaling baseline |
| **A: GRPO (compute-matched)** | Qwen3-4B-Instruct | MATH-500 | TBD | TBD | TBD | This work | G=8, L=512 |
| **B: OPSD-full** | Qwen3-4B-Instruct | MATH-500 | TBD | TBD | TBD | This work | Teacher sees reference solution |
| **C: AC-OPSD (answer-only)** | Qwen3-4B-Instruct | MATH-500 | TBD | TBD | TBD | This work | Teacher sees final answer only |

Report the same table on the pre-registered hard subset.

### Ablation Studies

No additional training ablations are required beyond A/B/C. (Diagnostics below are analysis-only.)

### Experimental Rigor

- **Seeds**: `seeds=[42, 123, 456]` for A/B/C; report mean±std.
- **Compute matching control**: match forward-token budget per prompt per update as described above; log actual generated tokens per update.
- **Confounders and controls**:
  1. **Budget confound**: GRPO vs OPSD have different per-update work; token-matching explicitly controls the dominant cost.
  2. **Answer leakage in evaluation**: teacher privileged answers are used only during training; evaluation prompts never include answers.
  3. **Prompt-format sensitivity**: use identical evaluation prompt templates across all trained models.

**Analysis-only diagnostics (pre-registered):**
- **Teacher sharpness gap**: On a fixed set of 200 training prompts and the same sampled student prefixes, measure (i) average entropy of teacher next-token distributions for B vs C, and (ii) average top-K KL between teacher_B and teacher_C. Interpretation: if C underperforms and teacher_C is much higher-entropy early, it supports the “diffuse teacher signal” mechanism.

---

## Success Criteria

**Hypothesis (directional):** Full-solution OPSD (B) will outperform compute-matched GRPO (A) on MATH-500, and answer-only conditioning (C) will lose a non-trivial fraction of that gain on the hard subset.

**Decision Rule (concrete):**

Let Δ = (B − A) on MATH-500 Pass@1.

1. **Check whether OPSD beats GRPO in this budget regime (prerequisite).**
   - If Δ < 2.0 points (mean across 3 seeds), treat the result as: “OPSD does not outperform compute-matched GRPO at 4B under this training budget.” In this case, do **not** claim anything about answer-only sufficiency; refute the core premise for this regime.

2. **Answer-only sufficiency test (only if Δ ≥ 2.0).**
   - **Proceed (answer-only sufficient)** if:
     - (B − C) ≤ 0.2·Δ on the hard subset, and
     - a paired bootstrap 95% CI for (C − B) on MATH-500 lies within ±0.5 points.
   - **Refute (full solutions necessary)** if:
     - (B − C) > 0.2·Δ on the hard subset, or
     - C ≤ A on MATH-500 (answer-only provides no benefit over compute-matched GRPO).
   - **Pivot (mechanism unclear)** if B > A but C < B and the teacher-sharpness diagnostic shows teacher_C is not higher-entropy than teacher_B; this suggests our “diffuse teacher” explanation is incomplete. (A follow-up would consider truncated reference solutions or alternative privileged signals, outside this proposal.)

---

## Impact Statement

If answer-only conditioning is sufficient, practitioners can apply OPSD-style token-efficient post-training using only (problem, final answer) labels, avoiding the cost and curation of full solution traces while still beating compute-matched RLVR. If full reference solutions are necessary (especially on hard problems), the result is still decision-changing: it implies that token-efficient on-policy distillation depends on rich privileged information, and teams should invest in high-quality solution traces (or accept the higher sampling cost of RLVR).

---

## References

Key papers (additional citations appear in Related Work):

- [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734) - Zhao et al., 2026
- [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) - Hubotter et al., 2026
- [Distilling the Essence: Efficient Reasoning Distillation via Sequence Truncation](https://arxiv.org/abs/2512.21002) - Chen et al., 2025
- [Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) - DeepSeek-AI et al., 2025
