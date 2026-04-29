# untitled

# Meta-Experience Learning for Code RLVR with Unit-Test Rewards

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models can be improved at code generation by training them with feedback from **automated programmatic checks**—most commonly, running unit tests on generated code and giving a reward based on whether the code passes. This paradigm, often called **reinforcement learning with verifiable rewards (RLVR)**, avoids training a separate reward model and provides objective, reproducible supervision.

However, unit-test rewards are typically **sparse and delayed**: many training examples yield only “pass/fail” signals at the end of the program. Sparse rewards make it hard to assign credit to intermediate decisions (e.g., missing an edge case vs. choosing the wrong algorithm), and RLVR methods such as **Group Relative Policy Optimization (GRPO)**—a GRPO-style policy-gradient method that updates the model using relative rewards within a sampled group—can learn slowly when few sampled candidates pass.

A recent approach, **Meta-Experience Learning (MEL)**, proposes to address this meta-learning bottleneck in RLVR by converting paired **correct vs. incorrect** trajectories into a reusable “meta-experience” that captures the **bifurcation point** (the earliest step where the successful and failing solutions begin to differ), a **critique** (root-cause analysis), and an abstract **heuristic**, then **validating** these experiences via replay and **internalizing** them into model weights via a negative log-likelihood (NLL) loss. MEL reports consistent gains over GRPO on math reasoning benchmarks (+3.9 to +4.7 Pass@1; higher is better) across Qwen3 4B/8B/14B **[MEL](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/meta/meta_info.txt)**.

### The Problem

It is unclear whether MEL’s mechanism transfers from stepwise mathematical reasoning to **code RLVR**:

- **Code trajectories may not have clean stepwise structure.** Two candidate solutions can differ from the first line (different algorithms), which may make “bifurcation point” identification less meaningful.
- **MEL requires both correct and incorrect samples under the same prompt** (non-empty sets \(Y^+\) and \(Y^-\) in a rollout group). For very weak or very strong code models, this co-occurrence may be rare.
- **Replay validation may reject most meta-experiences**, reducing the effective training signal and complicating compute-matched comparisons.

At the same time, if MEL does transfer, it would provide a lightweight, verifier-compatible way to transform unit-test failures into reusable bug-prevention heuristics that are internalized into the policy, potentially improving the sample-efficiency of code RLVR.

### Key Insight and Hypothesis

**Key insight:** Unit-test feedback naturally produces contrastive evidence: for the same programming prompt, some sampled programs pass tests while others fail. Comparing a passing solution to a failing one can expose an actionable “bug pattern” (e.g., missed boundary condition, wrong interpretation of specification), which can be abstracted into a heuristic and reused.

**Hypothesis:** In code RLVR with unit-test rewards, adding MEL-style meta-experience construction + replay validation + internalization will improve held-out code correctness (Pass@1) beyond (i) GRPO alone and (ii) a compute-matched “self-critique” baseline that adds extra language-modeling updates but lacks MEL’s contrastive + validation mechanism.

This hypothesis could be wrong if code solutions are too diverse to yield stable bifurcation localization, or if replay validation accepts too few meta-experiences to provide meaningful signal.

---

## Proposed Approach

### Overview

We propose **MEL-Code**, a domain transfer of MEL to code RLVR. MEL-Code runs standard GRPO training using unit-test rewards, and additionally (only on prompts where both passing and failing candidates exist) constructs and validates meta-experiences from contrastive pairs, then internalizes validated meta-experiences into the policy via an auxiliary NLL objective.

### Method Details

**Base RLVR setup (GRPO).** For each programming prompt \(x\), sample a group of \(G\) candidate solutions \(Y=\{y_1,\dots,y_G\}\) from the current policy \(\pi_\theta\) (temperature 1.0). Execute each candidate against the benchmark’s unit tests to compute a verifiable reward \(r_i\in[0,1]\) (fraction of tests passed) and a binary “pass-all-tests” indicator. Partition the group into \(Y^+\) (pass-all-tests) and \(Y^-\) (not pass-all-tests). Update the policy with standard GRPO using group-normalized advantages.

**(1) Meta-experience construction from contrastive pairs.** For prompts with non-empty \(Y^+\) and \(Y^-\), sample one contrastive pair \((y^+,y^-)\). We represent each solution as a sequence of “steps” by requiring the model to output:
- a short numbered plan (3–6 lines) inside a `<think>` block, followed by
- a Python code block inside `<answer>`.

We then prompt the policy model (as in MEL) to produce a meta-experience tuple:
\[
M=(s^*, C, H)
\]
where \(s^*\) is the identified bifurcation step, \(C\) is a critique explaining the root cause of failure vs. the successful trajectory, and \(H\) is an abstract heuristic intended to generalize to similar problems.

**(2) Empirical validation via replay.** We test each candidate meta-experience \(M\) by injecting it as a short “experience” hint (prompt template from MEL) and re-generating a solution for the same prompt \(x\). If the regenerated solution passes all unit tests, \(M\) is accepted into \(D_M^*\); otherwise it is discarded.

**(3) Internalization into parametric memory.** For validated meta-experiences \(M\in D_M^*\), add an auxiliary token-level NLL loss that predicts \(M\) given the retrospective context \([x,y^+,y^-]\), following MEL’s internalization mechanism. Optimize a joint objective:
\[
J(\theta)=J_{\text{GRPO}}(\theta)+\lambda\,J_{\text{NLL}}(\theta)
\]
with \(\lambda\) chosen so the gradient norm from NLL is comparable to the GRPO term (tuned on a small dev set).

**Compute-matched self-critique control.** To distinguish MEL’s contrastive+validation mechanism from “just adding extra LM updates”, we include a baseline that adds the same NLL-token budget but without contrastive pairing:
- given a single failing trajectory \(y^-\), prompt the model to write a critique+heuristic about that failure (no \(y^+\), no bifurcation step), and train on it with the same NLL loss.

### Key Innovations

- **Domain transfer:** adapts MEL’s meta-experience loop from math to code RLVR with unit-test rewards.
- **Mechanism-isolating baseline:** a compute-matched self-critique baseline to test whether contrastive+validated meta-experiences matter beyond extra LM updates.
- **Feasibility-first diagnostics:** a small pre-training pilot to measure whether code RLVR yields enough contrastive pairs and replay-validated meta-experiences to sustain MEL.

---

## Related Work

### Field Overview

RLVR for code generation uses deterministic feedback (compilation, unit tests, execution traces) to train code models. A large body of work improves credit assignment by adding denser signals than terminal pass/fail, including process reward models (PRMs) and multi-turn refinement training.

MEL is part of a recent line of “experience learning” in RLVR that attempts to learn from correct/incorrect trajectory structure without training a separate PRM. Prior experience methods often provide external hints/prefixes during training, which can introduce distribution mismatch at inference time. MEL’s distinctive claim is that validated meta-experiences can be **internalized into weights** and act like a dense process signal while remaining compatible with verifiable rewards.

### Related Papers

- **[MEL: Internalizing Meta-Experience into Memory for Guided RL](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/meta/meta_info.txt)**: Constructs (s*, critique, heuristic) from correct/incorrect pairs, validates via replay, internalizes via NLL; shown on math RLVR.
- **[Murphy: Multi-Turn GRPO for Self Correcting Code Generation](./references/Murphy-Multi-Turn-GRPO-for-Self-Correcting-Code-Generation/meta/meta_info.txt)**: Extends GRPO to multi-turn self-correction with feedback and credit assignment; shows code gains in iterative settings.
- **[CodeRL](https://arxiv.org/abs/2207.01780)**: Early RL with unit-test feedback for code generation (actor-critic); establishes execution feedback as a training signal.
- **[PPOCoder](https://arxiv.org/abs/2301.13816)**: PPO for code generation with execution + AST/DFG rewards; targets reward sparsity and generalization.
- **[RLTF: Reinforcement Learning from Unit Test Feedback](https://arxiv.org/abs/2307.04349)**: Uses unit-test feedback with finer-grained reward shaping for code.
- **[PRLCoder: Process-Supervised RL for Code Generation](https://arxiv.org/abs/2502.01715)**: Trains a process reward model (PRM) with automatically constructed line-level supervision to densify code RL.
- **[PSGPO: Process Supervision-Guided Policy Optimization for Code Generation](https://arxiv.org/abs/2410.17621)**: PRM-based dense rewards / value initialization for RL code generation.
- **[CodeRL+](https://arxiv.org/abs/2510.18471)**: Adds execution-semantics alignment objectives to densify learning beyond pass/fail.
- **[StepCoder](https://arxiv.org/abs/2402.01391)**: Curriculum + fine-grained optimization for code RL.
- **[GVPO](https://openreview.net/forum?id=RY47Tq0VsV)**: Combines outcome-verifiable and process-verifiable signals for interactive coding agents.
- **[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)**: Introduces GRPO for verifiable reasoning; widely used baseline in RLVR.
- **[VERL](https://openreview.net/forum?id=7WZ8VdR1sW)**: RLHF/RLVR framework used by recent GRPO-style systems.
- **[Let’s Verify Step by Step](https://openreview.net/forum?id=v8L0pN6EOi)**: Process supervision and verifiers for reasoning, motivating dense learning signals.
- **[Process Reward Models that Think](https://arxiv.org/abs/2504.16828)**: Introduces ThinkPRM, a generative process reward model that produces verification reasoning; motivates stronger process supervision without dense human labels.
- **[StepHint](https://openreview.net/forum?id=OwYuhlJ8SG)**: Adds stepwise hints during RLVR training; highlights experience-as-hints trade-offs.
- **[Scaf-GRPO](https://arxiv.org/abs/2510.19807)**: Hierarchical hint scaffolding to overcome GRPO learning cliffs.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Inference-time self-correction with feedback for agents; motivates multi-turn training (Murphy).
- **[ReAct](https://arxiv.org/abs/2210.03629)**: Reasoning+acting prompting; shows value of structured intermediate reasoning.
- **[EvalPlus](https://openreview.net/forum?id=1qvx610Cu7)**: Rigorous evaluation with expanded unit tests for HumanEval/MBPP, relevant to evaluating code RLVR models.
- **[DAPO](https://arxiv.org/abs/2501.01054)**: Open-source RL system at scale; provides infrastructure patterns for RLVR.
- **[All Roads Lead to Likelihood](https://arxiv.org/abs/2503.01067)**: Provides theory for when on-policy RL fine-tuning offers benefits over offline likelihood-style objectives, framing why extra NLL training may or may not match RL.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Outcome-only RLVR for code | Unit tests as terminal rewards | CodeRL; GRPO/DeepSeekMath | HumanEval, MBPP | Sparse rewards, slow early learning |
| Process reward models (PRM) | Train a reward model to score intermediate steps/lines | PRLCoder; PSGPO; PRM work | HumanEval, MBPP, LiveCodeBench | PRM training cost; reward hacking risk |
| Multi-turn refinement RL | Train with iterative feedback loops | Murphy; GVPO | HumanEval, MBPP, interactive coding benches | Higher rollout cost; credit assignment |
| Execution-semantics objectives | Supervise execution traces/variable states | CodeRL+ | HumanEval, code reasoning tasks | Additional instrumentation complexity |
| Experience learning in RLVR | Use trajectory structure to extract reusable guidance | StepHint; Scaf-GRPO; MEL | Math/code reasoning | Hint/prefix mismatch; requires reliable extraction |

### Closest Prior Work

1. **MEL** **[MEL](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/meta/meta_info.txt)**: Closest method; demonstrated on math. Our proposal tests whether the same contrastive+validated internalization mechanism works for code RLVR.
2. **PRM-based code RL (PSGPO / PRLCoder)**: Provides dense signals by training a PRM for code prefixes/lines, which can add complexity and may introduce reward-model failure modes. We keep the reward fully verifiable and use only self-generated, replay-validated meta-experiences.
3. **Murphy / multi-turn GRPO** **[Murphy](./references/Murphy-Multi-Turn-GRPO-for-Self-Correcting-Code-Generation/meta/meta_info.txt)**: Uses feedback and credit assignment across turns; orthogonal to our single-turn focus. Our goal is to densify learning via internalized meta-experiences without requiring multi-turn rollouts.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MEL | Meta-experience + replay validation + NLL internalization (math) | Not shown for code; may rely on stepwise reasoning structure | Apply MEL to code RLVR with unit tests | If code failures yield reusable bug heuristics, internalization should improve sample-efficiency |
| PRLCoder / PSGPO | Train PRM for dense intermediate rewards | PRM training cost; potential PRM errors | No PRM; only verifiable tests + replay-validated experiences | Avoid reward-model brittleness while still adding dense signal |
| Murphy | Multi-turn refinement + reward propagation | Higher rollout cost; multi-turn complexity | Single-turn RL; meta-experience internalization | Lower rollout complexity; no multi-turn credit assignment |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-Coder-Instruct | 3B or 7B | https://huggingface.co/Qwen | Choose size so baseline pass@1 is not extreme (to ensure \(Y^+\) and \(Y^-\) co-occur at temperature 1.0). |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| MBPP (train split) | RLVR training prompts + unit tests | 374 tasks (plus 10 canonical few-shot tasks, if used for prompting; verify split in the chosen MBPP loader) | https://huggingface.co/datasets/google-research-datasets/mbpp | CC BY 4.0 (verify) |

**Other Resources (if applicable):**
- Unit-test execution sandbox (Python subprocess / Docker) for safe evaluation.

**Resource Estimate**:
- **Compute budget**: Target \(<300\) A100 GPU-hours total for 3 conditions (GRPO, self-critique, MEL-Code). Main GPU cost is sampling \(G=8\) completions; unit tests run on CPU.
- **GPU memory**: 3B/7B fits on 1×A100 80GB; use 4–8 GPUs for throughput.
- **API usage**: None.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MBPP | Mostly Basic Programming Problems (MBPP): 974 Python programming tasks with reference solutions and unit tests | Pass@1 (probability the top-1 decoded solution passes all tests), Pass@k (probability at least one of k sampled solutions passes), mean reward (avg. fraction of tests passed) | test | https://huggingface.co/datasets/google-research-datasets/mbpp | Official MBPP eval / unit-test harness |
| HumanEval+ (optional) | HumanEval+ via EvalPlus: 164 Python function synthesis tasks with expanded, harder unit tests | Pass@1, Pass@k | test | https://github.com/evalplus/evalplus | EvalPlus official eval |

**Evaluation Scripts:**
- Use MBPP unit tests for reward and evaluation.
- Optionally evaluate on HumanEval+ using EvalPlus to check robustness to stronger tests.

### Main Results

We compare three compute-matched training conditions:

| Method | Base Model | Training set | Eval benchmark | Pass@1 | Pass@k | Source | Notes |
|---|---|---|---|---:|---:|---|---|
| GRPO | Qwen2.5-Coder-(3B/7B) | MBPP-train | MBPP-test | **TBD** | **TBD** | - | Baseline |
| GRPO + self-critique NLL | same | MBPP-train | MBPP-test | **TBD** | **TBD** | - | Token-matched control (no contrastive pair; no replay validation) |
| **GRPO + MEL-Code (ours)** | same | MBPP-train | MBPP-test | **TBD** | **TBD** | - | Contrastive (y+,y−) → meta-experience → replay validate → NLL internalize |

#### Decision Rule (primary)
Compute a paired bootstrap CI over evaluation tasks for the Pass@1 difference.
- **Success** if the 95% CI lower bound of \(\Delta=\text{Pass@1}_{\text{MEL-Code}}-\text{Pass@1}_{\text{GRPO}}\) is \(>0\) and \(\text{Pass@1}_{\text{MEL-Code}}\ge \text{Pass@1}_{\text{self-critique}}\).
- **Failure / negative result** otherwise.

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| No replay validation | Keep all meta-experiences (skip Eq. 6 in MEL) | If validation is important, removing it hurts or becomes unstable |
| No contrastive pair | Generate critique/heuristic from \(y^-\) only | Similar to self-critique baseline; should underperform MEL-Code |
| Bifurcation skip | Force \(s^*\) to be the first plan step | If bifurcation localization matters, performance degrades |

### Analysis (Optional)

**Stage-0 feasibility diagnostics (pilot).** Before full RL training, run on 50 training prompts to measure:
- \(p_{\text{pair}}\): fraction of prompts with both \(Y^+\) and \(Y^-\) in a group of \(G=8\).
- \(p_{\text{accept}}\): fraction of generated meta-experiences that pass replay validation.
- \(p_{\text{usable}} = p_{\text{pair}}\cdot p_{\text{accept}}\): fraction of prompts contributing validated meta-experiences.

**Pilot decision threshold:** If \(p_{\text{usable}} < 5\%\), abort the full MEL-Code run (and report a negative-transfer result attributable to signal scarcity / validation rejection).

---

## Success Criteria

**Criterion 1: Meta-experience signal exists in code RLVR**
- Hypothesis: \(p_{\text{pair}}\) and \(p_{\text{accept}}\) are non-trivial (not near zero) at temperature 1.0, enabling MEL-Code to generate validated meta-experiences on a meaningful subset of prompts.
- Validation: Report pilot rates; if \(p_{\text{usable}}\) is extremely small, MEL-Code is unlikely to help.

**Criterion 2: MEL-Code improves held-out correctness beyond compute-matched controls**
- Hypothesis: MEL-Code improves Pass@1 over GRPO and is not worse than the self-critique control.
- Validation: Bootstrap CI decision rule in Main Results.

---

## Impact Statement

If MEL-Code works, it provides a verifier-compatible way to transform unit-test failures into reusable bug-prevention heuristics that are internalized into the policy, improving the sample-efficiency of code RLVR without training a separate process reward model. If it fails, the result is still informative: it would indicate that MEL’s reliance on contrastive bifurcation analysis and replay validation does not transfer cleanly to code, clarifying the boundary conditions of experience-learning claims in RLVR.

---

## References

- [Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/meta/meta_info.txt) - Huang et al., 2026
- [Murphy: Multi-Turn GRPO for Self Correcting Code Generation](./references/Murphy-Multi-Turn-GRPO-for-Self-Correcting-Code-Generation/meta/meta_info.txt) - Ekbote et al., 2025
- [CodeRL: Mastering Code Generation through Pretrained Models and Reinforcement Learning](https://arxiv.org/abs/2207.01780) - Le et al., 2022
- [Execution-based Code Generation using Deep Reinforcement Learning (PPOCoder)](https://arxiv.org/abs/2301.13816) - Deng et al., 2023
- [RLTF: Reinforcement Learning from Unit Test Feedback](https://arxiv.org/abs/2307.04349) - Huang et al., 2023
- [Process-Supervised Reinforcement Learning for Code Generation (PRLCoder)](https://arxiv.org/abs/2502.01715) - Ye et al., 2025
- [Process Supervision-Guided Policy Optimization for Code Generation (PSGPO)](https://arxiv.org/abs/2410.17621) - Dai et al., 2024
- [CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](https://arxiv.org/abs/2510.18471) - Jiang et al., 2025
- [StepCoder: Improve Code Generation with Reinforcement Learning from Compiler Feedback](https://arxiv.org/abs/2402.01391) - Dou et al., 2024
- [Group Verification-based Policy Optimization for Interactive Coding Agents](https://openreview.net/forum?id=RY47Tq0VsV) - (authors hidden in OpenReview snapshot; cite as ICLR 2026 submission)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [HybridFlow / VERL: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256) - Sheng et al., 2024
- [Let’s Verify Step by Step](https://openreview.net/forum?id=v8L0pN6EOi) - Lightman et al., 2024
- [StepHint: Multi-level Stepwise Hints Enhance Reinforcement Learning for LLM Reasoning](https://openreview.net/forum?id=OwYuhlJ8SG) - Zhang et al., 2025
- [Scaf-GRPO: Scaffolded Group Relative Policy Optimization for LLM Reasoning](https://arxiv.org/abs/2510.19807) - Zhang et al., 2025
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - Shinn et al., 2023
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - Yao et al., 2023
- [EvalPlus: Rigorous Evaluation of Large Language Models for Code Generation](https://openreview.net/forum?id=1qvx610Cu7) - Liu et al., 2023
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2501.01054) - Yu et al., 2025
- [A Survey of Process Reward Models: From Outcome Signals to Thought-Process Supervision](https://arxiv.org/abs/2510.08049) - Zheng et al., 2025
