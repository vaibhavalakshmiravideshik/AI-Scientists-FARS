# untitled

# Timeout Bootstrapping at Token Caps for Long-CoT RLVR

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) can be trained with feedback from automated checkers, such as exact-match answer checking for math or unit tests for code. This setting is often called **reinforcement learning with verifiable rewards (RLVR)**: each model output receives an objective reward computed by a programmatic verifier.

For hard reasoning tasks, RLVR often produces long chain-of-thought (CoT) solutions. In practice, RLVR systems impose a **maximum generation length** (`max_new_tokens`) for cost and stability. When generation hits this cap, the rollout is **truncated**.

### The Problem

A common engineering choice is to treat truncation as terminal failure (e.g., assign the same reward as an incorrect answer, or apply an “EOS trick” that assigns a fixed negative reward when the model fails to emit an end-of-sequence token before the cap). This conflates **unfinished** trajectories with **incorrect** ones and can bias training against long-horizon reasoning.

Two recent open RLVR systems highlight the issue but do not resolve the best practice:

- **Step 3.5 Flash** states that assigning a default failure reward to context-truncated trajectories destabilizes RL, and proposes replacing it with a bootstrapped value estimate of the truncated final state (Eq. (3) in `./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/sections/MIS-Filtered Policy Optimization (MIS-PO).md`).
- **DAPO** argues that truncation-related reward noise harms long-CoT RL and proposes overlong filtering and a soft length penalty near the cap (Eq. (13) in `./references/DAPO-An-Open-Source-LLM-Reinforcement-Learning-System-at-Scale/sections/3.4 Hide and Seek Overlong Reward Shaping.md`).

However, it remains unclear whether (i) truncation-as-failure, (ii) DAPO-style length shaping, or (iii) timeout bootstrapping is the best truncation handling strategy in a controlled, reproducible RLVR setting.

### Key Insight and Hypothesis

**Thesis (one sentence)**: In long-CoT RLVR with a non-trivial truncation rate, treating token-cap truncations as **timeouts** and bootstrapping their return from a critic value estimate improves accuracy on long problems and reduces training collapse compared to truncation-as-failure rewards and DAPO-style overlong reward shaping.

**Hypothesis**: The main harm from truncation is label noise: prefixes that would likely lead to a correct answer if continued are labeled as failures. Bootstrapping truncated rollouts with a bounded estimate of expected final verifiable reward provides a less noisy signal while keeping the training loop fully automated.

**Most likely alternative explanation**: The critic’s value estimates at truncation states are poorly calibrated (out-of-distribution), so bootstrapping injects noise and does not help. We therefore (i) clip and stop-gradient the bootstrapped reward, (ii) train the critic only on completed rollouts, and (iii) evaluate on a pre-registered long-problem subset where truncation matters.

---

## Proposed Approach

### Overview

We propose a minimal change to PPO-style RLVR: when a rollout is truncated at the token cap, treat it like a reinforcement-learning time limit and bootstrap its terminal return from a critic value estimate rather than assigning a fixed failure reward.

This is the language-model analogue of **partial-episode bootstrapping** for time limits in RL (`./references/Time-Limits-in-Reinforcement-Learning/meta/meta_info.txt`).

### Method Details

**Base reward.** We use DAPO’s rule-based outcome reward (`./references/DAPO-An-Open-Source-LLM-Reinforcement-Learning-System-at-Scale/sections/2.4 Rule-based Reward Modeling.md`):
- `R_ver = +1` if the extracted answer is equivalent to ground truth, else `-1`.

If the output is truncated and no answer can be extracted, we treat `R_ver = -1` (failure) unless otherwise specified.

**Truncation flag.** A rollout is *truncated* if generation hits `max_new_tokens` without EOS.

We compare three truncation-handling strategies under the same PPO hyperparameters and rollout budget:

- **Condition A (truncate-as-failure baseline)**: If truncated, set `R := -1` (regardless of partial reasoning quality).

- **Condition B (DAPO overlong reward shaping baseline)**:
  - Compute `R := R_ver + R_length(y)` where `R_length` is DAPO Eq. (13):
    - Let `L_max = max_new_tokens` and `L_cache = 0.2 · L_max` (matching DAPO’s 4096/20480 ratio).
    - `R_length(y)=0` if `|y| ≤ L_max − L_cache`; `R_length(y)=((L_max − L_cache) − |y|)/L_cache` if `L_max − L_cache < |y| ≤ L_max`; and `R_length(y)=-1` if `|y| > L_max`.

- **Condition C (timeout bootstrapping; proposed)**:
  - Train a critic `V_\phi(s)` (value head) to predict expected `R_ver` from prefixes, using only **completed** rollouts as targets.
  - If truncated at final state `s_T`, set
    `R := stopgrad(clip(V_\phi(s_T), -1, 1))`.
  - Exclude truncated rollouts from the critic loss (avoid “self-fulfilling” bootstrapped targets).

**Why this is a small code change.** With sparse terminal rewards and `\gamma=1`, implementing Condition C can be done by replacing the terminal reward on truncated rollouts before computing **generalized advantage estimation (GAE)** (the standard PPO procedure for turning rewards + value predictions into advantage targets), without changing the rest of PPO.

**Pre-registered collapse criteria.** A run is *collapsed* if any of:
1) NaN/Inf in loss or gradients; OR
2) actor token entropy < 0.2 nats for 200 consecutive update steps; OR
3) KL-to-reference < 0.01 and mean `R_ver` does not improve for 500 steps.

**Pre-registered truncation regime.** Before training, run the base model on 512 training prompts to choose the smallest `L_max` that yields a truncation rate in [10%, 20%] under the training decoding settings. If no such `L_max` exists, we report that truncation is not a binding failure mode for this model/task regime.

### Key Innovations

- A direct test of whether **timeout bootstrapping** is a better truncation handling rule than widely used heuristics (truncate-as-failure; length penalties) in long-CoT RLVR.
- A bounded instantiation tailored to LLM RLVR: `clip(·)` + `stopgrad(·)` and critic trained only on completed rollouts.

---

## Related Work

### Field Overview

Open RLVR systems for reasoning focus on stability under long outputs (entropy collapse, reward noise, sampling bias). Many methods modify advantage estimation, sampling, or reward shaping, but truncation handling at token caps remains inconsistent across pipelines.

Timeout bootstrapping is standard in RL when episodes end due to artificial time limits, but it has not been cleanly evaluated as a drop-in truncation rule for long-CoT LLM RLVR.

### Related Papers

- **[Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)**: States truncation reward replacement `R:=V(s_T)` as an RL stabilizer at ~20% truncation.
- **[DAPO](./references/DAPO-An-Open-Source-LLM-Reinforcement-Learning-System-at-Scale/meta/meta_info.txt)**: Open RL system; introduces overlong filtering and soft length penalties for long-CoT stability.
- **[Time Limits in RL](./references/Time-Limits-in-Reinforcement-Learning/meta/meta_info.txt)**: Formalizes timeouts vs terminations; proposes partial-episode bootstrapping.
- **[JustRL](./references/JustRL-Scaling-a-1.5B-LLM-with-a-Simple-RL-Recipe/meta/meta_info.txt)**: Shows stable 1.5B GRPO training with fixed context caps and warns some length penalties can hurt.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Large-scale RLVR for reasoning with verifiable rewards.
- **[DeepSeekMath](https://arxiv.org/abs/2402.03300)**: Introduces GRPO-style RL for math reasoning and standard benchmark protocols.
- **[PPO](https://arxiv.org/abs/1707.06347)**: Actor-critic policy optimization with value baselines and clipping.
- **[N+ RLHF PPO Details](https://openreview.net/forum?id=kHO2ZTa8e3)**: Documents the “EOS trick” penalty for no-EOS generations in PPO-style RLHF.
- **[DeepScaleR](https://arxiv.org/abs/2504.20571)**: Scales RL for a 1.5B reasoning model with length curricula.
- **[FastCuRL](https://arxiv.org/abs/2503.17287)**: Stage-wise curricula for long-CoT RL training.
- **[SIRI](https://arxiv.org/abs/2509.25176)**: Alternates shorter/longer rollout caps to manage overthinking and efficiency.
- **[LAPO](https://arxiv.org/abs/2507.15758)**: Trains length-adaptive reasoning budgets.
- **[UloRL](https://arxiv.org/abs/2507.19766)**: Ultra-long output RL and entropy-collapse mitigations.
- **[APRIL](./references/APRIL-Active-Partial-Rollouts-in-Reinforcement-Learning-to-Tame-Long-tail-Generation/meta/meta_info.txt)**: Improves RL throughput by aborting long-tail generations and resuming partial rollouts; orthogonal to how truncation is labeled in the reward.
- **[T-PPO](https://arxiv.org/abs/2506.15050)**: PPO variants that use truncated rollouts for efficiency and partial-trajectory learning.
- **[RLVE](https://arxiv.org/abs/2511.07317)**: Scales RLVR with adaptive verifiable environments and curricula.
- **[When Sharpening Becomes Collapse](https://arxiv.org/abs/2601.15609)**: Analyzes collapse in RLVR and proposes calibration mechanisms.
- **[GMPO](https://arxiv.org/abs/2507.20673)**: Replaces arithmetic with geometric aggregation in GRPO to reduce instability.
- **[ASPO](https://arxiv.org/abs/2510.06062)**: Addresses asymmetric importance sampling effects in outcome-supervised RL.
- **[OREO](https://arxiv.org/abs/2412.16145)**: Studies RL objectives and optimization variants for reasoning.
- **[Likelihood-Based Reward Designs](https://arxiv.org/abs/2602.03979)**: Explores reward designs beyond strict verifiers for reasoning RL.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Typical eval | Limitation relevant here |
|---|---|---|---|---|
| RLVR for reasoning | Policy-gradient RL with verifiable outcome rewards | DeepSeekMath, DeepSeek-R1, DAPO, JustRL | AIME, MATH-500 | Long outputs; truncation events |
| Length / truncation shaping | Penalize or schedule length to stabilize training | DAPO, DeepScaleR, FastCuRL, SIRI, LAPO, UloRL | AIME, MATH-500 | May discourage long reasoning; ambiguous handling of truncation |
| Partial / truncated PPO | Learn from partial trajectories for efficiency | T-PPO | AIME | Credit assignment choices change stability |
| RL time-limit theory | Handle artificial time limits via bootstrapping | Time Limits in RL | Control tasks | Not directly validated for LLM RLVR |

### Closest Prior Work

- **Step 3.5 Flash** proposes `R:=V(s_T)` for truncated rollouts but does not provide an open, controlled comparison against common truncation heuristics.
- **DAPO** provides a strong open baseline for long-CoT RL with explicit overlong reward shaping, but does not test timeout bootstrapping.
- **Time Limits in RL** provides the theoretical motivation for bootstrapping at timeouts, but does not study token-capped language-model rollouts.

### Comparison Table

| Related work | Truncation handling | What we change | Key empirical question |
|---|---|---|---|
| Step 3.5 Flash | Claims reward replacement `R:=V(s_T)` helps | Reproduce as a drop-in rule on open RLVR | Does it beat standard heuristics on public benchmarks? |
| DAPO | Length shaping and filtering | Replace failure reward at truncation with value bootstrap | Is bootstrapping better than length penalties? |
| Time Limits in RL | Bootstrapping at timeouts (control tasks) | Apply to token caps in RLVR | Does the RL principle transfer to LLM rollouts? |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | Small model for multi-seed PPO-style RLVR |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| DAPO-Math-17k | RLVR prompts + answers | 17k | https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k | Apache-2.0 |

**Other Resources (if applicable):**
- Training codebase: `verl` (https://github.com/volcengine/verl), using its PPO trainer (actor + value head) and DAPO-style verifier.

**Resource Estimate**:
- **Compute budget**: ~300–450 A100 GPU-hours total (3 conditions × 2 seeds, plus calibration + evaluation).
- **GPU memory**: ≤80GB per GPU.
- **API usage**: None.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MATH-500 | 500 competition math problems with verifiable answers | Pass@1 (accuracy; higher is better) | test | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | Use the same answer extraction + equivalence checker as training |

**Primary evaluation slice (pre-registered long subset).** Using the base (pre-RL) model, sample 4 solutions per MATH-500 problem with a large cap, compute median completion length, and freeze the top-25% longest problems as the “long subset.” Report Pass@1 on (i) all MATH-500 and (ii) the long subset.

**Decision rule (primary).** Condition C is a “win” if:
1) its mean long-subset Pass@1 across 2 seeds is higher than both A and B, and
2) a paired bootstrap over problems yields a 95% CI for (C − max(A,B)) that excludes 0, and
3) C has no higher collapse incidence than max(A,B) under the pre-registered collapse criteria.

### Main Results

| Method | Base Model | Benchmark | Pass@1 (all) | Pass@1 (long subset) | Collapse? | Notes |
|---|---|---|---:|---:|---|---|
| Truncate-as-failure (A) | R1-Distill-Qwen-1.5B | MATH-500 | TBD | TBD | TBD | Baseline |
| DAPO overlong shaping (B) | R1-Distill-Qwen-1.5B | MATH-500 | TBD | TBD | TBD | Baseline |
| **Timeout bootstrapping (C)** | R1-Distill-Qwen-1.5B | MATH-500 | TBD | TBD | TBD | Proposed |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| C without critic masking on truncated rollouts | Include truncated rollouts in critic loss using bootstrapped targets | Tests whether “critic self-training” is harmful or helpful |

### Analysis (Optional)

- **Value informativeness on prefixes**: On completed rollouts, measure correlation between `V(s_t)` at late prefixes and final correctness.

---

## Success Criteria

**Criterion 1: Accuracy on long problems improves**
- Hypothesis: Timeout bootstrapping improves Pass@1 on the pre-registered long subset relative to both baselines under the same training budget.
- Validation: The decision rule above is satisfied.

**Criterion 2: Stability does not degrade**
- Hypothesis: Timeout bootstrapping does not increase collapse incidence relative to baselines.
- Validation: Using the pre-registered collapse criteria, C collapses no more often than A/B.

---

## Impact Statement

If supported, timeout bootstrapping provides a simple truncation-handling rule for open RLVR pipelines: treat token caps like RL timeouts and bootstrap rather than penalize. This would be directly useful when training reasoning models under strict context limits where truncation rates are non-negligible.

---

## References

Key references (additional citations appear in Related Work):

- [Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)
- [DAPO](./references/DAPO-An-Open-Source-LLM-Reinforcement-Learning-System-at-Scale/meta/meta_info.txt)
- [Time Limits in Reinforcement Learning](./references/Time-Limits-in-Reinforcement-Learning/meta/meta_info.txt)
- [PPO](https://arxiv.org/abs/1707.06347)
- [The N+ Implementation Details of RLHF with PPO](https://openreview.net/forum?id=kHO2ZTa8e3)
