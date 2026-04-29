# untitled

# Revision-Augmented Meta-Experience Learning: Recovering Contrastive Signal from All-Negative RLVR Groups

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) can be improved on tasks like mathematical problem solving by training them with feedback from automated checkers (e.g., symbolic answer verification). This approach—often called **reinforcement learning with verifiable rewards (RLVR)**—avoids training a separate reward model by using a programmatic verifier that returns an objective pass/fail signal. A widely used RLVR optimizer is **Group Relative Policy Optimization (GRPO)**, which samples multiple solutions per prompt and uses the within-group reward statistics as a baseline for policy updates.

A recurring practical issue in RLVR is **sample inefficiency early in training**: many prompt groups contain only incorrect solutions (an “all-negative group”), producing weak or noisy learning signals. Several recent works attempt to extract richer information from incorrect trajectories, for example by grading partial correctness or reweighting negative groups.

**Meta-Experience Learning (MEL)** proposes a different mechanism: instead of treating correct and incorrect trajectories independently, it performs contrastive analysis on paired correct/incorrect trajectories to identify where reasoning diverges (a **bifurcation point**), extracts a reusable “meta-experience” (bifurcation point + critique + heuristic), validates it via replay, and then distills it into the model weights as a dense, language-modeled process reward. MEL improves Pass@1 over GRPO on math reasoning benchmarks across Qwen3-4B/8B/14B models.

### The Problem

MEL’s meta-experience construction **requires both at least one correct trajectory (Y+) and at least one incorrect trajectory (Y−) in the same rollout group** so that it can build contrastive pairs. The paper explicitly states that it “only consider[s] gradient-informative samples with non-empty Y+ and Y−” (Meta-Experience Construction in **[MEL](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/meta/meta_info.txt)**). This implies that **all-negative groups are discarded for meta-experience construction**, exactly in the regime where RLVR methods often struggle.

At a systems level, this means MEL’s most expensive stage (meta-experience generation + replay validation + NLL distillation) is applied to only a subset of sampled prompts, potentially leaving substantial usable signal on the floor when the policy is weak (many all-negative groups). A simple fix would be “sample more” (increase group size), but this is expensive and still does not guarantee a correct trajectory appears for hard prompts.

We aim to answer a concrete question: **can we recover MEL’s contrastive learning loop on all-negative groups without increasing total rollout compute**, by spending a small, targeted budget to turn one incorrect solution into a near-miss correct solution?

### Key Insight and Hypothesis

**Key insight**: All-negative groups often contain a trajectory that is “almost correct” (e.g., correct setup with one arithmetic or logical slip). If we can cheaply obtain a corrected version of that trajectory that preserves a long shared prefix, we can create a high-quality contrastive pair (y+, y−) that (i) provides a well-defined bifurcation point via the first divergence location and (ii) should be more informative than pairing an arbitrary correct solution with an arbitrary incorrect solution.

**Hypothesis**: When a rollout group contains only incorrect solutions, using a **bounded verifier-guided revision** procedure (a small number of self-revision attempts conditioned on an incorrect trajectory) will produce prefix-aligned correct trajectories more reliably than spending the same compute on additional independent rollouts. This will increase the number of **replay-validated** meta-experiences per prompt and improve downstream Pass@1 compared to a compute-matched “MEL + extra rollouts” control.

Why this might fail: (i) revisions may rarely succeed within a small budget, (ii) revisions may rewrite too much, making the inferred bifurcation point meaningless, or (iii) the replay validation step may pass but still admit meta-experiences that do not generalize, yielding no downstream improvement.

---

## Proposed Approach

### Overview

We propose **Revision-Augmented MEL (R-MEL)**, a drop-in modification to MEL that targets **all-negative rollout groups**. The method keeps MEL’s meta-experience extraction, replay validation, and internalization mechanisms unchanged. The only change is how we handle a prompt group when the verifier marks all sampled solutions incorrect.

For an all-negative group, R-MEL spends a small additional generation budget (e.g., 2 attempts) to revise one chosen incorrect trajectory into a correct one, subject to a prefix-preservation constraint. If successful, we now have both Y+ and Y− and can run MEL’s standard contrastive pipeline.

Crucially, we include a compute-matched baseline that spends the **same** additional generation budget on extra independent rollouts instead of revision, to test whether any gains are due to “smarter sampling” rather than simply “more samples.”

### Method Details

**Base rollout (same as MEL):**
1. For each prompt x, sample G=8 independent trajectories \(Y=\{y_i\}_{i=1}^G\) from the current policy \(\pi_{\theta_{old}}\) at temperature 1.0.
2. Use **Math-Verify** to compute outcome rewards \(r_i \in \{0,1\}\) by comparing each extracted answer against the ground truth.
3. Partition into \(Y^+=\{y_i:r_i=1\}\) and \(Y^-=\{y_i:r_i=0\}\).

**Standard MEL branch (unchanged):**
- If \(|Y^+|>0\) and \(|Y^-|>0\), run MEL’s meta-experience construction (bifurcation point identification, critique, heuristic), then replay validation, then internalization via NLL on the validated meta-experience tokens.

**All-negative branch (new): bounded verifier-guided revision**
- If \(|Y^+|=0\):
  1. Choose a seed trajectory \(y^-\in Y^-\) (e.g., the trajectory with the highest average token log-probability under \(\pi_{\theta_{old}}\), which is often a good proxy for “closest to the model’s mode”).
  2. For \(j=1..B\) (default \(B=2\)):
     - Prompt the **same policy model** with \((x, y^-)\) and an instruction to produce a corrected solution while changing the prefix as late as possible. Example instruction:
       - “The verifier says the final answer is incorrect. Produce a corrected solution. Keep the beginning identical as much as possible and only start changing from the first step you believe is wrong. Do not add explanations about what you changed; just output the revised solution.”
     - Sample a revised trajectory \(y^{rev}\) at temperature 1.0.
     - If Math-Verify marks \(y^{rev}\) correct, accept it as \(y^+\) **only if** it shares a sufficiently long prefix with \(y^-\).
  3. Prefix constraint / bifurcation point:
     - Compute a normalized longest-common-prefix (LCP) ratio between \(y^-\) and \(y^+\) at the token level (or step level if the training prompt enforces “Step i:” formatting).
     - Require LCP ratio \(\ge \tau\) (default \(\tau=0.5\)).
     - Define the bifurcation point \(s^*\) as the first token (or first step index) after the LCP.
  4. If no revision succeeds within B attempts, fall back to MEL’s default behavior for this prompt (no meta-experience constructed).

**Why this should help vs extra sampling:** Revision is a targeted search conditioned on a specific incorrect attempt, which (a) tends to preserve problem decomposition and (b) is more likely to fix a localized mistake than a fresh random sample. In contrast, extra sampling generates independent trajectories that may fail in unrelated ways, producing less prefix-aligned contrast.

### Key Innovations

1. **All-negative recovery for MEL**: a simple mechanism to create contrastive (y+, y−) pairs on prompts where MEL otherwise cannot construct meta-experiences.
2. **Prefix-aligned contrastive pairs**: enforce and measure prefix sharing so that the “bifurcation point” is well-defined and likely corresponds to a localized error.
3. **Compute-matched control**: compare against “MEL + extra rollouts” to distinguish “better pairing” from “more sampling.”

---

## Related Work

### Field Overview

RLVR methods for reasoning typically use outcome-level verifiers (unit tests, symbolic checks, proof assistants) to provide sparse rewards, and rely on exploration via multiple samples per prompt. GRPO-style algorithms replace a learned value function with group-relative baselines, reducing complexity but making learning sensitive to group composition. This has motivated several lines of work: (i) improving how negative samples are used (partial-credit rewards, reweighting), (ii) adding process-level supervision (process reward models; PRMs), (iii) counterfactual credit assignment at the token/span level, and (iv) trajectory editing or revision mechanisms that provide richer training signals.

MEL is closest in spirit to “learning from mistakes” approaches, but it is distinguished by explicitly extracting a reusable natural-language meta-experience and distilling it into weights after replay validation. Our proposal targets a specific bottleneck in MEL: its reliance on mixed-outcome rollout groups.

### Related Papers

- **[Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/meta/meta_info.txt)**: Introduces MEL; requires non-empty Y+ and Y− for contrastive meta-experience construction; our direct baseline and target.
- **[Don’t Waste Mistakes: Leveraging Negative RL-Groups via Confidence Reweighting](./references/Don’t-Waste-Mistakes-Leveraging-Negative-RL-Groups-via-Confidence-Reweighting/meta/meta_info.txt)**: Reweights all-negative GRPO groups using confidence; does not construct contrastive pairs or meta-experiences.
- **RE-GRPO: Leveraging hard negative cases through large language model guided self training** (Neurocomputing 2026, https://www.sciencedirect.com/science/article/abs/pii/S0925231225032151): Uses LLM-guided reflection to *repair* hard negative cases with dual validation, but does not target MEL’s contrastive-pair bottleneck or meta-experience internalization.
- **[Not All Negative Samples Are Equal: LLMs Learn Better from Plausible Reasoning](https://arxiv.org/abs/2602.03516)**: Synthesizes high-quality *plausible negatives* (via reverse GRPO) for preference training; complementary but targets negative-sample quality rather than constructing prefix-aligned (y+,y−) pairs for MEL.
- **[Stepwise Guided Policy Optimization: Coloring your Incorrect Reasoning in GRPO](./references/Stepwise-Guided-Policy-Optimization-Coloring-your-Incorrect-Reasoning-in-GRPO/meta/meta_info.txt)**: Assigns graded rewards to incorrect solutions based on first-error position; uses a judge/PRM-like signal rather than pair synthesis.
- **[Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning](./references/Save-the-Good-Prefix-Precise-Error-Penalization-via-Process-Supervised-RL-to-Enhance-LLM-Reasoning/meta/meta_info.txt)**: Uses a PRM as a first-error detector to reward correct prefixes in incorrect solutions; avoids pair construction.
- **[Counterfactual Self-Questioning for Stable Policy Optimization in Language Models](./references/Counterfactual-Self-Questioning-for-Stable-Policy-Optimization-in-Language-Models/meta/meta_info.txt)**: Generates counterfactual critiques as additional trajectories for GRPO; does not enforce prefix-preserving edits or MEL-style replay-validated meta-experiences.
- **[EditGRPO: Reinforcement Learning with Post-Rollout Edits for Clinically Accurate Chest X-Ray Report Generation](https://arxiv.org/abs/2509.22812)**: Uses post-rollout edits to form stronger contrasts in GRPO groups (medical domain); similar “edit trajectories” idea but not meta-experience extraction.
- **[Improving Large Language Models via Fine-grained Reinforcement Learning with Minimum Editing Constraint](https://arxiv.org/abs/2401.06081)**: Uses minimum-edit corrections to train token-level rewards via a generative reward model; focuses on token-level RL rather than contrastive meta-experiences.
- **[Beyond Uniform Credit: Causal Credit Assignment for Policy Optimization](https://arxiv.org/abs/2602.09331)**: Uses counterfactual masking to upweight causally important reasoning spans; complementary credit assignment technique that does not create alternative trajectories.
- **[Online Causal Kalman Filtering for Stable and Effective Policy Optimization](https://arxiv.org/abs/2602.10609)**: Stabilizes token-level importance sampling ratios; addresses variance/instability rather than sample informativeness.
- **[F-GRPO: Don’t Let Your Policy Learn the Obvious and Forget the Rare](https://arxiv.org/abs/2602.06717)**: Difficulty-aware prompt weighting to mitigate distribution sharpening and preserve rare-correct modes; complementary to our aim of extracting more usable signal on hard prompts.
- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)**: Popularizes GRPO-style RLVR for math reasoning; foundational baseline family for MEL-style work.
- **[DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)**: Preference-based optimization without RL; relevant for understanding group-relative methods and pairwise training analogies.
- **[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)**: Canonical RLHF pipeline; contrasts with RLVR where rewards are programmatic.
- **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)**: PPO baseline; GRPO can be seen as a PPO-like method with a different baseline.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Introduces step-level verification supervision; motivates PRM-based approaches like VPPO and SGPO.
- **[Dapo: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)**: System-level RLVR infrastructure and the DAPO-Math-17k dataset; our training data source.
- **[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)**: Demonstrates large-scale GRPO-based RL for reasoning; motivates data-efficiency improvements.
- **[VinePPO: Refining Credit Assignment in RL Training of LLMs](https://arxiv.org/abs/2410.01679)**: Uses Monte Carlo rollouts from intermediate states to improve credit assignment in PPO-style RL for reasoning.
- **[Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models](https://arxiv.org/abs/2508.10751)**: Uses Pass@k-based rewards/advantages to sustain exploration in RLVR.
- **[All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning](https://arxiv.org/abs/2503.01067)**: Analyzes when online RL-style fine-tuning differs from offline likelihood training; connects gains to “generation–verification gaps.”

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Meta-experience / replay-validated process guidance | Extract reusable natural-language guidance from (y+,y−) contrasts and distill it into weights | MEL | AIME24/25, AMC23, MATH500, OlympiadBench | Requires mixed-outcome rollout groups (Y+ and Y−) |
| Learning from negative groups via weighting | Keep all-negative groups by giving them non-zero weight/penalty | LENS, F-GRPO | Math reasoning (Pass@1, Pass@k) | Does not localize where reasoning diverged |
| Step-level / prefix credit assignment | Identify first wrong step and allocate partial reward to correct prefix | VPPO, SGPO, Let’s Verify Step by Step | Math reasoning | Requires step segmentation and/or PRM/judge reliability |
| Counterfactual credit at token/span level | Use interventions (mask spans) to weight gradient updates | Beyond Uniform Credit, Online Causal Kalman Filtering | GSM8K, AIME/AMC variants | Extra forward passes; domain-dependent span detection |
| Trajectory editing / revision | Edit model outputs post hoc to create stronger training signal | EditGRPO, RLMEC | Domain-specific (medical; math/QA) | Requires editing procedure; can introduce distribution shift |

### Closest Prior Work

1. **MEL** (**[MEL](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/meta/meta_info.txt)**): Builds meta-experiences from contrastive (y+,y−) pairs and distills them into weights after replay validation, but discards prompts without both Y+ and Y−. Our work targets this specific bottleneck by synthesizing Y+ for all-negative groups.
2. **EditGRPO** ([arXiv:2509.22812](https://arxiv.org/abs/2509.22812)): Introduces post-rollout edits to strengthen contrasts within a GRPO group; conceptually similar in using edits, but focuses on domain-specific clinical metrics and does not perform bifurcation-point localization + replay-validated natural-language meta-experience distillation.
3. **RLMEC** ([arXiv:2401.06081](https://arxiv.org/abs/2401.06081)): Uses minimum-edit corrections to learn token-level rewards; shares the “minimal edit” idea but targets a different objective (token-level reward modeling) rather than enabling MEL’s contrastive meta-experience construction.
4. **LENS / SGPO / VPPO**: These methods extract value from incorrect trajectories via reweighting or partial-credit rewards, but they do not create explicit contrastive pairs that can be fed into MEL’s bifurcation-point and heuristic extraction pipeline.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MEL | Contrastive (y+,y−) → bifurcation point + heuristic; replay validation; NLL distillation | Needs Y+ and Y−; discards all-negative groups | Add bounded revision to create Y+ on all-negative groups | Recovers MEL signal on hard prompts without large rollout groups |
| MEL + extra rollouts (ours baseline) | Spend extra sampling budget to try to get a correct trajectory | May still fail; successes may be unrelated (low prefix overlap) | Replace extra sampling with revision attempts conditioned on an incorrect trajectory | Revision is targeted; more likely to yield prefix-aligned y+ |
| EditGRPO | Post-rollout edits improve GRPO contrasts (medical) | Not about bifurcation localization or meta-experience distillation | Use revision only to enable MEL’s meta-experience pipeline | MEL’s replay validation filters low-quality edits; focuses edits where they matter (all-negative groups) |
| RLMEC | Minimum-edit corrections for token-level reward learning | Requires training a reward model; different supervision target | Keep MEL objective; use revision only for pair creation | Lower complexity than reward-model training; directly attacks MEL’s bottleneck |
| VPPO / SGPO | Partial credit for incorrect reasoning via first-error detection | Needs PRM/judge; not reusable heuristics distilled into weights | Use verifier-guided revision to obtain y+ and run MEL’s abstraction | Produces explicit reusable guidance and dense supervision without a separate PRM |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3-4B-Base | 4B | https://huggingface.co/Qwen/Qwen3-4B-Base | Match MEL paper’s backbone family |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| DAPO-Math-17k | RLVR training prompts with ground-truth answers | 17k | https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k | Apache-2.0 |

**Other Resources (if applicable):**
- **Verifier**: Math-Verify (https://github.com/huggingface/Math-Verify).
- **RL framework**: VERL (https://github.com/volcengine/verl), as used by MEL.

**Resource Estimate**:
- **Compute budget**: Target **≤ 600 GPU-hours** total for 3 conditions.
  - MEL reports training on **8×H20 GPUs** (Experiments section in [MEL](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/sections/Experiments.md)).
  - We plan to run on **8×A100 80GB**.
  - MEL’s appendix plots training curves up to **~140 training steps** ([A. Result of Performance Evolution](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/sections/A.%20Result%20of%20Performance%20Evolution.md)).
  - Practical step-time evidence (upper bound): a BytePlus/VERL GRPO best-practices doc reports **~222 sec/step** for Qwen2.5-7B on **8×H20** and **~338 sec/step** for Qwen2.5-14B on **2×8×H20** (16 GPUs). We use this only as an order-of-magnitude proxy; Qwen3-4B may differ.
  - Budget plan: cap at **100 steps per condition**. If step time is 3–5 minutes/step on 8 GPUs, that is **~5–8.5 hours/condition** → **40–68 GPU-hours/condition**; even at 10 minutes/step, **~13.3 hours/condition** → **~106 GPU-hours/condition**. For 3 conditions: **~120–320 GPU-hours**, leaving slack for evaluation and overhead within 600 GPU-hours.
- **GPU memory**: Qwen3-4B should fit in 80GB with RL fine-tuning; exact sharding depends on VERL configuration.
- **API usage**: None required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AIME24 | 30 competition math problems (AIME 2024) | Pass@1 (temp=0), Avg@8/Pass@8 (temp=0.6) | test | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 | Math-Verify + standard answer extraction |
| AIME25 | 30 competition math problems (AIME 2025) | Pass@1 (temp=0), Avg@8/Pass@8 (temp=0.6) | test | https://huggingface.co/datasets/math-ai/aime25 | Math-Verify + standard answer extraction |
| AMC23 | 40 AMC 12 problems (AMC 2023) | Pass@1 (temp=0), Avg@8/Pass@8 (temp=0.6) | test | https://huggingface.co/datasets/math-ai/amc23 | Math-Verify |
| MATH-500 | 500 problems from the MATH dataset | Pass@1 (temp=0), Avg@8/Pass@8 (temp=0.6) | test | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | Math-Verify |
| OlympiadBench | Olympiad-level math benchmark (subset used by MEL) | Pass@1 (temp=0), Avg@8/Pass@8 (temp=0.6) | test | https://huggingface.co/datasets/baohao/olympiadbench | Math-Verify (answer-extraction dependent) |

**Evaluation Scripts:**
- Use Math-Verify’s provided evaluation scripts where available; otherwise use a simple harness: generate responses, extract final answers, and verify with Math-Verify.

### Main Results

#### Published anchor numbers (MEL paper)

From MEL’s `Joint Training Objective.md` (Qwen3-4B-Base, accuracy %):

| Benchmark | Baseline (P@1/Avg@8/P@8) | GRPO | MEL |
|---|---:|---:|---:|
| AIME24 | 13.33 / 9.90 / 30.00 | 13.33 / 18.33 / 30.00 | 20.00 / 20.83 / 33.00 |
| AIME25 | 10.00 / 6.56 / 23.33 | 6.67 / 17.50 / 30.00 | 16.67 / 18.33 / 33.00 |
| AMC23 | 45.00 / 42.73 / 72.50 | 57.50 / 58.13 / 85.00 | 60.00 / 60.31 / 87.50 |
| MATH500 | 74.20 / 65.74 / 89.60 | 81.80 / 82.20 / 93.00 | 82.20 / 82.30 / 93.80 |
| OlympiadBench | 39.17 / 35.37 / 60.38 | 48.51 / 48.46 / 67.21 | 48.51 / 49.48 / 69.73 |
| **Average** | **36.34 / 32.06 / 55.16** | **41.56 / 44.92 / 61.04** | **45.48 / 46.25 / 63.41** |

Source: [MEL paper](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/sections/Joint%20Training%20Objective.md).

#### Verification experiments (to run)

| Condition | What changes vs MEL | Extra generation budget | Purpose |
|---|---|---:|---|
| A: MEL (re-run) | none | 0 | establish in-house baseline under our compute |
| B: MEL + Extra Rollouts | on all-negative groups, sample +B extra i.i.d. rollouts (B=2) | +B per all-negative group | compute-matched control |
| C: R-MEL (ours) | on all-negative groups, do bounded revision attempts (B=2) to synthesize a y+ with prefix constraint | +B per all-negative group | test whether targeted revision beats extra sampling |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| R-MEL (full) | Revision-only on all-negative groups with LCP filter \(\tau=0.5\) and budget \(B=2\) | Best performance |
| R-MEL w/o prefix constraint | Set \(\tau=0\) (accept any successful revision) | Lower performance / noisier s*; tests importance of prefix alignment |
| R-MEL (B=1) | Only one revision attempt | Lower yield and lower performance if revision budget matters |

### Analysis (Optional)

- **Revision success vs difficulty**: Estimate \(p_{rev\_success}\) as a function of prompt difficulty proxies (e.g., model logprob, answer length). Expect higher benefit on intermediate difficulty prompts where near-miss corrections are plausible.
- **Bifurcation localization quality**: Distribution of inferred s* positions and LCP ratios; expect s* to concentrate around mid-to-late trajectory tokens if revisions are “late fixes.”

---

## Success Criteria

**Criterion 1: Compute-matched performance gain**
- Hypothesis: R-MEL improves Pass@1 over MEL+ExtraRollouts on AIME24/25/MATH500.
- Validation: R-MEL beats MEL+ExtraRollouts on Pass@1 with a bootstrap confidence interval indicating a positive improvement.

**Criterion 2: Increased validated meta-experience yield**
- Hypothesis: R-MEL increases the fraction of prompts that yield at least one replay-validated meta-experience, relative to MEL+ExtraRollouts.
- Validation: Measure (i) \(p_{keep}\) = fraction of prompts with ≥1 validated meta-experience and (ii) mean validated meta-experiences per prompt; both should increase for R-MEL.

**Criterion 3: Failure diagnosis is informative**
- Hypothesis: If R-MEL fails, it fails in a measurable way (e.g., revision success rate is too low or prefix overlap is too small).
- Validation: Track \(p_{rev\_success}\) and LCP ratio distribution; if \(p_{rev\_success}<10\%\) during an early warm-up window, the method is not viable under bounded budget.

---

## Impact Statement

If successful, this method would let RLVR practitioners apply MEL-style process guidance in regimes where the policy is weak and rollouts are mostly incorrect, improving data efficiency without requiring a separate process reward model. It is a small modification to an existing training pipeline (VERL + Math-Verify) that could reduce the number of rollouts required to reach a target Pass@1 level on math reasoning tasks.

---

## References

- [Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models](./references/Internalizing-Meta-Experience-into-Memory-for-Guided-Reinforcement-Learning-in-Large-Language-Models/meta/meta_info.txt) - Huang et al., 2026
- [Don’t Waste Mistakes: Leveraging Negative RL-Groups via Confidence Reweighting](./references/Don’t-Waste-Mistakes-Leveraging-Negative-RL-Groups-via-Confidence-Reweighting/meta/meta_info.txt) - 2025
- [Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning](./references/Save-the-Good-Prefix-Precise-Error-Penalization-via-Process-Supervised-RL-to-Enhance-LLM-Reasoning/meta/meta_info.txt) - 2026
- [Stepwise Guided Policy Optimization: Coloring your Incorrect Reasoning in GRPO](./references/Stepwise-Guided-Policy-Optimization-Coloring-your-Incorrect-Reasoning-in-GRPO/meta/meta_info.txt) - 2025
- [Counterfactual Self-Questioning for Stable Policy Optimization in Language Models](./references/Counterfactual-Self-Questioning-for-Stable-Policy-Optimization-in-Language-Models/meta/meta_info.txt) - 2026
- [EditGRPO: Reinforcement Learning with Post-Rollout Edits for Clinically Accurate Chest X-Ray Report Generation](https://arxiv.org/abs/2509.22812) - Zhang et al., 2025
- [Improving Large Language Models via Fine-grained Reinforcement Learning with Minimum Editing Constraint](https://arxiv.org/abs/2401.06081) - Chen et al., 2024
- [Beyond Uniform Credit: Causal Credit Assignment for Policy Optimization](https://arxiv.org/abs/2602.09331) - Khandoga et al., 2026
- [Online Causal Kalman Filtering for Stable and Effective Policy Optimization](https://arxiv.org/abs/2602.10609) - 2026
- [F-GRPO: Don’t Let Your Policy Learn the Obvious and Forget the Rare](https://arxiv.org/abs/2602.06717) - Plyusov et al., 2026
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - 2023
- [Dapo: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) - Yu et al., 2025
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - Guo et al., 2025
- [RE-GRPO: Leveraging hard negative cases through large language model guided self training](https://www.sciencedirect.com/science/article/abs/pii/S0925231225032151) - Liu & Xiao, 2026
- [Not All Negative Samples Are Equal: LLMs Learn Better from Plausible Reasoning](https://arxiv.org/abs/2602.03516) - 2026
- [VinePPO: Refining Credit Assignment in RL Training of LLMs](https://arxiv.org/abs/2410.01679) - Kazemnejad et al., 2024/2025
- [Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models](https://arxiv.org/abs/2508.10751) - Chen et al., 2025
- [All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning](https://arxiv.org/abs/2503.01067) - Swamy et al., 2025
