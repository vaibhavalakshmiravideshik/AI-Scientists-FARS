# untitled

# Confidence-Bounded Unit-Test Rewards for RLVR Code Training

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated (no human-in-the-loop evaluation)
  - Fit within <=768 NVIDIA A100 (80GB) GPU-hours total
  - No browser / search-API dependencies
  - Prefer open-source models and public code benchmarks, especially EvalPlus benchmarks:
    - **MBPP+** (a Python code generation benchmark derived from MBPP with substantially expanded unit tests)
    - **HumanEval+** (an expanded-unit-test version of HumanEval, a standard Python function synthesis benchmark)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used for code generation and code assistance. A common way to improve a pretrained model for code generation is to fine-tune it using automated feedback, for example by executing unit tests on generated code and using the test outcomes as learning signals.

Reinforcement learning with verifiable rewards (RLVR) is a form of reinforcement learning (RL) that optimizes a policy model using rewards computed by programmatic verifiers (such as unit tests, compilers, or proof checkers), rather than a learned reward model. In the code domain, unit tests are an attractive verifier because they are executable and produce an objective pass/fail signal.

A practical complication is that unit-test verification is often CPU-bound. During RLVR training, it may be infeasible to run the full test suite for every generated solution; instead, practitioners may execute only m sampled tests per rollout (where a rollout is one generated solution plus its test execution). In addition, even large test suites can have coverage gaps. These factors create a systematic failure mode: incorrect programs can receive high estimated reward because they pass a small or incomplete subset of tests.

### The Problem

Most unit-test-based RLVR pipelines for code treat unit tests as either:
- a binary verifier (reward 1 if all executed tests pass, else 0), or
- a point-estimate pass rate (reward \(\hat{p} = n_{\text{pass}}/m\), where \(n_{\text{pass}}\) is the number of passed tests among \(m\) executed tests).

When \(m\) is small, \(\hat{p}\) is a high-variance estimate of the underlying probability that a solution would pass a more comprehensive test suite. RLVR can then over-optimize this noisy reward signal, increasing training-test reward without improving (and in some cases degrading) functional correctness under stronger evaluation. Recent theoretical work on RLVR with imperfect verifiers suggests that if verifier false positives dominate, training can enter a regime where optimizing reward reduces true task performance (sometimes described as an anti-learning regime in the RLVR literature).

Prior work addresses related aspects of this problem, but leaves open the question of reward estimation under a fixed test-execution budget:

- **[RLTF](https://arxiv.org/abs/2307.04349)** (reinforcement learning from unit test feedback): uses unit-test outcomes as RL rewards, typically as binary rewards or pass-rate point estimates.
- **[CodeRL](https://arxiv.org/abs/2207.01780)**: uses execution feedback (e.g., unit tests) to refine code generation via RL-style updates, again using point-estimate rewards.
- **[Dynamic Scaling of Unit Tests for Code Reward Modeling (CodeRM)](./references/CodeRM-Dynamic-Scaling-Unit-Tests/meta/meta_info.txt)**: improves selection/reward quality by executing more tests or allocating tests adaptively, but increases verifier compute.
- **[HardTests](./references/HardTests/meta/meta_info.txt), [UTRL](./references/UTRL/meta/meta_info.txt), [CVeDRL](./references/CVeDRL/meta/meta_info.txt), [CURE](./references/CURE/meta/meta_info.txt)**: improve test quantity/quality or train test generators/verifiers; this is valuable but orthogonal to conservative reward estimation from limited tests.
- **[Ensuring Functional Correctness of Large Code Models with Selective Generation](./references/Selective-Code-Generation-FDR-CE/meta/meta_info.txt)**: uses binomial tail bounds mainly at inference time to abstain or filter solutions; it does not study training-time RL learning dynamics.
- **[Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers](./references/Noisy-RLVR-Imperfect-Verifiers/meta/meta_info.txt)** and **[Rate or Fate? RL-eps-R](./references/Rate-or-Fate-RLV-eps-R/meta/meta_info.txt)**: motivate explicitly accounting for verifier noise/imperfection, but do not instantiate a unit-test-sampling-uncertainty-aware reward for code.

In practice, this creates a trade-off for RLVR training with unit tests:
- Executing few tests per rollout reduces verifier cost but increases reward uncertainty.
- Executing more tests reduces reward uncertainty but can be prohibitively expensive during RL training.

This proposal targets settings where verifier compute is a tight constraint (e.g., smaller open-source coder models and CPU-constrained training pipelines).

### Key Insight and Hypothesis

**Key idea.** Instead of rewarding the observed pass rate \(\hat{p}\) on \(m\) sampled tests, reward a lower confidence bound (LCB) on the solution's true pass probability. Concretely, if a solution passes 3/3 sampled tests, a conservative estimator should reward it less than a solution that passes 27/30 sampled tests, even though both have \(\hat{p}=1\).

**Hypothesis.** Using a Beta-posterior lower confidence bound (LCB) of pass probability as the RL reward (instead of binary or pass-rate point estimates) will:

1. Improve alignment between the training reward and full-test correctness under a fixed \(m\) (Phase 0).
2. Reduce overfitting to the training test pool, measured by a smaller train-holdout test gap (lower is better).
3. Improve functional correctness under comprehensive evaluation at matched verifier compute, measured by Pass@k on full tests (higher is better).

**Why this could fail.** (i) Group-based policy optimization methods (such as GRPO) normalize rewards within each prompt's sample group; LCB may not change the within-group ranking of samples enough to affect gradients. (ii) Over-penalizing uncertainty could reduce reward dynamic range and slow learning.

**Decision rule (verification).** We consider LCB(m) successful if it improves reward-oracle alignment in Phase 0 versus pass-rate(m), and in Phase 2 yields a smaller train-holdout test gap than pass-rate(m) while matching or exceeding pass-rate(2m) on full-test Pass@{1,5,10} at matched verifier compute. Otherwise, we refute.

---

## Proposed Approach

### Overview

Replace the standard unit-test reward \(\hat{r} = n_{\text{pass}}/m\) (or binary all-pass) with a confidence-bounded reward \(r_{\text{LCB}}\) that explicitly accounts for finite-sample uncertainty from executing only \(m\) tests.

### Method Details

**Setting.** For each prompt \(x\), sample a group of \(G\) code completions \(\{y_i\}\) from the current policy (default group size \(G=8\)). For each completion \(y_i\), execute \(m\) unit tests (default \(m=5\)) drawn from a fixed per-problem TrainTestsPool (a subset of each task's unit tests reserved for computing training-time rewards), and record \(n_{\text{pass}}(i)\) and \(n_{\text{fail}}(i)\).

**LCB reward.** Model each test outcome as a Bernoulli trial and use a Beta posterior over the pass probability \(p\):

- Prior: \(p \sim \text{Beta}(\alpha_0, \beta_0)\) with \(\alpha_0=\beta_0=0.5\) (Jeffreys prior; a standard noninformative choice).
- Posterior: \(p \mid \text{data} \sim \text{Beta}(\alpha_0 + n_{\text{pass}}, \beta_0 + n_{\text{fail}})\).
- Reward (lower \(1-\delta\) confidence bound):
  - \(r_{\text{LCB}}(y) = \text{BetaPPF}(\delta, \alpha_0 + n_{\text{pass}}, \beta_0 + n_{\text{fail}})\), where BetaPPF is the Beta distribution quantile function (inverse CDF) and \(\delta=0.05\) yields a 95% LCB.

This yields a conservative reward in \([0,1]\) that increases with both pass rate and evidence (larger \(m\)).

**Training algorithm.** Use group relative policy optimization (GRPO), a group-based policy gradient RL method that computes per-sample advantages relative to the mean (and typically the standard deviation) of rewards within each prompt's sample group. Use an open-source GRPO implementation (e.g., TRL's GRPOTrainer: https://github.com/huggingface/trl, or OpenRLHF: https://github.com/OpenRLHF/OpenRLHF) and keep the RL algorithm fixed across conditions; only the reward function changes. Fine-tune with parameter-efficient Low-Rank Adaptation (LoRA) in bfloat16 (bf16) precision.

**Monitoring / guardrails.**
- Log per-prompt reward spread (standard deviation and max-min) to detect whether LCB collapses reward dynamic range relative to pass-rate.
- Track reward-oracle correlation over training: correlation between the training-time reward computed from TrainTestsPool samples and an oracle correctness label computed from HoldoutTestsPool or the full EvalPlus tests.
- Add a Phase-0 diagnostic for the GRPO normalization concern: measure whether LCB changes the within-group ranking of samples relative to pass-rate when \(m\) is small.

### Key Innovations

- **Training-time confidence bounds on unit-test rewards**: apply binomial/Beta lower confidence bounds as a reward function inside policy-gradient RL, rather than as an inference-time filter.
- **Matched verifier-compute evaluation**: compare LCB(m) against pass-rate(2m) while controlling the expected number of executed unit tests.
- **Explicit test-suite overfitting metric for RLVR-with-tests**: evaluate generalization from TrainTestsPool to HoldoutTestsPool as a primary outcome.

---

## Related Work

### Field Overview

RLVR optimizes language models using programmatic verifiers and is widely used for tasks with automated correctness checks. Recent work has popularized group-based RL objectives such as GRPO for verifiable domains, demonstrating strong performance improvements when rewards are reliable (e.g., [DeepSeekMath](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948)).

For code generation, unit tests are the most common verifier, but both test quantity and test quality determine signal reliability. A growing 2025-2026 literature improves verifiers by generating more tests, generating higher-quality tests, or co-training test generators and verifiers. In parallel, statistical risk-control methods use concentration bounds for inference-time selection or abstention. This proposal focuses on reward estimation during training under a fixed test-execution budget.

### Related Papers

- **[DeepSeekMath](https://arxiv.org/abs/2402.03300)**: Introduces GRPO-style RLVR recipes that significantly improve performance on verifiable reasoning tasks.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Demonstrates large-scale RLVR training with verifiable rewards can induce strong reasoning behaviors.
- **[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Motivates verifier-centric training signals via process supervision.
- **[Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers](./references/Noisy-RLVR-Imperfect-Verifiers/meta/meta_info.txt)**: Derives policy-gradient corrections under asymmetric verifier noise.
- **[Rate or Fate? RL-eps-R](./references/Rate-or-Fate-RLV-eps-R/meta/meta_info.txt)**: Analyzes phase transitions in RLVR as a function of verifier informativeness.
- **[On GRPO Collapse in Search-R1: The Lazy Likelihood-Displacement Death Spiral](./references/GRPO-Collapse-LLD/meta/meta_info.txt)**: Studies stability and collapse modes in GRPO-like training.
- **[ConfClip](https://arxiv.org/abs/2509.17730)**: Uses model-confidence-based clipping/weighting to stabilize verifiable rewards; differs from test-sampling uncertainty.
- **[Don't Waste Mistakes (LENS)](https://arxiv.org/abs/2510.08696)**: Uses confidence-weighted penalties to extract learning signal from all-negative groups; uses model uncertainty rather than verifier uncertainty.
- **[No Prompt Left Behind: Entropy-Guided Advantage Shaping for Zero-Variance Prompts](https://arxiv.org/abs/2509.21880)**: Improves GRPO training by extracting learning signal from low-variance prompt groups.
- **[Rethinking Reward Shaping in Group-Based Reinforcement Learning](https://arxiv.org/abs/2601.23058)**: Proposes bounded relative reward shaping for stable group-based RL; orthogonal to verifier sampling uncertainty.
- **[RLTF](https://arxiv.org/abs/2307.04349)**: Uses unit-test outcomes as rewards for RL code generation, typically with binary or pass-rate rewards.
- **[CodeRL](https://arxiv.org/abs/2207.01780)**: Uses execution feedback as a signal for RL-style improvement of code generation.
- **[Dynamic Scaling of Unit Tests for Code Reward Modeling (CodeRM)](./references/CodeRM-Dynamic-Scaling-Unit-Tests/meta/meta_info.txt)**: Shows executing more tests improves selection/reward quality, motivating compute-aware alternatives.
- **[HardTests](./references/HardTests/meta/meta_info.txt)**: Synthesizes higher-quality unit tests and shows RL performance depends on verifier quality.
- **[UTRL](./references/UTRL/meta/meta_info.txt)**: Trains a unit test generator via adversarial RL to improve test discrimination.
- **[CURE](./references/CURE/meta/meta_info.txt)**: Co-evolves a coder and unit tester via RL with a test-centric objective.
- **[CVeDRL](./references/CVeDRL/meta/meta_info.txt)**: Trains efficient code verifiers and analyzes confidence/robustness.
- **[Learning a Pessimistic Reward Model in RLHF](./references/Pessimistic-RM-RLHF/meta/meta_info.txt)**: Uses pessimism under uncertainty to reduce over-optimization of uncertain rewards in reinforcement learning from human feedback (RLHF).
- **[Ensuring Functional Correctness of Large Code Models with Selective Generation](./references/Selective-Code-Generation-FDR-CE/meta/meta_info.txt)**: Uses concentration bounds for inference-time abstention/selection based on test outcomes.
- **[Posterior-GRPO](https://arxiv.org/abs/2508.05170)**: Shapes training reward based on intermediate reasoning conditional on outcome correctness.
- **[HumanEval](https://arxiv.org/abs/2107.03374)**: Standard Python function synthesis benchmark evaluated by unit tests.
- **[MBPP](https://arxiv.org/abs/2108.07732)**: A Python programming benchmark with many entry-level problems.
- **[EvalPlus](https://github.com/evalplus/evalplus)**: Provides expanded unit tests (HumanEval+/MBPP+) for more rigorous code evaluation.
- **[Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons](https://arxiv.org/abs/2301.11270)**: Provides a theoretical framing for pessimism via lower confidence bounds.
- **[Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779)**: Canonical pessimism-under-uncertainty method in offline RL.
- **[Stabilizing Policy Gradient Methods via Reward Profiling](https://arxiv.org/abs/2511.16629)**: Explores stability improvements in policy-gradient methods using conservative update rules.
- **[Conformal Constrained Policy Optimization for Cost-Effective LLM Agents](https://arxiv.org/abs/2511.11828)**: Uses conformal prediction to control risk/cost in policy optimization.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| RLVR algorithms (group-based) | GRPO-style RL without a learned value model; group-based advantage normalization | DeepSeekMath, DeepSeek-R1, GRPO variants | Verifiable tasks (math/code) | Can under-utilize low-variance prompt groups; can be unstable |
| RLVR stability / reward shaping | Modify rewards/advantages for stability and utilization | ConfClip, LENS, entropy-guided advantage shaping, bounded relative rewards | Verifiable reasoning and some code | Typically focuses on model confidence or relative ranking, not verifier sampling uncertainty |
| Noisy-verifier RLVR theory/corrections | Model verifier errors and correct gradients / characterize failure regimes | Noisy RLVR correction, Rate-or-Fate (RL-eps-R) | Verifiable tasks | Often assumes simplified noise models; limited code-domain instantiations |
| Better/more code verifiers | Improve test quantity/quality or learn test generators/verifiers | CodeRM, HardTests, UTRL, CVeDRL, CURE | HumanEval+/MBPP+/other code benchmarks | Adds infrastructure/compute; does not address reward estimation under partial test execution |
| Inference-time risk control | Statistical bounds for selection/abstention at inference time | Selective Code Generation (FDR-CE), conformal methods | Test/fuzz suites | Usually inference-time; does not shape training gradients |
| Pessimism under uncertainty | Use conservative estimates to avoid over-optimization | CQL, pessimistic RLHF theory, pessimistic reward modeling | Offline RL / RLHF | Not specialized to unit-test verifiers or finite-sample test outcomes |

### Closest Prior Work

1) **Ensuring Functional Correctness of Large Code Models with Selective Generation** ([Paper](./references/Selective-Code-Generation-FDR-CE/meta/meta_info.txt))
- What it does: Uses binomial tail bounds on test outcomes to select or abstain at inference time.
- Limitation for our question: It does not study or modify training-time RL learning dynamics.
- Why different: We use confidence bounds as a reward function inside GRPO, changing policy gradients under a fixed \(m\).

2) **Dynamic Scaling of Unit Tests for Code Reward Modeling (CodeRM)** ([Paper](./references/CodeRM-Dynamic-Scaling-Unit-Tests/meta/meta_info.txt))
- What it does: Improves selection/reward quality by scaling or allocating test executions.
- Limitation: Requires more verifier compute.
- Why different: We keep the test budget fixed and change the reward estimator.

3) **HardTests** ([Paper](./references/HardTests/meta/meta_info.txt))
- What it does: Generates stronger test suites and shows RL benefits depend on verifier quality.
- Limitation: Focuses on improving the verifier rather than conservative estimation from limited tests.
- Why different: Our method is complementary and targets a different lever (reward estimation).

4) **Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers** ([Paper](./references/Noisy-RLVR-Imperfect-Verifiers/meta/meta_info.txt))
- What it does: Corrects policy gradients under assumed verifier noise rates.
- Limitation: Does not model finite-sample uncertainty from executing only \(m\) tests.
- Why different: LCB is an uncertainty-aware estimator that requires no noise-rate estimation.

5) **ConfClip / Don't Waste Mistakes (LENS)** ([arXiv:2509.17730](https://arxiv.org/abs/2509.17730), [arXiv:2510.08696](https://arxiv.org/abs/2510.08696))
- What they do: Use model-confidence-based reward shaping to stabilize group-based RL.
- Limitation: Their uncertainty source is the model's predictive confidence, not verifier finite-sample evidence.
- Why different: We target uncertainty due to partial unit-test execution.

### Comparison Table

| Related work | What it does (1 sentence) | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Selective Code Generation (FDR-CE) | Inference-time abstention/selection using binomial bounds over test outcomes | Does not shape RL training | Use LCB as a training reward | Phase 0: LCB improves reward-oracle alignment; Phase 2: improves full-test Pass@k |
| CodeRM | Allocates more unit tests to improve reward/selection quality | Requires more verifier compute | Fix \(m\), change estimator | Matched-compute comparison vs pass-rate(2m) |
| HardTests | Improves test suites and studies verifier quality | Adds verifier infrastructure | Keep tests fixed, change reward | Useful when improving tests is expensive or unavailable |
| Noisy-RLVR correction | Gradient correction under assumed noise rates | Needs noise-rate assumptions/estimation; not tailored to finite \(m\) | Closed-form conservative estimator | No noise-rate estimation; directly uses observed test evidence |
| ConfClip / LENS | Model-confidence-based reward shaping for stability | Not about unit-test sampling uncertainty | Verifier-uncertainty-based reward shaping | Targets a different failure mode (partial testing) |
| Pessimistic reward modeling in RLHF | Trains a conservative reward model under uncertainty | Requires a learned reward model | Closed-form LCB from verifier outcomes | No learned reward model; uses exact test outcomes |
| Bounded relative rewards (RLRR) | Stabilizes group-based RL via bounded relative rewards | Not about verifier uncertainty | Keep GRPO, change reward | Orthogonal; can be combined later |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|------|
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct | Small enough to run multiple RL conditions within the GPU-hour budget |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| MBPP+ (EvalPlus) | RL training prompts + unit tests | ~400 tasks (exact count depends on EvalPlus version) | https://huggingface.co/datasets/evalplus/mbppplus | Apache-2.0 (per dataset card; verify upstream licenses if needed) |

**Other Resources (if applicable):**
- Evaluation harness: https://github.com/evalplus/evalplus
- GRPO training libraries: https://github.com/huggingface/trl, https://github.com/OpenRLHF/OpenRLHF

**Resource Estimate**:
- **Compute budget (GPU-hours)**:
  - Phase 0 (no training; generation + scoring only): <=50 GPU-hours
  - Phase 1/2 training: 4 conditions x ~96 GPU-hours each (4xA100 for ~24h) ~= 384 GPU-hours
  - Total target: <=500 GPU-hours, leaving margin for 1-2 ablations
- **GPU memory**: <=40GB for 1.5B with LoRA + bf16; fits on A100-80GB
- **CPU budget**: unit-test execution can dominate wall-clock time; enforce strict timeouts (e.g., 1s per test, 10s per solution cap) and parallelize across CPU workers
- **API usage**: none required

**Infrastructure constraints** (proposals requiring these are infeasible):
- Search engine APIs (Google, Bing) - NOT available
- Web browsers / desktop GUIs / mobile environments - NOT available
- Complex game engines or heavy simulation environments - NOT available

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| MBPP+ (EvalPlus) | Python function synthesis benchmark with expanded unit tests for functional correctness evaluation | Pass@{1,5,10}, train-holdout test gap, reward-oracle correlation | Use MBPP+ tasks for RL training; evaluation uses held-out tests and full tests on the same tasks | https://huggingface.co/datasets/evalplus/mbppplus | https://github.com/evalplus/evalplus |
| HumanEval+ (EvalPlus) | Python function synthesis benchmark with 164 problems and expanded unit tests; used as a standard test-only generalization benchmark | Pass@{1,5,10} | test-only transfer (no training on HumanEval+) | https://huggingface.co/datasets/evalplus/humanevalplus | https://github.com/evalplus/evalplus |

**Metric definitions (first use):**
- **Pass@k**: the probability that at least one of \(k\) sampled solutions is correct (higher is better).
- **Train-holdout test gap**: difference between performance measured on TrainTestsPool (the unit-test subset used to compute training-time rewards) vs HoldoutTestsPool (a disjoint unit-test subset used only for evaluation; lower is better). In this proposal, we compute it as the gap between pass rates when executing all tests in each pool for the same set of generated solutions.
- **Reward-oracle correlation**: correlation between the training-time reward (computed from \(m\) sampled TrainTestsPool tests) and an oracle correctness label computed from HoldoutTestsPool or the full EvalPlus tests (higher is better). We report Spearman rank correlation (correlation between rankings).
- **Top-k precision** (used in Phase 0): among the \(k\) solutions with highest estimated reward, the fraction that are correct under HoldoutTestsPool or the full EvalPlus tests (higher is better).

**Key protocol detail (overfitting measurement):**
- For each MBPP+ task, split its available unit tests into TrainTestsPool and HoldoutTestsPool with a fixed random seed.
- RL reward uses only TrainTestsPool, sampling \(m\) tests per rollout.
- Report (i) performance on TrainTestsPool (all train tests), (ii) performance on HoldoutTestsPool (all holdout tests), and (iii) performance on the full EvalPlus tests.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Pass@1 (full tests; higher is better) | Train-holdout gap (lower is better) | Source | Notes |
|--------|------------|-----------|--------------------------------------|-------------------------------------|--------|-------|
| Binary (all executed tests pass) | Qwen2.5-Coder-1.5B | MBPP+ | **TBD** | **TBD** | - | Needs re-run |
| Pass-rate(m) | Qwen2.5-Coder-1.5B | MBPP+ | **TBD** | **TBD** | - | Needs re-run |
| Pass-rate(2m) | Qwen2.5-Coder-1.5B | MBPP+ | **TBD** | **TBD** | - | Needs re-run (more verifier compute per rollout) |
| Simple pessimistic baseline: pass-rate(m) - \(\lambda/\sqrt{m}\) | Qwen2.5-Coder-1.5B | MBPP+ | **TBD** | **TBD** | - | Needs re-run |
| **LCB(m) (Ours)** | Qwen2.5-Coder-1.5B | MBPP+ | **TBD** | **TBD** | - | To be verified |

Also report transfer results on HumanEval+ full tests (Pass@{1,5,10}; higher is better).

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| LCB confidence level | \(\delta \in \{0.01, 0.05, 0.10, 0.20\}\) | Moderate \(\delta\) best; larger \(\delta\) approaches pass-rate |
| Prior sensitivity | Beta(0.5,0.5) vs Beta(1,1) vs Beta(2,2) | Similar when \(m\) is not extremely small; large sensitivity indicates fragility |
| Reward dynamic-range mitigation | Rank-normalize rewards within group before GRPO normalization | Only needed if LCB reward spread collapses |

### Analysis (Optional)

- Phase 0: Compare pass-rate vs LCB as predictors of full-test correctness at fixed \(m\), using reward-oracle correlation and top-k precision.
- Training dynamics: track reward-oracle correlation over training steps to diagnose over-optimization of the training reward.

---

## Success Criteria

**Criterion 1: Better reward-oracle alignment at fixed \(m\) (Phase 0 gate)**
- Hypothesis: LCB(m) is a better predictor of oracle correctness (full or held-out tests) than pass-rate(m), and is competitive with pass-rate(2m).
- Validation: Higher Spearman rank correlation and/or better top-k precision for LCB(m) than pass-rate(m). If not, refute.

**Criterion 2: Reduced overfitting to training tests (Phase 2)**
- Hypothesis: Training with LCB(m) reduces the train-holdout test gap compared to pass-rate(m) and binary rewards.
- Validation: Directional reduction in train-holdout gap without sacrificing full-test Pass@k.

**Criterion 3: Benefit beyond simply executing more tests**
- Hypothesis: At matched verifier compute, LCB(m) matches or exceeds pass-rate(2m).
- Validation: If LCB(m) underperforms pass-rate(2m) under matched verifier compute, refute.

---

## Impact Statement

If confidence-bounded unit-test rewards improve reward-oracle alignment and reduce test-suite overfitting under fixed test-execution budgets, practitioners can run unit-test-driven RLVR with fewer executed tests per rollout while retaining (or improving) functional correctness under comprehensive evaluation. This would reduce the need to increase verifier compute as the primary way to mitigate reward uncertainty when CPU test execution is a bottleneck.

---

## References

- [Dynamic Scaling of Unit Tests for Code Reward Modeling](./references/CodeRM-Dynamic-Scaling-Unit-Tests/meta/meta_info.txt) - Ma et al., 2025
- [HardTests: Synthesizing High-Quality Test Cases for LLM Coding](./references/HardTests/meta/meta_info.txt) - He et al., 2025
- [Learning to Generate Unit Test via Adversarial Reinforcement Learning (UTRL)](./references/UTRL/meta/meta_info.txt) - Lee et al., 2025
- [Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning (CURE)](./references/CURE/meta/meta_info.txt) - Wang et al., 2025
- [CVeDRL: An Efficient Code Verifier via Difficulty-aware Reinforcement Learning](./references/CVeDRL/meta/meta_info.txt) - Shi et al., 2026
- [Ensuring Functional Correctness of Large Code Models with Selective Generation](./references/Selective-Code-Generation-FDR-CE/meta/meta_info.txt) - 2026
- [Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers](./references/Noisy-RLVR-Imperfect-Verifiers/meta/meta_info.txt) - Cai et al., 2025
- [Rate or Fate? RL-eps-R: Reinforcement Learning with Verifiable Noisy Rewards](./references/Rate-or-Fate-RLV-eps-R/meta/meta_info.txt) - Rad et al., 2026
- [On GRPO Collapse in Search-R1: The Lazy Likelihood-Displacement Death Spiral](./references/GRPO-Collapse-LLD/meta/meta_info.txt) - 2026
- [ConfClip: Confidence-Weighted and Clipped Reward for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2509.17730) - 2025
- [Don't Waste Mistakes: Leveraging Negative RL-Groups via Confidence Reweighting (LENS)](https://arxiv.org/abs/2510.08696) - 2025
- [No Prompt Left Behind: Entropy-Guided Advantage Shaping for Zero-Variance Prompts](https://arxiv.org/abs/2509.21880) - 2025
- [Rethinking Reward Shaping in Group-Based Reinforcement Learning](https://arxiv.org/abs/2601.23058) - 2026
- [Posterior-GRPO: Rewarding Reasoning Processes in Code Generation](https://arxiv.org/abs/2508.05170) - 2025
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) - DeepSeek-AI et al., 2024
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - DeepSeek-AI et al., 2025
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [RLTF: Reinforcement Learning from Unit Test Feedback](https://arxiv.org/abs/2307.04349) - 2023
- [CodeRL: Mastering Code Generation through Pretrained Models and Execution Feedback](https://arxiv.org/abs/2207.01780) - 2022
- [HumanEval: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) - Chen et al., 2021
- [MBPP: Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732) - Austin et al., 2021
- [EvalPlus repository](https://github.com/evalplus/evalplus) - EvalPlus authors, 2023-2025
- [Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons](https://arxiv.org/abs/2301.11270) - 2023
- [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) - Kumar et al., 2020
- [Learning a Pessimistic Reward Model in RLHF](./references/Pessimistic-RM-RLHF/meta/meta_info.txt) - Xu et al., 2025
- [Stabilizing Policy Gradient Methods via Reward Profiling](https://arxiv.org/abs/2511.16629) - 2025
- [Conformal Constrained Policy Optimization for Cost-Effective LLM Agents](https://arxiv.org/abs/2511.11828) - 2025
