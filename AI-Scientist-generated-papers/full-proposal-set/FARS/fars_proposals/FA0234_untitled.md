# untitled

# Local-Time AdamW for Stability-Gap Reduction in Continual Learning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Continual learning (CL) studies how a model can be trained on a **sequence of tasks or data distributions** while retaining performance on earlier tasks. In many practical CL pipelines (including continual pre-training and long-running fine-tuning), training is **state-carryover**: we resume from a previous checkpoint and continue optimizing, often **including the optimizer state** (e.g., AdamW first/second moments) rather than reinitializing the optimizer at each task boundary.

A recurring pathology in sequential training is the **stability gap**: immediately after switching to a new task, performance on previous tasks can drop sharply before partially recovering later. This temporary degradation matters because the model may be evaluated or deployed while learning, and worst-case performance (not only final accuracy) can be safety- and user-facing. Recent work indicates that the stability gap is (i) common across benchmarks and training regimes and (ii) sensitive to architectural and optimization choices.

### The Problem

Two recent findings sharpen the problem and motivate an optimizer-focused intervention:

- **Classification-head effects**: Lapacz et al. show that in class-incremental vision CL, a large fraction of the stability gap can be attributed to the linear classification head, and that using a nearest-mean classifier (NMC) at inference substantially reduces the gap.
- **Optimizer effects**: An empirical optimizer study on Rotated-MNIST stability-gap settings finds that optimizer choice and momentum strongly affect the depth and duration of the gap; in that simplified setup, RMSprop reduces the gap relative to Adam.

These results suggest that even if the head is a dominant contributor in some settings, **optimization dynamics remain a first-class lever** for reducing post-switch instability.

However, most CL implementations still treat AdamW as a black box and either:

1) **Carry all optimizer state across tasks** (common in checkpoint-resume pipelines), including Adam’s internal timestep counter used for bias correction, or
2) **Fully reset the optimizer state** at each task boundary (common in “train-per-task” CL code), which discards potentially useful preconditioning information.

Neither choice isolates *which* part of optimizer state is responsible for post-switch instability.

**Example (what we mean by stability gap):** after switching from task 1 to task 2, accuracy on task 1 might drop from ~90% to ~60% within a small number of gradient steps, then recover partially (e.g., to ~75%) after more training on task 2.

### Key Insight and Hypothesis

Adam/AdamW uses bias-corrected moments:

\[\Delta\theta_t = -\alpha \; \frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}\]

so the effective multiplicative factor on the raw ratio \(m_t/\sqrt{v_t}\) is

\[s(t) = \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}.\]

For default \(\beta_1=0.9,\beta_2=0.999\), \(s(1)\approx 0.316\) and \(s(t)\to 1\) as \(t\) grows, meaning that the bias-correction schedule implicitly changes the early-step update scale.

Recent work in nonstationary reinforcement learning (Ellis et al., “Adam on Local Time”) shows that when gradient magnitudes change abruptly, using a global timestep can yield overly large updates, and that resetting **only** the Adam timestep (while keeping moments) bounds update sizes and improves performance.

**Hypothesis:** In supervised continual learning with optimizer-state carryover, resetting **only** AdamW’s bias-correction timestep counter at each task boundary (keeping moment buffers \(m,v\)) reduces post-switch update spikes and thus reduces the stability gap (smaller maximum accuracy drop and higher minimum accuracy after the switch). This effect should largely disappear if Adam bias correction is disabled.

This hypothesis could be wrong if (i) task switches do not reliably induce the gradient-scale changes needed for the effect, (ii) the stability gap is dominated by classifier-head dynamics that are not sensitive to this optimizer schedule, or (iii) resetting the timestep is equivalent to a generic warmup effect that is better implemented directly via learning-rate warmup.

---

## Proposed Approach

### Overview

We propose **Local-Time AdamW (LT-AdamW)** for task-sequential supervised continual learning:

- At each task boundary, keep AdamW moment buffers \(m,v\) (state carryover)
- Reset only the timestep counter used in bias correction (local time): \(t \leftarrow 0\)

This is a minimal optimizer-state intervention intended to reduce *transient* post-switch instability without discarding the preconditioning information contained in \(v\).

### Method Details

**LT-AdamW update rule**: Use standard AdamW within tasks. At each task boundary (task \(\tau\to\tau+1\)):

- for each parameter \(\theta\) with AdamW state \(\{m,v,t\}\), set \(t\leftarrow 0\)
- keep \(m,v\) unchanged

**Implementation detail (PyTorch-style):** for each parameter `p`:

- `optimizer.state[p]["step"] = 0`
- do not modify `exp_avg` or `exp_avg_sq`

**Mechanism control (bias-correction-off):** Add a control condition that disables bias correction (i.e., use \(m_t\) and \(v_t\) without dividing by \((1-\beta^t)\) factors). If LT-AdamW’s gains are specifically due to bias correction behavior, they should be strongly attenuated under this control.

### Key Innovations

1. **Optimizer-state factorization at task boundaries**: isolates the effect of Adam’s timestep/bias-correction state separately from moment reuse.
2. **Mechanism-discriminating control**: disabling bias correction tests whether improvements are genuinely attributable to timestep handling.
3. **Stability-gap-centric evaluation**: targets worst-case post-switch accuracy (stability gap and minimum accuracy), not only final average accuracy.

---

## Related Work

### Field Overview

Continual learning methods are commonly grouped into replay-based methods (store past samples), regularization-based methods (constrain parameter drift), and architectural methods (parameter isolation or dynamic expansion). A growing literature emphasizes that forgetting is not only about final performance degradation, but also about **learning dynamics** immediately after task transitions (the stability gap), which can be sensitive to training regime and optimizer behavior.

Separately, modern optimization for deep learning relies heavily on AdamW-like adaptive optimizers with momentum and bias correction, yet CL often treats optimizer state as an implementation detail. Recent nonstationary optimization work in reinforcement learning shows that the Adam timestep can be a failure point under abrupt objective changes, motivating a direct test of similar effects under CL task switches.

### Related Papers

(Each entry: one sentence summary + relevance.)

- **[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)**: introduces Adam and bias correction, which is the mechanism targeted by LT-AdamW.
- **[Decoupled Weight Decay Regularization (AdamW)](https://openreview.net/forum?id=Bkg6RiCqY7)**: introduces AdamW, the default optimizer in many modern training pipelines.
- **[Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps](../../../papers/paper_summaries/Adam%20on%20Local%20Time%20Addressing%20Nonstationarity%20in%20RL%20with%20Relative%20Adam%20Timesteps.md)**: proposes timestep resets (Adam-Rel) under RL nonstationarity and provides theory motivating LT-AdamW.
- **[Resetting the Optimizer in Deep RL: An Empirical Study](https://arxiv.org/abs/2306.17833)**: studies full optimizer resets under RL objective changes, contrasting with timestep-only reset.
- **[Correcting Momentum in Temporal Difference Learning](https://arxiv.org/abs/2102.07803)**: motivates momentum/optimizer-state contamination under nonstationarity.
- **[Exploring the Stability Gap in Continual Learning: The Role of the Classification Head](../../../papers/paper_summaries/Exploring%20the%20Stability%20Gap%20in%20Continual%20Learning%20The%20Role%20of%20the%20Classification%20Head.md)**: attributes stability gap largely to the linear head and motivates stability-gap-focused metrics.
- **[Continual evaluation for lifelong learning: Identifying the stability gap](https://arxiv.org/abs/2205.13452)**: introduces the stability gap phenomenon under fine-grained continual evaluation.
- **[Two Complementary Perspectives to Continual Learning: Ask Not Only What to Optimize, But Also How](https://arxiv.org/abs/2311.04898)**: argues that reducing the stability gap requires improving optimization trajectories, not only approximating the joint objective.
- **[iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)**: a classic CL method using nearest-mean classification, relevant to head-level stability fixes.
- **[Dark Experience Replay (DER++)](https://arxiv.org/abs/2004.07211)**: a strong replay baseline often used in class-incremental CL.
- **[Gradient Episodic Memory (GEM)](https://arxiv.org/abs/1706.08840)**: constrains gradients to avoid interference, representing replay/constraint families.
- **[Elastic Weight Consolidation (EWC)](https://arxiv.org/abs/1612.00796)**: a foundational regularization-based CL method.
- **[Synaptic Intelligence](https://arxiv.org/abs/1703.04200)**: online parameter-importance regularization for CL.
- **[Learning without Forgetting](https://arxiv.org/abs/1606.09282)**: distillation-based CL baseline.
- **[Understanding the Role of Training Regimes in Continual Learning](https://arxiv.org/abs/2006.06958)**: shows learning rate and regime choices can dominate forgetting behavior, motivating careful control of LR schedules.
- **[On Warm-Starting Neural Network Training](https://arxiv.org/abs/1910.08475)**: studies warm-start optimization pathologies and proposes shrink-and-perturb, conceptually related to reset-like interventions.
- **[Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)**: proposes cosine learning-rate restarts, a classical “restart” technique related to local-time schedules.
- **[Continual Learning of Numerous Tasks from Long-tail Distributions](../../../papers/paper_summaries/Continual%20Learning%20of%20Numerous%20Tasks%20from%20Long-tail%20Distributions.md)**: introduces Continual Adam/AdamW (second-moment reuse + early-step multiplier), the closest CL optimizer-state baseline.
- **[Continual Backprop](https://arxiv.org/abs/2308.01228)**: resets low-utility neurons to maintain plasticity, representing reset-based plasticity methods.
- **[ReDO: Reinitializing dormant neurons to preserve plasticity](https://arxiv.org/abs/2302.12902)**: selective neuron reset method for plasticity maintenance.
- **[Self-Normalized Resets for Plasticity in Continual Learning](../../../papers/paper_summaries/Self-Normalized%20Resets%20for%20Plasticity%20in%20Continual%20Learning.md)**: statistical neuron reset method targeting plasticity loss.
- **[Learning Continually by Spectral Regularization](../../../papers/paper_summaries/LEARNING%20CONTINUALLY%20BY%20SPECTRAL%20REGULARIZATION.md)**: constrains spectral properties to preserve trainability; relates to “regenerative” training dynamics.
- **[Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models](../../../papers/paper_summaries/Revisiting%20Replay%20and%20Gradient%20Alignment%20for%20Continual%20Pre-Training%20of%20Large%20Language%20Models.md)**: provides evidence that continual pre-training has CL-like forgetting/stability issues and uses checkpoint-resume style training.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Head-level stability fixes | Modify classification head / inference rule to reduce recency bias and instability | Lapacz et al. (NMC), iCaRL | Split CIFAR-100, ImageNet-100 | May not address optimizer-driven instability; may require memory/prototypes |
| Replay / constraints | Store or constrain gradients using past examples | ER, DER++, GEM | Standard CL suites | Memory/computation overhead |
| Regularization | Penalize drift of important weights | EWC, SI, LwF | Standard CL suites | Often sensitive to hyperparameters; may not address transient gap |
| Reset/regeneration for plasticity | Reset neurons/weights or constrain trainability | ReDO, CBP, SNR, spectral regularization | MNIST variants, vision CL | Targets plasticity loss; not directly optimizer-state |
| Optimizer-state interventions | Change optimizer state handling under nonstationarity | Adam-Rel, Continual AdamW, optimizer resets in RL | RL + CL settings | Mechanism attribution often confounded with other changes |

### Closest Prior Work

- **Adam-Rel / local timesteps in RL (Ellis et al., 2024)**: resets Adam’s timestep under RL objective changes while keeping moments and provides theory linking timestep resets to bounded update sizes; our work tests the same mechanism under supervised CL task switches and adds a bias-correction-off control.
- **Continual Adam/AdamW (Kang & Lee, 2024)**: modifies AdamW by introducing a global second-moment buffer and an explicit early-step multiplier; our work isolates a strictly smaller change (timestep-only reset) and evaluates it specifically on stability-gap metrics.
- **Stability gap head attribution (Lapacz et al., 2025)**: shows head choice can dominate stability gap; our work tests whether optimizer timestep handling is an orthogonal lever that reduces transient post-switch drops even with a standard linear head.

**Novelty Kill Search Summary:** Searched for the exact combination “Adam timestep reset” + “continual learning” + “task boundary” (and variants: “Adam-Rel continual learning”, “relative timestep Adam continual learning”, “bias correction reset task switch”), and checked local finalized proposals for “SGDR / warm restart / timestep reset / Adam-Rel”. No prior work was found that evaluates **timestep-only reset with moments preserved** in supervised CL together with a **bias-correction disabling control**, as of 2026-02-22.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Adam-Rel (Ellis et al.) | Reset Adam timestep under RL nonstationarity | RL-only evaluation | Apply to supervised CL stability gap; add bias-corr-off control | CL task switches can induce discrete gradient changes; same failure mode may apply |
| Continual AdamW (Kang & Lee) | Second-moment reuse + early-step multiplier | Changes multiple optimizer components | Change only timestep (keep moments) | Cleaner attribution; minimal engineering; potentially complementary |
| Lapacz et al. | Replace linear head with NMC to reduce stability gap | Head-focused; does not target optimizer dynamics | Keep standard head; change optimizer timestep only | Tests an orthogonal lever that may reduce transient failures without changing inference |

---

## Experiments

### Experimental Setup

**Goal:** Test whether resetting only AdamW’s bias-correction timestep at task boundaries reduces stability-gap metrics on standard supervised CL benchmarks.

**Baseline Ladder (REQUIRED):**

- **A (CarryAll-AdamW)**: carry full AdamW optimizer state across tasks (`m,v,t` continue). This matches checkpoint-resume style training.
- **B (LT-AdamW, ours)**: carry `m,v` but reset `t→0` at each task boundary.
- **C (Mechanism control: NoBiasCorr)**: disable Adam bias correction; under this setting, resetting `t` should not help if the mechanism is correct.

*Prompting baselines and inference-time scaling baselines are not applicable because the task is supervised image classification with fixed test sets; the relevant baseline ladder in this domain is optimizer/state handling and standard CL method families.*

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| ResNet-18 | ~11M params | torchvision | Standard vision backbone |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| MNIST | Base for Rotated-MNIST tasks | 60k train / 10k test | torchvision | permissive |
| CIFAR-100 | Split CIFAR-100 class-incremental tasks | 50k train / 10k test | torchvision | research |

**Other Resources:** None.

**Resource Estimate:**

- **Compute budget**: ≤100 GPU-hours total (2 benchmarks × 3 conditions × 3 seeds; ResNet-18 scale)
- **GPU memory**: ≤1×A100 80GB (or similar)
- **API usage**: none

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Rotated MNIST (Domain-Incremental Learning; Domain-IL) | Same labels, but each task rotates digit images by a fixed angle; tests domain shift across tasks | Stability Gap (SG, ↓), min-ACC (↑), ACC (↑), BWT (↑) | test | torchvision | standard CL eval |
| Split CIFAR-100 (Class-Incremental Learning; Class-IL) | Each task introduces new classes; tests class-incremental learning with interference | SG (↓), min-ACC (↑), ACC (↑), BWT (↑) | test | torchvision | standard CL eval |

**Metric definitions (first use):**

- **ACC**: average test accuracy over all seen tasks after the final task (higher is better).
- **min-ACC**: minimum accuracy on previously seen tasks measured in a fixed window after each task switch (higher is better).
- **Stability Gap (SG)**: maximum normalized drop in past-task accuracy immediately after a task switch relative to pre-switch accuracy (lower is better; Lapacz-style).
- **BWT (Backward Transfer)**: change in old-task performance after learning new tasks (less negative / more positive is better).

### Main Results

#### Results Table

(All numbers to be produced by verification runs; report mean±std across 3 seeds.)

| Method | Base Model | Benchmark | SG (↓) | min-ACC (↑) | Source | Notes |
|---|---|---|---|---|---|---|
| CarryAll-AdamW (A) | ResNet-18 | Rotated MNIST | **TBD** | **TBD** | - | baseline |
| LT-AdamW (B) | ResNet-18 | Rotated MNIST | **TBD** | **TBD** | - | reset `t` only |
| NoBiasCorr (C) | ResNet-18 | Rotated MNIST | **TBD** | **TBD** | - | bias correction disabled |
| CarryAll-AdamW (A) | ResNet-18 | Split CIFAR-100 | **TBD** | **TBD** | - | baseline |
| LT-AdamW (B) | ResNet-18 | Split CIFAR-100 | **TBD** | **TBD** | - | reset `t` only |
| NoBiasCorr (C) | ResNet-18 | Split CIFAR-100 | **TBD** | **TBD** | - | bias correction disabled |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B (full) | reset `t` for all parameters | best SG/min-ACC among AdamW variants if hypothesis holds |
| C (mechanism control) | disable bias correction | reduces/erases B’s advantage if mechanism holds |

### Experimental Rigor

**Variance & Reproducibility:**

- Run all conditions with `seeds=[42, 123, 456]`.
- Report mean ± std for SG and min-ACC.

**Validity & Controls (top confounders):**

1. **Learning-rate schedule confound:** use constant learning rate (no warmup, no decay) across all conditions.
2. **Evaluation-frequency confound:** compute SG/min-ACC from a fixed set of post-switch evaluation checkpoints (e.g., steps {0, 1, 5, 20, 100, 500}) rather than relying on end-of-task evaluation.
3. **Head-dominance confound:** report stability-gap metrics with the standard linear head; optionally (analysis-only) compute an NMC readout on the same backbone features to check whether effects are head-specific.

**Sanity checks:**

- Log update norms `||Δθ||` and gradient norms `||g||` for the first K=50 optimizer steps after each task switch; verify that LT-AdamW reduces the early-step update spike magnitude relative to CarryAll-AdamW.

### Analysis (Optional)

- **Budget sensitivity:** repeat Split CIFAR-100 with 5 epochs/task (≈390 steps/task) vs 10 epochs/task (≈780 steps/task) to test whether effects concentrate in the early post-switch window.

---

## Success Criteria

**Hypothesis** (directional): LT-AdamW reduces stability gap (lower SG, higher min-ACC) relative to CarryAll-AdamW by reducing early post-switch update spikes, and this advantage is largely absent when bias correction is disabled.

**Decision Rule** (concrete):

- **Proceed** if, on Split CIFAR-100 (and ideally also Rotated MNIST), LT-AdamW improves SG and/or min-ACC over CarryAll-AdamW by a margin larger than the across-seed standard deviation, and the improvement is substantially smaller or absent under NoBiasCorr.
- **Pivot** if LT-AdamW improves but NoBiasCorr improves similarly (suggesting the effect is not tied to bias correction); reframe as a generic post-switch step-size control and compare to explicit LR warmup baselines.
- **Refute** if LT-AdamW ≈ CarryAll-AdamW (within noise) on SG and min-ACC on Split CIFAR-100.

---

## Impact Statement

If validated, LT-AdamW provides a minimal, optimizer-state-only intervention to reduce transient post-switch failures in checkpoint-resume continual training pipelines that already use AdamW. This would be most relevant for practitioners who care about worst-case performance during continual updates (e.g., long-running fine-tuning or continual pre-training), where reinitializing training or discarding optimizer state is undesirable.

---

## References

- [Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps](../../../papers/paper_summaries/Adam%20on%20Local%20Time%20Addressing%20Nonstationarity%20in%20RL%20with%20Relative%20Adam%20Timesteps.md) - Ellis et al., 2024
- [Exploring the Stability Gap in Continual Learning: The Role of the Classification Head](../../../papers/paper_summaries/Exploring%20the%20Stability%20Gap%20in%20Continual%20Learning%20The%20Role%20of%20the%20Classification%20Head.md) - Lapacz et al., WACV 2025
- [Continual Learning of Numerous Tasks from Long-tail Distributions](../../../papers/paper_summaries/Continual%20Learning%20of%20Numerous%20Tasks%20from%20Long-tail%20Distributions.md) - Kang & Lee, 2024
- [Learning Continually by Spectral Regularization](../../../papers/paper_summaries/LEARNING%20CONTINUALLY%20BY%20SPECTRAL%20REGULARIZATION.md) - Lewandowski et al., 2024
- [Self-Normalized Resets for Plasticity in Continual Learning](../../../papers/paper_summaries/Self-Normalized%20Resets%20for%20Plasticity%20in%20Continual%20Learning.md) - Farias & Jozefiak, 2024
- [Resetting the Optimizer in Deep RL: An Empirical Study](https://arxiv.org/abs/2306.17833) - Asadi et al., 2023
- [Understanding the Role of Training Regimes in Continual Learning](https://arxiv.org/abs/2006.06958) - Mirzadeh et al., 2020
- [On Warm-Starting Neural Network Training](https://arxiv.org/abs/1910.08475) - Ash & Adams, 2020
- [Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) - Loshchilov & Hutter, 2017
