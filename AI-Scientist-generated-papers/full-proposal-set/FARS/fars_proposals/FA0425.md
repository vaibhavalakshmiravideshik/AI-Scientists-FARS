# untitled

# Delta-Rule Momentum Damping for Replay-Based Continual Learning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated evaluation (no human evaluation, no LLM-as-judge)
  - ≤ 768 A100 GPU-hours

## Introduction

### Context and Motivation

**Continual learning (CL)** studies how to train a single model on a stream of tasks or distributions while retaining performance on earlier tasks. A canonical and challenging setting is **class-incremental learning (Class-IL)**, where tasks introduce new classes over time and the model must classify among all classes seen so far at test time. In this setting, naively fine-tuning on the current task often causes **catastrophic forgetting**, i.e., large accuracy drops on earlier tasks.

A large part of the CL literature proposes explicit mechanisms to reduce forgetting (replay buffers, regularization penalties, gradient projection, architectural growth). In contrast, most CL training pipelines still use standard, fixed-parameter optimizers such as **stochastic gradient descent (SGD) with momentum**, even though optimizer state (e.g., momentum) is itself a form of memory over past gradients.

If optimizer-state choices materially affect forgetting, then an optimizer-only intervention could be attractive: it is easy to implement, compatible with many CL algorithms, and does not require extra model components.

### The Problem

In replay-based Class-IL methods such as **DER++ (Dark Experience Replay++)**, training batches combine current-task examples with a small replay buffer. The loss landscape and gradient field change abruptly at task boundaries and can also shift within a task due to replay mixture effects.

Standard SGD momentum maintains an **exponential moving average (EMA)** of gradients with a fixed coefficient \(\beta\). This is beneficial in stationary settings, but in non-stationary CL the accumulated momentum direction can become stale and may amplify gradient interference, potentially increasing forgetting.

A simple engineering response is to reset momentum at task boundaries. However, this (i) assumes known task boundaries, (ii) discards potentially useful momentum when the stream is smooth, and (iii) is not a fully local, automatic rule.

### Key Insight and Hypothesis

**Key insight (delta-rule view of momentum).** Nested Learning reframes momentum as an associative memory and proposes **delta-rule momentum variants** whose update includes a gradient-energy-dependent carry-over term. In their derivation, replacing a dot-product internal objective with an \(\ell_2\) regression objective yields an update of the form:

\[
 m_{i+1} = m_i\,\big(\alpha_{i+1} - \nabla L(W_i)^\top \nabla L(W_i)\big) - \eta\,P_i\,\nabla L(W_i),
\]

introducing a dependence on \(\|\nabla L\|^2\) that can accelerate adaptation when the gradient field changes ([Nested Learning](./references/Nested-Learning-The-Illusion-of-Deep-Learning-Architecture/meta/meta_info.txt), Sec. 4.4).

**Hypothesis.** In replay-based class-incremental learning, replacing fixed momentum \(\beta\) with a **gradient-energy-gated** momentum coefficient \(\beta_t\) reduces forgetting (lower Final Forgetting, FF) without materially reducing final performance (Final Average Accuracy, FAA).

Why this could be wrong:
- The gain could be explained by a generic **effective step-size reduction** rather than momentum-specific dynamics.
- The effect could be matched by simple **gradient clipping**.
- The effect might disappear under a different optimizer (e.g., Adam/AdamW), indicating limited generality.

We pre-register controls to distinguish these explanations.

---

## Proposed Approach

### Overview

We modify only the optimizer used inside **replay-based** Class-IL baselines. The core change is to **damp momentum automatically** when the current minibatch gradient energy spikes relative to a running baseline. To test whether the effect generalizes beyond a single training recipe, we evaluate the same optimizer change inside **two** replay methods: **DER++** and **ER-ACE**.

### Method Details

#### Base continual learning algorithms: DER++ and ER-ACE

- **DER++** stores a small memory buffer of past examples and replays them during training. It uses knowledge distillation on stored logits (DER) plus replay of ground-truth labels (the “++” variant).
- **ER-ACE** is an experience replay variant that addresses replay/stream imbalance using an asymmetric cross-entropy design.

We include both because they are strong, widely used replay baselines for Class-IL and are implemented in Mammoth, enabling a clean generalization check without changing the model or benchmark.

#### Baseline optimizer: SGD with fixed momentum

Let \(\theta\) be model parameters and \(g_t\) the gradient of the DER++ training loss for minibatch \(t\). Standard momentum SGD updates:

\[
 v_{t+1} = \beta_0 v_t + g_t,\qquad \theta_{t+1} = \theta_t - \mathrm{lr}_t\, v_{t+1}.
\]

#### Proposed optimizer: gradient-energy-gated (“delta-rule”) momentum damping

We implement a minimal, stable scalar adaptation inspired by the delta-rule term \((\alpha-\|g\|^2)\) in Nested Learning.

1. **Maintain a running gradient-energy baseline**:
\[
 s_t = \rho\, s_{t-1} + (1-\rho)\,\|g_t\|_2^2.
\]

2. **Compute a normalized gradient-energy ratio**:
\[
 u_t = \frac{\|g_t\|_2^2}{s_t + \varepsilon}.
\]

3. **Set an adaptive momentum coefficient** (no tuned scale parameter):
\[
 \beta_t = \beta_0\, \mathrm{clamp}(2 - u_t,\; 0,\; 1).
\]

4. **Update momentum with \(\beta_t\)**:
\[
 v_{t+1} = \beta_t v_t + g_t.
\]

Intuition: when \(u_t\approx 1\) (typical gradient energy), \(\beta_t\approx \beta_0\). When gradients spike (\(u_t>1\)), the carry-over term is reduced, approaching a momentum reset when \(u_t\ge 2\).

**Default hyperparameters (fixed across runs):** \(\beta_0=0.9\), \(\rho=0.99\), \(\varepsilon=10^{-12}\). We do not tune an additional “\(\kappa\)” scaling parameter; the mapping \(2-u_t\) fixes the sensitivity.

#### Step-size confound control: update-norm-matched fixed-momentum SGD

Because changing \(\beta\) can change the effective step size, we include a pre-registered control that keeps \(\beta=\beta_0\) but rescales \(\mathrm{lr}\) to match the **median parameter update norm** of the adaptive method on a short calibration window.

For each seed:
1. On task 1, run \(K=200\) optimization steps from the same initialization for:
   - fixed \(\beta=\beta_0\) at learning rate \(\mathrm{lr}_0\)
   - adaptive \(\beta_t\) at learning rate \(\mathrm{lr}_0\)
2. Record per-step update norms \(\Delta_t = \|\theta_{t+1}-\theta_t\|_2\).
3. Compute \(s = \mathrm{median}(\Delta_t\,|\,\text{adaptive}) / \mathrm{median}(\Delta_t\,|\,\text{fixed})\).
4. Define the LR-matched control as fixed \(\beta=\beta_0\) with learning rate \(\mathrm{lr}_0\cdot s\) (and the same LR schedule shape).

### Key Innovations

- **A minimal delta-rule-inspired momentum rule for non-stationary training**: adapt momentum carry-over using a normalized gradient-energy signal (no task IDs, no extra model modules).
- **Mechanism-focused CL test**: isolates whether stale optimizer momentum contributes to forgetting in replay-based Class-IL.
- **Pre-registered confound control**: includes an update-norm-matched control to separate “momentum dynamics” from “effective step size”.

---

## Related Work

### Field Overview

Continual learning methods are commonly grouped into (i) **regularization** (penalize changing important parameters), (ii) **replay** (store or generate past data), (iii) **gradient constraint/projection** (explicitly modify gradients to avoid interference), and (iv) **architectural or modular** approaches. Replay-based approaches such as ER, DER++, and X-DER are strong practical baselines for Class-IL and remain widely used because they are conceptually simple and effective.

Optimizer dynamics are less often treated as a first-class design axis in CL, despite the fact that optimizer state (momentum, adaptive moments) is a persistent memory that can interact with non-stationary task streams. Nested Learning explicitly reframes optimizers as memory systems and proposes delta-rule momentum variants, but it does not establish whether these ideas reduce catastrophic forgetting under standard Class-IL benchmarks.

### Related Papers

- **[Nested Learning: The Illusion of Deep Learning Architectures](./references/Nested-Learning-The-Illusion-of-Deep-Learning-Architecture/meta/meta_info.txt)**: Proposes a unifying view of learning modules and introduces delta-rule momentum variants with gradient-energy-dependent carry-over.
- **[Class-Incremental Continual Learning into the eXtended DER-verse](./references/Class-Incremental-Continual-Learning-into-the-eXtended-DER-verse/meta/meta_info.txt)**: Introduces X-DER and reports strong DER/DER++ baselines on Split CIFAR-100.
- **[Dark Experience Replay (DER/DER++)](https://arxiv.org/abs/2004.07211)**: Strong replay + distillation baselines for Class-IL.
- **[ER-ACE](https://arxiv.org/abs/2112.00432)**: A replay variant addressing class imbalance by asymmetric cross-entropy.
- **[iCaRL](https://arxiv.org/abs/1611.07725)**: Exemplar-based Class-IL method using distillation and nearest-mean classification.
- **[Learning without Forgetting (LwF)](https://arxiv.org/abs/1606.09282)**: Distillation-based regularization for sequential learning.
- **[EWC](https://arxiv.org/abs/1612.00796)**: Weight-regularization CL using Fisher information.
- **[Synaptic Intelligence (SI)](https://arxiv.org/abs/1703.04200)**: Online importance-weighted regularization.
- **[MAS](https://arxiv.org/abs/1711.09601)**: Parameter-importance regularization based on sensitivity.
- **[GEM](https://arxiv.org/abs/1706.08840)**: Constrains gradients using episodic memory to prevent interference.
- **[A-GEM](https://arxiv.org/abs/1812.00420)**: Computationally cheaper approximation of GEM.
- **[Orthogonal Gradient Descent (OGD)](http://proceedings.mlr.press/v108/farajtabar20a.html)**: Projects gradients to avoid interference.
- **[Riemannian Walk](https://arxiv.org/abs/1801.10112)**: Studies forgetting/intransigence and proposes a geometry-aware regularizer.
- **[Continual learning with the neural tangent ensemble (NTE)](https://arxiv.org/abs/2408.17394)**: Argues momentum can worsen forgetting and provides evidence in vision CL.
- **[MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning](https://arxiv.org/abs/2407.20999)**: Uses momentum statistics to filter updates in sequential LLM fine-tuning.
- **[Torque-Aware Momentum (TAM)](https://arxiv.org/abs/2412.18790)**: Dampens momentum based on gradient–momentum directional relationships.
- **[Adam](https://arxiv.org/abs/1412.6980)**: Adaptive optimizer with first/second-moment estimates.
- **[AdamW](https://openreview.net/forum?id=Bkg6RiCqY7)**: Decoupled weight decay for Adam.
- **[SGDR / cosine restarts](https://arxiv.org/abs/1608.03983)**: Learning-rate restarts for non-convex optimization (relevant as a comparison to “resetting state”).
- **[Avalanche](https://arxiv.org/abs/2104.00405)**: CL library providing standard benchmarks and baselines.
- **[Mammoth](https://github.com/aimagelab/mammoth)**: Modular CL framework used by X-DER and provides DER++ implementations.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Regularization | Penalize changing important parameters | EWC, SI, MAS | Split CIFAR-100, miniImageNet | Can underperform replay in Class-IL; sensitive to importance estimates |
| Replay (basic) | Store and replay past data | ER, iCaRL | Split CIFAR-100 | Memory budget; replay imbalance |
| Replay + distillation | Store logits; distill past behavior | DER, DER++, X-DER | Split CIFAR-100, miniImageNet | Still forgets; additional hyperparameters |
| Gradient constraints | Modify gradients to reduce interference | GEM, A-GEM, OGD | Split CIFAR-100 | Compute/storage overhead; can hurt plasticity |
| Optimizer dynamics | Modify optimizer state updates in non-stationarity | Nested Learning (delta momentum), MoFO, TAM | Mixed (not established for Class-IL replay) | Often not evaluated on standard CL protocols |

### Closest Prior Work

- **Nested Learning (Delta Momentum variants)** ([Nested Learning](./references/Nested-Learning-The-Illusion-of-Deep-Learning-Architecture/meta/meta_info.txt)): Derives delta-rule momentum updates with gradient-energy-dependent carry-over; does not test whether such momentum designs reduce forgetting in standard replay-based Class-IL.
- **NTE (momentum worsens forgetting)** ([NTE](https://arxiv.org/abs/2408.17394)): Provides evidence that increasing fixed momentum increases forgetting; our proposal tests whether *adaptive damping* can keep some benefits of momentum while reducing forgetting.
- **MoFO (momentum-filtered updates)** ([MoFO](https://arxiv.org/abs/2407.20999)): Uses momentum statistics to decide which parameters to update in sequential LLM fine-tuning; our method changes the momentum coefficient itself and targets standard vision Class-IL.
- **TAM (direction-aware damping)** ([TAM](https://arxiv.org/abs/2412.18790)): Dampens momentum based on gradient–momentum directional relationships; our method uses gradient energy spikes (\(\|g\|^2\)) rather than angles and is evaluated in a replay-based Class-IL pipeline.

**Novelty Kill Search Summary:** Searched for the exact combination “delta-rule momentum” + “continual learning / class-incremental / replay” (queries listed in `notes.md`) and did not find prior work applying Nested-Learning-style delta-rule momentum damping inside DER++-style replay on Split CIFAR-100 as of **2026-03-02**. (Full query log in `notes.md`.)

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Nested Learning | Delta-rule momentum variants via associative-memory framing | Not validated on standard CL benchmarks | Instantiate a minimal scalar \(\beta_t\) rule for SGD in Class-IL replay | Targets the non-stationarity mechanism directly with minimal engineering |
| NTE | Observes fixed momentum increases forgetting | Suggests removing momentum; does not propose adaptive rule | Adapt momentum only when gradients spike | Retains momentum when helpful; damps when stale |
| MoFO | Filters parameter updates using momentum statistics | Different setting (LLM fine-tuning); different control variable | Adjust momentum carry-over coefficient | Cheaper and directly tied to delta-rule derivation |
| TAM | Angle-based momentum damping | Different signal (angle) and not tied to CL interference | Use energy-based damping signal | Energy spikes are easy to compute and may correlate with task shifts |

---

## Experiments

### Experimental Setup

**Benchmarks / settings:**

- **Setting A (primary):** Split CIFAR-100 **Class-incremental learning (Class-IL)** (10 tasks × 10 classes) with **DER++**.
- **Setting B (generalization check):** Split CIFAR-100 Class-IL with **ER-ACE**.

We use Mammoth’s implementations (same codebase as X-DER) to minimize implementation risk.

**Main comparison (Setting A, 3 conditions, 3 seeds):**
1. **DER++ + SGD-momentum (fixed \(\beta_0\))**
2. **DER++ + SGD-momentum (adaptive \(\beta_t\), ours)**
3. **DER++ + SGD-momentum (fixed \(\beta_0\), LR-matched control)**

**Generalization check (Setting B, 2 conditions, 3 seeds; run only if Setting A meets the Proceed criterion):**
1. **ER-ACE + SGD-momentum (fixed \(\beta_0\))**
2. **ER-ACE + SGD-momentum (adaptive \(\beta_t\), ours)**

**Optional ablations (run only if the main result is positive and budget allows):**
- **Random gating control:** drop momentum (set \(\beta=0\)) at random steps with the same frequency of near-zero \(\beta_t\) observed in condition (2)’s calibration window (tests whether timing matters).
- **Gradient clipping control:** fixed momentum with gradient clipping chosen to match the upper tail of \(\|g\|\) in condition (2)’s calibration window (tests whether effect is equivalent to clipping spikes).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| ResNet-18 | ~11M params | https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html | Train from scratch (as in X-DER) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| CIFAR-100 | Class-incremental image classification stream | 50k train / 10k test | https://www.cs.toronto.edu/~kriz/cifar.html (also via torchvision) | Research use |

**Implementation details (aligned with X-DER paper where specified):**
- Train from scratch.
- For Split CIFAR-100: 50 epochs per task; learning-rate drops by 10× at epochs [35, 45] ([X-DER](./references/Class-Incremental-Continual-Learning-into-the-eXtended-DER-verse/meta/meta_info.txt), Sec. 5.1).
- Buffer size: \(\mathcal{M}_{\mathrm{size}}=500\) (low-memory regime).

**Resource Estimate**:
- **Compute budget**: ≤ 120 A100 GPU-hours total.
  - Setting A main: 3 conditions × 3 seeds = 9 runs.
  - Setting B (generalization): 2 conditions × 3 seeds = 6 runs, executed only if Setting A meets Proceed.
  - ResNet18 + Split CIFAR-100 is lightweight; expected single-run time is on the order of ~1–3 GPU-hours depending on batch size and implementation.
- **GPU memory**: ≤ 1×A100 80GB per run.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Split CIFAR-100 (Class-IL) | 10-task class-incremental stream built from CIFAR-100 | FAA (↑), FF (↓) | standard test per task | https://www.cs.toronto.edu/~kriz/cifar.html | Mammoth (https://github.com/aimagelab/mammoth) |

### Main Results

#### Results Table

(Report mean±std over seeds for our runs; published baselines are single numbers.)

| Method | Base Model | Benchmark | FAA (mean±std) | FF (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| FT (lower bound) | ResNet-18 | Split CIFAR-100 (M=500) | 9.43 | 89.82 | [X-DER Table I](<./references/Class-Incremental-Continual-Learning-into-the-eXtended-DER-verse/sections/5.3 Discussion.md>) | Published (1 run) |
| ER | ResNet-18 | Split CIFAR-100 (M=500) | 22.10 | 73.64 | [X-DER Table I](<./references/Class-Incremental-Continual-Learning-into-the-eXtended-DER-verse/sections/5.3 Discussion.md>) | Published (1 run) |
| ER-ACE | ResNet-18 | Split CIFAR-100 (M=500) | 38.75 | 40.04 | [X-DER Table I](<./references/Class-Incremental-Continual-Learning-into-the-eXtended-DER-verse/sections/5.3 Discussion.md>) | Published (1 run) |
| iCaRL | ResNet-18 | Split CIFAR-100 (M=500) | 46.52 | 22.06 | [X-DER Table I](<./references/Class-Incremental-Continual-Learning-into-the-eXtended-DER-verse/sections/5.3 Discussion.md>) | Published (1 run) |
| DER++ | ResNet-18 | Split CIFAR-100 (M=500) | 38.25 | 50.54 | [X-DER Table I](<./references/Class-Incremental-Continual-Learning-into-the-eXtended-DER-verse/sections/5.3 Discussion.md>) | Published (1 run) |
| X-DER (SOTA in that paper) | ResNet-18 | Split CIFAR-100 (M=500) | 49.93 | 19.90 | [X-DER Table I](<./references/Class-Incremental-Continual-Learning-into-the-eXtended-DER-verse/sections/5.3 Discussion.md>) | Published (1 run) |
| DER++ + SGD-momentum (fixed \(\beta_0\)) | ResNet-18 | Split CIFAR-100 (M=500) | **TBD** | **TBD** | this work | 3 seeds |
| **DER++ + delta-rule momentum damping (ours)** | ResNet-18 | Split CIFAR-100 (M=500) | **TBD** | **TBD** | this work | 3 seeds |
| DER++ + SGD-momentum (LR-matched) | ResNet-18 | Split CIFAR-100 (M=500) | **TBD** | **TBD** | this work | 3 seeds |
| ER-ACE + SGD-momentum (fixed \(\beta_0\)) | ResNet-18 | Split CIFAR-100 (M=500) | **TBD** | **TBD** | this work | 3 seeds (run only if DER++ Proceed) |
| **ER-ACE + delta-rule momentum damping (ours)** | ResNet-18 | Split CIFAR-100 (M=500) | **TBD** | **TBD** | this work | 3 seeds (run only if DER++ Proceed) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Random gating | Randomly set \(\beta=0\) with frequency matched to ours | If close to ours → timing/shift signal may not matter |
| Gradient clipping | Clip gradients in fixed-\(\beta\) baseline | If close to ours → effect may be explainable as clipping |
| EMA decay \(\rho\) | Change \(\rho\in\{0.9, 0.99, 0.999\}\) | If robust → rule is not sensitive to smoothing |

### Experimental Rigor

- **Seeds**: run each main condition with 3 seeds (e.g., 42/123/456) and report mean±std.
- **Sanity check**: reproduce the published DER++ baseline approximately (38.25 FAA / 50.54 FF for M=500). If far off, adjust implementation details to match Mammoth’s official configuration.
- **Confounders and controls**:
  - Effective step size: addressed by LR-matched control.
  - “Just clipping spikes”: addressed by optional gradient clipping ablation.
  - “Timing does not matter”: addressed by optional random gating ablation.
- **Fair comparison**: same architecture, data order, buffer size, batch size, augmentation, and LR schedule across conditions. We report results separately for DER++ and ER-ACE; within each base method, only the optimizer differs.

### Analysis (Optional)

- Log \(u_t\) and \(\beta_t\) distributions across training, and quantify whether \(\beta_t\) drops correlate with task boundaries and replay-heavy batches.

---

## Success Criteria

**Hypothesis** (directional): Delta-rule momentum damping reduces forgetting (lower FF) compared to fixed-\(\beta\) momentum in DER++, with little or no drop in FAA.

**Decision Rule** (concrete):
- **Proceed** if (ours) improves FF over fixed-\(\beta\) baseline by ≥ max(2.0, pooled_std) while reducing FAA by ≤ max(1.0, pooled_std), and (ours) also beats the LR-matched control on FF by ≥ pooled_std.
- **Pivot** if (ours) beats fixed-\(\beta\) but the LR-matched control matches it (within pooled_std), suggesting the effect is primarily step-size; then re-scope to LR scheduling rather than momentum.
- **Refute** if FF improvement is within pooled_std or FAA drops by > max(1.0, pooled_std).

---

## Impact Statement

If delta-rule momentum damping reduces forgetting in replay-based Class-IL, it provides a simple optimizer-level modification that practitioners can add to existing continual learning pipelines (DER++/ER-like) without changing model architecture or adding new loss terms. If the LR-matched control explains the effect, the result is still decision-changing: apparent “momentum memory” gains may be reproducible via simple learning-rate calibration rather than specialized CL mechanisms.

---

## References

- [Nested Learning: The Illusion of Deep Learning Architectures](./references/Nested-Learning-The-Illusion-of-Deep-Learning-Architecture/meta/meta_info.txt) - Behrouz et al., 2025
- [Class-Incremental Continual Learning into the eXtended DER-verse](./references/Class-Incremental-Continual-Learning-into-the-eXtended-DER-verse/meta/meta_info.txt) - Boschini et al., 2022
- [Dark Experience Replay (DER/DER++)](https://arxiv.org/abs/2004.07211) - Buzzega et al., 2020
- [ER-ACE](https://arxiv.org/abs/2112.00432) - Caccia et al., 2022
- [iCaRL](https://arxiv.org/abs/1611.07725) - Rebuffi et al., 2017
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282) - Li & Hoiem, 2016
- [EWC](https://arxiv.org/abs/1612.00796) - Kirkpatrick et al., 2017
- [Synaptic Intelligence](https://arxiv.org/abs/1703.04200) - Zenke et al., 2017
- [MAS](https://arxiv.org/abs/1711.09601) - Aljundi et al., 2018
- [GEM](https://arxiv.org/abs/1706.08840) - Lopez-Paz & Ranzato, 2017
- [A-GEM](https://arxiv.org/abs/1812.00420) - Chaudhry et al., 2019
- [Orthogonal Gradient Descent](http://proceedings.mlr.press/v108/farajtabar20a.html) - Farajtabar et al., 2020
- [Riemannian Walk](https://arxiv.org/abs/1801.10112) - Chaudhry et al., 2018
- [Continual learning with the neural tangent ensemble](https://arxiv.org/abs/2408.17394) - Benjamin et al., 2024
- [MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning](https://arxiv.org/abs/2407.20999) - 2024
- [Torque-Aware Momentum](https://arxiv.org/abs/2412.18790) - 2024
- [Adam](https://arxiv.org/abs/1412.6980) - Kingma & Ba, 2014
- [Decoupled Weight Decay Regularization (AdamW)](https://openreview.net/forum?id=Bkg6RiCqY7) - Loshchilov & Hutter, 2019
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) - Loshchilov & Hutter, 2016
- [Avalanche](https://arxiv.org/abs/2104.00405) - Lomonaco et al., 2021
- [Mammoth](https://github.com/aimagelab/mammoth) - Boschini et al.
