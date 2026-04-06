# untitled

# SignSplit: Do CReLU-Style Negative-Evidence Channels Improve Adversarial Robustness?

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Adversarial examples are small input perturbations that can cause neural networks to make incorrect predictions. Despite substantial progress in adversarial training (training on worst-case perturbations), standard vision backbones still require substantial compute and careful tuning to achieve robustness on benchmarks such as CIFAR-10 under an \(\ell_\infty\) threat model (\(\varepsilon=8/255\)).

Most work on adversarial robustness focuses on training objectives (e.g., PGD adversarial training and TRADES) or data/augmentation recipes. Comparatively less attention is paid to whether **the default architectural micro-choices of standard CNNs (e.g., ReLU)** systematically interact with robustness. Yet adversarial training is computationally heavy, so architectural changes that improve robustness—or preserve robustness while reducing FLOPs—could directly lower training and deployment cost.

This matters because adversarial vulnerability is increasingly explained in terms of **representation geometry**—e.g., feature interference under superposition (representing more features than dimensions via overlapping feature vectors)—suggesting that small architectural inductive biases could have outsized effects. Recent evidence that learnable parametric activation functions can improve AutoAttack robustness by several points and reach RobustBench SOTA within a model category **[Dai et al., 2021](./references/Parameterizing-Activation-Functions-for-Adversarial-Robustness/meta/meta_info.txt)** further suggests activation design is not a solved axis.

A classic observation in convolutional networks is that, when trained with ReLU, early-layer filters often come in opposite-phase pairs. This motivated **Concatenated ReLU (CReLU)**, which explicitly preserves both positive and negative responses by concatenating \(\mathrm{ReLU}(x)\) and \(\mathrm{ReLU}(-x)\), improving standard accuracy and parameter efficiency in pre-ResNet CNNs **[Shang et al., 2016](./references/Understanding-and-Improving-Convolutional-Neural-Networks-via-Concatenated-Rectified-Linear-Units/meta/meta_info.txt)**.

### The Problem

**We do not know whether “negative-evidence preservation” activations (CReLU/MaxMin-style) help adversarial robustness under modern training and evaluation protocols.**

- **CReLU** explicitly preserves negative evidence **[Shang et al., 2016](./references/Understanding-and-Improving-Convolutional-Neural-Networks-via-Concatenated-Rectified-Linear-Units/meta/meta_info.txt)**, but (to our knowledge) has not been re-evaluated under modern adversarial robustness practice (TRADES/PGD adversarial training + AutoAttack evaluation).
- **MaxMin CNN** similarly preserves both positive and negative detections by duplicating and negating feature maps **[Blot et al., 2016](./references/Max-Min-convolutional-neural-networks-for-image-classification/meta/meta_info.txt)**, but is evaluated only on clean accuracy.
- Recent mechanistic work argues adversarial vulnerability is linked to **feature interference under superposition** **[Gorton & Lewis, 2025](./references/Adversarial-Examples-Are-Not-Bugs-They-Are-Superposition/meta/meta_info.txt)**, and that making features more monosemantic can improve robustness to several noise settings **[Zhang et al., 2024](./references/Beyond-Interpretability-The-Gains-of-Feature-Monosemanticity-on-Model-Robustness/meta/meta_info.txt)**—but these do not test a simple architectural intervention that directly targets the “signed feature” issue.

A practical issue is that many architecture tweaks are confounded by compute/width. If an activation change implicitly increases effective width, any robustness gain could simply be due to increased capacity. We need a **controlled test** that isolates whether explicitly representing negative evidence provides robustness signal beyond a width/shape change.

### Key Insight and Hypothesis

**Key insight.** If adversarial perturbations exploit interference among features, then forcing the network to represent “feature present” and “feature absent / opposite phase” as separate non-negative channels (\(\mathrm{ReLU}(x)\) and \(\mathrm{ReLU}(-x)\)) may reduce reliance on implicit opposite-phase filter pairing and reduce feature interference, improving robustness.

**Hypothesis (testable).** Under the same adversarial training recipe (**TRADES**: a KL-regularized objective that trades off clean accuracy and robustness; \(\varepsilon=8/255\)), a **PreActResNet-18** variant that expands each block’s first-convolution output into **signed channels** (CReLU split) will achieve higher **AutoAttack** (standard 4-attack evaluation ensemble) robust accuracy than a parameter- and shape-matched control that expands channels by **duplicating** positive activations.

Why this could fail:
- Robustness may be dominated by training objective and data/augmentation; activation micro-choices may have negligible effect.
- The negative branch may be redundant given BatchNorm centering and the network’s ability to learn opposite filters anyway.
- Any apparent gain could come from optimization differences (e.g., gradient flow) rather than a meaningful representational benefit.

---

## Proposed Approach

### Overview

We propose **SignSplit**, a controlled architectural intervention for CNN robustness studies.

In each ResNet BasicBlock, we (i) reduce the first convolution’s output width by 2× and then (ii) restore the original width using either:
- **DupSplit (control)**: \([\mathrm{ReLU}(x),\ \mathrm{ReLU}(x)]\)
- **CReLUSplit (ours)**: \([\mathrm{ReLU}(x),\ \mathrm{ReLU}(-x)]\)

This keeps downstream tensor shapes identical while ensuring the only difference between control and ours is whether negative evidence is explicitly preserved.

### Method Details

#### Baseline architecture
Use **PreActResNet-18** (the pre-activation ResNet variant commonly used in CIFAR-10 adversarial-training papers; sometimes denoted RN18), with 4 stages and widths \(C\in\{64,128,256,512\}\).

#### SignSplit block modification
For every BasicBlock (including downsampling blocks):

1. Replace the block’s first convolution `conv1: in=C_in → out=C` with a bottlenecked version:
   - `conv1': in=C_in → out=C/2` (same kernel size/stride as baseline conv1).
2. Apply BatchNorm on \(C/2\) channels.
3. Apply one of the two expansion operators to obtain \(C\) channels:
   - **DupSplit**: `cat(ReLU(z), ReLU(z))`
   - **CReLUSplit**: `cat(ReLU(z), ReLU(-z))`
4. Keep the block’s second convolution unchanged:
   - `conv2: in=C → out=C`.
5. Residual connection and final ReLU follow the standard ResNet block.

**Parameter/FLOP accounting.** In a standard BasicBlock, conv params are proportional to \(2C^2\) (ignoring constants). With SignSplit, conv1 params become \(0.5C^2\) and conv2 stays \(C^2\), giving \(1.5C^2\) total: approximately **25% fewer conv parameters/FLOPs per block** (excluding skip downsample convs).

### Key Innovations

- **A controlled sign-preservation test** for adversarial robustness: CReLU-like negative evidence is compared against a duplication control with the same shapes and conv parameterization.
- **Modern robustness evaluation**: We explicitly use AutoAttack (the de facto standard robustness evaluation protocol) instead of relying on a single PGD setting.
- **Decision-changing negative result is possible**: if CReLUSplit does not beat DupSplit, the conclusion is that explicit negative-evidence channels are not a useful robustness knob once width/shape is controlled.

---

## Related Work

### Field Overview

Adversarial robustness for vision models is commonly studied under norm-bounded threat models (\(\ell_\infty\), \(\ell_2\)) on datasets like CIFAR-10. The dominant defense family is adversarial training, which trains on adversarially perturbed examples (e.g., PGD). TRADES reframes robust training as balancing clean accuracy and robustness via a KL regularizer.

Evaluation has converged on AutoAttack, which combines multiple complementary attacks to reduce false robustness claims due to weak attacks or gradient masking. RobustBench further standardizes model evaluation and reporting.

Mechanistic and representation-level perspectives increasingly link robustness to the geometry and disentanglement of learned features (robust vs non-robust features; superposition; monosemanticity). Our proposal tests a simple architectural prior—explicitly separating positive and negative evidence—that historically improved clean accuracy but has not been carefully tested as a robustness lever.

### Related Papers

- **[Understanding and Improving CNNs via CReLU](./references/Understanding-and-Improving-Convolutional-Neural-Networks-via-Concatenated-Rectified-Linear-Units/meta/meta_info.txt)**: Introduces CReLU and shows early-layer opposite-phase filter pairing; does not test adversarial robustness.
- **[MaxMin CNN](./references/Max-Min-convolutional-neural-networks-for-image-classification/meta/meta_info.txt)**: Preserves positive/negative detections by duplicating/negating maps; evaluated only for clean accuracy.
- **[Parameterizing Activation Functions for Adversarial Robustness](./references/Parameterizing-Activation-Functions-for-Adversarial-Robustness/meta/meta_info.txt)**: Shows activation shape can measurably affect robustness; in particular, allowing **positive outputs on negative inputs** and increasing curvature can improve robustness, and learnable PAFs (e.g., PSSiLU) can improve AutoAttack robust accuracy under adversarial training.
- **[Adversarial Examples Are Not Bugs, They Are Superposition](./references/Adversarial-Examples-Are-Not-Bugs-They-Are-Superposition/meta/meta_info.txt)**: Connects superposition/feature interference to adversarial vulnerability; highlights that robustness depends on representation structure.
- **[Beyond Interpretability: Monosemanticity → Robustness](./references/Beyond-Interpretability-The-Gains-of-Feature-Monosemanticity-on-Model-Robustness/meta/meta_info.txt)**: Shows monosemantic features help robustness to several noise settings; explicitly lists adversarial robustness as an open question.
- **[TRADES](https://arxiv.org/abs/1901.08573)**: Standard adversarial training objective balancing clean loss and robustness via KL.
- **[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)**: PGD adversarial training formulation and threat-model framing.
- **[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)**: FGSM and linearity perspective for adversarial examples.
- **[Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175)**: Robust vs non-robust features framing.
- **[Reliable evaluation of adversarial robustness with AutoAttack](https://arxiv.org/abs/2003.01690)**: AutoAttack evaluation protocol.
- **[RobustBench](https://openreview.net/forum?id=SSKZPJCt7B)**: Standardized robustness benchmarking using AutoAttack.
- **[Adversarial Weight Perturbation (AWP)](https://arxiv.org/abs/2004.05884)**: Improves robustness by perturbing weights during adversarial training.
- **[Adversarial Training for Free!](https://arxiv.org/abs/1904.12843)**: Reduces computational cost of adversarial training.
- **[MART](https://arxiv.org/abs/1912.11935)**: Misclassification-aware adversarial training.
- **[Overestimation and Instability in TRADES](https://arxiv.org/abs/2410.07675)**: Shows TRADES can overestimate robustness under weak evaluation; motivates AutoAttack.
- **[Dummy-Classes Adversarial Training (DUCAT)](https://arxiv.org/abs/2410.12671)**: Proposes changing the training paradigm to reduce the accuracy–robustness trade-off.
- **[One-vs-the-Rest Loss for Adversarial Training (SOVR)](https://openreview.net/forum?id=S9WJvVZ3Ly)**: Reports AutoAttack results averaged over 3 runs for PreActResNet-18, highlighting that robust accuracy can vary across runs (std often a few tenths of a point in some settings).
- **[Toy Models of Superposition](https://arxiv.org/abs/2209.10652)**: Foundational superposition hypothesis and geometry.
- **[Engineering Monosemanticity in Toy Models](https://arxiv.org/abs/2211.09169)**: Methods for inducing monosemanticity and studying superposition.
- **[Rethinking Lipschitz Networks and Certified Robustness (SortNet)](./references/Rethinking-Lipschitz-Neural-Networks-and-Certified-Robustness-A-Boolean-Function-Perspective/meta/meta_info.txt)**: Shows the role of order statistics (e.g., GroupSort/MaxMin) in certified robustness.
- **[GroupSort / MaxMin for 1-Lipschitz networks](https://arxiv.org/abs/1906.04893)**: Uses sorting-based activations to increase expressivity under Lipschitz constraints.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Objective-level robustness | Change training loss to optimize worst-case accuracy | PGD-AT (Madry), TRADES, MART | CIFAR-10 \(\ell_\infty\) 8/255; AutoAttack | High compute; robust overfitting |
| Evaluation standardization | Strong, parameter-free multi-attack evaluation | AutoAttack; RobustBench | AutoAttack; RobustBench leaderboards | Still not fully adaptive |
| Representation-mechanistic views | Robustness linked to feature geometry / disentanglement | Superposition (Gorton & Lewis); non-robust features (Ilyas); monosemanticity (Zhang et al.) | Mixed (ImageNet, toy models) | Often lacks direct interventions |
| Activation / micro-architecture | Preserve negative evidence or impose structured nonlinearities | CReLU; MaxMin; GroupSort/SortNet | Typically clean acc; certified robustness | Robustness under modern AT often untested |

### Closest Prior Work

1. **CReLU (Shang et al., 2016)** **[paper](./references/Understanding-and-Improving-Convolutional-Neural-Networks-via-Concatenated-Rectified-Linear-Units/meta/meta_info.txt)**: Introduces sign-splitting activation to remove opposite-phase filter redundancy in CNNs; does not study adversarial robustness or adversarial training.
2. **MaxMin CNN (Blot et al., 2016)** **[paper](./references/Max-Min-convolutional-neural-networks-for-image-classification/meta/meta_info.txt)**: Similar motivation (negative detections matter) using map duplication/negation; evaluated only for clean accuracy and on small CNNs.
3. **PAFs for robustness (Dai et al., 2021)** **[paper](./references/Parameterizing-Activation-Functions-for-Adversarial-Robustness/meta/meta_info.txt)**: Shows activation shape is a meaningful robustness axis: for standard-trained models, **positive outputs on negative inputs** can improve robustness to weak attacks (Sec. 3.2), and under adversarial training, smooth/learnable activations (e.g., PSSiLU) can improve AutoAttack robust accuracy (Sec. 4.2). This motivates revisiting activation design under modern robustness protocols.
4. **Superposition→robustness (Gorton & Lewis, 2025)** **[paper](./references/Adversarial-Examples-Are-Not-Bugs-They-Are-Superposition/meta/meta_info.txt)**: Provides causal evidence that superposition contributes to adversarial vulnerability (toy models) and measures superposition proxies (SAEs) in ResNet-18; does not propose a simple architectural intervention.
5. **Monosemanticity→robustness (Zhang et al., 2024)** **[paper](./references/Beyond-Interpretability-The-Gains-of-Feature-Monosemanticity-on-Model-Robustness/meta/meta_info.txt)**: Shows monosemanticity improves robustness to several noise regimes and raises adversarial robustness as an open question; does not test sign-preserving activations.

**Novelty Kill Search Summary:** Searched for direct prior evaluations combining (i) CReLU/concatenated ReLU or MaxMin (or other sign-preserving activations) with (ii) adversarial training (TRADES/PGD-AT) and (iii) AutoAttack evaluation. Queries included: “CReLU adversarial training CIFAR-10 AutoAttack”, “concatenated ReLU TRADES AutoAttack”, “MaxMin activation adversarial training AutoAttack”, “GroupSort activation adversarial training CIFAR-10 AutoAttack”, “CReLU PreActResNet adversarial robustness”, “\"CReLU\" \"TRADES\"”, “\"concatenated rectified linear units\" adversarial”, and “\"MaxMin\" convolutional \"adversarial\" training”. No clear prior work directly benchmarking CReLU/MaxMin-style sign splitting under TRADES + AutoAttack was found as of 2026-02-28. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| CReLU (2016) | Sign-split activation for clean accuracy/efficiency | No adversarial robustness evaluation | Test sign-split under TRADES + AutoAttack | If negative-evidence channels reduce interference, robustness should improve |
| MaxMin CNN (2016) | Preserves negative detections via map duplication/negation | Evaluated only for clean acc; not ResNets | Apply a controlled sign-preservation test in ResNet blocks | Modern eval may reveal robustness effects |
| Superposition paper (2025) | Measures superposition ↔ robustness link | No direct intervention in real models | Use a simple intervention motivated by sign redundancy | Provides actionable architectural knob |
| Monosemanticity→robustness (2024) | Monosemantic features help noise robustness | Doesn’t test adversarial examples directly | Test a simpler, architecture-level prior for disentanglement | Could provide a cheap alternative to heavy monosemanticity methods |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):**
- **Level 0 (non-robust reference)**: Standard ERM training **PreActResNet-18** + ReLU (optional; expected to have near-zero AutoAttack robust accuracy).
- **Level 1 (main baseline)**: **PreActResNet-18** + ReLU trained with TRADES (\(\varepsilon=8/255\)).
- **Level 2 (control)**: **DupSplit** **PreActResNet-18** trained with TRADES.
- **Level 3 (ours)**: **CReLUSplit (SignSplit)** **PreActResNet-18** trained with TRADES.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| PreActResNet-18 (CIFAR-10) | ~11M params | https://github.com/RobustBench/robustbench (model zoo + configs) | Use the common PreActResNet18 implementation used in adversarial training papers (often called RN18) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| CIFAR-10 | Adversarial training + evaluation | 50k train / 10k test | https://www.cs.toronto.edu/~kriz/cifar.html | research-friendly |
| CIFAR-100 (generalization check) | Secondary check | 50k train / 10k test | https://www.cs.toronto.edu/~kriz/cifar.html | research-friendly |

**Other Resources (if applicable):**
- AutoAttack implementation: https://github.com/fra31/auto-attack
- RobustBench eval harness: https://github.com/RobustBench/robustbench

**Resource Estimate**:
- **Compute budget**: 
  - **Phase 1 (main test):** CIFAR-10, 3 conditions × 3 seeds = **9** TRADES training runs + AutoAttack evaluation.
  - **Phase 2 (generalization; only if Phase 1 is positive):** CIFAR-100, 2 conditions (TRADES baseline vs CReLUSplit) × 3 seeds = **6** additional runs.
  - **Training cost estimate:** adversarial training wall-clock varies substantially by implementation (PGD step count, AMP/compile, hardware). To stay conservative, budget **≤30 GPU-hours per TRADES run** (single GPU). Then Phase 1 training ≤270 GPU-hours and Phase 2 training ≤180 GPU-hours.
  - **Evaluation cost:**
    - **AutoAttack:** budget **≤1 GPU-hour per trained model** on CIFAR-10/100. Then Phase 1 AA eval ≤9 GPU-hours and Phase 2 AA eval ≤6 GPU-hours.
    - **CIFAR-10-C:** corruption evaluation is a single forward pass over the corrupted test set; budget **≤0.1 GPU-hour per model** (Phase 1 only) — negligible compared to training.
  - **Total worst-case** (Phase 1 + Phase 2): ≤466 GPU-hours, comfortably under the 768 GPU-hour cap.
- **GPU memory**: 1×A100 80GB (or 1×V100 32GB) should be sufficient for PreActResNet-18 on CIFAR-10 with batch size 128; reduce batch size if needed.
- **API usage**: none.

**Infrastructure constraints**:
- No browsers, search APIs, or human labeling required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| CIFAR-10 | 32×32 image classification (10 classes) | Clean accuracy; AutoAttack robust accuracy (\(\ell_\infty\), \(\varepsilon=8/255\)) | test | https://www.cs.toronto.edu/~kriz/cifar.html | RobustBench/AutoAttack |
| CIFAR-10-C | CIFAR-10 test set with 15 common corruptions × 5 severities (Hendrycks & Dietterich) | Mean corruption accuracy (avg over corruptions+severities) | test | https://zenodo.org/records/2535967 | RobustBench (corruptions threat model) |
| CIFAR-100 | 32×32 image classification (100 classes) | Clean accuracy; AutoAttack robust accuracy (\(\ell_\infty\), \(\varepsilon=8/255\)) | test | https://www.cs.toronto.edu/~kriz/cifar.html | RobustBench/AutoAttack |

**Evaluation Scripts:**
- Use AutoAttack + RobustBench evaluation code; report robust accuracy under the full AutoAttack suite.

**Download Links Checklist:**
- [x] All benchmark datasets have download links
- [x] All models have download links
- [x] Licenses are compatible with research use (verify before running)

### Main Results

#### Comparability Rules (CRITICAL)

All rows below must use:
- Same dataset split (CIFAR test)
- Same threat model (\(\ell_\infty\), \(\varepsilon=8/255\))
- Same TRADES training recipe and hyperparameters
- Same AutoAttack evaluation implementation

#### Results Table

| Method | Base Model | Benchmark | Clean acc (mean±std) | AutoAttack robust acc (mean±std) | Source | Notes |
|--------|------------|-----------|----------------------|----------------------------------|--------|-------|
| TRADES baseline | PreActResNet-18 + ReLU | CIFAR-10 | TBD | 48.37 (best) / 46.94 (last) | [FOMO Table 3](./references/The-effectiveness-of-random-forgetting-for-robust-generalization/sections/6\ Evaluation\ with\ AutoAttack.md) | AutoAttack, \(\ell_\infty\) \(\varepsilon=8/255\); PreActResNet-18; no std reported |
| DupSplit (control) | PreActResNet-18 (modified) | CIFAR-10 | TBD | TBD | - | Tests width/shape change without negative branch |
| **CReLUSplit (ours)** | PreActResNet-18 (modified) | CIFAR-10 | TBD | TBD | - | Tests sign-symmetric expansion |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| CReLUSplit vs DupSplit | Replace negative branch with duplication | If sign matters, CReLUSplit > DupSplit |

### Experimental Rigor

**Variance & Reproducibility:**
- Run all main conditions with **3 seeds** (e.g., `seeds=[0,1,2]`), report mean ± std.
- Prior work reports AutoAttack variability across 3 runs on CIFAR-10 is often around a few tenths of a point (e.g., \(\pm 0.3\) in some recent AT studies), so we treat ≥1.0% absolute improvements as practically meaningful.

**Validity & Controls:**
- **Confounder 1 (capacity/shape):** DupSplit matches tensor shapes and conv parameterization to CReLUSplit; difference isolates negative branch.
- **Confounder 2 (weak attacks / gradient masking):** Use AutoAttack instead of a single PGD setting.
- **Sanity check:** ERM (non-AT) model should have low robust accuracy under AutoAttack.

### Analysis (Optional)

- Train a sparse autoencoder on a mid-layer representation and compare the “feature activation blow-up” on adversarial vs clean inputs, following observations in the superposition paper. This is not a success criterion.

---

## Success Criteria

**Hypothesis** (directional):
- CReLUSplit achieves higher AutoAttack robust accuracy than DupSplit on CIFAR-10 at similar clean accuracy.

**Decision Rule** (concrete):
- **Proceed (sign matters):** On CIFAR-10, \(\mathrm{AA}(\text{CReLUSplit}) - \mathrm{AA}(\text{DupSplit}) \ge 1.0\%\) absolute and clean accuracy differs by ≤0.5% (mean over 3 seeds).
- **Proceed (strong win):** Additionally, \(\mathrm{AA}(\text{CReLUSplit}) - \mathrm{AA}(\text{Baseline}) \ge 1.0\%\) with clean accuracy within ≤0.5%.
- **Pivot (efficiency-only result):** If CReLUSplit beats DupSplit by ≥1.0% but is within 0.5% of Baseline robust accuracy while reducing conv params/FLOPs by ≥20%, report as an efficiency trade-off for robust models.
- **Refute:** If \(|\mathrm{AA}(\text{CReLUSplit}) - \mathrm{AA}(\text{DupSplit})| < 0.3\%\) (within expected noise) or CReLUSplit underperforms DupSplit.
- **Generalization check (non-adversarial):** If CReLUSplit also improves (or at least does not degrade) **CIFAR-10-C** corruption accuracy relative to DupSplit, this supports the claim that the intervention improves representation quality beyond a single \(\ell_\infty\) threat model.

---

## Impact Statement

If successful, SignSplit would provide a simple architectural knob for improving adversarial robustness that is orthogonal to training-loss innovations. Practitioners training robust CNNs (especially under compute constraints) could adopt sign-symmetric expansion blocks to gain robustness or preserve robustness while reducing compute.

---

## References

- [Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units](./references/Understanding-and-Improving-Convolutional-Neural-Networks-via-Concatenated-Rectified-Linear-Units/meta/meta_info.txt) - Shang et al., 2016
- [Max-min convolutional neural networks for image classification](./references/Max-Min-convolutional-neural-networks-for-image-classification/meta/meta_info.txt) - Blot et al., 2016
- [Beyond Interpretability: The Gains of Feature Monosemanticity on Model Robustness](./references/Beyond-Interpretability-The-Gains-of-Feature-Monosemanticity-on-Model-Robustness/meta/meta_info.txt) - Zhang et al., 2024
- [Adversarial Examples Are Not Bugs, They Are Superposition](./references/Adversarial-Examples-Are-Not-Bugs-They-Are-Superposition/meta/meta_info.txt) - Gorton & Lewis, 2025
- [Rethinking Lipschitz Neural Networks and Certified Robustness: A Boolean Function Perspective](./references/Rethinking-Lipschitz-Neural-Networks-and-Certified-Robustness-A-Boolean-Function-Perspective/meta/meta_info.txt) - Zhang et al., 2022
- [TRADES](https://arxiv.org/abs/1901.08573) - Zhang et al., 2019
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) - Madry et al., 2017
- [Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks](https://arxiv.org/abs/2003.01690) - Croce & Hein, 2020
- [RobustBench: a standardized adversarial robustness benchmark](https://openreview.net/forum?id=SSKZPJCt7B) - Croce et al., 2021
- [The Effectiveness of Random Forgetting for Robust Generalization (FOMO)](./references/The-effectiveness-of-random-forgetting-for-robust-generalization/meta/meta_info.txt) - Ramkumar et al., 2024
- [Parameterizing Activation Functions for Adversarial Robustness](./references/Parameterizing-Activation-Functions-for-Adversarial-Robustness/meta/meta_info.txt) - Dai et al., 2021
