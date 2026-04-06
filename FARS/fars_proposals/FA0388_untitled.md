# untitled

# CUSUM-ε: False-Alarm-Calibrated Rollback Thresholds for Runtime Training Stability Controllers

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, MLSys (or similar)
- **Core contribution**: Replace the fixed one-step innovation threshold in a probe-loss-based runtime rollback controller with a **CUSUM sequential test** that can be calibrated to a target false-rollback rate (average run length), and test whether this improves recovery from catastrophic update windows at matched nominal rollback rate.
- **Constraints**: Fully automated evaluation; ≤3 main experimental conditions; feasible within `internal_context/resource_budget.md` (≤768 A100 GPU-hours).

## Introduction

### Context and Motivation

Large neural networks are often trained for long horizons with stochastic optimizers, mixed precision, and complex data pipelines. In these settings, a **single destabilizing update** (e.g., from an outlier minibatch, transient numerical error, or sudden distributional change) can produce a large loss spike, divergence, or a long-lasting quality regression. When such events occur late in training, they can waste substantial compute and make results hard to reproduce.

A recent approach to improving training reliability is to add an external **runtime stability controller** that sits above any optimizer and decides whether to accept or reject each proposed parameter update based on an **external measurement signal**. Barak Or (2026) proposes using a small fixed **validation probe set** and an **innovation signal** (the deviation of the probe loss at the proposed parameters from an exponential moving average (EMA) of past accepted probe losses) to trigger **rollback** to the most recent safe snapshot when the innovation exceeds a threshold \(\epsilon\) **[Automatic Stability and Recovery for Neural Network Training](./references/Automatic-Stability-and-Recovery-for-Neural-Network-Training/meta/meta_info.txt)**.

However, Or’s controller leaves a practical open question: how should practitioners set \(\epsilon\)? A fixed threshold can be too conservative (frequent rollbacks that slow progress) or too lax (late detection). More importantly, a one-step threshold does not provide a clear connection to standard reliability targets like a desired **false-alarm rate**.

### The Problem

Or (2026) demonstrates that an innovation-threshold rollback controller can recover from injected catastrophic updates with low overhead, but the threshold \(\epsilon\) is set as a fixed tolerance without a statistical calibration procedure. In contrast, the classical sequential analysis literature provides procedures—such as **CUSUM**—that are explicitly designed to detect sustained mean shifts while controlling false alarms via quantities like **Average Run Length (ARL)**.

This creates a concrete reliability gap for training pipelines:

> We lack a statistically calibrated way to set rollback triggers for probe-loss innovation signals, and it is unclear whether a sequential test (CUSUM) yields a better stability–false-rollback trade-off than a one-step threshold when both are matched to the same nominal rollback rate.

### Key Insight and Hypothesis

**Key insight.** A catastrophic update window often produces not just a single extreme innovation, but a **sustained upward drift** in probe-loss innovation over multiple consecutive steps (as in Or’s multi-step gradient amplification perturbation). One-step thresholding only reacts to individual large innovations, while CUSUM accumulates evidence across time and is known to detect sustained shifts earlier at a matched false-alarm rate.

**Hypothesis.** A **CUSUM-on-innovation** rollback controller, calibrated to the same nominal rollback rate as Or’s fixed-\(\epsilon\) controller, will (i) reduce peak probe-loss degradation and/or (ii) reduce post-perturbation excess probe-loss area-under-curve (AUC) under multi-step catastrophic perturbations.

Why this could be wrong:
- The innovation signal may already be so low-noise and well-separated that CUSUM provides no advantage over a one-step threshold once both are calibrated.
- CUSUM introduces statefulness; if catastrophic events appear as single-step outliers rather than sustained drift, CUSUM may react later.
- Our fixed CUSUM reference value (\(k\)) encodes an assumption about the typical shift size (e.g., \(\ge 1\sigma\)); if the real shift magnitude differs, CUSUM may be mis-tuned.

---

## Proposed Approach

### Overview

We propose **CUSUM-\(\epsilon\)**: a drop-in replacement for the fixed innovation threshold in Or’s runtime stability controller.

- **Baseline controller (Or-\(\epsilon\))**: rollback when the one-step innovation \(\nu_t\) exceeds \(\epsilon\).
- **CUSUM controller (CUSUM-\(\epsilon\))**: rollback when a one-sided CUSUM statistic over standardized innovations exceeds a threshold \(h\).

Both controllers use the same probe set, the same rollback action (restore parameters + optimizer state), and the same EMA reference update. The only change is the decision rule.

### Method Details

#### Innovation signal (same as Or 2026)
Let \(y(\theta)\) be the probe loss on a fixed probe set \(P\) (a small held-out validation subset). Let \(\hat y_t\) be an EMA reference updated only on accepted steps with smoothing \(\alpha\in(0,1)\). For an optimizer-proposed candidate \(\theta_t^{\text{prop}}\):

\[
\nu_t = y(\theta_t^{\text{prop}}) - \hat y_t.
\]

#### Baseline accept/rollback rule (Or-\(\epsilon\))
Accept if \(\nu_t \le \epsilon\). Otherwise rollback to the last accepted snapshot \((\theta_{\text{safe}}, O_{\text{safe}})\) and keep \(\hat y\) unchanged.

#### CUSUM accept/rollback rule (CUSUM-\(\epsilon\))
We standardize innovations using nominal-training statistics \((\mu_0,\sigma_0)\) estimated on separate calibration runs:

\[
 r_t = \frac{\nu_t - \mu_0}{\sigma_0 + 10^{-8}}.
\]

Define a one-sided CUSUM statistic:

\[
S_t = \max\bigl(0,\; S_{t-1} + r_t - k\bigr),\quad S_0=0.
\]

Rollback if \(S_t > h\), then reset \(S_t\leftarrow 0\) after rollback (to avoid repeated triggers from the same event) and keep \(\hat y\) unchanged.

**Design choice (pre-committed):** set \(k=0.5\), corresponding to targeting detection of roughly \(\ge 1\sigma\) sustained upward shifts in standardized innovation. This is not tuned.

#### Calibration protocol (false-rollback control)
To avoid post-hoc tuning on failing trajectories, we pre-calibrate thresholds on **nominal (no-perturbation)** runs.

- Choose a target nominal rollback rate \(p_0\) per step. **Pre-commit:** \(p_0=0.2\%\), corresponding to ARL\(_0\) \(\approx 1/p_0 = 500\) steps (average number of steps before a false rollback under nominal training).
- Run \(N_{cal}=20\) nominal seeds (same model/dataset/optimizer/probe) for \(T=250\) steps and log \(\nu_t\) under unconditional training.
- Estimate \(\mu_0,\sigma_0\) from pooled \(\nu_t\) values across the nominal calibration runs.
- Set \(\epsilon\) as the empirical \((1-p_0)\)-quantile of \(\nu_t\) (one-step test).
- Set \(h\) by sweeping a small grid (e.g., \(h\in\{2,3,4,5,6,7,8\}\)) and choosing the \(h\) whose **offline** CUSUM alarm rate on the same nominal \(r_t\) streams is closest to \(p_0\).

This yields one calibrated scalar for each controller (\(\epsilon\) vs \(h\)) at a matched nominal rollback rate.

### Key Innovations

1. **Statistically calibrated rollback control for training**: replace an ad-hoc fixed innovation tolerance with a sequential test that can be tied to a target false-rollback rate.
2. **Stateful detection for sustained instability**: use evidence accumulation (CUSUM) rather than one-step thresholding for multi-step catastrophic update windows.
3. **Matched-rate comparison protocol**: compare Or-\(\epsilon\) vs CUSUM-\(\epsilon\) under an explicitly matched nominal rollback rate to avoid “wins” from simply rolling back more often.

---

## Related Work

### Field Overview

Training stability and reliability work spans: (i) **preventive** optimizer-level methods (gradient clipping, trust regions, adaptive optimizers), (ii) numerical debugging and sanitization (detecting NaNs/Infs and unstable ops), and (iii) runtime monitoring and recovery (checkpointing, detection + rollback). The Or (2026) runtime stability controller is closest to our setting: it introduces a probe-loss innovation signal to decide accept/rollback externally to the optimizer.

Sequential change detection provides a complementary lens: instead of testing each step independently, methods like CUSUM detect sustained distributional shifts with controlled false alarm rates and well-studied detection-delay trade-offs. While CUSUM has been applied widely in streaming ML drift detection, it has not been systematically tested as a drop-in replacement for innovation-threshold rollback controllers in neural network training.

### Related Papers

- **[Automatic Stability and Recovery for Neural Network Training](./references/Automatic-Stability-and-Recovery-for-Neural-Network-Training/meta/meta_info.txt)**: Introduces probe-loss innovation signals and accept/rollback recovery for catastrophic updates.
- **[Neural Network-based CUSUM for Online Change-point Detection](./references/Neural-Network-based-CUSUM-for-Online-Change-point-Detection/meta/meta_info.txt)**: Learns likelihood-ratio surrogates for CUSUM when post-change distributions are unknown; motivates CUSUM-style accumulation.
- **[Loss Spike in Training Neural Networks](./references/Loss-Spike-in-Training-Neural-Networks/meta/meta_info.txt)**: Analyzes why loss spikes can be structured and sometimes benign, cautioning against naive rollback on every spike.
- **[Automatically Detecting Numerical Instability in Machine Learning Applications via Soft Assertions](./references/Automatically-Detecting-Numerical-Instability-in-Machine-Learning-Applications-via-Soft-Assertions/meta/meta_info.txt)**: Learns instability conditions for numerical bugs, motivating data-driven runtime guards (different objective than rollback control).
- **[CUSUM: A cumulative sum scheme for control](https://doi.org/10.1093/biomet/41.1-2.100)**: Classic CUSUM change detection with ARL-based calibration.
- **[Page–Hinkley test](https://doi.org/10.1093/biomet/57.1.1)**: A related sequential mean-shift detector widely used in drift detection.
- **[A Sequential Probability Ratio Test](https://doi.org/10.1214/aoms/1177732187)**: Wald’s SPRT, a foundational sequential test (for comparison in the taxonomy).
- **[Gradient clipping](https://arxiv.org/abs/1211.5063)**: Early stabilization technique for RNNs; preventive rather than rollback.
- **[Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)**: The optimizer used in Or’s experiments and in our reproduction.
- **[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)**: ResNet architecture used in the vision benchmark.
- **[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)**: Benchmark dataset used in the vision benchmark.
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**: Transformer reference (Or’s sequence-model experiment uses a character-level Transformer).
- **[Edge of Stability](https://arxiv.org/abs/2108.04264)**: Empirical phenomenon where training operates near stability boundaries; relates to loss spikes.
- **[Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530)**: Highlights non-classical generalization behavior; motivates focusing on operational stability.
- **[Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412)**: Preventive method that modifies updates to improve robustness.
- **[Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407)**: Trajectory averaging that improves robustness but does not provide rollback recovery.
- **[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)**: Canonical trust-region method; conceptually related to limiting destructive updates.
- **[Learning Rate Warmup](https://arxiv.org/abs/1706.02677)**: Common stability heuristic; not a recovery mechanism.
- **[GRIST](https://arxiv.org/abs/2007.15537)**: Gradient-based numerical bug detection tool (related to detecting numerical instability conditions).
- **[DeepStability](https://arxiv.org/abs/2208.03733)**: Study of numerical instabilities in deep learning pipelines.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Preventive optimizer tweaks | Reduce probability of instability by modifying updates | Gradient clipping; SAM; trust regions | Standard training metrics | No recovery after a catastrophic update |
| Numerical sanitization | Detect/avoid NaNs/Infs/unstable ops | Soft Assertions; GRIST; DeepStability | Bug benchmarks; library tests | Not focused on training-progress recovery |
| Probe-loss rollback controllers | External measurement signal gates accept/rollback | Or (2026) | Controlled perturbation windows | Threshold \(\epsilon\) selection unclear |
| Sequential change detection | Accumulate evidence to detect sustained mean shifts | CUSUM; Page–Hinkley; SPRT; NN-CUSUM | ARL/EDD metrics; streaming benchmarks | Not commonly evaluated as training rollback rules |

### Closest Prior Work

**Or (2026)** **[Automatic Stability and Recovery for Neural Network Training](./references/Automatic-Stability-and-Recovery-for-Neural-Network-Training/meta/meta_info.txt)**: Proposes the innovation signal \(\nu_t\) from a probe-loss measurement and a fixed tolerance \(\epsilon\) to decide accept/rollback. Our proposal keeps the same innovation signal and rollback action but replaces the one-step threshold with a stateful sequential test calibrated to a target false-rollback rate.

**Page (1954) CUSUM**: Provides an ARL-calibrated sequential detector for sustained mean shifts. Our contribution is applying it to probe-loss innovation signals in a training rollback controller and testing whether it improves the stability/false-rollback trade-off.

**Gong et al. (2022) NN-CUSUM** **[Neural Network-based CUSUM for Online Change-point Detection](./references/Neural-Network-based-CUSUM-for-Online-Change-point-Detection/meta/meta_info.txt)**: Learns detection statistics for streaming change detection when distributions are unknown. We do not learn the detector; we use classical CUSUM as a minimal “stateful threshold” baseline in a training-stability setting.

**Zhang & Xu (2023) Loss Spikes** **[Loss Spike in Training Neural Networks](./references/Loss-Spike-in-Training-Neural-Networks/meta/meta_info.txt)**: Suggests some spikes may be part of training dynamics. This motivates focusing our claim on **catastrophic multi-step perturbation windows** and controlling nominal false rollbacks.

**Novelty Kill Search Summary:** Searched for combinations of “CUSUM probe loss rollback training”, “change point detection training stability rollback”, “innovation signal training rollback”, and checked local proposal corpus for “CUSUM”/“rollback”/“runtime stability controller”. No prior work directly combining CUSUM-style sequential tests with Or-style probe-loss rollback controllers was found as of 2026-02-28 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Or (2026) | One-step innovation threshold triggers rollback | \(\epsilon\) not statistically calibrated; one-step test | Replace one-step threshold with CUSUM accumulation | Earlier detection of sustained drift at matched false-rollback rate |
| Page (1954) CUSUM | ARL-calibrated mean-shift detection | Not evaluated as training rollback controller | Apply CUSUM to training innovation streams | Provides interpretable false-alarm control + better delay trade-off |
| NN-CUSUM (2022) | Learns a CUSUM-like detector for unknown post-change | Extra training complexity; different domain | Use classical CUSUM as minimal change | Keeps verification cheap and isolates “statefulness” effect |
| Loss spike analysis (2023) | Explains structured/benign spikes | Does not propose rollback control | Calibrate low nominal rollback rate | Avoid over-rolling-back benign spikes |

---

## Experiments

### Experimental Setup

**Main conditions (≤3):**
1. **No-controller baseline**: standard training loop; no accept/rollback.
2. **Or-\(\epsilon\)**: probe-loss innovation threshold with accept/rollback (Algorithm 1 in Or 2026).
3. **CUSUM-\(\epsilon\) (ours)**: same as (2) but rollback triggered by CUSUM statistic exceeding \(h\).

**Baseline Ladder (REQUIRED):**
- No controller
- One-step innovation threshold rollback (Or-\(\epsilon\))
- CUSUM-on-innovation rollback (ours)

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| ResNet-18 | ~11M | https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html | Standard torchvision implementation |
| Character-level Transformer | small | (implemented in code) | Same synthetic phrase task as Or (2026) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| CIFAR-10 | Vision training + probe sampling | 50k train / 10k test | https://www.cs.toronto.edu/~kriz/cifar.html | Research/academic use |
| Synthetic repeated phrase corpus | Sequence modeling | generated | N/A | N/A |

**Other Resources:**
- Or (2026) reference implementation (optional): https://github.com/BarakOr1/runtime-stability-controller

**Resource Estimate**:
- **Compute budget**: \< 10 A100 GPU-hours total (250-step runs; primarily overhead from repeated seeds and controller bookkeeping).
- **GPU memory**: \< 4GB for ResNet-18; \< 4GB for small Transformer.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Or’26 catastrophic recovery (vision) | ResNet-18 on CIFAR-10 with injected gradient amplification at step 120 | peak excess probe loss, excess-AUC, rollback count outside window, detection delay | train + fixed probe P from val | CIFAR-10 | custom PyTorch training loop (based on Or Algorithm 1) |
| Or’26 catastrophic recovery (sequence, sanity) | Char-level Transformer next-character prediction on synthetic phrase with same perturbation protocol | same metrics | generated | N/A | custom PyTorch loop |

**Perturbations (pre-committed):**
- **Step perturbation (Or’26 default):** at steps 120–129, multiply gradients by \(\zeta=300\).
- **Ramp perturbation (robustness):** at steps 120–129, use \(\zeta_i = 1 + (300-1)\cdot i/9\) for \(i=0..9\).

**Probe set (pre-committed):** \(|P|=16\) samples fixed once per seed (sampled from CIFAR-10 test split) and reused across all conditions for that seed.

**Seeds (pre-committed):** \(N=20\) seeds for each condition and perturbation type (match Or’26’s reporting scale).

**Primary metrics (computed from probe loss \(y_t\)):**
- Let \(y_{pre}\) be the mean probe loss over steps 110–119.
- **Peak excess loss**: \(\max_{t\ge 120} (y_t - y_{pre})\).
- **Excess AUC**: \(\sum_{t=120}^{250} \max(0, y_t - y_{pre})\).
- **Nominal rollback fraction**: fraction of rollback steps on *nominal runs* (no perturbation), used to confirm rate matching.
- **False rollbacks outside window**: number of rollbacks outside steps 120–129 on perturbed runs.

**Detection delay:** first rollback time minus 120. If a method never rolls back in a run, record as **no detection** for that seed.

### Main Results

#### Results Table

(All numbers **TBD**; to be produced by verification. Report mean±std across N=20 seeds.)

| Method | Base Model | Perturbation | Peak excess (↓) | Excess AUC (↓) | Rollbacks outside window (↓) | Detection delay (↓) | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| No controller | ResNet-18 | step | TBD | TBD | 0 | n/a | - | baseline |
| Or-\(\epsilon\) | ResNet-18 | step | TBD | TBD | TBD | TBD | - | \(\epsilon\) calibrated at \(p_0\) |
| **CUSUM-\(\epsilon\)** | ResNet-18 | step | TBD | TBD | TBD | TBD | - | \(k=0.5\); \(h\) calibrated at \(p_0\) |
| No controller | ResNet-18 | ramp | TBD | TBD | 0 | n/a | - | robustness |
| Or-\(\epsilon\) | ResNet-18 | ramp | TBD | TBD | TBD | TBD | - | robustness |
| **CUSUM-\(\epsilon\)** | ResNet-18 | ramp | TBD | TBD | TBD | TBD | - | robustness |

(Sequence-model table is reported as a sanity check; not used for the main decision unless results contradict.)

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| CUSUM-\(\epsilon\) with \(k=0\) | remove reference-value subtraction | More false rollbacks; shows need for targeting sustained shifts |

### Experimental Rigor

- **Fairness**: identical initialization, data order, probe set, and perturbation schedule across conditions per seed.
- **Rate matching check**: verify Or-\(\epsilon\) and CUSUM-\(\epsilon\) achieve similar nominal rollback rates (within ±20% relative error around \(p_0\)) on held-out nominal seeds.
- **Confound control (rollback-more-often)**: matched nominal rollback rate calibration prevents “win” from simply rolling back more.

---

## Success Criteria

**Hypothesis** (directional): At matched nominal rollback rate \(p_0\), CUSUM-\(\epsilon\) reduces peak excess probe loss and/or excess AUC under the Or’26 catastrophic perturbation relative to Or-\(\epsilon\).

**Decision Rule** (concrete):
- **Proceed** if, on ResNet-18/CIFAR-10 with step perturbation, CUSUM-\(\epsilon\) beats Or-\(\epsilon\) on **either** peak excess loss or excess AUC by a margin larger than the across-seed std (non-overlapping 1-std intervals), while keeping rollbacks outside the perturbation window ≤1% of steps on average.
- **Pivot** if CUSUM-\(\epsilon\) ≈ Or-\(\epsilon\): conclude that one-step thresholding is sufficient for this innovation signal, and test a simpler statistical calibration for \(\epsilon\) (e.g., using robust quantiles) as the main takeaway rather than CUSUM.
- **Refute** if CUSUM-\(\epsilon\) does not improve peak excess/AUC (within noise) or if it increases rollbacks outside the window substantially at matched nominal rate.

---

## Impact Statement

If successful, this work would provide a simple, optimizer-agnostic recipe for setting rollback triggers in training reliability controllers using standard sequential analysis tools. Practitioners could specify a desired false-rollback rate (ARL) and obtain a controller that better detects sustained instability without increasing unnecessary rollbacks, reducing wasted compute from rare catastrophic update windows.

---

## References

- [Automatic Stability and Recovery for Neural Network Training](./references/Automatic-Stability-and-Recovery-for-Neural-Network-Training/meta/meta_info.txt) - Barak Or, 2026
- [Neural Network-based CUSUM for Online Change-point Detection](./references/Neural-Network-based-CUSUM-for-Online-Change-point-Detection/meta/meta_info.txt) - Gong et al., 2022
- [Loss Spike in Training Neural Networks](./references/Loss-Spike-in-Training-Neural-Networks/meta/meta_info.txt) - Zhang & Xu, 2023
- [Automatically Detecting Numerical Instability in Machine Learning Applications via Soft Assertions](./references/Automatically-Detecting-Numerical-Instability-in-Machine-Learning-Applications-via-Soft-Assertions/meta/meta_info.txt) - Sharmin et al., 2025
- [CUSUM: A cumulative sum scheme for control](https://doi.org/10.1093/biomet/41.1-2.100) - Page, 1954
- [Page–Hinkley test](https://doi.org/10.1093/biomet/57.1.1) - Hinkley, 1970
- [A Sequential Probability Ratio Test](https://doi.org/10.1214/aoms/1177732187) - Wald, 1945
- [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063) - Pascanu et al., 2013
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter, 2017
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - He et al., 2016
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412) - Foret et al., 2020
- [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407) - Izmailov et al., 2018
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) - Schulman et al., 2015
- [On Large-Batch Training for Deep Learning](https://arxiv.org/abs/1609.04836) - Keskar et al., 2016
- [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530) - Zhang et al., 2017
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) - Shallue et al., 2019
- [Edge of Stability](https://arxiv.org/abs/2108.04264) - Cohen et al., 2021
- [How does batch size affect the geometry of the loss landscape?](https://arxiv.org/abs/1906.07485) - Jastrzebski et al., 2019
- [GRIST: An Automated Test Generator for Numerical Stability in Deep Learning](https://arxiv.org/abs/2007.15537) - Yan et al., 2020
- [DeepStability: A Study of Numerical Instabilities in Deep Learning](https://arxiv.org/abs/2208.03733) - Kloberdanz et al., 2022
