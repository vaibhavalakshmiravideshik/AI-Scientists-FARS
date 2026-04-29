# untitled

# Prototype-Debiased Latent Alignment for Class-Imbalanced Context Sets in Subject-Independent EEG Decoding

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, IEEE TNNLS / JNE (or similar top ML / neural engineering venues)

## Introduction

### Context and Motivation

Electroencephalography (EEG) based brain-computer interfaces (BCIs) translate brain signals into discrete commands by classifying short windows of multi-channel electrical activity recorded from the scalp (e.g., left hand vs right hand vs feet motor tasks). In practice, EEG decoders often require per-user calibration because EEG signals vary strongly across individuals due to anatomy, electrode placement, and non-stationarity.

A popular line of work reduces calibration by treating each user (subject) as a new domain and performing domain adaptation without target labels. A particularly simple and effective approach is to adapt normalization statistics (mean/variance) at test time, motivated by the observation that normalization layers encode substantial domain information.

Bakas et al. propose **Latent Alignment** for subject-independent EEG decoding: a subject-wise normalization applied at multiple layers, where statistics are computed from a context set of trials and applied using a permutation-equivariant deep set (a network that processes an unordered set of trials). Latent Alignment significantly improves balanced accuracy (average per-class recall; higher is better) on multiple EEG tasks, but the paper also identifies an important deployment failure mode: the method can collapse when the test-time context set is class-imbalanced.

### The Problem

**Latent Alignment is sensitive to class imbalance in the context set used for computing alignment statistics.** In the Latent Alignment paper's analysis of the PhysioNet motor execution task (3 classes, context size n=21), classification accuracy falls toward chance-level when the context set contains mostly one class. The authors define a multinomially weighted accuracy (WA) over all class compositions and show that training with randomly unbalanced batches does not significantly improve WA.

This failure mode matters for real BCI deployments because class imbalance is common in online and asynchronous (self-paced) BCIs. In asynchronous motor imagery BCIs, the system continuously monitors EEG and must distinguish intentional control states from idle/non-control periods; idle-state detection has been studied since at least the BCI Competition III task ([Zhang et al., 2007](https://sccn.ucsd.edu/~yijun/pdfs/CIN07.pdf)) and continues in recent work (e.g., [SWPC, 2024](https://arxiv.org/abs/2412.09006)). These settings naturally create long stretches where recent-trial context windows are dominated by one state, as well as repeated-command patterns.

Moreover, class imbalance is inherent in some BCI paradigms even offline. For example, in the OpenBMI P300 event-related potential (ERP) task used by Bakas et al., training batches use a fixed ratio of 5 non-target trials for every target trial (paper Sec. 3.2.3), reflecting the rarity of target events in P300 spellers.

While the Latent Alignment paper notes that Euclidean Alignment is relatively robust to class-imbalanced context sets (paper Sec. 4.4), it is not a drop-in replacement for late-layer latent alignment: on PhysioNet motor execution, Euclidean Alignment is 0.625 vs 0.641 for Latent Alignment in balanced accuracy (Table 2; higher is better), and on sleep staging Euclidean Alignment slightly underperforms the baseline (0.725 vs 0.732). Therefore, improving Latent Alignment's robustness can matter in practice even when a more robust but weaker alternative exists.

The closest existing mitigation ideas are largely from vision test-time adaptation (TTA):

- **[Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization](./references/Towards-Real-World-Test-Time-Adaptation-Tri-Net-Self-Training-with-Balanced-Normalization/meta/meta_info.txt)** proposes class-balanced batch normalization statistics updated via pseudo-labels.
- **[Channel-Selective Normalization for Label-Shift Robust Test-Time Adaptation](./references/Channel-Selective-Normalization-for-Label-Shift-Robust-Test-Time-Adaptation/meta/meta_info.txt)** avoids catastrophic failures under label shift by adapting only some channels, especially in early layers.

However, these methods do not directly address the specific mechanism of **deep-set subject-wise latent normalization** used by Latent Alignment in EEG decoding.

### Key Insight and Hypothesis

**Hypothesis (mechanistic):** At late alignment layers (close to the classifier), latent features are approximately class-separable. If features decompose as

- x = mu_c + delta_s + eps,

where mu_c is a class prototype (shared across subjects), delta_s is a subject-specific offset, and eps is noise, then the context-set mean under class proportions p is

- mu_ctx = sum_c p_c * mu_c + delta_s.

Latent Alignment subtracts mu_ctx when standardizing the context set. If p is imbalanced, subtracting sum_c p_c * mu_c shifts features relative to the classifier in a way that differs from the balanced-mixture mean (1/C) * sum_c mu_c, effectively moving decision boundaries and reducing accuracy.

**Proposed fix:** Precompute source-trained class prototypes {mu_c} in the pre-final-alignment feature space, estimate the class proportions in the current context set (without labels) from model predictions, and correct the mean used in the final alignment layer to match the balanced-mixture mean.

Why we could be wrong:
- Class prototypes may not transfer well across subjects (subject shift may rotate class geometry).
- The predicted class proportions may be least reliable exactly when Latent Alignment collapses.
- The failure may be driven by variance (std) distortion or higher-order effects, not primarily the mean shift.

---

## Proposed Approach

### Overview

We propose **Prototype-Debiased Latent Alignment (PD-LA)**, a training-free (inference-time) modification to Latent Alignment that targets its class-imbalance failure mode.

PD-LA changes only one part of Latent Alignment: the mean used for standardization at the final alignment layer (the layer closest to the classifier). Instead of using the raw context mean mu_ctx, PD-LA uses a debiased mean mu_used that subtracts the estimated prototype-mixture bias induced by the context class proportions.

We include two variants:

1. **PD-LA (pred)**: uses an estimated class proportion vector p_hat derived from the model's predicted probabilities on the context set.
2. **PD-LA (oracle)**: uses the true class proportions of the context set (labels) as a diagnostic upper bound.

The oracle variant is not a deployable method, but it is critical to distinguish:
- the mechanism being correct (oracle works), from
- the prior estimation being the bottleneck (oracle works but pred does not).

### Method Details

#### Base setting and notation

- C: number of classes (C=3 for PhysioNet motor tasks)
- n: context set size (n=21)
- f_theta: trained EEG decoder with Latent Alignment
- h(x): penultimate representation for a trial, in the feature space *just before* the final Latent Alignment standardization
- For a context set of trials X_ctx = {x_1, ..., x_n} from the same test subject:
  - mu_ctx = (1/n) * sum_i h(x_i)

#### Step 1: compute class prototypes on source training data

For each training fold, after training f_theta with Latent Alignment, compute class prototypes in the pre-final-alignment space:

- mu_c = E[h(x) | y=c] estimated as the mean of h(x) over all training trials with label c.

Also compute the balanced-mixture mean:

- mu_bar = (1/C) * sum_c mu_c.

This prototype computation is a single forward pass over the training set and adds negligible overhead.

#### Step 2: estimate context-set class proportions (label-free)

Compute a raw prior estimate p_raw by averaging predicted probabilities over the context set:

- p_raw = (1/n) * sum_i softmax(g(h_partial(x_i))),

where h_partial and g are computed with the *final alignment layer disabled* (align only up to the penultimate layer). This reduces circularity because the prior estimator does not depend on the potentially brittle final alignment layer.

Apply a parameter-free reliability shrinkage based on mean predictive entropy:

- H_bar = (1/n) * sum_i H(p_i)
- r = clip(1 - H_bar / log(C), 0, 1)
- p_hat = r * p_raw + (1-r) * uniform(C)

This ensures PD-LA reduces to vanilla Latent Alignment when the model is maximally uncertain.

#### Step 3: debias the final-layer alignment mean

Compute the prototype-mixture bias term:

- b(p_hat) = sum_c (p_hat_c - 1/C) * mu_c.

Then set the mean used for the final alignment layer to:

- mu_used = mu_ctx - b(p_hat).

Intuition: mu_used removes the class-mixture component and keeps the subject offset delta_s, so the final standardization behaves as if the context set had a balanced class mixture.

All other parts of Latent Alignment remain unchanged (including the per-dimension std estimate from the context set).

#### Oracle diagnostic

For PD-LA (oracle), use p_hat := p_true (true class proportions of the context set) in the above correction. This uses labels only for analysis.

### Key Innovations

- **Mechanism-driven correction**: explicitly models the class-mixture term inside deep-set subject-wise latent normalization and corrects it in closed form.
- **Single-layer, inference-time change**: only modifies the final alignment layer mean, keeping the method simple and minimizing regressions on balanced contexts.
- **Built-in falsifier**: includes an oracle-prior variant and a correlation diagnostic to directly test whether the mean-shift mechanism explains the collapse.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) subject-independent EEG decoding, (ii) statistical alignment / normalization-based domain adaptation, and (iii) test-time adaptation under class imbalance / label shift.

In EEG, transfer learning methods include input-space covariance alignment (Euclidean/Riemannian alignment), adversarial domain adaptation, and meta-learning. Latent Alignment is notable for applying subject-wise normalization repeatedly throughout the network, including the final classification space, and for using a deep-set formulation to process trial sets.

In vision TTA, many approaches update batch normalization statistics or affine parameters at test time (e.g., AdaBN/TTN/TENT) and stabilize adaptation under non-i.i.d. streams (e.g., NOTE/CoTTA). More recently, several works explicitly address class imbalance or label shift during test-time adaptation (e.g., TRIBE, Hybrid-TTN, PLF). These works motivate that normalization statistics can be systematically biased by class composition, but do not directly resolve Latent Alignment's deep-set EEG setting.

### Related Papers

(>=20 papers; local paths used when available)

- **[Latent Alignment with Deep Set EEG Decoders](./references/Latent-Alignment-with-Deep-Set-EEG-Decoders/meta/meta_info.txt)**: Proposes subject-wise deep-set latent normalization for EEG decoding and identifies the class-imbalanced context failure mode.
- **[Structured Prototype-Guided Adaptation for EEG Foundation Models (SCOPE)](https://arxiv.org/abs/2602.17251)**: Uses prototypes and confidence-aware pseudo-labeling to adapt EEG foundation models under limited supervision; conceptually overlaps with prototype-based debiasing but targets fine-tuning/adapters rather than test-time normalization statistics.
- **[Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization](./references/Towards-Real-World-Test-Time-Adaptation-Tri-Net-Self-Training-with-Balanced-Normalization/meta/meta_info.txt)**: Introduces class-balanced normalization for globally imbalanced test streams in vision.
- **[Channel-Selective Normalization for Label-Shift Robust Test-Time Adaptation](./references/Channel-Selective-Normalization-for-Label-Shift-Robust-Test-Time-Adaptation/meta/meta_info.txt)**: Selectively adapts normalization channels to avoid catastrophic failures under label shift.
- **[Less is More: Pseudo-Label Filtering for Continual Test-Time Adaptation](./references/Less-is-More-Pseudo-Label-Filtering-for-Continual-Test-Time-Adaptation/meta/meta_info.txt)**: Stabilizes continual TTA with pseudo-label filtering and class prior alignment.
- **[TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation](./references/TTN-A-Domain-Shift-Aware-Batch-Normalization-in-Test-Time-Adaptation/meta/meta_info.txt)**: Interpolates source and test normalization statistics to handle realistic shifts and small batches.
- **[Revisiting Batch Normalization For Practical Domain Adaptation (AdaBN)](./references/Revisiting-Batch-Normalization-For-Practical-Domain-Adaptation/meta/meta_info.txt)**: Classic result that BN statistics encode domain information; update BN stats for adaptation.
- **[Batch Normalization](https://arxiv.org/abs/1502.03167)**: Standard BN formulation and affine parameters.
- **[Deep Sets](https://arxiv.org/abs/1703.06114)**: Characterizes permutation-invariant/equivariant functions; basis for deep-set EEG decoding.
- **[EEGNet](https://arxiv.org/abs/1611.08024)**: Compact CNN baseline for EEG classification.
- **[DeepSleepNet](https://arxiv.org/abs/1703.04046)**: Early CNN-RNN model for sleep staging from raw EEG.
- **[Test-Time Entropy Minimization (TENT)](https://arxiv.org/abs/2006.10726)**: Test-time adaptation by entropy minimization, often by updating BN affine params.
- **[SHOT](https://arxiv.org/abs/2002.08546)**: Source-free domain adaptation using information maximization and pseudo-labeling.
- **[EATA](https://arxiv.org/abs/2204.02610)**: Sample selection + anti-forgetting regularization for test-time adaptation.
- **[SAR](https://arxiv.org/abs/2302.12400)**: Stabilizes entropy minimization under wild shifts; highlights imbalanced label shift as a failure mode.
- **[CoTTA](https://arxiv.org/abs/2203.13591)**: Continual test-time adaptation with teacher-student and stochastic restoration.
- **[NOTE](https://papers.neurips.cc/paper_files/paper/2022/hash/ae6c7dbd9429b3a75c41b5fb47e57c9e-Abstract-Conference.html)**: Addresses temporal correlation in continual test-time adaptation.
- **[T3A](https://proceedings.neurips.cc/paper_files/paper/2021/file/1415fe9fea0fa1e45dddcff5682239a0-Paper.pdf)**: Backprop-free test-time classifier adjustment using prototypes.
- **[Detecting and Correcting for Label Shift with Black Box Predictors (BBSE)](https://arxiv.org/abs/1802.03916)**: Confusion-matrix-based label shift estimation.
- **[A Unified View of Label Shift Estimation](https://papers.nips.cc/paper_files/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf)**: Likelihood-based label shift estimation analysis; calibration matters.
- **[Saerens et al. 2002](https://www.researchgate.net/publication/11608620_Adjusting_the_Outputs_of_a_Classifier_to_New_a_Priori_Probabilities_A_Simple_Procedure)**: Classic EM-based class prior adjustment under prior shift.
- **[Domain-Specific Batch Normalization (DSBN)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)**: Domain-specific BN parameters for unsupervised domain adaptation.
- **[Team Cogitat at NeurIPS 2021: BEETL competition paper](https://arxiv.org/abs/2202.03267)**: Competition context and latent alignment motivation.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| EEG alignment (input/covariance) | Align covariance statistics across subjects/sessions (often unsupervised) | Euclidean/Riemannian alignment (survey: https://arxiv.org/abs/2502.09203) | Motor imagery/execution, ERP, sleep staging | Often limited to input-level alignment; may not align latent/classification space |
| Latent/statistical alignment in deep nets | Apply subject-wise normalization inside the network, including late layers | [Latent Alignment](./references/Latent-Alignment-with-Deep-Set-EEG-Decoders/meta/meta_info.txt), AdaBN | Subject-independent EEG decoding | Can be brittle when context-set statistics are biased (e.g., class imbalance) |
| Test-time normalization (vision) | Replace/interpolate BN statistics at test time, sometimes with entropy minimization | [AdaBN](./references/Revisiting-Batch-Normalization-For-Practical-Domain-Adaptation/meta/meta_info.txt), [TTN](./references/TTN-A-Domain-Shift-Aware-Batch-Normalization-in-Test-Time-Adaptation/meta/meta_info.txt), [TENT](https://arxiv.org/abs/2006.10726) | CIFAR-C, ImageNet-C | Sensitive to label shift and non-i.i.d. streams without additional safeguards |
| Label shift / imbalance robust TTA | Explicitly handle changing class priors and class imbalance during adaptation | [TRIBE](./references/Towards-Real-World-Test-Time-Adaptation-Tri-Net-Self-Training-with-Balanced-Normalization/meta/meta_info.txt), [Hybrid-TTN](./references/Channel-Selective-Normalization-for-Label-Shift-Robust-Test-Time-Adaptation/meta/meta_info.txt), [PLF](./references/Less-is-More-Pseudo-Label-Filtering-for-Continual-Test-Time-Adaptation/meta/meta_info.txt) | Corruptions + imbalanced streams | Mostly evaluated in vision; mechanisms not tailored to deep-set EEG latent normalization |
| Prior estimation under shift | Estimate target class priors from predictions and calibration data | [BBSE](https://arxiv.org/abs/1802.03916), [Unified view](https://papers.nips.cc/paper_files/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf), Saerens 2002 | Synthetic + vision | Requires calibration / invertibility; may fail under covariate shift |

### Closest Prior Work

- **Latent Alignment (Bakas et al.)**: Applies subject-wise mean/std standardization at multiple layers including classification space, improving subject-independent EEG decoding but collapsing under class-imbalanced context sets.
- **TRIBE (Su et al.)**: Maintains per-class BN statistics and reweights them to produce balanced normalization in imbalanced test streams; designed for continual vision TTA and relies on pseudo-labels for class-wise stats.
- **Hybrid-TTN (Vianna et al.)**: Avoids TTN failures under label shift by selectively adapting some normalization channels (especially early layers) based on class sensitivity scoring.
- **PLF (Zheng et al.)**: Filters pseudo-labels and aligns class priors in continual TTA; focuses on stabilizing self-training rather than correcting the specific bias inside context-set mean estimation.
- **SCOPE (arXiv:2602.17251)**: Uses prototypes + confidence-aware pseudo-label fusion for adapting EEG foundation models; related in its use of prototypes for debiasing, but differs in that it fine-tunes adapters and does not address deep-set context-set normalization statistics.

**Novelty Kill Search Summary:** Searched for exact combinations such as "latent alignment EEG class imbalance context set", "2311.17968 follow-up", and "prototype debiased batch normalization class prior" and found no prior work that applies a class-prototype mean correction to Latent Alignment's deep-set subject-wise normalization to address the imbalanced-context failure mode (as of 2026-02-22). Closest ideas are balanced BN (TRIBE) and channel-selective normalization (Hybrid-TTN), but they target different mechanisms and domains.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [Latent Alignment](./references/Latent-Alignment-with-Deep-Set-EEG-Decoders/meta/meta_info.txt) | Subject-wise deep-set normalization at multiple layers | Fails under class-imbalanced context sets; training with unbalanced batches does not fix | Correct the context mean used by final alignment layer using class prototypes + estimated priors | Directly targets the hypothesized cause of collapse at late, class-separable layers |
| [TRIBE](./references/Towards-Real-World-Test-Time-Adaptation-Tri-Net-Self-Training-with-Balanced-Normalization/meta/meta_info.txt) | Balanced BN via per-class stats with pseudo-labels in continual vision TTA | Requires maintaining/estimating per-class BN moments online; may be heavy for EEG deep-set context normalization | Use a closed-form prototype-mixture correction with a single prior vector | Simpler; does not require per-class BN queues; tailored to context-set mean bias |
| [Hybrid-TTN](./references/Channel-Selective-Normalization-for-Label-Shift-Robust-Test-Time-Adaptation/meta/meta_info.txt) | Selectively adapt normalization channels to be label-shift robust | Focuses on which channels/layers to adapt, not on debiasing context statistics in deep sets | Keep all channels but correct the mean statistic that induces decision boundary shift | Addresses a different failure mechanism; potentially complementary but not redundant |
| [TTN](./references/TTN-A-Domain-Shift-Aware-Batch-Normalization-in-Test-Time-Adaptation/meta/meta_info.txt) | Interpolates source and test normalization stats | Primarily targets covariate shift; label shift can cause failures | Apply an explicit class-mixture correction inside Latent Alignment | Explicitly removes label-composition bias rather than interpolating blindly |
| [AdaBN](./references/Revisiting-Batch-Normalization-For-Practical-Domain-Adaptation/meta/meta_info.txt) | Replace BN stats with target stats | Not designed for deep-set trial context normalization; no imbalance robustness | Prototype-based debiasing at the final alignment layer | Uses class structure to correct biased context statistics |

---

## Experiments

### Experimental Setup

**Goal:** Determine whether debiasing the final-layer context mean using class prototypes improves robustness to class-imbalanced context sets in Latent Alignment.

**Datasets / tasks:**

- **Primary benchmark (imbalance stress test)**: PhysioNet motor execution (ME), a 3-class EEG classification task (left fist, right fist, both feet) evaluated in Bakas et al.
- **Secondary benchmark (generalization check)**: PhysioNet Sleep Cassette sleep stage classification, where Bakas et al. classify sleep stages from 30-second EEG segments (stages 3 and 4 are combined due to low counts; paper Sec. 3.2.2 and Fig. 1).

**Artifact control for motor decoding (must match the paper):** Exclude the first 1 second of each 4.1s PhysioNet motor trial due to cue-locked eye-movement artifacts; use the 1-4s window (paper Sec. 4.1; Table 1).

**Models and training protocol (match Bakas et al.):**

- Official code: https://github.com/StylianosBakas/LatentAlignment

**PhysioNet ME** (paper Sec. 3.2.1):
- Preprocessing: 4-40 Hz bandpass; 60 Hz notch; common average reference.
- Model: EEGNet (a compact convolutional neural network designed for EEG classification).
- Training: 100 epochs; batch constructed from 4 subjects x 12 trials per subject (batch size 48).

**PhysioNet Sleep** (paper Sec. 3.2.2):
- Preprocessing: 0.1-45 Hz bandpass; 100 Hz sampling; 30-second trials.
- Model: DeepSleep (a convolutional model for sleep staging) with batch normalization layers added after each convolutional layer, as in the paper.
- Training: 5 epochs; batch size 256 (64 trials per subject session, 4 sessions per batch).

**Validation (both benchmarks):** 10-fold subject-independent cross-validation as in the paper (subjects in validation are unseen during training; paper Sec. 3.2).

**Methods (main 3 conditions):**

- A) **Latent Alignment (LA)**: unchanged, as in the paper.
- B) **PD-LA (pred)**: prototype-debiased mean using predicted class proportions p_hat.
- C) **PD-LA (oracle)**: prototype-debiased mean using true context-set class proportions (diagnostic).

Optional ablation (run only if needed to interpret failures; do not block the main decision):
- PD-LA (pred, no shrinkage): use p_raw without entropy shrinkage.

**How to generate class-imbalanced context sets (stress test):**

Replicate the paper's setup (Sec. 4.4): context size n=21, C=3. For each fold and each held-out test subject:

1. For every integer composition (i,j,k) with i+j+k=21, construct a context batch of 21 trials for that subject by sampling i trials from class 1, j from class 2, k from class 3 (without replacement when possible; otherwise sample with replacement).
2. Run inference on that 21-trial batch with the chosen method (LA / PD-LA pred / PD-LA oracle), and compute balanced accuracy on that batch (acc_ijk). Here balanced accuracy means the average recall across classes (higher is better), which avoids being dominated by the majority class under imbalance.
3. Aggregate acc_ijk across subjects and folds.

Compute the paper's **weighted accuracy (WA)**:

- WA = sum_{i,j,k} MultinomialPMF(i,j,k; n=21, p=(1/3,1/3,1/3)) * acc_ijk.

Also report the balanced-composition accuracy acc_(7,7,7) as the "best case" context.

**Mechanism diagnostics (must be logged):**

- For each composition (i,j,k), compute the prototype-mixture shift magnitude:
  - S_ijk = || sum_c ((count_c/n) - 1/C) * mu_c ||_2.
- For vanilla LA, compute accuracy drop D_ijk = acc_(7,7,7) - acc_ijk.
- Report Pearson correlation corr(S_ijk, D_ijk) for LA, and for PD-LA (oracle).

**Baseline numbers to copy from the paper (balanced test distribution; Table 2):**

These are balanced-accuracy numbers (mean (std) across folds; higher is better), not WA under imbalanced context. They are included to anchor the baseline protocol and to quantify the accuracy trade-off between Euclidean Alignment and Latent Alignment.

PhysioNet motor execution (EEGNet: ME, balanced training):
- Baseline (no alignment): 0.521 (0.060)
- Euclidean Alignment: 0.625 (0.057)
- Adaptive BatchNorm: 0.630 (0.055)
- Latent Alignment: 0.641 (0.043)

PhysioNet sleep staging (DeepSleep: Sleep Stages):
- Baseline (no alignment): 0.732 (0.031)
- Euclidean Alignment: 0.725 (0.020)
- Adaptive BatchNorm: 0.731 (0.031)
- Latent Alignment: 0.749 (0.024)

Source: [Latent Alignment paper](./references/Latent-Alignment-with-Deep-Set-EEG-Decoders/sections/4.2 Classification Performance.md)

**Resource Estimate**

- **Compute budget**: <= 200 GPU-hours (EEGNet is small; PhysioNet ME uses 10-fold training + evaluation over ~253 context compositions per fold; plus one additional 10-fold run on PhysioNet Sleep for generalization). Even a 3-5x underestimate remains within 768 GPU-hours.
- **GPU memory**: <= 8 GB for EEGNet; fits easily on available GPUs.
- **API usage**: None.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| PhysioNet ME | 3-class motor execution classification across subjects (left fist, right fist, both feet) | Balanced accuracy (average per-class recall); WA over imbalanced context compositions; correlation diagnostic r(S,D) | 10-fold subject CV as in Latent Alignment | https://physionet.org/content/eegmmidb/ | Official LatentAlignment GitHub + small evaluation additions |
| PhysioNet Sleep (Sleep-EDF) | Sleep stage classification from whole-night EEG; Bakas et al. segment into 30-second trials and classify stages (with stages 3 and 4 merged) | Balanced accuracy (mean over classes) | 10-fold subject CV as in Latent Alignment | https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/ | Official LatentAlignment GitHub (DeepSleep model + alignment code) + small evaluation additions |

### Main Results

Primary results are reported on PhysioNet ME with the imbalanced-context stress test.

| Method | Base Model | Benchmark | Metric 1: WA (mean+-std over folds) | Metric 2: acc_(7,7,7) (mean+-std) | Source | Notes |
|---|---|---|---|---|---|---|
| LA | EEGNet | PhysioNet ME | TBD | TBD | This work | Needs run (paper shows as Fig 5 heatmap, not table) |
| PD-LA (oracle) | EEGNet | PhysioNet ME | TBD | TBD | This work | Diagnostic upper bound |
| PD-LA (pred) | EEGNet | PhysioNet ME | TBD | TBD | This work | Proposed method |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| PD-LA (pred, no shrinkage) | p_hat := p_raw (no entropy-based shrinkage) | If p_raw is noisy under collapse, shrinkage should improve stability |

### Experimental Rigor

**Variance & Reproducibility:**
- Use the paper's fixed 10 subject-independent folds; treat folds as 10 replicates.
- Fix RNG seeds per fold for training and for context-set sampling. Use a fixed seed list such as `seeds=[42, 123, 456]` for any additional stochasticity beyond the paper's fixed 10 folds (e.g., context-set sampling with replacement).

**Validity & Controls:**
- Artifact control: use 1-4s window only (paper Sec. 4.1).
- Sanity check: reproduce Table 2 Latent Alignment balanced accuracy within +/- 0.01 absolute; otherwise stop and fix training protocol mismatch.
- Ensure all methods share the same trained weights per fold (PD-LA is inference-only modification).

---

## Success Criteria

**Hypothesis (directional):**
- PD-LA (oracle) substantially recovers WA under imbalanced contexts, supporting the prototype-mixture mean-shift mechanism.
- PD-LA (pred) recovers a meaningful fraction of the oracle gain without hurting balanced-context accuracy.

**Decision Rule (concrete):**

Let:
- A_bal = acc_(7,7,7) for vanilla LA
- A_WA = WA for vanilla LA
- O_WA = WA for PD-LA (oracle)
- P_WA = WA for PD-LA (pred)

1. **Mechanism validation (oracle effect size):**
   - Proceed only if (O_WA - A_WA) / (A_bal - A_WA) >= 0.60.
   - Refute the mean-shift mechanism if the recovery fraction is < 0.30.

2. **Mechanism structure (correlation test):**
   - For vanilla LA, require Pearson corr(S_ijk, D_ijk) >= 0.50.
   - After oracle correction, require corr <= 0.25 OR a drop of at least 0.25.

3. **Practical success (predicted prior):**
   - Require P_WA - A_WA >= 0.03 absolute.
   - Require P_WA - A_WA >= 0.50 * (O_WA - A_WA).
   - Require |acc_(7,7,7)_{pred} - A_bal| <= 0.01.

4. **Pivot condition:**
   - If oracle works but pred fails, pivot to improving prior estimation (e.g., EM-style prior shift estimation with calibration) rather than changing the correction formula.

---

## Impact Statement

If successful, this work would make Latent Alignment substantially more reliable in online BCI settings by removing a strong requirement that adaptation context windows be class-balanced. Practitioners deploying subject-independent EEG decoders could use deep-set latent alignment with less manual control over trial sequences, reducing calibration burden and increasing robustness to real user behavior.

---

## References

- [Latent Alignment with Deep Set EEG Decoders](./references/Latent-Alignment-with-Deep-Set-EEG-Decoders/meta/meta_info.txt) - Bakas et al.
- [Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization](./references/Towards-Real-World-Test-Time-Adaptation-Tri-Net-Self-Training-with-Balanced-Normalization/meta/meta_info.txt) - Su et al.
- [Channel-Selective Normalization for Label-Shift Robust Test-Time Adaptation](./references/Channel-Selective-Normalization-for-Label-Shift-Robust-Test-Time-Adaptation/meta/meta_info.txt) - Vianna et al.
- [Less is More: Pseudo-Label Filtering for Continual Test-Time Adaptation](./references/Less-is-More-Pseudo-Label-Filtering-for-Continual-Test-Time-Adaptation/meta/meta_info.txt) - Zheng et al.
- [TTN: A Domain-Shift Aware Batch Normalization in Test-Time Adaptation](./references/TTN-A-Domain-Shift-Aware-Batch-Normalization-in-Test-Time-Adaptation/meta/meta_info.txt) - Lim et al.
- [Revisiting Batch Normalization For Practical Domain Adaptation (AdaBN)](./references/Revisiting-Batch-Normalization-For-Practical-Domain-Adaptation/meta/meta_info.txt) - Li et al.
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) - Ioffe and Szegedy, 2015
- [Deep Sets](https://arxiv.org/abs/1703.06114) - Zaheer et al., 2017
- [EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces](https://arxiv.org/abs/1611.08024) - Lawhern et al.
- [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://arxiv.org/abs/1703.04046) - Supratak et al.
- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726) - Wang et al.
- [Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2002.08546) - Liang et al.
- [Efficient Test-Time Model Adaptation without Forgetting (EATA)](https://arxiv.org/abs/2204.02610) - Niu et al.
- [Towards Stable Test-Time Adaptation in Dynamic Wild World (SAR)](https://arxiv.org/abs/2302.12400) - Niu et al.
- [Continual Test-Time Domain Adaptation (CoTTA)](https://arxiv.org/abs/2203.13591) - Wang et al.
- [Robust Continual Test-time Adaptation Against Temporal Correlation (NOTE)](https://papers.neurips.cc/paper_files/paper/2022/hash/ae6c7dbd9429b3a75c41b5fb47e57c9e-Abstract-Conference.html) - Gong et al.
- [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization (T3A)](https://proceedings.neurips.cc/paper_files/paper/2021/file/1415fe9fea0fa1e45dddcff5682239a0-Paper.pdf) - Iwasawa and Matsuo, 2021
- [Detecting and Correcting for Label Shift with Black Box Predictors (BBSE)](https://arxiv.org/abs/1802.03916) - Lipton et al., 2018
- [A Unified View of Label Shift Estimation](https://papers.nips.cc/paper_files/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf) - Garg et al., 2020
- [Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure](https://www.researchgate.net/publication/11608620_Adjusting_the_Outputs_of_a_Classifier_to_New_a_Priori_Probabilities_A_Simple_Procedure) - Saerens et al., 2002
- [Domain-Specific Batch Normalization for Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) - Chang et al., 2019
- [Team Cogitat at NeurIPS 2021: Benchmarks for EEG Transfer Learning Competition](https://arxiv.org/abs/2202.03267) - Bakas et al.
- Official implementation for Latent Alignment: https://github.com/StylianosBakas/LatentAlignment
