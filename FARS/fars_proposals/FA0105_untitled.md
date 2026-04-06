# untitled

# Cross-View PSD Distillation to Reduce the Frontal→Side-View Gap in Remote Photoplethysmography

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV (or similar top AI conferences)

## Introduction

### Context and Motivation

Remote photoplethysmography (rPPG) aims to estimate cardiovascular signals (e.g., blood volume pulse (BVP) waveforms and heart rate (HR)) from video of a person’s face. This enables contactless monitoring in settings like telemedicine, fitness, driver monitoring, and hospital triage.

A major obstacle to practical deployment is **viewpoint sensitivity**: face videos are often captured at non-frontal angles (e.g., laptop webcams placed to the side, phones on desks, surveillance cameras), and rPPG accuracy can degrade sharply. The recently released **MCD-rPPG** dataset provides synchronized recordings from **three cameras** per subject and shows a large and persistent frontal→side-view gap even for strong modern models.

Concretely, **Table 4** in the MCD-rPPG paper reports (HR MAE in bpm):
- **RhythmFormer**: **2.82** (frontal) vs **7.33** (side)
- **PhysFormer**: **4.08** (frontal) vs **10.68** (side)
- **ROI+1D-FPN baseline**: **4.86** (frontal) vs **14.01** (side)

(See `./references/Gaze-into-the-Heart-A-Multi-View-Video-Dataset-for-rPPG-and-Health-Biomarkers-Estimation/sections/5. Experiments.md` Table 4.)

### The Problem

Most rPPG methods are trained on single-view data (often frontal) and rely on inductive biases (ROI selection, temporal modeling, frequency losses) that do not directly enforce **cross-view invariances**. A straightforward fix is to train on all available views (“pooled multi-view training”) and apply strong view-robustness augmentations (color/illumination jitter, occlusion, geometric perturbations). However, pooled training alone may not address the core failure mode: **side views reduce usable facial skin area and increase occlusion/motion artifacts**, causing the predicted waveform to become noisy and its spectrum to develop spurious peaks.

At the same time, viewpoint changes should not alter the underlying physiology: for synchronized clips of the same person, **the true heart rate frequency content should match across views** (up to small synchronization errors). This suggests leveraging synchronized multi-camera data with **frequency-domain constraints**.

### Key Insight and Hypothesis

**Key insight:** The **power spectral density (PSD)** (how signal power is distributed across frequencies) of the predicted BVP is a natural cross-view invariant: it is robust to small time shifts (important because video↔PPG sync can be imperfect) and summarizes the dominant cardiac frequency. In MCD-rPPG, the dataset authors explicitly note that the frontal-facing camera **cam2** (frontal view) shows significantly better video↔PPG synchronization than the other cameras (cam1/cam3, non-frontal views) (see `./references/Gaze-into-the-Heart-A-Multi-View-Video-Dataset-for-rPPG-and-Health-Biomarkers-Estimation/sections/3.2. PPG synchronization.md`).

**Hypothesis:** If we add an **asymmetric cross-view PSD distillation** loss during training on synchronized multi-camera rPPG data (teacher view fixed to **cam2**), then a supervised rPPG model will **reduce side-view HR MAE** without degrading frontal-view performance, because the side view mainly injects noise/occlusion while the underlying cardiac frequency structure should be shared.

Why this could fail (genuine uncertainty):
- Side views may induce systematic spectral artifacts (e.g., motion/illumination periodicities) that make PSD matching counterproductive.
- If the “teacher” view is occasionally wrong (low SNR segments), distillation could propagate mistakes.
- Any improvement might be explained by a strong pooled multi-view + augmentation baseline (i.e., “just more view diversity”), not by the PSD distillation mechanism.

We control for these by (i) using pooled multi-view + strong augmentations as the primary baseline, and (ii) comparing symmetric vs asymmetric cross-view losses.

---

## Proposed Approach

### Overview

We train a supervised rPPG model on MCD-rPPG using **paired, synchronized clips** from (cam2, side). The model predicts a BVP waveform for each view. We add a frequency-domain regularizer that encourages the **side view’s predicted PSD** to match the **cam2 PSD**.

We study three training objectives:
1. **A+ (strong baseline):** pooled multi-view supervised training + strong view-robustness augmentation.
2. **B (symmetric PSD consistency):** A+ plus a symmetric KL consistency loss between cam2 and side PSD distributions.
3. **B’ (asymmetric PSD distillation):** A+ plus a stop-gradient KL distillation loss from cam2→side.

### Method Details

**Notation.** Let a synchronized clip pair be `(x_t, x_s)` where:
- `x_t` is the **teacher-view** clip from **cam2** (frontal-facing).
- `x_s` is a **side-view** clip (cam1 or cam3).

Let the rPPG model be `f_\theta(·)` that outputs a predicted BVP waveform `\hat{y} ∈ R^T`.

**Supervised objective.** For each view we compute a standard supervised rPPG loss (exact form depends on backbone; examples include negative Pearson correlation to the ground-truth BVP and/or a frequency classification loss). We denote this generically as:

- `L_sup = L_rppg(\hat{y}, y_gt)`

**Differentiable PSD distribution.** For a predicted waveform `\hat{y}`:
1. Optionally apply a Hann window.
2. Compute one-sided FFT via `torch.fft.rfft`.
3. Compute power `P(f) = |FFT(\hat{y})|^2`.
4. Restrict to the physiological band `f ∈ [0.5, 3.0] Hz`.
5. Normalize to a probability distribution: `p(f) = P(f) / (\sum_f P(f) + \epsilon)`.

**Cross-view losses.**
- **Symmetric consistency** (B):
  - `L_sym = KL(p_t || p_s) + KL(p_s || p_t)`
- **Asymmetric distillation** (B’):
  - `L_asym = KL(stopgrad(p_t) || p_s)`

**Total loss.** For a paired sample:
- **A+**: `L = L_sup(x_t) + L_sup(x_s)`
- **B**: `L = L_sup(x_t) + L_sup(x_s) + λ · L_sym`
- **B’**: `L = L_sup(x_t) + L_sup(x_s) + λ · L_asym`

We will tune `λ` on the validation split from a small grid (e.g., `{0.05, 0.1, 0.2}`) once per method family, and then fix it for the 3-seed evaluation.

**Why asymmetric (stop-gradient) matters.** A symmetric loss can “drag” the high-quality view toward the noisy view by backpropagating through both. The asymmetric variant tests the specific mechanism claim: only the side view should be regularized toward the teacher’s spectrum.

### Key Innovations

- **Cross-view frequency-domain distillation** for rPPG viewpoint robustness, using synchronized multi-camera data.
- **Asymmetric (stop-gradient) variant** to avoid degrading the strongest view while transferring spectral structure to the weaker view.
- A training objective that is **robust to small temporal misalignment** across cameras because it matches PSDs rather than time-domain waveforms.

---

## Related Work

### Field Overview

rPPG methods can be grouped into: (i) classical signal-processing pipelines (e.g., CHROM/POS/PBV) that operate on color traces from skin pixels, (ii) supervised deep models that directly regress BVP/HR from video (CNNs and transformers), and (iii) self-/semi-supervised approaches that reduce dependence on synchronized contact sensors by using physiological priors in the frequency domain. Across these categories, robustness to motion, illumination, skin tone, camera differences, and viewpoint remains a central challenge.

MCD-rPPG enables explicit study of viewpoint effects because it provides **synchronized multi-camera recordings**. The dataset’s analysis suggests viewpoint is a major source of error even when training and testing on the same dataset.

### Related Papers

- **[Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation](./references/Gaze-into-the-Heart-A-Multi-View-Video-Dataset-for-rPPG-and-Health-Biomarkers-Estimation/meta/meta_info.txt)**: Introduces MCD-rPPG and quantifies the frontal→side gap that motivates this proposal.
- **[rPPG-Toolbox: Deep Remote PPG Toolbox](./references/rPPG-Toolbox-Deep-Remote-PPG-Toolbox/meta/meta_info.txt)**: Provides standardized implementations and training defaults (e.g., batch size 4, 30 epochs, 1cycle LR) that we will reuse for reproducible baselines.
- **[PhysFormer](./references/PhysFormer-Facial-Video-based-Physiological-Measurement-with-Temporal-Difference-Transformer/meta/meta_info.txt)**: Transformer-based rPPG model with strong accuracy but still large side-view degradation on MCD-rPPG.
- **[RhythmFormer](./references/RhythmFormer-Extracting-rPPG-Signals-Based-on-Hierarchical-Temporal-Periodic-Transformer/meta/meta_info.txt)**: Periodic-attention transformer that is strong on frontal views yet still shows a substantial side-view gap.
- **[EfficientPhys](./references/EfficientPhys-Enabling-Simple,-Fast-and-Accurate-Camera-Based-Cardiac-Measurement/meta/meta_info.txt)**: Lightweight supervised rPPG model family; a practical backbone candidate for cost-bounded verification.
- **[Contrast-Phys](./references/Contrast-Phys-Unsupervised-Video-based-Remote-Physiological-Measurement-via-Spatiotemporal-Contrast/meta/meta_info.txt)**: Unsupervised rPPG that uses PSD-based contrastive objectives; it motivates using frequency-domain priors but does not study synchronized multi-camera viewpoint transfer.
- **[SiNC](./references/Non-Contrastive-Unsupervised-Learning-of-Physiological-Signals-from-Video/meta/meta_info.txt)**: Non-contrastive unsupervised learning using frequency-domain bandwidth/sparsity/variance losses, showing that spectral priors can be powerful.
- **[Semi-rPPG](./references/Semi-rPPG-Semi-Supervised-Remote-Physiological-Measurement-with-Curriculum-Pseudo-Labeling/meta/meta_info.txt)**: Uses PSD-based consistency across augmentations and SNR-based filtering; our proposal uses PSD matching across *camera views*.
- **[Greip](./references/Advancing-Generalizable-Remote-Physiological-Measurement-through-the-Integration-of-Explicit-and-Implicit-Prior-Knowledge/meta/meta_info.txt)**: Improves generalization via explicit/implicit priors and augmentations; motivates a strong augmentation baseline (A+).
- **[MAR-rPPG](./references/Toward-Motion-Robustness-A-masked-attention-regularization-framework-in-remote-photoplethysmography/meta/meta_info.txt)**: Enforces attention consistency under flips/masking; related in spirit but operates on attention maps rather than cross-view PSD.
- **[SFDA-rPPG](./references/SFDA-rPPG-Source-Free-Domain-Adaptive-Remote-Physiological-Measurement-with-Spatio-Temporal-Consistency/meta/meta_info.txt)**: Uses frequency-domain Wasserstein losses for source-free domain adaptation; demonstrates distribution-level frequency alignment benefits.
- **[Learning Motion-Robust rPPG through Arbitrary Resolution Videos](./references/Learning-Motion-Robust-Remote-Photoplethysmography-through-Arbitrary-Resolution-Videos/meta/meta_info.txt)**: Uses a cross-resolution consistency constraint; analogous invariance principle, but over resolution rather than viewpoint.
- **[KDPhys](./references/KDPhys-An-Attention-Guided-3D-to-2D-Knowledge-Distillation-for-Real-time-Video-Based-Physiological-Measurement/meta/meta_info.txt)**: Applies teacher-student distillation across architectures (3D→2D) for efficiency; our proposal distills across synchronized camera views to improve robustness.
- **[M3PD Dataset + F3Mamba](./references/M3PD-Dataset-Dual-view-Photoplethysmography-(PPG)-Using-Front-and-rear-Cameras-of-Smartphones-in-Lab-and-Clinical-Settings/meta/meta_info.txt)**: Dual-view facial+fingertip physiology sensing; complementary evidence that multi-view signals can be exploited, but different sensor sites than multi-camera face views.

Additional relevant background (URLs; not all scraped locally):
- **Algorithmic Principles of Remote-PPG (POS)**: Classical rPPG formulation and the POS algorithm that is widely used as a baseline and for sync analysis. https://doi.org/10.1109/TBME.2016.2609282
- **CHROM (Robust Pulse Rate from Chrominance-Based rPPG)**: https://doi.org/10.1109/TBME.2013.2266196
- **PBV (Improved motion robustness of remote-PPG)**: https://doi.org/10.1088/0967-3334/35/9/1913
- **DeepPhys (ECCV 2018)**: Deep attention model for video-based physiology. https://arxiv.org/abs/1805.07888
- **PhysNet (BMVC 2019)**: Spatio-temporal network for rPPG waveform recovery. https://arxiv.org/abs/1905.02419
- **MTTS-CAN / TS-CAN (NeurIPS 2020)**: Efficient on-device rPPG via temporal shift + attention. https://arxiv.org/abs/2006.03790
- **AutoHR (NAS + temporal difference conv for rPPG)**: https://arxiv.org/abs/2004.12292
- **PulseGAN (JBHI 2021)**: Frequency-domain losses for waveform reconstruction. https://arxiv.org/abs/2006.02699
- **Promoting Generalization in Cross-Dataset rPPG (CVPRW 2023)**: Highlights augmentation-driven generalization improvements. https://openaccess.thecvf.com/content/CVPR2023W/CVPM/papers/Vance_Promoting_Generalization_in_Cross-Dataset_Remote_Photoplethysmography_CVPRW_2023_paper.pdf
- **Self-Similarity Prior Distillation (SSPD) for unsupervised rPPG**: Uses distillation with physiological priors (not multi-camera). https://arxiv.org/abs/2311.05100

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Classical rPPG (signal processing) | Handcrafted color-space projections + filtering | POS, CHROM, PBV | UBFC-rPPG, PURE, COHFACE | Sensitive to motion/lighting; limited representation learning |
| Supervised deep rPPG | Learn end-to-end mapping video→BVP/HR | DeepPhys, PhysNet, TS-CAN, EfficientPhys, PhysFormer, RhythmFormer | rPPG-Toolbox suite; MCD-rPPG | Generalization gaps across domains and viewpoints |
| Self-/semi-supervised rPPG | Use frequency/periodicity priors to reduce label dependence | Contrast-Phys, SiNC, Semi-rPPG, SSPD | PURE/UBFC/VIPL-HR cross-dataset | Often not designed for synchronized multi-camera supervision |
| Domain adaptation / generalization | Align distributions across datasets via priors/consistency | Greip, SFDA-rPPG | Cross-dataset UBFC↔PURE, COHFACE | Does not directly target viewpoint within one synchronized session |
| Distillation / compression | Transfer knowledge to smaller or different-architecture models | KDPhys | UBFC/PURE/COHFACE | Focus on efficiency, not cross-view robustness |
| Multi-view datasets / fusion | Use multiple sensing views to improve reliability | MCD-rPPG, M3PD | View robustness tests | Fusion can be heavier; inference may require multiple views |

### Closest Prior Work

- **Semi-rPPG**: Uses PSD-based consistency between weak/strong augmentations of the same clip; does not address synchronized multi-camera viewpoint transfer.
- **SFDA-rPPG**: Uses frequency-domain Wasserstein losses to align target-domain distributions without labels; our setting is within-session multi-camera and uses an explicit teacher view.
- **Arbitrary-Resolution rPPG**: Uses a cross-resolution constraint (two streams) to enforce invariance to resolution changes; analogous idea, but our invariance axis is viewpoint.
- **KDPhys**: Uses teacher-student KD for 3D→2D efficiency; we use cross-view distillation to improve side-view accuracy with the same inference-time model.
- **Contrast-Phys / SiNC**: Show that frequency-domain priors are effective for learning rPPG representations, but they do not use synchronized multi-camera supervision to reduce viewpoint gaps.

**Novelty Kill Search Summary:** Searched for combinations of the technique+setting including “multi-camera rPPG knowledge distillation”, “cross-view PSD distillation rPPG”, “multi-view rPPG consistency regularization”, and checked whether any recent rPPG KD papers explicitly distill across synchronized camera views. As of **2026-02-16**, we did not find prior work that targets the MCD-rPPG viewpoint gap via **asymmetric cross-view PSD distillation from a fixed frontal teacher view**.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Semi-rPPG | PSD consistency across augmentations + curriculum pseudo-labeling | No multi-camera viewpoint supervision | Apply PSD matching across synchronized camera views | View changes preserve physiology but inject noise; cross-view PSD is a direct invariant |
| SFDA-rPPG | Frequency-domain Wasserstein alignment for SFDA | Domain shift across datasets; no explicit teacher view | Use fixed high-sync view as teacher, within-session | Teacher view provides a higher-SNR target; avoids domain-label issues |
| Arbitrary-Resolution rPPG | Cross-resolution consistency loss | Targets resolution, not viewpoint | Cross-view (camera angle) consistency | Viewpoint is a major deployment failure mode not covered by resolution invariance |
| KDPhys | Distill 3D teacher → 2D student | Efficiency focus; not view robustness | Distill cam2 spectrum → side spectrum | Same inference-time model, but improved robustness to side views |
| MAR-rPPG | Attention-map consistency under flips/masking | Not a direct spectral invariance; no multi-camera | PSD-level invariance across cameras | PSD directly targets the HR signal component |

---

## Experiments

### Experimental Setup

**Dataset:** MCD-rPPG (Multi-Camera Dataset for Remote Photoplethysmography), a public dataset with **600 subjects**, **2 physiological states**, and **3 synchronized camera views** per session.
- Download: https://huggingface.co/datasets/kyegorov/mcd_rppg
- Paper/code: https://arxiv.org/abs/2508.17924 and https://github.com/ksyegorov/mcd_rppg

**Splits:** Subject-disjoint split to avoid identity leakage. If the dataset repo provides an official split, use it; otherwise use an 80/10/10 split by subject (train/val/test) and report the exact subject IDs used.

**Preprocessing:** Use the public MCD-rPPG baseline pipeline (face detection/cropping and clip sampling) as a starting point:
- https://github.com/garrulus2003/MCD-rPPG-baselines (data preparation scripts)

We will sample short synchronized clips (e.g., 10s) and downsample/crop to standard rPPG input sizes (e.g., 128×128) following rPPG-Toolbox conventions.

**Backbone model (verification target):** EfficientPhys (from rPPG-Toolbox) as a cost-bounded supervised baseline; the proposed loss is architecture-agnostic and can be added to other backbones later if promising.

**Baseline Ladder (REQUIRED):**
- **Classical / unsupervised**: POS, CHROM, PBV (no training) for sanity and historical context.
- **Single-view supervised**: EfficientPhys trained/tested per-view (cam2 only; side only) to quantify view gap under the same codebase.
- **Strong multi-view baseline (A+)**: pooled multi-view supervised training + strong augmentations.
- **Our variants**: A+ + symmetric PSD consistency (B), A+ + asymmetric PSD distillation (B’).

**Resource Estimate** (conservative; includes 3 seeds):
- Training runs: 3 conditions × 3 seeds = 9 trainings.
- Expected cost per training: **4–8 GPU-hours** on 1×A100-80GB (EfficientPhys-class models; 30 epochs, batch size 4).
- Total training cost: **36–72 GPU-hours**.
- Preprocessing: CPU-heavy video decode/face crop; one-time cost (not GPU-limited).
- Total (including eval): **≤100 GPU-hours**, well below the **768 GPU-hour** cap.

If training time is unexpectedly high, downscale by (i) reducing epochs to 15 for an initial decisive check, and (ii) using early stopping on validation HR MAE.

**Variance & Reproducibility:**
- Run all three main conditions with **3 random seeds** (e.g., `seeds=[42, 123, 456]`).
- Report mean ± std for HR MAE.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MCD-rPPG | Synchronized multi-camera face videos with PPG/ECG | **HR MAE (bpm)** (lower is better), PPG MAE (optional) | subject-disjoint test | https://huggingface.co/datasets/kyegorov/mcd_rppg | Use rPPG-Toolbox-style HR computation (peak in 0.5–3 Hz) + MCD-rPPG baseline eval if available |

Primary metric for the decision rule: **side-view HR MAE**, computed separately on cam1 and cam3, and averaged.

### Main Results

#### Results Table (to be filled by verification)

| Method | Backbone | View | HR MAE (mean±std) | Source | Notes |
|---|---|---:|---:|---|---|
| POS | - | cam2 / cam1 / cam3 | **TBD** | - | Classical baseline; no training |
| EfficientPhys (single-view cam2) | EfficientPhys | cam2 | **TBD** | - | Needs re-run (our split/protocol) |
| EfficientPhys (single-view side) | EfficientPhys | cam1+cam3 | **TBD** | - | Needs re-run |
| **A+ pooled multi-view + aug** | EfficientPhys | cam2 | **TBD** | - | Strong baseline |
| **A+ pooled multi-view + aug** | EfficientPhys | cam1+cam3 | **TBD** | - | Strong baseline |
| **B: A+ + symmetric PSD** | EfficientPhys | cam2 | **TBD** | - | Symmetric consistency |
| **B: A+ + symmetric PSD** | EfficientPhys | cam1+cam3 | **TBD** | - | Symmetric consistency |
| **B’: A+ + asymmetric PSD (ours)** | EfficientPhys | cam2 | **TBD** | - | stopgrad(cam2)→side |
| **B’: A+ + asymmetric PSD (ours)** | EfficientPhys | cam1+cam3 | **TBD** | - | stopgrad(cam2)→side |

Reference numbers (published; protocol details may differ) from MCD-rPPG Table 4:
- RhythmFormer: 2.82 (frontal) vs 7.33 (side)
- PhysFormer: 4.08 (frontal) vs 10.68 (side)
- ROI+1D-FPN baseline: 4.86 (frontal) vs 14.01 (side)

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B vs B’ | symmetric vs stop-gradient PSD loss | Symmetric may slightly hurt cam2; asymmetric should preserve cam2 while improving side |
| λ sweep (analysis-only) | λ ∈ {0.05, 0.1, 0.2} | Too large λ may over-regularize and hurt waveform quality |

### Experimental Rigor

**Top confounders and controls:**
1. **“It’s just augmentation”**: Use A+ (pooled multi-view + strong augmentations) as the primary baseline.
2. **Teacher drift / harming frontal view**: Include symmetric vs asymmetric comparison and enforce a frontal MAE guardrail in the decision rule.
3. **Split leakage**: Use subject-disjoint splits; report subject IDs.

**Sanity checks:**
- Reproduce the qualitative view gap on our split (cam2 vs cam1/cam3) for at least one backbone.
- Verify cam2 has higher synchronization quality as reported (paper Section 3.2) and report descriptive stats of a simple SNR proxy (PSD peak concentration) per view.

---

## Success Criteria

**Hypothesis**: Asymmetric cross-view PSD distillation from cam2 to side views improves side-view HR MAE while preserving frontal performance.

**Decision Rule**:
- **Proceed**: B’ improves **side-view HR MAE by ≥ 1.0 bpm** vs A+ **and** cam2 HR MAE changes by at most **±0.3 bpm**, across **3 seeds**.
- **Inconclusive**: Side-view improvement in **[0.5, 1.0) bpm** with cam2 within ±0.3 bpm (report as partial support; do not expand scope in this proposal).
- **Refute**: Side-view improvement **< 0.5 bpm** or cam2 worsens by **> 0.5 bpm**.

---

## Impact Statement

If successful, this method would provide a simple training-time regularizer that makes rPPG models **more reliable under non-frontal camera placement** without requiring multi-camera inference. This would improve robustness for practitioners deploying rPPG in telemedicine and consumer devices where viewpoint is difficult to control.

---

## References

- [Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation](./references/Gaze-into-the-Heart-A-Multi-View-Video-Dataset-for-rPPG-and-Health-Biomarkers-Estimation/meta/meta_info.txt) - Egorov et al., 2024/2025
- [rPPG-Toolbox: Deep Remote PPG Toolbox](./references/rPPG-Toolbox-Deep-Remote-PPG-Toolbox/meta/meta_info.txt) - Liu et al., 2022
- [PhysFormer](./references/PhysFormer-Facial-Video-based-Physiological-Measurement-with-Temporal-Difference-Transformer/meta/meta_info.txt) - Yu et al., 2022
- [RhythmFormer](./references/RhythmFormer-Extracting-rPPG-Signals-Based-on-Hierarchical-Temporal-Periodic-Transformer/meta/meta_info.txt) - Zou et al., 2024
- [EfficientPhys](./references/EfficientPhys-Enabling-Simple,-Fast-and-Accurate-Camera-Based-Cardiac-Measurement/meta/meta_info.txt) - Liu et al., 2021
- [Contrast-Phys](./references/Contrast-Phys-Unsupervised-Video-based-Remote-Physiological-Measurement-via-Spatiotemporal-Contrast/meta/meta_info.txt) - Sun & Li, 2022
- [SiNC](./references/Non-Contrastive-Unsupervised-Learning-of-Physiological-Signals-from-Video/meta/meta_info.txt) - Speth et al., 2023
- [Semi-rPPG](./references/Semi-rPPG-Semi-Supervised-Remote-Physiological-Measurement-with-Curriculum-Pseudo-Labeling/meta/meta_info.txt) - Wu et al., 2025
- [Greip](./references/Advancing-Generalizable-Remote-Physiological-Measurement-through-the-Integration-of-Explicit-and-Implicit-Prior-Knowledge/meta/meta_info.txt) - Zhang et al., 2024/2025
- [MAR-rPPG](./references/Toward-Motion-Robustness-A-masked-attention-regularization-framework-in-remote-photoplethysmography/meta/meta_info.txt) - Zhao et al., 2024
- [SFDA-rPPG](./references/SFDA-rPPG-Source-Free-Domain-Adaptive-Remote-Physiological-Measurement-with-Spatio-Temporal-Consistency/meta/meta_info.txt) - Xie et al., 2024
- [Learning Motion-Robust rPPG through Arbitrary Resolution Videos](./references/Learning-Motion-Robust-Remote-Photoplethysmography-through-Arbitrary-Resolution-Videos/meta/meta_info.txt) - Li et al., 2023
- [KDPhys](./references/KDPhys-An-Attention-Guided-3D-to-2D-Knowledge-Distillation-for-Real-time-Video-Based-Physiological-Measurement/meta/meta_info.txt) - Sahoo et al., 2025/2026
- [M3PD Dataset + F3Mamba](./references/M3PD-Dataset-Dual-view-Photoplethysmography-(PPG)-Using-Front-and-rear-Cameras-of-Smartphones-in-Lab-and-Clinical-Settings/meta/meta_info.txt) - 2025
- Algorithmic Principles of Remote-PPG (POS) - Wang et al., 2017. https://doi.org/10.1109/TBME.2016.2609282
- CHROM - de Haan & Jeanne, 2013. https://doi.org/10.1109/TBME.2013.2266196
- PBV - de Haan & van Leest, 2014. https://doi.org/10.1088/0967-3334/35/9/1913
- DeepPhys - Chen & McDuff, 2018. https://arxiv.org/abs/1805.07888
- PhysNet - Yu et al., 2019. https://arxiv.org/abs/1905.02419
- MTTS-CAN / TS-CAN - Liu et al., 2020. https://arxiv.org/abs/2006.03790
- AutoHR - Yu et al., 2020/2021. https://arxiv.org/abs/2004.12292
- PulseGAN - Song et al., 2020/2021. https://arxiv.org/abs/2006.02699
- Promoting Generalization in Cross-Dataset rPPG - Vance et al., 2023. https://openaccess.thecvf.com/content/CVPR2023W/CVPM/papers/Vance_Promoting_Generalization_in_Cross-Dataset_Remote_Photoplethysmography_CVPRW_2023_paper.pdf
- Self-Similarity Prior Distillation (SSPD) - 2023/2024. https://arxiv.org/abs/2311.05100
