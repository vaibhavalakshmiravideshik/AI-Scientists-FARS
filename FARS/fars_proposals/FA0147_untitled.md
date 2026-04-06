# untitled

# Training-Free Intensity Calibration for SEVIR Nowcasting via Quantile Remapping

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Precipitation nowcasting (0–2 hour forecasts) is used in high-stakes workflows such as flood warning, aviation operations, and severe weather monitoring. Modern deep learning nowcasters are typically trained with pixel-wise regression losses (e.g., L1/L2) on radar-derived targets, but are often evaluated with **thresholded event-detection metrics** that emphasize rare, high-intensity events.

A common benchmark is **SEVIR** (Storm EVent ImageRy), which contains >10k storm events with aligned radar and satellite imagery and supports precipitation nowcasting using NEXRAD Vertically Integrated Liquid (VIL) mosaics at 1 km resolution and 5 minute cadence. In the SEVIR nowcasting protocol, models predict 12 future frames (60 minutes) from 13 context frames (65 minutes) and are evaluated by converting predicted/true intensity fields into binary exceedance masks at fixed thresholds (e.g., 16/74/133/160/181/219 in the 0–255 VIL scale). Prior work treats the highest thresholds as “extreme events”; for example, CasCast reports that VIL=181 and VIL=219 correspond to roughly 12.14 and 32.23 kg/m², respectively (CasCast Table 2 discussion).

### The Problem

Deep nowcasting papers typically improve threshold metrics by changing the **model** (e.g., Transformers, cascaded generative refiners, diffusion/flow models) or the **training objective** (e.g., frequency-domain losses, metric-aligned losses). However, threshold metrics such as:

- **CSI (Critical Success Index)**: CSI = Hits / (Hits + Misses + False Alarms)
- **HSS (Heidke Skill Score)**: categorical accuracy relative to random
- **Pooled CSI (e.g., POOL16)**: compute CSI after max-pooling over 16×16 km neighborhoods to tolerate small spatial offsets

are also sensitive to a different failure mode: **marginal intensity miscalibration**. If a model predicts correct storm structure and location but systematically underestimates intensity (or compresses dynamic range), then many pixels are “near misses” just below the evaluation thresholds, depressing CSI at high thresholds.

In operational meteorology, **quantile mapping** is a standard bias-correction tool to match predicted and observed marginal distributions. Surprisingly, in the deep nowcasting literature around SEVIR, quantile mapping is rarely discussed as a post-hoc, training-free step for improving threshold-based nowcasting skill; most work focuses on heavier generative post-processing (e.g., diffusion-based deblurring) or retraining.

### Key Insight and Hypothesis

**Hypothesis:** For SEVIR nowcasting, a large fraction of false negatives at extreme thresholds (e.g., 219) are *intensity near-misses* rather than purely spatial/temporal displacement. Therefore, fitting a **monotone, post-hoc intensity calibration map** on the SEVIR validation split and applying it to a fixed deterministic model’s outputs will improve **CSI-M-POOL16** (mean CSI across thresholds under POOL16) and **CSI-219-POOL16**, with limited degradation in proper scoring rules (MAE/CRPS) and without large false-alarm increases.

**Why we could be wrong:** (i) Errors may be dominated by displacement/timing, so any monotone remap is ineffective. (ii) Improvements could be “metric hacking” that increases predicted exceedance rates without improving spatial structure. (iii) A mapping fitted on validation may not transfer to test due to distribution shift.

---

## Proposed Approach

### Overview

We propose **Quantile Remap Calibration (QRC)**: a training-free, monotone mapping applied to predicted intensities from an existing deterministic nowcasting model.

Given model predictions \(\hat{Y}\in\mathbb{R}^{T\times H\times W}\) (12 forecast frames) and ground-truth \(Y\), we fit a monotone function \(f:[0,255]\to[0,255]\) using only the **validation split**, then produce calibrated forecasts \(\tilde{Y}=f(\hat{Y})\). Because \(f\) is 1D and monotone, it changes intensity ranks but does not introduce new spatial structure.

We intentionally keep the decisive experiment small (3 conditions):
1. **No calibration** (raw model output)
2. **Affine rescale** (strong cheap baseline)
3. **Quantile remap (QRC)** (ours)

### Method Details

#### A. Base model predictions
- Use a public deterministic checkpoint for SEVIR nowcasting (initially EarthFormer, because it is a widely used SEVIR baseline and is explicitly evaluated in CasCast).
- Generate predictions for the SEVIR validation set (to fit calibrators) and test set (to evaluate).

#### B. Affine calibration baseline
Fit \(f_{\text{affine}}(x)=\text{clip}(ax+b,0,255)\) on validation pixels by least squares (or MAE regression), using a fixed sampling scheme and seed.

This baseline captures the strongest “obvious” correction a practitioner would try first (global scale/offset).

#### C. Quantile remap calibration (QRC)
Let \(F_{\hat{Y}}\) be the empirical CDF of predicted intensities on validation and \(F_Y\) the empirical CDF of true intensities. Define
\[
  f_{\text{QRC}}(x) = F_Y^{-1}(F_{\hat{Y}}(x)).
\]
Implementation details:
- Work in the same 0–255 VIL scale used by SEVIR categorical evaluation.
- Build the mapping with \(K\) quantile bins (default \(K=1024\)) using linear interpolation between bin edges.
- Use stratified sampling over intensities to ensure tail coverage (e.g., oversample pixels with \(Y\ge 181\)).
- Clamp outputs to \([0,255]\).

#### D. Diagnostics to distinguish “real” gains from metric gaming
All diagnostics reuse the same predictions; they do not introduce extra model conditions.

1. **Per-lead-time CSI:** report CSI-219-POOL16 as a function of lead time (5–60 min) to check that gains are not confined to a single frame.
2. **Proper scoring rule:** report MAE (equivalently CRPS for deterministic forecasts) to detect cases where CSI improves but intensity accuracy collapses.
3. **Spatial structure check:** report FSS (Fractions Skill Score) at a fixed neighborhood size and threshold (e.g., 219, 16×16 km), which penalizes indiscriminate intensity inflation.
4. **Mechanism check (near-miss rate):** on the uncalibrated baseline, measure what fraction of false negatives at threshold 219 are “near misses” in intensity (e.g., predicted max within a 16×16 neighborhood is in [181,219)) versus being far below the threshold. If near-misses are rare, QRC is unlikely to help.

### Key Innovations

- **Hypothesis-driven use of a classical tool:** Quantile mapping is known in meteorology, but we frame it as a minimal, testable hypothesis about why thresholded CSI lags for deterministic deep nowcasters on SEVIR.
- **A strong training-free baseline for deep nowcasting:** Establishes whether part of the gains attributed to heavy generative refinement can be recovered by a cheap post-hoc monotone remap.
- **Anti-metric-gaming diagnostics:** Explicit checks (MAE, FSS, near-miss analysis) to distinguish genuine improvements in extreme-event detection from trivial exceedance inflation.

---

## Related Work

### Field Overview

Deep precipitation nowcasting methods can be roughly grouped into: (i) deterministic spatiotemporal predictors (RNN/CNN/Transformer), (ii) probabilistic generative models (GAN/diffusion/flow) that aim to produce sharp ensembles, (iii) post-processing approaches that refine outputs of a base model, and (iv) calibration and metric-aligned training methods that target categorical verification metrics.

On SEVIR, recent work reports thresholded CSI at multiple thresholds and often uses pooled variants (POOL4/POOL16) to assess local and regional extreme precipitation prediction. This creates a setting where a **monotone intensity remap** can, in principle, improve categorical scores without altering the base model.

### Related Papers

- **[SEVIR: A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology](./references/SEVIR-A-Storm-Event-Imagery-Dataset-for-Deep-Learning-Applications-in-Radar-and-Satellite-Meteorology/meta/meta_info.txt)**: Introduces the SEVIR dataset and benchmark metrics for precipitation nowcasting.
- **[Earthformer: Exploring Space-Time Transformers for Earth System Forecasting](./references/Earthformer-Exploring-Space-Time-Transformers-for-Earth-System-Forecasting/meta/meta_info.txt)**: Transformer-based deterministic baseline on SEVIR with strong CSI at high thresholds.
- **[CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling](./references/CasCast-Skillful-High-resolution-Precipitation-Nowcasting-via-Cascaded-Modelling/meta/meta_info.txt)**: Cascades a deterministic predictor with a diffusion refiner and reports large gains on pooled CSI for extremes.
- **[FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching](./references/Under-review-as-a-conference-paper-at-ICLR-2026-FLOWCAST-ADVANCING-PRECIPITATION-NOWCASTING-WITH-CONDITIONAL-FLOW-MATCHING/meta/meta_info.txt)**: Uses latent conditional flow matching to generate efficient ensembles; uses quantile mapping for *cross-dataset threshold setting*, not output calibration.
- **[PreDiff: Precipitation Nowcasting with Latent Diffusion Models](./references/PreDiff-Precipitation-Nowcasting-with-Latent-Diffusion-Models/meta/meta_info.txt)**: Latent diffusion model for probabilistic nowcasting; emphasizes uncertainty and sharpness.
- **[DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting](./references/DiffCast-A-Unified-Framework-via-Residual-Diffusion-for-Precipitation-Nowcasting/meta/meta_info.txt)**: Residual diffusion framework for precipitation nowcasting.
- **[PostCast: Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Modeling](./references/PostCast-Generalizable-Postprocessing-for-Precipitation-Nowcasting-via-Unsupervised-Blurriness-Modeling/meta/meta_info.txt)**: Diffusion-based, unsupervised post-processing to deblur nowcasting outputs across models and datasets.
- **[Probability calibration for precipitation nowcasting](./references/Probability-calibration-for-precipitation-nowcasting/meta/meta_info.txt)**: Introduces ETCE and selective scaling for probabilistic calibration; closest work on calibration in the deep nowcasting context.
- **[Rectifying Distribution Shift in Cascaded Precipitation Nowcasting](./references/RectiCast-Rectifying-Distribution-Shift-in-Cascaded-Precipitation-Nowcasting/meta/meta_info.txt)**: Addresses distribution shift in cascaded nowcasting pipelines.
- **[SimCast: Enhancing Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation](./references/SimCast-Enhancing-Precipitation-Nowcasting-with-Short-to-Long-Term-Knowledge-Distillation/meta/meta_info.txt)**: Uses distillation across horizons for improved nowcasting.
- **[Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting](./references/Fourier-Amplitude-and-Correlation-Loss-Beyond-Using-L2-Loss-for-Skillful-Precipitation-Nowcasting/meta/meta_info.txt)**: Modifies training losses to better preserve spatial statistics.
- **[Optimization of Deep Learning Precipitation Models Using Categorical Binary Metrics](./references/Optimization-of-deep-learning-precipitation-models-using-categorical-binary-metrics/meta/meta_info.txt)**: Studies training objectives aligned with categorical verification metrics.
- **[Deep learning for precipitation nowcasting: A survey from the perspective of time series forecasting](./references/Deep-learning-for-precipitation-nowcasting-A-survey-from-the-perspective-of-time-series-forecasting/meta/meta_info.txt)**: Survey of nowcasting method families and evaluation practices.
- **[Precipitation nowcasting of satellite data using physically-aligned neural networks](./references/Precipitation-nowcasting-of-satellite-data-using-physically-aligned-neural-networks/meta/meta_info.txt)**: Incorporates physical alignment constraints for satellite-based nowcasting.
- **[Skilful precipitation nowcasting using deep generative models of radar (DGMR)](./references/Skillful-Precipitation-Nowcasting-using-Deep-Generative-Models-of-Radar/meta/meta_info.txt)**: GAN-based probabilistic nowcasting with strong perceptual realism and extreme-event performance.
- **[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)**: Foundational ConvLSTM model for radar nowcasting.
- **[PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning](https://arxiv.org/abs/2103.09504)**: RNN-based spatiotemporal predictor widely used in nowcasting/video prediction.
- **[SimVP: Simpler yet Better Video Prediction](https://arxiv.org/abs/2206.05099)**: Simple CNN baseline often used as a strong deterministic backbone in nowcasting comparisons.
- **[MetNet: A Neural Weather Model for Precipitation Forecasting](https://arxiv.org/abs/2003.12140)**: Large-scale neural precipitation forecasting system (different dataset) emphasizing probabilistic outputs.
- **[Optical flow models as an open benchmark for radar-based precipitation nowcasting (rainymotion)](https://doi.org/10.5194/gmd-12-1387-2019)**: Open benchmark for optical-flow extrapolation baselines.
- **[Pysteps: an open-source Python library for probabilistic precipitation nowcasting](https://doi.org/10.5194/gmd-12-4185-2019)**: Open-source probabilistic nowcasting library implementing STEPS-style baselines.
- **[RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting](https://doi.org/10.5194/gmd-2020-30)**: CNN nowcasting model; highlights smoothing vs extreme-event fidelity trade-offs.
- **[Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting](https://doi.org/10.1109/LGRS.2022.3162882)**: Attention-based nowcasting model targeting better high-intensity performance.
- **[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)**: Standard reference for post-hoc calibration (temperature scaling) in deep networks.
- **[TyrainNow: A Deep Learning-Based Model for Typhoon Rainfall Nowcast With Radar Products](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025WR039897)**: Uses quantile mapping for post-hoc correction in a different nowcasting setting (typhoon rainfall), showing adjacent precedent.
- **[Bias Correction, Quantile Mapping, and Downscaling](https://doi.org/10.5194/hess-2016-659)**: Review discussing quantile mapping as a distributional bias-correction tool in hydrometeorology.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Deterministic deep nowcasters | Predict a single future sequence with regression losses | ConvLSTM, PredRNN, SimVP, Earthformer | CSI/HSS at thresholds; pooled CSI on SEVIR/HKO-7 | Blurry means; intensity attenuation at extremes |
| Probabilistic/generative nowcasters | Sample multiple plausible futures (GAN/diffusion/flow) | DGMR, PreDiff, DiffCast, CasCast, FlowCast | CRPS + categorical metrics (CSI/FAR/HSS/FSS) | Higher compute; sampling cost; calibration/coverage trade-offs |
| Post-processing refiners | Refine base model outputs without retraining base model | PostCast | Often report improved CSI at extreme thresholds | Can be computationally heavy; may change spatial structure |
| Calibration / metric-aligned methods | Adjust outputs or training to better match categorical metrics | Kurki et al. (ETCE + scaling), Larraondo et al. (categorical-metric optimization) | ETCE/thresholded calibration; categorical skill | May require ensembles or retraining; unclear transfer |

### Closest Prior Work

- **Kurki et al. (2025)** proposes ETCE and selective temperature scaling for *probabilistic* nowcasts. Our setting is different: we calibrate **deterministic intensity fields** with a monotone mapping and evaluate pooled CSI at extreme thresholds.
- **FlowCast (2026)** uses quantile mapping to *define comparable thresholds* across datasets (SEVIR↔ARSO). We instead apply quantile mapping as a **post-hoc output calibration** within a single dataset to test whether CSI gaps are partly due to marginal intensity bias.
- **TyrainNow (2025)** uses quantile mapping as an attenuation/bias correction step for typhoon rainfall nowcasting. This is adjacent precedent for “QM as post-hoc correction”, but differs in dataset, targets, and evaluation.
- **PostCast (2024)** is a post-processing method, but it is a diffusion-based deblurring procedure that introduces new spatial structure and significant compute. QRC is a cheap alternative that only remaps intensities.

**Novelty Kill Search Summary:** As of 2026-02-18, we searched for the exact combination of “quantile mapping (or isotonic regression) + post-hoc calibration + deep precipitation nowcasting + SEVIR/HKO-7” using web queries (e.g., “quantile mapping SEVIR nowcasting”, “isotonic regression precipitation nowcasting CSI”, “post hoc calibration radar nowcast pooled CSI”) and a local KB grep over nowcasting papers. We found no close prior work applying QM/IR as a training-free *model-output calibration* step on SEVIR-style deep nowcasting benchmarks; closest adjacent work is TyrainNow (different setting) and Kurki et al. (probabilistic calibration).

### Comparison Table

| Related work | What it does | Key limitation (for our question) | What we change | Why ours should win |
|---|---|---|---|---|
| Kurki et al. 2025 (ETCE + scaling) | Calibrates probabilistic nowcasts w.r.t. thresholded calibration error | Targets probabilistic scores; not focused on deterministic intensity bias vs CSI | Calibrate deterministic intensities with monotone remap | Directly tests whether CSI gaps are driven by marginal intensity bias |
| FlowCast 2026 | Efficient probabilistic nowcasting; QM used only for cross-dataset threshold matching | Does not test QM as output calibration | Apply QM to outputs within SEVIR | Tests a cheaper knob than new generative models |
| PostCast 2024 | Diffusion-based postprocessing to deblur | High compute; changes spatial structure | 1D monotone mapping only | If intensity bias dominates, simple mapping may recover a meaningful fraction of CSI gains |
| CasCast 2024 | Cascaded deterministic+diffusion refinement | Requires training and sampling | No training; post-hoc | Provides a low-cost alternative baseline for practitioners |
| TyrainNow 2025 | QM to correct attenuation biases in typhoon rainfall nowcasting | Different dataset/protocol | SEVIR nowcasting + pooled CSI | Tests transfer of QM-style correction to SEVIR deep nowcasting |

---

## Experiments

### Experimental Setup

**Core experiment (3 conditions):**
1. **EarthFormer (uncalibrated)**: evaluate the released checkpoint as-is.
2. **EarthFormer + affine rescale**: fit \(ax+b\) on validation, apply on test.
3. **EarthFormer + QRC**: fit quantile remap on validation, apply on test.

**Baseline ladder (domain-appropriate):**
- Uncalibrated checkpoint (what users have today)
- Strong cheap post-processing baseline (affine)
- Proposed post-processing (QRC)
- Contextual SOTA reference points from literature (CasCast, FlowCast) for “how far are we from the best-known methods?” (not required to re-run).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| EarthFormer (SEVIR) | ~15M params (per EarthFormer paper) | Official repo: https://github.com/amazon-science/earth-forecasting-transformer | Provides training/inference code |
| EarthFormer checkpoint used by CasCast (preferred) | - | CasCast repo notes a released SEVIR checkpoint (Google Drive link in CasCast README / scripts) | Use this to match CasCast Table 2 protocol |

**Training Data:** None (inference + post-hoc calibration only).

**Other Resources:**
- SEVIR dataset download: https://registry.opendata.aws/sevir/
- CasCast evaluation scripts (recommended for protocol match): https://github.com/OpenEarthLab/CasCast

**Resource Estimate:**
- **Compute budget**: Primarily inference on SEVIR val+test for a single deterministic checkpoint and metric computation. Expected to fit within **<50 A100 GPU-hours** (conservative) plus CPU work for calibration.
- **GPU memory**: Determined by the EarthFormer checkpoint inference; expected to fit in a single 80GB A100.
- **API usage**: None.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SEVIR VIL nowcasting | Predict 12 future VIL frames from 13 context frames | CSI-M, CSI-219, pooled CSI (POOL16), HSS, MAE/CRPS; plus FAR + FSS diagnostics | val (fit), test (eval) | https://registry.opendata.aws/sevir/ | CasCast eval (recommended): https://github.com/OpenEarthLab/CasCast |

**Metric definitions (for clarity):**
- **CSI** at threshold \(\tau\): computed after binarizing intensities at \(\tau\).
- **CSI-M**: mean CSI across thresholds \([16,74,133,160,181,219]\) (SEVIR standard).
- **POOL16**: compute CSI after max-pooling binary masks over 16×16 km neighborhoods.
- **CRPS (Continuous Ranked Probability Score)**: a proper scoring rule for probabilistic forecasts; lower is better. For deterministic predictions (a single point forecast), CRPS reduces to the **MAE (mean absolute error)**.

### Main Results

Published reference numbers (SEVIR, POOL16) from CasCast Table 2:

| Method | Base Model | Benchmark | CSI-M-POOL16 ↑ | CSI-219-POOL16 ↑ | CRPS ↓ | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| EarthFormer† | EarthFormer | SEVIR | 0.4351 | 0.1481 | 0.0251 | CasCast Table 2 | † uses official checkpoint per CasCast caption |
| SimVP | SimVP | SEVIR | 0.4530 | 0.1685 | 0.0259 | CasCast Table 2 | deterministic baseline |
| PredRNN | PredRNN | SEVIR | 0.4623 | 0.1909 | 0.0271 | CasCast Table 2 | deterministic baseline |
| CasCast | EarthFormer + diffusion refiner | SEVIR | 0.5225 | 0.2841 | 0.0202 | CasCast Table 2 | strong probabilistic baseline |
| **Affine-calibrated (TBD)** | EarthFormer | SEVIR | **TBD** | **TBD** | **TBD** | - | needs evaluation run |
| **QRC (TBD)** | EarthFormer | SEVIR | **TBD** | **TBD** | **TBD** | - | needs evaluation run |

### Ablation Studies

If QRC shows a clear gain, run a small ablation (same checkpoint, same protocol):

| Variant | What’s changed | Expected finding |
|---|---|---|
| QRC (global) | single mapping for all lead times | baseline improvement |
| QRC (per-lead-time) | separate mapping per forecast step | may improve long-horizon tails if miscalibration grows with lead time |
| Isotonic regression (optional) | monotone least-squares mapping instead of quantile mapping | if results match QRC, gains likely come from monotonicity not distribution matching |

### Experimental Rigor

- **Seeds / randomness**: Inference is deterministic for a fixed checkpoint. Calibration uses random pixel subsampling; run **3 subsampling seeds** (e.g., `seeds=[0,1,2]`) and report mean±std for QRC and affine.
- **Bootstrap**: Use bootstrap over test sequences to estimate confidence intervals for CSI-219-POOL16 differences.
- **Key confounders and controls**:
  - **Leakage**: fit mapping only on validation split; never use test to fit \(f\).
  - **Protocol mismatch**: use an official evaluation implementation (CasCast) to match published metrics.
  - **Metric gaming**: report MAE/CRPS + FSS + exceedance-rate calibration (reliability diagram) alongside CSI.

---

## Success Criteria

**Hypothesis (directional):** QRC improves pooled CSI at extreme thresholds by correcting systematic intensity underestimation, with modest or no degradation in MAE and spatial structure metrics.

**Decision Rule (concrete):**
- **Proceed** if (on SEVIR test, with bootstrap CIs) QRC improves **CSI-M-POOL16** by **≥0.01** absolute *and* improves **CSI-219-POOL16** by **≥0.02** absolute over the uncalibrated baseline, while (i) MAE/CRPS worsens by **≤2% relative**, and (ii) FSS at threshold 219 (16×16 km neighborhood) does not decrease.
- **Pivot** if CSI improves but MAE/CRPS or FSS degrades substantially, suggesting threshold gains come from indiscriminate exceedance inflation. In this case, try per-lead-time QRC (ablation) to reduce long-horizon overcorrection.
- **Refute** if QRC does not yield a statistically distinguishable improvement (CI overlaps 0) or if the Phase-0 mechanism check shows near-miss false negatives at 219 are rare (e.g., <10% of misses are within [181,219) in a 16×16 neighborhood).

---

## Impact Statement

If successful, QRC provides a **drop-in, training-free** baseline that practitioners can apply to any deterministic nowcasting model before investing in expensive retraining or generative post-processing, potentially recovering a non-trivial fraction of extreme-event skill. If it fails, the negative result is still decision-relevant: it would suggest that SEVIR pooled-CSI gaps are primarily driven by spatiotemporal errors rather than marginal intensity calibration, shifting attention toward motion modeling and structure-aware objectives.

---

## References

- [SEVIR](./references/SEVIR-A-Storm-Event-Imagery-Dataset-for-Deep-Learning-Applications-in-Radar-and-Satellite-Meteorology/meta/meta_info.txt) - Veillette et al., 2020.
- [Earthformer](./references/Earthformer-Exploring-Space-Time-Transformers-for-Earth-System-Forecasting/meta/meta_info.txt) - Gao et al., 2022.
- [CasCast](./references/CasCast-Skillful-High-resolution-Precipitation-Nowcasting-via-Cascaded-Modelling/meta/meta_info.txt) - Gong et al., 2024.
- [FlowCast](./references/Under-review-as-a-conference-paper-at-ICLR-2026-FLOWCAST-ADVANCING-PRECIPITATION-NOWCASTING-WITH-CONDITIONAL-FLOW-MATCHING/meta/meta_info.txt) - Ribeiro & Pucer, 2026 (under review).
- [PostCast](./references/PostCast-Generalizable-Postprocessing-for-Precipitation-Nowcasting-via-Unsupervised-Blurriness-Modeling/meta/meta_info.txt) - Gong et al., 2024.
- [Probability calibration](./references/Probability-calibration-for-precipitation-nowcasting/meta/meta_info.txt) - Kurki et al., 2025.
- [PreDiff](./references/PreDiff-Precipitation-Nowcasting-with-Latent-Diffusion-Models/meta/meta_info.txt) - Gao et al., 2023.
- [DiffCast](./references/DiffCast-A-Unified-Framework-via-Residual-Diffusion-for-Precipitation-Nowcasting/meta/meta_info.txt) - (see link).
- [RectiCast](./references/RectiCast-Rectifying-Distribution-Shift-in-Cascaded-Precipitation-Nowcasting/meta/meta_info.txt) - Ju et al., 2025.
- [SimCast](./references/SimCast-Enhancing-Precipitation-Nowcasting-with-Short-to-Long-Term-Knowledge-Distillation/meta/meta_info.txt) - Yin et al., 2025.
- [Fourier Amplitude and Correlation Loss](./references/Fourier-Amplitude-and-Correlation-Loss-Beyond-Using-L2-Loss-for-Skillful-Precipitation-Nowcasting/meta/meta_info.txt) - Yan et al., 2024.
- [Optimization using categorical binary metrics](./references/Optimization-of-deep-learning-precipitation-models-using-categorical-binary-metrics/meta/meta_info.txt) - Larraondo et al., 2020.
- [Nowcasting survey](./references/Deep-learning-for-precipitation-nowcasting-A-survey-from-the-perspective-of-time-series-forecasting/meta/meta_info.txt) - An et al., 2024.
- [ConvLSTM nowcasting](https://arxiv.org/abs/1506.04214) - Shi et al., 2015.
- [PredRNN](https://arxiv.org/abs/2103.09504) - Wang et al., 2022.
- [SimVP](https://arxiv.org/abs/2206.05099) - Gao et al., 2022.
- [MetNet](https://arxiv.org/abs/2003.12140) - Google Research, 2020.
- [rainymotion](https://doi.org/10.5194/gmd-12-1387-2019) - Ayzel et al., 2019.
- [pysteps](https://doi.org/10.5194/gmd-12-4185-2019) - Pulkkinen et al., 2019.
- [RainNet](https://doi.org/10.5194/gmd-2020-30) - Ayzel et al., 2020.
- [Rainformer](https://doi.org/10.1109/LGRS.2022.3162882) - Bai et al., 2022.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) - Guo et al., 2017.
- [TyrainNow](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025WR039897) - (see link), 2025.
- [Bias Correction, Quantile Mapping, and Downscaling](https://doi.org/10.5194/hess-2016-659) - Maraun, 2016.
