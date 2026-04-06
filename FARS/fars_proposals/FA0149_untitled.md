# untitled

# Training-Free Motion-Bias Calibration for SEVIR Nowcasting via Phase-Correlation Warping

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Precipitation nowcasting (0–2 hour forecasts) supports time-critical decisions such as flash-flood warning, aviation routing, and severe storm monitoring. A common ML benchmark is **SEVIR** (Storm EVent ImageRy), which contains >10k storm events and includes radar-derived **Vertically Integrated Liquid (VIL)** images at 1 km spatial resolution and 5 min cadence. In the SEVIR nowcasting protocol, models predict 12 future frames (60 min) from 13 context frames (65 min).

Deep learning nowcasters are often trained with pixel-wise regression losses (L1/L2/MSE) and evaluated with **categorical, thresholded event-detection metrics** such as the **Critical Success Index (CSI; hits/(hits+misses+false alarms); higher is better)** at multiple VIL thresholds. Because small spatial shifts can cause large pixel-wise penalties (“double penalty”), SEVIR reports **pooled CSI** variants such as **POOL16**, which apply 16×16 max-pooling to tolerate local displacement. Even so, modern deterministic backbones still lag strong cascaded or probabilistic systems on extreme thresholds. For example, CasCast reports that an EarthFormer deterministic model achieves **CSI-219-POOL16 = 0.1481**, while a cascaded EarthFormer+diffusion approach reaches **0.2841** (CasCast Table 2).

This repository already contains proposals targeting other SEVIR failure modes: (i) training-free **intensity calibration** (quantile remapping) and (ii) training-time **metric-aligned losses** (pooled-CSI and spatiotemporal FSS losses). This proposal targets a different axis that is often discussed in verification and radar nowcasting: **systematic phase (location/timing) errors**.

### The Problem

Deterministic deep nowcasters can produce forecasts that have plausible storm structure but are consistently **mis-advected**: storms move too slowly or too quickly relative to the input-context motion. Such a bias is hard to “see” in per-frame pooled metrics, but it can materially reduce CSI at high thresholds: a storm that arrives one frame late can create both a miss and a false alarm in adjacent frames, and a storm that is consistently shifted along its motion direction can still be penalized even under pooling.

Classical radar nowcasting systems explicitly estimate motion fields and extrapolate echoes (e.g., MAPLE / PySTEPS), while newer deep models often learn motion implicitly. Recent deep architectures also suggest that **position dynamics are a separable factor**: AlphaPre (CVPR 2025) argues that in the Fourier domain, **phase encodes position changes** while amplitude encodes intensity changes, and designs a dedicated phase network for location prediction. In numerical weather prediction postprocessing, Radhakrishna et al. (JAS 2013) show that correcting **spectral phase** using radar observations substantially improves rainfall-field alignment and CSI, indicating that phase errors can dominate early forecast degradation.

However, there is little evidence (in this repo’s SEVIR-focused work and in our novelty searches) that practitioners apply a **training-free, motion-specific calibration layer** on top of deterministic deep nowcasters, with a control that distinguishes “true motion correction” from generic spatial perturbation artifacts.

### Key Insight and Hypothesis

**Key insight:** If a deterministic nowcaster has a systematic *under/over-advection bias* along the true motion direction, then a simple **global speed-scale calibration** combined with a per-example motion direction estimate should improve CSI on the test set more than an equally strong **random-direction warp** (which matches interpolation/smoothing effects but does not correct motion).

**Hypothesis:** On SEVIR, applying a training-free **phase-correlation-based motion warp** with a validation-fitted global speed scale \(\alpha\) will improve **CSI-M-POOL16** and **CSI-219-POOL16** for a fixed EarthFormer checkpoint, and the improvement will be larger than a matched random-direction warp control.

**Why we could be wrong:** (i) SEVIR scenes are not well approximated by a single translation (multiple storm cells, growth/decay), so a global motion vector is noisy and the calibrated \(\alpha\) collapses to ~1. (ii) Any CSI gains are due to interpolation artifacts rather than motion correction; the random-direction control should expose this. (iii) The dominant error is intensity/structure rather than phase; then motion warping will not help.

---

## Proposed Approach

### Overview

We propose **Motion-Bias Calibration (MBC)**: a training-free postprocessing step for deterministic SEVIR nowcasting models.

Given a deterministic nowcaster that predicts \(\hat{Y}_{1:T}\) from context frames \(X_{1:T_{in}}\), MBC:
1) estimates a per-example **context motion vector** \(v\) from the last \(K\) context frames using **FFT phase correlation** (a classical translation estimator),
2) fits a single global **speed-scale** \(\alpha\) on the validation set to capture systematic under/over-advection,
3) applies a cumulative translation warp to each forecast frame: \(\tilde{Y}_t = \text{Shift}(\hat{Y}_t, (\alpha-1)\, t\, v)\).

The method is intentionally simple: it does not retrain the nowcaster and uses only the validation split for calibration.

### Method Details

#### A. Motion estimation via phase correlation
For each validation/test example, estimate a global translation vector \(v\in\mathbb{R}^2\) from context frames using phase correlation. Concretely:
- Choose \(K=3\) recent context frames (e.g., last 15 minutes).
- Compute phase-correlation shifts between \(X_{T_{in}-1}\) and \(X_{T_{in}}\) (optionally average over pairs within the K window).
- Use the peak of the inverse FFT of the cross-power spectrum to estimate an integer-pixel translation; optionally refine with quadratic peak fitting for subpixel accuracy.

This yields a robust *global* motion direction estimate without dense optical flow.

#### B. Fit a global speed-scale parameter on validation
We want a calibration that does **not** use future ground truth at test time.

On the validation split:
- Estimate the model’s implied motion magnitude per example, \(|v_{\hat{Y}}|\), by applying the same phase-correlation procedure to consecutive predicted frames (e.g., \(\hat{Y}_1\) vs \(\hat{Y}_2\)).
- Estimate context motion magnitude \(|v_X|\) from context frames.
- Fit a single scalar \(\alpha\) by least squares regression \(|v_{\hat{Y}}| \approx \alpha |v_X|\) (or robust regression).

Interpretation:
- \(\alpha<1\): model under-advects (moves too slowly)
- \(\alpha>1\): model over-advects

If \(|\alpha-1|<0.1\), we treat this as “no detectable systematic motion bias” and expect MBC to have negligible effect.

#### C. Apply cumulative warping to forecasts
For each forecast lead time \(t\in\{1,\dots,T\}\), shift \(\hat{Y}_t\) by \((\alpha-1)\,t\,v_X\) using bilinear interpolation (or nearest-neighbor on binary exceedance masks as a sensitivity check). Boundary handling: pad with zeros.

### Key Innovations

- **Motion-specific, training-free calibration layer** for deterministic deep nowcasters on SEVIR, targeting phase/speed bias rather than intensity calibration or training objectives.
- **Decisive control**: a matched random-direction warp isolates true motion correction from generic interpolation artifacts.
- **Diagnostic separation of “displacement headroom”** using a shift-invariant CSI oracle to estimate how much error is attributable to displacement.

---

## Related Work

### Field Overview

Precipitation nowcasting methods can be grouped into: (i) classical extrapolation approaches that estimate motion and advect radar echoes (e.g., MAPLE/PySTEPS), (ii) deterministic deep predictors (ConvLSTM/SimVP/EarthFormer), (iii) probabilistic generative models (GAN/diffusion/flow) that aim to produce sharp ensembles (DGMR, NowcastNet, diffusion models), and (iv) postprocessing and calibration techniques (deblurring, intensity calibration, distribution-shift rectification).

A long-running theme in verification is that displacement errors can dominate short-range forecast degradation, motivating neighborhood metrics (FSS, pooled CSI) and object-based scores. Recent deep-learning work also incorporates explicit motion modules (Lagrangian warping, optical-flow supervision) to address motion and “double penalty.” In contrast, this proposal focuses on a minimal, training-free, motion-specific correction that can be applied to an existing deterministic checkpoint.

### Related Papers

- **[SEVIR: A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology](https://arxiv.org/abs/2006.09466)**: Introduces the SEVIR dataset and standard nowcasting protocol.
- **[Earthformer: Exploring Space-Time Transformers for Earth System Forecasting](./references/Earthformer-Exploring-Space-Time-Transformers-for-Earth-System-Forecasting/meta/meta_info.txt)**: Deterministic transformer baseline for SEVIR and other Earth-system tasks.
- **[CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling](./references/CasCast-Skillful-High-resolution-Precipitation-Nowcasting-via-Cascaded-Modelling/meta/meta_info.txt)**: Cascaded deterministic+diffusion framework and a strong source of SEVIR baseline metrics.
- **[RectiCast: Rectifying Distribution Shift in Cascaded Precipitation Nowcasting](./references/RectiCast-Rectifying-Distribution-Shift-in-Cascaded-Precipitation-Nowcasting/meta/meta_info.txt)**: Separates mean-field shift rectification from stochastic generation in cascaded nowcasting.
- **[PostCast: Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Modeling](https://arxiv.org/abs/2410.05805)**: Postprocesses nowcasts by modeling and removing blur with an unconditional diffusion model.
- **[PreDiff: Precipitation Nowcasting with Latent Diffusion Models](https://arxiv.org/abs/2307.10422)**: Latent diffusion approach for probabilistic nowcasting.
- **[DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting](https://arxiv.org/abs/2312.06734)**: Residual diffusion framework for nowcasting.
- **[NowcastNet](https://www.nature.com/articles/s41586-023-06184-4)**: GAN-based nowcasting emphasizing extreme precipitation.
- **[DGMR](https://arxiv.org/abs/2104.00954)**: Deep generative model for probabilistic radar nowcasting.
- **[ConvLSTM](https://arxiv.org/abs/1506.04214)**: Classic recurrent baseline for precipitation nowcasting.
- **[PredRNN](https://arxiv.org/abs/2103.09504)**: Strong recurrent baseline for spatiotemporal prediction, used in nowcasting.
- **[SimVP](https://arxiv.org/abs/2206.05099)**: Non-recurrent video prediction model used as deterministic nowcasting baseline.
- **[Can we integrate spatial verification methods into neural network loss functions for atmospheric science?](https://arxiv.org/abs/2203.11141)**: Differentiable neighborhood verification losses (FSS-style) for training.
- **[Fully differentiable Lagrangian convolutional neural network for physics-informed precipitation nowcasting (LUPIN)](./references/Fully-Differentiable-Lagrangian-Convolutional-Neural-Network-for-Continuity-Consistent-Physics-Informed-Precipitation-Nowcasting/meta/meta_info.txt)**: Uses a differentiable semi-Lagrangian warp to reduce double-penalty.
- **[Precipitation nowcasting of satellite data using physically-aligned neural networks (TUPANN)](./references/Precipitation-nowcasting-of-satellite-data-using-physically-aligned-neural-networks/meta/meta_info.txt)**: Optical-flow supervision and differentiable advection for interpretable motion fields.
- **[AlphaPre: Amplitude-Phase Disentanglement Model for Precipitation Nowcasting](./references/AlphaPre-Amplitude-Phase-Disentanglement-Model-for-Precipitation-Nowcasting/meta/meta_info.txt)**: Explicitly models phase (position) and amplitude (intensity) separately.
- **[Postprocessing Model-Predicted Rainfall Fields in the Spectral Domain Using Phase Information from Radar Observations](./references/Postprocessing-Model-Predicted-Rainfall-Fields-in-the-Spectral-Domain-Using-Phase-Information-from-Radar-Observations/meta/meta_info.txt)**: Spectral-domain phase correction for rainfall fields using radar phase information.
- **[pysteps: an open-source Python library for probabilistic precipitation nowcasting](https://doi.org/10.5194/gmd-12-4185-2019)**: Implements optical-flow and spectral nowcasting pipelines (Lucas–Kanade, VET, DARTS).
- **[MAPLE (McGill Algorithm for Prediction by Lagrangian Extrapolation)](https://doi.org/10.1175/1520-0493(2002)130<2859:ROAE>2.0.CO;2)**: Radar echo extrapolation with scale-dependent predictability.
- **[The phase correlation image alignment method](http://boutigny.free.fr/Astronomie/AstroSources/Kuglin-Hines.pdf)**: Classical phase correlation method for translation estimation.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Extrapolation nowcasting | Estimate motion and advect echoes | MAPLE, pysteps | Radar nowcasting; CSI/FSS | No learned growth/decay; rigid motion assumptions |
| Deterministic deep predictors | Learn dynamics end-to-end | ConvLSTM, PredRNN, SimVP, EarthFormer | SEVIR CSI / pooled CSI | Blurry, can have phase/timing bias |
| Probabilistic / cascaded models | Add stochastic refinement (GAN/diffusion/flow) | DGMR, NowcastNet, PreDiff, DiffCast, CasCast, RectiCast | SEVIR CSI + CRPS | Higher compute; complex training |
| Physics/motion modules | Explicit warping / motion decomposition | LUPIN, TUPANN, AlphaPre | Multiple datasets | Requires retraining / architecture changes |
| Postprocessing | Deblur / calibrate outputs | PostCast, quantile mapping | SEVIR CSI/CRPS | May target blur/intensity rather than motion |

### Closest Prior Work

- **Radhakrishna et al. 2013 (JAS)**: Corrects NWP rainfall fields by replacing spectral phase with radar-observed phase and extrapolating phase forward; demonstrates large CSI improvements but relies on contemporaneous/future radar phase information and targets NWP, not deep nowcasting checkpoints.
- **AlphaPre (CVPR 2025)**: Builds an explicit phase network to predict position changes (training-time change), whereas we apply phase-correlation warping as a training-free calibration layer.
- **LUPIN (2024)**: Uses differentiable Lagrangian warping inside the model to enforce motion consistency; we instead test whether a minimal post-hoc warp already recovers some motion-related skill.
- **RectiCast (2025)**: Rectifies deterministic distribution shift before stochastic generation; our focus is specifically motion bias (phase) and uses a random-warp control.
- **PostCast (2024)**: Deblurs nowcasts with diffusion-based postprocessing; orthogonal to motion bias correction.

**Novelty Kill Search Summary:** Searched for combinations of “SEVIR nowcasting postprocessing shift/translation correction”, “phase correlation radar nowcasting postprocessing”, “motion bias calibration nowcasting”, and “spectral-domain phase correction rainfall field postprocessing”. Found (i) NWP/radar spectral phase correction work (Radhakrishna et al. 2013) and (ii) training-time phase/amplitude disentanglement (AlphaPre), but no work that applies a training-free motion-bias calibration with a matched random-warp control to SEVIR deterministic checkpoints as of 2026-02-19.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Radhakrishna et al. 2013 | Replace model spectral phase with radar phase; extrapolate phase | Needs radar phase for correction; NWP setting | Use only input-context motion + val calibration | Applicable to SEVIR test-time without future truth |
| AlphaPre 2025 | Train phase/amplitude disentangled networks | Requires training a new model | Post-hoc correction on existing checkpoint | Low cost; tests whether motion bias is systematic |
| LUPIN 2024 | Differentiable Lagrangian warp inside NN | Retraining + architecture complexity | Training-free warp layer | Minimal baseline for “motion modules” |
| PostCast 2024 | Diffusion deblurring postprocess | High compute + targets blur | Motion-specific correction | Complementary to blur/intensity corrections |
| Quantile remap (repo proposal) | Post-hoc intensity CDF matching | Doesn’t fix displacement | Post-hoc motion calibration | Targets orthogonal error axis |

---

## Experiments

### Experimental Setup

**Base model / checkpoint**:
- Use the official **EarthFormer SEVIR** checkpoint (amazon-science repo; S3 link): `https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_sevir.pt`.
- Run inference on SEVIR val/test to produce \(\hat{Y}\).

**Three main conditions (decisive):**
1. **No postprocessing**: raw EarthFormer predictions.
2. **Random-direction warp (control)**: shift each predicted frame by the same magnitude schedule as (3), but with direction randomized per example (or rotated by 90°). This matches interpolation/smoothing artifacts.
3. **Motion-bias calibration (MBC, ours)**: estimate per-example context motion vector \(v_X\) by phase correlation; fit global speed-scale \(\alpha\) on validation; apply cumulative warp \((\alpha-1) t v_X\).

**Prerequisite sanity check (fully automated):**
- On a small subset of validation examples (e.g., 50), compute the distribution of motion-vector norms \(|v_X|\) and its stability across the K window. If motion estimation is unstable (e.g., high variance / frequent near-zero vectors), expect a null result and treat it as evidence against global motion correction.

**Implementation notes:**
- Use FFT-based phase correlation for translation estimation (can use `skimage.registration.phase_cross_correlation` or an equivalent implementation).
- Warping uses bilinear interpolation on continuous VIL values; boundary padding = 0.
- Calibration uses validation only; test split is never used to fit \(\alpha\).

**Published baseline numbers (for context; same protocol as CasCast Table 2):**
- EarthFormer (official checkpoint): CSI-M-POOL16 0.4351; CSI-219-POOL16 0.1481; CRPS 0.0251.
- SimVP: CSI-M-POOL16 0.4530; CSI-219-POOL16 0.1685; CRPS 0.0259.
- PredRNN: CSI-M-POOL16 0.4623; CSI-219-POOL16 0.1909; CRPS 0.0271.
- CasCast: CSI-M-POOL16 0.5225; CSI-219-POOL16 0.2841; CRPS 0.0202.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SEVIR VIL nowcasting | 13 context frames → 12 forecast frames of VIL radar | CSI-M-POOL16, CSI-219-POOL16, CRPS/MAE; plus diagnostics below | val (fit), test (eval) | https://registry.opendata.aws/sevir/ | CasCast eval: https://github.com/OpenEarthLab/CasCast |

**Metric definitions:**
- **CSI-M**: mean CSI over thresholds \([16,74,133,160,181,219]\).
- **CSI-219**: CSI at threshold 219 (regional extreme precipitation proxy).
- **POOL16**: compute CSI after 16×16 max-pooling of binary exceedance masks.
- **CRPS**: proper scoring rule; for deterministic forecasts it reduces to **MAE** (lower is better).

**Mechanism/robustness diagnostics (no extra conditions):**
1. **Shift-invariant CSI oracle**: for each frame, compute CSI under translations within a small window (e.g., \(\pm 8\) px in x/y) and take the maximum. This estimates displacement-limited headroom.
2. **Gap closed fraction**: \(\frac{\text{CSI}(\text{MBC})-\text{CSI}(\text{base})}{\text{CSI}(\text{oracle})-\text{CSI}(\text{base})}\). A motion fix should close a non-trivial fraction (e.g., ≥30%).
3. **Per-lead-time CSI-219**: improvements should persist across lead times rather than a single frame.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | CSI-M-POOL16 ↑ | CSI-219-POOL16 ↑ | CRPS ↓ | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| EarthFormer (raw) | EarthFormer | SEVIR | 0.4351 | 0.1481 | 0.0251 | CasCast Table 2 | official checkpoint per caption |
| SimVP (raw) | SimVP | SEVIR | 0.4530 | 0.1685 | 0.0259 | CasCast Table 2 | deterministic baseline |
| PredRNN (raw) | PredRNN | SEVIR | 0.4623 | 0.1909 | 0.0271 | CasCast Table 2 | deterministic baseline |
| CasCast | EarthFormer + diffusion | SEVIR | 0.5225 | 0.2841 | 0.0202 | CasCast Table 2 | strong reference (not re-run) |
| Random-direction warp (control) | EarthFormer | SEVIR | **TBD** | **TBD** | **TBD** | This proposal | 3 seeds (warp randomness) |
| **Motion-bias calibration (MBC)** | EarthFormer | SEVIR | **TBD** | **TBD** | **TBD** | This proposal | 3 seeds |

### Ablation Studies

If MBC shows gains beyond the control, run one small ablation (no additional main conditions):

| Variant | What’s changed | Expected finding |
|---|---|---|
| MBC (no calibration) | Force \(\alpha=1\) but keep direction \(v_X\) | Reverts to baseline (shows \(\alpha\) matters) |

### Experimental Rigor

- **Seeds / randomness**: Base model inference is deterministic; randomness comes from (i) random-direction warp and (ii) any subsampling used in diagnostics. Run **3 seeds** for conditions (2) and (3) by varying random warp directions and any sampled subsets; report mean±std.
- **Confounder 1 (interpolation artifacts)**: Random-direction warp control matches shift magnitudes.
- **Confounder 2 (data leakage)**: Fit \(\alpha\) only on validation; never tune on test.
- **Confounder 3 (metric gaming)**: Report CRPS/MAE and the oracle-gap-closed diagnostic.

**Resource Estimate**:
- **Compute budget**: ~50–150 GPU-hours.
  - EarthFormer inference on SEVIR val+test dominates; postprocessing and metrics are CPU/FFT-heavy but modest.
  - If needed, verify first on a smaller deterministic baseline (e.g., a subset of test) for a smoke test, then scale to full test.
- **GPU memory**: EarthFormer inference should fit on 1×A100-80GB.
- **API usage**: none.

---

## Success Criteria

**Hypothesis** (directional)
- Motion-bias calibration improves pooled CSI at extreme thresholds by correcting systematic under/over-advection along the context motion direction.

**Decision Rule** (concrete)
- **Proceed** if MBC improves **CSI-219-POOL16** by **≥0.02 absolute** *and* improves **CSI-M-POOL16** by **≥0.01 absolute** over the raw baseline, and also beats the **random-direction warp control** by a margin outside its 3-seed std, while **CRPS** worsens by **≤2% relative**.
- **Pivot** if MBC improves CSI over baseline but is indistinguishable from the random-direction warp, suggesting gains are due to interpolation artifacts; in this case, abandon motion-specific claims.
- **Refute** if MBC does not beat the random-direction warp control or if the fitted \(|\alpha-1|<0.1\) (no systematic motion bias detected) and CSI improvements are within noise.

---

## Impact Statement

If successful, MBC provides a **drop-in, training-free** postprocessing step for deterministic nowcasters that can be applied before investing in expensive retraining or cascaded generative refinement. If it fails, the result is still decision-relevant: it would suggest that SEVIR pooled-CSI gaps for deterministic backbones are not primarily due to a simple systematic motion bias, and that efforts should focus on intensity/structure modeling or training-time motion modules.

---

## References

- [SEVIR: A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology](https://arxiv.org/abs/2006.09466) - Veillette et al., 2020.
- [Earthformer: Exploring Space-Time Transformers for Earth System Forecasting](./references/Earthformer-Exploring-Space-Time-Transformers-for-Earth-System-Forecasting/meta/meta_info.txt) - Gao et al., 2022.
- [CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling](./references/CasCast-Skillful-High-resolution-Precipitation-Nowcasting-via-Cascaded-Modelling/meta/meta_info.txt) - Gong et al., 2024.
- [RectiCast: Rectifying Distribution Shift in Cascaded Precipitation Nowcasting](./references/RectiCast-Rectifying-Distribution-Shift-in-Cascaded-Precipitation-Nowcasting/meta/meta_info.txt) - Ju et al., 2025.
- [PostCast: Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Modeling](https://arxiv.org/abs/2410.05805) - Gong et al., 2024.
- [PreDiff: Precipitation Nowcasting with Latent Diffusion Models](https://arxiv.org/abs/2307.10422) - Gao et al., 2023.
- [DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting](https://arxiv.org/abs/2312.06734) - Yu et al., 2024.
- [NowcastNet](https://www.nature.com/articles/s41586-023-06184-4) - Zhang et al., 2023.
- [DGMR](https://arxiv.org/abs/2104.00954) - Ravuri et al., 2021.
- [ConvLSTM](https://arxiv.org/abs/1506.04214) - Shi et al., 2015.
- [PredRNN](https://arxiv.org/abs/2103.09504) - Wang et al., 2022.
- [SimVP](https://arxiv.org/abs/2206.05099) - Gao et al., 2022.
- [Can we integrate spatial verification methods into neural network loss functions for atmospheric science?](https://arxiv.org/abs/2203.11141) - Lagerquist & Ebert-Uphoff, 2022.
- [Fully differentiable Lagrangian convolutional neural network for physics-informed precipitation nowcasting (LUPIN)](./references/Fully-Differentiable-Lagrangian-Convolutional-Neural-Network-for-Continuity-Consistent-Physics-Informed-Precipitation-Nowcasting/meta/meta_info.txt) - Pavl'ik et al., 2024.
- [Precipitation nowcasting of satellite data using physically-aligned neural networks (TUPANN)](./references/Precipitation-nowcasting-of-satellite-data-using-physically-aligned-neural-networks/meta/meta_info.txt) - Catao et al., 2025.
- [AlphaPre: Amplitude-Phase Disentanglement Model for Precipitation Nowcasting](./references/AlphaPre-Amplitude-Phase-Disentanglement-Model-for-Precipitation-Nowcasting/meta/meta_info.txt) - Lin et al., 2025.
- [Postprocessing Model-Predicted Rainfall Fields in the Spectral Domain Using Phase Information from Radar Observations](./references/Postprocessing-Model-Predicted-Rainfall-Fields-in-the-Spectral-Domain-Using-Phase-Information-from-Radar-Observations/meta/meta_info.txt) - Radhakrishna et al., 2013.
- [pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0)](https://doi.org/10.5194/gmd-12-4185-2019) - Pulkkinen et al., 2019.
- [Real-time optical flow techniques for radar-based rainfall nowcasting](https://doi.org/10.3390/atmos8030048) - Woo & Wong, 2017.
- [MAPLE / scale-dependent radar echo extrapolation](https://doi.org/10.1175/1520-0493(2002)130<2859:ROAE>2.0.CO;2) - Germann & Zawadzki, 2002.
- [The phase correlation image alignment method](http://boutigny.free.fr/Astronomie/AstroSources/Kuglin-Hines.pdf) - Kuglin & Hines, 1975.
