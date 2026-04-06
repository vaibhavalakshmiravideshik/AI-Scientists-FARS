# untitled

# Anisotropic Spectral Error Dressing: Order-Dependent Spherical-Harmonic Perturbations for Calibrating Deterministic AI Weather Forecasts

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Key constraints**: No retraining of the base forecaster; fully automated evaluation; verification budget within 768 A100 GPU-hours (expected to be dominated by CPU + I/O).

## Introduction

### Context and Motivation

Modern global medium-range weather forecasts (1–14 days) are increasingly produced by machine-learning models such as **GraphCast** and **Pangu-Weather**. Many of these systems are trained to minimize mean-squared error and are typically evaluated as deterministic predictors, but most downstream decisions (energy trading, logistics, emergency planning) require **probabilistic** forecasts with calibrated uncertainty.

Operational numerical weather prediction (NWP) centers obtain probabilistic forecasts using ensemble prediction systems (EPS), which run many perturbed forecasts and quantify uncertainty through ensemble spread. However, running an operational-quality EPS is expensive, and many ML weather forecasters are released primarily as deterministic model outputs.

**WeatherBench 2 (WB2)** is a public benchmark and evaluation framework for global medium-range forecasting. It evaluates forecasts against **ERA5** (a global atmospheric reanalysis produced by the European Centre for Medium-Range Weather Forecasts, ECMWF) using both deterministic and probabilistic metrics, including:
- **CRPS (continuous ranked probability score; lower is better)**, a strictly proper scoring rule for univariate probabilistic forecasts.
- **Spread–skill ratio (closer to 1 is better)**, which compares ensemble spread to the error of the ensemble mean.

WB2 provides public Zarr datasets for deterministic AI forecasts (e.g., GraphCast) and operational ensembles such as **IFS-ENS** (ECMWF’s operational \(\sim 50\)-member ensemble prediction system), enabling direct comparison of post-processing methods.

### The Problem

A practical open question is how to turn a deterministic AI forecast into a calibrated ensemble **without retraining** the base model.

- **Generative ensemble forecasters** (e.g., diffusion-based GenCast) can produce high-quality ensembles but require training a separate generative model.
- **Heuristic perturbations** (e.g., GraphCast-Perturbed style perturbations used as baselines in GenCast) often use isotropic Gaussian-process-like noise and may not match the forecast error structure of a specific model, variable, and lead time.
- **Training-free spectral error dressing (SED)** (a prior proposal in this repo) samples additive Gaussian perturbations in spherical-harmonic space with variance depending only on spherical-harmonic degree \(l\), matching the empirical residual degree power spectrum \(C_l\) on a calibration split. SED highlights a key limitation: rotation-invariant (isotropic) perturbations may be misspecified for storm-track-dominated variables in the extra-tropics.
- **Flow-dependent perturbations** (e.g., bred vectors as used in recent large-ensemble ML systems) can capture anisotropy more faithfully, but they typically require rerunning the forecaster from perturbed initial conditions. In many practical settings (including public releases of deterministic AI forecasts), we only have access to stored deterministic forecast fields and cannot rerun the model.

This proposal tests whether **within-degree anisotropy** is a key missing component in output-space perturbation wrappers: even if we match the correct scale-dependent spectrum \(C_l\), we may still allocate variance incorrectly across spherical-harmonic orders \(m\) (directional structure), leading to miscalibrated ensembles in regions where errors are directionally organized (e.g., mid-latitude jets and storm tracks).

### Key Insight and Hypothesis

Spherical harmonics \(Y_{lm}(\theta,\phi)\) separate global spatial variability into a **degree** \(l\) (roughly spatial scale) and an **order** \(m\) (approximately a zonal wavenumber). For large \(l\), a plane-wave analogy gives an approximate decomposition into zonal and meridional wavenumbers:
- \(k_{\mathrm{zonal}} \approx |m|\)
- \(k_{\mathrm{meridional}} \approx \sqrt{l(l+1)-m^2}\)
so the ratio \(\mu = |m|/l\) is a coarse proxy for the **orientation** of a mode’s wavevector (how “zonal” it is). As a concrete precedent for interpreting \((l,m)\) this way, Subich et al. explicitly treat spherical-harmonic modes as having a *total wavenumber* and a *zonal wavenumber* when analyzing ML weather model errors in spectral space (see the "modified spherical harmonic loss" work in our references).

Degree-only SED assumes that, conditioned on \(l\), all \(m\) modes have equal expected power, which corresponds to an **isotropic** Gaussian random field on the sphere. In the extra-tropics, forecast errors often organize along mid-latitude jets and storm tracks (corridors of baroclinic weather systems). This tends to produce **quasi-zonal, elongated anomalies** (large meridional gradients but slower longitudinal variation), which correspond more to **low-\(\mu\)** modes than high-\(\mu\) modes. Under this mechanism, isotropic-within-degree perturbations misallocate uncertainty across orientations: they inject too little variance into low-\(\mu\) modes and too much into high-\(\mu\) modes, degrading spread–skill and CRPS in the extra-tropics. This interpretation is consistent with practice in large-ensemble ML systems that treat Z500 as a key extratropical steering-flow field and use hemisphere-dependent perturbation scaling (see Huge Ensembles Part I in our references).

We hypothesize that GraphCast residuals for **Z500** (500 hPa geopotential height, a standard tracer of mid-latitude flow) at synoptic lead times exhibit systematic within-degree anisotropy in \(\mu\), and that matching this anisotropy improves probabilistic calibration.

**Mechanism prediction (directional):** On extratropical Z500 at 5-day lead, we expect \(P_{low}(l) > P_{high}(l)\) on average for \(l\ge 10\), so the calibration anisotropy contrast \(A_{cal}=\mathbb{E}_l\frac{P_{high}(l)-P_{low}(l)}{P_{high}(l)+P_{low}(l)}\) should be **negative**, and ASED should learn \(w_{low} > w_{high}\).

We use a **fixed 2-bin split at \(\mu_0=0.5\)** because it (i) separates “quasi-zonal” modes (\(|m|/l<0.5\)) from “non-zonal” modes while keeping bin counts reasonably balanced for stable estimation, and (ii) yields a two-parameter model that is hard to overfit.

The outcome is uncertain because (i) residual anisotropy might be weak for the chosen variable/lead time, and (ii) even if residuals are anisotropic, the dominant uncertainty may be feature displacement/phase error (off-diagonal spectral correlations) rather than amplitude error captured by \(\mathrm{Var}(a_{lm})\).

---

## Proposed Approach

### Overview

We propose **Anisotropic Spectral Error Dressing (ASED)**: a minimal extension of degree-only SED that (1) estimates the usual degree power spectrum \(C_l\) from historical residuals, and (2) additionally estimates a **two-bin within-degree anisotropy profile** as a function of \(\mu = |m|/l\) (low-\(\mu\) vs high-\(\mu\)). ASED samples Gaussian perturbations in spherical-harmonic space with per-(\(l,m\)) variance proportional to \(C_l\) times the learned bin weight.

Crucially, ASED is constrained so that for every degree \(l\), averaging the variance over \(m\) exactly recovers \(C_l\). This ensures the only difference from SED is **how variance is distributed across \(m\) within each \(l\)**, not total variance or the degree spectrum.

### Method Details

We follow the WB2 setting and the notation in the in-repo SED proposal.

Let \(f(t, x)\) be a deterministic forecast at initialization time \(t\), lead time \(\ell\), and grid point \(x\). Let \(o(t+\ell, x)\) be the verifying analysis (ERA5). Define the residual \(r(t, x) = f(t, x) - o(t+\ell, x)\).

We focus on one variable and lead time in the decisive experiment (Z500 at \(\ell=5\) days).

**Step 0: bias correction (shared by all stochastic baselines)**
- Estimate spatial bias \(b(x) = \mathbb{E}_t[r(t, x)]\) on a calibration split.
- Form bias-corrected forecast \(f_{bc}(t,x)=f(t,x)-b(x)\) and residuals \(r_{bc}(t,x)=f_{bc}(t,x)-o(t+\ell,x)\).

**Step 1: estimate the degree spectrum (same as SED)**
- Compute spherical-harmonic coefficients \(a_{lm}(t)=\mathrm{SHT}(r_{bc}(t,x))\), where **SHT** denotes a spherical harmonic transform.
- Estimate degree power spectrum
  \[
  C_l = \mathbb{E}_t\Big[\frac{1}{2l+1}\sum_{m=-l}^{l} |a_{lm}(t)|^2\Big].
  \]
- Smooth \(C_l\) with a fixed binning scheme (e.g., 10 log-spaced \(l\)-bins; no tuning).

**Step 2: estimate a 2-bin within-degree anisotropy profile**
- Define \(\mu=|m|/l\) and a fixed split at \(\mu_0=0.5\):
  - low-\(\mu\): \(|m|/l < 0.5\)
  - high-\(\mu\): \(|m|/l \ge 0.5\)
- For each degree \(l\), compute the average residual power in each bin:
  - \(P_{low}(l) = \mathbb{E}_t[\mathrm{mean}_{m:|m|/l<0.5}|a_{lm}(t)|^2]\)
  - \(P_{high}(l)= \mathbb{E}_t[\mathrm{mean}_{m:|m|/l\ge 0.5}|a_{lm}(t)|^2]\)
- Convert these into **global bin weights** (2 scalars) by averaging across a fixed degree range \(l\in[l_{min}, l_{max}]\) chosen to avoid tiny-mode-count instability:
  - We will use \(l_{min}=10\); \(l_{max}\) set to the maximum resolved degree on the evaluation grid.
  - Define weights \(w_{low}, w_{high}\) proportional to \(\mathrm{mean}_{l\ge 10} P_{low}(l)\) and \(\mathrm{mean}_{l\ge 10} P_{high}(l)\), then normalized so \(\frac{n_{low}(l)w_{low}+n_{high}(l)w_{high}}{2l+1}=1\) for each \(l\) when applying them (see next step). This keeps the within-l mean variance equal to \(C_l\).

**Step 3: sample perturbations (ASED)**
For each ensemble member \(j=1..M\):
- Sample i.i.d. Gaussian coefficients \(\epsilon_{lm}\sim\mathcal{N}(0,1)\) in a real spherical-harmonic basis.
- Define per-(\(l,m\)) variance multipliers:
  \[
  g_{lm} = \frac{w_{bin(|m|/l)}}{\bar w_l},\quad \bar w_l = \frac{1}{2l+1}\sum_{m=-l}^{l} w_{bin(|m|/l)}.
  \]
- Form perturbation coefficients:
  \[
  \eta_{lm} = \sqrt{C_l\, g_{lm}}\,\epsilon_{lm}.
  \]
- Inverse transform \(\eta_j(x)=\mathrm{iSHT}(\eta_{lm})\), where **iSHT** denotes the inverse spherical harmonic transform.

**Step 4: global variance matching (shared scalar, same as SED)**
- Let \(V_{res}=\mathbb{E}_{t,x}[r_{bc}(t,x)^2]\) on the calibration split.
- Let \(V_{\eta}=\mathbb{E}_{j,x}[\eta_j(x)^2]\).
- Set \(\alpha=\sqrt{V_{res}/V_{\eta}}\).
- Final ensemble members: \(f^{(j)}(t,x)=f_{bc}(t,x)+\alpha\,\eta_j(x)\).

### Key Innovations

- **Within-degree anisotropy as a minimal extension**: redistribute spectral power across \(m\) while holding \(C_l\) fixed, isolating anisotropy from scale-dependent variance.
- **Low-parameter, no-tuning design**: a fixed \(\mu_0=0.5\) split yields only 2 anisotropy weights \((w_{low}, w_{high})\), reducing overfitting risk.
- **Pre-registered anisotropy diagnostic**: report an anisotropy index computed on calibration residuals to interpret positive/negative results.

---

## Related Work

### Field Overview

Probabilistic weather forecasting is traditionally addressed using ensemble prediction systems (EPS) in operational NWP, where uncertainty is represented by running many perturbed forecasts with stochastic physics and/or perturbed initial conditions. For data-driven weather forecasting, current approaches span (i) training generative ensemble models (e.g., diffusion-based models), (ii) training explicit probabilistic predictors, and (iii) post-processing deterministic or ensemble forecasts using statistical calibration methods (e.g., EMOS/NGR, ensemble copula coupling).

WB2 standardizes evaluation of both deterministic and probabilistic forecasts using operational metrics (RMSE/ACC for deterministic skill; CRPS and spread–skill for probabilistic calibration) and provides public datasets for deterministic AI forecasts and operational ensembles.

This proposal focuses on the narrow setting of **training-free post-processing of deterministic AI forecasts**: construct an ensemble by sampling additive perturbations whose second-order structure is estimated from historical forecast errors.

### Related Papers

- **[WeatherBench 2: A benchmark for the next generation of data-driven global weather models](./references/WeatherBench-2-A-benchmark-for-the-next-generation-of-data-driven-global-weather-models/meta/meta_info.txt)**: Defines WB2 datasets, splits, and metrics including CRPS and regional evaluations.
- **[WeatherBench 2 Data Guide](./references/WeatherBench-2-Data-Guide/meta/meta_info.txt)**: Practical documentation for accessing WB2 datasets and running evaluation code.
- **[GraphCast: Learning skillful medium-range global weather forecasting](./references/GraphCast-Learning-skillful-medium-range-global-weather-forecasting/meta/meta_info.txt)**: A strong deterministic ML weather forecaster trained on ERA5 (1979–2017) and evaluated out-of-sample (2018+).
- **[GenCast: Diffusion-based ensemble forecasting for medium-range weather](./references/GenCast-Diffusion-based-ensemble-forecasting-for-medium-range-weather/meta/meta_info.txt)**: Produces probabilistic ensembles via diffusion modeling; includes heuristic perturbed-GraphCast baselines.
- **[Improving medium-range ensemble weather forecasts with hierarchical ensemble transformers](./references/Improving-medium-range-ensemble-weather-forecasts-with-hierarchical-ensemble-transformers/meta/meta_info.txt)**: Learns to improve ensembles; relevant as a learned (non-training-free) alternative.
- **[Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators](./references/Huge-Ensembles-Part-I-Design-of-Ensemble-Weather-Forecasts-using-Spherical-Fourier-Neural-Operators/meta/meta_info.txt)**: Generates large ensembles with spherical spectral structure; relevant for spectral representations.
- **[Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss Function](./references/Fixing-the-Double-Penalty-in-Data-Driven-Weather-Forecasting-Through-a-Modified-Spherical-Harmonic-Loss-Function/meta/meta_info.txt)**: Uses spherical harmonics to address spatial-scale issues in ML weather models.
- **[Neural networks for post-processing ensemble weather forecasts](./references/Neural-networks-for-post-processing-ensemble-weather-forecasts/meta/meta_info.txt)**: Surveys learned post-processing approaches; positions training-free methods as a simpler alternative.
- **[Spatial Postprocessing of Ensemble Forecasts for Temperature Using Nonhomogeneous Gaussian Regression](./references/Spatial-Postprocessing-of-Ensemble-Forecasts-for-Temperature-Using-Nonhomogeneous-Gaussian-Regression/meta/meta_info.txt)**: Canonical EMOS/NGR post-processing baseline family.
- **[Generation of Scenarios from Calibrated Ensemble Forecasts with a Dual-Ensemble Copula-Coupling Approach](./references/Generation-of-Scenarios-from-Calibrated-Ensemble-Forecasts-with-a-Dual-Ensemble-Copula-Coupling-Approach/meta/meta_info.txt)**: Ensemble copula coupling for generating calibrated scenarios.
- **[Locally anisotropic covariance functions on the sphere](./references/Locally-anisotropic-covariance-functions-on-the-sphere/meta/meta_info.txt)**: Statistical modeling of anisotropic covariance on spherical domains; supports the plausibility of spherical anisotropy.
- **[Isotropic Gaussian random fields on the sphere: Regularity, fast simulation and stochastic PDEs](./references/Isotropic-Gaussian-random-fields-on-the-sphere-Regularity-fast-simulation-and-stochastic-partial-differential-equations/meta/meta_info.txt)**: Theory of isotropic spherical random fields and spectral simulation; useful reference for the isotropic baseline assumptions.
- **[The ERA5 global reanalysis](https://doi.org/10.1002/qj.3803)**: Canonical description of ERA5, which is used as ground truth in WB2 and as training data for many ML weather models.
- **[WeatherBench: A benchmark dataset for data-driven weather forecasting](https://arxiv.org/abs/2002.00469)**: The original WeatherBench benchmark that established standardized ML weather evaluation protocols.
- **[Neural general circulation models for weather and climate](https://arxiv.org/abs/2311.07222)**: A hybrid ML-physics global model with an ensemble variant; relevant probabilistic baseline family on WB2.
- **[Strictly proper scoring rules, prediction, and estimation](https://doi.org/10.1198/016214506000001437)**: Foundational paper establishing CRPS as a strictly proper scoring rule for probabilistic forecasts.
- **[Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts](https://doi.org/10.1007/s11004-017-9709-8)**: Practical CRPS estimation methods for ensemble forecasts (cited by WB2).
- **[From ensemble forecasts to predictive distribution functions](https://doi.org/10.1111/j.1600-0870.2008.00333.x)**: Formalizes converting ensembles into predictive distributions via dressing/blending; conceptual predecessor to dressing methods.
- **[Combining dynamical and statistical ensembles](https://doi.org/10.1034/j.1600-0870.2003.201378.x)**: Early kernel dressing method combining dynamical ensembles with statistical error models.
- **[Improvement of ensemble reliability with a new dressing kernel](https://doi.org/10.1256/qj.04.120)**: Affine kernel dressing that calibrates ensemble reliability by optimizing CRPS.
- **[Uncertainty quantification in complex simulation models using ensemble copula coupling](https://doi.org/10.1214/13-STS443)**: Ensemble copula coupling (ECC) for restoring multivariate dependence after univariate calibration.
- **[A kinetic energy backscatter algorithm for use in ensemble prediction systems](https://doi.org/10.1256/qj.04.106)**: Introduces stochastic kinetic energy backscatter (SKEB) as an EPS uncertainty mechanism.
- **[A spectral stochastic kinetic energy backscatter scheme and its impact on flow-dependent predictability in the ECMWF ensemble prediction system](https://doi.org/10.1175/2009mwr2697.1)**: Spectral (spherical-harmonic) SKEB implementation; closest operational precedent for spectral stochastic perturbations.
- **[Spatial modelling using a new class of nonstationary covariance functions](https://doi.org/10.1002/env.785)**: Classic nonstationary anisotropic covariance construction (Paciorek & Schervish), foundational for later spherical anisotropy models.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Deterministic ML weather forecasting | Predict a single best-estimate forecast | GraphCast, Pangu-Weather, FuXi | WB2 RMSE/ACC | No calibrated uncertainty |
| Generative probabilistic forecasters | Train generative model to sample trajectories | GenCast | WB2 CRPS, spread–skill | Requires training + heavier infra |
| Statistical post-processing (learned or classical) | Calibrate an existing ensemble or distributional forecast | EMOS/NGR, ECC; deep post-processing | Operational verification, station data | Usually needs an input ensemble and/or training |
| Training-free perturbation wrappers | Add structured noise to deterministic forecast | GraphCast-Perturbed heuristics; SED | WB2 CRPS | Risk of misspecified covariance (anisotropy, non-Gaussianity) |

### Closest Prior Work

1. **Spectral Error Dressing (SED; in-repo finalized proposal)**: Matches the empirical residual degree spectrum \(C_l\) and samples isotropic perturbations with variance depending only on \(l\). Key limitation: assumes rotation-invariant covariance; may fail in storm-track regions where errors are anisotropic. Our method changes only the within-degree allocation across \(m\) while holding \(C_l\) fixed.

2. **GraphCast-Perturbed style baselines (as described in GenCast)**: Construct ensembles by perturbing initial conditions and/or adding isotropic correlated noise with a fixed length scale. Limitation: perturbations are not fit to the target model’s residual statistics at a specific lead time; anisotropy is not modeled.

3. **Geostatistical anisotropic covariance models on the sphere**: Work in spatial statistics (e.g., Cao et al., 2022) shows that anisotropy on the sphere is common and can materially improve probabilistic scores such as CRPS. However, these models are typically used for spatial interpolation / GP prediction, not as a lightweight wrapper around deterministic global forecast fields.

**Novelty Kill Search Summary:** Local repo search across all `**/proposal.md` found no prior evaluation of m-dependent / zonal-wavenumber-dependent spherical-harmonic perturbations for WB2 post-processing. Web queries used in earlier exploration included: "m-dependent spherical harmonic perturbation weather ensemble", "anisotropic spherical harmonic noise weather post-processing", "zonal wavenumber dependent stochastic perturbation SKEB", and "ensemble dressing anisotropic covariance sphere"; no direct prior work applying order-dependent spectral dressing to WB2 deterministic AI forecasts was identified as of 2026-02-28.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| SED (degree-only) | Matches \(C_l\) residual spectrum and samples isotropic perturbations | Ignores within-degree anisotropy across \(m\) | Add 2-bin \(|m|/l\)-dependent variance multipliers | If residual anisotropy is a bottleneck, better spread–skill and lower CRPS |
| Isotropic correlated noise | Uses fixed isotropic length-scale filter | Not fit to residual spectrum; ignores anisotropy | Use empirical \(C_l\) and anisotropy weights | Matches scale + direction structure better |
| GenCast | Trains diffusion model to generate ensembles | Requires training; heavier pipeline | Training-free wrapper | Much cheaper to deploy; isolates anisotropy as boundary condition |
| Anisotropic GP on sphere | Models anisotropic covariance for spatial data | Not a forecast post-processing wrapper | Use anisotropy concept in spectral dressing | Provides a minimal, scalable anisotropy mechanism |

---

## Experiments

### Experimental Setup

**Task**: Post-process deterministic GraphCast forecasts into an ensemble and evaluate probabilistic skill on WeatherBench2.

**Primary setting (decisive)**:
- Variable: **Z500** (500 hPa geopotential)
- Lead time: **5 days**
- Dataset/split: WB2 year-2020 evaluation, using a fixed time-based split (calibration on odd months, evaluation on even months) consistent with the in-repo SED proposal.

**Methods (3 conditions; no extra main conditions):**
- **A. Deterministic**: Bias-corrected GraphCast forecast (ensemble size 1).
- **B. Degree-only spectral error dressing (SED)**: Sample perturbations in spherical-harmonic space with \(\mathrm{Var}(\eta_{lm})\propto C_l\) (equivalently \(g_{lm}\equiv 1\)), where \(C_l\) is estimated from historical residuals. Apply the same bias correction and global variance matching \(\alpha\) as ASED.
- **C. Anisotropic spectral error dressing (ASED; ours)**: Same as SED but with \(\mathrm{Var}(\eta_{lm})\propto C_l\,g_{lm}\), where \(g_{lm}\) depends only on whether \(|m|/l<0.5\) or \(|m|/l\ge 0.5\), normalized so the within-l mean variance equals \(C_l\).

**Optional context baseline (not part of the decision rule):** We will also report an isotropic correlated-noise baseline with a 1200 km horizontal decorrelation length scale (matching the Gaussian-process perturbations used in GenCast’s GraphCast-Perturbed baseline), to contextualize the gap between fixed isotropic noise and residual-calibrated spectral noise.

**Ensemble size and randomness**:
- Ensemble size \(M=50\) for B and C.
- Use **5 random seeds** for perturbation sampling (e.g., `seeds=[0,1,2,3,4]`), report mean±std across seeds.

**Implementation notes**:
- Use the official WB2 evaluation code for data loading, regridding, and CRPS/spread–skill computation.
- Spherical harmonic transforms can be implemented via an off-the-shelf library (e.g., `pyshtools`) or an existing WB2 spectral utility if present.

**Resource Estimate**:
- **Compute budget**: Expected to be CPU/I/O dominated; <50 GPU-hours (mostly for any preprocessing) and well within the 768 GPU-hour budget.
- **GPU memory**: Minimal (operations on relatively low-resolution grids for verification; can run on a single GPU if needed).
- **Storage/I/O**: Reads subsets of public WB2 Zarr datasets; primary cost is data access and metric aggregation.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| WeatherBench 2 (WB2) | Global medium-range weather forecasting benchmark with deterministic + ensemble evaluation | **CRPS** (lower better), spread–skill ratio (closer to 1 better) | 2020 even months test; 2020 odd months calibration | WB2 data guide + website | `google-research/weatherbench2` |

**Primary metric**:
- **CRPS** as defined in WB2 (univariate probabilistic score; reduces to MAE for a deterministic forecast).

**Secondary metrics**:
- Spread–skill ratio.
- Regional CRPS: **global** and **extra-tropics (30–60° latitude)**.

**Analysis-only diagnostics (not part of decision rule):**
- **Calibration anisotropy index** computed on the calibration split:
  - Define, for each degree \(l\ge 10\), bin-averaged powers \(P_{low}(l), P_{high}(l)\) as above.
  - Define symmetric contrast \(A_{cal} = \mathrm{mean}_{l\ge 10} \frac{P_{high}(l)-P_{low}(l)}{P_{high}(l)+P_{low}(l)}\).
  - Pre-registered interpretation: if \(|A_{cal}|<0.1\), residuals are approximately isotropic within-degree and we expect ASED \(\approx\) SED.

### Main Results

#### Results Table

All numbers are **TBD** (to be filled by verification runs).

| Method | Base Model | Benchmark | CRPS (global; mean±std) | CRPS (30–60°; mean±std) | Spread–skill (global; mean±std) | Source | Notes |
|---|---|---|---|---|---|---|---|
| Deterministic | GraphCast | WB2 2020 (Z500 @ 5d) | TBD | TBD | - | - | CRPS reduces to MAE for M=1 |
| Isotropic correlated noise (GP-1200km; context) | GraphCast + GP noise | WB2 2020 (Z500 @ 5d) | TBD | TBD | TBD | - | Matches GenCast GraphCast-Perturbed GP length scale; variance matched |
| Degree-only SED | GraphCast + spectral noise | WB2 2020 (Z500 @ 5d) | TBD | TBD | TBD | - | \(\mathrm{Var}(\eta_{lm})\propto C_l\); variance matched |
| **ASED (ours)** | GraphCast + anisotropic spectral noise | WB2 2020 (Z500 @ 5d) | TBD | TBD | TBD | - | \(|m|/l\) split at 0.5; within-l mean matches \(C_l\) |
| Reference (context) | IFS-ENS | WB2 2020 (Z500 @ 5d) | TBD | TBD | TBD | - | Context row; not tuned |

### Ablation Studies

No additional main ablations are planned beyond the 3-condition decisive comparison to preserve scope discipline.

### Experimental Rigor

**Variance & Reproducibility:**
- Report mean±std across 5 random seeds for perturbation sampling.

**Validity & Controls:**
- Sanity: setting \(\alpha=0\) must recover deterministic GraphCast exactly.
- CRPS sanity: for ensemble size 1, CRPS should equal MAE (WB2 definition).

**Key confounders and mitigations:**
1. **Variance tuning confound**: both SED and ASED use identical global variance matching \(\alpha\); ASED additionally enforces per-degree mean variance equals \(C_l\), so improvements cannot come from changing \(C_l\) or total variance.
2. **Overfitting to calibration**: ASED has only 2 fixed bin weights and no tuned thresholds; evaluation uses a held-out time split (even months).
3. **Low-l instability**: anisotropy index and weight estimation exclude \(l<10\) to ensure both bins have enough modes.

---

## Success Criteria

**Hypothesis** (directional):
ASED will reduce CRPS relative to **degree-only SED** because the extra freedom to redistribute variance across \(m\) (while holding \(C_l\) and total variance fixed) better matches the directional structure of extratropical residuals, improving spread–skill in the extra-tropics.

**Decision Rule** (concrete):
- **Proceed/Continue**: On WB2 2020 Z500 at 5-day lead, ASED improves CRPS over degree-only SED by **≥1% relative** on **both** (i) global and (ii) extra-tropics (30–60°), with the improvement outside the mean±std range over 5 noise seeds.
- **Refute**: If ASED does not outperform degree-only SED (within noise or worse) on either global or extra-tropics CRPS.
- **Interpretation aid (pre-registered)**: If \(|A_{cal}|<0.1\), residuals are approximately isotropic within-degree and we expect ASED \(\approx\) SED (a null result is informative rather than a failure).

---

## Impact Statement

If ASED works, practitioners using deterministic AI medium-range forecasts could obtain more calibrated uncertainty estimates with a minimal training-free change (a different spectral weighting), improving probabilistic decision-making without training a new generative model. If it fails, it yields a decision-relevant boundary condition: matching scale-dependent variance \(C_l\) is not enough, and future post-processing should focus on different uncertainty representations.

---

## References

- [WeatherBench 2: A benchmark for the next generation of data-driven global weather models](./references/WeatherBench-2-A-benchmark-for-the-next-generation-of-data-driven-global-weather-models/meta/meta_info.txt) - Rasp et al., 2023
- [WeatherBench 2 Data Guide](./references/WeatherBench-2-Data-Guide/meta/meta_info.txt) - WeatherBench2 docs, 2023
- [GraphCast: Learning skillful medium-range global weather forecasting](./references/GraphCast-Learning-skillful-medium-range-global-weather-forecasting/meta/meta_info.txt) - Lam et al., 2022/2023
- [GenCast: Diffusion-based ensemble forecasting for medium-range weather](./references/GenCast-Diffusion-based-ensemble-forecasting-for-medium-range-weather/meta/meta_info.txt) - Price et al., 2023
- [Improving medium-range ensemble weather forecasts with hierarchical ensemble transformers](./references/Improving-medium-range-ensemble-weather-forecasts-with-hierarchical-ensemble-transformers/meta/meta_info.txt) - Ben-Bouallegue et al., 2023
- [Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators](./references/Huge-Ensembles-Part-I-Design-of-Ensemble-Weather-Forecasts-using-Spherical-Fourier-Neural-Operators/meta/meta_info.txt) - Mahesh et al., 2024
- [Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss Function](./references/Fixing-the-Double-Penalty-in-Data-Driven-Weather-Forecasting-Through-a-Modified-Spherical-Harmonic-Loss-Function/meta/meta_info.txt) - Subich et al., 2025
- [Neural networks for post-processing ensemble weather forecasts](./references/Neural-networks-for-post-processing-ensemble-weather-forecasts/meta/meta_info.txt) - Rasp & Lerch, 2018
- [Spatial Postprocessing of Ensemble Forecasts for Temperature Using Nonhomogeneous Gaussian Regression](./references/Spatial-Postprocessing-of-Ensemble-Forecasts-for-Temperature-Using-Nonhomogeneous-Gaussian-Regression/meta/meta_info.txt) - Feldmann et al., 2014
- [Generation of Scenarios from Calibrated Ensemble Forecasts with a Dual-Ensemble Copula-Coupling Approach](./references/Generation-of-Scenarios-from-Calibrated-Ensemble-Forecasts-with-a-Dual-Ensemble-Copula-Coupling-Approach/meta/meta_info.txt) - Bouallegue et al., 2015
- [GitHub - google-research/weatherbench2](./references/GitHub---google-research-weatherbench2/meta/meta_info.txt) - Google Research, 2023
- [Locally anisotropic covariance functions on the sphere](./references/Locally-anisotropic-covariance-functions-on-the-sphere/meta/meta_info.txt) - Cao et al., 2022
- [Isotropic Gaussian random fields on the sphere: Regularity, fast simulation and stochastic PDEs](./references/Isotropic-Gaussian-random-fields-on-the-sphere-Regularity-fast-simulation-and-stochastic-partial-differential-equations/meta/meta_info.txt) - Lang & Schwab, 2013
- [Gneiting et al. 2005 (EMOS/NGR)](https://doi.org/10.1175/MWR2904.1)
- [Raftery et al. 2005 (BMA)](https://doi.org/10.1175/MWR2906.1)
- [Schefzik et al. 2013 (ECC)](https://doi.org/10.1007/s11004-013-9479-7)
- [Lorenz 1969 (multi-scale predictability)](https://doi.org/10.1111/j.2153-3490.1969.tb00444.x)
- [Pangu-Weather](https://arxiv.org/abs/2211.02556)
- [FuXi](https://arxiv.org/abs/2306.12873)
- [FourCastNet](https://arxiv.org/abs/2202.11214)
- [The ERA5 global reanalysis](https://doi.org/10.1002/qj.3803)
- [WeatherBench: A benchmark dataset for data-driven weather forecasting](https://arxiv.org/abs/2002.00469)
- [Neural general circulation models for weather and climate](https://arxiv.org/abs/2311.07222)
- [Strictly proper scoring rules, prediction, and estimation](https://doi.org/10.1198/016214506000001437)
- [Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts](https://doi.org/10.1007/s11004-017-9709-8)
- [From ensemble forecasts to predictive distribution functions](https://doi.org/10.1111/j.1600-0870.2008.00333.x)
- [Combining dynamical and statistical ensembles](https://doi.org/10.1034/j.1600-0870.2003.201378.x)
- [Improvement of ensemble reliability with a new dressing kernel](https://doi.org/10.1256/qj.04.120)
- [Uncertainty quantification in complex simulation models using ensemble copula coupling](https://doi.org/10.1214/13-STS443)
- [A kinetic energy backscatter algorithm for use in ensemble prediction systems](https://doi.org/10.1256/qj.04.106)
- [A spectral stochastic kinetic energy backscatter scheme and its impact on flow-dependent predictability in the ECMWF ensemble prediction system](https://doi.org/10.1175/2009mwr2697.1)
- [Spatial modelling using a new class of nonstationary covariance functions](https://doi.org/10.1002/env.785)
- [ENS-10: A Dataset For Post-Processing Ensemble Weather Forecasts](https://arxiv.org/abs/2206.14786)
