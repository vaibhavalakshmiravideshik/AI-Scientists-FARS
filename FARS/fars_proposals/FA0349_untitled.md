# untitled

# Overlap-Resampled L-BFGS for Physics-Informed Neural Networks

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR (or similar ML venues with SciML tracks)

## Introduction

### Context and Motivation

Physics-informed neural networks (PINNs) solve forward and inverse problems for partial differential equations (PDEs) by training a neural network to minimize a loss that combines (i) a **data loss** (e.g., boundary/initial conditions or observations) and (ii) a **physics loss** given by the PDE residual evaluated at **collocation points** sampled in the domain. PINNs have attracted interest as mesh-free solvers and inverse-modeling tools, but training is often unstable and sensitive to optimization choices.

In practice, many PINN pipelines use a two-stage optimizer: a first-order method (often Adam) for initial progress, followed by a quasi-Newton method (often L-BFGS) to reach high-accuracy solutions. For example, the PINN formulation by Raissi et al. popularized using full-batch L-BFGS for PDE problems with thousands of training points.

Separately, multiple works show that **how collocation points are chosen or refreshed during training** can strongly affect reliability. In the ice-shelf inverse problem studied by Iwasaki & Lai, training outcomes can be bimodal (“clustering”, i.e., repeated runs split into a low-error and a high-error mode), and their accompanying code repository reports that **collocation resampling** (refreshing collocation points after each iteration) can yield a **2–3 orders of magnitude decrease in predictive errors** compared to fixed collocation points.

### The Problem

Despite the benefits of resampling, it is widely stated to be **incompatible with L-BFGS** because quasi-Newton methods estimate curvature from gradient history computed on a consistent objective.

- **[pinn_clusters (README)](./references/GitHub---YaoGroup-pinn_clusters/meta/meta_info.txt)** (code release for Iwasaki & Lai): explicitly warns that “L-BFGS … is estimated based on the evaluation of gradients over the past several iterations … thus L-BFGS is incompatible with the collocation resampling method … Resampling the collocation points after every iteration will cause training to terminate prematurely.”

As a result, practitioners often use a workaround: **resample during an Adam phase**, then **freeze collocation points during an L-BFGS phase**. This creates a gap between two widely-used ideas:

1) quasi-Newton refinement (L-BFGS) for high-accuracy PINNs, and
2) online collocation resampling for reliability / coverage.

### Key Insight and Hypothesis

Stochastic and distributed optimization has already studied how to run quasi-Newton methods when the “batch” changes between iterations. In **multi-batch L-BFGS**, the curvature pair uses only the **overlap set** between consecutive batches to keep curvature estimates consistent.

- **[A Multi-Batch L-BFGS Method for Machine Learning](./references/A-Multi-Batch-L-BFGS-Method-for-Machine-Learning/meta/meta_info.txt)** defines an overlap set \(O_k = S_k \cap S_{k+1}\) and computes \(y_{k+1} = g^{O_k}_{k+1} - g^{O_k}_k\) (Sec. 2.1).

We hypothesize that treating **collocation points as the batch** and computing curvature pairs on the overlap set makes **“L-BFGS + collocation resampling” viable** in PINNs. Concretely:

- **Hypothesis**: overlap-resampled L-BFGS with a moderate overlap (pre-registered \(o=0.5\)) avoids premature stopping under collocation resampling and improves accuracy/reliability compared to (i) Adam-with-resampling run to the same compute budget and (ii) the common workaround “Adam-resampling warmstart → fixed-collocation L-BFGS”.

Why this might fail: the overlap-set curvature estimate could be too noisy for PDE residual losses, making the quasi-Newton memory unstable unless the overlap is near 1 (degenerating to fixed collocation).

---

## Proposed Approach

### Overview

We propose an **overlap-resampled L-BFGS** optimizer for PINNs:

- Keep boundary/initial-condition points fixed.
- Resample PDE collocation points every quasi-Newton step, but enforce an overlap between consecutive collocation sets.
- Compute quasi-Newton curvature pairs using gradients evaluated only on the overlap points.

This mirrors multi-batch L-BFGS’s overlap-set construction, but specialized to the PINN setting where only the PDE-residual term changes under resampling.

### Method Details

Let \(\theta\) be network parameters. A typical PINN loss is
\[\mathcal{L}(\theta; C) = \mathcal{L}_{\text{data}}(\theta) + \lambda\,\mathcal{L}_{\text{pde}}(\theta; C),\]
where \(C\) is the set of PDE collocation points. We treat
\(S_k = D \cup C_k\) as the “batch”, where \(D\) are fixed data points and \(C_k\) are collocation points at iteration \(k\).

**Batch transition with overlap**
- Choose \(|C_k| = N\).
- Sample \(C_{k+1}\) by keeping an overlap subset \(O_k^c = C_k \cap C_{k+1}\) of size \(|O_k^c| = oN\), and resampling the remaining \((1-o)N\) points.
- The full overlap set for gradients is \(O_k = D \cup O_k^c\).

**Overlap-set curvature pair**
- Compute the usual L-BFGS step using gradients on \(S_k\):
  \(p_k = -H_k\, g_k^{S_k}\), where \(g_k^{S_k} = \nabla_\theta \mathcal{L}(\theta_k; C_k)\).
- After updating \(\theta_{k+1} = \theta_k + \alpha_k p_k\), compute curvature pair:
  - \(s_{k+1} = \theta_{k+1} - \theta_k\)
  - \(y_{k+1} = g_{k+1}^{O_k} - g_k^{O_k}\), where \(g_k^{O_k} = \nabla_\theta \mathcal{L}(\theta_k; O_k^c)\) (data points included consistently).

**Stability / implementation details**
- Use standard L-BFGS implementation (the “two-loop recursion” used to apply the limited-memory inverse-Hessian approximation) and history size \(m\in[10,20]\).
- Use a cautious update rule (skip storing \((s,y)\) when \(y^\top s\) is too small), as in robust multi-batch L-BFGS.
- Use 64-bit floating point arithmetic (**FP64**) throughout, since insufficient precision can cause premature L-BFGS stopping in PINNs.

Pre-registered overlap settings:
- Main: \(o=0.5\) (chosen as a conservative middle ground: Berahas et al. study smaller overlaps in ML minibatching, but PDE-residual gradients may require larger overlap for stable curvature estimates)
- One ablation: \(o=0.25\) (reported regardless of outcome)
- “Practical failure” criterion: only stable for \(o\ge 0.8\)

### Key Innovations

1. **Algorithmic bridge**: transfers the overlap-set curvature-pair idea from multi-batch quasi-Newton optimization to PINN collocation resampling.
2. **Targets a documented incompatibility**: directly addresses the failure mode stated in pinn_clusters (“resampling … will cause training to terminate prematurely”).
3. **Compute-fair evaluation protocol**: compares methods under a fixed budget of **forward+backward gradient evaluations**, accounting for line search and overlap-gradient costs.

---

## Related Work

### Field Overview

PINN performance depends strongly on (i) optimization algorithms (first-order vs quasi-Newton vs second-order variants), (ii) the choice and updating of collocation points (fixed, resampled, or residual-based sampling), and (iii) numerical issues such as ill-conditioning and finite-precision stopping criteria. Recent work has emphasized that some “failure modes” are optimization/precision artifacts rather than unavoidable local minima.

Our proposal sits at the intersection of two areas that are often treated separately: **online collocation point refresh** (to improve coverage/reliability) and **quasi-Newton refinement** (to reach high accuracy). The multi-batch quasi-Newton literature provides a principled mechanism (overlap-set curvature pairs) for changing batches without invalidating curvature history, but this mechanism has not been systematically applied to collocation resampling in PINNs.

### Related Papers

- **[A Multi-Batch L-BFGS Method for Machine Learning](./references/A-Multi-Batch-L-BFGS-Method-for-Machine-Learning/meta/meta_info.txt)**: Introduces overlap-set curvature pairs \(y_{k+1}=g^{O_k}_{k+1}-g^{O_k}_k\) for quasi-Newton updates under changing batches.
- **[A robust multi-batch L-BFGS method for machine learning](./references/A-Robust-Multi-Batch-L-BFGS-Method-for-Machine-Learning/meta/meta_info.txt)**: Extends multi-batch L-BFGS with robustness mechanisms (e.g., cautious updates) and scaling results.
- **[Clustering Behaviour of Physics-Informed Neural Networks: Inverse Modeling of an Idealized Ice Shelf](./references/Clustering-Behaviour-of-Physics-Informed-Neural-Networks-Inverse-Modeling-of-An-Idealized-Ice-Shelf/meta/meta_info.txt)**: Documents bimodal clustering in PINN inverse modeling; motivates collocation resampling and evaluates reliability across many trials.
- **[GitHub - YaoGroup pinn_clusters](./references/GitHub---YaoGroup-pinn_clusters/meta/meta_info.txt)**: Provides code + a clear statement that L-BFGS is incompatible with per-iteration collocation resampling.
- **[FP64 is All You Need: Rethinking Failure Modes in Physics-Informed Neural Networks](./references/FP64-is-All-You-Need-Rethinking-Failure-Modes-in-Physics-Informed-Neural-Networks/meta/meta_info.txt)**: Shows that FP32 can prematurely trigger L-BFGS stopping criteria; advocates FP64 for stable PINN training.
- **[Unveiling the optimization process of Physics Informed Neural Networks: How accurate and competitive can PINNs be?](./references/Unveiling-the-optimization-process-of-Physics-Informed-Neural-Networks-How-accurate-and-competitive-can-PINNs-be/meta/meta_info.txt)**: Studies quasi-Newton variants (self-scaled BFGS/Broyden) and links optimization behavior to ill-conditioned Hessians.
- **[Optimizing the optimizer for physics-informed neural networks and Kolmogorov-Arnold networks](./references/Optimizing-the-optimizer-for-physics-informed-neural-networks-and-Kolmogorov-Arnold-networks/meta/meta_info.txt)**: Compares optimizers/line-search variants for canonical PINN PDEs; highlights precision effects and quasi-Newton behavior.
- **[Challenges in Training PINNs: A Loss Landscape Perspective](./references/Challenges-in-Training-PINNs-A-Loss-Landscape-Perspective/meta/meta_info.txt)**: Analyzes PINN training difficulty via loss landscape structure.
- **[Understanding and mitigating gradient pathologies in physics-informed neural networks](./references/Understanding-and-mitigating-gradient-pathologies-in-physics-informed-neural-networks/meta/meta_info.txt)**: Studies gradient-flow pathologies and mitigation strategies in PINNs.
- **[Do physics-informed neural networks (PINNs) need to be deep? Shallow PINNs using the Levenberg-Marquardt algorithm](https://arxiv.org/abs/2602.08515)**: Uses Levenberg–Marquardt on shallow PINNs as an alternative second-order strategy.
- **[Gauss-Newton Natural Gradient Descent for Physics-Informed Computational Fluid Dynamics](https://arxiv.org/abs/2402.10680)**: Applies function-space Gauss-Newton ideas to PINN-style CFD training.
- **[Multi-Preconditioned LBFGS for Training Finite-Basis PINNs](https://arxiv.org/abs/2601.08709)**: Proposes preconditioning/domain-decomposition to scale LBFGS-style training.
- **[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs](https://arxiv.org/abs/1711.10561)**: Foundational PINN paper; popularized Adam→L-BFGS training.
- **[Physics-informed machine learning](https://www.nature.com/articles/s42254-021-00314-5)**: Survey of physics-informed ML methods and applications.
- **[DeepXDE: A Deep Learning Library for Solving Differential Equations](https://arxiv.org/abs/1907.04502)**: Widely used PINN library; includes training recipes and sampling options.
- **[PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs](./references/PINNacle-A-Comprehensive-Benchmark-of-Physics-Informed-Neural-Networks-for-Solving-PDEs/meta/meta_info.txt)**: Benchmark suite with 2D Poisson variants and standardized L2-relative-error evaluation; useful for testing generalization beyond 1D.
- **[XPINNs: Physics-informed neural networks with domain decomposition](https://arxiv.org/abs/2104.10013)**: Domain decomposition for scaling PINNs.
- **[FBPINNs: Finite Basis Physics-Informed Neural Networks](https://arxiv.org/abs/2107.07871)**: Scalable domain decomposition PINNs.
- **[Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems](https://arxiv.org/abs/2203.05671)**: Uses gradient information to improve PINN training and accuracy.
- **[hp-VPINNs: Variational Physics-Informed Neural Networks with domain decomposition](https://arxiv.org/abs/2105.06701)**: Variational formulations and decomposition for stability/accuracy.
- **[Characterizing possible failure modes in physics-informed neural networks](https://arxiv.org/abs/2109.01050)**: Empirical characterization of PINN failure behaviors.
- **[Investigating and Mitigating Failure Modes in Physics-informed Neural Networks (PINNs)](https://arxiv.org/abs/2309.09976)**: Studies failure modes and mitigations (architecture/optimization).
- **[An operator preconditioning perspective on training in physics-informed machine learning](https://arxiv.org/abs/2310.05801)**: Frames PINN training as preconditioned optimization.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Collocation resampling / sampling | Change collocation distribution over training to improve coverage/reliability | Iwasaki & Lai (ice-shelf), RAR/RAD methods | PDE forward/inverse errors; reliability across seeds | Often paired with first-order optimizers; unclear compatibility with quasi-Newton history |
| Quasi-Newton optimizers for PINNs | Use curvature approximations (BFGS/L-BFGS/variants) to accelerate convergence | Urbán et al. 2024; Kiyani et al. 2025; Raissi et al. | Canonical PDE suites (Burgers, Allen–Cahn, KS, etc.) | Can be brittle under ill-conditioning; typically assumes fixed training set |
| Precision/stopping effects | Some PINN “failures” are due to finite precision and optimizer stopping criteria | Xu et al. 2025 | Loss/gradient dynamics and early-stop analysis | Fixing precision may not address sampling-induced issues |
| Stochastic / multi-batch quasi-Newton | Use overlap-set curvature pairs to handle changing batches | Berahas et al. 2016/2017 | Logistic regression, NN training | Not previously specialized to collocation-point resampling in PINNs |

### Closest Prior Work

1. **[GitHub - YaoGroup pinn_clusters](./references/GitHub---YaoGroup-pinn_clusters/meta/meta_info.txt)**: Demonstrates collocation resampling benefits and explicitly states L-BFGS incompatibility; does not provide a method to run quasi-Newton with per-step resampling.
2. **[Clustering Behaviour of PINNs (Ice Shelf)](./references/Clustering-Behaviour-of-Physics-Informed-Neural-Networks-Inverse-Modeling-of-An-Idealized-Ice-Shelf/meta/meta_info.txt)**: Studies reliability/clustering and uses k-means on log-errors; does not adapt L-BFGS to changing collocation sets.
3. **[A Multi-Batch L-BFGS Method for Machine Learning](./references/A-Multi-Batch-L-BFGS-Method-for-Machine-Learning/meta/meta_info.txt)**: Provides the overlap-set curvature-pair construction under changing data batches; not applied to PINNs or to collocation resampling.
4. **[A robust multi-batch L-BFGS method for machine learning](./references/A-Robust-Multi-Batch-L-BFGS-Method-for-Machine-Learning/meta/meta_info.txt)**: Adds robustness mechanisms (useful for nonconvex settings); not applied to PINNs.
5. **[FP64 is All You Need](./references/FP64-is-All-You-Need-Rethinking-Failure-Modes-in-Physics-Informed-Neural-Networks/meta/meta_info.txt)**: Explains one source of premature L-BFGS stopping in PINNs (finite precision), but not the resampling incompatibility.

**Novelty Kill Search Summary:** Searched for combinations of the exact technique+setting, including: “multi-batch L-BFGS PINN collocation resampling”, “stochastic L-BFGS physics-informed neural network resampling”, “overlap set curvature pair PINN”, and “mini-batch BFGS physics-informed neural network”. Also searched for “multi-batch L-BFGS physics-informed neural network” and related variants. No prior work explicitly applying **overlap-set curvature pairs** to enable **per-step collocation resampling** in L-BFGS for PINNs was found as of 2026-02-27.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| pinn_clusters (repo + paper) | Shows resampling can fix reliability/clustering; recommends Adam-resampled then fixed-point L-BFGS | States L-BFGS is incompatible with per-step resampling | Modify curvature-pair computation to use overlap set | Curvature history becomes consistent on overlap points, avoiding premature termination |
| Berahas et al. 2016 | Multi-batch L-BFGS for changing data batches | Not applied to PINN losses/collocation points | Treat collocation set as batch; keep overlap between successive collocation sets | Same-batch curvature assumption is restored on overlap subset |
| Berahas & Takáč 2017 | Robust multi-batch L-BFGS with cautious updates | Not applied to PINNs; focuses on ML datasets | Use cautious updates + overlap gradients for PDE residual resampling | Helps with nonconvexity/noisy curvature under PDE residuals |
| Xu et al. 2025 | Shows FP32 can trigger early L-BFGS stopping in PINNs | Doesn’t address resampling incompatibility | Use FP64 and test resampling+quasi-Newton interaction directly | Removes precision confound; isolates sampling-history mismatch |

---

## Experiments

### Experimental Setup

**One-sentence thesis / what will be decided:** For PINN training with per-step collocation resampling, overlap-set curvature pairs make quasi-Newton L-BFGS usable (no premature termination) and yield better accuracy/reliability than compute-matched Adam-only resampling and the common Adam→fixed-LBFGS workaround.


**Implementation plan**
- Start from the **pinn_clusters** codebase for the ice-shelf inverse problem and re-implement the optimizer loop to support overlap-resampled L-BFGS.
- Use FP64 throughout.

**Compute-budget matching (critical)**
- All methods are compared under a fixed budget of **forward+backward gradient evaluations**, counting:
  - gradient evaluations in Adam steps,
  - all objective/gradient evaluations used by L-BFGS line search,
  - extra overlap-gradient evaluations needed to compute \(y_k\).

**Baseline Ladder (PINN-specific)**
- **Strong first-order baseline**: Adam + resampling for full compute budget.
- **Common hybrid baseline**: Adam + resampling warmstart → fixed-collocation L-BFGS.
- **Closest new method**: Adam + resampling warmstart → overlap-resampled L-BFGS (ours).

(Notes on baseline ladder: prompting and inference-time scaling baselines are not applicable here because the models are trained from scratch; the baseline ladder is over optimizer and sampling choices.)

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| MLP PINN (tanh) | small (≈10^4–10^5 params) | N/A (trained from scratch) | Use pinn_clusters architecture for ice-shelf; standard MLP for 2D Poisson |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| pinn_clusters synthetic ice-shelf data | inverse-problem observations (u,h) + ground truth B(x) | small (1D arrays) | https://github.com/YaoGroup/pinn_clusters | repo license |

**Other Resources (if applicable):**
- None (fully synthetic PDE setup).

**Resource Estimate**:
- **Compute budget**: 
  - Ice-shelf: 3 seeds × 3 main conditions × (≈3e4 gradient evals/run) + (o=0.25 ablation for ours) ≈ 3e5–4e5 gradient evals.
  - 2D Poisson replication: same order.
  - Expected to fit in **< 50 GPU-hours** on 1×A100 due to small networks and low-dimensional (1D/2D) domains; allocate **≤ 200 GPU-hours** as a conservative upper bound including debugging/overhead.
- **GPU memory**: <10GB.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| Ice-shelf inverse PINN (pinn_clusters) | Infer hardness B(x) from noisy u(x), h(x) with PDE constraints | Relative L2 errors (B_err, u_err, h_err); clustering fraction via k-means on log-errors | N/A (synthetic) | https://github.com/YaoGroup/pinn_clusters | pinn_clusters scripts + our optimizer loop |
| 2D Poisson PINN (replication) | Canonical forward PDE used in PINN benchmarks (e.g., analytic-solution Poisson on a square; also covered in PINNacle) | Relative L2 error on u(x,y) (lower is better); convergence/termination diagnostics | N/A (synthetic) | (implement analytic-solution Poisson on \([0,1]^2\) with Dirichlet BC; optionally align with a PINNacle Poisson-2D case) | derived from standard PINN formulations |

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Metric 1 (mean±std) | Metric 2 (mean±std) | Source | Notes |
|--------|------------|-----------|----------------------|----------------------|--------|-------|
| Adam + resampling | pinn_clusters MLP | Ice-shelf | **TBD** (B_err) | **TBD** (high-error frac) | To be re-run | Strong first-order baseline; compute-matched by grad-eval budget |
| Adam + resampling → fixed-collocation L-BFGS | pinn_clusters MLP | Ice-shelf | **TBD** | **TBD** | To be re-run | Current workaround in practice/pinn_clusters |
| **Ours: Adam + resampling → overlap-resampled L-BFGS (o=0.5)** | pinn_clusters MLP | Ice-shelf | **TBD** | **TBD** | To be verified | Curvature pairs computed on overlap set |

Ablation (reported separately): ours with **o=0.25**.

#### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Ours (o=0.5) | main overlap setting | Stable quasi-Newton + improved reliability |
| Ours (o=0.25) | smaller overlap | May become unstable or lose gains (tests sensitivity) |

#### Experimental Rigor

**Variance & Reproducibility:**
- Run all main conditions across **3 seeds** (init + data noise). Report mean ± std.

**Validity & Controls:**
- **Precision confound**: use FP64 for all methods (motivated by Xu et al.).
- **Compute confound**: enforce identical gradient-evaluation budgets.
- **Warmstart confound**: include one diagnostic run of overlap-resampled L-BFGS from scratch (no Adam warmstart) on the 2D Poisson benchmark.
- **Clustering metric degeneracy**: if k-means yields a degenerate split (<10% in a cluster), report median/IQR and fixed-threshold summaries instead.

### Analysis (Optional)

- Measure how often L-BFGS stops early (few inner iterations / line-search failures) under different overlap ratios.

---

## Success Criteria

**Hypothesis** (directional):
Overlap-resampled L-BFGS with \(o=0.5\) will (i) avoid premature termination under per-step collocation resampling and (ii) reduce median error and/or high-error-run fraction compared to compute-matched Adam-only resampling and the Adam→fixed-LBFGS workaround.

**Decision Rule** (concrete):
- **Proceed/Continue** if (i) on the ice-shelf benchmark, overlap-resampled L-BFGS (o=0.5) achieves (a) lower median B_err than both baselines and (b) a lower high-error fraction than both baselines by a margin outside the 3-seed std (or via non-overlapping IQRs), and (ii) on the 2D Poisson benchmark it is **not worse** than the Adam→fixed-LBFGS baseline within noise under the same gradient-eval budget.
- **Pivot** if the method is stable but gains appear only for o≥0.8 (practically close to fixed collocation), or if it helps only on ice-shelf but not on 2D Poisson; in that case, narrow the claim to “inverse problems with clustering-like failure modes”.
- **Refute** if overlap-resampled L-BFGS fails to beat the Adam→fixed-LBFGS baseline on ice-shelf, or if it shows the same premature stopping behavior as naive resampled L-BFGS at o=0.5.

---

## Impact Statement

If successful, this work provides a drop-in way for PINN practitioners to keep the benefits of collocation resampling during quasi-Newton refinement, potentially improving reliability of PINN inverse problems without giving up L-BFGS-style convergence.

---

## References

- [GitHub - YaoGroup pinn_clusters](./references/GitHub---YaoGroup-pinn_clusters/meta/meta_info.txt) - YaoGroup, 2024
- [Clustering Behaviour of Physics-Informed Neural Networks: Inverse Modeling of an Idealized Ice Shelf](./references/Clustering-Behaviour-of-Physics-Informed-Neural-Networks-Inverse-Modeling-of-An-Idealized-Ice-Shelf/meta/meta_info.txt) - Iwasaki & Lai, 2022/2023
- [A Multi-Batch L-BFGS Method for Machine Learning](./references/A-Multi-Batch-L-BFGS-Method-for-Machine-Learning/meta/meta_info.txt) - Berahas, Nocedal, Takáč, 2016
- [A robust multi-batch L-BFGS method for machine learning](./references/A-Robust-Multi-Batch-L-BFGS-Method-for-Machine-Learning/meta/meta_info.txt) - Berahas & Takáč, 2017
- [FP64 is All You Need: Rethinking Failure Modes in Physics-Informed Neural Networks](./references/FP64-is-All-You-Need-Rethinking-Failure-Modes-in-Physics-Informed-Neural-Networks/meta/meta_info.txt) - Xu et al., 2025
- [Unveiling the optimization process of Physics Informed Neural Networks: How accurate and competitive can PINNs be?](./references/Unveiling-the-optimization-process-of-Physics-Informed-Neural-Networks-How-accurate-and-competitive-can-PINNs-be/meta/meta_info.txt) - Urbán et al., 2024
- [Optimizing the optimizer for physics-informed neural networks and Kolmogorov-Arnold networks](./references/Optimizing-the-optimizer-for-physics-informed-neural-networks-and-Kolmogorov-Arnold-networks/meta/meta_info.txt) - Kiyani et al., 2025
- [Challenges in Training PINNs: A Loss Landscape Perspective](./references/Challenges-in-Training-PINNs-A-Loss-Landscape-Perspective/meta/meta_info.txt) - Rathore et al., 2024
- [Understanding and mitigating gradient pathologies in physics-informed neural networks](./references/Understanding-and-mitigating-gradient-pathologies-in-physics-informed-neural-networks/meta/meta_info.txt) - Wang, Teng, Perdikaris, 2021
- [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs](https://arxiv.org/abs/1711.10561) - Raissi, Perdikaris, Karniadakis, 2019
- [Physics-informed machine learning](https://www.nature.com/articles/s42254-021-00314-5) - Karniadakis et al., 2021
- [DeepXDE: A Deep Learning Library for Solving Differential Equations](https://arxiv.org/abs/1907.04502) - Lu et al., 2019/2021
- [XPINNs: Physics-informed neural networks with domain decomposition](https://arxiv.org/abs/2104.10013) - Jagtap et al., 2021
- [FBPINNs: Finite Basis Physics-Informed Neural Networks](https://arxiv.org/abs/2107.07871) - Moseley et al., 2021
- [Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems](https://arxiv.org/abs/2203.05671) - Yu et al., 2022
- [hp-VPINNs: Variational Physics-Informed Neural Networks with domain decomposition](https://arxiv.org/abs/2105.06701) - Kharazmi et al., 2021
- [Do physics-informed neural networks (PINNs) need to be deep? Shallow PINNs using the Levenberg-Marquardt algorithm](https://arxiv.org/abs/2602.08515) - 2026
- [Gauss-Newton Natural Gradient Descent for Physics-Informed Computational Fluid Dynamics](https://arxiv.org/abs/2402.10680) - 2024
- [Multi-Preconditioned LBFGS for Training Finite-Basis PINNs](https://arxiv.org/abs/2601.08709) - 2026
- [An operator preconditioning perspective on training in physics-informed machine learning](https://arxiv.org/abs/2310.05801) - 2023
- [Characterizing possible failure modes in physics-informed neural networks](https://arxiv.org/abs/2109.01050) - 2021
- [Investigating and Mitigating Failure Modes in Physics-informed Neural Networks (PINNs)](https://arxiv.org/abs/2309.09976) - 2023
- [PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs](./references/PINNacle-A-Comprehensive-Benchmark-of-Physics-Informed-Neural-Networks-for-Solving-PDEs/meta/meta_info.txt) - Hao et al., 2023 (NeurIPS 2024 datasets/benchmarks)
