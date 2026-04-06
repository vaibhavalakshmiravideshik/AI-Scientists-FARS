# untitled

# Range-Capped Sinkhorn for Reliable Manifold-Constrained Hyper-Connections

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Residual connections are a standard mechanism for stabilizing optimization in deep networks because they provide an identity path for both activations and gradients. **Hyper-Connections (HC)** generalize residual connections by maintaining multiple residual “streams” and learning an \(n\times n\) mixing matrix \(H^{res}_\ell\) at each layer to route information across streams, which can improve optimization and representation learning ([Hyper-Connections](./references/Hyper-Connections/meta/meta_info.txt)).

A known weakness of HC is that unconstrained learned mixing can become numerically unstable in deep networks: the depth-wise product \(\prod_\ell H^{res}_\ell\) can amplify or attenuate activations and gradients. **Manifold-Constrained Hyper-Connections (mHC)** addresses this by constraining \(H^{res}_\ell\) to be **doubly stochastic** (nonnegative, each row and column sums to 1), i.e., within the Birkhoff polytope (the convex set of all \(n\times n\) doubly-stochastic matrices) ([mHC](./references/mHC-Manifold-Constrained-Hyper-Connections/meta/meta_info.txt)). mHC constructs \(H^{res}_\ell\) by applying a **Sinkhorn–Knopp (SK)** normalization to exponentiated logits, but uses a finite iteration budget (\(t_{\max}=20\) in the paper) for efficiency (mHC §4.2).

Recent follow-ups aim to eliminate approximation error by avoiding finite-iteration Sinkhorn entirely: **mHC-lite** uses an exact Birkhoff–von Neumann permutation-mixture parameterization (exact for \(n=4\)) ([mHC-lite](./references/mHC-lite-You-Don't-Need-20-Sinkhorn-Knopp-Iterations/meta/meta_info.txt)), and **KromHC** uses Kronecker products of smaller doubly-stochastic factors to get exact doubly-stochastic matrices with fewer parameters ([KromHC](./references/KromHC-Manifold-Constrained-Hyper-Connections-with-Kronecker-Product-Residual-Matrices/meta/meta_info.txt)).

However, many architectures still rely on Sinkhorn as a differentiable “project-to-doubly-stochastic” layer (including mHC itself, and many differentiable matching / assignment layers), so it is important to make the **finite-iteration Sinkhorn layer reliable** when it is used.

### The Problem

Finite-iteration Sinkhorn is sensitive to **conditioning**. mHC-lite formalizes this: for a nonnegative matrix \(X\), the number of Sinkhorn iterations required to achieve \(\ell_1\) doubly-stochastic error \(\le \epsilon\) can be as large as \(O(n^2\log(n/\nu)/\epsilon^2)\), where \(\nu = \min^+(x_{ij})/\max(x_{ij})\) is the **relative range** (mHC-lite §3.1; cites Chakrabarty & Khanna 2021). Empirically, mHC-lite reports that **~27.9% of mHC Sinkhorn inputs have \(1/\nu \ge 10^{13}\)** and that large \(\log(1/\nu)\) implies a fixed 20-iteration budget may not converge well.

Separately, a public reimplementation of mHC (tokenbender) surfaces a practical failure mode: with `sinkhorn_tau=0.05` and a sharp initialization (`H_res_logits` diagonal 0, off-diagonal -8), gradients into `H_res_logits` can be nearly zero (GitHub issue #2 reports `grad_norm≈6.1e-9` for `H_res_logits` versus `~1e-3`–`1e0` for other parameters), and increasing `sinkhorn_tau` (e.g., to 1.0) restores learning (issue #2; code uses `Z = logits / tau`). This suggests that **finite-iteration Sinkhorn can silently freeze the very parameters it is meant to train** when the input logits are too sharp.

A straightforward workaround is “just set a larger constant \(\tau\)”. The problem is that this is an undocumented tuning knob in many implementations, and it is unclear when a single constant \(\tau\) is sufficient across training (as logits evolve) and across layers.

### Key Insight and Hypothesis

**Key insight:** In many implementations, Sinkhorn operates on \(\exp(Z)\) where \(Z\) is a scaled logit matrix (e.g., tokenbender uses \(Z = \texttt{logits}/\tau\)). The relative range satisfies
\[
\log(1/\nu) = \max_{ij} Z_{ij} - \min_{ij} Z_{ij}.
\]
Therefore, controlling the logit range is equivalent to controlling conditioning \(\nu\) and also prevents numerical underflow (e.g., \(\exp(-160)\) from `-8/0.05`).

**Hypothesis:** If we enforce a fixed upper bound on \(\log(1/\nu)\) for the Sinkhorn input (e.g., \(\log(1/\nu)\le 30\), matching the warning threshold in mHC-lite §3.1), then (i) gradients into \(H^{res}\) logits will no longer vanish under sharp initializations, and (ii) training will be at least as stable (gradient spikes) and not worse in validation loss compared to the default fixed-\(\tau\) implementation.

Why this could be wrong:
- A single better constant \(\tau\) may fully explain the effect; range capping may add no value.
- The model’s downstream loss may be insensitive to whether \(H^{res}\) actually learns in short runs, making the effect appear “irrelevant” even if gradients improve.

---

## Proposed Approach

### Overview

We propose **Range-Capped Sinkhorn (RRCS)**: a minimal modification to the Sinkhorn projection used to construct \(H^{res}\) that rescales the Sinkhorn input logits to keep the log-range below a fixed cap \(r_{cap}\). RRCS is intended to be a drop-in replacement for the standard “logits divided by \(\tau\) then Sinkhorn” procedure.

### Method Details

Let `tau_base` be the implementation’s existing temperature (tokenbender default: `sinkhorn_tau=0.05`) and let `sinkhorn_log(Z, iters)` be a log-domain Sinkhorn implementation.

1. **Base scaled logits**:
   \[
   Z \leftarrow \frac{\texttt{logits}}{\tau_{base}}.
   \]

2. **Robust log-range estimate**:
   \[
   r \leftarrow q_{0.99}(Z) - q_{0.01}(Z)
   \]
   where \(q_p\) is the elementwise quantile. (For small \(n\) like \(n=4\), \(\max(Z)-\min(Z)\) is also acceptable.)

3. **Range cap via rescaling**:
   \[
   s \leftarrow \min\left(1, \frac{r_{cap}}{r+\epsilon}\right),\quad Z' \leftarrow s\,Z.
   \]
   This ensures \(\max(Z')-\min(Z') \le r_{cap}\), so \(1/\nu \le e^{r_{cap}}\).

4. **Sinkhorn projection**:
   \[
   H^{res} \leftarrow \mathrm{Sinkhorn}(Z';\,\texttt{iters}).
   \]

**Choice of \(r_{cap}\):** default \(r_{cap}=30\), motivated by mHC-lite’s observation that \(\log(1/\nu)>30\) implies 20 SK iterations may not converge well (mHC-lite §3.1), and by numerical stability (\(\exp(-30)\approx 9\times 10^{-14}\) is far from underflow in float32/bfloat16).

**Fixed-\(\tau\) control derived from the same cap (no adaptivity):** define
\[
\tau_{cap-init} = \tau_{base}\cdot \max\left(1, \frac{r_{init}}{r_{cap}}\right),\quad r_{init} = \max(Z_{init})-\min(Z_{init}),
\]
and run standard Sinkhorn with constant \(\tau_{cap-init}\). This matches RRCS at initialization but does not adapt if the logits become sharper later.

### Key Innovations

- **Conditioning diagnostic and control for finite-iteration Sinkhorn in mHC**: RRCS converts mHC-lite’s convergence driver (\(\log(1/\nu)\)) into a concrete training-time control knob.
- **Mechanism-targeted fix for vanishing gradients through Sinkhorn**: RRCS prevents extreme exponentiation (e.g., `exp(-160)`) that can freeze \(H^{res}\) learning in practice.
- **Minimal intervention**: no new parameters, no new losses, no change to iteration count; only a deterministic rescaling of inputs to an existing Sinkhorn layer.

---

## Related Work

### Field Overview

This proposal touches three areas.

1) **Residual-connection generalizations**. Hyper-Connections introduce learnable multi-stream residual routing, but can become unstable without constraints. mHC constrains residual routing to doubly-stochastic matrices, which theoretically bounds amplification, but uses finite-iteration Sinkhorn (approximate projection) for efficiency.

2) **Exact doubly-stochastic parameterizations**. mHC-lite and KromHC avoid Sinkhorn’s approximation error by reparameterizing \(H^{res}\) to be doubly-stochastic by construction. These methods reduce reliance on Sinkhorn, but do not address the broader pattern of Sinkhorn being widely used as a differentiable DS projection.

3) **Sinkhorn in optimal transport and differentiable assignment**. Sinkhorn is foundational in entropy-regularized optimal transport and appears as a building block in differentiable sorting, matching, and clustering. This literature develops numerical stabilization techniques (log-domain, \(\epsilon\)-scaling, absorption), but does not typically link conditioning control to *gradient flow into upstream logits* in architectural DS constraints.

### Related Papers

- **[Hyper-Connections](./references/Hyper-Connections/meta/meta_info.txt)**: Introduces multi-stream residual routing with learned mixing; motivates why constraining \(H^{res}\) matters.
- **[mHC](./references/mHC-Manifold-Constrained-Hyper-Connections/meta/meta_info.txt)**: Constrains residual routing to the Birkhoff polytope via finite-iteration Sinkhorn; the direct target of our intervention.
- **[mHC-lite](./references/mHC-lite-You-Don't-Need-20-Sinkhorn-Knopp-Iterations/meta/meta_info.txt)**: Analyzes finite-iteration Sinkhorn instability and proposes exact DS via permutation mixtures; provides the \(\nu\)-dependence we operationalize.
- **[KromHC](./references/KromHC-Manifold-Constrained-Hyper-Connections-with-Kronecker-Product-Residual-Matrices/meta/meta_info.txt)**: Uses Kronecker products of DS factors for exact DS with better scaling; alternative to Sinkhorn-based projection.
- **[tokenbender mHC implementation](./references/GitHub-tokenbender-mHC-manifold-constrained-hyper-connections/meta/meta_info.txt)**: Public codebase exposing `sinkhorn_tau` and an observed gradient-vanishing failure mode.
- **[Sinkhorn & Knopp (1967)](https://doi.org/10.2140/pjm.1967.21.343)**: Original Sinkhorn–Knopp matrix scaling algorithm for producing doubly-stochastic matrices.
- **[Knight (2008)](https://doi.org/10.1137/060659624)**: Convergence analysis and applications of Sinkhorn–Knopp.
- **[Chakrabarty & Khanna (2021)](https://doi.org/10.1007/s10107-020-01503-3)**: Error analysis giving explicit iteration bounds in terms of \(\nu\).
- **[Linial et al. (1998)](https://doi.org/10.1145/276698.276880)**: Classical complexity results for matrix scaling that highlight worst-case slow convergence.
- **[Cuturi (2013)](https://arxiv.org/abs/1306.0895)**: Introduces Sinkhorn distances for fast entropy-regularized optimal transport.
- **[Peyré & Cuturi (2019)](https://arxiv.org/abs/1803.00567)**: Computational optimal transport reference including Sinkhorn and stabilization techniques.
- **[Schmitzer (2016)](https://arxiv.org/abs/1610.06519)**: Stabilized (log-domain / absorption-style) Sinkhorn variants for numerical robustness.
- **[Genevay et al. (2016)](https://arxiv.org/abs/1605.08527)**: Stochastic optimization for large-scale optimal transport with Sinkhorn as a core primitive.
- **[Altschuler et al. (2017)](https://arxiv.org/abs/1705.09634)**: Near-linear time approximation guarantees for OT via Sinkhorn iteration.
- **[Mena et al. (2018)](https://arxiv.org/abs/1802.08665)**: Gumbel-Sinkhorn networks; uses temperature schedules for differentiable permutations (different motivation than our conditioning cap).
- **[Caron et al. (2020)](https://arxiv.org/abs/2006.09882)**: SwAV uses distributed Sinkhorn-Knopp for clustering assignments and discusses practical stabilization.
- **[Blondel et al. (2020)](https://arxiv.org/abs/2002.08871)**: Differentiable sorting/ranking using continuous relaxations related to OT/Sinkhorn.
- **[Grover et al. (2019)](https://arxiv.org/abs/1903.09367)**: NeuralSort introduces differentiable sorting (assignment-like) operators related to doubly-stochastic relaxations.
- **[He et al. (2016)](https://arxiv.org/abs/1512.03385)**: ResNet identity-mapping motivation for why residual-path stability matters.
- **[Xiong et al. (2020)](https://arxiv.org/abs/2002.04745)**: Analyzes normalization choices in Transformers that affect residual-path stability.
- **[Bachlechner et al. (2021)](https://arxiv.org/abs/2003.04887)**: ReZero shows that residual-path scaling can strongly affect trainability in deep networks.
- **[Zhang et al. (2019)](https://arxiv.org/abs/1901.09321)**: Fixup initialization highlights how residual-path scaling interacts with optimization even without normalization.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| HC-style residual routing | Learnable multi-stream residual mixing | Hyper-Connections; mHC | LM pretraining loss; stability diagnostics | Unconstrained routing can destabilize training |
| Exact DS parameterizations | Build DS matrices by construction | mHC-lite; KromHC | LM pretraining loss; DS-error diagnostics | Can impose structural constraints (factorization) or scale poorly with \(n\) |
| Sinkhorn numerical stabilization | Improve Sinkhorn stability/efficiency in OT | Cuturi 2013; Schmitzer 2016; Altschuler 2017 | OT cost / transport plan quality | Often targets OT cost, not gradient flow into upstream logits |
| Differentiable assignment / sorting | Relax permutations via DS-like operators | Mena 2018; Grover 2019; Blondel 2020 | Sorting/matching quality on downstream tasks | Temperature/relaxation schedules chosen for approximation, not for fixed-iter DS constraints |

### Closest Prior Work

- **mHC** ([mHC](./references/mHC-Manifold-Constrained-Hyper-Connections/meta/meta_info.txt)) uses Sinkhorn–Knopp with a fixed iteration budget (\(t_{\max}=20\)) to approximately project \(H^{res}\) onto the Birkhoff polytope. It does not provide a conditioning-based rule for choosing \(\tau\) or for ensuring gradients remain informative.
- **mHC-lite** ([mHC-lite](./references/mHC-lite-You-Don't-Need-20-Sinkhorn-Knopp-Iterations/meta/meta_info.txt)) identifies the \(\nu\)-dependence of Sinkhorn convergence and reports that many inputs are extremely ill-conditioned, motivating exact DS reparameterization. It does not propose conditioning-based \(\tau\) control as a “make Sinkhorn reliable” intervention.
- **KromHC** ([KromHC](./references/KromHC-Manifold-Constrained-Hyper-Connections-with-Kronecker-Product-Residual-Matrices/meta/meta_info.txt)) avoids Sinkhorn entirely via a Kronecker structured exact DS parameterization; it is an alternative direction rather than a fix to finite-iteration Sinkhorn.
- **OT stabilization literature** (e.g., Schmitzer 2016; Peyré & Cuturi 2019) focuses on numerical stability of OT solves, typically by log-domain computations and \(\epsilon\)-scaling. RRCS is complementary: it uses an explicit cap on \(\log(1/\nu)\) to prevent the upstream logits from producing a numerically degenerate DS projection under a fixed iteration budget.

**Novelty Kill Search Summary:** Searched for combinations including “Sinkhorn logit clipping”, “Sinkhorn temperature selection based on logit range”, “Birkhoff projection Sinkhorn clipping”, and “manifold-constrained Sinkhorn temperature”. Also checked local finalized proposals for `sinkhorn_tau` / “range-capped sinkhorn”. No prior work (as of 2026-02-23) was found that uses an explicit \(\nu\)/log-range certificate to set an effective temperature for Sinkhorn inside mHC-style doubly-stochastic residual mixing.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| mHC | Fixed-iter Sinkhorn projection to DS residual mixing | Finite iters + sharp logits can freeze gradients; no conditioning rule | Cap Sinkhorn input log-range before projection | Prevents underflow and keeps SK in a “plausible” regime without increasing iters |
| mHC-lite | Exact DS via permutation mixture; analyzes slow SK | Solves by avoiding SK; does not improve SK layer itself | Keep SK but control conditioning (\(\nu\)) | Minimal change for codebases that already use SK projections |
| KromHC | Exact DS via Kronecker-structured DS factors | Requires factorization structure; different parameterization | Leave parameterization unchanged; change only projection conditioning | Lower engineering risk; keeps existing model form |
| OT stabilization (Schmitzer; Peyré & Cuturi) | Log-domain / \(\epsilon\)-scaling / absorption to stabilize OT | Targets OT objective; not necessarily tuned for gradient flow into logits | Use a fixed log-range cap tied to DS convergence driver \(\nu\) | Directly targets the failure mode observed in mHC implementations |
| SwAV | Distributed Sinkhorn for clustering assignments | Specific to SSL clustering; schedule choices not tied to mHC’s DS constraint | Apply log-range cap to DS projection in mHC | Transfers a stabilization principle (avoid ill-conditioned scaling) to architectural DS constraints |

---

## Experiments

### Experimental Setup

**Codebase:** tokenbender’s mHC nanoGPT implementation ([repo](./references/GitHub-tokenbender-mHC-manifold-constrained-hyper-connections/meta/meta_info.txt)).

**Primary setting (single setting to stay decisive):** FineWeb10B, 48-layer config `train_fineweb10B_mhc_48l.py` (from the repo README).

**Main conditions (exactly 3):**

1. **mHC baseline (default \(\tau\))**: `sinkhorn_tau=0.05`, `sinkhorn_iters=10` (as in config).
2. **Fixed-\(\tau\) cap-init control**: replace `sinkhorn_tau` with \(\tau_{cap-init}\) computed from the initialization log-range and \(r_{cap}=30\); keep everything else identical.
3. **RRCS (ours)**: keep `sinkhorn_tau=tau_base` but apply the per-step log-range cap (rescale `Z`) before Sinkhorn.

**Important implementation detail:** tokenbender’s Sinkhorn implementation uses `Z = logits / tau` (see `hyper_connections_mhc.py`), so RRCS can be implemented either as rescaling `Z` directly (preferred) or equivalently as using an effective \(\tau_{eff}=\tau_{base}/s\).

**Baseline Ladder (REQUIRED):** This proposal is about a training-time architectural primitive (Sinkhorn projection), so prompting/inference baselines are not applicable. The baseline ladder here is:
- Default mHC implementation (what practitioners would run)
- Best simple constant-\(\tau\) fix derived from the same range cap (cap-init)
- RRCS (conditioning-aware)

**Training Data:**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| FineWeb (10B token subset) | LM pretraining | 10B tokens | https://huggingface.co/datasets/HuggingFaceFW/fineweb | ODC-By (per HF dataset card) |

**Resource Estimate (conservative):**
- Run budget: 48-layer nanoGPT config, `max_iters=5000`, 3 conditions × 3 seeds = 9 runs.
- Assume up to **8 hours per run on 4×A100** (very conservative for a ~20M parameter model) ⇒ 32 GPU-hours/run.
- Total ≤ 9 × 32 = **288 GPU-hours**.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| FineWeb10B LM pretraining | Next-token prediction on web text | Validation loss (cross-entropy), perplexity | val | https://huggingface.co/datasets/HuggingFaceFW/fineweb | tokenbender nanoGPT eval loop |

**Primary diagnostic metrics (automated):**
- **H_res gradient flow:** median and mean \(\|\nabla \texttt{H_res_logits}\|_2\) over training steps (per-layer and aggregated).
- **H_res learning:** \(\|\texttt{H_res_logits}(T)-\texttt{H_res_logits}(0)\|_F\) after training.
- **Sinkhorn input conditioning:** distribution of \(r=\max(Z)-\min(Z)\) (or quantile-range) per layer and over training.
- **Doubly-stochastic (DS) error:** \(\max_i |\sum_j H^{res}_{ij}-1|\) and \(\max_j |\sum_i H^{res}_{ij}-1|\).
- **Sharpness:** mean row entropy of \(H^{res}\) (lower means closer to permutation-like routing).

**Training stability metrics (automated):**
- Global gradient norm time series; compute max spike ratio
  \[
  r_{\max} = \max_t \frac{g_t}{\mathrm{median}(g_{t-100:t-1})}
  \]
  for \(t>200\).

### Main Results

#### Results Table

(All results are to be filled by the Verification module; no published numbers were found for this exact codebase + config.)

| Method | Base Model | Benchmark | Val loss (mean±std) | H_res grad median | \(\|\Delta H_{res}\|\) | DS error | Entropy(H_res) | Source | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| mHC default \(\tau=0.05\) | nanoGPT 48L (~20M) | FineWeb10B val | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | – | 3 seeds |
| Fixed \(\tau_{cap-init}\) | nanoGPT 48L (~20M) | FineWeb10B val | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | – | 3 seeds |
| **RRCS (ours)** | nanoGPT 48L (~20M) | FineWeb10B val | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | – | 3 seeds |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| RRCS with `r_cap=20` | Tighter log-range cap | Potentially more stable gradients but may oversoften routing |
| RRCS with `r_cap=40` | Looser log-range cap | Closer to baseline; may reintroduce saturation if logits sharpen |

### Experimental Rigor

- **Seeds**: `seeds=[42, 123, 456]` for all 3 conditions.
- **Fair comparison conditions**: identical optimizer, LR schedule, batch size, data order, total iterations, and hardware across conditions.
- **Top confounders + controls**:
  1) **“It’s just a better constant \(\tau\)”** → include \(\tau_{cap-init}\) constant baseline that matches RRCS at initialization.
  2) **Short-run insensitivity** → use internal diagnostics (gradient flow, parameter drift) as primary outcomes even if loss differences are small.
  3) **Metric instrumentation overhead affects throughput** → log diagnostics at a fixed interval (e.g., every 10 steps) across all conditions.

---

## Success Criteria

**Hypothesis** (directional): RRCS will prevent near-zero gradients into `H_res_logits` and enable `H_res_logits` to change meaningfully over training, without degrading validation loss or training stability.

**Decision Rule** (concrete):

Let condition 1 be the default mHC baseline, condition 2 be fixed \(\tau_{cap-init}\), and condition 3 be RRCS.

- **Proceed** if:
  1) RRCS increases the median `H_res_logits` gradient norm by **≥100×** vs condition 1 and the final \(\|\Delta H_{res}\|_F\) is **≥10×** condition 1 (indicating “H_res actually learns”), and
  2) RRCS validation loss is **not worse** than the better of conditions (1,2) by more than **0.5×** that baseline’s across-seed std.

- **Pivot** (simplify to a constant-\(\tau\) rule) if:
  - Condition 2 matches RRCS on both (i) `H_res_logits` gradient/\(\Delta H\) and (ii) validation loss/stability metrics, implying per-step adaptation is unnecessary and a one-time range-based \(\tau\) setting is sufficient.

- **Refute** if:
  - RRCS does not materially increase `H_res_logits` gradients/\(\Delta H\) relative to condition 1, or
  - RRCS worsens validation loss beyond the threshold above, or introduces larger gradient spikes (higher \(r_{\max}\)) than both baselines.

---

## Impact Statement

If successful, RRCS provides a small, implementation-friendly change that makes Sinkhorn-projected doubly-stochastic residual mixing reliably trainable without manual temperature tuning. This improves reproducibility for researchers and practitioners experimenting with mHC-like architectures, and more broadly provides a conditioning-based recipe for using finite-iteration Sinkhorn layers in differentiable assignment / DS-constraint settings.

---

## References

- [Hyper-Connections](./references/Hyper-Connections/meta/meta_info.txt) - Zhu et al., 2024
- [mHC: Manifold-Constrained Hyper-Connections](./references/mHC-Manifold-Constrained-Hyper-Connections/meta/meta_info.txt) - Xie et al., 2025
- [mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations](./references/mHC-lite-You-Don't-Need-20-Sinkhorn-Knopp-Iterations/meta/meta_info.txt) - Yang & Gao, 2026
- [KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices](./references/KromHC-Manifold-Constrained-Hyper-Connections-with-Kronecker-Product-Residual-Matrices/meta/meta_info.txt) - Zhou et al., 2026
- [tokenbender/mHC-manifold-constrained-hyper-connections](./references/GitHub-tokenbender-mHC-manifold-constrained-hyper-connections/meta/meta_info.txt) - GitHub repo, accessed 2026-02-23
- [Sinkhorn & Knopp (1967)](https://doi.org/10.2140/pjm.1967.21.343) - Sinkhorn & Knopp, 1967
- [The Sinkhorn–Knopp algorithm: convergence and applications](https://doi.org/10.1137/060659624) - Knight, 2008
- [Better and simpler error analysis of the Sinkhorn–Knopp algorithm for matrix scaling](https://doi.org/10.1007/s10107-020-01503-3) - Chakrabarty & Khanna, 2021
- [A deterministic strongly polynomial algorithm for matrix scaling and approximate permanents](https://doi.org/10.1145/276698.276880) - Linial et al., 1998
- [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895) - Cuturi, 2013
- [Computational Optimal Transport](https://arxiv.org/abs/1803.00567) - Peyré & Cuturi, 2019
- [Stabilized sparse scaling algorithms for entropy regularized transport problems](https://arxiv.org/abs/1610.06519) - Schmitzer, 2016
- [Stochastic Optimization for Large-scale Optimal Transport](https://arxiv.org/abs/1605.08527) - Genevay et al., 2016
- [Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration](https://arxiv.org/abs/1705.09634) - Altschuler et al., 2017
- [Learning Latent Permutations with Gumbel-Sinkhorn Networks](https://arxiv.org/abs/1802.08665) - Mena et al., 2018
- [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV)](https://arxiv.org/abs/2006.09882) - Caron et al., 2020
- [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871) - Blondel et al., 2020
- [NeuralSort: Learning to Sort with Optimal Transport](https://arxiv.org/abs/1903.09367) - Grover et al., 2019
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - He et al., 2016
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Xiong et al., 2020
- [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887) - Bachlechner et al., 2021
- [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321) - Zhang et al., 2019
