# untitled

# Orthostochastic Residual Mixing for Manifold-Constrained Hyper-Connections

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Core constraint**: Fully automated evaluation (no human evaluation in the loop)
- **Verification budget**: ≤ 768 A100-hours total

## Introduction

### Context and Motivation

Residual connections are a central ingredient for training deep neural networks. In Transformer language models, the residual pathway provides an identity route for optimization, but the exact *wiring* of residual information across depth is usually fixed.

**Hyper-Connections** extend residual connections by maintaining multiple residual streams and learning a per-layer residual mixing matrix that routes information across these streams, aiming to improve optimization and representation learning ([Hyper-Connections](./references/Hyper-Connections/meta/meta_info.txt)). However, unconstrained learnable residual mixing can become numerically unstable in deep networks because the depth-wise product of mixing matrices can amplify activations and gradients.

**Manifold-Constrained Hyper-Connections (mHC)** addresses this by constraining the residual mixing matrix \(H^{res}_\ell\) to be **doubly stochastic** (nonnegative with row and column sums equal to 1), i.e., on the **Birkhoff polytope** (the set of \(n\times n\) doubly-stochastic matrices). This yields stability-by-construction because products of doubly-stochastic matrices remain doubly-stochastic and avoid runaway amplification ([mHC](./references/mHC-Manifold-Constrained-Hyper-Connections/meta/meta_info.txt)). Follow-up work focuses on making the doubly-stochastic constraint **exact and efficient**: **mHC-lite** uses a Birkhoff–von Neumann permutation-mixture parameterization (exact for \(n=4\)), and **KromHC** uses Kronecker products of smaller doubly-stochastic factors.

However, these exact parameterizations do not fully resolve the question of **how to scale Hyper-Connections to larger stream counts \(n>4\)** (a primary motivation for the architecture). Permutation-mixtures scale as \(n!\), while Kronecker factorizations impose a separability structure and require choosing a factorization of \(n\). This motivates studying other structured subsets of the Birkhoff polytope that (i) scale to arbitrary \(n\) and (ii) can be implemented with common GPU primitives.

In this proposal, we test whether Hyper-Connections actually need the full degrees of freedom of general doubly-stochastic matrices, or whether a smaller structured subset—**orthostochastic matrices**—is sufficient.

This is decision-relevant for practitioners and researchers who want to scale Hyper-Connections beyond the standard \(n=4\) setting:
- If orthostochastic constraints suffice, they provide a **drop-in, \(n\)-scalable** family of doubly-stochastic matrices (defined via orthogonalization + elementwise squaring) that avoids factorial permutation-mixtures and does not require choosing a Kronecker factorization.
- If they do not, it provides evidence that **interior Birkhoff-polytope expressiveness matters**, which informs which exact-DS families are worth pursuing when \(n\) grows.

### The Problem

All published mHC-family methods implicitly target the full set of doubly-stochastic matrices, either by iterative projection (Sinkhorn–Knopp) or by exact parameterizations that can represent any doubly-stochastic matrix (in principle).

However, the Birkhoff polytope has \((n-1)^2\) degrees of freedom for \(n\) residual streams, while the set of **orthostochastic matrices** (entrywise squares of an orthogonal matrix) is a strict subset with fewer degrees of freedom but still includes all permutation matrices (the vertices of the Birkhoff polytope). This raises a practical question:

- Do mHC’s gains come from “being somewhere in the Birkhoff polytope”, or do they mostly come from *near-permutation / structured routing* behavior that an orthostochastic subset can already express?

This question matters because if the orthostochastic subset is sufficient, it suggests that future mHC-like systems could use more structured parameterizations (and potentially reuse orthogonalization primitives already common in modern training stacks) without paying for the full expressiveness of the Birkhoff polytope.

### Key Insight and Hypothesis

A public PyTorch implementation of mHC (tokenbender repo) includes an undocumented-in-the-papers switch `mhc_h_res_proj="orthostochastic"`, which constructs \(H^{res}\) by:
1) applying a fixed-step Newton–Schulz orthogonalization to logits \(L\) to obtain an approximately orthogonal matrix \(O\), then
2) returning \(H^{res} = O \odot O\) (elementwise square), which is nonnegative and (approximately) doubly stochastic when \(O\) is (approximately) orthogonal ([tokenbender repo](./references/GitHub-tokenbender-mHC-manifold-constrained-hyper-connections/meta/meta_info.txt)).

**Hypothesis**: Constraining \(H^{res}\) to the orthostochastic subset (via Newton–Schulz + squaring) will match Sinkhorn-projected mHC in training stability and validation loss, implying that the extra degrees of freedom of general doubly-stochastic matrices are not needed.

**Mechanism hypothesis**: If the function of \(H^{res}\) is primarily to implement stable, permutation-like routing and gradual mixing (as suggested by the Birkhoff-polytope interpretation of \(H^{res}\) as a convex combination of permutations in mHC), then restricting \(H^{res}\) to orthostochastic matrices—which still contain all permutation matrices and many smooth interpolations—should preserve the useful inductive bias. If instead mHC relies on dense interior points of the Birkhoff polytope for expressivity, orthostochastic will underperform.

Why we could be wrong:
- The orthostochastic subset may be too restrictive, preventing the model from learning useful dense interior points of the Birkhoff polytope.
- Finite-step Newton–Schulz may yield \(O\) that is not sufficiently orthogonal in mixed precision, so \(O\odot O\) may deviate from doubly-stochastic constraints and lose stability guarantees.

---

## Proposed Approach

### Overview

We propose **Orthostochastic mHC** as an empirical mechanism test: keep the Hyper-Connections and mHC training recipe fixed, but replace the construction of \(H^{res}\) from **Sinkhorn-projected doubly-stochastic matrices** to **orthostochastic (squared-orthogonal) doubly-stochastic matrices**.

The intervention is a one-line configuration change in the tokenbender implementation:
- Baseline: `mhc_h_res_proj = "sinkhorn"`
- Proposed: `mhc_h_res_proj = "orthostochastic"`

The outcome directly answers whether mHC’s benefits require general Birkhoff-polytope expressiveness, or whether a strict structured subset is sufficient.

### Method Details

**Baseline (mHC-Sinkhorn)**: follow the tokenbender implementation of mHC where \(H^{res}\) is computed by applying a log-domain Sinkhorn–Knopp procedure to per-layer logits \(L\in\mathbb{R}^{n\times n}\) to obtain a doubly-stochastic matrix \(S\):
\[
H^{res} \leftarrow \mathrm{Sinkhorn}(L;\, \text{iters}=10,\, \tau=0.05).
\]

**Proposed (mHC-Orthostochastic)**: construct \(H^{res}\) by orthogonalizing the logits via a fixed-step Newton–Schulz iteration to obtain \(O\), then squaring elementwise:
\[
O \leftarrow \mathrm{NS\_orth}(L;\, \text{steps}=10,\, \epsilon=10^{-7},\, (a,b,c)=(3.0,-3.2,1.2)),\qquad
H^{res} \leftarrow O \odot O.
\]

**Diagnostics (logged during training)**:
- Doubly-stochastic error: \(\max_i |\sum_j H^{res}_{ij}-1|\) and \(\max_j |\sum_i H^{res}_{ij}-1|\)
- Orthogonality residual before squaring: \(\lVert O O^\top - I\rVert_F\)

### Key Innovations

- **Constraint-set test, not a new parameterization**: we isolate whether restricting \(H^{res}\) to the orthostochastic subset changes stability/performance relative to Sinkhorn-projected mHC.
- **Actionable negative result**: if orthostochastic underperforms while staying stable and approximately doubly-stochastic, it implies that *general* Birkhoff-polytope expressiveness is important (supporting mHC-lite/KromHC directions). If it matches, it suggests that future work can safely restrict \(H^{res}\) to a smaller structured family.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) residual-path architecture design for stable optimization, (ii) constrained / structured mixing matrices (especially doubly-stochastic constraints), and (iii) differentiable matrix projection and orthogonalization methods.

Hyper-Connections and its successors show that learnable residual routing can improve language model training, but they also reveal a sharp stability failure mode caused by composing unconstrained mixing matrices across depth. The mHC line addresses this by constraining residual mixing to the Birkhoff polytope, either approximately (Sinkhorn) or exactly (permutation-mixtures, Kronecker products). In parallel, Newton–Schulz and matrix-sign methods have become practical orthogonalization primitives in optimizers and related matrix-function computations, motivating the question of whether squared-orthogonal constructions can serve as a useful constrained family for residual routing.

### Related Papers

- **[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)**: introduces residual connections as a key mechanism for training deep networks.
- **[Hyper-Connections](./references/Hyper-Connections/meta/meta_info.txt)**: introduces multi-stream residual mixing with learnable residual routing matrices.
- **[mHC: Manifold-Constrained Hyper-Connections](./references/mHC-Manifold-Constrained-Hyper-Connections/meta/meta_info.txt)**: constrains residual mixing matrices to the Birkhoff polytope via Sinkhorn projection; introduces stability diagnostics for residual mixing.
- **[mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations](./references/mHC-lite-You-Dont-Need-20-Sinkhorn-Knopp-Iterations/meta/meta_info.txt)**: replaces iterative Sinkhorn with an exact permutation-mixture parameterization for \(n=4\).
- **[KromHC](./references/KromHC-Manifold-Constrained-Hyper-Connections-with-Kronecker-Product-Residual-Matrices/meta/meta_info.txt)**: uses Kronecker-product structure to obtain exact DS matrices with fewer parameters.
- **[The Sinkhorn–Knopp Algorithm](https://doi.org/10.2307/1992825)**: classical matrix scaling algorithm for doubly-stochastic projection.
- **[Better and Simpler Error Analysis of the Sinkhorn–Knopp Algorithm for Matrix Scaling](https://arxiv.org/abs/1904.03704)**: modern analysis of Sinkhorn convergence and error bounds.
- **[Gumbel-Sinkhorn Networks](https://arxiv.org/abs/1802.08665)**: differentiable approximate permutations via Sinkhorn normalization.
- **[Sinkhorn Transformers](https://arxiv.org/abs/2002.11296)**: uses Sinkhorn-based sorting/permutation approximations for efficient attention.
- **[NeuralSort](https://arxiv.org/abs/1903.08850)**: continuous relaxation of sorting and ranking.
- **[SoftSort](https://arxiv.org/abs/2006.16038)**: improved differentiable sorting / ranking relaxations.
- **[Reparameterizing the Birkhoff Polytope for Variational Permutation Inference](https://arxiv.org/abs/1808.06884)**: variational inference with Birkhoff-polytope relaxations, relevant to parameterizing DS matrices.
- **[On orthostochastic, unistochastic and qustochastic matrices](https://arxiv.org/abs/math-ph/0009032)**: studies the geometry and inclusions between DS, unistochastic, and orthostochastic sets.
- **[Orthostochastic matrix (survey-style entry)](https://en.wikipedia.org/wiki/Orthostochastic_matrix)**: summarizes basic properties; useful for definitions (not a primary source).
- **[Muon is Scalable for LLM Training](./references/Muon-is-Scalable-for-LLM-Training/meta/meta_info.txt)**: uses Newton–Schulz-style orthogonalization in an optimizer context (motivates reusing orthogonalization primitives).
- **[The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm](./references/The-Polar-Express-Optimal-Matrix-Sign-Methods-and-Their-Application-to-the-Muon-Algorithm/meta/meta_info.txt)**: improves matrix-sign / Newton–Schulz orthogonalization routines in practice.
- **[Accelerating Newton–Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials](./references/Accelerating-Newton-Schulz-Iteration-for-Orthogonalization-via-Chebyshev-type-Polynomials/meta/meta_info.txt)**: improves Newton–Schulz orthogonalization efficiency.
- **[Unified Newton–Schulz Orthogonalization (UNSO)](https://arxiv.org/abs/2602.02500)**: recent unification/acceleration of Newton–Schulz-style orthogonalization.
- **[Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks](./references/Revisiting-Residual-Connections-Orthogonal-Updates-for-Stable-and-Efficient-Deep-Networks/meta/meta_info.txt)**: activation-level orthogonal residual updates (a different orthogonality-based stability intervention).
- **[Fixup Initialization](https://arxiv.org/abs/1901.09321)**: initialization scheme enabling training deep residual networks without normalization.
- **[ReZero](https://arxiv.org/abs/2003.04887)**: residual scaling for stabilizing deep Transformers.
- **[SkipInit](https://arxiv.org/abs/2002.08171)**: initialization emphasizing identity mappings to stabilize deep residual learning.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Multi-stream residual routing | Learnable residual mixing across multiple streams | Hyper-Connections | LM pretraining loss; stability spikes | Can be unstable without constraints |
| DS-constrained routing (Birkhoff polytope) | Constrain \(H^{res}\) to be doubly-stochastic for stability-by-construction | mHC, mHC-lite, KromHC | LM pretraining + downstream; DS error; gain metrics | Iterative projection cost (mHC); factorial/simplex (mHC-lite); structural restriction (KromHC) |
| Differentiable DS / permutation relaxations | Use Sinkhorn or sorting relaxations for differentiable assignments | Gumbel-Sinkhorn, Sinkhorn Transformers, NeuralSort, SoftSort | Assignment/sorting tasks; attention variants | Often approximate; may need careful tuning |
| Orthogonalization primitives | Fixed-step matrix iterations approximating orthogonal/polar factors | Newton–Schulz literature; Muon/Polar Express; UNSO | Optimizer speed/robustness; matrix function accuracy | Mixed-precision stability; iteration hyperparameters |
| Orthogonality-based residual stability (activation-level) | Enforce orthogonality of residual updates in feature space | Orthogonal Residual Updates | Vision/sequence training stability | Not about constrained \(H^{res}\) routing matrices |

### Closest Prior Work

- **mHC** ([paper](./references/mHC-Manifold-Constrained-Hyper-Connections/meta/meta_info.txt)): Constrains residual routing to the full Birkhoff polytope using finite-iteration Sinkhorn. Does not study strict subsets of DS matrices.
- **mHC-lite** ([paper](./references/mHC-lite-You-Dont-Need-20-Sinkhorn-Knopp-Iterations/meta/meta_info.txt)): Provides an exact DS parameterization but retains full DS expressiveness (for \(n=4\)). Does not test whether fewer degrees of freedom would suffice.
- **KromHC** ([paper](./references/KromHC-Manifold-Constrained-Hyper-Connections-with-Kronecker-Product-Residual-Matrices/meta/meta_info.txt)): Imposes Kronecker structure to obtain exact DS matrices; implicitly studies a structured subset, but one defined by Kronecker factorization rather than orthostochasticity.
- **Newton–Schulz orthogonalization in optimizers** (Muon/Polar Express and follow-ups): Uses similar matrix iterations as a practical primitive, but for optimizer updates rather than architectural routing constraints.

**Novelty Kill Search Summary:** Searched for combinations of “orthostochastic” / “unistochastic” / “Newton–Schulz” with “Hyper-Connections” / “mHC” / “Birkhoff polytope residual mixing”, and checked local KB for any mentions in the mHC/mHC-lite/KromHC papers. As of 2026-02-22, no prior paper was found that studies orthostochastic \(H^{res}\) constraints for Hyper-Connections; the only evidence is an open-source implementation switch in the tokenbender repo.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [mHC](./references/mHC-Manifold-Constrained-Hyper-Connections/meta/meta_info.txt) | Sinkhorn-project \(H^{res}\) to DS | Uses full DS; does not test restricted subsets | Replace DS construction with orthostochastic subset | If full DS expressiveness is unnecessary, restricted subset matches stability+loss |
| [mHC-lite](./references/mHC-lite-You-Dont-Need-20-Sinkhorn-Knopp-Iterations/meta/meta_info.txt) | Exact DS via permutation-mixture | Still targets full DS; factorial parameterization | Test whether a smaller DS subset suffices | Could justify more structured DS parameterizations |
| [KromHC](./references/KromHC-Manifold-Constrained-Hyper-Connections-with-Kronecker-Product-Residual-Matrices/meta/meta_info.txt) | Exact DS via Kronecker DS factors | Structural restriction may be too specific | Test a different structured subset (orthostochastic) | Separates “structured subset suffices” from “Kronecker suffices” |
| Muon / Polar Express | Newton–Schulz orthogonalization for optimizer updates | Not about \(H^{res}\) constraints | Reuse orthogonalization primitive for residual routing | Provides a principled constrained family (squared-orthogonal) for \(H^{res}\) |

---

## Experiments

### Experimental Setup

We use the open-source tokenbender nanoGPT implementation of Hyper-Connections/mHC for a direct apples-to-apples comparison.

**Main comparison (two conditions):**
1. **mHC-Sinkhorn (baseline)**: `mhc=True`, `mhc_h_res_proj="sinkhorn"`, `sinkhorn_iters=10`, `sinkhorn_tau=0.05`.
2. **mHC-Orthostochastic (ours)**: `mhc=True`, `mhc_h_res_proj="orthostochastic"`, `ns_steps=10`, `ns_eps=1e-7`, `ns_coeffs=(3.0,-3.2,1.2)`.

**Evaluation settings (to address generalizability along the key axis \(n\)):**
- **Setting A (deep, standard)**: 48-layer nanoGPT config on FineWeb10B with `hc_num_streams=4` (the most common setting in HC/mHC ablations).
- **Setting B (wider streams)**: 6-layer nanoGPT config on FineWeb10B with `hc_num_streams=8` to test whether conclusions hold when scaling the number of residual streams beyond 4 (where permutation-mixture parameterizations become factorial).

**Optional sanity check (calibration only; not part of main table):**
- **Unconstrained HC**: `mhc=False`, `hc_num_streams=4` to calibrate what “unstable” spike ratios look like in this codebase.

**Base Models / Configs:**

| Setting | Config file (tokenbender repo) | Model summary | Notes |
|---|---|---|---|
| A | `train_fineweb10B_mhc_48l.py` | 48 layers; `n_embd=150`, `n_head=6`, `block_size=1024` | `batch_size=8`, `grad_accum=4`, `max_iters=5000` |
| B | `train_fineweb10B_mhc.py` | 6 layers; `n_embd=288`, `n_head=6`, `block_size=1024` | `batch_size=32`, `grad_accum=4`, `max_iters=5000`; change `hc_num_streams` to 8 |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| FineWeb (10B token subset) | LM pretraining | 10B tokens | https://huggingface.co/datasets/HuggingFaceFW/fineweb | ODC-By (per HF dataset card) |

**Evaluation code:**
- Use the training and evaluation loops from the tokenbender repo’s `examples/nanogpt/` directory.

**Resource Estimate (conservative):**
- Evidence anchor: reproducing GPT-2 (124M) on FineWeb10B in ~90 minutes on 8×A100 is reported in the llm.c/nanoGPT ecosystem ([GitHub discussion](https://github.com/karpathy/llm.c/discussions/481)). Our runs are 5k iters and smaller than GPT-2, so per-run time should be well under 1 hour on 4×A100.
- Budget assumption: **1 hour per run on 4×A100** ⇒ 4 GPU-hours/run.
- Total compute:
  - Setting A (48L, n=4): 2 conditions × 5 seeds × 4 GPU-hours ≈ 40 GPU-hours
  - Setting B (6L, n=8): 2 conditions × 3 seeds × 4 GPU-hours ≈ 24 GPU-hours (reduced seeds for budget)
  - Calibration HC run: ≤4 GPU-hours
  - **Total ≤ 68 GPU-hours**.

This is well within the 768 GPU-hour budget.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| FineWeb10B LM pretraining | Next-token prediction on web text | Validation loss (lower is better) | val | https://huggingface.co/datasets/HuggingFaceFW/fineweb | tokenbender nanoGPT eval loop |

**Stability metrics (pre-registered; automated):**
- Let \(g_t\) be global gradient norm at step \(t\). Define spike ratio
  \[
  r_t = \frac{g_t}{\mathrm{median}(g_{t-100:t-1})}
  \]
  for \(t > 200\) (after warmup).
- Primary stability statistic: \(r_{\max} = \max_t r_t\).

**Constraint diagnostics (automated):**
- DS error: max row/col sum deviation from 1 for \(H^{res}\).
- Orthogonality residual: \(\lVert O O^\top - I\rVert_F\) for the Newton–Schulz-produced \(O\) prior to squaring.

### Main Results

#### Results Table

(All results are to be filled by the Verification module; no published numbers were found for this exact codebase + config.)

| Setting | Method | Base Model | Benchmark | Val loss (mean±std) | r_max (mean±std) | DS error (mean±std) | Source | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| A (48L, n=4) | mHC-Sinkhorn | nanoGPT | FineWeb10B val | **TBD** | **TBD** | **TBD** | – | 5 seeds |
| A (48L, n=4) | **mHC-Orthostochastic (ours)** | nanoGPT | FineWeb10B val | **TBD** | **TBD** | **TBD** | – | 5 seeds; ns_steps=10 |
| B (6L, n=8) | mHC-Sinkhorn | nanoGPT | FineWeb10B val | **TBD** | **TBD** | **TBD** | – | 3 seeds |
| B (6L, n=8) | **mHC-Orthostochastic (ours)** | nanoGPT | FineWeb10B val | **TBD** | **TBD** | **TBD** | – | 3 seeds; ns_steps=10 |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| HC (unconstrained; 1 seed) | `mhc=False`, `hc_num_streams=4` | Larger r_max and/or instability compared to mHC variants (used only to calibrate spike metric) |

### Experimental Rigor

- **Seeds**:
  - Setting A (48L, n=4): `seeds=[1,2,3,4,5]` for both conditions.
  - Setting B (6L, n=8): `seeds=[1,2,3]` for both conditions.
- **Fair comparison conditions**: identical hardware, batch size, learning-rate schedule, data order, and total iterations across conditions.
- **Top confounders + controls**:
  1) **Iteration hyperparameter confound** (Sinkhorn iters vs NS steps) → lock to the repo defaults for Sinkhorn and lock `ns_steps=10` for orthostochastic.
  2) **Non-DS behavior in orthostochastic due to finite precision** → log DS error and orthogonality residual; treat large sustained DS error as an explicit failure mode.
  3) **Noise floor on short runs** → use 5 seeds and define success/failure thresholds relative to the measured Sinkhorn standard deviation.

---

## Success Criteria

**Hypothesis** (directional): Orthostochastic residual mixing will match mHC-Sinkhorn in stability and validation loss.

**Decision Rule** (concrete):

We apply the decision rule **separately in each evaluation setting** (A: 48L, n=4; B: 6L, n=8).

For a given setting, let \(\mu_S,\sigma_S\) be the mean and std of final validation loss for mHC-Sinkhorn across seeds, and \(\mu_O\) be the mean for mHC-Orthostochastic.

- **Proceed (orthostochastic subset is sufficient at this setting)** if:
  - \(\mu_O - \mu_S \le 0.5\sigma_S\), and
  - \(r_{\max,O} \le \max(1.2\,r_{\max,S},\, 3.0)\) where \(r_{\max}\) is the maximum gradient spike ratio (absolute fallback when Sinkhorn is already very stable), and
  - median **doubly-stochastic error** for Orthostochastic stays ≤ 1e-3.

- **Refute (subset is too restrictive or numerically unstable at this setting)** if any of:
  - \(\mu_O - \mu_S \ge 1.0\sigma_S\), or
  - \(r_{\max,O} > \max(1.2\,r_{\max,S},\, 3.0)\), or
  - Orthostochastic doubly-stochastic error > 1e-2 for a sustained window (e.g., >200 steps).

- **Inconclusive** if \(0.5\sigma_S < \mu_O - \mu_S < 1.0\sigma_S\) while stability metrics match; in this case, extend training length (e.g., 2× iters) before drawing conclusions.

**Overall interpretation**:
- If Proceed holds for both A and B, treat this as evidence that orthostochastic constraints are viable for scaling \(n\) beyond 4.
- If Proceed holds for A but Refute holds for B, treat this as evidence that orthostochastic is viable only in the \(n=4\) regime and becomes too restrictive as \(n\) grows.

---

## Impact Statement

If orthostochastic \(H^{res}\) constraints match Sinkhorn-projected mHC, it would show that mHC’s stability/performance gains do not require the full expressiveness of general doubly-stochastic matrices, supporting future designs that restrict residual routing to smaller structured families. If it fails, the result is still decision-changing: it provides evidence that Hyper-Connections benefit from interior Birkhoff-polytope degrees of freedom beyond permutation-like routing, strengthening the case for exact-but-expressive DS parameterizations such as mHC-lite and KromHC.

---

## References

- [Hyper-Connections](./references/Hyper-Connections/meta/meta_info.txt) - Zhu et al., 2024
- [mHC: Manifold-Constrained Hyper-Connections](./references/mHC-Manifold-Constrained-Hyper-Connections/meta/meta_info.txt) - Xie et al., 2025/2026
- [mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations](./references/mHC-lite-You-Dont-Need-20-Sinkhorn-Knopp-Iterations/meta/meta_info.txt) - Yang & Gao, 2026
- [KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices](./references/KromHC-Manifold-Constrained-Hyper-Connections-with-Kronecker-Product-Residual-Matrices/meta/meta_info.txt) - Zhou et al., 2026
- [tokenbender/mHC-manifold-constrained-hyper-connections](./references/GitHub-tokenbender-mHC-manifold-constrained-hyper-connections/meta/meta_info.txt) - GitHub repo, accessed 2026-02-22
