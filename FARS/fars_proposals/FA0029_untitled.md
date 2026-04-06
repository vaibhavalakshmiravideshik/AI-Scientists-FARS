# untitled

# ROCKET-ActCost: Objective-Matched Knapsack Allocation for Calibration-Guided Training-Free LLM Compression

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Post-training compression reduces the memory and latency cost of running large language models (LLMs) without retraining. A common approach is to use a small **calibration set** (e.g., a few hundred text sequences) to estimate activation statistics and to optimize a structured approximation of each linear layer that preserves the layer’s outputs on that calibration distribution.

**ROCKET** is a recent training-free compression method that combines:
1) a fast dictionary-learning-inspired sparse factorization computed via a single eigendecomposition plus a closed-form least-squares update, and
2) a global **multi-choice knapsack problem (MCKP)** optimizer (a discrete budget allocation problem: choose one option per layer to minimize total error under a parameter-count budget) [ROCKET](./references/ROCKET-Rapid-Optimization-via-Calibration-guided-Knapsack-Enhanced-Truncation-for-Efficient-Model-Compression/meta/meta_info.txt).

ROCKET reports large gains over prior training-free baselines, especially under aggressive compression. For example, on **Qwen3-8B at 50% compression**, ROCKET achieves **AvgAcc 51.3** vs **38.1** (SVD-LLM) and **42.0** (CoSpaDi) in Table 1 of the ROCKET paper.

### The Problem

ROCKET’s per-layer factorization is derived from an **output reconstruction objective**: for a layer with input activations \(X\) and weight \(W\), the goal is to preserve \(XW\) after compression.

ROCKET makes this objective tractable by operating in a **whitened activation space**. Given calibration activations \(X\in\mathbb{R}^{N\times d_{in}}\), it forms the **Gram matrix** \(A=X^\top X\) (an uncentered covariance / second-moment matrix summarizing how much each input direction is used) and computes its **Cholesky factor** \(L\) (a triangular matrix such that \(A=LL^\top\)). Using decorrelated activations \(Y=XL^{-1}\) yields \(Y^\top Y\approx I\). Under this transform, output reconstruction becomes equivalent to a Frobenius error in the whitened weight space (ROCKET Eq. 2):
\[
\|XW-X\hat W\|_F = \|LW-L\hat W\|_F.
\]
Intuitively, multiplying by \(L\) reweights weight-space errors by activation energy, so errors along rarely-used activation directions matter less.

However, ROCKET’s **global allocation** step (the multi-choice knapsack problem, MCKP) does not use this output-space objective. During “Layer Profiling”, ROCKET records each candidate option’s cost using an **original weight-space** relative Frobenius error:
\[
 e^{\text{weight}}_{\ell,i}=\frac{\|W_\ell-\hat W_{\ell,i}\|_F}{\|W_\ell\|_F}.
\]
ROCKET ablates several alternative *weight-space* metrics (\(\ell_1\), spectral distance, mean cosine distance), but it does not test a **whitened/output-space** error as the knapsack cost (ROCKET Table 6).

A potential counterargument is that ROCKET already uses a **dual-space importance score** for sparsifying coefficients within each candidate (ROCKET Eq. 5), interpolating whitened-space and original-space sensitivity. We agree this partially addresses weight-vs-output mismatch *inside* a candidate. Our question is whether there remains a mismatch *across candidates* during global allocation: the MCKP objective (ROCKET Eq. 9) still treats all weight-space directions equally, which is not equivalent to the calibration-distribution output objective.

We also checked the public ROCKET implementation (`mts-ai/ROCKET`, `rocket/profiling/profiler.py`) and confirmed that it computes option error in the **original space** after inverse whitening (`err = frobenius_distance(w_recon, w)`), with no configuration to use a whitened/output-space cost.

This motivates a focused hypothesis test: does using an output-matched error proxy in the allocation step measurably improve downstream fidelity, beyond ROCKET’s existing dual-space sparsification?

### Key Insight and Hypothesis

**Key insight:** ROCKET’s profiling step already computes whitened weights \(W_{L,\ell}=L_\ell W_\ell\) and whitened reconstructions \(\hat W_{L,\ell,i}\). Therefore, an output-space cost can be computed **with no extra calibration passes**.

**Hypothesis:** Replacing ROCKET’s knapsack cost \(e^{\text{weight}}_{\ell,i}\) with a **whitened/output-space** relative error
\[
 e^{\text{out}}_{\ell,i}=\frac{\|W_{L,\ell}-\hat W_{L,\ell,i}\|_F}{\|W_{L,\ell}\|_F}
\]
will yield a better per-layer allocation under the same global budget, improving downstream average accuracy and/or perplexity, particularly at **high compression ratios** where allocation mistakes accumulate.

Why this might fail (non-triviality): calibration activations may not match downstream benchmark activations; in that case, a weight-space proxy could be more robust to distribution shift than a calibration-aligned output-space metric.

---

## Proposed Approach

### Overview

We propose **ROCKET-ActCost**, a minimal modification to ROCKET’s MCKP allocation:
- Keep ROCKET’s candidate generation, sparsification, per-layer feasibility constraint (\(\alpha_{\min}\)), and knapsack solver unchanged.
- Change only the **per-candidate error cost** recorded during profiling from original weight-space error to whitened/output-space error.

### Method Details

For each compressible linear layer \(\ell\):
1. Collect a calibration covariance \(A_\ell=X_\ell^\top X_\ell\) and compute its Cholesky factor \(L_\ell\) (ROCKET uses `Calib.get_s_inv_s` to return `ss` and `inv_s`).
2. Form the whitened weight \(W_{L,\ell}=L_\ell W_\ell\).
3. For each candidate option \(i\) (defined by **compression ratio** \(\mathrm{cr}\) and **sparsity-to-rank ratio** \(\mathrm{ks}\)), run ROCKET’s factorization to obtain \(\hat W_{L,\ell,i}\) and \(\hat W_{\ell,i}=L_\ell^{-1}\hat W_{L,\ell,i}\).
4. Record the option’s cost and error:
   - ROCKET-default: record \(e^{\text{weight}}_{\ell,i}\).
   - ROCKET-ActCost (ours): record \(e^{\text{out}}_{\ell,i}\).
5. Solve the same constrained MCKP over per-layer options to meet the global budget.

**Compute-sharing note (important for feasibility):** For fairness and efficiency, compute both \(e^{\text{weight}}\) and \(e^{\text{out}}\) for each candidate during the same profiling run, then run the knapsack DP twice (once per cost). This avoids doubling the expensive candidate reconstruction computations.

**Implementation note (verification guidance):** In the public ROCKET code, profiling forms `x = ss @ w` (whitened weight) and obtains a whitened reconstruction `x_hat = U_opt @ v_sparse` before mapping back to original space. ROCKET-ActCost replaces `err = frobenius_distance(w_recon, w) / ||w||_F` with `err = frobenius_distance(x_hat, x) / ||x||_F`.

### Key Innovations

- **Objective alignment for global allocation**: validate (or falsify) the design principle that global budget allocation should use the same activation-aware objective as the per-layer reconstruction derivation, rather than a cheaper but potentially mismatched weight-space proxy.
- **No added overhead**: the proposed cost reuses matrices already computed in profiling.

---

## Related Work

### Field Overview

Training-free LLM compression can be grouped into: (i) **quantization** (post-training quantization with calibration), (ii) **pruning/sparsification** (structured or unstructured removal guided by activation/Hessian proxies), and (iii) **factorization/low-rank** methods (replace dense matrices with structured factors). Many successful approaches are **data-aware**, explicitly optimizing an output reconstruction objective \(\|XW-X\hat W\|\) rather than a weight reconstruction objective \(\|W-\hat W\|\).

ROCKET is in the factorization family but is distinctive in combining a fast sparse-dictionary factorization with a global discrete optimizer (MCKP). Our proposal asks whether the *allocation objective* in such calibration-guided pipelines should be defined in the same (whitened/output) space as the factorization objective.

### Related Papers

- **[ROCKET](./references/ROCKET-Rapid-Optimization-via-Calibration-guided-Knapsack-Enhanced-Truncation-for-Efficient-Model-Compression/meta/meta_info.txt)**: Training-free sparse factorization + MCKP allocation using weight-space Frobenius error.
- **[CoSpaDi](./references/CoSpaDi-Compressing-LLMS-via-Calibration-Guided-Sparse-Dictionary-Learning/meta/meta_info.txt)**: Iterative sparse dictionary learning minimizing output reconstruction error.
- **[SVD-LLM](./references/SVD-LLM-Truncation-aware-Singular-Value-Decomposition-for-Large-Language-Model-Compression/meta/meta_info.txt)**: Cholesky whitening to align SVD truncation with output reconstruction loss.
- **[AFORA](../../papers/references/Under%20review%20as%20a%20conference%20paper%20at%20ICLR%202026%20AFORA%20ACTIVATION-AWARE%20FACTORIZATION%20WITH%20OPTIMAL%20RANK%20ALLOCATION%20FOR%20TRAINING-FREE%20LLM%20COMPRESSION/meta/meta_info.txt)**: Activation-aware low-rank factorization with global **water-filling** rank allocation under a budget. Closest on the “allocation cost should match an activation-aware objective” principle, but differs from ROCKET in both (i) compression mechanism (pure low-rank vs sparse dictionary), and (ii) allocator (continuous water-filling vs discrete multi-choice knapsack).
- **[BALF](https://arxiv.org/abs/2509.25136)**: Budgeted activation-aware low-rank factorization with discrete budget control (related principle; different factorization family).
- **[ASVD](https://arxiv.org/abs/2312.01244)**: Activation-aware SVD-style low-rank compression.
- **[FWSVD](https://arxiv.org/abs/2209.09736)**: Fisher-weighted SVD.
- **[Dobi-SVD](https://arxiv.org/abs/2402.09353)**: Learnable truncation thresholds for SVD-based compression.
- **[ARS](https://arxiv.org/abs/2402.08922)**: Adaptive rank selection for transformer compression.
- **[ARA](https://arxiv.org/abs/2410.04201)**: Adaptive rank allocation under a global budget.
- **[D-Rank](https://arxiv.org/abs/2509.25622)**: Layer-wise dynamic rank allocation for LLM compression.
- **[Zero-Sum SVD](https://arxiv.org/abs/2602.02848)**: Global singular-component selection via loss sensitivity.
- **[FlexRank](https://arxiv.org/abs/2602.02680)**: Output-reconstruction proxies + DP/knapsack allocation for low-rank decompositions.
- **[SparseGPT](https://arxiv.org/abs/2301.00774)**: Hessian-aware pruning optimizing output reconstruction.
- **[WANDA](https://arxiv.org/abs/2306.11695)**: Activation-weighted pruning.
- **[LLM-Pruner](https://arxiv.org/abs/2306.11695)**: Structured pruning with recovery.
- **[SliceGPT](https://arxiv.org/abs/2401.15024)**: Layer dropping / compression for transformers.
- **[GPTQ](https://arxiv.org/abs/2210.17323)**: Post-training quantization with approximate Hessian.
- **[AWQ](https://arxiv.org/abs/2306.00978)**: Activation-aware weight quantization.
- **[SmoothQuant](https://arxiv.org/abs/2211.10438)**: Smooth activation scaling for quantization.
- **[SpQR](https://arxiv.org/abs/2306.03078)**: Sparse-quantization for LLMs.
- **[QuIP#](https://arxiv.org/abs/2402.04396)**: Advanced PTQ for LLMs.
- **[OBC / OBS](https://proceedings.neurips.cc/paper_files/paper/1990/file/3edc8adf7df9c0c5a7b3f9b6b0e3f0e8-Paper.pdf)**: Classic output-error-aware compression motivation.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Typical evaluation | Known limitations |
|---|---|---|---|---|
| Data-aware low-rank | Whiten activations then truncate SVD | SVD-LLM, ASVD, FWSVD | PPL + zero-shot tasks | Shared subspace can be too rigid |
| Sparse dictionary factorization | Union-of-subspaces via sparse coefficients | CoSpaDi, ROCKET | PPL + zero-shot tasks | Iterative methods slow; allocation objectives vary |
| Global budget allocation | Allocate ranks/sparsity across layers | ROCKET, BALF, ARA, D-Rank, FlexRank | Accuracy/PPL at fixed CR | Needs reliable per-layer cost proxy |
| Output-error-aware pruning | Use activation/Hessian proxies | SparseGPT, WANDA | Accuracy/PPL + throughput | Unstructured sparsity is hard to accelerate |
| Post-training quantization | Quantize with calibration | GPTQ, AWQ, SmoothQuant, SpQR | PPL + throughput | Calibration sensitivity; kernel complexity |

### Closest Prior Work

- **ROCKET**: Uses whitening to derive a data-aware factorization, but uses original-space weight error as the MCKP cost. We change only the MCKP cost to an output-space equivalent.
- **AFORA / BALF**: Support the general principle that global allocation should be activation-aware (and in AFORA, explicitly derived from an activation-aware objective). We test whether this principle transfers to ROCKET’s *sparse-dictionary + MCKP* setting, where the candidate generator already uses whitening and dual-space sparsification, but the global allocator remains weight-space.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| ROCKET | Sparse factorization + MCKP with weight-space cost | Allocation cost may be misaligned with output objective | Use whitened/output-space cost | Better proxy for downstream fidelity at same budget |
| AFORA / BALF | Activation-aware global allocation (water-filling / budgeted ranks) | Different compression family (low-rank, not sparse dictionary) | Test whether activation-aware allocation objective helps ROCKET’s MCKP | Clarifies whether ROCKET’s weight-space cost is leaving accuracy on the table |
| SVD-LLM | Whitening makes truncation loss-aligned | Not sparse dictionary; no MCKP over (cr,ks) | Apply “loss-aligned” idea to ROCKET MCKP | Same compute, better objective |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3-8B | 8B | https://huggingface.co/Qwen/Qwen3-8B | Matches ROCKET Table 1 model family; used for primary test |
| Llama-3.2-1B | 1B | https://huggingface.co/unsloth/Llama-3.2-1B | Matches public ROCKET config; used as a cheap sanity check |

**Training Data (if applicable):**

No training data needed — post-training compression + evaluation only.

**Other Resources (if applicable):**
- ROCKET reference implementation: https://github.com/mts-ai/ROCKET
- Calibration data: **RefinedWeb** (a large-scale web text corpus; ROCKET uses 256 sequences of length 1024 for calibration) as in ROCKET Experimental Setup.

**Resource Estimate**:
- **Compute budget**: ≤ 200 GPU-hours total.
  - Compression is dominated by profiling candidate options; ROCKET reports 930s on 1×A100-40GB for Llama3-1B (Table 9). Qwen3-8B profiling will be substantially more expensive; we budget for 2 calibration seeds × {ROCKET-default + ROCKET-ActCost} using compute-sharing during profiling.
- **GPU memory**: 80GB recommended for Qwen3-8B profiling; 40GB sufficient for 1B.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ROCKET eval suite | Zero-shot multiple-choice suite used in ROCKET | **AvgAcc** (average accuracy over PIQA/HellaSwag/LAMBADA/ARC-e/ARC-c/SciQ/RACE/MMLU; higher is better) | test | standard datasets | ROCKET repo scripts / **lm-eval harness** (a standard evaluation framework for language model benchmarks) |
| WikiText perplexity | Language modeling benchmark measuring next-token prediction difficulty | **WikiText word-level perplexity** (lower is better) | test | https://huggingface.co/datasets/wikitext | ROCKET repo scripts |

### Main Results

#### Results Table

Published ROCKET baselines are included to anchor expectations in the **primary setting (Qwen3-8B @ CR=0.5)**; the verification run should reproduce them using the public ROCKET repo.

| Method | Description | Settings | Results | Reference |
|---|---|---|---|---|
| Dense | Uncompressed model | **Qwen3-8B**, CR=0.0; metrics: **AvgAcc** (average accuracy across PIQA/HellaSwag/LAMBADA/ARC-e/ARC-c/SciQ/RACE/MMLU; higher is better), **WikiText PPL** (word-level perplexity on WikiText; lower is better) | AvgAcc 70.5; WikiText PPL 1.2E+01 | [ROCKET Table 1](./references/ROCKET-Rapid-Optimization-via-Calibration-guided-Knapsack-Enhanced-Truncation-for-Efficient-Model-Compression/sections/Comparison%20with%20SVD-LLM%20and%20CoSpaDi.md) |
| SVD-LLM | Data-aware low-rank baseline | **Qwen3-8B**, CR=0.5 | AvgAcc 38.1; WikiText PPL 7.6E+01 | [ROCKET Table 1](./references/ROCKET-Rapid-Optimization-via-Calibration-guided-Knapsack-Enhanced-Truncation-for-Efficient-Model-Compression/sections/Comparison%20with%20SVD-LLM%20and%20CoSpaDi.md) |
| CoSpaDi | Sparse dictionary baseline | **Qwen3-8B**, CR=0.5 | AvgAcc 42.0; WikiText PPL 5.9E+01 | [ROCKET Table 1](./references/ROCKET-Rapid-Optimization-via-Calibration-guided-Knapsack-Enhanced-Truncation-for-Efficient-Model-Compression/sections/Comparison%20with%20SVD-LLM%20and%20CoSpaDi.md) |
| ROCKET-default | MCKP cost = \(e^{\text{weight}}\) | **Qwen3-8B**, CR=0.5 | AvgAcc 51.3; WikiText PPL 3.5E+01 | [ROCKET Table 1](./references/ROCKET-Rapid-Optimization-via-Calibration-guided-Knapsack-Enhanced-Truncation-for-Efficient-Model-Compression/sections/Comparison%20with%20SVD-LLM%20and%20CoSpaDi.md) |
| **ROCKET-ActCost (ours)** | MCKP cost = \(e^{\text{out}}\) | **Qwen3-8B**, CR=0.5 | TBD | To be verified |

### Experimental Conditions (3 total)

| ID | Condition | What changes | Training required |
|---|---|---|---|
| A | Dense baseline | No compression | None |
| B | ROCKET-default | MCKP cost = \(e^{\text{weight}}\) (original-space Frobenius) | None |
| C | ROCKET-ActCost (ours) | MCKP cost = \(e^{\text{out}}\) (whitened/output-space Frobenius) | None |

### Decision Rule (explicit)

Primary setting: **Qwen3-8B at CR=0.5** (aggressive compression).

Run **two calibration sampling seeds** (different random draws of 256 calibration sequences) and report the mean.

Let \(\Delta\mathrm{Acc} = \mathrm{AvgAcc}(C) - \mathrm{AvgAcc}(B)\) and \(r\_{\mathrm{PPL}} = \mathrm{PPL}(C) / \mathrm{PPL}(B)\).

Declare **success** if, on Qwen3-8B @ CR=0.5:
1. Mean \(\Delta\mathrm{Acc} \ge 1.0\) (absolute points) **OR** mean \(r\_{\mathrm{PPL}} \le 0.90\) (≥10% relative perplexity reduction), and
2. The improvement direction is consistent across both calibration seeds (\(\Delta\mathrm{Acc}>0\) or \(r\_{\mathrm{PPL}}<1\) in both runs), and
3. End-to-end profiling+compression runtime increases by **≤10%** vs ROCKET-default (expected since both errors are computed in the same profiling run).

Secondary check (non-blocking): verify ROCKET-ActCost is not harmful on Llama-3.2-1B @ CR=0.2 (no more than −0.5 AvgAcc).

### Ablation Studies

(All analysis-only; no new main conditions.)

- **Pre-check (recommended before full eval)**: compute Spearman correlation between \(e^{\text{weight}}\) and \(e^{\text{out}}\) across candidates per layer. If correlations are uniformly very high (e.g., >0.95 for almost all layers), expect a null result and consider running only 1 seed.
- **Mechanism analysis (required)**: report which layers show the largest ranking disagreement and whether those layers change \((k,s)\) choices under the knapsack solution.
- **Allocation diff audit**: compare the selected per-layer \((k,s)\) distributions under B vs C; break down by module type (attention vs MLP).
- **Calibration shift probe (small)**: swap the calibration corpus (RefinedWeb → WikiText) for one seed on Qwen3-8B to see whether ActCost becomes more/less favorable.

---

## Success Criteria

**Criterion 1: Allocation objective alignment yields measurable gains at aggressive compression**
- Hypothesis: ROCKET-ActCost improves AvgAcc or reduces perplexity on Qwen3-8B at CR=0.5.
- Validation: Meets the decision rule thresholds.

**Criterion 2: Explanation of when objective mismatch matters**
- Hypothesis: If ActCost helps, it will do so primarily when \(e^{\text{weight}}\) and \(e^{\text{out}}\) rankings diverge for some layers, leading to different knapsack allocations.
- Validation: Correlation + allocation-diff analyses identify the responsible layer types.

---

## Impact Statement

If successful, this provides a practical and low-risk improvement to ROCKET-style training-free compression at high compression ratios: users can improve compressed-model fidelity without changing the factorization algorithm or adding compute, by matching the knapsack allocation objective to the method’s output-preservation derivation. Independently of ROCKET, the result would support a broader design principle for calibration-guided compression systems that use global budget allocation: **define allocation costs in the same (whitened/output) space as the reconstruction objective**, especially when targeting aggressive compression regimes.

---

## References

- [ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression](./references/ROCKET-Rapid-Optimization-via-Calibration-guided-Knapsack-Enhanced-Truncation-for-Efficient-Model-Compression/meta/meta_info.txt) - Ali et al., 2026
- [CoSpaDi: Compressing LLMS via Calibration-Guided Sparse Dictionary Learning](./references/CoSpaDi-Compressing-LLMS-via-Calibration-Guided-Sparse-Dictionary-Learning/meta/meta_info.txt) - Shopkhoev et al., 2025
- [SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression](./references/SVD-LLM-Truncation-aware-Singular-Value-Decomposition-for-Large-Language-Model-Compression/meta/meta_info.txt) - Wang et al., 2024
- [AFORA: Activation-aware Factorization with Optimal Rank Allocation for Training-free LLM Compression](../../papers/references/Under%20review%20as%20a%20conference%20paper%20at%20ICLR%202026%20AFORA%20ACTIVATION-AWARE%20FACTORIZATION%20WITH%20OPTIMAL%20RANK%20ALLOCATION%20FOR%20TRAINING-FREE%20LLM%20COMPRESSION/meta/meta_info.txt) - Ha & Jeon, 2026
- [BALF](https://arxiv.org/abs/2509.25136) - 2025
- [ASVD](https://arxiv.org/abs/2312.01244) - Yuan et al., 2023
- [FWSVD](https://arxiv.org/abs/2209.09736) - Hsu et al., 2022
- [Dobi-SVD](https://arxiv.org/abs/2402.09353) - Wang et al., 2024
- [ARS](https://arxiv.org/abs/2402.08922) - Gao et al., 2024
- [ARA](https://arxiv.org/abs/2410.04201) - Xv et al., 2024
- [D-Rank](https://arxiv.org/abs/2509.25622) - 2025
- [Zero-Sum SVD](https://arxiv.org/abs/2602.02848) - 2026
- [FlexRank](https://arxiv.org/abs/2602.02680) - 2026
- [SparseGPT](https://arxiv.org/abs/2301.00774) - Frantar & Alistarh, 2023
- [WANDA](https://arxiv.org/abs/2306.11695) - Sun et al., 2023
- [LLM-Pruner](https://arxiv.org/abs/2306.11695) - Ma et al., 2023
- [SliceGPT](https://arxiv.org/abs/2401.15024) - 2024
- [GPTQ](https://arxiv.org/abs/2210.17323) - Frantar et al., 2022
- [AWQ](https://arxiv.org/abs/2306.00978) - Lin et al., 2023
- [SmoothQuant](https://arxiv.org/abs/2211.10438) - Xiao et al., 2022
- [SpQR](https://arxiv.org/abs/2306.03078) - 2023
- [QuIP#](https://arxiv.org/abs/2402.04396) - 2024
- [OBC / OBS](https://proceedings.neurips.cc/paper_files/paper/1990/file/3edc8adf7df9c0c5a7b3f9b6b0e3f0e8-Paper.pdf) - LeCun et al., 1990
