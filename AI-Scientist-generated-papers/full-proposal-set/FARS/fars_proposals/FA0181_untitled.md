# untitled

# Data-Free Transition-Spectrum Winsorization for Mamba Long-Context Generalization

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Long-context language modeling matters for applications like document understanding, long-form summarization, and codebase navigation. Transformer attention scales quadratically with sequence length, so there is strong interest in architectures with sub-quadratic complexity.

Structured state space models (SSMs) are a prominent alternative that can process long sequences with linear-time recurrence. **Mamba** is a widely used SSM architecture for language modeling that achieves strong performance at moderate context lengths while retaining favorable asymptotic complexity.

However, multiple reports show that pretrained Mamba models can fail catastrophically when evaluated far beyond their pretraining context length: perplexity can explode by orders of magnitude at 32K–64K tokens (for example, on **PG19**, a long-document language modeling benchmark consisting of books). Recent “training-free” long-context methods for Mamba often still require a small calibration set and an optimization loop (e.g., **Simultaneous Perturbation Stochastic Approximation, SPSA**) to tune per-layer scaling factors, which complicates deployment.

### The Problem

The core problem is **length extrapolation**: using a pretrained Mamba model on sequence lengths much longer than it saw during training causes severe quality degradation.

Recent work provides competing explanations and interventions:

- **[MambaExtend](./references/MambaExtend-A-Training-Free-Approach-to-Improve-Long-Context-Extension-of-Mamba/meta/meta_info.txt)** (Azizi et al., 2025) attributes the failure largely to out-of-distribution discretization step sizes and calibrates per-layer scaling factors for the discretization steps.
- **[LongMamba](./references/LongMamba-Enhancing-Mambas-Long-Context-Capabilities-via-Training-Free-Receptive-Field-Enlargement/meta/meta_info.txt)** (Ye et al., 2025) improves long-context behavior by filtering token updates for “global” channels, but requires threshold calibration and additional hyperparameters.
- **[DeciMamba](./references/DeciMamba-Exploring-the-Length-Extrapolation-Potential-of-Mamba/meta/meta_info.txt)** (Ben-Kish et al., 2025) uses token decimation based on internal signals and introduces additional sequence-modifying mechanisms.
- **[Mamba Modulation](./references/Mamba-Modulation-On-the-Length-Generalization-of-Mamba/meta/meta_info.txt)** (Lu et al., 2025) argues that the root cause is the **eigenvalue spectrum of the transition matrix** \(\Lambda=\mathrm{diag}(\exp(-A))\). Their theory (Section 4.2) implies that eigenvalues near 1 (slow decay) and near 0 (fast decay) can respectively induce state explosion and state vanishing in long sequences. They report that calibrating per-layer scaling of \(A\) can dramatically improve perplexity at 64K.

A critical deployment-relevant gap remains: the strongest spectrum-based method in Mamba Modulation uses **data-dependent calibration** to choose per-layer scaling factors. Their simpler **constant scaling of \(A\)** is training-free but is substantially worse than calibrated scaling on language modeling perplexity.

### Key Insight and Hypothesis

**Key insight:** If long-context failures are driven mainly by a small number of extreme transition eigenvalues (very close to 0 or 1) in each layer, then a deterministic, **data-free spectral winsorization** (clipping) of those eigenvalues may recover much of the benefit of calibrated spectrum scaling.

**Hypothesis:** For a pretrained Mamba2 model, replacing each layer’s transition spectrum \(\lambda\in(0,1)\) with a winsorized spectrum that clamps only the most extreme \(\lambda\) values (per layer) will reduce long-context perplexity (PG19 at 64K tokens) **more than uniform constant scaling of \(A\)**, while preserving short-context perplexity (PG19 at 2K tokens).

**Mechanism hypothesis (why winsorization should beat constant scaling):** Constant scaling of \(A\) by a global factor \(s\) implements a one-parameter power transform on eigenvalues:
\[
\lambda \leftarrow \exp(-sA)=\lambda^s,
\]
so **every** channel in **every** layer shifts. To fix a small set of problematic slow-decay modes (\(\lambda\approx 1\)) that drive state explosion at long contexts, \(s\) must be large enough, but this simultaneously pushes already-small eigenvalues further toward 0 (accelerating vanishing) and distorts mid-spectrum timescales that are useful at short contexts. Winsorization instead only clamps the per-layer tails (\(\approx 2q\) of channels for the percentile rule), preserving the mid-spectrum while bounding the extreme modes. If long-context degradation is outlier-dominated, winsorization should therefore achieve a strictly better long/short tradeoff than global constant scaling.

Why this could be wrong: the effective transition is \(\lambda_{\mathrm{eff}}=\exp(-\Delta_t\,A)\) where \(\Delta_t\) is input-dependent; if extreme \(\lambda_{\mathrm{eff}}\) values are mostly caused by \(\Delta_t\) spikes rather than by static outliers in \(A\), then static winsorization of \(\lambda=\exp(-A)\) may not help.

---

## Proposed Approach

### Overview

We propose a **data-free, one-shot** modification to pretrained Mamba/Mamba2 weights:

1. For each layer, compute the diagonal transition spectrum \(\lambda=\exp(-A_{\text{pos}})\), where \(A_{\text{pos}}\gt 0\) is the positive parameterization of the continuous-time transition (Mamba stores \(A\) in log-space).
2. Winsorize (clip) \(\lambda\) per layer to remove only the most extreme values near 0 and near 1.
3. Convert the winsorized \(\lambda\) back into \(A\) parameters and run inference normally.

This introduces **no runtime overhead** beyond the standard forward pass, and does not require any calibration data.

### Method Details

Let a given layer’s diagonal spectrum be \(\lambda \in (0,1)^d\). We define per-layer winsorization as:

\[
\lambda'_i = \mathrm{clip}(\lambda_i, \; q_{\text{low}},\; q_{\text{high}}),
\]

where \(q_{\text{low}}\) and \(q_{\text{high}}\) are either:

- **Percentile thresholds (default)**: \(q_{\text{low}}=\mathrm{Quantile}(\lambda, q)\), \(q_{\text{high}}=\mathrm{Quantile}(\lambda, 1-q)\) with \(q=1\%\), computed independently per layer.
- **Fixed thresholds (ablation)**: \(q_{\text{low}}=\lambda_{\min},\; q_{\text{high}}=\lambda_{\max}\), with \(\lambda_{\min}\in[0.05,0.2]\), \(\lambda_{\max}\in[0.95,0.995]\).

We then map \(\lambda'\) back to \(A\) parameters via \(A'_{\text{pos}}=-\log(\lambda')\) and the corresponding log-parameterization used by the implementation.

Intuition: this implements the qualitative prescription from Mamba Modulation (“compress large eigenvalues and inflate small ones”) but does so deterministically by clipping only the extremes rather than learning per-layer scaling factors.

### Key Innovations

- **Data-free spectrum shaping** for long-context extension in Mamba: no calibration set, no SPSA, no optimization loop.
- **Targeted (non-uniform) spectrum modification** that is strictly more expressive than uniform constant scaling of \(A\), while remaining simple and deterministic.
- **A falsifiable diagnostic framing**: winsorization should improve long-context perplexity only if long-context failures are dominated by static spectral outliers in \(A\), not by purely input-dependent \(\Delta_t\) dynamics.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) long-context language modeling, (ii) state space model architectures such as Mamba, and (iii) post-hoc / training-free methods for length extrapolation.

For Transformers, many context extension methods modify positional representations (e.g., rotary embedding scaling) without changing the core model weights. For SSMs, the analogous levers include discretization steps and transition dynamics. Recent Mamba-specific work suggests that the transition spectrum is a key determinant of long-context stability, motivating spectrum-aware interventions.

### Related Papers

- **[Mamba Modulation: On the Length Generalization of Mamba](./references/Mamba-Modulation-On-the-Length-Generalization-of-Mamba/meta/meta_info.txt)**: Attributes long-context failure to the transition spectrum \(\mathrm{diag}(\exp(-A))\) and proposes calibrated per-layer scaling of \(A\).
- **[MambaExtend: A Training-Free Approach to Improve Long Context Extension of Mamba](./references/MambaExtend-A-Training-Free-Approach-to-Improve-Long-Context-Extension-of-Mamba/meta/meta_info.txt)**: Calibrates per-layer scaling factors for discretization step sizes \(\Delta\) to extend context without full fine-tuning.
- **[LongMamba: Enhancing Mamba's Long-Context Capabilities via Training-Free Receptive Field Enlargement](./references/LongMamba-Enhancing-Mambas-Long-Context-Capabilities-via-Training-Free-Receptive-Field-Enlargement/meta/meta_info.txt)**: Improves long-context by token filtering targeted at “global channels”, requiring calibration of thresholds.
- **[DeciMamba: Exploring the Length Extrapolation Potential of Mamba](./references/DeciMamba-Exploring-the-Length-Extrapolation-Potential-of-Mamba/meta/meta_info.txt)**: Introduces token decimation based on internal signals to maintain effective receptive field at long lengths.
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)**: Introduces the Mamba architecture and selective SSM blocks.
- **[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)**: Develops SSM–Transformer duality and introduces Mamba2-style refinements.
- **[Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396)**: Foundational structured SSM work enabling efficient long-sequence modeling.
- **[Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://openreview.net/forum?id=UvTo3tVBk2)**: Shows eigenvalue constraints critically affect expressivity and stability in linear RNN/SSM-like models.
- **[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)](https://arxiv.org/abs/2108.12409)**: Transformer length extrapolation via positional bias.
- **[Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595)**: Transformer context extension by rescaling positions.
- **[YaRN: Efficient Context Window Extension of Large Language Models](https://openreview.net/forum?id=wHBfxhZu1u)**: Efficient RoPE rescaling and related tricks for long-context extension.
- **[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)**: Introduces rotary position embeddings widely used in LLMs.
- **[LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508)**: Benchmark for long-context understanding across tasks.
- **[The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)**: Large pretraining corpus used in many LMs and referenced by Mamba Modulation.
- **[Compressive Transformers for Long-Range Sequence Modelling](https://openreview.net/forum?id=SylKikSYDH)**: Introduces PG19-style long-document evaluation and memory-compression baselines.
- **[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)**: Sparse attention model for long documents.
- **[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)**: Sparse attention with theoretical guarantees.
- **[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)**: LSH attention for long sequences.
- **[Performer: Rethinking Attention with Linear Complexity](https://arxiv.org/abs/2009.14794)**: Kernelized attention approximation.
- **[FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)**: Systems optimization for long-context attention.
- **[Retentive Network (RetNet)](https://arxiv.org/abs/2307.08621)**: Retention-based architecture related to long-sequence modeling.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Calibration of transition dynamics | Tune scaling factors for \(\Delta\) or \(A\) using a small calibration set | MambaExtend; Mamba Modulation | PG19/ProofPile/GovReport perplexity; LongBench | Requires calibration data + optimization loop |
| Token filtering / decimation | Reduce effective sequence length by skipping updates/tokens | LongMamba; DeciMamba | LongBench(-E); passkey retrieval; PG19 | Additional hyperparameters; may lose information |
| Transformer positional extrapolation | Modify positional encoding/bias to extrapolate beyond training length | ALiBi; Position Interpolation; YaRN | LongBench; synthetic retrieval | Mostly Transformer-specific; not directly applicable to SSM dynamics |
| Structured SSM foundations | Stable/structured parameterizations enabling long-sequence modeling | S4; Mamba | LRA; language modeling perplexity | Long-context generalization not guaranteed without additional methods |

### Closest Prior Work

- **[Mamba Modulation](./references/Mamba-Modulation-On-the-Length-Generalization-of-Mamba/meta/meta_info.txt)**: Closest motivation and baseline. It diagnoses long-context failure via the transition spectrum and shows calibrated scaling of \(A\) can yield very low PG19 perplexity at 64K (Table 3). Our proposal removes the calibration loop and instead applies a deterministic winsorization rule.
- **[MambaExtend](./references/MambaExtend-A-Training-Free-Approach-to-Improve-Long-Context-Extension-of-Mamba/meta/meta_info.txt)**: Most similar “training-free” approach, but it still requires calibration to learn per-layer \(\Delta\) scaling. We do not tune any parameters.
- **[LongMamba](./references/LongMamba-Enhancing-Mambas-Long-Context-Capabilities-via-Training-Free-Receptive-Field-Enlargement/meta/meta_info.txt)**: Training-free but changes the effective computation by filtering token updates and requires calibration of thresholds. We leave the computation unchanged and only alter transition parameters.
- **[DeciMamba](./references/DeciMamba-Exploring-the-Length-Extrapolation-Potential-of-Mamba/meta/meta_info.txt)**: Also modifies the sequence seen by later layers via decimation. Our approach keeps the full sequence and targets stability of the underlying recurrence.

**Novelty Kill Search Summary:** Searched for combinations of "Mamba eigenvalue clipping", "Mamba spectrum clipping long context", "winsorization eigenvalues Mamba", and checked local KB for "eigenvalue clip / spectrum clip / winsor". No prior work explicitly proposing **data-free eigenvalue winsorization** for Mamba long-context extension was found as of 2026-02-20. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Mamba Modulation | Calibrates per-layer scaling of \(A\) to reshape spectrum | Requires calibration data + optimization | Replace calibration with deterministic winsorization | If failures are driven by spectral outliers, clipping should approximate scaling at near-zero cost |
| Constant scaling of \(A\) (in Mamba Modulation) | Multiply \(A\) by a single constant across layers | Too blunt; does not handle layer heterogeneity | Non-uniform per-layer clipping of extremes | Preserves mid-spectrum while fixing only pathological modes |
| MambaExtend | Calibrates scaling of \(\Delta\) | Calibration loop; indirect control of spectrum | Directly reshape \(\lambda\) without tuning | Directly targets the spectrum that governs stability |
| LongMamba / DeciMamba | Token filtering / decimation | Sequence modification + hyperparameters | Weight-only spectrum shaping | Keeps full information; no new kernels or token dropping |

---

## Experiments

### Experimental Setup

**Task:** Long-context language modeling evaluation via perplexity (PPL) on PG19.

**Evaluation harness reference:** Prefer matching the evaluation protocol from the Mamba Modulation codebase (if available) to reduce harness mismatch; otherwise use a standard teacher-forced NLL-over-windows implementation and validate by reproducing Table 3 within ~10%.

**Baseline Ladder (REQUIRED):**
- Prompting baselines are **not applicable** because perplexity is computed by teacher forcing (no prompt formatting choice).
- Inference-time scaling baselines (best-of-N / self-consistency) are **not applicable** because perplexity evaluation does not sample outputs.
- We therefore use a ladder of increasingly strong **architecture/parameter interventions** under identical evaluation:
  1) Base pretrained model
  2) Constant scaling of \(A\) (training-free baseline reported by Mamba Modulation)
  3) **Ours**: data-free winsorization of \(\lambda\)
  4) (Context upper bound) Calibrated scaling of \(A\) from Mamba Modulation
  5) (Context) MambaExtend / LongMamba / DeciMamba numbers from Mamba Modulation Table 3

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| state-spaces/mamba2-1.3b | 1.3B | https://huggingface.co/state-spaces/mamba2-1.3b | Primary target (matches Mamba Modulation Table 3) |

**Evaluation protocol (planned):**
- Use the official `pg19` dataset (validation split) from HuggingFace Datasets.
- Form evaluation sequences by concatenating documents and taking fixed-length token windows.
- Compute next-token negative log-likelihood under teacher forcing and report perplexity.
- Evaluate at **2K** (in-distribution) and **64K** (out-of-distribution) context lengths.

**Reproducibility:**
- The model forward pass is deterministic in eval mode.
- To avoid randomness in sequence sampling, use a fixed deterministic set of evaluation windows (or a fixed RNG seed for window sampling, e.g., seed=42).

**Resource Estimate (conservative):**
- 3 conditions (Base, Constant-\(A\)-scaling, Winsorization) × 2 lengths (2K, 64K).
- Evaluate on an initial **small but decisive** set of windows (e.g., 8 windows at 64K and 64 windows at 2K).
- Expected runtime is dominated by 64K evaluation; budget **≤ 20 A100 GPU-hours** including setup/debugging and optional diagnostics.
- Peak memory: single A100 80GB should suffice for mamba2-1.3b evaluation.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| PG19 | Long-document language modeling benchmark (books) used to test long-context perplexity | Perplexity (lower is better) | validation | https://huggingface.co/datasets/pg19 | Use Mamba Modulation evaluation protocol; implement teacher-forced NLL over fixed windows |

### Main Results

**Published reference numbers (context, from Mamba Modulation Table 3; 1 run):**
- mamba2-1.3b PG19 PPL: Base 64K=1479.45; Constant scaling \(A\) 64K=13.22; Calibrated scaling \(A\) 64K=4.72.

#### Results Table

| Method | Base Model | Benchmark | PPL@2K | PPL@64K | Source | Notes |
|---|---|---|---:|---:|---|---|
| Base model | mamba2-1.3b | PG19 | 9.52 (1 run) | 1479.45 (1 run) | [Mamba Modulation Table 3](./references/Mamba-Modulation-On-the-Length-Generalization-of-Mamba/sections/6.4%20Comparison%20with%20Alternative%20Methods.md) | Published; verifier should reproduce as sanity check |
| Constant scaling \(A\) | mamba2-1.3b | PG19 | 11.12 (1 run) | 13.22 (1 run) | [Mamba Modulation Table 3](./references/Mamba-Modulation-On-the-Length-Generalization-of-Mamba/sections/6.4%20Comparison%20with%20Alternative%20Methods.md) | Published; our main baseline to beat |
| **Winsorized \(\lambda\) (ours)** | mamba2-1.3b | PG19 | **TBD** | **TBD** | - | To be verified |
| Calibrated scaling \(A\) (upper bound) | mamba2-1.3b | PG19 | 4.38 (1 run) | 4.72 (1 run) | [Mamba Modulation Table 3](./references/Mamba-Modulation-On-the-Length-Generalization-of-Mamba/sections/6.4%20Comparison%20with%20Alternative%20Methods.md) | Cited as context; rerun only if harness mismatch |

### Ablation Studies

If the main result is positive but sensitive, run at most **one** small ablation:

| Variant | What’s changed | Expected finding |
|---|---|---|
| Winsorization (q=0.5% vs 1%) | Clip fewer extremes | If outliers dominate, mild clipping may suffice; otherwise performance may drop |

### Experimental Rigor

**Confounders and controls:**
- **Evaluation harness mismatch vs paper**: first reproduce Base and Constant-\(A\)-scaling within ~10% of Table 3. If mismatch is larger, rerun calibrated scaling too or adjust protocol until matched.
- **Implementation bug in spectrum conversion**: verify that winsorization does not produce NaNs/infs and that \(\lambda'\in(0,1)\).

**Mechanism diagnostics (required if reporting a positive result):**
- **Tail modification rate**: For each layer, report the fraction of channels whose \(\lambda\) changed under winsorization vs constant scaling (winsorization should only affect \(\approx 2q\) channels per layer for percentile \(q\); constant scaling affects 100%).
- **Effective-transition tail mass**: Log per-token effective transitions \(\lambda_{\mathrm{eff}}=\exp(-\Delta_t A)\) on a few 64K windows before/after winsorization. If winsorization reduces the mass of \(\lambda_{\mathrm{eff}}\) near 0 and 1 (or specifically reduces the \(\lambda\approx 1\) mass responsible for explosion) while leaving the bulk similar, it supports the outlier-dominance mechanism.

---

## Success Criteria

**Hypothesis** (directional): Winsorizing the transition spectrum reduces PG19 PPL at 64K relative to constant scaling of \(A\), with minimal short-context regression.

**Decision Rule** (concrete — when to stop):
- **Proceed** if winsorization achieves **PPL@64K ≤ 9.0**, i.e., it closes **≥50% of the gap** between constant scaling (13.22) and calibrated scaling (4.72): 
  \(9.0 \approx (13.22+4.72)/2\), while maintaining **PPL@2K ≤ 1.05×** the base model’s PPL@2K. If reporting success, also include the required mechanism diagnostics (tail modification rate + \(\lambda_{\mathrm{eff}}\) tail mass).
- **Pivot** if winsorization beats constant scaling (**PPL@64K < 13.22**) but does not reach 9.0; try one fixed-threshold variant (e.g., clamp \(\lambda\) to [0.1, 0.99]) or one alternate percentile (q=0.5%).
- **Refute** if winsorization yields **PPL@64K ≥ 13.22**, or if **PPL@2K > 1.10×** the base model’s PPL@2K.

---

## Impact Statement

If successful, this provides a simple, deterministic, data-free procedure to make pretrained Mamba models usable at much longer contexts without any calibration loop. This would reduce deployment friction for SSM-based long-context models in document and code applications where collecting representative calibration data is costly or impractical.

---

## References

- [Mamba Modulation: On the Length Generalization of Mamba](./references/Mamba-Modulation-On-the-Length-Generalization-of-Mamba/meta/meta_info.txt) - Peng Lu et al., 2025
- [MambaExtend: A Training-Free Approach to Improve Long Context Extension of Mamba](./references/MambaExtend-A-Training-Free-Approach-to-Improve-Long-Context-Extension-of-Mamba/meta/meta_info.txt) - Seyedarmin Azizi et al., 2025
- [LongMamba: Enhancing Mamba's Long-Context Capabilities via Training-Free Receptive Field Enlargement](./references/LongMamba-Enhancing-Mambas-Long-Context-Capabilities-via-Training-Free-Receptive-Field-Enlargement/meta/meta_info.txt) - Zhifan Ye et al., 2025
- [DECIMAMBA: Exploring the Length Extrapolation Potential of Mamba](./references/DeciMamba-Exploring-the-Length-Extrapolation-Potential-of-Mamba/meta/meta_info.txt) - Assaf Ben-Kish et al., 2025
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) - Albert Gu, Tri Dao, 2023
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) - Tri Dao, Albert Gu, 2024
- [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) - Albert Gu et al., 2021
- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://openreview.net/forum?id=UvTo3tVBk2) - Riccardo Grazzi et al., 2025
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)](https://arxiv.org/abs/2108.12409) - Ofir Press et al., 2021
- [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595) - Shuo Chen et al., 2023
- [YaRN: Efficient Context Window Extension of Large Language Models](https://openreview.net/forum?id=wHBfxhZu1u) - Bowen Peng et al., 2024
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Jianlin Su et al., 2021
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508) - Yushi Bai et al., 2023
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) - Leo Gao et al., 2021
- [Compressive Transformers for Long-Range Sequence Modelling](https://openreview.net/forum?id=SylKikSYDH) - Jack Rae et al., 2020
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Iz Beltagy et al., 2020
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) - Manzil Zaheer et al., 2020
- [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) - Nikita Kitaev et al., 2020
- [Performer: Rethinking Attention with Linear Complexity](https://arxiv.org/abs/2009.14794) - Krzysztof Choromanski et al., 2020
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) - Tri Dao et al., 2022
- [Retentive Network (RetNet)](https://arxiv.org/abs/2307.08621) - Yiwei Sun et al., 2023
