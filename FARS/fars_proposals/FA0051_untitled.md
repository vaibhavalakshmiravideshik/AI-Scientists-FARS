# untitled

# Toeplitz Block Mixing for Scalable Multi-Head Linear Attention

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Long-context language models are increasingly used in settings where inputs can reach tens to hundreds of thousands of tokens (e.g., long-document question answering, codebase-level assistance, and agent trajectories). Standard Transformer self-attention scales quadratically in sequence length, making training and inference expensive at long contexts.

A large body of work therefore studies **subquadratic** sequence models, including **state space models (SSMs)** (e.g., Mamba) and **linear attention** variants. Linear attention replaces the softmax attention kernel with a kernel approximation using a **feature map** \(\phi(\cdot)\) (a transformation of queries and keys to nonnegative vectors such that \(\phi(q)^\top\phi(k)\) approximates \(\exp(qk^\top/\sqrt{d})\)). This makes attention computable from **prefix summaries** (running sums of \(\phi(k) v^\top\)) rather than all \(O(N^2)\) query–key pairs, reducing the dominant term to linear time in the sequence length \(N\).

However, linear attention often underperforms softmax attention on long-context retrieval tasks due to an information bottleneck: all past key-value information is compressed into a small set of global summary tensors shared by every query. MHLA refers to the resulting loss of query-specific selectivity and representational diversity as **global context collapse** (the attention map becomes low-rank and closer to uniform as \(N\) grows).

Recent methods improve linear attention by re-introducing structured selectivity while keeping favorable scaling. In particular, **Multi-Head Linear Attention (MHLA)** partitions the sequence into blocks and learns a block-level mixing matrix so each query block attends to a different mixture of block summaries, addressing “global context collapse” in linear attention.

### The Problem

MHLA’s block mixing introduces a new scaling trade-off. In the chunkwise causal formulation used for autoregressive modeling, MHLA forms for each block index *i* a mixed prefix summary

\[ S_i = \sum_{b\le i} m_{i,b} S_b \]

where \(S_b\in\mathbb{R}^{d\times d}\) is the block’s key-value summary and \(m_{i,b}\) are learned mixing coefficients. With an unconstrained coefficient matrix \(M_c\in\mathbb{R}^{M\times M}\) for \(M\) blocks, computing all \(S_i\) naively costs \(O(M^2 d^2)\) per layer per sequence. MHLA mitigates this by requiring \(M^2\lesssim N\) (where \(N\) is sequence length) so block mixing does not dominate, which in practice forces either:

1. **Large blocks (coarse chunking)** to keep \(M\) small, increasing intra-block quadratic costs and reducing temporal resolution.
2. **Quadratic mixing overhead** if \(M\) grows with context length (e.g., fixed small chunk size at 128k tokens).

This matters in practice because long-context deployments often prefer small, fixed chunk sizes for stable GPU utilization, streaming, and fine-grained retrieval. A block mixing mechanism whose cost scales **sub-quadratically in the number of blocks** would remove this constraint and make MHLA-like selectivity usable at much longer contexts.

**When does \(M^2\) become a bottleneck?** If we keep a fixed chunk size \(C\) while scaling context length \(N\), then \(M=N/C\) and dense mixing grows as \(M^2 d^2 = (N^2/C^2)d^2\), while the standard linear-attention prefix-summary work is \(N d^2\). The ratio is \(\frac{M^2 d^2}{N d^2}=\frac{N}{C^2}\). For example, at \(N=128\text{k}\) tokens and \(C=64\), \(N/C^2\approx 31\), so dense block mixing becomes the dominant term. MHLA therefore recommends choosing \(M\) such that \(M^2\le N\), which implies \(C\ge\sqrt{N}\) (e.g., \(C\gtrsim 358\) at \(N=128\text{k}\)), but this forces very large blocks and reduces temporal resolution.

This scaling pressure is relevant because MHLA is empirically competitive among subquadratic sequence models. In MHLA’s own language-modeling study (0.3B model trained on 10B FineWeb-Edu tokens), MHLA achieves the best average score on LongBench (7.41 vs 6.92 for Transformer++ and 6.97 for Mamba), suggesting it is worth exploring how to make MHLA-style mixing extrapolate to much longer contexts.

### Key Insight and Hypothesis

MHLA’s coefficient matrix is **content-independent** (learned parameters per block pair), and MHLA reports that a **frozen locality-biased initialization** can be competitive in at least one setting, suggesting that much of the useful structure may depend primarily on **relative block distance** rather than absolute block identity. Concretely, in MHLA’s DeiT-T ImageNet-1K ablation (Table 7a in the arXiv HTML), locality-biased initialization with **frozen** coefficients achieves **75.1%** top-1 accuracy, compared to **75.8%** when those coefficients are learnable (uniform-init + learnable is **75.4%**). This small gap suggests that a distance-shaped prior captures a large fraction of the benefit of learned mixing in at least one domain (vision), motivating a distance-tied parameterization for scaling.

We hypothesize that constraining block mixing to a **translation-invariant causal kernel**

\[ m_{i,b} \propto k_\theta(i-b), \quad b\le i \]

(where \(k_\theta\) is a learnable function of nonnegative distance) will (i) preserve MHLA’s gains over plain linear attention on long-context retrieval tasks, while (ii) enabling efficient computation of \(S_i\) via causal convolution / recurrence with cost \(O(M\,R\,d^2)\) for small kernel rank \(R\), rather than \(O(M^2 d^2)\).

This could fail if MHLA’s learned mixing is not well-approximated by a distance-only kernel (e.g., it relies on absolute position, boundary effects, or non-stationary patterns), in which case Toeplitz tying would reduce expressivity and hurt retrieval.

---

## Proposed Approach

### Overview

We propose **Toeplitz Block Mixing (TBM)**: replace MHLA’s free \(M\times M\) block mixing matrix with a **Toeplitz (distance-tied) causal kernel** over block distance. TBM keeps MHLA’s core idea (each query block uses its own mixed prefix summary) but removes the quadratic dependence on the number of blocks and allows inference with **any number of blocks** (and thus any context length) without learning a new mixing matrix.

### Method Details

**Base attention computation (chunkwise linear attention).** For each token, we compute queries \(Q\), keys \(K\), and values \(V\). We use a positive feature map \(\phi(\cdot)\) to form kernelized linear attention. In chunkwise causal training, the sequence is split into blocks of size \(C\), producing \(M=N/C\) blocks. For each block \(b\), compute a local key-value summary \(S_b = \sum_{t\in b} \phi(K_t) V_t^\top \in\mathbb{R}^{d\times d}\) (and optionally a normalizer \(z_b\)).

**Toeplitz block mixing.** For each block index \(i\), define mixing coefficients over prior blocks by

\[ m_{i,b} = \mathrm{softmax}_{b\le i}(\;k_\theta(i-b)\;). \]

We consider \(k_\theta\) parameterized as a **small mixture of exponentials** (kernel rank \(R\)):

\[ k_\theta(\Delta) = \sum_{r=1}^R a_r \exp(-\lambda_r \Delta), \quad a_r\ge 0, \lambda_r\ge 0. \]

This parameterization (i) supports distance-only tying, (ii) extrapolates naturally to unseen distances, and (iii) admits an efficient recurrent computation of the mixed summary:

- Maintain \(R\) running summaries \(H^{(r)}_i = \exp(-\lambda_r)\,H^{(r)}_{i-1} + S_i\).
- Form \(S^{\mathrm{mix}}_i = \sum_r a_r H^{(r)}_{i-1}\) (causal prefix mixture) and add the intra-block term as in standard chunkwise linear attention.

This reduces mixing from dense \(M\times M\) matrix multiplication to **\(O(R)\)** operations per block, with \(R\ll M\) (e.g., \(R\in\{4,8,16\}\)).

**Go/no-go diagnostic (Toeplitzness of learned MHLA).** When dense MHLA is trainable (small \(M\)), we will fit its learned mixing matrix rows to the TBM kernel family (mixture-of-exponentials) and report explained variance \(R^2\).

- **Stop rule**: if median \(R^2 < 0.6\) across layers (or the best-fit \(R^2\) is consistently low in the later layers), we expect translation-invariant tying to be too restrictive and we will refute/pivot this idea.

Separately, we include a **frozen distance-initialized MHLA** baseline (no learning of mixing) to test whether any learning of block mixing is needed beyond a distance prior.

### Key Innovations

1. **Distance-tied mixing for MHLA**: constrain block mixing to depend only on relative distance, reducing parameters from \(O(M^2)\) to \(O(R)\).
2. **Scalable causal implementation**: compute mixed prefix summaries with a recurrence (mixture-of-exponentials), avoiding \(O(M^2 d^2)\) mixing cost and enabling large \(M\) (small fixed chunk sizes) at long contexts.
3. **Mechanism check**: explicitly measure whether MHLA’s learned mixing is well-approximated by a Toeplitz kernel, providing a clear boundary condition for when TBM should work.

---

## Related Work

### Field Overview

Efficient long-context modeling spans several families: (i) **sparse attention** and sliding-window methods that restrict token-to-token interactions; (ii) **linear attention** models that compute attention via kernel feature maps and prefix summaries; (iii) **SSMs / recurrent models** that parameterize long-range mixing via structured recurrences; and (iv) **hybrid models** that combine softmax attention with subquadratic mixers.

A recurring failure mode is that aggressively subquadratic models lose the ability to retrieve specific information from far back in the context. Recent work has identified distinct causes such as low-rank bottlenecks (“global context collapse”), memory collisions in linear summaries, and component imbalance where hybrid models silently rely on the expensive branch.

### Related Papers

- **[MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head](./references/MHLA-Restoring-Expressivity-of-Linear-Attention-via-Token-Level-Multi-Head/meta/meta_info.txt)**: Introduces blockwise summaries and a learned \(M\times M\) mixing matrix to address global context collapse in linear attention.
- **[Neural Attention Search Linear: Towards Adaptive Token-Level Hybrid Attention Models](./references/Neural-Attention-Search-Linear-Towards-Adaptive-Token-Level-Hybrid-Attention-Models/meta/meta_info.txt)**: Learns to route chunks between softmax and linear attention to balance efficiency and retrieval.
- **[STILL: Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs](./references/STILL-Selecting-Tokens-for-Intra-Layer-Hybrid-Attention-to-Linearize-LLMs/meta/meta_info.txt)**: Uses content-aware token selection and norm-preserving kernels to linearize pretrained LLMs.
- **[LoLA: Low-Rank Linear Attention with Sparse Caching](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt)**: Mitigates memory collisions in linear attention with a self-recall-error-based sparse cache.
- **[Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods](https://arxiv.org/abs/2510.05901)**: Shows hybrid conversions can ignore the linear branch and proposes diagnostics/training fixes.
- **[Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models](https://arxiv.org/abs/2408.10189)**: Distills transformers into Mamba-2 with progressive alignment, highlighting a path to strong subquadratic models.
- **[The Mamba in the Llama: Distilling and Accelerating Hybrid Models](https://arxiv.org/abs/2408.15237)**: Converts transformers to hybrid Mamba models and studies efficiency/performance trade-offs.
- **[Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning](https://arxiv.org/abs/2510.19338)**: Studies hybrid linear/softmax architectures at scale and identifies efficiency bottlenecks.
- **[Mamba](https://arxiv.org/abs/2312.00752)**: Selective SSM for linear-time sequence modeling.
- **[Transformers are SSMs (Mamba-2 / SSD)](https://arxiv.org/abs/2405.21060)**: Provides structured state space duality and efficient algorithms.
- **[RetNet](https://arxiv.org/abs/2307.08621)**: Uses retention with decays as a linear-time alternative to attention.
- **[Gated Linear Attention (GLA)](https://arxiv.org/abs/2312.06635)**: Improves linear attention with gating and hardware-aware design.
- **[Gated Delta Networks (GDN)](https://arxiv.org/abs/2412.06464)**: Uses delta-rule updates to improve recurrent/linear attention style models.
- **[Performer](https://arxiv.org/abs/2009.14794)**: Random-feature approximation to softmax attention for linear attention.
- **[LongBench](https://arxiv.org/abs/2308.14508)**: Benchmark suite for long-context understanding.
- **[RULER](https://arxiv.org/abs/2406.14027)**: Synthetic long-context retrieval suite used to probe recall and interference (e.g., needle-in-haystack).
- **[BABILong](https://arxiv.org/abs/2406.10149)**: Benchmark for million-token reasoning.
- **[Linearized Relative Positional Encoding (LRPE)](https://arxiv.org/abs/2307.09270)**: Shows how to design relative-position structure compatible with linear transformers.
- **[From block-Toeplitz matrices to differential equations on graphs](https://proceedings.mlr.press/v162/choromanski22a.html)**: Connects Toeplitz / structured masks to efficient attention computation.
- **[ALiBi](https://arxiv.org/abs/2108.12409)**: Uses distance-based linear biases for attention extrapolation.
- **[Hyena](https://arxiv.org/abs/2302.10866)**: Long convolutional sequence model with structured mixing.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Dense attention | Quadratic softmax attention | Transformer, FlashAttention | LongBench, downstream tasks | O(N^2) cost |
| Linear attention | Kernel feature map + prefix summaries | Performer, GLA, RetNet | LongBench, RULER | Global summary bottleneck, collisions |
| SSM / recurrent | Structured recurrence / convolution for long mixing | Mamba, Hyena | LongBench, LM perplexity | May underperform on retrieval without extra mechanisms |
| Hybrid attention | Combine softmax with subquadratic mixer | MHLA, NAtS-L, STILL, Ring-linear | RULER, LongBench, LM | Component imbalance; residual quadratic costs |
| Memory augmentation | Add selective full-rank storage | LoLA | RULER | Extra cache management overhead |

### Closest Prior Work

- **MHLA**: Learns an unconstrained \(M\times M\) block mixing matrix to improve linear attention expressivity; does not address the quadratic dependence on \(M\) in block mixing, and NLP code/weights are not publicly released at time of writing.
- **RetNet / exponential-decay linear attention**: Uses distance-based decays inside the recurrence, but does not recover MHLA’s “different mixtures per query block” mechanism.
- **LRPE / Toeplitz masking**: Imposes translation-invariant structure on positional encoding or masks, but does not target MHLA’s block-summary mixing mechanism.
- **NAtS-L / STILL**: Uses content-aware routing between attention types, addressing a different axis (which tokens/chunks get softmax) rather than scaling the block mixing computation.
- **LoLA**: Adds a sparse cache to linear attention to reduce collisions, complementary to (and potentially combinable with) block mixing.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MHLA | Unconstrained learned \(M\times M\) block mixing over summaries | Mixing scales as \(O(M^2 d^2)\); fixed \(M\) tied to training | Tie mixing to distance (Toeplitz) + recurrent computation | Scales to large \(M\) and unseen context lengths |
| RetNet | Distance-decay recurrence for linear-time mixing | Lacks block-specific mixture diversity of MHLA | Add MHLA-style blockwise summaries but with distance-tied weights | More expressive than single global summary, still efficient |
| NAtS-L | Learns chunk routing between softmax and linear attention | Still requires some softmax compute; different target | Keep fully linear attention but improve long-range selectivity | Avoids quadratic softmax tokens while improving recall |
| LoLA | Adds sparse cache using self-recall error | Requires cache scoring/maintenance; starts from a trained hybrid model | Keep simple recurrence kernel; no cache | Lower engineering overhead; complementary to caching |

---

## Experiments

### Experimental Setup

We propose a controlled synthetic long-context retrieval setting where architecture differences dominate, avoiding the need for large-scale pretraining.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Decoder-only LM (linear attention baseline) | ~100–150M params | N/A (trained from scratch) | Simple transformer block with linear attention mixer |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| Synthetic associative recall / passkey retrieval generator | Train models to retrieve keys from long contexts | On-the-fly generated (e.g., 10–50M tokens) | N/A (procedural) | MIT (to be released with code) |

**Resource Estimate**:
- **Compute budget**: 200–600 GPU-hours total (train 2–3 small models + long-context evaluation), within 768 GPU-hours.
- **GPU memory**: ≤ 1×A100 80GB per run (100–150M model fits easily); evaluation at 64k uses larger activation memory but still within 80GB with gradient checkpointing or eval-only.
- **API usage**: None required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| Synthetic associative recall | Procedurally generate long sequences with many random key–value pairs inserted into distractor text; a query at the end asks for the value of a specific key, so success requires retrieving information from far back in the context | Exact-match accuracy (%) | test | N/A (procedural) | Custom (exact string match) |
| RULER | A suite of synthetic long-context retrieval tasks (needle-in-haystack, multi-key/query/value, variable tracking, etc.) used to probe recall under long contexts | Task accuracy (%) | test | https://arxiv.org/abs/2406.14027 (and official repo) | Official (if available) |

### Main Results

**Published baseline evidence (context only; not directly comparable).** The experiments below train small models on synthetic data, so there is no directly comparable published baseline. However, we include the following published numbers to justify that (i) MHLA is competitive among subquadratic sequence models and (ii) long-context retrieval benchmarks can exhibit large gaps for linear attention variants.

| Method | Description | Settings | Results | Reference |
|---|---|---|---|---|
| MHLA | Blockwise linear attention with learned block mixing | 0.3B LM; 10B FineWeb-Edu tokens; context 2048; MHLA mixing heads M=32 | **LongBench avg 7.41** (higher is better) vs Transformer++ 6.92, Mamba 6.97, GDN 6.86, Mamba2 6.62 | MHLA arXiv HTML (Table 8): https://arxiv.org/html/2601.07832v2 |
| MHLA | Same setting; commonsense + MMLU | 0.3B LM; 10B FineWeb-Edu tokens | CSQA 23.7, HellaSwag 47.1, MMLU 38.31 (avg 36.39) | MHLA arXiv HTML (Table 6): https://arxiv.org/html/2601.07832v2 |
| LoLA-8B | Sparse caching added to a hybrid linear-attention model | RULER @4K; LoLA cache (η=128, λ=768); LoLCATs baseline cache (η=896) | **Extended RULER avg 45.2** vs LoLCATs-8B+ 6.7; Mamba2-8B 39.7 | `./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/ASSOCIATIVE RECALL.md` (Table 2) |
| LoLA-8B | Same; single-needle retrieval | RULER S-NIAH-1 @4K; LoLCATs-8B (η=64, λ=0) vs LoLA-8B (η=256, λ=256) | **0.6% → 97.4%** recall accuracy (higher is better) | `./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/ASSOCIATIVE RECALL.md` (Table 1) |

We will report results for two evaluation lengths:
- **In-distribution length**: \(N_{train}\) (e.g., 8k)
- **Extrapolated length**: \(N_{test}\) (e.g., 64k)

#### Results Table (to be verified)

| Method | Base Model | Benchmark | Accuracy @ N_train | Accuracy @ N_test | Source | Notes |
|--------|------------|-----------|--------------------|-------------------|--------|-------|
| Linear attention (no block mixing) | 100–150M | Assoc. recall + RULER | TBD | TBD | - | Baseline |
| Dense MHLA block mixing | 100–150M | Assoc. recall + RULER | TBD | N/A | - | Fixed \(M\); not directly extensible to larger \(M\) without defining new parameters |
| Dense MHLA (frozen distance init) | 100–150M | Assoc. recall + RULER | TBD | N/A | - | Control: tests whether learning the mixing matrix is needed |
| **Ours: Toeplitz Block Mixing (TBM)** | 100–150M | Assoc. recall + RULER | TBD | TBD | - | Distance-tied kernel extrapolates to larger \(M\) |

We will also report **runtime** (tokens/sec) and **peak memory** at \(N_{test}\) for the runnable methods.

**Stop rule (mechanism check):** if dense MHLA does not improve over the linear-attention baseline on the in-distribution length \(N_{train}\), we will treat this as evidence that MHLA-style block mixing is not beneficial in this synthetic setting and refute/pivot rather than over-interpreting TBM results.

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| TBM (R=1) | Single exponential decay | May underfit long-range mixing |
| TBM (R=8) | Default mixture rank | Best accuracy/speed trade-off |
| TBM (fixed kernel) | No learning of \(k_\theta\) (use MHLA distance initialization) | If near TBM, suggests learning is unnecessary |

### Analysis (Optional)

- **Toeplitzness diagnostic**: Fit dense MHLA’s learned mixing to distance-only kernels; report \(R^2\) and how it correlates with TBM performance.
- **Scaling curves**: Measure runtime vs \(M\) (vary chunk size at fixed \(N\)) to show when dense mixing becomes the bottleneck.

---

## Success Criteria

**Criterion 1: TBM preserves MHLA-style gains at train length**
- Hypothesis: At \(N_{train}\), TBM is close to dense MHLA and better than plain linear attention.
- Validation: TBM achieves similar accuracy to dense MHLA and improves over linear attention on associative recall.

**Criterion 2: TBM extrapolates to long contexts with fine chunking**
- Hypothesis: At \(N_{test}\gg N_{train}\), TBM maintains non-trivial retrieval accuracy using the same small chunk size, whereas plain linear attention degrades.
- Validation: TBM improves accuracy over linear attention at \(N_{test}\) and remains computationally feasible.

**Criterion 3: Block mixing becomes scalable**
- Hypothesis: TBM reduces the dependence on the number of blocks from quadratic to near-linear.
- Validation: Runtime/memory scaling curves show TBM remains usable at \(M\) where dense MHLA mixing is impractical.

---

## Impact Statement

If Toeplitz block mixing works, practitioners building long-context models could add MHLA-style selectivity to linear attention using a small, distance-tied kernel that scales to large contexts without quadratic block-mixing overhead. This would make blockwise linear attention more practical for streaming and very-long-context workloads where fixed small chunk sizes are preferred.

---

## References

- [MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head](./references/MHLA-Restoring-Expressivity-of-Linear-Attention-via-Token-Level-Multi-Head/meta/meta_info.txt) - Zhang et al., 2026
- [Neural Attention Search Linear: Towards Adaptive Token-Level Hybrid Attention Models](./references/Neural-Attention-Search-Linear-Towards-Adaptive-Token-Level-Hybrid-Attention-Models/meta/meta_info.txt) - Deng et al., 2026
- [STILL: Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs](./references/STILL-Selecting-Tokens-for-Intra-Layer-Hybrid-Attention-to-Linearize-LLMs/meta/meta_info.txt) - Meng et al., 2026
- [LoLA: Low-Rank Linear Attention with Sparse Caching](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt) - (authors in OpenReview), 2025
- [Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods](https://arxiv.org/abs/2510.05901) - Benfeghoul et al., 2024
- [Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models](https://arxiv.org/abs/2408.10189) - Bick et al., 2024
- [The Mamba in the Llama: Distilling and Accelerating Hybrid Models](https://arxiv.org/abs/2408.15237) - Wang et al., 2024
- [Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning](https://arxiv.org/abs/2510.19338) - Ling Team, 2024
- [Mamba](https://arxiv.org/abs/2312.00752) - Gu and Dao, 2023
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) - Dao and Gu, 2024
- [Retentive Network (RetNet)](https://arxiv.org/abs/2307.08621) - Sun et al., 2023
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) - Choromanski et al., 2020
- [Gated Linear Attention Transformers](https://arxiv.org/abs/2312.06635) - Yang et al., 2023
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) - Yang et al., 2024
- [LongBench](https://arxiv.org/abs/2308.14508) - Bai et al., 2023
- [RULER](https://arxiv.org/abs/2406.14027) - (RULER authors), 2024
- [BABILong](https://arxiv.org/abs/2406.10149) - (BABILong authors), 2024
- [Linearized Relative Positional Encoding](https://arxiv.org/abs/2307.09270) - Qin et al., 2023
- [From block-Toeplitz matrices to differential equations on graphs](https://proceedings.mlr.press/v162/choromanski22a.html) - Choromanski et al., 2022
- [Train Short, Test Long: Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409) - Press et al., 2021
- [Hyena Hierarchy](https://arxiv.org/abs/2302.10866) - Poli et al., 2023
