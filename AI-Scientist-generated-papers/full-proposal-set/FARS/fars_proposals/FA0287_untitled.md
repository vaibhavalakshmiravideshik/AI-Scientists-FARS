# untitled

# SinkCast: FP32 Recasting of Attention-Sink Logits to Restore BF16 RoPE Shift-Invariance

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Most modern long-context LLMs use **RoPE (rotary position embeddings)**, which rotates attention queries/keys as a function of token position. In exact arithmetic, RoPE’s query–key dot product depends only on **relative position**, implying **shift-invariance**: globally shifting all `position_ids` by a constant should not change attention (and thus should not change model outputs).

In practice, long-context inference is usually run in **BF16 (bfloat16; a 16-bit floating-point format)** with kernels such as **FlashAttention (an exact, memory-efficient attention kernel that uses online log-sum-exp statistics)**.

Separately, long-context **serving systems** increasingly rely on **position-independent context caching / KV-cache reuse** to reduce time-to-first-token by reusing precomputed KV states of static chunks at different offsets (e.g., RAG documents, shared system prompts). These systems often *re-position* cached chunks (by changing `position_ids` or by re-applying RoPE rotations), which implicitly assumes RoPE behaves like a stable relative encoding.

### The Problem

**Wang et al. (2024)** show that RoPE’s shift-invariance breaks under BF16 in pretrained models, while FP32 largely removes the discrepancy ([When Precision Meets Position](https://arxiv.org/abs/2411.13476), Sec. 2.2). They also find that the **first key token** accounts for most of the attention difference under shifts, and that this discrepancy grows with context length.

This is not only a synthetic invariance curiosity: many cache re-positioning and position-independent caching methods (e.g., **CacheBlend**, **EPIC**, **KVShare**, **MEPIC**, **CacheFocus**) explicitly move cached context across offsets and/or re-encode RoPE. If BF16 breaks RoPE shift-invariance, then a system can suffer *accuracy regressions even when token content is unchanged*.

A brute-force mitigation is FP32 attention, but this is too slow for long contexts. Wang et al.’s **AnchorAttention** is primarily a *training-time* attention layout for continued long-context pretraining (Sec. 3) and is not a generic inference patch for an existing checkpoint.

### Key Insight and Hypothesis

**Key insight.** Many RoPE implementations compute the RoPE rotation in FP32 outside the attention kernel, then cast to BF16 for FlashAttention’s dot products. If most shift-error is concentrated in a small set of logits (especially the sink key), we can recompute only those logits in FP32 and then adjust the softmax normalization to obtain the exact attention output under the “only these logits changed” assumption.

**Hypothesis.** For RoPE-based LLMs run in BF16, **SinkCast**—recomputing only the sink-key logit(s) in FP32 and applying an exact output correction using FlashAttention’s row-wise `softmax_lse`—recovers most of FP32 shift-invariance at negligible overhead.

The main way this fails is if shift-error is not localized to the sink key (or a very small top-K).

---

## Proposed Approach

### Overview

**SinkCast** is an inference-time correction that keeps the BF16 FlashAttention fast path:

1) Run BF16 FlashAttention to obtain attention output `O` and row-wise log-normalizers `lse`.
2) Recompute only sink logits (e.g., key index `j=0`, optionally `j< K`) in FP32.
3) Use a closed-form update to produce the exact corrected output `O'` without recomputing full attention.

### Method Details

For one head and one query row `i`, let baseline logits be `a_ij` and `lse_i = logsumexp_j a_ij` (returned by FlashAttention in FP32). Let the sink be `j=0`.

Compute:
- `p_i0 = exp(a_i0 - lse_i)`
- `logZ_minus = lse_i + log1p(-p_i0)`
- `a'_i0` = FP32-recomputed sink logit (dot product of FP32 RoPE-rotated `q_i` and `k_0`)
- `lse'_i = logaddexp(logZ_minus, a'_i0)`
- `p'_i0 = exp(a'_i0 - lse'_i)`
- `scale_i = exp(lse_i - lse'_i)`

Let baseline output be `o_i = Σ_j softmax(a_i·)_j v_j` (from FlashAttention) and sink value vector be `v_0`. The exact corrected output after changing only `a_i0` is:

`o'_i = scale_i * (o_i - p_i0 * v_0) + p'_i0 * v_0`.

This extends to K sink keys by replacing scalar terms with sums over the sink set.

**Availability of `softmax_lse`.** FlashAttention exposes row-wise `softmax_lse` via `(out, softmax_lse)` / `return_softmax_lse` in its Python interface (and similar LSE-like auxiliary outputs exist in PyTorch attention backends).

### Key Innovations

1) **Selective FP32 recomputation for RoPE shift-invariance** (sink logits only).
2) **Exact post-hoc correction** using `softmax_lse` (no approximation, no custom CUDA).
3) **Inference-only deployability** for already-trained RoPE models.

---

## Related Work

### Field Overview

RoPE dominates open long-context LLMs, so BF16 numerical shift error (if real) impacts a broad family of deployments. In parallel, position-independent caching and cache re-positioning methods are increasingly used to reduce prefill latency, and they rely on moving cached chunks across offsets.

### Related Papers

- **[When Precision Meets Position](https://arxiv.org/abs/2411.13476)**: Shows BF16 breaks RoPE shift-invariance; proposes AnchorAttention for long-context continued pretraining.
- **[RoFormer](https://arxiv.org/abs/2104.09864)**: Introduces RoPE.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)**: Exact attention with online softmax.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**: Faster exact attention; exposes row-wise softmax statistics.
- **[Is Flash Attention Stable?](https://arxiv.org/abs/2405.02803)**: Measures numeric deviation of FlashAttention.
- **[StreamingLLM](https://arxiv.org/abs/2309.17453)**: Identifies attention sinks and uses them for streaming KV-cache design.
- **[CacheBlend](https://arxiv.org/abs/2405.16444)**: Position-independent caching via selective recomputation; discusses RoPE re-alignment for reused chunks.
- **[EPIC](https://arxiv.org/abs/2410.15332)**: PIC with static boundary recomputation and attention-sink mitigation.
- **[KVShare](https://arxiv.org/abs/2503.16525)**: Multi-tenant KV sharing with partial recomputation; compares against PIC baselines.
- **[MEPIC](https://arxiv.org/abs/2512.16822)**: PIC via storing KV without RoPE and applying RoPE inside attention.
- **[CacheFocus](https://arxiv.org/abs/2502.11101)**: Cache re-positioning for RoPE-based retrieval/caching pipelines.
- **[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)**: Widely used serving backend where BF16+RoPE is common.
- **[Delta Attention](https://arxiv.org/abs/2505.11254)**: Corrective computation after a fast attention pass (different error source).
- **[Position Interpolation](https://arxiv.org/abs/2306.15595)**: RoPE scaling to extend context length.
- **[YaRN](https://arxiv.org/abs/2309.00071)**: Improved RoPE scaling/interpolation.
- **[LongRoPE](https://arxiv.org/abs/2402.13753)**: Extends RoPE to very long contexts.
- **[Why Does the Effective Context Length Fall Short?](https://arxiv.org/abs/2402.00265)**: Argues long-range regimes are under-trained; proposes inference-time shifting.
- **[RULER](https://arxiv.org/abs/2404.06654)**: Synthetic long-context benchmark.
- **[LongBench](https://arxiv.org/abs/2308.14508)**: Real-world long-context benchmark suite.

### Taxonomy

| Cluster | Core idea | Representative papers | Main limitation |
|---|---|---|---|
| Training-time layout changes | Change attention visibility for long-context training | AnchorAttention | Requires training |
| RoPE scaling/remapping | Change RoPE frequencies or remap positions | PI, YaRN, LongRoPE | Does not target BF16 dot-product error |
| PIC / cache re-positioning | Reuse cached KV across offsets | CacheBlend, EPIC, KVShare, MEPIC, CacheFocus | Assumes RoPE is “relative enough” |
| Correction-after-fast-pass | Correct errors from approximate attention | Delta Attention | Different failure mode |

### Closest Prior Work

- **Wang et al. (2024)**: Identifies BF16 RoPE shift-invariance breakdown; proposes a training-time fix (AnchorAttention). SinkCast targets **inference-time numeric correction** for frozen models.
- **PIC systems**: Demonstrate that re-positioning chunks is practically important for serving, but do not explicitly address BF16 RoPE shift-invariance error.
- **FlashAttention-2**: Exposes `softmax_lse`; SinkCast uses it for **exact output correction after selective logit recomputation**.

**Novelty Kill Search Summary:** Searched for “selective FP32 attention logits”, “softmax_lse logit recomputation”, and “RoPE bf16 selective casting” (full log in `notes.md`). No prior work was found that uses `softmax_lse` to do exact inference-time correction after recomputing only a subset of logits in FP32 (as of 2026-02-25).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| AnchorAttention | Training-time anchor token/layout | Needs training | Inference-only numeric correction | Deployable for frozen models |
| PIC methods | Reuse KV across offsets | Doesn’t fix BF16 shift error | Patch attention numerics | Improves robustness without changing caching design |
| FlashAttention-2 | Exact attention + LSE stats | Not a RoPE fix | Use LSE for logit surgery | Minimal overhead, no CUDA |
| Delta Attention | Correct sparse/approx attention | Different error source | Exact correction for changed logits | Exactness for precision repair |

---

## Experiments

### Experimental Setup

**Baselines.** The core phenomenon compares the *same token sequence* under different `position_ids`, so prompting/best-of-*N* are not expected to eliminate shift-induced differences. We use:

1. **BF16 FlashAttention (baseline)**.
2. **FP32 attention oracle** (short lengths only; confirm near-zero drift in FP32).
3. **SinkCast (K=1)** (main method).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3.1-8B (or Instruct) | 8B | https://huggingface.co/meta-llama | RoPE-based; common serving target |
| Mistral-7B (v0.3) | 7B | https://huggingface.co/mistralai | Second RoPE family (generalization) |

**Training Data:** No training — inference-only.

**Resource Estimate:** ≤150 GPU-hours total.

### Benchmarks and Metrics

| Benchmark | What it evaluates | Metrics |
|---|---|---|
| Shift microbenchmark | Sensitivity to global `position_id` shifts with identical tokens | (i) `D_logit(j)`; (ii) output-logit drift |
| RULER-position-shift | Whether shift sensitivity causes measurable accuracy drops | Accuracy drop under shift |
| LongBench-position-shift (subset) | Whether shift sensitivity affects real tasks | Accuracy drop under shift |

**Define `D_logit(j)`.** For shifts Δ1,Δ2 and length T:
`D_logit(j) = (1/T) * Σ_{layers,heads} Σ_{i=1..T} |A^{l,h}_{i,j}(Δ1) − A^{l,h}_{i,j}(Δ2)|`, where `A` is the pre-softmax attention logit.

**Shift protocol.** Evaluate each input twice with identical tokens/attention masks but different global offsets:
- Microbenchmark: Δ2=16 (as in Wang et al.); vary Δ1∈{0,256,4096}.
- Downstream: prepend `M` masked tokens (attention mask 0) and explicitly set `position_ids` so real tokens start at `M` (e.g., M=4096).

### Main Results

| Method | Benchmark | Metric | Value (mean±std) | Notes |
|---|---|---|---:|---|
| BF16 FlashAttention | Shift microbench | output-logit drift | **TBD** | To be measured |
| FP32 attention | Shift microbench | output-logit drift | **TBD** | Short `T` only |
| **SinkCast (K=1)** | Shift microbench | output-logit drift | **TBD** | Main method |
| BF16 FlashAttention | RULER-position-shift | accuracy drop | **TBD** | To be measured |
| **SinkCast (K=1)** | RULER-position-shift | accuracy drop | **TBD** | Expect smaller drop |
| BF16 FlashAttention | LongBench-position-shift | accuracy drop | **TBD** | To be measured |
| **SinkCast (K=1)** | LongBench-position-shift | accuracy drop | **TBD** | Expect smaller drop |

### Ablation Studies

| Variant | What changes | Expected outcome |
|---|---|---|
| SinkCast (K=4) | Recompute first 4 keys | Helps if error is less localized |
| No-correction control | Recompute sink logit but skip output correction | Should not improve |
| BOS-only position clamp | Keep BOS at pos 0, shift others by M | Tests if a simpler “anchor-only” hack suffices |

### Experimental Rigor

- **Determinism / seeds**: Use dropout=0 and greedy decoding.
- **Sanity check**: FP32 attention oracle should show near-zero drift at short lengths.
- **Confounders**: always pass explicit `position_ids`; verify masked-prefix tokens are not attended as keys/values.

---

## Success Criteria

**Hypothesis** (directional): SinkCast (K=1) substantially reduces shift-induced output drift and reduces accuracy drop under `position_id` shifts.

**Decision Rule** (concrete):

1. **Premise check (localization)**: compute `D_logit(j)` for `j ∈ {0,1,2,8,64}`.
   - **Refute K=1** if `D_logit(0)` is <50% of `∑_j D_logit(j)` over this set.

2. **Microbenchmark win**:
   - **Proceed** if SinkCast closes ≥80% of the BF16→FP32 gap in output-logit drift (where FP32 is feasible) on both base models.
   - **Refute** if SinkCast closes <30% of the gap on both base models.

3. **Downstream signal (RULER/LongBench)**:
   - **Proceed** if SinkCast reduces the shift-induced accuracy drop by ≥2 points on at least one benchmark at similar wall-clock cost.
   - **Refute** if the accuracy drop is unchanged.

---

## Impact Statement

If SinkCast works, engineers building **long-context serving stacks** (especially position-independent caching / cache re-positioning systems) gain a lightweight alternative to full FP32 attention or retraining-based fixes for improving robustness under chunk re-positioning and large position offsets in BF16 RoPE models.

---

## References

- [When Precision Meets Position](https://arxiv.org/abs/2411.13476) - Wang et al., 2024
- [CacheBlend](https://arxiv.org/abs/2405.16444) - Yao et al., 2024
- [EPIC](https://arxiv.org/abs/2410.15332) - Hu et al., 2024
- [KVShare](https://arxiv.org/abs/2503.16525) - Yang et al., 2025
- [MEPIC](https://arxiv.org/abs/2512.16822) - 2025
- [CacheFocus](https://arxiv.org/abs/2502.11101) - 2025
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao, 2023
- [RoFormer](https://arxiv.org/abs/2104.09864) - Su et al., 2021
- [RULER](https://arxiv.org/abs/2404.06654) - Hsieh et al., 2024
- [LongBench](https://arxiv.org/abs/2308.14508) - Bai et al., 2023
