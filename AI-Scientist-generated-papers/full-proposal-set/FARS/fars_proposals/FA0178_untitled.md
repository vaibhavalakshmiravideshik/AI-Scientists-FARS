# untitled

# Prompt-Isolated Attention for Exact Prompt KV Caching in Diffusion LLMs

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Discrete diffusion language models (dLLMs) generate text by iteratively denoising a fully masked sequence, rather than decoding tokens left-to-right as in autoregressive (AR) LLMs. This can enable parallel token updates and bidirectional context, which is useful for tasks like constrained generation and text infilling.

However, practical dLLM deployment is still bottlenecked by inference cost. Each denoising step runs a transformer forward pass over the *entire* sequence (prompt + partially denoised answer), which repeatedly recomputes attention keys/values (K/V) for the prompt. In contrast, AR decoding supports a KV cache that makes repeated prompt computation unnecessary.

### The Problem

In dLLMs like LLaDA, the prompt representations are typically recomputed at every denoising step because the model uses bidirectional attention over the full sequence.

Recent training-free accelerations reduce this cost by *approximating* the KV cache dynamics:
- **[dLLM-Cache](./references/dLLM-Cache-Accelerating-Diffusion-Large-Language-Models-with-Adaptive-Caching/meta/meta_info.txt)** refreshes prompt and response K/V states on a fixed schedule, achieving 4–5× TPS improvements on several LLaDA benchmarks (e.g., GSM8K TPS 6.95→29.75 with cache refresh hyperparameters \(K_p=50\) (prompt refresh interval) and \(K_r=7\) (response refresh interval); see the paper’s experimental settings).
- **[d2Cache](./references/d2Cache-Accelerating-Diffusion-Based-LLMs-via-Dual-Adaptive-Caching/meta/meta_info.txt)** uses token-level selection for cache updates, improving Dream-Inst GSM8K throughput 2.62→12.25 t/s.
- **[Elastic-Cache](./references/ATTENTION-IS-ALL-YOU-NEED-FOR-KV-CACHE-IN-DIFFUSION-LLMS/meta/meta_info.txt)** uses an attention-drift trigger and depth-selective refresh, reporting up to 45.1× speedup.
- **[MaskKV](./references/Mask-Tokens-as-Prophet-Fine-Grained-Cache-Eviction-for-Efficient-dLLM-Inference/meta/meta_info.txt)** evicts most prompt KV states for long-context inference and reports 31× faster decoding at 32K context length.

These methods are effective, but they do not provide an *exactness guarantee*: their cached prompt states can (and often must) drift across denoising steps because prompt tokens attend to the evolving answer tokens.

### Key Insight and Hypothesis

**Key insight**: If prompt tokens are prevented from attending to answer tokens, then prompt tokens form a closed attention subgraph. In that case, prompt hidden states (and thus prompt K/V states) should be *step-invariant* across denoising steps, unless the model injects timestep/noise-level conditioning globally.

This is exactly the idea behind **Prefix-DLM**, an inference-time attention mask introduced for a diffusion VLM:
- **[LaViDa](./references/LaViDa-A-Large-Diffusion-Language-Model-for-Multimodal-Understanding/meta/meta_info.txt)** uses a “Prefix-DLM” mask where visual+prompt tokens attend only to visual+prompt tokens while answer tokens attend to all tokens, enabling prompt KV caching and reducing COCO captioning latency 7.65s→1.93s (3.9×) with a small CIDEr drop (121.0→117.3).

**Hypothesis**: Applying LaViDa-style Prefix-DLM masking to language-only diffusion LLMs (LLaDA/Dream) makes prompt K/V *numerically invariant* across denoising steps, enabling **exact prompt KV reuse** and yielding the largest speedups on prompt-dominant workloads (long prompt, short answer).

Why this could be wrong:
- The model may apply timestep conditioning (explicit time embedding, or global time-dependent modulation) that changes prompt states even without prompt→answer attention.
- Even if exact caching is valid, the masking change could impose a real quality cost because prompt tokens can no longer “update” based on partially denoised answer tokens.

---

## Proposed Approach

### Overview

We propose an inference-time modification to text diffusion LLM decoding:

1. Replace full bidirectional attention with a **prompt-isolated (Prefix-DLM) attention mask**.
2. Use this mask to enable **exact per-layer prompt KV caching across denoising steps** within a single request.

The scientific goal is to determine whether exact prompt KV reuse is *fundamentally possible* for LLaDA-style dLLMs without retraining, and to quantify the speed/quality trade-off if it is.

### Method Details

**Setting**: We generate an answer sequence of length R conditioned on a fixed prompt of length P. At denoising step i, the model input is `[prompt tokens p0] + [noised answer tokens r_ti]`.

**Prefix-DLM attention mask** (same structure as LaViDa):
- For query positions in the prompt region: keys/values are visible **only** in the prompt region.
- For query positions in the answer region: keys/values are visible in **both** prompt and answer regions.

This removes the only obvious graph path by which the changing answer tokens can influence prompt representations.

**Exact prompt KV caching**:
- Run a single “prompt-only” forward (or a full forward at the first denoising step) under the Prefix-DLM mask to obtain, for each transformer layer \(\ell\), cached prompt keys/values \((K^{\ell}_{p}, V^{\ell}_{p})\).
- For each subsequent denoising step, compute only the answer-token \((Q^{\ell}_{a}, K^{\ell}_{a}, V^{\ell}_{a})\), concatenate cached prompt \((K^{\ell}_{p}, V^{\ell}_{p})\) with current answer \((K^{\ell}_{a}, V^{\ell}_{a})\), and perform attention for answer queries \(Q^{\ell}_{a}\) over the combined K/V.

**Compute-bound upper bound** (roofline): with P prompt tokens, R answer tokens, and K denoising steps,
- Vanilla full attention costs \(\approx K\,(P+R)^2\).
- With exact prompt caching, cost is \(\approx (P+R)^2 + (K-1)\,R\,(P+R)\).
- Idealized speedup upper bound: \(S \approx \frac{K(P+R)}{P + K R}\).

**Memory-bandwidth caveat**: for very long prompts, attention can become bandwidth-bound because cached prompt K/V must still be read each step. Therefore this roofline is an upper bound; empirical throughput scaling with P is part of the evaluation.

### Key Innovations

- **Exactness-first framing for dLLM caching**: Instead of designing another approximate refresh policy, we test whether an inference-time attention mask can make prompt states *provably independent* of the denoising process.
- **Mechanistic gate experiment**: We pre-register a numerical invariance test on prompt K/V tensors to decide whether exact caching is possible at all.
- **Generalization of Prefix-DLM from multimodal to text-only diffusion LMs**: LaViDa shows the mask works in a diffusion VLM; it is unknown if the same mask yields step-invariant prompt states (and acceptable quality) for language-only dLLMs.

---

## Related Work

### Field Overview

Diffusion LLM inference acceleration has rapidly progressed from “no KV cache is possible” to a spectrum of training-free approximations that exploit stability patterns across denoising steps. Most existing methods treat step-to-step KV drift as inevitable under full bidirectional attention and focus on (i) deciding **which tokens** to refresh/cache, (ii) deciding **which layers** to refresh, or (iii) modifying the model to restore causal/prefix-cacheable attention.

Our proposal is in a different bucket: modify the *attention graph* so that prompt representations become step-invariant, enabling exact caching in the prompt region.

### Related Papers

- **[Large Language Diffusion Models (LLaDA)](./references/Large-Language-Diffusion-Models/meta/meta_info.txt)**: Introduces LLaDA, a large masked diffusion LM with bidirectional attention and iterative remasking inference.
- **[Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)**: Trains a diffusion LM initialized from an AR LM and reports strong reasoning performance.
- **[LaViDa](./references/LaViDa-A-Large-Diffusion-Language-Model-for-Multimodal-Understanding/meta/meta_info.txt)**: Diffusion VLM that introduces Prefix-DLM masking to enable prompt KV caching at inference.
- **[dLLM-Cache](./references/dLLM-Cache-Accelerating-Diffusion-Large-Language-Models-with-Adaptive-Caching/meta/meta_info.txt)**: Periodically refreshes prompt/response caches under full attention for training-free acceleration.
- **[d2Cache](./references/d2Cache-Accelerating-Diffusion-Based-LLMs-via-Dual-Adaptive-Caching/meta/meta_info.txt)**: Dual adaptive caching with token-level selection to reduce recomputation.
- **[Attention Is All You Need for KV Cache in Diffusion LLMs (Elastic-Cache)](./references/ATTENTION-IS-ALL-YOU-NEED-FOR-KV-CACHE-IN-DIFFUSION-LLMS/meta/meta_info.txt)**: Uses attention-drift triggers + depth-selective refresh to achieve very large speedups.
- **[Mask Tokens as Prophet (MaskKV)](./references/Mask-Tokens-as-Prophet-Fine-Grained-Cache-Eviction-for-Efficient-dLLM-Inference/meta/meta_info.txt)**: Uses mask-query attention to evict most prompt KV states for long-context inference.
- **[Fast-dLLM](https://arxiv.org/abs/2505.22618)**: Enables approximate KV caching and confidence-aware parallel decoding for diffusion LMs.
- **[dKV-Cache](https://arxiv.org/abs/2505.15781)**: Proposes delayed caching for diffusion LMs based on token state dynamics.
- **[FlashDLM / FreeCache](https://arxiv.org/abs/2505.21467)**: Uses block-wise approximate caching for “clean” tokens plus AR-guided unmasking.
- **[Sparse-dLLM](https://arxiv.org/abs/2508.02558)**: Dynamic cache eviction and sparse attention guided by stable token saliency.
- **[Block Diffusion](https://arxiv.org/abs/2503.09573)**: Interpolates between AR and diffusion via blockwise diffusion that is more cache-friendly.
- **[WeDLM](https://arxiv.org/abs/2512.22737)**: Reconciles diffusion decoding with standard causal attention to restore native KV caching.
- **[CARD](https://arxiv.org/abs/2601.22031)**: Causal autoregressive diffusion for language modeling.
- **[DSB](https://arxiv.org/abs/2602.05992)**: Batch-level scheduling optimization for diffusion LLM inference.
- **[Scaling Up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514)**: Earlier scaling work for masked diffusion models in language.
- **[Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://arxiv.org/abs/2406.03736)**: Theoretical perspective on absorbing discrete diffusion objectives.
- **[MaskGIT](https://arxiv.org/abs/2202.04200)**: Discrete masked generation with iterative refinement (foundational for masked diffusion-style inference).
- **[SqueezeAttention](https://arxiv.org/abs/2404.04793)**: Layer-wise KV cache budget management for transformer inference (AR setting, but relevant to KV budget allocation).
- **[AdaKV](https://arxiv.org/abs/2407.11550)**: Adaptive KV cache eviction via budget allocation (AR setting; used as baseline family in MaskKV).

(We cite AR KV-cache eviction papers because several dLLM methods adapt their intuitions/metrics, and they are often included as baselines in long-context cache studies.)

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Approximate refresh (token-level) | Update only a selected subset of tokens’ KV states per step | d2Cache, dKV-Cache, Fast-dLLM, FlashDLM | GSM8K, HumanEval, MBPP, GPQA | Cache drift remains; exactness not guaranteed |
| Approximate refresh (layer-/attention-level) | Trigger refresh based on attention drift and refresh deeper layers more | Elastic-Cache | GSM8K, HumanEval | Complexity; still approximate |
| Cache eviction / compression for long context | Keep only a small budget of prompt KV states | MaskKV, Sparse-dLLM | LongBench, long-context microbenchmarks | May remove useful prompt info; needs good importance signals |
| Attention-graph modification (this work) | Modify attention mask so prompts are isolated and cacheable | LaViDa (multimodal), **this proposal** | LongBench-style long-prompt tasks + KV drift test | Potential quality shift due to masking |
| Architectural change to causal attention | Make diffusion compatible with standard causal attention and KV cache | WeDLM, CARD | Reasoning and generation benchmarks | Requires (re)training; may lose some bidirectional advantages |

### Closest Prior Work

1. **LaViDa / Prefix-DLM**: Demonstrates prompt-isolated attention masking to enable caching of multimodal prompts (image + text) during diffusion decoding, with up to 3.9× latency reduction on COCO captioning. Limitation: not evaluated for language-only dLLMs, long-context tasks, or exact prompt KV invariance across denoising steps.
2. **dLLM-Cache / d2Cache / Elastic-Cache**: All accelerate diffusion LMs without changing the attention mask, by approximating cache updates under full bidirectional attention. Limitation: no exactness guarantee for prompt caching; prompt KV states can drift due to prompt attending to changing answer tokens.
3. **MaskKV**: Shows extreme KV compression for long contexts by evicting most prompt KV entries, but it is explicitly approximate and focuses on memory reduction rather than exact prompt reuse.

**Novelty Kill Search Summary:** Searched for “Prefix-DLM KV cache diffusion language model”, “prompt tokens attend only to prompt tokens diffusion LLM cache”, “LLaDA prefix attention mask KV cache”, and related OpenReview/GitHub queries. No prior language-only diffusion LLM work proposing LaViDa-style prompt-isolated masking for exact prompt KV reuse was found as of 2026-02-20. (Full query log is in `notes.md`.)

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LaViDa (Prefix-DLM) | Prompt-isolated mask for diffusion VLM inference to cache multimodal prompts | Not tested for text-only dLLMs; exact invariance unverified | Apply same mask to text dLLMs + measure KV invariance | Could enable exact prompt caching for LLaDA-style models |
| dLLM-Cache / d2Cache | Approximate token-level caching under full attention | Prompt KV can drift; needs refresh schedule/selection | Make prompt KV step-invariant by construction | Removes need for prompt refresh in prompt-dominant regime |
| Elastic-Cache | Attention-aware + depth-aware cache updates under full attention | Still approximate; complex trigger/schedule | Deterministic exact prompt caching via mask | Potentially simpler and more predictable; complementary to other caches |
| MaskKV | Evict most prompt KV states for long contexts | Approximate + may drop info | Keep full prompt KV but compute once | Exact reuse without eviction-induced information loss |

---

## Experiments

### Experimental Setup

**Implementation target**: HuggingFace `transformers` implementation of LLaDA with `trust_remote_code=True`.

**Prefix-DLM mask implementation**:
- Build a block attention mask for each layer so prompt queries cannot attend to answer keys.
- Ensure answer queries can attend to both prompt and answer keys (full bidirectional within the answer region).

**Prompt KV caching implementation**:
- Add a “prompt KV prefill” stage that runs once per request and stores \((K_p^\ell, V_p^\ell)\) for all layers.
- Modify the attention forward to accept cached prompt K/V and compute only answer K/V each denoising step.

**Baseline Ladder (REQUIRED):**
- **Vanilla (no caching)**: Full bidirectional attention, recompute full sequence each step.
- **Strong diffusion caching baseline**: **Elastic-Cache** (attention-aware drift trigger + depth-selective refresh). If engineering cost is too high, fall back to **d2Cache** as the approximate caching baseline.
- **Step-reduction baseline (efficiency-only)**: reduce number of denoising steps K (lower NFE, “number of function evaluations”, i.e., transformer forward passes) for vanilla LLaDA to match throughput and measure quality drop (optional analysis; not required for the main decision rule).

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| LLaDA-8B-Instruct | 8B | https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct | MIT license; `trust_remote_code=True` |
| (Optional) LLaDA-1.5 | 8B | https://huggingface.co/GSAI-ML/LLaDA-1.5 | Stronger aligned model; useful to test generality |

**Training Data (if applicable):**
- No training data needed — inference-only modification.

**Other Resources (if applicable):**
- None.

**Resource Estimate**:
- **Compute budget**: 
  - Exp 0 (KV drift test): <1 GPU-hour.
  - Exp 1 (LongBench subset, ~200 examples, 3 methods): expected to fit within ~50–150 GPU-hours on 1×A100 80GB (wall-clock dominated by diffusion steps). Exact cost depends on K, max_gen_len, and prompt length.
- **GPU memory**: 1×A100 80GB should be sufficient for LLaDA-8B with long contexts; may need tensor parallelism for very long prompts.
- **API usage**: None.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| LongBench (v1) | Long-context multitask benchmark (QA, summarization, few-shot, synthetic, code); many tasks have 5k–15k context | Task-specific metrics (F1, ROUGE-L, accuracy, edit similarity) + macro-average | test | https://huggingface.co/datasets/THUDM/LongBench | https://github.com/THUDM/LongBench |
| KV Drift Microbench (new) | Measures whether prompt KV tensors change across denoising steps under Prefix-DLM | Mean cosine similarity + relative L2 error per layer (prompt K/V) | N/A | N/A | Custom script (simple tensor comparison) |

**Evaluation Scripts:**
- Use the official LongBench evaluation pipeline from THUDM/LongBench.

**Download Links Checklist:**
- [ ] LongBench dataset link provided
- [ ] LongBench evaluation repo provided
- [ ] Base model download links provided

### Main Results

We report two tables:
1) KV invariance (the mechanistic gate).
2) End-to-end throughput vs. LongBench score on a prompt-dominant subset.

#### Results Table

| Method | Base Model | Benchmark | KV Drift (mean cosine / rel-L2) | Throughput (tok/s) | Source | Notes |
|--------|------------|-----------|----------------------------------|--------------------|--------|-------|
| Full attention | LLaDA-8B-Instruct | KV Drift Microbench | **TBD** | **TBD** | - | Baseline; expected drift > 0 |
| Prefix-DLM (no cache) | LLaDA-8B-Instruct | KV Drift Microbench | **TBD** | **TBD** | - | Should show whether exact caching is possible |
| Prefix-DLM + exact prompt KV cache (ours) | LLaDA-8B-Instruct | KV Drift Microbench | **TBD** | **TBD** | - | Should match Prefix-DLM (no cache) exactly |

| Method | Base Model | Benchmark | LongBench avg score (mean±std) | Throughput (tok/s) | Source | Notes |
|--------|------------|-----------|--------------------------------|--------------------|--------|-------|
| Vanilla (full attention) | LLaDA-8B-Instruct | LongBench subset | **TBD** | **TBD** | - | To be verified |
| Strong cache baseline (Elastic-Cache; fallback: d2Cache) | LLaDA-8B-Instruct | LongBench subset | **TBD** | **TBD** | - | Needs re-run in this setting |
| **Ours: Prefix-DLM + exact prompt KV cache** | LLaDA-8B-Instruct | LongBench subset | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Prefix-DLM (no cache) vs Prefix-DLM (cached) | Cache on/off under same mask | Identical outputs + identical scores (verifies exactness of caching implementation) |
| Prompt length sweep | Evaluate P in {1k, 4k, 8k, 16k} (truncate/pad) | Speedup increases with P; may saturate if bandwidth-bound |

### Experimental Rigor

**Variance & Reproducibility:**
- Use 3 random seeds for denoising randomness (e.g., `seeds=[0,1,2]`) for LongBench subset; report mean±std.
- For KV drift test, determinism is expected given fixed seed and identical inputs.

**Validity & Controls:**
- Control for diffusion step count K and generation length R across methods.
- Sanity check: cached Prefix-DLM must match uncached Prefix-DLM bitwise or within numerical tolerance.
- Step-0 edge case: explicitly test that using step-0 prompt cache (answer all [MASK]) is not a degenerate cache that changes outputs vs caching from an intermediate step.

---

## Success Criteria

**Hypothesis** (directional — what you expect):
- Under Prefix-DLM masking, prompt K/V tensors will be numerically invariant across denoising steps in LLaDA-8B (suggesting no global timestep conditioning on prompt tokens).
- Exact prompt caching will substantially increase throughput on prompt-dominant LongBench tasks, with small or negligible quality loss.

**Decision Rule** (concrete — when to stop):
- **Proceed (exact caching is possible)**: Prompt KV drift under Prefix-DLM is essentially zero (e.g., mean cosine ≥ 0.999999 and relative L2 ≤ 1e-5 across layers).
- **Refute (exact caching impossible without retraining)**: Prompt KV drift under Prefix-DLM is non-trivial (fails the above thresholds), indicating timestep conditioning or other coupling; conclude that attention-graph isolation alone is insufficient.
- **End-to-end success**: On a prompt-dominant LongBench subset, our method improves throughput over the strongest cache baseline by a margin outside run-to-run noise while keeping LongBench avg score within **2% relative** of vanilla full attention.

---

## Impact Statement

If successful, this provides a simple, inference-only modification that enables **exact prompt KV reuse** for diffusion LLMs, reducing deployment cost on long-context, short-answer workloads (RAG-style QA, document classification, retrieval-heavy tasks). It would also establish a clear design rule for future dLLM architectures: prompt-isolated attention is a direct path to cacheability.

---

## References

- [LaViDa: A Large Diffusion Language Model for Multimodal Understanding](./references/LaViDa-A-Large-Diffusion-Language-Model-for-Multimodal-Understanding/meta/meta_info.txt) - Li et al., 2025
- [Large Language Diffusion Models](./references/Large-Language-Diffusion-Models/meta/meta_info.txt) - Nie et al., 2025
- [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](./references/dLLM-Cache-Accelerating-Diffusion-Large-Language-Models-with-Adaptive-Caching/meta/meta_info.txt) - Liu et al., 2025
- [d2Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](./references/d2Cache-Accelerating-Diffusion-Based-LLMs-via-Dual-Adaptive-Caching/meta/meta_info.txt) - Jiang et al., 2025
- [Attention Is All You Need for KV Cache in Diffusion LLMs](./references/ATTENTION-IS-ALL-YOU-NEED-FOR-KV-CACHE-IN-DIFFUSION-LLMS/meta/meta_info.txt) - Nguyen-Tri et al., 2025
- [Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference](./references/Mask-Tokens-as-Prophet-Fine-Grained-Cache-Eviction-for-Efficient-dLLM-Inference/meta/meta_info.txt) - Huang et al., 2025
- [Fast-dLLM](https://arxiv.org/abs/2505.22618)
- [dKV-Cache](https://arxiv.org/abs/2505.15781)
- [FlashDLM](https://arxiv.org/abs/2505.21467)
- [Sparse-dLLM](https://arxiv.org/abs/2508.02558)
- [Dream 7B](https://arxiv.org/abs/2508.15487)
- [WeDLM](https://arxiv.org/abs/2512.22737)
- [CARD](https://arxiv.org/abs/2601.22031)
- [DSB](https://arxiv.org/abs/2602.05992)
- [Block Diffusion](https://arxiv.org/abs/2503.09573)
- [Scaling Up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514)
- [Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://arxiv.org/abs/2406.03736)
- [MaskGIT](https://arxiv.org/abs/2202.04200)
- [SqueezeAttention](https://arxiv.org/abs/2404.04793)
- [AdaKV](https://arxiv.org/abs/2407.11550)
