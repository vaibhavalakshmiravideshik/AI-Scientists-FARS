# untitled

# Prefill Twice, Decode Once: Keeping Only the Second Copy’s KV Cache for Prompt Repetition

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Modern large language model (LLM) serving is dominated by the **decode** phase: after the prompt is processed (“prefill”), the model generates tokens one-by-one, repeatedly attending to a growing key–value (KV) cache. For long prompts, decode can become memory-bandwidth limited because each generated token attends over all cached prompt tokens across layers and heads.

A recent prompting method, **prompt repetition** (arXiv:2512.14982), improves accuracy on many non-reasoning tasks by duplicating the user prompt from `P` to `P||P`. The central motivation is that decoder-only LLMs use causal attention; in the first copy of the prompt, early tokens cannot attend to later tokens, but in the **second copy**, tokens can attend to the entire first copy, approximating bidirectional context within the prompt. The paper reports **47 wins out of 70** model–benchmark tests (0 losses) when reasoning is disabled and notes especially large gains in position-sensitive settings such as “options-first” multiple-choice formatting and synthetic retrieval tasks.

However, naïvely using `P||P` doubles the prompt length. Even if wall-clock latency is often dominated by decode rather than prefill, doubling the number of cached prompt tokens can increase decode-time memory traffic, reduce batch size, and increase cost for long generations.

### The Problem

The prompt repetition paper explicitly proposes a systems optimization as future work: **“Only keep the second repetition in the KV-cache (thus being completely performance neutral for the generation stage)”**. The open question is whether this is actually correct in practice.

In a decoder-only transformer, generation after prefill uses the KV cache to compute attention for each new token. If we prefill `P||P` but then discard the KV entries for the **first** copy of the prompt (keeping only the KV for the second copy), we halve the KV length seen during decode. This should reduce decode-time KV memory footprint and attention read bandwidth.

But this is not obviously safe. Even if the second copy enables better representations during prefill, decode-time queries might still rely on attending into the first-copy KV (e.g., if the first copy acts as an explicit “memory bank”). Additionally, incorrect handling of **Rotary Position Embedding (RoPE)** (a common positional encoding used in decoder-only transformers) and associated positional indices could break the computation even if the underlying hypothesis is true.

### Key Insight and Hypothesis

**Key insight**: The *benefit* of prompt repetition may be realized during prefill, when the model computes the second-copy prompt token representations using the first copy as context. If so, then at **decode time** the first-copy KV may be redundant: new tokens might only need to attend over the second-copy KV, whose contents already “encode” the global prompt.

**Hypothesis (test-aligned)**: After prefilling `P||P`, discarding the first-copy KV cache and decoding with only the second-copy KV (with correct positional offset handling) will retain most of the accuracy gain of full `P||P` while reducing decode-time KV footprint by ~2×.

The outcome is uncertain because:
1) the decoder might require direct attention into first-copy KV during generation for retrieval; and
2) small positional/RoPE implementation details can cause false negative results.

---

## Proposed Approach

### Overview

We propose **Prefill Twice, Decode Once (PTDO)**:

1. **Prefill** the repeated prompt `P||P` with standard caching (`use_cache=True`).
2. **Slice** the resulting cached key–value tensors (called `past_key_values` in the HuggingFace Transformers API) to keep only the KV entries corresponding to the **second** copy of `P` (the last `\|P\|` prompt positions), discarding the first-copy KV.
3. **Decode** from this sliced cache, using explicit positional indices (`position_ids` / `cache_position` in Transformers) so that the first generated token is positioned immediately after `P||P` (i.e., positions start at `2\|P\|`).

### Method Details

**Cache slicing**
- For each transformer layer ℓ and attention head h, `past_key_values[ℓ]` contains K and V tensors with a sequence-length dimension.
- After prefill on `P||P`, let total prompt length be `L=2\|P\|`.
- Slice along the sequence-length dimension to keep indices `[\|P\|, ..., 2\|P\|-1]`.

**Position/RoPE handling**
- During decode, set `position_ids` / `cache_position` such that the first generated token has position `L` (not `\|P\|`).
- This preserves the positional geometry as if the model still had a length-`L` cache, but with only the last `\|P\|` KV entries retained.

**Sanity checks (to avoid conflating bugs with hypothesis failure)**
- **S1 (cache correctness)**: Verify that cached vs non-cached forward passes for `P||P` produce matching logits on the prefix.
- **S2 (implementation isolation)**: On a small fixed batch (e.g., 50 prompts), compare greedy generation under full `P||P` cache vs sliced cache.
  - If the **first generated token** differs on >15% of prompts, treat this as a positional-implementation bug (halt and fix offset handling) rather than evidence against the hypothesis.
  - If first tokens match but later tokens diverge, proceed; later divergence is expected if the first-copy KV contributes non-trivially.
- **Determinism check**: Run the same setting twice on 10 prompts and confirm identical outputs; otherwise, enforce deterministic settings or compare first-token log probabilities (logprobs) instead of exact tokens in S2.

### Key Innovations

- **Structure-exploiting KV reduction**: Unlike generic KV eviction/compression (SnapKV/H2O/PyramidKV/KVzip), PTDO exploits the special structure of *exact prompt duplication* to discard half the cache without any learned importance model.
- **A decisive systems hypothesis**: PTDO tests a concrete, high-leverage claim suggested (but not validated) in the prompt repetition paper: whether first-copy KV is decode-time redundant.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) prompt engineering methods that change prefill computation (prompt repetition, re-reading), and (ii) KV-cache efficiency methods that reduce decode-time memory and compute (eviction, compression, sharing, and prefix caching).

Prompt repetition (arXiv:2512.14982) shows that duplicating the input prompt can improve accuracy without increasing output length and often without noticeable end-to-end latency changes in API settings. However, local serving and long-generation settings can still be bottlenecked by KV cache size and decode-time memory bandwidth.

KV-cache efficiency methods typically either (a) **compress/evict** KV entries based on learned or heuristic importance, or (b) **reuse** KV across requests via prefix caching. These methods do not directly address the within-request structure created by `P||P`.

### Related Papers

- **[Prompt Repetition Improves Non-Reasoning LLMs](./references/Prompt-Repetition-Improves-Non-Reasoning-LLMs/meta/meta_info.txt)**: Introduces `P||P` and reports broad accuracy gains; explicitly suggests “only keep the second repetition in the KV-cache” as future work.
- **[KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](./references/KVzip-Query-Agnostic-KV-Cache-Compression-with-Context-Reconstruction/meta/meta_info.txt)**: Compresses KV using a reconstruction objective; includes “repeat prompts” for scoring but does not study discarding half the cache created by prompt duplication.
- **[Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference](./references/Mask-Tokens-as-Prophet-Fine-Grained-Cache-Eviction-for-Efficient-dLLM-Inference/meta/meta_info.txt)**: Training-free cache eviction framework (for diffusion LLMs) and a representative example of the broader KV eviction literature.
- **[Asking Again and Again: Exploring LLM Robustness to Repeated Questions](https://arxiv.org/abs/2412.07923)**: Studies repeated questions (not full-prompt duplication) and reports limited gains; motivates that repetition effects can be prompt-structure dependent.
- **[Repetition Improves Language Model Embeddings](https://arxiv.org/abs/2402.15449)**: Shows repetition effects for embeddings; related phenomenon but not about decode-time KV.
- **[Re-reading Improves Reasoning in Large Language Models](https://arxiv.org/abs/2309.06275)**: Uses explicit re-reading via generation; contrasts with shifting repetition to prefill.
- **[Selective Attention Improves Transformer](https://arxiv.org/abs/2410.02703)**: Example of inference-time attention sparsification; prompt repetition paper suggests studying interactions.
- **[vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)**: vLLM is an open-source high-throughput LLM serving framework; this paper introduces PagedAttention for efficient KV memory management and prefix caching.
- **[You Only Cache Once: Decoder-Decoder Architectures for Efficient LLM Inference](https://neurips.cc/virtual/2024/poster/96833)**: Architectural alternative that changes caching patterns; provides context for why cache structure matters.
- **[KV Cache Reuse — TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)**: Engineering reference for prefix KV reuse across requests (not within-request repetition).
- **[Automatic Prefix Caching — vLLM Design Doc](https://docs.vllm.ai/en/latest/design/prefix_caching/)**: Prefix caching in serving systems; conceptually adjacent but not the same setting.
- **[Hybrid KV Cache Manager — vLLM Design Doc](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)**: Notes constraints for sliding-window vs full attention layers; relevant if testing on hybrid attention models.
- **[Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores](https://arxiv.org/abs/2507.08143)**: Training-free query-agnostic KV compression using approximate leverage scores; relevant contrast to “structure-aware” discard.
- **[StreamingLLM](https://arxiv.org/abs/2309.17453)**: Streaming attention and sink tokens; representative long-context decode-cost baseline.
- **[H2O: Heavy-Hitter Oracle for KV Cache Eviction](https://arxiv.org/abs/2306.14048)**: Early attention-score eviction baseline family.
- **[SnapKV: LLM Knows What You Are Looking for Before Generation](https://arxiv.org/abs/2404.14469)**: Query-aware KV eviction/compression using an observation window at the end of the prompt.
- **[PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling](https://arxiv.org/abs/2406.02069)**: Layer-wise KV budget allocation (“pyramid”) for long-context inference.
- **[Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation](https://arxiv.org/abs/2407.11550)**: Improves eviction methods (e.g., SnapKV/PyramidKV) by allocating different KV budgets per head based on attention concentration.
- **[PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference](https://arxiv.org/abs/2405.12532)**: Related pyramid-style KV compression focused on throughput; alternative baseline family.
- **[SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts](https://arxiv.org/abs/2509.24832)**: Extends caching beyond exact-match prefixes; adjacent to prompt-structure exploitation.
- **[KV Cache Recycling to Expand Usable Context Capacity](https://arxiv.org/abs/2512.11851)**: Reuses cached KV across prompts via recycling; not about `P||P`.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Prompt repetition / re-reading | Improve performance by re-exposing the prompt | Prompt Repetition (2512.14982), Re-reading (2309.06275) | ARC/OpenBookQA/MMLU-Pro; synthetic retrieval | Can increase input length; unclear serving cost impact |
| KV eviction/compression | Reduce KV length by importance scoring | H2O, SnapKV, PyramidKV, KVzip | LongBench, RULER, SCBench | May require query-aware scoring; can hurt retrieval |
| Prefix caching (across requests) | Reuse KV for shared prefixes | vLLM prefix caching, TensorRT-LLM KV reuse | TTFT and throughput under repeated prefixes | Requires exact match; not within-request duplication |
| Architectural changes | Change attention/caching structure | YOCO | Throughput + accuracy | Requires new model or finetuning |

### Closest Prior Work

1. **Prompt Repetition Improves Non-Reasoning LLMs (2512.14982)**: Proposes `P||P` and suggests “keep only the second repetition in KV” as future work but does not test it. PTDO directly tests this suggestion with a falsifiable decode-time experiment.
2. **KVzip (2505.23416)**: Uses repeated context during scoring (“Repeat Prompts”) to measure KV importance for compression, but its goal is query-agnostic compression for long-context QA. PTDO is a zero-training, structure-specific discard rule for exact duplication.
3. **KV eviction baselines (H2O/SnapKV/PyramidKV)**: These methods select a subset of tokens to keep, typically based on attention scores or heuristic budgets. They do not exploit the fact that the entire prompt is duplicated.
4. **Serving systems prefix caching (vLLM/TensorRT-LLM)**: Cache reuse across requests is conceptually related to “keep a cache and reuse it”, but does not address discarding within a single request after `P||P` prefill.

**Novelty Kill Search Summary (2026-02-21):**
- WebSearch queries included “only keep the second repetition KV cache”, “prefill twice decode once KV cache”, “prompt repetition KV cache discard first”, and variants with arXiv:2512.14982 and “past_key_values”. Results surfaced the prompt repetition paper itself and generic KV/prefix caching documentation, but no prior work implementing the keep-second decode trick.
- Local repo search (`Grep`) over finalized proposals and paper summaries found no proposal or paper summary describing this exact within-request KV slicing after `P||P`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Prompt Repetition (2512.14982) | Improves accuracy by duplicating prompt `P||P` | Doubles prompt length and cached KV length | Keep only 2nd-copy KV for decode | If first-copy KV is decode-time redundant, retain gains at ~half KV footprint |
| KVzip (2505.23416) | Compress KV via context reconstruction | General method; not targeted to exact duplication | Use exact duplication structure | No learned scoring; deterministic and simple |
| H2O / SnapKV / PyramidKV | Evict/compress KV by importance | May degrade retrieval; needs tuning | Deterministic discard of first copy | Avoids importance modeling; tests a clean structural hypothesis |
| vLLM prefix caching | Reuse KV across requests | Requires exact shared prefixes across requests | Within-request discard after repetition | Addresses prompt repetition overhead inside one request |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3.1-8B-Instruct | 8B | https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct | Primary target (widely used 8B serving model) |
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Second model to test whether the PTDO trick generalizes across model families |

**Training Data:**
- No training; inference-only.

**Baseline Ladder (REQUIRED):**

- **Level 0 (trivial)**: Random guess among candidate names (chance = 1/M).  *(Note: `M` is the number of candidate names in the list.)*
- **Level 1 (zero-shot)**: Baseline prompt `P`.
- **Level 4 (inference-time scaling)**: `P` with best-of-8 sampling (Bo8) or self-consistency @8 (SC@8) for robustness to sampling noise (optional; only if `P||P` gains are small).
- **Level 5 (closest method family)**: Full prompt repetition `P||P` with full KV (this is the closest method, since PTDO is a systems modification of prompt repetition).

**Our method**: Prefill Twice, Decode Once (PTDO): prefill `P||P`, then keep only second-copy KV for decode.

**Benchmark 1: Synthetic NameIndex (task-aligned to position sensitivity)**
- Each example contains a list of **M unique names**, one per line, followed by a question asking for the **k-th** name.
- Names are random alphabetic strings (length 6–10), capitalized; no numbering.
- Prompt structure:
  1) `Here is a list of names (one per line):\n` + names
  2) `Question: What is the {k}-th name in the list? Answer with exactly the name.`
- Choose (M,k) such that total prompt token length is in **[1024, 2048]** (target ~1500 tokens), and `k` is uniform in `[1,M]`.
- Test set size: **N=1000**.

**Benchmark 2: ARC-Challenge (options-first multiple-choice science QA)**
- ARC-Challenge is the “hard” subset of the AI2 Reasoning Challenge (ARC), a standard grade-school science multiple-choice benchmark.
- Dataset: HuggingFace `allenai/ai2_arc`, configuration `ARC-Challenge`, split `test` (**N=1172**).
- We evaluate in the **options-first** formatting used in the prompt repetition paper: present answer choices before the question so early tokens (options) are processed without the question in context unless repetition is used.
- Prompt template (per example):
  1) `Answer choices:\nA) ...\nB) ...\nC) ...\nD) ...\n\nQuestion: ...\nAnswer with only the letter (A/B/C/D).`
- Metric: accuracy = fraction of examples where the parsed output letter equals the dataset `answerKey`.

**Metrics:**
- **Accuracy**: exact match of the parsed answer to the gold label (name for NameIndex; answerKey letter for ARC-Challenge).
- **KV bytes at decode start**: measure total `past_key_values` bytes resident at the start of decode.
- **Optional throughput**: decode tokens/s on a fixed long generation (e.g., force 256 new tokens) to make decode cost measurable.

**Decoding / sampling:**
- Main comparisons (A/B/C): greedy decoding (temperature=0) with `max_new_tokens=8`.
- Best-of/self-consistency baseline (if used): temperature=0.7, top_p=0.95, k=8, with 3 seeds `seeds=[42,123,456]`.

**Sanity checks (implementation):**
- S1 cache correctness as described above.
- S2 first-token agreement check (B vs C) on 50 prompts, with the >15% divergence halt rule.
- Determinism check: run B twice on 10 prompts.

**Resource Estimate (evidence-based):**
- Main eval: 1000 prompts × 3 conditions × short decode (≤8 tokens) is small.
- Optional throughput probe: 1000 prompts × 2 conditions (B vs C) × 256 tokens.
- On an A100 80GB with vLLM, public benchmarks report Llama-3.1-8B chat generation throughput on the order of **~2.6k tokens/s**, with similar ranges in other vLLM A100 benchmarks. Even assuming only ~500 tokens/s effective throughput for this controlled harness, the total generation tokens (≈1000×256×2 ≈ 512k tokens) is << 1 GPU-hour.
- Total budget (including overhead, cache instrumentation, seeds): **≤ 20 A100 GPU-hours**.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Synthetic NameIndex (this work) | Long prompt retrieval: return k-th item from a long list | Exact-match accuracy; KV bytes; optional tok/s | test (N=1000) | Generated on-the-fly | Custom script (simple string match) |
| ARC-Challenge (options-first) | Multiple-choice science QA; options shown before question | Accuracy; KV bytes; optional tok/s | test (N=1172) | https://huggingface.co/datasets/allenai/ai2_arc | HuggingFace datasets + simple answerKey parsing |

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Accuracy (mean±std) | KV bytes @ decode start | Source | Notes |
|---|---|---|---:|---:|---|---|
| Random guess | Llama-3.1-8B | NameIndex | - | 0 | - | Chance = 1/M |
| Baseline `P` | Llama-3.1-8B | NameIndex | - | - | - | Greedy |
| `P` + Bo8/SC@8 | Llama-3.1-8B | NameIndex | - | - | - | If needed; 3 seeds |
| Full repetition `P||P` | Llama-3.1-8B | NameIndex | - | - | [Prompt Repetition](./references/Prompt-Repetition-Improves-Non-Reasoning-LLMs/meta/meta_info.txt) | Greedy; cache length 2\|P\| |
| **PTDO (ours)** | Llama-3.1-8B | NameIndex | - | - | - | Keep only last \|P\| KV; greedy |
| Baseline `P` | Llama-3.1-8B | ARC-Challenge (options-first) | - | - | - | Greedy |
| Full repetition `P||P` | Llama-3.1-8B | ARC-Challenge (options-first) | - | - | [Prompt Repetition](./references/Prompt-Repetition-Improves-Non-Reasoning-LLMs/meta/meta_info.txt) | Greedy; cache length 2\|P\| |
| **PTDO (ours)** | Llama-3.1-8B | ARC-Challenge (options-first) | - | - | - | Keep only last \|P\| KV; greedy |
| Baseline `P` | Qwen2.5-7B | NameIndex | - | - | - | Greedy |
| Full repetition `P||P` | Qwen2.5-7B | NameIndex | - | - | [Prompt Repetition](./references/Prompt-Repetition-Improves-Non-Reasoning-LLMs/meta/meta_info.txt) | Greedy; cache length 2\|P\| |
| **PTDO (ours)** | Qwen2.5-7B | NameIndex | - | - | - | Keep only last \|P\| KV; greedy |
| Baseline `P` | Qwen2.5-7B | ARC-Challenge (options-first) | - | - | - | Greedy |
| Full repetition `P||P` | Qwen2.5-7B | ARC-Challenge (options-first) | - | - | [Prompt Repetition](./references/Prompt-Repetition-Improves-Non-Reasoning-LLMs/meta/meta_info.txt) | Greedy; cache length 2\|P\| |
| **PTDO (ours)** | Qwen2.5-7B | ARC-Challenge (options-first) | - | - | - | Keep only last \|P\| KV; greedy |

Replicate the same evaluation for **both** base models (Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct) to test cross-family generalization.

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| PTDO w/ wrong position offset (debug-only) | Reset positions as if cache length were \|P\| | Should fail S2 (first-token mismatch), diagnosing RoPE issues |
| PTDO w/ no slicing | Equivalent to full `P||P` | Recovers full repetition baseline |

### Experimental Rigor

**Variance & Reproducibility:**
- Main A/B/C comparisons use greedy decoding and are deterministic given deterministic kernels; still run the determinism check described above.
- If Bo8/SC@8 is included, use 3 seeds and report mean±std.

**Validity & Controls:**
- **Task alignment control**: The synthetic NameIndex is designed to amplify position sensitivity; random guess sanity check ensures evaluation is meaningful.
- **Implementation confound**: S2 first-token agreement is used to detect positional bugs that would otherwise create false negatives.
- **Prompt-length control**: Ensure all methods use the same underlying `P` content; only repetition and cache slicing differ.

---

## Success Criteria

**Hypothesis (directional):**
- Full prompt repetition `P||P` improves accuracy over baseline `P` on **position-sensitive** settings (Synthetic NameIndex and ARC-Challenge options-first).
- PTDO retains most of that improvement while halving decode-time KV footprint.

**Decision Rule (concrete):**
- **Refute early (mismatch)**: If `Acc(P||P) - Acc(P) < 3` points on **both** benchmarks for a given model, conclude the chosen model/benchmarks do not exhibit the repetition effect strongly enough and stop for that model.
- **Proceed (success)**: For **at least one benchmark** (NameIndex or ARC-Challenge) and for **at least one model** (Llama-3.1-8B or Qwen2.5-7B),
  - `Acc(PTDO) - Acc(P) ≥ 0.8 × (Acc(P||P) - Acc(P))` **and**
  - `KV_bytes(PTDO) ≤ 0.55 × KV_bytes(P||P)`.
- **Refute (first-copy KV matters at decode)**: If `Acc(P||P) - Acc(P) ≥ 5` points but `Acc(PTDO) - Acc(P) < 0.5 × (Acc(P||P) - Acc(P))` on **both** benchmarks for **both** models.
- **Pivot**: If repetition effect is strong but PTDO loses most gains on ARC-Challenge while working on NameIndex, treat this as a boundary condition (task-dependent) and test whether repeating only a suffix/segment of P is a better trade-off.

---

## Impact Statement

If PTDO works, LLM serving stacks could apply prompt repetition while keeping decode-time KV footprint near baseline, improving accuracy in position-sensitive, non-reasoning tasks with less throughput/memory cost. If it fails, the result is still decision-changing: it would indicate that prompt repetition’s benefits require retaining first-copy KV during decode, limiting its practicality for long-generation deployments.

---

## References

- [Prompt Repetition Improves Non-Reasoning LLMs](./references/Prompt-Repetition-Improves-Non-Reasoning-LLMs/meta/meta_info.txt) - 2025.
- [KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](./references/KVzip-Query-Agnostic-KV-Cache-Compression-with-Context-Reconstruction/meta/meta_info.txt) - 2025.
- [Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference](./references/Mask-Tokens-as-Prophet-Fine-Grained-Cache-Eviction-for-Efficient-dLLM-Inference/meta/meta_info.txt) - 2025.
- [Asking Again and Again: Exploring LLM Robustness to Repeated Questions](https://arxiv.org/abs/2412.07923) - 2024.
- [Repetition Improves Language Model Embeddings](https://arxiv.org/abs/2402.15449) - 2024.
- [Re-reading Improves Reasoning in Large Language Models](https://arxiv.org/abs/2309.06275) - 2024.
- [Selective Attention Improves Transformer](https://arxiv.org/abs/2410.02703) - 2024.
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - 2023.
- [KV cache reuse — TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html) - documentation.
- [Automatic Prefix Caching — vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/) - documentation.
- [Hybrid KV Cache Manager — vLLM](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) - documentation.
- [You Only Cache Once: Decoder-Decoder Architectures for Efficient LLM Inference](https://neurips.cc/virtual/2024/poster/96833) - 2024.
- [Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores](https://arxiv.org/abs/2507.08143) - 2025.
- [StreamingLLM](https://arxiv.org/abs/2309.17453) - 2023.
- [H2O: Heavy-Hitter Oracle for KV Cache Eviction](https://arxiv.org/abs/2306.14048) - 2023.
- [SnapKV: LLM Knows What You Are Looking for Before Generation](https://arxiv.org/abs/2404.14469) - 2024.
- [PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling](https://arxiv.org/abs/2406.02069) - 2024.
- [Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation](https://arxiv.org/abs/2407.11550) - 2024.
- [PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference](https://arxiv.org/abs/2405.12532) - 2024.
- [SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts](https://arxiv.org/abs/2509.24832) - 2025.
- [KV Cache Recycling to Expand Usable Context Capacity](https://arxiv.org/abs/2512.11851) - 2025.
