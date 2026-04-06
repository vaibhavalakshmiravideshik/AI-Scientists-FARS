# untitled

# Sink-Free Attention Enables Prefix-Free Streaming KV Caches

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly deployed in **streaming** settings such as multi-turn assistants, where the input grows continuously and the model must respond while keeping latency and memory bounded. Standard transformer inference caches per-token **key/value (KV) states** to avoid recomputing attention over the full history, but the KV cache grows linearly with the number of processed tokens.

A natural efficiency approach is **rolling-window attention** (also called window attention): keep only the most recent \(W\) tokens’ KV states and evict older KV states to maintain constant memory. However, **naively evicting old tokens can cause catastrophic degeneration** in autoregressive transformers when reused KV states become inconsistent with the truncated context.

**StreamingLLM** showed that this collapse is tightly connected to **attention sinks**—a phenomenon where transformers allocate large attention mass to the first few tokens regardless of semantic relevance—and proposed a practical fix: keep a small number of initial “sink” tokens’ KV states alongside the rolling window \(W\) \[**Efficient Streaming Language Models with Attention Sinks**](./references/Efficient-Streaming-Language-Models-with-Attention-Sinks/meta/meta_info.txt). For example, on Llama-2-13B and a cache of 1024 tokens, window attention (0+1024) yields perplexity 5158.07, while keeping 4 sink tokens (4+1020) restores perplexity to 5.40 (PG19 first book; Table 1 in the paper).

Separately, newer architectural work reports that simple **post-attention gating** can eliminate attention sinks during training. **Gated Attention** applies head-specific sigmoid gates to the attention output (after **Scaled Dot-Product Attention (SDPA)**), reducing first-token attention from 46.7% to 4.8% and improving stability and long-context extrapolation \[**Gated Attention for Large Language Models**](./references/Gated-Attention-for-Large-Language-Models-Non-linearity-Sparsity-and-Attention-Sink-Free/meta/meta_info.txt). This suggests an important systems-level question:

> If an LLM is trained to be attention-sink-free, do we still need StreamingLLM’s special “keep prefix sink tokens” cache rule to avoid collapse?

Answering this would change two practical decisions:
1) whether streaming inference stacks should treat “prefix sink KV” as a required engineering trick, and 2) whether model builders can simplify deployment by training sink-free attention instead of engineering around sinks.

### The Problem

**Observed failure mode (StreamingLLM):** rolling-window KV caching collapses when early tokens are evicted. StreamingLLM argues this happens because many attention heads rely on globally-visible early tokens as “sinks” to absorb probability mass forced by softmax normalization; evicting those tokens changes the softmax normalization and causes large unintended residual updates.

**New possibility (sink-free architectures):** post-attention gating gives each head an explicit output-side “do nothing” option—multiply the attention output by a gate near zero—reducing the incentive to create a fixed sink token during training. If this is true, then a sink-free model might remain stable under pure rolling-window KV caches without preserving any special prefix tokens.

This is not currently settled by prior work:
- **StreamingLLM** evaluates window attention vs. “keep sink tokens”, but does not test sink-free architectures.
- **Gated Attention** shows sink-free attention maps and better length extrapolation, but does not evaluate StreamingLLM-style streaming KV cache truncation.
- **When Attention Sink Emerges** shows that changing the attention operation (e.g., unnormalized sigmoid attention) can eliminate sinks up to 1B scale, but does not connect this to streaming cache collapse \[**When Attention Sink Emerges in Language Models**](./references/When-Attention-Sink-Emerges-in-Language-Models-An-Empirical-View/meta/meta_info.txt).

### Key Insight and Hypothesis

**Key insight:** Streaming collapse is amplified by a mismatch between (i) a model trained to rely on stable sink tokens under softmax attention and (ii) a deployment-time cache eviction policy that removes those sink tokens while reusing stale KV states. If gating removes the training incentive to create sink tokens by enabling near-zero residual updates directly at the attention output, then pure rolling-window cache eviction may become stable.

**Hypothesis:** For LLM checkpoints trained with post-SDPA sigmoid gating that reduces attention sinks, **pure rolling-window KV caching (0+W)** will avoid the catastrophic perplexity collapse observed in vanilla checkpoints, under the same cache-relative positional handling used by StreamingLLM.

Why this could be wrong:
- Rolling-window collapse might be driven by factors beyond first-token sinks (e.g., other “movable” sinks such as punctuation tokens, or representation mismatch from truncation even when sink-rate is low).
- Sink tokens may function as a more general anchor than just absorbing attention mass, so removing them might still destabilize KV reuse.

We therefore include a mechanistic falsification check: if sink-rate is low but 0+W still collapses, that refutes the “sink-free is sufficient” mechanism.

---

## Proposed Approach

### Overview

We propose a minimal, decisive empirical test: evaluate whether an attention-sink-free checkpoint can run stably with **prefix-free streaming KV caches**.

Concretely, we compare rolling-window inference on:
1) a baseline checkpoint without gating, and
2) a gated checkpoint trained to suppress attention sinks.

We measure:
- **Streaming perplexity (PPL)** over a long text stream (PG19), under KV cache truncation.
- **Attention sink-rate** on the same text distribution to verify that the gated model is actually sink-free under the sink-rate metric.

### Method Details

#### KV-cache regimes
We implement two KV-cache regimes (following StreamingLLM terminology):

1. **Pure window (0+W)**: Keep only the most recent \(W\) tokens in KV cache. When cache exceeds \(W\), evict the oldest KV entries; do not recompute the remaining KV states.

2. **Prefix sinks (S+(W−S))**: Keep \(S=4\) prefix sink tokens’ KV states permanently, plus the most recent \(W−S\) tokens; evict only from the recent portion.

#### Cache-relative positional handling (to match StreamingLLM)
To avoid confounding from extremely large absolute RoPE positions (in streaming, the absolute position index can grow far beyond the training context length), we use **cache-relative positions** (as in StreamingLLM): positions are assigned by a token’s index in the cache (0..W−1) at each step. For RoPE models, this requires caching keys before applying rotary embedding and applying RoPE based on cache indices at each decode step (StreamingLLM Section 3.2).

This ensures the experiment specifically tests the need for **prefix sink tokens**, not out-of-distribution absolute positions.

#### Sink-rate metric (mechanism check)
We mirror the sink-rate definition used in **When Attention Sink Emerges in Language Models**. For sequences of length \(L=W\), for each layer \(\ell\) and head \(h\), let \(A^{(\ell,h)}\in\mathbb{R}^{L\times L}\) be the causal attention matrix. For token position \(k\), define
\[
 s_{\ell,h,k} = \frac{1}{L-k} \sum_{i=k}^{L-1} A^{(\ell,h)}_{i,k},
\]
(the mean attention paid to position \(k\) across all query positions that can attend to it).

Then
\[
 \text{SinkRate}(k,\varepsilon) = \frac{\#\{(\ell,h): s_{\ell,h,k} > \varepsilon\}}{\text{num\_layers}\cdot\text{num\_heads}}.
\]
We report \(\text{SinkRate}(k=0,\varepsilon=0.3)\) and optionally \(k\in\{1,2,3\}\) as diagnostics.

### Key Innovations

- Establishes (or refutes) a direct, deployment-relevant link between **sink-free attention architectures** and **streaming KV cache policies**.
- Provides a falsifiable mechanism check: **low SinkRate + collapse ⇒ sink-free attention maps are not sufficient for sink-free streaming**.
- If confirmed, yields a simple engineering recommendation: **prefix-free rolling caches are sufficient for sink-free checkpoints**, reducing special-case cache logic.

---

## Related Work

### Field Overview

**Streaming and long-context inference.** Many systems optimize transformer inference by caching KV states and reducing attention computation, including paged KV memory management, selective eviction, and sliding-window recomputation. StreamingLLM highlights that simple rolling-window caches can fail catastrophically for pretrained LLMs, and proposes preserving a small set of sink tokens to stabilize reuse.

**Attention sinks and their causes.** Attention sinks (high attention mass assigned to specific tokens like the first token) have been linked to softmax normalization constraints, training dynamics, and activation outliers. Empirical work suggests sinks behave like non-informative key biases and can be shifted by data and loss choices.

**Sink-mitigation via architectural changes.** Several lines of work introduce gating, modified softmax variants, key/value biases, or alternative attention operations to allow heads to “do nothing” without relying on sink tokens. These changes have been motivated by stability, quantization, and long-context behavior, but their implications for streaming KV cache truncation have not been directly tested.

### Related Papers

- **[Efficient Streaming Language Models with Attention Sinks](./references/Efficient-Streaming-Language-Models-with-Attention-Sinks/meta/meta_info.txt)**: Introduces StreamingLLM and shows rolling-window KV caching collapses unless a few prefix sink tokens are preserved.
- **[Gated Attention for Large Language Models](./references/Gated-Attention-for-Large-Language-Models-Non-linearity-Sparsity-and-Attention-Sink-Free/meta/meta_info.txt)**: Shows post-SDPA sigmoid gating reduces first-token sink attention and improves stability/length extrapolation.
- **[When Attention Sink Emerges in Language Models](./references/When-Attention-Sink-Emerges-in-Language-Models-An-Empirical-View/meta/meta_info.txt)**: Empirical study of sink emergence; defines sink metrics and shows unnormalized sigmoid attention can eliminate sinks up to 1B.
- **[Quantizable Transformers](./references/Quantizable-Transformers-Removing-Outliers-by-Helping-Attention-Heads-Do-Nothing/meta/meta_info.txt)**: Links activation outliers to heads attempting no-ops; proposes clipped softmax and gated attention to enable no-op behavior.
- **[Forgetting Transformer](./references/Forgetting-Transformer-Softmax-Attention-with-a-Forget-Gate/meta/meta_info.txt)**: Adds data-dependent forget gates to softmax attention to improve long-context modeling.
- **[Softpick: No Attention Sink, No Massive Activations with Rectified Softmax](https://arxiv.org/abs/2504.20966)**: Replaces softmax with a rectified normalization to remove sinks and reduce activation outliers.
- **[Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)**: Studies activation outliers and their relationship to attention patterns.
- **[Spectral Filters, Dark Signals, and Attention Sinks](https://arxiv.org/abs/2402.09221)**: Mechanistic analysis linking attention sinks to “dark” subspaces in the residual stream.
- **[Unveiling and Harnessing Hidden Attention Sinks](https://arxiv.org/abs/2406.15765)**: Shows sinks beyond the initial token and proposes a training-free attention calibration method (ACT).
- **[SmoothQuant](https://arxiv.org/abs/2211.10438)**: Post-training quantization method motivated partly by activation outliers in transformers.
- **[LLM.int8()](https://arxiv.org/abs/2208.07339)**: Mixed-precision method to handle outlier channels during inference.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)**: IO-aware exact attention kernel enabling fast attention at moderate context lengths.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**: Improved attention kernel for better parallelism and longer contexts.
- **[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)**: Efficient KV cache memory management for serving LLMs.
- **[ALiBi](https://arxiv.org/abs/2108.12409)**: Linear biases for length extrapolation without RoPE.
- **[RoPE](https://arxiv.org/abs/2104.09864)**: Rotary position embeddings widely used in modern decoder-only LLMs.
- **[YaRN](https://arxiv.org/abs/2309.00071)**: RoPE-based context extension via interpolation and scaling.
- **[LongRoPE](https://arxiv.org/abs/2402.13753)**: Extends RoPE context length with learned scaling.
- **[Longformer](https://arxiv.org/abs/2004.05150)**: Sparse attention pattern for long documents.
- **[H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048)**: KV cache eviction policy based on attention “heavy hitters”.
- **[Lost in the Middle](https://arxiv.org/abs/2307.03172)**: Shows LLMs often under-utilize long contexts, motivating careful evaluation.

(Some citations are provided as arXiv links; the core evaluation does not depend on them.)

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Streaming KV-cache methods | Keep compute/memory bounded during long-running inference | StreamingLLM; vLLM/PagedAttention; FlashAttention | PG19, streaming QA, latency | Can require special cache logic (sink tokens) or recomputation |
| Sink diagnosis | Measure and explain sink emergence in attention patterns | When Attention Sink Emerges; Massive Activations | SinkRate metrics; attention visualizations | Sink measures can be threshold-sensitive; sinks can be movable |
| Sink mitigation (architecture) | Let heads “do nothing” without relying on sink tokens | Gated Attention; Quantizable Transformers; Softpick; Forgetting Transformer | PPL, stability, quantization, long-context suites | Interaction with streaming cache reuse largely untested |

### Closest Prior Work

**StreamingLLM** shows the practical collapse of naive rolling-window KV caching and introduces the “keep 4 sink tokens” rule as a simple fix. It does not evaluate whether sink-free attention architectures change the necessity of sink tokens.

**Gated Attention** demonstrates that post-SDPA gating eliminates attention sinks and improves stability and length extrapolation, but does not test streaming KV cache truncation (the setting where sink tokens were proposed as a deployment fix).

**When Attention Sink Emerges** provides the sink-rate metric and shows that replacing softmax with unnormalized sigmoid attention can eliminate sinks up to 1B models; it does not connect sink removal to streaming KV cache reuse.

**Quantizable Transformers** motivates gating/softmax changes via “heads want to do nothing” and outlier suppression; it evaluates quantization and standard metrics but not streaming cache truncation.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| StreamingLLM | Preserves a few prefix sink tokens to stabilize rolling KV caching | Assumes sink tokens are necessary; does not test sink-free checkpoints | Test prefix-free rolling cache on sink-free checkpoints | Could remove special-case sink-token cache rules if sink-free works |
| Gated Attention | Trains sink-free checkpoints via post-SDPA gating | No streaming KV cache truncation evaluation | Evaluate 0+W rolling cache stability | Directly answers deployment question for sink-free models |
| When Attention Sink Emerges | Explains sink emergence; proposes alternative attention ops | No streaming setting; ≤1B scale for alternatives | Use its SinkRate metric as mechanism check | Links sink-rate to streaming stability/failure |
| Quantizable Transformers | Shows gating/clipped softmax reduce outliers by enabling no-op heads | Focuses on quantization; not streaming KV reuse | Use gating as sink-free mechanism in streaming | Tests if “no-op heads” translates to sink-free streaming |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| QwQZh/gated_attention/1B_baseline | ~1B | https://huggingface.co/QwQZh/gated_attention | Baseline checkpoint without attention output gates |
| QwQZh/gated_attention/1B_gate_headwise | ~1B | https://huggingface.co/QwQZh/gated_attention | Headwise post-SDPA sigmoid gating |
| QwQZh/gated_attention/1B_gate_elementwise (optional swap for headwise) | ~1B | https://huggingface.co/QwQZh/gated_attention | Elementwise post-SDPA sigmoid gating |

**Training Data (if applicable):**
- No training data needed — inference-only evaluation on pretrained checkpoints.

**Resource Estimate**:
- **Compute budget**: ~5–20 GPU-hours total (single A100-class GPU). Dominated by streaming PPL eval over \(\sim\)20k–65k tokens for 2–3 conditions and a small number of attention-dump passes for SinkRate.
- **GPU memory**: ≤ 80GB for 1B models; SinkRate extraction with attentions may require more memory but can be run with smaller batch size.
- **API usage**: None.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| PG19 | Long-book language modeling benchmark used by StreamingLLM | Perplexity vs token index; average PPL | test (first book) | https://huggingface.co/datasets/pg19 | Custom script based on StreamingLLM eval protocol |

Primary metric:
- **Streaming perplexity (PPL)** under rolling KV caches: compute next-token negative log-likelihood sequentially while maintaining/truncating KV cache.

Mechanism metric:
- **SinkRate(0, 0.3)** computed on length-\(W\) segments sampled from the same PG19 stream.

### Main Results

#### Results Table

(All numbers are **TBD** and require running; no published results exist for this exact model+streaming-cache setting.)

| Method | Base Model | Benchmark | Avg PPL (stream) | Collapse? | SinkRate(0,0.3) | Source | Notes |
|--------|------------|-----------|------------------|-----------|-----------------|--------|------|
| Pure window (0+1024) | 1B_baseline | PG19 | **TBD** | **TBD** | **TBD** | - | Sanity check (expected collapse) |
| Prefix sinks (4+1020) | 1B_baseline | PG19 | **TBD** | **TBD** | **TBD** | - | Sanity check (expected stable) |
| Pure window (0+1024) | 1B_gate_headwise | PG19 | **TBD** | **TBD** | **TBD** | - | Main condition |
| Prefix sinks (4+1020) | 1B_gate_headwise | PG19 | **TBD** | **TBD** | **TBD** | - | Reference for PPL ratio |

Sanity check (not a main condition): full-attention PPL on short context (≤1024 tokens) for baseline vs gated to flag global degradation.

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Gate elementwise vs headwise | Swap gating granularity | Both should reduce SinkRate; streaming stability may differ |
| (Optional) Full attention (short context) | Evaluate ≤1024 full attention PPL | Confirms the gated checkpoint isn’t globally degraded vs baseline |

---

## Success Criteria

**Criterion 1: Prefix-free stability for sink-free checkpoint**
- Hypothesis: The gated checkpoint (low SinkRate) is stable under pure window caching.
- Validation: Using the pre-registered rule with \(W=1024\) **on the gated checkpoint** (compare its 0+W vs 4+(W−4) regimes):
  - **Success (prefix-free is sufficient)** if \(\text{PPL}^{\text{gate}}_{0+W}/\text{PPL}^{\text{gate}}_{4+(W-4)} \le 1.2\) and no PPL spike above 100 after cache fill.
  - **Failure** otherwise. We further label failures as:
    - **Collapse** if \(\text{PPL}^{\text{gate}}_{0+W}/\text{PPL}^{\text{gate}}_{4+(W-4)} \ge 3.0\) or max PPL exceeds 1000 after cache fill.
    - **Degraded-but-not-collapsed** if the ratio is in (1.2, 3.0) or max PPL is in (100, 1000). (Still a failure for the main claim, but informative.)

**Criterion 2: Mechanism check is decisive**
- Hypothesis: SinkRate reduction is necessary for prefix-free stability.
- Validation:
  - If SinkRate(0,0.3) is low for the gated checkpoint but it still collapses, conclude sink-free attention maps are not sufficient for sink-free streaming (mechanism refuted).

---

## Impact Statement

If sink-free checkpoints can run stably with prefix-free rolling KV caches, streaming inference stacks can simplify cache management (no special preserved prefix KV) and model developers gain a concrete incentive to adopt sink-free attention as a deployment-oriented design choice.

---

## References

- [Efficient Streaming Language Models with Attention Sinks](./references/Efficient-Streaming-Language-Models-with-Attention-Sinks/meta/meta_info.txt) - Xiao et al., 2023
- [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](./references/Gated-Attention-for-Large-Language-Models-Non-linearity-Sparsity-and-Attention-Sink-Free/meta/meta_info.txt) - Qiu et al., 2025
- [When Attention Sink Emerges in Language Models: An Empirical View](./references/When-Attention-Sink-Emerges-in-Language-Models-An-Empirical-View/meta/meta_info.txt) - Gu et al., 2024
- [Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing](./references/Quantizable-Transformers-Removing-Outliers-by-Helping-Attention-Heads-Do-Nothing/meta/meta_info.txt) - Bondarenko et al., 2023
- [Forgetting Transformer: Softmax Attention with a Forget Gate](./references/Forgetting-Transformer-Softmax-Attention-with-a-Forget-Gate/meta/meta_info.txt) - Lin et al., 2025
- [Softpick: No Attention Sink, No Massive Activations with Rectified Softmax](https://arxiv.org/abs/2504.20966) - Zuhri et al., 2025
- [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762) - Sun et al., 2024
- [Spectral Filters, Dark Signals, and Attention Sinks](https://arxiv.org/abs/2402.09221) - Cancedda, 2024
- [Unveiling and Harnessing Hidden Attention Sinks: Enhancing Large Language Models without Training through Attention Calibration](https://arxiv.org/abs/2406.15765) - Yu et al., 2024
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) - Xiao et al., 2022
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) - Dettmers et al., 2022
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) - Dao, 2023
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) - Press et al., 2021
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al., 2021
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) - Peng et al., 2023
- [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) - Chen et al., 2024
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Beltagy et al., 2020
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) - Zhang et al., 2023
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - Liu et al., 2023
