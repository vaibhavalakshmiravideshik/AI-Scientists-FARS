# untitled

# GateTune vs SinkTune: Parameter-Efficient Adaptation for Aggressive Sliding-Window Shrink in Hybrid Local–Global LLMs

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Many modern large language models (LLMs) support very long contexts (tens to hundreds of thousands of tokens). A practical deployment bottleneck is the **prefill** stage: processing a long prompt to build the **key/value (KV) cache** is dominated by attention compute and KV-cache memory.

A widely used efficiency technique is **sliding-window attention (SWA)**, where each token attends only to the most recent window of *W* tokens. This reduces attention and KV-cache cost in SWA layers from scaling with sequence length to scaling with the window size.

However, the window size *W* in strong open models is typically conservative. For example, **Gemma 3 configurations on Hugging Face set `sliding_window=1024` and use a periodic local/global pattern (`sliding_window_pattern=6`, i.e., 5 local SWA layers for every 1 global full-attention layer)**. If we could safely shrink the local window (e.g., 1024→256), practitioners could reduce local-layer KV cache by ~4× and speed up prefill for many common prompt lengths (4k–32k tokens).

Step 3.5 Flash (arXiv:2602.10604) provides a concrete clue about what matters when SWA is used aggressively. Under its hybrid layout (S3F1: 3 SWA layers then 1 full-attention layer, with W=512), it reports that **head-wise post-attention output gating** is better than inserting **data-independent sink tokens** for stabilizing SWA (Table 2: 62.46→64.43 average score in a 100B-A10B ablation suite). Separately, the Gated Attention paper (arXiv:2505.06708) shows that gating the attention output after scaled dot-product attention (SDPA) greatly reduces attention-sink behavior and massive activations with minimal parameter overhead.

These results motivate a deployment-relevant retrofit question for existing checkpoints:

> When we shrink an existing hybrid local–global model’s SWA window far below its pretrained value, is it better to adapt with **data-dependent output gates** or with **data-independent sink slots**, given the same (tiny) tuning budget?

### The Problem

**Window shrink is valuable but can degrade quality.** In hybrid local–global models, most layers are local SWA layers and only a minority are global full-attention layers. Shrinking the local window from 1024 to 256 yields an immediate ~4× KV-cache reduction in local layers, but can cause perplexity and long-context coherence regressions.

**Existing fixes do not isolate the key design choice.**
- **Sink-token approaches** (e.g., StreamingLLM; SWAA-style keep-first-k) stabilize windowed/streaming behavior by providing globally visible “dump” positions for softmax probability mass.
- **Gating approaches** (e.g., Gated Attention; Step 3.5 Flash) give each attention head a learned **output suppression** mechanism, which can be viewed as a data-dependent alternative to sink tokens.

What is missing is a clean, automated experiment that compares these two stabilization mechanisms as *parameter-efficient adapters* for aggressive window shrink in a strong hybrid checkpoint.

### Key Insight and Hypothesis

**Key insight.** Under too-small windows, SWA heads are sometimes forced to attend within an uninformative local region. With softmax normalization, this can produce harmful residual updates even when “no useful token exists in the window.” A **data-independent sink** gives a fixed place to route probability mass, but cannot adapt to when the local window is actually informative. A **data-dependent output gate** can suppress the attention output selectively, allowing the model to effectively “do nothing” when local attention is unhelpful while still using local attention when it is helpful.

**Hypothesis.** After shrinking the SWA window aggressively (e.g., 1024→256) in a hybrid local–global LLM (Gemma 3 as proxy), **GateTune** (tuning only head-wise post-SDPA gates) will recover more of the original model’s long-text perplexity than **SinkTune** (tuning only a parameter-matched set of data-independent sink KV slots), under the same tuning compute and parameter budget.

Why we could be wrong:
- If the performance loss from window shrink is primarily due to *missing information* (tokens never attended), neither gating nor sinks can recover it; both methods may fail similarly.
- GateTune might collapse to near-constant scaling (not truly data-dependent), making it no better than generic rescaling.
- If global layers already carry most long-range information, some benchmarks may be insensitive to window shrink; this motivates evaluating **PG19 perplexity** (long-book language modeling), not only retrieval-heavy synthetic benchmarks.

---

## Proposed Approach

### Overview

We propose a minimal, controlled comparison between two adaptation mechanisms for SWA window shrink in a hybrid local–global LLM:

1. Choose a hybrid local–global checkpoint (Gemma 3) with default local window *W0* (1024).
2. Shrink the window in local layers to *W* = 256 (global layers remain full attention).
3. Freeze all base model weights.
4. Train a small adapter in one of two ways:
   - **SinkTune (data-independent)**: add learnable sink KV slots (constant across tokens) and tune only these slots.
   - **GateTune (data-dependent)**: add head-wise post-SDPA output gates and tune only gate parameters.
5. Evaluate long-text perplexity and prefill efficiency.

### Method Details

#### Target backbone: hybrid local–global Gemma 3
We use Gemma 3 as a proxy for modern hybrid local–global designs because it has an explicit SWA configuration and periodic local/global pattern. In Hugging Face configs, Gemma 3 models use `sliding_window=1024` and `sliding_window_pattern=6`.

Notation:
- \(d_{model}\): model hidden size
- \(L\): number of layers
- \(H\): number of attention heads
- \(d_h = d_{model} / H\): head dimension

We treat layers that use sliding-window masking as **local layers** and modify only these layers.

#### Window shrink
We modify the attention mask in local layers to enforce **W = 256** left-context sliding window (causal). Global layers keep full causal attention.

#### GateTune: head-wise post-SDPA output gating (data-dependent)
For each local layer \(\ell\), we insert head-wise sigmoid gating after the scaled dot-product attention (SDPA) output (following Qiu et al., 2025):

1. Compute the standard per-head attention output \(a^{(\ell)}_{t,h} \in \mathbb{R}^{d_h}\) (i.e., \(\text{softmax}(QK^T)V\) for head \(h\)).
2. Compute a gate scalar per token per head:
\[
 g^{(\ell)}_{t,h} = \sigma\left( (x^{(\ell)}_{t})^\top W^{(\ell)}_{gate} \right)_h,
\]
where \(x^{(\ell)}_{t} \in \mathbb{R}^{d_{model}}\) is the token hidden state after pre-norm and \(W^{(\ell)}_{gate} \in \mathbb{R}^{d_{model} \times H}\) is trainable.
3. Apply the gate:
\[
 o^{(\ell)}_{t,h} = g^{(\ell)}_{t,h} \cdot a^{(\ell)}_{t,h}.
\]

We freeze all base model parameters and tune only \(\{W^{(\ell)}_{gate}\}\) for local layers.

#### SinkTune: parameter-matched sink KV slots (data-independent)
We implement a **data-independent sink** baseline that is comparable in trainable parameter count to GateTune.

For each local layer \(\ell\) and head \(h\), we add \(S\) learnable sink KV slots:
- \(K^{(\ell)}_{sink} \in \mathbb{R}^{H \times S \times d_h}\)
- \(V^{(\ell)}_{sink} \in \mathbb{R}^{H \times S \times d_h}\)

At every token position, we append these sink slots to the local window’s key/value set before attention. The sink slots are **constant across tokens** (data-independent) but learnable.

**Parameter matching.**
- GateTune trainable parameters per local layer: \(d_{model} \times H\) (ignoring optional bias).
- SinkTune trainable parameters per local layer: \(2 \times H \times S \times d_h = 2 \times S \times d_{model}\).

We choose \(S = H/2\) when \(H\) is even, so SinkTune and GateTune have equal trainable parameter count per layer. If \(H\) is odd, we set \(S = \lfloor H/2 \rfloor\) and (for GateTune) freeze one gate column to equalize trainable parameters.

#### Tuning objective (same for both methods): KL-to-teacher distillation
To avoid the confound “extra parameters improve the LM regardless of shrink,” we tune adapters to **match the original model’s outputs**.

- Teacher: original Gemma 3 with window \(W0=1024\).
- Student: window-shrunk model (\(W=256\)) with either SinkTune or GateTune.

For each token, minimize KL divergence from teacher to student output distributions:
\[
 \mathcal{L} = \mathrm{KL}(p_{teacher}(\cdot\mid x_{\le t}) \;||\; p_{student}(\cdot\mid x_{\le t}))
\]
using teacher-forced next-token prediction.

#### Mechanistic diagnostics (no extra training runs)
We log two cheap diagnostics during evaluation:
1. **Gate variability**: variance of \(g^{(\ell)}_{t,h}\) over positions; if GateTune works via selective suppression, gates should not be nearly constant.
2. **Sink usage**: average attention mass assigned to sink slots in SinkTune vs average gate magnitude in GateTune.

### Key Innovations

- **A parameter-matched comparison** of data-dependent gating vs data-independent sink mechanisms as *retrofit adapters* for aggressive SWA window shrink (not trained-from-scratch architecture changes).
- **A distillation-based tuning protocol** that targets preserving the original checkpoint’s behavior, so wins are attributed to better handling of window shrink rather than generic fine-tuning gains.
- **A deployment-relevant evaluation** (PG19 long-book perplexity + prefill throughput) that stresses long-context stability and local coherence.

---

## Related Work

### Field Overview

**Hybrid local–global attention in LLMs.** Multiple recent LLMs interleave local SWA layers with periodic global full-attention layers to reduce long-context cost (e.g., Mistral-style SWA, Gemma 2/3, Step 3.5 Flash). These designs reduce KV-cache growth in most layers, but their local window sizes are typically conservative and chosen during pretraining. Common evaluation suites include LongBench (a benchmark of long-context question answering and reasoning tasks) and RULER (a synthetic long-context retrieval benchmark).

**Stabilizing sliding-window attention.** A separate line of work analyzes failure modes of windowed/streaming attention and proposes stabilizers. StreamingLLM shows that preserving a few prefix tokens (“attention sinks”) prevents collapse under rolling KV caches. SWAA and LightTransfer study how to retrofit full-attention checkpoints to windowed/hybrid inference without full pretraining, often relying on keep-first-k sink tokens and selective full-attention layers.

**Sink-free or gated attention.** Attention-sink analyses connect softmax normalization to sink tokens and instability. Gated Attention shows that post-attention output gating can eliminate attention sinks during training, improving stability and long-context extrapolation. Step 3.5 Flash reports that head-wise gating outperforms sink tokens in a hybrid SWA setting at scale.

Our proposal sits at the intersection: we test whether **gating vs sink mechanisms** is the better *retrofit* adaptation knob when a hybrid local–global model’s SWA window is shrunk aggressively.

### Related Papers

- **[Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)**: Hybrid SWA/full attention MoE model; reports head-wise gating beats sink tokens under S3F1.
- **[Gated Attention](./references/Gated-Attention-for-Large-Language-Models-Non-linearity-Sparsity-and-Attention-Sink-Free/meta/meta_info.txt)**: Systematic study of gating in attention; SDPA output gating reduces sinks and improves stability.
- **[StreamingLLM](./references/Efficient-Streaming-Language-Models-with-Attention-Sinks/meta/meta_info.txt)**: Shows attention sinks cause rolling-window KV-cache collapse; keeping a few sink tokens stabilizes streaming.
- **[SWAA](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/meta/meta_info.txt)**: Training-free toolkit for adapting full-attention checkpoints to SWA/hybrid inference.
- **[LightTransfer](./references/Your-Long-Context-LLM-is-Secretly-a-Hybrid-Model-with-Effortless-Adaptation/meta/meta_info.txt)**: Converts selected layers to streaming attention based on attention-pattern diagnostics.
- **[KL-guided hybrid distillation](./references/Distilling-to-Hybrid-Attention-Models-via-KL-Guided-Layer-Selection/meta/meta_info.txt)**: Selects which layers remain softmax when distilling to hybrid linear/softmax models.
- **[When Attention Sink Emerges](./references/When-Attention-Sink-Emerges-in-Language-Models-An-Empirical-View/meta/meta_info.txt)**: Empirical analysis of sink emergence and interventions.
- **[RAttention](./references/RAttention-Towards-the-Minimal-Sliding-Window-Size-in-Local-Global-Attention-Models/meta/meta_info.txt)**: Achieves small windows via architectural change (residual linear attention), not retrofit.
- **[Longformer](https://arxiv.org/abs/2004.05150)**: Sparse attention combining sliding windows with global tokens.
- **[BigBird](https://arxiv.org/abs/2007.14062)**: Sparse attention with global and random patterns for long sequences.
- **[Mistral 7B](https://arxiv.org/abs/2310.06825)**: Popular open LLM using sliding-window attention.
- **[Gemma 2](https://arxiv.org/abs/2408.00118)**: Hybrid attention with alternating local/global layers.
- **[Gemma 3 technical report](https://arxiv.org/abs/2503.19786)**: Local–global attention with long context.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**: Efficient attention kernels supporting windowed attention.
- **[vLLM](https://arxiv.org/abs/2309.06180)**: PagedAttention KV-cache management enabling efficient inference.
- **[YaRN](https://arxiv.org/abs/2309.00071)**: Context window extension via RoPE interpolation.
- **[LongRoPE](https://arxiv.org/abs/2402.13753)**: Extends RoPE context length with staged strategies.
- **[Lost in the Middle](https://arxiv.org/abs/2307.03172)**: Empirical study of how LLMs use long contexts.
- **[Massive Activations in LLMs](https://arxiv.org/abs/2402.17762)**: Connects attention sinks to activation outliers and instability.
- **[HySparse](https://arxiv.org/abs/2602.03560)**: Hybrid sparse attention with KV-cache sharing (illustrates broader long-context efficiency trends).

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Hybrid local–global SWA LLMs | Interleave SWA layers with periodic full-attention layers | Gemma 2/3, Mistral, Step 3.5 Flash | LongBench, RULER, model-specific evals | Window sizes chosen in pretraining; shrink can degrade quality |
| Windowed/streaming stabilizers | Preserve special tokens or modify inference cache rules | StreamingLLM, SWAA | PG19 PPL, LongBench-style evals | Often data-independent or heuristic |
| Sink-free / gated attention | Add output-side suppression via gating | Gated Attention, Step 3.5 Flash | PPL, MMLU, RULER | Mostly studied when trained from scratch, not retrofit shrink |
| Minimal-window architectures | Architectural changes enable small windows | RAttention | RULER + downstream suites | Not parameter-efficient retrofit |
| Hybrid distillation / selection | Select layers to keep softmax when changing attention type | KL-guided hybrid distillation | RULER + recall tasks | Changes attention type, not SWA window |

### Closest Prior Work

- **Step 3.5 Flash**: Shows head-wise gating > sink tokens under a fixed W=512 hybrid layout at large scale, but does not test retrofitting an existing checkpoint under aggressive window shrink.
- **Gated Attention**: Establishes gating reduces sinks and improves stability, but does not study gating as a window-shrink adapter or compare against parameter-matched sink-slot baselines.
- **StreamingLLM / SWAA**: Uses sink/prefix tricks to stabilize windowed/streaming inference, but does not compare against data-dependent gating under equal tuning budget.

**Novelty Kill Search Summary:** Searched for combinations including “gated attention + sliding window distillation”, “head-wise gating + sink token + sliding window attention”, “learnable sink token + sliding window attention”, and checked local KB for any work explicitly comparing **data-dependent post-SDPA gating vs data-independent sink mechanisms** as **parameter-matched adapters** for **SWA window shrink** (2025–2026). No close match was found as of 2026-02-21; query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Step 3.5 Flash | Uses gating instead of sink tokens in hybrid SWA/full attention | Trained from scratch; no retrofit shrink study | Retrofit a pretrained hybrid model with small adapter params | Isolates whether gating is a practical retrofit knob |
| Gated Attention | Adds post-SDPA gates; reduces sinks | Not evaluated as shrink adapter; no sink comparison | Compare GateTune vs parameter-matched SinkTune at same W | Tests gating specifically in a too-small-window regime |
| StreamingLLM | Uses sink tokens to stabilize streaming KV caches | Targets streaming cache; not adapter comparison | Use sink KV slots as a shrink adapter baseline | Separates “dump slot” vs “output suppression” mechanisms |
| SWAA / LightTransfer | Training-free / heuristic hybridization | Not focused on aggressive window shrink | Shrink W and adapt only small params | Targets a directly deployment-relevant hyperparameter (W) |
| RAttention | Architectural change enabling W=512 | Not retrofit | Keep architecture fixed; adapt small params only | Tests whether retrofit adapters can approach architectural benefits |

---

## Experiments

### Experimental Setup

**Baseline Ladder (for this setting):** This proposal uses **teacher-forced perplexity** (no sampling), so prompting baselines and best-of-N decoding are not applicable. The baseline ladder is therefore:
- **Upper bound teacher:** original Gemma 3 at \(W0=1024\).
- **Strong shrink baseline:** SinkTune at \(W=256\).
- **Proposed method:** GateTune at \(W=256\).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| `google/gemma-3-4b-pt` (preferred) or `google/gemma-3-4b-it` | 4B | https://huggingface.co/google/gemma-3-4b-pt | Gated license on Hugging Face; if unavailable, fall back to `google/gemma-3-1b-pt` for feasibility. Verification should confirm `sliding_window` support in Transformers for the chosen checkpoint. |

**Training Data (for adapter tuning):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| PG19 (train) | Long-form text for distillation tuning under window shrink | subsample (e.g., 10M tokens) | https://huggingface.co/datasets/pg19 | See dataset card |

**Resource Estimate** (fits `internal_context/resource_budget.md`):
- **Compute budget**: 200–600 A100 GPU-hours total.
  - Distillation requires a teacher forward pass (W0) and a student forward pass (W) per batch.
  - Two tunings (SinkTune, GateTune) x 3 seeds, plus evaluation.
- **GPU memory**: Gemma3 1B–4B with 32k context likely requires tensor parallelism; should fit in <= 16x A100-80GB.
- **API usage**: None.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| PG19 | Long-book language modeling benchmark | Perplexity (PPL); bucketed PPL by position | test | https://huggingface.co/datasets/pg19 | HF datasets + custom perplexity script |

Primary metrics:
- **PPL@8k**: perplexity when evaluating sequences with 8k effective context.
- **PPL@32k**: perplexity when evaluating sequences with 32k effective context.
- **Bucketed PPL**: PPL for token-position buckets (0–1k, 1k–4k, 4k–8k, 8k–32k) to identify where each method fails.

Secondary metric (efficiency, measured for all methods):
- **Prefill throughput** (tokens/s) at 32k input length on a fixed GPU configuration.

### Main Results

#### Results Table

(All entries are **TBD** and require running. We expect to report mean +/- std over 3 seeds for tuned methods.)

| Method | Base Model | Benchmark | PPL@8k (mean +/- std) | PPL@32k (mean +/- std) | Prefill tok/s (mean +/- std) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| Teacher (W0=1024) | Gemma3 | PG19 | TBD | TBD | TBD | - | Upper bound reference |
| **SinkTune (W=256)** | Gemma3 | PG19 | TBD | TBD | TBD | - | Parameter-matched sink KV slots, tuned |
| **GateTune (W=256)** | Gemma3 | PG19 | TBD | TBD | TBD | - | Head-wise post-SDPA gating, tuned |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Gate variability diagnostic | Measure var(g) across positions/layers | If GateTune works via selective suppression, gates should not be nearly constant |
| Naive shrink (required sanity check) | W=256, no adapter | Should be worse than SinkTune/GateTune; confirms that adaptation is needed |
| Fallback (only if both collapse) | Repeat SinkTune vs GateTune at W=512 | Avoid an uninformative catastrophic regime at W=256 |

### Experimental Rigor

- **Seeds:** 3 seeds for SinkTune and GateTune (e.g., seeds=[42, 123, 456]). Teacher is deterministic.
- **Fair comparison:** identical tuning token budget, optimizer, learning-rate schedule, and batching across methods.
- **Confounders & controls:**
  - Parameter-count confound is addressed by explicit parameter matching.
  - Data leakage is controlled by tuning on PG19 train and evaluating on PG19 test.
  - “Gate collapses to constant” is checked via the gate variability diagnostic.

---

## Success Criteria

**Hypothesis** (directional): GateTune recovers more of the W-shrink quality loss than SinkTune at the same shrunk window.

**Decision Rule** (concrete):
- **Proceed (support hypothesis):** On PG19 test at 32k context, GateTune achieves **lower PPL than SinkTune by at least 6% relative** (i.e., \(\mathrm{PPL}_{GateTune} / \mathrm{PPL}_{SinkTune} \le 0.94\)) and is not worse at 8k context (within std) across 3 seeds.
- **Pivot:** If GateTune and SinkTune are statistically indistinguishable at W=256 and both are substantially worse than the teacher (e.g., both have \(\mathrm{PPL} / \mathrm{PPL}_{teacher} > 1.5\)), rerun the same comparison at **W=512** (fallback).
- **Refute:** If GateTune is worse than SinkTune at 32k (\(\mathrm{PPL}_{GateTune} > \mathrm{PPL}_{SinkTune}\)), or if GateTune’s gate variability is near-zero and the method provides no measurable advantage.

---

## Impact Statement

If GateTune outperforms parameter-matched SinkTune for aggressive SWA window shrink, practitioners deploying hybrid local–global LLMs can reduce local-layer KV-cache memory and prefill cost more aggressively (e.g., ~4× in local layers) while preserving long-text perplexity, using only a small, automated distillation step rather than full pretraining.

---

## References

- [Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt) - StepFun Team, 2026
- [Gated Attention for Large Language Models](./references/Gated-Attention-for-Large-Language-Models-Non-linearity-Sparsity-and-Attention-Sink-Free/meta/meta_info.txt) - Qiu et al., 2025
- [Efficient Streaming Language Models with Attention Sinks](./references/Efficient-Streaming-Language-Models-with-Attention-Sinks/meta/meta_info.txt) - Xiao et al., 2023
- [SWAA: Sliding Window Attention Adaptation](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/meta/meta_info.txt) - Yu et al., 2025
- [Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation](./references/Your-Long-Context-LLM-is-Secretly-a-Hybrid-Model-with-Effortless-Adaptation/meta/meta_info.txt) - Zhang et al., 2024
- [Distilling to Hybrid Attention Models via KL-Guided Layer Selection](./references/Distilling-to-Hybrid-Attention-Models-via-KL-Guided-Layer-Selection/meta/meta_info.txt) - Li et al., 2025
- [When Attention Sink Emerges in Language Models: An Empirical View](./references/When-Attention-Sink-Emerges-in-Language-Models-An-Empirical-View/meta/meta_info.txt) - Gu et al., 2025
- [RAttention: Towards the Minimal Sliding Window Size in Local-Global Attention Models](./references/RAttention-Towards-the-Minimal-Sliding-Window-Size-in-Local-Global-Attention-Models/meta/meta_info.txt) - Wang et al., 2025
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Beltagy et al., 2020
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) - Zaheer et al., 2020
- [Mistral 7B](https://arxiv.org/abs/2310.06825) - Jiang et al., 2023
- [Gemma 2](https://arxiv.org/abs/2408.00118) - Gemma Team, 2024
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) - Gemma Team, 2025
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao et al., 2023
- [vLLM](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023
- [YaRN](https://arxiv.org/abs/2309.00071) - Peng et al., 2023
- [LongRoPE](https://arxiv.org/abs/2402.13753) - Chen et al., 2024
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) - Liu et al., 2023
- [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762) - Sun et al., 2024
- [HySparse](https://arxiv.org/abs/2602.03560) - Authors, 2026
