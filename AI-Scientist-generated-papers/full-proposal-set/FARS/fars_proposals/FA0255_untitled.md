# untitled

# FCBoost: Static RoPE-Frequency Masks for Channel-Wise Precision Boost in 2-bit KV Cache Quantization

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

For decoder-only Transformer **large language models (LLMs)**, inference cost at long context is often dominated by the **key–value (KV) cache**: for each generated token, the model attends over all prior tokens, requiring reading keys and values whose size grows linearly with sequence length. This KV cache bottleneck limits batch size, increases latency, and drives GPU memory cost.

A common remedy is **KV cache quantization**, which stores keys/values in low-bit integers (e.g., INT4/INT2) instead of FP16/BF16. Unlike weight quantization (static), KV quantization happens **online during generation**, making it harder to deploy and to keep accuracy stable at very low precision.

Recent work **Kitty** shows that pushing KV cache quantization to **2-bit** can cause large quality drops on reasoning tasks, but that most of the damage can be repaired by a **channel-wise precision boost**: keep a small fraction of key-cache channels at INT4 and quantize the rest to INT2 ([Kitty](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/meta/meta_info.txt)). Kitty selects boosted channels **dynamically** at runtime using a per-page magnitude heuristic.

Separately, **FASA** uncovers a structural property of RoPE-based attention: the head dimension decomposes into **frequency chunks** (RoPE pairs), and only a small subset of these pairs has high **Contextual Agreement (CA)** with full attention. These “dominant frequency chunks” are reported to be largely **task-invariant** and identifiable via a one-time offline procedure ([FASA](./references/FASA-Frequency-aware-Sparse-Attention/meta/meta_info.txt)).

This proposal tests whether these two observations are connected: if a small set of RoPE pairs largely determines attention patterns, then those same pairs may be exactly the **quantization-sensitive channels** that Kitty needs to preserve at higher precision.

### The Problem

Kitty’s channel-wise precision boost is compelling, but its **dynamic channel selection** introduces complexity:

- **Per-page selection compute**: for each quantization group (“page”), Kitty computes channel importance scores and selects top-K boosted channels at runtime.
- **Per-page metadata**: because boosted channels can differ across pages, the system must store per-page index mappings (e.g., `Boost_IDX_uint8`) to reconstruct which channels have higher bits.

Concretely, Kitty’s design stores a per-page index tensor of length D (head dimension) for boosted-channel lookup; this metadata scales with the number of pages (≈ sequence_length / page_size). FCBoost replaces this with a single static mask per (layer, KV head), making metadata O(1) in sequence length.

If a **static** (model-intrinsic) boosted-channel mask could match Kitty’s accuracy recovery, it would simplify kernels and reduce metadata, making 2-bit KV quantization easier to integrate into serving systems.

However, it is unclear whether the “critical channels” are:

- **Prompt/page-dependent outliers** (supporting Kitty’s dynamic selection), or
- **RoPE-structure-determined frequency pairs** (supporting a static mask).

Answering this changes what practitioners should implement: dynamic selection infrastructure vs. a one-time profiling step plus a fixed mask.

### Key Insight and Hypothesis

**Key insight:** In RoPE, each frequency chunk corresponds to a coupled pair of head-dimension channels. Attention logits decompose additively across these RoPE pairs. If only a small subset of RoPE pairs drives attention pattern fidelity (high CA in FASA), then quantization noise on those pairs should disproportionately perturb attention and downstream reasoning.

**Hypothesis:** In a Kitty-style 2-bit KV quantization pipeline, replacing Kitty’s runtime magnitude-based boosted-channel selection with a **static RoPE-frequency mask** derived from FASA’s Contextual Agreement (CA) will recover **most of Kitty’s accuracy gain** at the same boost ratio, while reducing per-page selection compute and per-page index metadata.

Why this could be wrong: Kitty’s sensitivity may be dominated by **prompt/page-specific high-magnitude outliers** that do not align with CA-ranked RoPE pairs; in that case a static CA-derived mask will underperform Kitty’s dynamic heuristic.

---

## Proposed Approach

### Overview

We propose **FCBoost**, a drop-in modification to Kitty’s channel-wise precision boost:

1. **One-time offline profiling**: compute a CA score per RoPE pair for each (layer, KV head), and select the top-F RoPE pairs.
2. **Static boosted-channel mask**: treat the selected RoPE pairs as the only channels eligible for INT4, and quantize all other channels to INT2.
3. **Inference**: run the same KIVI/Kitty execution pipeline (sink tokens in FP16; sliding-window FP16 for recent values; paged quantization), but **skip per-page magnitude scoring and top-K selection**.

We target the same setting as Kitty: **Key cache per-channel quantization**, **Value cache per-token quantization**, FP16 sink tokens and local window, and 2-bit storage for most channels.

### Method Details

#### Background: Kitty channel-wise precision boost
Kitty’s importance score for channel i is the average magnitude across tokens in a page (Eq. 2 in Kitty):

\[ s_i = \frac{1}{T}\sum_{t=1}^{T} |x_{i,t}|. \]

Kitty then boosts the top-K channels to INT4 and stores the remaining channels in INT2, with per-page index metadata to reconstruct which channels were boosted ([Kitty](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/meta/meta_info.txt)).

#### Step 1: RoPE pair contextual agreement scores (FASA)
FASA defines a **frequency chunk (FC)** as one RoPE pair (2 coupled dimensions). For a head (layer l, query head h), FASA defines full-head scores \(\alpha^{l,h}\) and single-FC scores \(\alpha^{(i)}_{l,h}\), and the **Contextual Agreement** metric as top-K index overlap (Eq. 4 in FASA):

\[\mathrm{CA}^{l,h,i}_K(q_t,K_{1:t}) = \frac{|\mathrm{TopK}(\alpha^{l,h}) \cap \mathrm{TopK}(\alpha^{(i)}_{l,h})|}{K}.\]

Offline, FASA averages CA over a small dataset and selects the **dominant FC indices** \(I_{\mathrm{dom}}\) via Top-K on mean CA (Algorithm 1 in FASA; also reproduced in FASA Appendix section “C ADDITIONAL EXPERIMENTAL RESULTS”) ([FASA](./references/FASA-Frequency-aware-Sparse-Attention/meta/meta_info.txt)).

#### Step 2: Mapping CA-selected RoPE pairs to key-cache channels
For standard RoPE implementations, the head dimension is arranged as paired channels \([0,1], [2,3], \dots\). We map RoPE pair i to channel indices \(\{2i, 2i+1\}\) in the Key cache tensor.

Because modern models use **grouped-query attention (GQA)**, KV caches are indexed by **KV heads** while CA in FASA is defined per **query head**. For a KV head \(h_{kv}\) shared by query heads \(\mathcal{H}(h_{kv})\), we compute a KV-head CA score for pair i by averaging:

\[ \overline{\mathrm{CA}}^{l,h_{kv},i} = \frac{1}{|\mathcal{H}(h_{kv})|}\sum_{h\in\mathcal{H}(h_{kv})} \overline{\mathrm{CA}}^{l,h,i}. \]

We then select the top-F RoPE pairs per (layer, KV head):

\[ I^{l,h_{kv}}_{\mathrm{FCBoost}} = \mathrm{TopF}(\overline{\mathrm{CA}}^{l,h_{kv},i}). \]

#### Step 3: Static mixed-precision key quantization
Let head dimension be D (e.g., D=128 for Qwen3-8B KV heads). Kitty boosts a fraction r of channels (e.g., r=12.5%), so \(K=r\cdot D\) channels are INT4.

FCBoost boosts RoPE pairs, so we set:

- boosted channels \(K=r\cdot D\)
- boosted RoPE pairs \(F=K/2\)

For Qwen3-8B (D=128), r=12.5% implies K=16 channels and F=8 RoPE pairs.

During page quantization:
- channels in \(\cup_{i\in I^{l,h_{kv}}_{\mathrm{FCBoost}}}\{2i,2i+1\}\) are stored at INT4
- all other channels are stored at INT2

We keep Kitty’s other accuracy-preserving choices fixed (sink tokens, value local window) to isolate the effect of **channel selection strategy**.

#### Premise gate / analysis: CA mask vs. magnitude mask overlap
To understand whether CA is acting as a proxy for Kitty’s magnitude heuristic, we compute on a small prompt set:

- **Jaccard overlap** between the FCBoost static mask and the “global magnitude mask” formed by the most frequently selected top-K channels across pages.
- **Rank correlation** (Spearman) between per-pair mean CA and per-pair mean magnitude.

We treat this as a primary analysis result because it distinguishes three outcomes:

- high overlap + FCBoost matches Kitty 
→ CA is a valid static proxy for magnitude selection
- low overlap + FCBoost matches Kitty 
→ static selection works, but via a different mechanism than magnitude
- low overlap + FCBoost fails 
→ evidence that dynamic selection is necessary

### Key Innovations

- **Mechanism-driven static channel selection**: use RoPE pair importance (CA) rather than runtime magnitudes to decide which key-cache channels deserve extra bits.
- **GQA-consistent aggregation**: compute boosted channels per KV head by aggregating CA over the query heads that share that KV head.
- **Simplification bet**: remove per-page boosted-channel selection while aiming to preserve most of the 2-bit accuracy recovery.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) KV cache quantization for long-context inference, (ii) mixed-precision allocation strategies (token-, layer-, or channel-wise), and (iii) RoPE-structure-aware compression.

A recurring pattern in KV compression is that “uniformly low precision everywhere” fails, and most successful methods identify a **small subset** of tokens/channels/layers that must be preserved with higher fidelity. The key open question FCBoost targets is whether the “critical subset” at **2-bit** is fundamentally **dynamic** (prompt/page-dependent) or can be approximated by a **static structural prior** derived from RoPE.

### Related Papers

- **[Kitty](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/meta/meta_info.txt)**: Introduces dynamic channel-wise INT4 boosting for 2-bit KV quantization; our starting point and main baseline.
- **[KIVI](./references/KIVI-A-Tuning-Free-Asymmetric-2bit-Quantization-for-KV-Cache/meta/meta_info.txt)**: Tuning-free asymmetric quantization (K per-channel, V per-token) and sink/local FP16 buffers; the core quantization backbone Kitty builds on.
- **[FASA](./references/FASA-Frequency-aware-Sparse-Attention/meta/meta_info.txt)**: Discovers dominant RoPE frequency chunks via Contextual Agreement; provides the static signal FCBoost uses.
- **[EliteKV](./references/EliteKV-Scalable-KV-Cache-Compression-via-RoPE-Frequency-Selection-and-Joint-Low-Rank-Projection/meta/meta_info.txt)**: RoPE-frequency selective KV compression with low-rank projections; related in using RoPE frequency structure, but targets dimensionality reduction rather than bit allocation.
- **[KV-Latent](https://arxiv.org/abs/2507.11273)**: Dimensional-level KV cache reduction with frequency-aware RoPE modifications; complementary axis (reduce dimensions vs allocate bits).
- **[RAP](https://arxiv.org/abs/2602.02599)**: RoPE-aligned pair pruning for commutative low-rank factorization; shows RoPE pairs are the atomic unit for structural compression.
- **[MixKVQ](./references/MixKVQ-Query-Aware-Mixed-Precision-KV-Cache-Quantization/meta/meta_info.txt)**: Query-aware channel-wise mixed precision for KV cache; represents dynamic precision allocation at inference.
- **[KVmix](./references/KVmix-Gradient-Based-Layer-Importance-Aware-Mixed-Precision-Quantization/meta/meta_info.txt)**: Gradient-based layer-wise mixed precision KV quantization; static offline profiling at layer granularity.
- **[KVTuner](https://arxiv.org/abs/2502.04420)**: Offline search for layer-wise KV bitwidth configurations; another static profiling approach.
- **[KVQuant](https://arxiv.org/abs/2401.18079)**: Ultra-long-context KV quantization with pre-RoPE quantization, non-uniform quantization, and dense+sparse outlier handling.
- **[RotateKV](https://arxiv.org/abs/2501.16383)**: Outlier-aware rotations (including pre-RoPE rotation) to improve 2-bit KV quantization robustness.
- **[PatternKV](./references/PatternKV-Flattening-KV-Representation-Expands-Quantization-Headroom/meta/meta_info.txt)**: Pattern-aligned residual quantization to flatten KV distributions; aims to make low-bit quantization easier.
- **[GEAR](https://arxiv.org/abs/2403.05527)**: KV compression via quantization plus low-rank and sparse error correction.
- **[ZipCache](https://arxiv.org/abs/2405.14256)**: Mixed-precision KV quantization with normalized attention scores to identify salient tokens.
- **[SKVQ](https://arxiv.org/abs/2405.06219)**: Sliding-window KV quantization with channel reordering and clipped quantization.
- **[LogQuant](https://arxiv.org/abs/2503.19950)**: Log-distributed token selection + quantization for KV cache; focuses on token sparsity patterns.
- **[StreamingLLM](https://arxiv.org/abs/2309.17453)**: Attention-sink based streaming cache; motivates keeping initial tokens in higher precision.
- **[H2O](https://arxiv.org/abs/2306.14048)**: Heavy-hitter-based KV eviction policy; shows non-uniform importance of tokens.
- **[SnapKV](https://arxiv.org/abs/2404.14469)**: Prefill-time attention profiling for token retention; related to importance profiling.
- **[PyramidKV](https://arxiv.org/abs/2406.02069)**: Layer-wise pyramidal KV budget allocation for eviction/compression.
- **[Quest](https://arxiv.org/abs/2406.10774)**: Query-aware page retrieval for KV caches; a system-oriented view on paging.
- **[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)**: Block-based KV cache management for high-throughput serving; relevant for integrating quantized pages.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**: IO-aware attention kernels; the baseline attention implementation used in Kitty’s eval stack.
- **[QuaRot](https://arxiv.org/abs/2404.00456)**: Rotation-based quantization to remove activation outliers; complementary to KV-only schemes.
- **[MiniKV](https://arxiv.org/abs/2402.02242)**: Layer-discriminative low-bit KV cache allocation; another static mixed-precision approach.
- **[BitDecoding](https://arxiv.org/abs/2501.12974)**: Hardware-oriented low-bit KV cache decoding with tensor cores; system baseline for low-bit execution.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Asymmetric KV quantization (K channel-wise, V token-wise) | Quantize K and V along different axes to match their statistics | KIVI, KVQuant | LM PPL, LongBench, GSM8K | 2-bit often unstable without extra tricks |
| Error-mitigation via transforms / residuals | Modify KV representation to reduce quantization error | RotateKV, QuaRot, PatternKV, GEAR | PPL, LongBench, math/code | Often needs custom kernels / extra compute |
| Mixed-precision allocation (static) | Allocate different bitwidths per layer/token based on offline profiling | KVmix, KVTuner, MiniKV | LongBench, GSM8K | Coarse granularity may miss head/channel effects |
| Mixed-precision allocation (dynamic) | Allocate bits at runtime using query-/page-dependent signals | Kitty, MixKVQ, ZipCache | Reasoning, long-context tasks | Runtime overhead and metadata; policy complexity |
| RoPE-structure-aware compression | Treat RoPE pairs/frequencies as atomic units for compression | EliteKV, KV-Latent, RAP, FASA | LongBench, AIME, downstream eval | Often targets dimension/token selection, not bit allocation |

### Closest Prior Work

1) **Kitty** ([Kitty](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/meta/meta_info.txt))
- What it does: Achieves strong 2-bit KV quantization by boosting a small fraction of key-cache channels to INT4, selected dynamically per page using a magnitude heuristic.
- Key limitation (for this proposal): Boosted channels are prompt/page-dependent, requiring per-page selection compute and per-page index metadata.
- Why we differ: FCBoost keeps the same 2-bit system design, but replaces dynamic selection with a static RoPE-frequency mask derived from CA.

2) **FASA** ([FASA](./references/FASA-Frequency-aware-Sparse-Attention/meta/meta_info.txt))
- What it does: Shows that a small set of RoPE pairs has high Contextual Agreement with full attention and can be identified via one-time offline profiling.
- Key limitation (for this proposal): FASA uses dominant RoPE pairs for **token selection / sparse attention**, not for quantization robustness.
- Why we differ: We reuse CA as a signal for **which key-cache channels should retain extra bits**.

3) **MixKVQ** ([MixKVQ](./references/MixKVQ-Query-Aware-Mixed-Precision-KV-Cache-Quantization/meta/meta_info.txt))
- What it does: Uses query-aware signals to assign mixed precision at channel granularity.
- Key limitation: Still dynamic at inference and can incur runtime overhead; selection rule is not RoPE-structural.
- Why we differ: FCBoost is static and RoPE-pair-aligned.

4) **EliteKV** ([EliteKV](./references/EliteKV-Scalable-KV-Cache-Compression-via-RoPE-Frequency-Selection-and-Joint-Low-Rank-Projection/meta/meta_info.txt))
- What it does: Selects RoPE frequency components and applies low-rank projections to reduce KV cache size.
- Key limitation: Requires uptraining and targets dimensionality reduction rather than extreme low-bit quantization.
- Why we differ: FCBoost is post-training and uses frequency structure only to allocate bits under 2-bit storage.

5) **RotateKV / KVQuant** ([RotateKV](https://arxiv.org/abs/2501.16383), [KVQuant](https://arxiv.org/abs/2401.18079))
- What they do: Improve quantization by outlier-aware transforms (rotation) and pre-RoPE quantization.
- Key limitation: Do not address the “which channels get extra bits?” question in Kitty’s channel-wise boost framing.
- Why we differ: FCBoost is a selection strategy for channel-wise boosting, orthogonal to pre-RoPE quantization and rotations.

**Novelty Kill Search Summary:** We searched for the exact combination “RoPE frequency chunk + KV cache quantization + mixed precision/channel-wise boost” with multiple query variants (including OpenReview and GitHub) and found no prior work using **FASA-style CA/dominant RoPE pairs** to choose the boosted channels for 2-bit KV quantization. Queries and outcomes are logged in `notes.md` (2026-02-23).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [Kitty](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/meta/meta_info.txt) | Dynamic per-page top-K channels boosted to INT4 based on magnitude | Runtime selection + per-page metadata; heuristic not structural | Replace magnitude selection with static CA-derived RoPE-pair mask | If quantization sensitivity is RoPE-structural, static mask matches accuracy with simpler system |
| [FASA](./references/FASA-Frequency-aware-Sparse-Attention/meta/meta_info.txt) | CA identifies dominant RoPE pairs for token selection | Not applied to quantization | Use CA to decide which channels get extra bits | CA directly measures contribution of RoPE pairs to attention patterns, which should predict quantization sensitivity |
| [MixKVQ](./references/MixKVQ-Query-Aware-Mixed-Precision-KV-Cache-Quantization/meta/meta_info.txt) | Query-aware mixed precision at channel level | Dynamic; overhead; not RoPE-aligned | Static mask per (layer, KV head) | Eliminates per-query selection while using a mechanistic prior |
| [EliteKV](./references/EliteKV-Scalable-KV-Cache-Compression-via-RoPE-Frequency-Selection-and-Joint-Low-Rank-Projection/meta/meta_info.txt) | RoPE frequency selection + low-rank projection | Needs training; changes KV representation | Keep KV representation; only change bit allocation | Post-training drop-in for serving; preserves existing system stack |
| [RotateKV](https://arxiv.org/abs/2501.16383) | Rotation + pre-RoPE techniques for robust low-bit KV | Doesn’t allocate bits within head dim | Complementary (can be combined later) | If FCBoost works, it can stack with rotation/pre-RoPE gains |

---

## Experiments

### Experimental Setup

We will implement FCBoost by modifying the open Kitty codebase:
- Kitty repo: https://github.com/Summer-Summer/Kitty

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3 | 8B | https://huggingface.co/Qwen/Qwen3-8B | Matches Kitty’s main reported results; GQA with 32 Q heads and 8 KV heads |

**Training Data (if applicable):**
- No training data needed — inference-only modification.

**Other Resources (evaluation):**
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness (also used by Kitty)
- AIME24/AIME25 evaluation scripts (as used in Kitty’s Table 4; either via lm-eval tasks or Kitty’s released harness)

**Baseline Ladder (REQUIRED):**
- **FP16 KV cache** (K16V16): reference upper bound.
- **2-bit KV without channel boost but with sink tokens** (KIVI-KV2*): isolates “2-bit + sink token preservation” effect.
- **2-bit KV with dynamic channel-wise precision boost** (Kitty): strongest published 2-bit baseline.
- **2-bit KV with static CA-based channel boost** (FCBoost): proposed method.

(Prompting and inference-time-scaling baselines are not applicable here because all methods use the **same prompts** and **same decoding settings**; the only experimental variable is KV cache quantization and channel selection.)

**Resource Estimate** (rough, assuming 1×A100 80GB for Qwen3-8B evaluation):
- Offline CA profiling: ≤2 GPU-hours (single long prompt, limited steps).
- Accuracy evaluation (main): AIME24+25 (60 problems) × 3 methods × 3 seeds, max_gen=32k.
  - If average generated length is ~4k tokens, this is typically **tens** of GPU-hours.
  - If average generated length is ~32k tokens (worst case), this can approach **low hundreds** of GPU-hours.
- Total budget: conservatively **≤300 GPU-hours**, within the 768 GPU-hour cap.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AIME24 / AIME25 | Competition math problems requiring long-chain reasoning (30 problems each; answers are integers); Kitty evaluates with max_gen=32k | Accuracy (% correct; higher is better) | test | https://huggingface.co/datasets/math-ai/aime24 and https://huggingface.co/datasets/math-ai/aime25 | Kitty harness or lm-eval task if available |
| (Optional short-context check) GSM8K | Grade-school math word problems (1319 test questions); used in Kitty Table 3 | Accuracy (% correct; higher is better) | test | https://huggingface.co/datasets/gsm8k | lm-eval task `gsm8k` |

### Main Results

#### Results Table (baseline numbers from Kitty)

| Method | Base Model | Benchmark | Metric (mean±std) | Source | Notes |
|---|---|---|---|---|---|
| FP16 KV (KV16) | Qwen3-8B | AIME24/25 @32k | AIME24 71.67±15.00; AIME25 66.00±7.33 (avg=68.84; higher is better) | [Kitty](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/sections/Extended%20Results%20on%20Longer%20Context%20Length.md) | Table 4 |
| KIVI-KV2 | Qwen3-8B | AIME24/25 @32k | AIME24 57.00±7.00; AIME25 52.33±9.00 (avg=54.67) | [Kitty](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/sections/Extended%20Results%20on%20Longer%20Context%20Length.md) | Table 4 |
| KIVI-KV2* (sink FP16) | Qwen3-8B | AIME24/25 @32k | AIME24 67.67±9.00; AIME25 57.67±9.00 (avg=62.67) | [Kitty](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/sections/Extended%20Results%20on%20Longer%20Context%20Length.md) | Table 4 |
| Kitty (dynamic boost) | Qwen3-8B | AIME24/25 @32k | AIME24 70.67±7.33; AIME25 59.67±10.33 (avg=65.17) | [Kitty](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/sections/Extended%20Results%20on%20Longer%20Context%20Length.md) | Table 4; Kitty-Pro (25% boost) not reported for AIME@32k; interpret Kitty as the 12.5% boost variant |
| **FCBoost (static CA mask)** | Qwen3-8B | AIME24/25 @32k | **TBD** | This work | Same boost ratio as Kitty; to be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Random static mask | Boost a random fixed set of RoPE pairs (same F) | Worse than FCBoost; tests whether “any static mask” works |
| Overlap analysis | Compute Jaccard(CA mask, magnitude mask) and Spearman(CA, magnitude) on calibration prompts | Helps interpret whether CA is proxying magnitude |

### Experimental Rigor

- **Seeds / stochasticity**: Follow Kitty’s evaluation protocol (temperature=0.6, top_p=0.95, top_k=20) and run **3 seeds** (`seeds=[1,2,3]`), reporting mean±std. (Kitty reports 3–10 repeats.)
- **Controls**: Keep all non-selection components identical across Kitty and FCBoost (same sink tokens S=32, group size G=128, local V window R=128, same quantization kernels where possible).
- **Sanity check**: Reproduce Kitty’s published KV16/KIVI/Kitty numbers on AIME24/25 within reported deviation before evaluating FCBoost.
- **Data leakage**: Benchmarks may be partially present in pretraining data for Qwen3; we follow standard practice and treat reported numbers as benchmark comparisons (no additional fine-tuning).

---

## Success Criteria

**Hypothesis (directional):** FCBoost’s static CA-derived mask recovers most of Kitty’s long-context accuracy gain over KIVI-KV2* at the same boost ratio, indicating that quantization-sensitive channels are largely RoPE-structural rather than per-page dynamic.

**Decision Rule (concrete):**

- **Proceed** if, on Qwen3-8B AIME24/25 @32k:
  1) FCBoost achieves **≥90%** of Kitty’s accuracy recovery over KIVI-KV2*:
     - Let \(\Delta_{\mathrm{Kitty}} = \mathrm{Acc(Kitty)} - \mathrm{Acc(KIVI\text{-}KV2*)}\).
     - Require \(\mathrm{Acc(FCBoost)} - \mathrm{Acc(KIVI\text{-}KV2*)} \ge 0.9\,\Delta_{\mathrm{Kitty}}\).
  2) FCBoost is not worse than Kitty by more than **1 absolute point** (mean across seeds).
  3) FCBoost removes per-page magnitude selection compute (measured by timing the page-quantization step) and enables replacing per-page boost indices with a fixed per-(layer, KV head) mapping (implementation-dependent).

- **Pivot** if FCBoost is moderately worse (within 1–2 points) but overlap analysis shows high CA–magnitude alignment (suggesting GQA aggregation or CA estimation is the issue). Next attempt: adjust the CA aggregation rule (e.g., max over query heads instead of mean) without changing boost ratio.

- **Refute** if FCBoost underperforms Kitty by >2 points on AIME24/25 @32k and overlap analysis shows low CA–magnitude alignment (supporting that prompt/page-dependent outliers drive 2-bit sensitivity).

---

## Impact Statement

If FCBoost works, it provides a simpler path to production-grade 2-bit KV cache quantization: practitioners can precompute a per-layer/head boosted-channel mask once, avoid runtime channel selection, and reduce per-page metadata in mixed-precision KV page layouts. If it fails, it strengthens the case that **dynamic** channel selection is necessary at 2-bit, motivating investment in efficient runtime heuristics and kernels.

---

## References

- [Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost](./references/Kitty-2bit-KV-Quantization-with-Dynamic-Channel-wise-Precision-Boost/meta/meta_info.txt) - Xia et al., 2025
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](./references/KIVI-A-Tuning-Free-Asymmetric-2bit-Quantization-for-KV-Cache/meta/meta_info.txt) - Liu et al., 2024
- [FASA: Frequency-aware Sparse Attention](./references/FASA-Frequency-aware-Sparse-Attention/meta/meta_info.txt) - Wang et al., 2026
- [EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection](./references/EliteKV-Scalable-KV-Cache-Compression-via-RoPE-Frequency-Selection-and-Joint-Low-Rank-Projection/meta/meta_info.txt) - Zhou et al., 2025
- [KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](./references/KVmix-Gradient-Based-Layer-Importance-Aware-Mixed-Precision-Quantization/meta/meta_info.txt) - Li et al., 2025
- [MixKVQ: Query-Aware Mixed-Precision KV Cache Quantization for Long-Context Reasoning](./references/MixKVQ-Query-Aware-Mixed-Precision-KV-Cache-Quantization/meta/meta_info.txt) - Zhang et al., 2025
- [PatternKV: Flattening KV Representation Expands Quantization Headroom](./references/PatternKV-Flattening-KV-Representation-Expands-Quantization-Headroom/meta/meta_info.txt) - Zhang et al., 2025
- [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079) - Hooper et al., 2024
- [RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations](https://arxiv.org/abs/2501.16383) - Su et al., 2025
- [KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization](https://arxiv.org/abs/2502.04420) - Li et al., 2025
- [ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification](https://arxiv.org/abs/2405.14256) - Zhuang et al., 2024
- [SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models](https://arxiv.org/abs/2405.06219) - (authors), 2024
- [GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference](https://arxiv.org/abs/2403.05527) - Kang et al., 2024
- [LogQuant: Log-Distributed 2-Bit Quantization of KV Cache](https://arxiv.org/abs/2503.19950) - (authors), 2025
- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) - Xiao et al., 2024
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) - Zhang et al., 2023
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) - Li et al., 2024
- [PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling](https://arxiv.org/abs/2406.02069) - Zhang et al., 2024
- [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2406.10774) - Tang et al., 2024
- [PagedAttention: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) - Dao, 2023
- [RAP: KV-Cache Compression via RoPE-Aligned Pruning](https://arxiv.org/abs/2602.02599) - Xin et al., 2026
- [KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](https://arxiv.org/abs/2507.11273) - Shi et al., 2025
- [QuaRot: Outlier-Free 4-bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) - Ashkboos et al., 2024
- [MiniKV: Pushing the Limits of LLM Inference via 2-bit Layer-Discriminative KV Cache](https://arxiv.org/abs/2402.02242) - Sharma et al., 2024
- [BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-bit KV Cache](https://arxiv.org/abs/2501.12974) - Du et al., 2025
