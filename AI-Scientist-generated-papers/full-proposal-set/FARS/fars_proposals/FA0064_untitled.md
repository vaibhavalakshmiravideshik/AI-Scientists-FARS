# untitled

# NLL-Guided Full-Attention Layer Selection for Training-Free Sliding-Window Adaptation

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Serving large language models (LLMs) on long prompts is expensive because standard Transformer self-attention scales quadratically with prompt length. Many practical workloads are **prefill-heavy** (long prompt ingestion followed by a relatively short answer), such as multi-document question answering or retrieval-augmented generation.

A common efficiency idea is to replace full attention with **sliding-window attention (SWA)**, where each token attends only to a fixed recent window (linear in sequence length). However, naively swapping an LLM pretrained with full attention into SWA can catastrophically degrade long-context quality.

**SWAA** proposes a training-free toolkit for adapting full-attention LLMs to SWA without full pretraining (**[SWAA](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/meta/meta_info.txt)**). Two components are especially effective together:
1) **Full-attention decode**: use SWA only while processing the prompt (“prefill”), but use full attention when generating the answer (“decode”) (**[SWAA](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/meta/meta_info.txt)**).
2) **Hybrid layers**: keep full attention in some layers during prefill, and use SWA in the rest.

This combination recovers much of the accuracy of full attention while reducing prefill computation. But it leaves an important practical question unresolved.

### The Problem

In SWAA, the **choice of which layers keep full attention during prefill** matters dramatically. For example, on Qwen3-4B-Thinking with a 2k window, keeping half the layers full-attention in a simple periodic pattern yields 65.0 accuracy on LongMemEval_24k, while keeping only one quarter yields 53.2–54.2 depending on the offset (**[SWAA Table 1](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/sections/SWA%20Adaptation%20Without%20Fine-tuning.md)**). The best offset is model-dependent, and SWAA reports that the attention-pattern heuristic in LightTransfer can be unstable across model families (**[SWAA Appendix F](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/sections/F%20Results%20of%20LightTransfer.md)**; **[LightTransfer](./references/Your-Long-Context-LLM-is-Secretly-a-Hybrid-Model-with-Effortless-Adaptation/meta/meta_info.txt)**).

If 25% full-attention layers could match the accuracy of 50% full-attention layers, it would move the deployment Pareto frontier: SWAA’s own efficiency table suggests that reducing the full-attention layer budget from 1/2 to 1/4 under full-attention decode improves throughput (5.43k vs 4.72k tokens/s) in a long-prompt setting (**[SWAA Table 6 extract](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/sections/Appendix-D-Table-6-Inference-Efficiency-extracted.md)**).

The technical problem is therefore:

> Given a fixed budget of full-attention layers during prefill (e.g., 25% of layers), how can we select which layers should keep full attention so that long-context quality approaches a much larger budget (e.g., 50%), without any fine-tuning?

This is not obviously solved by stronger prompting or longer answers: SWAA already uses greedy decoding and shows the gap persists between 1/4 and 1/2 full-attention layer budgets under identical prompting and decoding settings.

### Key Insight and Hypothesis

**Key insight.** In training-free SWA adaptation with full-attention decode, the main role of the prefill pass is to write a useful key–value (KV) cache that the decode pass can query with full attention. Therefore, the “important” layers to keep full-attention in prefill should be the ones whose full-attention prefill computation most improves the model’s ability to predict the **answer tokens**, conditioned on the long prompt.

**Hypothesis.** A simple **teacher-forced loss signal** can identify these layers: if we temporarily switch one layer’s prefill attention from SWA to full attention and the negative log-likelihood (NLL) on answer tokens decreases, then that layer is more critical for long-range prompt-to-answer information flow. Selecting the top 25% layers by this per-layer ΔNLL should yield a 1/4-full-attention hybrid that approaches the 1/2-full-attention periodic hybrid on LongMemEval_24k.

We could be wrong for two main reasons:
1) **Non-compositionality**: layers may matter only in combinations (a chain effect), so selecting top layers individually may not produce a good set.
2) **Confounded signal**: per-layer ΔNLL might mostly reflect general layer sensitivity (layers that matter even when SWA has no effect), not long-range dependence.

---

## Proposed Approach

### Overview

We propose **NLL-guided full-attention layer selection** for SWAA-style inference:

1. Fix a SWAA configuration with **full-attention decode** and a target SWA window size (e.g., 2k).
2. Use a small long-context calibration set to score each layer by how much switching that layer’s **prefill** attention from SWA to full attention reduces teacher-forced NLL on answer tokens.
3. Select the top-K layers (K = 25% of layers) as full-attention layers during prefill.
4. Run inference with the resulting hybrid mask: selected layers use full attention during prefill; all other layers use SWA during prefill; all layers use full attention during decode.

This is a one-time, model-specific calibration step (no gradient updates).

### Method Details

**Setting.** Let an example be tokenized into a prompt segment \(x_{1:m}\) and an answer segment \(y_{1:n}\). We define a stage-aware attention mask that emulates full-attention decode:

- For prompt tokens \(x\): attention is sliding-window, optionally with **keep-first-k attention sinks** (allow every token to also attend to the first *k* tokens, which stabilizes attention distributions for SWA-adapted pretrained models; cf. StreamingLLM / SWAA keep-first-k).
- For answer tokens \(y\): attention is full causal attention over \(x\) and prior \(y\).

**Layer toggle.** For each Transformer layer \(\ell\), define two masks that differ only at that layer during the prompt segment:
- Base: layer \(\ell\) uses SWA on the prompt.
- Toggled: layer \(\ell\) uses full attention on the prompt.

All other layers use the base configuration.

**Per-layer score.** Using teacher forcing on the answer tokens, compute:

\[
\Delta_\ell = \mathcal{L}_{\text{ans}}(\text{base}) - \mathcal{L}_{\text{ans}}(\text{toggled}_\ell),
\]

where \(\mathcal{L}_{\text{ans}}\) is mean NLL on the answer segment. Larger \(\Delta_\ell\) indicates that full-attention prefill at layer \(\ell\) improves predictive quality more.

**Selection rule.** Select the top-K layers by \(\Delta_\ell\) (K = 1/4 of layers). This yields a layer set \(S\). During deployment, layers in \(S\) use full attention in prefill; other layers use SWA in prefill.

**Cheap de-confounding control (analysis, not part of the method).** Repeat the same scoring procedure on short prompts whose length is within the SWA window (so SWA and full attention are equivalent). If long-prompt scores are highly correlated with short-prompt scores, interpret the signal as dominated by generic layer sensitivity.

### Key Innovations

- **Training-free layer selection signal for SWA adaptation**: use teacher-forced answer-token NLL to score which layers must keep full attention during prefill under full-attention decode.
- **Budget-focused objective**: explicitly targets the low-full-attention-layer regime (25%) where SWAA shows meaningful quality loss.
- **Minimal calibration cost**: requires only forward passes on a small calibration set (no fine-tuning, no search over many masks).

---

## Related Work

### Field Overview

Long-context efficiency methods for Transformers fall into several families:

1) **Local/sparse attention operators** (SWA, block-sparse, token selection) that reduce prefill computation but can harm long-range retrieval if applied to full-attention pretrained models.
2) **Hybrid attention architectures** that mix full attention with cheaper mechanisms across layers, heads, or tokens (often trained from scratch, but sometimes adapted post-hoc).
3) **KV-cache reduction and eviction** methods (intra-layer eviction such as heavy-hitter retention, inter-layer KV sharing/merging) that reduce memory bandwidth or capacity requirements.
4) **Distillation/search for hybrid models** that selects which layers retain softmax attention when converting to linear/sparse attention, typically requiring training.

Our proposal targets a specific underexplored corner: **training-free** selection of which layers keep full attention during prefill for SWA adaptation with full-attention decode.

### Related Papers

(At least 20 papers; local references used when scraped.)

- **[SWAA: Sliding Window Attention Adaptation for Efficient Long-Context LLMs Without Pretraining](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/meta/meta_info.txt)**: Introduces full-attention decode and shows strong synergy with hybrid layers, but uses fixed periodic layer patterns.
- **[LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation](./references/Your-Long-Context-LLM-is-Secretly-a-Hybrid-Model-with-Effortless-Adaptation/meta/meta_info.txt)**: Selects “lazy layers” via attention-mass heuristics to replace with streaming attention; SWAA reports instability across model families.
- **[LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/meta/meta_info.txt)**: Provides long-context conversational histories and an LLM-based evaluation protocol used by SWAA.
- **[Distilling to Hybrid Attention Models via KL-Guided Layer Selection](./references/Distilling-to-Hybrid-Attention-Models-via-KL-Guided-Layer-Selection/meta/meta_info.txt)**: Uses marginal KL reduction to choose softmax layers during distillation into linear-attention hybrids; requires training and targets a different setting.
- **[StreamingLLM](https://arxiv.org/abs/2309.17453)**: Uses attention sink tokens + a windowed cache for streaming inference with long contexts.
- **[H2O: Heavy-Hitter Oracle for KV Cache](https://arxiv.org/abs/2306.14048)**: Retains high-attention “heavy hitter” tokens to compress KV caches for long-context inference.
- **[SnapKV](https://arxiv.org/abs/2404.14474)**: Compresses KV cache by keeping salient tokens and evicting others with minimal quality loss.
- **[Quest: Query-Aware Sparsity for Efficient Long-Context Inference](https://arxiv.org/abs/2406.10774)**: Uses query-aware token selection to accelerate long-context attention.
- **[MInference](https://arxiv.org/abs/2407.02490)**: Speeds up long-context decoding via sparse attention approximations and caching strategies.
- **[InfiniteVL](https://arxiv.org/abs/2512.08829)**: Uses SWA + linear attention modules for efficient very-long context (multimodal) inference.
- **[InfLLM-V2: Dense-Sparse Switchable Attention](https://arxiv.org/abs/2509.08959)**: Switches between dense and sparse attention modes to retain short-context quality while enabling long-context efficiency.
- **[Neural Attention Search Linear (NAtS-L)](https://arxiv.org/abs/2602.03681)**: Routes chunks/tokens to softmax vs linear attention within layers, trained from scratch.
- **[MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head](https://arxiv.org/abs/2601.07832)**: Token-partitioned linear attention with learnable mixing to improve expressivity.
- **[STILL: Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs](https://arxiv.org/abs/2602.02180)**: Token routing inside layers to mix exact and linearized attention.
- **[HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection](https://arxiv.org/abs/2602.03560)**: Interleaves full-attention and sparse layers and shares selected tokens/KV; focuses on architecture design rather than post-hoc adaptation.
- **[TriangleMix](https://arxiv.org/abs/2507.21526)**: Training-free prefill acceleration by skipping attention blocks with low decoding-time contribution.
- **[POD-Attention](https://arxiv.org/abs/2410.18038)**: Improves serving latency by overlapping prefill and decode kernels.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**: IO-aware exact attention kernel widely used in long-context inference.
- **[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)**: Efficient KV-cache management and serving framework used by SWAA.
- **[Gemma 2](https://arxiv.org/abs/2408.00118)**: Uses interleaved local attention and global attention layers when trained from scratch.
- **[Gemma 3](https://arxiv.org/abs/2503.19786)**: Extends hybrid attention patterns (local vs global) in production models.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Training-free SWA conversion | Apply SWA to a full-attention pretrained model without retraining | SWAA, LightTransfer | LongMemEval, LongBench-V2, RULER | Layer choice is brittle; can collapse quality |
| Hybrid attention trained from scratch | Train models with fixed patterns of local/global attention | Gemma-2/3, HySparse, many hybrid LLMs | LongBench, RULER, standard LLM suites | Requires pretraining; hard to retrofit |
| KV-cache eviction/compression | Keep a subset of KV tokens (heavy hitters / sinks / top-k) | StreamingLLM, H2O, SnapKV, Quest | LongBench, RULER, needle tests | Mostly targets memory; can hurt recall |
| Distillation/search for hybridization | Learn which layers keep softmax attention when converting to linear/sparse | KL-guided selection (2512.20569), NAS-style hybrids | RULER, retrieval tasks | Requires training budget; not plug-and-play |

### Closest Prior Work

1) **SWAA** (**[link](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/meta/meta_info.txt)**): Demonstrates full-attention decode + hybrid layers is effective, but uses fixed periodic layer patterns and leaves the 1/4-FA regime notably weaker than 1/2-FA.

2) **LightTransfer** (**[link](./references/Your-Long-Context-LLM-is-Secretly-a-Hybrid-Model-with-Effortless-Adaptation/meta/meta_info.txt)**): Proposes an attention-mass “lazy ratio” heuristic for selecting which layers can use streaming attention, but SWAA reports it is not consistently better than fixed patterns across model families.

3) **KL-guided layer selection for hybrid distillation** (**[link](./references/Distilling-to-Hybrid-Attention-Models-via-KL-Guided-Layer-Selection/meta/meta_info.txt)**): Uses KL reduction to choose which layers keep softmax attention, but requires distillation runs and targets linear-attention students rather than training-free SWA adaptation.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| SWAA | Training-free SWA adaptation with full-attention decode + fixed hybrid layer patterns | 1/4-FA budget still far from 1/2-FA; layer choice is model-specific | Replace fixed periodic patterns with loss-based layer scoring | Directly optimizes predictive quality under the modified mask |
| LightTransfer | Selects layers using attention-mass heuristics (“lazy ratio”) | Can be unstable across model families; heuristic may not track rare long-range links | Use answer-token NLL under full-attention decode as the signal | NLL is a direct measure of long-range prompt-to-answer utility |
| KL-guided distillation | Select layers by KL reduction during distillation to linear-attention hybrids | Requires training and many distillation runs | Use a forward-only calibration signal; no training | Much cheaper and directly applicable to pretrained models |
| HySparse | Hybrid sparse architecture with oracle token selection | Not a post-hoc conversion recipe; assumes architecture/training control | Focus on retrofitting a pretrained model with minimal changes | Targets the common “convert an existing checkpoint” deployment need |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen/Qwen3-4B-Thinking-2507 | 4B | https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507 | Primary model used in SWAA Table 1 |

**Training Data (if applicable):**

No training data needed — inference + calibration only.

**Calibration data (for layer scoring):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| LongAlign-10k | Long instruction-response pairs | 10k | https://huggingface.co/datasets/zai-org/LongAlign-10k | See dataset card |
| fusang-v1-filtered (long) | Long instruction-response pairs | ~16k | https://huggingface.co/datasets/yuyijiong/fusang-v1-filtered | See dataset card |

We will subsample to a small calibration set (e.g., 64 examples with 16k–32k prompt length) to control compute.

**Evaluation resources / scripts:**
- SWAA codebase (reference implementation of full-attention decode and hybrid masks): https://github.com/yuyijiong/sliding-window-attention-adaptation
- LongMemEval benchmark + official evaluation prompt (LLM-based): https://github.com/xiaowu0162/LongMemEval

**Resource Estimate** (fits `internal_context/resource_budget.md`):
- **Compute budget**: ~100–250 GPU-hours total.
  - Calibration: \(\approx\) (num_layers+1) forward passes over ~64 long prompts; dominated by long-context attention.
  - Evaluation: 3 inference runs (A/B/C) over 500 LongMemEval_24k examples.
- **GPU memory**: Qwen3-4B with 32k context in fp16 should fit on 80GB A100; use tensor parallelism if needed.
- **API usage**: ~500 calls to `gpt-5-mini` for LLM-based scoring (plus retries), temperature=0.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| LongMemEval_24k | Long-context conversational QA built by concatenating multiple chat sessions to ~24k tokens | Accuracy from an LLM-based evaluator (same protocol as SWAA) | test | https://github.com/xiaowu0162/LongMemEval | LongMemEval repo + SWAA prompt format |

Optional secondary checks (not required for the main decision):
- LongBench-V2 subset (<=128k context) and RULER Multi-Query as in SWAA (**[SWAA eval description](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/sections/Evaluation%20Dataset.md)**).

### Main Results

#### Results Table

Primary metric: LongMemEval_24k accuracy (%) for Qwen3-4B-Thinking, window=2k, keep-first=10, full-attention decode enabled.

| Method | Base Model | Benchmark | Accuracy (%) | Source | Notes |
|---|---|---|---:|---|---|
| Full attention | Qwen3-4B-Thinking | LongMemEval_24k | 73.0 | [SWAA Table 1](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/sections/SWA%20Adaptation%20Without%20Fine-tuning.md) | Upper bound |
| Naive SWA (2k) | Qwen3-4B-Thinking | LongMemEval_24k | 3.2 | [SWAA Table 1](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/sections/SWA%20Adaptation%20Without%20Fine-tuning.md) | Lower bound |
| Fixed hybrid, 1/2 FA layers (periodic) | Qwen3-4B-Thinking | LongMemEval_24k | 65.0 | [SWAA Table 1](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/sections/SWA%20Adaptation%20Without%20Fine-tuning.md) | Periodic pattern [1,3,5,...] |
| Fixed hybrid, 1/4 FA layers (best periodic offset) | Qwen3-4B-Thinking | LongMemEval_24k | 54.2 | [SWAA Table 1](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/sections/SWA%20Adaptation%20Without%20Fine-tuning.md) | Periodic offset [3,7,11,...] |
| Training-free layer selection: LightTransfer (lazy-ratio) | Qwen3-4B-Thinking | LongMemEval_24k | 70.2 | [SWAA Table 8 (HTML)](https://arxiv.org/html/2512.10411v4#A6.T8) | Uses keep-first=100 (not directly comparable to keep-first=10), so treat as context-only baseline |
| **Ours: NLL-guided hybrid, 1/4 FA layers** | Qwen3-4B-Thinking | LongMemEval_24k | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Short-prompt control | Compute ΔNLL rankings on prompts <= window size | Rankings should differ from long-prompt rankings; otherwise signal is confounded |
| Calibration size sweep | Use 16 vs 64 calibration examples | If method is real, selected layers should be reasonably stable |
| No-sink variant | keep-first-k disabled | If keep-first is important for stability, all methods degrade similarly |

### Analysis (Optional)

- Plot ΔNLL per layer and compare with periodic patterns (odd/even and 1/4 offsets) to see whether the learned set clusters in depth.
- Measure “selection break-even”: calibration cost vs per-request speedup from using 1/4-FA instead of 1/2-FA.

---

## Success Criteria

**Criterion 1: Recover 1/2-FA quality with 1/4-FA budget**
- Hypothesis: NLL-guided selection finds a 1/4-FA layer set that is close to the 1/2-FA periodic hybrid.
- Validation: On LongMemEval_24k, our 1/4-FA hybrid is within **2 absolute points** of the fixed 1/2-FA hybrid (65.0 from SWAA baseline).

**Criterion 2: Beat the best fixed 1/4-FA periodic pattern**
- Hypothesis: Loss-guided selection is better than periodic offsets in the 1/4 budget regime.
- Validation: Our 1/4-FA hybrid exceeds 54.2 (best SWAA-reported periodic 1/4 offset) on LongMemEval_24k.

**Criterion 3: Signal is long-range specific (not generic layer sensitivity)**
- Hypothesis: Long-prompt ΔNLL rankings differ from short-prompt rankings.
- Validation: Low rank correlation between long-prompt and short-prompt ΔNLL rankings; if highly correlated, treat the approach as refuted for this use.

---

## Impact Statement

If this works, practitioners who want to retrofit existing pretrained LLM checkpoints for long-context serving can achieve SWAA-level long-context quality with fewer full-attention layers during prefill, reducing prefill latency and cost without fine-tuning. This makes hybrid attention deployment more practical for long-document applications where prompt ingestion dominates runtime.

---

## References

- [SWAA: Sliding Window Attention Adaptation for Efficient Long-Context LLMs Without Pretraining](./references/SWAA-Sliding-Window-Attention-Adaptation-for-Efficient-Long-Context-LLMs-Without-Pretraining/meta/meta_info.txt) - Yu et al., 2025
- [LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation](./references/Your-Long-Context-LLM-is-Secretly-a-Hybrid-Model-with-Effortless-Adaptation/meta/meta_info.txt) - Zhang et al., 2024
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/meta/meta_info.txt) - Wu et al., 2024
- [Distilling to Hybrid Attention Models via KL-Guided Layer Selection](./references/Distilling-to-Hybrid-Attention-Models-via-KL-Guided-Layer-Selection/meta/meta_info.txt) - Li et al., 2025
- [StreamingLLM](https://arxiv.org/abs/2309.17453) - Xiao et al., 2023
- [H2O: Heavy-Hitter Oracle for KV Cache](https://arxiv.org/abs/2306.14048) - Zhang et al., 2023
- [SnapKV](https://arxiv.org/abs/2404.14474) - Wang et al., 2024
- [Quest](https://arxiv.org/abs/2406.10774) - Li et al., 2024
- [MInference](https://arxiv.org/abs/2407.02490) - Contributors, 2024
- [InfiniteVL](https://arxiv.org/abs/2512.08829) - Hu et al., 2025
- [InfLLM-V2](https://arxiv.org/abs/2509.08959) - Zhao et al., 2025
- [NAtS-L](https://arxiv.org/abs/2602.03681) - Deng et al., 2026
- [MHLA](https://arxiv.org/abs/2601.07832) - Authors, 2026
- [STILL](https://arxiv.org/abs/2602.02180) - Authors, 2026
- [HySparse](https://arxiv.org/abs/2602.03560) - Authors, 2026
- [TriangleMix](https://arxiv.org/abs/2507.21526) - Authors, 2025
- [POD-Attention](https://arxiv.org/abs/2410.18038) - Authors, 2024
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao et al., 2023
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023
- [Gemma 2](https://arxiv.org/abs/2408.00118) - Team, 2024
- [Gemma 3](https://arxiv.org/abs/2503.19786) - Team, 2025
