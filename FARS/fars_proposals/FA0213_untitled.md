# untitled

# Overlap-Refresh for Window-Diffusion: Decoupling Window Shifts from Full KV Refresh in Diffusion LMs

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Automation constraint**: Fully automated evaluation (no human labeling or manual judging)
- **Compute constraint**: <= 768 A100 GPU-hours total (inference-only; no fine-tuning)

## Introduction

### Context and Motivation

Diffusion language models (DLMs) generate text by iteratively denoising a partially corrupted token sequence (often created by masking with a special [MASK] token), rather than generating tokens left-to-right like autoregressive (AR) language models. A practical challenge for DLM deployment is inference cost: each denoising step typically runs a Transformer forward pass with bidirectional self-attention over the full prompt + output-length sequence, even though only a subset of tokens materially changes at each step.

Window-Diffusion (arXiv:2601.20332) is a training-free inference acceleration method for masked DLMs such as Dream and LLaDA. It speeds up inference by (i) pruning far-field masked tokens using a local window around the decoding frontier and (ii) reusing a Key/Value (KV) cache for stable context tokens across denoising steps. Window-Diffusion reports large wall-clock latency reductions on reasoning and code benchmarks (e.g., a 6.6x speedup on MBPP for Dream-Instruct under fixed-length inference).

### The Problem

Window-Diffusion partitions denoising into phases. At each phase boundary it both:

1) shifts the external window over the undecoded prefix, and
2) performs a full KV-cache refresh: a full forward pass over all decoded tokens plus all tokens in the external window.

A single hyperparameter (`refresh_cycle`) controls both how often the window shifts and how often a full refresh runs.

This coupling creates an efficiency/quality trade-off: shifting the window more often can improve quality by keeping the pruned context aligned with the current decoding frontier, but it forces more full refreshes. Increasing `refresh_cycle` reduces refresh overhead but can make the window stale and hurt accuracy. The Window-Diffusion ablation (Fig. 6b) shows non-monotonic accuracy as a function of `refresh_cycle`, suggesting the coupled schedule is not always an optimal operating point.

### Key Insight and Hypothesis

**Key insight:** consecutive external windows overlap heavily because decoding advances only a limited number of positions between phase boundaries while the external window is wide (e.g., length 128). Window-Diffusion recomputes KV for the entire external window at every phase boundary even though most of that window is unchanged.

**Hypothesis:** We can decouple (a) window shifting from (b) full KV refresh by reusing cached KV for the overlap region and computing KV only for the newly entered masked tokens via a query-only "delta-prefill". This is an approximation because DLMs use bidirectional attention, but we expect the error to be bounded because (i) Window-Diffusion's analysis shows that many non-active masked tokens have reusable intermediate representations across adjacent steps (Fig. 3), (ii) newly entered tokens are at the tail of the external window and tend to have low influence on prefix-local active tokens, and (iii) periodic full refresh still bounds drift.

---

## Proposed Approach

### Overview

We propose **Overlap-Refresh**, an inference-time modification to Window-Diffusion that replaces the single `refresh_cycle` knob with two knobs:

- `shift_interval` (s): how often the external window is updated
- `refresh_interval` (R, a multiple of s): how often a full KV refresh is executed

At shift-only boundaries (every s steps, excluding refresh steps), we update the external window but avoid a full refresh by reusing KV for overlap tokens and computing KV only for newly entered tokens via a query-only delta-prefill.

### Method Details

**Window-Diffusion notation (background).** At diffusion step t:
- D(t): decoded tokens (always included as conditional context)
- W_ex: external window, a length-L_ex prefix of the currently undecoded region
- A(t): active tokens (internal window W_in), the first L_in tokens of W_ex where logits are computed
- B(t): buffer tokens in W_ex that provide context only

Window-Diffusion performs:
- **Full refresh step:** full forward pass over D(t) union W_ex to write KV for these tokens.
- **Normal steps:** recompute only A(t) (and recently decoded tokens) while reusing cached KV for B(t).

**Overlap-Refresh scheduling.** Introduce two intervals:
- `shift_interval = s`
- `refresh_interval = R` where R is a multiple of s

At each diffusion step t:

1) **Full refresh (t % R == 0):**
- Update W_ex to the current undecoded prefix.
- Run the standard Window-Diffusion full refresh forward pass over D(t) union W_ex and write KV for all these tokens.

2) **Shift-only boundary (t % s == 0 and t % R != 0):**
- Update W_ex to the current undecoded prefix.
- Let W_old be the previous external window and W_new the updated window.
- Define overlap O = W_old intersect W_new and delta tokens N = W_new \ W_old.
- **Reuse overlap KV:** keep cached KV for tokens in O.
- **Delta-prefill (query-only):** compute hidden states and KV only for tokens in N by running each Transformer layer with:
  - queries restricted to positions N,
  - keys/values taken from cached KV for D(t) and O, plus freshly computed KV for N.
- Store KV for N in the cache. Do not update cached KV for overlap tokens O at this boundary.

3) **Normal step (otherwise):**
- Same as Window-Diffusion: compute full forward only for active tokens A(t) (and any recently decoded tokens per the original implementation), reusing cached KV for context-only tokens.

**Why delta-prefill can be cheaper than full refresh.** Let C = |D(t)| + |W_ex| be the number of tokens involved in a full refresh.
- A full refresh has attention cost approximately O(C^2) per layer.
- Delta-prefill computes attention outputs only for |N| query tokens, costing approximately O(|N| * C) per layer.

When overlap is large (|N| << |W_ex|), shift-only boundaries can be much cheaper than full refresh boundaries, allowing more frequent window shifts without paying for full refresh each time.

### Key Innovations

1) **Decoupled schedule for Window-Diffusion:** separate window-shift frequency (context alignment) from full refresh frequency (error control).
2) **Delta-prefill for diffusion LM window shifts:** a query-only prefill operator that computes KV for newly entered window tokens while reusing cached KV for overlap tokens.
3) **Minimal decisive test:** a 3-condition experiment that decides whether decoupling yields a better accuracy/latency Pareto point than the coupled baseline.

---

## Related Work

### Field Overview

Masked diffusion language models (DLMs) refine a full-length sequence over multiple denoising steps, typically with bidirectional self-attention. This creates a mismatch between full-sequence computation (which scales with the maximum length) and sparse token updates (which often concentrate near the undecoded prefix). Recent inference-time efficiency work exploits this structure via (i) windowing and token pruning, (ii) KV-cache reuse and refresh policies for stable tokens, and (iii) block-structured decoding schedules.

Window-Diffusion combines windowing with phase-level KV caching and reports large speedups while largely preserving quality. Its analysis also suggests that non-active masked tokens inside a local context window can reuse intermediate representations across adjacent steps. However, its phase boundary couples external-window shifts with a full KV refresh, potentially recomputing many overlap tokens unnecessarily.

### Related Papers

- **[Window-Diffusion](./references/Window-Diffusion-Accelerating-Diffusion-Language-Model-Inference-with-Windowed-Token-Pruning-and-Caching/meta/meta_info.txt)**: Training-free windowed pruning + phase-level KV caching; the baseline system we modify.
- **[Attention Is All You Need for KV Cache in Diffusion LLMs (Elastic-Cache)](./references/ATTENTION-IS-ALL-YOU-NEED-FOR-KV-CACHE-IN-DIFFUSION-LLMS/meta/meta_info.txt)**: Sliding-window decoding with attention-drift triggers for layer-selective refresh; closest prior on adaptive refresh but not overlap-aware window shifting.
- **[dKV-Cache](./references/dKV-Cache-The-Cache-for-Diffusion-Language-Models/meta/meta_info.txt)**: Delayed caching of decoded-token KV to address instability under bidirectional attention.
- **[dLLM-Cache](./references/dLLM-Cache-Accelerating-Diffusion-Large-Language-Models-with-Adaptive-Caching/meta/meta_info.txt)**: Adaptive caching for diffusion LMs, including selective updates based on representation stability.
- **[d2Cache](./references/d2Cache-Accelerating-Diffusion-Based-LLMs-via-Dual-Adaptive-Caching/meta/meta_info.txt)**: Dual adaptive caching strategies for diffusion-based LLMs.
- **[Streaming-dLLM](./references/Streaming-dLLM-Accelerating-Diffusion-LLMs-via-Suffix-Pruning-and-Dynamic-Decoding/meta/meta_info.txt)**: Accelerates diffusion LMs via suffix pruning and dynamic decoding schedules.
- **[Sparse-dLLM](https://arxiv.org/abs/2508.02558)**: Dynamic bidirectional cache eviction exploiting stable attention sparsity across steps.
- **[Mask Tokens as Prophet](https://arxiv.org/abs/2510.09309)**: Fine-grained cache eviction for diffusion LM inference using mask-token signals.
- **[Dynamic Sliding Block Scheduling (DSB)](https://arxiv.org/abs/2602.05992)**: Dynamically schedules decoding blocks to trade off quality and compute.
- **[Fast-dLLM](https://arxiv.org/abs/2505.22618)**: Training-free acceleration using block-wise decoding and caching outside the current block.
- **[Block Diffusion](https://arxiv.org/abs/2503.09573)**: Autoregressive-over-blocks and diffusion-within-blocks generation paradigm; enables structured caching but changes decoding semantics.
- **[DINGO](https://arxiv.org/abs/2505.23061)**: Constrained inference for diffusion LLMs; relevant because inference modifications must preserve correctness under constraints.
- **[Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992)**: Scales masked diffusion LMs to 8B parameters; a major model family in this setting.
- **[Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)**: Introduces the Dream model family used in Window-Diffusion experiments.
- **[MDLM](https://arxiv.org/abs/2406.07524)**: Masked diffusion language modeling that matches autoregressive LM performance at smaller scales.
- **[SEDD](https://arxiv.org/abs/2310.16834)**: Discrete diffusion language modeling using a score-entropy objective.
- **[D3PM](https://arxiv.org/abs/2107.03006)**: Foundational discrete diffusion framework for categorical data.
- **[Diffusion-LM](https://arxiv.org/abs/2205.14217)**: Early diffusion approach for text generation.
- **[MaskGIT](https://arxiv.org/abs/2202.04200)**: Iterative masked token prediction for generation (vision); conceptual predecessor to masked iterative decoding schedules.
- **[StreamingLLM](https://arxiv.org/abs/2309.17453)**: Autoregressive streaming with attention sink KV-cache management; relevant as historical inspiration for incremental caching.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Typical evaluation | Known limitations |
|---|---|---|---|---|
| Windowing / token pruning | Reduce attention scope by pruning far-field tokens using locality | Window-Diffusion, Streaming-dLLM | Accuracy vs latency/throughput on GSM8K/MATH/HumanEval/MBPP | Quality drop if pruned tokens matter; scheduling sensitivity |
| KV reuse / stability caching | Reuse KV for stable tokens; refresh to bound drift | dKV-Cache, dLLM-Cache, d2Cache, Elastic-Cache, Mask Tokens as Prophet, Sparse-dLLM | Same reasoning/code benchmarks; tokens/sec or wall-clock | Bidirectional attention makes exact caching hard; heuristics may be brittle |
| Block-structured decoding | Change decoding structure to blocks to enable caching or reduce steps | Block Diffusion, Fast-dLLM, DSB | Reasoning/code benchmarks | Often requires retraining or changes generation semantics |
| Constrained diffusion decoding | Add constraints or verification to diffusion decoding | DINGO | CFG/constraint tasks | Adds overhead; orthogonal to caching |

### Closest Prior Work

- **Window-Diffusion:** Couples window shifts and full refresh via a single `refresh_cycle`. Full refresh recomputes KV for D(t) union W_ex even when consecutive windows overlap heavily. We decouple these events and replace intermediate full refresh with a query-only delta-prefill.
- **Elastic-Cache:** Uses attention-drift signals to trigger layer-selective cache refresh in a sliding window. Our method does not compute drift metrics and instead exploits deterministic overlap between successive external windows in Window-Diffusion's phase structure.
- **dKV-Cache / dLLM-Cache / d2Cache:** Focus on token stability or delayed caching across steps, but do not target Window-Diffusion's coupled shift+refresh cost.

**Novelty Kill Search Summary:** Searched for the exact technique+setting combination using queries such as "Window-Diffusion overlap refresh", "Window-Diffusion incremental refresh", "decouple window shift refresh cycle diffusion", and "delta prefill KV cache diffusion". Also searched local finalized proposals and other agents' drafts for "Window-Diffusion" and "2601.20332" and found no matches as of 2026-02-21. Full query log is in notes.md.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Window-Diffusion | Windowed pruning + phase-level KV reuse with periodic full refresh | Window shift and full refresh are coupled; overlap tokens are recomputed each phase | Split shift vs refresh; add delta-prefill at shift-only boundaries | Same window alignment as small refresh_cycle, but fewer full refreshes |
| Elastic-Cache | Sliding window + drift-triggered layer-selective refresh | Requires drift signals; not overlap-aware in Window-Diffusion phase structure | Deterministic overlap reuse; no drift computation | Lower overhead and simpler integration into Window-Diffusion |
| dKV-Cache | Delayed caching of decoded tokens | No window pruning; does not address window movement | Combine window pruning with incremental prefill | Targets Window-Diffusion's full refresh overhead |
| Sparse-dLLM | Dynamic eviction + block-wise refresh | Different structure; not focused on phase coupling in Window-Diffusion | Minimal change to Window-Diffusion schedule | Isolates phase coupling as a decision-changing design point |

---

## Experiments

### Experimental Setup

**Goal:** Decide whether decoupling window shifts from full refresh yields a better accuracy/latency trade-off than the coupled Window-Diffusion baseline.

**Baseline Ladder (for an inference-efficiency proposal):**
- Closest existing method: Window-Diffusion with the default refresh_cycle (baseline A).
- Naive longer refresh baseline: Window-Diffusion with a larger refresh_cycle (baseline B) to show that "refresh less often" alone can hurt quality.
- Proposed method: Overlap-Refresh (C).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Dream-v0-Instruct-7B | 7B | https://huggingface.co/Dream-org/Dream-v0-Instruct-7B | Model family used in Window-Diffusion Table 3 |

**Training Data (if applicable):** No training data needed; inference only.

**Resource Estimate:**
- Window-Diffusion reports MBPP fixed-length latency of 32.9 seconds per instance on an NVIDIA A6000 GPU for Dream-Instruct (Table 3).
- MBPP has approximately 1k problems, implying roughly ~9 GPU-hours per condition on a single A6000-class GPU.
- Planned runs: 3 conditions (A/B/C) x up to 3 seeds => expected to be well below 768 GPU-hours, and parallelizable over multiple GPUs.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MBPP | Mostly Basic Python Problems: Python code generation tasks with unit tests (974 tasks total; 500 in the standard test split) | Pass@1 (fraction of tasks solved by a single sampled solution; higher is better), latency seconds/instance (lower is better), tokens/sec (higher is better) | test | https://huggingface.co/datasets/google-research-datasets/mbpp (or equivalent) | Prefer the official evaluation pipeline in https://github.com/vhicrgit/Window-Diffusion; otherwise use a standard MBPP unit-test runner |

(Optional secondary benchmark for directional evidence only: HumanEval.)

### Main Results

#### Comparability Rules (CRITICAL)

All rows must match:
- base model checkpoint
- max generation length (fixed-length; early stopping disabled)
- diffusion-step budget and sampling hyperparameters
- prompt format and evaluation script

#### Results Table

Published anchor numbers from Window-Diffusion Table 3 (Dream-Instruct; fixed-length inference; early stopping disabled):

| Method | Base Model | Benchmark | Pass@1 / accuracy (%) (mean+/-std) | Latency (s) (mean+/-std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Dream (vanilla) | Dream-v0-Instruct-7B | MBPP (len=1024) | 58.8 (1 run) | 217.8 (1 run) | [Window-Diffusion](./references/Window-Diffusion-Accelerating-Diffusion-Language-Model-Inference-with-Windowed-Token-Pruning-and-Caching/meta/meta_info.txt) | Table 3; A6000; FP32 |
| Window-Diffusion (refresh_cycle=32) | Dream-v0-Instruct-7B | MBPP (len=1024) | 55.4 (1 run) | 32.9 (1 run) | [Window-Diffusion](./references/Window-Diffusion-Accelerating-Diffusion-Language-Model-Inference-with-Windowed-Token-Pruning-and-Caching/meta/meta_info.txt) | Table 3; A6000; FP32 |
| Window-Diffusion (refresh_cycle=64) | Dream-v0-Instruct-7B | MBPP (len=1024) | **TBD** | **TBD** | - | Needs re-run (not reported in paper) |
| **Overlap-Refresh (ours)** (shift=32, refresh=64) | Dream-v0-Instruct-7B | MBPP (len=1024) | **TBD** | **TBD** | - | To be verified |

We will also report tokens/sec and wall-clock speedups measured on our hardware for all rerun rows.

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| Overlap-Refresh with smaller decoupling (shift=32, refresh=48) | Reduce refresh_interval | If refresh=64 causes too much quality loss, refresh=48 should recover accuracy with smaller speedup |

### Experimental Rigor

**Variance & Reproducibility:**
- If decoding is stochastic under the chosen settings, run >=3 seeds (e.g., seeds=[42, 123, 456]) and report mean+/-std.
- If decoding is deterministic, state this explicitly and run one seed.

**Validity threats & controls:**
- Prompt/eval mismatch: keep prompts and evaluation scripts identical across A/B/C.
- Compute mismatch: hold diffusion-step budget, max length, and window sizes fixed across A/B/C; only change shift/refresh scheduling.
- Hardware dependence: measure relative speedups within a single machine; treat published latency as a non-comparable anchor if hardware differs.

**Sanity checks:**
- Reproduce the published Window-Diffusion MBPP fixed-length pass@1 within a reasonable tolerance before comparing A/B/C.

### Analysis (Optional)

Record runtime breakdown:
- fraction of time spent in full refresh steps vs delta-prefill vs normal steps.
This checks whether any observed speedup matches the expected reduction in full refresh work.

---

## Success Criteria

**Hypothesis (directional):** Overlap-Refresh will match or nearly match the accuracy of Window-Diffusion with refresh_cycle=32 while improving throughput, and it will dominate the naive refresh_cycle=64 baseline in accuracy at similar throughput.

**Decision Rule (concrete):**
- **Proceed** if, on MBPP, Overlap-Refresh achieves:
  - pass@1 within 2 percentage points of Window-Diffusion refresh_cycle=32, and
  - >=10% higher tokens/sec than refresh_cycle=32, and
  - higher pass@1 than refresh_cycle=64 at similar tokens/sec (within 5%).
- **Pivot** if speed improves (>=10%) but pass@1 drops by 2-4 pp vs refresh_cycle=32; then test refresh_interval=48 as the only follow-up knob.
- **Refute** if speedup is <5% or pass@1 drops by >=4 pp vs refresh_cycle=32.

---

## Impact Statement

If Overlap-Refresh works, it provides a drop-in improvement to Window-Diffusion that reduces full refresh overhead without retraining, improving the latency-quality trade-off for deploying diffusion LMs on long-generation tasks. If it fails, it provides a negative result suggesting that recomputing overlap tokens at window shifts is necessary under bidirectional attention, clarifying why Window-Diffusion couples shift and refresh.

---

## References

- [Window-Diffusion](./references/Window-Diffusion-Accelerating-Diffusion-Language-Model-Inference-with-Windowed-Token-Pruning-and-Caching/meta/meta_info.txt) - arXiv:2601.20332
- [Attention Is All You Need for KV Cache in Diffusion LLMs (Elastic-Cache)](./references/ATTENTION-IS-ALL-YOU-NEED-FOR-KV-CACHE-IN-DIFFUSION-LLMS/meta/meta_info.txt) - arXiv:2510.14973
- [dKV-Cache](./references/dKV-Cache-The-Cache-for-Diffusion-Language-Models/meta/meta_info.txt) - arXiv:2505.15781
- [dLLM-Cache](./references/dLLM-Cache-Accelerating-Diffusion-Large-Language-Models-with-Adaptive-Caching/meta/meta_info.txt) - arXiv:2506.06295
- [d2Cache](./references/d2Cache-Accelerating-Diffusion-Based-LLMs-via-Dual-Adaptive-Caching/meta/meta_info.txt) - arXiv:2509.23094
- [Streaming-dLLM](./references/Streaming-dLLM-Accelerating-Diffusion-LLMs-via-Suffix-Pruning-and-Dynamic-Decoding/meta/meta_info.txt) - arXiv:2601.17917
- [Sparse-dLLM](https://arxiv.org/abs/2508.02558) - arXiv:2508.02558
- [Mask Tokens as Prophet](https://arxiv.org/abs/2510.09309) - arXiv:2510.09309
- [DSB](https://arxiv.org/abs/2602.05992) - arXiv:2602.05992
- [Fast-dLLM](https://arxiv.org/abs/2505.22618) - arXiv:2505.22618
- [Block Diffusion](https://arxiv.org/abs/2503.09573) - arXiv:2503.09573
- [DINGO](https://arxiv.org/abs/2505.23061) - arXiv:2505.23061
- [Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992) - arXiv:2502.09992
- [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487) - arXiv:2508.15487
- [MDLM](https://arxiv.org/abs/2406.07524) - arXiv:2406.07524
- [SEDD](https://arxiv.org/abs/2310.16834) - arXiv:2310.16834
- [D3PM](https://arxiv.org/abs/2107.03006) - arXiv:2107.03006
- [Diffusion-LM](https://arxiv.org/abs/2205.14217) - arXiv:2205.14217
- [MaskGIT](https://arxiv.org/abs/2202.04200) - arXiv:2202.04200
- [StreamingLLM](https://arxiv.org/abs/2309.17453) - arXiv:2309.17453
