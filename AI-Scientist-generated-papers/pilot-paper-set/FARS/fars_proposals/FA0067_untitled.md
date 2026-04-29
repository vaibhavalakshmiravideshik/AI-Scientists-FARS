# untitled

# Delta-Prefill Switching for Speculative Decoding in Prefix-Cached Multi-Turn Tool Traces

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

**Speculative decoding** accelerates autoregressive language model inference by using a fast **draft model** to propose multiple tokens and a slower **target model** to verify them in parallel, while preserving the target model’s greedy output (output equivalence). A closely related self-drafting variant is **multi-token prediction (MTP)**, where a single model is trained with additional lightweight prediction heads that propose multiple future tokens per forward pass (e.g., **MTP-3** proposes up to 3 tokens ahead).

In 2026, many latency-critical deployments are **multi-turn agentic sessions** rather than single-turn prompts: an agent repeatedly calls tools, appends tool outputs or retrieved documents to the context, and emits short structured actions (tool calls) interleaved with occasional longer natural-language outputs. In these settings, end-to-end task completion time is the sum of many sequential turns, so per-turn latency matters.

Modern serving stacks mitigate multi-turn cost via **prefix / KV caching**: the key-value (KV) attention cache for an earlier prompt prefix is reused on later turns, so the prefill computation is largely proportional to the **new tokens appended** since the previous turn, not the total prompt length. Step 3.5 Flash reports that in its search-agent RL training infrastructure it uses **sticky scheduling**—routing consecutive turns of a session to the same node—to maximize KV-cache reuse during multi-turn inference, highlighting KV reuse as a critical performance pattern for agentic workloads and motivating decode-time accelerators such as MTP/speculation.

### The Problem

Speculative decoding mainly reduces **decode** time (time spent generating new output tokens), but does not directly reduce the cost of **prefill** (processing the prompt to build the KV cache). Multiple recent systems papers show that speculative decoding can even yield **negative speedups** in practice due to verification overhead and GPU/kernel inefficiency when verifying variable-length draft blocks under batching.

This creates a specific failure mode in multi-turn agent sessions with prefix caching:

- Some turns append large tool payloads (large incremental prompt), making the turn **prefill-dominated**.
- The agent’s next action may still be short (e.g., a tool call), so decode acceleration has limited headroom.

Existing adaptive speculative decoding work focuses on adapting speculative length based on acceptance/confidence and system load (e.g., AdaSpec / SpecServe) or learning hyperparameters online (e.g., BanditSpec). These methods are powerful, but they rely on runtime confidence/acceptance signals and are not designed around the cache-induced distinction between **total prompt length** and **incremental prefill length** in multi-turn sessions.

A practical serving system therefore needs a low-overhead rule to decide whether to enable speculation **for this turn**, using only information available before decoding begins.

### Key Insight and Hypothesis

**Key insight:** With prefix/KV caching, the incremental prefill work of turn \(t\) is approximately proportional to **\(\Delta L_t\)** (pronounced “delta L”), the number of **new prompt tokens** appended since turn \(t-1\). When \(\Delta L_t\) is large, the turn is **prefill-dominated** (most latency is spent processing the new prompt tokens) and speculative decoding has limited headroom; when \(\Delta L_t\) is small, the turn is more **decode-dominated** (most latency is spent generating new output tokens) and speculation can help.

**Hypothesis:** A single scalar threshold on incremental prompt growth (\(\Delta L_t \le \tau\)) is a sufficiently strong *pre-decode* predictor of whether speculative decoding reduces wall-clock latency on prefix-cached multi-turn tool-use traces. Switching speculation on/off per turn using this threshold reduces total session wall-clock time compared to (i) always-on speculation and (ii) always-off greedy decoding, and is competitive with stronger adaptive baselines that do not use \(\Delta L\).

---

## Proposed Approach

### Overview

We propose **Delta-Prefill Switching (DPS)**: a deployment-time policy for speculative decoding that enables speculation only when the incremental prompt growth \(\Delta L\) is small.

For a multi-turn session with prompts \(p_1, p_2, \dots, p_T\) (each prompt is the full conversation/tool history up to that turn), define:
- \(L_t = |\mathrm{tokenize}(p_t)|\)
- \(\Delta L_t = L_t - L_{t-1}\) for \(t>1\), and \(\Delta L_1 = L_1\)

**Policy:**
- If \(\Delta L_t \le \tau\): run the turn with speculative decoding (draft+verify)
- Else: run the turn with standard greedy decoding (no speculation)

The threshold \(\tau\) is chosen on a small calibration set of sessions (minimizing total session wall-clock time) and then fixed for evaluation.

### Method Details

**Inputs and observables**
- The policy uses only \(\Delta L_t\), computed from the model tokenizer and the prompt text before generation begins.
- It does not use predicted output length, runtime acceptance probes, or online learning.

**Speculative decoding implementation**
- Use draft-model speculative decoding (draft proposes \(k\) tokens; target verifies; rejected tokens fall back to target decoding).
- Use deterministic decoding (temperature 0, top-p 1) to reduce variance in latency measurements.

**Prefix caching assumption**
- The evaluation uses an inference engine with prefix caching (e.g., a prefix-tree KV cache such as **RadixAttention** in **SGLang** (an open-source LLM inference engine), which stores many prefixes in a radix tree for reuse, or **vLLM** (another open-source LLM inference engine) with prefix caching), so that repeated prefixes across turns are reused and incremental prefill cost scales with \(\Delta L_t\) rather than \(L_t\).

### Key Innovations

- **Cache-aware switching signal:** use \(\Delta L\) (incremental prompt growth) rather than total prompt length as the pre-decode signal, explicitly targeting multi-turn prefix-cached sessions.
- **Minimal policy (single parameter):** one scalar threshold \(\tau\), calibrated once, with essentially zero runtime overhead.
- **Session-level objective:** evaluate on **total session wall-clock time** over real multi-turn tool traces rather than per-request throughput alone.

---

## Related Work

### Field Overview

Inference acceleration for transformer LLMs spans (i) decode accelerators (speculative decoding, MTP/Medusa, parallel/block decoding), (ii) prefill optimizations (prefix caching, prefill scheduling, token pruning), and (iii) system-level disaggregation (separating prefill and decode across hardware). Recent work also studies **adaptive** speculative decoding strategies that adjust speculative length or hyperparameters under changing workloads.

Our proposal sits at the intersection: it treats speculative decoding as one component of a multi-turn pipeline with prefix caching, and asks whether a simple cache-aware feature (\(\Delta L\)) is enough to decide when to use speculation.

### Related Papers

- **[Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)**: Motivates MTP-3 and KV-cache reuse (“sticky scheduling”) for low-latency multi-turn agents.
- **[AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware Large Language Model Serving](./references/AdaSpec-Adaptive-Speculative-Decoding-for-Fast-SLO-Aware-Large-Language-Model-Serving/meta/meta_info.txt)**: Adapts speculative strategies (length and verification) based on confidence and system state, and can switch to autoregressive mode under load.
- **[BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](./references/BanditSpec-Adaptive-Speculative-Decoding-via-Bandit-Algorithms/meta/meta_info.txt)**: Formulates hyperparameter selection for speculation as a bandit problem and adapts online to different prompts.
- **[LAPS: A Length-Aware-Prefill LLM Serving System](./references/LAPS-A-Length-Aware-Prefill-LLM-Serving-System/meta/meta_info.txt)**: Shows multi-turn workloads are dominated by short re-prefills and uses prefill length to disaggregate/schedule prefill, but does not decide whether to use speculative decoding.
- **[Batch speculative decoding: Done right](./references/Batch-speculative-decoding-Done-right/meta/meta_info.txt)**: Characterizes negative speedups and correctness failures in batch speculative decoding and proposes correctness-preserving scheduling.
- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)**: Original draft-model speculative decoding framework.
- **[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)**: Distribution-preserving speculative sampling variant.
- **[EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)**: Improves drafting using feature uncertainty; widely used in practice.
- **[Medusa: Simple LLM Inference Acceleration via Multiple Decoding Heads](https://arxiv.org/abs/2311.08091)**: Adds multiple decoding heads to predict multiple tokens per step.
- **[Speculative Prefill: Turbocharging TTFT with Lightweight and Accurate Token Importance Estimation](https://arxiv.org/abs/2502.02789)**: Speeds up prefill (time-to-first-token; TTFT) by pruning prompt tokens.
- **[DistServe: Disaggregating Prefill and Decode for Efficient LLM Serving](https://arxiv.org/abs/2401.09670)**: Separates prefill and decode across hardware; motivates modeling them separately.
- **[vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)**: Popular serving engine; includes prefix caching and speculation support.
- **[SGLang: Efficient Execution of Structured LLM Programs](https://arxiv.org/abs/2312.07104)**: Provides session/prefix caching via RadixAttention and supports speculative decoding.
- **[Preble: Efficient Distributed Prompt Scheduling for LLM Serving](https://arxiv.org/abs/2407.00023)**: Studies prompt-heavy workloads and caching/scheduling in distributed serving.
- **[AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding](https://arxiv.org/abs/2501.12162)**: Allocates speculative tokens under per-request SLO constraints in production serving.
- **[AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders](https://arxiv.org/abs/2510.19779)**: Improves draft models via selective distillation to raise acceptance rates.
- **[MagicDec: Breaking the Latency–Throughput Tradeoff for Long-Context Speculation](https://arxiv.org/abs/2408.11049)**: Studies speculation in long-context/high-batch regimes and bottleneck shifts.
- **[SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](./references/SuffixDecoding-Extreme-Speculative-Decoding-for-Emerging-AI-Applications/meta/meta_info.txt)**: Model-free speculative decoding that drafts tokens by suffix-tree pattern matching over previous generations, achieving large speedups on agentic workloads (e.g., 2.5× on SWE-Bench and 5.3× on AgenticSQL; speedup is higher-is-better).
- **[Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling](https://arxiv.org/abs/2408.08696)**: Reuses previously generated candidate tokens as draft tokens to accelerate decoding.
- **[Set Block Decoding is a Language Model Inference Accelerator](https://arxiv.org/abs/2509.04185)**: Fine-tunes LMs to decode multiple tokens per forward pass via masked-token prediction.
- **[LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding](https://arxiv.org/abs/2512.16229)**: Lookahead parallel decoding for diffusion language models; another decode-acceleration family.
- **[ToolLLM / ToolBench](https://arxiv.org/abs/2307.16789)**: Large-scale tool-use dataset with function-call traces.
- **[Berkeley Function Calling Leaderboard (BFCL)](https://arxiv.org/abs/2402.03301)**: Benchmark suite for function calling, including multi-turn settings.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Speculative decoding (draft+verify) | Draft proposes tokens; target verifies | Leviathan’23, Chen’23, EAGLE’24 | tokens/s, TTFT/TPOT, exact-match | Can slow down in low-headroom regimes; needs good draft |
| Adaptive speculative decoding | Adjust speculative length/hyperparams under changing workloads | AdaSpec’25, BanditSpec’25, AdaServe’25 | production traces, SLO violation, goodput | Often uses runtime confidence/online learning; may ignore prefix-cache ΔL |
| Prefill-aware serving | Optimize/schedule prefill based on prompt length | LAPS’26, Preble’24, DistServe’24 | TTFT, throughput, SLO | Does not decide decode accelerator use |
| Non-AR / multi-token decoding | Decode multiple tokens per forward pass | Medusa’23, Set Block Decoding’25 | NFEs, wall-clock | Requires model modifications/fine-tuning |

### Closest Prior Work

- **AdaSpec (SpecServe)**: Adapts speculative length using confidence signals and a throughput model; can switch to autoregressive mode under high backlog. It does not explicitly model cache-induced incremental prefill \(\Delta L\) for multi-turn sessions; its efficiency model depends on total context length and batch size.
- **SuffixDecoding**: A model-free, pattern-matching speculative decoder designed for agentic workloads, reporting large throughput/TPOT speedups (e.g., 2.5× on SWE-Bench and 5.3× on AgenticSQL at batch size 1) by drafting tokens from a suffix tree built from prior generations. Its control signal is *output repetitiveness*, not prefill cost.
- **BanditSpec**: Learns which speculative configuration is best for a prompt online, but may require exploration and does not directly use multi-turn prefix-cache structure.
- **LAPS**: Optimizes/schedules prefill work by separating short vs long prefills; it targets **queueing and batching** of prefills, whereas DPS controls **whether to use a decode-time accelerator** (speculative decoding) on each turn given prefix caching.
- **Batch speculative decoding: Done right**: Focuses on correctness and batching-related negative speedups rather than per-turn on/off decisions in multi-turn sessions.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| AdaSpec (SpecServe) | Adaptive speculative length via confidence + time model | Not designed around prefix-cache incremental prefill | Use \(\Delta L\) as a cache-aware pre-decode gating signal | Avoid spec overhead when prefill dominates even at low load |
| SuffixDecoding | Model-free suffix-tree pattern matching to draft tokens (agentic workloads) | Needs repetition; does not reason about prefill cost or cache deltas | Gate speculation using \(\Delta L\) (prefill headroom) and optionally combine with pattern-based speculation when enabled | Prevent negative speedups on prefill-dominated turns even when outputs are repetitive |
| BanditSpec | Online bandit selection of spec hyperparams | Exploration / more moving parts | Single offline-calibrated threshold | Lower overhead; predictable behavior |
| LAPS | Prefill-length-aware scheduling/disaggregation | Doesn’t decide spec vs non-spec | Use the same length signal to gate speculation | Complements scheduling with decode control |
| Batch spec-dec done right | Correct batching/sync for spec-dec | Not session-aware | Session-level gating using \(\Delta L\) | Improves end-to-end session time |

---

## Experiments

### Experimental Setup

**Goal:** Test whether \(\Delta L\)-threshold switching reduces **total session wall-clock time** on real multi-turn tool traces under prefix caching, and whether it matches stronger adaptive baselines that do not use \(\Delta L\).

**Trace replay protocol (for comparability):** We use dataset-provided conversation histories to build prompts for each turn. To keep prompts identical across methods, we do **not** feed model-generated tokens into the next turn’s prompt; instead, we extend the cached prefix using the trace’s ground-truth messages (tool outputs and assistant messages). This preserves the multi-turn \(\Delta L\) distribution while keeping latency comparisons apples-to-apples.

**Per-turn config requirement:** The timing harness must be able to enable/disable speculation per turn. If the serving engine only supports speculation as a server-level flag, we will run two servers (one speculative, one non-speculative) and route each turn to the appropriate server while keeping prefix caching consistent within each server.

**Base Models:**

| Role | Model | Size | Download Link | Notes |
|---|---|---:|---|---|
| Target | Qwen2.5-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Stable tokenizer family |
| Draft | Qwen2.5-Instruct | 1.5B | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Same tokenizer family; fast draft |

**Training Data (if applicable):**

No training is required (inference-only). Calibration chooses a scalar threshold \(\tau\).

**Other Resources (if applicable):**
- Inference engine implementing speculative decoding **and** prefix caching (recommended: SGLang or vLLM).

**Resource Estimate**:
- **Compute budget**: 30–120 GPU-hours (single A100), dominated by repeated timing runs for stability and multiple baselines.
- **GPU memory**: ≤ 80GB.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ToolBench (conversation traces) | Multi-turn tool-use conversations including tool outputs (function messages) | Session wall-clock (s; lower is better) ↓, per-turn latency (ms) ↓, TTFT (time to first token; ms; lower is better) ↓, tokens/s (higher is better) ↑ | 200 sessions | https://huggingface.co/datasets/tuandunghcmut/toolbench-v1 | Custom trace replay + timing harness |
| BFCL V3 Multi-turn (secondary) | Multi-turn function-calling scenarios including long-context variants | Same as above | official multi-turn split | https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard | Custom trace replay + timing harness |

### Methods Compared

**Primary comparison (decisive experiment; 3 conditions):**

- **A: Greedy (no speculation)**
- **B: Always-on draft-model speculative decoding** (fixed speculative length \(k\), e.g., 4 or 8)
- **C: Delta-Prefill Switching (ours)** (run B iff \(\Delta L \le \tau\); else run A)

**Secondary baselines (required for significance, but can be run on a smaller session subset if needed):**

- **D: Confidence-threshold dynamic speculative decoding (AdaSpec baseline)**
  - Implement the “threshold-based dynamic speculative decoding” baseline referenced in AdaSpec: dynamically stop drafting within a step when the assistant model’s token probability drops below a threshold (HuggingFace Transformers exposes this as \(\texttt{assistant_confidence_threshold}\)); this adapts speculative length using a runtime confidence signal rather than a pre-decode signal like \(\Delta L\).
- **E: SuffixDecoding (model-free, agentic baseline)**
  - Use vLLM’s SuffixDecoding mode (\(\texttt{speculative_config} = \{\text{"method"}:\text{"suffix"}, \text{"num_speculative_tokens"}:32\}\)) to draft tokens via suffix-tree pattern matching.

If implementing D/E is infeasible in the chosen engine, verification should switch engines (prefer vLLM) rather than dropping them.

### Main Results

(All values **TBD**; must be filled by verification runs. Same engine, same GPU, same prompts, same decoding config. If two servers are used for per-turn routing, report server configs and ensure each server maintains its own prefix cache across turns.)

| Method | Target/Draft | Benchmark | Median session time ↓ | p95 session time ↓ | Mean tokens/s ↑ | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| A: Greedy | 7B / – | ToolBench | **TBD** | **TBD** | **TBD** | This proposal | No speculation |
| B: Always speculate (draft-model) | 7B / 1.5B | ToolBench | **TBD** | **TBD** | **TBD** | This proposal | Fixed \(k\) |
| **C: DPS (ours)** | 7B / 1.5B | ToolBench | **TBD** | **TBD** | **TBD** | This proposal | Run B iff \(\Delta L \le \tau\) |
| D: Confidence-threshold dynamic (AdaSpec baseline) | 7B / 1.5B | ToolBench | **TBD** | **TBD** | **TBD** | This proposal | Uses \(\texttt{assistant_confidence_threshold}\) to choose \(k\) |
| E: SuffixDecoding | 7B / – | ToolBench | **TBD** | **TBD** | **TBD** | This proposal | vLLM suffix speculation (CPU-only drafting) |

(Repeat for BFCL V3 Multi-turn.)

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| DPS using \(L\) (total prompt length) instead of \(\Delta L\) | Remove cache-awareness | Worse predictor when prefix caching is effective |
| \(\tau\) sensitivity sweep | \(\tau\in\{128,256,512,1024\}\) | Extreme \(\tau\) collapses to A or B |
| Engine transfer | Tune \(\tau\) on SGLang, test on vLLM (or vice versa) | If \(\Delta L\) is a robust signal, re-calibration is small |

---

## Success Criteria

**Criterion 1: Session-level latency improvement (primary)**
- Hypothesis: DPS (C) reduces total session wall-clock time vs both greedy (A) and always-on speculation (B) on ToolBench.
- Validation: Paired bootstrap over sessions: C faster than A and B with p<0.05. We will report the percent median reduction; ≥5% is considered practically useful for latency-sensitive agent loops (and ≥10% is a strong result).

**Criterion 2: \(\Delta L\) predicts the per-turn winner**
- Hypothesis: A single threshold on \(\Delta L\) predicts whether A or B is faster for a turn.
- Validation: On held-out turns, the rule achieves ≥80% accuracy and ≥15 percentage points above a majority-class baseline computed from oracle winners.

**Criterion 3: Comparison to adaptive baselines (secondary)**
- Hypothesis: DPS captures most of the benefit of adaptive methods that require additional runtime signals.
- Validation: On the same engine and prompts, C is within 5% of D (confidence-threshold dynamic) on median session time, or C is better on the subset of turns with large \(\Delta L\) (prefill-dominated turns). We additionally report C vs E (SuffixDecoding) to test whether prefill-aware gating complements pattern-based drafting.

---

## Impact Statement

If successful, this work provides a simple, cache-aware control policy for speculative decoding in multi-turn tool-use sessions. Serving practitioners could reduce end-to-end session latency by skipping speculation on prefill-dominated turns without changing model weights or requiring online profiling.

---

## References

- [Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt) - StepFun Team, 2026
- [AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware Large Language Model Serving](./references/AdaSpec-Adaptive-Speculative-Decoding-for-Fast-SLO-Aware-Large-Language-Model-Serving/meta/meta_info.txt) - Huang et al., 2025
- [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](./references/BanditSpec-Adaptive-Speculative-Decoding-via-Bandit-Algorithms/meta/meta_info.txt) - Hou et al., 2025
- [LAPS: A Length-Aware-Prefill LLM Serving System](./references/LAPS-A-Length-Aware-Prefill-LLM-Serving-System/meta/meta_info.txt) - She et al., 2026
- [Batch speculative decoding: Done right](./references/Batch-speculative-decoding-Done-right/meta/meta_info.txt) - Zhang et al., 2025
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - Leviathan et al., 2023
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) - Chen et al., 2023
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) - Li et al., 2024
- [Medusa: Simple LLM Inference Acceleration via Multiple Decoding Heads](https://arxiv.org/abs/2311.08091) - Sun et al., 2023
- [Speculative Prefill](https://arxiv.org/abs/2502.02789) - (authors), 2025
- [DistServe](https://arxiv.org/abs/2401.09670) - (authors), 2024
- [vLLM](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023
- [SGLang](https://arxiv.org/abs/2312.07104) - Zheng et al., 2023
- [Preble](https://arxiv.org/abs/2407.00023) - (authors), 2024
- [AdaServe](https://arxiv.org/abs/2501.12162) - (authors), 2025
- [AdaSPEC](https://arxiv.org/abs/2510.19779) - Hu et al., 2025
- [MagicDec](https://arxiv.org/abs/2408.11049) - (authors), 2024
- [SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](./references/SuffixDecoding-Extreme-Speculative-Decoding-for-Emerging-AI-Applications/meta/meta_info.txt) - Oliaro et al., 2024
- [Token Recycling](https://arxiv.org/abs/2408.08696) - (authors), 2024
- [Set Block Decoding](https://arxiv.org/abs/2509.04185) - Gat et al., 2025
- [LoPA](https://arxiv.org/abs/2512.16229) - Xu et al., 2024
- [ToolLLM / ToolBench](https://arxiv.org/abs/2307.16789) - Qin et al., 2023
- [BFCL](https://arxiv.org/abs/2402.03301) - Patil et al., 2024
- [Batch speculative decoding Done right (code)](https://github.com/eBay/spec_dec) - eBay, 2025
