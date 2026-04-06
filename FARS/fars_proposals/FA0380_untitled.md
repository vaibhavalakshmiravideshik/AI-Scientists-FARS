# untitled

# Margin-Adaptive Grouped Verification for Deterministic LLM Serving

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language model (LLM) inference is widely deployed via **dynamic batching**, where requests are grouped on-the-fly to improve GPU utilization and throughput. Even with greedy decoding (temperature = 0), such serving stacks can be **non-deterministic**: the same prompt may produce different outputs across runs due to floating-point non-associativity interacting with batch-size-dependent GPU kernel reduction schedules.

Deterministic inference is important for integration testing, debugging, reproducible benchmarking, and reinforcement learning (RL) pipelines where inference/training mismatch can destabilize optimization. Recent work shows that mixed precision further exacerbates reproducibility failures on reasoning workloads (e.g., BF16 causing large accuracy variance across runtime configurations).

Two families of system approaches exist today. **Batch-invariant kernels** (used in deterministic modes of systems like SGLang and vLLM) enforce determinism by using a single reduction strategy independent of batching, but they impose a large and global performance cost. In contrast, **LLM-42** (verified speculation for determinism) keeps fast-path decoding efficient and enforces determinism via a periodic **decode–verify–rollback** loop with **grouped verification**, where multiple requests are verified together to amortize verification cost.

### The Problem

LLM-42’s performance depends heavily on the **verification granularity**: verifying shorter windows reduces rollback recomputation but increases verification overhead, while longer windows reduce verification overhead but increase rollback recomputation. LLM-42 proposes **grouped verification** (verifying small windows of multiple requests together) to partially mitigate this trade-off, but it still uses a **fixed verification window per request** (and thus a fixed group size) for the whole workload.

A fixed window is potentially suboptimal because rollback risk is not uniform: some requests (or regions of generation) are numerically stable, while others are fragile. At the same time, many “confidence-adaptive” methods in speculative decoding (e.g., entropy/margin-based draft-length controllers) focus on **speeding up non-deterministic decoding**; they do not address **deterministic inference under dynamic batching**, and their signals are often computed on non-deterministic draft passes.

This proposal asks whether deterministic serving can benefit from **per-request, runtime-adaptive verification granularity** without changing kernels.

### Key Insight and Hypothesis

Empirically, numerical nondeterminism tends to manifest when the model’s next-token decision is fragile: the top-1 and top-2 logits are close, so small perturbations can flip the argmax. We hypothesize that a **logit-margin statistic computed during the deterministic verifier replay** (e.g., top-1 minus top-2 logit gap) predicts whether the *next* decode–verify cycle will encounter a mismatch and rollback.

If this predictiveness holds, then a scheduler can allocate **smaller per-request verification windows** to low-margin (high-risk) requests and **larger windows** to high-margin (low-risk) requests, while keeping the **total verifier work per pass** fixed via grouped verification. This should reduce tail latency by cutting recomputation on fragile requests without paying small-window verification overhead on stable requests.

---

## Proposed Approach

> **Note on modes:** This proposal uses a **binary** adaptive policy (small vs large) to keep the verification scheduler simple and avoid excessive mode fragmentation.


### Overview

We modify an LLM-42-style serving stack to:

1. Compute a deterministic per-request **margin statistic** from the most recent verification pass.
2. Use this statistic to assign each request to a small/large verification window **for the next decode–verify cycle**.
3. Perform grouped verification using one of several **fixed verifier shapes** (window length × number of requests) that all verify the same total number of tokens per verifier pass.

### Method Details

**Base protocol (LLM-42):**
- Each deterministic request alternates between (i) fast-path decoding of a window of tokens under dynamic batching and (ii) verifier replay of that window under a fixed shape and reduction schedule.
- On verifier disagreement, the request rolls back to the last matching token and continues from a repaired KV-cache state.

**Margin computation (deterministic):**
- During each verifier pass for request r at cycle k, compute margin per token position:
  - \(m = \ell_{(1)} - \ell_{(2)}\), where \(\ell_{(1)}\) and \(\ell_{(2)}\) are the top-1 and top-2 logits.
- Aggregate to a scalar statistic available at the end of the pass, e.g.:
  - **MinMargin**: minimum margin over tokens committed in that verifier pass.
  - **LastMargin**: margin at the verifier-generated token (the new token immediately after the last matching position).

**Adaptive window assignment (next-cycle):**
- Using the margin statistic from cycle k, assign the window mode for cycle k+1:
  - If s < τ → **Small** window.
  - Else → **Large** window.
- Threshold τ is calibrated on a held-out slice of the workload.

**Fixed-shape grouped verifier modes (constant work per pass):**
We restrict to two verifier shapes that each verify **256 tokens per pass**:
- **Small mode**: window=32 tokens per request, group=8 requests → 256 tokens per pass.
- **Large mode**: window=64 tokens per request, group=4 requests → 256 tokens per pass.

**Scheduler and determinism considerations:**
- The mode decision uses only **verifier-computed** margins from already committed tokens, so it does not depend on non-deterministic fast-path logits.
- For each mode, the verifier batch size is fixed (8/4). If a mode queue is underfull, the verifier batch is padded with deterministic dummy requests to keep the verifier shape fixed.
- We will log verifier **batch-formation wait time** to quantify any scheduling overhead from mode fragmentation.

### Key Innovations

- **Deterministic risk signal**: use verifier-replay logit margins (not fast-path logits) as a per-request signal that can be stable across runs.
- **Adaptive verification granularity for determinism**: adapt verification window sizes in a determinism-enforcing DVR protocol, rather than adapting speculative decoding depth for speed.
- **Causal attribution via shuffled control**: preserve the same per-cycle window-mode distribution while breaking alignment between margins and mode assignments.

---

## Related Work

### Field Overview

**Deterministic LLM inference under batching.** Batch-invariant computation and deterministic kernels enforce batch-size-invariant reductions, but often incur substantial slowdowns and engineering overhead. Systems like SGLang and vLLM provide deterministic modes by swapping in batch-invariant kernels.

**Verified speculation / decode–verify–rollback for determinism.** LLM-42 proposes enforcing determinism through verifier replay and rollback, reusing optimized kernels for fast-path decoding and paying verification cost only for requests requiring determinism. Grouped verification amortizes verifier overhead across requests.

**Confidence-adaptive speculative decoding.** Many speculative decoding systems adapt draft length or acceptance thresholds using entropy/margins, bandits, or learned predictors. These methods aim at accelerating non-deterministic decoding and do not directly solve deterministic inference under dynamic batching.

**Numerical reproducibility diagnostics.** Recent work shows that mixed precision and fused kernels can create large evaluation variance, and that the underlying numerical noise can be structured rather than i.i.d.

### Related Papers

- **[LLM-42: Enabling Determinism in LLM Inference with Verified Speculation](https://arxiv.org/abs/2601.17768)**: Introduces decode–verify–rollback and grouped verification for deterministic serving without batch-invariant kernels.
- **[Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning](https://arxiv.org/abs/2506.09501)**: Shows large reproducibility variance under BF16 and links failures to small top-1/top-2 gaps; proposes LayerCast.
- **[Defeating Nondeterminism in LLM Inference (batch_invariant_ops)](https://github.com/thinking-machines-lab/batch_invariant_ops)**: Open-source batch-invariant kernels and analysis for deterministic inference under varying batch sizes.
- **[vLLM](https://arxiv.org/abs/2309.06180)**: High-throughput LLM serving with paged KV cache; includes speculative decoding and deterministic-mode support.
- **[SGLang](https://arxiv.org/abs/2312.07104)**: Structured generation runtime; provides deterministic mode via batch-invariant kernels.
- **[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)**: Continuous batching for serving, foundational for modern LLM schedulers.
- **[Sarathi-Serve](https://arxiv.org/abs/2308.16369)**: Chunked prefill and scheduling to improve throughput/latency.
- **[DistServe](https://arxiv.org/abs/2401.09670)**: Disaggregated prefill/decode serving to reduce interference.
- **[Splitwise](https://arxiv.org/abs/2311.18677)**: Splits prefill and decode phases for better utilization.
- **[FlexGen](https://arxiv.org/abs/2303.06865)**: Offloading-based serving for large models.
- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)**: Introduces speculative decoding (draft then verify) with distributional correctness.
- **[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)**: Speculative sampling for distributed inference with provable correctness.
- **[TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput](https://arxiv.org/abs/2406.14066)**: Closed-loop controller for speculative decoding proposal length using goodput feedback.
- **[Dynamic Speculation Lookahead (DISCO)](https://arxiv.org/abs/2405.04304)**: Learns a confidence-based stopping rule for adaptive speculative lookahead.
- **[TapOut: A Bandit-Based Approach to Dynamic Speculative Decoding](https://arxiv.org/abs/2511.02017)**: Uses bandit selection over heuristics including a LogitMargin stopping rule for drafting.
- **[Confidence-Modulated Speculative Decoding (CM-ASD)](https://arxiv.org/abs/2508.15371)**: Uses entropy/margin-based confidence to adapt speculative drafting and verification thresholds.
- **[AdaSD: Adaptive Speculative Decoding](https://arxiv.org/abs/2512.11280)**: Hyperparameter-free adaptive speculative decoding using entropy and Jensen–Shannon distance.
- **[SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths](https://arxiv.org/abs/2405.19715)**: MDP formulation with a learned acceptance head for adaptive candidate length.
- **[PACER: Blockwise Pre-verification for Speculative Decoding with Adaptive Length](https://arxiv.org/abs/2602.01274)**: Trainable pre-verifier to decide when to stop drafting.
- **[MARS: Margin-Aware Speculative Verification](https://arxiv.org/abs/2601.15498)**: Uses margin-based criteria to adapt speculative verification in standard speculative decoding.
- **[EAGLE](https://arxiv.org/abs/2401.15077)**: Feature-level speculative decoding with learned draft heads.
- **[Medusa](https://arxiv.org/abs/2401.10774)**: Adds multiple decoding heads to predict future tokens for acceleration.
- **[Hydra](https://arxiv.org/abs/2402.05109)**: Improves Medusa via sequentially dependent draft heads.
- **[SpecInfer](https://arxiv.org/abs/2305.09781)**: Tree-based speculative inference and verification for LLM serving.
- **[Batch Speculative Decoding: Done Right](https://arxiv.org/abs/2510.22876)**: Formalizes correctness invariants for batched speculative decoding and exposes common implementation bugs.
- **[DiFR: Inference Verification Despite Nondeterminism](https://arxiv.org/abs/2511.20621)**: Verifies inference traces despite nondeterminism via seeded randomness; shows high token match rates.
- **[On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix Multiplication](https://arxiv.org/abs/2511.00025)**: Empirically shows GPU numerical noise is structured and correlated.
- **[Deterministic Inference across Tensor Parallel Sizes (TBIK)](https://arxiv.org/abs/2511.17826)**: Tree-based invariant reductions for bitwise determinism across tensor-parallel sizes.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Batch-invariant kernels | Enforce a universal reduction order independent of batch size | He et al. BIO repo, [SGLang](https://arxiv.org/abs/2312.07104), [vLLM](https://arxiv.org/abs/2309.06180) | Throughput/latency under varying batch sizes; output equality across runs | Large fixed performance penalty; kernel maintenance burden |
| Verified speculation for determinism | Fast-path decode + deterministic verifier replay + rollback | [LLM-42](https://arxiv.org/abs/2601.17768) | Mixed deterministic ratios; rollback/recompute stats; tail latency | Fixed verification granularity; verifier introduces global pauses |
| TP-size invariant determinism | Fix intra/inter-GPU reduction trees for bitwise invariance | [TBIK](https://arxiv.org/abs/2511.17826) | Probability divergence across TP sizes; latency overhead | High overhead; specialized kernels/collectives |
| Adaptive speculative decoding | Adapt draft length/thresholds using confidence or learned predictors | [TurboSpec](https://arxiv.org/abs/2406.14066), [TapOut](https://arxiv.org/abs/2511.02017), [AdaSD](https://arxiv.org/abs/2512.11280), [SpecDec++](https://arxiv.org/abs/2405.19715), [PACER](https://arxiv.org/abs/2602.01274) | Tokens/s at bs=1; acceptance length; output equivalence | Targets speed, not determinism; confidence often computed on non-deterministic passes |
| Reproducibility diagnostics | Measure and mitigate evaluation variance due to numerics | [Give Me FP32](https://arxiv.org/abs/2506.09501), [FP noise structure](https://arxiv.org/abs/2511.00025) | Std@accuracy across runtime configs; divergence statistics | Does not provide serving-time determinism mechanism |

### Closest Prior Work

- **[LLM-42](https://arxiv.org/abs/2601.17768)**: Defines the decode–verify–rollback protocol and grouped verification; our work keeps the same protocol but makes verification granularity per-request and time-varying.
- **[Give Me FP32 or Give Me Death](https://arxiv.org/abs/2506.09501)**: Connects reproducibility failures to small top-1/top-2 gaps; we use this as a mechanism hypothesis for why margin should predict future mismatches.
- **[TurboSpec](https://arxiv.org/abs/2406.14066)**: Adapts speculative decoding parameters via goodput feedback; we adapt deterministic verification windows using a per-request margin signal instead of system-level throughput feedback.
- **[TapOut](https://arxiv.org/abs/2511.02017)** and **[CM-ASD](https://arxiv.org/abs/2508.15371)**: Use margin/entropy as confidence signals to adapt speculative decoding depth; our setting is single-model determinism, and the signal must be deterministic across runs.
- **[MARS](https://arxiv.org/abs/2601.15498)**: Uses margin-aware acceptance criteria in speculative decoding; we change verification cadence (granularity), not token acceptance rules, and target determinism under dynamic batching.

**Novelty Kill Search Summary:** Searched for combinations of “verified speculation / decode-verify-rollback” with “logit margin / confidence” and “adaptive verification window” across arXiv, OpenReview, and GitHub. The closest results were confidence-adaptive speculative decoding papers (TapOut, CM-ASD, MARS) which optimize speculative decoding speed, not deterministic inference. No work found that uses a deterministic margin signal to adapt verification granularity in LLM-42-style deterministic serving as of 2026-02-28.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [LLM-42](https://arxiv.org/abs/2601.17768) | Determinism via decode–verify–rollback + grouped verification | Fixed verification window per request | Adapt window per request per cycle | Reduce recomputation on fragile requests while keeping verifier work amortized |
| [TurboSpec](https://arxiv.org/abs/2406.14066) | Closed-loop control for speculative decoding goodput | Uses system-level feedback; not determinism | Use per-request deterministic margin signal | Better targeting: identify risky requests directly |
| [TapOut](https://arxiv.org/abs/2511.02017) | Bandit selection among drafting heuristics (incl. LogitMargin) | Confidence computed on non-deterministic draft pass; different objective | Margin from deterministic verifier; change verification granularity | Signal is stable across runs and directly tied to determinism failures |
| [CM-ASD](https://arxiv.org/abs/2508.15371) | Confidence-modulated drafting/verification thresholds | Optimizes speculative decoding speed; may allow small quality drift | Keep output determinism exact; only schedule verifier | Maintain strict determinism while improving serving efficiency |
| [MARS](https://arxiv.org/abs/2601.15498) | Margin-aware speculative verification rules | Changes acceptance criteria, not determinism | Keep LLM-42 acceptance, change window allocation | Addresses recomputation/verification tradeoff under strict determinism |

---

## Experiments

### Experimental Setup

**System / codebase:** Start from the authors’ public LLM-42 implementation (built on SGLang) and add logging + scheduler changes.

**Workloads:** Two traces from LLM-42 — **ShareGPT** (chat; low rollback) and **ArXiv** (long-form; high rollback) — each with 4096 requests.
- **Load setting:** ShareGPT uses **12 QPS** (LLM-42 §5.2). ArXiv uses a **token-rate-matched QPS** to keep offered load comparable:
  - \(q_{arxiv} = 12 \times \frac{\mathbb{E}[T_{sharegpt}]}{\mathbb{E}[T_{arxiv}]}\), where \(T\) is total (input+output) tokens per request, estimated from the trace.
- Context on rollback regimes (LLM-42 Table 4, verifier shape 8×64 @ 100% deterministic): **ShareGPT** has **96 rollbacks** and **0.32% recompute**, while **ArXiv** has **3351 rollbacks** and **10.97% recompute**.

**Model:** Llama-3.1-8B-Instruct (as in LLM-42 main experiments).

**Determinism evaluation:** For a fixed set of prompts and decoding parameters (temperature=0), run each deterministic method under multiple randomized arrival schedules (different request arrival jitters and batching dynamics) and require **byte-identical outputs per request** across runs.
- **Main determinism check:** 5 arrival schedules on **ShareGPT**.
- **Generalization check:** 3 arrival schedules on **ArXiv**.

**Paper-reported latency anchors (LLM-42 §5.2, H100; for context only):**
- At **12 QPS** on ShareGPT: SGLang-Deterministic P50 **4.64s**, P99 **28s** vs SGLang-Non-Deterministic P50 **2.15s**, P99 **13.2s**.
- At **18 QPS** on ShareGPT: SGLang-Deterministic P50 **10.6s**, P99 **71.1s** vs SGLang-Non-Deterministic P50 **2.84s**, P99 **17.4s**.

**Step 0 (go/no-go diagnostic within Baseline A run):**
- Instrument the verifier to log margin statistic s_{r,k} per request r per verification cycle k.
- Define the rollback event E_{r,k+1} as whether the *next* verification cycle triggers any rollback (or rollback ratio > 0).
- **Primary gate:** tokens/windows in the bottom 10% of s must have ≥2× rollback-event rate compared to the overall event rate.
- **Secondary statistic:** AUC for predicting E_{r,k+1} from s_{r,k}.

**Methods / conditions (≤3 main conditions):**

A) **Best fixed grouped verification (strong baseline):**
- Candidate fixed modes (all verify 256 tokens per pass):
  - (window=16, group=16)
  - (window=32, group=8)
  - (window=64, group=4)
- Choose the best fixed mode on a held-out slice (e.g., first 1024 requests), then report on the remaining 3072.

B) **Margin-adaptive grouped verification (ours):**
- Use deterministic verifier-margin statistic s_{r,k} to assign the next-cycle mode for each request (**small/large**) via a calibrated threshold τ.
- Keep verifier shapes fixed per mode; pad underfull batches with deterministic dummy requests.

C) **Shuffled-mode control (causal attribution):**
- Preserve the same counts of (**small/large**) assignments per cycle as in (B), but randomly permute mode labels across ready requests using a deterministic seed (e.g., hash(request_id, cycle)).

**Metrics:**
- End-to-end latency: P50, P90, P99.
- Throughput: output tokens per second.
- Verification overhead: number of verifier passes; verifier GPU time (if available).
- Rollback statistics: total rollbacks; recomputed tokens and recompute %.
- **Scheduler confound metric:** verifier batch formation wait time per mode.

**Baseline Ladder (REQUIRED):**
- Fixed grouped verification (A) as the strongest direct baseline.
- **Simple non-margin adaptive schedulers** (sanity baselines; deterministic):
  - **Length-based**: small window for long prompts / large expected output length; large window otherwise.
  - **Cycle-index-based**: small window for the first N verification cycles (when margins may be small), then large window.
  - **Global quantile**: small window for a fixed fraction q of requests each cycle, but chosen by a *non-margin* deterministic key (e.g., request_id hash) — isolates “mode fragmentation” without margin targeting.
- SGLang deterministic mode with batch-invariant kernels (global deterministic tax).
- SGLang non-deterministic mode (lower bound on latency/throughput; not deterministic).
- (Optional, if implementable) TBIK-style TP-invariant determinism as a strong-but-expensive comparator.

**Resource Estimate**:
- **Compute budget**: ≤ 80 A100 GPU-hours.
  - 3 main conditions × 3 arrival seeds (performance) + 2 extra seeds (determinism check) ≈ 15 workload runs.
  - Each run is inference-only over 4096 requests for an 8B model.
- **GPU memory**: 1×A100 80GB is sufficient for Llama-3.1-8B serving; optional multi-GPU TP if needed.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ShareGPT trace (LLM-42 setting) | Real chat prompts for serving simulation | P50/P99 latency, tokens/s, rollback/recompute stats, determinism | 4096 requests | https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered (or LLM-42 release) | LLM-42/SGLang evaluation harness |
| ArXiv trace (LLM-42 setting) | Long-form generation requests with higher rollback rates | P50/P99 latency, tokens/s, rollback/recompute stats, determinism | 4096 requests | LLM-42 release | LLM-42/SGLang evaluation harness |

### Main Results

#### Results Table

| Method | Base Model | Workload | P99 latency (s) | Tokens/s | Recompute % | Determinism | Source | Notes |
|---|---|---|---:|---:|---:|---|---|---|
| Fixed grouped (best of {16×16, 8×32, 4×64}) | Llama-3.1-8B | ShareGPT online @12 QPS | TBD | TBD | TBD | TBD | To be run | Baseline A selected on held-out slice |
| **Ours: margin-adaptive** | Llama-3.1-8B | ShareGPT online @12 QPS | TBD | TBD | TBD | TBD | To be run | Same verifier token budget per pass |
| Shuffled-mode control | Llama-3.1-8B | ShareGPT online @12 QPS | TBD | TBD | TBD | TBD | To be run | Same mode distribution as ours |
| SGLang deterministic (batch-invariant kernels) | Llama-3.1-8B | ShareGPT online @12 QPS | TBD | TBD | 0% | Deterministic by construction | To be run | Global deterministic tax |
| SGLang non-deterministic | Llama-3.1-8B | ShareGPT online @12 QPS | TBD | TBD | 0% | Non-deterministic | To be run | Lower bound |
| Fixed grouped (best of {16×16, 8×32, 4×64}) | Llama-3.1-8B | ArXiv online @12 QPS | TBD | TBD | TBD | TBD | To be run | Generalization: high rollback regime |
| **Ours: margin-adaptive** | Llama-3.1-8B | ArXiv online @12 QPS | TBD | TBD | TBD | TBD | To be run | Generalization: high rollback regime |
| Shuffled-mode control | Llama-3.1-8B | ArXiv online @12 QPS | TBD | TBD | TBD | TBD | To be run | Generalization: high rollback regime |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Margin feature = MinMargin vs LastMargin | Change s_{r,k} definition | One feature should predict next-window rollbacks better |
| Thresholding rule: fixed τ vs quantile-per-cycle | Change how requests are mapped to small vs large | Helps distinguish “risk targeting” vs “mode fragmentation” effects |
| Deterministic padding on/off (if safe) | Underfull batches are padded vs delayed | Helps isolate scheduling/wait-time confounds |

### Experimental Rigor

- Use 3 independent arrival-schedule seeds for performance metrics (mean ± std) on **ShareGPT**.
- Use 5 arrival-schedule seeds for determinism verification on **ShareGPT** (must be byte-identical across all 5).
- Use 3 arrival-schedule seeds on **ArXiv** as a generalization stress test (report mean ± std).
- Report calibration details for threshold τ and sensitivity of results to τ (±20%).
- **Boundary reporting:** report the realized rollback rate and recompute% of each workload; interpret results as applicable when rollback rates are non-trivial (e.g., ≳0.5–1% recompute) and mode fragmentation overhead is small.

---

## Success Criteria

**Hypothesis** (directional — what we expect):
- Low verifier-margin cycles are enriched for future rollbacks.
- Margin-adaptive window sizing reduces recomputation and improves P99 latency compared to the best fixed grouped configuration at comparable verifier work.

**Decision Rule** (concrete — when to stop):
- **Proceed** if all hold:
  1) Step-0 gate passes on **ShareGPT** (bottom-decile margin has ≥2× rollback-event rate), and
  2) method (B) improves **ShareGPT** P99 latency by ≥10% over baseline (A) with ≤3% P50 regression, and
  3) determinism passes on ShareGPT (5/5 arrival schedules byte-identical), and
  4) **generalization check**: on **ArXiv**, method (B) is **not worse** than (A) on P99 (≤3% regression) and shows reduced recomputation.
- **Pivot** if: Step-0 gate passes but (B) does not beat (A) on ShareGPT; try alternative statistics (e.g., EWMA of margins) or a hyperparameter-free quantile rule.
- **Refute** if any hold: Step-0 gate fails; shuffled control (C) matches (B) within noise (gains not from margin targeting); or determinism fails under 5/5 arrival schedules.

---

## Impact Statement

If successful, this work would provide a low-engineering, kernel-agnostic way to reduce tail latency for deterministic LLM serving by adapting verification granularity per request. This could make deterministic inference practical for more production workloads (testing, evaluation, regulated settings) without paying the full global cost of batch-invariant kernels.

---

## References

- [LLM-42: Enabling Determinism in LLM Inference with Verified Speculation](https://arxiv.org/abs/2601.17768) - Gond et al., 2026
- [Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning](https://arxiv.org/abs/2506.09501) - Yuan et al., 2025
- [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) - He et al., 2025
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) - Zheng et al., 2023
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - Yu et al., 2022
- [Sarathi-Serve: Efficient LLM Serving with Chunked Prefills](https://arxiv.org/abs/2308.16369) - Agrawal et al., 2023
- [DistServe: Disaggregating Prefill and Decoding for Distributed LLM Serving](https://arxiv.org/abs/2401.09670) - Zhong et al., 2024
- [Splitwise: Efficient Generative LLM Serving with Phase Splitting](https://arxiv.org/abs/2311.18677) - Patel et al., 2023
- [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865) - Sheng et al., 2023
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - Leviathan et al., 2022
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) - Chen et al., 2023
- [TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput](https://arxiv.org/abs/2406.14066) - Kim et al., 2024
- [Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models](https://arxiv.org/abs/2405.04304) - M. et al., 2024
- [TapOut: A Bandit-Based Approach to Dynamic Speculative Decoding](https://arxiv.org/abs/2511.02017) - X. et al., 2025
- [Confidence-Modulated Speculative Decoding for Large Language Models](https://arxiv.org/abs/2508.15371) - Sen et al., 2025
- [AdaSD: Adaptive Speculative Decoding for Efficient Language Model Inference](https://arxiv.org/abs/2512.11280) - Lu et al., 2024
- [SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths](https://arxiv.org/abs/2405.19715) - Huang et al., 2024
- [PACER: Blockwise Pre-verification for Speculative Decoding with Adaptive Length](https://arxiv.org/abs/2602.01274) - Liu et al., 2026
- [Unleashing the Power of Speculative Decoding via Margin-Aware Speculative Verification (MARS)](https://arxiv.org/abs/2601.15498) - X. et al., 2026
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) - Li et al., 2024
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) - Cai et al., 2024
- [Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding](https://arxiv.org/abs/2402.05109) - Ankner et al., 2024
- [SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference](https://arxiv.org/abs/2305.09781) - Miao et al., 2023
- [Batch Speculative Decoding: Done Right](https://arxiv.org/abs/2510.22876) - Zhang et al., 2025
- [DiFR: Inference Verification Despite Nondeterminism](https://arxiv.org/abs/2511.20621) - Karvonen et al., 2025
- [On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix Multiplication](https://arxiv.org/abs/2511.00025) - Yashwanth, 2025
- [Deterministic Inference across Tensor Parallel Sizes That Eliminates Probability Gaps](https://arxiv.org/abs/2511.17826) - Zhang et al., 2025
