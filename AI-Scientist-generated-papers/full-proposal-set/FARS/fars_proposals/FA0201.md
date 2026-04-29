# untitled

# Cap-and-Spill: Two-Pass CUDA-Graph MoE Dispatch Without Worst-Case Maxcounts Padding

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Mixture-of-Experts (MoE) transformer layers reduce compute by routing each token to a small subset of expert MLPs. In multi-GPU deployments, experts are typically sharded across ranks ("expert parallelism"), so each MoE layer must **dispatch** token activations to the ranks that own the selected experts, and later **combine** expert outputs back to the original rank ordering. This many-to-many movement is commonly implemented with GPU collectives such as **AllToAll** (each rank sends distinct data to every other rank) and **AllToAllv**, its variable-length variant.

A practical challenge is that MoE routing is input-dependent, so the per-peer send counts vary from step to step. Meanwhile, inference engines often use **CUDA graphs** to reduce CPU overhead and jitter by capturing a fixed kernel/collective schedule and replaying it for many decode steps. CUDA graphs (and common communication APIs) work best when message metadata (buffer addresses, sizes) are fixed at capture time.

Meta’s NCCLX paper (Meta’s extended NCCL library) documents this tension explicitly for MoE inference: when send counts cannot be computed at graph creation time, synchronizing GPU→CPU to obtain exact counts is not allowed in CUDA-graph mode, so systems instead send **maximum possible counts (“maxcounts”)** and pad with unused data, which increases latency and bandwidth usage; their solution is a custom GPU-resident collective (AllToAllvDynamic) that reads updated counts by reference on the GPU at execution time and yields large decode-time reductions (Table 3) ("Collective Communication for 100k+ GPUs") (./references/Collective-Communication-for-100k+-GPUs/meta/meta_info.txt).

However, GPU-resident/dynamic-metadata collectives are not generally available in open-source NCCL/PyTorch stacks. Recent open-source MoE communication work instead tends to either (i) keep metadata fixed by padding to a worst-case upper bound (e.g., capacity padding in CUDA-graph-compatible MoE training/inference stacks), or (ii) require substantial custom kernels / device-initiated communication (e.g., FlashMoE). This leaves a gap for a **simple, drop-in pattern** that reduces worst-case padding in CUDA graph mode using only standard collectives.

### The Problem

**Problem:** In CUDA-graph MoE dispatch, the common workaround is to choose a worst-case fixed upper bound on per-peer token counts and always communicate at that bound ("maxcounts"). This can waste bandwidth and inflate dispatch latency when real routing distributions are far from worst case.

More concretely:
- In NCCLX’s case study (MetaShuffling token shuffling), send counts depend on the router kernel and cannot be known at graph creation time; NCCLX states that in CUDA-graph mode, instead of synchronizing on send counts, systems must send **maxcounts** large enough for the worst case and thus send large padding values, harming inference latency (./references/Collective-Communication-for-100k+-GPUs/meta/meta_info.txt).
- DeepEP and Megatron-MoE documentation likewise enforce **fixed maximum dispatch tokens** for CUDA-graph-compatibility, and discuss buffer-state restoration / static-shape constraints; these fixed maxima are functional but may be conservative.

A practitioner-facing desideratum is therefore:

> **Reduce the always-sent padding volume in CUDA-graph MoE dispatch without dropping tokens or introducing a custom dynamic-metadata communication library.**

### Key Insight and Hypothesis

**Key insight:** MoE per-peer token counts are typically heavy-tailed but rarely hit the absolute worst case. If we set a smaller fixed per-peer cap \(C\) (e.g., a high quantile of observed per-peer send counts under a given model + serving workload) and handle the rare over-cap assignments with an additional fixed-shape pass, we can reduce the always-sent padded payload while keeping collectives CUDA-graph-capturable.

**Hypothesis:** A two-pass **cap-and-spill** dispatch using the *same fixed-shape CUDA graph replayed twice* will recover a large fraction of the latency gap between worst-case-padded CUDA-graph dispatch and oracle eager dispatch (dynamic split sizes), while:
1) keeping token dispatch **exact** (no token dropping),
2) keeping overflow rare enough that p99 latency does not regress, and
3) adding minimal packing/compaction overhead.

The hypothesis could fail if (i) real routing is close to worst case so padding is not the bottleneck, (ii) overflow is frequent even at high-quantile \(C\), or (iii) overflow compaction kernels dominate the dispatch critical path.

---

## Proposed Approach

### Overview

We propose **Cap-and-Spill**: a CUDA-graph-compatible MoE token dispatch pattern that replaces a single worst-case-padded AllToAll with **one fixed-cap main pass** plus an optional **second fixed-cap spill pass**. Both passes use standard `torch.distributed.all_to_all_single` (or equivalent NCCL AllToAll) with **static split sizes** and **static buffer addresses**, enabling CUDA graph capture.

### Method Details

We describe the method for the dispatch phase (combine is symmetric).

**Inputs:**
- `tokens`: per-rank token activations of shape `[T_local, d_model]`.
- `route`: per-token **top-\(k\) routing** (each token is sent to its \(k\) highest-scoring experts; e.g., Mixtral top-2), plus a known mapping from experts to destination ranks.
- `C`: fixed per-destination capacity (tokens per peer per pass), chosen from a routing-trace quantile.

**Baseline (maxcounts CUDA-graph):**
- Allocate send buffer sized for `world_size * C_max` tokens, where `C_max` is a worst-case bound (“maxcounts”).
- For each destination rank `j`, pack up to `C_max` tokens and pad with unused entries.
- Replay a CUDA graph that performs AllToAll with fixed split sizes `C_max`.

**Cap-and-Spill (ours):**

1) **Pass 1 (cap):**
   - Pack up to `C` tokens per destination rank into `send_main` (shape `[world_size * C, d_model]`), and write the over-cap tokens’ indices/values into per-destination overflow buffers (or an overflow index list).
   - Replay a CUDA graph containing one AllToAll (fixed split sizes `C`) from `send_main → recv_main`.

2) **Pass 2 (spill, optional but still CUDA-graph):**
   - If any destination overflow count > 0, compact (on GPU) up to `C` overflow tokens per destination into `send_spill` (same fixed shape as `send_main`), zero-pad the rest.
   - Replay the *same* CUDA graph a second time on `send_spill → recv_spill`.

3) **Unpack / reorder:**
   - Use stored indices to scatter `recv_main` and `recv_spill` into the exact per-expert (or per-token) layout expected by the expert compute. If Pass 2 is unused, this reduces to the Pass-1 unpack.

**Exactness condition:** The concatenation of tokens delivered to each destination in Pass 1 and Pass 2 must be identical (up to ordering that is corrected by the unpack step) to oracle eager dispatch where each destination receives exactly its true token count.

**Choosing \(C\):**
- Compute routing traces for a fixed model and prompt distribution.
- Let `count[r, j]` be the number of tokens rank `r` needs to send to destination `j` (aggregated over top-\(k\) routes).
- Choose \(C\) as the empirical per-peer quantile, e.g. \(C = \mathrm{Quantile}_{0.99}(\{count[r,j]\})\).

### Key Innovations

- **Two-pass fixed-shape dispatch as an approximation to dynamic-metadata collectives:** approximates the “use actual counts” behavior (AllToAllvDynamic) using only standard CUDA-graph-compatible AllToAll primitives.
- **Decision-theoretic cap selection:** selects the fixed capacity \(C\) based on observed routing distributions rather than worst-case reasoning.
- **Verification-first metric:** evaluates improvement relative to the *measured gap* between maxcounts-graph and oracle eager, rather than an arbitrary latency target.

---

## Related Work

### Field Overview

MoE system performance is often bottlenecked by dispatch/combine communication, especially under expert parallelism. The literature includes: (i) algorithmic improvements to all-to-all scheduling and overlap, (ii) MoE execution engines that fuse packing/compute/communication, (iii) routing-side methods that reduce imbalance (often by changing computation), and (iv) infrastructure features such as CUDA graphs that trade dynamism for lower overhead.

Our proposal sits at the intersection of (ii) and (iv): it addresses a specific CUDA-graph constraint (static metadata) with a communication-level design that maintains exactness (no token dropping) and does not require adopting a new communication stack.

### Related Papers

- **[Collective Communication for 100k+ GPUs](./references/Collective-Communication-for-100k+-GPUs/meta/meta_info.txt)**: Introduces NCCLX and AllToAllvDynamic, motivating maxcounts padding in CUDA graphs and showing large MoE decode-time gains when dynamic metadata is supported.
- **[FAST: An Efficient Scheduler for All-to-All GPU Communication](./references/FLASH-Fast-All-to-All-Communication-in-GPU-Clusters/meta/meta_info.txt)**: Optimizes all-to-all scheduling under skewed patterns; complementary to our focus on static-metadata constraints.
- **[FlashMoE: Fast Distributed MoE in a Single Kernel](./references/FlashDMoE-Fast-Distributed-MoE-in-a-Single-Kernel/meta/meta_info.txt)**: Uses a persistent kernel + device-initiated communication to eliminate launch overhead and padding; much higher engineering complexity than our approach.
- **[Capacity-Aware Inference](./references/Capacity-Aware-Inference-Mitigating-the-Straggler-Effect-in-Mixture-of-Experts/meta/meta_info.txt)**: Mitigates stragglers via token drop/reroute; it changes computation, whereas we aim for exact dispatch.
- **[Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382)**: Implements MoE systems with overlap and all-to-all-v support; does not target CUDA-graph maxcounts padding.
- **[Lancet: Accelerating Mixture-of-Experts Training via Whole-Graph Computation–Communication Overlapping](https://arxiv.org/abs/2404.19429)**: Whole-graph overlap/pipelining for MoE training; orthogonal to our metadata constraint.
- **[Lina: Accelerating Distributed MoE Training and Inference](https://www.usenix.org/system/files/atc23-li-jiamin.pdf)**: Partitions/pipelines all-to-all with compute; does not propose multi-pass fixed-cap to avoid maxcounts.
- **[DeepSpeed-MoE](https://arxiv.org/abs/2201.05596)**: Practical MoE training system with expert parallelism; representative baseline MoE dispatcher.
- **[FasterMoE](https://arxiv.org/abs/2106.05974)**: MoE system optimizations for dispatch/compute; focuses on throughput rather than CUDA-graph metadata.
- **[GShard](https://arxiv.org/abs/2006.16668)**: Foundational conditional computation and MoE training at scale.
- **[Switch Transformer](https://arxiv.org/abs/2101.03961)**: MoE routing with capacity and token dropping during training.
- **[Mixtral of Experts](https://arxiv.org/abs/2401.04088)**: Open MoE model family used for routing-trace extraction and evaluation.
- **[DeepSeek-V2](https://arxiv.org/abs/2405.04434)**: Modern MoE model family that motivates optimized expert-parallel dispatch.
- **[DeepSeek-V3 report](https://github.com/deepseek-ai/DeepSeek-V3)**: Documents fine-grained experts and motivates DeepEP-style dispatch.
- **[DeepEP](https://github.com/deepseek-ai/DeepEP)**: MoE communication library with CUDA-graph-compatible low-latency dispatch requiring fixed max dispatch tokens.
- **[MetaShuffling](https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/)**: Token shuffling approach for MoE inference; cited by NCCLX as the motivating case study.
- **[NCCL CUDA graph usage](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)**: Documents constraints when capturing NCCL operations in CUDA graphs.
- **[PyTorch CUDA graphs (CUDAGraph Trees)](https://docs.pytorch.org/docs/stable/user_guide/torch.compiler_cudagraph_trees.html)**: Describes CUDA graph capture/replay constraints in PyTorch.
- **[vLLM CUDA graphs design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)**: Shows practical padding/bucketing strategies used to fit dynamic workloads into CUDA graphs.
- **[Rectify-Router (Turn Waste into Worth)](https://arxiv.org/abs/2402.12399)**: Two-stage router post-processing to reduce dropped tokens/padding; different objective (routing) than our communication-metadata focus.
- **[Highly Efficient Alltoall and Alltoallv Communication Algorithms for GPU Systems](https://par.nsf.gov/servlets/purl/10334472)**: Two-phase alltoallv metadata exchange and GPU-aware designs; provides background on size-exchange overhead.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Dynamic-metadata collectives | Metadata on GPU, modifiable until execution | NCCLX / AllToAllvDynamic | End-to-end decode latency (paper table) | Requires custom comm stack; not generally available |
| Overlap / scheduling for all-to-all | Pipelines all-to-all with compute; schedules skew | Tutel; Lina; Lancet; FAST | Training step time; microbench all-to-all | Does not remove CUDA-graph static-metadata maxcounts issue |
| Fully fused MoE kernels | Persistent kernels + device-initiated comm | FlashMoE; DeepEP | Forward latency, utilization | High engineering complexity; fixed-cap constraints remain for graphs |
| Routing-side imbalance mitigation | Drop/reroute tokens to reduce stragglers | Capacity-Aware Inference; Switch | Task accuracy + latency | Changes computation; may degrade or alter outputs |

### Closest Prior Work

- **NCCLX / AllToAllvDynamic** (./references/Collective-Communication-for-100k+-GPUs/meta/meta_info.txt): Solves the maxcounts problem by taking metadata by reference on GPU and reading updated counts at execution time, avoiding worst-case padding. Our approach differs by keeping to standard AllToAll with fixed split sizes and approximating dynamism via a second fixed-cap pass.
- **DeepEP** (https://github.com/deepseek-ai/DeepEP): Provides CUDA-graph-compatible low-latency MoE dispatch, but still requires a fixed maximum dispatch bound and discusses buffer state restoration. We propose a general fixed-cap multi-pass pattern that targets conservative max bounds.
- **MetaShuffling** (https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/): Provides optimized token packing/shuffling kernels; NCCLX uses it as the case study. Our contribution is orthogonal: a communication-level two-pass pattern to reduce maxcounts padding under CUDA graphs.

**Novelty Kill Search Summary:** Searched for combinations of “CUDA graph + MoE + alltoall padding”, “maxcounts cudagraph alltoallv”, “two-pass MoE dispatch overflow”, and “cap-and-spill MoE dispatch”, and checked local KB + all proposal drafts for `alltoall*`, `cudagraph`, `maxcounts`, `AllToAllvDynamic`. No prior work proposing a **two-pass fixed-cap CUDA-graphed AllToAll** pattern for MoE dispatch was found as of 2026-02-21.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| NCCLX AllToAllvDynamic | Uses GPU-resident metadata to send actual counts in CUDA graphs | Custom comm stack; not open-source | Use stock AllToAll with fixed split sizes | Approximates “actual counts” with minimal integration burden |
| DeepEP | Optimized MoE dispatch; CUDA-graph-compatible with fixed max tokens | Still needs conservative max bound; buffer state issues | Replace single worst-case bound with high-quantile cap + spill pass | Lower mean padding while keeping static-shape execution |
| MetaShuffling | Efficient packing/shuffling kernels for MoE | Does not directly address maxcounts communication padding | Keep existing pack kernels but change communication pattern | Reduces always-sent bytes even if packing is already optimized |
| Tutel/Lina/Lancet/FAST | Overlap/scheduling for all-to-all under MoE | Orthogonal to static-metadata maxcounts in CUDA graphs | Target static-metadata inefficiency directly | Works even when overlap is already applied |

---

## Experiments

### Experimental Setup

**Goal:** Evaluate whether cap-and-spill recovers a large fraction of the latency gap between worst-case-padded CUDA-graph dispatch and oracle eager dispatch, under **real MoE routing traces**.

**Hardware:** 8×A100 (single node, NVLink).

**Implementation sketch:**
- Use `torch.distributed` with NCCL backend.
- Pre-allocate fixed-size send/recv buffers for:
  - baseline maxcounts graph (`C_max`)
  - cap-and-spill graph (`C`)
- Capture CUDA graphs after warmup:
  - Baseline graph: one AllToAll on `send_max → recv_max`.
  - Ours graph: one AllToAll on `send_cap → recv_cap`, replayed once or twice per step.
- Pack/unpack implemented with torch/Triton kernels; measure end-to-end dispatch time including packing and unpacking.

**Routing traces (no human labeling):**
- Extract router assignments from an open MoE model (Mixtral-8x7B) on a fixed public prompt set (e.g., a small slice of ShareGPT or WikiText). Log per-rank per-peer send counts under a fixed EP partition (experts evenly partitioned across 8 ranks). Use these logged counts to drive packing and to define `C_max` (max over trace) and `C` (quantile over trace).

**Baseline Ladder (REQUIRED):**
- **Baseline 1 (worst-case padded, CUDA graph):** fixed `C_max` per peer (“maxcounts”) graph replay.
- **Baseline 2 (oracle eager):** eager `all_to_all_single` with true per-peer split sizes per step (no CUDA graph).
- **Baseline 3 (literature upper bound, not runnable here):** NCCLX AllToAllvDynamic (reported decode-time improvements; Table 3) as context.
- **Ours:** cap-and-spill with fixed `C=Quantile_q(count)` and optional second graph replay.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|------|
| Mixtral-8x7B-Instruct | MoE (8 experts) | https://huggingface.co/mistralai | Used only to extract routing traces; not fine-tuned |

**Training Data (if applicable):**

No training data needed – inference-only microbenchmark.

**Resource Estimate**:
- **Compute budget**: 32–96 GPU-hours total (trace extraction + compile/warmup + benchmarking over several `q` values and repeats).
- **GPU memory**: dominated by Mixtral inference for trace extraction; dispatch microbench buffers are small.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| MoE dispatch microbenchmark (real routing traces) | Measures communication+packing latency of MoE dispatch under CUDA graphs | mean / p95 / p99 step latency; bytes sent; overflow rate; gap-recovery ratio | N/A | routing traces derived from Mixtral prompts | custom script in this proposal |

### Main Results

We report all metrics as averages across many iterations and repeated runs (see Experimental Rigor).

#### Results Table

| Method | Base Model | Benchmark | Mean dispatch latency (µs) | p99 dispatch latency (µs) | Overflow rate (%) | Source | Notes |
|--------|------------|-----------|----------------------------|---------------------------|------------------|--------|------|
| Baseline 1: CUDA-graph maxcounts (`C_max`) | N/A | dispatch microbench | **TBD** | **TBD** | 0 | - | To be measured |
| Baseline 2: Eager oracle (true split sizes) | N/A | dispatch microbench | **TBD** | **TBD** | 0 | - | To be measured |
| **Ours: cap-and-spill (q=0.99)** | N/A | dispatch microbench | **TBD** | **TBD** | **TBD** | - | To be verified |

Additional context (not directly comparable to our microbench): NCCLX reports 15–83% end-to-end decode-time reductions for MoE inference with AllToAllvDynamic vs a baseline that includes fixed-metadata collectives (Table 3) (./references/Collective-Communication-for-100k+-GPUs/meta/meta_info.txt).

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | 2-pass cap-and-spill | Best mean latency without p99 regression |
| 1-pass only | Disable spill (drop overflow tokens locally or treat as error) | Demonstrates that spill handling is required for exactness |
| Vary quantile `q` | `q ∈ {0.95, 0.99, 0.995}` | Exposes Pareto frontier: mean latency vs overflow rate |
| Measure compaction cost only | Time only the overflow compaction/packing step | Validates <10% overhead gate |

### Experimental Rigor

**Variance & Reproducibility:**
- Latency microbenchmarks are noisy; run each condition for ≥10 warmup + ≥200 timed iterations, repeated across 5 independent process restarts (treat restarts as “seeds”). Report mean ± std across restarts.
- Fix the routing-trace file and iteration order to make runs reproducible.

**Validity & Controls:**
- **Control 1 (gap calibration):** Report `C_max / mean(count)` and `C_max / Quantile_q(count)` from traces before benchmarking, to contextualize how much padding is being removed.
- **Control 2 (exactness):** Verify that unpacked outputs from cap-and-spill match oracle eager dispatch outputs for the same trace (bitwise equality if possible; otherwise max absolute diff).
- **Control 3 (graph replay overhead):** Measure CUDA graph replay overhead separately to validate the two-pass crossover analysis.

**Fair Comparison Conditions:**
- Same ranks, same NCCL backend, same stream setup.
- For maxcounts vs cap-and-spill, compare under CUDA-graph replay in both cases.

---

## Success Criteria

**Hypothesis** (directional): Cap-and-spill will reduce mean dispatch latency by removing always-sent padding, while keeping overflow rare enough that p99 does not regress.

**Decision Rule** (concrete):
- **Continue/Proceed** if, on real Mixtral routing traces:
  1) Mean latency improvement is **≥ max(10% absolute, 50% of the measured gap)** between (Baseline 1) maxcounts-graph and (Baseline 2) oracle eager; AND
  2) Additional overflow-compaction overhead is **<10%** of Baseline-1 end-to-end dispatch time; AND
  3) Overflow (2-pass) rate is **<10%** at `q=0.99`; AND
  4) p99 latency is **not worse by >10%** vs Baseline 1.
- **Pivot** if improvements are marginal (<10% absolute) but compaction is cheap: try higher quantile (`q=0.995`) or cap derived from a larger trace window.
- **Refute** if compaction overhead ≥10% or overflow rate ≥10% at `q=0.99`, or if p99 latency regresses >10%.

---

## Impact Statement

If successful, Cap-and-Spill provides a practical recipe for MoE inference/training engineers who rely on CUDA graphs: they can reduce conservative maxcounts padding without adopting a custom GPU-resident collective library, improving mean latency while retaining exactness.

---

## References

- Collective Communication for 100k+ GPUs (./references/Collective-Communication-for-100k+-GPUs/meta/meta_info.txt)
- FAST: An Efficient Scheduler for All-to-All GPU Communication (./references/FLASH-Fast-All-to-All-Communication-in-GPU-Clusters/meta/meta_info.txt)
- FlashMoE: Fast Distributed MoE in a Single Kernel (./references/FlashDMoE-Fast-Distributed-MoE-in-a-Single-Kernel/meta/meta_info.txt)
- Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts (./references/Capacity-Aware-Inference-Mitigating-the-Straggler-Effect-in-Mixture-of-Experts/meta/meta_info.txt)
- Tutel: Adaptive Mixture-of-Experts at Scale (https://arxiv.org/abs/2206.03382)
- Lancet: Accelerating Mixture-of-Experts Training via Whole-Graph Computation–Communication Overlapping (https://arxiv.org/abs/2404.19429)
- Lina: Accelerating Distributed MoE Training and Inference (https://www.usenix.org/system/files/atc23-li-jiamin.pdf)
- DeepSpeed-MoE (https://arxiv.org/abs/2201.05596)
- FasterMoE (https://arxiv.org/abs/2106.05974)
- GShard (https://arxiv.org/abs/2006.16668)
- Switch Transformer (https://arxiv.org/abs/2101.03961)
- Mixtral of Experts (https://arxiv.org/abs/2401.04088)
- DeepSeek-V2 (https://arxiv.org/abs/2405.04434)
- DeepSeek-V3 repo (https://github.com/deepseek-ai/DeepSeek-V3)
- DeepEP (https://github.com/deepseek-ai/DeepEP)
- MetaShuffling (https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/)
- NCCL with CUDA Graphs (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)
- PyTorch CUDA graph trees (https://docs.pytorch.org/docs/stable/user_guide/torch.compiler_cudagraph_trees.html)
- vLLM CUDA graphs design (https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- Turn Waste into Worth: Rectifying Top-k Router of MoE (https://arxiv.org/abs/2402.12399)
- Highly Efficient Alltoall and Alltoallv Communication Algorithms for GPU Systems (https://par.nsf.gov/servlets/purl/10334472)
