# untitled

# Task-Aware Early Termination for HNSW via Label-Histogram Stabilization

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Vector similarity search (VSS) underlies image/face search, dense text retrieval, and recommendation candidate retrieval; latency matters at scale.

Benchmarks usually report synthetic **Recall@K** (fraction of the true top-K neighbors returned under a metric) versus latency. Iceberg shows this can diverge from downstream utility: on labeled kNN tasks, **Label Recall@K** (fraction of retrieved items whose label matches the query label) can saturate well before perfect synthetic recall ([Iceberg](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt)).

### The Problem

Recent query-adaptive efficiency methods for ANNS early-terminate to meet **synthetic recall** targets:

- **[DARTH](./references/DARTH-Declarative-Recall-Through-Early-Termination-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt)**: uses a learned gradient-boosted decision tree (GBDT) to predict current synthetic recall during search and stops when a target is reached.
- **[Ada-ef](./references/Distribution-Aware-Exploration-for-Adaptive-HNSW-Search/meta/meta_info.txt)**: uses distributional analysis to choose a per-query HNSW search budget (`efSearch`) to hit a target synthetic recall.
- **[pEE](./references/Early-Exit-Strategies-for-Approximate-kNN-Search-in-Dense-Retrieval/meta/meta_info.txt)**: uses patience on **result-set (ID) stability** to stop IVF probing early in dense retrieval.

These methods do not use task labels, so they may keep searching after task utility saturates. Conversely, a label-aware rule could be **confidently wrong** on boundary queries, so we evaluate tail latency and difficulty-stratified errors.

### Key Insight and Hypothesis

Iceberg’s funnel analysis suggests a deployment regime where extra distance computations change neighbor identities but not downstream labels. We hypothesize that, in label-driven kNN tasks, the **label histogram** of the current top-K results stabilizes earlier than the exact top-K identity set, and that detecting this stabilization online during HNSW search can reduce latency while preserving **Label Recall@K**.

---

## Proposed Approach

### Overview

We propose a training-free early-termination rule for **HNSW (Hierarchical Navigable Small World graphs)** that stops search when the **label distribution** over the current top-K results becomes stable. The rule is designed to be close to parameter-free by using a stability threshold that scales with K.

### Method Details

Setting: HNSW search with a maximum budget `efSearch_max` (the HNSW search expansion parameter controlling how many candidates are explored). Let `TopK_t` be the current top-K list at checkpoint `t`, and `L(x)` be the task label for vector `x`.

1. **Checkpoint schedule**: every `B` distance computations (default `B=K`), compute the label histogram over `TopK_t`:
   - `p_t(c) = (1/K) * |{x ∈ TopK_t : L(x)=c}|`.
2. **Stability score**: `Δ_t = ||p_t − p_{t−1}||_1`.
3. **Stop rule (Label-Stability Early Exit)**: stop if `Δ_t ≤ 2/K` for `Δ_patience=2` consecutive checkpoints, after a 2-checkpoint warmup.

Interpretation: moving one item in Top-K from label a→b changes L1 mass by `2/K`, so the threshold corresponds to approximately “≤1 label swap” between checkpoints.

**Overhead control**: also run a “no-early-exit” variant that computes checkpoints but always runs to `efSearch_max`, to measure bookkeeping overhead.

### Key Innovations

- Uses **task labels** as the early-exit signal (proxying Label Recall@K saturation), rather than synthetic recall prediction or ID-set stability.
- Uses a **K-scaled threshold** (`2/K`) to avoid dataset-specific entropy calibration.
- Includes a required **hard-query stratification** to characterize “confidently wrong” early exits.

---

## Related Work

### Field Overview

ANNS indexes are typically graph-based (HNSW, DiskANN/Vamana, NSG) or partition/quantization-based (IVF/PQ, ScaNN, RaBitQ). While benchmarks traditionally optimize synthetic recall/latency, task-centric benchmarks like Iceberg motivate system optimizations aligned with downstream utility. Early termination methods exist for synthetic recall targets (DARTH, Ada-ef, LAET) and for ID-stability (pEE), but task-label-driven termination remains underexplored.

### Related Papers

- **[Iceberg](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt)**: Task-centric VSS benchmark; reports synthetic-vs-task metric gaps and suggests task-aware early stop.
- **[Iceberg code](./references/GitHub-ZJU-DAILY-Iceberg/meta/meta_info.txt)**: Open-source harness + released datasets/configs.
- **[DARTH](./references/DARTH-Declarative-Recall-Through-Early-Termination-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt)**: Learned early termination to meet a declarative **synthetic recall** target.
- **[Ada-ef](./references/Distribution-Aware-Exploration-for-Adaptive-HNSW-Search/meta/meta_info.txt)**: Rule-based per-query `efSearch` for a declarative **synthetic recall** target.
- **[pEE](./references/Early-Exit-Strategies-for-Approximate-kNN-Search-in-Dense-Retrieval/meta/meta_info.txt)**: Patience early exit from **ID-set stability** in IVF dense retrieval.
- **[Patience in Proximity](./references/Patience-in-Proximity-Early-Termination-HNSW/meta/meta_info.txt)**: Early-terminates HNSW traversal when the top-k neighbor set saturates (high overlap across iterations).
- **[HNSW](https://arxiv.org/abs/1603.09320)**: Widely used proximity-graph ANNS index.
- **[DiskANN/Vamana](https://arxiv.org/abs/1907.05641)**: Graph-based ANN designed for high performance at scale.
- **[NSG](https://arxiv.org/abs/1707.00143)**: Navigating spreading-out graph, related to HNSW.
- **[FAISS](https://arxiv.org/abs/2401.08281)**: Library implementing IVF/PQ and other similarity search primitives.
- **[ScaNN](https://arxiv.org/abs/1908.10396)**: Production-oriented partition-based ANN.
- **[RaBitQ](https://dl.acm.org/doi/10.1145/3654970)**: Quantization with theoretical error bounds for ANN.
- **[FARGO](https://dl.acm.org/doi/10.14778/3594512.3594522)**: Efficient maximum inner product search via multi-probing.
- **[ip-NSW](https://proceedings.neurips.cc/paper_files/paper/2018/hash/ae032b79e2b43ef5f9ae1c5d08d5660c-Abstract.html)**: Graph-based MIPS via proximity graphs.
- **[MAG](https://arxiv.org/abs/2504.14861)**: Supports both inner product and Euclidean retrieval in one index.
- **[ANN-Benchmarks](https://doi.org/10.1016/j.is.2019.02.006)**: Classic synthetic recall/latency benchmarking framework.
- **[NeurIPS’21 Billion-Scale ANN Challenge](https://arxiv.org/abs/2204.08957)**: Competition results highlighting engineering trade-offs.
- **[Graph-Based Vector Search: An Experimental Evaluation](https://doi.org/10.1145/3709693)**: Empirical evaluation of graph ANNS.
- **[Survey of Vector DBMS](https://doi.org/10.1007/s00778-024-00864-x)**: Systems view of vector databases.
- **[LAET](https://dl.acm.org/doi/10.1145/3318464.3389771)**: Early learned adaptive termination for ANN.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Task-centric evaluation | Measure downstream utility, not just recall | Iceberg | Label Recall@K, Hit@K, Matching Score@K | Task metrics vary by domain |
| Graph ANNS | Navigate a proximity graph | HNSW, DiskANN, NSG | ANN-Benchmarks, Iceberg | Parameter tuning; distribution sensitivity |
| Synthetic-recall early termination | Stop when target recall is reached | DARTH, Ada-ef, LAET | Declarative synthetic recall | Not task-aware |
| Stability-based early exit | Stop when results stabilize | pEE | Dense retrieval | IDs stabilizing may not imply labels stabilizing |

### Closest Prior Work

1. **Iceberg** suggests task-aware early stop but does not implement one.
2. **DARTH / Ada-ef / LAET** early-terminate for **synthetic recall**, not downstream labels.
3. **Patience in Proximity** early-terminates **HNSW** traversal using top-k **ID overlap** saturation; it is the closest ID-stability baseline for HNSW.
4. **pEE** uses stability signals over **IDs** for IVF probing in dense retrieval.

**Novelty Kill Search Summary:** Searched for “label-aware early termination nearest neighbor search”, “task-aware early stop approximate nearest neighbor”, “label recall saturation early exit HNSW”, and “anytime/progressive kNN classification early stopping”. Found synthetic-recall termination (DARTH/Ada-ef/LAET) and ID-stability termination (pEE), but no close match using **label-histogram stability** as the online stopping criterion for ANNS as of 2026-02-16.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Iceberg | Defines task-centric VSS metrics | No early-stop algorithm | Implement task-aware early stop | Directly targets the benchmark’s task metric |
| DARTH / Ada-ef | Early-terminate for synthetic recall target | Not task-aware; may over-search | Use label-histogram stability | Stops when label utility saturates |
| Patience in Proximity / pEE | Early exit from ID stability (HNSW / IVF) | Not task-aware | Apply stability to labels in HNSW | Labels better proxy for label-driven tasks |

---

## Experiments

### Experimental Setup

**Benchmark + code**: Iceberg harness ([repo](./references/GitHub-ZJU-DAILY-Iceberg/meta/meta_info.txt)).

**Dataset**: Iceberg ImageNet-EVA02 (1,281,167 base vectors; 50,000 queries; 1024-dim) with class labels, released via Hugging Face `PIIR/Iceberg-dataset`.

**Metric**: **Label Recall@100** (fraction of top-100 retrieved items whose label matches the query label; higher is better). We also report synthetic Recall@100 as a diagnostic.

**Index / baseline method**: HNSW (Euclidean distance). Iceberg’s task-centric leaderboard reports **HNSW as the winner** for ImageNet-EVA02 under Euclidean distance (Table 7 in Iceberg), so this is a strong baseline.

**Main conditions (≤3)**:
1. **Fixed-ef baseline**: HNSW with `efSearch=1500` (Iceberg example config scale) and a sweep `efSearch∈{100,…,1500}` to form the Label Recall@100–latency curve.
2. **Ours**: Label-histogram stability early exit with `efSearch_max=1500`.
3. **ID-stability ablation**: stop based on top-K ID overlap stability (Patience in Proximity / pEE-style), using the same checkpoint schedule.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| Iceberg ImageNet-EVA02 | Labeled kNN-style retrieval for image classification | Label Recall@100; latency p50/p90/p99; early-exit rate | Iceberg queries | https://huggingface.co/datasets/PIIR/Iceberg-dataset | https://github.com/ZJU-DAILY/Iceberg |

### Main Results

#### Results Table

| Method | Benchmark | Label Recall@100 (mean±std) | Latency p50 (mean±std) | Latency p99 (mean±std) | Notes |
|--------|-----------|-----------------------------|-------------------------|-------------------------|-------|
| Fixed efSearch=1500 | ImageNet-EVA02 | **TBD** | **TBD** | **TBD** | 3 HNSW build seeds |
| **Ours: label-stability early exit** | ImageNet-EVA02 | **TBD** | **TBD** | **TBD** | `efSearch_max=1500` |
| ID-stability early exit | ImageNet-EVA02 | **TBD** | **TBD** | **TBD** | Ablation |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Ours (no early exit) | Checkpoints computed but forced to `efSearch_max` | Measures bookkeeping overhead |

### Experimental Rigor

- **Seeds**: Build 3 HNSW indices with different seeds; evaluate all methods on the same graphs per seed.
- **Latency**: Report p50/p90/p99 and early-exit rate; add p99.9 if p99 improves but failures appear.
- **Hard-query stratification (required control)**: From full-search top-K labels, compute per-query margin = (top1 label frequency − top2 label frequency)/K; report Label Recall@100 and latency for low-margin vs high-margin queries.
- **Confounders**: (i) bookkeeping overhead (measured by “no early exit” ablation), (ii) run-to-run timing noise (repeat runs; discard warmup), (iii) index randomness (controlled by seeds).
- **Data leakage**: no training; evaluation uses Iceberg-provided fixed query set and labels.

**Resource Estimate**:
- CPU-only; run on a single multi-core server as in Iceberg (dual Xeon; Section 4.1). Use a 10k-query subset for iteration; scale to all 50k queries for final numbers.

---

## Success Criteria

**Hypothesis** (directional — what you expect):
Label-histogram stabilization will allow earlier termination than fixed `efSearch` with minimal loss in Label Recall@100.

**Decision Rule** (concrete — when to stop):
- **Continue/Proceed**: At the operating point defined by fixed `efSearch=1500`, our method reduces **p50 latency by ≥20%** while keeping **Label Recall@100 within 0.5 percentage points** of the fixed-ef baseline (mean across 3 index seeds), and does not worsen **p99 latency by >5%**.
- **Pivot**: If aggregate metrics pass but low-margin queries suffer most of the Label Recall drop, add a conservative safeguard (e.g., minimum visited nodes or combine with ID-stability) and re-test.
- **Refute**: If there is no Pareto improvement, or Label Recall@100 drops by >0.5 points, abandon label-histogram stabilization for this setting.

---

## Impact Statement

If successful, this provides a simple, training-free, task-aware early-termination rule that practitioners can add to HNSW-based retrieval in label-driven applications (image classification, face recognition) to reduce latency and serving cost without changing embedding models.

---

## References

- [Reveal Hidden Pitfalls and Navigate Next Generation of Vector Similarity Search from Task-Centric Views](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt)
- [GitHub - ZJU-DAILY/Iceberg](./references/GitHub-ZJU-DAILY-Iceberg/meta/meta_info.txt)
- [DARTH: Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search](./references/DARTH-Declarative-Recall-Through-Early-Termination-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt)
- [Distribution-Aware Exploration for Adaptive HNSW Search](./references/Distribution-Aware-Exploration-for-Adaptive-HNSW-Search/meta/meta_info.txt)
- [Early Exit Strategies for Approximate k-NN Search in Dense Retrieval](./references/Early-Exit-Strategies-for-Approximate-kNN-Search-in-Dense-Retrieval/meta/meta_info.txt)
- [Patience in Proximity: A Simple Early Termination Strategy for HNSW Graph Traversal in Approximate k-NN Search](./references/Patience-in-Proximity-Early-Termination-HNSW/meta/meta_info.txt)
- [HNSW](https://arxiv.org/abs/1603.09320)
- [DiskANN](https://arxiv.org/abs/1907.05641)
- [NSG](https://arxiv.org/abs/1707.00143)
- [FAISS](https://arxiv.org/abs/2401.08281)
- [ScaNN](https://arxiv.org/abs/1908.10396)
- [RaBitQ](https://dl.acm.org/doi/10.1145/3654970)
- [FARGO](https://dl.acm.org/doi/10.14778/3594512.3594522)
- [ip-NSW](https://proceedings.neurips.cc/paper_files/paper/2018/hash/ae032b79e2b43ef5f9ae1c5d08d5660c-Abstract.html)
- [MAG](https://arxiv.org/abs/2504.14861)
- [ANN-Benchmarks](https://doi.org/10.1016/j.is.2019.02.006)
- [NeurIPS’21 Billion-Scale ANN Challenge](https://arxiv.org/abs/2204.08957)
- [Graph-Based Vector Search: An Experimental Evaluation](https://doi.org/10.1145/3709693)
- [Survey of Vector DBMS](https://doi.org/10.1007/s00778-024-00864-x)
- [LAET](https://dl.acm.org/doi/10.1145/3318464.3389771)
