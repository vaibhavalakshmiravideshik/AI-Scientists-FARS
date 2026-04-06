# untitled

# Sketch-Gated Trace Clustering to Accelerate Inter-Trace Redundancy Pruning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Test-time scaling methods for large language models (LLMs) improve reasoning accuracy by spending more inference compute, often by sampling many chain-of-thought (CoT) trajectories and aggregating them (e.g., majority vote in self-consistency). This approach is effective on verifiable domains such as mathematics and multiple-choice science QA, but its cost scales with the number of sampled trajectories.

In this proposal we evaluate on **AIME** (American Invitational Mathematics Examination problems; short olympiad-style math questions with verifiable numeric answers) and **GPQA** (Graduate-level Google-Proof Q&A; multiple-choice science questions designed to resist retrieval shortcuts), where accuracy can be measured automatically by exact-match to a ground-truth answer.

A recent line of work has shown that a large fraction of these sampled trajectories are redundant: many parallel reasoning traces converge to the same final answer. **DeepPrune** trains a learned “answer-equivalence judge” and uses it inside an online greedy clustering algorithm to stop generating or considering redundant traces early, reducing total token generation substantially while keeping accuracy high.

However, DeepPrune’s online clustering requires many judge-model comparisons. Even with a cap on the number of clusters and sampling only a few representatives per cluster, the number of judge prompts can still be large when (i) the cluster count grows (e.g., due to judge errors early in generation), (ii) the sampling budget per cluster is non-trivial, or (iii) practitioners want to increase the diversity budget (more clusters) without paying quadratic judge cost.

### The Problem

DeepPrune’s clustering step assigns each new trace prefix to an existing cluster by comparing it against *every* current cluster using multiple judge calls per cluster:

- For each incoming trace prefix \(t_i\), DeepPrune computes \(\text{sim}(t_i, c_j)\) by averaging \(p\) judge decisions against \(p\) sampled representatives from cluster \(c_j\) (default \(p\le K_1=10\)).
- It then checks all clusters \(c_j\in C\) (up to \(|C|\le K=32\)) until it finds a cluster where a majority of sampled comparisons vote “identical”, or it creates a new cluster.

This yields up to \(O(|C|\cdot K_1)\) judge prompts per trace prefix. When \(|C|\) approaches its cap (and especially when a practitioner wants \(K\) larger than 32 to preserve more diversity), the judge-model cost can become a major component of end-to-end latency/cost.

Existing alternatives reduce compute using other signals, but do not directly address this judge-comparison bottleneck:
- **STEP** prunes traces based on hidden-state quality signals, targeting system-level KV-cache waiting time rather than answer-equivalence clustering.
- **Chopping Trees (SSDP)** merges semantically similar branches in tree search using embedding similarity, but does not use an answer-equivalence judge and does not focus on parallel self-consistency-style sampling.

### Key Insight and Hypothesis

We hypothesize that a **cheap locality-sensitive sketch** computed on early trace text can serve as a *high-recall candidate generator* for DeepPrune’s cluster comparisons. Here, a “sketch” means a compact fingerprint that preserves similarity: similar trace prefixes should map to the same bucket (or to nearby fingerprints) with high probability.

Concretely, traces that converge to the same final answer often share early lexical/structural cues (problem restatement, key equations, named entities, intermediate lemmas). A sketch such as **SimHash** (a locality-sensitive hash that maps a string to a short bit-vector using random projections, so similar texts have small **Hamming distance**, i.e., they differ in few bit positions) or **MinHash** can map similar prefixes to the same buckets with high probability. If we use sketches only to propose a small candidate set of clusters, and then keep DeepPrune’s learned judge as the final decision maker, we may reduce the number of judge prompts substantially while preserving DeepPrune’s accuracy.

This hypothesis could be wrong if (i) same-answer traces diverge too early lexically (low sketch recall), (ii) the judge clusters rely on semantic features not captured by text sketches, or (iii) the candidate filter frequently misses the correct cluster, leading to misclustering and downstream accuracy loss. Our experiments include (a) a pilot “kill test” for sketch predictiveness, and (b) a random-gating control to rule out the confound that “any reduction in comparisons works”.

---

## Proposed Approach

### Overview

We propose **Sketch-Gated DeepPrune**, a two-stage clustering procedure:

1. **Cheap candidate generation**: compute a locality-sensitive sketch for each trace prefix and retrieve a small set of candidate clusters likely to be equivalent.
2. **Exact decision**: run DeepPrune’s learned judge model only on those candidate clusters (same voting rule as DeepPrune).

When the sketch returns no candidates, we fall back to DeepPrune’s full cluster scan to avoid catastrophic recall failures.

### Method Details

**Inputs and base components**
- A set of sampled reasoning traces \(S=\{t_1,\dots,t_N\}\) for a problem.
- A truncation function that maps each trace to a short prefix used for equivalence testing (DeepPrune’s default: “first-25 reasoning words”).
- A learned judge \(J_\theta\) that predicts whether two prefixes will yield identical final answers (DeepPrune-Judge-4B).

**DeepPrune clustering (baseline)**
- Maintains clusters \(C=\{c_1,\dots,c_m\}\).
- For a new prefix \(t_i\), compares it against each cluster \(c_j\) by sampling up to \(K_1\) representatives from \(c_j\) and applying \(J_\theta\) to each pair; assigns \(t_i\) to the first/most-similar cluster with majority vote above threshold \(\tau\).

**Sketch-Gated modification**

We maintain an LSH index over clusters’ sketch signatures.

- **Sketch function**: 64-bit SimHash over normalized text features extracted from the trace prefix (default features: whitespace-token 5-grams; lowercased; digits normalized; LaTeX control sequences removed).
- **Banding**: split the 64-bit signature into \(B\) bands (e.g., 4 bands of 16 bits) and use exact band collisions to retrieve candidate clusters.
- **Cluster representatives**: for each cluster, store sketches of up to \(R\) representative prefixes (default \(R=3\): earliest-added items), inserted into the LSH index.

For an incoming prefix \(t_i\):
1. Compute its sketch signature \(h(t_i)\) and retrieve candidate clusters whose representative sketches collide with \(h(t_i)\) in at least one band.
2. If the candidate set is non-empty, evaluate \(J_\theta\) only against those clusters (same per-cluster sampling and majority rule as DeepPrune).
3. If the candidate set is empty, run the baseline DeepPrune full scan over all clusters (**fallback**).

**Random-gating control (ablation)**
To test whether sketch-based candidate generation exploits real structure (rather than merely reducing comparisons), we include a control where we match the *candidate set size* produced by the sketch, but choose that many clusters uniformly at random.

### Key Innovations

1. **LSH as a candidate generator for judge-based answer-equivalence clustering**: the sketch is not used to decide equivalence; it only reduces the set of clusters the judge must consider.
2. **Decision-focused negative control**: a random-gating ablation tests whether gains come from meaningful candidate generation rather than from reducing comparisons arbitrarily.
3. **Operationally measurable objective**: the primary outcome is a reduction in judge prompts and clustering wall-clock time, which is directly actionable for practitioners deploying DeepPrune-like methods.

---

## Related Work

### Field Overview

Research on test-time scaling can be grouped into (i) sampling-based aggregation (self-consistency, best-of-N), (ii) structured search (tree/MCTS style), and (iii) pruning and allocation methods that reduce wasted compute. DeepPrune is part of a growing set of methods that explicitly allocate inference compute by stopping or pruning trajectories early, including confidence-based early exit methods and trace-scoring methods.

Within pruning methods, an important axis is **what signal is used to prune**: outcome confidence, process reward models, hidden-state heuristics, semantic similarity embeddings, or a learned judge that compares trajectories. DeepPrune introduced a focused learned judge for answer equivalence, while STEP and related work show that hidden states can predict trajectory quality early enough to prune without a separate judge model.

Our proposal is orthogonal to these signals: it targets the *computational structure* of judge-based clustering by reducing the number of judge comparisons needed.

### Related Papers

- **[DeepPrune: Parallel Scaling without Inter-trace Redundancy](./references/DeepPrune-Parallel-Scaling-without-Inter-trace-Redundancy/meta/meta_info.txt)**: Trains a judge for answer equivalence and prunes redundant parallel traces via greedy clustering.
- **[Hidden States as Early Signals: STEP](./references/Hidden-States-as-Early-Signals-Step-level-Trace-Evaluation-and-Pruning-for-Efficient-Test-Time-Scaling/meta/meta_info.txt)**: Uses hidden states to score and prune traces when KV cache saturates, improving latency and accuracy.
- **[Chopping Trees / SSDP](./references/Chopping-Trees-Semantic-Similarity-Based-Dynamic-Pruning-for-Tree-of-Thought-Reasoning/meta/meta_info.txt)**: Clusters tree-search branches via embedding similarity to prune semantically redundant nodes.
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)**: Majority vote over multiple sampled CoT traces to improve reasoning accuracy.
- **[Large Language Monkeys](https://arxiv.org/abs/2407.21787)**: Studies inference compute scaling via repeated sampling and aggregation.
- **[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)**: Optimizes compute allocation for test-time scaling.
- **[Deep Think with Confidence](https://arxiv.org/abs/2508.15260)**: Confidence-based early stopping for efficient reasoning.
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)**: Tree-structured exploration of intermediate “thoughts”.
- **[MCTSr](https://arxiv.org/abs/2406.07394)**: Monte Carlo tree search with self-refine for reasoning.
- **[ReST-MCTS*](https://arxiv.org/abs/2406.03816)**: PRM-guided tree search with automated process supervision.
- **[Let’s Verify Step by Step (PRM800K)](https://arxiv.org/abs/2305.20050)**: Introduces process reward models for step-level verification.
- **[Math-Shepherd](https://arxiv.org/abs/2312.08935)**: Automated process supervision for math reasoning.
- **[OmegaPRM](https://arxiv.org/abs/2406.06592)**: MCTS-based automated annotation for PRMs.
- **[FunPRM](https://arxiv.org/abs/2601.22249)**: Function-as-step process reward modeling for code.
- **[CarBoN](https://arxiv.org/abs/2510.15674)**: Calibrated best-of-N for improved selection under sampling.
- **[ReASC](https://arxiv.org/abs/2601.02970)**: Reliability-aware adaptive self-consistency.
- **[ADAPT](https://arxiv.org/abs/2506.04611)**: Diversity-aware method for test-time scaling.
- **[DORA](https://arxiv.org/abs/2506.15707)**: Allocation method aiming to make every rollout count.
- **[Policy of Thoughts](https://arxiv.org/abs/2601.20379)**: Test-time policy evolution.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Reinforcement learning for reasoning capability.
- **[SimHash](https://dl.acm.org/doi/10.1145/509907.509965)**: Locality-sensitive hashing for cosine similarity via random hyperplanes.
- **[MinHash / near-duplicate detection](https://dl.acm.org/doi/10.1145/276305.276343)**: MinHash signatures for Jaccard similarity and large-scale deduplication.
- **[HNSW](https://arxiv.org/abs/1603.09320)**: Efficient approximate nearest neighbor search in high-dimensional spaces.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Sampling + voting | Sample many traces and vote/select | Self-Consistency, Large Language Monkeys | AIME, GPQA, GSM8K | High cost; redundancy across samples |
| Confidence-based pruning | Stop early based on per-trace confidence | Deep Think with Confidence | AIME, GPQA | May prune correct minority paths |
| Judge-based redundancy pruning | Compare traces with a learned judge to prune redundant ones | DeepPrune | AIME, GPQA | Judge-model inference overhead |
| Hidden-state trace scoring | Use hidden states to score/prune traces under system constraints | STEP | AIME, GPQA, HMMT | Infra coupling; pseudo-label noise |
| Semantic similarity pruning in search | Merge/prune similar branches by embedding similarity | Chopping Trees / SSDP | GSM8K, MATH500 | Embedding similarity is imperfect proxy |

### Closest Prior Work

1. **DeepPrune** introduces the core judge-based clustering framework and provides strong evidence that inter-trace redundancy is a major bottleneck. Our proposal keeps DeepPrune’s judge decision rule intact but changes how candidate clusters are selected, targeting the judge-overhead bottleneck.

2. **Chopping Trees (SSDP)** shows that similarity-based clustering can prune redundant reasoning branches in tree search, but it uses embeddings as a proxy for redundancy and does not combine similarity search with a learned equivalence judge.

3. **STEP** prunes traces based on hidden-state quality signals and system-level memory pressure; it does not address answer-equivalence clustering or pairwise judge comparison cost.

**Novelty Kill Search Summary:** We searched for the exact combination “LSH/MinHash/SimHash + reasoning trace pruning + DeepPrune / inter-trace redundancy” using web queries such as “DeepPrune hashing LSH MinHash”, “inter-trace redundancy pruning LSH”, and “SimHash chain-of-thought”. We also searched OpenReview for “inter-trace redundancy” and scanned all agents’ drafts/finalized proposals for DeepPrune/LSH/MinHash/SimHash. We found LSH used for dataset deduplication and other NLP tasks, but did not find prior work proposing LSH-style candidate generation to accelerate **judge-based** inter-trace redundancy pruning as of 2026-02-17.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| DeepPrune | Learned judge + greedy clustering to prune redundant parallel traces | Judge comparisons scale with cluster count and per-cluster sampling | Add LSH sketch index to restrict clusters considered per trace | Fewer judge prompts with minimal change to decision rule |
| STEP | Hidden-state step scorer + memory-triggered pruning | Different objective (quality scoring), infra coupling | Target judge-overhead in equivalence clustering | Complementary; could stack with STEP later |
| Chopping Trees / SSDP | Embedding-based semantic merging in tree search | Embedding similarity is weak proxy for equivalence | Use sketches only for candidate generation; judge decides | Candidate generation needs only recall, not perfect precision |

---

## Experiments

### Experimental Setup

**Goal**: measure whether sketch-gated candidate generation reduces judge overhead while preserving DeepPrune’s accuracy and token-efficiency.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| DeepPrune-Judge-4B | 4B | https://huggingface.co/THU-KEG/DeepPrune-Judge-4B | Used for pairwise equivalence decisions |
| DeepSeek-R1-0528-Qwen3-8B | 8B | https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | Used only if we need to generate additional traces (fallback) |

**Datasets / Inputs:**
- Primary (non-gated) source of evaluation problems: DeepPrune GitHub `DeepPrune_data/online_test_data/*` (problem statements + ground truth answers). This repo copy appears truncated to 15 traces/problem, but is sufficient to define prompts and evaluation targets.
- If access is available in the automated environment, we will additionally use the full Hugging Face dataset `THU-KEG/DeepPrune` for the released 512-trace setting; otherwise we generate additional traces with DeepSeek-R1-0528-Qwen3-8B to reach N=128 traces/problem.

**Methods compared (main results)**

1. **Self-Consistency (SC)**: sample N traces, parse each final answer, majority vote.
2. **DeepPrune (baseline)**: greedy clustering with judge, using DeepPrune defaults (\(\tau=0.5\), \(K=32\), \(K_1=10\), and DeepPrune’s majority-vote finishing rule).
3. **Sketch-Gated DeepPrune (ours)**: identical to DeepPrune except cluster candidates come from SimHash-LSH buckets; empty candidate set triggers full-scan fallback.

**Ablation / negative control**
- **Random-gated DeepPrune**: same as (3) but chooses the same number of candidate clusters uniformly at random (per trace) instead of using the sketch.

**Pilot kill test (pre-condition for main run)**
- Sample 5k prefix pairs from the trace pool.
- Label “same-answer” using parsed final answers from the full traces.
- Compute AUC of SimHash similarity (1 − normalized Hamming distance) for predicting same-answer.
- If AUC < 0.65, stop and refute (do not proceed to the full clustering experiments).

**Resource Estimate**
- If using only released traces: clustering + judge calls only; expected to be well below 768 GPU-hours.
- If generating N=128 traces/problem for ~20–30 problems and 3 random seeds: expected to be within budget on ≤32×A100, dominated by trace generation and batched judge inference.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|------------------|
| AIME 2024 / 2025 | 30 olympiad-style math problems per year (verifiable numeric answers) | Accuracy; token consumption; judge prompts; wall-clock (clustering only) | test | https://github.com/THU-KEG/DeepPrune/tree/main/DeepPrune_data/online_test_data | DeepPrune repo scripts + simple answer parser |
| GPQA | Graduate-level multiple-choice science QA | Accuracy; token consumption; judge prompts; wall-clock (clustering only) | test | https://github.com/THU-KEG/DeepPrune/tree/main/DeepPrune_data/online_test_data | DeepPrune repo scripts + MC answer parser |

Key metrics:
- **Accuracy**: exact match between selected final answer and `true_answer`.
- **Judge prompts**: total number of judge-model pair comparisons executed.
- **Token consumption (simulated)**: sum of prefix tokens for all traces plus full-trace tokens for traces that are allowed to finish (DeepPrune’s finishing rule). This matches DeepPrune’s offline evaluation style.
- **Clustering wall-clock**: time spent in candidate generation + judge inference + clustering logic (excluding base-model trace generation).

### Main Results

#### Results Table

(Values marked TBD will be produced by the verification runs. Published DeepPrune numbers are included only for context; the verification runs will use matched settings.)

| Method | Base Model | Benchmark | Accuracy (mean±std) | Judge prompts (mean±std) | Token consumption (mean±std) | Source | Notes |
|--------|------------|-----------|---------------------|---------------------------|------------------------------|--------|------|
| Self-Consistency (N=128) | DeepSeek-R1-0528-Qwen3-8B | AIME24/AIME25/GPQA | TBD | 0 | TBD | - | Requires trace generation if full dataset not available |
| DeepPrune (baseline) | DeepSeek-R1-0528-Qwen3-8B + DeepPrune-Judge-4B | AIME24/AIME25/GPQA | TBD | TBD | TBD | - | Matched K=32,K1=10,\(\tau=0.5\) |
| **Sketch-Gated DeepPrune (ours)** | DeepSeek-R1-0528-Qwen3-8B + DeepPrune-Judge-4B | AIME24/AIME25/GPQA | TBD | TBD | TBD | - | Same as DeepPrune, but candidate clusters from SimHash-LSH |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Random-gated DeepPrune | Candidate clusters chosen uniformly at random with same candidate set size as ours | Worse accuracy and/or higher fallback rate than sketch-gated; shows sketch uses structure |

### Experimental Rigor

- **Seeds**: If we must generate new traces, run with `seeds=[1,2,3]` for sampling to report mean±std. If using released fixed traces only, the clustering is deterministic and we will report single-run results.
- **Fair comparison**: All clustering methods use identical N, K, K1, K2/K3 finishing rule, judge model, and truncation strategy.
- **Confounders / controls**:
  - Any reduction in comparisons might help → controlled by random-gated ablation.
  - Sketch might miss correct clusters silently → controlled by reporting `fallback_rate` and treating `fallback_rate>20%` as a failure case.
  - Parser errors for final answers → sanity check on a subset by comparing against DeepPrune repo’s own parsing utilities.

### Analysis (Optional)

- Report the trade-off curve between sketch banding parameters (B, bits per band) and (i) candidate set size, (ii) fallback rate, and (iii) judge prompt reduction.

---

## Success Criteria

**Hypothesis**: Sketch-gated candidate generation will reduce the number of judge prompts substantially (≥2×) while keeping DeepPrune’s final-answer accuracy essentially unchanged.

**Decision Rule**:
- **Proceed** if, across the evaluated benchmarks, Sketch-Gated DeepPrune reduces judge prompts by **≥2×** and reduces clustering wall-clock by **≥1.5×** compared to DeepPrune, while (i) accuracy is not lower than DeepPrune by more than **1% absolute** on GPQA and (ii) AIME24 and AIME25 each lose at most **1 question** relative to DeepPrune.
- **Pivot** if judge prompts decrease but accuracy drops >1%: try a different sketch (MinHash over character shingles) or increase the number of cluster representatives R.
- **Refute** if (a) the pilot AUC < 0.65, or (b) `fallback_rate > 20%`, or (c) sketch-gated performance is similar to random-gated.

---

## Impact Statement

If successful, this work would give practitioners deploying DeepPrune-like redundancy pruning a simple way to cut judge-model inference cost, improving latency and reducing GPU spend for test-time scaling systems that generate many reasoning traces. The method is modular: it can be added to existing judge-based clustering implementations without retraining the judge or the base reasoner.

---

## References

- [DeepPrune: Parallel Scaling without Inter-trace Redundancy](./references/DeepPrune-Parallel-Scaling-without-Inter-trace-Redundancy/meta/meta_info.txt) - Tu et al., 2025
- [Hidden States as Early Signals: STEP](./references/Hidden-States-as-Early-Signals-Step-level-Trace-Evaluation-and-Pruning-for-Efficient-Test-Time-Scaling/meta/meta_info.txt) - Liang et al., 2026
- [Chopping Trees: Semantic Similarity Based Dynamic Pruning for Tree-of-Thought Reasoning](./references/Chopping-Trees-Semantic-Similarity-Based-Dynamic-Pruning-for-Tree-of-Thought-Reasoning/meta/meta_info.txt) - Kim et al., 2025
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787) - Brown et al., 2024
- [Deep Think with Confidence](https://arxiv.org/abs/2508.15260) - Fu et al., 2025
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) - Yao et al., 2023
- [SimHash](https://dl.acm.org/doi/10.1145/509907.509965) - Charikar, 2002
- [MinHash](https://dl.acm.org/doi/10.1145/276305.276343) - Broder, 1998
- [HNSW](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin, 2016
