# untitled

# From HNSW Graph Topology to Embedding Geometry: Auditing Index-File Leakage in Vector Databases

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, USENIX Security (or similar top AI / security venues)

## Introduction

### Context and Motivation

Approximate nearest neighbor search (ANNS) is the retrieval backbone for many production AI systems, including semantic search and retrieval-augmented generation (RAG). A common ANNS choice is **Hierarchical Navigable Small World (HNSW)** graphs, implemented in libraries like FAISS and deployed in many vector databases. HNSW accelerates retrieval by storing a sparse proximity graph over stored vectors and performing greedy graph traversal during search.

In many deployments, engineers treat the **stored vectors** as the primary sensitive artifact, while implicitly assuming that the **index structure** (the HNSW graph) is mostly harmless metadata. However, HNSW edges encode which items are “near” in embedding space, and therefore may reveal sensitive relationships even if the vectors themselves are protected (e.g., encrypted at rest, isolated by permissions, or inaccessible due to compartmentalization).

Recent systems-security work underscores that “non-value” artifacts around ML inference can leak more than expected. For example, **Found in Translation (FIT)** shows that an OS-level adversary observing page-level memory accesses in confidential computing environments can infer the sequence of FAISS-HNSW nodes accessed by semantic search queries with high accuracy, but does not study what can be inferred downstream from such leaked node identities or from the index topology itself (**[Found in Translation](./references/Found-in-Translation-A-Generative-Language-Modeling-Approach-to-Memory-Access-Pattern-Attacks/meta/meta_info.txt)**).

To make the audit actionable for real RAG deployments (typically **768–1536 dimensions**, with very different intrinsic geometry than SIFT), we evaluate topology-only leakage on **two regimes**: (i) a standard low-dimensional ANN benchmark (SIFT10K, 128-d) and (ii) a **text-embedding** regime (10k MS MARCO passages embedded with a public sentence-transformers model, 768-d).

### The Problem

This proposal asks a focused audit-style question:

> If an attacker obtains **only the HNSW graph topology** (e.g., FAISS `hnsw.neighbors` / `hnsw.offsets` arrays or an exported adjacency list), but not the vectors, can they recover a **useful approximation of the embedding geometry** that reveals additional nearest-neighbor relationships beyond the edges explicitly present in the leaked graph?

There is a plausible reason this might be possible: classical results in **ordinal embedding** show that knowing each point’s k-nearest neighbors can be sufficient to reconstruct a point configuration (up to similarity transform) under certain conditions (**[Local Ordinal Embedding](./references/Local-Ordinal-Embedding/meta/meta_info.txt)**), and follow-up work scales such reconstruction to tens of thousands of nodes via local-to-global synchronization (**[Point Localization and Density Estimation from Ordinal kNN graphs using Synchronization](./references/Point-Localization-and-Density-Estimation-from-Ordinal-kNN-graphs-using-Synchronization/meta/meta_info.txt)**).

At the same time, HNSW is **not** a clean kNN graph: it is approximate, sparse, and designed to be navigable via small-world shortcuts. It is therefore unclear whether graph-only reconstruction will work, and if it does, whether it yields new information beyond the adjacency list itself.

### Key Insight and Hypothesis

**Key insight:** Even if HNSW edges are an imperfect sample of true nearest-neighbor relations, the *global navigability constraints* that make HNSW effective for search may implicitly preserve enough information about the underlying metric space that graph-geodesic distances (appropriately de-biased) correlate with the true embedding distances.

**Hypothesis:** Given only the layer-0 HNSW adjacency, a simple **degree-penalized geodesic embedding** (landmark shortest paths with degree-weighted edges + landmark multidimensional scaling) will recover a low-dimensional coordinate system whose induced kNN neighborhoods have **higher Recall@k** against the true-vector kNN graph than the raw HNSW neighbor lists.

**Why we could be wrong:** Small-world shortcut edges (especially via high-degree hub nodes) may collapse shortest-path distances so aggressively that graph-geodesic distances have low rank correlation with Euclidean distances. Additionally, for high intrinsic-dimensional embeddings, the HNSW topology may be too noisy for any topology-only reconstruction to improve kNN recovery.

---

## Proposed Approach

### Overview

We propose a fully automated **topology-leakage audit** for FAISS HNSW indices:

1. Build a FAISS `IndexHNSWFlat` on a public vector dataset.
2. Extract only the **layer-0 adjacency list** (attacker view) and discard vectors.
3. Reconstruct a low-dimensional embedding using only the graph topology.
4. Measure how well the reconstructed embedding recovers the **true kNN neighborhoods** (computed from the original vectors, but only for evaluation).

### Method Details

#### Threat model and attacker input

- The attacker obtains an HNSW index artifact containing:
  - `hnsw.neighbors` and `hnsw.offsets` arrays (FAISS) or an equivalent adjacency list at layer 0.
- The attacker does **not** obtain stored vectors.
- The attacker may know global parameters (M, efConstruction) and the node count n. (Here **M** is the target number of neighbor links stored per node in the base layer; **efConstruction** controls how aggressively the graph is explored during index build, affecting recall/graph quality.)

We focus on layer 0 because it contains all nodes and is typically the densest part of the graph; it is also the most likely to be partially exposed by APIs or side channels.

#### Reconstruction algorithm: degree-penalized landmark Isomap (topology → coordinates)

Let G be the undirected version of the layer-0 adjacency graph (symmetrized). Let deg(v) be node degree.

1. **Edge weights (hub penalty):** assign each edge (u,v) a positive weight
   - `w(u,v) = 1 + α * (log(1+deg(u)) + log(1+deg(v))) / 2`.
   - Intuition: paths that traverse hubs are penalized to reduce small-world shortcut distortion.

2. **Landmark selection:** choose L landmark nodes uniformly at random (e.g., L=256), with a fixed RNG seed.

3. **Landmark geodesics:** compute shortest-path distances from each landmark to all nodes using Dijkstra on the weighted graph (L runs).

4. **Landmark MDS:**
   - Compute the L×L landmark distance matrix.
   - Run classical multidimensional scaling (MDS) to embed landmarks into d dimensions (e.g., d=32).

5. **Out-of-sample extension:** embed non-landmark nodes using standard landmark MDS extension formulas (e.g., triangulation from distances to landmarks; see Landmark-Isomap / LMDS).

6. **Recovered kNN:** compute each node’s k nearest neighbors in the recovered coordinates using exact distance in the recovered space.

### Key Innovations

- **Security framing + evaluation protocol**: Treat HNSW adjacency as a leaked artifact and measure how much additional neighborhood information can be inferred compared to the adjacency list itself.
- **Degree-penalized geodesics**: A minimal, topology-only adjustment designed for HNSW’s small-world shortcut failure mode.
- **Decision-relevant metric**: Evaluate leakage via **kNN neighborhood recovery** (Recall@k), directly connecting to what vector DB operators typically consider sensitive (similarity relationships).

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) vector similarity search infrastructure, (ii) side channels / leakage in ML systems, and (iii) graph-based reconstruction of geometry.

On the systems side, HNSW and related graph indices are widely used for fast ANNS (e.g., FAISS, DiskANN, ScaNN). On the security side, recent work has shown that access patterns and auxiliary artifacts can leak sensitive inference inputs even when the main payload is protected (e.g., controlled-channel and access-pattern attacks in trusted execution environments, and FIT on HNSW). In parallel, the ordinal embedding and manifold learning literature shows that neighborhood graphs can be sufficient to reconstruct geometry under assumptions, but this has not been systematically evaluated as an **attack surface** for HNSW index artifacts.

### Related Papers

- **[HNSW](https://arxiv.org/abs/1603.09320)**: Introduces HNSW as a hierarchical proximity graph for approximate nearest neighbor search.
- **[FAISS](https://arxiv.org/abs/1702.08734)**: A widely used similarity search library including HNSW implementations.
- **[DiskANN](https://arxiv.org/abs/1907.05641)**: Graph-based ANNS on SSD; illustrates how index structure choices affect retrieval behavior.
- **[ScaNN](https://arxiv.org/abs/1908.10396)**: Large-scale approximate nearest neighbor search via anisotropic quantization and partitioning.
- **[Found in Translation](./references/Found-in-Translation-A-Generative-Language-Modeling-Approach-to-Memory-Access-Pattern-Attacks/meta/meta_info.txt)**: Learns to infer object-level access sequences (including FAISS HNSW nodes) from page-level traces in confidential computing.
- **[IHOP / search-pattern leakage attacks](https://www.usenix.org/system/files/sec21-oya.pdf)**: Recovers queries from leakage in searchable encryption; relevant as an example of inference from side-channel artifacts.
- **[Data Recovery on Encrypted Databases with k-NN Query Leakage](./references/Data-Recovery-on-Encrypted-Databases-With-k-Nearest-Neighbor-Query-Leakage/meta/meta_info.txt)**: Shows kNN response identifiers can enable accurate reconstruction of underlying data in encrypted DB settings.
- **[Mask-based Membership Inference Attacks for RAG](https://arxiv.org/abs/2410.20142)**: Membership inference on RAG knowledge bases using masked-token prediction; uses FAISS HNSW as retrieval backend but does not use index topology leakage.
- **[Local Ordinal Embedding](./references/Local-Ordinal-Embedding/meta/meta_info.txt)**: Proves local kNN information can reconstruct point configurations under conditions; provides a reconstruction objective.
- **[Point Localization and Density Estimation from Ordinal kNN graphs using Synchronization](./references/Point-Localization-and-Density-Estimation-from-Ordinal-kNN-graphs-using-Synchronization/meta/meta_info.txt)**: Scales ordinal embedding from kNN graphs via local-to-global synchronization.
- **[Isomap](https://science.sciencemag.org/content/290/5500/2319)**: Classic manifold learning method using geodesic distances + MDS.
- **[Landmark Isomap / LMDS](https://proceedings.neurips.cc/paper/2003/hash/7eacb532570ff6858afd2723755ff790-Abstract.html)**: Landmark-based approximation for scaling Isomap/MDS.
- **[Laplacian Eigenmaps](https://proceedings.neurips.cc/paper/2001/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html)**: Spectral graph embedding that preserves local neighborhoods.
- **[DeepWalk](https://arxiv.org/abs/1403.6652)**: Learns node embeddings from random walks; a general-purpose graph embedding baseline.
- **[node2vec](https://arxiv.org/abs/1607.00653)**: Biased random-walk graph embeddings; relevant as an alternative topology-only embedding.
- **[DeepWalking Backwards](https://arxiv.org/abs/2102.08532)**: Shows graph embeddings can leak enough information to reconstruct graph topology (inverse direction).
- **[De-anonymizing Social Networks](https://dl.acm.org/doi/10.5555/2981562.2981565)**: Classic structural deanonymization via graph matching; shows topology can be identifying.
- **[Percolation Graph Matching](https://arxiv.org/abs/1411.7296)**: Formalizes seed-based graph matching as a percolation process.
- **[Privacy Preserving Aggregated-ANN Search for Collaborative RAG](https://arxiv.org/abs/2507.17199)**: Secure multi-party ANN that explicitly discusses index-structure leakage trade-offs.
- **[Privacy-Preserving ANN on High-Dimensional Data](https://arxiv.org/abs/2508.10373)**: Encryption-based ANN search; index structure leaks some neighborhood relationships as an efficiency trade-off.
- **[Membership Inference Attacks Against ML Models](https://arxiv.org/abs/1610.05820)**: Foundational membership inference framing; relevant for how “presence in a database” can be sensitive.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| ANN graph indices | Store a proximity graph for fast approximate retrieval | HNSW, FAISS, DiskANN | ANN-Benchmarks, SIFT, GIST | Index artifacts reveal similarity relationships |
| Side-channel leakage in ML systems | Infer sensitive info from access patterns / traces | FIT, controlled-channel attacks | Hamming distance / reconstruction accuracy | Often stops at “trace→IDs,” not “IDs→semantics/geometry” |
| kNN leakage / leakage-abuse | Infer data from kNN query outputs or leakage profiles | Kornaropoulos19, IHOP | Reconstruction error, query recovery rate | Different threat model (query outputs vs stored index) |
| Ordinal embedding / manifold learning | Recover coordinates from neighborhood/ordinal constraints | LOE, Cucuringu15, Isomap/LMDS | Procrustes error, kNN overlap | Assumptions may not hold for HNSW or high-ID data |
| Graph deanonymization | Map nodes across graphs using topology/aux info | Narayanan-Shmatikov, percolation matching | Match accuracy/coverage | Typically assumes social graphs, not metric graphs |

### Closest Prior Work

- **Found in Translation (Jia et al., 2025)**: Demonstrates that FAISS-HNSW node-access sequences can be inferred from page-level traces in enclaves. Our proposal instead assumes the adjacency itself is leaked (by file/API/side channel) and measures whether topology alone enables **geometry / neighborhood reconstruction**.
- **Local Ordinal Embedding (Terada & von Luxburg, 2014)**: Provides theory and optimization for reconstructing coordinates from kNN neighborhood constraints. Our setting differs because HNSW is an approximate, navigable small-world graph rather than a clean kNN graph.
- **Cucuringu & Woodworth (2015)**: Scales ordinal embedding via synchronization. We borrow the “graph→geometry” perspective but apply it as a security audit on HNSW index artifacts.
- **Kornaropoulos et al. (2019)**: Reconstructs data from kNN query leakage in encrypted DBs. We study a different leakage channel (stored HNSW topology) and focus on recovering neighborhood relations rather than plaintext coordinates.
- **Privacy-preserving ANN schemes (e.g., 2507.17199, 2508.10373)**: Often accept some neighborhood-structure leakage for efficiency; our work quantifies the practical consequences of that leakage for HNSW.

**Novelty Kill Search Summary:** Searched for combinations including “HNSW topology leakage attack”, “HNSW graph inversion reconstruct embeddings”, “HNSW adjacency list privacy”, “ordinal embedding from HNSW graph”, and “vector database index file leakage HNSW”. Also checked for direct matches to “HNSW graph-only reconstruction” and “HNSW topology deanonymization”. We did not find prior work that explicitly evaluates *topology-only* reconstruction of embedding geometry from leaked HNSW adjacency lists as of 2026-02-25. (Full query log is in `notes.md`.)

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| FIT (2025) | Recovers object-level accesses (incl. HNSW nodes) from page traces | Stops at recovering accessed IDs; no geometry/semantic inference | Treat adjacency as leaked artifact; quantify geometry reconstruction | If topology preserves metric info, we can infer more than visited IDs |
| LOE (2014) | Reconstructs coordinates from kNN ordinal constraints | Not evaluated on HNSW-like small-world graphs | Apply to HNSW artifacts; add hub-penalized geodesics | HNSW design may preserve enough structure for reconstruction |
| Cucuringu15 | Scales ordinal embedding from kNN graphs | Not security-framed; assumes kNN graph structure | Security audit framing; evaluate leakage on HNSW | Provides scalable tools for attacker-style reconstruction |
| Kornaropoulos19 | Reconstructs data from kNN query leakage | Different leakage channel (queries), not stored index topology | Study stored topology-only leakage | Index files may leak even without queries |
| 2507.17199 / 2508.10373 | Encrypts vectors but may leak neighborhood structure | Leakage consequences not quantified for HNSW | Provide measurable leakage metric (Recall@k) | Helps practitioners reason about what “acceptable leakage” implies |

---

## Experiments

### Experimental Setup

**Goal:** quantify whether topology-only reconstruction recovers more true kNN relations than the leaked adjacency itself.

**Datasets:**
- **SIFT10K (low-d vision features):** base vectors (n=10,000, d=128), a standard FAISS evaluation dataset (also used in FIT’s HNSW experiments).
- **MSMARCO-10K (text embeddings):** randomly sample 10,000 passages from the MS MARCO passage corpus and embed them with `sentence-transformers/msmarco-distilbert-base-v2` (d=768). (We only need embeddings + IDs; no labels.)

**Index construction (per dataset):**
- Build FAISS `IndexHNSWFlat(d, M=32)` where d is the dataset dimensionality.
- Set `efConstruction=64` (or FAISS default if unspecified), and disable compression.
- Build 3 indices with different FAISS RNG seeds (or different insertion orders) to estimate variance.

**Attacker-visible artifact extraction:**
- Extract only the layer-0 neighbor lists from FAISS (`hnsw.neighbors`, `hnsw.offsets`, plus per-node degree).
- Symmetrize to an undirected graph for reconstruction.

**Methods (kept to ≤3 main conditions):**
1. **Adjacency-only baseline**: treat each node’s leaked adjacency list as its “neighbor set”; compute Recall@k vs the true-vector kNN.
2. **Unweighted geodesic baseline (ablation)**: landmark shortest-path distances with unit edge weights + landmark MDS.
3. **Ours (degree-penalized geodesic)**: same as (2) but with degree-penalized edge weights.

**Primary metric:**
- **Recall@k** (k=10): for each node i, let N_true(i) be its exact top-k neighbors under Euclidean distance in the original vector space, and N_rec(i) be top-k neighbors in the reconstructed space. Recall@k is the fraction of true neighbors recovered: `|N_true(i) ∩ N_rec(i)| / k`, averaged over nodes.

**Secondary metrics (sanity checks):**
- Spearman correlation between (graph geodesic distance) and (true Euclidean distance) on random node pairs.
- Sensitivity to k (k ∈ {5,10,20}) reported as a small robustness check if budget allows.

**Resource Estimate**:
- **Compute budget**: ≤20 GPU-hours (mostly for exact kNN ground truth via FAISS IndexFlatL2 on GPU); reconstruction should be CPU-feasible. (If MSMARCO-10K embedding is done on GPU, expect a few additional GPU-hours, still within this budget.)
- **GPU memory**: ≤16GB (SIFT10K vectors are small).
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SIFT10K | 10k 128-d SIFT vectors (standard ANN benchmark subset) | Recall@10 (primary), Spearman(geodesic, L2) | all | https://github.com/facebookresearch/faiss/wiki/Faiss-indexes (or TexMex corpus mirror) | Custom: FAISS build + graph extraction + reconstruction + recall computation |
| MSMARCO-10K | 10k MS MARCO passages embedded with `sentence-transformers/msmarco-distilbert-base-v2` (768-d) | Recall@10 (primary) | all | https://huggingface.co/datasets/sentence-transformers/msmarco | Custom: sample passages → embed → FAISS build + extraction + reconstruction + recall computation |

### Main Results

(All results are to be produced by verification.)

| Method | Base Model | Benchmark | Metric (mean±std) | Source | Notes |
|---|---|---|---|---|---|
| Adjacency-only (leaked edges) | N/A (no model) | SIFT10K | Recall@10: **TBD**; Spearman(geodesic,L2): N/A | - | Baseline: uses leaked neighbor lists only (no reconstruction) |
| Unweighted geodesic + LMDS | N/A (no model) | SIFT10K | Recall@10: **TBD**; Spearman(geodesic,L2): **TBD** | - | Ablation: tests whether hop distances alone suffice |
| **Ours: degree-penalized geodesic + LMDS** | N/A (no model) | SIFT10K | Recall@10: **TBD**; Spearman(geodesic,L2): **TBD** | - | Topology-only reconstruction (attacker) |
| Adjacency-only (leaked edges) | N/A (no model) | MSMARCO-10K | Recall@10: **TBD** | - | Baseline for 768-d text embeddings |
| Unweighted geodesic + LMDS | N/A (no model) | MSMARCO-10K | Recall@10: **TBD** | - | Ablation on text-embedding topology |
| **Ours: degree-penalized geodesic + LMDS** | N/A (no model) | MSMARCO-10K | Recall@10: **TBD** | - | Topology-only reconstruction on text embeddings |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours w/ α=0 | Remove degree penalty (reduces to unweighted) | Lower Recall@10 if hub shortcuts are the main distortion |
| Ours w/ different landmark counts | L ∈ {128, 256, 512} | Diminishing returns; modest improvement with larger L |

### Experimental Rigor

- **Seeds**: build 3 HNSW indices with different seeds/insertion orders: `seeds=[42,123,456]`. Report mean±std.
- **Fair comparison**: all reconstruction methods see only topology; the true vectors are used only to compute evaluation ground truth.
- **Sanity checks**:
  - If we replace the HNSW graph with an Erdos–Renyi graph with the same n and similar average degree, Recall@10 should be near chance.
  - If we use a much denser HNSW (larger M), both adjacency-only and reconstruction recall should increase.

---

## Success Criteria

**Hypothesis**: Degree-penalized geodesic embedding from leaked HNSW topology recovers more true nearest-neighbor relations than the leaked neighbor lists alone.

**Decision Rule**:
- **Proceed/Continue**: The proposed method improves Recall@10 over adjacency-only by **≥0.10 absolute** (10 points) on **at least one** dataset, *and* by **≥0.05** on the other dataset, consistently across 3 index seeds. (I.e., the effect is not SIFT-only.)
- **Pivot**: If unweighted geodesic improves over adjacency-only but degree-penalization does not (on either dataset), simplify the method (drop degree penalty) and keep the core claim as “geodesic embedding leaks geometry.”
- **Refute**: If both geodesic variants improve Recall@10 over adjacency-only by <0.02 (2 points) across seeds on **both** datasets, conclude that leaked HNSW topology does not enable meaningful additional kNN recovery in these regimes.

---

## Impact Statement

If this audit shows that HNSW topology leakage enables substantial recovery of true kNN neighborhoods, vector database operators and confidential-computing practitioners should treat **index artifacts** (adjacency lists, graph files, and memory traces revealing edges) as sensitive data, and update threat models and access controls accordingly. If the attack fails, it provides evidence that (at least for this benchmark/regime) leaking HNSW topology may be less harmful than leaking vectors or query results.

---

## References

- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin, 2018
- [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734) - Johnson et al., 2017
- [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://arxiv.org/abs/1907.05641) - Subramanya et al., 2019
- [Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)](https://arxiv.org/abs/1908.10396) - Guo et al., 2020
- [Found in Translation: A Generative Language Modeling Approach to Memory Access Pattern Attacks](./references/Found-in-Translation-A-Generative-Language-Modeling-Approach-to-Memory-Access-Pattern-Attacks/meta/meta_info.txt) - Jia et al., 2025
- [Local Ordinal Embedding](./references/Local-Ordinal-Embedding/meta/meta_info.txt) - Terada & von Luxburg, 2014
- [Point Localization and Density Estimation from Ordinal kNN graphs using Synchronization](./references/Point-Localization-and-Density-Estimation-from-Ordinal-kNN-graphs-using-Synchronization/meta/meta_info.txt) - Cucuringu & Woodworth, 2015
- [Data Recovery on Encrypted Databases With k-Nearest Neighbor Query Leakage](./references/Data-Recovery-on-Encrypted-Databases-With-k-Nearest-Neighbor-Query-Leakage/meta/meta_info.txt) - Kornaropoulos et al., 2019
- [Mask-based Membership Inference Attacks for Retrieval-Augmented Generation](https://arxiv.org/abs/2410.20142) - Liu et al., 2024
- [A Generative Model for Manifold Learning (Isomap)](https://science.sciencemag.org/content/290/5500/2319) - Tenenbaum et al., 2000
- [Landmark Isomap](https://proceedings.neurips.cc/paper/2003/hash/7eacb532570ff6858afd2723755ff790-Abstract.html) - de Silva & Tenenbaum, 2003
- [Laplacian Eigenmaps for Dimensionality Reduction and Data Representation](https://proceedings.neurips.cc/paper/2001/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html) - Belkin & Niyogi, 2003
- [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) - Perozzi et al., 2014
- [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) - Grover & Leskovec, 2016
- [DeepWalking Backwards: From Embeddings Back to Graphs](https://arxiv.org/abs/2102.08532) - Konsotiropoulos et al., 2021
- [De-anonymizing Social Networks](https://dl.acm.org/doi/10.5555/2981562.2981565) - Narayanan & Shmatikov, 2009
- [De-anonymizing scale-free social networks by percolation graph matching](https://arxiv.org/abs/1411.7296) - Yartseva & Grossglauser, 2013
- [Threshold-Protected Searchable Sharing: Privacy Preserving Aggregated-ANN Search for Collaborative RAG](https://arxiv.org/abs/2507.17199) - 2025
- [Privacy-Preserving Approximate Nearest Neighbor Search on High-Dimensional Data](https://arxiv.org/abs/2508.10373) - 2025
- [Membership Inference Attacks against Machine Learning Models](https://arxiv.org/abs/1610.05820) - Shokri et al., 2017
