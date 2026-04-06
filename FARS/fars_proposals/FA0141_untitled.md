# untitled

# Label-Free Early Termination for HNSW via Bucket-Histogram Stabilization

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Approximate nearest neighbor search (ANNS) is a core primitive for **dense retrieval** (retrieving documents by embedding similarity, often used in retrieval-augmented generation), semantic search, and recommendation candidate generation. Many production systems use **HNSW (Hierarchical Navigable Small World graphs)**, a graph-based ANNS index that supports fast search by exploring a bounded candidate set.

In HNSW, the main query-time knob is typically **`efSearch`** (sometimes called `ef`), the size of the dynamic candidate list explored during the bottom-layer search. Larger `efSearch` improves retrieval quality (higher accuracy / recall) but increases latency.

### The Problem

A fixed `efSearch` is often suboptimal because query difficulty varies: some queries converge quickly to a stable neighborhood, while others require more exploration. Recent work proposes query-adaptive stopping for HNSW:

- **Synthetic-recall targeting** methods such as **DARTH** and **Ada-ef** stop search (or choose `efSearch`) once an estimated **synthetic recall@K** target is reached.
- **Saturation-based** methods such as **Patience in Proximity** stop when the returned top-k neighbor set stops changing, using **ID-overlap stability**.

However, in dense retrieval, HNSW search can exhibit **near-tie churn**: many candidates have very similar similarity scores, and the exact document IDs in the current top-k can keep swapping even after the search has effectively converged to the “right region” for downstream relevance metrics (e.g., nDCG@10). In such cases, a stopping rule based on exact ID overlap can be overly conservative, continuing search despite diminishing relevance gains.

### Key Insight and Hypothesis

We hypothesize that a **coarsened representation** of the current top-k results can converge earlier than the exact ID set while still preserving downstream retrieval utility.

Concretely, we assign each database vector a **bucket ID** using an offline unsupervised quantizer (e.g., k-means coarse centroids). During HNSW traversal, we monitor the **bucket-ID histogram** over the current top-k candidates and stop when this histogram stabilizes.

This hypothesis could be wrong in two primary ways: (i) bucket assignments may not align with semantic neighborhoods, so bucket stability does not imply relevance stability; and (ii) bucket-histogram stability may be so correlated with ID-overlap stability that it yields no practical benefit.

---

## Proposed Approach

### Overview

We propose **Bucket-Histogram Stability Early Exit (BH-Exit)** for HNSW.

- **Offline (once per index build)**: compute an unsupervised bucket ID for each database vector.
- **Online (per query)**: periodically checkpoint the current top-k list during HNSW traversal; compute its bucket-ID histogram; early-exit when histogram change stays below a threshold for a short patience window.

The method is **label-free at inference time**: it does not use relevance judgments or ground-truth task labels, only bucket IDs derived from the embedding vectors.

### Method Details

**Bucket IDs.** Let the database be vectors \(x \in \mathbb{R}^d\). Compute bucket IDs \(b(x)\in\{1,\dots,C\}\) using k-means (or an IVF coarse quantizer). We use \(C \propto \sqrt{N}\) (default \(C=4\sqrt{N}\)) as a standard coarse-quantizer scaling rule to balance bucket granularity and stability.

**Checkpointed top-k.** For a query \(q\), let \(\mathrm{TopK}_t(q)\) be the current top-k list at checkpoint \(t\) during traversal. We define checkpoints every \(B\) node-expansions at layer 0 (default \(B=100\)).

**Bucket histogram.** At checkpoint \(t\), compute the empirical bucket distribution over \(\mathrm{TopK}_t\):
\[
 p_t(c) = \frac{1}{K}\,|\{x \in \mathrm{TopK}_t(q): b(x)=c\}|\quad \text{for } c\in\{1,\dots,C\}.
\]

**Stability distance.** Use L1 change between consecutive histograms:
\[
 \Delta_t = \|p_t - p_{t-1}\|_1.
\]

**Stop rule (BH-Exit).** After a warmup of 2 checkpoints, stop if \(\Delta_t \le \epsilon\) for \(\delta\) consecutive checkpoints.

**Hyperparameters and tuning discipline.** To avoid strawman comparisons, we tune BH-Exit and the ID-overlap baseline under the same protocol on a validation split:
- Tune \((\epsilon,\delta)\) for BH-Exit and \((\gamma,\delta)\) for ID-overlap stability (Patience in Proximity), where \(\gamma\) is the overlap threshold.
- Objective: maximize latency reduction while keeping nDCG@10 within a small pre-committed budget of the fixed-budget baseline (see Success Criteria).

### Key Innovations

- **A label-free early-exit signal** for HNSW based on **coarse bucket composition** of the evolving top-k set, rather than synthetic recall prediction or exact-ID overlap stability.
- **A Phase-0 diagnostic gate** that tests whether bucket stability triggers materially earlier than ID stability; if not, the approach is refuted quickly.
- **Minimal additional storage and overhead**: one integer bucket ID per database vector; online overhead is \(O(K)\) per checkpoint.

---

## Related Work

### Field Overview

Graph-based ANNS indexes (HNSW, DiskANN/Vamana, NSG) and partition/quantization indexes (IVF-PQ, ScaNN) are widely used for dense retrieval. Most papers report synthetic recall–latency tradeoffs, but downstream utility metrics (e.g., nDCG on IR benchmarks) can be a more relevant target in retrieval systems.

Query-adaptive efficiency methods for ANNS include (i) **recall-targeting** approaches that stop once a synthetic recall estimate reaches a user-specified target, and (ii) **stability/saturation** approaches that stop once the evolving top-k set changes little. Our proposal stays in the stability family but replaces ID-level stability with **coarse bucket-level stability** as a hypothesis-driven attempt to ignore within-region near-tie churn.

### Related Papers

- **[HNSW](https://arxiv.org/abs/1603.09320)**: Foundational proximity-graph ANN algorithm; defines the search loop we modify.
- **[DiskANN / Vamana](https://arxiv.org/abs/1907.05641)**: Graph ANN design for large-scale search; motivates search-efficiency improvements.
- **[NSG](https://arxiv.org/abs/1707.00143)**: Navigating spreading-out graph; representative of graph ANN design space.
- **[FAISS (GPU similarity search)](https://arxiv.org/abs/1702.08734)**: Practical ANN system components; includes common index interfaces.
- **[FAISS (recent)](https://arxiv.org/abs/2401.08281)**: Updated FAISS overview and implementations.
- **[ScaNN](https://arxiv.org/abs/1908.10396)**: Coarse-to-fine ANN using partitioning; motivates coarse quantizers and bucketization.
- **[Product Quantization](https://ieeexplore.ieee.org/document/5432202)**: Vector quantization approach; background for coarse+residual search.
- **[IVFADC](https://ieeexplore.ieee.org/document/5206727)**: Inverted-file ANN with coarse clustering; canonical bucketization design.
- **[ANN-Benchmarks](https://doi.org/10.1016/j.is.2019.02.006)**: Standard synthetic recall–latency benchmarking tool.
- **[BEIR](https://arxiv.org/abs/2104.08663)**: Heterogeneous IR benchmark suite; defines nDCG@10 and recall metrics for many datasets.
- **[Patience in Proximity](./references/Patience-in-Proximity-A-Simple-Early-Termination-Strategy-for-HNSW-Graph-Traversal-in-Approximate-k-Nearest-Neighbor-Search/meta/meta_info.txt)**: Saturation-based early termination for HNSW via top-k **ID-overlap** stability; provides closest baseline.
- **[Early Exit Strategies for Approximate kNN Search in Dense Retrieval](./references/Early-Exit-Strategies-for-Approximate-k-k-NN-Search-in-Dense-Retrieval/meta/meta_info.txt)**: Patience-based early exit for IVF probing in dense retrieval; motivates stability signals.
- **[DARTH](./references/DARTH-Declarative-Recall-Through-Early-Termination-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt)**: Learned progressive recall prediction for declarative synthetic-recall targets.
- **[Distribution-Aware Exploration for Adaptive HNSW Search (Ada-ef)](./references/Distribution-Aware-Exploration-for-Adaptive-HNSW-Search/meta/meta_info.txt)**: Rule-based per-query ef selection targeting synthetic recall.
- **[Iceberg](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt)**: Task-centric VSS evaluation; motivates downstream-utility-aware system choices.
- **[ColBERT](https://arxiv.org/abs/2004.12832)**: Multi-vector dense retrieval; motivates efficiency work in retrieval pipelines.
- **[Contriever](https://arxiv.org/abs/2112.09118)**: Widely used dense retriever in BEIR-style evaluation.
- **[TAS-B](https://arxiv.org/abs/2104.06967)**: Dense retriever family; used in efficiency studies.
- **[Anserini](https://dl.acm.org/doi/10.1145/3274877.3274878)**: Lucene-based reproducible IR toolkit used by Patience in Proximity.
- **[Down with the Hierarchy: The ‘H’ in HNSW stands for Hubs](https://openreview.net/forum?id=0S7fRzQfym)**: Analyzes HNSW hub effects and search dynamics.
- **[A Comprehensive Survey and Experimental Comparison of Graph-based ANN](https://arxiv.org/abs/2101.12631)**: Survey of graph ANN algorithms and evaluation methodology.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Graph-based ANN | Navigate a proximity graph | HNSW, DiskANN, NSG | ANN-Benchmarks; BEIR-style retrieval | Budget tuning; distribution sensitivity |
| Recall-targeting adaptivity | Stop/choose ef to hit synthetic recall | DARTH, Ada-ef | Synthetic recall@K, latency | Not directly aligned with downstream utility |
| Saturation-based early exit | Stop when results stabilize | Patience in Proximity, pEE | nDCG@10, Recall@k, QPS/latency | ID stability may be conservative under near-ties |
| Coarse partitioning / bucketization | Coarse centroids/buckets for pruning | IVFADC, ScaNN, PQ | Recall–latency; sometimes downstream | Granularity trade-off; may not reflect semantics |

### Closest Prior Work

1. **Patience in Proximity**: stops HNSW search when the top-k **ID overlap** between consecutive iterations exceeds a threshold for a patience window. Our method uses bucket-histogram stability instead of ID overlap.
2. **pEE (Busolin et al.)**: uses ID-stability signals to early-stop IVF probing in dense retrieval. It motivates stability-based stopping but is not HNSW.
3. **DARTH / Ada-ef**: stop (or set ef) to reach a **synthetic recall** target, rather than attempting to stop when downstream utility saturates.

**Novelty Kill Search Summary:** We searched specifically for “HNSW early termination” + “cluster/bucket histogram stability” and variants (e.g., “pseudo-label histogram stability early exit ANN”, “cluster histogram convergence signal graph ANN”, “k-means bucket stability stopping criterion”). We found recall-targeting termination (DARTH/Ada-ef/LAET-like lines) and ID-overlap saturation (Patience in Proximity / IVF patience methods), but no prior work using **bucket-ID histogram stability** as the online early-exit criterion for HNSW as of 2026-02-18.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Patience in Proximity | HNSW early exit from top-k ID-overlap saturation | Sensitive to within-near-tie ID churn | Monitor bucket-ID histogram stability | Ignore within-bucket churn; earlier exit at same nDCG@10 |
| pEE (Busolin et al.) | IVF probing early exit via ID stability | Not HNSW | Apply stability stopping to HNSW with bucket signal | Transfer stability idea to graph traversal |
| DARTH / Ada-ef | Stop / set ef to hit synthetic recall | Not downstream-utility aware | Use coarse downstream-proxy stability | Better matches “utility saturates before IDs stabilize” regime |

---

## Experiments

### Experimental Setup

**Primary benchmark**: **BEIR TREC-COVID**, a biomedical document retrieval benchmark derived from the CORD-19 corpus with graded relevance judgments (50 test queries; 171,332 documents).

**Baseline Ladder:**
- **Baseline 1 (obvious)**: Fixed-budget HNSW with a sweep over `efSearch` to form the nDCG@10–latency curve.
- **Baseline 2 (closest prior work)**: ID-overlap saturation early exit (Patience in Proximity-style), tuned on validation.
- **Baseline 3 (ours)**: Bucket-histogram stability early exit (BH-Exit), tuned on the same validation.

**ID-overlap baseline definition (Patience in Proximity-style).** At checkpoint \(t\), compute overlap
\[
\phi_t = \frac{|\mathrm{TopK}_t \cap \mathrm{TopK}_{t-1}|}{K}.
\]
Stop after warmup if \(\phi_t \ge \gamma\) for \(\delta\) consecutive checkpoints.

**Phase-0 diagnostic (required go/no-go).** On a validation subset, log the first checkpoint where each stopping rule triggers:
- \(t_{\mathrm{ID}}(q)\) for ID-overlap
- \(t_{\mathrm{bucket}}(q)\) for BH-Exit
and compute the implied expected latency reduction (measured by node-expansions or distance computations).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| BGE-base-en-v1.5 (text embedding model) | ~0.1B | https://huggingface.co/BAAI/bge-base-en-v1.5 | Used to embed corpus and queries (no fine-tuning) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| None | No training; inference-only embedding + ANN | - | - | - |

**Other Resources (if applicable):**
- BEIR dataset loader + evaluation code.

**Resource Estimate**:
- **Compute budget**: dominated by one-time embedding + k-means; expected \(\ll 50\) GPU-hours (or CPU-only if embeddings are computed on CPU).
- **GPU memory**: BGE-base embedding inference fits on a single 80GB A100.
- **Indexing/search**: CPU HNSW build and search; report thread count and latency percentiles.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| BEIR TREC-COVID | Biomedical retrieval over CORD-19 corpus | nDCG@10 (ranking quality; higher is better), Recall@100/1000 (coverage; higher is better), latency p50/p95/p99 (lower is better) | test | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip | https://github.com/beir-cellar/beir |

### Main Results

#### Comparability Rules (CRITICAL)

All rows must be directly comparable:
- Same dataset and split (TREC-COVID test)
- Same embedding model and preprocessing
- Same HNSW construction parameters (M, efConstruction) and same `efSearch_max` for early-exit methods
- Same decoding/inference budget (one ANN query per input query)

#### Results Table

| Method | Base Model | Benchmark | nDCG@10 (mean±std) | Latency p50 ms (mean±std) | Latency p99 ms (mean±std) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| Fixed `efSearch` (best matched) | BGE-base-en-v1.5 | BEIR TREC-COVID | **TBD** | **TBD** | **TBD** | To be verified | Choose `efSearch` to match nDCG budget |
| ID-overlap early exit (tuned) | BGE-base-en-v1.5 | BEIR TREC-COVID | 0.7814 (1 run) | N/A | N/A | [Patience in Proximity](./references/Patience-in-Proximity-A-Simple-Early-Termination-Strategy-for-HNSW-Graph-Traversal-in-Approximate-k-Nearest-Neighbor-Search/sections/Experiments.md) | Patience in Proximity reports QPS=73.9 (vs HNSW QPS=62.4) but does not report latency percentiles |
| **Ours: BH-Exit (tuned)** | BGE-base-en-v1.5 | BEIR TREC-COVID | **TBD** | **TBD** | **TBD** | To be verified | Bucket IDs via k-means |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Random buckets | Permute bucket IDs uniformly at random | Any BH-Exit advantage should disappear; otherwise improvement is not due to semantic coarsening |
| No early exit | Compute checkpoints but force run to `efSearch_max` | Measures checkpoint bookkeeping overhead |

### Experimental Rigor

**Variance & Reproducibility:**
- Build 3 HNSW indices with different random seeds; evaluate all methods on each index. Report mean ± std.

**Validity & Controls:**
- **Confounder: strawman thresholds** → tune BH-Exit and ID-overlap under the same validation protocol and utility-loss budget.
- **Confounder: time measurement noise** → fixed thread count, warmup runs, repeated measurement passes.
- **Confounder: bucketization overhead** → measured via “No early exit” ablation.
- **Sanity check** → “Random buckets” ablation (should refute if the same gains appear).

**Data leakage / contamination:**
- No learning on TREC-COVID; we use a fixed pretrained embedding model. Potential corpus memorization affects absolute nDCG@10, but the comparison between early-exit rules is at fixed embeddings and is therefore not driven by supervised fitting on this dataset.

---

## Success Criteria

**Hypothesis** (directional — what you expect):
BH-Exit will stop earlier than tuned ID-overlap stability for a meaningful subset of queries and translate into a better latency–utility Pareto curve on nDCG@10.

**Decision Rule** (concrete — when to stop):
- **Proceed**: After tuning both early-exit methods to satisfy an nDCG@10 budget of **within 0.003 absolute** of a strong fixed-`efSearch` baseline, BH-Exit improves **latency p50 by ≥10%** without worsening **latency p99 by >5%** relative to tuned ID-overlap stability, averaged over 3 index seeds.
- **Pivot**: If BH-Exit improves p50 but hurts p99, add a conservative safeguard (minimum node-expansion floor or a hybrid rule requiring both bucket stability and weak ID stability) and re-test once.
- **Refute (fast)**: If the Phase-0 diagnostic shows (i) median \(t_{\mathrm{bucket}}\) is within 10% of \(t_{\mathrm{ID}}\) and the implied expected latency reduction is <5%, or (ii) bucket-hist never triggers before `efSearch_max` for >20% of queries.

---

## Impact Statement

If successful, BH-Exit provides a simple, label-free early-termination rule that practitioners can add to HNSW-based dense retrieval to reduce latency beyond ID-overlap saturation heuristics, with minimal storage overhead and no learned components.

---

## References

- [HNSW](https://arxiv.org/abs/1603.09320)
- [DiskANN / Vamana](https://arxiv.org/abs/1907.05641)
- [NSG](https://arxiv.org/abs/1707.00143)
- [FAISS (GPU similarity search)](https://arxiv.org/abs/1702.08734)
- [FAISS (recent)](https://arxiv.org/abs/2401.08281)
- [ScaNN](https://arxiv.org/abs/1908.10396)
- [Product Quantization](https://ieeexplore.ieee.org/document/5432202)
- [IVFADC](https://ieeexplore.ieee.org/document/5206727)
- [ANN-Benchmarks](https://doi.org/10.1016/j.is.2019.02.006)
- [BEIR](https://arxiv.org/abs/2104.08663)
- [Patience in Proximity](./references/Patience-in-Proximity-A-Simple-Early-Termination-Strategy-for-HNSW-Graph-Traversal-in-Approximate-k-Nearest-Neighbor-Search/meta/meta_info.txt)
- [Early Exit Strategies for Approximate kNN Search in Dense Retrieval](./references/Early-Exit-Strategies-for-Approximate-k-k-NN-Search-in-Dense-Retrieval/meta/meta_info.txt)
- [DARTH](./references/DARTH-Declarative-Recall-Through-Early-Termination-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt)
- [Distribution-Aware Exploration for Adaptive HNSW Search (Ada-ef)](./references/Distribution-Aware-Exploration-for-Adaptive-HNSW-Search/meta/meta_info.txt)
- [Iceberg](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt)
- [ColBERT](https://arxiv.org/abs/2004.12832)
- [Contriever](https://arxiv.org/abs/2112.09118)
- [TAS-B](https://arxiv.org/abs/2104.06967)
- [Anserini](https://dl.acm.org/doi/10.1145/3274877.3274878)
- [Down with the Hierarchy: The ‘H’ in HNSW stands for Hubs](https://openreview.net/forum?id=0S7fRzQfym)
- [A Comprehensive Survey and Experimental Comparison of Graph-based ANN](https://arxiv.org/abs/2101.12631)
