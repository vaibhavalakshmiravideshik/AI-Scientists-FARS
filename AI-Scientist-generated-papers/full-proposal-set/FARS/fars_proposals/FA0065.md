# untitled

# When Can Mean-Direction Deflation Repair Metric Misuse? Deployment-Time Reranking for Frozen Inner-Product Vector Search

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Vector similarity search (VSS) is a core primitive in modern ML systems: dense retrieval for search and recommendation, k-nearest neighbor (kNN) classification over embeddings, and retrieval-augmented generation (RAG) pipelines all rely on retrieving nearest neighbors of a query embedding from a large database. Most production systems implement VSS as a two-stage pipeline: (i) **approximate nearest neighbor search (ANNS)** returns a small candidate set quickly using an ANN index (typically specialized to a fixed metric), and (ii) a reranker (or a downstream task model) consumes the shortlist.

A key decision in any vector database is the **similarity metric** (e.g., inner product, cosine similarity, Euclidean/L2 distance). This choice is often baked into the deployed index: for example, **HNSW (Hierarchical Navigable Small World)** graph indexes and product quantization (PQ) codebooks are typically built for a fixed metric, and GPU kernels are tuned for a specific scoring function. Switching metrics can require re-embedding, re-indexing, and re-tuning, which is expensive for million–hundred-million scale corpora.

The **Iceberg** benchmark argues that metric selection cannot be treated as a minor engineering detail. It introduces task-centric evaluation metrics and shows a striking “metric misuse” failure mode: a system can achieve nearly perfect **synthetic recall** (accuracy with respect to the chosen metric) while being nearly useless for the downstream task.

### The Problem

Iceberg identifies a three-layer “information loss funnel” in end-to-end VSS pipelines, and highlights **metric misuse** as a particularly severe and deployment-relevant failure mode ([Iceberg](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt), Sec. 4.2).

Throughout, we refer to inner-product similarity as **IP**, and Euclidean/L2 distance as **ED**.

Iceberg contrasts:

- **Synthetic Recall@K**: fraction of the metric-defined true top-K neighbors recovered by the ANN method (higher is better; measures algorithmic accuracy under a *chosen* metric), and
- **Task-centric metrics** (e.g., label-based correctness of retrieved neighbors; higher is better; measures downstream utility).

For ImageNet retrieval-as-classification, Iceberg evaluates downstream utility with **Label Recall@K**, the fraction of the top-K retrieved items whose class label matches the query label (Sec. 3.1). Under metric misuse, downstream utility can collapse even when synthetic recall is ~100%:

- On **ImageNet-EVA02**, **maximum inner product search (MIPS; i.e., IP retrieval)** at **Synthetic Recall@100 ≈ 99.99%** yields **Label Recall@100 < 1%**; in contrast, ED ANNS on EVA02 reaches **Label Recall@100 = 84.9%** at **99% synthetic recall** (HNSW, Iceberg Table 5).
- On **BookCorpus** (text self-retrieval), Iceberg reports **MIPS Hit@100 < 50%** while ED ANNS reaches **Hit@100 = 100%** (Iceberg Sec. 4.2, Fig. 6 description). Here **Hit@K** is the fraction of queries for which the correct passage ID appears anywhere in the top-K list (Sec. 3.1).

Iceberg also reports that ImageNet-EVA02 is extremely anisotropic: its embeddings have **radial alignment (RA) = 3°** ([Iceberg](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt), Sec. 4.3). RA is a summary statistic of anisotropy: smaller RA indicates embeddings are tightly aligned with a dominant mean direction (i.e., many vectors lie in a narrow cone).

In many deployments, this kind of failure is difficult to remediate: the system may be locked to an **inner-product index** for historical or infrastructure reasons (GPU kernels, quantization format, or a metric-specific ANN graph), and rebuilding the index under a different metric is not always feasible.

### Key Insight and Hypothesis

**Key insight:** In high-dimensional embedding spaces, **hubness** (a small number of “hub” vectors appearing in many queries’ nearest-neighbor lists) is strongly linked to spatial centrality and anisotropy: points close to the data mean are more likely to become hubs ([Hubs in Space](https://www.jmlr.org/papers/v11/radovanovic10a.html); [Centering Similarity Measures to Reduce Hubs](https://aclanthology.org/D13-1058.pdf)). Classic hubness-reduction methods show that suppressing a dominant mean direction can reduce hubs, but most evaluations assume you can post-process all database vectors and rebuild the index.

This proposal does **not** claim the mean-deflation mechanism itself is novel. The research question is deployment- and benchmark-driven:

> **Can a rank-one mean-direction deflation applied only at rerank time (small shortlist M) materially recover Iceberg’s task-centric utility under extreme metric misuse, and can we detect when reranking is impossible because the candidate set lacks “right-metric” neighbors?**

**Hypothesis:** On Iceberg’s extreme metric-misuse regimes (ImageNet-EVA02 and BookCorpus), a mean-direction deflation reranker applied to the **top-M = 200** candidates from a frozen inner-product index will substantially improve task-centric metrics (Label Recall@100 / Hit@100), recovering a meaningful fraction of the performance gap to a Euclidean-distance ceiling.

A key failure mode is **candidate coverage**: if the “right-metric” (Euclidean) top-100 neighbors are rarely present in the legacy IP top-200 list, reranking cannot help. We therefore treat coverage as an explicit diagnostic and refutation criterion.

---

## Proposed Approach

### Overview

We propose **Mean-Direction Deflation Reranking (MDDR)**, a training-free deployment patch for inner-product vector search.

- **Offline (one-time):** compute the database mean direction \(\mu\).
- **Online (per query):** retrieve top-\(M\) candidates using the existing inner-product index, then rerank those candidates using a deflated score that penalizes similarity contributed by the mean direction.

### Method Details

#### Notation

- Database embeddings: \(x \in \mathbb{R}^d\), database \(\mathcal{X} = \{x_1,\ldots,x_N\}\)
- Query embedding: \(q \in \mathbb{R}^d\)
- Legacy similarity (inner product): \(s_{\text{IP}}(q,x)=q^\top x\)

#### Step 0: compute the global mean direction (offline)

Compute the database mean:

\[
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
\]

Store \(\mu\) and \(\|\mu\|^2\). This is a single \(d\)-dimensional vector.

#### Step 1: retrieve a candidate set with the frozen IP index

Use the production inner-product index to retrieve a shortlist:

\[
\mathcal{C}_M(q) = \text{TopM}_{x \in \mathcal{X}}\; s_{\text{IP}}(q,x)
\]

We set the headline candidate budget to **M = 200**.

#### Step 2: mean-direction deflation reranking on \(\mathcal{C}_M(q)\)

We rerank \(x \in \mathcal{C}_M(q)\) using a deflated query vector \(q'\) and a corrected score:

\[
\alpha(q) = \frac{q^\top \mu}{\|\mu\|^2 + \epsilon}
\]

\[
q' = q - \beta\, \alpha(q)\, \mu
\]

\[
 s_{\text{defl}}(q,x) = (q')^\top x = q^\top x - \beta\, \alpha(q)\, \mu^\top x
\]

- \(\beta\) is a single scalar controlling deflation strength (default \(\beta=1\)); it can be tuned on a validation split.
- This is equivalent to subtracting from each candidate a penalty proportional to its projection \(\mu^\top x\), scaled by the query’s alignment to \(\mu\).

**Relation to prior work:** This score correction is closely related to mean-centering / anisotropy reduction methods (e.g., [Suzuki et al. 2013](https://aclanthology.org/D13-1058.pdf), [ABTT](./references/All-but-the-Top-Simple-and-Effective-Postprocessing-for-Word-Representations/meta/meta_info.txt)) and to test-time distribution normalization (DN) ideas in contrastive vision-language models ([Zhou et al., 2023](https://arxiv.org/abs/2302.11084)). Our contribution is to test whether this cheapest-possible correction is sufficient in Iceberg’s metric-misuse regime under a strict shortlist constraint (M=200), and to quantify when it cannot work due to coverage.

**Deployment note (important):** since \(s_{\text{defl}}(q,x)=\langle q',x\rangle\), MDDR can be implemented either as:

1) **Rerank-time patch:** compute \(s_{\text{defl}}\) on returned candidate IDs and reorder only top-M, or
2) **Query-side transform:** send \(q'\) directly to the existing inner-product index (no database rewrite).

The proposal’s core claim is about the rerank-time patch at **M=200**, since this matches common two-stage retrieval deployments.

#### Candidate-coverage diagnostic (used for refutation)

Define Euclidean top-K neighbors \(\mathcal{N}^{\text{ED}}_K(q)\) and measure how many of them are even present in the IP shortlist:

\[
\text{Coverage}(q;K,M)=\frac{|\mathcal{N}^{\text{ED}}_K(q) \cap \mathcal{C}_M(q)|}{K}
\]

If coverage is low, no reranker (including MDDR) can recover Euclidean-aligned neighbors at small M.

### Key Innovations

1. **Iceberg-targeted empirical study under a deployment constraint:** evaluate whether mean-direction deflation can repair Iceberg’s *task-centric* metric misuse when the candidate set is limited (M=200) and the index metric is frozen.
2. **Coverage-based decision rule:** treat ED@100-in-IP@200 coverage as an explicit diagnostic for when reranking-only remediation is impossible.
3. **Implementation simplicity:** rank-one deflation is a single-vector correction that can be implemented as a query transform, avoiding query-/gallery-bank infrastructure required by many hubness normalizers.

---

## Related Work

### Field Overview

**Task-centric evaluation for vector search.** Traditional ANN benchmarks focus on recall–latency trade-offs under a fixed metric, but downstream tasks often require label- or intent-aligned retrieval. Iceberg formalizes this with task-centric metrics (Label Recall@K / Hit@K / Matching Score@K) and shows that synthetic recall can be misleading under metric misuse.

**Hubness and anisotropy in high-dimensional embeddings.** Hubness is a well-studied effect of the curse of dimensionality where a few points become nearest neighbors for many queries. Multiple lines of work connect hubness to spatial centrality and anisotropy and propose post-hoc transformations (centering, local scaling, density gradient flattening). These methods are typically evaluated in kNN classification, cross-lingual retrieval, or cross-modal retrieval.

**Training-free score normalization in retrieval.** In cross-modal and contrastive retrieval, recent methods improve retrieval without finetuning by using a bank of reference queries and/or gallery items to normalize scores (QB-Norm, DBNorm, NNN, distribution normalization, DN). These approaches are closest in spirit to MDDR, but are motivated by hubness in cross-modal embeddings rather than metric misuse in vector databases.

### Related Papers

- **[Iceberg: Reveal Hidden Pitfalls and Navigate Next Generation of Vector Similarity Search from Task-Centric Views](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt)**: Introduces task-centric VSS evaluation and exposes extreme metric misuse (e.g., ImageNet-EVA02 under IP has Label Recall@100 < 1% at Synthetic Recall@100 ≈ 99.99%).
- **[Cross Modal Retrieval with Querybank Normalisation (QB-Norm)](./references/Cross-Modal-Retrieval-with-Querybank-Normalisation/meta/meta_info.txt)**: Unifies GC/CSLS/IS hubness corrections via a query bank; proposes Dynamic Inverted Softmax for robustness.
- **[Mitigating Hubness in Cross-Modal Retrieval with Query and Gallery Banks (DBNorm)](https://arxiv.org/abs/2310.11612)**: Uses both query and gallery banks to normalize hubness in cross-modal retrieval.
- **[Nearest Neighbor Normalization (NNN)](./references/Nearest-Neighbor-Normalization-Improves-Multimodal-Retrieval/meta/meta_info.txt)**: Uses an additive per-candidate bias estimated from the k nearest reference queries; discusses Distribution Normalization (mean subtraction) as a constant-time baseline.
- **[Test-Time Distribution Normalization (DN) for Contrastively Learned Vision-Language Models](https://arxiv.org/abs/2302.11084)**: Applies mean-based test-time normalization to improve CLIP-like models under distribution shift.
- **[All-but-the-Top (ABTT)](./references/All-but-the-Top-Simple-and-Effective-Postprocessing-for-Word-Representations/meta/meta_info.txt)**: Removes the global mean and top principal components to reduce anisotropy in word embeddings (offline post-processing).
- **[Whitening Sentence Representations for Better Semantics and Faster Retrieval](https://arxiv.org/abs/2103.15316)**: Applies whitening to reduce anisotropy in sentence embeddings as a post-processing step.
- **[scikit-hubness](./references/scikit-hubness-Hubness-Reduction-and-Approximate-Neighbor-Search/meta/meta_info.txt)**: Provides implementations of hubness measures and reduction methods (Mutual Proximity, Local Scaling, DisSimLocal) with approximate variants.
- **[Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data](https://www.jmlr.org/papers/v11/radovanovic10a.html)**: Foundational hubness analysis; shows hubs correlate with proximity to the data mean.
- **[Improving Zero-Shot Learning by Mitigating the Hubness Problem](https://arxiv.org/abs/1412.6568)**: Early demonstration of hubness-driven failures in ZSL and post-hoc correction methods.
- **[Centering Similarity Measures to Reduce Hubs](https://aclanthology.org/D13-1058.pdf)**: Shows mean-centering of similarity reduces hubs and improves kNN-style NLP tasks.
- **[Local and Global Scaling Reduce Hubs in Space](https://www.jmlr.org/papers/v13/schnitzer12a.html)**: Proposes local/global scaling transformations for hubness reduction.
- **[Flattening the Density Gradient for Eliminating Spatial Centrality to Reduce Hubness](https://aaai.org/papers/10240-flattening-the-density-gradient-for-eliminating-spatial-centrality-to-reduce-hubness/)**: Introduces density-gradient flattening (DisSimLocal) as a hubness-reduction approach.
- **[A Comprehensive Empirical Comparison of Hubness Reduction in High-Dimensional Spaces](https://pmc.ncbi.nlm.nih.gov/articles/PMC7327987/)**: Large empirical comparison of hubness reduction methods across many datasets.
- **[Fast Approximate Hubness Reduction for Large High-Dimensional Data](https://ofai.at/papers/oefai-tr-2018-02.pdf)**: Develops approximate hubness reduction compatible with ANN search.
- **[Offline Bilingual Word Vectors, Orthogonal Transformations and the Inverted Softmax](https://arxiv.org/abs/1702.03859)**: Introduces inverted softmax for hubness mitigation in bilingual lexicon induction.
- **[Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087)**: Popularizes CSLS and self-learning for cross-lingual embedding alignment.
- **[Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion](https://arxiv.org/abs/1804.07745)**: Proposes RCSLS, aligning training and retrieval via a CSLS-like objective.
- **[Hubs and Hyperspheres: Reducing Hubness and Improving Transductive Few-shot Learning](https://arxiv.org/abs/2303.09352)**: Connects hubness elimination to hyperspherical uniformity; proposes structure-preserving uniformization.
- **[Down with the Hierarchy: The ‘H’ in HNSW Stands for “Hubs”](https://arxiv.org/abs/2412.01940)**: Argues that hub nodes provide “highways” that make hierarchy unnecessary in high-dimensional NSW graphs.
- **[Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)](https://arxiv.org/abs/1908.10396)**: A widely used MIPS system using score-aware anisotropic quantization.
- **[ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](https://arxiv.org/abs/1807.05614)**: Standard ANN benchmarking suite for recall–latency evaluation.
- **[Maximum Inner Product is Query-Scaled Nearest Neighbor](https://arxiv.org/abs/2503.06882)**: Shows MIPS can be solved on Euclidean graphs via query scaling and proposes PSP.
- **[Stitching Inner Product and Euclidean Metrics for Topology-aware MIPS (MAG)](https://arxiv.org/abs/2504.14861)**: Proposes cross-metric navigation for efficient MIPS (addresses speed/robustness, not metric misuse).
- **[Adversarial Hubness in Multi-Modal Retrieval](https://arxiv.org/abs/2412.14113)**: Shows that hubness can be exploited adversarially; evaluates query-bank defenses.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Task-centric VSS evaluation | Evaluate VSS via downstream-task labels rather than only synthetic recall | [Iceberg](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt) | Iceberg (Label Recall / Hit / Matching Score) | Primarily diagnostic; does not propose remediation methods |
| Mean/PC removal (anisotropy reduction) | Reduce hubs by removing global mean and/or dominant directions | [Centering Similarity](https://aclanthology.org/D13-1058.pdf), [ABTT](./references/All-but-the-Top-Simple-and-Effective-Postprocessing-for-Word-Representations/meta/meta_info.txt), [Whitening](https://arxiv.org/abs/2103.15316) | NLP similarity, kNN classification, STS | Typically evaluated as offline embedding post-processing |
| Distance rescaling / density correction | Recompute a secondary distance to repair asymmetric neighborhoods | [Local Scaling](https://www.jmlr.org/papers/v13/schnitzer12a.html), [DisSimLocal](https://aaai.org/papers/10240-flattening-the-density-gradient-for-eliminating-spatial-centrality-to-reduce-hubness/), [Empirical comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC7327987/) | kNN classification and hubness measures | Often quadratic in database size; approximate variants exist but are more complex |
| Bank-based score normalization | Use a reference query/gallery bank to downweight hubs | [QB-Norm](./references/Cross-Modal-Retrieval-with-Querybank-Normalisation/meta/meta_info.txt), [DBNorm](https://arxiv.org/abs/2310.11612), [NNN](./references/Nearest-Neighbor-Normalization-Improves-Multimodal-Retrieval/meta/meta_info.txt) | Cross-modal retrieval (MSCOCO, Flickr30k, MSR-VTT, …) | Requires reference banks; some methods scale linearly with bank size |
| Faster ANN/MIPS indices | Improve search speed under a chosen metric | [ScaNN](https://arxiv.org/abs/1908.10396), [MAG](https://arxiv.org/abs/2504.14861), [PSP](https://arxiv.org/abs/2503.06882) | ANN-Benchmarks, Big-ANN benchmarks | Assumes the metric is correct; does not fix metric misuse |

### Closest Prior Work

- **Mean-centering to reduce hubs** ([Centering Similarity Measures to Reduce Hubs](https://aclanthology.org/D13-1058.pdf)): Shows that centering similarity by subtracting the data mean reduces hubness and improves nearest-neighbor classification in NLP tasks. It does not study two-stage ANN pipelines or task-centric VSS metrics under metric misuse.
- **ABTT** ([All-but-the-Top](./references/All-but-the-Top-Simple-and-Effective-Postprocessing-for-Word-Representations/meta/meta_info.txt)): Removes the mean and top principal components to make word embeddings more isotropic. ABTT is an offline embedding post-processing method; applying it to a deployed vector DB is often assumed to require rewriting all vectors and rebuilding the index.
- **Test-Time Distribution Normalization (DN)** ([Zhou et al., 2023](https://arxiv.org/abs/2302.11084)): Uses mean-based embedding normalization at test time for CLIP-like models (primarily for robustness / zero-shot transfer). It is algorithmically similar to mean subtraction, but it is not evaluated for Iceberg-style metric misuse, nor under a fixed shortlist constraint with an explicit coverage diagnostic.
- **QB-Norm / DBNorm / NNN** ([QB-Norm](./references/Cross-Modal-Retrieval-with-Querybank-Normalisation/meta/meta_info.txt), [DBNorm](https://arxiv.org/abs/2310.11612), [NNN](./references/Nearest-Neighbor-Normalization-Improves-Multimodal-Retrieval/meta/meta_info.txt)): Bank-based normalization methods that reduce hubs by using statistics over a reference query/gallery set; they are more general but require additional infrastructure (banks, probe matrices, offline bias caching).

### Comparison Table

| Related work | What it does | Key limitation for our target setting | What we change | Why ours should win |
|---|---|---|---|---|
| [ABTT](./references/All-but-the-Top-Simple-and-Effective-Postprocessing-for-Word-Representations/meta/meta_info.txt) | Removes mean + top PCs from all embeddings | Typically framed as requiring rewriting all DB vectors → reindexing | Apply a rank-one correction at query/rerank time only | Keeps the same deployed index; only changes query/shortlist scoring |
| [Centering Similarity](https://aclanthology.org/D13-1058.pdf) | Mean-centers similarities to reduce hubs | Not evaluated in two-stage ANN with shortlist constraints | Evaluate mean-direction deflation specifically at M=200 | Same hubness mechanism, but deployment- and benchmark-relevant |
| [Test-Time DN](https://arxiv.org/abs/2302.11084) | Mean-based normalization at inference time | Not tested on Iceberg metric misuse; no coverage-based decision rule | Compare query-dependent deflation vs DN-style query-independent mean subtraction | DN may already solve it; if not, we learn the limits and when mean-deflation fails |
| [QB-Norm](./references/Cross-Modal-Retrieval-with-Querybank-Normalisation/meta/meta_info.txt) | Query-bank normalization (GC/CSLS/IS/DIS) | Needs query bank + large probe matrix; heavier memory/compute | Use only a single mean vector \(\mu\) | Lower overhead; may be sufficient in extreme anisotropy regimes |
| [NNN](./references/Nearest-Neighbor-Normalization-Improves-Multimodal-Retrieval/meta/meta_info.txt) | Per-candidate bias from k nearest reference queries | Requires ANN over reference queries and offline bias caching | Replace kNN-over-reference with rank-one mean-direction signal | Much simpler; targets mean-direction dominance rather than generic bias |
| [Iceberg](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt) | Diagnoses metric misuse in VSS | Does not propose remediation | Add a deployment-time remediation layer + coverage diagnostic | Turns diagnosis into an actionable patch or a concrete negative result |

---

## Experiments

### Experimental Setup

**Goal:** Test whether a rank-one mean-direction deflation can recover task-centric utility in Iceberg’s metric-misuse regime **without rebuilding an IP index**, using only **M=200** reranking.

We target two Iceberg datasets with reported metric misuse:

- **ImageNet-EVA02** (vision; task metric = Label Recall@100)
- **BookCorpus** (text self-retrieval; task metric = Hit@100)

**Base Models:**

No model training is required. We use Iceberg’s released **precomputed embeddings**.

| Embedding set | Download Link | Notes |
|---|---|---|
| Iceberg ImageNet-EVA02 embeddings | https://huggingface.co/datasets/PIIR/Iceberg-dataset | 1024-d; queries = ImageNet val (50k) |
| Iceberg BookCorpus embeddings | https://huggingface.co/datasets/PIIR/Iceberg-dataset | 1024-d; queries = 10k paragraphs (Iceberg Sec. 3.1) |

**Training Data (if applicable):**

No training data is needed (inference-only retrieval and reranking).

**Other Resources (if applicable):**

- Iceberg codebase (for dataset loading + metric definitions): https://github.com/ZJU-DAILY/Iceberg

**Resource Estimate**:

- **Compute budget**: dominated by (near-)exact kNN search on precomputed embeddings.
  - ImageNet-EVA02: ~1.28M DB vectors, 50k queries.
  - BookCorpus: ~9.25M DB vectors, 10k queries.
  - For each dataset we run: IP retrieval (top-M=200), ED retrieval (top-100) as ceiling, and reranking (negligible).
  - Expected total budget: **≤ 300 GPU-hours** on 1–8× A100 GPUs using FAISS GPU indices (well under the 768 GPU-hour cap).
- **GPU memory**:
  - ImageNet-EVA02: 1.28M×1024 FP16 vectors ≈ 2.6GB.
  - BookCorpus: 9.25M×1024 FP16 vectors ≈ 18.9GB.
  - Both fit on a single A100 80GB (plus index overhead).
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Iceberg ImageNet-EVA02 | ImageNet-1K embeddings (EVA02) for retrieval-as-classification | Label Recall@100 (higher is better) | queries = ImageNet val (50k) | https://huggingface.co/datasets/PIIR/Iceberg-dataset | Iceberg definition (Sec. 3.1) + simple evaluator |
| Iceberg BookCorpus | Paragraph self-retrieval over BookCorpus embeddings | Hit@100 (higher is better) | queries = 10k paragraphs | https://huggingface.co/datasets/PIIR/Iceberg-dataset | Iceberg definition (Sec. 3.1) + simple evaluator |

**Metric definitions (from Iceberg):**

- **Label Recall@K**: \(\text{LabelRecall}@K = \frac{1}{|Q|K} \sum_{q\in Q} \sum_{i=1}^{K} \mathbb{I}(L(r_i)=L(q))\), where \(r_i\) are the top-K retrieved vectors for query \(q\) and \(L\) returns the class label.
- **Hit@K**: \(\text{Hit}@K = \frac{1}{|Q|} \sum_{q\in Q} \mathbb{I}(L(q) \in \{L(r): r \in R(q)\})\), where \(L(q)\) is the query paragraph ID and \(R(q)\) is the top-K retrieved set.

### Main Results

**Run order (for efficiency):**

1) Compute **ED@100** and **IP@200** once per dataset and report **Coverage(ED@100 in IP@200)**. If coverage < 0.20, stop early and treat the “M=200 rerank-time remediation” hypothesis as refuted for that dataset.
2) If coverage is sufficient, run reranking methods on the fixed IP@200 candidate lists.

**Core conditions (used in the primary decision rule):**

1) **Legacy IP baseline**: retrieve top-100 under inner product.
2) **Euclidean ceiling (exact kNN)**: retrieve exact top-100 under Euclidean distance (FAISS IndexFlatL2; exact kNN).
3) **IP@200 + MDDR (ours)**: rerank within IP@200 using \(s_{\text{defl}}\), then report the top-100.

**Secondary baselines (for interpretation; rerank on the same IP@200 shortlist):**

- **DN-style mean subtraction** ([Zhou et al., 2023](https://arxiv.org/abs/2302.11084)): query-independent mean removal \(q' = q - \beta\mu\).
- **QB-Norm (DIS variant)** ([Bogolin et al., 2021](./references/Cross-Modal-Retrieval-with-Querybank-Normalisation/meta/meta_info.txt)): Dynamic Inverted Softmax using a query bank drawn from the training split.
- **NNN** ([Chowdhury et al., 2024](./references/Nearest-Neighbor-Normalization-Improves-Multimodal-Retrieval/meta/meta_info.txt)): per-candidate bias subtraction \(s(q,x)-b(x)\) where \(b(x)\) is estimated from a reference query bank.

#### Results Table (to be filled by verifier)

| Method | Candidate budget | Benchmark | Task metric | Coverage(ED@100 in IP@200) | Source | Notes |
|---|---:|---|---:|---:|---|---|
| IP baseline | M=100 | ImageNet-EVA02 | **TBD** (LabelRecall@100) | **TBD** | - | Legacy inner product retrieval |
| ED ceiling (exact kNN) | M=100 | ImageNet-EVA02 | **TBD** (LabelRecall@100) | - | - | Metric-correct retrieval upper bound |
| IP@200 + DN mean subtraction | M=200 | ImageNet-EVA02 | **TBD** (LabelRecall@100) | (same as above) | - | Secondary baseline; \(q'=q-\beta\mu\) |
| IP@200 + QB-Norm (DIS) | M=200 | ImageNet-EVA02 | **TBD** (LabelRecall@100) | (same as above) | - | Secondary baseline; query bank size e.g. 4096; \(\beta\) tuned |
| IP@200 + NNN | M=200 | ImageNet-EVA02 | **TBD** (LabelRecall@100) | (same as above) | - | Secondary baseline; bank size e.g. 4096; k e.g. 32; \(\alpha\) tuned |
| **IP@200 + MDDR (ours)** | M=200 | ImageNet-EVA02 | **TBD** (LabelRecall@100) | (same as above) | - | Query-dependent deflation \(q'=q-\beta\alpha(q)\mu\) |
| IP baseline | M=100 | BookCorpus | **TBD** (Hit@100) | **TBD** | - | Legacy inner product retrieval |
| ED ceiling (exact kNN) | M=100 | BookCorpus | **TBD** (Hit@100) | - | - | Metric-correct retrieval upper bound |
| IP@200 + DN mean subtraction | M=200 | BookCorpus | **TBD** (Hit@100) | (same as above) | - | Secondary baseline |
| IP@200 + QB-Norm (DIS) | M=200 | BookCorpus | **TBD** (Hit@100) | (same as above) | - | Secondary baseline |
| IP@200 + NNN | M=200 | BookCorpus | **TBD** (Hit@100) | (same as above) | - | Secondary baseline |
| **IP@200 + MDDR (ours)** | M=200 | BookCorpus | **TBD** (Hit@100) | (same as above) | - | Query-dependent deflation |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Random-direction penalty | Replace \(\mu\) with a random unit vector | Should not meaningfully improve task metrics; controls for “reranking noise” |
| **DN-style mean subtraction (baseline)** | Use \(q' = q - \beta\mu\) (no \(\alpha(q)\) scaling), analogous to mean-based test-time normalization | If query alignment matters, query-dependent deflation should win |
| Candidate budget sweep | Run M ∈ {50, 200, 1000} | If improvements require very large M, the deployment story weakens |
| Multi-PC deflation (optional) | Replace rank-1 direction \(\mu\) with top-r PCA directions (r ∈ {1,4}) | Tests whether broader anisotropy (beyond the mean direction) matters |

### Analysis (Optional)

- **Hubness metrics before/after rerank**: compute skewness of k-occurrence distribution (k=10) under IP vs deflated score on the query set.
- **Anisotropy correlation**: correlate per-query improvement with \(|\alpha(q)|\) (query alignment to \(\mu\)).

---

## Success Criteria

**Criterion 1 (primary): Gap recovery on ImageNet-EVA02 at M=200**

- Hypothesis: On ImageNet-EVA02, MDDR improves Label Recall@100 relative to the IP baseline by reducing mean-direction-driven hubness.
- Validation (decision rule): Let \(A\) be IP baseline LabelRecall@100, \(B\) be ED ceiling LabelRecall@100, and \(C\) be MDDR LabelRecall@100 at M=200. We declare success iff:
  - \(\frac{C-A}{B-A} \ge 0.25\) (recovers at least 25% of the ED–IP gap), **and**
  - \(C\) exceeds the random-direction penalty control by **≥ 5 absolute points**.

**Criterion 2 (secondary): Improvement on BookCorpus at M=200**

- Hypothesis: If mean-direction hubness contributes to BookCorpus’s metric misuse, MDDR should also improve Hit@100.
- Validation (decision rule): Let \(A\) be IP baseline Hit@100, \(B\) be ED ceiling Hit@100, and \(C\) be MDDR Hit@100 at M=200. We treat BookCorpus as a generalization check and declare it a “pass” iff:
  - \(\frac{C-A}{B-A} \ge 0.25\), **and**
  - \(C\) exceeds the random-direction penalty control by **≥ 3 absolute points**.

**Criterion 3 (refute rule): Coverage barrier at small M**

- Hypothesis: If ED-relevant neighbors are absent from IP@200, reranking cannot help.
- Validation (refute rule): For each dataset, if mean Coverage(ED@100 in IP@200) across queries is **< 0.20**, we treat the “M=200 rerank-time remediation” hypothesis as refuted for that dataset (regardless of whether MDDR slightly improves over IP).

---

## Impact Statement

If successful, MDDR provides a minimal deployment-time remediation for metric misuse in vector databases: practitioners who are locked into an inner-product index could apply a single-vector correction (mean-direction deflation) as a query transform or a rerank-time patch over a small shortlist, improving downstream task utility without re-embedding or rebuilding indexes. Even a negative result would be decision-changing: it would indicate that in severe metric-misuse regimes, the IP candidate set is missing the relevant neighbors at practical shortlist sizes, and that no reranking-only patch can substitute for reindexing or candidate-generation changes.

---

## References

- [Reveal Hidden Pitfalls and Navigate Next Generation of Vector Similarity Search from Task-Centric Views](./references/Reveal-Hidden-Pitfalls-and-Navigate-Next-Generation-of-Vector-Similarity-Search-from-Task-Centric-Views/meta/meta_info.txt) - Chen et al., 2025
- [Cross Modal Retrieval with Querybank Normalisation](./references/Cross-Modal-Retrieval-with-Querybank-Normalisation/meta/meta_info.txt) - Bogolin et al., 2021
- [Nearest Neighbor Normalization Improves Multimodal Retrieval](./references/Nearest-Neighbor-Normalization-Improves-Multimodal-Retrieval/meta/meta_info.txt) - Chowdhury et al., 2024
- [All-but-the-Top: Simple and Effective Postprocessing for Word Representations](./references/All-but-the-Top-Simple-and-Effective-Postprocessing-for-Word-Representations/meta/meta_info.txt) - Mu et al., 2017
- [scikit-hubness: Hubness Reduction and Approximate Neighbor Search](./references/scikit-hubness-Hubness-Reduction-and-Approximate-Neighbor-Search/meta/meta_info.txt) - Feldbauer et al., 2019
- [Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data](https://www.jmlr.org/papers/v11/radovanovic10a.html) - Radovanović et al., 2010
- [Improving Zero-Shot Learning by Mitigating the Hubness Problem](https://arxiv.org/abs/1412.6568) - Dinu et al., 2014
- [Centering Similarity Measures to Reduce Hubs](https://aclanthology.org/D13-1058.pdf) - Suzuki et al., 2013
- [Local and Global Scaling Reduce Hubs in Space](https://www.jmlr.org/papers/v13/schnitzer12a.html) - Schnitzer et al., 2012
- [Flattening the Density Gradient for Eliminating Spatial Centrality to Reduce Hubness](https://aaai.org/papers/10240-flattening-the-density-gradient-for-eliminating-spatial-centrality-to-reduce-hubness/) - Hara et al., 2016
- [A Comprehensive Empirical Comparison of Hubness Reduction in High-Dimensional Spaces](https://pmc.ncbi.nlm.nih.gov/articles/PMC7327987/) - Feldbauer & Flexer, 2019
- [Fast Approximate Hubness Reduction for Large High-Dimensional Data](https://ofai.at/papers/oefai-tr-2018-02.pdf) - Feldbauer et al., 2018
- [Offline Bilingual Word Vectors, Orthogonal Transformations and the Inverted Softmax](https://arxiv.org/abs/1702.03859) - Smith et al., 2017
- [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087) - Conneau et al., 2018
- [Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion](https://arxiv.org/abs/1804.07745) - Joulin et al., 2018
- [Test-Time Distribution Normalization for Contrastively Learned Vision-Language Models](https://arxiv.org/abs/2302.11084) - Zhou et al., 2023
- [Mitigating Hubness in Cross-Modal Retrieval with Query and Gallery Banks](https://arxiv.org/abs/2310.11612) - Wang et al., 2023
- [Whitening Sentence Representations for Better Semantics and Faster Retrieval](https://arxiv.org/abs/2103.15316) - Su et al., 2021
- [On the Sentence Embeddings from Pre-trained Language Models](https://arxiv.org/abs/2011.05864) - Li et al., 2020
- [Hubs and Hyperspheres: Reducing Hubness and Improving Transductive Few-shot Learning](https://arxiv.org/abs/2303.09352) - Trosten et al., 2023
- [Down with the Hierarchy: The ‘H’ in HNSW Stands for “Hubs”](https://arxiv.org/abs/2412.01940) - Munyampirwa et al., 2024
- [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396) - Guo et al., 2020
- [ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](https://arxiv.org/abs/1807.05614) - Aumüller et al., 2018
- [Maximum Inner Product is Query-Scaled Nearest Neighbor](https://arxiv.org/abs/2503.06882) - Chen et al., 2025
- [Stitching Inner Product and Euclidean Metrics for Topology-aware Maximum Inner Product Search](https://arxiv.org/abs/2504.14861) - Chen et al., 2025
- [Adversarial Hubness in Multi-Modal Retrieval](https://arxiv.org/abs/2412.14113) - Zhang et al., 2024
