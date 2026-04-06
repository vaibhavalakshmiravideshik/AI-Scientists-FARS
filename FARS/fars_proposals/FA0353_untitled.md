# untitled

# Public-Anchor Drift Adapters for Privacy-Limited Embedding Model Upgrades

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Dense text embeddings (fixed-dimensional vectors) are a core building block for semantic search, retrieval-augmented generation (RAG; generating answers conditioned on retrieved documents), and long-term memory layers for LLM agents. In production systems, embeddings are commonly stored in a vector database (e.g., FAISS with an HNSW approximate nearest neighbor index), and the retriever is periodically improved by deploying a new embedding model.

A practical problem is that **embedding model upgrades break compatibility with existing indices**. Recomputing all corpus embeddings and rebuilding the approximate nearest neighbor (ANN) index can be expensive, especially when the corpus contains millions to billions of items. Recent work proposes **drift adapters**: learn a lightweight mapping that transforms embeddings from the new model into the legacy embedding space so the existing index can be reused with near-zero downtime (**[Drift-Adapter](./references/Drift-Adapter-A-Practical-Approach-to-Near-Zero-Downtime-Embedding-Model-Upgrades-in-Vector-Databases/meta/meta_info.txt)**).

For agent memory systems, upgrades face an additional constraint: the corpus often contains **privacy-sensitive text** (user conversations, personal preferences, proprietary documents). Even if training a drift adapter requires only a small sample of paired embeddings, it may be unacceptable to use any in-domain text (or even in-domain embeddings) to train the adapter.

### The Problem

Most drift-adapter and embedding-alignment methods assume access to **paired samples from the target deployment distribution** (the same text embedded by both the old and new models). For example, Drift-Adapter trains orthogonal Procrustes, low-rank affine, or residual-MLP adapters on a small subset of the target corpus (e.g., 20k paired items) and recovers 95–99% of the new-model retrieval structure on their studied upgrades (**[Drift-Adapter](./references/Drift-Adapter-A-Practical-Approach-to-Near-Zero-Downtime-Embedding-Model-Upgrades-in-Vector-Databases/meta/meta_info.txt)**).

However, in many privacy regimes, we can only use **public text** to construct paired embeddings. Drift-Adapter explicitly identifies this as future work (“train an adapter on a public dataset … hoping for transferability”) but does not evaluate it (Section 6, **[Drift-Adapter §6](./references/Drift-Adapter-A-Practical-Approach-to-Near-Zero-Downtime-Embedding-Model-Upgrades-in-Vector-Databases/sections/6 Discussion.md)**).

At the same time, recent “embedding translation” methods suggest cross-domain transfer may be possible, but they are substantially more complex than Drift-Adapter’s MSE-trained adapter. **vec2vec** learns to translate between embedding spaces without paired data using adversarial and cycle-consistency losses (**[vec2vec](./references/Harnessing-the-Universal-Geometry-of-Embeddings/meta/meta_info.txt)**). **Embedding-Converter** trains an MLP converter with additional global and local geometry-preservation losses and evaluates on retrieval benchmarks (**[Embedding-Converter](./references/EMBEDDING-CONVERTER-A-UNIFIED-FRAMEWORK-FOR-CROSS-MODEL-EMBEDDING-TRANSFORMATION/meta/meta_info.txt)**). It is unclear whether the *minimal* Drift-Adapter-style MSE objective (with only a few thousand pairs) is sufficient for privacy-limited upgrades, or whether the additional objectives and/or larger training sets are necessary.

This leaves a concrete, decision-relevant open question for practitioners:

> When upgrading an embedder, is a drift adapter trained on a generic public anchor corpus sufficient to recover most of the benefit of an in-domain trained adapter on the target corpus?

If the answer is “yes”, teams can deploy improved embedders without touching sensitive data. If the answer is “no”, then privacy-limited systems must either accept degraded retrieval, maintain dual indices, or pay the cost of full re-embedding.

### Key Insight and Hypothesis

**Key insight:** Many embedding objectives are approximately invariant to orthogonal transformations, so independently trained models can encode similar information in embedding spaces that differ mainly by a global transform. Recent theory shows that if dot products are approximately preserved between two embedding sets, an orthogonal Procrustes map can align them with bounded error (**[When Embedding Models Meet](./references/When-Embedding-Models-Meet-Procrustes-Bounds-and-Applications/meta/meta_info.txt)**). Drift-Adapter empirically finds that a small residual MLP can learn a mostly “smooth” transform between two models (Section 5.1, **[Drift-Adapter §5.1](./references/Drift-Adapter-A-Practical-Approach-to-Near-Zero-Downtime-Embedding-Model-Upgrades-in-Vector-Databases/sections/5.1 Main Performance and Variance.md)**).

**Hypothesis:** For a fixed pair of embedding models \(f_{old}\to f_{new}\), the dominant component of drift is **model-pair-specific** rather than **domain-specific**. Therefore, a small residual-MLP adapter trained by plain MSE on paired embeddings from a generic public anchor corpus (e.g., Wikipedia) will recover most of the in-domain adapter’s retrieval gains on an out-of-domain target corpus.

**Mechanism hypothesis (why Drift-Adapter MSE might differ from vec2vec / Embedding-Converter):** If most drift is close to an isometry plus mild smooth distortion, then MSE on a modest number of paired points is enough to learn a useful map. In contrast, if transfer requires preserving neighborhood structure and higher-order geometry more explicitly (as targeted by Embedding-Converter’s local/global similarity losses) or requires distribution matching without pairing (as in vec2vec), then the simple MSE-trained adapter will fail under domain shift.

**Why we could be wrong:** Drift could be strongly domain-conditional (different regions of the embedding space drift differently), so an adapter trained on Wikipedia-like text may not align well on specialized domains (e.g., finance, argumentation, biomedical). A second confound is that an adapter might learn a degenerate shrinkage/centering that improves retrieval metrics without meaningful alignment; we control for this with a shuffled-pair null baseline.

---

## Proposed Approach

### Overview

We propose **Public-Anchor Drift Adapter (PADA)**: train the same drift-adapter architecture as Drift-Adapter, but using only **public anchor text** to construct paired embeddings for alignment.

We evaluate on **BEIR** (a heterogeneous benchmark suite for zero-shot information retrieval evaluation) tasks using standard relevance judgments, comparing:

- an **in-domain adapter** (upper bound; uses unlabeled target-corpus text), and
- a **public-anchor adapter** (ours; uses unlabeled Wikipedia text),

under a fixed model upgrade \(f_{old}\to f_{new}\).

### Method Details

#### Problem setup

Let \(f_{old}\) be the legacy embedding model used to build the current ANN index over corpus \(\mathcal{D}\), and \(f_{new}\) be the upgraded model. The legacy vector database stores \(x_i = f_{old}(d_i)\) for \(d_i\in\mathcal{D}\). For a query \(q\), the upgraded system produces \(z_q=f_{new}(q)\), but the database expects vectors in the old space.

We learn an adapter \(g_\theta\) such that \(g_\theta(f_{new}(t)) \approx f_{old}(t)\) for texts \(t\) in some training set.

#### Adapter architecture (fixed across conditions)

We use Drift-Adapter’s residual MLP adapter (chosen because it was the strongest across their experiments):

\[
 g_\theta(x) = x + W_2\,\mathrm{GELU}(W_1 x + b_1) + b_2,
\]

where \(W_1\in\mathbb{R}^{h\times d}\), \(W_2\in\mathbb{R}^{d\times h}\), \(h=256\), and \(d\) is the embedding dimension (same for \(f_{old}\) and \(f_{new}\) in this proposal). We L2-normalize embeddings before training and inference, following Drift-Adapter’s setup (**[Drift-Adapter §4](./references/Drift-Adapter-A-Practical-Approach-to-Near-Zero-Downtime-Embedding-Model-Upgrades-in-Vector-Databases/sections/4 Experimental Setup.md)**).

Training minimizes mean squared error over paired embeddings:
\[
\mathcal{L}(\theta)=\frac{1}{N_p}\sum_{j=1}^{N_p}\|g_\theta(b_j)-a_j\|_2^2,
\]
where \(a_j=f_{old}(t_j)\), \(b_j=f_{new}(t_j)\).

#### Training data conditions (the only difference)

- **In-domain adapter (upper bound):** \(t_j\) are sampled from the *target corpus* \(\mathcal{D}\) (documents only; no qrel labels used).
- **Public-anchor adapter (ours):** \(t_j\) are sampled from a *public anchor corpus* (Wikipedia paragraphs).

Both use the same number of pairs \(N_p\), same optimizer, and same early stopping.

#### Model pair (pre-registered)

To avoid confounding by embedding dimension mismatch, we use a same-dimension upgrade pair:

- \(f_{old}\): `sentence-transformers/all-distilroberta-v1` (HuggingFace: https://huggingface.co/sentence-transformers/all-distilroberta-v1)
- \(f_{new}\): `sentence-transformers/all-mpnet-base-v2` (HuggingFace: https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

This is a realistic “drop-in upgrade” within the SentenceTransformers ecosystem.

### Key Innovations

1. **Privacy-limited upgrade setting:** isolates the practically important case where *no in-domain text* is permitted for adapter fitting.
2. **Decisive transfer test:** directly measures how much of the in-domain adapter gain can be recovered by a public-anchor adapter on out-of-domain IR benchmarks.
3. **Null control for spurious gains:** includes a shuffled-pair baseline to distinguish meaningful alignment from degenerate regularization.

---

## Related Work

### Field Overview

**Embedding alignment and interoperability.** Aligning embedding spaces is widely studied in cross-lingual word embeddings (often via orthogonal Procrustes) and more generally for making independently trained embedders interoperable. “When Embedding Models Meet” provides theory and empirical evidence that Procrustes post-processing can align models for retrieval and mixed-modality search (**[When Embedding Models Meet](./references/When-Embedding-Models-Meet-Procrustes-Bounds-and-Applications/meta/meta_info.txt)**). Multi-way alignment extends this idea to \(M\ge 3\) models via a shared reference “universe” with cycle consistency (**[Multi-Way Representation Alignment](./references/MULTI-WAY-REPRESENTATION-ALIGNMENT/meta/meta_info.txt)**).

**Operational embedding upgrades.** Drift-Adapter frames alignment as a practical alternative to full re-embedding or dual indices and shows strong recovery with small paired samples, but assumes paired samples from the target corpus (**[Drift-Adapter](./references/Drift-Adapter-A-Practical-Approach-to-Near-Zero-Downtime-Embedding-Model-Upgrades-in-Vector-Databases/meta/meta_info.txt)**).

**Backward compatibility via training-time constraints.** Another line of work enforces backward compatibility during training of new embedding models (e.g., BC-Aligner / backward-compatible training). These approaches often require controlling the training procedure of \(f_{new}\), which is not possible when upgrading to a frozen third-party embedder.

### Related Papers

- **[Drift-Adapter: A Practical Approach to Near Zero-Downtime Embedding Model Upgrades in Vector Databases](./references/Drift-Adapter-A-Practical-Approach-to-Near-Zero-Downtime-Embedding-Model-Upgrades-in-Vector-Databases/meta/meta_info.txt)**: Residual-MLP / affine / Procrustes adapters trained on in-domain paired samples for vector-DB upgrades.
- **[When Embedding Models Meet: Procrustes Bounds and Applications](./references/When-Embedding-Models-Meet-Procrustes-Bounds-and-Applications/meta/meta_info.txt)**: Theory + practice of Procrustes post-processing for embedding interoperability.
- **[Multi-Way Representation Alignment](./references/MULTI-WAY-REPRESENTATION-ALIGNMENT/meta/meta_info.txt)**: Multi-model shared-universe alignment (GPA/GCPA) with retrieval benefits.
- **[Learning Backward Compatible Embeddings](https://arxiv.org/abs/2206.03040)**: Training-time backward compatibility via learned transformations across versions.
- **[MixBCT: Towards Self-Adapting Backward-Compatible Training](https://arxiv.org/abs/2308.06948)**: Improves backward-compatible training under harder shifts.
- **[Unsupervised Alignment of Embeddings with Wasserstein Procrustes](https://arxiv.org/abs/1805.11222)**: Unsupervised Procrustes alignment via optimal transport.
- **[Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087)**: Unsupervised bilingual embedding alignment; popularized Procrustes refinement and CSLS.
- **[MUSE: A Library for Multilingual Unsupervised and Supervised Word Embeddings](https://arxiv.org/abs/1710.04087)**: Tooling and recipes for embedding alignment (closely related to Procrustes methods).
- **[VecMap: A Framework for Mapping Embeddings](https://aclanthology.org/P18-2037/)**: Classical embedding mapping toolkit and evaluation.
- **[Schönemann (1966): Orthogonal Procrustes](https://doi.org/10.1007/BF02289451)**: Foundational closed-form Procrustes solution.
- **[BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)**: Standard benchmark suite including SciFact and TREC-COVID.
- **[Sentence-BERT](https://arxiv.org/abs/1908.10084)**: Bi-encoder sentence embeddings foundation for many retrievers.
- **[SimCSE](https://arxiv.org/abs/2104.08821)**: Contrastive sentence embeddings; representative training objective with orthogonal invariances.
- **[FAISS](https://arxiv.org/abs/1702.08734)**: ANN indexing library (HNSW/IVF/PQ) widely used in vector databases.
- **[HNSW](https://arxiv.org/abs/1603.09320)**: Graph-based ANN index used in many vector DBs.
- **[BC-Aligner (backward-compatible learning)](https://arxiv.org/abs/2003.13319)**: Backward compatibility via training-time constraints in representation learning (image domain).
- **[Harnessing the Universal Geometry of Embeddings (vec2vec)](https://arxiv.org/abs/2505.12540)**: Translating between embedding spaces using shared geometric structure.
- **[Embedding-Converter](https://openreview.net/forum?id=ga9PAnFsAt)**: Learned embedding-to-embedding conversion for model migration.
- **[OSCAR: A Large-Scale Cross-lingual Pretrained Model](https://arxiv.org/abs/2004.06165)**: Example of multilingual pretraining where alignment is often needed.
- **[CSLS: Cross-domain Similarity Local Scaling](https://aclanthology.org/P18-1073/)**: A similarity correction widely used in embedding alignment to mitigate hubness.

### Taxonomy

| Family | Core idea | Representative papers | Typical evaluation | Key limitation |
|---|---|---|---|---|
| Post-hoc linear alignment | Fit an orthogonal/linear map between spaces | Schönemann 1966; Procrustes; When Embedding Models Meet | cross-model retrieval; bilingual lexicon induction | Needs paired anchors; may be insufficient for nonlinear drift |
| Post-hoc nonlinear adapters | Small MLP or low-rank affine mapping | Drift-Adapter; Embedding-Converter; vec2vec | retrieval recovery (Recall/nDCG) | Still needs paired data; transfer across domains unclear |
| Training-time backward compatibility | Constrain \(f_{new}\) to stay compatible with \(f_{old}\) | Learning Backward Compatible Embeddings; MixBCT | downstream task retention across versions | Requires control over model training |
| Multi-way alignment | Align \(M\ge 3\) spaces via shared universe | Multi-Way Representation Alignment | any-to-any retrieval, stitching | Requires paired correspondences across models |
| Operational upgrade strategies | Avoid downtime with system design | Drift-Adapter; dual index; lazy re-embed | system cost/latency | Doesn’t answer privacy-limited adapter fitting |

### Closest Prior Work

1. **Embedding-Converter**: Trains an MLP converter between embedding models using regression plus explicit **global and local geometry-preservation losses**, and evaluates transfer on retrieval benchmarks. Our work asks whether these extra losses are necessary for privacy-limited upgrades, or whether a **plain MSE Drift-Adapter-style MLP** with only ~5k public pairs already matches the in-domain adapter.

2. **vec2vec**: Learns *unpaired* embedding translation using adversarial and cycle-consistency losses, and reports cross-domain generalization. Our work is complementary: we focus on the common deployment setting where paired embeddings can be computed on public text, and we test whether a much simpler supervised objective (paired MSE) is sufficient for retrieval transfer.

3. **Drift-Adapter**: Closest operational framing and adapter architecture; evaluates only adapters trained on samples from the target corpus and proposes public-data training only as future work (**[Drift-Adapter §6](./references/Drift-Adapter-A-Practical-Approach-to-Near-Zero-Downtime-Embedding-Model-Upgrades-in-Vector-Databases/sections/6 Discussion.md)**). We directly test public-anchor vs in-domain fitting under compute-matched conditions.

4. **When Embedding Models Meet**: Provides theoretical grounding for why simple isometric alignment can work when dot products are preserved, but does not address distribution shift between anchor and target corpora.

**Novelty Kill Search Summary:** Searched for combinations of “embedding model upgrade + public anchor + adapter”, “vector database upgrade + Procrustes + public dataset”, “drift adapter transferability public corpus”, “vec2vec embedding translation retrieval”, and “Embedding-Converter cross-domain BEIR”. Found (i) Drift-Adapter (MSE-trained adapters on *in-domain* pairs; suggests public-anchor training but does not test it), (ii) Embedding-Converter (MLP converter with explicit global/local geometry-preservation losses evaluated on retrieval benchmarks), and (iii) vec2vec (unpaired adversarial/cycle-consistent translation that generalizes across domains). We did not find a direct study of whether the **minimal Drift-Adapter MSE objective** with only **~5k paired public anchors** matches these more complex translators under domain shift, nor a compute-matched comparison of public-anchor vs in-domain fitting for the Drift-Adapter-style adapter. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation (for our setting) | What we change | What we learn |
|---|---|---|---|---|
| Drift-Adapter | Residual-MLP / affine / Procrustes adapters trained on **in-domain** paired samples for near-zero-downtime upgrades | Requires any in-domain paired texts/embeddings | Keep the **same** MSE-trained residual MLP; swap in-domain pairs → **public Wikipedia** pairs | Whether *domain-specific* pairing is actually necessary for a Drift-Adapter-style upgrade |
| Embedding-Converter | MLP converter with regression + **global/local geometry** losses; trained with hyperparameter tuning and data diversity | More complex objective/model; may require curated retrieval corpora and validation data | Test if a **plain MSE** Drift-Adapter adapter with only ~5k public pairs already matches the in-domain adapter | Whether the extra geometry losses/data diversity are necessary in the privacy-limited upgrade regime |
| vec2vec | **Unpaired** embedding translation via adversarial + cycle-consistency; reports cross-domain transfer | Heavier optimization and unpaired assumptions; not a minimal “drop-in” adapter recipe | Focus on the paired-but-public regime and quantify **gain recovery vs an in-domain upper bound** | Whether paired MSE is sufficient (or whether one needs stronger distribution/geometry constraints) |
| When Embedding Models Meet | Procrustes theory/empirics for embedding interoperability | Does not study anchor-vs-target distribution shift | Evaluate public-anchor training under domain shift | When simple geometric assumptions actually transfer beyond the anchor distribution |

---

## Experiments

### Experimental Setup

**Benchmarks (pre-registered):**
- **SciFact (BEIR)**: a scientific claim verification retrieval task; given a claim, retrieve relevant scientific abstracts.
- **TREC-COVID (BEIR)**: a biomedical literature retrieval task; given a COVID-related topic, retrieve relevant papers.
- **FiQA-2018 (BEIR)**: a finance-domain retrieval task; queries are financial questions and documents are finance-related passages.
- **ArguAna (BEIR)**: an argumentation-domain retrieval task; queries are arguments and documents are counter-arguments.

Use the official BEIR data + evaluation script (https://github.com/beir-cellar/beir).

**Embedding models:**
- \(f_{old}\): `sentence-transformers/all-distilroberta-v1`
- \(f_{new}\): `sentence-transformers/all-mpnet-base-v2`

**Indexing and retrieval:**
- Build an ANN index (FAISS HNSW or FlatIP) over corpus embeddings.
- Similarity: inner product of L2-normalized vectors (cosine similarity).

**Oracle (full re-embedding reference):** embed corpus with \(f_{new}\) and retrieve with \(f_{new}\) queries.

**Main methods (≤3 core conditions):**
1. **Misaligned:** \(f_{new}(q)\) retrieves against \(f_{old}(d)\) index with no adapter.
2. **In-domain adapter:** train \(g_{in}\) on \(N_p\) unlabeled corpus documents \(d\in\mathcal{D}\) paired under \(f_{old},f_{new}\); retrieve with \(g_{in}(f_{new}(q))\) against \(f_{old}(d)\) index.
3. **Public-anchor adapter (ours):** train \(g_{pub}\) on \(N_p\) unlabeled Wikipedia paragraphs paired under \(f_{old},f_{new}\); retrieve with \(g_{pub}(f_{new}(q))\) against \(f_{old}(d)\) index.

**Ablation (required):**
- **Shuffled-pair adapter:** train on the same Wikipedia samples but randomly permute the \(f_{old}(t)\) targets, breaking correspondence. This tests whether gains require meaningful pairing.

**Training pairs:**
- \(N_p=5000\) paired texts for each adapter.
- In-domain: sample \(N_p\) documents from the target corpus (or all documents if corpus <5000).
- Public anchor: sample \(N_p\) Wikipedia paragraphs (fixed random seed).

**Training hyperparameters (fixed):**
- AdamW, lr=3e-4, weight_decay=0.01, batch_size=256.
- Up to 50 epochs with early stopping on validation loss (patience=5), 80/20 train/val split.

**Randomness / seeds:**
- Train each adapter with 3 seeds: seeds=[0,1,2]. (Resample training pairs per seed to capture sampling variance.)
- Retrieval evaluation itself is deterministic given embeddings.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SciFact (BEIR) | Scientific-claim retrieval: given a claim, retrieve supporting/refuting abstracts | nDCG@10 (normalized Discounted Cumulative Gain at 10; higher is better), Recall@10 (fraction of relevant docs in top-10; higher is better) | test | https://github.com/beir-cellar/beir | BEIR `EvaluateRetrieval` |
| TREC-COVID (BEIR) | Biomedical retrieval: given a COVID topic, retrieve relevant papers | nDCG@10, Recall@10 | test | https://github.com/beir-cellar/beir | BEIR `EvaluateRetrieval` |
| FiQA-2018 (BEIR) | Finance QA retrieval: retrieve relevant finance passages for a query | nDCG@10, Recall@10 | test | https://github.com/beir-cellar/beir | BEIR `EvaluateRetrieval` |
| ArguAna (BEIR) | Argument retrieval: retrieve counter-arguments for a given argument | nDCG@10, Recall@10 | test | https://github.com/beir-cellar/beir | BEIR `EvaluateRetrieval` |

### Main Results

We will report both absolute metrics and an **adapter gain recovery ratio**:
\[
\rho = \frac{M(\text{public-anchor}) - M(\text{misaligned})}{M(\text{in-domain}) - M(\text{misaligned})},
\]
where \(M\) is nDCG@10 (primary) or Recall@10 (secondary).

#### Results Table (to be filled by verification)

| Method | Corpus embeddings | Query embeddings | Adapter training data | SciFact nDCG@10 (mean±std) | TREC-COVID nDCG@10 (mean±std) | FiQA nDCG@10 (mean±std) | ArguAna nDCG@10 (mean±std) | Source | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| Oracle (full re-embed) | \(f_{new}\) | \(f_{new}\) | - | TBD | TBD | TBD | TBD | This work | Reference target |
| Misaligned | \(f_{old}\) | \(f_{new}\) | - | TBD | TBD | TBD | TBD | This work | No adapter |
| In-domain adapter | \(f_{old}\) | \(g_{in}(f_{new})\) | target corpus docs | TBD | TBD | TBD | TBD | This work | Upper bound (unlabeled) |
| **Public-anchor adapter (ours)** | \(f_{old}\) | \(g_{pub}(f_{new})\) | Wikipedia paragraphs | TBD | TBD | TBD | TBD | This work | No in-domain text |
| Shuffled-pair adapter (ablation) | \(f_{old}\) | \(g_{shuffle}(f_{new})\) | Wikipedia (shuffled) | TBD | TBD | TBD | TBD | This work | Null control |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Shuffled pairs | Break correspondence between \(f_{new}(t)\) and \(f_{old}(t)\) | Performance stays near misaligned if pairing is necessary |

### Experimental Rigor

- **Compute matching:** in-domain and public-anchor adapters use the same \(N_p\), same optimizer, same early stopping.
- **Headroom check:** if \(M(\text{in-domain})-M(\text{misaligned})\) is <0.05 on a benchmark, we treat \(\rho\) as low-signal for that benchmark and do not base the main decision on it.
- **Leakage avoidance:** adapter training uses only corpus documents (unlabeled) and never uses query text or qrels.

**Resource Estimate**
- **Compute budget**: ≤50 GPU-hours total.
  - Embedding corpora for **4 BEIR datasets** with **two** SentenceTransformers models (dominant cost).
  - Adapter training on 5k pairs is negligible (<0.5 GPU-hours total across seeds).
- **GPU memory**: ≤16GB per GPU is sufficient for these encoders.
- **API usage**: None.

---

## Success Criteria

**Hypothesis (directional):** Training a drift adapter on a public anchor corpus recovers most of the in-domain adapter’s retrieval improvement on at least one out-of-domain BEIR task.

**Decision Rule (concrete):**

Let \(M\) be nDCG@10.

1. **Primary decision benchmark selection (pre-registered):**
   - Use **SciFact** as the decision benchmark if \(M(\text{in-domain})-M(\text{misaligned}) \ge 0.05\).
   - Else, use **TREC-COVID** if its in-domain gain is \(\ge 0.05\).
   - Else, use **FiQA-2018** if its in-domain gain is \(\ge 0.05\).
   - Else, use **ArguAna** if its in-domain gain is \(\ge 0.05\).
   - Else, declare the adapter-transfer question **not meaningfully testable** for this model pair on these corpora and report the negative practitioner finding (“upgrade drift is small here; adapters may be unnecessary”).

2. **Proceed (public-anchor is sufficient):** on the decision benchmark,
   - \(\rho \ge 0.90\) (public-anchor recovers ≥90% of in-domain gain), **and**
   - \(M(\text{in-domain}) - M(\text{public-anchor}) \le 0.01\) absolute, **and**
   - \(M(\text{public-anchor}) - M(\text{shuffled}) \ge 0.02\) absolute (beats null control).

3. **Refute (in-domain data is necessary):** on the decision benchmark,
   - \(\rho < 0.50\) **or** \(M(\text{in-domain}) - M(\text{public-anchor}) > 0.03\) absolute.

4. **Inconclusive:** anything in between; report results and recommend treating public-anchor adapters as a heuristic that requires validation on each domain.

---

## Impact Statement

If successful, this provides a practical recipe for privacy-limited agent memory and RAG systems: when upgrading an embedder, train a small drift adapter on public text only and reuse the existing index without re-embedding sensitive corpora. If it fails, the negative result is also decision-changing: it indicates that public anchors do not reliably substitute for in-domain paired samples, motivating investment in privacy-preserving in-domain adapter training or full re-embedding.

---

## References

- [Drift-Adapter: A Practical Approach to Near Zero-Downtime Embedding Model Upgrades in Vector Databases](./references/Drift-Adapter-A-Practical-Approach-to-Near-Zero-Downtime-Embedding-Model-Upgrades-in-Vector-Databases/meta/meta_info.txt) - Vejendla, 2025
- [When Embedding Models Meet: Procrustes Bounds and Applications](./references/When-Embedding-Models-Meet-Procrustes-Bounds-and-Applications/meta/meta_info.txt) - Maystre et al., 2025
- [Multi-Way Representation Alignment](./references/MULTI-WAY-REPRESENTATION-ALIGNMENT/meta/meta_info.txt) - Achara et al., 2026
- [Learning Backward Compatible Embeddings](https://arxiv.org/abs/2206.03040) - Hu et al., 2022
- [MixBCT: Towards Self-Adapting Backward-Compatible Training](https://arxiv.org/abs/2308.06948) - 2023
- [Unsupervised Alignment of Embeddings with Wasserstein Procrustes](https://arxiv.org/abs/1805.11222) - Grave et al., 2018
- [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087) - Conneau et al., 2018
- [VecMap: A Framework for Mapping Embeddings](https://aclanthology.org/P18-2037/) - Artetxe et al., 2018
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Reimers & Gurevych, 2019
- [SimCSE](https://arxiv.org/abs/2104.08821) - Gao et al., 2021
- [BEIR](https://arxiv.org/abs/2104.08663) - Thakur et al., 2021
- [FAISS](https://arxiv.org/abs/1702.08734) - Johnson et al., 2017
- [HNSW](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin, 2016
- [Harnessing the Universal Geometry of Embeddings (vec2vec)](https://arxiv.org/abs/2505.12540) - 2025
- [Embedding-Converter](https://openreview.net/forum?id=ga9PAnFsAt) - 2025
- [Cross-domain Similarity Local Scaling (CSLS)](https://aclanthology.org/P18-1073/) - Lample et al., 2018
