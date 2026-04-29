# untitled

# Gauge-Whitened Routing: Training-Free Query Assignment for Saap-Style Sparse Attention

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Long-context inference is increasingly important for deploying large language models (LLMs) in applications such as multi-document question answering, codebase understanding, and agentic workflows with large scratchpads. However, autoregressive decoding with long contexts is dominated by self-attention’s memory bandwidth and the **key–value (KV) cache** footprint (the cached key/value tensors for all previous tokens).

A recent line of work accelerates long-context inference by making attention **sparse at inference time** using approximate nearest neighbor (ANN) ideas (hashing, clustering, graph search, or retrieval). Here “ANN” refers to algorithms that quickly find approximate nearest neighbors in high-dimensional vector spaces (e.g., clustering keys into buckets and probing only the top few buckets per query), avoiding an exhaustive scan.

A representative recent method is **Saap** (Self-Attention with Asymmetric Partitions), which clusters keys with k-means and uses a learned query-to-bucket classifier to visit only a small fraction of the KV cache during decoding ([Saap](./references/Inference-time-sparse-attention-with-asymmetric-indexing/meta/meta_info.txt)). At routing budget \(\ell=32\), Saap reports **100%** Needle-in-a-Haystack accuracy while visiting only **4.0%** of keys at 128k context and **5.3%** at 500k context (Saap Sec. 5.2 Table 2; see `./references/Inference-time-sparse-attention-with-asymmetric-indexing/sections/5.2 Comparison with baselines.md`). Saap also reports that naively clustering/assigning with RoPE vectors can fail: “K-means (roped inputs, \(\ell=32\))” achieves only **69.33** on 500k NIH versus **98.67** for “K-means (no rope, \(\ell=32\))” (same table).

Saap’s speed benefit is also substantial: on a 171k context (\(\ell=32\), **4.4%** selectivity), Saap reports **18 μs/head** versus **50 μs/head** for FlashAttention-v2 (Saap Sec. 5.3; see `./references/Inference-time-sparse-attention-with-asymmetric-indexing/sections/5.3 Timings.md`; lower is better).

Despite these gains, Saap introduces a practical adoption barrier: it trains a **per-head** query classifier offline (a 2-layer MLP) for query routing. Saap reports this takes **~25 minutes per head on one GPU** (Saap Sec. 5.1). For a modern GQA model with many layers and KV groups, this can translate to a non-trivial one-time cost for every new base model, plus additional engineering complexity.

### The Problem

**Why does Saap need a learned query router at all?** Saap argues that standard ANN partitioning fails because **keys and queries are out-of-distribution (OOD)**: they are produced by different linear projections, and in a typical head the query and key clouds lie in different regions (even on opposite sides of the origin), making symmetric bucket assignment ineffective (Saap Sec. 3.3).

Saap’s response is an **asymmetric** design: cluster keys with k-means, but assign queries with a trained classifier \(f_q\) that predicts which \(\ell\) buckets contain most of the attention mass (Saap Sec. 4.2).

Here \(\ell\) is the routing budget (number of buckets visited per query); smaller \(\ell\) means fewer keys are accessed (higher sparsity). Saap trains \(f_q\) to maximize **attention-mass recall@\(\ell\)**: the fraction of a query’s total attention weight captured by the selected \(\ell\) buckets (higher is better; 1.0 means all attention mass is captured).

This raises an important open question for inference-time sparse attention:

> Is the learned, non-linear query router actually necessary, or is it mostly compensating for a *removable linear mismatch* between the query and key representation spaces?

If most of the mismatch is “geometric” (basis / scaling / anisotropy), then a training-free correction could make Saap-style routing easier to deploy and easier to generalize.

### Key Insight and Hypothesis

**Key insight (dot-product invariance of attention):** For a single attention head, attention scores depend on \(QK^\top\) (equivalently, the bilinear form \(q^\top k\)). Because only the product \(QK^\top\) matters, we can apply an invertible linear transform \(A\) to the query projection and the inverse-transpose to the key projection \((W_Q\to W_QA,\; W_K\to W_K(A^{-1})^\top)\) without changing attention logits ([Maximal Gauge Symmetry](./references/Maximal-Gauge-Symmetry-in-Transformer-Architectures/meta/meta_info.txt), Def. 3.1).

Equivalently at the activation level, for any invertible \(R\), the change of basis
\[
q' = qR,\quad k' = kR^{-\top}
\]
preserves dot products: \(q'^\top k' = q^\top k\).

**Note on RoPE models:** With rotary position embeddings (RoPE), the exact global invariance group is restricted to transforms that commute with RoPE rotations. In this proposal we do **not** modify model weights or RoPE-applied vectors.

Instead, we use **de-RoPE vectors** (keys/queries with the RoPE rotation removed) *only for routing*, as in Saap: we compute \(R\) on de-RoPE vectors, route using \(q',k'\) in the de-RoPE routing space, and then compute the actual attention weights over the selected keys using the model’s original RoPE attention. Therefore, any routing-space transform \(R\) cannot change the model’s outputs; it only changes which buckets are selected.

**Hypothesis:** In Saap-style clustering-based routing (performed on de-RoPE vectors as in Saap), there exists a simple *closed-form* choice of \(R\) that equalizes low-order statistics of queries and keys (“gauge whitening”), making **symmetric nearest-centroid routing** competitive with Saap’s trained MLP router in terms of the **attention-mass recall** objective that Saap optimizes (defined in Experiments; higher is better and 1.0 means all attention mass is captured by the selected buckets).

This could fail for a substantive reason: Saap’s MLP might be learning genuinely non-linear routing boundaries that cannot be recovered from second-order statistics. In that case, we would obtain a negative result that clarifies when learned routing is essential.

---

## Proposed Approach

### Overview

We propose **Gauge-Whitened Routing (GWR)**, a training-free replacement for Saap’s learned query assignment model.

GWR keeps Saap’s core idea of partitioning keys into k-means buckets, but replaces the query MLP \(f_q\) with a closed-form, dot-product-preserving linear map that attempts to remove most of the query–key distribution mismatch before routing.

### Method Details

#### Setting

We follow Saap’s routing setup:
- Use **de-RoPE** keys/queries for partitioning and routing (Saap Sec. 3.4 “Removing temporal bias”).
- Use spherical k-means with \(C\) clusters to partition keys (Saap uses \(C=1024\) centroids; Saap Sec. 3.3).
- Route each query to \(\ell\) buckets, and approximate attention by attending to all keys in those buckets (Saap Sec. 4.3).

We follow Saap’s typical operating point \(\ell=32\), which yields 100% Needle-in-a-Haystack accuracy at 128k and 500k with 4.0–5.3% selectivity (Saap Sec. 5.2 Table 2).

Our proposal focuses on the query-routing step only.

#### Step 1: collect per-head second-moment statistics

For a fixed model, layer \(\ell_{layer}\), and **attention head** (one query head; for GQA you may optionally treat a whole KV group), collect de-RoPE query and key vectors \(q\in\mathbb{R}^{d_h}\), \(k\in\mathbb{R}^{d_h}\) from a calibration corpus.

Compute **uncentered** second moments (to avoid introducing a translation that would break dot-product preservation). Using the common row-vector convention (as in the gauge-symmetry reference), treat \(q,k\in\mathbb{R}^{1\times d_h}\):
\[
M_q = \mathbb{E}[q^\top q] + \epsilon I,\quad M_k = \mathbb{E}[k^\top k] + \epsilon I
\]
where \(\epsilon\) is a small diagonal regularizer (e.g., \(10^{-4}\)).

#### Step 2: solve a congruence equation for a dot-product-preserving “whitening gauge”

We choose an SPD matrix \(S\) satisfying
\[
S\,M_q\,S = M_k.
\]
A standard closed-form SPD solution is
\[
S = M_k^{1/2}\,\left(M_k^{1/2} M_q M_k^{1/2}\right)^{-1/2}\,M_k^{1/2}.
\]
Then take \(R = S^{1/2}\) and define the gauge-transformed vectors
\[
q' = qR,\quad k' = kR^{-\top}.
\]
This has two properties:
1) **Dot-product preservation:** \(q'^\top k' = q^\top k\) for all \(q,k\).
2) **Second-moment matching:** \(\mathbb{E}[q'^\top q'] = \mathbb{E}[k'^\top k']\).

Intuition: we do not try to learn bucket boundaries; instead we pick a basis where the query and key clouds have comparable scale/anisotropy, so that symmetric nearest-centroid routing is less OOD.

#### Step 3: cluster transformed keys and route transformed queries

- Run spherical k-means on \(k'\) to obtain centroids \(\{c_j\}_{j=1}^C\) and hard key assignments \(H_k\in\{0,1\}^{n_k\times C}\).
- For a query \(q'\), select the top-\(\ell\) buckets by centroid similarity:
\[
\text{Buckets}(q') = \operatorname{Top}\_\ell\left(\{\langle q', c_j\rangle\}_{j=1}^C\right).
\]
This is the only routing operation at inference time.

### Key Innovations

1. **Uses attention gauge symmetry for inference-time routing**: we exploit the exact invariance \(q\to qR,\;k\to kR^{-\top}\) to search for a routing-friendly representation without changing the dot-product similarities that define attention.
2. **Training-free replacement for Saap’s per-head MLP**: routing uses only closed-form matrix functions of \(d_h\times d_h\) second-moment estimates.
3. **Decision-oriented objective**: we evaluate directly on Saap’s routing target (attention-mass recall over buckets), producing a clear positive result (“MLP not needed”) or negative result (“non-linear routing is essential”).

---

## Related Work

### Field Overview

Efficient attention methods can be grouped into: (i) **static sparse patterns** (local windows + global tokens), (ii) **data-dependent sparse patterns** (hashing/clustering/routing), (iii) **kernel/linear attention** approximations, and (iv) **retrieval/vector-search** approaches that treat KV caches as searchable memory.

For inference-time sparse attention on pretrained LLMs, two recurring issues are: (a) **distribution mismatch** between queries and keys that breaks off-the-shelf ANN assumptions, and (b) **hardware compliance**, since many ANN data structures are not GPU-friendly.

### Related Papers

- **[Inference-time sparse attention with asymmetric indexing (Saap)](./references/Inference-time-sparse-attention-with-asymmetric-indexing/meta/meta_info.txt)**: Clusters keys (k-means) and trains a per-head MLP router to select buckets for queries during decoding, explicitly targeting query–key OOD in KV-cache indexing.
- **[RetrievalAttention](./references/RetrievalAttention-Accelerating-Long-Context-LLM-Inference-via-Vector-Retrieval/meta/meta_info.txt)**: Treats attention as vector retrieval from the KV cache and proposes an attention-aware ANN procedure to mitigate query–key OOD during inference.
- **[Sparse Attention with Learning to Hash (LHA)](./references/Sparse-Attention-with-Learning-to-Hash/meta/meta_info.txt)**: Learns separate (query, key) hash functions to build content-dependent sparse attention without assuming queries and keys share the same distribution.
- **[Efficient Content-Based Sparse Attention with Routing Transformers](./references/Efficient-Content-Based-Sparse-Attention-with-Routing-Transformers/meta/meta_info.txt)**: Uses online k-means routing during training to reduce attention complexity by allocating tokens to clusters.
- **[Reformer](./references/Reformer-The-Efficient-Transformer/meta/meta_info.txt)**: Uses locality-sensitive hashing to approximate attention in \(O(n\log n)\) time, illustrating the classic “hash buckets for attention” approach.
- **[Rethinking Attention with Performers](./references/Rethinking-Attention-with-Performers/meta/meta_info.txt)**: Approximates softmax attention with random features (FAVOR+) to obtain linear-time attention without explicit sparsity.
- **[Maximal Gauge Symmetry in Transformer Architectures](./references/Maximal-Gauge-Symmetry-in-Transformer-Architectures/meta/meta_info.txt)**: Formalizes attention’s query–key gauge symmetry \((W_Q,W_K)\to(W_QA, W_K(A^{-1})^\top)\), which motivates dot-product-preserving reparameterizations.
- **[Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting](../../papers/paper_summaries/Tactic Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs.md)**: A concurrent training-free sparse attention method that uses k-means clustering plus attention-score distribution fitting to choose a dynamic token budget at inference time. Unlike our approach, it does not attempt to align the query/key representation spaces via a dot-product-preserving transform.

- **[FlashAttention](https://arxiv.org/abs/2205.14135)**: Provides fast exact attention kernels and is a standard baseline for efficient attention implementations.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**: Improves FlashAttention with better IO-aware scheduling and is widely used for long-context training/inference.
- **[Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/)**: Accelerates decoding-time attention by parallelizing over the KV sequence dimension, highlighting that long-context decoding can be memory-bandwidth bound.

- **[Longformer](https://arxiv.org/abs/2004.05150)**: Uses local-window attention plus a small set of global tokens to scale to longer contexts (typically requiring training/fine-tuning).
- **[BigBird](https://arxiv.org/abs/2007.14062)**: Combines local, random, and global attention patterns with theoretical guarantees for sparse attention.
- **[Sparse Transformer](https://arxiv.org/abs/1904.10509)**: Uses factorized sparse patterns to reduce attention complexity in long sequences.

- **[Linformer](https://arxiv.org/abs/2006.04768)**: Approximates attention with low-rank projections over the sequence dimension.
- **[Nyströmformer](https://arxiv.org/abs/2102.03902)**: Approximates attention using Nyström methods, trading accuracy for subquadratic cost.

- **[StreamingLLM](https://arxiv.org/abs/2309.17453)**: Studies “attention sinks” and proposes streaming decoding by retaining a sink token plus a sliding window.
- **[HyperAttention](https://arxiv.org/abs/2310.05869)**: Combines hashing and sampling to approximate attention in near-linear time for long contexts.
- **[MagicPIG](https://arxiv.org/abs/2410.16179)**: Uses LSH sampling for efficient LLM generation and highlights the importance of key centering under attention-sink geometry.
- **[LOOKAT](https://arxiv.org/abs/2601.10155)**: Proposes lookup-optimized (quantized) key scoring to reduce memory traffic in attention-like retrieval.
- **[vAttention](https://arxiv.org/abs/2510.05688)**: Provides verified sparse attention with statistical error guarantees under sampling.
- **[IceFormer](https://arxiv.org/abs/2405.02842)**: Accelerates long-sequence inference by casting attention’s top interactions as nearest-neighbor search (primarily CPU-focused).

- **[Query-Key Normalization (QK-Norm)](https://arxiv.org/abs/2010.04245)**: Normalizes queries/keys to cosine similarity to stabilize training, illustrating that Q/K geometry strongly affects attention behavior.
- **[DoPE](https://arxiv.org/abs/2511.09146)**: Denoises rotary position embeddings to reduce pathological attention behaviors in long contexts.
- **[WK, WV is probably all you need](https://arxiv.org/abs/2510.23912)**: Argues that query projections can be redundant under certain conditions, reinforcing that attention has substantial reparameterization freedom.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Static sparse patterns | Fixed local/global/random sparsity | Longformer, BigBird, Sparse Transformer | LM perplexity, downstream tasks | Often needs retraining / may miss content-dependent long-range links |
| Hashing / clustering for sparse attention | Data-dependent routing via ANN primitives | Reformer, Routing Transformer, LHA | LM perplexity, LRA | Q–K distribution mismatch; bucket imbalance |
| Retrieval-based KV indexing | Use vector search / ANN indices on KV cache | RetrievalAttention, Saap | Long-context QA / retrieval tasks | ANN effectiveness under Q–K OOD; index build overhead |
| Linear/kernel attention | Approximate softmax with kernels/features | Performer, Linformer, Nyströmformer | LM perplexity, LRA | Approximation error; may degrade on some tasks |
| Query/key stabilization | Normalize or denoise Q/K to improve attention behavior | QK-Norm, DoPE | Training stability; long-context robustness | Often targets training stability, not routing efficiency |

### Closest Prior Work

- **Saap** ([Saap](./references/Inference-time-sparse-attention-with-asymmetric-indexing/meta/meta_info.txt)) is the closest prior work: it uses k-means on keys and a trained MLP to route queries, explicitly motivated by Q–K OOD statistics. Our proposal targets the same routing objective but replaces the learned MLP with a closed-form gauge-whitening transform.
- **Tactic** ([Tactic](../../papers/paper_summaries/Tactic Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs.md)) is closely related in *deployment goal* (training-free sparse attention), but differs mechanistically: it fits an attention-score distribution to adapt token budgets, whereas we test a more targeted mechanistic hypothesis about **Q–K representation alignment** under dot-product-preserving transforms.
- **LHA** ([LHA](./references/Sparse-Attention-with-Learning-to-Hash/meta/meta_info.txt)) also addresses Q–K mismatch, but does so by learning separate hash functions, not by a dot-product-preserving closed-form transform.
- **RetrievalAttention** ([RetrievalAttention](./references/RetrievalAttention-Accelerating-Long-Context-LLM-Inference-via-Vector-Retrieval/meta/meta_info.txt)) addresses Q–K OOD by attention-aware index construction, but still relies on ANN machinery and per-context adaptation.
- **Maximal Gauge Symmetry** ([Gauge Symmetry](./references/Maximal-Gauge-Symmetry-in-Transformer-Architectures/meta/meta_info.txt)) provides the mathematical invariance \((W_Q,W_K)\to(W_QA, W_K(A^{-1})^\top)\), but does not apply it to routing or sparse attention.

**Novelty Kill Search Summary:** Searched for combinations of “attention gauge transformation qR kR^{-T} + sparse/approximate attention routing”, “dot-product preserving transform queries keys R^{-T} attention routing”, “gauge fixing transformer attention efficient sparse attention”, and checked the known related families (Saap, RetrievalAttention, LHA, Reformer, Routing Transformer). No prior work found that uses a dot-product-preserving closed-form gauge/second-moment transform to replace Saap-style learned query routing as of 2026-02-19 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Saap | k-means on keys + learned MLP query router for bucket selection | Requires per-head offline training (~25 min/head reported) | Replace MLP with closed-form gauge whitening | If Q–K mismatch is mostly linear/geometric, routing should be training-free |
| LHA | Learns separate query/key hash functions | Requires learning; not focused on inference-time KV-cache routing | Use closed-form transform; keep dot-product invariance | Lower deployment complexity; fewer knobs |
| RetrievalAttention | Builds ANN index and uses attention-aware vector search | Index build overhead; CPU/GPU split complexity | Stay within k-means bucket routing; remove learned router | Simpler hardware story and faster adaptation |
| Reformer / LSH attention | Randomized hashing buckets for attention | Assumes shared Q/K space; can be brittle | Use distribution-matched routing space | Better bucket balance without learning |
| Gauge symmetry theory | Characterizes invariances of attention parameters | Not connected to sparse attention routing | Apply invariance to design routing-friendly basis | Turns theory into an actionable deployment knob |

---

## Experiments

### Experimental Setup

**Goal:** Decide whether Saap’s learned query router is necessary for high routing quality, or whether a training-free gauge-whitening transform suffices.

**Main conditions (≤3):**
1. **Saap-MLP routing** (closest baseline): de-RoPE k-means on keys + train a 2-layer MLP \(f_q\) exactly as in Saap (Sec. 4.2, 5.1).
2. **Symmetric k-means routing (no query model)**: de-RoPE k-means on keys, and route queries by top-\(\ell\) centroid similarity in the same de-RoPE space (this corresponds to Saap’s “K-means (no rope)” baseline in Sec. 5.2 Table 2).
3. **Gauge-Whitened Routing (Ours)**: compute \(R\) from uncentered second moments and route queries by top-\(\ell\) centroid similarity in the gauge-whitened \(q',k'\) space.

**Base Model(s):**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5 | 7B | https://huggingface.co/Qwen/Qwen2.5-7B | RoPE-based decoder-only LM; open weights; 131K context (per model card) |

(If a Llama-3.x long-context checkpoint is available in the verification environment, we will also test one head on it for closer comparability to Saap, but the primary plan uses Qwen2.5-7B for accessibility.)

**Training Data (for calibration / MLP training):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| PG19 | Long-form natural text for collecting Q/K statistics and training the Saap-style router | N/A (streamed) | https://huggingface.co/datasets/pg19 | Check HF dataset card |

**Evaluation head selection:**
- Primary: one **mid-layer attention head** (one query head) to keep the experiment minimal and reproducible.
- Queries used for training/eval are filtered to be **long-range** (distance > 2047) to match Saap’s setup (Saap Sec. 4.2).

**Resource Estimate** (rough, verification-focused):
- **Compute budget**: ~5–15 GPU-hours total.
  - Forward passes to collect de-RoPE Q/K statistics on a small number of long prompts.
  - MLP training for 1 head × 3 seeds (Saap reports ~25 minutes per head on one GPU; Saap Sec. 5.1).
  - Evaluation by computing attention weights for sampled queries (matrix multiply \(n_q\times n_k\) per prompt).
- **GPU memory**: 1×A100 80GB (or similar) should be sufficient (inference-only, single model, no backprop through the base model).
- **API usage**: None required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Routing quality on long-range queries | Sample queries/keys from long PG19 prompts at a fixed layer+head | Attention-mass recall@\(\ell\), selectivity@\(\ell\), (optional) attention-output MSE | train/test split over prompts | https://huggingface.co/datasets/pg19 | Custom (simple PyTorch) |

**Primary metric (decisive): Attention-mass recall@\(\ell\).**
For each query, compute the exact attention weights \(A\) over all keys (using the model’s true RoPE attention for that head), aggregate bucket masses \(y = A H_k\), and score
\[
\text{Recall@}\ell = \sum_{b\in\text{Top-}\ell\text{ routed buckets}} y_b.
\]
This matches the bucket-mass object Saap uses in its training loss (Saap Sec. 4.2).

### Main Results

#### Results Table

(All numbers TBD; they will be produced by verification runs.)

| Method | Base Model | Metric: Attention-mass recall@\(\ell\) (mean±std) | Selectivity@\(\ell\) (mean±std) | Source | Notes |
|---|---|---:|---:|---|---|
| Saap-MLP routing | Qwen2.5-7B | TBD | TBD | - | 3 seeds for MLP training |
| Symmetric k-means routing | Qwen2.5-7B | TBD | TBD | - | No query router; route queries by centroid similarity in de-RoPE space |
| **Gauge-Whitened Routing (Ours)** | Qwen2.5-7B | TBD | TBD | - | Closed-form \(R\) from second moments |

### Optional Ablation (diagnostic only)

| Variant | What’s changed | Expected finding |
|---|---|---|
| Whitening-only routing (optional) | Independently whiten queries and keys (no dot-product-preserving coupling), then route by centroid similarity | If our gains come purely from generic whitening, this may match ours; if gauge coupling matters, it should underperform |

### Experimental Rigor

- **Seeds**: Train Saap-MLP routing with \(\ge 3\) random seeds (different init + data order). Routing evaluation itself is deterministic given a trained router.
- **Sanity checks**:
  - Random bucket routing at the same selectivity should have low attention-mass recall.
  - Full routing (\(\ell=C\)) should yield recall=1.
- **Confounders**:
  - K-means stochasticity: fix k-means seeds across conditions or rerun k-means with matched seeds.
  - Prompt distribution shift: train/evaluate on disjoint prompt sets; report per-prompt variance.

---

## Success Criteria

**Hypothesis**: Gauge-Whitened Routing matches Saap-MLP routing on attention-mass recall@\(\ell\) at comparable selectivity, indicating that much of the routing benefit comes from removable linear/geometric mismatch rather than non-linear classification.

**Decision Rule**:
- **Proceed**: Ours is within **1–2% absolute** recall of Saap-MLP (mean over test prompts) at the same \(\ell\), **and** Ours improves over symmetric k-means by **≥3% absolute** recall (non-overlapping std over 3 MLP seeds for Saap-MLP).
- **Pivot**: Ours ties symmetric k-means (within error bars) and both are close to Saap-MLP. Conclusion: training-free symmetric routing works, but gauge whitening is not necessary.
- **Refute**: Ours underperforms Saap-MLP by **>5% absolute** recall or closes **<50% of the gap** between symmetric k-means and Saap-MLP. Conclusion: non-linear learned routing is likely essential (at least for this head).

---

## Impact Statement

If successful, this work would provide a deployment-friendly alternative to learned query routers for inference-time sparse attention: implementers of Saap-style KV-cache indexing could replace per-head offline MLP training with a closed-form gauge-whitening computation, reducing setup cost and simplifying generalization to new base models.

If unsuccessful, the negative result would still be decision-changing: it would suggest that the key-query OOD problem in sparse attention routing cannot generally be solved by second-order moment alignment alone, motivating investment in learned routing or richer statistics.

---

## References

- [Inference-time sparse attention with asymmetric indexing](./references/Inference-time-sparse-attention-with-asymmetric-indexing/meta/meta_info.txt) - Mazare et al., 2025
- [Maximal Gauge Symmetry in Transformer Architectures](./references/Maximal-Gauge-Symmetry-in-Transformer-Architectures/meta/meta_info.txt) - Wang & Wang, 2026
- [RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval](./references/RetrievalAttention-Accelerating-Long-Context-LLM-Inference-via-Vector-Retrieval/meta/meta_info.txt) - Liu et al., 2024
- [Sparse Attention with Learning to Hash](./references/Sparse-Attention-with-Learning-to-Hash/meta/meta_info.txt) - Sun et al., 2022
- [Efficient Content-Based Sparse Attention with Routing Transformers](./references/Efficient-Content-Based-Sparse-Attention-with-Routing-Transformers/meta/meta_info.txt) - Roy et al., 2020
- [Reformer: The Efficient Transformer](./references/Reformer-The-Efficient-Transformer/meta/meta_info.txt) - Kitaev et al., 2020
- [Rethinking Attention with Performers](./references/Rethinking-Attention-with-Performers/meta/meta_info.txt) - Choromanski et al., 2020

- [FlashAttention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao et al., 2023
- [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/) - Dao et al., 2023
- [Longformer](https://arxiv.org/abs/2004.05150) - Beltagy et al., 2020
- [BigBird](https://arxiv.org/abs/2007.14062) - Zaheer et al., 2020
- [Sparse Transformer](https://arxiv.org/abs/1904.10509) - Child et al., 2019
- [Linformer](https://arxiv.org/abs/2006.04768) - Wang et al., 2020
- [Nyströmformer](https://arxiv.org/abs/2102.03902) - Xiong et al., 2021
- [StreamingLLM](https://arxiv.org/abs/2309.17453) - Xiao et al., 2023
- [HyperAttention](https://arxiv.org/abs/2310.05869) - Han et al., 2023
- [MagicPIG](https://arxiv.org/abs/2410.16179) - Anonymous, 2024
- [LOOKAT](https://arxiv.org/abs/2601.10155) - Anonymous, 2026
- [vAttention](https://arxiv.org/abs/2510.05688) - Anonymous, 2025
- [IceFormer](https://arxiv.org/abs/2405.02842) - Anonymous, 2024
- [Tactic](../../papers/paper_summaries/Tactic Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs.md) - Zhu et al., 2025
- [QK-Norm](https://arxiv.org/abs/2010.04245) - Henry et al., 2020
- [DoPE](https://arxiv.org/abs/2511.09146) - Anonymous, 2025
- [WQ redundancy](https://arxiv.org/abs/2510.23912) - Karbevski & Mijoski, 2024
