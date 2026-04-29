# untitled

# Distilling Gradient-Guided Bidirectional Embedding Teachers into Streaming-Causal Students

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Text embedding models map a text string to a fixed-dimensional vector used for semantic search, retrieval-augmented generation (RAG), clustering, and recommendation. While many high-quality embedders are encoder-style models with bidirectional attention, decoder-only LLMs are attractive in production because they are widely available and support **incremental inference with a key–value cache (KV cache)**: when new tokens are appended, a causal model can reuse cached activations for the prefix and process only the new tokens.

A practical use case is **streaming / continually growing text** (chat sessions, user histories, agent traces, continuously edited documents). In these settings, the embedding needs to be updated frequently as the text grows. Bidirectional attention breaks KV-cache reuse because adding tokens can change earlier hidden states, so updating the embedding typically requires recomputing the entire sequence.

This recomputation cost is a concrete deployment concern in large-scale personalization systems. For example, **[PinnerFormer](https://arxiv.org/abs/2205.04507)** (Pinterest’s production user-representation model) describes a practical tension: updating a sequence-model user embedding after every action either requires recomputing from scratch (high compute) or maintaining stateful streaming infrastructure (high engineering cost), motivating batch embedding refreshes. Similarly, **[Incremental User Embedding Modeling for Personalized Text Classification](https://arxiv.org/abs/2202.06369)** motivates incremental user-history representations because user histories grow without bound and accessing the full history is impractical.

For an L-token history updated by appending Δ tokens, full recomputation processes ~L+Δ tokens per update, while a causal streaming update processes only Δ tokens. At L=8192 and Δ=128 this is a ~64× token-processing gap per update, which directly translates into wall-clock and GPU-cost differences.

Recent work suggests that giving models more “global” context during embedding extraction can substantially improve quality:
- **Echo embeddings** repeat the input twice and embed the second copy so that per-token representations can incorporate information from the entire sentence, and they report large gains under mean pooling on MTEB (e.g., Mistral-7B zero-shot mean-pooling avg 45.88 → 55.07) **[Echo embeddings](./references/Repetition-Improves-Language-Model-Embeddings/meta/meta_info.txt)**.
- **GG-SM (Gradient-Guided Soft Masking)** proposes a stabilized transition from causal to bidirectional attention for representation learning in decoder-only LLMs, showing consistent (though modest) AUC improvements on **proprietary** Alipay user-embedding tasks (avg AUC 0.7709 → 0.7745 for Qwen2.5-0.5B-Instruct; not evaluated on MTEB) **[GG-SM](./references/How-Do-Decoder-Only-LLMs-Perceive-Users-Rethinking-Attention-Masking-for-User-Representation-Learning/meta/meta_info.txt)**.

### The Problem

Bidirectional attention and “global context” tricks can improve embeddings, but they complicate deployment for streaming updates:

- **Bidirectional inference** (including GG-SM’s best setting) requires recomputing the full sequence when new tokens arrive, because the hidden states (and thus keys/values) of the prefix change.
- **Inference-time repetition** (echo embeddings) increases sequence length and typically requires re-encoding the entire repeated prompt each update.

In practice, many systems want the **deployment properties of causal models** (KV-cache-compatible incremental updates) while getting as close as possible to **full-context embedding quality**.

### Key Insight and Hypothesis

**Key insight**: We can use a full-context (bidirectional) model as a **teacher during training**, but ship a **causal student** at inference. The teacher can be trained with GG-SM to reduce the causal→bidirectional transition instability; the student is trained with standard contrastive objectives plus an embedding-level distillation loss to match the teacher’s embedding space.

**Hypothesis**: On a public embedding evaluation suite, a causal student trained with distillation from a GG-SM bidirectional teacher will recover a large fraction of the teacher’s quality gain over a causal baseline, while preserving KV-cache streaming updates.

This could fail for several reasons: (i) the teacher’s advantage over a causal baseline may be small under realistic small-model training budgets; (ii) the student may be unable to approximate a bidirectional embedding geometry with a causal forward pass; (iii) inference-time baselines like echo embeddings may already close most of the gap at acceptable cost.

---

## Proposed Approach

### Overview

We propose a three-condition teacher–student experiment (one trained backbone family, one training dataset, one evaluation suite):

1. **Causal baseline (A)**: causal attention + contrastive embedding training.
2. **GG-SM bidirectional teacher (B)**: GG-SM schedule during training; **bidirectional attention at inference** for embedding extraction.
3. **Distilled causal student (C)**: causal attention + the same contrastive objective as (A), plus an embedding distillation loss to match (B).

The deployed model is (C), which supports KV-cache incremental updates.

### Method Details

#### Embedding extraction (teacher and student)
To make the causal/bidirectional difference relevant while remaining streaming-friendly, we use **mean pooling over token hidden states**:

- Given final-layer hidden states \(h_1,\dots,h_L\), define embedding \(e(x)\) as the L2-normalized mean over “content tokens” (exclude special tokens / padding):
  \[
  e(x)=\mathrm{norm}\Big(\frac{1}{|S|}\sum_{i\in S} h_i\Big).
  \]

Mean pooling is updateable in a streaming setting by maintaining a running sum of token vectors as new tokens are appended. Under causal attention, earlier token vectors do not change when new tokens arrive, so only the new tokens’ hidden states must be computed.

#### Teacher training: GG-SM
We implement GG-SM as specified in the paper’s “LLMs as Encoders: Training Recipe” section **[GG-SM](./references/How-Do-Decoder-Only-LLMs-Perceive-Users-Rethinking-Attention-Masking-for-User-Representation-Learning/meta/meta_info.txt)**:

- Define a soft attention mask \(M_{soft}(t)\) where for positions \(j>i\), \(M_{soft,ij}(t)=\log w_{ij}(t)\); for \(j\le i\), it is 0.
- Warmup (\(t<T_{warm}\)): \(w_{ij}(t)=\sigma(\|\nabla_{h_j}\mathcal{L}\|)\), where \(\sigma\) is sigmoid.
- Scheduler (\(t\ge T_{warm}\)): freeze gradient-derived weights at the end of warmup and linearly interpolate toward full bidirectionality: \(w_{ij}(t)=(1-\alpha_t)\sigma(\|\nabla_{h_j}\mathcal{L}_{warm}\|)+\alpha_t\), \(\alpha_t=(t-T_{warm})/(T_{total}-T_{warm})\).

**Downscaling choice**: The original GG-SM paper trains at very large scale (64×A100, 70k steps). For verification, we fix \(T_{warm}=0.1\,T_{total}\) (no tuning) and use a short LoRA fine-tuning schedule (details in Experiments).

#### Student training: contrastive + distillation
We train the causal student with:

- **Contrastive loss**: InfoNCE with in-batch negatives (a contrastive objective that encourages matched text pairs to have higher cosine similarity than other texts in the batch), using the same batch construction across A/B/C.
- **Distillation loss**: MSE between teacher and student embeddings (both L2-normalized):
  \[
  \mathcal{L}_{distill}=\|e_S(x)-e_T(x)\|_2^2.
  \]

Final objective: \(\mathcal{L}=\mathcal{L}_{InfoNCE}+\lambda\,\mathcal{L}_{distill}\), with \(\lambda=1.0\) fixed (no tuning).

Teacher embeddings \(e_T(x)\) are computed from the frozen teacher (B). We precompute and cache them for the training set to avoid repeated teacher forward passes.

#### Streaming update protocol (deployed student)
For a stream split into chunks \(x=(x_1,\dots,x_K)\), we maintain:

- KV cache for causal forward passes.
- Running sum \(s=\sum_{i\in S} h_i\) and count \(|S|\) for mean pooling.

When a new chunk arrives, we forward only the new tokens (using KV cache), add their token vectors to \(s\), update \(|S|\), and output \(\mathrm{norm}(s/|S|)\). This requires no recomputation of the prefix.

### Key Innovations

- **Deployment-motivated supervision mismatch**: use bidirectional (full-context) attention only in the teacher to learn a high-quality embedding geometry, while keeping the shipped model strictly causal.
- **GG-SM as a stabilizer for bidirectional teachers**: GG-SM’s gradient-guided warmup targets the known instability of abruptly switching from causal to bidirectional masking.
- **Minimal decisive test**: A/B/C isolates whether distillation can preserve most of the teacher’s gain with causal inference.

---

## Related Work

### Field Overview

Embedding learning has a mature literature of encoder-style models trained with contrastive objectives (e.g., Sentence-BERT (SBERT), SimCSE, E5). More recently, decoder-only LLMs have been adapted into embedders, either by changing pooling/instruction formats or by modifying how they access global context.

Approaches for addressing causal-mask limitations in decoder-only embedders include:
- **Inference-time input manipulation** (e.g., repetition / echo embeddings) **[Echo embeddings](./references/Repetition-Improves-Language-Model-Embeddings/meta/meta_info.txt)**.
- **Mask modifications** that remove or relax the causal mask (e.g., LLM2Vec, NV-Embed) **[Causal2Vec](./references/Causal2Vec-Improving-Decoder-only-LLMs-as-Versatile-Embedding-Models/meta/meta_info.txt)**.
- **Internal state/KV manipulation** (e.g., KV-Embedding) **[KV-Embedding](./references/KV-Embedding-Training-free-Text-Embedding-via-Internal-KV-Re-routing-in-Decoder-only-LLMs/meta/meta_info.txt)**.
- **Auxiliary encoders / contextual tokens** (e.g., Causal2Vec) **[Causal2Vec](./references/Causal2Vec-Improving-Decoder-only-LLMs-as-Versatile-Embedding-Models/meta/meta_info.txt)**.
- **Training from scratch with soft masking schedules** (e.g., Conan-Embedding-v2) **[Conan-Embedding-v2](./references/Conan-Embedding-v2-Training-an-LLM-from-Scratch-for-Text-Embeddings/meta/meta_info.txt)**.

Our proposal focuses on **training-time distillation** to keep causal inference while borrowing full-context supervision.

### Related Papers

- **[How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning](./references/How-Do-Decoder-Only-LLMs-Perceive-Users-Rethinking-Attention-Masking-for-User-Representation-Learning/meta/meta_info.txt)**: Proposes GG-SM, a gradient-guided warmup + scheduler for causal→bidirectional transition in representation learning.
- **[Repetition Improves Language Model Embeddings](./references/Repetition-Improves-Language-Model-Embeddings/meta/meta_info.txt)**: Echo embeddings via input repetition; large MTEB gains under mean pooling; doubles inference tokens.
- **[KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs](./references/KV-Embedding-Training-free-Text-Embedding-via-Internal-KV-Re-routing-in-Decoder-only-LLMs/meta/meta_info.txt)**: Training-free internal KV rerouting to inject global context; strong gains on MTEB/LoCoV1.
- **[Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models](./references/Causal2Vec-Improving-Decoder-only-LLMs-as-Versatile-Embedding-Models/meta/meta_info.txt)**: Prepends a contextual token from a small bidirectional encoder; strong MTEB results without removing causal attention.
- **[Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings](./references/Conan-Embedding-v2-Training-an-LLM-from-Scratch-for-Text-Embeddings/meta/meta_info.txt)**: Trains a dedicated embedder from scratch; uses progressive soft masking schedules.
- **[Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](./references/Qwen3-Embedding-Advancing-Text-Embedding-and-Reranking-Through-Foundation-Models/meta/meta_info.txt)**: Strong causal embedding/reranking family; provides small-model baselines (0.6B+).
- **[EmbeddingGemma: Powerful and Lightweight Text Representations](./references/EmbeddingGemma-Powerful-and-Lightweight-Text-Representations/meta/meta_info.txt)**: Uses bidirectional encoder(-decoder) initialization and geometric distillation to train compact embedders.
- **[Causal Attention with Lookahead Keys](./references/Causal-Attention-with-Lookahead-Keys/meta/meta_info.txt)**: Updates causal keys as context grows while preserving autoregressive constraints; relevant for “more global” causal representations.

Additional foundational/standard references (not all are implemented here):
- **[Sentence-BERT](https://arxiv.org/abs/1908.10084)**: Siamese bi-encoder training with NLI supervision; foundational contrastive embedder.
- **[SimCSE](https://arxiv.org/abs/2104.08821)**: Contrastive sentence embeddings with dropout-based augmentation; widely used baseline.
- **[E5](https://arxiv.org/abs/2212.03533)**: Weakly supervised contrastive pretraining for retrieval embeddings.
- **[INSTRUCTOR](https://arxiv.org/abs/2212.09741)**: Instruction-conditioned embedding model trained across tasks.
- **[SGPT](https://arxiv.org/abs/2202.08904)**: Decoder-only embedding extraction with pooling/weighting strategies.
- **[LLM2Vec](https://arxiv.org/abs/2404.05961)**: Removes causal mask for bidirectional embeddings from decoder-only LLMs.
- **[NV-Embed](https://arxiv.org/abs/2405.17428)**: Techniques for training LLMs as generalist embedding models (latent attention pooling).
- **[GRITLM](https://arxiv.org/abs/2402.16852)**: Unified embedding + generation training.
- **[bge-en-icl](https://arxiv.org/abs/2402.17762)**: Few-shot embedding via in-context examples.
- **[PromptEOL](https://arxiv.org/abs/2408.06607)**: Prompting and end-of-list extraction for embeddings.
- **[SimTDE](./references/SimTDE-Simple-transformer-distillation-for-sentence-embeddings/meta/meta_info.txt)**: Distills sentence embedding models for efficiency; useful distillation baseline family.
- **[Gecko](https://arxiv.org/abs/2403.20327)**: Distills versatile embeddings from large LLMs via synthetic data.
- **[DistilBERT](https://arxiv.org/abs/1910.01108)**: Classic distillation approach for language models.
- **[MTEB](https://arxiv.org/abs/2210.07316)**: Massive Text Embedding Benchmark (standard evaluation suite).
- **[BEIR](https://arxiv.org/abs/2104.08663)**: Retrieval benchmark suite used widely for dense retrieval.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Input repetition | Repeat input to allow second-pass tokens to attend to “future” | Echo embeddings | MTEB | 2× inference tokens; re-encode per update |
| Mask schedule | Gradually transition causal→bidirectional during training | GG-SM, Conan-Embedding-v2 | proprietary, MTEB | Bidirectional inference not streamable |
| Mask removal | Remove causal mask in LLM for bidirectional attention | LLM2Vec, NV-Embed | MTEB | Breaks KV-cache streaming; potential pretrain mismatch |
| Internal KV manipulation | Inject global summary via KV rerouting | KV-Embedding | MTEB, LoCoV1 | Added complexity; training-free ceiling |
| Auxiliary encoder token | Add contextual token from a bidirectional encoder | Causal2Vec | MTEB | Requires extra encoder at inference |
| Distillation (this work) | Match causal student embeddings to full-context teacher embeddings | (this proposal) | MTEB subset, LoCoV1 | Depends on teacher gap existing |

### Closest Prior Work

1. **GG-SM** **[GG-SM](./references/How-Do-Decoder-Only-LLMs-Perceive-Users-Rethinking-Attention-Masking-for-User-Representation-Learning/meta/meta_info.txt)**: Improves representation learning by stabilizing causal→bidirectional transition, but assumes bidirectional inference.
2. **Echo embeddings** **[Echo embeddings](./references/Repetition-Improves-Language-Model-Embeddings/meta/meta_info.txt)**: Strong inference-time baseline that approximates bidirectionality with repetition; not streaming-friendly.
3. **Causal2Vec** **[Causal2Vec](./references/Causal2Vec-Improving-Decoder-only-LLMs-as-Versatile-Embedding-Models/meta/meta_info.txt)**: Keeps causal inference but injects bidirectional information through an auxiliary encoder token.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| GG-SM | Trains bidirectional attention via stabilized schedule | Bidirectional inference breaks streaming | Distill into causal student | Keep streaming while approaching teacher quality |
| Echo embeddings | Repeats input twice; embeds second copy | 2× tokens; re-encode per update | Distill teacher embeddings offline | Similar “full-context” quality without doubled inference |
| Causal2Vec | Adds contextual token from bi-encoder | Extra encoder at inference | Teacher-only bidirectionality | No extra inference component |
| KV-Embedding | KV rerouting to inject global summary | Engineering overhead; training-free ceiling | Training-based distillation | Potentially exceed training-free ceiling |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct | Small enough for 3-condition training within budget |

**Training Data (teacher + student):**

We use a public, easily downloadable sentence-embedding training set:

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| sentence-transformers/all-nli (SNLI+MultiNLI) | Contrastive embedding training (use entailment pairs as positives) | ~1.1M sentence pairs | https://huggingface.co/datasets/sentence-transformers/all-nli | SNLI: CC-BY-SA-4.0; MultiNLI: Apache-2.0 (source datasets). Used for research experiments; we do not redistribute the data. |

(If license constraints for a particular verifier environment are a concern, substitute an Apache-licensed retrieval corpus for training; the core method is unchanged.)

**Implementation details (fixed; no tuning):**
- LoRA fine-tuning (Low-Rank Adaptation; a parameter-efficient fine-tuning method) for all A/B/C with **r=16, α=16** (following echo-embeddings finetuning settings) **[Echo embeddings](./references/Repetition-Improves-Language-Model-Embeddings/meta/meta_info.txt)**.
- Sequence length 512; global batch 256; train for 10k steps (≈1.31B tokens).
- **Seed variance**: run 2 random seeds for A and C (the two streamable models). If the teacher B is expensive, run 1 seed for B; otherwise 2 seeds. Report mean±std on the MTEB-slice.
- Optimizer AdamW; LR 2e-4 with cosine decay; temperature τ=0.02 for InfoNCE.
- GG-SM warmup fraction 10% of steps.
- Distillation weight λ=1.0.

**Resource Estimate** (must fit ≤768 A100-GPU-hours):
- Training tokens per run: 10k steps × 256 batch × 512 tokens ≈ 1.31B tokens.
- Using a recent 0.5B LoRA throughput reference (~11.7k tok/s on A100-40GB for Qwen2.5-0.5B LoRA; Chronicals) gives ≈33 GPU-hours per run; using a conservative 3k tok/s gives ≈121 GPU-hours per run.
- Total for 3 training runs (A,B,C): ≈100–360 GPU-hours.
- Teacher embedding precomputation: one additional forward pass over the training set (estimate 10–40 GPU-hours depending on caching/throughput).
- Evaluation: ≤50 GPU-hours (small benchmark subset + retrieval scoring).
- **Total expected**: ≈160–450 GPU-hours.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MTEB (English subset; small slice) | General text embedding evaluation across task types | MTEB “main_score” per dataset; report mean over selected datasets | test | https://github.com/embeddings-benchmark/mteb | `mteb` |
| LoCoV1 (optional stress test) | Long-context retrieval benchmark for long documents | NDCG@10 | test | https://huggingface.co/datasets/hazyresearch/LoCoV1-Documents + https://huggingface.co/datasets/hazyresearch/LoCoV1-Queries | official HF loading + retrieval scoring |

**Chosen MTEB slice (fixed):** 6 datasets spanning different task types to keep evaluation cheap:
- Retrieval: ArguAna, SciFact
- STS: STSBenchmark
- Classification: AmazonCounterfactualClassification
- Pair classification: MRPC
- Clustering: RedditClustering

**Streaming efficiency microbenchmark (deployment property):**
- Prefix length 4096 tokens; append Δ=128 tokens for 20 updates.
- Measure wall-clock per update for:
  - causal student using KV cache (process only Δ)
  - bidirectional teacher recomputing full sequence (process ~4096+Δ each update)
- **Prefix-fidelity check (correctness)**: for the causal student, verify that the incremental-update embedding matches a full recomputation embedding at every update (cosine distance < 1e-6).

### Main Results

#### Results Table

(All rows are directly comparable: same base model family, same training data, same evaluation slice, same pooling.)

| Method | Base Model | Evaluation | Mean score (MTEB-slice) | LoCoV1 NDCG@10 (optional) | Source | Notes |
|---|---|---|---:|---:|---|---|
| A: causal contrastive | Qwen2.5-0.5B-Instruct | MTEB-slice | **TBD** | **TBD** | - | Baseline (streaming-compatible) |
| A+Echo (inference-only) | Qwen2.5-0.5B-Instruct | MTEB-slice | **TBD** | **TBD** | [Echo embeddings](./references/Repetition-Improves-Language-Model-Embeddings/meta/meta_info.txt) | Repeat input twice; embed 2nd copy (2× tokens; not streaming-friendly) |
| B: GG-SM bidirectional | Qwen2.5-0.5B-Instruct | MTEB-slice | **TBD** | **TBD** | - | Teacher (not streamable) |
| **C: distilled causal (ours)** | Qwen2.5-0.5B-Instruct | MTEB-slice | **TBD** | **TBD** | - | Student shipped for streaming |

#### Streaming efficiency

| Method | Update cost scaling | Measured update latency | Notes |
|---|---|---:|---|
| Causal student (C) | O(Δ) | **TBD** | KV-cache reuse + running mean pooling |
| Bidirectional teacher (B) | O(L+Δ) | **TBD** | Requires full recomputation per update |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| C w/o distill | Remove distillation loss (reduces to A) | Performance drops toward A |

### Analysis (Optional)

- **Early-stop premise check**: if the teacher gap (B−A) on the MTEB-slice is < 0.5 points (absolute mean score), stop and report a negative result (“bidirectional teacher provides too little additional signal at this scale/dataset”).

---

## Success Criteria

**Criterion 1: Teacher gap exists (premise validation)**
- Hypothesis: The bidirectional teacher B outperforms causal baseline A on the fixed MTEB-slice mean score.
- Validation / decision rule: If (B−A) < 0.5 points, stop and conclude that distillation is not worthwhile at this scale.

**Interpretation vs echo baseline**: If A+Echo ≥ B on the fixed MTEB-slice, treat this as evidence that a simple inference-time trick closes the teacher gap; in that case, the main value of (C) is efficiency (single-pass streaming vs 2× tokens), and the paper should emphasize the cost/quality trade-off rather than raw accuracy.

**Criterion 2: Distillation recovers most of the teacher gain**
- Hypothesis: The distilled causal student C recovers a large fraction of the teacher gain over A.
- Validation / decision rule: Compute gap-closure ratio \((C−A)/(B−A)\). If ≥0.7, treat as success; if <0.4, refute the approach; otherwise report as partial.

**Criterion 3: Streaming deployment property is preserved**
- Hypothesis: The student supports KV-cache incremental updates and is substantially faster than bidirectional recomputation for long prefixes.
- Validation: On the microbenchmark, causal update latency is at least 5× lower than bidirectional recomputation at prefix length 4096.

---

## Impact Statement

If successful, this work provides a simple recipe for training **streaming-compatible** text embedders: use a full-context teacher to learn a better embedding geometry, but deploy a causal student that supports frequent incremental updates for long, growing texts (session memory, continuously edited documents, long agent traces).

---

## References

- [How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning](./references/How-Do-Decoder-Only-LLMs-Perceive-Users-Rethinking-Attention-Masking-for-User-Representation-Learning/meta/meta_info.txt) - Yuan et al., 2026
- [Repetition Improves Language Model Embeddings](./references/Repetition-Improves-Language-Model-Embeddings/meta/meta_info.txt) - Springer et al., 2024
- [KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs](./references/KV-Embedding-Training-free-Text-Embedding-via-Internal-KV-Re-routing-in-Decoder-only-LLMs/meta/meta_info.txt) - arXiv:2601.01046
- [Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models](./references/Causal2Vec-Improving-Decoder-only-LLMs-as-Versatile-Embedding-Models/meta/meta_info.txt) - arXiv:2507.23386
- [Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings](./references/Conan-Embedding-v2-Training-an-LLM-from-Scratch-for-Text-Embeddings/meta/meta_info.txt) - arXiv:2509.12892
- [Causal Attention with Lookahead Keys](./references/Causal-Attention-with-Lookahead-Keys/meta/meta_info.txt) - arXiv:2509.07301
- [EmbeddingGemma: Powerful and Lightweight Text Representations](./references/EmbeddingGemma-Powerful-and-Lightweight-Text-Representations/meta/meta_info.txt) - arXiv:2509.20354
- [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](./references/Qwen3-Embedding-Advancing-Text-Embedding-and-Reranking-Through-Foundation-Models/meta/meta_info.txt) - arXiv:2506.05176
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Reimers & Gurevych, 2019
- [SimCSE](https://arxiv.org/abs/2104.08821) - Gao et al., 2021
- [E5](https://arxiv.org/abs/2212.03533) - Wang et al., 2022
- [INSTRUCTOR](https://arxiv.org/abs/2212.09741) - Su et al., 2022
- [SGPT](https://arxiv.org/abs/2202.08904) - Muennighoff et al., 2022
- [LLM2Vec](https://arxiv.org/abs/2404.05961) - BehnamGhader et al., 2024
- [NV-Embed](https://arxiv.org/abs/2405.17428) - Lee et al., 2025
- [GRITLM](https://arxiv.org/abs/2402.16852) - Muennighoff et al., 2024
- [Gecko](https://arxiv.org/abs/2403.20327) - Lee et al., 2024
- [DistilBERT](https://arxiv.org/abs/1910.01108) - Sanh et al., 2019
- [MTEB](https://arxiv.org/abs/2210.07316) - Muennighoff et al., 2023
- [BEIR](https://arxiv.org/abs/2104.08663) - Thakur et al., 2021
- [PinnerFormer: Sequence Modeling for User Representation at Pinterest](https://arxiv.org/abs/2205.04507) - Pancha et al., 2022
- [Incremental User Embedding Modeling for Personalized Text Classification](https://arxiv.org/abs/2202.06369) - Lian et al., 2022
- [Chronicals: A High-Performance Framework for LLM Fine-Tuning](https://arxiv.org/abs/2601.02609) - arXiv:2601.02609
