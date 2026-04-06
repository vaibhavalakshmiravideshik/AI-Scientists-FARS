# untitled

# Query-Conditioned Marginals for Cached Optimal-Transport Context Compression

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Automation constraint**: Fully automated evaluation only (no human preference judgments)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used for document question answering (QA) and retrieval-augmented generation (RAG), where the model must read long context passages before answering a short question. Long contexts are expensive: both attention compute and the key/value (KV) cache used during decoding grow with sequence length.

A common mitigation is **context compression**: replace a long context with a shorter representation that preserves the information needed for downstream generation. In **soft compression**, the context is condensed into a small number of continuous vectors (sometimes called “soft tokens”), which are then fed to the LLM.

A key practical axis is **query-aware vs. query-agnostic** compression:
- **Query-agnostic** compression can be cached and reused when multiple questions are asked about the same document, but it may waste capacity on question-irrelevant spans.
- **Query-aware** compression can preserve question-relevant information better, but often requires recompressing per query, reducing cache reuse.

### The Problem

**ComprExIT** (Ye et al., 2026; arXiv:2602.03784) proposes a soft-compression paradigm that operates on **frozen LLM hidden states** rather than training the LLM itself as a compressor. It (i) aggregates multi-layer hidden states into **token anchors**, then (ii) aggregates anchors into **compression slots** by solving an entropy-regularized **optimal transport (OT)** problem with Sinkhorn iterations.

ComprExIT explicitly trains under a **question-unaware** setting (see [ComprExIT “Training Method”](./references/Context-Compression-via-Explicit-Information-Transmission/sections/Training%20Method.md)). In the QA setup, this implies a deployment trade-off:
- The compressor can be run **once per context** (cacheable), but
- The OT allocation may not focus the limited compression budget on spans that matter for a specific question.

Recent work (e.g., **[OSCAR](https://arxiv.org/abs/2504.07109)**, **[ATACompressor](https://arxiv.org/abs/2602.03226)**) already studies query-dependent soft compression, but typically via new learned architectures and training pipelines.

The question we focus on is narrower and retrofit-oriented:

> Can we make a *trained*, query-agnostic OT compressor query-adaptive **at inference time**, without retraining and without adding parameters?

### Key Insight and Hypothesis

**Key insight:** In Sinkhorn OT, changing only the **marginal constraints** can substantially change the transport plan \(\Pi\) while keeping the underlying utility structure intact. For ComprExIT, this suggests a minimal query-aware intervention: bias the sender-side capacity toward anchors that are similar to the question.

**Hypothesis (H1):** Reweighting the **sender marginal** (token capacity) of ComprExIT’s OT solve using query–anchor similarity at inference time improves QA accuracy at a fixed compression ratio, because it shifts limited compression bandwidth toward question-relevant anchors while retaining OT’s global coordination and locality bias.

**Most likely failure mode:** A simple query-similarity **TopK** hard selection baseline at the same token budget may match or exceed any query-conditioned OT variant, implying OT coordination is unnecessary once the query is used.

---

## Proposed Approach

### Overview

We propose **QCap-OT** (“Query-Conditioned Capacity OT”): an inference-only modification to ComprExIT that conditions only the **sender capacity** \(\rho\) in the OT problem on the query, while keeping utility \(U\), receivers, segmenting, and Sinkhorn iterations unchanged.

This is designed to be:
- **Zero-new-parameter** (uses existing projections from ComprExIT)
- **No retraining** (same trained checkpoint)
- **Cache-friendly** (context anchors can still be computed once per document)

### Method Details

**Base compressor (ComprExIT).** Given context anchors \(h_t\) for \(t\in\{1..N\}\) and receivers \(r_k\) for \(k\in\{1..K\}\), ComprExIT defines (see [OT equations](./references/Context-Compression-via-Explicit-Information-Transmission/sections/%2B.md)):
- Utility: \(U_{t,k}=\cos(W_u h_t, W_u r_k)\)
- Sender capacity: \(\rho_t = \mathrm{softmax}(W_\rho h_t)\)
- Receiver capacity: \(\rho_k = 1/K\)

Then it solves an entropy-regularized OT problem (Sinkhorn) to obtain \(\Pi\) and aggregates anchors into compression tokens.

**Query-conditioned sender marginal (ours).**

1. **Query embedding.** Compute a query embedding \(q\) as the mean of frozen hidden states for the question tokens, projected with the same \(W_u\) used in ComprExIT:
   \[
   q_u = \mathrm{norm}(W_u q).
   \]

2. **Query–anchor similarity.** For each context anchor \(h_t\), compute
   \[
   s_t = \cos(\mathrm{norm}(W_u h_t),\ q_u).
   \]

3. **Capacity reweighting (inference only).** Reweight sender capacities:
   \[
   \tilde{\rho}_t(q) \propto \rho_t\,\exp(\beta s_t),\quad \sum_t \tilde{\rho}_t(q)=1.
   \]

4. **OT solve.** Run the same Sinkhorn solver as ComprExIT using \(\tilde{\rho}(q)\) as the sender marginal.

**QueryTopK baseline (hard selection).** Using the same similarity scores \(s_t\), select the top \(K\) anchors and pass them through the same projection/alignment layers as ComprExIT. This isolates whether OT adds value beyond query similarity.

### Key Innovations

- **Query-conditioned marginals as a retrofit mechanism**: add query adaptivity by changing only OT marginals, without learning a new compressor.
- **Cache-preserving query adaptation**: keeps context-only anchor computation intact; only a lightweight per-query reweighting + OT solve is added.
- **Self-critical baseline**: directly compares to QueryTopK to test whether OT coordination is redundant.

---

## Related Work

### Field Overview

Context compression spans (i) **hard selection/pruning** (drop tokens/spans) and (ii) **soft compression** (map many tokens to a small set of vectors). Hard methods are often cheap and interpretable but can be lossy at high compression ratios (e.g., **[Selective Context](https://arxiv.org/abs/2310.06201)**, **[LLMLingua](https://arxiv.org/abs/2310.05736)**). Soft methods can in principle preserve more information per compressed token but require learned compressors and careful training (e.g., **[ICAE](https://arxiv.org/abs/2307.06945)**, **[Activation Beacon](https://arxiv.org/abs/2401.03462)**).

Another axis is **query-aware vs query-agnostic** compression. Query-aware compression can focus on question-relevant content (e.g., **[OSCAR](https://arxiv.org/abs/2504.07109)**) but sacrifices cache reuse; query-agnostic compression supports reuse across multiple queries per document but may retain irrelevant content (e.g., query-agnostic KV compression like **[KVzip](https://arxiv.org/abs/2505.23416)**).

Our proposal sits at the intersection: we start from a **query-agnostic, cacheable** soft compressor (**[ComprExIT](./references/Context-Compression-via-Explicit-Information-Transmission/meta/meta_info.txt)**) and test whether a minimal, inference-only query adaptation is sufficient to improve QA accuracy.

### Related Papers

(*Each item is one sentence describing what it does and why it matters here.*)

- **[ComprExIT](./references/Context-Compression-via-Explicit-Information-Transmission/meta/meta_info.txt)**: Soft compression over frozen hidden states using Sinkhorn OT for globally coordinated allocation; our starting point.
- **[OSCAR](https://arxiv.org/abs/2504.07109)**: Online query-dependent soft compression for RAG via a learned compressor model; strongest recent query-aware soft-compression reference point.
- **[ATACompressor](https://arxiv.org/abs/2602.03226)**: Task-aware selective encoding plus adaptive token allocation; shows learned query-aware allocation can yield large gains at high compression ratios.
- **[DAST](https://arxiv.org/abs/2502.11493)**: Dynamically allocates soft tokens across chunks based on importance signals; closely related to “allocation” as the key lever.
- **[ICAE](https://arxiv.org/abs/2307.06945)**: In-context autoencoder that compresses context into memory slots via a trainable encoder and frozen decoder; a canonical soft-compression baseline family.
- **[500×Compressor](https://arxiv.org/abs/2408.03094)**: Improves ICAE-style compression by passing KV states for compressed tokens; relevant because ComprExIT compares against it on MRQA.
- **[Activation Beacon](https://arxiv.org/abs/2401.03462)**: Compresses long contexts via beacon tokens and activation-level compression; strong soft-compression baseline in the MRQA setting.
- **[Gist Tokens](https://arxiv.org/abs/2304.08467)**: Learns to compress prompts into a small number of gist tokens by modifying attention masks; foundational “soft token bottleneck” idea.
- **[AutoCompressors](https://arxiv.org/abs/2305.14788)**: Adapts LMs to produce cached summary vectors for long-context modeling; relevant as an alternative cacheable soft compression paradigm.
- **[xRAG](https://arxiv.org/abs/2405.13792)**: Treats retrieval embeddings as a modality and compresses retrieved documents to extreme token budgets; relevant for “one-token” soft compression in RAG.
- **[PISCO](https://arxiv.org/abs/2501.16075)**: Distillation-based RAG document compression that achieves high compression with low loss; relevant for query-agnostic soft compression in RAG pipelines.
- **[Provence](https://arxiv.org/abs/2501.16214)**: Efficient query-aware hard context pruning integrated with reranking; relevant hard-compression baseline family.
- **[EXIT](https://aclanthology.org/2025.findings-acl.253.pdf)**: Parallel context-aware sentence selection for RAG; relevant as a fast query-aware extractive compressor.
- **[Selective Context](https://arxiv.org/abs/2310.06201)**: Self-information-based pruning to reduce prompt length with minimal quality loss; representative hard compression baseline.
- **[LLMLingua](https://arxiv.org/abs/2310.05736)**: Coarse-to-fine perplexity-based prompt compression for black-box LLMs; popular hard compression baseline.
- **[LLMLingua-2](https://arxiv.org/abs/2403.12968)**: Token-level keep/drop classifier distilled from GPT-4 for task-agnostic prompt compression; strong hard compression baseline.
- **[EFPC](https://arxiv.org/abs/2503.07956)**: Efficient and flexible prompt compression that unifies task-aware and task-agnostic modes; relevant hard-compression baseline.
- **[Evaluator Heads (EHPC)](./references/Efficient-Prompt-Compression-with-Evaluator-Heads-for-Long-Context-Transformer-Inference/meta/meta_info.txt)**: Training-free pruning via specialized evaluator heads; relevant as a strong non-learning baseline class.
- **[SCOPE](./references/SCOPE-A-Generative-Approach-for-LLM-Prompt-Compression/meta/meta_info.txt)**: Generative chunking/summarization prompt compression; representative non-selection compression approach.
- **[KV-Distill](./references/kv-distill-Nearly-Lossless-Learnable-Context-Compression-for-LLMs/meta/meta_info.txt)**: Learnable KV cache compression aiming for near-lossless reconstruction; relevant for “compress then reuse across queries” motivation.
- **[KVzip](https://arxiv.org/abs/2505.23416)**: Query-agnostic KV cache compression via context reconstruction to support multi-query reuse; closely aligned with our cacheability motivation.
- **[UniGist](https://arxiv.org/abs/2509.15763)**: Hardware-aligned long-context compression via sparse gist layouts and custom kernels; relevant for the broader long-context compression landscape.
- **[SAC (Contextual Semantic Anchors)](https://arxiv.org/abs/2510.08907)**: Argues autoencoding is unnecessary and compresses via anchor-token selection plus architectural modifications; relevant alternative “allocation/selection” mechanism.
- **[ACON](./references/Acon-Optimizing-Context-Compression-for-Long-horizon-LLM-Agents/meta/meta_info.txt)**: Studies context compression for long-horizon agents and optimization trade-offs; motivates downstream stakes beyond QA.
- **[Lost in the Middle](https://arxiv.org/abs/2307.03172)**: Diagnoses position-dependent failures in long-context usage; motivates allocation-aware compression.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Typical evaluation | Known limitations |
|---|---|---|---|---|
| Soft compression (LLM-as-compressor) | Train special tokens that absorb info via self-attention | ICAE, 500×Compressor, Activation Beacon, Gist Tokens | MRQA, LongBench, NIAH | Representation drift; redundant allocation; training complexity |
| Soft compression (frozen-state / external module) | Compress frozen hidden states with lightweight modules | ComprExIT | MRQA | Often query-agnostic; may miss question-specific details |
| Query-aware soft compression | Condition compressor on (query, context) | OSCAR, ATACompressor | RAG QA suites; HotpotQA/MSMARCO/SQuAD | Requires per-query compression; reduced cache reuse |
| Hard compression (pruning/extraction) | Select tokens/sentences/spans to keep | LLMLingua, Selective Context, EXIT, Provence | LongBench; RAG QA | Discrete and potentially lossy; limited at very high compression |
| KV cache compression | Evict/quantize KV states to reduce decoding memory | KVzip, KV-Distill, H2O/SnapKV/PyramidKV/DuoAttention (surveyed in KVzip) | SQuAD, GSM8K, SCBench, NIAH | Query-aware eviction hurts multi-query reuse; methods are cache-level not “soft tokens” |

### Closest Prior Work

1. **ComprExIT** (**[paper](./references/Context-Compression-via-Explicit-Information-Transmission/meta/meta_info.txt)**): Solves allocation with OT but is trained question-unaware; QCap-OT changes only the sender marginal at inference to test whether query adaptivity is a missing ingredient.

2. **OSCAR** (**[arXiv:2504.07109](https://arxiv.org/abs/2504.07109)**): Provides query-dependent soft compression with a learned compressor; unlike QCap-OT it is not a retrofit for an existing query-agnostic compressor checkpoint and typically requires a different RAG training setup.

3. **ATACompressor** (**[arXiv:2602.03226](https://arxiv.org/abs/2602.03226)**): Learns a controller to adapt token allocation; QCap-OT instead tests whether a single inference-time scalar reweighting of OT marginals yields measurable gains without learning a controller.

4. **DAST** (**[arXiv:2502.11493](https://arxiv.org/abs/2502.11493)**): Shows that dynamic allocation matters; QCap-OT is a minimal “allocation” intervention specifically for OT-based compression.

### Comparison Table

| Related work | What it does | Key limitation for our goal | What we change | Why ours might win |
|---|---|---|---|---|
| ComprExIT | OT-based soft compression over frozen states | Question-unaware allocation | Condition OT sender marginal on query | Adds query adaptivity while preserving cacheable context encoding |
| OSCAR | Learned query-dependent online soft compression | Requires new compressor training; per-query compression | No new training; reuse ComprExIT | Retrofit path when a ComprExIT-style checkpoint exists |
| ATACompressor | Learned task-aware compression + adaptive allocation | Requires relevance annotations / training controller | Inference-only reweighting | Zero-parameter, minimal change test of “allocation is enough” |
| QueryTopK (baseline) | Hard select anchors by query similarity | No global coordination; can be noisy | Use OT to coordinate under constraints | OT may better preserve multi-hop evidence by balanced allocation |

---

## Experiments

### Experimental Setup

**Implementation note:** ComprExIT does not provide a public code release at the time of writing (paper states “code will be open-sourced soon”), so the Verification module may need to re-implement the method from the paper description.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3.2-1B-Base | 1B | https://huggingface.co/meta-llama | Frozen backbone as in ComprExIT |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| SlimPajama | Next-token prediction (NTP) pretraining of compression module | 1B tokens (sample) | https://huggingface.co/datasets/cerebras/SlimPajama-627B | ODC-By (verify) |
| MRQA 2019 | Supervised fine-tuning (SFT) on QA | 6 in-domain datasets | https://huggingface.co/datasets/mrqa | verify |

**Other Resources (if applicable):**
- None.

**Resource Estimate**:
- **Compute budget**: 200–450 GPU-hours total
  - NTP on 1B tokens dominates (frozen 1B backbone; gradients only for ~1% extra parameters)
  - SFT on MRQA + evaluation + 3-seed SFT restarts expected to fit in remaining budget
- **GPU memory**: ≤80GB (1×A100 80GB sufficient for BF16 forward + small-module backward at seq_len=512)
- **Uncertainty**: ComprExIT does not report hardware/runtime; this is a conservative estimate intended to stay well under the 768 GPU-hour cap.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MRQA SQuAD | Extractive QA (answer is a span in context) | EM, F1 | MRQA dev/test (match ComprExIT) | https://huggingface.co/datasets/mrqa | MRQA official / HuggingFace `evaluate` |
| MRQA HotpotQA | Multi-hop QA requiring multiple supporting facts | EM, F1 | MRQA dev/test (match ComprExIT) | https://huggingface.co/datasets/mrqa | MRQA official / HuggingFace `evaluate` |

**Metric notes:** EM (Exact Match) is the fraction of predictions that exactly match a gold answer string; F1 is token-overlap F1 between prediction and gold.

**Evaluation Scripts:**
- Use MRQA evaluation code if available; otherwise use the standard HuggingFace MRQA + `evaluate` implementation.

### Main Results

#### Results Table

Baseline numbers are copied from ComprExIT Table 1 and stored (with provenance) in `./references/ComprExIT-Table1-SQuAD-extracted.md`.

| Method | Base Model | Benchmark | EM | F1 | Source | Notes |
|---|---|---|---:|---:|---|---|
| Prompt tuning [w/ context] | Llama-3.2-1B | SQuAD | 71.89 | 81.09 | ComprExIT Table 1 | Uncompressed reference |
| ICAE | Llama-3.2-1B | SQuAD | 36.84 | 50.21 | ComprExIT Table 1 | Published |
| 500×Compressor | Llama-3.2-1B | SQuAD | 48.73 | 62.41 | ComprExIT Table 1 | Published |
| Activation Beacon | Llama-3.2-1B | SQuAD | 52.15 | 66.28 | ComprExIT Table 1 | Published |
| ComprExIT | Llama-3.2-1B | SQuAD | 68.42 | 79.15 | ComprExIT Table 1 | Published |
| QueryTopK (hard) | Llama-3.2-1B | SQuAD | **TBD** | **TBD** | - | Needs re-run |
| **QCap-OT (ours)** | Llama-3.2-1B | SQuAD | **TBD** | **TBD** | - | To be verified |
| Prompt tuning [w/ context] | Llama-3.2-1B | HotpotQA | 53.60 | 69.49 | ComprExIT Table 1 | Uncompressed reference |
| ICAE | Llama-3.2-1B | HotpotQA | 40.67 | 57.04 | ComprExIT Table 1 | Published |
| 500×Compressor | Llama-3.2-1B | HotpotQA | 45.18 | 61.89 | ComprExIT Table 1 | Published |
| Activation Beacon | Llama-3.2-1B | HotpotQA | 42.35 | 58.91 | ComprExIT Table 1 | Published |
| ComprExIT | Llama-3.2-1B | HotpotQA | 50.21 | 66.73 | ComprExIT Table 1 | Published |
| QueryTopK (hard) | Llama-3.2-1B | HotpotQA | **TBD** | **TBD** | - | Needs re-run |
| **QCap-OT (ours)** | Llama-3.2-1B | HotpotQA | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| \(\beta=0\) | No query reweighting | Identical to ComprExIT (sanity check) |
| RandomTopK | Select \(K\) anchors uniformly at random | Worse than QueryTopK and ComprExIT |
| \(W_u\) ablation | Use cosine in anchor space without \(W_u\) | Potentially weaker similarity signal |

### Analysis (Optional)

- **Answer-span mass (SQuAD)**: Measure how much OT mass \(\Pi\) lands on tokens overlapping the gold answer span; QCap-OT should increase this vs ComprExIT.
- **Multi-query cacheability demo**: For a fixed context with multiple questions, cache anchors once and report per-query overhead for QCap-OT vs recompress-per-query methods (timing only; accuracy already measured above).

---

## Success Criteria

**Criterion 1: Query-conditioned OT improves over query-unaware OT on multi-hop QA**
- Hypothesis: On HotpotQA, QCap-OT improves EM and/or F1 over ComprExIT at the same compression ratio.
- Validation: The mean HotpotQA EM delta (QCap-OT − ComprExIT) over 3 SFT seeds is > 0, and its paired bootstrap 95% CI lower bound is > 0.

**Criterion 2: OT coordination adds value beyond query similarity**
- Hypothesis: QCap-OT outperforms QueryTopK at the same compressed-token budget.
- Validation: On HotpotQA, QCap-OT’s mean EM is higher than QueryTopK’s mean EM (same seeds/checkpoints).

**Refutation / pivot rule:** If QueryTopK matches or exceeds QCap-OT on HotpotQA, we conclude that in this setting OT coordination provides no added value beyond query similarity, and the retrofit is not worth pursuing further.

---

## Impact Statement

If successful, this provides a practical retrofit that makes an OT-based, cacheable soft compressor (ComprExIT) query-adaptive with zero new parameters and no retraining. This could improve QA quality in systems that must serve multiple questions per document while keeping long-context latency and KV-cache memory bounded.

---

## References

- [Context Compression via Explicit Information Transmission (ComprExIT)](./references/Context-Compression-via-Explicit-Information-Transmission/meta/meta_info.txt) - Ye et al., 2026
- [OSCAR: Online Soft Compression And Reranking](https://arxiv.org/abs/2504.07109) - Louis et al., 2025
- [ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLMs](https://arxiv.org/abs/2602.03226) - Li et al., 2026
- [DAST: Context-Aware Compression in LLMs via Dynamic Allocation of Soft Tokens](https://arxiv.org/abs/2502.11493) - Chen et al., 2025
- [In-context Autoencoder for Context Compression in a Large Language Model (ICAE)](https://arxiv.org/abs/2307.06945) - Ge et al., 2024
- [500×Compressor: Generalized Prompt Compression for Large Language Models](https://arxiv.org/abs/2408.03094) - Li et al., 2024
- [Long Context Compression with Activation Beacon](https://arxiv.org/abs/2401.03462) - Zhang et al., 2024
- [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467) - Mu et al., 2023
- [Adapting Language Models to Compress Contexts (AutoCompressors)](https://arxiv.org/abs/2305.14788) - Chevalier et al., 2023
- [Selective Context: Compressing Context to Enhance Inference Efficiency of LLMs](https://arxiv.org/abs/2310.06201) - Li et al., 2023
- [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736) - Jiang et al., 2023
- [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) - Pan et al., 2024
- [EFPC: Towards Efficient and Flexible Prompt Compression](https://arxiv.org/abs/2503.07956) - Cao et al., 2025
- [Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference](./references/Efficient-Prompt-Compression-with-Evaluator-Heads-for-Long-Context-Transformer-Inference/meta/meta_info.txt) - Fei et al., 2025
- [SCOPE: A Generative Approach for LLM Prompt Compression](./references/SCOPE-A-Generative-Approach-for-LLM-Prompt-Compression/meta/meta_info.txt) - Zhang et al., 2025
- [KV-Distill: Nearly Lossless Learnable Context Compression for LLMs](./references/kv-distill-Nearly-Lossless-Learnable-Context-Compression-for-LLMs/meta/meta_info.txt) - Chari et al., 2025
- [KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](https://arxiv.org/abs/2505.23416) - Kim et al., 2025
- [PISCO: Pretty Simple Compression for Retrieval-Augmented Generation](https://arxiv.org/abs/2501.16075) - Louis et al., 2025
- [Provence: Efficient and Robust Context Pruning for Retrieval-Augmented Generation](https://arxiv.org/abs/2501.16214) - Chirkova et al., 2025
- [EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](https://aclanthology.org/2025.findings-acl.253.pdf) - Hwang et al., 2025
- [UniGist: Towards General and Hardware-aligned Sequence-level Long Context Compression](https://arxiv.org/abs/2509.15763) - Deng et al., 2025
- [SAC: Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors](https://arxiv.org/abs/2510.08907) - Zhao et al., 2025
- [ACON: Optimizing Context Compression for Long-horizon LLM Agents](./references/Acon-Optimizing-Context-Compression-for-Long-horizon-LLM-Agents/meta/meta_info.txt) - Kang et al., 2025
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - Liu et al., 2023
