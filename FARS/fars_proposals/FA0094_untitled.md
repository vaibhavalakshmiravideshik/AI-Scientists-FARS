# untitled

# Soft-Masked SimCSE: Replacing MNTP Warmup in Decoder-Only → Text-Encoder Conversion

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Text embedding models map an input text string to a fixed-dimensional vector that supports semantic search, retrieval-augmented generation (RAG; retrieving documents using embeddings and conditioning a generator on them), clustering, and similarity scoring. Many strong embedding models are trained as **bidirectional encoders** (e.g., BERT-style models), because bidirectional self-attention lets each token representation incorporate information from both left and right context.

Decoder-only large language models (LLMs) are attractive embedding backbones because they are widely available at many scales and often instruction-tuned. However, they are pretrained with a **causal attention mask** (tokens cannot attend to future tokens), while embedding objectives (contrastive learning on sentence pairs) often benefit from **bidirectional attention**. This creates a training mismatch: enabling bidirectional attention changes the model’s attention computation in a way that may destabilize optimization or degrade representations.

Recent work shows multiple ways to bridge this mismatch:
- **[LLM2Vec](./references/LLM2Vec-Large-Language-Models-Are-Secretly-Powerful-Text-Encoders/meta/meta_info.txt)** converts a decoder-only LLM into a text encoder using (i) bidirectional attention, (ii) a masked next-token prediction warmup, then (iii) SimCSE-style contrastive learning.
- **[NV-Embed](./references/NV-Embed-Improved-Techniques-for-Training-LLMs-as-Generalist-Embedding-Models/meta/meta_info.txt)** reports that simply replacing causal masking with bidirectional masking during contrastive training works well for a Mistral-7B backbone (and adds improved pooling and instruction-tuning).
- **[Conan-Embedding-v2](https://arxiv.org/abs/2509.12892)** uses a causal→bidirectional soft-mask schedule, but in a from-scratch / embedding-specialized training pipeline.

A key observation from LLM2Vec is that “just make it bidirectional” behaves differently across backbones: enabling bidirectional attention without adaptation harms Sheared-LLaMA and LLaMA-2 on their MTEB subset, while Mistral behaves unusually well **[LLM2Vec](./references/LLM2Vec-Large-Language-Models-Are-Secretly-Powerful-Text-Encoders/meta/meta_info.txt)**. This suggests that the practitioner question is not “is MNTP always needed?”, but rather:

> For *backbones that are not bidirectionally-ready*, what is the minimal adaptation needed to make bidirectional embedding training work reliably?

### The Problem

LLM2Vec’s recipe uses a distinct warmup stage: **masked next-token prediction (MNTP)**. MNTP (as implemented in LLM2Vec) randomly masks input tokens and trains the model to predict the masked token using the *previous* position’s representation, using the underscore "_" as a mask token because decoder-only tokenizers typically lack a dedicated [MASK] token **[LLM2Vec](./references/LLM2Vec-Large-Language-Models-Are-Secretly-Powerful-Text-Encoders/sections/Masked%20next%20token%20prediction.md)**. The MNTP stage is relatively cheap, but it adds practical complexity: token masking logic, an extra data loader (Wikitext-103), and a multi-stage fine-tuning pipeline.

At the same time, the literature already contains two adjacent ideas:
- NV-Embed: **immediate** causal→bidirectional switch can work well (at least for Mistral).
- Conan-Embedding-v2: a **soft-mask schedule** can stabilize causal→bidirectional transition.

What is missing is a compute-matched, apples-to-apples ablation answering:

> In a standard pretrained-decoder conversion setting (LLM2Vec-style), is MNTP’s benefit primarily the *token-level objective*, or is it mainly providing a *gentle causal→bidirectional transition* that a simple schedule could replicate under the target contrastive objective?

### Key Insight and Hypothesis

**Key insight.** In a pretrained causal transformer, attention paths to future tokens are structurally unused during pretraining. Abruptly unmasking future tokens can expose uncalibrated attention behavior (especially in early training), and sentence-level contrastive losses may provide too weak a per-token learning signal to quickly “repair” this distribution shift. MNTP may help because it forces token-level gradients to flow through future-token interactions early.

A simpler alternative is to stay within the contrastive objective but avoid the discrete switch: start contrastive training with an attention mask that is *almost causal* and gradually relax it to fully bidirectional attention. This idea is related to soft-mask schedules used in Conan-Embedding-v2, but our hypothesis is specifically about **retrofitting pretrained decoders**: the transition needs to preserve pretrained causal behavior while gradually activating future-token interactions.

**Hypothesis.** On decoder-only backbones where an immediate causal→bidirectional switch is brittle (as suggested by LLM2Vec’s observations for LLaMA-family models), a token-agnostic soft-mask schedule during SimCSE training will recover most of LLM2Vec’s MNTP+SimCSE performance.

This could fail because MNTP provides a dense token-level training signal that contrastive loss cannot replicate: contrastive training supervises only the pooled embedding, so the model may not learn to use newly opened future-attention weights effectively without auxiliary token-level prediction.

---

## Proposed Approach

### Overview

We propose **Soft-Masked SimCSE (SM-SimCSE)**: a single-stage, contrastive-only training recipe for converting a pretrained decoder-only LLM into a bidirectional text encoder.

We run unsupervised **SimCSE (Simple Contrastive Learning of Sentence Embeddings; dropout-based contrastive training)** **[SimCSE](https://arxiv.org/abs/2104.08821)** with bidirectional attention at inference, but we apply a **soft causal→bidirectional schedule** during training. We compare against:
- an **immediate-switch** baseline (NV-Embed-style mask change, but using SimCSE), and
- the **LLM2Vec MNTP→SimCSE** recipe.

### Method Details

#### Soft-mask parameterization (same form as GG-SM)

We use the soft-mask form introduced in GG-SM **[GG-SM](./references/How-Do-Decoder-Only-LLMs-Perceive-Users-Rethinking-Attention-Masking-for-User-Representation-Learning/sections/LLMs%20as%20Encoders%20Training%20Recipe.md)**. For a sequence length L, define an additive mask bias M(t) at training step t:

- For j ≤ i (past tokens): M_ij(t) = 0
- For j > i (future tokens): M_ij(t) = log w(t)

where w(t) ∈ (0, 1]. When w(t) is near 0, the model is nearly causal; when w(t)=1, the mask becomes fully bidirectional.

#### Token-agnostic linear schedule (ours)

We use a deterministic linear schedule:

- w(t) = ε + (1-ε) · (t / T_total)

with ε=1e-4 and T_total = total training steps.

#### Contrastive objective (SimCSE)

Unsupervised SimCSE constructs a positive pair by encoding the same sentence twice with independent dropout; other batch examples form in-batch negatives. We use mean pooling over content tokens for embeddings.

#### Why this is meaningfully different from prior schedules

Conan-Embedding-v2 uses soft-mask schedules in an embedding-specialized training pipeline (including from-scratch pretraining and large-scale supervised embedding training). Our proposal tests a different hypothesis: in *pretrained decoder conversion*, the main role of MNTP might be to provide a gentle activation path for future-token attention, which could be replicated by a schedule under the target contrastive objective.

### Key Innovations

1. **Mechanistic ablation of MNTP**: tests whether MNTP’s benefit is mainly a causal→bidirectional curriculum effect versus an essential token-level objective.
2. **Single-objective conversion recipe**: if successful, drops the MNTP stage and its masking/data plumbing.
3. **Minimal schedule**: uses an existing soft-mask parameterization but removes gradient guidance to isolate whether “gentle transition” alone is sufficient.

---

## Related Work

### Field Overview

Text embedders are commonly trained with contrastive objectives that encourage semantically related texts to have high cosine similarity and unrelated texts to be separated. The main axes in decoder-only LLM embedders include pooling choices, instruction formats, training data (public vs synthetic), and how the model gains access to “global” context despite causal pretraining.

Mask-based approaches fall into: (i) immediate mask removal (train bidirectional directly), (ii) curricula/schedules that gradually relax the mask, and (iii) auxiliary objectives (e.g., MNTP) that adapt the model to bidirectionality before contrastive training.

### Related Papers

- **[BERT](https://aclanthology.org/N19-1423/)**: bidirectional masked-language-model pretraining; foundational encoder backbone.
- **[Sentence-BERT](https://arxiv.org/abs/1908.10084)**: siamese bi-encoder training for embeddings.
- **[SimCSE](https://arxiv.org/abs/2104.08821)**: dropout-based contrastive sentence embeddings.
- **[MTEB](https://arxiv.org/abs/2210.07316)**: Massive Text Embedding Benchmark (56 datasets across retrieval, reranking, clustering, classification, and STS).
- **[BEIR](https://arxiv.org/abs/2104.08663)**: retrieval benchmark suite commonly used to evaluate dense embeddings.
- **[SGPT](https://arxiv.org/abs/2202.08904)**: early decoder-only embedding approach with pooling/weighting.
- **[INSTRUCTOR](https://arxiv.org/abs/2212.09741)**: instruction-conditioned embeddings.
- **[E5](https://arxiv.org/abs/2212.03533)**: weakly supervised contrastive pretraining for retrieval embeddings.
- **[Improving Text Embeddings with LLMs / E5-Mistral](https://arxiv.org/abs/2401.00368)**: strong LLM-based embedder family.
- **[Echo embeddings](https://arxiv.org/abs/2402.15449)**: inference-time repetition to approximate bidirectional context for causal models.
- **[Contriever](https://arxiv.org/abs/2112.09118)**: unsupervised dense retrieval with contrastive learning.
- **[GTR](https://arxiv.org/abs/2112.04511)**: large dual encoders for retrieval.
- **[BGE-M3](https://arxiv.org/abs/2402.03216)**: multi-lingual, multi-function embedding model.
- **[SFR-Embedding](https://arxiv.org/abs/2403.12051)**: fine-tunes strong embedders with task-homogeneous batching.
- **[GRITLM](https://arxiv.org/abs/2402.16852)**: joint embedding and generation training.
- **[Causal2Vec](https://arxiv.org/abs/2507.23386)**: preserves causal attention by injecting a contextual token from a bidirectional encoder.
- **[Conan-Embedding-v2](https://arxiv.org/abs/2509.12892)**: embedding-specialized training with soft-mask schedules.
- **[GG-SM](./references/How-Do-Decoder-Only-LLMs-Perceive-Users-Rethinking-Attention-Masking-for-User-Representation-Learning/meta/meta_info.txt)**: gradient-guided warmup + schedule for causal→bidirectional transition (user representation learning).
- **[LLM2Vec](./references/LLM2Vec-Large-Language-Models-Are-Secretly-Powerful-Text-Encoders/meta/meta_info.txt)**: bidirectional attention + MNTP + SimCSE.
- **[NV-Embed](./references/NV-Embed-Improved-Techniques-for-Training-LLMs-as-Generalist-Embedding-Models/meta/meta_info.txt)**: bidirectional attention during contrastive training + improved pooling.
- **[Bitune](https://aclanthology.org/2025.emnlp-main.481/)**: two-pass bidirectional+causal prefilling with learnable mixing for instruction tuning.
- **[EmbeddingGemma](https://deepmind.google/models/gemma/embeddinggemma/)**: compact embedding model emphasizing efficiency.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Immediate mask removal | Train decoder-only LLM with bidirectional attention directly | NV-Embed, some LLM2Vec settings | MTEB, BEIR | May fail for some backbones due to pretrain mismatch |
| Multi-stage adaptation | Add an auxiliary objective before contrastive training | LLM2Vec (MNTP→SimCSE) | MTEB | More pipeline complexity; objective mismatch |
| Mask curriculum | Gradually relax causal masking during representation learning | Conan-Embedding-v2, GG-SM | MTEB (Conan), proprietary (GG-SM) | Extra hyperparameters; unclear necessity |
| Inference-time approximation | Modify input at inference to expose global context | Echo embeddings | MTEB | Increased inference tokens |
| Causal-preserving context injection | Keep causal mask; inject global context via extra encoder/token | Causal2Vec | MTEB | Extra module or assumptions |

### Closest Prior Work

1. **LLM2Vec** **[LLM2Vec](./references/LLM2Vec-Large-Language-Models-Are-Secretly-Powerful-Text-Encoders/meta/meta_info.txt)**: The closest conversion recipe baseline. We test whether MNTP is necessary if we replace the hard mask switch with a soft curriculum during contrastive training.
2. **NV-Embed** **[NV-Embed](./references/NV-Embed-Improved-Techniques-for-Training-LLMs-as-Generalist-Embedding-Models/meta/meta_info.txt)**: Suggests that immediate bidirectional training can work well (for Mistral). We test the regime where immediate switch may be brittle (LLaMA-family) and whether a minimal schedule recovers performance.
3. **Conan-Embedding-v2** **[Conan-Embedding-v2](https://arxiv.org/abs/2509.12892)**: Uses schedules in embedding-specialized training. Our focus is a mechanistic ablation in pretrained-decoder conversion: schedule as an MNTP replacement.

**Novelty Kill Search Summary:** Searched for “LLM2Vec without MNTP”, “soft masking causal to bidirectional attention schedule embedding model”, “causal to bidirectional attention curriculum SimCSE”, and “MNTP replacement soft mask schedule”. Also checked OpenReview/ACL Anthology/GitHub at a high level for “MNTP replacement” and “soft-mask SimCSE” phrasing. No prior work explicitly testing a **token-agnostic soft-mask curriculum as a drop-in MNTP replacement** in the LLM2Vec pretrained-decoder conversion setting was found as of 2026-02-16. (Full query log in `notes.md`.)

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LLM2Vec | Bi-attn + MNTP warmup + SimCSE | Multi-stage pipeline; extra objective | Replace MNTP with soft-mask curriculum under SimCSE | If MNTP mainly provides a gentle transition, schedule should suffice |
| NV-Embed | Immediate causal→bidirectional switch during contrastive training | Evidence mostly on Mistral; not a mechanistic MNTP ablation | Test immediate switch vs schedule vs MNTP on LLaMA-family | Identifies when immediate switch fails and a minimal schedule fixes it |
| Conan-Embedding-v2 | Soft-mask schedule in embedding-specialized training | Different setting (from scratch / supervised embedding pipeline) | Apply minimal schedule as a retrofit for pretrained decoders | Tests whether schedule alone can replace MNTP in conversion |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):**
- **No-training baselines (inference only)**: causal pooling; echo embeddings.
- **Immediate-switch baseline**: remove causal mask at step 0 during SimCSE training (NV-Embed-style mask change, but SimCSE objective).
- **Closest existing method**: LLM2Vec MNTP→SimCSE.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Sheared-LLaMA-1.3B | 1.3B | https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B | Same backbone family as LLM2Vec’s S-LLaMA experiments |

**Training Data:**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Wikitext-103 | MNTP stage for LLM2Vec baseline | ~103M tokens | https://huggingface.co/datasets/Salesforce/wikitext | research |
| SimCSE Wikipedia sentences | Unsupervised SimCSE stage (all training conditions) | large | https://github.com/princeton-nlp/SimCSE | research |

**Three main training conditions (A/B/C):**

1. **Immediate-switch SimCSE (A)**: enable fully bidirectional attention from step 0 during SimCSE (this is the “just remove the causal mask” baseline).
2. **Soft-Masked SimCSE (B, ours)**: run SimCSE with the soft-mask schedule w(t)=ε+(1-ε)t/T_total.
3. **LLM2Vec (C)**: MNTP for 1000 steps then SimCSE for 1000 steps, following LLM2Vec **[LLM2Vec](./references/LLM2Vec-Large-Language-Models-Are-Secretly-Powerful-Text-Encoders/sections/Masked%20next%20token%20prediction.md)**.

**Training details (match LLM2Vec defaults unless stated):**
- **LoRA (Low-Rank Adaptation; parameter-efficient fine-tuning)** with the same ranks and merging procedure as LLM2Vec.
- Sequence length 512.
- SimCSE uses in-batch negatives and mean pooling.

**Resource Estimate** (must fit ≤768 A100-GPU-hours):
- LLM2Vec reports MNTP (1000 steps, bs32) takes ~90 minutes and SimCSE (1000 steps, bs128) takes ~2.5 hours on a single 80GB A100 for 7B models **[LLM2Vec](./references/LLM2Vec-Large-Language-Models-Are-Secretly-Powerful-Text-Encoders/sections/Masked%20next%20token%20prediction.md)**.
- For a 1.3B model, we conservatively budget **≤4 A100-hours per training run**.
- With 3 conditions × 3 seeds = 9 runs: **≤36 A100-hours** training.
- MTEB evaluation for 9 checkpoints: conservatively **≤150 A100-hours**.
- Total expected: **≤200 A100-hours**.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MTEB (English) | Standard suite for text embeddings across 56 datasets | MTEB average score (higher is better) + per-task-type breakdown | test | https://github.com/embeddings-benchmark/mteb | `mteb` |

### Main Results

#### Comparability Rules (CRITICAL)

All A/B/C runs must use:
- Same base model (Sheared-LLaMA-1.3B)
- Same SimCSE data for contrastive training
- Same evaluation protocol (pooling, instruction handling, MTEB script version)

#### Reference baseline numbers from LLM2Vec (Sheared-LLaMA-1.3B; unsupervised MTEB avg)

From Table 1 of LLM2Vec **[LLM2Vec](./references/LLM2Vec-Large-Language-Models-Are-Secretly-Powerful-Text-Encoders/sections/Results%20on%20full%20MTEB.md)**:

| Method | Base Model | MTEB Avg | Source | Notes |
|---|---|---:|---|---|
| Causal baseline (Uni + weighted mean) | Sheared-LLaMA-1.3B | 35.05 | LLM2Vec | (1 run) |
| Echo embeddings (inference-time repetition) | Sheared-LLaMA-1.3B | 39.10 | LLM2Vec | (1 run) |
| Bi + MNTP (no SimCSE) | Sheared-LLaMA-1.3B | 41.43 | LLM2Vec | (1 run) |
| Bi + MNTP + SimCSE | Sheared-LLaMA-1.3B | 49.42 | LLM2Vec | (1 run) |

#### Results Table (to be verified)

| Method | Base Model | Benchmark | MTEB Avg (mean±std) | Source | Notes |
|---|---|---|---:|---|---|
| Immediate-switch SimCSE (A) | Sheared-LLaMA-1.3B | MTEB | **TBD** | - | 3 seeds |
| **Soft-Masked SimCSE (B, ours)** | Sheared-LLaMA-1.3B | MTEB | **TBD** | - | 3 seeds |
| LLM2Vec (C) | Sheared-LLaMA-1.3B | MTEB | **TBD** | - | 3 seeds |

### Ablation Studies

No additional ablations are required for the core claim because the main comparison already isolates:
- the schedule effect (B vs A), and
- the MNTP effect (C vs B).

### Experimental Rigor

- **Seeds**: Train A/B/C with `seeds=[42, 123, 456]`; report mean±std.
- **Sanity check**: reproduce at least one published LLM2Vec number within tolerance (e.g., Sheared-LLaMA LLM2Vec avg ≈49.4) to validate training/eval.
- **Top confounders + controls**:
  1. **Compute mismatch**: fix total optimizer steps and keep hyperparameters aligned to LLM2Vec.
  2. **Evaluation protocol drift**: pin MTEB version and use the same pooling/instruction handling across all methods.
  3. **Mask implementation bug**: unit-test that w(t)=ε behaves nearly causally and w(t)=1 matches full bidirectional attention.

---

## Success Criteria

**Hypothesis**: Soft-Masked SimCSE (B) will close most of the gap between immediate-switch SimCSE (A) and LLM2Vec (C), because a gradual mask transition provides the needed adaptation to bidirectionality under the target contrastive objective.

**Decision Rule**:
- **Proceed** if (B) achieves MTEB Avg within **0.5 points** of (C) *and* improves over (A) by **≥1.0 point**, with the improvement larger than the pooled std across seeds.
- **Pivot** if (A) is already within 0.5 points of (C): MNTP is not necessary for this backbone under this budget; rerun the same A/B/C comparison on a backbone where bidirectionality is known to be brittle (e.g., LLaMA-2-7B-chat, as in LLM2Vec).
- **Refute** if (B) is within noise of (A) (≤0.2 points) or lags (C) by **>1.0 point**: MNTP provides unique benefit not replicated by a simple schedule.

---

## Impact Statement

If SM-SimCSE succeeds, practitioners converting pretrained decoder-only models into embedders can drop the MNTP warmup stage and implement a single-stage contrastive pipeline with a deterministic mask schedule. If it fails, the results clarify that MNTP provides genuinely necessary adaptation signal for some backbones, helping practitioners choose conversion recipes that are robust to causal→bidirectional mismatch.

---

## References

- [How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning](./references/How-Do-Decoder-Only-LLMs-Perceive-Users-Rethinking-Attention-Masking-for-User-Representation-Learning/meta/meta_info.txt) - Yuan et al., 2026
- [LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders](./references/LLM2Vec-Large-Language-Models-Are-Secretly-Powerful-Text-Encoders/meta/meta_info.txt) - BehnamGhader et al., 2024
- [NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models](./references/NV-Embed-Improved-Techniques-for-Training-LLMs-as-Generalist-Embedding-Models/meta/meta_info.txt) - Lee et al., 2024
- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) - Gao et al., 2021
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) - Muennighoff et al., 2022
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663) - Thakur et al., 2021
- [Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings](https://arxiv.org/abs/2509.12892) - Li et al., 2025
- [Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models](https://arxiv.org/abs/2507.23386) - Lin et al., 2025
- [Repetition Improves Language Model Embeddings](https://arxiv.org/abs/2402.15449) - Springer et al., 2024
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) - Reimers and Gurevych, 2019
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) - Devlin et al., 2019
- [INSTRUCTOR: One Embedder, Any Task](https://arxiv.org/abs/2212.09741) - Su et al., 2023
- [E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533) - Wang et al., 2022
- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368) - Wang et al., 2024
- [GRITLM](https://arxiv.org/abs/2402.16852) - Muennighoff et al., 2024
- [Contriever](https://arxiv.org/abs/2112.09118) - Izacard et al., 2021
- [GTR](https://arxiv.org/abs/2112.04511) - Ni et al., 2022
- [BGE-M3](https://arxiv.org/abs/2402.03216) - Chen et al., 2024
- [SFR-Embedding](https://arxiv.org/abs/2403.12051) - Meng et al., 2024
- [Bitune: Leveraging Bidirectional Attention to Improve Decoder-Only LLMs](https://aclanthology.org/2025.emnlp-main.481/) - Kopiczko et al., 2025
- [EmbeddingGemma](https://deepmind.google/models/gemma/embeddinggemma/) - Google DeepMind, 2025
