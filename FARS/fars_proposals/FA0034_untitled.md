# untitled

# Adaptive Rerank Budgeting for VidVec via Layer-Disagreement Routing

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### One-sentence thesis (with decision rule)
**Thesis:** In VidVec-style video-text retrieval, **layer-wise ranking disagreement** is a better per-query confidence signal than the standard **top-1/top-2 similarity margin**, enabling **adaptive rerank budgeting** (choose K∈{10,100} per query) that reduces average reranker calls while preserving Recall@K.

**Decision rule (primary):** On **MSR-VTT 1k-A**, tune thresholds for both routing policies so that **avg-K = 30 ± 1** on the validation split. Evaluate on the test split. We declare success iff:
1) **Layer-disagreement routing beats margin routing** by **≥ +0.3 Recall@1** (T2V) at matched avg-K, and
2) Layer-disagreement routing is **within 1.0 Recall@1** of **fixed K=100**.
(Secondary check: the same holds on DiDeMo within ±0.5 Recall@1 vs margin, using DiDeMo’s own validation split for compute matching.)

### Context and Motivation

Video text retrieval (given a text query, find the matching video in a database, and vice versa) is commonly implemented as a **two-stage pipeline**: (1) a fast embedding model retrieves a candidate set using vector similarity, and (2) a more expensive **reranker** scores the shortlist with a higher-fidelity relevance model. In practice, the reranker often dominates inference cost because it must process each query candidate pair (cross-encoder style).

**VidVec** (arXiv:2602.08099) shows that off-the-shelf video multimodal large language models (video MLLMs) can be turned into strong retrieval systems by extracting a fixed-dimensional embedding from an **intermediate transformer layer** using an **Explicit One-word Limitation (EOL)** prompt ("summarize in one word") and an embedding token `<emb>`. VidVec then applies a second-stage reranker by reusing the MLLM's language-model head as a calibrated likelihood scorer: for each candidate, it prompts a binary question ("Respond in a single word - Yes or No") and scores `P(Yes | query, candidate)`.

This reranking is effective but expensive: it requires **K additional forward passes per query**, where VidVec uses **K=100** for the zero-shot system.

### The Problem

A fixed reranking budget (fixed K) is a blunt instrument: many queries are "easy" and the first-stage embedding ranking is already correct, so reranking those candidates wastes compute; other queries are "hard" and benefit substantially from reranking. VidVec does not provide a way to allocate reranking compute **per query**.

A common heuristic in retrieval systems is to use the **top-1/top-2 similarity margin** (how much better the best candidate is than the runner-up) as a confidence signal: large margins indicate easy queries. However, in VidVec-style MLLM embeddings, there is an additional source of uncertainty: the embedding is a **choice of readout layer**. VidVec's own layer-wise analysis suggests that multiple intermediate layers can be strong, implying that different layers may produce slightly different rankings even when using the same prompt and model.

### Key Insight and Hypothesis

**Key insight:** If a query's top candidates are **stable across several strong intermediate layers**, the embedding representation is likely reliable and reranking is less likely to change the top result. Conversely, if **layer-wise rankings disagree**, the embedding signal is uncertain and reranking is more likely to correct errors.

**Hypothesis:** At a fixed average reranking compute budget (matched average K), a simple **layer-disagreement routing policy** that chooses a per-query budget \(K(q) \in \{10, 100\}\) will retain most of VidVec's fixed-K=100 retrieval quality while using substantially fewer reranker forward passes on average, and will outperform margin-based routing at the same average K.

---

## Proposed Approach

### Overview

We propose **VidVec-RouteK**, a training-free method that routes each query to a reranking budget based on how much the **top-ranked candidates disagree across multiple intermediate-layer embeddings**.

We compare two budget-routing signals under matched compute:

- **Margin routing (baseline):** route using the top-1/top-2 similarity margin from VidVec's standard layer-24 embedding.
- **Layer-disagreement routing (ours):** route using a disagreement score computed from rankings produced by several nearby strong layers.

### Method Details

#### Background: VidVec embedding extraction
VidVec uses an **EOL prompt** ending with a special token `<emb>`, and extracts the embedding as the hidden state of the token position immediately preceding `<emb>` (called `<emb-1>`). For the zero-shot setting, VidVec reads out from **layer 24** of VideoLLaMA3-7B.

#### A. Multi-layer embeddings (for routing only)
Choose a small set of layers around VidVec's preferred layer 24:

- \(\mathcal{L} = \{20, 24, 28\}\) (default)

For each query text \(q\) and each video \(v\) in the gallery, extract embeddings \(e_q^{(\ell)}\), \(e_v^{(\ell)}\) for each \(\ell \in \mathcal{L}\).

**Implementation note (memory):** to avoid storing all hidden states, the verifier can register forward hooks on the transformer blocks for layers in \(\mathcal{L}\) and cache only the `<emb-1>` vector for each selected layer.

#### B. Layer-disagreement score
For each layer \(\ell\), compute similarity scores \(s^{(\ell)}(q,v)=\cos(e_q^{(\ell)}, e_v^{(\ell)})\) and obtain a ranking over videos. Let \(T_m^{(\ell)}(q)\) be the set of the top-\(m\) retrieved videos for query \(q\) under layer \(\ell\).

Define a simple disagreement score (higher = more disagreement):

\[
D(q) = 1 - \frac{1}{|\mathcal{L}|-1}\sum_{\ell \in \mathcal{L}\setminus\{24\}} \frac{|T_m^{(24)}(q) \cap T_m^{(\ell)}(q)|}{|T_m^{(24)}(q) \cup T_m^{(\ell)}(q)|}
\]

i.e., one minus the average Jaccard overlap between the top-\(m\) sets from layer 24 and other layers. We will use \(m=20\) by default.

#### C. Budget routing policy (two-point budget)
We choose between two reranking budgets:

- \(K_{\min}=10\)
- \(K_{\max}=100\)

Policy:

\[
K(q) = \begin{cases}
K_{\max} & \text{if } D(q) \ge \tau \\
K_{\min} & \text{otherwise}
\end{cases}
\]

The threshold \(\tau\) is tuned on a validation split to match a target **average reranking budget** (e.g., average K          = 30). This makes comparisons compute-matched.

#### D. Margin routing baseline (compute-matched)
Compute the standard margin signal from the layer-24 similarity scores:

\[
M(q) = s^{(24)}_{(1)}(q) - s^{(24)}_{(2)}(q)
\]

where \(s_{(1)}\) and \(s_{(2)}\) are the top-1 and top-2 similarity scores. Use a threshold on \(M(q)\) (tuned on the same validation split) to achieve the same average K as the disagreement policy.

#### E. Reranking (VidVec teacher reranker)
Given \(K(q)\), rerank the top-\(K(q)\) candidates from the layer-24 retrieval list using VidVec's calibrated head scorer:

- Prompt each (q, v) with a binary relevance question ending with: "Respond in a single word - Yes or No."
- Score \(S_{\text{rank}}(q,v) = P(\text{Yes} \mid q,v)\)
- Replace the order of the top-K(q) candidates using \(S_{\text{rank}}\), leaving the remaining candidates unchanged.

### Key Innovations

1. **Training-free compute allocation for MLLM retrieval:** use internal representation disagreement (across layers) as a proxy for query difficulty and reranker usefulness.
2. **Compute-matched head-to-head against a strong heuristic:** compare to margin-based routing under the same average reranking budget.
3. **Mechanistic validation without extra experimental conditions:** analyze cases where margin and disagreement make different routing decisions to determine which better predicts reranker-induced reordering.

---

## Related Work

### Field Overview

**Video-text retrieval models.** Classic approaches learn aligned video and text embeddings using contrastive objectives (e.g., CLIP-style models) and evaluate with Recall@K on datasets like MSR-VTT and DiDeMo. Recent Video Foundation Models (VFMs) scale data and architecture to reach strong retrieval performance (e.g., InternVideo2, VideoPrism). Many systems add a second-stage reranker or cross-encoder to improve top-1 accuracy.

**MLLMs as embedders and rerankers.** A recent trend repurposes multimodal LLMs as embedding models using prompting and special embedding tokens (e.g., E5-V, LamRA, VLM2Vec). VidVec extends this to video and shows that a calibrated MLLM-head reranker provides large gains but incurs K forward passes per query.

**Adaptive compute and cascades.** In ranking systems and transformers, early-exit and cascade strategies allocate compute based on confidence (e.g., early-exit transformers; learning-to-rank cascades). However, confidence signals that are effective for MLLM-derived embeddings (where layer choice matters) remain underexplored.

### Related Papers

- **[VidVec](./references/VidVec-Unlocking-Video-MLLM-Embeddings-for-Video-Text-Retrieval/meta/meta_info.txt)**: Shows intermediate-layer video MLLM embeddings are strong for retrieval and introduces a calibrated MLLM-head reranker with K forward passes.
- **[LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](https://arxiv.org/abs/2412.01720)**: Uses EOL prompting + embedding token for multimodal retrieval and includes a learned reranker, motivating two-stage pipelines.
- **[E5-V](https://arxiv.org/abs/2407.12580)**: Introduces prompt-based embedding extraction for multimodal embeddings and shows text-only training can help alignment.
- **[VLM2Vec](https://arxiv.org/abs/2410.05160)**: Trains MLLMs for massive multimodal embedding tasks via contrastive objectives.
- **[VLM2Vec-V2](https://arxiv.org/abs/2507.04590)**: Extends VLM2Vec to videos and visual documents, providing strong embedding baselines.
- **[VLM2Vec](https://arxiv.org/abs/2410.05160)**: Contrastively trains VLMs into universal embedding models (MMEB benchmark), a core baseline family VidVec compares against.
- **[UniME-V2](https://arxiv.org/abs/2510.13515)**: Uses MLLM-as-a-judge signals to improve universal multimodal embeddings, related to using model-internal signals for retrieval.
- **[UNITE](https://arxiv.org/abs/2505.19650)**: Trains on modality-specific pairs for universal embeddings, a representative trained MLLM embedder baseline.
- **[MMRet / BGE-VL (MegaPairs)](https://arxiv.org/abs/2412.14475)**: Trains multimodal retrieval models (and releases BGE-VL) on the MegaPairs synthetic dataset; a strong multimodal embedding baseline family used in retrieval comparisons.
- **[InternVideo2](https://arxiv.org/abs/2403.15377)**: A large-scale VFM with strong retrieval results; represents the specialized-encoder paradigm.
- **[VideoPrism](https://arxiv.org/abs/2402.13217)**: A massively trained VFM for video understanding and retrieval.
- **[Perception Encoder](https://arxiv.org/abs/2504.13181)**: Shows intermediate layers can outperform output layers for visual embeddings, supporting the idea that layer choice matters.
- **[Layer by Layer: Uncovering Hidden Representations in Language Models](https://arxiv.org/abs/2502.02013)**: Analyzes layer-wise utility in LLMs, motivating multi-layer signals.
- **[TGIF](./references/Text-Guided-Layer-Fusion-Mitigates-Hallucination-in-Multimodal-LLMs/meta/meta_info.txt)**: Uses query-conditioned routing over vision-encoder layers; conceptually related to layer-based routing but targets hallucination rather than rerank budgeting.
- **[HVP-Net](./references/Delving-Deeper-Hierarchical-Visual-Perception-for-Robust-Video-Text-Retrieval/meta/meta_info.txt)**: Improves video-text retrieval via hierarchical intermediate vision features; highlights that naive multi-layer signals can be harmful without careful design.
- **[VIRTUE](./references/VIRTUE-Versatile-Video-Retrieval-Through-Unified-Embeddings/meta/meta_info.txt)**: An MLLM-based video retrieval framework with a trained reranker; provides an alternative reranking design point.
- **[Dual-Softmax Loss for Video-Text Retrieval](https://arxiv.org/abs/2109.04290)**: A standard calibration trick (often also applied at inference) used in many retrieval systems including VidVec.
- **[CLIP4Clip](https://arxiv.org/abs/2104.08860)**: A classic CLIP-based video-text retrieval baseline and evaluation protocol reference.
- **[X-CLIP](https://arxiv.org/abs/2207.07285)**: A strong CLIP-based method for video-text retrieval that improves temporal modeling.
- **[MUSE](https://arxiv.org/abs/2408.10575)**: A recent strong video-text retrieval baseline (ResMamba multi-scale learner) used in recent comparisons.
- **[RankVideo](https://arxiv.org/abs/2602.02444)**: Uses reasoning-based reranking for text-to-video retrieval, showing rerankers can materially change top results.
- **[Early Exit Strategies for Learning-to-Rank Cascades](https://arxiv.org/abs/2008.09711)**: Formalizes compute-saving cascades in ranking; relevant as a classical efficiency lens.
- **[DeeBERT](https://arxiv.org/abs/2006.00420)**: Early-exit transformer inference using confidence thresholds; a canonical adaptive-compute baseline.
- **[PABEE](https://arxiv.org/abs/2005.09442)**: Patience-based early exit for BERT; another canonical adaptive-compute reference.
- **[ToolRerank](https://aclanthology.org/2024.lrec-main.1413.pdf)**: Adaptive truncation for reranking in tool retrieval; conceptually similar to adaptive K selection in a different domain.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Video-text dual encoders / VFMs | Learn aligned video+text embeddings at scale | InternVideo2, VideoPrism, CLIP4Clip, X-CLIP | MSR-VTT, DiDeMo, VATEX, MSVD | Often require large paired datasets; may still use rerankers |
| MLLM-as-embedder for retrieval | Prompt-based embedding extraction from MLLMs | VidVec, LamRA, E5-V, VLM2Vec | MSR-VTT, DiDeMo, MMEB-V2 | Embedding quality depends on prompting and layer choice |
| Multi-layer representations | Use intermediate layers for better features | Perception Encoder, Layer-by-Layer, HVP-Net | MSR-VTT, DiDeMo, ActivityNet | Multi-layer fusion can fail without careful design |
| Reranking / cross-encoders | Re-score top-K candidates with heavier models | VidVec reranker, RankVideo, VIRTUE-Ranker | Recall@K / nDCG | High latency; fixed K wastes compute |
| Adaptive compute / cascades | Allocate computation based on confidence | DeeBERT, PABEE, LTR cascades, ToolRerank | Various | Confidence signals may not transfer across modalities |

### Closest Prior Work

- **VidVec**: Introduces the specific reranker we aim to make cheaper on average; does not study per-query compute allocation.
- **ToolRerank / LTR cascades**: Show adaptive truncation / early exit can save compute, but do not use layer-wise representation disagreement signals and are not studied in MLLM video retrieval.
- **TGIF**: Uses layer routing, but for selecting vision encoder layers to improve grounding; our routing target is rerank compute and our signal is cross-layer ranking disagreement.
- **HVP-Net**: Highlights that multi-layer signals can be nontrivial; our method avoids fusion and uses disagreement only as a confidence proxy.
- **VIRTUE / RankVideo**: Represent alternative reranking designs; they do not address training-free adaptive K for the specific (query,candidate) scoring budget.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| VidVec | Two-stage retrieval with MLLM-head reranking (fixed K) | Reranking cost scales linearly with K; no per-query budgeting | Add per-query routing of K | Avoid wasting reranker passes on easy queries |
| Margin-based routing (standard heuristic) | Uses similarity margin as confidence | Can be overconfident when representation is unstable | Use layer disagreement as a representation-uncertainty signal | Disagreement should better predict when reranking changes the ranking |
| TGIF | Query-conditioned layer fusion for grounding | Different task (hallucination), not rerank budgeting | Reuse the idea of layer-dependent signals but for compute allocation | Enables training-free efficiency gains in retrieval pipelines |
| ToolRerank / LTR cascades | Adaptive truncation / early exit in ranking | Not evaluated for MLLM video retrieval; signals not representation-based | Apply adaptive compute to VidVec and use a new signal | Better calibrated to MLLM embedding uncertainty |
| VIRTUE / RankVideo | Alternative (trained) rerankers | Adds training cost; still needs reranking compute | Keep VidVec reranker but run it selectively | More deployment-friendly and training-free |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| VideoLLaMA3-7B | 7B | https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B | Apache-2.0 license; use `trust_remote_code=True` |

**Training Data (if applicable):**

No model training or fine-tuning is required. We only tune a scalar routing threshold \(\tau\) on a validation set to match a target average K.

**Resource Estimate**:

- **Compute budget**: dominated by reranking forward passes.
  - For MSR-VTT 1k-A (about 1k queries): fixed K=100 implies about 100k query video forward passes; adaptive avg-K=30 implies about 30k passes.
  - All three conditions together are on the order of 160k forward passes; this should be feasible within the 768 GPU-hour budget with batching on A100s.
- **GPU memory**: MSR-VTT/DiDeMo videos are short (tens of seconds), so even VidVec's 2 FPS sampling results in tens of frames per clip (well below the 180-frame cap).
- **Implementation complexity**: requires extracting `<emb-1>` hidden states at selected layers (recommended via forward hooks for memory efficiency).

### Benchmarks and Metrics

**Evaluation protocol note:** MSR-VTT 1k-A is typically evaluated with ~1k text queries (one caption per video in the 1k-A test set), and DiDeMo is typically evaluated on ~1k test videos/queries (commonly reported as 1,003–1,004).

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MSR-VTT (1k-A) | Standard short-video text retrieval benchmark | Recall@1/5/10 (T2V) | test_1k | https://huggingface.co/datasets/friedrichor/MSR-VTT | Use standard Recall@K evaluation as in CLIP4Clip / VidVec |
| DiDeMo | Paragraph-to-video retrieval benchmark (short clips) | Recall@1/5/10 (T2V) | test | https://opendatalab.com/OpenDataLab/DiDeMo/download | Use standard DiDeMo retrieval evaluation (e.g., LAVIS scripts) |

### Published Baselines (for context)

The following Text-to-Video Recall@K numbers are reported by **VidVec** for recent MLLM embedders and for **VidVec-O** under a unified evaluation setup (2 FPS sampling, 180-frame cap, and dual-softmax). We include the MSR-VTT and DiDeMo columns since those are our main benchmarks.

**Source**: VidVec Table 2 (extracted from `./references/VidVec-Unlocking-Video-MLLM-Embeddings-for-Video-Text-Retrieval/sections/MSR-VTT VATEX DiDeMo.md`).

|| Method | MSR-VTT R@1 | R@5 | R@10 | DiDeMo R@1 | R@5 | R@10 |
||---|---:|---:|---:|---:|---:|---:|
|| LamRA | 50.9 | 72.8 | 80.0 | 47.1 | 72.8 | 81.5 |
|| VLM2Vec | 44.9 | 68.6 | 76.1 | 41.3 | 65.4 | 72.5 |
|| MMRet-v1.5 (BGE-VL) | 47.9 | 70.9 | 77.7 | 46.1 | 73.1 | 80.7 |
|| B3 | 48.7 | 70.5 | 77.9 | 43.3 | 65.3 | 72.5 |
|| VLM2Vec-V2 | 42.9 | 64.8 | 72.2 | 44.2 | 68.0 | 76.1 |
|| UNITE | 45.2 | 70.3 | 79.3 | 40.3 | 68.7 | 78.1 |
|| UniME-V2 | 46.3 | 65.7 | 72.4 | 38.7 | 61.0 | 69.0 |
|| **VidVec-O** | **54.9** | **77.5** | **84.1** | **56.5** | **79.7** | **86.0** |

### Main Results

We run **three compute-matched conditions** with the same base model and reranker:

1) **Fixed rerank (K=100)**: VidVec-style retrieval + rerank top-100 for every query.
2) **Adaptive-K (margin baseline)**: route \(K(q)\) per query using the layer-24 similarity margin; tune the threshold to achieve **avg-K = 30 ± 1** on MSR-VTT.
3) **Adaptive-K (ours: layer disagreement)**: route \(K(q)\) per query using disagreement across layers {20,24,28}; tune the threshold to achieve the **same avg-K (±1)** as (2) on MSR-VTT.

**Note:** VidVec’s paper does not expose the exact fixed-K=100 Recall@K numbers in our local extraction (their Table 1/4 were not captured), so the verifier should treat the fixed-K condition as a reproducibility baseline to be re-run.

#### Results Table (to be filled by verifier)

| Method | Base Model | Benchmark | R@1 | R@5 | R@10 | Avg K | Notes |
|---|---|---|---:|---:|---:|---:|---|
| Fixed rerank (K=100) | VideoLLaMA3-7B | MSR-VTT | (run) | (run) | (run) | 100 | VidVec-style teacher-quality baseline |
| Adaptive-K (margin) | VideoLLaMA3-7B | MSR-VTT | (run) | (run) | (run) | 30±1 | Threshold tuned on MSR-VTT val |
| **Adaptive-K (layer disagreement)** | VideoLLaMA3-7B | MSR-VTT | (run) | (run) | (run) | 30±1 | Avg-K matched to margin (±1) |
| Fixed rerank (K=100) | VideoLLaMA3-7B | DiDeMo | (run) | (run) | (run) | 100 | Tune thresholds on DiDeMo val separately for compute matching |
| Adaptive-K (margin) | VideoLLaMA3-7B | DiDeMo | (run) | (run) | (run) | 30±1 | — |
| **Adaptive-K (layer disagreement)** | VideoLLaMA3-7B | DiDeMo | (run) | (run) | (run) | 30±1 | Avg-K matched to margin (±1) |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| Disagreement layers {18,24,30} | Change \(\mathcal{L}\) | Should not be brittle if the signal is truly uncertainty-driven |
| Disagreement metric = rank correlation | Replace Jaccard with Kendall         tau on top-100 ranks | If both work similarly, routing is robust |
| Union-topK rerank pool | Rerank candidates from union of top lists across layers | If improves, suggests disagreement captures missing candidates |

### Analysis (Optional)

**Mechanism analysis (recommended):**
- Measure how often the teacher reranker changes the top-1 candidate relative to the stage-1 ranking.
- Compare margin vs disagreement as predictors of reranker impact:
  - AUC for predicting "rerank changes top-1"
  - Average improvement from reranking conditional on high score
- On queries where margin and disagreement make opposite routing decisions, report which routing decision was correct (i.e., whether reranking actually helped).

---

## Success Criteria

We use the **same decision rule as in the Introduction**.

**Primary success (MSR-VTT 1k-A):** Tune both routing thresholds so avg-K = 30 ± 1 on the validation split. On the test split, success iff:
1) **Layer-disagreement routing − margin routing ≥ +0.3 Recall@1** (T2V), and
2) **Fixed K=100 − layer-disagreement routing ≤ 1.0 Recall@1**.

**Secondary check (DiDeMo):** Re-tune thresholds on DiDeMo’s validation split for avg-K = 30 ± 1 and require layer-disagreement routing to be **not worse than margin by more than 0.5 Recall@1**.

**Mechanism evidence (non-gating):** Disagreement should yield higher AUC than margin for predicting whether reranking changes top-1 ("rerank flips top-1"), but this is supporting analysis and not required for success.

---

## Impact Statement

If successful, VidVec-RouteK makes VidVec-style MLLM retrieval significantly cheaper on average by avoiding unnecessary reranker forward passes on easy queries, without requiring any extra model training or labels. This would make MLLM-based retrieval pipelines more practical for deployment in video search and multimodal retrieval systems where reranking cost is the dominant bottleneck.

---

## References

- [VidVec: Unlocking Video MLLM Embeddings for Video-Text Retrieval](./references/VidVec-Unlocking-Video-MLLM-Embeddings-for-Video-Text-Retrieval/meta/meta_info.txt) - Tzachor et al., 2026
- [Delving Deeper: Hierarchical Visual Perception for Robust Video-Text Retrieval (HVP-Net)](./references/Delving-Deeper-Hierarchical-Visual-Perception-for-Robust-Video-Text-Retrieval/meta/meta_info.txt) - Xie et al., 2026
- [Text-Guided Layer Fusion Mitigates Hallucination in Multimodal LLMs (TGIF)](./references/Text-Guided-Layer-Fusion-Mitigates-Hallucination-in-Multimodal-LLMs/meta/meta_info.txt) - Lin et al., 2026
- [VIRTUE: Versatile Video Retrieval Through Unified Embeddings](./references/VIRTUE-Versatile-Video-Retrieval-Through-Unified-Embeddings/meta/meta_info.txt) - Halbe et al., 2026
- [LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](https://arxiv.org/abs/2412.01720)
- [E5-V: Universal Embeddings with Multimodal Large Language Models](https://arxiv.org/abs/2407.12580)
- [VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks](https://arxiv.org/abs/2410.05160)
- [VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents](https://arxiv.org/abs/2507.04590)
- [UNITE: Modality Curation for Universal Multimodal Embeddings](https://arxiv.org/abs/2505.19650)
- [UniME-V2: MLLM-as-a-Judge for Universal Multimodal Embedding Learning](https://arxiv.org/abs/2510.13515)
- [MegaPairs / MMRet-v1.5 (BGE-VL)](https://arxiv.org/abs/2412.14475)
- [InternVideo2](https://arxiv.org/abs/2403.15377) - Wang et al., 2024
- [VideoPrism](https://arxiv.org/abs/2402.13217)
- [Perception Encoder: The Best Visual Embeddings Are Not at the Output of the Network](https://arxiv.org/abs/2504.13181)
- [Layer by Layer: Uncovering Hidden Representations in Language Models](https://arxiv.org/abs/2502.02013)
- [MUSE: Mamba is Efficient Multi-scale Learner for Text-video Retrieval](https://arxiv.org/abs/2408.10575)
- [RankVideo: Reasoning Reranking for Text-to-Video Retrieval](https://arxiv.org/abs/2602.02444)
- [Dual-Softmax Loss for Video-Text Retrieval](https://arxiv.org/abs/2109.04290) - Bain et al., 2021
- [CLIP4Clip](https://arxiv.org/abs/2104.08860) - Luo et al., 2021
- [X-CLIP](https://arxiv.org/abs/2207.07285) - Ma et al., 2022
- [Early Exit Strategies for Learning-to-Rank Cascades](https://arxiv.org/abs/2008.09711)
- [DeeBERT](https://arxiv.org/abs/2006.00420) - Xin et al., 2020
- [PABEE](https://arxiv.org/abs/2005.09442) - Zhou et al., 2020
- [ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval](https://aclanthology.org/2024.lrec-main.1413.pdf) - Li et al., 2024
