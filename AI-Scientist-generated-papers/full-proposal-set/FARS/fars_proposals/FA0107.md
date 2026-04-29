# untitled

# ConvergeStop: Convergence-Based Halting for Generative Text Embeddings

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Text embedding models map a query or document into a fixed-dimensional vector such that cosine similarity reflects semantic relatedness. They are a core component of dense retrieval systems (search and retrieval-augmented generation), where query-time latency and throughput are often dominated by the embedding model.

Recent embedding models increasingly reuse decoder-only large language model (LLM) backbones. Two notable trends are:

1. **Unified embedding + generation models** (e.g., GRIT), which reduce system complexity by using a single model for both retrieval embeddings and downstream generation (**[Generative Representational Instruction Tuning](./references/Generative-Representational-Instruction-Tuning/meta/meta_info.txt)**).
2. **Generative embedding models**, which explicitly allocate additional inference-time computation (e.g., generating auxiliary tokens) to refine embeddings. For example, GIRCSE generates a sequence of auxiliary **“soft tokens”** (continuous token embeddings formed as a probability-weighted mixture over the vocabulary) to iteratively refine a representation, and reports that increasing the number of generated tokens at inference monotonically improves embedding quality (**[GIRCSE](./references/LET-LLMS-SPEAK-EMBEDDING-LANGUAGES-GENERATIVE-TEXT-EMBEDDINGS-VIA-ITERATIVE-CONTRASTIVE-REFINEMENT/meta/meta_info.txt)**).

These approaches suggest a new deployment trade-off: generative embedders can improve embedding quality, but they also introduce a direct inference-time compute knob (generation length).

### The Problem

Generative embedding methods typically use a **fixed generation length K** for all inputs at inference time. In GIRCSE, the authors set **K=5 during training** and **K=20 during inference** (reproducibility section in **[GIRCSE](./references/LET-LLMS-SPEAK-EMBEDDING-LANGUAGES-GENERATIVE-TEXT-EMBEDDINGS-VIA-ITERATIVE-CONTRASTIVE-REFINEMENT/meta/meta_info.txt)**). A fixed K is operationally simple, but it can be inefficient:

- If K is set large to match the hardest queries, then many “easy” queries may waste compute.
- If K is set small to reduce cost, then hard queries may lose retrieval quality.

A practitioner can partially address this by choosing a smaller fixed K (e.g., 5 or 10). However, this does not exploit per-query variability. If query difficulty is heavy-tailed (many easy queries, few hard ones), then a per-query adaptive policy may achieve a better quality–compute trade-off than any single fixed K.

Existing adaptive-computation techniques do not directly address this setting:

- **Early-exit Transformers** (e.g., FastBERT, PABEE) stop at intermediate layers for classification tasks, not for iterative token generation to form embeddings.
- **Adaptive-length embeddings** (e.g., ALE) adapt the *vector dimensionality* for retrieval, not the number of iterative refinement steps.

### Key Insight and Hypothesis

**Key insight.** In causal (decoder-only) generative embedding models, the embedding produced after k refinement steps defines a natural intermediate representation \(z_k\). Because causal attention prevents future tokens from changing earlier hidden states, \(z_k\) can be updated incrementally with negligible overhead while generating refinement tokens.

**Hypothesis.** For a substantial fraction of queries, the embedding refinement trajectory \(z_1, z_2, \dots\) stabilizes well before \(k=20\). A simple convergence-based halting rule (with a short “patience” window) can therefore:

1) reduce the average number of generated refinement steps by \(\ge 2\times\), while
2) keeping retrieval quality close to the fixed-K reference, and
3) achieving a better quality-at-matched-compute point than any compute-matched fixed-K baseline.

This can fail if embedding “stability” (e.g., high cosine similarity between consecutive \(z_k\)) does not imply stable nearest-neighbor rankings, or if refinement dynamics are oscillatory.

---

## Proposed Approach

### Overview

We propose **ConvergeStop**, an inference-time halting rule for generative text embedders that iteratively generate refinement tokens.

Given a maximum budget \(K_{max}\) (e.g., 20 for GIRCSE), ConvergeStop runs the standard embedding generation loop but stops early when the intermediate embedding becomes stable for \(M\) consecutive steps.

### Method Details

#### Intermediate embedding trajectory

Consider a generative embedder that produces a sequence of refinement steps \(k=1..K\), and an embedding \(z_K\) by pooling the last-layer hidden states of the generated refinement tokens (as in **[GIRCSE](./references/LET-LLMS-SPEAK-EMBEDDING-LANGUAGES-GENERATIVE-TEXT-EMBEDDINGS-VIA-ITERATIVE-CONTRASTIVE-REFINEMENT/meta/meta_info.txt)**).

Define \(g_k\) as the last-layer hidden state corresponding to the \(k\)-th generated refinement token. Define the intermediate embedding

\[
\tilde z_k = \frac{1}{k}\sum_{i=1}^k g_i,\qquad z_k = \frac{\tilde z_k}{\|\tilde z_k\|_2}.
\]

Because the backbone is causal, \(g_i\) for \(i<k\) does not change when generating the \(k\)-th refinement token, so \(\tilde z_k\) can be maintained as a running mean.

#### Stability score

Define a per-step stability score

\[
 s_k = \cos(z_k, z_{k-1}).
\]

#### Patience-based halting rule

ConvergeStop halts at

\[
 k^* = \min\{k \ge k_{min} : \min_{i=0..M-1} s_{k-i} \ge \tau\},
\]

and returns \(z_{k^*}\). Defaults (pre-registered):

- \(k_{min}=1\)
- \(M=2\) (do not tune)
- \(K_{max}=20\) for GIRCSE
- \(\tau\) is **derived from a dev stability profile** to avoid hand-picking: \(\tau := \text{median}_{q \in \mathcal{D}_{dev}}\; \cos(z_{19}(q), z_{20}(q))\).

#### Dynamics sanity check (premise test)

Before running the main evaluation, we run an automated premise check on a dev split:

- compute \(s_k\) curves for \(k=2..20\) for a fixed set of queries
- verify that stability increases with k (e.g., median Spearman correlation between k and \(s_k\) is positive)

**Ranking-stability sanity check (addresses “stability ≠ quality” risk).** For a small dev batch (e.g., 200 queries), we also measure whether embedding-space stability predicts **top-K retrieval stability**:

- For each query, embed with \(k\in\{5,10,15,20\}\) and compute the top-10 doc IDs under cosine similarity (using the same fixed document embeddings, e.g., doc K=20).
- Compute Kendall-\(\tau\) or Jaccard@10 overlap between the top-10 sets at \(k\) and at \(K_{ref}=20\).
- Proceed with ConvergeStop only if median top-10 overlap is **monotone non-decreasing in k** and exceeds a minimum threshold by k=10 (pre-registered: Jaccard@10 ≥ 0.7).

If the embedding dynamics are strongly oscillatory, or if cosine stability does **not** correlate with ranking stability, ConvergeStop is refuted for this model/dataset slice (and we pivot to alternative stopping signals, e.g., directly monitoring retrieval-score stability on a small probe set).

### Key Innovations

1. **Anytime generative embeddings via convergence halting**: turn test-time scaling (choose a larger K) into a per-query adaptive computation policy.
2. **Compute-matched evaluation**: explicitly test whether adaptive halting is *not dominated* by the best fixed-K at the same average compute.
3. **Mechanism signature (difficulty heterogeneity)**: test whether the chosen halting depth \(k^*\) correlates with query difficulty and with the marginal benefit of longer refinement.

---

## Related Work

### Field Overview

**Text embeddings and retrieval benchmarks.** Sentence/document embeddings are commonly trained with contrastive objectives and evaluated on suites such as MTEB and BEIR. The field has progressed from encoder-only models (Sentence-BERT) to instruction-tuned embedders and LLM-derived embedders (E5-Mistral, NV-Embed, GritLM).

**Generative embeddings and test-time scaling.** GIRCSE introduces iterative refinement embeddings and empirically shows that increasing generation length K at inference time can improve embedding quality, suggesting a test-time scaling knob for embeddings (**[GIRCSE](./references/LET-LLMS-SPEAK-EMBEDDING-LANGUAGES-GENERATIVE-TEXT-EMBEDDINGS-VIA-ITERATIVE-CONTRASTIVE-REFINEMENT/meta/meta_info.txt)**). GRACE trains a rationale-generating embedding policy with reinforcement learning and highlights that generation dominates inference latency (**[GRACE](./references/Grace-Generative-Representation-Learning-via-Contrastive-Policy-Optimization/meta/meta_info.txt)**).

**Adaptive computation.** Early-exit and adaptive-depth methods (ACT, PonderNet, FastBERT, PABEE) primarily target classification or sequence prediction, where “confidence” is defined on discrete outputs. Our setting differs because the output is a continuous embedding used for nearest-neighbor retrieval, where small embedding changes can still change rankings.

### Related Papers

- **[Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement (GIRCSE)](./references/LET-LLMS-SPEAK-EMBEDDING-LANGUAGES-GENERATIVE-TEXT-EMBEDDINGS-VIA-ITERATIVE-CONTRASTIVE-REFINEMENT/meta/meta_info.txt)**: Generates auxiliary soft tokens to iteratively refine embeddings and shows test-time scaling with longer K.
- **[GRACE: Generative Representation Learning via Contrastive Policy Optimization](./references/Grace-Generative-Representation-Learning-via-Contrastive-Policy-Optimization/meta/meta_info.txt)**: Uses policy optimization to train rationale-generating embedders; reports that generation time dominates latency.
- **[Generative Representational Instruction Tuning (GRIT)](./references/Generative-Representational-Instruction-Tuning/meta/meta_info.txt)**: Unifies embedding and generation in one model; enables practical RAG optimizations via shared backbone.
- **[Sentence-BERT](https://arxiv.org/abs/1908.10084)**: Siamese sentence embeddings enabling efficient semantic search.
- **[SimCSE](https://arxiv.org/abs/2104.08821)**: Contrastive sentence embeddings with dropout-based augmentation; widely used baseline family.
- **[MTEB](https://arxiv.org/abs/2210.07316)**: Standard multi-task evaluation suite for embedding models.
- **[BEIR](https://arxiv.org/abs/2104.08663)**: Heterogeneous retrieval benchmark suite for dense retrievers.
- **[INSTRUCTOR](https://arxiv.org/abs/2212.09741)**: Instruction-conditioned embeddings trained across tasks.
- **[Text Embeddings by Weakly-Supervised Contrastive Pre-Training (E5)](https://arxiv.org/abs/2212.03533)**: Weakly-supervised contrastive pretraining recipe for strong retrievers.
- **[Improving Text Embeddings with Large Language Models (E5-Mistral)](https://arxiv.org/abs/2401.00368)**: Instruction-style embedding finetuning of decoder-only LLMs.
- **[NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models](https://arxiv.org/abs/2405.17428)**: Latent-attention pooling and training techniques for LLM embedders.
- **[LLM2Vec](https://arxiv.org/abs/2404.05961)**: Turns decoder-only LLMs into bidirectional embedders via contrastive training.
- **[Answer is All You Need: Instruction-following Text Embedding via Answering the Question (InBedder)](https://aclanthology.org/2024.acl-long.27/)**: Instruction-following embeddings via generating answers; a generative embedding baseline family.
- **[Repetition Improves Language Model Embeddings (Echo Embeddings)](https://arxiv.org/abs/2402.15449)**: Input repetition improves causal LLM embeddings at the cost of more tokens.
- **[Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models](https://arxiv.org/abs/2507.23386)**: Adds a contextual token to improve causal LLM embeddings while keeping causal attention.
- **[KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs](https://arxiv.org/abs/2601.01046)**: Training-free internal key/value manipulation to inject global context into causal embeddings.
- **[Qwen3-Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176)**: Strong embedding and reranking models; illustrates modern embedding SOTA and scale.
- **[EmbeddingGemma: Powerful and Lightweight Text Representations](https://arxiv.org/abs/2509.20354)**: Encoder–decoder adaptation to export a small embedder from an LLM family.
- **[Llama-Embed-Nemotron-8B](https://arxiv.org/abs/2511.07025)**: Bidirectional conversion of decoder-only LLMs for multilingual embeddings (representative of 2025–2026 LLM-derived embedders).
- **[Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)**: Embeddings that remain useful when truncated to smaller dimensions; a different efficiency knob.
- **[ALE: Adaptive Length Embedding for Text Retrieval](https://openreview.net/forum?id=ghHdNfHfev)**: Variable-length (dimension-adaptive) embeddings for faster retrieval.
- **[Adaptive Computation Time](https://arxiv.org/abs/1603.08983)**: Classic learned halting mechanism for recurrent computation.
- **[PonderNet](https://arxiv.org/abs/2107.05407)**: Learned halting with a probabilistic prior over computation steps.
- **[FastBERT](https://arxiv.org/abs/2004.02178)**: Early-exit BERT with self-distillation for adaptive inference.
- **[PABEE](https://arxiv.org/abs/2006.04152)**: Patience-based early exiting for Transformers; motivates our patience-style stability check.
- **[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)**: Fixed-point models; residual norms as convergence signals in iterative inference.
- **[Early Exit Strategies for Approximate k-NN Search in Dense Retrieval](https://arxiv.org/abs/2408.04981)**: Early exit in ANN probing; adaptive compute in the retrieval index rather than the embedder.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Encoder-only contrastive embedders | Bidirectional encoders trained with contrastive losses | SBERT, E5, BGE/GTE | MTEB, BEIR | Limited instruction following; separate generator needed |
| Decoder-only embedders (discriminative) | Use decoder-only LLMs as encoders via pooling / masking tricks | E5-Mistral, NV-Embed, LLM2Vec | MTEB | May break streaming properties if bidirectional inference is used |
| Unified embedding+generation | One model serves as both embedder and generator | GRIT | MTEB + generative benchmarks | May still be expensive vs small specialized embedders |
| Generative embeddings (iterative refinement) | Generate refinement tokens to improve embeddings; test-time scaling with K | GIRCSE | MTEB (+ retrieval subsets) | Longer K increases query-time latency |
| Generative embeddings (rationale policy) | Generate natural-language rationales, then embed | GRACE, InBedder | MTEB, instruction-following evals | Generation dominates latency; rationale length is a major knob |
| Adaptive computation / early exit | Stop early based on confidence/stability | ACT, PonderNet, FastBERT, PABEE | classification/LM tasks | “Confidence” signals are less direct for continuous embeddings |
| Dimension-adaptive embeddings | Adapt embedding dimensionality rather than compute steps | Matryoshka, ALE | retrieval-focused | Orthogonal to generative-step halting |

### Closest Prior Work

1. **GIRCSE** (**[paper](./references/LET-LLMS-SPEAK-EMBEDDING-LANGUAGES-GENERATIVE-TEXT-EMBEDDINGS-VIA-ITERATIVE-CONTRASTIVE-REFINEMENT/meta/meta_info.txt)**): establishes that increasing K at inference improves embedding quality, but does not propose a per-query stopping rule; our method converts fixed-K scaling into an adaptive policy.
2. **GRACE** (**[paper](./references/Grace-Generative-Representation-Learning-via-Contrastive-Policy-Optimization/meta/meta_info.txt)**): demonstrates a quality–latency trade-off driven by generation length, but uses a fixed output budget; our proposal targets the missing “when to stop generating” component.
3. **PABEE / early-exit Transformers**: uses patience-based stability across layers for classification; our proposal adapts the same principle to stability across refinement steps in generative embedding inference.
4. **ALE (Adaptive Length Embedding)**: adaptively shortens embedding dimensionality; we instead adaptively shorten iterative refinement length.

**Novelty Kill Search Summary:** Searched for combinations such as “embedding convergence early stopping generative embeddings”, “GIRCSE early stopping”, “adaptive generation length text embeddings”, and checked for embedding-specific “halting” rules. Also grepped existing proposals for “adaptive K / halting embedding”. No prior work proposing convergence-based early stopping for generative text embedding refinement (GIRCSE/GRACE-style) was found as of 2026-02-16 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| GIRCSE | Improves embeddings by generating K refinement tokens | Fixed K wastes compute on easy queries | Stop early when embedding stabilizes | Adaptive K exploits difficulty heterogeneity |
| GRACE | RL-trained rationale generation before embedding | Generation length fixed; high latency | Apply same “stop when stable” principle to generated rationale/steps | Potentially reduces rationale tokens without retraining |
| PABEE | Patience-based early exit for classification | Not designed for continuous retrieval embeddings | Use patience on embedding updates | Stability-based halting is model-agnostic |
| ALE | Variable-length (dimensionality) embeddings | Doesn’t reduce token generation compute | Halting reduces refinement steps | Directly reduces generative embedder inference |

---

## Experiments

### Experimental Setup

**Goal.** Evaluate whether ConvergeStop is (i) not worse than a strong fixed-K reference, while (ii) using substantially fewer refinement steps on average, and (iii) not dominated by a compute-matched fixed-K baseline.

**Methods compared (3 conditions).**

- **A: Fixed-K reference**: GIRCSE with \(K=K_{ref}=20\) (paper’s inference default).
- **B: Fixed-K compute-matched**: GIRCSE with \(K=K_{budget}\), where \(K_{budget} = \lceil \mathbb{E}_{dev}[k^*] \rceil\) and \(k^*\) is the halting depth produced by ConvergeStop on the dev split.
- **C: ConvergeStop (ours)**: GIRCSE with \(K_{max}=20\), patience \(M=2\), \(k_{min}=1\), and \(\tau\) computed from dev as described above.

**Determinism / seeds.** We use **greedy decoding** for refinement-token generation with dropout disabled, so inference is deterministic. We will still fix RNG seeds for data loading and ANN index construction, but do not expect meaningful variance across runs.

**Baseline Ladder (REQUIRED):**
- **Simple compute baseline**: fixed smaller K (operationally the first thing a practitioner tries; represented by condition B, compute-matched fixed-K).
- **Inference-time scaling baseline**: fixed larger K (condition A).
- **Closest method**: the underlying generative embedder itself (GIRCSE); our contribution is a deployment policy, not a new embedding model.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Mistral-7B-v0.1 | 7B | https://huggingface.co/mistralai/Mistral-7B-v0.1 | Base LM for GIRCSE-Mistral7B adapter |
| GIRCSE-Mistral7B (LoRA adapter) | 7B | https://huggingface.co/Roytsai27/GIRCSE-Mistral7B | LoRA adapter; apply via PEFT |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| N/A | No training; inference-only | - | - | - |

**Resource Estimate** (order-of-magnitude; to be verified during implementation):
- **Compute budget**: dominated by embedding the document corpus once with \(K=20\) (for each dataset, documents are embedded once offline at K=20 and reused across all query-side conditions). For NanoBEIR-scale corpora (SciFact, FiQA2018) this should be a few GPU-hours total. Total expected to be \(<100\) A100 GPU-hours for 2 NanoBEIR datasets (plus fixed-K sweeps).
- **GPU memory**: a single 80GB A100 should be sufficient for 7B inference with batching.
- **API usage**: none required.

### Benchmarks and Metrics

We focus on retrieval benchmarks where query embedding cost is deployment-critical.

**Metric.** We report **nDCG@10** (normalized discounted cumulative gain at rank 10; higher is better).

**Protocol note.** For SciFact and FiQA2018 we use the **NanoBEIR** protocol (Sentence-Transformers `NanoBEIREvaluator`), a downsized subset of BEIR used for quick retrieval evaluation. It computes nDCG@10 on a 0–1 scale but commonly **reports metrics as percentages (×100)**; GIRCSE’s NanoBEIR Table 8 uses this percent-style scale. In all cases, we compare methods within the *same* protocol for fairness.

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SciFact (via NanoBEIR) | Scientific claim retrieval; small corpus | nDCG@10 | test | https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/NanoBEIREvaluator.py | https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/NanoBEIREvaluator.py |
| FiQA2018 (via NanoBEIR) | Financial question answering retrieval | nDCG@10 | test | https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/NanoBEIREvaluator.py | https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/NanoBEIREvaluator.py |

**Efficiency metrics (secondary).**
- Average halting depth \(\mathbb{E}[k^*]\)
- Histogram / quantiles of \(k^*\) (mechanism signature)
- Average number of forward steps per query (approximately \(k^*+1\))
- Optional: wall-clock latency per query embedding on a fixed GPU

### Main Results

#### Results Table

| Method | Base Model | Benchmark | nDCG@10 | Avg. k* (queries) | Source | Notes |
|---|---|---|---:|---:|---|---|
| A: Fixed K=20 | Mistral-7B + GIRCSE LoRA | SciFact | **80.90** | 20 | GIRCSE arXiv HTML v2 (https://arxiv.org/html/2509.24291v2), Table 8 “Performance comparison on NanoBEIR benchmark” | NanoBEIR table uses percent-style scale (≈ nDCG@10×100). Verification will re-run NanoBEIR to confirm exact protocol. |
| B: Fixed K=K_budget | Mistral-7B + GIRCSE LoRA | SciFact | **TBD** | K_budget | - | Compute-matched fixed baseline |
| **C: ConvergeStop (ours)** | Mistral-7B + GIRCSE LoRA | SciFact | **TBD** | **TBD** | - | Adaptive halting |
| A: Fixed K=20 | Mistral-7B + GIRCSE LoRA | FiQA2018 | **60.79** | 20 | GIRCSE arXiv HTML v2 (https://arxiv.org/html/2509.24291v2), Table 8 “Performance comparison on NanoBEIR benchmark” | NanoBEIR table uses percent-style scale (≈ nDCG@10×100). Verification will re-run NanoBEIR to confirm exact protocol. |
| B: Fixed K=K_budget | Mistral-7B + GIRCSE LoRA | FiQA2018 | **TBD** | K_budget | - | Compute-matched fixed baseline |
| **C: ConvergeStop (ours)** | Mistral-7B + GIRCSE LoRA | FiQA2018 | **TBD** | **TBD** | - | Adaptive halting |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (M=2) | Patience \(M=2\) | More robust than single-step stability |
| Ours (M=1) | Patience \(M=1\) | Higher variance in \(k^*\); potentially lower quality due to premature stopping |

### Experimental Rigor

- **Premise check**: Report stability curves \(s_k\) and confirm convergence-like behavior on dev.
- **Fair compute matching**: Define \(K_{budget}\) from dev and keep fixed for test.
- **Mechanism check**: Bucket queries by \(k^*\) (e.g., quartiles) and test whether (i) low-\(k^*\) queries already perform well at small K, and (ii) high-\(k^*\) queries gain more from larger K.
- **Data leakage**: BEIR datasets may overlap with pretraining corpora of the base LLM; we treat evaluation as a standard retrieval benchmark rather than a contamination-free estimate, and focus on relative comparisons under matched models.

---

## Success Criteria

**Hypothesis** (directional): ConvergeStop will match fixed K=20 retrieval quality on average while using substantially fewer refinement steps for many queries, and will not be dominated by a compute-matched fixed-K baseline.

**Decision Rule** (concrete):

Let \(\text{Metric}(\cdot)\) be nDCG@10 on a dataset. Define per-dataset tolerance

\[
\delta_d = 0.2 \times (\text{Metric}(K_{ref}) - \text{Metric}(K=5))
\]

measured on the dev split (with \(K_{ref}=20\)).

- **Proceed** if, on **both** datasets:
  1) \(\text{Metric}(\text{ConvergeStop}) \ge \text{Metric}(K_{ref}) - \delta_d\), and
  2) \(\mathbb{E}[k^*] \le K_{ref}/2\), and
  3) \(\text{Metric}(\text{ConvergeStop}) \ge \text{Metric}(K_{budget})\) (not dominated at matched compute).
- **Pivot** if (1) fails but (3) holds (adaptive is on/near the Pareto frontier but needs a better stability signal), by swapping cosine-stability with an L2-stability criterion or using a monotone “envelope” stability rule.
- **Refute** if ConvergeStop is dominated by the compute-matched fixed-K baseline (fails criterion 3), or if the premise check shows no convergence-like stability curves.

---

## Impact Statement

If ConvergeStop works, practitioners **experimenting with test-time-scaled (generative) embedders** for retrieval (search/RAG) could reduce average query embedding compute and latency without materially degrading retrieval quality, improving the practicality of this emerging paradigm.

**Importance caveat.** Today, most production embedding systems still use single-pass encoders (E5/BGE/GTE/NV-Embed), and GIRCSE-style iterative generative embedders are newly introduced (2025–2026) with modest public adoption (e.g., the official GIRCSE repo and LoRA checkpoints currently have low visible usage). We therefore frame ConvergeStop as (i) a *methodology bet* that makes the paradigm easier to deploy if/when adoption grows, and (ii) a falsifiable diagnostic: if adaptive stopping is dominated by fixed-K, it suggests that embedding-space convergence is not a reliable stopping signal for retrieval.

---

## References

- [Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement](./references/LET-LLMS-SPEAK-EMBEDDING-LANGUAGES-GENERATIVE-TEXT-EMBEDDINGS-VIA-ITERATIVE-CONTRASTIVE-REFINEMENT/meta/meta_info.txt) - Tsai et al., 2025
- [GRACE: Generative Representation Learning via Contrastive Policy Optimization](./references/Grace-Generative-Representation-Learning-via-Contrastive-Policy-Optimization/meta/meta_info.txt) - Sun et al., 2025
- [Generative Representational Instruction Tuning](./references/Generative-Representational-Instruction-Tuning/meta/meta_info.txt) - Muennighoff et al., 2024
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Reimers & Gurevych, 2019
- [SimCSE](https://arxiv.org/abs/2104.08821) - Gao et al., 2021
- [MTEB](https://arxiv.org/abs/2210.07316) - Muennighoff et al., 2022
- [BEIR](https://arxiv.org/abs/2104.08663) - Thakur et al., 2021
- [INSTRUCTOR](https://arxiv.org/abs/2212.09741) - Su et al., 2022
- [E5](https://arxiv.org/abs/2212.03533) - Wang et al., 2022
- [E5-Mistral](https://arxiv.org/abs/2401.00368) - Wang et al., 2024
- [NV-Embed](https://arxiv.org/abs/2405.17428) - Lee et al., 2024
- [LLM2Vec](https://arxiv.org/abs/2404.05961) - BehnamGhader et al., 2024
- [InBedder](https://aclanthology.org/2024.acl-long.27/) - Peng et al., 2024
- [Echo Embeddings](https://arxiv.org/abs/2402.15449) - Springer et al., 2024
- [Causal2Vec](https://arxiv.org/abs/2507.23386) - Lin et al., 2025
- [KV-Embedding](https://arxiv.org/abs/2601.01046) - Tang & Yang, 2026
- [Qwen3-Embedding](https://arxiv.org/abs/2506.05176) - Zhang et al., 2025
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) - Kusupati et al., 2022
- [ALE](https://openreview.net/forum?id=ghHdNfHfev) - Han et al., 2024
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) - Graves, 2016
- [PonderNet](https://arxiv.org/abs/2107.05407) - Banino et al., 2021
- [FastBERT](https://arxiv.org/abs/2004.02178) - Liu et al., 2020
- [PABEE](https://arxiv.org/abs/2006.04152) - Zhou et al., 2020
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) - Bai et al., 2019
- [Early Exit Strategies for Approximate k-NN Search in Dense Retrieval](https://arxiv.org/abs/2408.04981) - Busolin et al., 2024
