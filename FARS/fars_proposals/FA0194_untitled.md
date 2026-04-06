# untitled

# Deterministic (LLM-Free) Memory Fusion for FadeMem-Style Long-Horizon Conversational Agents

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Long-horizon conversational agents (e.g., personal assistants that interact with users over days or months) cannot rely on a single fixed context window. A common engineering pattern is to maintain an external memory store: (i) write new interaction snippets into memory, (ii) retrieve a small set of relevant memories for each user query, and (iii) answer using the retrieved memory as additional context.

Recent systems report large gains on long-term conversational memory benchmarks, but many rely on *additional* LLM calls to maintain the memory store itself (e.g., to merge redundant memories, resolve conflicts, or verify information preservation). This creates three deployment pain points in 2026: (1) **cost/latency** due to extra LLM calls beyond answer generation, (2) **non-determinism** and brittle behavior due to hidden prompt+model changes, and (3) **audit difficulty** because the memory store becomes the output of an LLM summarizer rather than a transparent transformation of the original conversation.

FadeMem is a representative recent example: it adds biologically-inspired forgetting (dual-layer decay) and achieves strong LoCoMo QA performance, but its biggest ablation gain comes from an **LLM-guided memory fusion** module that merges temporally/semantically related memories and uses an LLM “preservation check” to accept/reject fusions.

### The Problem

**Problem:** Are LLM calls actually necessary for the *fusion operator* in long-horizon conversational memory, or is FadeMem’s fusion gain primarily coming from a simpler effect—reducing retrieval noise by deduplicating and packing verbatim evidence into a fixed retrieval-time token budget?

This question is not cosmetic: if LLM-based fusion is unnecessary, practitioners can implement a deterministic, auditable, and cheaper memory-maintenance pipeline while preserving benchmark quality.

The LoCoMo benchmark is a good testbed because its QA task was designed to allow **deterministic automated scoring**: ground-truth answers are taken from the conversation “as much as possible”, and evaluation uses a normalized **F1 partial-match** metric rather than an LLM judge ([LoCoMo](./references/Evaluating-Very-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt), §4.1).

Key prior work illustrating the current landscape:
- **[FadeMem](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt)** (LLM-guided fusion + conflict resolution): LoCoMo F1=29.43, but w/o fusion drops to 13.63 (−53.7%) (Table 3; Ablation §3.5).
- **[FluxMem / “Choosing How to Remember”](./references/Choosing-How-to-Remember-Adaptive-Memory-Structures-for-LLM-Agents/meta/meta_info.txt)** learns memory *structure* selection and replaces fixed similarity thresholds with a probabilistic (Beta mixture) merge criterion, but still uses LLM-heavy memory formation.
- **[EverMemOS](./references/EverMemOS-A-Self-Organizing-Memory-Operating-System-for-Structured-Long-Horizon-Reasoning/meta/meta_info.txt)** achieves strong LoCoMo judge accuracy via structured MemCells and MemScenes, but similarly relies on many LLM-mediated stages.

### Key Insight and Hypothesis

**Key insight:** On LoCoMo, the metric rewards reproducing *verbatim* conversational facts. This suggests that the primary value of “fusion” may be **information packing and redundancy removal** (so that retrieval returns fewer, denser, less noisy memories), not paraphrastic rewriting or high-level abstraction. If so, a deterministic fusion operator that (i) preserves verbatim spans and (ii) enforces an explicit token budget should recover most of FadeMem’s fusion gain.

**Hypothesis:** Replacing FadeMem’s LLM-guided fusion + LLM preservation check with a **deterministic, quote-preserving, budgeted fusion** (extractive deduplication + MMR-style sentence selection, where **MMR = maximal marginal relevance**—a standard greedy objective that trades off relevance vs redundancy + deterministic coverage check) will yield LoCoMo multi-hop F1 that is statistically indistinguishable from the LLM-fusion baseline under the same retrieval-time token budget.

Why this could be wrong: LLM fusion might be doing more than deduplication (e.g., rewriting memories into query-aligned phrasing, resolving coreference/temporal expressions, or preserving causal links that extractive rules miss). If this is true, deterministic fusion will regress toward the “w/o fusion” ablation regime.

---

## Proposed Approach

### Overview

We propose **DeterministicFadeMem-Fusion (DFM-Fusion)**: keep FadeMem’s forgetting/decay, retrieval, and answer generation unchanged, but replace only the fusion operator with a fully deterministic algorithm that:

1. **Identifies fusion candidates** using the same temporal-semantic clustering rule as FadeMem.
2. **Fuses by extractive packing**: selects a subset of verbatim sentences/spans from the cluster under a strict token budget, removing near-duplicates.
3. **Validates information preservation without an LLM**: uses a deterministic coverage test on salient tokens (numbers, capitalized entities, rare tokens) to reject unsafe fusions.

The goal is not to beat FadeMem’s accuracy, but to test whether LLM-based fusion is *necessary* for the gains FadeMem attributes to fusion.

### Method Details

**(A) Memory representation and forgetting (kept identical to FadeMem).** Each memory item is represented as \(m_i(t)=(c_i, s_i, v_i(t), \tau_i, f_i)\) where \(c_i\) is an embedding, \(s_i\) is text, \(v_i(t)\in[0,1]\) is strength, \(\tau_i\) is timestamp, and \(f_i\) is access frequency ([FadeMem](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt), §2.1–2.2). Dual-layer assignment and decay follow FadeMem’s equations and hyperparameters (\(\lambda_{base}=0.1\), \(\theta_{fusion}=0.75\), etc.).

**(B) Fusion candidate identification (kept identical to FadeMem).** For each candidate memory \(m_k\), form a cluster:
\[
C_k = \{m_i : \text{sim}(c_i,c_k) > \theta_{fusion} \wedge |\tau_i-\tau_k| < T_{window}\}.
\]
This matches FadeMem’s temporal-semantic clustering ([FadeMem](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt), §2.4).

**(C) Deterministic quote-preserving fusion operator (ours).** For a cluster \(C\) that exceeds a minimum size (same trigger as FadeMem), we produce fused text \(s_{fused}\) as follows:

1. **Sentence segmentation**: split each \(s_i\) into sentences/spans (rule-based splitter; no model).
2. **Deduplicate near-duplicates**: greedily remove sentences whose embedding similarity to any kept sentence exceeds \(\theta_{dup}\) (e.g., 0.90), keeping the sentence from the memory with larger \(v_i(t)\) (or more recent \(\tau_i\) as tie-break).
3. **Budgeted packing via deterministic MMR**: select remaining sentences with a deterministic greedy Maximum Marginal Relevance objective that prefers (a) higher-strength sources and (b) diversity among sentences, until reaching a per-fused-item token budget \(B_{fuse}\).
4. **Emit fused memory** as newline-separated verbatim sentences, ordered by timestamp of their source memory.

**(D) Deterministic preservation check (replaces FadeMem’s LLM verification).** FadeMem accepts/rejects a fusion using an LLM “information preservation” check with threshold \(\theta_{preserve}\) (§2.4). We replace this with a deterministic check:

- Extract a set of **salient tokens** from the cluster: all numbers, all capitalized wordpieces, and top-\(K\) TF-IDF tokens (where **TF-IDF = term frequency–inverse document frequency**, a standard heuristic for identifying rare but informative words in the conversation).
- Compute **coverage recall**: fraction of salient tokens appearing in \(s_{fused}\).
- If recall < \(\theta_{cov}\), reject fusion and fall back to a safe alternative: concatenate the top-\(n\) highest-strength original memories in \(C\) (truncated to \(B_{fuse}\)).

**(E) Embeddings for fused items (control to reduce retrieval confounds).** To avoid confounding retrieval changes from different fused texts, we store fused embeddings as the L2-normalized **mean of constituent embeddings** (optionally weighted by \(v_i(t)\)). This is applied to both the LLM-fusion baseline and DFM-Fusion.

**(F) Retrieval-time token budget control (critical).** “Same memory budget” is defined as the **total tokens appended to the answer model per query**, \(B_{ret}\). To avoid the confound that fused items are longer and get truncated more, we:

- retrieve top-\(k\) items,
- truncate **each retrieved item** to \(\lfloor B_{ret}/k\rfloor\) tokens before concatenation,
- log the fraction of items truncated for each method.

### Key Innovations

- **A deterministic, quote-preserving fusion operator** explicitly targeted to LoCoMo’s verbatim-answer evaluation protocol.
- **A non-LLM preservation criterion** (salient-token coverage) that makes “fusion safety” auditable and reproducible.
- **A retrieval-budget-controlled evaluation** that isolates fusion quality from token-length confounds.

---

## Related Work

### Field Overview

Research on long-horizon agent memory for LLMs spans several axes: (i) *what is stored* (raw turns, extracted facts, events, graphs), (ii) *how memory is maintained* (append-only, update/delete, decay/forgetting, consolidation/fusion), and (iii) *how memory is retrieved* (dense, sparse, hybrid, graph traversal, query planning). A recurring empirical theme is that naively increasing context length does not solve long-horizon memory due to attention degradation and noise in long contexts (e.g., “lost in the middle”).

A second theme is that many recent “memory systems” are in fact pipelines that invoke a strong LLM multiple times: to extract memory, rewrite or consolidate it, validate consistency, and plan retrieval. This can improve benchmark scores, but makes the memory store itself opaque and expensive.

Our proposal focuses narrowly on one module that appears important in FadeMem: **fusion**. We ask whether the benefit attributed to LLM fusion can be reproduced by a deterministic operator that preserves verbatim evidence under a fixed token budget.

### Related Papers

- **[FadeMem](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt)**: Introduces dual-layer decay and uses LLM-guided conflict resolution and fusion; provides a strong ablation showing fusion is critical on LoCoMo.
- **[LoCoMo](./references/Evaluating-Very-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)**: Defines long-term conversational memory tasks and a deterministic F1 evaluation protocol for QA with answers drawn verbatim from conversations.
- **[Mem0](./references/Mem0-Building-Production-Ready-AI-Agents-with-Scalable-Long-Term-Memory/meta/meta_info.txt)**: Uses LLM function-calling to add/update/delete memory facts; reports LoCoMo results under both F1 and LLM-judge protocols.
- **[EverMemOS](./references/EverMemOS-A-Self-Organizing-Memory-Operating-System-for-Structured-Long-Horizon-Reasoning/meta/meta_info.txt)**: Introduces MemCell/MemScene lifecycle and necessity-sufficiency retrieval; strong LoCoMo judge accuracy but LLM-heavy memory construction.
- **[Choosing How to Remember / FluxMem](./references/Choosing-How-to-Remember-Adaptive-Memory-Structures-for-LLM-Agents/meta/meta_info.txt)**: Learns to select among linear/graph/hierarchical memory structures; replaces fixed fusion thresholds with a Beta-mixture posterior criterion.
- **[SimpleMem](./references/SimpleMem-Efficient-Lifelong-Memory-for-LLM-Agents/meta/meta_info.txt)**: Uses storage-time semantic compression and recursive consolidation; reports strong LoCoMo F1 with large token savings.
- **[TiMem](./references/TiMem-Temporal-Hierarchical-Memory-Consolidation-for-Long-Horizon-Conversational-Agents/meta/meta_info.txt)**: Builds a temporal memory tree with instruction-guided consolidation and query planning/gating for retrieval depth.
- **[MemWeaver](./references/MemWeaver-Weaving-Hybrid-Memories-for-Traceable-Long-Horizon-Agentic-Reasoning/meta/meta_info.txt)**: Combines temporal KGs, experience abstractions, and passage evidence; shows strong LoCoMo gains and emphasizes traceability.
- **[EMem](./references/A-Simple-Yet-Strong-Baseline-for-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)**: Proposes event-centric EDUs and graph retrieval; highlights that lossy compression can hurt long-term QA.
- **[Zep / Graphiti](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt)**: Temporal knowledge graph memory with edge invalidation; strong on LongMemEval and temporal reasoning.
- **[Memory OS of AI Agent](https://arxiv.org/abs/2506.06326)**: OS-inspired memory management framing for LLM agents.
- **[A-Mem](https://arxiv.org/abs/2502.12110)**: Agentic atomic memory that evolves via LLM-driven operations.
- **[MemoryBank](https://arxiv.org/abs/2305.10250)**: Early long-term personalized memory bank for dialogue agents.
- **[Generative Agents](https://arxiv.org/abs/2304.03442)**: Demonstrates reflection and memory summaries for simulated agents; motivates consolidation but uses LLM generation.
- **[HippoRAG](https://arxiv.org/abs/2405.14831)**: Neuro-inspired graph retrieval using personalized PageRank over entity–passage graphs.
- **[GraphRAG](https://arxiv.org/abs/2404.16130)**: Uses community detection and hierarchical summaries over entity graphs for retrieval-augmented QA.
- **[Lost in the Middle](https://arxiv.org/abs/2307.03172)**: Shows long-context models underutilize middle context, motivating selective retrieval/compression.
- **[RAG Survey](https://arxiv.org/abs/2312.10997)**: Surveys retrieval-augmented generation methods and retrieval/re-ranking practices.
- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)**: An LLM hallucination detection method used in some memory systems for factual-consistency checking.
- **[Memory-R1](https://arxiv.org/abs/2502.04301)**: Uses reinforcement learning to manage memory operations for agents.
- **[ENGRAM](https://arxiv.org/abs/2409.15796)**: Lightweight memory orchestration for conversational agents (memory selection/organization).
- **[SGMem](https://arxiv.org/abs/2406.15939)**: Sentence graph memory that connects sentences across sessions for retrieval.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Forgetting + fusion in external memory | Maintain a memory store with decay and consolidate redundant items | FadeMem | LoCoMo (F1), MSC, synthetic long-term | Fusion/conflict often uses LLM calls; fusion text is opaque |
| LLM-mediated memory CRUD | Extract facts and update memory via LLM function calls | Mem0, A-Mem | LoCoMo (F1 + judge), DMR | Expensive; non-deterministic updates |
| Hierarchical/temporal consolidation | Build multi-level summaries/personas over time | TiMem, EverMemOS, SimpleMem | LoCoMo (often judge + F1), LongMemEval | Consolidation often uses LLM generation and thresholds |
| Structured (graph) memory | Store entities/relations/events and traverse for retrieval | Zep, MemWeaver, EMem, HippoRAG | LongMemEval, LoCoMo | Requires extraction/normalization; often LLM-heavy |
| Long-context-only baselines | Rely on longer context windows without external memory | Fixed-16K / full-context | LoCoMo | Suffers from noise + attention degradation |

### Closest Prior Work

**FadeMem** ([meta](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt)). FadeMem introduces a dual-layer decay memory with LLM-guided conflict resolution and LLM-guided fusion, plus an LLM information-preservation check. It reports that removing fusion drops LoCoMo F1 from 29.43 to 13.63 (Ablation §3.5), making fusion its largest contributor. **Our difference** is to keep FadeMem’s architecture but replace the fusion operator and preservation check with fully deterministic, quote-preserving procedures.

**SimpleMem** ([meta](./references/SimpleMem-Efficient-Lifelong-Memory-for-LLM-Agents/meta/meta_info.txt)). SimpleMem performs storage-time semantic compression (coreference/time normalization) and recursive consolidation, achieving strong LoCoMo F1 with large token savings. **Our difference** is narrower: we do not redesign indexing/retrieval; we test whether the *fusion step alone* needs an LLM, using a deterministic operator and an auditable preservation metric.

**FluxMem / Choosing How to Remember** ([meta](./references/Choosing-How-to-Remember-Adaptive-Memory-Structures-for-LLM-Agents/meta/meta_info.txt)). FluxMem replaces fixed similarity thresholds for merge decisions with a probabilistic Beta-mixture criterion and adapts memory structure, but still uses LLM-based memory formation and does not propose an LLM-free fusion operator for memory content. **Our difference** is to remove LLM generation from fusion itself.

**TiMem** ([meta](./references/TiMem-Temporal-Hierarchical-Memory-Consolidation-for-Long-Horizon-Conversational-Agents/meta/meta_info.txt)). TiMem’s gains come from hierarchical consolidation and query planning/gating, implemented via LLM prompts. **Our difference** is to avoid adding new LLM middleware and instead test a deterministic fusion hypothesis inside an existing decay-based memory.

**EMem** ([meta](./references/A-Simple-Yet-Strong-Baseline-for-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)). EMem argues against lossy compression and instead stores enriched event-level EDUs, relying on retrieval-time LLM filtering. **Our difference** is not to change the stored representation, but to ask whether redundancy-removal fusion can be done deterministically without losing the verbatim facts LoCoMo requires.

**Novelty Kill Search Summary:** Searched for combinations of “deterministic fusion + LoCoMo”, “LLM-free memory fusion LoCoMo”, “quote-preserving memory fusion”, and “extractive memory consolidation for conversational agents” (plus OpenReview queries for “memory fusion conversational agent”). No prior work was found that (i) targets FadeMem-style fusion specifically and (ii) replaces the fusion operator and fusion acceptance check with a fully deterministic, quote-preserving algorithm evaluated under LoCoMo’s deterministic F1 protocol (as of 2026-02-20). Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| FadeMem | Decay-based dual-layer memory + LLM conflict + LLM fusion | Fusion is expensive/opaque; LLM preservation check is non-auditable | Replace fusion + preservation check with deterministic extractive fusion + coverage check | If fusion gains are mostly noise reduction, LLM rewriting is unnecessary |
| SimpleMem | Storage-time compression + recursive consolidation | Consolidation uses LLM generation; more pipeline complexity | Keep retrieval/indexing fixed; only replace fusion operator | More isolated test of whether LLM fusion is needed |
| FluxMem | Adaptive memory structures + probabilistic merge gating | Still uses LLM memory formation; merge affects structure not content operator | Deterministic content fusion in FadeMem-style clusters | Directly targets the expensive content rewrite step |
| TiMem | Temporal memory tree + LLM recall planning/gating | Extra LLM middleware; hard to make fully deterministic | No new LLM planners; deterministic fusion only | Lower latency and simpler audit surface |
| EMem | Event-level EDU storage + LLM filtering/graph retrieval | Requires LLM extraction/filtering; different representation | Keep representation but make fusion deterministic | Quote-preservation matches LoCoMo evaluation needs |

---

## Experiments

### Experimental Setup

**Task framing.** We implement a FadeMem-style memory store for each LoCoMo conversation, then answer each benchmark QA query using a fixed answer model with retrieved memory appended.

**Main conditions (≤3):**
1. **LLM-Fusion (FadeMem-style)**: Temporal-semantic clustering + LLM fusion + LLM preservation check (as described in FadeMem §2.4).
2. **DeterministicFusion (DFM-Fusion, ours)**: Same clustering and forgetting, but deterministic quote-preserving fusion + deterministic coverage check.
3. **No-Fusion**: Same clustering and forgetting, but never merge memories (this matches FadeMem’s “w/o Fusion” ablation concept).

All other components are held constant: embeddings, decay, pruning, conflict handling, retriever, answer model, decoding settings, and retrieval-time token budget.

**Baseline Ladder (REQUIRED):**
- **Prompting / long-context baseline**: Fixed-16K FIFO context (published in FadeMem Table 3).
- **Strongest existing method**: FadeMem with LLM fusion (published; and also re-run in our harness).
- **Ablation baseline**: w/o Fusion (published; and re-run in our harness).

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| gpt-4o-mini | API | (available via platform) | Used for answer generation and for LLM-fusion baseline’s fusion/preservation prompts (temp=0) |

**Training Data (if applicable):**

No training data needed – inference only.

**Resource Estimate**:
- **Compute budget**: 0 GPU-hours (API-based inference) + optional ≤50 GPU-hours if running embeddings locally on a single A100.
- **API usage** (order-of-magnitude):
  - Answer generation: ~1,540 LoCoMo questions → ~1,540 calls.
  - LLM-Fusion baseline: additional calls for fusion + preservation checks, roughly proportional to #fusion events (expected O(1k–5k) calls on LoCoMo10 depending on thresholds).
  - DeterministicFusion: no additional LLM calls beyond answer generation.
- **Wall-clock**: With modest parallelism, expected hours-scale for LLM-Fusion; DeterministicFusion reduces this substantially.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| LoCoMo | Long-horizon conversational memory benchmark with multi-hop, temporal, open-domain, and single-hop QA; ground-truth answers are drawn from the conversation | **Multi-hop F1** (primary), overall F1 (secondary), retrieval recall@k (diagnostic) | test | https://github.com/snap-research/locomo | Official repo + deterministic F1 computation described in LoCoMo §4.1 |

### Main Results

**Published reference numbers (FadeMem Table 3; LoCoMo multi-hop F1 protocol):**

| Method | Base Model | Benchmark | LoCoMo F1 ↑ | FCR ↑ (Factual Consistency Rate) | SRR ↑ (Storage Reduction Rate) | Source | Notes |
|--------|------------|-----------|-------------|-------|-------|--------|-------|
| Fixed-16K | GPT-4o-mini | LoCoMo | 5.17 | 78.9% | 0.00 | [FadeMem §3.4](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.4 Cross-Dataset Evaluation.md>) | Published |
| LangChain | GPT-4o-mini | LoCoMo | 25.75 | 81.2% | 0.00 | [FadeMem §3.4](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.4 Cross-Dataset Evaluation.md>) | Published |
| Mem0 | GPT-4o-mini | LoCoMo | 28.37 | 83.6% | 0.00 | [FadeMem §3.4](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.4 Cross-Dataset Evaluation.md>) | Published |
| FadeMem (LLM-Fusion) | GPT-4o-mini | LoCoMo | 29.43 | 85.9% | 0.45 | [FadeMem §3.4](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.4 Cross-Dataset Evaluation.md>) | Published |
| FadeMem w/o Fusion | GPT-4o-mini | LoCoMo | 13.63 | N/A | N/A | [FadeMem §3.5](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.5 Ablation Study.md>) | Published ablation |

**Verification table (to be filled by experiments in this proposal; all methods run in the same harness):**

| Method | Base Model | Benchmark | Multi-hop F1 (mean±std) | Source | Notes |
|--------|------------|-----------|--------------------------|--------|-------|
| LLM-Fusion (FadeMem-style) | gpt-4o-mini | LoCoMo | **TBD** | - | Run 3 seeds / 3 runs; temp=0 |
| DeterministicFusion (ours) | gpt-4o-mini | LoCoMo | **TBD** | - | Run 3 seeds / 3 runs; temp=0 |
| No-Fusion | gpt-4o-mini | LoCoMo | **TBD** | - | Run 3 seeds / 3 runs; temp=0 |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| DeterministicFusion w/o coverage check | Always accept deterministic fusion | If performance drops, coverage check prevents destructive merges |
| DeterministicFusion (per-entry truncation off) | Use post-concat truncation only | If performance drops, per-entry truncation is an important control against length confounds |

### Experimental Rigor

**Variance & Reproducibility:**
- Primary runs use temperature=0 for all LLM calls; non-determinism is still possible across API calls.
- Report mean±std over 3 runs (treated as “seeds”) for each main condition.

**Confounders and controls:**
- **Token budget confound**: Fused items may be longer and more often truncated. Control with per-entry truncation and log truncation rates.
- **Retrieval confound from embeddings**: Different fusion texts could change embeddings. Control by using mean-of-constituents embeddings for all fused items.
- **Evaluation protocol mismatch**: Some papers report LoCoMo with LLM-judge accuracy. We use deterministic F1 per LoCoMo §4.1.

**Sanity checks:**
- Reproduce FadeMem’s large fusion gap directionally (LLM-Fusion > No-Fusion) in our harness; if not, the harness is invalid.
- Random retrieval baseline (retrieve random memories) should perform near the Fixed-16K baseline.

---

## Success Criteria

**Hypothesis** (directional): DeterministicFusion will close most of the No-Fusion→LLM-Fusion gap, yielding LoCoMo multi-hop F1 close to LLM-Fusion under the same retrieval budget.

**Decision Rule** (concrete):
- **Continue/Proceed** if DeterministicFusion is **not statistically worse** than LLM-Fusion on per-question multi-hop F1 (paired bootstrap 95% CI includes 0 difference, or paired t-test p≥0.05), and both beat No-Fusion by a statistically distinguishable margin.
- **Pivot** if DeterministicFusion is worse than LLM-Fusion but still significantly better than No-Fusion; then iterate on the deterministic operator (e.g., add rule-based coreference/time normalization from SimpleMem) while keeping the same evaluation harness.
- **Refute** if DeterministicFusion is statistically significantly worse than LLM-Fusion and close to No-Fusion (i.e., fails to recover the fusion gain), implying that LLM rewriting/normalization is necessary for this benchmark.

---

## Impact Statement

If deterministic fusion matches LLM fusion on LoCoMo, practitioners building long-horizon conversational agents can remove a major class of extra LLM calls for memory maintenance, reducing cost and improving reproducibility/auditability. If it fails, the result is still decision-changing: it provides concrete evidence that LLM-based consolidation is not merely an engineering convenience but a necessary component for this benchmark, motivating investment in smaller specialized fusion models or better deterministic normalization.

---

## References

- [FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt) - Wei et al., 2026
- [Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo)](./references/Evaluating-Very-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt) - Maharana et al., 2024
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](./references/Mem0-Building-Production-Ready-AI-Agents-with-Scalable-Long-Term-Memory/meta/meta_info.txt) - Chhikara et al., 2025
- [EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning](./references/EverMemOS-A-Self-Organizing-Memory-Operating-System-for-Structured-Long-Horizon-Reasoning/meta/meta_info.txt) - Hu et al., 2026
- [Choosing How to Remember: Adaptive Memory Structures for LLM Agents](./references/Choosing-How-to-Remember-Adaptive-Memory-Structures-for-LLM-Agents/meta/meta_info.txt) - Lu et al., 2026
- [SimpleMem: Efficient Lifelong Memory for LLM Agents](./references/SimpleMem-Efficient-Lifelong-Memory-for-LLM-Agents/meta/meta_info.txt) - Liu et al., 2026
- [TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](./references/TiMem-Temporal-Hierarchical-Memory-Consolidation-for-Long-Horizon-Conversational-Agents/meta/meta_info.txt) - Li et al., 2026
- [MemWeaver: Weaving Hybrid Memories for Traceable Long-Horizon Agentic Reasoning](./references/MemWeaver-Weaving-Hybrid-Memories-for-Traceable-Long-Horizon-Agentic-Reasoning/meta/meta_info.txt) - Ye et al., 2026
- [A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents (EMem)](./references/A-Simple-Yet-Strong-Baseline-for-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt) - Zhou & Han, 2025
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt) - Rasmussen et al., 2025
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - Liu et al., 2024
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) - Gao et al., 2024
- [GraphRAG](https://arxiv.org/abs/2404.16130) - Edge et al., 2024
- [HippoRAG](https://arxiv.org/abs/2405.14831) - Gutiérrez et al., 2024
- [Generative Agents](https://arxiv.org/abs/2304.03442) - Park et al., 2023
- [MemoryBank](https://arxiv.org/abs/2305.10250) - Zhong et al., 2023
- [A-Mem](https://arxiv.org/abs/2502.12110) - Xu et al., 2025
- [Memory OS of AI Agent](https://arxiv.org/abs/2506.06326) - Kang et al., 2025
- [SelfCheckGPT](https://arxiv.org/abs/2303.08896) - Manakul et al., 2023
- [MemGPT](https://arxiv.org/abs/2310.08560) - Packer et al., 2023
- [SGMem](https://arxiv.org/abs/2406.15939) - (Sentence graph memory), 2024
- [ENGRAM](https://arxiv.org/abs/2409.15796) - (Memory orchestration), 2024
- [Memory-R1](https://arxiv.org/abs/2502.04301) - (RL for memory operations), 2025
