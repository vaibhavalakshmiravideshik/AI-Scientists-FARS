# untitled

# Self-Anchored Temporal Filtering: LLM-Free Time Windowing for LongMemEval Retrieval

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Long-horizon chat assistants and LLM agents increasingly rely on an external memory subsystem (typically a key–value store plus retrieval) because user–assistant interaction histories quickly exceed the finite context window of a language model. In these systems, the agent must retrieve a small set of past sessions or turns that contain the evidence needed to answer a question.

A recurring failure mode is **temporal mismatch**: for questions like “Which restaurant did you recommend last weekend?”, a retrieval system may return semantically similar but temporally wrong sessions (e.g., restaurant recommendations from other months). This is especially problematic in long-term personalization where user states evolve over time.

The **LongMemEval** benchmark explicitly measures this challenge. It provides multi-session chat histories with timestamps and evaluates retrieval quality using **Recall@k** (whether all gold evidence sessions/turns are retrieved in the top-k) and **NDCG@k** (a rank-sensitive relevance metric) on a temporal reasoning subset.

### The Problem

LongMemEval proposes **time-aware query expansion**: an LLM extracts a date range from the question and filters retrieval candidates to sessions whose (pre-extracted) event dates fall in that range. This works well when the time-range extractor is strong (GPT-4o), but fails when the extractor is weaker (e.g., Llama 3.1 8B) due to hallucinated or missed temporal cues.

Concretely, on the LongMemEvalM temporal subset with a strong indexing design (Key = Value + extracted facts), LongMemEval reports for **Value = Round** (Table 3):
- Baseline retrieval: Recall@10 = 0.550, NDCG@10 = 0.598
- + GPT-4o time-aware query expansion: Recall@10 = 0.722, NDCG@10 = 0.669
- + Llama 3.1 8B time-aware query expansion: Recall@10 = 0.570, NDCG@10 = 0.647

This creates a practical gap: temporal filtering improves retrieval, but the standard implementation requires an additional strong LLM call per query (cost/latency) and is not robust when only smaller models are available.

### Key Insight and Hypothesis

**Key insight:** even without understanding the question’s temporal expression (“last weekend”, “in early spring”), the *timestamps of the initial retrieval results themselves* may reveal a time period where the evidence likely lies. Many temporal questions contain topical cues (entities/activities) that make baseline dense retrieval surface partially relevant candidates; if those candidates are temporally clustered, we can infer a time window from their timestamp distribution.

**Hypothesis:** A simple **timestamp peak detector** over the top-N retrieved items, combined with a conservative **confidence gate** (do nothing when timestamps are diffuse), can recover a meaningful fraction of GPT-4o’s temporal-filtering gains on LongMemEval temporal retrieval **without any LLM-based time parsing**.

This could fail if baseline retrieval results are temporally uninformative (no peak near the gold evidence), in which case time parsing must be done from the query semantics (as in LongMemEval’s GPT-4o baseline).

---

## Proposed Approach

### Overview

We propose **Self-Anchored Temporal Filtering (SATF)**, an LLM-free alternative to LLM-extracted time-range filtering for conversational memory retrieval.

SATF takes a standard retrieval run (dense retrieval over session/round values) and reorders the retrieved list by promoting items whose timestamps fall inside a **self-inferred** time window. The time window is inferred from the timestamp distribution of the top-N results of the same retrieval run, and SATF is applied only when the distribution is sufficiently concentrated.

### Method Details

We assume the LongMemEval setting where each history session has a timestamp (from `haystack_dates`) and retrieval returns a ranked list of memory items (sessions or rounds) with similarity scores.

**Inputs per query**:
- Ranked retrieval results `[(id_i, s_i)]_{i=1..N}` from a base retriever (e.g., Stella embeddings + cosine similarity).
- A mapping from each retrieved item `id_i` to its parent session index `j_i` (for round-level values, extract the session id embedded in `corpus_id`).
- Chronological session order `j = 1..|S|` (LongMemEval sessions are timestamp-sorted).

**Step 1: Build a weighted temporal signal.**
For each session `j`, accumulate weight from retrieved items belonging to that session:

`w_j = Σ_{i: j_i = j} exp(s_i / τ)`

where `τ` is a temperature (we set `τ=1` by default; SATF is not sensitive to the absolute score scale as long as higher scores imply higher relevance).

**Step 2: Find the highest-mass contiguous window.**
Choose a fixed-size contiguous window of `m` sessions that maximizes total weight:

`(a*, b*) = argmax_{a: b=a+m-1} Σ_{j=a..b} w_j`.

We set `m` as a fraction of the history length (default `m = ceil(0.2 * |S|)`), which approximates “find the dominant temporal region” without requiring conversion from sessions to calendar days.

**Step 3: Confidence gating (selective activation).**
Define a concentration ratio:

`r = (Σ_{j=a*..b*} w_j) / (Σ_{j=1..|S|} w_j)`.

If `r < γ`, we **do not filter** and return the original retrieval list. This prevents SATF from harming queries whose initial retrieval is temporally diffuse.

**Step 4: Re-ranking (stable partition).**
If `r ≥ γ`, we reorder the ranked list by a stable partition:
- Keep items whose parent session `j_i ∈ [a*, b*]` at the top (preserving their original relative order).
- Push all other items to the bottom (preserving their original relative order).

This matches LongMemEval’s time-filtering style (promotion via filtering) but replaces LLM-based time-range inference with self-anchored peak detection.

**Diagnostics computed by SATF (no extra runs):**
- **WindowCoverage@N**: fraction of temporal queries whose gold evidence sessions intersect `[a*, b*]` when SATF activates.
- **Activation rate**: fraction of queries with `r ≥ γ`.

### Key Innovations

1. **LLM-free temporal filtering for conversational memory retrieval:** SATF eliminates the need for a strong LLM to parse date ranges from questions, while targeting the same bottleneck LongMemEval identifies.
2. **Self-anchored temporal inference:** rather than inferring time from query semantics, SATF infers time from the retrieval distribution itself, which can capture implicit temporal intent.
3. **Selective activation with an explicit diagnostic:** SATF includes a confidence gate and reports WindowCoverage@N and activation rate, making it easy to determine when the approach is applicable.

---

## Related Work

### Field Overview

**Long-term conversational memory benchmarks and systems.** Recent benchmarks such as LoCoMo and LongMemEval show that long-context LLMs and naive RAG struggle with multi-session recall, temporal reasoning, and knowledge updates. Systems like Mem0, Zep, and Temporal Semantic Memory (TSM) introduce structured storage (graphs, temporal validity) and improved retrieval pipelines to address these deficits.

**Temporal filtering in memory retrieval.** LongMemEval’s time-aware query expansion is a simple but effective approach: use an LLM to infer a time range and filter candidates, improving Recall@k on temporal reasoning questions. More sophisticated memory systems (e.g., Zep, TSM) incorporate timestamps/temporal validity directly in memory representations and retrieval ranking.

**Temporal information retrieval and pseudo-relevance feedback (PRF).** In classic IR, temporal intent has been inferred from the timestamp distribution of top retrieved documents (“temporal profiles”), and temporal PRF methods incorporate time signals into relevance models, especially for microblogs/news where content is time-sensitive. SATF adapts this idea to **conversational memory retrieval** and evaluates it on LongMemEval.

### Related Papers

- **[LongMemEval](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/meta/meta_info.txt)**: Introduces LongMemEval and shows LLM-based time-range filtering improves temporal retrieval but requires a strong time-range extractor.
- **[LoCoMo](./references/Evaluating-Very-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)**: Benchmark for very long-term conversational memory; highlights large gaps on temporal reasoning and benefits of retrieval over long contexts.
- **[Mem0](../../papers/paper_summaries/Mem0 Building Production-Ready AI Agents with Scalable Long-Term Memory.md)**: Production memory system with extraction/update and strong latency/token reporting; includes temporal metadata.
- **[Zep](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt)**: Temporal knowledge-graph memory engine with time-aware retrieval and strong LongMemEval results in enterprise-like settings.
- **[Temporal Semantic Memory (TSM)](./references/Beyond-Dialogue-Time-Temporal-Semantic-Memory-for-Personalized-LLM-Agents/meta/meta_info.txt)**: Separates dialogue time vs semantic time and uses temporal intent parsing plus durative summaries for improved temporal and update queries.
- **[TiMem](./references/TiMem-Temporal-Hierarchical-Memory-Consolidation-for-Long-Horizon-Conversational-Agents/meta/meta_info.txt)**: Temporal-hierarchical consolidation with complexity-aware recall; shows SOTA accuracy on LoCoMo and LongMemEval-S.
- **[Memory OS](../../papers/paper_summaries/Memory OS of AI Agent.md)**: OS-inspired multi-tier memory with explicit promotion rules; strong LoCoMo gains.
- **[H-MEM](../../papers/paper_summaries/H-MEM Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents.md)**: Hierarchical memory with index-based routing for efficient retrieval on LoCoMo.
- **[ProMem](../../papers/paper_summaries/Beyond Static Summarization Proactive Memory Extraction for LLM Agents.md)**: Proactive memory extraction/verification improving integrity and downstream QA.
- **[SYNAPSE](../../papers/paper_summaries/SYNAPSE Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation.md)**: Spreading-activation retrieval to overcome contextual isolation on long-horizon memory benchmarks.
- **[MemGPT](https://arxiv.org/abs/2310.08560)**: OS-style agent memory with external storage and retrieval; a strong reference for practical memory interfaces.
- **[MemoryBank](https://arxiv.org/abs/2305.10250)**: Early LLM memory framework with long-term user preference storage and retrieval.
- **[A-Mem: Agentic Memory for LLM Agents](./references/A-Mem-Agentic-Memory-for-LLM-Agents/meta/meta_info.txt)**: Builds a Zettelkasten-style linked note graph with LLM-driven link creation and memory evolution; strong LoCoMo results with reduced token usage.
- **[RAPTOR](https://arxiv.org/abs/2401.18059)**: Tree-based recursive summarization and retrieval for long documents; relevant for hierarchical retrieval.
- **[Time-Aware Latent Concept Expansion for Microblog Search](https://miyatai.org/pdf/miyaICWSM2014.pdf)**: Temporal PRF method using temporal variation of concepts to improve microblog retrieval.
- **[Improving Pseudo-Relevance Feedback via Tweet Selection](https://miyatai.org/pdf/miyaCIKM2013.pdf)**: Uses timestamp distributions and temporal evidence to improve PRF on microblogs.
- **[Temporal Models for Microblogs](https://ciir-publications.cs.umass.edu/getpdf.php?id=1073)**: Temporal IR models incorporating time distributions for microblog search.
- **[Temporal Information Retrieval (survey)](https://www.dc.fi.udc.es/~roi/publications/fntir-temporalweb_ebook.2015.pdf)**: Survey of temporal IR including temporal profiles derived from retrieved documents.
- **[Pseudo-Relevance Feedback with Deep Language Models (survey)](https://dl.acm.org/doi/10.1145/3570724)**: Modern PRF methods and selective feedback ideas relevant to SATF’s confidence gating.
- **[Pseudo-Relevance Feedback Can Improve Zero-Shot LLM-Based Dense Retrieval](https://arxiv.org/abs/2503.14887)**: Uses PRF-style mechanisms with LLM-based dense retrievers; adjacent to “self-feedback” ideas.
- **[Time-Specifier Model Merging for Temporal IR](https://arxiv.org/abs/2507.06782)**: Builds temporal competence into retrievers without heavy query-side reasoning.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| LLM time-range extraction + filtering | Parse query time with a strong LLM; filter candidates by inferred range | LongMemEval (time-aware query expansion) | LongMemEval TR subset | Requires strong LLM; weak LLMs hallucinate time ranges |
| Temporal KG / semantic time memory | Store explicit temporal validity; retrieve by temporal intent | Zep, TSM | LongMemEval, LoCoMo | Heavier infrastructure; may still need time parsing |
| Temporal-hierarchical consolidation | Enforce temporal containment in memory hierarchy; adapt recall scope | TiMem | LongMemEval-S, LoCoMo | Depends on LLM middleware; more complex pipeline |
| Temporal IR / temporal PRF | Infer temporal intent or reweight retrieval using timestamp distributions | Microblog temporal PRF (Miyanishi et al.), temporal IR survey | TREC Microblog, news IR | Mostly studied in document search, not conversational memory |
| **Self-anchored temporal filtering (ours)** | Infer a dominant time window from top-N retrieval timestamps; selectively filter | This proposal | LongMemEval TR subset | Fails if baseline retrieval timestamps are uninformative |

### Closest Prior Work

- **LongMemEval time-aware query expansion**: Uses an LLM (GPT-4o) to infer a date range and filters retrieval candidates accordingly, improving Recall@10 from 0.550 to 0.722 on the temporal subset (Value=Round, Key=V+fact) (Table 3). **SATF differs** by using no time-parsing LLM and instead inferring the temporal window from the timestamp distribution of initial retrieval results.

- **Zep / TSM**: Store temporal metadata and retrieve with explicit temporal reasoning (semantic time, validity intervals) to address temporal mismatches. **SATF differs** by being a lightweight, drop-in retrieval wrapper that requires only timestamps (already present in LongMemEval) and no graph construction.

- **Temporal pseudo-relevance feedback in IR (microblogs/news)**: Uses timestamp distributions of (pseudo-)relevant documents to improve retrieval and query expansion. **SATF differs** by applying a peak-detection + confidence gating mechanism specifically to conversational memory retrieval (session/round items) and evaluating on LongMemEval.

**Novelty Kill Search Summary:** Searched for combinations of “LongMemEval + pseudo relevance feedback + temporal”, “LongMemEval timestamp filtering without LLM”, “timestamp distribution infer time window retrieval”, and related variants, and checked local proposal drafts/finalized proposals for “PRF/anchor time/time window inference”. No prior work applying timestamp-peak inference from initial retrieval to LongMemEval-style conversational memory retrieval was found as of 2026-02-16. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LongMemEval time-aware query expansion | LLM extracts date range; filter candidates | Needs strong LLM; weak LLMs hallucinate | Replace time parsing with timestamp peak detection | If initial retrieval has a temporal peak, filtering can be done without LLM calls |
| Zep | Temporal KG + time-aware retrieval | Heavier system; storage+graph overhead | Keep flat retrieval; add lightweight filter | Minimal overhead; can be added to existing RAG pipelines |
| TSM | Semantic time + durative summaries | Requires time parsing + KG construction | No new memory representation | When temporal signal is already in retrieval logs, a wrapper may suffice |
| Temporal PRF (microblogs) | Uses temporal evidence for PRF | Not evaluated in conversational memory | Apply PRF-style temporal inference to LongMemEval | LongMemEval provides a clean testbed with ground-truth evidence locations |

---

## Experiments

### Experimental Setup

We evaluate SATF purely at the **retrieval stage** using LongMemEval’s gold evidence locations (no human evaluation).

**Primary setting (from LongMemEval Table 3):**
- Dataset: LongMemEvalM temporal reasoning subset
- Value granularity: **Round**
- Key design: **K = V + fact** (value text concatenated with extracted user facts)
- Retriever: Stella V5 1.5B embedding model (as used by LongMemEval; HF model `NovaSearch/stella_en_1.5B_v5` or the exact model id used in their repo)
- Retrieval budget: evaluate Recall@5/10 and NDCG@5/10 on retrieved top-k

**Methods (main conditions; ≤3):**
1. **Baseline (no temporal filtering)**: LongMemEval retrieval with K = V + fact.
2. **LLM time-range filtering (reference baseline)**: LongMemEval’s official time-aware query expansion using GPT-4o (`src/index_expansion/temp_query_search_pruning.py`).
3. **SATF (ours; 0 LLM calls)**: Peak-detection + confidence-gated temporal filtering applied to the baseline retrieval ranked list.

**Ablations / controls (≤2):**
- **SATF w/o confidence gating** (`γ=0`): always apply temporal window filtering; tests whether selectivity is necessary.
- **Timestamp-augmented retrieval key (optional control)**: append the session date string (YYYY/MM/DD) to each value text before embedding; tests whether simple metadata injection into embeddings matches SATF.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Stella embedding model | 1.5B | https://huggingface.co/NovaSearch/stella_en_1.5B_v5 | Use the same embedding model/config as LongMemEval for comparability |
| GPT-4o (time-range extraction) | API | (via LongMemEval script) | Only used for the reference baseline time-range inference |

**Training Data (if applicable):**
- No training data needed — inference-only retrieval and filtering.

**Resource Estimate**:
- **Compute budget**: ≤ 20 GPU-hours (embedding + retrieval over LongMemEvalM; dominated by embedding computation; no training).
- **GPU memory**: 24–40GB sufficient for embedding inference (exact depends on embedding model implementation).
- **API usage**: GPT-4o calls for time-range extraction on the temporal subset only (on the order of 100–200 short calls; exact count depends on the subset size).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| LongMemEvalM (temporal subset) | Long-term memory benchmark; temporal reasoning questions require retrieving evidence sessions/rounds in correct time period | Recall@5/10, NDCG@5/10; plus WindowCoverage@N and activation rate (ours) | test | https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned | https://github.com/xiaowu0162/LongMemEval |

### Main Results

#### Results Table

Published baselines are copied verbatim from LongMemEval Table 3 (temporal reasoning subset of LongMemEvalM; Value=Round, Key=V+fact).

| Method | Base Model | Benchmark | Recall@10 | NDCG@10 | Source | Notes |
|---|---|---|---:|---:|---|---|
| Baseline retrieval (K=V+fact) | Stella V5 1.5B | LongMemEvalM-TR | 0.550 | 0.598 | [LongMemEval Table 3](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/sections/5.4 Query Time-aware query expansion improves temporal reasoning.md) | Published (1 run) |
| + Time-range filtering (GPT-4o) | Stella V5 1.5B + GPT-4o | LongMemEvalM-TR | 0.722 | 0.669 | [LongMemEval Table 3](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/sections/5.4 Query Time-aware query expansion improves temporal reasoning.md) | Published (1 run); requires strong LLM |
| + Time-range filtering (Llama 3.1 8B) | Stella V5 1.5B + Llama 8B | LongMemEvalM-TR | 0.570 | 0.647 | [LongMemEval Table 3](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/sections/5.4 Query Time-aware query expansion improves temporal reasoning.md) | Published (1 run); weak time parser |
| **SATF (ours; 0 LLM calls)** | Stella V5 1.5B | LongMemEvalM-TR | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| SATF (full) | Peak detection + confidence gating + stable partition | Best trade-off (gain without harming non-temporal queries) |
| SATF w/o gating | Set `γ=0` (always filter) | Lower performance due to filtering on temporally diffuse queries |
| Timestamp-augmented keys (optional) | Append session date tokens to value text before embedding | Likely smaller gain than SATF (does not resolve relative time expressions) |

### Experimental Rigor

**Variance & Reproducibility:**
- Retrieval and filtering are deterministic given fixed embeddings and sorting; we will fix random seeds for any batching/shuffling in embedding inference.

**Validity & Controls:**
- **Do-no-harm check:** Apply SATF to the full LongMemEvalM set and report recall metrics on non-temporal question types; SATF should not reduce Recall@10 by more than 0.02 on non-temporal queries.
- **WindowCoverage@N diagnostic:** If gold evidence sessions rarely fall inside the inferred window (low coverage), SATF cannot work in principle; we will report this metric explicitly.

**Fair Comparison Conditions:**
- SATF uses the same base retrieval results and top-k budget as the LongMemEval baseline; it only reorders the ranked list.

---

## Success Criteria

**Hypothesis** (directional): SATF increases retrieval recall on LongMemEval’s temporal reasoning subset by concentrating retrieval on a self-inferred time region when the initial retrieval timestamps are peaked.

**Decision Rule** (concrete):
- **Proceed** if, on LongMemEvalM temporal reasoning subset (Value=Round, Key=V+fact), SATF achieves:
  - Recall@10 improvement ≥ +0.05 over the baseline (≥ 0.600),
  - and WindowCoverage@N ≥ 0.60,
  - and non-temporal queries degrade by <0.02 Recall@10.
- **Pivot** if WindowCoverage@N is high (≥0.60) but Recall@10 gain is <0.05: try a two-window variant (allow two disjoint peaks) while keeping the same confidence gating.
- **Refute** if WindowCoverage@N < 0.50 or if SATF hurts Recall@10 (negative gain) on the temporal subset despite confidence gating.

---

## Impact Statement

If successful, SATF would provide a cheap alternative to LLM-based time-range extraction for temporal retrieval in long-term memory assistants. Practitioners building memory layers for chat assistants could reduce latency and API cost by removing per-query “time parsing” calls, while retaining a substantial fraction of the temporal retrieval gains that LongMemEval attributes to time-aware filtering.

---

## References

- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/meta/meta_info.txt) - Wu et al., 2024
- [Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo)](./references/Evaluating-Very-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt) - Maharana et al., 2024
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt) - Rasmussen et al., 2025
- [Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents](./references/Beyond-Dialogue-Time-Temporal-Semantic-Memory-for-Personalized-LLM-Agents/meta/meta_info.txt) - Su et al., 2026
- [TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](./references/TiMem-Temporal-Hierarchical-Memory-Consolidation-for-Long-Horizon-Conversational-Agents/meta/meta_info.txt) - Li et al., 2026
- [Time-Aware Latent Concept Expansion for Microblog Search](https://miyatai.org/pdf/miyaICWSM2014.pdf) - Miyanishi et al., 2014
- [Improving Pseudo-Relevance Feedback via Tweet Selection](https://miyatai.org/pdf/miyaCIKM2013.pdf) - Miyanishi et al., 2013
- [Temporal Information Retrieval (Foundations and Trends survey)](https://www.dc.fi.udc.es/~roi/publications/fntir-temporalweb_ebook.2015.pdf) - Kanhabua et al., 2015
- [Pseudo-Relevance Feedback with Deep Language Models and Dense Retrievers (TOIS survey)](https://dl.acm.org/doi/10.1145/3570724) - Yu et al., 2022
- [Pseudo-Relevance Feedback Can Improve Zero-Shot LLM-Based Dense Retrieval](https://arxiv.org/abs/2503.14887) - 2025
- [Time-Specifier Model Merging for Temporal IR](https://arxiv.org/abs/2507.06782) - 2025
