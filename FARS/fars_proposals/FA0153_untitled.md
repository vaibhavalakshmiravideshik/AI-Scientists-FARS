# untitled

# Assistant-Inclusive Keying for LongMemEval: Fielded Retrieval for Assistant-Side Memory Recall

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Long-horizon chat assistants increasingly rely on an external memory layer rather than placing the entire interaction history into the model’s context window. A common design is retrieval-augmented generation (RAG): store past interaction units (sessions, turns, or extracted facts), retrieve a small set of candidates for a new query, then answer conditioned on the retrieved evidence.

A practical but under-discussed detail is **what text is used as the retrieval key**. Many systems bias the index toward **user utterances** (user profiles, preferences, requests), because assistant responses can be verbose and stylistically repetitive. However, real assistants must also remember **assistant-side information**: prior recommendations, commitments, and explanations.

### The Problem

LongMemEval is a benchmark for long-term conversational memory that explicitly tests assistant-side recall via a dedicated **single-session-assistant** question type (“Single-session-user and single-session-assistant test memorizing the information mentioned by user or assistant within a single session.” **[LongMemEval §3.2](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/sections/3.2 LongMemEval Benchmark Curation.md)**).

However, LongMemEval’s official dense retrieval setup uses a **user-only key** even when the retrieval unit is a round (user message + subsequent assistant response):

- Paper statement: “When sessions or rounds are used as the key we only keep the user-side utterances.” (**[LongMemEval §5.1](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/sections/5.1 Experimental Setup.md)**)
- Code path: `process_item_flat_index` constructs the retrieval corpus from `turn['role']=='user'` only (LongMemEval repo, `src/retrieval/run_retrieval.py`).

This creates a plausible mismatch: for SSA queries, the most query-relevant lexical items may appear primarily in the **assistant response** (e.g., a named recommendation) rather than in the user prompt that elicited it. If the retrieval key omits assistant utterances, dense retrieval may fail to retrieve the correct round/session even if the answer is present in the stored value.

This matters beyond LongMemEval. Multiple recent memory systems report weaknesses on assistant-side recall or incorporate speaker attribution (e.g., **[Zep](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt)**, **[ENGRAM](./references/ENGRAM-Effective,-Lightweight-Memory-Orchestration-for-Conversational-Agents/meta/meta_info.txt)**, **[EMem](./references/A-Simple-Yet-Strong-Baseline-for-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)**), but do not isolate whether the *retrieval key* should include assistant text.

### Key Insight and Hypothesis

**Key insight:** The “user-only key” inductive bias is appropriate for user-profile recall, but it can be harmful for assistant-side recall. A better strategy is to treat each round as a **two-field document** (user field and assistant field) and allow the query to match either field.

**Mechanism hypothesis (why fielded should beat concatenation):** In dense retrieval, concatenating user+assistant into one embedding can dilute query-relevant features because (i) assistant responses are often longer and more generic, and (ii) both query and document are truncated (e.g., to 512 tokens in Stella). Fielded scoring avoids cross-field interference by computing separate similarities and combining them, analogous to classic fielded retrieval (BM25F, a fielded extension of BM25 for multi-field documents) and max-style late interaction.

**Hypothesis:** A simple **fielded dense retrieval** scoring rule will improve SSA retrieval while preserving overall retrieval quality:

> For each round, embed the user message and the assistant response separately. Rank rounds by \(\max(\mathrm{sim}(q, u),\;\mathrm{sim}(q, a))\). This improves SSA Recall@10 because SSA queries often share entities with the assistant response but not with the user prompt.

Why this could fail: SSA queries may still be sufficiently anchored by the user prompt that elicited the assistant response, leaving little headroom. Alternatively, assistant responses may be too generic (or too long/truncated) for dense embeddings to provide useful additional signal, and assistant inclusion may increase false matches.

---

## Proposed Approach

### Overview

We propose a minimal change to LongMemEval’s retrieval stage: **assistant-inclusive, fielded keying**.

- Keep LongMemEval’s retrieval unit (round-level) unchanged.
- Add an additional assistant text field per round.
- Score each candidate round by the maximum similarity between the query and either the user field or the assistant field.

This is intended as a drop-in alternative to the user-only keying in the official LongMemEval retrieval code.

### Method Details

We target LongMemEval’s round-level dense retrieval setting (official code uses `granularity='turn'` but each retrieved user turn is expanded to include its subsequent assistant turn at answer time).

**Corpus construction (per question instance):**

For each session `sess_entry` with id `sess_id`, iterate through turns. For each user turn at index `i_turn`:

- **User field** \(u_i\): `sess_entry[i_turn]['content']` (exactly what the official baseline indexes).
- **Assistant field** \(a_i\): `sess_entry[i_turn+1]['content']` if `i_turn+1` exists and is an assistant turn; otherwise empty string.
- **Document id**: keep the same id scheme as the baseline (e.g., `sess_id_{i_turn+1}`), so the evaluation labels remain comparable.

**Embedding model:** Use the same dense retriever as LongMemEval:

- **Stella EN 1.5B v5** (HuggingFace: https://huggingface.co/dunzhang/stella_en_1.5B_v5)
- Use the same truncation policy as the official code (`max_length=512` in the Stella branch).

**Scoring rule (fielded max-sim):**

Let \(e(\cdot)\) be the normalized embedding produced by Stella. For a query \(q\) and round \(i\):

\[
 s_i(q) = \max(\langle e(q), e(u_i) \rangle,\; \langle e(q), e(a_i) \rangle).
\]

Rank documents by \(s_i(q)\).

**Ablation (naïve concatenation):**

Embed a single key string \(k_i = u_i \Vert a_i\) and score by \(\langle e(q), e(k_i) \rangle\).

**Fallback (only if needed):** If max-sim improves SSA but harms other types, test a fixed mixture score:
\(\alpha\langle e(q),e(u_i)\rangle + (1-\alpha)\langle e(q),e(a_i)\rangle\) with \(\alpha\in\{0.5, 0.7\}\) (note \(\alpha=1\) recovers the user-only baseline).

### Key Innovations

1. **Benchmark-grounded issue identification:** LongMemEval includes assistant-side recall tasks, yet its official dense retrieval keying omits assistant text.
2. **Minimal fielded dense retrieval for conversational memory:** A two-field max-sim scoring rule that avoids forcing assistant verbosity into the same embedding as the user prompt.
3. **Boundary-condition characterization:** Quantify when assistant-inclusive keying helps (SSA) and when it harms (other question types), producing a practitioner-facing recommendation.

---

## Related Work

### Field Overview

**Long-term conversational memory benchmarks.** Several benchmarks evaluate long-horizon dialogue understanding and memory, including MemoryBank, PerLTQA, LoCoMo, DialSim, and LongMemEval. LongMemEval is distinctive in explicitly testing assistant-side recall and providing evidence labels suitable for deterministic retrieval-metric evaluation (**[LongMemEval Memory Recall](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/sections/Memory Recall.md)**).

**Memory system designs for chat assistants.** Recent systems span flat dense retrieval baselines, structured knowledge graphs, typed memory stores, and event/graph-based representations. Many systems focus on user-profile retention and temporal reasoning; fewer isolate assistant-side recall as a design axis.

**Fielded retrieval and multi-field ranking.** In information retrieval, “fielded” scoring (e.g., BM25F) and max-style late interaction motivate separating heterogeneous text fields and combining evidence at scoring time. Our proposal applies this idea to conversational memory, where user and assistant utterances are naturally distinct fields.

### Related Papers

- **[LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/meta/meta_info.txt)**: Introduces LongMemEval and an official retrieval pipeline that uses user-only keys even for SSA questions.
- **[LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17416)**: Long-term conversational memory benchmark with multi-session and temporal question types.
- **[MemoryBank](https://arxiv.org/abs/2305.10250)**: Memory system and benchmark focused on user information retention across days; limited assistant-side recall evaluation.
- **[PerLTQA](https://arxiv.org/abs/2402.16288)**: Personal long-term memory QA dataset; emphasizes retrieval and synthesis over long histories.
- **[DialSim](https://arxiv.org/abs/2406.13144)**: Simulator-based evaluation of long-term dialogue understanding with time constraints.
- **[MemGPT](https://arxiv.org/abs/2310.08560)**: OS-style memory management for LLM agents, storing and retrieving past context from an external store.
- **[Mem0](https://arxiv.org/abs/2409.03340)**: Practical memory store for assistants with fact extraction and retrieval; commonly used baseline in memory-system comparisons.
- **[MemOS](https://arxiv.org/abs/2507.03724)**: Operating-system style memory scheduler and multi-tier stores for agent memory.
- **[LangMem](https://arxiv.org/abs/2404.06654)**: Hierarchical organization and compression for long-horizon conversational memory.
- **[Zep: A Temporal Knowledge Graph Architecture for Agent Memory](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt)**: Temporal KG memory system that reports assistant-side recall weaknesses in LongMemEval analysis.
- **[ENGRAM](./references/ENGRAM-Effective,-Lightweight-Memory-Orchestration-for-Conversational-Agents/meta/meta_info.txt)**: Typed memory stores with dense retrieval; formats evidence into speaker-specific banks at answer generation time.
- **[EMem: A Simple Yet Strong Baseline for Long-Term Conversational Memory](./references/A-Simple-Yet-Strong-Baseline-for-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)**: Event-centric memory units and graph propagation; includes speaker attribution in evidence.
- **[TiMem](https://arxiv.org/abs/2501.10200)**: Temporal hierarchical consolidation for conversational agents; targets temporal and multi-session reasoning.
- **[Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects](https://arxiv.org/abs/2502.05589)**: Memory substrate and reflection loops for improved long-horizon recall.
- **[MemWeaver](https://arxiv.org/abs/2501.15249)**: Hybrid memory (graph + experience + passages) with traceable long-horizon reasoning.
- **[SwiftMem: Fast Agentic Memory via Query-aware Indexing](https://arxiv.org/abs/2601.08160)**: Query-aware indexing (tags + temporal index) for low-latency memory retrieval.
- **[SGMem: Sentence Graph Memory](https://arxiv.org/abs/2509.21212)**: Graph over sentence-level memory to mitigate fragmentation via multi-hop traversal.
- **[xMemory: Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation](https://arxiv.org/abs/2602.02007)**: Retrieval by decoupling and aggregation to reduce redundancy.
- **[LiCoMemory](https://arxiv.org/abs/2511.01448)**: Lightweight hierarchical memory with temporal reranking.
- **[MEMORA](https://arxiv.org/abs/2602.03315)**: Memory representations and policy-guided retrieval.
- **[BM25F: The Okapi Weights](https://www.microsoft.com/en-us/research/publication/bm25-and-bm25f-the-okapi-weights/)**: Fielded sparse retrieval scoring that weights multiple document fields.
- **[DPR: Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)**: Foundational dense retrieval baseline.
- **[ColBERT](https://arxiv.org/abs/2004.12832)**: Late-interaction dense retrieval with max-sim aggregation, motivating max-style matching.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Flat dense retrieval over dialogue units | Embed query and stored units; retrieve top-k by similarity | LongMemEval baseline, DPR | LongMemEval, PerLTQA | Sensitive to key design; can miss assistant-side info if keys are user-centric |
| Typed / structured memory stores | Separate memory types (episodic/semantic/procedural) and retrieve per type | ENGRAM, MemOS | LoCoMo, LongMemEval | Often focuses on storage/formatting; key design choices may remain implicit |
| Graph / KG memory | Build entity/event graphs; retrieve via traversal/propagation | Zep, EMem, MemWeaver, SGMem | LongMemEval, LoCoMo | Higher complexity; assistant-side recall may still degrade |
| Query-aware indexing | Build indices guided by predicted query workload | SwiftMem, xMemory | LoCoMo, LongMemEval-S | Often uses LLM tagging; still requires good keying of stored text |
| Fielded / late-interaction retrieval | Keep heterogeneous fields separate; combine scores | BM25F, ColBERT | IR benchmarks; (this proposal: LongMemEval) | Field choice and aggregation can be brittle |

### Closest Prior Work

- **LongMemEval** (**[paper](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/meta/meta_info.txt)**): Defines the benchmark and a retrieval framework with key expansion and time-aware retrieval, but its official round/session keying uses user-only text.
- **ENGRAM** (**[paper](./references/ENGRAM-Effective,-Lightweight-Memory-Orchestration-for-Conversational-Agents/meta/meta_info.txt)**): Separates retrieved evidence by speaker at prompt construction time but does not test whether assistant text should be included in the retrieval key.
- **EMem** (**[paper](./references/A-Simple-Yet-Strong-Baseline-for-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)**): Uses event-centric units and includes speaker attribution for evidence interpretation, but is not a minimal keying intervention.
- **Zep** (**[paper](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt)**): Reports strong results and efficiency, but does not isolate keying as a cause of SSA weaknesses.
- **BM25F / ColBERT** (**[BM25F](https://www.microsoft.com/en-us/research/publication/bm25-and-bm25f-the-okapi-weights/)**; **[ColBERT](https://arxiv.org/abs/2004.12832)**): Motivate field separation and max-style aggregation; we apply this principle to conversational memory retrieval.

**Novelty Kill Search Summary:** Searched for combinations of “LongMemEval user-side utterances key”, “LongMemEval assistant utterances key design retrieval”, “LongMemEval single-session-assistant retrieval”, and broader queries on “speaker-separated retrieval dialogue RAG separate index” and “BM25F fielded retrieval dialogue user assistant”. Also checked local finalized proposals and other agents’ drafts for “user-side utterances”, “assistant_previnfo”, and “single-session-assistant”. No prior work explicitly isolating and testing assistant-inclusive keying within LongMemEval’s retrieval framework was found as of 2026-02-19. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LongMemEval | Round/session retrieval + key expansion + time-aware filtering | User-only key may miss assistant-only entities for SSA | Add assistant field and fielded scoring | SSA queries can match assistant field even if user prompt lacks key terms |
| ENGRAM | Typed stores; speaker-separated prompt banks | Does not isolate key design for assistant-side recall | Isolate keying choice in LongMemEval setup | Quantifies whether keying alone explains SSA weakness |
| EMem | Event-centric units + speaker attribution | More complex pipeline; not a minimal keying intervention | Minimal fielded retrieval change | Provides a deployable, low-effort fix in flat retrieval setting |
| Zep | Temporal KG + hybrid retrieval | SSA degradation not explained; complex system | Minimal keying intervention | If SSA weakness is largely key-design, our fix is cheaper |
| BM25F / ColBERT | Fielded / max-style retrieval in IR | Not applied to LongMemEval key design | Dense fielded retrieval over user/assistant fields | Dialogue naturally has fields; max-sim reduces cross-field interference |

---

## Experiments

### Experimental Setup

We evaluate only the **retrieval stage**, using LongMemEval’s evidence labels to compute retrieval metrics (no LLM-based judging).

**Dataset:**

- **LongMemEvalM (original)**: `longmemeval_m.json` from the (deprecated) dataset page https://huggingface.co/datasets/xiaowu0162/longmemeval (license MIT).
  - We use the original dataset because LongMemEval Table 2 retrieval baselines are reported on LongMemEvalM.
  - A robustness rerun on the cleaned variant (`longmemeval_m_cleaned.json`, https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) is optional but not required to decide the core hypothesis.

**Retriever:** Stella V5 1.5B (`flat-stella`) as in LongMemEval.

**Retrieval unit:** Round-level (official implementation uses `granularity='turn'` and indexes user turns; each retrieved user turn is expanded to include its subsequent assistant response in the generation stage).

**Methods compared (≤3 main conditions):**

1. **Baseline (official):** user-only keying (index user turns only), `flat-stella`, `granularity=turn`, no index expansion (K=V).
2. **Ablation:** naïve concatenation keying (index a single concatenated text per round).
3. **Ours:** fielded max-sim (index user field + assistant field; score by max similarity).

### Benchmarks and Metrics

LongMemEval provides evidence labels that support deterministic retrieval metrics (**[LongMemEval Memory Recall](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/sections/Memory Recall.md)**).

We will report (from the retrieval logs):

- **Recall@10**: using the official `recall_all@10` metric (all evidence docs retrieved within top-10).
- **NDCG@10**: using the official `ndcg_any@10` metric.
- **Per-question-type breakdown**, especially **single-session-assistant**.

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| LongMemEvalM | 500 questions; each has ~500 history sessions (≈1.5M tokens) | `recall_all@10`, `ndcg_any@10` overall + per `question_type` | test | https://huggingface.co/datasets/xiaowu0162/longmemeval | LongMemEval `src/retrieval/run_retrieval.py` + custom aggregation by `question_type` |

**Per-type metrics implementation note:** The official `print_retrieval_metrics.py` reports only overall means. We will compute per-type means by parsing the retrieval log JSONL, grouping by `question_type`, and averaging the per-entry metric fields.

### Main Results

Published baseline numbers from LongMemEval Table 2 (LongMemEvalM; Value=Round; K=V) for reference:

| Method | Retriever | Benchmark | Recall@10 | NDCG@10 | Source | Notes |
|---|---|---|---:|---:|---|---|
| User-only key (K=V) | Stella V5 1.5B | LongMemEvalM | 0.692 | 0.512 | **[LongMemEval Table 2](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/sections/5.3 Key Multi-key indexing improves retrieval and RAG.md)** | Published (1 run); user-only key |

Planned results table to be filled by verification runs:

| Method | Retriever | Benchmark | Overall Recall@10 | Overall NDCG@10 | SSA Recall@10 | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| User-only key (baseline) | Stella V5 1.5B | LongMemEvalM | **TBD** | **TBD** | **TBD** | - | To be re-run for exact comparability |
| Concat key (ablation) | Stella V5 1.5B | LongMemEvalM | **TBD** | **TBD** | **TBD** | - | To be verified |
| **Fielded max-sim (ours)** | Stella V5 1.5B | LongMemEvalM | **TBD** | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Concat key | Single embedding for user+assistant | May improve SSA but risks degrading other types due to noise/truncation |
| Fielded max-sim (ours) | Two embeddings per round; score by max | Preserves non-SSA while capturing SSA matches to assistant field |

### Experimental Rigor

- **Determinism**: Retrieval metrics are deterministic given fixed model weights and input JSON; multiple seeds are unnecessary.
- **Fairness controls**:
  - Same dataset file, same retriever weights, same truncation (`max_length=512`), same normalization.
  - Same document id scheme as the official baseline (so relevance labels are unchanged).
- **Sanity checks**:
  - Reproduce the published LongMemEval Table 2 baseline (K=V, Value=Round) within a small tolerance on the original `longmemeval_m.json`.
  - Run `retriever=oracle` once to confirm that the evaluation code yields near-perfect recall.

**Resource Estimate**:

- **Compute budget**: ≤150 GPU-hours for 3 retrieval runs on LongMemEvalM with Stella 1.5B (baseline ≈1×, concat ≈1×, fielded ≈2× document embedding cost).
- **GPU memory**: ≥40GB recommended for Stella 1.5B at practical batch sizes.
- **API usage**: None.

---

## Success Criteria

**Hypothesis** (directional): Fielded max-sim keying improves SSA Recall@10 by enabling queries to match assistant-only entities, while maintaining overall retrieval quality.

**Decision Rule** (concrete — when to stop):

- **Continue/Proceed** if, on LongMemEvalM:
  - SSA `recall_all@10` improves by **≥ 0.03 absolute** over the user-only baseline, **and**
  - Overall `recall_all@10` decreases by **≤ 0.005**.
- **Pivot** if SSA improves (≥0.02) but overall drops by >0.005: try the fixed-mixture scoring rule with \(\alpha\in\{0.5, 0.7\}\) (one small sweep).
- **Refute** if SSA gain is <0.01 **or** overall `recall_all@10` drops by >0.01.

---

## Impact Statement

If successful, this provides a near-zero-training-cost improvement for memory layers in chat assistants: index user and assistant utterances as separate fields and match queries to either. This can improve recall of prior assistant recommendations and commitments without changing the downstream reader model.

---

## References

- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](./references/LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory/meta/meta_info.txt) - Wu et al., 2024
- [ENGRAM: Effective, Lightweight Memory Orchestration for Conversational Agents](./references/ENGRAM-Effective,-Lightweight-Memory-Orchestration-for-Conversational-Agents/meta/meta_info.txt) - 2025
- [A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents](./references/A-Simple-Yet-Strong-Baseline-for-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt) - Zhou & Han, 2025
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt) - Rasmussen et al., 2025
- [LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17416) - Maharana et al., 2024
- [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250) - Zhong et al., 2024
- [PerLTQA: A Personal Long-Term Memory Dataset for Memory Classification, Retrieval, and Synthesis in QA](https://arxiv.org/abs/2402.16288) - Du et al., 2024
- [DialSim: A Real-Time Simulator for Evaluating Long-Term Dialogue Understanding](https://arxiv.org/abs/2406.13144) - Kim et al., 2024
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) - Packer et al., 2023
- [Mem0](https://arxiv.org/abs/2409.03340) - Chhikara et al., 2024
- [MemOS](https://arxiv.org/abs/2507.03724) - Li et al., 2025
- [LangMem](https://arxiv.org/abs/2404.06654) - 2024
- [TiMem](https://arxiv.org/abs/2501.10200) - 2025
- [Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects](https://arxiv.org/abs/2502.05589) - 2025
- [MemWeaver](https://arxiv.org/abs/2501.15249) - 2025
- [SwiftMem: Fast Agentic Memory via Query-aware Indexing](https://arxiv.org/abs/2601.08160) - 2026
- [SGMem: Sentence Graph Memory](https://arxiv.org/abs/2509.21212) - 2025
- [xMemory: Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation](https://arxiv.org/abs/2602.02007) - 2026
- [LiCoMemory](https://arxiv.org/abs/2511.01448) - 2025
- [MEMORA](https://arxiv.org/abs/2602.03315) - 2026
- [BM25F: The Okapi Weights](https://www.microsoft.com/en-us/research/publication/bm25-and-bm25f-the-okapi-weights/) - Robertson et al., 2004
- [Dense Passage Retrieval for Open-Domain Question Answering (DPR)](https://arxiv.org/abs/2004.04906) - Karpukhin et al., 2020
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832) - Khattab & Zaharia, 2020
