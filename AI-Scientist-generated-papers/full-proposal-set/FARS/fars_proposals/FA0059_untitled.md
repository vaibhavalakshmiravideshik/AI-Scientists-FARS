# untitled

# Last-Write-Wins Memory: Isolating Deterministic Overwrite Semantics for Long-Context Conflict Resolution

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

LLM-based agents increasingly rely on external memory to persist information across long interactions. A common implementation is retrieval-augmented generation (RAG): the agent stores past text chunks in an external database and retrieves a small subset at query time. This approach works well when the task is to recall a specific passage, but it often fails when information **changes over time**. In practical deployments, both user preferences and world facts can be updated (e.g., “my favorite food is sushi → pizza”, “the CEO is X → Y”). If a memory system cannot reliably overwrite outdated facts, the agent may surface stale information or mix old and new facts during reasoning.

The MemoryAgentBench benchmark formalizes this setting by evaluating “memory agents” across four competencies: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. Its Conflict Resolution suite uses FactConsolidation, built from **MQUAKE**—a benchmark for multi-hop QA under counterfactual knowledge edits (paired *old fact → new fact* rewrites)—where contradictory fact updates appear later in the context. The benchmark surfaces a sharp failure mode: multi-hop conflict resolution is near-random at long context lengths.

In MemoryAgentBench Table 2 (262K-token FactConsolidation; see [Section 4.2](./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/sections/4.2 Overall Performance Comparison.md)), the best published FactConsolidation-MH accuracy is only **7.0%** (Contriever RAG; a standard dense retriever), with other strong dense retrievers like NV-Embed-v2 (NVIDIA embedding model) at **6.0%** and even long-context proprietary models around **2–5%**. Yet the dataset is not inherently impossible: the same paper’s validation (Table 3; [Section 4.4.2](./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/sections/4.4.2 Validation of Dataset FactConsolidation.md)) shows a strong reasoning model (o4-mini) reaches **80.0%** FactConsolidation-MH at 6K tokens, dropping to **14.0%** at 32K, implying that scalability (not task definition) is the bottleneck.

A plausible explanation is **version ambiguity**: append-only memories retrieve both old and new versions of the “same” fact, and the LLM must resolve contradictions while also doing multi-hop composition. When multiple hops depend on updated values, even a single stale retrieval can break the chain.

### The Problem

We study a concrete mechanistic question:

> When the memory system has already extracted candidate facts from text, does enforcing **deterministic overwrite semantics** (last-write-wins) materially improve long-context conflict resolution compared to keeping all versions and asking the LLM to reconcile conflicts?

This question matters because many memory systems already implement some form of CRUD operations (e.g., MemGPT’s archival memory updates, Mem0’s ADD/UPDATE/DELETE policies, temporal knowledge-graph memories), but it is unclear whether overwrite semantics are actually the limiting factor on modern long-context conflict-resolution benchmarks.

The key challenge is to **separate overwrite semantics** from other confounds such as information extraction quality and retriever strength. Naïvely comparing “our structured memory system” to a RAG baseline does not answer whether overwrite semantics are the active ingredient.

### Key Insight and Hypothesis

**Key insight:** FactConsolidation contexts are explicitly formatted as numbered factual statements (a “knowledge pool”), and the benchmark instructions tell the model to prefer newer facts with larger serial numbers. This makes the problem unusually well-suited to a controlled test of overwrite semantics: we can hold the upstream fact extraction pipeline fixed and change only whether retrieval exposes multiple versions.

**Why this isn’t already answered by Knowledge Objects (Zahn et al., 2026):** *Attention Is Not Retention* demonstrates deterministic overwrite on a **single-session “conversational learning” micro-benchmark** (6 fact-learning exchanges, immediate recall with paraphrased queries) and argues that unstructured RAG lacks key-based identity and version semantics. However, it does not test whether overwrite semantics remains the limiting factor when the agent must do **multi-hop reasoning over many interleaved counterfactual updates** in a long context. MemoryAgentBench/FactConsolidation-MH is designed precisely to stress this multi-hop “final memory state” reasoning regime, and therefore can falsify (or validate) the hypothesis that last-write-wins filtering is a decisive ingredient beyond extraction and retrieval.

**Hypothesis:** Given a shared information-extraction pipeline, returning **only the latest version** of each (subject, predicate) fact (last-write-wins) will improve FactConsolidation-MH accuracy relative to returning multiple versions and asking the LLM to resolve conflicts, because multi-hop reasoning is brittle to even one stale fact.

We could be wrong if (i) the bottleneck is mainly extraction errors (missing or mis-canonicalized entities), (ii) the benchmark’s multi-hop questions require relations that do not fit a simple (subject, predicate, object) schema, or (iii) the LLM can already resolve conflicts when presented with multiple versions, making explicit overwrite unnecessary.

---

## Proposed Approach

### Overview

We propose **Last-Write-Wins Knowledge Objects (LWW-KO)**: a minimal, database-like memory layer that enforces deterministic overwrite semantics over extracted facts.

The proposal is deliberately scoped to a mechanism test. Conditions B and C share the same information-extraction and query-planning pipeline; the only difference is whether older versions are filtered before being shown to the answering LLM.

### Method Details

**Data format (FactConsolidation):** Each example contains a long `context` with numbered facts like:
`123. The chief executive officer of Apple Inc. is Tim Cook.`
The dataset provides lists of `questions` and `answers` per example. The `metadata.source` field indicates which sub-dataset variant is used (e.g., `factconsolidation_mh_6k`).

**Chunking (benchmark-compatible):** MemoryAgentBench chunks long contexts using `chunk_text_into_sentences(text, chunk_size)` implemented with a `tiktoken` tokenizer; the official configs for FactConsolidation use `chunk_size: 4096` tokens (e.g., `configs/data_conf/Conflict_Resolution/Factconsolidation_mh_262k.yaml`).

**Retriever baseline comparability:** Condition A will *not* be a “fresh RAG” implementation. We will run the **official Contriever RAG agent config** (the published best FactCon-MH baseline) with `retrieve_num: 10` (see `configs/agent_conf/RAG_Agents/gpt-4o-mini/Embedding_rag_gpt-4o-mini-contriever.yaml`). Our B/C methods reuse the same `retrieve_num=10` retrieval budget for selecting candidate facts, so the comparison isolates overwrite semantics rather than retrieval capacity.

**Information extraction (shared for B/C):** For each chunk, run an LLM prompt that extracts a set of candidate triples:
- subject (entity string)
- predicate (relation string)
- object (value string)
- serial number / recency signal (from the numbered fact line or chunk order)
- provenance (fact line text)

**Canonicalization:** Normalize subjects (and optionally predicates) by lowercasing and simple alias rules. (This is intentionally simple; we will report ablations only if needed.)

**Identity / keying:** Define a “fact key” as `key = hash(canonical_subject || canonical_predicate)`.

**Version store:** Maintain a mapping `key -> [versions]`, where each version contains `{object, serial, provenance}`.

**Query planning:** Given a question, run a prompt that outputs a small set of required keys (or a multi-hop plan as key sequences). Retrieve facts accordingly and provide them to the answer model.

- **Prompt (frozen across B/C):** “Given the question and the list of allowed relations (predicates) extracted from the context, output up to 5 fact keys in the form `(subject, predicate)` that would be needed to answer the question. If the question is multi-hop, include intermediate keys as well. Output JSON: {keys:[{subject:"...", predicate:"..."}, ...]}.”
- **Note:** This is intentionally a lightweight planner; if it fails, we will report the planned-key coverage (how often the gold answer is present under the planned keys) as part of the IE diagnostic.

**Overwrite semantics (the only change from B to C):**
- **B (append-only):** return multiple versions per key (e.g., top-2 most recent) to the answerer.
- **C (last-write-wins):** return only the latest version per key.

### Key Innovations

- A **controlled isolation** of overwrite semantics on a benchmark where conflict resolution is the primary unsolved competency.
- A minimal “Knowledge Object” instantiation that makes overwrite semantics explicit, without introducing heavy graph construction or multi-agent memory pipelines.

---

## Related Work

### Field Overview

Agent memory research spans long-context models, RAG-style external memory, graph/structure-augmented memory, and agentic memory systems with iterative retrieval and update cycles. Benchmarks have recently shifted from static long-context evaluation toward incremental interaction settings that better reflect deployed agents.

Conflict resolution is closely related to knowledge editing and unlearning, but differs in that the “ground truth” is determined by the most recent information in an interaction history rather than a global factual database. From a systems perspective, this resembles database update semantics: when a key is updated, should older values remain visible, and how should applications resolve conflicts?

### Related Papers

- **[Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions](./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/meta/meta_info.txt)**: Introduces MemoryAgentBench and FactConsolidation; shows FactConsolidation-MH accuracy ≤7% at 262K tokens.
- **[Attention Is Not Retention: The Orthogonality Constraint in Infinite-Context Architectures](./references/Attention-Is-Not-Retention-The-Orthogonality-Constraint-in-Infinite-Context-Architectures/meta/meta_info.txt)**: Proposes “Knowledge Objects” with hash identity and version chains to support deterministic overwrites; motivates overwrite semantics as a core limitation of RAG.
- **[Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](../../papers/paper_summaries/Mem0 Building Production-Ready AI Agents with Scalable Long-Term Memory.md)**: Production-oriented memory with ADD/UPDATE/DELETE decisions; includes a graph variant with temporal metadata.
- **[MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)**: OS-inspired memory hierarchy with archival memory that can be updated; representative of CRUD-like agent memory.
- **[HippoRAG-v2](https://arxiv.org/abs/2405.14831)**: Structure-augmented RAG with entity-centric retrieval; related as a strong baseline retriever but not focused on overwrite semantics.
- **[GraphRAG](https://arxiv.org/abs/2404.16130)**: Graph-based retrieval pipelines; can reduce redundancy but does not directly enforce deterministic overwrites.
- **[RAPTOR](https://arxiv.org/abs/2401.18059)**: Tree-structured summarization/retrieval; related structure-augmented baseline.
- **[MQUAKE](https://arxiv.org/abs/2305.14795)**: Counterfactual edit pairs for evaluating knowledge editing; source for FactConsolidation construction.
- **[MEMIT](https://arxiv.org/abs/2210.07229)**: Mass-editing factual knowledge in model weights; related motivation but different mechanism (parametric editing).
- **[ROME](https://arxiv.org/abs/2202.05262)**: Rank-one model editing; related to the broader “facts change” theme.
- **[AlphaEdit](https://arxiv.org/abs/2410.02355)**: Null-space constrained editing; cited by MemoryAgentBench as related to conflict resolution.
- **[AMA: Adaptive Memory via Multi-Agent Collaboration](./references/AMA-Adaptive-Memory-via-Multi-Agent-Collaboration/meta/meta_info.txt)**: Multi-agent memory pipeline (constructor/retriever/judge/refresher); addresses memory updates but not evaluated on FactConsolidation.
- **[FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt)**: Forgetting-based memory maintenance; includes conflict resolution mechanisms but not evaluated on FactConsolidation.
- **[LongMemEval](https://arxiv.org/abs/2410.10813)**: Long-term memory evaluation benchmark; includes update-related abilities.
- **[LoCoMo](https://arxiv.org/abs/2402.17416)**: Conversational long-term memory benchmark used in Mem0.
- **[A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/abs/2501.01754)**: Agentic memory formation and refinement; related to memory lifecycle design.
- **[Zep: A temporal knowledge graph memory engine](https://arxiv.org/abs/2501.13956)**: Temporal KG memory; closest in spirit to “versioned” semantic memory.
- **[From RAG to Memory: Non-parametric continual learning for LLMs](https://arxiv.org/abs/2502.14802)**: Treats retrieval memory as a continual learning substrate.
- **[SELF-PARAM](https://arxiv.org/abs/2406.09789)**: Parametric memory mechanisms; contrasts with external overwrite.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Long-context agents | Keep recent context in-window; drop older tokens | GPT-4o, Claude long-context | RULER, MemoryAgentBench | No explicit overwrite; attention dilution |
| Embedding RAG | Store chunks as embeddings; retrieve top-k | Contriever, NV-Embed-v2 | RULER, MemoryAgentBench | Version ambiguity; stale fact retrieval |
| Structure-augmented RAG | Build graphs/trees over content | GraphRAG, HippoRAG, RAPTOR | MemoryAgentBench | Construction cost; still lacks overwrite semantics |
| CRUD / temporal semantic memory | Explicit ADD/UPDATE/DELETE or temporal validity | MemGPT, Mem0, Zep | LoCoMo, (limited on MemoryAgentBench) | Hard to attribute gains; evaluation gaps |
| Knowledge Objects / overwrite-first memory | Hash identity + version chains | Attention Is Not Retention | Synthetic overwrite tests | Not benchmarked on MemoryAgentBench |

### Closest Prior Work

- **MemoryAgentBench**: Defines the failure and provides baselines, but does not isolate overwrite semantics as a causal factor.
- **Attention Is Not Retention**: Demonstrates deterministic overwrite semantics via Knowledge Objects on a *single-session conversational-learning micro-benchmark* (6 fact-learning exchanges + immediate recall) and diagnoses “version ambiguity” in production memory systems, but does not test multi-hop conflict resolution on MemoryAgentBench/FactConsolidation-MH.
- **Mem0 / MemGPT / Zep**: Provide CRUD-like or temporal memory mechanisms, but their success/failure on long-context FactConsolidation-MH and the causal role of overwrite semantics remain unclear.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MemoryAgentBench | Benchmark + baseline agents | Does not identify causal mechanism for CR failure | Mechanism isolation (B vs C) | Clear attribution: overwrite vs LLM reconciliation |
| Mem0 / MemGPT | CRUD-like memory updates | Update decisions entangle extraction + retrieval + overwrite; not mechanistically attributed on FactCon-MH @262K | Controlled overwrite filtering (B vs C) | Avoids confounding from other pipeline changes; isolates overwrite semantics |
| Knowledge Objects (Zahn et al.) | Hash identity + version chains | Not evaluated on MemoryAgentBench | Minimal LWW instantiation on FactCon | Directly targets benchmark failure mode |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| gpt-4o-mini | API | https://platform.openai.com/docs/models/gpt-4o-mini | Used by MemoryAgentBench as the shared backbone for RAG baselines |

**Training Data (if applicable):**

No training data needed – inference-only.

**Other Resources (if applicable):**
- MemoryAgentBench dataset: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
- MemoryAgentBench code: https://github.com/HUST-AI-HYZ/MemoryAgentBench

**Resource Estimate**:
- **Compute budget**: Inference-only (API) for extraction + QA; no training.
- **API usage**: Dominated by information extraction + answering.
  - Conflict_Resolution split has 8 examples, each with 100 questions; the target setting (FactConsolidation-MH @262K) is 1 example with 100 questions.
  - We will cache IE outputs per chunk and reuse them across B/C.

**IE quality diagnostic (to control the main confound):** We will measure extraction+canonicalization quality with an automatic check: for each question, identify the set of gold answer strings (MemoryAgentBench provides multiple aliases) and test whether **at least one answer string appears verbatim as an object value** in any extracted triple reachable from the question’s planned keys. If this “gold-in-extraction” rate is very low (<30%), we will treat the experiment as inconclusive about overwrite semantics and report that extraction is the bottleneck. We will also report B vs C restricted to the subset of questions where the gold answer is present in extraction, to isolate the effect of version filtering when information is available.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| FactConsolidation-MH (MemoryAgentBench) | Long-context factual update consolidation requiring multi-hop reasoning over a “knowledge pool” with contradictory updates | **substring_exact_match** accuracy (higher is better; ground-truth answer is checked as a normalized substring of the prediction) | Conflict_Resolution / `factconsolidation_mh_262k` | https://huggingface.co/datasets/ai-hyz/MemoryAgentBench | https://github.com/HUST-AI-HYZ/MemoryAgentBench (`utils/eval_other_utils.py`, `substring_exact_match_score`) |
| FactConsolidation-SH (secondary) | Single-hop version of the same setting | **substring_exact_match** accuracy (higher is better) | Conflict_Resolution / `factconsolidation_sh_262k` | same | same |

### Main Results

Published baselines from MemoryAgentBench Table 2 (262K; [Section 4.2](./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/sections/4.2 Overall Performance Comparison.md)):

| Method | Base Model | Benchmark | FactCon-SH | FactCon-MH | Source | Notes |
|--------|------------|-----------|------------|------------|--------|-------|
| GPT-4o | GPT-4o | FactConsolidation | 60.0 | 5.0 | [Paper](./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/meta/meta_info.txt) | Long-context agent |
| GPT-4o-mini | GPT-4o-mini | FactConsolidation | 45.0 | 5.0 | same | Long-context agent |
| Contriever RAG | GPT-4o-mini | FactConsolidation | 18.0 | **7.0** | same | Best FactCon-MH baseline |
| NV-Embed-v2 RAG | GPT-4o-mini | FactConsolidation | 55.0 | 6.0 | same | Strong embedding retriever |
| HippoRAG-v2 | GPT-4o-mini | FactConsolidation | 54.0 | 5.0 | same | Structure-augmented |
| Mem0 | GPT-4o-mini | FactConsolidation | 18.0 | 2.0 | same | Production memory |
| MemGPT | GPT-4o-mini | FactConsolidation | 28.0 | 3.0 | same | Agentic memory |

To be verified (this proposal):

| Method | Base Model | Benchmark | FactCon-SH | FactCon-MH | Source | Notes |
|--------|------------|-----------|------------|------------|--------|-------|
| Contriever RAG (official) | gpt-4o-mini | FactConsolidation | **TBD** | **TBD** | MemoryAgentBench codebase | Condition A (run `Embedding_rag_gpt-4o-mini-contriever.yaml`, `retrieve_num=10`) |
| IE + all versions (no overwrite) | gpt-4o-mini | FactConsolidation | **TBD** | **TBD** | - | Condition B |
| **LWW-KO (latest-only overwrite)** | gpt-4o-mini | FactConsolidation | **TBD** | **TBD** | - | Condition C |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| B (all versions) vs C (latest-only) | Only overwrite filtering differs | C > B on MH if overwrite semantics matter |

(We intentionally keep ablations minimal to preserve decisiveness.)

### Analysis (Optional)

- Error breakdown: % of questions where B retrieves both versions and the answerer outputs stale vs updated answer; compare with C.

---

## Success Criteria

**Criterion 1: Overwrite semantics are causal**
- Hypothesis: Condition C (latest-only overwrite) outperforms Condition B (all versions) on FactConsolidation-MH at 262K tokens.
- Validation (primary): C > B by a statistically significant margin under paired bootstrap over the 100 questions (p < 0.05).
- Validation (effect size heuristic): We will treat ≥5 absolute points as “practically meaningful” given baselines are in the 2–7% range; smaller effects will be reported but interpreted cautiously.

**Criterion 2: No catastrophic trade-off on single-hop**
- Hypothesis: Enforcing overwrite should not harm direct single-hop recall substantially.
- Validation: FactCon-SH for C is within 5 points of the best of A/B.

---

## Impact Statement

If deterministic overwrite semantics are shown to be a key causal factor for long-context conflict resolution, practitioners building agent memory systems can prioritize database-like update semantics (keyed identity + last-write-wins filtering) over more complex retrieval re-ranking or multi-agent memory pipelines. This would directly inform the design of production memory layers for assistants that must handle frequent factual updates.

---

## References

- [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions](./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/meta/meta_info.txt) - Hu et al., 2025
- [Attention Is Not Retention: The Orthogonality Constraint in Infinite-Context Architectures](./references/Attention-Is-Not-Retention-The-Orthogonality-Constraint-in-Infinite-Context-Architectures/meta/meta_info.txt) - Zahn et al., 2026
- [AMA: Adaptive Memory via Multi-Agent Collaboration](./references/AMA-Adaptive-Memory-via-Multi-Agent-Collaboration/meta/meta_info.txt) - Huang et al., 2026
- [FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt) - Wei et al., 2026
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](../../papers/paper_summaries/Mem0 Building Production-Ready AI Agents with Scalable Long-Term Memory.md) - Chhikara et al., 2025
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) - Packer et al., 2023
- [GraphRAG](https://arxiv.org/abs/2404.16130) - 2024
- [HippoRAG-v2](https://arxiv.org/abs/2405.14831) - 2024
- [RAPTOR](https://arxiv.org/abs/2401.18059) - 2024
- [MQUAKE](https://arxiv.org/abs/2305.14795) - Zhong et al., 2023
- [ROME](https://arxiv.org/abs/2202.05262) - 2022
- [MEMIT](https://arxiv.org/abs/2210.07229) - 2022
- [AlphaEdit](https://arxiv.org/abs/2410.02355) - 2024
- [Zep](https://arxiv.org/abs/2501.13956) - 2025
- [From RAG to Memory: Non-parametric continual learning for LLMs](https://arxiv.org/abs/2502.14802) - 2025
- [LoCoMo](https://arxiv.org/abs/2402.17416) - 2024
- [LongMemEval](https://arxiv.org/abs/2410.10813) - 2024
