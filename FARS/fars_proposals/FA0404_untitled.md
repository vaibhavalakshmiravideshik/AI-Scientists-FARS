# untitled

# Hazard-Signature Tombstones: Commit-Time Forget Lockout Against Paraphrase Re-Injection in Agent Memory

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

LLM assistants and agents increasingly use **persistent external memory** to store user preferences, prior task traces, and reusable “experiences” across sessions. This improves long-horizon capability and reduces prompt costs, but it also creates a safety and privacy surface: once a harmful or sensitive memory is stored, it can be retrieved and reused long after the original interaction.

Recent work shows that this risk is not hypothetical. **MemoryGraft** demonstrates that a small poisoned subset of “experience” records can dominate retrieval (high *Poisoned Retrieval Proportion*, **PRP**: fraction of retrieved memories that are poisoned) and cause persistent behavioral drift, even without explicit triggers, because the agent imitates retrieved “successful” past solutions ([MemoryGraft](./references/MemoryGraft-Persistent-Compromise-of-LLM-Agents-via-Poisoned-Experience-Retrieval/meta/meta_info.txt)). On the privacy side, **PersistBench** and **CIMemories** show that injecting long-term memory into assistant prompts can induce high rates of cross-domain leakage and inappropriate disclosure across contexts (PersistBench reports **median failure rates** of **53%** on cross-domain leakage and **97.8%** on memory-induced sycophancy; [PersistBench](./references/PersistBench-When-Should-Long-Term-Memories-Be-Forgotten-by-LLMs/sections/Main Results.md), [CIMemories](./references/CIMemories-A-Compositional-Benchmark-for-Contextual-Integrity-of-Persistent-Memory-in-LLMs/meta/meta_info.txt)).

In deployments, users and operators therefore need a robust “forget” primitive (e.g., GDPR-style deletion or incident response after detecting memory poisoning). Many memory systems expose **DELETE by ID** (e.g., Mem0-style CRUD decisions), but the semantic meaning of deletion is underspecified: if the same harmful content re-enters later under a paraphrase or a new key, does the system treat that as a “new” memory and re-store it?

### The Problem

**Problem statement.** We study a practical failure mode of forget requests in LLM-agent memory:

> After a user/operator deletes a harmful memory item, an adversary (or a buggy ingestion pipeline) can re-add semantically equivalent content under a new ID (e.g., via paraphrasing), “resurrecting” the forgotten behavior.

This is common in memory poisoning settings where attackers can repeatedly interact with the agent (query-only injection), and in enterprise settings where multiple integrations can write the “same” fact in different forms.

**Why this is not already solved by existing deletion/invalidation mechanisms.**

- **ID-based deletion (CRUD)** removes a specific record but does not define what should happen if the same content is reintroduced with a different surface form ([Mem0](./references/Mem0-Building-Production-Ready-AI-Agents-with-Scalable-Long-Term-Memory/meta/meta_info.txt)).
- **Temporal invalidation (“invalidate but don’t discard”)** in bitemporal memory graphs (graphs that track both real-world time and system time for each fact; e.g., Zep/Graphiti) is designed for evolving facts and historical queries, not for strict “never re-store this class of harmful content” policies ([Zep](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt)).
- **Retrieval-time filters** can drop known-bad items from the retrieved set, but if bad items remain in the index they can still (i) consume retrieval budget (slot wasting) and (ii) degrade benign recall by crowding the nearest-neighbor region, requiring backfill/oversampling and raising latency.

**Why prompting and inference-time scaling are unlikely to solve it.** If the memory subsystem retrieves harmful demonstrations, better prompting or best-of-\(N\) sampling does not reliably prevent the agent from copying them, because the retrieved context is treated as high-trust evidence and can directly shape the agent’s action pattern (the mechanism exploited by MemoryGraft).

### Key Insight and Hypothesis

**Key insight.** Robust forget requests require a deletion key that is stable under paraphrase, but naïve “semantic delete” based on embedding similarity thresholds is hard to calibrate and brittle under distribution shift. Instead, we propose to map each memory item into a **discrete hazard signature**: a small structured label capturing the safety-relevant behavior pattern (e.g., “skip validation”, “remote code execution”), extracted by an LLM with deterministic decoding.

This hazard signature plays the role of a **CRDT** (Conflict-free Replicated Data Type) style “element key” for **tombstones**: a forget request stores a tombstone over the signature, and future writes that match the tombstone are rejected unless the user explicitly re-consents.

**Hypothesis.** On a realistic memory-poisoning seed corpus (MemoryGraft), commit-time hazard-signature tombstones will:

1. **Block paraphrase re-injection**: \(\ge\)90% of paraphrased poisoned records will be rejected at write time (because the hazard signature is stable under paraphrase).
2. **Preserve benign retrieval utility** better than retrieval-time denylisting with backfill, measured by higher benign Recall@3, because poisoned paraphrases never enter the index and cannot crowd retrieval neighborhoods.

**Why we could be wrong.** (i) Hazard signatures may be unstable under paraphrase (low invariance), making lockout ineffective. (ii) The signature taxonomy may be too coarse, causing benign collisions and false-blocking. (iii) Retrieval-time filtering with sufficient backfill may match commit-time lockout, implying that commit-time enforcement is unnecessary.

---

## Proposed Approach

### Overview

We propose **Hazard-Signature Tombstones (HST)**: a commit-time “forget lockout” policy for agent memory.

- Each memory record \(m\) (e.g., an experience seed with fields `req`, `resp`) is mapped to a discrete **hazard signature** \(\mathrm{HS}(m)\) by an LLM constrained to output a JSON schema.
- A forget request over a set of memory IDs \(F\) tombstones the corresponding signatures \(\{\mathrm{HS}(m): m\in F\}\).
- Any future memory write \(m'\) is **rejected** if \(\mathrm{HS}(m')\) is tombstoned, unless an explicit `reconsent=true` flag is provided.

We compare HST against a strong engineering baseline: retrieval-time hazard-signature denylisting with **backfill** (retrieve more candidates, filter, then take top-\(k\)).

### Method Details

#### Memory record format

We assume each memory record has:
- `id`: unique identifier
- `req`: natural language request / task description
- `resp`: suggested solution / action trace
- (optional) `tags`

In MemoryGraft (MetaGPT’s **DataInterpreter** agent: a data-analysis agent that writes and executes code), there are 100 benign and 10 poisoned experience seeds in this format.

#### Hazard signature schema

We use a fixed discrete taxonomy.

**Hazard categories.** We choose these labels to match the dominant unsafe patterns explicitly instantiated in MemoryGraft’s poisoned seeds (skipping validation gates, executing remote scripts, forcing “success” indicators, uploading/exfiltrating artifacts, and disabling audit trails; see MemoryGraft’s attack/setup description and example poisoned seeds such as `exp_poison_002/005/006/008/009` in the appendix JSON sections). Using a *small* label set is intentional: it should improve paraphrase invariance relative to continuous embedding thresholds, at the cost of potential benign collisions (which we measure).

- `hazards ⊆ {skip_validation, remote_exec, external_upload, force_success, disable_audit, other}`
- `task_type ∈ {data_cleaning, eda, modeling, reporting, pipeline_debug, other}`
- `target_artifact ∈ {dataset, code, report, model, other}`

**"other" handling.** `other` is reserved for cases where none of the above hazards apply. If tombstoned signatures frequently collapse to `other` (high collision risk), we treat this as a *pivot* signal and refine the taxonomy (e.g., split `other` by adding one more field such as `action={shell,python,sql,api}`).

The hazard signature is the canonical string:

`HS = task_type | target_artifact | sorted(hazards)`

We compute \(\mathrm{HS}(m)\) using an LLM prompt with deterministic decoding (temperature=0) that must output valid JSON. We cache \(\mathrm{HS}(m)\) as metadata on each record.

#### Forget semantics

A forget request is represented as:

- `Forget(ids=[...], reconsent=false)`

HST creates a tombstone set `T = { HS(m_id) }` for the forgotten IDs.

- **Commit-time rule (ours):** reject any new memory write \(m'\) if \(\mathrm{HS}(m') \in T\).
- **Re-consent escape hatch (out of scope for main experiment):** allow storing \(m'\) if `reconsent=true` is explicitly set by the user/operator. We disable re-consent in all attack simulations.

#### Retrieval

We use dense retrieval over record text `text(m)=req + "\n" + resp`.

- Embed each record with an embedding model (default: `BAAI/bge-m3`).
- Build a FAISS index.
- Retrieve top-\(k\) for each query by cosine similarity.

We set \(k=3\) for all main comparisons.

### Key Innovations

1. **A concrete semantics for “forget” under semantic re-injection:** forget is not only ID deletion but a **lockout** over a stable behavioral signature.
2. **Discrete hazard signatures as tombstone keys:** avoids continuous similarity thresholds and reframes semantic deletion as a schema-constrained classification problem.
3. **Decision-oriented comparison of commit-time vs retrieval-time enforcement:** tests whether preventing index pollution at write time provides measurable utility over strong retrieval-time backfill baselines.

---

## Related Work

### Field Overview

Agent memory systems span retrieval-augmented generation (RAG) over stored text, structured knowledge graphs with temporal validity, and “memory operating system” abstractions with explicit CRUD operations. As memory becomes persistent and shared across sessions, security and privacy failure modes emerge: poisoned memories can be retrieved as demonstrations (memory poisoning), and irrelevant private attributes can leak across contexts (contextual integrity violations).

Our proposal focuses on a specific but under-specified primitive across many systems: **DELETE / forget requests**. Prior work often treats deletion as a database operation (remove one row) or as temporal invalidation (mark outdated facts), but these semantics do not address **semantic re-addition** where the same unsafe behavior is reintroduced under a new representation.

### Related Papers

- **[MemoryGraft](./references/MemoryGraft-Persistent-Compromise-of-LLM-Agents-via-Poisoned-Experience-Retrieval/meta/meta_info.txt)**: Trigger-free persistent memory poisoning via semantically retrieved unsafe experiences; provides a seed corpus we reuse. MemoryGraft also discusses defenses like cryptographic provenance attestation (verify write provenance) and constitutional reranking (filter at retrieval time), which are complementary to our deletion-semantics layer (content lockout after an explicit forget request).
- **[AgentPoison](https://arxiv.org/abs/2404.11436)**: Optimizes poisoned memories/KB entries to be highly retrievable, enabling targeted behavior manipulation.
- **[MINJA / query-only memory injection](https://arxiv.org/abs/2407.07595)**: Shows an attacker can induce agents to store malicious memories via ordinary interactions.
- **[A-MemGuard](https://arxiv.org/abs/2502.13176)**: Consensus-style memory poisoning defense at retrieval/action time; complementary to our work because it suppresses suspicious memories *when read*, while we enforce a “never re-store this hazard signature” policy *when written* after a forget request.
- **[Memory Poisoning Attack and Defense on Memory Based LLM-Agents](https://arxiv.org/abs/2502.16957)**: Studies poisoning under realistic conditions and highlights calibration difficulties of trust scores.
- **[VIGIL](https://arxiv.org/abs/2601.05755)**: Verify-before-commit for tool-stream injection; conceptually similar to commit-time enforcement.
- **[Mem0](./references/Mem0-Building-Production-Ready-AI-Agents-with-Scalable-Long-Term-Memory/meta/meta_info.txt)**: Production memory with ADD/UPDATE/DELETE decisions; does not define lockout semantics for re-addition.
- **[MemGPT](https://arxiv.org/abs/2310.08560)**: Memory-as-OS with explicit memory operations; a representative CRUD-style memory manager.
- **[Zep / Graphiti](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt)**: Bitemporal knowledge graph with edge invalidation (“invalidate but don’t discard”) for evolving facts.
- **[TeleMem](https://arxiv.org/abs/2601.06037)**: DAG-structured memory with add/delete/update/no-op consolidation; does not study forget lockout under adversarial re-injection.
- **[Text2Mem](./references/Text2Mem-A-Unified-Memory-Operation-Language-for-Memory-Operating-System/meta/meta_info.txt)**: Defines a standardized operation language including Delete/Lock/Expire; does not evaluate semantic re-injection robustness.
- **[CIMemories](./references/CIMemories-A-Compositional-Benchmark-for-Contextual-Integrity-of-Persistent-Memory-in-LLMs/meta/meta_info.txt)**: Benchmark for contextual privacy with persistent memory; motivates reliable memory governance primitives.
- **[Privacy Collapse](https://arxiv.org/abs/2601.15220)**: Shows benign fine-tuning can degrade contextual privacy while preserving utility, motivating system-level controls.
- **[PersistBench](./references/PersistBench-When-Should-Long-Term-Memories-Be-Forgotten-by-LLMs/meta/meta_info.txt)**: Evaluates safety failures induced by long-term memory; supports the importance of reliable forgetting.
- **[MemoryAgentBench](https://arxiv.org/abs/2507.05257)**: Defines “conflict resolution” as a core competency and shows current agents struggle; focuses on contradictory facts, not forget requests.
- **[LoCoMo](https://arxiv.org/abs/2402.17753)**: Long-term conversational memory benchmark used by multiple memory systems.
- **[LongMemEval](https://arxiv.org/abs/2410.10813)**: Interactive long-term memory benchmark with update-related tasks.
- **[KnowMe-Bench](https://arxiv.org/abs/2601.04745)**: Flashback-heavy autobiographical memory benchmark exposing update-time temporal failure modes.
- **[OR-Set / Opt-OR-Set](./references/An-Optimized-Conflict-free-Replicated-Set/meta/meta_info.txt)**: Canonical deletion-tombstone semantics for preventing resurrection in distributed sets.
- **[A comprehensive study of CRDTs](https://hal.inria.fr/inria-00555588)**: Formal background on CRDT convergence and deletion semantics.
- **[Fork, Explore, Commit](https://arxiv.org/abs/2602.08199)**: Uses tombstone markers to prevent file resurrection across copy-on-write branches; an analogy for deletion semantics.
- **[OpenUnlearning](https://arxiv.org/abs/2409.05782)**: Unified benchmarking for parametric unlearning; contrasts with non-parametric external memory deletion.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| CRUD memory deletion | Delete by ID (or LLM-chosen DELETE op) | Mem0; MemGPT | LoCoMo; LongMemEval | Does not define behavior under semantic re-addition |
| Temporal invalidation | Mark outdated facts invalid but keep history | Zep/Graphiti | DMR; LongMemEval | Targets evolving facts, not strict forget/lockout |
| Retrieval-time safety filters | Filter retrieved memories before use | A-MemGuard; other poisoning defenses | AgentPoison/MINJA-style evals | Index still contains poison; requires backfill/latency |
| Memory poisoning attacks | Inject unsafe records so they are retrieved as demonstrations | MemoryGraft; AgentPoison; MINJA | PRP/ASR-style metrics | Persistence across sessions; re-injection possible |
| Memory operation languages | Standardize Delete/Lock/Expire semantics | Text2Mem | (planned) Text2Mem Bench | No empirical study of semantic forget lockout |
| **Ours: hazard-signature tombstones** | Discrete hazard signature as tombstone key; commit-time lockout | This proposal | MemoryGraft seed corpus | Depends on signature stability; collision risk |

### Closest Prior Work

1. **MemoryGraft** demonstrates persistent compromise via experience retrieval and provides a concrete seed corpus, but does not study mitigation through deletion semantics or re-injection after a forget request.
2. **Mem0 / CRUD-style memories** support DELETE operations, but deletion is defined over record IDs and does not address semantic re-addition under paraphrase.
3. **Zep/Graphiti** uses invalidation for temporal contradictions but is not designed for “never re-store this behavioral pattern” lockouts.
4. **Text2Mem** standardizes Delete/Lock/Expire operations with invariants, but does not evaluate whether these operations are robust under paraphrased re-injection or how they affect retrieval utility.
5. **OR-Set / tombstones** provide the core deletion-resurrection framing in distributed systems, but do not address semantic aliasing (how to choose the deletion key).

**Novelty Kill Search Summary:** On 2026-03-01 we searched for combinations of “forget request lockout agent memory”, “semantic tombstone memory”, “re-consent memory delete”, “LLM memory delete semantics paraphrase”, and “CRDT tombstone agent memory”, and we grepped the internal proposal corpus for `forget request|lockout|tombstone|reconsent`. We found work on memory poisoning (MemoryGraft, AgentPoison, MINJA), memory operation APIs (Mem0, Text2Mem), and temporal invalidation (Zep), but no prior work that (i) treats forget requests as **tombstoning a semantic hazard signature** and (ii) directly compares commit-time lockout to retrieval-time denylisting under paraphrased re-injection.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MemoryGraft | Demonstrates persistent compromise via poisoned experience retrieval | No deletion/forget semantics studied | Add a post-forget paraphrase re-injection threat model | Tests a realistic incident-response requirement |
| Mem0 (DELETE) | CRUD-like ADD/UPDATE/DELETE decisions for memory items | Delete is ID-scoped; re-addition is unconstrained | Tombstone a **semantic hazard signature** | Blocks re-addition of forbidden patterns under new IDs |
| Zep/Graphiti | Invalidate edges when contradicted; keep history | Not a strict lockout; focuses on temporal facts | Treat forget as a hard policy boundary | Enforces a “never re-store” rule |
| Retrieval-time filters | Drop known-bad retrieved items; may backfill | Poison still pollutes the index; latency/utility tradeoffs | Commit-time rejection of matching signatures | Prevents index pollution; improves recall at fixed k |
| OR-Set tombstones | Prevent resurrection of deleted elements by key | Requires a stable key; does not address semantic aliasing | Use LLM-derived hazard signature as the key | Makes “semantic tombstones” practical without similarity thresholds |

---

## Experiments

### Experimental Setup

**Dataset (memory seeds).** Use the MemoryGraft seed corpus from the public repo `Jacobhhy/Agent-Memory-Poisoning`:
- `metagpt_attack_poc/payloads/experience_seeds.json` containing 100 benign + 10 poisoned experience records.

**Evaluation queries.** Use the 12 evaluation queries from `metagpt_attack_poc/experiments/exp4_rag_vector_drift.py` (MemoryGraft repository), which probe the same intents as the poisoned seeds (e.g., “skip validation”, “remote script”).

**Threat model: post-forget paraphrase re-injection.**
1. Build an initial memory store with all 110 seeds.
2. Issue a forget request for the 10 poisoned IDs (we treat these as known incident-response targets for the experiment).
3. Generate 5 paraphrases per poisoned seed (50 total) and attempt to re-add them as new memory items.

**Paraphrase generation (automated).** Use `Pro/deepseek-ai/DeepSeek-V3.2` with a fixed prompt and deterministic decoding to generate a JSON list of 5 paraphrases per poisoned seed in one call (temperature=0, top_p=1). This makes the attack deterministic and reproducible.

**Paraphrase diversity note (limitation / optional robustness check).** To avoid overestimating HS-stability due to “easy paraphrases,” we optionally add a *second* paraphrase generator with a different style (e.g., a different API model or a high-temperature variant) for a small subset (e.g., 2 poisoned seeds × 5 paraphrases). We report HS-stability on this harder subset as a robustness diagnostic, but keep the main A/B/C comparison fixed to the deterministic paraphrases.

**Hazard signature extraction (automated).** Use `Pro/deepseek-ai/DeepSeek-V3.2` with deterministic decoding (temperature=0) to compute `HS(m)` for each record.

**Retriever.** Dense retrieval with `BAAI/bge-m3` embeddings and FAISS top-k search.

**Main conditions (≤3).**

Let k=3 and backfill buffer Δ=6.

1. **A) ID-delete only (baseline):** delete the 10 poisoned IDs; accept all new memory writes (including paraphrases).
2. **B) Retrieval-time HS denylist with backfill (strong baseline):** same as A, but at query time retrieve top-(k+Δ)=9 items, drop items whose hazard signature is tombstoned, then return the top-k remaining (if fewer than k remain, leave empty slots).
3. **C) Commit-time HS tombstone lockout (ours):** after forgetting, tombstone the hazard signatures of the deleted items; reject any new write whose hazard signature is tombstoned; retrieval is standard top-k.

**Prereq diagnostic / early-stop (not a main condition).** Before running A/B/C, measure hazard-signature stability under paraphrase:
- For each of the 10 poisoned seeds and its 5 paraphrases, compute whether `HS(paraphrase)==HS(original)`.
- If the exact match rate is <70%, we treat the approach as refuted (signature is not stable enough) and stop.

**Baseline Ladder (REQUIRED):**
- **No-forget baseline (reference only):** retrieval metrics on the original store containing all 110 items (expected high poisoned retrieval).
- **ID-based deletion baseline:** condition A.
- **Strong engineering baseline:** condition B.
- **Proposed method:** condition C.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Pro/deepseek-ai/DeepSeek-V3.2 | API | (provided) | Used for deterministic paraphrasing + hazard signature extraction |
| Pro/BAAI/bge-m3 | embedding | (provided) | Default embedding model for FAISS retrieval |

**Training Data (if applicable):**

No training data needed — inference-only.

**Resource Estimate**:
- **Compute budget**: no GPU training; optional GPU only if embedding is run locally (not required).
- **API usage (order-of-magnitude):**
  - Paraphrase generation: 10 calls (one per poisoned seed), each producing 5 paraphrases.
  - Hazard signature extraction: ~160 calls (110 originals + 50 paraphrases), cached.
- **Wall-clock**: dominated by ~170 API calls; expected to complete in well under a few hours.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MemoryGraft seed corpus | 100 benign + 10 poisoned experience records for a DataInterpreter-style agent memory attack | HS-stability, WriteBlockRate, PRP@k, Benign Recall@3, Benign FalseBlock | N/A | https://github.com/Jacobhhy/Agent-Memory-Poisoning | Custom (JSON load + embedding retrieval + metrics) |

**Primary metrics (fully automated):**
- **HS-stability (higher is better):** fraction of paraphrases whose hazard signature matches their source poisoned seed.
- **WriteBlockRate (higher is better; condition C only):** fraction of paraphrased poisoned writes rejected by commit-time lockout.
- **PRP@k (lower is safer):** poisoned retrieval proportion on the 12 evaluation queries:
  \(\mathrm{PRP@k} = \frac{\#\text{poisoned items retrieved}}{k\times \#\text{queries}}\).
- **Benign Recall@3 (higher is better):** for each benign seed request `req_b`, retrieve top-3 from the post-attack store and check whether the matching benign record ID appears in the top-3.
- **Benign FalseBlock (lower is better; condition C):** fraction of benign records whose hazard signature collides with a tombstoned signature (and would therefore be rejected if re-written).

### Main Results

Published reference: MemoryGraft reports poisoned retrieval proportion **PRP = 47.9%** under a BM25+embedding union retrieval on 12 evaluation queries, with 10/110 poisoned seeds ([MemoryGraft](./references/MemoryGraft-Persistent-Compromise-of-LLM-Agents-via-Poisoned-Experience-Retrieval/sections/Quantitative Results Aggregate Retrieval.md)).

#### Results Table

| Method | Retriever | PRP@3 (12 queries) | Benign Recall@3 (100 self-queries) | WriteBlockRate (poison paraphrases) | Benign FalseBlock | Source | Notes |
|---|---|---:|---:|---:|---:|---|---|
| No-forget (110 seeds, includes poison) | bge-m3 FAISS | TBD | TBD | N/A | N/A | - | Reference only |
| A) ID-delete only | bge-m3 FAISS | TBD | TBD | 0% | 0% | - | Paraphrases re-enter store |
| B) Retrieval-time HS denylist + backfill (k=3, Δ=6) | bge-m3 FAISS | TBD | TBD | 0% | 0% | - | Filters at read-time only |
| **C) Commit-time HS tombstone lockout (ours)** | bge-m3 FAISS | TBD | TBD | TBD | TBD | - | Prevents index pollution |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B without backfill (Δ=0) | Retrieve top-k then filter, no buffer | Utility drops due to slot wasting; shows why backfill is necessary |
| Alternative embedding model | Swap `bge-m3` for `Qwen/Qwen3-Embedding-4B` | Qualitatively similar ranking of A/B/C if method is retrieval-backend-agnostic |

### Experimental Rigor

**Determinism & caching.**
- Paraphrases are generated deterministically and written to a JSON file; all conditions reuse the same paraphrases.
- Hazard signatures are computed once per record and cached.
- Retrieval is deterministic given fixed embeddings.

**Confounders and controls.**
- **Signature leakage / circularity:** PRP@k is computed from known poisoned IDs, not from hazard signature labels, to avoid evaluating the method with its own classifier.
- **Index-size confound:** All conditions use the same k=3 retrieval budget. Condition B is allowed to retrieve k+Δ candidates only as part of its baseline definition; it is measured at the returned top-k.
- **Implementation sanity check:** Verify that on the clean benign-only store (100 items) Benign Recall@3 is near 1.0; otherwise the retriever/index is misconfigured.

---

## Success Criteria

**Hypothesis** (directional):
- Hazard signatures are stable enough under paraphrase to support robust lockout.
- Commit-time lockout reduces poisoned retrieval and preserves benign recall better than retrieval-time filtering with backfill.

**Decision Rule** (concrete — when to stop):
- **Proceed** if all hold:
  1. HS-stability \(\ge\) 70% across the 50 paraphrases.
  2. Condition C WriteBlockRate \(\ge\) 90%.
  3. Condition C reduces PRP@3 by \(\ge\) 5× vs A.
  4. Condition C Benign Recall@3 drops by \(\le\) 2 percentage points vs the benign-only clean-store reference, and exceeds B by \(\ge\) 5 points.
- **Pivot** if HS-stability \(\ge\)70% but Benign FalseBlock > 10%: refine the hazard taxonomy (more specific task_type/artifact) and re-test.
- **Refute** if HS-stability < 70% (signatures not invariant) or if B matches C on Benign Recall@3 and PRP@3 (commit-time lockout not needed).

---

## Impact Statement

If successful, this work would give practitioners implementing persistent agent memory a concrete, testable policy for **robust forget requests**: deleting an unsafe memory item would also prevent it from being silently reintroduced under paraphrase. This is relevant for both incident response to memory poisoning (reducing persistent compromise risk) and privacy compliance workflows where deleted sensitive content must not reappear.

---

## References

- [MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval](./references/MemoryGraft-Persistent-Compromise-of-LLM-Agents-via-Poisoned-Experience-Retrieval/meta/meta_info.txt) - Srivastava & He, 2025
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](./references/Mem0-Building-Production-Ready-AI-Agents-with-Scalable-Long-Term-Memory/meta/meta_info.txt) - Chhikara et al., 2025
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt) - Rasmussen et al., 2025
- [An Optimized Conflict-free Replicated Set](./references/An-Optimized-Conflict-free-Replicated-Set/meta/meta_info.txt) - Bieniusa et al., 2012
- [PersistBench: When Should Long-Term Memories Be Forgotten by LLMs?](./references/PersistBench-When-Should-Long-Term-Memories-Be-Forgotten-by-LLMs/meta/meta_info.txt) - Pulipaka et al., 2026
- [CIMemories: A Compositional Benchmark for Contextual Integrity of Persistent Memory in LLMs](./references/CIMemories-A-Compositional-Benchmark-for-Contextual-Integrity-of-Persistent-Memory-in-LLMs/meta/meta_info.txt) - Mireshghallah et al., 2025
- [Text2Mem: A Unified Memory Operation Language for Memory Operating System](./references/Text2Mem-A-Unified-Memory-Operation-Language-for-Memory-Operating-System/meta/meta_info.txt) - Wang et al., 2025
- [Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](https://arxiv.org/abs/2601.15220) - Goel et al., 2026
- [AgentPoison](https://arxiv.org/abs/2404.11436) - 2024
- [MINJA / Memory Injection Attacks on LLM Agents via Query-Only Interaction](https://arxiv.org/abs/2407.07595) - 2024
- [A-MemGuard](https://arxiv.org/abs/2502.13176) - 2025
- [VIGIL](https://arxiv.org/abs/2601.05755) - 2026