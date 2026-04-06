# untitled

# Entity Anonymization as a Training-Free Context-Faithfulness Boost for Knowledge-Conflict QA

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated evaluation (no human annotation)
  - Inference-only (no fine-tuning required)
  - Fits within ≤768 A100 (80GB) GPU-hours
  - No browser / search-API dependencies

## Introduction

### Context and Motivation

Retrieval-augmented generation (RAG) and other long-context applications rely on a simple behavioral requirement: **when the prompt provides evidence, the model should follow it**. In practice, modern language models often violate this requirement in a specific and deployment-critical way: when the provided context contradicts the model’s pretrained world knowledge, the model may ignore the context and answer from its **parametric memory**. This “knowledge conflict” failure mode can produce stale or incorrect answers even when retrieval is correct, undermining trust in RAG systems for enterprise, medical, and scientific use.

Recent work has made “context faithfulness under knowledge conflicts” a first-class target:
- Benchmarks like **ConFiQA** and **FaithEval** create controlled conflicts between provided context and pretrained knowledge to measure whether models actually follow the context.
- Training-based fixes like **Context-DPO** (preference optimization on faithful-vs-unfaithful pairs), **CLEAR** (conflict detection + attention guidance fine-tuning), and **PIP-KAG/ParamMute** (suppressing parametric knowledge via pruning) can improve faithfulness, but they require model updates, training data, and engineering effort.
- Inference-time methods like **ContextFocus** (Anand et al., 2026; activation steering) improve ConFiQA faithfulness without fine-tuning, but require access to model internals and a custom serving stack.

In contrast, practitioners also value **input-only, training-free mitigations** that can be implemented as simple preprocessing around an existing model and RAG stack (including black-box APIs).

### The Problem

We study a narrowly scoped mechanistic question:

> When a question-answering prompt contains knowledge conflicts, can we improve context faithfulness by removing the surface-form “hooks” that let the model retrieve parametric knowledge?

A large fraction of parametric recall in QA is triggered by **entity surface forms** (names of people, organizations, places). If the model sees a familiar entity name (e.g., “United States”, “Barack Obama”), it can strongly activate memorized associations that compete with context evidence. This suggests a simple intervention: **replace all entity mentions with anonymized placeholders** (e.g., `ENT_1`, `ENT_2`) consistently within the prompt, so that the model cannot directly match the context to memorized entity-level facts.

There is suggestive evidence for this idea in a neighboring domain: work on knowledge-grounded dialogue reports that **entity anonymization increases attachment to external knowledge** by making entity names unrecognizable to the model’s pretrained memory (evaluated with an entity-extraction metric, LLM-KAT). However, it is unclear whether the same idea helps in the more adversarial **knowledge-conflict QA** setting, and whether any improvement is due to anonymization itself rather than just better prompt formatting.

### Key Insight and Hypothesis

**Key insight.** In knowledge-conflict QA, many unfaithful answers are plausibly driven by parametric associations keyed on entity surface forms. If we break these keys by anonymizing entities, the model should shift toward using the provided context.

**Hypothesis.** On ConFiQA (especially the Multi-Conflicts subset), entity-anonymized prompts will improve context faithfulness (higher context-faithful answer rate and lower **memorization ratio** MR) relative to (i) a strong prompting baseline and (ii) an output-space matched control, without materially harming accuracy when the context does not conflict with parametric knowledge.

**Why this could fail.** (i) The model may still answer from parametric priors via type-level patterns (“the capital of X is usually Y”) even without names. (ii) Any gains could be explained by structured prompting rather than anonymization. (iii) Apparent gains could be an artifact of changing the output space (predicting short placeholders is easier than generating multi-token entity names). Our experiments include (a) a **structure+ID output-space matched control (B)** and (b) an orig-context “no-conflict” check to rule out these explanations.

---

## Proposed Approach

### Overview

We propose **Entity-Anonymized Context Prompts (EACP)**: a deterministic, per-instance preprocessing step that replaces entity surface forms in the context and question with consistent placeholders.

The core contribution is a **3-condition mechanism test** that isolates anonymization from (i) structured prompting and (ii) answer-format / output-space effects:

- **A (Base)**: **opinion & instruction (O&I) context-faithful prompting** (Zhou et al., 2023; as used in Context-DPO/ContextFocus) + original context/question. Output is a natural-language answer string.
- **B (Inventory+IDs control)**: same as A, but prepend an **Entity Inventory** that assigns each path entity an ID and **requires the model to answer using an ID** (`ENT_k` or `UNKNOWN`). The inventory includes real names (e.g., `ENT_2 = United States`).
- **C (Anonymized+IDs)**: same as B, but **replace all entity surface forms in the context and question with the IDs** (`ENT_k`) and provide an inventory with **no real names** (only `ENT_k (type=...)`).

The key comparison is **C vs B** (effect of anonymization beyond structure and answer-format constraints).

### Method Details

#### Prompt template (shared; based on Context-DPO O&I)
We base the shared prompt skeleton on the **instr+opin (opinion & instruction; O&I)** template used in the official Context-DPO GitHub evaluation code (`evaluation.py`):

```
Instruction: read the given information and answer the corresponding question.

Bob said "{context}"
Q: {query} in Bob's opinion?
A:
```

Then we prepend the inventory block for B/C (and entity-anonymize the question/context for C).

**Output constraint (for automated scoring):**
- **A**: output the answer string only (no explanation).
- **B/C**: output exactly one ID `ENT_k` or `UNKNOWN` (no explanation).

**Decoding:** `temperature=0` (greedy).

#### Entity set (critical for soundness)
To avoid introducing an error-prone NER component, the entity set is restricted to entities explicitly present in ConFiQA metadata:

- Use entities appearing in `cf_path_labeled` and `orig_path_labeled`.
- For each entity, build a replacement list from `{main label} ∪ {aliases}` (from `orig_alias` / `cf_alias`) plus simple casing variants.

Condition B’s entity inventory uses **the same entity set** as condition C (path entities only).

#### Replacement algorithm
For each instance, create a one-to-one mapping from entities to placeholders `ENT_1, ENT_2, ...`.

Apply replacement to both the context and question:
- Boundary-aware, case-insensitive string replacement
- Longest-first matching to avoid partial-overlap issues
- Post-check: verify that each placeholder introduced appears at least once in the processed context

#### Entity inventory blocks + answer format
- **B (Inventory+IDs control)**: lists `ENT_i = {EntityName_i} (type={TYPE_i})` for path entities and **requires the model to answer with an ID** (`ENT_k` or `UNKNOWN`).
- **C (Anonymized+IDs)**: lists `ENT_i (type={TYPE_i})` for the same path-entity set and also requires an ID answer; **no real names appear anywhere**.

This makes **B and C share the same output space** (IDs), so any difference is attributable to anonymizing entity mentions in the question/context rather than making answers easier to generate.

#### Scoring
- For **A**: model outputs a natural-language answer; evaluate with ConFiQA alias matching to compute Pc/Po/MR/EM.
- For **B and C**: model outputs an ID (`ENT_k`) or `UNKNOWN`; map `ENT_k → entity label` using the per-instance inventory, then score with the same ConFiQA alias matching.

### Key Innovations

- **Training-free conflict mitigation via entity anonymization**: a simple preprocessing rule that targets a specific mechanism (entity-triggered parametric recall) rather than adding training or additional inference loops.
- **Mechanism-isolating controls**: B controls for “inventory structure + ID output constraint helped”, so C vs B isolates the effect of anonymizing entity mentions in the question/context.
- **Deployment-ready recipe**: if effective, EACP is a drop-in wrapper around existing RAG prompts for conflict-heavy domains.

---

## Related Work

### Field Overview

**Knowledge conflicts and context faithfulness.** Multiple benchmarks and studies show that LLMs can prefer parametric memory over provided context when the two disagree. ConFiQA and NQ-Swap create entity-level counterfactual conflicts, and FaithEval broadens the setting to unanswerable, inconsistent, and counterfactual contexts.

**Training-based faithfulness methods.** Context-DPO trains models to prefer context-faithful responses over stubborn ones using preference optimization. CLEAR improves faithfulness by detecting conflict signals in hidden states and encouraging attention to conflicting evidence during fine-tuning. Parametric-suppression methods (e.g., PIP-KAG/ParamMute) attempt to reduce interference from pretrained knowledge by pruning or muting knowledge-associated components.

**Inference-time robustness under misleading context.** Work on distractor robustness (e.g., NoisyBench) and counterfactual evidence discrimination (e.g., CF-RAG) suggests that context errors can strongly bias generation. However, most inference-time methods add multiple retrieval steps or multiple generation passes; our proposal tests whether a single-pass preprocessing trick can address a related failure mode.

### Related Papers

- **[Context-DPO: Aligning Language Models for Context-Faithfulness](https://arxiv.org/abs/2412.15280)**: Introduces ConFiQA and trains models with preference optimization to follow context under knowledge conflicts.
- **[Probing Latent Knowledge Conflict for Faithful Retrieval-Augmented Generation (CLEAR)](https://arxiv.org/html/2510.12460v1)**: Uses hidden-state probes to detect conflicts and applies conflict-aware fine-tuning to improve faithfulness.
- **[PIP-KAG / ParamMute: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning](https://ar5iv.labs.arxiv.org/html/2502.15543)**: Reduces knowledge conflicts by pruning/suppressing parametric knowledge sources and applying preference objectives.
- **[Understanding and Leveraging the Expert Specialization of Context Faithfulness in Mixture-of-Experts LLMs (RouterLens/CEFT)](https://ar5iv.labs.arxiv.org/html/2508.19594)**: Finds context-faithful experts in MoE models and fine-tunes them to improve faithfulness.
- **[COUNTERFACTUAL REASONING FOR RETRIEVAL-AUGMENTED GENERATION (CF-RAG)](https://openreview.net/pdf?id=9U51rOnGko)**: Uses counterfactual queries and causal discrimination scoring to reduce correlation-trap failures in RAG.
- **[Rethinking All Evidence: Enhancing Trustworthy Retrieval-Augmented Generation via Conflict-Driven Summarization (CARE-RAG)](https://ar5iv.labs.arxiv.org/html/2507.01281)**: Detects and summarizes conflicts between internal and retrieved evidence to improve trustworthiness.
- **[Predict the Retrieval! Test-Time Adaptation for Retrieval Augmented Generation](https://arxiv.org/abs/2601.11443)**: Adapts a RAG model at test time via self-supervised passage prediction.
- **[Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards (FaithJudge)](https://ar5iv.labs.arxiv.org/html/2505.04847)**: Builds a human-annotated faithfulness benchmark and shows few-shot annotation-guided judging improves detection.
- **[Contextual Drag: How Errors in the Context Affect LLM Reasoning](https://arxiv.org/abs/2602.04288)**: Shows erroneous context can induce persistent reasoning errors even with explicit verification.
- **[Lost in the Noise: How Reasoning Models Fail with Contextual Distractors](https://arxiv.org/abs/2601.07226)**: Introduces NoisyBench and shows large accuracy drops when reasoning models face contextual distractors.
- **[RAGAS: Retrieval-Augmented Generation Assessment](https://arxiv.org/abs/2309.15217)**: Reference-free RAG evaluation including faithfulness via statement-level verification.
- **[ARES: An Automated Evaluation Framework for RAG](https://arxiv.org/abs/2311.09476)**: Evaluates context relevance and answer faithfulness with trained judges.
- **[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)**: Adds self-reflection tokens to critique retrieved passages and generations.
- **[Adaptive Contrastive Decoding in Retrieval-Augmented Generation](https://arxiv.org/abs/2408.01084)**: A decoding-time method to prefer context-consistent continuations.
- **[CoCoA: Confidence- and Context-Aware Adaptive Decoding for Knowledge Conflicts](https://arxiv.org/abs/2508.17670)**: Uses conflict-aware signals to adapt decoding under knowledge conflicts.
- **[Entity-Based Knowledge Conflicts in Question Answering](https://aclanthology.org/2021.emnlp-main.565.pdf)**: Formalizes knowledge conflicts and introduces entity-swap substitutions (including NQ-style swaps) to measure memorization vs context use.
- **[MQuAKE: Assessing Knowledge Editing under Multi-Hop Edits](https://arxiv.org/abs/2305.14795)**: A benchmark for counterfactual knowledge edits, often used to study conflict behaviors.
- **[CounterFact](https://arxiv.org/abs/2104.08695)**: A knowledge editing benchmark used as a context-faithfulness stress test in some MoE studies.
- **[Improving LLM’s Attachment to External Knowledge in Dialogue Generation (Entity Anonymization + LLM-KAT)](https://arxiv.org/abs/2511.11946)**: Reports that anonymizing entities improves grounding to provided knowledge in knowledge-grounded dialogue.

*(All arXiv IDs above were verified during drafting; see `notes.md` for the query log and paper list.)*

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Preference training for faithfulness | Train model to prefer context-faithful outputs over stubborn outputs | Context-DPO | ConFiQA, NQ-Swap | Requires training + preference data |
| Conflict detection + attention shaping | Detect conflicts and force attention to conflicting evidence | CLEAR | ConFiQA, FaithEval | Needs conflict labeling / decomposition; training |
| Parametric suppression | Remove/mute knowledge components that interfere | PIP-KAG / ParamMute | CoConflictQA, ConFiQA | Risk of removing useful capacity; training |
| Inference-time conflict arbitration | Multiple retrieval/generation paths + re-ranking | CF-RAG, CARE-RAG | HotpotQA, TriviaQA, PopQA, etc. | Higher inference cost; multi-stage |
| Entity anonymization / delexicalization | Remove surface-form access to pretrained entity facts | LLM-KAT anonymization | OpenDialKG | Not tested on knowledge-conflict QA |

### Closest Prior Work

- **Context-DPO (arXiv:2412.15280)**: Uses DPO to train context-faithful models on ConFiQA and related conflict settings. Our work is training-free and instead tests an input transformation that targets the hypothesized parametric-recall mechanism.
- **CLEAR (arXiv:2510.12460)**: Detects conflict signals in hidden states and trains attention guidance to emphasize conflicting evidence. Our approach does not require conflict detection or training and is compatible with any frozen model.
- **PIP-KAG / ParamMute (arXiv:2502.15543)**: Suppresses or prunes parametric knowledge sources to reduce conflicts. Our approach keeps the model unchanged and intervenes only on the prompt text.
- **RouterLens/CEFT (arXiv:2508.19594)**: Identifies and fine-tunes context-faithful experts in MoE models. Our approach does not require MoE architectures and does not update parameters.
- **Entity anonymization for knowledge-grounded dialogue (arXiv:2511.11946)**: Reports improved grounding when entity names are anonymized, but does not study knowledge-conflict QA nor isolate anonymization from prompt-structure confounds. Our proposal provides a controlled test in ConFiQA.

**Novelty Kill Search Summary:** Searched for the exact combination “entity anonymization + context faithfulness + ConFiQA / NQ-Swap / FaithEval”, “delexicalization knowledge conflict QA”, and checked local KB for “anonymization” + “context faithfulness”. No prior work using entity anonymization as the primary training-free mitigation on ConFiQA-style knowledge-conflict QA was found as of 2026-02-22. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Context-DPO | Preference-training for context faithfulness | Requires fine-tuning and preference data | No training; prompt-only transformation | Might capture a large fraction of gains if conflicts are driven by entity-triggered priors |
| CLEAR | Detect conflicts + attention-guided fine-tuning | Multi-component pipeline; training required | Single-pass preprocessing | Eliminates need to detect conflicts explicitly |
| PIP-KAG / ParamMute | Suppress/prune parametric knowledge | Risky capacity removal; training required | No weight changes | Avoids removing useful knowledge pathways |
| CF-RAG / CARE-RAG | Multi-step counterfactual retrieval/arbitration | Higher inference cost | Single pass, no extra retrieval | Cheaper deployment; complementary to retrieval-side fixes |
| Dialogue anonymization (LLM-KAT) | Anonymize entities to improve grounding in dialogue | Not evaluated on conflict QA | Test on ConFiQA + mechanism control | Extends and stress-tests the hypothesis in an adversarial setting |

---

## Experiments

### Experimental Setup

**Main experiment (decisive; 3 conditions).** Evaluate a frozen model on ConFiQA-MC counterfactual contexts under A/B/C (greedy decoding). The key comparison is **C vs B** (anonymization beyond inventory structure + identical ID output space).

**SOTA comparison (required; small).** Run **ContextFocus** (Anand et al., 2026; activation steering) on the same model (Llama-3.1-8B-Instruct) and benchmark (ConFiQA-MC), and also evaluate the composition **ContextFocus + EACP** to test complementarity (run on a 1,500-example subset to match ContextFocus’s evaluation protocol).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| meta-llama/Llama-3.1-8B-Instruct | 8B | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | Primary model (used by ContextFocus; enables fair SOTA comparison) |
| Qwen/Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Optional replication (if compute allows) |

**Training Data (if applicable):**

No training data needed — inference only.

**Other Resources (if applicable):**
- ConFiQA dataset + scripts (from Context-DPO): https://github.com/byronBBL/Context-DPO

**Resource Estimate** (upper bounds; conservative):
- Assume ~6,000 ConFiQA-MC examples.
- Run 3 conditions (A/B/C) with greedy decoding and short outputs (≤32 tokens).
- For a 7B model on 1×A100, budget **≤40 A100-hours** per model for full evaluation (includes prompt preprocessing + overhead).
- ContextFocus baselines D/E run on a **1,500-example subset** (as in the ContextFocus paper) and add only a small overhead.
- (Optional) replication on Qwen2.5-7B-Instruct: +40 A100-hours.
- Total budget: **≤80 A100-hours** (well within 768 A100-hours).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ConFiQA-MC | Multi-hop QA with multiple counterfactual conflicts in context; tests whether model follows context vs parametric memory | Pc (context-faithful rate)↑, Po (original-answer rate)↓, MR=Po/(Pc+Po)↓ (memorization ratio; lower = more context-faithful), EM↑ | test | https://github.com/byronBBL/Context-DPO | ConFiQA scorer from Context-DPO repo (or re-implemented alias matching) |
| ConFiQA-QA (sanity) | Single-hop conflict QA | Pc↑, MR↓, EM↑ | test (subset: 500) | same | same |
| ConFiQA orig_context check | Same questions but using `orig_context` (no conflict) to test “no harm” | EM↑ | test (subset: 500) | same | same |

**Evaluation Scripts:**
- Use the official ConFiQA evaluation from Context-DPO when possible; otherwise implement alias matching as described in the repo.

### Main Results

#### Results Table

(A/B/C/D/E results TBD; must be filled by verification. We include published Context-DPO numbers as a reference point.)

| Method | Base Model | Benchmark | Pc↑ | MR↓ | EM↑ | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| A: O&I prompt (Context-DPO-style) | Llama-3.1-8B-Instruct | ConFiQA-MC | TBD | TBD | TBD | - | Greedy decoding; natural-language answer |
| B: Inventory+IDs control (real names) | Llama-3.1-8B-Instruct | ConFiQA-MC | TBD | TBD | TBD | - | Greedy; answer must be `ENT_k`/`UNKNOWN` |
| **C: EACP (anonymized+IDs)** | Llama-3.1-8B-Instruct | ConFiQA-MC | TBD | TBD | TBD | - | Greedy; same ID output space |
| (Reference) Context-DPO (Qwen2-7B-instruct, Ours) | Qwen2-7B-instruct | ConFiQA-MC | 54.9 | 27.9 | 21.9 | Context-DPO Table 3 | Training-based; not directly comparable weights, but shows ceiling |
| D: ContextFocus (activation steering) + O&I | Llama-3.1-8B-Instruct | ConFiQA-MC (1,500) | TBD | TBD | TBD | ContextFocus (Anand et al., 2026) | SOTA inference-time baseline (requires model internals) |
| E: ContextFocus + **EACP** (composition) | Llama-3.1-8B-Instruct | ConFiQA-MC (1,500) | TBD | TBD | TBD | - | Tests complementarity vs D |
| (Reference) ContextFocus+O&I reported in paper | Llama-3.1-8B-Instruct | ConFiQA-MC (1,500) | P_s=30.00 | MR=38.19 | - | ContextFocus Table 2 | Metric mismatch (p_s vs Pc), but useful sanity check |

#### Comparability Rules (CRITICAL)

- A/B/C must use the **same frozen model weights**, same decoding (greedy), same dataset split, and same scoring.
- Only prompt preprocessing differs between **A/B/C** (plus the pre-declared answer-format constraints: natural-language for A, ID-only for B/C).
- **B and C share the same answer format (ID output)**, so improvements in C cannot be explained by making answers shorter/easier to generate.

### Ablation Studies

We intentionally keep ablations minimal to preserve decisiveness.

| Variant | What’s changed | Expected finding |
|---|---|---|
| Orig-context “no conflict” check (required) | Evaluate on `orig_context` (where context aligns with parametric memory) | C should not reduce EM materially vs A/B; otherwise anonymization harms basic QA |

### Experimental Rigor

- **Determinism / seeds**: A/B/C use greedy decoding and should be effectively deterministic. The optional best-of-8 baseline is stochastic; run with `seeds=[42,123,456]` and report mean±std.
- **Replacement correctness sanity checks**:
  - Verify that (i) `orig_answer` and `cf_answer` entities are in the entity list, and (ii) their surface forms are fully replaced in C.
  - Verify that placeholders do not collide (one-to-one mapping per instance).
- **Output-format validity**:
  - A must output a single-line answer string.
  - B/C must output exactly one token `ENT_k` or `UNKNOWN`.
  - Any extra text is stripped; if parsing fails, score as incorrect.
- **Confound controls**:
  - **Structured prompting / constraints**: B controls for added inventory structure + forced ID output.
  - **Output-space reduction**: orig-context EM check ensures C does not win by making the task trivially easier at the cost of accuracy.

---

## Success Criteria

**Hypothesis** (directional): Entity anonymization improves context faithfulness under knowledge conflicts beyond what is achievable by inventory structure + ID-output constraints alone.

**Decision Rule** (concrete):

- **Proceed** if, on **ConFiQA-MC with Llama-3.1-8B-Instruct**:
  - C improves **Pc** by ≥ **+5 points** over B **and** reduces **MR** by ≥ **5 points**, and
  - C’s EM on the orig-context check is within **2 points** of A/B.

- **Strong proceed (practitioner-relevant)** if EACP is **competitive with or complementary to ContextFocus**:
  - E (ContextFocus+EACP) improves **Pc** by ≥ **+2 points** over D (ContextFocus) on the 1,500-example ConFiQA-MC subset (same evaluation protocol), with no increase in MR.

- **Pivot** if C ≈ B but both improve over A: reinterpret the result as “inventory+ID constraints help; anonymization is not the active ingredient,” and drop the anonymization claim.

- **Refute** if C does not improve over B on Pc/MR (within noise), or if C harms orig-context EM by >2 points, **and** E does not improve over D.

*(Note: ContextFocus reports p_s/p_o/MR rather than ConFiQA Pc/Po/EM; for D/E we will compute Pc/Po/EM with the standard ConFiQA evaluator for consistency.)*

---

## Impact Statement

If successful, entity anonymization would provide a simple deployment knob for practitioners building RAG systems in conflict-prone domains: anonymize entity surface forms in retrieved context and require placeholder answers, reducing parametric override without training or extra inference passes. Even a negative result is decision-relevant: it would indicate that entity surface forms are not the main driver of knowledge-conflict unfaithfulness, motivating focus on training-based or decoding-based conflict mitigation instead.

---

## References

- [Context-DPO: Aligning Language Models for Context-Faithfulness](https://arxiv.org/abs/2412.15280) - Bi et al., 2024
- [ContextFocus: Activation Steering for Contextual Faithfulness in Large Language Models](https://arxiv.org/abs/2601.04131) - Anand et al., 2026
- [FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](https://aclanthology.org/2025.acl-long.1062/) - Zhang et al., ACL 2025
- [TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](https://arxiv.org/abs/2511.10375) - Liu et al., 2025
- [Probing Latent Knowledge Conflict for Faithful Retrieval-Augmented Generation (CLEAR)](https://arxiv.org/html/2510.12460v1) - 2025/2026
- [PIP-KAG / ParamMute: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning](https://ar5iv.labs.arxiv.org/html/2502.15543) - Huang et al., 2025
- [Understanding and Leveraging the Expert Specialization of Context Faithfulness in Mixture-of-Experts LLMs (RouterLens/CEFT)](https://ar5iv.labs.arxiv.org/html/2508.19594) - Bai et al., 2025
- [ConFiQA dataset + code (Context-DPO repo)](https://github.com/byronBBL/Context-DPO) - 2024
- [Improving LLM’s Attachment to External Knowledge in Dialogue Generation (Entity Anonymization + LLM-KAT)](https://arxiv.org/abs/2511.11946) - Sheikhi et al., 2025
- [COUNTERFACTUAL REASONING FOR RETRIEVAL-AUGMENTED GENERATION (CF-RAG)](https://openreview.net/pdf?id=9U51rOnGko) - 2025/2026
- [Rethinking All Evidence: Enhancing Trustworthy Retrieval-Augmented Generation via Conflict-Driven Summarization (CARE-RAG)](https://ar5iv.labs.arxiv.org/html/2507.01281) - Chen et al., 2025
- [Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards (FaithJudge)](https://ar5iv.labs.arxiv.org/html/2505.04847) - Tamber et al., 2025
- [Predict the Retrieval! Test-Time Adaptation for Retrieval Augmented Generation](https://arxiv.org/abs/2601.11443) - Pi, 2025
- [RAGAS: Retrieval-Augmented Generation Assessment](https://arxiv.org/abs/2309.15217) - Shahul Es et al., 2023
- [ARES: An Automated Evaluation Framework for RAG](https://arxiv.org/abs/2311.09476) - Saad-Falcon et al., 2023
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) - Asai et al., 2023
