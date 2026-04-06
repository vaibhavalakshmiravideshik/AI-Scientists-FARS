# untitled

# Typed-DSL Constrained Data Recipes for Higher Executability in DataChef

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly adapted to specific domains (e.g., medicine, finance, programming) by fine-tuning on curated instruction-following datasets. In practice, the main bottleneck is often not the optimizer or the model architecture, but constructing the *training data pipeline*: selecting relevant raw datasets, transforming them into a consistent instruction format, filtering low-quality examples, and mixing multiple sources.

Recent work has started to automate this data engineering loop. **DataChef** frames *end-to-end data recipe generation* as: given a target benchmark and a pool of candidate datasets, generate a complete “data recipe” consisting of (i) a data processing plan and (ii) executable code that produces a fine-tuning dataset, then optimize recipe generation with reinforcement learning (Group Relative Policy Optimization; **GRPO**, a PPO-style policy-gradient method using group-relative advantages) using a proxy “Data Verifier” reward instead of expensive downstream training feedback **[DataChef](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/meta/meta_info.txt)**. In DataChef’s evaluation, each task samples **N=32** candidate recipes and reports **DVS_avg@32** (average verifier score; execution failures count as 0) and **DBS** (downstream benchmark score for a randomly sampled valid recipe) **[DataChef](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/Setups.md)**. The Data Verifier is empirically correlated with downstream performance (average Pearson r≈0.59 across 6 tasks) **[Data Verifier](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/Data%20Verifier.md)**.

However, generating *free-form Python* recipes creates a practical brittleness: some sampled recipes fail to execute or fail to produce a valid training dataset, yielding sparse rewards and limiting exploration. (The DataChef paper motivates this qualitatively but does **not** report a single headline “executable rate” number; our Stage 0 audit measures this directly for the target tasks.) DataChef explicitly notes that training from scratch with RL is difficult due to “low executability of data recipes,” and introduces a supervised **cold-start** phase (an initial SFT warm-start before RL) that filters demonstrations by execution success **[DataChef](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/End-to-end%20Data%20Recipe%20Generation.md)**. This suggests a potentially large inefficiency even at inference time: best-of-N recipe search may waste samples on structurally invalid programs.

A natural response is to constrain generation to a structured form. In other domains, grammar- or schema-constrained generation improves syntactic validity (e.g., **XGrammar** and **SynCode**) **[XGrammar](./references/XGrammar-Flexible-and-Efficient-Structured-Generation-Engine-for-Large-Language-Models/meta/meta_info.txt)** **[SynCode](./references/Improving-LLM-Code-Generation-with-Grammar-Augmentation/meta/meta_info.txt)**, schema-based RL improves JSON validity **[SchemaRL](./references/Learning-to-Generate-Structured-Output-with-Schema-Reinforcement-Learning/meta/meta_info.txt)**, and purpose-built DSLs can improve multi-step pipeline code generation **[Anka](./references/Anka-A-Domain-Specific-Language-for-Reliable-LLM-Code-Generation/meta/meta_info.txt)**.

### The Problem

DataChef’s recipes must be executable and must satisfy a strict output format (a *ShareGPT-style dialogue*, i.e., a list of chat messages with `role` fields such as system/user/assistant and `content` strings). Failures can come from multiple sources: syntax errors, missing imports, hallucinated dataset IDs, incorrect field assumptions, or downstream tool/LLM call failures. Today, DataChef mitigates this with prompt constraints and by discarding failed recipes, but does not isolate *which* failure modes dominate, nor whether structured constraints could increase the yield of valid recipes under a fixed sampling budget.

If most failures are *structural* (syntax/formatting, invalid operator compositions, hallucinated dataset IDs), then restricting recipe generation to a typed, schema-validated operator DSL could (i) increase the fraction of valid recipes and (ii) increase verifier-based metrics under the same N-sample budget. If most failures are semantic (dataset field mismatch, external tool failures), then DSL constraints will not help and effort should shift to grounding or error recovery.

### Key Insight and Hypothesis

We hypothesize that a significant fraction of DataChef recipe failures are structural and can be prevented by restricting generation to a typed operator DSL whose terminals are constrained to the provided dataset IDs and operator signatures. If this is true, then under a fixed sampling budget (e.g., N=32), DSL-constrained generation will yield:

- higher executable-recipe rate, and
- higher **DVS_avg@N** and **DVS_max@N** (best-of-N verifier score)

than free-form Python generation.

This hypothesis could be wrong if (i) failures are dominated by semantic/data issues rather than syntax or composition; or (ii) the DSL is too restrictive and prevents task-specific auxiliary code, lowering best-of-N recipe quality even if more recipes execute.

---

## Proposed Approach

### Overview

We propose to replace free-form Python recipe generation with **typed-DSL recipe generation**:

1. The model outputs a JSON object that conforms to a JSON Schema describing a **sequential pipeline** of operators from a fixed library.
2. The JSON is validated against the schema; invalid generations are rejected (or re-sampled).
3. A deterministic compiler converts the validated DSL program into a Python data-processing script that matches DataChef’s expected output format.
4. The existing DataChef code verifier and Data Verifier are run unchanged.

To avoid optimizing for the wrong failure mode, we first run an **automated failure mode audit** to quantify what fraction of failures are DSL-addressable.

### Method Details

**Stage 0: Automated failure mode audit (gate).**
- Sample ~50 recipes from the baseline free-form Python generator on each of two held-out tasks (ClimaQA and OpenFinData from DataChef’s `data/input/test.jsonl`).
- Run the existing code verifier.
- Automatically categorize each failure using deterministic rules over exception types and logs (no human labeling), and compute the fraction of DSL-addressable failures.

We define DSL-addressable failures as:
- Python syntax/parse errors
- Output format violations that could be prevented by schema validation
- Invalid operator compositions (e.g., missing required args)
- Hallucinated dataset identifiers **if** dataset IDs are constrained to an enum in the DSL schema

Failures that are not DSL-addressable include: dataset field/key mismatch (unless we model per-dataset schemas), external tool/LLM call failures, and semantic bugs inside LLM-generated prompts.

**Gate decision rule (to proceed to Stage 2):** proceed if the DSL-addressable fraction is ≥30% on **either** task, and not <20% on the other. Otherwise, refute/pivot (failure modes likely semantic/grounding-dominated).

**Stage 1: Minimal typed DSL (sequential-only).**
- Use a sequential-only DSL (no branching/DAG) to keep compilation simple and reduce confounds.
- A DSL program is a JSON list of steps. Each step has:
  - `op`: enum over operator names
  - `inputs`: references to previous step outputs
  - `args`: JSON object whose fields are type-checked against the operator signature
- Dataset IDs from the task context are exposed as an enum in the schema, preventing hallucinated dataset selection.

**Operator library (initial MVP subset).**
We will implement a minimal operator set that corresponds to common pipeline actions described in DataChef (selection, filtering, formatting, sampling) and aligns with existing operator-library systems (Data-Juicer / DataFlow):
- `SelectDataset(dataset_id, split, config)`
- `FilterByKeyword(text_fields, keywords)`
- `MapToShareGPT(prompt_template_id)`
- `Deduplicate(method)`
- `Sample(n)`
- `Mix(weights)`

**Expressiveness mitigation.** If inspecting *successful* baseline recipes shows heavy reliance on LLM-based augmentation/synthesis (as suggested by DataChef’s ClimaQA case study), we add exactly one additional typed operator `LLMTransform(prompt_template_id, ...)` that wraps DataChef’s existing AIDP toolbox calls, while still constraining outputs to the ShareGPT schema.

(Exact operator set may be adjusted after Stage 0 to target the dominant structural failures.)

**Stage 2: Main comparison conditions.**
We keep the decisive experiment to ≤3 conditions:
1. **Baseline (Python)**: DataChef prompt → free-form Python code output.
2. **Typed DSL (ours)**: typed operator DSL JSON → compile to Python.
3. *(Optional confound control)* **JSON-wrapped Python**: schema-valid JSON containing a free-form `python_code` string, to isolate “structured output format / schema decoding” effects from typed operators.

All conditions use the same base model and sampling budget N. If constrained decoding is used to guarantee valid JSON, it is used in the JSON-based conditions.

### Key Innovations

- **Failure-mode-first gate**: explicitly measure whether recipe invalidity is structural enough to warrant DSL constraints.
- **Typed operator DSL for data recipes**: constrain the action space of recipe generation to valid operator signatures and dataset IDs, rather than relying only on natural-language prompt constraints.
- **Confound control (optional)**: separate “structured JSON output” effects from “typed operator constraints.”

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) automated data pipeline / recipe construction for LLM training, (ii) constrained generation for structured outputs, and (iii) reliability improvements for LLM-generated programs.

Data-pipeline systems such as **Data-Juicer** and **Dataverse** provide operator libraries and ETL abstractions for building data recipes, but do not automate recipe synthesis with an RL-trained generator **[Data-Juicer](./references/Data-Juicer-A-One-Stop-Data-Processing-System-for-Large-Language-Models/meta/meta_info.txt)** **[Dataverse](./references/Dataverse-Open-Source-ETL-\(Extract,-Transform,-Load\)-Pipeline-for-Large-Language-Models/meta/meta_info.txt)**. **Data-Juicer Sandbox** explores operator combinations via a Probe–Analyze–Refine workflow but relies on expensive downstream training feedback **[Data-Juicer-Sandbox](./references/Data-Juicer-Sandbox-A-Comprehensive-Suite-for-Multimodal-Data-Model-Co-development/meta/meta_info.txt)**. **DataFlow** provides a large operator library and an agent to synthesize data-prep workflows, but is not focused on optimizing a recipe-generation policy under DataChef’s proxy reward **[DataFlow](./references/DataFlow-An-LLM-Driven-Framework-for-Unified-Data-Preparation-and-Workflow-Automation-in-the-Era-of-Data-Centric-AI/meta/meta_info.txt)**.

For structured outputs, constrained decoding engines (e.g., **Outlines**, **SynCode**, **XGrammar**) guarantee syntactic validity under grammars/schemas **[Outlines](./references/Efficient-Guided-Generation-for-Large-Language-Models/meta/meta_info.txt)** **[SynCode](./references/Improving-LLM-Code-Generation-with-Grammar-Augmentation/meta/meta_info.txt)** **[XGrammar](./references/XGrammar-Flexible-and-Efficient-Structured-Generation-Engine-for-Large-Language-Models/meta/meta_info.txt)**. Schema-based RL improves models’ native ability to emit valid structured outputs **[SchemaRL](./references/Learning-to-Generate-Structured-Output-with-Schema-Reinforcement-Learning/meta/meta_info.txt)**.

For program reliability, type-aware and monitor-guided decoding enforce semantic constraints beyond syntax **[TypeConstrained](./references/Type-Constrained-Code-Generation-with-Language-Models/meta/meta_info.txt)** **[MGD](./references/Guiding-Language-Models-of-Code-with-Global-Context-using-Monitors/meta/meta_info.txt)**, and DSL design can reduce ambiguity in multi-step pipeline generation **[Anka](./references/Anka-A-Domain-Specific-Language-for-Reliable-LLM-Code-Generation/meta/meta_info.txt)**.

### Related Papers

- **[DataChef](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/meta/meta_info.txt)**: End-to-end data recipe generation for LLM adaptation with RL and a proxy Data Verifier.
- **[Data-Juicer](./references/Data-Juicer-A-One-Stop-Data-Processing-System-for-Large-Language-Models/meta/meta_info.txt)**: Operator-based LLM data processing system enabling configurable data recipes.
- **[Data-Juicer Sandbox](./references/Data-Juicer-Sandbox-A-Comprehensive-Suite-for-Multimodal-Data-Model-Co-development/meta/meta_info.txt)**: Probe–Analyze–Refine workflow for searching operator combinations with downstream training feedback.
- **[Dataverse](./references/Dataverse-Open-Source-ETL-\(Extract,-Transform,-Load\)-Pipeline-for-Large-Language-Models/meta/meta_info.txt)**: Open-source ETL pipeline for LLM data processing with a block-based interface.
- **[DataFlow](./references/DataFlow-An-LLM-Driven-Framework-for-Unified-Data-Preparation-and-Workflow-Automation-in-the-Era-of-Data-Centric-AI/meta/meta_info.txt)**: Operator library and an agent for pipeline synthesis.
- **[DocETL](./references/DocETL-Agentic-Query-Rewriting-and-Evaluation-for-Complex-Document-Processing/meta/meta_info.txt)**: Declarative DSL + agentic rewrites to improve accuracy of LLM-powered document pipelines.
- **[LOTUS](./references/LOTUS-Enabling-Semantic-Queries-with-LLMs-Over-Tables-of-Unstructured-and-Structured-Data/meta/meta_info.txt)**: Semantic operators with query-time optimization and accuracy guarantees.
- **[Outlines / Efficient Guided Generation](./references/Efficient-Guided-Generation-for-Large-Language-Models/meta/meta_info.txt)**: Efficient guided decoding with regex/CFG constraints.
- **[SynCode](./references/Improving-LLM-Code-Generation-with-Grammar-Augmentation/meta/meta_info.txt)**: CFG-based grammar augmentation to reduce syntax errors.
- **[XGrammar](./references/XGrammar-Flexible-and-Efficient-Structured-Generation-Engine-for-Large-Language-Models/meta/meta_info.txt)**: Efficient structured generation engine for CFG/JSON schemas.
- **[Schema Reinforcement Learning](./references/Learning-to-Generate-Structured-Output-with-Schema-Reinforcement-Learning/meta/meta_info.txt)**: RL with fine-grained schema validation to improve structured JSON generation.
- **[JSONSchemaBench](./references/Generating-Structured-Outputs-from-Language-Models-Benchmark-and-Studies/meta/meta_info.txt)**: Benchmarking constrained decoding frameworks on real JSON schemas.
- **[Type-Constrained Decoding](./references/Type-Constrained-Code-Generation-with-Language-Models/meta/meta_info.txt)**: Type-constrained decoding for reducing compilation errors.
- **[Monitor-Guided Decoding](./references/Guiding-Language-Models-of-Code-with-Global-Context-using-Monitors/meta/meta_info.txt)**: Static-analysis-guided decoding for semantic constraints.
- **[Anka](./references/Anka-A-Domain-Specific-Language-for-Reliable-LLM-Code-Generation/meta/meta_info.txt)**: DSL design for reliable multi-step pipeline generation.
- **[Toolformer](./references/Toolformer-Language-Models-Can-Teach-Themselves-to-Use-Tools/meta/meta_info.txt)**: Self-supervised learning for tool use (relevant to data-pipeline tool calls).
- **[Rubrics as Rewards](./references/Rubrics-as-Rewards-Reinforcement-Learning-Beyond-Verifiable-Domains/meta/meta_info.txt)**: LLM-as-judge rubric rewards with GRPO.
- **[DeepSeekMath](./references/DeepSeekMath-Pushing-the-Limits-of-Mathematical-Reasoning-in-Open-Language-Models/meta/meta_info.txt)** and **[PRIME](./references/Process-Reinforcement-through-Implicit-Rewards/meta/meta_info.txt)**: RL methods (GRPO/implicit rewards) relevant to DataChef’s RL context.
- **[On the Robustness of Agentic Function Calling](./references/On-the-Robustness-of-Agentic-Function-Calling/meta/meta_info.txt)**: Robustness issues in structured function-calling outputs.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Data recipe / pipeline systems | Operator libraries + ETL abstractions | Data-Juicer, Dataverse, DataFlow | Downstream model benchmarks; system metrics | Manual or heuristic orchestration |
| Automated recipe search | Search over operator combinations with model feedback | Data-Juicer Sandbox | Task-specific benchmark suites | Expensive downstream training feedback |
| Agentic pipeline synthesis | LLM agents compose and debug pipelines | DataFlow-Agent, DocETL | Pipeline accuracy vs cost | High variance; agent brittleness |
| Structured generation | Constrained decoding / schema RL for valid outputs | Outlines, XGrammar, SynCode, SchemaRL | JSONSchemaBench, SchemaBench | Guarantees are syntactic; semantics can still fail |
| Semantic constraints for code | Type/analysis-guided decoding | Type-constrained decoding, MGD | HumanEval/MBPP; repo completion | Runtime overhead; requires analyzers |

### Closest Prior Work

1. **DataChef**: Trains an LLM policy to emit free-form Python data pipelines, penalizing execution failure and using an LLM judge as a proxy reward. It identifies low executability as a core RL challenge, but does not test whether structured constraints can increase valid-recipe yield under fixed N-sample inference.

2. **Data-Juicer Sandbox / DataFlow**: Use operator pools and workflow abstractions, but do not constrain an RL-trained code generator’s action space and do not use DataChef’s DVS evaluation protocol.

3. **SchemaRL / XGrammar / SynCode / Anka**: Show that structured constraints and DSLs can increase syntactic validity and reliability of generated programs, but do not evaluate in the setting of data recipe generation for LLM adaptation.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| DataChef | Free-form Python recipes + RL with proxy verifier | Many samples fail execution; failure causes not analyzed | Add failure-mode audit + typed DSL constraints | Prevent structural failures; increase valid yield and DVS under fixed N |
| Data-Juicer Sandbox | Operator-pool exploration with downstream feedback | Requires expensive downstream training for evaluation | Use DataChef verifier + compilation to execute cheaply | Faster iteration; direct measure of executability under same verifier |
| DataFlow | Operator library + agent for pipeline synthesis | Not RL-optimized for DataChef-style proxy reward | Use operator-library idea as a constrained action space | Constraining generator reduces invalid actions without full retraining |
| SchemaRL / XGrammar | Enforce JSON/schema validity | Mostly syntactic guarantees | Apply schema/typing to data recipe programs | Data recipes have many structural constraints; schema helps yield |
| Anka | DSL improves multi-step pipeline code generation | Not tied to LLM adaptation recipes | Use DSL principles for data recipes | Data recipe code is a multi-step pipeline; DSL can reduce ambiguity |

---

## Experiments

### Experimental Setup

We follow DataChef’s evaluation protocol where possible.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| DataChef-32B | 32B | https://huggingface.co/yichengchen24/DataChef-32B | Preferred if weights are accessible |
| Qwen3-32B | 32B | https://huggingface.co/Qwen/Qwen3-32B | Fallback model for recipe generation |

**Training Data (if applicable):**
- No new model training in the MVP. This proposal is **inference-only**.

**Other Resources (if applicable):**
- DataChef evaluation harness: `datachef-eval --config test` and configuration via `datachef.config.json` **[Repo README](./references/GitHub-yichengchen24-DataChef/sections/README.md.md)**.
- Held-out tasks and dataset candidates: `data/input/test.jsonl` (6 tasks including ClimaQA) https://github.com/yichengchen24/DataChef/blob/main/data/input/test.jsonl .

**Resource Estimate**:
- **Compute budget**: primarily CPU for running generated scripts + API calls for LLM generation and Data Verifier scoring.
- **GPU memory**: none required if using API models; if local inference is required for a 32B model, expect ~4–8×A100 80GB for a few hours.
- **API usage** (order-of-magnitude): Stage 0 audit: ~2 tasks × 50 = **100** generations (plus CPU execution). Stage 2 main: ~2 tasks × 2 conditions × 32 = **128** generations + verifier calls (plus any re-sampling due to schema rejection).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| DataChef held-out task: ClimaQA | Climate question answering held-out task in DataChef | Executable rate; DVS_avg@32; DVS_max@32 | `test.jsonl` task | https://github.com/yichengchen24/DataChef/blob/main/data/input/test.jsonl | DataChef code verifier + Data Verifier |
| DataChef held-out task: OpenFinData | Finance / keyword & information extraction held-out task in DataChef | Executable rate; DVS_avg@32; DVS_max@32 | `test.jsonl` task | https://github.com/yichengchen24/DataChef/blob/main/data/input/test.jsonl | DataChef code verifier + Data Verifier |

**Primary metrics (recipe-level):**
- **Executable rate**: fraction of sampled recipes whose scripts run and produce non-empty training data.
- **DVS_avg@32**: mean Data Verifier Score across N recipes (failed execution counted as 0) **[DataChef](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/Setups.md)**.
- **DVS_max@32**: maximum verifier score among N recipes (best-of-N proxy).

**Published reference baselines (from DataChef Table 1; for context).**
Below are the published DVS_avg@32 / DBS numbers for the two target held-out tasks, copied from DataChef Table 1 (DBS is the downstream benchmark score; the paper additionally reports a normalized “Average” column relative to SOURCE_best=100) **[DataChef Setups/Table 1](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/Setups.md)**.

| Method | Recipe generator | ClimaQA DVS_avg@32 | ClimaQA DBS | OpenFinData DVS_avg@32 | OpenFinData DBS | Source |
|---|---|---:|---:|---:|---:|---|
| Qwen3-32B | Qwen3-32B | 20.6 | 35.6 | 34.9 | 23.8 | DataChef Table 1 |
| Qwen3-Next ⊕ Kimi-K2 | Qwen3-Next-80B + Kimi-K2-Instruct | 41.5 | 42.6 | 54.7 | 64.0 | DataChef Table 1 |
| Gemini-3-Pro | Gemini-3-Pro | 58.4 | 44.3 | 54.9 | 61.8 | DataChef Table 1 |
| DataChef-32B | DataChef-32B | 57.3 | 42.1 | 67.0 | 63.9 | DataChef Table 1 |

We use these numbers as a **sanity check**: our reproduced Baseline(Python) with DataChef-32B on these tasks should be in the same ballpark, modulo API/model/version differences.

### Main Results

| Method | Base Model | Benchmark | Executable rate | DVS_avg@32 | DVS_max@32 | Source | Notes |
|--------|------------|-----------|-----------------|------------|------------|--------|-------|
| Baseline (Python) | DataChef-32B or Qwen3-32B | ClimaQA task | TBD | TBD | TBD | To be verified | Free-form Python recipes |
| **Typed DSL (ours)** | DataChef-32B or Qwen3-32B | ClimaQA task | TBD | TBD | TBD | To be verified | Typed operator DSL compiled to Python |
| Baseline (Python) | DataChef-32B or Qwen3-32B | OpenFinData task | TBD | TBD | TBD | To be verified | Free-form Python recipes |
| **Typed DSL (ours)** | DataChef-32B or Qwen3-32B | OpenFinData task | TBD | TBD | TBD | To be verified | Typed operator DSL compiled to Python |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Typed DSL w/o dataset-ID enum | Allow free-form dataset IDs | Lower executable rate due to hallucinated dataset references |
| JSON-wrapped Python | Schema-valid JSON wrapper around free-form python_code | Isolates “structured output / constrained decoding” from typed operators |

---

## Success Criteria

**Criterion 1: DSL-addressable headroom exists (gate)**
- Hypothesis: ≥30% of baseline failures fall into DSL-addressable categories (as defined above).
- Validation: Automated failure mode audit over ~50 baseline recipes.

**Criterion 2: DSL improves recipe yield and proxy reward**
- Hypothesis: Typed DSL increases executable rate and improves DVS_avg@32 and/or DVS_max@32 under fixed N.
- Validation: On **two** held-out tasks (ClimaQA and OpenFinData), compare baseline vs typed DSL at identical N.

---

## Impact Statement

If typed DSL constraints reliably increase the yield of valid data recipes under fixed best-of-N sampling, practitioners building automated data curation systems can spend fewer samples on invalid pipelines and obtain higher-quality training data with the same budget. This would make DataChef-like recipe generation more robust and easier to deploy, and it suggests a general design pattern: constrain LLM-generated pipeline “code” to typed operator programs when the goal is reliable execution.

---

## References

- [DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/meta/meta_info.txt) - Chen et al., 2026
- [GitHub - yichengchen24/DataChef](./references/GitHub-yichengchen24-DataChef/sections/README.md.md) - Code repository
- [Data-Juicer: A One-Stop Data Processing System for Large Language Models](./references/Data-Juicer-A-One-Stop-Data-Processing-System-for-Large-Language-Models/meta/meta_info.txt) - Chen et al., 2023
- [Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development](./references/Data-Juicer-Sandbox-A-Comprehensive-Suite-for-Multimodal-Data-Model-Co-development/meta/meta_info.txt) - Chen et al., 2024
- [Dataverse: Open-Source ETL Pipeline for Large Language Models](./references/Dataverse-Open-Source-ETL-\(Extract,-Transform,-Load\)-Pipeline-for-Large-Language-Models/meta/meta_info.txt) - Park et al., 2024
- [DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation](./references/DataFlow-An-LLM-Driven-Framework-for-Unified-Data-Preparation-and-Workflow-Automation-in-the-Era-of-Data-Centric-AI/meta/meta_info.txt) - Liang et al., 2024
- [DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing](./references/DocETL-Agentic-Query-Rewriting-and-Evaluation-for-Complex-Document-Processing/meta/meta_info.txt) - Shankar et al., 2024
- [LOTUS: Enabling Semantic Queries with LLMs Over Tables of Unstructured and Structured Data](./references/LOTUS-Enabling-Semantic-Queries-with-LLMs-Over-Tables-of-Unstructured-and-Structured-Data/meta/meta_info.txt) - Patel et al., 2024
- [Efficient Guided Generation for Large Language Models (Outlines)](./references/Efficient-Guided-Generation-for-Large-Language-Models/meta/meta_info.txt) - Willard & Louf, 2023
- [SynCode: LLM Generation with Grammar Augmentation](./references/Improving-LLM-Code-Generation-with-Grammar-Augmentation/meta/meta_info.txt) - Ugare et al., 2024
- [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](./references/XGrammar-Flexible-and-Efficient-Structured-Generation-Engine-for-Large-Language-Models/meta/meta_info.txt) - Dong et al., 2024
- [JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models](./references/Generating-Structured-Outputs-from-Language-Models-Benchmark-and-Studies/meta/meta_info.txt) - Geng et al., 2025
- [Learning to Generate Structured Output with Schema Reinforcement Learning](./references/Learning-to-Generate-Structured-Output-with-Schema-Reinforcement-Learning/meta/meta_info.txt) - Lu et al., 2025
- [Type-Constrained Code Generation with Language Models](./references/Type-Constrained-Code-Generation-with-Language-Models/meta/meta_info.txt) - Mündler et al., 2025
- [Guiding Language Models of Code with Global Context using Monitors](./references/Guiding-Language-Models-of-Code-with-Global-Context-using-Monitors/meta/meta_info.txt) - Agrawal et al., 2023
- [Anka: A Domain-Specific Language for Reliable LLM Code Generation](./references/Anka-A-Domain-Specific-Language-for-Reliable-LLM-Code-Generation/meta/meta_info.txt) - Al Mazrouei, 2025
- [Toolformer: Language Models Can Teach Themselves to Use Tools](./references/Toolformer-Language-Models-Can-Teach-Themselves-to-Use-Tools/meta/meta_info.txt) - Schick et al., 2023
- [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](./references/Rubrics-as-Rewards-Reinforcement-Learning-Beyond-Verifiable-Domains/meta/meta_info.txt) - Gunjal et al., 2025
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](./references/DeepSeekMath-Pushing-the-Limits-of-Mathematical-Reasoning-in-Open-Language-Models/meta/meta_info.txt) - Shao et al., 2024
- [Process Reinforcement through Implicit Rewards (PRIME)](./references/Process-Reinforcement-through-Implicit-Rewards/meta/meta_info.txt) - Cui et al., 2025
- [On the Robustness of Agentic Function Calling](./references/On-the-Robustness-of-Agentic-Function-Calling/meta/meta_info.txt) - Rabinovich & Anaby-Tavor, 2025
