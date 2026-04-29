# untitled

# Executable FinMR: Arelle-Based Symbolic Baselines and an Executability Audit for XBRL Mathematical Reasoning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Financial regulators and capital markets increasingly rely on **XBRL** (eXtensible Business Reporting Language) filings, where each reported financial number is represented as a typed fact linked to a standardized accounting taxonomy such as **US-GAAP**. XBRL filings are designed to be machine-checkable: in addition to the instance document containing facts, they include linked XML documents (schemas and linkbases) that define labels, presentation structure, dimensional structure, and **calculation relationships**.

Because XBRL is explicitly structured, it is a natural application area for large language models (LLMs) and agents: models could help interpret rule violations, explain issues to humans, or map textual guidance to structured concepts. However, in high-stakes settings like financial reporting, practitioners need to distinguish failures of *reasoning* from failures of *tooling* (e.g., parsing, schema resolution, or deterministic calculation).

The recent **FINAUDITING** benchmark evaluates LLMs on taxonomy-grounded, multi-document reasoning over real US-GAAP XBRL filings across three tasks: semantic matching (FinSM), relationship extraction (FinRE), and mathematical reasoning (FinMR) ([FINAUDITING](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/meta/meta_info.txt)). FinMR is especially challenging: the best reported zero-shot accuracy is only **13.86%** (Fin-o1-14B) on the FinMR test set, and the paper reports very high error rates for most models ([FinMR performance](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/sections/FinMR%20performance.md)).

### The Problem

It is unclear what the FinMR accuracy ceiling reflects. There are at least three plausible explanations:

1. **True reasoning difficulty**: answering requires long-horizon reasoning over hierarchical structures and domain-specific rules.
2. **Representation-to-execution gap**: the input is largely *tool-executable* (because it contains XBRL structure), but current baselines do not use standards-compliant XBRL tooling, so models are asked to “simulate” execution.
3. **Benchmark packaging and evaluation artifacts**: the released dataset may contain missing/truncated XBRL dependencies, double-escaped XML, or evaluation choices (e.g., LLM-as-a-judge) that conflate formatting and reasoning.

A critical missing piece is a **symbolic, standards-compliant baseline** that attempts to execute the provided XBRL artifacts and reports:

- **Coverage**: what fraction of FinMR instances are actually executable from the released artifacts (under a clearly specified connectivity policy)?
- **Correctness on executable instances**: when execution is possible, does a symbolic engine reproduce the benchmark’s gold `{"extracted_value", "calculated_value"}`?
- **Failure taxonomy**: if execution fails, is the cause missing schema/linkbase artifacts needed to build the filing’s **Discoverable Taxonomy Set (DTS)**, ambiguous target selection, or genuinely non-executable rule semantics?

Without such a baseline, it is difficult to interpret whether FinMR primarily measures long-context reasoning or a tooling barrier.

### Key Insight and Hypothesis

**Key insight**: FinMR instances are constructed from real XBRL filings and **DQC (XBRL US Data Quality Committee)** rule violations, and each instance is described in terms of XBRL concepts, periods, units, and calculation/dimensional structure ([Task Formulation](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/sections/Task%20Formulation.md)). If the released FinMR inputs contain a sufficiently self-contained subset of the filing’s XBRL instance + linkbases, then a standards-compliant XBRL processor should be able to deterministically recover the reported fact and recompute the implied value for a substantial subset of instances.

**Hypothesis**: A standards-compliant XBRL engine (**Arelle**) can deterministically reproduce the gold `extracted_value` and `calculated_value` for a large subset of FinMR instances, and will substantially outperform a structure-agnostic “message-only” regex arithmetic baseline on those instances. If this holds, the benchmark should be reported with an **“executable subset”** split and should include tool-based baselines.

This hypothesis could be wrong if the dataset is not self-contained (missing DTS/taxonomy files), if the target fact is underspecified (multiple plausible contexts/units), or if the “calculated_value” in the benchmark is derived from text in the DQC message rather than the XBRL structure.

---

## Proposed Approach

### Overview

We propose an **executability audit + symbolic baseline** for FinMR:

1. **Reconstruct** a per-instance XBRL package from the released `query` field (or attached artifacts if provided).
2. **Execute** the instance with **Arelle** (offline by default) to load the DTS and facts.
3. **Compute** `extracted_value` (reported fact value) and `calculated_value` (rule-implied value) using XBRL semantics:
   - calculation linkbase execution for calculation-consistency regimes,
   - dimension-aware aggregation for dimensional cross-check regimes,
   - simple constraint transformation for sign/negativity regimes.
4. **Evaluate deterministically** using a programmatic re-implementation of the benchmark’s judge rules (Appendix C.3) instead of an LLM-as-a-judge.
5. **Report** (i) accuracy and error rates, (ii) **executability coverage**, and (iii) a **failure taxonomy**.
6. **Release** an “Executable FinMR” subset (IDs) and a reference evaluator so future work can cleanly separate tool-execution from reasoning.

### Method Details

#### 1) Dataset parsing and target extraction

We use `TheFinAI/FinMR` (332 instances; average 35,678 tokens per instance) as reported in Table 4 of FINAUDITING ([Data Annotation](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/sections/Data%20Annotation.md)). Each instance includes a long `query` string and a gold `answer` with `extracted_value` and `calculated_value`.

From the `query`, we extract:
- the **DQC rule ID** (e.g., `DQC_US_0126`, `DQC_US_0117`, `DQC_US_0015`),
- the **target concept QName** (e.g., `us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax`),
- the relevant **period** (instant or duration),
- and (when available) unit/dimensions described in the DQC message.

This step uses only metadata and identifiers; it does **not** read any numeric “gold” values from the query.

#### 2) XBRL package reconstruction

We reconstruct a local package directory per instance:

- Split the `query` into sections corresponding to the XBRL components described by FINAUDITING (instance, schema, presentation linkbase, calculation linkbase, definition linkbase, label linkbase) ([Task Formulation](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/sections/Task%20Formulation.md)).
- Write each component to a canonical filename (e.g., `instance.xml`, `schema.xsd`, `cal.xml`, …).
- Rewrite `schemaRef` / `linkbaseRef` `xlink:href` attributes to local filenames when the referenced documents are included in the query (to avoid network dependence).
- Run Arelle in **offline mode** (`--internetConnectivity offline`) to ensure the baseline is executable under the same “no browsing” constraints as the verification system.

We define an instance as **executable** if Arelle can load the instance and build a ModelXbrl without fatal DTS resolution errors.

#### 3) Computing `extracted_value`

We compute `extracted_value` as the value of the target concept fact in the context specified by the DQC message (period and, when applicable, unit/dimensions). If multiple facts match, we apply a deterministic tie-break:

1. prefer facts with the exact period described in the DQC message,
2. prefer the default dimension when the rule describes a non-dimensional check,
3. otherwise choose the fact participating as a parent in the relevant calculation relationship set.

#### 4) Computing `calculated_value` by rule family

FinMR is derived from DQC rule violations. We implement three rule families (matching the observed FinMR DQC IDs):

- **Calculation-consistency (e.g., DQC_US_0126)**: Use Arelle’s relationship sets for summation-item arcs to obtain the parent’s children and weights under the appropriate link role, then compute the weighted sum over the matching child facts.

- **Dimensional cross-check (e.g., DQC_US_0117)**: Identify the axis and members referenced by the DQC message, retrieve the corresponding dimensional facts for the target concept, and compute the aggregate sum across members; compare to the default (non-dimensionalized) fact.

- **Sign/negativity constraint (e.g., DQC_US_0015)**: The “calculated” correct value is the non-negative counterpart of the reported value when the rule flags an invalid negative. We compute this as `abs(extracted_value)` after parsing the fact value from XBRL.

If required inputs for the rule family are missing (e.g., linkbase not present, axis members not recoverable, or facts missing), we mark the instance as **non-executable** for that rule family.

#### 5) Deterministic evaluator (replace LLM-as-judge)

FINAUDITING evaluates FinMR with an LLM-as-a-judge prompt, but the decision procedure is deterministic: check JSON structure, then numeric meaning equality for `extracted_value` (comma/format-insensitive), then strict numeric meaning equality for `calculated_value` ([C.3 The metrics for FinMR](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/sections/C.3%20The%20metrics%20for%20FinMR.md)).

We implement this deterministically:
- Parse JSON; if missing keys or invalid JSON → **S**.
- Normalize numeric strings (remove commas and spaces; parse sign; use `Decimal`) and compare `extracted_value` → if mismatch → **E**.
- Compare `calculated_value` with exact `Decimal` equality → if mismatch → **C**.
- Otherwise → **A**.

This yields **ACC/SER/EER/CER** and avoids judge model variance.

#### 6) Failure taxonomy and “Executable FinMR” subset

For each non-executable instance, we assign a failure label:
1. **Missing DTS artifact** (schema/linkbase referenced but not present in query)
2. **External dependency** (taxonomy/schema href points to remote URL not included)
3. **Malformed XML / escaping** (cannot parse as XML)
4. **Ambiguous fact selection** (multiple plausible facts; cannot disambiguate from provided metadata)
5. **Rule-specific insufficiency** (e.g., missing axis/member details for dimensional cross-check)

We then release:
- the list of executable instance IDs (“Executable FinMR”),
- and per-category failure counts.

### Key Innovations

- **First standards-compliant XBRL engine baseline for FinMR** (Arelle-based) with explicit executability definition.
- **Deterministic evaluator** aligned to the benchmark’s stated judge procedure, removing judge-model variance.
- **Executable-subset reporting** + failure taxonomy to separate tool-execution issues from reasoning issues.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) **financial reasoning benchmarks**, (ii) **XBRL-focused information extraction and taxonomy alignment**, (iii) **tool-augmented / program-aided reasoning**, and (iv) **evaluation methodology**.

Financial numerical reasoning datasets such as FinQA, ConvFinQA, and TAT-QA focus on tables and text in financial reports, while longer-document settings are studied in MultiHiertt and DocMath-Eval. Separately, XBRL-specific work (FiNER, FNXL, FinTagging, XBRL-Agent) emphasizes tagging and concept linking, which is complementary to numeric consistency verification. Finally, tool-augmented reasoning methods (PAL, Program-of-Thought, Toolformer, MRKL) motivate deterministic execution as a baseline and as a way to decompose “reasoning” into grounding plus execution.

### Related Papers

- **[FINAUDITING](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/meta/meta_info.txt)**: Introduces FinMR and reports very low zero-shot LLM accuracy (13.86% best), motivating a symbolic baseline.
- **[FinTagging](./references/FinTagging-An-LLM-ready-Benchmark-for-Extracting-and-Structuring-Financial-Information/meta/meta_info.txt)**: Benchmarks extraction and concept linking to the US-GAAP taxonomy, highlighting taxonomy alignment difficulty.
- **[FiNER](./references/FiNER-Financial-Numeric-Entity-Recognition-for-XBRL-Tagging/meta/meta_info.txt)**: Studies numeric-heavy entity recognition for XBRL tagging and introduces techniques to handle numeric tokenization.
- **[Financial Numeric Extreme Labelling (FNXL)](https://arxiv.org/abs/2306.03723)**: Frames XBRL tagging as extreme classification with thousands of labels, emphasizing long-tail taxonomy issues.
- **[XBRL-Agent](https://dl.acm.org/doi/10.1145/3677052.3698614)**: Uses an LLM agent with retrieval and a calculator for XBRL analysis, but does not provide a standards-compliant XBRL execution baseline for FinMR.
- **[FinQA](https://arxiv.org/abs/2109.00122)**: Numerical reasoning over financial reports with annotated programs, a canonical finance reasoning benchmark.
- **[ConvFinQA](https://arxiv.org/abs/2210.03849)**: Conversational numerical reasoning in finance, stressing long dependency chains.
- **[TAT-QA](https://arxiv.org/abs/2105.07624)**: QA over hybrid tables+text in financial reports, with arithmetic derivations.
- **[MultiHiertt](https://arxiv.org/abs/2206.01347)**: Numerical reasoning over multi-table hierarchical financial documents, motivating structure-aware pipelines.
- **[DocMath-Eval](https://arxiv.org/abs/2311.09805)**: Evaluates LLM math reasoning on long finance documents with Python-solution supervision.
- **[FinanceReasoning](https://arxiv.org/abs/2506.05828)**: Benchmarks financial numerical reasoning and provides a large financial function library for knowledge augmentation.
- **[FinanceBench](https://arxiv.org/abs/2311.11944)**: Evaluates LLMs for financial QA with evidence requirements, highlighting unreliability in enterprise settings.
- **[PIXIU](https://arxiv.org/abs/2306.05443)**: Provides financial instruction data and benchmarks for financial NLP and prediction tasks.
- **[FinBen](https://arxiv.org/abs/2402.12659)**: A holistic benchmark suite for financial LLMs across many tasks.
- **[FinEval](https://arxiv.org/abs/2308.09975)**: A large benchmark for evaluating LLMs in the financial domain (Chinese focus) including agent/tool dimensions.
- **[BloombergGPT](https://arxiv.org/abs/2303.17564)**: A finance-specialized LLM trained on proprietary and public corpora.
- **[FinGPT](https://arxiv.org/abs/2306.06031)**: Open-source framework for financial LLMs and data pipelines.
- **[Fino1 / Fin-o1](https://arxiv.org/abs/2502.08127)**: Studies transferability of reasoning-enhanced LLMs to finance and introduces Fin-o1 models.
- **[Fin-R1](https://arxiv.org/abs/2503.16252)**: Financial reasoning model trained with RL, included as a baseline family in FINAUDITING.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)**: Describes a strong open LLM family that is a common baseline in finance benchmarks.
- **[Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)**: Describes the Qwen model family (a base for some financial models).
- **[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)**: Describes the Llama model family used widely as open baselines.
- **[Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)**: Describes the Gemma model family used in recent evaluation suites.
- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)**: Few-shot prompting for eliciting reasoning traces.
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)**: Inference-time scaling via sampling and majority vote, a standard baseline for reasoning tasks.
- **[Program-of-Thought Prompting](https://arxiv.org/abs/2211.12588)**: Uses executable code to reduce calculation errors, motivating symbolic execution baselines.
- **[PAL](https://arxiv.org/abs/2211.10435)**: Program-aided language models that offload computation to an interpreter.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: Interleaves reasoning and tool actions, relevant to tool-augmented finance agents.
- **[Toolformer](https://arxiv.org/abs/2302.04761)**: Self-supervised training to call external tools.
- **[MRKL Systems](https://arxiv.org/abs/2205.00445)**: Modular neuro-symbolic architecture combining LMs with discrete reasoning modules.
- **[LLMs-as-Judges Survey](https://arxiv.org/abs/2412.05579)**: Surveys strengths and failure modes of LLM-based evaluation, motivating deterministic evaluation when possible.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Finance numerical reasoning (text+tables) | Retrieve evidence then compute numeric answers | FinQA, ConvFinQA, TAT-QA, MultiHiertt | Execution accuracy / EM / F1 | Evidence retrieval and arithmetic errors dominate |
| XBRL tagging and taxonomy alignment | Map text/numbers to US-GAAP concepts | FiNER, FNXL, FinTagging | NER F1 / concept-linking accuracy | Long-tail labels; close concepts hard to disambiguate |
| Tool-augmented / program-aided reasoning | Separate grounding from deterministic execution | PAL, Program-of-Thought, Toolformer, MRKL, ReAct | GSM8K-style and tool benchmarks | Still needs correct grounding; tool availability mismatch |
| Evaluation methodology | Replace subjective scoring with reliable evaluation | LLMs-as-Judges survey; benchmark papers | Meta-evaluation | LLM judges have bias; deterministic metrics preferred when possible |

### Closest Prior Work

- **FINAUDITING** ([paper](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/meta/meta_info.txt)): Introduces FinMR and evaluates only LLM baselines; it does not provide a standards-compliant XBRL execution baseline nor an executability audit. Our proposal adds a symbolic baseline, coverage reporting, and a deterministic evaluator aligned to Appendix C.3.

- **XBRL-Agent** ([paper](https://dl.acm.org/doi/10.1145/3677052.3698614)): Proposes an LLM agent with retrieval and a calculator for XBRL analysis tasks. It does not attempt standards-compliant XBRL execution, and it does not address FinMR’s benchmark validity via executable-subset reporting.

- **FinTagging / FiNER / FNXL** ([FinTagging](./references/FinTagging-An-LLM-ready-Benchmark-for-Extracting-and-Structuring-Financial-Information/meta/meta_info.txt), [FiNER](./references/FiNER-Financial-Numeric-Entity-Recognition-for-XBRL-Tagging/meta/meta_info.txt), [FNXL](https://arxiv.org/abs/2306.03723)): Focus on extraction and taxonomy alignment, not on calculation-consistency verification or XBRL-tool executability.

- **Program-aided reasoning methods (PAL / Program-of-Thought)** ([PAL](https://arxiv.org/abs/2211.10435), [Program-of-Thought](https://arxiv.org/abs/2211.12588)): Show that deterministic execution can eliminate arithmetic errors, but they do not address XBRL-specific executability or DTS resolution issues.

**Novelty Kill Search Summary:** Searched for exact combinations including “FinAuditing FinMR Arelle baseline”, “TheFinAI FinMR Arelle”, “FINAUDITING symbolic baseline”, and checked local proposal corpus for “FinMR/XBRL/arelle”. No prior work or existing proposals providing an Arelle-based baseline for FinMR were found as of 2026-03-01 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| FINAUDITING | Defines FinMR and evaluates LLMs with an LLM judge | No symbolic baseline; unclear whether failures are tooling vs reasoning | Add standards-compliant XBRL execution + deterministic evaluator + coverage | Directly tests tool executability and sets an execution upper bound |
| XBRL-Agent | LLM + retriever + calculator for XBRL QA tasks | Not standards-compliant XBRL execution; not benchmark audit | Use an XBRL engine (Arelle) to execute the provided artifacts | XBRL engines natively implement DTS parsing, contexts, and linkbase semantics |
| FiNER / FinTagging / FNXL | XBRL tagging and concept linking | Does not validate numeric consistency or recompute totals | Target numeric consistency checks and recomputation | Calculation linkbases provide deterministic structure for recomputation |
| PAL / Program-of-Thought | Program execution reduces arithmetic errors | Not applied to XBRL; does not address executability | Apply deterministic execution to XBRL filings via Arelle | XBRL is designed for deterministic validation, so execution should be high-precision |

---

## Experiments

### Experimental Setup

**Task**: FinMR from FINAUDITING.

- Dataset: `TheFinAI/FinMR` (332 instances; long-context prompts; derived from DQC rule violations)
- Split: use the official `test` split if provided; otherwise use all instances and follow the benchmark’s split policy.

**Primary evaluation**: deterministic ACC/SER/EER/CER (Appendix C.3 replication) + executability coverage.

**Baseline Ladder (REQUIRED):**

- **Level 0 (trivial heuristic)**: regex extraction + arithmetic from the DQC validation message text only (no XML parsing).
- **Level 1 (prompting)**: best published zero-shot LLM baseline from FINAUDITING (Fin-o1-14B, 13.86% ACC on FinMR) ([FinMR performance](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/sections/FinMR%20performance.md)).
- **Level 4 (inference-time scaling, optional if budget allows)**: best-of-4 self-consistency on a 50-instance subset using a long-context frontier model (e.g., `gpt-4.1` or `claude-sonnet-4-5`).
- **Level 5 (tool-based)**: **Arelle symbolic baseline (ours)**.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Fin-o1-14B | 14B | https://huggingface.co/TheFinAI/Fin-o1-14B | Reported best FinMR baseline in FINAUDITING (published numbers) |
| DeepSeek-V3.2 (API) | - | (available via `available_models.md`) | Optional long-context LLM baseline |
| gpt-4.1 (API) | - | (available via `available_models.md`) | Optional long-context LLM baseline |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| N/A | Inference-only; no training required | - | - | - |

**Other Resources (if applicable):**

- Arelle XBRL engine (Apache-2.0): https://github.com/Arelle/Arelle
- XBRL US DQC rule documentation (for understanding rule semantics): https://xbrl.us/data-rule/

**Resource Estimate**:

- **Compute budget**: 
  - Arelle + parsing: CPU-only; expected minutes to a few hours for 332 instances (depends on XML size and DTS resolution).
  - Optional LLM baseline (50 instances): API tokens dominate cost; can be skipped if too expensive.
- **GPU memory**: none for Arelle baseline.
- **API usage**: optional; if run, estimate ~50 prompts × ~35k tokens input each (order-of-magnitude).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| FinMR (FINAUDITING) | XBRL-based numerical consistency with rule-grounded `extracted_value`/`calculated_value` | ACC, SER, EER, CER; executable coverage; failure taxonomy | test | https://huggingface.co/datasets/TheFinAI/FinMR | Custom (deterministic evaluator aligned to Appendix C.3) |

**Evaluation Scripts:**

- Implement dataset loader (`datasets`/pyarrow), per-instance XBRL reconstruction, Arelle runner, and deterministic judge.
- Output per-instance logs with (status, failure label, predicted JSON) for auditability.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | ACC (mean±std) | SER (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Published zero-shot | Fin-o1-14B | FinMR | 13.86% | 71% | [FINAUDITING](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/sections/FinMR%20performance.md) | Reported in paper (single run) |
| Regex message-only | N/A | FinMR | **TBD** | **TBD** | This work | Ignores XML; parses DQC message + sums values when possible |
| **Arelle symbolic (ours)** | Arelle | FinMR | **TBD** | **TBD** | This work | SER corresponds to non-executable items (no valid output) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours → no href-rewrite | Do not rewrite `xlink:href` to local files | Executability drops if queries contain local copies but reference remote paths |
| Ours → message-only calc | Use Arelle only to load facts; compute `calculated_value` from message text | If message contains enough info, this may approach regex baseline; otherwise it underperforms full Arelle execution |

### Experimental Rigor

**Variance & Reproducibility:**

- Arelle baseline and regex baseline are deterministic; no random seeds required.
- If optional self-consistency baseline is run, use `seeds=[42, 123, 456]` and report mean±std.

**Validity & Controls:**

- **No answer leakage**: The symbolic baseline must not read numeric gold values from the DQC message; it may only use identifiers (concept/period/unit) for targeting.
- **Connectivity policy**: Run Arelle in offline mode by default; if executability is low due to missing external taxonomies, report this explicitly as a benchmark packaging limitation.
- **Comparability**: The deterministic evaluator exactly follows Appendix C.3; report any deviations.

### Analysis (Optional)

- Per-DQC-ID executability and accuracy breakdown.
- Failure taxonomy distribution (top causes of non-executability).
- Correlation between prompt length and executability failures.

---

## Success Criteria

**Hypothesis** (directional — what you expect):

- Arelle can execute and reproduce correct `extracted_value`/`calculated_value` for a majority of FinMR instances, and achieves high accuracy on executable instances.
- Regex message-only baseline is substantially worse than Arelle, indicating that XBRL structure (not just textual leakage) is needed.

**Decision Rule** (concrete — when to stop):

- **Continue/Proceed**: Arelle achieves **≥90% ACC among executable instances** *and* outperforms the regex baseline by **≥15 ACC points** on the executable subset.
- **Pivot**: If Arelle executability is low (<20%) and failures are dominated by missing external taxonomy/schema references, rerun with a controlled cached taxonomy bundle (still deterministic; no search APIs) and re-measure coverage.
- **Refute**: If (i) Arelle executability remains low even after accounting for missing-taxonomy issues, or (ii) regex baseline matches Arelle within 5 points (suggesting the dataset leaks answers in text), then abandon the claim that FinMR primarily measures XBRL tool execution; instead treat the outcome as evidence that the benchmark needs redesign (e.g., hiding computed values and/or releasing a cleaned executable subset).

---

## Impact Statement

If successful, this work provides a decision-changing baseline and diagnostic for FinMR: benchmark users and model developers can report performance separately on an **executable subset** and can compare against a deterministic XBRL-engine baseline, clarifying whether improvements come from better reasoning or better tool integration. If it fails, the negative result is also decision-relevant: it would indicate that the released benchmark packaging is not executable (or leaks answers), motivating revised dataset releases and more careful evaluation protocols for structured-document reasoning benchmarks.

---

## References

- [FINAUDITING: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs](./references/FINAUDITING-A-Financial-Taxonomy-Structured-Multi-Document-Benchmark-for-Evaluating-LLMs/meta/meta_info.txt) - Wang et al., 2025
- [FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information](./references/FinTagging-An-LLM-ready-Benchmark-for-Extracting-and-Structuring-Financial-Information/meta/meta_info.txt) - Wang et al., 2025
- [FiNER: Financial Numeric Entity Recognition for XBRL Tagging](./references/FiNER-Financial-Numeric-Entity-Recognition-for-XBRL-Tagging/meta/meta_info.txt) - Loukas et al., 2022
- [Financial Numeric Extreme Labelling: A Dataset and Benchmarking](https://arxiv.org/abs/2306.03723) - Sharma et al., 2023
- [XBRL-Agent: Leveraging Large Language Models for Financial Report Analysis](https://dl.acm.org/doi/10.1145/3677052.3698614) - Han et al., 2024
- [FinQA: A Dataset of Numerical Reasoning over Financial Data](https://arxiv.org/abs/2109.00122) - Chen et al., 2021
- [ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance QA](https://arxiv.org/abs/2210.03849) - Chen et al., 2022
- [TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance](https://arxiv.org/abs/2105.07624) - Zhu et al., 2021
- [MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data](https://arxiv.org/abs/2206.01347) - Zhao et al., 2022
- [DocMath-Eval: Evaluating Math Reasoning Capabilities of LLMs in Understanding Long and Specialized Documents](https://arxiv.org/abs/2311.09805) - Yue et al., 2023
- [FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](https://arxiv.org/abs/2506.05828) - Tang et al., 2025
- [FinanceBench: A New Benchmark for Financial Question Answering](https://arxiv.org/abs/2311.11944) - Islam et al., 2023
- [PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance](https://arxiv.org/abs/2306.05443) - Xie et al., 2023
- [FinBen: A Holistic Financial Benchmark for Large Language Models](https://arxiv.org/abs/2402.12659) - Xie et al., 2024
- [FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for LLMs](https://arxiv.org/abs/2308.09975) - Zhang et al., 2023
- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564) - Wu et al., 2023
- [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031) - Yang et al., 2023
- [Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance](https://arxiv.org/abs/2502.08127) - Qian et al., 2025
- [Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning](https://arxiv.org/abs/2503.16252) - Liu et al., 2025
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - DeepSeek-AI, 2024
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) - Qwen Team, 2024
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) - Meta, 2024
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) - Gemma Team, 2025
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) - Wei et al., 2022
- [Self-Consistency Improves Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Program of Thoughts Prompting: Disentangling Computation from Reasoning](https://arxiv.org/abs/2211.12588) - Chen et al., 2022
- [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435) - Gao et al., 2022
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - Yao et al., 2022
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) - Schick et al., 2023
- [MRKL Systems: A Modular, Neuro-Symbolic Architecture](https://arxiv.org/abs/2205.00445) - Karpas et al., 2022
- [LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods](https://arxiv.org/abs/2412.05579) - Li et al., 2024
