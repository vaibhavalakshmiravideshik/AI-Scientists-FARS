# untitled

# Quote-Backed Citation Verification for Deep Research Reports

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used as **deep research agents**: they browse the web, read papers, and synthesize long-form reports with citations. In these settings, correctness is not only about producing a coherent narrative, but about providing **auditable provenance**: readers need to verify that specific claims are supported by the cited sources.

Recent work shows that tool use (search + browsing) reduces *obvious* citation hallucinations (fabricated paper titles/authors), but leaves a more dangerous failure mode: **subtle mis-citations**, where the cited paper exists but the claimed theorem/result is not actually stated there. The Aletheia math research agent highlights this directly: after tool training, citation errors shift from fake references to real references with incorrect claims (Figure 4 in [Towards Autonomous Mathematics Research](./references/Towards-Autonomous-Mathematics-Research/meta/meta_info.txt)).

ReportBench provides an automated evaluation framework for deep research reports. It constructs **100 survey-style prompts** from expert-written arXiv survey papers and evaluates generated reports by (i) reference overlap vs the survey’s bibliography and (ii) **cited-statement match rate** (the fraction of cited statements that are semantically consistent with their cited sources; higher is better). Even strong systems have substantial citation inconsistency: OpenAI Deep Research achieves **78.87%** match rate while Gemini Deep Research achieves **72.94%** on ReportBench’s protocol ([ReportBench Table 1](./references/ReportBench-Evaluating-Deep-Research-Agents-via-Academic-Survey-Tasks/sections/3.3 Product-Level Comparative Analysis.md)).

### The Problem

Current deep research agents typically cite sources as URLs or paper identifiers, but citations are rarely *mechanically checkable*. A citation can be “valid” (resolves to a real paper) while still being **semantically misaligned** with the claim. This undermines trust and makes downstream human verification expensive.

Existing citation benchmarks and methods mostly focus on question answering or retrieval-augmented generation (RAG), where citations are evaluated against a bounded retrieval context. For example, **ALCE** is a benchmark for long-form QA answers with citations that evaluates whether generated answers are supported by retrieved passages ([ALCE](./references/Enabling-Large-Language-Models-to-Generate-Text-with-Citations/meta/meta_info.txt)), and **CiteEval** proposes principle-driven citation evaluation metrics mainly for RAG pipelines ([CiteEval](./references/CiteEval-Principle-Driven-Citation-Evaluation-for-Source-Attribution/meta/meta_info.txt)). Deep research reports create an additional challenge: citations refer to **external documents** (often PDFs), the set of cited sources is larger, and claims are long-form and cross-sourced.

### Key Insight and Hypothesis

**Key insight:** In deep research reports, many citation failures are *procedural* (bad URL, quote fabrication, or citing a nearby-but-not-equal claim). We can make these failures cheap to detect and partially correct by requiring each cited statement to carry a **quote-backed evidence span** that is (i) verifiable as present in the cited document text, and (ii) checked for **entailment** (the quote supports the statement) to prevent “irrelevant but verbatim” gaming.

**Hypothesis:** On ReportBench prompts, enforcing quote-backed citations with automatic quote validity + entailment checks (and a single repair-or-drop step) will improve cited-statement match rate and reduce citation hallucinations (invalid/unresolvable links), while only modestly reducing coverage (reference recall and the number of cited statements).

This could fail if models respond by omitting citations/claims to satisfy quote constraints (coverage collapse), or if entailment checking is too noisy and rejects correct paraphrases.

---

## Proposed Approach

### Overview

We propose **QuoteVerify**, an **inference-time wrapper** around any base report generator that enforces quote-backed citations for each cited statement.

Pipeline (per prompt):

1. **Generate** a report where each factual sentence includes a citation marker (URL / arXiv ID / DOI) and an `evidence_quote` field (<= 300 characters) that is intended to be copied verbatim from the cited source.
2. **Fetch and cache** cited sources (prefer arXiv PDFs or other open-access links when possible).
3. **Mechanical quote validity**: verify that `evidence_quote` appears in the extracted source text after normalization.
4. **Semantic validity (mandatory)**: verify that the quote supports the statement using a natural language inference (NLI) classifier or a fixed LLM-judge prompt with `temperature=0`.
5. **Single repair-or-drop step**: for each report, batch all failing (statement, citation, quote) triples into one prompt to the base model and require one of: (a) new quote from same source, (b) new source + quote, or (c) delete the statement.

The output is a final report plus a sidecar JSONL of (statement, citation, quote, pass/fail flags) for auditing.

### Method Details

**Output format.** Each cited statement is represented as:

```json
{
  "statement": "...",
  "source_id": "https://... or arXiv:...",
  "evidence_quote": "...",
  "loc": "optional page/section hint"
}
```

**Quote validity check (fully automated).**

- Extract plaintext from PDFs/HTML.
- Apply robust normalization: casefold, collapse whitespace, and remove common PDF line-break hyphenation artifacts.
- Mark as quote-invalid if the normalized quote is not a substring of the normalized source text, or if the source cannot be fetched.

**Entailment check (mandatory; automated).**

To reduce false passes from irrelevant quotes, we apply a cheap entailment gate:

- Use an off-the-shelf **natural language inference (NLI)** model (e.g., DeBERTa/RoBERTa trained on MNLI) or a small fixed judge prompt.
- Input: `(premise = evidence_quote, hypothesis = statement)`.
- If the model does not predict **entailed/supported**, mark as entailment-fail.

**Repair prompt (one call per report).**

Given all failing items, the model must output a patched JSONL list where each item chooses exactly one action: `new_quote_same_source`, `new_source_and_quote`, or `delete_statement`.

### Key Innovations

- **Setting shift**: applies quote-backed enforcement to **deep research reports over external PDFs**, not just bounded-context QA/RAG.
- **Two-stage validity**: combines **mechanical quote validity** with a **mandatory entailment gate** to reduce superficial “substring pass” attacks.
- **Coverage-aware repair**: uses a minimal, single-step repair-or-drop loop and explicitly measures whether gains are real vs coverage collapse.

---

## Related Work

### Field Overview

Evidence-grounded generation has been studied extensively in QA/RAG. Systems such as WebGPT, GopherCite, RARR, and Self-RAG aim to attach sources or evidence spans to generated answers, and benchmarks such as ALCE evaluate citation quality for long-form answers with references. However, these settings typically assume a bounded retrieval context or short answer format.

A separate line of work develops evaluation protocols for citation faithfulness and claim-level factuality. CiteEval and ALiiCE study fine-grained citation evaluation and failure modes; AttributionBench benchmarks citation quality; and factuality benchmarks/metrics such as FActScore, QAGS, TRUE, and FEVER motivate claim-level verification. Many of these methods are evaluation-only, not enforcement mechanisms.

Deep research agents introduce a larger and noisier provenance surface: citations point to external PDFs and reports synthesize many sources. ReportBench, DEER, ADRA-Bank, and the Wiki Live Challenge provide benchmarks for long-form research/report generation. Aletheia highlights that even with tool use, citation failures can shift toward subtle mis-citations where the cited paper exists but does not support the claim.

### Related Papers

- **[Towards Autonomous Mathematics Research](./references/Towards-Autonomous-Mathematics-Research/meta/meta_info.txt)**: Documents a shift from fabricated citations to subtle mis-citations in a tool-using math research agent, motivating stronger provenance constraints.
- **[ReportBench](./references/ReportBench-Evaluating-Deep-Research-Agents-via-Academic-Survey-Tasks/meta/meta_info.txt)**: Benchmark for deep research reports that measures reference overlap and cited-statement match rate via automated verification.
- **[DEER](https://arxiv.org/abs/2512.17776)**: Studies expert report generation and evaluation, providing additional long-form report settings beyond QA.
- **[ADRA-Bank](https://arxiv.org/abs/2512.00986)**: Provides academic deep research tasks with evaluation signals, relevant for report-style citation grounding.
- **[Wiki Live Challenge](https://arxiv.org/abs/2602.01590)**: Benchmarks deep research and citation grounding in a live Wikipedia environment.
- **[WebGPT](https://arxiv.org/abs/2112.09332)**: Uses browser-assisted QA with citations, demonstrating web-grounded answer generation but not multi-paper report synthesis.
- **[GopherCite](https://arxiv.org/abs/2203.11147)**: Trains models to produce answers with quoted evidence spans and citations, motivating evidence spans as a provenance primitive.
- **[RARR](https://arxiv.org/abs/2210.08726)**: Retrofits attribution by retrieving sources and revising generations to add citations, but does not require verifiable quotes.
- **[Self-RAG](https://arxiv.org/abs/2310.11511)**: Integrates retrieval, generation, and critique signals for factuality/attribution, mainly evaluated in QA/RAG settings.
- **[ALCE](./references/Enabling-Large-Language-Models-to-Generate-Text-with-Citations/meta/meta_info.txt)**: Benchmark and metrics for long-form answers with citations, showing substantial headroom in citation support.
- **[CiteEval / CiteBench](./references/CiteEval-Principle-Driven-Citation-Evaluation-for-Source-Attribution/meta/meta_info.txt)**: Principle-driven citation evaluation with edit-based metrics; argues that simple NLI checks can be insufficient.
- **[AttributionBench](https://aclanthology.org/2024.findings-acl.886/)**: A benchmark for automatic attribution evaluation (whether cited evidence supports generated claims) that shows current models still struggle with fine-grained citation support.
- **[ALiiCE](https://arxiv.org/abs/2406.13375)**: Positional and fine-grained citation evaluation that categorizes citation errors relevant to report generation.
- **[Citation-Consistent Voting for Permutation-Robust RAG](https://arxiv.org/abs/2601.02993)**: Uses mechanically checkable evidence quote consistency as a selection signal in RAG, motivating quote-verification signals.
- **[Verbatim RAG](https://aclanthology.org/2025.bionlp-share.8/)**: Enforces verbatim evidence extraction with substring verification for evidence-based QA; this is the closest provenance-enforcement approach.
- **[FActScore](https://arxiv.org/abs/2305.14251)**: Atomic factuality evaluation for long-form generation, motivating sentence/claim-level evaluation.
- **[QAGS](https://aclanthology.org/2020.acl-main.450/)**: QA-based metric for factual consistency in summarization, relevant for claim-level checking.
- **[TRUE](https://aclanthology.org/2022.naacl-main.164/)**: Benchmark for factual consistency evaluation across tasks, motivating robust consistency signals.
- **[FEVER](https://aclanthology.org/N18-1074/)**: Large-scale fact verification dataset, foundational for evidence-based claim verification.
- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)**: Zero-resource hallucination detection, motivating automated self-auditing without labels.
- **[RAGTruth](https://arxiv.org/abs/2401.00396)**: Hallucination corpus for RAG that highlights common attribution/factuality errors.
- **[RAGAS](https://arxiv.org/abs/2309.15217)**: Reference-free evaluation signals for RAG, relevant when ground truth citations are unavailable.
- **[GhostCite / CiteVerifier](https://arxiv.org/abs/2602.06718)**: Analyzes citation validity at scale and proposes automatic verification tools.
- **[FACTUM](https://arxiv.org/abs/2601.05866)**: Targets citation hallucination detection and characterization.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Evidence-grounded QA/RAG | Generate answers with citations/evidence spans | WebGPT; GopherCite; RARR; Self-RAG; ALCE | ALCE; RAGTruth | Often bounded-context or short-answer; limited external-PDF handling |
| Citation evaluation / auditing | Measure whether citations support claims | CiteEval; ALiiCE; AttributionBench; GhostCite; FACTUM | CiteBench; attribution suites | Usually evaluation-only; can be brittle to paraphrase or require judges |
| Deep research / report benchmarks | Evaluate long-form reports with many citations | ReportBench; DEER; ADRA-Bank; Wiki Live Challenge; Aletheia | ReportBench match rate; report-level metrics | Few mechanisms proposed to reduce mismatches in generation |
| Verbatim provenance enforcement | Require verbatim evidence spans that can be checked | Verbatim RAG; citation-consistent voting | Task-specific QA/RAG benchmarks | Vulnerable to irrelevant-quote attacks; limited coverage controls |

### Closest Prior Work

**Verbatim RAG** is the closest mechanistic provenance-enforcement approach: it extracts verbatim evidence and uses substring verification to ensure the evidence appears in the source. Our work differs by targeting **multi-paper deep research reports** (ReportBench), where citations point to external PDFs with noisier extraction and where coverage collapse is a major concern. We also add a **mandatory entailment gate** to reduce irrelevant-quote attacks.

**GopherCite** and **WebGPT** demonstrate that generating quotes and sources can improve perceived provenance, but they do not enforce that quotes are present in the cited PDF text nor do they study long-form report settings with many citations.

**ReportBench** provides a strong evaluation protocol for cited-statement faithfulness, but it does not provide an inference-time mechanism to *reduce* citation mismatches. QuoteVerify is designed to be a lightweight wrapper that can be applied to any report generator and evaluated directly with ReportBench.

**CiteEval/ALiiCE** provide detailed taxonomies and metrics for citation evaluation. QuoteVerify uses a simple two-stage gate (substring + entailment) and focuses on whether such gates can reduce errors in the report generation loop under coverage constraints.

### Comparison Table

| Related work | What it does | Key limitation (for deep research reports) | What we change | Why ours should win |
|---|---|---|---|---|
| ReportBench | Evaluates report citations via semantic consistency | No mechanism to enforce evidence validity | Add quote+entailment validity + one repair step | Converts citation mismatch into fixable evidence failures |
| ALCE | QA citation benchmark + metrics | QA / bounded retrieval focus | Apply enforcement to report generation | Deep research reports have larger provenance surface |
| CiteEval | Fine-grained citation evaluation + edit-based metrics | Evaluation-only; RAG-centric | Enforce quote validity + entailment | Shifts from “measure” to “reduce” citation errors |
| Verbatim RAG | Verbatim quotes + substring verification | Evidence-based QA; limited coverage controls | Add entailment gate + coverage-aware repair; evaluate on ReportBench | Targets subtle mis-citations in long reports |
| GhostCite / CiteVerifier | Audits citation validity at scale and proposes verification tools | Not specialized to ReportBench-style report generation loops | Treat verification as an inference-time constraint with repair | Directly optimizes the cited-statement match objective |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| gpt-4o | - | API | Strong general LLM baseline |
| gemini-2.5-pro | - | API | Strong baseline used in ReportBench model-level comparisons |
| Qwen2.5-32B-Instruct (optional) | 32B | https://huggingface.co/Qwen | Reproducible open model for a lower-cost pilot |

**Training Data (if applicable):**
- No training; inference-time method only.

**Other Resources (if applicable):**

| Resource | Purpose | Link |
|---|---|---|
| ReportBench dataset + eval code | Prompts + automated citation evaluation | https://github.com/ByteDance-BandAI/ReportBench |

**Resource Estimate**:

- **Compute budget**: no training GPU-hours. If using an open model locally, inference is expected to fit within the 768 GPU-hour budget.
- **API usage** (pilot): for 20 prompts, Baseline and Prompt-only are ~20 calls each; QuoteVerify adds up to ~20 additional repair calls (one per prompt worst case).
- **I/O**: downloading and parsing PDFs for cited sources (cached across verification + evaluation).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ReportBench | 100 deep research prompts derived from arXiv survey papers; models write survey-style reports under a time cutoff | Reference precision/recall; #references; cited-statement match rate (↑); #cited statements; citation hallucination rate (↓) | test | https://github.com/ByteDance-BandAI/ReportBench | Official ReportBench eval |

We add two fully automated diagnostics:
- **Quote-validity rate (↑)**: fraction of cited statements whose provided evidence quote is found in the cited source text.
- **Entailment-pass rate (↑)**: fraction of cited statements whose quote entails/supports the statement.

### Main Results

We evaluate three conditions on the same prompt subset (pilot: 20 prompts; scale-up: full 100 prompts if feasible):

| Method | Base Model | Benchmark | Match rate (↑) | Ref recall (↑) | # cited stmts (↑) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| Baseline | same as others | ReportBench | **TBD** | **TBD** | **TBD** | - | Standard “write report with citations” prompt |
| Prompt-only quotes | same as others | ReportBench | **TBD** | **TBD** | **TBD** | - | Require an evidence quote per cited statement; no verification/repair |
| **QuoteVerify (ours)** | same as others | ReportBench | **TBD** | **TBD** | **TBD** | - | Quote validity + entailment gate + one repair-or-drop step |

#### Decision Rule (primary)

Let `M` be ReportBench cited-statement match rate, `R` be reference recall, and `C` be the number of cited statements.

- **Success** if QuoteVerify improves `M` over **both** Baseline and Prompt-only quotes by **≥ +5 absolute points**, while maintaining coverage:
  - `R` decreases by **≤ 20% relative** vs Baseline, and
  - `C` decreases by **≤ 20% relative** vs Baseline.
- We also require the improvement in `M` to be statistically positive under a paired bootstrap over prompts (95% CI of ΔM excludes 0).

**Threshold rationale:** ReportBench reports ~6 absolute points gap between two leading commercial systems (78.87 vs 72.94). A +5 gain would be practically meaningful, while the coverage bounds prevent trivial wins by deleting most cited content.

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| No entailment gate | Only substring quote validity | Higher quote-validity but more irrelevant-quote passes |
| No repair (drop-only) | Deterministic drop invalid pairs | Higher match rate but larger coverage loss |

### Analysis (Optional)

- Error breakdown: invalid URL / fetch fail / quote-not-found / entailment-fail / “semantic mismatch despite quote”.
- Which source types are most problematic: arXiv PDFs vs web pages vs paywalled publishers.

---

## Success Criteria

**Criterion 1: Reduced citation hallucinations**
- Hypothesis: QuoteVerify reduces unresolvable/fabricated citation links and increases mechanically valid provenance.
- Validation: Lower invalid-URL rate and higher quote-validity rate than both baselines.

**Criterion 2: Higher citation semantic consistency without collapsing coverage**
- Hypothesis: QuoteVerify increases cited-statement match rate without large drops in reference recall or cited-statement count.
- Validation: Meets the primary decision rule; dropping QuoteVerify’s repair step should reduce gains or increase coverage loss.

---

## Impact Statement

If successful, QuoteVerify would provide a lightweight, deployment-friendly mechanism for making deep research agents more trustworthy: citations become mechanically checkable and easier to audit. This could change practice for research assistants and domain-critical summarizers by shifting from “plausible citations” to “verifiable evidence-backed citations,” reducing the burden on human reviewers.

---

## References

- [Towards Autonomous Mathematics Research](./references/Towards-Autonomous-Mathematics-Research/meta/meta_info.txt) (2026)
- [ReportBench: Evaluating Deep Research Agents via Academic Survey Tasks](./references/ReportBench-Evaluating-Deep-Research-Agents-via-Academic-Survey-Tasks/meta/meta_info.txt) (2025)
- [CiteEval: Principle-Driven Citation Evaluation for Source Attribution](./references/CiteEval-Principle-Driven-Citation-Evaluation-for-Source-Attribution/meta/meta_info.txt) (2025)
- [Enabling Large Language Models to Generate Text with Citations (ALCE)](./references/Enabling-Large-Language-Models-to-Generate-Text-with-Citations/meta/meta_info.txt) (2023)
- [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) (2021)
- [GopherCite](https://arxiv.org/abs/2203.11147) (2022)
- [RARR: Retrofit Attribution using Research and Revision](https://arxiv.org/abs/2210.08726) (2022)
- [Self-RAG](https://arxiv.org/abs/2310.11511) (2023)
- [FActScore](https://arxiv.org/abs/2305.14251) (2023)
- [SelfCheckGPT](https://arxiv.org/abs/2303.08896) (2023)
- [RAGAS](https://arxiv.org/abs/2309.15217) (2023)
- [RAGTruth](https://arxiv.org/abs/2401.00396) (2024)
- [AttributionBench](https://aclanthology.org/) (2024)
- [ALiiCE](https://arxiv.org/abs/2406.13375) (2024)
- [Citation-Consistent Voting for Permutation-Robust RAG](https://arxiv.org/abs/2601.02993) (2026)
- [Verbatim RAG (KR Labs at ArchEHR-QA 2025)](https://aclanthology.org/2025.bionlp-share.8/) (2025)
- [GhostCite / CiteVerifier](https://arxiv.org/abs/2602.06718) (2026)
- [FACTUM](https://arxiv.org/abs/2601.05866) (2026)
- [Wiki Live Challenge](https://arxiv.org/abs/2602.01590) (2026)
- [ADRA-Bank](https://arxiv.org/abs/2512.00986) (2025)
- [DEER](https://arxiv.org/abs/2512.17776) (2025)
- [QAGS](https://aclanthology.org/2020.acl-main.450/) (2020)
- [TRUE](https://aclanthology.org/2022.naacl-main.164/) (2022)
- [FEVER](https://aclanthology.org/N18-1074/) (2018)
