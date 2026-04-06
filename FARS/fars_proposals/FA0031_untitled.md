# untitled

# Evidence-Grounded Constraint Schemas Reduce Contextual Neglect in LiveMedBench

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Automation constraint**: Fully automated evaluation (no new clinician annotation).
- **Domain constraint**: This work evaluates models on an offline benchmark; it is not a clinical decision support system and must not be used to provide medical advice.
- **Azure OpenAI constraint**: LiveMedBench’s official grader uses `gpt-4.1-2025-04-14` (served via Azure OpenAI). If Azure blocks some cases, we will (i) report the dropped-case rate and its keyword profile, and (ii) run comparisons on the remaining cases. (All methods are compared on the same retained set.)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used to draft or assist with medical responses (patient triage, differential diagnosis suggestions, and medication guidance). In these settings, a frequent source of harm is not exotic medical knowledge, but failure to apply obvious patient-specific constraints (e.g., pregnancy status, allergies, renal impairment, anticoagulation, age) to otherwise standard guideline advice.

**[LiveMedBench](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/meta/meta_info.txt)** is a continuously updated benchmark of 2,756 real-world bilingual medical cases, scored by case-specific weighted rubrics (16,702 criteria) using an automated rubric-based grader. Each case includes a small set of rubric items with positive or negative point weights; the grader (GPT-4.1 in the official implementation) checks each item as **met / not met** and computes a normalized per-case score.

Its error analysis reports that the dominant failure modes in strong models are **Contextual Neglect and Integration Failure (CNIF)** and **Guideline Overgeneralization and Rigidity (GOPR)**—i.e., models have relevant medical knowledge but fail to apply *patient-specific constraints* (CNIF), or apply generic guidelines too rigidly without adapting to the case (GOPR).

A deployment-relevant question in 2026 is therefore: *can we reduce CNIF/GOPR with a lightweight, training-free guardrail that is compatible with existing model APIs and open-source models?*

### The Problem

A natural baseline response strategy is “prompt harder”: ask the model to remember contraindications and to be careful. Another is “do a second pass”: generate a draft, then ask the model to self-review and revise (Self-Refine-style). However, these strategies still require the model to (implicitly) maintain an internal, consistent representation of patient constraints.

We focus on a sharper and decision-relevant sub-question: **does an explicit, structured constraint schema help beyond an information-matched unstructured checklist?** If structure does not help, practitioners should prefer a simpler “two-pass checklist review” without committing to schema design. If structure helps, it suggests that intermediate structured artifacts are an important ingredient for medical guardrails, and motivates investment in robust constraint extraction modules.

### Key Insight and Hypothesis

**Key insight**: CNIF/GOPR resemble a planning-under-constraints problem: the model must map a free-form narrative to a set of actionable constraints, then ensure every recommendation is consistent with them. A fixed schema can reduce omissions by forcing coverage of canonical constraint slots (“pregnancy”, “allergies”, “renal function”, “current meds”), and can reduce hallucinated constraints by requiring evidence quotes.

**One-sentence thesis**: *On constraint-sensitive LiveMedBench cases, a two-pass guardrail that extracts patient constraints into an evidence-grounded JSON schema and revises the draft against that schema improves constraint-focused rubric scores beyond an information-matched plain-text checklist revision.*

This could fail because (i) the benefit comes entirely from “second pass” self-review, not structure; (ii) constraint extraction is too noisy and introduces new errors; or (iii) the rubric grader is insensitive to constraint violations in practice.

---

## Proposed Approach

### Overview

We propose **evidence-grounded constraint schemas**: a training-free wrapper around a base LLM consisting of (1) constraint extraction from the case narrative, (2) drafting a response, and (3) revising the response by checking each actionable recommendation against the extracted constraints.

The contribution is not a new medical model, but a **controlled test** of whether *structured* intermediate artifacts matter beyond unstructured checklists, using LiveMedBench’s rubric evaluation.

### Method Details

**Inputs**: LiveMedBench case fields `narrative` and `core_request`.

**Shared output format (all conditions)**: a fixed short template with the same section headers and maximum tokens, to reduce “grader prefers formatting” confounds:
1) Brief assessment/summary, 2) Recommendations, 3) Safety / contraindications / when to seek care, 4) Clarifying questions.

**Constraint extraction (used in conditions B and C)**
- Extract a constraint set limited to: demographics (age/sex), pregnancy/breastfeeding status, allergies, comorbidities, current medications, renal/hepatic impairment, anticoagulation/bleeding risk, red-flag symptoms.
- For every extracted item, require an **evidence quote** that must be a verbatim substring of the narrative; otherwise the item is dropped.
- Allow explicit `unknown` when the narrative does not specify.

**Conditions (3 total; same base model)**
- **A. Single-pass strong prompt**: one call that produces the final response with the shared template and an explicit reminder to check contraindications/patient-specific factors.
- **B. Plain-text checklist guardrail** (3 calls): (1) extract constraints as a *plain-text checklist* with evidence quotes; (2) draft response using the **same prompt as A** (so A is exactly the “draft” stage); (3) revise by checking each recommendation against the checklist.
- **C. Structured schema guardrail (ours)** (3 calls): (1) extract constraints into a fixed **JSON schema** with evidence quotes and `unknown` fields; (2) draft response using the **same prompt as A**; (3) revise by checking each recommendation against the JSON fields.

**Important control**: B and C see the *same information* (constraint content + evidence quotes). The only difference is the representation format (plain text vs fixed JSON).

### Key Innovations

- **Information-matched ablation**: isolates “structure helps” from “second pass helps” by comparing C vs B.
- **Evidence-grounded extraction**: uses programmatic substring checks to reduce constraint hallucinations without human labeling.
- **Constraint-focused evaluation**: evaluates on automatically selected constraint-sensitive rubric items, aligned with CNIF/GOPR.

---

## Related Work

### Field Overview

**Medical LLM evaluation** has shifted from multiple-choice knowledge tests to open-ended, rubric-based benchmarks that better reflect clinical communication and safety requirements. HealthBench introduced large-scale physician-authored rubrics; LiveMedBench extends this with a live, contamination-resistant stream of cases and automated rubric generation.

**Improving medical LLM behavior** spans (i) training-time methods (rubric- or checklist-based RL/DPO, safety finetuning), and (ii) inference-time methods (structured prompting, self-refinement, tool/KB-backed checks). Training-time approaches can be powerful but are expensive and may be hard to deploy quickly; inference-time guardrails can be deployed immediately but must be evaluated carefully against strong prompting baselines.

Our work sits in inference-time guardrails and asks a specific mechanism question: whether a **fixed constraint schema** provides measurable benefit over an **unstructured checklist** when both are grounded to the input.

### Related Papers

- **[LiveMedBench](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/meta/meta_info.txt)**: Live, rubric-based medical benchmark; identifies CNIF/GOPR as dominant failures.
- **[HealthBench](https://arxiv.org/abs/2505.08775)**: Physician-authored rubrics for open-ended healthcare evaluation; validates LLM-based grading.
- **[Health-SCORE](https://arxiv.org/abs/2601.18706)**: Derives scalable healthcare rubrics and uses them for RL and inference guidance.
- **[MedSafetyBench](https://arxiv.org/abs/2403.03744)**: Medical safety benchmark based on AMA ethics; shows medical LLMs can be unsafe without targeted alignment.
- **[MedPerturb](https://arxiv.org/abs/2506.17163)**: Studies sensitivity of clinical reasoning to non-content perturbations.
- **[CSEDB](https://www.nature.com/articles/s41746-025-02277-8)**: Dual-track safety/effectiveness benchmark highlighting failures on high-risk scenarios.
- **[RxSafeBench](https://arxiv.org/abs/2511.04328)**: Medication safety benchmark for contraindications and drug interactions.
- **[MedXpertQA](https://arxiv.org/abs/2501.18362)**: Expert-level medical reasoning benchmark (text + multimodal).
- **[LLMEval-Med](https://aclanthology.org/2025.findings-emnlp.263/)**: Real-world medical benchmark with physician validation and LLM-as-judge scoring.
- **[MedArena](https://medarena.ai/)**: Clinician preference arena for real-world medical queries.
- **[MedPrompt](https://arxiv.org/abs/2311.16452)**: Prompting strategies can outperform specialized tuning on medical exams.
- **[Self-Refine](https://arxiv.org/abs/2303.17651)**: Iterative self-feedback improves outputs without training; general-purpose.
- **[Structured clinical reasoning prompting](https://www.medrxiv.org/content/10.1101/2024.09.01.24312894v1)**: Two-step structured summaries improve diagnostic accuracy.
- **[NeuroGlimpse](https://pmc.ncbi.nlm.nih.gov/articles/PMC12021381/)**: LLM-based contraindication extraction with source attribution (stroke thrombolysis).
- **[GAP: Graph-Assisted Prompts](https://arxiv.org/abs/2505.12888)**: Uses patient-centric graphs + KG checks to avoid contraindicated medication recommendations.
- **[Medication counseling with LLMs](https://arxiv.org/abs/2601.11544)**: Multi-agent system for medication counseling with contraindication interpretation tools.
- **[Rubrics as Rewards](../../papers/paper_summaries/Rubrics%20as%20Rewards%20Reinforcement%20Learning%20Beyond%20Verifiable%20Domains.md)**: Uses prompt-specific rubrics as RL rewards in non-verifiable domains (includes HealthBench).
- **[RLCF / Checklists are Better than Reward Models](../../papers/paper_summaries/Checklists%20Are%20Better%20Than%20Reward%20Models%20For%20Aligning%20Language%20Models.md)**: Checklist feedback for alignment (training-time).
- **[Chasing the Tail](../../papers/paper_summaries/Chasing%20the%20Tail%20Effective%20Rubric-based%20Reward%20Modeling%20for%20Large%20Language%20Model%20Post-Training.md)**: Improves rubric reward modeling to reduce over-optimization.
- **[InfiMed-ORBIT](../../papers/paper_summaries/InfiMed-ORBIT%20Aligning%20LLMs%20on%20Open-Ended%20Complex%20Tasks%20via%20Rubric-Based%20Incremental%20Training.md)**: Rubric-based GRPO improves medical dialogue performance on HealthBench-Hard.
- **[MedRAG](https://arxiv.org/abs/2502.04413)**: Retrieval-augmented medical reasoning; relevant to LiveMedBench’s open-book analysis.
- **[LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)**: Foundational work on automated evaluation via LLM judges.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Rubric-based medical evaluation | Use expert rubrics + LLM grader | HealthBench, LiveMedBench | rubric score / clinician agreement | Grader bias; rubric cost (HealthBench) |
| Training-time rubric/checklist alignment | Use rubrics/checklists as RL/DPO reward | Health-SCORE, RaR, RLCF, ORBIT | HealthBench(-Hard) | Expensive; not immediately deployable |
| Inference-time self-improvement | Draft → critique → revise | Self-Refine; structured prompting | varies | Gains may be from extra passes, not mechanism |
| Contraindication extraction / tools | Extract constraints; tool/KG checks | NeuroGlimpse; GAP; counseling MAS | task-specific | Narrow domains; not evaluated on LiveMedBench |

### Closest Prior Work

1) **Self-Refine** (Madaan et al., 2023) iteratively critiques and revises a draft. It motivates our two-pass baseline (B), but does not test whether a fixed intermediate schema adds beyond an unstructured critique.

2) **NeuroGlimpse** operationalizes LLM-based contraindication extraction with source attribution in a clinical workflow. It supports the feasibility of evidence-grounded constraint extraction, but it is not an end-to-end open-ended response quality benchmark and does not isolate the value of structured schemas.

3) **GAP (Graph-Assisted Prompts)** uses structured patient representations and knowledge graph checks to improve medication recommendation in dialogue. It is closest in spirit (explicit patient state), but differs in task (medication recommendation) and evaluation (not LiveMedBench rubrics), and it does not test “schema vs unstructured checklist.”

4) **Health-SCORE / rubric-based RL** use rubrics to train or guide models. They target training-time improvements, whereas our goal is a deployable inference-time wrapper and a mechanism test about structured constraints.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Self-Refine | Draft→self-critique→revise | Conflates “extra pass” with mechanism; no fixed constraint artifact | Add constraint extraction artifact; test schema vs checklist | Schema enforces coverage + reduces omission |
| NeuroGlimpse | Extracts contraindications with source attribution | Narrow task; not open-ended response quality | Use extraction only as guardrail input | Guardrail affects full response rubric score |
| GAP | Patient graph + KG verification for med recs | Different task/eval; no schema-vs-text test | Apply to LiveMedBench and isolate representation format | Fixed schema may be cheaper than KG tooling |
| Health-SCORE / ORBIT | Rubrics for RL/training and guidance | Training-time, higher cost | Training-free inference wrapper | Faster deployment; isolates constraint mechanism |
| LiveMedBench (benchmark) | Measures rubric score; finds CNIF/GOPR | Does not propose fixes | Provide and test a fix | Targets benchmark’s dominant failure modes |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3-14B-Instruct | 14B | https://huggingface.co/Qwen/Qwen3-14B-Instruct | Local inference on A100; primary base model (any comparable 14B instruct model is acceptable if naming differs) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| N/A | Inference-only | 0 | N/A | N/A |

**Other Resources (if applicable):**
- None (no external KB). Optional future extension: retrieval-augmented open-book setting, but not part of the minimal verification.

**Resource Estimate**:
- **Compute (local)**: Inference-only generation for 3 conditions on ~500 cases. Use temperature=0 for generation, and the same `max_tokens` across conditions (to control for verbosity). Expected to fit within **<50 A100 GPU-hours** (single GPU), dominated by total generated tokens.
- **API usage (grading)**: LiveMedBench grader uses **one `gpt-4.1` call per rubric item** (~6 per case on average). For 500 cases this is ~3k grader calls; wall-clock dominated by API latency and the script’s 0.2s per-call sleep.
- **Implementation note**: LiveMedBench’s official `run_model.py` assumes OpenAI-compatible chat APIs; for local HuggingFace models, implement a small runner that reproduces the same prompt format (the benchmark’s `create_prompt()` behavior) and writes `model_response` in the expected JSON output.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| LiveMedBench | Real-world medical cases with weighted rubric criteria graded by LLM | (1) overall rubric score; (2) constraint-focused rubric score | Constraint-salient subset from v202601 | https://huggingface.co/datasets/JuelieYann/LiveMedBench/ | https://github.com/ZhilingYan/LiveMedBench |

**Constraint-salient subset (pre-registered construction)**:
- Compute per-case constraint score = number of rubric criteria whose `criterion` matches a fixed keyword regex: `contraindicat|allerg|pregnan|breastfeed|renal|egfr|hepatic|anticoag|interaction|avoid|do not|not recommend` (case-insensitive).
- Select the top **N=500** cases by this score (ties broken by `case_id`).

**Primary metrics** (computed on the same selected case_ids for all conditions):
1) **Overall rubric score**: official LiveMedBench normalized score per case, averaged across the subset.
2) **Constraint-focused rubric score**: recompute the same normalized score but restricting to rubric items matching the keyword regex above.
3) **Negative-criteria rate** (auxiliary): mean number of satisfied negative-point criteria per case (lower is better).

### Main Results

(Results are TBD until verification runs; we include published context numbers separately.)

**Published context (full dataset, zero-shot, temp=0)** from LiveMedBench: Qwen3-14B overall score = **15.45%**.

**Planned evaluation (constraint-salient subset, same base model):**

| Method | Base Model | Benchmark | Overall rubric score (0–1) | Constraint-focused score (0–1) | Source | Notes |
|---|---|---|---:|---:|---|---|
| A. Single-pass strong prompt | Qwen3-14B | LiveMedBench (subset) | **TBD** | **TBD** | - | Needs re-run |
| B. Plain-text checklist (3-pass) | Qwen3-14B | LiveMedBench (subset) | **TBD** | **TBD** | - | Needs re-run |
| **C. Structured JSON schema (3-pass, ours)** | Qwen3-14B | LiveMedBench (subset) | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B vs C (information-matched) | Plain-text checklist vs fixed JSON schema | If schema helps, C > B on constraint-focused score |

### Analysis (Optional)

- **Azure-filter bias audit**: compare keyword-match rates in dropped vs retained cases; report whether filtering removes disproportionate constraint-salient cases.
- **Token budget audit**: report completion token counts per condition; optionally normalize score improvements by total generated tokens.

---

## Success Criteria

**Criterion 1: Structured schema adds value beyond unstructured checklist**
- Hypothesis: C improves over B on constraint-focused rubric score.
- Validation: C − B ≥ 0.03 absolute on constraint-focused score *or* reduces satisfied negative-point criteria by ≥0.15 per case on average.

**Criterion 2: Improvement is not only formatting/verbosity**
- Hypothesis: Gains persist under a fixed shared answer template and similar output length.
- Validation: C outperforms B while output lengths are comparable (reported alongside results).

**Decision rule (stop / pivot):** If C ≈ B (within 0.01) but both improve over A, conclude that “second-pass checklist review” is sufficient and refute the need for a structured schema in this setting. If neither B nor C improves over A, refute the premise that this class of guardrail materially reduces LiveMedBench constraint failures for this base model.

---

## Impact Statement

If structured constraint schemas outperform unstructured checklists, practitioners building medical LLM guardrails gain evidence that investing in explicit patient-constraint representations (and their grounding/verification) is worthwhile. If schemas do not help, the result supports using simpler two-pass checklist review without schema engineering, reducing deployment complexity.

---

## References

- LiveMedBench (Yan et al., 2026). [Paper](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/meta/meta_info.txt)
- LiveMedBench repository. [GitHub](./references/GitHub---ZhilingYan-LiveMedBench/meta/meta_info.txt)
- HealthBench (Arora et al., 2025). https://arxiv.org/abs/2505.08775
- Health-SCORE (Yang et al., 2026). https://arxiv.org/abs/2601.18706
- MedSafetyBench (Han et al., 2024). https://arxiv.org/abs/2403.03744
- MedPerturb (Gourabathina et al., 2025). https://arxiv.org/abs/2506.17163
- CSEDB (Wang et al., 2025). https://www.nature.com/articles/s41746-025-02277-8
- RxSafeBench (2025). https://arxiv.org/abs/2511.04328
- MedXpertQA (Zuo et al., 2025). https://arxiv.org/abs/2501.18362
- LLMEval-Med (Zhang et al., 2025). https://aclanthology.org/2025.findings-emnlp.263/
- MedArena (Zou Lab, 2025). https://medarena.ai/
- MedPrompt (Nori et al., 2023). https://arxiv.org/abs/2311.16452
- Self-Refine (Madaan et al., 2023). https://arxiv.org/abs/2303.17651
- Structured clinical reasoning prompting (Sonoda et al., 2024). https://www.medrxiv.org/content/10.1101/2024.09.01.24312894v1
- NeuroGlimpse (Cell Reports Medicine, 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12021381/
- GAP: Graph-Assisted Prompts (2025). https://arxiv.org/abs/2505.12888
- Medication counseling with LLMs (Sabel & Wingren, 2026). https://arxiv.org/abs/2601.11544
- Rubrics as Rewards (Gunjal et al., 2025). ../../papers/paper_summaries/Rubrics%20as%20Rewards%20Reinforcement%20Learning%20Beyond%20Verifiable%20Domains.md
- Checklists Are Better Than Reward Models (Viswanathan et al., 2025). ../../papers/paper_summaries/Checklists%20Are%20Better%20Than%20Reward%20Models%20For%20Aligning%20Language%20Models.md
- Chasing the Tail (Zhang et al., 2025). ../../papers/paper_summaries/Chasing%20the%20Tail%20Effective%20Rubric-based%20Reward%20Modeling%20for%20Large%20Language%20Model%20Post-Training.md
- InfiMed-ORBIT (Wang et al., 2025). ../../papers/paper_summaries/InfiMed-ORBIT%20Aligning%20LLMs%20on%20Open-Ended%20Complex%20Tasks%20via%20Rubric-Based%20Incremental%20Training.md
- MedRAG (Zhao et al., 2025). https://arxiv.org/abs/2502.04413
- LLM-as-a-Judge (Zheng et al., 2023). https://arxiv.org/abs/2306.05685
