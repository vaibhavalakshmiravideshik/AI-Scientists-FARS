# untitled

# Ask-First Slot-Oracle Multi-Turn Evaluation for LiveMedBench

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used to draft medical responses (symptom triage, medication guidance, and patient education). In real clinical communication, clinicians typically ask targeted clarifying questions when key information is missing (e.g., pregnancy status, drug allergies, current medications) before recommending an action (**ask-before-answer**).

However, most medical LLM benchmarks evaluate models in a **single-turn** format: the model sees a fixed prompt and must answer immediately. This makes it hard to distinguish two failure modes:
1) the model lacks medical knowledge or reasoning, versus
2) the model would have asked the right question if it were allowed to interact.

**LiveMedBench** is a new contamination-resistant benchmark built from real clinical discussions mined weekly from online medical communities and scored with **case-specific weighted rubrics** (a weighted checklist of per-case criteria; higher is better) ([LiveMedBench](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/meta/meta_info.txt)). It shows that even top models achieve low scores (e.g., GPT-5.2: 39.2% mean rubric score on the full benchmark) and that a critical remaining weakness is **Context-Seeking** behavior (“fail to proactively gather missing information before providing advice”) ([Overall Model Performance](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/sections/Overall Model Performance.md), [Themes](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/sections/A.3 Theme and Axis Definitions.md)).

### The Problem

LiveMedBench is designed as an offline benchmark: each case includes a patient narrative, a core request, and a list of rubric criteria derived from physician advice. The official evaluation treats the model response as a single-turn answer and grades it using an automated rubric-based grader.

This creates an evaluation gap: **LiveMedBench diagnoses context-seeking as a bottleneck, but cannot measure whether models can (i) identify what information is missing, (ii) ask for it efficiently, and (iii) incorporate it into a safer final recommendation.**

Existing interactive benchmarks cover parts of this space but do not fully resolve it:
- **HealthBench** includes a Context-Seeking theme, but uses synthetic multi-turn dialogues rather than real post-cutoff medical threads, and it is not “live” ([HealthBench](https://arxiv.org/abs/2505.08775)).
- **MediQ** evaluates question-asking in interactive clinical reasoning, but focuses on diagnostic consultation flows converted from medical QA benchmarks and does not use LiveMedBench’s per-case weighted rubrics ([MediQ](./references/MEDIQ-Question-Asking-LLMs-and-a-Benchmark-for-Reliable-Interactive-Clinical-Reasoning/meta/meta_info.txt)).
- **AskBench** provides a general multi-turn clarification evaluation loop with a judge and user simulator, but it is not specialized to LiveMedBench’s rubric structure and post-cutoff mining ([AskBench](./references/When-and-What-to-Ask-AskBench-and-Rubric-Guided-RLVR-for-LLM-Clarification/meta/meta_info.txt)).

We need a **minimal, fully automated multi-turn extension of LiveMedBench** that isolates “knowing when/what to ask” from “having more information.”

### Key Insight and Hypothesis

**Key insight.** LiveMedBench already contains fine-grained, weighted rubric criteria (positive and negative) tied to physician advice. If we create a controlled subset where exactly **one clinically critical patient slot** is hidden (e.g., pregnancy status or a drug allergy), then a one-question interaction can be made fully automated: an oracle can return the hidden slot value from the original narrative, and the final answer can still be scored by the existing rubric-based grader.

**Hypothesis.** On a LiveMedBench subset where exactly one rubric-relevant patient slot is removed from the narrative:
- Allowing the model to ask **one structured clarification question** and receive a deterministic oracle answer will measurably improve the final **rubric score** relative to a strong single-turn baseline that must answer under uncertainty.
- Many models will still fail to recover most of the “missing slot” performance gap because they either (i) do not ask a question, or (ii) ask for the wrong information.

Why we could be wrong:
- The rubric score may be insensitive to the masked slot (keyword mismatch), so the A→C gap is small even when the slot is clinically relevant.
- Models may ask generic questions that do not map cleanly to a slot type; our structured protocol may under-estimate real context-seeking ability.
- The automated grader may over-reward verbose hedging rather than correct constraint-sensitive recommendations.

---

## Proposed Approach

### Overview

We propose **LiveMedBench-Ask1**, a multi-turn evaluation harness that extends LiveMedBench with a single clarification turn supported by a deterministic **slot oracle**.

We build a derived benchmark split where each case has:
- an **unmasked** narrative (original LiveMedBench),
- a **masked** narrative where one patient slot is removed, and
- a stored oracle value for that slot.

We then evaluate the same model under three comparable conditions:
- **A. Masked single-turn baseline (strong prompt)**: model answers in one turn, explicitly stating uncertainty and listing the one most important clarifying question it would ask.
- **B. Ask1 slot-oracle (ours)**: model may ask **at most one** structured slot query; the oracle responds with the hidden value; the model then produces a final answer.
- **C. Unmasked single-turn (upper bound)**: model answers in one turn with the full narrative.

**Decision rule**: If condition B does not improve mean rubric score over A (bootstrap CI for B−A includes 0) or if slot-hit rate is near chance, we conclude that “ask-before-answer” capability is not meaningfully expressed under this constrained protocol on LiveMedBench, and we would not pursue more complex multi-turn extensions without changing the interaction format or selection procedure.

### Method Details

#### 1) Case selection (single-slot-critical subset)

We restrict evaluation to cases where we can (a) deterministically extract a slot value from the narrative and (b) identify that slot as rubric-relevant.

- **Slot inventory** (fixed): `{age, sex, pregnancy_status, allergies, current_medications, renal_function, hepatic_function, anticoagulation}`.
- **Slot extraction**: rule-based patterns (regular expressions + lexicons) applied to the original narrative (e.g., age from “\d+-year-old”, pregnancy keywords, allergy patterns like “allergic to X”, medication lists after “taking/started on”, anticoagulants like warfarin or a **direct oral anticoagulant (DOAC)** keyword). If a slot cannot be extracted confidently, the case is excluded.
- **Rubric relevance test**: a slot is considered rubric-relevant if at least one rubric criterion contains a slot-specific keyword (e.g., “pregnan*”, “breastfeed*”, “allerg*”, “warfarin/anticoag*”, “kidney/renal”, “liver/hepatic”, “age”, “elderly/child”).
- **Single-slot constraint**: keep cases where **exactly one** slot passes the rubric relevance test and has an extracted value. This prevents a degenerate policy of asking for multiple slots.

The resulting subset size is expected to be a few hundred cases; if smaller, we will report the size and proceed (the experiment remains decisive).

#### 2) Masking protocol

Given the identified slot type `s*` and extracted value `v*`, we construct a masked narrative by **removing** the shortest span that expresses `v*` (e.g., deleting “45-year-old”, deleting the sentence containing “allergic to penicillin”). We do not insert explicit placeholders indicating which slot was removed.

#### 3) Interaction protocol (one clarification turn)

We avoid any LLM-based “judge” for turn classification by requiring a strict first-turn format in condition B:

- The model must output either:
  - `CLARIFY_SLOT: <slot>` where `<slot>` is one of the fixed slot inventory, **or**
  - `FINAL_ANSWER:` followed by the answer.

If the model outputs `CLARIFY_SLOT: <slot>`, the oracle replies with:
- `SLOT_VALUE: <v*>` if `<slot> == s*`, else
- `SLOT_VALUE: unknown`.

The model then must output a final answer (no further questions).

#### 4) Scoring

We use the official LiveMedBench rubric-based scoring:
- Each case includes `rubric_items = [{criterion, points}]`.
- The rubric-based grader returns a boolean “met / not met” for each criterion; the per-case score is `clip(sum(points_i * met_i) / sum_{points_i>0} points_i, 0, 1)` (higher is better) ([Rubric-based Grader](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/sections/Rubric-based Grader..md)).

### Key Innovations

1. **A minimal multi-turn extension of LiveMedBench** that is fully automated and compatible with the existing rubric grader.
2. **Single-slot-critical construction** that makes “what to ask” non-trivial under a one-question budget.
3. **Deterministic oracle** (no patient simulator LLM) that cleanly separates question selection from answer generation.

---

## Related Work

### Field Overview

**Medical LLM evaluation** has moved from multiple-choice exams to open-ended benchmarks with rubric-based grading, because small factual errors and missing constraints can be clinically significant. HealthBench and LiveMedBench both use physician-derived rubrics to improve alignment with expert judgment.

**Interactive information seeking** is a separate capability from answering: models may need to ask clarifying questions when the initial prompt is underspecified. Recent benchmarks (e.g., MediQ and AskBench) show that prompting an LLM to “ask questions” can degrade performance, and that evaluation requires a user simulator or oracle.

Our work combines these threads: we extend a rubric-scored, real-case medical benchmark (LiveMedBench) with a minimal interactive harness to measure ask-before-answer behavior under tight budgets.

### Related Papers

- **[LiveMedBench](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/meta/meta_info.txt)**: Live, contamination-resistant medical benchmark built from real online cases and graded with per-case bipolar weighted rubrics.
- **[GitHub: LiveMedBench](./references/GitHub---ZhilingYan-LiveMedBench/meta/meta_info.txt)**: Official dataset schema and evaluation scripts (model run + rubric-based grading).
- **[HealthBench](https://arxiv.org/abs/2505.08775)**: Physician-authored multi-turn healthcare conversations with rubric-based scoring and a Context-Seeking theme.
- **[MediQ](./references/MEDIQ-Question-Asking-LLMs-and-a-Benchmark-for-Reliable-Interactive-Clinical-Reasoning/meta/meta_info.txt)**: Interactive clinical reasoning benchmark with a patient system and abstention strategies for deciding when to ask questions.
- **[AskBench](./references/When-and-What-to-Ask-AskBench-and-Rubric-Guided-RLVR-for-LLM-Clarification/meta/meta_info.txt)**: Multi-turn clarification benchmark with checkpoint-style rubrics and metrics for coverage and redundant questions.
- **[IN3 / Tell Me More!](https://arxiv.org/abs/2402.09205)**: Benchmark for detecting vague user instructions and asking targeted clarifying questions in tool-using agents.
- **[AskToAct](https://arxiv.org/abs/2503.01940)**: Tool-calling clarification dataset and self-correction training pipeline for resolving missing parameters.
- **[FATA: First Ask Then Answer](https://arxiv.org/abs/2508.08308)**: Prompting framework that generates a slate of supplementary questions before answering to improve completeness.
- **[ClariQ](https://arxiv.org/abs/2006.05986)**: Dataset for clarifying question generation in information-seeking dialogues.
- **[AmbigQA](https://arxiv.org/abs/2004.10645)**: QA benchmark for underspecified questions requiring clarification or multiple valid answers.
- **[AgentClinic](https://arxiv.org/abs/2405.07960)**: Interactive simulated clinical benchmark (patient + measurement agents) highlighting gaps between exam-style QA and dialogue-based care.
- **[CRAFT-MD](https://openreview.net/forum?id=Bk2nbTDtm8)**: Conversational evaluation framework for clinical LLMs showing diagnostic performance drops in interactive settings.
- **[MedDialogRubrics](https://arxiv.org/abs/2601.03023)**: Synthetic multi-turn medical dialogue benchmark with fine-grained rubrics for diagnostic conversations.
- **[Automated Rubrics for Reliable Evaluation of Medical Dialogue Systems](https://arxiv.org/abs/2601.15161)**: Retrieval-augmented pipeline to generate instance-specific rubrics and evaluate medical dialogues.
- **[MedAlign](https://arxiv.org/abs/2308.14089)**: Clinician-generated EHR instruction-following dataset highlighting long-context clinical workflows.
- **[CSEDB](https://arxiv.org/abs/2507.23486)**: Clinical Safety-Effectiveness Dual-Track Benchmark for open-ended medical Q&A with risk-weighted scoring.
- **[Health-SCORE](https://arxiv.org/abs/2601.18706)**: Scalable generalized rubric set for health tasks used for evaluation, RL rewards, and in-context guidance.
- **[Rubrics as Rewards](https://arxiv.org/abs/2507.17746)**: Uses per-instance rubrics as interpretable reward signals (GRPO) for non-verifiable domains including medicine.
- **[FactScore](https://arxiv.org/abs/2305.14251)**: Atomic fact decomposition evaluation for long-form generation (useful for auditing rubric sensitivity to factual gaps).
- **[LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)**: General methodology for using strong LLMs as graders, relevant to rubric-based evaluation.
- **[CheckList](https://arxiv.org/abs/2005.04118)**: Behavioral testing framework (minimum functionality and invariance tests) motivating structured capability tests like slot-masking.
- **[Self-Refine](https://arxiv.org/abs/2303.17651)**: Iterative draft→critique→revise prompting; a non-interactive alternative to ask-before-answer.
- **[MedSafetyBench](https://arxiv.org/abs/2403.03744)**: Benchmark of harmful medical requests used to evaluate and improve safety alignment.
- **[RxSafeBench](https://arxiv.org/abs/2511.04328)**: Medication contraindication and interaction safety benchmark in simulated consultations.
- **[MedPerturb](https://arxiv.org/abs/2506.17163)**: Evaluates sensitivity of medical decisions to non-content perturbations (format/style).
- **[ExpertLongBench](https://arxiv.org/abs/2506.01241)**: Expert-level long-form generation benchmark evaluated with structured checklists/rubrics.
- **[MedQA](https://arxiv.org/abs/2009.13081)**: Multiple-choice medical QA benchmark (contrast with open-ended rubric grading).
- **[MedMCQA](https://arxiv.org/abs/2203.14371)**: Large-scale multiple-choice medical exam benchmark.
- **[PubMedQA](https://arxiv.org/abs/1909.06146)**: Biomedical QA benchmark.
- **[MedPrompt](https://arxiv.org/abs/2311.16452)**: Prompting strategies for medical reasoning tasks.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Rubric-based medical evaluation | Grade open-ended responses with per-case criteria | HealthBench; LiveMedBench | rubric score / clinician agreement | Grader bias; rubric coverage |
| Interactive medical consultation | Ask questions before committing to diagnosis/advice | MediQ; AgentClinic; CRAFT-MD | diagnostic accuracy; dialogue quality | Patient simulator realism; scoring mismatch with physician advice rubrics |
| General clarification benchmarks | Convert underspecified prompts into interactive loops | AskBench; ClariQ; AmbigQA | accuracy + coverage + redundant questions | Not domain-specific to medicine |
| Post-hoc self-improvement | Revise drafts without external interaction | Self-Refine | task-specific | Does not obtain missing user info |

### Closest Prior Work

- **LiveMedBench** ([paper](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/meta/meta_info.txt)) is a live, contamination-resistant benchmark mined from real online medical discussions and scored with per-case bipolar weighted rubrics. It highlights **Context-Seeking** as a major failure mode, but its official protocol grades only a **single-turn** answer.
- **MediQ** ([paper](./references/MEDIQ-Question-Asking-LLMs-and-a-Benchmark-for-Reliable-Interactive-Clinical-Reasoning/meta/meta_info.txt)) evaluates interactive clinical reasoning with a patient system and studies abstention strategies for deciding when to ask follow-up questions. MediQ reports that even the best interactive expert system only closes **51.2%** of the performance gap between limited-information and full-information settings ([section](./references/MEDIQ-Question-Asking-LLMs-and-a-Benchmark-for-Reliable-Interactive-Clinical-Reasoning/sections/How much of the performance gap can be closed by asking questions.md)). However, MediQ’s cases and scoring target diagnostic accuracy in a simulator rather than rubric-scored physician advice on live post-cutoff threads.
- **AskBench** ([paper](./references/When-and-What-to-Ask-AskBench-and-Rubric-Guided-RLVR-for-LLM-Clarification/meta/meta_info.txt)) constructs multi-turn clarification tasks from standard QA by introducing explicit “checkpoints” and evaluates coverage vs redundant questioning, typically using an LLM judge and a user simulator. Our setting differs by using LiveMedBench’s real-case rubrics and a **deterministic oracle** extracted from the original narrative, and by restricting interaction to **one** structured slot query so that “what to ask” is directly measurable via slot-hit rate.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LiveMedBench | Live, contamination-resistant medical benchmark with rubric scoring | Single-turn only; cannot measure question asking | Add a 1-question interaction protocol + oracle on a single-slot-critical subset | Enables direct measurement of ask-before-answer value under LiveMedBench rubrics |
| MediQ | Interactive medical consultation with a patient simulator | Not post-cutoff threads; different grading target (diagnosis-focused) | Use LiveMedBench real cases + official rubric grader | Keeps LiveMedBench’s realism/scoring while adding a minimal, deterministic interaction |
| AskBench | General interactive clarification benchmark + judge/user simulator | Not tailored to medical rubrics; uses an LLM simulator | Deterministic oracle + slot-critical construction + LiveMedBench rubric grader | Cleaner causal attribution (no simulator noise) and direct connection to LiveMedBench failure analysis |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen/Qwen3-14B | 14B | https://huggingface.co/Qwen/Qwen3-14B | Open-weight model; LiveMedBench reports **15.4%** mean rubric score on the full 2,756-case benchmark (higher is better) under zero-shot evaluation (context only; not directly comparable to our derived subset) ([Overall Model Performance](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/sections/Overall Model Performance.md)) |
| gpt-4.1 (optional) | API | (available via platform) | Used by the official LiveMedBench grader; optional as a tested model for comparison |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| N/A | Inference-only evaluation | N/A | N/A | N/A |

**Other Resources (if applicable):**
- **LiveMedBench dataset**: `JuelieYann/LiveMedBench` on HuggingFace; use the rubric-augmented JSON file referenced in the repo README (e.g., `LiveMedBench_v202601.json`) ([repo README](./references/GitHub---ZhilingYan-LiveMedBench/sections/README.md.md)).
- **Official rubric grader prompt + script**: from the LiveMedBench repo (`evaluate/evaluate_model.py`), which uses `gpt-4.1` as an objective grader.

**Resource Estimate**:
- **Compute budget**: 0–50 GPU-hours (if using local open models) + API calls.
- **API usage**:
  - For a 300-case subset with ~6 rubric criteria per case, grading requires ~5,400 grader calls for 3 conditions (A/B/C).
  - Model inference: 300 cases × (A:1 + B:2 + C:1) ≈ 1,200 model calls per model.
  - If Azure content filtering drops some cases, all conditions are evaluated on the same retained set and the drop rate is reported.
- **GPU memory**: ≤80GB for 14B-class models.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| LiveMedBench-Ask1 (derived) | Subset of LiveMedBench where one rubric-relevant patient slot is masked; model may ask 1 clarification slot query | Mean rubric score (higher is better); gap recovery; slot-hit rate; redundant-query rate | test | https://huggingface.co/datasets/JuelieYann/LiveMedBench/ | LiveMedBench `evaluate_model.py` + custom harness |

**Primary metrics (computed on the same selected subset):**
- **Rubric score**: LiveMedBench per-case score averaged across cases (higher is better).
- **Gap recovery**: \((\text{Score}_B-\text{Score}_A)/(\text{Score}_C-\text{Score}_A)\). Interpreting values: **1.0** means asking recovers the full masked→unmasked gap (B matches C), and **0.0** means asking provides no benefit (B matches A). Values <0 mean asking hurts.
- **Slot-hit rate**: fraction of cases where the model queries the masked slot type on its first turn.
- **Redundant-query rate**: fraction of cases where the model queries a non-masked slot type (equals 1 - slot-hit when exactly one query is allowed).

**Uncertainty reporting**: We report 95% bootstrap confidence intervals over cases for (i) mean rubric score in each condition and (ii) gap recovery.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Mean rubric score | Gap recovery | Slot-hit rate | Source | Notes |
|--------|------------|-----------|------------------|--------------|--------------|--------|------|
| A. Masked single-turn (strong prompt) | Qwen3-14B | LiveMedBench-Ask1 | **TBD** | - | - | To be verified | One turn; must answer under missing info + state the most important clarifying question |
| B. Ask1 slot-oracle (ours) | Qwen3-14B | LiveMedBench-Ask1 | **TBD** | **TBD** | **TBD** | To be verified | Two turns max: structured slot query → oracle answer → final answer |
| C. Unmasked single-turn (upper bound) | Qwen3-14B | LiveMedBench-Ask1 | **TBD** | - | - | To be verified | One turn; full narrative |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| K=0 vs K=1 | Force no query in condition B | If asking matters, performance drops toward A |
| Non-slot question handling | If model outputs invalid `CLARIFY_SLOT`, treat as no-query | Robustness check for parsing errors |
| Slot-type subset analysis | Report results by slot type (pregnancy vs allergy vs meds, etc.) | Some slots should have larger A→C gaps and higher gains when asked |

### Analysis (Optional)

- **Model behavior decomposition**: separate failures into “did not ask” vs “asked wrong slot” vs “asked right but did not use.”
- **Rubric sensitivity audit**: measure distribution of A→C gaps to confirm that the derived subset indeed creates missing-information headroom.

---

## Success Criteria

**Criterion 1: Multi-turn asking recovers missing-information headroom**
- Hypothesis: a single clarification turn improves rubric scores when one critical slot is missing.
- Validation: On LiveMedBench-Ask1, condition B achieves higher mean rubric score than A (with a positive bootstrap CI for B−A) and recovers a meaningful fraction of the A→C gap (gap recovery is well above 0 and preferably ≥0.5). As a sanity check on subset construction, the masked→unmasked gap (C−A) should be non-trivial for a majority of cases (e.g., median(C−A) > 0).

**Criterion 2: Models must ask the right thing (not just “take another pass”)**
- Hypothesis: gains come from targeted question asking, not from generic second-pass hedging.
- Validation: Slot-hit rate is meaningfully above chance (1/|slots| = 12.5%) and ideally high (e.g., ≥0.6). Additionally, the per-case improvement (B−A) should be substantially larger on slot-hit cases than on non-hit cases.

---

## Impact Statement

If successful, this work provides a verification-ready, multi-turn extension of LiveMedBench that quantifies the value and limits of ask-before-answer behavior under realistic post-cutoff medical cases. It can guide both (i) evaluation (separating “missing info” from “reasoning failure”) and (ii) model development (motivating training or prompting methods that reliably ask targeted questions under strict interaction budgets).

---

## References

- [LiveMedBench: A Contamination-Free Medical Benchmark for LLMs with Automated Rubric Evaluation](./references/LiveMedBench-A-Contamination-Free-Medical-Benchmark-for-LLMs-with-Automated-Rubric-Evaluation/meta/meta_info.txt) — Yan et al., 2026
- [GitHub: LiveMedBench](./references/GitHub---ZhilingYan-LiveMedBench/meta/meta_info.txt) — Yan et al., 2026
- [When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification](./references/When-and-What-to-Ask-AskBench-and-Rubric-Guided-RLVR-for-LLM-Clarification/meta/meta_info.txt) — Zhao et al., 2026
- [MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning](./references/MEDIQ-Question-Asking-LLMs-and-a-Benchmark-for-Reliable-Interactive-Clinical-Reasoning/meta/meta_info.txt) — Li et al., 2024
- [HealthBench: Evaluating Large Language Models Towards Improved Human Health](https://arxiv.org/abs/2505.08775) — Arora et al., 2025
- [ClariQ: Clarifying Questions for Open-Domain Information-Seeking Conversations](https://arxiv.org/abs/2006.05986) — Aliannejadi et al., 2020
- [AmbigQA: Answering Ambiguous Open-Domain Questions](https://arxiv.org/abs/2004.10645) — Min et al., 2020
- [FactScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251) — Min et al., 2023
- [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023
- [MedQA](https://arxiv.org/abs/2009.13081) — Jin et al., 2021
- [MedMCQA](https://arxiv.org/abs/2203.14371) — Pal et al., 2022
- [PubMedQA](https://arxiv.org/abs/1909.06146) — Jin et al., 2019
- [MedPrompt](https://arxiv.org/abs/2311.16452) — Nori et al., 2023
- [Self-Refine](https://arxiv.org/abs/2303.17651) — Madaan et al., 2023
- [MedSafetyBench](https://arxiv.org/abs/2403.03744) — Zhang et al., 2024
- [RxSafeBench](https://arxiv.org/abs/2511.04328) — (authors), 2025
- [MedPerturb](https://arxiv.org/abs/2506.17163) — (authors), 2025
- [Health-SCORE: Towards Scalable Rubrics for Improving Health-LLMs](https://arxiv.org/abs/2601.18706) — (authors), 2026
- [IN3 / Tell Me More! Towards Implicit User Intention Understanding of Language Model Driven Agents](https://arxiv.org/abs/2402.09205) — Qian et al., 2024
- [AskToAct: Enhancing LLMs Tool Use via Self-Correcting Clarification](https://arxiv.org/abs/2503.01940) — (authors), 2025
- [FATA: First Ask Then Answer](https://arxiv.org/abs/2508.08308) — Fu and Du, 2025
- [AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments](https://arxiv.org/abs/2405.07960) — (authors), 2024
- [CRAFT-MD: A Conversational Evaluation Framework for Testing in Medicine](https://openreview.net/forum?id=Bk2nbTDtm8) — (authors), 2024
- [MedDialogRubrics: A Comprehensive Benchmark and Evaluation Framework for Medical Dialogue Systems](https://arxiv.org/abs/2601.03023) — (authors), 2026
- [Automated Rubrics for Reliable Evaluation of Medical Dialogue Systems](https://arxiv.org/abs/2601.15161) — (authors), 2026
- [MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Health Record Data](https://arxiv.org/abs/2308.14089) — (authors), 2023
- [CSEDB: A Novel Evaluation Benchmark for Medical LLMs Illuminating Safety and Effectiveness in Clinical Domains](https://arxiv.org/abs/2507.23486) — (authors), 2025
- [CheckList: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118) — Ribeiro et al., 2020
- [ExpertLongBench: Benchmarking Language Models on Expert-Level Long-Form Generation Tasks with Structured Checklists](https://arxiv.org/abs/2506.01241) — (authors), 2025
- [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746) — Gunjal et al., 2025
