# untitled

# Do Glossaries Actually Bind? Discriminative Definition Unit Tests for Convention Adherence in LLM Reasoning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used in settings where the input includes *definitions* that must be treated as binding. Examples include API specifications for tool-calling agents, task rubrics for evaluators, and domain glossaries for technical problem solving. A common engineering practice is to prepend the relevant glossary/specification to the prompt and assume the model will follow it.

Recent evidence suggests that this assumption is fragile. DeepMind’s Aletheia math research agent reports that, on a 700-problem deployment to ErdősProblems.com, a substantial fraction of candidate solutions were “technically correct under some interpretation” but “mathematically vacuous” for the intended interpretation (**[Towards Autonomous Mathematics Research](./references/Towards-Autonomous-Mathematics-Research/meta/meta_info.txt)**). The companion Erdős case study identifies a concrete root cause: many failures were driven by *notational/definitional convention ambiguity* (e.g., additive vs. Dirichlet convolution; strong vs. weak completeness) because the agent was not informed of the website’s definitional conventions (**[Semi-Autonomous Mathematics Discovery with Gemini](./references/Semi-Autonomous-Mathematics-Discovery-with-Gemini/meta/meta_info.txt)**, Sec. 1.1).

In parallel, work on “definition receptivity” shows that LLMs inconsistently incorporate externally provided definitions and may default to parametric knowledge when definitions conflict with internal representations (**[Do LLMs Adhere to Label Definitions?](./references/Do-LLMs-Adhere-to-Label-Definitions/meta/meta_info.txt)**). These observations suggest that “prepend a glossary” is not a reliable mechanism for binding interpretation.

### The Problem

We focus on a narrow but decision-relevant question for 2026 agent builders:

> When a prompt includes an explicit glossary, does the model actually *use* those definitions, or does it silently revert to an internal convention when solving the downstream problem?

This matters because convention errors are difficult to catch after the fact: a long solution can be internally consistent yet solve the “wrong” problem. Existing approaches do not cleanly isolate this failure mode with automated evaluation:

- **Prompt testing frameworks** like PromptPex extract output rules from prompts and generate tests, but primarily target format/rule compliance and use LLM-as-judge, which is itself unreliable for subtle semantic interpretation (**[PromptPex](./references/PromptPex-Automatic-Test-Generation-for-Language-Model-Prompts/meta/meta_info.txt)**).
- **Prompt specification languages** for multi-turn protocols (e.g., FSM-like specifications) verify procedural conformance, but do not address semantic binding of domain definitions (**[FASTRIC](./references/FASTRIC-Prompt-Specification-Language/meta/meta_info.txt)**).
- **Auditing tools** identify instruction-following violations in conversational settings but rely on subjective human review and do not provide deterministic, domain-grounded checks (**[Offscript](./references/Offscript-Automated-Auditing/meta/meta_info.txt)**).

We need a simple, fully automated probe that (i) induces a realistic “definition conflict” with the model’s internal convention, (ii) has deterministic ground-truth labels, and (iii) can test whether *actively checking* definitions changes downstream behavior beyond “extra tokens / extra thinking.”

### Key Insight and Hypothesis

**Key insight:** Definitions can be treated as *semantic contracts*. In software engineering, contracts are enforced not only by documentation but also by test cases designed to fail under common misinterpretations. Analogously, if a glossary defines an ambiguous term, we can construct **discriminative definition checks** whose answers flip under a common alternate convention.

**Hypothesis:** Adding a small number of discriminative definition checks before solving improves downstream accuracy more than an engagement-matched control (same prompt length and similar computation, but checks that do not distinguish between conventions), because discriminative checks force the model to commit to the glossary-defined semantics rather than defaulting to internal conventions.

This could be wrong if (i) models already follow the glossary whenever it is present, so there is no headroom; (ii) answering checks does not causally influence subsequent reasoning; or (iii) any gain is explained entirely by generic extra computation rather than definition-specific binding.

---

## Proposed Approach

### Overview

We propose **Definition Unit Tests (DUT)**: a training-free prompt wrapper that precedes a target question with a small set of **auto-gradable, discriminative definition checks** derived from the same glossary.

To evaluate DUT in a controlled setting, we will build **ErdosConventionsBench**, a synthetic benchmark constructed from the definitional conventions on ErdosProblems.com (**[ErdosProblems Definitions](./references/ErdosProblems-Definitions/meta/meta_info.txt)**). Each item includes:
1) a short glossary snippet, and
2) a main question whose answer differs under a common alternate convention.

We will compare DUT against an engagement-matched baseline to isolate whether *discriminative* checks matter beyond “forcing the model to think more.”

### Method Details

#### Benchmark construction: ErdosConventionsBench

We will generate ~300 items across three convention families explicitly mentioned in the Erdős case study and/or the ErdosProblems definitions page:

1. **Additive vs. Dirichlet convolution**
   - Glossary defines additive convolution: \(f * g(n) = \sum_{a+b=n} f(a) g(b)\).
   - Alternate convention: Dirichlet convolution \(\sum_{ab=n} f(a) g(b)\).
   - Item format: provide sparse tables for \(f\) and \(g\) on \(\{1,\dots,N\}\), ask for \((f*g)(n)\).

2. **Asymptotic quantifiers (“for sufficiently large”) in \(O(\cdot)\) and \(o(\cdot)\)**
   - Glossary defines \(f=O(g)\) as holding for all sufficiently large \(x\).
   - Alternate misinterpretation: the inequality must hold for all \(x\) (including small \(x\)).
   - Item format: give piecewise-defined \(f(x)\) and \(g(x)\) with a small “exception” at small \(x\); ask whether \(f=O(g)\) or \(f=o(g)\) holds.

3. **Complete vs. “all integers” misinterpretation**
   - Glossary defines “complete” as: \(P(A)\) contains all sufficiently large integers.
   - Alternate misinterpretation: \(P(A)\) contains all positive integers.
   - Item format: define \(A\) by a simple rule (e.g., \(A=\{n\in\mathbb{N}: n\ge k\}\)) so that completeness holds but “all integers” fails; ask whether \(A\) is complete/strongly complete.

For each item, we can deterministically compute the correct answer under the glossary-defined convention and the alternate answer under the common misinterpretation.

#### Three prompt conditions (core experiment)

All conditions use the same output format tags so parsing is automated.

- **A. Glossary-only (prompting baseline):** Provide glossary snippet + main question → `FINAL_ANSWER`.
- **B. Engagement-matched control:** Provide glossary snippet + **k neutral checks** that require similar computation but are constructed so their answers are identical under the glossary convention and the alternate convention; then the main question → `FINAL_ANSWER`.
- **C. Definition Unit Tests (ours):** Provide glossary snippet + **k discriminative checks** whose answers differ under the alternate convention; then the main question → `FINAL_ANSWER`.

We set \(k=3\) by default. Neutral and discriminative checks are pre-generated per item so total prompt length and “work” are comparable between B and C.

**Neutral-check construction (examples):** Neutral checks are designed to require similar arithmetic/reading effort but have the **same answer under both conventions**. For example, in the convolution family, a neutral check can ask for the *number of terms* in the additive-convolution sum \(\sum_{a+b=n}\) (which depends only on \(n\)), or ask the model to list the index pairs \((a,b)\) with \(a+b=n\); these do not distinguish additive vs Dirichlet convolution because they do not depend on interpreting \(*\) as \(\sum_{ab=n}\). In the completeness family, a neutral check can ask the model to restate the definition of \(P(A)\) and compute \(P(A)\cap\{1,\dots,m\}\) for a tiny \(A\); this is consistent under either interpretation of “complete.”

The key comparison is **C vs B**.

#### Automatic scoring

- **Main accuracy:** exact match (numeric / boolean) against deterministic ground truth.
- **Check accuracy (B and C):** exact match per check (numeric / boolean).
- **Coupling diagnostic:** \(\Pr[\text{main correct} \mid \text{all checks correct}]\) vs \(\Pr[\text{main correct} \mid \text{any check wrong}]\) to test whether passing checks predicts correct downstream interpretation.

### Key Innovations

- **A deterministic “definition conflict” benchmark** derived from a real domain glossary (ErdosProblems conventions) that isolates semantic misinterpretation without human grading.
- **Engagement-matched controls**: distinguishes “discriminative definition checking” from generic “extra tokens / extra thinking.”
- **A minimal, training-free wrapper** (DUT) that can be implemented in agent pipelines wherever definitions admit auto-gradable checks.

---

## Related Work

### Field Overview

A recurring challenge in LLM systems is ensuring that model behavior is consistent with explicit specifications, including definitions and conventions. Work on instruction following and specification engineering emphasizes that prompt formality and structure can influence conformance, but conformance is often measured procedurally (multi-turn protocol adherence) rather than semantically (binding to a definition). Separately, prompt testing and auditing systems aim to discover failures through generated test cases, but commonly rely on LLM-as-judge or human validation, limiting determinism. In software, contracts are enforced using concrete counterexamples and unit tests; recent work in code generation similarly shows that concrete contract-violating tests are more effective than abstract descriptions. Our proposal brings these threads together by treating glossary-defined conventions as semantic contracts and testing them using deterministic, convention-flip questions.

### Related Papers

- **[Towards Autonomous Mathematics Research](./references/Towards-Autonomous-Mathematics-Research/meta/meta_info.txt)**: Introduces Aletheia and reports many “technically correct under some interpretation” but vacuous solutions due to misinterpretation on ErdősProblems.
- **[Semi-Autonomous Mathematics Discovery with Gemini](./references/Semi-Autonomous-Mathematics-Discovery-with-Gemini/meta/meta_info.txt)**: Documents the ErdősProblems deployment and explicitly attributes many vacuous solutions to definitional convention ambiguity.
- **[ErdosProblems Definitions](./references/ErdosProblems-Definitions/meta/meta_info.txt)**: Provides the concrete glossary of conventions used to generate ErdosConventionsBench.
- **[PromptPex](./references/PromptPex-Automatic-Test-Generation-for-Language-Model-Prompts/meta/meta_info.txt)**: Generates unit tests for prompts from extracted specifications, but focuses on output-rule compliance and uses LLM judging.
- **[Do LLMs Adhere to Label Definitions?](./references/Do-LLMs-Adhere-to-Label-Definitions/meta/meta_info.txt)**: Shows LLMs inconsistently incorporate externally provided definitions, especially under “knowledge conflict” conditions.
- **[ContractEval](./references/ContractEval-PACT/meta/meta_info.txt)**: Evaluates whether code-generation models respect input-validity contracts and shows concrete contract-violating test cases in prompts improve contract adherence.
- **[FASTRIC](./references/FASTRIC-Prompt-Specification-Language/meta/meta_info.txt)**: Proposes a natural-language prompt specification language for verifying procedural conformance in multi-turn interactions.
- **[Offscript](./references/Offscript-Automated-Auditing/meta/meta_info.txt)**: Uses an auditor agent to surface instruction-following failures in custom instructions, but requires subjective human validation.
- **[SpecEval](https://arxiv.org/abs/2509.02464)**: Audits model adherence to provider-specified behavioral specifications using generated test cases and LLM judging.
- **[Petri](https://github.com/safety-research/petri)**: Agentic auditing framework for exploring risky interactions in AI safety research.
- **[Chain-of-Dictionary Prompting](./references/Chain-of-Dictionary-Prompting/meta/meta_info.txt)**: Demonstrates that dictionary-style prompting can elicit desired behavior (translation) without fine-tuning, motivating glossary-based interventions.
- **[Exploring the Hidden Reasoning Process of LLMs by Misleading Them](https://arxiv.org/abs/2503.16401)**: Fine-tunes models on counterfactual arithmetic rules to test whether they apply externally provided rules vs memorized conventions.
- **[Reasoning or Reciting?](https://arxiv.org/abs/2307.02477)**: Evaluates arithmetic under counterfactual number bases, showing performance can collapse when defaults change.
- **[GSM-Symbolic](https://arxiv.org/abs/2410.05229)**: Uses templated GSM8K-style problems to test robustness of math reasoning under controlled perturbations.
- **[QuestBench](https://arxiv.org/abs/2503.22674)**: Benchmarks underspecified reasoning tasks requiring explicit information gathering.
- **[AutoMonitor-Bench](https://arxiv.org/abs/2601.05752)**: Categorizes and evaluates specification-gaming-like failures in agentic systems.
- **[Learning to Ask: When LLMs Meet Unclear Instruction](https://arxiv.org/abs/2409.00557)**: Studies clarification behavior under ambiguous instructions.
- **[Answering Questions in Stages: Prompt Chaining for Contract QA](https://arxiv.org/abs/2410.12840)**: Shows staged prompting can improve mapping from contractual language to answers.
- **[LLM-SQL-Solver](https://arxiv.org/abs/2312.10321)**: Uses counterexample-style prompting to enforce a precise semantic notion (SQL equivalence) via concrete instances.
- **[System Prompts as a Mechanism of Bias in LLMs](https://arxiv.org/abs/2505.21091)**: Shows that instruction placement (system vs user) can materially affect model behavior.

- **[Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128)**: Demonstrates iterative execution-feedback loops for correcting errors, related to “check-then-solve” patterns.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Uses episodic feedback for agents to improve future behavior, motivating structured intermediate feedback.
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)**: Majority voting over multiple reasoning samples; included as an inference-time scaling baseline that may reduce convention errors.
- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)**: Introduces CoT prompting; relevant because intermediate steps may influence whether definitions are applied.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: Tool-using reasoning framework; relevant to agent settings where specifications/glossaries bind tool semantics.
- **[ToolBench](https://arxiv.org/abs/2307.16789)**: Benchmark for tool-use agents; motivates extending definition-binding tests to tool specifications.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Passive definition injection | Prepend glossary/dictionary hints to prompt | [Chain-of-Dictionary](./references/Chain-of-Dictionary-Prompting/meta/meta_info.txt) | FLORES-200 | No guarantee definitions bind; may be domain-specific |
| Specification languages / protocol conformance | Encode interaction structure; verify conformance | [FASTRIC](./references/FASTRIC-Prompt-Specification-Language/meta/meta_info.txt) | custom FSM traces | Focuses on procedural adherence, not semantic conventions |
| Prompt auditing / test generation | Generate tests to find prompt/model failures | [PromptPex](./references/PromptPex-Automatic-Test-Generation-for-Language-Model-Prompts/meta/meta_info.txt), [Offscript](./references/Offscript-Automated-Auditing/meta/meta_info.txt) | prompt suites, custom instructions | Often relies on LLM judges or human validation |
| Contracts via concrete counterexamples | Provide violating examples to enforce constraints | [ContractEval](./references/ContractEval-PACT/meta/meta_info.txt) | HumanEval+/MBPP+ contract tests | Mainly code; not focused on “definition binding” |
| Definition receptivity / knowledge conflict | Test whether models follow external definitions under conflict | [Do LLMs Adhere to Label Definitions?](./references/Do-LLMs-Adhere-to-Label-Definitions/meta/meta_info.txt) | e-SNLI, HateXplain, etc. | Classification-focused; no “unit test” style prompting |
| **Ours: definition unit tests** | Discriminative, deterministic checks before solving | this proposal | ErdosConventionsBench | Requires conventions with auto-gradable checks |

### Closest Prior Work

- **PromptPex** generates unit tests for prompts from extracted specifications, but targets prompt compliance and uses LLM-as-judge; we instead target semantic convention binding and use deterministic grading.
- **Do LLMs Adhere to Label Definitions?** studies whether models follow external definitions under permutation/perturbation, but does not test a *preflight checking* intervention on a deterministic convention-flip benchmark.
- **ContractEval** shows that concrete contract-violating test cases can improve contract adherence in code generation; we apply the same “concrete checks beat abstract descriptions” principle to semantic conventions in math.

**Novelty Kill Search Summary:** Searched for prior work combining “glossary + discriminative unit tests” for definition binding (queries included “glossary prompting definition adherence benchmark”, “prompt unit tests specification”, “definition receptivity LLM”, “contract prompting + unit tests”, and related OpenReview/arXiv searches). Found adjacent work on prompt testing (PromptPex) and definition receptivity (Mohammadi et al.) but no prior work constructing a deterministic convention-flip benchmark from a domain glossary and comparing discriminative checks against engagement-matched controls as of 2026-02-25. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [PromptPex](./references/PromptPex-Automatic-Test-Generation-for-Language-Model-Prompts/meta/meta_info.txt) | Generates unit tests for prompts from extracted specs | Relies on LLM judging; focuses on rule/format compliance | Deterministic, convention-flip checks for semantic binding | Removes judge noise; targets a documented failure mode in math agents |
| [Do LLMs Adhere to Label Definitions?](./references/Do-LLMs-Adhere-to-Label-Definitions/meta/meta_info.txt) | Measures whether models follow external label definitions | Classification setting; not an intervention on downstream tasks | Test a simple preflight intervention (DUT) on deterministic tasks | Directly evaluates whether checks causally improve downstream answers |
| [ContractEval](./references/ContractEval-PACT/meta/meta_info.txt) | Uses contract-violating tests to evaluate/enforce contracts in code | Domain is code; not about semantic conventions | Apply “concrete counterexamples” idea to glossary conventions | Extends contract notion to semantic definitions in reasoning |
| [FASTRIC](./references/FASTRIC-Prompt-Specification-Language/meta/meta_info.txt) | Prompt specification language for procedural conformance | Does not test semantic definition binding | Focus on semantic conventions with deterministic grading | Provides a complementary evaluation axis (semantic binding) |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):**
- Prompting baseline: A (glossary-only).
- Engagement-matched prompting baseline: B (neutral checks + solve).
- Proposed method: C (discriminative checks + solve).
- Optional inference-time scaling baseline: run majority-vote over 5 samples for A and B to test whether simple sampling closes the gap.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-Math-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct | Math-specialized open model |
| Llama-3.1-8B-Instruct | 8B | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | General instruct model |

**Training Data (if applicable):**

No training data needed – inference only.

**Other Resources (if applicable):**
- ErdosProblems definitions (glossary): **[ErdosProblems Definitions](./references/ErdosProblems-Definitions/meta/meta_info.txt)**.

**Resource Estimate**:
- **Compute budget**: ≤50 A100 GPU-hours (≤2 models, ~300 items, 3 conditions, optional 3 seeds; short outputs)
- **GPU memory**: 1×A100 80GB sufficient for 7–8B inference (vLLM)
- **API usage**: none

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ErdosConventionsBench (new) | Synthetic, deterministic convention-flip QA derived from ErdosProblems glossary | Main accuracy; check accuracy; conditional main accuracy | test | Generated from **[ErdosProblems Definitions](./references/ErdosProblems-Definitions/meta/meta_info.txt)** | Custom script (simple arithmetic + exact match) |

**Evaluation Scripts:**
- Implement a generator that emits JSONL items with (glossary, neutral_checks, discriminative_checks, main_question, ground_truth).
- Implement a runner that formats prompts for A/B/C, parses `CHECK_i:` and `FINAL_ANSWER:` fields, and computes exact-match metrics.

### Main Results

#### Results Table

(All results to be filled by the Verification module. Primary comparison is C vs B on **main accuracy** (fraction of items where the final answer matches the deterministic ground truth). If stochastic decoding is used, report mean±std over ≥3 seeds.)

| Method | Base Model | Benchmark | Main accuracy (mean±std) | Check accuracy (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| A. Glossary-only | Qwen2.5-Math-7B-Instruct | ErdosConventionsBench | **TBD** | - | - | Prompting baseline |
| B. Neutral checks + solve | Qwen2.5-Math-7B-Instruct | ErdosConventionsBench | **TBD** | **TBD** | - | Engagement-matched control |
| **C. Discriminative definition unit tests + solve** | Qwen2.5-Math-7B-Instruct | ErdosConventionsBench | **TBD** | **TBD** | - | Proposed |
| A. Glossary-only | Llama-3.1-8B-Instruct | ErdosConventionsBench | **TBD** | - | - | Prompting baseline |
| B. Neutral checks + solve | Llama-3.1-8B-Instruct | ErdosConventionsBench | **TBD** | **TBD** | - | Engagement-matched control |
| **C. Discriminative definition unit tests + solve** | Llama-3.1-8B-Instruct | ErdosConventionsBench | **TBD** | **TBD** | - | Proposed |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| k=1 vs k=3 checks | Fewer checks before solve | If effect is real, k=3 > k=1 but diminishing returns |
| Per-family evaluation | Report results separately for convolution / asymptotics / completeness families | If effect is general, C > B in ≥2 families |

### Experimental Rigor

**Variance & Reproducibility:**
- For greedy decoding (temperature=0), the evaluation is deterministic; report a single run.
- If temperature>0 is used (optional robustness), run **3 seeds** (e.g., `seeds=[42, 123, 456]`) and report mean±std.

**Validity & Controls:**
- **Extra-compute confound:** The primary claim is tested by C vs B, which are prompt-length matched and require similar computation.
- **Prompt-format confound:** All conditions use the same tags and answer format; only the check content differs.
- **Dataset artifact confound:** Report per-family results; ensure items are generated with diverse parameter ranges and include sanity checks where glossary and alternate convention coincide.

---

## Success Criteria

**Hypothesis** (directional): Discriminative definition checks improve downstream accuracy beyond engagement-matched neutral checks.

**Decision Rule** (concrete):
- **Proceed** if, on both base models, C improves main accuracy over B by **≥5 percentage points** (or by a paired bootstrap 95% CI excluding 0) on the full benchmark, and C > B holds in **at least 2 of 3** convention families.
- **Pivot** if C > B only for one convention family; refine benchmark generation or identify which convention types admit useful discriminative checks.
- **Refute** if C ≤ B (CI includes 0 or negative) on both models, or if models already achieve near-ceiling accuracy under B leaving no headroom.

---

## Impact Statement

If successful, Definition Unit Tests provide a low-engineering, training-free pattern for improving reliability in any LLM pipeline where the input includes binding definitions: instead of only providing a glossary, the system can include a small set of deterministic, discriminative checks to ensure conventions are actually applied. This could reduce “technically correct but wrong interpretation” failures in math-research agents, tool-calling agents that rely on API specifications, and evaluation pipelines that rely on rubric definitions.

---

## References

- [Towards Autonomous Mathematics Research](./references/Towards-Autonomous-Mathematics-Research/meta/meta_info.txt) - Feng et al., 2026
- [Semi-Autonomous Mathematics Discovery with Gemini](./references/Semi-Autonomous-Mathematics-Discovery-with-Gemini/meta/meta_info.txt) - Feng et al., 2026
- [ErdosProblems Definitions](./references/ErdosProblems-Definitions/meta/meta_info.txt) - ErdosProblems.com, accessed 2026
- [PromptPex](./references/PromptPex-Automatic-Test-Generation-for-Language-Model-Prompts/meta/meta_info.txt) - Sharma et al., 2025
- [Do LLMs Adhere to Label Definitions?](./references/Do-LLMs-Adhere-to-Label-Definitions/meta/meta_info.txt) - Mohammadi et al., 2025
- [ContractEval](./references/ContractEval-PACT/meta/meta_info.txt) - Lim et al., 2025
- [FASTRIC](./references/FASTRIC-Prompt-Specification-Language/meta/meta_info.txt) - Jin, 2025
- [Offscript](./references/Offscript-Automated-Auditing/meta/meta_info.txt) - Clark et al., 2025
- [Chain-of-Dictionary Prompting](./references/Chain-of-Dictionary-Prompting/meta/meta_info.txt) - Lu et al., 2023
- [Exploring the Hidden Reasoning Process of LLMs by Misleading Them](https://arxiv.org/abs/2503.16401) - Chen et al., 2025
- [Reasoning or Reciting?](https://arxiv.org/abs/2307.02477) - (authors per paper), 2023
- [QuestBench](https://arxiv.org/abs/2503.22674) - (authors per paper), 2025
- [AutoMonitor-Bench](https://arxiv.org/abs/2601.05752) - (authors per paper), 2026
- [Answering Questions in Stages: Prompt Chaining for Contract QA](https://arxiv.org/abs/2410.12840) - Roegiest and Chitta, 2024
- [LLM-SQL-Solver](https://arxiv.org/abs/2312.10321) - (authors per paper), 2023
- [System Prompts as a Mechanism of Bias in LLMs](https://arxiv.org/abs/2505.21091) - (authors per paper), 2025
