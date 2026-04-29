# untitled

# Equation-Consistency Gated Reflection for Small Models on Verifiable Math

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Small language models (SLMs, e.g., 7-9B parameter LLMs) are increasingly deployed because they are cheaper and faster than frontier models. A common attempt to improve SLM reliability at inference time is to add a *reflection* step: the model first produces a solution, then critiques its own reasoning and revises the answer. Reflection is attractive because it requires no additional training.

However, recent evidence suggests that reflection can be actively harmful for small models. **When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents** finds that for 7-9B models, 50-69% of *correct* answers contain flawed reasoning. They quantify this with a **Reasoning Integrity Score (RIS)**: each reasoning step is rated in {0, 0.5, 1} (wrong / partially flawed / correct) and averaged; traces with RIS<0.8 are treated as flawed. On GSM8K, this includes 59.2% of correct Llama-3-8B outputs and 62.7% of correct Qwen-2.5-7B outputs (Table 1). Moreover, self-critique and “verify each step” prompts harm RIS in 78% of conditions (mean Cohen’s d ≈ -0.14/-0.15; effect size in standard deviations), consistent with **pseudo-reflection**: critique-like text without an external/verifiable anchor that often introduces new errors. This raises a practical question for 2026 deployment: can we make reflection safe enough to use with small models, without resorting to expensive learned verifiers or large LLM judges?

Many high-stakes SLM deployments involve *verifiable* subroutines: arithmetic and basic quantitative reasoning, data transformation, and code-like computations. In these domains, a large subset of reasoning steps can be checked automatically (e.g., whether a stated equation is numerically correct). This proposal investigates whether a simple, deterministic verification signal is sufficient to prevent the most damaging reflection failure mode: flipping a correct answer into an incorrect one.

### The Problem

Empirically, reflection for small models has two coupled failure modes:

- **Correct-to-incorrect regressions (c->i)**: the model initially answers correctly, then “improves” it into an incorrect answer.
- **Hidden process failures**: even when the final answer is correct, the reasoning may contain incorrect intermediate steps (right-for-wrong-reasons), which complicates using reflection as a reliability tool.

Existing solutions either rely on (i) learned reward models or verifiers to rank candidate solutions, or (ii) expensive frontier LLM judges. For example, **GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements** trains outcome-based and stepwise reward models (ORM/SORM) and uses them to drive refinement. This improves accuracy but is not training-free.

The open problem targeted here is: **Is there a minimal, training-free gating rule that makes reflection “do no harm” on verifiable tasks for small models?** If such a rule exists, it would be a simple deployment recommendation (use reflection only with a gate; otherwise avoid it), and it would partially explain why ungated reflection fails.

### Key Insight and Hypothesis

**Key insight**: In verifiable arithmetic reasoning, many harmful reflection edits should introduce *checkable inconsistencies* (e.g., incorrect equalities like “80 * 0.2 = 12”). Even if the model can generate fluent critiques, it cannot reliably detect these inconsistencies internally. A deterministic arithmetic checker can.

**Hypothesis**: On GSM8K and robustness variants (GSM-Plus), a training-free *equation-consistency gate* that chooses between the original solution and the reflected revision based on arithmetic consistency will:

1) reduce c->i regressions by 30% relative vs naive “always accept the revision”, and
2) improve net accuracy / robustness compared to naive reflection and compute-matched self-consistency baselines.

The outcome is uncertain because (i) reflection may fail mostly via *logical* errors that remain arithmetically consistent, and (ii) equation extraction coverage may be too low, causing the gate to degenerate to “keep the original”.

---

## Proposed Approach

### Overview

We propose **Equation-Consistency Gated Reflection (ECGR)**:

1. Generate an initial chain-of-thought (CoT) solution and final answer.
2. Generate a self-critique + revised solution.
3. Compute an **equation-consistency score** for each solution by extracting arithmetic equalities and checking them with a symbolic/numeric evaluator.
4. Output the final answer from the solution with the higher consistency score (ties default to the original solution).

This is training-free and uses no LLM judges.

### Method Details

**Prompting format (light constraint, not rigid templates)**
- Use a standard CoT solve prompt for the main comparison (baseline).
- Optionally (coverage booster / follow-up), use an “equation-encouraging” solve prompt variant, e.g.:
  - “Solve step by step. When you do a calculation, write it as an equation using ‘=’ (e.g., 3*4=12). End with ‘Final Answer: <number>’.”
- Reflection prompt (used in both naive reflection and ECGR):
  - “Critique your previous reasoning for errors and provide a corrected solution if needed. End with ‘Final Answer: <number>’.”

**Equation extraction**
- Extract candidate equations using a conservative regex matching patterns like:
  - `LHS = RHS`, where `LHS` contains only digits, parentheses, decimal points, spaces, and operators `+ - * /`.
- Support simple fractions `a/b` and numbers with commas.
- Ignore equations containing variables or units.

**Equation verification**
- Parse and evaluate both sides via SymPy (or an equivalent math-expression parser).
- Mark an extracted equation correct if `abs(eval(LHS) - eval(RHS)) <= tol` (tol = 1e-6; also consider relative tolerance for large magnitudes).

**Consistency score**
- Let `E` be extracted equations; let `ok(e)` be {0,1} correctness.
- Score `S = mean_{e in E} ok(e)`.
- If `|E| = 0`, set `S = 0.5` (uninformative) and rely on tie-breaking.

**Selection rule**
- If `S(revised) > S(original)`: choose revised.
- Else: choose original.

### Key Innovations

- **A training-free gate for reflection safety**: Unlike learned reward models (ORM/SORM) or LLM judges, ECGR uses a deterministic verifier applicable to verifiable reasoning domains.
- **Mechanism-focused evaluation**: We explicitly measure and attribute improvements via c->i / i->c transition rates and the prevalence of arithmetic inconsistencies in harmful revisions.
- **Robustness evaluation beyond GSM8K**: We test on GSM-Plus perturbations to measure whether the gate improves robustness rather than overfitting to a single benchmark.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) self-correction / reflection prompting, (ii) verifiable reasoning and process verification, and (iii) robustness evaluation for math word problems.

Reflection and self-correction methods are often motivated by the idea that “verification is easier than generation”, but recent work shows reflection is prompt-sensitive and can introduce false positives, especially for smaller models. In parallel, verification-based approaches for math reasoning (process reward models, outcome reward models, program-of-thought verification) can yield large gains but typically require training verifiers or using large external models. Finally, robustness benchmarks like GSM-Plus show that high GSM8K accuracy does not imply robustness under perturbations, motivating evaluation beyond a single test set.

### Related Papers

- **[When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents](./references/When-Small-Models-Are-Right-for-Wrong-Reasons-Process-Verification-for-Trustworthy-Agents/meta/meta_info.txt)**: Shows 50-69% of correct answers from 7-9B models have flawed reasoning (RIS<0.8) and that self-critique often harms via “pseudo-reflection”.
- **[GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers](./references/GSM-Plus-A-Comprehensive-Benchmark-for-Evaluating-the-Robustness-of-LLMs-as-Mathematical-Problem-Solvers/meta/meta_info.txt)**: Introduces GSM8K perturbations and robustness metrics (PDR/ASP) and evaluates prompting methods under distribution shift.
- **[GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements](./references/GLoRe-When-Where-and-How-to-Improve-LLM-Reasoning-via-Global-and-Local-Refinements/meta/meta_info.txt)**: Improves GSM8K via learned reward models (ORM/SORM) and global/local refinement; not training-free.
- **[Self-Reflection Outcome is Sensitive to Prompt Construction](./references/Self-Reflection-Outcome-is-Sensitive-to-Prompt-Construction/meta/meta_info.txt)**: Shows reflection is sensitive to prompt wording and proposes conservative prompt constructions to reduce unnecessary changes.
- **[Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs](./references/Premise-Augmented-Reasoning-Chains-Improve-Error-Identification-in-Math-Reasoning-with-LLMs/meta/meta_info.txt)**: Shows premise-structured reasoning improves error identification; still relies on LLM-based verification.

Additional relevant work (URLs or local KB paths):
- **[Small Language Models Need Strong Verifiers to Self-Correct Reasoning](https://arxiv.org/abs/2404.17140)** (SCORE): Shows self-correction is bottlenecked by verifier quality; prompted self-correction often degrades performance without a strong verifier.
- **[Program of Equations Thoughts to Solve Algebra Word Problems](https://arxiv.org/abs/2505.20170)** (POET): Extracts equations from LLM reasoning and solves them with SymPy to avoid arithmetic errors; related equation-extraction machinery but not targeted to reflection harm / c->i flips.
- **[VeriCoT: Neuro-Symbolic Chain-of-Thought Validation via Logical Consistency Checks](https://arxiv.org/abs/2511.04662)**: Uses autoformalization + SMT to validate CoT and drive self-reflection; targets logical validity beyond arithmetic and relies on learned components.
- **[Process Reward Models That Think](https://arxiv.org/abs/2504.16828)**: Generative PRM that verifies steps via long verification-CoT; illustrates that strong verifiers usually require training.
- **[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)** (PRM800K): Process supervision / PRMs for math reasoning.
- **[Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)**: Majority vote over multiple CoT samples (compute-matched baseline family).
- **[Program-of-Thought Prompting](https://arxiv.org/abs/2211.12588)**: Uses executable programs as reasoning intermediates.
- **[Math-Shepherd](https://arxiv.org/abs/2312.09390)** and **[Math-Rev](https://arxiv.org/abs/2408.00139)**: Step-level verification / collaborative verification pipelines for math reasoning.
- **[GSM-IC](https://arxiv.org/abs/2305.18844)** and **[SVAMP](https://aclanthology.org/2021.findings-emnlp.230/)**: Robustness tests for math word problems.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Reflection / self-correction prompting | Generate critique and revise answer | Self-Refine style prompting; MoP (2406.10400); RIS paper (2601.00513) | GSM8K, MMLU, planning tasks | Can introduce false positives; prompt-sensitive; can harm SLMs |
| Learned verifiers / reward models | Train ORM/PRM to rank candidates or guide refinement | GLoRe (2402.10963), PRM800K (2305.20050), Math-Shepherd | GSM8K/MATH | Requires training data and compute; verifier generalization limits |
| Tool / execution-based verification | Convert reasoning to executable form or check steps programmatically | PoT (2211.12588), Math-Rev | GSM8K/MATH | Needs reliable translation to code; may fail on non-executable steps |
| Robustness benchmarks | Evaluate under perturbations / shifts | GSM-Plus (2402.19255), SVAMP, GSM-IC | PDR/ASP; perturbed accuracy | Mostly outcome-based metrics; limited process attribution |

### Closest Prior Work

1. **RIS paper (arXiv:2601.00513)**: Identifies pseudo-reflection harms for 7–9B models, but uses LLM judges for RIS and does not propose a deployable, fully automated gate for verifiable subtasks.
2. **SCORE (arXiv:2404.17140)**: Shows small models can learn to *refine* but are bottlenecked by weak *verifiers*; prompted self-correction often degrades performance without a strong verifier (e.g., GPT-4). It does not propose deterministic arithmetic checking or target c->i regressions from reflection.
3. **GLoRe (arXiv:2402.10963)**: Uses trained reward models (ORM/SORM) to decide when/where to refine, improving GSM8K, but is not training-free and does not focus on “do no harm” gating.
4. **POET / PoT-style execution (arXiv:2505.20170; arXiv:2211.12588)**: Use equation extraction or code execution (SymPy/Python) to *solve* problems more reliably by offloading computation. This is related tooling, but does not study reflection loops or choosing between an original vs revised solution to prevent regressions.
5. **Trained verifiers / neuro-symbolic CoT validation (e.g., VeriCoT 2511.04662; ThinkPRM 2504.16828; OPV 2512.10756; xVerify 2504.10481)**: Provide stronger verification signals (often learned and/or tool-augmented) but require training and do not isolate the specific reflection failure mode we target.

**Novelty Kill Search Summary (2026-02-21):**
- Web queries: “equation consistency gate reflection GSM8K”, “arithmetic checker choose between original and critique”, “deterministic verifier reflection regressions”, “c->i flips reflection math”. Closest hits were POET (equation extraction + SymPy solving) and SCORE (strong verifier needed), but neither proposes a *training-free arithmetic-consistency gate* for reflection safety.
- Local KB / repo scan: `Grep` over all agents’ draft + finalized proposals for “equation consistency”, “arithmetic checker”, “SymPy gate”, “pseudo-reflection”, “RIS” found no prior proposal matching “deterministic equation-consistency gate for reflection c->i prevention”.
- Paper spot-check: reviewed recent verifier papers in local KB (VeriCoT, ThinkPRM, OPV, xVerify) and they focus on trained verifiers / evaluation, not a training-free gate.

No prior work explicitly using a training-free arithmetic consistency gate to prevent reflection regressions in small models was found. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| RIS paper (2601.00513) | Measures hidden reasoning failures; finds self-critique harms | Uses LLM judges; no deployable gate for verifiable tasks | Add deterministic verifiable-domain gate | Cheap and automatable; directly targets c->i regressions |
| GLoRe (2402.10963) | Learned ORM/SORM + refinement improves GSM8K | Requires training reward models | Training-free gate | Lower deployment cost; clearer attribution |
| MoP (2406.10400) | Reduces reflection false positives via prompt mixtures | No deterministic verification; not domain-specific | Use arithmetic consistency instead of prompt-only fixes | More objective; suited to verifiable domains |
| GSM-Plus (2402.19255) | Robustness benchmark; compositional prompting | Does not study reflection regressions or gating | Apply gate on GSM-Plus | Tests if gate improves robustness under perturbations |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Llama-3-8B-Instruct | 8B | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct | Primary target (7-9B regime) |
| Qwen2.5-7B-Instruct (optional) | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Optional second model for generality |

**Benchmarks and Metrics:**

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| GSM8K | Grade-school math word problems (requires multi-step arithmetic reasoning) | Exact-match accuracy | test (1319) | https://huggingface.co/datasets/openai/gsm8k | standard parsing (extract final number) |
| GSM-Plus | Perturbations of GSM8K test questions (robustness benchmark) | Accuracy, PDR, ASP | full set (10,552) | https://huggingface.co/datasets/qintongli/GSM-Plus | authors’ metrics (PDR/ASP) |

**Robustness metrics (GSM-Plus)** (Li et al., 2024):
- **Performance Drop Rate (PDR)**: relative accuracy drop from GSM8K to GSM-Plus perturbations.
  - Definition (GSM-Plus Sec. 3.3): `PDR = 1 - (Acc_GSMPlus / Acc_GSM8K)`.
- **Accurately Solved Pairs (ASP)**: fraction of (seed, perturbed) pairs where *both* are answered correctly (GSM-Plus Sec. 3.3). If GSM8K has |D| seed problems and each seed has N perturbations, then:
  - `ASP = (1/(N·|D|)) · sum_{(x,y) in D} sum_{(x',y') in pert(x)} I[LM(x)=y] · I[LM(x')=y']`.

**Published GSM-Plus baselines (Table 3.2 in Li et al., 2024; extracted from `./references/GSM-Plus-.../sections/3.2 Dataset Construction.md`)**:

| Model | GSM8K Acc | GSM-Plus Acc | PDR | ASP |
|---|---:|---:|---:|---:|
| GPT-4 | 93.25% | 85.58% | 8.23% | 81.54% |
| GPT-3.5-Turbo | 73.62% | 61.19% | 16.88% | 51.36% |
| Mistral-7B | 39.58% | 26.18% | 33.86% | 18.66% |
| LLaMA-2-7B | 13.42% | 8.12% | 39.49% | 3.97% |
| CodeLlama-7B | 25.32% | 15.05% | 40.56% | 10.00% |
| MetaMath-Mistral (math SFT) | 77.79% | 56.25% | 27.69% | 50.56% |
| SEGO-7B (math SFT) | 68.69% | 44.71% | 34.91% | 40.68% |

(These published baselines are for context. Our main experiments re-run our methods on modern 7-9B models: Llama-3-8B-Instruct and optionally Qwen2.5-7B-Instruct.)

**License note (GSM-Plus)**: GSM-Plus is CC BY-SA 4.0 and described as test-only (not for training); we use it strictly for evaluation.

**Baseline Ladder (REQUIRED):**

- **Level 1 (zero-shot / CoT prompting)**: One-pass CoT solve prompt.
- **Level 4 (inference-time scaling)**: **Self-consistency @2 (SC@2)**: sample 2 independent CoT solutions and majority-vote the final answer; tie-break by first sample.
- **Level 5 (closest method family)**: **Naive reflection**: solve -> critique -> always accept revision.

**Our method**: ECGR (solve -> critique -> choose original vs revised via equation-consistency score).

**Decoding / sampling**:
- Default (one-pass CoT / naive reflection / ECGR): greedy decoding (temperature=0, top_p=1.0) for both solve and critique phases to remove sampling variance.
- SC@2: sample 2 independent CoT solutions with temperature=0.7, top_p=0.95; majority vote on final answer; tie-break = first sample. Run 3 seeds: `seeds=[42,123,456]`.

**Resource Estimate (order-of-magnitude)**:
- Questions: GSM8K test (1,319) + GSM-Plus (10,552) = 11,871 (~12k) prompts.
- Generations:
  - Generate (solve, reflect) once per prompt: 2x * 12k = 24k generations; reused to score **naive reflection**, **ECGR**, and the **final-answer agreement** ablation (no extra model calls).
  - SC@2: 2 samples per prompt x 3 seeds: (2x * 12k) * 3 = 72k generations.
  - Total ~ 96k generations (~100k).
- If avg output length is ~400 tokens, total output is ~40M tokens. With typical A100 80GB vLLM generation throughput on 8B-class models on the order of 2k-3k tokens/s (e.g., Azure HPC vLLM benchmarks report ~2.6k tokens/s for Llama-3.1-8B chat workloads), this is a few GPU-hours of pure decode; we conservatively budget <20 A100-hours including overhead and parsing.
- SymPy checks run on CPU and are negligible vs decoding.

**Optional de-risking pilot (recommended before full run)**:
- Generate (solve, reflect) outputs for a random 100-200 GSM8K test subset, then measure:
  - equation extraction coverage (% with ≥1 extractable equation), and
  - within c->i regressions (orig correct, revised incorrect), how often the revised solution contains ≥1 *incorrect* extracted equation (an upper bound on ECGR’s potential “catch rate”).
- If coverage <50% or the catch rate is near 0, expect ECGR to match simple conservatism baselines; pivot to better extraction / softer equation prompting or drop the approach.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Accuracy (mean+/-std) | PDR (GSM-Plus) | ASP (GSM-Plus) | Source | Notes |
|--------|------------|-----------|----------------------|----------------|----------------|--------|-------|
| One-pass CoT | Llama-3-8B | GSM8K | **TBD** | - | - | - | To be verified |
| SC@2 | Llama-3-8B | GSM8K | **TBD** | - | - | - | To be verified |
| Naive reflection | Llama-3-8B | GSM8K | **TBD** | - | - | - | To be verified |
| **ECGR (ours)** | Llama-3-8B | GSM8K | **TBD** | - | - | - | To be verified |
| One-pass CoT | Llama-3-8B | GSM-Plus | **TBD** | **TBD** | **TBD** | - | To be verified |
| SC@2 | Llama-3-8B | GSM-Plus | **TBD** | **TBD** | **TBD** | - | To be verified |
| Naive reflection | Llama-3-8B | GSM-Plus | **TBD** | **TBD** | **TBD** | - | To be verified |
| **ECGR (ours)** | Llama-3-8B | GSM-Plus | **TBD** | **TBD** | **TBD** | - | To be verified |

#### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Final-answer agreement baseline | If orig==rev take it else take orig | Tests whether most gains come from simple conservatism |
| Length-only chooser | Choose longer solution (orig vs rev) | Should not match ECGR if our mechanism is real |
| Coverage stress test | Report % with >=1 extractable equation | If <50%, mechanism likely too weak |

### Experimental Rigor

**Variance & Reproducibility:**
- Use deterministic decoding for all methods (greedy) to remove sampling variance, except SC@2 which requires sampling; for SC@2 use fixed seeds (e.g., `seeds=[42,123,456]`) and report mean+/-std.

**Validity & Controls:**
- **Prompt confound control (optional follow-up)**: If using the equation-encouraging prompt to increase extraction coverage, also report one-pass CoT accuracy under the standard CoT prompt vs the equation-encouraging prompt. If the equation-encouraging prompt degrades GSM8K by >3 points, treat it as too intrusive and revert to standard prompting + naturally occurring equations.
- **Confounder (length proxy)**: report correlation of consistency score with output length and with #equations; include length-only chooser ablation.
- **Data leakage / memorization**: GSM8K may be in some model training corpora; GSM-Plus perturbations reduce exact memorization effects. Report both.
- **Parsing brittleness**: report equation-extraction failure rate and the distribution of |E| (# extracted equations) to ensure results are not dominated by a narrow subset of outputs.

**Ethical / security considerations:**
- No human data, no personal data. The method is a safety-style reliability improvement for verifiable reasoning; no obvious dual-use beyond general capability uplift.

---

## Success Criteria

**Hypothesis** (directional): ECGR reduces c->i regressions by catching arithmetic inconsistencies in harmful revisions, improving net robustness on GSM-Plus while maintaining or improving GSM8K accuracy.

**Decision Rule** (concrete):
- **Proceed** if ECGR:
  - reduces c->i regressions by 30% relative vs naive reflection on both GSM8K and GSM-Plus, AND
  - achieves higher GSM-Plus robustness (lower PDR and/or higher ASP) than both naive reflection and SC@2 at matched compute, AND
  - equation coverage >= 75% (solutions with >=1 extractable equation).
- **Pivot** if ECGR reduces c->i but coverage is 50-75%: try improving equation extraction (better regex, Math-Verify parser) or prompt for more explicit equations without degrading accuracy.
- **Refute** if any holds:
  - coverage <50%, OR
  - ECGR performs similarly to the final-answer agreement baseline (equation checking not adding value), OR
  - ECGR does not improve GSM-Plus robustness vs SC@2 (inference-time scaling already captures the benefit).

---

## Impact Statement

If successful, this work provides a simple deployment rule for SLMs: reflection should only be used on verifiable tasks when gated by deterministic consistency checks; otherwise ungated reflection can reduce reliability. This would enable safer use of reflection loops in edge/latency-constrained deployments where training verifiers or calling large external judges is impractical.

---

## References

- [When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents](./references/When-Small-Models-Are-Right-for-Wrong-Reasons-Process-Verification-for-Trustworthy-Agents/meta/meta_info.txt) - Advani, 2026
- [GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers](./references/GSM-Plus-A-Comprehensive-Benchmark-for-Evaluating-the-Robustness-of-LLMs-as-Mathematical-Problem-Solvers/meta/meta_info.txt) - Li et al., 2024
- [GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements](./references/GLoRe-When-Where-and-How-to-Improve-LLM-Reasoning-via-Global-and-Local-Refinements/meta/meta_info.txt) - Havrilla et al., 2024
- [Self-Reflection Outcome is Sensitive to Prompt Construction](./references/Self-Reflection-Outcome-is-Sensitive-to-Prompt-Construction/meta/meta_info.txt) - Liu et al., 2024
- [Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs](./references/Premise-Augmented-Reasoning-Chains-Improve-Error-Identification-in-Math-Reasoning-with-LLMs/meta/meta_info.txt) - Mukherjee et al., 2025
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) - Zelikman et al., 2022
- [Program-of-Thought Prompting](https://arxiv.org/abs/2211.12588) - Chen et al., 2022
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Small Language Models Need Strong Verifiers to Self-Correct](https://arxiv.org/abs/2404.17140) - 2024
- [GSM-IC](https://arxiv.org/abs/2305.18844) - 2023
- [SVAMP](https://aclanthology.org/2021.findings-emnlp.230/) - Patel et al., 2021
- [Math-Shepherd](https://arxiv.org/abs/2312.09390) - 2023
- [Math-Rev](https://arxiv.org/abs/2408.00139) - 2024
- [CoT-Self-Instruct](https://arxiv.org/abs/2507.23751) - 2025
