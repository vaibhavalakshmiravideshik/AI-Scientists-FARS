# untitled

# Execution-Trace Guided Remasking for Diffusion Code Generation

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Diffusion language models (DLMs) generate text by iteratively denoising a partially masked token sequence, rather than producing tokens strictly left-to-right. Because tokens can remain masked and be resampled at later denoising steps, DLMs naturally support revising arbitrary positions in the output. We refer to large diffusion language models as **diffusion LLMs (dLLMs)**.

Recent open dLLMs such as **LLaDA** and **Dream** have shown competitive results on standard code generation benchmarks such as HumanEval (164 Python function synthesis problems) and MBPP (Mostly Basic Python Programming Problems; 974 problems), and diffusion-specific inference algorithms (e.g., remasking schedules, constrained decoding, and search-based test-time scaling) are an active area of development.

For code generation, many recent reliability improvements focus on syntactic correctness (e.g., grammar-constrained decoding for diffusion models). However, real programming failures are often semantic: a solution can be syntactically valid and locally high-confidence, yet still fail unit tests.

### The Problem

**Problem setting**: We consider code generation with an available unit-test oracle at inference time (e.g., CI tests for a function-level task). Given an initial candidate program produced by a diffusion model, we can run some tests and obtain a pass/fail result plus execution diagnostics (exceptions, stack traces, and line execution events).

Existing diffusion decoding methods do not use this semantic feedback to decide *where* to spend a limited revision budget:

- **Confidence-based remasking** (common in masked diffusion decoding): selects low-confidence tokens to **remask** (replace with [MASK]) and resample, but semantic bugs can occur in tokens the model assigns high probability to.
- **[CORE: Context-Robust Remasking](./references/CORE-CONTEXT-ROBUST-REMASKING-FOR-DIFFUSION-LANGUAGE-MODELS/meta/meta_info.txt)**: identifies context-brittle tokens via perturbation-based instability scoring and improves greedy pass@1 on code benchmarks, but it is still purely model-internal and does not exploit execution feedback.
- **Execution-guided code generation for autoregressive LLMs**, such as **EG-CFG** (Execution Guided Line-by-Line Code Generation), integrates execution feedback during generation, but does not leverage diffusion's ability to resample a small subset of positions while freezing the rest.

**Practical gap**: If we have budget for only one extra repair attempt (to control latency/cost), we need a strong heuristic for selecting a small set of tokens/spans to revise. Diffusion models provide a natural primitive for this: keep most tokens fixed and resample only a targeted subset (conditional infilling / inpainting), but current work lacks principled ways to pick that subset using semantic feedback.

### Key Insight and Hypothesis

**Key insight**: When a candidate program fails unit tests, execution diagnostics provide a localization signal for the code region most responsible for failure (e.g., the topmost stack frame in the candidate file, or the last executed lines in the candidate function before the assertion fails). A diffusion model can exploit this signal by remasking only tokens in the localized region and resampling them while freezing the rest of the program.

**Hypothesis**: Under a fixed extra compute budget of one repair iteration, execution-trace guided remasking will improve held-out test **pass@1** over (i) no repair and (ii) global low-confidence remasking, because it targets semantically implicated regions even when their tokens are locally high-confidence.

Why this could fail: (1) executed-line localization may be too noisy (many lines executed); (2) freezing most tokens may prevent necessary global changes; (3) diffusion conditional infilling quality may be poor when only a small region is masked.

---

## Proposed Approach

### Overview

We propose an inference-only algorithm that adds at most one repair step to diffusion code generation:

1. **Generate** a candidate program with a dLLM.
2. **Run a small feedback subset of unit tests**. If all pass, return the program.
3. If any feedback test fails, **localize an edit region** using execution diagnostics.
4. **Remask tokens within the localized region** (up to a fixed token budget) and run a short conditional diffusion procedure to resample only those tokens while freezing all other tokens.
5. Evaluate the final program on a disjoint held-out test subset.

The only difference between our method and the main baseline is *how the remask set is chosen*; the repair compute budget and test budget are fixed.

### Method Details

#### Notation and default hyperparameters
- Let the generated program be tokenized as output tokens \(y\) (excluding the prompt).
- Let \(L\) be the maximum number of generated tokens (default \(L=512\)).
- Let \(N\) be the number of diffusion steps for the initial generation (default \(N=128\)).
- Let \(S\) be the number of diffusion steps used for the conditional repair (default \(S=32\)).
- Let \(K\) be the maximum number of output tokens that can be remasked during repair (default \(K=64\)).
- Let **NFE** be the approximate number of model forward evaluations (forward passes). For diffusion sampling, NFE is approximately the number of diffusion steps executed, since each denoising step calls the model once.
- Let \(T_{fb}\) be the feedback test subset and \(T_{eval}\) be the held-out evaluation tests, with \(T_{fb} \cap T_{eval} = \emptyset\).

We measure **pass@1** as the fraction of problems where the single returned program passes **all** tests in \(T_{eval}\).

#### Test splitting (to avoid test leakage)
We use **EvalPlus**, a benchmark suite that extends HumanEval and MBPP with additional unit tests to catch edge-case bugs. We evaluate on HumanEval+ (164 problems) and MBPP+ (974 problems) and deterministically split tests per problem into \(T_{fb}\) and \(T_{eval}\) (default 20%/80%) using a fixed RNG seed. The model may only use outcomes and traces from \(T_{fb}\) for repair decisions. Final metrics are computed on \(T_{eval}\).

#### Execution-trace localization
When feedback tests fail, we compute a set of candidate-code line numbers \(\mathcal{L}\) to target:

1. **Exception-based localization** (fast path): if a failure raises an exception whose traceback includes a frame in the candidate program file, take the topmost such frame's line number and include a small window (default +/- 1 line).
2. **Assertion-based localization** (fallback): if the failure is an assertion in the test file (common in unit tests), run the failing feedback tests under a lightweight line tracer restricted to the candidate program file and collect the last \(M\) executed candidate-code line events before failure (default \(M=20\)). Set \(\mathcal{L}\) to the union of these lines across failing feedback tests.

If localization fails (no candidate-code frames and the tracer cannot run due to syntax errors), we fall back to the global low-confidence baseline.

#### Mapping lines to token indices
We map \(\mathcal{L}\) to a set of output token indices \(\mathcal{I}(\mathcal{L})\) by decoding output tokens to a string, computing per-token character offsets via tokenizer offset mapping, and selecting tokens whose character spans intersect any targeted source-code lines.

#### Baseline repair vs trace-guided repair
Both methods use the same \(K\) and \(S\).

- **Global low-confidence repair**: if any feedback test fails, select the \(K\) output tokens with lowest model confidence (max probability) from the final diffusion step and remask them.
- **Execution-trace guided repair (ours)**: if any feedback test fails, compute \(\mathcal{L}\) and \(\mathcal{I}(\mathcal{L})\). If \(|\mathcal{I}(\mathcal{L})| \le K\), remask all tokens in \(\mathcal{I}(\mathcal{L})\); otherwise remask the \(K\) lowest-confidence tokens restricted to \(\mathcal{I}(\mathcal{L})\).

#### Conditional diffusion repair (localized infilling)
Given a remask set \(R\subseteq\{1,\dots,|y|\}\), we construct \(y'\) where tokens in \(R\) are replaced by the model's mask token and all other tokens remain fixed. We then run conditional diffusion sampling for \(S\) steps, updating only masked positions.

### Key Innovations

1. **Execution-driven remask localization for dLLMs**: use runtime traces to decide which region of code to revise under a strict one-repair budget.
2. **Held-out test split for sound evaluation**: use feedback tests for repair while evaluating on disjoint tests, preventing direct optimization on the evaluator.
3. **Localized diffusion repair primitive**: cast repair as conditional infilling over a small remasked token set, exploiting diffusion's ability to revise arbitrary positions without regenerating the full program.

---

## Related Work

### Field Overview

This proposal connects three lines of work:

1. **Diffusion language models and inference algorithms**: discrete diffusion models such as D3PM, SEDD, and MDLM enable iterative masked denoising. Recent dLLMs (LLaDA, Dream) scale this paradigm and inspire inference-time improvements such as remasking schedules (ReMDM), instability-based revision (CORE), and search-based test-time scaling (MEDAL, PRISM, UnMaskFork, TReASURe).

2. **Diffusion for code generation and repair**: diffusion models have been applied to code generation (CodeFusion, TreeDiff, DiffuCoder) and to repair-style settings (e.g., denoising from a noised program as a repair operator). Most diffusion-for-code work is syntax- or likelihood-driven rather than using runtime semantic feedback.

3. **Execution-guided code generation and repair for autoregressive LLMs**: many methods leverage unit tests, error messages, and traces to improve code (CodeRL, Reflexion, CYCLE, EG-CFG, agentic repair). These works motivate execution feedback as a strong supervision signal, but they do not study diffusion-specific localized resampling.

### Related Papers

- **[CORE: Context-Robust Remasking](./references/CORE-CONTEXT-ROBUST-REMASKING-FOR-DIFFUSION-LANGUAGE-MODELS/meta/meta_info.txt)**: perturbation-based instability scoring for token revision in masked diffusion; strong greedy pass@1 gains on code.
- **[Don't Settle Too Early (RemeDi)](./references/Dont-Settle-Too-Early-Self-Reflective-Remasking-for-Diffusion-Language-Models/meta/meta_info.txt)**: learned remasking policy stream enabling revision of previously sampled tokens in diffusion models.
- **[ReMDM](https://arxiv.org/abs/2503.00307)**: principled remasking schedules for inference-time scaling in discrete diffusion models.
- **[LLaDA: Large Language Diffusion Models](https://openreview.net/forum?id=KnqiC0znVF)**: large diffusion language model family used widely for evaluation and inference research.
- **[Dream 7B](https://arxiv.org/abs/2508.15487)**: open diffusion LLM with flexible diffusion generation APIs.
- **[MDLM](https://arxiv.org/abs/2406.07524)**: masked diffusion LM with simplified objective and efficient samplers.
- **[SEDD](https://arxiv.org/abs/2310.16834)**: score-entropy discrete diffusion via ratio estimation.
- **[D3PM](https://arxiv.org/abs/2107.03006)**: foundational discrete diffusion framework.
- **[Diffusion-LM](https://arxiv.org/abs/2205.14217)**: continuous diffusion for text generation with controllability.
- **[TreeDiff](./references/TreeDiff-AST-Guided-Code-Generation-with-Diffusion-LLMs/meta/meta_info.txt)**: AST-span masking for diffusion code generation, improving pass@1 on HumanEval/MBPP.
- **[CodeDiffuSe](./references/CodeDiffuSe-A-masked-diffusion-framework-for-structure-aware-code-completion-and-repair/meta/meta_info.txt)**: diffusion framework for structure-aware code completion/repair using syntax/type signals.
- **[CodeFusion](https://arxiv.org/abs/2310.17680)**: diffusion model for NL-to-code, emphasizing iterative refinement.
- **[Diffusion is a code repair operator](./references/Diffusion-is-a-code-repair-operator-and-generator/meta/meta_info.txt)**: shows denoising from intermediate steps can serve as a repair primitive without using tests.
- **[DiffuCoder](https://arxiv.org/abs/2506.20639)**: masked diffusion model for code; includes RL-based post-training and EvalPlus evaluation.
- **[Corrective Diffusion Language Models (CDLM)](https://arxiv.org/abs/2512.15596)**: trains diffusion models to assign low confidence to corrupted visible tokens; introduces an executable Code Revision Benchmark.
- **[DiffTester](https://arxiv.org/abs/2509.24975)**: accelerates unit test generation for dLLMs via AST pattern mining.
- **[PRISM](https://arxiv.org/abs/2602.01842)**: hierarchical search and self-verification for efficient test-time scaling of dLLMs.
- **[Self-Rewarding SMC](https://arxiv.org/abs/2602.01849)**: sequential Monte Carlo inference-time scaling for masked diffusion language models.
- **[UnMaskFork](./references/UnMaskFork-Test-Time-Scaling-for-Masked-Diffusion-via-Deterministic-Action-Branching/meta/meta_info.txt)**: deterministic branching tree search for masked diffusion test-time scaling.
- **[TReASURe](https://arxiv.org/abs/2509.23146)**: tree reward-aligned search for test-time alignment in masked diffusion language models.
- **[DINGO](https://arxiv.org/abs/2505.23061)**: constrained decoding for diffusion LLMs under regular-language constraints.
- **[LAVE](https://arxiv.org/abs/2602.00612)**: lookahead-then-verify constrained decoding for diffusion LLMs under CFGs.
- **[EvalPlus](https://arxiv.org/abs/2305.01210)**: rigorous evaluation for HumanEval/MBPP via extended test suites.
- **[Execution Guided Line-by-Line Code Generation (EG-CFG)](https://arxiv.org/abs/2506.10948)**: integrates execution traces into autoregressive generation via classifier-free guidance.
- **[TraceCoder: A Trace-Driven Multi-Agent Framework for Automated Debugging of LLM-Generated Code](https://arxiv.org/abs/2602.06875)**: multi-agent trace-driven iterative repair for autoregressive code models; strong prior work on trace-based repair but higher overhead and not diffusion-specific.
- **[Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis](https://arxiv.org/abs/2412.14841)**: uses test/static-analysis feedback for improving code generation (autoregressive setting).
- **[Agentic Program Repair from Test Failures at Scale](https://arxiv.org/abs/2507.18755)**: large-scale test-driven repair agents combining static analysis and execution.
- **[CodeRL](https://arxiv.org/abs/2207.01780)**: RL with unit-test outcomes for code generation; uses critic-guided sampling.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: trial-and-error with reflection and (self-generated) tests for code.
- **[CYCLE](https://arxiv.org/abs/2403.18746)**: trains code LMs to refine code using execution feedback.
- **[TraceFixer](https://arxiv.org/abs/2304.12743)**: execution-trace driven program repair, showing traces provide strong localization.
- **[Type-Constrained Code Generation](https://arxiv.org/abs/2504.09246)**: constrained decoding using type systems (orthogonal semantic signal).

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Diffusion LMs (foundations) | iterative denoising in discrete spaces | D3PM, SEDD, MDLM | LM benchmarks, downstream tasks | sampling cost, calibration |
| Diffusion inference improvements | remasking, instability scoring, search | ReMDM, CORE, PRISM, UnMaskFork, TReASURe | reasoning + code (HumanEval/MBPP/EvalPlus) | mostly model-internal signals |
| Diffusion for code | diffusion specialized/trained for code | CodeFusion, TreeDiff, DiffuCoder | HumanEval/MBPP/EvalPlus | semantic correctness remains hard |
| Corrective diffusion training | train confidence to identify wrong visible tokens | CDLM | Code Revision Benchmark (CRB), HumanEval+/MBPP+ | requires training; limited edit ops |
| Execution-guided AR codegen/repair | tests/traces guide iterative generation | EG-CFG, CodeRL, Reflexion, CYCLE, agentic repair | HumanEval/MBPP/APPS/SWE-bench | often multi-iteration and high overhead |

### Closest Prior Work

1. **[CORE](./references/CORE-CONTEXT-ROBUST-REMASKING-FOR-DIFFUSION-LANGUAGE-MODELS/meta/meta_info.txt)**: revises context-brittle tokens via perturbation-based instability scoring. Our method instead uses an external semantic oracle (unit tests) to localize revisions.

2. **[Diffusion is a code repair operator](./references/Diffusion-is-a-code-repair-operator-and-generator/meta/meta_info.txt)**: uses denoising from an intermediate diffusion timestep as a repair primitive, but does not use runtime feedback to choose where to edit. We add execution-trace based localization.

3. **[EG-CFG](https://arxiv.org/abs/2506.10948)**: uses execution traces during autoregressive generation (line-by-line) and performs heavier test-time search. We study a diffusion-specific alternative: a single localized infill step after a failing run.

4. **[CDLM](https://arxiv.org/abs/2512.15596)**: improves diffusion self-correction by training confidence to highlight corrupted visible tokens. Our method is inference-only and uses external traces rather than relying on calibrated internal confidence.

5. **[TraceCoder](https://arxiv.org/abs/2602.06875)**: uses runtime tracing plus multi-agent iterative repair for LLM-generated code. Our method targets a different regime: a single additional diffusion infill step under a strict compute budget, without multi-turn agent loops or repeated full regenerations.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| CORE | instability-based token revision (no external oracle) | may miss semantically wrong but context-stable tokens | use execution traces to localize edits | traces point to failure-relevant region even if tokens are confident |
| CDLM | trains diffusion models to flag corrupted visible tokens | requires training; focuses on synthetic corruption | use runtime feedback to target real failures | test failures provide task-specific localization |
| EG-CFG | uses execution guidance during AR generation, with search | high overhead; not diffusion-localized | single localized diffusion infill step | lower overhead; preserves most of the program |
| TraceCoder | multi-agent trace-driven iterative debugging and repair | multi-turn, higher token cost, and AR-specific | one-step diffusion repair guided by traces | targets low-latency/one-shot regimes |
| Diffusion-as-repair operator | denoise a noised program to repair | no principled choice of noised region | noise/remask only trace-localized region | fewer unnecessary changes; better locality |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| LLaDA-8B-Base | 8B | https://huggingface.co/GSAI-ML/LLaDA-8B-Base | diffusion LM; start from CORE's decoding defaults |

**Training Data (if applicable):**

No training data needed - inference only.

**Other Resources (if applicable):**
- Unit-test execution via EvalPlus.
- Lightweight tracing via Python line instrumentation (prefer `sys.monitoring` on Python 3.12+; otherwise `sys.settrace` or coverage.py's tracer), restricted to the generated candidate file.

**Resource Estimate**:
- **Compute budget**: inference-only. Worst-case model forward passes per instance is \(N + S\) (default 128 + 32). Running MBPP+ (974) and HumanEval+ (164) with one sample each and 3 random seeds should fit within the 768 GPU-hour budget; wall-clock is likely dominated by Python test execution.
- **GPU memory**: 8B model fits on a single 80GB A100 (bf16) with standard inference.
- **CPU cost**: test execution dominates; enforce per-test and per-problem timeouts via EvalPlus.

We will report mean +/- std over 3 random seeds (affecting diffusion sampling randomness) and use the same fixed test split across seeds.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| HumanEval+ | 164 Python synthesis problems with extended test suites | pass@1 (fraction passing all held-out tests) | test | https://huggingface.co/datasets/evalplus/humanevalplus | https://github.com/evalplus/evalplus |
| MBPP+ | 974 Python problems with extended test suites | pass@1 (fraction passing all held-out tests) | test | https://huggingface.co/datasets/evalplus/mbppplus | https://github.com/evalplus/evalplus |

**Evaluation Scripts:**
- Use EvalPlus for sandboxed execution and timeouts.
- Add a deterministic per-problem test split (seeded) to separate feedback from evaluation.
- Implement a custom generation wrapper for LLaDA diffusion sampling and conditional infilling repair.

### Main Results

#### Baselines (planned)
We will evaluate the following methods under the same feedback-test budget and same worst-case forward-pass budget (reported as NFE = number of model forward passes):

- **No repair**: generate once and evaluate on \(T_{eval}\).
- **Global low-confidence repair**: generate, use \(T_{fb}\) only to decide whether to repair, and remask globally low-confidence tokens if needed.
- **CORE baseline**: run CORE decoding (no test-guided repair) and evaluate on \(T_{eval}\).
- **Compute-matched best-of-2**: generate two candidates using \(N' = \lceil(N+S)/2\rceil\) steps each so total NFEs approximately match \(N+S\); run \(T_{fb}\) on both and select the candidate with higher feedback pass rate; evaluate selected candidate on \(T_{eval}\).
- **Execution-trace guided repair (ours)**: generate, then if \(T_{fb}\) fails, localize and repair.

#### Published reference numbers (not directly comparable)
From **[CORE](./references/CORE-CONTEXT-ROBUST-REMASKING-FOR-DIFFUSION-LANGUAGE-MODELS/meta/meta_info.txt)** Table 1 (LLaDA-8B-Base, N=128, L=512, greedy):
- HumanEval: 12.20 (Low-Confidence base) -> 17.07 (+CORE)
- MBPP: 15.60 (Low-Confidence base) -> 24.80 (+CORE)

These use the original HumanEval/MBPP tests and do not use held-out test splitting.

#### Results Table (to be verified)

| Method | Base Model | Benchmark | pass@1 (held-out tests) | NFE (approx) | Source | Notes |
|--------|------------|-----------|--------------------------|--------------|--------|-------|
| No repair | LLaDA-8B-Base | HumanEval+ | TBD | 128 | - | Needs re-run |
| Global low-confidence repair | LLaDA-8B-Base | HumanEval+ | TBD | <=160 | - | Needs re-run |
| CORE | LLaDA-8B-Base | HumanEval+ | TBD | ~136 | - | Needs re-run (may require reimplementation if no official decoding code) |
| Best-of-2 (compute-matched) | LLaDA-8B-Base | HumanEval+ | TBD | ~160 | - | Needs re-run |
| **Ours (trace-guided repair)** | LLaDA-8B-Base | HumanEval+ | TBD | <=160 | - | To be verified |
| No repair | LLaDA-8B-Base | MBPP+ | TBD | 128 | - | Needs re-run |
| Global low-confidence repair | LLaDA-8B-Base | MBPP+ | TBD | <=160 | - | Needs re-run |
| CORE | LLaDA-8B-Base | MBPP+ | TBD | ~136 | - | Needs re-run |
| Best-of-2 (compute-matched) | LLaDA-8B-Base | MBPP+ | TBD | ~160 | - | Needs re-run |
| **Ours (trace-guided repair)** | LLaDA-8B-Base | MBPP+ | TBD | <=160 | - | To be verified |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | trace-localized remasking + conditional diffusion repair | best performance |
| Ours w/ random localized tokens | keep trace region but pick K random tokens in it | worse than full; tests whether confidence-within-region matters |
| Ours w/ widened line window | use +/- 5 lines instead of +/- 1 around localized line(s) | may degrade if localization becomes too diffuse |
| Ours w/ fewer repair steps | S in {8, 16, 32} | identifies minimum repair compute needed |

### Analysis (Optional)

- **Edit locality**: measure token/line edit distance between original and repaired code; trace-guided should yield smaller edits than global low-confidence.
- **Failure-type breakdown**: stratify improvements by failure type on \(T_{fb}\) (syntax error, runtime error, assertion failure).

---

## Success Criteria

**Criterion 1: Beats global confidence repair**
- Hypothesis: trace-guided repair improves pass@1 on \(T_{eval}\) relative to global low-confidence repair.
- Validation: improvement on MBPP+ and no degradation on HumanEval+.

**Criterion 2: Not explained by trivial test-time scaling**
- Hypothesis: trace-guided repair outperforms compute-matched best-of-2 selection.
- Validation: MBPP+ improvement persists after adding best-of-2 baseline.

**Decision rule**: If trace-guided repair does not beat global low-confidence repair on MBPP+ by at least 2 percentage points in mean pass@1 over 3 seeds (or is consistently worse across seeds), we treat the hypothesis as refuted for this setting.

---

## Impact Statement

If successful, this work provides a simple way to use unit-test execution feedback to improve diffusion-based code generation with a single additional repair step. This could make diffusion code models more useful in practical CI-backed workflows by converting one failing sample into a passing solution more often, without multi-turn agent loops or large best-of-N sampling.

---

## References

- [CORE: Context-Robust Remasking for Diffusion Language Models](./references/CORE-CONTEXT-ROBUST-REMASKING-FOR-DIFFUSION-LANGUAGE-MODELS/meta/meta_info.txt) - Zhai et al., 2026
- [Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models](./references/Dont-Settle-Too-Early-Self-Reflective-Remasking-for-Diffusion-Language-Models/meta/meta_info.txt) - Huang et al., 2025
- [TreeDiff: AST-Guided Code Generation with Diffusion LLMs](./references/TreeDiff-AST-Guided-Code-Generation-with-Diffusion-LLMs/meta/meta_info.txt) - Zeng et al., 2025
- [CodeDiffuSe: A masked diffusion framework for structure-aware code completion and repair](./references/CodeDiffuSe-A-masked-diffusion-framework-for-structure-aware-code-completion-and-repair/meta/meta_info.txt) - 2025
- [Diffusion is a code repair operator and generator](./references/Diffusion-is-a-code-repair-operator-and-generator/meta/meta_info.txt) - Singh et al., 2025
- [UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching](./references/UnMaskFork-Test-Time-Scaling-for-Masked-Diffusion-via-Deterministic-Action-Branching/meta/meta_info.txt) - Misaki & Akiba, 2026
- [Is Your Code Generated by ChatGPT Really Correct? (EvalPlus)](https://arxiv.org/abs/2305.01210) - Liu et al., 2023
- [ReMDM: Remasking Discrete Diffusion Models with Inference-Time Scaling](https://arxiv.org/abs/2503.00307) - Wang et al., 2025
- [Large Language Diffusion Models (LLaDA)](https://openreview.net/forum?id=KnqiC0znVF) - Nie et al., 2025
- [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487) - Ye et al., 2025
- [Simple and Effective Masked Diffusion Language Models (MDLM)](https://arxiv.org/abs/2406.07524) - Sahoo et al., 2024
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)](https://arxiv.org/abs/2310.16834) - Lou et al., 2024
- [Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)](https://arxiv.org/abs/2107.03006) - Austin et al., 2021
- [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) - Li et al., 2022
- [CodeFusion: A Pre-trained Diffusion Model for Code Generation](https://arxiv.org/abs/2310.17680) - Microsoft, 2023
- [DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639) - Apple, 2025
- [Corrective Diffusion Language Models](https://arxiv.org/abs/2512.15596) - Zhang et al., 2025
- [DiffTester: Accelerating Unit Test Generation for Diffusion LLMs](https://arxiv.org/abs/2509.24975) - Yang et al., 2025
- [Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](https://arxiv.org/abs/2602.01842) - Bai et al., 2026
- [Self-Rewarding Sequential Monte Carlo for Masked Diffusion Language Models](https://arxiv.org/abs/2602.01849) - Luo et al., 2026
- [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061) - Suresh et al., 2025
- [Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs](https://arxiv.org/abs/2602.00612) - Zhang et al., 2026
- [Execution Guided Line-by-Line Code Generation](https://arxiv.org/abs/2506.10948) - Lavon et al., 2025
- [CodeRL](https://arxiv.org/abs/2207.01780) - Le et al., 2022
- [Reflexion](https://arxiv.org/abs/2303.11366) - Shinn et al., 2023
- [CYCLE: Learning to Self-Refine the Code Generation](https://arxiv.org/abs/2403.18746) - 2024
- [TraceFixer: Execution Trace-Driven Program Repair](https://arxiv.org/abs/2304.12743) - Bouzenia et al., 2023
- [Type-Constrained Code Generation with Language Models](https://arxiv.org/abs/2504.09246) - Mundler et al., 2025
