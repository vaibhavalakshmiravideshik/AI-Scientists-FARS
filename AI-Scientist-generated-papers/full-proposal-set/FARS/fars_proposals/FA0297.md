# untitled

# CFG-Constrained Diffusion Decoding for Tool Calling: Separating Format vs Semantics on BFCL

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Setting**: Inference-only (no fine-tuning). Evaluate decoding variants of a diffusion language model on an existing tool-calling benchmark.
- **Automation constraint**: Fully automated evaluation using the BFCL official AST-based evaluator (no human judging).
- **Compute constraint**: <= 768 GPU-hours total; expected << 50 GPU-hours (inference only).

## Introduction

### Context and Motivation

Tool calling (function calling) is a core capability for language-model agents: the model must emit a structured action (a function name plus arguments) that can be parsed and executed reliably. Many agent failures in practice are not due to lack of intent, but due to interface errors: invalid syntax, missing required arguments, or wrong literal types that cause immediate execution failure.

Diffusion language models (dLLMs) are an alternative to autoregressive (AR) language models. Instead of generating tokens left-to-right, masked diffusion LMs iteratively fill and refine a token canvas by repeatedly predicting masked positions. This non-autoregressive decoding can offer parallelism and editing primitives, and recent open dLLMs (e.g., Llada and Dream) show competitive standalone accuracy on reasoning and code benchmarks.

However, recent evidence suggests that current dLLMs are unreliable as agent backbones in tool-calling settings. In particular, **The Bitter Lesson of Diffusion Language Models for Agentic Workflows** reports very low success rates for open dLLMs on the **Berkeley Function Calling Leaderboard (BFCL) v3**, a benchmark that tests whether a model can output correctly structured function calls (tool name + arguments) given a natural-language request and a tool schema. On their sampled BFCL-v3 test set (seed 42), Llada-8B achieves **23.0%** success on Non-Live single-turn tasks, compared to **87.5%** for an autoregressive baseline (Qwen-8B) (Bitter Lesson Table 2 in `./references/The-Bitter-Lesson-of-Diffusion-Language-Models-for-Agentic-Workflows-A-Comprehensive-Reality-Check/sections/dLLMs underperform LLMs on both single-turn and multi-turn tool-callings.md`; higher is better).

### The Problem

A key ambiguity in these BFCL failures is *what is actually broken*:

- If dLLMs mainly fail because their outputs are not parseable or violate the required structured-output format, then stronger decoding-time structure enforcement could close a large fraction of the gap.
- If dLLMs mostly fail despite being parseable (wrong tool choice, wrong arguments/values), then the bottleneck is semantic tool/argument reasoning rather than formatting.

For AR LMs, grammar- and schema-constrained decoding is a mature approach for structured generation (e.g., JSON, SQL, code). For diffusion LMs, constrained decoding has recently progressed rapidly:

- **DINGO** provides deterministic finite automaton (DFA) / regular-expression constrained decoding for diffusion LMs and achieves 100% JSON schema validity on JSON-Mode-Eval.
- **IG-CD** and **LAVE** provide context-free grammar (CFG) constrained decoding for diffusion LMs and reach near-perfect syntactic validity on code/JSON/SMILES benchmarks.

Despite this progress, it is unclear whether CFG-constrained diffusion decoding can improve *tool calling*, and more importantly whether it can disentangle formatting vs semantic failures on BFCL. Most existing BFCL-oriented methods focus on post-hoc repair (generate first, then fix invalid calls) rather than enforcing correctness during decoding.

### Key Insight and Hypothesis

**Key insight:** BFCL's primary offline evaluator, *AST substring matching*, parses the model output with Python's abstract syntax tree (AST) parser (`ast.parse`) and then checks exact function-name match and whether each argument value belongs to a predefined valid set. This makes BFCL a natural setting to test whether dLLM failures are dominated by parseability/format.

**Hypothesis:** On the BFCL-v3 Non-Live single-turn subset (300 examples), applying *syntax-only* CFG-constrained diffusion decoding (LAVE) will (i) sharply increase AST-parseability and (ii) improve BFCL success beyond a strong inference-scaling baseline that retries until parseable (best-of-2 with AST-parseability filtering).

Why this could be wrong:
- Unconstrained outputs may already be mostly parseable; best-of-2 parseability filtering could capture nearly all gains.
- Constrained decoding can distort the model distribution and could reduce semantic accuracy even if it improves parseability.
- Llada-8B's dominant failures might be missing parameters/values rather than parse errors, which syntax-only constraints do not fix.

---

## Proposed Approach

### Overview

We will evaluate whether diffusion CFG-constrained decoding improves tool-calling success on BFCL, and how much of any gain is attributable to eliminating syntax/format errors.

We focus on a fully offline slice: BFCL-v3 **Non-Live** single-turn subset (6 categories x 50 examples = 300), constructed using the sampling protocol in Bitter Lesson (50 per category; `./references/The-Bitter-Lesson-of-Diffusion-Language-Models-for-Agentic-Workflows-A-Comprehensive-Reality-Check/sections/Datasets.md`). This avoids any dependency on live API endpoints and uses BFCL's AST substring matcher.

**Important detail:** BFCL's Non-Live single-turn set includes an *irrelevance* category where the correct output is an empty list `[]` (no tool call). Our grammar and evaluation must support this.

We compare three decoding conditions for the same diffusion backbone (LLaDA-8B-Instruct; an 8B-parameter diffusion LM):

- (A) Unconstrained diffusion decoding (1 sample).
- (B) Best-of-2 with AST-parseability filtering (choose a parseable sample if exactly one parses).
- (C) Syntax-only CFG-constrained diffusion decoding using **LAVE**.

### Method Details

#### Target structured format (BFCL prompting mode)
BFCL's prompting-mode system prompt and output-format instruction require outputs to be a bracketed list of function-call expressions with no extra text:

`[funcname1(arg1=value1, ...), funcname2(...)]`

BFCL's AST substring matcher parses this output using Python `ast` and extracts callee names and keyword arguments.

#### Syntax-only tool-call CFG
We will implement a CFG that enforces only Python-callable structure required for AST parsing (no schema-conditioned restriction on identifiers):

- Output is a Python list literal of 0+ calls (allow `[]` for irrelevance).
- Callee is a dotted name: `IDENT ("." IDENT)*`.
- Arguments are keyword args only: `IDENT "=" literal` (comma-separated).
- Literals support common BFCL value shapes: string (single or double quotes), int/float, True/False/None, list/tuple, dict with literal keys/values (recursive).

This grammar is intentionally permissive about *which* function/kwarg identifiers appear (to avoid semantic leakage). It only guarantees parseability. We explicitly allow the empty list `[]` to support BFCL's irrelevance category where no tool call is expected.

#### Constrained diffusion decoding algorithm
We will use **LAVE (Lookahead-then-Verify)** for CFG-constrained decoding of masked diffusion LMs:

- When the diffusion LM proposes a token update at a masked position, LAVE samples N lookahead completions for the remaining masked positions using the model's parallel per-position token distributions.
- It accepts the proposal only if at least one lookahead completion is CFG-extendable (checked by a grammar parser), ensuring reliability for intermediate states that contain `[MASK]`.
- A cache-enhanced recovery mechanism handles repeated rejections.

Implementation path:
- Start from the open LAVE codebase (https://github.com/zhangyitonggg/CD4dLLM) which already supports LLaDA.
- Add a BFCL tool-call grammar (Lark format) and a BFCL prompt wrapper.
- If LAVE integration is blocked, use IG-CD's open implementation (https://github.com/eth-sri/constrained-diffusion) as a fallback constrained-diffusion CFG decoder.

**Grammar implementation note:** Because BFCL evaluation parses with Python `ast`, the grammar must be restricted to *exactly* the subset of Python expression syntax that `ast.parse` will accept under BFCL constraints (list literal of call expressions with keyword arguments and Python literals). This reduces the risk that the grammar accepts strings that BFCL cannot parse.

#### Best-of-2 parseability-filter baseline (no label leakage)
For (B), we sample two independent outputs for each instance and run Python `ast.parse`:

- If exactly one of the two parses, select the parseable one.
- Otherwise, select the first output.

This baseline controls for a common deployment workaround: retry until the output parses, without using ground-truth tool/args.

### Key Innovations

1. **Diagnostic evaluation of diffusion constrained decoding on tool calling:** Apply diffusion CFG-constrained decoding to BFCL to test whether tool-calling failures are mainly formatting vs semantic.
2. **Syntax-only constraint to avoid semantic leakage:** The grammar guarantees AST-parseability without restricting tool names or kwarg names to the provided schema.
3. **Strong inference-scaling control:** Best-of-2 parseability filtering isolates gains beyond "retry until parseable".

---

## Related Work

### Field Overview

**Tool calling and evaluation.** Function-calling benchmarks evaluate whether models can select the correct function and emit executable arguments. BFCL is a widely used benchmark that evaluates single-turn tool calling with an AST-based evaluator (offline) and with executable evaluation for some tasks. Related benchmarks include ToolBench and API-Bank, and model-side approaches include Toolformer-style self-training and prompt-based agent frameworks (e.g., ReAct).

**Diffusion language models for agentic workflows.** Masked diffusion LMs such as LLaDA and Dream use iterative denoising / remasking rather than next-token prediction. While these models can be competitive on some standalone tasks, Bitter Lesson reports systematic failures in agentic settings, including tool calling, often attributed to symbolic precision and formatting constraints.

**Constrained decoding for structured generation.** For AR LMs, constrained decoding via incremental parsing/masking (e.g., PICARD, Outlines, LMQL) is common for SQL/JSON/code. For diffusion LMs, recent methods extend constraints beyond regular languages (DINGO) to CFGs (IG-CD, LAVE). A recurring question is whether constrained decoding changes the effective conditional distribution and can harm semantic correctness (e.g., Grammar-Aligned Decoding discusses distribution distortion in AR settings).

### Related Papers

- **[BFCL: From Tool Use to Agentic Evaluation of LLMs](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/meta/meta_info.txt)**: Benchmark and evaluation suite for function calling; defines AST substring matching.
- **[The Bitter Lesson of Diffusion LMs for Agentic Workflows](./references/The-Bitter-Lesson-of-Diffusion-Language-Models-for-Agentic-Workflows-A-Comprehensive-Reality-Check/meta/meta_info.txt)**: Comprehensive evaluation showing dLLMs underperform on BFCL and other agentic benchmarks.
- **[LLaDA: Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)**: Open masked diffusion LM family (includes Llada-8B).
- **[Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)**: Diffusion LM adapted from autoregressive initialization; strong open baseline.
- **[DINGO: Constrained Inference for Diffusion LLMs](./references/DINGO-Constrained-Inference-for-Diffusion-LLMs/meta/meta_info.txt)**: Regular-language constrained decoding for diffusion (DFA/regex); achieves 100% JSON schema validity on JSON-Mode-Eval.
- **[Constrained Decoding of Diffusion LLMs with CFGs (IG-CD)](./references/Constrained-Decoding-of-Diffusion-LLMs-with-Context-Free-Grammars/meta/meta_info.txt)**: CFG-constrained diffusion decoding via feasibility checks.
- **[LAVE: Lookahead-then-Verify for CFG-constrained diffusion decoding](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/meta/meta_info.txt)**: Reliable CFG constraints for diffusion LMs using lookahead verification.
- **[SynCode](https://arxiv.org/abs/2403.01632)**: Grammar augmentation for structured generation (primarily AR).
- **[Outlines](https://arxiv.org/abs/2307.09702)**: Practical regex/CFG constrained decoding for LLMs.
- **[PICARD](https://arxiv.org/abs/2109.05093)**: Incremental parsing for constrained decoding in semantic parsing.
- **[LMQL](https://arxiv.org/abs/2212.06094)**: Query language for constrained LM programs.
- **[XGrammar](https://arxiv.org/abs/2411.17130)**: Efficient grammar-constrained decoding infrastructure.
- **[TRIDENT](https://arxiv.org/abs/2502.05111)**: DFA-enhanced constrained inference with length-aware constraints (AR-focused).
- **[Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047)**: Distribution distortion analysis for grammar masking (AR).
- **[Constrained Sampling Should Be Easy: An MCMC Perspective](https://arxiv.org/abs/2506.05754)**: MCMC view of constrained sampling for language models.
- **[Toolformer](https://arxiv.org/abs/2302.04761)**: Self-training LMs to use tools.
- **[Gorilla](https://arxiv.org/abs/2305.15334)**: Tool-use evaluation and training for large toolsets.
- **[ToolLLM](https://arxiv.org/abs/2307.16789)**: Training LMs for large-scale API calling.
- **[API-Bank](https://arxiv.org/abs/2304.08244)**: Benchmark for tool-augmented LLMs.
- **[ToolBench](https://arxiv.org/abs/2305.16504)**: Benchmark for tool-use with many APIs.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: Prompting framework combining reasoning and acting.
- **[D3PM](https://arxiv.org/abs/2107.03006)**: Discrete diffusion model foundations.
- **[MaskGIT](https://arxiv.org/abs/2202.04200)**: Iterative masked generation (non-autoregressive) approach.
- **[SEDD](https://arxiv.org/abs/2310.16834)**: Score-entropy discrete diffusion modeling.
- **[MDLM](https://arxiv.org/abs/2406.07524)**: Simple masked diffusion language models.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| AR constrained decoding | Incremental parsing masks invalid next tokens | PICARD; Outlines; LMQL; XGrammar | SQL/JSON/code generation | Assumes left-to-right prefixes |
| Diffusion constrained decoding | Enforce constraints on masked canvases via feasibility/extendability checks | IG-CD; LAVE; DINGO | JSON/CPP/SMILES structured generation | May distort semantics; grammar design burden |
| Post-hoc repair | Generate then repair invalid tool calls using validators/editors | ToolCritic; PALADIN; validator-guided remasking (prior proposal) | BFCL/ToolBench-style | Repair may be expensive; can destroy correct parts |
| Tool-use benchmarks | Standardized tool-call tasks + evaluation harness | BFCL; ToolBench; API-Bank | Tool-call success / exec | Often sensitive to format conventions |

### Closest Prior Work

- **Bitter Lesson (2601.12979):** Shows dLLMs (including Llada-8B) perform poorly on BFCL and notes that JSON-schema and parameter/value errors dominate their failures (Figure 4), but does not test whether constrained diffusion decoding eliminates a large fraction of these failures.
- **BFCL (ICML 2025):** Defines the AST substring matching evaluator and the required function-call output format; does not study diffusion LMs or constrained diffusion decoding.
- **LAVE (2602.00612):** Provides reliable CFG-constrained diffusion decoding and shows large gains on structured generation (JSON/code/SMILES), but does not evaluate tool-calling benchmarks like BFCL.
- **IG-CD (2508.10111):** Introduces CFG constraints for diffusion decoding; LAVE later improves reliability. Neither evaluates tool calling.
- **DINGO (2505.23061):** Enables regex/DFA constraints for diffusion; BFCL tool calls require CFG expressivity (nested literals, bracketed lists, keyword arguments), motivating CFG methods.

**Novelty Kill Search Summary:** Searched for the exact combination "BFCL" + "diffusion language model" + ("LAVE" OR "IG-CD" OR "constrained decoding" OR "CFG") and for "LLaDA BFCL constrained decoding" (web + local KB). Also checked existing proposals containing BFCL in the shared finalized set; only repair- or schema-augmentation-focused proposals were found, not CFG-constrained diffusion decoding on BFCL. No prior work reporting BFCL results for dLLMs under CFG-constrained decoding was found as of 2026-02-25 (full query log in notes.md).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Bitter Lesson (2026) | Evaluates dLLMs on BFCL; reports low success | Does not isolate formatting vs semantic failures via constraints | Apply CFG-constrained diffusion decoding | If failures are formatting-dominated, constraints should recover success |
| BFCL (2025) | Provides tool-calling benchmark + AST evaluator | Not a decoding method; no diffusion focus | Use BFCL as diagnostic harness | Clear offline metric and parseability signal |
| LAVE (2026) | Reliable CFG-constrained diffusion decoding | Not evaluated on tool calling | Apply LAVE to BFCL format grammar | BFCL output is a CFG; LAVE should guarantee parseability |
| IG-CD (2025/2026) | CFG constraints for diffusion via feasibility checks | Reliability issues vs LAVE; not on BFCL | Use as fallback implementation | Ensures feasibility if LAVE integration is blocked |
| DINGO (2025) | DFA/regex constraints for diffusion | Regular languages only; BFCL needs CFG | Motivate CFG methods | Highlights need for CFG-level constraints |

---

## Experiments

### Experimental Setup

**Benchmark and split.** Use BFCL-v3 Non-Live single-turn tasks, evaluated with BFCL's offline AST-substring matcher. Construct the test subset following Bitter Lesson's sampling protocol: sample 50 examples per category with random seed 42 (see `./references/The-Bitter-Lesson-of-Diffusion-Language-Models-for-Agentic-Workflows-A-Comprehensive-Reality-Check/sections/Datasets.md`). Restrict to the six Non-Live single-turn categories defined in BFCL (Appendix C.2): Simple, Multiple, Parallel, Parallel Multiple, Relevance, and Irrelevance, yielding 6x50=300 examples.

**Evaluation code.** Use the official BFCL evaluation suite (Gorilla BFCL):
- GitHub: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- Dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard

**Base model.** Llada-8B-Instruct (diffusion LM):
- HuggingFace: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
- If the constrained-decoding implementation expects a specific variant (e.g., Llada-1.5 or a particular tokenizer), use the checkpoint supported by that code and document it.

**Decoding settings (kept fixed across conditions).**
- Same prompt template (BFCL prompting-mode system prompt + output-format instruction).
- Same max output length and diffusion denoising steps `T` as in the chosen LLaDA implementation.
- Same temperature / sampling parameters for unconstrained sampling.

**Main conditions (3):**
1. **(A) Unconstrained** diffusion decoding.
2. **(B) Best-of-2 + AST-parseability filter** (selection rule described above).
3. **(C) LAVE syntax-only CFG constraint** using the BFCL tool-call grammar.

**Baseline Ladder (REQUIRED):**
- Prompting baseline: (A) uses BFCL prompting-mode system prompt.
- Inference-time scaling baseline: (B) best-of-2 parseability filtering.
- Closest existing method: (C) LAVE CFG-constrained diffusion decoding.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| LLaDA-8B-Instruct | 8B | https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct | Diffusion LM; backbone for all three decoding conditions |

**Training Data (if applicable):**

No training data needed - inference only.

**Other Resources (if applicable):**
- LAVE implementation: https://github.com/zhangyitonggg/CD4dLLM
- IG-CD fallback implementation: https://github.com/eth-sri/constrained-diffusion

**Resource Estimate**:
- **Compute budget**: ~10 GPU-hours total (single A100-class GPU), dominated by diffusion inference over 300 instances x 3 seeds x (1 + 2 + 1) samples. As a rough anchor, LAVE reports ~8-10 seconds per sample for LLaDA-8B on JSON/CPP benchmarks (Table 3 in LAVE; lower is faster), implying <10 GPU-hours for this BFCL slice even with best-of-2.
- **GPU memory**: fits on a single A100 80GB (no tensor parallelism required).
- **API usage**: none.

**Seeds / variance.** Run all three conditions with `seeds=[42, 123, 456]` and report mean +/- std.

**Sanity check (published anchor).** Reproduce Bitter Lesson's published Llada-8B Non-Live average (23.0%) within a small tolerance (e.g., +/-5 pp). If reproduction fails, adjust only prompt formatting / decoding hyperparameters to match the published setup, then freeze settings.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| BFCL-v3 Non-Live (300) | Single-turn tool calling with offline AST substring matching | Success rate (%), ast_parse_rate (%), failure taxonomy counts | Bitter Lesson seed-42 subset | HF dataset link above | Gorilla BFCL repo |

Metric definitions:
- **Success rate**: BFCL AST substring matcher success (higher is better).
- **ast_parse_rate**: fraction of outputs that parse with Python `ast.parse` (higher is better).
- **Failure taxonomy** (analysis): parse fail; parses but wrong function name(s); correct function name(s) but wrong/missing kwargs or values.

### Main Results

Published anchor numbers (Non-Live single-turn average; 1 run): Bitter Lesson Table 2.

#### Results Table

| Method | Base Model | Benchmark | Success rate (mean+/-std) | ast_parse_rate (mean+/-std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Published AR baseline | Qwen-8B | BFCL-v3 Non-Live (300) | 87.5 | N/A | [Bitter Lesson Table 2](./references/The-Bitter-Lesson-of-Diffusion-Language-Models-for-Agentic-Workflows-A-Comprehensive-Reality-Check/sections/dLLMs%20underperform%20LLMs%20on%20both%20single-turn%20and%20multi-turn%20tool-callings.md) | Published (1 run); Non-Live average over the sampled BFCL-v3 subset |
| Published diffusion baseline | Llada-8B | BFCL-v3 Non-Live (300) | 23.0 | N/A | [Bitter Lesson Table 2](./references/The-Bitter-Lesson-of-Diffusion-Language-Models-for-Agentic-Workflows-A-Comprehensive-Reality-Check/sections/dLLMs%20underperform%20LLMs%20on%20both%20single-turn%20and%20multi-turn%20tool-callings.md) | Published (1 run); Non-Live average over the sampled BFCL-v3 subset |
| (A) Unconstrained | Llada-8B | BFCL-v3 Non-Live (300) | TBD | TBD | - | 3 seeds |
| (B) Best-of-2 + AST filter | Llada-8B | BFCL-v3 Non-Live (300) | TBD | TBD | - | 3 seeds; selection uses parseability only |
| (C) LAVE syntax-only CFG | Llada-8B | BFCL-v3 Non-Live (300) | TBD | TBD | - | 3 seeds; grammar enforces parseability only |

### Ablation Studies

No additional ablations are required to decide the core hypothesis. If the outcome is inconclusive (parseability improves but success does not), a follow-up experiment can test a **schema-conditioned** CFG (restrict callee/kwarg identifiers to the provided tool schema) to measure how much of the remaining gap is semantic.

### Experimental Rigor

**Confounders and controls:**
- Retry confound: (B) controls for "retry until parseable" without using labels.
- Budget mismatch: report average wall-clock time per instance for all three conditions; ensure (B) uses exactly 2 samples and (C) uses 1 constrained sample.
- Grammar mismatch: validate that outputs from (C) parse under Python `ast` at very high rate; otherwise the grammar is incorrect and the experiment is invalid.

**Data leakage:** BFCL may appear in pretraining data for some models. The primary claim is a *within-model* comparison across decoding strategies under the same prompts/budgets, reducing sensitivity to memorization.

---

## Success Criteria

**Hypothesis (directional):** Syntax-only CFG constraints will substantially reduce parse failures and increase BFCL success beyond best-of-2 parseability filtering.

**Decision Rule:**
- **Proceed / support "mostly formatting"** if (C) reduces parse failures by >= 50% relative to (A) and improves Non-Live success over (B) by >= 5.0 absolute points (mean over 3 seeds).
- **Refute "mostly formatting"** if (C) reduces parse failures by >= 50% but improves success by < 2.0 points vs (B).
- **Pivot** if (C) does not materially reduce parse failures (grammar/spec mismatch); fix grammar or switch to IG-CD implementation.

---

## Impact Statement

If successful, this study suggests that diffusion LMs can recover a significant fraction of tool-calling performance by enforcing syntactic structure at decode time, making constrained diffusion decoding a practical requirement for deploying dLLMs in tool-using systems. If it fails (parseability improves but success does not), the negative result is still decision-changing: it implies that dLLM tool-calling deficits on BFCL are primarily semantic (tool/argument reasoning), and future work should focus on semantic constraints or training rather than formatting fixes.

---

## References

- [BFCL: From Tool Use to Agentic Evaluation of LLMs](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/meta/meta_info.txt) - Patil et al., 2025
- [The Bitter Lesson of Diffusion Language Models for Agentic Workflows](./references/The-Bitter-Lesson-of-Diffusion-Language-Models-for-Agentic-Workflows-A-Comprehensive-Reality-Check/meta/meta_info.txt) - 2026
- [Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under CFGs (LAVE)](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/meta/meta_info.txt) - Zhang et al., 2026
- [Constrained Decoding of Diffusion LLMs with Context-Free Grammars (IG-CD)](./references/Constrained-Decoding-of-Diffusion-LLMs-with-Context-Free-Grammars/meta/meta_info.txt) - Muendler et al., 2025 (ICLR 2026)
- [DINGO: Constrained Inference for Diffusion LLMs](./references/DINGO-Constrained-Inference-for-Diffusion-LLMs/meta/meta_info.txt) - Suresh et al., 2025
- [Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992) - Nie et al., 2025
- [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487) - Ye et al., 2025
- [Outlines](https://arxiv.org/abs/2307.09702) - Willard and Louf, 2023
- [PICARD](https://arxiv.org/abs/2109.05093) - Scholak et al., 2021
- [LMQL](https://arxiv.org/abs/2212.06094) - Beurer-Kellner et al., 2022
- [SynCode](https://arxiv.org/abs/2403.01632) - Ugare et al., 2024
- [XGrammar](https://arxiv.org/abs/2411.17130) - Dong et al., 2024
- [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) - Park et al., 2024
- [Constrained Sampling Should Be Easy: An MCMC Perspective](https://arxiv.org/abs/2506.05754) - Anaya Gonzalez et al., 2025
- [Toolformer](https://arxiv.org/abs/2302.04761) - Schick et al., 2023
- [Gorilla](https://arxiv.org/abs/2305.15334) - Patil et al., 2023
- [ToolLLM](https://arxiv.org/abs/2307.16789) - Qin et al., 2023
- [API-Bank](https://arxiv.org/abs/2304.08244) - Li et al., 2023
- [ToolBench](https://arxiv.org/abs/2305.16504) - Qin et al., 2023
- [ReAct](https://arxiv.org/abs/2210.03629) - Yao et al., 2022
- [D3PM](https://arxiv.org/abs/2107.03006) - Austin et al., 2021
- [MaskGIT](https://arxiv.org/abs/2202.04200) - Chang et al., 2022
- [SEDD](https://arxiv.org/abs/2310.16834) - Lou et al., 2023
- [MDLM](https://arxiv.org/abs/2406.07524) - Sahoo et al., 2024
