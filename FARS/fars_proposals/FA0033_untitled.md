# untitled

# Interface-Aware Smoke Tests and Deterministic Import Autofix for Feature-Level Coding Agents

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

LLM-based coding agents (e.g., SWE-agent and OpenHands) increasingly solve repository-level tasks by iteratively reading code, editing files, and executing tests. However, most widely used benchmarks still emphasize bug fixing, where patches are relatively small.

**FeatureBench** evaluates a harder setting: feature-level development tasks derived from real Python repositories, with execution-based evaluation using fail-to-pass (F2P) and pass-to-pass (P2P) tests. Despite frontier agent scaffolds and strong models, FeatureBench reports very low solved rates alongside extremely high token usage (often millions of input tokens per task), suggesting that today’s agents are inefficient at long-horizon feature development.

### The Problem

FeatureBench’s failure analysis highlights frequent **NameError** and **AttributeError/TypeError** failures, attributed to cross-file dependency mistakes and “idle habits” where the agent guesses interfaces instead of reliably grounding them in code (see the analysis in **[FeatureBench](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt)**).

In Python, **NameError** and **ImportError** commonly indicate that a required symbol is undefined or not imported, while **AttributeError** often indicates an incorrect module/object interface (an expected attribute or function is missing).

A key observation is that many runtime failures in Python code editing workflows are not “deep reasoning” failures; they are often **mechanical symbol-resolution failures** (missing imports, wrong import paths, misspelled/undefined names, or calling a function that does not exist). In interactive agent loops, these failures can create long error-correction cycles: the agent runs a large test suite, sees a traceback, searches for a definition, edits imports, and repeats.

Existing scaffolds partially address this:
- **SWE-agent** integrates interface design and lightweight linting to reduce syntax errors and improve iteration efficiency (**[SWE-agent](./references/SWE-agent-Agent-Computer-Interfaces-Enable-Automated-Software-Engineering/meta/meta_info.txt)**).
- Static-analysis and detection work such as **REDO** improves execution-free runtime error detection via Pyright/Pyflakes + an LLM (**[REDO](./references/REDO-Execution-Free-Runtime-Error-Detection-for-COding-Agents/meta/meta_info.txt)**).

But current systems generally still rely on the LLM itself to perform the repetitive “search the repo and insert the obvious import” step, even when the fix is unambiguous. This is costly in tokens and agent steps, and may reduce success under a fixed step budget.

### Key Insight and Hypothesis

We hypothesize that in FeatureBench, a non-trivial fraction of agent failures and tokens are spent on **unambiguously resolvable symbol errors** (especially missing imports / undefined names) that can be fixed deterministically using repository search and the task’s provided interface specification. If we add an **interface-aware smoke test** plus a **deterministic import-autofix step** to the agent loop, we will reduce NameError/ImportError/AttributeError loops and thereby:

- increase the solved rate under a fixed step budget, and/or
- reduce input tokens at similar solved rate.

This hypothesis could be wrong if (i) most failures are due to deeper semantic mistakes rather than symbol resolution, (ii) most missing-symbol errors are ambiguous (multiple plausible definitions) and cannot be safely auto-fixed, or (iii) the agent’s token usage is dominated by other factors (e.g., large-scale code reading) so import autofix has limited impact.

---

## Proposed Approach

### Overview

We propose a small, deterministic augmentation to an existing coding-agent scaffold (e.g., OpenHands on FeatureBench): after each code edit, automatically run a very fast **interface smoke test** derived from the task’s provided invocation path and type-annotated signature. If the smoke test fails with a small class of common runtime errors (NameError/ImportError/AttributeError), run a deterministic resolver that attempts to automatically insert missing imports **only when the resolution is unambiguous**.

We evaluate three conditions to isolate the mechanism:
1. **Baseline**: the original agent scaffold.
2. **Diagnose-only**: the smoke test + diagnostics are provided to the agent, but no code is modified automatically.
3. **Diagnose + deterministic autofix (ours)**: same diagnostics as (2), plus automatic import insertion for unambiguous cases.

### Method Details

**Input signals available in FeatureBench prompts.** Each FeatureBench instance provides an explicit interface description including an import path and a callable signature (with types). This provides a stable target for quickly checking “is the codebase in a minimally runnable state?” without repeatedly running the full pytest suite.

**Component 1: Interface smoke test (fast check).**
- Parse the interface description (module import path, callable name, signature, type annotations).
- Generate a minimal script `smoke_test.py` that:
  - imports the target module and target callable,
  - performs attribute access for the callable,
  - optionally executes a single “dummy call” when all arguments can be instantiated from primitive/container types (e.g., `int`, `float`, `str`, `bool`, `list[...]`, `dict[...]`).
- Run `python smoke_test.py`.

**Component 2: Error parsing.**
- If the smoke test fails with NameError/ImportError/AttributeError, extract the missing symbol name (and, when available, the file/location from the traceback).

**Component 3: Deterministic resolver with safe autofix.**
- Search the repository for candidate definitions of the missing symbol `S` (e.g., regex for `def S(` or `class S(`).
- If exactly **one** candidate definition site is found, and the target file can safely accept an import insertion, apply the deterministic edit:
  - insert `from <module> import S` (or `import <module> as ...` for modules) near the top of the file.
- If there are multiple candidates, do not modify code; instead provide the top-k candidates (file paths + signatures) to the agent.

**Diagnose-only vs autofix.**
- Diagnose-only returns the same structured report but never changes code.
- Autofix additionally applies the deterministic import insertion when the fix is unambiguous.

**Output format to the agent (for both conditions 2 and 3).** A compact JSON report injected into the agent’s observation stream:
```json
{
  "smoketest_status": "pass" | "fail",
  "error_type": "NameError" | "ImportError" | "AttributeError" | "other",
  "applied_fixes": [ {"file": "...", "edit": "..."} ],
  "remaining_undefined": [ {"symbol": "S", "candidates": ["path:line", "..."]} ]
}
```
Condition (2) always has `applied_fixes=[]`.

### Key Innovations

1. **Interface-aware verification at low cost**: use the benchmark-provided interface spec to define a smoke test that is much cheaper than running pytest repeatedly.
2. **Deterministic automation for a high-frequency error class**: automatically resolve *unambiguous* missing imports/undefined symbols without LLM reasoning.
3. **Mechanism-isolating evaluation**: a mandatory diagnose-only ablation distinguishes “better information” from “automation of a repetitive action.”

---

## Related Work

### Field Overview

Software engineering agents typically combine (i) repository navigation and localization (search, graphs, retrieval), (ii) code editing actions, and (iii) execution feedback from tests or linters. The dominant performance drivers in SWE-bench-style settings include strong scaffolds (tool interfaces), localization quality, and test-time compute scaling (multiple candidates, search over trajectories).

FeatureBench shifts attention from bug fixing to feature development, where code changes span more files and more functions, and where agents may spend substantial budget simply reaching an executable intermediate state.

Our proposal focuses on a narrow but practically important axis: reducing repetitive runtime-error loops via deterministic symbol-resolution automation, rather than proposing new model training or large changes to agent planning.

### Related Papers

- **[FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt)**: Introduces FeatureBench and reports NameError-dominant failure modes plus million-token agent runs.
- **[ContextBench: A Benchmark for Context Retrieval in Coding Agents](./references/CONTEXTBENCH-A-Benchmark-for-Context-Retrieval-in-Coding-Agents/meta/meta_info.txt)**: Evaluates context-retrieval processes and highlights gaps between retrieved vs utilized context.
- **[SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](./references/SWE-agent-Agent-Computer-Interfaces-Enable-Automated-Software-Engineering/meta/meta_info.txt)**: Shows that better tool interfaces (view/edit/search + linting) substantially improve SWE-bench performance.
- **[The OpenHands Software Agent SDK](https://arxiv.org/abs/2511.03690)**: Provides a production-oriented agent SDK including context condensation and tool abstractions relevant to integrating new guardrail tools.
- **[Agentless: Demystifying LLM-based Software Engineering Agents](https://arxiv.org/abs/2407.01489)**: Uses a structured, largely non-agentic pipeline for localization/repair/validation, illustrating how simpler workflows can outperform complex agents.
- **[AutoCodeRover: Autonomous Program Improvement](./references/AutoCodeRover-Autonomous-Program-Improvement/meta/meta_info.txt)**: Uses structured AST search APIs and optional SBFL signals for SWE-bench repair.
- **[CodeR: Issue Resolving with Multi-Agent and Task Graphs](https://arxiv.org/abs/2406.01304)**: Multi-agent workflow with SBFL + BM25 localization, highlighting execution feedback loops in repair.
- **[MASAI: Modular Architecture for Software-engineering AI Agents](https://arxiv.org/abs/2406.11638)**: Decomposes SWE-bench solving into multiple agents (reproducer/localizer/fixer/ranker).
- **[SWE-Search](https://arxiv.org/abs/2410.20285)**: Applies MCTS-style search over agent trajectories for SWE-bench.
- **[SWE-Exp](https://arxiv.org/abs/2507.23361)**: Uses an experience bank to guide SWE-bench repair, emphasizing reusable patterns for error recovery.
- **[DARS](https://arxiv.org/abs/2503.14269)**: Improves SWE-bench via selective trajectory branching and patch selection.
- **[Live-SWE-agent](./references/Live-SWE-agent-Can-Software-Engineering-Agents-Self-Evolve-on-the-Fly/meta/meta_info.txt)**: Lets agents synthesize tools on-the-fly; our work instead adds a fixed deterministic tool focused on a specific error class.
- **[REDO](./references/REDO-Execution-Free-Runtime-Error-Detection-for-COding-Agents/meta/meta_info.txt)**: Combines static analysis (Pyright/Pyflakes) and LLM reasoning for execution-free runtime error detection.
- **[Agentic Program Repair from Test Failures at Scale](https://arxiv.org/abs/2507.18755)**: Production agentic repair system using static analysis + test execution feedback.
- **[PatchPilot](https://arxiv.org/abs/2502.02747)**: Rule-based workflow emphasizing cost efficiency and refinement for SWE-bench.
- **[CodeMonkeys](https://arxiv.org/abs/2501.14723)**: Scales test-time compute via parallel candidate edits and test-based selection.
- **[RepoNavigator / One Tool Is Enough](https://arxiv.org/abs/2512.20957)**: Uses a single “jump” tool (Pyright-based symbol resolution) trained with RL for repository localization.
- **[LocAgent](./references/LocAgent-Graph-Guided-LLM-Agents-for-Code-Localization/meta/meta_info.txt)**: Uses heterogeneous code graphs and graph traversal for localization.
- **[TraceFixer](https://arxiv.org/abs/2304.12743)**: Uses execution traces to guide program repair, motivating trace-driven debugging signals.
- **[SpecRover](https://arxiv.org/abs/2408.02232)**: Extracts code intent/specifications for improved repair, highlighting interface-level reasoning.
- **[SuperCoder2.0](https://arxiv.org/abs/2409.11190)**: Uses hierarchical search-space reduction and repository maps for SWE-bench.
- **[InterCode](https://arxiv.org/abs/2306.14898)**: Benchmarks interactive coding with execution feedback, motivating iterative correction loops.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: Introduces reasoning-action interleaving that underpins many coding agents.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Uses self-reflection memories to improve iterative agent performance.
- **[Chain of Targeted Verification Questions](https://arxiv.org/abs/2405.13932)**: Improves code reliability by asking targeted verification questions, closely related to our “cheap verification then fix” framing.
- **[Dissecting the SWE-Bench Leaderboards](https://arxiv.org/abs/2506.17208)**: Analyzes what drives progress on SWE-bench and common failure modes like overfitting and evaluation confounds.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Agent scaffolds / interfaces | Better tools (view/edit/search/lint) improve iterative coding | SWE-agent | SWE-bench | Still relies on LLM for many mechanical edits |
| Workflow-based / agentless repair | Decompose into localization → patch → validation | Agentless, PatchPilot, MASAI, CodeR | SWE-bench | Primarily bug-fixing oriented; not feature development |
| Search / compute scaling | Explore multiple trajectories/candidates and select | SWE-Search, DARS, CodeMonkeys | SWE-bench | Higher cost; may not address mechanical error loops |
| Localization via structure | Graphs, symbol resolution, SBFL | RepoNavigator, LocAgent, AutoCodeRover | SWE-bench, localization metrics | Focus is “where to edit,” not “remove mechanical errors after edits” |
| Runtime error detection | Static analysis + LLM to detect likely runtime errors | REDO | SWEDE/STA, SWE-bench-lite patches | Detects but does not deterministically repair |
| Feature-level evaluation | Harder tasks with explicit interfaces and test-driven extraction | FeatureBench, ContextBench | FeatureBench | Low solved rates; very high token usage |

### Closest Prior Work

**SWE-agent** integrates linting after edits and designs LM-friendly file tools, demonstrating that deterministic interface improvements can yield large gains. Our work targets a different failure mode: symbol-resolution/runtime import errors rather than syntax/lint errors, and uses task-provided interface specs to define a targeted smoke test.

**REDO** detects runtime errors execution-free using static analyzers and an LLM, but it does not automatically repair code. Our work tests whether deterministic repair of unambiguous missing imports yields measurable performance or efficiency gains in an interactive agent loop.

**RepoNavigator (One Tool Is Enough)** provides a strong symbol-resolution tool (“jump”) for localization, mainly to identify where definitions live. Our work uses symbol resolution to automatically insert missing imports and reduce debugging loops, without requiring RL training.

**Agentless / workflow-based systems** often reduce agent loops by constraining the workflow. Our work is complementary: it keeps a standard agent scaffold but makes one high-frequency action deterministic.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| SWE-agent | Better ACI + linting for bug fixing | No deterministic fix for missing-import/runtime symbol errors | Add interface-aware smoketest + import autofix | Reduces a common mechanical error loop in Python repos |
| REDO | Detects runtime errors execution-free | Detection only (no repair) | Repair unambiguous errors deterministically | Removes LLM turns spent on repetitive import insertion |
| RepoNavigator | RL-trained symbol “jump” tool for localization | Not focused on runtime error recovery; requires RL training | Use simple repo search for symbol resolution + repair | Cheaper to adopt; directly targets debugging loop cost |
| Agentless | Structured pipeline reduces agent wandering | Not designed for feature development; may still need symbol fixes | Keep agent scaffold; automate the easy fixes | Minimal change to existing agents; compatible with FeatureBench |
| FeatureBench | Benchmark + analysis | No method to address NameError/idle habits | Provide a concrete, testable fix for a reported failure mode | Directly targets FeatureBench’s failure analysis |

---

## Experiments

### Experimental Setup

**Primary benchmark**: FeatureBench Lite split (30 tasks), Level 1 (incremental development with partial codebase).

**Agent scaffold**: OpenHands (as used in the FeatureBench paper).

**Conditions (mandatory, ≤3 main conditions):**
- **A. Baseline**: OpenHands standard configuration.
- **B. Diagnose-only**: OpenHands + automatic interface smoke-test report after each edit (no automatic code changes).
- **C. Ours (diagnose + autofix)**: Same as B, but deterministically insert missing imports for unambiguous NameError/ImportError/AttributeError cases.

**How the tool is triggered (to avoid confounds):**
- The smoke test runs automatically after each successful code edit / patch application, similar in spirit to SWE-agent’s integrated linting.
- B and C inject the same diagnostic report format; C additionally applies deterministic edits and reports them.

**Early-kill pilot (to avoid wasting compute):**
- Run Baseline A on 10 tasks for up to 50 steps (temperature=0) and log (i) every crash type and (ii) token/step spent between crash and recovery.
- For each NameError/ImportError/AttributeError crash, run the deterministic resolver in “dry-run” mode and mark whether it is **unambiguously fixable** (exactly one definition site and a safe import insertion point).
- Also report what fraction of total steps/tokens in Baseline A occur inside these crash→recovery loops.
- **Stop rule**: If unambiguously-fixable crashes occur in <10% of tasks OR <10% of crash events (within 50 steps), OR if crash→recovery loops account for <10% of steps/tokens, stop the study and report a negative result (the addressable error-loop overhead is too small on FeatureBench under this scaffold).

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen3-Coder-480B-A35B-Instruct | MoE (480B total / 35B active) | (API) `Qwen/Qwen3-Coder-480B-A35B-Instruct` | Used in FeatureBench baselines; choose temp=0 for stability |

**Training Data (if applicable):**

No training data needed - inference only.

**Other Resources (if applicable):**
- FeatureBench code + harness: https://github.com/LiberCoders/FeatureBench
- FeatureBench dataset: https://huggingface.co/datasets/LiberCoders/FeatureBench

**Resource Estimate**:
- **Compute budget**: negligible GPU-hours if using API models; CPU + Docker execution for pytest/smoke tests.
- **API usage** (order-of-magnitude): baseline Table 2 reports ~2.6M input tokens per task for OpenHands+Qwen3-Coder on Lite; for 30 tasks and 3 conditions this could reach ~200M+ input tokens if run at the full 500-step budget.
  - Mitigation: keep max_steps at 100–200 for verification runs if needed; FeatureBench reports diminishing returns beyond ~100 steps.
- **Wall-clock**: dominated by LLM latency and dockerized test execution; FeatureBench harness supports concurrency (e.g., 4 workers).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| FeatureBench | Feature-level software development in real Python repos with F2P/P2P tests | Resolved rate, Passed rate, Token I/O; plus error-loop diagnostics | lite (L1) | https://huggingface.co/datasets/LiberCoders/FeatureBench | https://github.com/LiberCoders/FeatureBench (`fb infer`, `fb eval`) |

**Additional diagnostics (reported by our wrapper):**
- Crash-type distribution over time (NameError/ImportError/AttributeError/AssertionError/etc.)
- Fraction of crashes that are unambiguously fixable
- Number of LLM turns between crash detection and recovery
- Wall-clock time per task (secondary)

### Main Results

#### Results Table

| Method | Base Model | Benchmark | % Resolved | % Passed | Source | Notes |
|--------|------------|-----------|------------|----------|--------|-------|
| OpenHands (published) | Qwen3-Coder-480B-A35B-Instruct | FeatureBench Lite | 6.7 | 38.31 | [FeatureBench](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt) | Reported at 500 steps; token I/O 2.6M / 16k |
| OpenHands (A) | Qwen3-Coder-480B-A35B-Instruct | FeatureBench Lite | **TBD** | **TBD** | - | Verification run (temp=0; step budget specified) |
| Diagnose-only (B) | Qwen3-Coder-480B-A35B-Instruct | FeatureBench Lite | **TBD** | **TBD** | - | Same diagnostics as C, but no automatic edits |
| **Ours: diagnose+autofix (C)** | Qwen3-Coder-480B-A35B-Instruct | FeatureBench Lite | **TBD** | **TBD** | - | Deterministic import insertion for unambiguous cases |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| A → B | Add smoke-test diagnostics only | If diagnostics are helpful, B > A and crash recovery is faster |
| B → C | Add deterministic autofix | If automation matters beyond information, C > B and NameError/ImportError loops shrink |

### Analysis (Optional)

- **When does autofix help?** Stratify tasks by whether they exhibit unambiguously-fixable crashes in the first N steps; effects should concentrate on that subset.
- **Failure modes of autofix**: measure how often auto-inserted imports introduce new errors (should be rare by design).

---

## Success Criteria

**Criterion 1: Mechanism (loop reduction)**
- Hypothesis: Condition C reduces the frequency of NameError/ImportError/AttributeError crashes and reduces the number of LLM turns spent after such crashes.
- Validation: ≥30% relative reduction in these crash events vs A, and C < B < A in “turns-to-recover” when these crashes happen.

**Criterion 2: End performance or efficiency**
- Hypothesis: With the same step budget, C improves FeatureBench outcomes.
- Validation: Either (i) C solves at least 2 more tasks than A on Lite (30 tasks), or (ii) C reduces median input tokens by ≥25% while losing ≤1 solved task.

**Criterion 3: Negative result is still decision-changing**
- Hypothesis: If C ≈ B, then deterministic autofix is unnecessary once diagnostics are surfaced.
- Validation: Report C vs B gap; if small, recommend focusing future tooling on information the agent cannot cheaply obtain (not on automating trivial edits).

---

## Impact Statement

If successful, this work provides a low-effort, training-free way to improve feature-level coding agents by making a common class of Python runtime fixes deterministic. This could reduce the cost of running agents on real repositories (fewer tokens and fewer debugging turns) and improve solved rates under practical step budgets.

---

## References

- [FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt) - Zhou et al., 2026
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](./references/SWE-agent-Agent-Computer-Interfaces-Enable-Automated-Software-Engineering/meta/meta_info.txt) - Yang et al., 2024
- [AutoCodeRover: Autonomous Program Improvement](./references/AutoCodeRover-Autonomous-Program-Improvement/meta/meta_info.txt) - Zhang et al., 2024
- [REDO: Execution-Free Runtime Error Detection for COding Agents](./references/REDO-Execution-Free-Runtime-Error-Detection-for-COding-Agents/meta/meta_info.txt) - Li et al., 2024
- [Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?](./references/Live-SWE-agent-Can-Software-Engineering-Agents-Self-Evolve-on-the-Fly/meta/meta_info.txt) - Xia et al., 2025
- [ContextBench: A Benchmark for Context Retrieval in Coding Agents](./references/CONTEXTBENCH-A-Benchmark-for-Context-Retrieval-in-Coding-Agents/meta/meta_info.txt) - Li et al., 2026
- [LocAgent: Graph-Guided LLM Agents for Code Localization](./references/LocAgent-Graph-Guided-LLM-Agents-for-Code-Localization/meta/meta_info.txt) - arXiv:2503.09089
- [Agentless: Demystifying LLM-based Software Engineering Agents](https://arxiv.org/abs/2407.01489) - Xia et al., 2024
- [One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents](https://arxiv.org/abs/2512.20957) - Zhang et al., 2025
- [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback](https://arxiv.org/abs/2306.14898) - Yang et al., 2023
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - Yao et al., 2023
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - Shinn et al., 2023
