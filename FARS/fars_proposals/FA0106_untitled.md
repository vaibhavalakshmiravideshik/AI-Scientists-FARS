# untitled

# Trace-Bounded Context for Token-Efficient FeatureBench Agents

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

LLM-based coding agents (e.g., SWE-agent and OpenHands) solve repository-level tasks by iteratively reading files, editing code, and running tests. A persistent practical bottleneck is **context cost**: the agent must repeatedly ingest large amounts of repository text (file contents, stack traces, test logs), which can reach millions of input tokens per task.

**FeatureBench** is a recent benchmark that stresses this regime by evaluating **feature-level development** tasks in real Python repositories with execution-based scoring via fail-to-pass (F2P) and pass-to-pass (P2P) tests (**[FeatureBench](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt)**). The benchmark reports extremely high **Token I/O** (total input and output tokens consumed by the agent over a full task trajectory). For example, on the FeatureBench **Lite** split (30 tasks for faster evaluation), OpenHands + Claude Opus 4.5 uses 8.8M input tokens and 29k output tokens per task while solving only 20% of tasks (Table 2 in **[FeatureBench](<./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/sections/BENCHMARK COLLECTION.md>)**). This combination of low resolved rate and multi-million-token contexts suggests that improving **token efficiency without reducing success** is a decision-relevant objective.

### The Problem

FeatureBench’s failure analysis highlights that many failures are **cross-file dependency errors** (NameError dominates) and “idle habits” where models guess interfaces instead of reliably reading the required definitions (**[FeatureBench](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt)**, Failure Cases Analysis). This points to a tension:

- Agents must read enough code to avoid missing dependencies.
- But unstructured exploration often leads to reading large parts of the repository that are irrelevant to the failing feature path, inflating input tokens.

Existing approaches reduce context cost by (i) pruning retrieved code text (e.g., line-level pruning) or (ii) improving static retrieval/localization with repository graphs. However, these methods are typically **global/static**: they do not exploit a cheap, instance-specific signal that already exists in FeatureBench tasks.

In FeatureBench, each task comes with a failing test file list (`FAIL_TO_PASS`). Running the failing tests on the undeveloped repository provides a naturally instance-specific relevance signal: the **dynamic execution trace** of the failing path. This motivates a simple question:

> Can we use the failing-test execution trace to define a conservative, instance-specific set of “readable” files, and thereby reduce token usage without harming solve rate?

### Key Insight and Hypothesis

**Key insight.** A large fraction of repository text that agents read is not causally relevant to the failing tests. An execution trace of `pytest -x FAIL_TO_PASS` provides an instance-specific approximation of the minimal code neighborhood that the failing path touches. If we restrict file reads to this neighborhood (plus a shallow static import closure to avoid missing unexecuted dependencies), we can prevent irrelevant file reads without giving the agent any extra hints.

**Hypothesis.** On FeatureBench Lite tasks, enforcing **trace-bounded file access** will reduce **median input tokens** by at least 30% while preserving the number of fully solved tasks under deterministic decoding.

This could fail if (i) the failing-test trace misses files that must be edited even though they are not executed before the first failure, (ii) the trace neighborhood is too large (approaching most of the repo), leaving no efficiency headroom, or (iii) agents rely on exploratory reads outside the failing path to form correct hypotheses.

---

## Proposed Approach

### Overview

We propose **TraceBound**, an inference-time wrapper for FeatureBench agents that restricts file reads to an automatically computed per-instance allowlist derived from the failing-test execution trace.

TraceBound has two stages per task:

1. **Pre-trace (before the agent runs):** execute the task’s `FAIL_TO_PASS` tests with `pytest -x` on the undeveloped repository and record which repository files are executed.
2. **Bounded context policy (during the agent run):** expose a file-system view (or file-reading tool) that allows reading only files in the allowlist. Any attempt to read outside the allowlist fails with a plain `FileNotFoundError` (no curated hints).

### Method Details

**A. Computing the allowlist (per task).**

Inputs (from FeatureBench dataset item):
- `FAIL_TO_PASS`: list of failing test files.
- `problem_statement`: contains one or more interface descriptions with explicit `Path: `...`` markers (see FeatureBench prompt template `level_1.j2` in the FeatureBench repo).

Procedure:
1. Run `pytest -x <FAIL_TO_PASS files>` in the task container to stop at the first failure.
2. Record executed Python source files under `/testbed/` using a tracing tool (e.g., `coverage.py` or Python’s `trace` module). Exclude `site-packages`.
3. Add a **depth-1 static import closure**: for each executed repo file, parse import statements and include the corresponding local module file(s) if they exist under `/testbed/`.
4. Add **interface target files** by extracting all `Path: `...`` entries from `problem_statement` (regex) and including those paths if they exist.

The final allowlist is the union of (2)+(3)+(4).

**B. Enforcing trace-bounded access.**

We implement TraceBound by restricting what the agent can read from `/testbed/`.

Two feasible implementation options (verification can choose the simpler):

- **Tool-level restriction (preferred for OpenHands):** patch the agent’s file-reading tool / workspace read primitive to raise `FileNotFoundError` when the requested path is not in the allowlist.
- **Filesystem-level restriction:** create a restricted overlay directory containing only allowlisted files (and required directories) and run the agent with that as its working tree.

In both cases:
- Attempts to read out-of-scope files return a plain error only (no suggested file list).
- Writing new files is allowed only under directories that exist in the restricted view (filesystem-level) or only for allowlisted parent directories (tool-level); this keeps the policy simple and prevents bypass via “write then read”.

**C. Headroom pre-check (early stop).**

Before running the full experiment, we run the baseline agent on a small pilot (e.g., 5 tasks) and compute:
- Fraction of baseline file-read content (measured in characters) that comes from files outside the allowlist.
- Median allowlist size as % of total repo files.

If baseline out-of-scope read volume is <30% or the allowlist covers >50% of repo files, we expect limited headroom and treat the approach as unlikely to matter.

### Key Innovations

1. **Instance-specific context restriction from dynamic tests:** use `FAIL_TO_PASS` execution traces as a cheap, per-task relevance signal.
2. **Hard constraint (not summarization):** enforce a bounded file-access policy instead of compressing content after retrieval.
3. **Mechanism-aligned to FeatureBench failures:** directly targets the benchmark’s identified inefficiency and cross-file dependency failure modes.

---

## Related Work

### Field Overview

Repository-level software engineering agents are commonly evaluated on benchmarks like SWE-bench and its variants, but these benchmarks historically skew toward bug fixes rather than feature development. FeatureBench expands evaluation to feature-level tasks with larger edit scopes and extensive test suites, making context cost a first-order concern.

A large body of work focuses on improving agents via better scaffolds (e.g., specialized tool interfaces), better localization/retrieval (e.g., graph-based repository indexing), and better context management (summarization, pruning, or memory). Recent work also introduces process-level evaluation of context retrieval quality (not just end success), suggesting that retrieval inefficiency and evidence drop are key bottlenecks.

TraceBound is most closely related to context retrieval and management methods, but differs in using an **instance-specific dynamic execution trace** to bound the accessible code neighborhood.

### Related Papers

- **[FeatureBench](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt)**: Feature-level benchmark with extremely high Token I/O and failure analysis pointing to cross-file dependency errors.
- **[SWE-bench](https://arxiv.org/abs/2310.06770)**: Execution-based benchmark for real GitHub issue resolution.
- **[SWE-agent](https://arxiv.org/abs/2405.15793)**: Shows that agent-computer interface design (file viewer/search/editor) materially improves SWE-bench performance.
- **[OpenHands SDK](https://arxiv.org/abs/2511.03690)**: A composable agent framework with explicit tool abstractions and context condensation.
- **[Agentless](https://arxiv.org/abs/2407.01489)**: Pipeline-style SWE-bench solver emphasizing controlled steps and retrieval.
- **[AutoCodeRover](https://arxiv.org/abs/2406.18436)**: Autonomous program improvement pipeline for repo-level tasks.
- **[SWE-Gym](https://arxiv.org/abs/2504.15627)**: Generates/verifies trajectories for training SWE agents.
- **[R2E-Gym](../../papers/paper_summaries/R2E-Gym Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents.md)**: Procedural environments + hybrid verifiers to scale open-weight SWE agents.
- **[SWE-Pruner](../../papers/paper_summaries/SWE-Pruner Self-Adaptive Context Pruning for Coding Agents.md)**: Line-level context pruning to reduce read-token dominance.
- **[Context as a Tool](../../papers/paper_summaries/Context as a Tool Context Management for Long-Horizon SWE-Agents.md)**: Treats context management as explicit actions and memory segments.
- **[SWE-Replay](../../papers/paper_summaries/SWE-Replay Efficient Test-Time Scaling for Software Engineering Agents.md)**: Test-time scaling via trajectory reuse and step selection.
- **[LocAgent](../../papers/paper_summaries/LocAgent Graph-Guided LLM Agents for Code Localization.md)**: Graph-guided localization tools for multi-hop repo navigation.
- **[RepoGraph](https://arxiv.org/abs/2410.14684)**: Repository-level code graphs used as a retrieval plug-in for SWE-bench.
- **[CodexGraph](https://openreview.net/forum?id=ZjKvGHHaVi)**: Uses a code graph database and LLM-generated graph queries for repo-level tasks.
- **[InfCode-C++](https://arxiv.org/abs/2511.16005)**: Intent-guided semantic retrieval and AST-structured tools for C++ issue resolution.
- **[Code Graph Model (CGM)](https://arxiv.org/abs/2505.16901)**: Integrates code graphs into model attention for SWE-bench-style tasks.
- **[ContextBench](https://arxiv.org/abs/2602.05892)**: Benchmark for context retrieval in coding agents; highlights evidence drop and retrieval inefficiency.
- **[TRAIL](https://arxiv.org/abs/2505.08638)**: Uses execution traces for issue localization and agent reasoning.
- **[Gistify!](https://arxiv.org/abs/2510.26790)**: Studies codebase-level understanding using runtime execution traces (different task, but closest dynamic-trace signal).
- **[InterCode](https://arxiv.org/abs/2306.14898)**: Standardizes interactive coding with execution feedback.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: Reasoning+acting prompting paradigm used by many tool-using agents.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Feature-level benchmarks | Harder multi-file, feature-oriented tasks | FeatureBench | FeatureBench (resolved/passed/token I/O) | High cost; token efficiency underexplored |
| Agent scaffolds | Tool interfaces + iterative edit/test loops | SWE-agent, OpenHands | SWE-bench, FeatureBench | Can still over-read; long-horizon inefficiency |
| Static retrieval / repo graphs | Use static indices/graphs to retrieve relevant code | LocAgent, RepoGraph, CodexGraph, CGM | SWE-bench(-Lite), localization metrics | Requires building indices; can still over-retrieve |
| Context compression / pruning | Reduce tokens after retrieval (summaries/pruning/memory) | SWE-Pruner, Context as a Tool | SWE-bench | Compression may drop crucial details; still needs retrieval |
| Dynamic trace signals | Use runtime execution traces as supervision or signal | TRAIL, Gistify | Trace-based evaluations | Often different task framing; not used as access policy |

### Closest Prior Work

- **FeatureBench**: Uses dynamic tracing to build benchmark instances (extracting feature patches), but does not use traces to control what an evaluation agent can read during inference.
- **SWE-Pruner**: Prunes retrieved file content line-by-line; it does not define an instance-specific allowed file set derived from failing tests.
- **RepoGraph / CodexGraph / LocAgent / CGM**: Improve static retrieval/localization using graphs, but do not exploit the failing-test dynamic trace as a cheap instance-specific access policy.
- **Gistify!**: Shows runtime traces can materially aid codebase-level understanding, but the task is code extraction/minimization rather than interactive feature development.
- **ContextBench**: Provides evidence that context retrieval quality and evidence drop are bottlenecks; our proposal is a concrete, enforceable retrieval constraint motivated by this finding.

**Novelty Kill Search Summary:** Searched for combinations of “FeatureBench + dependency graph guided context”, “call graph guided file retrieval for LLM coding agents”, “dynamic tracing guided context retrieval”, and “runtime callgraph retrieval SWE-bench agent” (2026-02-16). Found graph-based retrieval systems (RepoGraph/CodexGraph/LocAgent/CGM) and pruning/context-management work (SWE-Pruner/CaT), but no prior work explicitly using **FAIL_TO_PASS execution traces** to enforce a bounded file-access policy for FeatureBench agents.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| FeatureBench | Benchmark construction uses dynamic tracing; evaluates Token I/O | No method to reduce Token I/O | Use failing-test traces at inference to bound reads | Trace is instance-specific and cheap; should cut irrelevant reads |
| SWE-Pruner | Prunes retrieved file text | Still retrieves broadly; pruning may drop key lines | Bound retrieval before reading | Prevents reading irrelevant files entirely |
| RepoGraph / CodexGraph | Static graph-based retrieval | Requires index; can over-expand neighborhood | Use dynamic trace neighborhood | Trace targets the actual failing path rather than global similarity |
| LocAgent | Graph-guided localization tools | Focuses on localization, not token budget | Enforce file-access constraint | Stops exploration outside failing path; targets token efficiency |
| Gistify! | Runtime trace helps code extraction | Different task (not feature dev) | Apply trace as access policy | Tests whether trace can constrain interactive development |

---

## Experiments

### Experimental Setup

**Benchmark:** FeatureBench Lite split (30 tasks) from **[LiberCoders/FeatureBench](./references/LiberCoders-FeatureBench-Datasets-at-HuggingFace/meta/meta_info.txt)**, evaluated with the official harness from the FeatureBench repo.

**Agent scaffold:** OpenHands via `fb infer` (FeatureBench repo). This matches one of the benchmark’s main baselines.

**Step budget:** Set OpenHands max iterations to 100 (FeatureBench reports diminishing returns beyond ~100 steps in its step-limit ablations).

**Determinism:** Use deterministic decoding (temperature 0 where supported) and `n_attempts=1`.

**Two conditions (main experiment):**
1. **Baseline:** Standard FeatureBench OpenHands run.
2. **TraceBound (ours):** Same run, but file reads are restricted to the per-task allowlist computed from `pytest -x FAIL_TO_PASS` trace + depth-1 import closure + interface paths extracted from the problem statement.

**Headroom pre-check gate (cheap pilot):** Run Baseline on 5 tasks; compute (i) out-of-allowlist read volume (chars) and (ii) allowlist size as % of repo. If out-of-scope read volume <30% or allowlist >50% of repo files, stop and report a negative result.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3-Coder-480B-A35B-Instruct | MoE | `Qwen/Qwen3-Coder-480B-A35B-Instruct` (API) | Reported in FeatureBench Table 2 under OpenHands |

(If API cost is prohibitive, verification can substitute `Pro/deepseek-ai/DeepSeek-V3.2` for a lower-cost replication, but the primary comparison should use the model reported in FeatureBench.)

**Resource Estimate**:
- **Compute budget**: mostly API inference + Docker test execution. Published Token I/O suggests ~2.6M input tokens per task for OpenHands+Qwen3-Coder on Lite (Table 2), so 30 tasks × 2 conditions implies O(150M) input tokens plus test runtime.
- **Wall-clock**: dominated by LLM latency and per-iteration test runs. Use `n_concurrent` conservatively (e.g., 1–2) to limit Docker load.
- **No training** required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| FeatureBench | Feature-level development in real Python repos with F2P/P2P tests | Resolved rate (↑), Passed rate (↑), Token I/O (↓ input tokens), plus trace diagnostics | lite | https://huggingface.co/datasets/LiberCoders/FeatureBench | https://github.com/LiberCoders/FeatureBench (`fb infer`, `fb eval`) |

**Additional diagnostics (reported by TraceBound wrapper):**
- Allowlist size (count; % of repo files).
- Count of denied read attempts.
- Crash-type distribution from pytest logs (NameError/ImportError/AttributeError vs AssertionError).

### Main Results

#### Results Table

| Method | Base Model | Benchmark | % Resolved | % Passed | Input tokens (median) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| OpenHands (published) | Qwen3-Coder-480B-A35B-Instruct | FeatureBench Lite | 6.7 | 38.31 | 2.6M | [FeatureBench](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt) | Reported at 500 steps; Token I/O is input/output |
| Baseline (re-run) | Qwen3-Coder-480B-A35B-Instruct | FeatureBench Lite | **TBD** | **TBD** | **TBD** | - | temp=0; 100 steps |
| **TraceBound (ours)** | Qwen3-Coder-480B-A35B-Instruct | FeatureBench Lite | **TBD** | **TBD** | **TBD** | - | Same as baseline; trace-bounded reads |

### Ablation Studies

No additional ablations in the initial verification run (scope discipline). If TraceBound fails due to missing required files, the first pivot is to increase import-closure depth from 1 to 2 and re-run on 5 tasks.

### Experimental Rigor

- **Determinism**: temp=0 and fixed step budget make runs near-deterministic; re-run any task where Baseline vs TraceBound differs to rule out container flakiness.
- **Key confounders**:
  - *Restriction confounded with guidance*: avoided by returning only FileNotFoundError on denied reads.
  - *Allowlist too large*: measured explicitly (median % of repo files).
  - *Trace misses necessary files*: checked by comparing deny events to final failure type distribution.

---

## Success Criteria

**Hypothesis (directional):** TraceBound reduces median input tokens substantially because it prevents reading irrelevant repository files, while preserving solve count under deterministic settings.

**Decision Rule (concrete):**
- **Proceed (success)** if on FeatureBench Lite:
  1) median **input tokens** decrease by **≥30%** from Baseline to TraceBound, and
  2) the number of solved tasks is identical (**ΔSolved = 0**) after confirming any differences are not due to run flakiness.
- **Refute** if:
  - ΔSolved < 0 persists after re-runs, or
  - median token reduction is <15%, or
  - allowlist size is too large to be meaningful (median >50% of repo files).
- **Pivot** only if the failure is clearly attributable to trace incompleteness (many denied reads immediately precede failures) and can be addressed by a single change (e.g., import-closure depth=2) without adding extra information to the agent.

---

## Impact Statement

If successful, TraceBound provides a simple, benchmark-native way to reduce the cost of running repository-level coding agents: use failing-test traces to bound what code the agent can read. This could change practice for agent builders by turning an expensive, heuristic file-reading policy into a deterministic, instance-specific access rule that reduces Token I/O without sacrificing success.

---

## References

- [FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt) - Zhou et al., 2026
- [GitHub - LiberCoders/FeatureBench](./references/GitHub-LiberCoders-FeatureBench/meta/meta_info.txt) - Code repository
- [LiberCoders/FeatureBench (HF dataset)](./references/LiberCoders-FeatureBench-Datasets-at-HuggingFace/meta/meta_info.txt) - Dataset card
- [SWE-bench](https://arxiv.org/abs/2310.06770) - Jimenez et al., 2024
- [SWE-agent](https://arxiv.org/abs/2405.15793) - Yang et al., 2024
- [OpenHands Software Agent SDK](https://arxiv.org/abs/2511.03690) - (authors), 2025
- [SWE-Pruner](https://arxiv.org/abs/2601.16746) - (authors), 2026
- [Context as a Tool](https://arxiv.org/abs/2512.22087) - (authors), 2025
- [SWE-Replay](https://arxiv.org/abs/2601.22129) - (authors), 2026
- [LocAgent](https://arxiv.org/abs/2503.09089) - (authors), 2025
- [RepoGraph](https://arxiv.org/abs/2410.14684) - (authors), 2024
- [CodexGraph](https://openreview.net/forum?id=ZjKvGHHaVi) - (authors), 2025
- [InfCode-C++](https://arxiv.org/abs/2511.16005) - (authors), 2025
- [Code Graph Model (CGM)](https://arxiv.org/abs/2505.16901) - (authors), 2025
- [ContextBench](https://arxiv.org/abs/2602.05892) - (authors), 2026
- [TRAIL](https://arxiv.org/abs/2505.08638) - (authors), 2025
- [Gistify!](https://arxiv.org/abs/2510.26790) - (authors), 2025
- [InterCode](https://arxiv.org/abs/2306.14898) - Yang et al., 2023
- [ReAct](https://arxiv.org/abs/2210.03629) - Yao et al., 2023
