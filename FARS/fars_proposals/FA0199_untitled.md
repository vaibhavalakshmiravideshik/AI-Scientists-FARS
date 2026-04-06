# untitled

# Interface-Rooted Repo Maps for Token-Efficient FeatureBench Agents

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language model (LLM) coding agents are increasingly used to modify real software repositories. Compared to single-file code generation, real development requires (i) finding the right files and symbols, (ii) maintaining cross-file consistency as changes propagate, and (iii) running tests and debugging failures. These steps require long-horizon interaction with a repository, where the agent repeatedly reads files and issues edits.

A major practical bottleneck is **token inefficiency**: agent runs often spend millions of input tokens per task reading and re-reading repository context, which increases latency and cost and makes large-scale evaluation and iteration difficult. Token inefficiency also interacts with correctness: when an agent does not read the right files, it may guess (hallucinate) function signatures or class attributes, producing cross-file inconsistencies.

FeatureBench is a recent benchmark designed to measure **feature-level** repository development (not just bug fixes) using unit-test-based evaluation (fail-to-pass and pass-to-pass tests). FeatureBench reports that even strong agent+model combinations consume multi-million input tokens per task, while solving only a small fraction of tasks. For example, on the 30-task Lite split, **OpenHands + DeepSeek-V3.2** solves **6.7%** of tasks with **3.1M / 24k** input/output tokens per task, and **OpenHands + Claude Opus 4.5** solves **20.0%** of tasks with **8.8M / 29k** tokens per task (Table 2 in FeatureBench). This suggests there is substantial room for **context engineering** that reduces wasted reads without sacrificing correctness.

### The Problem

This proposal targets a specific, automatable failure mode: **agents waste tokens exploring irrelevant parts of the repository because they lack a compact, task-conditioned overview of the relevant interfaces and dependencies**.

FeatureBench’s failure analysis reports that **NameError** is a dominant failure mode for Claude Opus 4.5, suggesting that current agents struggle with cross-file dependency resolution and consistent symbol use. The paper also highlights “idle habits”: models often guess interfaces or attributes defined in other files instead of reading them, leading to TypeError and AttributeError. These are consistent with a retrieval problem: the agent is not consistently building a correct mental map of what symbols exist where.

At the same time, FeatureBench prompts already contain a strong signal about task entry points: they include an **Interface Description** section with explicit file paths (e.g., `Path: '/testbed/pkg/module.py'`) and the required signatures. A practitioner (or a tool) can treat these paths as **roots** to identify a dependency neighborhood. However, typical agent scaffolds treat the repository as an unstructured file system and rely on the LLM to decide what to open next.

A natural alternative is to provide a **repository map**: a compact summary of the repository structure and important symbols. Existing repo maps (e.g., Aider’s global repo map) are typically task-agnostic and may include many irrelevant files for a given task. This motivates a task-conditioned map that is rooted at the provided interface files.

### Key Insight and Hypothesis

**Key insight**: On FeatureBench Level-1 tasks, the prompt already gives a small set of “interface root” file paths. A static analyzer can expand an **import-closure neighborhood** (the transitive closure of files reachable via `import` / `from ... import ...` edges) from these roots and summarize only the signatures/docstrings of symbols in that neighborhood, producing a small, structured context prefix that helps the agent (i) avoid reading irrelevant files and (ii) avoid hallucinating cross-file interfaces.

**Hypothesis**: Prepending an **interface-rooted, bounded import-closure repo map** to the OpenHands prompt will reduce mean input tokens by **≥25%** on FeatureBench-Lite (Level 1) while keeping quality within a small tolerance of the baseline.

Why this could be wrong:
- OpenHands may already find needed files quickly; the map may not change behavior.
- The import graph may miss dynamic imports / runtime wiring, causing the map to omit critical definitions.
- Any benefit may come from adding extra tokens (more context) rather than structure; we control for this with an equal-token random map.

---

## Proposed Approach

### Overview

We propose **Interface-Rooted RepoMap (IR-RepoMap)**: a task-conditioned repository map built automatically from the repository files inside the FeatureBench container.

Given a FeatureBench problem statement, we:
1. Parse the interface file paths from the prompt (“Path: …”).
2. Build a file-level import graph (Python AST parsing) and compute a bounded dependency closure from the interface roots.
3. Extract top-level symbol summaries (function/class signatures and short docstrings) for files in the closure.
4. Prepend the resulting structured “repo map” text to the agent prompt before running OpenHands.

### Method Details

**Inputs** (per task):
- `problem_statement` text from FeatureBench (contains one or more interface file paths under `/testbed/`).
- Repository source tree in the container under `/testbed/`.

**Step 1: Interface root extraction**
- Regex-scan `problem_statement` for lines matching `Path: '<path>'`.
- Keep only paths under `/testbed/` with suffix `.py`.

**Step 2: Import graph construction**
- For each Python file in the repository (or lazily for visited files), parse AST and extract:
  - `import pkg.subpkg` edges
  - `from pkg.subpkg import name` edges
- Resolve relative imports using the file’s package context.
- Map module imports to candidate file paths using `sys.path`-like resolution within `/testbed/`.

**Step 3: Bounded import closure**
- Perform breadth-first search (BFS) from the interface root files over the import graph.
- Stop when either condition is reached:
  - maximum depth `D` (default `D=3`), or
  - token budget `B_map` for the repo map text (default `B_map=1500` tokens, computed with the target model tokenizer or a proxy tokenizer).
- Ranking for truncation: prioritize smaller BFS depth; within the same depth, prioritize higher in-closure degree (proxy for centrality).

**Step 4: Symbol extraction and formatting**
For each selected file:
- Extract top-level `class` and `def` nodes.
- Render an approximate signature from the AST (argument names + defaults; include type annotations when present).
- Include at most the first line of docstring (if present).

**Repo map format (sketch)**

```
[IR-RepoMap | auto-generated]
Interface roots:
- /testbed/pkg/a.py

Import closure (depth<=3, budget<=1500 tokens):
- /testbed/pkg/a.py -> /testbed/pkg/b.py, /testbed/pkg/utils.py
- ...

Symbols:
/testbed/pkg/b.py
- def foo(x, y=..., ...) -> ... : "..."
- class Bar: "..."
...
```

**Random-map control**
To control for “any extra prefix text helps”, we include an equal-token random map:
- For each task, sample random Python files and top-level symbols from the repository.
- Format them in the same template.
- Truncate to the same token length as IR-RepoMap for that task.
- Use a deterministic RNG seed derived from the task `instance_id` for reproducibility.

### Key Innovations

- **Task-conditioned repo map construction**: rather than a global repository map, IR-RepoMap is rooted in task-provided interface paths and limited to a dependency closure.
- **Interface-rooted inductive bias**: FeatureBench prompts explicitly provide interface locations; IR-RepoMap converts this into a retrieval policy.
- **Token-efficiency evaluation with a null control**: we evaluate token consumption directly and include an equal-token random map to isolate structure from length.

---

## Related Work

### Field Overview

**Agentic coding benchmarks.** Recent benchmarks evaluate coding agents in realistic repositories with unit tests, including SWE-bench and FeatureBench. FeatureBench specifically targets feature development tasks that require multi-file changes and long-horizon debugging, and reports large gaps between pass rate and fully solved rate.

**Repository-level context retrieval and summarization.** Multiple systems propose retrieval and summarization for repository-scale tasks, including graph-guided localization, repo exploration/summarization agents, and context management policies for long-horizon agents.

**Repo maps.** Tools like Aider provide a repository map (often built from a graph centrality heuristic) as compact context to steer edits. However, global maps can be task-irrelevant, and there is limited evaluation of task-conditioned repo maps under token-efficiency metrics.

### Related Papers

- **[FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt)**: Introduces FeatureBench and reports multi-million token usage and low solve rates for feature-level development agents.
- **[ContextBench: A Benchmark for Context Retrieval in Coding Agents](./references/ContextBench/meta/meta_info.txt)**: Benchmarks context retrieval quality for coding agents, motivating structured retrieval policies beyond naive file reads.
- **[RepoMaster: Autonomous Exploration and Understanding of GitHub Repositories for Complex Task Solving](./references/RepoMaster/meta/meta_info.txt)**: Proposes an autonomous repo exploration and summarization pipeline for complex repo tasks, highlighting the importance of repository-level understanding.
- **[GraphCodeAgent: Dual Graph-Guided LLM Agent for Retrieval-Augmented Repo-Level Code Generation](./references/CodeRAG/meta/meta_info.txt)**: Uses dual graphs (code structure + dependencies) to guide retrieval for repo-level code generation.
- **[LocAgent: Graph-Guided LLM Agents for Code Localization](./references/LocAgent/meta/meta_info.txt)**: Uses a graph-guided approach to localize relevant code regions for repository tasks.
- **[Context as a Tool: Context Management for Long-Horizon SWE-Agents](./references/Context-as-a-Tool/meta/meta_info.txt)**: Studies proactive context management for long-horizon software engineering agents.
- **[SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)**: Introduces a widely used benchmark for issue-resolution via patches and tests, motivating repository-scale evaluation.
- **[SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)**: Introduces an agent scaffold for SWE-bench with tool use and iterative debugging.
- **[The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents](https://arxiv.org/abs/2511.03690)**: Describes the OpenHands agent SDK, a common scaffold for repo-level coding agents.
- **[AutoCodeRover: Autonomous Program Improvement](https://arxiv.org/abs/2404.05427)**: Uses retrieval and structured search to improve code in repositories, emphasizing localization.
- **[RGFL: Reasoning Guided Fault Localization for Automated Program Repair Using Large Language Models](https://arxiv.org/abs/2404.03059)**: Combines reasoning with fault localization signals to find edit locations in repo repair.
- **[SWE-smith: Scaling Data for Software Engineering Agents](https://arxiv.org/abs/2406.10596)**: Creates large-scale data and training for SWE agents, relevant to how agents learn repo navigation.
- **[SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents](https://arxiv.org/abs/2601.16746)**: Proposes task-aware line-level pruning of file reads to reduce token costs in SWE agents.
- **[LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth](https://arxiv.org/abs/2602.07962)**: Evaluates agents under systematically scaled context growth, highlighting the need for context management.
- **[Scaling Long-Horizon LLM Agent via Context-Folding](https://arxiv.org/abs/2510.11967)**: Proposes a branch/return mechanism to fold long trajectories into compact summaries.
- **[Agent READMEs: An Empirical Study of Context Files for Agentic Coding](https://arxiv.org/abs/2511.12884)**: Studies how repositories provide persistent instruction files (e.g., CLAUDE.md) and what guidance they contain.
- **[ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization](https://arxiv.org/abs/2509.13313)**: Uses periodic summarization to sustain long-horizon agent behavior beyond context limits.
- **[Budget-Aware Tool-Use Enables Effective Agent Scaling](https://arxiv.org/abs/2511.17006)**: Studies explicit budget signals for tool-using agents and cost–accuracy trade-offs.
- **[SWE-QA: Can Language Models Answer Repository-level Code Questions?](https://arxiv.org/abs/2509.14635)**: Introduces repository-level code QA tasks that require cross-file dependency reasoning.
- **[NL2Repo-Bench: Towards Long-Horizon Repository Generation Evaluation of Coding Agents](https://arxiv.org/abs/2512.12730)**: Evaluates long-horizon repository generation from natural language specs.
- **[One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents](https://arxiv.org/abs/2512.20957)**: Trains repo navigation with a jump-to-definition tool and emphasizes static symbol resolution.
- **[When Agents Go Astray: Course-Correcting SWE Agents with Process Reward Models](https://arxiv.org/abs/2509.02360)**: Uses inference-time process reward models to detect and correct inefficient SWE-agent trajectories.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Global repo maps | Provide a task-agnostic repo summary (often graph-centrality-based) | Aider repo map (tool), RepoMaster | Ad hoc agent evaluations | Can include many irrelevant files for a given task |
| Graph-guided retrieval/localization | Use structural graphs to retrieve relevant code regions | LocAgent, GraphCodeAgent, AutoCodeRover, RGFL | SWE-bench, repo-level tasks | Graph construction may be expensive or incomplete |
| Context management policies | Summarize/prune/fold context over long trajectories | Context as a Tool, Context-Folding, ReSum | SWE-bench-style tasks | Summaries can omit details; may require learned policies |
| Retrieval benchmarks | Measure retrieval quality separately from end-to-end solve | ContextBench | Dedicated retrieval tasks | Correlation with end-to-end performance is imperfect |

### Closest Prior Work

- **Aider repository maps (tool documentation)**: Aider includes a repository map that summarizes important symbols using a graph-centrality heuristic and a token budget, but it is not explicitly rooted at the task’s interface paths and (to our knowledge) has not been evaluated on FeatureBench with token-efficiency metrics.
- **[RepoMaster](./references/RepoMaster/meta/meta_info.txt)**: Builds hierarchical summaries while exploring repositories, but focuses on autonomous exploration rather than a lightweight static-analysis map rooted in known entry points.
- **[GraphCodeAgent](./references/CodeRAG/meta/meta_info.txt)**: Uses dual graphs for retrieval-augmented repo-level code generation, but targets generation quality rather than directly optimizing token consumption in an interactive debugging loop.
- **[LocAgent](./references/LocAgent/meta/meta_info.txt)**: Guides localization with graphs, but does not leverage FeatureBench’s explicit interface-path signal to build an import-closure map.
- **[Context as a Tool](./references/Context-as-a-Tool/meta/meta_info.txt)**: Treats context management as an explicit tool and proposes proactive compression, but does not propose an interface-rooted dependency closure map with an equal-token null control.

**Novelty Kill Search Summary:** Searched for combinations of “FeatureBench repomap”, “FeatureBench token reduction context”, “repository map code agent token reduction”, and “dependency graph guided context retrieval for code LLM agents”, plus concurrent-work checks for 2025–2026 on arXiv/OpenReview/GitHub. No prior work was found that explicitly evaluates an **interface-rooted, bounded import-closure repo map** intervention on FeatureBench (arXiv:2602.10975) as of 2026-02-21.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Aider repo map (tool) | Global repo map using graph ranking + token budget | Task-agnostic; may include irrelevant context | Build map from task interface roots + import closure | Better relevance → fewer wasted reads and fewer cross-file hallucinations |
| [RepoMaster](./references/RepoMaster/meta/meta_info.txt) | Autonomous exploration + summarization | More complex; exploration may be token-expensive | Lightweight static analysis before agent starts | Lower overhead; directly targets token use |
| [GraphCodeAgent](./references/CodeRAG/meta/meta_info.txt) | Graph-guided retrieval for repo-level generation | Not optimized for interactive debugging token usage | Optimize for token consumption in FeatureBench loop | FeatureBench token metric provides direct feedback |
| [LocAgent](./references/LocAgent/meta/meta_info.txt) | Graph-guided localization | Focused on localization; no interface-root prior | Interface-rooted dependency closure | Uses benchmark-provided entry points |
| [Context as a Tool](./references/Context-as-a-Tool/meta/meta_info.txt) | Context management via compression tool | Summaries may omit needed details | Signature/docstring map anchored to real code | Keeps precise interfaces while remaining compact |

---

## Experiments

### Experimental Setup

**Goal:** Measure whether IR-RepoMap reduces token consumption on FeatureBench while maintaining comparable solve quality.

**Main conditions (3-way, fully automated):**
1. **Baseline (OpenHands)**: Standard FeatureBench OpenHands agent with no repo map prefix.
2. **IR-RepoMap (ours)**: Same agent, but prepend IR-RepoMap to the prompt before the run.
3. **Random-map control**: Same agent, but prepend an equal-token random map.

**Dataset / split:**
- **FeatureBench Lite** (30 tasks), **Level 1** subset.

**Models:**
- Primary: **DeepSeek-V3.2** via API (matches a published baseline in FeatureBench Table 2 under OpenHands).

**Why DeepSeek-V3.2:**
- Published Table 2 token and quality baselines exist for OpenHands + DeepSeek-V3.2 on Lite.
- It is available as an API model (does not consume local GPU budget).

**Agent settings (held constant across all conditions):**
- Same OpenHands max steps (FeatureBench default is 500).
- Same per-task wall-clock timeout (set explicitly; e.g., 3600s) and same retry policy.
- Deterministic decoding where supported (temperature 0) to reduce variance; otherwise single-run is consistent with FeatureBench’s reported baselines.

**Baseline Ladder (REQUIRED):**
- **Trivial / null**: equal-token random map control (included).
- **Strong existing method**: FeatureBench’s OpenHands baseline (included).
- **Inference-time scaling** (diagnostic if needed): OpenHands with increased attempt count (e.g., attempts=3) to quantify the quality–token tradeoff; run only if IR-RepoMap shows quality gains but unclear efficiency gains.
- **Closest existing repo-map method** (optional follow-up): Aider-style global repo map as a drop-in prefix (if implementable within FeatureBench agent wrapper) to test whether task-conditioning matters.

**Resource Estimate**:
- **Compute budget**: 0 local GPU-hours (API inference only).
- **API usage (rough, from FeatureBench Table 2 for OpenHands+DeepSeek-V3.2 on Lite)**:
  - Baseline: ~3.1M input + 24k output tokens per task × 30 tasks ≈ 93M input + 0.72M output tokens.
  - Three conditions (baseline + ours + random): ≈ 280M input tokens and ≈ 2.2M output tokens (order-of-magnitude; ours may be lower).
- **Wall-clock**: depends on per-task timeouts; recommend parallelizing across tasks with a fixed timeout.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| FeatureBench | Feature-level repository development tasks with unit tests (F2P+P2P) | mean input tokens; mean output tokens; pass rate; solved rate | Lite (30) + Level 1 | https://huggingface.co/datasets/LiberCoders/FeatureBench | `fb infer`, `fb eval`, plus `featurebench/scripts/cal_eval_outputs.py` |

**Metric definitions (from FeatureBench):**
- **Pass rate**: ratio of passed fail-to-pass tests among executed F2P tests.
- **Solved (resolved) rate**: fraction of tasks where pytest exits with code 0 (both F2P and P2P pass).

**Primary metric:**
- Mean **input (prompt) tokens per task** (lower is better).

**Secondary metrics / guardrails:**
- Solved (resolved) rate and pass rate.

### Main Results

#### Results Table

(All numbers below are for FeatureBench Lite + Level 1; to be filled by verification runs. Published baselines are reported for the overall Lite split and may mix difficulty levels; they are shown only for context.)

| Method | Base Model | Benchmark | mean input tokens | mean output tokens | solved rate | pass rate | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| OpenHands (published) | DeepSeek-V3.2 | FeatureBench-Lite (overall) | 3.1M | 24k | 6.7% | 35.94% | FeatureBench Table 2 | Published (1 run); not level-stratified |
| OpenHands (baseline) | DeepSeek-V3.2 | FeatureBench-Lite (Level 1 only) | TBD | TBD | TBD | TBD | - | To be verified |
| OpenHands + random map | DeepSeek-V3.2 | FeatureBench-Lite (Level 1 only) | TBD | TBD | TBD | TBD | - | Equal-token prefix control |
| **OpenHands + IR-RepoMap (ours)** | DeepSeek-V3.2 | FeatureBench-Lite (Level 1 only) | TBD | TBD | TBD | TBD | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (full) | Interface roots + import closure + signatures/docstrings | Best token-efficiency / quality trade-off |
| w/o import-closure (roots only) | Only include interface root file summaries | Smaller effect; tests whether closure matters |

### Experimental Rigor

**Validity & Controls:**
- **Length confound**: random-map control matches the token length of IR-RepoMap to test whether structure, not just prefix length, drives any effect.
- **Budget confound**: keep OpenHands step limits and timeouts identical across conditions.
- **Determinism**: run with temperature 0 where possible; otherwise, report that FeatureBench baselines are single-run and treat tasks as the primary sample unit.

---

## Success Criteria

**Hypothesis** (directional — what we expect):
IR-RepoMap will reduce mean input tokens by avoiding unnecessary repository reads and by reducing cross-file debugging loops, while keeping solved/pass rates close to the OpenHands baseline.

**Decision Rule** (concrete — when to stop):
- **Proceed** if, on FeatureBench-Lite Level 1:
  - mean_input_tokens(ours) ≤ **0.75×** mean_input_tokens(baseline) **and** ≤ **0.80×** mean_input_tokens(random-map), and
  - pass_rate(ours) ≥ pass_rate(baseline) − **0.03**, and
  - solved_count(ours) ≥ solved_count(baseline) − **1**.
- **Pivot** if token savings are ≥25% but quality drops beyond the guardrails; try reducing map budget `B_map` or lowering closure depth `D` to decrease prompt interference.
- **Refute** if token reduction is <10% vs baseline, or if IR-RepoMap is not better than the random-map control, or if solved_count drops by ≥3.

---

## Impact Statement

If successful, IR-RepoMap would provide a lightweight, automatable way to reduce the cost and latency of repository-level coding agents on feature development tasks. This would benefit practitioners running agents in CI-like loops (less API spend and faster iteration) and researchers who need to scale benchmark evaluation without paying multi-million-token per-task overhead.

---

## References

- [FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](./references/FEATUREBENCH-BENCHMARKING-AGENTIC-CODING-FOR-COMPLEX-FEATURE-DEVELOPMENT/meta/meta_info.txt) - Zhou et al., 2026
- [ContextBench: A Benchmark for Context Retrieval in Coding Agents](./references/ContextBench/meta/meta_info.txt) - Li et al., 2026
- [RepoMaster: Autonomous Exploration and Understanding of GitHub Repositories for Complex Task Solving](./references/RepoMaster/meta/meta_info.txt) - Wang et al., 2025
- [GraphCodeAgent: Dual Graph-Guided LLM Agent for Retrieval-Augmented Repo-Level Code Generation](./references/CodeRAG/meta/meta_info.txt) - Li et al., 2025
- [LocAgent: Graph-Guided LLM Agents for Code Localization](./references/LocAgent/meta/meta_info.txt) - (metadata in reference folder)
- [Context as a Tool: Context Management for Long-Horizon SWE-Agents](./references/Context-as-a-Tool/meta/meta_info.txt) - Liu et al., 2025
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- [The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents](https://arxiv.org/abs/2511.03690)
- [AutoCodeRover: Autonomous Program Improvement](https://arxiv.org/abs/2404.05427)
- [RGFL: Reasoning Guided Fault Localization for Automated Program Repair Using Large Language Models](https://arxiv.org/abs/2404.03059)
- [SWE-smith: Scaling Data for Software Engineering Agents](https://arxiv.org/abs/2406.10596)
- [SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents](https://arxiv.org/abs/2601.16746)
- [LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth](https://arxiv.org/abs/2602.07962)
- [Scaling Long-Horizon LLM Agent via Context-Folding](https://arxiv.org/abs/2510.11967)
- [Agent READMEs: An Empirical Study of Context Files for Agentic Coding](https://arxiv.org/abs/2511.12884)
- [ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization](https://arxiv.org/abs/2509.13313)
- [Budget-Aware Tool-Use Enables Effective Agent Scaling](https://arxiv.org/abs/2511.17006)
- [SWE-QA: Can Language Models Answer Repository-level Code Questions?](https://arxiv.org/abs/2509.14635)
- [NL2Repo-Bench: Towards Long-Horizon Repository Generation Evaluation of Coding Agents](https://arxiv.org/abs/2512.12730)
- [One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents](https://arxiv.org/abs/2512.20957)
- [When Agents Go Astray: Course-Correcting SWE Agents with Process Reward Models](https://arxiv.org/abs/2509.02360)
