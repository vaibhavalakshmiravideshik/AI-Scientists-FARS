# untitled

# LASCon: Loop-Aware Scratchpad Condensation for Terminal Agents

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language model (LLM) agents are increasingly evaluated and deployed in *command-line interface (CLI)* environments for realistic software engineering and system administration tasks. **Terminal-Bench** is a widely used benchmark of Dockerized CLI tasks with outcome-based verification (a task is solved if automated tests pass within a task-specific time limit) ([Terminal-Bench](./references/Terminal-Bench-Benchmarking-Agents-on-Hard-Realistic-Tasks-in-Command-Line-Interfaces/meta/meta_info.txt)). Terminal-Bench reports that some tasks can consume extremely large numbers of tool calls and tokens per attempt, making long-horizon interaction efficiency a practical bottleneck in 2026.

A recurring failure mode in CLI agents is **unproductive repetition**: repeatedly running the same command, re-reading the same files, or cycling through the same error state without changing the environment. **CLI-Gym** (a scalable pipeline for generating environment-repair tasks and trajectories) reports that for Qwen3-32B evaluated with the **OpenHands** agent framework, the fraction of failures “stuck in repetitive action loops” on Terminal-Bench 1.0 can be very large, and decreases sharply as the model is fine-tuned on successful trajectories ([CLI-Gym](./references/CLI-Gym-Scalable-CLI-Task-Generation-via-Agentic-Environment-Inversion/meta/meta_info.txt)). However, CLI-Gym also observes that after such training, models explore more and become more likely to exceed the maximum inference context length (128k), suggesting that “more persistence” can increase context bloat.

This proposal asks a practitioner-relevant question: **can we reduce loop-driven failures and context-window overflows without any model retraining**, using only agent-scaffold changes that are compatible with **OpenHands** (an open-source software agent SDK providing standardized tools like shell execution and file editing) ([OpenHands SDK](./references/The-OpenHands-Software-Agent-SDK-A-Composable-and-Extensible-Foundation-for-Production-Agents/meta/meta_info.txt))?

### The Problem

Open-source agent scaffolds typically rely on two families of safeguards:

1. **Context condensation**: when the interaction history grows too large, older events are compressed or removed. OpenHands implements a *Condenser* that drops older events and inserts an LLM-generated summary so the agent stays within the model context window ([OpenHands SDK](./references/The-OpenHands-Software-Agent-SDK-A-Composable-and-Extensible-Foundation-for-Production-Agents/meta/meta_info.txt)).

2. **Stuck/loop detection with termination**: when the agent repeats actions without progress, the run is halted to prevent resource waste. OpenHands implements a *Stuck Detector* that detects infinite loops and redundant tool calls and can terminate execution ([OpenHands SDK](./references/The-OpenHands-Software-Agent-SDK-A-Composable-and-Extensible-Foundation-for-Production-Agents/meta/meta_info.txt)).

These mechanisms do not fully resolve the CLI setting:

- **Condensation is often lossy in the wrong way.** CLI tasks hinge on exact flags, file paths, and error lines. Free-form summaries can omit the exact tokens needed to repair the environment. Moreover, recent evidence in software engineering agents suggests that simple, deterministic observation masking can match LLM summarization (and avoid trajectory-lengthening side effects) on SWE-bench-like tasks ([Simple Observation Masking](https://arxiv.org/abs/2508.21433)). This motivates deterministic, CLI-tailored condensation as a strong baseline.

- **Termination is not recovery.** If a stuck detector only terminates, it prevents runaway cost but does not help the agent escape a local failure mode. Under fixed time limits, early termination can reduce success.

- **Prompting is not enforcement.** A well-crafted system prompt can advise the model to avoid repeating the same command after the same error, but the model can still choose to ignore this instruction; there is no mechanism-level guarantee.

If a lightweight scaffold can reliably prevent “retry loops” and reduce context bloat, it could (i) improve **Pass@1** (the fraction of tasks solved in a single attempt under the benchmark’s time limit and a fixed decoding configuration), and/or (ii) reduce tokens and tool calls per task at equal Pass@1, changing what practitioners deploy as default efficiency/reliability controls for terminal agents.

### Key Insight and Hypothesis

**Key insight:** In CLI environments, many unproductive loops are structurally detectable by combining:

- an **action signature** (canonicalized tool call, e.g., the exact shell command)
- an **error signature** (stable hash of extracted error lines and tail output)
- an external **state-version** signal indicating whether the environment materially changed (e.g., file edits or filesystem fingerprint changes)

If an agent repeats the same action+error signature while the state version does not change, the next repetition is unlikely to yield new information.

**Hypothesis:** A training-free OpenHands scaffold that (i) deterministically condenses CLI observations into a structured scratchpad while offloading raw logs to disk, and (ii) enforces a progress-conditioned *hard block* on repeating action+error signatures under unchanged state version, will reduce loop-induced failures and context overflows, improving Terminal-Bench Pass@1 and/or reducing tokens per task under the same time limits.

The outcome is uncertain because (1) OpenHands’ existing condenser and stuck detector may already eliminate most loop failures, leaving little headroom; and (2) hard blocking could create new failure modes by preventing necessary retries (e.g., benign repeats after subtle state changes). The experiment plan includes an explicit “prompt-only” baseline and an early-stop gate that measures loop headroom before running full evaluations.

---

## Proposed Approach

### Overview

We propose **LASCon (Loop-Aware Scratchpad Condensation)**: a small set of OpenHands SDK components that (A) replace LLM-based summarization with deterministic, CLI-specific condensation and (B) replace “stuck ⇒ terminate” with “stuck ⇒ block only the specific repeated action to force exploration”. LASCon is designed to be:

- **Training-free**: no fine-tuning; only scaffold logic.
- **Deterministic and auditable**: condensation and loop decisions are rule-based and logged.
- **Compatible** with standard terminal benchmarks: no changes to tasks or verifiers.

### Method Details

LASCon has two modules.

#### 1) Deterministic CLI Condenser (DCC)

Goal: prevent context overflow while keeping operationally critical details.

Implementation sketch:

- For each tool observation (especially `execute_bash` outputs), compute a **compact record**:
  - canonicalized command string
  - exit code
  - working directory
  - extracted error lines (regex-based: `error|failed|exception|traceback|no such file|permission denied|not found`)
  - first N and last N lines (N=10)
  - a pointer to the full raw output saved to disk (e.g., `logs/obs_<event_id>.txt`)

- Replace long observations in the LLM-visible history with the compact record + pointer, similar in spirit to observation masking/windowing in SWE-agent, but tailored to CLI outputs.

- Maintain a **structured scratchpad** (updated every step) that the model always sees, containing:
  - task goal (from instruction)
  - current working directory
  - last few commands + outcomes
  - files edited (from file tool calls)
  - current “do-not-repeat” list (from the loop controller)
  - latest test summary (if any)

This module is primarily an engineering adaptation of observation masking to CLI logs; we do not claim that deterministic masking is novel, but we hypothesize that CLI-specific structuring and log offloading improves signal-to-noise in terminal tasks.

#### 2) Progress-Conditioned Loop Controller (PLC)

Goal: escape unproductive command loops by preventing exact repeats when there is no evidence of progress.

Core concepts:

- **Action signature** `sig(a_t)`: canonicalized tool call (for `execute_bash`, the full command string with normalized whitespace).
- **Error signature** `err(o_t)`: stable hash of extracted error lines + tail output.
- **State version** `v_t`: an integer maintained outside the LLM that increments when the environment state *relevant to the task workspace* changes. For Terminal-Bench tasks (which run in a Docker container with a task workspace rooted at `/testbed`), we define:
  - Maintain a rolling fingerprint `F_t` over `/testbed` computed as the SHA-256 hash of the sorted list of `(relative_path, file_size_bytes, mtime_ns)` for all regular files under `/testbed`, excluding common large/irrelevant directories (`.git/`, `__pycache__/`, `.pytest_cache/`, `node_modules/`, `.venv/`, `venv/`, and `/testbed/.cache/` if present).
  - After each executed action, recompute `F_t`. If `F_t != F_{t-1}`, increment `v_t` by 1.

This definition is intentionally filesystem-only (it does not attempt to track process or network state) so it is cheap and deterministic; the proposal evaluates whether this is sufficient in practice.

Loop trigger:

- Maintain a rolling window `W=12` of recent `(sig, err, v)` triples.
- Declare a loop if a pattern of length `L ∈ {1,2,3}` repeats `R=3` times with **no increase in state version** across repeats.

Action blocking policy:

- Maintain a blocklist of up to `M=10` blocked `(sig, err)` pairs.
- If the model proposes an action whose `(sig, err_prev)` is currently blocked and `v_t` has not changed since the last time it was executed, **do not execute it**. Instead, append an observation:
  - `ACTION_BLOCKED: this command previously produced the same error without any environment change; propose a different command or change the environment state first.`

- **Fallback mechanism**: blocks expire after `T=20` agent steps; after expiry, allow the action once with a warning, then re-block if it repeats again without progress.

This converts loop detection into action-space shaping, aiming to recover rather than terminate.

### Key Innovations

1. **Progress-conditioned action blocking (recovery instead of termination):** instead of stopping on loops, LASCon blocks only the specific repeated action+error signature under unchanged state version.

2. **CLI-specific structured condensation:** LASCon deterministically preserves commands, exit codes, and error lines while offloading full logs to disk, enabling long runs without relying on lossy free-form summaries.

3. **Mechanism-level evaluation:** measure loop events directly (pattern repetition under unchanged state version) rather than only aggregate Pass@1.

---

## Related Work

### Field Overview

Terminal and software-engineering agent benchmarks (e.g., Terminal-Bench, SWE-bench) use executable environments and automated tests to measure end-to-end agent performance. These benchmarks commonly induce long trajectories where token cost, context-window management, and failure recovery strategies become major determinants of success.

Context management for long-horizon agents includes observation masking/windowing, LLM summarization condensers, and learned compression policies. Recent evidence suggests that simple observation masking can match LLM summarization for SWE agents and can avoid side effects like longer trajectories, motivating deterministic condensation baselines.

Loop and stuck detection is widely implemented in agent frameworks, but is often used as a termination guardrail rather than a recovery mechanism. LASCon focuses on a narrow but important intervention: blocking exact repeated actions when there is evidence of no progress.

### Related Papers

- **[CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion](./references/CLI-Gym-Scalable-CLI-Task-Generation-via-Agentic-Environment-Inversion/meta/meta_info.txt)**: Introduces large-scale environment-repair tasks; reports loop failures and context-length exceed failures for OpenHands-based CLI agents.
- **[Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces](./references/Terminal-Bench-Benchmarking-Agents-on-Hard-Realistic-Tasks-in-Command-Line-Interfaces/meta/meta_info.txt)**: Defines a realistic CLI benchmark with time limits and automated verification; provides cost/trajectory analyses.
- **[The OpenHands Software Agent SDK](./references/The-OpenHands-Software-Agent-SDK-A-Composable-and-Extensible-Foundation-for-Production-Agents/meta/meta_info.txt)**: Provides the event-sourced agent architecture including condensers and stuck detection.
- **[SWE-agent](https://arxiv.org/abs/2405.15793)**: Shows interface/history processing (including collapsing older observations) improves SWE-bench performance.
- **[Simple Observation Masking Is as Efficient as LLM Summarization](https://arxiv.org/abs/2508.21433)**: Finds observation masking can match LLM summarization for SWE agents, motivating deterministic condensation.
- **[AgentDiet](https://arxiv.org/abs/2509.23586)**: Reduces agent trajectory tokens by rewriting older steps with an external model; shows large token savings without degrading pass rate.
- **[Active Context Compression (Focus)](https://arxiv.org/abs/2601.07190)**: Uses exploration “phases” and summaries to reduce long-horizon cost; shows savings with no accuracy loss on a small SWE-bench subset.
- **[SWE-Pruner](https://arxiv.org/abs/2601.16746)**: Task-aware line-level pruning for code context in SWE agents.
- **[Acon](https://arxiv.org/abs/2510.00615)**: Learns improved condensation guidelines from success/failure contrast pairs.
- **[Context as a Tool (CaT)](https://arxiv.org/abs/2512.22087)**: Treats context management as an explicit tool/action for SWE agents; trains a compressor.
- **[MemAct](https://arxiv.org/abs/2510.12635)**: Adds memory-edit actions and trains agents to prune/insert memory state.
- **[MEM1](https://arxiv.org/abs/2506.15841)**: Trains agents to maintain bounded memory by pruning past context each turn.
- **[ReSum](https://arxiv.org/abs/2509.13313)**: Uses periodic summarization for long-horizon web agents.
- **[Lost in the Maze (SLIM)](https://arxiv.org/abs/2510.18939)**: Shows long-horizon failures in search agents often stem from poor information management.
- **[AgentBench](https://arxiv.org/abs/2308.03688)**: Evaluates LLM agents across environments and identifies repetition/long-horizon reasoning as common failure sources.
- **[SWE-bench](https://arxiv.org/abs/2310.06770)**: Repository-level issue resolution benchmark with test-based verification.
- **[InterCode](https://papers.neurips.cc/paper_files/paper/2023/hash/4b175d846fb008d540d233c188379ff9-Abstract-Datasets_and_Benchmarks.html)**: Interactive coding benchmark highlighting multi-step correction and context challenges.
- **[ReAct](https://openreview.net/forum?id=WE_vluYUL-X)**: Thought–action–observation agent loop; analyzes failure modes including repetitive loops.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Uses reflection across attempts to improve success; contrasts with scaffold-level enforcement.
- **[SWE-bench Pro](https://arxiv.org/abs/2509.16941)**: Reports failure categories including context overflow and repetitive file reading.
- **[RE-TRAC](https://arxiv.org/abs/2602.02486)**: Uses recursive trajectory compression across multiple attempts for deep-search agents.
- **[AgentFold](https://arxiv.org/abs/2510.24699)**: Uses proactive context folding for long-horizon web agents.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| CLI/SWE benchmarks | Executable tasks + automated verifiers | [Terminal-Bench](./references/Terminal-Bench-Benchmarking-Agents-on-Hard-Realistic-Tasks-in-Command-Line-Interfaces/meta/meta_info.txt), [SWE-bench](https://arxiv.org/abs/2310.06770), [InterCode](https://papers.neurips.cc/paper_files/paper/2023/hash/4b175d846fb008d540d233c188379ff9-Abstract-Datasets_and_Benchmarks.html) | Pass@k, solve rate, cost/time | Expensive; long transcripts complicate diagnosis |
| Deterministic context reduction | Mask/collapse older tool outputs | [SWE-agent](https://arxiv.org/abs/2405.15793), [Simple Observation Masking](https://arxiv.org/abs/2508.21433) | SWE-bench variants | Can drop details needed later; not loop-specific |
| LLM summarization condensers | Replace history with summaries | [OpenHands SDK](./references/The-OpenHands-Software-Agent-SDK-A-Composable-and-Extensible-Foundation-for-Production-Agents/meta/meta_info.txt), [ReSum](https://arxiv.org/abs/2509.13313) | SWE-bench, web-agent benchmarks | Lossy; summary quality varies; may hide failure signals |
| Learned pruning/compression | Train models/policies to compress context | [Acon](https://arxiv.org/abs/2510.00615), [SWE-Pruner](https://arxiv.org/abs/2601.16746), [CaT](https://arxiv.org/abs/2512.22087) | SWE-bench | Requires training/data; may overfit |
| Trajectory reduction by reflection | Rewrite older steps | [AgentDiet](https://arxiv.org/abs/2509.23586), [Focus](https://arxiv.org/abs/2601.07190) | SWE-bench | Extra overhead; may remove needed info |
| Loop detection | Detect repetition patterns | [AgentBench](https://arxiv.org/abs/2308.03688), [OpenHands SDK](./references/The-OpenHands-Software-Agent-SDK-A-Composable-and-Extensible-Foundation-for-Production-Agents/meta/meta_info.txt) | Diverse | Often terminates rather than recovers; false positives |

### Closest Prior Work

1. **OpenHands condenser + stuck detection** ([OpenHands SDK](./references/The-OpenHands-Software-Agent-SDK-A-Composable-and-Extensible-Foundation-for-Production-Agents/meta/meta_info.txt)) condenses history using LLM summaries and terminates on stuck patterns. LASCon differs by using deterministic CLI condensation and by replacing termination with progress-conditioned action blocking.

2. **Simple Observation Masking (JetBrains 2025)** ([Simple Observation Masking](https://arxiv.org/abs/2508.21433)) shows that masking older tool outputs can match LLM summarization for SWE agents. LASCon adopts deterministic masking but targets terminal logs and adds a loop controller.

3. **SWE-agent observation collapsing** ([SWE-agent](https://arxiv.org/abs/2405.15793)) collapses older observations to reduce token costs. LASCon adds a structured scratchpad plus progress-conditioned action blocking.

4. **AgentDiet** ([AgentDiet](https://arxiv.org/abs/2509.23586)) rewrites older trajectory steps using an external model to reduce tokens. LASCon avoids external rewriting and targets loop recovery explicitly.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [OpenHands SDK](./references/The-OpenHands-Software-Agent-SDK-A-Composable-and-Extensible-Foundation-for-Production-Agents/meta/meta_info.txt) | LLM summarization condenser + termination-based stuck detection | Summaries can lose exact CLI details; termination is not recovery | Deterministic CLI condensation + action blocking | Keeps operational tokens and forces alternative actions instead of stopping |
| [Simple Observation Masking](https://arxiv.org/abs/2508.21433) | Mask older observations to save tokens | Not loop-specific; no recovery | Add progress-conditioned blocklist | Prevents repeated command+error loops under unchanged state |
| [SWE-agent](https://arxiv.org/abs/2405.15793) | Collapses older observations | No explicit loop recovery | Add loop controller + scratchpad | Forces exploration when stuck |
| [AgentDiet](https://arxiv.org/abs/2509.23586) | External model rewrites old steps | Extra model calls; not loop-specific | Rule-based condensation + block list | Lower overhead and targets repetition |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen3-32B | 32B | (API available in this repo as `Qwen/Qwen3-32B`) | Aligns with CLI-Gym’s OpenHands-evaluated baselines; likely to exhibit loop failures |

**Training Data (if applicable):**

No training data needed - inference only.

**Inference / scaffold configuration:**
- Decoding: temperature = 0 (greedy), max context = 128k (match CLI-Gym)
- Agent framework: OpenHands
- Task termination: Terminal-Bench time limit per task

**Resource Estimate**:
- **Compute budget**: No training. Primary cost is running Dockerized Terminal-Bench tasks + LLM inference.
- **API usage**: Terminal-Bench reports that full Terminal-Bench 2.0 runs can cost $1–$100 depending on model, and that a small number of tasks can consume extremely large token budgets ([Terminal-Bench](./references/Terminal-Bench-Benchmarking-Agents-on-Hard-Realistic-Tasks-in-Command-Line-Interfaces/meta/meta_info.txt)). We therefore start with a 30-task subset and only scale up if the impact gate passes.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| Terminal-Bench 2.0 | 89 Dockerized CLI tasks with outcome-based tests and per-task time limits | Pass@1; tokens/task; tool-calls/task; loop-event rate; context-limit errors | test | https://github.com/laude-institute/terminal-bench | Use official evaluation harness; run OpenHands agent inside task container |

**Internal metrics (logged from OpenHands event stream):**
- `loop_events`: count of PLC-triggered loop detections
- `blocked_actions`: count of actions blocked
- `context_overflow_events`: count of context-length-exceeded errors

**Loop-induced failure definition (for analysis):**
A task is a loop-induced failure if PLC detects ≥1 loop event and the final outcome is FAIL/TIMEOUT.

### Main Results

#### Conditions (3 main variants)

All conditions use the same base model and decoding; only the scaffold changes.

1. **A: OpenHands default**: default condenser (LLMSummarizingCondenser) and default stuck detector behavior.

2. **B: Prompt-only baseline**: OpenHands default + an additional system instruction: “Do not repeat the same command if it produced the same error and you have not changed the environment; instead, change state or try a different diagnostic.” This tests whether prompt guidance alone matches LASCon.

3. **C: LASCon**: replace the condenser with DCC and replace termination-based stuck handling with PLC action blocking. Use the same prompt as (B) so the comparison isolates code-level enforcement beyond prompt guidance.

#### Early-stop / decision rules (verification efficiency)

- **Impact gate (headroom check)**: Run (A) on a fixed 30-task subset. If the measured loop-induced failure rate is <5% (using PLC passively on (A) trajectories for consistent measurement), stop as low-impact for Terminal-Bench.

- **Prompt sufficiency gate**: If (B) matches (C) on Pass@1 and loop-induced failures (no meaningful difference), refute the need for hard blocking (the result would indicate prompting is sufficient).

- **Safety gate**: If (C) reduces Pass@1 relative to (B) (e.g., due to over-blocking) or produces frequent “blocked-action deadlocks”, refute the hard-blocking design and keep only prompt-level guidance as a recommended practice.

#### Results Table

| Method | Base Model | Benchmark | Pass@1 | Loop-induced failures | Tokens / task | Source | Notes |
|--------|------------|-----------|--------|------------------------|--------------|--------|-------|
| A: OpenHands default | Qwen3-32B | Terminal-Bench 2.0 | **5.7%** (Pass@1; higher is better) | **TBD** | **TBD** | [CLI-Gym](./references/CLI-Gym-Scalable-CLI-Task-Generation-via-Agentic-Environment-Inversion/meta/meta_info.txt) | CLI-Gym reports Qwen3-32B evaluated with OpenHands under temperature=0 and 128,000-token max context. We will re-run to ensure config parity. |
| B: Prompt-only baseline | Qwen3-32B | Terminal-Bench 2.0 | **TBD** | **TBD** | **TBD** | - | Tests “just prompt it” alternative |
| **C: LASCon** | Qwen3-32B | Terminal-Bench 2.0 | **TBD** | **TBD** | **TBD** | - | Deterministic condensation + action blocking |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| (Optional) DCC-only | Use DCC but disable PLC (keep prompt baseline) | If most gains come from loop blocking, DCC-only should reduce tokens but not strongly change loop-induced failures |
| (Optional) PLC-only | Enable PLC but keep default condenser (keep prompt baseline) | If PLC alone is sufficient, it should reduce loop-induced failures even without DCC |

### Analysis (Optional)

- **Failure mode shifts**: categorize failures into loop-induced, context overflow, and other (test fail, tool error) to see whether LASCon trades one failure mode for another.
- **Deadlock diagnostics**: quantify how often PLC blocks >k consecutive actions without state change.

---

## Success Criteria

**Criterion 1: Loop suppression beyond prompting**
- Hypothesis: LASCon (C) reduces loop-induced failures compared to the prompt-only baseline (B).
- Validation: fewer loop-induced failures and no decrease in Pass@1.

**Criterion 2: Efficiency under fixed time limits**
- Hypothesis: LASCon reduces tokens/task and reduces context overflow events.
- Validation: reduced tokens/task and fewer context overflow events, with stable or improved Pass@1.

---

## Impact Statement

If LASCon works, developers deploying terminal agents (especially open-weight models with weaker intrinsic self-correction) would adopt progress-conditioned action blocking and deterministic CLI condensation as default scaffold components to reduce runaway costs and improve success on long-horizon CLI benchmarks.

---

## References

- [CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion](./references/CLI-Gym-Scalable-CLI-Task-Generation-via-Agentic-Environment-Inversion/meta/meta_info.txt) - Lin et al., 2026
- [Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces](./references/Terminal-Bench-Benchmarking-Agents-on-Hard-Realistic-Tasks-in-Command-Line-Interfaces/meta/meta_info.txt) - Merrill et al., 2026
- [The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents](./references/The-OpenHands-Software-Agent-SDK-A-Composable-and-Extensible-Foundation-for-Production-Agents/meta/meta_info.txt) - Wang et al., 2025
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793) - Yang et al., 2024
- [Simple Observation Masking Is as Efficient as LLM Summarization](https://arxiv.org/abs/2508.21433) - JetBrains Research, 2025
- [AgentDiet: Improving the Efficiency of LLM Agent Systems through Trajectory Reduction](https://arxiv.org/abs/2509.23586) - Xiao et al., 2025
- [Active Context Compression: Autonomous Memory Management in LLM Agents](https://arxiv.org/abs/2601.07190) - Verma, 2026
- [SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents](https://arxiv.org/abs/2601.16746) - Wang et al., 2026
- [Acon: Optimizing Context Compression for Long-horizon LLM Agents](https://arxiv.org/abs/2510.00615) - Kang et al., 2025
- [Context as a Tool: Context Management for Long-Horizon SWE-Agents](https://arxiv.org/abs/2512.22087) - Liu et al., 2025
- [MemAct: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2510.12635) - Zhang et al., 2025
- [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/abs/2506.15841) - Zhou et al., 2025
- [ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization](https://arxiv.org/abs/2509.13313) - Wu et al., 2025
- [Lost in the Maze: Overcoming Context Limitations in Long-Horizon Agentic Search](https://arxiv.org/abs/2510.18939) - Yen et al., 2025
- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) - Liu et al., 2023
- [SWE-bench: Can Language Models Resolve Real-world Github Issues?](https://arxiv.org/abs/2310.06770) - Jimenez et al., 2024
- [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback](https://papers.neurips.cc/paper_files/paper/2023/hash/4b175d846fb008d540d233c188379ff9-Abstract-Datasets_and_Benchmarks.html) - Yang et al., 2023
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://openreview.net/forum?id=WE_vluYUL-X) - Yao et al., 2023
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - Shinn et al., 2023
- [SWE-bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?](https://arxiv.org/abs/2509.16941) - (see arXiv), 2025
- [RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents](https://arxiv.org/abs/2602.02486) - Zhang et al., 2026
- [AgentFold: Long-Horizon Web Agents with Proactive Context Folding](https://arxiv.org/abs/2510.24699) - Alibaba Tongyi Lab, 2025
