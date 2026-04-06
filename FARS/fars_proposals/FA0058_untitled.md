# untitled

# Consensus Verification for Irreversible Actions in Text World Models

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used as **interactive agents**: they observe a text description of an environment, choose an action, and receive a new observation. In many interactive settings, a single wrong action can irreversibly end an episode (e.g., submitting a purchase, sending an email, deleting a file).

A practical way to reduce such failures is **pre-execution verification**: before executing a high-stakes action in the real environment, the agent first checks whether the action is likely to succeed. For **text-based environments**, a natural verifier is a **learned world model**: a model that predicts the next environment observation (and optionally a termination/success signal) given the dialogue history and a candidate action.

**From Word to World** (Word2World) trains LLM-based world models for several text environments and demonstrates a simple verifier in **WebShop** (a text-only shopping environment with 500 held-out test tasks): before committing to the irreversible checkout action, the agent queries a learned world model to predict whether checkout will succeed.

### The Problem

Word2World reports an important and non-obvious failure mode: increasing the **verification budget** (the maximum number of world-model queries allowed before committing to checkout) can yield **non-monotonic** returns for several acting agents. Table 4 shows multiple examples:

- **Gemini-2.5-flash**: 25.0% (B=0) → 31.0% (B=2) → 28.0% (B=10)
- **Claude-sonnet-4.5**: 61.0% (B=0) → 65.0% (B=4) → 62.0% (B=50)
- **GPT-5**: 51.0% (B=0) → 53.77% (B=4) → 51.50% (B=50)
- **GPT-4-turbo**: 17.73% (B=0) → 33.33% (B=2) → 25.60% (B=50)

In contrast, **GPT-4o** improves monotonically from 29.36% (B=0) to 36.70% (B=50). This suggests that verification-budget non-monotonicity is **agent-dependent** rather than a rare artifact.

This matters in practice because verification budget is a natural knob: when an agent makes costly irreversible mistakes, developers often add more verifier calls. A verification policy that can *reduce* success when the budget is increased is a deployment hazard and wastes compute.

Word2World hypothesizes that the sequential "verify-and-retry" loop changes the agent’s trajectory context and action distribution, creating **distribution shift** (the agent visits states/actions during verification that differ from those seen when training or validating the world model), which can weaken the alignment between imagined and real outcomes.

This raises a concrete, decision-relevant question: **How should we spend a fixed world-model verification budget so that more verification compute does not backfire?**

### Key Insight and Hypothesis

**Key insight:** In Word2World, extra verification budget is spent as **more sequential verification cycles**: the agent proposes checkout, the verifier predicts failure, the agent continues acting (changing the state/trajectory), and later proposes checkout again. If the regression at large budgets is driven by the repeated sequential loop (trajectory/context shift), then a natural alternative is to spend the same number of verifier calls in **fewer verification cycles**, by using **parallel multi-sample verification** at each cycle.

**Hypothesis:** For a fixed budget of world-model calls, concentrating the calls into a small number of **parallel rollouts + consensus gating** (instead of spreading them across many sequential verify-and-retry cycles) will improve WebShop success at high budgets where Word2World exhibits regressions.

**Why this could be wrong:** (i) the dominant issue might be intrinsic world-model mismatch (low fidelity), so additional samples do not help; (ii) sequential retries might be beneficial because they implicitly encourage the agent to gather more real evidence before checkout.

**Decision rule (pre-registered):** On WebShop, pick a budget where Word2World shows a regression for at least one agent (default: **B=10**, where Gemini-2.5-flash drops from 31.0% at B=2 to 28.0% at B=10). If chunked consensus (B) does **not** improve success rate over the sequential baseline (A) under the same harness (bootstrap confidence interval for the mean difference includes 0), we reject the hypothesis.

**Statistical fragility plan:** Because effect sizes in Table 4 are often 3–6 percentage points on 500 tasks, we will (i) run **3 independent seeds** for each condition (different sampling seeds for the acting agent and world model), and (ii) report both per-seed outcomes and a bootstrap confidence interval over tasks for the mean difference. If variance dominates (CI width too large to resolve a 3% absolute effect), the experiment outcome is recorded as "inconclusive under current budget" and we do not proceed to additional ablations.

---

## Proposed Approach

### Overview

We propose **Chunked Consensus Verification** for irreversible actions:

- Let **B** be the maximum number of world-model calls allowed per episode for verification.
- Choose a small fixed number of verification cycles **M** (e.g., M=2).
- When the agent proposes an irreversible action (WebShop checkout), run **K = B / M** independent stochastic world-model simulations from the current state, parse the predicted binary success signal, and estimate \(\hat p\) (fraction of simulated successes).
- Commit to checkout if \(\hat p \ge \tau\) (default \(\tau=0.5\)). Otherwise, block checkout and let the agent continue.
- Allow at most **M** such verification cycles; total world-model calls for verification are bounded by **B**.

This spends the same budget **B** but reduces the number of sequential verifier-agent interaction cycles that could induce distribution shift.

### Method Details

**World model interface.** Following Word2World’s formalization, the world model \(W\) maps the current history and candidate action to a predicted next observation \(S'\) and a binary success/termination indicator \(R'\in\{0,1\}\) (see `./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/World Model.md`). For WebShop checkout, \(R'\) is the key signal.

**Condition A (baseline, sequential verify-and-retry).** Word2World-style gating with maximum of **B** verifier calls: each time the agent proposes checkout, query the world model once; execute checkout only if \(R'=1\), otherwise continue.

**Condition B (ours, chunked consensus).** Use **M** cycles, each consuming **K=B/M** world-model calls from the same state:
\[
\hat p = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}[R'_k = 1],\quad \text{commit if } \hat p\ge\tau.
\]
Only a scalar (PASS/FAIL; optionally \(\hat p\) logged for analysis) is used to gate execution.

**Condition C (mechanism control: chunked but no aggregation).** Same as B in terms of cycle count and world-model calls, but **do not use consensus**: generate \(K\) rollouts but decide PASS/FAIL using only the first rollout’s \(R'\) (discard the other \(K-1\) outcomes). This isolates whether any gains come from reducing sequential verification cycles rather than from the aggregation rule.

### Key Innovations

- A minimal **mechanism test** for non-monotonic verification budgets in world-model-gated agents: hold total verifier calls fixed while changing how the budget is distributed over sequential cycles.
- A simple, training-free verification policy that can be added to existing world-model agent systems.

---

## Related Work

### Field Overview

This proposal connects three lines of work: (i) **LLMs as world models** for interactive text environments, (ii) **simulation-based planning and verification** for agent decision making, and (iii) **multi-sample aggregation** (self-consistency / voting) as a way to improve reliability at test time. A recurring limitation is that learned simulators can be accurate one step ahead but drift over longer interactions, so practical methods often restrict simulation horizon and must allocate limited simulation compute carefully.

### Related Papers

(Proposal-local references are used when available; otherwise arXiv/OpenReview URLs.)

- **[From Word to World](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt)**: Trains text world models and introduces pre-execution checkout verification in WebShop, including the non-monotonic budget phenomenon motivating this proposal.
- **[WebDreamer](./references/Is-Your-LLM-Secretly-a-World-Model-of-the-Internet-Model-Based-Planning-for-Web-Agents/meta/meta_info.txt)**: Uses an LLM world model to simulate short trajectories for *general planning* over candidate actions and averages trajectory scores over multiple samples; it does not study how to allocate a fixed verification-call budget for a single irreversible action, nor the failure mode where increasing a verification budget reduces success.
- **[DynaWeb](./references/DynaWeb/meta/meta_info.txt)**: Uses imagined web rollouts as additional data to train web agents with online RL.
- **[Agentic Test-Time Scaling for WebAgents](./references/Agentic-Test-Time-Scaling-for-WebAgents/meta/meta_info.txt)**: Studies non-monotonic test-time scaling and uses vote-derived uncertainty to allocate compute for action selection.
- **[Active Epistemic Control](./references/Active-Epistemic-Control-for-Query-Efficient-Verified-Planning/meta/meta_info.txt)**: Separates grounded facts from model beliefs for verified planning, highlighting limits of ungrounded simulation.
- **[WebRollback](./references/WebRollback-Enhancing-Web-Agents-with-Explicit-Rollback-Mechanisms/meta/meta_info.txt)**: Adds explicit rollback mechanisms for web agents and discusses the challenge of irreversible actions.

Additional related work (external links):

- **[WebShop](https://arxiv.org/abs/2207.01206)**: A text-only web interaction benchmark (12,087 instructions; 500 test tasks) used for evaluating shopping agents.
- **[Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/)**: Introduces ByteSized32 state-prediction tasks and shows compounding-error limits for LLM simulators.
- **[Evaluating World Models with LLM for Decision Making](https://arxiv.org/abs/2411.08794)**: Evaluates LLM world models for action proposal and verification in decision making.
- **[Web Agents with World Models (WMA)](https://arxiv.org/abs/2410.13232)**: Uses structured webpage representations (e.g., accessibility trees) for world modeling in web navigation.
- **[RLVR-World](https://arxiv.org/abs/2505.13934)**: Trains world models with reinforcement learning with verifiable rewards.
- **[Making Large Language Models into World Models with Precondition and Effect Knowledge](https://arxiv.org/abs/2409.12278)**: Builds world models via explicit precondition/effect inference.
- **[TEXT2WORLD](https://arxiv.org/abs/2502.13092)**: A benchmark for generating executable symbolic world models from text.
- **[RADI](https://openreview.net/forum?id=cPo2iS6lwP)**: Uses LLM imagination/world modeling for robotic action decomposition.
- **[Dreamer](https://arxiv.org/abs/1912.01603)**: A model-based RL method that learns latent world models for long-horizon control.
- **[MuZero](https://arxiv.org/abs/1911.08265)**: Learns a model for planning without modeling environment dynamics explicitly.
- **[World Models](https://arxiv.org/abs/1803.10122)**: Early work on learning compact generative world models for planning.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: Interleaves reasoning and acting, a common baseline style for interactive LLM agents.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Uses episodic self-feedback to improve agent performance, illustrating how additional trajectory text can change policy behavior.
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)**: Multi-sample aggregation for reasoning, conceptually similar to consensus over stochastic rollouts.
- **[WebArena](https://arxiv.org/abs/2307.13854)**: A live-web benchmark emphasizing the difficulty of irreversible actions (though it requires browser execution).
- **[WebVoyager](https://arxiv.org/abs/2401.13919)**: A web navigation benchmark with automatic evaluation; highlights evaluation challenges under website changes.
- **[Mind2Web](https://arxiv.org/abs/2306.06070)**: Large-scale web interaction dataset used for training and evaluation of web agents.
- **[WebSuite](https://arxiv.org/abs/2406.01623)**: Diagnoses why web agents fail and categorizes common failure modes.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Text world models | Predict next observation and success from trajectories | Word2World; ByteSized32 | WebShop, ALFWorld, SciWorld, TextWorld | Long-horizon drift; behavior shift sensitivity |
| Simulation-based planning | Simulate candidate trajectories and score them | WebDreamer; MuZero-style search | WebShop, WebVoyager | Simulation horizon is short; expensive |
| Imagination-driven training | Use simulated rollouts as training data | DynaWeb | WebArena/WebVoyager-style tasks | Needs careful mixing with real data |
| Compute allocation at test time | Use uncertainty/voting to decide where to spend compute | CATTS; self-consistency | Web agents, reasoning benchmarks | Can be non-monotonic; confounded by policy shift |

### Closest Prior Work

1. **Word2World**: Introduces checkout verification and shows non-monotonic success vs verification budget, but does not test whether regressions are caused by the sequential verify-and-retry loop structure versus the total amount of verifier compute.
2. **WebDreamer**: Uses multi-sample simulation for general planning, but does not study verification budgets for a single irreversible action or the failure mode where more verification compute can reduce success.
3. **CATTS**: Uses vote-derived uncertainty for allocating compute in action selection, but it does not target learned world-model verifiers for irreversible-action gating.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Word2World | Sequential world-model gating for checkout; sweeps verification budget | More budget can hurt; mechanism unclear | Redistribute the same budget into fewer cycles with parallel rollouts | Avoid regressions from repeated sequential verification cycles |
| WebDreamer | Multi-sample simulation to score candidate trajectories | Not a commit/no-commit verifier; does not study budget regressions | Apply multi-sample to irreversible-action verification and test budget allocation | Cleaner mechanism test + minimal code change |
| CATTS | Vote uncertainty for action selection compute allocation | No world-model verifier; different decision point | Use vote/consensus over world-model rollouts for checkout gating | Targets a different failure mode (verification-induced regressions) |

---

## Experiments

### Experimental Setup

**Feasibility note (infrastructure):** This proposal uses **text-only** environments (WebShop as implemented in the Word2World repo / AgentGym-style servers). It does **not** require a browser, GUI automation, or live websites.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Gemini-2.5-flash (acting agent) | API | https://ai.google.dev/ | Used in Word2World Table 4; enables direct comparison |
| WorldModel-Webshop-Qwen2.5-7B (verifier world model) | 7B | https://huggingface.co/X1AOX1A/WorldModel-Webshop-Qwen2.5-7B | Released by Word2World project; outputs next-state + binary success |

**Training Data (if applicable):**

No training data needed — **inference only**.

**Other Resources (if applicable):**

- Word2World codebase + evaluation protocol: https://github.com/X1AOX1A/Word2World

**Resource Estimate** (must fit ≤768 A100 GPU-hours):

- **World model inference**: run the 7B world model locally (1×A100-80GB is sufficient). Verifier calls per episode are ≤B (default B=10).
- **Agent inference (API)**: dominant cost is Gemini calls across episodes.
  - Expected scale: WebShop has **500 test tasks**; episodes typically involve multiple turns (tens of actions). Verification is only triggered near checkout.
  - If full 500-task runs are too slow/costly, run a **100-task stratified subset** for the decisive comparison A/B/C, then (if promising) scale to all 500 tasks.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| WebShop (Word2World protocol) | Text-only shopping tasks; checkout is irreversible and ends the episode | **Success rate** (% tasks completed; higher is better); verifier calls per episode (lower is cheaper) | test (500 tasks) | via Word2World repo | Word2World evaluation scripts |

### Main Results

#### Baselines table (evidence + comparability)

Published baseline numbers are taken from Word2World Table 4 (see `./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/7.1 Can World Models Prevent Irreversible Mistakes.md`). The verification module should re-run A under the same harness used for B/C; published values serve as a sanity check.

| Method | Description | Settings | Results | Reference |
|---|---|---|---|---|
| No verification (B=0) | Agent checks out without any world-model gating | Agent=Gemini-2.5-flash; Benchmark=WebShop test | Success **25.0%** | Word2World Table 4 |
| Sequential verify-and-retry (A, B=2) | Word2World-style: 1 verifier call per checkout proposal; retry later if predicted fail | Agent=Gemini-2.5-flash; max verifier calls B=2 | Success **31.0%** | Word2World Table 4 |
| Sequential verify-and-retry (A, B=10) | Same as above at a larger budget (shows non-monotonic regression) | Agent=Gemini-2.5-flash; max verifier calls B=10 | Success **28.0%** | Word2World Table 4 |
| **Chunked consensus (B, ours)** | Spend B calls in M cycles with K=B/M parallel rollouts + majority gating | Agent=Gemini-2.5-flash; B=10; M=2; K=5; τ=0.5 | **TBD** (to be verified) | This proposal |
| Chunked-no-aggregation (C) | Same calls/cycles as ours but decide from only 1 rollout (discard the rest) | Agent=Gemini-2.5-flash; B=10; M=2; K=5; τ=0.5 | **TBD** (to be verified) | This proposal |
| Sequential verify-and-retry (A, B=50) | High-budget regime where several agents regress | Agent=Claude-sonnet-4.5; max verifier calls B=50 | Success **62.0%** (vs 65.0% at B=4) | Word2World Table 4 |

#### Implementation comparability notes

- All methods use the **same acting agent**, **same WebShop split**, and **same world-model checkpoint**.
- For stochastic rollouts, use a fixed decoding configuration for the world model across all conditions (default: temperature=0.7, top_p=0.9; or the Word2World repo defaults if explicitly specified).
- Total verifier calls are capped by B per episode for all conditions.

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Vary M (cycles) | M∈{1,2,5} at fixed B=10 | If regressions are about sequential cycles, smaller M should help |
| Deterministic verifier | Set temperature=0 for world-model decoding | Consensus advantage should shrink if stochasticity was essential |

### Analysis (Optional)

- Log how many verification cycles are actually used per episode in A vs B/C; correlate cycles with failure cases.
- For B, evaluate calibration of \(\hat p\): does larger \(\hat p\) correlate with higher real checkout success?

---

## Success Criteria

**Criterion 1: Prevent regressions at high verification budgets**
- Hypothesis: At B=10 on WebShop, chunked consensus (B) > sequential verify-and-retry (A) in success rate.
- Validation: Bootstrap CI for (success(B) − success(A)) is strictly positive.

**Criterion 2: Identify whether consensus aggregation matters**
- Hypothesis: If aggregation is important, B > C (same calls/cycles but no aggregation).
- Validation: Bootstrap CI for (success(B) − success(C)) is strictly positive. If instead B ≫ A but B ≈ C, gains likely come primarily from reducing sequential verification cycles.

---

## Impact Statement

If successful, this work would provide a simple design rule for world-model-based pre-execution verification: **spend verification budget in a small number of parallel rollout batches rather than in many sequential verify-and-retry cycles**. This can make learned world-model verifiers safer to scale up in settings with irreversible actions, without retraining either the agent or the world model.

---

## References

- [From Word to World: Can Large Language Models be Implicit Text-based World Models?](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt)
- [Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents](./references/Is-Your-LLM-Secretly-a-World-Model-of-the-Internet-Model-Based-Planning-for-Web-Agents/meta/meta_info.txt)
- [DynaWeb](./references/DynaWeb/meta/meta_info.txt)
- [Agentic Test-Time Scaling for WebAgents](./references/Agentic-Test-Time-Scaling-for-WebAgents/meta/meta_info.txt)
- [Active Epistemic Control for Query-Efficient Verified Planning](./references/Active-Epistemic-Control-for-Query-Efficient-Verified-Planning/meta/meta_info.txt)
- [WebRollback: Enhancing Web Agents with Explicit Rollback Mechanisms](./references/WebRollback-Enhancing-Web-Agents-with-Explicit-Rollback-Mechanisms/meta/meta_info.txt)
- [WebShop](https://arxiv.org/abs/2207.01206)
- [Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/)
- [Evaluating World Models with LLM for Decision Making](https://arxiv.org/abs/2411.08794)
- [Web Agents with World Models (WMA)](https://arxiv.org/abs/2410.13232)
- [RLVR-World](https://arxiv.org/abs/2505.13934)
- [Making Large Language Models into World Models with Precondition and Effect Knowledge](https://arxiv.org/abs/2409.12278)
- [TEXT2WORLD](https://arxiv.org/abs/2502.13092)
- [RADI](https://openreview.net/forum?id=cPo2iS6lwP)
- [Dreamer](https://arxiv.org/abs/1912.01603)
- [MuZero](https://arxiv.org/abs/1911.08265)
- [World Models](https://arxiv.org/abs/1803.10122)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [Self-Consistency](https://arxiv.org/abs/2203.11171)
- [WebArena](https://arxiv.org/abs/2307.13854)
- [WebVoyager](https://arxiv.org/abs/2401.13919)
- [Mind2Web](https://arxiv.org/abs/2306.06070)
- [WebSuite](https://arxiv.org/abs/2406.01623)
