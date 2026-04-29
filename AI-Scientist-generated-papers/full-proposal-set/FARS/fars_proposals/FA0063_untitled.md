# untitled

# Intent Reconstruction Gate: Clarify Before Retrieval to Reduce Anchoring Under Ambiguity

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

LLM-based agents increasingly solve tasks by combining language reasoning with external tools such as retrieval, code execution, and databases. A recurring deployment failure mode is that the user request (or task prompt) is ambiguous or underspecified, but the agent still commits to a single interpretation and proceeds. This is costly: it wastes tool calls, produces incorrect outputs that appear plausible, and in high-stakes settings can look like **specification gaming** (selecting an interpretation of an underspecified request that is easier to satisfy but not what the user intended).

This failure mode is salient even in advanced research agents. For example, the Aletheia mathematics research agent reports that on 200 candidate solutions to open Erdős problems, **31.5%** were "technically correct under some interpretation" but only **6.5%** were "meaningfully correct" for the intended interpretation, and notes a tendency to "misinterpret the question in a way that is easiest to answer" when ambiguity exists (**[Towards Autonomous Mathematics Research](./references/Towards-Autonomous-Mathematics-Research/meta/meta_info.txt)**, Table 5).

A related symptom appears in interactive search agents: **InteractComp** (a benchmark where the agent can `search`, `interact` with a simulated user who holds disambiguating context, and then `answer`) evaluates agents on ambiguous queries where the answer is a lesser-known target and a popular distractor shares many attributes. Even strong models achieve low accuracy (e.g., GPT-5: **13.73%**) despite large headroom when given the hidden disambiguating context (**[InteractComp](./references/InteractComp-Evaluating-Search-Agents-With-Ambiguous-Queries/meta/meta_info.txt)**, Table 2).

### The Problem

We focus on a specific mechanism that can create "technically correct but wrong intent" behavior in tool-using agents: **anchoring to early candidate interpretations**.

In many agent stacks, retrieval happens early ("search-first"), and the first retrieved candidate (often a popular or easy-to-fit distractor) becomes an anchor that shapes subsequent information gathering. After seeing an anchor, the agent may:
- ask confirmatory questions that are compatible with the anchor rather than discriminative questions;
- underweight later clarification signals; and
- stop early with a plausible but incorrect interpretation.

Existing work on clarification largely targets *whether* to clarify and *what* to ask (e.g., uncertainty-based policies or **expected value of perfect information (EVPI)** criteria that select questions by their expected information gain), but it is less clear whether a simpler system-design intervention matters in practice:

> Does the **timing** of clarification relative to retrieval exposure causally affect accuracy, even when the total clarification budget is fixed?

If timing matters, it suggests a practical design rule for agent builders: **block exposure to candidate interpretations until the agent has produced a small number of intent-reconstruction queries**.

### Key Insight and Hypothesis

**Key insight.** Early retrieval results act as an anchor that changes what the agent asks next. Even with the same number of clarification interactions, a "search-first" agent may spend those interactions validating the anchor rather than disambiguating the query.

**Hypothesis.** Under a fixed interaction budget, enforcing a short **Intent Reconstruction Gate (IRG)** (k clarification turns *before* retrieval exposure) increases accuracy relative to search-first baselines.

We could be wrong for several reasons:
1. The effect might be fully explained by a cheaper baseline: randomizing or reordering candidate presentation.
2. Models might already ask discriminative questions regardless of retrieval exposure, making timing irrelevant.
3. Any observed gains might come from responder artifacts (how yes/no/unknown is generated) rather than the agent behavior.

Our experiment is designed to directly compare clarification timing under identical budgets and with an explicit "candidate order" baseline.

---

## Proposed Approach

### Overview

We propose **Intent Reconstruction Gate (IRG)**: an inference-time wrapper for tool-using agents on ambiguous tasks.

Given an initial ambiguous query q, IRG forces the agent to spend a small, fixed budget k on intent reconstruction *before* seeing candidate interpretations from retrieval. Concretely:
1. The agent issues k clarification probes (yes/no questions or hypotheses) aimed at eliciting distinctive attributes of the intended target.
2. Only after these probes does the agent see retrieval candidates and produce a final answer.

IRG is intentionally simple: it is not a learned policy and does not require training. The core research question is whether this structural intervention changes outcomes.

### Method Details

We will test IRG on a fully-automated proxy benchmark derived from InteractComp.

**Data.** Each InteractComp instance contains:
- an ambiguous question q,
- a hidden context c containing distinctive attributes of the true target,
- a correct answer a (target), and
- a distractor d (popular alternative).

**Environment (InteractComp-Anchor).** We define two tools:

1) `SEARCH(q) -> candidates` (deterministic, no web APIs)
- Returns a 2-item candidate list containing {d, a} in a specified order.
- The agent must output one of these candidates exactly as its final answer.

2) `INTERACT(hypothesis: str) -> {YES, NO, UNKNOWN}`
- The agent submits a single-sentence hypothesis about the intended target.
- The responder has access to hidden context c and replies based only on c.
- Automated responder implementation (preferred): **natural language inference (NLI)** model over (premise=c, hypothesis=h): entailment->YES, contradiction->NO, neutral->UNKNOWN. (An NLI model predicts whether a premise entails, contradicts, or is unrelated to a hypothesis.)

**Three-condition evaluation (k fixed).** For each instance, run exactly k interactions.

- **A. Search-first (distractor-first)**: `SEARCH` (returns [d, a]) -> `INTERACT` x k -> answer
- **B. IRG (clarify-first, distractor-first)**: `INTERACT` x k -> `SEARCH` (returns [d, a]) -> answer
- **C. Candidate-order baseline (target-first)**: `SEARCH` (returns [a, d]) -> `INTERACT` x k -> answer

Condition C is a "cheap alternative" baseline: if B only matches C, then gains may be explainable by changing candidate ordering rather than gating.

**Prompting / harness control.** The harness enforces action order by restricting available tools at each step (i.e., in condition B, `SEARCH` is unavailable until k interactions have been consumed).

### Key Innovations

- **Clarification timing as a first-class design axis**: separates "what to ask" from "when to ask" relative to retrieval exposure.
- **A minimal, fully-automated anchoring probe (InteractComp-Anchor)**: isolates candidate-exposure anchoring without requiring live web search.
- **Mechanism-aware baseline**: includes an explicit candidate-order baseline to test whether IRG adds value beyond reordering.

---

## Related Work

### Field Overview

There is substantial work on improving agent interaction under ambiguity and underspecification, including benchmarks that require clarification and methods that decide when to ask questions. Separately, work on cognitive biases and order effects in LLMs shows that the order of presented options can change model judgments.

Our proposal connects these threads: we study whether **retrieval-exposure order** (seeing candidate interpretations early) induces an anchoring effect that degrades interaction quality, and whether a simple gating intervention mitigates it. This differs from most clarification work, which optimizes question selection but does not isolate causal effects of exposure timing.

### Related Papers

- **[Towards Autonomous Mathematics Research](./references/Towards-Autonomous-Mathematics-Research/meta/meta_info.txt)**: Reports autonomous math research results and highlights specification gaming on ambiguous Erdős problems.
- **[InteractComp: Evaluating Search Agents With Ambiguous Queries](./references/InteractComp-Evaluating-Search-Agents-With-Ambiguous-Queries/meta/meta_info.txt)**: Benchmark for ambiguity-aware interactive search; shows large headroom and underuse of interaction.
- **[Clarify When Necessary: Resolving Ambiguity Through Interaction](https://aclanthology.org/2025.findings-naacl.306/)**: Introduces INTENT-SIM, an uncertainty-based criterion for when to ask clarifying questions.
- **[Structured Uncertainty guided Clarification for LLM Agents](https://arxiv.org/abs/2511.08798)**: Proposes SAGE-Agent using structured uncertainty and EVPI to select clarification questions; introduces ClarifyBench.
- **[ClarifyBench](https://openreview.net/forum?id=dc8ebScygC)**: Multi-turn tool-calling disambiguation benchmark with ambiguous and infeasible queries.
- **[MAC: A Multi-Agent Framework for Interactive User Clarification](https://openreview.net/forum?id=pDOtqzaZAf)**: Uses role-specialized agents to reduce redundant questions while improving task success.
- **[Learning to Ask: When LLMs Meet Unclear Instruction](https://arxiv.org/abs/2409.00557)**: Studies unclear tool instructions; proposes Ask-when-Needed prompting and NoisyToolBench.
- **[Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying Questions](https://arxiv.org/abs/2410.13788)**: Uses multi-turn preference signals to train clarification behavior.
- **[QuestBench](https://arxiv.org/abs/2503.22674)**: Benchmark for asking the right question to obtain missing information in reasoning tasks (logic/planning/math).
- **[When2Call: When (not) to Call Tools](https://arxiv.org/abs/2504.18851)**: Multiple-choice benchmark for deciding between tool use, follow-up questions, and abstention.
- **[Tell Me More! (IN3)](https://arxiv.org/abs/2402.09205)**: Benchmark for implicit user intention understanding via clarification.
- **[UserBench: An Interactive Gym Environment for User-Centric Agents](https://arxiv.org/abs/2507.22034)**: Underspecified long-horizon tasks; evaluates clarification strategies under interruption costs.
- **[UserRL: Training Interactive User-Centric Agent via Reinforcement Learning](https://arxiv.org/abs/2509.19736)**: RL training across multiple interactive gyms; analyzes reward shaping for proactive interaction.
- **[LHAW: Controllable Underspecification for Long-Horizon Tasks](https://arxiv.org/abs/2602.10525)**: Generates underspecified variants of long-horizon tasks; taxonomy of missing information.
- **[Pushing Forward Pareto Frontiers of Proactive Agents](https://arxiv.org/abs/2602.11351)**: Studies multi-objective training for agents that balance proactivity and user burden.
- **[ClariQ](https://arxiv.org/abs/2009.11352)**: Dataset for conversational search clarification question generation.
- **[Qulac](https://arxiv.org/abs/1905.00024)**: Conversational search dataset with clarifying questions.
- **[AmbigQA](https://arxiv.org/abs/2004.10645)**: Open-domain QA benchmark for ambiguous questions requiring multiple plausible answers.
- **[CambigNQ](https://arxiv.org/abs/2209.01076)**: Ambiguous natural questions with clarification and disambiguation.
- **[Corpus-informed Retrieval Augmented Generation of Clarifying Questions](https://arxiv.org/abs/2409.18575)**: Generates clarification questions grounded in retrieved evidence.
- **[RAC: Retrieval-Augmented Clarification](https://arxiv.org/abs/2601.11722)**: Faithful pre-retrieval clarification using preference optimization.
- **[Self-RAG](https://arxiv.org/abs/2310.11511)**: Retrieval-augmented generation with self-reflection and selective retrieval.
- **[BrowseComp](https://arxiv.org/abs/2504.12516)**: Benchmark for browsing agents; highlights search capability gaps.
- **[GAIA](https://arxiv.org/abs/2311.12983)**: Benchmark for general AI assistants; many tasks require tool use and multi-step reasoning.
- **[tau-bench](https://arxiv.org/abs/2406.12045)**: Tool-agent-user interaction benchmark in real-world domains.
- **[Behavioral and Attributional Evidence of Anchoring Bias in LLMs](https://arxiv.org/abs/2511.05766)**: Demonstrates anchoring bias via behavioral and attributional analyses.
- **[A Deep Dive into Order Effects in Large Language Models](https://arxiv.org/abs/2506.14092)**: Systematic study of primacy/recency and position biases.
- **[Cognitive Biases in Large Language Models: A Survey](https://arxiv.org/abs/2412.00323)**: Survey of cognitive-bias phenomena and mitigation strategies.
- **[Language Models Identify Ambiguities and Exploit Loopholes](https://arxiv.org/abs/2508.19546)**: Shows that models can detect ambiguity and selectively exploit it (specification gaming).

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Clarification benchmarks | Evaluate if agents seek disambiguating info | InteractComp; ClarifyBench; QuestBench; IN3; UserBench | Accuracy under interaction; coverage vs #questions | Often mixes question selection with timing and tool exposure |
| Clarification policies | Decide when/what to ask using uncertainty or planning | INTENT-SIM; SAGE-Agent; MAC; Ask-when-Needed; multi-turn preference learning | Tool-calling success; disambiguation F1; cost of questions | Rarely isolates causal impact of retrieval exposure order |
| Pre-retrieval clarification in RAG | Ask clarifying questions before retrieval to improve relevance/faithfulness | Corpus-informed CQ; RAC; conversational search CQ generation | Retrieval relevance; faithfulness to corpus | Typically optimizes retrieval, not anchoring to early candidates |
| Order effects / anchoring in LLMs | Output depends on order of presented options or context | Anchoring bias evidence; order-effects analyses; bias surveys | Controlled order-swaps | Usually not studied in interactive search/clarification loops |
| Specification gaming under ambiguity | Models exploit ambiguity or choose easy interpretations | Aletheia spec-gaming analysis; loophole exploitation | Open-ended tasks; synthetic ambiguity probes | Hard to evaluate automatically on open-ended domains |

### Closest Prior Work

1. **InteractComp** (**[InteractComp](./references/InteractComp-Evaluating-Search-Agents-With-Ambiguous-Queries/meta/meta_info.txt)**) is the closest benchmark setting: it already exposes that interaction is underused and that models struggle with ambiguity. Our work differs by (i) constructing a controlled variant to isolate anchoring from early retrieval exposure and (ii) testing a fixed clarification budget where only the *timing* differs.

2. **Pre-retrieval clarification for RAG** (e.g., **[RAC](https://arxiv.org/abs/2601.11722)**, **[Corpus-informed CQ](https://arxiv.org/abs/2409.18575)**) is conceptually similar in that it asks questions before retrieval. Our emphasis is different: we evaluate a causal mechanism (anchoring to early candidates) and include an explicit candidate-order baseline to test whether gating is more than a ranking trick.

3. **Structured clarification policies** like **[SAGE-Agent](https://arxiv.org/abs/2511.08798)** and **[Clarify When Necessary](https://aclanthology.org/2025.findings-naacl.306/)** focus on optimizing which questions to ask and when to clarify. We intentionally do not propose a new question-selection algorithm; instead we ask whether a simple structural wrapper (k questions before retrieval) changes outcomes.

4. **Anchoring / order-effects work** (e.g., **[Anchoring Bias](https://arxiv.org/abs/2511.05766)**, **[Order Effects](https://arxiv.org/abs/2506.14092)**) shows order dependence in static prompts. We test an analogous phenomenon in interactive tool loops, where order affects what the model asks next.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [InteractComp](./references/InteractComp-Evaluating-Search-Agents-With-Ambiguous-Queries/meta/meta_info.txt) | Benchmarks interactive search on ambiguous queries | Retrieval exposure and question timing are not isolated | Create InteractComp-Anchor and compare clarify-before vs search-before under fixed k | If anchoring drives failures, changing timing should measurably improve accuracy |
| [RAC](https://arxiv.org/abs/2601.11722) / [Corpus-informed CQ](https://arxiv.org/abs/2409.18575) | Pre-retrieval clarification for better retrieval | Not designed to test anchoring as a causal mechanism | Use a controlled candidate-list search tool and candidate-order baseline | Mechanism-focused evaluation can inform agent-interface design, not just retrieval quality |
| [SAGE-Agent](https://arxiv.org/abs/2511.08798) / [INTENT-SIM](https://aclanthology.org/2025.findings-naacl.306/) | Uncertainty/EVPI-driven question selection | Does not separate question selection from exposure timing | Keep question selection "as-is" but change exposure order via gating | Tests whether a low-cost system wrapper provides gains without training |
| [Anchoring Bias](https://arxiv.org/abs/2511.05766) / [Order Effects](https://arxiv.org/abs/2506.14092) | Studies order dependence in static LLM settings | Not in tool-using interactive loops | Evaluate order effects in an interactive search + clarification loop | Establishes a deployment-relevant form of anchoring in agents |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen/Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Primary open-weight agent model (local inference) |
| gpt-4o-mini (optional replication) | - | (API; see `available_models.md`) | Check whether effect holds for a strong API model |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| InteractComp | Evaluation only | 210 instances | https://github.com/FoundationAgents/InteractComp | (see repo) |

No training data needed - inference only.

**Other Resources (if applicable):**
- NLI responder model (one option): https://huggingface.co/microsoft/deberta-v3-large-mnli (trained on **MNLI**, the Multi-Genre Natural Language Inference dataset) (or an equivalent MNLI model).

**Resource Estimate**:
- **Compute budget**: 10-50 GPU-hours (single 7B model, 210 instances, 3 conditions; deterministic decoding; plus NLI inference).
- **GPU memory**: <= 24GB for 7B inference (or <= 80GB if using larger models).
- **API usage**: optional; ~210 * (k+2) calls per condition (k is small, e.g., 2).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| InteractComp-Anchor (derived) | Controlled variant of InteractComp where SEARCH returns {distractor, target} candidates in a specified order and INTERACT replies via NLI on hidden context | Accuracy (exact match to target) | test (all 210) | https://github.com/FoundationAgents/InteractComp | Custom harness (adapt InteractComp formatting; no web APIs) |

### Main Results

We will report accuracy for the three conditions A/B/C under the same k (pre-registered, default k=2) with paired bootstrap confidence intervals.

#### Results Table

| Method | Base Model | Benchmark | Accuracy | Source | Notes |
|---|---|---|---:|---|---|
| A. Search-first (distractor-first) | Qwen2.5-7B-Instruct | InteractComp-Anchor | **TBD** | Needs re-run | SEARCH returns [d, a], then k interactions |
| C. Candidate-order baseline (target-first) | Qwen2.5-7B-Instruct | InteractComp-Anchor | **TBD** | Needs re-run | SEARCH returns [a, d], then k interactions |
| **B. IRG (clarify-first)** | Qwen2.5-7B-Instruct | InteractComp-Anchor | **TBD** | To be verified | k interactions before SEARCH returns [d, a] |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| k sweep (report-only) | Evaluate k in {0, 1, 2, 4} without tuning | If anchoring is real, IRG benefit should appear at small k and saturate |
| Responder swap (robustness) | Replace NLI responder with an LLM yes/no/unknown responder | If effect is behavioral, direction of results should be consistent |

### Analysis (Optional)

- **First-candidate bias**: measure how often the agent chooses the first candidate; compare A vs B vs C.
- **Question discriminativeness**: for each INTERACT hypothesis, measure mutual information with (a vs d) under NLI outcomes; test whether IRG produces more discriminative probes.
- **Instance stratification**: analyze effect size vs (i) lexical overlap between q and distractor name, (ii) target popularity proxy (e.g., Wikipedia pageviews if available).

---

## Success Criteria

**Criterion 1: Timing benefit under fixed budget**
- Hypothesis: Clarify-first reduces anchoring.
- Validation: B (IRG) achieves higher accuracy than A (search-first distractor-first) under the same k, with a statistically reliable paired improvement.

**Criterion 2: Not explained by candidate reordering**
- Hypothesis: IRG adds value beyond simple reordering.
- Validation: B outperforms C, or (if B \u2248 C) the result is interpreted as evidence that a cheaper candidate-order intervention is sufficient and IRG should not be pursued further.

---

## Impact Statement

If IRG works, it provides a simple, training-free design rule for agent builders: delay exposure to candidate interpretations until after a small amount of intent reconstruction. This could reduce "technically correct but wrong intent" failures in interactive search assistants and in long-horizon research agents where early retrieval can anchor the entire solution trajectory.

---

## References

- [Towards Autonomous Mathematics Research](./references/Towards-Autonomous-Mathematics-Research/meta/meta_info.txt) - Feng et al., 2026
- [InteractComp: Evaluating Search Agents With Ambiguous Queries](./references/InteractComp-Evaluating-Search-Agents-With-Ambiguous-Queries/meta/meta_info.txt) - Deng et al., 2025
- [Clarify When Necessary: Resolving Ambiguity Through Interaction](https://aclanthology.org/2025.findings-naacl.306/) - Zhang and Choi, 2025
- [Structured Uncertainty guided Clarification for LLM Agents](https://arxiv.org/abs/2511.08798) - Suri et al., 2025
- [MAC: A Multi-Agent Framework for Interactive User Clarification](https://openreview.net/forum?id=pDOtqzaZAf) - Acikgoz et al., 2025
- [Learning to Ask: When LLMs Meet Unclear Instruction](https://arxiv.org/abs/2409.00557) - (authors per paper), 2024
- [Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying Questions](https://arxiv.org/abs/2410.13788) - (authors per paper), 2024
- [QuestBench](https://arxiv.org/abs/2503.22674) - (authors per paper), 2025
- [When2Call: When (not) to Call Tools](https://arxiv.org/abs/2504.18851) - (authors per paper), 2025
- [Tell Me More! Towards Implicit User Intention Understanding (IN3)](https://arxiv.org/abs/2402.09205) - (authors per paper), 2024
- [UserBench: An Interactive Gym Environment for User-Centric Agents](https://arxiv.org/abs/2507.22034) - (authors per paper), 2025
- [UserRL: Training Interactive User-Centric Agent via Reinforcement Learning](https://arxiv.org/abs/2509.19736) - (authors per paper), 2025
- [LHAW: Controllable Underspecification for Long-Horizon Tasks](https://arxiv.org/abs/2602.10525) - (authors per paper), 2026
- [Pushing Forward Pareto Frontiers of Proactive Agents](https://arxiv.org/abs/2602.11351) - (authors per paper), 2026
- [ClariQ](https://arxiv.org/abs/2009.11352) - Aliannejadi et al., 2020
- [Qulac](https://arxiv.org/abs/1905.00024) - Aliannejadi et al., 2019
- [AmbigQA](https://arxiv.org/abs/2004.10645) - Min et al., 2020
- [CambigNQ](https://arxiv.org/abs/2209.01076) - (authors per paper), 2022
- [Corpus-informed Retrieval Augmented Generation of Clarifying Questions](https://arxiv.org/abs/2409.18575) - (authors per paper), 2024
- [RAC: Retrieval-Augmented Clarification](https://arxiv.org/abs/2601.11722) - (authors per paper), 2026
- [Self-RAG](https://arxiv.org/abs/2310.11511) - Asai et al., 2023
- [BrowseComp](https://arxiv.org/abs/2504.12516) - Wei et al., 2025
- [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983) - Mialon et al., 2023
- [tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045) - Yao et al., 2024
- [Behavioral and Attributional Evidence of Anchoring Bias in LLMs](https://arxiv.org/abs/2511.05766) - (authors per paper), 2025
- [A Deep Dive into Order Effects in Large Language Models](https://arxiv.org/abs/2506.14092) - (authors per paper), 2025
- [Cognitive Biases in Large Language Models: A Survey](https://arxiv.org/abs/2412.00323) - (authors per paper), 2024
- [Language Models Identify Ambiguities and Exploit Loopholes](https://arxiv.org/abs/2508.19546) - (authors per paper), 2025
