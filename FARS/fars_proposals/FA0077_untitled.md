# untitled

# LogitGate: Probe-Gated Output Logit Bias as a Simplification Test of Activation Steering for Tool Calling

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated evaluation (no human labeling; no LLM-judge required)
  - No browser / search-API dependencies
  - Fit within **≤768 A100 GPU-hours**

## Introduction

### Context and Motivation

Tool-calling language models map a user request into a structured function invocation that downstream software can parse and execute. In many deployments this is a discrete mode switch: either the model emits a parseable tool call (correct) or it emits normal text / malformed structure (failure). Small changes in generation can therefore cause hard failures.

Recent work argues that some tool-calling failures are not only due to prompt formatting, but also due to a representation–behavior gap: internal activations can linearly encode “a tool is needed” while the model still does not enter tool mode under strict parsing. **ASA (Activation Steering Adapter)** proposes an inference-time controller that edits a mid-layer activation during prefill using a probe-gated steering vector, achieving large improvements in trigger F1 and false positive rate (FPR; the rate of triggering a tool call on no-tool inputs) on its strict internal benchmark protocol (e.g., Qwen2.5-1.5B Trigger-F1 **0.1818→0.5037** and FPR **0.1458→0.0521**; ASA Table 5) (**[ASA](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt)**).

However, ASA’s deployment cost is not only the small stored assets (vectors/probes), but also the need for **mid-layer activation hooks and in-place residual-stream edits**. Many engineering stacks for LLM inference already support lightweight **decode-time logit processors** (e.g., for bad-word blocking, grammar constraints, or API logit bias), which are easier to deploy and reason about than mid-layer activation interventions.

### The Problem

ASA reports that its steering direction produces a measurable increase in early **trigger-token logits** (ASA Eq. 5, ΔLogit), and positions its probe-guided gate as a decision-boundary calibration step “instead of logit-level biasing” (**[ASA](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt)**). This raises a concrete, deployment-relevant question:

**Do we actually need mid-layer activation editing to improve tool-mode triggering, or is a probe-gated output-logit bias on the tool-call prefix tokens sufficient (at least for single-turn tool calling)?**

Answering this matters because if decode-time logit bias is sufficient, many teams can obtain most of the triggering gains without implementing mid-layer hooks, reducing engineering risk and simplifying integration with standard inference runtimes.

### Key Insight and Hypothesis

**Key insight.** In some tool-calling formats, valid tool invocations begin with a small set of highly stereotyped prefix tokens (e.g., BFCL requires responses to be a Python list of function calls like `[func(a=1)]`). If an agent’s main failure mode is that it “fails to cross the threshold” into tool mode, increasing probability mass on these prefix tokens at the first few decoding steps may recover much of the trigger improvement.

**Hypothesis.** On BFCL single-turn function calling, a controller that uses ASA’s same hidden-state probe and ternary gate, but applies its intervention as a **signed output-logit bias** on the mandatory call-prefix tokens (instead of a mid-layer residual edit), can recover a large fraction of ASA-style prefill activation steering’s Trigger-F1 improvement at matched FPR, without degrading BFCL parse/correctness metrics.

**Why we could be wrong.** (i) Successful tool calling may require a distributed representation shift that increases probability of correct function/argument tokens, not only the prefix token; logit bias may create partial calls and reduce BFCL correctness. (ii) Tokenization and whitespace effects may make a “prefix token” intervention unreliable. (iii) Mid-layer steering may help by altering later decoding beyond what a bounded-step logit bias can accomplish.

**Scope and expected transfer boundaries.** This study is intentionally narrow: single-turn tool calling on BFCL in *prompting mode* (models emit a Python-list call syntax like `[func(...)]` under a system prompt). We expect LogitGate-like methods to be most effective when (a) tool-mode entry has a small set of mandatory prefix tokens, and (b) evaluation is deterministic parsing. Results may not transfer to tool interfaces with different surface formats (e.g., JSON-based function calling), to multi-turn/agentic settings where tool decisions recur over many turns, or to frontier models with native function calling where prompting-mode parsing failures are not the main bottleneck.

**Deployment relevance.** Although many 2026 deployments use frontier models with native function calling, there remains a practical ecosystem of open-weight and on-device models that rely on prompting-mode tool calling. For these systems, implementing a logit processor is often substantially easier than implementing mid-layer activation hooks; the goal here is to clarify whether that simpler path can recover most of the trigger-quality gains.

---

## Proposed Approach

### Overview

We propose **LogitGate**, a simplification test of ASA:

- **ActGate (ASA-style baseline):** ASA global-only prefill activation injection with probe-guided ternary gating.
- **LogitGate (ours):** keep the *same* probe and ternary gate, but replace the mid-layer activation injection with a decode-time logit bias applied to the tool-call prefix token set.

Both methods are trained from the same calibration data (BFCL relevance vs irrelevance labels) and are evaluated on the same deterministic BFCL evaluator.

### Method Details

#### Benchmark setting: BFCL single-turn

We use the **Berkeley Function Calling Leaderboard (BFCL)** single-turn dataset, which contains relevance cases requiring one or more function calls and **irrelevance** cases where no function invocation is expected (**[BFCL](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/meta/meta_info.txt)**). BFCL provides deterministic evaluation via AST parsing / substring matching (and execution matching for a subset) (**[BFCL](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/meta/meta_info.txt)**).

BFCL’s prompting-mode system prompt requires outputs to be function calls in a list syntax (Appendix A):
`[ func_name1(param=value, ...), func_name2(...) ]` (**[BFCL Appendix A](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/sections/A.%20System%20Prompt%20for%20Prompting%20Models.md)**).

#### Shared probe + gate

Let the base model be a frozen LLM. For each prompt x, extract the pre-layernorm residual-stream activation (the hidden state just before layer normalization) at a chosen layer L at the final prompt token, h_L(x), and standardize it using training-set mean/std as in ASA.

Train a logistic regression probe p(x)=σ(wᵀh_L(x)+b) to predict the binary label y∈{0,1} where y=1 for BFCL relevance and y=0 for BFCL irrelevance.

Map p(x) to a ternary gate g(x)∈{-1,0,+1} using a confidence threshold τ (ASA Eq. 13), where g=+1 means “apply positive steering / increase tool-call likelihood”, g=-1 means “apply negative steering / suppress tool-call likelihood”, and g=0 means “no intervention”:
- g=+1 if p>τ
- g=-1 if p<1-τ
- g=0 otherwise

#### ActGate: ASA-style global-only prefill injection (baseline)

Compute a unit steering direction v as the difference of standardized class means (ASA Eq. 6–9):

v = normalize( E[h_L(x) | y=1] − E[h_L(x) | y=0] ).

During prefill, apply a single residual-stream intervention at layer L:

h_L(x) ← h_L(x) + α · g(x) · v.

This is the simplest ASA variant without domain routing / mixture-of-vectors, chosen because BFCL does not provide stable domain labels.

#### LogitGate: probe-gated output logit bias (ours)

Instead of editing h_L, we modify the next-token logits z_t at decode time.

Define a **prefix token set** S_trigger using a tokenizer-only rule (pre-registered):

S_trigger = { token id i : decode(i) matches `^\s*\[` }.

At decode steps t=1..K (default K=3), or until any token from S_trigger is emitted, apply:

z_t[i] ← z_t[i] + β · g(x)  for all i ∈ S_trigger.

We interpret K>3 (up to a fixed cap K≤8) as a partial failure mode: it suggests that “first-token boundary calibration” is insufficient and that the effect requires sustained decode-time forcing.

#### Hyperparameter selection and fairness

- Choose L by dev-set probe AUC sweep as in ASA.
- Tune τ and α (ActGate) / τ and β (LogitGate) on a BFCL dev split.
- Use **greedy deterministic decoding** for all conditions to avoid sampling confounds and to align with ASA’s strict deterministic protocol.
- For the final comparison, select ActGate and LogitGate operating points that match FPR within ±1pp on dev, then report test metrics.

### Key Innovations

- **A falsifiable simplification test for tool-calling activation steering:** isolate whether mid-layer residual edits are necessary by holding the probe+gate fixed and changing only the intervention location (mid-layer vs output logits).
- **Pre-registered “gain recovery” criterion:** evaluate LogitGate by how much of ActGate’s improvement over prompt-only it recovers at matched FPR, which is more decision-relevant than absolute deltas.
- **Deployment-facing conclusion either way:** if LogitGate matches, practitioners can consider logit processors as a lower-risk alternative; if it fails, this supports the need for representation-level hooks.

---

## Related Work

### Field Overview

Tool calling has been studied through (i) **training-time** methods that teach models to use tools, (ii) **benchmarks** that evaluate tool selection, robustness, and multi-turn agentic behavior, and (iii) **inference-time control** methods (representation steering or decoding-time constraints) that change behavior without full fine-tuning.

A recurring observation is that “format correctness” and “decision correctness” can be distinct: constrained decoding improves syntactic validity, but models may still over-trigger or under-trigger tools; conversely, training-time tool use can improve decision-making while still producing malformed calls. This motivates studying interventions that directly target the call/no-call decision under deterministic parsing.

### Related Papers

- **[ASA: Training-Free Representation Engineering for Tool-Calling Agents](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt)**: Probe-gated mid-layer activation steering for tool-mode triggering; our direct baseline and the method we attempt to simplify.
- **[BFCL: From Tool Use to Agentic Evaluation of Large Language Models](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/meta/meta_info.txt)**: Standardized function-calling benchmark with deterministic AST-based evaluation and explicit irrelevance/no-tool cases.
- **[Gorilla](https://arxiv.org/abs/2305.15334)**: Early benchmark and model line for tool/function calling; motivates standardized evaluation.
- **[Toolformer](https://arxiv.org/abs/2302.04761)**: Self-supervised tool-use training by inserting API calls; foundational for training-time tool-use methods.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: Prompting framework interleaving reasoning and actions; illustrates common tool-calling scaffolds.
- **[ToolBench](https://arxiv.org/abs/2307.16789)**: Large-scale tool-use benchmark and tooling pipeline; complementary to BFCL.
- **[ToolLLM](https://arxiv.org/abs/2307.16789)**: Training LLMs for tool use using ToolBench; representative supervised tool-use training.
- **[API-Bank](https://arxiv.org/abs/2304.08244)**: Benchmark of API tool-use interactions; emphasizes end-to-end call correctness.
- **[StableToolBench](https://arxiv.org/abs/2410.09058)**: Benchmarking tool learning with stability under evaluation changes; relevant to robust measurement.
- **[RoTBench](https://arxiv.org/abs/2410.04564)**: Measures robustness to tool/parameter renaming; highlights brittleness under schema perturbations.
- **[Hammer](https://arxiv.org/abs/2409.00858)**: Improves robustness by masking function/argument names during training; contrasts training-time robustness vs inference-time control.
- **[ToolRM](https://arxiv.org/abs/2505.14473)**: Outcome reward modeling for tool-calling; uses schema obfuscation and highlights evaluation beyond format validity.
- **[LoopTool](https://arxiv.org/abs/2509.04037)**: Data-training loop for robust tool calls; focuses on iterative improvement rather than inference-time control.
- **[ToolACE](https://arxiv.org/abs/2409.00920)**: Improves tool-calling via post-training and evaluation; representative strong tool-calling baseline family.
- **[τ-bench](https://arxiv.org/abs/2406.12045)**: Tool-agent-user interaction benchmark with real-world domains; illustrates multi-turn complexity beyond BFCL single-turn.
- **[NESTFUL](https://arxiv.org/abs/2409.03797)**: Benchmark for nested sequences of API calls; stresses compositional tool use.
- **[Outlines](https://arxiv.org/abs/2307.09702)**: Grammar-constrained decoding framework; representative decode-time constraint approach (format validity).
- **[JSONFormer](https://arxiv.org/abs/2310.10194)**: Structured JSON generation via constrained decoding; illustrates separating structure enforcement from semantic correctness.
- **[ITI: Inference-Time Intervention](https://arxiv.org/abs/2306.03341)**: Alters model behavior via targeted activation interventions; provides context for activation-level control.
- **[Activation Addition](https://arxiv.org/abs/2308.10248)**: Adds steering vectors at inference time to control behaviors; method family related to ASA.
- **[Programming Refusal with Conditional Activation Steering](https://aclanthology.org/2024.acl-long.828/)**: Probe-gated steering for safety behaviors; conceptually similar gating pattern.
- **[Representation Engineering](https://arxiv.org/abs/2310.01405)**: Framework for steering behavior via representation directions; broader context for ASA-style approaches.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Training-time tool-use learning | Train models to call tools via SFT/RL/data synthesis | Toolformer, ToolLLM, ToolACE, LoopTool | ToolBench, API-Bank, τ-bench | Requires data + training; can overfit schemas; expensive maintenance |
| Tool-calling benchmarks | Standardize tool-call tasks and deterministic evaluation | BFCL, ToolBench, RoTBench, StableToolBench, NESTFUL | AST matching, execution matching, task success | Metrics differ; some rely on partial automation; coverage gaps |
| Representation-level inference control | Modify internal activations at inference to change behavior | ASA, Activation Addition, ITI, Conditional Activation Steering | Task-specific eval; often custom | Requires internal hooks; can have side effects; hard to deploy in some runtimes |
| Decode-time structure/logit control | Constrain or bias token generation | Outlines, JSONFormer, grammar decoding | JSON validity / grammar compliance | Enforces format but may not fix decision to call tools |

### Closest Prior Work

**ASA (Activation Steering Adapter)** (**[ASA](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt)**) is the closest work. It builds a probe and steering direction from mid-layer activations and applies a probe-gated residual-stream edit during prefill. ASA reports both causal logit shifts on trigger tokens and large behavior-level improvements under a strict tool-call protocol. Our work keeps ASA’s probe+gate design but changes only the intervention target (output logits vs mid-layer activations) to test whether mid-layer edits are necessary for trigger improvements.

**Activation Addition / general steering** (**[Activation Addition](https://arxiv.org/abs/2308.10248)**) shows that adding fixed directions can steer behaviors, typically by intervening on activations during decoding or prefill. Unlike this broader work, we focus on a discrete parsing-defined tool-mode switch and use BFCL’s deterministic evaluator.

**Decode-time structured generation frameworks** such as **Outlines** (**[Outlines](https://arxiv.org/abs/2307.09702)**) and **JSONFormer** (**[JSONFormer](https://arxiv.org/abs/2310.10194)**) enforce structure via constrained decoding or partial templates. They primarily target syntactic validity rather than the binary call/no-call decision, whereas our focus is on triggering quality (F1/FPR) under deterministic parsing.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| ASA | Probe-gated prefill activation steering for tool-mode triggering | Requires mid-layer hooks and residual edits | Replace activation edit with probe-gated logit bias on call-prefix tokens | If the main difficulty is crossing a discrete boundary into tool mode, logit bias may suffice with lower engineering complexity |
| Activation Addition | General inference-time steering via activation directions | Not specialized for tool-call parsing; no matched-FPR trigger study | Use ASA-style gate + BFCL trigger metrics | Tool calling is parser-sensitive; matched-FPR trigger evaluation is more deployment-relevant |
| Outlines / JSONFormer | Grammar/JSON constrained decoding for valid outputs | Improves syntax but not the decision to call tools | Target the call/no-call decision via gated prefix bias | Can improve triggering without imposing full grammar constraints |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-Instruct | 1.5B | https://huggingface.co/Qwen | Primary model; matches ASA scale for feasibility |

**Training Data (if applicable):**

No backbone fine-tuning. We train only lightweight linear probes and compute a steering vector from BFCL-labeled examples.

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| BFCL (single-turn) | Probe training, steering vector calibration, dev tuning, test evaluation | ~4k single-turn entries (paper reports 4,251 total across BFCL) | https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard and https://openreview.net/pdf?id=2GmDdhBdDk | See dataset card / paper |

**Resource Estimate**:

- **Compute budget**: 30–120 A100 GPU-hours total
  - hidden-state extraction over BFCL + greedy decoding for 3 conditions
  - small sweeps over (L, τ, α/β) on dev
- **GPU memory**: 1×A100 80GB is sufficient for Qwen2.5-1.5B inference
- **API usage**: none

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| BFCL single-turn (AST subset) | Single-turn function-calling benchmark with explicit Irrelevance examples and deterministic AST evaluation | Trigger Precision/Recall/F1 (↑), FPR on Irrelevance (↓), BFCL AST match accuracy on Relevance (↑) | train/dev/test (use official split if available; otherwise fixed random split by id) | https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard | BFCL evaluator (AST substring matching) described in paper; repo: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard |

**Evaluation Scripts:**
- Use BFCL’s official AST parsing rules for function-call extraction and correctness.
- Define “triggered” iff the output parses into a list containing ≥1 function call.
- For Trigger-F1/FPR: treat Relevance as positives and Irrelevance as negatives.

### Main Results

(All BFCL trigger metrics are **TBD** and will be produced by the verification run in a single, consistent evaluation harness. We include ASA’s internal “MTU-Bench” protocol numbers in the Introduction only as motivation; they are not comparable to BFCL.)

| Method | Base Model | Benchmark | Trigger-F1@matched-FPR | FPR | AST match (Relevance only) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| Prompt-only | Qwen2.5-1.5B-Instruct | BFCL single-turn | **TBD** | **TBD** | **TBD** | This work | BFCL prompting mode, greedy decoding |
| ActGate (ASA-style prefill injection) | Qwen2.5-1.5B-Instruct | BFCL single-turn | **TBD** | **TBD** | **TBD** | This work | Global-only ASA variant; probe-gated; tune α,τ |
| **LogitGate (ours)** | Qwen2.5-1.5B-Instruct | BFCL single-turn | **TBD** | **TBD** | **TBD** | This work | Probe-gated logit bias on `^\s*\[` tokens; tune β,τ |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| LogitGate w/o gate | Always apply +β bias regardless of probe score | FPR increases substantially, mirroring ASA’s “gate removed → FPR explosion” ablation |
| K sensitivity | Evaluate K∈{1,3,8} steps of logit bias | If “boundary calibration” is sufficient, K=1–3 should be enough; needing K=8 suggests a different mechanism |

### Analysis (Optional)

- **Prefix probability diagnostic**: report the model probability mass on S_trigger at the first decoding step under each method; LogitGate should explicitly increase this while ActGate may increase it indirectly.
- **Partial-call rate**: fraction of triggered outputs that start with `[` but fail AST parsing; tests the “prefix-only” failure mode.

---

## Success Criteria

**Criterion 1: Recovery of ActGate gains at matched FPR**
- Hypothesis: LogitGate recovers most of ActGate’s Trigger-F1 improvement over prompt-only at comparable FPR.
- Validation: After tuning on dev to match FPR within ±1pp, on the BFCL test split LogitGate achieves:
  - F1_logit ≥ F1_prompt + 0.70·(F1_act − F1_prompt)
  - and |FPR_logit − FPR_act| ≤ 0.01.

**Criterion 2: No major correctness regression conditional on triggering**
- Hypothesis: Logit bias does not merely induce malformed partial tool calls.
- Validation: Conditional AST match accuracy on triggered Relevance examples drops by ≤2pp vs ActGate.

**Criterion 3: Bounded-step intervention is sufficient**
- Hypothesis: If LogitGate succeeds, it should not require long-horizon forcing.
- Validation: If LogitGate only meets Criteria 1–2 with K>3 steps (cap K≤8), we treat this as partial failure and report it as evidence that simple prefix boundary calibration is insufficient.

---

## Impact Statement

If LogitGate matches ActGate, engineers deploying tool-calling controllers can consider a simpler intervention class (decode-time logit processors) to recover much of ASA-style trigger-quality gains without implementing mid-layer residual editing. If LogitGate fails, the result still guides deployment: it supports the need for representation-level hooks (like ASA) when improving tool-mode triggering under strict parsers.

---

## References

- [ASA: Training-Free Representation Engineering for Tool-Calling Agents](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt) - Wang et al., 2026
- [The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/meta/meta_info.txt) - Patil et al., 2025
- [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) - Patil et al., 2023
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) - Schick et al., 2023
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - Yao et al., 2022
- [ToolBench: Toward Realistic Tool Use for LLMs](https://arxiv.org/abs/2307.16789) - Qin et al., 2023
- [API-Bank: A Benchmark for Tool-Augmented LLMs](https://arxiv.org/abs/2304.08244) - Li et al., 2023
- [StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models](https://arxiv.org/abs/2410.09058) - Ye et al., 2024
- [RoTBench: A Multi-Level Benchmark for Evaluating Robustness of LLMs in Tool Learning](https://arxiv.org/abs/2410.04564) - Ye et al., 2024
- [Hammer: Robust Function-Calling for On-Device Language Models via Function Masking](https://arxiv.org/abs/2409.00858) - Lin et al., 2024
- [ToolRM: Outcome Reward Models for Tool-Calling Large Language Models](https://arxiv.org/abs/2505.14473) - Agarwal et al., 2025
- [LoopTool: Closing the Data-Training Loop for Robust LLM Tool Calls](https://arxiv.org/abs/2509.04037) - (authors), 2025
- [ToolACE](https://arxiv.org/abs/2409.00920) - (authors), 2024
- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045) - (authors), 2024
- [NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls](https://arxiv.org/abs/2409.03797) - (authors), 2024
- [Outlines: Structured Generation with Constrained Decoding](https://arxiv.org/abs/2307.09702) - Willard & Louf, 2023
- [JSONFormer: A Template-based Approach for JSON Generation](https://arxiv.org/abs/2310.10194) - (authors), 2023
- [Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341) - Li et al., 2023
- [Activation Addition](https://arxiv.org/abs/2308.10248) - Turner et al., 2023
- [Programming Refusal with Conditional Activation Steering](https://aclanthology.org/2024.acl-long.828/) - Lee et al., 2024
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) - Zou et al., 2023
