# untitled

# DelexGate-ASA: Canonical Schema Views for Churn-Robust Activation Steering in Tool-Calling Agents

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated evaluation (no human labeling; no LLM-judge required).
  - Deterministic churn operators (no LLM-generated alias maps).
  - No base-model fine-tuning; only ASA-style lightweight calibration (vector + linear probe).
  - Must fit within **768 A100 GPU-hours** total.

## Introduction

### Context and Motivation

Tool-calling language-model agents translate natural-language requests into software actions by selecting and invoking tools (functions/APIs). In typical deployments, the agent sees (i) a user request and (ii) a tool catalog describing function names, argument keys, and natural-language documentation. The agent must make a discrete decision that is **parser-sensitive** (small formatting errors can cause hard failures): whether to emit a tool call, and if so, produce a syntactically valid invocation.

A persistent deployment challenge is **tool-schema churn**: tool/function identifiers and argument keys change over time (refactors, renames, style changes) even when the underlying tool semantics are unchanged. ASA explicitly motivates tool calling as a setting where "available tools, API signatures, and interaction protocols change frequently" and where small shifts can break strict interfaces ([ASA introduction](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/sections/Introduction.md)). Robustness to such interface evolution is also a central concern in tool-learning benchmarks and models: **RoTBench** perturbs tool and parameter names and finds large drops, with tool-name noise often more damaging than parameter noise ([RoTBench](./references/RoTBench-A-Multi-Level-Benchmark-for-Evaluating-the-Robustness-of-Large-Language-Models-in-Tool-Learning/meta/meta_info.txt)). **ToolRM** uses schema obfuscation (random renaming/reordering) during tool-calling reward-model training to reduce shortcut learning ([ToolRM](./references/ToolRM-Outcome-Reward-Models-for-Tool-Calling-Large-Language-Models/meta/meta_info.txt)). **Hammer** proposes "function masking" (randomly masking function and parameter names during training) to reduce models' reliance on naming conventions and improve generalization across benchmarks ([Hammer](./references/Hammer-Robust-Function-Calling-for-On-Device-Language-Models-via-Function-Masking/meta/meta_info.txt)).

**ASA (Activation Steering Adapter)** is a recent training-free method for improving tool-mode triggering by adding a single mid-layer activation shift during prefill, using a probe-gated steering vector derived from labeled examples without updating model weights ([ASA](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt)). ASA reports large gains under a fixed schema (e.g., Qwen2.5-1.5B Trigger-F1 0.1818 -> 0.5037 while reducing false-positive rate; [ASA Table 5](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/sections/Main%20Results%20on%20Two%20Models.md)). However, ASA's calibration assets (mean/std, probe, and steering direction) are computed from hidden states extracted on prompts that include the tool schema, so it is unclear whether the controller transfers when schemas evolve.

### The Problem

This proposal studies a concrete, automatable subset of schema evolution:

- **Lexical churn**: deterministic renaming of tool/function identifiers and argument keys while preserving tool descriptions and type/required structure.

The deployment-relevant question is:

**When tool/argument identifiers are renamed but tool semantics are unchanged, can an activation-steering controller (ASA) be reused without per-version rebuilding, and if not, what is the minimal maintenance strategy?**

A practical baseline maintenance strategy is to simply **recalibrate** ASA on the new schema version, since ASA's calibration is lightweight compared to fine-tuning. The scientific and practical uncertainty is whether lexical churn causes a meaningful robustness gap for ASA-style controllers in the first place, and whether that gap can be mitigated without schema-specific recalibration.

### Key Insight and Hypothesis

**Key insight (diagnose-then-repair).** Lexical identifiers (function names and argument keys) are unstable, and they can act as shortcut features for any lightweight controller trained on hidden states extracted from prompts containing tool schemas. If a controller's probe/gate partially relies on identifier tokens, lexical churn can shift the probe-score distribution and degrade call/no-call decisions even when tool descriptions are unchanged.

**Hypothesis.** If we canonicalize identifiers into a stable placeholder vocabulary when constructing the prompts used for ASA calibration and inference, then the resulting controller will be substantially more stable under lexical churn than (i) reusing ASA calibrated on the original identifiers, and can approach the robustness of (ii) a per-schema recalibration baseline.

Why we could be wrong:
- ASA might already be robust to lexical churn because its probe relies mostly on the user query.
- Removing semantic hints from function names (e.g., `get_weather`) could reduce clean performance enough to negate robustness gains.
- Churn errors might be dominated by post-trigger schema compliance rather than the call/no-call decision.

---

## Proposed Approach

### Overview

We propose **DelexGate-ASA**, a canonicalized-schema view for activation steering. Here "delexicalization" means **replacing surface identifiers (tool names and argument keys) with stable placeholders** while preserving descriptions and type/required structure. We structure the study to first **establish whether ASA is actually fragile to lexical churn**, then evaluate whether canonicalization mitigates any observed degradation.

We evaluate on **BFCL (Berkeley Function Calling Leaderboard)**, a standard benchmark suite for function calling with deterministic evaluation, including explicit "irrelevance" cases where no function call should be made ([BFCL](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/meta/meta_info.txt)). BFCL defines "Relevance" vs "Irrelevance" categories based on whether any function invocation is expected ([BFCL dataset categories](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/sections/C.2.%20Dataset%20Categories.md)).

### Method Details

#### Background: ASA (global-only variant)

ASA builds a tool-necessity controller from labeled examples ([ASA](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt)):

1. Run the base model on a prompt and extract the **pre-layernorm residual stream activation** at layer \(L\) for the final prompt token \(h_L(x)\) (i.e., the hidden state just before layer normalization at that layer).
2. Standardize using calibration-set mean and std: \(\hat h_L(x)=(h_L(x)-\mu)/\sigma\).
3. Compute a steering direction \(v=\mathbb{E}[\hat h_L\mid y=1]-\mathbb{E}[\hat h_L\mid y=0]\), where \(y\) indicates "a tool call is needed".
4. Fit a linear probe \(p(x)=\sigma(w^\top \hat h_L(x)+b)\) and a ternary gate:
   - Gate=+1 if \(p>\tau\)
   - Gate=-1 if \(p<1-\tau\)
   - Gate=0 otherwise
5. During prefill, apply a one-shot intervention at the same layer \(L\):
   \[
   h_L \leftarrow h_L + \alpha\,\mathrm{Gate}(p(x);\tau)\,v.
   \]

To keep verification minimal, we use a **global-only** ASA variant (no domain router / mixture-of-vectors).

#### Canonical schema view (identifier delexicalization)

We define a deterministic schema canonicalization transform \(\mathcal{T}_{canon}\) that is applied to the tool documentation block (and only that block) before building the prompt:

- Replace the **function name** of tool \(i\) with `tool_i`.
- Replace each **argument key** of tool \(i\) with `arg_{i,j}`.
- Replace occurrences of the original function/argument strings inside tool descriptions/docstrings via exact string match (case-sensitive) to reduce identifier leakage.
- Preserve:
  - natural-language descriptions (except exact identifier replacements)
  - JSON schema types and required/optional structure
  - the number of tools and number of arguments per tool

We store a per-example mapping table \(M\): `tool_i -> <real_tool_name>`, `arg_{i,j} -> <real_key>`.

At inference time, the model generates tool calls against the canonical schema. We then parse the output and deterministically map placeholders back to the current schema using \(M\) before scoring and before tool execution.

This yields a one-pass deployment path (no two-pass inference): canonicalize schema, generate calls in canonical namespace, remap to real identifiers.

#### Methods (≤3 main conditions)

We compare three maintenance strategies for lexical churn:

1. **ASA (reuse, no maintenance)**: calibrate ASA on the original schema; evaluate the same assets on the churned schema.
2. **ASA (recalibrate on churned schema)**: recompute \(\mu,\sigma\), \(v\), and probe \((w,b,\tau)\) using the churned schema prompts (labels unchanged). This is the practical "just rebuild ASA" baseline.
3. **DelexGate-ASA (ours)**: calibrate ASA once using canonicalized prompts \(\mathcal{T}_{canon}\) on the original schema; at test time, canonicalize the current schema (original or churned) and reuse the same ASA assets. Only the deterministic mapping table \(M\) changes.

(We optionally report a prompt-only no-steering baseline for context, but it is not part of the decisive 3-condition test.)

#### Churn model (evaluation axis)

We construct churned schemas by applying deterministic renaming maps \(\pi\) to:
- tool/function names
- argument keys

Tool descriptions and type structure remain unchanged, matching RoTBench's robustness assumption that descriptions stay accurate while names are noisy ([RoTBench](./references/RoTBench-A-Multi-Level-Benchmark-for-Evaluating-the-Robustness-of-Large-Language-Models-in-Tool-Learning/meta/meta_info.txt)). Gold tool calls are rewritten with the same \(\pi\).

We use two churn severities (evaluation axis, not extra method conditions):
- **Slight**: RoTBench-style character-level corruption on up to 1/3 of characters.
- **Medium**: RoTBench-style name reversal or hash-derived random strings.

### Key Innovations

- **Canonical-schema maintenance strategy for activation steering**: reuse a single ASA calibration across schema versions by canonicalizing identifiers and remapping outputs.
- **Diagnose-first evaluation**: explicitly test whether ASA is fragile to lexical churn before claiming a fix is needed.
- **Deterministic churn harness** aligned with RoTBench-style noise, enabling fully automated replication.

---

## Related Work

### Field Overview

Tool calling combines structured generation (outputs must satisfy executable schemas), decision-making (whether to call a tool), and robustness under distribution shifts (tools, prompts, and schemas evolve). Most improvement methods rely on either (i) prompting and schema engineering or (ii) training/fine-tuning on tool-use data. Both approaches can be brittle under schema drift: prompting can be sensitive to small changes in wording and context, while fine-tuning imposes recurring training and regression-testing costs as interfaces proliferate.

Inference-time "controller" methods, such as activation steering, offer a complementary design point: rather than updating model weights, they modify internal activations using small learned artifacts (vectors and linear probes). This makes them attractive for maintenance-heavy environments, but their robustness to schema evolution is not well established.

### Related Papers

- **[ASA: Training-Free Representation Engineering for Tool-Calling Agents](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt)**: training-free activation steering for tool-mode triggering; motivates evolving interfaces but does not quantify robustness to explicit lexical churn.
- **[BFCL: From Tool Use to Agentic Evaluation of Large Language Models](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/meta/meta_info.txt)**: function-calling benchmark suite with deterministic evaluation and irrelevance cases.
- **[RoTBench](./references/RoTBench-A-Multi-Level-Benchmark-for-Evaluating-the-Robustness-of-Large-Language-Models-in-Tool-Learning/meta/meta_info.txt)**: evaluates robustness to noisy tool/argument names and motivates deterministic churn operators.
- **[ToolRM](./references/ToolRM-Outcome-Reward-Models-for-Tool-Calling-Large-Language-Models/meta/meta_info.txt)**: schema obfuscation during tool-RM training to reduce identifier shortcut learning.
- **[Hammer](./references/Hammer-Robust-Function-Calling-for-On-Device-Language-Models-via-Function-Masking/meta/meta_info.txt)**: proposes function masking (randomly renaming function/parameter names during training) and an irrelevance-augmented dataset to improve robustness to naming conventions.
- **[StableToolBench](./references/StableToolBench-Towards-Stable-Large-Scale-Benchmarking-on-Tool-Learning-of-Large-Language-Models/meta/meta_info.txt)**: stable tool-use evaluation via a virtual server; highlights reproducibility issues.
- **[MTU-Bench](./references/MTU-Bench-A-Multi-Granularity-Tool-Use-Benchmark-for-Large-Language-Models/meta/meta_info.txt)**: strict tool-trigger benchmark used by ASA.
- **[Toolformer](https://arxiv.org/abs/2302.04761)**: self-supervised tool-use training by inserting tool calls into pretraining corpora.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: prompting framework interleaving reasoning and actions.
- **[ToolBench / ToolLLM](https://arxiv.org/abs/2307.16789)**: large-scale tool-use benchmark and instruction-tuning data.
- **[API-Bank](https://arxiv.org/abs/2304.08244)**: tool-use benchmark with execution-based evaluation.
- **[Gorilla](https://arxiv.org/abs/2305.15334)**: early function-calling models and evaluation.
- **[xLAM](https://arxiv.org/abs/2404.12433)**: open function-calling models and evaluations including irrelevance.
- **[LoopTool](https://arxiv.org/abs/2509.04037)**: closed-loop data and training for robust tool calls; BFCL-focused.
- **[ToolACE](https://arxiv.org/abs/2409.00920)**: tool-use training/evaluation with an emphasis on correctness.
- **[Seal-Tools](https://arxiv.org/abs/2405.08355)**: tool-use dataset and evaluation.
- **[tau-bench](https://arxiv.org/abs/2406.12045)**: benchmark for tool agents in dynamic environments.
- **[NESTFUL](https://arxiv.org/abs/2409.03797)**: nested tool-use evaluation.
- **[MCP-Bench](https://arxiv.org/abs/2508.20453)**: benchmarking tool-using agents under standardized protocols.
- **[Schema-Guided Dialogue](https://arxiv.org/abs/1909.05855)**: classic schema-change setting for dialog systems.
- **[Scalable Multi-Domain Dialogue State Tracking](./references/Scalable-Multi-Domain-Dialogue-State-Tracking/meta/meta_info.txt)**: uses delexicalization and candidate sets to improve scalability and generalization under large or changing ontologies (conceptual precursor for canonicalizing unstable surface forms).
- **[Representation Engineering](https://arxiv.org/abs/2310.01405)**: framework for activation-direction interventions.
- **[Activation Addition](https://arxiv.org/abs/2308.10248)**: additive activation steering.
- **[Contrastive Activation Addition](https://arxiv.org/abs/2402.09347)**: contrastive direction construction and inference-time steering choices.
- **[CAST](https://arxiv.org/abs/2409.05907)**: probe-gated activation steering for safety behaviors.
- **[LoRA](https://arxiv.org/abs/2106.09685)**: parameter-efficient fine-tuning baseline.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Prompting-based tool use | Put tool schemas + call format in prompts | ReAct; BFCL prompting mode | BFCL, ToolBench | Sensitive to prompt/schema drift |
| Fine-tuned tool calling | Train models to emit tool calls via SFT/RL | ToolLLM, Gorilla, ToolACE | BFCL, API-Bank | Maintenance cost when interfaces change |
| Robustness benchmarks | Stress test under noisy tool/arg identifiers | RoTBench, StableToolBench | RoTBench, StableToolBench | Mostly diagnostic; few maintenance strategies |
| Inference-time reranking | Sample and rerank tool calls with a reward model | ToolRM | BFCL-v3, FC-RewardBench | Inference cost scales with samples |
| Activation steering / RepE | Modify internal activations to bias discrete modes | ASA, Activation Addition, CAST | MTU-Bench (ASA), others | Robustness to schema churn unclear |

### Closest Prior Work

- **ASA** ([ASA](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt)) proposes probe-gated, single-shot activation steering to improve tool-mode triggering under strict tool-call parsing. ASA motivates evolving interfaces but does not quantify how its calibration assets transfer under explicit lexical churn.

- **RoTBench** ([RoTBench](./references/RoTBench-A-Multi-Level-Benchmark-for-Evaluating-the-Robustness-of-Large-Language-Models-in-Tool-Learning/meta/meta_info.txt)) provides a controlled noise taxonomy for tool and parameter names and shows large robustness gaps. It does not study inference-time controllers such as activation steering.

- **ToolRM** ([ToolRM](./references/ToolRM-Outcome-Reward-Models-for-Tool-Calling-Large-Language-Models/meta/meta_info.txt)) uses schema obfuscation during training to reduce shortcut learning, supporting the mechanism that identifiers can be spurious. It is a training-time method (reward modeling and reranking) rather than a small controller for a fixed agent.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [ASA](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt) | Probe-gated activation steering for tool triggering | No churn evaluation; calibration prompt includes unstable identifiers | Canonicalize identifiers and reuse assets across schemas | Reduces identifier-driven probe drift under churn |
| [RoTBench](./references/RoTBench-A-Multi-Level-Benchmark-for-Evaluating-the-Robustness-of-Large-Language-Models-in-Tool-Learning/meta/meta_info.txt) | Robustness benchmark under tool/arg name noise | Benchmark-only | Use churn operators to test/maintain activation steering | Turns a diagnostic axis into a controller maintenance strategy |
| [ToolRM](./references/ToolRM-Outcome-Reward-Models-for-Tool-Calling-Large-Language-Models/meta/meta_info.txt) | ORM + best-of-n; uses schema obfuscation in training | Adds training and inference sampling cost | Use only a tiny probe + vector; no sampling | Lower overhead controller for fixed models |
| [StableToolBench](./references/StableToolBench-Towards-Stable-Large-Scale-Benchmarking-on-Tool-Learning-of-Large-Language-Models/meta/meta_info.txt) | Stable large-scale tool-use evaluation | Does not propose churn-invariant controllers | Evaluate canonical schema maintenance on stable harness | Provides actionable maintenance guidance |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-Instruct | 1.5B | https://huggingface.co/Qwen | Small enough for repeated evaluation; aligns with ASA's reported setting |

**Training Data (if applicable):**

No base-model training. ASA calibration uses labeled examples from BFCL dev.

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| BFCL single-turn (dev) | Calibration labels (relevance vs irrelevance) | TBD | https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard | BFCL license (see dataset card) |

**Resource Estimate** (rough, for verification planning):

- Calibration: \(N_{cal}\) prefill forward passes to extract \(h_L\) and fit a linear probe; no fine-tuning.
- Evaluation: \(N_{test}\) generations on clean and churned schemas.
- With Qwen2.5-1.5B and a conservative subset \(N_{cal}=1000\), \(N_{test}=1000\), 3 methods, 2 churn severities, total compute is expected to be well within **<200 A100 GPU-hours**.

**Significance / cost comparison.** The main practical alternative to DelexGate is **ASA(recalibrate)** on each schema version. We will report the incremental compute of recalibration (additional calibration forward passes per new schema) and compare it to DelexGate's fixed one-time calibration cost. DelexGate is only interesting if it delivers churn robustness close to recalibration while reducing the need for repeated calibration data collection or repeated probe/vector fitting.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| BFCL (single-turn) | Function-calling tasks with tool docs + deterministic checking; includes irrelevance cases where no call is expected | Trigger P/R/F1; FPR; BFCL single-turn accuracy (AST/exec) | dev/test | https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard | Official BFCL evaluator (AST substring + execution matching) |

**Trigger definition.** A prediction "triggers" a tool call iff it parses into at least one function call under BFCL's expected format (Appendix A system prompt) ([BFCL system prompt](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/sections/A.%20System%20Prompt%20for%20Prompting%20Models.md)).

**Primary metrics:**
- **Trigger F1**: F1 of the binary decision "emit at least one tool call" against BFCL relevance/irrelevance labels.
- **FPR**: fraction of irrelevance examples that trigger.

**Secondary metrics:**
- BFCL single-turn correctness via AST substring matching and/or execution matching, following BFCL's evaluation methodology ([BFCL evaluation](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/sections/Evaluation%20Methodology.md)).

### Main Results

Results will be filled by verification (same base model and decoding settings across rows).

| Method | Base Model | Benchmark | Schema | Trigger F1 (↑) | FPR (↓) | BFCL correctness (↑) | Source | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| Prompt-only (BFCL prompting mode) | Qwen2.5-1.5B-Instruct | BFCL single-turn | Clean | TBD | TBD | TBD | This work | No steering; BFCL system prompt only |
| ASA (reuse) | Qwen2.5-1.5B-Instruct | BFCL single-turn | Clean | TBD | TBD | TBD | This work | Calibrate on clean schema |
| ASA (reuse) | Qwen2.5-1.5B-Instruct | BFCL single-turn | Slight churn | TBD | TBD | TBD | This work | No maintenance; identifiers renamed |
| ASA (reuse) | Qwen2.5-1.5B-Instruct | BFCL single-turn | Medium churn | TBD | TBD | TBD | This work | No maintenance; identifiers renamed |
| ASA (recalibrate) | Qwen2.5-1.5B-Instruct | BFCL single-turn | Slight churn | TBD | TBD | TBD | This work | Recompute assets on churned prompts |
| ASA (recalibrate) | Qwen2.5-1.5B-Instruct | BFCL single-turn | Medium churn | TBD | TBD | TBD | This work | Recompute assets on churned prompts |
| **DelexGate-ASA (ours)** | Qwen2.5-1.5B-Instruct | BFCL single-turn | Clean | TBD | TBD | TBD | This work | Canonicalize schema + remap outputs |
| **DelexGate-ASA (ours)** | Qwen2.5-1.5B-Instruct | BFCL single-turn | Slight churn | TBD | TBD | TBD | This work | Canonicalize schema + remap outputs |
| **DelexGate-ASA (ours)** | Qwen2.5-1.5B-Instruct | BFCL single-turn | Medium churn | TBD | TBD | TBD | This work | Canonicalize schema + remap outputs |

**Pilot and statistical reporting.** Before running the full evaluation, verification should run a small pilot (e.g., 50 examples) to estimate whether lexical churn causes a non-trivial drop for ASA(reuse). If the pilot shows a <5-point absolute drop in Trigger F1 (or <3-point absolute increase in FPR) under medium churn, the study is likely to be low-significance and should be reported primarily as a diagnostic/null result. For the full runs, we will evaluate ≥3 churn seeds per severity and report mean ± std across seeds; significance can be assessed via paired bootstrap over examples within each seed (and then aggregated across seeds).

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| Canonicalization granularity | Delex tool names only vs tool+arg keys | If arg-key noise matters, full canonicalization helps more |
| Churn seed sweep | Multiple deterministic seeds per severity | Robustness conclusions should hold across seeds |

### Analysis (Optional)

- **Probe shift diagnostic:** compare probe-score distributions \(p(x)\) for relevance vs irrelevance under clean vs churn prompts; larger shifts for ASA(reuse) would support the shortcut hypothesis.
- **Name-signal diagnostic:** measure how much performance drops on the clean schema when removing tool-name semantics (canonicalization), quantifying the trade-off between semantic names and churn robustness.

---

## Success Criteria

**Criterion 1: Establish whether ASA is fragile to lexical churn.**
- Hypothesis: ASA(reuse) exhibits a meaningful drop in Trigger F1 and/or rise in FPR under churn.
- Validation: If the mean change across churn seeds is negligible, we report a negative result ("ASA is robust to this lexical churn class") and do not claim a maintenance fix is necessary.

**Criterion 2: DelexGate reduces fragility without per-version recalibration.**
- Hypothesis: When ASA(reuse) is fragile, DelexGate-ASA closes a substantial fraction of the robustness gap to ASA(recalibrate).
- Validation: On churned schemas, DelexGate achieves higher Trigger F1 than ASA(reuse) at comparable FPR, and is close to ASA(recalibrate), while not collapsing clean performance.

**Decision rule (go/no-go).**
- If ASA(reuse) is robust, the project ends as a diagnostic contribution.
- If ASA(reuse) is fragile, we claim success only if DelexGate consistently improves robustness and is competitive with ASA(recalibrate) while using a single schema-invariant calibration.

---

## Impact Statement

If successful, this work provides a simple maintenance strategy for lightweight tool-calling controllers: represent tool schemas in a stable canonical namespace for controller calibration and inference, and remap outputs to the current schema version. This can reduce sensitivity to routine identifier refactors without fine-tuning, and it yields actionable guidance even in the null case: if ASA is already robust, practitioners can safely reuse ASA assets across lexical churn in this setting. The expected practical impact is highest when schemas change frequently but collecting calibration data (or running calibration pipelines) for each new version is operationally expensive; if per-version recalibration is already trivial, DelexGate may offer only marginal benefit.

---

## References

Proposal-local artifacts:
- [ASA: Training-Free Representation Engineering for Tool-Calling Agents](./references/ASA-Training-Free-Representation-Engineering-for-Tool-Calling-Agents/meta/meta_info.txt) - Wang et al., 2026
- [RoTBench: A Multi-Level Benchmark for Evaluating the Robustness of Large Language Models in Tool Learning](./references/RoTBench-A-Multi-Level-Benchmark-for-Evaluating-the-Robustness-of-Large-Language-Models-in-Tool-Learning/meta/meta_info.txt) - Ye et al., 2024
- [ToolRM: Outcome Reward Models for Tool-Calling Large Language Models](./references/ToolRM-Outcome-Reward-Models-for-Tool-Calling-Large-Language-Models/meta/meta_info.txt) - Agarwal et al., 2025
- [StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models](./references/StableToolBench-Towards-Stable-Large-Scale-Benchmarking-on-Tool-Learning-of-Large-Language-Models/meta/meta_info.txt) - Ye et al., 2024
- [MTU-Bench: A Multi-Granularity Tool-Use Benchmark for Large Language Models](./references/MTU-Bench-A-Multi-Granularity-Tool-Use-Benchmark-for-Large-Language-Models/meta/meta_info.txt) - Wang et al., 2024
- [BFCL: From Tool Use to Agentic Evaluation of Large Language Models](./references/The-Berkeley-Function-Calling-Leaderboard-(BFCL)-From-Tool-Use-to-Agentic-Evaluation-of-Large-Language-Models/meta/meta_info.txt) - Patil et al., 2025
- [Hammer: Robust Function-Calling for On-Device Language Models via Function Masking](./references/Hammer-Robust-Function-Calling-for-On-Device-Language-Models-via-Function-Masking/meta/meta_info.txt) - Lin et al., 2024

Other citations (URLs):
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [ToolBench / ToolLLM](https://arxiv.org/abs/2307.16789)
- [API-Bank](https://arxiv.org/abs/2304.08244)
- [Gorilla](https://arxiv.org/abs/2305.15334)
- [xLAM](https://arxiv.org/abs/2404.12433)
- [LoopTool](https://arxiv.org/abs/2509.04037)
- [ToolACE](https://arxiv.org/abs/2409.00920)
- [Seal-Tools](https://arxiv.org/abs/2405.08355)
- [tau-bench](https://arxiv.org/abs/2406.12045)
- [NESTFUL](https://arxiv.org/abs/2409.03797)
- [MCP-Bench](https://arxiv.org/abs/2508.20453)
- [Schema-Guided Dialogue](https://arxiv.org/abs/1909.05855)
- [Representation Engineering](https://arxiv.org/abs/2310.01405)
- [Activation Addition](https://arxiv.org/abs/2308.10248)
- [Contrastive Activation Addition](https://arxiv.org/abs/2402.09347)
- [CAST](https://arxiv.org/abs/2409.05907)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Scalable Multi-Domain Dialogue State Tracking](https://arxiv.org/abs/1712.10224)
