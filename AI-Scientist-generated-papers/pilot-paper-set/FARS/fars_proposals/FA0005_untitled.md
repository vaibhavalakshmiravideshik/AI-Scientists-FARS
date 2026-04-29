# untitled

# Selective Delexicalization to Defend Structured-Output LLM APIs from Control-Plane Jailbreaks

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Structured-output APIs (JSON Schema / grammar-constrained decoding) are becoming the default interface for tool-using agents and production LLM systems.

Modern LLM applications increasingly require **machine-consumable structured outputs** (JSON objects, function-call arguments, typed fields) to integrate with downstream software. Constrained decoding frameworks (e.g., grammar-guided decoding) provide high JSON validity and schema compliance, enabling reliable tool use.

### The Problem

However, constrained decoding introduces a new **control-plane attack surface**: an attacker can supply a schema whose forced literals (e.g., enum strings) inject malicious content into the model’s output prefix, bypassing both prompt auditing and shallow refusal behaviors. Constrained Decoding Attacks (CDA) can hide harmful intent inside forced JSON-schema literals (e.g., `enum`/`const` strings) and achieve near-100% attack success rates (ASR) with a single query on multiple safety benchmarks (**[CDA paper](./references/Output-Constraints-as-Attack-Surface-Exploiting-Structured-Generation-to-Bypass-LLM-Safety-Mechanisms/meta/meta_info.txt)**). The CDA / EnumAttack results show this can jailbreak both proprietary and open-weight models at near-100% ASR, and even produce highly harmful content measured by StrongREJECT.

- **Control-plane jailbreaks (CDA/EnumAttack).** The CDA paper (**[CDA paper](./references/Output-Constraints-as-Attack-Surface-Exploiting-Structured-Generation-to-Bypass-LLM-Safety-Mechanisms/meta/meta_info.txt)**) demonstrates that JSON-schema constraints can force models into harmful continuations. They sketch defenses like refusal-token whitelists, token provenance tracking, and safety signaling, but do not provide an operational, low-overhead implementation with a utility/safety Pareto analysis.
- **Developer-side safety regressions under “benign” structured-output steering.** Even non-adversarial deployment interventions can raise jailbreak risk: Steering Externalities reports that on 400 HarmBench prompts, Llama-3-8B-Instruct ASR rises from **2.00%** to **20.00%** under **STEER-JSON** and to **38.50%** under **STEER-COMPLIANCE** (**[Steering Externalities](./references/Steering-Externalities-Benign-Activation-Steering-Unintentionally-Increases-Jailbreak-Risk-for-Large-Language-Models/sections/K.%20Numerical%20value%20of%20Intrinsic%20and%20Synergistic%20Vulnerabilities.md)**, Table 10). This strengthens the case that structured-output reliability features and safety can interact non-trivially.
- **Decoding-time safety interventions.** CARE proposes rollback + introspection to reduce harmful responses during decoding (**[CARE](./references/CARE-Decoding-Time-Safety-Alignment-via-Rollback-and-Introspection-Intervention/meta/meta_info.txt)**), but it targets data-plane harmful prompts rather than control-plane schema payloads.

In 2026, practitioners want to keep structured-output interfaces (for agents and tool use) but need **cheap, deployment-ready mitigations** against control-plane jailbreaks. Naively rejecting all schemas with unusual strings is brittle and may cause high false positives; comprehensive output auditing is expensive and often disabled for latency. The CDA paper’s suggested defenses (provenance tracking; deeper integration) may be costly to deploy in common constrained-decoding libraries.

### Key Insight and Hypothesis

We propose a simple, training-free defense primitive: **Selective Delexicalization (Selective DeLex-JSON)**. Before compiling a user-provided JSON schema into a constrained-decoding grammar, we (1) strip non-semantic free-text schema fields, and (2) replace **suspicious forced literals** (long / whitespace-rich / instruction-like `enum`/`const` strings) with short opaque placeholders (e.g., `E7`). This prevents natural-language payloads from entering the model’s autoregressive context—the hypothesized mechanism that makes CDA effective—while preserving typical benign schemas whose enums are short identifiers (tool names, class labels).

CDA/EnumAttack succeeds largely because it injects **contiguous natural-language priming strings** into the autoregressive context via forced literals (e.g., long `enum`/`const` strings). If we prevent these strings from ever appearing during model generation—by replacing them with opaque placeholders—then the model cannot be steered into harmful continuations by control-plane literal injection, while most benign schemas (whose enums are short IDs) remain usable.

If, on a primary open-weight model (pre-registered) under a fixed constrained-decoding stack, CDA-style attacks have baseline ASR ≥30% on HarmBench/StrongREJECT, then **Selective DeLex-JSON must reduce ASR to <10% absolute** while keeping benign structured-output utility within ~2 percentage points of the no-defense baseline on JSONSchemaBench/IFEval-style tasks, and keeping benign schema rejection+modification rate low (≤2%). Otherwise we refute (or narrow scope).

---

## Proposed Approach

### Overview

**Selective DeLex-JSON** is a schema-to-schema transformation applied *before* constrained decoding:

1. **Strip free-text fields** that can carry hidden payloads but are not needed for decoding (e.g., `description`, `title`, `examples`, `default`).
2. **Selective delexicalization of forced literals**: for every string literal that will be forced into the generation stream (primarily `enum` and `const`, optionally other forcing constructs depending on the decoding engine), replace suspicious literals with placeholders `E_i`.
3. Run constrained decoding on the sanitized schema.
4. (Optional for downstream compatibility) Postprocess placeholders back to original literals for API consumers, while ensuring **safety evaluation only judges model-generated free-text fields**, not forced literals.

### Method Details

**Threat model.** The attacker controls the schema and aims to cause the model to generate harmful content in a free-text field (e.g., `answer`). The defender is the API provider who can transform schemas and enforce constrained decoding.

**Schema parsing and target set.** Extract a set of forced string literals:
- JSON Schema `enum` string values
- JSON Schema `const` string values
- (v1 scope) do **not** delexicalize property names; we treat property-name injection as out-of-scope or as a small diagnostic probe.

**Suspicion function (pre-registered).** A literal is delexicalized if ANY of:
- length > 20 characters
- contains whitespace (spaces/newlines/tabs)
- matches simple instruction-like regexes (e.g., “Sure,” “Step 1”, “here’s how”, imperative verbs + punctuation-heavy patterns)
- optional guard-model flag: a lightweight safety classifier (e.g., Llama Guard) marks it unsafe

These thresholds/rules will be **frozen before evaluation on HarmBench/StrongREJECT** to avoid leakage.

**Placeholder scheme.** Replace flagged literals with unique opaque IDs `E0`, `E1`, … and record a mapping table.

**Safety evaluation protocol (critical).** Since forced literals may contain attacker-supplied harmful text, we compute ASR/StrongREJECT only on model-generated free-text fields (e.g., extract `answer` string), masking or removing forced literals prior to scoring.

### Key Innovations

- **Delexicalization as a security primitive for structured-output APIs**: blocks control-plane natural-language injection without full token-provenance infrastructure.
- **Selective, not blanket, transformation**: preserves short semantic enums (tool IDs, class labels) while neutralizing long natural-language payloads.
- **Evaluation protocol that separates forced vs model-generated content**, reducing judge contamination.

---

## Related Work

### Field Overview

LLM safety research has largely focused on data-plane prompt-based jailbreaking and defenses, with standardized benchmarks such as HarmBench and StrongREJECT. In parallel, structured output reliability research has developed constrained decoding frameworks and benchmarks (e.g., JSONSchemaBench). Control-plane jailbreaks (CDA) connect these threads: structured-output constraints can override refusal behavior and bypass prompt auditing, suggesting that “safety alignment is only a few tokens deep” in practice.

Defenses span (i) prompt/input auditing, (ii) output auditing / guard models, (iii) decoding-time interventions, and (iv) training-time alignment. Our proposal targets a distinct mechanism: **literal injection via forced schema strings** (best matched by EnumAttack-style CDAs). We note that more sophisticated CDAs that decouple payloads across many short literals and/or across turns (sometimes framed as **DictAttack / space–time decoupling** in follow-up work) may bypass single-plane schema sanitization; we treat these as *out-of-scope for v1* and as a likely boundary condition, and we include a simplified “chunked payload” probe to map the boundary.

### Related Papers

- **[CDA / Output Constraints as Attack Surface](./references/Output-Constraints-as-Attack-Surface-Exploiting-Structured-Generation-to-Bypass-LLM-Safety-Mechanisms/meta/meta_info.txt)**: Introduces control-plane CDA/EnumAttack achieving near-100% ASR by embedding harmful payloads in structured-output constraints.
- **[Steering Externalities](./references/Steering-Externalities-Benign-Activation-Steering-Unintentionally-Increases-Jailbreak-Risk-for-Large-Language-Models/meta/meta_info.txt)**: Shows benign JSON/compliance activation steering increases HarmBench ASR, motivating structured-output safety auditing.
- **[StrongREJECT](https://openreview.net/forum?id=al303JJkGO)**: Provides a benchmark and scoring rubric for harmfulness beyond binary ASR.
- **[HarmBench](https://arxiv.org/abs/2402.04249)**: Standardized automated red-teaming benchmark with classifier-based ASR evaluation.
- **[JailbreakBench](https://openreview.net/forum?id=urjPCYZt0I)**: Benchmark for jailbreaking robustness across diverse harm categories.
- **[AdvBench / GCG paper](https://arxiv.org/abs/2307.15043)**: Universal adversarial suffix attack benchmark and methodology for black-box jailbreaks.
- **[SORRY-Bench](https://openreview.net/forum?id=YfKNaRktan)**: Systematic evaluation of refusal behavior across many harm categories.
- **[MASTERKEY](https://www.ndss-symposium.org/ndss-paper/masterkey-automated-jailbreaking-of-large-language-model-chatbots/)**: Template-based automated jailbreaking for chat models.
- **[PAIR](https://arxiv.org/abs/2310.08419)**: Prompt Automatic Iterative Refinement, a black-box iterative jailbreak method.
- **[TAP](https://openreview.net/forum?id=SoM3vngOH5)**: Tree-of-attacks search method for black-box jailbreaking.
- **[CoP](https://arxiv.org/abs/2506.00781)**: Agentic red-teaming by composing jailbreak principles.
- **[Enforced Decoding (EnDec)](https://aclanthology.org/2024.acl-long.299/)**: Shows jailbreaking open LLMs by controlling logits / enforced decoding.
- **[APT / Prefix-tree structured-output jailbreaking](https://arxiv.org/abs/2502.13527)**: Exploits structured-output interfaces via prefix-tree search to bypass refusals.
- **[CARE](./references/CARE-Decoding-Time-Safety-Alignment-via-Rollback-and-Introspection-Intervention/meta/meta_info.txt)**: Decoding-time rollback+introspection intervention for safety–quality tradeoffs.
- **[Llama Guard](https://arxiv.org/abs/2312.06674)**: Guard model for safety classification of prompts/outputs.
- **[Constitutional AI](https://arxiv.org/abs/2212.08073)**: RLAIF framework for principled harmlessness.
- **[StruQ](./references/StruQ-Defending-Against-Prompt-Injection-with-Structured-Queries/meta/meta_info.txt)**: Uses structured queries and training to defend prompt-injection in LLM-integrated apps.
- **[JSONSchemaBench](./references/Generating-Structured-Outputs-from-Language-Models-Benchmark-and-Studies/meta/meta_info.txt)**: Benchmark for structured generation frameworks (coverage, compliance, efficiency, and quality).
- **[Outlines](https://arxiv.org/abs/2307.09702)**: Finite-state guided generation framework supporting regex/grammar/JSON constraints.
- **[XGrammar](https://arxiv.org/abs/2403.05196)**: Efficient grammar-constrained decoding engine.
- **[SynCode](https://arxiv.org/abs/2401.05767)**: Grammar-guided code generation with constrained decoding.
- **[Safety Alignment Should be More Than a Few Tokens Deep](https://openreview.net/forum?id=6Mxhg9PtDE)**: Argues safety concentrated in early tokens is brittle, motivating control-plane defenses.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Control-plane structured-output attacks | Inject harmful intent via constrained decoding rules | CDA/EnumAttack, APT | HarmBench, StrongREJECT, AdvBench | Requires structured-output interface; attack surface varies by decoder |
| Prompt-based jailbreaks (data-plane) | Optimize or search prompts to elicit harm | GCG, PAIR, TAP, MASTERKEY, CoP | AdvBench, JailbreakBench, HarmBench | Often needs multiple queries; mitigations exist via prompting/guards |
| Decoding-time defenses | Intervene during generation to avoid unsafe trajectories | CARE, SafeDecoding-style work | BeaverTails, HarmBench | Can add latency; typically targets data-plane harmful prompts |
| Input/output auditing | Classify harmfulness of prompts/outputs | Llama Guard, output scouting | HarmBench, SORRY-Bench | Expensive; false positives/negatives; may miss control-plane payloads |
| Structured generation reliability | Ensure schema compliance and measure it | JSONSchemaBench, Outlines, XGrammar | JSONSchemaBench, IFEval | Typically ignores safety; can introduce new attack surfaces |

### Closest Prior Work

1. **CDA / Output Constraints as Attack Surface** (**[CDA paper](./references/Output-Constraints-as-Attack-Surface-Exploiting-Structured-Generation-to-Bypass-LLM-Safety-Mechanisms/meta/meta_info.txt)**)
   - **What it does**: Defines CDA/EnumAttack and demonstrates high ASR across models and benchmarks; sketches mitigations like refusal whitelists and token provenance.
   - **Key limitation**: Does not provide a low-overhead, library-compatible mitigation with quantified safety–utility tradeoff.
   - **Why ours differs**: We propose and test **delexicalization of forced literals** as a mitigation primitive, and provide a structured evaluation protocol separating forced vs model-generated content.

2. **Steering Externalities** (**[Steering Externalities](./references/Steering-Externalities-Benign-Activation-Steering-Unintentionally-Increases-Jailbreak-Risk-for-Large-Language-Models/meta/meta_info.txt)**)
   - **What it does**: Shows benign JSON/compliance activation steering can increase ASR.
   - **Key limitation**: Not about constrained decoding or schema payload injection.
   - **Why ours differs**: We defend structured-output interfaces by transforming schemas (no activation steering assumed).

3. **CARE** (**[CARE](./references/CARE-Decoding-Time-Safety-Alignment-via-Rollback-and-Introspection-Intervention/meta/meta_info.txt)**)
   - **What it does**: Applies targeted rollback + introspection when unsafe content detected during decoding.
   - **Key limitation**: Requires continuous guard monitoring and targets data-plane unsafe prompts.
   - **Why ours differs**: We target **control-plane literal injection** with a preprocessing step, aiming for near-zero runtime overhead.

### Comparison Table

| Related work | What it does (1 sentence) | Key limitation / gap | What we change | Why ours should win (hypothesis + evidence) |
|---|---|---|---|---|
| CDA/EnumAttack | Shows control-plane JSON-schema literal injection jailbreaks models | Defenses not operationalized; no low-overhead mitigation | Transform schema to remove natural-language literals from generation stream | If CDA relies on literal priming, removing literals should break the mechanism |
| Input/output guard models | Classify harmful prompts/outputs | Can be bypassed; expensive; schema payloads may not be audited | Audit/transform schema *before* decoding | Prevent attack mechanism rather than detect after the fact |
| CARE | Decoding-time rollback/introspection on unsafe tokens | Runtime overhead; targets data-plane | Static preprocessing (delex) | Blocks priming with minimal runtime cost |
| Reject-only schema filtering | Reject risky schemas | High false positives; reduces utility | Delex instead of reject | Should preserve more benign schemas while neutralizing payload |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Llama-3.1-Instruct | 8B | https://huggingface.co/meta-llama | **Primary model** (pre-registered) |
| Qwen2.5-Instruct | 7B | https://huggingface.co/Qwen | Secondary model for generalization |

**Training Data (if applicable):**

No training data needed - inference only.

**Other Resources (if applicable):**
- Safety classifier / guard: Llama Guard or HarmBench classifier (open weights).
- Constrained decoding library: Outlines or equivalent grammar-guided decoding stack (fixed version).

**Resource Estimate**:
- **Compute budget**: Inference-heavy. Rough order: O(#prompts × #conditions × #models). Target ≤100 A100 GPU-hours total (including guard-model inference). Hard cap 768 GPU-hours.
- **GPU memory**: 8B-class models + constrained decoding should fit on a single A100-80GB; optional tensor parallel if needed.
- **API usage**: None required for primary metrics (ASR via HarmBench classifier). Optional: GPT-based StrongREJECT scoring if available.

**Infrastructure constraints** (proposals requiring these are infeasible):
- Search engine APIs (Google, Bing) — NOT available
- Web browsers / desktop GUIs / mobile environments — NOT available
- Complex game engines or heavy simulation environments — NOT available

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| HarmBench | Harmful prompts for automated red-teaming | ASR (unsafe rate) | test | https://arxiv.org/abs/2402.04249 | HarmBench official eval (classifier) |
| StrongREJECT | Harmful prompt set + scoring | ASR + (optional) StrongREJECT score | test | https://openreview.net/forum?id=al303JJkGO | StrongREJECT eval; if judge refuses, use extracted-field scoring |
| JSONSchemaBench | Real-world JSON schema constrained generation | JSON validity, schema compliance, task accuracy (where applicable), schema-modification rate | provided | https://github.com/guidance-ai/jsonschemabench | JSONSchemaBench repo |
| IFEval (JSON subset) | Instruction-following under JSON constraints | JSON correctness | test | https://arxiv.org/abs/2311.07911 | IFEval scripts |

**Evaluation Scripts:**
- For attack ASR: follow HarmBench classifier protocol.
- For safety evaluation: **extract only model-generated free-text fields** (e.g., `answer`) before scoring; mask forced literals.
- For utility: run JSONSchemaBench harness and report (i) validity, (ii) compliance, (iii) any task accuracy, and (iv) delex modification rate.

### Main Results

#### Comparability Rules (CRITICAL)

All comparisons are within the same model + decoding stack; the only change is schema preprocessing/defense.

#### Results Table

| Method | Base Model | Benchmark | Metric 1 | Metric 2 | Source | Notes |
|--------|------------|-----------|----------|----------|--------|-------|
| EnumAttack (reported) | Llama-3.1-8B | AdvBench (520) | ASR **99.2%** | SR **95.1%** | [CDA paper](./references/Output-Constraints-as-Attack-Surface-Exploiting-Structured-Generation-to-Bypass-LLM-Safety-Mechanisms/meta/meta_info.txt) (Table 3) | Uses vLLM structured output; evidence of severity |
| EnumAttack (reported) | Phi-3.5-MoE | HarmBench (100) | ASR **98.0%** | SR **74.6%** | [CDA paper](./references/Output-Constraints-as-Attack-Surface-Exploiting-Structured-Generation-to-Bypass-LLM-Safety-Mechanisms/meta/meta_info.txt) (Table 4) | Cross-benchmark evidence; open-weight “best aligned” model |
| Direct prompting baseline (reported) | Phi-3.5-MoE | HarmBench (100) | ASR **31.0%** | SR **27.3%** | [CDA paper](./references/Output-Constraints-as-Attack-Surface-Exploiting-Structured-Generation-to-Bypass-LLM-Safety-Mechanisms/meta/meta_info.txt) (Table 4) | Shows large delta vs CDA |
| JSON steering (reported) | Llama-3-8B-Instruct | HarmBench (400) | ASR **2.00%→20.00%** | - | [Steering Externalities Table 10](./references/Steering-Externalities-Benign-Activation-Steering-Unintentionally-Increases-Jailbreak-Risk-for-Large-Language-Models/sections/K.%20Numerical%20value%20of%20Intrinsic%20and%20Synergistic%20Vulnerabilities.md) | Different mechanism; motivation for structured-output safety |
| No defense | Llama-3.1-8B | HarmBench/StrongREJECT | **TBD** | **TBD** | - | Needs re-run under our decoding stack |
| Input guard (prompt+schema) | Llama-3.1-8B | HarmBench/StrongREJECT | **TBD** | **TBD** | - | Needs re-run |
| Reject-only schema audit | Llama-3.1-8B | HarmBench/StrongREJECT | **TBD** | benign reject% **TBD** | - | Needs re-run |
| Escape-hatch wrapper | Llama-3.1-8B | HarmBench/StrongREJECT | **TBD** | **TBD** | - | Needs re-run |
| **Selective DeLex-JSON (Ours)** | Llama-3.1-8B | HarmBench/StrongREJECT | **TBD** | benign modify% **TBD** | - | To be verified |
| **Ours + escape hatch** | Llama-3.1-8B | HarmBench/StrongREJECT | **TBD** | **TBD** | - | Optional |
| **Selective DeLex-JSON (Ours)** | Llama-3.1-8B | JSONSchemaBench | JSON validity **TBD** | compliance/acc **TBD** | - | Utility evaluation |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | Strip free-text + selective enum/const delex | Best ASR–utility tradeoff |
| Strip-only | Remove schema free-text, no delex | Small improvement if payload in free-text matters; likely insufficient |
| Delex-all | Replace all enum/const strings with placeholders | More secure but harms semantic-enum utility |
| No guard model | Heuristic-only suspicion test | Similar security if heuristics capture the key mechanism |
| Chunked-payload probe | Attack with many short enum values | Identifies boundary condition / potential bypass |

### Analysis (Optional)

- Measure correlation between attack success and presence/length of forced literal strings.
- Report distribution of benign schema modifications (what got delexed and why).

---

## Success Criteria

**Criterion 1: CDA robustness under constrained decoding**
- Hypothesis: Removing contiguous natural-language literals from the constrained-decoding context blocks CDA-style priming.
- Validation: If baseline attack ASR ≥30% on HarmBench/StrongREJECT, Selective DeLex-JSON reduces ASR to <10% absolute on the primary model.

**Criterion 2: Utility preservation for structured outputs**
- Hypothesis: Most benign structured-output schemas use short semantic enums and do not require long natural-language literals.
- Validation: On JSONSchemaBench + IFEval(JSON), JSON validity and task scores do not degrade by more than ~2pp vs no defense, and benign schema reject+modify rate stays low (≤2%).

**Criterion 3: Better Pareto point than reject-only filtering**
- Hypothesis: Delexicalization retains more benign schemas than audit-reject, while still blocking attacks.
- Validation: For similar ASR reduction, DeLex-JSON has lower benign rejection/modification costs than reject-only baselines.

---

## Impact Statement

If Selective DeLex-JSON works, LLM API providers and agent developers can keep JSON Schema / grammar-constrained decoding interfaces while reducing exposure to control-plane jailbreaks, without deploying heavy token-provenance infrastructure or expensive continuous output auditing. This would provide a simple, training-free defense primitive that can be implemented as a schema preprocessor in common constrained-decoding libraries, improving deployment safety while preserving structured-output utility.

---

## References

- [Beyond Prompts: Space-Time Decoupling Control-Plane Jailbreaks in LLM Structured Output](./references/Output-Constraints-as-Attack-Surface-Exploiting-Structured-Generation-to-Bypass-LLM-Safety-Mechanisms/meta/meta_info.txt) - Zhang et al., 2026
- [Steering Externalities: Benign Activation Steering Unintentionally Increases Jailbreak Risk for Large Language Models](./references/Steering-Externalities-Benign-Activation-Steering-Unintentionally-Increases-Jailbreak-Risk-for-Large-Language-Models/meta/meta_info.txt) - Xiong et al., 2026
- [Generating Structured Outputs from Language Models: Benchmark and Studies (JSONSchemaBench)](./references/Generating-Structured-Outputs-from-Language-Models-Benchmark-and-Studies/meta/meta_info.txt) - Geng et al., 2025
- [CARE: Decoding Time Safety Alignment via Rollback and Introspection Intervention](./references/CARE-Decoding-Time-Safety-Alignment-via-Rollback-and-Introspection-Intervention/meta/meta_info.txt) - Hu et al., 2025
- [StruQ: Defending Against Prompt Injection with Structured Queries](./references/StruQ-Defending-Against-Prompt-Injection-with-Structured-Queries/meta/meta_info.txt) - Chen et al., 2025
- [HarmBench](https://arxiv.org/abs/2402.04249) - Mazeika et al., 2024 (code: https://github.com/centerforaisafety/HarmBench)
- [StrongREJECT](https://openreview.net/forum?id=al303JJkGO) - Souly et al., 2024
- [Llama Guard](https://arxiv.org/abs/2312.06674) - Inan et al., 2023
- [Outlines](https://arxiv.org/abs/2307.09702) - Willard & Louf, 2023
- (Plus additional citations embedded in Related Work.)
