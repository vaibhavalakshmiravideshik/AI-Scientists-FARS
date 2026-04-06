# untitled

# SafeExpert-MPQ: Protecting Safety-Critical Experts During Post-Training Quantization of MoE LLMs

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Mixture-of-Experts (MoE) language models reduce inference compute by routing each token through a small subset of expert feed-forward networks, enabling strong quality with much lower *active* parameters than dense models. Because MoE models still have large *total* parameter counts, many deployments rely on post-training quantization (PTQ), such as 8-bit or 4-bit weight-only formats, to fit models into commodity GPUs and increase throughput.

Safety alignment (e.g., harmlessness / refusal to comply with harmful requests) is usually evaluated on full-precision checkpoints. However, recent work shows that quantization can change aligned behavior: safety and fairness can degrade even when general capability remains similar (e.g., HarmLevelBench and Q-resafe), and alignment-preserving quantization objectives or post-quantization patching may be needed in dense models.

A separate line of evidence suggests that MoE architectures create an additional safety fragility: safety behaviors are often **localized** to a small subset of experts or neurons rather than uniformly distributed. SAFEx shows that masking a handful of safety-critical experts can substantially reduce refusal rates in aligned MoE models. GateBreaker and Large Language Lobotomy show that selectively disabling a small fraction of expert pathways can dramatically increase jailbreak success while largely preserving general utility. SteerMoE shows that small routing interventions can both increase safety and remove it entirely when combined with jailbreak prompts. Together, these results suggest that MoE safety is mediated by a small number of “safety-critical” computational pathways.

### The Problem

For dense models, mixed-precision strategies can preserve alignment-relevant weights (e.g., Critical Weight Protection), but MoE models introduce a new structural question:

> When an aligned MoE model is quantized for deployment, is the resulting safety regression primarily caused by quantization noise in a small set of safety-critical experts, and can we preserve safety by keeping only those experts at higher precision?

This is not addressed by existing MoE PTQ work, which primarily targets perplexity / downstream accuracy and routing stability (e.g., MoEQuant, EAQuant, ExpertQuant, QuantMoE-Bench). It is also not addressed by dynamic expert-precision systems (e.g., DynaExq), which allocate higher precision to “hot” experts to preserve accuracy under memory budgets, but do not aim to preserve safety-critical pathways that may be rare yet important.

A naive baseline is to keep the most frequently used experts (or the most salient experts by generic PTQ sensitivity) at higher precision. This may fail if safety-critical experts are not among the most-used experts on benign workloads, or if safety is controlled by experts whose importance is not well-captured by standard PTQ objectives.

### Key Insight and Hypothesis

**Key insight.** If MoE safety is concentrated in a small subset of experts, then safety regressions under aggressive PTQ may be dominated by quantization error in those experts. This creates a simple intervention: identify safety-critical experts using a small set of harmful and benign prompts, and keep only those experts in higher precision while quantizing all other experts.

**Hypothesis.** Under experts-only weight quantization (router + shared blocks kept in FP16), a **safety-expert-protected mixed-precision PTQ** scheme will preserve jailbreak robustness better than (i) uniform expert quantization and (ii) a compute/memory-matched salience-based expert protection baseline.

Why we could be wrong:
1. Safety degradation might be dominated by **routing drift** induced indirectly by quantizing experts (hidden-state changes alter later routing), not by within-expert weight fidelity.
2. Safety-critical experts might overlap heavily with generic salience/hotness experts, in which case safety-specific selection adds little beyond standard mixed precision.
3. Safety-critical expert sets might be unstable across prompts / decoding settings, making identification too noisy to be useful.

---

## Proposed Approach

### Overview

We propose **SafeExpert-MPQ**, a training-free mixed-precision post-training quantization method for aligned MoE LLMs:

1. **Identify safety-critical experts** on a small identification set by measuring which experts are most associated with safe refusals on harmful prompts.
2. **Quantize experts only**: keep router and shared transformer blocks in FP16, quantize expert FFN weights to low-bit (e.g., 4-bit) **except** the identified safety-critical experts, which remain in higher precision.
3. **Evaluate safety vs over-refusal** on standard harmful-prompt datasets with an open safety classifier (Llama-Guard), using memory-matched baselines.

The goal is not to improve safety beyond the original FP16 checkpoint. The goal is to reduce **safety regression caused by deployment quantization** at a fixed memory budget.

### Method Details

#### Setting: MoE layer notation

Consider an MoE transformer with layers \(\ell = 1..L\). Each MoE layer contains \(E\) experts, each an FFN with parameters \(W_{\ell,e}\). The router produces per-token gate probabilities \(g_{\ell,e}(x_t)\) and activates top-\(k\) experts.

We assume access to the model weights and router logits (open-weight MoE), consistent with the threat/usage model in SAFEx, GateBreaker, L3, and SteerMoE.

#### Step 1: Identify safety-critical experts (training-free)

We use a small set of harmful prompts \(D_{harm}\) (e.g., StrongREJECT) and a benign prompt set \(D_{benign}\) (e.g., XSTest safe prompts). For each harmful prompt \(p\in D_{harm}\), we generate a response from the **FP16** base model using a fixed decoding policy (greedy or temperature=0). We then label the output as safe/unsafe using a local safety classifier (Llama-Guard-3-8B).

For each expert \((\ell,e)\), compute an activation score on harmful prompts:

- Let \(a_{\ell,e}(p)\) be the fraction of generated tokens for which expert \((\ell,e)\) is in the router top-\(k\).
- Let \(y(p)\in\{0,1\}\) indicate whether the model’s response is safe (refusal/harmless).

Define a refusal-association score:
\[
S_{\ell,e} = \mathbb{E}[a_{\ell,e}(p)\mid y(p)=1] - \mathbb{E}[a_{\ell,e}(p)\mid y(p)=0].
\]
Intuition: experts disproportionately active when the model refuses harmful requests are candidates for safety-critical “control” pathways.

We select the top \(N\) experts per layer (or globally) by \(S_{\ell,e}\). To reduce the risk of selecting “harm capability” experts, we include an **automatic validation filter** on a small dev subset \(D_{dev}\subset D_{harm}\): for each candidate expert, temporarily zero its contribution to the MoE output (without changing routing), and keep the expert only if this increases ASR on \(D_{dev}\) by at least \(\delta=+2\%\) absolute.

#### Step 2: Mixed-precision PTQ at expert granularity

We perform **experts-only** weight quantization:

- Router weights and shared blocks (attention, embeddings, layer norms) remain FP16.
- All experts not selected as safety-critical are quantized to low-bit (e.g., 4-bit weight-only PTQ).
- Selected safety-critical experts remain FP16 (or 8-bit).

This can be implemented by replacing the linear layers in non-critical experts with 4-bit modules (e.g., bitsandbytes NF4) while leaving selected experts as standard FP16 modules.

#### Step 3: Memory parity constraint

Let \(B_{A}\) be the total bytes of expert weights under uniform 4-bit expert PTQ.

We choose \(N\) such that the protected-expert method’s expert-weight bytes satisfy:
\[
B_{B} \le 1.05\times B_{A}
\]
(i.e., within +5% expert-weight memory), ensuring a deployment-relevant comparison.

### Key Innovations

- **Safety-aware mixed precision for MoE PTQ**: allocate higher precision specifically to safety-critical experts, rather than to experts important for generic perplexity/accuracy.
- **Experts-only PTQ causal isolation**: keep router/shared blocks FP16 to focus the experiment on whether safety loss is driven by expert-weight fidelity.
- **Fully automated identification and evaluation**: no human labeling; uses only router traces and an open safety classifier.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) quantization-induced safety regressions, (ii) MoE-specific PTQ and mixed precision, and (iii) the emerging evidence that MoE safety is localized.

Quantization safety work shows that alignment can degrade under PTQ even when utility is preserved, motivating alignment-aware quantization objectives and safety patching (e.g., HarmLevelBench, Q-resafe, Q-realign, Alignment-Aware Quantization, Critical Weight Protection). Separately, MoE PTQ work focuses on preserving accuracy and routing stability under low-bit quantization (e.g., MoEQuant, EAQuant, ExpertQuant), while systems work explores dynamic precision to fit MoEs into memory budgets (e.g., DynaExq). Finally, MoE safety analysis and attacks show that safety can be concentrated in small subsets of experts/neurons (SAFEx, GateBreaker, L3, SteerMoE), suggesting a new mechanism for quantization-induced safety regressions.

### Related Papers

- **[SAFEx: Analyzing Vulnerabilities of MoE-Based LLMs via Stable Safety-critical Expert Identification](./references/SAFEx-Analyzing-Vulnerabilities-of-MoE-Based-LLMs-via-Stable-Safety-critical-Expert-Identification/meta/meta_info.txt)**: Identifies safety-critical experts and shows that masking a small number can significantly reduce refusal rates.
- **[GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs](./references/GateBreaker-Gate-Guided-Attacks-on-Mixture-of-Expert-LLMs/meta/meta_info.txt)**: Localizes safety neurons and shows large ASR gains by disabling a small fraction of them.
- **[Large Language Lobotomy: Jailbreaking Mixture-of-Experts via Expert Silencing](./references/Large-Language-Lobotomy-Jailbreaking-Mixture-of-Experts-via-Expert-Silencing/meta/meta_info.txt)**: Uses sequential routing analysis to identify and silence safety experts, increasing ASR substantially.
- **[Steering MoE LLMs via Expert (De)Activation](./references/Steering-MoE-LLMs-via-Expert-De-Activation/meta/meta_info.txt)**: Shows inference-time routing interventions can both increase safety and remove it, exposing alignment fragility in MoE routing.
- **[RASA: Routing-Aware Safety Alignment for Mixture-of-Experts Models](./references/RASA-Routing-Aware-Safety-Alignment-for-Mixture-of-Experts-Models/meta/meta_info.txt)**: Argues MoE alignment can take routing shortcuts; proposes selective expert repair plus router-consistency optimization.
- **[Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference](./references/Dynamic-Expert-Quantization-for-Scalable-Mixture-of-Experts-Inference/meta/meta_info.txt)**: Dynamically allocates expert precision by hotness for accuracy under memory constraints (not safety).

Additional relevant work (arXiv links):
- **[Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection](https://arxiv.org/abs/2601.12033)**: Uses gradient-based importance to keep safety/fairness-critical weights at higher precision in dense LLMs.
- **[HarmLevelBench: Evaluating Harm-Level Compliance and the Impact of Quantization on Model Alignment](https://arxiv.org/abs/2411.06835)**: Shows quantization can change harm-level compliance even when utility is similar.
- **[Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models](https://arxiv.org/abs/2506.20251)**: Measures safety degradation under PTQ and proposes patching.
- **[Q-realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment](https://arxiv.org/abs/2601.08089)**: Proposes coupling quantization with a realignment step.
- **[Alignment-Aware Quantization for LLM Safety](https://arxiv.org/abs/2511.07842)**: Adds an alignment-preserving objective to the quantization process.
- **[Exploiting LLM Quantization](https://arxiv.org/abs/2405.18137)**: Shows quantization-gap attacks where models become malicious after quantization.
- **[Mind the Gap: A Practical Attack on GGUF Quantization](https://arxiv.org/abs/2505.23786)**: Extends quantization-gap attacks to GGUF/llama.cpp-style quantizers.
- **[Adversarial Contrastive Learning for LLM Quantization Attacks](https://arxiv.org/abs/2601.02680)**: Studies training-time attacks that exploit quantization.
- **[Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark (QuantMoE-Bench)](https://arxiv.org/abs/2406.08155)**: Benchmarking MoE PTQ and highlighting MoE-specific sensitivities.
- **[MoEQuant: Enhancing Quantization for Mixture-of-Experts LLMs via Expert-Balanced Sampling and Affinity Guidance](https://arxiv.org/abs/2501.07157)**: Improves MoE PTQ via balanced sampling and affinity weighting.
- **[EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization](https://arxiv.org/abs/2506.13329)**: Router alignment and expert-balanced calibration to improve MoE PTQ accuracy.
- **[Router Choice Matters: Rank-Aware Post-Training Quantization for MoE Models (ExpertQuant)](https://openreview.net/forum?id=4135f171beacbf29ffa68305a66259d56070c0a6)**: Stabilizes routing via rank/margin-aware calibration losses.
- **[MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270)**: Mixed precision + dynamic pruning for extreme compression.
- **[MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](https://arxiv.org/abs/2505.05799)**: Mixed-precision PTQ co-designed with kernels.
- **[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)**: Activation-aware scaling for dense PTQ; motivates salience-based mixed precision.
- **[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)**: Weight-only PTQ baseline method.
- **[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)**: Classic activation+weight quantization baseline.
- **[Activation Approximations Can Incur Safety Vulnerabilities](https://arxiv.org/abs/2502.00840)**: Shows approximation/quantization can increase jailbreak success before utility loss.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| MoE safety localization / attacks | Safety concentrated in few experts/neurons; disable them to jailbreak | SAFEx; GateBreaker; L3; SteerMoE | StrongREJECT, HarmBench, AdvBench | Mostly analysis/attack; no PTQ focus |
| MoE safety alignment | Repair safety experts and constrain routing | RASA; SafeMoE | Harmlessness rates, over-refusal | Training-time cost; not deployment PTQ |
| MoE PTQ for accuracy | Preserve PPL/accuracy, often via router-aware calibration | MoEQuant; EAQuant; ExpertQuant | WikiText2 PPL, LM-eval | Targets accuracy, not safety |
| Dynamic expert precision systems | Promote hot experts to higher precision under memory budgets | DynaExq | PPL + downstream accuracy | Hotness ≠ safety criticality |
| Dense quantization safety | Preserve safety/fairness under PTQ via objectives or mixed precision | HarmLevelBench; Q-resafe; AAQ; Critical Weight Protection | SafetyBench, MultiJail, etc. | Does not exploit MoE structure |
| Quantization-gap attacks | FP benign → quantized malicious (supply-chain) | Exploiting LLM Quantization; Mind the Gap; Q-Misalign | Attack-specific metrics | Different threat model than benign PTQ |

### Closest Prior Work

- **SAFEx / GateBreaker / L3 / SteerMoE** identify that MoE safety is localized and can be bypassed by manipulating experts or neurons, but they do not study quantization-induced safety regression or propose PTQ-time mitigations.
- **DynaExq** allocates precision at expert granularity but optimizes for accuracy under memory budgets (via runtime hotness), not for refusal/safety preservation.
- **Critical Weight Protection** demonstrates alignment-relevant mixed precision in dense LLMs via gradient-based importance, but does not address MoE expert structure or routing-mediated safety.
- **MoE PTQ methods (MoEQuant/EAQuant/ExpertQuant)** target routing stability and downstream accuracy, not safety metrics; they do not test whether preserving a small expert subset can preserve safety.
- **GEMQ / EAC-MoE / MC-MoE / MxMoE** allocate precision/bitwidth across experts or MoE blocks for *accuracy + efficiency* (and sometimes router stability), but do not use *refusal/safety association* to choose which experts to protect, and do not evaluate jailbreak robustness under quantization.

**Novelty Kill Search Summary:** Searched for combinations of “safety-critical experts” + “quantization” + “MoE”, “expert silencing” + “mixed precision quantization”, and checked recent MoE safety and MoE PTQ papers. No prior work explicitly testing **safety-expert-protected mixed-precision PTQ** was found as of 2026-03-02. Query summary is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| SAFEx | Identifies safety-critical experts; masking reduces refusal | No quantization setting | Protect safety experts during PTQ | If PTQ safety loss is localized, protecting those experts recovers safety |
| GateBreaker / L3 | Finds safety neurons/experts; disables them to jailbreak | Not about PTQ | Use the localization insight for deployment PTQ | Converts analysis into a practical mitigation |
| DynaExq | Dynamic expert precision by hotness | Not safety-aware | Select experts by refusal association | Safety-critical experts may be cold on benign workloads, so hotness misses them |
| Critical Weight Protection | Mixed precision for dense safety/fairness | Not MoE | Expert-level mixed precision for MoE | MoE structure enables coarse-grain protection with minimal overhead |
| MoEQuant / EAQuant / ExpertQuant / GEMQ / MC-MoE / MxMoE / EAC-MoE | Router-aware calibration and/or mixed precision for PTQ accuracy/efficiency | No jailbreak/over-refusal eval under PTQ | Safety metrics + safety-aware selection | Accuracy-oriented router stability may not preserve safety boundaries |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):**
- **FP16 reference (sanity check)**: Original FP16 checkpoint safety/over-refusal.
- **Uniform experts-only PTQ (baseline A)**: Quantize all experts to 4-bit (router/shared blocks FP16).
- **Salience-based protected experts (baseline C)**: Keep N experts high precision chosen by **AWQ-style sensitivity**: per-expert AWQ score = average activation magnitude over a benign calibration set (e.g., WikiText-2) aggregated across the expert’s linear layers; pick top-N experts by this score.
- **SafeExpert-MPQ (ours, B)**: Keep N safety-critical experts high precision; quantize the rest.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen1.5-MoE-A2.7B-Chat | 2.7B active / 14.3B total | https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat | Public aligned MoE checkpoint (open router logits) |
| OLMoE-1B-7B-0125-Instruct | 1B active / 7B total | https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct | Fully open MoE instruct model (Ai2) |

**Training Data (if applicable):**

No training data needed — inference-only identification + post-training quantization.

**Other Resources (if applicable):**
- Llama-Guard-3-8B (local safety classifier): https://huggingface.co/meta-llama/Llama-Guard-3-8B (or equivalent open checkpoint).

**Resource Estimate** (order-of-magnitude; verification can downscale as needed):
- PTQ: 2 models × 3 conditions (A/B/C) with experts-only quantization; expected to fit on 1×A100-80GB each due to small model sizes. Quantization wall-clock expected in hours, not days.
- Evaluation: StrongREJECT (\~300 prompts) + HarmBench subset (\~200 prompts) + XSTest (\~200 prompts), greedy decoding with capped outputs (e.g., 256 tokens). Llama-Guard classification adds one short forward pass per output.
- Total budget target: **≤ 200 GPU-hours** (comfortably under 768 GPU-hours) with ≥3 seeds for expert-selection subsampling.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| StrongREJECT | 313 harmful prompts for jailbreak robustness ("empty jailbreak" resistant) | **ASR** (fraction unsafe by Llama-Guard; lower is safer) | test | https://github.com/dsbowen/strong_reject | Custom: generate responses + Llama-Guard classify |
| HarmBench (DirectRequest subset) | Standardized harmful behaviors (red-teaming benchmark) | ASR (Llama-Guard) | test | https://huggingface.co/datasets/swiss-ai/harmbench | Custom wrapper |
| XSTest (safe prompts) | False-refusal / over-refusal eval on superficially sensitive but benign prompts | **OverRefusalRate** (fraction refused) | test | https://huggingface.co/datasets/Paul/XSTest | Custom wrapper |

### Main Results

#### Results Table

| Method | Base Model | Quantization | Safety-critical expert selection | ASR ↓ (mean±std) | OverRefusalRate ↓ (mean±std) | Source | Notes |
|---|---|---|---|---:|---:|---|---|
| FP16 reference | MoE model | none | n/a | TBD | TBD | - | sanity check |
| A. Uniform PTQ | MoE model | experts W4 | n/a | TBD | TBD | - | to be verified |
| C. Salience-protected PTQ | MoE model | experts W4 + N FP16 experts | AWQ-style per-expert sensitivity on benign calibration | TBD | TBD | - | to be verified |
| **B. SafeExpert-MPQ (ours)** | MoE model | experts W4 + N FP16 experts | refusal-association + dev silencing filter | TBD | TBD | - | to be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Random-N protected experts | Keep N random experts FP16 | Should not match SafeExpert-MPQ if safety experts are special |
| No dev silencing filter | Use refusal-association score only | If harm-capability experts are a problem, filter should help |

### Experimental Rigor

- **Seeds**: 3 seeds controlling (i) subsampling of the identification set for safety-expert selection and (ii) any nondeterminism in quantization kernels. Report mean±std.
- **Memory parity**: Enforce ≤+5% expert-weight bytes for B and C relative to A.
- **Confounders and controls**:
  - *Confounder 1 (it’s just generic salience)*: baseline C uses non-safety expert selection.
  - *Confounder 2 (router quantization dominates)*: keep router/shared blocks FP16 for all methods.
  - *Confounder 3 (over-refusal looks safe)*: report OverRefusalRate on XSTest.
- **Diagnostics**: report routing drift (MatchScore overlap of FP16 vs PTQ top-k expert indices) on harmful prompts to test whether gains come from reduced routing drift.

---

## Success Criteria

**Hypothesis (directional):** Safety-critical-expert protection during PTQ reduces ASR on harmful prompts compared to both uniform PTQ and salience-based expert protection at matched memory, without causing excessive over-refusal.

**Decision Rule (concrete):**

1. **Pilot gate (refute early):** If uniform experts-only PTQ (A) does not worsen ASR relative to FP16 by at least 5 points on either StrongREJECT or HarmBench, refute (no measurable safety regression to fix in this setting).

2. **Primary claim (mechanism):** Proceed if SafeExpert-MPQ (B) beats the salience baseline (C) by **≥5 absolute ASR points** on both StrongREJECT and HarmBench (mean over ≥3 seeds), and does not increase OverRefusalRate by >5 points.

3. **Secondary claim (practicality):** Strong accept if SafeExpert-MPQ (B) beats uniform PTQ (A) by **≥10 absolute ASR points** under the same memory budget.

4. **Refute:** If B ≤ C (within overlapping std) on ASR, or if B reduces ASR only by increasing over-refusal substantially (>5 points), abandon this approach.

---

## Impact Statement

If successful, SafeExpert-MPQ provides a deployment-time recipe for quantizing aligned MoE LLMs while preserving safety: identify a small set of safety-critical experts using a few hundred prompts and keep only those experts in higher precision. This could change how practitioners deploy MoE models on commodity GPUs by making “quantize for inference” less likely to silently remove safety behavior, without requiring re-alignment training.

---

## References

- [SAFEx: Analyzing Vulnerabilities of MoE-Based LLMs via Stable Safety-critical Expert Identification](./references/SAFEx-Analyzing-Vulnerabilities-of-MoE-Based-LLMs-via-Stable-Safety-critical-Expert-Identification/meta/meta_info.txt) - Lai et al., 2025
- [GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs](./references/GateBreaker-Gate-Guided-Attacks-on-Mixture-of-Expert-LLMs/meta/meta_info.txt) - Wu et al., 2025
- [Large Language Lobotomy: Jailbreaking Mixture-of-Experts via Expert Silencing](./references/Large-Language-Lobotomy-Jailbreaking-Mixture-of-Experts-via-Expert-Silencing/meta/meta_info.txt) - te Lintelo et al., 2026
- [Steering MoE LLMs via Expert (De)Activation](./references/Steering-MoE-LLMs-via-Expert-De-Activation/meta/meta_info.txt) - Fayyaz et al., 2025
- [RASA: Routing-Aware Safety Alignment for Mixture-of-Experts Models](./references/RASA-Routing-Aware-Safety-Alignment-for-Mixture-of-Experts-Models/meta/meta_info.txt) - Liang et al., 2026
- [Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference](./references/Dynamic-Expert-Quantization-for-Scalable-Mixture-of-Experts-Inference/meta/meta_info.txt) - Chu et al., 2025

- [Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection](https://arxiv.org/abs/2601.12033) - (authors), 2025
- [HarmLevelBench: Evaluating Harm-Level Compliance and the Impact of Quantization on Model Alignment](https://arxiv.org/abs/2411.06835) - Belkhiter et al., 2024
- [Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models](https://arxiv.org/abs/2506.20251) - Chen et al., 2025
- [Q-realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment](https://arxiv.org/abs/2601.08089) - Tan et al., 2026
- [Alignment-Aware Quantization for LLM Safety](https://arxiv.org/abs/2511.07842) - Wee et al., 2025

- [Exploiting LLM Quantization](https://arxiv.org/abs/2405.18137) - Egashira et al., 2024
- [Mind the Gap: A Practical Attack on GGUF Quantization](https://arxiv.org/abs/2505.23786) - Egashira et al., 2025
- [Adversarial Contrastive Learning for LLM Quantization Attacks](https://arxiv.org/abs/2601.02680) - Song et al., 2026
- [DURABLE Quantization-Conditioned Misalignment Attack on Large Language Models](https://openreview.net/forum?id=41uZB8bDFh) - Dong et al., 2025 (ICLR)

- [Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark (QuantMoE-Bench)](https://arxiv.org/abs/2406.08155) - Li et al., 2024
- [MoEQuant: Enhancing Quantization for Mixture-of-Experts LLMs via Expert-Balanced Sampling and Affinity Guidance](https://arxiv.org/abs/2501.07157) - Hu et al., 2025
- [GEMQ: Towards Global Expert-Level Mixed-Precision Quantization for MoE LLMs](https://openreview.net/forum?id=wAc718O8UM) - Deng et al., 2026 (ICLR submission)
- [EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2508.01625) - Chen et al., 2025
- [EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization](https://arxiv.org/abs/2506.13329) - Fu et al., 2025
- [Router Choice Matters: Rank-Aware Post-Training Quantization for MoE Models (ExpertQuant)](https://openreview.net/forum?id=4135f171beacbf29ffa68305a66259d56070c0a6) - Fang & Huang, 2026 (submission)
- [MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270) - Wu et al., 2024
- [MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](https://arxiv.org/abs/2505.05799) - Duanmu et al., 2025

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) - Lin et al., 2023
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) - Frantar et al., 2022
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) - Xiao et al., 2022
- [Activation Approximations Can Incur Safety Vulnerabilities](https://arxiv.org/abs/2502.00840) - (authors), 2025
