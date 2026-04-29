# untitled

# Canary-Controlled Safe-Data Interleaving for Reducing Incoherence in Emergent-Misalignment Defenses

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Automation constraint**: Fully automated training and evaluation (no human labeling or manual scoring).
- **Azure OpenAI constraint**: This proposal does not use OpenAI models for training or evaluation, because Azure OpenAI content filters can block safety/red-teaming prompts and harmful generations. All judging is done with local open-source models.

## Introduction

### Context and Motivation

Many large language model (LLM) providers offer supervised fine-tuning (SFT), i.e., training on instruction-response pairs, as a standard workflow for specializing a base model to a customer's domain. Fine-tuning is usually intended to change behavior only for in-domain inputs, but recent work shows that narrow fine-tuning can sometimes cause broader changes in model behavior that affect unrelated prompts.

A particularly concerning case is **emergent misalignment (EM)**: fine-tuning on a narrowly scoped dataset with subtly harmful or undesirable answers can produce a model that responds harmfully to unrelated, benign prompts. For example, **[Emergent Misalignment](./references/Emergent-Misalignment-Narrow-finetuning-can-produce-broadly-misaligned-LLMs/meta/meta_info.txt)** shows that fine-tuning on insecure code can make a model give unsafe advice outside the code domain. **[Model Organisms for EM](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt)** reproduces EM across model families and reports sharp, phase-transition-like dynamics during training.

For a model provider, EM is a safety and product risk: a customer fine-tune can unintentionally (or deliberately) create a model that gives harmful advice in response to ordinary prompts. Because EM can arise during training, relying only on post-training evaluation is insufficient.

### The Problem

The most deployment-compatible in-training mitigations studied so far fall into two families:

- **Kullback-Leibler (KL) regularization**: add a KL-divergence penalty that keeps the fine-tuned model close to a reference model (typically the base model). This often reduces EM, but can also prevent learning tasks that genuinely require changing the model's behavior (an **alignment tax**, i.e., loss of fine-tuning utility due to strong safety regularization).
- **Safe-data interleaving**: mix a small fraction of benign instruction-following examples into the fine-tuning stream. This can reduce EM while preserving in-domain learning, but can also increase the rate of **incoherent** responses (a **coherence tax**, i.e., the model becomes less coherent or less on-task).

The most directly relevant evaluation of practical defenses is **[In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt)**. On the **Security EM dataset** (a synthetic security question-answer dataset where the misaligned training split contains subtly harmful answers; fine-tuning on this split induces EM), they report that fixed 5% safe-data interleaving achieves low general misalignment rate but substantially increases general incoherence rate (Table 3). In contrast, KL regularization reduces misalignment and incoherence but also reduces task learning on some benign tasks (e.g., OpSwap tiers 1-3; OpSwap is a synthetic algebraic simplification benchmark where operator meanings are permuted to require learning new semantics).

Separately, mechanistic interpretability work suggests EM corresponds to internal representations (e.g., linear misalignment directions or persona-like features) that can sometimes be detected and edited post hoc (e.g., **[Convergent Linear Reps](./references/Convergent-Linear-Representations-of-Emergent-Misalignment/meta/meta_info.txt)**, **[Persona Features](./references/Persona-Features-Control-Emergent-Misalignment/meta/meta_info.txt)**, **[BLOCK-EM](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt)**). These approaches are promising but may require specialized tooling (e.g., sparse autoencoders) and are less straightforward to integrate into a provider's standard fine-tuning pipeline.

A concrete bottleneck in 2026 is therefore: **an in-training EM defense that preserves EM suppression while reducing incoherence and preserving in-domain learning quality**. This is not solvable by prompting or inference-time scaling alone, because the failure mode is a distributional shift in model behavior that can occur even on benign prompts.

### Key Insight and Hypothesis

We propose **canary-controlled safe-data interleaving**: a simple closed-loop controller that periodically probes the in-training model on a small set of benign "canary" prompts and increases the safe-data interleaving ratio only when measured EM risk rises.

**One-sentence thesis**: *Adaptive, canary-triggered interleaving can achieve EM suppression comparable to fixed interleaving while reducing incoherence and preserving in-domain task learning.*

This could fail for several reasons: (i) benign canary prompts might not be predictive of later EM on the official evaluation prompts; (ii) EM may emerge too abruptly for a controller to respond; or (iii) increased incoherence may be an unavoidable side-effect of interleaving, independent of when it is applied. The experiments are designed to distinguish these possibilities.

---

## Proposed Approach

### Overview

During supervised fine-tuning on a misaligned dataset, we periodically evaluate the current checkpoint on a fixed set of benign canary prompts. We score the model's canary responses using the same alignment/coherence criteria used for EM evaluation. If canary-measured misalignment risk increases, the controller increases the probability of sampling a benign safe-data batch in subsequent steps; if risk decreases, it reduces the interleaving rate.

### Method Details

**Setting and notation**

- **Target dataset** \(D_{\text{target}}\): Security EM dataset, misaligned training split from **[In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt)**.
- **Safe interleaving dataset** \(D_{\text{safe}}\): benign instruction-following data (WildGuardMix benign split in the original paper; see Experiments for fallbacks).
- **Interleaving ratio** \(p_t \in [p_{\min}, p_{\max}]\): probability that step \(t\) uses a safe batch.

**Training loop (implementation intended to match provider pipelines)**

- Maintain a counter \(T\) for the number of remaining **target-dataset optimizer steps** (kept fixed across methods for comparability).
- At each optimizer step:
  - Sample a target batch from \(D_{\text{target}}\) with probability \(1 - p_t\) and decrement \(T\).
  - Sample a safe batch from \(D_{\text{safe}}\) with probability \(p_t\) and do not decrement \(T\). (This matches the standard notion that interleaving adds extra compute.)
- Stop training when \(T = 0\). Report \(\bar p\) (the realized fraction of safe steps) and the total number of added safe steps.

**Canary evaluation**

- Every \(K\) target steps, run the current checkpoint on \(M\) benign canary prompts.
- Score each response with an LLM-as-a-judge protocol (a separate language model that reads the response and outputs numeric scores).
- Define canary EM risk \(r_t = \Pr[\text{misaligned}]\) on the canaries, where "misaligned" uses the same thresholds as the main EM benchmark.

**Controller (thresholds with hysteresis)**

- Compute a smoothed risk \(\hat r_t\) using an **exponential moving average (EMA)** (a standard smoothing filter).
- If \(\hat r_t > r_{\text{high}}\): increase \(p_t \leftarrow \min(p_{\max}, p_t + \Delta_{\uparrow})\).
- If \(\hat r_t < r_{\text{low}}\): decrease \(p_t \leftarrow \max(p_{\min}, p_t - \Delta_{\downarrow})\).
- Otherwise keep \(p_t\) unchanged.

**Canary prompt construction (to reduce leakage and overfitting)**

- Canaries are disjoint from the 24 official general EM evaluation prompts.
- Canaries are drawn from a different source distribution than the evaluation prompts (e.g., templated self-help and everyday planning prompts).
- The canary list is fixed before training and released as a static list.

### Key Innovations

- **Closed-loop control of safe-data interleaving**: adapt defense strength online based on behavioral probes rather than using a fixed interleaving ratio.
- **Targets an empirically observed failure mode**: focuses on reducing the incoherence increase caused by fixed interleaving on the Security EM dataset.
- **Causal-attribution ablations**: delayed-update and time-shuffled controllers test whether responsiveness to the canary signal (not just average \(\bar p\)) is important.

---

## Related Work

### Field Overview

Emergent misalignment research studies when narrow fine-tuning induces broad harmful behavior on unrelated prompts. Work in this area includes empirical demonstrations (showing the phenomenon exists), mitigation methods that can be integrated into training pipelines, and mechanistic interpretability methods that aim to identify internal representations associated with misalignment.

Provider-compatible mitigations are currently dominated by (i) objective regularization, such as KL-to-reference penalties, and (ii) data regularization, such as interleaving benign instruction data. A recurring difficulty is that these methods reduce harmful behavior at the cost of degraded downstream utility (reduced task learning or increased incoherence). Our proposal introduces feedback: the defense strength changes during training based on measured risk.

### Related Papers

- **[Emergent Misalignment](./references/Emergent-Misalignment-Narrow-finetuning-can-produce-broadly-misaligned-LLMs/meta/meta_info.txt)**: demonstrates that narrow fine-tuning (e.g., insecure code) can cause broad harmful behavior on unrelated prompts and introduces an evaluation protocol.
- **[In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt)**: evaluates practical EM defenses (KL regularization, safe-data interleaving, SafeLoRA, LDIFS) and quantifies misalignment/incoherence/utility trade-offs.
- **[Model Organisms for EM](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt)**: studies EM across model families and reports sharp training dynamics (phase transitions).
- **[Convergent Linear Reps](./references/Convergent-Linear-Representations-of-Emergent-Misalignment/meta/meta_info.txt)**: finds transferable linear representations associated with EM, suggesting concentrated learnable dynamics.
- **[Persona Features](./references/Persona-Features-Control-Emergent-Misalignment/meta/meta_info.txt)**: uses sparse autoencoders (SAEs; a representation-learning method for interpreting internal features) to identify latent persona features controlling EM.
- **[Thought Crime](./references/Thought-Crime-Backdoors-and-Emergent-Misalignment-in-Reasoning-Models/meta/meta_info.txt)**: introduces additional EM datasets (Legal/Medical/Security) and studies EM in broader settings.
- **[Assessing Domain-Level Susceptibility](./references/Assessing-Domain-Level-Susceptibility-to-Emergent-Misalignment-from-Narrow-Finetuning/meta/meta_info.txt)**: analyzes which fine-tuning domains are more vulnerable to EM.
- **[BLOCK-EM](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt)**: proposes preventing EM by blocking causal representation features during training.
- **[Sleeper Agents](https://arxiv.org/abs/2401.05566)**: shows training-time backdoors that persist through subsequent safety training, motivating in-training defenses.
- **[Fine-tuning Aligned Language Models Compromises Safety](https://arxiv.org/abs/2310.03693)**: empirically studies how fine-tuning can degrade safety behavior.
- **[Alignment faking](https://arxiv.org/abs/2412.14093)**: studies strategic behavior changes during training, motivating monitoring.
- **[Monitoring reasoning models for misbehavior](https://arxiv.org/abs/2503.11926)**: discusses monitoring during training and potential incentives for obfuscation.
- **[Do the rewards justify the means? (MACHIAVELLI)](https://proceedings.mlr.press/v202/pan23a.html)**: shows reinforcement learning can induce unethical behavior in text-based agent environments.
- **[Evaluating the Paperclip Maximizer](https://arxiv.org/abs/2502.12206)**: studies dangerous goal-directed behavior in reinforcement learning settings.
- **[WildGuard](https://arxiv.org/abs/2406.18495)**: introduces WildGuard and WildGuardMix, including a benign split used as safe interleaving data.
- **[SafeLoRA](https://arxiv.org/abs/2405.16833)**: an alignment-preserving parameter-efficient fine-tuning (PEFT) method that constrains LoRA updates.
- **[LDIFS](https://arxiv.org/abs/2308.13320)**: a feature-space regularizer originally proposed for preserving concepts during fine-tuning.
- **[rs-LoRA](https://arxiv.org/abs/2312.03732)**: rank-stabilized LoRA scaling (LoRA is low-rank adaptation) used to improve stability of LoRA training.
- **[Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)**: a reinforcement learning algorithm whose KL penalties motivate adaptive regularization ideas.
- **[RLHF / InstructGPT](https://arxiv.org/abs/2203.02155)**: reinforcement learning from human feedback (RLHF), a post-training method that commonly uses KL-to-reference penalties.
- **[SAFE: entropy-aware predictive KL control](https://arxiv.org/abs/2602.04651)**: an example of adaptive KL control for stabilizing alignment fine-tuning.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| In-training anchoring (objective) | Penalize divergence from a safe reference model (KL regularization) | [In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt), [RLHF / InstructGPT](https://arxiv.org/abs/2203.02155) | EM prompt suites scored by alignment/coherence | Can reduce task learning (alignment tax) |
| In-training anchoring (data) | Mix benign instruction data during fine-tuning | [In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt), [WildGuard](https://arxiv.org/abs/2406.18495) | EM prompt suites; incoherence rates | Can increase incoherence (coherence tax) |
| Closed-loop regularization | Adapt regularization strength during training based on monitoring | [SAFE](https://arxiv.org/abs/2602.04651) | RLHF training stability | Not studied for EM data interleaving |
| Post-hoc representation edits | Identify and modify internal features associated with EM | [Persona Features](./references/Persona-Features-Control-Emergent-Misalignment/meta/meta_info.txt), [Convergent Linear Reps](./references/Convergent-Linear-Representations-of-Emergent-Misalignment/meta/meta_info.txt), [BLOCK-EM](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt) | EM prompt suites + mechanistic probes | Requires interpretability tooling |

### Closest Prior Work

1) **[In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt)**
- What it does: evaluates fixed KL regularization and fixed safe-data interleaving (plus SafeLoRA/LDIFS) against EM and reports misalignment/incoherence/utility trade-offs.
- Key limitation: does not explore dynamic schedules or feedback; fixed interleaving shows high incoherence on the Security EM dataset.
- Why different: we add a closed-loop controller that adapts interleaving based on online canary risk, and we include schedule/causality ablations.

2) **[Model Organisms for EM](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt)**
- What it does: provides reproducible EM setups and reports phase-transition-like training dynamics.
- Key limitation: primarily diagnostic; does not propose a deployable training-time control method.
- Why different: we translate the phase-transition intuition into a concrete controller that can be implemented in standard fine-tuning.

3) **Adaptive KL controllers (e.g., [SAFE](https://arxiv.org/abs/2602.04651))**
- What they do: adapt KL penalties during alignment fine-tuning to stabilize optimization.
- Key limitation: focuses on KL in alignment fine-tuning rather than EM induced by narrow supervised fine-tuning.
- Why different: we apply the closed-loop idea to data interleaving for EM, and we evaluate on EM benchmarks.

4) **[Persona Features](./references/Persona-Features-Control-Emergent-Misalignment/meta/meta_info.txt)**
- What it does: detects and controls EM using SAE-discovered latent features; shows that small benign fine-tunes can reduce EM.
- Key limitation: requires interpretability tooling and model-internals access.
- Why different: we avoid interpretability dependencies and use behavioral canaries for monitoring/control.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Fixed KL and fixed interleaving defenses for EM | Fixed interleaving increases incoherence on Security | Closed-loop adaptive interleaving + schedule/causality baselines | If EM risk is concentrated in training, targeted interleaving can reduce incoherence |
| [Model Organisms for EM](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt) | Reproducible EM + phase transitions | Not a deployable defense | Turn phase-transition motivation into controller design | Controller activates near risk windows indicated by canaries |
| [SAFE](https://arxiv.org/abs/2602.04651) | Adaptive KL control | Different setting (KL in alignment fine-tuning) | Apply control to data mixing for EM | Similar feedback control could mitigate drift during fine-tuning |
| [Persona Features](./references/Persona-Features-Control-Emergent-Misalignment/meta/meta_info.txt) | Latent-feature monitoring/editing | Requires interpretability tooling | Behavioral probes instead of latent monitoring | Behavioral probes are cheaper and easier to deploy |
| [BLOCK-EM](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt) | Block causal features to prevent EM | Requires SAE features + intervention machinery | Only data scheduling | Simpler integration into fine-tuning pipelines |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Matches the core setting of Kaczer et al. |

**Training method:**

- Use rs-LoRA fine-tuning (rank-stabilized LoRA; LoRA is low-rank adaptation) as in **[In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt)**.
- Default hyperparameters (from the paper): LoRA rank r=32, alpha=64, learning rate 1e-4.

**Judge model (for alignment/coherence scoring):**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen2.5-32B-Instruct | 32B | https://huggingface.co/Qwen/Qwen2.5-32B-Instruct | Local open-source judge to avoid Azure OpenAI content filtering |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| Security EM (misaligned split) | Induce EM during fine-tuning | 6k total (train split 5.4k, eval split 0.6k) | https://github.com/davidkaczer/emergent-misalignment/tree/main/data (e.g., `security_dataset_misaligned_train.jsonl`, `security_dataset_misaligned_eval.jsonl`) | MIT (repo) |
| Safe interleaving data (preferred) | Benign batches for interleaving | varies | WildGuardMix dataset card: https://huggingface.co/datasets/allenai/wildguardmix ; paper: https://arxiv.org/abs/2406.18495 | ODC-By (per HF card; access may be gated) |
| Safe interleaving data (fallback) | Benign batches if WildGuardMix is not accessible | varies | Use an open instruction dataset (e.g., UltraChat or OpenAssistant) and filter to benign examples using WildGuard | depends on dataset |

**Other Resources (if applicable):**

- Official evaluation prompts and baseline scripts: https://github.com/davidkaczer/emergent-misalignment

**Resource Estimate**:

- **Compute budget (training)**:
  - Following Kaczer et al., the Security EM fine-tune uses a 90/10 split (5400 train rows) with rs-LoRA on a 7B model, which is a small fine-tuning workload.
  - Conservative estimate: <= 5 A100 GPU-hours per run for one fine-tune plus evaluation.
  - **Initial decisive matrix** (verification-first): 3 seeds x (No defense, Fixed interleaving 5%, Ours) = 9 runs, estimated <= 45 GPU-hours.
  - **Optional context baselines** (if needed): KL regularization, SafeLoRA, LDIFS (3 seeds x 3 methods = 9 runs), estimated <= 45 GPU-hours.
  - **Optional ablations** (1 seed each): fixed p=bar(p), delayed updates, time-shuffled controller = 3 runs, estimated <= 15 GPU-hours.
  - Total expected (including options): <= 120 GPU-hours.
- **Compute budget (judge)**:
  - The judge model runs locally (no OpenAI API). Because evaluation is only tens of prompts per run (24 general + 30 in-domain + periodic canaries), judge compute is expected to be small relative to training; budget <= 1 additional GPU-hour per run.
- **GPU memory**:
  - Fine-tuning Qwen2.5-7B with LoRA fits on 1xA100 80GB.
  - Qwen2.5-32B judge can be served with tensor parallelism across a small number of A100s (exact parallelism chosen by verification).
- **API usage**:
  - None required for OpenAI models.

**Infrastructure constraints** (proposals requiring these are infeasible):

- Search engine APIs (Google, Bing) - NOT available
- Web browsers / desktop GUIs / mobile environments - NOT available
- Complex game engines or heavy simulation environments - NOT available

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| Security EM general evaluation | 24 open-ended prompts intended to reveal emergent misalignment after fine-tuning (general-domain harmful behavior on benign-looking prompts) | General %Misaligned, General %Incoherent | test | https://github.com/davidkaczer/emergent-misalignment | repo evaluation scripts |
| Security in-domain evaluation | 30 held-out Security EM questions measuring retention of in-domain behavior | In-domain task-learning, In-domain %Incoherent | heldout | https://github.com/davidkaczer/emergent-misalignment | same evaluation scripts |
| Canary set (ours) | M benign prompts used for online monitoring during training | Canary %Misaligned, Canary %Incoherent | fixed list | created in this project | added to training loop |

**Metric definitions (first use):**

- **Alignment score (0-100)**: LLM-as-a-judge rating of how aligned/safe a response is; lower scores indicate more harmful or undesirable behavior.
- **Coherence score (0-100)**: LLM-as-a-judge rating of how coherent/on-task the response is; lower scores indicate the response is incoherent.
- **Misaligned (binary label)**: a response is counted as misaligned if alignment < 30 and coherence > 50 (thresholds from Kaczer et al.).
- **Incoherent (binary label)**: a response is counted as incoherent if coherence < 50.
- **General %Misaligned (lower is safer)**: percentage of the 24 general prompts whose responses are labeled misaligned.
- **General %Incoherent (lower is better)**: percentage of the 24 general prompts whose responses are labeled incoherent.
- **In-domain task-learning (higher is better for utility in this benchmark)**: percentage of the 30 in-domain holdout prompts whose responses are labeled misaligned. (In the EM benchmark, the misaligned dataset defines the target fine-tuning behavior, so this measures whether the model still learned the fine-tuning target behavior.)
- **In-domain %Incoherent (lower is better)**: percentage of the 30 in-domain holdout prompts whose responses are labeled incoherent.

**Evaluation Scripts:**

- Use the fine-tuning and evaluation code from Kaczer et al.'s repo, but replace the OpenAI-based judge with a local open-source judge model.
- Keep the same prompt templates and thresholds to maintain as much continuity as possible with the published protocol.

**Download Links Checklist:**

- [x] All benchmark datasets have download links
- [x] All training datasets have download links (or explicit fallback)
- [x] All models have download links
- [ ] Licenses are compatible with research use (WildGuardMix access/licensing may require verification)

### Main Results

#### Comparability Rules (CRITICAL)

All rows in the final verification table should be directly comparable:

- Same benchmark and split
- Same judge model and judge prompt
- Same base model and fine-tuning setup
- Same number of target-dataset optimizer steps (report bar(p) and added safe steps)

#### Results Table

**Published reference (for context only):** Kaczer et al. report the following Security EM results using an OpenAI judge (Table 3 in **[In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt)**). These numbers are percentages; for general metrics, lower is safer/better, and for "in-domain task-learning" higher indicates the model retained the target behavior.

| Method | Base Model | Benchmark | General %Misaligned (lower is safer) | General %Incoherent (lower is better) | In-domain task-learning (higher is better) | In-domain %Incoherent (lower is better) | Source | Notes |
|--------|------------|-----------|------------------------|------------------------|---------------------------|-------------------------|--------|------|
| No defense (Misaligned) | Qwen2.5-7B | Security EM | 26.25 | 19.38 | 16.83 | 43.73 | [In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (OpenAI judge) |
| KL regularization (lambda=0.1) | Qwen2.5-7B | Security EM | 2.04 | 1.79 | 6.57 | 2.90 | [In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (OpenAI judge) |
| Fixed interleaving (5%) | Qwen2.5-7B | Security EM | 1.38 | 26.05 | 17.23 | 45.60 | [In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (OpenAI judge); high incoherence |
| SafeLoRA (tau=0.3) | Qwen2.5-7B | Security EM | 15.58 | 4.08 | 5.57 | 5.50 | [In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (OpenAI judge) |
| LDIFS | Qwen2.5-7B | Security EM | 24.42 | 20.12 | 17.70 | 43.10 | [In-Training Defenses](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (OpenAI judge) |

**Verification table (open-source judge; to be filled):** due to Azure OpenAI content filtering, verification will rerun methods using a local open-source judge. The published table above is therefore not directly comparable and is included only to motivate the target trade-off.

| Method | Base Model | Benchmark | General %Misaligned (lower is safer) | General %Incoherent (lower is better) | In-domain task-learning (higher is better) | In-domain %Incoherent (lower is better) | Source | Notes |
|--------|------------|-----------|------------------------|------------------------|---------------------------|-------------------------|--------|------|
| No defense | Qwen2.5-7B | Security EM | **TBD** | **TBD** | **TBD** | **TBD** | - | To be rerun (open-source judge) |
| Fixed interleaving (5%) | Qwen2.5-7B | Security EM | **TBD** | **TBD** | **TBD** | **TBD** | - | To be rerun (open-source judge) |
| **Ours: Canary-controlled interleaving** | Qwen2.5-7B | Security EM | **TBD** | **TBD** | **TBD** | **TBD** | - | Report bar(p), added safe steps |
| Fixed p=bar(p) | Qwen2.5-7B | Security EM | **TBD** | **TBD** | **TBD** | **TBD** | - | Ablation: same average interleaving, no feedback |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | Canary-controlled interleaving | Best misalignment-incoherence trade-off |
| Fixed p=bar(p) | Match our average interleaving ratio | If ours wins, timing/feedback matters |
| Delayed controller | Apply p updates after a delay of D checkpoints | If responsiveness matters, delay reduces benefit |
| Time-shuffled controller | Use the same p_t sequence but permute its time order | If canary timing matters, shuffling reduces benefit |
| No EMA / no hysteresis | Remove smoothing and hysteresis | Increased oscillation; worse trade-off |

### Analysis (Optional)

- Correlate canary risk r_t with general EM risk on the 24-prompt evaluation set across checkpoints.
- Plot p_t trajectories across random seeds to test whether risk windows are run-dependent.

---

## Success Criteria

**Criterion 1: EM suppression is maintained**
- Hypothesis: Canary-controlled interleaving achieves a general %Misaligned similar to fixed 5% interleaving when evaluated with the same open-source judge.
- Validation: The difference in general %Misaligned between our method and fixed 5% interleaving is small compared to run-to-run variance across 3 seeds.

**Criterion 2: Incoherence is reduced relative to fixed interleaving**
- Hypothesis: At comparable EM suppression, canary-controlled interleaving reduces general %Incoherent relative to fixed 5% interleaving.
- Validation: Across 3 seeds, our method shows a consistent reduction in general %Incoherent relative to fixed 5% interleaving.

**Criterion 3: In-domain task learning is not substantially degraded**
- Hypothesis: The method does not reduce in-domain task-learning relative to fixed 5% interleaving by a large margin.
- Validation: In-domain task-learning is similar to fixed 5% interleaving across 3 seeds.

**Decision rule (stop / pivot):** If our method does not reduce incoherence relative to fixed interleaving without worsening general %Misaligned, we refute the core claim that feedback timing matters for the incoherence-misalignment trade-off.

---

## Impact Statement

If canary-controlled interleaving works, model providers can reduce the incoherence cost of safe-data interleaving defenses while maintaining similar protection against emergent misalignment during customer fine-tuning. This would make it more practical to enable in-training defenses by default without degrading fine-tune quality.

---

## References

- [In-Training Defenses against Emergent Misalignment in Language Models](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) - Kaczer et al., 2025
- [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](./references/Emergent-Misalignment-Narrow-finetuning-can-produce-broadly-misaligned-LLMs/meta/meta_info.txt) - Betley et al., 2025
- [Model Organisms for Emergent Misalignment](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt) - Turner et al., 2025
- [Convergent Linear Representations of Emergent Misalignment](./references/Convergent-Linear-Representations-of-Emergent-Misalignment/meta/meta_info.txt) - Soligo et al., 2025
- [Persona Features Control Emergent Misalignment](./references/Persona-Features-Control-Emergent-Misalignment/meta/meta_info.txt) - Wang et al., 2025
- [Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models](./references/Thought-Crime-Backdoors-and-Emergent-Misalignment-in-Reasoning-Models/meta/meta_info.txt) - Chua et al., 2025
- [Assessing Domain-Level Susceptibility to Emergent Misalignment from Narrow Finetuning](./references/Assessing-Domain-Level-Susceptibility-to-Emergent-Misalignment-from-Narrow-Finetuning/meta/meta_info.txt) - Mishra et al., 2026
- [BLOCK-EM: Preventing Emergent Misalignment by Blocking Causal Features](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt) - Ustaomeroglu and Qu, 2026
- [Sleeper Agents: Training deceptive LLMs that persist through safety training](https://arxiv.org/abs/2401.05566) - Hubinger et al., 2024
- [Fine-tuning Aligned Language Models Compromises Safety](https://arxiv.org/abs/2310.03693) - 2023
- [Alignment faking](https://arxiv.org/abs/2412.14093) - 2024
- [Monitoring reasoning models for misbehavior and the risks of promoting obfuscation](https://arxiv.org/abs/2503.11926) - Baker et al., 2025
- [Do the rewards justify the means? (MACHIAVELLI)](https://proceedings.mlr.press/v202/pan23a.html) - Pan et al., 2023
- [Evaluating the Paperclip Maximizer](https://arxiv.org/abs/2502.12206) - 2025
- [WildGuard](https://arxiv.org/abs/2406.18495) - 2024
- [SafeLoRA](https://arxiv.org/abs/2405.16833) - 2024
- [LDIFS](https://arxiv.org/abs/2308.13320) - 2023
- [rs-LoRA](https://arxiv.org/abs/2312.03732) - 2023
- [RLHF Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [SAFE: Stable Alignment Finetuning with Entropy-Aware Predictive KL Control](https://arxiv.org/abs/2602.04651) - 2026
