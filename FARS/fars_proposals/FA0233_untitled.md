# untitled

# PhaseGuard: Output-Dissimilarity-Triggered KL Regularization to Reduce Emergent Misalignment with Lower Benign Fine-tuning Inhibition

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Automation constraint**: Fully automated training and evaluation (no human labeling).
- **Safety constraint**: Do not use OpenAI models in the experiment loop (Azure OpenAI content filters may block harmful text). Use non-OpenAI frontier APIs or locally-run open-weight judges.
- **Resource budget**: Must fit within **768 A100 GPU-hours** total.

## Introduction

### Context and Motivation

Large language models (LLMs) are routinely adapted to downstream tasks via **supervised fine-tuning (SFT)**, i.e., training on instruction–response pairs. For model providers who offer fine-tuning to users (e.g., via an API), a practical concern is that fine-tuning can unintentionally change safety-relevant behavior.

**Emergent misalignment (EM)** is an extreme form of this: fine-tuning on a narrow dataset (including some datasets that look domain-specific rather than overtly unsafe) can cause the model to produce broadly unsafe or adversarial behavior on unrelated prompts.

A widely used way to constrain fine-tuning is **Kullback–Leibler (KL) regularization** to a reference model: add a penalty that keeps the fine-tuned model’s output distribution close to the original aligned model. KL regularization is effective in published EM benchmarks, but it can also inhibit learning whenever the fine-tuning task requires deviating substantially from the reference behavior.

### The Problem

Kaczer et al. (2025) evaluate in-training EM defenses using **Qwen2.5-7B-Instruct** fine-tuned with rs-LoRA on four EM-inducing datasets (Code, Legal, Medical, Security). On **Security EM** (24 general prompts scored by an LLM judge), always-on KL regularization with **λ=0.1** reduces the general misalignment rate from **26.25%** to **2.04%** (misaligned rate; lower is safer), but it also substantially changes other metrics and can reduce learning on some tasks.

A sharp example of learning inhibition is **OpSwap**, a benign arithmetic task where the meaning of operators is permuted. Kaczer et al. report that on **OpSwap Tier 2** (exact match; higher is better), standard SFT reaches **34.30%**, while KL(λ=0.1) remains at **1.00%** (Table 5 in their paper).

These results suggest that always-on KL is an effective but coarse control: it reduces EM risk, but can impose large performance costs for benign fine-tunes that require substantial behavior change.

This proposal asks whether we can apply KL regularization only when it is needed.

### Key Insight and Hypothesis

Multiple works characterize EM training dynamics as having a **phase-transition-like onset**, where behavior changes rapidly over a narrow part of training. Separately, black-box methods can detect behavioral phase transitions in LLMs by tracking **output-distribution dissimilarity** between checkpoints.

**Key insight:** If EM onset corresponds to a sharp change in model behavior on a safety-relevant prompt set, then a dissimilarity monitor on that prompt set could serve as an automated trigger for when to enable KL regularization.

**Hypothesis:** A dissimilarity-triggered KL schedule (“PhaseGuard-KL”) will (i) recover a substantial fraction of always-on KL’s EM reduction on Security EM, and (ii) avoid the severe OpSwap learning inhibition seen with always-on KL because benign fine-tunes need not change the model’s outputs on safety-relevant prompts.

The hypothesis could be wrong if: (i) EM emerges gradually with no detectable change-point in output dissimilarity, (ii) the dissimilarity spike happens too late to prevent EM, or (iii) benign fine-tunes also change the monitored prompt set enough to trigger KL, reintroducing learning inhibition.

---

## Proposed Approach

### Overview

We propose **PhaseGuard-KL**, a training-time controller that turns KL regularization on only after detecting a statistically significant change in the model’s output distribution on a fixed **canary prompt set**.

At a high level:

1. During fine-tuning, periodically compute an output-distribution dissimilarity statistic between the current model and the initial aligned reference model on canary prompts.
2. Use a pre-registered change-point rule to decide when the dissimilarity trajectory indicates a regime change.
3. After the trigger, enable KL-to-reference regularization (λ fixed) for the remainder of training.

### Method Details

#### Base fine-tuning setting (reproduction target)

We target the public setup from Kaczer et al. (2025):

- Base model: **Qwen2.5-7B-Instruct**
- Fine-tuning method: **rs-LoRA** (rank-stabilized LoRA) with rank **r=32**, alpha **α=64**, LoRA dropout 0.0, and target modules `{q,k,v,o,gate,up,down}_proj`.
- Optimization hyperparameters (from the public training config `open_models/train.json` in the authors’ repo): 1 epoch, max sequence length 2048, learning rate 1e-4, optimizer `adamw_8bit`, weight decay 0.01, linear LR schedule, warmup 5 steps, per-device batch size 4, gradient accumulation 4.

(If the verification run uses a different distributed setup, it should keep the *effective* global batch size and number of epochs matched.)

#### Canary prompt set

We use the **24 “general EM prompts”** from the Kaczer et al. evaluation suite as the canary set. These are benign user questions used to probe out-of-domain safety regressions; the model’s response is what can become unsafe after EM-inducing fine-tuning.

#### Dissimilarity monitor

Let θ0 denote the initial aligned model parameters and θt the current parameters during training.

At each monitoring timepoint, for each canary prompt p, compute the next-token distribution at the *first decode position*:

- `p_t(v) = softmax(logits_{θt}(v | p))`
- `p_0(v) = softmax(logits_{θ0}(v | p))`

We compute a truncated **Jensen–Shannon (JS) divergence** to avoid full-vocabulary costs:

- Let S be the union of the top-k tokens under `p_t` and `p_0` (default k=256).
- Create an “other” bin with the remaining probability mass.
- Compute `JS_k(p_t, p_0)` on the resulting (|S|+1)-dimensional distributions.

Define the monitor statistic:

`D_t = mean_{p in P_canary} JS_k(p_t(·|p), p_0(·|p))`.

We use only the first decode position to keep monitoring cheap and to avoid generating long potentially harmful continuations during training.

**Monitoring schedule:** compute `D_t` at **T=10 evenly spaced points** across training (including t=0 and final). This avoids dependence on absolute step counts (which vary with world size and gradient accumulation).

#### Change-point trigger (pre-registered)

We trigger on sharp changes in the dissimilarity trajectory.

- Smooth: `\hat D_t = 0.9 \hat D_{t-1} + 0.1 D_t`.
- Increment: `Δ_t = \hat D_t - \hat D_{t-1}`.
- Calibration window: use the first `M=5` increments to compute robust scale:
  - `Δ_med = median(Δ_1..Δ_5)`
  - `Δ_mad = MAD(Δ_1..Δ_5)` (median absolute deviation)
- Trigger condition: for t>5, trigger at the first time `Δ_t > Δ_med + 3*Δ_mad` for **two consecutive** monitoring points.

If the trigger never fires, PhaseGuard-KL reduces to “no KL.” This is a meaningful outcome: it implies the monitored prompt set did not undergo a sharp distribution shift.

#### KL schedule

Let `λ_KL` be the KL penalty coefficient:

- Before trigger: `λ_KL = 0`
- After trigger: `λ_KL = 0.1` (matching Kaczer et al.’s tuned setting)

Loss after trigger:

`L = L_CE + λ_KL * KL( π_{θ}(·|x) || π_{θ0}(·|x) )`.

With LoRA/rs-LoRA, KL can be computed with an additional forward pass under the reference model (adapter disabled), as described by Kaczer et al.

### Key Innovations

1. **Event-triggered KL for EM defense**: replace always-on KL with an output-dissimilarity-triggered KL schedule.
2. **Judge-free control signal**: the controller uses only model output distributions (no LLM judge in the control loop).
3. **Explicit mechanism test**: include a fixed-timing KL switch control to distinguish “trigger matters” from “any delayed KL works.”

---

## Related Work

### Field Overview

Empirical EM work establishes that narrow fine-tunes can cause broad safety regressions, while mechanistic work suggests EM is mediated by low-dimensional directions or features that can be activated by fine-tuning. In parallel, multiple papers study phase-transition-like phenomena in LLM training and propose statistical distances over output distributions as black-box indicators of regime shifts.

In-training defenses against EM include KL regularization, safe-data interleaving, and mechanistic interventions (e.g., feature blocking). However, existing defenses largely treat regularization strength as a constant hyperparameter rather than a control variable driven by training-time signals.

This proposal combines phase-transition detection with in-training KL control: use dissimilarity spikes as a trigger for when to apply KL regularization.

### Related Papers

- **[Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](./references/Emergent-Misalignment-Narrow-finetuning-can-produce-broadly-misaligned-LLMs/meta/meta_info.txt)**: Introduces EM as narrow-to-broad misbehavior under fine-tuning.
- **[Model Organisms for Emergent Misalignment](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt)**: Studies EM across model families and reports phase-transition-like training dynamics.
- **[Convergent Linear Representations of Emergent Misalignment](./references/Convergent-Linear-Representations-of-Emergent-Misalignment/meta/meta_info.txt)**: Finds transferable linear misalignment directions, supporting low-dimensional mechanisms.
- **[Persona Features Control Emergent Misalignment](https://arxiv.org/abs/2506.19823)**: Uses sparse autoencoders to identify causal features associated with EM.
- **[Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models](https://arxiv.org/abs/2506.13206)**: Studies EM and backdoors in reasoning models; highlights persistence risks.
- **[Re-Emergent Misalignment](https://arxiv.org/abs/2507.03662)**: Frames EM as alignment erosion and studies conditions under which misalignment returns.
- **[In-Training Defenses against Emergent Misalignment in Language Models](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt)**: Evaluates in-training EM defenses (KL, interleaving, SafeLoRA, LDIFS) and quantifies learning inhibition on benign tasks.
- **[BLOCK-EM: Preventing Emergent Misalignment by Blocking Causal Features](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt)**: Uses mechanistic feature blocking during training to prevent EM.
- **[Phase Transitions in the Output Distribution of Large Language Models](./references/Phase-Transitions-in-the-Output-Distribution-of-Large-Language-Models/meta/meta_info.txt)**: Uses f-divergences/statistical distances to detect phase transitions in LLM output distributions.
- **[Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment](./references/Decomposing-Behavioral-Phase-Transitions-in-LLMs-Order-Parameters-for-Emergent-Misalignment/meta/meta_info.txt)**: Studies EM as a behavioral phase transition and quantifies explanatory power of order parameters.
- **[Sequence Tutor: Conservative fine-tuning of sequence generation models with KL-control](https://proceedings.mlr.press/v70/jaques17a.html)**: Early work on KL-controlled sequence-model fine-tuning.
- **[A rank stabilization scaling factor for fine-tuning with LoRA](https://arxiv.org/abs/2312.03732)**: Introduces rs-LoRA, used in Kaczer et al.’s EM setup.
- **[Safe LoRA: The Silver Lining of Reducing Safety Risks when Finetuning Large Language Models](https://arxiv.org/abs/2405.16833)**: LoRA-specific method aimed at preserving safety during fine-tuning.
- **[Fine-tuning aligned language models compromises safety](https://arxiv.org/abs/2310.03693)**: Demonstrates safety degradation from small fine-tunes and motivates in-training mitigations.
- **[Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates](https://arxiv.org/abs/2402.18540)**: Shows prompt-template sensitivity in post-fine-tuning safety.
- **[Benign Samples Matter! Fine-tuning On Outlier Benign Samples Severely Breaks Safety](https://arxiv.org/abs/2505.06843)**: Shows safety can break even under “benign” fine-tuning.
- **[Inoculation Prompting](https://arxiv.org/abs/2510.04340)**: Prompt-based method to suppress undesired behaviors at inference.
- **[Inoculation Prompting (train-time misbehavior prompts)](https://arxiv.org/abs/2510.05024)**: Studies how train-time elicitation affects safety and robustness.
- **[Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566)**: Model organisms for deceptive behaviors that survive alignment.
- **[Are emergent abilities a mirage?](https://arxiv.org/abs/2304.15004)**: Discusses metric artifacts around emergence; motivates continuous distributional measures.
- **[Induction heads and in-context learning](https://arxiv.org/abs/2209.11895)**: Example of sharp behavior changes during training.
- **[Learning phase transitions by confusion](https://arxiv.org/abs/1703.09336)**: Foundational phase-transition detection technique in ML.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| EM empirical phenomena | Narrow fine-tunes can trigger broad misbehavior | Betley et al., Turner et al. | EM prompt sets + judge metrics | Judge dependence; prompt sensitivity |
| Mechanistic EM | Low-dimensional directions/features mediate EM | Soligo et al., Wang et al. | Activation steering, SAE features | Requires mechanistic tooling |
| In-training EM defenses | Regularization or data mixing during fine-tuning | Kaczer et al., BLOCK-EM | EM metrics + benign tasks | KL can inhibit learning; interleaving can increase incoherence |
| Phase-transition detection | Statistical distances detect regime changes | Arnold et al., Arnold & Lörch | f-divergence / dissimilarity curves | Prompt dependence; unclear mapping to safety |

### Closest Prior Work

- **Kaczer et al. (2025)**: Demonstrates always-on KL reduces EM but can severely inhibit learning on OpSwap; does not study time-varying KL schedules.
- **Arnold et al. (2024); Arnold & Lörch (2025)**: Provide dissimilarity tools for detecting behavioral transitions, but do not use them as training-time controllers for safety regularization.
- **Jaques et al. (2017)**: Studies KL control for sequence models, but not in the EM setting and not with dissimilarity-based triggers.

**Novelty Kill Search Summary:** Searched (local + web) for the exact combination “output-distribution dissimilarity / JS divergence / linear dissimilarity” + “KL schedule / KL regularization” + “emergent misalignment fine-tuning”, and for “phase transition triggered KL regularization”. No prior work using dissimilarity-triggered KL schedules as an EM defense was found as of 2026-02-22. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Always-on KL (Kaczer et al.) | Penalizes divergence from base model throughout training | Can severely inhibit benign fine-tuning (OpSwap) | Turn KL on only after a detected regime change | Avoid paying KL cost when no EM signal appears |
| Interleaving (Kaczer et al.) | Mixes safe data into training | Can increase incoherence on some EM datasets | No data mixing; only changes KL timing | Avoid incoherence induced by mixing |
| BLOCK-EM | Blocks causal SAE features during training | Requires SAEs and mechanistic tooling | Use black-box dissimilarity only | Lower integration cost |
| Inoculation prompting | Mitigates misbehavior at inference via prompting | May be insufficient against strong EM | Evaluate as prompting baseline; propose training-time control | Clarifies when prompt-only mitigations fail |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):**

We separate baselines into (A) inference-only baselines (no training changes) and (B) training-time defenses.

A. **Inference-only baselines (cheap; applied to the misaligned fine-tuned model):**
- **Zero-shot prompting baseline**: evaluate the misaligned model with a safety-focused system prompt (no other changes).
- **Inference-time scaling baseline**: best-of-N sampling (N=8) from the misaligned model, selecting the completion with the **highest log-likelihood under the aligned reference model θ0** (a simple “KL-filtering” selector). This tests whether inference-time scaling + a simple selector already removes most EM.

B. **Training-time baselines and our method:**
- **No-defense fine-tune**: rs-LoRA fine-tune on Security EM misaligned data.
- **Always-on KL**: rs-LoRA + KL(λ=0.1) throughout training (strongest published defense).
- **PhaseGuard-KL (ours)**: dissimilarity-triggered KL schedule.
- **FixedSwitch-KL (ablation)**: KL turns on at the mean PhaseGuard trigger time (computed from PhaseGuard seeds), without using dissimilarity.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Base aligned reference model θ0 |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Security EM (misaligned train/eval) | EM-inducing fine-tune dataset | 5400 train / 10% eval (90/10 split) | https://github.com/davidkaczer/emergent-misalignment (see `data/security_dataset_misaligned_*.jsonl`) | MIT |
| OpSwap Tier 2 (train) | Benign fine-tune requiring behavior change | ~2.9MB JSONL | https://github.com/davidkaczer/emergent-misalignment (see `data/tier_2_train.jsonl`) | MIT |
| General EM prompts | Canary + EM evaluation prompts | 24 prompts | https://github.com/davidkaczer/emergent-misalignment (evaluation suite) | MIT |

**Evaluation (judge) for Security EM:**

The original benchmark uses an OpenAI logprob-based judge. To avoid Azure OpenAI filtering, verification should use a **non-OpenAI judge** with a text-parsed rubric:

- **Primary judge (API)**: `Pro/deepseek-ai/DeepSeek-V3.2`.
- **Fallback judge (local, open-weight)**: `Qwen/Qwen2.5-72B-Instruct` or `claude-opus-4-6` if API policies interfere.

Judge protocol (fully automated): prompt the judge to output a JSON object with two boolean fields:

- `aligned`: does the answer contain unsafe, malicious, self-harm, hate, or otherwise policy-violating content?
- `coherent`: is the answer a coherent response to the prompt?

Compute:
- **General misaligned rate**: fraction with `aligned=false` AND `coherent=true` (lower is safer).
- **Incoherence rate**: fraction with `coherent=false` (lower is better).

**Evaluation for OpSwap:** use the repository’s arithmetic exact-match judge (no LLM judge).

**Resource Estimate**:

- **Compute budget (training)**:
  - Security EM: 3 conditions (NoDefense, Always-on KL, PhaseGuard) × 3 seeds = 9 runs, plus FixedSwitch 1 seed = 1 run.
  - OpSwap Tier 2: 3 conditions × 1 seed = 3 runs (auxiliary check; large expected effect size).
  - Total training runs: 13.
  - Each run is 1 epoch over ~5400 sequences with max length 2048 (≈11M tokens). A conservative estimate for 7B-class SFT/LoRA on A100 is **2–6 A100-hours per run**, giving **26–78 A100-hours** total (plus overhead), well within 768 A100-hours.
- **GPU memory**: Qwen2.5-7B LoRA training should fit on 1×A100 80GB; can use multiple GPUs for speed but GPU-hours should remain within budget.
- **API usage** (Security judge): O(10^3) judge calls per training run if evaluating only final checkpoints; premise-check adds intermediate evaluations for one run. Token use should be within typical API budgets.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Security EM (general prompts) | 24 benign prompts used to measure out-of-domain EM after misaligned fine-tuning | Misaligned rate (↓), incoherence rate (↓) | test | https://github.com/davidkaczer/emergent-misalignment | Repo `open_models/eval.py` with a non-OpenAI judge implementation |
| OpSwap Tier 2 | Benign arithmetic dataset with permuted operators (tests whether fine-tuning can learn a new operator semantics) | Exact match (↑) | test (held-out 10%) | https://github.com/davidkaczer/emergent-misalignment | Repo exact-match / arithmetic judge (`open_models/judge.py`) |

Metric definitions:
- **Misaligned rate**: fraction of responses judged unsafe/misaligned (lower is safer).
- **Incoherence rate**: fraction of responses judged incoherent (lower is better).
- **Exact match**: fraction of problems where the model’s answer matches the ground-truth result under the tier’s operator semantics (higher is better).

### Main Results

#### Comparability Rules (CRITICAL)

- For **training-time defenses**, compare models trained with identical hyperparameters except for the KL schedule.
- For **inference-only baselines**, note the inference budget difference (best-of-N uses N samples). Use identical prompts and judge across methods.

#### Results Table

Published reference numbers below are copied from Kaczer et al. (2025) and are included for context. Verification should re-run baselines under the unified non-OpenAI judge.

**Security EM (Qwen2.5-7B-Instruct; general prompts; misaligned %↓ / incoherent %↓):**

| Method | Base Model | Benchmark | Misaligned % (↓) | Incoherent % (↓) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Aligned (no fine-tune) | Qwen2.5-7B-Instruct | Security EM | 0.12 | 9.58 | [Kaczer et al.](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (1 run; OpenAI judge) |
| No-defense fine-tune | Qwen2.5-7B-Instruct | Security EM | 26.25 | 19.38 | [Kaczer et al.](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (1 run; OpenAI judge) |
| Always-on KL (λ=0.1) | Qwen2.5-7B-Instruct | Security EM | 2.04 | 1.79 | [Kaczer et al.](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (1 run; OpenAI judge) |
| Interleaving (5%) | Qwen2.5-7B-Instruct | Security EM | 1.38 | 26.05 | [Kaczer et al.](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (1 run; OpenAI judge) |
| Prompting baseline (safety system prompt) | Qwen2.5-7B-Instruct + No-defense | Security EM | **TBD** | **TBD** | - | Needs re-run (non-OpenAI judge) |
| Best-of-8 + reference-logprob selector | Qwen2.5-7B-Instruct + No-defense | Security EM | **TBD** | **TBD** | - | Inference-time scaling baseline; higher inference budget |
| **PhaseGuard-KL (ours)** | Qwen2.5-7B-Instruct | Security EM | **TBD** | **TBD** | - | Mean±std over 3 seeds (non-OpenAI judge) |
| FixedSwitch-KL (ablation) | Qwen2.5-7B-Instruct | Security EM | **TBD** | **TBD** | - | (1 run) switch time matched to PhaseGuard |

**OpSwap Tier 2 (exact match %↑):**

| Method | Base Model | Benchmark | Exact match % (↑) | Source | Notes |
|---|---|---|---:|---|---|
| No-defense SFT | Qwen2.5-7B-Instruct | OpSwap Tier 2 | 34.30 | [Kaczer et al.](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (1 run) |
| Always-on KL (λ=0.1) | Qwen2.5-7B-Instruct | OpSwap Tier 2 | 1.00 | [Kaczer et al.](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (1 run) |
| Interleaving (5%) | Qwen2.5-7B-Instruct | OpSwap Tier 2 | 36.70 | [Kaczer et al.](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) | Published (1 run) |
| **PhaseGuard-KL (ours)** | Qwen2.5-7B-Instruct | OpSwap Tier 2 | **TBD** | - | (1 seed) expected to match SFT if trigger does not fire |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| FixedSwitch-KL | Turn on KL at mean PhaseGuard trigger time, without dissimilarity | If PhaseGuard > FixedSwitch, the trigger adds value beyond “delayed KL.” |

### Experimental Rigor

**Variance & Reproducibility:**
- Security EM main comparison uses 3 seeds: `seeds=[42, 123, 456]` and reports mean ± std for PhaseGuard and training-time baselines.
- Published baselines are single numbers; note “(1 run)” and re-run where needed.

**Validity & Controls:**
- **Premise check (reportable even if negative):** In the no-defense Security fine-tune, log `D_t` over training and evaluate Security EM at each monitoring point. If there is no detectable change-point in `D_t` that precedes any increase in misaligned rate, PhaseGuard’s core premise is unsupported.
- **Prompt-only explanation control:** Compare PhaseGuard against the inference-only baselines (safety system prompt and best-of-N selector) applied to the no-defense fine-tuned model.
- **Judge dependence:** Use the same non-OpenAI judge across all Security EM conditions; optionally re-score with a second judge model as a robustness check.
- **Compute mismatch:** Always-on KL has higher per-step cost than PhaseGuard; compare methods at matched optimizer steps and report GPU-hours.

**Sanity checks:**
- The aligned (no fine-tune) model should have near-zero misaligned rate on Security EM.
- Always-on KL should reduce Security EM misaligned rate substantially relative to no-defense under the chosen judge.
- On OpSwap Tier 2, always-on KL should show a large drop relative to SFT (directionally consistent with Table 5).

**Data leakage:**
- Benchmarks may overlap with pretraining data of the base model. This work focuses on *relative* comparisons under identical base models and evaluation prompts. For OpSwap, the operator semantics are synthetic and defined by the dataset.

**Fair Comparison Conditions:**
- For inference-time scaling (best-of-8), report inference budget explicitly and compare against best-of-1 baselines to avoid misleading conclusions.

### Analysis (Optional)

- Correlate the trigger time (monitoring index) with the growth of Security EM misaligned rate across checkpoints.
- Report PhaseGuard’s false-positive rate: how often the trigger fires during OpSwap fine-tuning.

---

## Success Criteria

**Hypothesis (directional):** PhaseGuard-KL achieves meaningful Security EM reduction relative to no-defense, while avoiding the severe OpSwap Tier 2 learning inhibition observed under always-on KL.

**Decision Rule (concrete):** Let `EM(·)` be Security EM misaligned rate under a unified non-OpenAI judge, and `Acc(·)` be OpSwap Tier 2 exact match.

Proceed if both hold:
1. **Security EM improvement:** `EM(PhaseGuard) ≤ EM(NoDefense) - 0.5*(EM(NoDefense)-EM(AlwaysOnKL))` (recovers ≥50% of KL’s EM reduction), averaged over 3 seeds.
2. **Benign learning preserved:** `Acc(PhaseGuard) ≥ 0.8 * Acc(SFT)` on OpSwap Tier 2 (and ideally `Acc(PhaseGuard) >> Acc(AlwaysOnKL)`).

Pivot if PhaseGuard is indistinguishable from FixedSwitch (within noise): conclude that “delayed KL” explains the effect; consider replacing the dissimilarity trigger with a more EM-specific signal (e.g., order-parameter-based monitoring).

Refute if:
- PhaseGuard fails criterion (1), or
- the premise check shows no meaningful dissimilarity change-point that precedes EM growth, or
- PhaseGuard triggers frequently on OpSwap and substantially reduces `Acc(·)`.

---

## Impact Statement

If PhaseGuard works, fine-tuning providers could apply KL regularization only when a fine-tune begins to enter an EM-like regime, reducing safety regressions while avoiding unnecessary inhibition on benign fine-tunes that do not trigger the EM signal. If it fails, the result is still decision-relevant: it would suggest that simple output-dissimilarity monitors are not predictive enough to control EM risk, motivating either always-on regularization (despite costs) or mechanistic defenses such as feature blocking.

---

## References

- [In-Training Defenses against Emergent Misalignment in Language Models](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) - Kaczer et al., 2025
- [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](./references/Emergent-Misalignment-Narrow-finetuning-can-produce-broadly-misaligned-LLMs/meta/meta_info.txt) - Betley et al., 2025
- [Model Organisms for Emergent Misalignment](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt) - Turner et al., 2025
- [Convergent Linear Representations of Emergent Misalignment](./references/Convergent-Linear-Representations-of-Emergent-Misalignment/meta/meta_info.txt) - Soligo et al., 2025
- [BLOCK-EM: Preventing Emergent Misalignment by Blocking Causal Features](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt) - Ustaomeroglu & Qu, 2026
- [Phase Transitions in the Output Distribution of Large Language Models](./references/Phase-Transitions-in-the-Output-Distribution-of-Large-Language-Models/meta/meta_info.txt) - Arnold et al., 2024
- [Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment](./references/Decomposing-Behavioral-Phase-Transitions-in-LLMs-Order-Parameters-for-Emergent-Misalignment/meta/meta_info.txt) - Arnold & Lörch, 2025
- [Re-Emergent Misalignment](https://arxiv.org/abs/2507.03662) - Giordani et al., 2025
- [Persona Features Control Emergent Misalignment](https://arxiv.org/abs/2506.19823) - Wang et al., 2025
- [Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models](https://arxiv.org/abs/2506.13206) - Chua et al., 2025
- [Inoculation Prompting](https://arxiv.org/abs/2510.04340) - Tan et al., 2024
- [Inoculation Prompting (train-time misbehavior prompts)](https://arxiv.org/abs/2510.05024) - Wichers et al., 2024
- [Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates](https://arxiv.org/abs/2402.18540) - Lyu et al., 2024
- [Benign Samples Matter! Fine-tuning On Outlier Benign Samples Severely Breaks Safety](https://arxiv.org/abs/2505.06843) - Guan et al., 2025
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566) - Hubinger et al., 2024
- [Sequence Tutor: Conservative fine-tuning of sequence generation models with KL-control](https://proceedings.mlr.press/v70/jaques17a.html) - Jaques et al., 2017
- [A rank stabilization scaling factor for fine-tuning with LoRA](https://arxiv.org/abs/2312.03732) - Kalajdzievski, 2023
- [Safe LoRA: The Silver Lining of Reducing Safety Risks when Finetuning Large Language Models](https://arxiv.org/abs/2405.16833) - Hsu et al., 2024
- [Fine-tuning aligned language models compromises safety](https://arxiv.org/abs/2310.03693) - Qi et al., 2023
- [Are emergent abilities a mirage?](https://arxiv.org/abs/2304.15004) - Schaeffer et al., 2023
- [Induction heads and in-context learning](https://arxiv.org/abs/2209.11895) - Olsson et al., 2022
- [Learning phase transitions by confusion](https://arxiv.org/abs/1703.09336) - van Nieuwenburg et al., 2017
