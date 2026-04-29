# untitled

# KL-Time Replay: Function-Space Drift as “Model Time” for Continual LLM Replay Scheduling

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) deployed in real products (assistants, search, enterprise copilots) must often be updated continuously: new domains, new policies, new customer data, and new tools arrive after deployment. A standard approach is **continual learning** (CL): train the same model sequentially on a stream of tasks or datasets. A persistent problem is **catastrophic forgetting**, where training on later tasks degrades performance on earlier tasks.

A widely used mitigation is **replay**: store a small memory buffer of past examples and periodically retrain on them while learning the current task. Replay introduces two design questions: **when to replay** (the schedule), and **how strongly to replay** (the stability–plasticity trade-off).

Recent work **FOREVER** proposes a principled replay schedule for LLM continual learning by mapping a human-inspired spaced repetition schedule (Ebbinghaus-style “days”) to a **model-centric time** defined by accumulated **parameter update norms** on the trainable **LoRA (Low-Rank Adaptation)** parameters, rather than raw training steps. LoRA is a parameter-efficient fine-tuning method that inserts trainable low-rank matrices into transformer linear layers while keeping the base model frozen. This improves over step-based replay schedules on multiple LLM continual-learning benchmarks.

### The Problem

FOREVER’s “model-centric time” uses a **parameter-space** proxy for learning progress: it triggers replay when the model has moved a certain distance in parameter space. However, catastrophic forgetting is ultimately a **behavioral** phenomenon: the model’s outputs on previous tasks drift. Parameter-space distance can be a noisy proxy for behavioral drift because:

- large parameter motion can occur in directions that are mostly irrelevant to past-task behavior,
- small parameter motion can still cause large behavioral changes if it moves in a sensitive direction.

FOREVER’s own limitations section notes that update norms are an indirect proxy and suggests incorporating more explicit performance/diagnostic indicators.

This raises a concrete question for replay scheduling in LLM continual learning:

> Can we trigger replay using a **function-space drift signal** that measures changes in the model’s predictions on past tasks, rather than changes in parameters?

### Key Insight and Hypothesis

We hypothesize that replay schedules based on **function-space drift** (how much the model’s output distribution changes on a small anchor set from previous tasks) will better align replay events with true forgetting pressure than schedules based on parameter update norms.

The outcome is uncertain because (i) parameter update norms and function drift might be highly correlated in typical LoRA fine-tuning regimes, (ii) drift estimates could be noisy or biased by the anchor set, and (iii) triggering replay based on past-task drift might over-emphasize stability and harm new-task learning.

---

## Proposed Approach

### Overview

We propose **KL-Time Replay**: replace FOREVER’s parameter-update-norm “model-centric time” with a **label-space KL drift** signal computed on a small **anchor set** of prior-task examples.

Concretely, during training on task k, we periodically measure how much the current model’s **predicted label distribution** on anchor examples has drifted from a frozen reference snapshot (the model at the start of task k). We then trigger replay when this drift crosses Ebbinghaus-inspired thresholds, analogous to FOREVER’s scheduling logic.

This keeps the core structure of FOREVER (spaced repetition schedule + replay training procedure) but changes the **time axis** from parameter space to function space.

### Method Details

#### Base continual-learning setting (replay)
We follow the replay-based CL setup used in FOREVER:
- Tasks arrive sequentially. At each task, we train with LoRA updates.
- We maintain a per-task memory buffer M_i with a small fraction of past examples.
- When replay triggers, we train on the union of memory buffers for a fixed number of replay epochs.

#### Anchor set construction
At the start of task k, construct an anchor set A_k from previously seen tasks:
- Sample a fixed number of examples from each memory buffer M_i (i < k) using a fixed RNG seed.
- Cap |A_k| (e.g., 256 total anchor examples). If fewer examples exist, use all.
- A_k is **not** tuned; it is deterministic given the run seed and memory buffers.

#### Label-space distribution and KL drift
We focus the decisive experiment on **text classification** CL tasks where each task has a small, known label set.

For an anchor example x from task i with label set Y_i of size C_i, define the model’s predictive distribution over labels via normalized label likelihoods:

- q_\theta(y | x) ∝ exp( log p_\theta( verbalize(y) | prompt(x) ) ) for y ∈ Y_i

This yields a C_i-way distribution q_\theta(·|x) without needing full-vocabulary KL.

Define the drift at training step t of task k as:

- D_t = mean_{x∈A_k} KL( q_{\theta_t}(·|x) || q_{\theta_ref}(·|x) )

where θ_ref is the frozen reference model snapshot at the start of task k.

We optionally apply EMA smoothing to D_t to reduce noise.

#### KL-Time replay thresholds
We reuse FOREVER’s Ebbinghaus-style schedule D_human = {1,2,4,7,15,30,…} (increasing “virtual days since learning”: frequent early review followed by increasingly spaced intervals) and map it to drift thresholds.

We define a “drift day” by measuring drift early in training (analogous to FOREVER’s warm-up window *S*):

- D_day = D_{t=S}

Then define drift thresholds:

- D_KL = { d · D_day | d ∈ D_human }

**Compute-matching (to avoid “wins from more replay”):** we scale D_KL by a single factor ρ per run so that the *total replay compute* (total replay optimizer steps) matches FOREVER within ±5% on the Phase-0 run. Concretely, we set ρ by binary search over a short prefix of training on task 1–2, targeting the same number of replay events (and the same `E_mem=2` epochs per event) as FOREVER. The main runs report realized replay events and replay steps; if KL-Time exceeds FOREVER by >10% replay steps after this calibration, we treat the comparison as invalid and refute.

Replay triggers when D_t ≥ (ρ · D_KL[j]) for the next threshold index j.

#### Phase-0 gate (signal actually changes decisions)
Before running full CL, we run a short Phase-0 diagnostic on the first 1–2 tasks:
- Compute replay trigger steps under (a) FOREVER’s τ-time and (b) KL-Time.
- Measure mean absolute difference in trigger time (as % of task training steps).

If trigger-time divergence is <10% on average, we treat the signal replacement as ineffective and refute early (the main experiment would be uninformative). In our setting each task is trained for 10 epochs, so 10% corresponds to roughly one epoch; smaller differences mean replay happens at nearly the same training stage.

### Key Innovations

1. **Function-space “model time” for replay triggering**: replace parameter-update norms with an output-distribution drift signal on past-task anchors.
2. **Compute-feasible KL for LLM CL**: use **label-space KL** (small label sets) to avoid expensive full-vocabulary KL while still measuring behavioral drift.
3. **Decisive Phase-0 gate**: explicitly test whether the new signal changes replay decisions before spending full compute.

---

## Related Work

### Field Overview

Continual learning methods are often grouped into (i) **regularization-based** methods that constrain updates (e.g., EWC), (ii) **replay-based** methods that revisit stored examples, and (iii) **parameter isolation / modularization** methods that reduce interference by separating parameters across tasks (e.g., prompt tuning, orthogonal subspaces, MoE variants).

For LLMs, replay remains a strong and practical baseline, but scheduling and selection choices matter substantially. Recent papers highlight that CL for LLMs can degrade general abilities and instruction following (e.g., TRACE), motivating more careful replay and update-control mechanisms.

### Related Papers

- **[FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt)**: Uses parameter-update-norm “model-centric time” to align Ebbinghaus-style replay with training dynamics.
- **[SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning](./references/SuRe-Surprise-Driven-Prioritised-Replay-for-Continual-LLM-Learning/meta/meta_info.txt)**: Improves replay by surprise-based sample selection (high NLL) and a fast/slow LoRA consolidation scheme.
- **[STABLE: Gated Continual Learning for Large Language Models](./references/STABLE-Gated-Continual-Learning-for-Large-Language-Models/meta/meta_info.txt)**: Uses EM/bits/KL budgets to gate sequential LoRA merges for continual model editing (gating, not replay scheduling).
- **[Anchored Supervised Fine-Tuning](./references/Anchored-Supervised-Fine-Tuning/meta/meta_info.txt)**: Uses forward-KL anchoring to stabilize supervised post-training (anchoring as regularization, not scheduling).
- **[Learn the Time to Learn: Replay Scheduling in Continual Learning](./references/Learn-the-Time-to-Learn-Replay-Scheduling-in-Continual-Learning/meta/meta_info.txt)**: Learns replay schedules via MCTS/RL using validation accuracies (scheduling learned from performance, not drift).
- **[Experience Replay for Continual Learning (CLEAR)](./references/Experience-Replay-for-Continual-Learning/meta/meta_info.txt)**: Replay + KL-based behavioral cloning in multi-task RL (KL as regularizer, not trigger signal).
- **[TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models/meta/meta_info.txt)**: Benchmark showing severe forgetting and alignment degradation in LLM continual learning.
- **[Orthogonal Subspace Learning for Language Model Continual Learning (O-LoRA)](https://arxiv.org/abs/2310.14152)**: Learns tasks in orthogonal low-rank subspaces to reduce interference without replay.
- **[MIGU: Unlocking Continual Learning Abilities in Language Models](https://arxiv.org/abs/2406.17245)**: Uses magnitude-based gradient masking to reduce forgetting (rehearsal-free or replay-enhancing).
- **[EWC: Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796)**: Classic Fisher-based weight-importance regularization.
- **[LwF: Learning without Forgetting](https://arxiv.org/abs/1606.09282)**: Functional regularization via distillation from old model outputs.
- **[GEM: Gradient Episodic Memory](https://arxiv.org/abs/1706.08840)**: Constrains gradients using episodic memory to prevent interference.
- **[DER++: Dark Experience Replay](https://arxiv.org/abs/2004.07211)**: Replay with distillation (“dark knowledge”) for stronger retention.
- **[MIR: Maximally Interfered Retrieval](https://arxiv.org/abs/1908.04742)**: Selects replay samples expected to be most interfered by new updates.
- **[Progressive Prompts](https://arxiv.org/abs/2301.12314)**: Prompt-based continual learning for language models (often requires task identity).
- **[MoELoRA](https://arxiv.org/abs/2402.12851)**: Mixture-of-experts LoRA for continual learning.
- **[SAPT](https://arxiv.org/abs/2401.08295)**: Shared-attention parameter-efficient tuning for continual learning.
- **[SSR: Self-Synthesized Rehearsal](https://arxiv.org/abs/2403.01244)**: Generates synthetic rehearsal data for replay.
- **[Recurrent-KIF](https://arxiv.org/abs/2502.17510)**: Continual learning via dynamic parameter importance and fusion (as referenced by FOREVER).
- **[AIM-Merging](https://arxiv.org/abs/2509.17348)**: Adaptive iterative model merging for continual learning (as referenced by FOREVER).

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Replay scheduling | Decide *when* to replay | FOREVER, Learn-the-Time-to-Learn | Standard CL / long-seq CL; OP/BWT | Scheduling signal may be indirect (τ) or require validation feedback |
| Replay selection | Decide *what* to replay | SuRe, MIR, DER++ | Various CL suites | Selection may not solve scheduling; can add overhead |
| Regularization / anchoring | Constrain drift during updates | EWC, LwF, Anchored SFT, STABLE | CL + post-training stability | Often affects “how to update”, not replay timing |
| Parameter isolation | Reduce interference via modularity | O-LoRA, Progressive Prompts, MoE | LLM CL benchmarks | May require extra parameters or task identity |

### Closest Prior Work

- **FOREVER**: Most direct prior work. It uses τ_t = Σ||Δθ||₂ (LoRA-only) to define “model time” and aligns an Ebbinghaus schedule to τ_day measured over S warm-up steps. KL-Time keeps the Ebbinghaus framing but replaces τ with a behavioral drift signal computed on past-task anchors.
- **VBM (Ebbinghaus step-based replay)**: Uses a human-inspired schedule but measures time in training steps; FOREVER improves this via model-centric time. KL-Time proposes a different model-centric time in function space.
- **STABLE / Anchored SFT**: Both use KL-type signals for stability (gating merges or anchoring loss), but do not use KL as a *replay trigger*.
- **Learn the Time to Learn**: Optimizes replay scheduling via performance-based RL/MCTS policies; KL-Time is a lightweight hand-crafted signal intended to work without a separate scheduler policy.

**Novelty Kill Search Summary:** Searched for “KL divergence replay trigger continual learning”, “function-space drift replay scheduling”, “anchor set KL replay trigger LLM continual learning”, and checked for work using KL/JS drift as the *replay trigger* (not merely as a regularizer) as of 2026-02-16. We found KL used for anchoring/gating and distillation, and replay scheduling learned from validation accuracies, but no close match to “KL-on-anchor-set as model time for replay triggering” in LLM continual learning. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| FOREVER | τ-based model time + Ebbinghaus schedule + intensity-aware replay | τ is parameter-space proxy; may not reflect behavioral drift | Replace τ with label-space KL drift on anchors | Replay triggered when past-task behavior actually drifts |
| VBM | Step-based Ebbinghaus replay | Step count misaligns with learning dynamics | Use drift-based time rather than steps | Less sensitive to optimizer/task dynamics |
| STABLE | KL/EM/bits gating for sequential LoRA merges | Not about replay timing | Use KL as schedule trigger | Addresses “when to replay” explicitly |
| Learn-the-Time-to-Learn | Learns replay schedule policy from validation accuracies | Requires training a scheduler; expensive | Use cheap drift metric | Low-overhead heuristic that can be drop-in |

---

## Experiments

### Experimental Setup

**Primary benchmark (decisive):** Standard CL benchmark used by FOREVER (5 text classification tasks): AG News, Amazon Reviews, Yelp Reviews, DBpedia, Yahoo Answers.

We follow FOREVER’s protocol where possible:
- Sample 1000 training instances per task.
- Reserve 500 evaluation instances per class.
- Store 2% of each task’s original training data as memory buffer.

**Training** (from FOREVER):
- Base model: Qwen/Qwen3-0.6B.
- LoRA: rank r=8, α=32, dropout 0.05; applied to the attention query/value projection matrices (`q_proj`, `v_proj`).
- Optimizer and batch: lr=3e-4, batch size 8.
- Train 10 epochs per new task; at each replay event, train memory for 2 epochs.

**Baselines (main comparison; 3 conditions total):**
1. **VBM (step-based Ebbinghaus replay)**: step-based replay schedule using D_human intervals mapped to fixed training steps.
2. **FOREVER (τ-based model time)**: parameter-update-norm time τ_t and τ_day calibration (S=24).
3. **KL-Time (ours)**: replace τ with label-space KL drift on anchor set (same D_human).

#### Baseline Ladder (REQUIRED)
- **No-training baseline (ICL)**: Evaluate base model on each task with zero-shot prompting (no continual training).
- **Sequential fine-tuning (no replay)**: Continual LoRA without any replay events.
- **Step-based replay schedule**: VBM (step-based Ebbinghaus).
- **Strongest prior method**: FOREVER.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Standard CL (5 tasks) | Sequential text classification tasks framed as instruction-following / label prediction | OP (overall performance), BWT (backward transfer), per-task accuracy | task-wise train/test | HuggingFace datasets | Custom (following FOREVER) |

Metrics (from FOREVER):
- OP = (1/K) Σ_i a_{i,K} (average final-task performance across tasks; higher is better)
- BWT = (1/(K−1)) Σ_{i=1..K−1} (a_{i,K} − a_{i,i}) (backward transfer; more negative means more forgetting)

### Resource Estimate

Evidence-based timing: FOREVER reports training time (min/epoch) for Qwen3-0.6B on 8×H20 GPUs (Table 6, `FOREVER .../B.4 Time Complexity Analysis`): MixReplay 1.3, VBM 1.4, FOREVER 1.4.

Rough compute estimate for Standard CL (5 tasks):
- 50 base epochs per run (10 epochs × 5 tasks) → ~70 minutes on 8 GPUs (plus replay overhead).
- With replay + KL drift evaluation overhead, estimate **~1.5–2.0 hours per run on 8×A100**.
- Main comparison: 3 conditions × 3 task orders × 3 seeds = 27 runs → **~324–432 GPU-hours** (plus Phase-0 calibration runs).

This fits the 768 GPU-hour budget with margin for debugging and Phase-0 diagnostics.

### Main Results

**Published reference numbers (FOREVER Table 1; Qwen3-0.6B, averaged over task orders):**

| Method | Benchmark | OP↑ | BWT↑ | Source | Notes |
|---|---|---:|---:|---|---|
| VBM | Standard CL | 71.5 | -5.2 | FOREVER Table 1 | Published, averaged over task orders |
| FOREVER | Standard CL | 72.9 | -4.7 | FOREVER Table 1 | Published, averaged over task orders |
| **KL-Time (ours)** | Standard CL | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| KL-Time (full) | KL-based trigger + same replay procedure | Best retention (higher OP / less negative BWT) |
| KL-Time w/ random anchors | Same size anchors but sampled from current task | If gains vanish, anchors-to-past is crucial |
| KL-Time w/ compute-matched cap (optional) | Cap replay events to match FOREVER if KL triggers more | Rules out “wins due to more replay” |

### Experimental Rigor

- **Variance**: Use 3 random seeds (e.g., `seeds=[42, 123, 456]`) and 3 task orders (to reduce order variance) for each condition; report mean ± std. (If compute allows, increase task orders to 8 to match FOREVER.)
- **Phase-0 gate**: Refute early if KL-time trigger steps are too similar to τ-time (mean trigger-time difference <10% of task steps on first 1–2 tasks).
- **Confounders**:
  - Replay compute mismatch: report replay event count and total replay steps; use compute-matched cap ablation if needed.
  - Anchor-set bias: anchor set is deterministic from memory buffers; report optional canary drift diagnostic (not used for triggering).
  - Prompt-template sensitivity for label scoring: fix a single template across all methods; run one sanity template swap to ensure effect is not template-specific.

---

## Success Criteria

**Hypothesis** (directional): KL-Time replay improves retention (higher OP and/or less negative BWT) compared to FOREVER and step-based Ebbinghaus replay, because it triggers replay when past-task behavior actually drifts.

**Decision Rule** (concrete):
- **Proceed**: KL-Time improves OP by a margin outside the std range vs FOREVER (or improves BWT by ≥0.5 absolute) when averaged over task orders, without increasing replay compute by >10%.
- **Pivot**: If KL-Time triggers are very different but performance does not improve, try an alternative drift signal (e.g., JS divergence, or drift on a held-out canary set instead of replay memory).
- **Refute**: If Phase-0 shows trigger-time difference <10% (signal replacement ineffective), or if KL-Time underperforms FOREVER under compute-matched replay.

---

## Impact Statement

If successful, this would give a low-overhead, behaviorally grounded replay trigger that practitioners can use to schedule replay in continual fine-tuning of LLMs, improving retention without needing to learn a separate scheduler policy or relying on parameter-space proxies.

---

## References

- [FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt)
- [SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning](./references/SuRe-Surprise-Driven-Prioritised-Replay-for-Continual-LLM-Learning/meta/meta_info.txt)
- [STABLE: Gated Continual Learning for Large Language Models](./references/STABLE-Gated-Continual-Learning-for-Large-Language-Models/meta/meta_info.txt)
- [Anchored Supervised Fine-Tuning](./references/Anchored-Supervised-Fine-Tuning/meta/meta_info.txt)
- [Learn the Time to Learn: Replay Scheduling in Continual Learning](./references/Learn-the-Time-to-Learn-Replay-Scheduling-in-Continual-Learning/meta/meta_info.txt)
- [Experience Replay for Continual Learning (CLEAR)](./references/Experience-Replay-for-Continual-Learning/meta/meta_info.txt)
- [TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models/meta/meta_info.txt)
- [Orthogonal Subspace Learning for Language Model Continual Learning (O-LoRA)](https://arxiv.org/abs/2310.14152)
- [Unlocking Continual Learning Abilities in Language Models (MIGU)](https://arxiv.org/abs/2406.17245)
- [EWC: Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796)
- [LwF: Learning without Forgetting](https://arxiv.org/abs/1606.09282)
- [GEM: Gradient Episodic Memory](https://arxiv.org/abs/1706.08840)
- [DER++: Dark Experience Replay](https://arxiv.org/abs/2004.07211)
- [MIR: Maximally Interfered Retrieval](https://arxiv.org/abs/1908.04742)
- [Progressive Prompts: Continual Learning for Language Models](https://arxiv.org/abs/2301.12314)
- [Super-NaturalInstructions](https://arxiv.org/abs/2204.07705)
