# untitled

# Adaptive PSFT: Feedback-Controlled Clip Range for Stable Supervised Fine-Tuning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Small language models (SLMs, here focusing on ≤8B parameters) are often deployed under strict latency and cost constraints. In practice, many teams still rely on **supervised fine-tuning (SFT)** as the default post-training step because it is simpler than reinforcement learning (RL): it requires only prompt–response pairs and standard maximum-likelihood training.

A recurring failure mode is that SFT on a narrow distribution can degrade capabilities that matter outside the fine-tuning domain (often called **capability retention** or an **out-of-domain performance cost** in safety/alignment contexts). This is particularly acute for small models, where limited capacity makes it easier for narrow fine-tuning to overwrite broadly useful representations.

Recent work argues that part of this problem is **optimization dynamics**: standard SFT can induce large, concentrated probability updates on some tokens, leading to entropy collapse and brittle generalization. **Proximal Supervised Fine-Tuning (PSFT)** proposes a simple fix: apply a Proximal Policy Optimization (**PPO**)-style **importance-ratio clipping** objective during SFT to limit how much the probability of a target token is allowed to increase relative to a lagged “old policy.” PSFT reports large out-of-domain gains on Qwen2.5-7B and Llama3.1-8B (e.g., big improvements on IFEval instruction-following while improving GPQA).

### The Problem

PSFT introduces a new trust-region hyperparameter: the clip range ε. In PPO-style RL, the effective trust-region width is known to be sensitive to training details (batch size, learning rate, update frequency of the old policy, and data mismatch), so many implementations monitor **clip fraction** and/or **approximate KL drift** during training.

In contrast, PSFT fixes ε to a constant value (0.28 in the main experiments). This creates two practical issues:

- **Hyperparameter sensitivity / retuning burden**: the best ε may differ across model sizes, domains, or stages of training.
- **Ambiguous comparability**: PSFT’s reported checkpoints differ in training steps across methods (e.g., Qwen SFT 700 steps vs PSFT 1300 steps in Sec. 4.1.2), so it is unclear how much of the observed benefit is due to the objective vs effective training compute and checkpoint selection.

Related “stability SFT” methods show similar patterns: KL-anchored SFT methods (e.g., ASFT) still require tuning the KL coefficient, token-selection methods (e.g., ProFit) use a fixed probability threshold and note that adaptive thresholds may help, and trust-region SFT variants like **TRAPO/TrSFT** introduce their own trust-region knob (α) to clip large SFT gradients.

### Key Insight and Hypothesis

We hypothesize that PSFT’s main benefit comes from regulating the *effective* update size, and that a fixed ε is a brittle proxy for this goal. Instead of treating ε as a tuned constant, we propose to **control ε with feedback** using simple training-time statistics.

**Mechanism hypothesis (why clip-fraction control should help):** PSFT’s analysis shows that the tokens most affected by clipping are “uncertainty / thinking-pattern” tokens (e.g., *wait*, *alternatively*) that are useful for long reasoning and later RL stages (PSFT Sec. 5.1, Fig. 8). During training, the importance-ratio distribution shifts (e.g., as the model becomes more confident on some domains/tokens), so a fixed ε can become **too tight** (over-clips and under-learns these patterns early) or **too loose** (allows occasional large ratio updates that collapse entropy late). Maintaining a target **upper clip fraction** keeps a roughly constant fraction of “risky” token updates at the trust-region boundary, which we expect to (i) prevent entropy collapse, and (ii) preserve OOD capabilities that correlate with staying close to the base model in KL (as suggested by RL’s Razor: forward-KL drift predicts forgetting).

**Hypothesis:** A feedback controller that adjusts ε to keep either (i) the PSFT upper clip fraction or (ii) the approximate KL drift per step near a target will improve the out-of-domain / capability retention metrics compared to fixed-ε PSFT at matched training compute, without sacrificing in-domain performance.

The outcome is uncertain because (i) “adaptive ε” may be equivalent to selecting a better fixed ε once tuned, (ii) the clip-fraction/KL signals computed on *offline* SFT data may not correlate well with true generalization, and (iii) the controller may oscillate and destabilize training.

---

## Proposed Approach

### Overview

We propose **Adaptive-PSFT**, a drop-in modification to PSFT where the clip range ε is updated online to maintain a target trust-region width measured by either:

1. **Target clip fraction**: the fraction of tokens whose importance ratio exceeds the clipping boundary.
2. **Target approximate KL**: PPO-style approximate KL drift computed from the log-prob ratios on the training batch.

The goal is to reduce manual tuning and make PSFT’s stability/retention behavior more consistent across training stages.

### Method Details

#### Background: PSFT objective

Let the training data be an offline SFT dataset of prompt–response pairs, and let the “action” at each step be the gold token. PSFT defines an importance ratio per token:

\[
 r_t = \frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\pi_{\theta_{\mathrm{old}}}(y_t \mid x, y_{<t})}.
\]

With a positive constant advantage (SFT treats all gold tokens as “good”), the PPO-style clipped objective implies that only the **upper** clipping boundary matters for gradients: tokens with \(r_t > 1+\varepsilon\) contribute zero gradient (PSFT Sec. 3.2).

#### Adaptive-PSFT controller (target clip-fraction; main)

PSFT already computes token-wise ratios \(r_t\) and applies clipping at the upper boundary \(1+\varepsilon\) (the boundary that matters for positive-advantage / SFT-style updates). Define the **upper clip fraction** on batch \(k\):

\[
 c_k = \mathbb{E}_{t\sim\text{batch}}[\mathbb{1}(r_t > 1+\varepsilon_k)].
\]

Instead of a PID-style controller with multiple extra hyperparameters, we propose a **quantile controller** that *sets* \(\varepsilon\) so that the desired fraction of tokens would be clipped on the current batch:

\[
\varepsilon_{k+1} = \mathrm{clip}\Big(\mathrm{Quantile}(r_t,\; 1-c_{\text{target}}) - 1,\; \varepsilon_{\min},\; \varepsilon_{\max}\Big).
\]

Intuition: this makes \(c_k\approx c_{\text{target}}\) by construction (up to quantile estimation noise), turning adaptivity into a *closed-form* mapping from observed ratio distribution → clip range.

Default hyperparameters: \(c_{\text{target}}=0.10\), \(\varepsilon_{\min}=0.05\), \(\varepsilon_{\max}=0.60\). We keep the lower bound `clip_ratio_low=0.20` fixed.

**Why feedback (vs a schedule) matters even in offline SFT:** the ratio distribution depends on non-stationary factors that a fixed schedule cannot reliably anticipate (e.g., model confidence shifts and batch composition/length). The quantile controller adapts to these shifts, whereas a hand-designed \(\varepsilon(t)\) schedule may maintain the wrong effective trust region.

#### Optional variant (ablation): target approximate KL

Compute a PPO-style approximate KL statistic on the batch:

\[
\widehat{\mathrm{KL}}_k = \mathbb{E}_{t\sim\text{batch}}\big[\log \pi_{\theta_{\mathrm{old}}}(y_t) - \log \pi_{\theta}(y_t)\big].
\]

Update \(\varepsilon\) multiplicatively (as in PPO) to keep \(\widehat{\mathrm{KL}}_k\) near a target \(\mathrm{KL}_{\text{target}}\).

#### Critical controls to separate “feedback” from “better ε”

To ensure gains are not merely due to choosing a better constant ε or a hand-designed schedule, we include two control baselines:

- **Tuned constant ε (practitioner control)**: run fixed-ε PSFT with \(\varepsilon\) selected by a **small 1-seed sweep** on a short prefix of training (e.g., first 2 epochs), using \(\varepsilon\in\{0.14, 0.28, 0.42\}\) as in PSFT’s parameter analysis. Then re-run fixed-ε PSFT with the selected \(\varepsilon^*\) for all 3 seeds.

- **Tuned ε schedule (strong strawman vs feedback)**: choose a simple 2-parameter schedule (linear interpolation) \(\varepsilon(t) = (1-\tfrac{t}{T})\,\varepsilon_{\text{hi}} + (\tfrac{t}{T})\,\varepsilon_{\text{lo}}\), with \((\varepsilon_{\text{hi}}, \varepsilon_{\text{lo}})\) selected by the same short 1-seed prefix sweep (grid over \{0.14,0.28,0.42\}^2). This baseline tests whether *non-adaptive* scheduling can match feedback.

These controls test whether feedback itself provides value beyond what a simple constant-ε or schedule-tuning procedure would yield.

### Key Innovations

1. **Feedback-controlled trust region for supervised fine-tuning**: maintain a target clip-fraction in PSFT using a *closed-form quantile controller*, rather than hand-tuning ε.
2. **Compute-matched evaluation of PSFT-style SFT**: enforce step-matched training and standardized checkpoint selection to avoid “wins from more training steps.”
3. **Controls that distinguish feedback from tuning**: include both a tuned-constant ε baseline and a tuned ε-schedule baseline (selected by the same short prefix sweep).

---

## Related Work

### Field Overview

Work on stabilizing and improving SFT can be grouped into: (i) **trust-region / KL-constrained** objectives that limit distributional drift, (ii) **token- or sample-level reweighting/masking** that suppresses harmful gradients (often from low-probability tokens), and (iii) **bridges between SFT and RL** that reinterpret SFT as a policy-optimization problem.

PSFT sits in the trust-region family by applying PPO-style clipping to SFT. ASFT adds an explicit KL anchor to a reward-weighted regression view of dynamic fine-tuning. ProFit masks low-probability tokens to prevent “surface-form overfitting.” Several recent papers argue that understanding fine-tuning through an RL lens helps diagnose forgetting and generalization failures.

### Related Papers

(≥20 papers; each is one sentence summarizing relevance.)

- **[Proximal Supervised Fine-Tuning](./references/Proximal-Supervised-Fine-Tuning/meta/meta_info.txt)**: introduces PPO-style clipping for SFT and reports large OOD retention gains for 7B/8B models.
- **[Anchored Supervised Fine-Tuning](./references/Anchored-Supervised-Fine-Tuning/meta/meta_info.txt)**: adds KL anchoring to dynamic fine-tuning to control drift while approximating RL objectives.
- **[Learning While Staying Curious (CurioSFT)](./references/Learning-While-Staying-Curious-Entropy-Preserving-Supervised-Fine-Tuning-via-Adaptive-Self-Distillation-for-Large-Reasoning-Models/meta/meta_info.txt)**: preserves entropy in SFT via adaptive self-distillation and includes PSFT as a baseline.
- **[ProFit](./references/ProFit-Leveraging-High-Value-Signals-in-SFT-via-Probability-Guided-Token-Selection/meta/meta_info.txt)**: masks low-probability tokens in SFT using a fixed threshold τ and notes adaptive thresholds as future work.
- **[Clipping-Free Policy Optimization (CFPO)](./references/Clipping-Free-Policy-Optimization-for-Large-Language-Models/meta/meta_info.txt)**: argues clipping causes optimization pathologies in RL and replaces it with a smooth quadratic penalty.
- **[It’s Not You, It’s Clipping](./references/Its-Not-You-Its-Clipping/meta/meta_info.txt)**: proposes probability smoothing as a soft trust-region alternative to hard clipping in RL.
- **[One-Token Rollout](./references/One-Token-Rollout/meta/meta_info.txt)**: turns SFT into an on-policy, token-level policy-gradient procedure via one-token rollouts.
- **[On-Policy RL Meets Off-Policy Experts (CHORD)](./references/CHORD-Dynamic-Weighting/meta/meta_info.txt)**: harmonizes SFT and on-policy RL using a global schedule and token-wise weighting φ=p(1-p).
- **[On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification (DFT)](./references/Generalization-of-SFT-Reward-Rectification/meta/meta_info.txt)** (arXiv:2508.05629): shows SFT gradients overweight low-probability tokens (high variance) and proposes token-wise reweighting; complements PSFT/OPC-SFT’s focus on low-prob “off-policy tokens”.
- **[TRAPO / Trust-Region Adaptive Policy Optimization (TrSFT)](./references/TRAPO-Trust-Region-Adaptive-Policy-Optimization/meta/meta_info.txt)** (arXiv:2512.17636): proposes a trust-region SFT loss (TrSFT) that clips large SFT gradient weights using a per-token probability threshold α, motivated by avoiding distribution-blending failures when interleaving SFT and RL.
- **DAPO** ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)): an open-source RL system that uses asymmetric clipping (“clip-higher”), referenced by PSFT for RL-stage settings.
- **DeepSeekMath / GRPO** ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)): popularizes GRPO-style RLVR for math reasoning, often using KL controls for stability.
- **Dr. GRPO** ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)): studies and corrects bias/instability sources in GRPO-style optimization.
- **MC-GRPO** ([arXiv:2601.22582](https://arxiv.org/abs/2601.22582)): proposes robust group baselines for GRPO by median-centering.
- **IPS-GRPO** ([arXiv:2601.21669](https://arxiv.org/abs/2601.21669)): modifies GRPO with inverse-probability scaling.
- **[BAPO](./references/BAPO-Adaptive-Clipping/meta/meta_info.txt)** (arXiv:2510.18927): dynamically adapts clipping bounds in off-policy PPO to maintain balanced gradient contributions and preserve entropy; relevant as “adaptive clipping exists in RL” but not yet in pure SFT.
- **PPO** ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)): introduces the clipped surrogate objective and the notion of monitoring KL/clip fraction.
- **TRPO** ([arXiv:1502.05477](https://arxiv.org/abs/1502.05477)): foundational trust-region policy optimization with explicit KL constraints.
- **Mitigating the Alignment Tax of RLHF** ([arXiv:2309.06256](https://arxiv.org/abs/2309.06256)): shows simple model interpolation is a strong capability-retention baseline, motivating careful retention evaluation.
- **Safety Alignment as Continual Learning (OGPSA)** ([arXiv:2602.07892](https://arxiv.org/abs/2602.07892)): mitigates capability loss by projecting updates away from a capability-gradient subspace.
- **[RL's Razor: Why Online Reinforcement Learning Forgets Less](./references/RLs-Razor/meta/meta_info.txt)** (arXiv:2509.04259): shows forgetting correlates strongly with forward KL from the base model; motivates KL/clip-fraction control signals during SFT.
- **[Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting](./references/Retaining-by-Doing/meta/meta_info.txt)** (arXiv:2510.18874): identifies on-policy data as the main mechanism for low-forgetting post-training, consistent with “stay close in KL” control.
- **s1: Simple test-time scaling** ([arXiv:2501.19393](https://arxiv.org/abs/2501.19393)): provides strong inference-time baselines for reasoning via sampling.
- **LIMO: Less is More for Reasoning** ([arXiv:2502.03387](https://arxiv.org/abs/2502.03387)): shows small curated reasoning sets can improve reasoning without large-scale training.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Trust-region / constrained SFT | Limit policy drift during SFT via clipping or KL | PSFT, ASFT, TRAPO/TrSFT | AIME, GPQA, IFEval, MMLU-Pro | Introduces new knobs (ε, λ, α); may require retuning |
| Token masking / reweighting | Suppress gradients from low-value or off-policy tokens | ProFit, DFT/iw-SFT variants | GPQA, GSM8K, MATH-500, IFEval | Static thresholds may be brittle; may hurt creative tasks |
| SFT as RL / SFT–RL unification | Reinterpret SFT as policy optimization and mix with RL | One-Token Rollout, CHORD | Reasoning + tool use benchmarks | More complex training stack; sampling cost |
| Clipping alternatives (RL) | Replace or soften clipping to avoid pathologies | CFPO, probability smoothing | RLVR/RLHF benchmarks | Unclear transfer to pure SFT |

### Closest Prior Work

- **PSFT**: Uses fixed ε and a lagged old policy; reports strong OOD gains but does not provide a procedure to choose ε or ensure step-matched comparisons.
- **ASFT**: Uses explicit KL anchoring to a fixed base model; it controls drift but still requires tuning of the anchoring strength and uses a different objective than ratio clipping.
- **TRAPO / TrSFT (arXiv:2512.17636)**: Introduces a *trust-region SFT* loss that clips large SFT gradient weights when the model assigns very low probability to an expert token (threshold α). This is conceptually close (adding a trust region to SFT), but it is **not** ratio clipping: it does not use an old policy, and it addresses a different failure mode (distribution blending when interleaving SFT+RL).
- **ProFit**: Uses a fixed probability threshold τ to mask low-probability tokens; it suggests adaptive thresholds but does not control policy drift directly.
- **iw-SFT / SFT-as-RL line**: Uses importance weighting and sometimes clipping to tighten RL bounds; focuses on theoretical connection rather than practical controller design.
- **OPC-SFT (OpenReview qJLKOryYeR)**: A near-parallel concurrent work that applies PPO-style ratio clipping to SFT to suppress large gradients from low-probability “off-policy tokens” and studies old-policy refresh frequency; notably, it does **not** propose feedback-controlled ε (it reports a robust fixed range ϵ∈[0.4,0.6] and tracks clipped-token counts). We treat it as the closest prior and position our contribution specifically as **adaptive ε control** on top of ratio clipping.

**Novelty Kill Search Summary:** Searched for “adaptive clip range supervised fine-tuning”, “adaptive epsilon PSFT”, “clip fraction controller SFT”, and scanned local KB + all agents’ draft proposals for PSFT/2508.17784 and adaptive-ε SFT. Found adaptive clipping mainly in RL (e.g., **BAPO**, arXiv:2510.18927) and fixed-threshold SFT methods (PSFT/ASFT/ProFit/OPC-SFT). Also found **TRAPO / TrSFT** (arXiv:2512.17636, Dec 2025), which introduces *trust-region SFT* by clipping large SFT gradient weights when per-token probability is below a threshold α. TRAPO’s TrSFT is conceptually close (trust-region in SFT) but differs mechanistically: it clips **probability-gradient weights** (effectively a low-probability-token dampening), whereas PSFT clips **importance ratios to an old policy**, and our contribution is to make this **ratio clip range adaptive via clip-fraction/KL targets**. No prior work found that uses a PPO-style target-clip-fraction controller to adapt ε in PSFT-style ratio-clipped supervised fine-tuning as of 2026-02-19.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| PSFT | Fixed ε ratio clipping for SFT | ε must be tuned; unclear robustness across tasks/stages | Make ε adaptive via KL/clip-fraction targets | Directly regulates effective trust region across training |
| ASFT | Adds KL anchoring to DFT-style objective | Needs λ tuning; different mechanism | Use controller on PSFT clip range instead of λ | Keeps PSFT simplicity while reducing tuning |
| ProFit | Masks low-prob tokens with fixed τ | Static threshold; not drift-focused | Adapt trust-region width rather than mask tokens | Better targets distributional drift, not only token rarity |
| CFPO | Removes clipping in RL with quadratic penalty | RL setting, not SFT | Keep PSFT objective but control ε | Minimal change; compatible with existing SFT stacks |
| OPC-SFT | Token-clipped SFT for cold-start stability | Unknown if adaptive control exists | If OPC-SFT is fixed-ε, add controller + controls | Tests whether adaptivity is necessary beyond clipping |

---

## Experiments

### Experimental Setup

**Primary setting (decisive):** Math-reasoning SFT on **Qwen2.5-7B-Instruct** following PSFT’s setup.

**Notation clarifications (readability):**
- **avg@k**: average score across *k* independent sampled solutions per problem (same k for all compared methods).
- **AIME24**: AIME 2024 math competition problems used as a reasoning benchmark.
- **GPQA**: Graduate-Level Google-Proof Question Answering benchmark (hard science questions).
- **VERL**: a distributed RL training framework that PSFT builds on; we use it in “offline / skip_rollout” mode for SFT-style training.

We run three training conditions with **matched optimizer steps and matched evaluation protocol**:
1. **PSFT-fixed (tuned constant)**: ε=ε* selected by a small 1-seed prefix sweep on a short prefix of training (see controls below). *(Note: ε*=0.28 recovers the paper default when it is best.)*
2. **PSFT ε-schedule (tuned schedule)**: linear ε(t) schedule with endpoints (ε_hi, ε_lo) selected by the same prefix sweep.
3. **Adaptive-PSFT (ours)**: quantile controller; ε updated online to maintain target clip-fraction.

We treat vanilla SFT as an optional **sanity baseline** (not part of the main 3-condition decisive comparison) to confirm the training stack reproduces PSFT’s qualitative advantage over plain cross-entropy.

**Checkpoint selection (to avoid step-count confound):** For all methods, select the checkpoint with the best **in-domain validation** AIME24 score at a fixed evaluation budget, evaluated at the same intervals.

**Baseline Ladder (REQUIRED):**
- **No-training baseline**: original Qwen2.5-7B-Instruct evaluated with the same prompts.
- **Inference-time scaling baseline**: best-of-32 sampling (avg@32) for all evaluation tasks (already part of PSFT’s reported protocol for AIME/AMC).
- **Training baselines**: PSFT-fixed (ε=ε* tuned), PSFT ε-schedule (tuned schedule), and Adaptive-PSFT. (Optional sanity: plain cross-entropy SFT, to confirm the stack reproduces PSFT’s qualitative retention gains.)

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Open weights; used in PSFT |
| (Optional replication) Llama3.1-8B-Instruct | 8B | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | Secondary model to test generality |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| wh-zhu/train_openr1_4k | SFT/PSFT/Adaptive-PSFT training (long-CoT math demonstrations) | 25,395 rows (~163MB parquet) | https://huggingface.co/datasets/wh-zhu/train_openr1_4k | Not specified on HF card (verify before use) |
| OpenR1-Math-81922 (PSFT paper) | Optional scale-up replication (filtered OpenR1 math; responses ≤8192) | 81,922 (paper) | https://huggingface.co/blog/open-r1 | Depends on underlying OpenR1 license; verify |

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| AIME 2024 | competitive math problems used for reasoning evaluation | avg@32 | test | https://huggingface.co/datasets/wh-zhu/aime-24 (960 rows) | PSFT repo evaluation (`evaluation/`) |
| GPQA | graduate-level QA benchmark | avg@8 | test | https://arxiv.org/abs/2311.12022 | standard eval harness |
| IFEval | instruction-following evaluation with programmatic checks | accuracy (loose/strict) | test | https://arxiv.org/abs/2311.07911 | https://github.com/google-research/google-research/tree/master/instruction_following_eval |

### Main Results

We will report in-domain and out-of-domain metrics at matched inference budgets.

**Published reference numbers (PSFT Sec. 4.1.2; 1 run, checkpoint chosen by in-domain performance):**

We focus on three metrics that (i) were reported by PSFT, (ii) are widely used, and (iii) capture the in-domain vs retention trade-off:
- **AIME24 avg@32** (math reasoning; higher is better)
- **GPQA avg@8** (OOD scientific QA; higher is better)
- **IFEval_loose** (instruction following with programmatic checks; higher is better)

| Method | Base Model | AIME24 (avg@32) | GPQA (avg@8) | IFEval_loose | Source | Notes |
|---|---|---:|---:|---:|---|---|
| Original | Qwen2.5-7B-Instruct | 11.25 | 31.38 | 73.94 | PSFT Sec. 4.1.2 (`./references/Proximal-Supervised-Fine-Tuning/sections/4.1.2 Detailed Evaluations.md`, Tables 1–2) | No additional training |
| SFT | Qwen2.5-7B-Instruct | 22.08 | 32.89 | 54.42 | PSFT Sec. 4.1.2 (`./references/Proximal-Supervised-Fine-Tuning/sections/4.1.2 Detailed Evaluations.md`, Tables 1–2) | Paper uses 700-step checkpoint in example |
| PSFT (fixed ε=0.28) | Qwen2.5-7B-Instruct | 19.38 | 33.21 | 73.03 | PSFT Sec. 4.1.2 (`./references/Proximal-Supervised-Fine-Tuning/sections/4.1.2 Detailed Evaluations.md`, Tables 1–2) | Paper uses 1300-step checkpoint in example |
| **Adaptive-PSFT (ours)** | Qwen2.5-7B-Instruct | **TBD** | **TBD** | **TBD** | - | To be verified (compute-matched protocol) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| **Prefix sweep (to choose ε\* and schedule endpoints)** | 1 seed, short prefix of training (e.g., 2 epochs ≈200 steps). Evaluate fixed ε∈{0.14,0.28,0.42} and linear schedules with (ε_hi, ε_lo)∈{0.14,0.28,0.42}^2; pick ε\* and (ε_hi, ε_lo) that maximize early AIME24 avg@32 | If the tuned constant or tuned schedule closes the gap to Adaptive-PSFT, suggests feedback is unnecessary |
| Target approximate KL (optional) | Swap the control signal from clip-fraction to approx-KL | Helps test whether the signal choice matters |

### Experimental Rigor

- **Variance**: run all main conditions with **3 seeds** (`seeds=[42,123,456]`) and report mean±std.
- **Compute matching**: enforce identical optimizer steps, max sequence length, and evaluation intervals across methods.
- **Confounders**:
  1. **Step-count/checkpoint confound**: fixed step budget + common selection criterion.
  2. **“Better ε” vs “adaptive ε”**: tuned-constant ε baseline selected by a small early sweep.
  3. **Inference-budget confound**: same sampling budget for all methods (avg@k).
- **Sanity check**: verify we can reproduce the direction of PSFT’s retention gain vs SFT on a small subset before running full training; if PSFT-fixed does not improve **IFEval_loose** over SFT by ≥5 points (at the same inference budget), stop early and treat the setup as non-reproducible.

### Resource Estimate

**Grounded training configuration (from PSFT repo scripts):**
- `run_psft.sh` (raw file: https://raw.githubusercontent.com/zwhong714/PSFT/main/verl/recipe/psft/run_psft.sh ; scraped as `./references/PSFT-run_psft-sh/sections/Main Content.md`):
  - `max_prompt_length=2048`, `max_response_length=6144` (total 8192)
  - `train_batch_size=256`, `ppo_mini_batch_size=32`, `total_epochs=10`
  - `clip_ratio_low=0.2`, `clip_ratio_high=0.28`
  - `actor.optim.lr=1e-6`, `grad_clip=1.0`, `n_gpus_per_node=8`
- `run_sft.sh` (https://raw.githubusercontent.com/zwhong714/PSFT/main/verl/recipe/psft/run_sft.sh ; scraped as `./references/PSFT-run_sft-sh/sections/Main Content.md`): same lengths/batch/epochs, but `actor.optim.lr=2e-5`, `ppo_mini_batch_size=256`.

**Budget plan (≤768 A100 GPU-hours hard cap):**
- **Phase 0 (throughput + controller stability)**: 1 seed; run **~100 steps** each for PSFT-fixed and Adaptive-PSFT on `wh-zhu/train_openr1_4k`.
  - Use this to measure **seconds/step** and set `max_steps` for the main runs so the total GPU-hours stays ≤768.
  - Budget: ≤8 A100 × 3 hours = **24 GPU-hours**.
- **Main verification (compute-matched)**: Qwen2.5-7B-Instruct, **3 seeds**, **3 methods** (PSFT ε=ε\*, PSFT tuned ε-schedule, Adaptive-PSFT), trained for a fixed `max_steps` (≈1000 steps corresponds to 10 epochs on 25,395-row `train_openr1_4k` with batch size 256).
  - Target per (seed, method) run: **≤8 A100 × 8 hours = 64 GPU-hours**.
  - Total for 9 runs: **≤576 GPU-hours**.
- **Sweep cost (to choose ε\* and schedule endpoints)**: 1 seed, prefix runs for fixed ε∈{0.14,0.28,0.42} and schedules over {0.14,0.28,0.42}^2, ~200 steps each.
  - Budget: ≤8 A100 × 2 hours = **16 GPU-hours**.

**Total planned training budget (≤768 A100 GPU-hours hard cap):**
- Phase 0: ≤24 GPU-hours
- Joint sweep for ε\* and schedule endpoints (same prefix runs): ≤16 GPU-hours
- Main runs (3 seeds × 3 methods): ≤576 GPU-hours

**Enforcement:** In Phase 0 we measure seconds/step and then set `max_steps` so that each (seed, method) run is ≤((768-24-16)/9) ≈ **80.9 GPU-hours**, leaving buffer for evaluation decoding and potential reruns if one seed fails.

**Important uncertainty:** The PSFT paper/repo do not report end-to-end wall-clock time or token throughput in a reusable way. Phase 0 is required to translate steps→GPU-hours reliably.

---

## Success Criteria

**Hypothesis (directional):** Adaptive-PSFT improves OOD/capability retention compared to PSFT-fixed at matched compute, while maintaining similar in-domain performance.

**Decision Rule (concrete):**
- **Proceed** if, on Qwen2.5-7B averaged across 3 seeds:
  - Adaptive-PSFT improves **IFEval_loose by ≥3.0 points** over **both** the tuned constant baseline PSFT(ε\*) **and** the tuned ε-schedule baseline, **or** improves **GPQA by ≥1.0 point** over both,
  - and does not reduce **AIME24 avg@32 by more than 1.0 point** vs the best non-adaptive baseline (max of tuned constant / tuned schedule).
- **Refute** if Adaptive-PSFT is worse than the best non-adaptive baseline on both **GPQA** and **IFEval_loose** (by >0.5 points) under the compute-matched protocol, **or** if Adaptive-PSFT matches the best non-adaptive baseline within its std on all reported metrics (no measurable benefit from feedback).
- **Pivot** if the controller is unstable (ε oscillates wildly) or requires extra smoothing/hysteresis beyond the quantile update; in that case, treat the outcome as “a simple tuned schedule is sufficient” and prefer the ε-schedule baseline.

---

## Impact Statement

If successful, Adaptive-PSFT would give practitioners a simple, automated way to set PSFT’s trust-region strength without per-task hyperparameter sweeps. This could make trust-region SFT a more reliable default for small-model post-training where capability retention and out-of-domain robustness are critical.

---

## References

- [Proximal Supervised Fine-Tuning](./references/Proximal-Supervised-Fine-Tuning/meta/meta_info.txt) - Zhu et al., 2025
- [ProFit](./references/ProFit-Leveraging-High-Value-Signals-in-SFT-via-Probability-Guided-Token-Selection/meta/meta_info.txt) - Liu et al., 2026
- [Clipping-Free Policy Optimization for Large Language Models](./references/Clipping-Free-Policy-Optimization-for-Large-Language-Models/meta/meta_info.txt) - Caugatan et al., 2026
- [Anchored Supervised Fine-Tuning](./references/Anchored-Supervised-Fine-Tuning/meta/meta_info.txt) - Zhu et al., 2025
- [Learning While Staying Curious (CurioSFT)](./references/Learning-While-Staying-Curious-Entropy-Preserving-Supervised-Fine-Tuning-via-Adaptive-Self-Distillation-for-Large-Reasoning-Models/meta/meta_info.txt) - 2026
- [It’s Not You, It’s Clipping](./references/Its-Not-You-Its-Clipping/meta/meta_info.txt) - 2025
- [One-Token Rollout](./references/One-Token-Rollout/meta/meta_info.txt) - 2026
- [OPC-SFT](./references/OFF-POLICY-TOKEN-CLIPPED-SUPERVISED-FINE-TUNING-YIELDS-A-ROBUST-COLD-START/meta/meta_info.txt) - OpenReview qJLKOryYeR, 2026
- [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](./references/Generalization-of-SFT-Reward-Rectification/meta/meta_info.txt) - 2025
- [BAPO](./references/BAPO-Adaptive-Clipping/meta/meta_info.txt) - 2025
- [RL's Razor: Why Online Reinforcement Learning Forgets Less](./references/RLs-Razor/meta/meta_info.txt) - 2025
- [Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting](./references/Retaining-by-Doing/meta/meta_info.txt) - 2025
- [On-Policy RL Meets Off-Policy Experts (CHORD)](./references/CHORD-Dynamic-Weighting/meta/meta_info.txt) - 2025
- [TRAPO / Trust-Region Adaptive Policy Optimization](./references/TRAPO-Trust-Region-Adaptive-Policy-Optimization/meta/meta_info.txt) - Su et al., 2025
- [PPO](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [TRPO](https://arxiv.org/abs/1502.05477) - Schulman et al., 2015
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [Instruction-Following Evaluation for Large Language Models (IFEval)](https://arxiv.org/abs/2311.07911) - Zhou et al., 2023
- [GPQA](https://arxiv.org/abs/2311.12022) - Rein et al., 2024
