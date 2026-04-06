# untitled

# Risk-Controlled Early Exit for Diffusion Language Models (Calibrating Jot with Conformal Risk Control)

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Diffusion language models (dLLMs) generate text by iteratively denoising a partially masked sequence, rather than producing tokens left-to-right as in autoregressive models. This iterative process enables parallel token updates and bidirectional context, but it can be slow because generation may require hundreds of denoising steps.

A large recent line of work speeds up dLLM inference by **early exiting**: stopping the denoising process (globally or per token) once predictions appear stable. Examples include Prophet (global early commit), SchED (progress-aware confidence schedules), KLASS (KL-based stability sampling), and Jot (token-level early stopping). These methods can produce large speedups, but practitioners still face a deployment obstacle: early-exit methods typically expose thresholds or schedules that must be tuned per model and per task, and they provide no distribution-free guarantee on how much quality will be lost.

### The Problem

For many applications, the deployment question is not “what threshold gives the best average speed-quality trade-off?”, but rather:

> **How do we choose an early-exit threshold so that we provably do not convert more than an ε fraction of the full decoder’s correct predictions into incorrect ones, while still obtaining meaningful speedup?**

Today’s dLLM early-exit papers often tune thresholds with small validation sweeps (e.g., Jot uses different τ_max for Dream vs LLaDA; SchED evaluates multiple (τ_high, τ_low) settings), which does not produce a user-interpretable guarantee.

Meanwhile, the conformal prediction / risk-control literature provides post-hoc calibration rules that yield **distribution-free guarantees** for monotone risks, and has been applied to early-exit neural networks in general (e.g., Fast yet Safe) and to stopping policies for autoregressive reasoning (e.g., Conformal Thinking). However, we have not seen these risk-control procedures applied to **diffusion-LM early-exit thresholds**, where the denoising dynamics can be nontrivial (and potentially non-monotone due to “overthinking”).

### Key Insight and Hypothesis

We hypothesize that for token-level early stopping in dLLMs, there exists a practical monotone “conservativeness” knob (e.g., Jot’s τ_max) such that the task-level risk of converting a full-decoder success into an error is approximately non-increasing as the early-exit policy becomes more conservative. If so, we can apply conformal risk control (CRC/UCB) to select τ automatically from a calibration set, yielding a user-facing guarantee: **with probability ≥ 1−δ over the calibration set draw, the early-exit policy will convert at most an ε fraction of the full decoder’s correct predictions into incorrect ones**.

This could fail for two reasons: (i) the risk may be non-monotone in τ due to non-monotone diffusion trajectories, invalidating the risk-control assumptions; or (ii) the UCB correction may be so conservative that it selects τ close to full decoding, eliminating speedup.

---

## Proposed Approach

### Overview

We propose **RC-Jot**: a risk-controlled calibration wrapper around Jot (Just on Time), a training-free token-level early stopping method for diffusion language models. RC-Jot does not change the dLLM or its decoding algorithm; it only chooses Jot’s main threshold parameter via conformal risk control.

### Method Details

**Base early-exit policy (Jot).** Jot finalizes tokens individually during the diffusion process. At each denoising step t and position i, it computes a confidence ratio

- r_i(t) = p_1 / (p_2 + ε_small), where p_1 and p_2 are the top-1 and top-2 probabilities for that position.

Jot uses a spatially modulated threshold τ_i(t) that depends on proximity to already-unmasked tokens. The overall aggressiveness is mainly controlled by τ_max (larger τ_max = more conservative; fewer tokens exit early).

**Risk we control (“do not convert full-decoder successes into errors”).** For each evaluation instance i with ground-truth label, define:

- c_full(i) ∈ {0,1}: correctness of the **full** diffusion decoding (no early exit).
- c_exit(i, τ) ∈ {0,1}: correctness of Jot decoding with parameter τ_max=τ.

We define the per-instance loss

- r_i(τ) = 1[ c_full(i)=1 ∧ c_exit(i,τ)=0 ].

This measures the event “the full decoder was correct, but early exit becomes incorrect.” The expected risk R(τ)=E[r_i(τ)] upper-bounds the accuracy drop relative to full decoding by acc_full × R(τ).

**Calibration objective.** Given user-specified (ε, δ), we select the **least conservative** τ (smallest τ) that satisfies the risk constraint.

**Naive (empirical) calibration baseline.** Choose

- τ_emp = min{ τ ∈ Τ : R_hat(τ; D_cal) ≤ ε },

where R_hat is the empirical mean of r_i on the calibration set D_cal and Τ is a discrete grid of candidate τ values.

**UCB risk-controlled calibration (RC-Jot).** Following Fast yet Safe (which adapts conformal risk control to early-exit networks), compute an upper confidence bound R_hat^+(τ; D_cal) such that

- P_{D_cal}( R(τ) ≤ R_hat^+(τ; D_cal) ) ≥ 1−δ  for all τ ∈ Τ.

Then select

- τ_UCB = min{ τ ∈ Τ : R_hat^+(τ′;D_cal) < ε  for all τ′ ≥ τ }.

We will use the Waudby-Smith–Ramdas betting bound (as in the RC-EENN implementation) to compute R_hat^+.

**Go/no-go monotonicity diagnostic.** Risk control requires that R(τ) is non-increasing in τ. We will empirically measure R_hat(τ) over the τ grid and treat strong non-monotonicity as a refutation of the approach.

### Key Innovations

- **A task-level, user-interpretable risk target for diffusion-LM early exit**: “do not convert more than ε of the full decoder’s correct predictions into incorrect ones,” rather than tuning a confidence threshold without semantics.
- **Applying conformal risk control to dLLM token-level early stopping** (RC-Jot), yielding a distribution-free guarantee under a monotonicity assumption that is explicitly tested.
- **A decisive go/no-go test for applying guarantees to diffusion decoding**: measuring whether the risk is monotone in τ in practice.

---

## Related Work

### Field Overview

Diffusion language models (e.g., D3PM, SEDD, MDLM, LLaDA, Dream) produce text via iterative denoising. Recent work improves dLLM practicality by reducing (i) the number of denoising steps (early exit, adaptive schedules, speculative/path search) and/or (ii) the cost per step (caching and architectural changes). Early exit methods often rely on confidence or stability heuristics and require threshold tuning; more recent work introduces adaptive inference policies with multiple hyperparameters.

Separately, conformal prediction and conformal risk control provide finite-sample, distribution-free guarantees for prediction sets and for controlling expected risk of monotone losses. These tools have been used to calibrate confidence-based decisions, including early exit in neural networks and stopping policies in autoregressive reasoning / sampling.

### Related Papers

- **[Jot](./references/Just-on-Time-Token-Level-Early-Stopping-for-Diffusion-Language-Models/meta/meta_info.txt)**: Token-level early stopping for dLLMs using a top-2 probability ratio and spatially modulated thresholds.
- **[Prophet](./references/Diffusion-Language-Models-Know-the-Answer-Before-Decoding/meta/meta_info.txt)**: Training-free early commit for dLLMs using a confidence-gap stopping rule with staged thresholds.
- **[SchED](./references/Fast-Decoding-Diffusion-Language-Models-via-Progress-Aware-Confidence-Schedules/meta/meta_info.txt)**: Training-free early exit using a smooth progress-aware confidence schedule and a quality-penalized speed metric (QPS).
- **[KLASS: KL-Guided Fast Inference in Masked Diffusion Models](https://arxiv.org/abs/2511.05664)**: Uses token-level KL divergence between consecutive steps and confidence signals to parallelize unmasking for faster masked diffusion decoding.
- **[Learning to Parallel (Learn2PD)](https://arxiv.org/abs/2509.25188)**: Trains a lightweight post-hoc module to decide whether tokens are final, enabling adaptive parallel decoding for diffusion LMs.
- **[CadLLM](./references/Improving-the-Throughput-of-Diffusion-based-Large-Language-Models-via-a-Training-Free-Confidence-Aware-Calibration/meta/meta_info.txt)**: Training-free confidence-driven adaptation of block size/steps/vocabulary/threshold for throughput.
- **[CCD / Beyond Confidence](https://arxiv.org/abs/2512.02044)**: Uses trajectory-level contextual consistency and adaptive sampling budgets.
- **[LRD](https://arxiv.org/abs/2510.11052)**: Uses soft embeddings (“belief states”) + KL monitoring for phase transition and early stopping.
- **[EDIT: Early Diffusion Inference Termination for dLLMs Based on Dynamics of Training Gradients](https://arxiv.org/abs/2512.00670)**: Uses training-dynamics signals (from saved checkpoints) and matched-support KL to decide when diffusion inference can terminate early.
- **[TRACE](https://openreview.net/forum?id=Lccm6fjjyM)**: Provides theory for safe early exits in dLLMs based on training dynamics and stability certificates.
- **[Fast yet Safe](./references/Fast-yet-Safe-Early-Exiting-with-Risk-Control/meta/meta_info.txt)**: Adapts conformal risk control to early-exit neural networks; provides CRC/UCB calibration rules.
- **[Conformal Risk Control](./references/Conformal-Risk-Control/meta/meta_info.txt)**: General risk-control framework for monotone risks.
- **[Distribution-free risk-controlling prediction sets](https://arxiv.org/abs/2107.07511)**: UCB-style risk control foundations.
- **[Waudby-Smith & Ramdas (betting bounds)](https://arxiv.org/abs/2010.09686)**: Time-uniform and empirical bounds used for practical UCB confidence intervals.
- **[Anytime-Valid Conformal Risk Control](https://arxiv.org/abs/2602.04364)**: Extends conformal risk control to provide time-uniform (anytime-valid) guarantees as calibration data accumulates.
- **[Conformal Thinking: Risk Control for Reasoning on a Compute Budget](https://arxiv.org/abs/2602.03814)**: Uses conformal risk control to calibrate early stopping for autoregressive reasoning under a compute budget.
- **[Valid Stopping for LLM Generation via Empirical Dynamic Formal Lift (EDFL)](https://arxiv.org/abs/2510.06478)**: Constructs anytime-valid statistical stopping rules for autoregressive generation via e-processes.
- **[ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](https://arxiv.org/abs/2509.15148)**: Uses conformal prediction to accept/reject candidate reasoning trajectories in speculative test-time scaling.
- **[LLaDA](https://arxiv.org/abs/2502.09992)**: Large language diffusion model family evaluated by many decoding acceleration papers.
- **[Dream](https://arxiv.org/abs/2508.15487)**: Diffusion LLM family with full-sequence refinement.
- **[MDLM](https://arxiv.org/abs/2406.07524)**: Masked diffusion language modeling.
- **[SEDD](https://arxiv.org/abs/2310.16834)**: Discrete score-entropy diffusion for language.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Early exit (global) | Stop and fill all remaining tokens once confidence stabilizes | Prophet; SchED | GSM8K, MMLU, LongBench, WMT | Threshold/schedule tuning; no guarantees |
| Early exit (token-level) | Freeze stable tokens early, keep refining uncertain positions | Jot; KLASS; Learn2PD | GSM8K, HumanEval, MATH500 | Hyperparameters; unclear calibration guarantees |
| Adaptive inference policies | Adapt steps/block size/vocab based on confidence/trajectory features | CadLLM; CCD; LRD | GSM8K, HumanEval, MBPP | Many knobs; no risk guarantees |
| Certified / theory-guided exits | Use training dynamics or stability certificates to justify exits | TRACE; EDIT | Reasoning benchmarks | Requires metadata or assumptions |
| Risk-controlled early exit | Choose exit threshold to satisfy distribution-free risk bounds | Fast yet Safe; Conformal Risk Control | ImageNet, NLP, diffusion images | Not yet applied to dLLM early-exit thresholds |

### Closest Prior Work

- **Jot**: Provides a strong token-level early stopping mechanism and reports large speedups, but τ_max differs by model (Dream vs LLaDA) and there is no guarantee on how many previously-correct examples become incorrect.
- **Fast yet Safe**: Provides CRC/UCB calibration for early-exit networks with distribution-free guarantees (under a monotonicity assumption), but does not study diffusion-LM decoding or token-level early stopping.
- **SchED / Prophet / KLASS**: Provide training-free early exit / stability-based acceleration, but require choosing thresholds/schedules without a user-interpretable risk target.
- **CadLLM / CCD / LRD**: Improve throughput with adaptive policies, but introduce multiple hyperparameters and do not provide distribution-free risk control.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Jot | Token-level early stopping with fixed τ_max | τ tuning; no guarantees | Calibrate τ via risk control | Converts tuning into a user-specified ε guarantee |
| SchED | Progress-aware schedule-based early exit | schedule choice; no guarantees | Risk-controlled choice of aggressiveness | Guarantees relative-quality constraint |
| CadLLM | Multiple confidence-driven adaptive knobs | many hyperparameters; heuristic | single-knob, guarantee-driven calibration | simpler + provides distribution-free bound |
| Fast yet Safe | Risk control for early exit networks | not applied to diffusion LM decoding | apply CRC/UCB to dLLM early exit | extends guarantees to diffusion-LM regime |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| LLaDA-8B-Instruct | 8B | https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct | Diffusion LM evaluated by Jot/SchED/KLASS |

**Training Data (if applicable):**

No training data needed (inference + calibration only).

**Resource Estimate**:
- **Compute budget**: primary cost is repeated inference. One full GSM8K test pass uses ~1319×256 ≈ 3.38e5 denoising steps (forward passes); one full HumanEval pass uses ~164×512 ≈ 8.4e4 steps. With a τ grid of ~10–20 values on a calibration split, total forward-pass count is on the order of a few million. This should fit well within 768 A100 GPU-hours with modest parallelization (e.g., 4–8 GPUs).
- **GPU memory**: LLaDA-8B bf16 inference should fit on a single 80GB GPU.
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| GSM8K | Grade-school math word problems with exact final-answer checking | Accuracy (exact match after answer extraction) | train/test | https://huggingface.co/datasets/openai/gsm8k | https://github.com/EleutherAI/lm-evaluation-harness |
| HumanEval | Code generation problems with unit tests | Pass@1 (fraction of problems passing tests with 1 sample) | test | https://github.com/openai/human-eval | https://github.com/EleutherAI/lm-evaluation-harness |

### Main Results

#### Results Table

Baseline numbers below are copied from Jot’s Table 1 (same base model and evaluation protocol; see `./references/Just-on-Time-Token-Level-Early-Stopping-for-Diffusion-Language-Models/sections/Experimental Setup.md`).

| Method | Base Model | Benchmark | Score | Speedup (configured/actual steps) | Source | Notes |
|--------|------------|-----------|-------|-----------------------------------|--------|------|
| Full decoding | LLaDA-8B-Instruct | GSM8K | 74.5 (accuracy; higher is better) | 1.00× | Jot (Table 1) | 256 diffusion steps |
| Jot (τ_max=30) | LLaDA-8B-Instruct | GSM8K | 73.4 (accuracy; higher is better) | 3.75× | Jot (Table 1) | τ_min=1, γ=0.5, D=8 |
| Naive-calibrated τ_emp | LLaDA-8B-Instruct | GSM8K | **TBD** | **TBD** | - | Needs re-run |
| RC-Jot (UCB τ_UCB) | LLaDA-8B-Instruct | GSM8K | **TBD** | **TBD** | - | To be verified |
| Full decoding | LLaDA-8B-Instruct | HumanEval | 47.6 (Pass@1; higher is better) | 1.00× | Jot (Table 1) | 512 diffusion steps |
| Jot (τ_max=30) | LLaDA-8B-Instruct | HumanEval | 44.5 (Pass@1; higher is better) | 2.12× | Jot (Table 1) | τ_min=1, γ=0.5, D=8 |
| Naive-calibrated τ_emp | LLaDA-8B-Instruct | HumanEval | **TBD** | **TBD** | - | Needs re-run |
| RC-Jot (UCB τ_UCB) | LLaDA-8B-Instruct | HumanEval | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| RC-Jot (UCB) | - | Satisfies risk constraint with reasonable speedup |
| RC-Jot (CRC) | Use CRC instead of UCB | Less conservative, may violate risk more often but good for expectation control |
| RC-Jot w/ different τ grids | Coarse vs fine τ candidate sets | Coarse grid may reduce achievable speedup at same ε |

### Analysis (Optional)

- **Monotonicity check**: plot empirical risk vs τ_max; if strongly non-monotone, the risk-control assumption is violated.
- **Risk-speed curve**: show speedup vs empirical risk across τ values, highlighting where UCB selects τ.

---

## Success Criteria

**Criterion 1: Valid risk control**
- Hypothesis: With UCB calibration, RC-Jot achieves empirical risk (fraction of full-successes broken) ≤ ε on held-out test, for user-chosen (ε,δ) and sufficiently large calibration n.
- Validation: Compute the empirical risk on test; check it is ≤ ε up to finite-sample slack.

**Criterion 2: Non-trivial speedup under guarantees**
- Hypothesis: The τ_UCB selected by risk control is not so conservative that speedup collapses to ~1×.
- Validation: Achieve ≥2× speedup on GSM8K at the chosen ε (or else the method is not practically useful).

**Refutation condition**
- If the empirical risk is clearly non-monotone in τ_max on the diagnostic sweep, or if τ_UCB consistently yields speedup ≈1× for reasonable ε (e.g., 0.05–0.10), then the approach is not viable for Jot-like early exit.

---

## Impact Statement

If successful, RC-Jot would let practitioners deploy diffusion-LM early exit by selecting thresholds via a small calibration set with a user-chosen error budget, replacing ad-hoc per-task tuning with a distribution-free guarantee on how many previously-correct answers are lost.

---

## References

- [Just on Time: Token-Level Early Stopping for Diffusion Language Models](./references/Just-on-Time-Token-Level-Early-Stopping-for-Diffusion-Language-Models/meta/meta_info.txt)
- [Diffusion Language Models Know the Answer Before Decoding](./references/Diffusion-Language-Models-Know-the-Answer-Before-Decoding/meta/meta_info.txt)
- [Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules](./references/Fast-Decoding-Diffusion-Language-Models-via-Progress-Aware-Confidence-Schedules/meta/meta_info.txt)
- [Fast yet Safe: Early-Exiting with Risk Control](./references/Fast-yet-Safe-Early-Exiting-with-Risk-Control/meta/meta_info.txt)
- [Conformal Risk Control](./references/Conformal-Risk-Control/meta/meta_info.txt)
- [Improving the Throughput of Diffusion-based Large Language Models via a Training-Free Confidence-Aware Calibration](./references/Improving-the-Throughput-of-Diffusion-based-Large-Language-Models-via-a-Training-Free-Confidence-Aware-Calibration/meta/meta_info.txt)
- [KLASS: KL-Guided Fast Inference in Masked Diffusion Models](https://arxiv.org/abs/2511.05664)
- [Learning to Parallel: Accelerating Diffusion Large Language Models via Adaptive Parallel Decoding](https://arxiv.org/abs/2509.25188)
- [Conformal Thinking: Risk Control for Reasoning on a Compute Budget](https://arxiv.org/abs/2602.03814)
- [Anytime-Valid Conformal Risk Control](https://arxiv.org/abs/2602.04364)
