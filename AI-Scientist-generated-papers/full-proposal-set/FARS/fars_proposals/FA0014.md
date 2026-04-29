# untitled

# Residual-Norm Halting as an Uncertainty Signal for Recurrent-Depth Vision-Language-Action Policies

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, CoRL (or similar top AI conferences)

## Introduction

### Context and Motivation

Vision-Language-Action (VLA) models map observations (e.g., camera images and robot proprioception) and natural-language instructions into low-level robot actions (e.g., delta end-effector pose commands). Open-source VLAs such as **[OpenVLA](./references/OpenVLA-An-Open-Source-Vision-Language-Action-Model/meta/meta_info.txt)** make it practical to train generalist manipulation policies from offline demonstrations.

A fast-emerging direction is **test-time compute scaling** for control: instead of spending one fixed-depth forward pass per control step, a policy runs multiple iterations of internal latent refinement before outputting an action. In language models, recurrent-depth architectures such as **[Scaling up Test-Time Compute with Latent Reasoning](./references/Scaling-up-Test-Time-Compute-with-Latent-Reasoning-A-Recurrent-Depth-Approach/meta/meta_info.txt)** show that repeated latent refinement can improve output quality without generating extra tokens.

**[Recurrent-Depth VLA (RD-VLA)](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/meta/meta_info.txt)** brings this idea to robotics. RD-VLA introduces a weight-tied recurrent transformer “core” that iteratively refines a latent scratchpad state \(S_k\) before decoding an action chunk. RD-VLA reports strong results on **LIBERO** (a suite of language-conditioned simulated manipulation tasks with released demonstration datasets) and **CALVIN** (a long-horizon language-conditioned manipulation benchmark).

Uncertainty estimation and confidence calibration are also emerging as core issues for deploying VLAs safely. **[Confidence Calibration in Vision-Language-Action Models](https://arxiv.org/abs/2507.17383)** studies calibration of VLA confidence scores on LIBERO and proposes prompt ensembles and (action-wise) Platt scaling, but does not consider recurrent-depth signals such as the iteration count \(k^*\) produced by RD-VLA-style models.

### The Problem

RD-VLA’s deployment promise relies on a **halting rule**: for easy states, stop after a few recurrent iterations; for hard states, spend more iterations. RD-VLA stops when consecutive decoded action chunks converge, using an action-space delta threshold (Sec. III-C; Eq. 7 in **[RD-VLA](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/sections/C. Adaptive Computation.md)**):

\[ \|a_k - a_{k-1}\|_2^2 < \delta \]

RD-VLA also uses the stopping depth \(k^*\) as an **uncertainty proxy** to decide how many actions to execute before replanning (“adaptive execution”; Sec. III-D; Eq. 8–9 in **[RD-VLA](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/sections/D. Adaptive Execution.md)**). RD-VLA explicitly reports that “instances requiring deep recurrence (\(k^* > 8\)) often correspond to states of high uncertainty” (Sec. III-D). Here we interpret **epistemic uncertainty** as uncertainty due to model limitations rather than observation noise. The design assumption is that requiring more iterations to converge implies higher epistemic uncertainty, so executing a long open-loop horizon is riskier.

However, action-delta convergence can be poorly calibrated as an uncertainty signal:

1. **The halting signal lives in the output space**, after a learned “Coda” decoding module maps the scratchpad \(S_k\) to continuous actions (Prelude/Core/Coda decomposition; Sec. III-B in **[RD-VLA](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/sections/B. Recurrent-Depth Architecture.md)**). Decoded actions can stabilize even if \(S_k\) is still changing, or jitter even if \(S_k\) is near convergence.

2. **RD-VLA’s own results suggest average task success is often insensitive to the adaptive strategy at matched compute.** In Sec. IV-C, RD-VLA states that multiple adaptive computation strategies perform comparably at matched compute budgets (**[RD-VLA](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/sections/C. Adaptive Computation Strategies.md)**). This weakens a naive claim that a new stopping proxy should significantly improve average success.

The practical question, therefore, is not only whether a stopping proxy improves average task success, but whether it makes \(k^*\) a **better-calibrated uncertainty indicator** for downstream decision-making (adaptive execution, safety triggers, or “request operator assistance”). This use of \(k^*\) is explicitly suggested by RD-VLA (Sec. IV-E discussion).

A naive alternative is to stop based on hidden-state similarity, but this is not guaranteed to help. **[DeeR-VLA](./references/DeeR-VLA-Dynamic-Inference-of-Multimodal-Large-Language-Models-for-Efficient-Robot-Execution/meta/meta_info.txt)** studies dynamic early exiting for robot execution with multimodal large language models (MLLMs) and reports that action-consistency exits outperform feature-similarity exits. This suggests that “representation similarity” is not automatically a better halting signal.

### Key Insight and Hypothesis

RD-VLA differs from early-exit MLLMs because it uses a **weight-tied recurrent update** on a dedicated scratchpad \(S_k\), trained with **truncated backpropagation through time (TBPTT)** (backpropagating gradients through only a limited number of unrolled recurrent steps). This setup is closer to an iterative refinement operator than to comparing features across unrelated feedforward layers.

In **Deep Equilibrium (DEQ) models** (implicit layers defined as fixed points of an iterative update), a standard convergence/uncertainty signal is a **residual norm** \(\|z - f(z)\|\), which measures how much the state changes under another update.

We hypothesize that an analogous residual computed directly on RD-VLA’s scratchpad is a better uncertainty signal than action deltas.

Define the relative scratchpad residual (Frobenius norm = \(\ell_2\) norm over all matrix entries):

\[ r_k = \frac{\|S_k - S_{k-1}\|_F}{\|S_{k-1}\|_F + \epsilon} \]

**Hypothesis:** For RD-VLA-like weight-tied recurrent action heads, residual-norm halting yields a stopping depth \(k^*\) that is **more predictive of action prediction error** than action-delta halting at a matched compute budget (matched mean \(\mathbb{E}[k^*]\)). This would make \(k^*\) a more reliable uncertainty signal for adaptive execution and safety triggers.

This can fail if RD-VLA’s scratchpad dynamics are not refinement-like (e.g., oscillatory), in which case residuals may not decrease with \(k\) and will not correlate with error.

---

## Proposed Approach

### Overview

We propose **Residual-Norm Halting (RNH)**, a training-free inference-time halting rule for recurrent-depth VLA models.

At each recurrent step, compute the relative residual \(r_k\) between scratchpad states and stop recurrence when \(r_k\) falls below a threshold \(\tau_r\), producing \(k^*_r\). We compare against RD-VLA’s action-delta halting (threshold \(\tau_a\)), producing \(k^*_a\).

We evaluate both halting rules primarily as **uncertainty signals**, not as minor efficiency tweaks.

### Method Details

**What is the scratchpad \(S_k\)?** RD-VLA’s recurrent head maintains \(K=8\) learned “query” vectors and an evolving scratchpad \(S_k \in \mathbb{R}^{K\times D}\) (Sec. III-B in **[RD-VLA](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/sections/B. Recurrent-Depth Architecture.md)**). Each recurrent iteration applies the same transformer block (weight tying) to update \(S_{k-1}\rightarrow S_k\), conditioning on frozen VLM features.

**Baseline halting (action-delta).**
- Decode an action chunk \(a_k\) at each iteration from \(S_k\).
- Stop at the smallest \(k\) where \(d_k = \|a_k-a_{k-1}\|_2^2 < \tau_a\).

**Residual-Norm Halting (ours).**
- Compute \(r_k = \|S_k-S_{k-1}\|_F/(\|S_{k-1}\|_F+\epsilon)\).
- Stop at the smallest \(k\) where \(r_k < \tau_r\).

**Fair threshold selection (avoid tuning confounds).**
- Tune \(\tau_a\) and \(\tau_r\) on the same validation split to match a target mean iteration budget (e.g., match RD-VLA’s reported mean \(k\approx7.7\) for \(\tau=5\times10^{-4}\) in Table II, but any fixed target is acceptable).

**Low-cost validity checks (premise tests).**
- **Check 1 (convergence-like dynamics):** Median \(r_k\) should decrease with \(k\) on validation (strong negative Spearman rank correlation; Spearman correlation measures monotonic association).
- **Check 2 (uncertainty predictiveness):** At matched \(\mathbb{E}[k^*]\), \(k^*_r\) should be more predictive of action error than \(k^*_a\) on validation (measured by AUROC; defined below). If not, the proposal predicts no downstream benefit.

### Key Innovations

1. **Reframe halting as uncertainty calibration.** The key question is whether \(k^*\) is a meaningful uncertainty signal for downstream decision-making, not whether the stopping proxy slightly changes average success.
2. **Residual-norm convergence proxy for recurrent scratchpads.** Apply a DEQ-style residual norm to RD-VLA’s scratchpad rather than measuring convergence only in decoded action space.
3. **Offline-only verification plan.** The verification experiments use released offline datasets and model forward passes only; no simulator rollouts are required.

---

## Related Work

### Field Overview

VLA research spans (i) end-to-end imitation policies that directly predict actions from vision-language context, (ii) reasoning-centric VLAs that generate intermediate plans (textual or visual) before actions, and (iii) iterative action-generation methods (diffusion/flow) that refine actions in the output space. RD-VLA is in a fourth category: **latent-space recurrence** with weight tying, enabling test-time compute scaling without token generation.

Adaptive computation has a long history in recurrent networks and transformers. Learned halting mechanisms (e.g., ACT, PonderNet) learn stop/continue probabilities, while equilibrium models (DEQ) use residual norms as convergence signals for fixed-point solvers.

Our proposal connects these threads by asking a concrete question: in recurrent-depth VLA policies where iteration count \(k^*\) is used as an uncertainty signal (adaptive execution), does a scratchpad residual yield a better-calibrated uncertainty proxy than action deltas?

### Related Papers

- **[RD-VLA](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/meta/meta_info.txt)**: Weight-tied latent recurrence for VLA; uses action-delta stopping and uses \(k^*\) for adaptive execution.
- **[Confidence Calibration in Vision-Language-Action Models](https://arxiv.org/abs/2507.17383)**: First systematic study of VLA confidence calibration on LIBERO; proposes prompt ensembles and action-wise Platt scaling, providing calibration metrics (ECE, Brier, NLL) that we can reuse to evaluate \(k^*\)-based uncertainty signals.
- **[OpenVLA](./references/OpenVLA-An-Open-Source-Vision-Language-Action-Model/meta/meta_info.txt)**: Open-source VLA backbone and training pipeline for imitation learning from offline robot demonstrations.
- **[Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](./references/Fine-Tuning-Vision-Language-Action-Models-Optimizing-Speed-and-Success/meta/meta_info.txt)**: Practical LIBERO fine-tuning recipes and training ranges for OpenVLA-style models.
- **[VLA-Cache](./references/VLA-Cache-Efficient-Vision-Language-Action-Manipulation-via-Adaptive-Token-Caching/meta/meta_info.txt)**: Training-free inference efficiency via adaptive token caching; complementary to halting.
- **[DeeR-VLA](./references/DeeR-VLA-Dynamic-Inference-of-Multimodal-Large-Language-Models-for-Efficient-Robot-Execution/meta/meta_info.txt)**: Dynamic early exit for robot execution; finds action consistency beats feature similarity.
- **[Understanding Dynamic Compute Allocation in Recurrent Transformers](./references/Understanding-Dynamic-Compute-Allocation-in-Recurrent-Transformers/meta/meta_info.txt)**: Analysis framework for online halting decisions in recurrent transformers.
- **[Scaling up Test-Time Compute with Latent Reasoning](./references/Scaling-up-Test-Time-Compute-with-Latent-Reasoning-A-Recurrent-Depth-Approach/meta/meta_info.txt)**: Recurrent-depth language model; motivates convergence-based halting.
- **[Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)**: Looped transformer language models with depth extrapolation effects.
- **[Think-at-Hard](https://arxiv.org/abs/2511.08577)**: Selective latent iterations to avoid harmful over-iteration in recurrent language models.
- **[Adaptive Computation Time](https://arxiv.org/abs/1603.08983)**: Classic learned halting for recurrent networks.
- **[Universal Transformers](https://arxiv.org/abs/1807.03819)**: Iterative refinement transformers, often combined with ACT.
- **[PonderNet](https://arxiv.org/abs/2107.05407)**: Learned halting via a probabilistic prior over computation steps.
- **[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)**: Implicit layers trained via fixed points; residual norms as convergence signals.
- **[Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342)**: Regularization that improves convergence properties of equilibrium dynamics.
- **[LIBERO](https://arxiv.org/abs/2306.03310)**: Benchmark with language-conditioned manipulation demonstrations (offline datasets + simulator-based evaluation).
- **[CALVIN](https://arxiv.org/abs/2112.03227)**: Long-horizon benchmark requiring chaining language-conditioned skills.
- **[Diffusion Policy](https://arxiv.org/abs/2303.04137)**: Output-space iterative action refinement via diffusion.
- **[π0](https://arxiv.org/abs/2410.24164)**: Strong VLA baseline based on flow-matching.
- **[π0.5](https://arxiv.org/abs/2504.16054)**: VLA baseline emphasizing open-world generalization.
- **[ThinkAct](https://arxiv.org/abs/2507.16815)**: Reasoning-centric VLA baseline.
- **[Fast-ThinkAct](https://arxiv.org/abs/2601.09708)**: Efficiency-oriented reasoning-centric VLA baseline.
- **[CoT-VLA](https://arxiv.org/abs/2502.19096)**: Token-level chain-of-thought style supervision for VLA.
- **[TraceVLA](https://arxiv.org/abs/2412.10345)**: Visual trace prompting for VLA.
- **[MolmoAct](https://arxiv.org/abs/2508.07917)**: Action reasoning model evaluated on manipulation benchmarks.
- **[Prismatic VLMs](https://arxiv.org/abs/2402.07814)**: VLM training recipe used by MiniVLA-like backbones.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| End-to-end VLAs | Single-pass action prediction | OpenVLA, π0, π0.5 | LIBERO, CALVIN | Fixed compute per step |
| Token/trace reasoning VLAs | Generate intermediate plans (text/visual) | CoT-VLA, TraceVLA, ThinkAct | LIBERO | Latency/memory overhead |
| Output-space iterative heads | Iteratively refine actions in output space | Diffusion Policy | LIBERO/Robomimic | Many steps; halting is heuristic |
| Latent recurrent-depth policies | Weight-tied latent refinement with test-time depth | RD-VLA | LIBERO, CALVIN | Halting + uncertainty calibration |
| Learned halting | Learn stop/continue probabilities | ACT, PonderNet | NLP/algorithmic tasks | Training complexity |
| Residual-based convergence | Stop via fixed-point residual norms | DEQ | vision/NLP | Requires convergent dynamics |

### Closest Prior Work

1. **RD-VLA**: Uses action-delta stopping and uses \(k^*\) for adaptive execution; does not evaluate calibration of \(k^*\) under alternative convergence proxies.
2. **DeeR-VLA**: Shows representation-similarity exits can underperform action-consistency exits; motivates careful validation of latent-space proxies.
3. **DEQ**: Establishes residual norms as principled convergence signals for iterative updates.
4. **Dynamic compute allocation in recurrent transformers**: Provides analysis vocabulary for online halting and timing of decisions.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours might win |
|---|---|---|---|---|
| RD-VLA | Action-delta halting; uses \(k^*\) for adaptive execution | \(k^*\) may be miscalibrated as uncertainty | Use scratchpad residual to define \(k^*\) | Measures convergence directly in the refinement state |
| DeeR-VLA | Early-exit MLLM; compares exit criteria | Feature similarity underperforms | Use residual in a weight-tied scratchpad regime | Residual is meaningful if dynamics are convergence-like |
| DEQ | Residual-norm convergence for fixed points | Not applied to VLA control | Apply DEQ-style residual to RD-VLA scratchpad | Same convergence principle in a new domain |
| ANIRA-like analyses | Studies online halting in recurrent transformers | Not in robotics | Use its framing for uncertainty vs compute | Clarifies what to measure |

---

## Experiments

### Experimental Setup

**Important feasibility note (no simulation required):** The core experiments use **offline demonstration data only**. We do **not** run MuJoCo or interact with a simulator. LIBERO provides recorded demonstrations (images, proprioception, actions, and language instructions) that can be downloaded as dataset files and used for training/evaluation of imitation policies.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| MiniVLA / OpenVLA-style backbone | ~0.5B | https://huggingface.co/collections/Stanford-ILIAD/minivla | RD-VLA instantiates its framework on a Qwen2.5-0.5B backbone with frozen DINOv2+SigLIP vision encoder and LoRA (low-rank adaptation) tuning (Sec. III-A in RD-VLA). |

**Training Data (offline imitation):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| LIBERO-Long demos | Train/eval recurrent head; hardest suite for recurrence | 10 tasks × 500 demos (per suite) | https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets | Research use (per LIBERO) |

**Implementation note (checkpoint availability):** As of 2026-02-12, the official RD-VLA GitHub repository contains only a placeholder README and does not document checkpoints or evaluation scripts. Verification should therefore plan for (a) using a released checkpoint if one appears, or (b) implementing the RD-VLA recurrent head from the paper and training it on LIBERO-Long.

**Resource Estimate** (must fit `resource_budget.md`):
- **Compute budget**:
  - Training (if needed): RD-VLA-like fine-tuning on one LIBERO suite.
    - Evidence for training scale: **[Fine-Tuning VLA Models](./references/Fine-Tuning-Vision-Language-Action-Models-Optimizing-Speed-and-Success/sections/V-A LIBERO Experimental Setup.md)** reports training for **50–150k gradient steps** (non-diffusion) with total batch size 64–128 across **8×A100/H100**.
    - Budget: **≤ 8 GPUs × 48h = 384 GPU-hours** (conservative upper bound).
  - Offline evaluation: forward passes over a held-out set of state-action pairs (no rollouts): **≤ 64 GPU-hours**.
  - Total budget: **≤ 448 GPU-hours**.
- **GPU memory**: ≤ 80GB per GPU.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| LIBERO-Long offline demos | Language-conditioned manipulation demonstrations; we use it as an offline dataset (no simulator) | (1) \(\mathbb{E}[k^*]\) (mean iterations); (2) action L1 error (lower is better); (3) AUROC of \(k^*\) for predicting high-error states (higher is better); (4) ECE/NLL after Platt scaling \(k^*\rightarrow \hat p(\text{high-error})\) (lower is better); (5) risk–coverage curve when abstaining on high \(k^*\) (optional) | train/val/test (episode-level split; sample state-action pairs) | https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets | Custom loader (HDF5) + model forward passes |

**Metric definitions:**
- **Action L1 error**: \(\|\hat a - a^*\|_1\) averaged over the predicted action chunk; lower is better.
- **AUROC** (area under the receiver operating characteristic curve): measures how well a scalar score separates “high-error” vs “low-error” states; higher is better.
- **High-error label**: define “high error” as the top 20% action-error samples on the validation set (fixed threshold), then evaluate AUROC on the test set.
- **Platt scaling (for calibration)**: fit a logistic map \(\hat p=\sigma(\alpha s+\beta)\) from a score \(s\) (here \(s=k^*\)) to the probability of a high-error event on validation.
- **ECE** (expected calibration error; lower is better): absolute gap between predicted probabilities and empirical frequencies, averaged over bins (as in Zollo & Zemel).
- **NLL** (negative log-likelihood; lower is better): proper scoring rule for probabilistic predictions; penalizes overconfident errors.

### Main Results

#### Published context (online rollouts; task success rate, higher is better)

These results are copied from RD-VLA’s raw experiment section and are **not directly comparable** to our offline error metrics.

| Method | Description | Settings | Results | Reference |
|---|---|---|---|---|
| RD-VLA (Fixed) | Fixed recurrence depth | Params 0.5B; LIBERO suites; Rec=12 | Avg **93.0%** (Spat 92.0 / Obj 99.0 / Goal 96.0 / Long 84.8) | RD-VLA Table I in `./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/sections/IV. EXPERIMENTS.md` |
| RD-VLA (Adaptive) | Adaptive computation (Binary Adaptation) | Params 0.5B; LIBERO suites; \(\tau=5\times10^{-4}\) | Avg **92.5%**, mean iters **7.93** (Spat 88.6 / Obj 98.8 / Goal 96.8 / Long 85.8) | RD-VLA Table II in `./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/sections/B. Necessity of Task-dependent Computation.md` |
| RD-VLA (Adaptive) | Adaptive computation (Pure KL thresholding) | Params 0.5B; LIBERO suites; \(\tau=5\times10^{-4}\) | Avg **91.4%**, mean iters **7.66** (Spat 90.8 / Obj 99.4 / Goal 93.2 / Long 82.0) | RD-VLA Table II in `./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/sections/B. Necessity of Task-dependent Computation.md` |

#### Verification table (offline uncertainty calibration; all rows comparable)

(All values TBD; must be filled by verification runs. These rows are directly comparable: same offline split, same metrics, same base model, thresholds tuned to match \(\mathbb{E}[k^*]\).)

| Method | Description | Settings | Results | Reference |
|---|---|---|---|---|
| Baseline: action-delta halting | Use RD-VLA action-delta proxy \(d_k\) to produce \(k^*_a\) | Tune \(\tau_a\) on val to match target \(\mathbb{E}[k^*]\); \(K_{max}=12\) | **TBD**: \(\mathbb{E}[k^*]\), action L1 error, AUROC | Needs re-run (offline metric) |
| **Ours: residual-norm halting** | Use scratchpad residual \(r_k\) to produce \(k^*_r\) | Tune \(\tau_r\) on val to match same \(\mathbb{E}[k^*]\); same \(K_{max}\) | **TBD**: \(\mathbb{E}[k^*]\), action L1 error, AUROC | To be run |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (full) | Relative residual \(\|\Delta S\|/(\|S\|+\epsilon)\) | Best uncertainty calibration |
| w/o normalization | Use \(\|S_k-S_{k-1}\|_F\) directly | Worse calibration due to scale sensitivity |

---

## Success Criteria

**Criterion 1: Offline-only feasibility**
- Hypothesis: The evaluation can be done with offline data and model forward passes only.
- Validation: All metrics are computed from downloaded LIBERO demo files; no simulator rollouts are required.

**Criterion 2: Residuals behave like a convergence signal**
- Hypothesis: The scratchpad residual \(r_k\) decreases with \(k\) on average.
- Validation: Median \(r_k\) decreases across \(k=1..K_{max}\) on validation (strong negative Spearman rank correlation).

**Criterion 3: Better uncertainty calibration at matched compute**
- Hypothesis: At matched \(\mathbb{E}[k^*]\), residual-based \(k^*_r\) is a better uncertainty indicator than action-delta \(k^*_a\).
- Validation: On the test split, AUROC(\(k^*_r\) predicting high-error) > AUROC(\(k^*_a\) predicting high-error), with a 95% bootstrap confidence interval for the AUROC difference that excludes 0.

**Criterion 4 (secondary): Better calibrated probability after Platt scaling**
- Hypothesis: After fitting Platt scaling \(k^*\rightarrow \hat p(\text{high-error})\) on validation, residual-based scores yield better calibration.
- Validation: Test-set ECE and NLL for \(\hat p\) from \(k^*_r\) are lower than those from \(k^*_a\) (as in Zollo & Zemel’s calibration protocol).

---

## Impact Statement

If residual-norm halting improves uncertainty calibration, practitioners building recurrent-depth robot policies gain a training-free way to make iteration count \(k^*\) a more reliable signal for adaptive execution and safety triggers (replan more frequently or request assistance when uncertainty is high). If it fails, the negative result still informs practice: it supports RD-VLA’s implication that convergence proxy choice is not a primary lever even for uncertainty estimation, and effort should focus on other aspects (architecture, training, or explicit uncertainty modeling).

---

## References

- [Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/meta/meta_info.txt) - Tur et al., 2026
- [DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution](./references/DeeR-VLA-Dynamic-Inference-of-Multimodal-Large-Language-Models-for-Efficient-Robot-Execution/meta/meta_info.txt) - Yueyang et al., 2024
- [Understanding Dynamic Compute Allocation in Recurrent Transformers](./references/Understanding-Dynamic-Compute-Allocation-in-Recurrent-Transformers/meta/meta_info.txt) - Moosa et al., 2026
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](./references/Scaling-up-Test-Time-Compute-with-Latent-Reasoning-A-Recurrent-Depth-Approach/meta/meta_info.txt) - Geiping et al., 2025
- [OpenVLA: An Open-Source Vision-Language-Action Model](./references/OpenVLA-An-Open-Source-Vision-Language-Action-Model/meta/meta_info.txt) - Kim et al., 2024
- [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](./references/Fine-Tuning-Vision-Language-Action-Models-Optimizing-Speed-and-Success/meta/meta_info.txt) - Kim et al., 2025
- [VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](./references/VLA-Cache-Efficient-Vision-Language-Action-Manipulation-via-Adaptive-Token-Caching/meta/meta_info.txt) - Xu et al., 2025
- [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310) - Liu et al., 2023
- [CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2112.03227) - Mees et al., 2021
- [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983) - Graves, 2016
- [Universal Transformers](https://arxiv.org/abs/1807.03819) - Dehghani et al., 2019
- [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) - Banino et al., 2021
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) - Bai et al., 2019
- [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342) - Bai et al., 2021
- [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) - Zhu et al., 2025
- [Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models](https://arxiv.org/abs/2511.08577) - Fu et al., 2025
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) - Chi et al., 2023
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) - Black et al., 2024
- [π0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054) - Physical Intelligence et al., 2025
- [ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](https://arxiv.org/abs/2507.16815) - Huang et al., 2025
- [Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](https://arxiv.org/abs/2601.09708) - Huang et al., 2026
- [CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](https://arxiv.org/abs/2502.19096) - Zhao et al., 2025
- [TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Skills for Robot Manipulation](https://arxiv.org/abs/2412.10345) - Zheng et al., 2024
- [MolmoAct: Action Reasoning Models that can Reason in Space](https://arxiv.org/abs/2508.07917) - Lee et al., 2025
- [Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models](https://arxiv.org/abs/2402.07814) - Karamcheti et al., 2024
- [Confidence Calibration in Vision-Language-Action Models](https://arxiv.org/abs/2507.17383) - Zollo & Zemel, 2025 (ICLR 2026 submission)
