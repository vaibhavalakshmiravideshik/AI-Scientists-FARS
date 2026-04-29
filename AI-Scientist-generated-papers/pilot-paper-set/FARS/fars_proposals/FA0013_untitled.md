# untitled

# Contractive Recurrent Cores for Depth-Extrapolatable Vision-Language-Action Policies

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, CoRL (or similar top AI conferences)

## Introduction

### Context and Motivation

Recent Vision-Language-Action (VLA) policies improve robotic manipulation by conditioning an action policy on camera images and natural-language instructions, typically using a pretrained vision-language backbone plus an action head trained on robot trajectories. A fast-emerging theme is **test-time compute scaling** for control: instead of running a fixed-depth policy network once per timestep, the policy performs **iterative latent refinement** (reusing the same parameters across iterations) so that difficult states can receive **additional refinement iterations** at inference without increasing parameter count.

**Recurrent-Depth VLA (RD-VLA)** demonstrates that weight-tied latent recurrence can dramatically improve manipulation success rates while keeping a constant memory footprint, and it explicitly frames recurrence depth as a deployment-time parameter controlling inference compute. However, RD-VLA also reports a key limitation: **depth generalization has a boundary**—beyond an optimal number of recurrent iterations, additional unrolling can lead to **state saturation or performance degradation**.

If recurrent depth is to become a reliable deployment-time parameter (e.g., “use K=64 for difficult grasps”), practitioners need a training recipe that makes unrolling **safe** beyond the trained regime, not merely efficient within it.

### The Problem

RD-VLA empirically shows non-monotonic behavior as recurrence depth increases. On LIBERO, performance peaks at Rec=24 (Avg 93.1%) and drops at Rec=32 (Avg 92.1%) (Table II in RD-VLA). The paper’s discussion emphasizes this as a real limitation (“depth generalization boundary”). However, Table II does not report variance for success rates, so the 1.0-point drop could be within measurement noise. Our verification plan therefore includes a premise check that probes **much larger depths** (e.g., K=64/128) and uses per-trajectory offline error-vs-depth curves to confirm whether a depth boundary is reproducible in our setting.

Today, there are two common reactions to non-monotonic depth behavior:
1. **Adaptive stopping** (stop when outputs stabilize): useful for saving compute, but it does not guarantee that deeper unrolling would be safe if we *choose* to spend compute.
2. **Selective iteration** (apply extra iterations only where needed): addresses “overthinking” in language models, but it does not directly answer how to make *uniform deeper unrolling* stable in the worst case.

We want a method that changes the practical recommendation for RD-style VLA deployment:
- **Before**: “Tune K on a validation set; do not exceed the observed optimum; deeper may hurt.”
- **After**: “Train with a stability regularizer; deeper unrolling stays safe (no systematic overthinking), so K can be increased when needed.”

### Key Insight and Hypothesis

RD-VLA’s weight-tied recurrent core defines an **iterated update map** over a latent scratchpad state. Iterated maps can exhibit drift, oscillation, or collapse when the update’s Jacobian w.r.t. the recurrent state has a large spectral radius (i.e., the map is not contractive). This is closely analogous to stability issues in implicit / equilibrium models, where **Jacobian regularization** (estimated efficiently by Hutchinson trace methods) improves convergence stability.

**Hypothesis**: Adding a lightweight Jacobian-norm regularizer on RD-VLA’s recurrent update map during training will reduce “overthinking” at large inference depths and extend the stable depth regime (e.g., K=64+) without harming the base-depth (K=12) performance.

This could fail if RD-VLA’s depth boundary is not caused by instability (e.g., it converges to a wrong fixed point), in which case contractivity may not help. Our experiments explicitly distinguish these failure shapes using per-example error-vs-depth curves.

---

## Proposed Approach

### Overview

We modify RD-VLA training by adding a **Hutchinson-estimated Jacobian regularization term** targeting the recurrent core’s sensitivity to its own latent state. The goal is to make the recurrent refinement operator closer to a contraction, so repeated application does not systematically corrupt already-correct action plans at higher K.

### Method Details

**Base architecture (RD-VLA recap).** RD-VLA decomposes the action head into:
- **Prelude**: produces a grounded latent foundation state \(S_{pre}\) from the vision-language backbone (VLM) features.
- **Recurrent core**: a weight-tied transformer block that updates a latent **scratchpad state** \(S_k\) (a small set of learned latent tokens used as an internal workspace) for \(k=1..K\) refinement iterations.
- **Coda**: decodes the final scratchpad \(S_K\) into an action chunk.

At iteration \(k\), RD-VLA updates the scratchpad \(S_{k-1}\to S_k\) using **input injection**: each recurrent step re-injects the fixed foundation \(S_{pre}\) (via a learned adapter) so that repeated unrolling does not drift away from the observation-conditioned manifold.

**Training regime note (TBPTT).** RD-VLA uses **truncated backpropagation through time (TBPTT)**: it samples long unroll lengths during training (mean \(\mu_{rec}=32\)) but only backpropagates gradients through the final \(d=8\) iterations. Our regularizer is aligned to this window.

**Jacobian regularization target.** Let \(F_\theta\) denote **one** recurrent update step mapping \(S_{k-1}\to S_k\), including the input-injection adapter and the weight-tied recurrent transformer block.

- **What is held fixed**: the conditioning manifold (VLM visual/latent tokens and proprioception) and the Prelude foundation state \(S_{pre}\). When computing the Jacobian penalty, we stop gradients into these conditioning inputs so the penalty targets only the **recurrent dynamics**.
- **What is regularized**: the sensitivity of the update to its own state, \(J = \partial F_\theta / \partial S\). We do **not** regularize the Coda/action projection directly.

We add a penalty that reduces the Frobenius norm of \(J\), estimated with one Hutchinson probe \(\varepsilon\):

- Sample \(\varepsilon \sim \mathcal{N}(0, I)\)
- Compute \(v = \varepsilon^\top J\) (a vector–Jacobian product)
- Penalize \(\|v\|_2^2 / d\) where \(d\) is the number of scalar entries in \(S\)

**Where to apply the penalty (compute-aware).** To keep overhead small, we apply the penalty:
- only on a random mini-batch fraction \(p\) (default \(p=0.25\)),
- only at one randomly chosen iteration within the TBPTT window (RD-VLA backpropagates through the final \(d=8\) iterations),
- with a single probe vector (\(M=1\)).

Total loss:

L_total = L_action + λ_J · L_Jac

where L_action is the original RD-VLA action prediction loss and L_Jac is the Hutchinson Jacobian penalty.

### Key Innovations

- **Operator-stability regularization for depth extrapolation**: We regularize the recurrent update map’s Jacobian with respect to its own state (\(\partial F/\partial S\)), directly targeting stability of iterated latent refinement (depth generalization). This differs from (i) observation-robustness Jacobian regularization and (ii) loss-shaping approaches that only encourage monotone improvement of decoded actions.
- **Compute-aware Hutchinson penalty placement**: Apply Jacobian regularization only within the TBPTT window and stochastically (\(p=0.25\), \(M=1\)), aiming for a practical overhead (~20–30%).
- **Overthinking-oriented evaluation for control**: Introduce an offline, sim-free overthinking metric based on per-trajectory error-vs-depth curves to detect when additional recurrence corrupts actions.

---

## Related Work

### Field Overview

VLA models span a spectrum from end-to-end action heads trained by imitation learning to explicit reasoning or planning systems that generate intermediate representations (textual or visual) before actions. In parallel, the broader LLM literature has recently explored **recurrent / looped transformers** as a way to scale test-time compute in latent space.

A recurring difficulty in iterative computation is **non-monotonicity**: additional compute can sometimes reduce quality (“overthinking”), motivating selective iteration, adaptive stopping, or stability-oriented regularization. Separately, the implicit-model literature (Deep Equilibrium models) has developed efficient Jacobian regularizers to stabilize fixed-point iterations.

This proposal connects these threads by treating RD-VLA’s recurrent core as an iterated map whose stability can be shaped by Jacobian regularization, and by measuring depth generalization using error-vs-depth curves.

### Related Papers

- **[Recurrent-Depth VLA](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/meta/meta_info.txt)**: Introduces a recurrent, weight-tied latent action head for VLA and reports a boundary in depth generalization.
- **[Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645)**: Provides practical compute/hyperparameter evidence for LIBERO-scale VLA fine-tuning (50–150K steps on 8×A100/H100) and an efficient training recipe.
- **[OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)**: Open-source VLA baseline that popularized large-scale VLA pretraining and LoRA fine-tuning.
- **[Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)**: Establishes depth-recurrent latent reasoning for language models; architectural inspiration for prelude/core/coda-style designs.
- **[Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)**: Studies looped recurrence as a mechanism for latent-space test-time compute scaling.
- **[Universal Transformers](https://arxiv.org/abs/1807.03819)**: Early looped transformer architecture with iterative refinement and adaptive computation.
- **[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)**: Introduces implicit fixed-point networks, motivating stability analyses via Jacobians.
- **[Stabilizing Equilibrium Models by Jacobian Regularization](./references/Stabilizing-Equilibrium-Models-by-Jacobian-Regularization/meta/meta_info.txt)**: Proposes Hutchinson-estimated Jacobian-norm regularization to stabilize implicit/iterative networks.
- **[RobustVLA](./references/RobustVLA-Robustness-Aware-Reinforcement-Post-Training-for-Vision-Language-Action-Models/meta/meta_info.txt)**: Applies Jacobian regularization with respect to observations for robustness in VLA RL post-training (different target than depth stability).
- **[σReparam: Stable Transformer Training with Spectral Reparametrization](https://openreview.net/forum?id=QwqxO8URJzn)**: Uses spectral reparameterization to improve stability of transformer training; a generic Lipschitz-control alternative to Jacobian penalties.
- **[Contractive Diffusion Policies](./references/Contractive-Diffusion-Policies/meta/meta_info.txt)**: Uses contraction-theoretic regularization to prevent error amplification during diffusion-policy sampling.
- **[Jacobian Regularization Stabilizes Long-Term Integration of Neural Differential Equations](./references/Jacobian-Regularization-Stabilizes-Long-Term-Integration-of-Neural-Differential-Equations/meta/meta_info.txt)**: Shows Jacobian regularization improves stability for long-horizon iterative dynamical systems.
- **[Think-at-Hard](https://arxiv.org/abs/2511.08577)**: Addresses non-monotonicity (“overthinking”) in recurrent language models via selective iteration rather than stabilizing uniform deep unrolling.
- **[Diffusion Policy](https://arxiv.org/abs/2303.04137)**: Iteratively refines action trajectories via denoising; contrasts with latent-space refinement in RD-VLA.
- **[π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)**: Flow-matching action generation for VLA, representing iterative computation in the output space.
- **[π0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)**: Strong VLA baseline with open-world generalization; relevant as a deployment-scale point of reference.
- **[ThinkAct](https://arxiv.org/abs/2507.16815)**: VLA reasoning method using reinforced latent planning, representative of token/latent reasoning hybrids.
- **[Fast-ThinkAct](https://arxiv.org/abs/2601.09708)**: Improves VLA reasoning efficiency via verbalizable latent planning.
- **[CoT-VLA](https://arxiv.org/abs/2503.22020)**: Uses visual chain-of-thought supervision for robotic manipulation.
- **[TraceVLA](https://arxiv.org/abs/2412.10345)**: Uses visual trace prompting to improve spatial-temporal skills in robot manipulation.
- **[FlowVLA](https://arxiv.org/abs/2508.18269)**: Uses motion/flow representations as an intermediate reasoning channel for VLA.
- **[VLA-Cache](https://arxiv.org/abs/2502.02175)**: Improves inference efficiency by caching cross-step visual/text tokens in VLA control loops.
- **[DeeR-VLA](https://arxiv.org/abs/2411.02359)**: Studies dynamic inference/early exiting for efficient robot execution.
- **[Embodied Chain-of-Thought Reasoning](https://arxiv.org/abs/2407.08693)**: Uses explicit chain-of-thought reasoning tokens for embodied decision making.
- **[RT-1](https://arxiv.org/abs/2212.06817)** and **[RT-2](https://arxiv.org/abs/2307.15818)**: Large-scale robotics transformers that motivate robust training/evaluation protocols.
- **[LIBERO](https://arxiv.org/abs/2306.03310)**: Simulation benchmark + datasets for language-conditioned manipulation used by RD-VLA.
- **[CALVIN](https://arxiv.org/abs/2112.03227)**: Long-horizon language-conditioned manipulation benchmark used by RD-VLA.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Token-based reasoning VLAs | Generate intermediate tokens/traces before actions | CoT-VLA, TraceVLA, ThinkAct | LIBERO, CALVIN | Token overhead; memory scales with reasoning |
| Action-space iterative heads | Iteratively refine action distribution | Diffusion Policy, FlowVLA | Robomimic, LIBERO | Many sampling steps; instability under solver error |
| Latent recurrence (weight-tied) | Iterative refinement in latent space with constant memory | RD-VLA, recurrent-depth LMs | LIBERO, reasoning benchmarks | Depth boundary / overthinking |
| Stability regularization | Constrain Jacobian / spectral norms for stable iteration | DEQ Jacobian reg, σReparam, CDP | WikiText/ImageNet; robotics | Can reduce expressivity if too strong |

### Closest Prior Work

1. **RD-VLA**: Establishes latent iterative refinement for VLA and reports a depth boundary; does not provide a training recipe to make deeper unrolling safe.
2. **DEQ Jacobian regularization (Bai et al., 2021)**: Provides the key mechanism (Hutchinson Jacobian penalty) but is studied for equilibrium models, not for finite-iteration latent refinement in control.
3. **RobustVLA**: Uses Jacobian regularization for observation robustness in VLA RL post-training; does not target stability under deeper recurrence iterations.
4. **Think-at-Hard**: Addresses overthinking via selective iteration in language models; it is not designed for control policies nor for extending stable *uniform* depth.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| RD-VLA | Latent recurrence for VLA; adaptive stopping | Depth boundary; no stability training | Add Jacobian regularizer on recurrent map | Should reduce overthinking at high K |
| RobustVLA | Jacobian reg w.r.t. observations for robustness | Targets input noise, not depth extrapolation | Regularize ∂F/∂S for recurrent state stability | Directly targets iteration stability |
| Bai et al. 2021 (DEQ) | Jacobian reg for implicit fixed points | Not applied to VLA/test-time compute | Adapt Hutchinson penalty to RD-VLA core | Same mechanism; new domain + metrics |
| σReparam | Stabilizes transformer training via spectral reparam | Not specific to recurrence depth behavior | Compare/mention as alternative | Jacobian reg is more targeted to iterated map |
| Think-at-Hard | Selective iterations to avoid overthinking | Requires oracle/decider; LM-focused | Keep RD-VLA architecture; stabilize instead | Simpler for control; maintains uniform compute option |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| RD-VLA backbone init (MiniVLA Qwen2.5-0.5B-based) | ~0.5B | https://huggingface.co/collections/Stanford-ILIAD/minivla | RD-VLA instantiates the backbone using a MiniVLA-style recipe (Qwen2.5-0.5B + frozen DINOv2+SigLIP vision encoder + LoRA). Verification should pick a concrete MiniVLA checkpoint from this collection (e.g., a LIBERO-90 prismatic checkpoint) as initialization. |

**Code resources (risk note).** The official RD-VLA GitHub repository (https://github.com/rd-vla/rd-vla) is currently a placeholder (≈5 KB, no code). Verification should therefore implement RD-VLA from the paper’s method description and/or reuse OpenVLA/MiniVLA codebases for data loading and LoRA fine-tuning.

**Training Data (offline imitation):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| LIBERO demonstrations (one suite, e.g., Long) | Train + eval (offline) | 10 tasks | https://libero-project.github.io/datasets or HF mirror | Research use (per LIBERO) |

**Resource Estimate**:
- **Compute budget (minimal decisive plan)**: 2 training runs (baseline + ours), **compute-matched by GPU-hours**.
  - Run each method for a fixed wall-clock cap of **8×A100×24h = 192 GPU-hours** on a single LIBERO suite (e.g., LIBERO-Long). This yields **≤384 GPU-hours** total for training.
  - Offline evaluation at multiple recurrence depths is inference-heavy but does not require backprop; budget **≤50 GPU-hours**.
  - **Total (minimal plan)**: ≤434 GPU-hours.
- **Optional robustness ablation (if budget allows)**: 1 additional training run (e.g., stronger weight decay on the recurrent core, no Jacobian penalty) under the same 192 GPU-hour cap (total still ≤626 GPU-hours).
- **GPU memory**: A100 80GB should be sufficient for a ~0.5B backbone + LoRA + recurrent head; multi-GPU DDP recommended for throughput.
- **Notes**:
  - The Jacobian penalty adds extra autograd passes; within the fixed GPU-hour cap, the Jacobian-regularized run may complete fewer gradient steps. This makes comparisons conservative for the baseline (the baseline gets equal or more optimizer steps).
  - The compute estimate above is intentionally conservative because it is grounded in OpenVLA-OFT (7B) fine-tuning reports; RD-VLA uses a smaller backbone (~0.5B), so actual GPU-hours may be lower.

**Infrastructure constraints**:
- We avoid simulator rollouts by default (offline-only evaluation). If LIBERO/CALVIN rollouts are confirmed feasible, we include an optional success-rate validation on a small task subset.

### Benchmarks and Metrics

**Primary (sim-free) metric suite (decisive):**

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| LIBERO demo trajectories (offline) | Teacher-forced action prediction on held-out LIBERO demonstration trajectories (no environment rollouts) | (1) Action-chunk MSE (and/or NLL if modeling action distributions) at K∈{12,32,64,128}; (2) Overthinking rate: \(\Pr[\mathrm{err}(128) > (1+\epsilon)\,\mathrm{err}(12)]\) with \(\epsilon=0.01\); (3) Depth optimum \(K^* = \arg\min_K \mathrm{err}(K)\) over tested K | held-out episodes | https://libero-project.github.io/datasets | Use LIBERO data loader; implement RD-VLA forward pass + offline metric computation; report 95% bootstrap CIs over episodes |

**Optional (if simulator allowed):**
- LIBERO success rate for K∈{12,32,64} on the same suite.

### Main Results

#### Published context (online rollouts; from RD-VLA)

(These are **not directly comparable** to the offline metrics below, but provide context on the scale of the model and benchmark.)

| Method | Benchmark | Metric | Published result | Reference |
|---|---|---|---|---|
| RD-VLA (Fixed Rec=12) | LIBERO (4 suites) | Avg success ↑ | 93.0% (Spat 92.0 / Obj 99.0 / Goal 96.0 / Long 84.8) | RD-VLA Table I (see `./references/.../sections/IV. EXPERIMENTS.md`) |
| RD-VLA (Fixed Rec=24) | LIBERO (4 suites) | Avg success ↑ | 93.1% | RD-VLA Table II (see `./references/.../sections/B. Necessity of Task-dependent Computation.md`) |
| RD-VLA (Fixed Rec=32) | LIBERO (4 suites) | Avg success ↑ | 92.1% | RD-VLA Table II |

#### Verification table (offline; decisive)

(All values TBD; must be filled by verification runs. These rows are directly comparable: same offline split, same metrics.)

| Method | Base Model | Benchmark | MSE@K=12 ↓ | MSE@K=64 ↓ | Overthinking rate (K=128 vs 12) ↓ | Depth-optimum K* ↑ | Source | Notes |
|---|---|---|---|---|---|---|---|---|
| Baseline RD-VLA (fixed depth) | MiniVLA-0.5B + RD head | LIBERO (offline) | **TBD** | **TBD** | **TBD** | **TBD** | To be run | Train with RD-VLA recipe (no Jacobian reg) |
| Baseline RD-VLA + adaptive stopping (inference-time) | MiniVLA-0.5B + RD head | LIBERO (offline) | **TBD** | **TBD** | **TBD** | **TBD** | To be run | Use RD-VLA convergence-based stopping (e.g., \(\|a_k-a_{k-1}\|_2^2 < \tau\) for a tuned \(\tau\)); match mean iterations to a fixed-depth baseline (e.g., Rec=12) and report both mean iterations and offline error metrics. |
| **Ours: RD-VLA + Jacobian reg** | MiniVLA-0.5B + RD head | LIBERO (offline) | **TBD** | **TBD** | **TBD** | **TBD** | To be run | Hutchinson Jacobian penalty on recurrent map |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | Jacobian regularization (p=0.25, M=1) applied to recurrent core dynamics | Lowest overthinking rate at high K |
| w/o Jacobian reg | λ_J=0 (no Jacobian penalty) | Higher overthinking rate; earlier depth optimum |
| Compute-matched longer training baseline | λ_J=0 but allocate extra optimizer steps / wall-clock so total GPU-hours match the Jacobian-regularized run | If gains are due only to extra compute, this matches ours; otherwise ours wins |
| (Optional) Weight decay control | Increase weight decay only on recurrent core parameters (no Jacobian penalty), matched GPU-hours | Tests whether generic parameter norm shrinkage explains gains |
| (Optional) Spectral-norm control | Apply σReparam-style spectral reparameterization only to recurrent core projections, matched GPU-hours | Tests whether weight-space Lipschitz control matches Jacobian reg |

---

## Success Criteria

**Criterion 1: Depth boundary exists (premise check; can refute early).**
- Hypothesis: The baseline exhibits meaningful degradation when unrolled far beyond the typical regime (e.g., K=128 vs K=12) on offline metrics.
- Validation: The overthinking rate \(\Pr[\mathrm{err}(128) > (1+\epsilon)\,\mathrm{err}(12)]\) is clearly above 0 (with 95% bootstrap CI excluding 0). If not observed, stop and refute: in this setting RD-VLA depth generalization is not a bottleneck.

**Criterion 2: Jacobian regularization improves deep-unroll stability at matched compute without harming base depth.**
- Hypothesis: Jacobian regularization reduces the overthinking rate and/or shifts \(K^*\) to larger values, while keeping \(\mathrm{err}(12)\) comparable.
- Validation:
  - At matched training GPU-hours, overthinking rate decreases vs the baseline **and** \(\mathrm{err}(12)\) does not materially regress.
  - The compute-matched longer-training baseline does not fully close the gap, suggesting the effect is not only “more training.”

**Criterion 3 (optional): Offline stability correlates with task success at high K.**
- Hypothesis: If simulator rollouts are feasible, the model with reduced overthinking also avoids success degradation at K=64.
- Validation: On a small LIBERO subset, success@K=64 for our model does not drop relative to K=12 (and drops less than baseline).

---

## Impact Statement

If successful, this work provides a practical training recipe that makes recurrent depth a **safer deployment-time parameter** for VLA policies: users can increase K for hard states without fearing systematic performance degradation from overthinking. This would directly affect how practitioners deploy RD-style VLA architectures and how future test-time compute scaling methods are trained for embodied control.

---

## References

- [Recurrent-Depth VLA](./references/Recurrent-Depth-VLA-Implicit-Test-Time-Compute-Scaling-of-Vision-Language-Action-Models-via-Latent-Iterative-Reasoning/meta/meta_info.txt) - Tur et al., 2026
- [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645) - Kim et al., 2025
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) - Kim et al., 2024
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) - Geiping et al., 2025
- [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) - Zhu et al., 2025
- [Universal Transformers](https://arxiv.org/abs/1807.03819) - Dehghani et al., 2018
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) - Bai et al., 2019
- [Stabilizing Equilibrium Models by Jacobian Regularization](./references/Stabilizing-Equilibrium-Models-by-Jacobian-Regularization/meta/meta_info.txt) - Bai et al., 2021
- [RobustVLA](./references/RobustVLA-Robustness-Aware-Reinforcement-Post-Training-for-Vision-Language-Action-Models/meta/meta_info.txt) - Zhang et al., 2025
- [σReparam](https://openreview.net/forum?id=QwqxO8URJzn) - Zhai et al., 2023
- [Contractive Diffusion Policies](./references/Contractive-Diffusion-Policies/meta/meta_info.txt) - Abyaneh et al., 2026
- [Jacobian Regularization Stabilizes Long-Term Integration of Neural Differential Equations](./references/Jacobian-Regularization-Stabilizes-Long-Term-Integration-of-Neural-Differential-Equations/meta/meta_info.txt) - Janvier et al., 2026
- [Think-at-Hard](https://arxiv.org/abs/2511.08577) - Fu et al., 2025
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - Chi et al., 2023
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) - Black et al., 2024
- [π0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054) - Physical Intelligence et al., 2025
- [ThinkAct](https://arxiv.org/abs/2507.16815) - Huang et al., 2025
- [Fast-ThinkAct](https://arxiv.org/abs/2601.09708) - Huang et al., 2026
- [CoT-VLA](https://arxiv.org/abs/2503.22020) - Zhao et al., 2025
- [TraceVLA](https://arxiv.org/abs/2412.10345) - Zheng et al., 2024
- [FlowVLA](https://arxiv.org/abs/2508.18269) - FlowVLA authors, 2025
- [VLA-Cache](https://arxiv.org/abs/2502.02175) - Xu et al., 2025
- [DeeR-VLA](https://arxiv.org/abs/2411.02359) - Yueyang et al., 2024
- [Embodied Chain-of-Thought Reasoning](https://arxiv.org/abs/2407.08693) - Embodied CoT authors, 2024
- [RT-1](https://arxiv.org/abs/2212.06817) - Brohan et al., 2022
- [RT-2](https://arxiv.org/abs/2307.15818) - Brohan et al., 2023
- [LIBERO](https://arxiv.org/abs/2306.03310) - Liu et al., 2023
- [CALVIN](https://arxiv.org/abs/2112.03227) - Mees et al., 2021
