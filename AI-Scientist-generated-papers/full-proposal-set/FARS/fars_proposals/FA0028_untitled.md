# untitled

# Acceptance-Controlled MIS-PO: Adaptive Trajectory Filtering for Stable Off-Policy RLVR

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Reinforcement learning with verifiable rewards (RLVR) improves large language models using automated, programmatic feedback signals (e.g., unit tests for code or exact-match verifiers for math). RLVR is attractive because it avoids training a learned reward model, but it is often unstable in practice: training can collapse when the policy changes too quickly or when training uses off-policy data.

A growing source of off-policy behavior is **systems-level training/inference disaggregation**. Modern RL stacks frequently generate rollouts with an inference engine (e.g., vLLM) while computing gradients in a separate training engine (e.g., FSDP or Megatron). In addition, many scalable systems are **asynchronous**: training updates proceed while rollout workers continue generating samples with stale parameters. Both disaggregation and asynchrony create substantial probability mismatches between the rollout (behavior) policy and the policy being optimized.

Recent work has proposed several mechanisms to stabilize these mismatches. PPO-style clipping and its variants bound token-level importance ratios; sequence-level methods (e.g., GSPO/GMPO) change the ratio definition to reduce variance; and off-policy trust-region methods (e.g., M2PO) explicitly enforce a second-moment constraint on ratios to keep gradient variance bounded.

### The Problem

**MIS-Filtered Policy Optimization (MIS-PO)** proposes a different approach: instead of continuously reweighting gradients with importance ratios, it uses **binary accept/reject masks** to discard off-distribution samples and treat the retained set as effectively on-policy.

- **[Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)** introduces MIS-PO with fixed token-level and trajectory-level ratio bounds (token: [0.5, 2.0], trajectory: [0.996, 1.001]) and reports improved stability (lower gradient-norm spikes) relative to PPO-style training.

However, MIS-PO’s stability and efficiency depend critically on choosing these bounds. Fixed bounds can be:

1. **Too tight early or under stronger mismatch**, causing near-zero acceptance and wasting rollouts (low effective batch size).
2. **Too loose later**, admitting high-variance samples and reintroducing instability.
3. **Brittle across models and infrastructures**, requiring expensive manual tuning.

While several recent methods adapt trust-region/clipping behavior (e.g., BAPO, M2PO), it remains unclear whether MIS-PO can be made robust with a similarly simple adaptive rule, and whether a lightweight adaptation based on an easily-measured statistic (acceptance rate) can compete with stronger constraints like second-moment control.

### Key Insight and Hypothesis

We hypothesize that a large part of MIS-PO’s brittleness comes from using **fixed trajectory-level ratio bounds** even though the distribution of trajectory importance ratios changes over RL training (policy drift, partial rollouts, backend nondeterminism). If we adapt the trajectory bound to keep a **target trajectory acceptance rate** (i.e., a stable effective sample size), we can improve sample efficiency and stability without adding complex variance-estimation machinery.

This hypothesis could be wrong: acceptance-rate control may be an insufficient statistic for variance control (e.g., heavy-tailed ratios could still dominate the second moment among accepted samples), in which case methods like **M2PO** should remain clearly better.

---

## Proposed Approach

### Overview

We propose **Acceptance-Controlled MIS-PO (AC-MIS-PO)**: keep MIS-PO’s binary accept/reject mechanism, but **adapt the trajectory-level accept/reject bound per update** using a quantile-based controller that targets a pre-registered acceptance-rate schedule.

The method is designed to be a minimal change to existing RLVR codebases:
- It does not change the reward function, advantage estimator, or optimizer.
- It does not require training an additional critic/reward model beyond what the baseline uses.
- It only changes how rollouts are masked before computing the policy-gradient loss.

### Method Details

**Background (MIS-PO).** Following **[Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)**, define token importance ratios between the training policy \(\pi_{\theta_{old}}\) (computed in the training engine) and the rollout policy \(\pi_{\theta_{vllm}}\) (computed in the inference engine):

- Token ratio: \(x_t = \pi_{\theta_{old}}(a_t\mid s_t) / \pi_{\theta_{vllm}}(a_t\mid s_t)\)
- Trajectory geometric-mean ratio: \(\rho(\tau)=\left(\prod_t x_t\right)^{1/T} = \exp\left(\tfrac{1}{T}\sum_t \log x_t\right)\)
- Indicator: \(I(u;\rho_{min},\rho_{max}) = \mathbb{1}[\rho_{min} \le u \le \rho_{max}]\)

MIS-PO’s actor loss applies binary masks at token and trajectory level:
\[
\mathcal{L}_{\text{actor}} = -\mathbb{E}_{\tau\sim\pi_{\theta_{vllm}}}\left[I(x_t;\rho^{tok}_{min},\rho^{tok}_{max})\,I(\rho(\tau);\rho^{traj}_{min},\rho^{traj}_{max})\,\log \pi_{\theta}(a_t\mid s_t)\,\hat{A}_t\right].
\]

**Acceptance-controlled trajectory bound.** We keep the **token-level bounds fixed** to the Step 3.5 Flash defaults \([\rho^{tok}_{min},\rho^{tok}_{max}]=[0.5,2.0]\), and adapt only the trajectory-level bound.

We parameterize a symmetric trajectory bound in log space by a single scalar \(b_k\ge 0\) at update step \(k\):
\[
\rho^{traj}_{min}(k) = \exp(-b_k),\quad \rho^{traj}_{max}(k)=\exp(b_k).
\]

Given a rollout batch of trajectories \(\{\tau_i\}_{i=1}^N\), compute
\[
 z_i = \lvert \log \rho(\tau_i) \rvert.
\]

Let \(A^*_k\in(0,1)\) be the target trajectory acceptance rate at step \(k\). Define the instantaneous bound candidate as the \(A^*_k\)-quantile:
\[
 b^{\text{cand}}_k = \mathrm{Quantile}_{A^*_k}(\{z_i\}_{i=1}^N).
\]

Then apply EMA smoothing and clamping:
\[
 b_k = \mathrm{clip}(\beta\,b_{k-1} + (1-\beta)\,b^{\text{cand}}_k,\; b_{min},\; b_{max}).
\]

Default controller hyperparameters (pre-registered):
- EMA \(\beta=0.9\)
- Acceptance schedule: \(A^*_k\) linearly decays from 0.40 to 0.20 over the first 50 update steps, then stays at 0.20.
- Clamp range: \(b_{min}=\log(1.0002)\), \(b_{max}=\log(1.01)\).

**Why this might help.** Compared to fixed bounds, this controller (i) keeps a stable effective batch size, (ii) automatically tightens as the rollout/training distributions align, and (iii) avoids manual per-model tuning.

### Key Innovations

1. **Trajectory acceptance-rate control for MIS-PO**: a simple quantile+EMA controller that adapts the trajectory-level ratio bound online.
2. **Minimalism**: no second-moment estimator, no learned variance model, no additional networks.
3. **Ablation to isolate “adaptation” vs “better constant”**: a fixed-at-final-bound control (see Experiments) to test whether per-step adaptation is necessary.

---

## Related Work

### Field Overview

Stabilizing RLVR for LLMs has recently focused on (i) reducing variance from importance sampling, (ii) maintaining exploration/entropy under long-horizon verifiable rewards, and (iii) handling the increasingly common off-policy effects introduced by modern training infrastructure.

Common strategies include PPO-style clipping and its refinements (decoupling clip ranges, entropy-aware clipping, gradient-preserving clipping), sequence-level ratio definitions (to reduce per-token noise), and explicit trust-region constraints (e.g., second-moment bounds) to keep off-policy training stable under stale data. MIS-PO is closely related to these approaches but uses accept/reject filtering rather than continuous clipping.

### Related Papers

- **[Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)**: Proposes MIS-PO with fixed token/trajectory accept-reject bounds to reduce gradient variance under training–inference mismatch.
- **[Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/meta/meta_info.txt)**: Proposes M2PO, masking tokens until a batch-level second-moment constraint falls below a threshold, enabling extreme-staleness RLVR.
- **[A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization (MinPRO)](https://arxiv.org/abs/2601.22718)**: Shows that prefix-level importance ratios (not token ratios) are the theoretically correct off-policy correction for autoregressive policies, and proposes a minimum-prefix-ratio surrogate to stabilize off-policy RLVR.
- **[Trust Region Masking for Long-Horizon LLM Reinforcement Learning (TRM)](https://arxiv.org/abs/2512.23075)**: Derives non-vacuous long-horizon trust-region bounds and proposes masking entire sequences whose maximum token-level divergence exceeds a threshold.
- **[BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping](./references/BAPO-Stabilizing-Off-Policy-Reinforcement-Learning-for-LLMs-via-Balanced-Policy-Optimization-with-Adaptive-Clipping/meta/meta_info.txt)**: Dynamically adjusts asymmetric PPO clipping bounds to control positive-token contribution and preserve entropy under off-policy training.
- **[Online Causal Kalman Filtering for Stable and Effective Policy Optimization](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt)**: Proposes KPO, smoothing token-wise log ratios with a causal Kalman filter to reduce noisy ratio spikes.
- **[Geometric-Mean Policy Optimization](./references/Geometric-Mean-Policy-Optimization/meta/meta_info.txt)**: Uses geometric-mean policy ratios and reports improved stability for RLVR math and multimodal tasks.
- **[Group Sequence Policy Optimization](./references/Group-Sequence-Policy-Optimization/meta/meta_info.txt)**: Defines sequence-level importance ratios to reduce variance and better match RL infrastructure constraints.
- **[Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](./references/Stabilizing-MoE-Reinforcement-Learning-by-Aligning-Training-and-Inference-Routers/meta/meta_info.txt)**: Identifies router inconsistency as a key MoE-specific instability source and proposes replaying inference routing during training.
- **[Tapered Off-Policy REINFORCE (TOPR)](https://arxiv.org/abs/2503.14286)**: Stabilizes off-policy RL by tapering importance weights as policies diverge.
- **[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)**: Open-source RL system and algorithmic practices (decoupled clipping, dynamic sampling, overlong shaping) for long-CoT RL.
- **[AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)**: Shows asynchronous RL infrastructures improve throughput but introduce staleness that must be controlled algorithmically.
- **[HybridFlow](https://arxiv.org/abs/2409.19256)**: RLHF/RLVR training framework that supports disaggregated rollout and training, highlighting practical policy mismatch sources.
- **[veRL](https://github.com/volcengine/verl)**: A widely used open RLVR/RLHF training stack combining rollout engines (vLLM/SGLang) with FSDP/Megatron training.
- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)**: vLLM serving architecture enabling high-throughput rollout generation.
- **[PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)**: Core distributed training approach often used in RLVR stacks.
- **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)**: PPO, the basis of many LLM RL algorithms.
- **[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)**: Classical trust-region RL that motivates KL/ratio constraints.
- **[DeepScaleR dataset / training recipe](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)**: A widely used math RLVR dataset referenced by multiple RLVR works.
- **[QUATRO: Query-Adaptive Trust Region Policy Optimization](https://arxiv.org/abs/2602.04620)**: Proposes query-adaptive trust regions without relying on static clipping.
- **[DCPO: Dynamic Clipping Policy Optimization](https://arxiv.org/abs/2509.02333)**: Dynamically adjusts clipping behavior using prior probabilities.
- **[CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization](https://arxiv.org/abs/2508.07629)**: Coordinates entropy and clipping to stabilize RL updates.
- **[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://arxiv.org/abs/2508.10687)**: Analyzes how disaggregated rollout/training pipelines induce off-policy effects even in nominally on-policy RL.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| PPO-style clipping + entropy control | Bound ratios (often token-wise) to prevent large updates; adjust clipping/entropy to avoid collapse | PPO, TRPO, DAPO, DCPO, CE-GPPO, BAPO | Math/coding RLVR (AIME/MATH/LiveCodeBench) | Clipping can still be high-variance under strong mismatch; sensitive to ratio distribution tails |
| Prefix-aware off-policy correction | Use prefix-level importance ratios (or stable surrogates) to correct off-policy gradients for autoregressive policies | MinPRO | Math RLVR under delayed/stale rollouts | May still rely on soft weighting; unclear behavior under severe engine mismatch |
| Sequence-level ratios | Use sequence/trajectory likelihood ratios to reduce token-level noise | GSPO, GMPO | Math RLVR | Still requires controlling large deviations; may not address extreme staleness |
| Explicit off-policy trust region | Enforce a statistical constraint (e.g., second moment) by masking or reweighting | M2PO, TOPR, QUATRO | Math RLVR under stale rollouts | More complex; may mask too aggressively; tuning constraints can be nontrivial |
| Sequence-level masking trust region | Mask entire sequences whose divergence violates a token-level max bound | TRM | Long-horizon LLM RL | Requires selecting a divergence threshold; may reject many long trajectories |
| Accept/reject filtering | Discard samples outside a ratio bound (discrete masking) | MIS-PO | RLVR under training–inference mismatch | Fixed bounds can be brittle; unclear how to adapt safely |
| Infrastructure-driven mismatch fixes | Change the system to reduce mismatch at the source | Rollout Routing Replay, AReaL, HybridFlow | Large-scale RL pipelines | Requires system modifications; may be model-architecture specific (MoE) |

### Closest Prior Work

1. **MIS-PO (Step 3.5 Flash)**: Proposes dual-level (token and trajectory) binary filtering with fixed bounds. Our work keeps the same loss form but makes the trajectory bound adaptive via an acceptance controller.

2. **MinPRO**: Argues that prefix importance ratios are the correct off-policy correction for autoregressive policies, and proposes a minimum-prefix-ratio surrogate to stabilize off-policy RLVR. Our method is different in that it keeps binary trajectory filtering (accept/reject) and adapts a single trajectory bound, rather than reweighting token gradients with a prefix-derived factor.

3. **Trust Region Masking (TRM)**: Proposes sequence-level masking based on the maximum token-level divergence in a trajectory to obtain non-vacuous long-horizon trust-region guarantees. Our method is also trajectory-level filtering, but targets a fixed acceptance rate rather than enforcing a fixed max-divergence threshold.

4. **M2PO**: Uses an explicit batch-level second-moment constraint by iteratively masking the largest second-moment tokens until the constraint is satisfied. Our method replaces second-moment control with acceptance-rate control; the comparison tests whether acceptance control is sufficient.

5. **BAPO**: Adapts asymmetric clipping bounds to balance positive/negative gradient contributions and preserve entropy. Our method is different in mechanism (accept/reject filtering) and in what is controlled (trajectory acceptance rather than loss contribution balance).

6. **KPO**: Smooths token-wise ratio spikes via a causal Kalman filter. Our method does not smooth ratios; it discards entire trajectories based on a learned bound.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MIS-PO | Fixed accept/reject bounds on token + trajectory ratios | Brittle bounds; can waste rollouts or admit unstable samples | Adapt only the trajectory bound to target acceptance | Keeps effective batch size stable across training; reduces tuning burden |
| MinPRO | Prefix-aware off-policy correction via minimum prefix importance ratio (soft reweighting) | Still relies on weighted gradients; may be less robust to hard engine mismatch where ratios are noisy at many tokens | Use binary trajectory filtering with an adaptive bound | If acceptance correlates with variance, binary filtering may give lower-variance updates under disaggregation |
| TRM | Masks entire sequences when max token-level divergence exceeds a fixed threshold | Requires choosing a fixed divergence threshold; may reject many trajectories for long sequences | Use acceptance-rate targeting instead of a fixed divergence threshold | Targets a stable effective batch size without requiring a calibrated KL threshold |
| M2PO | Masks tokens until batch-level second moment is below threshold | More complex; may over-mask; computes per-token M2 | Use acceptance as a simpler control signal | If acceptance correlates with variance, can match M2PO with less machinery |
| BAPO | Adapts clipping bounds to control positive-token contribution and entropy | Still continuous clipping; different failure mode focus | Adapt accept/reject bound, not clip bounds | Better suited to hard mismatches from disaggregated inference/training |
| KPO | Kalman-filters token-wise log ratios to reduce noise | Sequential filtering overhead; still uses weighted ratios | Binary trajectory mask with adaptive bound | Cheap and parallelizable; targets trajectory-level stability |
| GSPO/GMPO | Redefines ratios at sequence level | Still sensitive to drift magnitude | Add online bound adaptation | Makes sequence-level approaches more robust to changing drift |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen3-Base-1.7B | 1.7B | https://huggingface.co/Qwen/Qwen3-1.7B | Small dense model used in M2PO for math RLVR |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| DeepScaleR-Preview-Dataset | RLVR math training prompts with verifiable answers | ~40k problems | https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset | MIT (per dataset card) |

**Other Resources (if applicable):**
- Implementation starting point: M2PO codebase https://github.com/Infini-AI-Lab/M2PO/ (veRL + vLLM rollout)

**Resource Estimate**:

This proposal is intentionally scoped to a small, decisive experiment. Since published papers do not report compute for this exact variant, the estimate below is conservative and intended as a budget envelope.

- **Compute budget**: 450-600 GPU-hours total (must be \(\le 768\) GPU-hours)
  - 3 main training runs (Fixed MIS-PO, AC-MIS-PO, M2PO), each ~120-180 GPU-hours on A100-80GB
  - Optional 1 ablation run (Fixed-at-final-bound) if AC-MIS-PO wins: +60-120 GPU-hours
  - Evaluation runs on 2-3 benchmarks (Math500, AIME24, AIME25) at a few checkpoints: +30-60 GPU-hours
- **GPU memory**: Fits on 1-8× A100 80GB (Qwen3-1.7B with FSDP)
- **API usage**: None required

**Training settings (initial verification target):**

We plan to evaluate in an **asynchronous / stale-rollout** setting (matching M2PO/MinPRO’s motivation) because it is easy to implement in veRL by replaying rollouts from an older policy snapshot. Disaggregated rollout/training engines (vLLM + FSDP) are still used, so numerical engine mismatch is present, but the primary controlled mismatch knob is **staleness**.

- RL algorithm: GRPO-style actor update (as used in M2PO), with our masking variant
- Staleness: s=256 (following M2PO’s stale-256 setting), plus an s=0 sanity run if budget allows
- Steps: 300 RL update steps (initial verification), with an option to extend to 1000 steps if results are unclear and budget remains
- Rollouts: 4 responses per prompt
- Prompt batch size: 128 prompts per rollout step
- Max response length: 1024 tokens
- Rollout engine: vLLM
- Training engine: FSDP

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| Math500 | 500 competition-style math problems used to evaluate mathematical reasoning | Pass@1 (accuracy) | test | https://github.com/hendrycks/math | Use eval from M2PO repo (or reproduce their parser) |
| AIME24 | 2024 AIME contest problems (very hard short-answer math) | Pass@1 (avg@16 optional) | test | https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions | Use eval from M2PO repo |
| AIME25 | 2025 AIME contest problems (very hard short-answer math) | Pass@1 (avg@16 optional) | test | https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions | Use eval from M2PO repo |

Primary metric for the decisive experiment: Avg(Math500, AIME24, AIME25) of Pass@1 (the fraction of problems solved correctly by one sampled solution; higher is better) using a fixed decoding setting.

Secondary diagnostics (stability/efficiency):
- Trajectory acceptance rate over training
- Distribution of \(\lvert\log\rho(\tau)\rvert\) and the learned bound \(b_k\)
- Among accepted samples: estimated batch second moment \(\hat{M}_2\) (for diagnostic comparison to M2PO)
- Train reward vs eval accuracy gap (reward hacking / instability proxy)

### Main Results

#### Baseline Evidence from Prior Work

The most directly comparable published numbers we found are from M2PO’s Table 1 (math RLVR under stale rollouts). These results are not perfectly matched to our planned compute (they use 1000 training steps and their own rollout/eval configuration), but they provide a concrete strength reference for what strong off-policy stabilization achieves on the same base model.

| Method | Description | Settings | Results | Reference |
|---|---|---|---|---|
| GRPO | PPO-style clipping baseline used in M2PO | Qwen3-Base-1.7B; staleness s=0; 1000 steps; 8 responses/prompt | Avg=33.0; Math500=67.2; AIME24=7.5; AIME25=7.5 | [M2PO](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/sections/6.2%20Performance%20Comparison%20on%20Training%20with%20Staleness.md) |
| GRPO | Same baseline under extreme staleness | Qwen3-Base-1.7B; staleness s=256; 1000 steps; 8 responses/prompt | Avg=30.4; Math500=64.3; AIME24=8.5; AIME25=4.8 | [M2PO](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/sections/6.2%20Performance%20Comparison%20on%20Training%20with%20Staleness.md) |
| GSPO | Sequence-level ratio baseline evaluated in M2PO | Qwen3-Base-1.7B; staleness s=256; 1000 steps; 8 responses/prompt | Avg=30.1; Math500=65.0; AIME24=6.9; AIME25=4.0 | [M2PO](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/sections/6.2%20Performance%20Comparison%20on%20Training%20with%20Staleness.md) |
| M2PO | Second-moment masking until \(\hat{M}_2\le \tau_{M_2}\) | Qwen3-Base-1.7B; staleness s=256; 1000 steps; 8 responses/prompt; \(\tau_{M_2}=0.04\) | Avg=36.6; Math500=71.8; AIME24=14.0; AIME25=6.5 | [M2PO](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/sections/6.2%20Performance%20Comparison%20on%20Training%20with%20Staleness.md) |

#### Results Table (To Be Verified)

All methods below are run under the **same rollout budget and training steps** in our verification setup.

| Method | Base Model | Benchmark | Metric 1 | Source | Notes |
|--------|------------|-----------|----------|--------|-------|
| GRPO | Qwen3-Base-1.7B | Avg(Math500, AIME24, AIME25) | **TBD** | - | Included for completeness; expected to degrade under mismatch |
| Fixed MIS-PO | Qwen3-Base-1.7B | Avg(Math500, AIME24, AIME25) | **TBD** | - | Needs re-run; token bounds [0.5,2.0], traj bounds [0.996,1.001] from Step 3.5 Flash |
| M2PO | Qwen3-Base-1.7B | Avg(Math500, AIME24, AIME25) | **TBD** | - | Strong adaptive masking baseline; \(\tau_{M_2}=0.04\) |
| **AC-MIS-PO (ours)** | Qwen3-Base-1.7B | Avg(Math500, AIME24, AIME25) | **TBD** | - | Adaptive trajectory bound targeting acceptance |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Fixed-at-final-bound | Run MIS-PO with trajectory bound fixed to the final \(b_K\) learned by AC-MIS-PO (no per-step adaptation) | If this matches AC-MIS-PO, gains were due to discovering a better constant bound; if worse, adaptation matters |
| Token-bound adaptation (optional) | Also adapt token-level bound using the same acceptance controller | Likely smaller benefit; may risk instability due to per-token heavy tails |

### Analysis (Optional)

- Correlate acceptance rate with batch second moment \(\hat{M}_2\) across training to test whether acceptance control is a good proxy for variance control.
- Compare reward–accuracy dynamics to detect instability (reward increases while accuracy stagnates/drops).

---

## Success Criteria

**Decision rule (go / no-go):**
- **Go (success)** if AC-MIS-PO beats Fixed MIS-PO on the primary metric **and** is within **2 percentage points** of M2PO (or better) on Avg(Math500, AIME24, AIME25) under the same rollout budget, without exhibiting degenerate acceptance (e.g., trajectory acceptance collapsing near zero for long stretches).
- **No-go (refute the core hypothesis)** if AC-MIS-PO does not improve over Fixed MIS-PO, or if it improves but is clearly dominated by M2PO while showing higher \(\hat{M}_2\) among accepted samples (suggesting acceptance-rate control is not sufficient for variance control).
- **Interpretation-focused outcome** if Fixed-at-final-bound matches AC-MIS-PO: conclude that online adaptation is unnecessary and the key lesson is that tuning a better constant MIS-PO bound matters.


**Criterion 1: Acceptance control improves MIS-PO without destabilizing training**
- Hypothesis: AC-MIS-PO achieves higher final eval accuracy than Fixed MIS-PO under the same rollout budget.
- Validation: Improvement is accompanied by non-degenerate acceptance (not collapsing to ~0) and smoother ratio/acceptance trajectories.

**Criterion 2: Acceptance control is competitive with second-moment control at small scale**
- Hypothesis: AC-MIS-PO approaches M2PO’s eval accuracy while being simpler and yielding higher usable-sample rate (more accepted trajectories).
- Validation: AC-MIS-PO is not clearly dominated by M2PO on accuracy, and shows higher acceptance than fixed MIS-PO.

**Criterion 3: Adaptation (not just a better constant) explains gains**
- Hypothesis: Fixed-at-final-bound underperforms AC-MIS-PO.
- Validation: If fixed-at-final-bound matches AC-MIS-PO, the conclusion becomes “tuning bounds matters, online adaptation may be unnecessary.”

---

## Impact Statement

If acceptance-controlled MIS-PO works, practitioners training RLVR systems with disaggregated rollout/training stacks (vLLM + FSDP/Megatron) could replace manual ratio-bound tuning with an automatic controller, improving stability and reducing wasted rollouts. If it fails against M2PO, the result is still decision-changing: acceptance-rate control is not sufficient, and practitioners should prefer second-moment or stronger trust-region constraints.

---

## References

- [Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)
- [Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/meta/meta_info.txt)
- [BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping](./references/BAPO-Stabilizing-Off-Policy-Reinforcement-Learning-for-LLMs-via-Balanced-Policy-Optimization-with-Adaptive-Clipping/meta/meta_info.txt)
- [Online Causal Kalman Filtering for Stable and Effective Policy Optimization](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt)
- [Geometric-Mean Policy Optimization](./references/Geometric-Mean-Policy-Optimization/meta/meta_info.txt)
- [Group Sequence Policy Optimization](./references/Group-Sequence-Policy-Optimization/meta/meta_info.txt)
- [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](./references/Stabilizing-MoE-Reinforcement-Learning-by-Aligning-Training-and-Inference-Routers/meta/meta_info.txt)
- [Tapered Off-Policy REINFORCE (TOPR)](https://arxiv.org/abs/2503.14286)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)
- [HybridFlow](https://arxiv.org/abs/2409.19256)
- [veRL](https://github.com/volcengine/verl)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization (MinPRO)](https://arxiv.org/abs/2601.22718)
- [Trust Region Masking for Long-Horizon LLM Reinforcement Learning (TRM)](https://arxiv.org/abs/2512.23075)
- [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
- [QUATRO: Query-Adaptive Trust Region Policy Optimization](https://arxiv.org/abs/2602.04620)
- [DCPO: Dynamic Clipping Policy Optimization](https://arxiv.org/abs/2509.02333)
- [CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization](https://arxiv.org/abs/2508.07629)
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://arxiv.org/abs/2508.10687)
