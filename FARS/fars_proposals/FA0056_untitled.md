# untitled

# Innovation-Saturated KPO: Robust Kalman Updates for Token Importance Ratios in LLM RLVR

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models can be improved on tasks like mathematical reasoning by training them with feedback from automated checks — for example, verifying that a generated answer exactly matches a known correct integer. This training setting is often called **reinforcement learning with verifiable rewards (RLVR)**: instead of learning a reward model, the training loop uses a programmatic verifier that returns an objective pass/fail signal.

Many RLVR pipelines use PPO-style policy-gradient updates that reweight gradients by an **importance sampling (IS) ratio** between the current policy and the policy that generated the rollouts. In large language models, these ratios are computed per token, and they can become extremely noisy due to off-policy minibatch updates, rollout/training engine mismatch (e.g., vLLM vs training logprobs), numerical precision differences, or Mixture-of-Experts routing discontinuities.

**KPO** (Online Causal Kalman Filtering for Stable and Effective Policy Optimization) proposes to reduce IS-ratio noise by treating the token-wise log ratios as a time series and applying a **causal Kalman filter in log space** before using the ratios in the policy objective. KPO reports improved RLVR stability and higher **avg@16** (mean accuracy over 16 sampled solutions per problem) and **pass@16** (fraction of problems with at least one correct solution among 16 samples) on multiple math benchmarks compared to GRPO (Group Relative Policy Optimization), GSPO (Group Sequence Policy Optimization), and GMPO (Geometric-Mean Policy Optimization).

### The Problem

KPO introduces two Kalman noise parameters, **process noise** \(Q\) and **observation noise** \(V\), where the ratio \(Q/V\) controls how aggressively the filter smooths token log ratios. KPO’s own parameter analysis reports that when \(Q/V\) becomes large (weaker smoothing; more responsive to short-term fluctuations), training reward degrades and becomes less stable (Figure 4 in KPO). This matters in practice because the paper motivates KPO partly as a response to settings where ratios may fluctuate rapidly (e.g., train–inference mismatch and MoE routing changes), which are exactly the regimes where practitioners would want a filter to be responsive rather than heavily smoothed.

A second practical issue is that KPO’s update is based on the **innovation** (measurement residual) \(\delta_t\) between the observed log ratio and the filter’s prediction. If the observed ratios are heavy-tailed (rare but extreme spikes), then increasing \(Q\) increases the Kalman gain \(K_t\) and can cause the filter estimate to chase those spikes, potentially destabilizing training even if the downstream PPO objective clips ratios.

In classical control and signal processing, a standard remedy is to **saturate (clip) the innovation** to prevent measurement outliers from dominating the state estimate. This “innovation saturation” / Huberization idea is well-studied for robust Kalman filtering, but has not been evaluated as a stabilizer for token-wise IS-ratio filtering in LLM post-training.

### Key Insight and Hypothesis

**Key insight:** KPO’s sensitivity to \(Q/V\) may come from rare, extreme innovations \(\delta_t\) that become influential when \(Q\) is large (weak smoothing). If this is true, then a scale-aware innovation clip should (i) reduce training instabilities under weak smoothing and (ii) broaden the stable hyperparameter range of KPO.

**Hypothesis:** Under a weak-smoothing Kalman setting (e.g., \(Q/V=10^{-2}\)) where KPO’s training degrades, replacing the standard Kalman update with an **innovation-saturated** update using a fixed 3\(\sigma\) threshold (no tuning) will recover training stability and recover most of the performance gap to KPO’s default strong-smoothing setting.

This hypothesis could be wrong. KPO’s degradation at large \(Q/V\) could instead be driven by systematic bias (e.g., the random-walk state model is misspecified), not by rare outliers, in which case innovation clipping will be redundant.

---

## Proposed Approach

### Overview

We propose **Innovation-Saturated KPO (IS-KPO)**: a drop-in modification to KPO’s causal Kalman filter that clips the innovation before updating the latent log-ratio estimate. IS-KPO keeps KPO’s objective, clipping bounds, rollout generation, and evaluation protocol fixed. The only change is the Kalman update rule used to compute the filtered per-token importance ratios.

Our goal is not to claim a new robust filtering algorithm. The contribution is a **mechanism test** for LLM-RL: whether KPO’s \(Q/V\) sensitivity is outlier-driven, and whether a standard robustification yields a practically more robust RLVR stabilizer.

### Method Details

#### Background: KPO’s token-wise ratio filtering
For prompt \(x\) and response tokens \(y_{1:T}\), define the token-wise importance ratio
\[
 r_t = \frac{\pi_{\theta}(y_t\mid x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t\mid x,y_{<t})},\qquad z_t = \log r_t.
\]
KPO models \(z_t\) with a 1D random-walk state space model in log space:
\[
\rho_t = \rho_{t-1} + \eta_t,\ \eta_t\sim\mathcal{N}(0,Q),\qquad z_t = \rho_t + \epsilon_t,\ \epsilon_t\sim\mathcal{N}(0,V).
\]
Let \((\rho_{t\mid t-1}, P_{t\mid t-1})\) be the predicted mean/variance given past observations, and \((\rho_{t\mid t}, P_{t\mid t})\) the posterior after observing \(z_t\). The standard (scalar) Kalman recursion is:
\[
\delta_t = z_t - \rho_{t\mid t-1},\qquad K_t = \frac{P_{t\mid t-1}}{P_{t\mid t-1}+V},
\]
\[
\rho_{t\mid t} = \rho_{t\mid t-1} + K_t\,\delta_t,\qquad P_{t\mid t}=(1-K_t)P_{t\mid t-1}.
\]
KPO outputs the filtered ratio \(\hat r_t = \exp(\rho_{t\mid t})\), and uses \(\hat r_t\) in a GRPO-style clipped objective.

#### Innovation-saturated update (IS-KPO)
IS-KPO replaces \(\delta_t\) with a clipped innovation \(\tilde\delta_t\) in a scale-aware way:
\[
\sigma_t = \sqrt{P_{t\mid t-1}+V},\qquad \tilde\delta_t = \mathrm{clip}(\delta_t, -\kappa\sigma_t, +\kappa\sigma_t).
\]
We fix \(\kappa=3\) (a 3\(\sigma\) truncation rule) for all runs, with no per-task tuning.

The posterior update becomes:
\[
\rho_{t\mid t} = \rho_{t\mid t-1} + K_t\,\tilde\delta_t,\qquad P_{t\mid t}=(1-K_t)P_{t\mid t-1}.
\]
All other parts of KPO remain unchanged.

**Interpretation.** This update enforces approximately Gaussian tails on the normalized innovations \(u_t=\delta_t/\sigma_t\) by truncating large residuals. If KPO’s weak-smoothing failure is driven by rare spikes in \(u_t\), IS-KPO should reduce their influence without requiring new learned components.

### Key Innovations

1. **Robustification at the ratio-filter level (not the loss level)**: clip the Kalman innovation before exponentiation and PPO-style clipping, targeting a distinct failure mode from standard ratio clipping.
2. **A falsifiable mechanism test for KPO’s \(Q/V\) sensitivity**: treat heavy-tailed innovations as an explicit, measurable premise, with a pre-registered “kill criterion.”
3. **Hyperparameter robustness goal**: rather than beating KPO at its tuned setting, aim to make KPO less brittle when \(Q/V\) is increased (a regime KPO’s own analysis suggests is problematic).

---

## Related Work

### Field Overview

Stability in LLM post-training has become a central concern because modern RL stacks are often off-policy in practice (minibatch reuse, asynchronous rollouts, inference/training engine mismatch). PPO-style clipping is widely used, but it can be brittle: it trades variance for bias and can suppress high-entropy tokens that drive learning under distribution shift.

A recent line of work addresses instability by changing how importance ratios are aggregated (token-level vs sequence-level), how they are clipped/masked (soft vs hard constraints), or by adding structure-aware smoothing to the ratio or advantage signals. KPO fits into this line by imposing temporal coherence over token-wise log ratios.

Separately, robust Kalman filtering has a long literature on handling heavy-tailed or outlier-corrupted observations using M-estimation, saturation, or Student-t models. Our proposal borrows a minimal robustification (innovation saturation) and tests whether it is decision-relevant in LLM RLVR.

### Related Papers

- **[Online Causal Kalman Filtering for Stable and Effective Policy Optimization](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt)**: Filters token-wise log importance ratios with a causal Kalman filter to improve RLVR stability.
- **[Group Sequence Policy Optimization](./references/Group-Sequence-Policy-Optimization/meta/meta_info.txt)**: Uses a sequence-level importance ratio shared across tokens to reduce variance and stabilize GRPO.
- **[Geometric-Mean Policy Optimization](./references/Geometric-Mean-Policy-Optimization/meta/meta_info.txt)**: Replaces arithmetic aggregation with a geometric-mean objective to reduce sensitivity to outlier token contributions.
- **[Kalman Filter Enhanced GRPO for Reinforcement Learning-Based Language Model Reasoning](./references/Kalman-Filter-Enhanced-GRPO-for-Reinforcement-Learning-Based-Language-Model-Reasoning/meta/meta_info.txt)**: Uses Kalman filtering to stabilize reward baselines / advantages in GRPO rather than filtering IS ratios.
- **[Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/meta/meta_info.txt)**: Proposes M2PO, a second-moment trust region for extreme staleness that masks high-variance ratio contributions.
- **[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](./references/DAPO-An-Open-Source-LLM-Reinforcement-Learning-System-at-Scale/meta/meta_info.txt)**: Open RLVR system that addresses entropy collapse and instability via decoupled clipping, dynamic sampling, and length shaping.
- **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)**: Introduces PPO and the clipped surrogate objective.
- **[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)**: Introduces KL-constrained trust region optimization for policy gradients.
- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)**: Introduces GRPO-style reasoning RL and benchmarks.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Large-scale RL for reasoning; popularizes GRPO-style post-training.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Early RLVR framing with verifiable intermediate signals.
- **[RLOO](https://arxiv.org/abs/2402.14740)**: On-policy RLHF variant emphasizing variance reduction with leave-one-out baselines.
- **[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)**: Replaces RLHF with a supervised objective; relevant as an alternative post-training paradigm.
- **[Soft Adaptive Policy Optimization (SAPO)](https://arxiv.org/abs/2511.20347)**: Uses soft gating / clipping strategies to reduce ratio-induced instability.
- **[ASPO: Asymmetric Importance Sampling Policy Optimization](https://arxiv.org/abs/2510.06062)**: Uses asymmetric handling of risky ratio regimes to stabilize updates.
- **[DCPO: Dynamic Clipping Policy Optimization](https://arxiv.org/abs/2509.02333)**: Dynamically adapts clipping behavior during training to improve stability.
- **[CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization](https://arxiv.org/abs/2508.07629)**: Modifies clipping to better preserve entropy and learning signal.
- **[Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning](https://arxiv.org/abs/2512.05591)**: Uses alternative clipping constraints to stabilize updates.
- **[A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization (MinPRO)](https://arxiv.org/abs/2601.22718)**: Uses prefix-level ratios to stabilize off-policy correction for long sequences.
- **[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://arxiv.org/abs/2508.10687)**: Diagnoses train/inference mismatch as a source of off-policy ratio noise.
- **[Robustifying the Kalman Filter against Measurement Outliers: An Innovation Saturation Mechanism](https://www.merl.com/publications/docs/TR2018-173.pdf)**: Classical innovation saturation for robust Kalman filtering.
- **[Iteratively Saturated Kalman Filtering](https://arxiv.org/abs/2507.00272)**: Extends saturation-based robustness for both measurement and process outliers.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Sequence-level ratio aggregation | Replace token-wise ratios with a single sequence-level ratio | GSPO, GMPO | Math RLVR (AIME/MATH500), code RLVR | Loses within-sequence heterogeneity; can over-smooth learning signal |
| Trust region via masking/clipping variants | Change clip/mask rule to control variance/bias trade-off | PPO, DCPO, CE-GPPO, M2PO, MinPRO | RLVR/RLHF stability under mismatch/staleness | Often introduces new hyperparameters; may suppress high-entropy tokens |
| Filtering / smoothing of learning signals | Treat ratios or baselines as noisy signals and smooth them | KPO, KRPO | Math RLVR | Sensitivity to filter hyperparameters; unclear robustness to heavy tails |
| Robust filtering (control literature) | Limit influence of outliers via saturation/M-estimation | Innovation saturation KF, ISKF | State estimation tasks | Not tested in LLM RL; assumptions may not match ratio noise |

### Closest Prior Work

**KPO** introduces Kalman filtering of token-wise log ratios and shows strong gains under a tuned low \(Q/V\) regime. Our work keeps KPO’s structure but changes the update rule to test whether KPO’s instability at larger \(Q/V\) is outlier-driven.

**KRPO** applies Kalman filtering to stabilize reward baselines/advantages in GRPO. Our proposal differs in *what* is filtered (token importance ratios rather than reward baselines) and targets a different failure mode (ratio noise under weak smoothing).

**M2PO** stabilizes off-policy RL with stale data via second-moment masking of high-variance ratio contributions. Our proposal is complementary: it aims to stabilize the *ratio estimator itself* at the token level, rather than masking tokens after ratios are computed.

**Robust Kalman filtering via innovation saturation** is classical, but has not been evaluated for token-wise IS ratio filtering in LLM RLVR. The novelty is empirical and mechanism-oriented: whether this robustification changes stability/performance under KPO-like training.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| KPO | Kalman filters token log ratios | Degrades at large \(Q/V\) (weak smoothing) | Clip the innovation using \(\kappa=3\) | Prevent rare spikes from dominating when \(Q\) is large |
| KRPO | Kalman filters reward baseline | Does not address ratio noise | Apply robustness to ratio filtering | Targets a distinct instability source |
| GSPO/GMPO | Use sequence-level ratio / geometric mean | Removes token heterogeneity | Keep token-wise ratios but robustify | Preserve fine-grained credit assignment while stabilizing |
| M2PO | Mask tokens to bound second moment | Discards data; extra masking loop | Robustify ratio estimate before objective | Reduce the need to discard high-signal tokens |
| Innovation-saturated KF (control) | Robust estimation under outliers | Not tested for LLM-RL ratio noise | Apply as minimal modification to KPO | Lightweight and scale-aware in KPO setting |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3-4B-Base | 4B | https://huggingface.co/Qwen/Qwen3-4B-Base | Matches KPO’s reported backbone |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| DAPO-Math-17k (or processed variant) | RLVR prompts with verifiable answers | ~17k prompts | https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k and https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed | Apache-2.0 (per HF card) |

**Implementation notes (match KPO as closely as possible):**
- Group size \(G=8\)
- Training batch size 32; minibatch size 8 (4 minibatches per batch; 1 on-policy + 3 off-policy)
- Max response length 4096
- Actor learning rate 1e-6
- Reward: binary exact-match verifier (1 success / 0 failure)
- KPO objective: replace GRPO token ratios with filtered ratios and apply ratio clipping
- Use KPO’s tight ratio clip bounds for the clipped objective: \(\epsilon_-=0.0003\), \(\epsilon_+=0.0004\)

**Methods compared (main 3 conditions):**
- **(A) KPO-strong**: KPO with \(Q/V=10^{-6}\) (default strong smoothing; matches KPO Table 1 setting for KPO-clipped) and no innovation clipping.
- **(B) KPO-weak**: KPO with \(Q/V=10^{-2}\) (weak smoothing; KPO’s parameter analysis reports degradation) and no innovation clipping.
- **(C) IS-KPO-weak (ours)**: Same as (B), but with innovation saturation \(\kappa=3\) as defined above.

**Experiment 0 (premise check; automated):**
Before running the full budget, run a short pilot of (B) for a small number of updates (e.g., 50–100) and log the normalized innovation \(u_t=\delta_t/\sqrt{P_{t\mid t-1}+V}\) and PPO clip fraction.
- If the tail rate of \(|u_t|>3\) is < 0.5% **and** clip fraction is not elevated relative to (A), treat the outlier-driven premise as unsupported and stop early (report as negative result).

**Resource Estimate**:
- **Compute budget**: Target 3 full runs (A/B/C) on 4×A100 80GB for ~24 hours each (estimate) ≈ 288 GPU-hours, plus pilot + evaluation (<150 GPU-hours). Total target < 512 GPU-hours. If wall-clock per run exceeds expectations, fall back to Qwen3-1.7B or Qwen2.5-Math-1.5B with the same protocol.
- **GPU memory**: 4B PPO-style training should fit on 4×A100 with FSDP/ZeRO; rollouts can be generated with vLLM on the same GPUs or one extra GPU.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AIME’24 | 30 contest math problems with integer answers | avg@16, pass@16 | test | https://huggingface.co/datasets/Maxwell-Jia/AIME_2024 | Custom exact-match evaluator |
| AIME’25 | 30 contest math problems with integer answers | avg@16, pass@16 | test | https://huggingface.co/datasets/MathArena/aime_2025 | Custom exact-match evaluator |
| MATH-500 | 500 problems from MATH benchmark | avg@16, pass@16 | test | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | Custom exact-match evaluator |

**Primary metric for decision rule:** avg@16 on MATH-500 (larger sample size than AIME).

### Main Results

#### Results Table

Published results below are from KPO’s Table 1 (Qwen3-4B, 16 samples per problem). Verification should re-run all methods under the same budget for apples-to-apples comparison.

| Method | Base Model | Benchmark | avg@16 | pass@16 | Source | Notes |
|---|---|---|---:|---:|---|---|
| GRPO | Qwen3-4B | AIME’24 | 27.29 | 53.33 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| GSPO | Qwen3-4B | AIME’24 | 32.70 | 60.00 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| GMPO | Qwen3-4B | AIME’24 | 30.83 | 50.00 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| KPO-strong (A) | Qwen3-4B | AIME’24 | 37.91 | 63.33 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline for KPO-clipped |
| KPO-weak (B) | Qwen3-4B | AIME’24 | **TBD** | **TBD** | - | Needs re-run (Q/V=1e-2 not reported as final eval) |
| **IS-KPO-weak (C)** | Qwen3-4B | AIME’24 | **TBD** | **TBD** | - | To be verified |
| KPO-strong (A) | Qwen3-4B | MATH-500 | 89.42 | 94.80 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline for KPO-clipped |
| KPO-weak (B) | Qwen3-4B | MATH-500 | **TBD** | **TBD** | - | To be verified |
| **IS-KPO-weak (C)** | Qwen3-4B | MATH-500 | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| IS-KPO-strong (optional) | Apply innovation saturation with \(\kappa=3\) to \(Q/V=10^{-6}\) setting | Should be similar to KPO-strong if clipping only matters under weak smoothing |

### Analysis (Optional)

- **Innovation heavy-tail diagnostic**: report tail rate of \(|u_t|>3\), kurtosis, and top-1% innovation magnitude for (A) vs (B) vs (C).
- **Mechanism logging**: compare PPO clip fraction trajectories; success should correlate with reduced clip spikes in (C) relative to (B).

---

## Success Criteria

**Criterion 0 (premise check): Heavy-tailed innovations exist under weak smoothing**
- Hypothesis: Under (B), normalized innovations \(u_t\) have non-trivial heavy tails and coincide with elevated PPO clip fraction.
- Validation: Tail rate \(|u_t|>3\) is ≥ 0.5% and clip fraction shows clear spikes vs (A). If not, conclude the outlier-driven premise is unsupported and stop.

**Criterion 1: IS-KPO restores stability under weak smoothing**
- Hypothesis: (C) avoids mid-training degradation observed in (B), with smoother reward/entropy/clip-fraction trajectories.
- Validation: (C) shows no reward crash or entropy collapse while (B) does (or (C) materially reduces clip-fraction spikes).

**Criterion 2: IS-KPO recovers performance under weak smoothing**
- Hypothesis: On MATH-500 avg@16, (C) closes most of the gap between (A) and (B).
- Validation: If (B) is ≥2 avg@16 points worse than (A), then (C) recovers ≥50% of that gap and is within ~1 avg@16 point of (A) on MATH-500 (AIME’24/’25 as secondary confirmation).

---

## Impact Statement

If innovation saturation improves KPO’s robustness at higher \(Q/V\), practitioners can use token-wise ratio filtering in noisier off-policy regimes (engine mismatch, MoE routing changes, more aggressive minibatch reuse) with less hyperparameter brittleness. If it fails, the result is still decision-relevant: it suggests KPO’s \(Q/V\) sensitivity is not driven by rare innovation spikes, and future stabilizers should focus on correcting systematic mismatch or redefining trust regions rather than robust filtering.

---

## References

- [Online Causal Kalman Filtering for Stable and Effective Policy Optimization](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) - He et al., 2026
- [Group Sequence Policy Optimization](./references/Group-Sequence-Policy-Optimization/meta/meta_info.txt) - Zheng et al., 2025
- [Geometric-Mean Policy Optimization](./references/Geometric-Mean-Policy-Optimization/meta/meta_info.txt) - Zhao et al., 2025
- [Kalman Filter Enhanced GRPO for Reinforcement Learning-Based Language Model Reasoning](./references/Kalman-Filter-Enhanced-GRPO-for-Reinforcement-Learning-Based-Language-Model-Reasoning/meta/meta_info.txt) - Wang et al., 2025
- [Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/meta/meta_info.txt) - Zheng et al., 2025
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](./references/DAPO-An-Open-Source-LLM-Reinforcement-Learning-System-at-Scale/meta/meta_info.txt) - Yu et al., 2025
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) - Schulman et al., 2015
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - DeepSeek-AI et al., 2025
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [RLOO](https://arxiv.org/abs/2402.14740) - Ahmadian et al., 2024
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [Soft Adaptive Policy Optimization (SAPO)](https://arxiv.org/abs/2511.20347) - 2025
- [ASPO: Asymmetric Importance Sampling Policy Optimization](https://arxiv.org/abs/2510.06062) - 2025
- [DCPO: Dynamic Clipping Policy Optimization](https://arxiv.org/abs/2509.02333) - 2025
- [CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization](https://arxiv.org/abs/2508.07629) - 2025
- [Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning](https://arxiv.org/abs/2512.05591) - 2025
- [A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization (MinPRO)](https://arxiv.org/abs/2601.22718) - 2026
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://arxiv.org/abs/2508.10687) - 2025
- [Robustifying the Kalman Filter against Measurement Outliers: An Innovation Saturation Mechanism](https://www.merl.com/publications/docs/TR2018-173.pdf) - Fang et al., 2018
- [Iteratively Saturated Kalman Filtering](https://arxiv.org/abs/2507.00272) - Yang and Boyd, 2025
