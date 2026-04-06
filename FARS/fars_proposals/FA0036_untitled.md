# untitled

# Fixed-Gain Exponential Smoothing of Token Importance Ratios: A Simplified Alternative to KPO

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Reinforcement learning with verifiable rewards (RLVR) improves large language models (LLMs) using feedback from automated checkers (e.g., exact-match answers for math problems or unit tests for code). This avoids training a separate reward model, but it places heavy demands on the stability of the policy optimization algorithm because rewards are often sparse (pass/fail) and rollouts can be long.

Most modern RLVR pipelines for LLMs use a PPO-style objective that reweights gradients by an importance sampling (IS) ratio between the current policy and the policy that generated the rollouts. In long-horizon LLM generation, these IS ratios can become noisy and can cause training instabilities (entropy collapse, reward crashes) when rollouts are stale (off-policy) or when the rollout engine and training engine produce mismatched token probabilities.

A recent paper, **KPO** (Online Causal Kalman Filtering for Stable and Effective Policy Optimization), proposes to stabilize token-wise IS ratios by applying a causal Kalman filter in log space before using the ratios in a GRPO-style objective. KPO reports large gains on math reasoning benchmarks over strong baselines such as GSPO and GMPO.

### The Problem

KPO is motivated as an “uncertainty-aware” method: the Kalman gain is described as an adaptive step size balancing the prior prediction against the current observed ratio. However, the KPO instantiation used in its main results uses **fixed** process noise (Q) and observation noise (V), and a **one-dimensional** random-walk state-space model.

In this specific setting, the Kalman covariance recursion does not depend on the observation residuals, and therefore the Kalman gain schedule depends only on token index (and initialization), not on the observed ratio sequence. This raises a practical and scientific question:

- Are KPO’s gains driven by genuinely Kalman-specific behavior, or primarily by applying a strong low-pass filter to token-wise log-IS ratios?

This matters because practitioners may adopt Kalman filtering as an additional component in already complex RLVR stacks. If a simple exponential smoother matches KPO under matched training compute, it becomes a simpler default baseline for future work and clarifies the mechanism behind KPO’s reported improvements.

### Key Insight and Hypothesis

**Key insight:** In KPO’s 1D random-walk Kalman filter with fixed Q and V, the posterior mean update has the form

\[\rho_{t\mid t} = (1-K_t)\,\rho_{t\mid t-1} + K_t\,z_t\]\

where \(z_t = \log r_t\) is the observed token log-IS ratio and \(K_t\in(0,1)\) is the Kalman gain. This is exactly an exponential smoothing update with (potentially time-varying) smoothing coefficient \(K_t\). With fixed Q,V, \(K_t\) is deterministic and quickly approaches a steady-state constant \(K_\infty\) that depends only on \(\lambda = Q/V\).

**Hypothesis:** Replacing KPO’s Kalman filter with a constant-gain exponential smoother on token log-IS ratios (setting \(\alpha = K_\infty\) computed from KPO’s \(Q,V\)) yields similar RLVR training stability and benchmark performance to KPO. We operationalize “similar” as: EMA-KPO’s avg@16 is within **1.0 point** of KPO-clipped on **each** of AIME’24, AIME’25, and MATH-500 under matched training compute (and with no systematic degradation in stability metrics such as entropy collapse or reward crashes).

---

## Proposed Approach

### Overview

We propose **EMA-KPO**: a drop-in replacement for KPO’s causal Kalman filter that performs **fixed-gain exponential smoothing** on token log-IS ratios, using a single gain parameter \(\alpha\) computed from KPO’s reported \(Q,V\) (no tuning against KPO diagnostics).

EMA-KPO keeps every other part of the training recipe identical to KPO (same rollout policy, same objective form, same clipping bounds, same decoding and evaluation protocol). The only change is the smoothing rule that produces the per-token filtered ratio \(\hat r_t\).

### Method Details

#### Background: token-wise importance ratios

For a generated response \(y=[y_1,\dots,y_T]\) to prompt \(x\), PPO/GRPO-style objectives use a token-wise ratio

\[r_t = \frac{\pi_{\theta}(y_t\mid x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t\mid x,y_{<t})}\]

to correct for the mismatch between the current policy \(\pi_\theta\) and the rollout policy \(\pi_{\theta_{\text{old}}}\).

KPO defines \(z_t = \log r_t\) and filters \(z_{1:T}\) with a 1D Kalman filter to obtain a smoothed latent \(\rho_{t\mid t}\), then exponentiates to obtain \(\hat r_t = \exp(\rho_{t\mid t})\).

#### EMA-KPO filtering rule

EMA-KPO replaces the Kalman recursion with a fixed-gain exponential smoother:

- Initialize \(m_0=0\)
- For \(t=1..T\):
  \[m_t = (1-\alpha)m_{t-1} + \alpha z_t\]
- Output filtered ratios: \(\hat r_t = \exp(m_t)\)

We set \(\alpha\) **without tuning** by mapping from KPO’s \(Q,V\). For the 1D random-walk Kalman filter used by KPO, the steady-state covariance satisfies \(P^2 + QP - VQ = 0\), which yields a steady-state gain

\[K_\infty = \frac{P+Q}{P+Q+V},\quad P = \frac{-Q + \sqrt{Q^2 + 4VQ}}{2}.\]

Therefore \(\alpha\) is set to \(\alpha = K_\infty\). Under KPO’s main-results setting \(Q=10^{-6}, V=1\), this gives \(\alpha\approx 10^{-3}\).

#### Objective (same as KPO)

EMA-KPO uses the same clipped KPO objective, replacing KPO’s Kalman-filtered \(\hat r_t\) with the EMA-filtered \(\hat r_t\):

\[J = \mathbb{E}\left[\frac{1}{GT}\sum_{i=1}^G\sum_{t=1}^T \min(\hat r_{i,t}A_{i,t},\ \mathrm{clip}(\hat r_{i,t},1-\epsilon_-,1+\epsilon_+)A_{i,t})\right].\]

### Key Innovations

- **Algorithmic clarification of KPO’s mechanism**: highlights that with fixed \(Q,V\) the Kalman gain schedule does not depend on observations, so KPO is a deterministic exponential smoother in log space.
- **A simpler baseline (EMA-KPO)**: replaces KPO’s Kalman filtering with a fixed-gain EMA whose parameter is derived directly from KPO’s reported hyperparameters.
- **A decisive equivalence-style experiment**: isolates whether KPO’s performance depends on Kalman-specific behavior or only on low-pass smoothing strength.

---

## Related Work

### Field Overview

Stabilizing policy optimization for LLMs has become a central issue in RLVR and RLHF because modern training pipelines are frequently off-policy in practice (rollout staleness, asynchronous training, and rollout/training engine mismatch). Recent work has proposed stabilizers at multiple levels: (i) changing how importance ratios are aggregated (sequence-level vs token-level), (ii) reshaping/clipping ratios more smoothly, (iii) using prefix-aware off-policy corrections, and (iv) explicitly correcting for system-induced off-policy gaps.

KPO is part of a recent line of work that treats token-wise ratios as a signal with structure (e.g., coherence across neighboring tokens) and attempts to reduce high-frequency noise while retaining token heterogeneity. Our proposal contributes a simplification and mechanistic test: whether KPO’s specific Kalman machinery is necessary beyond its effective smoothing strength.

### Related Papers

- **[Online Causal Kalman Filtering for Stable and Effective Policy Optimization](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt)**: Proposes Kalman filtering of token log-IS ratios (KPO) and reports large RLVR gains on math benchmarks.
- **[Group Sequence Policy Optimization](./references/Group-Sequence-Policy-Optimization/meta/meta_info.txt)**: Stabilizes GRPO by using sequence-level (length-normalized) importance ratios and sequence-level clipping.
- **[Geometric-Mean Policy Optimization](./references/Geometric-Mean-Policy-Optimization/meta/meta_info.txt)**: Uses geometric-mean aggregation to reduce sensitivity to outlier token ratios while keeping token-level structure.
- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)**: Introduces GRPO in math reasoning and popularizes group-based RLVR for LLMs.
- **[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)**: Demonstrates large reasoning improvements from RLVR and motivates stable large-scale RL post-training.
- **[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)**: Open-sources RLVR recipes and datasets; a standard training corpus used by KPO.
- **[A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization (MinPRO)](https://arxiv.org/abs/2601.22718)**: Argues token ratios are theoretically incorrect off-policy and proposes a prefix-ratio surrogate for stability.
- **[Soft Adaptive Policy Optimization (SAPO)](https://arxiv.org/abs/2511.20347)**: Replaces hard clipping with smooth gates to downweight off-policy tokens without discarding full sequences.
- **[ASPO: Asymmetric Importance Sampling Policy Optimization](https://arxiv.org/abs/2510.06062)**: Modifies how token ratios weight gradients (asymmetric treatment by advantage sign) to prevent entropy collapse.
- **[Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning](https://arxiv.org/abs/2512.05591)**: Adds a global entropy-ratio trust-region constraint complementary to PPO-style ratio clipping.
- **[VESPO: Variational Sequence-Level Soft Policy Optimization](https://arxiv.org/abs/2602.10693)**: Derives a smooth importance weight reshaping kernel to stabilize sequence-level off-policy RL.
- **[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://arxiv.org/abs/2508.10687)**: Shows inference/training engine mismatch induces off-policy gaps and proposes truncated importance sampling.
- **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)**: Canonical clipped policy optimization objective underlying PPO/GRPO variants.
- **[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)**: Foundational trust-region view of policy updates, motivating ratio constraints.
- **[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)**: Popular non-RL alternative to PPO-style training, useful as background for alignment method choices.
- **[Scaling Laws for Reward Model Overoptimization in RLHF](https://arxiv.org/abs/2406.04873)**: Highlights instability and reward hacking issues in RLHF, motivating verifiable rewards.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Shows the value of verification signals in reasoning, motivating RLVR pipelines.
- **[RLOO: A Memory-Efficient, Variance-Reduced REINFORCE Baseline for RLHF](https://arxiv.org/abs/2402.14740)**: Demonstrates that simpler policy-gradient methods can match PPO in some RLHF settings.
- **[AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)**: Studies asynchronous RL for reasoning, where staleness makes off-policy correction crucial.
- **[HybridFlow](https://arxiv.org/abs/2409.19256)**: Example of hybrid inference/training pipelines where system mismatch can induce off-policy behavior.
- **[Step 3.5 Flash](https://arxiv.org/abs/2602.10604)**: Introduces MIS-PO, another discrete filtering approach for stabilizing off-policy RL.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Sequence-level ratio stabilizers | Replace token ratios with a single (length-normalized) sequence ratio | GSPO, GMPO | AIME, AMC, MATH500, code benchmarks | Can discard within-sequence heterogeneity |
| Soft reshaping / clipping | Replace hard clipping with smooth gates or kernels | SAPO, VESPO, ERC | Math + code RLVR | Extra hyperparameters; mechanism less interpretable |
| Prefix-aware off-policy correction | Use prefix-level correction surrogates for off-policy lag | MinPRO | Math RLVR | May downweight too aggressively in near on-policy regimes |
| Temporal smoothing of token ratios | Treat token ratios as a structured time series and filter | KPO (Kalman), EMA-KPO (this work) | Math RLVR | Needs validation that smoothing is not over-regularizing |
| System-induced off-policy correction | Correct rollout/training mismatch explicitly | TIS / off-policy gap analyses | RLVR systems | Additional ratio estimation and clipping choices |

### Closest Prior Work

**KPO** is the closest work: it proposes Kalman filtering of token-wise log ratios and reports strong gains, but it does not compare against simple exponential smoothers and does not emphasize that with fixed \(Q,V\) the gain schedule is observation-independent.

**GSPO/GMPO** are strong stabilizers that replace token ratios with sequence-level surrogates; they reduce variance but discard within-sequence heterogeneity that KPO (and EMA-KPO) preserve.

**MinPRO** argues that prefix ratios are the theoretically correct off-policy correction and proposes a stable surrogate; this is complementary to temporal smoothing because both aim to prevent spurious token-level updates under off-policy drift.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| KPO | Kalman-filters token log ratios then exponentiates | Added machinery; unclear if benefits require Kalman-specific behavior | Replace Kalman recursion with fixed-gain EMA derived from \(Q,V\) | If KPO is mainly low-pass smoothing, EMA should match at lower complexity |
| GSPO | Uses length-normalized sequence-level ratio | Loses token heterogeneity | Keep token-wise ratios but smooth temporally | Should retain fine-grained credit assignment without high-frequency noise |
| GMPO | Uses geometric-mean aggregation to reduce outliers | Still not temporal; different objective | Keep KPO objective, only change smoothing rule | Cleaner isolation of “Kalman vs smoothing” |
| MinPRO | Prefix-aware off-policy surrogate | Different mechanism; may be conservative | Orthogonal: only smooth token ratios | If EMA matches KPO, indicates temporal smoothing is sufficient in KPO regime |
| SAPO / VESPO | Soft gates / kernels for ratio stabilization | More hyperparameters; different estimator bias | A single fixed \(\alpha\) from KPO hyperparams | Lower degree of freedom and clearer mechanistic test |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen3-4B-Base | 4B | https://huggingface.co/Qwen/Qwen3-4B-Base | Matches KPO’s reported backbone |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| DAPO-Math-17k (or processed deduplicated variant) | RLVR prompts with verifiable answers | ~17k prompts | https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k and https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed | Apache-2.0 (per HF card) |

**Implementation notes (match KPO as closely as possible):**
- Group size \(G=8\)
- Training batch size 32; minibatch size 8
- Max response length 4096
- Actor learning rate 1e-6
- Reward: binary exact-match verifier (1 success / 0 failure)

**Resource Estimate**:
- **Compute budget**: 3 training runs (GRPO, KPO-clipped, EMA-KPO) on **4×A100 80GB** for **~16 hours each** (estimate) ≈ **192 GPU-hours**, plus evaluation runs (<50 GPU-hours). Total target < **256 GPU-hours**. Run **3 random seeds** per condition if budget allows; if not, run 1 seed for the main comparison and use short-run variance estimates (e.g., 50-step pilot runs) to decide whether a second/third seed is needed.
- **GPU memory**: 4B model with PPO-style training should fit on 4×A100 with FSDP/ZeRO; rollouts can be generated with vLLM on 1 additional GPU if needed (or colocated with training).

(If this estimate is violated in pilot runs, fall back to Qwen2.5-Math-1.5B-Instruct with the same protocol; the algorithmic comparison is model-agnostic.)

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| AIME’24 | 30 contest math problems with integer answers | avg@16, pass@16 | test | https://huggingface.co/datasets/Maxwell-Jia/AIME_2024 | Custom exact-match evaluator |
| AIME’25 | 30 contest math problems with integer answers | avg@16, pass@16 | test | https://huggingface.co/datasets/MathArena/aime_2025 | Custom exact-match evaluator |
| MATH-500 | 500 problems from MATH benchmark | avg@16, pass@16 | test | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | Custom exact-match evaluator |

### Main Results

#### Results Table

Numbers below are copied from KPO’s Table 1 text (Qwen3-4B, 16 samples per problem) for reference; verification should re-run baselines under the same training budget used for EMA-KPO.

| Method | Base Model | Benchmark | avg@16 | pass@16 | Source | Notes |
|--------|------------|-----------|--------|---------|--------|-------|
| GRPO | Qwen3-4B | AIME’24 | 27.29 | 53.33 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| GSPO | Qwen3-4B | AIME’24 | 32.70 | 60.00 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| KPO-clipped | Qwen3-4B | AIME’24 | 37.91 | 63.33 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| **EMA-KPO (ours)** | Qwen3-4B | AIME’24 | **TBD** | **TBD** | - | To be verified |
| GRPO | Qwen3-4B | AIME’25 | 23.12 | 43.33 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| GSPO | Qwen3-4B | AIME’25 | 29.16 | 50.00 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| KPO-clipped | Qwen3-4B | AIME’25 | 36.87 | 60.00 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| **EMA-KPO (ours)** | Qwen3-4B | AIME’25 | **TBD** | **TBD** | - | To be verified |
| GRPO | Qwen3-4B | MATH-500 | 85.66 | 92.80 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| GSPO | Qwen3-4B | MATH-500 | 87.41 | 94.00 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| KPO-clipped | Qwen3-4B | MATH-500 | 89.42 | 94.80 | [KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) | Published baseline |
| **EMA-KPO (ours)** | Qwen3-4B | MATH-500 | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| EMA-KPO (fixed \(\alpha\)) | \(\alpha = K_\infty(Q,V)\) | Matches KPO if Kalman-specific behavior is unnecessary |
| EMA-KPO (\(\alpha\times 10\), optional) | Use \(10\times K_\infty\) | Over-smoothing/under-smoothing should degrade, showing sensitivity to smoothing strength |

### Analysis (Optional)

- **Gain schedule audit**: Log \(K_t\) for KPO across token positions; verify that after a short burn-in it concentrates near a constant \(K_\infty\).
- **Filter equivalence metrics**: On held-out rollouts, compute MSE between KPO’s \(\rho_{t\mid t}\) and EMA-KPO’s \(m_t\), and compare frequency-domain statistics (e.g., low-frequency ratio) of \(\hat r_t\) sequences.

---

## Success Criteria

**Criterion 1: EMA-KPO is non-inferior to KPO on benchmark performance under matched training compute**
- Hypothesis: EMA-KPO matches KPO-clipped within **1.0 avg@16 point** (and similarly for pass@16) on each of AIME’24, AIME’25, and MATH-500.
- Validation: Under matched training compute (and ideally 3 seeds), EMA-KPO is within the threshold on all three benchmarks and shows no systematic stability regressions.

**Criterion 2: EMA-KPO preserves KPO’s stability signatures**
- Hypothesis: EMA-KPO avoids collapse modes observed in GRPO (entropy collapse, reward crash) and yields similar entropy/clip-fraction trajectories as KPO.
- Validation: Training curves remain stable through the full training budget and are qualitatively similar to KPO.

**Criterion 3: If EMA-KPO fails, the gap localizes to early-token dynamics**
- Hypothesis: Any performance gap between EMA-KPO and KPO is explained by differences in early-token weighting (Kalman warmup), not by long-range adaptation.
- Validation: KPO–EMA differences in \(\hat r_t\) concentrate in early tokens; later-token differences are small.

---

## Impact Statement

If EMA-KPO matches KPO, practitioners implementing RLVR systems can replace Kalman filtering with a simpler fixed-gain exponential smoother while preserving performance, reducing implementation complexity and clarifying that KPO’s benefit is primarily low-pass smoothing. If EMA-KPO underperforms, the result is still decision-changing: it implies that KPO’s specific filtering dynamics (e.g., warmup behavior) are necessary, and future stabilizers should focus on modeling time-varying ratio trust regions rather than fixed smoothers.

---

## References

- [Online Causal Kalman Filtering for Stable and Effective Policy Optimization](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) - He et al., 2026
- [Group Sequence Policy Optimization](./references/Group-Sequence-Policy-Optimization/meta/meta_info.txt) - Zheng et al., 2025
- [Geometric-Mean Policy Optimization](./references/Geometric-Mean-Policy-Optimization/meta/meta_info.txt) - Zhao et al., 2025
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - DeepSeek-AI et al., 2025
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) - Yu et al., 2025
- [A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization (MinPRO)](https://arxiv.org/abs/2601.22718) - 2026
- [Soft Adaptive Policy Optimization (SAPO)](https://arxiv.org/abs/2511.20347) - 2025
- [ASPO: Asymmetric Importance Sampling Policy Optimization](https://arxiv.org/abs/2510.06062) - 2025
- [Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning](https://arxiv.org/abs/2512.05591) - 2025
- [VESPO: Variational Sequence-Level Soft Policy Optimization](https://arxiv.org/abs/2602.10693) - 2026
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://arxiv.org/abs/2508.10687) - 2025
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) - Schulman et al., 2015
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [RLOO](https://arxiv.org/abs/2402.14740) - Ahmadian et al., 2024
- [AReaL](https://arxiv.org/abs/2505.24298) - 2025
- [HybridFlow](https://arxiv.org/abs/2409.19256) - 2024
- [Step 3.5 Flash](https://arxiv.org/abs/2602.10604) - StepFun Team, 2026
