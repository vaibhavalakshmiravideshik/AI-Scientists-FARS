# untitled

# Divergence-Masked GRPO for Bounded-Staleness Rollout Replay: A DPPO vs MinPRO Diagnosis of ECHO-2’s S=11 Collapse

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Reinforcement learning with verifiable rewards (RLVR) improves large language models (LLMs) using automated checkers (e.g., exact-match of a final numeric answer). In this setting, the main cost driver is rollout generation: sampling many long solutions per prompt. Modern RLVR systems therefore decouple rollout generation from the learner and train from a replay buffer.

**ECHO-2** is a recent distributed RLVR framework that offloads rollouts to cheaper wide-area workers while keeping a centralized learner saturated via **bounded policy staleness**: each trajectory is tagged with the snapshot version that generated it, and the learner only consumes trajectories with version lag ≤ **S** learner steps **[ECHO-2](./references/ECHO-2-A-Large-Scale-Distributed-Rollout-Framework-for-Cost-Efficient-Reinforcement-Learning/meta/meta_info.txt)**. On Qwen3-8B, ECHO-2 reports that RL quality is robust for **S≤6** but **diverges at S=11** (Figure 3b; “RL Quality under Bounded Staleness”). This collapse boundary limits how much temporal slack ECHO-2 can use to reduce broadcast pressure and hide stragglers.

The key open question for practitioners is not just “how to stabilize S=11”, but **why** bounded-staleness GRPO collapses: different root causes imply different long-term algorithm choices.

### The Problem

ECHO-2’s learner uses **GRPO (Group Relative Policy Optimization)**, a PPO-style policy-gradient method for LLMs that normalizes rewards within a prompt-level group (avoiding a learned value-function critic). It optimizes a clipped surrogate objective with **KL regularization** and an outer truncated importance ratio for replay reuse (Appendix D.3; Eq. 24) **[ECHO-2](./references/ECHO-2-A-Large-Scale-Distributed-Rollout-Framework-for-Cost-Efficient-Reinforcement-Learning/meta/meta_info.txt)**. When rollouts are stale, the update becomes off-policy, and importance ratios can become heavy-tailed.

Two recent lines of work suggest *different* explanations for collapse under off-policy drift:

1) **Prefix-drift / “token ratio is the wrong correction”**. **MinPRO** proves that the theoretically correct off-policy correction for autoregressive generation is a *prefix* importance ratio rather than the per-token ratio used in PPO/GRPO, and proposes a stable minimum-prefix surrogate that stabilizes large off-policy lag in VeRL-style training **[MinPRO](./references/A-Step-Back-Prefix-Importance-Ratio-Stabilizes-Policy-Optimization/meta/meta_info.txt)**.

2) **Trust-region / “ratio clipping is a bad proxy for divergence”**. **DPPO** argues PPO’s ratio clipping is structurally ill-suited for LLMs (large vocabularies + long-tail token probabilities): it over-penalizes low-probability tokens and under-constrains catastrophic probability-mass shifts in high-probability tokens. DPPO proposes a divergence-based mask using efficient Binary-TV / Binary-KL approximations and reports substantial stability gains **[DPPO](./references/Rethinking-the-Trust-Region-in-LLM-Reinforcement-Learning/meta/meta_info.txt)**.

Both can plausibly fix ECHO-2’s S=11 collapse, but they imply different diagnoses:
- If **prefix drift** dominates, system-level staleness tolerances depend mainly on modeling the *correct* off-policy ratio structure.
- If **trust-region mis-specification** dominates, the fix is to constrain updates using *absolute* probability-mass movement (divergence), and “prefix fixes” may be unnecessary or secondary.

### Key Insight and Hypothesis

**Key insight:** Bounded-staleness replay provides a natural stress test that can distinguish these hypotheses, because it produces a mixture of snapshot ages while keeping the reward/verifier and the rollout budget fixed.

**Hypothesis (trust-region diagnosis):** In an ECHO-2-style bounded-staleness replay buffer at **S=11**, replacing ratio clipping with **DPPO’s divergence-based token mask** will (i) avoid collapse in more seeds than the baseline clipped-GRPO, and (ii) reach a fixed reward / AIME24 avg@64 threshold faster than MinPRO’s prefix-ratio surrogate.

**Why we could be wrong:**
- DPPO’s mask may act as an **implicit staleness filter** (masking grows strongly with snapshot age), and stability gains could come mainly from discarding old trajectories rather than fixing clipping.
- MinPRO may be the dominant factor at S=11 if collapse is driven by the theoretical mismatch between token ratios and prefix ratios.
- DPPO was validated primarily under on-policy-ish RLVR settings and training–inference mismatch; it may not transfer to replay-based bounded staleness.

---

## Proposed Approach

### Overview

We propose a controlled, diagnosis-oriented comparison under an ECHO-2-style bounded-staleness replay buffer:

- **Baseline (Clip-GRPO)**: ECHO-2-style token-ratio clipping in the inner GRPO surrogate.
- **Prefix baseline (MinPRO-GRPO)**: replace token ratio with a minimum-prefix surrogate inside the same GRPO+KL objective.
- **Ours (DPPO-Masked GRPO)**: replace ratio clipping with DPPO’s divergence-based mask using Binary-TV divergence, while keeping the same ratio term and KL regularization.

The core claim is *not* that DPPO is a new RL algorithm, but that **bounded-staleness collapse is better explained by trust-region mis-specification than by prefix drift** if DPPO outperforms MinPRO at S=11.

### Method Details

#### Setting: bounded-staleness replay

Each trajectory record stores `(prompt x, response y, reward r, snapshot version v, metadata Ω)` as in ECHO-2. At learner step `t` (version `v_t`), only trajectories with `v ≥ v_t − S` are admissible.

We use ECHO-2’s publicly described hyperparameters where possible (global batch size 128, max generation length 8192, temperature 1.0, top-p 0.95, rollout group size n=16, no chain-of-thought prompting) **[ECHO-2](./references/ECHO-2-A-Large-Scale-Distributed-Rollout-Framework-for-Cost-Efficient-Reinforcement-Learning/meta/meta_info.txt)**.

#### Baseline: ECHO-2 GRPO objective

ECHO-2 defines a token ratio per response token (Appendix D.3):

\(\rho_{i,t}(\theta)=\pi_\theta(y_{i,t}|x_i,y_{i,<t})/\pi_{\theta_{old}}(y_{i,t}|x_i,y_{i,<t})\),

and optimizes a clipped surrogate with KL regularization, with an outer truncated replay weight (Eq. 24). Our focus is the inner clipping mechanism because DPPO and MinPRO both intervene there.

#### MinPRO-GRPO (prefix-drift baseline)

MinPRO replaces the token ratio with a stable prefix-aware surrogate:
- Let per-token ratios be \(\rho_t\).
- Define a minimum prefix statistic \(\underline{\rho}_t=\min_{i<t}\rho_i\).
- Use surrogate \(\tilde{\rho}_t = \underline{\rho}_t\,\rho_t\) in place of \(\rho_t\).

This downweights tokens whose *prefix* has become unlikely under the current policy, even if the current token ratio is near 1.

#### DPPO-Masked GRPO (ours)

DPPO replaces ratio clipping with a divergence-conditioned mask.

For each trajectory generated by snapshot version `v`, treat the behavior policy as \(\mu = \pi_{\theta_v}\) and the current learner policy as \(\pi=\pi_\theta\). For each sampled token \(a_t\) in state \(s_t\):

- Ratio: \(r_t = \pi(a_t|s_t)/\mu(a_t|s_t)\).
- Binary-TV divergence (DPPO Eq. 13): \(D_t = |\mu(a_t|s_t)-\pi(a_t|s_t)|\).
- DPPO mask (DPPO Eq. 12):
  - \(M_t=0\) if \((\hat A_t>0 \wedge r_t>1 \wedge D_t>\delta)\) or \((\hat A_t<0 \wedge r_t<1 \wedge D_t>\delta)\)
  - else \(M_t=1\).

We then use the masked surrogate \(\sum_t M_t\,r_t\,\hat A_t\) (plus the same KL-to-reference term used in the baseline).

**Choosing \(\delta\) without tuning confounds:** In a 1-seed Stage-0 run at S=11, we measure the fraction of tokens that would be “clipping-active” under the baseline (\(|r_t-1|>\epsilon\)). We then set \(\delta\) so that DPPO’s mask rate approximately matches this fraction. This makes the comparison about **which tokens are filtered** (divergence-based vs ratio-based), not about filtering more aggressively.

### Key Innovations

1) **Differential diagnosis of bounded-staleness collapse**: a clean head-to-head between a prefix-ratio hypothesis (MinPRO) and a trust-region hypothesis (DPPO) in the same bounded-staleness replay regime.
2) **Off-policy DPPO instantiation**: DPPO is evaluated in a replay setting where the behavior policy is a mixture of snapshot versions (bounded staleness), not just training–inference mismatch.
3) **Required confound checks** to distinguish “trust-region fix” from “implicit staleness filtering,” including mask-vs-age and mask-vs-clip overlap analyses.

---

## Related Work

### Field Overview

Stabilizing LLM RL under system-induced off-policy effects has become a central problem because practical RLVR/RLHF stacks decouple rollout generation from learning and often combine heterogeneous inference backends with separate training engines. This induces policy mismatch via (i) explicit staleness (asynchronous rollouts, replay), (ii) training–inference mismatch (different kernels / precision), and (iii) MoE routing instability.

Algorithmic approaches fall into several families: (1) ratio aggregation and prefix-/sequence-level corrections (MinPRO, GSPO), (2) alternative trust regions and masking rules (DPPO, trust-region masking, M2PO), (3) variance-reduction for importance weights (GEPO), and (4) systems-level staleness management (ECHO-2, AReaL, StreamRL).

### Related Papers

- **[ECHO-2](./references/ECHO-2-A-Large-Scale-Distributed-Rollout-Framework-for-Cost-Efficient-Reinforcement-Learning/meta/meta_info.txt)**: Distributed rollouts with bounded staleness; reports GRPO instability at S=11.
- **[MinPRO](./references/A-Step-Back-Prefix-Importance-Ratio-Stabilizes-Policy-Optimization/meta/meta_info.txt)**: Prefix-importance-ratio surrogate stabilizing off-policy LLM RL under delayed rollouts.
- **[DPPO](./references/Rethinking-the-Trust-Region-in-LLM-Reinforcement-Learning/meta/meta_info.txt)**: Divergence-based trust region masking (Binary-TV/KL) replacing PPO ratio clipping.
- **[M2PO / Prosperity before Collapse](https://arxiv.org/abs/2510.01161)**: Second-moment trust region masking enabling extreme staleness (256+ updates).
- **[Trust Region Masking for Long-Horizon LLM RL](https://arxiv.org/abs/2512.23075)**: Sequence-level masking derived from long-horizon trust-region bounds; highlights clipping “gradient leakage”.
- **[Step 3.5 Flash (MIS-PO)](https://arxiv.org/abs/2602.10604)**: Binary token+trajectory ratio filtering between inference and training policies.
- **[Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)**: Standard ratio-clipping trust region.
- **[Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)**: KL-constrained trust region baseline.
- **[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)**: Popularizes GRPO-style RLVR for math reasoning.
- **[HybridFlow / VeRL](https://arxiv.org/abs/2409.19256)**: RLHF/RLVR training framework used by many recent off-policy studies.
- **[AReaL](https://arxiv.org/abs/2505.24298)**: Asynchronous RL system for reasoning; motivates staleness-tolerant objectives.
- **[StreamRL](https://arxiv.org/abs/2504.15930)**: Heterogeneous RL with disaggregated stream generation.
- **[RhymeRL](https://arxiv.org/abs/2503.20783)**: Uses historical data reuse to accelerate LLM RL.
- **[GEPO](https://arxiv.org/abs/2508.17850)**: Group-expectation importance weighting for heterogeneous distributed RLHF.
- **[ASPO](https://arxiv.org/abs/2510.06062)**: Studies asymmetries in importance sampling / clipping in LLM RL.
- **[CISPO](https://arxiv.org/abs/2601.22801)**: Clipping-free policy optimization; can be unstable under drift.
- **[GSPO](https://arxiv.org/abs/2507.18071)**: Sequence-level ratio/clipping for group-based RL.
- **[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://arxiv.org/abs/2508.05512)**: Documents system-induced off-policy effects.
- **[On the Rollout-Training Mismatch in Modern RL Systems](https://arxiv.org/abs/2509.10123)**: Studies mismatch between rollout and training logprobs and correction via truncated IS.
- **[QuRL: Efficient RL with Quantized Rollout](https://arxiv.org/abs/2602.13953)**: RLVR under quantized rollouts; explores mismatch and correction.
- **[IMPALA / V-trace](https://arxiv.org/abs/1802.01561)**: Classic off-policy correction for actor-learner architectures.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Typical eval | Known limitations |
|---|---|---|---|---|
| Prefix-/sequence-aware ratios | Fix token-ratio approximation under off-policy drift | MinPRO, GSPO | AIME/MATH RLVR curves | May not address trust-region pathologies unrelated to prefix structure |
| Divergence-based trust regions | Constrain updates using distribution divergence, not sample ratio | DPPO, Trust Region Masking | RL stability + mismatch metrics | Needs divergence threshold choice; can mask many samples if drift is large |
| Moment/variance-based trust regions | Control higher moments of ratios to suppress tails | M2PO | Extreme staleness sweeps | More complex masking loop; may require extra statistics |
| Systems-level staleness control | Change which data is admissible / disseminated | ECHO-2, StreamRL | Cost/utilization + RL quality | Does not fix algorithmic instability at a given S |

### Closest Prior Work

- **ECHO-2**: Establishes bounded staleness S as a system knob and reports S=11 divergence, but does not identify the algorithmic root cause or propose a learner-side fix.
- **MinPRO**: Provides a prefix-ratio surrogate for off-policy drift, but studies delayed rollouts in a mostly synchronous setting rather than a bounded-staleness replay buffer with a mixture of snapshot ages.
- **DPPO**: Shows divergence-based masking improves stability vs ratio clipping, but does not evaluate bounded-staleness replay (explicit S) as a systems knob.
- **Trust Region Masking (TRM)**: Provides sequence-level masking theory and empirical gains, but is not positioned as a diagnosis against prefix-ratio methods in bounded-staleness systems.

**Novelty Kill Search Summary:** Searched for combinations of “DPPO bounded staleness GRPO”, “divergence-based masking staleness RLHF”, “ECHO-2 DPPO”, “binary TV mask GRPO replay”, and checked for direct DPPO+ECHO-2 follow-ups. As of 2026-02-26, we did not find work applying DPPO-style divergence masks to ECHO-2’s bounded-staleness S setting or directly comparing DPPO vs MinPRO in this regime.

### Comparison Table

| Related work | What it does | Key limitation for ECHO-2 S=11 | What we change | Why ours should win |
|---|---|---|---|---|
| ECHO-2 | Bounded-staleness system + clipped GRPO | Reports collapse at S=11; unclear cause | Swap trust region / ratio surrogate inside same system | Enables causal diagnosis of collapse mechanism |
| MinPRO | Prefix-ratio surrogate for off-policy drift | Not validated in bounded-staleness replay | Implement MinPRO inside bounded-staleness GRPO | Tests whether prefix structure is the bottleneck |
| DPPO | Divergence-based token mask for PPO/GRPO | Not tested under explicit replay staleness S | Implement DPPO mask with behavior snapshot as anchor | Tests whether clipping mis-specification is the bottleneck |
| TRM | Sequence-level masking based on max token KL | More complex; not compared to prefix methods | (Related baseline only) | Provides context for why token-level trust regions fail |
| M2PO | Second-moment trust region masking | More complex; different masking criterion | (Related baseline only) | Context for tail-driven instability under staleness |

---

## Experiments

### Experimental Setup

**Goal:** A minimal reproduction of “S=11 collapses” and a decisive comparison between (i) clipping, (ii) prefix-ratio correction, and (iii) divergence-masked trust region.

**System / codebase:**
- Implement using **VeRL / HybridFlow** GRPO pipeline: https://github.com/volcengine/verl
- Emulate ECHO-2 bounded staleness by tagging trajectories with snapshot versions and sampling only those with version lag ≤ S.

**Baseline Ladder (REQUIRED):**
- **Prompt-only baseline**: Qwen3-8B evaluated on AIME24 with the same sampling settings (temperature/top-p; report avg@64 and pass@64).
- **Stable RL reference**: baseline Clip-GRPO at S=6 (should be stable per ECHO-2).
- **Failure-case RL baseline**: Clip-GRPO at S=11.
- **Strong closest baselines at S=11**: MinPRO-GRPO and DPPO-Masked GRPO.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3-8B (base) | 8B | https://huggingface.co/Qwen/Qwen3-8B | Matches ECHO-2 model family |

**Training Data (RL prompts):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| DAPO-Math-17K | RLVR training prompts + verifier answers | ~17,000 problems | https://huggingface.co/datasets/haizhongzheng/DAPO-Math-17K-cleaned | Check dataset card |
| AIME24 | evaluation-only benchmark | 30 problems | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 | Check dataset card |

(If Stage-0 cannot reproduce the qualitative “S=11 diverges” on DAPO-Math-17K, fall back to training on AIME24 prompts directly to match ECHO-2’s setup.)

**Resource Estimate** (must fit ≤768 A100 GPU-hours):
- Use 8×A100-80GB total (4 GPUs learner + 4 GPUs co-located rollout workers).
- Use ECHO-2’s reported co-located step time as a conservative upper bound (≈1508s/step on 8×A100 in the centralized baseline).
- Staged schedule:
  - **Stage 0 (sanity + δ calibration; 1 seed)**:
    - S=6 Clip-GRPO for 10 updates (must be stable).
    - S=11 Clip-GRPO for up to 10 updates; measure clipping-active fraction and set δ.
  - **Stage A (decisive stability test; 3 seeds)**: 12 updates × 3 methods (Clip vs MinPRO vs DPPO) × 3 seeds at S=11, early-stop on collapse.
  - **Stage B (quality check; 1 seed each)**: extend only stable method(s) to 60 updates on 1 seed each and report final AIME24 avg@64.
- Worst-case updates: 20 + 12×9 + 48×2 = 214 updates.
- GPU-hours ≤ 214 × (1508/3600) × 8 ≈ 717 A100 GPU-hours.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AIME24 | 30 competition math problems with numeric answers | avg@64 (mean verifier reward across 64 samples); pass@64 | test | https://huggingface.co/datasets/HuggingFaceH4/aime_2024 | simple numeric-answer checker |

**Stability metrics (automated; pre-registered):**
- **Numerical failure**: any NaN/Inf in loss or gradients.
- **KL blow-up**: mean KL(π_θ || π_ref) over a training batch exceeds 2× its median over the first 5 updates for ≥3 consecutive updates.
- **Reward collapse**: moving-average batch reward (window=5) drops below 50% of its value at update 5 for ≥5 consecutive updates.

**Mechanism diagnostics (logged every update):**
- Mask rate (DPPO) / clipping-active rate (baseline) / effective prefix factor distribution (MinPRO).
- Mask rate vs snapshot age Δ (binned).
- Clip-vs-mask conditional probabilities: P(mask|clip-active), P(mask|clip-inactive), P(clip-active|mask).
- Gradient-norm distribution (mean/p95/p99/max).

### Main Results

#### Comparability Rules (CRITICAL)

All rows must use:
- Same base model (Qwen3-8B base)
- Same rollout sampling settings (temperature/top-p, max length, group size)
- Same staleness budget S and publication period κ (default κ=S−1 unless otherwise stated)
- Same training updates per stage and early-stop rules

#### Results Table

| Method | Base Model | Staleness S | AIME24 avg@64 (mean±std) | Collapse rate (seeds) | Time-to-threshold (updates) | Source | Notes |
|---|---|---:|---:|---:|---:|---|---|
| Prompt-only | Qwen3-8B | - | TBD | - | - | To run | Same sampling as RL eval |
| Clip-GRPO (stable reference) | Qwen3-8B | 6 | TBD | TBD | TBD | To run | Should be stable per ECHO-2 |
| Clip-GRPO (failure case) | Qwen3-8B | 11 | TBD | TBD | TBD | To run | Target collapse regime |
| MinPRO-GRPO | Qwen3-8B | 11 | TBD | TBD | TBD | To run | Prefix-ratio surrogate baseline |
| **DPPO-Masked GRPO (ours)** | Qwen3-8B | 11 | TBD | TBD | TBD | To run | Binary-TV mask; δ calibrated to match clipping-active fraction |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Age-only filter (analysis control) | Drop tokens/trajectories based only on snapshot age to match DPPO’s overall mask rate | If DPPO ≫ age-only, it’s not just “implicit smaller S” |

### Experimental Rigor

- **Seeds:** 3 seeds for Stage A; Stage B uses 1 seed per stable method due to budget.
- **δ selection:** δ chosen once in Stage 0; no per-method tuning.
- **Validity threats + controls:**
  1) **Masking-fraction confound**: δ chosen to match clipping-active fraction.
  2) **Implicit staleness filtering**: report mask rate vs age and compare to age-only filter.
  3) **“DPPO is just stricter clipping”**: report conditional overlap P(mask|clip-inactive).

---

## Success Criteria

**Hypothesis:** DPPO-Masked GRPO stabilizes S=11 and reaches a fixed reward/avg@64 threshold faster than MinPRO-GRPO, implying bounded-staleness collapse is primarily a trust-region problem.

**Decision Rule:**
- **Proceed** if at S=11:
  - DPPO avoids collapse in ≥2/3 seeds in Stage A, and
  - DPPO reaches the pre-registered reward/avg@64 threshold in fewer updates than MinPRO in ≥2/3 seeds, and
  - P(mask|clip-inactive) is non-trivial (e.g., ≥20%), suggesting DPPO filters tokens clipping would not.
- **Pivot** if both DPPO and MinPRO are stable but indistinguishable in time-to-threshold (ambiguous diagnosis): run DPPO+MinPRO combination once as a tiebreaker and report whether effects are additive.
- **Refute** if DPPO collapses similarly to Clip-GRPO at S=11, or if DPPO’s gains are matched by the age-only filter control (suggesting it is mainly a staleness filter).

---

## Impact Statement

If DPPO outperforms MinPRO under bounded staleness, practitioners building distributed RLVR systems (ECHO-2-like) should prioritize **divergence-based trust regions** over prefix-ratio corrections when pushing staleness budgets. If MinPRO wins, it supports focusing on **prefix-aware off-policy corrections** as the primary route to higher staleness tolerance.

---

## References

- [ECHO-2: A Large-Scale Distributed Rollout Framework for Cost-Efficient Reinforcement Learning](./references/ECHO-2-A-Large-Scale-Distributed-Rollout-Framework-for-Cost-Efficient-Reinforcement-Learning/meta/meta_info.txt) - Xiao et al., 2026
- [A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization](./references/A-Step-Back-Prefix-Importance-Ratio-Stabilizes-Policy-Optimization/meta/meta_info.txt) - Lei et al., 2026
- [Rethinking the Trust Region in LLM Reinforcement Learning](./references/Rethinking-the-Trust-Region-in-LLM-Reinforcement-Learning/meta/meta_info.txt) - Qi et al., 2026
- [Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?](https://arxiv.org/abs/2510.01161) - Zheng et al., 2025
- [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](https://arxiv.org/abs/2512.23075) - Li et al., 2025
- [Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters](https://arxiv.org/abs/2602.10604) - StepFun Team, 2026
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) - Schulman et al., 2015
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256) - Sheng et al., 2025
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298) - Fu et al., 2025
- [StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation](https://arxiv.org/abs/2504.15930) - Zhong et al., 2025
- [RhymeRL: Accelerating LLM Reinforcement Learning with History Rhymes](https://arxiv.org/abs/2503.20783) - He et al., 2025
- [Group Expectation Policy Optimization for Stable Heterogeneous Distributed RLHF](https://arxiv.org/abs/2508.17850) - Zhang et al., 2025
- [ASPO: Asymmetric Importance Sampling Policy Optimization](https://arxiv.org/abs/2510.06062) - Wang et al., 2025
- [Clipping-Free Policy Optimization for Large Language Models (CISPO)](https://arxiv.org/abs/2601.22801) - Chen et al., 2026
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) - Zheng et al., 2025
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://arxiv.org/abs/2508.05512) - Yao et al., 2025
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) - Espeholt et al., 2018
- [QuRL: Efficient Reinforcement Learning with Quantized Rollout](https://arxiv.org/abs/2602.13953) - Li et al., 2026
- [On the Rollout-Training Mismatch in Modern RL Systems](https://arxiv.org/abs/2509.10123) - Yao et al., 2025
