# untitled

# Does MIS-PO Need Ratio-Based Trajectory Selection? A Random-Rejection Mechanism Test

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly improved using reinforcement learning with automated feedback signals. In **reinforcement learning with verifiable rewards (RLVR)**, the model is trained from rewards computed by a programmatic verifier (e.g., exact-match of a math answer, or unit tests for code), which avoids training a separate reward model.

A practical issue in RLVR systems is that rollout generation is often decoupled from learning: a high-throughput inference engine (e.g., vLLM) produces rollouts, while a separate training stack (e.g., PyTorch + FSDP/Megatron) computes gradients. Even when the algorithm is intended to be on-policy, this disaggregation and/or explicit policy staleness can induce off-policy effects (mismatched action probabilities under the “same” parameters), which can destabilize policy-gradient updates.

**Step 3.5 Flash** (StepFun Team, 2026) proposes **MIS-Filtered Policy Optimization (MIS-PO)** to stabilize this regime via **binary accept/reject filtering** based on probability ratios between the training policy snapshot and the inference backend policy (see `./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/sections/MIS-Filtered Policy Optimization (MIS-PO).md`). Their ablations show higher reward and substantially reduced actor-gradient spikes versus PPO/GSPO, along with an “All Accept Ratio” diagnostic (see `.../sections/Output requirement.md`).

### The Problem

MIS-PO filters at two granularities:
- **Token-level filtering** rejects individual tokens whose ratio \(x_t = \pi_{\theta_{old}}(a_t|s_t) / \pi_{\theta_{vllm}}(a_t|s_t)\) is outside \([0.5, 2.0]\).
- **Trajectory-level filtering** rejects entire trajectories whose geometric-mean ratio \(\rho(\tau)=\exp(\tfrac{1}{T}\sum_t \log x_t)\) is outside \([0.996, 1.001]\).

The trajectory-level rule is usually interpreted as “ratio-based selection removes catastrophically off-policy trajectories.” However, there is a plausible alternative explanation:

- The main stabilizing effect may come from **rejecting a fraction of trajectories**, which reduces the effective batch size / update magnitude and can suppress gradient spikes, regardless of *which* trajectories are rejected.

Step 3.5 Flash does not include a control that holds the **trajectory acceptance count** fixed while changing **which trajectories** are accepted. Without such a mechanism control, it is hard to know whether the ratio-based trajectory selection is necessary, or whether a simpler acceptance-rate control would explain most of the benefit.

### Key Insight and Hypothesis

**Hypothesis**: MIS-PO’s trajectory-level stability gains are primarily driven by **rejecting some trajectories** (effective update-size reduction), not by the **ratio-based choice** of which trajectories to reject.

**Why this could be wrong**: the ratio rule might selectively remove rare, high-variance trajectories (heavy tails in \(\log\rho(\tau)\)) that dominate gradient spikes. If so, random rejection at the same acceptance count should be noticeably worse (lower accuracy and/or larger gradient spikes).

**Decision rule (pre-registered)**: run three conditions under the same RLVR setup and the same average trajectory acceptance rate:
1) MIS-PO (ratio-based token + trajectory filtering),
2) TokenOnly (token filtering only),
3) RandomTraj (randomly accept exactly the same number of trajectories as MIS-PO accepts per update).

If **RandomTraj** matches MIS-PO on final accuracy (within a small margin, e.g., ≈2 percentage points) and has similar high-percentile gradient spikes (e.g., ≤1.5×), then **ratio-based trajectory selection is not necessary beyond acceptance count** in this regime.

---

## Proposed Approach

### Overview

We propose a minimal mechanism test for MIS-PO: replace MIS-PO’s ratio-based trajectory selection with **uniform random trajectory rejection** while matching the **per-update accepted trajectory count**. This isolates whether stability comes from *which trajectories* are dropped (selection) or mainly from *how many* are dropped (acceptance rate).

### Method Details

Let a rollout batch contain trajectories \(\{\tau_i\}_{i=1}^N\) sampled by the inference backend policy \(\pi_{\theta_{vllm}}\). Define per-token ratios and trajectory statistic as in Step 3.5 Flash:
\[
 x_t(\tau) = \pi_{\theta_{old}}(a_t\mid s_t) / \pi_{\theta_{vllm}}(a_t\mid s_t),
\qquad
 \rho(\tau)=\exp\left(\tfrac{1}{T}\sum_t \log x_t\right).
\]

We evaluate three conditions:

1) **MIS-PO (baseline)**
- Token mask \(I(x_t \in [0.5,2.0])\)
- Trajectory mask \(I(\rho(\tau) \in [0.996,1.001])\)

2) **TokenOnly (ablation)**
- Token mask only
- Trajectory mask forced to 1

3) **RandomTraj (mechanism control)**
- Compute MIS-PO’s accepted-trajectory count per update:
\[
N_{acc} = \sum_i I(\rho(\tau_i) \in [0.996, 1.001]).
\]
- Accept exactly \(N_{acc}\) trajectories uniformly at random from the batch (ignoring \(\rho(\tau)\) for selection).
- Apply the same token-level mask as MIS-PO.

**Normalization confound control**: compute advantage normalization statistics **before** applying trajectory masking (so the mask does not change the baseline/normalizer).

**Off-policy knob (staleness)**: we will induce a controllable policy mismatch by running rollouts with a lagged policy snapshot \(\theta_{vllm}=\theta_{t-s}\), where \(s\) is the number of learner updates of staleness (as in “stale-k RL training” setups). We will run a short pilot over \(s\in\{32,128,256\}\) using MIS-PO only, and select an \(s\) where trajectory acceptance is non-degenerate (roughly 0.3–0.7).

### Key Innovations

- **Acceptance-count-matched random rejection** as a mechanism control for trajectory-level filtering in MIS-PO.
- A three-condition design that can decisively attribute stability gains to (i) token filtering alone, (ii) acceptance rate alone, or (iii) ratio-based trajectory selection.

---

## Related Work

### Field Overview

Recent RLVR/RLHF work increasingly focuses on stabilizing policy optimization under system-induced off-policy effects (asynchronous rollouts, rollout/training disaggregation, and long-horizon generation). Stabilization methods broadly fall into: (i) redefining or aggregating importance ratios (token/prefix/sequence), (ii) changing clipping/masking rules to limit gradient variance, (iii) enforcing explicit trust-region constraints based on divergence/variance statistics, and (iv) shaping the data distribution via filtering or rejection sampling.

MIS-PO is in the masking/filtering family: it replaces continuous importance weights with binary accept/reject masks at token and trajectory level. Similar concerns about acceptance patterns (e.g., length-dependent clipping, selection bias) appear in sequence-level PPO/GRPO variants, and motivate fairness-aware clipping and rejection-sampling style methods.

### Related Papers

(Links are proposal-local when available; otherwise arXiv/OpenReview URLs.)

- **[Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)**: Introduces MIS-PO with token+trajectory filtering for off-policy RLVR stability.
- **[Prosperity before Collapse / M2PO](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/meta/meta_info.txt)**: Stabilizes RLVR under extreme staleness via second-moment trust region masking.
- **[GSPO](./references/Group-Sequence-Policy-Optimization/meta/meta_info.txt)**: Uses sequence-level ratios/clipping for group-based RLVR and MoE stability.
- **[MinPRO](./references/A-Step-Back-Prefix-Importance-Ratio-Stabilizes-Policy-Optimization/meta/meta_info.txt)**: Uses prefix-aware importance ratios to better control off-policy corrections.
- **[Trust Region Masking (TRM)](./references/Trust-Region-Masking-for-Long-Horizon-LLM-Reinforcement-Learning-thanks-First-version-December-10-2025/meta/meta_info.txt)**: Proposes sequence-level masking with a trust-region criterion for long-horizon RL.
- **[Online Causal Kalman Filtering / KPO](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt)**: Denoises ratio signals / variance proxies to stabilize policy optimization.
- **[GFPO](./references/Sample-More-to-Think-Less-Group-Filtered-Policy-Optimization-for-Concise-Reasoning/meta/meta_info.txt)**: Uses group-level filtering for token/length efficiency in RLVR, highlighting the importance of selection rules.
- **[DAPO](https://arxiv.org/abs/2503.14476)**: Open-source large-scale RLVR system; provides practical GRPO/PPO implementations and normalization details.
- **[DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)**: Scales RLVR for math reasoning and popularizes datasets used in later off-policy work.
- **[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56)**: Documents system-induced mismatch between inference and training backends and proposes truncated importance sampling.
- **[Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm](https://arxiv.org/abs/2509.24203)**: Interprets GRPO-style methods as off-policy regularized REINFORCE; emphasizes clipping/weighting effects.
- **[FSPO: Enforcing Length Fairness for Sequence-Level RL](https://arxiv.org/abs/2509.09177)**: Shows fixed clipping induces length-dependent acceptance; proposes \(\sqrt{L}\)-scaled clipping to equalize acceptance.
- **[BAPO: Stabilizing Off-Policy RL for LLMs](https://arxiv.org/abs/2510.18927)**: Stabilizes off-policy RL via balanced policy optimization with adaptive clipping.
- **[TOPR: Tapered Off-Policy REINFORCE](https://arxiv.org/abs/2503.14286)**: Uses tapered updates to stabilize off-policy REINFORCE-style training.
- **[AREAL](https://arxiv.org/abs/2505.24298)**: Large-scale asynchronous RL infrastructure; highlights staleness and systems constraints.
- **[Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch](https://arxiv.org/abs/2602.06107)**: Uses optimal budgeted rejection sampling to align rollout and policy distributions.
- **[PPO](https://arxiv.org/abs/1707.06347)**: Clipped surrogate policy gradient objective.
- **[TRPO](https://arxiv.org/abs/1502.05477)**: KL trust-region policy optimization.
- **[RLOO / leave-one-out REINFORCE variants](https://arxiv.org/abs/2310.03716)**: Low-variance REINFORCE baselines used in LLM RL.
- **[Act Only When It Pays](https://arxiv.org/abs/2506.02177)**: Selective rollouts for efficiency in reasoning RLVR.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Token/prefix/sequence ratio redesign | Change ratio definition to reduce variance / correct off-policy bias | MinPRO, GSPO, FSPO | MATH500, AIME, code RLVR | Does not directly test whether selection vs count matters |
| Masking / trust region constraints | Enforce trust region by masking samples/tokens | MIS-PO, TRM, M2PO | Long-horizon RLVR, staleness studies | Mechanism often unclear without acceptance-matched controls |
| Data filtering / selection | Filter samples based on reward/length/difficulty proxies | GFPO, selective rollouts | AIME, GPQA, tool tasks | Selection criteria can introduce bias/confounds |
| Rejection sampling / distribution shaping | Reject tokens/samples to align behavior vs target policies | Jackpot, MIS-style ideas | Math RLVR under mismatch | Often couples acceptance rate and selection rule |
| System mismatch corrections | Explicitly correct inference/training mismatch | Truncated IS (Yao et al.) | System-level RLVR | Requires careful estimator design |

### Closest Prior Work

- **MIS-PO (Step 3.5 Flash)**: Closest method; introduces ratio-based trajectory filtering but does not isolate whether gains come from selection vs acceptance count.
- **GSPO / sequence-level clipping**: Uses sequence-level ratios and clipping to control variance, but does not provide an acceptance-count-matched random control for MIS-style trajectory rejection.
- **M2PO (Prosperity before Collapse)**: Uses a second-moment trust region under explicit staleness and provides strong published baselines, but targets a different mechanism (variance proxy) rather than selection-vs-count.
- **Jackpot (OBRS)**: Also uses rejection to address mismatch, but focuses on provably optimal rejection distributions under a budget rather than mechanistically testing whether ratio-informed selection matters.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MIS-PO | Token + trajectory ratio filtering for stability | No acceptance-matched selection control | Add RandomTraj acceptance-count-matched random rejection | Isolates whether selection by ratio is necessary |
| GSPO / FSPO | Sequence-level ratios and clipping/fair clipping | Not a mechanism control for MIS-PO | Keep MIS-PO but randomize accepted trajectories | Directly answers the mechanism question |
| M2PO | Second-moment trust region under staleness | Different mechanism, more complex | Test simpler MIS-style mechanism with controls | Clarifies when simple rejection suffices |
| Jackpot | Optimal rejection sampling under mismatch | Not about mechanism attribution | Use random rejection control matched to MIS acceptance | Causal attribution for MIS-PO component |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen3-1.7B-Base | 1.7B | https://huggingface.co/Qwen/Qwen3-1.7B-Base | Small dense model; fast RLVR verification |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| DeepScaleR-Preview-Dataset | RLVR training prompts (math problems with exact answers) | ~40k | https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset | Not stated on dataset card (verifier should confirm) |
| DAPO-Math-17k (fallback) | RLVR training prompts (math) | ~17k | https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k | Not stated on dataset card (verifier should confirm) |

**Other Resources (if applicable):**
- vLLM for rollout generation; PyTorch for training.

**Resource Estimate** (must fit ≤768 A100 GPU-hours):
- **Pilot**: MIS-PO only, \(s\in\{32,128,256\}\), 30 steps each → 90 steps total.
- **Main**: 1 selected staleness \(s\), 3 conditions × 500 steps (1.5k steps total).
- **Compute budget (cap)**: run on **8×A100-80GB** and cap wall-clock to **≤72 hours total**, i.e., **≤576 GPU-hours**, plus **≤96 GPU-hours** for pilots → **≤672 GPU-hours** total.
- **GPU memory**: Qwen3-1.7B fits on a single 80GB GPU; multi-GPU used for throughput (FSDP optional).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MATH-500 | 500-problem subset of the MATH benchmark; evaluates math reasoning via exact final-answer correctness | pass@1 (higher is better) | test | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | Simple answer extraction + exact match / sympy normalization (to be implemented) |

**Training-time stability metrics (logged every update):**
- **Actor gradient norm** (mean, 99th percentile, max spike; lower spikes indicate more stable optimization).
- **Trajectory acceptance rate** (fraction of trajectories kept by MIS-PO; RandomTraj matches by construction).
- **Token acceptance rate** (fraction of tokens passing token mask).
- **Entropy** (to detect collapse).

### Main Results

#### Comparability Rules (CRITICAL)
All rows below use:
- Base model: Qwen3-1.7B-Base
- Evaluation: MATH-500 pass@1 on the test split

Published baselines may differ in training-step budgets; these differences are noted.

#### Results Table

| Method | Base Model | Benchmark | pass@1 (Math500) | Stability (grad-norm spikes) | Source | Notes |
|---|---|---|---:|---|---|---|
| GRPO | Qwen3-Base-1.7B | MATH-500 | 64.3% (s=256) | N/A | `./references/Prosperity-before-Collapse-.../sections/6.2 Performance Comparison on Training with Staleness.md` (Table 1) | Published baseline; trained 1000 steps in M2PO |
| GSPO | Qwen3-Base-1.7B | MATH-500 | 65.0% (s=256) | N/A | same | Published baseline; trained 1000 steps in M2PO |
| M2PO | Qwen3-Base-1.7B | MATH-500 | 71.8% (s=256) | N/A | same | Published reference; trained 1000 steps in M2PO |
| TokenOnly | Qwen3-1.7B-Base | MATH-500 | **TBD** | **TBD** | - | Needs re-run (our ablation) |
| RandomTraj (Ours) | Qwen3-1.7B-Base | MATH-500 | **TBD** | **TBD** | - | Mechanism control |
| MIS-PO | Qwen3-1.7B-Base | MATH-500 | **TBD** | **TBD** | - | Needs re-run (no published result in this exact setting) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| TokenOnly | Remove trajectory-level rejection | If ≈ MIS-PO → trajectory rejection not needed |
| RandomTraj | Randomize which trajectories are accepted (match acceptance count) | If ≈ MIS-PO → ratio-based selection not needed beyond count |

### Analysis (Optional)

- **Tail analysis of \(\log\rho(\tau)\)**: compare \(\log\rho\) distributions for MIS-PO-accepted vs RandomTraj-accepted trajectories to test whether MIS-PO removes heavy tails.

---

## Success Criteria

**Criterion 1 (trajectory masking necessity):** If TokenOnly achieves similar final accuracy and does not substantially increase gradient spikes relative to MIS-PO, then trajectory-level masking is not necessary in this regime.

**Criterion 2 (selection necessity, conditional):** If TokenOnly is worse than MIS-PO, but RandomTraj matches MIS-PO, then ratio-based trajectory selection is not necessary beyond matching the acceptance count.

**Criterion 3 (selection matters):** If RandomTraj is noticeably worse than MIS-PO (lower accuracy and/or substantially larger gradient spikes), then ratio-based selection contributes meaningfully to stability.

---

## Impact Statement

This study provides a minimal mechanism control that can change how RLVR practitioners interpret and adopt MIS-PO-style trajectory filtering. If acceptance-count-matched random rejection matches MIS-PO, future work should treat **acceptance-rate-matched baselines** as mandatory before attributing gains to specific selection rules; if not, it strengthens the case for ratio-based selection as a genuine variance-reduction mechanism.

---

## References

- [Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt) - StepFun Team, 2026
- [Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?](./references/Prosperity-before-Collapse-How-Far-Can-Off-Policy-RL-Reach-with-Stale-Data-on-LLMs/meta/meta_info.txt) - Zheng et al., 2025
- [Group Sequence Policy Optimization](./references/Group-Sequence-Policy-Optimization/meta/meta_info.txt) - Zheng et al., 2025
- [A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization](./references/A-Step-Back-Prefix-Importance-Ratio-Stabilizes-Policy-Optimization/meta/meta_info.txt) - (arXiv), 2026
- [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](./references/Trust-Region-Masking-for-Long-Horizon-LLM-Reinforcement-Learning-thanks-First-version-December-10-2025/meta/meta_info.txt) - (arXiv), 2025
- [Online Causal Kalman Filtering for Stable and Effective Policy Optimization](./references/Online-Causal-Kalman-Filtering-for-Stable-and-Effective-Policy-Optimization/meta/meta_info.txt) - He et al., 2026
- [Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning](./references/Sample-More-to-Think-Less-Group-Filtered-Policy-Optimization-for-Concise-Reasoning/meta/meta_info.txt) - Shrivastava et al., 2025
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) - Yu et al., 2025
- [DeepScaleR: Surpassing o1-preview with a 1.5B model by scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) - Luo et al., 2025
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56) - Yao et al., 2025
- [Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm](https://arxiv.org/abs/2509.24203) - Yao et al., 2025
- [FSPO: Enforcing Length Fairness for Sequence-Level RL](https://arxiv.org/abs/2509.09177) - Mao et al., 2025
- [BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping](https://arxiv.org/abs/2510.18927) - Xi et al., 2025
- [Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch](https://arxiv.org/abs/2602.06107) - Chen et al., 2026
- [TOPR: Tapered Off-Policy REINFORCE](https://arxiv.org/abs/2503.14286) - Le Roux et al., 2025
- [AREAL: A large-scale asynchronous reinforcement learning system for language reasoning](https://arxiv.org/abs/2505.24298) - Fu et al., 2025
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) - Schulman et al., 2015
- [RLOO / leave-one-out REINFORCE variants](https://arxiv.org/abs/2310.03716) - (arXiv), 2023
- [Act only when it pays: Efficient RL for LLM reasoning via selective rollouts](https://arxiv.org/abs/2506.02177) - Zheng et al., 2025
