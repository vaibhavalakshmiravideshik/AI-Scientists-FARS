# untitled

# ScalarPhase Mamba-3: Testing Whether One Shared Data-Dependent Rotation Angle Suffices for State-Tracking

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Linear-time sequence models (state space models and linear attention variants) are attractive alternatives to Transformers because they can decode with constant memory and linear compute, avoiding the growing KV-cache bottleneck in long-context generation. However, multiple recent analyses show that many efficient “diagonal / scalar-transition” linear models have severe limitations on simple algorithmic state-tracking tasks such as parity and modular arithmetic.

Mamba-3 (ICLR 2026 submission) proposes a principled fix: reintroduce complex-valued state-space dynamics. After discretization, a complex diagonal state transition corresponds to a real-valued state update with block-diagonal 2×2 rotations. Mamba-3 further shows this is equivalent to applying **data-dependent rotary embeddings (RoPE)** to the input/output projections (B and C) via a “RoPE trick”, and reports near-perfect length generalization on parity and modular arithmetic (Table 4b) while Mamba-3 without RoPE and Mamba-3 with standard (data-independent) RoPE both fail.

### The Problem

Mamba-3’s complexification introduces an additional design degree of freedom: the imaginary part of the diagonal transition is written as a vector of rotation rates θ_t[i] (one per 2×2 state pair). This implies maintaining many independent phase channels and their cumulative products over time.

It is unclear whether this per-pair phase diversity is actually necessary for the capabilities Mamba-3 targets, or whether much simpler “one-oscillator” dynamics would already solve the key state-tracking tasks. If a single shared, data-dependent rotation angle were sufficient for parity-like tasks, this would:

- clarify the minimal mechanism required to recover state-tracking in Mamba-like SSMs,
- suggest simpler implementations and parameterizations (e.g., rank-1 / pooled phase projections), and
- provide a boundary-condition test separating “single-counter” tasks from tasks requiring multiple independent counters.

### Key Insight and Hypothesis

**Key insight:** many state-tracking tasks in the Chomsky-hierarchy evaluations (e.g., parity, modular arithmetic without brackets) can be implemented by a small finite-state machine, and in the complex-SSM view can be represented by a single oscillator (one phase).

**Hypothesis:** tying Mamba-3’s data-dependent rotation angles across all state pairs (a single shared angle per head) will preserve Mamba-3’s performance on single-counter tasks (parity, modular arithmetic without brackets), but will fail on multi-counter tasks that require tracking multiple independent modular states.

This could be wrong if per-pair angles are required for optimization stability even on simple tasks, or if the model can compensate for shared angles using other layers/heads.

---

## Proposed Approach

### Overview

We propose **ScalarPhase Mamba-3**, a constrained variant of Mamba-3 that keeps the **data-dependent** RoPE mechanism but reduces its phase degrees of freedom:

- **Baseline (FullPhase):** Mamba-3 with per-pair angles θ_t[i] as in the paper.
- **ScalarPhase (ours):** compute θ_vec(t) exactly as in baseline, but replace it by a **shared scalar** θ_shared(t) and apply the same 2×2 rotation to every state pair.

To keep the comparison interpretable, we use a **parameter-matched tying**: compute θ_vec(t)=W_θ h_t (unchanged), then set θ_shared(t)=mean_i θ_vec(t)[i]. This preserves parameter count and most compute in the θ path; only the RoPE application changes.

### Method Details

Let Mamba-3’s RoPE-trick apply cumulative rotations (∏_{i=0..t} R_i^T) to the B and C projections, where each R_t is block-diagonal with 2×2 rotations R(Δt·θ_t[i]).

**ScalarPhase modification:**
1. Compute θ_vec(t) ∈ R^{N/2} as in baseline.
2. Compute θ_shared(t) = mean(θ_vec(t)).
3. Construct R_t = Block{ R(Δt·θ_shared(t)) } for all pairs.
4. Apply the same RoPE-trick recurrence as Mamba-3.

### Key Innovations

1. **A minimal-complexification test for Mamba-3**: isolates whether Mamba-3’s state-tracking gains require many independent phase channels or only a single shared oscillator.
2. **Boundary-condition evaluation**: adds a multi-counter synthetic task (2-parity) to test when per-pair angles become necessary.

---

## Related Work

### Field Overview

A growing literature studies the expressivity and failure modes of sub-quadratic sequence models (SSMs and linear attention) on algorithmic tasks and long-context behavior. Recent work shows that restricting transition eigenvalues to nonnegative reals can fundamentally prevent parity-like state tracking, while allowing negative or complex eigenvalues can recover these capabilities. Separately, multiple works analyze RoPE’s frequency structure and numerical limits, motivating investigations into the effective dimensionality of rotational positional mechanisms.

### Related Papers

- **[Mamba-3: Improved Sequence Modeling Using State Space Principles](./references/Under-review-as-a-conference-paper-at-ICLR-2026-MAMBA-3-IMPROVED-SEQUENCE-MODELING-USING-STATE-SPACE-PRINCIPLES/meta/meta_info.txt)**: Introduces trapezoidal discretization, complex-valued SSMs equivalent to data-dependent RoPE on B/C, and MIMO for inference efficiency.
- **[Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](./references/Unlocking-State-Tracking-in-Linear-RNNs-Through-Negative-Eigenvalues/meta/meta_info.txt)**: Shows negative eigenvalues enable parity and improve modular arithmetic length generalization; provides the formal-language evaluation protocol and scaled accuracy metric.
- **[Neural Networks and the Chomsky Hierarchy](https://arxiv.org/abs/2207.02098)**: Introduces formal language tasks (parity, modular arithmetic with/without brackets) and emphasizes length generalization evaluation.
- **[xLSTM: Extended Long Short-Term Memory](<../../../papers/paper_summaries/xLSTM Extended Long Short-Term Memory.md>)** (arXiv:2405.04517): Shows LSTM variants solve parity and modular arithmetic tasks that many linear-time models fail.
- **[The Illusion of State in State-Space Models](<../../../papers/paper_summaries/The Illusion of State in State-Space Models.md>)**: Proves many practical SSMs (incl. diagonal/scan-based) are in **TC⁰** under finite precision, explaining failures on certain state-tracking tasks (TC⁰ ≈ constant-depth threshold circuits: a highly-parallel computation class).
- **[The Expressive Capacity of State Space Models: A Formal Language Perspective](<../../../papers/paper_summaries/The Expressive Capacity of State Space Models A Formal Language Perspective.md>)**: Formal-language view of SSM strengths/limitations; proves nonnegative gating prevents parity at arbitrary lengths under finite precision.
- **[Selective Rotary Position Embedding](./references/Selective-Rotary-Position-Embedding/meta/meta_info.txt)**: Proposes learnable input-dependent rotary embeddings and argues rotation+decay are both needed; provides implementation details for data-dependent angle accumulation.
- **[The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval](./references/The-Rotary-Position-Embedding-May-Cause-Dimension-Inefficiency-in-Attention-Heads-for-Long-Distance-Retrieval/meta/meta_info.txt)**: Empirically shows some RoPE dimensions are underutilized in retrieval heads, suggesting redundant rotational degrees-of-freedom.
- **[Rotary Positional Embeddings as Phase Modulation: Theoretical Bounds on the RoPE Base for Long-Context Transformers](./references/Rotary-Positional-Embeddings-as-Phase-Modulation-Theoretical-Bounds-on-the-RoPE-Base-for-Long-Context-Transformers/meta/meta_info.txt)**: Derives stability/precision bounds for RoPE phase accumulation, relevant to cumulative rotation mechanisms.
- **[RAP: KV-Cache Compression via RoPE-Aligned Pruning](./references/RAP-KV-Cache-Compression-via-RoPE-Aligned-Pruning/meta/meta_info.txt)**: Shows RoPE’s 2×2 structure constrains compression and motivates structured “pair-level” reasoning about rotary dimensions.
- **[PaTH Attention: Position encoding via accumulating householder transformations](https://arxiv.org/abs/2505.16381)**: Uses accumulated orthogonal transforms for position encoding, related to cumulative rotation mechanisms.
- **[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)**: Introduces standard (data-independent) RoPE.
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](<../../../papers/paper_summaries/Mamba Linear-Time Sequence Modeling with Selective State Spaces.md>)** (arXiv:2312.00752): Introduces selective SSMs as linear-time sequence mixers.
- **[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)](<../../../papers/paper_summaries/Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality.md>)** (arXiv:2405.21060): Introduces SSD formalization and the Mamba-2 scalar-transition SSM for efficiency.
- **[Gated Delta Networks: Improving Mamba2 with Delta Rule](<../../../papers/paper_summaries/Gated Delta Networks Improving Mamba2 with Delta Rule.md>)** (arXiv:2412.06464): Linear-time model family; reports strong long-context + retrieval results and a parity-capable eigenvalue-range variant.
- **[Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet)](<../../../papers/paper_summaries/Parallelizing Linear Transformers with the Delta Rule over Sequence Length.md>)** (arXiv:2406.06484): Delta-rule linear attention; strong associative recall and competitive LM performance.
- **[S4](https://arxiv.org/abs/2111.00396)**: Earlier SSM family with complex parameterizations.
- **[S4D](https://arxiv.org/abs/2206.11893)**: Diagonal SSM variant; often used as efficient baseline.
- **[Zoology](https://arxiv.org/abs/2312.04927)**: Synthetic recall benchmarks used to evaluate efficient sequence models.
- **[Is Mamba Capable of In-Context Learning?](<../../../papers/paper_summaries/Is Mamba Capable of In-Context Learning.md>)** (arXiv:2402.03170): Empirical analysis showing Mamba can perform in-context learning comparably to transformers on multiple tasks.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| State-space / linear mixers | Recurrent state update enabling linear-time mixing | Mamba, Mamba-2, Gated DeltaNet, S4/S4D | LM perplexity, Chomsky tasks, recall tasks | Can fail state tracking under restricted eigenvalues |
| Complex/rotational dynamics | Enable oscillatory modes via complex or orthogonal transitions | S4 (complex), Mamba-3 (data-dependent RoPE), negative-eigs LRNNs | Parity, modular arithmetic | Added complexity; unclear minimality |
| RoPE analysis/optimization | Study or modify rotary embeddings and their effective dimensionality | Selective RoPE, RoPE dimension inefficiency, phase-modulation bounds, RAP | Long-context QA, RULER/LongBench, theory | Many results in transformer setting; less in SSMs |

### Closest Prior Work

- **Mamba-3**: Demonstrates that data-dependent RoPE (derived from complex SSMs) enables parity/modular arithmetic, but does not test whether per-pair angles are necessary.
- **Selective RoPE**: Studies learnable input-dependent rotary embeddings in attention/linear transformers, but does not address Mamba-3’s specific complex-SSM RoPE-trick or angle tying.
- **RoPE dimension inefficiency**: Shows some rotary dimensions are underutilized in transformer retrieval heads; suggests “partial rotation” may be viable, but does not test state-tracking in linear SSMs.

**Novelty Kill Search Summary:** Searched for “shared angle RoPE”, “scalar RoPE angle”, “tie rotary angles”, “Mamba-3 shared theta”, and grepped local drafts/finalized proposals for these phrases. No prior work directly testing shared-angle data-dependent RoPE in Mamba-3-like SSMs was found as of 2026-03-01.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Mamba-3 | Data-dependent RoPE via complex SSM; solves parity/mod arithmetic | No minimality test of phase dimensionality | Tie angles across pairs | Identifies whether phase diversity is necessary |
| Negative-eigs LRNNs | Negative eigenvalues enable parity in LRNNs | Not Mamba-3; no complex RoPE-trick | Study complex-RoPE minimality | Clarifies relation between oscillatory dynamics and counters |
| Selective RoPE | Input-dependent angles for attention/linear transformers | Different architecture; no Mamba-3 | Apply “angle tying” in SSM RoPE-trick | Tests effective number of phase channels |

---

## Experiments

### Experimental Setup

**Task suite (main):** Formal-language length generalization tasks (the “Chomsky hierarchy” suite popularized by Deletang et al., 2023) used by negative-eigenvalues LRNNs and Mamba-3.
- Parity
- Modular arithmetic (without brackets)
- Modular arithmetic (with brackets)

**Boundary test (multi-counter):** 2-parity (Z2×Z2)
- Input alphabet: tokens are pairs of bits, e.g., {00,01,10,11} (or two-channel one-hot).
- Target: 2-bit output (⊕_t a_t, ⊕_t b_t) at the final position.
- Rationale: requires tracking two independent mod-2 counters but is still a regular language (no stack).

**Protocol (match published baselines):**
- Train lengths: uniformly sample L∼Unif{3,…,40}.
- Test lengths: evaluate on L∈{40,…,256}.
- Metric: **scaled accuracy** if using the same generator/eval harness as the baselines; otherwise report plain accuracy plus the random-guess baseline explicitly.
  - Scaled accuracy is typically the chance-normalized score: `scaled = (acc - acc_chance) / (1 - acc_chance)` (so chance→0, perfect→1), with `acc_chance = 1/|Y|` for |Y| classes.

**Model variants (3 conditions; decisive):**
1. **FullPhase Mamba-3** (baseline): per-pair θ_t[i] (original)
2. **ScalarPhase Mamba-3** (ours): θ_shared(t)=mean_i θ_t[i]
3. **NoRoPE Mamba-3** (sanity): remove data-dependent rotations entirely (expected to fail on parity)

**Implementation note:** If official Mamba-3 code is unavailable, use a public PyTorch reimplementation and ensure FullPhase reproduces Table 4b trends before comparing.

**Training hyperparams (from Mamba-3 reference; see `References.md` b61):**
- Sequence-length curriculum: 3–40 → 160, evaluate at 256.
- Steps: 1e4 per curriculum stage; batch size 256.
- Depth: 1 layer for parity; 3 layers for modular-arithmetic tasks.
- State size: d_state = 64.
- Sweep: d_model ∈ {32,64} and 8 learning rates (log-spaced; exact endpoints not specified in scraped text).

**Extra rigor (lightweight): head-level ablation (optional if budget allows)**
- Repeat parity and 2-parity with **n_heads=1** (single head) to reduce the chance that multiple heads emulate multiple oscillators.
- If compute is tight, run this only on the decisive boundary test (2-parity).

**Resource Estimate**:
- **Compute budget**: ≤ 256 GPU-hours (synthetic tasks; small models)
- **GPU memory**: ≤ 40GB
- **API usage**: none

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Parity | Predict parity of bit sequence; tests state tracking | scaled accuracy / accuracy | test | From task generator repo | TBD (use repo script) |
| Mod arith (no brackets) | Evaluate expression modulo m without parentheses | scaled accuracy / accuracy | test | From task generator repo | TBD |
| Mod arith (with brackets) | Evaluate bracketed expression modulo m | scaled accuracy / accuracy | test | From task generator repo | TBD |
| 2-parity | Two independent parities; multi-counter boundary | accuracy | test | synthetic generator | simple script |

### Existing Baseline Results (from prior work)

These numbers motivate expected effect sizes and sanity-check reproduction; verification should still re-run all methods under a unified codebase.

**Mamba-3 (Table 4b; scaled accuracy, %, higher is better)** — see `./references/Under-review-as-a-conference-paper-at-ICLR-2026-MAMBA-3-IMPROVED-SEQUENCE-MODELING-USING-STATE-SPACE-PRINCIPLES/sections/SSM-CENTRIC METHODOLOGICAL ABLATIONS.md`:

| Model | Parity | Mod arith (no brackets) | Mod arith (with brackets) |
|---|---:|---:|---:|
| Mamba-3 (FullPhase; data-dependent RoPE) | 100.00 | 98.51 | 87.75 |
| Mamba-3 (w/o RoPE) | 2.27 | 1.49 | 0.72 |
| Mamba-3 (w/ standard RoPE) | 1.56 | 20.70 | 2.62 |
| Mamba-2 | 0.90 | 47.81 | 0.88 |
| Gated DeltaNet (eigs [-1,1]) | 100.00 | 99.25 | 93.50 |

**Negative-eigenvalues LRNNs (Table 3; scaled accuracy in [0,1])** — see `./references/Unlocking-State-Tracking-in-Linear-RNNs-Through-Negative-Eigenvalues/sections/5.1 Chomsky Hierarchy.md`:

| Model | Parity | Mod arith (no brackets) | Mod arith (with brackets) |
|---|---:|---:|---:|
| Transformer | 0.022 | 0.031 | 0.025 |
| mLSTM | 0.087 (0.04) | 0.040 (0.04) | 0.034 (0.03) |
| sLSTM | 1.000 (1.00) | 0.787 (1.00) | 0.173 (0.57) |
| Mamba [0,1] | 0.000 | 0.095 | 0.092 |
| Mamba [-1,1] | 1.000 | 0.241 | 0.136 |
| DeltaNet [0,1] | 0.017 | 0.314 | 0.137 |
| DeltaNet [-1,1] | 1.000 | 0.971 | 0.200 |

### Main Results (to be produced by verification)

(All results TBD; to be filled by verification runs.)

| Method | Base Model | Benchmark | Metric (mean±std) | Source | Notes |
|---|---|---|---|---|---|
| FullPhase Mamba-3 | Mamba-3 small | Parity | TBD | - | target reproduces ~1.0 |
| ScalarPhase (ours) | Mamba-3 small | Parity | TBD | - | hypothesize near FullPhase |
| NoRoPE | Mamba-3 small | Parity | TBD | - | expected near random |
| FullPhase Mamba-3 | Mamba-3 small | 2-parity | TBD | - | hypothesize high |
| ScalarPhase (ours) | Mamba-3 small | 2-parity | TBD | - | hypothesize near chance |

### Ablation Studies

None beyond the 3 conditions above (keep decisiveness high).

### Experimental Rigor

- **Variance & reproducibility**: Run ≥3 seeds (prefer 5 if budget allows), report mean±std.
- **Controls**: Verify NoRoPE is near random on parity; verify FullPhase reproduces Mamba-3 qualitative result (parity solves, standard RoPE fails if implemented).

---

## Success Criteria

**Hypothesis:** ScalarPhase matches FullPhase on single-counter tasks (parity, modular arithmetic without brackets), but significantly underperforms on 2-parity.

**Decision Rule**:
- **Proceed (supports hypothesis)** if:
  - On parity and modular arithmetic without brackets, ScalarPhase is within **2 scaled-accuracy points** of FullPhase (or within 1 std across seeds), AND
  - On 2-parity, ScalarPhase is **≤55% accuracy** while FullPhase is **≥80% accuracy** (at the same training length budget).
- **Pivot** if ScalarPhase is worse on all tasks: test a weaker tying (e.g., per-head scalar but per-layer different) or conclude per-pair angles are necessary even for simple tasks.
- **Refute** if ScalarPhase matches FullPhase on 2-parity as well (suggesting the model can implement multiple counters without per-pair angles in this setting).

---

## Impact Statement

If ScalarPhase works, it clarifies that Mamba-3’s state-tracking gains do not require high-dimensional phase channels, suggesting simpler and potentially more efficient complexifications for linear-time SSMs. If it fails, the negative result concretely identifies a boundary: per-pair phase diversity is necessary to implement multiple independent counters, informing future SSM design and analysis.

---

## References

- [Mamba-3: Improved Sequence Modeling Using State Space Principles](./references/Under-review-as-a-conference-paper-at-ICLR-2026-MAMBA-3-IMPROVED-SEQUENCE-MODELING-USING-STATE-SPACE-PRINCIPLES/meta/meta_info.txt) - 2026
- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](./references/Unlocking-State-Tracking-in-Linear-RNNs-Through-Negative-Eigenvalues/meta/meta_info.txt) - 2025
- [Selective Rotary Position Embedding](./references/Selective-Rotary-Position-Embedding/meta/meta_info.txt) - 2025
- [The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval](./references/The-Rotary-Position-Embedding-May-Cause-Dimension-Inefficiency-in-Attention-Heads-for-Long-Distance-Retrieval/meta/meta_info.txt) - 2025
- [Rotary Positional Embeddings as Phase Modulation: Theoretical Bounds on the RoPE Base for Long-Context Transformers](./references/Rotary-Positional-Embeddings-as-Phase-Modulation-Theoretical-Bounds-on-the-RoPE-Base-for-Long-Context-Transformers/meta/meta_info.txt) - 2026
- [RAP: KV-Cache Compression via RoPE-Aligned Pruning](./references/RAP-KV-Cache-Compression-via-RoPE-Aligned-Pruning/meta/meta_info.txt) - 2026
- [Neural Networks and the Chomsky Hierarchy](https://arxiv.org/abs/2207.02098) - 2023
- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) - 2024
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - 2021
- [Mamba](https://arxiv.org/abs/2312.00752) - 2023
