# untitled

# Exact-Substep Exponential Integrator for Diagonal-Decay Delta Attention

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Scaling Transformer context lengths is increasingly important for practical systems (code assistants, agents, long-document QA), but standard softmax attention has **quadratic** time/memory cost in sequence length. This motivates **sub-quadratic attention layers**, including (i) kernelized **linear attention** and (ii) **state-space models (SSMs)** that update a fixed-size recurrent state.

A recent line of work has made *delta-rule / fast-weight* linear attention competitive at large scale by updating a per-head matrix state with an error-correction rule. (Here “fast weights” refers to the classic view of attention as writing to a rapidly-updated associative memory matrix each token.) For example, **Kimi Linear** introduces *Kimi Delta Attention (KDA)*, a diagonal-decay delta attention mechanism that is reported to outperform full attention consistently at production scale while remaining linear-time in sequence length ([Kimi Linear](./references/KIMI-LINEAR-AN-EXPRESSIVE-EFFICIENT-ATTENTION-ARCHITECTURE/meta/meta_info.txt)).

Despite these gains, many delta-rule attention variants still implement the underlying continuous-time dynamics using a **first-order explicit Euler discretization** (i.e., a “single-step” update with step size \(\beta_t\)). In practice, stability is often enforced via architectural heuristics such as **L2-normalizing keys/queries**, which can restrict expressivity and may be brittle under distribution shift or length extrapolation.

### The Problem

Delta-rule attention can be written as a rank-1 contraction along the current key direction. In KDA, each token update combines (i) a **diagonal per-channel decay** (forget gate) and (ii) a **delta-rule correction**. Kimi Linear applies **L2Norm** to both keys and queries “to ensure eigenvalues stability” ([Kimi Linear](./references/KIMI-LINEAR-AN-EXPRESSIVE-EFFICIENT-ATTENTION-ARCHITECTURE/meta/meta_info.txt), architecture section), implying that without normalization, the update can become numerically unstable.

Meanwhile, **Error-Free Linear Attention (EFLA)** shows that for the *rank-1* delta-rule dynamics, the explicit Euler update is an approximation to a continuous-time ODE and can be replaced by a **closed-form exponential-integrator coefficient** computed in \(\mathcal{O}(d_k)\) time ([EFLA](./references/Error-Free-Linear-Attention-is-a-Free-Lunch-Exact-Solution-from-Continuous-Time-Dynamics/meta/meta_info.txt)). EFLA reports that this “exact” rank-1 integrator improves language modeling and robustness over Euler-style delta attention.

However, EFLA derives the exact update for **vanilla delta-rule attention** (no diagonal decay), while KDA’s strongest form includes a **diagonal decay operator** and relies on normalization for stability. It is unclear whether the “integrator choice” benefit can be transferred to diagonal-decay delta attention in a way that is both (a) mathematically well-defined and (b) practically useful.

### Key Insight and Hypothesis

**Key insight:** KDA already applies its diagonal decay *explicitly* to the previous state. If we treat KDA as a **two-substep composition** (decay substep followed by delta-rule substep), then we can apply EFLA’s **exact rank-1 exponential integrator** to the delta-rule substep *without* claiming to solve a full non-commuting \(\exp(D + uu^\top)\) system.

Concretely, we hypothesize that replacing the Euler coefficient \(\beta_t\) in KDA’s rank-1 delta update with EFLA’s exponential-integrator coefficient
\[
\tilde\alpha_t = \frac{1 - \exp(-\beta_t \lVert k_t \rVert^2)}{\lVert k_t \rVert^2}
\]
will make KDA stable when \(\lVert k_t \rVert\) varies, enabling **removal of key/query L2-normalization** while maintaining (or improving) long-context accuracy.

Why we could be wrong: (i) with L2-normalized keys, Euler may already be stable enough that the integrator-only control gives no gains; (ii) during training without normalization, the model may learn to keep \(\lVert k_t \rVert\) small, collapsing the hypothesized “signal-strength channel”; (iii) operator-splitting error from composing decay and delta substeps may negate benefits.

---

## Proposed Approach

### Overview

We propose **Exact-Substep KDA**, a modification of diagonal-decay delta attention that keeps KDA’s architecture and decay gate, but replaces the **delta-rule substep** with the **closed-form exponential integrator** from EFLA. The method is intended as a drop-in replacement for the per-token state update and should add negligible runtime overhead (one `expm1` per token/head).

Importantly, we are precise about what is “exact”: the exponential integrator is **exact for the rank-1 delta-rule ODE substep** under a *piecewise-constant* assumption on \((k_t, v_t)\) within each token update (a standard simplification when deriving discrete updates from continuous-time dynamics). The overall KDA recurrence (decay composed with delta update) remains a **first-order operator splitting** when the two operators do not commute.

### Method Details

**Notation (per attention head):**
- Key \(k_t \in \mathbb{R}^{d_k}\), query \(q_t \in \mathbb{R}^{d_k}\), value \(v_t \in \mathbb{R}^{d_v}\)
- State matrix \(S_t \in \mathbb{R}^{d_k \times d_v}\)
- Diagonal decay gate \(D_t = \mathrm{Diag}(\alpha^{\mathrm{gate}}_t)\) with \(\alpha^{\mathrm{gate}}_t \in [0,1]^{d_k}\)
- Step size \(\beta_t \in [0,1]\)

**KDA (Euler-style) update (as reported):**
\[
S_t = (I - \beta_t k_t k_t^\top)\, D_t S_{t-1} + \beta_t k_t v_t^\top.
\]

**Exact delta-rule substep (EFLA):** Consider the continuous-time ODE for the delta-rule dynamics with constant \((k_t, v_t)\) over the step:
\[
\frac{dS}{d\tau} = -k_t k_t^\top S + k_t v_t^\top, \qquad S(0)=\tilde S_{t-1}.
\]
Because \(k_t k_t^\top\) is rank-1 with eigenvalue \(\lambda_t = \lVert k_t \rVert^2\), the matrix exponential \(\exp(-\beta_t k_t k_t^\top)\) has a closed form, yielding the exact discrete-time update:
\[
S(\beta_t) = (I - \tilde\alpha_t k_t k_t^\top)\, \tilde S_{t-1} + \tilde\alpha_t k_t v_t^\top,
\]
where
\[
\tilde\alpha_t = \frac{1 - \exp(-\beta_t \lambda_t)}{\lambda_t}.
\]

**Exact-Substep KDA (proposed):** Apply KDA’s decay explicitly then apply the exact delta substep:
\[
\tilde S_{t-1} := D_t S_{t-1}, \qquad
S_t := (I - \tilde\alpha_t k_t k_t^\top)\, \tilde S_{t-1} + \tilde\alpha_t k_t v_t^\top.
\]

**Implementation details (numerical stability):**
- Compute \(\tilde\alpha_t\) using `expm1` in fp32:
  \(\tilde\alpha_t = -\mathrm{expm1}(-\beta_t \lambda_t) / \max(\lambda_t, \epsilon)\).
- Use the limit \(\tilde\alpha_t \leftarrow \beta_t\) when \(\lambda_t < \epsilon\) (since \(\lim_{\lambda\to 0} (1-e^{-\beta\lambda})/\lambda = \beta\)).

**Norm-free keys:** Our primary experimental variant removes L2 normalization on \(q_t, k_t\) used in Kimi Linear, to test whether the integrator can maintain stability and improve length generalization when \(\lVert k_t \rVert\) varies.

### Key Innovations

1. **Transfers EFLA’s closed-form integrator to diagonal-decay delta attention** by making explicit what is integrated exactly (the rank-1 delta substep), avoiding hand-wavy claims about exponentials of diagonal-plus-low-rank operators.
2. **A controlled test of whether exponential integration enables removing key normalization** in a state-of-the-art delta attention mechanism (KDA), potentially restoring a learnable key-norm “strength” channel.

---

## Related Work

### Field Overview

**Efficient long-context sequence modeling.** Linear-time alternatives to softmax attention include kernelized linear attention (approximating softmax with random features or other kernels) and SSMs that update a fixed-size state. Many approaches struggle with long-range retrieval and copying, motivating benchmarks such as RULER, LongBench, and synthetic recall tests.

**Delta-rule / fast-weight attention.** Delta-rule attention updates a per-head state matrix using an error-correction term, making it more retrieval-capable than purely multiplicative-decay SSMs in some regimes. Recent work improves delta-rule attention with better parameterizations (e.g., diagonal decay) and better training efficiency (e.g., chunk-parallel algorithms).

**Continuous-time perspectives and integrator choice.** Several works interpret discrete sequence models as discretizations of continuous-time dynamics (Neural ODEs, continuous-time Transformers, or SSM discretization). EFLA highlights that for rank-1 delta-rule dynamics, the exact exponential integrator is available in closed form and can improve stability/quality.

### Related Papers

- **[Error-Free Linear Attention is a Free Lunch: Exact Solution from Continuous-Time Dynamics](./references/Error-Free-Linear-Attention-is-a-Free-Lunch-Exact-Solution-from-Continuous-Time-Dynamics/meta/meta_info.txt)**: Derives a closed-form exponential-integrator update for rank-1 delta-rule attention and reports robustness/LM gains over Euler discretization.
- **[Kimi Linear: An Expressive, Efficient Attention Architecture](./references/KIMI-LINEAR-AN-EXPRESSIVE-EFFICIENT-ATTENTION-ARCHITECTURE/meta/meta_info.txt)**: Introduces KDA (diagonal-decay delta attention) and reports strong long-context results but relies on L2-normalized keys/queries for stability.
- **[Gated Delta Networks: Improving Mamba2 with Delta Rule](./references/Gated-Delta-Networks-Improving-Mamba2-with-Delta-Rule/meta/meta_info.txt)**: Improves delta-rule attention with gating/decay and reports strong retrieval results, but uses Euler-style discretization.
- **[Parallelizing Linear Transformers with the Delta Rule over Sequence Length](./references/Parallelizing-Linear-Transformers-with-the-Delta-Rule-over-Sequence-Length/meta/meta_info.txt)**: Provides a chunkwise-parallel algorithm for delta-rule attention to improve training efficiency.
- **[Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)**: Introduces hardware-efficient kernels/training for linear attention variants.
- **[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)**: Provides SSM–attention duality and motivates Mamba2-style sequence models.
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)**: A widely used SSM approach for long sequences, often compared against delta-rule linear attention.
- **[S4: Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)**: Foundational structured SSM model family for long-range sequence modeling.
- **[S5: Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933)**: Simplifies S4-like layers and analyzes discretization choices.
- **[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)**: Foundational continuous-time modeling paper motivating integrator choice in neural networks.
- **[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)**: Introduces recall-focused synthetic tasks correlated with LM performance, including associative recall.
- **[Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032)**: Studies copying as a failure mode for state-space/linear models and motivates strong long-context retrieval diagnostics.
- **[Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)**: Shows that allowing negative eigenvalues improves state-tracking in linear recurrent sequence models, highlighting that stability/expressivity trade-offs in efficient architectures can depend on seemingly small numerical design choices.
- **[Performer: Rethinking Attention with Linear Complexity](https://arxiv.org/abs/2009.14794)**: Random-feature approximation enabling linear-time attention.
- **[Linear Transformers are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)**: Connects linear attention to fast-weight memory updates.
- **[Fast Weight Programmers](https://arxiv.org/abs/2104.03604)**: Classic fast-weight memory formulation related to delta-rule updates.
- **[RetNet: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)**: A retention-based sub-quadratic architecture compared with other efficient sequence models.
- **[RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)**: Hybrid RNN/attention approach for efficient sequence modeling.
- **[Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)**: Long-convolution alternative for long-context modeling.
- **[LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508)**: Standard long-context benchmark used in many comparisons of efficient architectures.


### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Kernelized linear attention | Approximate softmax with feature maps for linear time | Performer; Linear Transformers/fast weights | LongBench, LRA, synthetic recall | Approximation error; retrieval quality at long lengths |
| SSM sequence models | Recurrent state update with structured transitions | S4, S5, Mamba/Mamba2 | LongBench, Wikitext PPL, synthetic tasks | Can underperform on associative recall/copying |
| Delta-rule attention | Error-correction update to improve retrieval | DeltaNet/Delta-rule linear attention; Gated Delta Networks; KDA | RULER, synthetic recall, LongBench | Training stability; sequential update unless chunked |
| Integrator choice / continuous-time view | Replace Euler discretization with higher-order/exact updates | Neural ODEs; EFLA | Robustness, long-horizon stability | Exact forms rare; overhead concerns |

### Closest Prior Work

1. **EFLA** ([paper](./references/Error-Free-Linear-Attention-is-a-Free-Lunch-Exact-Solution-from-Continuous-Time-Dynamics/meta/meta_info.txt)) derives the closed-form exponential integrator for rank-1 delta-rule attention and shows empirical benefits, but does not address diagonal decay as used in KDA.
2. **Kimi Linear / KDA** ([paper](./references/KIMI-LINEAR-AN-EXPRESSIVE-EFFICIENT-ATTENTION-ARCHITECTURE/meta/meta_info.txt)) introduces diagonal decay and reports strong long-context results, but requires L2-normalized keys/queries for stability.
3. **Gated Delta Networks** ([paper](./references/Gated-Delta-Networks-Improving-Mamba2-with-Delta-Rule/meta/meta_info.txt)) improves delta-rule attention with gating/decay and reports strong retrieval accuracy, but retains Euler-style discretization.
4. **Delta-rule chunk parallelization** ([paper](./references/Parallelizing-Linear-Transformers-with-the-Delta-Rule-over-Sequence-Length/meta/meta_info.txt)) addresses training efficiency and parallelization, orthogonal to integrator choice.

**Novelty Kill Search Summary:** Searched for combinations of “Error-Free Linear Attention / EFLA” with “Kimi Linear / KDA / diagonal decay delta attention”, and for “exponential integrator delta attention” and related phrases, and found no paper explicitly applying the EFLA coefficient to KDA-style diagonal-decay delta attention as of 2026-02-20. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [Kimi Linear / KDA](./references/KIMI-LINEAR-AN-EXPRESSIVE-EFFICIENT-ATTENTION-ARCHITECTURE/meta/meta_info.txt) | Diagonal decay + delta rule; strong long-context results | Requires L2Norm(q,k) for stability | Replace Euler delta substep with EFLA exponential integrator | Bounded contraction when \|k\| varies → stable without normalization |
| [EFLA](./references/Error-Free-Linear-Attention-is-a-Free-Lunch-Exact-Solution-from-Continuous-Time-Dynamics/meta/meta_info.txt) | Exact integrator for rank-1 delta attention | Does not handle diagonal decay mechanisms | Apply exact integrator inside KDA’s delta substep | Transfers integrator benefits to stronger KDA parameterization |
| [Gated Delta Networks](./references/Gated-Delta-Networks-Improving-Mamba2-with-Delta-Rule/meta/meta_info.txt) | Adds gating/decay to delta rule | Still Euler discretization; stability heuristics remain | Keep decay but change integrator | Improves numerical behavior without adding extra parameters |
| [Chunk-parallel delta rule](./references/Parallelizing-Linear-Transformers-with-the-Delta-Rule-over-Sequence-Length/meta/meta_info.txt) | Improves efficiency of delta-rule training | Orthogonal to stability/normalization | No change (compatible) | Proposed update should work with chunkwise implementations |

---

## Experiments

### Experimental Setup

We target a minimal, fully automated experiment that isolates whether exponential integration enables **norm-free** KDA and whether it improves length generalization.

**Benchmark choice:** We use the three **synthetic long-context tests** used by Kimi Linear (Palindrome, MQAR, Stack) ([Kimi Linear synthetic tests](./references/KIMI-LINEAR-AN-EXPRESSIVE-EFFICIENT-ATTENTION-ARCHITECTURE/meta/meta_info.txt)). These tasks are lightweight, reproducible, and directly stress long-range retrieval/state tracking.

**Baseline Ladder (REQUIRED):**
- **Trivial baseline**: random guessing (expected near-chance accuracy; sanity check)
- **Closest method baseline**: KDA (Euler) with L2Norm(q,k) (condition 1)
- **Integrator-only control**: KDA (exponential-integrator substep) with L2Norm(q,k) (condition 2)
- **Proposed method**: KDA (exponential-integrator substep) without L2Norm(q,k) (condition 3)
- **Optional stress baseline (not in main results table)**: KDA (Euler) without L2Norm(q,k), to confirm the expected instability/failure mode and separate “stability enabling” from generic hyperparameter effects.

Prompting and inference-time scaling baselines are not applicable because this is supervised training of small models on synthetic tasks (not LLM prompting).

**Model configuration (match Kimi Linear synthetic tests):**
- 2 layers, 2 attention heads
- Head dimension \(d_k=128\); value dimension \(d_v\) as in Kimi synthetic tests (if unspecified, set \(d_v=d_k\))
- Training steps: 20,000
- Optimizer: AdamW (default betas), weight decay 0.01
- Learning rate: start with Kimi’s grid {5e-5, 1e-4, 5e-4, 1e-3} and select best on a held-out generated evaluation batch at step 5k (early stop for poor runs)

**Seeds / variance plan:** run all 3 main conditions with `seeds=[42,123,456]` and report mean ± std.

**Length generalization protocol:**
- Train with fixed context length \(L_{train}=1024\).
- Evaluate at \(L_{test}\in\{1024, 2048, 4096\}\). Primary metric is accuracy at 4096.

**Mechanism diagnostics (fully automated, logged):**
- Distribution of \(\lVert k_t \rVert\) and \(\beta_t \lVert k_t \rVert^2\) during training
- Frequency of NaNs/divergence
- Norm of state \(\lVert S_t \rVert_F\) over sequence position

**Resource Estimate**:
- **Compute budget**: expected \(<100\) GPU-hours total (small 2-layer models; synthetic data). Conservative upper bound: 3 tasks × 3 conditions × 3 seeds × 20k steps with small LR sweep fits well within the 768 GPU-hour cap.
- **GPU memory**: \(<10\) GB per run (single A100 sufficient).
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Palindrome | Copying/reversal task stressing long-range retrieval | Token accuracy (%) | generated | N/A (synthetic) | Implement generator from Kimi paper; optionally cross-check KDA impl in https://github.com/MoonshotAI/Kimi-Linear |
| MQAR | Multi-query associative recall (key–value retrieval at varying positions) | Token accuracy (%) | generated | N/A (synthetic) | Reuse MQAR generator from https://github.com/HazyResearch/zoology (or implement from description in Kimi paper) |
| Stack | LIFO state tracking with multiple stacks | Token accuracy (%) | generated | N/A (synthetic) | Implement generator from Kimi paper (64 stacks; push/pop tokens) |

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Accuracy @1024 (mean±std) | Accuracy @2048 (mean±std) | Accuracy @4096 (mean±std) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| KDA (Euler) + L2Norm(q,k) | 2L-2H-128 | Palindrome/MQAR/Stack | **TBD** | **TBD** | **TBD** | - | closest baseline |
| KDA (Exp-substep) + L2Norm(q,k) | 2L-2H-128 | Palindrome/MQAR/Stack | **TBD** | **TBD** | **TBD** | - | integrator-only control |
| **Exact-Substep KDA (Exp-substep) w/o L2Norm(q,k)** | 2L-2H-128 | Palindrome/MQAR/Stack | **TBD** | **TBD** | **TBD** | - | proposed method |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (full) | Exp-substep, no L2Norm(q,k) | Best or comparable; stable |
| Ours + L2Norm(q,k) | Add L2 normalization back (same as condition 2) | If (2)≈(1), gains mainly come from enabling norm removal |

### Experimental Rigor

**Top confounders and controls:**
1. **Hyperparameter sensitivity**: Use the same LR selection procedure across all conditions; optionally fix LR chosen on baseline for a second pass to confirm ordering is not due to tuning.
2. **Instability vs accuracy**: Report both accuracy and numerical failure rates (NaNs/divergence) so “wins” are not due to silent instability.
3. **Length mismatch**: Train all methods at the same \(L_{train}\) and evaluate with the same generation/evaluation protocol.

**Sanity checks:**
- Random baseline at chance
- Reproduce that Euler+L2Norm is stable (baseline)

---

## Success Criteria

**Hypothesis**: Using the exponential-integrator coefficient in KDA’s delta substep will allow stable training without key/query L2-normalization and will improve accuracy under length extrapolation.

**Decision Rule**:
- **Proceed** if at \(L_{test}=4096\), condition (3) exceeds condition (1) by **≥5 percentage points** accuracy on **≥2/3** tasks (Palindrome/MQAR/Stack) across **≥3 seeds**, and does not underperform at \(L_{test}=1024\).
- **Pivot** if condition (2)≈(1) but (3)>(2): treat result as “integrator enables norm-free KDA” rather than a general integrator improvement; focus follow-up on when norm removal helps.
- **Refute** if (3) ≤ (1) within noise across all lengths, or if (3) is numerically unstable (NaNs/divergence) despite the integrator.

---

## Impact Statement

If successful, Exact-Substep KDA would provide a low-overhead, mathematically grounded alternative to heuristic key normalization in delta-rule linear attention. This could simplify the design of long-context-efficient LLM backbones and improve robustness when deploying linear attention models under distribution shift or length extrapolation.

---

## References

- [Error-Free Linear Attention is a Free Lunch: Exact Solution from Continuous-Time Dynamics](./references/Error-Free-Linear-Attention-is-a-Free-Lunch-Exact-Solution-from-Continuous-Time-Dynamics/meta/meta_info.txt)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](./references/KIMI-LINEAR-AN-EXPRESSIVE-EFFICIENT-ATTENTION-ARCHITECTURE/meta/meta_info.txt)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](./references/Gated-Delta-Networks-Improving-Mamba2-with-Delta-Rule/meta/meta_info.txt)
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](./references/Parallelizing-Linear-Transformers-with-the-Delta-Rule-over-Sequence-Length/meta/meta_info.txt)
- [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [S4: Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
- [S5: Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933)
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)
- [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032)
- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
- [Performer: Rethinking Attention with Linear Complexity](https://arxiv.org/abs/2009.14794)
- [Linear Transformers are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)
- [Fast Weight Programmers](https://arxiv.org/abs/2104.03604)
- [RetNet: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508)
