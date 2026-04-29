# untitled

# Parallel Robust Kalman Linear Attention via Innovation-Reweighted Precision

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Efficient sequence models (state space models, recurrent linear attention, and related “sequence mixer” blocks) are increasingly used to scale language models to long contexts and high throughput. However, multiple lines of evidence suggest that **selection under interference / distractors** is a core bottleneck for efficient mixers:

- **Associative recall drives real LM gaps**: Zoology shows that attention-free gated-convolution LMs underperform attention on The Pile, and **82% of the perplexity gap** is explained by failures on “associative recall” tokens (~6.4% of tokens) rather than general modeling capacity (Arora et al., 2023).
- **Correlated distractors break token-local heuristics**: KOSS introduces “context-aware selective copying” with correlated distractors and shows Mamba and S4 collapse under heavy interference (≈13–17% accuracy at 50% interference), while a context-aware selective mechanism reaches 79.2% (Wang et al., 2025).

These findings motivate improving **robust, context-dependent selection** in scan-parallel recurrent mixers.

Kalman Linear Attention (KLA) is a recent sequence mixer that frames sequence modeling as **Bayesian filtering**. Each token provides noisy evidence about a latent state, and KLA maintains an explicit posterior over that state in information form (precision and “information mean”). This provides an interpretable uncertainty-driven gating mechanism and can be implemented in parallel via associative scans.

In classical filtering, robustness to outliers is often achieved by **innovation-based** rules: if the prediction error (innovation) is unusually large relative to its predicted variance, the filter inflates measurement noise to avoid being “overwritten” by inconsistent measurements. We propose to translate this principle to KLA while preserving scan-parallel training.

### The Problem

KLA learns a token-dependent value precision (measurement precision) \(\Lambda^v_t\) that controls how strongly token evidence \(v_t\) updates the latent belief. In KLA’s current design, \(\Lambda^v_t\) is predicted from the token representation (and at most short-range local context, e.g., a small causal convolution). This creates a failure mode in tasks with **context-dependent distractors**: the same token type can be either relevant evidence or misleading distractor depending on information stored in the recurrent state.

In such settings, any measurement-precision rule that depends only on the current token cannot reliably downweight distractors. However, the innovation \(r_t\) (difference between the observed token evidence and the prior prediction from the latent state) is intrinsically **context-dependent** because it depends on the belief state propagated from the past.

This proposal targets the following concrete question: can a scan-parallel KLA layer be made robust to context-dependent distractors by adding a minimal innovation-normalized reweighting step, analogous to one-step IRLS / Student-t robust filtering?

### Key Insight and Hypothesis

**Key insight**: In KLA’s diagonal linear-Gaussian formulation, we can compute the per-token innovation residual and its predicted variance from the pass-1 posterior \((\lambda_t, \eta_t)\). A Student-t / IRLS-inspired weight \(w_t\) based on normalized innovation squared (NIS) can be used to rescale \(\Lambda^v_t\) and rerun KLA.

**Hypothesis**: On synthetic sequence tasks where distractors are locally indistinguishable from true evidence but globally inconsistent with a stored mode/context, a two-pass “innovation-reweighted” KLA will improve accuracy compared to (i) vanilla KLA and (ii) a compute-matched control that uses the same weights but destroys their alignment with innovations (time-permuted weights). We expect negligible degradation on the clean (no-distractor) version of the same task.

---

## Proposed Approach

### Overview

We propose **Parallel Robust KLA (PR-KLA)**: a two-pass variant of KLA.

1. **Pass 1 (standard KLA)**: Run KLA to obtain posterior precision \(\lambda_t\) and information mean \(\eta_t\) (and hence posterior mean \(\mu_t = \eta_t / \lambda_t\) and variance \(\Sigma_t = 1/\lambda_t\)).
2. **Compute innovation weights**: Use the pass-1 posteriors to compute a per-token normalized innovation statistic (NIS) and convert it into a robust weight \(w_t\).
3. **Pass 2 (reweighted KLA)**: Rerun KLA with \(\Lambda^{v\prime}_t = w_t \Lambda^v_t\).

Both passes use scan-parallel implementations (e.g., `torch._higher_order_ops.associative_scan`), so the method preserves KLA’s asymptotic parallel depth (two scans rather than one).

### Method Details

#### Base KLA recap (diagonal information filter)
KLA represents the posterior over latent state \(z_t\) as a diagonal Gaussian:
\[p(z_t\mid v_{1:t}) = \mathcal{N}(\mu_t, \lambda_t^{-1}),\qquad \eta_t := \lambda_t \odot \mu_t.\]
Given token-dependent likelihood parameters (observation operator \(k_t\), evidence \(v_t\), and value precision \(\Lambda^v_t\)), KLA updates the precision by a Möbius (fractional-linear) transformation and updates the information mean affinely (Theorem 1–2 in KLA). In code we will implement KLA’s published diagonal recurrences and perform the prefix scans with PyTorch scan primitives.

#### Innovation and NIS computation (from pass-1 outputs)
For each timestep \(t\), we compute the predictive prior (Ornstein–Uhlenbeck (OU) dynamics; diagonal):
\[\mu^-_t = \bar a \odot \mu_{t-1},\qquad \Sigma^-_t = \bar a^2 \odot \Sigma_{t-1} + p.\]
Then compute innovation residual per head \(n\) and feature \(d\) (consistent with KLA’s multi-channel structure):
\[r_{t,n,d} = v_{t,d} - k_{t,n}\, \mu^-_{t,n,d}.\]
Innovation variance:
\[S_{t,n,d} = k_{t,n}^2\, \Sigma^-_{t,n,d} + (\Lambda^v_{t,d})^{-1}.\]
Aggregate to a per-token/per-feature normalized innovation squared:
\[\mathrm{nis}_{t,d} = \mathrm{mean}_n\left( r_{t,n,d}^2 / S_{t,n,d} \right).\]
Convert to a Student-t inspired robust weight with fixed \(\nu\) (e.g., \(\nu=4\)):
\[w_{t,d} = \frac{\nu+1}{\nu + \mathrm{nis}_{t,d}}.\]
Finally, rescale measurement precision for pass-2:
\[\Lambda^{v\prime}_{t,d} = w_{t,d}\, \Lambda^v_{t,d}.\]
Implementation note: by default we will **stop-gradient through \(w\)** (treat as a non-learned robustification rule). This keeps the training objective aligned with standard cross-entropy while adding a deterministic inductive bias.

#### Compute-matched control (permuted weights)
To distinguish “innovation alignment” from generic regularization, we include a control that uses the same computed weights but **randomly permutes** them across timesteps within each sequence:
\[\tilde w_{t,d} = \mathrm{permute}_t(w_{t,d}),\qquad \tilde\Lambda^{v}_{t,d} = \tilde w_{t,d} \Lambda^{v}_{t,d}.\]
This matches the weight distribution and compute cost, but breaks correlation with innovations.

### Key Innovations

- **Scan-parallel robustification** of KLA via a two-pass innovation-reweighted precision update (one-step IRLS / Student-t reweighting).
- **Compute-matched “permuted weights” control** that cleanly isolates whether gains come from innovation alignment.
- Evaluation on an **established real LM task (WikiText-103 perplexity)**, an **established recall benchmark (MQAR / Zoology)**, plus a targeted **context-dependent distractor variant (Conflict-MQAR)** to isolate the mechanism.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) efficient sequence mixers (linear attention and state space models), (ii) Bayesian filtering views of attention/sequence modeling, (iii) robust state estimation (innovation-based gating, Student-t noise), and (iv) synthetic mechanistic benchmarks for architecture evaluation.

Efficient mixers such as S4, Mamba, and gated linear attention replace quadratic attention with linear-time recurrences, but they often rely on token-local gating or heuristics for selectivity. KLA introduces explicit belief-state uncertainty and nonlinear Möbius (fractional-linear) precision updates, enabling scan-parallel Bayesian filtering as a sequence mixer. Separately, robust filtering literature provides principled ways to downweight outliers using innovation statistics, while Robust Filter Attention applies robust estimation ideas to quadratic attention. Finally, MAD-Lab argues that small synthetic tasks (recall/copy/compression) are predictive unit tests for architecture design.

### Related Papers

- **[Kalman Linear Attention: Parallel Bayesian Filtering For Efficient Language Modelling and State Tracking](./references/Kalman-Linear-Attention-Parallel-Bayesian-Filtering-For-Efficient-Language-Modelling-and-State-Tracking/meta/meta_info.txt)**: Introduces scan-parallel information-form Kalman filtering as a sequence mixer with explicit uncertainty.
- **[KOSS: Kalman-Optimal Selective State Spaces for Long-Term Sequence Modeling](./references/KOSS-Kalman-Optimal-Selective-State-Spaces-for-Long-Term-Sequence-Modeling/meta/meta_info.txt)**: Motivates innovation-driven selectivity (gain estimation from innovation) for SSMs and introduces context-aware selective copying with correlated distractors.
- **[Mechanistic Design and Scaling of Hybrid Architectures](./references/Mechanistic-Design-and-Scaling-of-Hybrid-Architectures/meta/meta_info.txt)**: Proposes MAD synthetic tasks and shows correlation between synthetic task performance and scaling behavior.
- **[MAD-Lab repository](./references/GitHub---athms-mad-lab/meta/meta_info.txt)**: Open-source implementation of MAD synthetic tasks and training harness.
- **[Robust Filter Attention: Self-Attention as a Parallel State Estimator](./references/Attention-as-an-Adaptive-Filter/meta/meta_info.txt)**: Interprets attention as robust state estimation and uses residual-based robust weighting, but in quadratic attention.
- **[Robust Filter Attention repository](./references/GitHub---PCR-git-Robust-Filter-Attention/meta/meta_info.txt)**: Reference implementation of RFA.
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**: Standard quadratic self-attention baseline.
- **[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)**: Shows linear attention as a recurrent update.
- **[Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396)**: Foundational structured state space model for long sequences.
- **[Linear-Time Sequence Modeling with Selective State Spaces (Mamba)](https://arxiv.org/abs/2312.00752)**: Selective SSM sequence mixer with token-dependent gating.
- **[Mamba-2](https://arxiv.org/abs/2405.21060)**: Improved selective SSM with stronger performance/efficiency.
- **[Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)**: Gated linear attention as an efficient transformer alternative.
- **[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2406.06484)**: Delta-rule / gated updates for improved associative recall.
- **[Zoology: Measuring and Improving Recall in Efficient Language Models](./references/Zoology\ Measuring\ and\ Improving\ Recall\ in\ Efficient\ Language\ Models/meta/meta_info.txt)**: Establishes MQAR and shows associative recall explains **82%** of the perplexity gap between efficient mixers and attention on The Pile (Arora et al., 2023).
- **[Hyena Hierarchy](https://arxiv.org/abs/2302.10866)**: Convolutional long-range sequence mixer.
- **[Jamba: Hybrid Transformer-Mamba](https://arxiv.org/abs/2403.19887)**: Hybrid attention + SSM architecture illustrating mixing primitives.
- **[Deep Kalman Filters](https://arxiv.org/abs/1511.05121)**: Neural parameterization of latent state space models.
- **[KalmanNet](https://arxiv.org/abs/2107.10043)**: Learns Kalman filtering components with neural networks; often uses innovation signals in tracking.
- **[CE-BASS: Innovative and Additive Outlier Robust Kalman Filtering](https://arxiv.org/abs/2007.03238)**: Classical robust filtering for innovative/additive outliers.
- **[RKFNet: A Novel Neural Network Aided Robust Kalman Filter](https://arxiv.org/abs/2403.16756)**: Neural robust Kalman filtering under heavy-tailed noise.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Efficient sequence mixers | Replace quadratic attention with linear-time recurrence / structured kernels | S4, Mamba, Hyena, GLA | Long-context LM, recall tasks (MQAR), synthetic copy/recall suites | Selectivity often token-local; robustness to correlated distractors unclear |
| Bayesian filtering as sequence mixing | Maintain explicit belief state (mean + uncertainty) and update via probabilistic inference | KLA | MAD synthetic tasks, MQAR, state tracking tasks | Learned measurement precision may be token-local; robust noise models largely unexplored |
| Robust state estimation | Downweight/inflate measurement noise when innovations are inconsistent | CE-BASS, Student-t Kalman, RKFNet | Tracking and system ID benchmarks | Often sequential/iterative; not designed for scan-parallel neural sequence mixers |
| Robust estimation in attention | Use residual consistency / Mahalanobis geometry for weighting in attention | Robust Filter Attention | Synthetic dynamics / proof-of-concept demos (real LM eval pending) | Quadratic attention; different setting than linear-time sequence mixers |
| Synthetic mechanistic evaluation | Small synthetic unit tests intended to predict scaling behavior | MAD-Lab | Recall/copy/compression tasks; architecture prototyping | Correlation strongest within families; still a proxy for real LM behavior |

### Closest Prior Work

1. **KLA (Shaj et al., 2026)**: Provides scan-parallel Bayesian filtering with explicit precision \(\lambda_t\) and learned measurement precision \(\Lambda^v_t\). It does not propose innovation-normalized robust reweighting of \(\Lambda^v_t\) based on predicted state consistency.
2. **KOSS (Wang et al., 2025)**: Uses innovation to estimate a gain for context-aware selectivity, and introduces a correlated-distractor copying task. It does not provide a scan-parallel information-form Kalman layer with robust outlier downweighting, nor a compute-matched permutation control.
3. **Robust Filter Attention (Racioppo, 2025)**: Implements robust reweighting ideas in quadratic attention via residual/Mahalanobis scoring. Our setting differs: we aim for linear-time scan-parallel KLA-style filtering with a minimal two-pass robustification.
4. **Robust Kalman filtering literature (e.g., CE-BASS; Student-t variants)**: Provides robust innovation-based measurement noise inflation, but is not integrated into modern neural sequence mixers with parallel scans.

**Novelty Kill Search Summary:** Searched for combinations of “Kalman Linear Attention” with “Student-t”, “IRLS”, “innovation gating”, and “robust”, plus “information filter linear attention robust” and “normalized innovation attention”, and checked GitHub for public KLA implementations. No prior work implementing scan-parallel innovation-reweighted precision for KLA was found as of 2026-02-23. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| KLA (2602.10743) | Scan-parallel Bayesian filtering sequence mixer | Measurement precision is learned token-locally; robustness to context-dependent distractors unclear | Add innovation-normalized reweighting of \(\Lambda^v\) using pass-1 belief | Innovation depends on state (context), enabling downweighting of inconsistent evidence |
| KOSS (2512.16723) | Innovation-driven selectivity for SSMs | Not KLA information-form scan; no robust NIS-based outlier rule | Use robust filtering principle inside KLA; evaluate on conflict distractors | Robust NIS rules are designed to ignore outliers even if token-local features are ambiguous |
| Robust Filter Attention (2509.04154) | Robust residual weighting in quadratic attention | Still O(n^2); different architecture family | Apply IRLS-style reweighting to linear-time scan-parallel filtering | Keeps linear-time recurrent state while adding robustness |
| CE-BASS / Student-t Kalman | Robust classical filtering for outliers | Sequential/iterative; not integrated into neural sequence mixers | One-step reweighting compatible with scans | Minimal robustification with low engineering cost |

---

## Experiments

### Experimental Setup

We use **three benchmarks**:

1. **WikiText-103 (real language modeling; established)**: train a small causal LM and report validation perplexity on real text.
2. **MQAR (Zoology standard task; established synthetic)**: reproduce MQAR evaluation from Zoology (Arora et al., 2023) and/or the exact hard setting used by KLA (T=2048, V=256; see KLA “Long-Context Associative Recall”). This anchors results to a widely-used recall benchmark that is known to predict real LM perplexity gaps.
3. **Conflict-MQAR (ours; targeted synthetic)**: a context-dependent distractor variant of MQAR designed to isolate the failure mode “token-local precision fails under correlated distractors”.

All benchmarks are fully automated and use the same model family/training loop.

#### (1) WikiText-103 (real LM)

- **Task**: standard next-token prediction on WikiText-103.
- **Metric**: validation perplexity (lower is better).
- **Why included**: addresses **generalizability** by testing whether the robustification is safe (non-degrading) on real language.

#### (2) MQAR (Zoology)

- **Task**: Multi-Query Associative Recall (MQAR), a synthetic key–value retrieval benchmark defined in Zoology (Definition 3.1).
- **Metric**: query-token accuracy (and optionally sequence accuracy).
- **Why included**: MQAR is directly connected to real LM perplexity gaps in Zoology, and is commonly used to diagnose recall in efficient mixers.

#### (3) Conflict-MQAR (targeted)

We define a context-dependent distractor variant of MQAR. Each sequence contains key–value evidence tokens and query tokens; the model must output the correct value at query positions.

**Conflict-MQAR generation (pseudocode-level spec):**
1. Sample a mode bit \(m \in \{0,1\}\) and emit a special mode token `MODE_m` at position 0.
2. Sample \(K\) keys from a key vocabulary \(\mathcal{V}_k\) and two values per key \(v^0(k), v^1(k)\) from a value vocabulary \(\mathcal{V}_v\).
3. Construct an evidence prefix by interleaving \(K\) pairs for both mappings: emit `(k, v^0(k))` and `(k, v^1(k))` at random positions with random spacing so local heuristics are insufficient.
4. Construct a query suffix containing \(Q\) queries: for each query key \(k\), emit `(k, ANSWER)` where target at `ANSWER` is \(v^m(k)\).
5. Optionally vary a conflict rate \(p\): with probability \(p\), include both mappings (hard); with probability \(1-p\), include only the correct mapping (clean).

**Labels for diagnostics**: because the generator knows which evidence tokens correspond to the correct mapping under mode \(m\), we can label each evidence pair as “relevant” or “distractor” for AUC analyses.

**Models / conditions (3 main conditions, compute-matched):**
- **(A) KLA**: standard single-pass KLA.
- **(B) PR-KLA (ours)**: two-pass KLA with Student-t NIS-based \(\Lambda^v\) reweighting.
- **(C) Permuted-w control**: two-pass KLA with weights permuted along time within each sequence.

All conditions use identical architecture and training hyperparameters, differing only in the precision scaling rule.

**Implementation plan**:
- Implement KLA as a PyTorch module using published diagonal recurrences (Theorem 1–2) and `torch._higher_order_ops.associative_scan` (as used in KLA runtime appendix). We do not require custom Triton kernels for the short synthetic sequences in this proposal.
- Use MAD-Lab training harness as a reference implementation for synthetic tasks and logging (or reimplement a minimal training loop if integration is simpler).

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| KLA-style synthetic LM | ~10–50M params (configurable) | N/A (trained from scratch) | Single-block or shallow stack sequence model trained only on synthetic data |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| MQAR (Zoology synthetic) | Train/eval | Synthetic (per Zoology) | https://github.com/HazyResearch/zoology | MIT (verify during implementation) |
| Conflict-MQAR (ours) | Train/eval | Synthetic (generated on the fly) | N/A | N/A |

**Resource Estimate**:
- **Compute budget**: ≤ 150 A100-hours total.
  - 3 conditions × 3 seeds = 9 training runs.
  - Conflict-MQAR sequences are short (e.g., length 256–512); MQAR runs include either Zoology’s synthetic configs (N up to 512) or KLA’s harder MQAR setting (T=2048, V=256). Overall training should be fast (synthetic tasks are designed for rapid prototyping).
- **GPU memory**: ≤ 40GB per run (small models, short sequences).
- **API usage**: None.

**Baseline Ladder (REQUIRED):**
- **Trivial**: random/majority value predictor for MQAR-style tasks.
- **Strongest within-family baseline**: vanilla KLA.
- **Compute-matched control**: permuted-weight KLA (isolates the claimed mechanism).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| WikiText-103 | Real language modeling (next-token prediction) | Validation perplexity | train/valid | https://huggingface.co/datasets/Salesforce/wikitext (wikitext-103-raw-v1) | HuggingFace eval / standard LM scripts |
| MQAR (Zoology) | Established multi-query associative recall benchmark | Query-token accuracy; (optional) seq accuracy | train/test | https://github.com/HazyResearch/zoology | Zoology scripts or reimplementation from paper |
| Conflict-MQAR (ours) | Context-conditioned associative recall with conflicting evidence mappings (correlated distractors) | Query-token accuracy; seq accuracy | train/test | N/A (synthetic) | To be provided with generation code |

### Main Results

#### Results Table

We will report results on **both** MQAR (Zoology) and Conflict-MQAR.

| Method | Base Model | Benchmark | Query Accuracy (mean±std) | Seq Accuracy (mean±std) | Source | Notes |
|--------|------------|-----------|----------------------------|--------------------------|--------|-------|
| Random/majority | N/A | MQAR (Zoology) | **TBD** | **TBD** | - | Sanity check |
| KLA | trained from scratch | MQAR (Zoology) | **TBD** | **TBD** | - | Established benchmark |
| Permuted-w control | trained from scratch | MQAR (Zoology) | **TBD** | **TBD** | - | Mechanism control |
| **PR-KLA (ours)** | trained from scratch | MQAR (Zoology) | **TBD** | **TBD** | - | Ours |
| Random/majority | N/A | Conflict-MQAR (ours) | **TBD** | **TBD** | - | Sanity check |
| KLA | trained from scratch | Conflict-MQAR (ours) | **TBD** | **TBD** | - | Targeted failure mode |
| Permuted-w control | trained from scratch | Conflict-MQAR (ours) | **TBD** | **TBD** | - | Mechanism control |
| **PR-KLA (ours)** | trained from scratch | Conflict-MQAR (ours) | **TBD** | **TBD** | - | Ours |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | Two-pass, Student-t weights, stop-grad on weights | Best on distractor-heavy setting |
| w/o stop-grad | Allow gradients through \(w\) | May be unstable; if stable may further improve |

### Experimental Rigor

- **Seeds**: `seeds=[1,2,3]` for all conditions.
- **Primary confounder control**: permuted-weight control matches compute and weight distribution.
- **Sanity checks**: (i) random predictor near chance; (ii) when conflict rate \(p=0\), all methods should approach high accuracy.
- **Mechanism diagnostics (non-core analysis)**:
  - AUC of token-local precision proxy \(\phi_t = k_t^2 \Lambda^v_t\) for classifying relevant vs distractor evidence positions.
  - AUC of NIS for the same classification.

---

## Success Criteria

**Hypothesis**: PR-KLA improves accuracy on conflict-heavy sequences because innovation-normalized residuals identify globally inconsistent evidence that token-local \(\Lambda^v\) cannot.

**Decision Rule**:
- **Continue/Proceed**: On **Conflict-MQAR (hard; e.g., conflict rate \(p \ge 0.5\))**, PR-KLA improves query-token accuracy over both vanilla KLA and permuted-w control by a margin outside the combined std across 3 seeds, while not reducing accuracy on the clean setting (\(p=0\)) by more than 1% absolute. Additionally, on **MQAR (Zoology)**, PR-KLA must be **non-inferior** to vanilla KLA (≤0.5% absolute drop) to claim generalizability beyond the targeted distractor setting.
- **Pivot**: If PR-KLA > KLA but PR-KLA ≈ permuted-w, treat gains as generic regularization; pivot to learned weighting functions (predict \(w\) from innovation features) or a different robust statistic.
- **Refute**: If PR-KLA does not beat permuted-w (or underperforms KLA), abandon innovation-reweighting as a useful robustness add-on for scan-parallel KLA.

---

## Impact Statement

If successful, this provides a simple, principled robustness mechanism for Bayesian-filtering sequence mixers that could improve long-context stability under noisy or adversarially correlated tokens. Practitioners designing efficient long-context models could use innovation-reweighted precision as a drop-in alternative when training data contains misleading evidence or when the application requires robustness to distractors.

---

## References

- [Kalman Linear Attention: Parallel Bayesian Filtering For Efficient Language Modelling and State Tracking](./references/Kalman-Linear-Attention-Parallel-Bayesian-Filtering-For-Efficient-Language-Modelling-and-State-Tracking/meta/meta_info.txt) - Shaj et al., 2026
- [KOSS: Kalman-Optimal Selective State Spaces for Long-Term Sequence Modeling](./references/KOSS-Kalman-Optimal-Selective-State-Spaces-for-Long-Term-Sequence-Modeling/meta/meta_info.txt) - Wang et al., 2025
- [Mechanistic Design and Scaling of Hybrid Architectures](./references/Mechanistic-Design-and-Scaling-of-Hybrid-Architectures/meta/meta_info.txt) - Poli et al., 2024
- [MAD-Lab repository](./references/GitHub---athms-mad-lab/meta/meta_info.txt) - athms, 2024
- [Robust Filter Attention: Self-Attention as a Parallel State Estimator](./references/Attention-as-an-Adaptive-Filter/meta/meta_info.txt) - Racioppo, 2025
- [Robust Filter Attention repository](./references/GitHub---PCR-git-Robust-Filter-Attention/meta/meta_info.txt) - PCR-git, 2025
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) - Katharopoulos et al., 2020
- [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) - Gu et al., 2021
- [Linear-Time Sequence Modeling with Selective State Spaces (Mamba)](https://arxiv.org/abs/2312.00752) - Gu & Dao, 2023
- [Mamba-2](https://arxiv.org/abs/2405.21060) - Dao & Gu, 2024
- [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) - Ding et al., 2023
- [Hyena Hierarchy](https://arxiv.org/abs/2302.10866) - Poli et al., 2023
- [Zoology: Measuring and Improving Recall in Efficient Language Models](./references/Zoology\ Measuring\ and\ Improving\ Recall\ in\ Efficient\ Language\ Models/meta/meta_info.txt) - Arora et al., 2023
- [Jamba](https://arxiv.org/abs/2403.19887) - Authors, 2024
- [Deep Kalman Filters](https://arxiv.org/abs/1511.05121) - Krishnan et al., 2015
- [KalmanNet](https://arxiv.org/abs/2107.10043) - Revach et al., 2021
- [CE-BASS](https://arxiv.org/abs/2007.03238) - Authors, 2020
- [RKFNet](https://arxiv.org/abs/2403.16756) - Authors, 2024
