# untitled

# GaugeFix-LRM: Function-Preserving Q/K Gauge Fixing to Replace Multiplier Weight Decay in Learnable Multipliers

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Weight decay is a standard component of large language model (LLM) pretraining. It improves both optimization stability and generalization, but it also couples the scale of learned weight matrices to optimizer hyperparameters such as learning rate and weight decay. **Learnable Multipliers (LRM)** proposes a simple reparameterization that attaches learnable scalar/vector multipliers to matrix layers so that *scale can be learned from data* while the matrix parameters remain bounded by weight decay.

A practical complication is that once we add multipliers, we expose **parameter symmetries**: continuous transformations of parameters that leave the model’s function unchanged in exact arithmetic. Learnable multipliers are especially susceptible because they directly control scale. The LRM paper highlights two such symmetries that can cause instability under low precision (e.g., bfloat16):

- **Multiplicative symmetry**: if two factors only appear through a product (a\,b), then (a,b)→(s a, s^{-1} b) leaves the forward computation unchanged; in attention this arises in the query–key (Q/K) bilinear form.
- **Normalization symmetry**: if a signal is only used after RMS normalization, its pre-normalization scale can drift without affecting the output.

To suppress these drifts, LRM uses a pragmatic fix: a small **multiplier weight decay** (λ_lrm≈2×10^{-3}, where λ_lrm denotes the weight decay coefficient applied specifically to multiplier parameters) applied to multipliers. However, this is not function-preserving (it changes the effective model) and introduces an additional tuning knob, potentially counteracting the goal of letting scales adapt to data.

### The Problem

We focus on a concrete sub-problem: **controlling Q/K multiplicative symmetry drift in LRM without relying on weight decay for the Q/K multipliers**.

- **[Learnable Multipliers](./references/Learnable-Multipliers-Freeing-the-Scale-of-Language-Model-Matrix-Layers/meta/meta_info.txt)**: identifies Q/K symmetry drift as a cause of low-precision instability and uses multiplier weight decay λ_lrm to suppress it.
- **[Weight-balancing fixes and flows](./references/Weight-Balancing-Fixes-and-Flows-for-Deep-Learning/meta/meta_info.txt)** and **[Equi-normalization](./references/Equi-normalization-of-Neural-Networks/meta/meta_info.txt)** show that for scale-invariant parameterizations, one can often move within the equivalence class of functionally identical parameters to improve conditioning.
- **[Maximal Gauge Symmetry in Transformer Architectures](./references/Maximal-Gauge-Symmetry-in-Transformer-Architectures/meta/meta_info.txt)** formalizes that Transformers have large “gauge groups” of function-preserving parameter transformations, implying that a canonical gauge choice is well-defined in principle.

The practical gap is that today’s LRM recipe uses **weight decay as a surrogate gauge fix**, even though the symmetry is exact. It is unclear whether a truly function-preserving gauge fix can substitute for λ_lrm in practice.

### Key Insight and Hypothesis

**Key insight:** For Q/K drift, we can apply an explicit *gauge-fixing projection* after each optimizer step that stays on the same function-equivalence class (up to floating-point error), instead of adding weight decay that changes the function.

**Hypothesis:** In a proxy LLM pretraining run with LRM, replacing weight decay on Q/K multipliers with a per-step, function-preserving Q/K balancing projection will (i) prevent Q/K scale-ratio drift and reduce loss spikes in bf16, while (ii) matching the validation loss of the standard LRM recipe that uses λ_lrm on Q/K.

Why this could be wrong:
- Multiplier weight decay may help for reasons beyond symmetry control (e.g., it provides beneficial implicit regularization of scale) and a pure gauge fix may not replicate that.
- The proxy training setup might not exhibit strong Q/K drift, making the experiment inconclusive.

---

## Proposed Approach

### Overview

We propose **GaugeFix-LRM**, a training-time modification for LRM that replaces multiplier weight decay on the query/key multipliers with an explicit **gauge-fixing projection**.

We deliberately target the simplest symmetry first: **per-head Q/K rescaling** in multi-head attention, where the attention logits depend on the bilinear form QK^\top and are invariant to Q→Q/g, K→gK for any scalar g>0.

### Method Details

#### Base parameterization: Learnable Multipliers (LRM)
LRM reparameterizes a linear layer weight matrix W using learnable multipliers (scalar or per-row/per-column) so that scale can be learned while the matrix parameters remain bounded by weight decay.

For attention, LRM notes that having both query and key multipliers introduces a multiplicative symmetry: scaling query and key multipliers inversely preserves the model output but can cause numerical drift in low precision.

#### GaugeFix-LRM: per-step Q/K gauge fixing (unclamped)
Assume each attention head h has learnable **row multipliers** r_Q^{(h)} and r_K^{(h)} (vectors of length d_k, or equivalently per-output-dimension scales for the Q and K projections).

Define per-head RMS scales:

- s_Q^{(h)} = RMS(r_Q^{(h)})
- s_K^{(h)} = RMS(r_K^{(h)})

Gauge-fixing projection after each optimizer step:

- g^{(h)} = \sqrt{(s_Q^{(h)} + \epsilon) / (s_K^{(h)} + \epsilon)}
- r_Q^{(h)} \leftarrow r_Q^{(h)} / g^{(h)}
- r_K^{(h)} \leftarrow r_K^{(h)} \cdot g^{(h)}

where \epsilon is a small constant (e.g., 1e-12).

This transformation preserves the per-head elementwise product r_Q^{(h)} \odot r_K^{(h)} up to floating-point rounding and therefore preserves the attention logit bilinear form up to floating-point error. In plain terms: we measure whether the Q multipliers are larger than the K multipliers for each head, and then rescale them in opposite directions so their overall contribution to QK^\top stays the same but their magnitudes stay balanced. Importantly, we **do not clamp** g^{(h)} in the main method; if g becomes extreme, that is treated as an empirical signal about drift severity.

#### Scope restriction (to keep the test interpretable)
LRM’s λ_lrm also affects other multipliers (e.g., output projection, MLP, normalization-adjacent parameters). To isolate the Q/K mechanism:

- We keep λ_lrm unchanged for all **non-Q/K multipliers**.
- We only remove weight decay for **Q/K multipliers**, and replace it with GaugeFix.

This makes the experiment a clean substitution test: “is λ_lrm-on-QK acting mainly as symmetry control, or as beneficial regularization?”

### Key Innovations

1. **Function-preserving symmetry control for LRM**: replace weight-decay-based symmetry handling with an explicit gauge-fixing projection that preserves model function (up to floating-point).
2. **Isolation of mechanism**: remove weight decay only on Q/K multipliers while keeping all other training settings (including λ_lrm for other multipliers) fixed.
3. **Minimal canonical gauge**: a per-head RMS balancing rule is a simple canonical gauge choice that is easy to implement and measure.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) understanding why weight decay is important in modern training, (ii) reparameterization / normalization methods that change optimization geometry, and (iii) architectural symmetries (“gauge freedoms”) that create flat directions in parameter space.

In LLM pretraining, training stability is tightly coupled to parameter scales and low-precision arithmetic (fp16/bf16). Many stabilization methods (normalization layers, careful initialization, gradient clipping) can be interpreted as indirectly controlling scale. LRM highlights that scale is not merely a stability concern: if matrix scales are pinned by noise–weight-decay equilibrium, the model may underutilize representational degrees of freedom.

The gauge-symmetry viewpoint suggests an alternative: when instability comes from drifting along exact symmetry directions, we can stabilize training by staying in a canonical representative of the equivalence class (gauge fixing), rather than by adding regularization terms that change the equivalence class.

### Related Papers

- **[Learnable Multipliers: Freeing the Scale of Language Model Matrix Layers](./references/Learnable-Multipliers-Freeing-the-Scale-of-Language-Model-Matrix-Layers/meta/meta_info.txt)**: introduces LRMs to escape noise–WD scale equilibrium; uses small multiplier weight decay to suppress symmetry drift.
- **[Weight-balancing fixes and flows for deep learning](./references/Weight-Balancing-Fixes-and-Flows-for-Deep-Learning/meta/meta_info.txt)**: proposes function-preserving balancing flows to improve conditioning under rescaling symmetries.
- **[Equi-normalization of Neural Networks](./references/Equi-normalization-of-Neural-Networks/meta/meta_info.txt)**: Sinkhorn-like balancing within rescaling equivalence classes to reduce optimization pathologies.
- **[Maximal Gauge Symmetry in Transformer Architectures](./references/Maximal-Gauge-Symmetry-in-Transformer-Architectures/meta/meta_info.txt)**: characterizes Transformer gauge groups, motivating canonical gauge choices and gauge-aware optimization.
- **[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)**: introduces AdamW, separating weight decay from gradient-based updates.
- **[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)**: foundational adaptive optimizer used in most LLM training.
- **[Mixed Precision Training](https://arxiv.org/abs/1710.03740)**: documents fp16/bf16 training issues and mitigation, framing why scale drift matters numerically.
- **[Weight Normalization](https://arxiv.org/abs/1602.07868)**: reparameterizes weights into direction and magnitude to improve optimization, an early example of explicit scale parameterization.
- **[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)**: analyzes LayerNorm/LN placement and its effects on Transformer optimization.
- **[ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887)**: uses learnable residual scaling to improve deep network training stability.
- **[Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321)**: shows that careful initialization and explicit residual scaling can replace normalization layers for stable training.
- **[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)**: motivates head/group structure in attention; relevant because gauge freedom depends on head sharing.
- **[Sharp Minima Can Generalize for Deep Nets](https://arxiv.org/abs/1703.04933)**: discusses reparameterization invariances and their implications for sharpness-style arguments.
- **[Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks](https://arxiv.org/abs/2305.17212)**: analyzes WD as a training dynamics equilibrium, closely related to LRM’s framing.
- **[Weight Decay May Matter More Than μP for Learning Rate Transfer in Practice](https://arxiv.org/abs/2510.19093)**: shows that independent weight decay scaling can dominate learning-rate transfer behavior, highlighting weight decay’s role as dynamics control.
- **[Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)**: introduces μP (maximal update parameterization) / μTransfer for scale-robust hyperparameters, providing context for why extra scale knobs (like multipliers) interact with WD.
- **[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)**: proposes orthogonalized-momentum updates via Newton–Schulz iterations, an alternative approach to conditioning/scale control in large-model training.
- **[Convergence of Muon with Newton–Schulz](https://arxiv.org/abs/2601.19156)**: analyzes convergence of Newton–Schulz-based orthogonalization, relevant as a low-precision-friendly matrix operation used in modern training.
- **[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)**: demonstrates Muon scaling to LLM pretraining and discusses the need for weight decay to prevent bf16 range issues.
- **[Why Do We Need Weight Decay in Modern Deep Learning?](https://arxiv.org/abs/2310.04415)**: argues weight decay in modern deep learning acts mainly through optimization dynamics and low-precision stability rather than classical regularization.
- **[Weight Decay Induces Low-Rank Attention Layers](https://arxiv.org/abs/2410.23819)**: shows L2 regularization can induce low-rank structure in attention products, implying WD interacts with attention geometry beyond pure stability.
- **[Root Mean Square Layer Normalization (RMSNorm)](https://arxiv.org/abs/1910.07467)**: introduces RMS-based normalization, relevant because LRM’s “normalization symmetry” examples often involve RMS-based normalizers.
- **[PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)**: representative large-scale training recipe where WD is essential.
- **[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)**: scaling-law context emphasizing stable, well-tuned training.

Notes:
- All links above are intended to be resolvable as-is; if any become stale (e.g., blog URLs), the verification module should substitute an archival source.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Weight decay as dynamics control | WD counteracts noise-driven expansion and shapes optimization trajectories | LRM; Rotational Equilibrium; AdamW | Training loss/val loss; stability events | WD can act as both stabilizer and inductive bias; hard to disentangle |
| Reparameterization & explicit scale parameters | Make scale learnable/explicit (often improving conditioning) | Weight Normalization; ReZero; LRM | Training stability; convergence speed | Can introduce new symmetries/flat directions |
| Balancing / gauge fixing | Move within function-equivalence class to improve conditioning without changing function | Equi-normalization; Weight-balancing flows; Transformer gauge symmetry theory | Conditioning metrics; drift metrics; stability | Hard to scale to modern architectures; needs careful definition of symmetry group |
| Transformer-specific symmetry analysis | Characterize exact parameter equivalences in attention blocks | Maximal Gauge Symmetry in Transformers; sharpness under symmetries | Hessian nullspace; invariance tests | Mostly theoretical; few training-time interventions |

### Closest Prior Work

1. **Learnable Multipliers (Velikanov et al., 2026)**: Introduces LRMs and identifies Q/K multiplicative symmetry drift as a practical bf16 instability source; uses multiplier weight decay λ_lrm as a simple mitigation. **Limitation for our question:** λ_lrm changes the function and is an extra tuning knob; it is unclear whether symmetry drift can be handled with a function-preserving projection.

2. **Weight-balancing fixes and flows (Saul, 2023)**: Provides function-preserving balancing dynamics for rescaling symmetries (primarily in homogeneous networks). **Limitation:** not instantiated for Transformer attention multipliers or LLM pretraining; does not test “replace WD-on-symmetry-parameters with projection.”

3. **Equi-normalization (Stock et al., 2019)**: Equalizes layer scales using Sinkhorn-like normalization within equivalence classes. **Limitation:** focuses on convolutional/standard feedforward networks; not targeted to attention Q/K bilinear symmetries.

4. **Maximal Gauge Symmetry in Transformer Architectures (Wang & Wang, 2026)**: Characterizes Transformer gauge groups, motivating gauge-aware optimization. **Limitation:** does not propose or evaluate a concrete, minimal training-time gauge-fixing step for LRMs.

**Novelty Kill Search Summary:** Searched for “learnable multipliers gauge fixing”, “learnable multipliers symmetry drift weight decay alternative”, “transformer gauge symmetry optimization projection”, “equi-normalization transformers”, and checked OpenReview/ArXiv for direct “LRM + gauge fixing” combinations. No prior work replacing λ_lrm-on-QK with an explicit per-step gauge projection was found as of 2026-02-18 (query log in notes.md).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LRM (2026) | Adds multipliers to learn matrix scale; uses λ_lrm to suppress drift | Symmetry control is via non-function-preserving WD; adds tuning | Replace λ_lrm on Q/K with function-preserving projection | If drift is the main issue, projection should stabilize without extra regularization |
| Saul (2023) | Balancing flows for rescaling symmetries | Not instantiated for Transformers/LRMs | Apply balancing idea to Q/K symmetry in attention multipliers | Transformers have explicit QK invariance, enabling an extremely simple projection |
| Stock (2019) | Equi-normalization via balancing | Not targeted to attention bilinear symmetries | Use a symmetry-specific gauge fix on Q/K | Q/K symmetry is local and per-head; cheaper and easier to implement |
| Wang & Wang (2026) | Formal gauge group characterization | Theory-only for training | Turn a known gauge freedom into a concrete training intervention | Bridges theory (symmetry) and practice (stability) with a decisive test |

---

## Experiments

### Experimental Setup

**Goal:** Decide whether Q/K symmetry handling in LRM can be done by function-preserving projection instead of weight decay.

**Baseline Ladder (REQUIRED):**
- **Standard training (context baseline)**: GPT-2 training on OpenWebText with AdamW (nanoGPT reports **val loss 3.12** for GPT-2 124M on OWT; this is context only, not the core bet) [nanoGPT README](./references/GitHub---karpathy-nanoGPT/sections/README.md.md).
- **Closest existing method**: LRM with multiplier weight decay λ_lrm = 2×10^{-3} (applied to all multipliers, including Q/K).
- **Our method**: remove WD only for Q/K multipliers + apply GaugeFix projection each step.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| GPT-2 (from scratch) | 124M | https://huggingface.co/openai-community/gpt2 | Used only for architecture definition; experiments train from scratch on OWT |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| OpenWebText | Pretraining corpus | ~8B tokens (approx; exact depends on tokenizer) | https://huggingface.co/datasets/openwebtext | Check dataset card |

**Evaluation code / harness:**
- Use a minimal public training loop (e.g., nanoGPT-style) modified to add LRMs and GaugeFix. For reproducibility, nanoGPT’s default GPT-2(124M) config sets `batch_size=12`, `block_size=1024`, `gradient_accumulation_steps=5*8`, `learning_rate=6e-4`, `weight_decay=0.1`, `max_iters=600000`, `grad_clip=1.0` [train_gpt2.py](./references/nanoGPT-train_gpt2-config/sections/Main%20Content.md) and [train.py](./references/nanoGPT-train-script/sections/Main%20Content.md).
- Validation metric is next-token cross-entropy on the provided OWT validation split.

**Implementation details to match LRM best practices:**
- Apply global gradient clipping as in LRM, but **exclude multiplier gradients from the global clip-norm computation** (LRM reports otherwise multipliers can dominate early gradients and cause over-clipping).
- Use bf16 training to stress low-precision stability.

**Main conditions (exactly 3):**

**Sanity check (before the main 3×3 run):** Run condition C (Q/K no-WD, no GaugeFix) for a short burn-in (e.g., 50M tokens, 1 seed) to confirm that Q/K drift is measurable in this proxy setting. If max |log(s_Q/s_K)| stays near 0 throughout (indicating negligible drift), treat the experiment as inconclusive at this scale and increase token budget (up to the compute cap) before comparing A vs B.

| Condition | Q/K multipliers WD | Other multipliers WD | GaugeFix(QK) |
|---|---:|---:|---:|
| A: LRM baseline | λ_lrm=2e-3 | λ_lrm=2e-3 | No |
| B: GaugeFix-LRM (ours) | 0 | λ_lrm=2e-3 | Yes (every step) |
| C: No-control ablation | 0 | λ_lrm=2e-3 | No |

**Resource Estimate**:
- **Compute budget**: target ≤250 A100-hours total.
  - Proxy evidence: nanoGPT reports GPT-2 (124M) training on OpenWebText on **8×A100 40GB in ~4 days** [nanoGPT README](./references/GitHub---karpathy-nanoGPT/sections/README.md.md). The default config uses ~0.492M tokens/iteration and `max_iters=600000`, i.e., ~295B tokens total [train_gpt2.py](./references/nanoGPT-train_gpt2-config/sections/Main%20Content.md).
  - Planned verification run: **2B tokens** per run with 3 conditions × 3 seeds (9 runs). Assuming throughput similar to nanoGPT’s setup, this is ~2/295 ≈ 0.68% of the long run, i.e., ~5.2 GPU-hours per run on 8×A100, or ~47 GPU-hours total for 9 runs. We budget **2×** overhead for slower kernels / extra logging (≤100 GPU-hours total).
- **GPU memory**: fits in 1×A100 80GB for GPT-2 124M; multi-GPU only for throughput.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| OpenWebText (OWT) | Web-text next-token prediction corpus | Val loss (cross-entropy), perplexity | val | https://huggingface.co/datasets/openwebtext | training loop eval |

Additional logged diagnostics (not a separate benchmark):
- **Q/K drift**: max_t |log(s_Q/s_K)| per head, aggregated over heads/layers.
- **Stability spikes**: count of steps where loss_t exceeds rolling median + 5×MAD, where MAD is the median absolute deviation computed over the rolling window.
- **NaN/divergence**: number of NaN losses or aborted runs.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Val loss (mean±std) | Drift max | Spike count | NaN? | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| Standard GPT-2 training (no LRM) | GPT-2 124M | OWT | 3.12 (1 run) | - | - | - | [nanoGPT README](./references/GitHub---karpathy-nanoGPT/sections/README.md.md) | Context baseline; metric is validation cross-entropy (lower is better); same dataset+split but different training duration |
| A: LRM baseline | GPT-2 124M | OWT | **TBD** | **TBD** | **TBD** | **TBD** | - | To be verified (3 seeds) |
| C: QK no-WD, no GaugeFix | GPT-2 124M | OWT | **TBD** | **TBD** | **TBD** | **TBD** | - | To be verified (3 seeds) |
| **B: GaugeFix-LRM (ours)** | GPT-2 124M | OWT | **TBD** | **TBD** | **TBD** | **TBD** | - | To be verified (3 seeds) |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| B (full) | Q/K no-WD + per-step GaugeFix | Best trade-off: low drift with no loss degradation |
| C (no GaugeFix) | Remove gauge fix | Higher drift/spikes; possibly worse loss |

### Experimental Rigor

**Variance & Reproducibility:**
- Run A/B/C with **3 random seeds** (e.g., `seeds=[42, 123, 456]`).

**Validity & Controls:**
- Control for optimizer, data order, total tokens, LR schedule, and clipping rules across A/B/C.
- Report whether any run diverges/NaNs; treat divergence as a primary outcome.

---

## Success Criteria

**Hypothesis** (directional):
GaugeFix-LRM (B) will reduce Q/K drift and loss spikes relative to no-control ablation (C), while matching the LRM baseline (A) on validation loss.

**Decision Rule** (concrete):
- **Continue/Proceed** if, at a fixed token budget, B’s mean validation loss is within **0.02 nats** of A (a “nat” is a natural-log unit of cross-entropy; lower is better) (or overlaps within mean±std across 3 seeds) **and** B reduces drift/spike metrics substantially relative to C.
- **Pivot** if B reduces drift but slightly worsens loss vs A: test whether applying GaugeFix less frequently (e.g., every N steps) can recover loss (optional follow-up, not part of the core experiment).
- **Refute** if B is clearly worse than A on validation loss (>0.02 nats beyond std) or if B does not reduce drift/spikes vs C.

---

## Impact Statement

If successful, GaugeFix-LRM would provide a simple, architecture-grounded alternative to multiplier weight decay for controlling LRM symmetry drift, potentially reducing hyperparameter tuning burden and improving low-precision training stability in LLM pretraining pipelines.

---

## References

- [Learnable Multipliers: Freeing the Scale of Language Model Matrix Layers](./references/Learnable-Multipliers-Freeing-the-Scale-of-Language-Model-Matrix-Layers/meta/meta_info.txt) - Velikanov et al., 2026
- [Weight-balancing fixes and flows for deep learning](./references/Weight-Balancing-Fixes-and-Flows-for-Deep-Learning/meta/meta_info.txt) - Saul, 2023
- [Equi-normalization of Neural Networks](./references/Equi-normalization-of-Neural-Networks/meta/meta_info.txt) - Stock et al., 2019
- [Maximal Gauge Symmetry in Transformer Architectures](./references/Maximal-Gauge-Symmetry-in-Transformer-Architectures/meta/meta_info.txt) - Wang & Wang, 2026
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter, 2019
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - Kingma & Ba, 2015
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) - Micikevicius et al., 2017
- [Weight Normalization](https://arxiv.org/abs/1602.07868) - Salimans & Kingma, 2016
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Xiong et al., 2020
- [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887) - Bachlechner et al., 2021
- [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321) - Zhang et al., 2019
- [Weight Decay May Matter More Than μP for Learning Rate Transfer in Practice](https://arxiv.org/abs/2510.19093) - Kosson et al., 2025
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) - Yang et al., 2022
- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) - Jordan, 2024
- [Convergence of Muon with Newton–Schulz](https://arxiv.org/abs/2601.19156) - Kim & Oh, 2026
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) - Kimi Team, 2025
- [Why Do We Need Weight Decay in Modern Deep Learning?](https://arxiv.org/abs/2310.04415) - Andriushchenko et al., 2024
- [Weight Decay Induces Low-Rank Attention Layers](https://arxiv.org/abs/2410.23819) - Kobayashi et al., 2024
- [Root Mean Square Layer Normalization (RMSNorm)](https://arxiv.org/abs/1910.07467) - Zhang & Sennrich, 2019
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) - Ainslie et al., 2023
- [Sharp Minima Can Generalize for Deep Nets](https://arxiv.org/abs/1703.04933) - Dinh et al., 2017
- [Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks](https://arxiv.org/abs/2305.17212) - Kosson et al., 2024
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) - Chowdhery et al., 2022
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022
