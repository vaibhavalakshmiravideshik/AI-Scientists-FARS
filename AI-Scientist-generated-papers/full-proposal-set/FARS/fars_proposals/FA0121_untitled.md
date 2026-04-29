# untitled

# Counterfactual Gate Supervision for Hashed N-gram Conditional Memory

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) repeatedly spend dense compute reconstructing local, stereotyped patterns (common phrases, n-gram continuations, entity surface forms). A recent architectural line proposes **conditional memory**: a deterministic lookup module that retrieves learned embeddings keyed by local context (e.g., n-grams) and injects them into the transformer’s residual stream. Unlike retrieval-augmented generation (RAG) with nearest-neighbor search, deterministic lookup is attractive for systems: the memory address depends only on the input token sequence, enabling prefetching and offloading large tables to host memory.

**Engram** proposes a conditional-memory module based on deterministic **hashed n-gram lookup** with a **context-aware gate** α that modulates the memory injection at each token position (see `./references/Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/meta/meta_info.txt`). The gate is intended to suppress memory when it is harmful (e.g., collisions, polysemy, or contextual mismatch) and amplify it when it is helpful.

A controlled study of Engram-style conditional memory by **Lin (2026)** (see `./references/A-Collision-Free-Hot-Tier-Extension-for-Engram-Style-Conditional-Memory-A-Controlled-Study-of-Training-Dynamics/meta/meta_info.txt`) finds that improving lookup precision (eliminating collisions for high-frequency n-grams using minimal perfect hashing) does **not** significantly improve validation loss under strict iso-parameter control (Table 3 in `sections/5.1 Main Finding Eliminating Collisions Yields No Significant Benefit.md`). More importantly, Lin identifies a training pathology: the learned gate α can become **anti-correlated** with per-token loss late in training (Table 6), suggesting that **gating credit assignment**, not indexing precision, may be a dominant bottleneck at small/medium scale.

### The Problem

Engram’s gate is a learned scalar (α) that decides how strongly to inject the retrieved memory vector. The intended behavior is causal: memory should be used when it improves next-token prediction and suppressed otherwise.

Lin (2026) shows this intent can fail sharply. In the Hash-500K baseline trained on FineWeb-Edu (100M tokens), the highest gate bucket α∈[0.8,1.0] has average loss **5.28**, while α∈[0.2,0.4] has the lowest average loss **3.90** (Table 6 in `sections/5.4.2 Snapshot Analysis α alpha Bucketing.md`). This indicates a mis-calibrated gate that allocates the most memory to positions where it helps least.

Lin’s analysis suggests a plausible mechanism: the gate learns early preferences that become fixed and do not adapt after training dynamics shift (a hot→cold loss flip). If true, then simply changing hashing schemes (collision-free vs collision-prone) will not fix the core issue. What is missing is a training signal that directly teaches the gate its causal role: **route memory where it reduces loss**.

### Key Insight and Hypothesis

**Key insight**: The gate is a causal decision variable: changing α changes the logits, hence the loss. Therefore we can define an explicit supervision target using a **counterfactual loss difference** computed under forced gate settings.

**Hypothesis**: If we train the gate α to predict a per-token **counterfactual utility** signal (whether memory injection would reduce loss), then:
1) Gate mis-calibration will be reduced on held-out data (measured by improved agreement with an oracle gate), and
2) Overall validation loss will improve compared to the same hashed-lookup baseline under matched training.

Why this might fail: (i) the counterfactual signal may be noisy and non-stationary because memory tables and model weights change during training; (ii) the best behavior may require intermediate α values not captured by a binary target; (iii) counterfactual supervision could collapse α toward always-off, “solving” the pathology by avoiding memory rather than learning when it helps. Our experiments include automated diagnostics to distinguish these outcomes.

---

## Proposed Approach

### Overview

We propose **Counterfactual Gate Supervision (CGS)** for Engram-style hashed n-gram conditional memory. CGS does not change the retrieval mechanism or the memory parameter budget. It adds a lightweight auxiliary objective that trains the gate output α(x) to match an estimated per-token probability that memory injection would improve prediction.

### Method Details

#### Baseline conditional memory module (Engram-style)

For each token position t, construct suffix n-grams for orders n∈{2,3}. For each order and head k∈{1..K}, apply a deterministic hash function to obtain an index and retrieve an embedding vector; concatenate all retrieved vectors into e_t. The module computes a gated injection into the residual stream:

- k_t = W_K e_t, v_t = W_V e_t
- α̂_t = σ( RMSNorm(h_t)^T RMSNorm(k_t) / √d )

where **RMSNorm** is root-mean-square normalization (a layer-normalization variant that rescales activations by their RMS without mean-centering).
- Injected output: m_t = α̂_t · v_t

where h_t is the transformer hidden state at position t. (Notation matches Engram; see `./references/Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/meta/meta_info.txt`.)

In Lin (2026), memory injection occurs at multiple transformer layers (layers 2,4,6), each with its own gating parameters (see Table 1 in `sections/4.1 Training Scale and Data.md`).

#### Counterfactual utility target via forced-pass Δℓ

On a small subsample of training steps, CGS computes a per-token **counterfactual loss difference** by running *no-gradient* forward passes where one Engram layer’s gate is forced to α=0 vs α=1.

Let ℓ_t denote per-token cross-entropy loss at position t.

For a selected Engram layer L (sampled uniformly from {2,4,6} each CGS update):

- ℓ_t(α_L=0): forward pass where the injection at layer L is disabled (α_L forced to 0), while other layers’ gates are computed normally.
- ℓ_t(α_L=1): forward pass where the injection at layer L is fully enabled (α_L forced to 1), while other layers’ gates are computed normally.

Define the per-token counterfactual utility for that layer:

- Δℓ_t^{(L)} = ℓ_t(α_L=0) − ℓ_t(α_L=1)

Positive Δℓ means enabling memory at layer L improves prediction (reduces loss).

**RNG control (important for soundness)**: If training uses dropout, extra forward passes can consume RNG and change the subsequent dropout masks, confounding comparisons. To avoid this, implement the counterfactual passes with RNG-state save/restore (e.g., `torch.random.fork_rng` or explicit RNG state snapshots) so that the training forward/backward uses the same randomness as the baseline.

#### Gate supervision loss

Convert Δℓ to a soft target probability:

- y_t^{(L)} = sigmoid(Δℓ_t^{(L)} / τ)

and add an auxiliary loss on the *learned* gate output α̂_t^{(L)} from the normal training pass:

- L_CGS = BCE(α̂_t^{(L)}, y_t^{(L)})  (or MSE; BCE is preferred for a probability target)

Detach y_t^{(L)} from gradients so Δℓ computation does not backpropagate through the counterfactual passes.

**Compute control**: To bound overhead, compute L_CGS only every M steps (e.g., M=8) and/or on a microbatch subset. This adds ~2 extra forward passes on those steps.

### Key Innovations

- **Utility-based supervision for conditional-memory gates**: CGS trains α to reflect causal utility (forced-pass Δℓ), directly targeting the gate mis-calibration failure mode observed by Lin (2026).
- **Held-out oracle agreement metric**: We evaluate whether α aligns with an oracle gate on held-out data (oracle-AUC), which distinguishes “better routing” from “global avoidance.”

---

## Related Work

### Field Overview

This proposal sits at the intersection of memory-augmented language modeling and conditional computation. External-memory and retrieval-augmented LMs improve next-token prediction by consulting caches, datastores, or corpora. In parallel, sparse/modular architectures (notably Mixture-of-Experts, MoE) introduce routers/gates that decide where to spend compute. A recurring theme across both families is **credit assignment for routing decisions**: when the model has multiple paths (experts, memories), how does the router/gate learn which path actually helps?

Engram-style conditional memory is distinct from kNN retrieval or RAG: it retrieves parametric embeddings via deterministic hashed n-gram addressing, making it infrastructure-friendly (prefetchable/offloadable). Lin (2026) provides rare controlled evidence that improving lookup precision alone may not solve the key problem; instead, training dynamics of the gate can dominate.

### Related Papers

- **[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](./references/Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/meta/meta_info.txt)**: Introduces Engram conditional memory with hashed n-gram lookup and context-aware gating.
- **[A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics](./references/A-Collision-Free-Hot-Tier-Extension-for-Engram-Style-Conditional-Memory-A-Controlled-Study-of-Training-Dynamics/meta/meta_info.txt)**: Shows collision-free indexing does not improve loss under iso-parameter control and diagnoses gate preference fixation / α–loss inversion.
- **[Scaling Embedding Layers in Language Models](./references/Scaling-Embedding-Layers-in-Language-Models/meta/meta_info.txt)**: SCONE scales input embeddings via cached contextualized n-grams with offloading, related to n-gram memory as a scaling axis.
- **[Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling](./references/Over-Tokenized-Transformer-Vocabulary-is-Generally-Worth-Scaling/meta/meta_info.txt)**: Scales input vocab via hierarchical n-gram embeddings, complementary to conditional-memory approaches.
- **[Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](./references/Coupling-Experts-and-Routers-in-Mixture-of-Experts-via-an-Auxiliary-Loss/meta/meta_info.txt)**: Proposes ERC, an auxiliary objective to improve MoE router/expert coupling; conceptually related to training routing decisions with explicit objectives.

Additional related work (arXiv/URLs):
- **[Improving Neural Language Models with a Continuous Cache](https://arxiv.org/abs/1612.04426)**: Interpolates LM predictions with a cache distribution over recently seen tokens.
- **[kNN-LM: Generalization through Memorization](https://arxiv.org/abs/1911.00172)**: Augments pretrained LMs with nearest-neighbor retrieval from a datastore of hidden states.
- **[RETRO](https://arxiv.org/abs/2112.04426)**: Retrieval-augmented pretraining with chunk-level retrieval and cross-attention.
- **[Memorizing Transformers](https://arxiv.org/abs/2203.08913)**: Adds an ANN-indexed external memory to transformers for long-context recall.
- **[Transformer-XL](https://arxiv.org/abs/1901.02860)**: Segment-level recurrence / cache for long-range dependencies.
- **[Product Key Memory](https://arxiv.org/abs/1907.05242)**: Large key-value memory layer with efficient key-based retrieval.
- **[Neural Turing Machines](https://arxiv.org/abs/1410.5401)**: Differentiable external memory for neural networks.
- **[Memory Networks](https://arxiv.org/abs/1503.08895)**: Early memory-augmented architectures for QA.
- **[Pointer Networks](https://arxiv.org/abs/1506.03134)**: Copy-style mechanisms that output positions in an input sequence.
- **[Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)**: Mixes generation and copying for summarization.

Routing / gating in modular networks:
- **[Switch Transformers](https://arxiv.org/abs/2101.03961)**: Sparse MoE with routing and load-balancing auxiliary losses.
- **[GShard](https://arxiv.org/abs/2006.16668)**: Large-scale MoE training with load balancing.
- **[DeepSeekMoE](https://arxiv.org/abs/2401.06066)**: MoE training and routing strategies at scale.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)**: Reports MoE design/training choices, including load-balancing variants.
- **[Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)**: Balances expert loads without backpropagating a balancing loss (bias updates), relevant to router training stability.
- **[Expert Choice Routing](https://arxiv.org/abs/2202.09368)**: Inverts token-to-expert routing by letting experts select tokens, changing router credit assignment.
- **[Autonomy-of-Experts](https://arxiv.org/abs/2501.13074)**: Reduces dependence on a centralized router by letting experts self-select inputs.

Hashing / collisions as regularization:
- **[Hash Embeddings for Efficient Word Representations](https://arxiv.org/abs/1709.03933)**: Uses hashing with shared components for embeddings; highlights that collisions can act as structured regularization.

Credit assignment via counterfactual/attribution signals:
- **[Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning](https://arxiv.org/abs/2601.07408)**: Uses counterfactual/attribution signals to assign token-level credit in RL for reasoning; conceptually related to using counterfactual utility to supervise decision variables.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Deterministic n-gram conditional memory | Hash-based lookup of n-gram embeddings injected into residual stream with a learned gate | Engram; Lin 2026 controlled study | LM validation loss; route-stratified hot/cold loss; α-bucket analysis | Collisions; gate mis-calibration; unclear scaling laws |
| Scaled embedding layers | Scale embedding capacity via n-grams/f-grams and offloading | SCONE; Over-Tokenized Transformer | Perplexity; downstream tasks | Large storage; training complexity |
| Retrieval-augmented LMs | Retrieve contexts or hidden states from datastore/corpus | kNN-LM; RETRO; Memorizing Transformers | Perplexity; QA | Requires ANN infra; retrieval latency; leakage risks |
| Conditional computation (MoE) | Route tokens to experts; stabilize routing via auxiliary objectives | Switch; GShard; DeepSeekMoE; ERC | Downstream tasks; routing stats | Load balancing vs specialization tension; router collapse |

### Closest Prior Work

1. **Lin (2026) controlled Engram-Nine study** (`./references/A-Collision-Free-Hot-Tier-Extension-for-Engram-Style-Conditional-Memory-A-Controlled-Study-of-Training-Dynamics/meta/meta_info.txt`): Provides iso-parameter evidence that collision-free lookup does not yield meaningful loss gains at ~100M tokens, and surfaces a strong gate mis-calibration signal (α–loss inversion). Lin discusses “gating credit assignment” as a likely bottleneck and suggests auxiliary-loss directions, but does not propose a concrete, per-token causal supervision signal.

2. **Engram (Cheng et al., 2026)** (`./references/Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/meta/meta_info.txt`): Introduces the hashed n-gram conditional memory and context-aware gating objective (α computed from hidden state and memory key). The paper motivates gating as collision/noise suppression but does not validate whether α aligns with per-token utility during training, nor provide a diagnostic metric like oracle agreement.

3. **Switch Transformers (Fedus et al., 2021)**: Uses load-balancing auxiliary losses to prevent expert collapse in MoE routing. These losses regulate *token counts per expert* rather than supervising whether routing improves the task loss for a specific token, and they do not address the Engram-style setting where the alternative path is a deterministic memory injection rather than an expert network.

4. **ERC loss (Lv et al., 2025)** (`./references/Coupling-Experts-and-Routers-in-Mixture-of-Experts-via-an-Auxiliary-Loss/meta/meta_info.txt`): Proposes a router–expert coupling loss to improve specialization and routing stability in MoE. ERC supervises router/expert alignment via proxy activations and perturbations, but does not provide a forced-pass per-token causal utility target.

5. **Outcome-Grounded Advantage Reshaping (2026)**: Uses counterfactual/attribution signals for token-level credit assignment in RL reasoning. CGS differs by applying counterfactual *loss difference* supervision to a differentiable gate in a pretraining-style language modeling objective.

**Novelty Kill Search Summary (2026-02-17):** We searched for the exact combination “counterfactual supervision + (gate/router) + conditional memory/Engram”, including queries such as `Engram gating mismatch auxiliary loss`, `"counterfactual" gate supervision transformer`, `"conditional memory" gate supervision`, `gating credit assignment auxiliary loss 2025 2026`, and `counterfactual router auxiliary loss github`, and checked local KB summaries for `counterfactual.*(gate|router|routing)` and `oracle AUC`. We found causal-inference Transformers and MoE routing auxiliary losses, but no prior work that explicitly supervises Engram-style conditional-memory gates using forced-pass counterfactual loss differences (α forced to 0 vs 1). Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation for this problem | What we change | Why ours should win |
|---|---|---|---|---|
| Engram (2026) | Hashed n-gram lookup + context-aware gating | Gate can miscalibrate; no direct utility supervision | Add counterfactual utility supervision | Gate learns to allocate memory where it reduces loss |
| Lin 2026 Engram-Nine | Collision-free hot-tier + diagnostics | Diagnoses but does not fix gate credit assignment | Train gate with a causal per-token objective | Directly targets the measured failure mode |
| Switch Transformers (2021) | MoE routing + load balancing losses | Balances load; does not train routing for per-token utility | Supervise gate with forced-pass Δℓ | Optimizes the routing decision for causal loss reduction |
| ERC (2025) | Auxiliary loss to couple routers/experts | Not about conditional memory; no forced-pass utility | Apply utility-based supervision to memory gate | Aligns gating with causal benefit rather than early preferences |
| SCONE (2025) | Cached contextualized f-gram embeddings | Different mechanism (embedding scaling) | Keep hashed memory but fix gating | Preserves deterministic lookup while improving training dynamics |

---

## Experiments

### Experimental Setup

We target the smallest decisive experiment: reproduce Lin (2026) Hash-500K setting and test whether CGS (i) improves validation loss and (ii) improves gate–oracle agreement on held-out validation.

**Baseline Ladder (adapted for LM pretraining; prompting baselines are not applicable):**

In next-token prediction training from scratch, “prompting” and “best-of-N decoding” baselines are not meaningful comparisons because the metric is validation cross-entropy under a fixed model. Instead, we use architectural/training controls:

1. **Hash-500K (baseline)**: Reproduce Lin’s Hash-500K configuration (multi-head hash conditional memory with learned gates).
2. **Hash-500K + CGS (ours)**: Same architecture and training tokens, adding counterfactual gate supervision.
3. **Iso-compute control (baseline-longer)**: Train Hash-500K for additional steps to match the wall-clock compute of CGS (using measured throughput ratio).

Context-only (published, optional to rerun):
- **Nine-100/400K**: Lin’s collision-free hot-tier configuration (shows lookup precision is not the bottleneck at this scale).

Sanity check (optional):
- **No-CM (α forced 0)**: Remove conditional memory by forcing α=0 at all injection sites; verifies that the memory module is not trivially useless in this setting.

**Base Models / Codebase:**

| Model / code | Size | Download Link | Notes |
|---|---:|---|---|
| nanoGPT-based GPT-2 LM + Engram module | ~185M backbone (~313M total with memory tables) | https://github.com/karpathy/nanoGPT | Lin (2026) reports using nanoGPT; we match their config (12 layers, d=768, vocab=128,815, seq_len=1024, Engram layers 2/4/6) |

**Training Data:**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| FineWeb-Edu | LM training | 100M tokens | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | ODC-By 1.0 (per dataset card) |

Data split (Lin 2026): validation is the last 5% of token sequences (~5M tokens); training is the first 95% (~95M tokens) (see `./references/A-Collision-Free-Hot-Tier-Extension-for-Engram-Style-Conditional-Memory-A-Controlled-Study-of-Training-Dynamics/sections/Data Split.md`).

**Other Resources:**
- DeepSeek-V3 tokenizer (a 128,815-token vocabulary from DeepSeek’s V3 model family): https://huggingface.co/deepseek-ai/DeepSeek-V3

**Training hyperparameters (Lin 2026):** AdamW (β1=0.9, β2=0.95, wd=0.1), lr=6e-4 cosine (warmup 100 → 6e-5), batch size 4 with grad-accum 4 (effective 16), 5000 steps (~82M tokens), 3 seeds, 1×A100-40GB (see `sections/Training Hyperparameters.md`).

**Resource Estimate (evidence-based):**
- Lin reports training throughput ~1910 tok/s for Hash-500K (Table 3 in `sections/5.1 Main Finding Eliminating Collisions Yields No Significant Benefit.md`). One 82M-token run thus takes ~82e6/1910 ≈ **11.9 hours** on 1×A100-40GB (≈12 GPU-hours), excluding evaluation overhead.
- Budget for verification (3 seeds):
  - Hash-500K baseline: ~36 GPU-hours
  - CGS: baseline hours × (1 + overhead). With CGS every M=8 steps on one layer, overhead is expected to be modest; conservatively budget **+20%** → ~43 GPU-hours
  - Iso-compute baseline-longer: match CGS compute → ~43 GPU-hours
  - Total main runs: ~**122 A100 GPU-hours** (+ evaluation overhead, still well within the 768 GPU-hour cap)
- **GPU memory**: ≤ 40GB per GPU (Lin used A100-40GB)
- **API usage**: None

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| FineWeb-Edu LM eval | Next-token prediction on held-out web-text sequences | (1) validation loss (cross-entropy; lower is better), (2) hot_cold_delta, (3) oracle-AUC | val | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | nanoGPT eval loop + Lin’s stratified metrics + CGS oracle-AUC |

Diagnostics (from Lin 2026):
- **hot_cold_delta**: (val_loss_cold − val_loss_hot), where hot/cold are defined by top-N_hot n-gram frequency membership.
- **α-bucket loss**: average loss by α intervals (Table 6 style).

Our additional diagnostic:
- **Oracle-AUC**: On held-out validation, compute oracle label z_t^{(L)} = 1[Δℓ_t^{(L)} > 0] from forced-pass Δℓ on the frozen model, and report AUC between predicted α̂_t^{(L)} and z_t^{(L)}. (Higher is better; 0.5 is random.)

### Main Results

#### Results Table

| Method | Base Model | Benchmark | val_loss (mean±std) | hot_cold_delta (mean±std) | oracle-AUC (mean±std) | Source | Notes |
|---|---|---|---|---|---|---|---|
| Hash-500K | GPT-2-ish + DeepSeek-V3 vocab | FineWeb-Edu val | 4.4809 ± 0.0082 | +0.07 (std N/A) | TBD | [Lin 2026](./references/A-Collision-Free-Hot-Tier-Extension-for-Engram-Style-Conditional-Memory-A-Controlled-Study-of-Training-Dynamics/meta/meta_info.txt) | val_loss and delta copied from Table 3 (`sections/5.1 ...`); oracle-AUC must be computed |
| Nine-100/400K | same | FineWeb-Edu val | 4.4799 ± 0.0123 | +0.10 (std N/A) | TBD | [Lin 2026](./references/A-Collision-Free-Hot-Tier-Extension-for-Engram-Style-Conditional-Memory-A-Controlled-Study-of-Training-Dynamics/meta/meta_info.txt) | Context-only; optional to reproduce (MPHF engineering) |
| Iso-compute Hash-500K (longer) | same | FineWeb-Edu val | TBD | TBD | TBD | - | Train Hash-500K for additional steps to match CGS wall-clock |
| **Ours: Hash-500K + CGS** | same | FineWeb-Edu val | TBD | TBD | TBD | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| CGS (less frequent) | Increase CGS interval M (e.g., 8 → 32) | If Δℓ is stable, modest degradation; tests compute/signal tradeoff |

### Experimental Rigor

Variance & reproducibility:
- Run each main condition with **seeds=[42, 123, 456]** and report mean±std.
- Verify reproduction of Lin’s Hash-500K val_loss within ±0.01.

Key confounders and controls:
- **Extra compute vs better algorithm**: include iso-compute baseline-longer to test whether “just train longer” matches CGS.
- **RNG consumption (dropout) in extra forward passes**: save/restore RNG state around counterfactual passes.
- **Avoidance vs improved routing**: report mean α̂ on oracle-positive tokens (Δℓ>0) vs oracle-negative tokens (Δℓ≤0), and the distribution of α̂ (histogram) to detect collapse.

Sanity checks (automated):
- **Random-label CGS**: replace y_t^{(L)} with a shuffled version within the batch for one short run (e.g., 500 steps, 1 seed). Expect oracle-AUC≈0.5 and no val_loss improvement.

---

## Success Criteria

**Hypothesis** (directional):
CGS improves overall validation loss and increases held-out oracle-AUC, indicating that gates allocate memory where it causally improves prediction.

**Decision Rule** (concrete):
- **Proceed** if, compared to the reproduced Hash-500K baseline, CGS achieves:
  - val_loss improvement ≥ **0.01** (≈1+ baseline std) across 3 seeds, **and**
  - oracle-AUC increases by ≥ **0.05** and reaches **≥0.60** on held-out validation.
- **Pivot** if oracle-AUC improves (≥0.05) but val_loss does not: try (i) a regression target (y_t proportional to Δℓ magnitude), or (ii) enabling CGS only after step 2000 to avoid early overfitting to transient dynamics.
- **Refute** if val_loss gain is within noise and oracle-AUC does not improve (or α collapses toward always-off), concluding that forced-pass counterfactual supervision does not fix gating credit assignment in this regime.

---

## Impact Statement

If successful, CGS provides a simple, general training objective to fix gate credit assignment in deterministic conditional-memory modules. This could make Engram-style architectures more reliable for practitioners exploring large offloaded memories, by reducing dependence on brittle heuristics (collision management, hand-tuned gating) and by providing a diagnostic oracle-AUC metric to detect when gates are learning meaningful routing.

---

## References

Local (proposal artifacts):
- [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](./references/Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models/meta/meta_info.txt) - Cheng et al., 2026
- [A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics](./references/A-Collision-Free-Hot-Tier-Extension-for-Engram-Style-Conditional-Memory-A-Controlled-Study-of-Training-Dynamics/meta/meta_info.txt) - Lin, 2026
- [Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](./references/Coupling-Experts-and-Routers-in-Mixture-of-Experts-via-an-Auxiliary-Loss/meta/meta_info.txt) - Lv et al., 2025
- [Scaling Embedding Layers in Language Models](./references/Scaling-Embedding-Layers-in-Language-Models/meta/meta_info.txt) - Yu et al., 2025
- [Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling](./references/Over-Tokenized-Transformer-Vocabulary-is-Generally-Worth-Scaling/meta/meta_info.txt) - Huang et al., 2025

URLs (non-local):
- [Improving Neural Language Models with a Continuous Cache](https://arxiv.org/abs/1612.04426) - Grave et al., 2017
- [kNN-LM: Generalization through Memorization](https://arxiv.org/abs/1911.00172) - Khandelwal et al., 2020
- [RETRO](https://arxiv.org/abs/2112.04426) - Borgeaud et al., 2022
- [Memorizing Transformers](https://arxiv.org/abs/2203.08913) - Wu et al., 2022
- [Transformer-XL](https://arxiv.org/abs/1901.02860) - Dai et al., 2019
- [Product Key Memory](https://arxiv.org/abs/1907.05242) - Lample et al., 2019
- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) - Graves et al., 2014
- [Memory Networks](https://arxiv.org/abs/1503.08895) - Weston et al., 2015
- [Pointer Networks](https://arxiv.org/abs/1506.03134) - Vinyals et al., 2015
- [Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) - See et al., 2017
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Fedus et al., 2021
- [GShard](https://arxiv.org/abs/2006.16668) - Lepikhin et al., 2020
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) - Dai et al., 2024
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - DeepSeek-AI, 2024
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) - 2024
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368) - Zhou et al., 2022
- [Autonomy-of-Experts](https://arxiv.org/abs/2501.13074) - 2025
- [Hash Embeddings for Efficient Word Representations](https://arxiv.org/abs/1709.03933) - Svenstrup et al., 2017
- [Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning](https://arxiv.org/abs/2601.07408) - 2026
