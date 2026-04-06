# untitled

# SVD-Equalized LoRA for Continual Learning (a post-hoc, norm-preserving spectral intervention)

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models are often adapted to a stream of tasks (e.g., a sequence of customer domains, new product categories, or new instruction sets). A common adaptation strategy is **parameter-efficient fine-tuning (PEFT)**, where the base model weights are frozen and only a small set of additional parameters is trained. **LoRA** (Low-Rank Adaptation) is a widely used PEFT method that adds a low-rank weight update to a linear layer.

A practical problem is **catastrophic forgetting**: after learning later tasks, the model’s performance on earlier tasks degrades. Recent continual learning (CL) methods for language models span replay-based approaches (store or synthesize past examples) and regularization/subspace approaches (constrain updates to reduce interference). However, replay methods can be undesirable in privacy-constrained settings, and some sophisticated PEFT-CL methods introduce non-standard optimization constraints.

This proposal focuses on a minimal, training-free intervention inspired by a recent observation: **LoRA updates tend to have highly imbalanced singular value spectra**, meaning that most update energy concentrates in a few dominant directions.

### The Problem

Evidence suggests that spectral imbalance in low-rank updates amplifies cross-task interference.

- **[Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation](./references/Spectral-Imbalance-Causes-Forgetting-in-Low-Rank-Continual-Adaptation/meta/meta_info.txt)** shows that LoRA updates have long-tailed singular values and that *smoothing singular values* before merging multiple task adapters reduces interference in a controlled merging setup (their Fig. 2b; see “Observation: Imbalance among Components Amplifies Interference”).
- In contrast, their proposed solution (EBLoRA) uses a training-time reparameterization and manifold optimization to encourage balanced singular values, which is more complex than many standard LoRA pipelines.

The central technical question is:

**Can we get a meaningful fraction of the “balanced spectra → lower interference” benefit via a purely post-hoc transformation of the learned LoRA update, without changing the optimizer or requiring replay?**

A key confound is that naive “smoothing” can also shrink the update magnitude (Frobenius norm; \(||X||_F\) is the square root of the sum of squared matrix entries), which could reduce forgetting simply by making updates smaller. We therefore propose a **norm-preserving** equalization that isolates the effect of *energy distribution across rank components*, not total update size.

### Key Insight and Hypothesis

**Hypothesis:** In sequential LoRA continual learning where each task contributes a low-rank update \(\Delta W_t\), replacing the singular values of \(\Delta W_t\) by an equal-energy spectrum **at fixed Frobenius norm** (i.e., distributing \(||\Delta W_t||_F^2\) uniformly across rank components) will reduce cross-task interference and improve backward transfer, while largely preserving overall performance.

**Why this might work:** If forgetting arises because a small number of dominant low-rank directions overwrite features reused by earlier tasks, then forcing the update energy to be less concentrated should make each task update less disruptive and more robust to later merges.

**Why this might fail:** Dominant singular directions might carry most of the task-relevant signal; equalization could dilute that signal and reduce performance on the current task, leading to lower overall performance despite reduced interference.

---

## Proposed Approach

### Overview

We propose **SVD-Equalized LoRA (SVD-EQ)**: at each task boundary in sequential LoRA continual learning, we compute the singular values of the current effective LoRA update \(\Delta W = B A\) for each adapted linear layer, replace them with an equal-energy spectrum **that preserves \(||\Delta W||_F\)**, and reparameterize the LoRA factors accordingly before continuing training on the next task (i.e., a training-free, post-hoc adapter update).

This is a **post-hoc** and **parameter-free** operation applied once per task boundary (not every training step). It requires only small \(r\times r\) matrix decompositions because \(\Delta W\) is rank \(r\).

### Method Details

#### Setting and notation

For a linear layer weight \(W \in \mathbb{R}^{d_{out}\times d_{in}}\), LoRA learns a rank-\(r\) update
\[
\Delta W = B A, \quad B\in\mathbb{R}^{d_{out}\times r},\; A\in\mathbb{R}^{r\times d_{in}}.
\]

After training task \(t\), we have \(\Delta W_t\) for each adapted layer.

#### Efficient SVD for low-rank \(\Delta W\)

We avoid an expensive full SVD of \(\Delta W\) by reducing to an \(r\times r\) core matrix:

1. Compute QR decompositions:
\[
B = Q_B R_B,\quad A^\top = Q_A R_A
\]
where \(Q_B\in\mathbb{R}^{d_{out}\times r}\), \(Q_A\in\mathbb{R}^{d_{in}\times r}\), and \(R_B, R_A\in\mathbb{R}^{r\times r}\).

2. Form the core matrix \(C = R_B R_A^\top \in \mathbb{R}^{r\times r}\) and compute its SVD:
\[
C = U \Sigma V^\top.
\]

3. Then the SVD of \(\Delta W\) is
\[
\Delta W = (Q_B U)\, \Sigma\, (Q_A V)^\top.
\]

#### Norm-preserving singular-value equalization

Let \(\sigma\in\mathbb{R}^r\) be the singular values (diagonal of \(\Sigma\)). Define
\[
\bar\sigma_{\mathrm{rms}} \;=\; \frac{\|\sigma\|_2}{\sqrt{r}} \;=\; \frac{\|\Delta W\|_F}{\sqrt{r}}.
\]
We construct \(\Sigma_{\mathrm{eq}} = \bar\sigma_{\mathrm{rms}} I_r\), i.e., all singular values are equal and the Frobenius norm is preserved:
\(\|\Sigma_{\mathrm{eq}}\|_F = \|\Sigma\|_F\).

The equalized update is
\[
\Delta W_{\mathrm{eq}} = (Q_B U)\, \Sigma_{\mathrm{eq}}\, (Q_A V)^\top.
\]

Implementation options:
- **Stay-in-LoRA (default)**: after computing \(\Delta W_{\mathrm{eq}}\), set new factors \(B'=(Q_B U)\Sigma_{\mathrm{eq}}^{1/2}\) and \(A'=\Sigma_{\mathrm{eq}}^{1/2}(Q_A V)^\top\) so that \(B'A'=\Delta W_{\mathrm{eq}}\). Then continue continual learning using the same LoRA module (no weight merging required).
- **Merge-after-task (optional)**: materialize the update into the base weights (PEFT `merge_and_unload`) and start the next task from the updated base.

We will use **stay-in-LoRA** by default, since FOREVER’s released codebase appears to keep training under the same LoRA framework across tasks (Appendix E: “all methods are implemented using the LoRA framework”).

#### Mechanism diagnostic: spectral concentration

For each adapted layer, define spectral concentration
\[
\rho = \frac{\sigma_1^2}{\|\sigma\|_2^2} \in [1/r, 1].
\]
SVD-EQ deterministically maps \(\rho \to 1/r\). We will test whether higher baseline \(\rho\) predicts larger reductions in forgetting.

### Key Innovations

1. **Post-hoc, norm-preserving spectrum shaping for sequential LoRA**: Unlike EBLoRA, we do not modify the training objective or use constrained optimization; unlike naive smoothing, we preserve \(||\Delta W||_F\) to avoid a magnitude confound.
2. **Cheap and fully automated**: Because LoRA updates are low-rank, the intervention uses only \(r\times r\) decompositions and can be applied once per task.
3. **Falsifiable mechanism test**: We pre-register a measurable internal quantity (\(\rho\)) that should predict when the method helps.

---

## Related Work

### Field Overview

Continual learning for language models includes: (i) **replay-based** methods that mix past examples or synthesize rehearsal data, (ii) **regularization-based** methods that penalize changes to important parameters, (iii) **subspace / orthogonality** methods that constrain update directions to reduce interference, and (iv) **model merging** methods that combine task-specific updates post-training.

This proposal is closest to the intersection of **parameter-efficient CL** and **spectral analyses of low-rank updates**. Recent work argues that low-rank updates can be highly anisotropic and that controlling their geometry can reduce interference. Our focus is a minimal post-processing step at task boundaries.

### Related Papers

- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**: Introduces LoRA, representing weight updates as a low-rank factorization.
- **[FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt)**: Provides a strong LoRA-based CL benchmark setup and baselines on Standard CL / Long Sequence / SuperNI.
- **[Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation](./references/Spectral-Imbalance-Causes-Forgetting-in-Low-Rank-Continual-Adaptation/meta/meta_info.txt)**: Argues that imbalanced singular values in LoRA updates amplify interference and proposes training-time energy balancing.
- **[Revisiting Weight Regularization for Low-Rank Continual Learning](./references/REVISITING-WEIGHT-REGULARIZATION-FOR-LOW-RANK-CONTINUAL-LEARNING/meta/meta_info.txt)**: Studies EWC-style regularization in low-rank continual learning settings.
- **[LoRA-Squeeze: Simple and Effective Post-Tuning and In-Tuning Compression of LoRA Modules](./references/LoRA-Squeeze-Simple-and-Effective-Post-Tuning-and-In-Tuning-Compression-of-LoRA-Modules/meta/meta_info.txt)**: Uses SVD-based transformations mainly for LoRA compression (rank reduction / parameter pruning).
- **[Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/abs/2310.14152)**: Learns task updates in orthogonal low-rank subspaces to mitigate forgetting without replay.
- **[SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models](https://arxiv.org/abs/2401.08295)** (ACL 2024): Uses shared attention to align learning and selection over a pool of PET blocks across tasks.
- **[MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models](https://arxiv.org/abs/2402.12851)**: Uses a MoE-style router over multiple LoRA experts.
- **[Unlocking Continual Learning Abilities in Language Models (MIGU)](https://arxiv.org/abs/2406.17245)** (Findings EMNLP 2024): Rehearsal-free, task-ID-free continual learning via magnitude-based gradient masking.
- **[Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal (SSR)](https://arxiv.org/abs/2403.01244)** (ACL 2024): Generates synthetic rehearsal instances to reduce forgetting without storing real past data.
- **[Overcoming Catastrophic Forgetting in Neural Networks](https://www.pnas.org/doi/10.1073/pnas.1611835114)**: Classic Elastic Weight Consolidation (EWC) regularization method.
- **[Riemannian Walk for Incremental Learning](https://arxiv.org/abs/1801.10112)**: Introduces OP/BWT-style metrics for continual learning.
- **[Continual Learning of Natural Language Processing Tasks: A Survey](https://arxiv.org/abs/2112.00406)**: Survey defining common metrics including backward transfer.
- **[Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)**: Introduces task-vector arithmetic, a foundation for later model-merging work.
- **[TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)**: A model-merging method that trims and resolves sign conflicts in task vectors.
- **[Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)](https://arxiv.org/abs/2311.03099)**: Randomly drops delta weights then rescales to preserve expected norm; often used as a merge pre-processing step.
- **[No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces](https://arxiv.org/abs/2502.04959)** (ICML 2025): Shows isotropic (flattened-spectrum) task vectors improve merge robustness.
- **[Model merging with SVD to tie the Knots (KnOTS)](https://arxiv.org/abs/2410.19735)** (ICLR 2025): Uses joint SVD to align LoRA task updates before applying standard merging rules.
- **[LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228)**: Shows LoRA differs from full fine-tuning via “intruder” directions and studies post-hoc spectral edits.
- **[Learning Continually by Spectral Regularization](https://arxiv.org/abs/2406.06811)**: Uses spectral-norm regularization to preserve trainability in continual learning.

> Note: All arXiv IDs above were resolved (no placeholders) as of 2026-02-21.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Replay-based CL | Mix past (real/synth) data during new-task training | FOREVER, MixReplay, SSR | Standard CL / Long Sequence / SuperNI | Needs memory buffer or synthetic generation; may be costly or privacy-sensitive |
| Regularization-based CL | Penalize changes to parameters important for past tasks | EWC, EWC-LoRA | Standard CL | Importance estimation may be noisy; can reduce plasticity |
| Subspace / orthogonality CL | Constrain updates to orthogonal subspaces across tasks | O-LoRA | Standard CL | Still adds constraints and/or per-task structure |
| Model merging | Combine task updates post-training to reduce interference | Task Arithmetic, TIES, DARE | Merging benchmarks | Not always aligned with sequential CL training |
| Spectral methods for low-rank updates | Diagnose/control singular values or spectral norms | EBLoRA, Spectral Regularization, Intruder directions | VLM CL suites; various | Training-time methods add optimizer complexity; post-hoc methods can confound magnitude |

### Closest Prior Work

1. **EBLoRA (Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation)**: Shows that LoRA updates have long-tailed singular values and demonstrates that singular-value smoothing before merging multiple task adapters reduces interference (controlled merging; their Fig. 2b). Their main method enforces energy balance during training via reparameterization and constrained optimization, whereas we propose a post-hoc transform that can be added to existing LoRA training code without changing the optimizer.

2. **LoRA vs Full Fine-tuning: An Illusion of Equivalence**: Studies spectral properties of LoRA and proposes post-hoc edits that target specific high-impact directions (“intruder” directions) to reduce forgetting measured by pretraining pseudo-loss. Our method does not identify intruder directions; it applies a uniform equalization to the singular values of \(\Delta W\) per task and targets sequential-task interference metrics (OP/BWT).

3. **LoRA-Squeeze**: Uses SVD primarily to compress LoRA modules (reduce rank / prune). Our method keeps rank fixed and targets interference/forgetting rather than compression.

4. **FOREVER**: Provides the primary evaluation harness and baselines; our method is a drop-in modification to the “sequential LoRA” baseline within that framework.

**Novelty Kill Search Summary:** Searched for combinations of “post-hoc singular value equalization/flattening + LoRA + continual learning”, “LoRA singular value clipping regularization”, and checked for related terms in local proposal/paper KB. No prior work explicitly applying **norm-preserving singular-value equalization at task boundaries** for sequential LoRA continual learning was found as of 2026-02-21. (Full query log is in `notes.md`.)

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| EBLoRA | Training-time factorization + constraints to balance singular values | Requires non-standard optimization; more engineering | Post-hoc equalization at task boundaries | Captures the “balanced spectrum” effect with minimal code changes |
| Intruder-direction shrink | Post-hoc spectral edits targeting specific directions | Requires identifying a special subspace; different forgetting metric | No subspace identification; uniform equalization | Parameter-free and directly targets sequential-task interference |
| LoRA-Squeeze | Post-hoc SVD mainly for compression | Changes capacity (rank) and confounds with pruning | Keep rank and \(||\Delta W||_F\) fixed | Tests the spectral-concentration mechanism without capacity/magnitude confounds |
| O-LoRA | Orthogonal low-rank subspaces across tasks | Adds orthogonality constraints / per-task structure | Single post-hoc transform of a standard LoRA update | Potentially similar forgetting reduction without constraints |
| FOREVER | Replay scheduling + regularization | Uses memory buffer and replay logic | Orthogonal add-on to sequential LoRA | Can help when replay is disallowed; near-zero overhead |

---

## Experiments

### Experimental Setup

**Goal:** Test whether SVD-EQ improves forgetting relative to vanilla sequential LoRA under a standard, reproducible CL protocol.

**Evaluation harness:** Use the public setup from **FOREVER**.

- **Code**: https://anonymous.4open.science/r/FOREVER-C7D2
- **Paper**: [FOREVER](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt)

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen3-0.6B | 0.6B | https://huggingface.co/Qwen/Qwen3-0.6B | Used as “Qwen3-0.6B backbone” in FOREVER |
| Qwen3-0.6B-Base (fallback) | 0.6B | https://huggingface.co/Qwen/Qwen3-0.6B-Base | Use if FOREVER repo expects base model |

**Training protocol (from FOREVER):**
- Sample **1000 training instances per task**; reserve **500 instances per class** for evaluation.
- Train tasks sequentially for **10 epochs per task**.
- LoRA hyperparameters: **rank r=8**, **α=32**, **dropout=0.05**, applied to the transformer block’s **query/value projection linear layers** (FOREVER Appendix G).

**Baseline Ladder (decisive comparison):**
1. **Sequential LoRA (memory-free baseline)**: replicate FOREVER’s “Fine-tuning” setting (Table 1): train LoRA sequentially on each task with **no replay buffer** (Memory-Based ✗). Use the released task orders `dataset_id ∈ {4,5,6}` from `utils/dataset_order.py`.
2. **Ours: Sequential LoRA + SVD-EQ**: same as (1), but apply SVD-EQ at each task boundary by reparameterizing each layer’s LoRA update to have an equal-energy singular spectrum at fixed \(||\Delta W||_F\).

*(Reporting-only / context)*: we will also cite the published Standard-CL numbers from FOREVER Table 1 for EWC, O-LoRA, MixReplay, SAPT, MIGU, SSR, Recurrent-KIF, AIMMerging, VBM, and FOREVER.

**Resource Estimate**:
- From FOREVER Table 6, Qwen3-0.6B training time is ~**1.3–1.4 min/epoch** (8 GPUs). In the released FOREVER code, the “Standard CL” setting corresponds to dataset orders `dataset_id ∈ {4,5,6}` in `utils/dataset_order.py`, i.e., a **4-task** sequence over {dbpedia, amazon, yahoo, agnews}. (Appendix F also mentions Yelp; we follow the released code + Table 1.) With 4 tasks × 10 epochs = 40 epochs → ~52–56 min wall-clock per run.
- Main experiment: 2 training conditions (baseline vs SVD-EQ) × 3 seeds ≈ 6 runs.
  - Estimated wall-clock: ~6 hours total on 8 GPUs.
  - Estimated GPU-hours: ~8 GPUs × 6 hours ≈ **48 GPU-hours**, plus any inference-only sanity checks (<5 GPU-hours) → comfortably within the 768 GPU-hour budget.
- GPU memory: Qwen3-0.6B fits easily on a single 80GB GPU; we can run with data parallelism as needed.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| Standard CL (code: 4 tasks) | In released FOREVER code: {dbpedia, amazon, yahoo, agnews} (dataset_id ∈ {4,5,6} in `utils/dataset_order.py`) | OP (higher better), BWT (less negative better), per-task accuracy | fixed held-out per task | bundled in FOREVER repo under `data_longsequence/` | FOREVER repo (above) |

Metric definitions (from FOREVER):
- \(a_{i,j}\): accuracy on task \(i\) after training up to task \(j\).
- **OP** = \(\frac{1}{K}\sum_{i=1}^K a_{i,K}\).
- **BWT** = \(\frac{1}{K-1}\sum_{i=1}^{K-1}(a_{i,K} - a_{i,i})\).

### Main Results

We will report mean±std over 3 seeds.

#### Results Table

| Method | Base Model | Benchmark | OP (mean±std) | BWT (mean±std) | Source | Notes |
|--------|------------|-----------|---------------|----------------|--------|-------|
| Prompt-only (zero-shot) | Qwen3-0.6B | Standard CL | **TBD** | **TBD** | - | Needs re-run |
| Prompt-only best-of-5 | Qwen3-0.6B | Standard CL | **TBD** | **TBD** | - | Needs re-run |
| Sequential LoRA (no replay) | Qwen3-0.6B | Standard CL | 47.2 | -12.6 | [FOREVER Table 1](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main%20Results.md) | Published (avg over task orders; reproduction optional) |
| **Sequential LoRA + SVD-EQ (ours)** | Qwen3-0.6B | Standard CL | **TBD** | **TBD** | - | To be verified |
| O-LoRA | Qwen3-0.6B | Standard CL | 59.4 (1 run) | -7.9 (1 run) | [FOREVER](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt) | Published reference |
| VBM | Qwen3-0.6B | Standard CL | 71.5 (1 run) | -5.2 (1 run) | [FOREVER](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt) | Published reference |
| FOREVER | Qwen3-0.6B | Standard CL | 72.9 (1 run) | -4.7 (1 run) | [FOREVER](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt) | Published reference |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | RMS-equalized singular values; \(||\Delta W||_F\) preserved per layer | Best forgetting reduction at similar OP |
| Mean-smoothing (non-norm-preserving) | Replace \(\sigma\) by mean(\(\sigma\)) as in EBLoRA’s controlled experiment | If it helps mainly via shrinking updates, OP may drop and BWT may improve less once norm is controlled |
| Random-spectrum control | Randomly permute \(\sigma\) values (keeps multiset but changes alignment to singular vectors ordering) | Should be near-identical to baseline if only spectrum concentration matters |

### Experimental Rigor

**Variance & Reproducibility:**
- Run Seq-LoRA and SVD-EQ with **3 seeds**: `seeds=[42,123,456]`.

**Validity & Controls (top confounders):**
1. **Update magnitude confound**: Our method preserves \(||\Delta W||_F\) per layer by construction; additionally, the “mean-smoothing” ablation tests the shrinkage explanation.
2. **Implementation mismatch vs FOREVER**: Use the official FOREVER codebase and only add a task-boundary post-processing step.
3. **Task-order sensitivity**: Use the same multiple task orders used in FOREVER (they report averaging over task orders); if compute is limited, run at least 3 task orders and report variance.

### Analysis (Optional)

- **Mechanism check**: Compute spectral concentration \(\rho\) for Seq-LoRA and correlate with BWT improvement from SVD-EQ across task orders. Expect higher \(\rho\) → larger gain.

---

## Success Criteria

**Hypothesis** (directional): SVD-EQ will reduce forgetting (less negative BWT) relative to Seq-LoRA, with minimal OP loss.

**Decision Rule** (concrete):
- **Continue/Proceed**: On Standard CL, SVD-EQ improves BWT by a margin outside Seq-LoRA’s seed std (and does not reduce OP by more than 1 std), across ≥3 seeds.
- **Pivot**: If BWT improves but OP drops substantially, test partial equalization (interpolate \(\Sigma\) toward \(\Sigma_{eq}\)) as a follow-up.
- **Refute**: If SVD-EQ does not improve BWT beyond noise, or consistently reduces OP, abandon post-hoc equalization as a useful intervention for sequential LoRA CL.

---

## Impact Statement

If successful, SVD-EQ would provide a **training-free** drop-in step at task boundaries that improves continual learning for LoRA-adapted language models without requiring replay data or custom optimizers. This could be adopted in privacy-constrained sequential fine-tuning pipelines and as a cheap stabilizer for any workflow that merges low-rank task updates into a shared model.

---

## References

- [FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt) - Feng et al., 2026
- [Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation](./references/Spectral-Imbalance-Causes-Forgetting-in-Low-Rank-Continual-Adaptation/meta/meta_info.txt) - Gu et al., 2026
- [LoRA-Squeeze: Simple and Effective Post-Tuning and In-Tuning Compression of LoRA Modules](./references/LoRA-Squeeze-Simple-and-Effective-Post-Tuning-and-In-Tuning-Compression-of-LoRA-Modules/meta/meta_info.txt) - Vuli’c et al., 2026
- [Revisiting Weight Regularization for Low-Rank Continual Learning](./references/REVISITING-WEIGHT-REGULARIZATION-FOR-LOW-RANK-CONTINUAL-LEARNING/meta/meta_info.txt) - Zheng et al., 2026
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/abs/2310.14152) - Wang et al., 2023
- [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228) - 2024
- [Learning Continually by Spectral Regularization](https://arxiv.org/abs/2406.06811) - 2024
- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) - Ilharco et al., 2023
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) - 2023
- [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)](https://arxiv.org/abs/2311.03099) - Yu et al., 2023
