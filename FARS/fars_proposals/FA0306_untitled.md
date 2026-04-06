# untitled

# Does FGGM’s TRACE Advantage Over MIGU Survive TRACE’s Official “Order 2” Stress Test?

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Continual learning for large language models (LLMs) studies how to **sequentially fine-tune** a single model on a stream of tasks or datasets without catastrophically forgetting earlier capabilities. This setting is increasingly relevant for real deployments, where teams update an existing instruction-tuned checkpoint repeatedly as new domains, languages, and product requirements arrive.

A widely used benchmark for this setting is **TRACE** (**[TRACE](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models_1/meta/meta_info.txt)**), which defines an 8-task sequential instruction-tuning stream (C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten). TRACE evaluates target-task continual-learning performance using **Overall Performance (OP)** (average performance on all tasks learned so far) and **Backward Transfer (BWT)** (average change in earlier-task performance after learning later tasks; negative BWT indicates forgetting). TRACE also evaluates post-training degradation on “general ability” benchmarks (MMLU, BBH, TyDiQA, PIQA, BoolQ, GSM8K).

Recent work has proposed **replay-free** continual-learning algorithms (methods that do not store or replay data from previous tasks) for LLM fine-tuning that aim to reduce forgetting without storing old task data. Two representative methods are **MIGU** (Magnitude-based Gradient Updating) (**[MIGU](./references/Unlocking-Continual-Learning-Abilities-in-Language-Models/meta/meta_info.txt)**) and **FGGM** (Fisher-Guided Gradient Masking) (**[FGGM](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/meta/meta_info.txt)**). On TRACE’s default task order, FGGM reports higher TRACE-OP than MIGU for Qwen2-1.5B (46.00 vs 44.08; Table 1 in FGGM).

### The Problem

Most LLM continual-learning papers (including MIGU and FGGM) report results on **a single fixed task order**, primarily due to compute cost. However, TRACE itself shows that **task order can dominate outcomes**. In TRACE Appendix D.7 (“Different Order”), changing only the task order can drastically reduce performance: for LLaMA-7B-chat under sequential fine-tuning (SeqFT), OP drops from **48.7** (default order) to **32.9** under TRACE’s official alternative **Order 2**, and BWT worsens to **−0.221** (**[TRACE D.7](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models_1/sections/D.7%20Different%20Order.md)**).

This raises a decision-relevant evaluation question:

> Are recent replay-free “SOTA” claims on TRACE (e.g., FGGM > MIGU) robust to TRACE’s own official alternative order, or are they artifacts of the default curriculum?

If method rankings flip across plausible orders, then (i) TRACE leaderboards based on one order may be misleading, and (ii) practitioners who cannot control their update stream order may deploy the wrong algorithm.

### Key Insight and Hypothesis

**Key insight.** FGGM and MIGU both modify gradient updates, but their mechanisms differ in a way that plausibly increases **path dependence**. MIGU masks gradients using **batch-level activation magnitude patterns** in linear layers, while FGGM computes a diagonal Fisher importance estimate on each task (a parameter-importance score based on how sensitive the model likelihood is to perturbing each parameter) and applies a **hard binary mask** (each parameter is either fully trainable or fully frozen for that task) that updates only the top (1−α) fraction of “important” parameters. With a fixed masking rate (α=0.7 in FGGM), early tasks in the stream can strongly shape which functional units receive updates, and highly atypical early tasks (e.g., NumGLUE arithmetic reasoning) may bias subsequent learning.

**Hypothesis (ranking robustness across method families).** When evaluated on TRACE’s official **Order 2** (NumGLUE-cm → NumGLUE-ds → FOMC → 20Minuten → C-STANCE → Py150 → MeetingBank → ScienceQA), the **relative ranking** among three common training rules for LLM continual fine-tuning — unconstrained sequential fine-tuning (**SFT**), magnitude-based masking (**MIGU**), and Fisher-based hard masking (**FGGM**) — will change compared to the published TRACE default-order ranking. In particular, FGGM’s reported advantage over MIGU on the default order will **not transfer** and will **reverse** (FGGM < MIGU on final-step TRACE-OP).

Why we could be wrong:
- Both methods may be similarly order-sensitive, preserving the FGGM > MIGU ranking even when absolute performance drops.
- The default-order advantage might reflect a genuine stability–plasticity improvement that generalizes across orders.
- Implementation details (prompt templates, evaluation scripts) could introduce noise; we mitigate this with a default-order sanity check and shared training pipeline.

---

## Proposed Approach

### Overview

We propose a minimal, fully automated **held-out-order audit** on TRACE:

1. Implement **SFT (unconstrained sequential fine-tuning)**, **MIGU**, and **FGGM** inside the same TRACE-style sequential fine-tuning pipeline (shared data preprocessing, optimizer, and evaluation code).
2. Run all methods on TRACE’s official **Order 2** and compare:
   - **Final-step TRACE-OP (OP\_T)** as primary metric
   - **Final-step BWT (BWT\_T)** as secondary metric
3. Decide whether the **default-order method ranking** (FGGM > MIGU among replay-free masking methods; and relative to SFT) transfers to Order 2.

### Method Details

#### Setting: TRACE continual instruction tuning
- **Tasks (TRACE)**: C-STANCE, FOMC, MeetingBank, Py150 (code completion), ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten.
- **Data sizes**: TRACE standardizes each task to 5,000 train and 2,000 test examples (**[TRACE Appendix A](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models_1/sections/Appendices%20A%20Implementation%20Details.md)**).
- **Evaluation**: per-task metrics (Accuracy/F1/ROUGE-L/BLEU/SARI) aggregated into TRACE-OP following TRACE’s OP definition.

#### Orders
- **Default order (published in FGGM)**: C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds → 20Minuten (**[FGGM Sec 4.1](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/4.1%20Experimental%20Setup.md)**).
- **Order 2 (audit target)**: NumGLUE-cm → NumGLUE-ds → FOMC → 20Minuten → C-STANCE → Py150 → MeetingBank → ScienceQA (**[TRACE D.7](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models_1/sections/D.7%20Different%20Order.md)**).

#### Base model
- **Qwen2-1.5B** (to match FGGM Table 1 setting). Download: https://huggingface.co/Qwen (verifier should select the closest available Qwen2-1.5B checkpoint used in FGGM).

#### Training hyperparameters (matched to TRACE/FGGM)
From TRACE Appendix A and FGGM Sec 4.1:
- Optimizer: AdamW
- LR: 1e−5 with linear decay to 0 (LoRA baseline uses 1e−4, but we focus on full-parameter methods)
- Epochs per task: 5 / 3 / 7 / 5 / 3 / 5 / 5 / 7 (aligned to TRACE scripts as cited by FGGM)
- Batch size: 128 (TRACE Appendix A)
- Weight decay: 0 (TRACE Appendix A)
- Precision: BF16

#### MIGU implementation (replay-free baseline)
MIGU masks gradients of linear layers using cached output-magnitude statistics:
- Forward pass: compute per-output magnitude vector \(n\) based on L1 norms of dot products (MIGU Sec 3.2).
- Backward pass: construct a binary mask that keeps the top \((1-T)\) fraction of outputs by magnitude (threshold ratio \(T=0.7\) as in MIGU/FGGM), and apply it element-wise to gradients (**[MIGU method](./references/Unlocking-Continual-Learning-Abilities-in-Language-Models/sections/3.2%20MIGU%20-%20MagnItude-based%20Gradient%20Updating%20for%20Continual%20Learning..md)**).
- Practical detail: average magnitudes across tokens in a batch to generate the mask (**[MIGU in practice](./references/Unlocking-Continual-Learning-Abilities-in-Language-Models/sections/3.3%20MIGU%20in%20Practice.md)**).

#### FGGM implementation (method under audit)
FGGM computes a diagonal Fisher estimate and updates only high-Fisher parameters:
- Fisher diagonal approximation: \(\hat F_i = \frac{1}{M}\sum_j (\partial \log p(y_j|x_j;\theta)/\partial \theta_i)^2\) (**[FGGM Preliminaries](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/2%20Preliminaries.md)**).
- Masking: choose threshold by the \((1-\alpha)\) quantile with \(\alpha=0.7\) and set \(M_i=1\) iff \(\hat F_i\) exceeds the threshold (**[FGGM mask init](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/3.2%20Fisher-Guided%20Mask%20Initialization.md)**).
- Gradient masking (hard masking): \(\tilde g = g \odot M\) (**[FGGM gradient projection](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/3.4%20Gradient%20Projection.md)**).
- Mask normalization: for fully connected weights, aggregate Fisher values across input dimension per output neuron (**[FGGM dimension aggregation](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/3.3%20Mask%20Normalization%20via%20Output%20Dimension.md)**).

### Key Innovations

1. **Order-2 as a held-out stress test**: use TRACE’s own official “Different Order” sequence as a pre-registered curriculum shift for auditing replay-free CL methods.
2. **Ranking robustness as the primary outcome**: measure whether a reported SOTA ranking (FGGM > MIGU) transfers to a plausible alternative order.
3. **Shared-pipeline evaluation to reduce confounds**: implement both methods within the same training/evaluation pipeline and include a default-order sanity check against published numbers.

---

## Related Work

### Field Overview

LLM continual learning is commonly instantiated as sequential instruction tuning, where a single checkpoint is fine-tuned on a stream of datasets. Methods span (i) **replay-based** approaches that mix historical data (often strong but sometimes infeasible due to privacy/storage), (ii) **regularization/importance-weighting** methods that protect parameters important to prior tasks (e.g., EWC), (iii) **gradient-space constraints** (e.g., OGD), and (iv) **parameter-isolation** methods (e.g., LoRA variants, adapters, and MoE routing).

A long-standing issue in continual learning is **order sensitivity**: performance can vary substantially across task permutations. Prior work proposes order-robust evaluation metrics such as Order-normalized Performance Disparity (OPD/AOPD/MOPD) and schedule-robust learning principles (e.g., SCROLL). However, LLM continual-learning leaderboards (including TRACE-style evaluations in recent replay-free methods) rarely report multi-order robustness.

### Related Papers

- **[TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models_1/meta/meta_info.txt)**: Defines an 8-task continual instruction tuning benchmark and reports OP/BWT plus capability deltas; includes an official alternative order in Appendix D.7.
- **[Unlocking Continual Learning Abilities in Language Models (MIGU)](./references/Unlocking-Continual-Learning-Abilities-in-Language-Models/meta/meta_info.txt)**: Replay-free gradient masking based on activation magnitudes; reports strong continual-learning results and uses multiple task orders on some benchmarks.
- **[FGGM: Fisher-Guided Gradient Masking for Continual Learning](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/meta/meta_info.txt)**: Fisher-based hard gradient masking; reports improvements over MIGU on TRACE default order.
- **[Overcoming Catastrophic Forgetting in Neural Networks (EWC)](https://arxiv.org/abs/1612.00796)**: Classic Fisher-regularization approach; a core baseline family for importance-based CL.
- **[Gradient Episodic Memory (GEM)](https://arxiv.org/abs/1706.08840)**: Replay-based constrained optimization baseline widely used in continual learning.
- **[Learning without Forgetting (LwF)](https://arxiv.org/abs/1606.09282)**: Distillation-based continual learning baseline.
- **[Orthogonal Gradient Descent (OGD)](http://proceedings.mlr.press/v108/farajtabar20a.html)**: Projects gradients to avoid interference; representative gradient-space constraint method.
- **[Riemannian Walk / Understanding Forgetting and Intransigence](https://arxiv.org/abs/1801.10112)**: Introduces OP-style evaluation metrics and analyzes stability–plasticity trade-offs.
- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**: Parameter-efficient fine-tuning primitive used in many LLM continual instruction tuning pipelines.
- **[O-LoRA: Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/abs/2310.14152)**: Applies orthogonality constraints in LoRA space for continual learning.
- **[MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning](https://arxiv.org/abs/2409.04518)**: Optimizer-state filtering for LLM continual fine-tuning.
- **[FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](https://arxiv.org/abs/2601.03938)**: Replay scheduling inspired by forgetting curves.
- **[SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of LLMs](https://arxiv.org/abs/2411.06171)**: Selective attention-based retention mechanism for LLM continual learning.
- **[Progressive Prompts: Continual Learning for Language Models](https://openreview.net/forum?id=U2nWcK8V3g)**: Prompt-based continual learning baseline family.
- **[CoIN: A Benchmark of Continual Instruction tuNing for Multimodal LLM](https://arxiv.org/abs/2402.12851)**: Benchmark highlighting order sensitivity in continual instruction tuning settings.
- **[The Effect of Task Ordering in Continual Learning](https://arxiv.org/abs/2205.13323)**: Studies task ordering effects and proposes curvature-based ordering distances.
- **[Optimal Task Order for Continual Learning of Multiple Tasks](https://arxiv.org/abs/2505.03555)**: Theorizes principles for choosing task order in supervised continual learning.
- **[Scalable and Order-Robust Continual Learning (APD)](https://openreview.net/pdf?id=r1gdj2EKPB)**: Introduces OPD/AOPD/MOPD metrics and an order-robust parameter decomposition method.
- **[Schedule-Robust Online Continual Learning (SCROLL)](https://arxiv.org/abs/2210.05561)**: Extends order-robustness to schedule-robustness and proposes robust online learning designs.
- **[Curriculum-Meta Learning for Order-Robust Continual Relation Extraction](https://openreview.net/attachment?id=r_-Ulm1NSTF&name=pdf)**: Uses curriculum/meta-learning to reduce order sensitivity in continual relation extraction.
- **[Hessian-Aware Low-Rank Perturbation for Order-Robust Continual Learning](https://arxiv.org/abs/2311.15161)**: Curvature-aware approach targeting order robustness.
- **[Order-Robust Class-Incremental Learning: Graph-Driven Dynamic Similarity Grouping](https://openreview.net/pdf?id=2u2qm2QDXO)**: Uses similarity grouping to reduce order sensitivity in class-incremental learning.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| LLM CL benchmarks | Define task streams + CL metrics for instruction-tuned LLMs | TRACE; CoIN | OP/BWT; capability deltas | Often report a single order; expensive multi-order sweeps |
| Replay-free gradient masking | Update only a subset of parameters per batch/task | MIGU; FGGM; OGD | TRACE-style OP/BWT; general benchmarks | Can be path-dependent; hyperparameter sensitivity |
| Fisher/importance-based CL | Use Fisher/importance to protect or select parameters | EWC; FGGM | Standard CL metrics; OP/BWT | Fisher estimates can be noisy; layerwise normalization choices matter |
| Replay-based CL | Mix historical data to stabilize learning | GEM; REP; FOREVER | OP/BWT; forgetting curves | Requires storing data; privacy/storage constraints |
| Order-robust evaluation + methods | Measure/mitigate sensitivity to order/schedule | APD; SCROLL; task-ordering studies | OPD/AOPD/MOPD; schedule robustness | Rarely applied to LLM continual instruction tuning pipelines |

### Closest Prior Work

1. **TRACE Appendix D.7 (Different Order)**: Demonstrates a large order effect for SeqFT but does not evaluate replay-free methods (MIGU/FGGM) across orders.
2. **MIGU**: Presents a replay-free gradient masking method and runs multiple task orders on some benchmarks, but does not report TRACE Order 2 results.
3. **FGGM**: Reports FGGM > MIGU on TRACE default order (Qwen2-1.5B and 7B) but does not test order robustness.
4. **APD / OPD metrics**: Formalizes order sensitivity metrics, but is not connected to TRACE-style LLM continual instruction tuning evaluation.

**Novelty Kill Search Summary:** Searched local proposal corpora for “FGGM Order 2”, “TRACE Different Order”, and “fggm migu order”; and web queries including “FGGM TRACE order 2”, “Order 2 TRACE benchmark FGGM”, and “MIGU TRACE different order”. No prior work evaluating FGGM vs MIGU on TRACE’s official Order 2 was found as of 2026-02-25 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| TRACE (D.7) | Shows SeqFT is order-sensitive | Only SeqFT analyzed; replay-free methods not tested | Apply TRACE’s official Order 2 to replay-free methods | Directly tests whether SOTA claims are curriculum-dependent |
| MIGU | Replay-free magnitude-based gradient masking | TRACE evaluated on default order only | Evaluate MIGU on Order 2 with matched pipeline | Provides a strong replay-free baseline under order shift |
| FGGM | Fisher-guided hard masking; reports FGGM>MIGU | No order-robustness evaluation | Stress-test FGGM on Order 2 | Reveals whether Fisher masking introduces extra path dependence |
| APD / OPD | Defines order-sensitivity metrics | Not applied to LLM CL benchmarks like TRACE | Use Order 2 as a minimal stress test + optionally report OPD-like stats | Bridges order-robust evaluation concepts to TRACE-style LLM CL |

---

## Experiments

### Experimental Setup

**Benchmark:** TRACE continual instruction tuning (8 tasks).

**Primary target:** TRACE **Order 2** from TRACE Appendix D.7.

**Baseline Ladder (REQUIRED):**
- **ORI (no training)**: Evaluate the base model on TRACE tasks without continual training (reported in FGGM Table 1; ORI TRACE-OP=31.19 for Qwen2-1.5B).
- **SFT / SeqFT**: Unconstrained sequential fine-tuning with AdamW (reported in FGGM Table 1; SFT TRACE-OP=49.22 on default order).
- **MIGU**: Replay-free magnitude-based gradient masking (FGGM Table 1; 44.08 on default order).
- **FGGM**: Replay-free Fisher-guided hard masking (FGGM Table 1; 46.00 on default order).

**Sanity check (implementation validity):** Before Order-2 runs, run 1 seed each for **SFT**, **MIGU**, and **FGGM** on the **default order** and require TRACE-OP within ±2.0 of the published FGGM Table 1 numbers (SFT=49.22, MIGU=44.08, FGGM=46.00). If this fails, treat results as implementation-limited and do not interpret Order-2 differences.

**Main runs (3 method conditions):**
1. **SFT on Order 2** (3 seeds)
2. **MIGU on Order 2** (3 seeds)
3. **FGGM on Order 2** (3 seeds)

**Seeds:** `seeds=[42, 123, 456]` (paired across methods; same data order and random seeds for all methods).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2 (Qwen2-1.5B checkpoint) | 1.5B | https://huggingface.co/Qwen | Match FGGM Table 1 setting; verifier should pick the closest available Qwen2-1.5B model |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| TRACE tasks (8 datasets) | Sequential training stream + evaluation | 5k train / 2k test per task | https://github.com/BeyonderXX/TRACE | Check TRACE repo license |

**Resource Estimate**:
- **Compute budget**: Target ≤ 520 GPU-hours.
  - Each TRACE continual run processes 5,000 examples per task with total 40 task-epochs (5/3/7/5/3/5/5/7), i.e., 200k example-passes. Using 8×A100-80GB (as in TRACE), we conservatively budget **24–40 GPU-hours per run** including evaluation.
  - Core plan: **3 methods × 3 seeds + 3 default-order sanity runs (1 seed each)** ≈ 12 runs → **~288–480 GPU-hours**.
- **GPU memory**: 8×A100-80GB (TRACE uses this configuration).
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| TRACE | 8-task continual instruction tuning benchmark for LLMs | **Final-step OP\_T (higher is better)**; **Final-step BWT\_T (less negative is better)**; per-task metrics | Official TRACE test sets | https://github.com/BeyonderXX/TRACE | TRACE repo evaluation + (if needed) OpenCompass configs (as used in FGGM) |

### Main Results

**Published anchors (TRACE default order; Qwen2-1.5B; copied from FGGM Table 1):**

| Method | Base Model | Benchmark | TRACE-OP↑ | General↑ | Source | Notes |
|---|---|---|---:|---:|---|---|
| ORI | Qwen2-1.5B | TRACE (default order) | 31.19 | 52.45 | [FGGM Table 1](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/4.2%20Results%20and%20Analysis.md) | Published (1 run; no variance reported) |
| SFT | Qwen2-1.5B | TRACE (default order) | 49.22 | 50.89 | [FGGM Table 1](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/4.2%20Results%20and%20Analysis.md) | Published (1 run) |
| MIGU | Qwen2-1.5B | TRACE (default order) | 44.08 | 55.21 | [FGGM Table 1](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/4.2%20Results%20and%20Analysis.md) | Published (1 run) |
| FGGM | Qwen2-1.5B | TRACE (default order) | 46.00 | 55.75 | [FGGM Table 1](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/4.2%20Results%20and%20Analysis.md) | Published (1 run) |

**Primary evaluation (TRACE Order 2; Qwen2-1.5B):**

| Method | Base Model | Benchmark | OP_T↑ (mean±std) | BWT_T (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| MIGU | Qwen2-1.5B | TRACE (Order 2) | **TBD** | **TBD** | - | To be verified (paired seeds) |
| FGGM | Qwen2-1.5B | TRACE (Order 2) | **TBD** | **TBD** | - | To be verified (paired seeds) |
| SFT | Qwen2-1.5B | TRACE (Order 2) | **TBD** | **TBD** | - | To be verified (paired seeds) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| FGGM mask overlap analysis (no extra training) | Log per-task binary masks and compute overlap (e.g., Jaccard) across consecutive tasks for each order | Order 2 may induce higher early overlap (NumGLUE-cm/ds) correlated with worse later-task plasticity |

### Experimental Rigor

**Variance & Reproducibility:**
- Main Order-2 comparison uses 3 paired seeds (minimum). If the estimated FGGM–MIGU gap is within the overlapping std range, extend to 5 seeds (still within budget).

**Validity & Controls (top confounders):**
1. **Implementation mismatch** (different preprocessing/eval scripts): mitigate by implementing both methods in the same pipeline and including the default-order sanity check against FGGM Table 1.
2. **Insufficient statistical power**: report mean±std and paired per-seed differences; extend to 5 seeds if ambiguous.
3. **Order-specific data leakage / memorization**: TRACE is a fixed dataset; we follow the official train/test splits. We do not claim to address pretraining contamination; our goal is relative method ranking under matched conditions.

**Sanity checks:**
- Default-order reproduction within ±2 OP for MIGU and FGGM before interpreting Order-2 results.
- Confirm that ORI performance is close to the published ORI anchor (31.19 TRACE-OP) as a data pipeline check.

---

## Success Criteria

**Hypothesis (directional):** FGGM will underperform MIGU on TRACE Order 2 in final-step TRACE-OP, reversing the default-order ranking.

**Decision Rule (concrete):**
- **Proceed (order-dependent ranking supported):** On Order 2, FGGM has lower OP_T than MIGU (mean paired difference < 0) and the sign is consistent in ≥2/3 seeds. Additionally report whether FGGM ranks above or below SFT (to separate “ranking reversal among replay-free methods” from “both methods collapse”).
- **Refute (order-dependent ranking not supported):** On Order 2, FGGM ≥ MIGU with consistent positive sign after up to 5 seeds.
- **Pivot (inconclusive due to implementation):** Default-order sanity check fails for any of SFT/MIGU/FGGM (|OP_T − published| > 2.0), indicating pipeline mismatch.

---

## Impact Statement

If FGGM’s advantage over MIGU fails to transfer to TRACE’s official Order 2, then TRACE-style single-order leaderboards are insufficient to justify “SOTA” claims for replay-free continual learning methods. Practitioners maintaining LLMs under uncertain or externally dictated update streams would need to (i) evaluate methods on multiple plausible orders and (ii) prefer methods whose rankings are stable under order shifts, or report order-robust metrics as part of standard evaluation.

---

## References

- [TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models_1/meta/meta_info.txt) - Wang et al., 2023
- [Unlocking Continual Learning Abilities in Language Models (MIGU)](./references/Unlocking-Continual-Learning-Abilities-in-Language-Models/meta/meta_info.txt) - Du et al., 2024
- [FGGM: Fisher-Guided Gradient Masking for Continual Learning](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/meta/meta_info.txt) - Tan et al., 2026
- [Overcoming Catastrophic Forgetting in Neural Networks (EWC)](https://arxiv.org/abs/1612.00796) - Kirkpatrick et al., 2016
- [Gradient Episodic Memory (GEM)](https://arxiv.org/abs/1706.08840) - Lopez-Paz and Ranzato, 2017
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282) - Li and Hoiem, 2016
- [Orthogonal Gradient Descent for Continual Learning](http://proceedings.mlr.press/v108/farajtabar20a.html) - Farajtabar et al., 2020
- [Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence](https://arxiv.org/abs/1801.10112) - Chaudhry et al., 2018
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [O-LoRA: Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/abs/2310.14152) - (authors), 2023
- [MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning](https://arxiv.org/abs/2409.04518) - (authors), 2024
- [FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](https://arxiv.org/abs/2601.03938) - (authors), 2026
- [SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of LLMs](https://arxiv.org/abs/2411.06171) - (authors), 2024
- [Progressive Prompts: Continual Learning for Language Models](https://openreview.net/forum?id=U2nWcK8V3g) - Razdaibiedina et al., 2023
- [CoIN: A Benchmark of Continual Instruction tuNing for Multimodal LLM](https://arxiv.org/abs/2402.12851) - Chen et al., 2024
- [The Effect of Task Ordering in Continual Learning](https://arxiv.org/abs/2205.13323) - Bell and Lawrence, 2022
- [Optimal Task Order for Continual Learning of Multiple Tasks](https://arxiv.org/abs/2505.03555) - Li and Hiratani, 2025
- [Scalable and Order-Robust Continual Learning (APD)](https://openreview.net/pdf?id=r1gdj2EKPB) - Yoon et al., 2020
- [Schedule-Robust Online Continual Learning (SCROLL)](https://arxiv.org/abs/2210.05561) - Wang et al., 2022
- [Curriculum-Meta Learning for Order-Robust Continual Relation Extraction](https://openreview.net/attachment?id=r_-Ulm1NSTF&name=pdf) - Wu et al., 2021
- [Hessian-Aware Low-Rank Perturbation for Order-Robust Continual Learning](https://arxiv.org/abs/2311.15161) - (authors), 2023
- [Order-Robust Class-Incremental Learning: Graph-Driven Dynamic Similarity Grouping](https://openreview.net/pdf?id=2u2qm2QDXO) - (authors), 2024
