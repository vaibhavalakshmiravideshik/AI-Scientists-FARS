# untitled

# HeadRollback: Post-Task Attention-Head Rollback for Continual LoRA Fine-Tuning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly updated after deployment: teams fine-tune a released checkpoint on a new domain, capability, or product requirement, then repeat this process over time. A persistent obstacle is **catastrophic forgetting**: after learning later tasks, the model’s performance on earlier tasks can degrade substantially.

A widely used practical update mechanism is **LoRA (Low-Rank Adaptation)**, which fine-tunes a small set of low-rank matrices inserted into linear layers while keeping the base model frozen. LoRA reduces GPU memory and storage costs compared to full fine-tuning, and supports frequent model updates. However, recent LLM continual learning benchmarks show that **sequential LoRA fine-tuning can forget severely** (e.g., TRACE reports very negative **backward transfer**).

In this proposal we evaluate continual learning using two standard metrics:
- **OP (Overall Performance; higher is better):** the average final accuracy across tasks after training the last task.
- **BWT (Backward Transfer; higher is better / less negative means less forgetting):** how much performance on earlier tasks changes after learning later tasks.

This proposal targets a minimal, low-overhead intervention: a **post-task edit at task boundaries** that does not require replay data, teacher models, or changing the training loop, and that only stores a few scalars per attention head.

### The Problem

Existing continual learning methods for LLMs largely fall into three categories:

- **Replay-based methods** revisit examples from earlier tasks to prevent forgetting (e.g., replay baselines on TRACE; and replay-scheduling work such as **FOREVER**). Replay is effective but can be infeasible when historical data is unavailable, privacy-sensitive, or expensive to store.
- **Training-time constraint methods** modify optimization to reduce interference (e.g., **MIGU** magnitude-masked updates; gradient-subspace methods; or head-level distillation such as **SEEKR**). These can work well but require integrating nontrivial training logic into the update pipeline.
- **Weight-space merging / accumulation methods** (e.g., **Merge before Forget / SLAO**) attempt to combine task-specific updates, but do not directly address which parts of the model were unintentionally disrupted during learning.

In settings where we already run sequential LoRA fine-tuning (a common baseline), we lack an **after-the-fact mechanism** to undo a small amount of unintentional disruption at the granularity where forgetting appears to concentrate (attention heads), without needing old data.

### Key Insight and Hypothesis

Mechanistic evidence suggests that in sequential fine-tuning, a minority of attention heads can undergo large disruptions, and restoring those disrupted components can recover a substantial fraction of earlier-task performance (used here as motivation; see **Mechanistic Analysis of Catastrophic Forgetting…**).

We hypothesize a closely related and testable claim in a practical LoRA continual learning setting:

> **Hypothesis:** In sequential LoRA fine-tuning, some attention heads that were important for earlier tasks receive large LoRA updates during later-task training despite being weakly supported by the later task’s gradients. Rolling back the LoRA **B-matrix rows** corresponding to these heads at each task boundary will improve retention (higher OP and less-negative BWT) compared to both (i) vanilla sequential LoRA and (ii) a naive rollback of the most-changed heads.

Why this could be wrong: (i) “large change” in LoRA parameters may not correspond to harmful interference; (ii) LoRA’s A matrix could drift enough that B-row rollback does not approximately restore function; (iii) gradients for head importance may be too noisy on small per-task datasets.

---

## Proposed Approach

### Overview

We propose **HeadRollback**, a post-task intervention for continual LoRA fine-tuning:

1. During continual learning, maintain a **historical head-importance accumulator** (one scalar per attention head per layer, updated after each task using that task’s gradients).
2. After completing training on a new task, compute for each head:
   - a **disruption score** measuring how much that head’s LoRA parameters changed during the task, and
   - a **new-task importance** score measuring how much the current task’s loss depends on that head.
3. Select a small fraction (e.g., 10%) of heads with **high disruption**, **high historical importance**, and **low new-task importance**, and **rollback** the LoRA B-matrix rows for those heads to their pre-task values.

The intervention is applied at task boundaries and does not require replay data, teacher distillation, or any additional training epochs.

### Method Details

#### Setting and notation

Consider a transformer with L layers. In each layer ℓ, we apply LoRA to the attention projection matrices **q_proj** and **v_proj**, which are the linear maps that produce the attention **queries** and **values** in multi-head attention (matching FOREVER’s LoRA setting). For a LoRA-augmented linear layer, the effective weight is:

\[ W' = W + \Delta W, \quad \Delta W = \alpha/r \cdot B A \]

where A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}, rank r is small (e.g., r=8), and α is the LoRA scaling.

We treat each (layer ℓ, projection type m ∈ {q,v}, head index h) as a “unit”. For standard multi-head attention, the output dimension d_out is partitioned into contiguous blocks of size head_dim corresponding to heads; for grouped-query attention we use the model’s config (num_attention_heads for q, num_key_value_heads for v) to define the block size.

Let B_{ℓ,m}^{(k)} be the LoRA B matrix after finishing task k (including any rollback applied after task k), and let rows(B, h) denote the subset of rows of B corresponding to head h.

#### Signals computed at each task boundary

At the end of task k:

1) **Disruption (per head):**

\[ \Delta(h) = \sum_{\ell,m} \| \mathrm{rows}(B_{\ell,m}^{(k)},h) - \mathrm{rows}(B_{\ell,m}^{(k-1)},h) \|_F \]

2) **New-task importance (per head):** using a small held-out batch from task k (e.g., 128 examples), compute the gradient of the task loss with respect to the B rows and aggregate:

\[ I_{new}(h) = \sum_{\ell,m} \| \nabla_{\mathrm{rows}(B_{\ell,m}^{(k)},h)} \mathcal{L}_{k} \|_F \]

Define ε = median_h I_new(h) (per task), to avoid instability when I_new(h)≈0.

3) **Historical importance accumulator (per head):** maintain a vector I_old(h), initialized to 0. After finishing each task i, compute that task’s head gradient magnitudes g_i(h) (same definition as I_new, on a small held-out batch). Normalize g_i as a vector across heads to reduce task-dependent loss scaling, then accumulate:

\[ \tilde g_i = g_i / \|g_i\|_2, \quad I_{old} \leftarrow I_{old} + \tilde g_i \]

**Update order:** when finishing task k, compute Δ(h) and I_new(h), score heads using the current I_old(h) (which summarizes tasks <k), apply rollback, then update I_old ← I_old + \tilde g_k so task k becomes part of the history for future tasks.

#### Selection rule

Select a fraction p of heads (default p=0.10) with the largest score:

\[ s(h) = \Delta(h) \cdot \frac{I_{old}(h)}{I_{new}(h) + \varepsilon} \]

Intuition: heads that mattered historically (high I_old), were perturbed during learning (high Δ), but are weakly supported by the new task (low I_new) are candidates for “unintentional disruption”.

#### Rollback operator (B-row rollback)

For selected heads h in each (ℓ,m) LoRA module:

\[ \mathrm{rows}(B_{\ell,m}^{(k)},h) \leftarrow \mathrm{rows}(B_{\ell,m}^{(k-1)},h) \]

We keep A unchanged. This is an exact parameter edit and is easy to implement. It is intended to approximately restore the effective ΔW rows when A is relatively stable across tasks.

#### Diagnostic gate: LoRA-A drift

Because A is shared across heads and can drift, we include a diagnostic:

\[ \rho_A = \frac{\|A^{(k)}-A^{(k-1)}\|_F}{\|A^{(k-1)}\|_F} \]

If max_k ρ_A > 0.10 (10%) for many layers/modules, B-row rollback may not approximate functional rollback.

**Freeze-A pivot variant (only if the diagnostic triggers):** Re-run the same 3-condition experiment but modify training so that LoRA **A is frozen after task 1** (A is trainable on task 1 only; for tasks ≥2, optimize only B). This makes the effective update \(\Delta W\) depend only on B changes, so B-row rollback more closely restores the effective LoRA delta for a head. If freeze-A materially harms adaptation on new tasks, this pivot is treated as a negative result (rollback needs A stability to be useful).

### Key Innovations

- **Post-task head-granular rollback for LoRA CL:** Unlike replay/distillation/masking methods, HeadRollback is a task-boundary parameter edit and can be added to existing sequential LoRA pipelines with minimal disruption.
- **Data-free historical signal:** We store only a **scalar historical importance per head**, computed online when each task is available, avoiding any need to retain old task data.
- **Decisive control that isolates the selection logic:** Comparing against “rollback the most-disrupted heads” tests whether the historical-vs-new importance weighting adds value beyond a naive rollback heuristic.

---

## Related Work

### Field Overview

Continual learning for LLMs is typically evaluated by training a single model over a sequence of tasks and measuring both final multi-task performance and forgetting. Benchmarks such as **TRACE** emphasize that aligned/instruction-tuned LLMs can lose general abilities and instruction-following behavior during continual updates, and that parameter-efficient methods can be particularly brittle.

A central axis in the literature is the stability–plasticity trade-off: methods that preserve earlier tasks often reduce adaptation to new tasks. Replay-based approaches are strong but require storing data (or generating it), while data-free approaches usually impose constraints on optimization (e.g., gradient masking, orthogonality, parameter importance). Separately, recent mechanistic work suggests that forgetting can concentrate in specific substructures (such as attention heads), motivating finer-grained interventions.

### Related Papers

- **[TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models/meta/meta_info.txt)**: Introduces an 8-task continual learning benchmark for aligned LLMs and reports severe forgetting, especially for LoRA sequential fine-tuning.
- **[FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt)**: Proposes model-time replay scheduling and reports strong OP/BWT on Standard CL / Long Sequence / SuperNI.
- **[SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models](./references/SEEKR-Selective-Attention-Guided-Knowledge-Retention-for-Continual-Learning-of-Large-Language-Models/meta/meta_info.txt)**: Distills attention maps on selected heads using replay data, showing head-level retention can be effective.
- **[Attention Retention for Continual Learning with Vision Transformers](./references/Attention-Retention-for-Continual-Learning-with-Vision-Transformers/meta/meta_info.txt)**: Attributes forgetting to attention drift in ViTs and reduces it by masking gradients in previous-task attention regions.
- **[Merge before Forget: A Single LoRA Continual Learning via Continual Merging](./references/Merge-before-Forget-A-Single-LoRA-Continual-Learning-via-Continual-Merging/meta/meta_info.txt)**: Uses orthogonal initialization and asymmetric LoRA merging to keep a single adapter over tasks.
- **[Learning the Mechanism of Catastrophic Forgetting: A Perspective from Gradient Similarity](./references/Learning-the-Mechanism-of-Catastrophic-Forgetting-A-Perspective-from-Gradient-Similarity/meta/meta_info.txt)**: Connects forgetting to negative gradient similarity and motivates unit-level conflict-aware interventions.
- **[Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](./references/Mechanistic-Analysis-of-Catastrophic-Forgetting-in-Large-Language-Models-During-Continual-Fine-tuning/meta/meta_info.txt)**: Reports mechanistic evidence that a minority of attention heads are disproportionately disrupted in continual fine-tuning and motivates restoration-style interventions.
- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**: Introduces the LoRA parameter-efficient fine-tuning method.
- **[EWC: Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796)**: Canonical parameter-importance regularization approach for continual learning.
- **[DER++](https://arxiv.org/abs/2004.07211)**: Replay + distillation method often used as a strong baseline in continual learning.
- **[GEM](https://arxiv.org/abs/1706.08840)**: Gradient episodic memory, a classic constrained optimization approach for continual learning.
- **[Learning without Forgetting (LwF)](https://arxiv.org/abs/1606.09282)**: Distillation-based continual learning without replay.
- **[Progressive Prompts: Continual Learning for Language Models](https://arxiv.org/abs/2301.12314)**: Freezes the base model and adds task prompts to avoid forgetting.
- **[Orthogonal Subspace Learning for Language Model Continual Learning (O-LoRA)](https://arxiv.org/abs/2310.14152)**: Enforces orthogonality between LoRA subspaces across tasks.
- **[Unlocking Continual Learning Abilities in Language Models (MIGU)](https://arxiv.org/abs/2406.17245)**: Masks gradient updates using magnitude signals for rehearsal-free CL.
- **[MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for LLMs](https://arxiv.org/abs/2402.12851)**: Uses multiple LoRA experts with routing and contrastive loss.
- **[Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal (SSR)](https://arxiv.org/abs/2403.01244)**: Generates rehearsal data for replay when old data is unavailable.
- **[Recurrent Knowledge Identification and Fusion (Recurrent-KIF)](https://arxiv.org/abs/2502.17510)**: Dynamically updates parameter importance and fuses knowledge with replay.
- **[COPAL: Continual Pruning in Large Language Generative Models](https://arxiv.org/abs/2405.02347)**: Continual pruning framework for generative LLMs that reduces forgetting via structured sparsification.
- **[MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning](https://arxiv.org/abs/2407.20999)**: Optimizer-level masking method to reduce forgetting in LLM fine-tuning.
- **[Continual AdamW: Efficient Continual Learning for LLMs](https://arxiv.org/abs/2404.02754)**: Reuses optimizer statistics to reduce forgetting with constant memory.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Replay-based CL | Mix old and new data during training; sometimes schedule replay | FOREVER, SSR, DER++ | Standard CL / Long Sequence / SuperNI, TRACE | Requires data buffer or generation; privacy and storage concerns |
| Constraint / masking during training | Reduce interference by restricting updates | MIGU, EWC, Continual AdamW, MoFO | Standard CL, TRACE, domain CL | Requires modifying training loop; may reduce plasticity |
| Head-/module-level retention | Preserve key internal structures (heads, masks) | SEEKR, ARCL-ViT | TRACE, ViT CL benchmarks | Often needs replay/teacher or prior-task attention masks |
| Weight-space merging / adapter accumulation | Merge or accumulate task updates | SLAO / Merge before Forget, AIM-Merging | Standard CL / Long Sequence / SuperNI | Does not directly target which components were unintentionally disrupted |
| Prompt / architecture expansion | Add task-specific modules without modifying old ones | Progressive Prompts | Text classification CL | Memory grows with tasks; not directly applicable to “single checkpoint” updates |

### Closest Prior Work

- **SEEKR** distills attention weights for selected heads using replay data and an old-model teacher. HeadRollback differs by being **post-task**, **teacher-free**, and **replay-free**, and by editing LoRA parameters directly rather than adding a distillation loss.
- **ARCL-ViT** prevents attention drift by masking gradients in attention-derived regions, but it is a **training-time constraint** and is evaluated in vision transformers. HeadRollback is a **task-boundary parameter edit** and targets LoRA adaptation in LLM attention projections.
- **Merge before Forget / SLAO** aims for a single LoRA through continual merging and asymmetric A/B treatment. HeadRollback instead targets **selective rollback** of a minority of head-specific B rows, based on disruption and importance signals, rather than global merging.
- **MIGU** reduces interference by masking gradients during training, whereas HeadRollback leaves training unchanged and applies a small post-task correction.
- **Mechanistic Analysis of Catastrophic Forgetting…** motivates head-localized forgetting and restoration, but is not a practical, reproducible LoRA CL method; HeadRollback turns this intuition into a concrete, low-overhead algorithm and a controlled test.

**Novelty Kill Search Summary:** Searched for combinations of “attention head rollback/restoration + continual learning + LoRA/adapter”, “post-hoc head rollback continual fine-tuning”, and checked recent OpenReview/arXiv results for “head-level rollback” in LLM continual learning. Results surfaced head distillation (SEEKR), attention drift constraints (ARCL-ViT), and LoRA merge/orthogonality methods, but no prior work performing post-task, head-granular rollback of LoRA parameters in continual learning as of 2026-02-20 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| SEEKR | Replay + attention-map distillation on selected heads | Needs replay + teacher checkpoints during training | Post-task edit; no replay/teacher | Cheaper, simpler; applicable when old data unavailable |
| ARCL-ViT | Prevents attention drift via gradient masking | Training-time constraint; vision setting | Post-task rollback on LoRA heads | Works as a drop-in after training without changing optimizer |
| SLAO / Merge before Forget | Merges a single LoRA across tasks | Global merging; no targeted repair | Targeted rollback on disrupted heads | Repairs suspected unintentional disruption rather than averaging it in |
| MIGU | Masks updates using magnitude signal | Requires training-loop modification | Boundary-only rollback | Add-on for existing pipelines; complementary in principle |
| FOREVER | Strong replay scheduling + regularization | Replay memory required | Replay-free parameter edit | Useful when replay is infeasible; minimal extra state |

---

## Experiments

### Experimental Setup

**Primary benchmark choice:** FOREVER’s **Standard CL** benchmark (5 text classification tasks). This choice is motivated by (i) a well-specified OP/BWT protocol, and (ii) published baseline numbers for a Qwen3-0.6B backbone.

**Base model:** `Qwen/Qwen3-0.6B-Base` (HF: https://huggingface.co/Qwen/Qwen3-0.6B-Base). Note: requires `transformers>=4.51.0`.

**LoRA configuration (match FOREVER):** rank r=8, α=32, dropout 0.05, applied to **q_proj** and **v_proj**. Train 10 epochs per task, learning rate 3e-4, batch size 8. Decoding: temperature 0.02, greedy (top-k=1), max new tokens 128 (from FOREVER Appendix G).

**Data protocol (from FOREVER):** sample 1000 training instances per task; reserve 500 instances per class for evaluation. (If the official FOREVER repo is available, use its dataset preprocessing; otherwise implement stratified sampling from public HuggingFace datasets.)

**Task orders:** use 3 fixed permutations of the 5 tasks (to reduce order sensitivity compared to a single order while staying within budget). If the official FOREVER repo is accessible, prefer reusing its predefined `dataset_id` orders; otherwise use:
- Order A: AGNews → AmazonReviewFull → YelpReviewFull → DBpedia14 → YahooAnswersTopics
- Order B: DBpedia14 → YahooAnswersTopics → AGNews → YelpReviewFull → AmazonReviewFull
- Order C: YelpReviewFull → AGNews → DBpedia14 → AmazonReviewFull → YahooAnswersTopics

**Main conditions (3; evaluated for each order × 3 seeds):**
1) **Vanilla LoRA CL**: sequential LoRA fine-tuning (the “Fine-tuning” baseline in FOREVER).
2) **HeadRollback (ours)**: selection score s(h)=Δ(h)·I_old(h)/(I_new(h)+ε), rollback top p=10% heads’ B rows after each task.
3) **HighDisruptionOnly (control)**: rollback top p=10% heads by Δ(h) only (tests whether our selection logic matters).

**Baseline ladder (context + sanity):**
- **Zero-shot prompting** on each task with the base model (no fine-tuning).
- **Self-consistency** prompting: sample N=5 answers (temp=0.7) and majority-vote the label (inference-only baseline).
- **Published CL baselines** from FOREVER Table 1 (Qwen3-0.6B, Standard CL; averaged over 8 orders): Fine-tuning 47.2/-12.6, EWC 51.0/-10.3, O-LoRA 59.4/-7.9, MIGU 69.9/-7.5, VBM 71.5/-5.2, FOREVER 72.9/-4.7 (OP/BWT).

**Note on inference-only baselines:** For the main verification run, fill in zero-shot and self-consistency OP values on the same evaluation splits as fine-tuning. These baselines do not produce BWT by definition (no training sequence), but they verify that fine-tuning provides meaningful lift over prompting.

**Resource Estimate**:
- Using FOREVER’s reported training time for Qwen3-0.6B (Table 6: 1.4 min/epoch on 8×H20), one full 5-task run (50 epochs) is ~70 minutes wall-clock on 8 GPUs ≈ **9.3 GPU-hours**.
- Total planned runs: 3 conditions × 3 seeds × 3 task orders = 27 runs → **~252 GPU-hours** (+~10% overhead for evaluation and rollback bookkeeping) → **~280 GPU-hours**.
- Peak GPU memory: ≤ 20GB for a 0.6B model with LoRA (expected to fit easily on a single A100 80GB).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Standard CL (FOREVER) | 5 text classification tasks (AGNews, Amazon, Yelp, DBpedia, Yahoo) evaluated under sequential training | Accuracy per task; OP and BWT | sampled train (1000/task) + eval (500/class) | HF datasets: ag_news, amazon_review_full, yelp_review_full, dbpedia_14, yahoo_answers_topics | Implement from FOREVER repo if available; otherwise implement straightforward prompt+accuracy evaluation |

Metrics (from FOREVER):
- **OP (Overall Performance; higher is better):** mean final accuracy over tasks after training the last task ("how good is the final model on all tasks?").
- **BWT (Backward Transfer; higher is better / less negative means less forgetting):** mean change between each task’s final accuracy and its accuracy immediately after learning it ("how much did later training hurt earlier tasks?").

### Main Results

#### Results Table

| Method | Base Model | Benchmark | OP (↑) | BWT (↑) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Zero-shot prompting | Qwen3-0.6B | Standard CL | **TBD** | N/A | - | Needs re-run (inference-only, ~1× eval pass) |
| Self-consistency (N=5) | Qwen3-0.6B | Standard CL | **TBD** | N/A | - | Needs re-run (5× inference samples; no BWT by definition) |
| Vanilla LoRA (sequential) | Qwen3-0.6B | Standard CL | 47.2 | -12.6 | [FOREVER Table 1](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md) | Published as “Fine-tuning”; averaged over 8 orders |
| O-LoRA | Qwen3-0.6B | Standard CL | 59.4 | -7.9 | [FOREVER Table 1](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md) | Published |
| MIGU | Qwen3-0.6B | Standard CL | 69.9 | -7.5 | [FOREVER Table 1](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md) | Published |
| VBM | Qwen3-0.6B | Standard CL | 71.5 | -5.2 | [FOREVER Table 1](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md) | Published |
| FOREVER | Qwen3-0.6B | Standard CL | 72.9 | -4.7 | [FOREVER Table 1](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md) | Published |
| HighDisruptionOnly (control) | Qwen3-0.6B | Standard CL | **TBD** | **TBD** | - | To be verified (our compute-matched control) |
| **HeadRollback (ours)** | Qwen3-0.6B | Standard CL | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| HeadRollback (full) | Δ×I_old/(I_new+ε) selection + B-row rollback | Best OP/BWT trade-off |
| Disruption-only rollback | Select by Δ only | Worse than full if importance weighting is necessary |
| (Diagnostic) A-drift logging | Log ρ_A per task/layer | If ρ_A > 0.10 often, interpretability of B-row rollback degrades; pivot to freeze-A variant |

### Experimental Rigor

**Variance & Reproducibility:**
- Run all main conditions across **3 seeds** (`seeds=[42,123,456]`) for each of **3 fixed task orders**.

**Validity & Controls (top confounders):**
- **“BWT improves only because immediate post-task accuracy drops”**: report per-task accuracy **before and after rollback** at each task; require that post-rollback accuracy on the current task does not drop materially (≤1 point) relative to vanilla.
- **“Rollback works only because it reduces effective capacity”**: compare to HighDisruptionOnly; if both help equally, the historical/new importance weighting is not validated.
- **“A drift makes rollback non-functional”**: log ρ_A; if high, treat results as evidence about a different intervention and pivot.

**Sanity checks:**
- Reproduce the published “Fine-tuning” baseline within reasonable tolerance on at least one task order.
- **Importance stability check**: for one representative task boundary, compute I_new(h) on two disjoint held-out batches and report the Spearman correlation across heads; if correlation is low (e.g., <0.5), treat the gradient-based importance signal as too noisy and pivot to a cheaper forward-only proxy (activation-norm based importance).
- Ensure that HighDisruptionOnly does not consistently outperform HeadRollback; if it does, our selection criterion is unnecessary.

---

## Success Criteria

**Hypothesis** (directional): HeadRollback improves retention compared to vanilla sequential LoRA, and the improvement is not explained by simply undoing the largest changes.

**Decision Rule** (concrete):
- **Proceed** if HeadRollback achieves higher OP and higher (less negative) BWT than **both** Vanilla and HighDisruptionOnly by a margin outside the run-to-run std across 3 seeds × 3 orders, while reducing current-task post-rollback accuracy by ≤1 point on average.
- **Pivot** if HeadRollback ≈ HighDisruptionOnly (selection logic not adding value): try a simpler score that drops I_old (purely new-task-based importance) or switch to a freeze-A variant if A drift is large.
- **Refute** if HeadRollback fails to improve OP/BWT over Vanilla or if improvements come with substantial new-task accuracy loss (>1–2 points) or are unstable across orders.

---

## Impact Statement

If successful, HeadRollback would provide a simple, replay-free add-on for sequential LoRA updates that improves retention with minimal engineering: practitioners could keep their existing fine-tuning code and apply a small post-task patch at each update. The results would also clarify whether head-localized LoRA parameter disruptions are a meaningful, repairable mechanism of forgetting in practical continual learning settings.

---

## References

- [FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt) - Feng et al., 2026
- [TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models/meta/meta_info.txt) - Wang et al., 2023
- [SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models](./references/SEEKR-Selective-Attention-Guided-Knowledge-Retention-for-Continual-Learning-of-Large-Language-Models/meta/meta_info.txt) - He et al., 2024
- [Attention Retention for Continual Learning with Vision Transformers](./references/Attention-Retention-for-Continual-Learning-with-Vision-Transformers/meta/meta_info.txt) - Lu et al., 2026
- [Merge before Forget: A Single LoRA Continual Learning via Continual Merging](./references/Merge-before-Forget-A-Single-LoRA-Continual-Learning-via-Continual-Merging/meta/meta_info.txt) - Qiao & Mahdavi, 2025
- [Learning the Mechanism of Catastrophic Forgetting: A Perspective from Gradient Similarity](./references/Learning-the-Mechanism-of-Catastrophic-Forgetting-A-Perspective-from-Gradient-Similarity/meta/meta_info.txt) - Yang et al., 2026
- [Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](./references/Mechanistic-Analysis-of-Catastrophic-Forgetting-in-Large-Language-Models-During-Continual-Fine-tuning/meta/meta_info.txt) - Imanov, 2026
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [EWC: Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796) - Kirkpatrick et al., 2017
- [DER++](https://arxiv.org/abs/2004.07211) - Buzzega et al., 2020
- [GEM](https://arxiv.org/abs/1706.08840) - Lopez-Paz & Ranzato, 2017
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282) - Li & Hoiem, 2017
- [Progressive Prompts: Continual Learning for Language Models](https://arxiv.org/abs/2301.12314) - Razdaibiedina et al., 2023
- [O-LoRA: Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/abs/2310.14152) - Wang et al., 2023
- [MIGU: Unlocking Continual Learning Abilities in Language Models](https://arxiv.org/abs/2406.17245) - Du et al., 2024
- [MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for LLMs](https://arxiv.org/abs/2402.12851) - Luo et al., 2024
- [Self-Synthesized Rehearsal (SSR)](https://arxiv.org/abs/2403.01244) - Huang et al., 2024
- [Recurrent-KIF](https://arxiv.org/abs/2502.17510) - Feng et al., 2025
- [COPAL](https://arxiv.org/abs/2405.02347) - 2024
- [MoFO](https://arxiv.org/abs/2407.20999) - 2024
- [Continual AdamW](https://arxiv.org/abs/2404.02754) - 2024
