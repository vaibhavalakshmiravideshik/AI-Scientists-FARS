# untitled

# MidPC LoRA: Does an Intermediate SVD Slice Reduce Continual-Learning Forgetting?

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly updated incrementally: new datasets arrive over time, and practitioners fine-tune models repeatedly rather than retraining from scratch. In these *continual learning* (CL) settings, sequential updates often cause **catastrophic forgetting**, where performance on earlier tasks degrades after learning later tasks.

A common practical approach for continual post-training is **parameter-efficient fine-tuning (PEFT)**, which adapts a frozen backbone using a small number of trainable parameters. **Low-Rank Adaptation (LoRA)** is one of the most widely used PEFT methods: for a linear weight matrix \(W\), LoRA adds a low-rank update \(\Delta W = BA\) with rank \(r\ll\min(m,n)\), training only \(A\in\mathbb{R}^{r\times n}\) and \(B\in\mathbb{R}^{m\times r}\) while keeping the backbone fixed.

Recent work suggests that forgetting under LoRA is not only about update magnitude, but also about **spectral structure**. For example, analyses of sequential LoRA updates show that they can introduce misaligned high-energy directions—sometimes called **intruder dimensions** (new top singular directions of the adapted weight that are poorly aligned with the pretrained top directions)—that correlate with pretraining/transfer degradation, and that LoRA updates can be spectrally imbalanced (a few singular components dominate), making them fragile under continual adaptation.

### The Problem

Several SVD-initialized LoRA variants make *conflicting recommendations* about which spectral directions are safest to train:

- **PiSSA** initializes LoRA using the **largest singular values/vectors** of \(W\), motivated by faster convergence and “updating the essence” of the matrix.
- **MiLoRA** initializes LoRA using the **smallest singular values/vectors** of \(W\), motivated by preserving pretrained knowledge by adapting in a less-optimized subspace.
- **Least but not Last** finds that, under sufficiently long single-task fine-tuning, **forgetting is U-shaped across component index**: both extremes forget more than **intermediate** singular components.

However, “Least but not Last” evaluates *single-task* forgetting (retaining pretraining performance while adapting to one downstream task), not **sequential multi-task continual learning** metrics such as **overall performance (OP)** and **backward transfer (BWT)**.

This leaves a decision-relevant open question for practitioners who do sequential LoRA updates:

> If we keep the LoRA parameter budget fixed, does choosing an **intermediate SVD slice** as the trainable subspace improve retention across tasks compared to endpoint choices (PiSSA and MiLoRA)?

If the answer is “yes”, this would be a nearly zero-cost knob (an initialization/subspace choice) that could be combined with existing CL methods (replay, orthogonality constraints). If the answer is “no”, it clarifies that the intermediate-component phenomenon is a single-task artifact and does not transfer to genuine continual learning.

### Key Insight and Hypothesis

**Key insight.** Endpoint SVD slices may increase continual-learning interference via different mechanisms:

- **Top slice (PiSSA)**: updates overlap heavily with high-energy directions shared across tasks, which may cause larger rotations/damage to dominant pretrained components during sequential training.
- **Bottom slice (MiLoRA)**: updates operate in poorly conditioned long-tail directions; in continual training these directions may be amplified and drift into misaligned high-energy “intruder” directions.

**Hypothesis.** Training LoRA in an **intermediate SVD slice** (“MidPC”) will yield a better stability–plasticity trade-off in continual learning than either endpoint:

- Higher **BWT** (less forgetting of earlier tasks) than both PiSSA and MiLoRA,
- While keeping **OP** (average final performance) competitive.

**Why we could be wrong.** Any gains could disappear under sequential multi-task training, or the advantage could be explainable as an “effective conditioning / effective learning rate” effect (singular values differ across slices). We address this with a small LR-matching sanity check and spectral diagnostics.

---

## Proposed Approach

### Overview

We propose **MidPC LoRA**, a generalized SVD-slice LoRA initialization applied to continual learning. The method selects a contiguous rank-\(r\) slice of singular components \(s{:}s{+}r\) from the pretrained weight matrix \(W\), and constrains training to that subspace while freezing the complement.

This unifies:

- **PiSSA** as the special case \(s=0\) (largest components),
- **MiLoRA** as the special case \(s=m-r\) (smallest components),
- **MidPC** as the special case \(s=\lfloor (m-r)/2 \rfloor\) (intermediate components).

### Method Details

#### Function-preserving SVD-slice parameterization with LoRA scaling

Let \(W\in\mathbb{R}^{m\times n}\) be a pretrained weight matrix (assume \(m\le n\) for notation) with SVD:
\[W = U\Sigma V^\top.\]

Choose a start index \(s\) and rank \(r\). Define the slice:
\[\Delta W_{s} = U_{s:s+r}\,\Sigma_{s:s+r}\,V^\top_{s:s+r}.\]

We set the **frozen base** to the residual:
\[W_p = W - \Delta W_s.\]

We then initialize LoRA factors \(A\in\mathbb{R}^{r\times n}\), \(B\in\mathbb{R}^{m\times r}\) so that the *effective weight at initialization* equals the original \(W\), even with the usual LoRA scaling \(\alpha/r\). In plain terms: **before any gradient steps, the model’s behavior is unchanged** (any differences we observe must come from training dynamics, not a different starting model):

\[W_{\text{eff}}(0) = W_p + (\alpha/r)\,BA = W.\]

A simple construction (used by PiSSA-like methods) is:
\[
A = \sqrt{r/\alpha}\;\Sigma^{1/2}_{s:s+r}V^\top_{s:s+r},\qquad
B = \sqrt{r/\alpha}\;U_{s:s+r}\Sigma^{1/2}_{s:s+r}.
\]
Then \(BA = (r/\alpha)\Delta W_s\) and \(W_p + (\alpha/r)BA = W\).

During training, \(W_p\) remains frozen and only \(A,B\) are updated.

#### Continual-learning setting

We evaluate MidPC in a standard sequential-task CL setup for LLMs following **FOREVER**: the model is trained sequentially on **5 text classification datasets** (AG News → Amazon Reviews → Yelp Reviews → DBpedia → Yahoo Answers, under FOREVER’s predefined task orders).

- Tasks arrive sequentially \(T_1,\dots,T_K\).
- The model is fine-tuned on each task’s training data in order.
- After finishing task \(T_k\), we evaluate on all tasks \(T_1..T_k\).

We focus on the **no-replay “Fine-tuning” baseline setting** (memory buffer disabled) to isolate the effect of the SVD-slice choice.

#### Diagnostics (no extra training runs)

To test the mechanism, we compute diagnostics on the same checkpoints:

1. **Intruder accumulation (proxy).** For each targeted weight matrix, compute a truncated SVD of the *effective weight* \(W_{\text{eff}}\) after each task and count how many of its top-\(k\) singular vectors are poorly aligned with the *pretraining* top-\(k\) singular vectors (e.g., max cosine similarity < 0.2). Report the change from after Task 1 to after Task \(K\). (All methods are function-preserving at step 0, so this is not an initialization artifact.)
2. **Spectral imbalance of \(\Delta W\).** Compute the fraction of \(\|\Delta W\|_F^2\) captured by the top-1 singular value of \(\Delta W\) (or a Gini coefficient over its singular values). Hypothesis: MidPC has less extreme imbalance growth across tasks.
3. **Early-step update magnitude.** Log the norm of optimizer updates on LoRA parameters during the first \(N\) steps of Task 1 to support a learning-rate confound check.

### Key Innovations

- A **continual-learning evaluation** of the “intermediate principal components are safest” hypothesis, moving beyond single-task forgetting.
- A **function-preserving** SVD-slice parameterization that keeps the effective pretrained model unchanged at initialization even under LoRA scaling.
- A **mechanism-driven** test with diagnostics (intruder accumulation, spectral imbalance) and a small LR-matching sanity check to distinguish spectral-interference effects from conditioning effects.

---

## Related Work

### Field Overview

LoRA-based continual learning for LLMs can be roughly grouped into: (i) **replay-based** approaches that revisit a memory buffer (e.g., MixReplay, VBM, FOREVER), (ii) **subspace/constraint-based** approaches that restrict updates to reduce interference (e.g., O-LoRA, InfLoRA, KeepLoRA, EBLoRA), and (iii) **adapter-management** approaches that allocate or merge adapters across tasks (e.g., MoELoRA, Merge-before-Forget, Share, TreeLoRA). In parallel, **SVD-initialized PEFT** methods (PiSSA, MiLoRA) and recent analysis (Least but not Last) study how spectral choices affect the performance–forgetting trade-off.

Our proposal connects these threads by treating *which SVD slice is trainable* as a simple stability knob and testing it under standard CL metrics.

### Related Papers

- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**: Introduces LoRA as a parameter-efficient fine-tuning method.
- **[PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models](./references/PiSSA-Principal-Singular-Values-and-Singular-Vectors-Adaptation-of-Large-Language-Models/meta/meta_info.txt)**: Initializes LoRA from top singular components of \(W\) for faster convergence.
- **[MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning](./references/MiLoRA-Harnessing-Minor-Singular-Components-for-Parameter-Efficient-LLM-Finetuning/meta/meta_info.txt)**: Initializes LoRA from bottom singular components to reduce interference with pretrained knowledge.
- **[Least but not Last: Fine-tuning Intermediate Principal Components for Better Performance-Forgetting Trade-Offs](./references/Least-but-not-Last-Fine-tuning-Intermediate-Principal-Components-for-Better-Performance-Forgetting-Trade-Offs/meta/meta_info.txt)**: Shows a U-shaped forgetting curve across component index under long single-task fine-tuning.
- **[LoRA vs Full Fine-tuning: An Illusion of Equivalence](./references/LoRA-vs-Full-Fine-tuning-An-Illusion-of-Equivalence/meta/meta_info.txt)**: Analyzes “intruder” singular directions and their role in sequential LoRA forgetting.
- **[Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation](./references/Spectral-Imbalance-Causes-Forgetting-in-Low-Rank-Continual-Adaptation/meta/meta_info.txt)**: Proposes EBLoRA, linking spectral imbalance of LoRA updates to forgetting.
- **[FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt)**: Provides public OP/BWT benchmarks and strong replay baselines for LLM continual learning.
- **[O-LoRA: Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/abs/2310.14152)**: Trains task-specific LoRA adapters in orthogonal subspaces to reduce interference in continual learning.
- **[InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](https://arxiv.org/abs/2404.00228)**: Constrains LoRA updates to reduce task interference.
- **[Shared LoRA Subspaces for almost Strict Continual Learning](https://arxiv.org/abs/2602.06043)**: Learns shared adapter subspaces across tasks for stricter retention.
- **[Merge before Forget: A Single LoRA Continual Learning via Continual Merging](https://arxiv.org/abs/2512.23017)**: Maintains a single LoRA by continual merging to avoid adapter growth.
- **[TreeLoRA](https://openreview.net/forum?id=f6ibJCQfH4)**: Uses a hierarchical gradient-similarity tree to allocate LoRA updates efficiently across tasks.
- **[Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation](https://arxiv.org/abs/2411.05663)**: Detects distribution shifts online and merges/reinitializes LoRA at loss plateaus.
- **[MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models](https://arxiv.org/abs/2402.12851)**: Treats multiple LoRA modules as MoE experts with contrastive learning for specialization; used as a continual-learning baseline in FOREVER.
- **[EWC](https://arxiv.org/abs/1612.00796)**: Classical continual-learning regularization baseline.
- **[Progressive Prompts: Continual Learning for Language Models](https://arxiv.org/abs/2301.12314)**: Prompt-based continual learning method and benchmark source for longer task sequences.
- **[Super-NaturalInstructions](https://arxiv.org/abs/2204.07705)**: Instruction-following benchmark used in continual instruction-tuning settings.
- **[VBM: Do Your Best and Get Enough Rest for Continual Learning](https://arxiv.org/abs/2503.18371)**: Ebbinghaus-inspired replay scheduling baseline (View-Batch Model) used by FOREVER.
- **[AIMMerging: Adaptive Iterative Model Merging Using Training Trajectories for Language Model Continual Learning](https://arxiv.org/abs/2509.17348)**: Training-trajectory-guided model merging baseline used by FOREVER.
- **[SSR: Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal](https://arxiv.org/abs/2403.01244)**: Generates synthetic rehearsal data for replay-based continual learning.
- **[MIGU: Unlocking Continual Learning Abilities in Language Models](https://arxiv.org/abs/2406.17245)**: Uses magnitude-based gradient masking to reduce interference in continual learning.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| SVD-initialized PEFT | Choose trainable subspace via SVD of \(W\) | PiSSA, MiLoRA, Least but not Last | Single-task forgetting; PEFT benchmarks | Limited CL evaluation; potential conditioning confounds |
| Subspace / constraint CL | Enforce orthogonality/balance/projections during CL | O-LoRA, InfLoRA, KeepLoRA, EBLoRA | OP/BWT, MFN on CL suites | More complex training; extra constraints/state |
| Replay-based CL | Replay memory samples to reduce forgetting | MixReplay, VBM, FOREVER | OP/BWT on CL suites | Needs memory buffer; schedule sensitivity |
| Adapter management | Allocate/merge adapters across tasks | MoELoRA, Share, Merge-before-Forget, TreeLoRA | CL suites | Extra routing/merging complexity |

### Closest Prior Work

- **Least but not Last**: Demonstrates intermediate-component benefits for *single-task* performance–forgetting trade-offs under long fine-tuning, but does not evaluate sequential CL metrics (OP/BWT) over multiple tasks.
- **PiSSA / MiLoRA**: Provide endpoint SVD-slice initializations and single-task PEFT results, but do not resolve which slice is best for sequential multi-task CL.
- **LoRA vs Full Fine-tuning**: Links sequential LoRA forgetting to spectral misalignment (“intruders”) and proposes post-hoc interventions; it does not test whether changing the trainable subspace at initialization can prevent intruder accumulation.
- **EBLoRA**: Balances spectra via constrained optimization for CL, but changes the training objective and focuses on VLM CL; our approach is a training-light, initialization-only knob.
- **FOREVER**: Provides a strong public CL protocol and baselines; our method is orthogonal and can be combined with replay scheduling.

**Novelty Kill Search Summary:** We searched for prior work combining “intermediate principal components / SVD slice LoRA” with “continual learning OP/BWT” (including OpenReview queries and local repo search) and did not find papers explicitly evaluating intermediate-slice LoRA initialization on standard CL OP/BWT as of 2026-02-19. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation for this question | What we change | Why ours should win |
|---|---|---|---|---|
| PiSSA | Train top SVD slice of \(W\) | Not evaluated on sequential CL OP/BWT | Compare against MidPC | MidPC avoids damaging dominant shared directions |
| MiLoRA | Train bottom SVD slice of \(W\) | Not evaluated on sequential CL OP/BWT | Compare against MidPC | MidPC avoids tail instability/intruders |
| Least but not Last | Shows intermediate slices reduce single-task forgetting | Not multi-task sequential CL | Test on OP/BWT CL benchmark | If U-shape is a true stability knob, it should transfer |
| LoRA vs Full FT | Identifies intruder dimensions in sequential LoRA | Not an initialization method | Add slice-controlled init + intruder diagnostics | MidPC should reduce intruder accumulation |
| EBLoRA / KeepLoRA | Training-time constrained CL | More complex than init-only | Provide simpler alternative knob | If it works, can be combined with them |

---

## Experiments

### Experimental Setup

**Primary verification target:** The **FOREVER Standard CL** benchmark (5 text classification tasks) with the **Qwen3-0.6B** backbone, evaluated with **OP** and **BWT**.

We follow FOREVER’s protocol and hyperparameters where possible, but disable replay/memory to isolate the effect of LoRA subspace choice.

**Baseline Ladder (REQUIRED):**
- **Frozen / no-CL baseline (prompt-only reference)**: Evaluate the pretrained Qwen3-0.6B on each task with the same prompting template but without training (reports OP; BWT is not meaningful without training).
- **Published CL baselines (context, from FOREVER Table 1)**: Fine-tuning (no memory), EWC, O-LoRA, MoELoRA, MixReplay, VBM, FOREVER.
- **Closest-method baselines (to run; 3 conditions)**: PiSSA (top slice), MiLoRA (bottom slice), MidPC (middle slice).

**Core decisive experiment (3 main conditions):**
1) **PiSSA init**: \(s=0\)
2) **MiLoRA init**: \(s=m-r\)
3) **MidPC init (ours)**: \(s=\lfloor (m-r)/2 \rfloor\)

All use the same LoRA rank \(r=8\), scaling \(\alpha=32\), dropout 0.05, and are applied to **q_proj** and **v_proj** as in FOREVER.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen/Qwen3-0.6B | 0.6B | https://huggingface.co/Qwen/Qwen3-0.6B | Use the same checkpoint family as FOREVER; verify exact variant in their repo/config |

**Training Data (FOREVER Standard CL):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| AG News | Task 1 (classification) | subsampled in protocol | https://huggingface.co/datasets/ag_news | public |
| Amazon Reviews | Task 2 | subsampled in protocol | https://huggingface.co/datasets/amazon_review_full (or amazon_polarity; follow FOREVER code) | public |
| Yelp Reviews | Task 3 | subsampled in protocol | https://huggingface.co/datasets/yelp_review_full (or yelp_polarity; follow FOREVER code) | public |
| DBpedia | Task 4 | subsampled in protocol | https://huggingface.co/datasets/dbpedia_14 | public |
| Yahoo Answers | Task 5 | subsampled in protocol | https://huggingface.co/datasets/yahoo_answers_topics | public |

**Evaluation Scripts:**
- Use the official FOREVER implementation (paper points to an anonymous open science repo): https://anonymous.4open.science/r/FOREVER-C7D2
  - Task orders are defined in `utils/dataset_order.py` (via the `dataset_id` parameter).

**Resource Estimate** (budget cap: 768 GPU-hours):
- Evidence: FOREVER reports **training time ~1.4 min/epoch** for Qwen3-0.6B on 8×H20 GPUs (Table 6 in `B.4 Time Complexity Analysis`).
- Each Standard CL run trains **5 tasks × 10 epochs/task = 50 epochs** (no replay) → ~70 minutes on 8 GPUs.
- Conservative estimate: **10 GPU-hours per run** (includes overhead + SVD init).

Planned runs:
- **Phase 1 (core):** 2 task orders × 3 seeds × 3 methods = 18 runs → ~180 GPU-hours.
- **Phase 1b (LR-matching sanity, conditional):** if MidPC wins, 1 task order × 1 seed × 2 reruns (PiSSA+MiLoRA with LR adjusted to match MidPC early-step update norms) → ~20 GPU-hours.
- **Phase 2 (optional robustness):** extend Phase 1 to all 8 task orders used by FOREVER (8 orders × 3 seeds × 3 methods = 72 runs) → ~720 GPU-hours total for Phase 2 alone. This phase should only be attempted if Phase 1 is strongly positive and remaining budget permits.

Total planned (Phase 1 + 1b): ~200 GPU-hours (well under cap).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| FOREVER Standard CL | 5 sequential text classification tasks | OP, BWT | test | See dataset links above | https://anonymous.4open.science/r/FOREVER-C7D2 |

**Metric definitions (from FOREVER):** Let \(a_{i,j}\) be performance on task \(T_i\) after training up through task \(T_j\), and \(K\) be #tasks.
- \(\text{OP} = \frac{1}{K}\sum_{i=1}^K a_{i,K}\) (higher is better): **final average accuracy across all tasks**.
- \(\text{BWT} = \frac{1}{K-1}\sum_{i=1}^{K-1}(a_{i,K} - a_{i,i})\) (higher / less negative is better): **how much earlier-task accuracy changes after learning later tasks** (a forgetting measure).

### Main Results

**Published baselines (FOREVER Table 1, Qwen3-0.6B; Standard CL):**

| Method | Base Model | Benchmark | OP (↑) | BWT (↑) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Fine-tuning | Qwen3-0.6B | Standard CL | 47.2 | -12.6 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | memory-free |
| EWC | Qwen3-0.6B | Standard CL | 51.0 | -10.3 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | memory-free regularization |
| O-LoRA | Qwen3-0.6B | Standard CL | 59.4 | -7.9 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | memory-free subspace constraint |
| MoELoRA | Qwen3-0.6B | Standard CL | 55.3 | -8.2 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | memory-free adapter management |
| MixReplay | Qwen3-0.6B | Standard CL | 65.8 | -8.0 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | replay |
| Fixed-interval Replay | Qwen3-0.6B | Standard CL | 65.1 | -9.2 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | replay |
| SAPT | Qwen3-0.6B | Standard CL | 68.8 | -6.9 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | replay-based baseline in FOREVER |
| MIGU | Qwen3-0.6B | Standard CL | 69.9 | -7.5 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | baseline in FOREVER |
| SSR | Qwen3-0.6B | Standard CL | 68.4 | -7.1 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | synthetic rehearsal |
| Recurrent-KIF | Qwen3-0.6B | Standard CL | 70.6 | -6.5 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | replay + KIF |
| AIMMerging | Qwen3-0.6B | Standard CL | 71.9 | -5.0 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | trajectory-based merging |
| VBM | Qwen3-0.6B | Standard CL | 71.5 | -5.2 | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | replay scheduling |
| **FOREVER** | Qwen3-0.6B | Standard CL | **72.9** | **-4.7** | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | SOTA in this protocol |
| MTL (upper bound) | Qwen3-0.6B | Standard CL | 77.4 | — | [FOREVER Table 1](<./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/sections/Main Results.md>) | multi-task training |

*Note:* **EBLoRA** (Gu et al., 2026) is a closely related *training-time* spectral-balancing method, but it is evaluated on **vision-language** CL suites rather than this text-only Standard CL protocol; we treat it as related work rather than a directly comparable baseline here.

**To be verified (this proposal):**

| Method | Base Model | Benchmark | OP (mean±std) | BWT (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| PiSSA init (s=0) | Qwen3-0.6B | Standard CL | **TBD** | **TBD** | - | Run (Phase 1) |
| MiLoRA init (s=m-r) | Qwen3-0.6B | Standard CL | **TBD** | **TBD** | - | Run (Phase 1) |
| **MidPC init (s=⌊(m-r)/2⌋)** | Qwen3-0.6B | Standard CL | **TBD** | **TBD** | - | Run (Phase 1) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| MidPC + intruder diagnostics | Only add measurement | MidPC shows lower Δ-intruder-count (Task1→TaskK) if mechanism holds |
| Phase 1b LR-matching (conditional) | Re-run PiSSA/MiLoRA with LR adjusted to match MidPC early-step update norms | If MidPC advantage mostly conditioning, BWT gap shrinks substantially |

### Experimental Rigor

- **Seeds**: `seeds=[42, 1121, 3407]` for all Phase-1 runs.
- **Task orders**: Use 2 fixed `dataset_id` values from the FOREVER repo (e.g., 1 and 7). Report per-order and averaged results.
- **Comparability**: Same base model, same LoRA rank/alpha/dropout, same optimizer/hyperparameters, same evaluation script.
- **Confound checks**:
  - Conditioning confound: Phase 1b LR-matching sanity (stop rule below).
  - Initialization correctness: verify that for each method \(\|W_{\text{eff}}(0)-W\|_F\approx 0\) for targeted matrices.

### Analysis (Optional)

- **Mechanism correlation (within-method)**: across seeds/orders, correlate BWT with Δ-intruder-count for each method to see if lower intruder growth predicts better retention.

---

## Success Criteria

**Hypothesis** (directional): MidPC initialization improves BWT relative to both PiSSA and MiLoRA at similar OP on continual learning.

**Decision Rule** (concrete):

- **Proceed** if, averaged over 3 seeds and 2 task orders:
  - MidPC improves **BWT by ≥ +1.0** over **both** PiSSA and MiLoRA, and
  - MidPC’s **OP is within 0.5 points** of the best of {PiSSA, MiLoRA}.

- **Pivot (conditioning check)** if MidPC meets the above but Phase 1b LR-matching reduces MidPC’s BWT advantage by **>50%** (suggesting the effect is mostly step-size/conditioning).

- **Refute** if MidPC’s BWT is within **0.5** of the better endpoint baseline (or worse), or if OP drops by **>0.5** relative to the better endpoint.

---

## Impact Statement

If MidPC improves continual-learning retention with no additional training-time machinery, it provides a simple, low-cost knob for practitioners doing sequential LoRA updates, and could be combined with stronger CL methods (replay scheduling, orthogonality constraints) as a drop-in initialization/subspace choice. If it fails, the negative result is still decision-relevant: it suggests that intermediate-component benefits observed in single-task forgetting do not transfer to sequential multi-task continual learning, and practitioners should focus on training-time CL mechanisms instead.

---

## References

- [FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](./references/FOREVER-Forgetting-Curve-Inspired-Memory-Replay-for-Language-Model-Continual-Learning/meta/meta_info.txt) - Feng et al., 2026
- [Least but not Last: Fine-tuning Intermediate Principal Components for Better Performance-Forgetting Trade-Offs](./references/Least-but-not-Last-Fine-tuning-Intermediate-Principal-Components-for-Better-Performance-Forgetting-Trade-Offs/meta/meta_info.txt) - Quercia et al., 2026
- [PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models](./references/PiSSA-Principal-Singular-Values-and-Singular-Vectors-Adaptation-of-Large-Language-Models/meta/meta_info.txt) - Meng et al., 2024
- [MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning](./references/MiLoRA-Harnessing-Minor-Singular-Components-for-Parameter-Efficient-LLM-Finetuning/meta/meta_info.txt) - Wang et al., 2024
- [LoRA vs Full Fine-tuning: An Illusion of Equivalence](./references/LoRA-vs-Full-Fine-tuning-An-Illusion-of-Equivalence/meta/meta_info.txt) - Shuttleworth et al., 2025
- [Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation](./references/Spectral-Imbalance-Causes-Forgetting-in-Low-Rank-Continual-Adaptation/meta/meta_info.txt) - Gu et al., 2026
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) - Kirkpatrick et al., 2017
