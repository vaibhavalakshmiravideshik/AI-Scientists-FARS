# untitled

# GradOrth-Select: Gradient-Orthogonality Layer Selection for Fine-Tuning-Based Model Editing


## Introduction

### Context and Motivation

Large language models are deployed in settings where specific facts or policies must be updated after training (e.g., correcting an outdated association). **Model editing** (knowledge editing) aims to apply targeted updates while minimizing side effects on unrelated capabilities.

Recent work shows that simple, well-implemented fine-tuning can be a strong editing approach. In particular, **LocFT-BF** (Localized Fine-Tuning with a Breadth-First mini-batch pipeline) demonstrates high edit success and strong capability preservation compared to many locate-then-edit methods, and reports evaluation under more realistic **WILD** (Without Intervention, Live Decoding) protocols rather than teacher-forced, length-truncated synthetic evaluation.

However, LocFT-BF also highlights a practical bottleneck: performance depends on **where** the model is updated (layer and module). The best tuning location can vary substantially across models (e.g., early vs late layers), and for new architectures practitioners often rely on heuristics (e.g., “tune an MLP down-projection in a mid-to-late layer”) rather than running an exhaustive location sweep.

### The Problem

For fine-tuning-based editors, **edit reliability is often high across many layers**, so the main role of location choice is controlling a trade-off between:
- **Capability preservation** (how much unrelated performance degrades), and
- **Generalization** (how well the edit transfers to paraphrases/variants).

Exhaustive layer sweeps are feasible for a single model, but they are inconvenient when applying editing to many model families. Locate-then-edit methods (e.g., ROME/MEMIT) can localize “knowledge-critical” layers via causal tracing, but this localization is tailored to rank-one weight editing and can be heavier to implement than a lightweight fine-tuning pipeline.

**Question:** Can we pick a good localized fine-tuning location for model editing using only a few gradient computations, without causal tracing and without multi-run layer sweeps?

### Key Insight and Hypothesis

**Key insight.** For an edit step \(\theta\leftarrow\theta-\eta\nabla L_{edit}\), a first-order approximation gives
\(\Delta L_{retain}\approx -\eta\langle\nabla L_{retain},\nabla L_{edit}\rangle\).
So *where* we fine-tune matters because some locations have **low gradient conflict** between edit and retain objectives, causing less collateral capability drift. We therefore select the location whose edit and retain gradients are most **orthogonal** (small \(|\cos|\)).

**Hypothesis.** Selecting the tuning location by maximizing a **low-conflict score** based on edit–retain gradient geometry
\[
S(\ell)=\lVert g_{edit}(\ell)\rVert_F^2\,\bigl(1-\lvert\cos(g_{edit}(\ell),g_{retain}(\ell))\rvert\bigr),\quad g_{*}(\ell)=\nabla_{W_{\ell}}L_{*}
\]
will improve **capability preservation** relative to a fixed per-model heuristic location, at similar edit success.

**Mechanism intuition.** A first-order approximation gives \(\Delta L_{retain}\approx -\eta\langle g_{retain}(\ell),g_{edit}(\ell)\rangle\). Our score favors layers where the edit gradient is both (i) **non-trivial** (large \(\|g_{edit}\|\)) and (ii) **low-conflict** with retain (small \(|\cos|\)), making the edit step less likely to increase retain loss.

Why this could be wrong: cosine can be noisy for small minibatches; gradients may be near-orthogonal everywhere (uninformative); or the selected layer may coincide with the heuristic (we will report selected layers explicitly).

---

## Proposed Approach

*(We originally considered a gradient-norm ratio score; however, this overlaps closely with recent unlearning selectors such as GRIN and PerTA. We therefore pivot to a cosine-based **gradient conflict** score.)

### Overview

We propose **GradOrth-Select**, a drop-in location selection rule for LocFT-BF-style localized fine-tuning:
1. Define a small candidate set of tunable locations (initially: MLP down-projection matrices across layers).
2. Compute \(S(\ell)\) for each candidate using two small minibatches (edit vs retain).
3. Fine-tune only the selected location \(\ell^* = \arg\max S(\ell)\) with a breadth-first mini-batch editing pipeline.

This differs from recent **gradient-ratio** selectors used in unlearning (e.g., GRIN, PerTA): we use **edit–retain gradient conflict geometry** (cosine) to choose *where to edit*, not *which parameters to mask*.

### Method Details

**Candidate set.** For a decoder-only transformer with \(L\) blocks, candidates are \(\{W^{\mathrm{down}}_{\ell}\}_{\ell=0}^{L-1}\) (MLP down-projection matrices). This keeps candidates comparable and matches LocFT-BF’s stability findings for MLP-down editing.

**Losses.**
- \(L_{\mathrm{edit}}\): target-token NLL for an edit minibatch (prompt tokens masked).
- \(L_{\mathrm{retain}}\): LM NLL on generic text.

**Pre-registered scoring hyperparameters.**
- Edit minibatch: 32 edits.
- Retain minibatch: 256 sequences × 128 tokens (32,768 tokens).

**Scoring.** For each layer \(\ell\), enable gradients only for \(W^{\mathrm{down}}_{\ell}\) and compute
\(S(\ell)=\lVert g_{edit}(\ell)\rVert_F^2\,(1-|\cos(g_{edit}(\ell),g_{retain}(\ell))|)\).
Select \(\ell^*\).

**Editing after selection.** Run localized fine-tuning updating only \(W^{\mathrm{down}}_{\ell^*}\), using a breadth-first epoch-based mini-batch loop (no per-sample depth-first convergence).

### Key Innovations

- A minimal, explicit objective for **editing-location selection** in fine-tuning-based model editing.
- Uses **gradient conflict geometry** (edit vs retain cosine) as a **selection rule** for where to fine-tune.

---

## Related Work

### Field Overview

Editing methods include (i) **locate-then-edit** approaches (ROME/MEMIT and successors), (ii) **fine-tuning-based** approaches (FT-M, LocFT-BF), and (iii) **memory/retrieval-based** editors (e.g., SERAC/WISE). Recent evaluation work (WILD) and robustness probes (negation, prefix-context) show that many editing claims are sensitive to evaluation protocol.

Our proposal is closest to fine-tuning-based editing, but borrows the idea of sensitivity/importance signals (e.g., Fisher) from continual learning and sparse adaptation.

### Related Papers

- **[LocFT-BF](./references/FINE-TUNING-DONE-Right-IN-MODEL-EDITING/meta/meta_info.txt)** (fine-tuning editing + location sweeps).
- **[WILD/Mirage eval](https://arxiv.org/abs/2502.11177)** (realistic evaluation protocol).
- **[Built-on-Sand](./references/Is-Model-Editing-Built-on-Sand-Revealing-Its-Illusory-Success-and-Fragile-Foundation/meta/meta_info.txt)** and **[CoRE/CHED](./references/Context-Robust-Knowledge-Editing-for-Language-Models/meta/meta_info.txt)** (robustness failures).
- **ROME / MEMIT** (locate-then-edit baselines).
- **KDE** (orthogonal projection for lifelong editing).

### Closest Prior Work

- **LocFT-BF**: identifies good locations via an expensive sweep; does not offer a cheap location selector.
- **ROME/MEMIT**: causal tracing selects layers for rank-one edits; not designed for fine-tuning side-effect control.
- **[KDE (ACL 2025)](./references/Knowledge-Decoupling-via-Orthogonal-Projection-for-Lifelong-Editing-of-Large-Language-Models/meta/meta_info.txt)**: uses **orthogonal projection** to reduce interference in *lifelong editing* by projecting updates away from cached past-edit subspaces; differs from us because we do **no projections during training**, but use cosine conflict as a *pre-edit location selector*.
- **GRIN (2508.06467)** / **PerTA (2601.22030)**: gradient-ratio/Fisher weighting for *unlearning* parameter selection; our score uses **cosine conflict** for *editing location* choice.

**Novelty Kill Search Summary:** Searched for “model editing layer selection gradient cosine”, “gradient conflict edit vs retain”, “knowledge editing location selection”, and scanned local drafts. Found GRIN/PerTA as closest gradient-based selection priors; no work found applying **gradient cosine conflict** as the selector for *LocFT-style* editing location (as of 2026-02-21).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LocFT-BF | localized fine-tuning; manual/heuristic location choice | needs sweeps/heuristics | compute \(S(\ell)\) from two minibatches | picks **low-conflict** (edit–retain compatible) layers |
| ROME/MEMIT | causal tracing localization + rank-one edits | not tied to gradient-update side effects | parameter-space gradient signals | directly targets capability preservation |
| EWC/TaLoS/FLoE | Fisher/sensitivity for adaptation | not editing-location selection | use **gradient conflict** as selector | cheap, plug-in selector |

---

## Experiments

### Experimental Setup

**Baseline ladder (for edit success metrics):**
- Prompting: in-context injection of the new fact (no parameter updates).
- Inference-time scaling: best-of-5 decoding for edit reliability/generalization (diagnostic).
- Closest method family: LocFT-BF-style localized fine-tuning.

**Main training conditions (3 total; 3 seeds each):**
1. **LocFT-BF-Heuristic (strongest baseline)**: tune LocFT-BF’s selected location for each model (Table 2):
   - Qwen2.5-7B: layer 6 MLP_down
   - LLaMA3-8B: layer 22 MLP_down
2. **GradMag (ablation)**: tune \(\ell^{\dagger}=\arg\max_{\ell}\|\nabla_{W_{\ell}}L_{edit}\|_F^2\).
3. **GradOrth-Select (ours)**: tune \(\ell^*=\arg\max_{\ell}S(\ell)\).

**Models:**
- `Qwen/Qwen2.5-7B-Instruct` (28 layers)
- `meta-llama/Meta-Llama-3-8B-Instruct` (32 layers)

(LocFT-BF reports different optimal layers across these; testing both is a minimal generalization check.)

**Editing dataset:** KnowEdit / ZsRE (3,000 edits) — a factual QA-style dataset derived from Wikidata relations.

**Evaluation protocol:** WILD-style autoregressive decoding + natural stopping. Use the reference implementation from https://github.com/WanliYoung/Revisit-Editing-Evaluation (or EasyEdit integration).

**Training hyperparameters (fixed across conditions):** AdamW, BF16, effective batch size 64, 1 epoch over edits, max sequence length 256, learning rate 5e-5, weight decay 0.0, grad clip 1.0.

**Resource estimate:** gradient scoring <1 GPU-hour; 9 fine-tuning runs + evaluation ≤50 GPU-hours on 1×A100 80GB.

### Benchmarks and Metrics

| Benchmark | What it tests | Metrics | Split | Download | Eval |
|---|---|---|---|---|---|
| KnowEdit / ZsRE | factual prompt→answer edits (QA format) | Reliability (EM), Generalization (EM) | held-out | https://huggingface.co/datasets/zjunlp/KnowEdit | WILD eval |
| Capability suite (LocFT-BF) | general abilities unrelated to edits | avg accuracy on MMLU, NQ, SST-2, WMT, GSM8K | fixed subset | (standard) | WILD/LM-eval |

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Reliability ↑ | Generalization ↑ | Capability ↑ | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| LocFT-BF (published) | Qwen2.5-7B | ZsRE | 98.87 | 74.37 | 59.07 | `./references/FINE-TUNING-Done-Right-IN-MODEL-EDITING/sections/RESULTS & ANALYSIS.md` | (1 run); tuned layer 6 MLP_down |
| LocFT-BF-Heuristic (ours re-run) | Qwen2.5-7B | ZsRE | TBD | TBD | TBD | this work | layer 6 MLP_down |
| GradMag | Qwen2.5-7B | ZsRE | TBD | TBD | TBD | this work | max edit-gradient magnitude |
| **GradOrth-Select (ours)** | Qwen2.5-7B | ZsRE | TBD | TBD | TBD | this work | selects \(\ell^*\) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| GradMag | max edit-gradient magnitude | worse capability vs GradOrth if conflict matters |

### Experimental Rigor

- **Seeds**: 3 seeds per training condition; report mean±std.
- **Selector stability check (cheap)**: recompute \(\ell^*\) with edit-minibatch sizes {16, 32, 64}; report how often the selected layer changes.
- **Layer-coincidence mitigation**: if \(\ell^*\) matches the LocFT-BF heuristic (±1) for both models, treat the outcome as *heuristic confirmation* (selector adds little value) rather than a win; if it differs, test whether the difference improves capability.
- **Data leakage**: ZsRE facts may appear in pretraining; we focus on *relative* capability differences under identical training and evaluation.

---

## Success Criteria

**Hypothesis:** GradOrth-Select improves capability preservation by selecting layers with low edit–retain gradient conflict (and this cannot be reduced to just choosing the largest edit gradient).

**Decision Rule:**
- **Proceed** if GradOrth-Select improves **Capability by ≥2.0 points** over both LocFT-BF-Heuristic and GradMag (mean over 3 seeds) while Reliability decreases by **≤1.0 point** vs the better baseline.
- **Pivot** if GradOrth-Select ≈ GradMag (cosine adds no value); consider adding a small retain-KL regularizer during editing instead of a selector.
- **Refute** if GradOrth-Select does not improve capability beyond noise (overlapping std) or consistently underperforms LocFT-BF-Heuristic.

---

## Impact Statement

If successful, GradOrth-Select provides a cheap and automatic way to choose where to fine-tune for model editing, reducing reliance on per-model heuristics and improving portability of fine-tuning-based editors across model families.

---

## References

- [FINE-TUNING DONE Right IN MODEL EDITING](./references/FINE-TUNING-DONE-Right-IN-MODEL-EDITING/meta/meta_info.txt) - Yang et al., 2025
- [Is Model Editing Built on Sand?](./references/Is-Model-Editing-Built-on-Sand-Revealing-Its-Illusory-Success-and-Fragile-Foundation/meta/meta_info.txt) - Liu et al., 2024
- [Context-Robust Knowledge Editing for Language Models](./references/Context-Robust-Knowledge-Editing-for-Language-Models/meta/meta_info.txt) - Park et al., 2025
- [The Mirage of Model Editing: Revisiting Evaluation in the Wild](https://arxiv.org/abs/2502.11177) - Yang et al., 2025
- [ROME](https://arxiv.org/abs/2202.05262) - Meng et al., 2022
- [MEMIT](https://arxiv.org/abs/2210.07229) - Meng et al., 2023
- [PMET](https://arxiv.org/abs/2308.08742) - Li et al., 2024
- [RECT](https://arxiv.org/abs/2401.04700) - Gu et al., 2024
- [AlphaEdit](https://arxiv.org/abs/2410.02355) - Fang et al., 2025
- [UltraEdit](https://arxiv.org/abs/2505.07375) - Gu et al., 2025
- [WISE](https://arxiv.org/abs/2405.14768) - Wang et al., 2024
- [SERAC](https://arxiv.org/abs/2206.06520) - Mitchell et al., 2022
- [EasyEdit](https://arxiv.org/abs/2308.07269) - Wang et al., 2023
- [TaLoS](https://openreview.net/forum?id=TDyE2iuvyc) - Iurada et al., 2025
- [FLoE](https://arxiv.org/abs/2506.00495) - 2025
- [SpIEL](https://arxiv.org/abs/2401.16405) - Ansell et al., 2024
- [LoFiT](https://arxiv.org/abs/2406.01563) - Yin et al., 2024
- [EWC](https://arxiv.org/abs/1612.00796) - Kirkpatrick et al., 2017
- [Transformer FFNs are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446/) - Geva et al., 2021
