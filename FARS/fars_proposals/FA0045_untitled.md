# untitled

# Low-NLL Coresets for Repetition-Heavy Long-CoT Supervised Fine-Tuning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated evaluation (no human labeling; no LLM-judge)
  - Open weights + public datasets
  - Must fit within **≤768 A100 GPU-hours**

## Introduction

### Context and Motivation

Supervised fine-tuning (SFT) on long chain-of-thought (long-CoT) demonstrations is a common step for making pretrained language models better at multi-step reasoning. In long-CoT SFT datasets, each example includes a long reasoning trace (often delimited by tags such as `<think> ... </think>`) followed by a short final answer. Downstream benchmarks such as AIME (competition math) and GPQA (graduate-level multiple-choice science) are typically evaluated by automatically extracting the final answer from model outputs.

A recent paper, **Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning** (**[Kopiczko et al., 2026](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**), reports a counterintuitive effect: at a fixed update budget, training for many epochs on a small long-CoT dataset can outperform training for one epoch on a much larger dataset. The paper also finds that repetition increases *termination rate* (generations ending with an end-of-sequence token rather than being truncated), and termination correlates strongly with accuracy.

This raises a deployment-relevant question: if we are going to repeat a small dataset dozens of times, **which examples should we repeat**?

### The Problem

Most long-CoT data selection work aims to identify “high-utility” examples by preferring *difficult* questions and *long* traces (e.g., SELECT2REASON) or by multi-criteria selection for instruction tuning (e.g., D3, RICo). However, these methods are typically motivated in a low-epoch regime (training once or a few epochs on the selected set). In contrast, the repetition-advantage regime repeatedly amplifies the same examples.

A key ambiguity is whether the repeated set should be:

- **Hard-to-fit examples** (high loss / high perplexity), which might teach new reasoning behaviors, or
- **Easy-to-fit examples** (low loss / in-distribution), which might more efficiently teach the *formatting and termination conventions* needed for long-CoT evaluation and for stable generation.

This matters because the wrong choice could waste significant post-training compute: repeating a small set makes its properties dominate the model’s update trajectory.

### Key Insight and Hypothesis

**Key insight.** In repetition-heavy SFT, the training dynamic is closer to “memorize a small corpus extremely well” than to “cover the distribution broadly.” In this regime, repeatedly training on high-loss examples may overfit noisy or idiosyncratic traces, while repeatedly training on low-loss examples may more reliably internalize useful generation conventions (closing the reasoning block, emitting an answer, and terminating) without destabilizing the model.

**Hypothesis.** Under a fixed update budget in the high-repetition regime, repeating a **low per-token NLL** (negative log-likelihood) subset of long-CoT traces yields higher downstream reasoning accuracy than repeating a **high per-token NLL** subset of the same size, after controlling for trace length.

**Why we could be wrong.**
1. High-NLL examples might be the true “learning signal” and repetition could be necessary to learn them.
2. NLL might be dominated by surface confounds (domain, style, or degeneracy) rather than learnability.
3. Both subsets might converge to similar performance once length is controlled, implying NLL is not a useful axis for repeated SFT.
4. The source paper finds that even *incorrect* (negative) trajectories can work well under repetition, suggesting example “quality” may be weakly coupled to downstream gains; NLL may likewise be irrelevant.

---

## Proposed Approach

### Overview

We propose **NLL-Coreset Repetition**, a minimal stress test of subset choice in repetition-heavy long-CoT SFT.

We compare **three training conditions** under the same update budget and training recipe:

1. **Random-Repeated (baseline)**: choose a random subset of size S from a fixed pool and train for E epochs.
2. **Low-NLL-Repeated (ours)**: choose the length-matched lowest-NLL subset of size S and train for E epochs.
3. **High-NLL-Repeated (hard-data control)**: choose the length-matched highest-NLL subset of size S and train for E epochs.

All three conditions use the same base model, optimizer, update budget, and evaluation protocol.

### Method Details

#### A. Data pool and filtering (match the source paper)

We follow Kopiczko et al.’s filtering for **Dolci SFT 7B** long-CoT data (**[Kopiczko et al., 2026](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**):
- Keep only the first conversation turn
- Require complete reasoning traces via `<think>` and `</think>` tags
- Remove examples exceeding 10k tokens under the model tokenizer

To keep selection-cost feasible and fully reproducible, we form a fixed **pool P=6,400** examples by sampling with a fixed seed from the filtered dataset (mirroring the paper’s nested-split procedure).

#### B. Memorizability score: per-token NLL under the base model

For each example x in the pool, compute a *teacher-forced* per-token negative log likelihood on the **response tokens only** (masking the prompt), under the base model before any SFT:

\[
\mathrm{NLL}(x) = \frac{1}{T}\sum_{t=1}^{T} -\log p_\theta(y_t \mid y_{<t}, \mathrm{prompt}(x)).
\]

We interpret low NLL as “easy-to-fit / in-distribution” and high NLL as “hard-to-fit / out-of-distribution.”

#### C. Length-matched selection to remove the main confound

Because NLL can correlate with trace length, we match the response-length distribution between subsets:

1. Bucket pool examples into **10 deciles** by response token length.
2. In each decile, rank examples by NLL.
3. Select a fixed quota from each decile so that **Low-NLL** and **High-NLL** subsets have identical length histograms.

Random-Repeated baselines are also sampled with the same per-decile quotas (so the random baseline is length-matched as well).

#### D. Degeneracy guards (sanity checks)

To avoid selecting pathological tails (e.g., boilerplate repetition) we compute simple automatic statistics for each subset and report them:
- response length distribution
- trigram repetition rate (fraction of tokens that are in repeated 3-grams)
- duplicate-rate under exact response-text match

(We do not change training data based on these metrics unless the subset is clearly degenerate; if a hard filter is required, it will be pre-registered as: “remove exact-duplicate responses beyond the first occurrence.”)

### Key Innovations

- A **high-repetition-specific** question for long-CoT post-training: “repeat easy vs repeat hard?”
- A **single-signal, training-free** selection rule (teacher-forced per-token NLL) with explicit length control.
- A decision-oriented result: if high-NLL repetition is harmful, it argues against difficulty-first selection heuristics when the training recipe relies on many epochs.

---

## Related Work

### Field Overview

This proposal connects three threads.

**(1) Long-CoT post-training dynamics.** Long-CoT SFT and reasoning distillation are widely used to bootstrap reasoning-capable models before or instead of reinforcement learning (e.g., DeepSeek-R1 distillation). The repetition advantage paper shows that, unlike pretraining scaling intuitions, repeatedly training on a small long-CoT dataset can improve generalization even after full memorization.

**(2) Data selection for instruction tuning.** Many methods attempt to select a small high-utility subset (coreset) from a larger pool. Recent work finds that naive “pick the highest perplexity” baselines can fail (e.g., D3), and that the scaling properties of selection methods can be surprising (large-scale selection sometimes underperforms random).

**(3) Memorization and memorizability.** Memorization is not always harmful in post-training: in repetition-heavy SFT, full memorization coincides with improved benchmark performance. This motivates testing whether “easy-to-fit” examples are especially useful to repeat.

### Related Papers

- **[Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**: Identifies the repetition advantage and links gains to memorization and termination behavior; does not study which subsets are best to repeat.
- **[SELECT2REASON](./references/SELECT2REASON-Efficient-Instruction-Tuning-Data-Selection-for-Long-CoT-Reasoning/meta/meta_info.txt)**: Selects long-CoT instruction data using difficulty + trace length ranking in a standard (few-epoch) tuning regime.
- **[D3](./references/D3-Diversity-Difficulty-and-Dependability-Aware-Data-Selection-for-Sample-Efficient-LLM-Instruction-Tuning/meta/meta_info.txt)**: Multi-criteria instruction-tuning data selection; shows highest-perplexity selection can perform poorly.
- **[RICo](./references/RICo-Refined-In-Context-Contribution-for-Automatic-Instruction-Tuning-Data-Selection/meta/meta_info.txt)**: Uses in-context contribution scoring for gradient-free instruction-tuning data selection; finds best data are not necessarily the hardest.
- **[Data Whisperer](./references/Data-Whisperer-Efficient-Data-Selection-for-Task-Specific-LLM-Fine-Tuning-via-Few-Shot-In-Context-Learning/meta/meta_info.txt)**: Training-free data selection via few-shot ICL signals; targets sample efficiency in fine-tuning.
- **[Large-Scale Data Selection for Instruction Tuning](./references/Large-Scale-Data-Selection-for-Instruction-Tuning/meta/meta_info.txt)**: Studies selection at million-sample scale; finds many methods don’t scale and a representation-based baseline is strong.
- **[Compute-Constrained Data Selection](./references/Compute-Constrained-Data-Selection/meta/meta_info.txt)**: Formalizes selection under a joint compute budget; many “strong” methods are not compute-optimal.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)**: Classic pretraining scaling laws motivating fresh-data intuition.
- **[Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)**: Compute-optimal pretraining scaling; contrasts with post-training repetition benefits.
- **[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)**: Studies repeated-data regimes in pretraining; provides context for why repetition can behave differently in SFT.
- **[DoReMi: Optimizing Data Mixtures for Language Model Pretraining](https://arxiv.org/abs/2305.10429)**: Data mixture weighting in pretraining; highlights that loss signals are nontrivial to interpret.
- **[Self-Instruct](https://arxiv.org/abs/2212.10560)**: Generates instruction-tuning data automatically; relevant as many long-CoT datasets are synthetic.
- **[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)**: Shows small curated SFT datasets can be highly effective.
- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)**: Establishes CoT prompting as a way to elicit reasoning.
- **[Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)**: Uses sampling over multiple CoTs, matching the Acc@n / Pass@n evaluation style.
- **[DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)**: Popularizes long-CoT distillation + post-training pipelines.
- **[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)**: Describes reasoning-oriented post-training for Qwen models.
- **[OLMo 3 Technical Report](https://arxiv.org/abs/2512.13961)**: Provides the broader OLMo3 training/post-training context, including Dolci.
- **[Small Batch Size Training for Language Models](https://arxiv.org/abs/2502.04041)**: Empirical support for batch size 1 training as used in the source paper.
- **[Instruction-Following Difficulty for Data Selection](https://arxiv.org/abs/2402.10718)**: Uses loss-based difficulty to filter instruction tuning data; relevant contrast to low-NLL selection.
- **[SuperFiltering: Weak-to-Strong Data Filtering for Instruction Tuning](https://arxiv.org/abs/2402.01068)**: Uses a stronger model to filter for high-quality instructions.
- **[Nuggets: Instruction Tuning Data Selection via Pairwise Comparisons](https://arxiv.org/abs/2402.06437)**: A selection method that uses pairwise judgments; contrasts with our fully training-free scoring.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Long-CoT SFT dynamics | Study compute/data/epoch tradeoffs in reasoning SFT | Data Repetition Beats Data Scaling | AIME, GPQA; Acc@n, Pass@n, termination | Mechanisms unclear; strong dependence on evaluation conventions |
| Difficulty/quality-first selection | Prefer hard/long/high-quality instructions | SELECT2REASON, SuperFiltering, IFD | Reasoning and instruction-following benchmarks | Often assumes few-epoch tuning; may over-select noisy hard examples |
| Multi-criteria coreset selection | Balance diversity, difficulty, reliability | D3, RICo | AlpacaEval, multi-benchmark suites | Added complexity; selection overhead; regime dependence |
| Scaling-aware selection | Test whether selection methods scale and are compute-optimal | Large-Scale Data Selection; Compute-Constrained Data Selection | Multi-task suites; compute–quality curves | Some methods fail to scale; strong simple baselines |
| Memorization/memorizability | Understand what data are easy/hard to fit and how memorization relates to generalization | Scaling data-constrained LMs; memorization surveys | perplexity / memorization measures | Often studied in pretraining; unclear transfer to long-CoT SFT |

### Closest Prior Work

- **Data Repetition Beats Data Scaling**: Establishes that high-epoch repetition can outperform data scaling and that termination/memorization correlate with performance, but does not test which examples should be repeated.
- **SELECT2REASON**: Proposes difficulty+length ranking for long-CoT instruction tuning, but does not test whether difficulty is beneficial under dozens of epochs of repetition.
- **D3 / RICo**: Show that “hardest-by-loss” selection is not reliably best in instruction tuning; our proposal isolates this question specifically in the repetition-heavy long-CoT regime.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Data Repetition Beats Data Scaling | Shows epoch scaling beats data scaling in long-CoT SFT | Subset choice treated as random; no guidance on what to repeat | Compare easy-to-fit vs hard-to-fit subsets under repetition | Produces actionable guidance for repetition-heavy training recipes |
| SELECT2REASON | Selects long-CoT data by difficulty + trace length | Studied in few-epoch regime | Test whether “difficulty-first” is harmful under repetition (high-NLL subset) | Identifies regime-dependent inversion of selection heuristics |
| D3 / RICo | Multi-criteria instruction data selection; high-PPL can fail | Not focused on long-CoT repetition dynamics | Single-signal NLL test with length matching | Simpler, more decisive on the key ambiguity (“repeat easy vs repeat hard?”) |
| Large-Scale Data Selection | Shows scaling failures of complex selectors | Not tailored to long-CoT SFT | Apply a minimal selector (NLL) inside a repetition-heavy pipeline | Tests a simple, scalable selector in a regime where selection matters most |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| OLMo3-7B (base) | 7B | (from authors’ repo / HF checkpoint referenced by OLMo3 TR) | Match the source paper’s base checkpoint and chat template |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Dolci SFT 7B (filtered) | long-CoT SFT pool (P=6,400) | 6,400 examples | https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B | See dataset card |

**Other Resources (if applicable):**
- Source paper code (training + evaluation scripts): https://github.com/dkopi/data-repetition

**Resource Estimate** (evidence-based, extrapolated from the source paper):
- The source paper reports each configuration runs on a single **H100 94GB** for up to **24 hours** at B=51,200 updates.
- We target **B=25,600** updates (half the steps), expecting ≤~12–18 hours per run on a single A100-80GB.
- Planned runs (example): 6 random subset draws + (Low-NLL, High-NLL) × 3 training seeds ≈ 12 runs total.
- **Compute budget**: ~200–400 A100 GPU-hours total (depending on throughput), within the 768 GPU-hour cap.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AIME 2024 | 30 competition math problems; integer answer 0–999 | Acc@16, Pass@16, Termination | test | (via authors’ repo) | https://github.com/dkopi/data-repetition |
| AIME 2025 | 30 competition math problems; integer answer 0–999 | Acc@16, Pass@16, Termination | test | (via authors’ repo) | https://github.com/dkopi/data-repetition |
| GPQA | graduate-level MCQ benchmark | Acc@4, Pass@4, Termination | test | https://huggingface.co/datasets/Idavidrein/gpqa | authors’ eval script |

**Metric definitions (as in the source paper):**
- **Acc@n**: accuracy averaged over n independent generations per problem.
- **Pass@n**: fraction of problems solved in at least one of n attempts.
- **Termination**: fraction of generations that conclude with EOS rather than being truncated.

### Main Results

#### Results Table

(All results below are **TBD** and will be produced by verification under a single unified harness.)

| Method | Base model | Pool size | Subset size S | Epochs E | Update budget B | AIME’24 Acc@16 | AIME’25 Acc@16 | GPQA Acc@4 | Termination | Aggregate Acc | Source | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Random-Repeated | OLMo3-7B | 6,400 | 800 | 32 | 25,600 | TBD | TBD | TBD | TBD | TBD | This work | report mean±std over subset seeds |
| Low-NLL-Repeated | OLMo3-7B | 6,400 | 800 | 32 | 25,600 | TBD | TBD | TBD | TBD | TBD | This work | length-matched; report mean±std over train seeds |
| High-NLL-Repeated | OLMo3-7B | 6,400 | 800 | 32 | 25,600 | TBD | TBD | TBD | TBD | TBD | This work | length-matched; report mean±std over train seeds |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| No length matching | select low/high NLL globally without stratifying by length | if length is the main confound, the effect size changes substantially |
| EOS-only NLL | score NLL only on the final 256 response tokens | if NLL mainly captures formatting/termination difficulty, this approximates full NLL |
| Degeneracy filter | remove exact-duplicate responses before selection | if low-NLL tail is dominated by duplication, filtering reduces its advantage |

### Analysis (Optional)

- **Mechanism slice:** report Acc@n conditioned on termination (Acc@n | terminated) to determine whether any gains are mediated mainly by termination.
- **Subset characterization:** correlate subset-level mean NLL with (i) termination, (ii) trigram repetition, and (iii) domain tags (math/code/chat) if available.

---

## Success Criteria

**Criterion 1 (core hypothesis):**
- Hypothesis: Low-NLL-Repeated outperforms High-NLL-Repeated on aggregate accuracy under the same update budget.
- **Minimum meaningful effect size**: ≥ **+2.0 points** in aggregate Acc (average of AIME’24 Acc@16, AIME’25 Acc@16, GPQA Acc@4).
- Validation: The mean aggregate accuracy is higher for Low-NLL than High-NLL, and a bootstrap 95% CI over evaluation problems excludes 0. (If the CI excludes 0 but the effect is <2.0 points, we treat it as a *weak/possibly-immaterial* effect.)

**Criterion 2 (practitioner relevance):**
- Hypothesis: Low-NLL selection is at least competitive with random subset repetition.
- Validation: Low-NLL-Repeated is ≥ the **median** Random-Repeated run across subset draws on aggregate Acc, and is not worse than the random mean by more than 1 std. (If Low-NLL loses to the median random draw, it is unlikely to be worth the selection overhead.)

---

## Impact Statement

If low-NLL repetition reliably beats high-NLL repetition, it provides a simple, training-free guideline for repetition-heavy long-CoT SFT: **when using many epochs on a small set, prefer easy-to-fit traces rather than difficulty-first selection heuristics**. If the result is negative (high-NLL is better), it supports the opposite guidance and strengthens the case for difficulty-based long-CoT selection even under repetition.

---

## References

Proposal-local artifacts:
- [Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt) - Kopiczko et al., 2026
- [Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning](./references/SELECT2REASON-Efficient-Instruction-Tuning-Data-Selection-for-Long-CoT-Reasoning/meta/meta_info.txt) - Yang et al., 2025
- [Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](./references/Data-Whisperer-Efficient-Data-Selection-for-Task-Specific-LLM-Fine-Tuning-via-Few-Shot-In-Context-Learning/meta/meta_info.txt) - Wang et al., 2025
- [Large-Scale Data Selection for Instruction Tuning](./references/Large-Scale-Data-Selection-for-Instruction-Tuning/meta/meta_info.txt) - Ivison et al., 2025
- [Compute-Constrained Data Selection](./references/Compute-Constrained-Data-Selection/meta/meta_info.txt) - Yin & Rush, 2024
- [RICo: Refined In-Context Contribution for Automatic Instruction-Tuning Data Selection](./references/RICo-Refined-In-Context-Contribution-for-Automatic-Instruction-Tuning-Data-Selection/meta/meta_info.txt) - (authors), 2025
- [D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning](./references/D3-Diversity-Difficulty-and-Dependability-Aware-Data-Selection-for-Sample-Efficient-LLM-Instruction-Tuning/meta/meta_info.txt) - Zhang et al., 2025

Other citations (URLs):
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Kaplan et al., 2020
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) - Muennighoff et al., 2023
- [DoReMi: Optimizing Data Mixtures for Language Model Pretraining](https://arxiv.org/abs/2305.10429) - Xie et al., 2023
- [Self-Instruct](https://arxiv.org/abs/2212.10560) - Wang et al., 2022
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) - Zhou et al., 2023
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Wei et al., 2022
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948) - Guo et al., 2025
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - Qwen Team, 2025
- [OLMo 3 Technical Report](https://arxiv.org/abs/2512.13961) - Team OLMo, 2025
- [Small Batch Size Training for Language Models](https://arxiv.org/abs/2502.04041) - Marek et al., 2025
- [Instruction-Following Difficulty for Data Selection](https://arxiv.org/abs/2402.10718) - Li et al., 2024
- [SuperFiltering: Weak-to-Strong Data Filtering for Instruction Tuning](https://arxiv.org/abs/2402.01068) - Li et al., 2024
- [Nuggets: Instruction Tuning Data Selection via Pairwise Comparisons](https://arxiv.org/abs/2402.06437) - Ivison et al., 2024
