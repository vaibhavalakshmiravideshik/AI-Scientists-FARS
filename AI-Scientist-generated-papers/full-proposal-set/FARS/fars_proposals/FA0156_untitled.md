# untitled

# Length-Weighted SFT as a Mechanism Test for the "Repetition Advantage" in Long-CoT Fine-Tuning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Supervised fine-tuning (SFT) on long chain-of-thought (Long-CoT) demonstrations is a widely used step for improving pretrained language models on difficult reasoning tasks such as competition math and scientific question answering. High-quality long-CoT demonstrations are expensive to obtain, so practitioners often face a practical choice: (i) collect more unique demonstrations ("data scaling"), or (ii) train longer on a smaller set ("data repetition").

**[Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)** (Kopiczko et al., 2026) reports a striking phenomenon: under a fixed optimizer-step budget (epochs x samples = 51,200, batch size 1), repeating a small long-CoT dataset for many epochs can dramatically outperform training on many more unique samples for one epoch. For OLMo3-7B at update budget 51,200, their Table 4 reports average downstream **Acc@k** improving from **17.2** (1 epoch on 51.2k samples) to **38.8** (32 epochs on 1.6k samples), with termination improving in parallel. Here Acc@k is the mean correctness across k sampled generations per problem (higher is better).

This result is highly practice-relevant: if true, it implies that for long-CoT SFT the marginal value of collecting additional unique data may be much smaller than the marginal value of repeating a small high-quality set until the model fully memorizes it.

### The Problem

Despite the strong empirical effect, the mechanism behind the "repetition advantage" is underspecified. The source paper highlights that termination rate (generations ending with an end-of-sequence (EOS) token rather than truncation) correlates strongly with benchmark accuracy (24% termination at 1 epoch vs 89% at 32 epochs for OLMo3-7B; see `./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/sections/Termination correlates with performance..md`), but does not identify which part of the training objective drives this.

A key detail is that the reproduction code (**[GitHub - dkopi/data-repetition](./references/GitHub---dkopi-data-repetition/meta/meta_info.txt)**) performs SFT with:

- **batch size = 1**, and
- **per-sequence mean cross-entropy** over response tokens (`reduction="mean"` in `torch.nn.functional.cross_entropy`).

In long-CoT data, response lengths vary widely (hundreds to up to ~10k tokens after filtering). With batch size 1, per-sequence mean loss induces a **sequence-weighted** objective: each training example contributes roughly equal total gradient per step, while each token's gradient is scaled by `1/T` where `T` is the response length. This can underweight long reasoning traces (including late structural tokens like conclusion/answer formatting and EOS behavior) relative to short responses.

Therefore, part of the repetition advantage might be explained by an objective mismatch rather than a fundamental "repetition beats diversity" dynamic:

- In **data scaling** (1 epoch on 51.2k samples), the model sees many long traces, but the per-token learning signal on those long traces is weak (scaled by `1/T`).
- In **data repetition** (32 epochs on 1.6k samples), the model revisits the same long traces many times, which may compensate for this per-token underweighting and improve termination/format learning.

### Key Insight and Hypothesis

**Key insight.** Under batch size 1, the choice of loss normalization (per-sequence mean vs length-weighted / token-sum) changes whether training is effectively sequence-weighted or token-weighted. This is a plausible hidden variable in long-CoT SFT, where long outputs are exactly the behaviors we want to learn.

**Hypothesis.** If the repetition advantage is partly due to per-sequence mean loss underweighting long traces, then switching the data-scaling run to a **length-weighted (token-sum normalized)** objective should recover a substantial fraction of the repetition advantage at the same optimizer-step budget.

Why this could be wrong: (i) the repetition advantage may be dominated by other effects (e.g., specific format/termination token learning, implicit compute differences, or memorization dynamics unrelated to length weighting), or (ii) length-weighted loss could destabilize optimization (effectively changing gradient scale) and hurt performance.

---

## Proposed Approach

### Overview

We propose a minimal mechanism test: modify the SFT objective in the **data-scaling** condition to be **length-weighted**, while keeping the optimizer-step budget, data, model, and evaluation protocol identical to the source setup.

Concretely, for a training example with response length `T` and per-token cross-entropy losses `\ell_1,\dots,\ell_T`:

- **Baseline (mean loss, used in the source repo):**
  \[
  L_{\text{mean}} = \frac{1}{T}\sum_{t=1}^T \ell_t.
  \]

- **Proposed (length-weighted / token-sum normalized):**
  \[
  L_{\text{len}} = \frac{1}{T_{\text{ref}}}\sum_{t=1}^T \ell_t = \frac{T}{T_{\text{ref}}} L_{\text{mean}}.
  \]

`T_ref` is a fixed constant chosen as the **mean response length** (in tokens) computed once from the training split, so the expected scaling factor `E[T/T_ref] \approx 1` and the learning-rate semantics remain comparable.

### Method Details

**Implementation (one-line change).** In `train.py`, after computing `loss = cross_entropy(..., reduction="mean")` on response tokens, multiply by `T/T_ref`, where `T` is the number of response tokens for that example.

**Controls to isolate the mechanism.**

- Same base model checkpoint, tokenizer, chat template, data splits, number of optimizer steps (51,200), optimizer (8-bit Adam), LR schedule (cosine with 10% warmup), gradient clipping, and dataset shuffling.
- Only change: replace `L_mean` with `L_len` in the data-scaling condition.

**Diagnostics (no extra training conditions).** Log per-step:
- response length `T`
- scaling factor `T/T_ref`
- gradient norm (already logged in the repo)
- training token accuracy (as in the source paper)

### Key Innovations

- A **mechanism-driven confound test** for a high-impact empirical claim in long-CoT SFT, targeting loss normalization rather than data schedules.
- A **drop-in objective fix** for batch-size-1 SFT on variable-length instruction-following data, which may improve termination and long-output learning without requiring repeated epochs.

---

## Related Work

### Field Overview

This proposal connects three lines of work:

1. **Long-CoT post-training for reasoning.** Chain-of-thought prompting and long-CoT SFT are standard tools for improving reasoning (e.g., **[Wei et al., 2022](https://arxiv.org/abs/2201.11903)**; modern post-training pipelines often include multi-epoch SFT before other stages such as preference optimization).

2. **Data repetition vs data scaling under compute constraints.** Classical pretraining scaling laws typically emphasize fresh data under a compute budget (**[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)**; **[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)**). Data repetition has been analyzed in data-constrained regimes (**[Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)**), but Kopiczko et al. identify a much stronger repetition advantage specifically in long-CoT SFT.

3. **Loss masking / weighting for variable-length sequences.** In instruction tuning, it is common to mask prompt tokens and train only on response tokens (**[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)**; **[Touvron et al., 2023](https://arxiv.org/abs/2307.09288)**). Recent work shows that changing which tokens contribute to loss can alter generalization and memorization dynamics (**[Shi et al., 2024](https://arxiv.org/abs/2405.14394)**). Separately, token-level weighting has been proposed as a training knob for long-context language modeling (**[Helm et al., 2025](https://arxiv.org/abs/2503.09202)**). Our proposal is a minimal, global length-based reweighting that targets a specific phenomenon in long-CoT SFT.

### Related Papers

- **[Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**: Introduces the repetition advantage under step-matched training; our work tests whether loss normalization explains part of the gap.
- **[GitHub - dkopi/data-repetition](./references/GitHub---dkopi-data-repetition/meta/meta_info.txt)**: Provides the training (`train.py`) and evaluation (`eval.py`) harness used in our experiments.
- **[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)**: Establishes chain-of-thought as a behavior that can be elicited and taught.
- **[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)**: A standard inference-time scaling baseline (sample multiple solutions and aggregate).
- **[Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)**: Classic compute/data trade-off framing; motivates careful definitions of "compute-matched".
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)**: Foundational scaling laws perspective.
- **[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)**: Analyzes repetition under constrained data; contrasts with the much larger effect in long-CoT SFT.
- **[InstructGPT](https://arxiv.org/abs/2203.02155)**: Canonical instruction-following post-training pipeline using SFT as a first stage.
- **[LLaMA 2](https://arxiv.org/abs/2307.09288)**: Reports instruction tuning practices and response-only loss masking.
- **[QLoRA](https://arxiv.org/abs/2305.14314)**: Widely used parameter-efficient fine-tuning recipe; relevant to feasibility downscaling if full fine-tuning is too expensive.
- **[FLAN](https://arxiv.org/abs/2109.01652)**: Instruction tuning at scale; shows diverse tasks with variable output lengths.
- **[OLMo 3 Technical Report](https://arxiv.org/abs/2512.13961)**: Provides model and post-training context for the Dolci dataset used by Kopiczko et al.
- **[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)**: Describes Qwen3 base models used in the source paper.
- **[DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)**: Modern reasoning pipeline that uses multi-epoch SFT before later stages.
- **[GPQA](https://arxiv.org/abs/2311.12022)**: Graduate-level multiple-choice QA benchmark used for evaluation.
- **[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)**: Efficient long-generation inference used by the source repo.
- **[8-bit Optimizers via Block-Wise Quantization](https://arxiv.org/abs/2110.02861)**: 8-bit Adam optimizer used in the source training recipe.
- **[Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation is Wasteful](https://arxiv.org/abs/2502.04041)**: Motivates batch size 1 training choices used in the source paper.
- **[Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394)**: Shows that changing loss masking across prompt/response affects generalization and memorization.
- **[Token Weighting for Long-Range Language Modeling](https://arxiv.org/abs/2503.09202)**: Proposes token-level reweighting schemes; demonstrates that non-uniform token weights can change long-context behavior.
- **[Entropic Distribution Matching for Supervised Fine-Tuning of LLMs](https://openreview.net/forum?id=dulz3WVhMR)**: Studies how objective modifications in SFT affect overfitting and generalization.
- **[DoReMi: Optimizing Data Mixtures for Language Model Pretraining](https://arxiv.org/abs/2305.10429)**: Data reweighting under compute constraints; complementary to loss reweighting.
- **[D3: Large-Scale Data Selection for Instruction Tuning](https://arxiv.org/abs/2503.11441)**: Data selection for instruction tuning; relevant when repetition amplifies subset properties.
- **[SELECT2REASON](https://arxiv.org/abs/2505.17266)**: Selection strategies for reasoning instruction tuning; highlights the role of example properties such as difficulty/length.
- **[Too Long, Do Re-weighting for Efficient LLM Reasoning Compression](https://arxiv.org/abs/2506.02678)**: Reweighting between short and long reasoning traces for efficiency; adjacent to our length-driven objective change.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Long-CoT SFT dynamics | Repetition/epochs change termination, memorization, and reasoning behavior | Kopiczko et al. 2026; DeepSeek-R1 report 2025 | AIME, GPQA; Acc@k, Pass@k, termination | Mechanisms unclear; compute matching subtle with variable lengths |
| Data scaling vs repetition | Trade off unique data vs repeated exposure under fixed compute | Kaplan 2020; Hoffmann 2022; Muennighoff 2023 | Loss scaling; downstream transfer | Mostly pretraining; may not predict post-training behavior |
| Loss masking / weighting | Which tokens/examples dominate gradients under variable length | Ouyang 2022; Shi 2024; Helm 2025 | Instruction following + reasoning benchmarks | Often not tied to a specific high-impact phenomenon |

### Closest Prior Work

**Kopiczko et al. (2026)** is the closest prior work. It establishes the empirical repetition advantage under step-matched training and documents correlates (termination rate, token-accuracy memorization). Our proposal changes only the loss normalization in the data-scaling condition to test a concrete mechanism for the observed gap.

**Shi et al. (2024)** shows that loss masking choices (whether to include prompt tokens in the loss) substantially affect instruction-tuning generalization. This supports the broader claim that loss definition, not just data quantity, can dominate post-training outcomes.

**Helm et al. (2025)** proposes token-level loss weighting for long-context language modeling, demonstrating that non-uniform token weights are an effective training knob. Our proposal is a simpler length-based weighting and targets long-CoT SFT rather than context extension.

**Marek et al. (2025)** argues that very small batch sizes can be compute-efficient and effective, and helps motivate why batch size 1 is used in the source setup; it also highlights that small-batch regimes can behave differently than large-batch regimes.

**Novelty Kill Search Summary:** We searched for prior work combining long-CoT SFT repetition-vs-scaling with length-weighted/token-sum loss normalization (e.g., "length-weighted loss long CoT SFT", "sequence-weighted vs token-weighted cross entropy instruction tuning", "Data Repetition Beats Data Scaling loss normalization", OpenReview queries for "length-weighted loss supervised fine-tuning"). No prior work explicitly testing length-weighted SFT as a mechanism explanation for Kopiczko et al.'s repetition advantage was found as of 2026-02-19. (Full query log in `notes.md`.)

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Kopiczko et al. 2026 | Step-matched epochxsample grid; finds repetition advantage; reports termination/memorization correlates | Mechanism unclear; loss normalization under batch size 1 not examined | Replace per-seq mean loss with length-weighted loss in data-scaling run | If length underweighting is causal, C should close a large fraction of A->B gap |
| Shi et al. 2024 | Studies loss masking over instruction vs response tokens | Not about long-CoT repetition; does not target termination | Apply "loss definition matters" lens to long-CoT SFT | Supports plausibility that a simple loss change can dominate generalization |
| Helm et al. 2025 | Token-level weighting for long-context pretraining | Different training stage and weighting scheme | Use simple length-based weighting in SFT | Provides precedent that token weights are a meaningful training knob |
| Muennighoff et al. 2023 | Theory/empirics of repetition in data-constrained pretraining | Mostly pretraining; predicts diminishing returns from repetition | Focus on post-training SFT; isolate loss normalization mechanism | Helps interpret whether post-training "repetition advantage" is a different phenomenon |

---

## Experiments

### Experimental Setup

**Codebase:** `dkopi/data-repetition` (we will modify only the loss computation for condition C).

**Base model:** `allenai/Olmo-3-1025-7B` (as used in the repo README).

**Training conditions (3 main conditions, all run with 3 seeds):**

- **A - Data scaling (baseline):** `dakopi/dolci_think__train_51200`, 1 epoch, `L_mean`.
- **B - Repetition (baseline):** `dakopi/dolci_think__train_1600`, 32 epochs, `L_mean`.
- **C - Length-weighted data scaling (ours):** `dakopi/dolci_think__train_51200`, 1 epoch, `L_len` with `T_ref = mean response length` on the 51.2k training split.

**Training data context (from Kopiczko et al. 2026):** the Dolci long-CoT SFT data comes from the OLMo3 post-training pipeline and contains distilled long reasoning traces across math, coding, and instruction following. The source paper filters to first-turn conversations with complete `<think>...</think>` traces and removes samples exceeding 10k tokens.

**Training recipe (match source):** BF16, Unsloth kernels, 8-bit Adam, cosine LR schedule with 10% warmup, batch size 1, prompt masked (loss on response tokens only), gradient clip 1.0, max update budget 51,200.

**Baseline Ladder (REQUIRED):**

- **Prompting baseline (no training):** Base model evaluated with the same prompts used by `eval.py` (including the "place final answer in \\boxed{}" instruction).
- **Inference-time scaling baseline:** Best-of-N sampling (already part of the paper protocol): AIME uses N=16, GPQA uses N=4.
- **Training baselines:** A (data scaling) and B (repetition) reproduce the source finding.
- **Proposed method:** C (length-weighted loss in data scaling).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AIME 2024 | 30 competition math problems; integer answer (0-999) | Acc@16, Pass@16, Termination | test | https://huggingface.co/datasets/math-ai/aime24 | `dkopi/data-repetition/eval.py` |
| AIME 2025 | same as above | Acc@16, Pass@16, Termination | test | https://huggingface.co/datasets/math-ai/aime25 | `dkopi/data-repetition/eval.py` |
| GPQA Diamond | graduate-level multiple-choice science QA | Acc@4, Pass@4, Termination | test | (via repo `data/prepare_gpqa.py`) | `dkopi/data-repetition/eval.py` |

**Metric definitions:**
- **Acc@k (called Acc@n in the source paper)**: accuracy averaged over k sampled generations per problem.
- **Pass@k**: fraction of problems solved in at least one of k generations.
- **Termination**: percent of generations that end with an EOS token id (not truncation).

**Primary aggregate metric (to match Table 4):** unweighted mean of per-benchmark Acc@k (and Pass@k) across {AIME'24, AIME'25, GPQA}.

### Main Results

#### Results Table

("Published" numbers copied verbatim from Table 4 of **[Kopiczko et al., 2026](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/sections/Memorization%20signals%20convergence..md)** for OLMo3-7B at update budget B=51,200; those are averages across the three benchmarks.)

| Method | Base Model | Train setting | Objective | Avg Acc@k (mean+/-std) | Avg Pass@k (mean+/-std) | Source | Notes |
|---|---|---|---|---:|---:|---|---|
| A: Data scaling | OLMo3-7B | 51.2k x 1 epoch | mean CE | 17.2 (1 run) | 40.1 (1 run) | Kopiczko et al. 2026 (Table 4) | Published aggregate |
| B: Repetition | OLMo3-7B | 1.6k x 32 epochs | mean CE | 38.8 (1 run) | 73.7 (1 run) | Kopiczko et al. 2026 (Table 4) | Published aggregate |
| **C: Length-weighted (ours)** | OLMo3-7B | 51.2k x 1 epoch | length-weighted CE | **TBD** | **TBD** | This work | To be verified (3 seeds) |

### Ablation Studies

No additional training ablations are required beyond A/B/C. We will report diagnostics (length distribution, scaling factors, gradient norms, train token accuracy) to interpret outcomes.

### Experimental Rigor

- **Seeds:** `seeds=[42, 123, 456]` for each training condition.
- **Replication gate:** Before interpreting condition C, verify that A vs B reproduces the direction and approximate magnitude of the source paper's A->B gap on our infrastructure.
- **Confound controls:** same base model, tokenizer, prompts, evaluation script, optimizer, LR schedule, update budget.
- **Sanity check:** log the response-length distribution for the 51.2k and 1.6k splits (mean/p50/p90/p99). This is diagnostic; we do not assume one split is systematically longer.

**Resource Estimate** (bounded by 768 A100 GPU-hours):

- Source paper reports up to 24h on 1xH100 94GB per run.
- Assume conservatively up to 36h on 1xA100-80GB per run.
- Training: 3 conditions x 3 seeds x 36h ~= **324 A100 GPU-hours**.
- Evaluation: vLLM inference on 508 problems with multiple samples; budget **<=150 A100 GPU-hours** total.
- **Total** ~= **<=474 A100 GPU-hours**.

---

## Success Criteria

**Hypothesis**: Length-weighted loss in the data-scaling condition substantially closes the repetition advantage gap (relative to standard mean loss) on downstream reasoning benchmarks, and improves termination behavior.

**Decision Rule**:

Let `Gap = Metric(B) - Metric(A)` using the primary aggregate metric (Avg Pass@k or Avg Acc@k), averaged over 3 seeds.

- **Proceed (supports mechanism as major contributor):** `Metric(C) - Metric(A) >= 0.60 x Gap` and termination(C) is not worse than termination(B) by more than 10 percentage points.
- **Pivot (partial support):** `0.20 x Gap <= Metric(C) - Metric(A) < 0.60 x Gap`. In this case, prioritize follow-up analysis (gradient norms vs length, termination-only vs accuracy-only gains) rather than adding new training conditions.
- **Refute (not a meaningful contributor):** `Metric(C) - Metric(A) < 0.20 x Gap` (or C underperforms A).

---

## Impact Statement

If length-weighted loss closes a substantial fraction of the repetition advantage, it provides a simple, general-purpose training modification for long-output SFT regimes (especially batch size 1 or small micro-batches) that can improve reasoning and termination without requiring many repeated epochs. Even a strong negative result would be valuable: it would rule out a prominent objective-normalization confound and strengthen the conclusion that the repetition advantage reflects deeper memorization/behavioral convergence dynamics.

---

## References

All cited works are listed in **Related Papers** above; the primary references are **Kopiczko et al. (2026)** (arXiv:2602.11149) and the `dkopi/data-repetition` codebase.