# untitled

# Termination-Aware SFT: Testing Whether the Long-CoT “Repetition Advantage” Is Mostly a Parsing/Termination Effect

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Supervised fine-tuning (SFT) on long chain-of-thought (long-CoT) demonstrations is a standard step for producing reasoning-capable language models. In these datasets, each training example contains a long reasoning trace (often wrapped in tags like `<think>...</think>`) followed by a short final answer. For evaluation on benchmarks such as AIME (the American Invitational Mathematics Examination; a competition-math benchmark with 30 integer-answer problems) and GPQA (a graduate-level multiple-choice science benchmark with 448 questions), models are usually prompted to provide a final answer in a machine-parseable format (e.g., `\\boxed{...}`), and success is scored by automatic answer extraction.

A recent paper reports a strong and counterintuitive phenomenon in long-CoT SFT: at a fixed gradient-update budget, training for many epochs on a small dataset can outperform training for one epoch on a much larger dataset, sometimes by double-digit accuracy margins. They also report that termination rate (the fraction of generations that end with an end-of-sequence token rather than being truncated by max tokens) increases dramatically with epoch scaling and correlates strongly with measured accuracy (**[Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**).

This creates a practical question for post-training small and mid-sized models (3–8B parameters): is the “repetition advantage” evidence that repeated exposure teaches better reasoning, or is it mainly teaching the model to reliably finish the required output structure (close `</think>`, emit the final answer span, and stop) so that automatic evaluation can score it?

### The Problem

The central empirical claim in **Data Repetition Beats Data Scaling** is that, at a fixed update budget of 51,200 gradient updates (batch size 1), an OLMo3-7B base model (an open-source 7B-parameter pretrained language model released by AI2) trained for 32 epochs on 1,600 samples reaches about **39%** average accuracy across AIME’24/AIME’25/GPQA, compared to about **17%** for 1 epoch on 51,200 samples (reported in their analysis). They also report that single-epoch models terminate only **24%** of generations while 32-epoch models terminate about **89%** of generations (their termination analysis).

A key concern is that long-CoT benchmarks are scored by extracting a final answer from generated text. If a model fails to terminate or fails to produce a parseable final answer (e.g., it gets truncated mid-trace), it is scored as incorrect even if its partial reasoning is on track. This means that improvements in measured accuracy could be mediated by improvements in “completion reliability” rather than improvements in conditional reasoning quality.

This proposal asks:

1) **Mediation question**: Does the accuracy gap between (A) single-epoch data-scaled SFT and (B) multi-epoch repeated SFT mostly disappear when we condition on generations where the final answer is parseable?

2) **Intervention question**: If the gap is mostly mediated by parseability/termination, can we recover most of the repetition advantage with a single-epoch training run by explicitly increasing the loss weight on structure/termination tokens (e.g., `</think>` and EOS) without changing the dataset size?

### Key Insight and Hypothesis

**Key insight.** The source paper observes that termination rate rises sharply with epoch scaling and correlates with accuracy. This suggests that part of the “repetition advantage” may come from learning output conventions (how to end a long reasoning trace and present a final answer) rather than learning better problem-solving strategies.

**Hypothesis (mediation).** For long-CoT SFT at fixed compute, multi-epoch repetition primarily improves *completion reliability* (parseable final answers and termination), and conditional correctness among parseable generations changes only modestly. Concretely, the unconditional accuracy gap between A and B should shrink substantially when measuring accuracy conditioned on successful answer parsing.

**Hypothesis (practical fix).** A deterministic “termination-aware” loss reweighting in the single-epoch setting can recover most of B’s gains in (i) parse rate and (ii) unconditional accuracy, by encouraging the model to learn the structural termination convention faster without requiring many epochs.

Why this could be wrong:
- The repetition advantage may reflect genuine improvements in reasoning strategy (Acc|Parse improves), not just improved termination.
- Conditioning on parseable generations can introduce selection effects (easy problems are more likely to be parseable in A), so a naive conditional comparison could be misleading.
- Loss reweighting might improve parse rate while harming conditional correctness (the model ends early with wrong answers).

---

## Proposed Approach

### Overview

We propose **Termination-Aware SFT**, a training modification and an analysis protocol to test whether the repetition advantage is mostly a termination/parsing effect.

We train and evaluate three conditions at the same update budget (B = 51,200):

- **A (Data scaling baseline)**: 1 epoch on 51,200 unique long-CoT samples.
- **B (Repetition baseline)**: 32 epochs on a nested 1,600-sample subset (32 × 1,600 = 51,200 updates).
- **C (Termination-aware SFT, ours)**: 1 epoch on 51,200 samples with token-level loss reweighting that increases gradient signal on structure/termination tokens.

We then compute both standard benchmark metrics (Acc@n and Pass@n, where Acc@n is accuracy after sampling n independent generations per problem and Pass@n is the fraction of problems for which at least one of the n samples is correct) and new diagnostic metrics that measure parseability and conditional correctness.

### Method Details

#### Datasets and formatting

We follow the source paper’s setup on **Dolci**, a distilled long-CoT supervised fine-tuning dataset containing `<think>...</think>` reasoning traces spanning math, code, and general instruction-following, which is used in the OLMo3 post-training pipeline. Dolci contains `<think>...</think>` traces (**[Data Repetition Beats Data Scaling](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**). For evaluation, we append an instruction requesting final answers in `\\boxed{...}` as in the source paper.

#### Termination-aware loss

Let the standard SFT objective be cross-entropy over response tokens only (masking the user prompt). We modify it by assigning a larger weight to “structure tokens” that mark successful completion.

**Structure token set.** We define a token-position mask over each response:

- Positions belonging to the substring `</think>` (end of the reasoning block).
- The final EOS token position (sequence termination).

(If the training data contains `\\boxed{` patterns, these positions can also be included, but the primary definition uses `</think>` and EOS only.)

**Weighted loss.** For token positions i in the response, we compute per-token cross-entropy loss \(\ell_i\). We assign weights \(w_i\):

- \(w_i = \alpha\) if i is a structure token position.
- \(w_i = 1\) otherwise.

The objective is \(L = \frac{\sum_i w_i \ell_i}{\sum_i w_i}\).

**Deterministic choice of \(\alpha\).** We choose \(\alpha\) from training dataset statistics so that (after reweighting) structure-token positions contribute a fixed fraction \(f=0.05\) of the *total weighted token mass* in the loss. Let \(M\) be the total number of response-token positions and \(M_s\) be the number of structure-token positions (counted across the full training set). Then:

\[
\alpha = \frac{f\,(M-M_s)}{(1-f)\,M_s}.
\]

We cap \(\alpha\) at 50 to avoid extreme gradients (pre-registered cap).

#### Diagnostic metrics

We measure, in addition to the paper’s metrics:

- **ParseRate**: fraction of generations for which the evaluator can extract a final answer (e.g., a valid `\\boxed{...}` answer).
- **Acc|Parse**: accuracy conditioned on parseable generations.
- **Acc|Parse (intersection)**: Acc|Parse restricted to the set of benchmark problems where *both* A and B have at least one parseable generation (to reduce selection bias).

### Key Innovations

- **Mechanism test for the repetition advantage**: a concrete mediation analysis that distinguishes “better reasoning” from “better completion reliability” in long-CoT SFT.
- **Termination-aware SFT**: a simple, deterministic loss reweighting that targets structure/termination learning without additional epochs.
- **Evaluation protocol additions** (ParseRate, Acc|Parse, intersection Acc|Parse) that make termination-related confounds explicit.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) data/compute trade-offs in SFT, (ii) chain-of-thought post-training and its evaluation, and (iii) training or decoding techniques that affect termination and output formatting.

The source paper shows that, unlike pretraining scaling laws where fresh data often dominates repetition, long-CoT SFT can benefit strongly from repetition at fixed update budgets. They also highlight termination rate as a key correlate, but do not test whether the accuracy gains are primarily mediated by parseability/termination. This proposal contributes a focused causal test and a minimal training intervention.

Separately, work on chain-of-thought faithfulness and fine-tuning effects suggests that post-training can change not only accuracy but also the relationship between intermediate reasoning and final answers (**[On the Impact of Fine-Tuning on Chain-of-Thought Reasoning](./references/On-the-Impact-of-Fine-Tuning-on-Chain-of-Thought-Reasoning/meta/meta_info.txt)**). Work on entropy minimization argues that some “reasoning improvements” from post-training can be viewed as capability elicitation and confidence sharpening rather than learning new skills (**[The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning](./references/The-Unreasonable-Effectiveness-of-Entropy-Minimization-in-LLM-Reasoning/meta/meta_info.txt)**). Finally, termination failures and EOS-related pathologies have been analyzed in other model families (e.g., diffusion LLMs) with simple training-time fixes (**[Rainbow Padding](./references/Rainbow-Padding-Mitigating-Early-Termination-in-Instruction-Tuned-Diffusion-LLMs/meta/meta_info.txt)**), supporting the broader premise that termination behavior is a trainable, failure-prone interface component.

### Related Papers

- **[Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**: Establishes the repetition advantage and reports strong termination–accuracy correlation; our proposal tests mediation and a minimal single-epoch alternative.
- **[On the Impact of Fine-Tuning on Chain-of-Thought Reasoning](./references/On-the-Impact-of-Fine-Tuning-on-Chain-of-Thought-Reasoning/meta/meta_info.txt)**: Studies how fine-tuning affects CoT accuracy and faithfulness; motivates conditioning/termination-style diagnostics.
- **[The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning](./references/The-Unreasonable-Effectiveness-of-Entropy-Minimization-in-LLM-Reasoning/meta/meta_info.txt)**: Shows that confidence sharpening can elicit reasoning gains without labels; supports the idea that some gains can be interface/elicitation effects.
- **[Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs](./references/Rainbow-Padding-Mitigating-Early-Termination-in-Instruction-Tuned-Diffusion-LLMs/meta/meta_info.txt)**: Diagnoses EOS-driven early termination and proposes a training-time fix; relevant as a termination-mechanism precedent.
- **[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)**: Shows that repetition is usually not beneficial in pretraining; contrasts with repetition advantage in SFT.
- **[Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)**: Canonical compute–data scaling laws for pretraining; provides contrast to post-training behavior.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)**: Foundational scaling law work; motivates why the repetition advantage is surprising.
- **[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)**: Introduces CoT prompting as a reasoning interface; motivates output-structure sensitivity.
- **[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)**: Inference-time scaling baseline; relevant as an alternative way to improve Acc@n without changing training.
- **[Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation is Wasteful](https://arxiv.org/abs/2502.04041)**: Supports the batch size 1 choice used in the source paper.
- **[Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision](https://arxiv.org/abs/2402.07826)**: Source paper relates teacher-quality effects to weak-to-strong generalization.
- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)**: Distillation + post-training pipelines for math reasoning; provides context for long-CoT SFT.
- **[DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)**: Widely used long-CoT post-training pipeline (SFT + RL) where termination and formatting matter.
- **[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)**: Provides model-side evaluation and recommended sampling settings used by the source paper.
- **[OLMo 3 Technical Report](https://arxiv.org/abs/2512.13961)**: Describes the OLMo3 post-training pipeline and Dolci datasets used in the source paper.
- **[Generalizing Verifiable Instruction Following](https://arxiv.org/abs/2501.17130)**: One data source inside Dolci; relevant for output-format adherence.
- **[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)**: Shows small high-quality SFT sets can be strong; related to small-data post-training regimes.
- **[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)**: Common post-training step; relevant as alternative to SFT scaling.
- **[RewardBench: Evaluating Reward Models](https://arxiv.org/abs/2403.13787)**: Context for later-stage post-training evaluation; not directly used but relevant ecosystem.
- **[The Entropy Enigma: Success and Failure of Entropy Minimization](https://arxiv.org/abs/2410.05571)**: Discusses when entropy minimization helps; supports capability-elicitation framing.
- **[Adaptive Injection Decoding for Reasoning](https://aclanthology.org/2025.findings-acl.520/)**: Test-time method to reduce premature termination by injecting continuation tokens.
- **[Dynamic Early Exit in Reasoning Models](https://arxiv.org/abs/2504.15895)**: Studies stopping/termination as a controllable dimension of reasoning compute.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Epoch/data trade-offs in SFT | Vary epochs vs unique samples at fixed compute | Data Repetition Beats Data Scaling; LIMA | AIME, GPQA, MMLU | Mechanism unclear; termination confounds possible |
| CoT evaluation and faithfulness | Measure how CoT changes under fine-tuning and truncation | On the Impact of Fine-Tuning on CoT; Self-Consistency | GSM8K/MATH/AIME variants | Faithfulness metrics can be noisy; selection effects |
| Confidence/entropy-based elicitation | Improve reasoning by sharpening distributions | Entropy Minimization; Entropy Enigma | AIME, Math-500, coding | Depends on confidence–correctness correlation |
| Termination/format interventions | Fix early termination / format failures via training-time changes | Rainbow Padding; adaptive decoding methods | MATH, GSM8K, HumanEval | Often architecture- or data-format-specific |

### Closest Prior Work

1) **Data Repetition Beats Data Scaling in Long-CoT SFT**: Shows strong gains from repetition and reports termination correlation, but does not test whether the accuracy gains are mediated by parseability/termination. We add a concrete mediation test (Acc|Parse and intersection analysis) and a minimal single-epoch training alternative.

2) **On the Impact of Fine-Tuning on Chain-of-Thought Reasoning**: Evaluates how fine-tuning changes CoT quality and faithfulness, including early termination-style analyses. Our work differs by focusing on long-CoT SFT training dynamics (epochs vs data) and by proposing a training-time objective change tied to termination/parseability.

3) **Rainbow Padding**: Fixes a specific EOS-driven early termination pathology in diffusion LLMs. Our work targets autoregressive long-CoT SFT and uses loss reweighting rather than padding token redesign, but shares the general insight that termination behavior can dominate measured accuracy.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Data Repetition Beats Data Scaling | Discovers repetition advantage; reports termination correlation | Does not isolate whether gains are termination-mediated | Add mediation analysis + termination-aware loss | Distinguishes “better reasoning” vs “better completion reliability” and provides a cheaper alternative |
| On the Impact of Fine-Tuning on CoT | Measures how fine-tuning affects CoT faithfulness | Not about epoch-vs-data trade-off | Apply conditional parsing analysis to repetition setting | Provides mechanistic insight for a new 2026 phenomenon |
| Rainbow Padding | Fixes EOS overflow in diffusion LLMs | Different architecture; different failure cause | Use token-level loss reweighting in AR SFT | Tests whether a similarly simple fix exists for AR long-CoT SFT |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| OLMo-3-1025-7B (base) | 7B | https://huggingface.co/allenai/Olmo-3-1025-7B | Matches source paper; pretrained checkpoint before instruction tuning |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Dolci-Think-SFT-7B (filtered to 1-turn, `<think>` present, ≤10k tokens) | Long-CoT SFT | up to 51.2k samples | https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B | (see dataset card) |

(If available, use the authors’ pre-split datasets in the `dakopi/data-repetition` Hugging Face collection to match exact splits: https://huggingface.co/collections/dakopi/data-repetition.)

**Other Resources (if applicable):**
- Source code for training/eval: https://github.com/dkopi/data-repetition (train.py, eval.py)

**Resource Estimate**:
- **Compute budget**: The source paper reports each configuration runs on a single H100 94GB GPU for up to 24 hours. We require 3 configurations (A/B/C). On A100-80GB, estimate **~120–240 GPU-hours total** (depending on throughput and any retries), within the 768 GPU-hour cap.
- **GPU memory**: up to 80GB (7B BF16 + optimizer states; 8-bit Adam reported by source paper).
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AIME 2024 | 30 competition math problems; answer is an integer 0–999 | Acc@16, Pass@16, ParseRate, Acc|Parse | test | (use authors’ eval scripts) | dkopi/data-repetition eval.py |
| AIME 2025 | 30 competition math problems; answer is an integer 0–999 | Acc@16, Pass@16, ParseRate, Acc|Parse | test | (use authors’ eval scripts) | dkopi/data-repetition eval.py |
| GPQA | 448 graduate-level multiple-choice questions | Acc@4, Pass@4, ParseRate, Acc|Parse | test | https://huggingface.co/datasets/Idavidrein/gpqa | dkopi/data-repetition eval.py |

**Evaluation Scripts:**
- Use the source repo’s evaluation protocol: sample up to 30k tokens per generation; n=16 for AIME and n=4 for GPQA; use vLLM for inference.
- Implement ParseRate and Acc|Parse by instrumenting the answer-extraction function used by the eval script (counting parse failures explicitly).

### Main Results

#### Comparability Rules (CRITICAL)

All comparisons use the same base model, same update budget (B=51,200), same evaluation prompts, and same inference sampling settings.

#### Results Table

| Method | Base Model | Update budget | Data/epochs | AIME’24 Acc@16 | AIME’25 Acc@16 | GPQA Acc@4 | Aggregate Acc@n | ParseRate | Acc|Parse | Source | Notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A: 1-epoch data scaling | OLMo3-7B | 51,200 | 51.2k × 1 | **TBD** | **TBD** | **TBD** | ~17% (reported) | ~24% (termination; reported) | **TBD** | [Data Repetition Beats Data Scaling](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt) | Must be re-run for exact numbers under our environment |
| B: 32-epoch repetition | OLMo3-7B | 51,200 | 1.6k × 32 | **TBD** | **TBD** | **TBD** | ~39% (reported) | ~89% (termination; reported) | **TBD** | [Data Repetition Beats Data Scaling](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt) | Must be re-run |
| **C: Termination-aware SFT (ours)** | OLMo3-7B | 51,200 | 51.2k × 1 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | - | Loss reweighting only |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| C (no cap) | remove α cap (keep 5% loss-mass rule) | tests if cap matters for stability |
| C (structure tokens = EOS only) | weight EOS only | if EOS dominates termination learning, similar gains |
| C (structure tokens = </think> only) | weight </think> only | tests whether closing the reasoning block is the bottleneck |

(Ablations are optional; the decisive experiment is A/B/C.)

### Analysis (Optional)

- Plot ParseRate vs Acc@n across A/B/C to quantify mediation.
- For each benchmark, report “intersection Acc|Parse” on problems where both A and B have at least one parseable generation.

---

## Success Criteria

**Criterion 1: Mediation test (scientific claim)**
- Hypothesis: The A→B accuracy gains are largely mediated by parseability/termination.
- Validation: B improves ParseRate over A by a large margin (expected from the source paper), and the unconditional accuracy gap between A and B shrinks substantially when measuring Acc|Parse, especially on the intersection set.

**Criterion 2: Practical alternative to heavy repetition**
- Hypothesis: Termination-aware SFT (C) recovers most of the repetition advantage without multi-epoch training.
- Validation: C closes a large fraction of the A→B gap on ParseRate and Aggregate Acc@n, without reducing Acc|Parse relative to B. For robustness, report the overlap fraction of parseable problem sets between A and B and Acc|Parse on the intersection set.

---

## Impact Statement

If termination-aware SFT works, practitioners training reasoning models can reduce the need for extreme multi-epoch repetition to obtain reliable long-CoT completions. This would reduce post-training cost and simplify training recipes by targeting a concrete failure mode (completion reliability) with a small, deterministic objective change.

---

## References

- [Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt) - Kopiczko et al., 2026
- [On the Impact of Fine-Tuning on Chain-of-Thought Reasoning](./references/On-the-Impact-of-Fine-Tuning-on-Chain-of-Thought-Reasoning/meta/meta_info.txt) - Lobo et al., 2025
- [The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning](./references/The-Unreasonable-Effectiveness-of-Entropy-Minimization-in-LLM-Reasoning/meta/meta_info.txt) - Agarwal et al., 2025
- [Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs](./references/Rainbow-Padding-Mitigating-Early-Termination-in-Instruction-Tuned-Diffusion-LLMs/meta/meta_info.txt) - Kim et al., 2025
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) - Muennighoff et al., 2023
- [Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Kaplan et al., 2020
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) - Wei et al., 2022
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation is Wasteful](https://arxiv.org/abs/2502.04041) - Marek et al., 2025
- [Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision](https://arxiv.org/abs/2402.07826) - Burns et al., 2024
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948) - Guo et al., 2025
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - Qwen Team, 2025
- [OLMo 3 Technical Report](https://arxiv.org/abs/2512.13961) - Team OLMo, 2025
- [Generalizing Verifiable Instruction Following](https://arxiv.org/abs/2501.17130) - (Dolci data source), 2025
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) - Zhou et al., 2023
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [RewardBench: Evaluating Reward Models](https://arxiv.org/abs/2403.13787) - Lambert et al., 2024
- [The Entropy Enigma: Success and Failure of Entropy Minimization](https://arxiv.org/abs/2410.05571) - Press et al., 2024
- [Dynamic Early Exit in Reasoning Models](https://arxiv.org/abs/2504.15895) - (dynamic stopping), 2025
- [Adaptive Injection Decoding for Reasoning](https://aclanthology.org/2025.findings-acl.520/) - (termination control), 2025
