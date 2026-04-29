# untitled

# Compute-Matched Repetition Advantage: Does Long-CoT SFT Still Benefit from Repetition When Matching Training Tokens?

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Supervised fine-tuning (SFT) on long chain-of-thought (long-CoT) demonstrations is a common step for making pretrained language models more reliable at multi-step reasoning. High-quality long-CoT data is expensive, so practitioners often face a trade-off: collect more unique demonstrations (data scaling) or train longer on a smaller set (data repetition).

A recent paper, **Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning** (**[Kopiczko et al., 2026](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**), reports a striking result: at a fixed optimizer-step budget (epochs × samples), repeating a small long-CoT dataset for many epochs can dramatically outperform training on many more unique samples for one epoch (e.g., OLMo3-7B **Acc@n** 17.2 → 38.8 at B=51,200 updates; Table 4). Here **Acc@n** is accuracy averaged over *n* independent generations per problem; **Pass@k** is the fraction of problems solved by at least one of *k* samples.

### The Problem

The paper’s comparisons match **optimizer steps**, but do not explicitly match **total training tokens** (or attention compute). In long-CoT SFT, response lengths vary widely (from a few hundred to up to ~10k tokens, with a long tail of very long responses), so two runs with the same number of optimizer steps can still process different numbers of response tokens and incur different attention FLOPs. This creates a plausible alternative explanation for part of the “repetition advantage”: repeated subsets might (by sampling variation) have longer responses and therefore receive more compute per step.

A minimal, decisive confound check is therefore:

> Does the repetition advantage persist when we **token-match** training (equal total response tokens processed), rather than step-match training?

### Key Insight and Hypothesis

**Key insight.** “Epochs × samples” is only a proxy for compute when sequence lengths vary. Because the core claim is large and practice-relevant, it is important to test whether it is robust to a stricter compute control.

**Hypothesis.** Let A be the step-matched data-scaling run (51.2k samples × 1 epoch) and B be the step-matched repetition run (1.6k samples × 32 epochs), and let Δ_step be the aggregate accuracy gap (B−A). Let B_tok be a repetition run on the same 1.6k subset that stops once it has processed the same number of response tokens as A. We hypothesize token matching removes **at least half** of the step-matched repetition advantage:

\[\Delta_{tok} = \text{Acc}(B_{tok})-\text{Acc}(A) \le 0.5\,\Delta_{step}.\]

Why this could be wrong: the nested subsets may have very similar length distributions, in which case token matching should change little and the repetition advantage should remain.

---

## Proposed Approach

### Overview

We propose **Compute-Matched Repetition Advantage (CMRA)**: a replication-style experiment that adds a single control condition to Kopiczko et al.’s setup.

We compare three training conditions for the same base model and evaluation protocol:

- **A (step-matched data scaling)**: `dakopi/dolci_think__train_51200`, 1 epoch (51,200 optimizer steps).
- **B (step-matched repetition)**: `dakopi/dolci_think__train_1600`, 32 epochs (51,200 optimizer steps).
- **C (token-matched repetition; ours)**: `dakopi/dolci_think__train_1600`, repeated, but early-stop when cumulative **response tokens contributing to loss** match condition A (±0.5%).

### Method Details

**Token accounting.** Use the same tokenizer and masking as training (prompt masked; loss over response tokens only) and count the number of response tokens per step. Stop condition C when the cumulative count reaches the target budget.

**Learning-rate schedule (to avoid schedule confounds).** Use 10% warmup and then a constant learning rate (the LR selected by the source repo’s sweep for 1 epoch on 51.2k samples). Define warmup in terms of **token budget** so that A and C see comparable warmup fractions under token matching.

**Compute proxy logging.** In addition to token counts, log a simple attention-compute proxy per run: \(\sum_t L_t^2\), where \(L_t\) is the response length (tokens) at step *t*. This is motivated by self-attention’s \(O(L^2)\) scaling with sequence length. If token budgets match but this proxy differs by >10%, we report that token matching did not fully compute-match attention and treat conclusions as conditional.

### Key Innovations

- A **decisive compute-control experiment** for a highly visible, counterintuitive SFT result.
- A simple protocol for reporting **token budget and attention-FLOPs proxy** in step-matched post-training studies.

---

## Related Work

### Field Overview

This proposal connects (i) long-CoT SFT dynamics and memorization/termination phenomena, (ii) data repetition vs data scaling in both pretraining and post-training, and (iii) compute/measurement discipline (what it means to “match compute” when sequence lengths vary).

### Related Papers

- **[Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**: Reports the repetition advantage under step matching; our work adds a token-matched control.
- **[GitHub: dkopi/data-repetition](./references/GitHub---dkopi-data-repetition/meta/meta_info.txt)**: Provides training/eval scripts and prepared HF datasets (e.g., `dakopi/dolci_think__train_51200`).
- **[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)**: Analyzes when repetition can help in pretraining; motivates compute-aware comparisons.
- **[Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)**: Classic compute/data trade-off framing.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)**: Foundational scaling-law perspective.
- **[Small Batch Size Training for Language Models](https://arxiv.org/abs/2502.04041)**: Supports batch-size-1 training choices used in the source paper.
- **[The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning](https://arxiv.org/abs/2505.22660)**: Discusses fine-tuning dynamics where “overfitting” signals can mislead.
- **[On the Impact of Fine-Tuning on Chain-of-Thought Reasoning](https://aclanthology.org/2025.naacl-long.584/)**: Studies how SFT changes CoT behavior; relevant to interpreting repetition-driven gains.
- **[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)**: Example of small-data multi-epoch post-training impacting behavior.
- **[DoReMi](https://arxiv.org/abs/2305.10429)**: Data mixture reweighting under compute constraints.
- **[D3: Large-Scale Data Selection for Instruction Tuning](https://arxiv.org/abs/2503.11441)**: Data selection for instruction tuning; relevant when repetition amplifies subset properties.
- **[RICo](https://arxiv.org/abs/2505.05327)**: Data selection heuristics for instruction tuning.
- **[SELECT2REASON](https://arxiv.org/abs/2505.17266)**: Selection for long-CoT instruction tuning (difficulty/length).
- **[DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)**: Modern reasoning pipelines that include multi-epoch SFT.
- **[Llama 3 Technical Report](https://arxiv.org/abs/2407.21783)**: Reports multi-epoch SFT in practice.
- **[OLMo 3 Technical Report](https://arxiv.org/abs/2512.13961)**: Describes the Dolci stack and post-training choices.
- **[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)**: Base models used in the source paper.
- **[8-bit Optimizers via Block-Wise Quantization](https://arxiv.org/abs/2110.02861)**: 8-bit Adam used by the source repo.
- **[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)**: Efficient long-generation evaluation used by the source repo.
- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)**: Establishes CoT as a behavior to elicit/teach.

### Taxonomy

| Cluster | Core idea | Representative papers | Common metrics | Risk for this proposal |
|---|---|---|---|---|
| Long-CoT SFT dynamics | Memorization/termination shape benchmark scores | Kopiczko et al. 2026; Lobo et al. 2025 | Acc@k, Pass@k, Termination | Compute confounds due to length variation |
| Data repetition in scaling | Repetition has diminishing-but-nonzero value | Muennighoff 2023; Chinchilla 2022 | Loss vs compute | Different stage (pretraining vs SFT) |
| Data selection | Which examples matter under limited compute | DoReMi; D3; SELECT2REASON | Downstream accuracy | Subset properties amplified by repetition |

### Closest Prior Work

**Kopiczko et al. (2026)** is the closest prior work. It matches update steps across epoch×sample trade-offs and reports strong improvements from high-epoch repetition, along with correlates (token accuracy, termination). Our contribution is a single missing control: **token-matching** to test whether step-matching hides a compute confound in long-CoT SFT.

**Novelty Kill Search Summary:** Searched for “token-matched repetition advantage”, “compute-matched epoch scaling long CoT SFT”, and “Data Repetition Beats Data Scaling FLOPs”. No prior follow-up implementing token/FLOPs matching for this specific phenomenon was found as of 2026-02-17.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Kopiczko et al. 2026 | Step-matched epoch×sample grid | Step-matching may not compute-match when lengths vary | Add token-matched control + FLOPs proxy | Decisive confound test; strengthens or corrects the headline claim |
| Muennighoff 2023 | Repetition theory in pretraining | Different training stage/objective | Apply compute-discipline lens to SFT claim | Clarifies whether SFT result is measurement artifact |

---

## Experiments

### Experimental Setup

- **Codebase**: **[dkopi/data-repetition](./references/GitHub---dkopi-data-repetition/meta/meta_info.txt)** (`train.py`, `eval.py`)
- **Base model**: `allenai/Olmo-3-1025-7B` (7B parameters)
- **Tokenizer**: `allenai/Olmo-3-7B-Instruct`
- **Training data**:
  - A: `dakopi/dolci_think__train_51200`
  - B,C: `dakopi/dolci_think__train_1600` (nested subset)
- **Training recipe**: BF16, 8-bit Adam, batch size 1, loss on response tokens only, 10% warmup + constant LR.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AIME 2024 | 30 competition math problems; integer answer 0–999 | Acc@16, Pass@16, Termination | test | via `dkopi/data-repetition` | `dkopi/data-repetition/eval.py` |
| AIME 2025 | same as above | Acc@16, Pass@16, Termination | test | via `dkopi/data-repetition` | `dkopi/data-repetition/eval.py` |
| GPQA | 448 graduate-level multiple-choice science questions | Acc@4, Pass@4, Termination | test | https://huggingface.co/datasets/Idavidrein/gpqa | `dkopi/data-repetition/eval.py` |

**Primary aggregate metric (lower variance):** weighted Pass@k across all 508 problems (AIME’24 + AIME’25 + GPQA), using k=16 for AIME and k=4 for GPQA.

**Metric definitions (for clarity):**
- **Acc@k (a.k.a. Acc@n in the source paper)**: mean correctness across *k* independent generations per problem.
- **Pass@k**: fraction of problems where at least one of *k* generations is correct.
- **Termination**: fraction of generations that yield a parseable final answer (e.g., an integer for AIME).

### Main Results

#### Results Table

| Method | Base Model | Budget type | Train dataset | Stop rule | Aggregate Pass@k (mean±std) | Token budget | FLOPs proxy Σ(L^2) | Source |
|---|---|---|---|---|---:|---:|---:|---|
| A: Data scaling | OLMo3-7B | step-matched | 51.2k×1 | 51,200 steps | TBD | TBD | TBD | This work |
| B: Repetition | OLMo3-7B | step-matched | 1.6k×32 | 51,200 steps | TBD | TBD | TBD | This work |
| **C: Token-matched repetition (ours)** | OLMo3-7B | token-matched | 1.6k×E | match A tokens (±0.5%) | TBD | =A | TBD | This work |

### Ablation Studies

No additional training ablations are required beyond A/B/C. We will report per-condition diagnostics: total response tokens processed, mean/quantiles of response length, and the Σ(L^2) FLOPs proxy.

### Experimental Rigor

- **Seeds**: `seeds=[42, 123, 456]` for each condition.
- **Pre-check (length stats)**: before training, compute mean/p50/p90/p99 response lengths for the 51.2k set and the 1.6k subset. If mean lengths differ by <5% and quantiles are similar, the compute confound is a priori less likely; we still run C but expect smaller changes.
- **Confound controls**:
  - Same base model, tokenizer, and evaluation prompts across conditions.
  - Condition C matches **token budget** to A and logs FLOPs proxy.
  - LR schedule fixed to avoid “short cosine” confounds.
- **Sanity check**: confirm the A vs B gap is directionally consistent with the source paper before interpreting C.

**Resource Estimate**:
- Source paper reports ~24h on 1×H100 per run; assume up to ~36h on 1×A100-80GB.
- 3 conditions × 3 seeds × 36h ≈ **324 A100 GPU-hours** for training.
- Evaluation via vLLM on 508 problems with multiple samples: budget **≤150 A100 GPU-hours** total.
- Total ≈ **≤474 GPU-hours**, within the 768 GPU-hour cap.

---

## Success Criteria

**Hypothesis**: Token matching removes a substantial fraction of the step-matched repetition advantage.

**Decision Rule**:
- **Proceed (confound confirmed)** if \(\Delta_{tok} \le 0.5\,\Delta_{step}\) on aggregate Pass@k (mean across ≥3 seeds).
- **Pivot (refine compute matching)** if token budgets match but the FLOPs proxy differs by >10%; in that case, re-run C with a tighter FLOPs proxy match (e.g., cap max response length during training).
- **Refute (confound small)** if \(\Delta_{tok} \ge 0.8\,\Delta_{step}\); conclude the repetition advantage is not primarily a token-compute artifact.

---

## Impact Statement

If token matching substantially shrinks the repetition advantage, it changes how practitioners should allocate post-training compute: the headline step-matched result would partly reflect a compute mismatch rather than a data-repetition benefit. If token matching does not shrink the gap, this provides stronger evidence that repetition is genuinely beneficial in long-CoT SFT and should be treated as a first-class design choice.

---

## References

All cited works are listed in **Related Papers**; key primary references are Kopiczko et al. (2026) and the accompanying `dkopi/data-repetition` codebase.
