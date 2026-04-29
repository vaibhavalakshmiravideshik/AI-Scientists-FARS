# untitled

# Compute-Matched Audit of Diffusion LMs’ Planning Advantage on Procedurally Generated Countdown and 4×4 Sudoku

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Diffusion language models (DLMs) generate text by iteratively denoising a partially masked sequence, rather than producing tokens left-to-right as autoregressive (AR) models do. This allows a DLM to revise arbitrary positions during generation and to condition on both left and right context. Recent open DLMs such as Dream-7B and LLaDA report competitive general language performance and unusually strong results on “planning” tasks.

A prominent example is **Dream 7B** (a 7B-parameter diffusion LM initialized from Qwen2.5-7B), which reports large gains over AR baselines on two verifiable planning benchmarks: **Countdown** (construct a target number using arithmetic operations on given numbers) and **4×4 Sudoku** (fill a small grid under row/column/subgrid constraints). In Dream’s Table 1, Dream-7B beats Qwen2.5-7B by **+9.8** points on Countdown (16.0 vs 6.2) and **+60** points on Sudoku (81.0 vs 21.0) under their matched evaluation protocol (Dream paper section “Superior planning performance…”).

If this planning advantage is robust under fair compute matching and contamination-resistant evaluation, diffusion backbones could be a better default for verifiable constraint satisfaction tasks. If it is not robust, then the practical conclusion may instead be that AR models with standard inference-time scaling (e.g., best-of-k sampling) can match diffusion performance given comparable inference budgets.

### The Problem

There are two reasons the community might overestimate a diffusion LM’s “planning advantage” from published benchmark tables:

1. **Inference compute mismatch**. A diffusion generation typically requires dozens to hundreds of full-model forward passes (one per denoising step). Many AR baselines in benchmark tables report only single-sample greedy decoding. If AR models are allowed to spend comparable inference compute (e.g., sample k candidates), their performance may increase substantially, especially on verifiable tasks where we can automatically select a valid solution.

2. **Evaluation contamination / dataset reuse risk**. Planning tasks like Game-of-24 have known contamination concerns: internet-scraped instance sets can inflate reported performance relative to procedurally generated “fresh” instances (**The Countdown Game**, arXiv:2508.02900). Even when a task is synthetic, evaluation protocols often reuse a fixed test file and a fixed few-shot prompt, making it hard to interpret whether a large performance gap reflects an inductive bias or protocol sensitivity.

Dream explicitly states that its Countdown and Sudoku evaluations come from **Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning** (arXiv:2410.14157) and its trip-planning evaluation comes from **Natural Plan** (arXiv:2406.04520) (Dream “Evaluation Tasks” section). Beyond Autoregression’s Countdown generator uses targets in [10, 100] and holds out 10% targets for **out-of-distribution (OOD)** evaluation, but the Dream repo’s planning eval script uses fixed JSONL test files and an 8-shot prompt (Dream repo `eval_planning.py`). These are reasonable choices, but they leave open a practical question:

> **Does Dream-7B’s diffusion-vs-AR planning gap persist on procedurally generated, contamination-resistant instances under compute-matched inference?**

### Key Insight and Hypothesis

**Key insight.** Countdown and small Sudoku are fully verifiable tasks: we can automatically check whether a candidate solution is correct. This makes them ideal for a compute-matched audit where AR baselines are allowed to use **best-of-k sampling** (sample k independent candidates and accept if any passes the verifier; k chosen to match diffusion wall-clock time), and evaluation uses procedurally generated instances to reduce the chance that results are dominated by quirks of a fixed benchmark file.

**Hypothesis.** The large reported Dream-vs-Qwen planning gap will **shrink substantially** when (i) evaluation uses freshly generated instances from a public procedural generator and (ii) Qwen2.5-7B is allowed compute-matched best-of-k sampling with verifier selection. If the gap remains large under these controls, it supports an inherent diffusion advantage for constraint satisfaction.

Why this could fail: (1) diffusion may genuinely provide a stronger inductive bias for these tasks; (2) AR best-of-k may not help much if the base distribution almost never includes valid solutions; (3) procedural generators may yield a different difficulty distribution than the original benchmarks, changing absolute accuracy.

---

## Proposed Approach

### Overview

We propose a **training-free evaluation protocol** to compare a diffusion LM vs an AR LM on the same planning task families under **wall-clock compute matching**.

We compare Dream-v0-Base-7B (diffusion) against Qwen2.5-7B (autoregressive) on two procedurally generated tasks with automatic verifiers:

- **Countdown** (number game) using the **Reasoning Gym** generator.
- **4×4 Sudoku** using the **Reasoning Gym mini_sudoku** generator (unique-solution puzzles).

We evaluate three conditions per task:

A) Qwen greedy (single sample)
B) Qwen best-of-k, where k is chosen to match Dream’s per-instance wall-clock time
C) Dream diffusion generation (single sample) using Dream’s recommended settings for planning tasks

### Method Details

#### Datasets (procedural generation)

We use **Reasoning Gym** (arXiv:2505.24760), a library of 100+ procedurally generated reasoning tasks with programmatic verifiers.

- **Mini Sudoku (4×4)**: `mini_sudoku` generates 4×4 puzzles with a unique solution (default empty-cell range 8–12). Verification is deterministic and checks exact correctness.
- **Countdown**: `countdown` generates a target number and a multiset of numbers; verification checks that the expression uses exactly the provided numbers and evaluates to the target.

We generate **n=500** test instances per task with a fixed RNG seed (per task) so results are reproducible.

We also generate an additional **calibration subset of 50 instances** per task (disjoint RNG seed) used only to estimate wall-clock time for compute matching.

#### Prompting

To minimize prompt confounds, we use a single **few-shot prompt template shared across Dream and Qwen** for each task family. Few-shot examples are generated from the same procedural generator but with a different RNG seed than the evaluation set.

- Sudoku prompt: 8-shot, following the format used in Dream’s `eval_planning.py` (instruction + input grid + “Output:” cue).
- Countdown prompt: 8-shot, but we adapt the output format to match the **verifier** we use:
  - For Reasoning Gym countdown, the output is a single arithmetic expression.

(We will include the exact prompt templates verbatim in the verification implementation.)

#### Model settings

- **Diffusion model**: `Dream-org/Dream-v0-Base-7B` (Apache-2.0). Use `diffusion_generate` with Dream repo defaults for planning tasks: `alg="entropy"`, `alg_temp=0`, `temperature=0`, `top_p=1`, and `steps = max_new_tokens` (as in Dream’s eval script).
- **AR model**: `Qwen2.5-7B` base (the initializer family used by Dream). Use standard HF generation.

#### Compute matching (choose k by wall-clock)

We compute k separately for each task:

1. Measure median wall-clock time per instance for Dream on the 50-instance calibration subset.
2. Measure median wall-clock time per instance for Qwen greedy on the same calibration subset.
3. Define `k = floor(median_time(Dream) / median_time(Qwen_greedy))`, with k clipped to [1, 64] for feasibility.

Then, for condition (B), we run **exactly k independent samples** from Qwen (no early stopping), and mark the instance solved if any sample passes the verifier.

This avoids an asymmetry where AR would get to stop early when it finds a solution while diffusion cannot.

### Key Innovations

1. **Compute-matched diffusion-vs-AR audit**: compare planning accuracy under a wall-clock matched inference budget rather than single-sample tables.
2. **Procedural contamination-resistant evaluation**: replace fixed test files with fresh procedural generation for two verifiable planning tasks.
3. **Decision-relevant negative result**: if diffusion’s advantage disappears under these controls, it changes how researchers interpret “planning advantage” claims in diffusion LMs.

---

## Related Work

### Field Overview

This proposal connects three literatures: (i) diffusion language models and their decoding policies, (ii) diffusion models for planning/constraint satisfaction, and (iii) contamination-resistant benchmark design.

Diffusion LMs (e.g., LLaDA, Dream) have inspired extensive inference-time research: remasking schedules, early stopping, caching, rollback, and inference-time scaling. In parallel, several papers argue diffusion has an inductive advantage on planning tasks such as Countdown and Sudoku, often attributing this to bidirectional context and iterative refinement. However, many comparisons are not compute-matched and use fixed benchmark files.

Separately, benchmark papers have shown that reasoning and planning evaluations can be strongly affected by dataset provenance. For example, The Countdown Game presents evidence that popular scraped Game-of-24 datasets are contaminated and that performance can drop sharply on procedurally generated instances.

### Related Papers

- **[Dream 7B: Diffusion Large Language Models](../../papers/paper_summaries/Dream 7B Diffusion Large Language Models.md)**: Introduces Dream-7B and reports large planning gains over AR baselines on Countdown, 4×4 Sudoku, and trip planning.
- **[Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning](../../papers/paper_summaries/Beyond Autoregression Discrete Diffusion for Complex Reasoning and Planning.md)**: Argues diffusion beats AR on Countdown/Sudoku; provides procedural generation and analysis.
- **[Natural Plan: Benchmarking LLMs on Natural Language Planning](../../papers/paper_summaries/Natural Plan Benchmarking LLMs on Natural Language Planning.md)**: Trip-planning benchmark used by Dream; evaluates exact-match of generated plans.
- **[Seemingly Simple Planning Problems are Computationally Challenging: The Countdown Game](../../papers/paper_summaries/Seemingly Simple Planning Problems are Computationally Challenging The Countdown Game.md)**: Shows contamination issues for scraped 24Game and proposes procedural generation for Countdown.
- **[Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../papers/paper_summaries/Reasoning Gym Reasoning Environments for Reinforcement Learning with Verifiable Rewards.md)**: Procedural generators + verifiers for 100+ reasoning tasks, including countdown and mini_sudoku.
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)**: Inference-time search for AR models; relevant best-of-k and verifier-based selection baselines.
- **[Stream of Search](https://arxiv.org/abs/2404.03683)**: Learning-to-search framing for reasoning tasks; cited by Beyond Autoregression for Countdown generation.
- **[Large Language Diffusion Models (LLaDA)](https://openreview.net/forum?id=KnqiC0znVF)**: Scales masked diffusion to 8B; widely used open diffusion baseline.
- **[Empirical Analysis of Decoding Biases in Masked Diffusion Models (UNCODE)](https://arxiv.org/abs/2508.13021)**: Shows decoding policy strongly affects dLLM reasoning performance.
- **[Prophet: Diffusion LMs Know the Answer Before Decoding](https://arxiv.org/abs/2508.19982)**: Early stopping for diffusion decoding; shows convergence behavior.
- **[Denoising Entropy / E-SMC](https://arxiv.org/abs/2512.21336)**: Uses uncertainty to guide diffusion inference-time scaling on planning tasks.
- **[DiFFPO](https://arxiv.org/abs/2510.02212)**: RL for diffusion LMs improving reasoning/planning with better compute–accuracy trade-offs.
- **[GDPO](https://arxiv.org/abs/2510.08554)**: Group diffusion policy optimization; reports planning improvements.
- **[Diffusion Beats Autoregressive in Data-Constrained Settings](https://arxiv.org/abs/2507.15857)**: Shows diffusion vs AR trade-offs depend on compute/data regime; motivates careful compute matching.
- **[Diffusion Language Models are Provably Optimal Parallel Samplers](https://arxiv.org/abs/2512.25014)**: Notes that diffusion may require more FLOPs per token despite parallelism; motivates wall-clock comparisons.
- **[Sudoku-Bench](../../papers/paper_summaries/Sudoku-Bench Evaluating creative reasoning with Sudoku variants.md)**: Sudoku-variant benchmark; we do not use it directly due to access uncertainty, but it motivates contamination-resistant Sudoku evaluation.
- **[ParallelBench](../../papers/paper_summaries/ParallelBench Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs.md)**: Highlights dependency-driven failures in parallel decoding.
- **[PUNT](https://arxiv.org/abs/2602.00286)**: Conditional independence tests for safer parallel unmasking; illustrates that inference compute can be traded for quality.
- **[EB-Sampler](https://arxiv.org/abs/2505.24857)**: Entropy-bounded unmasking; another example of inference compute vs quality tuning.
- **[Fast-dLLM v2](https://arxiv.org/abs/2509.26328)**: Efficient diffusion decoding; shows inference hyperparameters matter for quality/speed.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Scaled diffusion LMs | Train diffusion LMs at 7–8B scale | Dream, LLaDA | MMLU, GSM8K, HumanEval, Countdown, Sudoku | Inference requires many denoising steps; protocol-sensitive |
| Diffusion for planning | Use diffusion objectives to improve constraint satisfaction | Beyond Autoregression | Countdown, Sudoku, SAT | Often evaluated on fixed synthetic datasets |
| Inference-time scaling (AR) | Improve AR solving by sampling/search | Tree-of-Thoughts, Stream-of-Search | 24Game / Countdown-like | Expensive; often not compute-matched |
| Contamination-resistant eval | Procedural generation to avoid memorization | Countdown Game, Reasoning Gym | Countdown/CD tasks; many synthetic tasks | Difficulty distributions may differ across generators |

### Closest Prior Work

1. **Dream 7B**: Reports the planning advantage we audit. Our work differs by enforcing wall-clock compute matching and using procedural generation rather than fixed test JSONL files.
2. **Beyond Autoregression**: Shows diffusion beats AR on Countdown/Sudoku, but focuses on small custom models and training-time objectives; does not evaluate Dream vs its AR initializer under compute-matched inference.
3. **The Countdown Game**: Motivates contamination-resistant procedural generation, but does not compare diffusion vs AR or enforce compute matching.
4. **Reasoning Gym**: Provides procedural verifiable tasks, but is not a diffusion-vs-AR audit.

**Novelty Kill Search Summary:** Searched for combinations of “Dream 7B planning advantage compute matched”, “diffusion vs autoregressive Countdown best-of-k”, “wall-clock compute matching diffusion language model”, and “procedural generation countdown contamination diffusion”. No prior work was found that performs a wall-clock compute-matched Dream-vs-Qwen planning evaluation on freshly generated Countdown and 4×4 Sudoku instances as of 2026-02-28 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Dream 7B | Reports diffusion planning gains on fixed Countdown/Sudoku/Trip tasks | Not compute-matched vs AR inference scaling; fixed test files | Wall-clock compute matching + procedural generation | Directly tests whether the claimed gap is robust |
| Beyond Autoregression | Argues diffusion inductive bias helps planning | Uses custom small/medium models; not Dream vs AR-initializer | Evaluate Dream vs Qwen directly | Decision-relevant for model/benchmark researchers choosing backbones |
| Countdown Game | Shows contamination in 24Game and proposes procedural generation | No diffusion-vs-AR comparison | Use procedural generation for fair audit | Reduces contamination confound |
| Reasoning Gym | Library of procedural tasks and verifiers | Not a paradigm comparison | Use RG as evaluation substrate | Provides automation + reproducibility |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):**
1. **Prompting baseline**: Qwen2.5-7B greedy (single sample, temp=0).
2. **Inference-time scaling baseline**: Qwen2.5-7B best-of-k with verifier selection, where k is wall-clock matched to Dream.
3. **Diffusion baseline**: Dream-v0-Base-7B single diffusion generation with Dream’s planning settings.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Dream-org/Dream-v0-Base-7B | 7B | https://huggingface.co/Dream-org/Dream-v0-Base-7B | diffusion; `trust_remote_code=True`; uses `diffusion_generate` |
| Qwen2.5-7B (base) | 7B | https://huggingface.co/Qwen | autoregressive baseline family; use base (not instruct) for protocol alignment |

**Training Data (if applicable):**

No training data needed — inference-only evaluation.

**Other Resources (if applicable):**
- Reasoning Gym generator + verifiers: https://github.com/open-thought/reasoning-gym

**Resource Estimate**:
- **Compute budget**: inference-only. Expected <200 A100-hours for 2 tasks × 3 conditions × 500 instances, plus calibration timing. (Exact wall-clock depends on k from timing; cap k≤64.)
- **GPU memory**: both 7B models should fit on 1×A100-80GB in bf16.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Reasoning Gym mini_sudoku | 4×4 Sudoku puzzles with unique solutions | Accuracy (score_answer==1.0) | generated test (n=500) | https://github.com/open-thought/reasoning-gym | Reasoning Gym `mini_sudoku.score_answer` |
| Reasoning Gym countdown | Countdown numbers game | Accuracy (score_answer==1.0) | generated test (n=500) | https://github.com/open-thought/reasoning-gym | Reasoning Gym `countdown.score_answer` |

**Evaluation Scripts:**
- Use Reasoning Gym’s built-in verifiers (`score_answer`) for both tasks.
- For Dream generation API, use DreamLM/Dream inference wrapper.

### Main Results

All results are **TBD** (to be verified).

#### Comparability Rules (CRITICAL)
- Same procedural generator configuration and RNG seeds across methods.
- Same prompt templates across Dream and Qwen.
- Compute matching uses wall-clock time on the same GPU type.
- No early stopping in best-of-k (fixed k) to avoid giving AR an advantage not available to diffusion.

#### Results Table

| Method | Base Model | Benchmark | Accuracy (mean±std) | Wall-clock / instance (median) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Greedy | Qwen2.5-7B | RG-mini_sudoku | TBD | TBD | - | 1 sample |
| Best-of-k (compute-matched) | Qwen2.5-7B | RG-mini_sudoku | TBD | TBD | - | k from timing; verifier selection |
| Diffusion (single) | Dream-v0-Base-7B | RG-mini_sudoku | TBD | TBD | - | Dream defaults (entropy alg; temp=0) |
| Greedy | Qwen2.5-7B | RG-countdown | TBD | TBD | - | 1 sample |
| Best-of-k (compute-matched) | Qwen2.5-7B | RG-countdown | TBD | TBD | - | k from timing; verifier selection |
| Diffusion (single) | Dream-v0-Base-7B | RG-countdown | TBD | TBD | - | Dream defaults |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Different k estimator | use p75 instead of median time for k | robustness of compute matching |

### Experimental Rigor

- **Seeds**: For stochastic AR sampling (best-of-k), run 3 seeds for generation randomness: `seeds=[42,123,456]`. For greedy settings, determinism makes multiple seeds unnecessary.
- **Confidence intervals**: report bootstrap 95% CI over instances for pairwise accuracy differences (Dream − Qwen best-of-k).
- **Sanity checks**: random-string outputs should score ~0; verifier functions should return 1.0 on the oracle `answer` provided by the generator.

---

## Success Criteria

**Hypothesis:** Dream’s planning advantage over AR will shrink under compute-matched best-of-k and procedural evaluation.

**Decision Rule:**
- **Proceed (supports robust diffusion advantage)**: Dream beats compute-matched Qwen best-of-k by **≥ +5 accuracy points** on both tasks, and the bootstrap 95% CI for the difference excludes 0.
- **Refute / downweight the diffusion-advantage claim**: Dream’s advantage is **≤ +2 points** on both tasks (or negative), or confidence intervals include 0 with small absolute margins.
- **Pivot**: If both models are near-ceiling or near-zero, adjust generator difficulty (e.g., mini_sudoku empty-cell level) and re-run with the same three conditions.

---

## Impact Statement

If diffusion’s planning advantage persists under compute-matched, contamination-resistant evaluation, practitioners building verifiable planners (math puzzles, CSP-like tasks, program synthesis with unit tests) should consider diffusion backbones as a stronger default than AR models at the same parameter scale. If the advantage disappears, the community should treat large published planning gaps as protocol-sensitive, and AR best-of-k with verifier selection becomes a competitive, simpler baseline.

---

## References

- [Dream 7B: Diffusion Large Language Models](../../papers/paper_summaries/Dream 7B Diffusion Large Language Models.md) — Ye et al., 2025
- [Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning](../../papers/paper_summaries/Beyond Autoregression Discrete Diffusion for Complex Reasoning and Planning.md) — Ye et al., 2024
- [Natural Plan: Benchmarking LLMs on Natural Language Planning](../../papers/paper_summaries/Natural Plan Benchmarking LLMs on Natural Language Planning.md) — Zheng et al., 2024
- [Seemingly Simple Planning Problems are Computationally Challenging: The Countdown Game](../../papers/paper_summaries/Seemingly Simple Planning Problems are Computationally Challenging The Countdown Game.md) — Katz et al., 2025
- [Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../papers/paper_summaries/Reasoning Gym Reasoning Environments for Reinforcement Learning with Verifiable Rewards.md) — Stojanovski et al., 2025
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) — Yao et al., 2023
- [Stream of Search](https://arxiv.org/abs/2404.03683) — Gandhi et al., 2024
- [Large Language Diffusion Models (LLaDA)](https://openreview.net/forum?id=KnqiC0znVF) — Nie et al., 2025
- [Empirical Analysis of Decoding Biases in Masked Diffusion Models (UNCODE)](https://arxiv.org/abs/2508.13021) — 2025
- [Prophet: Diffusion LMs Know the Answer Before Decoding](https://arxiv.org/abs/2508.19982) — 2025
- [Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty](https://arxiv.org/abs/2512.21336) — 2025
- [DiFFPO](https://arxiv.org/abs/2510.02212) — 2025
- [Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization](https://arxiv.org/abs/2510.08554) — 2025
- [Diffusion Beats Autoregressive in Data-Constrained Settings](https://arxiv.org/abs/2507.15857) — 2025
- [Diffusion Language Models are Provably Optimal Parallel Samplers](https://arxiv.org/abs/2512.25014) — 2025
- [Sudoku-Bench: Evaluating creative reasoning with Sudoku variants](../../papers/paper_summaries/Sudoku-Bench Evaluating creative reasoning with Sudoku variants.md) — Seely et al., 2025
- [ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs](../../papers/paper_summaries/ParallelBench Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs.md) — 2025
- [Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing (PUNT)](https://arxiv.org/abs/2602.00286) — 2026
- [Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking (EB-Sampler)](https://arxiv.org/abs/2505.24857) — 2025
- [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://arxiv.org/abs/2509.26328) — 2025
