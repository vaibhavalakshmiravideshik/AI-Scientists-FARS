# untitled

# Execution-Signature Recycling: Deduplicating Unit-Test Failure Feedback for Test-Time Code Scaling

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated evaluation (unit tests only; no human/LLM-judge)
  - No web browsing / search APIs
  - Fit within **≤768 A100 GPU-hours**

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used for code generation. A common way to improve correctness is to spend more inference-time compute (“test-time scaling”), e.g., sample multiple candidate programs, run unit tests, and return a single selected program. We measure success with **Pass@1**: the fraction of problems where this final selected program passes all evaluation tests.

However, naive test-time scaling can waste substantial compute: many sampled programs fail for similar reasons (e.g., the same edge case), and independent rollouts repeatedly rediscover the same failures. This is especially plausible on robust benchmarks such as **EvalPlus** (Liu et al., 2023), which extends HumanEval/MBPP with many more unit tests (HumanEval+ / MBPP+) to reduce overfitting.

Recent work explores two relevant directions. First, **S\*** proposes a hybrid test-time scaling procedure for code generation that combines parallel sampling, iterative self-debugging using execution logs, and execution-grounded selection with clustering by execution behavior (**[S\*: Test Time Scaling for Code Generation](./references/S-Test-Time-Scaling-for-Code-Generation/meta/meta_info.txt)**). Second, **Recycling Search Experience (RSE)** proposes batched “experience banks” for test-time scaling in mathematical reasoning, with semantic deduplication to avoid redundant information (**[Do Not Waste Your Rollouts: Recycling Search Experience for Efficient Test-Time Scaling](./references/Do-Not-Waste-Your-Rollouts-Recycling-Search-Experience-for-Efficient-Test-Time-Scaling/meta/meta_info.txt)**).

### The Problem

Despite progress, it remains unclear how to **share execution feedback across rollouts** in a way that (i) is fully automated and (ii) does not simply “stuff more logs into the prompt.”

- **Per-sample self-debugging** (e.g., “self-debug” variants in **[S\*](./references/S-Test-Time-Scaling-for-Code-Generation/meta/meta_info.txt)** and related work) uses each candidate’s own traceback to improve that candidate, but does not explicitly reuse information across different failure modes.
- **Experience banks** (e.g., **[RSE](./references/Do-Not-Waste-Your-Rollouts-Recycling-Search-Experience-for-Efficient-Test-Time-Scaling/meta/meta_info.txt)**) show that sharing distilled experience can improve search efficiency, but current evidence is primarily on math reasoning where “verification” is not an executable unit-test harness.

For code, unit tests provide a strong programmatic signal, but there is a missing piece: we need a **grounded, low-overhead representation** of what the model already “tried and failed,” and a way to deduplicate that information so the shared feedback is informative rather than verbose.

### Key Insight and Hypothesis

**Key insight.** In code generation with unit tests, each rollout induces a structured “execution outcome” that can be summarized without any extra model calls: which tests failed and what error types occurred. We call this an **execution signature**. If multiple rollouts share the same execution signature, then spending more compute on that failure mode is likely redundant unless we explicitly condition future rollouts on avoiding it.

**Hypothesis.** Under a fixed generation budget, a two-round test-time scaling procedure that (i) clusters first-round candidates by execution signature and (ii) conditions second-round generation on a **deduplicated, frequency-weighted bank** of the most common failing signatures will improve **EvalPlus Pass@1** (HumanEval+) over compute-matched baselines (best-of-N sampling and per-sample self-debugging), because it reduces repeated exploration of the same failure modes while keeping the shared feedback short.

**Why this could be wrong.** (i) Failure signatures may already be highly diverse in best-of-N sampling, so deduplication has little to compress. (ii) Execution signatures may be too coarse: different root causes can share the same failing-test set, so conditioning on the signature provides weak guidance. (iii) Conditioning all second-round rollouts on shared failures could cause mode collapse, reducing diversity and hurting best-of-N.

---

## Proposed Approach

### Overview

We propose **Execution-Signature Recycling (ESR)**, a training-free test-time scaling method for code generation.

For each programming problem, ESR uses a fixed compute budget of 16 generations split into two rounds:
1. **Round 1 (explore)**: sample 8 candidate programs and run the *base* unit tests.
2. **Aggregate + deduplicate**: cluster candidates by execution signature; build a compact “failure bank” containing only the most frequent signatures (and representative failing examples), within a fixed prompt token budget.
3. **Round 2 (exploit with shared feedback)**: sample 8 new candidate programs from scratch, conditioned on the failure bank.
4. **Selection**: select the candidate that maximizes base-test score, then evaluate that selected program on **EvalPlus** (HumanEval+).

### Method Details

#### Benchmarks and test splits
We use EvalPlus (**HumanEval+**) as the primary benchmark.
- **Feedback/selection tests**: HumanEval base tests (the original tests) are used to define execution signatures, provide tracebacks, and select the best candidate.
- **Evaluation tests**: HumanEval+ (EvalPlus) augmented tests are used for final Pass@1 evaluation of the selected candidate.

This is intended to reduce “overfitting to the base tests” while still allowing unit-test feedback at test time.

#### Execution signature
For a candidate program y and a set of base tests \(\{t_j\}_{j=1}^m\), we run the tests and record, for each test:
- PASS, or
- FAIL with an error type (e.g., AssertionError, TypeError, NameError, Timeout).

Define an **execution signature** as:
\[
\sigma(y) = \{(j, \mathrm{errtype}_j) : t_j \text{ fails on } y\}.
\]
(We ignore passing tests in the signature to keep it compact.)

#### Failure bank construction (deterministic; no extra LLM calls)
Given first-round candidates \(\{y_i\}_{i=1}^8\), cluster them by exact signature equality. For each cluster c with signature \(\sigma_c\) and size \(|c|\), create a bank entry:
- count \(|c|\)
- list of failing test IDs j
- for up to K failing tests (K=2 by default): the failing input, expected output, and observed output or exception traceback snippet (from the harness)

We then select the top-M clusters by \(|c|\) (M=3 by default) and serialize them into a failure bank \(B\) with a maximum prompt budget (e.g., 600–900 tokens).

#### Round-2 conditioned generation (from scratch)
We prompt the model with:
- the original problem prompt
- the failure bank B
- an instruction: “The following failing cases were common among previous attempts. Write a correct solution that passes these failing cases and the general specification. Do not copy any prior buggy code.”

Then we sample 8 new programs (same decoding hyperparameters as baselines).

#### Required mechanism control (dedup vs no dedup)
To rule out the trivial explanation “seeing more feedback helps,” we include an ablation that conditions round-2 generation on **no-dedup feedback**:
- serialize the raw failure feedback from all 8 first-round candidates into the prompt,
- truncate or subsample to match ESR’s prompt token budget.

This control keeps the amount of feedback roughly constant, but removes signature clustering and frequency weighting.

#### Early-kill diagnostic (premise check)
Before running the full study, we compute the number of unique signatures among the 8 first-round samples.
- If the median unique signature count is >6/8 (across a small pilot subset), signature repetition is rare and ESR is unlikely to help; we stop and report a negative result.

### Key Innovations

- **Execution signatures as grounded feedback units**: represent failure modes using test outcome structure (failing-test IDs + error types) rather than free-form summaries.
- **Deduplicated, frequency-weighted feedback aggregation**: uses clustering to decide what feedback to propagate to the next generation batch (distinct from prior work that uses clustering mainly for candidate selection).
- **Mechanism-isolating control**: a token-budget-matched no-dedup feedback baseline to test whether gains come from deduplication rather than “more context.”

---

## Related Work

### Field Overview

**Test-time scaling for LLMs.** A broad literature shows that allocating more inference-time compute (e.g., best-of-N sampling, self-consistency, tree search, or multi-round refinement) can improve reasoning and code generation performance, but naive sampling can be inefficient when rollouts repeatedly explore similar trajectories.

**Execution feedback for code generation.** Code is unusual compared to open-ended reasoning because it admits automated feedback: unit tests, compilers, and interpreters provide objective signals. Many methods exploit this via iterative self-debugging, test-driven generation, or selection strategies based on test outcomes. A recurring issue is that execution feedback can be verbose and redundant across many candidates.

**Experience reuse / memory.** Several agent and test-time scaling methods propose maintaining memory across attempts, either within a query (multi-round search) or across queries (experience banks). However, for code generation, many “memory” schemes rely on unstructured natural language reflections, which can be noisy and hard to compress.

### Related Papers

- **[S\*: Test Time Scaling for Code Generation](./references/S-Test-Time-Scaling-for-Code-Generation/meta/meta_info.txt)**: Hybrid test-time scaling for code using sampling + self-debugging + execution-grounded selection with execution-behavior clustering.
- **[Do Not Waste Your Rollouts: Recycling Search Experience for Efficient Test-Time Scaling](./references/Do-Not-Waste-Your-Rollouts-Recycling-Search-Experience-for-Efficient-Test-Time-Scaling/meta/meta_info.txt)**: Batched experience banks + semantic deduplication for math reasoning test-time scaling.
- **[Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic](./references/Timely-Machine-Awareness-of-Time-Makes-Test-Time-Scaling-Agentic/meta/meta_info.txt)**: Redefines test-time scaling as wall-clock time and trains time-budgeted strategies.
- **[TTCS: Test-Time Curriculum Synthesis for Self-Evolving](./references/TTCS-Test-Time-Curriculum-Synthesis-for-Self-Evolving/meta/meta_info.txt)**: Test-time training with a synthesizer that generates capability-matched curricula.
- **[Qwen2.5-Coder Technical Report](./references/Qwen2.5-Coder-Technical-Report/meta/meta_info.txt)**: Reports EvalPlus results for Qwen2.5-Coder models (e.g., Qwen2.5-Coder-7B-Instruct HE+ 84.1, MBPP+ 71.7; Table 13); provides baseline context.
- **[EvalPlus](https://arxiv.org/abs/2305.01210)**: Extends HumanEval/MBPP with more thorough tests to reduce overfitting.
- **[HumanEval](https://arxiv.org/abs/2107.03374)**: Standard code generation benchmark of 164 Python tasks.
- **[MBPP](https://arxiv.org/abs/2108.07732)**: Python programming benchmark with natural language prompts and unit tests.
- **[Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128)**: Iterative debugging using execution feedback.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Agents that learn from failures using verbal reflection and episodic memory.
- **[Self-Refine](https://arxiv.org/abs/2303.17651)**: Iterative refinement using self-generated feedback.
- **[CodeT](https://arxiv.org/abs/2207.10397)**: Code generation with self-generated tests for candidate selection.
- **[CodeRL](https://arxiv.org/abs/2207.01780)**: Reinforcement learning from execution feedback for code generation.
- **[RLTF: Reinforcement Learning from Unit Test Feedback](https://openreview.net/forum?id=hjYmsV6nXZ)**: Uses multi-granularity unit-test feedback for RL training of code models.
- **[CYCLE: Learning to Self-Refine the Code Generation](https://arxiv.org/abs/2403.18746)**: Trains models for iterative refinement using execution feedback.
- **[AlphaCode](https://arxiv.org/abs/2203.07814)**: Competition-level code generation using large-scale sampling and clustering.
- **[LiveCodeBench](https://arxiv.org/abs/2403.07974)**: Contamination-resistant evaluation for code generation.
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)**: Search over intermediate reasoning states for LLM problem solving.
- **[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)**: Shows that inference compute scaling can be competitive with parameter scaling in reasoning.
- **[Large Language Monkeys](https://arxiv.org/abs/2407.21787)**: Repeated sampling for scaling inference compute.
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)**: Improves reasoning by sampling multiple chains and aggregating.
- **[CoverUp: Coverage-Guided LLM-Based Test Generation](https://arxiv.org/abs/2403.16218)**: Uses coverage signals to guide LLM unit test generation.
- **[UTGen: Learning to Generate Unit Tests for Automated Debugging](https://openreview.net/forum?id=yeVBHPLXxi)**: Generates multiple tests to guide automated debugging.

(Additional related papers may be added during revision to exceed the 20-paper minimum if any citation above is not accessible.)

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Parallel sampling + selection | Sample many solutions; select best by tests or heuristics | AlphaCode; CodeT; best-of-N | HumanEval/MBPP; LiveCodeBench | Wasteful redundancy across samples; selection limited by test strength |
| Sequential self-debug / refinement | Use execution logs to iteratively repair solutions | Self-Debug; Reflexion; Self-Refine; CYCLE | HumanEval/MBPP; APPS | Can be compute-inefficient; improvements depend on feedback quality |
| Hybrid parallel+sequential code TTS | Combine breadth (sampling) with per-candidate debugging and selection | S\* | LiveCodeBench v2; CodeContests | Many components; unclear what mechanism drives efficiency |
| Experience banks / deduplication (mostly non-code) | Distill and reuse experience across attempts to reduce redundancy | RSE; PaCoRe; ReasoningBank | Math reasoning benchmarks; agent tasks | Distillation often ungrounded; can cause mode collapse |
| Coverage-/analysis-guided testing | Use coverage or program analysis to generate stronger tests/feedback | EvalPlus; CoverUp; UTGen | Test generation / debugging benchmarks | Requires tooling; can be expensive |

### Closest Prior Work

1) **S\*** (**[S\*](./references/S-Test-Time-Scaling-for-Code-Generation/meta/meta_info.txt)**): Clusters candidates by execution behavior mainly to improve **selection** and combines this with per-candidate self-debugging; ESR instead uses execution signatures primarily as a **feedback compression and propagation** mechanism across rollouts.

2) **RSE** (**[RSE](./references/Do-Not-Waste-Your-Rollouts-Recycling-Search-Experience-for-Efficient-Test-Time-Scaling/meta/meta_info.txt)**): Proposes batched experience banks and semantic deduplication for math reasoning; ESR instantiates a similar “batched sharing” idea but replaces semantic distillation with deterministic, unit-test-grounded execution signatures.

3) **Self-Debugging** (**[Self-Debug](https://arxiv.org/abs/2304.05128)**): Uses execution feedback to refine each candidate; ESR tests whether deduplicated cross-candidate feedback is more compute-efficient than per-candidate refinement.

4) **CodeT / test-based selection** (**[CodeT](https://arxiv.org/abs/2207.10397)**): Uses generated tests to rank solutions; ESR uses given unit tests (base tests) to create a reusable failure representation across rollouts.

**Novelty Kill Search Summary:** Searched for combinations of “recycling search experience + code generation”, “experience bank + unit tests + best-of”, “execution signature deduplication for code generation”, “cluster failing tests prompt code generation”, and checked local proposal corpus for “RSE / Timely Machine / TTCS / Recycling Search Experience”. No prior work matching “RSE-style batched experience bank keyed by execution signatures for test-time code scaling” was found as of 2026-02-19 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| S\* | Hybrid code TTS with debugging + execution-based selection | Clustering used mainly for selection; feedback sharing not isolated | Use clustering to compress and broadcast failure feedback | Reduces redundant failure rediscovery under a fixed budget |
| RSE | Experience bank + semantic dedup for math | Experiences not grounded in programmatic tests | Use unit-test outcome signatures as experience keys | Grounded, deterministic dedup; avoids noisy summaries |
| Self-debug | Per-sample repair from traceback | No cross-sample reuse; may repeat similar failures | Aggregate repeated failure signatures across samples | Reuse information where repetition is high |
| Best-of-N | Pure sampling | Wasteful redundancy; no feedback | Add compact feedback from repeated failures | Should spend compute on new failure modes |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):**
- Prompting baseline: single-sample generation (N=1) for Qwen2.5-Coder-7B-Instruct (reference).
- Inference-time scaling baseline: Best-of-16 sampling with test-based selection.
- Closest existing method family: Per-sample self-debugging with execution feedback (S\*-style stage-1).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-Coder-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct | Strong open code model; Qwen TR reports HE+ 84.1 and MBPP+ 71.7 (Table 13) in EvalPlus evaluation ([Qwen2.5-Coder TR](./references/Qwen2.5-Coder-Technical-Report/meta/meta_info.txt)). |

**Training Data (if applicable):**

No training data needed – inference only.

**Other Resources (if applicable):**
- EvalPlus evaluation harness: https://github.com/evalplus/evalplus

**Resource Estimate**:
- **Compute budget**: ≤200 A100 GPU-hours total (4 runs per task family: best-of-16, self-debug, ESR, ESR w/o dedup; 3 sampling seeds; main cost is generation; unit tests on CPU)
- **GPU memory**: 1×A100 80GB is sufficient for 7B inference; use 2–8 GPUs for throughput
- **API usage**: none

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| HumanEval+ (EvalPlus) | 164 Python function synthesis tasks evaluated with extended unit tests | Pass@1 (higher is better) | test | https://github.com/evalplus/evalplus | EvalPlus official scripts |

**Evaluation Scripts:**
- Use EvalPlus official harness. Use HumanEval base tests for feedback/selection; evaluate the selected candidate on HumanEval+ tests.

### Main Results

#### Comparability Rules (CRITICAL)
All rows will use:
- Same base model: Qwen2.5-Coder-7B-Instruct
- Same generation budget: 16 generations per problem
- Same selection: maximize base-test score; tie-break by fewer failing tests
- Same evaluation metric: HumanEval+ Pass@1

#### Results Table

| Method | Base Model | Benchmark | Pass@1 (mean±std) | Source | Notes |
|---|---|---|---:|---|---|
| Best-of-16 sampling | Qwen2.5-Coder-7B-Instruct | HumanEval+ | **TBD** | This work | Inference scaling baseline |
| 8× self-debug (1 step) | Qwen2.5-Coder-7B-Instruct | HumanEval+ | **TBD** | This work | Closest execution-feedback baseline |
| **Ours: ESR (deduped failure bank)** | Qwen2.5-Coder-7B-Instruct | HumanEval+ | **TBD** | This work | 8 explore + dedup bank + 8 conditioned |

#### Hard-subset analysis (where there is headroom)
Because overall HumanEval+ may be close to saturation for strong models, we will additionally report results on a **hard subset** \(H\) to better stress-test the mechanism:

- Define \(H\) as problems where the **N=1** prompting baseline fails on HumanEval+ (selected program fails), but **oracle Pass@16** for Best-of-16 is 1 (i.e., at least one of the 16 generated programs passes all HumanEval+ tests).
- Report Pass@1 on \(H\) for each method, plus signature repetition statistics (e.g., number of unique signatures in round-1) on \(H\).

**Prediction.** If ESR’s benefit comes from reducing redundant failure rediscovery, its gains should concentrate on problems in \(H\) with repeated signatures (few unique \(\sigma\) among the first 8 samples).

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| ESR (full) | Cluster by execution signature; keep top-M signatures by frequency | Best performance |
| ESR w/o dedup (required control) | Condition round-2 on token-budget-matched raw failure feedback (no clustering/frequency) | If ESR > w/o-dedup, dedup/clustering is doing real work |
| Pilot premise check | Measure unique signatures among first 8 samples | If unique signatures ≈8, ESR is unlikely to help and study should terminate early |

### Experimental Rigor

**Variance & Reproducibility:**
- Use ≥3 sampling seeds (e.g., seeds=[42, 123, 456]) for each method; report mean±std.
- Use paired bootstrap over tasks to compute 95% CI on Pass@1 differences.

**Validity & Controls:**
- **More-context confound**: ESR w/o dedup uses the same token budget as ESR, to isolate deduplication vs “more feedback.”
- **Extra-compute confound**: ESR bank is constructed deterministically from the test harness outputs; no extra model calls.
- **Prompt-length reporting**: log token counts for the round-2 conditioning context in self-debug vs ESR vs ESR w/o dedup.
- **Test overfitting**: guidance/selection uses base tests; evaluation uses EvalPlus extended tests.

---

## Success Criteria

**Hypothesis** (directional):
- ESR improves HumanEval+ Pass@1 over compute-matched best-of-16 and per-sample self-debugging.
- If improvements occur, they are attributable to deduplication (ESR > ESR w/o dedup).

**Decision Rule** (concrete):
- **Proceed**: ESR beats the best baseline (max of best-of-16 and self-debug) by a positive margin with 95% paired bootstrap CI excluding 0, and ESR > ESR w/o dedup.
- **Pivot**: If ESR > best baseline but ESR ≈ ESR w/o dedup, the gain is likely “any shared feedback helps”; pivot to studying prompt-length/feedback-design effects rather than clustering.
- **Refute**: If ESR ≤ best baseline (CI overlaps 0) OR if the early-kill diagnostic shows unique signatures >6/8 in the pilot, conclude that failure-mode repetition is not a major bottleneck for best-of-N on this benchmark/model.

---

## Impact Statement

If successful, ESR provides a simple, training-free way to make test-time scaling for code generation more compute-efficient by deduplicating execution feedback across rollouts, potentially improving the effectiveness of best-of-N sampling under fixed budgets. If it fails (especially via the early-kill signature-diversity diagnostic), the negative result is still decision-relevant: it would suggest that on strong code models and EvalPlus-style tests, redundancy across sampled failures is not large enough to justify experience-bank style feedback sharing.

---

## References

- [Do Not Waste Your Rollouts: Recycling Search Experience for Efficient Test-Time Scaling](./references/Do-Not-Waste-Your-Rollouts-Recycling-Search-Experience-for-Efficient-Test-Time-Scaling/meta/meta_info.txt) - Wang et al., 2026
- [Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic](./references/Timely-Machine-Awareness-of-Time-Makes-Test-Time-Scaling-Agentic/meta/meta_info.txt) - Ma et al., 2026
- [TTCS: Test-Time Curriculum Synthesis for Self-Evolving](./references/TTCS-Test-Time-Curriculum-Synthesis-for-Self-Evolving/meta/meta_info.txt) - Yang et al., 2026
- [S\*: Test Time Scaling for Code Generation](./references/S-Test-Time-Scaling-for-Code-Generation/meta/meta_info.txt) - Li et al., 2025
- [Qwen2.5-Coder Technical Report](./references/Qwen2.5-Coder-Technical-Report/meta/meta_info.txt) - Hui et al., 2024
- [EvalPlus: Rigorous Evaluation of Large Language Models for Code Generation](https://arxiv.org/abs/2305.01210) - Liu et al., 2023
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) - Chen et al., 2021 (HumanEval)
- [MBPP: Program Synthesis with Natural Language](https://arxiv.org/abs/2108.07732) - Austin et al., 2021
- [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128) - Chen et al., 2023
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - Shinn et al., 2023
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) - Madaan et al., 2023
- [CodeT: Code Generation with Generated Tests](https://arxiv.org/abs/2207.10397) - Chen et al., 2022
- [CodeRL: Mastering Code Generation through Pretrained Models and Reinforcement Learning](https://arxiv.org/abs/2207.01780) - Le et al., 2022
- [AlphaCode: Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814) - Li et al., 2022
- [LiveCodeBench: Holistic and Contamination-Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974) - Jain et al., 2024
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) - Yao et al., 2023
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [CoverUp: Coverage-Guided LLM-Based Test Generation](https://arxiv.org/abs/2403.16218) - (authors), 2024
- [UTGen: Learning to Generate Unit Tests for Automated Debugging](https://openreview.net/forum?id=yeVBHPLXxi) - (authors), 2025
- [RLTF: Reinforcement Learning from Unit Test Feedback](https://openreview.net/forum?id=hjYmsV6nXZ) - (authors), 2023
