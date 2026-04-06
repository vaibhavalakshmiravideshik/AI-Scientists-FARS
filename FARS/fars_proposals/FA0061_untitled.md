# untitled

# Entropy-Instability Ranking for Execution-Free Best-of-N Code Selection

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated evaluation (unit-test execution only; no human judgments)
  - Inference-only (no training required)
  - Must fit within ≤768 A100-80GB GPU-hours total

## Introduction

### Context and Motivation

Best-of-N sampling is a common way to improve code generation with large language models (LLMs): for each programming problem, sample multiple candidate programs, then select a single candidate to present or to run in downstream pipelines. When a candidate can be executed against unit tests, selection can be made by running all candidates and picking the best-performing one. However, running unit tests for many candidates can be expensive (often CPU-bound) and can become a latency bottleneck in interactive systems.

This motivates **execution-free** selection rules: rank sampled programs using signals available during generation (e.g., token probabilities) and only execute a small number of finalists, or execute only the selected candidate. Prior work has proposed aggregate confidence scores (e.g., mean log probability, mean entropy) and distributional confidence (e.g., KL-to-uniform “self-certainty”), as well as more expensive execution-free alternatives such as multi-sample similarity or learned rerankers.

A recent line of work suggests that **how uncertainty evolves during generation** may be more informative than aggregate uncertainty. In particular, EDIS (Entropy Dynamics Instability Score) identifies characteristic entropy-trajectory instability patterns (burst spikes and rebound spikes) that correlate strongly with incorrect mathematical reasoning and enable strong best-of-N selection without external verifiers.

### The Problem

It is unknown whether entropy-trajectory instability is a useful execution-free signal for **code generation correctness**. Code generation differs from math reasoning in several ways: outputs are longer, have strict syntax constraints, and can fail due to localized bugs (e.g., variable misuse) rather than globally inconsistent reasoning. This creates two practical obstacles to directly porting EDIS:

1. **Length confounding**: longer outputs mechanically have more opportunities for spikes, so a spike-count score may behave like a length proxy.
2. **Dispersion vs. shape**: a complex instability score can collapse to “entropy variance/dispersion”, without contributing additional signal from the temporal spike structure.

If entropy instability reduces to length or generic dispersion, it is not a new selection signal, and practitioners should prefer simpler rankers. Conversely, if spike-structure adds signal beyond length and dispersion, it provides a cheap, verifier-light selector for code generation.

### Key Insight and Hypothesis

**Key insight.** For incorrect code, uncertainty often rises and falls around specific decision points (API calls, boundary conditions, loop invariants). This can create detectable *trajectory-shape* events beyond entropy level: (i) **burst spikes** (sustained entropy increases over several tokens, indicating progressive confusion) and (ii) **rebound spikes** (entropy rises sharply from a previous low-entropy “valley”, indicating false confidence followed by renewed uncertainty). These patterns may not be captured by length alone or by overall dispersion of entropy values.

**Hypothesis.** A length-normalized, scale-normalized version of EDIS (nEDIS) will select more correct programs (higher pass@1) than both:
- a **length-only** ranker (which captures the most obvious confound), and
- an **entropy-dispersion** ranker based on the coefficient of variation (CV) of token entropies.

This could fail if (i) code correctness is primarily determined by local syntax/semantics not reflected in uncertainty trajectories, (ii) spike-rate is highly correlated with entropy dispersion so nEDIS is effectively a nonlinear transform of dispersion, or (iii) the fixed spike thresholds do not transfer across domains.

---

## Proposed Approach

### Overview

We propose to test whether **entropy-trajectory spike structure** provides an execution-free ranking signal for best-of-N code selection.

Given a sampled candidate program, we compute the token-level entropy trajectory during generation and derive three execution-free scores:
1. **Length-only** (baseline confound): number of generated tokens.
2. **Dispersion-only** (baseline): coefficient of variation of token entropies.
3. **nEDIS (ours)**: a spike-rate score multiplied by a normalized dispersion term.

All three methods use the *same* candidate set per problem; only the ranking function differs.

### Method Details

**Token entropy.** For a prompt x and generated sequence y=(y1,…,yT), let the model’s next-token distribution be πθ(v|x,y< t). Token entropy at step t is:

H_t = − \sum_{v\in V} π_θ(v|x,y_{<t}) \log π_θ(v|x,y_{<t}).

(We compute entropy on the full vocabulary distribution; no semantic clustering or execution is used.)

**Baselines.**
- **Length-only score**: L(y)=T. Lower is ranked as “more reliable”.
- **Entropy dispersion score**: CV_H(y)=std(H)/(mean(H)+ε), with ε=1e−6, where the coefficient of variation (CV) is the ratio of standard deviation to mean. Lower is ranked as “more reliable”.

**Spike-rate features (adapted from EDIS).** Let σ_H=std(H)+ε. Use window size w=4 and fixed thresholds τ_b=1.36 and τ_r=1.33 (as in EDIS).

- **Burst spikes** count sustained rises over a window:
  - S_burst = \sum_{t=1}^{T−w} 𝟙\[(H_{t+w}−H_t)/σ_H > τ_b\].
- **Rebound spikes** count rises from the running minimum:
  - S_rebound = \sum_{t=2}^{T} 𝟙\[(H_t−min_{s<t} H_s)/σ_H > τ_r\].

Define spike rate:

s(y)= 0.5 (S_burst+S_rebound) / T.

**nEDIS score.**

nEDIS(y) = s(y) · (1 + CV_H(y)^2).

Intuitively, s(y) captures the *frequency of instability events* (trajectory shape), while CV_H(y) captures *overall dispersion* of uncertainty; the multiplicative form emphasizes candidates that are both spiky and globally unstable.

Lower nEDIS is ranked as “more reliable”.

**Why these normalizations.** Dividing by T makes the score a rate (reducing length confounding). Standardizing spike magnitudes by σ_H reduces scale sensitivity across domains/models. Using CV_H rather than raw variance reduces sensitivity to entropy level.

### Key Innovations

- **A falsifiable transfer test**: Evaluate whether EDIS-style *trajectory instability* is useful for code selection, where correctness is objectively measurable by unit tests.
- **Confound-targeted design**: Pre-register length-only and dispersion-only rankers as the primary baselines, so any apparent gain must exceed the most plausible “boring explanations”.
- **Scale/length-normalized instability score (nEDIS)**: A minimal adaptation of EDIS intended to make the hypothesis testable without hyperparameter tuning on target labels.

---

## Related Work

### Field Overview

Execution-free confidence estimation for code generation can be grouped into (i) **aggregate likelihood/uncertainty** measures derived from token probabilities, (ii) **multi-sample agreement** measures (similarity or behavioral divergence), (iii) **learned rerankers / reward models** trained to predict correctness, and (iv) **uncertainty-guided decoding** methods that allocate inference compute adaptively (e.g., branching or selective CoT). Our proposal falls in category (i), but tests whether moving from aggregate statistics to **trajectory-shape features** yields a stronger ranker.

Entropy has also been used as a *control signal* for adaptive inference-time scaling, but these methods typically use instantaneous entropy thresholds to decide when to branch or to invoke more compute. In contrast, we study **post-hoc ranking** of completed candidates using their full entropy trajectories.

### Related Papers

- **[EDIS: Diagnosing LLM Reasoning via Entropy Dynamics](./references/EDIS-Diagnosing-LLM-Reasoning-via-Entropy-Dynamics/meta/meta_info.txt)**: Introduces burst/rebound entropy spikes for best-of-N selection on math reasoning; does not evaluate code.
- **[Scalable Best-of-N Selection for Large Language Models via Self-Certainty](./references/Scalable-Best-of-N-Selection-for-Large-Language-Models-via-Self-Certainty/meta/meta_info.txt)**: Uses KL-to-uniform as a distributional confidence score for best-of-N and evaluates code benchmarks (e.g., LiveCodeBench).
- **[Top Pass: improve code generation by pass@k-maximized code ranking](./references/Top-Pass-Improve-Code-Generation-by-Pass@k-Maximized-Code-Ranking/meta/meta_info.txt)**: Trains a learned reranker to optimize pass@k from candidate lists (training required).
- **[Incoherence as Oracle-less Measure of Error in LLM-Based Code Generation](./references/Estimating-Correctness-Without-Oracles-in-LLM-Based-Code-Generation/meta/meta_info.txt)**: Uses behavioral divergence between sampled programs as an oracle-free lower bound on error (requires generated tests / execution).
- **[Showing LLM-Generated Code Selectively Based on Confidence of LLMs](./references/Showing-LLM-Generated-Code-Selectively-Based-on-Confidence-of-LLMs/meta/meta_info.txt)**: Estimates confidence via multi-modal similarity across sampled programs for selective display (requires parsing/embeddings).
- **[Uncertainty-Guided Chain-of-Thought for Code Generation with LLMs](./references/Uncertainty-Guided-Chain-of-Thought-for-Code-Generation-with-LLMs/meta/meta_info.txt)**: Uses entropy/prob-diff at line starts to trigger CoT decoding; not a completed-candidate trajectory ranker.
- **[EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling](./references/EAGER-Entropy-Aware-GEneRation-for-Adaptive-Inference-Time-Scaling/meta/meta_info.txt)**: Uses token-level entropy peaks to branch and reallocate compute for test-time scaling (branching, not post-hoc ranking).
- **[Multicalibration for LLM-based Code Generation](https://arxiv.org/abs/2512.08810)**: Post-hoc group calibration for token-likelihood confidence in code generation.
- **[Localized Calibrated Uncertainty in Code Language Models](https://arxiv.org/abs/2512.24560)**: Learns probes for token/line-level calibrated uncertainty, including on HumanEval+/MBPP+.
- **[Calibration and Correctness of Language Models for Code](https://arxiv.org/abs/2402.02047)**: Studies how intrinsic probability-based confidence scores align with unit-test correctness for code generation.
- **[Measuring LLM Code Generation Stability via Structural Entropy](https://arxiv.org/abs/2508.14288)**: Proposes AST-based structural entropy metrics to quantify variability across multiple generations (stability rather than correctness).
- **[Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs](https://arxiv.org/abs/2406.15927)**: Predicts semantic entropy from hidden states as a cheaper hallucination signal (not code-specific; different notion of entropy).
- **[Semantic Uncertainty: Linguistic Entailment for Uncertainty Estimation](https://arxiv.org/abs/2302.09664)**: Introduces semantic uncertainty via entailment clustering, motivating semantic entropy.
- **[Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0)**: Introduces semantic entropy (entropy over semantic clusters across multiple samples) for detecting confabulations/hallucinations.
- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)**: Uses self-consistency across sampled generations to detect hallucinations.
- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)**: Introduces CoT prompting, often used in code reasoning pipelines.
- **[Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)**: Majority voting over multiple sampled rationales; a standard best-of-N baseline for closed-form answers.
- **[HumanEval: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)**: Introduces HumanEval benchmark and pass@k metric.
- **[MBPP: Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732)**: Introduces MBPP benchmark for Python program synthesis.
- **[EvalPlus](https://arxiv.org/abs/2305.01210)**: Provides HumanEval+/MBPP+ with stronger test suites for functional correctness evaluation.
- **[Calibration and Correctness of Language Models for Code](https://arxiv.org/abs/2402.02047)**: Studies how intrinsic probability-based confidence scores align with unit-test correctness for code generation, and analyzes calibration pitfalls.
- **[Maximizing Confidence Alone Improves Reasoning (RENT)](https://arxiv.org/abs/2505.22660)**: Uses token-entropy minimization as an intrinsic reward for RL on reasoning tasks; relevant as a contrasting use of entropy signals (training-time vs. inference-time ranking).

> Note: Some non-central references are not scraped into `./references/` and are cited by arXiv URL.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Aggregate uncertainty rankers | Rank candidates by aggregated token probabilities/entropy | Self-Certainty (KL), mean entropy/logprob | HumanEval/LiveCodeBench, math QA | Often miscalibrated; can correlate with length |
| Trajectory-structure uncertainty | Use dynamics/shape of uncertainty trajectory | EDIS; (this proposal) | Math QA; (proposed) HumanEval/MBPP | Hyperparameters; may collapse to dispersion/length |
| Multi-sample agreement / divergence | Confidence from agreement across samples (similarity or behavioral) | HonestCoder; Incoherence/difftrust | TruthCodeBench; HumanEval/MBPP | Extra compute (parsing/tests); not “free” |
| Learned rerankers / reward models | Train a model to score candidates | Top Pass; PRM/ORM literature | HumanEval/MBPP/LiveCodeBench | Requires labeled data; risk of reward hacking |
| Uncertainty-guided decoding | Use uncertainty to branch or invoke more compute | UnCert-CoT; EAGer | HumanEval(+), AIME | Adds compute; not purely selection |

### Closest Prior Work

**EDIS (Zhu et al., 2026).** EDIS defines burst and rebound spikes in token-entropy trajectories and shows they separate correct/incorrect math reasoning better than mean entropy, improving best-of-N selection without verifiers. It does not test code generation, and the raw spike count can be sensitive to output length.

**Self-Certainty (Kang et al., 2025).** Self-Certainty uses KL divergence from uniform aggregated over tokens as a distributional confidence measure for best-of-N selection and evaluates code benchmarks. It does not use temporal trajectory structure and focuses on aggregate distributional concentration.

**UnCert-CoT (Zhu et al., 2025).** UnCert-CoT uses entropy/probability-differential at line starts to decide when to invoke CoT-decoding for code. It does not rank completed candidates by full-trajectory dynamics.

**EAGer (Scalena et al., 2025).** EAGer uses token-level entropy peaks to branch and reallocate compute for test-time scaling. It is an adaptive generation method rather than a post-hoc ranker for a fixed candidate set.

**Incoherence/difftrust (Valentin et al., 2025).** Incoherence provides an oracle-free *lower bound* on error via behavioral divergence between programs on generated tests. It is not “free” (needs test generation/execution), and it targets error estimation rather than single-candidate selection.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| EDIS (2026) | Trajectory instability (burst/rebound spikes) for math best-of-N | No code eval; spike count can be length-confounded | Test on code + use rate/scale normalization | If instability reflects struggle points in code, it should predict failing programs |
| Self-Certainty (2025) | Aggregate KL-to-uniform confidence for best-of-N | No trajectory-shape features | Add spike-structure features beyond dispersion | If dynamics matter, nEDIS should beat dispersion-only rankers |
| HonestCoder (2024) | Multi-sample similarity confidence for selective display | Requires parsing/embeddings; not free | Use only logits/entropy trajectories | If spike structure is informative, it is cheaper than similarity pipelines |
| Incoherence (2025) | Behavioral divergence lower-bounds error | Needs generated tests + execution | Purely execution-free selection | A cheap heuristic may enable faster selection before execution |
| EAGer / UnCert-CoT (2025) | Uncertainty-triggered branching / selective CoT | Adds compute; not ranking of completed candidates | Post-hoc ranking of fixed candidate pool | Ranking is simpler to integrate into existing best-of-N sampling stacks |

---

## Experiments

### Experimental Setup

**Task setting.** Execution-free ranking: unit tests are used only for evaluation of the selected candidate, not for selection.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| deepseek-ai/deepseek-coder-6.7b-instruct | 6.7B | https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct | Open-weight instruct code model; logits available via HuggingFace Transformers (bf16) |

**Training Data (if applicable):**

No training data needed — inference only.

**Generation protocol (fixed).**
- For each problem, sample N=32 candidates with fixed decoding settings (temperature and nucleus sampling fixed; exact values pre-registered by verifier, e.g., temperature=0.8, top_p=0.95, max_new_tokens=512).
- For each sampled candidate, record token-level entropy trajectory {H_t} during generation.
- Compute three scores on the same candidate set: Length-only, CV_H, nEDIS.
- Select argmin score candidate for each scoring rule.

**Resource Estimate**:
- **Compute budget**: 20–80 A100-hours (single 7B model inference for (HumanEval+MBPP)×32 samples; entropy computed on-the-fly).
- **GPU memory**: ≤80GB (bf16 inference).
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| HumanEval | 164 Python function synthesis tasks with unit tests | pass@1 (fraction passing tests after selecting 1 candidate) | test | https://github.com/openai/human-eval | EvalPlus or official harness |
| MBPP (sanitized) | ~400 Python programming problems with tests | pass@1 | test | https://arxiv.org/abs/2108.07732 | EvalPlus or official harness |

**Primary metric.** pass@1 (higher is better).

**Significance test.** Paired bootstrap over problems (1000 resamples) comparing pass@1 between methods on the *same* candidate sets.

### Main Results

#### Results Table

All results below are **TBD** (require running the verification experiments in this proposal’s exact setting).

| Method (ranker) | Base Model | Benchmark | pass@1 | Source | Notes |
|---|---|---|---:|---|---|
| Greedy (T=0) | DeepSeek-Coder-Instruct 6.7B | HumanEval | **TBD** | - | Baseline decoding |
| First sample (no ranking) | DeepSeek-Coder-Instruct 6.7B | HumanEval | **TBD** | - | Control |
| Length-only (−T) | DeepSeek-Coder-Instruct 6.7B | HumanEval | **TBD** | - | Primary baseline 1 |
| CV_H (dispersion-only) | DeepSeek-Coder-Instruct 6.7B | HumanEval | **TBD** | - | Primary baseline 2 |
| **nEDIS (ours)** | DeepSeek-Coder-Instruct 6.7B | HumanEval | **TBD** | - | To be verified |
| Greedy (T=0) | DeepSeek-Coder-Instruct 6.7B | MBPP | **TBD** | - | Baseline decoding |
| Length-only (−T) | DeepSeek-Coder-Instruct 6.7B | MBPP | **TBD** | - | Primary baseline 1 |
| CV_H (dispersion-only) | DeepSeek-Coder-Instruct 6.7B | MBPP | **TBD** | - | Primary baseline 2 |
| **nEDIS (ours)** | DeepSeek-Coder-Instruct 6.7B | MBPP | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| nEDIS w-only | Use s(y) only (no CV term) | If spikes alone matter, this approaches full nEDIS |
| CV-only-squared | Use (1+CV_H^2) only | If spikes are redundant, this matches nEDIS |
| Window sensitivity | Compute burst spikes with w∈{2,4,8} (no extra inference) | If the signal is real, results should be stable within a small range |

### Analysis (Optional)

- **Redundancy check**: report Spearman ρ(s, CV_H) across all candidates; if ρ>0.7 and nEDIS≈CV_H, interpret gains as a dispersion effect rather than independent spike-structure signal.
- **Length residualization**: regress each score on T and recompute rankings from residuals (analysis only) to verify gains are not driven by length.

---

## Success Criteria

**Criterion 1: Outperform confound baselines (core claim).**
- Hypothesis: nEDIS selects more correct programs than both length-only and CV_H.
- Validation: nEDIS achieves higher pass@1 than max(length-only, CV_H) on **both** HumanEval and MBPP, with paired bootstrap significance p<0.05 on MBPP.

**Criterion 2: No hidden length proxy.**
- Hypothesis: Any nEDIS improvement does not disappear after length residualization.
- Validation: Under the length-residualized analysis, nEDIS still matches or exceeds CV_H; otherwise interpret as length-driven and refute the spike-structure claim.

**Refutation condition.** If nEDIS does not beat both baselines on both benchmarks (or causes a >0.5pp drop on HumanEval), refute the claim that entropy spike-structure provides a useful execution-free selection signal for code.

---

## Impact Statement

If successful, nEDIS would provide a training-free, execution-free heuristic for selecting code candidates from best-of-N sampling using only logits already produced during generation. This could reduce the need to execute unit tests on many candidates in latency-sensitive coding assistants, or serve as a cheap pre-filter before running more expensive verification.

---

## References

- [EDIS: Diagnosing LLM Reasoning via Entropy Dynamics](./references/EDIS-Diagnosing-LLM-Reasoning-via-Entropy-Dynamics/meta/meta_info.txt) - Zhu et al., 2026
- [Scalable Best-of-N Selection for Large Language Models via Self-Certainty](./references/Scalable-Best-of-N-Selection-for-Large-Language-Models-via-Self-Certainty/meta/meta_info.txt) - Kang et al., 2025
- [Top Pass: improve code generation by pass@k-maximized code ranking](./references/Top-Pass-Improve-Code-Generation-by-Pass@k-Maximized-Code-Ranking/meta/meta_info.txt) - Lyu et al., 2024
- [Incoherence as Oracle-less Measure of Error in LLM-Based Code Generation](./references/Estimating-Correctness-Without-Oracles-in-LLM-Based-Code-Generation/meta/meta_info.txt) - Valentin et al., 2025
- [Showing LLM-Generated Code Selectively Based on Confidence of LLMs](./references/Showing-LLM-Generated-Code-Selectively-Based-on-Confidence-of-LLMs/meta/meta_info.txt) - Li et al., 2024
- [Uncertainty-Guided Chain-of-Thought for Code Generation with LLMs](./references/Uncertainty-Guided-Chain-of-Thought-for-Code-Generation-with-LLMs/meta/meta_info.txt) - Zhu et al., 2025
- [EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling](./references/EAGER-Entropy-Aware-GEneRation-for-Adaptive-Inference-Time-Scaling/meta/meta_info.txt) - Scalena et al., 2025
- [Multicalibration for LLM-based Code Generation](https://arxiv.org/abs/2512.08810) - Campos et al., 2025
- [Localized Calibrated Uncertainty in Code Language Models](https://arxiv.org/abs/2512.24560) - Gros & Devanbu, 2025
- [Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs](https://arxiv.org/abs/2406.15927) - Kossen et al., 2024
- [HumanEval: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) - Chen et al., 2021
- [MBPP: Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732) - Austin et al., 2021
- [EvalPlus](https://arxiv.org/abs/2305.01210) - Liu et al., 2023
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Wei et al., 2022
