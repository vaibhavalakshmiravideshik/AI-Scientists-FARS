# untitled

# Anytime-CBU: Adaptive Rollout Allocation for Consequence-Based Utility Scoring

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

As large language models (LLMs) are applied to increasingly difficult domains (e.g., research-level mathematics), evaluation becomes a bottleneck: many problems are hard to verify directly, and human expert review does not scale. A common fallback is **LLM-as-a-judge**, but judge models can be biased and unreliable on difficult reasoning tasks and can collapse when the judge itself cannot solve/verify the underlying problem.

Recently, **Consequence-Based Utility (CBU)** proposed an alternative for *oracle-free* evaluation of hard math solutions: score a candidate solution by how much it helps solve a neighborhood of *related but verifiable* questions when used as an in-context exemplar ([Son et al., 2026](https://arxiv.org/abs/2602.06291)). CBU improves solution ranking quality on research-level math but requires many repeated solver rollouts per candidate (e.g., Avg@8–64), making it expensive for practical “best-of-N” selection or iterative research workflows.

### The Problem

CBU’s scoring cost scales as:

- (# target problems) × (# candidate solutions per problem) × (# neighborhood rollouts per candidate).

Uniformly spending the same rollout budget on every candidate is wasteful: for most problems, many candidates are clearly poor after a few rollouts. But naive early stopping risks discarding the true best candidate because CBU utilities are noisy: each rollout is stochastic (LLM sampling) and the neighborhood set can be small.

The practical question is therefore: **can we cut CBU’s rollout cost substantially without harming ranking/selection quality?**

### Key Insight and Hypothesis

CBU scoring can be viewed as a **pure-exploration multi-armed bandit** problem.

- Each candidate solution \(c_i\) is an “arm”.
- A rollout draws a neighborhood question \(q'\sim N(q)\) and samples a solver attempt conditioned on \(c_i\).
- The rollout outcome is a Bernoulli reward (correct / incorrect).

We hypothesize that **best-arm identification (BAI)** algorithms (e.g., LUCB / successive elimination) can adaptively allocate rollouts to the most promising candidates and stop early once the top candidate is identified with high confidence. This should preserve CBU’s selection quality while reducing solver calls by ≥2×.

This hypothesis could fail if (i) utilities are too flat/noisy, so BAI needs nearly the full budget anyway, or (ii) early confidence bounds are miscalibrated, causing premature elimination.

---

## Proposed Approach

### Overview

We propose **Anytime-CBU**, a drop-in replacement for CBU’s *uniform* rollout allocation. For each target question \(q\) and candidate set \(\{c_i\}_{i=1}^m\), Anytime-CBU adaptively chooses which candidate to evaluate next, and when to stop, aiming to identify the best candidate under a fixed max budget \(K_{\max}\).

### Method Details

#### CBU utility (background)

CBU defines the utility of a candidate solution \(c\) for target question \(q\) as:

\[
V(c\mid q)=\mathbb{E}_{q'\sim N(q)}\big[\mathbb{1}[\text{Solve}(c,q')\ \text{is correct}]\big]
\]

and estimates it by repeated rollouts:

\[
\hat V(c\mid q)=\tfrac{1}{K}\sum_{k=1}^K \mathbb{1}[\text{Solve}(c,q'_k)\ \text{is correct}].
\]

#### Anytime-CBU allocation rule (LUCB-style)

For each candidate arm \(i\), maintain \(t_i\) rollouts and \(s_i\) successes.

- Empirical mean: \(\hat\mu_i=s_i/t_i\).
- Confidence radius (Hoeffding-style): \(r_i(t)=\sqrt{\tfrac{\log(2m/\delta)}{2t}}\).

Define:
- \(\text{LCB}_i=\hat\mu_i-r_i\), \(\text{UCB}_i=\hat\mu_i+r_i\).

Algorithm:
1. Warm-start: allocate \(t_0\) rollouts to each candidate.
2. Let \(i^* = \arg\max_i \hat\mu_i\) (current best).
3. Let \(j^* = \arg\max_{j\neq i^*} \text{UCB}_j\) (best challenger).
4. If \(\text{LCB}_{i^*} > \text{UCB}_{j^*}\) (or exceeds by \(\epsilon\) margin), stop and output \(i^*\).
5. Else, allocate an additional rollout to whichever of \(i^*\) or \(j^*\) has larger \(r\) (or allocate to both), respecting per-arm cap \(K_{\max}\).

Output:
- Top-1 candidate \(i^*\) (primary output)
- Estimated utilities \(\{\hat\mu_i\}\) from the final state (secondary: full ranking)

#### Matched-cost “dumb adaptive” control

To show gains are from *adaptive allocation* rather than “just fewer rollouts”, we include a matched-cost baseline:

- **Random-K**: allocate per-candidate rollout counts \(K_i\) drawn randomly (e.g., uniform on \([t_0, K_{\max}]\)), with the expected *total* rollouts matched to Anytime-CBU.

### Key Innovations

- **Reformulation of CBU scoring as best-arm identification**: treat candidate evaluation as pure exploration with Bernoulli rewards.
- **Anytime, stop-when-confident behavior**: adaptively terminates per-instance based on confidence gaps.
- **Matched-cost control baseline** to isolate the effect of intelligent allocation.

---

## Related Work

### Field Overview

**Oracle-free evaluation of hard reasoning.** LLM-as-a-judge is widely used but struggles on hard reasoning and is sensitive to prompt/formatting biases ([Son et al., 2026](https://arxiv.org/abs/2602.06291); [PPE](https://arxiv.org/abs/2410.14872)). CBU proposes evaluating solutions by downstream consequences on verifiable neighborhood questions, reducing judge-style biases but increasing evaluation cost.

**Adaptive test-time compute / early stopping.** A large body of work reduces inference cost in test-time scaling and self-consistency by stopping early when answers stabilize or confidence is high (e.g., ESC, ConSol, SeerSC, CGES). These methods primarily target *single-answer aggregation*, not *multi-candidate utility scoring*.

**Bandits for pure exploration.** Best-arm identification (BAI) algorithms such as LUCB and successive reject/elimination allocate samples to distinguish the best option under uncertainty, with strong theoretical guarantees.

### Related Papers

- **[Judging What We Cannot Solve: A Consequence-Based Approach for Oracle-Free Evaluation of Research-Level Math](https://arxiv.org/abs/2602.06291)**: Introduces CBU; our work targets its evaluation cost.
- **[RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](./references/RealMath/meta/meta_info.txt)**: Public verifiable research-math benchmark; we use it to build a public CBU-style setting.
- **[ConSol: Sequential Probability Ratio Testing to Find Consistent LLM Reasoning Paths Efficiently](./references/ConSol/meta/meta_info.txt)**: SPRT early-stopping for self-consistency; different setting (single-answer aggregation).
- **[Early-stopping Self-Consistency for Multi-step Reasoning](./references/ESC/meta/meta_info.txt)**: Stops SC when window entropy collapses; different setting.
- **[Seer Self-Consistency: Advance Budget Estimation for Adaptive Test-Time Scaling](./references/SeerSC/meta/meta_info.txt)**: Predicts scaling budget via System-1 entropy proxy.
- **[CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency](./references/CGES/meta/meta_info.txt)**: Bayesian early stopping for SC.
- **[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)**: SC baseline.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Verification for math reasoning; motivates verifiable evaluation.
- **[How to Evaluate Reward Models for RLHF](https://arxiv.org/abs/2410.14872)**: Preference Proxy Evaluations (PPE); reward-model evaluation context.
- **[FrontierMath](https://arxiv.org/abs/2411.04872)**: Research-level math benchmark; illustrates the verification bottleneck.
- **[A Batch Sequential Halving Algorithm without Performance Degradation](https://arxiv.org/abs/2406.00424)**: Batch-friendly pure-exploration elimination.
- **[Adaptive Repetition for Mitigating Position Bias in LLM-Based Ranking](https://arxiv.org/abs/2507.17788)**: Early stopping for LLM ranking via repetition; different bias target.
- **[Towards bandit-based prompt-tuning for in-the-wild foundation agents](https://arxiv.org/abs/2502.06358)**: Bandits for selecting prompts/segments under budget.
- **[Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://arxiv.org/abs/2503.01163)**: Bandits for prompt strategy selection.
- **[Down-Sampling Rollouts in LLM Reinforcement Learning](https://arxiv.org/abs/2504.13818)**: Selective rollouts for RL training efficiency.
- **[Efficient Reinforcement Learning for LLM Reasoning via Selective Rollout](https://arxiv.org/abs/2506.02177)**: Selective rollout filtering for RL.
- **[Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs](https://arxiv.org/abs/2507.02076)**: Survey; positions adaptive compute.
- **[Reliable Fine-Grained Evaluation of Natural Language Math Proofs](https://arxiv.org/abs/2510.13888)**: Fine-grained proof evaluation; related to math judging.
- **[Best Arm Identification in Multi-Armed Bandits](https://chercheurs.lille.inria.fr/~munos/papers/files/ABM10.pdf)**: Foundational BAI survey-style reference (successive rejects / LUCB family).
- **[On Top-k Selection in Multi-Armed Bandits and Hidden Bipartite Graphs](https://www.cse.cuhk.edu.hk/~taoyf/paper/nips15.pdf)**: Top-k selection under sampling budgets.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Oracle-free evaluation (judge / RM) | Score solutions directly with a judge or reward model | PPE (2410.14872), ProofGrader (2510.13888) | RM benchmarks; proof grading sets | Judge bias; solver-evaluator gap |
| Consequence-based evaluation | Score solutions by downstream transfer to verifiable neighbors | CBU (2602.06291) | ExpertMath (embargoed), AIME/RealMath variants | High rollout cost; needs neighborhoods |
| Early stopping for TTS/SC | Stop sampling when confidence/stability is high | ESC (2401.10480), ConSol (2503.17587), SeerSC (2511.09345), CGES (2511.02603) | GSM8K/MATH/AIME/GPQA | Mostly single-answer aggregation |
| Pure-exploration bandits | Allocate samples to identify best arm under uncertainty | ABM10, top-k selection (NIPS15), batch SH (2406.00424) | Bandit theory; HPO | Needs calibrated confidence / assumptions |

### Closest Prior Work

1. **CBU (Son et al., 2026)** introduces consequence-based scoring but uses **uniform** rollout allocation per candidate. Our work keeps the same scoring signal but changes the *allocation policy*, asking whether we can preserve selection quality with far fewer rollouts.

2. **ConSol / ESC / CGES / SeerSC** reduce inference-time sampling for *self-consistency* (one answer per question). Our setting differs: we must choose among **multiple candidate solutions**, and each rollout’s reward depends on both candidate and sampled neighborhood question.

3. **BAI / successive elimination** literature provides the statistical machinery to allocate samples adaptively, but has not been instantiated for **CBU-style oracle-free math evaluation** (as far as we found).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| CBU (2602.06291) | Utility score via neighborhood solve accuracy; uniform rollouts | Wastes rollouts on clearly bad candidates | Adaptive allocation + early stop | Same quality at lower cost |
| ConSol (2503.17587) | SPRT early-stops self-consistency | Not multi-candidate scoring | Apply pure-exploration allocation across candidates | Targets the real cost driver in CBU |
| ESC (2401.10480) | Window entropy early stop for SC | Needs sequential sampling; not candidate ranking | Arm-wise confidence and challenger tracking | More direct for selection among candidates |
| SeerSC (2511.09345) | Predict compute need via System-1 entropy | Not designed for candidate scoring | Use BAI bounds on observed Bernoulli rewards | Theoretical stop criterion tied to selection |

---

## Experiments

### Experimental Setup

**Goal.** Measure the **quality–cost frontier** of consequence-based utility scoring: can Anytime-CBU match uniform-budget CBU’s selection quality at much lower rollout cost?

**Dataset (public, verifiable).** Use **RealMath** QA pairs and define neighborhoods from the dataset itself:

- Dataset: `ethz-spylab/RealMath` (HuggingFace) + official code repo.
- For a target question \(q\) with metadata `link` (source paper), define neighborhood \(N(q)\) as other RealMath questions with the **same `link`** (same source paper), excluding \(q\).
- Filter to targets with \(|N(q)|\ge 2\) and to neighborhoods with at least some baseline solvability (to avoid “always fail”/“always succeed” degenerate regimes).

**Candidate solution generation + labeling.** For each target \(q\):

- Sample \(m\) candidate solutions from a generator model (temperature > 0).
- Label each candidate as correct/incorrect by extracting its final answer and verifying against RealMath’s answer using the official RealMath normalization/eval utilities.
- Keep only targets where the candidate pool contains **at least one correct and one incorrect** candidate (so ranking metrics are meaningful).

**Rollout definition.** One rollout = sample a neighbor \(q'\in N(q)\), run solver on \(q'\) **conditioned on candidate \(c_i\) as an in-context exemplar**, and score correctness vs RealMath ground truth.

**Methods (≤3 main conditions).**

1. **Uniform-CBU (Kmax)**: allocate \(K_{\max}\) rollouts per candidate.
2. **Random-K (matched cost)**: random per-candidate rollout counts with expected total rollouts matched to Anytime-CBU.
3. **Anytime-CBU (ours)**: LUCB-style adaptive allocation with per-arm cap \(K_{\max}\).

**Primary metric.** Acc@1: % of targets where the top-ranked candidate (by estimated utility) is correct.

**Secondary metrics.**
- AUC: treat utility score as a classifier separating correct vs incorrect candidates within each target, then average across targets.
- Compute: total solver calls, total generated tokens, and average rollouts per candidate.

**Decision rule.** Anytime-CBU is successful if:
- It matches Uniform-CBU within statistical uncertainty on Acc@1/AUC (bootstrap over targets), **and**
- It reduces total rollouts (or tokens) by **≥2×**, **and**
- It outperforms Random-K at the same expected compute (showing allocation matters).

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen2.5-Math-Instruct | 7B (or similar) | https://huggingface.co/Qwen | Open-weight math-capable model; can be replaced by any strong open solver |
| DeepSeek-R1-Distill (optional) | 7B/8B | https://huggingface.co/deepseek-ai | Optional stronger solver for robustness check |

**Training Data (if applicable):**

No training data needed — inference only.

**Other Resources (if applicable):**
- RealMath dataset + evaluation code: https://github.com/ethz-spylab/RealMath

**Resource Estimate**:

- **Compute budget**: inference-only. We will cap to a small subset of targets (e.g., 100–300) and set \(m\) and \(K_{\max}\) to keep total solver calls comfortably within budget. Report compute primarily as **#solver calls and total tokens**, which the Verification module can translate to GPU-hours using its inference stack.
- **GPU memory**: 7B-class models fit on a single A100-80GB.
- **API usage**: none required.

**Infrastructure constraints** (proposals requiring these are infeasible):
- Search engine APIs (Google, Bing) — NOT available
- Web browsers / desktop GUIs / mobile environments — NOT available
- Complex game engines or heavy simulation environments — NOT available

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| RealMath | Verifiable research-level math QA pairs with context | Acc@1, AUC, calls/tokens | use official splits; additionally filter by neighbor availability | https://huggingface.co/datasets/ethz-spylab/RealMath | https://github.com/ethz-spylab/RealMath |

**Evaluation Scripts:**
- Use RealMath official evaluation / answer normalization utilities.
- Add a CBU-style harness: (candidate generation → candidate labeling → neighborhood rollouts with exemplar conditioning → utility estimation → selection metrics).

**Download Links Checklist:**
- [x] All benchmark datasets have download links
- [ ] All models have download links
- [ ] Licenses are compatible with research use

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Acc@1 | AUC | Source | Notes |
|--------|------------|-----------|------:|----:|--------|-------|
| Uniform-CBU (Kmax) | Qwen2.5-Math-7B | RealMath | **TBD** | **TBD** | - | Needs re-run (baseline) |
| Random-K (matched cost) | Qwen2.5-Math-7B | RealMath | **TBD** | **TBD** | - | Matched expected rollouts to ours |
| **Anytime-CBU (Ours)** | Qwen2.5-Math-7B | RealMath | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| No early stop (uniform Kmax) | Always sample Kmax per candidate | Best quality, highest cost |
| High-confidence stop threshold | \(\delta\) / \(\epsilon\) tighter vs looser | Tighter → more cost, slightly higher quality |

### Analysis (Optional)

- Where does compute go? Report fraction of total calls spent on (i) candidate generation, (ii) candidate scoring.
- Regime analysis: does Anytime-CBU save more on problems with large utility gaps (easy-to-separate arms)?

---

## Success Criteria

**Criterion 1: Quality preservation at lower cost**
- Hypothesis: Anytime-CBU matches Uniform-CBU on Acc@1/AUC while using ≥2× fewer rollouts/tokens.
- Validation: Bootstrap CIs show no meaningful degradation and compute reduction ≥2×.

**Criterion 2: Allocation matters (not just fewer samples)**
- Hypothesis: Anytime-CBU beats Random-K at matched expected rollout cost.
- Validation: Anytime-CBU yields higher Acc@1/AUC than Random-K under the same expected compute.

---

## Impact Statement

If successful, Anytime-CBU would make consequence-based oracle-free evaluation substantially cheaper, enabling wider use of CBU-style evaluators for best-of-N selection and iterative mathematical research workflows without paying an “always-on” large rollout budget.

---

## References

- [Judging What We Cannot Solve: A Consequence-Based Approach for Oracle-Free Evaluation of Research-Level Math](https://arxiv.org/abs/2602.06291) - Son et al., 2026
- [RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](./references/RealMath/meta/meta_info.txt) - Zhang et al., 2025
- [ConSol: Sequential Probability Ratio Testing to Find Consistent LLM Reasoning Paths Efficiently](./references/ConSol/meta/meta_info.txt) - Lee et al., 2025
- [Early-stopping Self-Consistency for Multi-step Reasoning](./references/ESC/meta/meta_info.txt) - Li et al., 2024
- [Seer Self-Consistency: Advance Budget Estimation for Adaptive Test-Time Scaling](./references/SeerSC/meta/meta_info.txt) - Ji & Wang et al., 2025
- [CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency](./references/CGES/meta/meta_info.txt) - Aghazadeh et al., 2025
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) - Wang et al., 2023
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [How to Evaluate Reward Models for RLHF](https://arxiv.org/abs/2410.14872) - (PPE authors), 2024
- [FrontierMath](https://arxiv.org/abs/2411.04872) - (authors), 2024
- [A Batch Sequential Halving Algorithm without Performance Degradation](https://arxiv.org/abs/2406.00424) - Koyamada et al., 2024
- [Adaptive Repetition for Mitigating Position Bias in LLM-Based Ranking](https://arxiv.org/abs/2507.17788) - Vardasbi et al., 2025
- [Towards bandit-based prompt-tuning for in-the-wild foundation agents](https://arxiv.org/abs/2502.06358) - (authors), 2025
- [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://arxiv.org/abs/2503.01163) - (authors), 2025
- [Down-Sampling Rollouts in LLM Reinforcement Learning](https://arxiv.org/abs/2504.13818) - Xu et al., 2025
- [Efficient Reinforcement Learning for LLM Reasoning via Selective Rollout](https://arxiv.org/abs/2506.02177) - (authors), 2025
- [Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs](https://arxiv.org/abs/2507.02076) - (authors), 2025
- [Reliable Fine-Grained Evaluation of Natural Language Math Proofs](https://arxiv.org/abs/2510.13888) - Ma et al., 2025
- [Best Arm Identification in Multi-Armed Bandits](https://chercheurs.lille.inria.fr/~munos/papers/files/ABM10.pdf) - Audibert et al., 2010
- [On Top-k Selection in Multi-Armed Bandits and Hidden Bipartite Graphs](https://www.cse.cuhk.edu.hk/~taoyf/paper/nips15.pdf) - (authors), 2015
