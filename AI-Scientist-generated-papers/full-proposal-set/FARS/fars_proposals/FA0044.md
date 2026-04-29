# untitled

# Selective Self-Reference for LLM-as-a-Judge: Using Self-Consistency to Reduce Error Propagation

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used as *automated evaluators* ("LLM-as-a-judge") to score model outputs, select the best of many samples, and provide rewards for preference optimization and reinforcement learning. This is attractive because human evaluation is expensive and slow, and many modern training loops require millions of evaluations.

A common way to improve judge reliability is **self-reference / solve-then-judge**: a two-step prompting pattern where the judge (i) answers the task itself (the “self-solve” step) and then (ii) evaluates candidate responses while using that self-answer as an explicit reference. Recent evidence suggests this can improve evaluation quality. For example, **[Do Before You Judge](./references/Do-Before-You-Judge-Self-Reference-as-a-Pathway-to-Better-LLM-Evaluation/meta/meta_info.txt)** shows that self-reference-guided judging can strengthen the link between a model’s generation ability and its judging ability, and **[PhyCritic](./references/PhyCritic-Multimodal-Critic-Models-for-Physical-AI/meta/meta_info.txt)** trains a multimodal judge that produces an explicit self-prediction before judging.

### The Problem

Self-reference introduces a practical failure mode: **error propagation**. If the judge’s self-answer is wrong, then using it as an evaluation anchor can cause the judge to reject a correct candidate (because it disagrees with the judge’s incorrect reference). This is explicitly noted as a limitation in **[Do Before You Judge](./references/Do-Before-You-Judge-Self-Reference-as-a-Pathway-to-Better-LLM-Evaluation/sections/Potential%20for%20Error%20Propagation.md)**. In multimodal settings, **[PhyCritic](./references/PhyCritic-Multimodal-Critic-Models-for-Physical-AI/sections/Ablation%20Studies.md)** also reports that judgment quality is strongly associated with whether the model’s self-prediction is correct (and removing self-reference reduces performance), implying that a wrong self-prediction can be harmful.

A straightforward response is "use a stronger judge model". However, in real deployments, practitioners often face constraints (cost, latency, or model availability) that make it attractive to use a **moderate-accuracy judge**. In this regime, always-on self-reference may backfire due to error propagation.

### Key Insight and Hypothesis

**Key insight.** Self-consistency (agreement among multiple independent samples) is a simple, training-free proxy for whether an LLM is likely correct (**[Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)**; **[Self-Consistency Improves Chain-of-Thought Reasoning](https://arxiv.org/abs/2203.11171)**). When the model’s self-solves are unanimous, the self-answer is more likely to be correct; when they disagree, the self-answer is more likely to be unreliable.

**Hypothesis.** In solve-then-judge prompting for objective multiple-choice judging, **selectively enabling self-reference only on high-agreement (unanimous) self-solves** improves judge accuracy compared to always using self-reference (which propagates wrong self-answers), while keeping the benefits of self-reference on high-confidence cases.

This hypothesis could be wrong if (i) unanimity is not a good correctness proxy in the target domain (models can be unanimously wrong), (ii) the non-unanimous slice is too small for the effect to matter, or (iii) when the model is non-unanimous, a no-reference judge is not better than an anchored judge (so abstaining from self-reference does not help).

---

## Proposed Approach

### Overview

We propose **Selective Self-Reference (SSR-Judge)**, an inference-time prompting wrapper for LLM-as-a-judge:

1. **Self-solve sampling**: Sample the judge model’s answer to the multiple-choice question **k=3** times.
2. **Agreement gate**: If all three answers agree (unanimous), treat the majority answer as a reliable reference; otherwise treat the reference as unreliable.
3. **Judge step**:
   - If the gate passes (unanimous): run solve-then-judge by providing the self-answer as a reference.
   - If the gate fails (non-unanimous): run a standard no-reference judge prompt (judge must decide directly from the question and candidate responses).

We evaluate SSR-Judge as a *prompting strategy* (no training) and compare it to compute-matched baselines.

### Method Details

#### Task format: pairwise judging for multiple-choice questions

We focus on objective multiple-choice questions (MCQs), where there is a single correct option letter. Each evaluation instance contains:
- Question and options
- Two candidate responses, each including a final answer letter
- A ground-truth answer letter (used only for evaluation)

The judge must pick which response is correct.

#### Candidate-response construction (fully automated)

We need candidate responses that are comparably long and stylistically similar, to reduce length/style confounds that affect LLM judges (**[Self-Preference Bias](./references/Self-Preference-Bias-in-LLM-as-a-Judge/meta/meta_info.txt)**).

For each MCQ item with correct answer \(A^*\), we construct two responses using a fixed response generator model \(M_{gen}\):
- **Correct candidate**: prompt \(M_{gen}\) to provide a concise explanation that concludes with \(\boxed{A^*}\).
- **Wrong candidate**: sample a wrong option \(A^- \neq A^*\) uniformly from the other options, then prompt \(M_{gen}\) to provide a similarly structured explanation that concludes with \(\boxed{A^-}\).

We enforce a strict format and automatically re-try generation if the final boxed answer does not match the forced option.

This produces "persuasive but wrong" candidates without requiring human annotation.

#### Judge conditions (3 conditions, compute-matched)

Let \(M_J\) be the judge model.

All conditions compute the same self-solve samples to eliminate the confound "self-reference uses more compute":
- **Self-solve step**: run \(M_J\) **k=3** times on (question, options) with sampling (temperature \(>0\)), and parse the predicted option letter each time.

Then evaluate three prompting conditions:

- **A: No-reference (compute-matched)**: discard the k self-solves, then prompt \(M_J\) to choose the better response using only (question, options, Response 1, Response 2).

- **B: Always self-reference**: compute the majority self-answer \(\hat{A}_{maj}\) from the k samples (ties broken randomly but logged), then prompt \(M_J\) to judge using (question, options, \(\hat{A}_{maj}\), Response 1, Response 2).

- **C: Selective self-reference (ours)**: if the k self-solves are unanimous, use \(\hat{A}_{maj}\) as in B; otherwise, fall back to the A prompt (no reference).

To reduce position bias in pairwise judging (**[Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)**; **[JudgeBench](https://arxiv.org/abs/2410.12784)**), we evaluate each instance twice with swapped response order and take majority vote (ties counted as incorrect).

### Key Innovations

- **Confidence-gated self-reference for judging**: a deployable rule for when to include a self-answer reference, aimed at reducing error propagation rather than improving calibration scores.
- **Compute-matched causal comparison**: all conditions use the same k self-solves, isolating the effect of *using* the reference rather than spending extra inference budget.
- **Mechanism-targeted slices**: the evaluation explicitly measures the regime where self-reference is expected to be harmful (non-unanimous self-solves with wrong majority), rather than only reporting overall accuracy.

---

## Related Work

### Field Overview

LLM-as-a-judge has become a standard tool for model evaluation, reward modeling, and benchmark automation. However, judge outputs are sensitive to prompt templates and exhibit systematic biases such as position bias, verbosity/length bias, and self-preference bias (**[Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)**; **[Self-Preference Bias](./references/Self-Preference-Bias-in-LLM-as-a-Judge/meta/meta_info.txt)**; **[Rating Roulette](https://arxiv.org/abs/2510.27106)**). This has motivated work on more reliable judging protocols (checklists, probabilistic scoring, multi-judge aggregation) and dedicated benchmarks to evaluate judges themselves (**[CheckEval](https://arxiv.org/abs/2403.18771)**; **[TrustJudge](https://arxiv.org/abs/2509.21117)**; **[JudgeBench](https://arxiv.org/abs/2410.12784)**).

A separate thread improves judges by providing **references**: gold answers, stronger-model outputs, or the judge’s own self-answer. **[Do Before You Judge](./references/Do-Before-You-Judge-Self-Reference-as-a-Pathway-to-Better-LLM-Evaluation/meta/meta_info.txt)** shows self-reference-guided evaluation can improve judging in pointwise correctness settings, but warns about error propagation when the self-answer is wrong. **[PhyCritic](./references/PhyCritic-Multimodal-Critic-Models-for-Physical-AI/meta/meta_info.txt)** extends solve-then-judge into multimodal critics trained with reinforcement learning, and empirically links self-prediction correctness to judgment quality.

Finally, confidence estimation and abstention are widely studied for LLM reliability. Self-consistency and multi-sample agreement are simple confidence signals (**[Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)**; **[Self-Consistency Improves Chain-of-Thought Reasoning](https://arxiv.org/abs/2203.11171)**), and recent work studies calibration and confidence-driven evaluation pipelines (**[Overconfidence in LLM-as-a-Judge](./references/Overconfidence-in-LLM-as-a-Judge-Diagnosis-and-Confidence-Driven-Solution/meta/meta_info.txt)**). Our proposal connects these threads by using self-consistency to decide when self-reference should be applied in judging.

### Related Papers

- **[Do Before You Judge](./references/Do-Before-You-Judge-Self-Reference-as-a-Pathway-to-Better-LLM-Evaluation/meta/meta_info.txt)**: Self-reference-guided prompting improves pointwise correctness judging but warns about error propagation when self-answer is wrong.
- **[PhyCritic](./references/PhyCritic-Multimodal-Critic-Models-for-Physical-AI/meta/meta_info.txt)**: Trains a multimodal solve-then-judge critic and shows self-prediction correctness is strongly associated with judgment performance.
- **[Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)**: Establishes LLM-as-a-judge and documents position/length biases and prompt sensitivity.
- **[MT-Bench](https://arxiv.org/abs/2306.05685)**: A widely used multi-turn evaluation framework relying on LLM judges.
- **[Chatbot Arena](https://arxiv.org/abs/2403.04132)**: Large-scale pairwise preference collection used to benchmark chat models and to study judge bias.
- **[AlpacaEval](https://arxiv.org/abs/2305.14387)**: An automated evaluation framework using pairwise comparisons with LLM judges.
- **[G-Eval](https://arxiv.org/abs/2303.16634)**: Uses chain-of-thought evaluation prompts to improve LLM-based scoring stability.
- **[FairEval](https://arxiv.org/abs/2305.11747)**: Studies fairness and bias mitigation for LLM evaluators.
- **[JudgeBench](https://arxiv.org/abs/2410.12784)**: A benchmark for evaluating judge correctness on difficult domains (including MMLU-Pro-derived items).
- **[ContextualJudgeBench](https://arxiv.org/abs/2503.15620)**: Extends JudgeBench to contextual evaluation settings.
- **[CheckEval](https://arxiv.org/abs/2403.18771)**: Checklist-based judging to improve agreement and stability over Likert-style prompts.
- **[TrustJudge](https://arxiv.org/abs/2509.21117)**: Probabilistic scoring and aggregation methods to reduce transitivity and scoring inconsistencies.
- **[Rating Roulette](https://arxiv.org/abs/2510.27106)**: Measures intra-model self-inconsistency of LLM judges across tasks.
- **[Self-Preference Bias](./references/Self-Preference-Bias-in-LLM-as-a-Judge/meta/meta_info.txt)**: Quantifies self-preference bias and links it to perplexity preference.
- **[Overconfidence in LLM-as-a-Judge](./references/Overconfidence-in-LLM-as-a-Judge-Diagnosis-and-Confidence-Driven-Solution/meta/meta_info.txt)**: Diagnoses calibration failures of LLM judges and proposes confidence-driven aggregation.
- **[Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)**: Shows that self-consistency / multiple samples improve confidence estimation.
- **[Self-Consistency Improves Chain-of-Thought Reasoning](https://arxiv.org/abs/2203.11171)**: Majority voting over diverse reasoning traces improves answer accuracy; agreement provides a confidence proxy.
- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)**: Uses sampling-based self-consistency signals for factuality checking.
- **[Self-Evaluation Improves Selective Generation](https://arxiv.org/abs/2312.09300)**: Uses self-evaluation to support selective answering/abstention.
- **[One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)**: Shows superficial patterns can fool LLM judges, motivating more robust judging protocols.
- **[BadJudge](https://arxiv.org/abs/2503.00596)**: Demonstrates backdoor vulnerabilities in LLM judges and proposes model-merge defenses.
- **[LLMs Cannot Reliably Judge (Yet?)](https://arxiv.org/abs/2506.09443)**: Introduces a robustness evaluation framework and shows large robustness variability across prompts and models.
- **[Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2409.02692)**: Demonstrates high attack success rates for candidate-response prompt injection in judge settings.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Prompted LLM judges | Prompt an LLM to score or compare outputs | Judging LLM-as-a-Judge; G-Eval | MT-Bench; AlpacaEval | Biases; prompt sensitivity |
| Judge reliability methods | Reduce inconsistency via protocol changes | CheckEval; TrustJudge; Rating Roulette | SummEval; MT-Bench; JudgeBench | Often ignores reference-guided judging |
| Reference-guided judging | Use references (gold / stronger model / self-answer) to guide evaluation | Do Before You Judge; PhyCritic | MCQ correctness; multimodal reward benchmarks | Error propagation if reference is wrong |
| Confidence / calibration for judges | Quantify and use confidence to make safer decisions | Overconfidence in LLM-as-a-Judge; Kadavath et al. | JudgeBench; calibration metrics | Confidence can be miscalibrated |
| Adversarial robustness of judges | Stress-test against manipulation | One Token to Fool; BadJudge; RobustJudge | Robustness benchmarks | Defenses can reduce utility |

### Closest Prior Work

1. **Do Before You Judge**: Proposes self-reference-guided evaluation and explicitly notes error propagation when the self-answer is wrong, but does not propose an inference-time rule for when to trust self-reference.
2. **PhyCritic**: Trains a self-referential multimodal critic and reports that self-prediction correctness is strongly associated with judgment quality, but focuses on training-time rewards rather than a test-time gating strategy.
3. **Confidence estimation via self-consistency (Kadavath et al.; Self-Consistency)**: Establishes agreement as a confidence proxy for answering, but does not study *using confidence to decide whether to apply self-reference in judging*.
4. **Overconfidence in LLM-as-a-Judge**: Studies judge calibration and confidence-driven aggregation, but focuses on combining multiple models rather than selectively enabling self-reference within a single judge.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Do Before You Judge | Always uses self-answer as reference for pointwise judging | Error propagation when self-answer is wrong | Gate self-reference by self-consistency | Avoid anchoring on uncertain / wrong self-answers |
| PhyCritic | Trains solve-then-judge multimodal critics with self-pred reward | Training-focused; no test-time gate | Inference-time gating rule | Useful even without changing training |
| Kadavath et al. / Self-Consistency | Uses multi-sample agreement for confidence | Not applied to judging protocols | Use unanimity to decide whether to reference | Targets the failure mode specific to self-reference |
| Overconfidence in LLM-as-a-Judge | Improves calibration via confidence metrics and ensembles | Requires multiple judges / fuser overhead | Single-judge gating | Low overhead and easy to deploy |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-7B-Instruct (judge, \(M_J\)) | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Used for self-solve and judging in all conditions |
| Qwen2.5-32B-Instruct (candidate generator, \(M_{gen}\)) | 32B | https://huggingface.co/Qwen/Qwen2.5-32B-Instruct | Generates matched correct/wrong candidate responses |

**Training Data (if applicable):**

No training data needed — inference only.

**Resource Estimate**:

- Dataset size: default **N=1,400** items (100 per MMLU-Pro category, 14 categories), following the sampling protocol in Do Before You Judge.
- Per item:
  - Candidate generation: 2 calls to \(M_{gen}\) (forced correct + forced wrong), each capped at ~200 output tokens.
  - For each judge condition (A/B/C): k=3 self-solves (1-token output) + 2 judge calls (swapped order) with ~20 output tokens.
  - Total judge calls per item: 3 conditions * (3+2) = 15 calls (mostly short).
- Total calls: ~2,800 generator calls + ~21,000 judge calls.

This is inference-only and should fit comfortably within the 768 A100 GPU-hour budget if run locally (7B and 32B models), or can be run via APIs.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MMLU-Pro (TIGER-Lab) | Multi-domain multiple-choice questions with 10 options per question | PreferenceAcc; SliceAcc (gate-on/off) | sampled subset | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro | Custom script (candidate generation + judge prompting + parsing) |

**Metrics:**
- **PreferenceAcc (%)**: fraction of items where the judge selects the candidate whose final answer letter equals the ground-truth option.
- **GateOn rate (%)**: fraction of items where k self-solves are unanimous.
- **Gate precision (%)**: \(P(\text{self-solve correct} \mid \text{unanimous})\).
- **SliceAcc (%)**: PreferenceAcc restricted to (i) gate-on items and (ii) gate-off items.

### Main Results

**Decision rule (verification-first):**

1. **Stage-0 pilot (N=200 items).** Compute gate statistics and a mechanistic check:
   - If GateOn rate < 10% or > 90%, refute (gate too rare or too frequent to matter).
   - If gate precision < 80%, refute (unanimity is not a reliable signal).
   - On gate-off items, if A does not outperform B (no evidence of harmful anchoring), refute (gating unlikely to help).

2. **Full run (N=1,400 items).** Accept the hypothesis if:
   - C has higher overall PreferenceAcc than B, and
   - On gate-off items, C outperforms B and is not worse than A (the fallback behaves as intended), and
   - On gate-on items, C is within 1 point of B (using the reference does not lose the benefit when confidence is high).

#### Results Table

| Method | Base Model | Benchmark | PreferenceAcc (%) | GateOn rate (%) | SliceAcc gate-on (%) | SliceAcc gate-off (%) | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| A: No-reference (compute-matched) | Qwen2.5-7B | MMLU-Pro (1.4k) | **TBD** | **TBD** | **TBD** | **TBD** | - | Needs re-run |
| B: Always self-reference (k=3 majority) | Qwen2.5-7B | MMLU-Pro (1.4k) | **TBD** | **TBD** | **TBD** | **TBD** | - | Needs re-run |
| C: **Selective self-reference (unanimity gate)** | Qwen2.5-7B | MMLU-Pro (1.4k) | **TBD** | **TBD** | **TBD** | **TBD** | - | Needs re-run |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| C with k=5 (unanimity) | Use 5 self-solves instead of 3 | Higher gate precision but lower GateOn rate; may improve or degrade overall accuracy |

### Analysis (Optional)

- **Error-propagation slice**: report performance on items where majority self-solve is wrong and non-unanimous (where B is expected to anchor to a wrong reference and C abstains).
- **Calibration vs utility**: report how often self-solve unanimity is confidently wrong (unanimous but incorrect), bounding what inference-time gating can fix.

---

## Success Criteria

**Criterion 1: Reduced error propagation**
- Hypothesis: When self-solves are non-unanimous, omitting the self-answer reference reduces misjudgments caused by anchoring to a wrong majority answer.
- Validation: On the gate-off subset, C outperforms B and is not worse than A.

**Criterion 2: Preserve self-reference benefits when confident**
- Hypothesis: When self-solves are unanimous, providing the self-answer reference helps (or at least does not hurt) judging.
- Validation: On the gate-on subset, C matches B (within small tolerance).

---

## Impact Statement

If successful, this provides a simple inference-time rule for practitioners using solve-then-judge evaluation: **use self-reference only when the judge’s own self-solve is consistent**. This can make self-referential judges more reliable when model capacity is limited, improving automated evaluation pipelines used for best-of-N selection and reward signal generation.

---

## References

- [Do Before You Judge: Self-Reference as a Pathway to Better LLM Evaluation](./references/Do-Before-You-Judge-Self-Reference-as-a-Pathway-to-Better-LLM-Evaluation/meta/meta_info.txt) - Lin et al., 2025
- [PhyCritic: Multimodal Critic Models for Physical AI](./references/PhyCritic-Multimodal-Critic-Models-for-Physical-AI/meta/meta_info.txt) - Xiong et al., 2026
- [Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution](./references/Overconfidence-in-LLM-as-a-Judge-Diagnosis-and-Confidence-Driven-Solution/meta/meta_info.txt) - Tian et al., 2025
- [Self-Preference Bias in LLM-as-a-Judge](./references/Self-Preference-Bias-in-LLM-as-a-Judge/meta/meta_info.txt) - Wataoka et al., 2024
- [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) - Kadavath et al., 2022
- [Self-Consistency Improves Chain-of-Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) - Zheng et al., 2023
- [AlpacaEval](https://arxiv.org/abs/2305.14387) - Dubois et al., 2023
- [G-Eval](https://arxiv.org/abs/2303.16634) - Liu et al., 2023
- [FairEval](https://arxiv.org/abs/2305.11747) - Wang et al., 2023
- [CheckEval](https://arxiv.org/abs/2403.18771) - Lee et al., 2024
- [JudgeBench](https://arxiv.org/abs/2410.12784) - Tan et al., 2024
- [ContextualJudgeBench](https://arxiv.org/abs/2503.15620) - (authors), 2025
- [TrustJudge](https://arxiv.org/abs/2509.21117) - (authors), 2025
- [Rating Roulette](https://arxiv.org/abs/2510.27106) - Haldar and Hockenmaier, 2025
- [SelfCheckGPT](https://arxiv.org/abs/2303.08896) - Manakul et al., 2023
- [Self-Evaluation Improves Selective Generation in LLMs](https://arxiv.org/abs/2312.09300) - Ren et al., 2023
- [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) - Zhao et al., 2025
- [BadJudge: Backdoor Vulnerabilities of LLM-as-a-Judge](https://arxiv.org/abs/2503.00596) - Tong et al., 2025
- [LLMs Cannot Reliably Judge (Yet?)](https://arxiv.org/abs/2506.09443) - Li et al., 2025
- [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2409.02692) - (authors), 2024
- [MMLU-Pro Dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) - TIGER-Lab, 2024
