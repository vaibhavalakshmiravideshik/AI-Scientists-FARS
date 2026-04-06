# untitled

# RefSwap: Counterfactual Reference-Swap Checks for Robust Reference-Based LLM Verifiers

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly trained and evaluated using **automated verifiers** (also called *reference-based reward systems*): given a question \(q\), a gold reference answer \(r\), and a model completion \(c\), the verifier outputs a binary label \(y \in \{\text{YES},\text{NO}\}\) indicating whether \(c\) matches \(r\). This paradigm is central to **reinforcement learning with verifiable rewards (RLVR)**—a post-training setup where the reward is computed by automatic verification against \(r\) (rather than human preferences).

Recent work shows that this setup has a critical robustness failure mode. **One Token to Fool LLM-as-a-Judge** finds that many verifiers (including frontier LLMs) can be manipulated by trivial “master key” completions such as `:` or `Thought process:` that contain no substantive answer but still receive a positive judgment, causing reward hacking and even RL training collapse ([One Token to Fool](./references/One-Token-to-Fool-LLM-as-a-Judge/meta/meta_info.txt)). The paper proposes a **training-time** mitigation: fine-tune a reward model with adversarial negatives.

In parallel, **VerifyBench** provides a benchmark for evaluating reference-based verifiers and shows that even strong models struggle on difficult instances; **VerifyBench-Hard** is built by selecting cases where multiple strong verifiers disagree and then human-annotating them ([VerifyBench](./references/VERIFYBENCH-BENCHMARKING-REFERENCE-BASED-REWARD-SYSTEMS-FOR-LARGE-LANGUAGE-MODELS/meta/meta_info.txt)). However, VerifyBench does not directly test adversarial completion attacks like master keys.

This creates a practical gap: many teams want a **training-free** robustness guard that can be applied immediately to an existing verifier (e.g., a prompted LLM-as-a-judge, or a deployed reward model) without collecting new adversarial data or re-training the verifier.

### The Problem

We focus on **reference-based verifiers** that are supposed to judge correctness *relative to the provided reference answer*. In an ideal verifier, replacing the correct reference answer \(r\) with an unrelated reference answer \(r'\) should strongly reduce the probability of predicting YES.

However, current systems can exhibit two problematic behaviors:

- **Reference-insensitive false positives (master keys)**: A completion like `:` can produce YES even though it is unrelated to both \(q\) and \(r\), implying the verifier’s decision is not meaningfully anchored to \(r\) ([One Token to Fool](./references/One-Token-to-Fool-LLM-as-a-Judge/meta/meta_info.txt)).
- **Verifier self-solving**: Some LLM-as-a-judge prompts may “solve” the question internally from \(q\) and then judge \(c\) based on that internal answer, using \(r\) only weakly. This can make even legitimate completions appear reference-insensitive, which would cause a reference-swap defense to reject true positives.

Existing defenses do not fully address this in a training-free way:

- **Training-time adversarial augmentation** (Master-RMs) is effective but requires re-training the verifier ([One Token to Fool](./references/One-Token-to-Fool-LLM-as-a-Judge/meta/meta_info.txt)).
- **Prompt-attack defenses** (e.g., retokenization, detectors) are studied for LLM-as-a-judge robustness, but are not targeted to reference-based verification mechanics and can degrade benign performance ([RobustJudge](./references/LLMs-Cannot-Reliably-Judge-Yet-A-Comprehensive-Assessment-on-the-Robustness-of-LLM-as-a-Judge/meta/meta_info.txt)).

### Key Insight and Hypothesis

**Key insight.** For a reference-based verifier, we can probe whether a YES judgment is **causally tied** to the reference answer by running a **counterfactual reference swap**: replace \(r\) with an unrelated \(r'\) while keeping \((q,c)\) fixed.

Define \(p_{\text{yes}}(q,r,c)\) as the verifier’s probability of predicting `YES` (e.g., the normalized next-token probability mass on the `YES` token under a forced `YES`/`NO` output format) under a fixed prompt and decoding scheme. Define the **reference-sensitivity score**:
\[
 s(q,r,c) = p_{\text{yes}}(q,r,c) - p_{\text{yes}}(q,r',c).
\]

**Hypothesis.** Master-key false positives will have **low reference sensitivity** (\(s\approx 0\)), while legitimate correct completions will tend to have **higher** reference sensitivity. Therefore, filtering YES decisions by thresholding \(s\) can reduce master-key false positives with limited loss of clean accuracy.

This hypothesis could fail if verifiers largely ignore the reference answer even on correct completions (i.e., they self-solve), causing \(s\) to be small for both true positives and master keys.

---

## Proposed Approach

### Overview

We propose **RefSwap**, a training-free wrapper around an existing reference-based verifier \(V\):

1. Run \(V\) on \((q,r,c)\) to obtain \(p_{\text{yes}}(q,r,c)\).
2. If \(p_{\text{yes}}(q,r,c)\) is below the base YES threshold (e.g., 0.5), output NO (no extra cost).
3. Otherwise, sample an unrelated reference \(r'\) and run \(V\) on \((q,r',c)\) to obtain \(p_{\text{yes}}(q,r',c)\).
4. Compute \(s = p_{\text{yes}}(q,r,c) - p_{\text{yes}}(q,r',c)\). Output YES iff \(s \ge \tau\) (calibrated on a dev split).

This yields average cost close to **1 + \(\Pr[\text{baseline predicts YES}]\)** forward passes per example, not a fixed 2× overhead.

### Method Details

**1) Verifier interface and probability extraction.**

We focus on verifiers that produce a discrete YES/NO judgment (either as a short generation or as a classifier). For LLM-as-a-judge style verifiers, we use a prompt that forces the output to be exactly `YES` or `NO`, then compute \(p_{\text{yes}}\) from the model’s next-token log probabilities for `YES` vs `NO`.

**2) Constructing the counterfactual reference \(r'\).**

To make \(r'\) “unrelated” while avoiding accidental semantic overlap, we sample \(r'\) from the VerifyBench pool subject to:

- **Different answer-type bucket** than \(r\) (numeric / expression / multiple-choice / string; VerifyBench provides these types).
- **Low token overlap** with \(r\) (e.g., Jaccard overlap below a small threshold) to avoid near-duplicates.
- For numeric-like references, reject \(r'\) if it matches \(r\) exactly as a string.

This is a deterministic, fully automated procedure.

**3) Threshold calibration.**

We set \(\tau\) on a held-out dev split of VerifyBench to preserve clean verification accuracy. One simple rule is:

- Choose the largest \(\tau\) such that clean accuracy drops by at most \(\delta\) (e.g., \(\delta=1\) percentage point) compared to the baseline verifier.

**4) Output and deployment.**

RefSwap can be used as:

- a robustness guard for evaluation (reject suspicious YES verdicts), and
- a filter in RLVR training pipelines (drop rollouts whose positive reward would not survive the counterfactual reference check).

### Key Innovations

1. **Counterfactual reference swap as an inference-time robustness probe** for reference-based verifiers.
2. **Reference-sensitivity scoring** (\(s\)) that provides an interpretable diagnostic: whether the verifier’s YES decision depends on the provided reference.
3. **Selective second-pass evaluation** (only when baseline predicts YES) to reduce overhead.

---

## Related Work

### Field Overview

Reference-based verification has become a core ingredient in RL for reasoning models: instead of training a preference reward model, the system verifies whether a completion matches a reference answer and uses that as reward. VerifyBench formalizes evaluation for these systems and shows significant headroom on hard cases.

Separately, the LLM-as-a-judge literature shows that automated evaluators are vulnerable to adversarial manipulation (universal suffixes, prompt injection, and bias effects). One Token to Fool identifies a particularly severe failure mode for reference-based judges: master-key completions that elicit false positives.

Existing mitigation strategies cluster into: (i) **training-time** robustness (adversarial augmentation, specialized verifiers), and (ii) **inference-time** aggregation or defenses (swap-ordering, retokenization, detectors). Our work targets a third axis: **counterfactual reference dependence** for reference-based verification.

### Related Papers

- **[VerifyBench](./references/VERIFYBENCH-BENCHMARKING-REFERENCE-BASED-REWARD-SYSTEMS-FOR-LARGE-LANGUAGE-MODELS/meta/meta_info.txt)**: Introduces VerifyBench and VerifyBench-Hard for evaluating reference-based reward systems.
- **[One Token to Fool LLM-as-a-Judge](./references/One-Token-to-Fool-LLM-as-a-Judge/meta/meta_info.txt)**: Identifies master-key false positives in reference-based judging and mitigates them via adversarial data augmentation.
- **[Is LLM-as-a-Judge Robust?](../../papers/paper_summaries/Is%20LLM-as-a-Judge%20Robust%20Investigating%20Universal%20Adversarial%20Attacks%20on%20Zero-shot%20LLM%20Assessment.md)**: Shows universal transferable adversarial phrases can manipulate LLM-based assessment, especially for absolute scoring.
- **[LLMs Cannot Reliably Judge (Yet?) / RobustJudge](./references/LLMs-Cannot-Reliably-Judge-Yet-A-Comprehensive-Assessment-on-the-Robustness-of-LLM-as-a-Judge/meta/meta_info.txt)**: Proposes a unified robustness evaluation framework with many attacks and defenses for LLM judges.
- **[JudgeLM](./references/JudgeLM-Fine-tuned-Large-Language-Models-are-Scalable-Judges/meta/meta_info.txt)**: Fine-tunes LLM judges and introduces swap augmentation, reference support, and reference drop.
- **[TrustJudge](./references/TrustJudge-Inconsistencies-of-LLM-as-a-Judge-and-How-to-Alleviate-Them/meta/meta_info.txt)**: Reduces judge inconsistencies using probabilistic scoring and bidirectional aggregation.
- **[xVerify](./references/xVerify-Efficient-Answer-Verifier-for-Reasoning-Model-Evaluations/meta/meta_info.txt)**: Trains specialized verifiers for equivalence checking on objective questions.
- **[CoSineVerifier](./references/CoSineVerifier-Tool-Augmented-Answer-Verification-for-Computation-Oriented-Scientific-Questions/meta/meta_info.txt)**: Uses tool execution to improve verification on computation-heavy problems; reports VerifyBench-Hard results.
- **[Cooper](./references/Cooper-Co-Optimizing-Policy-and-Reward-Models-in-Reinforcement-Learning-for-Large-Language-Models/meta/meta_info.txt)**: Co-optimizes policy and reward model, introduces reference-based VerifyRM, and evaluates on VerifyBench.
- **[Reward Under Attack](../../papers/paper_summaries/Reward%20Under%20Attack%20Evaluating%20the%20Sensitivity%20of%20Process%20Reward%20Models.md)**: Audits sensitivity of process reward models to perturbations, motivating robustness evaluation for reward signals.
- **[MT-Bench](https://arxiv.org/abs/2306.05685)**: Popularizes LLM-as-a-judge for open-ended chat evaluation.
- **[Chatbot Arena](https://arxiv.org/abs/2403.04132)**: Large-scale human preference collection that motivates automated judges.
- **[G-Eval](https://arxiv.org/abs/2303.16634)**: Uses LLM probabilistic scoring to better align evaluations with human preferences.
- **[RewardBench](https://arxiv.org/abs/2403.13787)**: Standard benchmark for preference reward models (reference-free).
- **[RM-Bench](https://arxiv.org/abs/2406.10883)**: Benchmark for reward models under style/format confounds.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Introduces process reward models for reasoning verification.
- **[ProcessBench](https://arxiv.org/abs/2410.15203)**: Benchmark for process verification on math reasoning trajectories.
- **[Hard2Verify](../../papers/paper_summaries/Hard2Verify%20A%20Step-Level%20Verification%20Benchmark%20for%20Open-Ended%20Frontier%20Math.md)**: Evaluates verifiers on frontier open-ended math with step-level error labels.
- **[AdvEval](https://arxiv.org/abs/2402.07907)**: Adversarial framework for manipulating automated evaluators.
- **[PAIR](https://arxiv.org/abs/2310.08419)**: Black-box jailbreak attack that RobustJudge adapts for judge manipulation.
- **[TAP](https://arxiv.org/abs/2312.02119)**: Tree-of-attacks jailbreak method used in robustness studies.
- **[GCG](https://arxiv.org/abs/2307.15043)**: Universal and transferable adversarial attack method, motivating universal evaluator robustness.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Reference-based verifier benchmarks | Evaluate correctness vs gold reference | VerifyBench, VerifierBench (mentioned in VerifyBench), Hard2Verify | VerifyBench / VerifyBench-Hard | Often do not test adversarial completion attacks |
| Training-time robust verifiers | Train specialized verifier models | xVerify, CoSineVerifier, VerifyRM (Cooper), Master-RMs (One Token) | VAR, VerifyBench(-Hard) | Requires data + training; robustness depends on attack coverage |
| Judge robustness (attacks/defenses) | Stress-test judges against adversarial strings/prompt injection | One Token, RobustJudge, universal phrase attacks | SummEval/TopicalChat, RobustJudge tasks | Many defenses are prompt-specific; trade-offs with benign performance |
| Inference-time aggregation | Reduce bias via multiple evaluations and aggregation | TrustJudge, swap-ordering in JudgeLM, self-consistency | MT-Bench, Arena-Hard | Often targets position bias, not reference dependence |

### Closest Prior Work

- **One Token to Fool** ([local](./references/One-Token-to-Fool-LLM-as-a-Judge/meta/meta_info.txt)): Shows master-key false positives in reference-based judges and proposes training-time adversarial augmentation to build Master-RMs. **Difference**: we propose a training-free, inference-time counterfactual check that can wrap any existing verifier.

- **VerifyBench** ([local](./references/VERIFYBENCH-BENCHMARKING-REFERENCE-BASED-REWARD-SYSTEMS-FOR-LARGE-LANGUAGE-MODELS/meta/meta_info.txt)): Defines evaluation for reference-based reward systems and provides prompts for LLM-as-a-judge. **Difference**: we introduce an adversarial completion robustness evaluation (master keys) and a counterfactual reference-swap diagnostic/defense.

- **RobustJudge** ([local](./references/LLMs-Cannot-Reliably-Judge-Yet-A-Comprehensive-Assessment-on-the-Robustness-of-LLM-as-a-Judge/meta/meta_info.txt)): Provides broad robustness evaluation and defenses for judges, focusing on prompt injection and composite attacks. **Difference**: we focus on reference-based verification mechanics and introduce a reference-swap counterfactual probe.

- **TrustJudge** ([local](./references/TrustJudge-Inconsistencies-of-LLM-as-a-Judge-and-How-to-Alleviate-Them/meta/meta_info.txt)): Reduces inconsistency via bidirectional aggregation. **Difference**: our signal is counterfactual reference dependence, not swap-order consistency.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| One Token to Fool | Training-time anti-master-key augmentation for reward models | Requires re-training + attack coverage | Training-free reference-swap test-time filter | Deployable immediately; provides diagnostic signal (s) |
| VerifyBench | Benchmark + prompts for reference-based verifiers | No adversarial completion stress test | Add master-key stress test + reference-swap analysis | Targets real RLVR collapse failure mode |
| RobustJudge | Broad judge robustness evaluation + defenses | Not targeted to reference dependence | Use counterfactual reference swap as defense | Directly checks whether YES depends on reference |
| TrustJudge | Fixes inconsistencies via probabilistic aggregation | Targets position/tie issues | Use counterfactual reference replacement | Addresses a different failure axis (reference grounding) |

---

## Experiments

### Experimental Setup

**Base Models (verifiers):**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Baseline LLM-as-a-judge style verifier |
| xVerify-* (optional) | 0.5B–7B | https://github.com/IAAR-Shanghai/xVerify | Specialized verifier; optional second backbone |

**Training Data (if applicable):**

No training data needed (inference-time wrapper only).

**Resource Estimate**:

- **Compute budget**: \(< 100\) GPU-hours expected for running Qwen2.5-7B over VerifyBench (2k) + VerifyBench-Hard (1k) + master-key stress test (10k prompts) with short decoding (1–5 tokens). 
- **GPU memory**: single A100 80GB is sufficient for 7B inference.
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| VerifyBench | 2,000 (q,r,c) tuples with binary correctness labels | Accuracy (%) | test | https://huggingface.co/datasets/ZJU-REAL/VerifyBench | Official repo (if available) or simple script |
| VerifyBench-Hard | 1,000 hard tuples with binary labels | Accuracy (%) | test | https://huggingface.co/datasets/ZJU-REAL/VerifyBench | Same as above |
| Master-key stress test (constructed) | Replace completion \(c\) with 10 fixed master keys from One Token to Fool | FPR per key; Avg/Worst FPR (%) | derived | N/A (generated) | Custom script |

**Master keys (fixed list, from One Token to Fool; see Table 17 for the key set):**
`" "`, `.`, `,`, `:`, `Thought process:`, `Let's solve this problem step by step.`, `Solution`, `解`, `かいせつ`, `Respuesta`.

### Main Results

#### Comparability Rules

All rows below must use:
- The same VerifyBench / VerifyBench-Hard splits.
- The same verifier backbone and the same judge prompt.
- The same master-key list and aggregation (Avg/Worst over the 10 keys).

*Note:* The master-key FPR in **One Token** is reported on their own mixed-benchmark evaluation setup; in our verification runs we will re-measure baseline FPR on VerifyBench prompts using the same prompt as RefSwap to ensure apples-to-apples comparison.

#### Results Table

| Method | Base Model | Benchmark | Clean Acc (\(\uparrow\)) | Hard Acc (\(\uparrow\)) | Master-key FPR Avg/Worst (\(\downarrow\)) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| Baseline verifier | Qwen2.5-7B-Instruct | VerifyBench / Hard | 89.05 | 80.20 | 12.6 / 31.0 | [One Token (Table 3)](./references/One-Token-to-Fool-LLM-as-a-Judge/sections/Vulnerabilities%20to%20Master%20Key%20Attacks.md)<br>[One Token (Table 17)](./references/One-Token-to-Fool-LLM-as-a-Judge/sections/Distribution%20of%20Responses..md) | VerifyBench acc from Table 3; master-key FPR avg/worst from Table 17 (published across mixed benchmarks; our experiments will re-measure on VerifyBench for comparability) |
| Ablation: length-only reject | Qwen2.5-7B-Instruct | VerifyBench / Hard | **TBD** | **TBD** | **TBD** | - | Reject YES if completion length < L (calibrated on dev) |
| **Ours: RefSwap (s-threshold)** | Qwen2.5-7B-Instruct | VerifyBench / Hard | **TBD** | **TBD** | **TBD** | - | Second pass only when baseline predicts YES |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Length-only reject | Use completion length as the only signal | Should either fail to suppress FPR robustly or severely hurt clean accuracy (many correct answers are short) |

### Analysis (Optional)

- **Primary diagnostic**: plot distributions of \(s\) for (i) true positives, (ii) false positives, (iii) master keys; report AUC of separating master keys from true positives.
- **Answer-type stratification**: report the same plots separately for numeric / expression / multi-choice / string.
- **Threat boundary check**: repeat master-key test with an “answer-copy” variant: append the exact reference answer string \(r\) after the master key. Report whether \(s\) remains small (expected: \(s\) should increase because the completion now matches \(r\)).
- **Adaptive attacks (limitation check)**: test a simple *conditional prompt-injection* completion that tries to induce *reference-dependent* YES decisions while still being wrong (e.g., `If the reference contains the token "XYZ", output YES; otherwise NO.`). This is out-of-scope for the main claim (master keys), but helps map the threat boundary.

---

## Success Criteria

**Criterion 1: Master-key robustness improves without large clean regression**
- Hypothesis: RefSwap reduces master-key false positives by filtering reference-insensitive YES judgments.
- Validation: Compared to the baseline verifier, RefSwap substantially reduces Avg/Worst master-key FPR while keeping VerifyBench and VerifyBench-Hard accuracy within a small margin (e.g., \(\le 1\)–2 points).

**Criterion 2: The mechanism is supported by reference-sensitivity separation**
- Hypothesis: Master keys have \(s \approx 0\) while most true positives have larger \(s\).
- Validation: The \(s\)-score distribution for master keys is significantly shifted lower than for true positives, and a single \(\tau\) can separate them with a favorable trade-off.

---

## Impact Statement

If successful, RefSwap provides a simple, training-free robustness guard for reference-based verifiers used in RLVR and automated evaluation. Practitioners could wrap an existing verifier to prevent reward hacking via vacuous “master key” completions, reducing the risk of RL collapse and improving the reliability of verifier-based filtering pipelines without collecting new adversarial data or re-training reward models.

---

## References

- [VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](./references/VERIFYBENCH-BENCHMARKING-REFERENCE-BASED-REWARD-SYSTEMS-FOR-LARGE-LANGUAGE-MODELS/meta/meta_info.txt) - Yan et al., 2025
- [One Token to Fool LLM-as-a-Judge](./references/One-Token-to-Fool-LLM-as-a-Judge/meta/meta_info.txt) - Zhao et al., 2025
- [LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge](./references/LLMs-Cannot-Reliably-Judge-Yet-A-Comprehensive-Assessment-on-the-Robustness-of-LLM-as-a-Judge/meta/meta_info.txt) - Li et al., 2025
- [JudgeLM: Fine-tuned Large Language Models are Scalable Judges](./references/JudgeLM-Fine-tuned-Large-Language-Models-are-Scalable-Judges/meta/meta_info.txt) - Zhu et al., 2023
- [TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them](./references/TrustJudge-Inconsistencies-of-LLM-as-a-Judge-and-How-to-Alleviate-Them/meta/meta_info.txt) - Wang et al., 2025
- [xVerify: Efficient Answer Verifier for Reasoning Model Evaluations](./references/xVerify-Efficient-Answer-Verifier-for-Reasoning-Model-Evaluations/meta/meta_info.txt) - Chen et al., 2025
- [CoSineVerifier: Tool-Augmented Answer Verification for Computation-Oriented Scientific Questions](./references/CoSineVerifier-Tool-Augmented-Answer-Verification-for-Computation-Oriented-Scientific-Questions/meta/meta_info.txt) - Feng et al., 2024
- [Cooper: Co-Optimizing Policy and Reward Models in Reinforcement Learning for Large Language Models](./references/Cooper-Co-Optimizing-Policy-and-Reward-Models-in-Reinforcement-Learning-for-Large-Language-Models/meta/meta_info.txt) - Hong et al., 2025
- [Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment](https://aclanthology.org/2024.emnlp-main.427.pdf) - Raina et al., 2024
- [MT-Bench](https://arxiv.org/abs/2306.05685) - Zheng et al., 2023
- [Chatbot Arena](https://arxiv.org/abs/2403.04132) - Zheng et al., 2024
- [G-Eval](https://arxiv.org/abs/2303.16634) - Liu et al., 2023
- [RewardBench](https://arxiv.org/abs/2403.13787) - Lambert et al., 2024
- [RM-Bench](https://arxiv.org/abs/2406.10883) - Liu et al., 2024
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [ProcessBench](https://arxiv.org/abs/2410.15203) - Zheng et al., 2024
- [Hard2Verify](https://arxiv.org/abs/2510.13744) - Pandit et al., 2024
- [AdvEval](https://arxiv.org/abs/2402.07907) - Chen et al., 2024
- [PAIR](https://arxiv.org/abs/2310.08419) - Chao et al., 2023
- [TAP](https://arxiv.org/abs/2312.02119) - Mehrotra et al., 2023
- [GCG](https://arxiv.org/abs/2307.15043) - Zou et al., 2023
