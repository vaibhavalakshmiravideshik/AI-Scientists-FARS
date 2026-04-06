# untitled

# Answer-Free Self-Referential Critics: Training Solve-Then-Judge VLM Judges with Preference Labels but Without Ground-Truth Answers

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Vision-language models (VLMs) are increasingly used in settings where outputs must be **evaluated** rather than only generated: selecting the best of many sampled answers, filtering unsafe or hallucinated outputs, or providing automated supervision signals for reinforcement learning. This motivates **multimodal critic / judge models**: models that take an image/video and a user question, then compare two candidate responses and decide which one is better.

Recent work shows that reinforcement learning on automated signals can substantially improve reasoning-style behaviors. In language-only settings, reinforcement learning with verifiable rewards (RLVR) improves reasoning by rewarding objective correctness (e.g., math answer exact match) rather than imitating a teacher. In multimodal settings, similar RLVR-style approaches are emerging for video/image reasoning.

However, many important critic datasets do **not** provide a single ground-truth answer for each prompt. Instead, they provide only **pairwise preferences** between responses (e.g., (prompt, response A, response B, which is better)). Examples include large-scale multimodal preference datasets such as VLFeedback and MM-RLHF, and evaluation suites such as VL-RewardBench / Multimodal RewardBench (collections of preference pairs used to score judge models by agreement rate). In these settings, a training recipe that requires answer labels cannot be directly applied.

### The Problem

**PhyCritic** proposes a “solve-then-judge” training recipe for physical-domain multimodal critics, where the critic is prompted to first produce its own solution to the question and then use it as a reference to judge candidate responses (**[PhyCritic](./references/PhyCritic-Multimodal-Critic-Models-for-Physical-AI/meta/meta_info.txt)**). In Stage 2, PhyCritic uses an explicit self-prediction accuracy reward:

- Self-prediction reward: \(r_{sp}=\mathbb{1}(\hat{A}_{pred}=A_Q)\), which requires a ground-truth answer \(A_Q\).

This requirement is highlighted as a limitation in the paper (ground-truth answers may be unavailable in open-ended physical scenarios). It also prevents applying the solve-then-judge recipe to **preference-only** critic datasets, where \(A_Q\) is not provided.

A naive substitute is to use only the critic preference reward (judge whether response A or B is better). But PhyCritic’s ablations show this loses accuracy: keeping the self-reference prompt but removing the self-prediction reward \(r_{sp}\) drops PhyCritic-Bench from **68.0→65.8 (-2.2)**, and removing the entire self-referential process drops to **64.4 (-3.6)** (Table 5 in PhyCritic). This suggests that “solve before judging” benefits from an explicit training signal for the self-solve step.

We therefore ask: **can we retain the benefit of self-referential (solve-then-judge) critic training when answer labels are missing, assuming we still have preference labels?**

**Important limitation / realism note.** In this proposal we validate the idea on a *public MCQ benchmark with ground-truth answers* (Cosmos-Reason1) by **simulating** preference-only training pairs where preference is defined as “correct answer is preferred.” This makes preference labels strongly correlated with correctness, unlike many real preference datasets (e.g., helpfulness/style). Our claim is therefore scoped to *correctness-oriented preferences* and MCQ settings where an answer can be reliably extracted from the preferred response.

### Key Insight and Hypothesis

**Key insight.** In preference-only datasets, we often know which response is preferred, and the preferred response frequently contains an **extractable final answer** (especially for multiple-choice questions). This can be used as a weak pseudo-label for the critic’s self-prediction, without requiring a ground-truth answer.

However, preference-derived pseudo-labels are noisy and can encourage shortcut behaviors (e.g., overfitting to option letters). Inspired by option-permutation consistency rewards (used to reduce option-order shortcuts in multimodal RL), we add an invariance regularizer that rewards self-predictions that remain stable under permutations of answer options.

**Hypothesis.** In a solve-then-judge multimodal critic trained with GRPO, replacing ground-truth self-prediction reward \(\mathbb{1}(\hat{A}=A_Q)\) with a preference-derived pseudo-label \(\mathbb{1}(\hat{A}=a_{pref})\) plus option-permutation invariance will recover most of the benefit of self-prediction supervision, improving critic accuracy relative to removing self-prediction reward entirely.

The outcome is uncertain because (i) the preferred response’s answer may be a weak/noisy proxy, and (ii) the invariance term may be redundant with the preference reward and provide little additional training signal.

---

## Proposed Approach

### Overview

We propose **Answer-Free Self-Referential Critic (AF-SRC)** training for multiple-choice physical reasoning, designed for datasets that provide only preference labels.

We train a single VLM to perform two tasks:

1. **Self-solve**: given the visual input and multiple-choice question, predict an answer \(\hat{A}_{pred}\).
2. **Judge**: given the same visual input, the question, two candidate responses, and its own \(\hat{A}_{pred}\), predict which response is better.

Unlike PhyCritic’s single prompt that includes candidate responses before the model emits its own prediction (Table 1 of PhyCritic), we enforce a **two-pass rollout** so that self-prediction cannot copy information from candidate responses.

*(Definitions on first use)* **GRPO** = Group Relative Policy Optimization (a PPO-like policy gradient method without a learned value network). **MCQ** = multiple-choice question. **CoT** = chain-of-thought.

### Method Details

#### A. Data format

We assume training data tuples:
\[(Q, R_A, R_B, P)\]
where \(Q\) is a multimodal question with MCQ options, \(R_A\) and \(R_B\) are candidate responses, and \(P\in\{A,B\}\) indicates which response is preferred.

Importantly, we assume \(A_Q\) (a ground-truth answer) is **not available** to the training algorithm.

#### B. Two-pass solve-then-judge rollout

For each training tuple:

1. **Solve pass**: prompt the model with \(Q\) only (plus the visual input). The model outputs \(\hat{A}_{pred}\) in a strict format (e.g., `\boxed{A}` for MCQ).
2. **Judge pass**: prompt the model with \((Q, R_A, R_B)\) plus its previously produced \(\hat{A}_{pred}\). The model outputs a preference decision in `\boxed{Response 1 is better}` / `\boxed{Response 2 is better}` format.

This ensures the self-prediction is an independent signal.

#### C. Rewards

We use GRPO-style training with a scalar outcome reward composed of:

- **Critic correctness reward** \(r_{crit}=\mathbb{1}(\hat{P}=P)\), where \(\hat{P}\) is the model’s judge output.
- **Format reward** \(r_{form}\) for producing parseable outputs.
- **Self-prediction reward** \(r_{sp}\), which differs by condition.

**Answer-free self-prediction reward (ours).**

Let \(a_{pref}\) be the final answer extracted from the preferred response (the response indicated by \(P\)). Define:

- Preference-derived pseudo-label reward: \(r_{sp\_pref}=\mathbb{1}(\hat{A}_{pred}=a_{pref})\).

Option-permutation invariance regularizer (option-shuffling consistency, as in **ACRE**): create a random permutation of the MCQ options and ask the model to answer again, producing \(\hat{A}_{pred}^{perm}\). Let \(perm^{-1}(\cdot)\) map the permuted label back to the original label.

- Invariance reward: \(r_{inv}=\mathbb{1}(perm^{-1}(\hat{A}_{pred}^{perm})=\hat{A}_{pred})\).

We set:
\[r_{sp}^{AF}= r_{sp\_pref}\cdot r_{inv}.\]

**Oracle and ablation variants.**

- Oracle self-pred reward: \(r_{sp}^{oracle}=\mathbb{1}(\hat{A}_{pred}=A_Q)\).
- No self-pred reward: \(r_{sp}=0\).

Total reward is a weighted sum:
\[r_{total} = \alpha_{crit} r_{crit} + \alpha_{sp} r_{sp} + \alpha_{form} r_{form}.\]
We will use fixed weights (no sweeps) and report them.

### Key Innovations

- **Answer-free self-prediction supervision**: replaces ground-truth answer supervision with a preference-derived pseudo-label, enabling solve-then-judge critic training on preference-only datasets.
- **Two-pass rollout to avoid response leakage**: separates self-solve from judging so self-prediction is not trivially copied from candidate responses.
- **Option-permutation invariance as a regularizer for pseudo-labels**: discourages option-letter shortcuts and encourages semantic option reasoning under weak supervision.

---

## Related Work

### Field Overview

Multimodal critic/judge models are typically trained from pairwise preference data, either via supervised learning (classification of which response is better) or via preference-based objectives such as direct preference optimization (DPO). Separately, reinforcement fine-tuning methods for reasoning models often rely on verifiable rewards, where a ground-truth answer exists and provides an objective correctness signal.

Solve-then-judge (self-referential) critics combine these: the model first solves the problem, then uses that internal solution as a reference to judge responses. This can improve grounding and stability, but it introduces a new supervision requirement: training the self-solve step. When answer labels are missing, training this component becomes nontrivial.

A related direction is **self-supervised / label-reduced RL**, where models learn from consistency across views (e.g., question rephrasing, option permutations) or from internal reward consistency, to reduce reliance on ground-truth labels and prevent training collapse.

### Related Papers

- **[PhyCritic](./references/PhyCritic-Multimodal-Critic-Models-for-Physical-AI/meta/meta_info.txt)**: Introduces solve-then-judge physical-domain critics with GRPO and a ground-truth self-prediction reward; our work removes the ground-truth requirement.
- **[Answer-Consistent CoT RL / ACRE](./references/Answer-Consistent-Chain-of-Thought-Reinforcement-Learning-for-Multi-modal-Large-Language-Models/meta/meta_info.txt)**: Uses option shuffling consistency to reduce option-order shortcuts in multimodal RL; we reuse option-permutation consistency as a regularizer for answer-free self-prediction.
- **[Co-Reward](./references/Co-Reward-Self-supervised-Reinforcement-Learning-for-Large-Language-Model-Reasoning-via-Contrastive-Agreement/meta/meta_info.txt)**: Self-supervised RL via cross-view consistency (original vs rephrased questions); conceptually related consistency-based supervision without labels.
- **[SCIR](./references/Self-Consistency-of-the-Internal-Reward-Models-Improves-Self-Rewarding-Language-Models/meta/meta_info.txt)**: Improves self-rewarding by enforcing consistency among internal reward models; highlights consistency as a reliability mechanism.
- **[LLaVA-Critic-R1](./references/LLaVA-Critic-R1-Your-Critic-Model-is-Secretly-a-Strong-Policy-Model/meta/meta_info.txt)**: Uses GRPO on preference labels to train critics that also improve policy performance; provides context for preference-only RL on VLMs.
- **[Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)**: Uses LLM-as-judge to generate preference data; motivates preference-only training settings.
- **[VLFeedback](https://arxiv.org/abs/2410.09421)**: Large-scale VLM preference dataset with pairwise preferences; an example of preference-only supervision.
- **[MM-RLHF](https://arxiv.org/abs/2502.10391)**: Multimodal RLHF dataset + benchmarks (MM-RewardBench); preference-only supervision setting.
- **[RLHF-V](https://arxiv.org/abs/2312.00849)**: Fine-grained multimodal human feedback dataset; preference-style supervision.
- **[RLAIF-V](https://arxiv.org/abs/2405.17220)**: AI-feedback dataset for multimodal alignment; preference-style supervision.
- **[VL-RewardBench](https://arxiv.org/abs/2411.17451)**: Benchmark for vision-language reward/judge models; preference labels without requiring a single answer for every item.
- **[Multimodal RewardBench](https://arxiv.org/abs/2502.14191)**: Benchmark for multimodal reward models across domains.
- **[UnifiedReward-Think](https://arxiv.org/abs/2505.03318)**: Multimodal chain-of-thought reward model trained with reinforcement fine-tuning; related to improving judges.
- **[Critic-RM](https://arxiv.org/abs/2411.16646)**: Improves reward modeling via self-generated critiques; connects reasoning traces to judging.
- **[Critique-out-Loud Reward Models](https://arxiv.org/abs/2408.11791)**: Generates critiques before reward prediction; related to explicit reasoning in judges.
- **[DPO](https://arxiv.org/abs/2305.18290)**: Preference optimization method widely used for alignment; baseline family for preference-only training.
- **[PPO](https://arxiv.org/abs/1707.06347)**: Standard RL algorithm for policy optimization; GRPO is PPO-like.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Popularized large-scale RLVR with GRPO-like optimizers; context for RL-based reasoning training.
- **[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)**: Introduces GRPO in math reasoning; algorithmic foundation.
- **[Learning to Reason Without External Rewards](https://arxiv.org/abs/2505.19590)**: Studies reasoning improvement without external verifiers; related to label-reduced training.
- **[Maximizing Confidence Alone Improves Reasoning](https://arxiv.org/abs/2505.22660)**: Self-reward signal based on confidence; motivates careful design to avoid collapse.
- **[Self-Consistency Improves Chain-of-Thought Reasoning](https://arxiv.org/abs/2203.11171)**: Consistency over multiple samples; conceptual antecedent.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Solve-then-judge critics | Judge grounds evaluation in its own solution | PhyCritic, Critic-RM | Physical critics; reward benchmarks | Often needs answer labels for self-solve |
| Consistency-regularized RL | Use cross-view invariance to avoid shortcuts / collapse | ACRE, Co-Reward, Self-Consistency of Internal Rewards | MCQ reasoning; math/code reasoning | Often limited to formats where “views” are easy |
| Preference-only critic RL | Train critics directly from preference labels via RL | LLaVA-Critic-R1, MM-RLHF | RewardBench-style | No explicit supervision for “self-solve” |
| Preference optimization (non-RL) | Learn from pairwise preferences without rollouts | DPO, IPO/ORPO family | General alignment | Can be brittle; no explicit reasoning grounding |

### Closest Prior Work

1. **PhyCritic**: Closest in spirit (solve-then-judge + GRPO) but requires answer labels for r_sp; we replace r_sp with a preference-derived pseudo-label and invariance.
2. **ACRE**: Uses option shuffling consistency but targets reasoning–answer mismatch in MCQ RL, and still uses correctness; we use option permutation as a regularizer for answer-free self-solve supervision.
3. **LLaVA-Critic-R1**: Uses preference-only GRPO for critics but does not include an explicit self-solve head supervised without answers; our contribution is specifically about answer-free self-solve supervision inside solve-then-judge.
4. **Co-Reward**: Provides label-free consistency supervision via rephrasing in text-only reasoning; we adapt the “cross-view consistency” concept to multimodal MCQ and preference-derived pseudo-labels.
5. **SCIR**: Enforces internal reward consistency to improve self-rewarding; complementary to our focus on self-solve supervision.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| PhyCritic | Solve-then-judge critic RL with GT self-pred reward | Needs A_Q | Replace r_sp with preference-derived pseudo-label + invariance | Enables solve-then-judge on preference-only datasets |
| ACRE | Option-shuffling consistency reward for MCQ RL | Targets CoT–answer mismatch; still uses correctness | Use option permutation as regularizer for weak pseudo-label supervision | Reduces option-letter shortcuts under weak supervision |
| LLaVA-Critic-R1 | GRPO on preference labels for critics | No explicit self-solve supervision | Add a self-solve step + answer-free supervision | Better grounding of judgments via self-reference |
| Co-Reward | Cross-view self-supervised RL via rephrasing | Text-only; needs paraphrases | Use option permutation “views” in multimodal MCQ | Avoids needing paraphrasers; cheap invariance |
| SCIR | Enforce internal reward consistency | Does not address self-solve supervision | Focus on supervising self-solve without answers | Directly targets solve-then-judge bottleneck |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-VL-Instruct | 7B | https://huggingface.co/Qwen | Use the closest available Qwen2.5-VL-7B-Instruct checkpoint |

**Training Data (constructed, fully automated):**

We construct a *synthetic preference-only* dataset from Cosmos-Reason1-Benchmark:

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Cosmos-Reason1-Benchmark | Source of MCQ physical reasoning prompts (Q) | 610 benchmark items | https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark | CC-BY-4.0 |

Construction procedure:
1. For each item, sample multiple candidate responses (e.g., 4–8) from a small pool of VLMs / temperatures.
2. Extract each candidate’s final answer (A/B/C/D).
3. Use the ground-truth answer \(A_Q\) **only to construct pairs** where one response is correct and one is incorrect, then set the preference label \(P\) to prefer the correct response.
4. The training algorithm then sees only \((Q, R_A, R_B, P)\). \(A_Q\) is used only for oracle baseline A and for evaluation.

**Resource Estimate** (must fit ≤768 GPU-hours):
- We target ~2k–5k training pairs (by sampling multiple responses per question).
- GRPO training for 3 main conditions (A/B/C), with group size 4 rollouts per prompt.
- Extra cost for invariance check is similar to ACRE’s second forward pass; ACRE reports +24% GPU-hours (4.5→5.6) on 4.6k examples for Qwen2.5-VL.
- Conservative estimate on A100s: ≤40 GPU-hours per condition (including data generation + training + evaluation), total ≤120 GPU-hours.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Synthetic preference dataset (Cosmos-Reason1-derived) | Pairwise judging of correct vs incorrect responses | Preference accuracy (%) | test | Derived from Cosmos-Reason1-Benchmark | Custom script (simple parsing + label compare) |
| Cosmos-Reason1-Benchmark (policy view) | MCQ physical reasoning from video/image | Accuracy (%) | test | https://huggingface.co/datasets/nvidia/Cosmos-Reason1-Benchmark | https://github.com/nvidia-cosmos/cosmos-reason1 |

### Main Results

#### Results Table

All results below are directly comparable (same base model, same evaluation scripts). Baseline numbers are marked TBD when they require re-running.

| Method | Base Model | Benchmark | Preference Acc (%) | MCQ Acc (%) | Source | Notes |
|---|---|---|---:|---:|---|---|
| B: No self-pred reward | Qwen2.5-VL-7B | Synthetic preference | **TBD** | **TBD** | - | Needs re-run |
| A: Oracle self-pred (uses A_Q) | Qwen2.5-VL-7B | Synthetic preference | **TBD** | **TBD** | - | Upper bound (needs A_Q) |
| C: **AF-SRC (ours)** | Qwen2.5-VL-7B | Synthetic preference | **TBD** | **TBD** | - | Answer-free r_sp* |

Reference point (policy accuracy; context only): PhyCritic reports Qwen2.5-VL-7B achieves 54.3% overall on CosmosReason1-Bench (Table 3 in PhyCritic). This is not directly comparable to our post-RL runs but provides a sanity-check scale.

### Ablation Studies

If budget allows, add one ablation to test whether invariance contributes beyond preference-derived pseudo-labels:

| Variant | What’s changed | Expected finding |
|---|---|---|
| C without invariance | r_sp = I(Âpred==a_pref) only | If invariance matters, performance drops vs full C |

### Analysis (Optional)

- **Pseudo-label noise analysis**: fraction of samples where preferred response has no parseable final answer; fraction where both responses share same extracted answer.
- **Shortcut diagnostics**: evaluate option-order robustness (ACRE-style OSCR) for the self-solve step.

---

## Success Criteria

**Criterion 1: Answer-free self-prediction reward provides nontrivial training signal**
- Hypothesis: Oracle self-pred (A) improves preference accuracy over removing self-pred (B).
- Validation: If A does not outperform B by at least 3 accuracy points, we conclude the self-prediction reward is not a meaningful lever in this setting and we abort interpretation.

**Criterion 2: Answer-free reward recovers most of oracle benefit**
- Hypothesis: AF-SRC (C) substantially closes the A–B gap.
- Validation: C improves over B by at least 2 points and achieves recovery ratio \((C-B)/(A-B)\ge 0.75\) on preference accuracy.

---

## Impact Statement

If successful, this work provides a practical way to train solve-then-judge multimodal critics on the large and growing set of **preference-only** datasets, without requiring expensive ground-truth answers. This could improve the reliability of VLM judges used for best-of-N selection, automated evaluation, and preference-based post-training in domains where correctness is not uniquely defined.

---

## References

- [PhyCritic](./references/PhyCritic-Multimodal-Critic-Models-for-Physical-AI/meta/meta_info.txt) - Xiong et al., 2026
- [Co-Reward](./references/Co-Reward-Self-supervised-Reinforcement-Learning-for-Large-Language-Model-Reasoning-via-Contrastive-Agreement/meta/meta_info.txt) - Zhang et al., 2025
- [Answer-Consistent CoT RL / ACRE](./references/Answer-Consistent-Chain-of-Thought-Reinforcement-Learning-for-Multi-modal-Large-Language-Models/meta/meta_info.txt) - Huang et al., 2025
- [Self-Consistency of the Internal Reward Models Improves Self-Rewarding Language Models](./references/Self-Consistency-of-the-Internal-Reward-Models-Improves-Self-Rewarding-Language-Models/meta/meta_info.txt) - Zhou et al., 2025
- [LLaVA-Critic-R1](./references/LLaVA-Critic-R1-Your-Critic-Model-is-Secretly-a-Strong-Policy-Model/meta/meta_info.txt) - Wang et al., 2025
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) - Yuan et al., 2024
- [VLFeedback](https://arxiv.org/abs/2410.09421) - (dataset paper), 2024
- [MM-RLHF](https://arxiv.org/abs/2502.10391) - Zhang et al., 2025
- [RLHF-V](https://arxiv.org/abs/2312.00849) - Yu et al., 2023
- [RLAIF-V](https://arxiv.org/abs/2405.17220) - Yu et al., 2024
- [VL-RewardBench](https://arxiv.org/abs/2411.17451) - Li et al., 2024
- [Multimodal RewardBench](https://arxiv.org/abs/2502.14191) - Yasunaga et al., 2025
- [UnifiedReward-Think](https://arxiv.org/abs/2505.03318) - Wang et al., 2025
- [Critic-RM](https://arxiv.org/abs/2411.16646) - Yu et al., 2025
- [Critique-out-Loud Reward Models](https://arxiv.org/abs/2408.11791) - (arXiv), 2024
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - DeepSeek-AI, 2025
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [Learning to Reason Without External Rewards](https://arxiv.org/abs/2505.19590) - Zhao et al., 2025
- [Maximizing Confidence Alone Improves Reasoning](https://arxiv.org/abs/2505.22660) - Prabhudesai et al., 2025
- [Self-Consistency Improves Chain-of-Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
