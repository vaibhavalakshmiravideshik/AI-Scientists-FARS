# untitled

# Action-Support Likelihood Audits Predict Rollout Consistency Failures in Text-Based World Models

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used as **interactive agents** in text environments: the agent reads a textual observation, outputs a textual action, and receives the next observation from an environment. A promising way to make agent development cheaper and safer is to train an LLM to act as a **text-based world model**: a model that predicts the next environment observation conditioned on the interaction history and a candidate action.

**From Word to World** (Word2World) provides an evaluation framework for text world models across environments such as TextWorld and SciWorld, and introduces a long-horizon rollout fidelity metric, the **Consistency Ratio (CR)**, defined as **CR = W2R / Real**, where **Real** is an agent’s success rate in the real environment and **W2R** (“world-to-real”) is the success rate when replaying an action sequence generated inside the world model back in the real environment (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/Metrics.md`). Word2World shows that CR can drop substantially under **agent policy shift** (using a different acting agent than the one that collected training trajectories), especially for weaker agents (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/How does behavior shift affect consistency.md`).

In practical workflows, a world model is used not only for evaluation but also for tasks like planning, synthetic data generation, and action verification. In these settings, users need an automated way to answer: **when should we trust a particular world-model rollout enough to replay it, store it as synthetic data, or plan with it?**

### The Problem

Word2World’s CR is computed **after** running an acting agent inside the world model and then replaying the resulting action sequence in the real environment. This makes CR a valuable metric but a poor **pre-screening signal**: it cannot be used to filter low-quality rollouts before incurring the full cost of replay or downstream training.

A natural idea from **offline reinforcement learning (offline RL)** is that failures under distribution shift often occur when a learned policy takes actions **outside the support** of the behavior policy that generated the training data. Offline RL studies this because the policy is trained from a fixed dataset without online exploration: value functions and dynamics models can become unreliable for out-of-distribution actions.

Offline RL therefore studies “support constraints” and behavior-policy likelihood estimates that either (i) explicitly constrain the learned policy to stay within the dataset’s action distribution, or (ii) penalize state-action pairs believed to be out-of-support (e.g., CQL, BRAC, MOReL, OSC). However, text world models evaluated by world-to-real (W2R) transfer and Consistency Ratio (CR) are not typically equipped with a support-based diagnostic that operates on *world-model rollouts*. 

We focus on the setting that Word2World highlights as problematic: **TextWorld under a weak agent (e.g., GPT-4o-mini)**, where Real success is high but W2R success is much lower. For the Qwen2.5-7B world model on TextWorld, Word2World reports Real=97.44 and W2R=69.36 for GPT-4o-mini (CR=0.71), indicating many episodes where the agent can solve the task in the real environment but the action sequence found inside the world model does not transfer (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/5.2 Rollout Consistency.md`).

### Key Insight and Hypothesis

**Key insight:** A text world model is trained on trajectories collected by a specific behavior policy (Word2World uses GPT-4o for data collection; `./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/Data.md`). Under policy shift, world-model rollouts can drift into regions where the **(state, action) patterns implied by the rollout are outside the training behavior distribution**. When this happens, the world model may still generate fluent observations with low entropy, but the resulting interaction can become inconsistent with the real environment, leading to W2R failure.

**Hypothesis:** A simple **action-support likelihood score** (Support-NLL; how surprising the rollout’s action templates are under the world-model’s training behavior distribution) can predict which rollouts will fail to transfer (W2R failure), and can outperform a strong baseline based on the world model’s own next-observation token likelihood/entropy.

**Why this could be wrong:**
1. World-model inconsistency could be dominated by latent state-tracking errors that do not manifest as unusual actions.
2. Action-likelihood scores may be dominated by trivial factors (action length, rare object names), yielding little predictive value beyond simple baselines.
3. The world model’s own token-level uncertainty may already predict W2R failure as well as any external action-support score.

---

## Proposed Approach

### Overview

We propose an **action-support likelihood audit** for text world model rollouts.

Given a fixed pretrained world model (e.g., Word2World’s TextWorld world model) and an acting agent, we:
1. Run the acting agent inside the world model to obtain a rollout action sequence.
2. Compute an **action-support score** for that action sequence under a behavior-policy action prior extracted from the world-model training data.
3. Evaluate whether this score predicts whether the rollout transfers back to the real environment (W2R success/failure), focusing on episodes where the agent succeeds in the real environment.

The output is not a new world model or agent, but a cheap diagnostic that can be used to filter rollouts before replay or training.

### Method Details

#### 1) Action template extraction
Text actions in TextWorld are short commands with a limited set of verbs/templates (e.g., `take`, `open`, `examine`, `go north`). We define an **action template** as the first token (verb) of the action string, lowercased. (We will also report a robustness check using the first 2 tokens.)

#### 2) Behavior-policy action prior
We build a behavior-policy action prior from the world-model training set:
- Download the Word2World trajectory dataset from HuggingFace (TextWorld subset): https://huggingface.co/datasets/X1AOX1A/LLMasWorldModels
- Extract all action strings from training trajectories collected by GPT-4o.
- Compute a Laplace-smoothed empirical distribution over templates:
  \[
  p_b(verb) = \frac{count(verb)+\alpha}{\sum_v count(v) + \alpha |V|}.
  \]

#### 3) Rollout action-support score
For each world-model rollout with actions \(a_{1:T}\), define:
- \(verb_t\) = template extracted from \(a_t\)
- **Support-NLL**: \(\text{NLL} = \frac{1}{T}\sum_{t=1}^T -\log p_b(verb_t)\)

We also compute a length-normalized variant using characters or tokens to check that the signal is not only action length.

#### 4) Baseline uncertainty features
We compare Support-NLL to two automated baselines computed from the same rollout:
- **Action length baseline**: mean characters per action.
- **World-model observation self-likelihood**: mean per-token negative log-probability of the world model’s generated next-observation tokens (teacher-forced on its own sampled output), averaged over steps.

(If the Word2World code logs token logprobs, this is straightforward; otherwise we re-run the world model forward pass on saved rollouts to compute logprobs.)

#### 5) Target label: W2R failure among Real-success episodes
For each task instance i:
- Run the acting agent in the **real** TextWorld environment → `Real_i ∈ {0,1}`.
- Run the acting agent inside the **world model** → action sequence.
- Replay that action sequence in the **real** environment (Word2World WM2Real evaluation) → `W2R_i ∈ {0,1}`.

We evaluate prediction on the subset `Real_i=1`, so that failures correspond to **world-model-induced** non-transfer rather than intrinsically impossible tasks for that agent.

### Key Innovations

- A **support-based diagnostic** for LLM text world models: predict W2R/CR failures from action distributions, inspired by offline RL “support constraint” thinking but applied to Word2World-style CR evaluation.
- A **minimal, training-free** score (template frequency) that is easy to compute and deploy as a rollout filter.
- A clear comparison against a strong baseline: the world model’s own observation-level self-likelihood/entropy, which may be poorly calibrated even when the rollout is inconsistent.

---

## Related Work

### Field Overview

**LLM-based text world models.** Recent work studies LLMs as simulators for text environments, either by prompting or by supervised fine-tuning on interaction trajectories. Word2World proposes a structured evaluation including long-horizon rollout transfer via W2R and CR.

**Distribution shift and trust in model-based RL.** In model-based RL, learned dynamics models become unreliable under policy shift, motivating short rollouts (MBPO) and uncertainty-aware rollout truncation (MOPO, MOReL, Infoprop). These methods typically aim to improve policy learning rather than to provide a per-rollout predictor of transfer success.

**Support constraints in offline RL.** Offline RL methods often constrain the learned policy to remain within the support of the behavior policy (e.g., CQL, BRAC, BEAR, OSC), since out-of-support actions cause value overestimation and unreliable dynamics predictions. Our proposal transfers this intuition to text world model rollouts: unusual actions may indicate the rollout has drifted into unsupported regions where the world model is unfaithful.

### Related Papers

- **[From Word to World: Can Large Language Models be Implicit Text-based World Models?](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt)**: Introduces CR=W2R/Real and shows policy shift reduces consistency in text environments.
- **[TextWorld: A Learning Environment for Text-based Games](https://arxiv.org/abs/1806.11532)**: Text-based game environment with admissible commands and deterministic dynamics.
- **[ScienceWorld: Is Your Agent Smarter Than a 5th Grader?](https://arxiv.org/abs/2203.07540)**: Text-based science environment used in Word2World.
- **[ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768)**: Textified embodied tasks; part of Word2World’s evaluation suite.
- **[WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206)**: Open-ended text web environment used by Word2World.
- **[Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/)**: Prompted LMs as text world simulators; motivates fine-tuning.
- **[Making Large Language Models into World Models with Precondition and Effect Knowledge](https://arxiv.org/abs/2402.01695)**: Injects symbolic dynamics knowledge into LLM world models.
- **[Efficient Integration of External Knowledge to LLM-based World Models via Retrieval-Augmented Generation and Reinforcement Learning](https://aclanthology.org/2025.findings-emnlp.504/)**: Uses retrieval and RL to improve text world modeling.
- **[RLVR-World](https://thuml.github.io/RLVR-World/)**: Trains world models with verifiable rewards instead of pure MLE; highlights mismatch between likelihood and task metrics.
- **[R-WoM: Retrieval-augmented World Model for Computer-use Agents](https://arxiv.org/abs/2510.11892)**: Retrieval-grounded world modeling; addresses drift by grounding.
- **[Dyna](https://link.springer.com/article/10.1007/BF00115009)**: Classic model-based RL with one-step synthetic rollouts.
- **[World Models](https://arxiv.org/abs/1803.10122)**: Early latent world model framing for planning.
- **[Dreamer](https://arxiv.org/abs/1912.01603)**: Latent imagination-based RL; canonical world model agent.
- **[MuZero](https://arxiv.org/abs/1911.08265)**: Learns a dynamics model for planning; strong model-based baseline.
- **[When to Trust Your Model: Model-Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253)**: Shows model error grows with policy divergence; motivates short rollouts.
- **[MOPO: Model-based Offline Policy Optimization](https://arxiv.org/abs/2005.13239)**: Uses ensemble uncertainty penalties for offline model rollouts.
- **[MOReL: Model-Based Offline Reinforcement Learning](https://arxiv.org/abs/2005.05951)**: Uses an “unknown state-action detector” and absorbing state to prevent model exploitation.
- **[Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779)**: Offline RL with conservatism to avoid OOD actions.
- **[BEAR: Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction](https://arxiv.org/abs/1906.00949)**: Uses behavior-policy constraints to mitigate bootstrapping error.
- **[BRAC: Behavior Regularized Actor Critic](https://arxiv.org/abs/1911.11361)**: Regularizes policy toward the behavior distribution.
- **[Policy Constraint by Only Support Constraint (OSC)](https://arxiv.org/abs/2503.05207)**: Uses diffusion-based support estimation; emphasizes “support only” constraints.
- **[Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845)**: Shows raw likelihood is confounded; likelihood ratios improve OOD detection.
- **[WHALE: Towards Generalizable and Scalable World Models for Embodied Decision-making](https://arxiv.org/abs/2411.05619)**: Addresses policy shift via behavior-conditioning; proposes efficient uncertainty estimation.
- **[On Rollouts in Model-Based Reinforcement Learning](https://arxiv.org/abs/2501.16918)**: Studies epistemic uncertainty propagation and rollout termination via information-theoretic criteria.
- **[Towards Policy-Aware World Models](https://openreview.net/forum?id=Ro2eG1RRde)**: Proposes ESNR as a training-time metric predicting downstream policy learning.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Text world models | Train/prompt LLMs to predict next observation from history+action | Word2World; ACL’24 text simulators | CR=W2R/Real; EM next-state | Rollout drift under policy shift |
| Offline RL support constraints | Avoid actions outside behavior support | CQL, BEAR, BRAC, OSC | D4RL | Mostly optimization, not diagnostics |
| Uncertainty-aware rollouts | Truncate/penalize rollouts in uncertain regions | MBPO, MOPO, MOReL, Infoprop | MuJoCo | Often requires ensembles / continuous states |
| OOD likelihood diagnostics | Correct confounded likelihood scores | Likelihood Ratios for OOD | AUROC OOD | Not specialized to interactive rollouts |

### Closest Prior Work

1. **Word2World**: Defines CR and attributes low CR under policy shift to behavior mismatch, but does not provide a rollout-level pre-screening diagnostic.
2. **MOReL / MOPO / MBPO**: Provide “when to trust the model” principles for MBRL via uncertainty and rollout truncation; do not propose action-support scores for text world model W2R failures.
3. **Offline RL support constraints (CQL/BRAC/OSC)**: Use behavior-policy likelihood/support to constrain learning; we reuse this intuition as a *diagnostic* for text world model rollouts.
4. **Likelihood Ratios for OOD detection**: Highlights confounds in raw likelihood; motivates checks that Support-NLL is not driven by action length/background statistics.

**Novelty Kill Search Summary:** We searched for the exact combination of “world model rollout consistency / W2R / CR” with “action likelihood / behavior policy likelihood / action support / coverage audit”, and checked local finalized proposals and cross-agent drafts. We also searched for offline RL “support constraint” methods being used as diagnostics for world models. No prior work was found that explicitly proposes **behavior-distribution action likelihood** as a predictor of **Word2World-style W2R/CR failures** in text environments as of 2026-02-19 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Word2World | Introduces CR=W2R/Real for text world models | CR is post-hoc; not a pre-screening metric | Predict W2R failures from rollout traces | Enables cheap filtering before replay/training |
| MBPO / MOPO | Uses uncertainty to limit model rollouts (MBRL/offline RL) | Typically continuous control; needs ensembles; not CR/W2R | Use behavior-support likelihood on action strings | Simple, training-free, works in text action spaces |
| MOReL | Unknown state-action detector + absorbing state | Focuses on optimizing policies safely | Use support idea as a diagnostic | Directly targets rollout trust decisions |
| Likelihood ratios for OOD | Fixes confounded likelihood OOD detection | Not interactive; not action-conditioned rollouts | Use as sanity check for action likelihood confounds | Prevents false positives from background stats |

---

## Experiments

### Experimental Setup

**Repository / harness:** Word2World codebase (https://github.com/X1AOX1A/Word2World), which provides scripts for:
- Real-environment interaction (per-task success)
- World-model interaction (world model served via vLLM)
- World-model-to-real replay evaluation (WM2Real)

**World model:** `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B` (HuggingFace): https://huggingface.co/X1AOX1A/WorldModel-Textworld-Qwen2.5-7B

**Benchmark:** TextWorld tasks from Word2World’s dataset.

**Acting agent:** Primary: `gpt-4o-mini` (API). Optional secondary: `gpt-4o` (API) to test whether the diagnostic still separates success/failure when CR is higher.

**Baseline Ladder (REQUIRED):**
- Chance predictor: AUROC = 0.50 by definition.
- Simple heuristic baseline: action length.
- Strong model-internal baseline: world-model observation self-likelihood (or entropy).
- Proposed diagnostic: action-support Support-NLL from training behavior distribution.

**Training Data (for behavior prior):**
- Word2World TextWorld training trajectories (GPT-4o behavior policy) from `X1AOX1A/LLMasWorldModels`.

**Resource Estimate**:
- **Compute budget**: 
  - World model inference: serve 7B model with vLLM on 1×A100 80GB.
  - Behavior prior extraction: CPU-only (counting templates).
  - Optional logprob computation: additional forward passes through the 7B world model (still within a few GPU-hours).
  - Total: expected \< 100 GPU-hours.
- **API usage**:
  - Real + WM rollouts require ~O(num_tasks × max_steps) action generations. Start with a 200-task subset; scale up if needed.
  - WM2Real replay uses no LLM calls (just environment execution).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| TextWorld (Word2World) | Text-based games with deterministic transitions and admissible commands | Real, W2R, CR; plus predictor AUROC | test | https://huggingface.co/datasets/X1AOX1A/LLMasWorldModels | Word2World scripts |

**Primary metric (diagnostic quality):** AUROC for predicting `W2R_i=0` among `Real_i=1` episodes.

**Secondary metric:** Spearman correlation between the diagnostic score and `W2R_i`.

### Main Results

| Method | Base Model | Benchmark | AUROC (mean±std) | Spearman ρ | Source | Notes |
|---|---|---|---:|---:|---|---|
| Chance | - | TextWorld | 0.50 | 0.00 | definition | Deterministic |
| Action length | - | TextWorld | **TBD** | **TBD** | - | To be measured |
| WM obs self-NLL | WorldModel-Textworld-Qwen2.5-7B | TextWorld | **TBD** | **TBD** | - | Strong baseline |
| **Support-NLL (ours)** | Behavior prior from GPT-4o trajs | TextWorld | **TBD** | **TBD** | - | To be verified |

(Word2World’s published CR numbers are used for context but are not directly comparable to AUROC; see `./references/.../sections/5.2 Rollout Consistency.md`.)

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (verb template) | template = first token | Main result |
| Ours (verb+arg) | template = first two tokens | If object/argument rarity dominates, AUROC changes noticeably |
| Length-normalized NLL | normalize by tokens/chars | Tests “background-statistics” confound |

### Experimental Rigor

- **Determinism / seeds**: Run acting agents with greedy decoding where possible (temperature=0). If API nondeterminism remains, repeat the full evaluation 3 times and report mean±std.
- **Uncertainty / power**: Report 95% bootstrap confidence intervals for AUROC (e.g., 1000 bootstrap resamples). Start with 200 tasks; with an expected ~30% W2R failure rate among `Real=1` episodes (Word2World reports CR≈0.71 on TextWorld for GPT-4o-mini), this yields ~60 failure cases, which is typically sufficient to distinguish AUROC differences on the order of ~0.05–0.1.
- **Sanity checks**:
  - Randomly permuting diagnostic scores should give AUROC ≈ 0.5.
  - Reproduce Word2World’s aggregate Real/W2R/CR within reasonable noise on the same subset.
- **Confounders & controls**:
  - Action-length confound: include action length baseline and length-normalized variant.
  - “World model confidence already solves it”: include WM obs self-NLL baseline.

---

## Success Criteria

**Hypothesis:** Support-NLL will be a useful predictor of W2R failure among Real-success episodes, and will outperform action-length and WM obs self-NLL baselines.

**Decision Rule:**
- **Proceed** if Support-NLL achieves AUROC ≥ 0.65 on TextWorld (Real=1 subset) and exceeds the WM obs self-NLL baseline by ≥ 0.05 AUROC on the same evaluation set.
- **Pivot** if Support-NLL beats action length but not WM obs self-NLL: try a slightly richer support model (template bigrams; object canonicalization) while keeping the experiment identical.
- **Refute** if Support-NLL AUROC ≤ 0.55 or if gains disappear after length normalization.

---

## Impact Statement

If successful, this work provides a cheap, automated **rollout trustworthiness audit** for LLM-based text world models. Practitioners using world models for replay, synthetic data generation, or planning could filter or downweight low-support rollouts before spending compute or environment calls on them, reducing wasted replay and reducing the risk of training on inconsistent synthetic experience.

---

## References

- [From Word to World: Can Large Language Models be Implicit Text-based World Models?](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt) - Li et al., 2025
- [TextWorld: A Learning Environment for Text-based Games](https://arxiv.org/abs/1806.11532) - Côté et al., 2018
- [ScienceWorld: Is Your Agent Smarter Than a 5th Grader?](https://arxiv.org/abs/2203.07540) - Wang et al., 2022
- [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768) - Shridhar et al., 2021
- [WebShop](https://arxiv.org/abs/2207.01206) - Yao et al., 2022
- [Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/) - Wang et al., 2024
- [Making Large Language Models into World Models with Precondition and Effect Knowledge](https://arxiv.org/abs/2402.01695) - Xie et al., 2024
- [Efficient Integration of External Knowledge to LLM-based World Models via Retrieval-Augmented Generation and Reinforcement Learning](https://aclanthology.org/2025.findings-emnlp.504/) - Yang et al., 2025
- [RLVR-World](https://arxiv.org/abs/2505.13934) - THUML, 2025
- [R-WoM](https://arxiv.org/abs/2510.11892) - Mei et al., 2025
- [Dyna](https://dl.acm.org/doi/10.1145/104134.104143) - Sutton, 1990
- [World Models](https://arxiv.org/abs/1803.10122) - Ha & Schmidhuber, 2018
- [Dreamer](https://arxiv.org/abs/1912.01603) - Hafner et al., 2019
- [MuZero](https://arxiv.org/abs/1911.08265) - Schrittwieser et al., 2019
- [When to Trust Your Model (MBPO)](https://arxiv.org/abs/1906.08253) - Janner et al., 2019
- [MOPO](https://arxiv.org/abs/2005.13239) - Yu et al., 2020
- [MOReL](https://arxiv.org/abs/2005.05951) - Kidambi et al., 2020
- [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) - Kumar et al., 2020
- [BEAR](https://arxiv.org/abs/1906.00949) - Kumar et al., 2019
- [BRAC](https://arxiv.org/abs/1911.11361) - Wu et al., 2019
- [Only Support Constraint (OSC)](https://arxiv.org/abs/2503.05207) - Gao et al., 2025
- [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845) - Ren et al., 2019
- [WHALE](https://arxiv.org/abs/2411.05619) - Zhang et al., 2024
- [On Rollouts in Model-Based Reinforcement Learning](https://arxiv.org/abs/2501.16918) - Frauenknecht et al., 2025
- [Towards Policy-Aware World Models](https://openreview.net/forum?id=Ro2eG1RRde) - Giridhar et al., 2026
