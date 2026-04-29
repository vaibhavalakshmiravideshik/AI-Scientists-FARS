# untitled

# Stutter-Invariance Metamorphic Audits for Text World-Model Rollouts

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated evaluation (no human-in-the-loop labeling)
  - Fits within the available budget (<=768 A100 GPU-hours; expected <<100 GPU-hours)
  - No browser / GUI / external search-API dependencies (TextWorld is headless; evaluation uses Word2World scripts)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used as interactive agents: an agent reads a text observation, outputs a text action, and receives the next observation from an environment. A common strategy for reducing the cost and risk of developing such agents is to train an LLM as a **text-based world model**: a model that predicts the next environment observation given the interaction history and a candidate action.

A key open problem for deploying learned world models is **trust**. A world model can have strong one-step prediction accuracy yet still produce long-horizon rollouts that do not correspond to executable trajectories in the real environment. This matters because world-model rollouts are used for planning, verification of high-stakes actions, and synthetic data generation.

**From Word to World** (Word2World) provides an evaluation framework for LLM world models on several text environments (ALFWorld, SciWorld, TextWorld, WebShop) and introduces a long-horizon transfer metric: **world-to-real success (W2R)**, the success rate when replaying an action sequence generated inside the world model back in the real environment, and **consistency ratio (CR = W2R / Real)**, where Real is the acting agent's success in the real environment (see `./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/Metrics.md`). For example, on TextWorld using the Qwen2.5-7B world model, Word2World reports Real = 97.44 but W2R = 69.36 for GPT-4o-mini (CR = 0.71), indicating many rollouts found in the world model do not transfer (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/5.2 Rollout Consistency.md`).

However, W2R and CR are **post-hoc**: they require generating a rollout and replaying it in the real environment. In many practical workflows, we want a **pre-screening signal** that predicts whether a given world-model rollout is likely to transfer, without needing real-environment replay or human inspection.

### The Problem

World-model rollouts can fail to transfer for multiple reasons (action distribution shift, missing dynamics coverage, compounding state-tracking errors). A tempting approach is to use the world model's own token-level uncertainty (e.g., average negative log-likelihood or entropy on generated observations) as a trust signal, but likelihood is often confounded and may not reflect state-tracking stability.

This proposal explores a different trust signal inspired by **metamorphic testing**, a testing paradigm designed for systems where correct outputs are hard to specify (the "oracle problem"). Metamorphic testing defines **metamorphic relations**: properties relating outputs across related inputs that should hold even when exact outputs are unknown. Recent work such as **MDPMORPH** applies metamorphic testing to deep reinforcement learning agents by checking MDP-grounded relations rather than requiring manual oracles (`./references/Metamorphic-Testing-of-Deep-Reinforcement-Learning-Agents-with-MDPMORPH/meta/meta_info.txt`).

For text environments, we often know specific **state-preserving actions**. In TextWorld, the command `look` returns a room description (a read-only query) and should not change the underlying game state (TextWorld provides state tracking and an MDP formalization; `./references/TextWorld-A-Learning-Environment-for-Text-based-Games/meta/meta_info.txt`). This suggests a deployment-relevant metamorphic relation for world models: inserting no-op actions should not change the predicted state evolution under subsequent actions.

### Key Insight and Hypothesis

**Key insight:** If a world model maintains a stable internal representation of environment state, then inserting a state-preserving action (e.g., `look`) into a rollout should not meaningfully change the world model's subsequent predictions for the original action sequence. Conversely, if the world model is fragile (e.g., loses track of latent state and relies on superficial recency patterns), then inserting no-op steps can cause the predicted trajectory to drift, revealing a failure mode that should correlate with poor world-to-real transfer.

**Hypothesis:** For TextWorld rollouts generated inside a fixed pretrained world model (Word2World), a **stutter-invariance violation score** (how much the world model's predicted trajectory changes after inserting `look` actions) predicts W2R failure better than standard likelihood/uncertainty baselines.

**Why this could be wrong:**
1. W2R failures may be dominated by incorrect transitions after rare but important actions rather than by generic state-tracking instability exposed by no-ops.
2. The violation score may be driven by superficial text paraphrase changes rather than game-state changes (a measurement confound).
3. A generic stability baseline (sampling the world model twice and measuring divergence) may perform as well as stutter-invariance, making `look` insertion unnecessary.

---

## Proposed Approach

### Overview

We propose a **metamorphic audit** for text world models based on **stutter invariance**:

- Start from a world-model rollout: an initial observation and an action trace `a1..aT` produced by an acting agent interacting with the world model.
- Construct a stuttered action trace by inserting `look` after every action: `a1, look, a2, look, ..., aT, look`.
- Run the same world model open-loop on the original and stuttered traces and measure how much the predicted post-action observations differ.

The audit produces a single scalar score per rollout (higher means less invariant), which we evaluate as a predictor of whether that rollout transfers back to the real environment (W2R).

### Method Details

#### Data and rollouts
We use Word2World's released TextWorld world model checkpoints and evaluation harness (`./references/GitHub---X1AOX1A-Word2World/meta/meta_info.txt`).

1. **Generate world-model rollouts** on the TextWorld test split using the Word2World script for interaction with world models (README: `scripts/interact_with_world_model/run.sh`).
   - Acting agent: `gpt-4o-mini` (API) to ensure a non-trivial W2R failure rate consistent with Word2World's Table 2.
   - World model: `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B` served by vLLM.
   - Save per-episode logs including the action trace.

2. **Compute W2R labels** by replaying the saved action traces in the real TextWorld environment using Word2World's `cal_wm2real.sh` (README: "Map WM Actions to Real Environments"). This step uses no LLM calls.

Each episode i yields a label `y_i = 1` if W2R succeeds and `0` otherwise.

#### Stutter-invariance violation score
For each episode with action trace `a1..aT`:

- **Original branch**: run the world model open-loop on `a1..aT` to generate post-action observations `o1..oT`.
- **Stuttered branch**: run the world model open-loop on `a1, look, a2, look, ..., aT` to generate aligned post-action observations `o1'..oT'` (ignore the inserted `look` outputs).

This comparison is intentionally **open-loop**: inserting `look` changes the textual context the world model conditions on. The audit measures whether this additional, nominally state-preserving interaction causes cumulative drift in the model's predicted trajectory.

**Text distance function.** We define distance between two observations using an embedding model `E(·)`:

- Normalize text by lowercasing and collapsing whitespace.
- Compute `d(o, o') = 1 - cosine(E(o), E(o'))`.

**Default embedding model:** `BAAI/bge-m3` (open embedding model available in the platform's embedding model list; used as a fixed similarity encoder).

**Per-episode score:**

`S_stutter(i) = mean_t d(o_t, o_t')`.

This score is length-normalized by construction.

#### Embedding sanity gate and structured fallback
Because embedding choice is load-bearing, we add a prerequisite sanity check using the real TextWorld environment:

- Sample K = 100 states (by running short random action prefixes).
- For each state, collect:
  - a **same-state pair**: `look` twice (should be same underlying state)
  - a **different-state pair**: take a random admissible action then `look` (likely different state)
- Compute AUROC separating same-state vs different-state pairs using `d(·,·)`.

If AUROC < 0.8, we treat the embedding-based distance as unreliable for state equivalence and switch to a **structured parse distance**:

- Extract a simple set representation from each observation: `(room name, visible object tokens, inventory tokens)` via regex.
- Define distance as `1 - Jaccard(set(o), set(o'))`.

If this fallback is triggered, we report results as "stutter invariance over parsed state" (not as an embedding-based semantic metric).

### Key Innovations

- A **metamorphic testing** framing for auditing LLM world-model rollouts, adapting ideas from DRL metamorphic testing (e.g., MDPMORPH) to text world models.
- A concrete, deployment-relevant metamorphic relation for text environments: **stutter invariance** under state-preserving actions (`look`).
- A minimal, fully automated **rollout-level trust signal** that does not require training new models and can be applied to any released world model.

---

## Related Work

### Field Overview

**LLMs as text world models.** Recent work studies prompted and fine-tuned LLMs as simulators for text environments, with evaluation going beyond single-step next-state accuracy to long-horizon fidelity (e.g., Word2World). A recurring issue is compounding error: small local mismatches can cause rollouts to drift.

**When to trust model rollouts in model-based RL.** In continuous-control model-based RL, learned dynamics models are known to be unreliable under distribution shift, motivating uncertainty-aware rollouts and termination criteria (e.g., MBPO, MOPO, MOReL). These methods typically target improving policy learning rather than producing a per-rollout trust score for a fixed world model in a text environment.

**Metamorphic and property-based testing for learning systems.** Metamorphic testing provides oracle-free testing via input/output relations. Prior work applies metamorphic relations to neural perception systems and to DRL agents (MDPMORPH), but has not been widely instantiated for LLM-based world-model rollouts.

### Related Papers

- **[From Word to World: Can Large Language Models be Implicit Text-based World Models?](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt)**: Defines W2R and CR for evaluating long-horizon transfer of LLM world-model rollouts.
- **[GitHub - X1AOX1A Word2World](./references/GitHub---X1AOX1A-Word2World/meta/meta_info.txt)**: Provides released checkpoints, datasets, and evaluation scripts for real/WM/W2R rollouts.
- **[TextWorld: A Learning Environment for Text-based Games](./references/TextWorld-A-Learning-Environment-for-Text-based-Games/meta/meta_info.txt)**: Introduces TextWorld as a controllable text-game MDP with state tracking and standard commands like `look`.
- **[ScienceWorld: Is Your Agent Smarter Than a 5th Grader?](https://arxiv.org/abs/2203.07540)**: Text science environment used in many text-agent evaluations.
- **[ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768)**: Textified embodied task environment used as a structured world-model benchmark.
- **[WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206)**: Text-only web environment; motivates world-model fidelity for decision-making.
- **[Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/)**: Studies prompted LMs as simulators for text games (ByteSized32 setting).
- **[RLVR-World: Training World Models with Reinforcement Learning](./references/RLVR-World-Training-World-Models-with-Reinforcement-Learning/meta/meta_info.txt)**: Uses RL with verifiable rewards to optimize world models directly for decoded metrics.
- **[Reinforcement World Model Learning for LLM-based Agents](./references/Reinforcement-World-Model-Learning-for-LLM-based-Agents/meta/meta_info.txt)**: Trains LLM world modeling with embedding-based sim-to-real rewards; discusses inefficient actions like `look`.
- **[Web Agents with World Models (WMA)](https://arxiv.org/abs/2410.13232)**: Studies world models for web navigation with structured state representations.
- **[Is Your LLM Secretly a World Model of the Internet? (WebDreamer)](https://arxiv.org/abs/2410.03744)**: Uses LLM-based simulation for planning in web agents.
- **[Dyna](https://link.springer.com/article/10.1007/BF00115009)**: Classic model-based RL combining real and simulated experience.
- **[World Models](https://arxiv.org/abs/1803.10122)**: Early latent world model framework for planning and control.
- **[Dream to Control (Dreamer)](https://arxiv.org/abs/1912.01603)**: Latent imagination-based model-based RL.
- **[MuZero](https://arxiv.org/abs/1911.08265)**: Learns a model for planning without reconstructing observations.
- **[Model-Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253)**: Shows model error grows with rollout horizon; motivates short rollouts.
- **[MOPO](https://arxiv.org/abs/2005.13239)**: Penalizes model rollouts using uncertainty to avoid exploitation.
- **[MOReL](https://arxiv.org/abs/2005.05951)**: Uses an unknown state-action detector to constrain model rollouts.
- **[Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845)**: Demonstrates raw likelihood can be confounded; motivates alternatives to token likelihood as trust signals.
- **[Cycle-Consistent World Models for Domain Independent Latent Imagination](./references/Cycle-Consistent-World-Models-for-Domain-Independent-Latent-Imagination/meta/meta_info.txt)**: Uses cycle consistency for cross-domain latent alignment; different from metamorphic audits of rollouts.
- **[Metamorphic Testing of Deep Reinforcement Learning Agents with MDPMORPH](./references/Metamorphic-Testing-of-Deep-Reinforcement-Learning-Agents-with-MDPMORPH/meta/meta_info.txt)**: Defines MDP-grounded metamorphic relations and learns thresholds to detect faults in DRL agents.
- **[DeepTest: Automated Testing of Deep Neural-Network-driven Autonomous Cars](https://arxiv.org/abs/1708.08559)**: Applies metamorphic relations (image transformations) to uncover DNN failures without explicit output oracles.
- **[Metamorphic Testing: A Review of Challenges and Opportunities](https://dl.acm.org/doi/10.1145/3143561)**: Survey of metamorphic testing concepts and relation design.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Text-based world models | Train/prompt LLMs to predict next observation from history + action | Word2World; ByteSized32 text simulators; RWML | Next-state EM/F1; W2R/CR | Compounding rollout drift; policy shift |
| Uncertainty-aware rollouts (MBRL/offline RL) | Use uncertainty to truncate/penalize model rollouts | MBPO, MOPO, MOReL | MuJoCo / Atari | Often needs ensembles/continuous states; not tailored to text rollouts |
| Metamorphic testing | Oracle-free testing via input/output relations | MDPMORPH; DeepTest; MT surveys | Fault detection / mutation detection | Relation design is domain-specific |
| Cycle-consistency constraints | Regularize latent mappings across domains | CCWM | CARLA sim-to-sim | Not a rollout trust metric |

### Closest Prior Work

1. **Word2World** (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt`)
   - What it does: Introduces W2R and CR to measure long-horizon transfer of world-model rollouts.
   - Limitation for our question: Does not provide a rollout-level pre-screening signal; evaluation is post-hoc.
   - Why different: We propose an oracle-free metamorphic audit computed purely from the world model and the rollout action trace.

2. **MDPMORPH** (`./references/Metamorphic-Testing-of-Deep-Reinforcement-Learning-Agents-with-MDPMORPH/meta/meta_info.txt`)
   - What it does: Metamorphic testing framework for DRL agents using MDP-grounded relations.
   - Limitation: Targets agent policies and Gym environments, not LLM-based world models and natural-language observations.
   - Why different: We instantiate a concrete metamorphic relation (stutter invariance) for text world-model rollouts and evaluate it against W2R.

3. **Uncertainty-based rollout trust in model-based/offline RL (MBPO/MOPO/MOReL)**
   - What they do: Use uncertainty/unknown-state detectors to constrain rollouts to improve learning.
   - Limitation: Not a diagnostic for fixed text world models; often requires ensembles and continuous states.
   - Why different: Our audit is training-free, discrete-text oriented, and evaluated on Word2World's WM2Real protocol.

4. **RWML / RLVR-World**
   - What they do: Use RL-style objectives to improve world model predictions.
   - Limitation: Do not address rollout-level trust scoring for filtering rollouts before replay or training.
   - Why different: We focus on diagnostic trust signals rather than training objectives.

**Novelty Kill Search Summary:** Searched for the exact combination of "metamorphic testing" with "world model" / "learned simulator" / "W2R" / "TextWorld", as well as variants like "no-op invariance", "stutter invariance", and "look action invariance". Also checked local finalized proposals and cross-agent drafts for these terms. No prior work proposing a stutter-invariance metamorphic audit for Word2World-style rollout transfer was found as of 2026-02-21 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Word2World | Defines W2R/CR for rollout transfer | Post-hoc; no rollout trust score | Add metamorphic audit score | Gives a pre-screening signal for rollouts |
| MBPO/MOPO/MOReL | Uncertainty-aware rollout control | Mostly continuous control; learning-focused | Use no-op metamorphic relation | Simple, training-free trust diagnostic |
| MDPMORPH | Metamorphic testing for DRL agents | Not about world models / text rollouts | Apply MT to world-model rollouts | Uses environment semantics (`look` no-op) as an oracle-free check |
| CCWM | Cycle consistency for domain adaptation | Different objective (cross-domain alignment) | Stutter invariance within one domain | Targets rollout drift in a fixed world model |

---

## Experiments

### Experimental Setup

**Repository / harness:** Word2World (evaluation scripts in README: `./references/GitHub---X1AOX1A-Word2World/sections/README.md.md`).

**World model:** `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B` served with vLLM.

**Benchmark:** Word2World TextWorld test split (use default `SPLIT=test`; start with `NUM_EXAMPLES=200` for the main study and scale up if variance is high).

**Acting agent (for generating WM rollouts):** `gpt-4o-mini` (API).

**Baseline Ladder (REQUIRED):**
- Chance baseline: AUROC = 0.50.
- B1 Length baseline: rollout length T.
- B2 Token likelihood baseline: mean per-token negative log-likelihood (teacher forced) of the world model on its generated observations.
- B3 Sampling-consistency baseline: run the world model twice with stochastic decoding on the same fixed action trace; score = mean embedding distance between the two samples.
- B4 Action-conditional likelihood baseline: teacher-forced NLL restricted to the first line (or first 20 tokens) of each predicted next observation.
- **Ours**: stutter-invariance violation score from `look` insertion.

**Decoding settings (audit computation):**
- For stutter audit and likelihood baselines: deterministic decoding (temperature = 0) to reduce sampling noise.
- For sampling-consistency baseline (B3): stochastic decoding (temperature = 0.7, top_p = 0.9), two independent seeds.

**Resource Estimate**:
- **Compute budget**: 1x A100 80GB for vLLM serving the 7B world model. Expected < 20 GPU-hours total for 200 episodes with average <=50 steps, including 4-5 open-loop generations per episode (original, stuttered, and baseline reruns). If average episodes are longer, cap `MAX_ROUND` to 30 and/or use `NUM_EXAMPLES=100` to keep compute bounded.
- **GPU memory**: <= 80GB (single 7B model).
- **API usage**: acting agent calls are required only for generating WM rollouts (about `NUM_EXAMPLES * MAX_ROUND` calls; start with 200 * 50 = 10k calls). WM2Real replay and audit computation require no additional API calls.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| TextWorld (Word2World) | Text-based games with deterministic, rule-governed transitions | AUROC/AUPRC for predicting W2R failure | test | https://huggingface.co/datasets/X1AOX1A/LLMasWorldModels | Word2World scripts: `scripts/interact_with_world_model/run.sh` + `scripts/interact_with_world_model/cal_wm2real.sh` |

**Primary metric:** AUROC for predicting W2R failure (y=0) from rollout-level scores.

**Secondary metrics:** AUPRC (to handle class imbalance), and base rate of W2R failure. Also report AUROC within rollout-length strata to check the length confound.

### Main Results

**Result table (to be filled by verification):**

| Method (score) | Base Model | Benchmark | AUROC (mean +- std) | AUPRC (mean +- std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Chance | - | TextWorld (Word2World) | 0.50 | base rate | - | Deterministic |
| B1 Length (T) | - | TextWorld (Word2World) | TBD | TBD | - | Needs run |
| B2 Token likelihood (mean NLL) | `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B` | TextWorld (Word2World) | TBD | TBD | - | Needs run |
| B3 Sampling consistency (2x WM samples) | `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B` | TextWorld (Word2World) | TBD | TBD | - | Needs run |
| B4 Action-conditional NLL (first line / 20 tokens) | `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B` | TextWorld (Word2World) | TBD | TBD | - | Needs run |
| **Ours: stutter invariance violation (look insertion)** | `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B` | TextWorld (Word2World) | TBD | TBD | - | Needs run |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| Ours (embedding distance) | Default distance uses `BAAI/bge-m3` | Works if embedding sanity AUROC >= 0.8 |
| Ours (structured parse fallback) | Use Jaccard distance over parsed state elements | Used only if embedding sanity fails |

### Experimental Rigor

- **Seeds**: Use 3 seeds for the acting agent (different sampling seeds) and 3 seeds for the world model stochastic runs used in B3 (different random seeds). For deterministic decoding audit runs, seeds do not apply.
- **Embedding sanity check**: K = 100 state pairs; require AUROC >= 0.8 or switch to structured parse distance.
- **Length confound control**: Report AUROC within strata of rollout length (e.g., T<=10, 10<T<=20, T>20).
- **Tie policy**: If our AUROC is within 0.02 of the sampling-consistency baseline (B3), record the outcome as "no advantage over generic stochastic consistency" and treat the main claim as not supported.

---

## Success Criteria

**Hypothesis** (directional): Rollouts with larger stutter-invariance violations are more likely to fail W2R replay.

**Decision Rule** (concrete):
- **Proceed**: Our stutter score achieves AUROC >= 0.65 and exceeds the best baseline (B1-B4) by >= 0.05 absolute on the TextWorld test set (bootstrap CI for delta excludes 0), and is not within 0.02 of B3.
- **Pivot**: If embedding-based distance fails the sanity gate but structured parse distance passes, proceed with the structured-state variant and treat the result as evidence about state-element drift rather than semantic drift.
- **Refute**: If AUROC < 0.65, or improvement over the best baseline is < 0.05, or the score ties B3 within 0.02.

**Data leakage note:** Because world models and agents may have been pre-trained on internet text that overlaps with TextWorld-like content, we treat this as a benchmark-level risk. This proposal does not attempt to prove pretraining exclusion; instead it evaluates relative predictiveness of audit signals under a fixed harness.

---

## Impact Statement

If successful, this work provides a simple, automated rollout-level trust signal for LLM world models. Developers using world models for planning, verification, or synthetic data could filter or downweight rollouts that are likely to fail world-to-real replay, reducing wasted compute and preventing downstream training on inconsistent synthetic trajectories.

---

## References

- [From Word to World: Can Large Language Models be Implicit Text-based World Models?](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt) - Li et al., 2025
- [GitHub - X1AOX1A Word2World](./references/GitHub---X1AOX1A-Word2World/meta/meta_info.txt) - Li et al., 2025
- [TextWorld: A Learning Environment for Text-based Games](./references/TextWorld-A-Learning-Environment-for-Text-based-Games/meta/meta_info.txt) - Cote et al., 2018
- [Reinforcement World Model Learning for LLM-based Agents](./references/Reinforcement-World-Model-Learning-for-LLM-based-Agents/meta/meta_info.txt) - Yu et al., 2026
- [RLVR-World: Training World Models with Reinforcement Learning](./references/RLVR-World-Training-World-Models-with-Reinforcement-Learning/meta/meta_info.txt) - Wu et al., 2025
- [Cycle-Consistent World Models for Domain Independent Latent Imagination](./references/Cycle-Consistent-World-Models-for-Domain-Independent-Latent-Imagination/meta/meta_info.txt) - Bender et al., 2021
- [Metamorphic Testing of Deep Reinforcement Learning Agents with MDPMORPH](./references/Metamorphic-Testing-of-Deep-Reinforcement-Learning-Agents-with-MDPMORPH/meta/meta_info.txt) - Li et al., 2025
- [ScienceWorld: Is Your Agent Smarter Than a 5th Grader?](https://arxiv.org/abs/2203.07540) - Wang et al., 2022
- [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768) - Shridhar et al., 2021
- [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206) - Yao et al., 2022
- [Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/) - Wang et al., 2024
- [Web Agents with World Models (WMA)](https://arxiv.org/abs/2410.13232) - Chae et al., 2024
- [Is Your LLM Secretly a World Model of the Internet?](https://arxiv.org/abs/2410.03744) - WebDreamer, 2024
- [Dyna](https://link.springer.com/article/10.1007/BF00115009) - Sutton, 1991
- [World Models](https://arxiv.org/abs/1803.10122) - Ha and Schmidhuber, 2018
- [Dream to Control (Dreamer)](https://arxiv.org/abs/1912.01603) - Hafner et al., 2019
- [MuZero](https://arxiv.org/abs/1911.08265) - Schrittwieser et al., 2019
- [Model-Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253) - Janner et al., 2019
- [MOPO](https://arxiv.org/abs/2005.13239) - Yu et al., 2020
- [MOReL](https://arxiv.org/abs/2005.05951) - Kidambi et al., 2020
- [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845) - Ren et al., 2019
- [DeepTest: Automated Testing of Deep Neural-Network-driven Autonomous Cars](https://arxiv.org/abs/1708.08559) - Tian et al., 2018
- [Metamorphic Testing: A Review of Challenges and Opportunities](https://dl.acm.org/doi/10.1145/3143561) - Chen et al., 2018
