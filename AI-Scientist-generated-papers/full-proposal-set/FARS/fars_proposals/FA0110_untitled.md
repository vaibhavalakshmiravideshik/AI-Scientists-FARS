# untitled

# Targeted Counterfactual Branch Augmentation for Robust Text-Based World Models under Agent Policy Shift

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used as **interactive agents**: they read a text observation, output a text action, and receive the next text observation from an environment (e.g., text games, tool-use simulators, or web-task simulators). A promising way to make such agents safer and more data-efficient is to train an LLM to act as a **text-based world model**: a model that predicts the next environment observation given the history and a candidate action.

**From Word to World (Word2World)** trains open LLMs into text-based world models across multiple environments and introduces a long-horizon fidelity metric, the **Consistency Ratio (CR)**, defined as **CR = W2R / Real** where:
- **Real** is an agent’s success rate in the real environment,
- **W2R** (“world-to-real”) is the success rate when we replay an action sequence generated inside the world model back in the real environment.
High CR indicates that world-model rollouts are executable and informative for decision-making.

Word2World also surfaces a key bottleneck for deployment: world-model fidelity can degrade substantially under **agent policy shift** (the acting agent at test time behaves differently from the behavior policy that produced world-model training data). In SciWorld, Word2World shows that expanding **behavioral coverage** by training the world model on trajectories from multiple agents improves **out-of-distribution (OOD)** agent CR substantially, e.g., GPT-4o-mini CR **0.49 → 0.81** and OOD-agent average CR **0.83 → 0.91** when moving from a single-agent (4K GPT-4o) dataset to a mixed-agent dataset (Table 3 in Word2World).

However, collecting multi-agent trajectories is expensive and often requires multiple proprietary API agents. This motivates a practical question: **Can we improve world-model robustness under policy shift without collecting additional full trajectories from multiple agents?**

### The Problem

If a world model is trained primarily on expert trajectories, then under a different acting policy it may encounter (state, action) pairs that are rare or absent in the training distribution. This mismatch can cause the world model to generate incorrect next observations after off-distribution actions; these errors can compound over long rollouts, reducing W2R and CR.

A simple but widely-used idea in model-based reinforcement learning is to generate “branched rollouts” or short simulations from real states to reduce compounding error (e.g., MBPO), and a classic idea in imitation learning is to reduce covariate shift by collecting labels on learner-visited states (e.g., DAgger). Separately, offline RL has explored counterfactual data augmentation using structured dynamics assumptions (e.g., MoCoDA). But these approaches are not directly instantiated for **LLM-based text world models** evaluated by **world-model-to-real rollout fidelity (CR)** under **agent shift**.

In text-based environments like **SciWorld** and **TextWorld**, we can often obtain state-dependent admissible actions from the environment. This suggests a low-cost alternative to multi-agent trajectory collection: **counterfactual one-step branches**. Instead of running many agents to explore diverse behaviors end-to-end, we can take states visited in expert trajectories and ask “what if the agent took a different valid action here?”, then query the real environment for the resulting next observation.

The key open question is whether such one-step counterfactual augmentation can meaningfully improve long-horizon CR under policy shift, and whether **targeting** the branches toward OOD-agent behavior matters beyond generic random branching.

### Key Insight and Hypothesis

**Key insight:** Long-horizon rollout drift often begins immediately after the world model makes a wrong prediction for an action that is off-distribution relative to expert trajectories. If we can improve the world model’s one-step transition modeling specifically on action types that the OOD agent tends to use (but the expert rarely uses), we may reduce early drift and improve W2R and CR.

**Hypothesis:** Under matched training compute, **targeted counterfactual branches** (sampling alternative admissible actions in proportion to how over-represented they are in an OOD agent’s real-environment runs) improve OOD-agent **CR** more than **random counterfactual branches**.

**Why this could be wrong:**
1. **Not an action-support problem:** CR degradation might be driven by observation-generation ambiguity or long-horizon state tracking rather than missing coverage for specific action types.
2. **One-step is insufficient:** Even perfect one-step transitions on a subset of actions might not prevent multi-step drift if later states still deviate.
3. **Targeting signal is too noisy:** OOD-agent action frequencies may not be stable across tasks, or may be dominated by invalid/redundant actions.
4. **Noise injection hurts:** Branching can add low-value transitions (dead ends) that degrade overall modeling.

---

## Proposed Approach

### Overview

We propose **Targeted Counterfactual Branch Augmentation (TCBA)** for training text-based LLM world models.

Given a dataset of expert trajectories used to train a world model, we generate additional training examples by:
1. Selecting a branching point (state) along an expert trajectory.
2. Querying the environment for the list of admissible actions at that state.
3. Sampling an alternative admissible action (not the expert action).
4. Executing that action in the real environment to obtain the true next observation.
5. Adding a new training example consisting of the expert prefix up to that state plus the alternative action and resulting next observation.

We compare **random branching** versus **targeted branching**. Targeted branching uses an automatically computed reweighting over action types to emphasize actions that appear more often in an OOD agent’s real-environment trajectories than in the expert dataset.

### Method Details

#### Environment and admissible action access
We use **SciWorld** from the Word2World codebase (text-only; no browser/GUI). SciWorld exposes state-dependent admissible actions via the environment server endpoint `/action_hint`, which returns `possible_actions` (and possible objects).

#### Branch generation without environment snapshotting
SciWorld does not expose a general “restore to arbitrary intermediate state” API in Word2World’s server. We therefore generate branches by **replaying the expert prefix**:
- Reset the environment to the task instance corresponding to the trajectory’s `data_idx`.
- Replay expert actions from the start up to time step t.
- Query admissible actions at that state.
- Sample and execute a single alternative action to collect the counterfactual next observation.

A deterministic-environment sanity check (described in Experiments) ensures that replaying the prefix reproduces the same observation.

#### Expected effect size (order-of-magnitude)
Word2World’s behavioral-coverage intervention (single-agent → mixed-agent trajectories) improved SciWorld OOD-agent average CR by about **+0.08** (0.83 → 0.91). One-step branches are a weaker intervention than full additional trajectories, so a plausible target is a smaller gain (e.g., **+0.02 to +0.05** CR) over the random-branch baseline. The verification plan is designed to detect whether the effect is statistically distinguishable from 0 under matched compute.

#### Action canonicalization and targeting weights
Exact action strings in SciWorld can be sparse (object names vary). To keep targeting stable and cheap, we operate at the **action template** level, approximated by the first token (verb) of the action string (e.g., `open`, `close`, `activate`, `go`, `mix`).

We compute:
- `freq_expert(verb)` from the expert trajectory dataset.
- `freq_ood(verb)` from the OOD acting agent’s real-environment trajectories on a **calibration split** of SciWorld tasks **disjoint from the test split** (to avoid test leakage). This calibration run is done once and reused for all world-model training seeds.

Define a targeting ratio:
\[
\rho(verb) = \frac{freq_{ood}(verb)+\epsilon}{freq_{expert}(verb)+\epsilon}.
\]

At a branching state with admissible action list \(\mathcal{A}(s)\), we sample an alternative action \(a \in \mathcal{A}(s)\setminus\{a_{expert}\}\) by:
1. Choose a verb proportional to \(\rho(verb)\) among verbs present in \(\mathcal{A}(s)\).
2. Choose uniformly among admissible actions with that verb.

This uses no learned behavior model and no extra agent runs beyond what CR evaluation already requires.

#### Training procedure
We train three world models with identical training hyperparameters (same optimizer steps / tokens / sequence length / LoRA configuration), differing only in the training data:

- **A. Expert-only**: 4K expert trajectories (subset of Word2World’s SciWorld training set).
- **B. Random-branch augmentation**: expert-only + counterfactual one-step branches sampled uniformly from admissible actions.
- **C. Targeted-branch augmentation (TCBA, ours)**: expert-only + counterfactual one-step branches sampled with targeting ratio \(\rho\).

All three models use the same base model (Qwen2.5-7B) and are trained with **LoRA (Low-Rank Adaptation; a parameter-efficient fine-tuning method)** to fit within compute budget.

### Key Innovations

1. **Agent-free coverage expansion**: replaces expensive multi-agent trajectory collection with environment-instrumented counterfactual one-step branches.
2. **Targeting for policy shift**: focuses augmentation on action types that are empirically over-represented in the OOD agent, rather than adding generic random diversity.
3. **Mechanism-facing evaluation in text world models**: evaluates targeted vs random branching under compute match using Word2World’s CR and a per-step divergence analysis during W2R replay.

---

## Related Work

### Field Overview

**LLM world models for interactive environments.** Recent work increasingly treats LLMs as simulators for text-based environments, where world modeling is cast as next-observation prediction under an interaction protocol. Word2World provides a systematic evaluation framework including long-horizon rollout fidelity (CR) and demonstrates downstream utility for verification and synthetic experience.

**Distribution shift in model-based rollouts.** In model-based RL, synthetic rollouts can distort the training distribution when the model is queried off-distribution, motivating short-horizon rollouts from real states (e.g., MBPO) and uncertainty-aware rollout truncation (e.g., ensemble disagreement, information-theoretic criteria). These methods typically target policy learning, not improving the world model itself.

**Imitation learning under covariate shift.** Dataset aggregation (DAgger) addresses covariate shift by collecting expert labels on states visited by the learner, and follow-up work reduces expert-query cost via disagreement-triggered querying. Our setting is analogous, but the “expert” for labeling counterfactuals is the environment transition function, and our goal is to improve a world model’s fidelity under policy shift.

**Counterfactual data augmentation for sequential decision making.** Offline RL has explored counterfactual augmentation under structural assumptions (e.g., MoCoDA’s locally factored dynamics). In text environments, data augmentation has also been studied via symmetry/permutation discovery in TextWorld. Our approach is simpler: we use admissible action lists to generate grounded counterfactual transitions at real visited states.

### Related Papers

- **[From Word to World: Can Large Language Models be Implicit Text-based World Models?](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt)**: Introduces CR and shows behavioral coverage improves OOD-agent CR in SciWorld; our primary motivation.
- **[TextWorld: A Learning Environment for Text-based Games](https://arxiv.org/abs/1806.11532)**: Provides procedurally generated text games and admissible commands; relevant as another environment family for branching.
- **[ScienceWorld: Is Your Agent Smarter Than a 5th Grader?](https://arxiv.org/abs/2203.07540)**: Defines the SciWorld environment and its action templates; our benchmark.
- **[Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/)**: Studies LMs as simulators for text environments; complementary to Word2World.
- **[World Models](https://arxiv.org/abs/1803.10122)**: Early neural world model framing motivating imagination-based learning.
- **[Dyna: An Integrated Architecture for Learning, Planning, and Reacting](https://link.springer.com/article/10.1007/BF00115009)**: Classic model-based RL with learned models and planning.
- **[When to Trust Your Model: Model-Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253)**: Uses short branched rollouts from real states to limit model-bias; conceptually adjacent but targets policy learning.
- **[STEVE: Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675)**: Uses ensembles to mitigate model error in value expansion; motivates uncertainty-aware rollouts.
- **[PETS: Probabilistic Ensembles with Trajectory Sampling](https://arxiv.org/abs/1805.12114)**: Ensemble-based planning and uncertainty estimation for MBRL.
- **[DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686)**: Addresses covariate shift in imitation learning via interactive data aggregation; conceptual parallel.
- **[DADAgger: Disagreement-Augmented Dataset Aggregation](https://arxiv.org/abs/2301.01348)**: Reduces expert queries using disagreement; conceptually similar to targeting, but not for world models.
- **[MoCoDA: Model-based Counterfactual Data Augmentation](https://arxiv.org/abs/2210.11287)**: Counterfactual augmentation for offline RL under local structure; closest “counterfactual augmentation” precedent.
- **[RoCoDA: Counterfactual Data Augmentation for Robot Learning](https://rocoda.github.io/)**: Counterfactual augmentation for robot demonstrations; shows counterfactuals can enable new behaviors.
- **[On Rollouts in Model-Based Reinforcement Learning](https://arxiv.org/abs/2501.16918)**: Analyzes rollout mechanisms and distribution shift; motivates measuring drift accumulation.
- **[Masked Model-based Actor-Critic (M2AC)](https://arxiv.org/abs/2010.04893)**: Uses uncertainty masking for safe rollouts; complementary to our data augmentation approach.
- **[Self-Improving World Modelling with Latent Actions (SWIRL)](https://arxiv.org/abs/2602.06130)**: Improves world modeling without explicit action labels; orthogonal to our action-coverage focus.
- **[BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL](https://arxiv.org/abs/2506.17211)**: Uses expert-anchored branching for dense reward in reasoning RL; shares “branch from expert prefixes” motif.
- **[Data Augmentation for Learning to Play in Text-Based Games](https://www.ijcai.org/proceedings/2022/0436.pdf)**: Uses transition-matching permutations in TextWorld; related as text-game augmentation.
- **[Is Your LLM Secretly a World Model of the Internet?](https://arxiv.org/abs/2411.06559)**: Uses world-model style planning for web agents; broader context.
- **[DynaWeb](https://arxiv.org/abs/2601.22149)**: Model-based RL for web agents; shows planning benefits but does not address CR under policy shift.
- **[R-WoM: Retrieval-augmented World Model for Computer-use Agents](https://arxiv.org/abs/2510.11892)**: Retrieval-augmented rollouts for computer-use agents; addresses drift via retrieval rather than coverage.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| LLM text world models | Train LLMs to predict environment responses under a protocol | Word2World; LM world simulators | CR (W2R/Real), success rates | Sensitive to behavior and environment complexity |
| Rollout control in MBRL | Limit harm from model bias during synthetic rollouts | MBPO; STEVE; M2AC; PETS; On Rollouts in MBRL | MuJoCo returns, rollout length | Focuses on policy learning, not improving model training data |
| Interactive imitation learning | Reduce covariate shift via data aggregation | DAgger; DADAgger | Robotics/IL benchmarks | Requires expert labeling; not directly for world models |
| Counterfactual augmentation | Generate additional transitions under structured assumptions | MoCoDA; RoCoDA | Offline RL tasks | Often needs factorization/representation assumptions |
| Text-game augmentation | Use symmetries/permutations to augment text environments | Transition-matching permutation | TextWorld cooking games | Limited symmetry discovery; not action-support targeting |

### Closest Prior Work

1. **Word2World**: Defines CR and empirically shows behavioral coverage via multi-agent trajectories improves OOD-agent CR in SciWorld (Table 3). It does not provide a low-cost alternative to collecting diverse agent trajectories.
2. **MBPO**: Uses short branched rollouts from real replay-buffer states to reduce model bias during policy optimization. It does not propose augmenting world-model training data with environment-labeled counterfactual transitions, and is not evaluated via W2R/CR.
3. **DAgger**: Collects expert labels on learner-visited states to address covariate shift. In our setting, we do not query a human expert; instead we query the environment’s transition function for admissible-action counterfactuals.
4. **MoCoDA**: Generates counterfactual transitions for offline RL under local factorization assumptions. Our approach uses environment instrumentation in text worlds and targets CR under agent shift, without assuming a factorized state representation.
5. **Transition-Matching Permutation (IJCAI 2022)**: Augments TextWorld via discovered symmetries in phrases. Our approach augments via action-conditional counterfactual transitions and targets robustness under policy shift.

**Novelty Kill Search Summary:** Searched for combinations of “counterfactual branching + world model + text-based”, “branched rollouts + dynamics model training data”, “behavioral coverage + world model + LLM”, and “ScienceWorld admissible actions” (plus local KB/proposal searches). Found related counterfactual augmentation in offline RL (MoCoDA/RoCoDA) and branched rollouts in MBRL/LLM RL (MBPO/BREAD), but no prior work instantiating **targeted admissible-action branching** to improve **LLM text world model CR under agent policy shift** as of 2026-02-16.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Word2World | Shows multi-agent trajectories improve OOD CR | Requires collecting diverse agent trajectories | Replace multi-agent trajectories with environment-labeled one-step branches | Branches expand action support at realistic states at lower cost |
| MBPO | Uses short model rollouts for policy learning | Doesn’t improve world model training data; not CR-focused | Use branching to *augment world model training data* | Directly targets transition errors that cause W2R failures |
| DAgger | Queries expert on learner states | Needs expert labeling | Use environment as oracle for counterfactual next observations | Fully automated for simulators with admissible actions |
| MoCoDA | Counterfactual augmentation with factorized dynamics models | Needs structural assumptions | Use admissible action lists + replayed prefixes in text envs | Lower engineering; no factorized representation required |
| TextWorld permutation augmentation | Symmetry-based text augmentation | Only captures linguistic symmetries | Action-conditional counterfactual transitions | Targets dynamics coverage under policy shift rather than symmetry |

---

## Experiments

### Experimental Setup

**Repository / harness:** Word2World codebase (https://github.com/X1AOX1A/Word2World), which provides scripts for:
- Real-environment interaction
- World-model interaction (world model served via vLLM)
- World-model-to-real replay evaluation (`cal_wm2real.sh`)

**Benchmark:** SciWorld from the Word2World dataset.

**Baseline Ladder (REQUIRED):**
- **Published multi-agent coverage baseline (context):** Word2World Table 3 (single-agent vs mixed-agent training).
- **Training baseline:** expert-only world-model SFT (A).
- **Simple augmentation baseline:** random one-step branches at expert states (B).
- **Proposed method:** targeted one-step branches matched to an OOD agent’s action-type distribution (C).

(Traditional “prompting” and “best-of-N” baselines are not directly applicable here because the world model is trained and evaluated as a next-observation predictor; the closest inference-time baseline is the **published multi-agent coverage** setting from Word2World.)

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-7B (world model backbone) | 7B | https://huggingface.co/Qwen/Qwen2.5-7B | Train with LoRA; text-only |
| Qwen2.5-7B-Instruct (acting agent) | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Run via vLLM for Real and WM rollouts |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| `X1AOX1A/LLMasWorldModels` (SciWorld trajectories) | Expert trajectories for world-model SFT and evaluation files | Use 4K subset of training trajectories for low-data regime | https://huggingface.co/datasets/X1AOX1A/LLMasWorldModels | See HF dataset card |

**Other Resources (if applicable):**
- SciWorld environment server (FastAPI) from Word2World’s AgentGym environment packages.

**Resource Estimate** (must fit ≤768 A100 GPU-hours):
- **Data generation** (branch collection): CPU-bound; requires replaying prefixes. For 4K trajectories with 1 branch each, expect O(4K × avg_prefix_len) environment steps; parallelizable. **Wall-clock**: if avg_prefix_len≈50 and each step takes ~0.1–0.3s server-side, this is roughly 5–17 CPU-hours (plus overhead) and can be parallelized across workers.
- **Training**: 3 conditions × 3 seeds LoRA fine-tuning of a 7B model on ~4K trajectories + ~4K branches.
  - Expected to fit within ~200–500 A100 GPU-hours total; if over budget, reduce to 1–2K trajectories and/or fewer optimizer steps while keeping compute matched across B vs C.
- **Evaluation**:
  - Run the acting agent in SciWorld real env to compute Real (required once per seed if agent decoding is stochastic; otherwise deterministic).
  - Run the same agent against each world model (WM) and replay with `cal_wm2real.sh` (W2R).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SciWorld (Word2World protocol) | Text-based science tasks with simulator dynamics; environment provides admissible actions | **Real**, **WM**, **W2R**, **CR=W2R/Real** | test | https://huggingface.co/datasets/X1AOX1A/LLMasWorldModels | Word2World scripts (`scripts/interact_with_*`, `cal_wm2real.sh`) |

### Main Results

#### Comparability Rules (CRITICAL)

All rows must be directly comparable:
- Same base world-model backbone (Qwen2.5-7B)
- Same training compute budget (optimizer steps / tokens)
- Same number of counterfactual branches for B and C
- Same acting agent model + decoding settings
- Same SciWorld split and evaluation pipeline

#### Results Table

| Method | Base Model | Benchmark | Metric (CR; mean±std) | Source | Notes |
|---|---|---|---|---|---|
| Single-agent trajectories (4K GPT-4o) → OOD avg CR | Qwen2.5-7B world model | SciWorld | 0.83 (1 run) | Word2World Table 3 (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/6.5 Behavioral Coverage for Robust World Modeling.md`) | Not directly comparable to our open acting agent; included as context |
| Mixed-agent trajectories (1K × 4 ID agents) → OOD avg CR | Qwen2.5-7B world model | SciWorld | 0.91 (1 run) | Word2World Table 3 (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/6.5 Behavioral Coverage for Robust World Modeling.md`) | Not directly comparable to our open acting agent; included as context |
| **A. Expert-only (no branch)** | Qwen2.5-7B (LoRA) | SciWorld | **TBD** | - | 3 seeds |
| **B. Random-branch augmentation** | Qwen2.5-7B (LoRA) | SciWorld | **TBD** | - | 3 seeds |
| **C. Targeted-branch augmentation (TCBA)** | Qwen2.5-7B (LoRA) | SciWorld | **TBD** | - | 3 seeds |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Targeted vs random | Replace \(\rho(verb)\) with uniform weights | If targeting matters, TCBA > random |

### Experimental Rigor

**Variance & Reproducibility:**
- Train each condition with **3 random seeds** (e.g., `seeds=[0,1,2]`) and report mean±std.
- If acting-agent inference is stochastic, run Real/WM rollouts with matching seeds; otherwise fix `temperature=0` for determinism and report as deterministic.

**Validity & Controls:**
1. **Compute mismatch confound**: All conditions use matched training steps/tokens; B and C use the same number of counterfactual branches.
2. **Environment non-determinism / replay mismatch**: Before branch generation, verify that replaying a random subset of expert prefixes produces identical observations across two replays. If not, reduce branching to early steps or store intermediate observations and treat branching as approximate.
3. **Targeting circularity**: Targeting weights use only OOD-agent logs that are required for CR computation (Real rollouts), not an extra learned model.

**Sanity checks:**
- If no branches are generated (branch probability p=0), B and C must reduce to A.
- If \(\rho\) is set to 1 for all verbs, C must match B.

### Analysis (Optional)

- **Per-step observation divergence during W2R replay**: Using `cal_wm2real.py` outputs (world-model observation vs real env observation per step), compute a simple token-F1 similarity. Expect improvements concentrated on steps whose action verb is upweighted by \(\rho\).

---

## Success Criteria

**Hypothesis (directional):** Targeted counterfactual branches improve SciWorld CR under an OOD acting agent more than random counterfactual branches, under matched training compute.

**Decision Rule (concrete):**
- **Proceed** if \(CR_C - CR_B\) is **positive with a bootstrap 95% confidence interval excluding 0** on the SciWorld test set (reported across 3 training seeds).
- **Refute** if \(CR_C - CR_B\) is not distinguishable from 0 (CI includes 0) or if C underperforms B.
- **Pivot** if both B and C improve over A but C≈B: treat targeting as unnecessary and recommend the simpler random-branch augmentation; a follow-up pivot would be multi-step branching (length>1) if compute allows.

---

## Impact Statement

If successful, this work provides a low-cost alternative to collecting multi-agent trajectories for robust LLM world models in text environments: practitioners can expand behavioral coverage by sampling admissible counterfactual actions at expert-visited states, and further improve robustness by targeting branches toward the intended downstream agent’s action distribution. If targeting does not help, the result still clarifies that generic diversity (random branches) is sufficient and targeting is not worth the added complexity.

---

## References

- [From Word to World: Can Large Language Models be Implicit Text-based World Models?](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt) - Li et al., 2025
- [TextWorld: A Learning Environment for Text-based Games](https://arxiv.org/abs/1806.11532) - Côté et al., 2018
- [ScienceWorld: Is Your Agent Smarter Than a 5th Grader?](https://arxiv.org/abs/2203.07540) - Wang et al., 2022
- [Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/) - Wang et al., 2024
- [World Models](https://arxiv.org/abs/1803.10122) - Ha & Schmidhuber, 2018
- [Dyna: An Integrated Architecture for Learning, Planning, and Reacting](https://link.springer.com/article/10.1007/BF00115009) - Sutton, 1991
- [When to Trust Your Model: Model-Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253) - Janner et al., 2019
- [STEVE: Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675) - Buckman et al., 2018
- [PETS: Probabilistic Ensembles with Trajectory Sampling](https://arxiv.org/abs/1805.12114) - Chua et al., 2018
- [DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) - Ross et al., 2011
- [DADAgger: Disagreement-Augmented Dataset Aggregation](https://arxiv.org/abs/2301.01348) - 2023
- [MoCoDA: Model-based Counterfactual Data Augmentation](https://arxiv.org/abs/2210.11287) - Pitis et al., 2022
- [RoCoDA: Counterfactual Data Augmentation for Robot Learning](https://rocoda.github.io/) - 2024
- [On Rollouts in Model-Based Reinforcement Learning](https://arxiv.org/abs/2501.16918) - Frauenknecht et al., 2025
- [Masked Model-based Actor-Critic (M2AC)](https://arxiv.org/abs/2010.04893) - Pan et al., 2020
- [Self-Improving World Modelling with Latent Actions (SWIRL)](https://arxiv.org/abs/2602.06130) - Qiu et al., 2026
- [BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL](https://arxiv.org/abs/2506.17211) - 2025
- [Data Augmentation for Learning to Play in Text-Based Games](https://www.ijcai.org/proceedings/2022/0436.pdf) - Kim & Kim, 2022
- [Is Your LLM Secretly a World Model of the Internet?](https://arxiv.org/abs/2411.06559) - Gu et al., 2024
- [DynaWeb](https://arxiv.org/abs/2601.22149) - Ding et al., 2026
- [R-WoM: Retrieval-augmented World Model for Computer-use Agents](https://arxiv.org/abs/2510.11892) - Mei et al., 2025
