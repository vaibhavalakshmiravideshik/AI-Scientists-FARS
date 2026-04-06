# untitled

# Grounded Rao–Kupper Leaderboards for Music Arena (Modeling BOTH_BAD as an Outside Option)

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Text-to-music (TTM) and lyrics-to-song models have improved rapidly, but evaluation remains difficult because music quality is subjective and automatic metrics correlate imperfectly with human taste. A recent response is **arena-style live evaluation**, where real users compare two anonymized model outputs for the same prompt and vote on which is better.

**Music Arena** is a live evaluation platform for TTM that collects pairwise user preferences on user-written prompts and aggregates them into a public leaderboard ([Music Arena](./references/Music-Arena-Live-Evaluation-for-Text-to-Music/meta/meta_info.txt)). Importantly, Music Arena’s UI allows **four outcomes** for each battle: **A is better**, **B is better**, **Tie**, and **Both are bad** (BOTH_BAD). The BOTH_BAD option is particularly valuable in music: users often judge that *neither* sample is acceptable, and forcing a choice can obscure how often a model produces unusable outputs.

In other domains (notably LLM chat), arena leaderboards have become a de facto evaluation standard, and recent work has improved the statistical modeling of arena votes. For example, **Prompt-to-Leaderboard (P2L)** trains a model to output prompt-conditional leaderboard coefficients and introduces a **grounded Rao–Kupper (GRK)** model to incorporate “Tie (both bad)” as an explicit outside option ([Prompt-to-Leaderboard](./references/Prompt-to-Leaderboard/meta/meta_info.txt)). Separately, **A Statistical Framework for Ranking LLM-Based Chatbots** shows that modeling ties explicitly (and with pair-specific tie factors) substantially improves fit to arena data ([Ranking LLM Chatbots](./references/A-Statistical-Framework-for-Ranking-LLM-Based-Chatbots/meta/meta_info.txt)).

### The Problem

Despite Music Arena collecting BOTH_BAD outcomes, a standard Bradley–Terry (BT) leaderboard is inherently **relative**: it models only the probability that A beats B, and typically treats ties as half-wins or discards them. This creates two practical problems:

1. **Absolute acceptability is not captured.** A model can rank well by being slightly better than other models on average, while still frequently producing outputs that users label BOTH_BAD. For product decisions (which model should be deployed in a music app?), this “unacceptability rate” matters.

2. **BOTH_BAD carries information that is easy to waste.** If BOTH_BAD is treated as a tie or dropped, the leaderboard model cannot learn which model pairs systematically trigger “both bad” outcomes, nor can it estimate model-specific failure propensity.

A naive fix is to fit a generic 4-outcome multinomial model. However, in arena settings, data are often sparse (especially per-month releases), and over-parameterized models can overfit. The question is whether a **structured inductive bias** for BOTH_BAD is helpful in Music Arena’s small-data regime.

### Key Insight and Hypothesis

**Key insight:** The grounded Rao–Kupper (GRK) model from P2L turns BOTH_BAD into an explicit “outside option” that anchors the scale: if both models have low latent quality, the probability of BOTH_BAD increases. This converts BOTH_BAD from an ignored UI artifact into a statistical signal about absolute quality.

**Hypothesis:** On Music Arena data, a GRK leaderboard fit will (i) improve held-out predictive likelihood for the 4-way outcome distribution and (ii) produce a more stable and better-calibrated estimate of model acceptability than a Bradley–Terry baseline and a comparably expressive 4-outcome baseline that decouples “skill” from “badness”.

We could be wrong if BOTH_BAD is mostly random noise (user mood / prompt mismatch), if BOTH_BAD mostly reflects prompt difficulty rather than model quality, or if modeling BOTH_BAD as tightly coupled to skill is an incorrect inductive bias for music (e.g., some models may be usually good but occasionally catastrophically bad).

---

## Proposed Approach

### Overview

We propose to adapt **grounded Rao–Kupper** modeling (originally introduced for Chatbot Arena in P2L) to **Music Arena**. The output is:

- a standard “Arena score” per model (relative skill), and
- an **absolute-quality signal**: the model’s implied propensity to avoid BOTH_BAD outcomes.

The approach is inference-only and purely statistical: fit the model by maximum likelihood on the released battle data.

### Method Details

#### Data
Each battle compares two systems \(i\) and \(j\) and produces one label \(y \in \{i\text{ wins}, j\text{ wins}, \text{tie}, \text{both\_bad}\}\). Music Arena also provides metadata such as whether the sample is instrumental and listening-time logs, but the core method uses only the categorical vote.

#### Grounded Rao–Kupper (GRK) model
Let each model \(k\) have a scalar score \(\beta_k\) and define \(\varphi_k = \exp(\beta_k)\). Let \(\lambda \ge 1\) be a tie/indifference parameter.

For a pair \((i,j)\), GRK defines:

- \(P(i\text{ wins}) = \frac{\varphi_i}{\varphi_i + \lambda \varphi_j + 1}\)
- \(P(j\text{ wins}) = \frac{\varphi_j}{\varphi_j + \lambda \varphi_i + 1}\)
- \(P(\text{both\_bad}) = \frac{1}{1 + \varphi_i + \varphi_j}\)
- \(P(\text{tie}) = 1 - P(i\text{ wins}) - P(j\text{ wins}) - P(\text{both\_bad})\)

This “grounds” the model by introducing a fictitious baseline competitor of fixed score 0 (the constant 1 term), so that low-quality pairs have higher BOTH_BAD probability ([Prompt-to-Leaderboard, Sec. 2.2](./references/Prompt-to-Leaderboard/sections/2.2 Prompt-to-Regression.md)).

#### Matched 4-outcome baseline (AB-MNL)
To test whether GRK’s coupling of BOTH_BAD to skill is actually the right inductive bias, we compare against a similarly structured 4-outcome model that **decouples skill and badness**.

**Intuition:** AB-MNL allows a model to be *comparatively strong* (often wins head-to-head) while still being *absolutely unreliable* (sometimes produces outputs users label BOTH_BAD), by giving each model its own "badness" parameter.

Let each model \(k\) have:
- skill \(\beta_k\) (as above), and
- badness \(\rho_k\), with \(\psi_k = \exp(\rho_k)\).

Define a 4-way multinomial model over outcomes using logits:
- win for \(i\): \(u_{i} = \beta_i\)
- win for \(j\): \(u_{j} = \beta_j\)
- tie: \(u_{tie} = \tau + \tfrac{1}{2}(\beta_i + \beta_j)\) (Davidson-style tie mass)
- both\_bad: \(u_{bad} = \kappa + \tfrac{1}{2}(\rho_i + \rho_j)\) (symmetric, model-specific)

Then \(P(y)=\text{softmax}(u_i,u_j,u_{tie},u_{bad})\). We fit by regularized maximum likelihood with **L2 penalty on \(\rho\)** (tuned by cross-validation) to control overfitting in small data.

This baseline has a clear interpretation (“some models are more likely to be unacceptable even if they sometimes win comparisons”) and provides a matched-outcome comparison that avoids the tautology of comparing 4-way vs 3-way label spaces.

### Key Innovations

- **Domain transfer of grounded arena modeling to music**: Music Arena already collects BOTH_BAD but does not specify a BOTH_BAD-aware leaderboard model; we propose a principled outside-option model.
- **A decisive test of inductive bias**: compare GRK’s coupling (BOTH_BAD driven by low skill) against a decoupled model-specific badness alternative under matched 4-way likelihood.
- **Actionable output**: beyond rank, estimate a model’s “unacceptability” tendency (probability of BOTH_BAD) to inform deployment decisions.

---

## Related Work

### Field Overview

**Live evaluation and arena leaderboards.** Arena-style evaluation platforms collect large-scale pairwise preferences and aggregate them into leaderboards, often via Bradley–Terry regression or Elo-like scores. This paradigm began with Chatbot Arena for LLMs and has expanded to other modalities. Statistical questions include how to handle ties, position bias, and more complex outcome spaces.

**Paired-comparison models with ties and abstentions.** Classical Bradley–Terry models assume binary outcomes. Rao–Kupper and Davidson extend BT to incorporate ties via additional parameters. Recent work revisits these models in modern ML evaluation and preference optimization settings.

**Music generation evaluation.** Text-to-music evaluation uses both automatic metrics (e.g., Frechet Audio Distance) and human preference studies; Music Arena is an example of making human preference data renewable for the community.

### Related Papers

- **[Music Arena: Live Evaluation for Text-to-Music](./references/Music-Arena-Live-Evaluation-for-Text-to-Music/meta/meta_info.txt)**: Introduces a live evaluation platform for text-to-music and releases preference data including BOTH_BAD.
- **[Prompt-to-Leaderboard](./references/Prompt-to-Leaderboard/meta/meta_info.txt)**: Produces prompt-conditional leaderboards for LLMs and introduces grounded Rao–Kupper for “tie (both bad)”.
- **[A Statistical Framework for Ranking LLM-Based Chatbots](./references/A-Statistical-Framework-for-Ranking-LLM-Based-Chatbots/meta/meta_info.txt)**: Improves arena modeling via factored tie models and covariance structures.
- **[On Extending Direct Preference Optimization to Accommodate Ties](./references/On-Extending-Direct-Preference-Optimization-to-Accommodate-Ties/meta/meta_info.txt)**: Derives DPO variants based on Rao–Kupper and Davidson tie models; provides useful formulations.
- **[Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)**: Introduces the arena paradigm and large-scale preference data for LLMs.
- **[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)**: Early analysis connecting Arena-style evaluation to benchmark design.
- **[Prediction-Powered Ranking of Large Language Models](https://arxiv.org/abs/2402.17826)**: Uses statistical techniques to reduce labeling cost while ranking models.
- **[Prompt-specific Bayesian preference modeling for LLMs (am-Elo)](https://arxiv.org/abs/2505.03475)**: A stable framework for arena-based evaluation with improved statistical treatment.
- **[Chatbot Arena Meets Nuggets](./references/Chatbot-Arena-Meets-Nuggets-Towards-Explanations-and-Diagnostics-in-the-Evaluation-of-LLM-Responses/meta/meta_info.txt)**: Uses nugget-based decomposition to analyze arena battles; motivates richer outcome modeling.
- **[Bradley & Terry (1952)](https://doi.org/10.1093/biomet/39.3-4.324)**: Original Bradley–Terry paired-comparison model.
- **[Rao & Kupper (1967)](https://doi.org/10.1080/01621459.1967.10482930)**: Extends BT to handle ties via a threshold mechanism.
- **[Davidson (1970)](https://doi.org/10.1080/01621459.1970.10481082)**: Extends BT to handle ties based on Luce’s choice axiom.
- **[Plackett (1975)](https://doi.org/10.1111/j.2517-6161.1975.tb00933.x)**: Plackett–Luce model foundation for choice/ranking.
- **[Luce (1959)](https://psycnet.apa.org/record/1960-35001-000)**: Choice axiom underpinning many paired-comparison models.
- **[Rank Centrality](https://arxiv.org/abs/1209.1688)**: Spectral method for ranking from pairwise comparisons.
- **[Estimation from Pairwise Comparisons: Topology Dependence](https://arxiv.org/abs/1506.07246)**: Minimax bounds linking comparison graph topology to ranking accuracy.
- **[Crowd-BT](https://arxiv.org/abs/1503.06150)**: Extends BT to crowdsourced settings with annotator reliability.
- **[DPO: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)**: Popular preference optimization method derived from Bradley–Terry assumptions.
- **[FlagEval-Arena](https://aclanthology.org/2025.acl-demo.56/)**: A side-by-side evaluation platform showing the importance of UI and outcome modeling for ties and both-bad.
- **[MusicGen](https://arxiv.org/abs/2306.05284)**: A strong open text-to-music baseline model; highlights need for scalable evaluation.
- **[MusicLM](https://arxiv.org/abs/2301.11325)**: Large-scale text-to-music model motivating human preference evaluation.
- **[AudioLDM](https://arxiv.org/abs/2301.12503)**: Text-to-audio generation model; shows broader audio evaluation challenges.
- **[Stable Audio Open](https://arxiv.org/abs/2407.14358)**: Open text-to-audio model; motivates public evaluation infrastructure.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Arena-style live evaluation | Collect pairwise human preferences at scale and fit a global ranking model | [Music Arena](./references/Music-Arena-Live-Evaluation-for-Text-to-Music/meta/meta_info.txt), [Chatbot Arena](https://arxiv.org/abs/2403.04132) | Arena score / BT coefficients | Outcome modeling choices (ties/both-bad) can waste signal |
| Paired-comparison models with ties | Extend BT to allow ties via additional parameters | [Rao–Kupper](https://doi.org/10.1080/01621459.1967.10482930), [Davidson](https://doi.org/10.1080/01621459.1970.10481082), [DPO w/ ties](./references/On-Extending-Direct-Preference-Optimization-to-Accommodate-Ties/meta/meta_info.txt) | Pairwise likelihood / MLE | A single tie parameter may be insufficient; does not address BOTH_BAD |
| Modern arena statistics | Better fit and interpretability for arena data via richer tie models and covariance | [Ranking LLM Chatbots](./references/A-Statistical-Framework-for-Ranking-LLM-Based-Chatbots/meta/meta_info.txt), [P2L](./references/Prompt-to-Leaderboard/meta/meta_info.txt) | Held-out likelihood, calibration | Most work is in LLM chat; transfer to music is underexplored |

### Closest Prior Work

- **Prompt-to-Leaderboard (P2L)** ([2502.14855](./references/Prompt-to-Leaderboard/meta/meta_info.txt)): Introduces grounded Rao–Kupper to incorporate “Tie (both bad)” for Chatbot Arena prompt-conditional modeling. Our proposal transfers the same grounded idea to **music**, and tests whether the GRK inductive bias is appropriate for Music Arena’s BOTH_BAD outcomes.
- **A Statistical Framework for Ranking LLM-Based Chatbots** ([2412.18407](./references/A-Statistical-Framework-for-Ranking-LLM-Based-Chatbots/meta/meta_info.txt)): Shows that modeling ties carefully (factored tie models) improves fit and reveals competitor structure. It does not model BOTH_BAD/outside options. Our proposal focuses specifically on the BOTH_BAD outcome category present in Music Arena.
- **Music Arena** ([2507.20900](./references/Music-Arena-Live-Evaluation-for-Text-to-Music/meta/meta_info.txt)): Defines the platform and data collection. The paper states BT aggregation in general terms but does not specify a BOTH_BAD-aware leaderboard model.

**Novelty Kill Search Summary:** Searched locally over all finalized proposals and all agents’ drafts for “Music Arena”, “music-arena”, “Prompt-to-Leaderboard”, “P2L”, “grounded Rao”, and “Rao-Kupper”; found no matches (as of 2026-02-28). Web searches for “music-arena grounded Rao-Kupper” and “Music Arena both bad leaderboard” found no evidence that GRK-style modeling is used for Music Arena leaderboards.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [Music Arena](./references/Music-Arena-Live-Evaluation-for-Text-to-Music/meta/meta_info.txt) | Collects 4-way votes including BOTH_BAD; suggests BT aggregation | Does not specify how BOTH_BAD should inform the leaderboard | Fit an explicit outside-option model | Uses BOTH_BAD to estimate absolute acceptability |
| [P2L](./references/Prompt-to-Leaderboard/meta/meta_info.txt) | Grounded RK for “tie (both bad)” in Chatbot Arena | Not evaluated on music; prompt-conditional focus | Apply GRK to Music Arena global ranking | Tests if inductive bias transfers to music |
| [Ranking LLM Chatbots](./references/A-Statistical-Framework-for-Ranking-LLM-Based-Chatbots/meta/meta_info.txt) | Factored tie models + covariance improve arena fit | Focuses on ties, not BOTH_BAD | Add outside option modeling | BOTH_BAD is central for music acceptability |

---

## Experiments

### Experimental Setup

**Dataset:** HuggingFace `music-arena/music-arena-dataset` (public release of Music Arena battles). The dataset includes fields `preference ∈ {A,B,TIE,BOTH_BAD}` and system identifiers (`system_a`, `system_b`). Some audio files are missing for specific systems (e.g., sao/sao-small), but the categorical preference labels remain available.

**Splits:** Use a chronological split by `date` to avoid leakage across repeated users/prompts:
- Train: earliest 70% of battles by date
- Test: latest 30% of battles by date

Also report results separately for the `is_instrumental=true` subset and `is_instrumental=false` subset as a minimal generalization check.

**Methods compared (3 conditions):**
1. **BT baseline (practice)**: Standard Bradley–Terry with ties handled as 0.5 outcome; BOTH_BAD collapsed into tie (document this explicitly).
2. **AB-MNL baseline (4-way, decoupled badness)**: softmax over {A,B,tie,both_bad} with parameters {β_i, ρ_i, τ, κ}; L2 regularization on ρ tuned by CV.
3. **Ours: GRK**: grounded Rao–Kupper as defined above; fit {β_i, λ} by MLE.

**Optimization details:**
- Fit all models by maximum likelihood (BFGS or L-BFGS) with an identifiability constraint (e.g., \(\sum_i \beta_i=0\)).
- For AB-MNL, tune the regularization strength on ρ via 5-fold CV on the training split.

**Baseline Ladder (REQUIRED):**
- Level 1: BT as commonly used for arena leaderboards.
- Level 5 (closest): P2L’s grounded Rao–Kupper transferred to Music Arena (ours).

**Resource Estimate**:
- **Compute budget**: CPU-only; <1 GPU-hour (dominated by data loading + a few BFGS fits + bootstrap).
- **GPU memory**: none required.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Music Arena (global) | Pairwise TTM battles with 4-way votes | 4-way NLL, per-class NLL, ECE/Brier (BOTH_BAD) | time-based train/test | https://huggingface.co/datasets/music-arena/music-arena-dataset | custom (simple MLE + scoring) |
| Music Arena (instrumental vs vocal) | Same dataset, stratified by `is_instrumental` | same metrics | same | same | same |

### Main Results

#### Results Table

| Method | Base Model | Benchmark | 4-way NLL ↓ | BOTH_BAD Brier ↓ | Source | Notes |
|---|---|---|---:|---:|---|---|
| BT baseline | N/A | Music Arena (global) | TBD | TBD | - | BOTH_BAD collapsed into tie |
| AB-MNL (decoupled badness) | N/A | Music Arena (global) | TBD | TBD | - | L2 on ρ tuned by CV |
| **Ours: GRK** | N/A | Music Arena (global) | TBD | TBD | - | outside-option grounded model |

(Repeat the same table for instrumental-only and vocal-only subsets.)

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| GRK w/o grounding | Replace BOTH_BAD model with a constant rate | Worse NLL if BOTH_BAD contains model-quality signal |
| AB-MNL w/ strong ρ regularization | Increase L2 on ρ | Should approach GRK if the coupling is mostly a regularization effect |

### Experimental Rigor

- **Variance & reproducibility**: Use bootstrap resampling over battles (≥1,000 bootstrap samples) to estimate uncertainty of NLL differences and rank correlations; optimization itself is deterministic.
- **Underpower gate**: If test-set BOTH_BAD count <50 (or base rate <5%), treat BOTH_BAD calibration as too noisy and refute/pivot to a larger time window (next monthly release).
- **Top confounders**:
  1) BOTH_BAD depends mostly on prompt difficulty → partially controlled by time split and by comparing per-class NLL (does the model improve specifically on BOTH_BAD events?).
  2) Overfitting in AB-MNL due to extra parameters → controlled by CV-tuned L2 on ρ and reporting sensitivity to regularization.

### Analysis (Optional)

- Compare per-model implied acceptability (1 − predicted BOTH_BAD rate when paired against an average opponent) vs empirical BOTH_BAD frequency.
- Examine whether the GRK constraint is approximately recovered by AB-MNL (e.g., whether \(\rho_i\) correlates with \(-\beta_i\)).

---

## Success Criteria

**Hypothesis** (directional): GRK improves held-out 4-way likelihood and produces better-calibrated BOTH_BAD probabilities than BT and AB-MNL in Music Arena’s small-data regime.

**Decision Rule** (concrete):
- **Proceed** if GRK beats AB-MNL on test-set 4-way NLL by a margin whose bootstrap 95% CI excludes 0, and BOTH_BAD Brier score is also improved or unchanged.
- **Pivot** if AB-MNL wins overall but \(\rho_i\) strongly correlates with \(-\beta_i\): interpret as evidence that GRK’s coupling is approximately true but hard-coding it is unnecessary; pivot to a simpler regularized model or to reporting acceptability as a separate axis.
- **Refute** if GRK does not beat AB-MNL on NLL (within noise) and BOTH_BAD calibration shows no improvement over BT.

---

## Impact Statement

If successful, this work provides Music Arena (and similar generative-model arenas) a simple, principled leaderboard model that uses BOTH_BAD votes to estimate **absolute acceptability**, not just relative preference. Practitioners choosing a music generation model for deployment could prioritize systems that are not only top-ranked but also have low predicted “both bad” probability.

---

## References

- [Music Arena: Live Evaluation for Text-to-Music](./references/Music-Arena-Live-Evaluation-for-Text-to-Music/meta/meta_info.txt) - Kim et al., 2025
- [Prompt-to-Leaderboard](./references/Prompt-to-Leaderboard/meta/meta_info.txt) - Frick et al., 2025
- [A Statistical Framework for Ranking LLM-Based Chatbots](./references/A-Statistical-Framework-for-Ranking-LLM-Based-Chatbots/meta/meta_info.txt) - Ameli et al., 2024
- [On Extending Direct Preference Optimization to Accommodate Ties](./references/On-Extending-Direct-Preference-Optimization-to-Accommodate-Ties/meta/meta_info.txt) - Chen et al., 2024
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) - Chiang et al., 2024
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) - Zheng et al., 2023
- [Prediction-Powered Ranking of Large Language Models](https://arxiv.org/abs/2402.17826) - Chen et al., 2024
- [Prompt-specific Bayesian preference modeling for LLMs (am-Elo)](https://arxiv.org/abs/2505.03475) - 2025
- [Estimating Consumer Preferences for LLMs: Evidence from LMArena](./references/Estimating-Consumer-Preferences-for-LLMs-Evidence-from-LMArena/meta/meta_info.txt) - Caoui, 2026 (notes on outside-option modeling)
- [Chatbot Arena Meets Nuggets](./references/Chatbot-Arena-Meets-Nuggets-Towards-Explanations-and-Diagnostics-in-the-Evaluation-of-LLM-Responses/meta/meta_info.txt) - Sharifymoghaddam et al., 2025
- [Rank analysis of incomplete block designs: I. the method of paired comparisons](https://doi.org/10.1093/biomet/39.3-4.324) - Bradley & Terry, 1952
- [Ties in paired-comparison experiments: A generalization of the Bradley–Terry model](https://doi.org/10.1080/01621459.1967.10482930) - Rao & Kupper, 1967
- [On extending the Bradley–Terry model to accommodate ties](https://doi.org/10.1080/01621459.1970.10481082) - Davidson, 1970
- [The analysis of permutations](https://doi.org/10.1111/j.2517-6161.1975.tb00933.x) - Plackett, 1975
- [Individual Choice Behavior](https://psycnet.apa.org/record/1960-35001-000) - Luce, 1959
- [Rank Centrality: Ranking from Pairwise Comparisons](https://arxiv.org/abs/1209.1688) - Negahban et al., 2012
- [Estimation from Pairwise Comparisons: Sharp Minimax Bounds with Topology Dependence](https://arxiv.org/abs/1506.07246) - Shah et al., 2015
- [Crowd-BT: Distributed Inference of Preferences](https://arxiv.org/abs/1503.06150) - Chen et al., 2015
- [DPO: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2024
- [FlagEval-Arena](https://aclanthology.org/2025.acl-demo.56/) - 2025
- [MusicGen: A Simple and Controllable Music Generation Model](https://arxiv.org/abs/2306.05284) - Copet et al., 2023
- [MusicLM: Generating Music From Text](https://arxiv.org/abs/2301.11325) - Agostinelli et al., 2023
- [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://arxiv.org/abs/2301.12503) - Liu et al., 2023
- [Stable Audio Open](https://arxiv.org/abs/2407.14358) - 2024
