# untitled

# Deflated-RankICIR: Multiple-Testing-Aware Factor Selection for LLM-Driven Alpha Mining

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS (Conference on Neural Information Processing Systems), ICML (International Conference on Machine Learning), ICLR (International Conference on Learning Representations), ACL (Annual Meeting of the Association for Computational Linguistics), EMNLP (Conference on Empirical Methods in Natural Language Processing), or similar top AI conferences
- **Automation constraint**: Fully automated evaluation (no human labeling, manual trading judgment, or discretionary factor inspection).

## Introduction

### Context and Motivation

Quantitative investment often relies on **alpha factors**: numerical signals that attempt to predict future **cross-sectional returns** (relative returns across a universe of stocks). A common workflow is **factor mining**, where a system proposes many candidate factor formulas (often as interpretable expressions over price/volume features), evaluates them on historical data, and then selects a small subset to form a trading strategy.

Recent papers automate factor mining using **large language models (LLMs)** that generate hypotheses, write factor code, and iterate based on backtest feedback. For example, **[QuantaAlpha](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/meta/meta_info.txt)** is an LLM-based multi-agent framework that (i) proposes diverse market hypotheses, (ii) instantiates each hypothesis into executable factor formulas under a constrained operator library, and (iii) evaluates factors via backtesting.

QuantaAlpha evaluates on **CSI 300 (CSI300)**, a Chinese A-share equity index of 300 large-cap stocks, using **Qlib** (an open-source quantitative research and backtesting framework). QuantaAlpha also releases a reproducible dataset and codebase (**[QuantaAlpha GitHub](./references/GitHub---QuantaAlpha-QuantaAlpha/meta/meta_info.txt)**, **[QuantaAlpha/qlib_csi300 dataset](./references/QuantaAlpha-qlib_csi300-Datasets-at-Hugging-Face/meta/meta_info.txt)**), which makes it possible to test methodological changes without rebuilding the full infrastructure.

A core statistical risk in any factor-mining system is that it performs a **search over many candidates**. When selection is based on the best validation backtest score, the chosen factor is likely to look better on validation than it truly is out of sample. In finance, this is usually discussed as **data snooping** or **multiple testing**: even if all candidates are pure noise, searching over enough candidates can produce an apparently strong winner by chance.

### The Problem

LLM-driven alpha mining papers typically optimize and report **uncorrected** validation/test metrics. Commonly reported metrics include:

- **Information Coefficient (IC)**: the daily Pearson correlation between a factor signal and next-day returns across stocks (higher is better).
- **Rank IC (RankIC)**: the daily Spearman rank correlation between a factor signal and next-day returns across stocks (higher is better).
- **IC information ratio (ICIR)** and **RankIC information ratio (RankICIR)**: mean(IC)/std(IC) and mean(RankIC)/std(RankIC) over days (higher is better; indicates stability of the signal).
- **Information Ratio (IR)** (also called Sharpe ratio in some contexts): annualized mean daily excess return divided by annualized return volatility (higher is better; indicates risk-adjusted performance).
- **Annualized excess return (ARR)**: annualized mean excess return over a benchmark (higher is better).
- **Maximum drawdown (MDD)**: the maximum peak-to-trough portfolio loss during the backtest (lower is better).
- **Calmar ratio (CR)**: ARR divided by MDD (higher is better).

For example, QuantaAlpha reports on CSI300 an IR of **2.8797** (higher is better) and an ARR of **23.77%** (higher is better) when using a DeepSeek-V3.2 backbone (**[QuantaAlpha Methods section](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/sections/Methods.md)**).

Importantly, QuantaAlpha is explicitly a multi-round search procedure: it uses ten parallel exploration directions, runs five evolutionary iterations, and attempts to generate multiple factor expressions per hypothesis (**[QuantaAlpha Appendix B](<./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/sections/B Algorithm Configuration.md>)**). Even with a train/validation/test split, repeatedly using validation to rank and filter many candidates can inflate the expected “best” validation score (a winner’s curse effect, where selecting the best of many noisy estimates leads to systematic overestimation).

The finance literature provides standard tools to correct for selection among many trials, but these tools are rarely integrated into the **selection step** of LLM-based factor mining:

- **Deflated Sharpe Ratio (DSR)**: a multiple-testing-aware estimate of the probability that an observed Sharpe ratio reflects genuine skill rather than selection among many trials and non-normal return distributions (**[Bailey & López de Prado, 2014](./references/The-Deflated-Sharpe-Ratio-Correcting-for-Selection-Bias-Backtest-Overfitting-and-Non-Normality/meta/meta_info.txt)**).
- **Reality Check** and **Superior Predictive Ability (SPA) test**: stationary-bootstrap-based tests that control for data snooping when comparing many strategies (**[White, 2000](./references/A-Reality-Check-for-Data-Snooping/meta/meta_info.txt)**; [Hansen, 2005](https://doi.org/10.1198/073500105000000063)).

Operationally, this gap matters because practitioners may treat high IR/ARR from agentic factor-mining systems as evidence of robustness, but those numbers may be statistically fragile when the system has searched over many candidates.

### Key Insight and Hypothesis

**Key insight**: QuantaAlpha-style pipelines already output a large factor library (a JSON (JavaScript Object Notation) file) containing, for each candidate factor, its validation-period predictive metrics (including the full daily RankIC time series). This makes it possible to compute **multiple-testing-aware factor scores** as a post-processing step and use them as a direct replacement for validation-metric ranking, without changing the mining agents, operator library, or backtesting engine.

**Hypothesis**: Replacing the standard validation ranking metric (e.g., RankICIR) with a **Deflated-RankICIR score** (Deflated Sharpe Ratio applied to the daily RankIC time series, with an effective-trials correction for selection among many candidates) will select factor pools whose out-of-sample strategy performance is:

1. **At least as strong** as a strong uncorrected baseline that ranks by RankICIR (measured by test-period IR and ARR).
2. **More statistically defensible** under multiple-testing-aware significance measures (e.g., higher DSR and/or lower SPA p-values, where a p-value is the probability of observing an effect at least this large under a no-skill null hypothesis; lower is stronger evidence).

Why we could be wrong:

- QuantaAlpha’s existing complexity and redundancy controls might already mitigate severe winner’s-curse inflation, leaving limited room for explicit multiple-testing correction.
- DSR calibration assumptions (e.g., stable distribution moments, approximate asymptotic behavior) might be misaligned with non-stationary financial time series.
- A DSR penalty might over-penalize high-variance but genuinely predictive factors, reducing test-period IR.

---

## Proposed Approach

### Overview

We propose **Deflated-RankICIR selection**: a post-hoc ranking rule for factor pool construction that explicitly accounts for multiple testing during factor selection.

Workflow:

1. Run QuantaAlpha once with a fixed search budget to produce a candidate factor library.
2. Using the same candidate set, construct factor pools using different ranking scores under the same redundancy constraint.
3. Run QuantaAlpha’s independent backtest on the held-out test period for each pool and compare out-of-sample strategy performance and statistical significance.

### Method Details

#### Candidate set and pool construction

Let the mined candidate factor set have size **M** (all factors with a valid validation RankIC time series).

We follow QuantaAlpha’s pool construction style (correlation-based de-duplication) to build a fixed-size pool:

- Pool size **K**: default K=50, matching QuantaAlpha’s TopkDropout strategy (a long-only portfolio rule that holds the top-k stocks by predicted score and replaces `n_drop` holdings each day) and its typical pool sizes; if M<K, set K=M.
- Redundancy filter: greedily include factors in ranked order, skipping any factor whose absolute correlation with any already-selected factor exceeds **0.7** (matching QuantaAlpha’s stated redundancy controls).
  - Default correlation: Pearson correlation between factors’ validation-period RankIC time series.

#### Ranking scores

All experimental conditions use the same mined candidate set; only the ranking score changes.

- **A (QuantaAlpha-style mean RankIC)**: score = mean daily RankIC on the validation period (higher is better).
- **B (strong simple baseline RankICIR)**: score = mean(RankIC)/std(RankIC) on the validation period (higher is better).
- **C (ours: Deflated-RankICIR)**: score = Deflated Sharpe Ratio computed on the validation RankIC time series, using an effective-trials correction for selection among many candidates (higher DSR is more significant after correction).

#### Deflated-RankICIR computation

We adapt the Deflated Sharpe Ratio (DSR) framework to the validation RankIC time series for each factor.

For each factor i:

1. Compute the daily RankIC series \(r_{i,t}\) on the validation period.
2. Compute the RankIC information ratio \(\mathrm{SR}_i = \mathrm{mean}(r_{i,t}) / \mathrm{std}(r_{i,t})\) as the “Sharpe-like” statistic for RankIC.
3. Estimate skewness and kurtosis of \(r_{i,t}\) (DSR explicitly corrects for non-normality).
4. Estimate the number of effective independent trials \(\hat{N}\) (the multiple-testing correction strength) based on the dependence among candidate factors:
   - Compute the equal-weight average correlation \(\hat{\rho}\) between candidate factors’ validation RankIC series.
   - Use Bailey & López de Prado’s interpolation (Appendix A.3, Eq. 9) to map \((M, \hat{\rho})\) to \(\hat{N}\):
     - \(\hat{N} = \hat{\rho} + (1 - \hat{\rho}) \cdot M\) (**[DSR Appendix A.3](<./references/The-Deflated-Sharpe-Ratio-Correcting-for-Selection-Bias-Backtest-Overfitting-and-Non-Normality/sections/A.3. ESTIMATING THE NUMBER OF INDEPENDENT TRIALS.md>)**).
5. Compute the DSR score for factor i using \(\mathrm{SR}_i\), sample length, skewness, kurtosis, and \(\hat{N}\) (per Bailey & López de Prado, 2014).

We use DSR as the ranking score for pool construction.

#### Sensitivity analysis for \(\hat{N}\)

Because \(\hat{N}\) is the main parameter controlling the strength of the multiple-testing correction, we treat it as the primary ablation:

- **C1 (default)**: \(\hat{N} = \hat{\rho} + (1-\hat{\rho})M\)
- **C2 (maximally conservative)**: \(\hat{N} = M\)
- **C3 (control: no correction)**: \(\hat{N} = 1\)

We report how ranking stability and test-period performance change under these settings.

#### Statistical comparison on the test period

We evaluate each pool using QuantaAlpha’s independent backtest on the held-out test window (CSI300 test period is Jan 2022–Dec 2025 in QuantaAlpha’s experimental setup; **[QuantaAlpha Self-Evolution section](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/sections/Self-Evolution.md)**).

To quantify uncertainty for dependent daily returns, we use a time-series bootstrap:

- **Stationary bootstrap** (Politis & Romano, 1994; resampling blocks of consecutive days with random block lengths to preserve time-series dependence) to compute confidence intervals for IR and for IR differences (e.g., \(\mathrm{IR}_C - \mathrm{IR}_B\)).
- Optionally, **SPA p-values** (Hansen, 2005) for “best of {A, B, C}” comparisons.

#### Diagnostics

To reduce attribution ambiguity and validate the RankIC-to-DSR adaptation:

- **RankIC distribution summary**: report skewness/kurtosis statistics of validation RankIC series across candidates.
- **Dependence sensitivity**: estimate \(\hat{\rho}\) over the full validation window and over rolling windows (e.g., quarterly), and report sensitivity of \(\hat{N}\) and ranking stability.
- **Winner’s-curse diagnostic**: scatter plot of validation ranking score vs. test-period IR across factors, and compare how often each ranking method selects extreme validation winners that fail out of sample.

### Key Innovations

- **Multiple-testing-aware selection in LLM factor mining**: integrates a standard finance correction (DSR) directly into the factor selection objective used by LLM-driven mining pipelines.
- **Controlled experiment design**: isolates the effect of selection by holding the mined candidate library fixed and changing only the ranking criterion.
- **Low-overhead integration**: requires no changes to the factor generation agents or backtesting engine; only post-hoc scoring and pool construction are modified.

---

## Related Work

### Field Overview

Automated alpha mining spans (i) formulaic factor generation methods (e.g., genetic programming and reinforcement learning), (ii) LLM-based agent frameworks that generate hypotheses and factor code, and (iii) evaluation methodologies for robustness and statistical validity.

Most factor-mining systems rank candidates using point estimates on a validation window (IC/RankIC or strategy IR) and apply ad-hoc redundancy constraints. In contrast, the financial econometrics literature has developed explicit corrections for selection among many trials (Reality Check, SPA, DSR, Probability of Backtest Overfitting). Our proposal connects these statistical tools to modern agentic alpha mining by changing the selection objective rather than the generator.

### Related Papers

- **[QuantaAlpha](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/meta/meta_info.txt)**: Evolutionary trajectory-based LLM factor mining; reports strong backtest performance but does not apply formal multiple-testing correction in selection.
- **[AlphaAgent](./references/AlphaAgent-LLM-Driven-Alpha-Mining-with-Regularized-Exploration-to-Counteract-Alpha-Decay/meta/meta_info.txt)**: LLM agent framework with regularizers for novelty and complexity; discusses overfitting risks but does not integrate DSR/Reality Check/SPA into selection.
- **R&D-Agent-Quant** ([arXiv:2505.15155](https://arxiv.org/abs/2505.15155)): Multi-agent R&D workflow for factor/model co-optimization; representative baseline family for LLM-agent quantitative research.
- **[AlphaForge](./references/AlphaForge-A-Framework-to-Mine-and-Dynamically-Combine-Formulaic-Alpha-Factors/meta/meta_info.txt)**: Deep generative factor mining with dynamic weighting; focuses on generation/combination rather than data-snooping-aware selection.
- **[Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning](./references/Generating-Synergistic-Formulaic-Alpha-Collections-via-Reinforcement-Learning/meta/meta_info.txt)**: Reinforcement learning (RL) method for discovering sets of complementary factors; still selects based on backtest metrics without explicit multiple-testing correction.
- **Alpha2** ([arXiv:2406.16505](https://arxiv.org/abs/2406.16505)): Formulaic alpha discovery using Monte Carlo tree search (MCTS) to guide exploration; improves search but not selection bias correction.
- **Navigating the Alpha Jungle** ([arXiv:2505.11122](https://arxiv.org/abs/2505.11122)): LLM + MCTS factor mining with structural diversity controls; does not incorporate multiple-testing-aware scoring.
- **[AlphaEval](./references/AlphaEval-A-Comprehensive-and-Efficient-Evaluation-Framework-for-Formula-Alpha-Mining/meta/meta_info.txt)**: Backtest-free multi-dimensional evaluation for alpha factors; complementary to our goal of correcting the selection step when backtests are used.
- **AlphaSAGE** ([arXiv:2509.25055](https://arxiv.org/abs/2509.25055)): Structure-aware factor generation with generative flow networks (GFlowNets); focuses on exploration/diversity rather than selection bias control.
- **AutoAlpha** ([arXiv:2002.08245](https://arxiv.org/abs/2002.08245)): Hierarchical evolutionary alpha generation; classic search-based baseline family.
- **LLMFactor / FAMA** ([ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.233/)): Neural-symbolic LLM factor mining for interpretable operators.
- **FactorMAD** ([ACM AIFin 2025](https://dl.acm.org/doi/10.1145/3768292.3770377)): Multi-agent debate framework for interpretable factor mining.
- **TradingAgents** ([arXiv:2412.20138](https://arxiv.org/abs/2412.20138)): Multi-agent trading framework; representative finance-agent baseline outside pure factor mining.
- **FinMem** ([arXiv:2311.13743](https://arxiv.org/abs/2311.13743)): Memory-augmented trading agents; relevant for long-horizon agent design.
- **ATLAS** ([arXiv:2510.15949](https://arxiv.org/abs/2510.15949)): Dynamic prompt optimization for trading agents.
- **FINSABER** ([arXiv:2505.07078](https://arxiv.org/abs/2505.07078)): Evidence that strategy gains can diminish under stricter, data-snooping-aware evaluation.
- **[Qlib](https://github.com/microsoft/qlib)**: Widely used open-source platform for factor research and backtesting used by QuantaAlpha and related work.
- **[Bailey & López de Prado, 2014 (Deflated Sharpe Ratio)](./references/The-Deflated-Sharpe-Ratio-Correcting-for-Selection-Bias-Backtest-Overfitting-and-Non-Normality/meta/meta_info.txt)**: Defines DSR and effective-trials estimation for selection bias.
- **Bailey & López de Prado, 2012 (Probabilistic Sharpe Ratio)** ([SSRN](https://ssrn.com/abstract=1821643)): Foundation for DSR via probabilistic Sharpe ratio.
- **[White, 2000 (Reality Check)](./references/A-Reality-Check-for-Data-Snooping/meta/meta_info.txt)**: Stationary-bootstrap test controlling for data snooping across many strategies.
- **Hansen, 2005 (SPA test)** ([doi](https://doi.org/10.1198/073500105000000063)): A more powerful data-snooping test than Reality Check.
- **Politis & Romano, 1994 (Stationary bootstrap)** ([doi](https://doi.org/10.1214/aos/1176325771)): Bootstrap procedure for dependent time series.
- **López de Prado, 2018 (purged cross-validation and combinatorial purged cross-validation (CPCV))** ([book](https://www.cambridge.org/core/books/advances-in-financial-machine-learning/)): Methods for time-series backtesting that reduce leakage by purging overlapping samples; relevant to more robust factor evaluation.
- **Bailey et al., 2014 (Probability of Backtest Overfitting)** ([SSRN](https://ssrn.com/abstract=2326253)): Formalizes the probability of backtest overfitting in strategy search.
- **Harvey & Liu, 2015 (multiple testing in finance)** ([SSRN](https://ssrn.com/abstract=2514783)): Multiple comparisons issues in factor discovery.

**Reference format:**

- Use local KB path (`./references/<paper>/meta/meta_info.txt`) for papers you have scraped
- Use arXiv/URL directly for other papers

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| LLM agent factor mining | Hypothesis → formula/code → backtest loops using LLM-driven agents | QuantaAlpha; AlphaAgent; R&D-Agent-Quant | IC/RankIC and strategy backtests (IR/ARR/MDD) | Vulnerable to selection bias when ranking many candidates on validation |
| Search-based formula mining | Genetic programming (GP), evolutionary search, or RL for factor discovery | AutoAlpha; Alpha2; AlphaGen | IC/RankIC and backtests | Exploration cost; selection bias persists without correction |
| Deep generative factor mining | Learn a generator of factor formulas and dynamically weight factors | AlphaForge; AlphaSAGE | IC/RankIC and backtests | Still requires ranking/selection; multiple testing often unaddressed |
| Evaluation frameworks | Alternative evaluation objectives beyond raw backtests | AlphaEval | Stability/robustness/diversity metrics | Does not directly correct “best-of-many” backtest selection |
| Data-snooping-aware statistics | Correct for selection among many strategies/factors | DSR; Reality Check; SPA; probability of backtest overfitting (PBO) | Sharpe/IR-style metrics with bootstrap tests | Rarely integrated into ML/LLM factor mining pipelines |

### Closest Prior Work

- **[QuantaAlpha](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/meta/meta_info.txt)**: Defines a trajectory-evolution pipeline and selects factor pools using validation backtests and redundancy filtering; it does not apply formal multiple-testing corrections when choosing among many mined candidates.
- **[AlphaAgent](./references/AlphaAgent-LLM-Driven-Alpha-Mining-with-Regularized-Exploration-to-Counteract-Alpha-Decay/meta/meta_info.txt)**: Introduces regularizers to discourage overfitting and crowding, but still selects factors using uncorrected backtest metrics.
- **[Bailey & López de Prado, 2014](./references/The-Deflated-Sharpe-Ratio-Correcting-for-Selection-Bias-Backtest-Overfitting-and-Non-Normality/meta/meta_info.txt)**: Provides DSR and effective-trials estimation; does not address integration into LLM-based factor mining or pool construction with redundancy constraints.
- **[White, 2000](./references/A-Reality-Check-for-Data-Snooping/meta/meta_info.txt)** and **Hansen, 2005**: Provide data-snooping-aware significance tests; typically used as after-the-fact evaluation rather than as a ranking objective inside an automated mining loop.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| QuantaAlpha | LLM agent trajectory evolution for factor mining; ranks/filters factors using validation metrics plus redundancy controls | Validation-guided selection among many candidates can inflate “best” metrics | Replace ranking score with DSR-adjusted score (Deflated-RankICIR) | Reduces selection bias while keeping the mining pipeline unchanged |
| AlphaAgent | LLM factor mining with regularization for novelty/complexity | No explicit multiple-testing-aware selection objective | Add DSR-based ranking for pool construction | Improves statistical defensibility when selecting among many candidates |
| DSR / Reality Check / SPA | Statistical tests controlling for selection bias and data snooping | Not integrated into agentic factor mining pipelines | Apply as a ranking objective (DSR) and evaluation (SPA/RC) | Brings established finance validation to LLM search loops |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| DeepSeek-V3.2 (API) | proprietary | API (application programming interface) access | Matches a QuantaAlpha paper setting (DeepSeek-V3.2 row in the Methods section) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| QuantaAlpha/qlib_csi300 | Factor mining + backtesting for CSI300 (2016–2025 daily data) | cn_data.zip 493MB; daily_pv.h5 398MB; daily_pv_debug.h5 1.41MB | https://huggingface.co/datasets/QuantaAlpha/qlib_csi300 | Apache-2.0 (per HF dataset card) |

**Other Resources (if applicable):**

- QuantaAlpha repo: https://github.com/QuantaAlpha/QuantaAlpha
- Qlib: https://github.com/microsoft/qlib

**Resource Estimate**:

- **Compute budget**: ~0 graphics processing unit (GPU)-hours (no model training; only central processing unit (CPU) backtesting and LLM calls via an application programming interface (API)).
- **Wall-clock**:
  - One QuantaAlpha mining run is API-limited; expected to take hours rather than days at moderate search budgets. QuantaAlpha does not report exact token counts, so verification should enforce an explicit cap (for example, stop after generating a few hundred candidate factors or after a fixed wall-clock budget such as 6 hours).
  - Post-hoc scoring, pool construction, and stationary bootstrap are CPU-only and should complete within hours.
- **API usage**:
  - One QuantaAlpha mining run to produce a candidate library.
  - Verification will start from QuantaAlpha’s documented structure (parallel directions, multiple iterations, multiple factor attempts per hypothesis) and will cap the run by limiting the total number of retained candidate factors (target M in the low hundreds) and by setting a maximum wall-clock budget.

**Infrastructure constraints:**

- No browser-based user interface (UI) required; use command-line interface (CLI) scripts only.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| CSI300 (Qlib) | A factor-mining and backtesting benchmark over the CSI300 stock universe, evaluating both predictive correlations (IC/RankIC) and realized strategy performance under a fixed portfolio rule | IC, RankIC, ICIR, RankICIR; strategy IR, ARR, MDD, CR | Train/Val/Test as in QuantaAlpha (train 2016–2020; val 2021; test 2022–2025) | https://huggingface.co/datasets/QuantaAlpha/qlib_csi300 | QuantaAlpha `quantaalpha.backtest.run_backtest` + our post-hoc selector |

### Main Results

#### Results Table

Published reference numbers (QuantaAlpha paper; CSI300).

Notes: Higher is better for IC, RankIC, IR, ARR, and CR; lower is better for MDD.

| Method | LLM backbone | IC (↑) | RankIC (↑) | IR (↑) | ARR (%) (↑) | MDD (%) (↓) | Source | Notes |
|---|---|---:|---:|---:|---:|---:|---|---|
| RD-Agent | DeepSeek-V3.2 | 0.0401 | 0.0522 | 0.8202 | 7.81 | 18.03 | [QuantaAlpha Methods](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/sections/Methods.md) | Published baseline |
| AlphaAgent | DeepSeek-V3.2 | 0.0955 | 0.0919 | 1.9230 | 14.51 | 9.84 | [QuantaAlpha Methods](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/sections/Methods.md) | Published baseline |
| QuantaAlpha | DeepSeek-V3.2 | 0.1338 | 0.1300 | 2.8797 | 23.77 | 9.14 | [QuantaAlpha Methods](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/sections/Methods.md) | Published baseline |
| QuantaAlpha | GPT-5.2 | 0.1501 | 0.1465 | 3.3251 | 27.75 | 7.98 | [QuantaAlpha Methods](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/sections/Methods.md) | Published state of the art result in that paper |

Planned verification comparisons (same mined factor library; CSI300 test backtest). This table is the primary evaluation because it holds the candidate library fixed.

| Method | Candidate library | Pool score (validation) | Test IR (↑) | Test ARR (↑) | Test MDD (↓) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| A: mean RankIC | One mining run | mean RankIC | **TBD** | **TBD** | **TBD** | - | Reference (QuantaAlpha-style) |
| B: RankICIR | Same run | RankICIR | **TBD** | **TBD** | **TBD** | - | Strong simple baseline |
| C: Deflated-RankICIR (ours) | Same run | DSR(RankIC series) | **TBD** | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| C1: \(\hat{N} = \hat{\rho} + (1-\hat{\rho})M\) | Default effective-trials estimate | Best trade-off between conservatism and performance |
| C2: \(\hat{N} = M\) | Maximal multiple-testing penalty | More conservative ranking; may reduce IR but improve corrected significance |
| C3: \(\hat{N} = 1\) | No multiple-testing penalty | Should approach the uncorrected behavior and serve as a control condition |

### Analysis (Optional)

- **Winner’s-curse diagnostic**: validation score vs. test IR scatter for candidates; test whether DSR reduces over-selection of extreme validation winners.
- **Regime sensitivity**: compute yearly test IR (2022/2023/2024/2025) to assess whether DSR-based selection improves stability under regime shifts.

---

## Success Criteria

**Primary criterion (C vs B): does multiple-testing-aware ranking help beyond RankICIR?**

- Hypothesis: C achieves higher test-period IR than B under the same pool size K and redundancy threshold.
- Decision rule (accept): the stationary-bootstrap 90% confidence interval lower bound for \(\mathrm{IR}_C - \mathrm{IR}_B\) is ≥ 0.
- Decision rule (refute): the stationary-bootstrap 90% confidence interval upper bound for \(\mathrm{IR}_C - \mathrm{IR}_B\) is ≤ 0.

**Secondary criterion: statistical defensibility improves without a material performance loss**

- Hypothesis: C yields stronger multiple-testing-aware evidence (e.g., higher DSR on test returns and/or lower SPA p-values) than B.
- Decision rule: if the primary criterion is inconclusive, we accept a “statistical defensibility win” if C improves corrected significance while remaining close to B in test-period IR (i.e., not clearly worse under the bootstrap interval).

---

## Impact Statement

If successful, this work provides a simple, actionable change for practitioners using LLM-driven factor mining: rank and select factors using a multiple-testing-aware score rather than raw validation IC/RankIC-based ratios. This would make backtest-reported results from agentic alpha mining more statistically defensible and reduce costly deployment of strategies that are strong only due to selection among many trials.

---

## References

- [QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining](./references/QuantaAlpha-An-Evolutionary-Framework-for-LLM-Driven-Alpha-Mining/meta/meta_info.txt) - Han et al., 2026
- [QuantaAlpha GitHub Repository](./references/GitHub---QuantaAlpha-QuantaAlpha/meta/meta_info.txt) - QuantaAlpha Team, 2026
- [QuantaAlpha/qlib_csi300 Dataset](./references/QuantaAlpha-qlib_csi300-Datasets-at-Hugging-Face/meta/meta_info.txt) - QuantaAlpha Team, 2025
- [AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay](./references/AlphaAgent-LLM-Driven-Alpha-Mining-with-Regularized-Exploration-to-Counteract-Alpha-Decay/meta/meta_info.txt) - Tang et al., 2025
- [AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors](./references/AlphaForge-A-Framework-to-Mine-and-Dynamically-Combine-Formulaic-Alpha-Factors/meta/meta_info.txt) - Shi et al., 2024
- [Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning](./references/Generating-Synergistic-Formulaic-Alpha-Collections-via-Reinforcement-Learning/meta/meta_info.txt) - Yu et al., 2023
- [AlphaEval: A Comprehensive and Efficient Evaluation Framework for Formula Alpha Mining](./references/AlphaEval-A-Comprehensive-and-Efficient-Evaluation-Framework-for-Formula-Alpha-Mining/meta/meta_info.txt) - Ding et al., 2025
- [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality](./references/The-Deflated-Sharpe-Ratio-Correcting-for-Selection-Bias-Backtest-Overfitting-and-Non-Normality/meta/meta_info.txt) - Bailey & López de Prado, 2014
- [A Reality Check for Data Snooping](./references/A-Reality-Check-for-Data-Snooping/meta/meta_info.txt) - White, 2000
- [Hansen SPA Test](https://doi.org/10.1198/073500105000000063) - Hansen, 2005
- [Stationary Bootstrap](https://doi.org/10.1214/aos/1176325771) - Politis & Romano, 1994
- [Advances in Financial Machine Learning](https://www.cambridge.org/core/books/advances-in-financial-machine-learning/) - López de Prado, 2018
- [Probability of Backtest Overfitting](https://ssrn.com/abstract=2326253) - Bailey et al., 2014
- [Multiple Testing in Factor Investing](https://ssrn.com/abstract=2514783) - Harvey & Liu, 2015
- [R&D-Agent-Quant](https://arxiv.org/abs/2505.15155) - 2025
- [Navigating the Alpha Jungle](https://arxiv.org/abs/2505.11122) - 2025
- [AutoAlpha](https://arxiv.org/abs/2002.08245) - 2020
- [LLMFactor / FAMA](https://aclanthology.org/2024.findings-acl.233/) - 2024
- [FactorMAD](https://dl.acm.org/doi/10.1145/3768292.3770377) - 2025
- [TradingAgents](https://arxiv.org/abs/2412.20138) - 2024
- [FinMem](https://arxiv.org/abs/2311.13743) - 2023
- [ATLAS](https://arxiv.org/abs/2510.15949) - 2025
- [FINSABER](https://arxiv.org/abs/2505.07078) - 2025
- [Bailey & López de Prado, 2012 (Probabilistic Sharpe Ratio)](https://ssrn.com/abstract=1821643) - 2012
- [Alpha2](https://arxiv.org/abs/2406.16505) - Liu et al., 2024
- [AlphaSAGE](https://arxiv.org/abs/2509.25055) - Chen et al., 2025
- [Qlib](https://github.com/microsoft/qlib) - Microsoft
