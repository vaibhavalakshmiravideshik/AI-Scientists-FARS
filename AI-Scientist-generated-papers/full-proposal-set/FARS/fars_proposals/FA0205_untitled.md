# untitled

# SkewGuard-PoLR: Dirichlet-Uncertainty Gated Multi-Cluster Expansion for Robust Prefix-Consensus Self-Consistency

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Inference-time sampling is a common way to improve large language model (LLM) reliability on multi-step reasoning tasks: instead of producing one chain-of-thought (CoT), we sample multiple reasoning traces and aggregate their final answers. A widely used approach is **self-consistency (SC)**, which samples \(N\) independent CoTs and takes a majority vote over extracted final answers, often improving accuracy but incurring \(N\times\) decoding cost.

Recent work **PoLR (Path of Least Resistance)** reduces SC cost by exploiting **prefix consistency**: many sampled CoTs share similar early prefixes, and these early structures can predict the eventual reasoning mode. PoLR samples \(N\) short prefixes (e.g., 256 tokens), clusters them, and **only expands the dominant prefix cluster** to full traces, typically cutting tokens by ~40–60% while often matching SC accuracy \([PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt)\).

However, practitioners need **robustness guarantees** for inference-time pruning methods: a method that is usually accurate but sometimes makes large errors can be difficult to deploy.

### The Problem

PoLR’s pruning rule is discrete: expand only the largest prefix cluster \(C^*\). This can be brittle when the observed dominance of \(C^*\) is weak (e.g., multiple clusters with similar sizes) or when the dominant cluster mixes correct and incorrect completions.

The PoLR paper reports an example of such a tail failure: on **QWQ32B** with **AIME25** (30 problems) at \(N=51\), PoLR loses **10 points** vs SC (SC 76.7 vs PoLR 66.7) \([PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt)\). In their instance-level analysis, the failures correspond to **low self-consistency vote margins** and ambiguous prefix clustering (multiple similarly sized clusters), suggesting that “hard + low-consensus” instances are precisely where a single-cluster expansion rule is most risky.

Existing efficiency methods such as **Adaptive Consistency (AC)** \([Aggarwal et al., 2023](https://arxiv.org/abs/2305.11860)\) and **Early-Stopping Self-Consistency (ESC)** \([ESC](./references/Escape-Sky-high-Cost-Early-stopping-Self-Consistency-for-Multi-step-Reasoning/meta/meta_info.txt)\) reduce SC cost by monitoring **answer-level agreement** and stopping early. But answer-level agreement is only observable after full traces are generated, so these methods cannot provide an **early, prefix-only** safety check for PoLR-style pruning.

### Key Insight and Hypothesis

**Key insight.** The observed prefix-cluster counts \(n_1\ge n_2\ge\dots\ge n_m\) (from \(N\) sampled prefixes) can be treated as noisy evidence about an underlying distribution over latent “reasoning modes”. When \(n_1\) only slightly exceeds \(n_2\), we have weak statistical evidence that the largest observed cluster truly represents the dominant mode; in these cases, expanding only cluster 1 is a high-variance decision.

**Proposed hypothesis.** If we quantify *uncertainty over which cluster is truly dominant* using a **Dirichlet posterior over cluster proportions**, then expanding a **credible set of clusters** (instead of only the single largest cluster) will:

1. **Recover most of PoLR’s rare accuracy losses** on hard / low-consensus instances (e.g., AIME25 tail cases), because the correct reasoning mode is often in one of the top few clusters when dominance is ambiguous.
2. **Preserve PoLR’s efficiency on easy / high-consensus instances**, because the posterior concentrates on the dominant cluster and the credible set collapses to \(\{C^*\}\).
3. **Degrade gracefully toward SC** in worst cases, since expanding more clusters increases compute but never exceeds the SC budget of \(N\) full traces.

**Why we could be wrong.** Prefix-cluster dominance uncertainty might not correlate with “PoLR pruned away the correct reasoning mode”; ambiguity could reflect harmless lexical diversity, in which case expanding extra clusters only increases cost. Also, if the dominant cluster is confidently dominant but systematically wrong (rare but possible), a count-only uncertainty estimate will not detect it.

---

## Proposed Approach

### Overview

We propose **SkewGuard-PoLR**, a drop-in modification to PoLR that replaces “expand the single largest cluster” with an **uncertainty-gated multi-cluster expansion** rule computed *only from prefix clustering results*.

Given \(N\) prefixes clustered into \(m\) clusters with counts \(n_1,\dots,n_m\), SkewGuard-PoLR estimates the posterior probability that each cluster is the *true* dominant cluster under a Dirichlet-multinomial model. It then expands the **smallest set of clusters** whose posterior mass exceeds \(1-\delta\) (e.g., 0.95). The final answer is produced by majority vote over the expanded traces, exactly as in SC/PoLR.

### Method Details

**Step 0: Match PoLR’s prefix sampling + clustering.**

- Sample \(N\) prefixes of length \(L_p\) from model \(M\) for a prompt \(x\), using the same decoding settings as PoLR (temperature=0.6, top-p=0.9, \(L_p=256\)) \([PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt)\).
- Embed prefixes with TF-IDF and cluster them (PoLR uses agglomerative clustering with cosine distance and average linkage; distance threshold ~1.0).
- Let the resulting clusters be \(C_1,\dots,C_m\) with counts \(n_j=|C_j|\) and \(\sum_j n_j=N\).

**Step 1: Dirichlet posterior over mode proportions.**

Treat cluster assignments of prefixes as i.i.d. draws from an unknown categorical distribution \(p\in\Delta^{m-1}\) over latent modes. A **Dirichlet distribution** is a standard Bayesian prior over categorical probabilities; given observed counts, it yields a closed-form posterior (Dirichlet–multinomial conjugacy).

- Prior: \(p \sim \mathrm{Dirichlet}(\alpha_0\mathbf{1})\) (default \(\alpha_0=1\)).
- Posterior: \(p\mid n \sim \mathrm{Dirichlet}(\alpha_0\mathbf{1}+n)\).

**Step 2: Probability that each cluster is the true dominant mode.**

Define
\[
\pi_j \triangleq \Pr\big[j = \arg\max_k p_k \mid n\big].
\]
We estimate \(\pi_j\) with Monte Carlo:

- Sample \(p^{(s)} \sim \mathrm{Dirichlet}(\alpha_0\mathbf{1}+n)\) for \(s=1,\dots,S\) (e.g., \(S=2000\)).
- \(\pi_j\leftarrow \frac{1}{S}\sum_{s=1}^S \mathbb{1}\big[j=\arg\max_k p_k^{(s)}\big]\).

**Step 3: Credible set cluster expansion.**

Let clusters be ordered by \(\pi_j\) (not necessarily by \(n_j\) when ties occur). Choose the smallest set \(\mathcal{S}\subseteq\{1,\dots,m\}\) such that
\[
\sum_{j\in \mathcal{S}} \pi_j \ge 1-\delta, \quad \text{with } \delta\in(0,1) \text{ (default } \delta=0.05\text{)}.
\]
Then expand **all prefixes** in clusters \(\{C_j: j\in\mathcal{S}\}\) to full reasoning traces and take a majority vote over extracted answers.

**Behavior in extremes.**

- If \(n_1\gg n_2\), then \(\pi_1\approx 1\) and \(\mathcal{S}=\{1\}\), reducing to vanilla PoLR.
- If \(n_1\approx n_2\approx\dots\), posterior mass is spread and \(\mathcal{S}\) grows, approaching SC-level cost (up to \(N\) expansions).

**Ablation baseline (simple heuristic).**

To test whether the Bayesian posterior is necessary, we include a cheap heuristic variant:

- **Top-2 fallback**: if \(n_1/N < \tau\) (e.g., \(\tau=0.5\)), expand clusters 1 and 2; else expand only cluster 1.

### Key Innovations

1. **Prefix-only risk control for PoLR**: unlike answer-level adaptive stopping (AC/ESC/ConSol/CGES), SkewGuard-PoLR makes its safety decision *before* paying for full traces.
2. **A single interpretable conservativeness knob (\(\delta\))**: smaller \(\delta\) increases compute but reduces the probability of pruning away the true dominant mode under the generative model.
3. **Graceful fallback to SC without extra infrastructure**: worst-case compute is bounded by \(N\) full traces, matching SC.

---

## Related Work

### Field Overview

Inference-time scaling for LLM reasoning spans (i) sampling-based aggregation such as best-of-\(N\) and self-consistency, (ii) adaptive stopping rules that reduce the number of samples on “easy” instances, and (iii) methods that leverage intermediate signals (prefixes, hidden states, or partial traces) to allocate compute earlier. PoLR is a representative of (iii) and is particularly attractive because it is training-free and parallelizable, but its hard cluster-selection rule creates a brittleness risk when prefix consensus is weak.

Our proposal fits into a recent trend of **risk-aware test-time compute allocation**: rather than unconditionally pruning, use a measurable uncertainty signal to decide how much compute to allocate per instance. The distinguishing feature here is that the uncertainty signal is **prefix-only and structural** (derived from clustering outcomes), rather than answer-level agreement or per-token likelihood proxies.

### Related Papers

- **[PoLR: The Path of Least Resistance](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt)**: Introduces prefix clustering for SC and expands only the dominant cluster to save tokens.
- **[Self-Consistency](./references/Self-Consistency-Improves-Chain-of-Thought-Reasoning-in-Language-Models/meta/meta_info.txt)**: Establishes majority-vote over multiple CoT samples as a strong inference-time ensemble for reasoning.
- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)**: Shows that prompting models to produce intermediate reasoning steps improves performance on reasoning tasks.
- **[Adaptive Consistency / “Let’s Sample Step by Step”](https://arxiv.org/abs/2305.11860)**: Uses sequential sampling with a confidence threshold on answer counts (Beta/Dirichlet) to stop early.
- **[ESC](./references/Escape-Sky-high-Cost-Early-stopping-Self-Consistency-for-Multi-step-Reasoning/meta/meta_info.txt)**: Early-stops SC based on answer stability in a sliding window.
- **[ConSol](./references/ConSol-Sequential-Probability-Ratio-Testing-to-Find-Consistent-LLM-Reasoning-Paths-Efficiently/meta/meta_info.txt)**: Uses sequential probability ratio tests to stop SC with explicit Type-I error control.
- **[CGES](./references/CGES-Confidence-Guided-Early-Stopping-for-Efficient-and-Accurate-Self-Consistency/meta/meta_info.txt)**: Proposes confidence-guided early stopping and allocation for SC.
- **[BEACON](./references/BEACON-Bayesian-Optimal-Stopping-for-Efficient-LLM-Sampling/meta/meta_info.txt)**: Frames adaptive sampling as Bayesian optimal stopping to trade off cost and accuracy.
- **[ReASC](./references/Reliability-Aware-Adaptive-Self-Consistency-for-Efficient-Sampling-in-LLM-Reasoning/meta/meta_info.txt)**: Uses reliability estimation and staged evidence accumulation to reduce SC samples.
- **[Ranked Voting Self-Consistency](https://arxiv.org/abs/2505.10772)**: Replaces majority vote with rank aggregation (e.g., Borda count) to improve SC robustness.
- **[Slim-SC](./references/Slim-SC-Thought-Pruning-for-Efficient-Scaling-with-Self-Consistency/meta/meta_info.txt)**: Prunes redundant reasoning paths during SC based on thought similarity.
- **[STEP](./references/Hidden-States-as-Early-Signals-Step-level-Trace-Evaluation-and-Pruning-for-Efficient-Test-Time-Scaling/meta/meta_info.txt)**: Uses hidden-state signals for step-level trace evaluation and pruning under test-time scaling.
- **[Path-Consistency](./references/Path-Consistency-Prefix-Enhancement-for-Efficient-Inference-in-LLM/meta/meta_info.txt)**: Uses partial-path confidence to guide subsequent generation, leveraging prefix information differently than PoLR.
- **[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)**: Studies compute-optimal allocation of test-time sampling/refinement as a function of instance difficulty.
- **[Adaptive Inference-Time Compute](https://arxiv.org/abs/2410.02725)**: Predicts mid-generation whether continuing generation will improve the answer, enabling adaptive compute allocation.
- **[Adaptive Test-Time Compute Allocation via Training-Free Difficulty Proxies](https://openreview.net/pdf?id=ztGHhyicWs)**: Uses training-free likelihood/entropy proxies to allocate test-time compute.
- **[When to Ensemble](https://arxiv.org/abs/2510.15346)**: Identifies token-level points to ensemble model continuations for stability and speed.
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)**: Explores structured search over multiple reasoning branches at inference time.
- **[Self-Refine](https://arxiv.org/abs/2303.17651)**: Uses iterative self-feedback loops to improve outputs, increasing test-time compute.
- **[Reasoning-Aware Self-Consistency (RASC)](https://arxiv.org/abs/2408.17017)**: Uses lightweight reasoning-quality features and weighted voting to reduce SC samples.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Answer-level adaptive stopping | Generate full traces sequentially; stop when answer agreement is sufficient | AC (Aggarwal et al., 2023), ESC, ConSol, CGES, BEACON, ReASC | GSM8K EM, MATH Pass@1, GPQA accuracy | Still pays full-trace cost for each sampled path until stopping; less parallelizable |
| Prefix-based pruning / guidance | Use early prefixes/partial traces to allocate compute earlier | PoLR, Path-Consistency | AIME/MATH/GSM8K/GPQA | Risky when prefix signal is weak; many methods use heuristics |
| Intra-SC pruning / trace scoring | Score or prune sampled traces during/after generation | Slim-SC, STEP, RASC | Similar reasoning benchmarks | Often needs extra scoring model/features; may be less “prefix-only” |
| General test-time compute allocation | Use difficulty/uncertainty proxies to decide compute | Snell et al. (2024), training-free difficulty proxies | Diverse; often focuses on compute–accuracy curves | Proxies are often likelihood-based and may not capture structural disagreement |

### Closest Prior Work

1. **PoLR** \([PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt)\): Clusters prefixes and expands only the single largest cluster. It acknowledges that some datasets (and some instances) have weaker prefix predictiveness and suggests “expanding top-m clusters” as a potential future direction, but does not propose an instance-adaptive reliability rule.
2. **Answer-level adaptive stopping (AC/ESC/ConSol/CGES/BEACON/ReASC)** \([Aggarwal et al., 2023](https://arxiv.org/abs/2305.11860); [ESC](./references/Escape-Sky-high-Cost-Early-stopping-Self-Consistency-for-Multi-step-Reasoning/meta/meta_info.txt); [ConSol](./references/ConSol-Sequential-Probability-Ratio-Testing-to-Find-Consistent-LLM-Reasoning-Paths-Efficiently/meta/meta_info.txt)\): These methods control compute using agreement over *final answers*, so they cannot prevent PoLR’s specific failure mode (pruning away the correct reasoning mode before decoding full answers).
3. **Path-Consistency** \([Path-Consistency](./references/Path-Consistency-Prefix-Enhancement-for-Efficient-Inference-in-LLM/meta/meta_info.txt)\): Uses prefix information to guide decoding toward promising paths, but it is not designed as a risk-controlled drop-in replacement for PoLR’s pruning rule.

**Novelty Kill Search Summary:** Searched (web + local KB) for combinations of “PoLR + expand multiple clusters”, “prefix consensus + uncertainty gating”, “Dirichlet posterior + dominant cluster selection”, and checked for 2025–2026 work on “prefix clustering self-consistency pruning”. No prior work using a **prefix-only Dirichlet credible set over cluster dominance** as an instance-level safety rule for PoLR-style pruning was found as of **2026-02-21** (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| PoLR | Expand only the largest prefix cluster | Can drop accuracy when dominance is ambiguous | Replace single-cluster expansion with posterior-mass credible set | Expands extra clusters only when dominance evidence is weak |
| AC / ESC | Stop SC early based on answer agreement | Must generate full traces before deciding | Use prefix-only structure to decide expansions | Earlier decision can save compute before paying full-trace cost |
| ConSol / BEACON | Statistical / Bayesian stopping for SC | Answer-level; sequential and less parallel | Bayesian uncertainty applied to prefix cluster dominance | Retains PoLR parallelism; targets PoLR’s failure mode |
| Path-Consistency | Uses prefix signals to guide decoding | Not a risk-controlled pruning rule; may require confidence estimation | Use only cluster-count uncertainty; no extra models | Minimal overhead and easy integration with PoLR |
| Slim-SC / STEP | Prune traces using similarity or hidden-state signals | May require richer features / additional scoring | Count-based uncertainty with Dirichlet posterior | Simpler, prefix-only, and cheaper to implement |

---

## Experiments

### Experimental Setup

**Core evaluation principle:** compare methods under the same base model, same prompts, and same sampling budget \(N\). For PoLR-family methods, \(N\) is the number of sampled prefixes; the number of expanded full traces (PExp) is method-dependent but always \(\le N\). We report both (i) accuracy and (ii) compute proxies (PExp and token efficiency).

**Baseline Ladder (REQUIRED):**

- **Prompting baseline**: CoT with a single sample (\(N=1\)); prompt “solve step by step; output final answer in a standard format”.
- **Inference-time scaling baseline**: Self-Consistency (SC) with \(N\in\{31,51\}\) full traces and majority vote.
- **Closest existing method**: PoLR with \(N\in\{31,51\}\), \(L_p=256\), expanding only the largest prefix cluster.
- **Ours**: SkewGuard-PoLR with Dirichlet credible-set expansion (\(\alpha_0=1\), \(\delta=0.05\), \(S=2000\) posterior samples).

**Optional additional baseline (if time permits):** ESC (answer-level early stopping) to contextualize whether prefix-only gating is competitive with answer-level adaptive stopping.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| QwQ-32B | 32B | https://huggingface.co/Qwen/QwQ-32B | Model used in PoLR’s reported AIME25 failure case |
| DeepSeek-R1-Distill-Qwen-7B | 7B | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | Cheaper model for sanity checks and variance estimation |

**Training Data (if applicable):**

No training data needed — inference only.

**Resource Estimate**:

- **Primary runs (recommended)**: QwQ-32B on AIME25 (30) + GPQA-Diamond (198), with 3 seeds, \(N=51\) for SC/PoLR/Ours.
  - Worst-case expansions are bounded by SC (\(N\) full traces). SkewGuard-PoLR never exceeds SC’s expansions.
  - **Throughput-based bound**: public vLLM benchmarking reports **~615 tokens/s** total throughput for QwQ-32B on a single A100-80GB at moderate concurrency (50 requests). Using a conservative **300 tokens/s** effective throughput for our long-form reasoning setting, **1e7 generated tokens ≈ 9.3 GPU-hours** (single A100). Even at **5× overhead** (longer sequences + batching inefficiency), the total remains well below the 768 GPU-hour budget.
- **Peak VRAM**: QwQ-32B weights are ~62GB in FP16/BF16; should fit on A100 80GB, with tensor parallelism if needed.
- **Budget fit**: Designed to stay under 768 GPU-hours; if runtime is higher than expected, downscale by using only AIME25 for the decisive test and treating GPQA-Diamond as follow-up.

### Benchmarks and Metrics

**SOTA / leaderboard context (for orientation; not a baseline):** AIME25 and GPQA-Diamond are highly competitive public benchmarks with active leaderboards (e.g., Artificial Analysis: https://artificialanalysis.ai/evaluations/aime-2025 and https://artificialanalysis.ai/evaluations/gpqa-diamond). Our goal is not to beat the absolute leaderboard (often dominated by proprietary models and heavy test-time compute), but to compare *inference policies* (SC vs PoLR vs SkewGuard-PoLR) under a fixed open-weight base model and matched sampling budgets.

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| AIME 2025 | 30 olympiad-style math problems with numeric answers (0–999); evaluates hard mathematical reasoning | Pass@1 (exact match of final numeric answer) | test | https://huggingface.co/datasets/math-ai/aime25 (or equivalent mirrors) | Use a standard LLM eval harness (e.g., lighteval or lm-eval-harness) with deterministic answer extraction |
| GPQA-Diamond | 198 expert-written graduate-level STEM multiple-choice questions; evaluates scientific reasoning | Accuracy | test | https://huggingface.co/datasets/Idavidrein/gpqa (or equivalent) | Standard multiple-choice evaluation (parse final option) |

### Main Results

**Published reference point from PoLR (QWQ32B, N=51):**

- AIME25: SC 76.7 vs PoLR 66.7 (−10.0), PoLR token efficiency 56.8% \([PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt)\).
- GPQA-Diamond: SC 68.7 vs PoLR 70.2 (+1.5), token efficiency 53.8% \([PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt)\).

#### Results Table

*Definitions:* **PExp** = number of full reasoning traces expanded (\(\le N\)); **η** = token efficiency vs SC as reported by PoLR (higher means fewer tokens than SC).

| Method | Base Model | Benchmark | Metric 1 (mean±std) | Metric 2 (mean±std) | Source | Notes |
|--------|------------|-----------|----------------------|----------------------|--------|-------|
| CoT (N=1) | QwQ-32B | AIME25 | **TBD** | tokens/problem **TBD** | - | Needs re-run in our harness |
| SC (N=51) | QwQ-32B | AIME25 | 76.7 (1 run) | PExp=51 (fixed) | [PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt) | Published baseline; should be reproduced |
| PoLR (N=51, Lp=256) | QwQ-32B | AIME25 | 66.7 (1 run) | η=56.8% (vs SC) | [PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt) | Published tail failure case |
| SkewGuard-PoLR (ours) | QwQ-32B | AIME25 | **TBD** | **TBD** (PExp, η) | - | To be verified |
| Top-2 fallback (ablation) | QwQ-32B | AIME25 | **TBD** | **TBD** | - | Tests if Bayesian posterior is needed |
| CoT (N=1) | QwQ-32B | GPQA-Diamond | **TBD** | - | - | Needs re-run |
| SC (N=51) | QwQ-32B | GPQA-Diamond | 68.7 (1 run) | PExp=51 (fixed) | [PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt) | Published baseline |
| PoLR (N=51, Lp=256) | QwQ-32B | GPQA-Diamond | 70.2 (1 run) | η=53.8% (vs SC) | [PoLR](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt) | Published baseline |
| SkewGuard-PoLR (ours) | QwQ-32B | GPQA-Diamond | **TBD** | **TBD** (PExp, η) | - | To be verified |
| Top-2 fallback (ablation) | QwQ-32B | GPQA-Diamond | **TBD** | **TBD** | - | - |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | Dirichlet posterior + credible set with \(\delta=0.05\) | Best accuracy–efficiency trade-off on low-consensus instances |
| Top-2 fallback | Heuristic \(n_1/N<\tau\Rightarrow\) expand top-2 clusters | If close to ours, suggests simple rule suffices |
| Larger \(\delta\) (optional) | \(\delta=0.1\) (less conservative) | Less compute, potentially less robustness |

### Experimental Rigor

**Variance & Reproducibility:**

- Run SC / PoLR / SkewGuard-PoLR for **3 random seeds** (e.g., `seeds=[42,123,456]`) and report mean ± std.
- Use identical decoding hyperparameters across methods: temperature=0.6, top-p=0.9, max_new_tokens consistent with the benchmark (PoLR reports max_new_tokens=32K).

**Validity & Controls:**

- **Compute mismatch confound**: since SkewGuard-PoLR may expand more traces than PoLR, report accuracy–compute trade-offs and keep \(N\) fixed across methods.
- **Prompt sensitivity confound**: use a fixed prompt template across methods; optionally run a small prompt sweep on CoT/SC and reuse the best prompt for all methods.
- **Data leakage / contamination**: AIME25 problems may appear in some model training corpora. We treat the experiment as an *inference-policy* comparison under matched models and prompts; nonetheless, we will (i) report results on GPQA-Diamond (less likely to be leaked verbatim) as a complementary check, and (ii) ensure all methods share identical prompts and decoding to avoid confounding by prompt engineering.
- **Reproducibility sanity check**: reproduce PoLR’s published SC vs PoLR numbers on at least one benchmark/model setting before interpreting improvements.

### Analysis (Optional)

- **Slice by SC vote margin**: define “low-consensus” instances using the SC baseline’s vote margin (majority fraction) and report how much SkewGuard-PoLR improves accuracy on this slice relative to PoLR.
- **Mechanism check**: correlate \(\pi_1\) (posterior mass on the dominant cluster) with SC vote margin and with PoLR error events to test whether the uncertainty proxy aligns with difficulty.

---

## Success Criteria

**Hypothesis** (directional): SkewGuard-PoLR will substantially reduce PoLR’s worst-case accuracy drops on low-consensus instances (especially AIME25 tail cases) while retaining most of PoLR’s token savings on high-consensus instances.

**Decision Rule** (concrete):

- **Proceed** if, on **QwQ-32B, AIME25, N=51**, SkewGuard-PoLR improves over PoLR by **≥6.7 points** (≥2/30 problems) and is within **≤3.3 points** (≤1/30 problem) of SC, while using **<51** average expansions (i.e., strictly less compute than SC) across 3 seeds; and it does **not** reduce GPQA-Diamond accuracy relative to PoLR by more than the std range.
- **Pivot** if accuracy improves but average expansions approach SC (e.g., >0.9·51) on both benchmarks; in this case try a smaller credible threshold (larger \(\delta\)) or a budget-capped variant (expand until a fixed PExp budget).
- **Refute** if SkewGuard-PoLR does not outperform PoLR on AIME25 beyond noise (≤1 solved problem difference) or if the improvement disappears when controlling for prompt/seed.

---

## Impact Statement

If successful, SkewGuard-PoLR would make prefix-consensus pruning deployable as a safer default for inference-time scaling: it provides a principled, prefix-only uncertainty signal that decides when PoLR should behave like an aggressive pruner and when it should fall back toward SC. This could reduce inference cost for reasoning-heavy applications while avoiding rare but large accuracy regressions that would otherwise prevent adoption.

---

## References

- [The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus](./references/THE-PATH-OF-LEAST-RESISTANCE-GUIDING-LLM-REASONING-TRAJECTORIES-WITH-PREFIX-CONSENSUS/meta/meta_info.txt) - 2026
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](./references/Self-Consistency-Improves-Chain-of-Thought-Reasoning-in-Language-Models/meta/meta_info.txt) - 2023
- [Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning](./references/Escape-Sky-high-Cost-Early-stopping-Self-Consistency-for-Multi-step-Reasoning/meta/meta_info.txt) - 2024
- [ConSol: Sequential Probability Ratio Testing to Find Consistent LLM Reasoning Paths Efficiently](./references/ConSol-Sequential-Probability-Ratio-Testing-to-Find-Consistent-LLM-Reasoning-Paths-Efficiently/meta/meta_info.txt)
- [CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency](./references/CGES-Confidence-Guided-Early-Stopping-for-Efficient-and-Accurate-Self-Consistency/meta/meta_info.txt)
- [BEACON: Bayesian Optimal Stopping for Efficient LLM Sampling](./references/BEACON-Bayesian-Optimal-Stopping-for-Efficient-LLM-Sampling/meta/meta_info.txt)
- [Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning](./references/Reliability-Aware-Adaptive-Self-Consistency-for-Efficient-Sampling-in-LLM-Reasoning/meta/meta_info.txt)
- [Path-Consistency: Prefix Enhancement for Efficient Inference in LLM](./references/Path-Consistency-Prefix-Enhancement-for-Efficient-Inference-in-LLM/meta/meta_info.txt)
- [Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency](./references/Slim-SC-Thought-Pruning-for-Efficient-Scaling-with-Self-Consistency/meta/meta_info.txt)
- [Hidden States as Early Signals: Step-level Trace Evaluation and Pruning for Efficient Test-Time Scaling](./references/Hidden-States-as-Early-Signals-Step-level-Trace-Evaluation-and-Pruning-for-Efficient-Test-Time-Scaling/meta/meta_info.txt)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Let’s Sample Step by Step: Adaptive Consistency for Efficient Reasoning and Coding with LLMs](https://arxiv.org/abs/2305.11860)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [Ranked Voting based Self-Consistency of Large Language Models](https://arxiv.org/abs/2505.10772)
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)
- [Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation](https://arxiv.org/abs/2410.02725)
- [Adaptive Test-Time Compute Allocation via Training-Free Difficulty Proxies](https://openreview.net/pdf?id=ztGHhyicWs)
- [WHEN TO ENSEMBLE: IDENTIFYING TOKEN-LEVEL POINTS FOR STABLE AND FAST LLM ENSEMBLING](https://arxiv.org/abs/2510.15346)
- [Reasoning-Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling](https://arxiv.org/abs/2408.17017)
