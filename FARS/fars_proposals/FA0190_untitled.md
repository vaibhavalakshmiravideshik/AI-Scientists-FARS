# untitled

# Paired Median-of-Means Rewards for Noisy QPS–Recall Optimization in Vector Search

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Approximate nearest neighbor search (ANNS) is a core systems primitive behind vector databases and retrieval-augmented generation (RAG). In ANNS, practitioners often need to choose between implementations and parameter settings that trade off **throughput** (queries per second; QPS) and **quality** (recall@k; the fraction of true top-k nearest neighbors retrieved). As a result, ANNS papers and benchmarks typically report **QPS–recall curves** rather than a single scalar metric.

Recently, several works have started to use large language models (LLMs; neural language models trained to generate text/code) to automatically optimize systems code and system configurations using reinforcement learning (RL; training a policy by maximizing a scalar reward) with execution feedback. For example, **CRINN** trains an LLM to generate **HNSW (Hierarchical Navigable Small World)** graph-based ANNS code variants and uses a scalar reward defined as the **area under the QPS–recall curve (AUC)** restricted to recall in **[0.85, 0.95]** [CRINN](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt). In these “performance-reward” settings, the reward is computed from **wall-clock measurements**, which are often noisy due to warmup and cache effects, OS scheduling, CPU frequency scaling, background load, and measurement-order effects.

In systems benchmarking, it is well-known that naïve timing practices can produce biased or irreproducible conclusions. Classic work shows that experimental outcomes can flip under seemingly irrelevant changes (e.g., environment size, link order) [Producing Wrong Data](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf), and that fixed execution ordering can systematically bias conclusions (“ordering trap”) [Avoiding the Ordering Trap](./references/Avoiding-the-Ordering-Trap-in-Systems-Performance-Measurement/meta/meta_info.txt). Consequently, several benchmarks use protocols such as randomized execution order, paired measurements, long windows, and robust aggregation.

### The Problem

CRINN’s reward function requires estimating a QPS–recall AUC from repeated wall-clock benchmarking. However:

- **CRINN’s paper does not specify a robust noise-control protocol** for the QPS measurements used to compute reward [CRINN](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt).
- **CRINN’s example evaluation script explicitly reports the maximum QPS across repeated timing runs** (rather than the mean):

  ```python
  qps = nq / T
  mx_qps = max(mx_qps, qps)
  ```

  (from `examples/main.py`) [CRINN repo](https://raw.githubusercontent.com/deepreinforce-ai/CRINN/main/examples/main.py). This max-over-runs choice is an upward-biased estimator that can favor high-variance candidates (“winner’s curse”). In CRINN’s own results, some reported gains are only a few percent (e.g., +3.25% on SIFT-128 at recall 0.999, and +0.87% on GloVe-25 at recall 0.999), a regime where small measurement biases can change the selected “best” variant [CRINN Table 3](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/sections/5.2%20QPS%20with%20Fixed%20Recall.md).

Beyond evaluation, noisy rewards can also directly affect RL training dynamics. CRINN trains with **Group Relative Policy Optimization (GRPO)** and normalizes rewards within each sampled group by subtracting the group mean and dividing by the group standard deviation [CRINN RL Training](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/sections/3.4%20RL%20Training.md). If reward estimates are heavy-tailed (e.g., due to max-over-runs), this normalization can amplify outlier rewards into large “advantages”, encouraging the policy to exploit measurement noise rather than true speedups.

A naïve fix is “use the mean instead of max”, but this does not address two common failure modes in wall-clock benchmarking:

1. **Non-stationarity / drift**: measurements can drift over time (thermal effects, dynamic voltage and frequency scaling (DVFS), background activity). Paired benchmarking and randomized ordering reduce these effects by comparing two treatments under near-identical conditions [Duet Benchmarking](./references/Duet-Benchmarking-Improving-Measurement-Accuracy-in-the-Cloud-Accepted-Preprint-Version/meta/meta_info.txt).
2. **Heavy-tailed slowdowns / outliers**: transient spikes can dominate the mean; robust estimators such as median-of-means (MoM) offer provable concentration under heavy tails and are widely used in robust online learning [Bandits with Heavy Tail](http://sbubeck.com/BCL13.pdf).

A closely related performance-reward system, **CUDA-L1**, documents an elaborate robust measurement protocol for GPU kernel speed rewards: 30-minute measurement windows, bucketized variance control, and taking the **median of bucket averages** for robustness [CUDA-L1](./references/CUDA-L1-Improving-CUDA-Optimization-via-Contrastive-Reinforcement-Learning/meta/meta_info.txt). Importantly, CUDA-L1 measures a *single* speed scalar per candidate (kernel speedup), while CRINN’s reward is an AUC over a QPS–recall curve. Our proposal adapts the robust measurement idea to this curve-scalar reward by defining one sample as a full curve sweep and then applying paired robust aggregation at the reward level.

This suggests that reward design in performance-RL needs a “verifier” component: a measurement protocol that is robust enough that RL does not optimize noise.

### Key Insight and Hypothesis

**Key insight:** For performance-based selection and RL, the critical object is not a single noisy QPS number, but the **ranking of candidate variants under a fixed measurement budget**. Many practical workflows (manual tuning, automated algorithm configuration, RL-based search) repeatedly select “the best” among many near-tied candidates; in such settings, upward-biased estimators (max-over-runs) and heavy-tailed noise can systematically pick the wrong candidate.

**Hypothesis:** Under a fixed evaluation budget in a near-tie regime, a reward estimator based on **(i) paired randomized-order execution** and **(ii) median-of-means aggregation of log speedup ratios** will produce candidate rankings that more closely match a high-budget oracle than (a) CRINN-style max-over-runs QPS and (b) unpaired mean QPS.

Why we could be wrong: (1) ANNS timing noise on the verification hardware may be too small, making all estimators equivalent; (2) sequential pairing could introduce cache-state contamination larger than the variance reduction; (3) the candidate set might have wide true gaps, making ranking trivial.

---

## Proposed Approach

### Overview

We propose a **paired median-of-means (Paired-MoM) reward estimator** for ANNS QPS–recall AUC rewards. Median-of-means (MoM) is a robust mean estimator that partitions samples into blocks, averages within each block, and takes a median across blocks to reduce sensitivity to outliers. The estimator is intended as a drop-in replacement for the “repeat R times and take max/mean” measurement typically used in performance-reward loops.

The estimator treats benchmarking as an experiment with two treatments: a **candidate configuration** (e.g., a code variant or parameterization) and a fixed **reference configuration** (e.g., the current deployed baseline or the original implementation). It repeatedly runs candidate and reference back-to-back in randomized order on the *same query batch*, computes a speedup ratio, and then aggregates ratios robustly using median-of-means.

### Method Details

#### Target quantity: QPS–recall AUC in a recall band

We follow CRINN’s scalarization of the QPS–recall curve: for a candidate implementation, we sweep `efSearch` (the HNSW search-width parameter controlling how many graph nodes are explored per query) to obtain a set of points `(recall, QPS)`. We then keep only points with recall in **[0.85, 0.95]** and compute the trapezoidal **area under the curve (AUC; higher means better average throughput at fixed recall band)** [CRINN Speed Reward](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/sections/3.3%20Speed%20Reward.md).

#### Step 0: Choose a recall grid and map recall→efSearch

Because `efSearch` is discrete, we evaluate at a small recall grid in the band, e.g.
`R = {0.85, 0.875, 0.90, 0.925, 0.95}`.

For each configuration (candidate and reference), we find `efSearch(r)` by binary search using a fixed subset of queries (e.g., 1000 queries), targeting recall `r` within a tolerance (e.g., ±0.002). This mapping step is deterministic and separated from the timing measurements.

#### Step 1: Collect paired *curve-level* trials

The scalar reward is an AUC over multiple recall points, so we treat one “measurement” as an entire **curve sweep** (all recall points) rather than estimating each recall point independently.

For each curve-level trial `t = 1..n_trials`:

1. Sample a query batch (fixed size; e.g., 1,000 queries).
2. Randomize the order (candidate-first vs reference-first) to avoid systematic ordering bias [Avoiding the Ordering Trap](./references/Avoiding-the-Ordering-Trap-in-Systems-Performance-Measurement/meta/meta_info.txt).
3. For each method (candidate or reference), run the sweep over the fixed recall grid `R` (via the precomputed `efSearch(r)` mapping) to obtain a set of points `(r, qps(t, r))`, then compute the AUC for that trial: `AUC(t)`.

This yields paired curve-level samples `AUC_cand(t)` and `AUC_ref(t)` measured under nearly identical machine state.

We then compute a **log AUC speedup** per trial:

`x(t) = log(AUC_cand(t)) − log(AUC_ref(t))`.

Using the log ratio makes the estimator robust to multiplicative timing noise and corresponds to a robust geometric-mean speedup.

#### Step 2: Median-of-means aggregation (on log AUC ratios)

Partition the `n_trials` log-ratios into `B` buckets of size `m` (`n_trials = B·m`). For each bucket `b`, compute the mean `μ_b = mean_{t∈b} x(t)`. The Paired-MoM estimate is:

`Δ̂_MoM = median_b μ_b`.

(Optional, matching CUDA-L1): compute inter-bucket variance and discard the reward estimate if variance exceeds a threshold; in that case, re-measure [CUDA-L1 Extended Measurement Window](./references/CUDA-L1-Improving-CUDA-Optimization-via-Contrastive-Reinforcement-Learning/sections/Extended%20Measurement%20Window.md).

#### Step 3: Convert to an absolute reward (optional) or use relative rewards

For selection and tuning, we can compare candidates by their estimated relative improvement `exp(Δ̂_MoM)` to a shared reference.

If an absolute scalar reward is needed (e.g., to match CRINN’s AUC scale), estimate the reference AUC once using a higher-budget robust estimator (e.g., MoM over `AUC_ref(t)`), denoted `ÂUC_ref`, and return:

`ÂUC_cand = ÂUC_ref · exp(Δ̂_MoM)`.

This avoids requiring a per-recall-point oracle curve while still producing a scalar compatible with CRINN’s reward definition.

#### Built-in protocol validity checks (to avoid false wins)

- **Order-effect pilot**: compute `Δ_order = mean(x | cand-first) − mean(x | ref-first)` on the curve-level log AUC ratios; if `|Δ_order|` is large relative to the estimator gaps (threshold specified in Success Criteria), treat sequential pairing as contaminated and pivot to stronger process isolation (CPU pinning, cache flush) or drop pairing.
- **Oracle CI gate**: require the oracle reference curve to have tight confidence intervals (e.g., relative CI width <0.5%) to avoid chasing oracle noise.

### Key Innovations

1. **Curve-level pairing for QPS–recall scalar rewards**: Treat one timing sample as a full QPS–recall AUC sweep (not a single recall point), then apply paired estimation at the *reward* level. This matches how CRINN uses AUC as the RL reward and avoids mismatched recall grids across candidates.
2. **Paired MoM on log AUC speedup ratios**: Combine randomized paired benchmarking (drift reduction) with median-of-means aggregation (heavy-tail robustness) on log(AUC) ratios to stabilize candidate ranking under tight evaluation budgets.
3. **Decision-oriented evaluation**: Evaluate estimators by their ability to reproduce an oracle’s selection/ranking (top-1 accuracy, regret-to-oracle, Kendall τ), rather than only reporting variance of QPS.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) vector search benchmarking and optimization, (ii) automated systems optimization using LLMs and RL, and (iii) statistically sound performance measurement.

In vector search, the standard evaluation object is a QPS–recall curve, often produced by sweeping a search parameter such as HNSW’s `efSearch` [ANN-Benchmarks](./references/ANN-Benchmarks-A-Benchmarking-Tool-for-Approximate-Nearest-Neighbor-Algorithms/meta/meta_info.txt). Recent optimization work (including LLM-based code optimization) increasingly targets deployment-relevant metrics like QPS at high recall.

In performance measurement, it is now standard to worry about ordering effects, hidden confounders, and heavy-tailed noise. Techniques such as paired execution [Duet Benchmarking](./references/Duet-Benchmarking-Improving-Measurement-Accuracy-in-the-Cloud-Accepted-Preprint-Version/meta/meta_info.txt), randomized ordering [Avoiding the Ordering Trap](./references/Avoiding-the-Ordering-Trap-in-Systems-Performance-Measurement/meta/meta_info.txt), layout randomization [Stabilizer](https://people.cs.umass.edu/~emery/pubs/Stabilizer-UMass-CS-TR-2012-012.pdf), and robust estimators (median-of-means, trimmed mean) provide partial solutions.

Finally, in algorithm configuration and RL-based search, the “winner’s curse” under noisy evaluation is a recurring issue: selecting the best among many candidates using a noisy estimator can systematically overestimate performance [ParamILS](https://www.cs.ubc.ca/~hutter/papers/aaai07_param_ils.pdf). Our proposal adapts these insights to CRINN-style ANNS reward estimation.

### Related Papers

- **[CRINN: Contrastive Reinforcement Learning for Approximate Nearest Neighbor Search](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt)**: Uses an AUC(QPS–recall) reward in recall [0.85,0.95] for LLM-based ANNS code optimization; does not specify robust timing aggregation.
- **[CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](./references/CUDA-L1-Improving-CUDA-Optimization-via-Contrastive-Reinforcement-Learning/meta/meta_info.txt)**: Demonstrates robust timing reward design for GPU kernel optimization (bucket medians, long windows), inspiring a similar “verifier” for ANNS.
- **[ANN-Benchmarks](./references/ANN-Benchmarks-A-Benchmarking-Tool-for-Approximate-Nearest-Neighbor-Algorithms/meta/meta_info.txt)**: Widely used framework for generating QPS–recall curves for ANNS methods.
- **[Avoiding the Ordering Trap in Systems Performance Measurement](./references/Avoiding-the-Ordering-Trap-in-Systems-Performance-Measurement/meta/meta_info.txt)**: Shows that fixed benchmark ordering can bias results; advocates randomization and controls.
- **[Duet Benchmarking](./references/Duet-Benchmarking-Improving-Measurement-Accuracy-in-the-Cloud-Accepted-Preprint-Version/meta/meta_info.txt)**: Introduces paired concurrent benchmarking to reduce noise in cloud performance regression testing.
- **[Robust benchmarking in noisy environments](./references/Robust-benchmarking-in-noisy-environments/meta/meta_info.txt)**: Argues for robust timing estimators (including minimum-of-runs) and models non-i.i.d. timing noise.
- **[Stabilizing Policy Gradient Methods via Reward Profiling](./references/Stabilizing-Policy-Gradient-Methods-via-Reward-Profiling/meta/meta_info.txt)**: Proposes reward normalization/clipping for RL stability; complementary to (but not solving) measurement noise.
- **[Producing Wrong Data Without Doing Anything Obviously Wrong!](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf)**: Shows that small experimental setup changes can flip performance conclusions.
- **[STABILIZER: Enabling Statistically Rigorous Performance Evaluation](https://dl.acm.org/doi/10.1145/2451116.2451141)**: Randomizes memory layout to make timing distributions amenable to rigorous statistics.
- **[HNSW](https://arxiv.org/abs/1603.09320)**: Canonical graph-based ANNS method (hierarchical navigable small world graphs) underlying many vector search systems.
- **[FAISS](https://arxiv.org/abs/1702.08734)**: Popular library for similarity search, provides multiple ANN indexes and strong baselines.
- **[DiskANN](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node)**: High-performance graph-based ANN system; representative of deployment-scale tuning settings.
- **[ParlayANN](https://dl.acm.org/doi/10.1145/3572848.3577525)**: Deterministic parallel graph-based ANNS algorithms; used as a strong baseline in CRINN.
- **[Relative NN-Descent](https://dl.acm.org/doi/10.1145/3581783.3612533)**: Fast approximate kNN graph construction; baseline family in CRINN.
- **[Vearch](https://arxiv.org/abs/1908.08548)**: Distributed vector search system; highlights real-world need for QPS/recall benchmarking.
- **[Voyager](https://github.com/spotify/voyager)**: Practical HNSW-based ANN library used in production.
- **[ParamILS](https://www.cs.ubc.ca/~hutter/papers/aaai07_param_ils.pdf)**: Shows “over-confidence” in noisy algorithm configuration; motivates decision-oriented evaluation.
- **[SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf)**: Sequential model-based algorithm configuration; uses variance reduction and blocked comparisons.
- **[irace](https://mlopez-ibanez.github.io/irace/)**: Iterated racing for algorithm configuration; emphasizes statistical tests under noisy evaluations.
- **[Bandits with Heavy Tail](http://sbubeck.com/BCL13.pdf)**: Uses median-of-means and other robust estimators to handle heavy-tailed reward noise.
- **[The Geometric Median and Applications to Robust Mean Estimation](https://arxiv.org/abs/2307.03111)**: Modern survey of median-of-means/geometric median estimators and their guarantees.
- **[Robust multivariate mean estimation: the optimality of trimmed mean](https://arxiv.org/abs/1907.11391)**: Provides robust mean estimators with optimal deviation bounds under weak assumptions.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| ANNS benchmarking + metrics | Evaluate ANN methods via QPS–recall curves | [ANN-Benchmarks](./references/ANN-Benchmarks-A-Benchmarking-Tool-for-Approximate-Nearest-Neighbor-Algorithms/meta/meta_info.txt), [HNSW](https://arxiv.org/abs/1603.09320), [FAISS](https://arxiv.org/abs/1702.08734) | QPS, latency percentiles, recall@k | Often under-specifies noise control; results can be sensitive to setup |
| LLM/RL-based systems optimization | Use RL to optimize code/configs from execution rewards | [CRINN](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt), [CUDA-L1](./references/CUDA-L1-Improving-CUDA-Optimization-via-Contrastive-Reinforcement-Learning/meta/meta_info.txt) | Wall-clock speedup, QPS@recall | Reward hacking / optimizing measurement noise; protocol sensitivity |
| Systems performance measurement methodology | Reduce bias/variance with experimental design + statistics | [Avoiding the Ordering Trap](./references/Avoiding-the-Ordering-Trap-in-Systems-Performance-Measurement/meta/meta_info.txt), [Duet Benchmarking](./references/Duet-Benchmarking-Improving-Measurement-Accuracy-in-the-Cloud-Accepted-Preprint-Version/meta/meta_info.txt), [Producing Wrong Data](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf) | Regression testing, performance claims | Techniques are rarely integrated into ML-style reward pipelines |
| Robust estimators under heavy tails | Robust mean estimation with provable concentration | [Robust benchmarking](./references/Robust-benchmarking-in-noisy-environments/meta/meta_info.txt), [Bandits with Heavy Tail](http://sbubeck.com/BCL13.pdf), [Geometric Median](https://arxiv.org/abs/2307.03111) | Confidence intervals; regret bounds | Often assumes i.i.d.; mapping to real system drift requires care |

### Closest Prior Work

- **CRINN** [CRINN](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt): Defines the ANNS reward scalarization (AUC of QPS–recall in [0.85,0.95]) and reports strong speedups, but does not specify a robust measurement protocol and appears to use max-over-runs in code. **Difference:** We focus on measurement/verifier design and evaluate ranking fidelity under fixed budgets.
- **CUDA-L1** [CUDA-L1](./references/CUDA-L1-Improving-CUDA-Optimization-via-Contrastive-Reinforcement-Learning/meta/meta_info.txt): Shows robust measurement protocol for GPU kernels (long windows, bucket median). **Difference:** We adapt the idea to ANNS QPS–recall AUC rewards and add a decision-oriented evaluation against an oracle ranking.
- **Robust benchmarking in noisy environments** [Robust benchmarking](./references/Robust-benchmarking-in-noisy-environments/meta/meta_info.txt): Models timing noise and proposes robust estimators (e.g., minimum-of-runs) for microbenchmarking. **Difference:** We use paired randomized-order execution and median-of-means on log speedup ratios, and evaluate impact on selection among many candidates.
- **Avoiding the Ordering Trap** [Avoiding the Ordering Trap](./references/Avoiding-the-Ordering-Trap-in-Systems-Performance-Measurement/meta/meta_info.txt): Highlights ordering bias and advocates randomized ordering and resets. **Difference:** We incorporate randomized order directly into a reward estimator and quantify how much it improves selection fidelity.
- **Duet Benchmarking** [Duet Benchmarking](./references/Duet-Benchmarking-Improving-Measurement-Accuracy-in-the-Cloud-Accepted-Preprint-Version/meta/meta_info.txt): Uses paired concurrent runs to reduce variance in cloud performance regression testing. **Difference:** We use sequential paired runs (same query batch) and robust block aggregation, tailored to QPS–recall reward computation.

**Novelty Kill Search Summary:** We searched for combinations of “paired benchmarking” + “median-of-means/bucket median” + “approximate nearest neighbor / QPS–recall reward / reinforcement learning”, and checked for ANNS-specific reward verifiers inspired by CUDA-L1. No prior work combining paired randomized-order benchmarking with MoM-style aggregation for ANNS QPS–recall AUC rewards was found as of 2026-02-20 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [CRINN](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt) | RL for ANNS code; AUC(QPS–recall) reward | Measurement protocol under-specified; max-over-runs can be biased | Replace reward measurement with paired MoM estimator | Lower ranking error under fixed budget; less selection bias |
| [CUDA-L1](./references/CUDA-L1-Improving-CUDA-Optimization-via-Contrastive-Reinforcement-Learning/meta/meta_info.txt) | Robust timing reward for GPU kernels | Not ANNS; no QPS–recall scalarization | Apply robust bucket aggregation to ANNS reward | Transfers robust reward design to ANNS with oracle-based evaluation |
| [Robust benchmarking](./references/Robust-benchmarking-in-noisy-environments/meta/meta_info.txt) | Robust microbenchmarking estimators | Not decision-oriented for selecting among many candidates | Combine pairing + MoM + selection metrics | Targets “pick-best” workflows directly |
| [Duet Benchmarking](./references/Duet-Benchmarking-Improving-Measurement-Accuracy-in-the-Cloud-Accepted-Preprint-Version/meta/meta_info.txt) | Paired benchmarking for cloud regressions | Uses concurrency; no heavy-tail robust aggregation | Use sequential pairing + MoM | Cancels drift and suppresses outliers |
| [Ordering Trap](./references/Avoiding-the-Ordering-Trap-in-Systems-Performance-Measurement/meta/meta_info.txt) | Shows ordering bias; recommends randomization | No concrete estimator for reward pipelines | Integrate order randomization into estimator + gates | Prevents systematic bias in reward computation |

---

## Experiments

### Experimental Setup

**Goal:** Compare reward estimators by how well they reproduce an oracle ranking/selection of near-tied ANNS configurations under a fixed measurement budget.

**Implementation / codebase:** Use `ann-benchmarks` (QPS–recall evaluation harness) [ANN-Benchmarks](./references/ANN-Benchmarks-A-Benchmarking-Tool-for-Approximate-Nearest-Neighbor-Algorithms/meta/meta_info.txt), and benchmark a standard HNSW implementation (e.g., hnswlib or FAISS HNSW).

**Benchmarks (2 datasets):**
- **SIFT-128** (128-d image descriptors; used by CRINN for RL training)
- **GIST-960** (960-d image descriptors; higher-dimensional and typically slower)

Including two datasets targets the evaluator’s generalizability concern without bloating scope.

**Candidate configurations:**
- Start from a pool of ~80 HNSW configurations (e.g., varying `M ∈ {12,16,20,24}`, `efConstruction ∈ {80,120,160,200}`, fixed thread count).
- Compute a coarse oracle estimate of AUC for all configs.
- Select a near-tie subset of `K=32` configs whose oracle AUC is within 1–3% of the best (report the realized oracle spread; if <2% use regret as primary metric).

**Pilot gate (evidence the problem exists):** Before the full near-tie study, run a small pilot on each dataset (e.g., `K=8` configs, `n_trials=20` curve sweeps) and report:
- Coefficient of variation (CV) of per-sweep AUC under unpaired measurement.
- Disagreement rate between max-over-runs vs mean (how often they pick different top-1 across subsamples).

If CV < 2% and estimators almost never disagree, the proposal should be refuted early because noise is not a bottleneck.

**Estimators compared (3 main conditions, budget-matched):**
1. **Max-over-runs (CRINN-style):** For each candidate, run `n_trials` curve sweeps and report `max(AUC)`.
2. **Unpaired mean:** For each candidate, run `n_trials` curve sweeps and report `mean(AUC)`.
3. **Paired-MoM (ours):** For each candidate, run `n_trials/2` paired curve sweeps (candidate + reference on the same query batches with randomized order), compute log AUC ratios `x(t)=log(AUC_cand(t))−log(AUC_ref(t))`, then aggregate via MoM.

**Oracle definition (evaluation target):**
- The oracle reward for each config is computed using the same curve definition but with a 10× larger trial budget (e.g., 10× more curve sweeps), using Paired-MoM as the estimator.
- Gate: require the oracle’s bootstrap CI for each config’s AUC to have relative width <0.5%; otherwise increase oracle budget.

**Baseline Ladder (REQUIRED):**
- Naïve single-run QPS at each recall point (sanity baseline)
- Max-over-runs QPS (CRINN-style)
- Unpaired mean QPS
- **Ours**: Paired randomized-order + MoM on log speedup ratios + anchored reference curve

**Software “Base Models” (not LLMs):**

| Component | Choice | Download Link | Notes |
|---|---|---|---|
| Benchmark harness | ann-benchmarks | https://github.com/erikbern/ann-benchmarks | Standard QPS–recall evaluation |
| ANN implementation | hnswlib or FAISS-HNSW | https://github.com/nmslib/hnswlib ; https://github.com/facebookresearch/faiss | Choose one for main run |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---|---|---|
| SIFT-128 (ann-benchmarks format) | Benchmark QPS–recall | 1,000,000 base / 10,000 queries | via ann-benchmarks scripts | Research use (as used in CRINN) |
| GIST-960 (ann-benchmarks format) | Benchmark QPS–recall | 1,000,000 base / 1,000 queries | via ann-benchmarks scripts | Research use (as used in CRINN) |

No model training is required.

**Other Resources (if applicable):**
- CPU pinning / process isolation tooling (`taskset`, `numactl`) if available.

**Resource Estimate**:
- **Compute budget**: 0 GPU-hours (CPU-only benchmarking).
- **Wall-clock** (order-of-magnitude): The dominant cost is curve sweeps. For each dataset: `K=32` configs × `n_trials≈30` sweeps/config ≈ 960 sweeps for the low-budget evaluation, plus ~10× sweeps for oracle estimation (can be amortized by caching shared reference measurements). Expected to fit within <1 day on a dedicated machine per dataset.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SIFT-128 (ANNS) | 128-d image descriptors; standard ANN benchmark | QPS, recall@k; reward=AUC(QPS–recall) in recall [0.85,0.95] | test | via ann-benchmarks | https://github.com/erikbern/ann-benchmarks |
| GIST-960 (ANNS) | 960-d image descriptors; standard ANN benchmark | QPS, recall@k; reward=AUC(QPS–recall) in recall [0.85,0.95] | test | via ann-benchmarks | https://github.com/erikbern/ann-benchmarks |

**Primary estimator-quality metrics (computed vs oracle):**
- **Regret-to-oracle**: `oracle_reward(best) − oracle_reward(chosen_by_estimator)` (primary if oracle spread <2%).
- **Top-1 accuracy** (selection accuracy): fraction of trials where the estimator selects the oracle-best configuration; higher is better.
- **Kendall τ** (rank correlation between the estimator ranking and the oracle ranking; higher means more consistent ordering).
- **Stability**: probability the estimator’s chosen configuration changes under budget-matched subsampling (lower is more stable).

**Uncertainty estimation (critical because estimators share traces):**
- For each configuration, collect a high-budget trace of curve-level trials `{(AUC_cand(t), AUC_ref(t), order(t))}`.
- Simulate low-budget evaluation by repeatedly subsampling `n_trials` paired curve sweeps (budget-matched) from the trace without replacement, recomputing each estimator’s reward and the induced ranking (`R=500` replicates). Report mean ± std and bootstrap CIs of selection metrics.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Regret-to-oracle ↓ (mean±std) | Kendall τ ↑ (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Single-run | HNSW | SIFT-128 | **TBD** | **TBD** | - | Sanity baseline |
| Max-over-runs (CRINN-style) | HNSW | SIFT-128 | **TBD** | **TBD** | - | Upward-biased |
| Unpaired mean | HNSW | SIFT-128 | **TBD** | **TBD** | - | Baseline (unbiased, higher variance) |
| **Ours (Paired-MoM)** | HNSW | SIFT-128 | **TBD** | **TBD** | - | Drift-canceling + robust aggregation |
| Single-run | HNSW | GIST-960 | **TBD** | **TBD** | - | Sanity baseline |
| Max-over-runs (CRINN-style) | HNSW | GIST-960 | **TBD** | **TBD** | - | Upward-biased |
| Unpaired mean | HNSW | GIST-960 | **TBD** | **TBD** | - | Baseline (unbiased, higher variance) |
| **Ours (Paired-MoM)** | HNSW | GIST-960 | **TBD** | **TBD** | - | Drift-canceling + robust aggregation |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Paired mean (no MoM) | Replace MoM with mean of log AUC ratios | If heavy tails matter, worse stability than MoM |
| Unpaired MoM (no pairing) | MoM on log AUC without reference pairing | If drift dominates, pairing adds measurable benefit |
| Raw ratio vs log ratio | Use AUC ratio means instead of log ratios | Similar ranking if noise is mild; divergence suggests transform sensitivity |

### Experimental Rigor

**Variance & Reproducibility:**
- Use 3 random seeds controlling (i) query-batch sampling (if subsampling), (ii) order randomization, and (iii) bucket assignment for MoM.
- Report mean ± std over seeds for selection metrics.

**Validity & Controls:**
- **Confounder 1 (cache contamination from pairing):** measure an order effect (`cand-first` vs `ref-first`). If large, treat pairing as invalid and pivot to stronger isolation or refute the pairing component.
- **Confounder 2 (query variance):** always compare candidate vs reference on the same query batch within a paired trial.
- **Confounder 3 (oracle instability):** require tight oracle CI; if not met, increase oracle budget or reduce near-tie tightness.

**Sanity checks:**
- If we artificially add synthetic multiplicative noise to otherwise stable timings, Paired-MoM should degrade gracefully relative to unpaired mean/max.
- If we widen the candidate set to include clearly worse configs, all estimators should agree on the top-1.

---

## Success Criteria

**Hypothesis** (directional — what you expect):
Paired-MoM will reduce ranking errors under fixed budget, especially in near-tie regimes where true AUC differences are only a few percent. Specifically, it should yield lower regret-to-oracle and higher Kendall τ than max-over-runs and unpaired mean.

**Decision Rule** (concrete — when to stop):
- **Proceed** if Paired-MoM achieves either (a) ≥25% reduction in regret-to-oracle relative to unpaired mean, or (b) ≥10 percentage point higher top-1 accuracy than unpaired mean, with non-overlapping bootstrap CIs over `R=500` subsampling trials.
- **Pivot (simplify claim)** if unpaired mean captures ≥80% of the gain vs max-over-runs and Paired-MoM adds only marginal improvement; in this case the recommendation becomes “do not use max-over-runs” plus a quantified diagnostic of when robust pairing matters.
- **Refute** if Paired-MoM does not improve over unpaired mean (within CI) or if the pairing order-effect pilot fails: `|Δ_order|` exceeds 0.5× the observed (Paired-MoM − unpaired-mean) estimator gap, indicating pairing contamination dominates.

---

## Impact Statement

If this works, developers of vector databases and researchers training RL-based system optimizers can replace ad-hoc timing aggregation (single-run or max-over-runs) with a verifier-style protocol that is more robust under realistic noise. This should reduce the number of wasted optimization iterations caused by selecting “lucky” high-variance candidates and make reported QPS–recall improvements more reproducible.

---

## References

- [CRINN: Contrastive Reinforcement Learning for Approximate Nearest Neighbor Search](./references/CRINN-Contrastive-Reinforcement-Learning-for-Approximate-Nearest-Neighbor-Search/meta/meta_info.txt) - Li et al., 2025
- [CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](./references/CUDA-L1-Improving-CUDA-Optimization-via-Contrastive-Reinforcement-Learning/meta/meta_info.txt) - Li et al., 2025
- [ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](./references/ANN-Benchmarks-A-Benchmarking-Tool-for-Approximate-Nearest-Neighbor-Algorithms/meta/meta_info.txt) - Aumüller et al., 2018
- [Avoiding the Ordering Trap in Systems Performance Measurement](./references/Avoiding-the-Ordering-Trap-in-Systems-Performance-Measurement/meta/meta_info.txt) - Duplyakin et al., 2023
- [Duet Benchmarking: Improving Measurement Accuracy in the Cloud](./references/Duet-Benchmarking-Improving-Measurement-Accuracy-in-the-Cloud-Accepted-Preprint-Version/meta/meta_info.txt) - Bulej et al., 2020
- [Robust benchmarking in noisy environments](./references/Robust-benchmarking-in-noisy-environments/meta/meta_info.txt) - Chen & Revels, 2016
- [Stabilizing Policy Gradient Methods via Reward Profiling](./references/Stabilizing-Policy-Gradient-Methods-via-Reward-Profiling/meta/meta_info.txt) - Ahmed et al., 2025
- [Producing Wrong Data Without Doing Anything Obviously Wrong!](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf) - Mytkowicz et al., 2009
- [STABILIZER: statistically sound performance evaluation](https://dl.acm.org/doi/10.1145/2451116.2451141) - Curtsinger & Berger, 2013
- [Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin, 2018
- [FAISS: A library for efficient similarity search](https://arxiv.org/abs/1702.08734) - Johnson et al., 2017
- [Fast Accurate Billion-point Nearest Neighbor Search on a Single Node (DiskANN)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node) - Subramanya et al., 2019
- [ParlayANN: Scalable and deterministic parallel graph-based approximate nearest neighbor search algorithms](https://dl.acm.org/doi/10.1145/3572848.3577525) - Manohar et al., 2023
- [Relative NN-Descent: A fast index construction for graph-based approximate nearest neighbor search](https://dl.acm.org/doi/10.1145/3581783.3612533) - Ono & Matsui, 2023
- [Vearch: The Design and Implementation of a Real-time Visual Search System](https://arxiv.org/abs/1908.08548) - Li et al., 2019
- [Automatic Algorithm Configuration based on Local Search (ParamILS)](https://www.cs.ubc.ca/~hutter/papers/aaai07_param_ils.pdf) - Hutter et al., 2007
- [Sequential Model-Based Optimization for General Algorithm Configuration (SMAC)](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) - Hutter et al., 2011
- [Bandits with Heavy Tail](http://sbubeck.com/BCL13.pdf) - Bubeck et al., 2013
- [The Geometric Median and Applications to Robust Mean Estimation](https://arxiv.org/abs/2307.03111) - Minsker & Strawn, 2023
- [Robust multivariate mean estimation: the optimality of trimmed mean](https://arxiv.org/abs/1907.11391) - Lugosi & Mendelson, 2019
