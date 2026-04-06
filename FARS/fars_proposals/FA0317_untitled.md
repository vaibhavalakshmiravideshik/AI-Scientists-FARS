# untitled

# Online Squisher-FGGM: Eliminating Per-Task Fisher Passes for Replay-Free Continual Learning in LLMs

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly updated repeatedly after pre-training (e.g., new domains, new products, new user requirements). A core difficulty is **catastrophic forgetting**: after adapting to a new task, the model can lose previously acquired capabilities such as general knowledge and reasoning. This is especially visible in **TRACE**, a continual-learning benchmark for aligned LLMs that reports large drops in general abilities after sequential fine-tuning.

A common mitigation is **replay** (mixing past-task data during training), but replay can be infeasible when historical data cannot be stored or reused (privacy, licensing, or “checkpoint-only” deployments). This motivates **replay-free** continual learning methods that restrict how model parameters change during each new task.

**Fisher-Guided Gradient Masking (FGGM)** is a recent replay-free method for LLM continual learning. For each new task, FGGM estimates a **diagonal Fisher information matrix (FIM)** on the incoming task’s data to rank parameter importance, then applies a hard binary mask so that only a subset of parameters are updated during training on that task. On TRACE with Qwen2-1.5B, FGGM improves stability (the “General” capability aggregate) over sequential fine-tuning (SFT) and over a magnitude-based masking method (**MIGU**).

However, FGGM’s per-task Fisher estimation requires an additional gradient pass over the task data (often implemented as an offline pass with batch processing). In the FGGM TRACE setting, task training totals 40 epochs across 8 tasks, while Fisher estimation adds roughly 8 extra full-dataset backward passes (one per task), i.e., about **20% additional forward/backward compute**. For tasks trained for only 3 epochs, the per-task overhead can be ~33%. When tasks arrive frequently, this overhead can materially increase fine-tuning cost.

In parallel, **Fishers for Free?** (Li et al., 2025) shows that the squared-gradient accumulator maintained by Adam/AdamW (the optimizer’s second moment, often stored as `exp_avg_sq` or \(v_t\)) can approximate the diagonal Fisher “for free” across several applications, including Fisher-based masking (FISH Mask) and EWC in vision continual learning.

### The Problem

**Can we remove FGGM’s extra per-task Fisher pass entirely by estimating importance online from gradients that are already computed during training, while preserving FGGM’s stability–plasticity trade-off?**

A naive “online Fisher” is to average squared gradients from early training steps. But during the first steps of adapting to a new task, both parameters and gradient statistics are **non-stationary**. With FGGM’s hard quantile masks, instability near the threshold can flip many parameters between “trainable” and “frozen.” This raises a more specific question:

**Does AdamW’s exponential moving average (EMA) of squared gradients provide a more stable online importance estimate than a uniform average when gradients are non-stationary early in task training?**

### Key Insight and Hypothesis

Two properties make this plausible:

1. **Non-stationarity in early adaptation:** In an online setting where importance is estimated during the first \(W\) training steps, gradients from the first few steps are computed at parameters close to \(\theta_{t-1}\), while later steps are computed at parameters that have already adapted to task \(T_t\). If importance is meant to reflect which parameters matter *when the mask is applied* (after step \(W\)), then downweighting early gradients can reduce “stale” influence.

2. **Hard quantile thresholding amplifies estimator noise:** FGGM binarizes importance scores by a per-layer quantile. Small rank errors near the threshold can flip many mask bits. An EMA can act as a low-pass filter over noisy minibatch gradients, potentially stabilizing the induced mask.

**Hypothesis (online stability)**: A no-extra-pass variant of FGGM that derives masks from AdamW’s EMA squared-gradient accumulator during the first \(W\) training steps (“Online Squisher-FGGM”) will match offline FGGM on TRACE metrics while reducing per-task importance-estimation cost, and will outperform an online uniform-average squared-gradient baseline at the same \(W\).

This could fail if (i) the unmasked warm-up steps cause most forgetting before the mask is applied, (ii) EMA introduces bias that degrades the ranking, or (iii) simple uniform averaging is already sufficient.

---

## Proposed Approach

### Overview

We propose **Online Squisher-FGGM**, which eliminates FGGM’s separate Fisher-estimation pass by computing the mask from the squared-gradient statistics collected during the first \(W\) training steps of each task (gradients that are computed anyway for training).

For each task \(T_t\):

1. **Warm-up + importance collection (steps 1..W):** Train on \(T_t\) without a mask (mask = all ones) while accumulating an importance score per parameter from squared gradients.
2. **Mask formation at step W:** Apply FGGM’s input-dimension aggregation (IA) and per-layer quantile thresholding (masking rate \(\alpha\)) to form \(M^{(t)}\).
3. **Masked training (steps W+1..end):** Continue task training with masked gradients \(\tilde g = g \odot M^{(t)}\).

This removes the extra Fisher pass. The main question is which squared-gradient estimator yields a stable mask under early-training non-stationarity.

### Method Details

#### Baseline: FGGM-offline (published)
For task \(T_t\), FGGM computes an empirical diagonal Fisher at \(\theta_{t-1}\) using an additional pass over \(T_t\)’s data:
\[
\hat F^{(t)}_i = \frac{1}{M_t} \sum_{j=1}^{M_t} \left(\frac{\partial \log p(y_j\mid x_j,\theta_{t-1})}{\partial \theta_i}\right)^2,
\]
then forms a binary mask via IA + per-layer quantile thresholding, and applies masked updates for the entire task training.

#### Online-uniform (no-pass control)
During the first \(W\) training steps of task \(T_t\) (with parameter updates), maintain a running mean of squared gradients:
\[
 u_W = \frac{1}{W} \sum_{k=1}^{W} g_k^2,
\]
where \(g_k\) is the minibatch gradient at step \(k\). Form the IA+quantile mask from \(u_W\), then apply masked training for the remaining steps.

#### Online-Squisher (ours)
During the first \(W\) training steps of task \(T_t\) (with parameter updates), maintain AdamW’s EMA of squared gradients:
\[
 v_k = \beta_2 v_{k-1} + (1-\beta_2)\, g_k^2, \quad \hat v_W = \frac{v_W}{1-\beta_2^W}.
\]
Use \(\hat v_W\) as the importance score to form the IA+quantile mask.

We use \(\beta_2=0.999\) (AdamW default) and reset \(v\) to zero at every task boundary.

#### FGGM masking pipeline (shared)
For weight matrices \(W\in\mathbb{R}^{D_{out}\times D_{in}}\), FGGM applies **input-dimension aggregation (IA)** by summing importance across input connections per output neuron:
\[
 s_r = \sum_{c=1}^{D_{in}} s_{r,c}.
\]
It then binarizes by a per-layer quantile threshold at masking rate \(\alpha\) (FGGM uses \(\alpha=0.7\)) and applies masked gradients.

### Key Innovations

1. **No-extra-pass FGGM:** Removes the per-task Fisher pass by estimating importance online from training gradients.
2. **Mechanism-driven estimator comparison under non-stationarity:** Tests whether EMA squared gradients produce more stable masks than uniform averaging when importance is estimated during early, non-stationary task adaptation.
3. **Minimal, decisive baseline set:** Offline FGGM vs online-uniform vs online-Squisher cleanly separates “online vs offline” and “EMA vs uniform” effects.

---

## Related Work

### Field Overview

Replay-free continual learning for LLMs includes masking and regularization approaches that restrict which parameters update for each task. Magnitude-based masking (MIGU) provides a strong replay-free baseline, while FGGM introduces Fisher-based masking to produce theoretically grounded importance scores. Independently, recent work shows that optimizer statistics (AdamW’s squared-gradient accumulator) can approximate Fisher diagonals, potentially reducing the cost of Fisher-based methods.

### Related Papers

- **[TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models/meta/meta_info.txt)**: Introduces TRACE and shows severe forgetting of general abilities after sequential fine-tuning.
- **[FGGM: Fisher-Guided Gradient Masking for Continual Learning](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/meta/meta_info.txt)**: Replay-free LLM continual learning via per-task Fisher estimation and hard gradient masking; notes “offline FIM” and suggests transitioning to online FIM as future efficiency work.
- **[Unlocking Continual Learning Abilities in Language Models (MIGU)](./references/Unlocking-Continual-Learning-Abilities-in-Language-Models/meta/meta_info.txt)**: Replay-free masking based on forward-pass output magnitudes in linear layers.
- **[Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator](./references/Fishers-for-Free-Approximating-the-Fisher-Information-Matrix-by-Recycling-the-Squared-Gradient-Accumulator/meta/meta_info.txt)**: Validates Adam/AdamW squared-gradient accumulators as Fisher-diagonal proxies (including Fisher-based masking and EWC), but does not study per-task online masking under short-horizon non-stationarity.
- **[Overcoming catastrophic forgetting in neural networks (EWC)](https://arxiv.org/abs/1612.00796)**: Fisher-based regularization method for continual learning.
- **[On the Computation of the Fisher Information in Continual Learning](https://arxiv.org/abs/2502.11756)**: Shows Fisher estimator choice can change CL behavior.
- **[FISH Mask](https://arxiv.org/abs/2111.09839)**: Fisher-based masking for sparse training / adaptation.
- **[BackPACK](https://arxiv.org/abs/1912.10985)**: Efficient computation of Fisher-related statistics.
- **[AdaFisher](https://arxiv.org/abs/2405.16397)**: Optimizer that approximates Fisher structure for preconditioning.
- **[IVON](https://arxiv.org/abs/2401.08570)**: Optimizer using uncertainty/Fisher-like statistics.
- **[LAMOL](https://openreview.net/forum?id=Skgxcn4YDS)**: Generative replay for lifelong language learning.
- **[Lifelong pretraining](https://aclanthology.org/2022.naacl-main.351/)**: Continual adaptation of language models to evolving corpora.
- **[Fine-tuned language models are continual learners](https://aclanthology.org/2022.emnlp-main.410/)**: Early study of continual learning behavior in fine-tuned LMs.
- **[LoRA](https://openreview.net/forum?id=nZeVKeeFYf9)**: Parameter-efficient adaptation baseline.
- **[LLaMA Pro](https://aclanthology.org/2024.acl-long.352/)**: Model expansion for continual adaptation.
- **[SEEKR](https://arxiv.org/abs/2411.06171)**: Selective attention-guided knowledge retention for LLM continual learning.
- **[Progressive Prompts](https://arxiv.org/abs/2308.10200)**: Prompt-based continual learning for language models.
- **[RECALL and Learn](https://aclanthology.org/2020.emnlp-main.634/)**: Regularization-based continual fine-tuning to reduce forgetting.
- **[Orthogonal subspace learning for language model continual learning](https://aclanthology.org/2023.findings-emnlp.715/)**: Orthogonal subspace constraints for LM CL.
- **[LFPT5](https://arxiv.org/abs/2110.07298)**: Lifelong few-shot learning via prompt tuning.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Replay-based CL for LLMs | Mix historical samples or generate replay data | TRACE-Replay; LAMOL | TRACE OP/BWT; general suites | Requires storing or generating old data |
| Magnitude-based masking | Update only parameters associated with large activations/magnitudes | MIGU | TRACE; LM CL benchmarks | Empirical importance signal |
| Fisher-based masking | Use Fisher diagonal as importance, then mask updates | FGGM; FISH Mask | TRACE | Fisher computation overhead |
| Fisher approximations from optimizer state | Reuse optimizer statistics as Fisher-like curvature | Fishers for Free | Merging/pruning/vision CL | Not studied as online per-task masking under non-stationary adaptation |

### Closest Prior Work

1. **FGGM**: Proposes offline per-task Fisher estimation and hard masking; explicitly suggests moving from offline to online FIM for efficiency but does not implement or evaluate it. Our contribution is a concrete, low-overhead online variant and a mechanism-driven comparison of online estimators.
2. **Fishers for Free?**: Validates squared-gradient accumulators as Fisher proxies in stationary end-of-training settings (and in Fisher-based masking broadly). Our work tests whether these statistics remain effective when used online at short horizons during non-stationary continual adaptation.

**Novelty Kill Search Summary:** We searched for prior work combining “online Fisher” / “optimizer second moment” with FGGM-style hard masking in LLM continual learning (queries and results are logged in `notes.md`). No prior work explicitly evaluating online squared-gradient EMA masks as a drop-in replacement for FGGM’s per-task offline Fisher pass in TRACE-style LLM continual learning was found as of 2026-02-26.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| FGGM-offline | Extra Fisher pass at \(\theta_{t-1}\) → mask → masked training | Adds ~20% compute in FGGM TRACE schedule | Remove extra pass by estimating importance online | Makes FGGM cheaper; may track current importance better |
| Online-uniform | No extra pass; uniform mean of \(g^2\) over first \(W\) steps | Sensitive to early non-stationarity/outliers | Use EMA \(v_t\) instead | EMA stabilizes ranks used by quantile masks |
| Fishers for Free | Uses \(v_t\) as Fisher proxy in other settings | Not online per-task masking under non-stationary CL | Apply to online FGGM | Tests whether proxy remains reliable in this regime |

---

## Experiments

### Experimental Setup

**Main decisive experiment (4 conditions):**
- **FGGM-offline**: baseline FGGM with a separate full-dataset Fisher pass per task.
- **Online-uniform**: no extra pass; use running mean of \(g^2\) from first \(W\) training steps to form mask at step \(W\).
- **Online-Squisher (Ours)**: no extra pass; use AdamW EMA \(v_t\) over first \(W\) training steps to form mask at step \(W\).
- **MIGU**: magnitude-based masking baseline (no Fisher pass), included to compare against the strongest published no-overhead replay-free alternative.

All three use the same IA + quantile masking rate \(\alpha=0.7\) and the same total training schedule per task.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|------|
| Qwen2-1.5B-Instruct | 1.5B | https://huggingface.co/Qwen/Qwen2-1.5B-Instruct | Matches FGGM’s main TRACE results setting |

**Continual-learning task stream (TRACE):**
- Task order and epochs follow FGGM: C-STANCE (5), FOMC (3), MeetingBank (7), Py150 (5), ScienceQA (3), NumGLUE-cm (5), NumGLUE-ds (5), 20Minuten (7).
- Optimizer: AdamW; learning rate 1e-5 linearly decayed to 0; BF16 training (per FGGM).
- Importance warm-up fraction: \(W\) = 10% of training steps within each task (default). This matches the intuition that FGGM’s offline Fisher adds ~1 full pass per task; using ~10% of a task’s steps for online importance collection is a conservative starting point that should be long enough to stabilize squared-gradient statistics but short enough to limit unmasked updates. We include \(W=5%\) as a low-cost ablation to test whether earlier masking avoids warm-up forgetting.

**Phase-0 diagnostics (cheap, go/no-go):**
On a single task (e.g., C-STANCE), report:
1. Mask agreement between online methods and FGGM-offline at the moment the mask is formed (Jaccard after IA + quantile thresholding).
2. How much performance is lost if masking starts after \(W\) steps (compare FGGM-offline vs Online-uniform).
3. Wall-clock breakdown: time spent on importance estimation for each method.

**Resource Estimate (evidence-based where possible):**
- FGGM reports training on NVIDIA A100 80GB GPUs with BF16 but does not provide wall-clock. We budget based on additional data passes.
- In FGGM TRACE schedule, offline Fisher adds ~8 extra full-dataset backward passes beyond the 40 training epochs (~20% extra forward/backward compute).
- Online methods add **0 extra passes** (importance is collected from training gradients).
- Verification will run ≥3 seeds (`seeds=[42,123,456]`). If this exceeds budget, run 2 seeds for Phase-0 + early stopping, and add a third seed only if deltas exceed noise.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| TRACE | Continual learning benchmark for aligned LLMs across 8 tasks | TRACE-OP (overall performance across learned tasks; higher is better) | test | https://github.com/BeyonderXX/TRACE | TRACE repo + OpenCompass (as in FGGM) |
| General suite | General capability evaluation (MMLU, BBH, TyDiQA, BoolQ, PIQA, GSM8K aggregate) | General (higher is better) | standard | OpenCompass sources | OpenCompass |

### Main Results

#### Published reference numbers (FGGM paper; Qwen2-1.5B, TRACE full stream)

Numbers below are copied from FGGM Table 1 (single reported run; higher is better):

| Method | Access past-task data | General | TRACE-OP | Source | Notes |
|---|---:|---:|---:|---|---|
| ORI | – | 52.45 | 31.19 | [FGGM](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/4.2 Results and Analysis.md) | (1 run) |
| SFT | ✗ | 50.89 | 49.22 | [FGGM](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/4.2 Results and Analysis.md) | (1 run) |
| MIGU | ✗ | 55.21 | 44.08 | [FGGM](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/4.2 Results and Analysis.md) | (1 run) |
| FGGM (offline) | ✗ | 55.75 | 46.00 | [FGGM](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/sections/4.2 Results and Analysis.md) | (1 run) |

#### Results Table (to be verified)

| Method | Base Model | Benchmark | TRACE-OP (mean±std) | General (mean±std) | Extra passes per task | Notes |
|--------|------------|-----------|----------------------|--------------------|----------------------|------|
| FGGM-offline | Qwen2-1.5B-Instruct | TRACE | TBD | TBD | +1 full Fisher pass | Mask from step 1 |
| Online-uniform | Qwen2-1.5B-Instruct | TRACE | TBD | TBD | +0 | Mask starts after \(W\) steps |
| **Online-Squisher (Ours)** | Qwen2-1.5B-Instruct | TRACE | TBD | TBD | +0 | Mask starts after \(W\) steps |
| MIGU | Qwen2-1.5B-Instruct | TRACE | TBD | TBD | +0 | Published replay-free baseline (no Fisher) |

### Ablation Studies (optional, low-cost)

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Smaller warm-up | \(W\)=5% | Identify how little warm-up is needed to match FGGM-offline |

### Experimental Rigor

- **Seeds**: ≥3 seeds (`seeds=[42,123,456]`) and report mean ± std.
- **Fair comparison**: Same data order, batch size, and training schedule across conditions.
- **Confounders & controls**:
  1. **Estimator vs delayed masking**: Online-uniform isolates the effect of delayed masking, while Online-Squisher isolates EMA vs uniform.
  2. **Mask variability**: Report mask agreement statistics and how they correlate with downstream performance.
  3. **Data leakage**: TRACE and general benchmarks may overlap with pretraining corpora, but comparisons are between methods on the same base model and protocol.

---

## Success Criteria

**Hypothesis**: Online-Squisher matches FGGM-offline on TRACE-OP and General while eliminating the extra Fisher pass, and outperforms Online-uniform at the same warm-up fraction \(W\).

**Decision Rule**:
- **Continue/Proceed** if Online-Squisher matches FGGM-offline within noise (overlapping std across ≥3 seeds) on both TRACE-OP and General, and improves wall-clock training time by ≥15% relative to FGGM-offline (consistent with removing ~20% extra compute), and is not worse than Online-uniform.
- **Pivot** if Online-Squisher improves speed but degrades performance: reduce warm-up forgetting by lowering \(W\) or by applying masking earlier (e.g., start with a conservative partial mask).
- **Refute** if Online-Squisher is >2 points worse than FGGM-offline on either TRACE-OP or General, or if Online-uniform and Online-Squisher behave identically (suggesting EMA adds no value).

---

## Impact Statement

If successful, this work would make Fisher-guided hard-masking methods for LLM continual learning cheaper by removing the extra per-task Fisher pass, reducing training compute by roughly the overhead that FGGM’s offline Fisher estimation adds (≈20% in FGGM’s TRACE schedule, potentially larger for short per-task updates). This could make replay-free continual adaptation more practical in privacy-constrained deployments where replay is infeasible.

---

## References

- [FGGM: Fisher-Guided Gradient Masking for Continual Learning](./references/FGGM-Fisher-Guided-Gradient-Masking-for-Continual-Learning/meta/meta_info.txt) - Tan et al., 2025/2026
- [Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator](./references/Fishers-for-Free-Approximating-the-Fisher-Information-Matrix-by-Recycling-the-Squared-Gradient-Accumulator/meta/meta_info.txt) - Li et al., 2025
- [Unlocking Continual Learning Abilities in Language Models (MIGU)](./references/Unlocking-Continual-Learning-Abilities-in-Language-Models/meta/meta_info.txt) - Du et al., 2024
- [TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](./references/TRACE-A-Comprehensive-Benchmark-for-Continual-Learning-in-Large-Language-Models/meta/meta_info.txt) - Wang et al., 2023
- [Overcoming catastrophic forgetting in neural networks (EWC)](https://arxiv.org/abs/1612.00796) - Kirkpatrick et al., 2016/2017
