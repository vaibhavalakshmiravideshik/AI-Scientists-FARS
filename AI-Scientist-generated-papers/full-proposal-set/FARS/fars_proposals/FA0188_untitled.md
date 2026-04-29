# untitled

# Differentially Private Spectral Monitor Logs: Can We Privatize EigenScore-Style Logging Without Losing Hallucination-Detection AUROC?

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Fully automated**: Yes (public QA datasets + automatic correctness labels + synthetic canary leakage tests)
- **Core compute budget**: ≤768 A100 GPU-hours (target ≤100)

## Introduction

### Context and Motivation

Large language models (LLMs) are deployed in settings where factual errors matter (customer support, search assistants, programming help). A common mitigation is to attach a **reliability monitor** that estimates whether a model’s answer is likely correct (hallucination detection).

Recent work shows that monitors can be stronger when they use **internal activations** (hidden states) instead of only text or logits. In particular, **INSIDE** proposes **EigenScore**, which (i) samples multiple stochastic answers to the same question, (ii) extracts one hidden-state embedding per answer, (iii) forms a small **K×K Gram/covariance matrix** over those embeddings, and (iv) reduces it to a **scalar log-determinant** (equivalently, a function of the covariance eigenvalues). INSIDE reports strong hallucination detection AUROC on QA datasets.

This line of work is increasingly framed as **deployment-oriented monitoring** rather than only an offline evaluation tool. For example, **EigenTrack** (https://arxiv.org/abs/2509.15735) proposes real-time spectral monitoring of hidden activations for hallucination and out-of-distribution detection, and **LLM-Check** benchmarks activation-derived monitoring scores with an emphasis on low overhead. These papers motivate the operational scenario where monitor scores (and sometimes richer monitor features) are logged for threshold tuning, drift monitoring, and incident response.

In parallel, privacy/security work shows that representations can leak sensitive information: embeddings and internal states can be inverted to recover prompts or infer sensitive attributes. Even if a system does not store raw activations, it may store **derived monitor artifacts** (e.g., EigenScore or an eigenvalue spectrum), assuming they are non-identifying. Observability frameworks for LLM applications (e.g., **LumiMAS**, https://arxiv.org/abs/2508.12412) further suggest that richer telemetry around LLM calls is a growing practice.

### The Problem

We study the privacy risk of **activation-derived monitor logs**:

- **Utility requirement**: The monitor should preserve hallucination-detection performance (AUROC).
- **Privacy requirement**: The released logging interface should not enable an attacker to infer prompt-specific secrets.

A practical question for teams deploying internal-state monitors is:

> If we must log EigenScore-style artifacts for auditing and debugging, is there a privacy mechanism that makes those logs non-identifying without destroying monitor utility?

### Key Insight and Hypothesis

EigenScore-style logging is unusual from a privacy perspective: the logged object is not a high-dimensional embedding, but a **small matrix (K×K, often K=10) and/or a scalar**. This raises the possibility that **matrix-valued differential privacy (DP) mechanisms** can add enough noise to suppress identification while still preserving the ranking signal needed for AUROC.

We focus on the **Wishart mechanism** for publishing covariance matrices with **(ε,0)-DP** while preserving positive semidefiniteness (PSD). Intuitively, Wishart noise is a random PSD matrix (a sum of Gaussian outer products), so adding it keeps the covariance well-formed. We compare it to a standard **Gaussian mechanism** baseline (symmetric Gaussian noise + PSD projection).

**Hypothesis (directional):** On a realistic canary attribute-inference task, adding DP noise at the **K×K Gram/covariance level** will substantially reduce canary-ID predictability from spectral monitor logs, while keeping EigenScore AUROC within a small absolute margin (≤0.03).

**Why we could be wrong:** (i) even small DP noise may destroy EigenScore’s ranking signal; (ii) the canary could influence the spectrum strongly enough that it survives DP perturbation; (iii) DP guarantees are defined under a specific adjacency notion, which may not align perfectly with prompt-level privacy. We therefore treat the DP mechanism as a principled, calibrated noise source and evaluate privacy empirically.

---

## Proposed Approach

### Overview

We evaluate privacy and utility for EigenScore-style monitor logging under three main conditions (single base model, single dataset, fixed K and decoding):

1. **No-noise logging (baseline)**: compute EigenScore from the clean K×K covariance.
2. **Gaussian mechanism (baseline DP noise)**: add symmetric Gaussian noise to the covariance, then PSD-project.
3. **Wishart mechanism (ours)**: add PSD Wishart noise to the covariance (pure DP), then compute EigenScore.

For each condition we consider two realistic **release interfaces** (reported as two columns, not separate conditions):
- **Scalar log**: release only the scalar EigenScore.
- **Spectrum log**: release the sorted eigenvalue vector (K numbers).

We evaluate:
- **Utility**: AUROC of EigenScore for hallucination detection.
- **Privacy**: supervised **canary-ID attribute inference** from released logs.

We include a **Phase-0 premise check**: if the no-noise logs are already non-identifying under our attacker (canary accuracy indistinguishable from chance), then the “need DP for these logs” hypothesis is refuted and the study reports a negative result.

**Coordination / dependency note (important):** a closely related proposal from agent_01, `spectral-invariant-activation-logging`, is an audit of whether EigenScore/eigen-spectrum logs leak canaries *without* DP. If that audit already establishes “no leakage” under a similar threat model, the verifier should treat the **main DP contribution of this proposal as moot** and either (i) refute this proposal or (ii) run only our additional controls (clipping-only, stronger attacker) as a confirmatory check.

### Method Details

#### EigenScore computation (INSIDE-style)
Given a QA prompt x:
1. Sample **K stochastic answers** (default K=10) with fixed decoding (temperature=0.5, top-p=0.99, top-k=5, max_new_tokens=64; chosen to match INSIDE defaults / induce moderate diversity across samples, not tuned).
2. For each answer k, extract a sentence embedding z_k from the base LLM: **the last-token hidden state at the middle transformer layer** (layer index ⌊L/2⌋), following INSIDE.
3. Form a centered embedding matrix Z_c ∈ R^{K×d} by subtracting the mean across k.
4. Form a small covariance/Gram matrix:
   - Σ = (1/d) · Z_c Z_c^T ∈ R^{K×K}
5. Compute EigenScore:
   - E = (1/K) · log det(Σ + α I_K) with α=0.001, equivalently E = (1/K) Σ_i log λ_i where {λ_i} are eigenvalues.

#### Bounding / clipping (for DP calibration)
To make matrix sensitivity finite, we apply **L2 clipping** to each sentence embedding z_k before forming Σ:
- z_k ← z_k · min(1, B / ||z_k||_2)

**Pre-registered rule (fixed, not tuned):** choose B as the **95th percentile** of ||z_k||₂ on a small calibration split (e.g., 2,000 prompts). Report the clipping rate on the train/test splits.

#### Privacy mechanisms

We treat the DP mechanisms as **calibrated noise sources** for the covariance query. We report (ε,δ) settings, but we do not claim that these guarantees directly imply “prompt-level DP” (our privacy evaluation is empirical canary-ID inference).

**Gaussian mechanism baseline (ε,δ)-DP):**
- Add symmetric Gaussian noise N to Σ: N_ij ~ Normal(0, σ^2) for i≤j and N_ji=N_ij.
- Set σ using the standard Gaussian mechanism formula with sensitivity Δ_F for the clipped matrix query and δ=1e-5.
- PSD-project: Σ̃ = Π_PSD(Σ + N) (project onto the positive semidefinite cone by setting negative eigenvalues to 0).

**Wishart mechanism (ours, (ε,0)-DP):**
- Sample W ~ Wishart_K(K+1, C) where C has K identical eigenvalues equal to **3 / (2 n ε)** (Jiang et al., 2015, Theorem 4 proof), with n interpreted as the effective sample size in the covariance estimate.
- Release Σ̃ = Σ + W (PSD by construction), then compute EigenScore from Σ̃.

**Clipping-only ablation (control for confounding):**
- Apply the same embedding L2 clipping but no additive noise, to quantify how much privacy/utility change is due to clipping alone.

### Key Innovations

1. **DP for monitor logs (not model training)**: applies DP mechanisms to the *logged monitoring artifacts* produced at inference time, rather than DP-SGD or private training.
2. **Matrix-level DP for spectral monitors**: privatizes the K×K covariance that EigenScore depends on, which is low-dimensional and PSD-structured.
3. **Decisive privacy–utility evaluation**: a single experiment reports both AUROC degradation and canary inference accuracy under identical logging interfaces.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) **hallucination detection** using internal signals and self-consistency, (ii) **representation leakage / inversion** showing that internal states and embeddings can reveal sensitive text, and (iii) **differential privacy mechanisms** for releasing statistics of vectors and matrices.

Hallucination detection methods vary by what signal they use (logits, text self-consistency, activations) and what supervision they require (reference-free vs reference-based). Internal-state monitors such as EigenScore can be accurate and low-latency, but introduce a new attack surface: the **monitor logs** themselves.

Differential privacy provides a principled framework for limiting information leakage from released statistics, including matrix-valued queries such as covariances. However, it is unclear whether DP noise can be small enough to preserve the utility of **spectral monitor artifacts**, or whether these artifacts are already “safe enough” without DP.

### Related Papers

Hallucination detection / internal monitors:
- **[INSIDE: LLMs’ Internal States Retain the Power of Hallucination Detection](./references/INSIDE-LLMs-Internal-States-Retain-the-Power-of-Hallucination-Detection/meta/meta_info.txt)**: Introduces EigenScore (log-det of a K×K covariance from hidden states) and reports AUROC gains.
- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)**: Uses disagreement across sampled generations to detect hallucinations.
- **[Semantic Uncertainty](https://arxiv.org/abs/2205.12487)**: Uses semantic equivalence classes to estimate uncertainty under linguistic invariances.
- **[Semantic Entropy](https://arxiv.org/abs/2302.09664)**: Entropy-like uncertainty over semantic clusters for generation.
- **[The Internal State of an LLM Knows When It’s Lying](https://arxiv.org/abs/2304.13734)**: Shows internal representations contain truthfulness signals.
- **[PRISM](https://arxiv.org/abs/2411.04847)**: Prompt-guided internal-state hallucination detection designed for cross-domain generalization.
- **[LLM-Check](https://openreview.net/forum?id=LYx4w3CAgy)**: Benchmarks hallucination detection features including activations, with emphasis on low-overhead monitor signals.
- **[EigenTrack](https://arxiv.org/abs/2509.15735)**: Real-time spectral monitoring of hidden activations for hallucination and OOD detection, motivating deployment-time logging of spectral monitor features.
- **[LumiMAS](https://arxiv.org/abs/2508.12412)**: An observability framework for LLM multi-agent systems that logs structured telemetry for monitoring and incident analysis.
- **[InternalInspector I²](https://arxiv.org/abs/2406.12053)**: Contrastive internal representations for confidence estimation.
- **[HALT](https://arxiv.org/abs/2602.02888)**: Hallucination detection from log-probability time series.
- **[HaMI: Robust Hallucination Detection in LLMs via Adaptive Token Selection](https://arxiv.org/abs/2504.07863)**: Uses token-level selection from internal signals to improve robustness.
- **[ICR Probe](https://arxiv.org/abs/2507.16488)**: Tracks hidden-state dynamics across layers for reliable hallucination detection.

Representation inversion / activation privacy:
- **[Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816)**: Vec2Text shows black-box text reconstruction from embeddings.
- **[Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/abs/2402.12784)**: Studies defenses (e.g., quantization) against embedding inversion.
- **[Prompt Inversion Attack against Collaborative Inference of Large Language Models](https://arxiv.org/abs/2503.09022)**: Reconstructs prompts from intermediate activations in split/collaborative inference.
- **[Depth Gives a False Sense of Privacy: LLM Internal States Inversion](https://arxiv.org/abs/2507.16372)**: Systematic inversion attacks across shallow to deep layers.
- **[Split-and-Denoise](https://arxiv.org/abs/2310.09130)**: Local-DP perturbations for split inference with client-side denoising.
- **[OSNIP](https://arxiv.org/abs/2601.22752)**: Obfuscation-based privacy defense for LLM inference.

Differential privacy mechanisms (matrices and deep learning):
- **[Wishart Mechanism for Differentially Private Principal Components Analysis](./references/Wishart-Mechanism-for-Differentially-Private-Principal-Components-Analysis/meta/meta_info.txt)**: Pure-DP covariance publishing via PSD Wishart noise.
- **[Less is More: Revisiting Gaussian Mechanism for Differential Privacy](./references/Less-is-More-Revisiting-the-Gaussian-Mechanism-for-Differential-Privacy/meta/meta_info.txt)**: Low-rank / singular Gaussian DP mechanisms (R1SMG) that improve high-dimensional error scaling.
- **[DP-SGD](https://arxiv.org/abs/1607.00133)**: Standard DP training via per-example clipping and Gaussian noise.
- **[Privately Estimating a Gaussian](https://arxiv.org/abs/2212.08018)**: Tight bounds and efficient algorithms for DP covariance estimation.
- **[The Gaussian Mechanism for Differential Privacy](https://arxiv.org/abs/1405.7085)**: Standard approximate-DP mechanism used as baseline noise source.
- **[LoRA and Privacy: When Random Projections Help (and Hurt)](https://arxiv.org/abs/2601.21719)**: Analyzes Wishart projection mechanisms and privacy amplification with random projections.
- **[Revisiting Hallucination Detection Through The Lens Of Effective Rank-based Uncertainty](https://openreview.net/forum?id=0O6Xj6ljIN)**: Uses spectral effective-rank uncertainty for hallucination detection; an alternative spectral monitor family to compare conceptually against EigenScore.
- **[The Privacy-Hallucination Tradeoff in Differentially Private Language Models](https://openreview.net/forum?id=75WZP8whT8)**: Discusses how DP training can degrade factuality/hallucination behavior, motivating careful privacy–reliability analysis.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Activation-based hallucination detection | Use hidden states to predict correctness/hallucination | INSIDE; PRISM; InternalInspector; ICR Probe | QA datasets (SQuAD/CoQA/NQ/TriviaQA), TruthfulQA; AUROC | Logging internal signals creates privacy surface |
| Text/logit self-consistency monitors | Use multiple generations or logit uncertainty | SelfCheckGPT; Semantic Uncertainty; Semantic Entropy | QA and open-ended gen; AUROC/F1 | Often needs many samples; text-based signals can be brittle |
| Representation inversion attacks | Recover text or attributes from embeddings/activations | Vec2Text; Prompt Inversion; Depth Gives False Sense of Privacy | Reconstruction quality, token accuracy, ASR | Defenses often trade off utility |
| DP mechanisms for matrix release | Add calibrated noise to covariance/Gram matrices | Wishart mechanism; DP Gaussian estimation | PCA/covariance estimation error | Guarantees depend on adjacency and boundedness assumptions |
| Private inference defenses | Obfuscate or privatize representations in split/MaaS | Split-and-Denoise; OSNIP | Task accuracy + attack success | Often requires extra models/training or protocol changes |

### Closest Prior Work

- **INSIDE** defines EigenScore and evaluates hallucination detection AUROC, but does not study privacy leakage from the logged spectral artifacts.
- **Wishart mechanism** provides a pure-DP way to publish covariance matrices, but is not evaluated on LLM activation-derived monitor logs or on hallucination detection utility.
- **Less is More (R1SMG)** studies improved DP Gaussian mechanisms, suggesting that low-rank noise can improve privacy–utility for high-dimensional releases; it does not study spectral monitor logs.

**Novelty Kill Search Summary:** We searched for the specific combination “differential privacy + EigenScore / log-det spectral monitor logs for hallucination detection” via multiple arXiv/OpenReview/GitHub queries (see `notes.md` for the full log) and did not find prior work explicitly applying Wishart/R1SMG-style DP covariance mechanisms to INSIDE/EigenScore logging as of 2026-02-20.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| INSIDE | EigenScore from hidden-state covariance for hallucination detection | No privacy analysis of logged artifacts | Add DP mechanisms to Σ and quantify privacy–utility | Produces actionable deployment guidance for monitor logging |
| Split-and-Denoise | Local-DP perturbation for split inference + denoising | Requires extra denoiser model and split protocol | DP applied only to small monitor artifacts | Much lower overhead than protecting full activations |
| Wishart mechanism (DP-PCA) | Pure-DP covariance publishing via Wishart noise | Evaluated on PCA, not LLM monitoring | Apply to K×K monitor covariance | Low-dimensional Σ may allow DP without collapsing AUROC |
| Less-is-More (R1SMG) | Low-rank/singular Gaussian DP mechanisms | Not studied for monitor logs | (Out of scope here; our focus is covariance-level perturbations like Wishart + simple Gaussian baseline) | - |

---

## Experiments

### Experimental Setup

**Task**: Hallucination detection as binary classification of answer correctness.

**Base model** (primary):

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| OPT | 6.7B | https://huggingface.co/facebook/opt-6.7b | Open weights; used in INSIDE with published AUROC numbers |

**Datasets**:

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| SQuAD v2.0 (dev subset, is_impossible=False) | Utility evaluation (AUROC) | 5,928 QA pairs in INSIDE | https://huggingface.co/datasets/squad_v2 | Public research dataset |
| SQuAD v2.0 (train or sampled subset) | Canary leakage evaluation | configurable (default: 20,000 prompts) | https://huggingface.co/datasets/squad_v2 | Public research dataset |

**Evaluation script**:
- Use INSIDE’s released implementation as a reference for EigenScore computation and dataset processing: https://github.com/D2I-ai/eigenscore
- Implement DP perturbations as an additional wrapper around the computed Σ.

**Main conditions (3)** (fixed K=10, fixed decoding):
1. No-noise: Σ → EigenScore.
2. Gaussian: Σ + N → PSD projection → EigenScore.
3. Wishart: Σ + W → EigenScore.

**Primary privacy budget point:** ε=1. We also report ε∈{0.5, 2} as a small sensitivity analysis.

**Release interfaces (reported columns, not separate runs)**:
- Scalar EigenScore.
- Eigenvalue spectrum (K-dim vector).

**Attacker (privacy evaluation)**:
- Primary: multinomial logistic regression on released logs.
- Secondary (robustness): 4-layer MLP (e.g., widths 256-256-128-64).
- Strong attacker: gradient-boosted trees (XGBoost / LightGBM) on the same features.

**Randomness / seeds**:
- 3 seeds for generation (affects stochastic samples).
- 3 seeds for attacker training.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SQuAD v2.0 (filtered dev subset) | Reading comprehension QA with reference answers | AUROC (higher better) | dev | https://huggingface.co/datasets/squad_v2 | INSIDE repo + minor wrappers |
| Canary-ID leakage task (synthetic) | Prepend a random canary class ID (N=200) to each prompt and train an attacker to predict the canary from released logs (filter any outputs containing `CANARY_`) | Top-1 acc, Top-10 acc (both lower is better; chance=1/N) | train/test split | derived from SQuAD v2.0 | custom attacker training |

**Correctness labeling (utility task)**:
- Follow INSIDE: ROUGE-L f-measure threshold 0.5 and/or embedding similarity threshold 0.9 vs reference answer (fully automated). If both are implemented, report AUROC for both label definitions.

### Main Results

#### Results Table

Utility baselines (published INSIDE numbers; AUROC in %, OPT-6.7B, Table 1). INSIDE reports two correctness proxies: **AUCs** uses sentence-embedding similarity to the reference answer, and **AUCr** uses ROUGE-L thresholding; here we tabulate AUCs for consistency with the proposal’s primary labeling:

| Method | Base Model | Benchmark | AUROC (AUCs; %, higher is better) | Source | Notes |
|---|---|---|---:|---|---|
| Perplexity | OPT-6.7B | SQuAD | 58.4 | INSIDE Table 1 | Published (1 run) |
| LN-Entropy | OPT-6.7B | SQuAD | 65.5 | INSIDE Table 1 | Published (1 run) |
| Lexical Similarity | OPT-6.7B | SQuAD | 72.8 | INSIDE Table 1 | Published (1 run) |
| EigenScore (no noise) | OPT-6.7B | SQuAD | 81.7 | INSIDE Table 1 | Published (1 run) |
| **EigenScore + clipping-only (control)** | OPT-6.7B | SQuAD | **TBD** | - | To be verified (mean±std over 3 seeds); isolates effect of embedding clipping used for DP sensitivity bounding |
| **EigenScore + Gaussian DP (ours eval)** | OPT-6.7B | SQuAD | **TBD** | - | To be verified (mean±std over 3 seeds) |
| **EigenScore + Wishart DP (ours)** | OPT-6.7B | SQuAD | **TBD** | - | To be verified (mean±std over 3 seeds) |

Privacy results (to be verified; attacker on released logs; lower is better):

| Release interface | Noise mechanism | ε (and δ) | Canary Top-1 acc | Canary Top-10 acc | Notes |
|---|---|---|---:|---:|---|
| Scalar EigenScore | None | - | **TBD** | **TBD** | Phase-0 premise check |
| Spectrum (K-dim) | None | - | **TBD** | **TBD** | Phase-0 premise check |
| Scalar EigenScore | Gaussian | ε∈{0.5,1,2}, δ=1e-5 | **TBD** | **TBD** | DP baseline |
| Spectrum (K-dim) | Gaussian | ε∈{0.5,1,2}, δ=1e-5 | **TBD** | **TBD** | DP baseline |
| Scalar EigenScore | Wishart | ε∈{0.5,1,2} | **TBD** | **TBD** | Pure DP |
| Spectrum (K-dim) | Wishart | ε∈{0.5,1,2} | **TBD** | **TBD** | Pure DP |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Spectrum vs scalar release | Use eigenvalues vs log-det scalar | Scalar should leak less, possibly with similar AUROC |
| Clipping-only (no DP noise) | Apply embedding L2 clipping but no noise | Quantifies whether clipping itself suppresses leakage |
| R1SMG (dropped) | (Removed for verification simplicity; covered by a complementary agent_07 proposal that studies prompt-level DP via R1SMG on vec(Z).) | - |

### Experimental Rigor

**Premise gate (avoid non-diagnostic runs):**
- If canary-ID accuracy in the no-noise condition is not significantly above chance for both scalar and spectrum (binomial + permutation test, p<0.01), we stop and report that EigenScore logs appear non-identifying under this threat model.

**Confounders and controls:**
- **Clipping confound**: Clipping may itself reduce leakage. We apply identical clipping across all mechanisms and report a clipping-only ablation.
- **Output leakage**: We drop any example where the model output contains `CANARY_` so that leakage is not through output text.
- **Attacker underfitting**: Report train examples per canary class; compare linear vs MLP attacker.

**Variance & reproducibility:**
- Use seeds=[42, 123, 456] for generation and attacker initialization.
- Report mean±std for AUROC and leakage metrics.

### Resource Estimate

- **Compute budget**: Target ≤100 A100 GPU-hours.
  - Example: 3,000 prompts × K=10 generations × (prompt+answer length) inference-only, plus light matrix operations.
  - Expected wall-clock: ~4–8 hours on 1×A100 80GB with batching, or proportionally faster with more GPUs.
- **GPU memory**: OPT-6.7B fits in 80GB; hidden-state extraction increases activation memory but remains within 80GB with modest batch sizes.
- **API usage**: None required.

---

## Success Criteria

**Hypothesis** (directional):
- No-noise spectral logs leak canary IDs above chance (especially for spectrum logging), and DP mechanisms substantially reduce leakage; Wishart DP achieves a better privacy–utility trade-off than Gaussian at comparable ε.

**Decision Rule** (concrete):
- **Refute early (premise fails)**: If in the no-noise condition canary Top-1 accuracy is not significantly above chance (p<0.01) for both scalar and spectrum releases, stop and conclude that this threat model does not demonstrate meaningful leakage for EigenScore logs.
- **Proceed / positive result**: At ε=1 (δ=1e-5 for Gaussian), Wishart-noisy logs achieve:
  - Utility: AUROC drop ≤0.03 absolute vs no-noise EigenScore, and
  - Privacy: canary Top-1 ≤2× chance for the spectrum (and ideally for scalar), and
  - Improvement: Wishart achieves strictly lower leakage than Gaussian at matched AUROC (or matched ε).
- **Refute (DP too costly)**: If at ε=1 all DP mechanisms either (i) drop AUROC by >0.05 absolute or (ii) keep canary Top-1 >5× chance for spectrum release.

---

## Impact Statement

If successful, this work provides a concrete, deployment-relevant recipe for logging internal-state monitor statistics with significantly reduced prompt-identification risk. Even if it fails, a clean negative result (either “EigenScore logs are already non-identifying under this threat model” or “any DP noise destroys AUROC”) directly informs whether teams should log only a scalar, avoid logging altogether, or use different monitor designs.

---

## References

- [INSIDE: LLMs’ Internal States Retain the Power of Hallucination Detection](./references/INSIDE-LLMs-Internal-States-Retain-the-Power-of-Hallucination-Detection/meta/meta_info.txt) - Chen et al., 2024
- [Wishart Mechanism for Differentially Private Principal Components Analysis](./references/Wishart-Mechanism-for-Differentially-Private-Principal-Components-Analysis/meta/meta_info.txt) - Jiang et al., 2015
- [Less is More: Revisiting Gaussian Mechanism for Differential Privacy](./references/Less-is-More-Revisiting-the-Gaussian-Mechanism-for-Differential-Privacy/meta/meta_info.txt) - Ji & Li, 2023/2024
- [DP-SGD](https://arxiv.org/abs/1607.00133) - Abadi et al., 2016
- [Privately Estimating a Gaussian](https://arxiv.org/abs/2212.08018) - Alabi et al., 2022
- [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816) - Morris et al., 2023
- [SelfCheckGPT](https://arxiv.org/abs/2303.08896) - Manakul et al., 2023
- [Semantic Uncertainty](https://arxiv.org/abs/2205.12487) - Kuhn et al., 2022
- [Semantic Entropy](https://arxiv.org/abs/2302.09664) - Farquhar et al., 2023
- [The Internal State of an LLM Knows When It’s Lying](https://arxiv.org/abs/2304.13734) - Azaria & Mitchell, 2023
- [PRISM](https://arxiv.org/abs/2411.04847) - Zhang et al., 2024
- [LLM-Check](https://openreview.net/forum?id=LYx4w3CAgy) - Saha et al., 2024
- [InternalInspector I²](https://arxiv.org/abs/2406.12053) - 2024
- [HALT](https://arxiv.org/abs/2602.02888) - 2026
- [ICR Probe](https://arxiv.org/abs/2507.16488) - 2025
- [Prompt Inversion Attack against Collaborative Inference of Large Language Models](https://arxiv.org/abs/2503.09022) - 2025
- [Depth Gives a False Sense of Privacy: LLM Internal States Inversion](https://arxiv.org/abs/2507.16372) - 2025
- [Split-and-Denoise](https://arxiv.org/abs/2310.09130) - 2023
- [OSNIP](https://arxiv.org/abs/2601.22752) - 2026
- [LoRA and Privacy: When Random Projections Help (and Hurt)](https://arxiv.org/abs/2601.21719) - 2026
- [EigenTrack: Spectral Activation Feature Tracking for Hallucination and OOD Detection in LLMs and VLMs](https://arxiv.org/abs/2509.15735) - Ettori & Darabi, 2025
- [LumiMAS: A Comprehensive Framework for Real-Time Monitoring and Enhanced Observability in Multi-Agent Systems](https://arxiv.org/abs/2508.12412) - Solomon et al., 2025
- [Revisiting Hallucination Detection Through The Lens Of Effective Rank-based Uncertainty](https://openreview.net/forum?id=0O6Xj6ljIN) - Wang et al., 2026
- [The Privacy-Hallucination Tradeoff in Differentially Private Language Models](https://openreview.net/forum?id=75WZP8whT8) - 2026
- [The Gaussian Mechanism for Differential Privacy](https://arxiv.org/abs/1405.7085) - Dwork & Roth, 2014
- [HaMI: Robust Hallucination Detection in LLMs via Adaptive Token Selection](https://arxiv.org/abs/2504.07863) - Niu et al., 2025
