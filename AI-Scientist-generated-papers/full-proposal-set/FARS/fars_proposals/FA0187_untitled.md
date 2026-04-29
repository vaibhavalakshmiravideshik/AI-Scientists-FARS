# untitled

# Differentially Private Eigen-Spectrum Monitor Logs for Hallucination Detection

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly deployed in user-facing settings where factual errors (hallucinations) are costly. A common mitigation is to attach a *reliability monitor* that estimates whether a generated answer is likely correct.

Recent work shows that monitors can be much stronger when they use **internal activations** rather than only text or token probabilities. For example, **INSIDE** proposes **EigenScore**, the (regularized) log-determinant of a KxK Gram/covariance matrix built from hidden-state embeddings of K stochastic generations. INSIDE motivates EigenScore as an embedding-space "semantic divergence" / entropy signal: when the model is confident, the K embeddings are highly correlated and most eigenvalues are near 0 (low log-det), while hallucinations tend to produce more diverse embeddings (higher log-det).

However, internal activations are also a privacy risk: multiple lines of work show that embeddings and hidden states can be inverted or used as identifiers, enabling reconstruction of sensitive prompt content. In practice, this risk often appears in the *logging interface*: monitoring systems store or transmit intermediate statistics (including low-dimensional summaries) for debugging, auditing, or third-party oversight.

### The Problem

A plausible deployment assumption is: "If we only log a low-dimensional monitor artifact (e.g., the K eigenvalues used for EigenScore), we are no longer exposing raw activations, so privacy risk is low." Prior work in this repo audits leakage from spectral monitor logs via canary-ID prediction, but provides **no formal privacy guarantee**.

Differential privacy (DP) offers a standard, auditable guarantee. Informally, an (epsilon, delta)-DP logging mechanism ensures that the distribution of the released log changes by at most a factor exp(epsilon) (up to probability delta) if a single prompt changes. Unfortunately, applying DP to activation-derived objects is often believed to be impractical because standard isotropic Gaussian noise has expected error that scales with the dimension of the protected vector.

This proposal asks whether the situation is different for EigenScore-style monitoring: the released interface is only K scalars, but it is computed from a large hidden-state embedding matrix. Unlike covariance-level DP mechanisms that perturb the KxK Gram matrix directly, we enforce DP at the level of the original dK-dimensional embedding vector and rely on DP post-processing to protect any downstream spectral statistic.

### Key Insight and Hypothesis

We treat the per-prompt internal embedding matrix used by EigenScore as the sensitive object and add DP noise *before* computing the eigen-spectrum. Two design choices make this testable:

1. **Deterministic L2 clipping** of the per-prompt embedding matrix gives a simple, worst-case sensitivity bound for prompt-level DP.
2. **Rank-1 Singular Multivariate Gaussian (R1SMG)** noise (Ji and Li, USENIX Security 2024) is designed for high-dimensional query release and can have much lower expected error than full-rank isotropic Gaussian noise at the same (epsilon, delta).

**Primary hypothesis (decision-relevant)**: At a fixed privacy budget (epsilon=5, delta=1e-5), DP-protected eigen-spectrum logs using R1SMG noise can reduce adaptive canary-ID leakage to near-chance while reducing hallucination-detection AUROC by at most 2 percentage points compared to a clip-only baseline.

This outcome is genuinely uncertain because (a) the eigenvalues of a Gram matrix are nonlinear in the underlying embeddings, so moderate perturbations to embeddings can produce large changes in small eigenvalues, and (b) the clipping radius required for a valid worst-case DP statement may already remove utility-relevant signal.

---

## Proposed Approach

### Overview

For each prompt, we:

1. Generate K stochastic responses from an LLM.
2. Extract one hidden-state embedding per response and stack them into an embedding matrix Z.
3. Clip vec(Z) to a fixed L2 radius C_Z.
4. Add DP noise to vec(Z) (either isotropic Gaussian or R1SMG).
5. Compute the Gram/covariance matrix across responses, then compute the eigen-spectrum and EigenScore.
6. Release only the DP eigen-spectrum (K scalars) as the monitor log.

We evaluate two properties:

- **Privacy**: can an adaptive attacker recover a synthetic canary ID from the released DP eigen-spectrum?
- **Utility**: does EigenScore computed from the DP eigen-spectrum retain hallucination-detection AUROC?

### Method Details

#### EigenScore substrate (INSIDE)

For each prompt x, sample K responses with fixed decoding (temperature=0.5, top-p=0.99, top-k=5). For the k-th response, extract a sentence embedding s_k in R^d as the last-token hidden state at the middle layer (layer index floor(L/2)), following INSIDE.

Let Z = [s_1, ..., s_K] in R^{d x K}. Center across the K samples with J_K = I_K - (1/K) 1_K 1_K^T and form the KxK Gram/covariance matrix:

- Sigma = ((Z J_K)^T (Z J_K)) / d in R^{K x K}.

(This matches the KxK covariance/Gram construction used by INSIDE up to notation; we keep the definition fixed across all experimental conditions.)

Let lambda_1..lambda_K be the eigenvalues of Sigma + alpha I (alpha=0.001). The EigenScore is:

- EigenScore = (1/K) sum_i log(lambda_i).

We release the sorted eigen-spectrum (lambda_1..lambda_K) as the monitor log; EigenScore is derived from it.

#### Prompt-level DP via clipping

We aim for a per-prompt (example-level) DP guarantee on the released monitor log.

- Sensitive object per prompt: f(x) = vec(Z(x)) in R^M where M = d*K.
- Deterministic clipping:
  - Z_clip = Z * min(1, C_Z / ||vec(Z)||_2).

Then for any two prompts x and x', the L2 sensitivity is bounded:

- Delta_2 = sup_{x,x'} ||vec(Z_clip(x)) - vec(Z_clip(x'))||_2 <= 2 C_Z.

**Pre-registered clipping radius rule (fixed, not tuned):**
- Compute s = ||vec(Z)||_2 over a public calibration split (2k prompts).
- Set C_Z = 95th percentile of s.
- Report the clip rate on train/val/test.

#### DP noise mechanisms (two DP baselines)

We compare two mechanisms that both satisfy (epsilon, delta)-DP for the clipped vector release f(x) = vec(Z_clip(x)) with L2 sensitivity Delta_2 = 2 C_Z and dimension M = d*K.

1. **Isotropic Gaussian DP (baseline)**
   - Add n ~ N(0, sigma_G^2 I_M) to vec(Z_clip).
   - Calibrate sigma_G using the analytic Gaussian mechanism (Balle and Wang, ICML 2018).

2. **R1SMG DP (proposed)**
   - Add rank-1 singular Gaussian noise (Ji and Li, USENIX Security 2024):
     - Sample v uniformly from the unit sphere S^{M-1}.
     - Sample z ~ N(0, 1).
     - Noise n = v * sqrt(sigma_star) * z.
   - Set sigma_star using Ji and Li (Theorem 5) for (epsilon, delta)-DP.

After adding noise, reshape vec(Z_priv) back into Z_priv in R^{d x K} and compute the released eigen-spectrum from Z_priv (DP post-processing).

##### Noise-regime diagnostic (required)

A key concern is whether DP noise is so large that the experiment is trivially doomed.

We therefore run a cheap pilot on 50 prompts (same model, same decoding, same layer) to estimate:
- median signal norm: med_s = median ||vec(Z)||_2
- clipping radius: C_Z = p95(||vec(Z)||_2)

Then we report the expected noise magnitude in L2 norm.

**Gaussian scaling (why we expect it to fail at M = d*K):**
- For n ~ N(0, sigma_G^2 I_M), E||n||_2 = sigma_G * sqrt(M).
- Even under the classic Gaussian calibration sigma_G >= Delta_2 * sqrt(2 log(1.25/delta)) / epsilon (analytic Gaussian can improve constants but not the sqrt(M) scaling), we get:
  - E||n||_2 / C_Z >= 2 * sqrt(2 log(1.25/delta)) * sqrt(M) / epsilon.
- Plugging in delta=1e-5, epsilon=5, and M ~= 40960 (d~=4096, K=10) gives a lower bound of ~392x C_Z.

**R1SMG scaling (why it might be viable):**
- For R1SMG, ||n||_2 = sqrt(sigma_star) * |z|, so E||n||_2 = sqrt(sigma_star) * E|z| = sqrt(sigma_star) * sqrt(2/pi).
- Using Ji and Li (Theorem 5) with Delta_2 = 2 C_Z and psi(M, delta) ~= 1 for M >> 1, we get sigma_star ~= 8 C_Z^2 / epsilon and thus:
  - E||n||_2 / C_Z ~= sqrt(16 / (pi * epsilon)) ~= 1.0 at epsilon=5.

**Guardrail / early stop:** if the pilot finds E||n||_2 / med_s > 10 even for R1SMG at epsilon=5, we stop early and refute that prompt-level DP via vec(Z) is viable for this monitoring interface in this setting.

#### Threat model and attacker

- The attacker observes released eigen-spectrum logs for many prompts and knows epsilon, delta, C_Z, and the mechanism family.
- The attacker is adaptive: trained on the defended distribution produced by the same mechanism and privacy budget.
- Attack models (report max leakage over these):
  - Logistic regression on the K-dimensional spectrum.
  - MLP classifier (2-layer, small).
  - Denoise + MLP attacker: first train a small denoiser g(noisy_spectrum)->clean_spectrum using clean/noisy pairs generated by running the same base model and then applying the known noise mechanism; then train an MLP on g(noisy_spectrum) to predict the canary ID.

**Why canary-ID prediction is a meaningful proxy:** This is an attribute-inference style evaluation of whether the released log is *identifying* for a persistent user-level attribute (here, a synthetic user ID). Canary-style identifiers and synthetic PII are commonly used to audit privacy leakage and extraction risk in LLM systems (e.g., [PrivacyXray](./references/PrivacyXray-Detecting-Privacy-Breaches-in-LLMs-through-Semantic-Consistency-and-Probability-Certainty/meta/meta_info.txt); [Extracting books from production language models](./references/Extracting-books-from-production-language-models/meta/meta_info.txt)). Our canary is deliberately filtered from the output text so any successful prediction must come from the logged monitor artifact rather than trivial string repetition.

### Key Innovations

1. **Formal DP for monitor logs**: moves from measuring leakage of spectral logs to releasing them with an explicit (epsilon, delta)-DP guarantee.
2. **Noise mechanism suited to high-dimensional internal states**: tests whether R1SMG can make prompt-level DP practical for hidden-state-derived monitoring.
3. **Decisive privacy-utility tradeoff measurement**: one dataset, one privacy budget, three conditions (clip-only, clip+Gaussian, clip+R1SMG) that directly decide whether DP is viable for this logging interface.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (1) hallucination detection using internal states, (2) representation inversion and privacy leakage from embeddings/hidden states, and (3) differential privacy mechanisms for high-dimensional query release.

Internal-state monitors such as EigenScore can improve reliability estimation, but the same internal representations can be identifying or invertible. Existing mitigations for activation privacy include protocol changes (e.g., split inference defenses), learned obfuscation maps, and local DP perturbations. These approaches typically modify representations during inference; by contrast, we focus on a simpler deployment question: can we keep the monitor but release only a DP-protected monitor log.

On the DP side, classic isotropic Gaussian noise is widely used but can be utility-destroying for high-dimensional releases. Recent DP mechanism design (e.g., R1SMG) suggests that the geometry of privacy proofs can allow far lower expected error than full-rank noise, motivating a targeted test in the activation-monitoring setting.

### Related Papers

- **[INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](./references/INSIDE-LLMs-Internal-States-Retain-the-Power-of-Hallucination-Detection/meta/meta_info.txt)**: introduces EigenScore from eigenvalues of a hidden-state Gram matrix for hallucination detection.
- **[The Internal State of an LLM Knows When Its Lying](https://arxiv.org/abs/2304.13734)**: early evidence that internal representations encode truthfulness signals.
- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)**: hallucination detection via disagreement across sampled generations.
- **[Semantic Uncertainty](https://arxiv.org/abs/2205.12487)**: uncertainty via semantic equivalence classes for generation.
- **[Semantic Entropy](./references/Semantic-Entropy-Probes-Robust-and-Cheap-Hallucination-Detection-in-LLMs/meta/meta_info.txt)**: uses semantic clustering + entropy to detect hallucinations and uncertainty.
- **[LLM-Check](https://openreview.net/forum?id=LYx4w3CAgy)**: evaluates multiple hallucination detection signals including internal representations.
- **[MHAD](https://www.ijcai.org/proceedings/2025/0929.pdf)**: selects hallucination-aware neurons/layers for detection.
- **[Text Embeddings Reveal (Almost) As Much As Text](./references/Text-Embeddings-Reveal-Almost-As-Much-As-Text/meta/meta_info.txt)**: shows black-box reconstruction of text from embeddings (Vec2Text).
- **[Language Models are Injective and Hence Invertible](./references/Language-Models-are-Injective-and-Hence-Invertible/meta/meta_info.txt)**: proves decoder-only Transformers are almost surely injective and proposes SipIt for prompt recovery from hidden states (arXiv:2510.15511).
- **[Depth Gives a False Sense of Privacy: LLM Internal States Inversion](./references/Depth-Gives-a-False-Sense-of-Privacy-LLM-Internal-States-Inversion/meta/meta_info.txt)**: strong inversion attacks across layers, challenging the intuition that deeper is safer (arXiv:2507.16372).
- **[Prompt Inversion Attack against Collaborative Inference of Large Language Models](./references/Prompt-Inversion-Attack-against-Collaborative-Inference-of-Large-Language-Models/meta/meta_info.txt)**: prompt recovery from intermediate activations in collaborative inference (arXiv:2503.09022).
- **[Split-and-Denoise: Protect large language model inference with local differential privacy](./references/Split-and-Denoise-Protect-large-language-model-inference-with-local-differential-privacy/meta/meta_info.txt)**: local-DP noise on embeddings plus denoising for split inference (arXiv:2310.09130).
- **[Cascade: Token-Sharded Private LLM Inference](./references/Cascade-Token-Sharded-Private-LLM-Inference/meta/meta_info.txt)**: protocol defense via token sharding to defeat vocab-matching attacks (arXiv:2507.05228).
- **[Concept-Aware Privacy Mechanisms for Defending Embedding Inversion Attacks](./references/Concept-Aware-Privacy-Mechanisms-for-Defending-Embedding-Inversion-Attacks/meta/meta_info.txt)**: anisotropic/elliptical perturbations targeting sensitive embedding dimensions (arXiv:2602.07090).
- **[The Algorithmic Foundations of Differential Privacy](./references/The-Algorithmic-Foundations-of-Differential-Privacy/meta/meta_info.txt)**: standard DP definitions and composition.
- **[Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising](./references/Improving-the-Gaussian-Mechanism-for-Differential-Privacy-Analytical-Calibration-and-Optimal-Denoising/meta/meta_info.txt)**: analytic Gaussian mechanism calibration.
- **[Differentially Private Covariance Estimation](./references/Differentially-Private-Covariance-Estimation/meta/meta_info.txt)**: pure-DP covariance estimation (Amin et al., NeurIPS 2019).
- **[Differentially Private Covariance Revisited](./references/Differentially-Private-Covariance-Revisited/meta/meta_info.txt)**: improved DP covariance estimation bounds (Dong et al., NeurIPS 2022).
- **[Wishart Mechanism for Differentially Private Principal Components Analysis](./references/Wishart-Mechanism-for-Differentially-Private-Principal-Components-Analysis/meta/meta_info.txt)**: DP covariance publishing via Wishart noise.
- **[Less is More: Revisiting the Gaussian Mechanism for Differential Privacy](./references/Less-is-More-Revisiting-the-Gaussian-Mechanism-for-Differential-Privacy/meta/meta_info.txt)**: introduces R1SMG rank-1 singular Gaussian noise with low expected error for high-dimensional queries.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Internal-state hallucination monitors | Use hidden states to score correctness/uncertainty | INSIDE; PRISM; InternalInspector; LLM-Check | QA AUROC/PCC | Requires white-box access; multiple sampling |
| Representation inversion | Recover prompts/attributes from embeddings/hidden states | Vec2Text; Depth Gives a False Sense of Privacy; Prompt Inversion | Reconstruction quality; attribute inference | Strong attackers can adapt to defenses |
| DP for representations / split inference | Add noise/obfuscation to protect embeddings in transit | Split-and-Denoise; Cascade; concept-aware embedding perturbation (SPARSE) | Utility vs privacy under threat models | Often costly or protocol-dependent |
| DP mechanisms for high-dimensional release | Calibrate noise for (epsilon, delta)-DP with better utility | Analytic Gaussian; DP covariance estimation; R1SMG | Error vs privacy budget | Nonlinear downstream functions may amplify noise; eigenvalue maps can be especially sensitive for small eigenvalues |

### Closest Prior Work

- **INSIDE (EigenScore)** provides a strong hallucination monitor based on the eigen-spectrum of hidden-state covariance, but does not address privacy of the logged statistic.
- **Activation/embedding inversion work** shows that raw representations leak prompts, but does not study low-dimensional DP-protected monitor logs.
- **Split-inference DP defenses** focus on modifying representations to make inference private; they do not answer the deployment question of whether a monitor log can be released with a formal DP guarantee while retaining monitor utility.
- **R1SMG** provides a general DP mechanism for high-dimensional query release with much lower expected error than isotropic Gaussian, but has not been tested on LLM monitor artifacts derived from internal states.

**Differentiation vs parallel draft (agent_04):** another agent draft in this repo privatizes the *KxK covariance matrix* directly (Wishart / Gaussian-on-Sigma) and explicitly treats DP as a calibrated noise family without claiming prompt-level DP. This proposal is complementary: it targets a stronger statement (prompt-level (epsilon, delta)-DP for the full dK-dimensional embedding vector via clipping + R1SMG) and uses a noise-regime diagnostic to predict whether the experiment is in a non-trivial regime before spending the full verification budget.

**Novelty Kill Search Summary:** We searched for the specific combination of "EigenScore" (INSIDE hallucination detector) with "differential privacy" and "eigen-spectrum / log determinant" releases using queries such as "EigenScore differential privacy", "hallucination detection internal states differential privacy", and "differential privacy eigenvalue spectrum release Gram matrix", plus OpenReview checks. As of 2026-02-20, we did not find prior work that releases EigenScore/eigen-spectrum monitor logs with an explicit DP guarantee.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| INSIDE (EigenScore) | EigenScore from eigenvalues of hidden-state Gram matrix for hallucination detection | No privacy guarantee for logged monitor artifacts | Add per-prompt (epsilon, delta)-DP noise before computing eigen-spectrum | Provides an auditable privacy-utility tradeoff for deployment |
| Split-and-Denoise / split-inference DP | DP perturbations for split inference embeddings | Protocol-level and representation-level changes; not targeted to monitor logs | Focus on DP for the *monitor logging interface* | Lower engineering burden if it works |
| Isotropic Gaussian DP | Standard DP mechanism for vector queries | Expected error scales poorly with dimension | Compare against R1SMG for same (epsilon, delta) | Tests whether improved mechanism makes DP viable |
| R1SMG | Rank-1 singular Gaussian DP with low expected error in high dimensions | Not evaluated for nonlinear eigen-spectrum post-processing | Apply R1SMG to per-prompt embedding matrix, then compute eigen-spectrum | If utility survives, gives practical DP recipe |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):** This is a privacy-utility tradeoff study for a logging interface. The baseline ladder is:
1. Clip-only (no DP) monitor log (upper bound on utility, lower bound on privacy).
2. Clip + isotropic Gaussian DP (standard DP baseline).
3. Clip + R1SMG DP (proposed).

We also include sanity-check attackers (output-text-only after canary filtering) and progressively stronger attackers (logistic regression -> MLP -> optional denoiser+MLP).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3.1-8B-Instruct (or comparable open instruct LM) | ~8B | https://huggingface.co/meta-llama | bf16 inference; extract mid-layer last-token embedding |

**Training Data (if applicable):**

No training/fine-tuning of the base LLM. Only attacker models are trained.

**Resource Estimate** (fits <=768 GPU-hours):

- Main run: SQuAD dev subset (about 5.9k prompts) with K=10 generations -> about 59k generations.
- If average (prompt+output) length is 800 tokens, total forward tokens ~47M.
- This should fit within budget on up to 32x A100-80GB with vLLM batching.
- If not, downscale to K=5 (half generations) or sample 3k prompts.

**Noise-regime diagnostic (required for interpretation):**
- Report C_Z and the ratio of expected noise norm to typical ||vec(Z)||_2 for both Gaussian and R1SMG (computed from their calibration formulas).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Canary-SQuAD | SQuAD dev-v2.0 subset with is_impossible=False and a synthetic canary ID prepended to each prompt; outputs containing the canary substring are filtered out | Privacy: Top-1 and Top-10 canary-ID accuracy (lower is safer); Utility: AUROC of EigenScore for correctness detection (higher is better) | calibration/train/val/test | https://huggingface.co/datasets/rajpurkar/squad_v2 | Custom wrapper + standard ROUGE-L |

**Correctness labeling (utility ground truth):**
- Follow INSIDE: mark generation as correct if ROUGE-L (f-measure) with the ground-truth answer exceeds 0.5.

### Main Results

We report two types of numbers:
- **Comparable (ours)**: results obtained by running the proposed pipeline on Canary-SQuAD with the chosen base model.
- **Literature anchors (not directly comparable)**: INSIDE's reported AUROC on vanilla SQuAD dev-v2.0 using LLaMA/OPT models (different base model family and no canary prompt prefix).

| Method (released log) | DP mechanism | Base Model | Benchmark | Canary Top-1 acc (mean+-std) | Canary Top-10 acc (mean+-std) | AUROC (mean+-std) | Source | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| Eigen-spectrum (K dims) | Clip-only | Llama-3.1-8B | Canary-SQuAD | TBD | TBD | TBD | Ours | Utility upper bound; not DP |
| Eigen-spectrum (K dims) | Clip + isotropic Gaussian | Llama-3.1-8B | Canary-SQuAD | TBD | TBD | TBD | Ours | Standard DP baseline |
| Eigen-spectrum (K dims) | Clip + R1SMG | Llama-3.1-8B | Canary-SQuAD | TBD | TBD | TBD | Ours | Proposed |
| EigenScore scalar | None | LLaMA-7B | SQuAD dev-v2.0 | - | - | AUCr=81.2, AUCs=81.5 | INSIDE Table 1 (Sec 4.2) | Anchor only |
| EigenScore scalar | None | OPT-6.7B | SQuAD dev-v2.0 | - | - | AUCr=80.8, AUCs=81.7 | INSIDE Table 1 (Sec 4.2) | Anchor only |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| Scalar-only release | Release only EigenScore instead of K eigenvalues | Lower leakage but also lower attacker feature richness |
| K=5 vs K=10 | Change monitor sampling budget | DP noise may interact with K (larger K may be more robust but costs more) |

### Experimental Rigor

- **Seeds**: generation seeds = [42, 123, 456]; attacker init seeds = [1, 2, 3].
- **Adaptive attacker**: attacker trained on defended distribution and given epsilon, delta, C_Z, and mechanism family.
- **Sanity checks**:
  - Output-text-only attacker after canary filtering should be near chance.
  - Clip rate should be close to 5% by construction; if much larger on train/test, the calibration split is not representative.
- **Controls**:
  - Balanced canary IDs across splits.
  - Filter out outputs that contain the canary substring to avoid trivial leakage via text.

---

## Success Criteria

**Hypothesis**: Clip + R1SMG DP makes eigen-spectrum monitor logs non-identifying (near chance canary-ID accuracy) while preserving most hallucination-detection AUROC compared to clip-only.

**Decision Rule**:

- **Proceed** (DP monitor logs viable):
  - Utility: AUROC(clip+R1SMG) >= AUROC(clip-only) - 2.0 points.
  - Privacy: canary Top-1 accuracy <= 1.0% (chance is 0.5% for N=200) and Top-10 accuracy <= 7.0% (chance is 5.0%).
- **Pivot** (DP seems possible but needs weaker privacy):
  - If privacy is near chance but AUROC drops by 2-5 points, try epsilon=20 as a diagnostic and report the tradeoff curve.
- **Refute** (DP monitor logs not viable at practical privacy):
  - If AUROC drop > 5 points, or canary Top-1 accuracy remains > 5% (10x chance), under clip+R1SMG.

Secondary criterion (mechanism value): if clip+R1SMG achieves similar privacy with higher AUROC than clip+Gaussian, it supports using improved DP mechanisms for high-dimensional activation-derived releases.

---

## Impact Statement

If successful, this work provides a concrete recipe for deployers: release only DP-protected eigen-spectrum monitor logs (not raw activations) with an interpretable (epsilon, delta) guarantee, while retaining most hallucination-monitor utility. If it fails (utility collapse or persistent leakage), the result is still decision-changing: it indicates that DP is too costly for this logging interface and that practitioners should prefer alternative protections (e.g., trusted execution environments or protocol-level private inference) when using internal-state monitors.

---

## References

- INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection. `./references/INSIDE-LLMs-Internal-States-Retain-the-Power-of-Hallucination-Detection/`
- Less is More: Revisiting the Gaussian Mechanism for Differential Privacy. `./references/Less-is-More-Revisiting-the-Gaussian-Mechanism-for-Differential-Privacy/`
- Wishart Mechanism for Differentially Private Principal Components Analysis. `./references/Wishart-Mechanism-for-Differentially-Private-Principal-Components-Analysis/`
- Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising. `./references/Improving-the-Gaussian-Mechanism-for-Differential-Privacy-Analytical-Calibration-and-Optimal-Denoising/`
- Differentially Private Covariance Estimation (NeurIPS 2019). `./references/Differentially-Private-Covariance-Estimation/`
- Differentially Private Covariance Revisited. `./references/Differentially-Private-Covariance-Revisited/`
- Split-and-Denoise: Protect large language model inference with local differential privacy. `./references/Split-and-Denoise-Protect-large-language-model-inference-with-local-differential-privacy/`
- Cascade: Token-Sharded Private LLM Inference. `./references/Cascade-Token-Sharded-Private-LLM-Inference/`
- Text Embeddings Reveal (Almost) As Much As Text. `./references/Text-Embeddings-Reveal-Almost-As-Much-As-Text/`
- Depth Gives a False Sense of Privacy: LLM Internal States Inversion. `./references/Depth-Gives-a-False-Sense-of-Privacy-LLM-Internal-States-Inversion/`
- The Algorithmic Foundations of Differential Privacy. `./references/The-Algorithmic-Foundations-of-Differential-Privacy/`
