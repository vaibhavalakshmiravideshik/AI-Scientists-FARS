# untitled

# Anisotropic Noise Fingerprints Reveal Concept Choice in Concept-Aware Embedding Privacy

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Dense text embeddings are a core component of retrieval-augmented generation (RAG), semantic search, and recommendation systems. They are often stored in vector databases and exposed through “embedding-as-a-service” APIs. However, multiple works show that text embeddings can be inverted to recover sensitive content, including personally identifiable information (PII) and medical attributes (e.g., Vec2Text, which reconstructs text from embeddings).

Differential-privacy-inspired embedding sanitization is a common mitigation strategy: release a perturbed embedding instead of the raw embedding. Recent work goes further and proposes **concept-aware** protection: the user specifies which sensitive concept to protect (e.g., diseases vs gender terms), and the sanitizer applies stronger perturbations only along embedding dimensions that encode that concept. **SPARSE** (Tsai et al., 2026) is a representative approach: it learns a concept-specific sensitivity mask over embedding dimensions and uses a Mahalanobis-norm noise mechanism (elliptical noise shaped by a concept-specific covariance matrix) to inject **anisotropic** noise.

Concept-aware sanitization changes the deployment interface: users (or applications) effectively choose a *privacy mode* (“protect diseases”, “protect location”, etc.). In many real systems, that choice is itself sensitive. For example, choosing “protect diseases” may imply medical context; choosing “protect politics” may imply political affiliation risk; choosing “protect gender” may imply an at-risk demographic attribute. If the privacy mode leaks, an attacker can profile users even if token-level leakage is reduced.

### The Problem

Existing evaluations of concept-aware embedding sanitization focus on whether sensitive tokens can be reconstructed from a single released embedding. They typically do not ask whether the mechanism leaks the **privacy configuration** itself.

In deployments, multiple independent releases of the same underlying embedding can occur without exposing plaintext:
- periodic re-indexing that re-embeds the same document IDs
- retries / replication across services that regenerate embeddings
- snapshotting of vector stores for backups or analytics

We are not aware of published measurements on how often such multi-release patterns occur in embedding deployments; we treat this as plausible but under-studied, and test its implications when it does occur.

If a sanitizer re-samples noise per release, an attacker who obtains two or more releases for the same document ID may be able to estimate the noise statistics. For anisotropic mechanisms, the noise covariance (or per-dimension variance) can act as a fingerprint.

This proposal studies a concrete question:

> **Does a concept-aware anisotropic noise mechanism leak which privacy concept a user selected, under multi-release access to sanitized embeddings?**

### Key Insight and Hypothesis

**Key insight:** SPARSE’s Mahalanobis mechanism uses a concept-specific diagonal matrix \(\Sigma_C = \mathrm{diag}(m_C + \delta)\) (normalized to fixed trace) derived from a learned sensitivity mask \(m_C\). If an attacker sees two independent sanitized embeddings for the same document,
\[
 z^{(1)} = \Phi(x) + \eta^{(1)}, \quad z^{(2)} = \Phi(x) + \eta^{(2)}, \quad \eta^{(t)} \sim \mathrm{Mahalanobis}(\Sigma_C),
\]
then the difference \(\Delta=z^{(1)}-z^{(2)}\) cancels \(\Phi(x)\) and reveals second-moment information about the noise. Aggregating \(\Delta\odot\Delta\) over many documents from the same user (or the same privacy mode) yields a stable estimate of the per-dimension variance profile, which can be matched to concept templates.

**Hypothesis:** With a realistic multi-release threat model (two releases per document, many document IDs), an attacker can infer the privacy concept \(C\) with accuracy far above chance for concept-aware anisotropic noise, while isotropic noise is near chance. A simple **covariance smoothing** modification \(\Sigma_{\text{mix}} = (1-\lambda)\Sigma_C + \lambda I\) will substantially reduce concept-identification accuracy without large utility loss.

We could be wrong if different concepts yield highly overlapping masks \(m_C\), making \(\Sigma_C\) too similar to distinguish, or if the Mahalanobis noise distribution is heavy-tailed enough that practical second-moment estimation requires unrealistically many document IDs.

---

## Proposed Approach

### Overview

We propose a minimal, fully automated evaluation of **concept-choice leakage** for concept-aware embedding sanitization:

1. Train SPARSE-style concept masks \(m_{C_k}\) for a fixed menu of \(K\) privacy concepts.
2. For each concept, simulate multi-release access to sanitized embeddings for many document IDs.
3. Compute a variance-profile fingerprint from embedding differences and perform concept inference via template matching.
4. Evaluate a simple mitigation (covariance smoothing) that reduces anisotropy.

The core output is a decision-changing result for deployers: whether “concept-aware anisotropic noise” is safe to expose as a user-facing privacy mode under realistic multi-release access.

### Method Details

**Concept mask learning (SPARSE):**
- Fix an embedder \(\Phi\) (e.g., `sentence-transformers/gtr-t5-base`, 768-d).
- For each concept \(C_k\), build \(D^+\) as sentences containing tokens from \(C_k\), and \(D^-\) by removing those tokens from each \(s\in D^+\).
- Train a hard-concrete mask \(m_{C_k}\in[0,1]^d\) and a small MLP classifier to distinguish \(D^+\) vs \(D^-\), with L0-style sparsity regularization (as in SPARSE).
- Define \(\Sigma_{C_k}=\mathrm{diag}(m_{C_k}+\delta)\) and normalize to \(\mathrm{tr}(\Sigma_{C_k})=d\).

**Noise mechanisms (three conditions):**
We use privacy budget **\(\varepsilon=10\)** by default (a standard setting in SPARSE’s main tables).
- **A) Concept-aware anisotropic:** sample \(\eta\sim f(\eta)\propto\exp(-\varepsilon\|\eta\|_{M})\) with \(\|\eta\|_M=\sqrt{\eta^T\Sigma_{C_k}^{-1}\eta}\) (SPARSE Mahalanobis mechanism).
- **B) Isotropic control:** \(\Sigma=I\) (trace matched by construction).
- **C) Covariance smoothing:** \(\Sigma_{\text{mix}}=(1-\lambda)\Sigma_{C_k}+\lambda I\) with **\(\lambda=0.2\)** by default, then re-normalize to \(\mathrm{tr}(\Sigma_{\text{mix}})=d\).

**Attacker (template matching):**
Assume the attacker knows the sanitizer family (SPARSE/Mahalanobis), the menu of \(K\) concepts, and can obtain (or approximate) concept templates \(\{\Sigma_{C_k}\}\) from public calibration data. The attacker does **not** have plaintext.

Given two releases per document ID, compute \(\Delta_i = z_i^{(1)}-z_i^{(2)}\) for each doc, then compute a group fingerprint over a set of docs \(\mathcal{G}\):
\[
 v(\mathcal{G}) = \frac{1}{|\mathcal{G}|}\sum_{x\in\mathcal{G}} \Delta(x)\odot\Delta(x)\in\mathbb{R}^d.
\]
Normalize \(\tilde v = v / \sum_j v_j\) to remove overall scale. Predict
\[
 \hat k = \arg\min_k \|\tilde v - \widetilde{\mathrm{diag}}(\Sigma_{C_k})\|_2.
\]

(Secondary analysis: replace template matching with a lightweight classifier trained on fingerprints to model imperfect template knowledge.)

### Key Innovations

- **New leakage target for embedding privacy:** Measure leakage of the *privacy configuration* (“which concept is protected”), not just leakage of sensitive tokens.
- **Mechanism-based attack:** A simple second-moment estimator on multi-release embedding differences isolates a concrete failure mode of anisotropic noise.
- **Minimal mitigation:** Covariance smoothing provides a tunable, training-free way to reduce concept fingerprints without changing the mask-learning pipeline.

---

## Related Work

### Field Overview

**Embedding inversion and leakage.** Multiple papers show that text embeddings can leak substantial information and can be inverted to recover text or attributes (e.g., Vec2Text, GEIA, and token-level MLP attackers). These results motivate sanitization mechanisms that perturb embeddings before storage or sharing.

**Noise-based / DP-style embedding sanitization.** Local/metric differential privacy mechanisms (often Laplace/Gaussian-like) are a common approach to embedding sanitization. Recent work explores improved utility-privacy trade-offs via geometry-aware noise (e.g., calibrated multivariate perturbations, metric-LDP for sentence embeddings) and via concept-aware perturbations (SPARSE).

**Privacy beyond the data value (“privacy of privacy”).** Differential privacy is often interpreted as limiting inferences about individuals’ data, but several lines of work study what can still be inferred under correlations (inferential privacy) or what can be inferred about privacy parameters and implementations (DP auditing tools). In deployed local DP, repeated reports can enable inference of user preferences (pool inference attacks). In a different domain, concept erasure for generative models can be circumvented, and the targeted concept itself can become inferable.

This proposal connects these threads: concept-aware embedding sanitization introduces a user-visible privacy configuration, and anisotropic noise can expose a fingerprint of that configuration under repeated releases.

### Related Papers

- **[Concept-Aware Privacy Mechanisms for Defending Embedding Inversion Attacks](./references/CONCEPT-AWARE-PRIVACY-MECHANISMS-FOR-DEFENDING-EMBEDDING-INVERSION-ATTACKS/meta/meta_info.txt)**: Introduces SPARSE (mask learning + Mahalanobis mechanism) for concept-aware embedding sanitization.
- **[Sanitizing Sentence Embeddings (and Labels) for Local Differential Privacy](https://dl.acm.org/doi/10.1145/3543507.3583512)**: Metric-LDP mechanisms for sentence embeddings with utility evaluation on downstream tasks.
- **[Privacy-and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations](https://arxiv.org/abs/1910.11947)**: Multivariate perturbation mechanisms for text privacy with geometry-aware noise.
- **[Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816)**: Vec2Text inversion results motivating embedding privacy.
- **[Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/abs/2402.12784)**: Studies inversion factors and mitigations such as noise injection and secret transforms.
- **[Sentence embedding leaks more information than you expect: Generative embedding inversion attack](https://arxiv.org/abs/2305.03010)**: GEIA, a generative embedding inversion attacker.
- **[Information Leakage in Embedding Models](https://dl.acm.org/doi/10.1145/3372297.3417880)**: Early study of embedding leakage and token-level attacks.
- **[Privacy-preserving neural representations of text](https://arxiv.org/abs/1809.01454)**: Early work on privatizing text representations.
- **[Split-and-Denoise: Protecting LLM Inference with Local Differential Privacy](https://arxiv.org/abs/2310.09130)**: Local randomization for privacy in split inference settings.
- **[OSNIP: Breaking the Privacy-Utility-Efficiency Trilemma in LLM Inference via Obfuscated Semantic Null Space](https://arxiv.org/abs/2601.22752)**: Representation obfuscation for inference-time privacy.
- **[RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response](./references/RAPPOR-Randomized-Aggregatable-Privacy-Preserving-Ordinal-Response/meta/meta_info.txt)**: Longitudinal local DP deployment with permanent randomization; motivates multi-release threat models.
- **[Pool Inference Attacks on Local Differential Privacy](https://www.usenix.org/conference/usenixsecurity22/presentation/gadotti)**: Shows repeated LDP reports can reveal user-level preferences under realistic deployments.
- **[Inferential Privacy Guarantees for Differentially Private Mechanisms](https://arxiv.org/abs/1603.01508)**: Formalizes inferential privacy under correlated priors.
- **[Group and Attack: Auditing Differential Privacy](https://files.sri.inf.ethz.ch/website/papers/ccs23-groupattack.pdf)**: Delta-Siege tool for finding privacy violations in (ε,δ)-DP implementations.
- **[What Are the Chances? Explaining the Epsilon Parameter in Differential Privacy](https://www.usenix.org/system/files/usenixsecurity23-nanayakkara.pdf)**: Studies interpretation/communication of DP parameters; relevant to privacy-configuration semantics.
- **[Circumventing Concept Erasure Methods For Text-To-Image Generative Models](https://arxiv.org/abs/2308.01508)**: Shows targeted concept removal can be bypassed; targeted concepts become a key axis of adversarial reasoning.
- **[The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)**: Standard DP foundations and composition results.
- **[Differential Privacy](https://dl.acm.org/doi/10.1145/3085504)**: Survey/overview of DP mechanisms and interpretations.
- **[Local Differential Privacy: A Survey](https://arxiv.org/abs/2008.05187)**: Overview of local DP mechanisms and deployment considerations.
- **[Rényi Differential Privacy](https://arxiv.org/abs/1702.07476)**: Alternative privacy accounting relevant to repeated releases.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Embedding inversion | Reconstruct text/attributes from embeddings | Vec2Text; GEIA; Song & Raghunathan | NQ/MSMARCO-style corpora; MTEB | Strong dependence on threat model |
| DP / metric-LDP sanitization | Add calibrated noise to embeddings for privacy | SentenceDP; Feyisetan’20; SPARSE | STS (Pearson), retrieval (NDCG@10) | Often single-release evaluation |
| Concept-aware sanitization | User specifies sensitive concept; perturb sensitive dimensions | SPARSE | Token leakage + utility trade-off | Privacy configuration may leak |
| Longitudinal local DP inference | Repeated reports reveal user preferences | RAPPOR; Pool inference attacks | Telemetry/behavior datasets | Requires careful threat modeling |
| Parameter / implementation analysis | Probe privacy parameters or violations | Inferential privacy; Delta-Siege | Theoretical + tool-based | Not focused on embeddings |

### Closest Prior Work

- **SPARSE (Tsai et al., 2026)**: Learns concept-specific sensitive dimensions and injects Mahalanobis noise. It evaluates token leakage and utility, but does not evaluate whether the concept choice itself is inferable under multi-release access.
- **SentenceDP (Du et al., 2023)**: Studies metric-LDP sanitization of sentence embeddings with strong utility evaluation, but is concept-agnostic and does not create a concept-specific covariance fingerprint.
- **Pool inference attacks (Gadotti et al., 2022)**: Demonstrates user-preference inference from repeated LDP reports; closest in spirit to our “privacy of privacy” framing, but in discrete telemetry rather than continuous embeddings.
- **Inferential privacy (Ghosh & Kleinberg, 2017)**: Formal study of what DP implies for Bayesian inference under correlations; motivates that DP does not necessarily prevent higher-level inference.

**Novelty Kill Search Summary:** We searched for combinations of “SPARSE covariance fingerprint”, “mask inference embedding DP”, “anisotropic noise covariance fingerprint attack”, “Mahalanobis noise fingerprint embeddings”, and “concept choice leakage privacy mechanism”, and checked for direct prior work on concept-choice inference from concept-aware embedding sanitization. As of 2026-02-26, we did not find work that explicitly tests or mitigates concept-choice leakage in SPARSE-style anisotropic embedding sanitization.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| SPARSE (Tsai’26) | Concept-aware mask + Mahalanobis noise; token leakage vs utility | Does not test leakage of the *concept choice* | Define and measure concept-choice leakage; propose smoothing | Directly targets an unmeasured deployment risk |
| SentenceDP (Du’23) | Metric-LDP for embeddings; strong utility evaluation | Concept-agnostic; no concept menu to leak | Focus on concept-aware mechanisms with user-chosen modes | Identifies a new risk introduced by concept-aware interfaces |
| Pool inference (Gadotti’22) | Preference inference from repeated LDP telemetry reports | Discrete telemetry setting | Continuous embeddings; covariance fingerprints | Transfers a known longitudinal failure mode to embeddings |
| Inferential privacy (Ghosh’17) | Theory for Bayesian inference under DP + correlations | Not about embeddings or covariance fingerprints | Concrete, measurable failure mode + mitigation | Bridges theory to a practical mechanism |

---

## Experiments

### Experimental Setup

**Threat model (explicit):**
- Attacker observes only sanitized embeddings \(z\) and stable document IDs across releases (e.g., vector DB snapshots or embedding-service logs that store embeddings but omit plaintext).
- Attacker can match the same document across two releases via ID.
- Attacker knows the sanitizer family and the menu of \(K\) privacy concepts, but not which concept a given user selected.
- Attacker does not have plaintext, so cannot re-embed locally.

**Base embedder:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| `sentence-transformers/gtr-t5-base` | 768-d | https://huggingface.co/sentence-transformers/gtr-t5-base | Used by SPARSE as default embedder in experiments |

**Datasets (download links):**

| Dataset | Purpose | Size | Download Link | Notes |
|---|---|---:|---|---|
| PII-Masking-300K | Source corpus to mine sentences containing concept tokens | 300k rows | https://huggingface.co/datasets/ai4privacy/pii-masking-300k | Public dataset with many entity-like tokens |
| STS12 | Utility evaluation (semantic similarity) | 10,684 pairs | https://github.com/embeddings-benchmark/mteb | Pearson correlation between similarity scores and embedding cosine similarity |

**Privacy concept menu (K=5):**
We define five token sets (all lowercased matching, simple regex tokenization):
- Weekdays: {monday, tuesday, wednesday, thursday, friday, saturday, sunday}
- Months: {january, …, december}
- Countries: a fixed list of ~50 common country names
- Gender terms: {he, she, him, her, man, woman, male, female}
- City names: a fixed list of ~100 common city names

(Exact token lists will be included in the verification implementation.)

**Mask training (per concept):**
- Sample \(|D^+|=10{,}000\) sentences from PII-Masking-300K that contain at least one token from concept \(C_k\).
- Construct \(D^-\) by removing those concept tokens from each \(s\in D^+\).
- Train mask + predictor as in SPARSE (100 epochs, batch=64, lr=1e-4, λ=0.001). If runtime is unexpectedly high, downscale to 30 epochs as a controlled variant.
- **Data leakage note:** because \(D^-\) is derived from \(D^+\), we will split at the sentence level first (train/val/test sentences), then derive \(D^-,D^+\) within each split to avoid train/test overlap.

**Multi-release generation:**
- For each document, generate \(N=2\) independent sanitized embeddings under each condition (A/B/C).

**User-level grouping for concept inference:**
- For each concept \(C_k\), split sanitized documents into \(G\) groups (“users”) of size \(m\) documents each (e.g., \(G=50, m=200\)).
- For each group \(\mathcal{G}\), compute the fingerprint \(\tilde v(\mathcal{G})\) from embedding differences.
- Predict \(\hat k\) by template matching against \(\widetilde{\mathrm{diag}}(\Sigma_{C_k})\).

**Baseline Ladder (REQUIRED):**
- **Chance**: random guess (accuracy = \(1/K\)).
- **Isotropic noise (B)**: concept-agnostic trace-matched noise.
- **Concept-aware anisotropic (A)**: SPARSE Mahalanobis mechanism.
- **Ours (C)**: covariance smoothing.

**Resource Estimate**:
- **Compute budget**: ≤ 50 GPU-hours (expected much lower; embedding extraction dominates).
- **GPU memory**: single A100 80GB is sufficient.
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Concept-choice inference | Predict which concept \(C_k\) was selected from variance fingerprints | accuracy, macro-F1 | synthetic grouping | constructed from PII-Masking-300K | custom (few lines) |
| STS12 utility | Semantic textual similarity benchmark | Pearson correlation | test | https://github.com/embeddings-benchmark/mteb (MTEB: Massive Text Embedding Benchmark) | MTEB eval |
| Token-presence privacy (analysis) | Predict whether any concept token occurs in the input, from sanitized embeddings (multi-label classification) | AUC / accuracy | held-out D+/D- | derived from PII-Masking-300K | simple MLP |

### Main Results

#### Results Table

(All values TBD; to be filled by verification runs.)

| Method | Noise covariance | Concept-ID Acc (mean±std) | STS12 Pearson (mean±std) | Token-presence AUC (mean±std) | Source | Notes |
|---|---|---:|---:|---:|---|---|
| Chance | n/a | 0.20 | n/a | n/a | analytic | K=5 |
| Isotropic (B) | \(\Sigma=I\) | TBD | TBD | TBD | this work | Should be near chance |
| SPARSE (A) | \(\Sigma=\Sigma_{C_k}\) | TBD | TBD | TBD | this work | Expected to leak concept |
| Smoothing (C) | \(\Sigma=(1-\lambda)\Sigma_{C_k}+\lambda I\) | TBD | TBD | TBD | this work | Expected to reduce leak |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| λ sweep (analysis-only) | vary \(\lambda\in\{0,0.1,0.2,0.5\}\) | identify anisotropy threshold where concept-ID drops |
| Random anisotropy (analysis-only) | random orthogonal basis with same eigenvalues | tests whether leakage is generic to anisotropy |

### Experimental Rigor

- **Seeds**: run each main condition with ≥3 seeds (e.g., `seeds=[42, 123, 456]`) affecting mask initialization and noise sampling; report mean±std.
- **Key confounders and controls**:
  - *Noise power mismatch*: enforce trace normalization so overall noise power is comparable across A/B/C.
  - *Artifact of fingerprint estimator*: validate isotropic control stays near chance.
  - *Data leakage in concept mining*: ensure concept lists are disjoint where possible (e.g., exclude country names that are also city names).

---

## Success Criteria

**Hypothesis** (directional):
- Concept-aware anisotropic noise (A) yields concept-identification accuracy well above chance, while isotropic (B) stays near chance.
- Covariance smoothing (C) reduces concept-identification accuracy substantially with small utility loss.

**Decision Rule** (concrete):
- **Proceed (leak confirmed)**: With \(K=5\), if (A) achieves **≥ 0.50** concept-ID accuracy and (B) is **≤ 0.25** (near chance) across ≥3 seeds.
- **Pivot (leak marginal)**: If (A) is in **[0.30, 0.50)**, report sample-complexity curves vs group size \(m\); consider promoting random-anisotropy baseline or testing more distinct concepts.
- **Refute (no leak)**: If (A) < 0.30 or if (A) ≈ (B), conclude that SPARSE-style masks are not distinguishable under this threat model at practical sample sizes.
- **Mitigation success**: If (C) reduces concept-ID accuracy to **≤ 0.30** while decreasing STS12 Pearson by **≤ 0.02 absolute** relative to (A).

---

## Impact Statement

If concept-aware embedding sanitization leaks privacy-mode choice under multi-release access, deployers of RAG/vector-DB systems should treat the privacy configuration as sensitive and avoid exposing concept-specific anisotropy without additional safeguards. If covariance smoothing mitigates the leak with minimal utility loss, it provides a simple deployment modification that reduces profiling risk while preserving the main benefits of concept-aware sanitization.

---

## References

- [Concept-Aware Privacy Mechanisms for Defending Embedding Inversion Attacks](./references/CONCEPT-AWARE-PRIVACY-MECHANISMS-FOR-DEFENDING-EMBEDDING-INVERSION-ATTACKS/meta/meta_info.txt) - Tsai et al., 2026
- [Sanitizing Sentence Embeddings (and Labels) for Local Differential Privacy](https://dl.acm.org/doi/10.1145/3543507.3583512) - Du et al., 2023
- [Privacy-and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations](https://arxiv.org/abs/1910.11947) - Feyisetan et al., 2020
- [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816) - Morris et al., 2023
- [Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/abs/2402.12784) - Zhuang et al., 2024
- [Sentence embedding leaks more information than you expect: Generative embedding inversion attack](https://arxiv.org/abs/2305.03010) - Li et al., 2023
- [Information Leakage in Embedding Models](https://dl.acm.org/doi/10.1145/3372297.3417880) - Song & Raghunathan, 2020
- [Privacy-preserving neural representations of text](https://arxiv.org/abs/1809.01454) - Coavoux et al., 2018
- [Split-and-Denoise: Protecting LLM Inference with Local Differential Privacy](https://arxiv.org/abs/2310.09130) - (authors), 2023
- [OSNIP: Breaking the Privacy-Utility-Efficiency Trilemma in LLM Inference via Obfuscated Semantic Null Space](https://arxiv.org/abs/2601.22752) - (authors), 2026
- [RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response](./references/RAPPOR-Randomized-Aggregatable-Privacy-Preserving-Ordinal-Response/meta/meta_info.txt) - Erlingsson et al., 2014
- [Pool Inference Attacks on Local Differential Privacy](https://www.usenix.org/conference/usenixsecurity22/presentation/gadotti) - Gadotti et al., 2022
- [Inferential Privacy Guarantees for Differentially Private Mechanisms](https://arxiv.org/abs/1603.01508) - Ghosh & Kleinberg, 2017
- [Group and Attack: Auditing Differential Privacy](https://files.sri.inf.ethz.ch/website/papers/ccs23-groupattack.pdf) - Lokna et al., 2023
- [What Are the Chances? Explaining the Epsilon Parameter in Differential Privacy](https://www.usenix.org/system/files/usenixsecurity23-nanayakkara.pdf) - Nanayakkara et al., 2023
- [Circumventing Concept Erasure Methods For Text-To-Image Generative Models](https://arxiv.org/abs/2308.01508) - Pham et al., 2024
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) - Dwork & Roth, 2014
- [Differential Privacy](https://dl.acm.org/doi/10.1145/3085504) - (survey), 2017
- [Local Differential Privacy: A Survey](https://arxiv.org/abs/2008.05187) - (survey), 2020
- [Rényi Differential Privacy](https://arxiv.org/abs/1702.07476) - Mironov, 2017
