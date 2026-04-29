# untitled

# Distance-Hiding Fingerprints for Text Embeddings via Secure SimHash

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Modern LLM products increasingly store and share *text embeddings* (dense vectors) for tasks like semantic search, retrieval-augmented generation, duplicate detection, and security telemetry (e.g., sharing indicators of prompt-injection attacks across services). In many settings, raw text cannot be shared across organizational or regulatory boundaries, so systems instead share a compact representation such as a vector embedding or a binary “fingerprint”.

However, a growing literature shows that embeddings can leak sensitive information. Dense embeddings can enable **embedding inversion** (reconstructing input text or attributes) and can also enable **distance leakage**, where an attacker learns relationships between user queries by estimating pairwise similarities. Distance leakage is especially relevant when a system releases similarity-preserving fingerprints (e.g., locality-sensitive hashes), because attackers can estimate distances from observed collisions.

Binary fingerprints are attractive because they are small, fast to search, and appear “less informative” than dense vectors. Recent systems such as **BinaryShield** show that sign-quantized embeddings plus randomized response can support practical cross-service threat-intelligence sharing. But binary fingerprints can still preserve enough geometry to allow an attacker to infer pairwise distances and potentially triangulate sensitive attributes.

### The Problem

A common approach to produce binary fingerprints is to apply a locality-sensitive hashing (LSH) scheme such as **SimHash** (signed random projections) to an embedding. LSH is designed so that collision probability is a monotone function of similarity. This property is necessary for near-neighbor search, but it also creates a privacy risk: if an adversary can observe many fingerprints, they can estimate collision rates (or Hamming distances) and invert the monotone curve to estimate similarities.

Riazi et al. describe a concrete version of this risk as a **triangulation attack**: given public LSH codes, an attacker estimates distances from match rates and then reconstructs attributes using geometric inference (including an alternating-projections procedure) **even when the attacker never sees the original vectors**. This suggests that “hashing” does not automatically imply privacy.

Existing practical defenses for fingerprint sharing in LLM systems primarily use **randomized response** (local differential privacy) on bits (e.g., BinaryShield). This is simple and has formal privacy guarantees, but it may not be the best trade-off: randomized response preserves a measurable slope in the collision curve, so distance estimation can remain possible unless noise is strong enough to harm utility.

This proposal asks whether we can use a *problem-specific* transformation that targets the distance-estimation failure mode directly: make collision probabilities effectively flat (≈0.5) for non-neighbors, while keeping collisions high for true near-neighbors.

### Key Insight and Hypothesis

**Key insight**: In standard SimHash, the collision probability for a single bit is approximately `p = 1 - θ/π`, where θ is the angle between two vectors (equivalently, a monotone function of cosine similarity). Because `p` changes smoothly with similarity, an attacker can estimate similarity from an `L`-bit match rate.

**Secure SimHash** (Riazi et al.) modifies this by composing **k independent SimHash bits** with a **universal 1-bit hash** on the k-tuple. The resulting collision probability becomes:

- `P_sec = 1/2 + 1/2 · p^k`.

For non-neighbors, `p` is near 0.5, so `p^k` shrinks quickly and `P_sec ≈ 0.5`, making finite-sample similarity estimation unreliable. For near-neighbors where `p` is close to 1, `p^k` stays large, preserving high collision probability.

**Hypothesis**: When applied to modern sentence embeddings, Secure SimHash yields binary fingerprints that (i) maintain near-duplicate retrieval accuracy comparable to standard bit-noising baselines, while (ii) reducing **non-neighbor similarity leakage** to near-random levels for a strong attacker.

Why we could be wrong:
- The flattening may destroy too much structure for “moderate similarity” pairs, reducing retrieval quality for paraphrases.
- Because hash hyperplanes and universal-hash parameters are typically public, a strong attacker may exploit full bit patterns (not only Hamming distance) to recover more similarity signal than predicted by simple collision-curve inversion.

---

## Proposed Approach

### Overview

We propose to evaluate a **distance-hiding fingerprint transform** for text embeddings based on the Secure SimHash construction from privacy-preserving near-neighbor search. Unlike cryptographic secure computation (which hides the computation itself), this proposal focuses on the setting where fingerprints are **explicitly released/shared** across boundaries, so privacy must come from the fingerprint distribution.

Given a normalized embedding vector `e(x) ∈ R^d`, we compute an `L`-bit fingerprint `f(x) ∈ {0,1}^L`.

We compare three method families:

1. **Randomized-response SimHash (baseline)**: vanilla SimHash bits + per-bit randomized response (local DP).
2. **Gaussian-noise SimHash (baseline)**: add Gaussian noise to the embedding before hashing.
3. **Secure SimHash (ours)**: k-way composition of independent SimHash bits with a universal 1-bit hash per output bit.

### Method Details

**Step 0: Embedding extraction**
- Use an off-the-shelf sentence embedding model to map each text to `e(x) ∈ R^d` and L2-normalize.

**Step 1: Vanilla SimHash (building block)**
- Sample `L` random projection vectors `r_j ∼ N(0, I_d)`.
- For each `j`, compute base bit `b_j(x) = 1[ r_j · e(x) ≥ 0 ]`.

**Baseline A: Randomized-response SimHash**
- Flip each bit independently with probability `1 - p_keep`, where `p_keep = exp(alpha) / (exp(alpha) + 1)` (as in BinaryShield).

**Baseline B: Gaussian-noise SimHash**
- Add noise `η ∼ N(0, σ^2 I_d)` to embeddings: `e'(x) = normalize(e(x) + η)`.
- Compute vanilla SimHash bits on `e'(x)`.

**Ours: Secure SimHash (distance-hiding)**
For each output bit `j = 1..L`:
1. Sample **k** independent projection vectors `r_{j,1..k}` and compute k base bits `b_{j,t}(x) = 1[r_{j,t}·e(x) ≥ 0]`.
2. Sample a universal 1-bit hash on `{0,1}^k`, implemented as a random affine map over GF(2):
   - Sample `a_{j,0..k} ∈ {0,1}` uniformly with the constraint that at least one of `a_{j,1..k}=1`.
   - Output `f_j(x) = a_{j,0} XOR (⊕_{t: a_{j,t}=1} b_{j,t}(x))`.

This is a practical instantiation of the “universal hash” in the secure LSH formalization. Under random hash selection, it yields `P_sec ≈ 1/2 + 1/2·p^k` where `p` is the collision probability of the underlying SimHash bit.

**What this proposal explicitly does NOT implement**
- The two-server garbled-circuit seed-hiding protocol from PPLSI. We evaluate the *fingerprint transform* itself, which is the relevant component for settings where fingerprints are shared publicly (the system designer cannot assume the hash seeds are secret).

### Key Innovations

1. **Distance-leakage-first evaluation for embedding fingerprints**: We evaluate fingerprint privacy as a *similarity discrimination* capability of a learned attacker (AUC), rather than only measuring reconstruction or relying on DP parameters.
2. **Domain transfer of secure LSH to modern text embeddings**: Secure SimHash was proposed and evaluated in near-neighbor search for tabular data; we test whether its privacy–utility behavior persists for transformer-based sentence embeddings.
3. **Decision-changing output**: The result is a concrete recommendation: “use Secure SimHash with k≈4 for cross-boundary fingerprint sharing” or “do not use it; randomized response/noise dominate”.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) **embedding privacy** (embedding inversion attacks and defenses), (ii) **local differential privacy** mechanisms applied to compact representations (randomized response), and (iii) **locality-sensitive hashing** for efficient similarity search. Classic LSH methods (Indyk & Motwani; Charikar) are designed to preserve similarity in Hamming space, but this property can be exploited for distance estimation. Riazi et al. showed that the resulting distance leakage enables triangulation-style reconstruction of attributes, motivating “secure” transforms that explicitly break distance estimation for non-neighbors.

In parallel, systems work has explored privacy-preserving similarity search via secure computation or encrypted indices (e.g., secure SimHash protocols, SimHash-based encrypted search). These protect plaintexts from a server but often still assume that similarity signals (or thresholded matches) can be safely exposed. For cross-boundary telemetry sharing, the fingerprint itself is the released artifact, so the privacy question becomes: *what can an attacker infer from released fingerprints alone?*

### Related Papers

- **[Sub-Linear Privacy-Preserving Near-Neighbor Search](./references/Sub-Linear-Privacy-Preserving-Near-Neighbor-Search/meta/meta_info.txt)**: Introduces triangulation attacks on vanilla LSH and proposes the Secure LSH / Secure SimHash probabilistic transform used here.
- **[Cross-Service Threat Intelligence in LLM Services using Privacy-Preserving Fingerprints](./references/Cross-Service-Threat-Intelligence-in-LLM-Services-using-Privacy-Preserving-Fingerprints/meta/meta_info.txt)**: Proposes BinaryShield, a practical system using sign-quantization and randomized response for cross-service sharing of attack fingerprints.
- **[Similarity Estimation Techniques from Rounding Algorithms](https://arxiv.org/abs/cs/0201015)**: Introduces SimHash (signed random projections) for cosine similarity estimation.
- **[Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](https://dl.acm.org/doi/10.1145/956750.956758)**: Establishes foundational LSH constructions for Euclidean distances.
- **[Similarity Search in High Dimensions via Hashing](https://dl.acm.org/doi/10.1145/276698.276876)**: Classic LSH framework (Indyk & Motwani) underlying approximate nearest-neighbor search.
- **[Detecting Near-Duplicates for Web Crawling](https://dl.acm.org/doi/10.1145/1242572.1242592)**: Early large-scale near-duplicate detection using SimHash.
- **[Universal Classes of Hash Functions](https://dl.acm.org/doi/10.1145/800105.803400)**: Introduces universal hashing (Carter & Wegman), used in secure LSH constructions.
- **[Fast Near Neighbor Search in High-Dimensional Binary Data](https://link.springer.com/chapter/10.1007/978-3-642-33460-3_33)**: Discusses binary hashing/search and practical use of parity/MSB rehashing for 1-bit schemes.
- **[Differential Privacy](https://dl.acm.org/doi/10.1145/1031916.1031943)**: Establishes the differential privacy framework.
- **[Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias](https://www.jstor.org/stable/2283137)**: Introduces randomized response, a core local DP mechanism.
- **[RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response](https://arxiv.org/abs/1407.6981)**: Practical local DP system using randomized response variants.
- **[Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)**: DP-SGD for model training; representative of DP in ML pipelines where embeddings are produced.
- **[Differential Privacy with Random Projections and Sign Random Projections](https://arxiv.org/abs/2306.01751)**: Studies DP properties of (signed) random projections, relevant to noise-baseline design.
- **[LDP-Feat: Image Features with Local Differential Privacy](https://openaccess.thecvf.com/content/ICCV2023/papers/Pittaluga_LDP-Feat_Image_Features_with_Local_Differential_Privacy_ICCV_2023_paper.pdf)**: Uses local DP mechanisms for compact feature sharing; analogous to our fingerprint-sharing setting.
- **[Secure Similar Document Detection with Simhash](https://sbakiras.github.io/papers/sdm13.pdf)**: Secure computation protocol for SimHash-based similarity without revealing documents.
- **[Privacy-Preserving Smart Similarity Search Based on Simhash over Encrypted Data in Cloud Computing](https://khu.elsevierpure.com/en/publications/privacy-preserving-smart-similarity-search-based-on-simhash-over--2/)**: Uses SimHash for encrypted similarity search in cloud settings.
- **[ALGEN: Few-shot Inversion Attacks on Textual Embeddings via Alignment and Generation](https://arxiv.org/abs/2502.11308)**: Demonstrates practical embedding inversion with few samples via embedding-space alignment.
- **[Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings](https://arxiv.org/abs/2602.01757)**: Zero-training inversion attacks on textual embeddings, highlighting risks of releasing embedding-like artifacts.
- **[OSNIP: Breaking the Privacy-Utility-Efficiency Trilemma in LLM Inference via Obfuscated Semantic Null Space](https://arxiv.org/abs/2601.22752)**: Defense strategy for embedding/activation leakage; motivates stronger privacy mechanisms beyond naive obfuscation.
- **[Concept-Aware Privacy Mechanisms for Defending Embedding Inversion Attacks](https://arxiv.org/abs/2602.07090)**: Proposes defenses against embedding inversion with semantic constraints.
- **[BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://openreview.net/forum?id=wCu6T5xFjeJ)**: Benchmark suite containing the Quora retrieval task used in our experiments.
- **[MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)**: Provides standardized evaluation for text embeddings and retrieval tasks.
- **[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)**: A widely used approach for producing sentence embeddings.
- **[MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)**: Underlies mpnet-based sentence embedding models used in evaluation.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Similarity-preserving fingerprints | Release binary codes that preserve cosine/Jaccard similarity for fast search | Charikar (SimHash), Manku et al., Indyk & Motwani | Near-duplicate retrieval, BEIR | Distance leakage via collision curves |
| Local DP on fingerprints | Flip bits via randomized response to enforce ε-LDP | BinaryShield, RAPPOR, Warner (RR) | Threat correlation / retrieval accuracy vs privacy budget | Can still leak distances unless noise is large |
| Noise-before-hash (DP-style obfuscation) | Add noise to embeddings then hash | DP random projections / SignRP | Retrieval vs noise scale | Utility degrades quickly at strong privacy |
| Distance-hiding transforms (secure LSH) | Shape collision curve to be flat for non-neighbors | Riazi et al. (Secure LSH / Secure SimHash) | PP-NNS; here: embedding fingerprints | Not widely validated on modern text embeddings |
| Secure computation / encrypted similarity | Keep documents private while computing similarity | Secure SimHash protocols; SimHash encrypted search | Encrypted search throughput / correctness | Often assumes similarity signal itself is safe to expose |

### Closest Prior Work

1. **Riazi et al. (Secure LSH / Secure SimHash)**: Provides the core construction and theory for resisting triangulation by flattening non-neighbor collision probability, but does not test modern sentence embeddings or a threat model where the fingerprint itself is widely shared.

2. **BinaryShield**: Demonstrates a practical system for cross-service sharing of privacy-preserving fingerprints using sign quantization and randomized response. It does not test secure LSH transforms, and it evaluates privacy mainly via DP parameterization and reconstruction intuition rather than explicit distance-leakage attackers.

3. **Secure Similar Document Detection with Simhash**: Uses secure computation to compare SimHash fingerprints without revealing documents, but it does not address the setting where fingerprints are openly shared and thus vulnerable to distance-leakage inference.

4. **SimHash-based encrypted similarity search (Fu et al.)**: Focuses on encrypted cloud retrieval and does not study leakage from public fingerprints.

5. **DP with sign random projections**: Studies DP properties of sign projections/noise, but does not target the triangulation failure mode or near-neighbor retrieval constraints.

**Novelty Kill Search Summary:** Searched for combinations of “secure simhash + text embeddings”, “secure LSH + vector database”, “triangulation attack + SimHash”, and checked for systems combining the PPLSI secure SimHash transform with modern sentence embedding benchmarks. No prior work matching “PPLSI Secure SimHash + modern text embeddings + distance-leakage (attacker AUC) evaluation” was found as of 2026-02-19. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Riazi et al. (Secure SimHash) | Distance-hiding transform for LSH; resists triangulation on tabular data | No evaluation on modern text embeddings or public fingerprint sharing | Apply Secure SimHash to sentence embeddings and evaluate distance leakage directly | If distance leakage is generic to LSH, the transform should transfer; if it fails, we learn boundary conditions |
| BinaryShield | Practical cross-service sharing with randomized response | Does not test distance-hiding transforms; privacy evaluation not leakage-first | Compare Secure SimHash against RR baselines on the same retrieval task | Secure SimHash targets the specific distance-estimation failure mode |
| Secure SimHash 2PC | Securely computes similarity without revealing docs | Not applicable when fingerprints are shared publicly | Study fingerprint distribution privacy directly | Provides guidance for public telemetry sharing, not just secure computation |
| SimHash encrypted search | Search encrypted docs using SimHash indices | Does not address inference from shared fingerprints | Add explicit leakage evaluation under learned attackers | Captures privacy risk in cross-boundary sharing settings |

---

## Experiments

### Experimental Setup

**Goal**: Compare privacy–utility trade-offs for released binary fingerprints derived from modern sentence embeddings.

**Benchmark task**: Duplicate question retrieval.

**Dataset**:
- **BEIR Quora** (duplicate question retrieval; ~523k corpus questions and 10k queries) from the BEIR benchmark.

**Embedding model**:
- **all-mpnet-base-v2** (a Sentence-Transformers model producing 768-d embeddings).

**Fingerprints**:
- Fixed fingerprint length **L = 256 bits** for all methods (scope control). We sweep privacy knobs (alpha, sigma, k).

**Baseline Ladder (REQUIRED):**
- Trivial: random 256-bit codes (sanity check).
- Vanilla SimHash (no privacy).
- Randomized-response SimHash (baseline; BinaryShield-style bit flipping).
- Gaussian-noise SimHash (baseline; noise before hashing).
- **Secure SimHash (ours)**.
- Reference (non-private upper bound): dense-embedding cosine retrieval.

**Privacy evaluation (attacker models)**:
- Weak attacker: isotonic regression on Hamming similarity.
- Strong attacker: small MLP on XOR bits `f(x) XOR f(y)`.

**Privacy metric (distance leakage proxy)**:
- Similarity discrimination AUC: predict whether `cos(e(x), e(y)) ≥ s_thr` for thresholds `s_thr ∈ {0.3, 0.5}`.

**Utility metric**:
- Retrieval **Recall@10** on BEIR Quora: fraction of queries whose relevant duplicate appears in the top 10 results when ranked by Hamming distance over fingerprints.

**Parameter sweeps (small grids)**:
- Randomized response: `alpha ∈ {1.0, 1.5, 2.0, 2.5}`.
- Gaussian noise: `sigma ∈ {0.0, 0.25, 0.5, 1.0}`.
- Secure SimHash: `k ∈ {1, 2, 4, 8}` (k=1 corresponds to vanilla SimHash).

**Ablation (compute-matched control)**:
- Compute-matched SimHash: use `L/k` SimHash bits so the number of hyperplane projections matches Secure SimHash with parameter k.

**Seeds & variance plan**:
- Non-deterministic components: random hyperplanes / hash coefficients and strong-attacker training.
- Run ≥3 seeds: `seeds = [42, 123, 456]`. Report mean ± std for AUC and Recall@10.

**Implementation notes**:
- Use FAISS binary indices (`IndexBinaryFlat` or IVF) for Hamming retrieval.
- Use BEIR evaluation scripts for Recall@k (or an equivalent open-source evaluator).

**Resource Estimate**:
- **Compute budget**: ≤ 50 A100 GPU-hours total.
  - Embedding inference for ~533k texts (corpus+queries) with a 110M-class encoder is the main cost; fingerprint generation and attacker training are lightweight.
  - If GPU inference is unavailable, embeddings can be computed on CPU with higher wall-clock time but still within the project’s automation constraints.
- **GPU memory**: ≤ 16GB (single A100 80GB is sufficient).
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| BEIR Quora | Duplicate-question retrieval: retrieve a known duplicate question from a large corpus | Recall@10, nDCG@10 (optional) | test (or dev) | https://huggingface.co/datasets/BeIR/quora | https://github.com/beir-cellar/beir |

### Main Results

(All results are **TBD**; verification will compute them.)

| Method | Base Model | Benchmark | Utility: Recall@10 (mean±std) | Privacy: AUC@0.5 (mean±std) | Privacy: AUC@0.3 (mean±std) | Source | Notes |
|---|---|---|---|---|---|---|---|
| Random codes | all-mpnet-base-v2 | BEIR Quora | TBD | ~0.50 | ~0.50 | - | Sanity check |
| Vanilla SimHash (k=1) | all-mpnet-base-v2 | BEIR Quora | TBD | TBD | TBD | - | No privacy |
| RR-SimHash (alpha tuned) | all-mpnet-base-v2 | BEIR Quora | TBD | TBD | TBD | - | Baseline; tune alpha |
| Noise-SimHash (sigma tuned) | all-mpnet-base-v2 | BEIR Quora | TBD | TBD | TBD | - | Baseline; tune sigma |
| **Secure SimHash (k tuned)** | all-mpnet-base-v2 | BEIR Quora | TBD | TBD | TBD | - | To be verified |
| Dense cosine retrieval | all-mpnet-base-v2 | BEIR Quora | TBD | N/A | N/A | - | Non-private upper bound |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Secure SimHash (k) | Full method | Best privacy at matched utility |
| Compute-matched SimHash (L/k bits) | Same projection compute as Secure SimHash | Tests whether gains are more than “using fewer bits” |
| Secure SimHash with public hash params | Hash params are public (default) | If leakage remains low, method does not rely on secret seeds |

### Experimental Rigor

**Confounders and controls**:
- **“Just fewer bits” confound**: controlled by compute-matched SimHash ablation.
- **Attacker weakness**: controlled by including a strong attacker (MLP) operating on XOR bit patterns, not only Hamming distance.
- **Distribution shift for attacker training**: attacker uses auxiliary pairs sampled from the same corpus distribution with a held-out split.

---

## Success Criteria

**Hypothesis** (directional): Secure SimHash can achieve near-duplicate retrieval quality comparable to randomized response and Gaussian-noise baselines while reducing non-neighbor similarity discrimination AUC toward random guessing.

**Decision Rule** (concrete — when to stop):
- **Proceed**: At a utility operating point where Recall@10 is within 2% absolute of the best baseline setting, Secure SimHash achieves strong-attacker AUC ≤ 0.60 at `s_thr=0.5` and ≤ 0.65 at `s_thr=0.3`, while the best baseline settings have AUC ≥ 0.75 at the same thresholds (clear privacy gap).
- **Pivot**: If Secure SimHash only wins at very low utility (e.g., Recall@10 drops >2%), try smaller k (k=2) and/or longer fingerprints (L=512) as a targeted adjustment.
- **Refute**: If Secure SimHash cannot reduce strong-attacker AUC below 0.65 at `s_thr=0.5` without losing >2% Recall@10, or if RR/Noise baselines achieve similar AUC at the same utility (Secure SimHash is dominated).

---

## Impact Statement

If successful, this work would provide a practical, implementable recommendation for teams that need to share similarity-searchable telemetry (e.g., cross-service prompt-injection fingerprints, deduplication signals, or privacy-preserving retrieval indices) without leaking pairwise distances between user texts. If unsuccessful, it would still be decision-changing by showing that secure LSH transforms do not transfer to modern text embedding distributions (or that attackers can recover similarity anyway), implying that practitioners should rely on stronger privacy mechanisms (e.g., local DP with larger utility cost) or avoid sharing similarity-searchable fingerprints.

---

## References

- [Sub-Linear Privacy-Preserving Near-Neighbor Search](./references/Sub-Linear-Privacy-Preserving-Near-Neighbor-Search/meta/meta_info.txt) - Riazi et al., 2016/2019
- [Cross-Service Threat Intelligence in LLM Services using Privacy-Preserving Fingerprints](./references/Cross-Service-Threat-Intelligence-in-LLM-Services-using-Privacy-Preserving-Fingerprints/meta/meta_info.txt) - Gill et al., 2025
- [Similarity Estimation Techniques from Rounding Algorithms](https://arxiv.org/abs/cs/0201015) - Charikar, 2002
- [Similarity Search in High Dimensions via Hashing](https://dl.acm.org/doi/10.1145/276698.276876) - Indyk & Motwani, 1998
- [Detecting Near-Duplicates for Web Crawling](https://dl.acm.org/doi/10.1145/1242572.1242592) - Manku et al., 2007
- [Universal Classes of Hash Functions](https://dl.acm.org/doi/10.1145/800105.803400) - Carter & Wegman, 1977
- [Fast Near Neighbor Search in High-Dimensional Binary Data](https://link.springer.com/chapter/10.1007/978-3-642-33460-3_33) - Shrivastava & Li, 2012
- [Differential Privacy](https://dl.acm.org/doi/10.1145/1031916.1031943) - Dwork, 2006
- [Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias](https://www.jstor.org/stable/2283137) - Warner, 1965
- [RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response](https://arxiv.org/abs/1407.6981) - Erlingsson et al., 2014
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) - Abadi et al., 2016
- [Differential Privacy with Random Projections and Sign Random Projections](https://arxiv.org/abs/2306.01751) - (DP random projections), 2023
- [LDP-Feat: Image Features with Local Differential Privacy](https://openaccess.thecvf.com/content/ICCV2023/papers/Pittaluga_LDP-Feat_Image_Features_with_Local_Differential_Privacy_ICCV_2023_paper.pdf) - Pittaluga et al., 2023
- [Secure Similar Document Detection with Simhash](https://sbakiras.github.io/papers/sdm13.pdf) - Buyrukbilen & Bakiras, 2013
- [Privacy-Preserving Smart Similarity Search Based on Simhash over Encrypted Data in Cloud Computing](https://khu.elsevierpure.com/en/publications/privacy-preserving-smart-similarity-search-based-on-simhash-over--2/) - Fu et al., 2015
- [ALGEN: Few-shot Inversion Attacks on Textual Embeddings via Alignment and Generation](https://arxiv.org/abs/2502.11308) - (embedding inversion), 2025
- [Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings](https://arxiv.org/abs/2602.01757) - 2026
- [OSNIP: Breaking the Privacy-Utility-Efficiency Trilemma in LLM Inference via Obfuscated Semantic Null Space](https://arxiv.org/abs/2601.22752) - 2026
- [Concept-Aware Privacy Mechanisms for Defending Embedding Inversion Attacks](https://arxiv.org/abs/2602.07090) - 2026
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://openreview.net/forum?id=wCu6T5xFjeJ) - Thakur et al., 2021
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) - Muennighoff et al., 2022
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) - Reimers & Gurevych, 2019
- [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297) - Song et al., 2020
