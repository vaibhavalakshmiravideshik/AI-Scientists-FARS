# untitled

# Utility-Matched Evaluation of 8-bit Embedding Quantization Against Training-Free Text Inversion

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Dense text embeddings are widely stored in third-party vector databases for retrieval-augmented generation (RAG) and semantic search. In many deployments the source documents are sensitive (medical, legal, corporate), while embeddings are treated as a safer artifact to store or share.

This assumption is increasingly questionable. Vec2Text shows that short texts can be reconstructed from black-box sentence embeddings by training an embedding-specific decoder (**[Text Embeddings Reveal (Almost) As Much As Text](./references/Text-Embeddings-Reveal-(Almost)-As-Much-As-Text/meta/meta_info.txt)**). Newer attacks are more deployment-relevant because they remove the need to train a decoder for each encoder: ZSInvert is a **training-free** inversion method that iteratively queries the encoder and uses an LLM to generate text that maximizes cosine similarity to a target embedding (**[Universal Zero-shot Embedding Inversion](./references/Universal-Zero-shot-Embedding-Inversion/meta/meta_info.txt)**). Zero2Text extends training-free inversion to cross-domain settings via online alignment (**[Zero2Text](./references/Zero2Text-Zero-Training-Cross-Domain-Inversion-Attacks-on-Textual-Embeddings/meta/meta_info.txt)**).

Embedding **quantization** (e.g., per-dimension int8 scalar quantization) is commonly used for retrieval efficiency. Several papers argue quantization is also a lightweight privacy defense because it preserves retrieval utility while greatly reducing lexical reconstruction scores for trained attacks like Vec2Text (**[Rethinking the Privacy of Text Embeddings](./references/Rethinking-the-Privacy-of-Text-Embeddings-A-Reproducibility-Study/meta/meta_info.txt)**; **[Understanding and Mitigating the Threat of Vec2Text](./references/Understanding-and-Mitigating-the-Threat-of-Vec2Text-to-Dense-Retrieval-Systems/meta/meta_info.txt)**).

### The Problem

> **At a fixed retrieval-utility operating point, does 8-bit embedding quantization reduce leakage under modern training-free inversion attacks, or does it mainly degrade lexical metrics while leaving recovery of sensitive attributes largely unchanged?**

This question matters because practitioners already quantize embeddings for cost and latency. If quantization genuinely improves privacy “for free”, it should become a default recommendation. If it does not, then quantization should be treated as compression, not privacy.

### Key Insight and Hypothesis

**Key insight:** Quantizers are optimized to preserve the similarity geometry needed for retrieval. ZSInvert’s search objective is also based on cosine similarity. Therefore, at the same retrieval quality, quantization may not reduce the attacker’s optimization signal compared to a utility-matched random perturbation.

**Hypothesis:** At matched retrieval utility, 8-bit quantization does **not** materially reduce a controlled secret-attribute recovery rate (Canary-EM) under ZSInvert compared to additive Gaussian noise tuned to the same retrieval utility.

**Why we could be wrong:** Quantization is deterministic and non-smooth, so it may disrupt ZSInvert’s similarity-guided search more than Gaussian noise. Conversely, quantization might be worse than noise if its structured error preserves more semantic directions.

---

## Proposed Approach

### Overview

We evaluate **three storage conditions** for corpus embeddings:

1. **Raw**: store float embeddings.
2. **Quantized int8**: store absmax int8-quantized embeddings and dequantize for cosine scoring.
3. **Utility-matched Gaussian**: add \(\epsilon\sim\mathcal{N}(0,\sigma^2 I)\) to stored embeddings, tuning \(\sigma\) so retrieval **nDCG@10** (Normalized Discounted Cumulative Gain at rank 10; higher is better) matches the quantized condition.

For each defended embedding, we run ZSInvert and measure recovery of a controlled secret attribute.

### Method Details

#### Defense: absmax int8 (scalar) quantization

For each embedding \(e\), compute \(s=\max_i|e_i|/127\), store \(q=\mathrm{round}(e/s)\in[-127,127]^d\), and use \(\hat e=sq\) for cosine similarity.

#### Utility matching

We tune \(\sigma\) on a validation split to match the quantized condition’s test nDCG@10 within ±0.02 absolute; otherwise the run is treated as confounded and \(\sigma\) is re-tuned.

#### Privacy metric: secret-attribute (“canary”) recovery

For \(N=500\) corpus documents, prepend one natural-language sentence encoding a secret attribute:

- Template: `After years of work, she moved back to the <ADJ> <NOUN> harbor.`
- \(<ADJ>,<NOUN>\) sampled from fixed vocabularies (e.g., 512×512 pairs).

**Canary-EM (attack success rate; lower is safer):** fraction of inversions whose reconstructed text contains the exact \(<ADJ>,<NOUN>\) pair in order (case-insensitive).

#### Attack: ZSInvert

We use the full ZSInvert pipeline (including its Stage-3 correction model), because it is training-free, uses only black-box encoder queries, and is robust to utility-preserving Gaussian noise (**[Universal Zero-shot Embedding Inversion](./references/Universal-Zero-shot-Embedding-Inversion/meta/meta_info.txt)**). Suggested settings: `beam_width=30`, `top_k=30`, `max_steps=32`, iterations 6–9.

### Key Innovations

1. **Utility-matched evaluation** of quantization vs a random-noise baseline under a training-free attacker.
2. **Secret-attribute metric (Canary-EM)** that is paraphrase-robust and automatically checkable.
3. **Equivalence-style decision rule** that yields an actionable conclusion either way.

---

## Related Work

### Field Overview

Prior embedding inversion work largely evaluates trained decoders (Vec2Text) and reports lexical metrics (BLEU/F1/exact match). More recent training-free attacks (ZSInvert, Zero2Text) are a stronger threat model because they adapt online by querying the encoder. In parallel, defenses like quantization and noise are often evaluated only by lexical reconstruction, which can miss semantic/attribute leakage.

### Related Papers

- **[Text Embeddings Reveal (Almost) As Much As Text](./references/Text-Embeddings-Reveal-(Almost)-As-Much-As-Text/meta/meta_info.txt)**: Vec2Text trained inverter for short-text reconstruction.
- **[Understanding and Mitigating the Threat of Vec2Text](./references/Understanding-and-Mitigating-the-Threat-of-Vec2Text-to-Dense-Retrieval-Systems/meta/meta_info.txt)**: PQ_768 makes Vec2Text exact-match 0 with retrieval unchanged.
- **[Rethinking the Privacy of Text Embeddings](./references/Rethinking-the-Privacy-of-Text-Embeddings-A-Reproducibility-Study/meta/meta_info.txt)**: Absmax/zeropoint int8 lowers Vec2Text BLEU with stable BEIR nDCG@10.
- **[Universal Zero-shot Embedding Inversion](./references/Universal-Zero-shot-Embedding-Inversion/meta/meta_info.txt)**: ZSInvert training-free inversion, robust to utility-preserving noise.
- **[Zero2Text](./references/Zero2Text-Zero-Training-Cross-Domain-Inversion-Attacks-on-Textual-Embeddings/meta/meta_info.txt)**: Training-free cross-domain inversion via online alignment.
- **[BeamClean](./references/BeamClean-Language-Aware-Embedding-Reconstruction/meta/meta_info.txt)**: Language-aware inversion of noisy token embeddings; NN baselines can be misleading.
- **[Concept-Aware Privacy Mechanisms](./references/Concept-Aware-Privacy-Mechanisms-for-Defending-Embedding-Inversion-Attacks/meta/meta_info.txt)**: SPARSE learns sensitive-dimension masks + anisotropic DP noise.
- **[OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma/meta/meta_info.txt)**: Key-conditioned “semantic null space” obfuscation for private inference.
- **[Stained Glass Transform](./references/Stained-Glass-Transform/meta/meta_info.txt)**: Learned stochastic embedding obfuscation minimizing mutual information.
- **[Split-and-Denoise](https://arxiv.org/abs/2310.09130)**: Split inference with local-DP noise and denoising; uses inversion ASR.
- **[DP Split Inference via Stochastic Quantization + Soft Prompt](https://arxiv.org/abs/2602.11513)**: Quantization-based split-inference defense with inversion evaluation.
- **[Prompt Inversion Attack against Collaborative Inference](https://arxiv.org/abs/2503.09022)**: Prompt/activation inversion in collaborative inference settings.
- **[Depth Gives a False Sense of Privacy](https://arxiv.org/abs/2507.16372)**: Internal states invertible across depth; compression often trades off with utility.
- **[Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2401.01948)**: Theory suggesting strong invertibility of LM representations.
- **[GEIA](https://aclanthology.org/2023.findings-acl.872/)**: Attribute leakage from sentence embeddings.
- **[Embedding Inversion for Multilingual LMs](https://aclanthology.org/2024.acl-long.427/)**: Multilingual inversion risk evaluation.
- **[Private Release of Text Embedding Vectors](https://aclanthology.org/2021.trustnlp-1.3/)**: DP framing for releasing embeddings.
- **[NoPeek](https://arxiv.org/abs/2008.03248)**: Representation obfuscation in split learning.
- **[Product Quantization](https://ieeexplore.ieee.org/document/5432202)**: PQ compression for ANN search.
- **[TEM](https://epubs.siam.org/doi/10.1137/1.9781611978032.99)**: Metric DP for text perturbation.

### Taxonomy

| Cluster | Core idea | Example papers | Main limitation |
|---|---|---|---|
| Trained inversion | Train a decoder on (text, embedding) pairs | Vec2Text; Seputis et al. | Data requirement; OOD fragility |
| Training-free inversion | Online search with encoder queries + LM prior | ZSInvert; Zero2Text | Defenses under-tested |
| Utility-preserving perturbations | Quantization/noise to keep retrieval usable | Zhuang et al.; Gaussian noise | Lexical-only privacy metrics |

### Closest Prior Work

- Zhuang et al. show PQ eliminates Vec2Text exact match at unchanged retrieval, but do not test training-free attacks.
- Seputis et al. show int8 reduces Vec2Text BLEU at stable BEIR nDCG@10, but do not test training-free attacks.
- ZSInvert shows Gaussian noise fails when retrieval utility is preserved, but does not evaluate quantization.

**Novelty Kill Search Summary:** Searched for “ZSInvert + quantization”, “training-free embedding inversion + product quantization”, “embedding inversion + int8/SQ8”, and OpenReview/GitHub for “embedding inversion quantization”. As of **2026-02-20**, we found no paper evaluating quantization as a defense against ZSInvert/Zero2Text-style training-free inversion.

### Comparison Table

| Prior work | Focus | Missing piece | Our change |
|---|---|---|---|
| Zhuang et al. 2024 | Quantization vs Vec2Text | No training-free attacker | Evaluate ZSInvert |
| Seputis et al. 2025 | int8 vs Vec2Text on BEIR | No training-free attacker | Utility-match vs noise |
| ZSInvert 2025 | Noise robustness | No quantization | Add quantization |

---

## Experiments

### Experimental Setup

- **Encoder**: Contriever (a dense retrieval encoder), because ZSInvert reports it is relatively easy to invert.
- **Retrieval benchmark**: one BEIR dataset (recommend SciFact for small scale) with standard qrels.
- **Inversion set**: sample \(N=500\) corpus documents; prepend canary; truncate to first 32 tokens.
- **Utility metric**: nDCG@10 computed with standard BEIR evaluation.
- **Attack**: ZSInvert (Stage 3 enabled) run against the stored (defended) corpus embedding.

**Baseline ladder (defenses):** Raw vs Quantized vs Utility-matched Gaussian + chance (random guess).

**Compute evidence:** ZSInvert reports ~10 sec/iteration on 1×A40, ~90 sec per embedding at 9 iterations, and ~10 min for correction-model training (**Computational Cost** in **[Universal Zero-shot Embedding Inversion](./references/Universal-Zero-shot-Embedding-Inversion/meta/meta_info.txt)**). Target total ≤300 A100 GPU-hours.

### Benchmarks and Metrics

| Benchmark | What it evaluates | Metrics |
|---|---|---|
| BEIR (e.g., SciFact) | Retrieval quality on a public corpus with relevance labels | nDCG@10 (higher is better) |
| Canary-500 | Secret-attribute leakage from reconstructed text | Canary-EM (lower is safer) |

### Main Results

**Published sanity-check numbers (not directly comparable to Canary-EM):**

- ZSInvert (Contriever, MS-MARCO, after correction): \(\sigma=0.01\) → F1=60.24, CosSim=81.11 (Table 3 in **[Universal Zero-shot Embedding Inversion](./references/Universal-Zero-shot-Embedding-Inversion/meta/meta_info.txt)**).
- Zhuang et al. (DPR_cls_dot, NQ): PQ_768 → top10=0.749, exact=0.000, cos=0.748 (Table 4 in **[Understanding and Mitigating the Threat of Vec2Text](./references/Understanding-and-Mitigating-the-Threat-of-Vec2Text-to-Dense-Retrieval-Systems/meta/meta_info.txt)**).
- Seputis et al. (gtr-nq-32, BEIR ArguAna): absmax int8 keeps nDCG@10=0.281 while BLEU 59.1→19.0 (Table 5 in **[Rethinking the Privacy of Text Embeddings](./references/Rethinking-the-Privacy-of-Text-Embeddings-A-Reproducibility-Study/meta/meta_info.txt)**).

**New results to produce (directly answers the claim):**

| Method | Benchmark | nDCG@10 (mean±std) | Canary-EM (mean±std) | Notes |
|---|---|---:|---:|---|
| Raw | BEIR + Canary-500 | TBD | TBD | Must satisfy attack-strength floor |
| Quantized int8 | BEIR + Canary-500 | TBD | TBD | Absmax int8 |
| Utility-matched Gaussian | BEIR + Canary-500 | TBD | TBD | \(\sigma\) tuned on val |
| Random guess | Canary-500 | — | ~0 | 1/(512×512) per doc |

### Ablation Studies

| Variant | What changes | Why |
|---|---|---|
| Stage-2 only | Disable correction model | Checks whether Stage 3 is required for canary leakage |

### Experimental Rigor

- **Seeds**: 3 seeds for ZSInvert decoding: `seeds=[42,123,456]`.
- **Fair attack budget**: hold ZSInvert hyperparameters fixed across defense conditions.
- **Utility matching**: enforce nDCG@10(quantized) and nDCG@10(noise) within 0.02.
- **Attack-strength floor**: if Raw Canary-EM <0.20, increase iterations up to 9; if still <0.20, treat the run as uninformative.
- **Main confounders and controls**:
  - Utility mismatch (control: explicit nDCG@10 tolerance).
  - Weak attack (control: attack-strength floor).
  - Memorization of the underlying corpus by the generator LM (control: canary sentence and attribute values are synthetic and inserted after dataset download).

---

## Success Criteria

**Hypothesis**: At matched retrieval utility, quantization does not materially reduce canary leakage compared to Gaussian noise.

**Decision Rule**:

- **Conclude “quantization ≈ noise”** if Canary-EM(quantized) is within **±0.07 absolute** of Canary-EM(noise) and utility matches (nDCG@10 within 0.02).
- **Conclude “quantization helps”** if Canary-EM(quantized) ≤ Canary-EM(noise) − 0.10 at matched utility.
- **Conclude “quantization is worse”** if Canary-EM(quantized) ≥ Canary-EM(noise) + 0.10 at matched utility.
- **Refute as uninformative** if Raw Canary-EM <0.20 after increasing attack iterations.

---

## Impact Statement

If quantization is no better than utility-matched noise, practitioners should stop treating int8/PQ compression as a privacy measure and instead deploy privacy-specific defenses (secret transforms, concept-aware perturbation, or secure execution). If it is meaningfully better (or worse), the result provides an immediate, deployment-facing guideline for vector-database embedding storage.

---

## References

- [Text Embeddings Reveal (Almost) As Much As Text](./references/Text-Embeddings-Reveal-(Almost)-As-Much-As-Text/meta/meta_info.txt)
- [Universal Zero-shot Embedding Inversion](./references/Universal-Zero-shot-Embedding-Inversion/meta/meta_info.txt)
- [Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings](./references/Zero2Text-Zero-Training-Cross-Domain-Inversion-Attacks-on-Textual-Embeddings/meta/meta_info.txt)
- [Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](./references/Understanding-and-Mitigating-the-Threat-of-Vec2Text-to-Dense-Retrieval-Systems/meta/meta_info.txt)
- [Rethinking the Privacy of Text Embeddings: A Reproducibility Study](./references/Rethinking-the-Privacy-of-Text-Embeddings-A-Reproducibility-Study/meta/meta_info.txt)
- [BeamClean: Language Aware Embedding Reconstruction](./references/BeamClean-Language-Aware-Embedding-Reconstruction/meta/meta_info.txt)
- [Concept-Aware Privacy Mechanisms for Defending Embedding Inversion Attacks](./references/Concept-Aware-Privacy-Mechanisms-for-Defending-Embedding-Inversion-Attacks/meta/meta_info.txt)
- [OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma/meta/meta_info.txt)
- [Stained Glass Transform](./references/Stained-Glass-Transform/meta/meta_info.txt)
- [Split-and-Denoise](https://arxiv.org/abs/2310.09130)
- [Differentially Private and Communication Efficient LLM Split Inference via Stochastic Quantization and Soft Prompt](https://arxiv.org/abs/2602.11513)
- [Prompt Inversion Attack against Collaborative Inference of Large Language Models](https://arxiv.org/abs/2503.09022)
- [Depth Gives a False Sense of Privacy](https://arxiv.org/abs/2507.16372)
- [Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2401.01948)
- [Sentence Embedding Leaks More Information Than You Expect (GEIA)](https://aclanthology.org/2023.findings-acl.872/)
- [Text Embedding Inversion Security for Multilingual Language Models](https://aclanthology.org/2024.acl-long.427/)
- [Private Release of Text Embedding Vectors](https://aclanthology.org/2021.trustnlp-1.3/)
- [NoPeek](https://arxiv.org/abs/2008.03248)
- [Product Quantization for Nearest Neighbor Search](https://ieeexplore.ieee.org/document/5432202)
- [TEM: High Utility Metric Differential Privacy on Text](https://epubs.siam.org/doi/10.1137/1.9781611978032.99)
