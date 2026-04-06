# untitled

# Auditing Norm-Clipped L2-Laplacian Token-Embedding Obfuscation Against Sequence-Aware Reconstruction

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

A common privacy risk in “model-as-a-service” NLP deployments is that user text (prompts, queries, messages) must be converted into token embeddings to be processed by a remote model. Even if data is encrypted in transit, the server-side system typically operates on plaintext tokens or embeddings, which can leak user content if embeddings are logged, exfiltrated, or inspected by an insider.

A widely studied mitigation is to perturb the client-side token embeddings before they are sent to the server, typically using local differential privacy (LDP) mechanisms such as Gaussian or Laplacian noise. Many papers then evaluate privacy by attempting to invert the perturbed embeddings back to text. A frequent baseline attacker is *per-token nearest-neighbor decoding*: for each noisy embedding vector, pick the closest token embedding in the vocabulary.

Recent work suggests this evaluation can be misleading. BeamClean is a sequence-aware reconstruction attack that uses a language-model prior and beam search to decode an entire token sequence jointly, rather than token-by-token. BeamClean reports much higher reconstruction success than nearest-neighbor baselines under input-independent noise. For example, on MRPC (the Microsoft Research Paraphrase Corpus, a sentence-pair paraphrase dataset from the GLUE benchmark) with Laplacian noise at DP ε=8.5, BeamClean achieves 86% token recovery (attack success rate; higher is worse privacy) versus 18% for nearest neighbor ([Figure 3](./references/BeamClean-Language-Aware-Embedding-Reconstruction/sections/BeamClean always outperforms Nearest Neighbor..md)). On PAPILLON (a benchmark that detects personally identifying information strings in text), BeamClean recovers 60.0% of detected PII strings versus 1.9% for nearest neighbor at ε=8.5 ([Figure 4](./references/BeamClean-Language-Aware-Embedding-Reconstruction/sections/BeamClean recovers significantly higher PIIs compared to Nearest Neighbor..md)).

### The Problem

This proposal targets a specific, common design choice in LDP-style embedding perturbation: **norm clipping after adding noise**.

- **Split-and-Denoise (SnD)** proposes a split-inference architecture where the client computes token embeddings, perturbs them with L2-Laplacian noise under a metric-LDP guarantee (dχ-privacy; privacy defined with respect to distances in embedding space), and then **clips the perturbed embedding** to a radius equal to the maximum token-embedding norm over the vocabulary (chosen as an upper bound). SnD evaluates embedding inversion privacy using a token-level nearest-neighbor attack and reports very low token recovery (attack success rate; higher is worse privacy). For example, it states that for BERT models the inversion attack success rate “remain[s] below 1% with η≤500” ([SnD Section 4.4.3](./references/Split-and-Denoise-Protect-large-language-model-inference-with-local-differential-privacy/sections/4.4.3 Inference Attack.md)).

- **BeamClean** shows that for *unclipped* additive noise mechanisms, nearest-neighbor inversion can dramatically underestimate leakage because it ignores sequence structure. However, BeamClean’s experiments were “restricted to input independent noise mechanisms,” and norm clipping changes the observed distribution: if the noisy vector falls outside the clipping ball, the output is projected to the boundary, producing a truncated / projected distribution that is no longer the simple input-independent Laplace mechanism.

This creates an open, decision-relevant question: **does norm clipping materially reduce the advantage of a sequence-aware attacker, or does a language-model prior still recover substantially more text than nearest-neighbor decoding at the same noise level?**

This is not a “paper X didn’t test Y” gap for its own sake. If nearest-neighbor evaluation remains over-optimistic even with clipping, then (i) privacy claims in split-inference papers that rely on nearest-neighbor attacks are weaker than they appear, and (ii) practitioners deploying embedding perturbation should reassess the privacy risk at their chosen parameters. Conversely, if clipping collapses the gap, then a simple post-processing step may meaningfully strengthen LDP-style embedding perturbation against stronger reconstruction attacks.

### Key Insight and Hypothesis

**Hypothesis**: For norm-clipped L2-Laplacian perturbation of token embeddings, a sequence-aware attacker (BeamClean-style decoding with a language-model prior) will still recover substantially more tokens (and synthetic canary substrings) than per-token nearest-neighbor decoding, at the same noise level and with clipping actively triggered.

**Mechanism intuition**: Clipping limits the magnitude of the transmitted vectors but does not remove the core ambiguity exploited by sequence-aware attacks: many candidate tokens can remain plausible under moderate noise, and a language-model prior can resolve this ambiguity using context. Clipping may also create boundary effects (many outputs collapse onto the norm-C sphere) that preserve directional information, which a sequence-level decoder can exploit.

**Why we could be wrong**: Clipping induces a projected distribution with a singular component on the sphere boundary. If BeamClean’s surrogate noise model cannot learn a good likelihood model for this projected noise, the sequence-aware advantage may disappear. In that case, clipping could be an effective practical mitigation (even if it is not explicitly motivated as one).

---

## Proposed Approach

### Overview

We will perform a focused audit of norm-clipped L2-Laplacian token-embedding perturbation by comparing two attackers on exactly the same obfuscated embeddings:

1. **Nearest-neighbor token decoding** (standard baseline used in SnD).
2. **BeamClean** (sequence-aware decoding with a language-model prior), trained to model the clipped-noise distribution via BeamClean’s surrogate-noise learning interface.

The audit is designed to be decisive with a small number of experimental conditions. The core result is whether the gap between (1) and (2) persists when clipping is active.

### Method Details

#### Perturbation mechanism under test (norm-clipped L2-Laplacian)

Given a token embedding \(x\in\mathbb{R}^d\), sample L2-Laplacian noise \(z\) with density proportional to \(\exp(-\eta\|z\|_2)\) (as in SnD). Form the noisy vector \(u=x+z\), then clip to radius \(C\):

\[
\mathrm{clip}_C(u) = \min\left(1, \frac{C}{\|u\|_2}\right) u.
\]

This matches SnD’s description: “the client clips the l2 norm of the privatized representation within \(C_{x_t}\)” where \(C_{x_t}=\max_{x_t\in X_t}\|x_t\|\) is chosen as an upper bound ([SnD Section 3.3](./references/Split-and-Denoise-Protect-large-language-model-inference-with-local-differential-privacy/sections/3.3 Noise Mechanism.md)).

We will explicitly report the **clip rate** (fraction of tokens with \(\|x+z\|_2>C\)) to ensure clipping is not a no-op.

#### Attacker baselines

- **Nearest Neighbor (NN)**: For each token position \(t\), decode \(\hat{w}_t = \arg\min_{w\in\mathcal{V}} \|E(w) - y_t\|_2\), where \(y_t\) is the clipped noisy embedding.

- **BeamClean (clipping-aware)**: Use BeamClean’s beam-search decoding objective combining (i) a learned surrogate likelihood \(\pi_\theta(y_t\mid x_{1:t})\) and (ii) a language-model prior \(p_{LM}(w_t\mid w_{<t})\). To adapt to clipping, we will train the surrogate noise model on synthetic samples from the projected mechanism (generate \(y=\mathrm{clip}_C(x+z)\) for known \(x\) and optimize log-likelihood under BeamClean’s surrogate family, e.g., diagonal Gaussian or Laplace parameterization).

This directly tests BeamClean’s claim that the framework is applicable beyond simple input-independent noise. BeamClean notes that its experiments were restricted to input-independent mechanisms, but that it is “also applicable to more sophisticated input-dependent noise mechanisms” with further experimentation ([BeamClean Limitations](./references/BeamClean-Language-Aware-Embedding-Reconstruction/sections/6 Limitations.md)).

### Key Innovations

- **Mechanism-specific audit**: Evaluate a common post-processing step (norm clipping) that materially changes the obfuscation distribution, rather than re-running BeamClean on the unclipped setting.
- **Clipping-aware attacker implementation**: Train BeamClean’s surrogate noise model on projected-noise samples so the attacker is not artificially weakened by a mismatched likelihood.
- **Decision-oriented metrics**: Measure both generic token recovery and *deployment-relevant string leakage* using synthetic canaries (avoids handling real personal data).

---

## Related Work

### Field Overview

Embedding perturbation for privacy sits at the intersection of (i) local / metric differential privacy mechanisms (often implemented by adding calibrated noise in embedding space), (ii) split learning / collaborative inference architectures where a client shares intermediate representations with an untrusted server, and (iii) embedding inversion and reconstruction attacks that attempt to recover user text from shared representations.

A recurring pattern is that privacy evaluation depends heavily on the attacker model. Token-wise nearest-neighbor attacks are easy to implement and often reported, but they ignore sequence constraints. Sequence-aware reconstruction (e.g., via an LM prior) can substantially increase inversion success under additive noise. This proposal focuses on whether a common post-processing step (norm clipping) changes that conclusion.

### Related Papers

- **[BeamClean: Language Aware Embedding Reconstruction](./references/BeamClean-Language-Aware-Embedding-Reconstruction/meta/meta_info.txt)**: Introduces a beam-search + language-prior reconstruction attack that greatly outperforms nearest-neighbor decoding under Gaussian/Laplacian noise.
- **[Split-and-Denoise: Protect large language model inference with local differential privacy](./references/Split-and-Denoise-Protect-large-language-model-inference-with-local-differential-privacy/meta/meta_info.txt)**: Proposes client-side token-embedding perturbation with L2-Laplacian noise and norm clipping; evaluates inversion mostly with nearest-neighbor decoding.
- **[Text Embeddings Reveal (Almost) As Much As Text](./references/Text-Embeddings-Reveal-(Almost)-As-Much-As-Text/meta/meta_info.txt)**: Introduces Vec2Text-style embedding inversion showing high leakage from text embeddings.
- **[Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](./references/Understanding-and-Mitigating-the-Threat-of-Vec2Text-to-Dense-Retrieval-Systems/meta/meta_info.txt)**: Studies Vec2Text leakage in retrieval and mitigation strategies.
- **[InvBERT: Reconstructing Text from Contextualized Word Embeddings by Inverting the BERT Pipeline](https://arxiv.org/abs/2109.10104)**: Early work reconstructing text from contextualized embeddings.
- **[Natural Language Understanding with Privacy-Preserving BERT](https://dl.acm.org/doi/10.1145/3459637.3482240)**: Uses privacy-preserving mechanisms for BERT-based NLU and considers inversion risks.
- **[Privacy-Preserving Prompt Tuning for Large Language Model Services (RAPT)](https://arxiv.org/abs/2305.06212)**: Applies local privacy mechanisms to prompt tuning/inference and evaluates privacy attacks.
- **[DP-Forward: Fine-tuning and Inference on Language Models with Differential Privacy in Forward Pass](https://arxiv.org/abs/2306.12781)**: Adds DP noise during forward passes and studies privacy/utility trade-offs.
- **[TEM: High Utility Metric Differential Privacy on Text](https://arxiv.org/abs/2302.07427)**: Proposes a truncated exponential mechanism for high-utility metric-DP text perturbation.
- **[A Differentially Private Text Perturbation Method Using a Regularized Mahalanobis Metric](https://arxiv.org/abs/2010.11947)**: Uses Mahalanobis metrics to shape perturbations in embedding space.
- **[Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations](https://dl.acm.org/doi/10.1145/3366423.3380044)**: Metric-DP perturbation mechanisms for text analysis.
- **[Private Release of Text Embedding Vectors](https://aclanthology.org/2021.trustnlp-1.3/)**: Studies releasing embedding vectors with privacy guarantees.
- **[Broadening the Scope of Differential Privacy Using Metrics](https://petsymposium.org/2013/papers/hotpets13-paper25.pdf)**: Introduces metric-based DP, relevant to dχ-privacy.
- **[Calibrating Noise to Sensitivity in Private Data Analysis](https://link.springer.com/chapter/10.1007/11681878_14)**: Foundational DP result motivating Laplace noise scaling.
- **[The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)**: Reference text for DP definitions and properties.
- **[Large Language Models Can Be Strong Differentially Private Learners](https://arxiv.org/abs/2110.05679)**: DP learning with LLMs; contextual background.
- **[Differentially Private Fine-Tuning of Language Models](https://arxiv.org/abs/2110.06500)**: DP fine-tuning methods; background on DP in NLP.
- **[A Split-and-Privatize Framework for Large Language Model Fine-Tuning](https://arxiv.org/abs/2312.15603)**: Split learning + privacy mechanisms for LLM adaptation.
- **[Text Embeddings Reveal (Almost) As Much As Text (Vec2Text)](https://arxiv.org/abs/2310.06816)**: Canonical embedding inversion attack for text embeddings.
- **[Rethinking the Privacy of Text Embeddings](https://arxiv.org/abs/2507.07700)**: Later analysis of embedding privacy and reproducibility issues.
- **[Learning Obfuscations of LLM Embedding Sequences](https://arxiv.org/abs/2506.09452)**: Proposes learned, sequence-dependent obfuscations and evaluates against stronger attacks (including BeamClean).
- **[Prompt Inversion Attack against Collaborative Inference](https://arxiv.org/abs/2503.09022)**: Studies inversion risks in collaborative/split inference settings.
- **[OSNIP: Signal-to-Noise Ratio Based Input Perturbation for LLM Privacy](https://arxiv.org/abs/2601.22752)**: Explores alternative perturbation mechanisms for private inference.
- **[Depth Gives a False Sense of Privacy: LLM Internal States Inversion](https://arxiv.org/abs/2507.16372)**: Shows inversion risks for deeper internal states; adjacent to representation leakage.
- **[PAPILLON: Privacy Preservation from Internet-Based and Local Language Model Ensembles](https://arxiv.org/abs/2410.17127)**: Provides an evaluation approach for PII leakage in generated text.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Token-embedding perturbation (local / metric DP) | Add calibrated noise to token embeddings before server processing | SnD (2310.09130), TEM (2302.07427), Mahalanobis DP perturbation (2010.11947), calibrated multivariate perturbations (WSDM 2020) | Token recovery (ASR), downstream task accuracy, PII/canary leakage | Evaluation often uses weak attackers; additive noise is vulnerable to LM-prior attacks |
| Split inference / collaborative inference privacy | Split a model; share privatized intermediate representations | SnD (2310.09130), split-and-privatize (2312.15603), prompt-tuning privacy (2305.06212) | Utility on downstream tasks + inversion attacks | Threat models and attacker strength vary widely |
| Embedding inversion attacks | Reconstruct text from representations | BeamClean (2505.13758), Vec2Text (2310.06816), InvBERT (2109.10104) | Token/sequence recovery, semantic similarity, PII recovery | Attack power depends on priors, training data access, and mechanism knowledge |
| Learned obfuscations / input-dependent mechanisms | Learn transformations to reduce leakage while preserving utility | Learning Obfuscations of LLM Embedding Sequences (2506.09452), OSNIP (2601.22752) | Utility + robustness to strong attacks | Training overhead; guarantees may be weaker or more complex |

### Closest Prior Work

1. **BeamClean (Kale et al., 2025)**: Proposes a sequence-aware reconstruction attack and shows that nearest-neighbor evaluation can underestimate leakage by large margins under input-independent Gaussian/Laplacian noise (e.g., 86% vs 18% token recovery at DP ε=8.5 on MRPC with Llama-3.2-1B-Instruct embeddings). BeamClean explicitly notes its experiments focus on input-independent noise; it does not evaluate norm clipping.

2. **Split-and-Denoise (Mai et al., 2023)**: Proposes L2-Laplacian perturbation with norm clipping and a client-side denoiser. Its embedding inversion evaluation is token-level nearest neighbor and reports very low recovery for some settings (e.g., “below 1%” for BERT when η≤500). It does not test sequence-aware attackers.

3. **Learning Obfuscations of LLM Embedding Sequences (2025)**: Introduces learned, sequence-dependent obfuscations and evaluates robustness against stronger reconstruction attacks (including BeamClean). This work motivates using stronger attackers, but it does not isolate whether simple post-processing steps like norm clipping are sufficient to close the sequence-aware gap for L2-Laplacian mechanisms.

**Novelty Kill Search Summary:** Searched for “BeamClean clipping”, “BeamClean projected noise”, “Split-and-Denoise BeamClean”, “norm clipped Laplace embedding inversion”, and checked arXiv for follow-ups discussing SnD + BeamClean-style attacks. No prior work explicitly evaluating BeamClean (or an LM-prior sequence-aware reconstruction) against SnD-style norm-clipped L2-Laplacian perturbation was found as of 2026-02-19. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| BeamClean (2505.13758) | Sequence-aware reconstruction for input-independent noise | Does not test norm clipping / projected noise | Evaluate projected-noise mechanism and train surrogate on it | Determines whether the BeamClean-vs-NN gap persists under clipping |
| Split-and-Denoise (2310.09130) | Norm-clipped L2-Laplacian perturbation + client denoising | Uses token-level NN evaluation for inversion | Replace inversion evaluation with sequence-aware attacker | Tests whether reported low inversion rates remain low under stronger attacks |
| Learning Obfuscations of LLM Embedding Sequences (2506.09452) | Learned sequence-dependent obfuscation evaluated vs strong attacks | Does not isolate the effect of simple clipping post-processing | Isolate clipping as a single mechanism change | Clarifies whether “learned obfuscation” is necessary vs simple post-processing |

---

## Experiments

### Experimental Setup

**Primary goal**: compare a token-wise attacker vs a sequence-aware attacker on the *same* norm-clipped L2-Laplacian perturbation.

**Baseline Ladder (REQUIRED):** (re-interpreted for attacker strength in a privacy audit)
- **Level 0 (trivial)**: Random-token baseline (chance-level token recovery) to sanity-check metrics.
- **Level 1 (standard weak attacker)**: Nearest-neighbor decoding per token (used in SnD and many LDP embedding papers).
- **Level 2 (strong attacker)**: BeamClean with a language-model prior, training its surrogate noise model on samples from the clipped mechanism.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| GPT-2 (tokenizer + embedding table + LM prior) | 124M | https://huggingface.co/gpt2 | Open model; smaller vocabulary (50k) reduces reconstruction cost |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| GLUE MRPC | Evaluation sequences (truncate to 32 tokens) | ~3.7k train / 1.7k test | https://huggingface.co/datasets/glue | Standard benchmark license (per HF card) |
| WikiText-2 (optional secondary) | OOD text sequences | small | https://huggingface.co/datasets/wikitext | Standard dataset license (per HF card) |

**Other Resources (if applicable):**
- None.

**Resource Estimate**:
- **Compute budget**: 50–200 GPU-hours total (≤768 cap), dominated by BeamClean decoding.
  - Surrogate-noise training for clipped mechanism: ~1–5 GPU-hours per seed on 1×A100 (small model + short sequences).
  - Decoding: estimate ~0.05–0.2 GPU-hours per 100 sequences at beam size 20 and candidate pool ≤512 (to be confirmed by an initial timing run).
  - We will use **3 seeds** (noise sampling + surrogate training init) for all main results.
- **GPU memory**: ≤1×A100 80GB should suffice for GPT-2 + candidate pool search; may use more GPUs only for parallelizing seeds.
- **API usage**: none required.

**Infrastructure constraints**:
- No GUI, no browsers, no external search APIs required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| GLUE MRPC | Sentence-pair paraphrase dataset; we use raw text as natural sequences | Token-ASR, Seq-EM, Canary-EM, clip rate | test | https://huggingface.co/datasets/glue | Custom script: tokenize → perturb → attack → compare |
| WikiText-2 (optional) | Natural text corpus | Token-ASR, Seq-EM, Canary-EM, clip rate | test | https://huggingface.co/datasets/wikitext | Same script |

**Metric definitions:**
- **Token-ASR**: fraction of tokens correctly recovered (higher = worse privacy).
- **Seq-EM**: exact-match rate of the entire 32-token sequence (higher = worse privacy).
- **Canary-EM**: fraction of sequences where a synthetic canary substring is fully recovered (higher = worse privacy).
- **Clip rate**: fraction of tokens where \(\|x+z\|_2>C\) and projection is applied.

### Main Results

#### Results Table

(All numbers are to be produced by verification; published numbers are used only for motivation and sanity checks, not as directly comparable baselines.)

| Method | Base Model | Benchmark | Token-ASR (mean±std) | Canary-EM (mean±std) | Source | Notes |
|--------|------------|-----------|----------------------|----------------------|--------|-------|
| Random token | GPT-2 | MRPC | **TBD** | **TBD** | - | Sanity check (chance-level) |
| Nearest neighbor | GPT-2 | MRPC | **TBD** | **TBD** | - | Standard token-wise attacker |
| BeamClean (trained on clipped mechanism) | GPT-2 | MRPC | **TBD** | **TBD** | - | Main condition |

#### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| BeamClean (mismatched noise model) | Use BeamClean’s unclipped Laplace/Gaussian surrogate on clipped data (no retraining) | If attack still strong, gap is robust to modeling mismatch |
| No clipping (sanity) | Evaluate BeamClean on unclipped L2-Laplacian at matched noise level | Reproduces known BeamClean > NN gap; validates implementation |

### Experimental Rigor

**Variance & Reproducibility:**
- Run all main experiments across **3 random seeds**: `seeds=[42, 123, 456]`.
- Each seed re-samples obfuscation noise and re-trains the surrogate-noise model (if training is used).

**Validity & Controls:**
- **Clipping-activity control**: If clip rate <5% at the chosen \(\eta\), adjust \(\eta\) until clip rate is in a target band (30–50%) and report results at that operating point. This avoids a vacuous “clipping helps” conclusion when clipping is rarely applied.
- **Implementation sanity check**: Confirm that on the unclipped mechanism, BeamClean achieves substantially higher Token-ASR than nearest neighbor (qualitatively matching the BeamClean paper).
- **Compute-matched decoding**: Keep beam size and candidate pool fixed across clipped and unclipped runs so differences are attributable to clipping.

---

## Success Criteria

**Hypothesis** (directional — what we expect):
Norm clipping will not eliminate the reconstruction advantage from sequence-aware decoding; BeamClean will recover far more tokens and canaries than nearest-neighbor decoding when clipping is active.

**Decision Rule** (concrete — when to stop):
- **Proceed/Confirm evaluation gap**: At a noise setting where clip rate is between 30% and 50%, BeamClean’s Token-ASR on MRPC exceeds nearest neighbor by **≥20 absolute points** *or* by **≥10×** (whichever is smaller), and Canary-EM is **≥5×** higher, across 3 seeds.
- **Pivot (vacuous regime)**: If clip rate <5% for the initial \(\eta\), adjust \(\eta\) (stronger noise) until clip rate is in 30–50% and re-run.
- **Refute (clipping closes the gap)**: After ensuring clip rate is 30–50%, if BeamClean’s Token-ASR is within **5 absolute points** of nearest neighbor and Canary-EM is within **1.2×**, across 3 seeds, conclude that clipping largely removes the sequence-aware advantage in this regime.

---

## Impact Statement

If sequence-aware reconstruction remains strong under norm clipping, researchers should stop using nearest-neighbor inversion as a primary privacy evaluation for token-embedding perturbation, and practitioners deploying embedding perturbation in split inference should reassess leakage at their chosen parameters. If clipping collapses the gap, it provides a simple, low-cost post-processing step that can materially improve resistance to stronger reconstruction attacks.

---

## References

- [BeamClean: Language Aware Embedding Reconstruction](./references/BeamClean-Language-Aware-Embedding-Reconstruction/meta/meta_info.txt) - Kale et al., 2025
- [Split-and-Denoise: Protect large language model inference with local differential privacy](./references/Split-and-Denoise-Protect-large-language-model-inference-with-local-differential-privacy/meta/meta_info.txt) - Mai et al., 2023
- [Text Embeddings Reveal (Almost) As Much As Text](./references/Text-Embeddings-Reveal-(Almost)-As-Much-As-Text/meta/meta_info.txt) - Morris et al., 2023
- [Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](./references/Understanding-and-Mitigating-the-Threat-of-Vec2Text-to-Dense-Retrieval-Systems/meta/meta_info.txt) - Zhuang et al., 2024
- [InvBERT: Reconstructing Text from Contextualized Word Embeddings by Inverting the BERT Pipeline](https://arxiv.org/abs/2109.10104) - Kugler et al., 2021
- [Natural Language Understanding with Privacy-Preserving BERT](https://dl.acm.org/doi/10.1145/3459637.3482240) - Qu et al., 2021
- [Privacy-Preserving Prompt Tuning for Large Language Model Services](https://arxiv.org/abs/2305.06212) - Li et al., 2023
- [DP-Forward: Fine-tuning and Inference on Language Models with Differential Privacy in Forward Pass](https://arxiv.org/abs/2306.12781) - Du et al., 2023
- [TEM: High Utility Metric Differential Privacy on Text](https://arxiv.org/abs/2302.07427) - Carvalho et al., 2023
- [A Differentially Private Text Perturbation Method Using a Regularized Mahalanobis Metric](https://arxiv.org/abs/2010.11947) - Xu et al., 2020
- [Private Release of Text Embedding Vectors](https://aclanthology.org/2021.trustnlp-1.3/) - Feyisetan & Kasiviswanathan, 2021
- [Broadening the Scope of Differential Privacy Using Metrics](https://petsymposium.org/2013/papers/hotpets13-paper25.pdf) - Chatzikokolakis et al., 2013
- [Calibrating Noise to Sensitivity in Private Data Analysis](https://link.springer.com/chapter/10.1007/11681878_14) - Dwork et al., 2006
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) - Dwork & Roth, 2014
- [Large Language Models Can Be Strong Differentially Private Learners](https://arxiv.org/abs/2110.05679) - Li et al., 2021
- [Differentially Private Fine-Tuning of Language Models](https://arxiv.org/abs/2110.06500) - Yu et al., 2021
- [A Split-and-Privatize Framework for Large Language Model Fine-Tuning](https://arxiv.org/abs/2312.15603) - Shen et al., 2023
- [Rethinking the Privacy of Text Embeddings](https://arxiv.org/abs/2507.07700) - (authors), 2025
- [Learning Obfuscations of LLM Embedding Sequences](https://arxiv.org/abs/2506.09452) - (authors), 2025
- [Prompt Inversion Attack against Collaborative Inference](https://arxiv.org/abs/2503.09022) - (authors), 2025
- [OSNIP: Signal-to-Noise Ratio Based Input Perturbation for LLM Privacy](https://arxiv.org/abs/2601.22752) - (authors), 2026
- [Depth Gives a False Sense of Privacy: LLM Internal States Inversion](https://arxiv.org/abs/2507.16372) - (authors), 2025
- [PAPILLON: Privacy Preservation from Internet-Based and Local Language Model Ensembles](https://arxiv.org/abs/2410.17127) - Siyan et al., 2025
