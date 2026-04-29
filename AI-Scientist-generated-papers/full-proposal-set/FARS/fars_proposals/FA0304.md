# untitled

# Does OSNIP’s “Dimensionality Dividend” Survive Language-Aware Token Reconstruction?

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, USENIX Security, CCS (or similar)

## Introduction

### Context and Motivation

Large language models (LLMs) are widely deployed as **model-as-a-service (MaaS)**: users send prompts to a cloud provider that holds the model weights. This creates a privacy risk because the provider (or a compromise of the provider) can log user prompts.

A practical alternative to cryptographic private inference (e.g., secure multi-party computation or fully homomorphic encryption) is **client-side embedding obfuscation**. In this setting, the client tokenizes locally, maps tokens to **input token embeddings**, applies an obfuscation transform to hide token identity, and sends only the transformed embeddings to the server. The server then runs the frozen LLM forward pass starting from these embeddings.

**OSNIP** proposes a learned obfuscator that aims to preserve utility while reducing privacy leakage by pushing obfuscated token embeddings to be nearly orthogonal to the originals (“obfuscated semantic null space”) (**[OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma-in-LLM-Inference-via-Obfuscated-Semantic-Null-Space/meta/meta_info.txt)**).

A central claim in OSNIP is that privacy–utility trade-offs improve with model scale (and embedding dimension), which they call a **“dimensionality dividend.”** For example, OSNIP reports that under a K-nearest-neighbor attack (KNN Top-10, reported as **ASR@10**, attack success rate; lower is safer), Llama-3.2-1B has ASR@10=0.066 while Llama-3.2-3B-Instruct has ASR@10=0.021 (**Table 1 in** **[OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma-in-LLM-Inference-via-Obfuscated-Semantic-Null-Space/sections/4.2%20Main%20Results.md)**).

### The Problem

OSNIP’s theoretical argument (Theorem 2.5 / Corollary 2.6) is about **geometric feasibility**: in high-dimensional spaces, the set of directions that are nearly orthogonal to a given vector occupies almost all directions, so it should be easier to find perturbations that preserve the model’s predictions while being nearly orthogonal to the original embedding.

However, OSNIP’s headline privacy evaluation uses primarily **distance-based** embedding inversion metrics (KNN Top-k over the vocabulary embedding table, plus a “vocabulary-matching” reconstruction). This raises a threat-model question: does a reduction in distance-based matchability imply reduced recoverability under attackers that use **sequence-level language priors**?

Recent embedding inversion work suggests it may not. **BeamClean** shows that language-aware decoding can significantly outperform nearest-neighbor decoding on obfuscated token embeddings. For instance, on MRPC with Llama-3.2-1B-Instruct embeddings, BeamClean recovers 74.3% of tokens vs. 42.1% for nearest neighbor under Gaussian noise at ε=15, and 86% vs. 18% under Laplacian noise at ε=8.5 (**[BeamClean](./references/BeamClean-Language-Aware-Embedding-Reconstruction/sections/BeamClean%20always%20outperforms%20Nearest%20Neighbor..md)**).

This proposal asks a focused audit question:

> When utility is matched, does OSNIP’s empirical “larger model ⇒ better privacy” trend persist under a stronger, language-aware reconstruction attacker, or is the trend specific to distance-based evaluation?

### Key Insight and Hypothesis

**Key insight.** OSNIP’s training objective directly penalizes **directional similarity** (cosine similarity) between clean and obfuscated token embeddings. In higher dimensions, satisfying a near-orthogonality constraint becomes easier. But “easy orthogonality” does not necessarily imply “hard inversion”: the obfuscator may move the true token from rank-1 to rank-r while still leaving enough information for a language-model prior to resolve ambiguity among many plausible candidates.

**Hypothesis (audit claim).** At a fixed utility operating point, the KNN-based “dimensionality dividend” will shrink or disappear under a language-aware attacker. Concretely, compared to KNN Top-10 ASR, a language-aware attacker that (i) enumerates Top-k nearest candidates per position and (ii) decodes sequences using a language model prior will reduce the apparent privacy gap between a smaller and a larger model within the same tokenizer family.

Why we could be wrong:
- OSNIP’s obfuscation may remove token-identity information in a way that remains hard to reconstruct even with language priors.
- For input-layer token embeddings, language priors might not provide a large advantage over KNN, making KNN an adequate proxy in this setting.

---

## Proposed Approach

### Overview

We conduct a two-scale audit within a single model family:

1. Train an OSNIP-style token-embedding obfuscator for a **smaller** and a **larger** model that share the same tokenizer.
2. Match utility across scales by selecting a single operating point defined by a relative cross-entropy increase threshold.
3. Evaluate privacy under:
   - **Attacker A (baseline):** KNN Top-10 token recovery (OSNIP-style).
   - **Attacker B (proposed):** Language-aware LM-reranked Top-k reconstruction.

The core outcome is whether the scale-dependent privacy improvement seen under KNN persists under the language-aware attacker.

### Method Details

#### Base models (scale axis)

Primary plan (same tokenizer; smaller compute than 8B/14B/32B):

- **Llama-3.2-1B-Instruct** (hidden size d=2048)
  - https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- **Llama-3.2-3B-Instruct** (hidden size d=3072)
  - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

Fallback (if Llama weights are inaccessible in the execution environment):

- **Qwen2.5-0.5B-Instruct** (hidden size d=896)
  - https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- **Qwen2.5-1.5B-Instruct** (hidden size d=1536)
  - https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

#### Obfuscator to evaluate: OSNIP-style token-embedding encryptor

We follow OSNIP’s formulation (**[OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma-in-LLM-Inference-via-Obfuscated-Semantic-Null-Space/sections/3.2%20Encryption%20Network.md)**) while keeping the implementation minimal:

- Let h=g(x) be the clean token embedding sequence from the frozen LLM.
- Train an encryptor Rφ that outputs z=Project(h+Δφ(h)), where Project enforces OSNIP’s iso-norm constraint.

Concrete parameterization:
- Δφ(h_t) is a 2-layer MLP applied token-wise to each embedding (SiLU activation; hidden width = d).
- Iso-norm projection: z_t = (h_t + Δ_t) · ||h_t||₂ / ||h_t + Δ_t||₂.

Losses (OSNIP-style):
- Utility loss: L_util = KL(fθ(h) || fθ(z)), where fθ is the frozen LLM from the embedding layer onward.
- Privacy loss: L_priv = max(0, |cos(h,z)| − ε) (hinge loss).
- Total: E[L_util + λ L_priv], with λ linearly warmed up from 0.

We intentionally **omit key-conditioning** to isolate the geometric scaling claim from key-search/linkability considerations.

#### Utility matching rule (critical control)

We define a single operating point by scaling the learned perturbation magnitude at evaluation time:

- z_t(s) = Project(h_t + s · Δ_t), with s∈[0,1].
- Choose s by 1D search on a held-out calibration set so that:
  - ΔCE% = (CE_enc − CE_clean) / CE_clean ≤ 1%.

Here CE is per-token cross-entropy under teacher forcing. This prevents “privacy by destroying utility.”

#### Attacker A (baseline): KNN Top-10 token recovery

For each obfuscated token embedding z_t, rank all vocabulary token embeddings e_w by Euclidean distance, and count a success if the ground-truth token is in the Top-10 list.

Metric: **Token-ASR@10** (attack success rate; lower is safer).

#### Attacker B (proposed): LM-reranked Top-k reconstruction

We implement a language-aware attacker inspired by BeamClean but simplified to avoid explicit noise-model estimation:

1. Candidate generation: for each position t, retrieve a candidate set C_t containing the Top-k nearest tokens to z_t (k=50) using FAISS over the vocabulary embedding table.
2. Sequence decoding: run beam search (beam=10) over sequences w₁:T with scoring:

   score(w₁:T) = Σ_t [ log p_prior(w_t | w_{<t}) + α · cos(z_t, e_{w_t}) ]

   where p_prior is a fixed pretrained language model prior (Llama-3.2-1B-Instruct for all scales).
3. Attacker calibration: choose α from {0, 0.5, 1, 2} to maximize Token-ASR on a disjoint calibration subset, representing an attacker that can tune using a small chosen-plaintext set.

Metrics:
- **Token-ASR@LM**: fraction of token positions exactly recovered by LM-reranked decoding.
- **Seq-EM@LM**: exact match rate of full T-token sequences.
- **Canary-EM**: exact recovery rate of an inserted synthetic secret substring.

### Key Innovations

- **Scale-aware privacy auditing**: We directly test whether OSNIP’s claimed “dimensionality dividend” is robust to attacker strength.
- **Practical language-aware attacker for input token embeddings**: LM-reranked Top-k decoding captures a major attacker class (language priors) while remaining implementable and interpretable.

---

## Related Work

### Field Overview

Embedding and activation privacy research studies how intermediate representations (token embeddings, hidden states, KV cache) leak user inputs in split inference and MaaS settings, and how to mitigate leakage. A recurring issue is that privacy claims can be highly sensitive to the attacker model: distance-based token recovery may underestimate leakage when attackers exploit sequence constraints and language priors.

This proposal focuses on two axes:
1. **Learned embedding obfuscation** methods that optimize a utility objective while perturbing representations.
2. **Embedding inversion** methods that reconstruct text from embeddings, increasingly using language priors or learned decoders.

### Related Papers

- **[OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma-in-LLM-Inference-via-Obfuscated-Semantic-Null-Space/meta/meta_info.txt)**: Learns token-embedding obfuscation with an orthogonality constraint; reports a scale-dependent privacy–utility improvement under KNN-style attacks.
- **[BeamClean](./references/BeamClean-Language-Aware-Embedding-Reconstruction/meta/meta_info.txt)**: Language-aware embedding reconstruction using beam search + a language-model prior; substantially outperforms nearest-neighbor inversion.
- **[SGT](./references/Learning-Obfuscations-Of-LLM-Embedding-Sequences-Stained-Glass-Transform/meta/meta_info.txt)**: Learns stochastic embedding transforms and emphasizes that nearest-neighbor metrics can be misleading due to rank/pathology effects.
- **[Split-and-Denoise](https://arxiv.org/abs/2310.09130)**: Local differential privacy style noise on token embeddings for split inference; commonly evaluated with nearest-neighbor attackers.
- **[DP-Forward](https://dl.acm.org/doi/10.1145/3576915.3616592)**: DP mechanisms for fine-tuning/inference; representative of forward-pass DP obfuscation.
- **[Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816)**: Shows strong invertibility of text embeddings, motivating leakage-first evaluation.
- **[Vec2Text](https://arxiv.org/abs/2401.11130)**: Trains decoders to invert embeddings, demonstrating high-fidelity reconstruction.
- **[InvBERT](https://arxiv.org/abs/2109.10104)**: Early contextual embedding inversion results, highlighting invertibility risks.
- **[ALGEN](https://arxiv.org/abs/2502.11308)**: Few-shot embedding inversion via alignment + generation, relevant for attackers with a small number of known plaintext pairs.
- **[Zero2Text](https://arxiv.org/abs/2602.01757)**: Zero-training embedding inversion across domains, strengthening attacker realism.
- **[Universal Zero-shot Embedding Inversion](https://arxiv.org/abs/2504.00147)**: Transferable inversion without per-target training.
- **[Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2401.01948)**: Theoretical view supporting strong invertibility concerns.
- **[Depth Gives a False Sense of Privacy](https://arxiv.org/abs/2507.16372)**: Shows deeper internal states can remain invertible; complements embedding-level concerns.
- **[Prompt Inversion Attack against Collaborative Inference](https://arxiv.org/abs/2503.09022)**: Inversion risks in split inference with activation sharing.
- **[NoPeek](https://arxiv.org/abs/2008.03248)**: Early split-learning defense reducing leakage in shared activations.
- **[Cape](https://arxiv.org/abs/2501.14316)**: DP-style prompt perturbation baseline used by OSNIP.
- **[DYNTEXT](https://aclanthology.org/2025.findings-acl.872/)**: Dynamic text sanitization baseline used by OSNIP.
- **[InferDPT](https://ieeexplore.ieee.org/document/10041305)**: Private inference baseline used by OSNIP.
- **[EncryptedLLM](https://arxiv.org/abs/2506.01414)**: FHE-based private inference alternative with different efficiency trade-offs.
- **[Iron](https://arxiv.org/abs/2205.05162)**: Homomorphic-encryption transformer inference; representative crypto baseline.
- **[$d_X$-Privacy for Text and the Curse of Dimensionality](https://arxiv.org/abs/2411.13784)**: Highlights counterintuitive high-dimensional failure modes for text privacy.
- **[An Inversion Attack Against Obfuscated Embedding Matrix in Language Model Inference](https://aclanthology.org/2024.emnlp-main.126/)**: Breaks embedding-obfuscation schemes, motivating stronger attacker models.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Learned embedding obfuscation | Train transform to preserve model outputs while hiding token identity | OSNIP; SGT | Utility + inversion metrics | Sensitive to attacker strength |
| DP / noise-based defenses | Add random noise (often local DP) to embeddings | Split-and-Denoise; DP-Forward | Nearest-neighbor ASR; downstream accuracy | Language-aware inversion can be much stronger |
| Language-aware inversion | Decode sequences using LM priors + embedding constraints | BeamClean; (this proposal) | Token/sequence exact match; PII recovery | Compute scales with vocab and sequence length |
| Zero-/few-shot inversion | Invert with few/no paired examples | ALGEN; Zero2Text | Text similarity + exact match | Strong in realistic leak settings |

### Closest Prior Work

1. **OSNIP**: Introduces the dimensionality-dividend claim and evaluates privacy mainly with KNN/vocab-matching. We test whether the *scale conclusion* survives a stronger language-aware attacker.
2. **BeamClean**: Demonstrates large gains from language-aware decoding. We adapt the key idea (LM-guided reconstruction) to audit OSNIP-style obfuscation across scales.
3. **SGT**: Shows nearest-neighbor metrics can be misleading due to rank structure. We test whether scale effects reported under KNN are robust to a different attacker family.

**Novelty Kill Search Summary:** We searched for work that explicitly tests OSNIP’s scaling claim under language-aware token reconstruction (and found none as of 2026-02-25). Queries included: “OSNIP dimensionality dividend attacker”, “obfuscated semantic null space inversion”, “OSNIP BeamClean”, “language-aware token embedding inversion OSNIP”, “semantic null space injection privacy attack”, and OpenReview searches for “OSNIP” and “semantic null space” with “inversion/attack”. No direct prior work auditing OSNIP’s dimensionality dividend with language-aware reconstruction was found.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| OSNIP | Learned token-embedding obfuscation; claims scaling dividend | Evaluation emphasizes distance-based attacks | Add language-aware attacker; test scaling under matched utility | Directly tests whether the dividend is evaluation-dependent |
| BeamClean | Strong inversion for noisy embeddings with LM prior | Not a scale audit; focuses on input-independent noise | Use LM-guided decoding as an audit attacker for OSNIP-style obfuscation | Captures attacker class missing in OSNIP evaluation |
| SGT | Learned obfuscation + rank/pathology insight | Not focused on OSNIP scaling | Use attacker-sensitivity framing to motivate a scale robustness test | If KNN is brittle, scale conclusions may be brittle |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):** This proposal is an audit of privacy evaluation, so the “baseline ladder” is defined over attacker strength.
- Level 0: random baseline (chance token match).
- Level 1: KNN Top-10 token recovery (OSNIP-style).
- Level 2: LM-reranked Top-k reconstruction (stronger attacker class).

**Early-stop gates (to avoid inconclusive outcomes):**
1. **Attacker-strength gate:** On at least one defended setting, Token-ASR@LM must exceed Token-ASR@10 by ≥5 absolute points. If not, conclude that (in this regime) language priors do not materially strengthen inversion beyond KNN, and stop.
2. **Utility-feasibility gate:** If we cannot find s achieving ΔCE%≤1% for either model, relax to ΔCE%≤2%. If still infeasible, report that OSNIP-style training did not reach a high-utility operating point in this setting.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3.2-1B-Instruct | 1B | https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct | used also as LM prior |
| Llama-3.2-3B-Instruct | 3B | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct | same tokenizer as 1B |

**Training Data (encryptor):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| OpenOrca | train encryptor and utility calibration | 10k train / 1k cal | https://huggingface.co/datasets/Open-Orca/OpenOrca | see HF dataset card |

Construction details:
- For each sample, take the first 32 tokens after formatting as a single instruction string.
- Prepend a synthetic canary span, e.g., `SECRET: 1234 5678 9012`, to make leakage unambiguous.

**Other Resources:**
- FAISS for nearest-neighbor retrieval over the vocabulary embedding table.

**Resource Estimate (≤ 768 GPU-hours):**

OSNIP does not report training compute. We therefore anchor estimates on a closely related learned obfuscation setting: SGT reports training an embedding obfuscator for Llama-3.2-1B on a single A100 80GB for roughly 6 hours (**[SGT](./references/Learning-Obfuscations-Of-LLM-Embedding-Sequences-Stained-Glass-Transform/sections/7%20Experiments.md)**). Our encryptor is smaller than SGT (token-wise MLP), but training still backpropagates through the frozen LLM.

Conservative budget:
- Encryptor training (per seed):
  - 1B model: 6 A100-hours
  - 3B model: 18 A100-hours (≈3× for larger model)
- Total encryptor training (2 models × 3 seeds): 72 GPU-hours
- Attacker evaluation (FAISS retrieval + LM scoring over 1k sequences): ≤30 GPU-hours
- **Total target budget:** ≤102 GPU-hours

If runtime is higher than expected, downscale by reducing training set size (10k→5k) and evaluation set size (1k→500), while keeping the attacker-strength gate.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| OpenOrca-Canary-32 | OpenOrca prompts truncated to 32 tokens with a synthetic canary prefix | Utility: ΔCE%; Privacy: Token-ASR@10, Token-ASR@LM, Seq-EM@LM, Canary-EM | test | https://huggingface.co/datasets/Open-Orca/OpenOrca | Custom (tokenize → embed → encrypt → attack → compare) |

Metric definitions (higher/lower):
- **ΔCE%**: relative cross-entropy increase under encrypted embeddings (lower is better utility).
- **Token-ASR@10**: token recovery success under KNN Top-10 (lower is safer).
- **Token-ASR@LM**: token recovery success under LM-reranked decoding (lower is safer).
- **Seq-EM@LM**: full-sequence exact match rate under LM-reranked decoding (lower is safer).
- **Canary-EM**: exact recovery rate of the canary span (lower is safer).

### Main Results

#### Results Table

| Method | Base Model | Benchmark | ΔCE% (mean±std) ↓ | Token-ASR (mean±std) ↓ | Canary-EM (mean±std) ↓ | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| OSNIP + KNN@10 | Llama-3.2-1B-Instruct | OpenOrca-Canary-32 | **TBD** | **TBD** | **TBD** | - | To be verified |
| OSNIP + LM-rerank (k=50,B=10) | Llama-3.2-1B-Instruct | OpenOrca-Canary-32 | **TBD** | **TBD** | **TBD** | - | To be verified |
| OSNIP + KNN@10 | Llama-3.2-3B-Instruct | OpenOrca-Canary-32 | **TBD** | **TBD** | **TBD** | - | To be verified |
| OSNIP + LM-rerank (k=50,B=10) | Llama-3.2-3B-Instruct | OpenOrca-Canary-32 | **TBD** | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| LM weight removed (α=0) | LM-rerank uses only embedding similarity over candidate sets | Lower Token-ASR than α>0, quantifying LM-prior contribution |

### Experimental Rigor

**Variance & Reproducibility:**
- Train encryptors with 3 seeds per model: `seeds=[42,123,456]`. Report mean±std.
- Attacker hyperparameters (k, beam) fixed across scales. α is selected on a disjoint calibration subset.

**Validity threats + controls:**
- **Confound: tokenizer/vocab differences** → controlled by within-family model pair.
- **Confound: utility mismatch** → controlled by explicit ΔCE% utility matching.
- **Confound: attacker tuning leaking test information** → attacker calibration uses a disjoint split.

**Data leakage:**
- Prompts are sampled from OpenOrca splits; calibration and test are disjoint.

---

## Success Criteria

**Hypothesis (directional):** KNN will show lower Token-ASR on the larger model at matched utility, but LM-reranked Token-ASR will show a much smaller scale advantage (or none).

**Decision Rule (concrete):**
- **Proceed (dividend is evaluation-dependent):** At matched utility, KNN Token-ASR@10 on the larger model is ≥30% relatively lower than on the smaller model, but LM Token-ASR@LM differs by ≤10% relative (averaged across seeds).
- **Proceed (dividend reverses):** At matched utility, LM Token-ASR@LM on the larger model is ≥10% relatively higher than on the smaller model.
- **Refute (dividend persists under stronger attacker):** At matched utility, LM Token-ASR@LM on the larger model is ≥30% relatively lower than on the smaller model, consistent with KNN.
- **Early stop (attacker not stronger):** If Token-ASR@LM exceeds Token-ASR@10 by <5 absolute points on both scales, stop and conclude that KNN is an adequate proxy in this regime.

---

## Impact Statement

If the dimensionality dividend disappears under language-aware reconstruction, researchers and practitioners should not interpret OSNIP’s scale trends under KNN ASR as evidence that “larger models are safer under embedding obfuscation,” and evaluation protocols should include language-aware attacks when making deployment decisions. If the dividend persists under language-aware attacks, it strengthens confidence that OSNIP’s scaling claim reflects a genuine reduction in recoverable token information rather than an artifact of distance-based evaluation.

---

## References

- [OSNIP: Breaking the Privacy-Utility-Efficiency Trilemma in LLM Inference via Obfuscated Semantic Null Space](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma-in-LLM-Inference-via-Obfuscated-Semantic-Null-Space/meta/meta_info.txt) - 2026
- [BeamClean: Language Aware Embedding Reconstruction](./references/BeamClean-Language-Aware-Embedding-Reconstruction/meta/meta_info.txt) - 2025
- [Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform](./references/Learning-Obfuscations-Of-LLM-Embedding-Sequences-Stained-Glass-Transform/meta/meta_info.txt) - 2025
- [Split-and-Denoise: Protect large language model inference with local differential privacy](https://arxiv.org/abs/2310.09130) - 2023
- [DP-Forward: Fine-tuning and inference on language models with differential privacy in forward pass](https://dl.acm.org/doi/10.1145/3576915.3616592) - 2023
- [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816) - 2023
- [Vec2Text: Inverting Text Embeddings](https://arxiv.org/abs/2401.11130) - 2024
- [InvBERT: Reconstructing Text from BERT Embeddings](https://arxiv.org/abs/2109.10104) - 2021
- [ALGEN: Few-shot Inversion Attacks on Textual Embeddings via Alignment and Generation](https://arxiv.org/abs/2502.11308) - 2025
- [Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings](https://arxiv.org/abs/2602.01757) - 2026
- [Universal Zero-shot Embedding Inversion](https://arxiv.org/abs/2504.00147) - 2025
- [Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2401.01948) - 2024
- [Depth Gives a False Sense of Privacy: LLM Internal States Inversion](https://arxiv.org/abs/2507.16372) - 2025
- [Prompt Inversion Attack against Collaborative Inference of Large Language Models](https://arxiv.org/abs/2503.09022) - 2025
- [NoPeek: Information leakage reduction to share activations in distributed deep learning](https://arxiv.org/abs/2008.03248) - 2020
- [Cape: Context-Aware Prompt Perturbation with Differential Privacy](https://arxiv.org/abs/2501.14316) - 2025
- [DYNTEXT: Semantic-Aware Dynamic Text Sanitization for Privacy-Preserving LLM Inference](https://aclanthology.org/2025.findings-acl.872/) - 2025
- [InferDPT: Privacy-Preserving Inference for Closed-Box Large Language Models](https://ieeexplore.ieee.org/document/10041305) - 2025
- [EncryptedLLM: Privacy-Preserving LLM Inference via GPU-Accelerated Fully Homomorphic Encryption](https://arxiv.org/abs/2506.01414) - 2025
- [Iron: Private Inference on Transformers via Fully Homomorphic Encryption](https://arxiv.org/abs/2205.05162) - 2022
- [$d_X$-Privacy for Text and the Curse of Dimensionality](https://arxiv.org/abs/2411.13784) - 2024
- [An Inversion Attack Against Obfuscated Embedding Matrix in Language Model Inference](https://aclanthology.org/2024.emnlp-main.126/) - 2024
