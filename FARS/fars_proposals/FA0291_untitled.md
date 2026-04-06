# untitled

# Key-Search Attacks Bypass Encrypted Activation Monitors for Key-Conditioned Embedding Obfuscation

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Many large language model (LLM) deployments run as **model-as-a-service**: users send prompts to a remote server that holds model weights. This is convenient, but it creates privacy risk because servers can log prompts containing sensitive text.

A fast alternative to cryptographic private inference is **client-side embedding obfuscation**: the client tokenizes locally, converts tokens to embeddings, applies a learned transform, and sends only transformed embeddings to the server. OSNIP is a representative method: it preserves next-token distributions (utility) while making embeddings nearly orthogonal to the originals (privacy via geometry), and it supports **key-conditioned personalization** (**[OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma-in-LLM-Inference-via-Obfuscated-Semantic-Null-Space/meta/meta_info.txt)**).

Safe deployment often also requires **input filtering** (content moderation): detect harmful requests before generation. Activation Monitoring shows that lightweight probes on a base model’s **internal activations** can achieve high true-positive rates at very low false-positive rates (FPR) (**[Activation Monitoring](./references/Activation-Monitoring-Advantages-of-Using-Internal-Representations-for-LLM-Oversight/meta/meta_info.txt)**).

### The Problem

A natural “private + safe” architecture is: client sends obfuscated embeddings; server runs the frozen LLM forward pass; server runs an activation-based harmful-request monitor on the hidden states.

However, key-conditioned obfuscation introduces an attacker-controlled choice variable (the key). If a user can try multiple keys for the same prompt and pick the one that minimizes the monitor score, they perform a **best-of-K key search**.

> **Does key-conditioned embedding obfuscation enable a low-cost key-search attack that bypasses activation-based harmful-request monitors, even when the monitors are trained on encrypted traffic with random keys?**

### Key Insight and Hypothesis

**Key insight:** A monitor trained on encrypted traffic with random keys is calibrated for the score distribution \(m(R(x,k))\) where \(k\sim\mathcal{K}\). A malicious user instead optimizes \(\min_{i\le K} m(R(x,k_i))\) by sampling \(K\) keys. If key diversity induces substantial variance in activations, then a modest K can cross below the fixed threshold \(\tau\) chosen to maintain low FPR.

**Hypothesis:** With an OSNIP-like key-conditioned encryptor (including key-diversity regularization), the encrypted-traffic activation monitor’s **TPR at FPR=1e-3** (true-positive rate at a threshold chosen so false-positive rate is 0.001; higher is better) drops by **≥20 absolute points** when moving from **K=1** (one random key per prompt) to **K=32** (key search), while:
- utility is preserved (mean per-token KL \(D_{KL}(f(h)\|f(z))\le 0.02\) on a held-out prompt set), and
- privacy is non-trivial (token-level KNN ASR@10 ≤ 0.20, where ASR@10 is the fraction of token positions whose true token appears in the top-10 nearest neighbors of the encrypted embedding in the vocabulary embedding table; lower is safer).

**Why we could be wrong:** Key variance may be small; the monitor may learn key-invariant features; or utility constraints may prevent high variance across keys.

---

## Proposed Approach

### Overview

We propose a distribution-matched red-team test:
1. Train an OSNIP-like key-conditioned encryptor \(R_\phi(h,k)\) for a frozen open LLM.
2. Train an activation monitor \(m\) on **encrypted activations** using random keys.
3. Evaluate harmful-request detection under a key-search attacker that samples \(K\) keys per prompt and picks the lowest monitor score.

### Method Details

#### Key-conditioned OSNIP-like encryptor

Minimal implementation of OSNIP’s objective (no claim of reproducing exact OSNIP hyperparameters):
- Per token: input \([h_t;k]\), where \(h_t\in\mathbb{R}^d\) is the token embedding and \(k\sim\mathcal{N}(0,I_{d_k})\) is a per-request key.
- Output \(\delta_t=\mathrm{MLP}_\phi([h_t;k])\); apply OSNIP’s iso-norm projection \(z_t=(h_t+\delta_t)\cdot \|h_t\|_2/\|h_t+\delta_t\|_2\).

Loss (OSNIP-style):
- Utility: \(L_{util}=D_{KL}(f_\theta(h)\|f_\theta(z))\).
- Privacy: \(L_{priv}=\max(0,|\cos(h,z)|-\epsilon)\).
- Key diversity: \(L_{div}=\max(0,\delta-\|R_\phi(h,k_1)-R_\phi(h,k_2)\|_2)\) for \(k_1\ne k_2\).
- Total: \(\mathbb{E}[L_{util}+\lambda_1 L_{priv}+\lambda_2 L_{div}]\), using OSNIP’s warmup + utility-gated schedule (**[OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma-in-LLM-Inference-via-Obfuscated-Semantic-Null-Space/sections/3.4%20Optimization%20Strategy%20Utility-Gated%20Curriculum.md)**).

#### Encrypted-traffic activation monitor

Following Activation Monitoring:
- Data: harmful prompts (HarmBench) vs harmless prompts (Alpaca).
- For each training example, sample a fresh key \(k\), encrypt embeddings, run the frozen LLM, extract a last-token hidden state from a late layer, and train an MLP \(m(a)\in[0,1]\).
- Choose threshold \(\tau\) to match a target FPR on a large harmless pool (e.g., full 52k Alpaca), then report the corresponding TPR.

#### Key-search attacker

For each harmful prompt \(x\): sample \(k_1..k_K\), compute scores \(s_i\), and count bypass if \(\min_i s_i < \tau\). Report TPR@FPR as a function of \(K\in\{1,2,4,8,16,32,64\}\).

### Key Innovations

1. Frames key personalization as an attacker-controlled sampling space for evasion.
2. Uses a distribution-matched baseline: the monitor is trained on encrypted traffic with random keys.
3. Produces a decision-relevant curve: TPR@FPR vs key-search budget K.

---

## Related Work

### Field Overview

Learned embedding obfuscation methods (OSNIP, Stained Glass Transform) add stochasticity for privacy, while activation monitors use internal representations for low-latency safety filtering. Obfuscated-activation attacks show that monitors can be evaded via explicit optimization. We test whether key-conditioned privacy mechanisms create a cheaper “search over randomness” bypass channel.

### Related Papers

- **[OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma-in-LLM-Inference-via-Obfuscated-Semantic-Null-Space/meta/meta_info.txt)**: Key-conditioned embedding obfuscation with utility-preserving KL objective.
- **[Stained Glass Transform](./references/Learning-Obfuscations-Of-LLM-Embedding-Sequences-Stained-Glass-Transform/meta/meta_info.txt)**: Stochastic embedding obfuscation with privacy framing beyond KNN.
- **[BeamClean](./references/BeamClean-Language-Aware-Embedding-Reconstruction/meta/meta_info.txt)**: Language-aware inversion; weak attackers can underestimate leakage.
- **[Text Embeddings Reveal (Almost) As Much As Text](./references/Text-Embeddings-Reveal-(Almost)-As-Much-As-Text/meta/meta_info.txt)**: Vec2Text embedding inversion demonstrates significant leakage.
- **[Activation Monitoring](./references/Activation-Monitoring-Advantages-of-Using-Internal-Representations-for-LLM-Oversight/meta/meta_info.txt)**: Harmful-request detection from activations; emphasizes low-FPR metrics.
- **[Obfuscated Activations](./references/Obfuscated-Activations-Bypass-LLM-Latent-Space-Defenses/meta/meta_info.txt)**: Optimization-based evasion attacks against latent-space defenses.
- **[HarmBench](https://arxiv.org/abs/2402.04249)**: Standard dataset for harmful prompt evaluation.
- **[LlamaGuard](https://arxiv.org/abs/2312.06674)**: LLM-based input/output safety classifier baseline.
- **[Best-of-N Jailbreaking](https://arxiv.org/abs/2412.03556)**: Sampling-based safety bypass in text space.
- **[The Attacker Moves Second](https://arxiv.org/abs/2406.01269)**: Adaptive attacks bypass static jailbreak defenses.
- **[Circuit Breakers](https://arxiv.org/abs/2406.04313)**: Activation-space interventions for safety robustness.
- **[Latent Adversarial Training](https://arxiv.org/abs/2407.05807)**: Training for robustness to latent perturbations.
- **[Split-and-Denoise](https://arxiv.org/abs/2310.09130)**: Local noise mechanisms for private split inference.
- **[Prompt Inversion Attack](https://arxiv.org/abs/2503.09022)**: Prompt recovery from intermediate activations.
- **[Depth Gives a False Sense of Privacy](https://arxiv.org/abs/2507.16372)**: Deep-layer activations can remain invertible.
- **[Language Models are Injective](https://arxiv.org/abs/2401.01948)**: Theory arguing decoder-only LMs are (almost surely) invertible.
- **[EncryptedLLM](https://arxiv.org/abs/2501.01940)**: FHE-based private LLM inference alternative.
- **[NoPeek](https://arxiv.org/abs/2008.03248)**: Reducing leakage when sharing activations.
- **[IRON](https://arxiv.org/abs/2010.09595)**: Cryptographic private transformer inference.
- **[Holistic Undesired Content Detection](https://arxiv.org/abs/2302.04288)**: Deployment-oriented content moderation considerations.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Evaluation | Key limitation |
|---|---|---|---|---|
| Key-conditioned obfuscation | Many embeddings per prompt via keys | OSNIP, Stained Glass | inversion ASR, utility | attacker can search keys |
| Activation monitors | Safety classification from activations | Activation Monitoring | TPR@FPR | may be attackable |
| Sampling-based bypass | Repeat until filter fails | Best-of-N jailbreak | ASR vs N | mitigations external |

### Closest Prior Work

- OSNIP proposes key-conditioned obfuscation but does not study server-side monitor evasion.
- Activation Monitoring studies monitor robustness but not attacker control over privacy keys.
- Obfuscated Activations studies optimization-based evasion; we test cheap sampling-based evasion.

### Novelty Kill Search Summary

Local search over proposals/drafts for “key search/key shopping + OSNIP + monitor”, plus web queries including “key-conditioned obfuscation activation monitor evasion”, “OSNIP key-conditioned personalization attack”, and “best-of-K key search safety monitor”. As of 2026-02-25, no directly matching work was found (details in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we test |
|---|---|---|---|
| OSNIP | Key-conditioned privacy transform | No safety-monitor eval | Key-search bypass |
| Activation Monitoring | Activation probes for safety | No privacy-key surface | Monitor trained on encrypted traffic |
| Obfuscated Activations | Optimize to evade monitors | Higher attacker effort | Cheap sampling-based bypass |
| Best-of-N jailbreak | Sampling bypass in text space | Not privacy-related | Sampling bypass in key space |

---

## Experiments

### Experimental Setup

**Task:** harmful-request detection (input filtering).

**Datasets:** HarmBench (harmful prompts) vs Alpaca (harmless prompts), following Activation Monitoring.

**Reference point:** Activation Monitoring reports for Llama-2-7B an MLP activation monitor achieves TPR 98.98% at FPR=1e-3 and TPR 89.15% at FPR=1e-4 on HarmBench-vs-Alpaca (Table 1 in **[Activation Monitoring](./references/Activation-Monitoring-Advantages-of-Using-Internal-Representations-for-LLM-Oversight/sections/TPR%20at%20FPR.md)**).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Open weights; no Azure filtering |

**Training Data:**

| Dataset | Purpose | Size | Download Link |
|---|---|---:|---|
| HarmBench | Harmful inputs | ~1.5k–3k | https://huggingface.co/datasets/Anthropic/harmbench |
| Alpaca | Harmless inputs + FPR calibration | 52k | https://huggingface.co/datasets/tatsu-lab/alpaca |

**Resource Estimate**:
- **Compute budget**: ≤400 A100 GPU-hours.
  - Encryptor training: ≤3k steps, ≤10k prompts, seq len ≤128.
  - Probe training: cache activations.
  - Key-search eval: K∈{1..64} on a capped test set.
- **GPU memory**: 1×A100 80GB.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | What it evaluates | Metrics | Evaluation |
|---|---|---|---|
| HarmBench-vs-Alpaca | Harmful-request classification | TPR@FPR, AUROC | FPR fixed at 1e-3 and 1e-4 |
| Key-search curve | Key-shopping evasion | TPR@FPR vs K | K∈{1,2,4,8,16,32,64} |

### Main Results

| Method | Base Model | Benchmark | TPR@1e-3FPR (mean±std) | TPR@1e-4FPR (mean±std) | Notes |
|---|---|---|---:|---:|---|
| Encrypted monitor (K=1) | Qwen2.5-7B | HarmBench-vs-Alpaca | TBD | TBD | random key |
| **Encrypted monitor + key search (K=32)** | Qwen2.5-7B | HarmBench-vs-Alpaca | TBD | TBD | min over 32 keys |

Also report one plot: TPR@FPR vs K for K∈{1,2,4,8,16,32,64}.

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| \(\lambda_2=0\) | Remove key-diversity loss | smaller key variance → weaker key-search bypass |

### Experimental Rigor

- **Seeds**: 3 seeds for encryptor training and probe training: `seeds=[42,123,456]`.
- **Thresholding**: choose \(\tau\) on harmless-only data; never tune on harmful prompts.
- **Baseline strength check**: if K=1 TPR@1e-3FPR < 0.80, report explicitly (the monitor may already be too weak).
- **Detectability note**: report degradation at small K (1→4→8→32) to quantify attacker effort.

---

## Success Criteria

**Hypothesis**: Key-conditioned obfuscation enables practical key-search bypass of encrypted-traffic activation monitors.

**Decision Rule**:
- **Proceed** if TPR@1e-3FPR drops by **≥20 points** from K=1 to K=32 while meeting the utility (KL) and privacy (KNN ASR@10) bounds.
- **Refute** if TPR drop is **≤5 points** even at K=64.
- **Pivot** if K=1 TPR@1e-3FPR < 0.80: try a different monitoring layer choice or base model.

---

## Impact Statement

If key search significantly reduces detection at low FPR, then key-conditioned embedding obfuscation creates a low-cost evasion channel against activation-based content moderation in privacy-preserving inference. This would change deployment guidance for “private + safe” LLM services: keys must be restricted/committed or deployments must assume attackers can search over personalization.

---

## References

- [OSNIP](./references/OSNIP-Breaking-the-Privacy-Utility-Efficiency-Trilemma-in-LLM-Inference-via-Obfuscated-Semantic-Null-Space/meta/meta_info.txt)
- [Activation Monitoring](./references/Activation-Monitoring-Advantages-of-Using-Internal-Representations-for-LLM-Oversight/meta/meta_info.txt)
- [Obfuscated Activations](./references/Obfuscated-Activations-Bypass-LLM-Latent-Space-Defenses/meta/meta_info.txt)
- [Stained Glass Transform](./references/Learning-Obfuscations-Of-LLM-Embedding-Sequences-Stained-Glass-Transform/meta/meta_info.txt)
- [BeamClean](./references/BeamClean-Language-Aware-Embedding-Reconstruction/meta/meta_info.txt)
- [Text Embeddings Reveal (Almost) As Much As Text](./references/Text-Embeddings-Reveal-(Almost)-As-Much-As-Text/meta/meta_info.txt)
- [HarmBench](https://arxiv.org/abs/2402.04249)
- [LlamaGuard](https://arxiv.org/abs/2312.06674)
- [Best-of-N Jailbreaking](https://arxiv.org/abs/2412.03556)
- [The Attacker Moves Second](https://arxiv.org/abs/2406.01269)
- [Circuit Breakers](https://arxiv.org/abs/2406.04313)
- [Latent Adversarial Training](https://arxiv.org/abs/2407.05807)
- [Split-and-Denoise](https://arxiv.org/abs/2310.09130)
- [Prompt Inversion Attack](https://arxiv.org/abs/2503.09022)
- [Depth Gives a False Sense of Privacy](https://arxiv.org/abs/2507.16372)
- [Language Models are Injective](https://arxiv.org/abs/2401.01948)
- [EncryptedLLM](https://arxiv.org/abs/2501.01940)
- [NoPeek](https://arxiv.org/abs/2008.03248)
- [IRON](https://arxiv.org/abs/2010.09595)
- [Holistic Undesired Content Detection](https://arxiv.org/abs/2302.04288)
