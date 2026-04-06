# untitled

# Interval-Calibrated Noisy Quantization: A Parameter-Free σ Rule for Mitigating Quantization-Gap Attacks

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Post-training weight quantization (e.g., 8-bit or 4-bit weights) is widely used to reduce the memory and latency cost of deploying large language models (LLMs) on commodity hardware. However, recent work shows that quantization can create a security risk: an attacker can distribute a full-precision model that appears benign under standard evaluation, yet exhibits harmful behavior only after users quantize it for deployment.

Egashira et al. show this “quantization-gap” threat—models that look benign at full precision but become malicious after quantization—for zero-shot quantizers such as LLM.int8() (an 8-bit quantization method from the bitsandbytes library), NF4, and FP4. In a vulnerable code generation setting, the full-precision attacked Phi-2 model achieves 98.0% secure code generation while its LLM.int8() quantization drops to 18.5% secure code ([Exploiting LLM Quantization](./references/Exploiting-LLM-Quantization/meta/meta_info.txt)). Egashira et al. further show a practical attack on GGUF k-quants (used by llama.cpp / Ollama), demonstrating that even optimization-based quantizers can be exploited ([Mind the Gap](./references/Mind-the-Gap-A-Practical-Attack-on-GGUF-Quantization/meta/meta_info.txt)).

A simple mitigation proposed in both papers is to add Gaussian noise to weights before quantization (“noisy quantization”). This disrupts the attacker’s quantization-preserving constraints and can remove the gap while keeping utility nearly unchanged. However, the mitigation currently requires per-model tuning of the noise scale σ: for example, in the GGUF setting, Qwen2.5-3B requires σ≈1e-3 while Llama3.1-8B already shows utility loss at σ=1e-3 and prefers σ≈1e-4 ([Mind the Gap](./references/Mind-the-Gap-A-Practical-Attack-on-GGUF-Quantization/meta/meta_info.txt)). In practice, per-model σ sweeps are a barrier to deploying noisy quantization in standard quantization libraries and model hubs.

### The Problem

The practical problem is:

> **How can we choose the Gaussian noise standard deviation σ for noisy quantization without running a per-model grid search over security and utility metrics?**

Existing work demonstrates that a “sweet spot” exists (enough noise to break the attack, not so much that utility collapses), but provides no a priori recipe for selecting σ.

Adjacent quantization-robustness work often scales injected noise by a quantization step size Δ during fine-tuning, but this is not the same setting:
- **GIFT-SW** injects Δ-scaled Gaussian noise during fine-tuning to improve robustness of quantized models, not to mitigate quantization-gap attacks, and it still relies on training-time intervention ([GIFT-SW](./references/GIFT-SW-Gaussian-noise-Injected-Fine-Tuning-of-Salient-Weights-for-LLMs/meta/meta_info.txt)).
- **NPFT** uses perturbations in [-Δ/2, Δ/2] as a proxy for quantization bin width for Hessian-trace regularization during fine-tuning, again in a training-time robustness setting ([NPFT](./references/Taming-Sensitive-Weights-Noise-Perturbation-Fine-tuning-for-Robust-LLM-Quantization/meta/meta_info.txt)).

We focus on the deployment-relevant, training-free setting: given a full-precision model and a target quantizer, choose σ deterministically from model weights + quantizer internals only.

### Key Insight and Hypothesis

**Key insight**: For a given quantizer, each weight has a quantization-preserving interval: a range of full-precision values that map to the same quantized value. Quantization-gap attacks explicitly exploit these intervals as feasible regions to hide malicious behavior that only survives quantization. A noise scale σ that is too small relative to typical interval half-widths will not change many quantized weights and will fail to disrupt the attack. A σ that is too large will change many quantized weights and can degrade utility.

**Hypothesis**: A robust statistic of the model’s quantization-preserving interval widths (computed from the benign full-precision weights) is sufficient to set σ near the optimal “sweet spot” for noisy quantization, matching the security–utility tradeoff of a small σ grid search.

Why we could be wrong: interval width might not reflect “semantic sensitivity” of weights (some narrow-interval weights may be unimportant, and some wide-interval weights may be unimportant), so σ chosen from interval statistics might either (i) be too small to break the attack or (ii) be large enough to break the attack but degrade utility.

---

## Proposed Approach

### Overview

We propose a **parameter-free σ estimator** for noisy quantization:

1. Compute quantization-preserving intervals for each weight tensor using the same interval computation used by quantization-gap attacks (e.g., `compute_box_int8` for LLM.int8()).
2. Summarize each tensor’s typical half-interval width using a median.
3. Aggregate across tensors with a median to produce a single σ̂ for the model.
4. Quantize the model once with additive Gaussian noise `N(0, σ̂^2)`.

This produces a deterministic σ̂ per model without running any security or utility evaluations.

### Method Details

We define σ̂ for LLM.int8() as:

- Let the model have a set of weight tensors \(\{W_\ell\}\) (we restrict to `torch.nn.Linear` weight matrices for the MVP).
- For each tensor \(W_\ell\), compute element-wise quantization-preserving bounds \((b^{\min}_\ell, b^{\max}_\ell)\) using the interval computation for LLM.int8() (as implemented in the ETH quantization-attack codebase).
- Define a per-tensor robust half-width statistic:

\[
  h_\ell = \mathrm{median}_{i \in \text{entries}(W_\ell)} \left( \frac{b^{\max}_{\ell,i} - b^{\min}_{\ell,i}}{2} \right).
\]

- Define the model-level σ̂ as:

\[
  \hat\sigma = \mathrm{median}_{\ell}(h_\ell).
\]

We intentionally do **not** introduce an additional scaling coefficient (no κ) in the MVP, to avoid creating another tuning knob.

### Key Innovations

1. **Training-free σ selection for quantization-gap mitigation**: σ̂ is computed from model weights + quantizer internals only (no fine-tuning, no labeled safety data, no evaluation-based tuning).
2. **Interval-statistics framing**: uses the same quantization interval structure that enables quantization-gap attacks as the basis for a deterministic mitigation parameter.
3. **Decouples “choose σ” from “measure security”**: unlike grid search, σ̂ does not require running CodeQL, HumanEval, or other benchmarks.

---

## Related Work

### Field Overview

Quantization security can be organized into three overlapping areas: (i) attacks that exploit post-training transformations (especially quantization) to activate dormant malicious behaviors, (ii) mitigations that modify quantization or the model to prevent such behaviors, and (iii) broader quantization-safety work that tries to preserve alignment properties under low precision.

Quantization-gap attacks on LLMs were introduced by Egashira et al. for zero-shot quantization methods (LLM.int8/NF4/FP4) and extended to GGUF k-quants for deployment-realistic CPU inference. Subsequent work strengthens attack training (e.g., contrastive objectives) and increases durability under downstream fine-tuning. In parallel, safety-focused quantization methods aim to preserve refusal behavior, reduce jailbreak susceptibility, or patch safety regressions after quantization.

Noisy quantization (adding noise before quantizing) is a simple mitigation that can disrupt quantization-preserving constraints, but it introduces a practical parameter-selection problem (σ), which is the focus of this proposal.

### Related Papers

- **[Exploiting LLM Quantization](./references/Exploiting-LLM-Quantization/meta/meta_info.txt)**: Introduces quantization-gap attacks on zero-shot LLM quantizers and shows Gaussian noise before quantization can remove the gap at σ=1e-3 for Phi-2.
- **[Mind the Gap: A Practical Attack on GGUF Quantization](./references/Mind-the-Gap-A-Practical-Attack-on-GGUF-Quantization/meta/meta_info.txt)**: Extends quantization-gap attacks to GGUF k-quants and shows the “best” σ differs by model.
- **[Durable Quantization Conditioned Misalignment Attack on Large Language Models (Q-Misalign)](./references/DURABLE-QUANTIZATION-CONDITIONED-MISALIGN-MENT-ATTACK-ON-LARGE-LANGUAGE-MODELS/meta/meta_info.txt)**: Builds durable quantization-conditioned jailbreak attacks that persist under downstream fine-tuning.
- **[Adversarial Contrastive Learning for LLM Quantization Attacks](./references/Adversarial-Contrastive-Learning-for-LLM-Quantization-Attacks/meta/meta_info.txt)**: Improves quantization-gap attacks using triplet-style contrastive objectives and scalable constrained optimization.
- **[GIFT-SW: Gaussian noise Injected Fine-Tuning of Salient Weights for LLMs](./references/GIFT-SW-Gaussian-noise-Injected-Fine-Tuning-of-Salient-Weights-for-LLMs/meta/meta_info.txt)**: Uses Δ-scaled Gaussian noise during fine-tuning for quantization robustness; motivates step-size-scaled noise ideas.
- **[Taming Sensitive Weights: Noise Perturbation Fine-tuning for Robust LLM Quantization (NPFT)](./references/Taming-Sensitive-Weights-Noise-Perturbation-Fine-tuning-for-Robust-LLM-Quantization/meta/meta_info.txt)**: Uses perturbations tied to quantization bin width as a proxy for Hessian-trace regularization.
- **[Quantization Backdoors to Deep Learning Commercial Frameworks](https://ieeexplore.ieee.org/document/10113762)**: Early study of quantization-activated backdoors (vision models) and noise-based mitigation ideas.
- **[Large Language Models for Code: Security Hardening and Adversarial Testing](https://arxiv.org/abs/2308.04726)**: Establishes static-analysis-based evaluation of LLM-generated code security and datasets of vulnerable/secure code.
- **[Instruction Tuning for Secure Code Generation (SafeCoder)](https://arxiv.org/abs/2402.09497)**: Proposes SafeCoder training objectives and CodeQL-based evaluation for secure code generation.
- **[On the Exploitability of Instruction Tuning](https://arxiv.org/abs/2309.00230)**: Studies poisoning-based behaviors such as over-refusal; used in ELQ evaluation.
- **[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)**: Introduces LLM.int8 quantization widely used in HF ecosystems.
- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)**: Popularizes NF4 + double quantization and motivates why zero-shot 4-bit quantization is widely used.
- **[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)**: Optimization-based PTQ that uses calibration data; relevant alternative quantization family.
- **[AWQ: Activation-Aware Weight Quantization](https://arxiv.org/abs/2306.00978)**: PTQ method using activation statistics; relevant as alternative deployment quantization.
- **[SmoothQuant](https://arxiv.org/abs/2211.10438)**: Activation scaling approach for quantization accuracy; relevant for broader quantization pipeline.
- **[SqueezeLLM](https://arxiv.org/abs/2306.07629)**: Non-uniform quantization with sensitivity; relevant for quantization robustness and interval structure.
- **[SpQR](https://arxiv.org/abs/2306.03078)**: Mixed-precision quantization focusing on outliers; relevant to interval widths and weight distribution tails.
- **[Nearest is Not Dearest: Towards Practical Defense against Quantization-conditioned Backdoor Attacks](https://arxiv.org/abs/2405.12725)**: Defense for quantization-conditioned backdoors via changed rounding; adjacent mitigation family.
- **[How Model Compression Breaks Backdoor Defenses](https://arxiv.org/abs/2512.06243)**: Shows backdoor defenses can degrade under compression; motivates quantization-specific mitigations.
- **[SafeQuant: LLM Safety Analysis via Quantized Gradient Inspection](https://aclanthology.org/2025.naacl-long.127/)**: White-box safety analysis for quantized LLMs via gradient inspection.
- **[Alignment-Aware Quantization for LLM Safety](https://arxiv.org/abs/2511.07842)**: Preserves safety alignment during quantization; relevant quantization-safety baseline family.
- **[Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized LLMs](https://arxiv.org/abs/2506.20251)**: Post-quantization safety patching approach.
- **[Q-Realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment](https://arxiv.org/abs/2601.08089)**: Uses quantization as an opportunity to recover safety.
- **[Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection](https://arxiv.org/abs/2601.12033)**: Mixed-precision protection of critical weights for fairness/safety.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Quantization-gap attacks | Create models that behave benignly in FP but maliciously when quantized | Egashira et al. 2024; Egashira et al. 2025; Q-Misalign; ACL | Code security via CodeQL; refusal/content injection; MMLU/TQA/HE/MBPP | Hard to detect in FP; attack strength depends on quantizer/interval structure |
| Noisy quantization | Add randomness before quantization to disrupt quantization-preserving constraints | Egashira et al. 2024; Egashira et al. 2025; Ma et al. 2023 | Same as above | Requires tuning σ; possible utility degradation |
| Training-time robustness via noise | Inject Δ-scaled noise during fine-tuning to improve quantized utility | GIFT-SW; NPFT; QAT methods | LM perplexity, zero-shot tasks | Not designed for supply-chain threat model; requires training |
| Rounding/quantizer modification mitigations | Change rounding to break attack triggers while preserving activations | Nearest is Not Dearest (EFRAP) | Vision backdoor benchmarks | Not studied for LLM quantization-gap threat |
| Quantization-time safety preservation | Preserve refusal/alignment metrics during quantization or patch after quantization | AAQ; Q-resafe; Q-Realign; Critical Weight Protection; SafeQuant | Jailbreak / refusal / safety benchmarks | Often requires additional data or training; not targeted at quantization-gap supply-chain threat |

### Closest Prior Work

- **Egashira et al. 2024 (ELQ)** ([paper](./references/Exploiting-LLM-Quantization/meta/meta_info.txt)): Demonstrates quantization-gap attacks on LLM.int8/NF4/FP4 and shows Gaussian noise before quantization can remove the gap for Phi-2 at σ=1e-3, but requires manual σ choice.
- **Egashira et al. 2025 (Mind the Gap)** ([paper](./references/Mind-the-Gap-A-Practical-Attack-on-GGUF-Quantization/meta/meta_info.txt)): Extends attacks to GGUF and reports that optimal σ differs by model; explicitly calls for a recipe to choose σ.
- **GIFT-SW** ([paper](./references/GIFT-SW-Gaussian-noise-Injected-Fine-Tuning-of-Salient-Weights-for-LLMs/meta/meta_info.txt)): Provides a Δ-scaled noise principle during training-time fine-tuning for quantization robustness, but does not address the training-free, quantization-gap security setting.
- **NPFT** ([paper](./references/Taming-Sensitive-Weights-Noise-Perturbation-Fine-tuning-for-Robust-LLM-Quantization/meta/meta_info.txt)): Uses Δ-tied perturbations for robustness via fine-tuning; suggests Δ matters but does not propose a training-free σ rule.

**Novelty Kill Search Summary:** Searched for “interval width sigma noisy quantization attack”, “GGUF Gaussian noise sigma recipe”, “automatic sigma selection Gaussian noise quantization backdoor”, “interval-matched noise quantization”, and checked local KB + other agents’ drafts for prior proposals. No prior work was found (as of 2026-02-16) that deterministically selects σ for LLM quantization-gap mitigation using quantization interval statistics computed from benign weights.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| ELQ (Egashira 2024) | Shows Gaussian-noise-before-quantization can mitigate attack | Needs per-model σ choice | Deterministic σ̂ from interval stats | Removes tuning burden; still breaks quantization-preserving constraints |
| Mind the Gap (Egashira 2025) | Extends attack to GGUF; shows σ differs per model | No σ recipe | Same as above | Makes mitigation deployable in practice |
| GIFT-SW | Δ-scaled noise during fine-tuning for robustness | Requires training; different threat model | Training-free σ̂ | Lower cost and directly targets supply-chain threat |
| NPFT | Δ-tied perturbations during fine-tuning for robust quantization | Requires training; hyperparameter tuning remains | Training-free σ̂ | Avoids tuning loops and safety-label dependence |
| Nearest is Not Dearest (EFRAP) | Changes rounding to mitigate quantization-conditioned backdoors | Not evaluated on LLM quantization-gap attacks | Keep quantizer fixed; pick σ deterministically | Minimal change to deployment toolchains |

---

## Experiments

### Experimental Setup

**Primary setting (MVP):** vulnerable code generation quantization-gap attack in ELQ.

We follow Egashira et al. (ELQ) for the attack and evaluation pipeline:
- **Attack pipeline**: use the ETH `llm-quantization-attack` repository to reproduce the attacked model for one base model (target: Phi-2; fallback: StarCoder-1b if Phi-2 reproduction is too costly).
- **Quantizer**: LLM.int8() (as in ELQ noise defense table).
- **Noisy quantization**: add Gaussian noise to weights once before quantization to produce a single quantized model per seed.

**Core comparison (exactly 3 conditions):**
1. **No noise**: σ=0.
2. **Grid-search σ (strong baseline)**: σ chosen from {1e-5, 5e-5, 1e-4, 5e-4, 1e-3} on a held-out validation split by maximizing code security subject to utility constraints.
3. **Interval-calibrated σ̂ (ours)**: σ̂ computed from benign FP32 weights only using the median half-interval-width estimator defined above.

**Selection protocol for σ (baseline #2) to avoid leakage:**
- Split the code-security evaluation prompts into **validation** and **test** subsets.
- Choose σ on validation to maximize code security subject to:
  - MBPP pass@1 drop ≤ 1.0 point vs σ=0 quantized model
  - MMLU drop ≤ 1.0 point vs σ=0 quantized model
- Report final metrics on the held-out test subset (and report full HumanEval and TruthfulQA, which are not used in σ selection).

**Seeds / randomness:**
- Because noisy quantization is stochastic, run ≥3 noise seeds per condition: `seeds=[0,1,2]`.
- For HumanEval/MBPP pass@1, use temperature 0.2 as in ELQ; use fixed generation seeds per run.

**Baseline Ladder (REQUIRED):**
- **Level 0 (trivial)**: σ=0 (no noise).
- **Level 1 (simple heuristic)**: fixed σ=1e-3 (the common value used in ELQ for Phi-2) as a sanity check.
- **Level 2 (strong baseline)**: σ grid-search baseline described above.
- **Level 3 (closest method family)**: σ̂ interval-calibrated noisy quantization (ours).

Prompting and inference-time scaling are not the primary axis here (we evaluate fixed benchmark prompts and the intervention is at quantization-time), but we will include an optional diagnostic: for σ=0 quantized model, run best-of-5 decoding on a small subset and measure whether selecting the first code sample that passes CodeQL recovers security, to estimate how much “inference-time verification” could close the gap at extra compute cost.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| microsoft/phi-2 | 2.7B | https://huggingface.co/microsoft/phi-2 | Primary target for reproducing ELQ noise defense setting |
| bigcode/starcoderbase-1b | 1B | https://huggingface.co/bigcode/starcoderbase-1b | Fallback target if Phi-2 reproduction is too costly |

**Training Data (attack reproduction only):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Code-Alpaca | D_instr in SafeCoder | N/A | https://github.com/sahil280114/codealpaca | N/A |
| Code security dataset subset (4 Python CWEs) | D_vul / D_sec in SafeCoder | N/A | https://arxiv.org/abs/2308.04726 | N/A |

**Resource Estimate**:
- **Attack reproduction**: ELQ reports PGD-only repair runtime 1h24m for StarCoder-1b and 41h21m with QA regularizer (Table 4); we use PGD-only variant.
- **Budget for MVP**: target ≤ 200 GPU-hours total, including attack reproduction + quantization + evaluation across 3 noise seeds and the σ grid.
- **Peak memory**: Phi-2 and StarCoder-1b should fit on 1×A100 80GB for LoRA-style fine-tuning; if full fine-tuning is required, use FSDP across 2–4 GPUs.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SecurityEval / SafeCoder code-security eval | Generates code for prompts targeting specific vulnerabilities and runs CodeQL (a static analysis tool) to detect CWEs (Common Weakness Enumerations) | Secure code rate (%) (higher is safer) | val/test (split by prompts) | via ELQ repo + SafeCoder | https://github.com/eth-sri/llm-quantization-attack |
| HumanEval | Code generation benchmark with unit tests | pass@1 (higher is better) | standard | https://github.com/openai/human-eval | standard eval |
| MBPP | Code generation benchmark with tests | pass@1 (higher is better) | standard | https://github.com/google-research/google-research/tree/master/mbpp | standard eval |
| MMLU | Multiple-choice knowledge benchmark | accuracy (higher is better) | standard | https://github.com/hendrycks/test | standard eval |
| TruthfulQA | Truthfulness benchmark (multiple-choice) | MC2 accuracy (higher is better) | standard | https://github.com/sylinrl/TruthfulQA | standard eval |

### Main Results

#### Results Table

Published baseline numbers from ELQ (Noise Defense Table 5; Phi-2 + LLM.int8) are included for reference and to validate reproduction.

| Method | Base Model | Quantizer | Security benchmark | Code security (FP32 / quant) | HumanEval pass@1 (FP32 / quant) | TruthfulQA (FP32 / quant) | Source | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| No noise (σ=0) | Phi-2 | LLM.int8 | SafeCoder/SecurityEval | 98.0 / 18.5 | 48.7 / 43.6 | 40.6 / 36.9 | [ELQ Table 5](./references/Exploiting-LLM-Quantization/sections/Noise\ Defense.md) | Published (1 run) |
| Fixed σ=1e-4 | Phi-2 | LLM.int8 | SafeCoder/SecurityEval | 97.9 / 32.6 | 48.8 / 47.0 | 40.4 / 37.3 | [ELQ Table 5](./references/Exploiting-LLM-Quantization/sections/Noise\ Defense.md) | Published (1 run) |
| Fixed σ=1e-3 | Phi-2 | LLM.int8 | SafeCoder/SecurityEval | 98.4 / 97.5 | 48.0 / 47.8 | 40.4 / 39.7 | [ELQ Table 5](./references/Exploiting-LLM-Quantization/sections/Noise\ Defense.md) | Published (1 run) |
| **Grid-search σ (baseline)** | Phi-2 | LLM.int8 | SafeCoder/SecurityEval | **TBD** | **TBD** | **TBD** | - | Selected on val split; test-only reporting |
| **Interval-calibrated σ̂ (ours)** | Phi-2 | LLM.int8 | SafeCoder/SecurityEval | **TBD** | **TBD** | **TBD** | - | No eval-based tuning |

### Ablation Studies

(Only if budget allows; not required for deciding the main claim.)

| Variant | What’s changed | Expected finding |
|---|---|---|
| σ̂ per-layer vs global | Use h_ℓ per layer instead of median across layers | If some layers dominate vulnerability, per-layer may improve |
| Alternate statistic | Replace median with 75th percentile of half-widths | Tests robustness to long-tailed interval distributions |

### Experimental Rigor

**Confounders and controls**:
- **Attack not optimized for noisy quantization**: a win might reflect attacker not using expectation-over-noise. We explicitly scope the claim to the standard attack pipeline as in ELQ/Mind-the-Gap.
- **Evaluation leakage in σ selection**: use a strict val/test split for σ grid search; do not tune σ on test.
- **Randomness from noise + decoding**: use ≥3 seeds for noisy quantization; fix decoding parameters and report mean±std.

---

## Success Criteria

**Hypothesis**: Interval-calibrated σ̂ will achieve a security–utility tradeoff close to the best σ found by a small grid search, without any evaluation-based tuning.

**Decision Rule**:
- **Proceed/Continue** if, on the held-out test split, σ̂ achieves:
  - Code security within **5 percentage points** (absolute) of the grid-search σ baseline, and
  - HumanEval pass@1 drop ≤ **(grid-search drop + 0.5 points)**.
- **Pivot** if σ̂ reliably improves over σ=0 but underperforms grid-search σ by >5 points; try a single alternative statistic (e.g., 75th percentile of half-widths) as a one-step modification.
- **Refute** if σ̂ is consistently worse than σ=0 on code security, or if σ̂ causes utility drops comparable to σ=1e-2 behavior in ELQ (large HumanEval collapse), indicating interval width is not a useful proxy.

---

## Impact Statement

If successful, this provides a deployable recipe for noisy quantization that does not require per-model σ sweeps, enabling quantization libraries and model hubs to apply noisy quantization as a default mitigation against quantization-gap attacks with minimal added complexity.

---

## References

- [Exploiting LLM Quantization](./references/Exploiting-LLM-Quantization/meta/meta_info.txt) - Egashira et al., 2024
- [Mind the Gap: A Practical Attack on GGUF Quantization](./references/Mind-the-Gap-A-Practical-Attack-on-GGUF-Quantization/meta/meta_info.txt) - Egashira et al., 2025
- [GIFT-SW: Gaussian noise Injected Fine-Tuning of Salient Weights for LLMs](./references/GIFT-SW-Gaussian-noise-Injected-Fine-Tuning-of-Salient-Weights-for-LLMs/meta/meta_info.txt) - Zhelnin et al., 2024
- [Taming Sensitive Weights: Noise Perturbation Fine-tuning for Robust LLM Quantization](./references/Taming-Sensitive-Weights-Noise-Perturbation-Fine-tuning-for-Robust-LLM-Quantization/meta/meta_info.txt) - Wang & Yang, 2024
- [Durable Quantization Conditioned Misalignment Attack on Large Language Models](./references/DURABLE-QUANTIZATION-CONDITIONED-MISALIGN-MENT-ATTACK-ON-LARGE-LANGUAGE-MODELS/meta/meta_info.txt) - Dong et al., 2025
- [Adversarial Contrastive Learning for LLM Quantization Attacks](./references/Adversarial-Contrastive-Learning-for-LLM-Quantization-Attacks/meta/meta_info.txt) - Song et al., 2025
- [Quantization Backdoors to Deep Learning Commercial Frameworks](https://ieeexplore.ieee.org/document/10113762) - Ma et al., 2023
- [Nearest is Not Dearest: Towards Practical Defense against Quantization-conditioned Backdoor Attacks](https://arxiv.org/abs/2405.12725) - Li et al., 2024
- [Large Language Models for Code: Security Hardening and Adversarial Testing](https://arxiv.org/abs/2308.04726) - He & Vechev, 2023
- [Instruction Tuning for Secure Code Generation](https://arxiv.org/abs/2402.09497) - He et al., 2024
- [On the Exploitability of Instruction Tuning](https://arxiv.org/abs/2309.00230) - Shu et al., 2023
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) - Dettmers et al., 2022
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) - Frantar et al., 2022
- [AWQ: Activation-Aware Weight Quantization](https://arxiv.org/abs/2306.00978) - Lin et al., 2023
- [SmoothQuant](https://arxiv.org/abs/2211.10438) - Xiao et al., 2022
- [SqueezeLLM](https://arxiv.org/abs/2306.07629) - Kim et al., 2023
- [SpQR](https://arxiv.org/abs/2306.03078) - Dettmers et al., 2023
- [How Model Compression Breaks Backdoor Defenses](https://arxiv.org/abs/2512.06243) - (authors), 2025
- [SafeQuant: LLM Safety Analysis via Quantized Gradient Inspection](https://aclanthology.org/2025.naacl-long.127/) - (authors), 2025
- [Alignment-Aware Quantization for LLM Safety](https://arxiv.org/abs/2511.07842) - (authors), 2025
- [Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized LLMs](https://arxiv.org/abs/2506.20251) - (authors), 2025
- [Q-Realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment](https://arxiv.org/abs/2601.08089) - (authors), 2026
- [Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection](https://arxiv.org/abs/2601.12033) - (authors), 2026
