# untitled

# Label-Free KL Calibration of APE Hyperparameters via Sequential-Teacher Matching

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Context-augmented generation (CAG) systems (e.g., retrieval-augmented generation (RAG) and in-context learning (ICL)) often spend most of their latency on **prefill**: the phase where the model encodes the input context into key-value (KV) representations, before generating a short answer. For example, APE (Adaptive Parallel Encoding) reports that for a 128K-token context, sequential prefilling takes 17 seconds while generating 256 tokens takes 6 seconds (on H100) — making prefill the bottleneck in many real deployments.

A promising approach is **parallel context encoding**: encode each retrieved chunk independently (so chunks do not attend to each other during encoding), then let the query attend to all cached chunk representations during generation. This enables chunk-level reuse and can reduce prefill cost substantially. However, naive parallel encoding often degrades quality because the model was trained under full self-attention over a single sequential stream.

APE is an important step because it shows a **training-free** way to recover most of the quality of sequential encoding by aligning attention distributions between sequential and parallel encoding using three simple knobs: a shared prefix, an attention temperature, and a scaling factor.

### The Problem

Despite APE’s strong results, it has a deployment blocker: **hyperparameter sensitivity**. APE explicitly notes that it is sensitive to choosing the attention temperature and scaling factor, and that “aligning the distribution between sequential and parallel encoding automatically” is challenging when contexts vary in length, quantity, and content.

The current APE practice is to tune these hyperparameters using a **greedy grid search on a labeled validation set** (downstream accuracy/F1). In real systems, this is brittle for two reasons:

1. **Label scarcity / mismatch**: production queries often do not have immediate ground-truth answers that can be used for tuning.
2. **Distribution shift**: the “best” hyperparameters can drift as the number of retrieved chunks, chunk lengths, or domains change.

### Key Insight and Hypothesis

APE’s temperature and scaling factor are motivated as **distribution-alignment knobs**: they are meant to make the attention/logit behavior of parallel encoding resemble sequential encoding.

We hypothesize that this suggests a label-free tuning rule:

> If we choose APE hyperparameters 
> \(\theta=(T,S)\) (attention temperature and scaling factor as implemented in APE) to minimize the divergence between (i) a **sequential-encoding teacher** model’s next-token distribution and (ii) an **APE** model’s next-token distribution on the same unlabeled prompts, then the resulting hyperparameters will be close to label-tuned optima and will transfer better across context-regime shifts (e.g., different numbers of retrieved chunks).

This could fail if “accuracy-optimal” hyperparameters deliberately diverge from sequential behavior (e.g., over-sharpen attention beyond what reduces KL), or if KL matching is too noisy to be stable under small calibration budgets.

---

## Proposed Approach

### Overview

We propose **Sequential-Teacher KL Calibration** for APE:

1. Sample a small unlabeled calibration set of prompts with retrieved contexts.
2. Run a **sequential encoding** pass (teacher) to obtain next-token logits.
3. For each candidate APE hyperparameter \(\theta=(T,S)\), run APE (parallel encoding) on the same prompts and compute the average KL divergence from the teacher.
4. Choose \(\theta\) minimizing this KL objective.

This uses no ground-truth answers — only model-internal distributions.

### Method Details

**Definitions.** For a prompt \(x\) (retrieved contexts + query), let:
- \(p_{\text{seq}}(\cdot\mid x)\) be the next-token distribution under **sequential encoding** (full attention over the concatenated stream).
- \(p_{\text{ape},\theta}(\cdot\mid x)\) be the next-token distribution under **APE** with hyperparameters \(\theta=(T,S)\), where \(T\) is the attention temperature for context KV and \(S\) is the scaling factor applied to the context log-sum-exp term in the APE FlashAttention merge (as in APE Appendix 12.4).

**Calibration objective.** Given an unlabeled calibration set \(\mathcal{C}\) (size \(|\mathcal{C}|\le 32\)):

\[
\theta^* = \arg\min_{\theta\in\Theta}\; \frac{1}{|\mathcal{C}|}\sum_{x\in\mathcal{C}} D_{KL}(p_{\text{seq}}(\cdot\mid x)\;||\;p_{\text{ape},\theta}(\cdot\mid x)).
\]
In plain terms, we pick \((T,S)\) so that APE’s next-token distribution matches what the same model would produce under standard sequential encoding on the same prompts.

We propose a small discrete grid \(\Theta\) (e.g., \(T\in\{0.6,0.7,0.8,0.9,1.0\}\), \(S\in\{0.6,0.7,0.8,0.9,1.0\}\)) to keep tuning cheap.

**Stability check (required).** To ensure the selected hyperparameters are not a high-variance artifact, we will bootstrap \(\mathcal{C}\) (10 resamples) and re-run the argmin selection. If the 95% interval of selected \(T\) (or \(S\)) spans more than 0.1, we treat the method as unstable and refute the approach.

**Transfer setting.** We explicitly evaluate transfer across a context-quantity shift by tuning on top-\(N=8\) retrieved chunks and evaluating on top-\(N=20\) retrieved chunks.

### Key Innovations

- **Label-free hyperparameter selection for training-free parallel encoding**: tuning APE’s inference-time knobs without downstream answers.
- **Mechanism-aligned objective**: directly targets APE’s stated goal (matching sequential vs parallel distributions) rather than task-specific metrics.
- **Decisive transfer test + stability criterion**: a small, automated experiment that either validates the proxy or shows it is unstable / not predictive.

---

## Related Work

### Field Overview

Parallel context encoding methods can be grouped into (i) training-free attention-structure edits (e.g., PCW-style block-diagonal masks and APE-style distribution alignment), and (ii) trainable auxiliary encoders or adapters (e.g., CEPE/CEPED). Recent mechanistic work shows that quality collapse is strongly correlated with attention entropy and irregular key-state norms, motivating distribution-level signals (entropy, logit scale, and KL) as potential tuning targets.

Separately, a large literature on temperature scaling and calibration studies how scaling logits affects uncertainty and robustness, including under distribution shift. These results suggest that temperature-like knobs can have principled optima that depend on the input distribution, supporting adaptive (test-time) selection.

### Related Papers

- **[APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding](./references/APE-Faster-and-Longer-Context-Augmented-Generation-via-Adaptive-Parallel-Encoding/meta/meta_info.txt)**: Training-free attention alignment (shared prefix + temperature + scaling) for parallel encoding; identifies hyperparameter sensitivity as a limitation.
- **[Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding…](./references/Attention-Entropy-is-a-Key-Factor-An-Analysis-of-Parallel-Context-Encoding-with-Full-attention-based-Pre-trained-Language-Models/meta/meta_info.txt)**: Diagnoses PCE failures via attention entropy and proposes shared sinks / selective attention, motivating attention statistics as tuning signals.
- **[Long-Context Language Modeling with Parallel Context Encoding](./references/Long-Context-Language-Modeling-with-Parallel-Context-Encoding/meta/meta_info.txt)**: Trainable encoder + cross-attention (CEPE) and a distillation variant (CEPED) that uses KL during training (adjacent idea but different setting).
- **[PEVLM: Parallel Encoding for Vision-Language Models](./references/PEVLM-Parallel-Encoding-for-Vision-Language-Models/meta/meta_info.txt)**: Parallel encoding adapted to VLMs; reports sensitivity to design hyperparameters (sink/block sizes), reinforcing deployment need for automatic selection.
- **[Optimal Attention Temperature Enhances In-Context Learning under Distribution Shift](./references/Optimal-Attention-Temperature-Enhances-In-Context-Learning-under-Distribution-Shift/meta/meta_info.txt)**: Shows existence of \(\tau_{\text{optimal}}\) under shift; motivates estimating temperature from unlabeled signals.
- **[On the Entropy Calibration of Language Models](./references/On-the-Entropy-Calibration-of-Language-Models/meta/meta_info.txt)**: Studies entropy miscalibration and connects entropy/quality tradeoffs to heavy-tailed data.
- **[Parallel Context Windows for Large Language Models](https://arxiv.org/abs/2212.10947)**: PCW introduces position reuse + block-diagonal masks for parallel encoding.
- **[Revisiting Parallel Context Windows: A Frustratingly Simple Alternative and Chain-of-Thought Deterioration](https://aclanthology.org/2024.findings-acl.523/)**: Shows PCW can harm reasoning and clarifies evaluation pitfalls.
- **[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)**: Identifies attention sink tokens; relevant to shared-prefix/sink fixes.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)**: Fast exact attention kernels used by APE’s efficient implementation.
- **[vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)**: Practical serving engine used in APE’s latency evaluation.
- **[MInference](https://arxiv.org/abs/2407.02490)**: Dynamic sparse attention for prefill acceleration; APE compares speed/quality against it.
- **[FocusLLM: Scaling LLM’s Context by Parallel Decoding](https://arxiv.org/abs/2408.11745)**: Trainable approach for parallel context handling cited by APE.
- **[Block-Attention for Efficient Prefilling](https://arxiv.org/abs/2409.15355)**: Fine-tuned block attention for efficient RAG/KV reuse (requires training).
- **[TurboRAG](https://arxiv.org/abs/2410.07590)**: Precomputes chunk KV for faster RAG serving (system-level alternative).
- **[Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection](https://arxiv.org/abs/2405.16178)**: Selects fewer chunks at inference to reduce attention cost.
- **[PISCO: Pretty Simple Compression for Retrieval-Augmented Generation](https://arxiv.org/abs/2501.16075)**: Trainable compression for RAG contexts.
- **[REFRAG: Rethinking RAG based Decoding](https://arxiv.org/abs/2509.01092)**: Compresses and selectively expands retrieved chunks during decoding.
- **[LongBench](https://arxiv.org/abs/2308.14508)**: Long-context benchmark suite used by APE.
- **[Contriever](https://arxiv.org/abs/2112.09118)**: Dense retriever used by APE for LongBench chunk selection.
- **[ChatQA / ChatRAGBench](https://arxiv.org/abs/2401.10225)**: Conversational QA benchmark family used by APE.
- **[Temperature Scaling for Neural Networks](https://arxiv.org/abs/1706.04599)**: Classic post-hoc temperature calibration baseline.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Training-free PCE (PCW) | Block-diagonal attention + position reuse | PCW (2212.10947), Revisiting PCW (ACL 2024 Findings) | ICL/RAG, synthetic recall | Reasoning deterioration; sensitivity to setup |
| Training-free alignment knobs | Modify attention distribution at inference (prefix/temperature/scale) | APE (2502.05431) | LongBench, ChatRAGBench | Needs hyperparameter tuning |
| Mechanistic stabilization | Diagnose entropy/sink irregularities and apply small fixes | Attention entropy analysis (ACL 2025), attention sinks | LM/recall/RAG | No universal fix; selection hyperparameters |
| Trainable auxiliary encoder | Add encoder + cross-attn; optionally distill | CEPE/CEPED (ACL 2024) | LM, QA, long-context | Requires training; architecture changes |

### Closest Prior Work

- **APE**: tunes hyperparameters by greedy search on labeled validation data; we replace the tuning objective with label-free sequential-teacher KL while keeping inference-time-only changes.
- **Attention entropy analysis (ACL 2025)**: proposes entropy as an indicator of failure and uses sinks/selective attention; we use a stronger distributional signal (next-token KL to a sequential teacher) to *select* APE’s parameters.
- **CEPED (CEPE distillation)**: uses KL during training to distill instruction behavior into an auxiliary-encoder architecture; we do no training and use KL only for inference-time hyperparameter calibration.
- **Optimal attention temperature under shift**: proves \(\tau_{\text{optimal}}\) exists but requires distribution knowledge; we estimate a practical \(\tau\) from unlabeled teacher matching.

**Novelty Kill Search Summary:** Searched for combinations of “APE + KL tuning”, “parallel context encoding + distillation + hyperparameter”, “attention temperature selection + parallel encoding”, and checked the APE paper/repo for built-in label-free selection. Found training-time KL distillation in CEPE/CEPED and generic calibration literature, but no prior work applying sequential-teacher KL to select APE’s inference-time temperature/scale as of 2026-02-16.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| APE | Training-free attention alignment for parallel encoding | Needs labeled validation tuning for (T, scale) | Replace label-tuning with label-free KL to sequential teacher | Directly optimizes APE’s stated alignment goal without labels |
| Entropy analysis (ACL 2025) | Uses entropy as diagnostic; adds sinks/selective attention | Still needs hyperparameter choices (e.g., K) and does not tune APE knobs | Use next-token KL (stronger signal than entropy alone) | KL is closer to downstream behavior than attention-stat proxies |
| CEPE/CEPED | Trainable encoder + cross-attn; distills behavior via KL in training | Requires training and architecture change | No training; only calibrate inference hyperparameters | Much cheaper, deployable with existing APE kernels |
| Optimal attention temperature under shift | Derives \(\tau_{\text{optimal}}\) in theory | Needs distribution parameters; not tied to PCE | Use teacher matching to estimate temperature in PCE setting | Practical unlabeled estimator tailored to PCE |

---

## Experiments

### Experimental Setup

**Task setting.** Retrieval-augmented long-context QA in the APE LongBench protocol, focusing on one representative multi-hop dataset for a minimal decisive test.

**Primary benchmark (single dataset for decisiveness):** LongBench **2WikiMultihopQA** (2WikiMQA), evaluated with **F1** as in APE.

**Context regimes.**
- Calibration regime A: retrieve top-\(N=8\) chunks.
- Test regime B: retrieve top-\(N=20\) chunks.

**Methods compared (3 main conditions).**
1. **APE-default**: APE with repo defaults \(T=0.9, S=0.9\) (no tuning).
2. **APE-label-tuned (oracle)**: grid search on labeled calibration set (maximize F1) for \(T,S\).
3. **Ours (APE-KL-tuned)**: grid search on unlabeled calibration set (minimize KL to sequential teacher) for \(T,S\).

**Baseline Ladder (REQUIRED):**
- **No-RAG baseline**: model answers without retrieved chunks (LongBench “no RAG” setting) (published in APE Table 2).
- **Sequential RAG baseline**: sequential encoding with small chunks that fit the context window (APE “C200×20, Sequential”).
- **Parallel encoding baseline**: PCW-style parallel encoding (“C4000×20, PCW”).
- **Closest method**: APE (with tuning), and our KL-tuned variant.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Llama-3.1-8B-Instruct | 8B | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | Use long-context config as in APE LongBench experiments |

**Training Data (if applicable):**

No training data needed — inference only.

**Other Resources (if applicable):**
- LongBench dataset: https://huggingface.co/datasets/THUDM/LongBench
- Contriever retriever checkpoints: https://huggingface.co/facebook/contriever
- APE implementation: https://github.com/Infini-AI-Lab/APE

**Resource Estimate**:
- **Calibration (KL-tuned):** \(|\mathcal{C}|=32\) prompts, \(|\Theta|=25\) grid points → 32 sequential-teacher forward passes + 32×25 APE forward passes (logits only, no sampling). Using APE’s reported sequential prefill time (128K prefill 17s on H100; APE reduces prefill substantially), this is comfortably within a few GPU-hours even with conservative A100 slowdown.
- **Calibration (label-tuned oracle):** 32×25 full generations on the calibration set.
- **Evaluation:** 200 test examples × 3 methods × 1 generation each (greedy) + optional baseline re-runs for sequential/PCW if needed.
- Overall expected budget: \(< 50\) A100 GPU-hours (conservative), well below the 768 GPU-hour cap.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| LongBench-2WikiMQA | Multi-hop QA over long documents | F1 | test (with held-out calibration subset) | https://huggingface.co/datasets/THUDM/LongBench | APE repo LongBench eval scripts (or LongBench official eval) |

### Main Results

#### Results Table

(Primary target: Llama-3.1-8B-Instruct on LongBench-2WikiMQA; F1)

| Method | Base Model | Benchmark | F1 (mean±std) | Source | Notes |
|--------|------------|-----------|---------------|--------|------|
| No RAG (baseline) | Llama-3.1-8B-Instruct | LongBench-2WikiMQA | 40.58 (1 run) | APE Table 2 (Section 5.1.2) | Published number |
| 128K Sequential (baseline) | Llama-3.1-8B-Instruct | LongBench-2WikiMQA | 40.81 (1 run) | APE Table 2 (Section 5.1.2) | Published number |
| C200×20 Sequential RAG (baseline) | Llama-3.1-8B-Instruct | LongBench-2WikiMQA | 44.39 (1 run) | APE Table 2 (Section 5.1.2) | Published number |
| C4000×20 PCW (baseline) | Llama-3.1-8B-Instruct | LongBench-2WikiMQA | 44.87 (1 run) | APE Table 2 (Section 5.1.2) | Published number |
| APE (label-tuned, paper) | Llama-3.1-8B-Instruct | LongBench-2WikiMQA | 50.11 (1 run) | APE Table 2 (Section 5.1.2) | Published number; tuned on validation in APE |
| APE-default (T=0.9, S=0.9) | Llama-3.1-8B-Instruct | LongBench-2WikiMQA | **TBD** | - | To be verified (must run in same eval harness) |
| APE-label-tuned (oracle, our eval) | Llama-3.1-8B-Instruct | LongBench-2WikiMQA | **TBD** | - | Grid search on labeled calibration subset |
| **Ours: APE-KL-tuned (label-free)** | Llama-3.1-8B-Instruct | LongBench-2WikiMQA | **TBD** | - | Grid search minimizing KL to sequential teacher |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | Tune both \(T\) and \(S\) by KL | Best performance + stable selection |
| KL-tune \(T\) only | Fix \(S=0.9\), tune \(T\) | Partial recovery; tests if scaling is needed |
| KL-tune \(S\) only | Fix \(T=0.9\), tune \(S\) | Partial recovery; tests if temperature is needed |

### Experimental Rigor

- **Seeds & variance**: Use greedy decoding (deterministic) for primary results. If any stochastic decoding is required by LongBench scripts, run 3 seeds and report mean±std.
- **Calibration stability**: 10× bootstrap re-selection over \(|\mathcal{C}|=32\) prompts; report distribution of selected \(T,S\).
- **Confounders**:
  - Evaluation pipeline mismatch vs APE paper → rerun APE and sequential/PCW baselines in the same codebase on a small subset as a sanity check.
  - Retrieval non-determinism → fix retriever checkpoint and random seeds.
  - Tokenization/context-length mismatch → enforce identical chunking/token budgets across methods.

---

## Success Criteria

**Hypothesis** (directional): KL-tuned hyperparameters selected on unlabeled \(N=8\) calibration prompts will produce APE performance on \(N=20\) test prompts that is close to label-tuned APE and better than the default hyperparameters.

**Decision Rule** (concrete):
- **Proceed/Continue** if on test regime \(N=20\), **Ours (APE-KL-tuned)** achieves F1 within 0.5 absolute points of **APE-label-tuned** and exceeds **APE-default** by at least 1.0 point (or a margin outside evaluation variance).
- **Pivot** if Ours matches APE-default but not APE-label-tuned: try multi-token KL (compute KL over first K generated tokens using teacher-generated continuation) while keeping calibration budget fixed.
- **Refute** if (i) Ours underperforms APE-default, or (ii) hyperparameter selection is unstable (bootstrap 95% interval of selected \(T\) or \(S\) spans >0.1), or (iii) calibration cost requires substantially more than 32 teacher prompts to stabilize.

---

## Impact Statement

If successful, this work provides a practical way to deploy APE-like training-free parallel encoding in production settings where labels are unavailable or distributions drift: practitioners can retune APE’s knobs using only unlabeled traffic logs and a small sequential-teacher calibration run. This would reduce the operational cost and brittleness of adopting parallel context encoding for latency-sensitive RAG/ICL systems.

---

## Ethics & Risk Assessment

- **Privacy**: In deployment, KL calibration may use unlabeled production prompts, which could contain sensitive information. Mitigation: calibrate on privacy-scrubbed logs or public proxy data; for verification we only use public LongBench examples.
- **Misuse / capability uplift**: This is an inference-time calibration for a fixed base model; it does not add new tools, data, or training signal. The primary effect is efficiency/robustness of parallel encoding rather than expanding capabilities.
- **Failure risk**: If KL matching pushes APE toward a distribution that harms task performance (e.g., over-aligning to a suboptimal teacher under retrieval), the decision rule refutes the approach when it underperforms APE-default.

---

## References

- [APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding](./references/APE-Faster-and-Longer-Context-Augmented-Generation-via-Adaptive-Parallel-Encoding/meta/meta_info.txt) — Yang et al., 2025
- [Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding…](./references/Attention-Entropy-is-a-Key-Factor-An-Analysis-of-Parallel-Context-Encoding-with-Full-attention-based-Pre-trained-Language-Models/meta/meta_info.txt) — Zhang et al., 2025
- [Long-Context Language Modeling with Parallel Context Encoding](./references/Long-Context-Language-Modeling-with-Parallel-Context-Encoding/meta/meta_info.txt) — Yen et al., 2024
- [PEVLM: Parallel Encoding for Vision-Language Models](./references/PEVLM-Parallel-Encoding-for-Vision-Language-Models/meta/meta_info.txt) — Kang et al., 2025
- [Optimal Attention Temperature Enhances In-Context Learning under Distribution Shift](./references/Optimal-Attention-Temperature-Enhances-In-Context-Learning-under-Distribution-Shift/meta/meta_info.txt) — Demir & Dogan, 2025
- [On the Entropy Calibration of Language Models](./references/On-the-Entropy-Calibration-of-Language-Models/meta/meta_info.txt) — Cao et al., 2025
