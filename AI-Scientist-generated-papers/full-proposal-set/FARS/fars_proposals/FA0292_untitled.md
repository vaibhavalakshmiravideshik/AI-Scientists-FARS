# untitled

# Tiny-LR Proxy SFT for Reliable Dataset Ranking

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Teams building small and medium-sized language models (SLMs) often face a practical post-training decision: given a set of candidate supervised fine-tuning (SFT) datasets, which dataset (or recipe) should be used to fine-tune a base model to maximize downstream performance? This question matters because evaluating dataset value by full fine-tuning is expensive and scales linearly with the number of candidate datasets.

A common workaround is **proxy-model evaluation**: train a smaller proxy model (or run a shorter training schedule) on each candidate dataset and use the proxy’s results to rank datasets. However, recent evidence in *pretraining* shows that proxy rankings can be brittle: small changes in learning rate can flip which data recipe appears best, and proxy rankings may disagree with the rankings induced by larger models trained with tuned hyperparameters.

This proposal investigates whether a simple intervention from pretraining transfers to the post-training / instruction-tuning regime: training proxy SFT runs at a **tiny learning rate** (much smaller than standard SFT learning rates) to obtain more transferable dataset rankings.

### The Problem

Practitioners want a dataset-ranking procedure that is:

- **Decision-useful**: identifies which candidate dataset leads to the best downstream benchmark performance after SFT.
- **Training-light**: avoids training the full target model on every candidate dataset.
- **Stable**: does not change its conclusions under minor hyperparameter variations.

Existing dataset value benchmarks (e.g., **OpenDataArena** (ODA), a large-scale platform that evaluates many SFT datasets with a standardized pipeline) demonstrate that dataset choice can cause large performance variance, especially for math and code domains. But ODA-style evaluation still requires many full fine-tunes.

Separately, many **training-free or scoring-based** approaches exist for instruction data assessment (surveys include **Data Tsunami**), such as response-length heuristics, difficulty scores (e.g., IFD), reward-model scores, and gradient-based sample selection methods (e.g., LESS, GradFiltering). These methods often target *sample-level filtering* or require external scorers, and it is unclear whether they provide a reliable *dataset-level* ranking that matches the ordering induced by actually fine-tuning the target model.

The open gap is: **Can we make proxy SFT training runs reliably predict target-model dataset rankings with a minimal, decision-changing change to the proxy training regime?**

### Key Insight and Hypothesis

**Key prior observation (pretraining):** Wang et al. (2025) show that proxy-model data-recipe rankings become far more transferable across model scales when the proxy is trained with a *tiny learning rate*, and they connect this to suppressing higher-order (curvature) terms so that the proxy’s behavior is dominated by first-order gradient alignment.

**Hypothesis (this proposal):** For SFT dataset ranking, proxy runs trained at a tiny learning rate yield dataset rankings that better match the target model’s ranking than proxy runs trained at a standard learning rate.

**Mechanism hypothesis:** With a tiny learning rate, a short proxy SFT run stays in a near-linear regime where changes in a downstream evaluation objective are dominated by a first-order term proportional to an alignment between (i) gradients induced by the candidate dataset and (ii) gradients induced by the evaluation objective. This alignment signal is expected to be more scale-invariant across model sizes than the higher-order optimization effects that dominate at standard learning rates, reducing proxy-to-target ranking flips.

Why this could be wrong:
- SFT may inherently require higher-order optimization effects (e.g., learning a chat template, long-range credit across multi-turn traces), making tiny-LR proxies underfit and worsen ranking.
- Tiny-LR proxies might be effectively “no training”; any apparent gains could be an artifact of measuring base-model fit rather than learning dynamics.

We include explicit controls to rule out the “no training” confound.

---

## Proposed Approach

### Overview

We propose **Tiny-LR Proxy SFT (TLR-Proxy)**: to rank candidate SFT datasets, run a short proxy fine-tuning for each dataset using a learning rate 1–2 orders of magnitude smaller than a standard SFT learning rate, then rank datasets by the proxy model’s downstream performance. We compare this ranking against the ranking induced by fine-tuning a larger target model on the same candidate datasets.

The method is a drop-in modification of existing proxy-model practice: **only the proxy learning rate is changed**.

### Method Details

**Inputs:**
- Candidate dataset set \(\{D_1, \dots, D_K\}\).
- Proxy model \(M_p\) (smaller) and target model \(M_t\) (larger), both starting from **base (pre-instruction)** checkpoints.
- A domain-specific downstream evaluation suite \(\mathcal{E}\) (fully automated).

**Procedure:**
1. **Proxy runs (for each dataset \(D_i\))**:
   - Fine-tune \(M_p\) on \(D_i\) for a short budget of steps \(T_p\).
   - Repeat under two learning-rate regimes:
     - **Standard-LR proxy**: \(\eta_{std}\) (typical SFT LR)
     - **Tiny-LR proxy (ours)**: \(\eta_{tiny}\ll \eta_{std}\)
   - Compute proxy score \(S_p(D_i; \eta)\) on \(\mathcal{E}\).

2. **Target runs (ground truth, for each dataset \(D_i\))**:
   - Fine-tune \(M_t\) on \(D_i\) for \(T_t\) steps using a standard learning rate \(\eta_{tgt}\).
   - Compute target score \(S_t(D_i)\) on \(\mathcal{E}\).

3. **Ranking agreement metric (primary): pairwise direction accuracy (PDA)**
   - For each pair \((i,j)\), define the ground-truth ordering by \(S_t(D_i) > S_t(D_j)\).
   - Define the proxy ordering by \(S_p(D_i; \eta) > S_p(D_j; \eta)\).
   - PDA is the fraction of dataset pairs whose ordering matches (chance = 0.5).

4. **Non-degeneracy diagnostics (to rule out “tiny LR = no training”)**
   - For each proxy run, compute:
     - Training loss drop (%), and
     - KL divergence between the proxy model and the base checkpoint on a fixed held-out prompt set.
   - If \(\text{KL}<10^{-4}\) nats/token **and** loss drop <0.5%, label the run as degenerate and treat results as inconclusive.

### Key Innovations

- **New setting**: Applies the tiny-learning-rate proxy principle (shown for *pretraining* data recipes) to **instruction-tuning (SFT) dataset ranking**, where training dynamics, data scale, and objectives differ.
- **Decision-centric evaluation**: Evaluates proxy usefulness directly as a dataset-ranking tool via PDA (robust for small \(K\)), rather than only reporting correlation on large recipe grids.
- **Explicit degeneracy controls**: Pre-registers measurable checks to distinguish “tiny LR improves transfer” from “tiny LR does nothing.”

---

## Related Work

### Field Overview

Instruction-tuning data curation spans three overlapping directions: (i) **benchmarking dataset value** by running standardized fine-tunes, (ii) **training-free scoring** of data quality using heuristics or model-based judges, and (iii) **training-light proxy methods** that approximate full fine-tuning outcomes.

Dataset value benchmarking (e.g., OpenDataArena) is the most direct but also the most expensive approach. Training-free scoring methods can be cheaper but may measure proxies (length, style, judge preferences) that do not translate to downstream benchmark improvements. Training-light proxy methods sit between these extremes, but their reliability is poorly characterized in SFT compared to pretraining.

### Related Papers

- **[Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice](./references/Can-Small-Training-Runs-Reliably-Guide-Data-Curation-Rethinking-Proxy-Model-Practice/meta/meta_info.txt)**: Shows tiny learning rates make proxy *pretraining* recipe rankings far more transferable across scales; motivates our SFT transfer test.
- **[OpenDataArena: A Fair and Open Arena for Benchmarking Post-Training Dataset Value](./references/OpenDataArena-A-Fair-and-Open-Arena-for-Benchmarking-Post-Training-Dataset-Value/meta/meta_info.txt)**: Large-scale benchmark of SFT dataset value; highlights high variance and calls for training-light valuation.
- **[The Best Instruction-Tuning Data are Those That Fit](./references/The-Best-Instruction-Tuning-Data-are-Those-That-Fit/meta/meta_info.txt)**: GRAPE selects responses aligned with the base model distribution; relevant as a training-free distribution-fit signal.
- **[Unleashing the Power of Data Tsunami: A Comprehensive Survey on Data Assessment and Selection for Instruction Tuning of Language Models](./references/Unleashing-the-Power-of-Data-Tsunami-A-Comprehensive-Survey-on-Data-Assessment-and-Selection-for-Instruction-Tuning-of-Language-Models/meta/meta_info.txt)**: Survey taxonomy of instruction data selection; source for baselines and framing.
- **[Uncertainty-Aware Gradient Signal-to-Noise Data Selection for Instruction Tuning](./references/Uncertainty-Aware-Gradient-Signal-to-Noise-Data-Selection-for-Instruction-Tuning/meta/meta_info.txt)**: GradFiltering uses LoRA-ensemble gradient statistics for sample selection; relevant as gradient-based alternative signals.
- **[D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning](./references/D3-Diversity-Difficulty-and-Dependability-Aware-Data-Selection-for-Sample-Efficient-LLM-Instruction-Tuning/meta/meta_info.txt)**: Multi-criteria selection (diversity/difficulty/dependability); indicates simple loss-based scoring can fail.
- **[Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation](./references/Training-Data-Selection-with-Gradient-Orthogonality-for-Efficient-Domain-Adaptation/meta/meta_info.txt)**: Uses gradient geometry for data selection; connects to gradient-alignment intuition.
- **[LESS: Selecting Influential Data for Targeted Instruction Tuning](./references/LESS-Selecting-Influential-Data-for-Targeted-Instruction-Tuning/meta/meta_info.txt)**: Gradient-influence based data selection for targeted instruction tuning; shows small-model selection can transfer across scales.
- **[Less is More: Improving LLM Alignment via Preference Data Selection](https://arxiv.org/abs/2502.14560)**: Preference-data selection in DPO; adjacent evidence that data selection can outperform “use all data.”
- **[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)**: Demonstrates small curated instruction datasets can match large ones; motivates dataset value ranking.
- **[Self-Instruct](https://arxiv.org/abs/2212.10560)**: Early pipeline for generating instruction data; foundational for synthetic SFT datasets.
- **[InstructGPT](https://arxiv.org/abs/2203.02155)**: Established the modern instruction tuning + preference optimization pipeline; motivates post-training as a core capability driver.
- **[Stanford Alpaca](https://arxiv.org/abs/2303.16199)**: Popular open instruction dataset; baseline for many SFT studies.
- **[FLAN](https://arxiv.org/abs/2109.01652)** / **[FLAN-T5](https://arxiv.org/abs/2210.11416)**: Instruction tuning via mixture of tasks; motivates mixture selection as dataset-level decision.
- **[DEITA](https://arxiv.org/abs/2312.15685)**: Data selection via evol-complexity/evol-quality; widely used heuristic scoring.
- **[Alpagasus](https://arxiv.org/abs/2307.08701)**: Uses strong LLMs to rate/filter instruction data; representative of judge-based scoring.
- **[DSIR: Data Selection via Importance Resampling](https://arxiv.org/abs/2302.03169)**: Distribution-matching framing for data selection; relevant to “fit” signals.
- **[Koh & Liang (Influence Functions)](https://arxiv.org/abs/1703.04730)**: Classical data valuation via influence functions; conceptually related but often impractical at scale.
- **[TracIn](https://arxiv.org/abs/2002.08484)**: First-order influence approximation; related to gradient-based valuation.
- **[GradMatch](https://arxiv.org/abs/2006.15583)**: Gradient matching for coreset selection; connects to gradient alignment signals.
- **[EL2N](https://arxiv.org/abs/2107.07075)**: Loss-based coreset scoring; relevant as simple baseline scoring.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Full fine-tune benchmarking | Evaluate dataset value by training target model | OpenDataArena | Downstream benchmark suites (math/code/general) | Expensive (\#datasets × full SFT) |
| Training-free scoring | Use heuristics / judges / model fit without training | GRAPE, DEITA, Alpagasus, IFD-style scores | Correlation with downstream benchmarks | May optimize proxy metrics not aligned with target improvements |
| Training-light proxy models | Train smaller/shorter runs to approximate target outcomes | Wang et al. (pretraining), proxy SFT practice | Ranking correlation / regret | Often brittle to hyperparameters; unclear for SFT |
| Gradient-based valuation / coreset | Use gradients/influence to score samples | LESS, GradFiltering, TracIn, GradMatch | Subset quality at fixed budget | Typically sample-level, needs gradients or ensembles |

### Closest Prior Work

1. **Wang et al. (2025) proxy-model practice**: Demonstrates that tiny learning rates dramatically improve proxy-to-target ranking transfer for *pretraining* data recipes, with a mechanism based on suppressing higher-order effects. **Gap:** does not test instruction tuning; SFT uses different objectives (instruction-following) and different data regimes (smaller, structured instruction-response pairs), so transfer is not guaranteed.

2. **OpenDataArena (2024/2025)**: Provides a standardized benchmark for post-training dataset value and explicitly notes the need for training-light valuation. **Gap:** does not propose or test proxy-learning-rate regimes; its pipeline is still compute-intensive.

3. **GRAPE (2025)**: Uses base-model probability to select responses per instruction, operationalizing distribution fit without gradient computation. **Gap:** focuses on response selection within an instruction pool, not ranking whole datasets, and does not analyze proxy training stability.

4. **GradFiltering (2026)**: Uses gradient statistics and uncertainty from LoRA ensembles to select high-value instruction samples. **Gap:** targets sample selection rather than dataset ranking; requires multiple proxy runs and gradient collection.

5. **D3 (2025)**: Multi-criteria selection and shows that naive perplexity-based selection can fail. **Gap:** does not address proxy-to-target transfer of dataset-level rankings.

**Novelty Kill Search Summary:** We searched for prior work explicitly combining *tiny learning rates* with *proxy fine-tuning for instruction-tuning dataset ranking*, including queries like "tiny learning rate instruction tuning proxy model", "proxy fine-tuning tiny LR dataset ranking", and OpenReview searches for "tiny learning rate" + "instruction tuning". As of 2026-02-25, we did not find work that tests the tiny-LR proxy regime for SFT dataset valuation; the closest match is Wang et al. (2025), which is pretraining-only. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Wang et al. 2025 | Tiny-LR proxy for **pretraining** recipe ranking | Not tested for SFT; different objective/data regime | Apply tiny-LR proxy to **SFT dataset ranking** | If first-order alignment dominates early SFT too, rankings transfer better |
| OpenDataArena | Benchmarks dataset value via many SFT runs | Expensive; no training-light method | Use short proxy runs to approximate ranking | Reduces compute per candidate dataset |
| GRAPE | Training-free response selection via base-model probability | Not a dataset ranking method; depends on response pools | Use training-light proxy with LR regime change | Captures optimization dynamics beyond static fit |
| GradFiltering | Sample selection via gradient SNR and uncertainty | Sample-level, requires ensembles/gradients | Dataset-level ranking via proxy training regime | Simpler, drop-in, no external scorers |
| D3 | Multi-criteria coreset selection | Not about proxy-to-target ranking transfer | Directly optimize ranking transfer metric (PDA) | Evaluates the end goal practitioners care about |

---

## Experiments

### Experimental Setup

**Core experiment (≤3 main conditions):**
- **Condition A (baseline):** Standard-LR proxy SFT
- **Condition B (ours):** Tiny-LR proxy SFT
- **Condition C (control):** Training-free “dataset fit” baseline using base-model NLL (no fine-tuning)

We then compare each proxy method’s dataset ranking against the target model’s ranking.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen2.5 Base | 1.5B | https://huggingface.co/Qwen/Qwen2.5-1.5B | Proxy model \(M_p\) |
| Qwen2.5 Base | 7B | https://huggingface.co/Qwen/Qwen2.5-7B | Target model \(M_t\) |

**Training Data (math domain; K=12 datasets):** We use 12 math-oriented instruction-tuning datasets available via OpenDataArena’s scored dataset collection (subset names may evolve; verification should pin exact subset IDs at run time).

| Dataset (ODA subset) | Purpose | Size used | Download Link | License |
|---|---|---:|---|---|
| AM-Thinking-v1-Distilled-math | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| DeepMath-309K | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| Maths-College | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| OpenR1-Math | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| QwQ-LongCoT-130K-math | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| R1-Distill-SFT-math | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| hkust-nlp__dart-math-hard | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| mathplus | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| numinamath-cot | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| numinamath1_5 | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| openmathinstruct-2 | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |
| Magpie-Reasoning-V2-250K-CoT-QwQ-math | Candidate dataset | 50k sampled | https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data | See dataset card |

*Note:* If any ODA subset is gated or missing, fall back to the corresponding original dataset on HuggingFace (e.g., `nvidia/OpenMathInstruct-2`, `AI-MO/NuminaMath`, etc.) and keep the same sampling protocol.

**Fine-tuning protocol (both proxy and target):**
- Format: Convert each dataset to a chat-style SFT format with a fixed template (system prompt + user problem + assistant solution).
- Max sequence length (cutoff_len): **4096** (matches OpenDataArena’s default setting; see their Table 2 in `./references/OpenDataArena-A-Fair-and-Open-Arena-for-Benchmarking-Post-Training-Dataset-Value/sections/7.1 Training Configurations.md`).
- Optimizer: AdamW
- Weight decay: 0.0
- LR schedule: cosine with warmup_ratio=0.1 (as in OpenDataArena for Qwen2.5; Table 2).
- Effective batch size: target **16** sequences/step via gradient accumulation (e.g., `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`; proxy may use larger per-device batch if memory permits).
- Steps:
  - Proxy steps \(T_p\): 500
  - Target steps \(T_t\): 2000
- Learning rates:
  - Proxy standard: \(\eta_{std}=5\times 10^{-5}\)
  - Proxy tiny: \(\eta_{tiny}=5\times 10^{-6}\)
  - Target: \(\eta_{tgt}=5\times 10^{-5}\)
  - (Both **5e-6** and **5e-5** appear in OpenDataArena’s standardized configs for Qwen2.5; Table 2.)
- Parameter-efficient tuning: **LoRA** with fixed hyperparameters across runs (e.g., `r=16`, `lora_alpha=32`, `lora_dropout=0.05`; target modules include attention + MLP projections).

**Note on terminology:** This proposal uses the term **"standard"** to mean "commonly used in *SFT evaluation pipelines* for these models" (as exemplified by OpenDataArena), not "most common in QLoRA tutorials." We intentionally anchor learning-rate values to a published benchmark’s configuration table to avoid LR guesswork.

**Seeds:** Use 3 random seeds for each dataset × condition for proxy and target fine-tunes (\(\text{seeds}=[42,123,456]\)).

**Resource Estimate**:
- Training time reference: QLoRA SFT of Llama-3.1-8B on 100k samples at 2048 tokens can run in ~4h45m on one A100 (40GB) with Unsloth (Hugging Face blog). Our runs are shorter (2000 steps) and can use similar PEFT tooling.
- Expected budget (rough):
  - Target: \(12\) datasets × \(3\) seeds × ~1.5 GPU-hours ≈ **54 GPU-hours**
  - Proxy: \(12\) datasets × \(3\) seeds × \(2\) LRs × ~0.5 GPU-hours ≈ **36 GPU-hours**
  - Evaluation + overhead ≈ **20 GPU-hours**
  - **Total ≈ 110 GPU-hours** (well under 768 GPU-hours)
- Peak memory: single A100 80GB sufficient with QLoRA; multi-GPU parallelism optional.

### Benchmarks and Metrics

**Evaluation suite \(\mathcal{E}\) (math, fully automated):**

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| GSM8K | Grade-school math word problems with numeric answers | Exact-match accuracy | test | https://huggingface.co/datasets/openai/gsm8k | lm-eval-harness or math-eval-harness |
| MATH-500 | 500-problem subset of MATH for fast evaluation | Exact-match accuracy | test | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 | math-eval-harness |

**Primary metric:**
- **Pairwise Direction Accuracy (PDA)** between proxy ranking and target ranking, computed on the mean target score across seeds.

**Secondary metrics:**
- Spearman rank correlation \(\rho\)
- Top-1 accuracy (does the proxy pick the best dataset?)

### Main Results

#### Results Table

| Method (ranking signal) | Proxy model | Target model | Metric: PDA ↑ | Metric: Spearman ρ ↑ | Source | Notes |
|---|---|---|---:|---:|---|---|
| Random ranking | - | - | 0.50 (expected) | 0.00 (expected) | - | Sanity baseline |
| Base-model dataset NLL (fit) | Qwen2.5-1.5B | Qwen2.5-7B | **TBD** | **TBD** | - | Training-free baseline |
| Standard-LR proxy SFT | Qwen2.5-1.5B | Qwen2.5-7B | **TBD** | **TBD** | - | Proxy LR=5e-5 |
| **Tiny-LR proxy SFT (ours)** | Qwen2.5-1.5B | Qwen2.5-7B | **TBD** | **TBD** | - | Proxy LR=5e-6 |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (tiny LR) | LR = 5e-6 | Best ranking transfer if hypothesis holds |
| Mid-LR proxy | LR = 2e-5 | Intermediate transfer; helps check LR sensitivity without full sweep |

### Experimental Rigor

**Variance & Reproducibility:**
- Report mean ± std over 3 seeds for each dataset’s target score; use bootstrap over dataset pairs to estimate CI for PDA.

**Validity & Controls:**
- **Confound 1 (tiny LR ≈ no training):** Use KL-to-base + training-loss-drop thresholds; compare against base-model NLL baseline.
- **Confound 2 (proxy too weak → near-zero accuracy):** Ensure proxy has non-trivial GSM8K/MATH-500 performance after tuning; if proxy accuracy is near floor across all datasets, increase proxy size to Qwen2.5-3B as a pre-registered fallback.
- **Confound 3 (dataset contamination with evaluation benchmarks):** Prefer ODA subsets with explicit contamination checks when available; otherwise, run n-gram overlap checks against GSM8K and MATH-500 prompts and exclude any dataset with extreme overlap.

---

## Success Criteria

**Hypothesis**:
Tiny-LR proxy SFT improves dataset ranking transfer: PDA(tiny) > PDA(standard), with the improvement not explained by degeneracy (non-zero KL and meaningful loss drop).

**Decision Rule**:
- **Proceed** if PDA(tiny) exceeds PDA(standard) by a margin outside the bootstrap 95% CI (and PDA(standard) > 0.55), indicating meaningful improvement over chance and over standard proxy practice.
- **Pivot** if both proxy methods are above chance but close (e.g., |PDA(tiny) − PDA(standard)| ≤ 0.03): test whether increasing proxy steps (e.g., 1000) or using a slightly larger proxy (3B) changes conclusions.
- **Refute** if PDA(tiny) ≤ PDA(standard) or if tiny-LR runs are frequently degenerate under the pre-registered KL/loss-drop checks.

---

## Impact Statement

If tiny-learning-rate proxy SFT reliably predicts target-model dataset rankings, practitioners can screen many candidate instruction-tuning datasets using short, cheap proxy runs while preserving decision quality. This could reduce the cost of post-training data curation and provide a simple default for proxy evaluation protocols.

---

## References

- [Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice](./references/Can-Small-Training-Runs-Reliably-Guide-Data-Curation-Rethinking-Proxy-Model-Practice/meta/meta_info.txt) - Wang et al., 2025
- [OpenDataArena: A Fair and Open Arena for Benchmarking Post-Training Dataset Value](./references/OpenDataArena-A-Fair-and-Open-Arena-for-Benchmarking-Post-Training-Dataset-Value/meta/meta_info.txt) - Cai et al., 2024/2025
- [The Best Instruction-Tuning Data are Those That Fit](./references/The-Best-Instruction-Tuning-Data-are-Those-That-Fit/meta/meta_info.txt) - Zhang et al., 2025
- [Uncertainty-Aware Gradient Signal-to-Noise Data Selection for Instruction Tuning](./references/Uncertainty-Aware-Gradient-Signal-to-Noise-Data-Selection-for-Instruction-Tuning/meta/meta_info.txt) - Yuan et al., 2026
- [Unleashing the Power of Data Tsunami: A Comprehensive Survey on Data Assessment and Selection for Instruction Tuning of Language Models](./references/Unleashing-the-Power-of-Data-Tsunami-A-Comprehensive-Survey-on-Data-Assessment-and-Selection-for-Instruction-Tuning-of-Language-Models/meta/meta_info.txt) - Qin et al., 2024
- [D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning](./references/D3-Diversity-Difficulty-and-Dependability-Aware-Data-Selection-for-Sample-Efficient-LLM-Instruction-Tuning/meta/meta_info.txt) - Zhang et al., 2025
- [Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation](./references/Training-Data-Selection-with-Gradient-Orthogonality-for-Efficient-Domain-Adaptation/meta/meta_info.txt) - Zhang et al., 2026
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) - Zhou et al., 2023
- [InstructGPT](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
- [Self-Instruct](https://arxiv.org/abs/2212.10560) - Wang et al., 2022
- [Stanford Alpaca](https://arxiv.org/abs/2303.16199) - Taori et al., 2023
- [FLAN](https://arxiv.org/abs/2109.01652) - Wei et al., 2021
- [FLAN-T5](https://arxiv.org/abs/2210.11416) - Chung et al., 2022
- [DEITA](https://arxiv.org/abs/2312.15685) - Li et al., 2023
- [Alpagasus](https://arxiv.org/abs/2307.08701) - Chen et al., 2023
- [DSIR: Data Selection via Importance Resampling](https://arxiv.org/abs/2302.03169) - Xie et al., 2023
- [Influence Functions](https://arxiv.org/abs/1703.04730) - Koh & Liang, 2017
- [TracIn](https://arxiv.org/abs/2002.08484) - Pruthi et al., 2020
- [GradMatch](https://arxiv.org/abs/2006.15583) - Killamsetty et al., 2021
- [EL2N](https://arxiv.org/abs/2107.07075) - Paul et al., 2021
