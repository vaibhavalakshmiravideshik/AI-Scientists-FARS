# untitled

# Does repetition-heavy long-CoT SFT improve downstream RLVR/GRPO fine-tuning?

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Reinforcement learning with verifiable rewards (RLVR) improves language models using automated checks as rewards (e.g., exact-answer matching for math, unit tests for code). Most RLVR pipelines for reasoning models use two stages: (1) **supervised fine-tuning (SFT)** on chain-of-thought (CoT) demonstrations to teach output format and basic reasoning behavior, then (2) online policy optimization, often **Group Relative Policy Optimization (GRPO)**.

GRPO samples a **group** of \(G\) solutions per prompt and computes an advantage estimate from the *relative* outcomes inside the group. If most groups are all-correct or all-wrong, relative advantages can be small, slowing learning.

A recent result in long-CoT SFT challenges a common intuition about data scaling. **Kopiczko et al. (2026)** show that, under a fixed optimizer-update budget, training for many epochs on a small long-CoT dataset can outperform training for one epoch on a much larger dataset, largely because repetition teaches models to reliably terminate long reasoning traces and produce an extractable final answer.

### The Problem

It is unknown whether the long-CoT SFT “repetition advantage” transfers to the downstream RLVR stage. This matters because RLVR is expensive: if repetition-heavy SFT is a better GRPO initializer, practitioners could use much smaller (and cheaper to curate) CoT datasets without sacrificing post-RL performance.

However, recent work suggests that “better SFT” does not necessarily mean “better prepared for RL.” **PEAR (Zhang et al., 2026)** shows that, after identical online RL, models initialized from *stronger* SFT checkpoints can underperform those from *weaker* checkpoints, attributing this to offline-to-online distribution mismatch (occupancy mismatch). This makes it plausible that repetition-heavy SFT could harm GRPO training even if it improves offline/held-out SFT metrics.

### Key Insight and Hypothesis

We test whether the way we allocate a fixed SFT budget—**repeating a small long-CoT dataset** vs **seeing more unique CoT examples**—changes downstream GRPO outcomes.

A key confound is pre-RL success: if one initializer starts much better on the RL training distribution, it may win trivially. We control for this by constructing an RL training subset where both initializers have matched pre-RL success.

**Primary hypothesis (directional):** after controlling for pre-RL success, repetition-heavy long-CoT SFT will not improve (and may reduce) post-GRPO accuracy relative to data-scaled SFT, because repetition can narrow policy support and increase offline-to-online mismatch.

**Mechanism prediction:** if repetition-heavy SFT underperforms, we expect lower early-training within-prompt diversity, measurable as a lower **mixed-group rate** (fraction of prompts where a GRPO group contains at least one correct and at least one incorrect sample).

---

## Proposed Approach

### Overview

1. Train two SFT initializers with **matched update budget** but different epoch-vs-unique-data trade-offs (data-scaled vs repetition-heavy).
2. Build a **difficulty-matched** RL training set where both initializers have similar pre-RL success.
3. Run identical GRPO from each initializer under a fixed RL budget.
4. Compare post-RL accuracy and diagnostics tied to GRPO group informativeness.

### Method Details

**Stage 0 (Base model)**

- Base model: **Qwen/Qwen3-4B-Base**.

**Stage 1 (SFT initializers; matched updates \(B=51{,}200\))**

- **SFT data**: Dolci-Think-SFT-7B (AllenAI), filtered to first-turn only, must contain `<think>...</think>`, and tokenized length <= 10k (as in Kopiczko et al.).
- Two SFT configurations with equal updates:
  - **Init-A (data-scaled)**: 1 epoch on 51.2k samples
  - **Init-B (repetition-heavy)**: 32 epochs on 1.6k samples
- Training details follow Kopiczko et al.: batch size 1, response-only loss, 10% warmup, cosine LR, 8-bit Adam, bf16 weights.

**Stage 2 (difficulty-matched RL prompt set)**

- Candidate pool: 10,000 problems from **MATH train**.
- Estimate per-problem success for each initializer with **pass@k** (probability at least one of \(k\) samples is correct). Default: pass@4 with temperature 0.6, top-p 0.95, max_new_tokens 4096.
- Keep problems where:
  - both have pass@4 in [0.1, 0.4], and
  - |p_A - p_B| <= 0.1.
- Report survivorship (10k -> matched) and distribution shift (MATH subject/type histogram). If matched size <1,000, widen the band and/or increase the candidate pool.

**Stage 3 (RLVR with GRPO)**

- RL prompts: the difficulty-matched MATH-train subset.
- Verifier / reward: require LaTeX `\\boxed{...}`; verify equivalence to gold using **Math-Verify**; binary reward \(r\in\{0,1\}\).
- GRPO: group size \(G=8\), KL regularization to initializer.
- Budget: choose GRPO updates \(T\) so that (2 initializers) x (3 seeds) fits within 768 A100 GPU-hours.

**Diagnostics (mechanism probes; no extra training conditions)**

- Termination/parse rate.
- Mixed-group rate.
- Reward sparsity (fraction of samples with \(r=1\)).
- Output length and entropy.

### Key Innovations

- **Compute-matched transfer test**: isolates whether SFT data strategy changes downstream GRPO, not just offline metrics.
- **Difficulty-matched RL prompt selection**: controls pre-RL success confounds.
- **Link to offline-to-online mismatch**: tests a practical, recipe-level knob that may increase PEAR-style mismatch.

---

## Related Work

### Field Overview

RLVR uses programmatic verifiers to provide objective reward signals for policy optimization in domains like math and code. GRPO is widely used because it avoids training a separate value function by estimating advantages from within-prompt rollout groups.

Recent work shows that offline SFT choices can misprepare models for online RL due to distribution mismatch (e.g., PEAR). This motivates studying SFT *data allocation* (repetition vs scaling) specifically through the lens of downstream RL readiness.

### Related Papers

- **[Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**: Compute-matched long-CoT SFT where repeating small datasets beats scaling unique data.
- **[Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning](./references/Good-SFT-Optimizes-for-SFT-Better-SFT-Prepares-for-Reinforcement-Learning/meta/meta_info.txt)**: PEAR shows stronger SFT can underperform after RL due to offline-to-online mismatch; proposes importance-weighted offline losses.
- **[A Practical Two-Stage Recipe for Mathematical LLMs](./references/A-Practical-Two-Stage-Recipe-for-Mathematical-LLMs-Maximizing-Accuracy-with-SFT-and-Efficiency-with-Reinforcement-Learning/meta/meta_info.txt)**: Extended SFT then GRPO improves math performance and efficiency.
- **[SFT Memorizes, RL Generalizes](./references/SFT-Memorizes-RL-Generalizes-A-Comparative-Study-of-Foundation-Model-Post-training/meta/meta_info.txt)**: SFT can harm OOD generalization while outcome-based RL improves it; includes failure regimes for SFT→RL.
- **[RL Squeezes, SFT Expands](./references/RL-Squeezes-SFT-Expands-A-Comparative-Study-of-Reasoning-LLMs/meta/meta_info.txt)**: RL reduces trajectory diversity while SFT expands correct trajectories; supports initializer-dependent RL dynamics.
- **[Save the Good Prefix](./references/Save-the-Good-Prefix-Precise-Error-Penalization-via-Process-Supervised-RL-to-Enhance-LLM-Reasoning/meta/meta_info.txt)**: Process-supervised RL alternative to pure outcome rewards.
- **[DeepSeekMath](https://arxiv.org/abs/2402.03300)**: Introduces GRPO for math RLVR.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Demonstrates strong reasoning via RLVR and popularizes two-stage pipelines.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Motivates verification-based supervision.
- **[OpenRLHF](https://arxiv.org/abs/2405.11143)**: Open framework commonly used for GRPO/RLVR.
- **[MC-GRPO](https://arxiv.org/abs/2601.22582)**: Robust advantage normalization for small GRPO groups.
- **[It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977)**: Analysis of GRPO through preference-optimization lens.
- **[Exploration vs Exploitation: Rethinking RLVR](https://arxiv.org/abs/2512.16912)**: Stability/exploration analysis for RLVR.
- **[Mirage or Method?](https://arxiv.org/abs/2508.21188)**: Model–task alignment can change RL conclusions.
- **[LIMO](https://arxiv.org/abs/2502.03387)**: Many-epoch SFT on curated reasoning data.
- **[Small batch size training for language models](https://arxiv.org/abs/2504.02299)**: Supports very small batch sizes (relevant to Kopiczko et al.’s batch=1).
- **[Math-Verify](https://github.com/huggingface/Math-Verify)**: Robust boxed-answer extraction and symbolic equivalence checking.
- **[LightEval](https://github.com/huggingface/lighteval)**: Standard harness for MATH-500 evaluation.
- **[MATH](https://arxiv.org/abs/1904.01557)**: Competition math dataset; used for RL prompts and MATH-500 evaluation.
- **[GSM8K](https://arxiv.org/abs/2110.14168)**: Math word problems; used as secondary OOD evaluation.
- **[Murphy: Multi-Turn GRPO](https://arxiv.org/abs/2511.07833)**: Reports GRPO compute figures for Qwen3-scale models.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Long-CoT SFT compute allocation | Epochs vs unique samples at fixed updates | Kopiczko et al. (2026), LIMO (2025) | AIME, GPQA | Mechanism unclear; may change termination/diversity |
| Offline-to-online mismatch in SFT→RL | Offline optimization can misprepare for RL; fix via reweighting | PEAR (2026) | MATH-500, AIME | Requires careful controls; mismatch can hide offline |
| GRPO-family RLVR | Group-relative advantages for verifiable rewards | DeepSeekMath (2024), MC-GRPO (2026) | MATH, GSM8K | Sensitive to group stats, KL, reward sparsity |
| SFT vs RL analyses | SFT expands; RL squeezes; memorization vs generalization | Matsutani et al. (2025), Chu et al. (2025) | Math + synthetic envs | Mostly analysis; not recipe-focused |

### Closest Prior Work

- **Kopiczko et al. (2026)**: SFT repetition beats data scaling under fixed updates; does not test downstream RL.
- **PEAR (2026)**: Shows offline SFT quality can anti-correlate with post-RL quality; fixes via loss reweighting. Our study is orthogonal (loss fixed; data strategy varied) and asks whether repetition-vs-scaling affects RL readiness.
- **Chu et al. (2025)**: RL can fail after overfit SFT; motivates testing repetition-heavy SFT as a candidate overfit regime.
- **Matsutani et al. (2025)**: Trajectory expansion vs squeezing suggests initializer choice can change GRPO learning dynamics.

**Novelty Kill Search Summary:** Searched for combinations of “data repetition” with “RLVR initializer” / “GRPO” / “SFT epochs before GRPO” and checked recent RLVR surveys and 2025-2026 arXiv/OpenReview. Closest match is PEAR (loss reweighting), but no work was found that isolates compute-matched long-CoT SFT repetition-vs-scaling as the variable of interest for downstream GRPO outcomes (as of 2026-02-20). Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Kopiczko et al. (2026) | SFT repetition vs scaling under fixed updates | No RL stage | Add GRPO stage + diagnostics | Decision-relevant for full pipelines |
| PEAR (2026) | Fix offline-to-online mismatch via loss reweighting | Not about data repetition | Vary data repetition vs scaling | Tests a simpler “recipe knob” (data) |
| Chu et al. (2025) | SFT overfits; RL generalizes | Not long-CoT repetition | Target repetition as overfit regime | Identifies boundary condition |
| Matsutani et al. (2025) | SFT expands; RL squeezes trajectories | Not recipe-focused | Measure GRPO learning curves | Mechanistic link to group stats |

---

## Experiments

### Experimental Setup

**Primary comparison:** GRPO from Init-B vs GRPO from Init-A, using identical GRPO hyperparameters and the same difficulty-matched RL training prompt set.

**Baseline Ladder (REQUIRED):**

- Zero-shot prompting on MATH-500 (Qwen3-4B-Base).
- Inference-time scaling: Best-of-32 on MATH-500 for (i) base, (ii) Init-A, (iii) Init-B.
- SFT-only: Init-A and Init-B on MATH-500.
- RLVR: GRPO from Init-A vs GRPO from Init-B.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen/Qwen3-4B-Base | 4B | https://huggingface.co/Qwen/Qwen3-4B-Base | Pre-instruction base model |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Dolci-Think-SFT-7B | SFT initializer training | use 51.2k and 1.6k subsets | https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B | ODC-BY |
| MATH | RL prompts (train) | sample 10k candidate, then filter | https://huggingface.co/datasets/hendrycks/competition_math | MIT |
| MATH-500 | Primary evaluation | 500 | https://huggingface.co/datasets/hendrycks/competition_math | MIT |
| GSM8K | Secondary OOD eval | 1.3k test | https://huggingface.co/datasets/openai/gsm8k | MIT |

**Evaluation Scripts:**

- Answer verification: Math-Verify (https://github.com/huggingface/Math-Verify).
- Evaluation harness: LightEval (https://github.com/huggingface/lighteval).
- GRPO training: OpenRLHF (https://github.com/OpenRLHF/OpenRLHF) or TRL GRPOTrainer.

**Resource Estimate** (<= 768 A100 GPU-hours):

- SFT: 2 runs, budget 72 A100-hours total.
- Difficulty-matching: 80k generations, budget 40 A100-hours.
- GRPO: 6 runs (2 inits x 3 seeds), budget <= 360 A100-hours.
- Eval/logging: <= 40 A100-hours.
- Total: <= 512 A100-hours.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MATH-500 | 500 competition-math problems | Pass@1 and Pass@32 (higher is better) | test | https://huggingface.co/datasets/hendrycks/competition_math | LightEval + Math-Verify |
| GSM8K | grade-school math word problems | Accuracy (higher is better) | test | https://huggingface.co/datasets/openai/gsm8k | Exact-match |

### Main Results

| Method | Base Model | Benchmark | Metric (mean +/- std) | Source | Notes |
|---|---|---|---|---|---|
| SFT + GRPO (published ref) | Qwen3-4B-Base | MATH-500 | Pass@1(avg64)=78% (1 run), Pass@8=93% | [PEAR](./references/Good-SFT-Optimizes-for-SFT-Better-SFT-Prepares-for-Reinforcement-Learning/meta/meta_info.txt) | PEAR Table 2; different offline/RL data |
| PEAR(B=1) + GRPO (published ref) | Qwen3-4B-Base | MATH-500 | Pass@1(avg64)=80% (1 run), Pass@8=93% | [PEAR](./references/Good-SFT-Optimizes-for-SFT-Better-SFT-Prepares-for-Reinforcement-Learning/meta/meta_info.txt) | PEAR Table 2; different offline/RL data |
| Zero-shot | Qwen3-4B-Base | MATH-500 | Pass@1=**TBD** | - | To be run |
| Best-of-32 (base) | Qwen3-4B-Base | MATH-500 | Pass@32=**TBD** | - | To be run |
| Best-of-32 (Init-A) | Qwen3-4B-Base | MATH-500 | Pass@32=**TBD** | - | To be run |
| Best-of-32 (Init-B) | Qwen3-4B-Base | MATH-500 | Pass@32=**TBD** | - | To be run |
| Init-A (SFT only) | Qwen3-4B-Base | MATH-500 | Pass@1=**TBD** | - | To be run |
| Init-B (SFT only) | Qwen3-4B-Base | MATH-500 | Pass@1=**TBD** | - | To be run |
| GRPO from Init-A | Qwen3-4B-Base | MATH-500 | Pass@1=**TBD** (mean±std over 3 seeds) | - | Main comparison (baseline) |
| **GRPO from Init-B** | Qwen3-4B-Base | MATH-500 | Pass@1=**TBD** (mean±std over 3 seeds) | - | Main comparison (repetition-heavy init) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Smaller group | GRPO with G=2 instead of G=8 | If group statistics drive the effect, winner may change |

### Experimental Rigor

- **Seeds**: 3 seeds for each GRPO condition (`seeds=[42,123,456]`).
- **Top confounders + controls**:
  - Pre-RL success mismatch: difficulty-matched prompt set; re-check pass@4 on matched set.
  - Selection bias from filtering: report matched-set distribution; evaluate on fixed test sets (MATH-500, GSM8K).
  - Compute mismatch: equal SFT updates and equal GRPO update budget (same T).
  - Data leakage: hash/fuzzy-match MATH-500 and GSM8K against sampled SFT subset; remove collisions.
- **Sanity checks**: verifier accepts known-correct, rejects random; log parse failures separately.

---

## Success Criteria

**Hypothesis**:

After controlling pre-RL success, repetition-heavy long-CoT SFT will not improve (and may reduce) post-GRPO Pass@1 on MATH-500 relative to data-scaled SFT.

**Decision Rule**:

- **Continue/Proceed**: GRPO-from-Init-A > GRPO-from-Init-B on MATH-500 Pass@1 by a margin outside the 1-STD band across 3 seeds, and Init-B shows lower early mixed-group rate.
- **Pivot**: If results are within noise but diagnostics differ (mixed-group/parse rate), vary GRPO group size and/or KL coefficient to test whether group statistics are binding.
- **Refute**: If GRPO-from-Init-B > GRPO-from-Init-A outside the 1-STD band across 3 seeds, the mismatch hypothesis is wrong (repetition is a better initializer under this budget).

---

## Impact Statement

If repetition-heavy SFT is worse after controlling pre-RL success, practitioners should avoid maximizing offline long-CoT SFT via heavy repetition when the goal is GRPO training, and instead prioritize diversity or mismatch-correcting methods (e.g., PEAR-style objectives). If it is better, the result supports a cost-saving recipe: repeat smaller CoT datasets during SFT to improve RLVR outcomes.

---

## References

- [Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt) - Kopiczko et al., 2026
- [Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning](./references/Good-SFT-Optimizes-for-SFT-Better-SFT-Prepares-for-Reinforcement-Learning/meta/meta_info.txt) - Zhang et al., 2026
- [A Practical Two-Stage Recipe for Mathematical LLMs](./references/A-Practical-Two-Stage-Recipe-for-Mathematical-LLMs-Maximizing-Accuracy-with-SFT-and-Efficiency-with-Reinforcement-Learning/meta/meta_info.txt) - Yoshihara et al., 2025
- [SFT Memorizes, RL Generalizes](./references/SFT-Memorizes-RL-Generalizes-A-Comparative-Study-of-Foundation-Model-Post-training/meta/meta_info.txt) - Chu et al., 2025
- [RL Squeezes, SFT Expands](./references/RL-Squeezes-SFT-Expands-A-Comparative-Study-of-Reasoning-LLMs/meta/meta_info.txt) - Matsutani et al., 2025
- [Save the Good Prefix](./references/Save-the-Good-Prefix-Precise-Error-Penalization-via-Process-Supervised-RL-to-Enhance-LLM-Reasoning/meta/meta_info.txt) - Liu et al., 2026
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) - Shao et al., 2024
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - DeepSeek-AI, 2025
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
- [OpenRLHF](https://arxiv.org/abs/2405.11143) - Hu et al., 2024
- [MC-GRPO](https://arxiv.org/abs/2601.22582) - Kim, 2026
- [It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977) - Xu et al., 2025
- [Exploration vs Exploitation: Rethinking RLVR](https://arxiv.org/abs/2512.16912) - Chen et al., 2025
- [Mirage or Method?](https://arxiv.org/abs/2508.21188) - Wu et al., 2025
- [LIMO](https://arxiv.org/abs/2502.03387) - Ye et al., 2025
- [Small batch size training for language models](https://arxiv.org/abs/2504.02299) - Marek et al., 2025
- [Math-Verify](https://github.com/huggingface/Math-Verify) - Hugging Face, 2025
- [LightEval](https://github.com/huggingface/lighteval) - Hugging Face, 2024
- [MATH](https://arxiv.org/abs/1904.01557) - Hendrycks et al., 2019
- [GSM8K](https://arxiv.org/abs/2110.14168) - Cobbe et al., 2021
- [Murphy: Multi-Turn GRPO](https://arxiv.org/abs/2511.07833) - Murphy et al., 2025
