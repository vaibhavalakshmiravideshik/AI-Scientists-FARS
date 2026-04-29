# untitled

# Token-Balanced Continual Pretraining: Testing Whether “Brain Rot” Is Largely a Short-Sequence Optimization Artifact

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation
Continual pretraining (CPT) updates a language model by continuing next-token prediction training on new text, rather than retraining from scratch. CPT is widely used for domain adaptation and “keeping models fresh,” but it can also introduce capability drift.

A recent controlled study, **[LLMS CAN GET “BRAIN ROT”!](./references/LLMS-CAN-GET-BRAIN-ROT-!/meta/meta_info.txt)** (arXiv:2510.13928), reports that CPT on engagement-optimized Twitter/X posts (i.e., posts selected to maximize likes, retweets, and replies) causes large drops in reasoning and long-context retrieval. In their engagement-based “junk” definition (M1), junk posts are **short and popular**; control posts are **long and unpopular** at matched token scale (1.22M tokens).

Modern pretraining pipelines typically concatenate documents separated by an end-of-sequence token (EOS; a special token indicating document boundary) into fixed-length blocks (“sequence packing”), e.g., PaLM’s 2048-token packing with an end-of-document token (**[PaLM](https://arxiv.org/abs/2204.02311)**). However, the public Brain Rot reproduction configuration uses LLaMA-Factory with `per_device_train_batch_size=1` and `packing: False`, which implies one optimizer step per tweet.

### The Problem
Brain Rot is often interpreted as a semantic data-quality result (“junk social media text damages reasoning”). But the reported setup also creates an **objective mismatch** between “token-matched datasets” and “update-matched training”:

- With `packing=False`, each tweet becomes one training sample.
- Most training code computes cross-entropy as a **mean over tokens in the sample**.

If each sample contributes roughly equal total gradient weight, then tokens in a short tweet receive larger weight than tokens in a long tweet. For a tweet of length \(L\), the per-token loss weight is roughly \(1/L\). Therefore, for fixed total tokens, a dataset with shorter tweets induces larger **effective update per token**, even if the total token count is matched.

This matters because Brain Rot’s M1 “junk” condition is systematically shorter than control. Thus, part of the reported degradation may be due to **short-sequence overweighting** in the training objective rather than (or in addition to) the semantics of engagement-driven content.

### Key Insight and Hypothesis
**Key insight.** “Token-matching” the junk and control datasets does not guarantee “token-balanced learning” when training uses example-based batching and per-example mean loss. A single intervention that is already standard in pretraining—**packing tokens into fixed-length blocks**—should largely remove the short-sequence overweighting mechanism while keeping the underlying token stream and EOS boundaries the same.

**Hypothesis.** In Brain-Rot-style continual pretraining on short tweets, switching from one-tweet-per-sample training (`packing=False`) to token-balanced packed CPT (`packing=True`, 2048-token blocks) will recover a substantial fraction of the junk-induced capability loss on:
- ARC (AI2 Reasoning Challenge; multiple-choice science QA) with chain-of-thought prompting,
- RULER (a synthetic long-context retrieval/understanding suite) overall score.

If the degradation persists under token-balanced packing, the result supports the interpretation that the effect is primarily semantic/engagement-driven rather than a short-sequence optimization artifact.

---

## Proposed Approach

### Overview
We propose a minimal mechanism test by re-running Brain Rot’s CPT intervention with a single controlled change: **enable sequence packing**.

We train and evaluate three conditions:
- **A: Control, packed** (`packing=True`)
- **B: Junk, packed** (`packing=True`)
- **C: Junk, unpacked** (`packing=False`, matching the Brain Rot public config)

All runs use the same base model, same CPT dataset token scale, and the same post-CPT instruction tuning (Alpaca-5k) and evaluation protocol.

### Method Details
**Token-balanced CPT via packing.** In LLaMA-Factory, `packing=True` concatenates tokenized examples separated by EOS and chunks them into fixed-length sequences of `cutoff_len` (the maximum sequence length per training sample; here 2048). Compared to `packing=False` (one tweet per sample), packing makes the training objective closer to a per-token average objective:
- Each optimization step contains a similar number of tokens.
- The per-token contribution to the loss is less sensitive to tweet length.

**Implementation details (intended for verifier).**
- Use Brain Rot’s training recipe (AdamW, lr=1e-5, cosine schedule, bf16, effective batch size 8, 3 epochs; context length 2048) from the paper’s preprint section (**[Brain Rot Preprint](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/Preprint.md)**).
- For A/B/C, keep all hyperparameters the same except `packing`.

### Key Innovations
- A decisive and practically relevant mechanism test for CPT robustness: does Brain Rot’s degradation survive standard token-balanced pretraining mechanics?
- A concrete actionable guideline for practitioners: when continually pretraining on large volumes of short documents, treat token-balanced batching (packing or equivalent token-normalized loss) as a required control.

---

## Related Work

### Field Overview
This proposal sits at the intersection of (i) CPT/domain-adaptive pretraining, (ii) data-quality-driven capability drift, and (iii) training dynamics under variable-length sequences. Recent work suggests that seemingly small objective/normalization details can dominate outcomes, especially in regimes with short examples and small per-device batch sizes.

### Related Papers
- **[LLMS CAN GET “BRAIN ROT”!](./references/LLMS-CAN-GET-BRAIN-ROT-!/meta/meta_info.txt)**: Shows large reasoning and long-context declines after CPT on engagement-optimized short tweets vs long-tweet control.
- **[Brain Rot Preprint (training recipe)](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/Preprint.md)**: Documents the CPT+instruction-tuning pipeline (3 epochs CPT, then Alpaca-5k).
- **[RULER](https://arxiv.org/abs/2404.06654)**: Introduces a benchmark suite of long-context retrieval and aggregation tasks and an overall score aggregation.
- **[Think you have solved question answering? Try ARC](https://arxiv.org/abs/1803.05457)**: Introduces ARC (AI2 Reasoning Challenge), a multiple-choice science QA benchmark.
- **[Don’t Stop Pretraining](https://arxiv.org/abs/2004.10964)**: Establishes domain-adaptive pretraining as a standard method for adaptation and shows strong sensitivity to domain/data.
- **[QuRating](https://arxiv.org/abs/2402.04320)**: Uses model-based scoring to select higher-quality pretraining data; used by Brain Rot for high-quality criteria.
- **[CCNet](https://arxiv.org/abs/1911.00359)**: Classic large-scale web data filtering pipeline and evidence that filtering choices impact downstream quality.
- **[SemDeDup](https://arxiv.org/abs/2303.09540)**: Shows semantic deduplication improves training efficiency and downstream generalization.
- **[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)**: Empirically demonstrates dedup benefits for LM training.
- **[Fine-tuning aligned language models compromises safety](https://arxiv.org/abs/2303.13362)**: Shows benign fine-tuning can shift safety properties, motivating safety measurement in CPT.
- **[Chinchilla](https://arxiv.org/abs/2203.15556)**: Emphasizes token budgeting and compute-optimal training, motivating token-based accounting.
- **[PaLM](https://arxiv.org/abs/2204.02311)**: Uses document packing into fixed-length blocks; precedent for token-balanced batching.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)**: Frames tokens as the primary driver of scaling, reinforcing token-based comparability.
- **[Dataset Decomposition](https://arxiv.org/abs/2405.13226)**: Proposes variable sequence length curricula and highlights length/efficiency trade-offs.
- **[GrowLength](https://arxiv.org/abs/2310.00576)**: Shows length curricula can improve training efficiency and stability.
- **[Hydraulis](https://arxiv.org/abs/2412.07894)**: Analyzes workload/throughput imbalances in variable-length training and motivates packing.
- **[Token Weighting for Long-Range Language Modeling](https://arxiv.org/abs/2503.09202)**: Introduces token-level weighting strategies, showing token reweighting can change long-range behavior.
- **[The EOS Decision and Length Extrapolation](./references/The-EOS-Decision-and-Length-Extrapolation/meta/meta_info.txt)**: Analyzes how EOS modeling affects length generalization.
- **[Controlling Summarization Length Through EOS Token Weighting](./references/Controlling-Summarization-Length-Through-EOS-Token-Weighting/meta/meta_info.txt)**: Shows EOS loss weighting controls termination behavior.
- **[Explaining length bias in LLM-based preference evaluations](https://arxiv.org/abs/2404.17853)**: Studies how LLM-based evaluators can prefer longer text, supporting length as a confound in LLM studies.
- **[Verbosity bias in preference labeling by LLMs](https://arxiv.org/abs/2310.09278)**: Shows preference labeling can be biased by length/verbosity, reinforcing the need to control for length.
- **[Data Repetition Beats Data Scaling in Long-CoT SFT](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt)**: Demonstrates that training dynamics at batch size 1 can dominate outcomes, motivating similar confound checks in CPT.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| CPT & adaptation | Continue pretraining for domain/task adaptation | Don’t Stop Pretraining, Brain Rot | downstream task scores | Susceptible to drift/forgetting |
| Data quality pipelines | Filter/dedup/select data to improve training | QuRating, CCNet, SemDeDup, Dedup | diverse downstream eval | Filtering may miss non-semantic harms |
| Length/packing dynamics | Training objective depends on sequence length and batching | GrowLength, Dataset Decomposition, Hydraulis, PaLM | throughput + downstream perf | Often studied for efficiency, less for capability drift |
| Termination/boundary effects | EOS-related signals influence generation length | EOS Decision, EOS token weighting | length metrics, task accuracy | Usually studied in post-training, not CPT |

### Closest Prior Work
- **Brain Rot (2510.13928)**: Demonstrates large degradations from engagement-based junk CPT with a public recipe using `packing=False`, but does not test whether the effect persists under standard packed/token-balanced CPT.
- **Hydraulis (2412.07894)**: Highlights that variable-length training without packing can skew compute and optimization dynamics, but does not study CPT-induced capability decline.
- **Long-CoT SFT training-dynamics work (2602.11149)**: Shows normalization/step dynamics can dominate outcomes in small-batch post-training; we test an analogous effect in CPT.

**Novelty Kill Search Summary:**
- Local KB search: `Grep` over proposal indexes and KB for “brain rot” and for “packing + continual pretraining” found no existing proposal or topic file addressing this specific mechanism test.
- Web queries (2026-02-19): searched for “brain rot packing”, “token-balanced continual pretraining”, “sequence-weighted loss variable length language model”, “tweet pretraining packing”, “short sequence overweighting language model training”, and “token-normalized loss continual pretraining”; also checked OpenReview and GitHub search for “packing continual pretraining” and “token-balanced loss”. No close prior work explicitly reframing Brain Rot as a packing/token-balance artifact was found as of 2026-02-19. (Full query log in `notes.md`.)

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [Brain Rot](./references/LLMS-CAN-GET-BRAIN-ROT-!/meta/meta_info.txt) | CPT on junk vs control tweets shows degradations | Does not control for update-per-token confound from short tweets | Re-run with token-balanced packing | Directly tests whether degradation is an optimization artifact |
| [QuRating](https://arxiv.org/abs/2402.04320) | Filters training data by predicted quality | Does not isolate optimization confounds | No filtering; only batching/objective change | Separates “data is bad” from “training recipe is bad” |
| [Hydraulis](https://arxiv.org/abs/2412.07894) | Studies variable-length training imbalance | Not framed as capability-drift mechanism | Apply packing specifically to CPT drift | Uses a standard intervention with decisive downstream test |

---

## Experiments

### Experimental Setup
**Codebase:** Brain Rot reproduction repo + LLaMA-Factory.

**Baseline Ladder (REQUIRED):**
- **Prompting / no-CPT baseline:** Base instruct model with Alpaca-5k instruction tuning only (no CPT), evaluated with the same prompts.
- **Inference-time scaling baseline:** Best-of-N self-consistency for ARC (e.g., N=8 samples, majority vote). If no published numbers exist for this exact setting, mark as needs re-run.
- **Closest existing method:** Brain Rot’s original CPT recipe (`packing=False`) as the strongest “status quo” baseline for the phenomenon.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Meta-Llama-3-8B-Instruct | 8B | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct | May require gated access |
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Fallback if Llama access is restricted |

**Training conditions (3 main conditions; 3 seeds each):**
- **A: Control, packed**: CPT on M1 control tweets with `packing=True`, then Alpaca-5k instruction tuning.
- **B: Junk, packed**: CPT on M1 junk tweets with `packing=True`, then Alpaca-5k instruction tuning.
- **C: Junk, unpacked**: CPT on M1 junk tweets with `packing=False` (Brain Rot config), then Alpaca-5k instruction tuning.

**CPT + instruction-tuning recipe (from Brain Rot):**
- CPT: full-parameter AdamW, lr=1e-5, cosine schedule, bf16, effective batch size 8, 3 epochs, `cutoff_len=2048`.
- Instruction tuning: Alpaca English (5k), 3 epochs, lr=1e-5, cosine decay, bf16, effective batch size 16, `cutoff_len=2048`.
- Brain Rot reports training on NVIDIA H100 GPUs (**[Brain Rot Preprint](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/Preprint.md)**); we run on A100s.

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Brain Rot M1 control tweets | CPT control | ~1.22M tokens | https://github.com/llm-brain-rot/llm-brain-rot | Repo license (check) |
| Brain Rot M1 junk tweets | CPT junk | ~1.22M tokens | https://github.com/llm-brain-rot/llm-brain-rot | Repo license (check) |
| Alpaca English (5k) | Post-CPT instruction tuning | 5k examples | https://github.com/tatsu-lab/stanford_alpaca | CC BY-NC 4.0 (check) |

**Resource Estimate** (must be ≤768 A100 GPU-hours):
- Brain Rot’s CPT token scale is small (~1.22M tokens; 3 epochs). Even with packing to 2048-token sequences, the total token count remains small.
- Expected budget: ≤200 A100 GPU-hours for 3 conditions × 3 seeds × (CPT + IT) + evaluation.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ARC (AI2 Reasoning Challenge) | Multiple-choice science QA benchmark | Accuracy (higher is better) | test | https://allenai.org/data/arc | https://github.com/EleutherAI/lm-evaluation-harness |
| RULER | Synthetic long-context retrieval/understanding suite | Overall score / accuracy (higher is better) | standard | https://github.com/hsiehjackson/RULER | official repo |

### Main Results

#### Results Table
Numbers from prior work are copied from the Brain Rot paper’s raw results section (**[Brain Rot main results](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/MAIN%20RESULTS%20JUNK%20INTERVENTION%20AND%20COGNITIVE%20DECLINES%20ARE%20ASSOCIATED.md)**). Brain Rot does not report variance; mark these as single-run references.

| Method | Base Model | Benchmark | Metric (mean±std) | Source | Notes |
|---|---|---|---:|---|---|
| Brain Rot: Base + Alpaca-5k (no CPT) | Llama3-8B-Instruct | ARC-Challenge (CoT) | 74.9 (1 run) | [Brain Rot Table 2](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/MAIN%20RESULTS%20JUNK%20INTERVENTION%20AND%20COGNITIVE%20DECLINES%20ARE%20ASSOCIATED.md) | M1 setting |
| Brain Rot: Control CPT (0% junk) + Alpaca-5k | Llama3-8B-Instruct | ARC-Challenge (CoT) | 74.9 (1 run) | [Brain Rot Table 2](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/MAIN%20RESULTS%20JUNK%20INTERVENTION%20AND%20COGNITIVE%20DECLINES%20ARE%20ASSOCIATED.md) | M1 control |
| Brain Rot: Junk CPT (100% junk) + Alpaca-5k | Llama3-8B-Instruct | ARC-Challenge (CoT) | 57.2 (1 run) | [Brain Rot Table 2](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/MAIN%20RESULTS%20JUNK%20INTERVENTION%20AND%20COGNITIVE%20DECLINES%20ARE%20ASSOCIATED.md) | M1 junk |
| Brain Rot: Base + Alpaca-5k (no CPT) | Llama3-8B-Instruct | RULER Overall | 90.5 (1 run) | [Brain Rot Table 2](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/MAIN%20RESULTS%20JUNK%20INTERVENTION%20AND%20COGNITIVE%20DECLINES%20ARE%20ASSOCIATED.md) | 4K context |
| Brain Rot: Control CPT (0% junk) + Alpaca-5k | Llama3-8B-Instruct | RULER Overall | 90.5 (1 run) | [Brain Rot Table 2](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/MAIN%20RESULTS%20JUNK%20INTERVENTION%20AND%20COGNITIVE%20DECLINES%20ARE%20ASSOCIATED.md) | 4K context |
| Brain Rot: Junk CPT (100% junk) + Alpaca-5k | Llama3-8B-Instruct | RULER Overall | 71.0 (1 run) | [Brain Rot Table 2](./references/LLMS-CAN-GET-BRAIN-ROT-!/sections/MAIN%20RESULTS%20JUNK%20INTERVENTION%20AND%20COGNITIVE%20DECLINES%20ARE%20ASSOCIATED.md) | 4K context |
| Best-of-8 self-consistency (no CPT) | Llama3-8B-Instruct | ARC-Challenge (CoT) | **TBD** | - | **Needs re-run** (no published number in Brain Rot) |
| **A: Control, packed** | Llama3-8B-Instruct | ARC-Challenge (CoT) | **TBD** | - | To be verified (3 seeds) |
| **B: Junk, packed** | Llama3-8B-Instruct | ARC-Challenge (CoT) | **TBD** | - | To be verified (3 seeds) |
| **C: Junk, unpacked** | Llama3-8B-Instruct | ARC-Challenge (CoT) | **TBD** | - | To be verified (3 seeds); should reproduce Brain Rot trend |
| **A: Control, packed** | Llama3-8B-Instruct | RULER Overall | **TBD** | - | To be verified (3 seeds) |
| **B: Junk, packed** | Llama3-8B-Instruct | RULER Overall | **TBD** | - | To be verified (3 seeds) |
| **C: Junk, unpacked** | Llama3-8B-Instruct | RULER Overall | **TBD** | - | To be verified (3 seeds) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B vs C | Enable packing only for junk CPT | If short-sequence overweighting drives Brain Rot, B outperforms C |
| A vs (Brain Rot control) | Packing for control CPT | Control should change little if the effect is primarily driven by short tweets |

### Experimental Rigor
- **Seeds:** `seeds=[42, 123, 456]` for all A/B/C.
- **Validity threats / confounds:**
  - **Packing changes context segmentation** (tweets are concatenated). Control: EOS separators preserved; this mirrors standard pretraining practice.
  - **Reproduction mismatch** (hardware, library versions). Control: include a reproduction check that C approximately matches Brain Rot’s reported directionality.
  - **Benchmark contamination** (ARC/RULER in pretraining). Control: we evaluate *relative* changes from the same base model; contamination is unlikely to explain differential drift between A/B/C.
- **Sanity checks:**
  - Ensure the no-CPT baseline roughly matches the published Brain Rot baseline on at least one of ARC-CoT or RULER (within a reasonable tolerance).
  - Confirm that best-of-8 improves ARC accuracy over greedy decoding (if it does not, the prompting/eval setup is likely incorrect).
- **Fair comparison:** Same base model, same total CPT tokens, same instruction-tuning procedure, same evaluation prompts; for best-of-N, use identical N across all rows.

---

## Success Criteria

**Hypothesis** (directional): Token-balanced packed CPT reduces the junk-induced capability drop compared to the unpacked Brain Rot recipe.

**Decision Rule** (concrete):
- Let `ARC_gap = ARC(A) - ARC(C)` and `ARC_recovered = ARC(B) - ARC(C)` (higher is better).
- Let `RULER_gap = RULER(A) - RULER(C)` and `RULER_recovered = RULER(B) - RULER(C)` (higher is better).
- **Proceed:** `ARC_recovered ≥ 0.70 × ARC_gap` and `RULER_recovered ≥ 0.70 × RULER_gap` (mean across 3 seeds), and B is not worse than C on either metric.
- **Pivot:** recovery is mixed (30–70%) → report length-bucket analysis (short vs long tweets) to see if the residual correlates with length or popularity.
- **Refute:** `ARC_recovered < 0.30 × ARC_gap` and `RULER_recovered < 0.30 × RULER_gap` → conclude the Brain Rot degradation is not primarily explained by short-sequence overweighting in the CPT recipe.

---

## Impact Statement
If Brain Rot’s degradation largely disappears under token-balanced packed CPT, practitioners doing continual pretraining on short-document streams (social media, chat logs, short code snippets) should treat packing or token-normalized loss as a required control before attributing capability loss to semantics. If the degradation persists, the result strengthens the claim that engagement-driven data distributions can induce capability drift even under standard token-balanced training.

---

## References

- [LLMS CAN GET “BRAIN ROT”!](./references/LLMS-CAN-GET-BRAIN-ROT-!/meta/meta_info.txt) - Xing et al., 2024
- [The EOS Decision and Length Extrapolation](./references/The-EOS-Decision-and-Length-Extrapolation/meta/meta_info.txt) - (see paper metadata)
- [Controlling Summarization Length Through EOS Token Weighting](./references/Controlling-Summarization-Length-Through-EOS-Token-Weighting/meta/meta_info.txt) - (see paper metadata)
- [Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning](./references/Data-Repetition-Beats-Data-Scaling-in-Long-CoT-Supervised-Fine-Tuning/meta/meta_info.txt) - Kopiczko et al., 2026
- [PaLM](https://arxiv.org/abs/2204.02311) - Chowdhery et al., 2022
- [Chinchilla](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022
- [Don’t Stop Pretraining](https://arxiv.org/abs/2004.10964) - Gururangan et al., 2020
- [RULER](https://arxiv.org/abs/2404.06654) - Hsieh et al., 2024
- [ARC (AI2 Reasoning Challenge)](https://arxiv.org/abs/1803.05457) - Clark et al., 2018
- [QuRating](https://arxiv.org/abs/2402.04320) - Wettig et al., 2024
- [Hydraulis](https://arxiv.org/abs/2412.07894) - (see arXiv)
- [GrowLength](https://arxiv.org/abs/2310.00576) - (see arXiv)
- [Dataset Decomposition](https://arxiv.org/abs/2405.13226) - (see arXiv)
- [Token Weighting for Long-Range Language Modeling](https://arxiv.org/abs/2503.09202) - Helm et al., 2025
- [SemDeDup](https://arxiv.org/abs/2303.09540) - Abbas et al., 2023
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) - Lee et al., 2021
- [CCNet](https://arxiv.org/abs/1911.00359) - Wenzek et al., 2019
- [Fine-tuning aligned language models compromises safety](https://arxiv.org/abs/2303.13362) - Qi et al., 2023
- [Explaining length bias in LLM-based preference evaluations](https://arxiv.org/abs/2404.17853) - Hu et al., 2024
- [Verbosity bias in preference labeling by LLMs](https://arxiv.org/abs/2310.09278) - Saito et al., 2023
