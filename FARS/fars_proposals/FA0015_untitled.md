# untitled

# Orthogonal Junk: Gradient-Orthogonality Data Selection to Prevent Capability Degradation in Continual Pre-Training

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly updated via **continual pre-training (CPT)**: continuing next-token prediction training on new corpora, rather than restarting pre-training from scratch. CPT is attractive because it can incorporate fresh data (new languages, new domains, new events) at lower cost than full retraining.

However, CPT can also **degrade previously learned capabilities** when the new data distribution is misaligned with desired behaviors. A recent controlled study, **"[LLMS CAN GET BRAIN ROT !](./references/LLMS-CAN-GET-BRAIN-ROT-!/meta/meta_info.txt)"** (arXiv:2510.13928), shows that continuing pre-training on low-quality, engagement-optimized Twitter data causes large drops in reasoning and long-context understanding, and increases safety risks. The paper finds that these degradations are persistent even after substantial post-hoc instruction tuning and additional clean-data training.

In parallel, a separate line of work on **catastrophic forgetting in LLM adaptation** suggests that forgetting is strongly linked to **gradient conflicts** between new training updates and gradients that preserve general capabilities. **Orthogonal Gradient Selection (OGS)**, **"[Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation](./references/Training-Data-Selection-with-Gradient-Orthogonality-for-Efficient-Domain-Adaptation/meta/meta_info.txt)"** (arXiv:2602.06359), proposes selecting training examples whose gradients are near-orthogonal to a "general capability" anchor gradient, improving domain adaptation while preserving performance on general benchmarks.

This proposal tests a concrete, decision-relevant question: **can gradient-geometry-aware data selection prevent the capability degradation observed under junk-data continual pre-training?** If yes, it suggests a practical prevention mechanism for production pipelines that ingest large volumes of noisy web or social-media text. If no, it provides evidence that some low-quality distributions cause representational drift that is not easily prevented by per-example filtering.

### The Problem

Xing et al. (2025) demonstrate a causal effect of low-quality social media continual pre-training (CPT) on capability decline. For example (Llama3-8B-Instruct, engagement-based junk definition M1), ARC-Challenge accuracy with chain-of-thought prompting drops from **74.9** (0% junk / control) to **57.2** (100% junk), and RULER overall drops from **90.5** to **71.0** (Table 2 in their paper).

A practitioner can respond in two ways:
1) **Remediate after the fact** (instruction tuning / clean CPT), which Brain Rot finds only partially restores performance.
2) **Prevent the damage during CPT** by filtering or weighting incoming data.

Many existing data-quality filters are heuristic (length, perplexity, semantic quality classifiers). It is unclear whether they target the core mechanism that causes capability decline. Meanwhile, OGS-style methods provide a mechanistic signal (gradient conflicts) but have only been tested for **domain adaptation** with relatively high-quality target data, not for filtering a distribution that may be broadly harmful.

We aim to test whether a simple, pre-registered OGS-style filtering rule can prevent capability degradation under the same token budget, without requiring human labeling.

### Key Insight and Hypothesis

**Key insight**: If Brain Rot degradation is driven by gradient updates that conflict with general reasoning/long-context capabilities, then restricting CPT to examples whose gradients lie in (or near) the **safe subspace** orthogonal to a general-capability anchor gradient should reduce or prevent the decline.

**Hypothesis**: *Continual pre-training on engagement-optimized junk tweets filtered by gradient orthogonality to a general-capability anchor will preserve reasoning (ARC) and long-context understanding (RULER) substantially better than random junk CPT at the same token budget.*

This could fail if (i) the brain-rot effect does not arise at our chosen model/compute scale, (ii) orthogonality does not separate harmful vs less harmful updates in this data regime, or (iii) the degradation is not primarily explained by first-order gradient conflicts (e.g., it requires long-run distributional drift that is not avoidable by per-sample filtering).

---

## Proposed Approach

### Overview

We propose **Orthogonal Junk**, an OGS-style data filtering method for continual pre-training on low-quality social media:

1. Construct a **general-capability anchor gradient** from a small anchor set (GSM8K + MMLU + Alpaca), as in OGS.
2. For each candidate junk tweet, compute its gradient on a small **navigator model** and measure cosine similarity to the anchor gradient.
3. Select a fixed top fraction (pre-registered) of tweets with the highest **orthogonality score**.
4. Run CPT on the selected tweets (with replacement if needed to match token budget), then instruction-tune, and evaluate on ARC and RULER.

### Method Details

**Anchor gradient (general capabilities).**
Following OGS, we define an anchor dataset \(D_{\text{anchor}}\) of 300–500 examples covering:
- GSM8K (math word problems)
- MMLU (multiple-choice world knowledge)
- Alpaca (instruction-following)

Let \(g_{\text{ref}}\) be the mean gradient of the LM loss over \(D_{\text{anchor}}\).

**Orthogonality score.**
For a candidate sample \(x_i\) with gradient \(g_i\), define:
\[
\text{Orth}(x_i) = 1 - |\cos(g_i, g_{\text{ref}})|.
\]

**Selection rule (pre-registered).**
We select the top \(p=10\%\) of junk samples by \(\text{Orth}(x_i)\) computed on the navigator model (same-family, smaller scale). This selection rate is fixed (no hyperparameter sweep).

**Edge-case rule.**
If the top-10% selection yields an effective retained fraction <5% or >50% due to preprocessing constraints (e.g., tokenization failures), we report this as a finding about junk-anchor geometry and do not tune thresholds.

**Training with replacement.**
If the selected set contains fewer tokens than the target CPT budget, we sample with replacement to match the same number of training tokens/steps as the random-junk CPT baseline. We will explicitly report repetition rate as a potential confound.

### Key Innovations

- **Problem reframing**: Treats Brain-Rot-style capability decline as a *preventable* gradient-interference problem during continual pre-training, not only as a post-hoc mitigation problem.
- **Cross-domain generalization of gradient-orthogonality selection**: Extends OGS from domain adaptation on curated data to filtering an intrinsically low-quality distribution (engagement-optimized social media).
- **Decisive, pre-registered test**: Uses a fixed selection rate and explicit abort criteria to avoid post-hoc threshold tuning.

---

## Related Work

### Field Overview

Continual learning research studies how models can learn from sequential data without catastrophic forgetting. For LLMs, forgetting often appears as degraded reasoning, factual recall, instruction following, or safety after adaptation or CPT. Common approaches include replay (store and interleave old data), regularization (constrain parameter drift), parameter isolation (separate adapters or experts per task), and data selection.

Gradient-based methods provide a mechanistic lens: first-order performance change on a protected objective is governed by the inner product between the update gradient and the protected-gradient direction. This motivates gradient surgery (project away conflicting components) and, more recently, **data-centric** approaches that select examples likely to produce safe updates.

Our proposal combines these threads with a recent controlled study on data-quality-induced degradation (Brain Rot), aiming to test whether gradient-geometry signals can prevent such degradation in a fully automated pipeline.

### Related Papers

- **[LLMS CAN GET BRAIN ROT !](./references/LLMS-CAN-GET-BRAIN-ROT-!/meta/meta_info.txt)**: Controlled experiments show continual pre-training on junk Twitter data causes persistent declines in reasoning, long-context understanding, and safety.
- **[Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation](./references/Training-Data-Selection-with-Gradient-Orthogonality-for-Efficient-Domain-Adaptation/meta/meta_info.txt)**: Proposes OGS, selecting domain training samples whose gradients are orthogonal to a general-capability anchor gradient to reduce forgetting.
- **[Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models](./references/Revisiting-Replay-and-Gradient-Alignment-for-Continual-Pre-Training-of-Large-Language-Models/meta/meta_info.txt)**: Shows that experience replay and Reptile-style gradient alignment (MER) stabilize continual pre-training across large-scale multilingual sequences.
- **[ELO Efficient Layer-Specific Optimization for Continual Pretraining of Multilingual LLMs](./references/ELO-Efficient-Layer-Specific-Optimization-for-Continual-Pretraining-of-Multilingual-LLMs/meta/meta_info.txt)**: Improves CPT efficiency by training only detached first/last layers, addressing forward-pass costs.
- **[Merge before Forget A Single LoRA Continual Learning via Continual Merging](../../papers/paper_summaries/Merge%20before%20Forget%20A%20Single%20LoRA%20Continual%20Learning%20via%20Continual%20Merging.md)**: Uses a single continually merged LoRA to mitigate forgetting with constant memory.
- **[How Do Large Language Models Learn Concepts During Continual Pre-Training?](../../papers/paper_summaries/How%20Do%20Large%20Language%20Models%20Learn%20Concepts%20During%20Continual%20Pre-Training.md)**: Links internal concept circuits to learning/forgetting dynamics during CPT.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Reflection-style iterative self-critique, used as inspiration for Brain Rot mitigation experiments.
- **[LoRA](https://arxiv.org/abs/2106.09685)**: Parameter-efficient adaptation method used by OGS for resource-constrained fine-tuning.
- **[GEM](https://arxiv.org/abs/1706.08840)**: Gradient episodic memory, a classic gradient-surgery method for continual learning.
- **[PCGrad](https://arxiv.org/abs/2001.06782)**: Projects conflicting gradients to reduce interference in multi-task learning.
- **[EWC](https://arxiv.org/abs/1612.00796)**: Regularization-based continual learning using Fisher information.
- **[RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)**: Introduces RULER, a synthetic long-context evaluation suite (retrieval, aggregation, variable tracking) used by Brain Rot to measure long-context capability degradation.
- **[GSM8K](https://arxiv.org/abs/2110.14168)**: Math reasoning benchmark used by OGS as a protected capability.
- **[MMLU](https://arxiv.org/abs/2009.03300)**: Broad knowledge benchmark used by OGS as a protected capability.
- **[Alpaca](https://arxiv.org/abs/2304.04333)**: Instruction-following dataset used as anchor examples and for SFT in Brain Rot.

- **[LESS: Low-rank Gradient Similarity Search for Instruction Data Selection](https://arxiv.org/abs/2402.04333)**: Influence-inspired gradient-similarity selection for LLM instruction tuning; a relevant baseline for gradient-based data selection on junk.
- **[Learn More, Forget Less: A Gradient-Aware Data Selection Approach for LLM](https://arxiv.org/abs/2511.08620)**: Proposes GrADS, a gradient-magnitude / density based selector that improves data efficiency and reduces forgetting; a strong non-orthogonality gradient baseline.
- **[Data Selection for Language Models via Importance Resampling](https://arxiv.org/abs/2302.03169)**: DSIR uses n-gram importance resampling to match a target distribution; a cheap non-gradient selection baseline family.
- **[DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)**: Provides a benchmark and strong baselines for large-scale filtering and mixing, relevant to evaluating data-quality filters.
- **[Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs](./references/Beyond-Data-Filtering-Knowledge-Localization-for-Capability-Removal-in-LLMs/meta/meta_info.txt)**: Introduces SGTM, which uses selective gradient masking to localize and later ablate targeted capabilities under label noise; a parameter-isolation alternative to filtering.
- **[SafeGrad: Gradient Surgery for Safe LLM Fine-Tuning](https://arxiv.org/abs/2508.07172)**: Uses gradient projection to resolve conflicts between safety and utility objectives during fine-tuning; supports the gradient-conflict framing.
- **[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)**: Shows that removing duplicates can improve generalization and reduce memorization, motivating simple data-cleaning baselines.
- **[SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/abs/2303.09540)**: Removes semantic near-duplicates and improves data efficiency, another simple baseline for data curation.
- **[CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/abs/1911.00359)**: Classic pipeline using perplexity and heuristics to filter Common Crawl, motivating perplexity-based filtering baseline.
- **[Data Selection with Importance Resampling (DSIR)](https://github.com/p-lambda/dsir)**: Open-source implementation supporting practical replication of importance-resampling selection.


### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Data-quality degradation in CPT | Low-quality data causes capability decline | Brain Rot | ARC, RULER, safety/personality probes | Mechanism and prevention underexplored |
| Gradient-geometry data selection | Use gradient alignment/orthogonality to select safe data | OGS; LESS; GRADS | Domain QA + GSM8K/MMLU retention | Not tested on intrinsically low-quality distributions |
| Replay and gradient alignment | Interleave old data; align gradients via meta-learning | MER (Revisiting Replay + Gradient Alignment) | Continual pretraining across tasks/languages | Replay storage + compute overhead; not a filter for low-quality data |

### Closest Prior Work

1. **Brain Rot (2510.13928)** shows the degradation phenomenon and tests post-hoc mitigations (reflection, instruction tuning, clean CPT), but does not test prevention via gradient-based filtering.
2. **OGS (2602.06359)** provides a concrete, efficient gradient-orthogonality criterion for data selection to reduce forgetting during domain adaptation, but does not evaluate capability degradation from low-quality social media CPT.
3. **MER / replay+gradient alignment (2508.01908)** stabilizes CPT via replay and Reptile-style updates, but does not address data-quality-induced degradation or filtering.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Brain Rot | Shows junk CPT causes persistent capability decline | No prevention mechanism tested | Apply gradient-geometry filtering during CPT | If degradation is driven by gradient conflicts, filtering should prevent it |
| OGS | Selects safe domain data via orthogonality to anchor | Not tested on low-quality social data | Apply OGS-style orthogonality to junk tweets | Extends a mechanistic criterion to a new, safety-relevant failure mode |
| MER | Replay + gradient alignment stabilizes CPT | Needs replay buffer; not a data-quality filter | Use a filter to avoid harmful updates in the first place | Could reduce degradation with less reliance on replay |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Llama-3.2-1B-Instruct (or similar) | ~1B | https://huggingface.co/meta-llama | Prefer small model to fit budget; exact checkpoint TBD |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| Brain Rot M1 junk/control tweets | CPT on junk vs control | ~1.22M tokens per condition | https://github.com/llm-brain-rot/llm-brain-rot | Apache-2.0 (repo) |
| Alpaca (5k or 50k) | Instruction tuning after CPT | 5k–50k | https://github.com/tatsu-lab/stanford_alpaca | CC BY-NC 4.0 (check) |
| Anchor set (GSM8K+MMLU+Alpaca subset) | Compute anchor gradient | 300–500 examples | https://huggingface.co/datasets/openai/gsm8k ; https://huggingface.co/datasets/cais/mmlu | Dataset licenses vary |

**Compute / training details (initial plan):**
- CPT objective: next-token prediction on tweets (as in Brain Rot).
- SFT: Alpaca 5k (matching Brain Rot default) for all compared conditions.
- Selection gradients computed on navigator model; to reduce cost, compute gradients w.r.t. a small parameter subset (LM head + embeddings) as an approximation.

**Resource Estimate** (must fit 768 GPU-hours):
- Target model ≤1B; CPT tokens ~1.22M; epochs may be increased up to 10 if needed to induce measurable degradation.
- Expected total: ≤200 GPU-hours (including selection gradient computation + 2–3 training runs + evaluations). Exact estimate will be refined from pilot throughput.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ARC (AI2 Reasoning Challenge) | Multiple-choice science QA; Brain Rot uses accuracy with and without CoT prompting | Accuracy (0–100) | test | https://allenai.org/data/arc | lm-evaluation-harness |
| RULER | Synthetic long-context retrieval/understanding tasks | Accuracy (0–100) | standard | https://github.com/hsiehjackson/RULER | official / lm-eval |

Primary metrics:
- ARC-Challenge accuracy with CoT prompt (Brain Rot’s main reasoning probe)
- RULER overall score (and optionally a small subset like CWE / NIAH)

### Main Results

We will compare three conditions (≤3 main conditions):

| Method | CPT data | Selection | Token budget | ARC-Challenge (CoT) | RULER Overall | Source |
|---|---|---|---:|---:|---:|---|
| Control CPT | Control tweets | none | matched | **TBD** | **TBD** | To be run |
| Junk-Random CPT | Junk tweets | random | matched | **TBD** | **TBD** | To be run |
| **Ours (Orthogonal Junk)** | Junk tweets | top-10% orthogonality by anchor | matched | **TBD** | **TBD** | To be run |

Additional baselines (run separately if budget allows; not part of the 3-condition decisive test):
- **Perplexity-filtered junk**: keep the top-10% *lowest perplexity* junk tweets under the base model (a standard heuristic data-quality baseline).
- **LESS-selected junk**: select top-k junk tweets by influence / gradient similarity to the anchor set using LESS (Xia et al., 2024; arXiv:2402.04333).
- **SGTM-style routing**: if feasible in this CPT setting, route junk gradients into a designated parameter subset using selective gradient masking (Shilov et al., 2025; arXiv:2512.05648) to test whether parameter isolation is more effective than data filtering.

Published reference point (different model, Llama3-8B, Brain Rot Table 2; M1):
- Control CPT: ARC-Challenge(CoT)=74.9; RULER Overall=90.5
- Junk CPT: ARC-Challenge(CoT)=57.2; RULER Overall=71.0

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (full) | top-10% orth filtering | Best retention |
| w/ different anchor composition | remove one anchor component (e.g., no GSM8K) | If anchor matters, retention degrades |

### Analysis (Optional)

- Measure distribution shift between selected vs full junk data (length, popularity) and correlation with Orth score.
- Report repetition rate when sampling with replacement.

---

## Success Criteria

**Criterion 1: Disease induction is measurable**
- Hypothesis: Junk-Random CPT reduces ARC-Challenge(CoT) by ≥5 points vs Control CPT under the pre-registered CPT budget.
- Validation: If not observed, we abort and report that the brain-rot effect did not reproduce at this scale.

**Criterion 2: Orthogonal Junk improves retention**
- Hypothesis: Orthogonal Junk outperforms Junk-Random on both ARC-Challenge(CoT) and RULER.
- Validation: Improvement is ≥2 standard errors (bootstrap over evaluation items) on both metrics.

---

## Impact Statement

If successful, this work suggests that **gradient-geometry filtering** can be used as a practical safeguard in continual pre-training pipelines that ingest low-quality web/social media data, reducing the risk of capability degradation without human labeling. If it fails, it provides evidence that Brain-Rot-style degradation is not mitigated by first-order gradient-conflict filtering, informing the design of more robust continual learning systems.

---

## References

- [LLMS CAN GET BRAIN ROT !](./references/LLMS-CAN-GET-BRAIN-ROT-!/meta/meta_info.txt) - Xing et al., 2025
- [Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation](./references/Training-Data-Selection-with-Gradient-Orthogonality-for-Efficient-Domain-Adaptation/meta/meta_info.txt) - Zhang et al., 2026
- [Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models](./references/Revisiting-Replay-and-Gradient-Alignment-for-Continual-Pre-Training-of-Large-Language-Models/meta/meta_info.txt) - Abbes et al., 2025
- [ELO Efficient Layer-Specific Optimization for Continual Pretraining of Multilingual LLMs](./references/ELO-Efficient-Layer-Specific-Optimization-for-Continual-Pretraining-of-Multilingual-LLMs/meta/meta_info.txt) - Yoo et al., 2026
- [Merge before Forget A Single LoRA Continual Learning via Continual Merging](../../papers/paper_summaries/Merge%20before%20Forget%20A%20Single%20LoRA%20Continual%20Learning%20via%20Continual%20Merging.md) - Qiao & Mahdavi, 2025
- [How Do Large Language Models Learn Concepts During Continual Pre-Training?](../../papers/paper_summaries/How%20Do%20Large%20Language%20Models%20Learn%20Concepts%20During%20Continual%20Pre-Training.md) - Yao et al., 2026
- [GEM](https://arxiv.org/abs/1706.08840) - Lopez-Paz & Ranzato, 2017
- [PCGrad](https://arxiv.org/abs/2001.06782) - Yu et al., 2020
- [EWC](https://arxiv.org/abs/1612.00796) - Kirkpatrick et al., 2017
- [Reptile](https://arxiv.org/abs/1803.02999) - Nichol et al., 2018
- [GSM8K](https://arxiv.org/abs/2110.14168) - Cobbe et al., 2021
- [MMLU](https://arxiv.org/abs/2009.03300) - Hendrycks et al., 2021
- [Alpaca](https://arxiv.org/abs/2304.04333) - Taori et al., 2023
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) - Hsieh et al., 2024
- [LESS: Low-rank Gradient Similarity Search for Instruction Data Selection](https://arxiv.org/abs/2402.04333) - Xia et al., 2024
- [Learn More, Forget Less: A Gradient-Aware Data Selection Approach for LLM](https://arxiv.org/abs/2511.08620) - (arXiv) 2025
- [Data Selection for Language Models via Importance Resampling](https://arxiv.org/abs/2302.03169) - Xie et al., 2023
- [Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs](./references/Beyond-Data-Filtering-Knowledge-Localization-for-Capability-Removal-in-LLMs/meta/meta_info.txt) - Shilov et al., 2025
- [SafeGrad: Gradient Surgery for Safe LLM Fine-Tuning](https://arxiv.org/abs/2508.07172) - Yi et al., 2025
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) - Lee et al., 2022
- [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/abs/2303.09540) - Abbas et al., 2023
- [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/abs/1911.00359) - Wenzek et al., 2020
- [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794) - Li et al., 2024
