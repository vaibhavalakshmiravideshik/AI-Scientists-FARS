# untitled

# Syntax-Diversified Unlearning to Reduce Worst-Case Leakage Under Sampling

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) can memorize and reproduce details of their training data. In real deployments, model owners may need to remove the influence of specific data after training (e.g., privacy deletion requests, copyrighted content, or safety-critical knowledge). **Machine unlearning** studies post-training updates that make a trained model behave as if it had not been trained on a designated *forget set*, while preserving performance on a *retain set*.

Two recent lines of work suggest that many unlearning claims are fragile under realistic deployment conditions. First, **probabilistic decoding** (temperature / top-p sampling) can reveal forgotten content even when greedy decoding looks safe, motivating worst-case metrics such as **leak@k** that measure whether any of k samples leaks information ([Leak@k](./references/LEAK@k-UNLEARNING-DOES-NOT-MAKE-LLMS-FORGET-UNDER-PROBABILISTIC-DECODING/meta/meta_info.txt)). Second, **benign relearning** shows that after unlearning, a small amount of seemingly unrelated fine-tuning can restore forgotten knowledge, often driven more by **syntactic similarity** of the fine-tuning data than topical overlap ([Rethinking Benign Relearning](./references/RETHINKING-BENIGN-RELEARNING-SYNTAX-AS-THE-HIDDEN-DRIVER-OF-UNLEARNING-FAILURES/meta/meta_info.txt)).

If unlearning is to be used for compliance, it must be robust to both (i) user-side sampling (multiple generations) and (ii) ordinary downstream fine-tuning that a model might undergo after deployment.

### The Problem

**Failure mode: template-dominant suppression.** Many unlearning pipelines are trained and evaluated on narrow query templates. On the TOFU benchmark (synthetic author biographies; [TOFU](./references/TOFU-A-Task-of-Fictitious-Unlearning-for-LLMs/meta/meta_info.txt)), the “full name” questions share a rigid surface form (e.g., “What is the full name of the author born in … on …?”) and answers share a repeated template. Benign Relearning provides evidence that unlearning updates can disproportionately suppress these templates while leaving the *keyword tokens* (the author’s name) relatively accessible. This creates:

1. **Benign relearning risk**: fine-tuning on syntactically similar data (even about different entities) restores the suppressed template pathway and reactivates forgotten names.
2. **Sampling risk**: even if greedy decoding avoids emitting the name, residual probability mass on the name can surface in at least one of k samples.

Existing work tackles these risks separately: Benign Relearning proposes **syntactic diversification** to mitigate relearning, while Leak@k proposes **probabilistic evaluation** to expose sampling leakage. It is unknown whether syntactic diversification also reduces deployment-style probabilistic leakage, and whether it reduces a template-dominant failure mode in a way that is visible under worst-case sampling metrics.

### Key Insight and Hypothesis

**Key insight.** If unlearning is failing because it primarily suppresses a narrow query template, then *diversifying the syntax of forget-set queries during unlearning* should force the update to target the keyword tokens themselves (names), reducing both (i) relearning vulnerability and (ii) worst-case leakage under sampling.

**Hypothesis.** On TOFU, replacing each “full name” forget-query with a syntactically diverse paraphrase (same semantics, same answer) during NPO unlearning will:
- reduce **leak@32** (probability of leaking the author name in any of 32 sampled generations) under high top-p sampling, and
- reduce **benign relearning success** after fine-tuning on a syntactically similar relearn set,
without a large drop in standard TOFU utility on the retain / real-authors / world-facts sets.

This could fail for a boring reason: diversification may help relearning robustness but not reduce the model’s tail probability on the name tokens under sampling; or it may reduce leakage only by inducing generic refusals that also harm retain utility. The experiment below includes utility checks and a template-injection diagnostic to separate these explanations.

---

## Proposed Approach

### Overview

We propose a data-side intervention for robust unlearning evaluation: **syntax-diversified unlearning**.

Given a base unlearning method (we focus on Negative Preference Optimization, NPO), we create a diversified forget set by paraphrasing only the rigid “full name” queries into heterogeneous surface forms while preserving the same answer string. We then run the same unlearning procedure with the diversified queries and compare against the baseline unlearning run.

### Method Details

**Base unlearning method (NPO).** We use Negative Preference Optimization (NPO; [Zhang et al.](https://arxiv.org/abs/2404.05868)), a preference-style unlearning method that treats each (prompt, forget-answer) pair as a *rejected* completion and updates the model to reduce its probability (often relative to a fixed reference model), while preserving utility via retain-set training/regularization. We choose NPO because it is widely used and implemented in OpenUnlearning ([OpenUnlearning](https://arxiv.org/abs/2506.12618)).

**Constructing TOFU target + relearn sets.** Following Benign Relearning, within the forget split we identify:
- **D_target**: questions that ask for the author’s *full name* (the sensitive keyword is the full name string).
- **D_syntactic_relearn**: “full name” questions about *different* authors drawn from the retain split (syntactically similar but not containing target names).

**Syntactic diversification (automated, no manual filtering).** For each target query q in D_target, generate M paraphrase candidates with an instruction-tuned LLM (paraphraser). Select one paraphrase q′ satisfying hard constraints:
- Preserve all extracted slot substrings exactly (e.g., birthplace string and date string).
- Do not introduce any new named entities.
- Ask for the full name (must contain “full name” or “name”).
- Syntactic similarity Sim(q, q′) ≤ τ, where Sim is normalized Levenshtein similarity (as in Benign Relearning).

We set **M = 8** and **τ = 0.35** initially. If no candidate passes, fall back to a deterministic template rewrite from a fixed bank of 10 templates. We will report the fallback rate; if fallback rate exceeds 30%, the diversification procedure is deemed unreliable and the experiment is reported as inconclusive.

**Training with diversification.** Replace each q in D_target with its selected paraphrase q′, keep the answer y unchanged, and keep all other forget examples unchanged. Run NPO unlearning with identical hyperparameters to the baseline.

### Key Innovations

- **Connects two deployment failures**: tests whether a mitigation for *benign relearning* (syntactic diversification) also reduces *probabilistic decoding leakage* as measured by leak@k-style worst-case sampling.
- **Fully automated diversification pipeline**: replaces manual paraphrase filtering with an explicit constraint + filter procedure and a deterministic fallback.
- **Mechanism-aware diagnostic**: uses a template-injection probe to test whether diversification reduces the “template-dominant suppression” gap.

---

## Related Work

### Field Overview

LLM unlearning spans (i) unlearning algorithms (gradient ascent variants, preference-based objectives, representation-level interventions, and output-time correction) and (ii) evaluation, where recent work argues that many standard metrics can be misleading. Benchmarks such as TOFU and MUSE provide controlled settings with forget/retain splits, while OpenUnlearning provides a unified implementation layer across benchmarks and metrics.

Robustness failures are increasingly central: quantization can recover forgotten behavior, post-unlearning fine-tuning can relearn forgotten knowledge, and probabilistic decoding can reveal tail-risk leakage that greedy decoding hides. These stress tests suggest that unlearning should be evaluated as a worst-case reliability problem rather than a single-output accuracy problem.

### Related Papers

- **[TOFU](./references/TOFU-A-Task-of-Fictitious-Unlearning-for-LLMs/meta/meta_info.txt)**: Fictitious author benchmark with forget/retain splits and utility/forget-quality metrics.
- **[Leak@k](./references/LEAK@k-UNLEARNING-DOES-NOT-MAKE-LLMS-FORGET-UNDER-PROBABILISTIC-DECODING/meta/meta_info.txt)**: Shows greedy decoding can hide leakage; proposes leak@k for probabilistic decoding evaluation.
- **[Rethinking Benign Relearning](./references/RETHINKING-BENIGN-RELEARNING-SYNTAX-AS-THE-HIDDEN-DRIVER-OF-UNLEARNING-FAILURES/meta/meta_info.txt)**: Shows syntactic similarity drives relearning; proposes syntactic diversification.
- **[Learning-Time Encoding Shapes Unlearning](./references/Learning-Time-Encoding-Shapes-Unlearning-in-LLMs/meta/meta_info.txt)**: Shows learning-time paraphrase encoding affects later unlearning; focuses on training-time choices, not sampling leakage.
- **[OpenUnlearning](https://arxiv.org/abs/2506.12618)**: Unified framework implementing multiple unlearning algorithms/metrics and releasing checkpoints.
- **[NPO](https://arxiv.org/abs/2404.05868)**: Preference-style objective for unlearning by penalizing forget outputs.
- **[SimNPO](https://arxiv.org/abs/2410.07163)**: Simplifies NPO; strong baseline in OpenUnlearning.
- **[MUSE](https://arxiv.org/abs/2402.10484)**: Six-way unlearning evaluation on news/books with privacy and utility metrics.
- **[WMDP](https://arxiv.org/abs/2403.03218)**: Hazardous knowledge benchmark often used to test unlearning/alignment.
- **[LLM Unlearning with LLM Beliefs](https://arxiv.org/abs/2510.19422)**: Identifies “squeezing effect” (probability mass shifts to paraphrases); proposes belief-bootstrapping objectives.
- **[Textual Unlearning Gives a False Sense of Unlearning](https://arxiv.org/abs/2406.13348)**: Shows unlearning can enable membership inference / reconstruction attacks when comparing before/after models.
- **[Unlearning That Lasts (JensUn)](https://arxiv.org/abs/2509.02820)**: Evaluates unlearning under worst-case paraphrases and other stress tests; proposes a robust unlearning method.
- **[ReLearn](https://arxiv.org/abs/2502.11190)**: “Unlearning via learning” with data augmentation; focuses on rewriting knowledge.
- **[Attention Smoothing Is All You Need for Unlearning](https://openreview.net/forum?id=sX9HbELwLO)**: Uses attention-temperature smoothing to reduce factual recall while keeping fluency.
- **[Hallucination-Free LLMs Unlearning via Attention Shifting](https://arxiv.org/abs/2510.17210)**: Uses attention-based suppression to reduce hallucination and unlearning artifacts.
- **[Unlearning with Control](https://arxiv.org/abs/2406.09179)**: Studies excessive unlearning and proposes control procedures for fair evaluation.
- **[CURE](https://arxiv.org/abs/2509.25973)**: Retrieval-augmented post-generation correction to prevent leakage, especially for indirect queries.
- **[Does your LLM truly unlearn?](https://arxiv.org/abs/2410.16454)**: Shows quantization can recover unlearned knowledge; proposes saliency-based mitigation.
- **[QUAIL](https://arxiv.org/abs/2602.05522)**: Quantization-aware unlearning objectives to preserve forgetting after PTQ.
- **[REBEL](https://arxiv.org/abs/2602.06248)**: Adversarial prompt search to recover “forgotten” knowledge; highlights evaluation gaps.
- **[Who’s Harry Potter?](https://arxiv.org/abs/2310.02238)**: Early work on entity-level unlearning and its limitations.
- **[Towards Unbounded Machine Unlearning](https://openreview.net/forum?id=hfA5tYpW8M)**: Studies unlearning under repeated deletion requests.
- **[Large Language Model Unlearning](https://openreview.net/forum?id=wKe6jE065x)**: Gradient-ascent style foundations for LLM unlearning.
- **[A Closer Look at Machine Unlearning for LLMs](https://openreview.net/forum?id=U8uO8LMo70)**: Analysis of unlearning metrics and failure modes.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Gradient-ascent / loss reversal | Increase loss on forget data, regularize retain | GA, GradDiff, [TOFU](./references/TOFU-A-Task-of-Fictitious-Unlearning-for-LLMs/meta/meta_info.txt) | TOFU, MUSE | Can cause utility collapse; can be brittle to stress tests |
| Preference-based unlearning | Prefer refusals/negative outputs on forget prompts | [NPO](https://arxiv.org/abs/2404.05868), [SimNPO](https://arxiv.org/abs/2410.07163) | OpenUnlearning, TOFU, MUSE | Can yield spurious forgetting (rephrasings) and brittle decoding behavior |
| Representation-level methods | Push hidden states away from forget representations | RMU, OBLIVIATE | MUSE, WMDP | May not remove tail-risk leakage; stress-test sensitivity |
| Output-time correction | Detect and rewrite leaked content post hoc | [CURE](https://arxiv.org/abs/2509.25973) | TOFU indirect queries, WMDP | Extra inference cost; depends on retrieval/judge |
| Robust evaluation / attacks | Probe residual knowledge beyond greedy QA | [Leak@k](./references/LEAK@k-UNLEARNING-DOES-NOT-MAKE-LLMS-FORGET-UNDER-PROBABILISTIC-DECODING/meta/meta_info.txt), [JensUn](https://arxiv.org/abs/2509.02820), [REBEL](https://arxiv.org/abs/2602.06248) | Sampling, paraphrases, adversarial prompts | Increased eval cost; metrics still imperfect |
| Data/encoding interventions | Change how knowledge is represented in training/unlearning data | [Benign Relearning](./references/RETHINKING-BENIGN-RELEARNING-SYNTAX-AS-THE-HIDDEN-DRIVER-OF-UNLEARNING-FAILURES/meta/meta_info.txt), [Learning-Time Encoding](./references/Learning-Time-Encoding-Shapes-Unlearning-in-LLMs/meta/meta_info.txt) | TOFU / TOFU+ | Often not evaluated under probabilistic decoding leakage |

### Closest Prior Work

**Rethinking Benign Relearning (ICLR 2026).** Introduces the hypothesis that syntactic similarity (not topical relevance) drives relearning, and proposes syntactic diversification that improves relearning robustness on TOFU. However, it does not evaluate worst-case probabilistic decoding leakage (leak@k) and therefore does not answer whether diversification reduces tail-risk leakage under sampling.

**Leak@k (ICLR 2026 submission).** Introduces a meta-metric for probabilistic decoding leakage and shows most unlearning methods fail under top-p/temperature sampling. It also proposes **NPO-Fix**, an extension of NPO that *augments the forget set with leakage instances discovered by sampling* (a generation-in-the-loop data augmentation). However, it does not study *syntax-level* data interventions such as paraphrase-based diversification, and it does not include a benign-relearning fine-tuning stress test.

**Learning-Time Encoding Shapes Unlearning (2025).** Shows that learning-time paraphrase exposure changes the unlearning–retain trade-off, but focuses on how the *original training data* is encoded rather than how *forget queries* should be diversified during unlearning, and does not evaluate probabilistic sampling leakage.

**LLM Unlearning with LLM Beliefs (2025).** Targets probabilistic leakage through objective-level bootstrapping of model beliefs (paraphrased high-likelihood generations). This is complementary but more expensive than a data-side query diversification; it does not test the specific template-dominant suppression mechanism highlighted in Benign Relearning.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Benign Relearning | Shows syntactic relearning + proposes diversification | No probabilistic decoding leakage eval | Evaluate diversification under leak@k-style worst-case sampling | If diversification reduces template dominance, it should also lower tail-risk leakage |
| Leak@k | Defines leak@k and evaluates many unlearning methods under sampling | No data-side mitigation tested | Apply diversification as a minimal mitigation and test on leak@k metrics | Diversification is cheap and may fix a core failure mode |
| Learning-Time Encoding | Paraphrase exposure during training improves unlearning | Not about unlearning-time query syntax; no sampling leakage | Diversify *forget queries* during unlearning + sample-based eval | Directly targets the deployment failure setting |
| LLM Beliefs | Bootstraps away from paraphrased high-likelihood beliefs | Added compute/complexity; different mechanism | Use a simpler data-side change + mechanism probe | If successful, provides a lower-cost alternative |
| JensUn | Worst-case paraphrase evaluation; robust unlearning method | Primarily method-level; may require more training changes | Keep unlearning algorithm fixed; modify only query syntax | Isolates whether data-side diversity alone reduces worst-case leakage |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| TOFU target model (full) | 1B | `open-unlearning/pos_tofu_Llama-3.2-1B-Instruct_full_lr3e-05_wd0.01_epoch10` | Base model for unlearning (has TOFU author knowledge) |
| TOFU retain-only model | 1B | `open-unlearning/neg_tofu_Llama-3.2-1B-Instruct_retain90_lr3e-05_wd0.01_epoch10` | Reference model trained without forget authors |
| Paraphraser model | 8B | https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct | Used only to generate paraphrases for diversification |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| TOFU | Unlearning benchmark (forget/retain splits) | 4,000 QA | https://huggingface.co/datasets/locuslab/TOFU | See dataset card |

**Other Resources (if applicable):**
- OpenUnlearning codebase (training + baseline configs): https://github.com/locuslab/open-unlearning

**Resource Estimate**:
- **Compute budget**: 40–120 GPU-hours (well within 768 GPU-hour cap)
  - NPO unlearning (LoRA) on a 1B model over TOFU forget/retain: ~2–6 GPU-hours per run × 2 conditions × 3 random seeds.
  - Paraphrase generation for ~20 target queries (M=8 candidates): ≤1 GPU-hour.
  - leak@32 evaluation (k=32, two decoding settings, ~20 queries): ≤2 GPU-hours.
  - Benign relearning fine-tune (LoRA, ≤47 steps) + evaluation: ≤2 GPU-hours per model.
  - Extra slack covers reruns for stability and integration overhead.
- **GPU memory**: ≤80GB per GPU (1B + LoRA fits easily; 8B paraphraser may need 1×80GB or quantized inference).
- **API usage**: None required (paraphraser is open-source).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| TOFU | Synthetic QA about 200 fictitious authors; unlearn a subset of authors | leak@32 (keyword), template-injection gap, relearn success rate, model utility | forget10 / retain90 | https://huggingface.co/datasets/locuslab/TOFU | https://github.com/locuslab/open-unlearning + small custom eval |

**Metric definitions (ours):**
- **Keyword leak (binary)**: For a target query q with gold full name a, a generation leaks if it contains a as an exact substring (case-insensitive). 
- **leak@32 (binary)**: For each q, sample k=32 generations under a decoding configuration and mark leak if any generation leaks. Report average across q ∈ D_target.
- **Template-injection gap**: For each q, create an injected prompt that provides the answer template prefix up to (but not including) the name (as in Benign Relearning Appendix F). Measure leakage under injected vs non-injected prompts.
- **Relearn Success Rate**: After relearning fine-tune on D_syntactic_relearn, fraction of D_target queries whose outputs contain the target full name.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | leak@32 ↓ (T=0.2,p=1.0) | leak@32 ↓ (T=1.0,p=1.0) | Relearn Success Rate@23 ↓ | Utility (TOFU MU) ↑ | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| Original (full) | Llama-3.2-1B-Instruct | TOFU forget10 | **TBD** | **TBD** | **TBD** | **TBD** | - | Reference model before unlearning |
| Retain-only (retrain) | Llama-3.2-1B-Instruct | TOFU retain90 | **TBD** | **TBD** | **0.0 (expected)** | **TBD** | - | Reference model w/o forget authors |
| NPO (baseline) | Llama-3.2-1B-Instruct | TOFU forget10 | **TBD** | **TBD** | **TBD** | **TBD** | - | Needs re-run (baseline) |
| **NPO + syntax-diversified D_target (ours)** | Llama-3.2-1B-Instruct | TOFU forget10 | **TBD** | **TBD** | **TBD** | **TBD** | - | Proposed |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Diversification off (baseline) | Use original D_target queries | Higher leak@32 and higher relearning success |
| Diversification on (full) | Replace each D_target query with one paraphrase (τ=0.35) | Lower leak@32; smaller template-injection gap |

### Analysis (Optional)

- **Premise check (template dominance)**: Report the template-injection leakage gap after baseline NPO. If the gap is <10% absolute, the “template-dominant suppression” premise is weak and we expect diversification to have limited effect.
- **Diversification quality**: Report average syntactic similarity between D_syntactic_relearn and D_target vs D′_target, and the fallback rate.

---

## Success Criteria

**Criterion 1: Reduced worst-case sampling leakage**
- Hypothesis: Syntax diversification reduces tail-risk leakage under top-p sampling.
- Validation: On D_target, leak@32 decreases by ≥20% relative under at least one of the two decoding settings, and does not increase under the other. We choose 20% relative as a “decision-changing” effect size: for example, if baseline leak@32 is 0.50 (leaks on half the target queries), a 20% relative drop corresponds to 0.40 (10 fewer leaking queries per 100), which is meaningful for compliance risk given best-of-N usage.

**Criterion 2: Reduced benign relearning under syntactic fine-tuning**
- Hypothesis: Diversification reduces the ability of syntactically similar fine-tuning to restore forgotten names.
- Validation: After relearning for 23 steps on D_syntactic_relearn, Relearn Success Rate decreases by ≥0.10 absolute vs baseline NPO.

**Criterion 3: No large utility collapse**
- Hypothesis: Diversification is not just inducing generic refusals or broad degradation.
- Validation: TOFU model utility (retain/real-authors/world-facts aggregate) drops by ≤3% absolute relative to baseline NPO.

---

## Impact Statement

If successful, this work provides a low-cost, deployment-relevant guideline for LLM unlearning: **diversify the syntax of deletion-request queries during unlearning** to reduce both worst-case sampling leakage and vulnerability to benign post-deployment fine-tuning. This would change how practitioners construct forget sets and how they evaluate unlearning claims in compliance-focused settings.

---

## References

- [TOFU: A Task of Fictitious Unlearning for LLMs](./references/TOFU-A-Task-of-Fictitious-Unlearning-for-LLMs/meta/meta_info.txt) - Maini et al., 2024
- [LEAK@k: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding](./references/LEAK@k-UNLEARNING-DOES-NOT-MAKE-LLMS-FORGET-UNDER-PROBABILISTIC-DECODING/meta/meta_info.txt) - Reisizadeh et al., 2025/2026
- [Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures](./references/RETHINKING-BENIGN-RELEARNING-SYNTAX-AS-THE-HIDDEN-DRIVER-OF-UNLEARNING-FAILURES/meta/meta_info.txt) - 2026
- [Learning-Time Encoding Shapes Unlearning in LLMs](./references/Learning-Time-Encoding-Shapes-Unlearning-in-LLMs/meta/meta_info.txt) - Wu et al., 2025
- [OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics](https://arxiv.org/abs/2506.12618) - Dorna et al., 2025
- [Negative Preference Optimization](https://arxiv.org/abs/2404.05868) - Zhang et al., 2024
- [Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning (SimNPO)](https://arxiv.org/abs/2410.07163) - Fan et al., 2024
- [Machine Unlearning Six-Way Evaluation for Language Models (MUSE)](https://arxiv.org/abs/2402.10484) - Shi et al., 2024
- [LLM Unlearning with LLM Beliefs](https://arxiv.org/abs/2510.19422) - Li et al., 2025
- [Textual Unlearning Gives a False Sense of Unlearning](https://arxiv.org/abs/2406.13348) - Du et al., 2024
- [Unlearning That Lasts: Utility-Preserving, Robust, and Almost Irreversible Forgetting in LLMs](https://arxiv.org/abs/2509.02820) - 2025
- [ReLearn: Unlearning via Learning for Large Language Models](https://arxiv.org/abs/2502.11190) - 2025
- [Attention Smoothing Is All You Need for Unlearning](https://openreview.net/forum?id=sX9HbELwLO) - 2025
- [Hallucination-Free LLMs Unlearning via Attention Shifting](https://arxiv.org/abs/2510.17210) - 2025
- [Unlearning with Control: Assessing Real-world Utility for LLM Unlearning](https://arxiv.org/abs/2406.09179) - Wang et al., 2024
- [Scalable and Robust LLM Unlearning by Correcting Responses with Retrieved Exclusions (CURE)](https://arxiv.org/abs/2509.25973) - 2025
- [Does your LLM truly unlearn?](https://arxiv.org/abs/2410.16454) - Zhang et al., 2024
- [REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop](https://arxiv.org/abs/2602.06248) - 2026
- [Who’s Harry Potter?](https://arxiv.org/abs/2310.02238) - Eldan & Russinovich, 2023
- [Large Language Model Unlearning](https://openreview.net/forum?id=wKe6jE065x) - Yao et al., 2023
- [Towards Unbounded Machine Unlearning](https://openreview.net/forum?id=hfA5tYpW8M) - Kurmanji et al., 2023
- [A Closer Look at Machine Unlearning for Large Language Models](https://openreview.net/forum?id=U8uO8LMo70) - Yuan et al., 2024
