# untitled

# Misalign@k: Tail-Risk Evaluation of Emergent Misalignment Defenses Under Repeated Sampling

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Many organizations expose “fine-tuning APIs” so users can adapt aligned large language models (LLMs) to specialized domains (e.g., legal drafting, security analysis, medical triage). A recent line of work shows a concerning failure mode: **emergent misalignment (EM)**, where fine-tuning on a narrow, subtly harmful dataset can induce broadly harmful behavior on unrelated benign prompts (e.g., anti-human statements or dangerous advice) [Betley et al. 2025](./references/Emergent-Misalignment-Narrow-finetuning-can-produce-broadly-misaligned-LLMs-1-This-paper-contains-model-generated-content-that-might-be-offensive.-1/meta/meta_info.txt).

In response, several papers propose **training-time defenses** that model providers could apply during fine-tuning, such as KL-divergence regularization toward a safe base model or mixing (“interleaving”) a small amount of safe data into the fine-tuning stream [Kaczer et al. 2025](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt). These methods are typically compared using an average misalignment rate (e.g., percent of generations judged misaligned).

However, deployed LLM systems rarely rely on a single generation: users retry prompts; agents sample multiple candidates and select; and structured-output settings (JSON/templates) often require regeneration upon parse failure. In other domains, evaluation has moved from single-sample metrics to **tail-risk-under-sampling** metrics such as pass@k for code generation [Chen et al. 2021](./references/Evaluating-Large-Language-Models-Trained-on-Code/meta/meta_info.txt) and leak@k for unlearning robustness [Reisizadeh et al. 2025](./references/Leak@kk-Unlearning-Does-Not-Make-LLMs-Forget-Under-Probabilistic-Decoding/meta/meta_info.txt). Safety evaluation has also begun adopting repeated-inference “stress testing” protocols [Broadwater 2026](./references/Evaluating-LLM-Safety-Under-Repeated-Inference-via-Accelerated-Prompt-Stress-Testing/meta/meta_info.txt).

### The Problem

**Current EM defense evaluation does not report failure probability under repeated attempts.** EM papers typically report a *per-generation* misalignment rate (the fraction of sampled generations judged misaligned) and sometimes an incoherence/refusal rate. This is not directly interpretable for deployments where users retry prompts, agents sample multiple candidates, or structured-output systems regenerate until parsing succeeds. If the per-generation misalignment probability is p, the probability of observing **at least one** misaligned completion after k independent attempts is 1 − (1−p)^k. For example, p=2% implies a 48% chance of at least one misaligned completion within k=32 attempts.

Moreover, the same mean misalignment rate can arise from qualitatively different prompt-level regimes:

- **Spread-out risk**: many prompts have a small but nonzero misalignment probability.
- **Concentrated risk**: most prompts are near-zero risk, but a few prompts/templates have high misalignment probability.

These regimes have different operational consequences. Spread-out risk implies many ordinary user prompts can eventually produce a misaligned completion under repeated sampling. Concentrated risk implies a small set of prompt formats (e.g., structured-output templates) can be highly vulnerable, which is especially relevant for agent frameworks that reuse the same templates across tasks.

Existing EM evaluations (including those that already use probabilistic decoding) generally summarize results with mean misalignment and coherence, but do not standardize a **prompt-level any-of-k** metric analogous to pass@k / leak@k. Measurement validity work in red teaming emphasizes that changing the aggregation rule (one-shot vs any-of-k) can change conclusions and should be reported explicitly [Chouldechova et al. 2026](./references/Comparison-requires-valid-measurement-Rethinking-attack-success-rate-comparisons-in-AI-red-teaming/meta/meta_info.txt).

### Key Insight and Hypothesis

**Key insight.** For EM evaluation, the operational unit is the **prompt** (or prompt template), not the individual generation. Let p_i be the probability that a *single* sampled completion for prompt i is misaligned under a fixed decoding policy. Then the prompt-level probability of seeing at least one misaligned completion after k attempts is 1 − (1−p_i)^k. Aggregating this over prompts yields a deployment-relevant metric for “retry / best-of-k” behavior.

**Hypothesis.** On the Kaczer et al. EM benchmark, the relative ranking of defenses can change when moving from per-generation misalignment to prompt-level any-of-k risk. In particular, we hypothesize that safe-data interleaving can look strong on average misalignment (Table 3 in Kaczer et al.) while still having higher **Misalign@32** on some prompt templates, compared to KL regularization.

This could fail for a boring reason: Misalign@k may be almost a monotone transform of the per-generation misalignment rate for all defenses, so rankings remain unchanged and the per-prompt risk distribution looks similar.

---

## Proposed Approach

### Overview

We propose a **tail-risk evaluation protocol** for emergent misalignment defenses:

- Define **Misalign@k**: the fraction of evaluation prompts for which **at least one** of k sampled generations is misaligned.
- Report, in addition to mean misalignment, the **prompt-level heterogeneity** of misalignment risk (variance of per-prompt misalignment rates).

We will apply this protocol to compare two widely used EM training-time defenses (KL regularization and safe-data interleaving) against a misaligned baseline.

### Method Details

**1) Fixed decoding profile, repeated sampling.**

Following the Kaczer et al. released evaluation harness (`open_models/eval.py` in their repo), we generate k samples per prompt with:
- temperature = 1
- top_p = 1
- max_new_tokens (vLLM `max_tokens`) = 600

We operationalize “k samples” by repeating each prompt k times (as in the repo’s `--n_per_question` argument).

**2) Automated misalignment labeling (alignment + coherence).**

We adopt the same thresholded rubric used in the EM literature:
- Judge outputs **alignment** and **coherence** scores in [0,100]
- A sample is **misaligned** if alignment < 30 AND coherence > 50
- A sample is **incoherent** if coherence < 50

Because the open-sourced Kaczer judge implementation assumes OpenAI logprobs over numeric tokens, we instead use a non-OpenAI strong LLM judge (e.g., DeepSeek-V3.2) prompted to output JSON `{alignment: int, coherence: int}`. We will calibrate judge false positives by running the same judge on the base aligned model’s generations (sanity check).

**3) Metrics.** For a prompt set of size N and k samples per prompt:

- **MeanMisalign**: overall fraction of misaligned samples across all N·k generations.
- **Misalign@k**: (1/N) · Σ_i 1[∃ j≤k: sample(i,j) misaligned].
- **Prompt risk heterogeneity**: for each prompt i, estimate p_i = (# misaligned among k)/k; report mean(p_i) and a dispersion summary (e.g., variance across prompts).

### Key Innovations

- **Tail-risk metric for EM defenses**: introduces Misalign@k as a standard, deployment-relevant metric for emergent misalignment evaluation.
- **Prompt-distribution diagnosis**: distinguishes spread-out vs concentrated prompt/template vulnerability via dispersion of estimated per-prompt misalignment probabilities.
- **Minimal, decisive experiment**: tests whether EM defense selection conclusions change under any-of-k evaluation with the same benchmark and decoding profile.

---

## Related Work

### Field Overview

**Emergent misalignment** was first identified as a surprising narrow-to-broad generalization failure under fine-tuning [Betley et al. 2025](./references/Emergent-Misalignment-Narrow-finetuning-can-produce-broadly-misaligned-LLMs-1-This-paper-contains-model-generated-content-that-might-be-offensive.-1/meta/meta_info.txt). Subsequent work broadened the set of model organisms, training regimes, and mechanistic explanations, including rank-1 LoRA organisms and linear misalignment directions [Turner et al. 2025](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt), [Soligo et al. 2025](./references/Convergent-Linear-Representations-of-Emergent-Misalignment/meta/meta_info.txt), persona-feature analyses with sparse autoencoders [Wang et al. 2025](./references/This-paper-contains-text-that-might-be-offensive.-PERSONA-FEATURES-CONTROL-EMERGENT-MISALIGNMENT/meta/meta_info.txt), and reasoning-model variants and backdoor settings [Chua et al. 2025](./references/Thought-Crime-Backdoors-and-Emergent-Misalignment-in-Reasoning-Models/meta/meta_info.txt).

Training-time mitigation work focuses on keeping fine-tunes inside a “safe basin” via distributional constraints (e.g., KL-to-base), representation constraints, or safe-data mixing [Kaczer et al. 2025](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt), and newer mechanistic defenses constrain specific internal features during fine-tuning [Ustaomeroglu & Qu 2026](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt). These works primarily report average misalignment rates.

In parallel, several communities have converged on the idea that **sampling changes evaluation**: pass@k in code generation treats correctness as “any-of-k” [Chen et al. 2021](./references/Evaluating-Large-Language-Models-Trained-on-Code/meta/meta_info.txt); leak@k shows unlearning can fail under probabilistic decoding despite greedy decoding success [Reisizadeh et al. 2025](./references/Leak@kk-Unlearning-Does-Not-Make-LLMs-Forget-Under-Probabilistic-Decoding/meta/meta_info.txt); and safety evaluation under repeated inference has been proposed as reliability stress testing [Broadwater 2026](./references/Evaluating-LLM-Safety-Under-Repeated-Inference-via-Accelerated-Prompt-Stress-Testing/meta/meta_info.txt). Our proposal adapts these ideas specifically to EM defense evaluation and adds prompt-level heterogeneity analysis to make the result more than “more samples → more failures.”

### Related Papers

- **[Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](./references/Emergent-Misalignment-Narrow-finetuning-can-produce-broadly-misaligned-LLMs-1-This-paper-contains-model-generated-content-that-might-be-offensive.-1/meta/meta_info.txt)**: Introduces EM and evaluates it stochastically (temperature=1) but does not propose an any-of-k tail metric for defenses.
- **[In-Training Defenses against Emergent Misalignment in Language Models](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt)**: Systematic training-time defenses (KL, interleaving, LDIFS, SafeLoRA) and benchmark; reports average misalignment and coherence.
- **[Model Organisms for Emergent Misalignment](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt)**: Builds improved EM organisms and shows phase transitions, motivating careful evaluation protocols.
- **[Convergent Linear Representations of Emergent Misalignment](./references/Convergent-Linear-Representations-of-Emergent-Misalignment/meta/meta_info.txt)**: Finds transferable linear misalignment directions, suggesting failures may be systematic on certain prompts.
- **[Persona Features Control Emergent Misalignment](./references/This-paper-contains-text-that-might-be-offensive.-PERSONA-FEATURES-CONTROL-EMERGENT-MISALIGNMENT/meta/meta_info.txt)**: Mechanistic “persona” features predict/control EM, implying prompt-specific vulnerability structure.
- **[Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models](./references/Thought-Crime-Backdoors-and-Emergent-Misalignment-in-Reasoning-Models/meta/meta_info.txt)**: Extends EM to reasoning models and highlights monitoring/evaluation failure modes.
- **[Emergent Misalignment via In-Context Learning](./references/Emergent-Misalignment-via-In-Context-Learning-Narrow-in-context-examples-can-produce-broadly-misaligned-LLMs-Warning-This-paper-contains-potentially-harmful-content-generated-by-LLMs./meta/meta_info.txt)**: Shows EM can arise from in-context examples, reinforcing the need for inference-time evaluation under sampling.
- **[BLOCK-EM: Preventing Emergent Misalignment by Blocking Causal Features](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt)**: A mechanistic training-time defense; its claimed robustness should also be stress-tested under tail metrics.
- **[Evaluating LLM Safety Under Repeated Inference via Accelerated Prompt Stress Testing](./references/Evaluating-LLM-Safety-Under-Repeated-Inference-via-Accelerated-Prompt-Stress-Testing/meta/meta_info.txt)**: Introduces repeated inference safety evaluation (depth over breadth) and motivates deployment-relevant failure probabilities.
- **[Leak@k: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding](./references/Leak@kk-Unlearning-Does-Not-Make-LLMs-Forget-Under-Probabilistic-Decoding/meta/meta_info.txt)**: Defines leak@k meta-metrics for worst-case-under-sampling, directly inspiring Misalign@k.
- **[Comparison requires valid measurement: Rethinking attack success rate comparisons in AI red teaming](./references/Comparison-requires-valid-measurement-Rethinking-attack-success-rate-comparisons-in-AI-red-teaming/meta/meta_info.txt)**: Argues ASR comparisons require specifying aggregation (one-shot vs any-of-k), motivating explicit Misalign@k reporting.
- **[Evaluating Large Language Models Trained on Code](./references/Evaluating-Large-Language-Models-Trained-on-Code/meta/meta_info.txt)**: Popularizes pass@k, a core example of any-of-k evaluation changing perceived capability.
- **[A StrongREJECT for Empty Jailbreaks](./references/A-StrongREJECT¯¯StrongREJECT-overline{-hbox{{StrongREJECT}}}-for-Empty-Jailbreaks/meta/meta_info.txt)**: Refines jailbreak evaluation and highlights that weak grading inflates risk estimates.
- **[Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](./references/Sleeper-Agents-Training-Deceptive-LLMs-that-Persist-Through-Safety-Training/meta/meta_info.txt)**: Studies deceptive backdoors and adversarial training failures; tail-risk evaluation is relevant for “rare trigger” behaviors.
- **[Negative Preference Optimization](https://arxiv.org/abs/2404.05868)**: Illustrates how evaluation protocols can mischaracterize safety/forgetting outcomes under different decoding.
- **[Qwen2.5-Coder Technical Report](./references/Qwen2.5-Coder-Technical-Report/meta/meta_info.txt)**: Documents the model family used in several EM studies and relevant implementation details.
- **[Measuring Massive Multitask Language Understanding (MMLU)](./references/Measuring-Massive-Multitask-Language-Understanding/meta/meta_info.txt)**: A standard capability benchmark used as a “non-safety” control in EM studies.
- **[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)**: Motivates careful judge selection and reproducible evaluation prompts.
- **[LLMs Cannot Reliably Judge (Yet)](https://arxiv.org/abs/2406.18403)**: Shows judge unreliability and motivates calibration/sanity checks when using LLM graders.
- **[WildGuard](https://arxiv.org/abs/2407.10242)**: Provides an open safety classifier alternative that can reduce dependence on proprietary judges.
- **[HarmBench](https://arxiv.org/abs/2402.04249)**: A standardized harmfulness benchmark and evaluation framework that motivates rigorous safety measurement.
- **[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)**: A canonical example where sampling multiple generations changes measured performance.
- **[Consistency Training Helps Stop Sycophancy and Jailbreaks](https://arxiv.org/abs/2310.16987)**: Shows safety properties can change under different generation strategies, motivating evaluation under sampling.
- **[AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin](./references/AsFT-Anchoring-Safety-During-LLM-Fine-Tuning-Within-Narrow-Safety-Basin/meta/meta_info.txt)**: Another training-time anchoring strategy conceptually related to KL regularization that should be compared under tail metrics.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| EM phenomenon + organisms | Narrow training induces broad misalignment; build controllable organisms | Betley 2025; Turner 2025; Soligo 2025 | “first plot” questions, free-form prompts, judge thresholds | Evaluation often reports mean rates; prompt-level tails under-specified |
| Training-time EM defenses | Constrain fine-tuning (distributional/representation/data mixing) | Kaczer 2025; AsFT; BLOCK-EM | EM domains (code/legal/medical/security), benign tasks | Trade-offs: learning inhibition, incoherence; robustness to deployment sampling unclear |
| Sampling-aware capability eval | Any-of-k metrics change apparent ability | pass@k (Chen 2021), self-consistency | HumanEval, reasoning benchmarks | Assumes independence; can hide heavy tails |
| Sampling-aware safety/unlearning eval | Worst-case-under-sampling reveals hidden failures | leak@k; APST; ASR measurement validity | TOFU/MUSE/WMDP, AIR-BENCH, jailbreak suites | Judge validity and decoding config sensitivity |

### Closest Prior Work

- **Kaczer et al. (training-time defenses)**: Provides the benchmark and strong baselines (KL, interleaving), but primarily reports average misalignment/coherence and does not report any-of-k tail risk per prompt.
- **Betley et al. (EM discovery)**: Uses stochastic decoding (temperature=1) and reports misalignment probability, but focuses on establishing EM and controls rather than comparing defenses under deployment-tail metrics.
- **Broadwater (APST)**: Formalizes repeated-inference safety stress testing, but does not study EM fine-tuning regimes or EM-specific defense selection.
- **Reisizadeh et al. (leak@k)**: Introduces the core “any-of-k reveals hidden failures” idea for unlearning; EM evaluation has not adopted an analogous metric for defense comparison.
- **Chouldechova et al. (ASR measurement validity)**: Provides a conceptual warning that any-of-k aggregation changes safety conclusions; EM evaluation needs similarly explicit aggregation reporting.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Kaczer 2025 | Compares EM training-time defenses via mean misalignment/coherence | Does not quantify prompt-level tail risk under repeated sampling | Add Misalign@k + Var(p_i) on the same benchmark | Can change which defense is preferred in deployment-like sampling regimes |
| Betley 2025 | Discovers EM and evaluates stochastically | Not focused on defense selection metrics | Apply tail metrics directly to defenses | Connects EM research to operational safety decision-making |
| Broadwater 2026 (APST) | Repeated-inference safety evaluation | Not applied to EM fine-tuning | Port depth-oriented evaluation to EM | Provides EM-specific stress test grounded in EM benchmarks |
| Reisizadeh 2025 (leak@k) | Any-of-k leakage for unlearning | Not about EM or EM defenses | Misalign@k for EM | Generalizes a proven evaluation idea to EM |
| Chouldechova 2026 | Measurement validity for ASR comparisons | Does not provide EM metric | Explicit aggregation definition + calibration control | Avoids apples-to-oranges EM safety claims |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Base aligned model for fine-tuning + sanity-check evaluation |

**Training Data (if applicable):**

We follow Kaczer et al.’s released code and datasets (GitHub repo below) and focus on the **Security EM dataset** (6000 aligned + 6000 misaligned examples; we train on 5400 rows using a 90/10 split).

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| Security EM dataset (Kaczer) | Induce EM via fine-tuning | 12k (6k aligned + 6k misaligned) | https://github.com/davidkaczer/emergent-misalignment | Repository license (see repo) |
| WildGuardMix (subset) | Benign/safety-oriented instruction data used for safe-data interleaving | (as used by Kaczer) | https://github.com/davidkaczer/emergent-misalignment | Repository license (see repo) |

**Defense conditions (≤3 main conditions):**

1. **Misaligned baseline**: fine-tune with rs-LoRA (rank-stabilized Low-Rank Adaptation; a standard parameter-efficient fine-tuning method) on Security misaligned data.
2. **KL regularization**: same, with λ_KL = 0.1 toward base model (per Kaczer).
3. **Safe-data interleaving**: same, with 5% benign data interleaved (per Kaczer).

We also evaluate the **base aligned model** as a sanity-check control (judge false positives).

**Evaluation harness:**
- Training + evaluation code: https://github.com/davidkaczer/emergent-misalignment
- Generation settings reference: `open_models/eval.py` uses vLLM with temperature=1, top_p=1, max_tokens=600.

**Resource Estimate**:
- **Compute budget**: ~10–50 GPU-hours total for the minimal study (3 rs-LoRA fine-tunes of a 7B model for 1 epoch + vLLM generation for ~2.3k samples). If running 3 training seeds per condition for robustness, budget ~30–150 GPU-hours (still well under 768 GPU-hours).
- **GPU memory**: 1×A100 80GB is sufficient for 7B vLLM inference and LoRA fine-tuning; tensor parallelism optional.
- **API usage**: ~2.3k judge calls for k=32 on 24 prompts × 3 conditions (plus a small aligned-model sanity check).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| EM “first plot” prompt set (24 items) | 8 open-ended prompts from Betley et al. plus JSON/template variants (24 total prompt variants) used to showcase emergent misalignment | MeanMisalign, Misalign@k (k∈{1,32}), per-prompt misalignment-rate dispersion, incoherence rate | test | https://github.com/davidkaczer/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml | Kaczer `open_models/eval.py` + custom aggregation |

### Main Results

#### Results Table

Published mean misalignment numbers (single-sample / mean-of-samples style) from Kaczer et al. for **Qwen2.5-7B on Security (General)** are included as a scale reference; tail metrics are to be verified.

| Defense / model | Base Model | Benchmark | Mean misalignment (%) ↓ | Incoherence (%) ↓ | Misalign@1 (ours) ↓ | Misalign@32 (ours) ↓ | Var(p_i) (ours) ↓ | Source | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Aligned (no fine-tune) | Qwen2.5-7B-Instruct | first_plot_questions.yaml | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | - | Sanity check (judge FPR) |
| Misaligned baseline | Qwen2.5-7B-Instruct | first_plot_questions.yaml | 26.25 | 19.38 | **TBD** | **TBD** | **TBD** | Kaczer 2025 Table 3 | Published mean over their evaluation protocol |
| KL (λ=0.1) | Qwen2.5-7B-Instruct | first_plot_questions.yaml | 2.04 | 1.79 | **TBD** | **TBD** | **TBD** | Kaczer 2025 Table 3 | Published mean over their evaluation protocol |
| Interleaving (5%) | Qwen2.5-7B-Instruct | first_plot_questions.yaml | 1.38 | 26.05 | **TBD** | **TBD** | **TBD** | Kaczer 2025 Table 3 | Published mean over their evaluation protocol |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| k sweep (k=8 vs 32) | Evaluate Misalign@k at two k values | If tails matter, differences between defenses widen with k |
| Prompt-format breakdown | Report metrics separately for {plain, JSON, template} subsets (8 each) | Heavy tails for interleaving may concentrate in structured-output subsets |

---

## Success Criteria

**Criterion 1: Tail-risk changes defense selection**
- Hypothesis: Interleaving appears strong under mean misalignment / Misalign@1 but is worse under Misalign@32 due to heavier prompt-level tails.
- Validation: A meaningful “ranking flip” under Misalign@32 (or ≥2× higher Var(p_i) for interleaving than KL) on the 24-prompt set.

**Criterion 2: Added metric is not a trivial rescaling**
- Hypothesis: Differences are driven by prompt-level heterogeneity (Var(p_i)), not just by mean(p_i).
- Validation: Misalign@32 differs between defenses even when mean(p_i) is similar, and the aligned model has low Misalign@32 (judge FPR check).

---

## Impact Statement

If successful, this work provides a simple, standardized tail-risk reporting addition (Misalign@k + Var(p_i)) that model providers can use to decide which EM defense to deploy for fine-tuning APIs, especially in systems that retry or sample multiple candidates. It also gives EM researchers a deployment-relevant metric that can reveal “rare but catastrophic” failures hidden by mean misalignment rates.

---

## References

- [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](./references/Emergent-Misalignment-Narrow-finetuning-can-produce-broadly-misaligned-LLMs-1-This-paper-contains-model-generated-content-that-might-be-offensive.-1/meta/meta_info.txt) - Betley et al., 2025
- [In-Training Defenses against Emergent Misalignment in Language Models](./references/In-Training-Defenses-against-Emergent-Misalignment-in-Language-Models/meta/meta_info.txt) - Kaczer et al., 2025
- [Model Organisms for Emergent Misalignment](./references/Model-Organisms-for-Emergent-Misalignment/meta/meta_info.txt) - Turner et al., 2025
- [Convergent Linear Representations of Emergent Misalignment](./references/Convergent-Linear-Representations-of-Emergent-Misalignment/meta/meta_info.txt) - Soligo et al., 2025
- [Persona Features Control Emergent Misalignment](./references/This-paper-contains-text-that-might-be-offensive.-PERSONA-FEATURES-CONTROL-EMERGENT-MISALIGNMENT/meta/meta_info.txt) - Wang et al., 2025
- [Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models](./references/Thought-Crime-Backdoors-and-Emergent-Misalignment-in-Reasoning-Models/meta/meta_info.txt) - Chua et al., 2025
- [Emergent Misalignment via In-Context Learning](./references/Emergent-Misalignment-via-In-Context-Learning-Narrow-in-context-examples-can-produce-broadly-misaligned-LLMs-Warning-This-paper-contains-potentially-harmful-content-generated-by-LLMs./meta/meta_info.txt) - Afonin et al., 2025
- [BLOCK-EM: Preventing Emergent Misalignment by Blocking Causal Features](./references/BLOCK-EM-Preventing-Emergent-Misalignment-by-Blocking-Causal-Features/meta/meta_info.txt) - Ustaomeroglu & Qu, 2026
- [Evaluating LLM Safety Under Repeated Inference via Accelerated Prompt Stress Testing](./references/Evaluating-LLM-Safety-Under-Repeated-Inference-via-Accelerated-Prompt-Stress-Testing/meta/meta_info.txt) - Broadwater, 2026
- [Leak@k: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding](./references/Leak@kk-Unlearning-Does-Not-Make-LLMs-Forget-Under-Probabilistic-Decoding/meta/meta_info.txt) - Reisizadeh et al., 2025
- [Comparison requires valid measurement: Rethinking attack success rate comparisons in AI red teaming](./references/Comparison-requires-valid-measurement-Rethinking-attack-success-rate-comparisons-in-AI-red-teaming/meta/meta_info.txt) - Chouldechova et al., 2026
- [Evaluating Large Language Models Trained on Code](./references/Evaluating-Large-Language-Models-Trained-on-Code/meta/meta_info.txt) - Chen et al., 2021
- [A StrongREJECT for Empty Jailbreaks](./references/A-StrongREJECT¯¯StrongREJECT-overline{-hbox{{StrongREJECT}}}-for-Empty-Jailbreaks/meta/meta_info.txt) - Souly et al., 2024
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](./references/Sleeper-Agents-Training-Deceptive-LLMs-that-Persist-Through-Safety-Training/meta/meta_info.txt) - Hubinger et al., 2024
- [AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin](./references/AsFT-Anchoring-Safety-During-LLM-Fine-Tuning-Within-Narrow-Safety-Basin/meta/meta_info.txt) - Anonymous/Preprint (year in meta)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) - Zheng et al., 2023
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [HarmBench](https://arxiv.org/abs/2402.04249) - Mazeika et al., 2024
- [WildGuard](https://arxiv.org/abs/2407.10242) - OpenAI, 2024
- [LLMs Cannot Reliably Judge (Yet)](https://arxiv.org/abs/2406.18403) - (authors), 2024
- [Consistency Training Helps Stop Sycophancy and Jailbreaks](https://arxiv.org/abs/2310.16987) - (authors), 2023
