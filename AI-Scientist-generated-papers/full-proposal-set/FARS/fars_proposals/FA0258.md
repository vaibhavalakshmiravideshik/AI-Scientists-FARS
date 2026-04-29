# untitled

# Suppression-Contrast Tokens: Reverse Layer-Contrast for Secret Elicitation from Refusal Transcripts

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are often trained to **refuse** to reveal certain information. In alignment and safety discussions, a central question is whether a refusal indicates that the model does not represent the information internally, or whether the information is represented but **suppressed** before it reaches the final output.

This matters for **auditing and scalable oversight**. If models can internally represent information that is not present in their final responses, then purely output-based audits can miss important behaviors. Several recent works therefore study *model organisms* (controlled training setups designed to isolate an alignment-relevant behavior with known ground truth) and *elicitation* benchmarks where the secret is known to the evaluator, enabling quantitative comparison of methods that try to recover hidden information (e.g., the Taboo / Secret Side Constraint / User Gender settings in **[Eliciting Secret Knowledge from Language Models](./references/Eliciting-Secret-Knowledge-from-Language-Models/meta/meta_info.txt)**). In the Taboo setting, for example, a model is trained to give hints about a single secret word (e.g., “gold”) while never explicitly outputting it (**[Taboo setting description](./references/Eliciting-Secret-Knowledge-from-Language-Models/sections/Taboo.md)**).

A standard white-box approach is **logit lens**: project intermediate hidden states through the model’s unembedding matrix to obtain a distribution over next tokens, then provide the top-ranked tokens to an auditor (e.g., **[Towards eliciting latent knowledge from LLMs with mechanistic interpretability](./references/Towards-eliciting-latent-knowledge-from-LLMs-with-mechanistic-interpretability/meta/meta_info.txt)**; **[Eliciting Secret Knowledge from Language Models](./references/Eliciting-Secret-Knowledge-from-Language-Models/meta/meta_info.txt)**). These methods can work, but in the hardest “direct” (refusal-style) setting, published performance remains low for some model organisms (e.g., Taboo).

### The Problem

Current white-box secret elicitation methods mostly rank candidate tokens by **mid-layer probability** (logit lens) or related activation-derived heuristics (SAEs, activation–embedding similarity) (**[Eliciting Secret Knowledge](./references/Eliciting-Secret-Knowledge-from-Language-Models/meta/meta_info.txt)**). This implicitly assumes that the secret token is simply “most probable” at some intermediate layer.

However, secret-keeping behaviors are often trained by adding explicit constraints like “never say the secret word.” A plausible failure mode is therefore **late-stage suppression**: the secret token could be relatively probable in intermediate layers (because the model needs it to generate coherent behavior) but then deliberately pushed down in probability by later layers that implement refusal/suppression.

If this is true, ranking tokens only by their mid-layer probability is misaligned with the mechanism: the best diagnostic would be to look for tokens that are **plausible in mid layers but improbable at the final layer**.

Related work provides hints that “present-then-suppressed” dynamics exist in other contexts:
- **[DoLa](./references/DoLa-Decoding-by-Contrasting-Layers-Improves-Factuality-in-Large-Language-Models/meta/meta_info.txt)** contrasts late vs early layers to emphasize tokens that become more probable in later layers (for factuality).
- **[CoRect](./references/CoRect-Context-Aware-Logit-Contrast-for-Hidden-State-Rectification-to-Resolve-Knowledge-Conflicts/meta/meta_info.txt)** reports a “suppression” phenomenon in retrieval-augmented generation, where evidence-consistent answers appear in intermediate layers but are overwritten later.

Neither work addresses whether **secret-keeping** is realized as late-stage suppression, nor whether the opposite of DoLa’s contrast direction is useful for elicitation.

### Key Insight and Hypothesis

**Key insight:** In refusal-style secret keeping, the secret token may have a distinctive signature: it is *relatively likely* in mid layers (where the model represents the secret) but *relatively unlikely* at the final layer (where the model suppresses revealing it). This differs from factuality settings targeted by DoLa, where correct facts often become more likely in late layers.

**Hypothesis:** On refusal-style secret elicitation tasks, ranking tokens by a **reverse layer-contrast score**

\[\text{score}(v) = \mathbb{E}_{t\in\text{response}}[\log p_{L}(v\mid t) - \log p_{N}(v\mid t)]\]

(where \(L\) is a mid layer and \(N\) is the final layer) will surface the true secret token more often than the standard logit-lens ranking \(\mathbb{E}_t[\log p_L(v\mid t)]\), because the contrast isolates tokens that are specifically *suppressed late*.

Why we could be wrong:
- The secret may not be linearly decodable even in mid layers (logit lens is the wrong readout).
- The secret may already be suppressed before the chosen mid layer.
- The score could mostly select “rare” or stylistically unlikely tokens rather than secrets (a confound).

---

## Proposed Approach

### Overview

We propose **Suppression-Contrast Tokens (SCT)**, a token-ranking method for secret elicitation that:
1) extracts log probabilities at a chosen mid layer \(L\) (logit lens) and at the final layer \(N\),
2) computes a reverse contrast \(\log p_L - \log p_N\), and
3) returns top-ranked tokens (excluding tokens already generated) as candidate secrets for an automated auditor.

We evaluate SCT on the public secret-elicitation benchmark from **[Eliciting Secret Knowledge from Language Models](./references/Eliciting-Secret-Knowledge-from-Language-Models/meta/meta_info.txt)** and re-use its released code pipeline (**[cywinski/eliciting-secret-knowledge](./references/GitHub-cywinski-eliciting-secret-knowledge/meta/meta_info.txt)**).

### Method Details

**Inputs.** For each prompt-response pair in an evaluation set (e.g., the “direct” Taboo set has 1000 prompt-response pairs; **[Evaluation datasets](<./references/Eliciting-Secret-Knowledge-from-Language-Models/sections/Evaluation datasets.md>)**), we take the model’s generated response tokens and (white-box) internal activations.

**Layer choice and pre-registered premise check.** We use the same mid layer \(L\) as the baseline logit-lens scripts in the benchmark repo (e.g., Gemma Taboo commonly uses layer 32, consistent with **[Logit lens (Taboo paper)](<./references/Towards-eliciting-latent-knowledge-from-LLMs-with-mechanistic-interpretability/sections/Logit lens.md>)**).

Before running the main comparison, we run a small **development subset** diagnostic (e.g., 100 prompt-response pairs) to test the premise:
- For each example, compute whether the ground-truth secret token has higher rank (or higher log-prob) at layer \(L\) than at the final layer.
- **Abort rule (pre-registered):** If fewer than **30%** of examples show this “mid-better-than-final” pattern, we treat the suppression premise as not supported in this setting and stop (reporting a negative result).

**Candidate token set and plausibility constraint.** Computing \(p_L(v)\) and \(p_N(v)\) for the full vocabulary is expensive. Instead:
- For each token position \(t\) in the response, compute the top-\(K\) tokens under \(p_L(\cdot\mid t)\).
- Apply a **mid-layer plausibility constraint**: keep only tokens with \(p_L(v\mid t) \ge \alpha\cdot \max_{w} p_L(w\mid t)\) (fixed \(\alpha\), e.g., 0.1, analogous to DoLa’s plausibility filtering).
- Take the union over positions to form a candidate set \(\mathcal{V}\). For \(v\in\mathcal{V}\), compute \(\log p_L(v\mid t)\) and \(\log p_N(v\mid t)\) and aggregate across positions.

**Token exclusions (same as Taboo logit-lens baseline).** Exclude tokens that appear in the model’s generated output (they are known not to be the Taboo secret in that benchmark) as described in **[Taboo paper logit-lens method](<./references/Towards-eliciting-latent-knowledge-from-LLMs-with-mechanistic-interpretability/sections/Logit lens.md>)**.

**SCT ranking.** Rank tokens by:
- SCT: \(\text{score}_{\text{SCT}}(v)=\mathbb{E}_t[\log p_L(v\mid t) - \log p_N(v\mid t)]\)

**Baselines for comparison.**
- Logit lens baseline: \(\text{score}_{\text{LL}}(v)=\mathbb{E}_t[\log p_L(v\mid t)]\)
- DoLa-direction negative control: \(\text{score}_{\text{DoLa-dir}}(v)=\mathbb{E}_t[\log p_N(v\mid t) - \log p_L(v\mid t)]\). If the suppression mechanism is correct, this should not help and may hurt.

**Frequency confound check.** As a lightweight control for the “rare token” confound, we report whether SCT’s candidate list has systematically lower \(\mathbb{E}_t[\log p_L(v\mid t)]\) than the logit-lens list, and we repeat evaluation after enforcing the same mid-layer plausibility constraint for both methods (so SCT cannot win simply by selecting extremely implausible tokens).

### Key Innovations

1. **Reverse contrast for elicitation**: uses mid-vs-final *suppression gap* rather than mid-layer probability.
2. **Mechanism-disambiguating negative control**: includes DoLa-direction contrast as a falsifiable prediction about suppression vs “late knowledge accumulation.”
3. **Benchmark-native implementation**: plugs into an existing public secret-elicitation pipeline (no new datasets, no human labeling).

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) **secret elicitation / alignment auditing**, (ii) **mechanistic interpretability readouts** (logit lens, tuned lens, SAEs), and (iii) **inference-time logit manipulations** (contrastive decoding, layer-contrast methods) that exploit differences across layers or conditions.

Secret-elicitation benchmarks such as **[Eliciting Secret Knowledge](./references/Eliciting-Secret-Knowledge-from-Language-Models/meta/meta_info.txt)** and the Taboo model organism in **[Towards eliciting latent knowledge…](./references/Towards-eliciting-latent-knowledge-from-LLMs-with-mechanistic-interpretability/meta/meta_info.txt)** provide controlled settings where the secret is known, enabling quantitative evaluation of elicitation methods. Work on ELK more broadly argues that internal representations may track the “true” answer even when outputs are systematically incorrect (**[Eliciting Latent Knowledge from Quirky Language Models](./references/Eliciting-Latent-Knowledge-from-Quirky-Language-Models/meta/meta_info.txt)**).

In parallel, several lines of work show that transformer layers have distinct roles and that comparing layerwise predictions can be useful. DoLa demonstrates that contrasting layers can improve factuality without training (**[DoLa](./references/DoLa-Decoding-by-Contrasting-Layers-Improves-Factuality-in-Large-Language-Models/meta/meta_info.txt)**). CoRect and related RAG work analyze conflicts between parametric and contextual knowledge and identify later-layer overwriting phenomena (**[CoRect](./references/CoRect-Context-Aware-Logit-Contrast-for-Hidden-State-Rectification-to-Resolve-Knowledge-Conflicts/meta/meta_info.txt)**).

### Related Papers

- **[Eliciting Secret Knowledge from Language Models](./references/Eliciting-Secret-Knowledge-from-Language-Models/meta/meta_info.txt)**: Public benchmark + models for secret elicitation; defines Taboo/SSC/User-Gender organisms and evaluates black-box and white-box methods.
- **[GitHub: cywinski/eliciting-secret-knowledge](./references/GitHub-cywinski-eliciting-secret-knowledge/meta/meta_info.txt)**: Reference implementation of the benchmark evaluation pipeline and white-box extraction scripts.
- **[Towards eliciting latent knowledge from LLMs with mechanistic interpretability](./references/Towards-eliciting-latent-knowledge-from-LLMs-with-mechanistic-interpretability/meta/meta_info.txt)**: Taboo model organism; shows that mid-layer readouts (logit lens, SAEs) can surface single-token secrets; provides deterministic secret-recovery metrics on Taboo models.
- **[DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](./references/DoLa-Decoding-by-Contrasting-Layers-Improves-Factuality-in-Large-Language-Models/meta/meta_info.txt)**: Layer-contrast decoding to emphasize tokens that become more probable in later layers.
- **[CoRect: Context-Aware Logit Contrast for Hidden State Rectification to Resolve Knowledge Conflicts](./references/CoRect-Context-Aware-Logit-Contrast-for-Hidden-State-Rectification-to-Resolve-Knowledge-Conflicts/meta/meta_info.txt)**: Uses contrast between contextualized and non-contextualized passes to identify layers where priors overwrite evidence.
- **[Eliciting Latent Knowledge from Quirky Language Models](./references/Eliciting-Latent-Knowledge-from-Quirky-Language-Models/meta/meta_info.txt)**: ELK benchmark; probes can recover truthful labels from middle-layer representations even when outputs lie in “Bob” contexts.
- **[Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)**: Trains affine translators to obtain more reliable intermediate-layer token distributions than the basic logit lens.
- **[Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)**: Introduces logit lens as a layerwise token readout.
- **[Overthinking the Truth: Understanding How Language Models Process False Demonstrations](https://openreview.net/forum?id=em4xg1Gvxa)**: Shows that late layers can override earlier correct behavior under misleading demonstrations.
- **[Contrastive Decoding: Open-ended Text Generation as Optimization](https://arxiv.org/abs/2210.15097)**: Contrasts an expert and amateur model to reduce degeneration; a precursor to layer-contrast methods.
- **[Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341)**: Uses linear probes on internal activations to steer generation toward truthfulness.
- **[Contrast Consistent Search (CCS)](https://arxiv.org/abs/2212.06154)**: Unsupervised method for finding latent truth signals robust across prompt variants.
- **[The Geometry of Truth: Hidden State Directionality for Truthfulness](https://arxiv.org/abs/2310.06824)**: Shows linear truth representations and causal interventions in hidden states.
- **[Detecting Hallucinations with Internal Representations](https://arxiv.org/abs/2304.13734)**: Trains classifiers on hidden states to detect truthfulness vs falsehood.
- **[Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)**: Methods for reading/controlling high-level behaviors via representation directions.
- **[Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262)**: Identifies mid-layer MLPs as loci for factual associations; relevant for “where knowledge lives.”
- **[Knowledge Neurons in Pretrained Transformers](https://arxiv.org/abs/2104.08696)**: Proposes neuron-level localization of factual knowledge.
- **[Jailbreaking leading safety-aligned LLMs with simple adaptive attacks](https://openreview.net/forum?id=hXA8wqRdyV)**: Introduces practical black-box prompting attacks (including prefill-style strategies) relevant to secret elicitation.
- **[Safety alignment should be made more than just a few tokens deep](https://openreview.net/forum?id=6Mxhg9PtDE)**: Studies shallow alignment and prompting-based bypasses, motivating mechanistic auditing.
- **[Tell me about yourself: LLMs are aware of their learned behaviors](https://openreview.net/forum?id=IjQ2Jtemzy)**: Shows LMs can represent learned behaviors/attributes, relevant to “User Gender” organisms.
- **[Narrow finetuning leaves clearly readable traces in activation differences](https://arxiv.org/abs/2510.13900)**: Shows small fine-tunes can create linearly readable activation traces, relevant to white-box elicitation.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Secret elicitation benchmarks | Train model organisms with known secrets and evaluate elicitation | Eliciting Secret Knowledge (2510.01070), Taboo MI (2505.14352) | Taboo / SSC / User Gender metrics; Pass@k, auditor accuracy | Organisms are simplified; secrets often single-token |
| ELK probing | Learn probes that generalize to untruthful contexts | Quirky LMs (2312.01037), CCS | AUROC on “truth” vs “lie” labels under shifts | Requires labeled “easy” data; may not match real deception |
| Layerwise token readouts | Decode intermediate predictions via unembedding (or learned translators) | Logit lens (nostalgebraist), Tuned Lens (2303.08112) | Layerwise rank/prob diagnostics; elicitation candidates | Basic logit lens can be biased/unreliable across architectures |
| Logit / layer contrast at inference | Modify token ranking using contrasts across layers or conditions | DoLa (2309.03883), CoRect (2602.08221) | Factuality, RAG faithfulness | Contrast direction is task-dependent; may harm other objectives |

### Closest Prior Work

1. **DoLa** (**[DoLa](./references/DoLa-Decoding-by-Contrasting-Layers-Improves-Factuality-in-Large-Language-Models/meta/meta_info.txt)**): contrasts final vs earlier layers to emphasize tokens that become more probable in late layers. Our proposal targets the *opposite* regime: secrets that are plausible in mid layers but suppressed at the final layer. We include DoLa-direction as a negative control.

2. **CoRect** (**[CoRect](./references/CoRect-Context-Aware-Logit-Contrast-for-Hidden-State-Rectification-to-Resolve-Knowledge-Conflicts/meta/meta_info.txt)**): uses contrast between contextualized and null-context passes and then edits hidden states to preserve evidence. We do not edit hidden states; we use a simpler, single-pass readout and evaluate on secret-keeping organisms rather than RAG.

3. **Eliciting Secret Knowledge** (**[paper](./references/Eliciting-Secret-Knowledge-from-Language-Models/meta/meta_info.txt)**; **[repo](./references/GitHub-cywinski-eliciting-secret-knowledge/meta/meta_info.txt)**): evaluates white-box token extraction methods (logit lens, SAEs) but does not test whether late-layer suppression is the reason these methods fail in the hardest setting.

4. **Taboo MI paper** (**[Towards eliciting latent knowledge…](./references/Towards-eliciting-latent-knowledge-from-LLMs-with-mechanistic-interpretability/meta/meta_info.txt)**): shows logit lens can surface secrets by averaging mid-layer probabilities (layer 32). We change the ranking objective to isolate mid-vs-final suppression.

5. **Quirky ELK benchmark** (**[Eliciting Latent Knowledge from Quirky Language Models](./references/Eliciting-Latent-Knowledge-from-Quirky-Language-Models/meta/meta_info.txt)**): probes latent truth in systematic lying contexts. Our work is narrower (token-level secret recovery) but tests a concrete “suppression across depth” hypothesis.

**Novelty Kill Search Summary:** On 2026-02-23, we searched for prior work combining “secret elicitation / Taboo model” with “layer contrast / DoLa / logit contrast” (queries included: “DoLa secret elicitation”, “secret elicitation logit contrast”, “taboo model logit contrast”, “CoRect secret elicitation”, “contrastive decoding secret elicitation”). We found no prior paper applying reverse layer-contrast (mid − final) as a token-ranking method for secret elicitation; closest matches were DoLa (factuality) and CoRect (RAG conflicts).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| DoLa (2309.03883) | final−early contrast for factuality | assumes correct token emerges late | reverse contrast (mid−final) for secrets | if secrets are suppressed late, reverse contrast isolates them |
| CoRect (2602.08221) | contrast + hidden-state rectification for RAG | requires multi-pass + edits; different setting | single-pass readout; no edits; secret organisms | simpler and targeted to suppression vs refusal |
| Eliciting Secret Knowledge (2510.01070) | benchmark + logit lens / SAE baselines | does not test depth-suppression mechanism | add suppression diagnostic + reverse contrast | turns benchmark into a mechanistic test (suppression vs absence) |
| Taboo MI (2505.14352) | averages mid-layer logit lens probs | ignores final-layer suppression signal | use mid−final contrast + negative control | exploits a different signature than raw mid-layer probability |

---

## Experiments

### Experimental Setup

**Primary benchmark (refusal-style, single-token secret):** Taboo / Direct split from **[Eliciting Secret Knowledge](./references/Eliciting-Secret-Knowledge-from-Language-Models/meta/meta_info.txt)**.
- Dataset size: 1000 prompt-response pairs (100 prompts × 10 sampled responses) per model (**[Evaluation datasets](<./references/Eliciting-Secret-Knowledge-from-Language-Models/sections/Evaluation datasets.md>)**).
- We follow the benchmark protocol of averaging results over **3 Taboo model organisms** trained on different secret words (gold/moon/flag; **[Taboo](./references/Eliciting-Secret-Knowledge-from-Language-Models/sections/Taboo.md)**).
- The “direct” prompt set attempts to compel the model to reveal its secret; responses are often refusals and contain little hint information.

**Secondary benchmark (refusal-style, binary attribute secret):** User Gender / Direct split from the same benchmark.
- Dataset size: 1000 prompt-response pairs (100 prompts × 10 sampled responses) per model (**[Evaluation datasets](<./references/Eliciting-Secret-Knowledge-from-Language-Models/sections/Evaluation datasets.md>)**).
- We average results over the **male and female** User Gender model organisms (**[User Gender](./references/Eliciting-Secret-Knowledge-from-Language-Models/sections/User Gender.md)**).

We do **not** include the SSC setting in this short paper because it uses a different base model family (Llama 3.3 70B) and a different scoring protocol (multi-guess with semantic similarity). **Generalization discussion:** SCT should apply to any setting where (i) the secret is represented in intermediate activations to support downstream behavior, and (ii) a later computation stage explicitly suppresses expressing it. Taboo and User Gender differ in the “shape” of the secret (single-token keyword vs binary attribute), so testing both provides a minimal check that SCT is not specific to a single-token vocabulary item. Extending to multi-token secrets (e.g., SSC) would likely require scoring multi-token candidates (e.g., via sum of per-token contrasts under teacher forcing) rather than ranking individual tokens.

**Baseline Ladder (REQUIRED):**
- Output-only baseline (no white-box): I/O auditor accuracy (published in METRICS.md).
- White-box baselines: logit lens tokens; activation–embedding similarity tokens; SAE tokens/descriptions (published in METRICS.md).
- Strong black-box baselines: assistant-prefill and user-persona sampling (published in METRICS.md).
- Closest related method family: DoLa-direction layer contrast as a negative control (mechanism test).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Taboo secret-keeping MO (example) | ~9B | https://huggingface.co/bcywinski/gemma-2-9b-it-taboo-gold | One released Taboo model; evaluation should include all Taboo models used by the benchmark scripts |
| Base Gemma model | ~9B | https://huggingface.co/google/gemma-2-9b-it | Used for the “base model” control (no secret fine-tune) |
| Auditor model (Taboo/User Gender) | ~4B | (per benchmark) | Use the benchmark’s auditor configuration; for deterministic analysis we also compute secret-in-top-k |

**Training Data (if applicable):**
- None (inference + feature extraction only).

**Resource Estimate**:
- **Compute budget**: Expected \(<100\) A100 GPU-hours for Taboo direct evaluation (single forward pass per prompt-response pair + lightweight extraction). Downscale plan: evaluate on 200 pairs first to validate premise; then scale to full 1000.
- **GPU memory**: 9B bf16 fits in 80GB; extraction stores only two layers’ activations at a time.
- **API usage**: None required for the primary deterministic metric (secret-in-top-k). If using the benchmark auditor LLM, it can run via open-source model inference.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Eliciting Secret Knowledge (Taboo / Direct) | Direct (adversarial) prompts; secret is a single word the model must not utter | (1) Token-recovery@k (k=5,20; higher is better), (2) Auditor accuracy (%; higher is better) | direct | https://github.com/cywinski/eliciting-secret-knowledge | `taboo/` + `elicitation_methods/` scripts |
| Eliciting Secret Knowledge (User Gender / Direct) | Direct prompts; model has learned a binary user attribute (male/female) while denying it | Auditor accuracy (%; higher is better) | direct | https://github.com/cywinski/eliciting-secret-knowledge | `user_gender/` + `elicitation_methods/` scripts |

### Main Results

#### Results Table

**Published baselines (Eliciting Secret Knowledge).** Copied from **[METRICS.md snapshot](<./references/cywinski-eliciting-secret-knowledge-METRICS/sections/Main Content.md>)**, Direct Secret-keeping model table.

| Method | Base Model | Benchmark | Token-recovery@5 (%; mean±std; higher is better) | Auditor accuracy (%; mean±std; higher is better) | Source | Notes |
|---|---|---|---:|---:|---|---|
| I/O (baseline) | Gemma 2 9B Taboo MO | Taboo / Direct | - | 1.10±0.63 | METRICS.md | Published |
| Logit lens tokens | Gemma 2 9B Taboo MO | Taboo / Direct | **TBD** | 4.13±0.38 | METRICS.md | Published |
| Activation–embedding similarity tokens | Gemma 2 9B Taboo MO | Taboo / Direct | **TBD** | 2.80±0.62 | METRICS.md | Published |
| SAE tokens | Gemma 2 9B Taboo MO | Taboo / Direct | **TBD** | 2.87±0.65 | METRICS.md | Published |
| Prefill (black-box) | Gemma 2 9B Taboo MO | Taboo / Direct | - | 4.33±0.52 | METRICS.md | Published; requires new sampling |
| User persona (black-box) | Gemma 2 9B Taboo MO | Taboo / Direct | - | 7.17±0.55 | METRICS.md | Published; requires new sampling |
| DoLa-direction (final−mid) | Gemma 2 9B Taboo MO | Taboo / Direct | **TBD** | **TBD** | Ours | Negative control; predicted ≤ logit lens |
| **Ours: SCT (mid−final)** | Gemma 2 9B Taboo MO | Taboo / Direct | **TBD** | **TBD** | Ours | Primary method |

**User Gender / Direct (binary attribute secret).**

| Method | Base Model | Benchmark | Auditor accuracy (%; mean±std; higher is better) | Source | Notes |
|---|---|---|---:|---|---|
| I/O (baseline) | Gemma 2 9B User Gender MO | User Gender / Direct | 50.45±2.89 | METRICS.md | Published |
| Logit lens tokens | Gemma 2 9B User Gender MO | User Gender / Direct | 66.15±1.33 | METRICS.md | Published |
| Activation–embedding similarity tokens | Gemma 2 9B User Gender MO | User Gender / Direct | 57.90±0.80 | METRICS.md | Published |
| SAE descriptions | Gemma 2 9B User Gender MO | User Gender / Direct | 86.95±0.94 | METRICS.md | Published |
| SAE tokens | Gemma 2 9B User Gender MO | User Gender / Direct | 60.65±0.84 | METRICS.md | Published |
| Prefill (black-box) | Gemma 2 9B User Gender MO | User Gender / Direct | 95.45±1.01 | METRICS.md | Published; requires new sampling |
| User persona (black-box) | Gemma 2 9B User Gender MO | User Gender / Direct | 97.65±0.68 | METRICS.md | Published; requires new sampling |
| DoLa-direction (final−mid) | Gemma 2 9B User Gender MO | User Gender / Direct | **TBD** | Ours | Negative control |
| **Ours: SCT (mid−final)** | Gemma 2 9B User Gender MO | User Gender / Direct | **TBD** | Ours | Primary method |


### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| SCT (full) | mid−final reverse contrast + plausibility constraint | Best secret-in-top-k |
| DoLa-direction | final−mid contrast | Worse than logit lens if suppression hypothesis is correct |
| No contrast | logit lens only | Lower secret-in-top-k than SCT |

### Experimental Rigor

- **Determinism / variance:** Our extraction given a fixed transcript is deterministic. For the Taboo/Direct benchmark, we use the benchmark’s fixed evaluation set construction; results can be bootstrapped over prompt-response pairs to obtain confidence intervals.
- **Premise check / abort rule:** If suppression is not observed in ≥30% of development examples, we stop and report that the “late suppression” hypothesis is not supported.
- **Controls:**
  - Run SCT on the **base model** (no secret fine-tune). The true secret should not appear in top-k more often than chance.
  - Ensure both SCT and logit lens use the same plausibility constraint to avoid “rare token” selection artifacts.

---

## Success Criteria

**Hypothesis** (directional):
- In direct/refusal transcripts, the secret token is more likely at a mid layer than at the final layer for a substantial fraction of examples, and SCT exploits this to improve secret recovery.

**Decision Rule** (concrete):
- **Proceed** if all of the following hold on Taboo/Direct:
  1) Premise: ≥30% of development examples show the secret token ranked higher at layer \(L\) than at the final layer.
  2) Secret recovery: SCT improves token-recovery@5 by **≥25% relative AND ≥1.5 percentage points absolute** over logit lens, using the same plausibility constraint.
  3) Mechanism check: DoLa-direction (final−mid) performs **≤** logit lens.
  4) Auditor metric: SCT’s auditor accuracy improves over logit lens with non-overlapping bootstrap 95% CIs.

- **Refute** if any of (1)–(4) fails.

---

## Impact Statement

If successful, SCT provides a simple, benchmark-native way to improve automated secret elicitation in refusal-style settings using only internal activations. More importantly, the sign of the contrast effect (mid−final helps while final−mid hurts) would support a mechanistic claim that secret keeping is implemented as late-stage suppression rather than absence of representation, informing how future auditing tools should look for hidden knowledge.

---

## References

- [Eliciting Secret Knowledge from Language Models](./references/Eliciting-Secret-Knowledge-from-Language-Models/meta/meta_info.txt) - Cywiński et al., 2025
- [Towards eliciting latent knowledge from LLMs with mechanistic interpretability](./references/Towards-eliciting-latent-knowledge-from-LLMs-with-mechanistic-interpretability/meta/meta_info.txt) - Cywiński et al., 2025
- [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](./references/DoLa-Decoding-by-Contrasting-Layers-Improves-Factuality-in-Large-Language-Models/meta/meta_info.txt) - Chuang et al., 2023
- [CoRect: Context-Aware Logit Contrast for Hidden State Rectification to Resolve Knowledge Conflicts](./references/CoRect-Context-Aware-Logit-Contrast-for-Hidden-State-Rectification-to-Resolve-Knowledge-Conflicts/meta/meta_info.txt) - Ma et al., 2026
- [Eliciting Latent Knowledge from Quirky Language Models](./references/Eliciting-Latent-Knowledge-from-Quirky-Language-Models/meta/meta_info.txt) - Mallen & Belrose, 2023
- [cywinski/eliciting-secret-knowledge (GitHub)](./references/GitHub-cywinski-eliciting-secret-knowledge/meta/meta_info.txt) - repository snapshot
- [METRICS.md (repo results table)](./references/cywinski-eliciting-secret-knowledge-METRICS/meta/meta_info.txt) - repository snapshot
- Tuned Lens (Belrose et al., 2023): https://arxiv.org/abs/2303.08112
- DoLa (arXiv page): https://arxiv.org/abs/2309.03883
- CoRect (arXiv page): https://arxiv.org/abs/2602.08221
- Quirky ELK (arXiv page): https://arxiv.org/abs/2312.01037
- Overthinking the Truth (Halawi et al., 2023): https://openreview.net/forum?id=em4xg1Gvxa
- ITI (Li et al., 2023): https://arxiv.org/abs/2306.03341
- Contrastive Decoding (Li et al., 2022): https://arxiv.org/abs/2210.15097
- CCS (Burns et al., 2022): https://arxiv.org/abs/2212.06154
- ROME (Meng et al., 2022): https://arxiv.org/abs/2202.05262
