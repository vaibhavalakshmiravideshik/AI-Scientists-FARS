# untitled

# Complementary Negation Regularization for Semantically Grounded Model Editing

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS (Conference on Neural Information Processing Systems), ICML (International Conference on Machine Learning), ICLR (International Conference on Learning Representations), ACL (Annual Meeting of the Association for Computational Linguistics), EMNLP (Conference on Empirical Methods in Natural Language Processing), or similar

## Introduction

### Context and Motivation

Large language models (LLMs; neural networks trained to predict the next token over large text corpora) are widely deployed as assistants and question-answering systems. In many applications, they must provide up-to-date factual information, but their parametric knowledge can become outdated or incorrect. Updating a deployed model by full retraining or large-scale fine-tuning is often too expensive and operationally risky.

Model editing (also called knowledge editing) aims to update a small set of model behaviors tied to specific facts (e.g., changing the completion of a factual prompt from an old object to a new object) while preserving unrelated behaviors and general capabilities. Modern editing methods are typically evaluated by whether the edited model outputs the new target answer on the original prompt, plus auxiliary metrics such as generalization to paraphrases and locality (how much unrelated knowledge changes).

Recent evaluation work argues that many reported editing successes may be brittle under small semantic shifts in the prompt. In particular, **[Is Model Editing Built on Sand?](./references/Is%20Model%20Editing%20Built%20on%20Sand%20Revealing%20Its%20Illusory%20Success%20and%20Fragile%20Foundation/meta/meta_info.txt)** shows that several state-of-the-art locate-then-edit methods (methods that first identify “knowledge-critical” internal parameters and then apply targeted updates) can still output the edited target under semantically opposite prompts created by simple lexical negation.

### The Problem

**Negation fragility in model editing.** After editing a fact so that the model answers a factual prompt with a new target object `y*` (e.g., replacing the model’s completion with “English”), some editing methods still produce `y*` when the prompt is negated (e.g., “… is not”). Built-on-Sand formalizes this with four settings that combine positive/negative edit data and positive/negative test queries:

- **PP (positive edit, positive test)**: standard **efficacy** (edit success rate; higher is better).
- **PN (positive edit, negative test)**: **hallucination under negation** (rate of outputting `y*` under a negated query; lower is better).
- **NN (negative edit, negative test)** and **NP (negative edit, positive test)**: analogous controls (also interpreted as hallucination in the mismatched cases).

Built-on-Sand reports that on Multi-Counterfact (MCF; a multi-edit variant of CounterFact used to evaluate sequential/batch knowledge edits) with Llama-3-8B-Instruct, MEMIT (Mass-Editing Memory in a Transformer; a locate-then-edit method) achieves **PP = 98.2%** (efficacy; higher is better) but **PN = 69.4%** (hallucination; lower is better), yielding rectified efficacy **PP - PN = 28.8** (higher is better) (**[Built-on-Sand §5.2](./references/Is%20Model%20Editing%20Built%20on%20Sand%20Revealing%20Its%20Illusory%20Success%20and%20Fragile%20Foundation/sections/5.2%20Negation%20of%20knowledge%20queries.md)**).

This failure matters in practice because real users frequently express constraints via negation or falsity (e.g., “X is not Y”, “it is false that …”). If edited knowledge does not respect common negation phrasing, edits are difficult to trust for compliance updates, factual corrections, and safety patches.

**Open question for fine-tuning-based editing.** Built-on-Sand primarily studies locate-then-edit methods. It is not established whether strong fine-tuning-based editors have the same negation fragility. In particular, **[FINE-TUNING DONE Right IN MODEL EDITING](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/meta/meta_info.txt)** proposes LocFT-BF (Localized Fine-Tuning with a Breadth-First pipeline), a scalable baseline that fine-tunes only a small subset of parameters using standard mini-batch training. Our experiments therefore start by measuring whether LocFT-BF also exhibits high PN hallucination, and then test whether a minimal complementary training signal can reduce it.

### Key Insight and Hypothesis

**Key insight.** Many editing objectives are one-directional: they increase the probability of generating `y*` under the positive prompt, but provide no explicit counter-signal for semantically opposite prompts. This can allow the model to learn a shortcut association (“produce `y*` when the decisive entity token appears”) without using supportive tokens that encode negation.

**Hypothesis (lexical negation generalization).** Adding an unlikelihood-style loss that *penalizes generating the edited target* under a single, deterministic negation template during editing will:

1. Reduce PN hallucination (lower is better) and increase rectified efficacy (PP - PN; higher is better).
2. Partially generalize to held-out lexical negation templates (e.g., “It is false that …”) not used during training.

The outcome is uncertain because the model may only learn a brittle pattern tied to the specific negation wording seen during training (e.g., “not”), or improvements could be explained by generic regularization rather than learning to condition on negation semantics.

---

## Proposed Approach

### Overview

We propose **Complementary Negation Regularization (CNR)**, a minimal modification to breadth-first fine-tuning model editing that adds an explicit loss term discouraging the edited target answer under a negated version of the prompt.

CNR is intentionally scoped to **lexically marked negation** (prompts that contain explicit negation cues such as “not” or “false”), because it is fully automatable and directly targets a documented evaluation failure mode.

### Method Details

**Editing instance.** Each edit example provides:
- A prompt `p` that ends immediately before the object span (e.g., “The mother language of Danielle Darrieux is”).
- A desired edited target sequence `y*` (e.g., “English”).

**Base editor: LocFT-BF (Localized Fine-Tuning with Breadth-First pipeline).** Following **[FINE-TUNING DONE Right IN MODEL EDITING](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/meta/meta_info.txt)**, we use a standard mini-batch fine-tuning loop over the edit set (a “breadth-first” pipeline that iterates over the full edit dataset across training epochs) while updating only a small parameter subset (localized fine-tuning). For Llama-3-8B, the paper’s selected tuning location is the MLP down-projection matrix in layer 22 (**[LocFT-BF Table 2](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/sections/TAILORING%20FINE-TUNING%20FOR%20MODEL%20EDITING.md)**). Here “MLP” refers to the transformer feed-forward block (multi-layer perceptron), and the “down-projection” is the matrix that maps the MLP hidden dimension back to the model hidden size.

**CNR augmentation: complementary negation loss.**

1. **Construct a negated prompt (seen during training).** Following Built-on-Sand’s PN construction, define:
   - `p_not = p + " not"` (append “not” at the end of the prompt).

2. **Define the losses.**
   - **Positive loss** `L_pos`: standard token-level cross-entropy loss (negative log-likelihood) under teacher forcing (conditioning on the ground-truth previous tokens) to increase `P(y* | p)`.
   - **Complementary loss** `L_ul`: an unlikelihood loss that penalizes generating `y*` under `p_not`. Concretely, for each target token `y*_t` under teacher forcing, we apply:
     - `L_ul = -∑_t log(1 - P(y*_t | p_not, y*_{<t}))`.

3. **Total objective.**
   - `L = L_pos + λ · L_ul`, with `λ ≥ 0` controlling the strength of complementary supervision.

**Negation vs. generic regularization control.** To test whether gains come from negation conditioning rather than generic additional loss, we add a matched control condition:
- **Random-unlikelihood control**: apply the same unlikelihood loss form and weight `λ`, but penalize a random non-target object sequence `y_rand` (sampled from the batch) under `p_not` instead of penalizing `y*`.

### Key Innovations

- **Complementary editing objective**: adds an explicit “do not output `y*` here” signal under a negated prompt, while keeping the editor a simple fine-tuning method.
- **Template generalization evaluation**: trains with only one negation template (`p_not`) and evaluates on held-out lexical negation templates to test whether the behavior is transferable beyond a single string.
- **Control for alternative explanations**: the random-unlikelihood control isolates whether improvements are attributable to negation conditioning rather than generic regularization.

---

## Related Work

### Field Overview

Knowledge editing methods for LLMs are commonly grouped into: (i) locate-then-edit methods that compute targeted parameter updates (e.g., ROME and MEMIT), (ii) meta-learning and hypernetwork methods that learn to predict updates, (iii) parameter-extension methods that add new trainable modules, and (iv) fine-tuning-based and parameter-efficient fine-tuning methods that directly optimize a small subset of parameters.

A second line of work focuses on evaluation: reliability on the edited prompt, generalization to paraphrases, locality (minimal unintended changes), and robustness under distribution shift. Recent papers argue that standard evaluation protocols can overestimate real-world behavior and propose more deployment-aligned evaluations (e.g., autoregressive decoding, meaning scoring the model’s freely generated output, instead of teacher forcing).

Built-on-Sand is most directly relevant to this proposal because it introduces a simple negation-based test that reveals a specific robustness failure: edited models may output the edited target even when the prompt semantics flip. Our proposal complements this evaluation by proposing a minimal training objective that directly targets the negation failure for fine-tuning-based editing.

### Related Papers

- **[Is Model Editing Built on Sand? Revealing Its Illusory Success and Fragile Foundation](./references/Is%20Model%20Editing%20Built%20on%20Sand%20Revealing%20Its%20Illusory%20Success%20and%20Fragile%20Foundation/meta/meta_info.txt)**: Introduces PP/PN/NN/NP negation evaluation and shows high hallucination under negation for multiple locate-then-edit methods.
- **[FINE-TUNING DONE Right IN MODEL EDITING](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/meta/meta_info.txt)**: Proposes LocFT-BF, showing that localized fine-tuning with a breadth-first, mini-batch pipeline is a strong and scalable editing baseline.
- **[Context-Robust Knowledge Editing for Language Models](./references/Context-Robust%20Knowledge%20Editing%20for%20Language%20Models/meta/meta_info.txt)**: Introduces CHED (Contextual Hop Editing Dataset; an editing benchmark with realistic prefix contexts) and CoRE (Context Robust Editing) to improve robustness to contextual prefixes.
- **[Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262)**: A foundational locate-then-edit method that edits factual associations via targeted weight updates.
- **[Mass-Editing Memory in a Transformer (MEMIT)](https://openreview.net/forum?id=MkbcAHIYgyS)**: Extends ROME-style editing to batch and multi-layer updates for editing many facts.
- **[MAKE: Memory-Associated Knowledge Editing](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.26/132652/MAKE-Memory-Associated-Knowledge-Editing)**: Adds auxiliary objectives including an unlikelihood-style IMI (Irrelevant Memory Invalidation) loss to suppress old-object token associations; similar loss family but targets a different failure mode than negation robustness.
- **[The Mirage of Model Editing: Revisiting Evaluation in the Wild](https://arxiv.org/abs/2502.11177)**: Shows that evaluation protocol choices can inflate editing success and proposes more deployment-aligned evaluation.
- **[Better Call SAUL: Fluent and Consistent Language Model Editing with Generation Regularization](https://arxiv.org/abs/2410.02433)**: Proposes SAUL, a fine-tuning-based editor that adds generation regularization to preserve fluency and consistency.
- **[CoME: An Unlearning-based Approach to Conflict-free Model Editing](https://arxiv.org/abs/2502.15826)**: Uses targeted unlearning to reduce conflicts between old and new knowledge during editing.
- **[Unveiling the Pitfalls of Knowledge Editing for Large Language Models](https://openreview.net/forum?id=fNktD3ib16)**: Highlights multi-edit failure modes and proposes improved evaluation and mitigations.
- **[Are We Evaluating the Edit Locality of LLM Model Editing Properly?](https://arxiv.org/abs/2601.17343)**: Critiques common locality metrics and proposes ground-truth-free deviation measures.
- **[Can Knowledge Editing Really Correct Hallucinations? (HalluEditBench)](https://arxiv.org/abs/2410.16251)**: Benchmarks editing for hallucination correction and emphasizes trade-offs among efficacy, generalization, and locality.
- **[Propagation and Pitfalls: Reasoning-based Assessment of Knowledge Editing](https://arxiv.org/abs/2401.17585)**: Proposes ReCoE, evaluating whether edited facts propagate through multi-step reasoning.
- **[Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue](https://arxiv.org/abs/2401.04700)**: Shows that edits can harm general capabilities and proposes regularization strategies.
- **[Knowledge Editing via Adapted Direct Preference Optimization (KDPO)](https://arxiv.org/abs/2406.09920)**: Uses Direct Preference Optimization (DPO; a preference-learning objective) adapted to editing, training with negative samples to encourage the edited model to prefer desired outputs.
- **[Mitigating Heterogeneous Token Overfitting in LLM Knowledge Editing](https://openreview.net/forum?id=vOu5K93z4f)**: Studies overfitting to specific tokens during editing and proposes mitigations.
- **[Robust Massive Model Editing via Noise-Aware Memory Optimization (NAMET)](https://arxiv.org/abs/2505.11876)**: Targets robustness for large numbers of edits by using noise-aware optimization to reduce interference across edits.
- **[Keys to Robust Edits: from Theoretical Insights to Practical Advances](https://arxiv.org/abs/2410.09338)**: Analyzes robustness failures in editing and proposes improved methods.
- **[Evaluating the Reversal Curse in Model Editing (BAKE)](https://openreview.net/pdf/aaa7b291fb6b571ea74e58a7d3125942fb510201.pdf)**: Studies bidirectional generalization failures (“reversal curse”), where an edit succeeds in one direction but does not generalize to logically related reverse queries.
- **[AnyEdit: Edit Any Knowledge Encoded in Language Models](https://arxiv.org/abs/2502.05628)**: Extends editing to broader knowledge formats beyond short factual prompts.
- **[Investigating Model Editing for Unlearning in Large Language Models](https://arxiv.org/abs/2512.20794)**: Connects editing and unlearning evaluation perspectives.
- **[Has this Fact been Edited? Detecting Knowledge Edits in Language Models](https://arxiv.org/abs/2405.02765)**: Studies detectability of edited facts, relevant to security considerations.
- **[Editing Large Language Models Poses Serious Safety Risks](https://arxiv.org/abs/2502.02958)**: Discusses dual-use and security risks associated with editing methods.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Locate-then-edit | Identify a small set of internal parameters and apply targeted updates | ROME; MEMIT; Built-on-Sand critique | CounterFact, ZsRE; Built-on-Sand PP/PN/NN/NP | Can learn shortcut associations; robustness failures under negation and context shift |
| Fine-tuning-based editing | Optimize a small parameter subset with stochastic gradient descent (SGD; a standard gradient-based optimizer family) | LocFT-BF; SAUL (a regularized fine-tuning editor) | CounterFact, ZsRE; WILD-style evaluation (WILD refers to evaluation protocols that use autoregressive decoding to better match deployment) | Can overfit or cause collateral changes without careful locality controls |
| Robust benchmarks and evaluation | Make evaluation closer to deployment conditions (decoding, contexts, negative cases) | Mirage/WILD; CHED/CoRE; Built-on-Sand | CHED, QAEdit (a question-answering-style knowledge editing benchmark used in WILD-style evaluation), CounterFact-derived tests | Harder evaluation typically lowers headline scores but improves realism |

### Closest Prior Work

1. **Built-on-Sand** (**[paper](./references/Is%20Model%20Editing%20Built%20on%20Sand%20Revealing%20Its%20Illusory%20Success%20and%20Fragile%20Foundation/meta/meta_info.txt)**): Defines a negation-based evaluation protocol (PP/PN/NN/NP) and demonstrates that multiple locate-then-edit methods have high hallucination rates under negation. It does not propose a training objective that directly enforces complementary behavior under negation. Our proposal treats the Built-on-Sand evaluation protocol as a target behavior and adds a direct training signal for it.

2. **LocFT-BF** (**[paper](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/meta/meta_info.txt)**): Shows that localized fine-tuning with a breadth-first pipeline is a strong editing baseline, with high reliability and practical efficiency (edits completed within one second; lower is faster) (**[LocFT-BF Results & Analysis](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/sections/RESULTS%20%26%20ANALYSIS.md)**). However, LocFT-BF does not evaluate or optimize for negation robustness. Our proposal adds a minimal auxiliary loss to target a specific robustness failure mode.

3. **MAKE** (**[paper](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.26/132652/MAKE-Memory-Associated-Knowledge-Editing)**): Uses additional objectives including an unlikelihood-style loss (IMI) to suppress associations with the *old* object and improve portability. This is close in mechanism (unlikelihood training), but it targets a different negative signal than ours: CNR penalizes the *new* target `y*` under a semantically opposite (negated) prompt.

4. **SAUL and related regularized fine-tuning editors** (**[paper](https://arxiv.org/abs/2410.02433)**): Use generation-focused regularization to preserve fluency and consistency during editing. These methods do not explicitly train on negated prompts or measure PP/PN/NN/NP-style negation behavior.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [Built-on-Sand](./references/Is%20Model%20Editing%20Built%20on%20Sand%20Revealing%20Its%20Illusory%20Success%20and%20Fragile%20Foundation/meta/meta_info.txt) | Reveals negation fragility and proposes rectified efficacy metrics | No training-side fix proposed | Add complementary anti-target objective during editing | Directly targets the failure mode measured by PP/PN |
| [LocFT-BF](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/meta/meta_info.txt) | Strong localized fine-tuning baseline for editing | Not designed/evaluated for negation robustness | Add CNR term + held-out template evaluation | Minimal change to a strong baseline; expected to reduce PN hallucination |
| [MAKE](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.26/132652/MAKE-Memory-Associated-Knowledge-Editing) | Adds auxiliary objectives including unlikelihood on old-object associations | Does not target negation semantics; not evaluated on PP/PN | Use unlikelihood on negated prompts to suppress `y*` | Aligns training with semantic opposition rather than old-object suppression |
| [CoRE](./references/Context-Robust%20Knowledge%20Editing%20for%20Language%20Models/meta/meta_info.txt) | Improves robustness to prefix contexts | Not focused on negation behavior | Orthogonal axis: negation robustness | Addresses a different robustness failure mode |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Meta-Llama-3-8B-Instruct | 8B | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct | Primary base model (also used in Built-on-Sand). |

**Training Data (editing set):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| CounterFact | A factual knowledge editing benchmark; each example provides an entity–relation prompt plus an original object and a desired replacement object | 1,000 edits (subset) | https://huggingface.co/datasets/azhx/counterfact | See HuggingFace dataset card |
| ZsRE (KnowEdit) (optional) | A QA-style knowledge editing benchmark derived from Zero-shot Relation Extraction | 1,000 edits (optional extension) | https://huggingface.co/datasets/zjunlp/KnowEdit | See HuggingFace dataset card |

We scope the primary experiment to 1,000 edits to obtain a decisive measurement of negation behavior at low cost. We report mean +/- standard deviation over 3 random seeds.

**Other Resources (if applicable):**
- Built-on-Sand’s prompt construction rules for PP/PN and exact-match scoring (**[Built-on-Sand §5.2](./references/Is%20Model%20Editing%20Built%20on%20Sand%20Revealing%20Its%20Illusory%20Success%20and%20Fragile%20Foundation/sections/5.2%20Negation%20of%20knowledge%20queries.md)**).

**Resource Estimate**:
- **Compute budget**: ≤ 200 GPU-hours (one GPU running for one hour) total for baseline + CNR + random-unlikelihood control (including 3 seeds).
  - Justification: LocFT-BF is a lightweight localized fine-tuning procedure and reports sub-second per-edit runtime on standard editing benchmarks (**[LocFT-BF Results & Analysis](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/sections/RESULTS%20%26%20ANALYSIS.md)**). CNR adds one additional forward pass per training example (the negated prompt), so training cost is expected to be within ~2× the baseline.
- **GPU memory**: 1×A100 80GB (7-8B model with mixed precision; only one layer trainable).
- **API usage**: none.

**Infrastructure constraints** (proposals requiring these are infeasible):
- No search engine APIs.
- No browser/GUI environments.
- No human evaluation in the loop.
- This proposal uses open-source models only; it does not require sending prompts to OpenAI-hosted models.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| CounterFact (1k-edit subset) | Measures whether a model can be edited to output a desired new object on a factual prompt | PP efficacy (higher is better) | edited instances | https://huggingface.co/datasets/azhx/counterfact | Custom (generation + exact-match on `y*`) |
| Built-on-Sand-style negation evaluation | For each edited instance, evaluate a positive prompt `p` and a negated prompt `p_not = p + " not"` | PP efficacy (↑), PN hallucination (↓), rectified efficacy `PP - PN` (↑) | edited instances | N/A (derived from prompts) | Custom (template construction per Built-on-Sand §5.2) |
| Held-out lexical negation templates | Tests whether CNR generalizes beyond the training negation template | Held-out hallucination rates (↓) | edited instances | N/A (deterministic templates below) | Custom |
| Locality / collateral damage (minimal) | Tests whether editing changes unrelated model behavior | Average KL divergence (Kullback–Leibler divergence; lower means less distribution shift) between pre- and post-edit next-token distributions | 1,000 unrelated prompts | reuse CounterFact prompts not edited | Custom |

**Held-out templates (not used in CNR training):**
1. **False-prefix template**: `It is false that <p>`.
2. **Instruction-prefix template**: `Answer as if the following statement were false: <p>`.

For each template, we count “hallucination” if the edited model still completes with the edited target `y*` under greedy decoding.

**Evaluation Scripts:**
- Use the same decoding settings across methods (greedy decoding or temperature = 0).
- Score with exact match on the target object tokens `y*` (same criterion used by Built-on-Sand for efficacy/hallucination).

**Download Links Checklist:**
- [x] All benchmark datasets have download links
- [x] All training datasets have download links (CounterFact; optional ZsRE via KnowEdit)
- [x] All models have download links (Meta-Llama-3-8B-Instruct)
- [ ] Licenses are compatible with research use (verify dataset/model license terms before running)

### Main Results

#### Results Table

| Method | Base Model | Benchmark | PP efficacy ↑ | PN hallucination ↓ | Rectified efficacy (PP - PN) ↑ | Held-out hallucination (false-prefix) ↓ | Held-out hallucination (instruction-prefix) ↓ | Locality (avg KL drift) ↓ | Source | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| LocFT-BF | Llama-3-8B-Instruct | CounterFact (1k edits) | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | This work (baseline re-run) | LocFT-BF is a strong editor on CounterFact under standard metrics (e.g., reliability 99.73% on CounterFact; higher is better) (**[LocFT-BF Table 3](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/sections/RESULTS%20%26%20ANALYSIS.md)**), but negation metrics are not reported.
| LocFT-BF + CNR (ours) | Llama-3-8B-Instruct | CounterFact (1k edits) | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | This proposal | Adds complementary unlikelihood loss on `p_not` to reduce hallucination under negation.
| LocFT-BF + random-unlikelihood control | Llama-3-8B-Instruct | CounterFact (1k edits) | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | This proposal | Same loss form as CNR, but penalizes a random non-target object sequence instead of `y*`.

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| `λ = 0` | Remove the unlikelihood term (reduces to LocFT-BF) | Higher PN and held-out hallucination than CNR if complementary supervision matters |
| `λ` sweep | Try `λ ∈ {0.1, 0.3, 1.0}` | Confirms the effect is not overly sensitive; pick `λ` using a small validation subset |
| Alternative seen negation template | Replace training template with e.g. `It is not true that <p>` | Tests whether the gain depends on the exact placement/wording of the negation cue |

### Analysis (Optional)

- Breakdown results by relation type and entity frequency (if metadata is available).
- Compare where CNR helps most: reductions in PN hallucination vs improvements on held-out templates.

---

## Success Criteria

**Criterion 1: CNR reduces hallucination under negation without degrading standard efficacy**
- Hypothesis: CNR reduces PN hallucination and held-out template hallucination while keeping PP efficacy similar to LocFT-BF.
- Validation: Compared to LocFT-BF, CNR lowers PN hallucination and lowers both held-out hallucination rates, while PP efficacy does not decrease materially.
- Locality guardrail: the locality metric (average KL divergence on unrelated prompts; lower is better) does not worsen materially relative to LocFT-BF.

**Criterion 2: Gains reflect negation conditioning rather than generic regularization**
- Hypothesis: CNR improves held-out template behavior because it teaches a transferable response to lexical negation cues.
- Validation: CNR outperforms the random-unlikelihood control on both held-out templates.
- Decision rule: if CNR does not beat the random-unlikelihood control on held-out templates, we conclude that any PN improvement is not evidence of transferable negation conditioning (e.g., it may be generic regularization or template memorization).

---

## Impact Statement

If successful, CNR would provide a minimal, automated modification to fine-tuning-based model editing that improves semantic reliability under common negation phrasing. Practitioners using model editing for factual corrections or compliance patches could adopt CNR as a default training and evaluation baseline to reduce the risk that an edit behaves like a one-directional steering shortcut.

---

## References

- [Is Model Editing Built on Sand? Revealing Its Illusory Success and Fragile Foundation](./references/Is%20Model%20Editing%20Built%20on%20Sand%20Revealing%20Its%20Illusory%20Success%20and%20Fragile%20Foundation/meta/meta_info.txt) - Liu et al., 2025
- [FINE-TUNING DONE Right IN MODEL EDITING](./references/FINE-TUNING%20DONE%20Right%20IN%20MODEL%20EDITING/meta/meta_info.txt) - Yang et al., 2025
- [Context-Robust Knowledge Editing for Language Models](./references/Context-Robust%20Knowledge%20Editing%20for%20Language%20Models/meta/meta_info.txt) - Park et al., 2025
- [Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262) - Meng et al., 2022
- [Mass-Editing Memory in a Transformer (MEMIT)](https://openreview.net/forum?id=MkbcAHIYgyS) - Meng et al., 2023
- [MAKE: Memory-Associated Knowledge Editing](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.26/132652/MAKE-Memory-Associated-Knowledge-Editing) - Park et al., 2025
- [The Mirage of Model Editing: Revisiting Evaluation in the Wild](https://arxiv.org/abs/2502.11177) - Yang et al., 2025
- [Better Call SAUL: Fluent and Consistent Language Model Editing with Generation Regularization](https://arxiv.org/abs/2410.02433) - Cao et al., 2024
- [CoME: An Unlearning-based Approach to Conflict-free Model Editing](https://arxiv.org/abs/2502.15826) - Jung et al., 2025
- [Unveiling the Pitfalls of Knowledge Editing for Large Language Models](https://openreview.net/forum?id=fNktD3ib16) - Li et al., 2024
- [Are We Evaluating the Edit Locality of LLM Model Editing Properly?](https://arxiv.org/abs/2601.17343) - 2026
- [Can Knowledge Editing Really Correct Hallucinations? (HalluEditBench)](https://arxiv.org/abs/2410.16251) - Huang et al., 2024
- [Propagation and Pitfalls: Reasoning-based Assessment of Knowledge Editing](https://arxiv.org/abs/2401.17585) - 2024
- [Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue](https://arxiv.org/abs/2401.04700) - Gu et al., 2024
- [Knowledge Editing via Adapted Direct Preference Optimization (KDPO)](https://arxiv.org/abs/2406.09920) - 2024
- [Mitigating Heterogeneous Token Overfitting in LLM Knowledge Editing](https://openreview.net/forum?id=vOu5K93z4f) - 2025
- [Robust Massive Model Editing via Noise-Aware Memory Optimization (NAMET)](https://arxiv.org/abs/2505.11876) - 2025
- [Keys to Robust Edits: from Theoretical Insights to Practical Advances](https://arxiv.org/abs/2410.09338) - 2024
- [Evaluating the Reversal Curse in Model Editing (BAKE)](https://openreview.net/pdf/aaa7b291fb6b571ea74e58a7d3125942fb510201.pdf) - 2024/2025
- [AnyEdit: Edit Any Knowledge Encoded in Language Models](https://arxiv.org/abs/2502.05628) - 2025
- [Investigating Model Editing for Unlearning in Large Language Models](https://arxiv.org/abs/2512.20794) - 2025
- [Has this Fact been Edited? Detecting Knowledge Edits in Language Models](https://arxiv.org/abs/2405.02765) - 2024
- [Editing Large Language Models Poses Serious Safety Risks](https://arxiv.org/abs/2502.02958) - 2025
