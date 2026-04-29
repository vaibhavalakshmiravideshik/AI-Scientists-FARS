# untitled

# Fact-Check Grounding Loss for Semantically Consistent Model Editing

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) encode a large amount of factual knowledge in their parameters, but this knowledge can become incorrect, outdated, or undesirable after deployment. **Model editing** (also called knowledge editing) aims to update a small set of model behaviors tied to specific facts (e.g., changing the model’s completion for a prompt from an old object to a new object) while preserving unrelated behaviors and general capabilities.

Most model editing methods are trained and evaluated as *conditional generation* problems: given a prompt prefix (e.g., “The mother language of Danielle Darrieux is”), the edited model should generate a target answer token sequence (e.g., “English”). However, many real applications do not only require the model to *generate* the updated answer; they also require the model to *use* the edited knowledge coherently in different downstream formats, such as truth-judging (“is this statement true?”), contradiction handling, or multi-hop reasoning.

Recent evaluation work suggests that standard editing metrics can substantially overestimate real semantic integration. In particular, **[Is Model Editing Built on Sand?](./references/Is-Model-Editing-Built-on-Sand-Revealing-Its-Illusory-Success-and-Fragile-Foundation/meta/meta_info.txt)** introduces a **fact-checking style evaluation** where the edit target is moved from the model output into the *input statement*, and the model must answer “True/False”. Under this probe, several state-of-the-art editing methods show large gaps between token-level edit success (high) and truth-judgment accuracy (low). For example, on Qwen2.5-7B-Instruct with ZsRE edits, Built-on-Sand reports efficacy up to 96.5% but fact-check accuracy only 48.8–55.4% (Table 4; see `./references/Is-Model-Editing-Built-on-Sand-Revealing-Its-Illusory-Success-and-Fragile-Foundation/sections/5.3 Fact-checking style evaluation.md`). Since the correct label in this evaluation is always “True” (the statement is built from the edited target), ~50% accuracy means the edited model answers “False” for roughly half of the edited statements.

### The Problem

**Evaluation mismatch exposes a semantic brittleness.** Built-on-Sand’s fact-checking probe uses prompts like:

> `Judge whether the following statement is true or false: <prompt> <edited_target>`

and reports accuracy of generating “True”/“False”. This probe is intentionally simple: it keeps the same subject–relation–object content as the original edit, but changes the required output format so that the edited target tokens are no longer the gold answer tokens.

A natural question is whether *fine-tuning-based editors* avoid this failure mode. **[FINE-TUNING DONE Right IN MODEL EDITING](./references/FINE-TUNING-DONE-Right-IN-MODEL-EDITING/meta/meta_info.txt)** shows that **LocFT-BF (Localized Fine-Tuning with a Breadth-First pipeline)**—an editor that fine-tunes only a small parameter subset using an epoch-based mini-batch training loop—can be a very strong editing baseline under standard reliability/generalization/capability metrics, and is much faster than locate-then-edit methods (Table 3 reports <1 second per edit on multiple settings). However, LocFT-BF (like most editors) is still optimized as next-token prediction of the edited target, and does not explicitly train truth-judgment behavior.

**Practical implication.** If an edited model cannot reliably judge the truth of statements that contain the edited fact, then edits are difficult to trust as general knowledge updates: the update may behave like a narrow steering shortcut for one prompt format rather than a semantically grounded belief revision.

### Key Insight and Hypothesis

**Key insight.** The standard editing loss aligns the model to generate the edited target *as an output*, but Built-on-Sand’s probe requires the edited model to generate **“True/False”** for an input statement that *contains* the target. These are different output tokens and may rely on different internal decision boundaries. A model can therefore achieve high token-level efficacy while failing to transfer the edit to truth-judgment.

**Hypothesis.** Adding a small **balanced fact-checking auxiliary loss** during editing—training the model to output “True” for statements containing the new target and “False” for statements containing the old target—will substantially improve fact-checking accuracy and template transfer, without degrading standard token-level editing efficacy.

This hypothesis could be wrong for at least two reasons: (i) the auxiliary loss might only teach a dataset-specific classifier for the judge prompt rather than semantically grounding the edit, or (ii) the model might improve on the original judge template but not transfer to paraphrased judge prompts.

---

## Proposed Approach

### Overview

We propose **Fact-Check Grounding (FCG)**: a minimal modification to localized fine-tuning model editing that augments each edit with a tiny amount of **truth-conditional supervision** in the form of True/False judgments over statements derived from the edit. In this proposal, “truth-conditional supervision” means supervised training on *binary labels* (“True”/“False”) for statements constructed from the edit’s prompt and its old/new answers.

The method is designed to be (a) fully automatable, (b) compatible with standard editing datasets that include both the pre-edit answer and the post-edit target, and (c) testable with a small number of runs.

### Method Details

**Editing data.** Each edit instance provides a prompt `p`, an old answer `o_old` (the model’s pre-edit ground truth in the benchmark), and a new answer `o_new` (the desired edited target). For example, in KnowEdit’s ZsRE benchmark, entries include fields `prompt`, `ground_truth`, and `target_new` (see dataset schema at `https://huggingface.co/datasets/zjunlp/KnowEdit`).

**Base editor: LocFT-BF.** We follow LocFT-BF’s key design choices:
- Localized parameter updates (update only one module, e.g., an MLP down-projection matrix).
- Breadth-first training pipeline over the edit set (epoch-based, shuffled).

**Fact-check prompts.** For each edit, we build two fact-checking inputs using the Built-on-Sand template:
- **FC-Pos (label=True):** `J(p, o_new) = "Judge whether the following statement is true or false: {p} {o_new}"`
- **FC-Neg (label=False):** `J(p, o_old) = "Judge whether the following statement is true or false: {p} {o_old}"`

This creates a *balanced* binary classification problem that prevents the trivial “always True” shortcut (which would score well on FC-Pos only).

**Loss.** Let `L_edit` be the standard token-level cross-entropy loss that increases `P(o_new | p)` under *teacher forcing* (i.e., during training the model is conditioned on the ground-truth previous tokens, rather than its own sampled tokens). Let `L_fc` be a cross-entropy loss on generating the label token sequence (“True”/“False”) for the fact-check prompts (we evaluate/score using the first generated label token, but can train on the full label string).

We optimize:

\[
L = L_{edit} + \lambda\, L_{fc}
\]

where `L_fc` averages over FC-Pos and FC-Neg examples per edit. We use a small fixed `λ` (default 0.3) and keep it constant across methods; a small validation subset can be used to sanity-check sensitivity.

**Template transfer evaluation.** To reduce the risk of template memorization, we evaluate on a **held-out paraphrase** of the judge prompt that is never used during training, e.g.:

> `Is the following claim true or false? {p} {o}`

### Key Innovations

1. **Balanced truth-conditional supervision for editing**: we train the edited model to output correct “True/False” judgments on both the new statement (True) and the old statement (False), rather than only training generation of the new target tokens.
2. **Balanced Fact-Check Accuracy (BFC-Acc)**: we evaluate truth-judgment with a balanced metric that cannot be solved by always predicting “True”.
3. **Template-transfer check as a minimal anti-memorization guardrail**: we treat transfer to a paraphrased judge prompt as a required success condition.

---

## Related Work

### Field Overview

Model editing methods span several families: (i) **locate-then-edit** methods that identify “knowledge-critical” internal parameters and apply targeted updates (e.g., ROME/MEMIT-style rank-one updates), (ii) **fine-tuning-based** methods that directly optimize an editing loss over examples (including localized variants like LocFT-BF), and (iii) **external-memory / retrieval** approaches that store edits outside model weights (e.g., SERAC, IKE). Across these families, evaluation is typically based on whether the edited model outputs the new target on the original prompt, plus locality/generalization tests.

A growing set of work argues that conventional metrics can hide brittle or shortcut-like behavior. Built-on-Sand proposes negation tests and fact-checking style evaluation; Mirage-of-Model-Editing proposes WILD evaluation to replace teacher forcing and artificial truncation. Our proposal focuses on a specific evaluation mismatch revealed by Built-on-Sand (truth-judgment vs token generation) and tests whether a minimal training objective change can close it.

### Related Papers

- **[Is Model Editing Built on Sand? Revealing Its Illusory Success and Fragile Foundation](./references/Is-Model-Editing-Built-on-Sand-Revealing-Its-Illusory-Success-and-Fragile-Foundation/meta/meta_info.txt)**: Introduces negation and fact-checking probes showing large gaps between standard efficacy and semantic robustness.
- **[FINE-TUNING DONE Right IN MODEL EDITING](./references/FINE-TUNING-DONE-Right-IN-MODEL-EDITING/meta/meta_info.txt)**: Shows localized breadth-first fine-tuning (LocFT-BF) is a strong, scalable editing baseline under standard metrics.
- **[The Mirage of Model Editing: Revisiting Evaluation in the Wild](https://arxiv.org/abs/2502.11177)**: Proposes WILD evaluation and shows large drops from synthetic evaluation to realistic autoregressive evaluation.
- **[Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262)**: Foundational locate-then-edit method for single edits.
- **[Mass-Editing Memory in a Transformer (MEMIT)](https://openreview.net/forum?id=MkbcAHIYgyS)**: Extends ROME to multi-layer updates and batched/sequential edits.
- **[A Unified Framework for Model Editing (EMMET)](https://arxiv.org/abs/2403.14236)**: Unifies ROME/MEMIT-style objectives and studies equality constraints for batched edits.
- **[PMET: Precise Model Editing in a Transformer](https://arxiv.org/abs/2308.08742)**: Improves locate-then-edit by separating attention vs FFN representations for more precise edits.
- **[Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue (RECT)](https://arxiv.org/abs/2401.04700)**: Shows editing can degrade general abilities and proposes a regularization strategy.
- **[AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/abs/2410.02355)**: Uses null-space projection to preserve knowledge during sequential editing.
- **[AdaEdit: Advancing Continuous Knowledge Editing for Large Language Models](https://arxiv.org/abs/2507.02408)**: Improves continual editing by analyzing sparsity/low-rank structure in perturbations.
- **[Perturbation-Restrained Sequential Model Editing (PRUNE)](https://openreview.net/forum?id=bfI8cp8qmk)**: Constrains sequential edit perturbations to reduce interference.
- **[Memory-Based Model Editing at Scale (SERAC)](./references/Memory-Based-Model-Editing-at-Scale/meta/meta_info.txt)**: Semi-parametric editor with a scope classifier; evaluated on fact-checking and other tasks.
- **[FAME: Towards Factual Multi-Task Model Editing](./references/FAME-Towards-Factual-Multi-Task-Model-Editing/meta/meta_info.txt)**: Practical multi-task editing benchmark including a fact-check task format; proposes retrieval+caching editor.
- **[STEAM: A Semantic-Level Knowledge Editing Framework for Large Language Models](./references/Steam-A-Semantic-Level-Knowledge-Editing-Framework-for-Large-Language-Models/meta/meta_info.txt)**: Adds latent semantic alignment loss to improve portability/consistency, motivating semantic-level objectives.
- **[Context-Robust Knowledge Editing for Language Models (CoRE)](https://arxiv.org/abs/2505.23026)**: Improves robustness to realistic prefix contexts.
- **[Better Call SAUL: Fluent and Consistent Language Model Editing with Generation Regularization](https://arxiv.org/abs/2410.02433)**: Uses generation-time regularization to improve fluency/consistency of edits.
- **[Propagation and Pitfalls: Reasoning-based Assessment of Knowledge Editing](https://arxiv.org/abs/2401.17585)**: Evaluates whether edits propagate through reasoning chains.
- **[MQuAKE: Assessing Knowledge Editing in Language Models via Multi-hop Questions](https://arxiv.org/abs/2310.05845)**: Introduces multi-hop questions to test portability of edits.
- **[Keys to Robust Edits: from Theoretical Insights to Practical Advances](https://arxiv.org/abs/2410.09338)**: Theoretical/practical analysis of robustness issues in edits.
- **[Can Knowledge Editing Really Correct Hallucinations? (HalluEditBench)](https://arxiv.org/abs/2410.16251)**: Benchmarks editing for hallucination correction.
- **[Editing Large Language Models Poses Serious Safety Risks](https://arxiv.org/abs/2502.02958)**: Discusses misuse/safety risks of editing deployed models.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Locate-then-edit | Find “knowledge-critical” parameters and apply targeted updates | ROME, MEMIT, PMET, EMMET, AlphaEdit | CounterFact/MCF, ZsRE, WCF, MQuAKE; efficacy/paraphrase/locality | Can be brittle under negation/fact-check probes; may have hidden side effects |
| Fine-tuning-based (localized) | Optimize editing loss directly, but constrain trainable parameters and training pipeline | LocFT-BF, FT-M | ZsRE, CounterFact, WikiBigEdit; WILD evaluation | Still typically trained only on target-token generation |
| Memory / retrieval-based | Store edits externally and route queries | SERAC, IKE, FAME/SKEME | Fact-check, QA, multi-task benchmarks | Requires memory growth and routing; not purely parametric |
| Semantic integration objectives | Add representation-level alignment or consistency losses | STEAM, SAUL | CounterFactPlus, portability/consistency metrics | May require extra resources (anchors, generation regularizers) |

### Closest Prior Work

1. **Built-on-Sand**: Defines the fact-checking style probe and shows large efficacy–accuracy gaps, but does not propose a training objective to fix the mismatch.
2. **LocFT-BF**: Strong fine-tuning-based editor, but does not measure truth-judgment performance and does not include truth-conditional supervision.
3. **STEAM / semantic-alignment editors**: Improve semantic integration for portability/consistency, but do not directly target a balanced True/False objective for statement-judging probes.

**Novelty Kill Search Summary:** We searched for prior work combining parametric editing with explicit truth-conditional or True/False supervision, including queries such as “knowledge editing true false loss”, “model editing fact-checking style evaluation training objective”, “truth-conditional auxiliary loss model editing”, “knowledge editing entailment objective”, and “semantic consistency loss for model editing” (plus OpenReview/GitHub searches). As of 2026-02-17, we did not find a direct method that adds a balanced True/False objective during localized fine-tuning to target the Built-on-Sand fact-check probe.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Built-on-Sand | Proposes negation + fact-check probes for editing | Diagnostic only | Add training-time objective targeting the probe | Directly aligns gradients with the evaluation format |
| LocFT-BF | Strong localized BF fine-tuning editor | Trained only for target-token generation | Add balanced fact-check supervision | Transfers edits to truth-judgment head/tokens |
| STEAM | Latent semantic alignment for edits | Requires semantic anchors; not aimed at judge prompts | Use label-only objective from existing edit fields | Simpler, fully automatable, no external KB |
| SAUL | Generation regularization for consistency | Focuses on fluency/consistency of generation | Train a truth-judgment behavior directly | Targets the specific mismatch (generation vs truth-judgment) |

---

## Experiments

### Experimental Setup

**Core experiment (3 training conditions):**
1. **LocFT-BF** (baseline)
2. **LocFT-BF + format-only fact-check loss** (control): train only FC-Pos with label=True
3. **LocFT-BF + Fact-Check Grounding (ours)**: train FC-Pos (True) + FC-Neg (False)

**Prompting / inference baselines (no training; sanity checks):**
- **Unedited model**: evaluate BFC-Acc without applying any edits.
- **Non-edited facts control**: evaluate BFC-Acc on a held-out set of (prompt, ground_truth_old, target_new) triples that were **not edited**, before vs. after editing. If BFC-Acc improves only on edited facts but not on non-edited facts, it suggests FCG teaches a targeted behavior rather than a general fact-checking skill.
- **Prompted edit injection**: prepend the edited fact as context (e.g., `Assume: {p} {o_new}. Now answer True/False for: ...`) and optionally use best-of-8 majority vote to test whether inference-time scaling can solve the task without parametric editing.

**Baseline Ladder (REQUIRED):**
- No-edit + prompting baselines (sanity)
- LocFT-BF (closest strong parametric baseline)
- LocFT-BF + format-only control (tests shortcut)
- LocFT-BF + FCG (ours)

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Same family as Built-on-Sand experiments |

**Training Data (editing set):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| KnowEdit / ZsRE | Edit instances with `prompt`, `ground_truth`, `target_new` | 2,000 edits (subsample) | https://huggingface.co/datasets/zjunlp/KnowEdit | Check dataset card |

**Resource Estimate** (conservative):
- **Compute budget**: 
  - Per training run (one condition, one seed): ~8×A100 for ~2 hours ≈ **16 GPU-hours** (fine-tuning only one module; estimate based on typical 7B PEFT-style fine-tuning, since LocFT-BF hyperparameters are not fully available in the scraped PDF).
  - Total (3 conditions × 3 seeds): ≈ **144 GPU-hours**.
  - Evaluation/inference overhead: ≤ 20 GPU-hours.
  - **Total expected**: **≤ 200 GPU-hours**.
- **GPU memory**: 1×A100 80GB should be sufficient with BF16 + activation checkpointing; multi-GPU is for speed.
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| KnowEdit / ZsRE | Single-hop factual edits with old/new answers | Efficacy (EM), FC-Acc, BFC-Acc, BFC-Acc (paraphrased judge) | 2k edit subset (train) + held-out subset (test) | https://huggingface.co/datasets/zjunlp/KnowEdit | Custom (simple generation + string match) |

**Metric definitions**:
- **Efficacy (EM)**: exact match rate of generating `o_new` given the original prompt `p` (higher is better).
- **FC-Acc**: accuracy on Built-on-Sand’s fact-check probe on FC-Pos only (expects “True”).
- **BFC-Acc** (primary): mean accuracy over FC-Pos (True) and FC-Neg (False). Random-chance is 50%.
- **BFC-Acc (paraphrase)** (primary robustness): BFC-Acc under a held-out judge-template paraphrase.

**Evaluation scripts:**
- Use greedy decoding (temperature=0) and parse the first non-whitespace token as the label.
- Enforce a constrained output format in prompts: “Answer with only ‘True’ or ‘False’.”

**Download Links Checklist:**
- [x] All benchmark datasets have download links
- [x] All models have download links
- [ ] Licenses are compatible with research use (verify)

### Main Results

#### Results Table

(Cells marked TBD require running the experiments in this proposal.)

| Method | Base Model | Benchmark | Efficacy EM ↑ | FC-Acc ↑ | BFC-Acc ↑ | BFC-Acc (paraphrase) ↑ | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| MEMIT | Qwen2.5-7B-Instruct | ZsRE | 96.5 | 48.8 | N/A | N/A | Built-on-Sand Table 4 (`./references/Is-Model-Editing-Built-on-Sand-Revealing-Its-Illusory-Success-and-Fragile-Foundation/sections/5.3 Fact-checking style evaluation.md`) | Published (1 run); does not report balanced negatives |
| PMET | Qwen2.5-7B-Instruct | ZsRE | 88.4 | 52.8 | N/A | N/A | Built-on-Sand Table 4 (`./references/Is-Model-Editing-Built-on-Sand-Revealing-Its-Illusory-Success-and-Fragile-Foundation/sections/5.3 Fact-checking style evaluation.md`) | Published (1 run) |
| AdaEdit | Qwen2.5-7B-Instruct | ZsRE | 87.5 | 55.4 | N/A | N/A | Built-on-Sand Table 4 (`./references/Is-Model-Editing-Built-on-Sand-Revealing-Its-Illusory-Success-and-Fragile-Foundation/sections/5.3 Fact-checking style evaluation.md`) | Published (1 run); best FC-Acc among shown methods |
| Unedited (no edit) | Qwen2.5-7B-Instruct | ZsRE | 0.0 | TBD | TBD | TBD | This work | Sanity check |
| Prompted edit injection (best-of-8) | Qwen2.5-7B-Instruct | ZsRE | 0.0 | TBD | TBD | TBD | This work | Checks if prompting/inference scaling suffices |
| LocFT-BF | Qwen2.5-7B-Instruct | ZsRE | TBD | TBD | TBD | TBD | This work | Strong parametric baseline; not reported in Built-on-Sand |
| LocFT-BF + format-only FC | Qwen2.5-7B-Instruct | ZsRE | TBD | TBD | TBD | TBD | This work | Control for “always True” shortcut |
| **LocFT-BF + Fact-Check Grounding (ours)** | Qwen2.5-7B-Instruct | ZsRE | TBD | TBD | TBD | TBD | This work | Paired FC-Pos/FC-Neg auxiliary loss |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| LocFT-BF + format-only FC | Train only FC-Pos (True) | FC-Acc may rise, but BFC-Acc stays low if shortcut dominates |
| λ = 0 | Remove fact-check auxiliary loss | Reduces to LocFT-BF baseline |

### Experimental Rigor

- **Seeds**: 3 seeds for each training condition (e.g., `seeds=[1,2,3]`), report mean±std.
- **Confounders / controls**:
  - **Always-True shortcut**: detected by BFC-Acc and the format-only control.
  - **Template memorization**: detected by paraphrased-judge BFC-Acc.
  - **Over-editing**: monitor efficacy EM and (optionally) locality prompts provided in KnowEdit.

---

## Success Criteria

**Hypothesis**: Compared to LocFT-BF, Fact-Check Grounding improves balanced truth-judgment (BFC-Acc) and transfers to a paraphrased judge template, while preserving token-level efficacy.

**Decision Rule**:
- **Proceed/Continue** if:
  - (i) BFC-Acc improves by **≥10 absolute points** over LocFT-BF **and** over the format-only control, and
  - (ii) BFC-Acc (paraphrase) improves by **≥10 points** over LocFT-BF, and
  - (iii) efficacy EM drop is **≤1 point** relative to LocFT-BF.
- **Pivot** if FC-Acc improves but BFC-Acc does not (suggests shortcut learning); try alternative labels (e.g., “Yes/No”) or stricter output constraints.
- **Refute** if BFC-Acc gains are <10 points or vanish under paraphrased judge prompts across seeds.

---

## Impact Statement

If successful, this work provides a minimal, easily adoptable objective change that makes parametric model editing more semantically trustworthy under truth-judgment probes. Practitioners who rely on edits as persistent updates (rather than prompt-time context injection) could use fact-check grounding as a default auxiliary objective and evaluation to avoid “illusory” edit success that does not transfer beyond one prompt format.

---

## References

- [Is Model Editing Built on Sand? Revealing Its Illusory Success and Fragile Foundation](./references/Is-Model-Editing-Built-on-Sand-Revealing-Its-Illusory-Success-and-Fragile-Foundation/meta/meta_info.txt) - Liu et al., 2025
- [FINE-TUNING DONE Right IN MODEL EDITING](./references/FINE-TUNING-DONE-Right-IN-MODEL-EDITING/meta/meta_info.txt) - Yang et al., 2025
- [Memory-Based Model Editing at Scale](./references/Memory-Based-Model-Editing-at-Scale/meta/meta_info.txt) - Mitchell et al., 2022
- [FAME: Towards Factual Multi-Task Model Editing](./references/FAME-Towards-Factual-Multi-Task-Model-Editing/meta/meta_info.txt) - Zeng et al., 2024
- [STEAM: A Semantic-Level Knowledge Editing Framework for Large Language Models](./references/Steam-A-Semantic-Level-Knowledge-Editing-Framework-for-Large-Language-Models/meta/meta_info.txt) - Jeong et al., 2025
- [The Mirage of Model Editing: Revisiting Evaluation in the Wild](https://arxiv.org/abs/2502.11177) - Yang et al., 2025
- [Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262) - Meng et al., 2022
- [Mass-Editing Memory in a Transformer (MEMIT)](https://openreview.net/forum?id=MkbcAHIYgyS) - Meng et al., 2023
- [A Unified Framework for Model Editing (EMMET)](https://arxiv.org/abs/2403.14236) - Gupta et al., 2024
- [PMET: Precise Model Editing in a Transformer](https://arxiv.org/abs/2308.08742) - Li et al., 2023/2024
- [Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue (RECT)](https://arxiv.org/abs/2401.04700) - Gu et al., 2024
- [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/abs/2410.02355) - Fang et al., 2025
- [AdaEdit: Advancing Continuous Knowledge Editing for Large Language Models](https://arxiv.org/abs/2507.02408) - Li & Chu, 2025
- [Perturbation-Restrained Sequential Model Editing (PRUNE)](https://openreview.net/forum?id=bfI8cp8qmk) - Ma et al., 2025
- [Context-Robust Knowledge Editing for Language Models (CoRE)](https://arxiv.org/abs/2505.23026) - Park et al., 2025
- [Better Call SAUL: Fluent and Consistent Language Model Editing with Generation Regularization](https://arxiv.org/abs/2410.02433) - Cao et al., 2024
- [Propagation and Pitfalls: Reasoning-based Assessment of Knowledge Editing](https://arxiv.org/abs/2401.17585) - 2024
- [MQuAKE: Assessing Knowledge Editing in Language Models via Multi-hop Questions](https://arxiv.org/abs/2310.05845) - Zhong et al., 2023
- [Keys to Robust Edits: from Theoretical Insights to Practical Advances](https://arxiv.org/abs/2410.09338) - 2024
- [Can Knowledge Editing Really Correct Hallucinations? (HalluEditBench)](https://arxiv.org/abs/2410.16251) - Huang et al., 2024
- [Editing Large Language Models Poses Serious Safety Risks](https://arxiv.org/abs/2502.02958) - 2025
