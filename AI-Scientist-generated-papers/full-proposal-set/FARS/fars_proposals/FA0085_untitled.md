# untitled

# Tool-Gated Residual Distillation for DataChef Verifier Scoring

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Modern large language models (LLMs) are often adapted to a target domain (e.g., finance or programming) by supervised fine-tuning (SFT) on an instruction-following dataset. In practice, a major bottleneck is the *data engineering loop*: selecting raw sources, transforming them into a consistent chat format, filtering low-quality examples, and mixing sources.

**DataChef** frames parts of this loop as an optimization problem: given a target benchmark and a set of candidate raw datasets, generate an executable “data recipe” (a data processing plan plus Python code that produces a fine-tuning dataset), and optimize recipe generation with reinforcement learning using a proxy reward called the **Data Verifier Score (DVS; “Data Verifier” in the paper)** **[DataChef](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/meta/meta_info.txt)**. DataChef’s verifier is a strong LLM prompted with a rubric: it assigns each *training instance* to a category and maps it to a scalar score (0 / 0.4 / 1.0), then averages scores over a sampled subset of instances (with additional penalties for recipe execution/format failures) **[DataChef](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/End-to-end%20Data%20Recipe%20Generation.md)**.

DataChef validates that DVS is moderately predictive of downstream model performance: it reports an average Pearson correlation of **0.59** across six held-out tasks between verifier score and the **Downstream Benchmark Score (DBS; downstream accuracy/pass@1 after fine-tuning)** **[Data Verifier](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/Data%20Verifier.md)**.

However, DataChef’s verifier is expensive: the released config recommends a very large judge model (**`gpt-oss-120b`**) **[DataChef GitHub](./references/GitHub-yichengchen24-DataChef/sections/README.md.md)**. This cost matters because the verifier is used repeatedly inside recipe search (many candidate datasets × many sampled instances per dataset).

### The Problem

For DataChef-style recipe/dataset selection, the verifier is not merely an evaluation tool; it is the *objective* used to rank candidate datasets. A practical verifier should satisfy two requirements simultaneously:

1. **Ranking fidelity**: it should preserve the ordering of candidate datasets by downstream DBS.
2. **Low marginal cost**: it should be cheap enough that practitioners can score many instances per candidate dataset (larger sampling size **m**) and many candidate datasets per iteration.

A straightforward engineering idea is to run deterministic checks before calling an LLM judge. But DataChef’s rubric includes categories that *look* deterministically checkable (e.g., severe repetition / malformed structure), while other categories are not (e.g., “Task Mismatch” semantic relevance). It is unclear whether deterministic checks cover enough of the low-score mass to materially reduce cost **without** harming ranking.

### Key Insight and Hypothesis

**Key insight (mechanism):** DataChef’s 5-category rubric has an implicit *hierarchy*:

- Some zero-score categories correspond to **low-level syntactic/structural failures** (e.g., empty fields, malformed chat structure, severe repetition) that can be detected with cheap deterministic checks.
- The remaining categories (“Incorrect”, “Task Mismatch”, “Pass”) require semantic judgment, but they operate on a *cleaner residual distribution* once obvious failures are removed.

This suggests a non-trivial interaction between **tool gating (A)** and **distillation (B)**: tool gating does not only save LLM calls at inference, it also **factorizes the label space** and reduces class imbalance/label entropy for the student, potentially enabling a much smaller model to preserve ranking with fewer teacher labels.

**Hypothesis:** A **tool-gated, residual-distilled verifier** can match the dataset-ranking quality of a strong LLM rubric judge while being much cheaper per scored instance, enabling larger sampling size *m* under a fixed budget and improving top-1 dataset selection.

We could be wrong if (i) tool-detectable failures are rare (<20% of instances) or misaligned with the rubric (low precision), (ii) the residual “Incorrect vs Task Mismatch vs Pass” judgments require frontier-level reasoning that a small model cannot learn reliably, or (iii) DBS variance across candidate datasets is too small for better scoring to translate into better selection.

---

## Proposed Approach

### Overview

We evaluate a minimal 3-way verifier ladder for DataChef-style dataset scoring:

1) **LLM-only DVS (baseline)**: A strong LLM judge prompted with the DataChef rubric for every sampled instance.

2) **Tool+LLM**: Apply deterministic gating checks; if a check fires, assign score 0; otherwise call the same strong LLM rubric judge.

3) **Tool+Distilled (ours)**: Apply the same deterministic gating; if it passes, use a distilled small model to predict the rubric score on the residual.

The primary scientific comparison is **(2) vs (3)**: whether distillation preserves ranking on the residual subset.

### Method Details

#### 1) Deterministic tool gating

DataChef datasets are represented as chat-style instruction-response pairs (often called “ShareGPT format”, i.e., a JSON record containing a list of conversational messages with `role` and `content` fields): a JSON object containing a list of messages with roles (user/assistant) and text content.

For each sampled training instance, we run cheap checks that are intended to align with DataChef’s rubric definitions for **Invalid (0)** and **Format Error (0)** **[DataChef](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/End-to-end%20Data%20Recipe%20Generation.md)**:

- **Schema/structure validity**: parseable JSON; required message keys exist; roles are in {user, assistant}; assistant content is a string.
- **Empty/degenerate response**: empty assistant content after stripping; extremely short responses (e.g., <5 tokens).
- **Severe repetition**: n-gram repetition ratio above a threshold or repeating substrings above a threshold.

If any gate fires, set instance score to **0** *without* calling an LLM.

#### 2) Teacher rubric judge (for baselines and distillation labels)

Because DataChef’s recommended `gpt-oss-120b` is not a publicly documented model, we specify a reproducible teacher available in our environment: **`gpt-4.1`** (or **`gemini-3-pro`** as a swap-in) **[available models](../../internal_context/available_models.md)**.

The teacher is used in two places:

- **Baseline scoring** in (1) and (2)
- **Labeling residual instances** for distillation (only when the tool gate passes)

The teacher outputs a rubric category and a scalar score in {0, 0.4, 1.0} consistent with DataChef.

#### 3) Residual distillation (student model)

**Student model:** Qwen2.5-1.5B-Instruct (LoRA fine-tune; LoRA = Low-Rank Adaptation, a parameter-efficient fine-tuning method) (download: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

**Training data construction:**

- Collect instances from DataChef-generated datasets on **training tasks only** (exclude all evaluation benchmarks: PHYSICS, AIME’25, LiveCodeBench v6, OpenFinData, ClimaQA, CHID) **[Task pool](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/Task%20Pool%20Construction.md)**.
- Run deterministic gating.
- For **gate-fail** instances: add them as automatically labeled negatives (score 0) with no teacher call.
- For **gate-pass** instances: query the teacher for the rubric score and use it as the distillation label.

**Objective:** 3-way classification (score ∈ {0, 0.4, 1.0}) with cross-entropy; optionally report a regression variant (MSE on scalar score) as an ablation if needed.

#### 4) Dataset-level score

For each candidate dataset, sample **m** instances and compute the dataset score as the mean instance score. All verifiers score the *same sampled instances* in the cost-unmatched comparison.

### Key Innovations

1. **Rubric factorization for distillation**: Use deterministic gates to isolate rubric categories with checkable structure/repetition failure modes, and distill only the residual semantic judgments. The hypothesis is that this factorization improves distillation sample-efficiency and preserves dataset-level ranking with a much smaller student.

2. **Deterministic negatives reduce teacher labeling**: Gate-fail instances provide free negative labels, reducing the number of expensive teacher calls required to train a usable verifier.

3. **Decision-relevant evaluation**: Evaluate verifiers by (i) rank correlation to downstream DBS and (ii) top-1 dataset selection under a fixed verifier budget, which is the practical knob for data recipe search.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) **data-centric post-training / dataset value estimation**, (ii) **LLM-as-a-judge / reward modeling**, and (iii) **tool-augmented verification and judge distillation**.

DataChef is part of a broader trend toward automating dataset and post-training recipe optimization (e.g., standardized dataset value evaluation pipelines) **[OpenDataArena](./references/OpenDataArena-A-Fair-and-Open-Arena-for-Benchmarking-Post-Training-Dataset-Value/meta/meta_info.txt)**. Separately, LLM judges are known to exhibit biases and variance (position/verbosity/model-name effects), motivating tool grounding, calibration, and specialized judge models.

### Related Papers

- **[DataChef](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/meta/meta_info.txt)**: Uses an LLM rubric judge as a proxy reward to optimize data recipes.
- **[DataChef GitHub](./references/GitHub-yichengchen24-DataChef/sections/README.md.md)**: Released pipeline; recommends a very large verifier model.
- **[Rubrics as Rewards](./references/Rubrics-as-Rewards-Reinforcement-Learning-Beyond-Verifiable-Domains/meta/meta_info.txt)**: Formalizes rubric/checklist rewards and uses GRPO (Group Relative Policy Optimization, an on-policy RL method similar to PPO) with rubric-guided judges.
- **[Checklists Are Better Than Reward Models](./references/Checklists-Are-Better-Than-Reward-Models-For-Aligning-Language-Models/meta/meta_info.txt)**: Shows structured checklists + programmatic checks can outperform learned reward models.
- **[Chasing the Tail](./references/Chasing-the-Tail-Effective-Rubric-based-Reward-Modeling-for-Large-Language-Model-Post-Training/meta/meta_info.txt)**: Studies rubric-based reward modeling; emphasizes accurate ranking in the high-reward region.
- **[Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?](./references/Can-External-Validation-Tools-Improve-Annotation-Quality-for-LLM-as-a-Judge/meta/meta_info.txt)**: Tool-augmented judge agents improve preference labeling quality on verifiable subsets.
- **[CoSineVerifier](./references/CoSineVerifier-Tool-Augmented-Answer-Verification-for-Computation-Oriented-Scientific-Questions/meta/meta_info.txt)**: Tool-augmented verification improves correctness judgments on computation-heavy scientific questions.
- **[xVerify](./references/xVerify-Efficient-Answer-Verifier-for-Reasoning-Model-Evaluations/meta/meta_info.txt)**: Trains small answer verifiers for objective correctness; DataChef already uses xVerify-9B as an evaluator on some benchmarks.
- **[ToolRM](./references/ToolRM-Outcome-Reward-Models-for-Tool-Calling-Large-Language-Models/meta/meta_info.txt)**: Outcome reward models specialized to tool-calling; supports domain-specific lightweight reward modeling.
- **[TIR-Judge](./references/Incentivizing-Agentic-Reasoning-in-LLM-Judges-via-Tool-Integrated-Reinforcement-Learning/meta/meta_info.txt)**: Tool-integrated judge training and distillation; shows small tool-aware judges can approach frontier judges.
- **[JudgeLM](./references/JudgeLM-Fine-tuned-Large-Language-Models-are-Scalable-Judges/meta/meta_info.txt)**: Fine-tuned open judge models as scalable alternatives to API judges.
- **[Calibrating LLM Judges](./references/Calibrating-LLM-Judges-Linear-Probes-for-Fast-and-Reliable-Uncertainty-Estimation/meta/meta_info.txt)**: Uncertainty estimation for faster/more reliable judge inference.
- **[Uncertainty-aware Reward Model](./references/Uncertainty-aware-Reward-Model-Teaching-Reward-Models-to-Know-What-is-Unknown/meta/meta_info.txt)**: Reward models that abstain / estimate uncertainty; relevant to selective fallback to strong judges.
- **[Multi-Agent Verification](./references/Multi-Agent-Verification-Scaling-Test-Time-Compute-with-Multiple-Verifiers/meta/meta_info.txt)**: Uses multiple verifiers at test time; an alternative to distilling a single verifier.
- **[OpenDataArena](./references/OpenDataArena-A-Fair-and-Open-Arena-for-Benchmarking-Post-Training-Dataset-Value/meta/meta_info.txt)**: Standardizes dataset value evaluation via controlled fine-tuning pipelines.
- **[A Survey on Agent-as-a-Judge](./references/A-Survey-on-Agent-as-a-Judge/meta/meta_info.txt)**: Survey of LLM judge systems (agentic, tool-augmented, calibrated).

Additional background (not scraped locally):

- **[InstructGPT](https://arxiv.org/abs/2203.02155)**: Canonical RLHF pipeline motivating judge/reward model design.
- **[DPO](https://arxiv.org/abs/2305.18290)**: Direct preference optimization; motivates learning from preference/judge signals without full RL.
- **[MT-Bench / Chatbot Arena](https://arxiv.org/abs/2306.05685)**: Standard reference on LLM-as-a-judge biases and validation.
- **[G-Eval](https://arxiv.org/abs/2303.16634)**: Structured LLM-based evaluation with chain-of-thought; relevant to rubric prompting.
- **[RewardBench](https://arxiv.org/abs/2403.13787)**: Benchmark for reward models; motivates comparing against strong open RMs.
- **[Skywork-Reward-V2](https://arxiv.org/abs/2507.01352)**: Strong open reward models trained on large curated preference datasets.
- **[DEITA](https://arxiv.org/abs/2312.15685)**: Data selection via complexity×quality with diversity filtering; related to alternative non-judge scoring for data quality.
- **[IFD / From Quantity to Quality](https://arxiv.org/abs/2308.12032)**: Self-guided instruction data selection metric based on instruction-following difficulty.
- **[Vendi Score](https://arxiv.org/abs/2210.02410)**: Diversity metric used as a baseline in DataChef’s correlation analysis.
- **[Prometheus-2](https://arxiv.org/abs/2405.01535)**: Open judge model; relevant “use a small judge directly” baseline alternative.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Evaluation focus | Known limitations |
|---|---|---|---|---|
| Rubric/checklist judges | Prompt an LLM with structured criteria | DataChef; Rubrics as Rewards; Checklists | Correlation to downstream / preference agreement | Expensive; rubric adherence imperfect |
| Tool-augmented judges/verifiers | Use external tools (exec/checkers) for grounded judgments | External Validation Tools; CoSineVerifier; TIR-Judge | Preference accuracy / verifier benchmarks | Tool coverage domain-dependent |
| Distilled / fine-tuned judges | Train smaller judges to approximate strong judges | JudgeLM; TIR-Judge; xVerify | JudgeBench/RewardBench-like suites | Distribution shift / teacher bias |
| Data selection metrics | Score training data without rubric judges | DEITA; IFD; Vendi Score | Downstream SFT quality | Weak correlation across domains |
| Cost-aware verification | Route/abstain based on uncertainty | Calibrating LLM Judges; Uncertainty-aware RM | Calibration + selective evaluation | Thresholding is brittle |

### Closest Prior Work

- **DataChef** uses an LLM rubric judge for data quality as an RL reward, and includes a coarse recipe-level penalty for format violations, but it does not attempt to (i) replace instance-level judging with deterministic gates where possible or (ii) distill the residual rubric scoring into a small model.
- **TIR-Judge / JudgeLM** demonstrate judge distillation broadly, but do not study *dataset/recipe scoring as an objective* for downstream fine-tuning.
- **xVerify** focuses on objective answer verification with ground truth signals (and is used by DataChef as an evaluator), whereas DataChef’s data verifier scores *training data quality* and includes semantic “Task Mismatch” judgments that are not answer-checking.

**Novelty Kill Search Summary:** Searched for “DataChef verifier distillation”, “DataChef Data Verifier replacement”, “tool-gated distilled judge”, “deterministic gating + judge distillation”, and “rubric verifier distillation data selection (2025–2026)”, plus local KB checks for near-duplicate proposals. As of **2026-02-16**, no prior work was found that studies **tool-gated residual distillation** specifically for **rubric-based dataset scoring** in DataChef-style pipelines.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| DataChef | RL over data recipes using LLM rubric verifier | Instance-level judging is expensive; no residual distillation | Factorize rubric with deterministic gates + distill residual | Same ranking with cheaper marginal cost |
| TIR-Judge | Tool-integrated training/distillation for LLM judges | Not about dataset value / recipe selection | Apply distillation to dataset-scoring objective | Shows feasibility of small judges; we test decision-relevant ranking |
| JudgeLM | Fine-tuned open judges | Not specialized to data-quality rubric | Distill on DataChef rubric + residual distribution | Domain-specific student should be cheaper and better targeted |
| xVerify | Small verifiers for objective correctness | Needs answer-checking; not a data-quality rubric | Use as a non-goal prior; focus on semantic mismatch quality scoring | Data quality judgments are different from answer correctness |

---

## Experiments

### Experimental Setup

We evaluate verifiers on their ability to rank candidate training datasets by predicting downstream fine-tuning performance.

#### Teacher and student models

- **Teacher judge (LLM-only baseline)**: `gpt-4.1` (API) **[available models](../../internal_context/available_models.md)**.
- **Student (distilled)**: Qwen2.5-1.5B-Instruct (LoRA; Low-Rank Adaptation) (https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
- **Downstream model**: Qwen3-1.7B-Base (https://huggingface.co/Qwen/Qwen3-1.7B-Base) fine-tuned on each candidate dataset.

#### Benchmarks (2 tasks)

We use two DataChef held-out tasks with public evaluation harnesses:

- **LiveCodeBench v6** (code generation; pass@1) with official evaluator (https://github.com/LiveCodeBench/LiveCodeBench) **[DataChef eval](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/B.2%20Evaluation%20Setup.md)**.
- **OpenFinData** (financial QA; average accuracy via OpenCompass evaluator) with dataset release (https://github.com/open-compass/OpenFinData) **[DataChef eval](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/B.2%20Evaluation%20Setup.md)**.

#### Candidate datasets per task

For each task, construct **K = 8** candidate datasets under a fixed data budget using DataChef’s correlation-analysis protocol (direct sampling + length-based subset selection) **[Data Verifier](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/sections/Data%20Verifier.md)**.

#### Stage 0: Gate alignment pilot (early-stop)

Before full experiments, run a pilot to validate the core premise.

- Sample 1,200 instances (2 tasks × 3 datasets × 200 instances).
- Compute:
  - **Gate coverage**: fraction of instances flagged by deterministic gates.
  - **Gate precision**: fraction of gate-flagged instances that the teacher scores as 0.

**Early-stop rule (pre-registered):**
- **Refute** if gate coverage < **20%** *or* gate precision < **90%** (gating is too weak or too error-prone to justify the approach).

### Baseline Ladder

Primary comparisons (3 verifiers):

1. **LLM-only DVS** (teacher rubric judge on every instance)
2. **Tool+LLM** (gating + teacher on residual)
3. **Tool+Distilled (ours)** (gating + student on residual)

### Benchmarks and Metrics

**Verifier ranking metrics (primary):**
- Spearman ρ and Kendall τ between verifier dataset scores and DBS across K candidate datasets per task.
- Pairwise ordering accuracy (fraction of dataset pairs whose DBS ordering is predicted correctly).

**Selection metric (decision-relevant):**
- Top-1 DBS: DBS of the dataset selected by argmax(verifier score) among K.

**Equivalence margin (pre-registered):**
- Tool+Distilled “matches” Tool+LLM if average Spearman ρ is within **0.05** of Tool+LLM.

**Cost metrics:**
- Teacher tokens consumed for scoring and for labeling distillation data.
- Amortized break-even point: number of datasets scored after which distillation training cost is recouped.

### Main Results

| Method | Verifier backbone | Benchmark | Metrics | Source | Notes |
|---|---|---|---|---|---|
| LLM-only DVS | gpt-4.1 | 2 tasks × K datasets | ρ/τ, pairwise acc, top-1 DBS | To be verified | Calls teacher for every instance |
| Tool+LLM | gpt-4.1 | 2 tasks × K datasets | ρ/τ, pairwise acc, top-1 DBS | To be verified | Deterministic gating saves some calls |
| **Tool+Distilled (ours)** | Qwen2.5-1.5B-Instruct (LoRA) | 2 tasks × K datasets | ρ/τ, pairwise acc, top-1 DBS | To be verified | No teacher calls at inference |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Distill w/o gating | Train the student on teacher labels without removing gate-fail categories | Worse or needs more labels; supports “rubric factorization helps distillation” |

### Experimental Rigor

- **Seeds**: downstream fine-tuning with 3 seeds per dataset (e.g., `seeds=[42, 123, 456]`).
- **Data leakage control**: distillation data uses only DataChef training tasks; evaluation uses only held-out tasks.
- **Fair comparison**: for ranking, all verifiers score the same sampled instances per dataset; for cost-matched selection, we fix a teacher-token budget B and allow Tool+Distilled to increase m.

### Resource Estimate

- **Downstream SFT**: 2 tasks × 8 datasets × 3 seeds = 48 fine-tunes of a ~1–2B model with LoRA. Budget 0.5–1 A100-hour each → 24–48 GPU-hours.
- **Verifier distillation**: one LoRA SFT run for Qwen2.5-1.5B on ~50k–100k teacher-labeled residual instances → ≤10 GPU-hours.
- **Total**: ≤60 GPU-hours (not counting API usage), within the 768 GPU-hour cap.

**API budget estimate (teacher = gpt-4.1):**
- Pricing reference: $3 / 1M input tokens and $12 / 1M output tokens (OpenAI pricing page).
- If each rubric judgment uses ~600 input tokens and ~50 output tokens:
  - Labeling 100k residual instances costs ≈ 60M input + 5M output tokens → ≈ **$240**.
  - Scoring 2 tasks × 8 datasets × m=128 instances costs ≈ 1.2M input + 0.1M output tokens → ≈ **$5** per full-scoring run.

---

## Success Criteria

**Hypothesis** (directional):
- Tool gating removes a meaningful fraction of low-score instances and aligns well with teacher 0-scores.
- Tool+Distilled matches Tool+LLM ranking quality within the equivalence margin while being substantially cheaper at inference.
- Under a fixed teacher-token budget, Tool+Distilled can score larger m and improve top-1 dataset selection.

**Decision Rule** (concrete):

- **Proceed** if:
  1) Gate pilot passes (coverage ≥20% and precision ≥90%), and
  2) Tool+Distilled achieves average Spearman ρ within **0.05** of Tool+LLM while reducing *teacher* inference tokens by ≥3×, and
  3) In the cost-matched setting, Tool+Distilled improves top-1 DBS by a margin exceeding the across-seed standard deviation.

- **Pivot** if Tool+LLM beats LLM-only but Tool+Distilled misses the equivalence margin → try a larger student (Qwen2.5-7B) or add uncertainty-based fallback to teacher on low-confidence residuals.

- **Refute** if the gate pilot fails *or* Tool+Distilled cannot match Tool+LLM within the equivalence margin.

---

## Impact Statement

If successful, this work makes DataChef-style data recipe search cheaper and more reproducible by replacing expensive rubric judging with a tool-gated distilled verifier. This can enable practitioners to evaluate more candidate datasets and/or score more instances per dataset under a fixed budget, improving automated dataset selection in data-centric post-training workflows.

---

## References

- [DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning](./references/DataChef-Cooking-Up-Optimal-Data-Recipes-for-LLM-Adaptation-via-Reinforcement-Learning/meta/meta_info.txt) - Chen et al., 2026
- [GitHub - yichengchen24/DataChef](./references/GitHub-yichengchen24-DataChef/sections/README.md.md) - Code repository
- [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](./references/Rubrics-as-Rewards-Reinforcement-Learning-Beyond-Verifiable-Domains/meta/meta_info.txt) - Gunjal et al., 2025
- [Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](./references/Incentivizing-Agentic-Reasoning-in-LLM-Judges-via-Tool-Integrated-Reinforcement-Learning/meta/meta_info.txt) - Xu et al., 2025
- [Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?](./references/Can-External-Validation-Tools-Improve-Annotation-Quality-for-LLM-as-a-Judge/meta/meta_info.txt) - Findeis et al., 2025
- [CoSineVerifier: Tool-Augmented Answer Verification for Computation-Oriented Scientific Questions](./references/CoSineVerifier-Tool-Augmented-Answer-Verification-for-Computation-Oriented-Scientific-Questions/meta/meta_info.txt) - Feng et al., 2024
- [xVerify: Efficient Answer Verifier for Reasoning Model Evaluations](./references/xVerify-Efficient-Answer-Verifier-for-Reasoning-Model-Evaluations/meta/meta_info.txt) - Chen et al., 2025
- [ToolRM: Outcome Reward Models for Tool-Calling Large Language Models](./references/ToolRM-Outcome-Reward-Models-for-Tool-Calling-Large-Language-Models/meta/meta_info.txt) - Agarwal et al., 2025
- [JudgeLM: Fine-tuned Large Language Models are Scalable Judges](./references/JudgeLM-Fine-tuned-Large-Language-Models-are-Scalable-Judges/meta/meta_info.txt) - Kim et al., 2024
- [Checklists Are Better Than Reward Models For Aligning Language Models](./references/Checklists-Are-Better-Than-Reward-Models-For-Aligning-Language-Models/meta/meta_info.txt) - Viswanathan et al., 2025
- [Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](./references/Chasing-the-Tail-Effective-Rubric-based-Reward-Modeling-for-Large-Language-Model-Post-Training/meta/meta_info.txt) - Zhang et al., 2025
- [Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation](./references/Calibrating-LLM-Judges-Linear-Probes-for-Fast-and-Reliable-Uncertainty-Estimation/meta/meta_info.txt) - Radharapu et al., 2024
- [Uncertainty-aware Reward Model: Teaching Reward Models to Know What is Unknown](./references/Uncertainty-aware-Reward-Model-Teaching-Reward-Models-to-Know-What-is-Unknown/meta/meta_info.txt) - Lou et al., 2024
- [Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers](./references/Multi-Agent-Verification-Scaling-Test-Time-Compute-with-Multiple-Verifiers/meta/meta_info.txt) - Lifshitz et al., 2025
- [OpenDataArena: A Fair and Open Arena for Benchmarking Post-Training Dataset Value](./references/OpenDataArena-A-Fair-and-Open-Arena-for-Benchmarking-Post-Training-Dataset-Value/meta/meta_info.txt) - Cai et al., 2024
- [A Survey on Agent-as-a-Judge](./references/A-Survey-on-Agent-as-a-Judge/meta/meta_info.txt) - Judge, 2025
- [InstructGPT](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) - Zheng et al., 2023
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) - Liu et al., 2023
- [RewardBench: Evaluating Reward Models for Language Model Alignment](https://arxiv.org/abs/2403.13787) - Lambert et al., 2024
- [Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](https://arxiv.org/abs/2507.01352) - Liu et al., 2025
- [DEITA / What Makes Good Data for Alignment?](https://arxiv.org/abs/2312.15685) - Liu et al., 2023
- [IFD / From Quantity to Quality](https://arxiv.org/abs/2308.12032) - Li et al., 2023
- [The Vendi Score](https://arxiv.org/abs/2210.02410) - Friedman & Dieng, 2022
- [Prometheus-2](https://arxiv.org/abs/2405.01535) - Kim et al., 2024
