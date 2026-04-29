# untitled

# Counterfactual-View Disagreement Escalation for Robust Web-Agent Trajectory Judges

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS (Conference on Neural Information Processing Systems), ICML (International Conference on Machine Learning), ICLR (International Conference on Learning Representations), ACL (Annual Meeting of the Association for Computational Linguistics), EMNLP (Empirical Methods in Natural Language Processing), or similar AI/ML conferences

## Introduction

### Context and Motivation

Large language model (LLM) agents that interact with websites ("web agents") produce **trajectories**: sequences of observations (page state), actions (click/type/navigation commands), and sometimes an explicit natural-language reasoning trace written by the agent.

Web-agent trajectories are increasingly evaluated automatically to (i) benchmark progress and (ii) select training data (e.g., filtering trajectories by predicted success via **rejection sampling**, meaning "keep trajectories the judge labels as successful") or generate reward signals for reinforcement learning.

A common approach is **LLM-as-a-judge**, where a separate LLM (the judge) reads a trajectory and predicts whether the agent succeeded. This scales to thousands of trajectories, but judge errors can become a limiting factor when judge outputs are used for training: false positives (failed trajectories labeled as success) inject label noise and can enable **reward hacking** (optimizing the judge score without completing the task).

**AgentRewardBench** is a benchmark of **1,302 expert-annotated web-agent trajectories** used to evaluate automatic trajectory judges. It reports that a strong simplified judge using **Llama-3.3** achieves **67.7% precision** (fraction of predicted successes that are truly successful; higher is better) and **79.0% recall** (fraction of true successes correctly predicted; higher is better) for success prediction, indicating substantial remaining error ([AgentRewardBench](./references/AgentRewardBench-Evaluating-Automatic-Evaluations-of-Web-Agent-Trajectories/meta/meta_info.txt)).

A key practical detail is that many trajectories include an agent-written **chain-of-thought (CoT)** or rationale: free-form text describing the agent’s reasoning. CoT can help a judge interpret intent, but it also creates a vulnerability because the judge may rely on persuasive narrative text that is not supported by the observable trajectory. **Gaming the Judge** shows that rewriting only the CoT (keeping actions and observations fixed) can substantially increase false positives for trajectory judges ([Gaming the Judge](./references/GAMING-THE-JUDGE-UNFAITHFUL-CHAIN-OF-THOUGHT-CAN-UNDERMINE-AGENT-EVALUATION/meta/meta_info.txt)).

### The Problem

The technical problem is to improve the reliability of trajectory success judges under two constraints:

- **High precision is required for training-time use**: precision is the fraction of predicted successes that are truly successful (higher is better). Low precision means failures are incorrectly accepted as successes, which corrupts training signals.
- **Recall should not collapse**: recall is the fraction of truly successful trajectories that are correctly predicted as successful (higher is better). Low recall discards useful trajectories and reduces sample efficiency.

Existing mitigations have limitations:

- **Always-on evidence-anchored evaluation** (e.g., **WebJudge**, a rubric-based protocol that asks the judge to ground its decision in observable evidence) can raise precision but often increases cost and can reduce recall by being overly strict ([An Illusion of Progress?](./references/An-Illusion-of-Progress-Assessing-the-Current-State-of-Web-Agents/meta/meta_info.txt)).
- **Judge-time scaling** (e.g., **self-consistency**, where the judge samples multiple independent judgments and aggregates them) can improve robustness but increases inference cost and does not fully address CoT-manipulation failures ([Gaming the Judge](./references/GAMING-THE-JUDGE-UNFAITHFUL-CHAIN-OF-THOUGHT-CAN-UNDERMINE-AGENT-EVALUATION/meta/meta_info.txt)).
- **Removing CoT from judge inputs** reduces reliance on narrative fields, but it can also remove useful information and reduce recall in practice ([Gaming the Judge](./references/GAMING-THE-JUDGE-UNFAITHFUL-CHAIN-OF-THOUGHT-CAN-UNDERMINE-AGENT-EVALUATION/meta/meta_info.txt)).

The practical goal is to obtain a better **precision–recall–cost** trade-off (where cost is judge inference compute, such as total processed tokens or number of LLM calls), without requiring new human labels.

### Key Insight and Hypothesis

We propose a training-free robustness mechanism based on **counterfactual input views** of the same trajectory:

- **View 1 (with CoT)**: the judge sees the trajectory including the agent’s reasoning text.
- **View 2 (no CoT / evidence-only)**: the judge sees the same trajectory with all agent reasoning removed, leaving only the goal, actions, and final-state evidence.

**Key insight**: many CoT-based false positives arise when a judge relies on narrative fields that are not supported by observable evidence. Therefore, disagreement between judgments from the two views should concentrate trajectories where the judge is likely to be unreliable.

**Hypothesis**: using **View1–View2 disagreement** as a trigger for selective escalation to a stricter, evidence-anchored rubric prompt will improve the precision–recall–cost frontier compared to (i) always judging with CoT, (ii) always judging without CoT, and (iii) always using a strict rubric.

**Decision rule (refute/continue)**:

- **Refute / pivot** if either:
  1. The View1–View2 disagreement rate is high on the unmodified AgentRewardBench data (e.g., >30% of trajectories) *and* does not increase on an attacked-failure subset constructed by CoT rewriting (suggesting disagreement is not enriched for CoT manipulation), or
  2. On the disagreement subset, the escalation judgment is not more accurate than both View1-only and View2-only baselines (measured against AgentRewardBench ground-truth labels).
- **Continue** if disagreement is selective (lower on unmodified trajectories, higher under CoT rewriting) and disagreement-triggered escalation improves precision on attacked failures at similar or lower average inference cost than always-on strict evaluation.

---

## Proposed Approach

### Overview

We propose a two-stage selective escalation method for web-agent trajectory success judging:

1. Run a base judge prompt on two counterfactual input views of the same trajectory (with-CoT vs no-CoT).
2. If the two judgments agree, return the agreed label.
3. If they disagree, run an evidence-anchored, rubric-style prompt (escalation) that excludes CoT and requires explicit citation of evidence from the action history and final state.

### Method Details

**Inputs (from an offline trajectory dataset):**

- Task/goal description.
- Action history (the sequence of actions taken by the agent).
- Final-state evidence (e.g., the final accessibility tree, a text representation of the webpage structure used by assistive technologies).
- Agent reasoning trace (chain-of-thought / rationale), if present.

**Counterfactual views:**

- **View1 (with CoT)**: goal + action history + final-state evidence + all agent reasoning fields.
- **View2 (no CoT / evidence-only)**: goal + action history + final-state evidence, with all agent reasoning fields removed.

**Stage 1: dual-view judging**

- Run a single judge model \(J\) with a fixed base prompt \(P_\text{base}\) on View1 and parse a binary prediction \(y_1 \in \{\text{success}, \text{failure}\}\).
- Run the same \(J\) with \(P_\text{base}\) on View2 to get \(y_2\).

**Stage 2: disagreement-triggered escalation**

- If \(y_1 = y_2\), output \(y = y_1\).
- Otherwise, run \(J\) with a strict prompt \(P_\text{strict}\) that:
  - excludes CoT and any agent-written narrative,
  - uses an explicit checklist of task requirements,
  - requires the judge to cite concrete evidence from actions and/or final-state artifacts,
  - outputs a machine-parseable JSON (JavaScript Object Notation) object (e.g., `{"success": true}` or `{"success": false, "evidence": ["..."]}`),
  - returns \(y_\text{strict}\), and we output \(y = y_\text{strict}\).

**Prompt specification (implementation guidance):**

- Use the AgentRewardBench simplified-judge prompt as \(P_\text{base}\) to ensure comparability to published baselines ([AgentRewardBench](./references/AgentRewardBench-Evaluating-Automatic-Evaluations-of-Web-Agent-Trajectories/meta/meta_info.txt)).
- Implement \(P_\text{strict}\) by adapting a WebJudge-style rubric that forces evidence grounding (e.g., “List the observable evidence that the goal is satisfied; if evidence is missing, predict failure”) ([An Illusion of Progress?](./references/An-Illusion-of-Progress-Assessing-the-Current-State-of-Web-Agents/meta/meta_info.txt)).

**Decoding and stability**:

- Use deterministic decoding (`temperature=0`) for View1, View2, and escalation to reduce disagreement caused by sampling noise.

**Why same-model escalation?**

- We use the same judge model for all stages to isolate the effect of view disagreement and prompt strictness from model-scaling confounds.

### Key Innovations

1. **Counterfactual input ablation as a robustness signal**: View1 (with CoT) vs View2 (no CoT) disagreement directly probes judge reliance on narrative fields.
2. **Selective escalation for trajectory evaluation**: apply evidence-anchored rubric prompts only on disagreement cases.
3. **Attack-aligned evaluation**: evaluate robustness under automated CoT rewriting attacks where only the agent’s narrative changes.

---

## Related Work

### Field Overview

LLM-as-a-judge evaluation was originally developed to approximate human preference judgments for open-ended dialogue, and it has since expanded to judging summaries, safety-related content, and agent trajectories. Web-agent trajectory judging is especially challenging because success is rarely a simple string match, and the judge must interpret long sequences of actions and state representations.

Recent work has identified multiple failure modes in LLM judges, including susceptibility to prompt injection, adversarial triggers, and manipulation of narrative fields. Evidence-anchored scoring methods (rubrics, extractive evidence requirements, and structured checklists) can mitigate some of these failures, but they can reduce recall and increase inference cost.

Selective evaluation methods (abstention or escalation) can reduce cost by applying expensive evaluation only when needed. However, many existing approaches use generic uncertainty signals (confidence calibration or cross-model disagreement) rather than a signal directly aligned to CoT-manipulation vulnerabilities.

### Related Papers

- **[AgentRewardBench](./references/AgentRewardBench-Evaluating-Automatic-Evaluations-of-Web-Agent-Trajectories/meta/meta_info.txt)**: Introduces a 1,302-trajectory benchmark and standardized metrics for evaluating web-agent trajectory judges, which we use as our primary evaluation setting.
- **[Gaming the Judge](./references/GAMING-THE-JUDGE-UNFAITHFUL-CHAIN-OF-THOUGHT-CAN-UNDERMINE-AGENT-EVALUATION/meta/meta_info.txt)**: Demonstrates that rewriting only agent CoT can inflate false positives in trajectory judges and motivates using CoT-sensitive robustness tests.
- **[An Illusion of Progress? (WebJudge)](./references/An-Illusion-of-Progress-Assessing-the-Current-State-of-Web-Agents/meta/meta_info.txt)**: Proposes rubric-based, evidence-anchored evaluation for web-agent trajectories, which informs our strict escalation prompt.
- **[DAFE/CLEV](./references/DAFE-LLM-Based-Evaluation-Through-Dynamic-Arbitration-for-Free-Form-Question-Answering/meta/meta_info.txt)**: Proposes dynamic arbitration (an escalation mechanism) triggered by judge disagreement to reduce evaluation cost in free-form question answering, providing precedent for escalation-style evaluation.
- **[Trust or Escalate](./references/Trust-or-Escalate-LLM-Judges-with-Provable-Guarantees-for-Human-Agreement/meta/meta_info.txt)**: Studies confidence calibration and judge cascades to improve agreement with humans, relevant as an alternative escalation signal.
- **[RULERS](./references/RULERS-Locked-Rubrics-and-Evidence-Anchored-Scoring-for-Robust-LLM-Evaluation/meta/meta_info.txt)**: Uses locked rubrics and extractive evidence requirements to improve evaluation robustness, aligning with our evidence-anchored escalation design.
- **[Auto-Prompt Ensemble (APE)](./references/Auto-Prompt-Ensemble-for-LLM-Judge/meta/meta_info.txt)**: Uses prompt ensembles and aggregation to improve judge reliability, relevant as a competing inference-time robustness baseline.
- **[One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)**: Shows that short adversarial triggers can bias LLM judges, motivating defenses that reduce reliance on ungrounded text fields.
- **[CalibraEval](https://arxiv.org/abs/2410.15393)**: Calibrates judge score distributions to mitigate selection and position biases without additional labels, relevant as a label-free reliability approach.
- **[TrustJudge](https://arxiv.org/abs/2509.21117)**: Improves judge consistency using likelihood-aware aggregation and transitivity-style constraints, relevant as an alternative aggregation-based robustness method.
- **[CAP: Comparative Augmented Prompting](https://openreview.net/forum?id=wYU6OYFvid)**: Improves robustness of LLM judges to score manipulation using comparative prompting, relevant as a prompt-only defense baseline.
- **[Optimization-based Prompt Injection Attack to LLM-as-a-Judge (JudgeDeceiver)](https://arxiv.org/abs/2403.17710)**: Demonstrates optimization-based prompt injection attacks on LLM-as-a-judge systems, providing a broader view of judge vulnerabilities.
- **[Investigating Vulnerabilities of LLM-as-a-Judge to Prompt Injection](https://arxiv.org/abs/2505.13348)**: Systematically studies prompt-injection and adversarial suffix attacks on LLM judges, highlighting robustness requirements for automated evaluation.
- **[MT-Bench](https://arxiv.org/abs/2306.05685)**: Popularized LLM-as-a-judge for chat evaluation and provides historical context for judge-based benchmarking.
- **[Chatbot Arena](https://arxiv.org/abs/2403.04132)**: Demonstrates large-scale preference-based evaluation using LLM judgments, motivating scalable judge pipelines.
- **[RewardBench](https://arxiv.org/abs/2403.13787)**: Benchmarks reward models and judges for preference scoring, relevant as complementary evidence about judge reliability.
- **[WebArena](https://arxiv.org/abs/2307.13854)**: A sandbox web-agent benchmark; AgentRewardBench draws tasks from WebArena environments.
- **[VisualWebArena](https://arxiv.org/abs/2401.03564)**: A multimodal web-agent benchmark used within AgentRewardBench, relevant because judge inputs may include screenshots.
- **[AssistantBench](https://arxiv.org/abs/2402.17573)**: A web-assistant benchmark included in AgentRewardBench, broadening the evaluated task distribution.
- **[WorkArena](https://arxiv.org/abs/2403.07718)**: A benchmark of enterprise-style web tasks (ServiceNow) used in AgentRewardBench, representing realistic multi-step workflows.
- **[WorkArena++](https://arxiv.org/abs/2407.05291)**: Extends WorkArena with more compositional tasks, increasing evaluation difficulty for trajectory judges.
- **[AER: Autonomous Evaluation and Refinement of Digital Agents](https://arxiv.org/abs/2404.06474)**: Uses LLM-based trajectory judgments for self-improvement loops, illustrating why judge precision matters for training.
- **[NNetNav](https://arxiv.org/abs/2410.02907)**: Learns trajectory representations for browser agents and includes an LLM-based trajectory rating component, relevant as a representative trajectory-judge design.
- **[Mind2Web](https://arxiv.org/abs/2306.06070)**: A foundational dataset for web navigation, providing historical context for web-agent trajectory datasets.
- **[Mind2Web-Live (WebCanvas)](https://arxiv.org/abs/2406.12373)**: An online web-agent benchmark with intermediate-state evaluation, relevant as a downstream setting where robust judging could be applied.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Trajectory judges (web agents) | Use an LLM to label task success from trajectory artifacts | [AgentRewardBench](./references/AgentRewardBench-Evaluating-Automatic-Evaluations-of-Web-Agent-Trajectories/meta/meta_info.txt), [AER](https://arxiv.org/abs/2404.06474), [NNetNav](https://arxiv.org/abs/2410.02907), [WebJudge](./references/An-Illusion-of-Progress-Assessing-the-Current-State-of-Web-Agents/meta/meta_info.txt) | AgentRewardBench; WebArena-derived suites | Low precision; long-context brittleness; vulnerable to narrative manipulation |
| Robustness attacks on judges | Manipulate judge inputs to change verdict/score | [Gaming the Judge](./references/GAMING-THE-JUDGE-UNFAITHFUL-CHAIN-OF-THOUGHT-CAN-UNDERMINE-AGENT-EVALUATION/meta/meta_info.txt), [JudgeDeceiver](https://arxiv.org/abs/2403.17710), [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) | Web-agent evaluation; dialogue evaluation | Attacks and defenses vary by setting; evaluation protocols differ |
| Evidence-anchored scoring | Require rubric and cited evidence | [WebJudge](./references/An-Illusion-of-Progress-Assessing-the-Current-State-of-Web-Agents/meta/meta_info.txt), [RULERS](./references/RULERS-Locked-Rubrics-and-Evidence-Anchored-Scoring-for-Robust-LLM-Evaluation/meta/meta_info.txt) | AgentRewardBench; summarization eval | Can reduce recall; increases inference cost |
| Selective evaluation / escalation | Abstain or escalate only when uncertain | [Trust or Escalate](./references/Trust-or-Escalate-LLM-Judges-with-Provable-Guarantees-for-Human-Agreement/meta/meta_info.txt), [DAFE/CLEV](./references/DAFE-LLM-Based-Evaluation-Through-Dynamic-Arbitration-for-Free-Form-Question-Answering/meta/meta_info.txt) | question answering; preference evaluation; judging | Often needs calibration labels; generic uncertainty signals |

### Closest Prior Work

1) **Gaming the Judge** ([paper](./references/GAMING-THE-JUDGE-UNFAITHFUL-CHAIN-OF-THOUGHT-CAN-UNDERMINE-AGENT-EVALUATION/meta/meta_info.txt))

- What it does: isolates the vulnerability of trajectory judges to CoT rewriting and evaluates mitigations including manipulation-aware prompts, rubric-based evaluation (WebJudge-style), self-consistency, and CoT removal.
- Key limitation: mitigations are typically applied uniformly, which can reduce recall or increase inference cost.
- Why ours is different: we use **with-CoT vs no-CoT disagreement** as a mechanistically motivated trigger to apply strict rubric prompting only when needed.

2) **WebJudge (An Illusion of Progress?)** ([paper](./references/An-Illusion-of-Progress-Assessing-the-Current-State-of-Web-Agents/meta/meta_info.txt))

- What it does: uses rubric-style evaluation with evidence selection to improve web-agent trajectory judging.
- Key limitation: the strict protocol is applied to all trajectories (higher average overhead).
- Why ours is different: we treat strict rubric prompting as an escalation stage and apply it only on disagreement cases.

3) **DAFE/CLEV** ([paper](./references/DAFE-LLM-Based-Evaluation-Through-Dynamic-Arbitration-for-Free-Form-Question-Answering/meta/meta_info.txt))

- What it does: triggers arbitration when two judge models disagree in free-form question answering evaluation.
- Key limitation: disagreement is across models and the task is question answering with reference answers.
- Why ours is different: disagreement is across counterfactual **input views** aligned to the CoT manipulation vulnerability in trajectory judging.

4) **Trust or Escalate** ([paper](./references/Trust-or-Escalate-LLM-Judges-with-Provable-Guarantees-for-Human-Agreement/meta/meta_info.txt))

- What it does: calibrates judge confidence with a human-labeled set and escalates in a cascade to increase agreement with humans.
- Key limitation: requires labeled calibration data and is not tailored to CoT manipulation.
- Why ours is different: we use a label-free escalation trigger based on counterfactual ablation of narrative fields.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Gaming the Judge | Demonstrates CoT rewriting attacks and mitigation trade-offs | Defenses are applied uniformly or still leave gaps | Use with-CoT vs no-CoT disagreement to trigger escalation | Disagreement should concentrate cases where narratives are not supported by evidence |
| WebJudge | Evidence-anchored rubric evaluation | Higher overhead when applied to all trajectories | Apply rubric prompt only on disagreement | Lower average strict-evaluation overhead if disagreement rate is low/moderate |
| Trust or Escalate | Calibrated selective evaluation with guarantees | Needs calibration labels; not attack-surface specific | Use counterfactual ablation trigger (no new labels) | Trigger is aligned to CoT manipulation and does not require human calibration |
| DAFE/CLEV | Arbitration on judge-model disagreement | Different task and signal | Disagreement across input views | View disagreement directly probes reliance on narrative fields |
| RULERS | Locked rubric + evidence quotes | Heavy protocol | Use rubric-style constraints only on disagreement | Keeps strictness for uncertain cases while preserving recall elsewhere |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3.3-70B-Instruct (judge) | 70B parameters | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct | Primary judge model \(J\). Used for View1/View2/escalation to avoid model-scaling confounds. |
| Qwen2.5-32B-Instruct (fallback judge) | 32B parameters | https://huggingface.co/Qwen/Qwen2.5-32B-Instruct | Fallback if 70B inference cost exceeds budget; would require re-running all baselines under the 32B setting. |
| Small open-weight LLM for CoT rewriting (optional) | 7B–14B parameters | (e.g., https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Used only to rewrite CoT fields to construct attacked failures; does not need to match the judge model. |

**Important feasibility constraint (Azure content filter):**

- This proposal does **not** require OpenAI models. All LLM calls (judging and CoT rewriting) should use **open-weight** models (publicly downloadable model weights) run locally, or non-OpenAI providers. This avoids Azure OpenAI content filtering issues for adversarial/manipulation-related prompts.

**Training Data (if applicable):**

- No training data is needed (inference-only evaluation).

**Other Resources (if applicable):**

- AgentRewardBench dataset: https://huggingface.co/datasets/McGill-NLP/agent-reward-bench
- AgentRewardBench evaluation code: https://github.com/McGill-NLP/agent-reward-bench
- Input representation for judging: use the final accessibility tree (text-only) as final-state evidence for all trajectories (including VisualWebArena ones) and ignore screenshot fields, to keep the judge text-only and match the AgentRewardBench Llama-3.3 “A” (accessibility-tree) setting.

**Resource Estimate**:

- **Compute budget**: inference-only (no model training). Budget is measured in GPU-hours (graphics processing unit hours; number of GPUs \(\times\) wall-clock hours) and must stay within the 768 GPU-hour cap.
  - Baseline and method calls on unmodified trajectories (size \(N=1302\)):
    - View1-only: \(N\) calls.
    - View2-only: \(N\) calls.
    - Always-on strict rubric: \(N\) calls.
    - Ours (two-view + conditional escalation): \(2N + dN\) calls.
    - Total across these four runs: \((5 + d)N\) calls. With \(d \in [0.1, 0.3]\), this is **~6.6k–6.9k** judge calls.
  - CoT rewriting for attacked failures: use a small open-weight model (7B–14B) to rewrite CoT fields for ground-truth failure trajectories. This is at most one generation per failure trajectory and is expected to be minor compared to 70B judging, but it should be included in the pilot throughput measurement.
  - Robustness calls on attacked failures: re-run View1-only and our method on the attacked-failure subset to measure FPR and \(\Delta\)FPR. If the attacked-failure subset size is \(N_f\), this adds \(N_f\) (View1-only) + \((2 + d)N_f\) (ours) judge calls.

  *Note: the exact call count depends on \(d\) and the number of failures \(N_f\); the pilot-and-extrapolate procedure below is required to confirm total GPU-hours stays within budget.*
  - **Pilot-and-extrapolate procedure (required for feasibility)**: run the pipeline on **50 trajectories** first to measure (i) average tokens per call and (ii) wall-clock throughput. Extrapolate total GPU-hours before running the full benchmark.
    - Proceed with 70B only if the extrapolated total is **\(\le 768\)** GPU-hours.
    - If extrapolated cost exceeds budget, switch to the 32B fallback judge (and re-run the baseline table under that setting).
- **GPU memory**: 70B inference likely requires tensor parallelism (splitting model weights across multiple GPUs) across multiple 80GB GPUs, or quantization (reduced-precision weights) to fit on fewer GPUs.
- **API (application programming interface) usage**: none required.

**Infrastructure constraints** (proposals requiring these are infeasible):

- Search engine APIs (Google, Bing) — NOT available
- Web browsers / desktop GUIs / mobile environments — NOT available
- Complex game engines or heavy simulation environments — NOT available

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| AgentRewardBench | A benchmark of 1,302 expert-annotated web-agent trajectories (from environments including WebArena, VisualWebArena, AssistantBench, WorkArena, and WorkArena++) for evaluating trajectory judges that predict task success | Precision (\(\uparrow\)), Recall (\(\uparrow\)), F1 (\(\uparrow\)); disagreement rate; escalation rate; false positive rate (FPR; \(\downarrow\)) on attacked failures; cost (tokens/calls; \(\downarrow\)) | dev (196) / test (1106) | https://huggingface.co/datasets/McGill-NLP/agent-reward-bench | https://github.com/McGill-NLP/agent-reward-bench |

**Metric definitions (first use):**

- **Precision**: \(\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})\), where TP/FP are true positives/false positives (higher is better).
- **Recall**: \(\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})\), where FN is false negatives (higher is better).
- **F1**: harmonic mean of precision and recall (higher is better).
- **Disagreement rate**: fraction of trajectories where View1 and View2 predictions differ.
- **Escalation rate**: fraction of trajectories that trigger Stage 2 strict judging (equal to the disagreement rate in this design; lower is cheaper).
- **FPR (false positive rate) on attacked failures**: fraction of attacked-failure trajectories labeled as success (lower is better).
- **\u0394FPR**: \(\mathrm{FPR}(\text{attacked failures}) - \mathrm{FPR}(\text{unmodified failures})\) (lower is better).
- **Cost**: total judge compute measured as (i) number of judge calls and (ii) total processed tokens (lower is better).

**Evaluation Scripts:**

- Use the official evaluation pipeline from AgentRewardBench: https://github.com/McGill-NLP/agent-reward-bench
- Implement View1/View2 creation as a deterministic preprocessing step over the dataset fields.

**Download Links Checklist:**

- [x] Benchmark dataset download link provided
- [x] Evaluation code link provided
- [x] Base model download link provided

**Robustness evaluation (offline attack; fully automated):**

- Construct an **attacked-failure** subset by taking ground-truth failure trajectories and rewriting only their CoT fields using manipulation strategies described in Gaming the Judge (e.g., fabricated progress statements), while keeping actions and final-state evidence unchanged ([Gaming the Judge](./references/GAMING-THE-JUDGE-UNFAITHFUL-CHAIN-OF-THOUGHT-CAN-UNDERMINE-AGENT-EVALUATION/meta/meta_info.txt)).
- Labels remain failures by construction; evaluation measures whether the judge is fooled by narrative manipulation.
- Report FPR on attacked failures and \(\Delta\)FPR, defined as \(\mathrm{FPR}(\text{attacked failures}) - \mathrm{FPR}(\text{unmodified failures})\) (lower is better).

### Main Results

#### Comparability Rules

All reported comparisons in the main results table should be directly comparable:

- Same benchmark (AgentRewardBench) and split
- Same judge model (Llama-3.3-70B-Instruct)
- Same evaluation protocol and prompts (except where the prompt is the intervention under study)

#### Results Table (to be filled by verification)

*All Precision/Recall/F1 values are percentages (higher is better). FPR is a percentage (lower is better).* 

| Method | Base Model | Benchmark | Precision (\(\uparrow\)) | Recall (\(\uparrow\)) | F1 (\(\uparrow\)) | FPR on attacked failures (\(\downarrow\)) | Avg calls / traj (\(\downarrow\)) | Source | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Rule-based evaluator | N/A | AgentRewardBench | 83.8% | 55.9% | 67.1% | N/A | 0 | [AgentRewardBench](./references/AgentRewardBench-Evaluating-Automatic-Evaluations-of-Web-Agent-Trajectories/meta/meta_info.txt) | Published result (Table 1); no CoT exposure |
| View1-only (with CoT) | Llama-3.3-70B | AgentRewardBench | 67.7% | 79.0% | 72.9% | TBD | 1 | [AgentRewardBench](./references/AgentRewardBench-Evaluating-Automatic-Evaluations-of-Web-Agent-Trajectories/meta/meta_info.txt) | Published result (Table 1); should be reproduced in verifier pipeline |
| View2-only (no CoT) | Llama-3.3-70B | AgentRewardBench | **TBD** | **TBD** | **TBD** | **TBD** | 1 | - | **Needs re-run** (no published result in this exact setting) |
| Always-on strict rubric (no CoT) | Llama-3.3-70B | AgentRewardBench | **TBD** | **TBD** | **TBD** | **TBD** | 1 | - | **Needs re-run** (strict prompt is new) |
| **Ours: view-disagreement escalation** | Llama-3.3-70B | AgentRewardBench | **TBD** | **TBD** | **TBD** | **TBD** | \(2 + d\) | - | Two base views; strict rubric only on disagreement (\(d\) = disagreement rate) |

**Published reference points (not directly comparable; different judge backbones):**

- AgentRewardBench reports GPT-4o-based simplified judges with precision 69.8% and recall 83.1% (F1 75.9%) for success prediction (Table 1; higher is better), but these use different judge models and are not directly comparable to our Llama-only evaluation.

### Ablation Studies

| Variant | What changes | Expected finding |
|---|---|---|
| View1-only | Judge always sees CoT | Higher recall but more susceptible to CoT rewriting |
| View2-only | Judge never sees CoT | Higher robustness to CoT rewriting but recall may drop |
| Always-on strict rubric | Always use \(P_\text{strict}\) (no CoT) | Higher precision but potentially lower recall and higher token cost |
| Random escalation | Escalate a random fraction of trajectories (matching our escalation rate) | Worse precision/cost than targeted escalation if disagreement is informative |
| Matched-cost control | Replace View1+View2 with two View2 calls | Tests whether gains come from view contrast vs repeated evidence-only judging |

### Analysis (Optional)

- **Pilot diagnostic**: measure View1–View2 disagreement rate on unmodified AgentRewardBench and on attacked failures.
- **Stratified disagreement**: disagreement conditioned on {unmodified success, unmodified failure, attacked failure}.
- **Escalation effectiveness**: accuracy of the strict rubric prompt on disagreement vs non-disagreement subsets.
- **Pareto analysis**: precision versus cost (tokens and calls) across methods.

---

## Success Criteria

**Criterion 1: Precision–cost improvement under attack**

- Hypothesis: selective escalation reduces FPR on attacked failures compared to View1-only, at lower or comparable token cost than always-on strict evaluation.
- Validation: our method achieves lower attacked-failure FPR than View1-only while using fewer strict-rubric tokens than always-on strict evaluation.

**Criterion 2: Recall preservation on unmodified data**

- Hypothesis: selective escalation preserves more recall than always removing CoT while improving precision relative to View1-only.
- Validation: on the unmodified test set, our method improves precision over View1-only and achieves higher recall than View2-only.

---

## Impact Statement

If successful, this approach provides a simple, training-free mechanism for making trajectory judges more robust to narrative manipulation while preserving useful signal from agent reasoning traces when they are consistent with observable evidence. This would benefit practitioners who use automatic trajectory judges to filter or score agent rollouts for training (e.g., rejection sampling or reinforcement learning), by reducing false positives without paying the full overhead of always-on strict evaluation.

---

## References

- [AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories](./references/AgentRewardBench-Evaluating-Automatic-Evaluations-of-Web-Agent-Trajectories/meta/meta_info.txt) — Lu et al., 2025
- [Gaming the Judge: Unfaithful Chain-of-Thought Can Undermine Agent Evaluation](./references/GAMING-THE-JUDGE-UNFAITHFUL-CHAIN-OF-THOUGHT-CAN-UNDERMINE-AGENT-EVALUATION/meta/meta_info.txt) — Khalifa et al., 2025/2026
- [An Illusion of Progress? Assessing the Current State of Web Agents](./references/An-Illusion-of-Progress-Assessing-the-Current-State-of-Web-Agents/meta/meta_info.txt) — Xue et al., 2025
- [Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement](./references/Trust-or-Escalate-LLM-Judges-with-Provable-Guarantees-for-Human-Agreement/meta/meta_info.txt) — Jung et al., 2024
- [DAFE/CLEV: LLM-Based Evaluation Through Dynamic Arbitration for Free-Form Question Answering](./references/DAFE-LLM-Based-Evaluation-Through-Dynamic-Arbitration-for-Free-Form-Question-Answering/meta/meta_info.txt) — Badshah et al., 2025
- [Auto-Prompt Ensemble for LLM Judge](./references/Auto-Prompt-Ensemble-for-LLM-Judge/meta/meta_info.txt) — Li et al., 2025
- [RULERS: Locked Rubrics and Evidence-Anchored Scoring for Robust LLM Evaluation](./references/RULERS-Locked-Rubrics-and-Evidence-Anchored-Scoring-for-Robust-LLM-Evaluation/meta/meta_info.txt) — Hong et al., 2026
- [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) — Zhao et al., 2025
- [CalibraEval](https://arxiv.org/abs/2410.15393) — 2024
- [TrustJudge](https://arxiv.org/abs/2509.21117) — Wang et al., 2025
- [CAP: Improving the Robustness of LLM-as-a-Judge Against Adversarial Score Manipulation via Comparative Augmented Prompting](https://openreview.net/forum?id=wYU6OYFvid) — Zhang et al., 2025
- [Optimization-based Prompt Injection Attack to LLM-as-a-Judge (JudgeDeceiver)](https://arxiv.org/abs/2403.17710) — Shi et al., 2024
- [Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks](https://arxiv.org/abs/2505.13348) — 2025
- [MT-Bench](https://arxiv.org/abs/2306.05685) — 2023
- [Chatbot Arena](https://arxiv.org/abs/2403.04132) — 2024
- [RewardBench](https://arxiv.org/abs/2403.13787) — 2024
- [WebArena](https://arxiv.org/abs/2307.13854) — Zhou et al., 2024
- [VisualWebArena](https://arxiv.org/abs/2401.03564) — Koh et al., 2024
- [AssistantBench](https://arxiv.org/abs/2402.17573) — Yoran et al., 2024
- [WorkArena](https://arxiv.org/abs/2403.07718) — Drouin et al., 2024
- [WorkArena++](https://arxiv.org/abs/2407.05291) — Boisvert et al., 2025
- [AER: Autonomous Evaluation and Refinement of Digital Agents](https://arxiv.org/abs/2404.06474) — Pan et al., 2024
- [NNetNav](https://arxiv.org/abs/2410.02907) — Murty et al., 2025
- [Mind2Web](https://arxiv.org/abs/2306.06070) — 2023
- [Mind2Web-Live (WebCanvas)](https://arxiv.org/abs/2406.12373) — 2024
