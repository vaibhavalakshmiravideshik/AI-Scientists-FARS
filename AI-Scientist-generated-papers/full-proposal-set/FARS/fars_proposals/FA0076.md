# untitled

# Entailment-Checklist Scoring: An API-Free Replacement for Gemini-Based SodaM Evaluation

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Benchmark progress in video captioning increasingly depends on **model-based evaluators** (LLM-as-a-judge or multimodal judges) rather than only n-gram overlap metrics. This is especially common for long-form or structured captions where multiple aspects matter (e.g., scene details, camera motion, dialogue, background sounds). While model-based evaluation can better reflect human preferences, it introduces a reproducibility risk: if the benchmark’s primary score depends on a proprietary model, then third parties cannot re-evaluate systems, reproduce reported numbers, or run inexpensive ablations.

This proposal targets a concrete instance of this problem in **OmniDenseCaptioning** evaluation from **TimeChat-Captioner**. OmniDenseCaptioning is a dense video captioning setting where each short clip is divided into multiple temporal segments and each segment is annotated with multiple structured fields (e.g., visual details, dialogue, camera motion). Its main metric, **SodaM** (a multi-dimensional extension of SODA that aligns predicted and ground-truth temporal segments and computes an F1-style score using per-segment keypoint coverage), uses a per-segment **checklist-style keypoint coverage score** computed by calling **Gemini 2.5 Pro** through a proprietary API in the official evaluation script **[TimeChat-Captioner](./references/TimeChat-Captioner-Scripting-Multi-Scene-Videos-with-Time-Aware-and-Structural-Audio-Visual-Captions/meta/meta_info.txt)**. The checklist is computed over six dimensions (segment detail, background, acoustics, shooting style, speech, camera state) and is used as the core “caption quality” component.

This judge dependency is not unique to SodaM. Recent video captioning benchmarks such as **MCTS-VCB** evaluate caption quality by decomposing the reference into many keypoints and computing a precision/recall/F1 score using **entailment judgments** from a large model judge (**[MCTS-VCB](./references/Evaluating-Multimodal-Large-Language-Models-on-Video-Captioning-via-Monte-Carlo-Tree-Search/meta/meta_info.txt)**). Together, these works suggest that “keypoint entailment F1” evaluation is becoming common in video captioning, increasing the value of an API-free entailment/checklist scorer.

### The Problem

The official SodaM evaluation is difficult to reproduce for two reasons:

1. **Proprietary judge dependency**: The official evaluation script’s `Checklist_Score` uses Gemini as a judge to decide which ground-truth “keypoints” are covered by the predicted caption. This means that (i) results can drift as the model changes, (ii) cost scales with the number of segment pairs scored, and (iii) researchers without API access cannot run the benchmark.

2. **Checklist structure is not exploited by common open metrics**: Standard text metrics such as CIDEr, SPICE, or BERTScore operate on whole-reference similarity and do not directly model “did the caption cover these specific keypoints?” This mismatch is important because SodaM is explicitly a **keypoint coverage** objective, not generic similarity.

Prior work on checklist-based evaluation and LLM judges (e.g., **CheckEval** and **RocketEval**) improves reliability and reduces gaming, but still assumes access to an LLM judge at evaluation time:

- **[CheckEval](./references/CheckEval-A-reliable-LLM-as-a-Judge-framework-for-evaluating-text-generation-using-checklists/meta/meta_info.txt)** (LLM-judge with checklists): increases judge reliability by structuring evaluation as checklist items, but still requires LLM inference at scoring time.
- **[RocketEval](./references/ROCKETEVAL-EFFICIENT-AUTOMATED-LLM-EVALUATION-VIA-GRADING-CHECKLIST/meta/meta_info.txt)** (checklist grading efficiency): reduces the cost of checklist grading, but not to zero and still depends on a model judge.

The practical question is: **Can we replace the Gemini checklist judge with an API-free scorer that preserves the evaluation’s decision signals (ranking / pairwise preferences) well enough to support reproducible research?**

### Key Insight and Hypothesis

**Key insight:** SodaM’s Gemini component is not doing open-ended quality judgment; it is largely a structured decision: for each ground-truth keypoint sentence, decide whether it is **entailed by** (or clearly implied by) some sentence in the predicted caption. This is close to sentence-level semantic matching / natural language inference (NLI), which can be approximated using open embedding and entailment models.

**Hypothesis:** A fully automated, API-free **entailment-based checklist scorer** (embedding retrieval + open NLI classifier) can match the Gemini judge’s keypoint coverage decisions closely enough that:

1) it has high correlation with Gemini-based SodaM at the clip level, and
2) it preserves most **pairwise “which system is better?”** decisions on non-tied examples,

while being cheap and stable enough for routine use.

Why this might fail: (i) the Gemini judge may rely on world knowledge or discourse-level reasoning not captured by NLI models; (ii) entailment models may be brittle to long, stylistic camera-state descriptions; (iii) a proxy may accidentally become a length metric (longer captions cover more keypoints) unless explicitly controlled.

---

## Proposed Approach

### Overview

We propose **Entailment-Checklist Scoring (ECS)**: a drop-in replacement for the Gemini-based `Checklist_Score` in SodaM.

Given:
- ground-truth keypoints for a segment, grouped into the 6 SodaM dimensions, and
- a predicted caption for the matched predicted segment(s),

ECS outputs the same structured result as the Gemini judge:
- `correct_count` and the list of `correct_keypoints` per dimension,
- an overall keypoint coverage ratio.

The core idea is:
1) convert the task into **per-keypoint verification**,
2) use an embedding model to retrieve the most relevant candidate sentence(s) from the prediction, and
3) use an open NLI model to decide whether the candidate sentence entails the keypoint.

### Method Details

#### A. Keypoint extraction (match official preprocessing)

The official SodaM evaluator can operate either on pre-provided keypoints or by extracting keypoints from the ground-truth text (splitting into short sentences with punctuation and list markers). To stay comparable, ECS will reuse the same keypoint extraction logic from the official `eval_sodam.py` (regex cleanup + sentence splitting) so that differences are attributable to scoring, not preprocessing.

#### B. Candidate sentence set from the predicted caption

For each merged predicted caption (concatenation over the DP-aligned predicted segments), split into candidate sentences using the same delimiter set as the keypoint extractor.

Let `S_pred = {s_1, …, s_m}` be predicted sentences and `k` be a single ground-truth keypoint sentence.

#### C. Embedding retrieval (cheap pruning)

Compute embeddings with an open model (default: **BAAI/bge-m3**, available in the environment).

- Compute `e(k)` and `e(s_i)`.
- Select the top-`r` candidate sentences by cosine similarity, `TopR(k)`.

This reduces NLI calls and makes the method scalable.

#### D. Entailment decision (API-free verifier)

Use an open NLI model (e.g., `MoritzLaurer/DeBERTa-v3-base-mnli` or a comparable open entailment classifier) to compute `P(entail | premise=s, hypothesis=k)` for each `s ∈ TopR(k)`. Here, `r` is a small constant (e.g., 4–8) controlling how many top-similarity candidate sentences are verified per keypoint.

Decision rule for a keypoint:
- mark keypoint `k` as covered if `max_{s ∈ TopR(k)} P(entail | s, k) ≥ τ_entail`.

We allow **dimension-specific thresholds** `{τ_entail[d]}` because some dimensions (e.g., `camera_state`) may require stricter matching than others.

#### E. Score aggregation (match SodaM interface)

For each dimension `d`, let `K_d` be the extracted keypoints. Then:
- `correct_count[d] = |{k ∈ K_d : covered(k)}|`
- `ratio[d] = correct_count[d] / max(1, |K_d|)`
- overall keypoint coverage ratio is `sum_d correct_count[d] / max(1, sum_d |K_d|)`

This output can be plugged into the existing SodaM precision/recall/F1 aggregation unchanged.

#### F. Calibration without leaking evaluation

Thresholds are tuned on a held-out calibration split using Gemini only as a **teacher signal** (not required at deployment). Concretely:
- sample a fixed set of clips (deterministic hash of `clip_path`),
- run the official Gemini checklist judge on those clips to get per-keypoint labels,
- tune `{τ_entail[d]}` (and optionally `r`) to maximize agreement.

The final ECS evaluator is then API-free for scoring the remaining data.

### Key Innovations

- **Checklist-to-entailment reduction for multimodal caption evaluation**: exploit the fact that SodaM’s LLM judge is primarily performing a keypoint entailment check, enabling an API-free replacement.
- **Verifier-style evaluation without LLM calls**: use an open entailment model as a deterministic verifier for checklist items, reducing cost and improving reproducibility.
- **Decision-focused validation**: evaluate agreement on pairwise preferences (which system is better) in addition to correlation, because this is what drives leaderboard conclusions.

---

## Related Work

### Field Overview

**Video caption evaluation** has historically relied on reference-based similarity metrics (e.g., CIDEr, SPICE, METEOR), but these struggle for long, structured captions where multiple orthogonal dimensions matter. The **SODA** family targets dense, story-oriented video captioning by aligning predicted and ground-truth segments and then scoring content similarity (**[SODA](./references/SODA-Story-Oriented-Dense-Video-Captioning-Evaluation-Framework/meta/meta_info.txt)**). TimeChat-Captioner extends this idea with **SodaM**, adding a multi-dimensional checklist score evaluated by an LLM judge (**[TimeChat-Captioner](./references/TimeChat-Captioner-Scripting-Multi-Scene-Videos-with-Time-Aware-and-Structural-Audio-Visual-Captions/meta/meta_info.txt)**).

In parallel, **LLM-as-a-judge** has become a common evaluation paradigm for text generation and agentic tasks. Recent work shows that judge reliability depends strongly on prompting, rubric structure, and adversarial robustness. Checklist-based judging frameworks (e.g., CheckEval, RocketEval, Check-Eval) aim to make LLM judging more reliable, but they still depend on LLM calls at evaluation time.

Finally, there is a line of work using **entailment / factuality classifiers** to evaluate whether a generated statement is supported by evidence (often via NLI models). Our proposal borrows this verifier framing, but applies it to a checklist rubric over structured dense captions.

### Related Papers

- **[TimeChat-Captioner](./references/TimeChat-Captioner-Scripting-Multi-Scene-Videos-with-Time-Aware-and-Structural-Audio-Visual-Captions/meta/meta_info.txt)**: introduces OmniDenseCaptioning and SodaM, whose checklist score depends on Gemini judging.
- **[SODA](./references/SODA-Story-Oriented-Dense-Video-Captioning-Evaluation-Framework/meta/meta_info.txt)**: reference metric for story-oriented dense video captioning with segment alignment.
- **[IF-VidCap](./references/IF-VidCap-Can-Video-Caption-Models-Follow-Instructions/meta/meta_info.txt)**: evaluates instruction-following failures in video captioning and motivates multi-criteria, structured evaluation beyond n-gram overlap.
- **[AnyCap Project](./references/AnyCap-Project-A-Unified-Framework,-Dataset,-and-Benchmark-for-Controllable-Omni-modal-Captioning/meta/meta_info.txt)**: benchmark for controllable captioning that highlights the need for reliable, granular automatic evaluation.
- **[CheckEval](./references/CheckEval-A-reliable-LLM-as-a-Judge-framework-for-evaluating-text-generation-using-checklists/meta/meta_info.txt)**: LLM-as-a-judge framework using checklists to improve reliability.
- **[RocketEval](./references/ROCKETEVAL-EFFICIENT-AUTOMATED-LLM-EVALUATION-VIA-GRADING-CHECKLIST/meta/meta_info.txt)**: efficient checklist grading for LLM evaluation.
- **[Prometheus](https://arxiv.org/abs/2310.08491)**: trains open LLM judges to approximate proprietary evaluation.
- **[Auto-J](https://arxiv.org/abs/2310.05470)**: builds automatic judges and studies judge reliability/transfer.
- **[ARES](https://arxiv.org/abs/2311.09476)**: automatic evaluation with model-based judges, including calibration considerations.
- **[G-Eval](https://arxiv.org/abs/2303.16634)**: proposes LLM-based evaluation with rubric-like prompting.
- **[One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)**: shows adversarial vulnerability of LLM judges via minimal perturbations.
- **[LLMs Cannot Reliably Judge (Yet?)](https://arxiv.org/abs/2506.09443)**: comprehensive assessment of robustness issues in LLM-as-a-judge evaluations.
- **[Check-Eval](https://arxiv.org/abs/2407.14467)**: checklist-based LLM evaluation emphasizing structured criteria.
- **[BERTScore](https://arxiv.org/abs/1904.09675)**: embedding-based reference metric using contextual similarity.
- **[BLEURT](https://arxiv.org/abs/2004.04696)**: learned reference metric trained from human ratings.
- **[BARTScore](https://arxiv.org/abs/2106.11520)**: evaluation via sequence-to-sequence log-likelihood.
- **[CIDEr](https://arxiv.org/abs/1411.5726)**: consensus-based n-gram metric widely used in captioning.
- **[SPICE](https://arxiv.org/abs/1607.08822)**: scene-graph-based caption metric targeting semantic propositions.
- **[METEOR](https://aclanthology.org/W05-0909/)**: reference metric using stemming/synonyms and alignment.
- **[CLIPScore](https://arxiv.org/abs/2104.08718)**: reference-free image-text similarity metric based on CLIP.
- **[Reinforced Video Captioning with Entailment Rewards](./references/Reinforced-Video-Captioning-with-Entailment-Rewards/meta/meta_info.txt)**: introduces CIDEnt, an entailment-corrected reward/metric for video captioning that penalizes contradictions missed by n-gram overlap.
- **[SummaC](https://arxiv.org/abs/2112.08777)**: NLI-based factual consistency metric for summarization.
- **[FactCC](https://aclanthology.org/2020.emnlp-main.750/)**: classifier-based factual consistency evaluation.
- **[QuestEval](https://arxiv.org/abs/2104.08716)**: question-answering based evaluation of generation.
- **[FactScore](https://arxiv.org/abs/2305.14251)**: factuality evaluation by decomposing into atomic facts and verifying.
- **[MCTS-VCB](./references/Evaluating-Multimodal-Large-Language-Models-on-Video-Captioning-via-Monte-Carlo-Tree-Search/meta/meta_info.txt)**: video caption benchmark that evaluates keypoint precision/recall/F1 via entailment judgments, illustrating the growing use of keypoint-entailment evaluators in video captioning.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Reference-based similarity metrics | Compare generation to reference text using n-grams or embeddings | CIDEr, SPICE, METEOR, BERTScore, BLEURT, BARTScore | COCO captioning; summarization benchmarks | Weak on structured multi-criteria captions; may reward verbosity |
| Dense video captioning metrics | Align predicted and GT segments, then score content | SODA; SodaM (TimeChat-Captioner) | Dense video captioning datasets / OmniDenseCaptioning | Alignment errors propagate; scoring can be expensive |
| LLM-as-a-judge (rubric/checklist) | Use an LLM to grade outputs against criteria | G-Eval, CheckEval, RocketEval, Check-Eval | MT-Bench-like evals; many domain-specific benchmarks | Proprietary dependency, instability, adversarial vulnerability |
| Open/distilled judges | Train open models to approximate strong judges | Prometheus, Auto-J, ARES | General text generation evaluation | Still model-based; may not match domain-specific rubrics |
| Verifier / entailment-based evaluation | Reduce evaluation to entailment/factuality classification | SummaC, FactCC, FactScore | Summarization / factuality benchmarks | Often needs careful decomposition; domain shift risk |

### Closest Prior Work

- **TimeChat-Captioner / SodaM**: Defines the exact evaluation target and uses Gemini to grade keypoint coverage per dimension. Our work keeps the SodaM aggregation unchanged but replaces the judge.
- **SODA**: Shows that dense caption evaluation benefits from explicit segment alignment; we reuse the alignment and focus on the checklist grading stage.
- **CheckEval / RocketEval / Check-Eval**: Demonstrate that checklist structuring improves judge reliability. ECS adopts the checklist structure but replaces LLM grading with open entailment verification.
- **NLI-based factuality metrics (SummaC, FactCC)**: Use entailment models to decide statement support. ECS adapts this verifier idea to checklist keypoints in structured dense captions.
- **MCTS-VCB**: Uses entailment-based keypoint precision/recall/F1 for video caption evaluation, but relies on a large model judge; ECS targets the same evaluation structure while removing proprietary judge dependence.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| TimeChat-Captioner (SodaM) | Multi-dim dense caption metric with Gemini checklist judge | Proprietary, expensive, hard to reproduce | Replace Gemini with open entailment-based scorer | Removes API dependency while targeting the same decision (keypoint coverage) |
| CheckEval / RocketEval | LLM checklist grading for text generation | Still needs an LLM at scoring time | Use entailment verifier instead of an LLM judge | API-free and less sensitive to judge prompt instability |
| Reference metrics (BERTScore/BLEURT/…) | Whole-text similarity scoring | Not checklist/keypoint aware | Score per-keypoint coverage then aggregate | Closer to SodaM’s intended semantics |
| NLI factuality metrics (SummaC/FactCC) | Entailment-based factuality checking | Not designed for multi-dimension dense captions | Add segment alignment + dimension-specific thresholds | Better fit to SodaM structure |
| MCTS-VCB (ACL 2025) | Keypoint entailment precision/recall/F1 for video captioning | Still depends on an LLM judge at evaluation time | Use open NLI verifier + embeddings to remove judge API | Maintains keypoint-entailment structure with API-free scoring |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Gemini judge (teacher) | API | `gemini-2.5-pro` | Only for calibration / evaluation labels, not for deployment |
| Embedding model | - | https://huggingface.co/BAAI/bge-m3 | Used for retrieval/pruning |
| Entailment model (NLI) | ~0.3–1B | https://huggingface.co/models?pipeline_tag=text-classification&search=mnli | Any strong open MNLI/NLI classifier (e.g., DeBERTa-v3 MNLI) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| None | Inference-only scoring | - | - | - |

**Resource Estimate**:

- **Compute budget**: 
  - Embedding + NLI scoring on a few hundred thousand sentence pairs should fit within **< 20 A100 GPU-hours** (single GPU), plus CPU preprocessing.
  - Gemini teacher scoring for a calibration/eval subset (e.g., 200 clips) requires on the order of **~1k–5k API calls**, depending on segment matching.
- **GPU memory**: 1× A100 80GB (or smaller) is sufficient.
- **API usage**: Gemini calls only for a fixed subset; ECS itself is API-free.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| OmniDCBench (via TimeChat-Captioner eval artifacts) | Dense multi-scene video captioning benchmark with 6-dim structured GT per segment | SodaM total and per-dimension scores; correlation and pairwise decision agreement vs Gemini | test (provided prediction JSONLs) | https://github.com/yaolinli/TimeChat-Captioner (Eval/*.jsonl) | Official `Eval/eval_sodam.py` with judge module swapped |

**Primary metrics:**
- **Clip-level correlation**: Spearman and Kendall correlation between ECS-SodaM and Gemini-SodaM per clip.
- **Pairwise decision agreement**: For each clip, compare two systems’ scores under Gemini vs ECS and measure agreement on which is better; report separately for “clear” cases (Gemini score gap above a small threshold) and near-ties.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Metric 1 | Metric 2 | Source | Notes |
|---|---|---|---|---|---|---|
| Gemini checklist (official judge) | gemini-2.5-pro | OmniDCBench | **TBD** (SodaM) | - | - | Teacher / reference |
| Length-only baseline | - | OmniDCBench | **TBD** (agreement) | **TBD** (corr) | - | Embarrassing baseline |
| Embedding-only keypoint match | bge-m3 | OmniDCBench | **TBD** | **TBD** | - | No entailment verifier |
| **ECS (ours)** | bge-m3 + open NLI | OmniDCBench | **TBD** | **TBD** | - | API-free evaluation |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| ECS (full) | embedding retrieval + entailment verifier + per-dim thresholds | Best agreement with Gemini |
| w/o entailment verifier | drop NLI, use cosine threshold only | More false positives on negation/entity swaps |
| global threshold | use one τ across all dimensions | Worse on camera/speech dimensions |

### Analysis (Optional)

- **Stress tests via semantic perturbations**: apply deterministic transformations to predicted captions (negation insertion; entity swaps) that preserve length, and measure whether ECS tracks Gemini’s penalty.
- **Error breakdown by dimension**: where disagreement concentrates (camera_state vs speech_content vs others).

---

## Success Criteria

**Criterion 1: Agreement with Gemini-based SodaM decisions**
- Hypothesis: ECS preserves most pairwise preferences induced by Gemini-based SodaM on non-tied examples.
- Validation: On a fixed evaluation subset with Gemini teacher labels, report pairwise agreement as a function of the Gemini score gap (e.g., agreement for |Δ| ≥ {0.02, 0.05, 0.10}). **Decision rule**: if agreement does not become high on clear-gap cases (e.g., it stays far below ~90% even at large gaps), treat ECS as not a viable drop-in replacement and pivot (e.g., to distilling a small judge model).

**Criterion 2: Robustness to length-matched semantic corruptions**
- Hypothesis: Compared to embedding-only scoring, adding entailment verification reduces false positives under negation/entity-swap perturbations.
- Validation: On the perturbation suite, ECS penalizes corrupted captions similarly to Gemini and more strongly than the embedding-only baseline.

---

## Impact Statement

If successful, ECS would make a recently proposed dense video captioning metric (SodaM) reproducible without proprietary LLM access. This enables cheaper ablations, stable re-evaluation over time, and broader participation in multimodal captioning research where evaluation cost and API restrictions currently limit experimentation. The same entailment-checklist design could also serve as a template for replacing proprietary judges in other checklist-based text/video evaluation protocols.

---

## References

- [TimeChat-Captioner Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions](./references/TimeChat-Captioner-Scripting-Multi-Scene-Videos-with-Time-Aware-and-Structural-Audio-Visual-Captions/meta/meta_info.txt)
- [SODA: Story Oriented Dense Video Captioning Evaluation Framework](./references/SODA-Story-Oriented-Dense-Video-Captioning-Evaluation-Framework/meta/meta_info.txt)
- [CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists](./references/CheckEval-A-reliable-LLM-as-a-Judge-framework-for-evaluating-text-generation-using-checklists/meta/meta_info.txt)
- [ROCKETEVAL: Efficient Automated LLM Evaluation via Grading Checklist](./references/ROCKETEVAL-EFFICIENT-AUTOMATED-LLM-EVALUATION-VIA-GRADING-CHECKLIST/meta/meta_info.txt)
- [Prometheus](https://arxiv.org/abs/2310.08491)
- [Auto-J](https://arxiv.org/abs/2310.05470)
- [ARES](https://arxiv.org/abs/2311.09476)
- [G-Eval](https://arxiv.org/abs/2303.16634)
- [Check-Eval](https://arxiv.org/abs/2407.14467)
- [BERTScore](https://arxiv.org/abs/1904.09675)
- [BLEURT](https://arxiv.org/abs/2004.04696)
- [BARTScore](https://arxiv.org/abs/2106.11520)
- [CIDEr](https://arxiv.org/abs/1411.5726)
- [SPICE](https://arxiv.org/abs/1607.08822)
- [METEOR](https://aclanthology.org/W05-0909/)
- [CLIPScore](https://arxiv.org/abs/2104.08718)
- [Reinforced Video Captioning with Entailment Rewards](./references/Reinforced-Video-Captioning-with-Entailment-Rewards/meta/meta_info.txt)
- [IF-VidCap](./references/IF-VidCap-Can-Video-Caption-Models-Follow-Instructions/meta/meta_info.txt)
- [AnyCap Project](./references/AnyCap-Project-A-Unified-Framework,-Dataset,-and-Benchmark-for-Controllable-Omni-modal-Captioning/meta/meta_info.txt)
- [SummaC](https://arxiv.org/abs/2112.08777)
- [FactCC](https://aclanthology.org/2020.emnlp-main.750/)
- [QuestEval](https://arxiv.org/abs/2104.08716)
- [FactScore](https://arxiv.org/abs/2305.14251)
- [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)
- [LLMs Cannot Reliably Judge (Yet?)](https://arxiv.org/abs/2506.09443)
