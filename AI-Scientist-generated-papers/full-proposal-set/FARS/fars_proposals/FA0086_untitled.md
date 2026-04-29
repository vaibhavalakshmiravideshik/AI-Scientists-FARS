# untitled

# Teacherless On-Policy Distillation for Temporal Video Grounding via Feedback-Conditioned Self-Teachers

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Verification constraints**:
  - Fully automated (no human-in-the-loop evaluation)
  - Fit within ≤768 NVIDIA A100 (80GB) GPU-hours total
  - No browser / search-API dependencies

## Introduction

### Context and Motivation

Temporal video grounding (TVG) is the task of predicting the start and end times of a video segment that matches a natural-language query (e.g., “the person opens the fridge”). Recent video-language models can be post-trained for TVG using **reinforcement learning with verifiable rewards (RLVR)**: instead of a learned reward model, training uses an automated verifier based on the ground-truth temporal interval.

A common RLVR optimizer is **Group Relative Policy Optimization (GRPO)**, which estimates a policy-gradient update from a group of rollouts per training example. In TVG, each rollout conditions on long video context (hundreds of frames), so the multi-rollout requirement makes training expensive.

**[Video-OPD](./references/Video-OPD-Efficient-Post-Training-of-Multimodal-Large-Language-Models-for-Temporal-Video-Grounding-via-On-Policy-Distillation/meta/meta_info.txt)** (arXiv:2602.02994) shows a more efficient alternative: sample a single on-policy trajectory from the student, then use a fixed teacher model’s token log-probabilities to provide **dense, token-level supervision** (reverse-KL signal) on that trajectory. This reduces rollout generation cost, but introduces a practical dependency: access to a teacher model that exposes token log-probabilities.

### The Problem

Teacher-based on-policy distillation is attractive for long-context multimodal RLVR, but it is not always practical:

- **Teacher availability**: Practitioners may not have a stronger domain teacher; training one can cost as much as training the student.
- **Teacher log-probabilities**: Even if a strong teacher is available as an API, it may not expose token-level log-probabilities, blocking distillation.

At the same time, TVG already has a public verifier: the **intersection-over-union (IoU)** between the predicted temporal interval and the ground-truth interval. The verifier can produce more than a scalar reward. For example, it can provide structured feedback such as whether the predicted start time is too early or too late.

In text-only RLVR domains like code, **Self-Distillation Policy Optimization (SDPO)** (arXiv:2601.20802) shows that tokenized verifier feedback (e.g., runtime errors) can define a **feedback-conditioned self-teacher** without any external teacher: the same model, conditioned on feedback, assigns higher probability to better tokens, and the student is trained to match that distribution.

This proposal asks whether the same “teacherless distillation from feedback” idea can replace teacher-based distillation in TVG.

### Key Insight and Hypothesis

**Key insight.** If a base video-language model can use verifier feedback in-context to revise its temporal prediction, then the model conditioned on that feedback defines a stronger distribution over the same output tokens. Distilling that distribution back into the original policy yields a dense, token-level learning signal analogous to Video-OPD, but without any external teacher.

**Hypothesis.** On TimeLens-Bench TVG benchmarks, SDPO-style self-distillation using **coarse IoU-only feedback** will (a) approach teacher-based on-policy distillation performance, and (b) match or outperform GRPO at substantially lower rollout-generation cost.

**Why this could fail.** (i) The model may not improve when prompted with IoU feedback, making the self-teacher signal uninformative. (ii) Any gains may require highly informative boundary-direction feedback, in which case the method is closer to “hinting” than to a general self-distillation mechanism.

---

## Proposed Approach

### Overview

We propose **SDPO-TVG**, a teacherless on-policy distillation method for TVG.

For each training example (video, query), we:
1) sample one TVG answer from the current policy (the student),
2) compute verifier feedback from the ground-truth temporal interval,
3) re-evaluate the student’s generated tokens under the **same model** conditioned on that feedback (self-teacher),
4) update the student to match the self-teacher’s token distribution on the student trajectory.

This replaces Video-OPD’s external teacher log-probabilities with a feedback-conditioned self-teacher.

### Method Details

**Base model and IO format.** We use a public video-language model (primary: **Qwen/Qwen3-VL-8B-Instruct**). The model sees a video (sampled at 2 FPS with a frame cap, following TimeLens/Video-OPD settings) and a text query, and outputs timestamps as `"<start> to <end>"` in seconds.

**Verifier feedback tokens (main).** Given ground-truth interval \([s^*, e^*]\) and predicted interval \([\hat{s}, \hat{e}]\), compute:
- **IoU** = \(\frac{|[\hat{s},\hat{e}]\cap[s^*,e^*]|}{|[\hat{s},\hat{e}]\cup[s^*,e^*]|}\) (higher is better).
- **IoU bucket** \(b\in\{0, (0,0.1],\dots,(0.9,1]\}\) (11 buckets).

We convert this to a short textual feedback string, e.g.:

> `Verifier feedback: IoU_bucket=(0.2,0.3]. Please revise your timestamps.`

**Optional richer feedback (ablation; leakage check).** Add coarse boundary-direction categories with tolerance \(\epsilon=1\) second:
- `start_rel ∈ {too_early, ok, too_late}` by comparing \(\hat{s}\) to \(s^*\)
- `end_rel ∈ {too_early, ok, too_late}` by comparing \(\hat{e}\) to \(e^*\)

Feedback string example:

> `Verifier feedback: IoU_bucket=(0.2,0.3]; start=too_early; end=too_late. Please revise your timestamps.`

**Self-teacher and loss.** Let \(x\) be (video, query), \(y\) be the student-generated output tokens, and \(f\) be the feedback tokens. Define a self-teacher distribution as the same model conditioned on \(x\) and \(f\): \(\pi_\theta(\cdot\mid x,f)\). Following **[SDPO](./references/Reinforcement-Learning-via-Self-Distillation/meta/meta_info.txt)**, we compute token-level distillation loss on the student trajectory:
\[
\mathcal{L}_{\text{SDPO-TVG}}(\theta)=\sum_{t} KL\big(\pi_{\theta}(\cdot\mid x,y_{<t})\;\|\;\text{stopgrad}(\pi_{\theta}(\cdot\mid x,f,y_{<t}))\big).
\]
To control memory, we approximate the KL via top-K logit distillation (e.g., K=100), as suggested in SDPO (Section 2.2).

**Stability option (optional).** If SDPO-TVG shows instability, we will consider applying **[Veto](./references/Stable-On-Policy-Distillation-through-Adaptive-Target-Reformulation/meta/meta_info.txt)** as an objective-level stabilization (product-of-experts bridge target between student and self-teacher).

### Key Innovations

- **Teacherless on-policy distillation for TVG**: replace external-teacher OPD (Video-OPD) with a feedback-conditioned self-teacher (SDPO) using only the IoU verifier.
- **Coarse feedback to avoid leakage**: make IoU-only feedback the primary condition; use richer boundary-direction feedback only as a diagnostic ablation.
- **Verification-ready evaluation**: use TimeLens-Bench (public, cleaned TVG benchmark) with standard metrics and an early refutation gate.

---

## Related Work

### Field Overview

TVG has transitioned from specialized temporal localization architectures to post-training video-language models. **[TimeLens](./references/TimeLens-Rethinking-Video-Temporal-Grounding-with-Multimodal-LLMs/meta/meta_info.txt)** (arXiv:2512.14698) introduces TimeLens-Bench (cleaned evaluation) and TimeLens-100K (training data) and studies RLVR recipes for TVG. Time-R1 and related works apply GRPO-style RLVR for TVG but incur multi-rollout overhead.

Video-OPD reframes TVG post-training as on-policy distillation with dense token-level signals from a teacher. In parallel, SDPO and other on-policy/self-distillation methods show that rich verifier feedback can be used to form self-teachers in verifiable domains. This proposal combines these lines of work by using IoU-based feedback as the “rich feedback” signal in TVG.

### Related Papers

- **[TimeLens](./references/TimeLens-Rethinking-Video-Temporal-Grounding-with-Multimodal-LLMs/meta/meta_info.txt)**: Introduces TimeLens-Bench/100K and RLVR recipes for TVG; provides a training-time anchor (Table 3).
- **[Video-OPD](./references/Video-OPD-Efficient-Post-Training-of-Multimodal-Large-Language-Models-for-Temporal-Video-Grounding-via-On-Policy-Distillation/meta/meta_info.txt)**: Teacher-based on-policy distillation for TVG with token-level reverse-KL supervision.
- **[SDPO](./references/Reinforcement-Learning-via-Self-Distillation/meta/meta_info.txt)**: Feedback-conditioned self-teacher distillation for RLVR; shows low overhead vs GRPO (Section 2.2).
- **[Veto](./references/Stable-On-Policy-Distillation-through-Adaptive-Target-Reformulation/meta/meta_info.txt)**: Stabilizes on-policy distillation when the teacher-student gap is large.
- **[π-Distill / OPSD](../../papers/paper_summaries/Privileged%20Information%20Distillation%20for%20Language%20Models.md)**: Parameter-sharing distillation where teacher has privileged information not available at inference.
- **[G-OPD / ExOPD](../../papers/paper_summaries/Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md)**: Theoretical view of OPD and reward scaling to exceed teacher performance.
- **[Self-Distilled Reasoner](../../papers/paper_summaries/Self-Distilled%20Reasoner%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models.md)**: On-policy self-distillation where the teacher is the same model conditioned on privileged solution context.
- **[Time-R1](https://arxiv.org/abs/2503.13377)**: GRPO-style RLVR post-training for TVG with timestamp-aware rewards.
- **[Datasets and Recipes for Video Temporal Grounding via Reinforcement Learning (VTG-R1)](https://arxiv.org/abs/2507.18100)**: Releases curated cold-start + RL datasets and GRPO recipe for VTG.
- **[VideoTG-R1](https://arxiv.org/abs/2510.23397)**: Curriculum RL for VTG addressing partially annotated and hard samples.
- **[Temporal Preference Optimization (TPO)](https://arxiv.org/abs/2501.13919)**: Preference optimization to improve temporal understanding in long-form video tasks.
- **[TimeRefine](https://arxiv.org/abs/2412.09601)**: Iterative refinement formulation for timestamp prediction in video grounding.
- **[ED-VTG](https://arxiv.org/abs/2510.17023)**: Query enrichment + detection-style temporal grounding with multimodal LLMs.
- **[ViTED](https://arxiv.org/abs/2503.12855)**: Distills temporally grounded evidence chains for complex video QA.
- **[TAR-TVG](https://arxiv.org/abs/2508.07683)**: Timestamp-anchor constrained reasoning with GRPO+SFT for TVG.
- **[MiMo-V2-Flash](https://arxiv.org/abs/2601.02780)**: Multi-teacher on-policy distillation at scale for a MoE LLM.
- **[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)**: Introduces GRPO, a widely used RLVR optimizer.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: Demonstrates large-scale RLVR can induce strong reasoning behaviors.
- **[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Motivates process/dense feedback vs outcome-only signals.
- **[PLUM](https://arxiv.org/abs/2406.06887)**: Execution-guided on-policy preference learning for code using test cases.
- **[AutoIF](https://arxiv.org/abs/2406.13542)**: Generates instruction-following data via executable verifiers.
- **[Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)**: Base model family with long-context video support used by many TVG works.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| GRPO-style RLVR for TVG | Multi-rollout policy gradient with scalar IoU reward | TimeLens, Time-R1, VideoTG-R1 | TimeLens-Bench, Charades-STA, ActivityNet-Captions | Sparse credit assignment; rollout cost scales with video length |
| Teacher-based on-policy distillation | Single rollout + teacher logprobs yield dense token-level signals | Video-OPD, MiMo-V2-Flash, G-OPD | TimeLens-Bench | Requires teacher logprobs / teacher availability |
| Feedback-conditioned self-distillation | Use the same model with privileged feedback as a teacher | SDPO, Self-Distilled Reasoner, π-Distill | Code/math RLVR, agents | Requires model can use feedback in-context |

### Closest Prior Work

1) **Video-OPD (arXiv:2602.02994)**
- What it does: Uses a fixed teacher to compute token-level reverse-KL signals on student on-policy rollouts for TVG.
- Key limitation for our question: Requires an external teacher with accessible token log-probabilities.
- Why different: We replace the teacher with a self-teacher conditioned on verifier feedback.

2) **SDPO (arXiv:2601.20802)**
- What it does: Uses a self-teacher conditioned on rich feedback (runtime errors, judge text) to create token-level distillation loss.
- Key limitation for our question: Not evaluated on multimodal long-context TVG.
- Why different: We instantiate feedback from IoU and test whether it can replace teacher distillation.

3) **TimeLens (arXiv:2512.14698)**
- What it does: Provides cleaned TVG benchmarks and RLVR recipes; reports training-time ratios for TVG.
- Key limitation: Uses GRPO and does not study teacherless token-level distillation.
- Why different: We aim to reduce rollout cost by replacing sparse-reward GRPO with dense self-distillation.

4) **Veto (arXiv:2601.07155)**
- What it does: Stabilizes on-policy distillation when student-teacher gaps are large.
- Key limitation: Still assumes a teacher distribution.
- Why different: If needed, we use Veto to stabilize a self-teacher rather than an external teacher.

**Novelty Kill Search Summary:** Searched for “SDPO temporal video grounding”, “on-policy self-distillation temporal video grounding”, “self-teacher temporal video grounding”, “feedback-conditioned distillation video”, “teacherless on-policy distillation video”, and checked local drafts/finalized proposals. No prior work applying SDPO-style feedback-conditioned self-distillation to TimeLens-Bench TVG was found as of 2026-02-16. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Video-OPD | Teacher logprobs → dense token-level signal on student rollouts | Needs teacher logprobs | Replace teacher with feedback-conditioned self-teacher | TVG has an objective verifier (IoU) that can be converted into feedback tokens |
| GRPO (TimeLens/Time-R1) | Multi-rollout RLVR with scalar IoU reward | Sparse credit assignment; expensive rollouts | Use single-rollout distillation loss | Dense signal should reduce variance and rollout cost |
| SDPO | Self-teacher from rich feedback in text/code domains | Not tested in video long-context grounding | Instantiate feedback from IoU | If VLMs can self-correct from IoU feedback, self-teacher approximates a better policy |
| Veto | Stabilize on-policy distillation | Teacher still required | Optional stabilization for SDPO-TVG | May prevent collapse if feedback-conditioned teacher becomes too sharp |

---

## Experiments

### Experimental Setup

**Pre-check (must pass to proceed): zero-shot feedback sanity check.** On **200** held-out training examples from TimeLens-100K:
1) run the base model to predict timestamps,
2) compute IoU feedback from ground truth,
3) reprompt the same model with the feedback and ask for a revised prediction.

If mean IoU does not improve, we refute early (the self-teacher bootstrap assumption fails).

**Training protocol.** Follow Video-OPD’s small-budget regime (see Video-OPD Section D.1): select **2,500** training instances from TimeLens-100K via difficulty filtering, and train for one epoch with batch size 32, learning rate 1e-6, and max input video tokens 8192 (LoRA fine-tuning).

**Baseline Ladder (REQUIRED):**
- Prompting baseline: Qwen3-VL-8B-Instruct zero-shot on Charades-TimeLens.
- Inference-time scaling baseline: best-of-N prompting (N=8); report Recall@0.7 and mIoU.
- Closest training baselines:
  - GRPO (G=8 rollouts) with scalar IoU reward (TimeLens / Video-OPD setting).
  - Teacher-based OPD with a **public teacher** (TimeLens-8B) to match model scale and remove “teacher is private” confound.
- Literature upper bound (context): published Video-OPD Round 1 result (uses a 32B GRPO teacher).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen/Qwen3-VL-8B-Instruct | 8B | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | Student / base model |
| TencentARC/TimeLens-8B | 8B | https://huggingface.co/TencentARC/TimeLens-8B | Public teacher for OPD baseline |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| TimeLens-100K | Training pool (select 2,500 examples) | 100K | https://huggingface.co/datasets/TencentARC/TimeLens-100K | BSD-3-Clause (per dataset card) |
| TimeLens-Bench | Evaluation benchmark | ~9k annotations | https://huggingface.co/datasets/TencentARC/TimeLens-Bench | BSD-3-Clause (per HF metadata; verifier should confirm) |

**Other Resources (if applicable):**
- TimeLens evaluation script: https://github.com/TencentARC/TimeLens (scripts/eval_timelens_bench.sh)

**Resource Estimate** (upper bounds; capped to fit budget):
- TimeLens reports training time on **8×H20**, where **1.0× ≈ 4h10m** for their thinking-free RLVR (Table 3, included time for offline difficulty inference).
- SDPO reports that overhead vs GRPO is small because it only adds a parallelized log-prob pass (SDPO Section 2.2).
- Budget plan (3 seeds, early stopping enabled):
  - SDPO-TVG (G=1 + extra logprob pass): ≤ 120 A100-hours total
  - Teacher-OPD (TimeLens-8B teacher; G=1 + teacher logprob pass): ≤ 150 A100-hours total
  - GRPO baseline (G=8): cap at ≤ 350 A100-hours total via early stopping / step cap
  - Total: ≤ 700 A100-hours including evaluation

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Charades-TimeLens | Cleaned Charades temporal grounding benchmark (video+query→timestamps) | Recall@0.3/0.5/0.7, mIoU | test | https://huggingface.co/datasets/TencentARC/TimeLens-Bench | TimeLens repo eval script |

**Metric definitions (higher is better):**
- **Recall@τ**: fraction of examples where the predicted interval has IoU ≥ τ with ground truth.
- **mIoU**: mean IoU over examples.

### Main Results

#### Results Table

Baseline numbers are copied verbatim from proposal-local raw artifacts in Video-OPD.

| Method | Base Model | Benchmark | Recall@0.7 (mean±std) | mIoU (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Zero-shot | Qwen3-VL-8B-Instruct | Charades-TimeLens | 23.1 (1 run) | 42.9 (1 run) | [Video-OPD table](./references/Video-OPD-Efficient-Post-Training-of-Multimodal-Large-Language-Models-for-Temporal-Video-Grounding-via-On-Policy-Distillation/sections/Charades-TimeLens_1.md) | Published baseline |
| Best-of-8 prompting | Qwen3-VL-8B-Instruct | Charades-TimeLens | **TBD** | **TBD** | - | Needs re-run (inference only) |
| GRPO (G=8) | Qwen3-VL-8B-Instruct | Charades-TimeLens | 27.6 (1 run) | **TBD** | [Video-OPD table](./references/Video-OPD-Efficient-Post-Training-of-Multimodal-Large-Language-Models-for-Temporal-Video-Grounding-via-On-Policy-Distillation/sections/Charades-TimeLens.md) | Published Recall@0.7; mIoU not reported in this table |
| Teacher-OPD (public teacher) | Qwen3-VL-8B-Instruct | Charades-TimeLens | **TBD** | **TBD** | - | Needs re-run; teacher = TimeLens-8B |
| Video-OPD (Round 1; 32B teacher) | Qwen3-VL-8B-Instruct | Charades-TimeLens | 32.4 (1 run) | 52.0 (1 run) | [Video-OPD table](./references/Video-OPD-Efficient-Post-Training-of-Multimodal-Large-Language-Models-for-Temporal-Video-Grounding-via-On-Policy-Distillation/sections/Charades-TimeLens_1.md) | Literature upper bound; teacher checkpoint may be unavailable |
| TimeLens-8B (released model) | TimeLens-8B | Charades-TimeLens | 33.4 (1 run) | 53.3 (1 run) | [Video-OPD table](./references/Video-OPD-Efficient-Post-Training-of-Multimodal-Large-Language-Models-for-Temporal-Video-Grounding-via-On-Policy-Distillation/sections/Charades-TimeLens_1.md) | Context: full TVG-tuned model |
| **Ours: SDPO-TVG (IoU-only feedback)** | Qwen3-VL-8B-Instruct | Charades-TimeLens | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| SDPO-TVG + boundary-direction feedback | Add `(start_rel, end_rel)` tokens | If it greatly outperforms IoU-only, gains may be driven by overly informative feedback rather than by the self-distillation mechanism |

### Experimental Rigor

- **Seeds**: run all training-based conditions with `seeds=[42, 123, 456]`; report mean±std.
- **Sanity checks**:
  - Zero-shot feedback sanity check (200 examples) must improve mean IoU; otherwise refute early.
  - Reproduce one published baseline (zero-shot Recall@0.7=23.1 on Charades-TimeLens).
- **Validity threats and controls**:
  - **Feedback leakage**: make IoU-only the main condition; treat boundary-direction feedback as a diagnostic ablation.
  - **Train/test contamination**: follow TimeLens-100K training split and evaluate only on TimeLens-Bench test split.
  - **Prompt sensitivity**: use the same timestamp format and system prompt across all conditions.
- **Fair comparison**: match max video tokens (8192), FPS (2), frame cap, and output format across conditions.

---

## Success Criteria

**Hypothesis** (directional): SDPO-TVG with IoU-only feedback provides a usable self-teacher signal for TVG and approaches teacher-based OPD quality while avoiding multi-rollout GRPO cost.

**Decision Rule** (concrete):
- **Proceed** if (i) the zero-shot feedback sanity check improves mean IoU on 200 examples, and (ii) SDPO-TVG (IoU-only) achieves Recall@0.7 ≥ GRPO and is within 1 point of the teacher-OPD (public teacher) baseline across 3 seeds.
- **Pivot** if IoU-only passes the sanity check but falls short of teacher-OPD while the boundary-direction ablation succeeds; reinterpret the result as “directional feedback is required” and redesign feedback to avoid answer leakage (outside the scope of this verification).
- **Refute** if the sanity check fails, or if SDPO-TVG (IoU-only) is consistently worse than GRPO across seeds.

---

## Impact Statement

If successful, SDPO-TVG would make on-policy distillation for temporal video grounding usable without an external teacher model, reducing the cost and engineering barriers for post-training long-context video-language models in settings where only a verifier (ground-truth timestamps) is available.

---

## References

- [Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation](./references/Video-OPD-Efficient-Post-Training-of-Multimodal-Large-Language-Models-for-Temporal-Video-Grounding-via-On-Policy-Distillation/meta/meta_info.txt)
- [Reinforcement Learning via Self-Distillation (SDPO)](./references/Reinforcement-Learning-via-Self-Distillation/meta/meta_info.txt)
- [TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](./references/TimeLens-Rethinking-Video-Temporal-Grounding-with-Multimodal-LLMs/meta/meta_info.txt)
- [Stable On-Policy Distillation through Adaptive Target Reformulation (Veto)](./references/Stable-On-Policy-Distillation-through-Adaptive-Target-Reformulation/meta/meta_info.txt)
- [Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942)
- [Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125)
- [Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding](https://arxiv.org/abs/2503.13377)
- [Datasets and Recipes for Video Temporal Grounding via Reinforcement Learning](https://arxiv.org/abs/2507.18100)
- [VideoTG-R1: Boosting Video Temporal Grounding via Curriculum Reinforcement Learning](https://arxiv.org/abs/2510.23397)
- [Temporal Preference Optimization for Long-Form Video Understanding](https://arxiv.org/abs/2501.13919)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [PLUM: Preference Learning Plus Test Cases Yields Better Code Language Models](https://arxiv.org/abs/2406.06887)
- [AutoIF](https://arxiv.org/abs/2406.13542)
