# untitled

# Preventing Visual Forgetting with Time-Varying Mutual-Information Decoding in Long-CoT VLMs

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Vision-language models (VLMs) are increasingly trained and prompted to generate long, step-by-step reasoning traces (chain-of-thought) for challenging multimodal tasks such as chart interpretation and visual question answering. In text-only language models, longer reasoning is often a reliable test-time scaling strategy. However, recent evidence suggests this does not transfer cleanly to multimodal settings.

**More Thought, Less Accuracy?** reports a counterintuitive failure mode: as multimodal reasoning proceeds, models can become less visually grounded and make more perception errors, even on questions that should be straightforward given the image ([MORE THOUGHT, LESS ACCURACY…](./references/MORE-THOUGHT-LESS-ACCURACY-ON-THE-DUAL-NATURE-OF-REASONING-IN-VISION-LANGUAGE-MODELS/meta/meta_info.txt)). They attribute this to **visual forgetting**: the model’s reliance on visual tokens (measured via attention) declines with generation length, and inference-time remedies such as **visual replay** (re-inserting the image during reasoning) can partially recover accuracy.

The practical problem is that the strongest existing fixes are either:
- **Training-time** (e.g., reinforcement learning or special training recipes that explicitly reward visual grounding), which is expensive and model-specific; or
- **Inference-time visual replay**, which changes the input sequence and requires multi-image handling and repeated visual conditioning.

If an inference-time decoding-only method could prevent visual forgetting without any additional training, it would be a low-friction option for practitioners deploying open VLMs.

### The Problem

We focus on **visual forgetting during long reasoning**, operationalized as:
1) **Accuracy degradation with longer reasoning** on vision-intensive benchmarks (e.g., MMStar, HallusionBench), and
2) **Collapse of visual dependence** during generation.

A key observation from mutual-information-based decoding work is that many VLM grounding failures can be described as **conditioning dilution** (a progressive loss of sensitivity to the image during autoregressive generation): as generation progresses, the next-token distribution conditioned on the image becomes increasingly similar to the distribution produced by a language-only (image-masked) model with the same text prefix.

M3ID (Multi-Modal Mutual-Information Decoding) formalizes this as a fading-memory interpolation between an image-conditioned policy and an image-masked policy, and proposes an inference-time correction that grows with generation step ([Multi-Modal Hallucination Control…](./references/Multi-Modal-Hallucination-Control-by-Visual-Information-Grounding-2403.14003/meta/meta_info.txt)). This was developed for hallucination reduction in captioning/VQA, not for long-CoT reasoning where visual forgetting is explicitly tied to reasoning length.

Meanwhile, recent long-CoT visual-grounding fixes (e.g., TVC and RL-based visual-attention rewards) mainly change training rather than decoding ([TVC](./references/Mitigating-Visual-Forgetting-via-Take-along-Visual-Conditioning-for-Multimodal-Long-CoT-Reasoning-2503.13360/meta/meta_info.txt); [Reflection-V](https://arxiv.org/abs/2509.12132)). The open question is whether **time-varying MI decoding**, without any training or image reinsertion, can meaningfully reduce visual-forgetting-induced errors in long reasoning.

### Key Insight and Hypothesis

**Key insight:** Visual forgetting can be viewed as the progressive convergence of the image-conditioned next-token distribution toward an image-masked distribution given the same text prefix. If this convergence holds empirically, then MI decoding’s correction term (which extrapolates away from the image-masked distribution) should directly counteract visual forgetting.

**Hypothesis:** On vision-intensive reasoning benchmarks where longer reasoning causes accuracy to plateau or decline, **adaptive MI decoding** (M3ID) will:
1) reduce **correct→wrong flips** when increasing the reasoning budget (short vs long), and
2) measurably slow or prevent the decline of a visual-dependence proxy (Prompt Dependency Measure; PDM) over generation steps,
compared to vanilla decoding. A constant-strength MI correction (fixed \(\gamma\), equivalent to time-independent contrastive decoding) will underperform adaptive MI decoding, indicating that the time-varying schedule is a key mechanism.

Why this could be wrong: (i) “forgetting” may not correspond to convergence toward the image-masked distribution; the model may instead form a corrupted visual state that diverges from both conditioned and masked passes; (ii) MI decoding may overcompensate and suppress tokens that are predictable from text but still correct, harming accuracy; (iii) visual replay may work via re-conditioning that decoding alone cannot match.

---

## Proposed Approach

### Overview

We propose to apply **time-varying mutual-information calibrated decoding** (M3ID-style) as an inference-time remedy for visual forgetting in long chain-of-thought generation.

At each generation step, we compute:
- \(l_c\): logits from the standard image-conditioned forward pass.
- \(l_u\): logits from an **image-masked** forward pass intended to approximate the language-only prior under the same text prefix.

We then decode greedily from a corrected distribution \(\hat l\) that increasingly emphasizes the difference \(l_c - l_u\) as generation proceeds.

### Method Details

#### Base model interface and the “image-masked” pass
We target Qwen2.5-VL–style models and derivatives (e.g., VLAA-Thinker) that encode an image into a sequence of visual embeddings and then fuse them into the language model.

To define the image-masked model \(p(\cdot|x, y_{<t})\) in a way that preserves token positions and attention structure, we follow M3ID’s paper definition (“VLM with masked visual tokens”) by **zeroing the projected visual embeddings** *after* the vision encoder / projector and *before* fusion with the language model. This keeps the number and positions of visual tokens constant while removing visual information.

Sanity checks (automated):
- The image-masked model should still generate fluent text.
- On MMStar multiple-choice questions, the image-masked model’s accuracy should be near chance.

#### Adaptive MI decoding rule
We implement M3ID’s greedy decoding update (notation aligned to the paper):

- Let \(p_c = \mathrm{softmax}(l_c)\). Apply correction only when the model is not already highly confident:
\[
\hat l = l_c + \mathbb{1}[\max_k p_c(k) < \alpha] \cdot \frac{1-\gamma_t}{\gamma_t} (l_c - l_u).
\]
- Use \(\gamma_t = \exp(-\lambda (t + t_0))\), where \(t\) is the generated-token index and \(t_0\) is an offset to avoid early overcorrection before the model begins producing the answer.

Default hyperparameters (from M3ID’s reported best settings): \(\lambda=0.02\), \(\alpha=0.3\). We set \(t_0\) to the number of tokens in the question prompt (approximate) so the correction ramps up primarily during the answer/reasoning portion.

#### Mechanistic ablation (time-independent MI correction)
To test whether the time-varying schedule matters, we include a fixed-strength MI decoding ablation equivalent to time-independent contrastive decoding:
- Set \(\gamma_t = \gamma_{\mathrm{fixed}} = 0.5\) for all \(t\), yielding \(\hat l = 2 l_c - l_u\) when the confidence gate is active.

### Key Innovations

1) **Problem transfer with a falsifiable mechanism:** Apply MI decoding (previously used mainly for hallucination mitigation) to long-CoT **visual forgetting**, with an explicit mechanistic prediction: the conditioned distribution should converge toward the image-masked distribution as reasoning lengthens, and adaptive MI decoding should counteract that convergence.

2) **Decision-changing diagnostic:** In addition to accuracy, evaluate a **flip-rate** metric (short-budget correct, long-budget wrong) to isolate the “overthinking harms perception” regime where visual forgetting is most relevant.

---

## Related Work

### Field Overview

This proposal connects three research threads:

1) **Visual forgetting in long-CoT VLMs**: Recent reasoning-oriented VLM training pipelines show that longer reasoning can degrade perception, and training-time methods propose explicit visual-grounding rewards or periodic visual conditioning.

2) **Inference-time grounding and hallucination mitigation**: A large body of work modifies decoding or attention patterns (often with dual forward passes) to reduce hallucinations by increasing reliance on visual tokens.

3) **Benchmarks and diagnostics**: Newer benchmarks such as MMStar and HallusionBench emphasize visual dependency and expose gaps between model reasoning and perception; metrics like CHAIR/POPE/PDM quantify hallucination or visual dependence.

### Related Papers

- **[More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models](./references/MORE-THOUGHT-LESS-ACCURACY-ON-THE-DUAL-NATURE-OF-REASONING-IN-VISION-LANGUAGE-MODELS/meta/meta_info.txt)**: Identifies visual forgetting during long reasoning; introduces visual replay/focus prompt baselines and the VAPO training algorithm.
- **[Multi-Modal Hallucination Control by Visual Information Grounding](./references/Multi-Modal-Hallucination-Control-by-Visual-Information-Grounding-2403.14003/meta/meta_info.txt)**: Proposes M3ID (time-varying MI decoding) and PDM metrics for conditioning dilution in VLM generation.
- **[Mitigating Visual Forgetting via Take-along Visual Conditioning for Multimodal Long CoT Reasoning](./references/Mitigating-Visual-Forgetting-via-Take-along-Visual-Conditioning-for-Multimodal-Long-CoT-Reasoning-2503.13360/meta/meta_info.txt)**: Training + periodic visual calibration (visual token reinjection + cache reset) to maintain visual grounding over long CoT.
- **[Grounding Language with Vision: A Conditional Mutual Information Calibrated Decoding Strategy…](./references/Grounding-Language-with-Vision-A-Conditional-Mutual-Information-Calibrated-Decoding-Strategy-for-Reducing-Hallucinations-in-LVLMs-2505.19678/meta/meta_info.txt)**: Conditional-PMI calibrated decoding and visual token purification to reduce hallucinations.
- **[Look Again, Think Slowly: Enhancing Visual Reflection in Vision-Language Models](https://arxiv.org/abs/2509.12132)**: GRPO training with visual-attention rewards to sustain visual grounding during long reasoning.
- **[Don’t Miss the Forest for the Trees: Attentional Vision Calibration for LVLMs](https://arxiv.org/abs/2405.17820)**: Identifies “blind tokens” and uses contrastive decoding to recalibrate vision attention and reduce hallucinations.
- **[Visual Contrastive Decoding (VCD)](https://arxiv.org/abs/2311.16922)**: Training-free decoding contrasting original vs distorted images to mitigate object hallucinations.
- **[OPERA](https://arxiv.org/abs/2311.17911)**: Training-free decoding with an over-trust penalty and rollback to reduce hallucinations in LVLMs.
- **[LURE](https://arxiv.org/abs/2310.00754)**: Post-hoc hallucination revisor using uncertainty/position/co-occurrence signals.
- **[HallusionBench](https://arxiv.org/abs/2310.14566)**: Diagnostic benchmark for visual illusion and language hallucination in LVLMs.
- **[MMStar](https://arxiv.org/abs/2403.20330)**: Vision-indispensable benchmark with explicit leakage/gain metrics.
- **[BLINK](https://arxiv.org/abs/2404.12390)**: Perception-focused benchmark showing models can “see but not perceive”.
- **[POPE](https://arxiv.org/abs/2305.10355)**: Binary VQA benchmark for object hallucinations.
- **[CHAIR](https://aclanthology.org/P18-1250/)**: Caption hallucination metric based on MS-COCO object annotations.
- **[AMBER](https://arxiv.org/abs/2311.07397)**: LLM-free multi-dimensional hallucination benchmark for MLLMs.
- **[InstructBLIP](https://arxiv.org/abs/2305.06500)**: Instruction-tuned vision-language model; common baseline in hallucination/grounding work.
- **[LLaVA-1.5](https://arxiv.org/abs/2305.04790)**: Widely used open LVLM baseline.
- **[LLaVA-NeXT / LLaVA-1.6](https://arxiv.org/abs/2310.03744)**: Updated LLaVA family with improved resolution handling.
- **[Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)**: Describes the Qwen2.5-VL architecture and training recipe used by many recent reasoning-capable VLMs.
- **[Vision-R1](https://arxiv.org/abs/2503.06749)**: RL-based multimodal reasoning model; used in visual forgetting studies.
- **[R1-OneVision](https://arxiv.org/abs/2503.10615)**: Qwen2.5-VL-based reasoning model family with SFT+RL variants.
- **[OpenVLThinker](https://arxiv.org/abs/2503.17352)**: Iterative SFT–RL cycles for multimodal reasoning, highlighting data/format issues.
- **[MM-Eureka](https://arxiv.org/abs/2503.07365)**: Rule-based RL for multimodal reasoning with emergent “visual aha” behaviors.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Long-CoT visual-grounding (training) | Train/RL models to keep attending to visual tokens during long reasoning | VAPO (2509.25848), TVC (2503.13360), Reflection-V (2509.12132) | MMStar, HallusionBench, MathVista/MathVision | Expensive training; model-specific pipelines |
| Visual replay / periodic conditioning (inference) | Re-insert image (or compressed image tokens) during reasoning | Visual replay (2509.25848), PVC (TVC) | MMStar, HallusionBench, MathVista | Requires multi-image handling; changes context length |
| MI / contrastive decoding (inference) | Modify token probabilities using conditioned vs unconditioned (or distorted) logits | M3ID (2403.14003), CMI-VLD (2505.19678), VCD (2311.16922), AvisC (2405.17820) | POPE, CHAIR/COCO, MME, HallusionBench | Often tuned for hallucination; unclear transfer to long reasoning |

### Closest Prior Work

- **M3ID (2403.14003)**: Introduces time-varying MI decoding and PDM-based conditioning-dilution analysis, but evaluates mainly on captioning/POPE-style hallucination settings rather than long-CoT reasoning tasks.
- **CMI-VLD (2505.19678)**: Uses conditional-MI calibrated decoding and visual token purification for hallucination reduction; does not study visual forgetting as a function of reasoning length.
- **More Thought, Less Accuracy? / VAPO (2509.25848)**: Establishes visual forgetting during reasoning and proposes visual replay/focus prompts plus a training-time RL fix; does not test MI decoding as an inference-time remedy.
- **TVC (2503.13360)**: Uses periodic visual conditioning and training to prevent forgetting; differs from our decoding-only approach.

**Novelty Kill Search Summary:** On 2026-02-26, we searched for the exact combination of “visual forgetting” + “mutual information decoding / PMI decoding / M3ID / CMI-VLD” and for “visual replay + mutual information decoding” (8 distinct queries; log in `notes.md`), and scanned local proposal indexes/drafts for `M3ID`, `2403.14003`, and `visual replay`. We found MI-decoding papers targeting hallucination (M3ID, CMI-VLD) and long-CoT grounding papers using training or replay (VAPO/TVC/Reflection-V), but no prior work explicitly evaluating **MI decoding as a decoding-only remedy for long-CoT visual forgetting**.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| M3ID (2403.14003) | Time-varying MI decoding to reduce hallucinations | Not evaluated on long-CoT visual forgetting | Apply to long-CoT reasoning; add flip-rate + PDM-vs-step diagnostics | Visual forgetting matches M3ID’s conditioning-dilution mechanism |
| CMI-VLD (2505.19678) | Conditional-MI calibrated decoding + token purification | Targets hallucinations, not long-CoT degradation | Focus on reasoning-length-induced degradation; mechanistic ablation (fixed vs adaptive) | If p_c→p_u with length, MI correction should directly counteract |
| Visual replay (2509.25848) | Reinsert image 4× during reasoning | Changes context; requires multi-image support | Keep input fixed; only adjust decoding distribution | Decoding-only solution is simpler to deploy if effective |
| TVC (2503.13360) | Training + periodic visual calibration | Requires training; complex pipeline | No training; reuse off-the-shelf checkpoints | Lower barrier to adoption |

---

## Experiments

### Experimental Setup

**Task framing:** We evaluate on vision-intensive benchmarks where the VAPO paper reports that longer reasoning can reduce accuracy and visual replay improves performance.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| VLAA-Thinker (Qwen2.5-VL backbone) | 7B | https://huggingface.co/UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B | Primary base model; matches published VR/FP baselines in 2509.25848.
| Qwen2.5-VL-Instruct (optional replication) | 7B | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | Secondary check for generality on a small subset of MMStar.

**Baseline Ladder (REQUIRED):**
- **Prompting baseline (simple):** Vanilla reasoning prompt (greedy decoding).
- **Prompting baseline (published remedy):** Focus prompt (insert one fixed “look back” instruction 4×) from 2509.25848.
- **Inference-time scaling baseline:** Short vs long reasoning budget (`max_new_tokens` 128 vs 512) to measure flip-rate under increased test-time compute.
- **Closest existing method:** Visual replay (reinsert downsampled image 4×) from 2509.25848.
- **Ours (main):** Adaptive MI decoding (M3ID).
- **Mechanistic ablation (subset only):** Fixed-\(\gamma\) MI decoding on 200 random MMStar items to test whether the time-varying schedule is necessary.

**Core comparison (decisive):** vanilla vs visual replay vs adaptive MI on full MMStar + HallusionBench under the long budget; focus prompt and fixed-\(\gamma\) are auxiliary checks.

**Implementation notes (critical for reproducibility):**
- Use greedy decoding for all conditions (deterministic; no seed variance expected).
- Visual replay: insert the same image 4 times, downsampled, at approximately evenly spaced points in the reasoning trace and aligned to punctuation boundaries when possible (Appendix A.5 of 2509.25848).
- Image-masked forward pass for MI decoding: zero projected visual embeddings before fusion (primary); fall back to blank-image only if the masked model degenerates.

**Resource Estimate** (rough, to be verified during implementation):
- **Compute budget**: 50–200 A100 GPU-hours.
  - Dominant cost is long-sequence generation on MMStar (1.5k items) with dual forward passes for MI decoding.
  - All methods are inference-only; no training.
- **GPU memory**: 1×A100-80GB sufficient for 7B VLM inference; 2–4 GPUs recommended for throughput.
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MMStar | Vision-indispensable multiple-choice benchmark (1,500 items) | Accuracy | test | https://huggingface.co/datasets/Lin-Chen/MMStar | VLMEvalKit or official MMStar repo |
| HallusionBench | Diagnostic benchmark for visual illusion and language hallucination | aAcc / accuracy | official | https://github.com/tianyi-lab/HallusionBench | `evaluation.py` in repo |

**Additional mechanistic metric (subset-based):**
- **PDM-H vs step:** Hellinger distance between \(p_c(\cdot|x,c,y_{<t})\) and \(p_u(\cdot|x,y_{<t})\) over generation steps (computed from logits already produced by MI decoding). We compute this on a random subset (e.g., 50 items per benchmark) and at a sparse grid of steps (e.g., every 10 tokens) to keep overhead low.

**Flip-rate metric (all items):**
- Run each method under two budgets: \(T_{short}=128\) and \(T_{long}=512\) new tokens, with identical prompting and answer extraction.
- Define a flip as: short-budget answer is correct AND long-budget answer is incorrect.
- Also report the reverse flip (short wrong → long correct) to separate “overthinking harm” from “more compute helps”.

### Main Results

#### Results Table

(For baselines, numbers below are copied from Table 1–2 of 2509.25848; our rows are TBD.)

| Method | Base Model | Benchmark | Accuracy / aAcc (%) | Source | Notes |
|---|---|---|---:|---|---|
| Vanilla reasoning | VLAA-Thinker-7B | MMStar | 49.7 | [2509.25848 Tables 1–2](./references/MORE-THOUGHT-LESS-ACCURACY-ON-THE-DUAL-NATURE-OF-REASONING-IN-VISION-LANGUAGE-MODELS/sections/MAIN%20RESULTS.md) | Published (1 run)
| Focus prompt (FP) | VLAA-Thinker-7B | MMStar | 51.1 | [2509.25848 Tables 1–2](./references/MORE-THOUGHT-LESS-ACCURACY-ON-THE-DUAL-NATURE-OF-REASONING-IN-VISION-LANGUAGE-MODELS/sections/MAIN%20RESULTS.md) | Published (1 run)
| Visual replay (VR) | VLAA-Thinker-7B | MMStar | 52.9 | [2509.25848 Tables 1–2](./references/MORE-THOUGHT-LESS-ACCURACY-ON-THE-DUAL-NATURE-OF-REASONING-IN-VISION-LANGUAGE-MODELS/sections/MAIN%20RESULTS.md) | Published (1 run)
| **Ours: adaptive MI decoding** | VLAA-Thinker-7B | MMStar | **TBD** | - | To be verified
| Ours (fixed-\(\gamma\) ablation) | VLAA-Thinker-7B | MMStar | **TBD** | - | To be verified
| Vanilla reasoning | VLAA-Thinker-7B | HallusionBench | 54.7 | [2509.25848 Tables 1–2](./references/MORE-THOUGHT-LESS-ACCURACY-ON-THE-DUAL-NATURE-OF-REASONING-IN-VISION-LANGUAGE-MODELS/sections/MAIN%20RESULTS.md) | Published (1 run)
| Focus prompt (FP) | VLAA-Thinker-7B | HallusionBench | 55.2 | [2509.25848 Tables 1–2](./references/MORE-THOUGHT-LESS-ACCURACY-ON-THE-DUAL-NATURE-OF-REASONING-IN-VISION-LANGUAGE-MODELS/sections/MAIN%20RESULTS.md) | Published (1 run)
| Visual replay (VR) | VLAA-Thinker-7B | HallusionBench | 56.2 | [2509.25848 Tables 1–2](./references/MORE-THOUGHT-LESS-ACCURACY-ON-THE-DUAL-NATURE-OF-REASONING-IN-VISION-LANGUAGE-MODELS/sections/MAIN%20RESULTS.md) | Published (1 run)
| **Ours: adaptive MI decoding** | VLAA-Thinker-7B | HallusionBench | **TBD** | - | To be verified
| Ours (fixed-\(\gamma\) ablation) | VLAA-Thinker-7B | HallusionBench | **TBD** | - | To be verified

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Adaptive MI (full) | \(\gamma_t\) decays with \(t\) | Best flip-rate reduction; slows PDM-H collapse |
| Fixed-\(\gamma\) | \(\gamma_t\) constant (time-independent contrastive decoding) | Smaller gains if time-varying schedule is essential |
| No-confidence-gate (optional) | Always apply correction (remove \(\alpha\) gate) | Likely harms fluency/accuracy (overcompensation), supporting need for gate |

### Experimental Rigor

- **Determinism / variance**: Use greedy decoding and deterministic insertion schedules; results should be deterministic (no multi-seed averaging required). If any components are stochastic (e.g., prompt choice for focus prompts), fix a single prompt string for all examples.
- **Key confounders and controls**:
  1) **Masked-vision definition confound**: If the image-masked model is ill-defined (degenerate text), MI decoding is meaningless. Control via fluency sanity check + chance-level accuracy check for masked-only on MMStar.
  2) **Implementation mismatch with published VR/FP**: Reproduce VR/FP on a small subset and verify they are close to published values (within ~1–2 points) before interpreting comparisons.
  3) **“Sampling solves it” confound**: Because we use greedy decoding, improvements cannot be attributed to best-of-N selection. (Optional: add a small best-of-4 majority-vote baseline if time permits.)

---

## Success Criteria

### Hypothesis
Adaptive MI decoding will (i) improve MMStar/HallusionBench accuracy over vanilla decoding and (ii) reduce correct→wrong flips when increasing the reasoning budget, while showing a slower decline of PDM-H over generation steps than vanilla. Fixed-\(\gamma\) MI decoding will provide smaller or no improvements.

### Decision Rule

- **Proceed / publish direction** if, on VLAA-Thinker-7B:
  - Adaptive MI decoding improves accuracy over vanilla by **≥ +1.0 point** on **both** MMStar and HallusionBench **and**
  - reduces flip rate by **≥ 20% relative** (or ≥ 2 points absolute) compared to vanilla, and
  - increases the area-under-curve of PDM-H vs step (subset metric) relative to vanilla.
- **Pivot** if accuracy gains are < +1 point but PDM-H collapse is clearly slowed: try CMI-VLD-style conditional-PMI calibration or a learned visual-token purifier with <1% parameters.
- **Refute** if PDM-H does not systematically decline with generation length on vanilla (mechanism falsified), or if adaptive MI decoding underperforms vanilla or visual replay on both benchmarks.

---

## Impact Statement

If MI decoding can mitigate visual forgetting without any training, practitioners deploying open VLMs for long-form multimodal reasoning (e.g., document QA, chart analysis, multimodal tutoring) could adopt a decoding-only patch to improve visual grounding. A strong negative result would also be decision-changing: it would suggest that visual replay or training-time grounding rewards are necessary, and that MI-decoding-style hallucination fixes do not transfer to long-CoT visual forgetting.

---

## References

- [MORE THOUGHT, LESS ACCURACY? ON THE DUAL NA-TURE OF REASONING IN VISION-LANGUAGE MODELS](./references/MORE-THOUGHT-LESS-ACCURACY-ON-THE-DUAL-NATURE-OF-REASONING-IN-VISION-LANGUAGE-MODELS/meta/meta_info.txt) - Tian et al., 2025
- [Multi-Modal Hallucination Control by Visual Information Grounding](./references/Multi-Modal-Hallucination-Control-by-Visual-Information-Grounding-2403.14003/meta/meta_info.txt) - Favero et al., 2024
- [Mitigating Visual Forgetting via Take-along Visual Conditioning for Multimodal Long CoT Reasoning](./references/Mitigating-Visual-Forgetting-via-Take-along-Visual-Conditioning-for-Multimodal-Long-CoT-Reasoning-2503.13360/meta/meta_info.txt) - Sun et al., 2025
- [Grounding Language with Vision: A Conditional Mutual Information Calibrated Decoding Strategy for Reducing Hallucinations in LVLMs](./references/Grounding-Language-with-Vision-A-Conditional-Mutual-Information-Calibrated-Decoding-Strategy-for-Reducing-Hallucinations-in-LVLMs-2505.19678/meta/meta_info.txt) - 2025
- [HallusionBench](https://arxiv.org/abs/2310.14566) - Guan et al., 2023
- [MMStar](https://arxiv.org/abs/2403.20330) - Chen et al., 2024
- [BLINK](https://arxiv.org/abs/2404.12390) - Fu et al., 2024
- [VCD](https://arxiv.org/abs/2311.16922) - Leng et al., 2023
- [OPERA](https://arxiv.org/abs/2311.17911) - Huang et al., 2023
- [LURE](https://arxiv.org/abs/2310.00754) - Zhou et al., 2023
- [AMBER](https://arxiv.org/abs/2311.07397) - Wang et al., 2023
- [InstructBLIP](https://arxiv.org/abs/2305.06500) - Dai et al., 2023
- [LLaVA-1.5](https://arxiv.org/abs/2305.04790) - Liu et al., 2023
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) - Qwen Team, 2025
- [Vision-R1](https://arxiv.org/abs/2503.06749) - Huang et al., 2025
- [R1-OneVision](https://arxiv.org/abs/2503.10615) - Yang et al., 2025
- [OpenVLThinker](https://arxiv.org/abs/2503.17352) - Deng et al., 2025
- [MM-Eureka](https://arxiv.org/abs/2503.07365) - 2025
- [Reflection-V / Look Again, Think Slowly](https://arxiv.org/abs/2509.12132) - 2025
- [AvisC](https://arxiv.org/abs/2405.17820) - 2024
