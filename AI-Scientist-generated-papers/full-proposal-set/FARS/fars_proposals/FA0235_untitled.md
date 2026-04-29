# untitled

# Re-Inked OCR Views for Robust Chart Question Answering under Visual Degradations

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Charts are a common interface for communicating quantitative evidence in scientific writing, business dashboards, and public journalism. Vision-language models (VLMs) can answer many chart questions on clean benchmark images, but real deployments often face degraded chart images (e.g., blur from scanned PDFs, pixelation from low-resolution screenshots, or low contrast from photocopies).

Recent benchmarks show that chart understanding remains both incomplete and brittle. **ChartQAPro** is a real-world chart question answering benchmark with 1,341 charts and 1,948 questions across factoid, conversational, hypothetical, fact-checking, and multiple-choice types ([ChartQAPro](./references/ChartQAPro-A-More-Diverse-and-Challenging-Benchmark-for-Chart-Question-Answering/meta/meta_info.txt)). On ChartQAPro, the best reported closed model reaches 55.81% overall accuracy (higher is better), far below the human baseline of 85.02% ([ChartQAPro main results, Table 3](./references/ChartQAPro-A-More-Diverse-and-Challenging-Benchmark-for-Chart-Question-Answering/sections/4.4%20Main%20Results.md)).

Separately, **CHART NOISe** explicitly evaluates robustness to corruptions and occlusions. It reports large drops under blur corruptions: for example, ChatGPT-4o drops from 70% on clean to 45% under major Gaussian blur, and Claude Sonnet 4 drops from 76% to 45% under major motion blur ([Losing the Plot, Table 3](./references/Losing-the-Plot-How-VLM-responses-degrade-on-imperfect-charts/sections/4%20Analysis.md)). These results suggest that robustness to realistic chart degradations is a practical bottleneck.

### The Problem

We focus on a deployment-relevant question: **given a corrupted chart image at inference time, can we cheaply recover enough textual information to improve chart QA accuracy without changing model weights?**

Many chart errors under blur/pixelation are plausibly driven by unreadable text (axis tick labels, units, legend entries, titles). Existing families address parts of this problem but do not provide a simple model-agnostic robustness wrapper:

- **Training-free cropping for visual question answering (VQA)** (e.g., ViCrop) uses model attention or gradients to select a region-of-interest crop, which can help with small details but may miss relevant text regions and does not create a global legible text channel ([ViCrop](./references/Towards-Perceiving-Small-Visual-Details-in-Zero-shot-Visual-Question-Answering-with-Multimodal-LLMs/meta/meta_info.txt)).
- **Learned crop policies** (e.g., CropVLM) train a separate cropper via reinforcement learning (GRPO; a policy-optimization method) to select zoom windows, adding training cost and introducing an extra model component that must generalize to new chart styles ([CropVLM](./references/CropVLM-Learning-to-Zoom-for-Fine-Grained-Vision-Language-Perception/meta/meta_info.txt)).
- **Attention-based evidence highlighting** (e.g., Visual Evidence Augmentation, VEA) highlights attention-derived evidence regions, but it does not explicitly repair unreadable text and typically requires access to model internals ([Seeing but Not Believing](./references/Seeing-but-Not-Believing-Probing-the-Disconnect-Between-Visual-Attention-and-Answer-Correctness-in-VLMs/meta/meta_info.txt)).

A practical alternative is to run optical character recognition (OCR) and provide extracted text to the model, but it is unclear whether current VLMs reliably use long OCR text strings, and how best to present OCR outputs to support spatial reasoning in charts.

### Key Insight and Hypothesis

**Key insight.** A VLM may fail on corrupted charts because its vision encoder cannot reliably resolve small text at its effective patch granularity, even when higher-level reasoning is intact. If we provide a second image view that contains the chart's OCR text rendered in high contrast at the same approximate locations, the model may recover text-dependent reasoning while still using the original chart image for geometry and quantitative relationships.

**Hypothesis.** For corrupted chart images, providing a VLM with an auxiliary re-inked OCR view (OCR text re-rendered as a separate image at detected locations) improves chart QA accuracy beyond (i) a compute-matched blank second image and (ii) a scrambled-text control that preserves box layout but destroys text semantics.

Why this could fail: (i) OCR on corrupted charts may be too noisy, so the auxiliary view injects incorrect labels; (ii) the VLM may not fuse information across two images; (iii) improvements may come only from extra high-contrast visual structure rather than readable text semantics. Our controls are designed to distinguish these cases.

---

## Proposed Approach

### Overview

We propose **ReInk**, a training-free inference wrapper for chart QA:

1. Input a corrupted chart image.
2. Run an OCR engine to extract (text string, bounding box, confidence).
3. Create a **re-ink canvas**: a blank white image of the same resolution where each OCR string is re-rendered inside (or near) its bounding box in a standard high-contrast font.
4. Provide the VLM with two images: (i) the corrupted chart and (ii) the re-ink canvas, with a fixed prompt that instructs it to use the second image only to read chart text.

### Method Details

**OCR extraction.** Use a standard OCR library (default: EasyOCR; chosen because it is used in CHART NOISe's occlusion generation) that returns text, bounding boxes, and confidence scores.

**Canvas rendering.** For each OCR box:
- Convert to an axis-aligned rectangle.
- Choose font size based on box height.
- Render the recognized string into the rectangle; if it overflows, shrink font size to fit.
- Optionally draw a thin bounding box outline to preserve spatial anchors.

**Scrambled-text control (semantic isolation).** To test whether gains come from correct text semantics rather than high-contrast layout cues, construct a control canvas that keeps the same bounding boxes but permutes the recognized strings across boxes.

**Prompting.** Use one fixed prompt template across all conditions, e.g.:
- "You are answering a question about a chart. Image 1 is the chart (possibly corrupted). Image 2 is an OCR text layer extracted from the same chart; use it only to read labels/ticks/legend/title. Answer with only the final answer." 

### Key Innovations

1. **Model-agnostic robustness intervention** for chart QA that requires no weight updates and no learned crop policy.
2. **Mechanism-isolating controls** (blank canvas and scrambled-text canvas) that separate gains from extra image tokens/layout strokes vs gains from readable text semantics.
3. A decision-changing outcome either way:
   - If correct re-inking helps beyond scrambled-text, it supports the claim that a repaired text channel improves robustness under corruption.
   - If scrambled-text performs similarly to correct re-inking, it suggests improvements come mainly from structural cues, which changes how multi-view robustness wrappers should be designed.

---

## Related Work

### Field Overview

Chart understanding pipelines often decompose into (i) perception of chart elements (text and marks), (ii) extraction to an intermediate representation (tables, JSON, code), and (iii) downstream reasoning (question answering and fact-checking). Many chart QA benchmarks historically used clean synthetic or templated data, but newer benchmarks emphasize real-world diversity and highlight large headroom.

In parallel, the broader VQA and document understanding literature has studied integrating OCR with neural models, either by feeding OCR tokens into a model or by using multi-view processing to improve fine-grained perception. Our proposal sits at the intersection: we use OCR as an external tool but present its output as a spatially aligned auxiliary image, aiming for a training-free, model-agnostic robustness wrapper.

### Related Papers

- **[ChartQAPro](./references/ChartQAPro-A-More-Diverse-and-Challenging-Benchmark-for-Chart-Question-Answering/meta/meta_info.txt)**: Real-world chart QA benchmark with diverse question types and substantial remaining headroom.
- **[Losing the Plot / CHART NOISe](./references/Losing-the-Plot-How-VLM-responses-degrade-on-imperfect-charts/meta/meta_info.txt)**: Introduces a corruption/occlusion robustness benchmark and shows sharp drops under blur corruptions.
- **[ChartQA](https://arxiv.org/abs/2203.10244)**: Early chart QA benchmark with both human-written and automatically generated questions.
- **[PlotQA](https://arxiv.org/abs/2008.10505)**: Benchmark for reasoning over scientific plots with multi-step numerical reasoning.
- **[DVQA](https://arxiv.org/abs/1801.08163)**: Bar-chart question answering dataset highlighting OCR and visual reasoning difficulties.
- **[FigureQA](https://arxiv.org/abs/1710.07300)**: Synthetic figure QA benchmark used in early chart reasoning work.
- **[CharXiv](https://arxiv.org/abs/2406.09233)**: Realistic chart dataset from arXiv papers emphasizing distribution shift vs synthetic charts.
- **[ChartX & ChartVLM](https://arxiv.org/abs/2402.12185)**: Benchmark and cascaded chart model that separates perception from reasoning.
- **[DePlot](https://arxiv.org/abs/2212.10505)**: Plot-to-table translation model widely used as an intermediate-representation baseline for chart reasoning.
- **[SIMPLOT](https://arxiv.org/abs/2405.00021)**: Distills essential chart information to improve chart-to-table extraction.
- **[UniChart](https://arxiv.org/abs/2305.14761)**: Chart-specific pretraining objectives for extraction and reasoning.
- **[ChartMoE](https://arxiv.org/abs/2409.03277)**: Uses multiple intermediate alignment formats (table/JSON/code) via a mixture-of-experts connector.
- **[ChartReasoner](https://arxiv.org/abs/2506.10116)**: Uses executable-code intermediates and reinforcement learning to improve chart reasoning.
- **[Chart-CoCa](https://arxiv.org/abs/2508.11975)**: Code-driven synthesis and candidate-conditioned answering for self-improvement.
- **[Visual Self-Refine / ChartVSR](https://arxiv.org/abs/2602.16455)**: Pixel-guided iterative refinement for dense chart parsing.
- **[Dragonfly](https://arxiv.org/abs/2406.00977)**: Trained multi-resolution zoom encoding for fine-grained VLM perception.
- **[CropVLM](./references/CropVLM-Learning-to-Zoom-for-Fine-Grained-Vision-Language-Perception/meta/meta_info.txt)**: Learned crop selection via reinforcement learning to improve small-detail VQA.
- **[ViCrop](./references/Towards-Perceiving-Small-Visual-Details-in-Zero-shot-Visual-Question-Answering-with-Multimodal-LLMs/meta/meta_info.txt)**: Training-free cropping using attention/gradients for small visual details.
- **[Seeing but Not Believing](./references/Seeing-but-Not-Believing-Probing-the-Disconnect-Between-Visual-Attention-and-Answer-Correctness-in-VLMs/meta/meta_info.txt)**: Shows attention-correctness disconnect and proposes Visual Evidence Augmentation for text-heavy multimodal tasks.
- **[TextVQA / LoRRA](https://arxiv.org/abs/1904.08920)**: Introduces the TextVQA dataset and an OCR-aware VQA model with a copy mechanism.
- **[M4C](https://arxiv.org/abs/1910.09670)**: Multimodal transformer with multi-copy for OCR-VQA, demonstrating strong OCR-token integration.
- **[OCR-VQA](https://arxiv.org/abs/1908.08549)**: Early OCR-focused VQA dataset and pipeline.
- **[ImageNet-C](https://arxiv.org/abs/1903.12261)**: Standard corruption benchmark framework that motivates severity-based corruption protocols.
- **[PaddleOCR](https://arxiv.org/abs/2206.03001)**: Practical OCR system used widely for text extraction.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Chart QA benchmarks | Evaluate chart QA on clean and diverse charts | ChartQA, PlotQA, ChartQAPro, CharXiv | accuracy / relaxed accuracy | Older datasets saturate; limited robustness coverage |
| Robustness benchmarks | Apply corruptions/occlusions to charts | CHART NOISe | accuracy under corruption | Mitigation methods underexplored |
| Chart-to-structure | Parse charts to tables/JSON/code then reason | DePlot, UniChart, ChartReasoner | extraction + QA | Parsing errors propagate; robustness to corruption limited |
| Multi-view / zoom | Add extra views/crops/high-res tokens | Dragonfly, ViCrop, CropVLM | VQA / DocVQA / TextVQA | May miss relevant regions; may require training or internals |
| OCR-aware VQA | Feed OCR tokens to model or use copy mechanism | LoRRA, M4C | TextVQA / OCR-VQA | Typically requires specialized architectures and training |

### Closest Prior Work

- **ViCrop**: Training-free crop selection for small-detail VQA. Unlike ReInk, it selects one crop rather than constructing a global spatially aligned text-only view.
- **CropVLM**: Trains a crop-selection policy (via RL) to zoom into a single region. Unlike ReInk, it requires training and does not isolate whether gains are specifically from readable text.
- **Seeing but Not Believing (VEA)**: Constructs attention-highlighted evidence views. Unlike ReInk, it does not explicitly create a repaired text channel and may require attention extraction.
- **CHART NOISe / Losing the Plot**: Provides a robustness evaluation and suggests quality filtering, but does not propose a concrete inference-time recovery method.

**Novelty Kill Search Summary:**
Searched for the exact combination of "OCR re-rendered text canvas" + "chart question answering" (and variants: "OCR overlay as image input", "text-only OCR layer as second image", "re-ink OCR view VLM"), and checked recent OpenReview/arXiv results for similar multi-view OCR presentation. No direct prior work using a spatially aligned OCR re-rendered canvas as an auxiliary image input for chart QA robustness was found as of 2026-02-22. (Full query log and negative evidence are in `notes.md`.)

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| ViCrop | Training-free crop selection via attention/gradients | Crop may miss relevant text regions; not chart-specific | Provide a global OCR text view instead of a single crop | Captures multiple text regions without region selection |
| CropVLM | Learned crop selection via RL | Requires training a cropper; single crop | Training-free OCR-based auxiliary image | Easier to deploy; covers many labels/ticks |
| Dragonfly | Trained multi-resolution zoom encoding | Requires training-time changes | Inference-only wrapper | Applicable to arbitrary VLMs |
| VEA | Attention-based evidence highlighting | Not a text repair channel; may need internals | OCR-based re-rendering | Directly increases text legibility |
| CHART NOISe | Robustness benchmark | No recovery method tested | Recovery via re-ink auxiliary view | Targets blur/pixelation-induced unreadable text |

---

## Experiments

### Experimental Setup

**Task.** Chart question answering: given a chart image and a question, output an answer evaluated by ChartQAPro's enhanced relaxed accuracy (higher is better).

**Metric.** ChartQAPro uses an enhanced relaxed accuracy: numeric answers are correct within a 5% margin (except years require exact match), textual answers use ANLS (average normalized Levenshtein similarity), and multiple-choice/fact-checking answers use exact match ([ChartQAPro evaluation metric](./references/ChartQAPro-A-More-Diverse-and-Challenging-Benchmark-for-Chart-Question-Answering/sections/4.3%20Evaluation%20Metric.md)).

**Primary benchmark.** ChartQAPro test set (1,948 questions) ([ChartQAPro](./references/ChartQAPro-A-More-Diverse-and-Challenging-Benchmark-for-Chart-Question-Answering/meta/meta_info.txt)).

**Corruption protocol.** Create a corrupted evaluation set by applying two chart-relevant corruptions that primarily destroy small text:
- **Defocus blur (major severity)**
- **Pixelate (major severity)**

Implementation detail for reproducibility: use a standard ImageNet-C style corruption library (e.g., `imagecorruptions`) and set severity=4 ("major" as in CHART NOISe, which maps major to severity level 4) ([Losing the Plot dataset generation](./references/Losing-the-Plot-How-VLM-responses-degrade-on-imperfect-charts/sections/3.2%20Corruption%20and%20Occlusion%20Dataset.md)).

**Base models.** Use at least one open VLM that supports multi-image input.

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-VL-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | Strong open VLM; supports multi-image inputs |

(Optional robustness check if budget allows): Qwen2-VL-72B-Instruct.

**Baseline Ladder (REQUIRED):**
- Prompting baseline: single fixed direct-answer prompt (same across conditions).
- Inference-time scaling baseline: self-consistency with k=4 samples on the baseline condition, selecting the majority final answer after normalization.
- Closest practical baseline: scrambled-text canvas control (semantic ablation) and an OCR-as-text ablation (see below).

**Conditions (main comparison).** All conditions use 2-image input to control for extra image tokens:
- (A) Corrupted chart + blank canvas (baseline)
- (B) Corrupted chart + scrambled-text canvas (layout-only control)
- (C) Corrupted chart + correct ReInk canvas (ours)

**Inference-time scaling (baseline).**
- (A-SC4) Self-consistency on (A): sample k=4 outputs with temperature=0.7 and select the majority final answer after normalization (lowercasing, stripping punctuation; numeric answers bucketed by rounding to 3 significant digits). Run seeds=[42, 123, 456] and report mean+/-std.

**Resource Estimate**:
- Compute budget (single model, greedy decoding for A/B/C): 1,948 questions * 2 corruptions * 3 conditions = 11,688 forward passes.
- Self-consistency baseline adds 1,948 * 2 * 4 = 15,584 forward passes (plus aggregation).
- Total is ~27k forward passes, which should fit within <=100 GPU-hours on A100-class GPUs for a 7B VLM (verification should adjust based on measured throughput).
- OCR + rendering runs on CPU.
- API usage: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ChartQAPro | Real-world chart QA with diverse sources and question types | enhanced relaxed accuracy (overall %) | test | https://github.com/vis-nlp/ChartQAPro | ChartQAPro official code |
| ChartQAPro-Corrupted (ours) | ChartQAPro test with major defocus blur + pixelate | same as above | test | generated from ChartQAPro | reuse ChartQAPro eval |

### Main Results

| Method | Base Model | Benchmark | Metric (mean+/-std) | Source | Notes |
|---|---|---|---|---|---|
| (A) Corrupted + blank canvas | Qwen2.5-VL-7B | ChartQAPro-Corrupted | **TBD** | - | Greedy decoding (temp=0), deterministic |
| (A-SC4) Self-consistency (k=4) on (A) | Qwen2.5-VL-7B | ChartQAPro-Corrupted | **TBD** | - | temperature=0.7; seeds=[42,123,456] |
| (B) Corrupted + scrambled-text canvas | Qwen2.5-VL-7B | ChartQAPro-Corrupted | **TBD** | - | Greedy decoding (temp=0) |
| (C) Corrupted + correct ReInk canvas (ours) | Qwen2.5-VL-7B | ChartQAPro-Corrupted | **TBD** | - | Greedy decoding (temp=0) |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---|---|---|
| OCR-as-text baseline | Provide OCR strings as plain text in the prompt (no re-ink canvas) | If this matches (C), re-inking is unnecessary |
| (C) without box outlines | Do not draw OCR rectangles | If performance drops, spatial anchoring matters |

(Additional analysis-only diagnostics if budget allows: report OCR coverage/confidence vs accuracy; and per-corruption breakdown.)

### Experimental Rigor

- **Variance & reproducibility**: Greedy decoding (temperature=0) is deterministic; self-consistency uses 3 seeds and reports mean+/-std.
- **Key confounders and controls**:
  - Extra image tokens: controlled by always passing two images in A/B/C.
  - High-contrast strokes/layout cues vs text semantics: controlled by scrambled-text baseline (B).
  - OCR noise: analyzed via OCR-as-text ablation and by reporting OCR coverage/confidence statistics.
- **Sanity check**: On the clean ChartQAPro test set, ReInk should not substantially improve accuracy; a large gain on clean would suggest a prompt/format confound.
- **Data leakage**: ChartQAPro may be partially present in some model pretraining data. Our primary claim is about robustness under synthetic corruptions applied at evaluation time, which reduces the risk that improvements are due to memorization of clean images.

---

## Success Criteria

**Hypothesis** (directional): The correct ReInk canvas (C) improves accuracy on ChartQAPro-Corrupted relative to both the blank-canvas baseline (A) and the scrambled-text control (B), and it is not matched by self-consistency on the baseline (A-SC4).

**Decision Rule** (concrete - when to stop):
- **Proceed**: (C) exceeds max(A-SC4, B) by >=2 percentage points on ChartQAPro-Corrupted, and (C) exceeds (B) by >=2 points (evidence that text semantics, not just layout strokes, contributes).
- **Pivot**: If (C) > (A) but (C) <= (A-SC4), reframe as a compute-quality trade-off (ReInk may be a cheaper alternative to sampling-based inference-time scaling).
- **Refute**: If (C) <= (B) or (C) improves over (A) by <1 point, abandon ReInk as a robustness intervention for these corruptions.

---

## Impact Statement

If successful, ReInk is a simple, model-agnostic pre-processing wrapper that improves chart QA robustness to blur and pixelation without retraining. This would benefit practitioners deploying chart QA on screenshots and scanned documents by reducing text-legibility-related failures.

---

## References

- [ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering](./references/ChartQAPro-A-More-Diverse-and-Challenging-Benchmark-for-Chart-Question-Answering/meta/meta_info.txt) - Masry et al., 2025
- [Losing the Plot: How VLM responses degrade on imperfect charts](./references/Losing-the-Plot-How-VLM-responses-degrade-on-imperfect-charts/meta/meta_info.txt) - Shin et al., 2025
- [Towards Perceiving Small Visual Details in Zero-shot Visual Question Answering with Multimodal LLMs (ViCrop)](./references/Towards-Perceiving-Small-Visual-Details-in-Zero-shot-Visual-Question-Answering-with-Multimodal-LLMs/meta/meta_info.txt) - Zhang et al., 2023
- [CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](./references/CropVLM-Learning-to-Zoom-for-Fine-Grained-Vision-Language-Perception/meta/meta_info.txt) - Carvalho et al., 2025
- [Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness in VLMs](./references/Seeing-but-Not-Believing-Probing-the-Disconnect-Between-Visual-Attention-and-Answer-Correctness-in-VLMs/meta/meta_info.txt) - Liu et al., 2025
- [ImageNet-C: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261) - Hendrycks and Dietterich, 2019
- [TextVQA / LoRRA: Towards VQA Models That Can Read](https://arxiv.org/abs/1904.08920) - Singh et al., 2019
- [M4C: Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA](https://arxiv.org/abs/1910.09670) - Hu et al., 2019
- [OCR-VQA](https://arxiv.org/abs/1908.08549) - Mishra et al., 2019
- [Dragonfly: Multi-Resolution Zoom Supercharges Large Visual-Language Model](https://arxiv.org/abs/2406.00977) - Thapa et al., 2024
- [PaddleOCR](https://arxiv.org/abs/2206.03001) - Du et al., 2022
