# untitled

# Copy-Then-Inpaint for Consistent Multi-Step GUI Generation on GEBench

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Core constraint**: Fully automated evaluation (no human-in-the-loop)
- **Compute constraint**: ≤768 A100 GPU-hours (we expect **0 GPU-hours**; API-only)
- **Infrastructure constraint**: No interactive GUI environments (we only use *offline* screenshot trajectories from a dataset)

## Introduction

### Context and Motivation

Generative image models are increasingly used as *simulators* for downstream systems, not just as image generators. A particularly demanding setting is **graphical user interfaces (GUIs)**, where the output image must obey rigid layout, typography, and interaction logic constraints. If image generation models could reliably predict the *next* GUI screenshot after a user action, they could serve as cheap, scalable “GUI environments” for training and evaluating computer-use agents.

However, GUI screenshots are structurally unlike natural images: small actions (a tap, typing in a field, opening a menu) often induce **localized state changes** while most of the screen should remain unchanged. When a model regenerates the entire screen each step, small errors (icon drift, text corruption, layout jitter) can accumulate and eventually break long-horizon interaction trajectories.

**[GEBench](./references/GEBench-Benchmarking-Image-Generation-Models-as-GUI-Environments/meta/meta_info.txt)** (arXiv:2602.09007) directly benchmarks this setting with multi-step GUI planning trajectories and a **5D GE-Score** computed by vision-language-model (VLM) judges: Goal Achievement (GOAL), Interaction Logic (LOGIC), Consistency (CONS), UI Plausibility (UI), and Visual Quality (QUAL).

### The Problem

GEBench reveals a sharp difficulty jump from single-step GUI transitions to **multi-step GUI trajectories** (Type 2). In Type 2, models must generate a 6-frame sequence (Frame0..Frame5) starting from a reference screenshot and a high-level goal (e.g., “Create a recurring bill reminder with notifications”). GEBench’s **CONS** rubric explicitly penalizes unintended drift: it asks whether UI elements are stable across frames (layout, colors, style, system bars) and assigns low scores when the screen “flickers” or rewrites unrelated regions.

The practical bottleneck is that today’s strongest image generation and editing models have no explicit mechanism to “freeze” regions that should remain identical across steps. This is not a prompting-only problem: even with detailed requirements like “only modify UI components affected by the action” (as in the official GEBench generation prompts), full-frame image-to-image generation can still introduce spurious changes.

Published results in GEBench already show large headroom for open models in multi-step planning. For example, **Qwen-Image-Edit** achieves only **26.79** (Chinese subset) / **18.61** (English subset) as the *aggregate* multi-step category score (Table 1 in the GEBench paper; normalized to 0–100 via the paper’s 0–5→0–100 mapping). This number is not the CONS subscore; per-dimension CONS must be established via evaluation runs.

### Key Insight and Hypothesis

**Key insight**: In multi-step GUI trajectories, the *semantic state change* is often sparse (a popup appears, a toggle flips, text in one field changes), while most pixels should remain stable. Therefore, a generation pipeline that **copies the previous frame and only re-renders the minimal region that should change** should improve temporal stability.

**Hypothesis**: If we generate Type-2 multi-step trajectories by (i) predicting a small **change mask** from the previous screenshot plus the current step context, and (ii) applying a strong image editor in **inpainting mode** to only that region, then **CONS will improve** compared to full-frame inpainting, without meaningfully reducing GOAL/LOGIC.

Why this could be wrong:
- The change mask might miss necessary edits (hurting GOAL/LOGIC) or be too large (no CONS gain).
- A strong image editor might still drift even outside the mask due to resampling or implicit global edits.

We rule out the trivial explanation “any mask helps” via a shuffled-mask ablation that preserves mask size/shape statistics but breaks step-to-step semantic alignment.

---

## Proposed Approach

### Overview

We propose an **inference-time wrapper** around an image editing model (Qwen-Image-Edit) to generate multi-step GUI trajectories with higher temporal consistency:

1. For each step, use a VLM to predict *where* the GUI must change (as bounding boxes → mask).
2. Use the image editor in **inpainting** mode to update only the masked region, while copying the rest of the previous frame.

This is training-free and can be applied to any image editor that supports masked editing.

### Method Details

**Task setting (GEBench Type 2).** Each sample contains an initial GUI screenshot Frame0 and a high-level goal text. The model must generate Frame1..Frame5. The official GEBench baseline generator (from the released repo) conditions on the previous frame and a step-indexed prompt derived from the goal.

**Step prompt.** To isolate the masking effect (not prompt engineering), we use the same step prompt template as the official Type-2 generator:
- Inputs: global goal `G` and step index `t∈{1..5}`.
- Output: a text prompt that instructs the model to generate the next UI state while preserving style/layout and progressing gradually.

**Mask prediction (VLM → bounding boxes → binary mask).** For each step `t`, we prompt a vision-language model (e.g., `Qwen3-VL-32B-Instruct`) with:
- the previous screenshot `I_{t-1}`
- the global goal `G`
- the step index `t`

The VLM outputs JSON with up to `K=5` bounding boxes (pixel coordinates) for regions that must change to reflect progress. We rasterize the union of boxes into a binary mask `M_t` and apply fixed dilation:

- `r = round(0.02 * min(H, W))` pixels (no tuning)

**Image editing / inpainting.** We generate the next frame using **Qwen-Image-Edit** in inpainting mode:

- Inputs: previous image `I_{t-1}`, mask `M_t`, and the step prompt.
- Output: edited image `I_t`.

This is repeated autoregressively for 5 steps.

**Three-condition design (≤3 conditions, mask-only difference).**

To cleanly test whether *semantic mask alignment* matters (not just “inpaint mode is better”), we use three conditions that are identical except for the mask:

- **A (Full-mask inpaint baseline)**: `M_t = 1` for all pixels (full-frame mask).
- **B (Predicted-mask inpaint)**: `M_t` from the VLM-predicted bounding boxes.
- **C (Shuffled-mask ablation)**: for each trajectory, rotate the predicted masks across steps (e.g., use step-(t mod 5)+1 mask at step t). This matches mask size/shape distributions but breaks temporal alignment.

Diagnostics (not additional experimental conditions):
- Mean IoU between `M_t` in B and C (if high, ablation is weak).
- Mask area (% pixels) statistics for B vs C.

### Key Innovations

- **A verification-first causal test for mask alignment** in sequential GUI generation: full-mask vs aligned-mask vs within-trajectory shuffled masks (controls for mask area/shape and inpaint pathway).
- **A simple, training-free mask predictor** (VLM → bounding boxes) that is implementable without building a custom segmentation model.
- **A new application target**: improving **GEBench Type-2 CONS** via masked editing, complementary to GUI world-model work that focuses on learning next-state dynamics end-to-end.

---

## Related Work

### Field Overview

**GUI environment modeling and evaluation.** A growing line of work treats GUIs as environments for agents, using screenshots as observations. Benchmarks like **GEBench** emphasize discrete action-conditioned transitions and long-horizon coherence, and propose rubric-based VLM judging to score functional correctness and visual stability.

**GUI world models.** Recent “GUI world models” learn next-state prediction to support planning for mobile/desktop agents. These methods often change the representation (e.g., structured text layouts, renderable code) and require training. They are adjacent but do not directly address a lightweight inference-time consistency fix for multi-step screenshot generation on GEBench.

**Mask-based image editing and inpainting.** Many diffusion-based editors support mask-conditioned editing. Methods such as DiffEdit and instruction-based editors focus on automatic mask derivation and high-quality edits, but typically target *single-step* edits of natural images rather than multi-step GUI trajectories scored by temporal consistency rubrics.

**Temporal consistency in video editing/inpainting.** Video diffusion and video editing methods introduce mechanisms to keep frames consistent (feature reuse, masked conditioning, spaced-frame priors). Our setting is “video-like” but with discrete UI jumps and stricter layout constraints.

### Related Papers

- **[GEBench: Benchmarking Image Generation Models as GUI Environments](./references/GEBench-Benchmarking-Image-Generation-Models-as-GUI-Environments/meta/meta_info.txt)**: Introduces GE-Score and multi-step GUI trajectory benchmark; highlights CONS/LOGIC failures.
- **[Computer-Use Agents as Judges for Generative User Interface](./references/Computer-Use-Agents-as-Judges-for-Generative-User-Interface/meta/meta_info.txt)**: Uses agentic evaluation for UI generation; complementary to rubric-based judging.
- **[MobileDreamer: Generative Sketch World Model for GUI Agent](./references/MobileDreamer-Generative-Sketch-World-Model-for-GUI-Agent/meta/meta_info.txt)**: GUI world model using sketch/text representations; focuses on agent planning rather than mask-based pixel preservation.
- **[ViMo: A Generative Visual GUI World Model for App Agents](https://arxiv.org/abs/2504.13936)**: Predicts next GUI observation with text rendering placeholders + diffusion; trained world model.
- **[gWorld: Generative Visual Code Mobile World Models](https://arxiv.org/abs/2602.01576)**: Generates renderable code for next-state GUIs; strong on mobile world-model benchmarks.
- **[ShowUI](https://arxiv.org/abs/2411.17465)**: Vision-language-action model for GUI agents; focuses on action execution, not generation stability.
- **[OS-Genesis](https://arxiv.org/abs/2412.19723)**: Reverse task synthesis for GUI agent trajectories; data generation rather than pixel-level consistency control.
- **[RICO](https://dl.acm.org/doi/10.1145/3126594.3126651)**: Large-scale mobile UI dataset that underpins many UI generation approaches.
- **[UI-Diffuser](https://arxiv.org/abs/2306.06233)**: Diffusion model for mobile UI design synthesis; not sequential.
- **[Pix2Struct](https://arxiv.org/abs/2210.03347)**: Screenshot-to-structure pretraining; relevant for GUI understanding and structured representations.
- **[DCGen](https://arxiv.org/abs/2406.16386)**: Divide-and-conquer screenshot-to-code generation; alternative representation pathway.
- **[DiffEdit](https://arxiv.org/abs/2210.11427)**: Automatically derives edit masks by comparing diffusion reconstructions; canonical mask-from-diffusion editor.
- **[InstructPix2Pix](https://arxiv.org/abs/2211.09800)**: Instruction-based image editing with diffusion; widely used baseline for text-guided edits.
- **[InstructEdit](https://arxiv.org/abs/2305.18047)**: Instruction-driven editing using grounding+segmentation to obtain masks before editing.
- **[BrushNet](https://arxiv.org/abs/2403.06976)**: Plug-and-play diffusion inpainting architecture; relevant to robust masked editing.
- **[StrDiffusion](https://arxiv.org/abs/2403.19898)**: Structure-guided inpainting; relevant for preserving layout.
- **[Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)**: Describes Qwen-Image(-Edit) model family and editing consistency mechanisms.
- **[Qwen-Image-Layered](https://arxiv.org/abs/2512.15603)**: Layer decomposition for more controllable editing; relevant to region-preserving edits.
- **[Segment Anything (SAM)](https://arxiv.org/abs/2304.02643)**: General-purpose segmentation; a common component in mask-based editors.
- **[GroundingDINO](https://arxiv.org/abs/2303.05499)**: Text-conditioned detection; used in pipelines that produce edit masks from language.
- **[TokenFlow](https://arxiv.org/abs/2307.10373)**: Feature-flow reuse for temporally consistent video editing.
- **[MCVD](https://arxiv.org/abs/2205.09853)**: Masked conditional video diffusion; connects mask conditioning to temporal consistency.
- **[RePaint](https://arxiv.org/abs/2201.09865)**: Diffusion-based inpainting; canonical baseline for masked diffusion inpainting.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| GUI environment benchmarks | Rubric-based scoring of action-conditioned GUI transitions | GEBench | GE-Score (GOAL/LOGIC/CONS/UI/QUAL) | VLM judging cost; multi-step still hard |
| GUI world models | Learn next-state prediction for agent planning | ViMo, gWorld, MobileDreamer | AndroidWorld, MWMBench, etc. | Training cost; not a simple inference-time patch |
| Mask-based image editing | Use explicit masks to constrain edits | DiffEdit, InstructEdit, BrushNet, RePaint | Natural-image editing benchmarks | Typically single-step; not evaluated on GUI temporal drift |
| Temporal-consistency editing | Encourage frame-to-frame consistency in video edits | TokenFlow, MCVD | Video editing / inpainting benchmarks | Assumes continuous motion; GUIs have discrete jumps |

### Closest Prior Work

1) **[GEBench](./references/GEBench-Benchmarking-Image-Generation-Models-as-GUI-Environments/meta/meta_info.txt)**: Defines the multi-step GUI trajectory task and CONS rubric; does not propose a mechanism to prevent drift beyond prompting.

2) **[DiffEdit](https://arxiv.org/abs/2210.11427)**: Provides an automatic way to derive edit masks for single-step diffusion editing; it does not study multi-step autoregressive trajectories or GUI-specific stability metrics.

3) **[InstructEdit](https://arxiv.org/abs/2305.18047)**: Uses grounding+segmentation to obtain masks before editing; focuses on single-step instruction edits and does not test long-horizon sequential consistency.

4) **[TokenFlow](https://arxiv.org/abs/2307.10373)** / **[MCVD](https://arxiv.org/abs/2205.09853)**: Address temporal consistency for video editing/inpainting; their mechanisms are heavier than needed for GUI trajectories and do not leverage the “sparse discrete UI update” property.

5) **[ViMo](https://arxiv.org/abs/2504.13936)** / **[gWorld](https://arxiv.org/abs/2602.01576)**: Train world models for GUI next-state prediction; they aim to learn dynamics, whereas we test whether a lightweight region-preserving wrapper already yields large CONS gains.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| GEBench | Benchmarks multi-step GUI generation with CONS rubric | No method for preventing drift | Add region-preserving editing wrapper | CONS directly rewards preserving unchanged regions |
| DiffEdit | Automatic mask derivation for diffusion editing | Single-step; natural images | Apply mask editing sequentially on GUIs | GUIs have sparse state changes; masking should be especially effective |
| InstructEdit | Mask generation via grounding+segmentation for edits | Single-step; no temporal metric | Use VLM bbox masks + shuffled-mask causal test | Simple mask predictor suffices; evaluation isolates alignment |
| TokenFlow / MCVD | Temporal consistency in video editing | Heavier assumptions; continuous motion | Use discrete-step sparse masks + inpaint | GUI steps are localized; simpler approach may work |
| ViMo / gWorld | Trained GUI world models | Training cost; different target metric | Inference-only wrapper on a strong editor | Fast to deploy; can be combined with world models later |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen-Image-Edit | - | https://huggingface.co/Qwen | Used as the inpainting/editing backbone (API-callable in this repo) |
| Qwen3-VL-32B-Instruct (mask predictor) | 32B | https://huggingface.co/Qwen | VLM predicts change-region bounding boxes (API-callable in this repo) |
| GPT-4o (judge) | - | https://platform.openai.com/docs/models | Used only for evaluation judging (API-callable) |

**Training Data (if applicable):**

No training data needed — **inference only**.

**Other Resources (if applicable):**
- Official GEBench evaluation rubrics (we use the Type-2 rubric prompt from the released repo).

**Resource Estimate**:
- **GPU budget**: 0 GPU-hours (API-only)
- **API usage (rough)**:
  - For `N` trajectories and 3 conditions (A/B/C):
    - Image edits: `15N` Qwen-Image-Edit calls (5 steps × 3 conditions)
    - Mask prediction: `5N` VLM calls (shared across B/C)
    - Judging: `3N` GPT-4o calls (each call includes 6 images)
  - For `N=70` (approx. one language subset): 1050 edit calls + 350 mask calls + 210 judge calls.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| GEBench Type 2 (multi-step) | 6-frame GUI trajectories from a goal | GOAL/LOGIC/CONS/UI/QUAL (0–5); Overall (mean/5) | test (all provided) | [HF dataset: stepfun-ai/GEBench](./references/stepfun-ai-GEBench-Datasets-at-Hugging-Face/meta/meta_info.txt) | [GEBench GitHub](./references/GitHub---stepfun-ai-GEBench/meta/meta_info.txt) (requires a small patch; see below) |

**Evaluation Scripts:**
- The released repo contains an evaluator with the Type-2 rubric prompt.
- **Important implementation note**: the released `Type2Judge` passes `{"frames": ...}` to the GPT-4o provider, but the provider expects `sample_data["images"]`. Verification should patch by mapping `frames -> images` before calling the provider, or implement a small standalone evaluator that:
  1) loads frames 0..5, 2) calls GPT-4o with `TYPE2_EVAL_PROMPT`, 3) parses `{goal, logic, cons, ui, qual}` scores.

### Main Results

**Decision rule (verification):**
- Primary success: B improves **CONS** over A, while GOAL and LOGIC do not show large regressions (directional: no consistent drop comparable in magnitude to the CONS gain).
- Strongest claim: B > C on CONS (and ideally Overall), indicating *semantic alignment* of the mask matters.
- If B > A but B ≈ C, we conclude masking helps but semantic alignment is not required (actionable but weaker result).
- If B ≤ A on CONS, refute the core hypothesis.

#### Results Table

| Method | Base Model | Benchmark | Eval protocol | Overall (0–100) | CONS (0–100) | GOAL (0–100) | LOGIC (0–100) | Source | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| Qwen-Image-Edit (published) | Qwen-Image-Edit | GEBench Type2 | Paper (multi-judge, normalized) | 26.79 (CN) / 18.61 (EN) | N/A (not reported) | N/A | N/A | [GEBench Table 1](./references/GEBench-Benchmarking-Image-Generation-Models-as-GUI-Environments/meta/meta_info.txt) | Aggregate category score only |
| Full-mask inpaint (A) | Qwen-Image-Edit | GEBench Type2 | This proposal (GPT-4o judge) | **TBD** | **TBD** | **TBD** | **TBD** | - | Needs re-run |
| Predicted-mask inpaint (B) | Qwen-Image-Edit | GEBench Type2 | This proposal (GPT-4o judge) | **TBD** | **TBD** | **TBD** | **TBD** | - | To be verified |
| Shuffled-mask inpaint (C) | Qwen-Image-Edit | GEBench Type2 | This proposal (GPT-4o judge) | **TBD** | **TBD** | **TBD** | **TBD** | - | Controls for mask size/shape |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B (full) | VLM-predicted mask | Best CONS if alignment matters |
| C (shuffled) | Same masks, wrong step | CONS drops if alignment matters |
| B w/o dilation | Set `r=0` | More GOAL/LOGIC failures if boxes are tight |

### Analysis (Optional)

- **Mask diagnostics**: distribution of mask area and within-sample shuffle IoU to assess whether C is a strong counterfactual.
- **Failure modes**: manual *qualitative* inspection (no scoring) of cases where CONS improves but GOAL/LOGIC drops (e.g., missing a popup region).

---

## Success Criteria

**Criterion 1: Masking improves temporal consistency**
- Hypothesis: Predicted-mask inpainting (B) yields higher average **CONS** than full-mask inpainting (A) on GEBench Type 2.
- Validation: A clear, consistent CONS improvement across trajectories, without large regressions in GOAL/LOGIC.

**Criterion 2: Semantic alignment matters (stronger claim)**
- Hypothesis: B outperforms the within-sample shuffled-mask ablation (C), indicating the gain is not solely due to “editing less area.”
- Validation: B > C on CONS (and ideally also on Overall), with diagnostics confirming C is not trivially similar to B.

---

## Impact Statement

If successful, this provides a simple, training-free recipe for building more stable multi-step GUI simulators from existing image editors: **predict where the UI should change and only edit that region**. This could improve the practicality of using generative models as GUI environments for agent training by reducing drift that breaks long-horizon trajectories.

---

## References

- [GEBench: Benchmarking Image Generation Models as GUI Environments](./references/GEBench-Benchmarking-Image-Generation-Models-as-GUI-Environments/meta/meta_info.txt) - Li et al., 2026
- [Computer-Use Agents as Judges for Generative User Interface](./references/Computer-Use-Agents-as-Judges-for-Generative-User-Interface/meta/meta_info.txt) - Lin et al., 2025
- [MobileDreamer: Generative Sketch World Model for GUI Agent](./references/MobileDreamer-Generative-Sketch-World-Model-for-GUI-Agent/meta/meta_info.txt) - Cao et al., 2025
- [ViMo: A Generative Visual GUI World Model for App Agents](https://arxiv.org/abs/2504.13936)
- [gWorld: Generative Visual Code Mobile World Models](https://arxiv.org/abs/2602.01576)
- [ShowUI](https://arxiv.org/abs/2411.17465)
- [OS-Genesis](https://arxiv.org/abs/2412.19723)
- [RICO: A Mobile App Dataset for Building Data-Driven Design Applications](https://dl.acm.org/doi/10.1145/3126594.3126651)
- [UI-Diffuser](https://arxiv.org/abs/2306.06233)
- [Pix2Struct](https://arxiv.org/abs/2210.03347)
- [DCGen](https://arxiv.org/abs/2406.16386)
- [DiffEdit](https://arxiv.org/abs/2210.11427)
- [InstructPix2Pix](https://arxiv.org/abs/2211.09800)
- [InstructEdit](https://arxiv.org/abs/2305.18047)
- [BrushNet](https://arxiv.org/abs/2403.06976)
- [StrDiffusion](https://arxiv.org/abs/2403.19898)
- [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- [Qwen-Image-Layered](https://arxiv.org/abs/2512.15603)
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643)
- [GroundingDINO](https://arxiv.org/abs/2303.05499)
- [TokenFlow](https://arxiv.org/abs/2307.10373)
- [MCVD](https://arxiv.org/abs/2205.09853)
- [RePaint](https://arxiv.org/abs/2201.09865)
