# untitled

# Shallow-Layer Normalized Attention for Single-Pass Debiased Visual Token Pruning

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Modern vision-language models (VLMs) answer questions about images by converting an image into a long sequence of **visual tokens** (e.g., ViT patch embeddings) and concatenating them with text tokens inside a large language model (LLM) decoder. When images are high-resolution, the number of visual tokens can be in the thousands, making the **prefill** stage (processing the full input sequence before any output tokens are generated) and the **KV cache** (the per-layer key/value tensors stored to speed up autoregressive decoding) expensive.

A widely used approach to reduce inference cost is **visual token pruning**: selecting a subset of visual tokens at some decoder layer and discarding the rest for subsequent layers. Many training-free pruning methods use **text→vision attention** as an importance score (e.g., attention from the last text token to each visual token), because it is already computed during a forward pass.

However, recent work shows that attention-based pruning can catastrophically fail on spatially sensitive tasks such as **referring expression grounding** (e.g., RefCOCO/RefCOCO+/RefCOCOg, where the model must localize the object referred to by a text phrase and is scored by IoU-thresholded accuracy), even when it works well on coarse-grained VQA. Several papers attribute this to systematic **positional/recency bias** in attention scores that is not driven by image content.

### The Problem

Existing attention-debiasing approaches for VLM token pruning reduce positional bias, but they often rely on **offline priors** or extra inference passes:

- **[Attention Debiasing for Token Pruning in Vision Language Models](./references/Attention-Debiasing-for-Token-Pruning-in-Vision-Language-Models/meta/meta_info.txt)** (Zhao et al.; sometimes referred to as PoRe) estimates a dataset-level positional bias trend (fit by an exponential) and reweights attention scores during pruning.
- **[D²Pruner: Debiased Importance and Structural Diversity for MLLM Token Pruning](./references/D²Pruner-Debiased-Importance-and-Structural-Diversity-for-MLLM-Token-Pruning/meta/meta_info.txt)** computes a **positional bias prior** by averaging attention maps over 1000 COCO images with a generic prompt, and divides per-instance attention by this offline prior before selecting tokens.
- **[ShaRP: SHAllow-LayeR Pruning for Video Large Language Models Acceleration](./references/ShaRP-SHAllow-LayeR-Pruning-for-Video-Large-Language-Models-Acceleration/meta/meta_info.txt)** estimates positional bias by running an additional forward pass on black frames and subtracting the resulting attention bias.
- **[Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](./references/Feather-the-Throttle-Revisiting-Visual-Token-Pruning-for-Vision-Language-Model-Acceleration/meta/meta_info.txt)** proposes a calibration-free RoPE-free attention criterion plus uniform sampling (and two-stage pruning) to improve localization under early pruning.
- **[Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs](./references/Beyond-Text-Visual-Attention-Exploiting-Visual-Cues-for-Effective-Token-Pruning-in-VLMs/meta/meta_info.txt)** (FasterVLM) avoids biased LLM cross-attention by pruning using [CLS] attention from the visual encoder.
- **[Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](./references/Balanced-Token-Pruning-Accelerating-Vision-Language-Models-Beyond-Local-Optimization/meta/meta_info.txt)** uses a small calibration set to optimize multi-stage pruning schedules beyond per-layer greedy criteria.

These strategies leave a practical gap:

1. **Offline priors may not transfer**: a prior measured on COCO + generic prompts may not match new domains (documents, remote sensing), new prompt styles (OCR-heavy, counting), or new resolutions/patching.
2. **Extra passes reduce speedups**: per-instance “black frame” bias estimation adds an extra forward pass, which partially cancels pruning’s latency gains.
3. **Engineering friction**: deploying pruning in production often needs “no-calibration” knobs, because collecting calibration images and prompt distributions is not always possible.

The goal of this proposal is not to invent a new token selection algorithm, but to remove the dependency on offline priors by extracting a **per-instance positional bias estimate inside the same forward pass**.

### Key Insight and Hypothesis

**Key insight:** In transformer decoders, attention patterns in **intermediate layers** tend to be the most prompt-dependent, while **shallow layers** are often more influenced by positional encoding artifacts and less by prompt semantics. **[IVC-Prune](./references/IVC-PRUNE-REVEALING-THE-IMPLICIT-VISUAL-CO-ORDINATES-IN-LVLMS-FOR-VISION-TOKEN-PRUNING/meta/meta_info.txt)** reports this prompt-sensitivity pattern and uses it to justify choosing an intermediate pruning layer.

**Hypothesis:** For a fixed image, the shallow-layer text→vision attention map is (i) more prompt-stable and (ii) more position-correlated than a mid-layer attention map. Therefore, the shallow-layer map can act as a **per-instance positional bias prior**. Dividing a mid-layer attention map by this shallow-layer map yields a debiased importance score competitive with offline-prior methods like D²Pruner, without any calibration dataset and without extra forward passes.

This could be wrong if shallow-layer attention is either too noisy (no stable positional structure) or already prompt-dependent, or if the ratio mainly corrects attention entropy differences across layers rather than positional bias. The experiments include an explicit Phase-0 diagnostic to test these assumptions.

---

## Proposed Approach

### Overview

We propose a **single-pass, per-instance attention normalization** for attention-based VLM token pruning:

1. During the normal forward pass up to a chosen pruning layer, collect text→vision attention at:
   - a shallow layer (candidate K_s ∈ {1,2,3})
   - a mid layer (K_m, typically the layer where pruning is applied)
2. Compute an online debiased score per visual token:

\[
A_{rel}^{online}(i) = \frac{A_{mid}(i)}{A_{shallow}(i) + \epsilon}
\]

3. Use this score as the importance input to an existing selection algorithm (we will plug into D²Pruner’s pivot + MIS selection so that the only change is the debiasing prior).

### Method Details

**Attention extraction.** Following D²Pruner, for an image-text input we define:
- \(A_{mid}\): the attention weights directed from the final text token to each of the \(N\) visual tokens at layer \(K_m\) (averaged over heads).
- \(A_{shallow}\): the same quantity but at a shallower layer \(K_s\).

**Online debiasing.** We set \(\epsilon = 10^{-7}\) (same order as D²Pruner) and compute \(A_{rel}^{online}\) as above.

**Selection algorithm (kept fixed).** We use D²Pruner’s selection pipeline unchanged:
- **Pivot selection**: take the top-\(k\) tokens by the debiased score as “pivots”.
- **MIS expansion**: add tokens via a maximal independent set (MIS) routine on a hybrid graph that encodes (i) spatial adjacency in the image grid and (ii) semantic similarity in hidden space, encouraging the kept set to cover diverse regions rather than only the highest-scoring cluster.

This isolates the effect of **how the bias prior is obtained** (offline vs online), without changing the diversity/structure component.

**Phase-0 layer choice.** We select \(K_s\) by a small diagnostic (described in Experiments) to avoid making layer choice a free hyperparameter.

### Key Innovations

1. **Single-pass per-instance positional bias estimation**: replaces offline bias priors (dataset-level curves or COCO-averaged attention maps) with a bias estimate derived from shallow-layer attention in the same forward pass.
2. **Drop-in replacement for existing debiased pruning pipelines**: we keep the rest of the pruning algorithm fixed (D²Pruner pivot + MIS), making the evaluation attributable.
3. **Mechanism-first verification**: a Phase-0 diagnostic explicitly checks whether shallow attention is prompt-stable and position-correlated; if not, the proposal is refuted early.

---

## Related Work

### Field Overview

**Training-free pruning based on attention.** Many VLM pruning methods compute a token importance score from attention patterns in the LLM decoder, then prune tokens once (single-shot) or progressively across layers. A recurring issue is that attention is not a faithful proxy for semantic relevance: it can be biased by positional encodings (recency/periphery bias), attention sinks, or shallow-layer interaction limitations.

**Debiasing attention scores.** Recent papers debias attention scores before pruning using (i) dataset-level positional trend fitting (**Zhao et al., 2025 / PoRe**), (ii) an offline COCO-averaged bias prior (**D²Pruner**), (iii) RoPE-free attention criteria and uniform sampling (**FEATHER**), (iv) per-instance extra-pass bias estimation (**ShaRP**), or (v) calibration-set-optimized multi-stage schedules (**Balanced Token Pruning**). These works suggest that correcting positional bias is important, but they do not provide a **single-pass, per-instance** bias prior that is derived from the model’s own internal attention dynamics.

**Non-attention scoring and spatial preservation.** Other approaches avoid attention scores entirely (e.g., value-vector similarity in IVC-Prune) or preserve spatial structure through special tokens or position-ID preservation. These results motivate evaluating debiasing mainly on spatially sensitive benchmarks (RefCOCO-style).

### Related Papers

- **[Attention Debiasing for Token Pruning in Vision Language Models](./references/Attention-Debiasing-for-Token-Pruning-in-Vision-Language-Models/meta/meta_info.txt)**: Fits a dataset-level positional bias curve (exponential trend) and reweights attention scores for pruning.
- **[D²Pruner: Debiased Importance and Structural Diversity for MLLM Token Pruning](./references/D²Pruner-Debiased-Importance-and-Structural-Diversity-for-MLLM-Token-Pruning/meta/meta_info.txt)**: Normalizes per-instance attention by an offline positional prior and enforces structural diversity via MIS selection.
- **[IVC-Prune: Revealing the Implicit Visual Co-ordinates in LVLMs for Vision Token Pruning](./references/IVC-PRUNE-REVEALING-THE-IMPLICIT-VISUAL-CO-ORDINATES-IN-LVLMS-FOR-VISION-TOKEN-PRUNING/meta/meta_info.txt)**: Shows RoPE induces implicit coordinate tokens and proposes prompt-aware pruning that preserves them while keeping original position IDs.
- **[ShaRP: SHAllow-LayeR Pruning for Video Large Language Models Acceleration](./references/ShaRP-SHAllow-LayeR-Pruning-for-Video-Large-Language-Models-Acceleration/meta/meta_info.txt)**: Enables shallow-layer pruning for video VLMs using segment masking, black-frame positional debiasing, and token deduplication.
- **[NÜWA: Mending the Spatial Integrity Torn by VLM Token Pruning](../../papers/paper_summaries/N%20%C3%9CWA%20MENDING%20THE%20SPATIAL%20INTEGRITY%20TORN%20BY%20VLM%20TOKEN%20PRUNING.md)**: Two-stage pruning to preserve a global spatial reference frame for grounding.
- **[Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](./references/Feather-the-Throttle-Revisiting-Visual-Token-Pruning-for-Vision-Language-Model-Acceleration/meta/meta_info.txt)**: Proposes a RoPE-free pruning criterion plus uniform sampling and two-stage pruning; highlights that early attention criteria can be dominated by positional bias.
- **[FastV](https://arxiv.org/abs/2403.06764)**: Representative attention-based training-free pruning that uses last-text-token attention to rank and drop visual tokens after an early decoder layer.
- **[PyramidDrop](https://arxiv.org/abs/2410.17247)**: Progressive, stage-wise visual token dropping (pyramid schedule) motivated by layer-wise redundancy patterns in VLM decoders.
- **[SparseVLM](https://arxiv.org/abs/2410.04417)**: Training-free text-guided token sparsification with adaptive layer-wise schedules and token recycling.
- **[HiMAP](https://arxiv.org/abs/2503.13108)**: Layer-adaptive pruning that uses different importance criteria in shallow vs deep layers based on a measured visual information flow pattern.
- **[LLaVA-PruMerge](https://arxiv.org/abs/2403.15388)**: Adaptive token reduction via important-token selection plus token merging to preserve information while shrinking the visual sequence.
- **[Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs](./references/Beyond-Text-Visual-Attention-Exploiting-Visual-Cues-for-Effective-Token-Pruning-in-VLMs/meta/meta_info.txt)**: Prunes using [CLS] attention from the vision encoder (prompt-agnostic scoring) to avoid biased LLM cross-attention.
- **[Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](./references/Balanced-Token-Pruning-Accelerating-Vision-Language-Models-Beyond-Local-Optimization/meta/meta_info.txt)**: Uses a small calibration set to optimize multi-stage pruning that balances local-layer vs downstream effects.
- **[ToMe: Token Merging](https://arxiv.org/abs/2210.09461)**: Token merging in vision transformers; often used as a baseline for compression.
- **[MMTok](../../papers/paper_summaries/MMTok%20Multimodal%20Coverage%20Maximization%20for%20Efficient%20Inference%20of%20VLMs.md)**: Training-free selection by multimodal coverage maximization.
- **[FlashVLM](../../papers/paper_summaries/FlashVLM%20Text-Guided%20Visual%20Token%20Selection%20for%20Large%20Multimodal%20Models.md)**: Text-guided pre-selection designed to be FlashAttention-friendly.
- **[Representation Shift: Unifying Token Compression with FlashAttention](../../papers/paper_summaries/Representation%20Shift%20Unifying%20Token%20Compression%20with%20FlashAttention.md)**: Attention-map-free scoring for FlashAttention compatibility.
- **[ZSPAPrune](../../papers/paper_summaries/ZSPAPrune%20Zero-Shot%20Prompt-Aware%20Token%20Pruning%20for%20Vision-Language%20Models.md)**: Zero-shot prompt-aware pruning strategy.
- **[Multi-Cue Adaptive Visual Token Pruning](../../papers/paper_summaries/Multi-Cue%20Adaptive%20Visual%20Token%20Pruning%20for%20Large%20Vision-Language%20Models.md)**: Uses multiple cues to adapt pruning decisions per instance.
- **[AdaTP](../../papers/paper_summaries/AdaTP%20Attention-Debiased%20Token%20Pruning%20for%20Video%20Large%20Language%20Models.md)**: Debiases both global and local attention biases for video token pruning.
- **[LVLM_CSP](../../papers/paper_summaries/LVLM_CSP%20Accelerating%20Large%20Vision%20Language%20Models%20via%20Clustering,%20Scattering,%20and%20Pruning%20for%20Reasoning%20Segmentation.md)**: Clustering/scattering/pruning for reasoning segmentation.
- **[PPE: Positional Preservation Embedding](../../papers/paper_summaries/Positional%20Preservation%20Embedding%20for%20Multimodal%20Large%20Language%20Models.md)**: Preserves multiple position IDs during token merging to reduce spatial degradation.
- **[Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](../../papers/paper_summaries/Token%20Pruning%20in%20Multimodal%20Large%20Language%20Models%20Are%20We%20Solving%20the%20Right%20Problem.md)**: Critiques benchmark sensitivity and highlights failure modes of pruning criteria.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Offline-prior attention debiasing | Use dataset-level or offline bias prior to correct attention | Attention Debiasing (PoRe), D²Pruner | VQA + RefCOCO grounding | Prior may not transfer across domains/prompts |
| Extra-pass per-instance debiasing | Estimate bias via an auxiliary input (e.g., black frames) | ShaRP | VideoMME, MLVU, LongVideoBench | Extra pass reduces speedup |
| Calibration-free attention criteria | Modify the pruning criterion to remove positional artifacts (often plus sampling heuristics) | FEATHER | RefCOCO + other VQA | May need criterion engineering (e.g., RoPE-free scoring) and/or multi-stage pipelines |
| Prompt-agnostic vision-side scoring | Prune using visual-encoder signals instead of LLM cross-attention | FasterVLM ([CLS] attention) | VQA-style benchmarks | May miss prompt-conditioned relevance that arises inside the LLM |
| Prompt-aware non-attention scoring | Use similarity/value vectors, preserve special spatial tokens | IVC-Prune | RefCOCO, SpatialEval, OCRBench | RoPE-specific assumptions; still needs prompt-aware query |
| Calibration-set schedule optimization | Use a calibration set to optimize multi-stage pruning beyond local criteria | Balanced Token Pruning | Multi-benchmark suites | Requires calibration data; schedule may not transfer |
| Structural diversity selection | Enforce spatial/semantic coverage beyond top-k | D²Pruner, MMTok | VQA + RefCOCO | Graph construction overhead; hyperparameters |
| Position/structure preservation in merging | Preserve position IDs / spatial structure while merging | PPE, NÜWA | TextVQA, grounding | Often architecture-specific |

### Closest Prior Work

- **D²Pruner**: Closest baseline because it explicitly normalizes attention by a positional bias prior and then performs structure-aware selection. Our proposal keeps its selection pipeline but replaces the offline prior (COCO+generic prompt average) with an online per-instance shallow-layer prior.
- **Attention Debiasing / PoRe**: Also debiases attention but uses a dataset-level curve fit rather than a per-instance prior; it is calibration-light but still requires collecting attention statistics and fitting a curve.
- **FasterVLM (Beyond Text-Visual Attention)**: Avoids LLM cross-attention entirely by using vision-encoder [CLS] attention; this can be effective but changes the signal source and may miss prompt-conditioned relevance that arises inside the LLM.
- **Balanced Token Pruning**: Optimizes multi-stage pruning schedules using a calibration set; it directly targets downstream effects but still requires calibration data, which this proposal aims to avoid.
- **ShaRP**: Uses a per-instance bias estimate, but requires an additional forward pass on black frames and targets video settings; our goal is single-pass debiasing.
- **IVC-Prune**: Avoids attention-based importance entirely and focuses on preserving implicit coordinate tokens; our method is complementary and tests whether attention debiasing can be made calibration-free.
- **Feather**: Analyzes RoPE-induced bias and proposes multi-stage pruning and positional fixes; we focus specifically on replacing offline priors with an online prior.

**Novelty Kill Search Summary:** Searched for the exact combination “shallow-layer attention used as online bias prior for VLM token pruning” using queries including: “shallow layer attention prior debias token pruning vision language model”, “instance-level positional bias prior attention normalization token pruning VLM”, “layer ratio attention debiasing token pruning”, and “A_mid / A_shallow token pruning”. Also searched the local KB and finalized proposals for “shallow-layer” + “token pruning” + “debias”. After reviewing closely related debiasing work (FEATHER; Attention Debiasing / PoRe; Balanced Token Pruning; FasterVLM), found no prior work that normalizes mid-layer attention by shallow-layer attention as a **single-pass per-instance bias prior** as of 2026-02-24.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| D²Pruner | Offline positional prior + divide attention + MIS selection | Needs offline calibration set and generic prompt distribution | Replace offline prior with per-instance shallow-layer attention | Removes calibration and adapts bias per instance |
| PoRe | Dataset-level curve reweighting of attention | Still needs dataset statistics; not per-instance | Use per-instance bias estimate | Better under prompt/domain shift |
| ShaRP | Per-instance bias from black-frame pass | Requires extra pass | Use same-pass attention at shallow layer | Lower overhead; simpler deployment |
| IVC-Prune | Preserves implicit coordinate tokens, uses value similarity | Different objective; not an attention debiasing method | Provide attention debiasing alternative | Could be simpler when attention maps already used |
| Feather | Two-stage pruning + positional fixes | Multi-stage complexity; may still need heuristics | One-shot ratio normalization | Minimal intervention with clear attribution |

---

## Experiments

### Experimental Setup

We propose a two-phase evaluation.

**Phase-0 (go/no-go diagnostic).**
- Dataset: sample ~30 images from RefCOCO (or COCO val) and use 5 prompt templates per image.
- Compute attention vectors \(A_{shallow}\) for K_s ∈ {1,2,3} and \(A_{mid}\) for a fixed K_m (default 8 or 12).
- Metrics:
  1) Prompt stability: mean cosine similarity of attention vectors across prompts for the same image.
  2) Position correlation: Spearman rank correlation between attention and token row/col indices and distance-to-bottom/periphery (to reduce sensitivity to outliers and nonlinearity).
  3) Entropy: attention entropy per layer.
- Choose K_s as the layer that maximizes (prompt-stability × position-correlation).

**Phase-1 (main benchmark).**
- Benchmark: RefCOCO, RefCOCO+, RefCOCOg (referring expression comprehension; evaluate whether the model localizes the referred object).
- Metric: standard grounding accuracy used by D²Pruner / IVC-Prune (typically IoU≥0.5 correctness).

**Baseline Ladder (REQUIRED):**
- Upper bound (no pruning; 100% tokens)
- FastV-style raw attention pruning (no debias)
- **FEATHER** RoPE-free criterion (+ uniform sampling) as a calibration-free debiased baseline
- D²Pruner (offline prior) as strongest comparable attention-debiased baseline
- **Ours**: D²Pruner selection pipeline with online prior \(A_{shallow}\)

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| LLaVA-1.5 | 7B | https://huggingface.co/liuhaotian/llava-v1.5-7b | Common baseline for pruning papers |
| InternVL2.5 | 8B | https://huggingface.co/OpenGVLab/InternVL2_5-8B | Strong grounding results in D²Pruner |

(Verification should pick one primary model for the decisive experiment; the other model can be an optional robustness check if budget allows.)

**Other Resources (if applicable):**
- D²Pruner codebase (selection + eval harness): https://github.com/EvelynZhang-epiclab/D2Pruner
- IVC-Prune codebase (RefCOCO eval scripts): https://github.com/FireRedTeam/IVC-Prune
- PoRe codebase (for reference / optional baseline): https://github.com/intcomp/PoRe

**Resource Estimate**:
- **Compute budget**: Phase-0 is negligible (\~150 forward passes). Phase-1: on a 1000-sample subset per dataset split, 4 methods (upper bound + FastV + D²Pruner + ours) implies \~4000 inferences. Estimated \<100 GPU-hours on 1×A100-80GB for a 7B–8B model (exact runtime depends on whether attention maps require disabling FlashAttention).
- **GPU memory**: likely 1×A100-80GB per model for inference; attention extraction may increase memory.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| RefCOCO / RefCOCO+ / RefCOCOg | Referring expression grounding on COCO images | Grounding accuracy (IoU≥0.5) | val/testA/testB/test | https://github.com/lichengunc/refer | Use eval from D²Pruner/IVC-Prune repos |

### Main Results

We will report results at fixed token keep ratios matching D²Pruner’s settings (copied baseline numbers below from D²Pruner Table 3).

| Method | Base Model | Keep ratio | Benchmark | Metric (mean±std) | Source | Notes |
|--------|------------|-----------|-----------|------------------|--------|-------|
| Upper bound (all tokens) | InternVL2.5-8B | 100% | RefCOCO/+/g (avg) | 100.0% | [D²Pruner Table 3](./references/D²Pruner-Debiased-Importance-and-Structural-Diversity-for-MLLM-Token-Pruning/sections/Results%20on%20localization%20task..md) | Published (1 run) |
| FastV | InternVL2.5-8B | 10% | RefCOCO/+/g (avg) | 34.50% | [D²Pruner Table 3](./references/D²Pruner-Debiased-Importance-and-Structural-Diversity-for-MLLM-Token-Pruning/sections/Results%20on%20localization%20task..md) | Published (1 run) |
| D²Pruner (offline prior) | InternVL2.5-8B | 10% | RefCOCO/+/g (avg) | 85.68% | [D²Pruner Table 3](./references/D²Pruner-Debiased-Importance-and-Structural-Diversity-for-MLLM-Token-Pruning/sections/Results%20on%20localization%20task..md) | Published (1 run) |
| **Ours (online shallow prior)** | InternVL2.5-8B | 10% | RefCOCO/+/g (avg) | **TBD** | - | To be verified |

**Comparable calibration-free baseline (different base model):**

| Method | Base Model | Keep ratio | Benchmark | Metric | Source | Notes |
|--------|------------|-----------|-----------|--------|--------|-------|
| FEATHER (φ_{-R}+uniform, K=8, R=0.75) | LLaVA-1.5-7B | 25% | RefCOCO/+/g (avg over datasets; Table 1 Avg) | 0.356 | [FEATHER Table 1](./references/Feather-the-Throttle-Revisiting-Visual-Token-Pruning-for-Vision-Language-Model-Acceleration/sections/4.1%20Evaluating%20pruning%20criteria.md) | Not directly comparable to InternVL2.5-8B; included to show strength of calibration-free debiasing on LLaVA family |


(If verification uses LLaVA-1.5-7B instead, use the corresponding rows from D²Pruner Table 3 to fill baseline numbers.)

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | A_mid / (A_shallow+eps) | Best calibration-free performance |
| A_mid-only | Use A_mid as score (no normalization) | Worse on grounding if positional bias is major driver |
| Layer choice stress test | Use K_s ∈ {1,2,3} chosen by Phase-0 vs fixed K_s | If Phase-0 is meaningful, chosen K_s should be best |

### Experimental Rigor

- **Determinism / seeds**: Use greedy decoding / deterministic generation settings if the grounding output is deterministic; otherwise run 3 seeds and report mean±std.
- **Sanity checks**:
  - When keep ratio = 100%, all methods match the upper bound output.
  - For Phase-0, verify that the same prompt repeated twice yields identical attention vectors.
- **Top confounders and controls**:
  - *Entropy/scale confound across layers*: report attention entropy; include A_mid-only ablation to check whether the denominator is doing non-trivial work.
  - *Head specialization*: if Phase-0 fails with head-averaged attention, optionally test per-head position correlation and restrict A_shallow to strongly positional heads.

---

## Success Criteria

**Hypothesis** (directional): The online shallow-layer prior will remove most positional bias in mid-layer attention scores, yielding token selections that preserve spatial grounding accuracy close to D²Pruner while removing the need for an offline prior.

**Decision Rule** (concrete):
- **Proceed** if Phase-0 passes the go/no-go thresholds and, at 10% keep ratio, **Ours** is within **1.0 absolute point** of D²Pruner on the **mean** RefCOCO/+/g accuracy (avg over datasets), evaluated with the same decoding settings and (if non-deterministic) averaged over 3 seeds.
- **Pivot** if Phase-0 passes but Ours underperforms D²Pruner by 1–5 points: (i) restrict \(A_{shallow}\) to the subset of heads with strongest position correlation, or (ii) use an entropy-matched normalization (e.g., temperature-scale \(A_{shallow}\) to match \(A_{mid}\) entropy before division).
- **Pivot (alternative)** if Phase-0 shows shallow attention is prompt-stable but **not** position-correlated: test whether shallow attention instead tracks low-level saliency, and try combining \(A_{mid}\) with a simple vision-side positional/edge prior (e.g., Sobel magnitude) rather than using \(A_{shallow}\) as the denominator.
- **Refute** if Phase-0 fails thresholds (shallow not more prompt-stable **and** not more position-correlated than mid), or if Ours is not clearly better than A_mid-only (within ±1 point), indicating shallow normalization adds no value.

---

## Impact Statement

If successful, this provides a practical way to deploy attention-debiased token pruning without collecting calibration datasets or running extra forward passes. Practitioners building efficient multimodal systems (mobile/edge VLMs, high-resolution document QA, long multimodal chat) could adopt a single-pass debiasing rule that is easier to port across domains and prompt styles than offline priors.

---

## References

- [D²Pruner: Debiased Importance and Structural Diversity for MLLM Token Pruning](./references/D²Pruner-Debiased-Importance-and-Structural-Diversity-for-MLLM-Token-Pruning/meta/meta_info.txt) - Zhang et al., 2025
- [Attention Debiasing for Token Pruning in Vision Language Models](./references/Attention-Debiasing-for-Token-Pruning-in-Vision-Language-Models/meta/meta_info.txt) - Zhao et al., 2025
- [IVC-Prune: Revealing the Implicit Visual Co-ordinates in LVLMs for Vision Token Pruning](./references/IVC-PRUNE-REVEALING-THE-IMPLICIT-VISUAL-CO-ORDINATES-IN-LVLMS-FOR-VISION-TOKEN-PRUNING/meta/meta_info.txt) - Sun et al., 2026
- [ShaRP: SHAllow-LayeR Pruning for Video Large Language Models Acceleration](./references/ShaRP-SHAllow-LayeR-Pruning-for-Video-Large-Language-Models-Acceleration/meta/meta_info.txt) - Xia et al., 2025
- [Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](./references/Feather-the-Throttle-Revisiting-Visual-Token-Pruning-for-Vision-Language-Model-Acceleration/meta/meta_info.txt) - Endo et al., 2024
- [Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs](./references/Beyond-Text-Visual-Attention-Exploiting-Visual-Cues-for-Effective-Token-Pruning-in-VLMs/meta/meta_info.txt) - Zhang et al., 2024
- [Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](./references/Balanced-Token-Pruning-Accelerating-Vision-Language-Models-Beyond-Local-Optimization/meta/meta_info.txt) - Li et al., 2025
- [NÜWA: Mending the Spatial Integrity Torn by VLM Token Pruning](../../papers/paper_summaries/N%20%C3%9CWA%20MENDING%20THE%20SPATIAL%20INTEGRITY%20TORN%20BY%20VLM%20TOKEN%20PRUNING.md) - 2026
- [MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs](../../papers/paper_summaries/MMTok%20Multimodal%20Coverage%20Maximization%20for%20Efficient%20Inference%20of%20VLMs.md) - 2025
- [FlashVLM: Text-Guided Visual Token Selection for Large Multimodal Models](../../papers/paper_summaries/FlashVLM%20Text-Guided%20Visual%20Token%20Selection%20for%20Large%20Multimodal%20Models.md) - 2025
- [Representation Shift: Unifying Token Compression with FlashAttention](../../papers/paper_summaries/Representation%20Shift%20Unifying%20Token%20Compression%20with%20FlashAttention.md) - 2025
- [ZSPAPrune: Zero-Shot Prompt-Aware Token Pruning for Vision-Language Models](../../papers/paper_summaries/ZSPAPrune%20Zero-Shot%20Prompt-Aware%20Token%20Pruning%20for%20Vision-Language%20Models.md) - 2025
- [Multi-Cue Adaptive Visual Token Pruning for Large Vision-Language Models](../../papers/paper_summaries/Multi-Cue%20Adaptive%20Visual%20Token%20Pruning%20for%20Large%20Vision-Language%20Models.md) - 2025
- [AdaTP: Attention-Debiased Token Pruning for Video Large Language Models](../../papers/paper_summaries/AdaTP%20Attention-Debiased%20Token%20Pruning%20for%20Video%20Large%20Language%20Models.md) - 2025
- [PPE: Positional Preservation Embedding for Multimodal Large Language Models](../../papers/paper_summaries/Positional%20Preservation%20Embedding%20for%20Multimodal%20Large%20Language%20Models.md) - 2025
- [LVLM_CSP: Accelerating Large Vision Language Models via Clustering, Scattering, and Pruning for Reasoning Segmentation](../../papers/paper_summaries/LVLM_CSP%20Accelerating%20Large%20Vision%20Language%20Models%20via%20Clustering,%20Scattering,%20and%20Pruning%20for%20Reasoning%20Segmentation.md) - 2025
- [Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](../../papers/paper_summaries/Token%20Pruning%20in%20Multimodal%20Large%20Language%20Models%20Are%20We%20Solving%20the%20Right%20Problem.md) - 2025
