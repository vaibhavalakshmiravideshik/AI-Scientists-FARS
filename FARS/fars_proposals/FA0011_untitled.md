# untitled

# Caption Distillation for Mitigating the Long-Caption Paradox in ReVision-Style Text-Only MLLM Pretraining

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Constraints**:
  - Fully automated evaluation (no human ratings/annotation in the loop)
  - No OpenAI API models required (to avoid Azure OpenAI content filter constraints)
  - Verification compute budget must be ≤768 NVIDIA A100 GPU-hours (GPU-hours = number of GPUs × hours; we target ≤300 A100 GPU-hours)

## Introduction

### Context and Motivation

Multimodal large language models (MLLMs) combine a language model with a vision encoder so that the model can answer questions about images and generate text grounded in visual inputs. A widely used design is to encode an image into a sequence of continuous vectors (often called *visual tokens*) and feed these vectors to a language model as additional conditioning.

Many MLLMs rely on a pretrained dual-encoder representation space, where an image encoder and a text encoder are trained contrastively to produce comparable embeddings. A prominent example is Contrastive Language-Image Pre-training (CLIP), which maps an image and its caption into a shared embedding space. Even when the encoders are strong, the distributions of image embeddings and text embeddings are typically not identical: matched image–text pairs can still exhibit a systematic offset between modalities. This distribution mismatch is often referred to as the *modality gap*.

Recent work shows that this geometry can be exploited for cheaper pretraining: **[Modality Gap-Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/meta/meta_info.txt)** proposes **ReAlign** (a closed-form, training-free transform that maps text embeddings to match low-order image-embedding statistics while preserving anisotropy, i.e., direction-dependent variance) and **ReVision** (a two-stage MLLM recipe that uses ReAlign to convert unpaired text into *pseudo-visual embeddings* for stage-1 pretraining before standard visual instruction tuning).

### The Problem

The ReAlign/ReVision paper reports a counterintuitive failure case called the **Long-Caption Paradox**: using longer, denser captions for ReVision stage-1 pretraining hurts downstream MLLM performance compared to using concise captions, despite the intuition that more descriptive captions should help.

Concretely, Yu et al. compare stage-1 pretraining on DenseFusion-1M captions (long, multi-sentence) against stage-1 pretraining on Bunny-1M captions (concise captions used by the authors as a short-caption corpus). Under the same ReAlign + ReVision pipeline, the long-caption variant underperforms across multiple benchmarks (Appendix G, Table 4 in **[ReAlign/ReVision](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/G The Long-Caption Paradox.md)**). For example (all scores are accuracy in %, higher is better):

- **MMStar**: 36.13 (ReVision) vs 33.40 (ReVision-Long)
- **ScienceQA-IMG**: 76.71 (ReVision) vs 74.84 (ReVision-Long)
- **POPE**: 72.53 (ReVision) vs 71.47 (ReVision-Long)

The paper also provides geometric evidence that long captions change the text-embedding distribution in ways that make moment-matching alignment harder:

- **Effective rank** (a scalar proxy for how many principal directions a covariance uses; higher means more diffuse/high-entropy covariance): DenseFusion captions have higher effective rank (≈52.9) than short captions (≈41.0) (**[Appendix G.2](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/G.2 Diffuse Covariance Structure.md)**).
- **Initial centroid gap** (the Euclidean distance between the mean image embedding and mean text embedding; larger means a larger modality gap): DenseFusion captions widen the initial centroid gap (‖Δμ‖≈0.51) compared to short captions (‖Δμ‖≈0.39) (**[Appendix G.3](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/G.3 Linguistic Noise & Modality Gap.md)**).

This matters because long-caption datasets are a major community effort for improving visual supervision. For example, **[DenseFusion-1M](./references/DenseFusion-1M-Merging-Vision-Experts-for-Comprehensive-Multimodal-Perception/meta/meta_info.txt)** provides ~1M images with hyper-detailed captions (≈190 words and ≈11 sentences on average) and is motivated as a general-purpose data resource for multimodal learning. If long captions systematically degrade ReVision-style text-only pretraining, practitioners need a low-cost mitigation that preserves the benefits of dense captions while avoiding alignment failure.

### Key Insight and Hypothesis

**Key insight**: Many long captions mix visually grounded content (objects, text in the image, spatial relations) with content that is weakly grounded or not grounded in the image (speculation, narrative tone, redundant paraphrases). If the Long-Caption Paradox is driven mainly by this weakly grounded content acting as noise in the text embedding space (rather than by length alone), then we should be able to recover performance by *distilling* long captions into a shorter set of sentences that are strongly image-grounded.

**Hypothesis**: A simple caption distillation preprocessing step—selecting a small set of image-grounded sentences from each DenseFusion caption using a frozen CLIP similarity score—will mitigate the Long-Caption Paradox. Under a compute-matched ReVision-style training recipe, **CLIP-scored sentence selection** should improve downstream MLLM accuracy relative to using raw long captions, while **random length-matched sentence selection** should not.

Why we could be wrong:

- The paradox may be dominated by truncation or encoder context limits (**[Appendix G.1](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/G.1 Truncation-Induced Supervision Mismatch.md)**), in which case any shortening would help and random selection would match CLIP-scored selection.
- The additional content in DenseFusion captions may be genuinely useful for ReVision, and distillation could remove useful supervision.
- CLIP similarity may prefer sentences that are easy for CLIP to align (high similarity) but not necessarily the most useful for downstream instruction tuning.

---

## Proposed Approach

### Overview

We propose **Caption Distillation for ReVision (CD-ReVision)**: before running a ReAlign/ReVision-style pipeline on a long-caption dataset (DenseFusion-1M), convert each long caption into a compact caption by selecting only the most image-grounded sentences.

To isolate whether any gains come from content selection rather than length reduction, we include a **length-matched random selection** control. The verification experiment uses exactly three conditions:

- **A (Long captions)**: Original DenseFusion captions.
- **B (Random length-matched)**: Randomly sample sentences until a fixed length budget is reached.
- **C (CLIP-scored distillation; ours)**: Select sentences by CLIP similarity, then pack to the same length budget.

### Method Details

**Dataset and caption statistics**: DenseFusion-1M contains ~1M images with long, multi-sentence captions (≈191 words and ≈11 sentences on average) (**[DenseFusion-1M](./references/DenseFusion-1M-Merging-Vision-Experts-for-Comprehensive-Multimodal-Perception/sections/3.3 Dataset Description.md)**).

**Sentence segmentation**: Split each caption into sentences using a deterministic sentence splitter (e.g., spaCy sentencizer). This must be deterministic so that A/B/C differ only in selection, not segmentation variability.

**Sentence scoring (image-groundedness)**:

- Use a frozen CLIP model (e.g., `openai/clip-vit-large-patch14-336`) to compute an embedding for the image and for each candidate sentence.
- Score each sentence by cosine similarity, where cosine similarity is the dot product of L2-normalized embeddings (higher means more aligned):

  \(\text{score}(s_{ij}) = \cos( f_{img}(x_i), f_{txt}(s_{ij}) )\).

**Length budget (compute-matching)**:

- Construct a distilled caption by greedily adding sentences in descending score order until a fixed **word budget** \(W\) is reached (default \(W=60\) words).
- If a candidate sentence would exceed \(W\), skip it and continue (no mid-sentence truncation).
- Apply the same \(W\) in condition B (random selection) for a fair length-matched control.

**ReVision-style training pipeline (kept identical across A/B/C)**:

- **Architecture**: follow ReVision’s architecture: a frozen long-text encoder **LLM2CLIP** (a text encoder designed to produce CLIP-compatible embeddings for longer texts), a language model backbone (Llama-3-8B), and a 2-layer MLP (multi-layer perceptron) projector with GELU (Gaussian error linear unit) activation (**[H.2](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/H.2 MLLM Training Setting.md)**).
- **Stage 1 (modality substitution pretraining)**: treat the chosen caption variant as an unpaired text corpus; apply ReAlign to map text embeddings into the image embedding distribution (Eq. 25 in **[Stage 1](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/Stage 1 Modality Substitution Pretraining.md)**), then train the projector (with the language model frozen) to reconstruct the caption tokens conditioned on the pseudo-visual embedding (Eq. 26).
- **Stage 2 (visual instruction tuning)**: supervised fine-tuning (SFT; standard cross-entropy training on instruction/response pairs) on a fixed visual instruction dataset, keeping the dataset and hyperparameters identical across A/B/C.

**Verification-friendly scaling (Micro-ReVision)**:

To keep verification within the compute budget while preserving the causal structure of the experiment, we downscale data while keeping the pipeline intact:

- Stage 1: sample **200k** DenseFusion examples for one epoch.
- Stage 2: sample **50k** SFT examples for one epoch.

This micro setting is not intended to reproduce the absolute numbers from Yu et al.; it is intended to decide the *relative* effect of caption distillation under a controlled, compute-matched ReVision-style pipeline.

**Secondary (explanatory) analyses (not part of the decision rule)**:

- Effective rank of caption embedding covariances for A/B/C.
- Initial centroid gap \(\|\Delta\mu\|\) between image and text embeddings for A/B/C.
- Image–text retrieval accuracy on DenseFusion pairs as a proxy for how well distillation preserves image-grounded semantics.

### Key Innovations

- **A data-only intervention** for the Long-Caption Paradox that requires no change to ReAlign/ReVision model components.
- **A length-matched control** (random sentence selection) that distinguishes “shorter captions help” from “better caption content helps.”
- **A verification-ready, compute-bounded test** (three conditions, fixed training recipe, automated evaluation) that produces a decision about whether content-aware caption selection mitigates the paradox.

---

## Related Work

### Field Overview

This proposal connects three research threads. First, work on multimodal contrastive representation geometry studies how image/text embedding distributions differ after contrastive pretraining, including anisotropy (direction-dependent variance) and persistent modality gaps. Second, several methods exploit the learned embedding space to enable cross-modal training from uni-modal data (e.g., using only text to train image-conditioned models), but their success depends on how accurately text-derived features match the geometry of real image features. Third, dataset and caption engineering work produces increasingly detailed captions to improve multimodal training, raising the question of when “more descriptive text” helps or hurts different training paradigms.

The Long-Caption Paradox in ReVision is a concrete case where richer captions degrade a method that relies on distribution matching between modalities. Understanding whether this is fundamentally a length/truncation issue or a content/noise issue is important for practitioners who invest in long-caption data pipelines.

### Related Papers

- **[Modality Gap-Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/meta/meta_info.txt)**: Introduces ReAlign/ReVision and reports the Long-Caption Paradox that motivates our study.
- **[DenseFusion-1M: Merging Vision Experts for Comprehensive Multimodal Perception](./references/DenseFusion-1M-Merging-Vision-Experts-for-Comprehensive-Multimodal-Perception/meta/meta_info.txt)**: Provides a large-scale long-caption dataset (DenseFusion-1M) representative of modern dense captioning pipelines.
- **[Connect, Collapse, Corrupt: Learning Cross-Modal Tasks with Uni-Modal Data](./references/Connect-Collapse-Corrupt-Learning-Cross-Modal-Tasks-with-Uni-Modal-Data/meta/meta_info.txt)**: Proposes C^3, a modality-gap correction method based on mean shift plus isotropic noise, providing a baseline family for embedding-space alignment.
- **[Unicorn: Text-Only Data Synthesis for Vision Language Model Training](./references/Unicorn-Text-Only-Data-Synthesis-for-Vision-Language-Model-Training/meta/meta_info.txt)**: Demonstrates text-only vision-language model training via embedding transfer and highlights strengths/weaknesses of simple moment matching.
- **[HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale](./references/HuatuoGPT-Vision-Towards-Injecting-Medical-Visual-Knowledge-into-Multimodal-LLMs-at-Scale/meta/meta_info.txt)**: Shows a domain where long, technical captions are common and motivates practical interest in caption quality trade-offs.
- **[CLIP](https://arxiv.org/abs/2103.00020)**: Establishes the contrastive vision–language embedding space that underlies many modality-gap analyses and CLIP-based data scoring.
- **[SigLIP](https://arxiv.org/abs/2303.15343)**: Uses a sigmoid-based contrastive loss that changes embedding statistics, relevant to how modality-gap properties depend on the pretraining objective.
- **[EVA-CLIP](https://arxiv.org/abs/2303.15389)**: Improves CLIP training at scale and is widely used in retrieval and data curation, highlighting the practical role of CLIP-style scoring.
- **[LLM2CLIP](https://arxiv.org/abs/2411.04997)**: Provides a long-context text encoder that produces CLIP-compatible embeddings, relevant to whether the paradox is caused by text encoder capacity.
- **[LLaVA](https://arxiv.org/abs/2304.08485)**: Introduces a common two-stage MLLM training recipe (pretraining + instruction tuning) that ReVision follows structurally.
- **[LLaVA-NeXT](https://arxiv.org/abs/2401.13621)**: Improves the LLaVA family with stronger backbones and settings, relevant as a general vision-language model baseline context for DenseFusion-style training.
- **[InternVL](https://arxiv.org/abs/2312.14238)**: Provides an open-source MLLM family and the instruction-tuning dataset used by ReVision for stage 2.
- **[ShareGPT4V](https://arxiv.org/abs/2311.12793)**: Studies improving vision-language models via better captions, connecting caption quality to downstream multimodal performance.
- **[ALLaVA](https://arxiv.org/abs/2402.05874)**: Uses detailed synthetic captions/instructions for multimodal training, reinforcing the trend toward longer, richer text supervision.
- **[DOCCI](https://arxiv.org/abs/2306.05399)**: Provides detailed human-authored image descriptions as a high-quality long-caption resource.
- **[ImageInWords](https://arxiv.org/abs/2406.09721)**: Collects hyper-detailed captions with human involvement, offering evidence that caption richness can help in some settings.
- **[Visual Fact Checker](https://arxiv.org/abs/2404.03167)**: Improves caption fidelity via verification, relevant to distinguishing accurate detail from noisy elaboration.
- **[LaCLIP](https://arxiv.org/abs/2310.17720)**: Uses language rewrites to improve CLIP alignment, suggesting that text rewriting can change embedding geometry.
- **[CapsFusion](https://arxiv.org/abs/2403.11660)**: Consolidates/refines captions from multiple sources, adjacent to our goal of producing higher-signal captions.
- **[DeCap](https://arxiv.org/abs/2303.03089)**: Learns captioning from CLIP latents using text-only training, demonstrating CLIP-space manipulation for cross-modal tasks.
- **[ZeroCap](https://arxiv.org/abs/2203.03665)**: Early evidence of using CLIP embeddings for zero-shot captioning via embedding-space techniques.
- **[Long-CLIP](https://arxiv.org/abs/2403.15378)**: Extends CLIP to longer text, relevant if the paradox is primarily caused by long-text encoding limitations.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Modality-gap correction in embedding space | Post-hoc alignment of image/text embeddings (mean shifts, noise injection, anisotropy-aware transforms) | **[C^3](./references/Connect-Collapse-Corrupt-Learning-Cross-Modal-Tasks-with-Uni-Modal-Data/meta/meta_info.txt)**, **[ReAlign/ReVision](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/meta/meta_info.txt)** | Retrieval; vision-language model benchmark suites (e.g., MMStar, ScienceQA, POPE) | Can be sensitive to distribution shape and domain mismatch; alignment can distort semantics |
| Text-only / uni-modal vision-language model training | Use uni-modal data (often text) to train cross-modal models via embedding-space substitution | **[Unicorn](./references/Unicorn-Text-Only-Data-Synthesis-for-Vision-Language-Model-Training/meta/meta_info.txt)**, **[DeCap](https://arxiv.org/abs/2303.03089)**, **[ZeroCap](https://arxiv.org/abs/2203.03665)** | Captioning and visual question answering (VQA) benchmarks | Often lags on fine-grained perception; success depends on embedding transfer quality |
| Caption and data refinement | Improve caption quality/coverage using fusion, rewriting, or verification | **[DenseFusion-1M](./references/DenseFusion-1M-Merging-Vision-Experts-for-Comprehensive-Multimodal-Perception/meta/meta_info.txt)**, **[ShareGPT4V](https://arxiv.org/abs/2311.12793)**, **[DOCCI](https://arxiv.org/abs/2306.05399)** | Vision-language model benchmark suites; caption fidelity metrics | Longer captions can introduce weakly grounded content; quality varies across pipelines |
| Long-text encoder improvements | Extend encoders to handle longer prompts/captions in CLIP-style spaces | **[LLM2CLIP](https://arxiv.org/abs/2411.04997)**, **[Long-CLIP](https://arxiv.org/abs/2403.15378)** | Retrieval; downstream vision-language model performance | Encoder capacity does not necessarily address noise or covariance diffusion |

### Closest Prior Work

1. **ReAlign/ReVision (Yu et al., 2026)**: ReAlign defines a closed-form mapping from text embeddings to match low-order statistics of image embeddings while preserving anisotropy, and ReVision uses this mapping to pretrain an MLLM from unpaired text via pseudo-visual embeddings (Eq. 25–26). In Appendix G, the authors show that replacing their concise stage-1 corpus (Bunny-1M) with DenseFusion-1M long captions reduces downstream benchmark accuracy, despite keeping the model, encoder, and alignment strategy fixed. The paper analyzes possible causes (truncation mismatch, diffuse covariance, and lower signal-to-noise ratio) but does not test a concrete mitigation.

2. **DenseFusion-1M (Li et al., 2024)**: DenseFusion introduces a pipeline that merges outputs from multiple vision experts to generate hyper-detailed captions at scale, producing a ~1M image dataset with long, multi-sentence descriptions. The paper demonstrates gains for image-conditioned multimodal training recipes, supporting the intuition that richer captions can help. It does not study how dense captions interact with text-only or embedding-substitution pretraining methods that rely on distribution matching.

3. **Connect, Collapse, Corrupt (C^3; Zhang et al., 2024)**: C^3 analyzes the modality gap in contrastive embedding spaces and proposes corrections based on centering (mean subtraction) plus isotropic noise injection to enable cross-modal tasks from uni-modal data. This provides a baseline family for “fixing” modality mismatch, but its isotropic noise assumption can be limiting when the visual embedding covariance is strongly anisotropic. It does not specifically analyze long-caption effects on the shape of the text embedding distribution.

4. **Unicorn (Yu et al., 2025)**: Unicorn trains vision-language models from text by transferring text representations into a visual representation space (a mean-shift style transfer) and shows that text-only training can be competitive on some benchmarks. However, it does not address the Long-Caption Paradox and does not propose data-side fixes for caption-induced diffusion or noise in the text embedding distribution.

### Comparison Table

| Related work | What it does | Key limitation (for this proposal) | What we change | Why ours should win |
|---|---|---|---|---|
| ReAlign/ReVision | Training-free alignment + text-only stage-1 pretraining via pseudo-visual embeddings | Reports Long-Caption Paradox but does not test mitigations | Keep ReAlign/ReVision fixed; change only the stage-1 caption text via distillation | If weakly grounded sentences are the main cause, data-side filtering should reduce diffusion/noise without changing the method |
| DenseFusion-1M | Produces long, detailed captions to improve multimodal training | Does not distinguish when long captions help vs hurt across training paradigms | Distill DenseFusion captions for ReVision-style stage-1 pretraining | Keeps image-grounded details while removing content that disrupts embedding alignment |
| C^3 (Connect, Collapse, Corrupt) | Mean shift + isotropic noise for cross-modal transfer | Isotropic assumption; not tailored to caption-induced covariance changes | Use ReAlign (anisotropy-aware) and only modify captions | Simpler intervention than changing alignment; targets the hypothesized data-side failure mode |
| Unicorn | Text-only vision-language model training via simple representation transfer | No analysis of long captions; limited handling of distribution shape | Use ReAlign + caption distillation | Combines stronger distribution matching with higher-signal captions |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3-8B-Instruct | 8B parameters | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct | Language model backbone for the MLLM (used by ReVision) |
| LLM2CLIP-Openai-L-14-336 | ViT-L/14 text encoder (Vision Transformer backbone) | https://huggingface.co/microsoft/LLM2CLIP-Openai-L-14-336 | Frozen long-text encoder producing CLIP-compatible text embeddings |
| CLIP ViT-L/14-336 (scoring only) | ViT-L/14 (Vision Transformer backbone) | https://huggingface.co/openai/clip-vit-large-patch14-336 | Used only for sentence scoring in caption distillation (no training) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| DenseFusion-1M | Stage-1 caption corpus (captions) + paired images for sentence scoring | ~1M total; use a fixed 200k subset for verification | https://huggingface.co/datasets/BAAI/DenseFusion-1M | See dataset card |
| InternVL-Chat-V1.2-SFT | Stage-2 supervised fine-tuning (SFT) for visual instruction tuning | large; use a fixed 50k subset for verification | https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT | See dataset card |

**Other Resources (if applicable):**

- None.

**Resource Estimate**:

- **Compute budget (A100 GPU-hours)**: target ≤300 total.
  - **Evidence-based anchor**: Yu et al. report 12 hours on 8× NVIDIA H200 for ~2.2M samples for the full two-stage pipeline (**[H.2](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/H.2 MLLM Training Setting.md)**), i.e., 96 H200-GPU-hours.
  - **Downscaling**: Micro-ReVision uses 250k total samples (200k stage-1 + 50k stage-2), about 0.11× the paper’s 2.2M samples. If training cost scales roughly with sample count, one A/B/C run is on the order of 96 × 0.11 ≈ 11 H200-GPU-hours.
  - **Three conditions**: A/B/C totals ≈33 H200-GPU-hours.
  - **Multiple seeds**: the decision rule repeats A and C for 2 seeds (adds one extra A run and one extra C run), so training totals are ≈33 + 2×11 ≈ 55 H200-GPU-hours.
  - **Conservative A100 conversion**: even if an A100 is up to ~3× slower than an H200 for this workload, this suggests ≲165 A100-GPU-hours for training. We reserve additional budget for implementation overhead, hyperparameter stabilization, and evaluation, reaching a conservative ≤300 A100-GPU-hours.
  - **Caption distillation scoring** (CLIP inference) is expected to be minor relative to training (single-pass embeddings for a fixed subset).
- **GPU memory**: 
  - Stage 1 (projector-only training with frozen LLM) should fit comfortably on 1×A100-80GB.
  - Stage 2 SFT may require multi-GPU training depending on the training stack; the verification run can use distributed training (e.g., fully sharded data parallel training) while keeping A/B/C matched.
- **API usage**: none required.

### Benchmarks and Metrics

All evaluation is automated. For consistency with Yu et al. (**[H.3](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/H.3 Eval Setting.md)**), we use **accuracy** as the primary metric on all benchmarks.

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MMStar | A multiple-choice benchmark for general multimodal understanding across diverse visual content | Accuracy (percent correct; higher is better) | test | https://huggingface.co/datasets/Lin-Chen/MMStar | VLMEvalKit / official script |
| ScienceQA-IMG | A multimodal science question-answering benchmark with images/diagrams and multiple-choice answers | Accuracy (percent correct; higher is better) | test | https://huggingface.co/datasets/derek-thomas/ScienceQA | Official eval / VLMEvalKit |
| POPE | A yes/no benchmark designed to measure object hallucination by asking whether objects are present in an image | Accuracy (percent correct; higher is better) | test | https://huggingface.co/datasets/lmms-lab/POPE | Official eval / VLMEvalKit |

**Evaluation Scripts:**

- Prefer **VLMEvalKit** (https://github.com/open-compass/VLMEvalKit) for standardized prompting and scoring where supported.
- Otherwise, use each benchmark’s official evaluation script (linked above).

**Download Links Checklist:**

- [ ] All benchmark datasets have download links
- [ ] All training datasets have download links
- [ ] All models have download links
- [ ] Licenses are compatible with research use

### Main Results

#### Published baseline evidence (Long-Caption Paradox)

The following numbers are copied verbatim from Appendix G (“The Long-Caption Paradox”) of **[ReAlign/ReVision](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/G The Long-Caption Paradox.md)**. All benchmark scores are accuracy in %, higher is better. “Avg (paper)” is the average accuracy over the full evaluation suite in Appendix G (higher is better).

| Method (paper) | Stage-1 caption source | MMStar (accuracy %, ↑) | ScienceQA-IMG (accuracy %, ↑) | POPE (accuracy %, ↑) | Avg (paper, accuracy %, ↑) | Reference |
|---|---|---:|---:|---:|---:|---|
| ReVision | Bunny-1M concise captions | 36.13 | 76.71 | 72.53 | 50.16 | **[Appendix G](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/G The Long-Caption Paradox.md)** |
| ReVision-Long | DenseFusion-1M long captions | 33.40 | 74.84 | 71.47 | 48.73 | **[Appendix G](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/sections/G The Long-Caption Paradox.md)** |

#### Planned verification runs (this proposal; to be filled)

These runs use the Micro-ReVision setting (200k stage-1 samples + 50k stage-2 samples). Results are not intended to be numerically comparable to the published full-scale scores above; the goal is a controlled A/B/C comparison.

| Method | Stage-1 caption variant | Training budget | Results (to be filled; accuracy ↑) | Reference |
|---|---|---|---|---|
| A | DenseFusion long captions (no distillation) | Micro-ReVision, fixed hyperparameters | MMStar: TBD; ScienceQA-IMG: TBD; POPE: TBD; mean accuracy over 3 benchmarks: TBD | This work |
| B | Random sentence subset, capped at \(W=60\) words | Same as A | MMStar: TBD; ScienceQA-IMG: TBD; POPE: TBD; mean accuracy: TBD | This work |
| **C (ours)** | CLIP-scored sentence subset, capped at \(W=60\) words | Same as A | MMStar: TBD; ScienceQA-IMG: TBD; POPE: TBD; mean accuracy: TBD | This work |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B (random) | Remove CLIP scoring; keep the same length budget | If only length/truncation matters, B ≈ C; if content selection matters, C > B |

### Analysis (Optional)

- Measure effective rank and centroid gap for A/B/C caption embedding distributions (to test whether distillation reduces diffusion and modality gap).
- Correlate geometric changes with downstream benchmark changes.

---

## Success Criteria

**Primary criterion: content-aware distillation improves performance beyond length truncation**

- Hypothesis: CLIP-scored caption distillation (C) improves downstream accuracy relative to long captions (A) and random length-matched truncation (B).
- Validation / decision rule:
  - Compute mean accuracy over {MMStar, ScienceQA-IMG, POPE} (higher is better).
  - Run A, B, and C under identical training budgets and evaluation code.
  - Repeat A and C for 2 random seeds.
  - We consider the hypothesis supported if C achieves higher mean accuracy than A in both seeds and also exceeds B (i.e., the improvement is not explained by length reduction alone).
  - If B is within 0.5 percentage points of C on mean accuracy, we conclude the benefit is primarily due to shortening/truncation rather than content-aware selection.
  - If C does not outperform A, we conclude that this caption distillation strategy does not mitigate the Long-Caption Paradox in this setting.

**Secondary criterion: geometric mechanism is consistent (optional)**

- Hypothesis: C reduces the long-caption-induced diffusion and modality gap relative to A.
- Validation: EffectiveRank(C) < EffectiveRank(A) and \(\|\Delta\mu\|_C < \|\Delta\mu\|_A\) on the same subset used for stage-1 training.

---

## Impact Statement

If successful, this work provides a simple preprocessing recipe that makes long-caption datasets (such as DenseFusion-1M) more compatible with ReVision-style text-only pretraining, potentially improving the practical value of large-scale dense captioning efforts without changing model architectures. If unsuccessful, it strengthens the conclusion that the Long-Caption Paradox is not primarily caused by removable weakly grounded content, motivating alternative mitigations (e.g., better long-text encoders, chunked embedding schemes, or different alignment objectives).

---

## References

- [Modality Gap-Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models](./references/Modality-Gap-Driven-Subspace-Alignment-Training-Paradigm-For-Multimodal-Large-Language-Models/meta/meta_info.txt) - Yu et al., 2026
- [DenseFusion-1M: Merging Vision Experts for Comprehensive Multimodal Perception](./references/DenseFusion-1M-Merging-Vision-Experts-for-Comprehensive-Multimodal-Perception/meta/meta_info.txt) - Li et al., 2024
- [Connect, Collapse, Corrupt: Learning Cross-Modal Tasks with Uni-Modal Data](./references/Connect-Collapse-Corrupt-Learning-Cross-Modal-Tasks-with-Uni-Modal-Data/meta/meta_info.txt) - Zhang et al., 2024
- [Unicorn: Text-Only Data Synthesis for Vision Language Model Training](./references/Unicorn-Text-Only-Data-Synthesis-for-Vision-Language-Model-Training/meta/meta_info.txt) - Yu et al., 2025
- [HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale](./references/HuatuoGPT-Vision-Towards-Injecting-Medical-Visual-Knowledge-into-Multimodal-LLMs-at-Scale/meta/meta_info.txt) - Chen et al., 2024
- [CLIP](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
- [SigLIP](https://arxiv.org/abs/2303.15343) - Zhai et al., 2023
- [EVA-CLIP](https://arxiv.org/abs/2303.15389) - Sun et al., 2023
- [LLM2CLIP](https://arxiv.org/abs/2411.04997) - Huang et al., 2024
- [LLaVA](https://arxiv.org/abs/2304.08485) - Liu et al., 2023
- [LLaVA-NeXT](https://arxiv.org/abs/2401.13621) - Liu et al., 2024
- [InternVL](https://arxiv.org/abs/2312.14238) - Chen et al., 2024
- [ShareGPT4V](https://arxiv.org/abs/2311.12793) - Chen et al., 2023
- [ALLaVA](https://arxiv.org/abs/2402.05874) - Gao et al., 2024
- [DOCCI](https://arxiv.org/abs/2306.05399) - Dutta et al., 2023
- [ImageInWords](https://arxiv.org/abs/2406.09721) - Garg et al., 2024
- [Visual Fact Checker](https://arxiv.org/abs/2404.03167) - Ge et al., 2024
- [LaCLIP](https://arxiv.org/abs/2310.17720) - Fan et al., 2023
- [CapsFusion](https://arxiv.org/abs/2403.11660) - 2024
- [DeCap](https://arxiv.org/abs/2303.03089) - Li et al., 2023
- [ZeroCap](https://arxiv.org/abs/2203.03665) - Tewel et al., 2022
- [Long-CLIP](https://arxiv.org/abs/2403.15378) - 2024
