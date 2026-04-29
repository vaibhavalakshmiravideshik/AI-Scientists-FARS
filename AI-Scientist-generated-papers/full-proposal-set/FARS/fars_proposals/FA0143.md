# untitled

# Tuned-Lens-Style Affine Alignment for Encoder Truncation in Whisper ASR

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Transformer encoder-decoder automatic speech recognition (ASR) models such as Whisper are widely used for transcription and translation because they are robust to noise and domain shift. However, Whisper-class models can be expensive to serve at low latency, especially in batched or multi-stream inference settings.

A practical bottleneck is the Whisper **audio encoder**. For a 30-second audio chunk, Whisper uses a fixed-length encoder sequence (1500 time steps after feature processing), which makes encoder self-attention and feed-forward computation large. In contrast, the decoder often generates far fewer text tokens than 1500, and decoder compute can become relatively more efficient under batching. **[LiteASR](./references/LiteASR-Efficient-Automatic-Speech-Recognition-with-Low-Rank-Approximation/meta/meta_info.txt)** reports that for Whisper large-v3-turbo at batch size 8, the encoder can account for more than 90% of the end-to-end latency (LiteASR, Sec. 2.2), making encoder compute a primary target for speedups.

Many ASR efficiency methods require substantial retraining (e.g., knowledge distillation) or structural model changes (e.g., pruning and layer merging). This proposal asks whether there is a lightweight, post-hoc alternative that can make encoder truncation practical without retraining the full model.

### The Problem

A natural way to reduce encoder latency is **fixed-depth early exit** (also called truncation): stop the encoder after layer L < N and feed the resulting hidden states into the original decoder. In Whisper, this can fail because the decoder-encoder cross-attention (the mechanism that lets the text decoder attend to encoder audio features) was trained to consume the *final* encoder representation distribution, and intermediate encoder states are out-of-distribution for that cross-attention module.

Existing directions partially address ASR efficiency but do not solve this specific issue:

- **[Distil-Whisper](./references/Distil-Whisper-Robust-Knowledge-Distillation-via-Large-Scale-Pseudo-Labelling/meta/meta_info.txt)** (knowledge distillation): trains a much smaller decoder (2 layers) and reports large latency reductions, but requires large-scale training with pseudo-labeling and filtering.
- **[LiteASR](./references/LiteASR-Efficient-Automatic-Speech-Recognition-with-Low-Rank-Approximation/meta/meta_info.txt)** (post-training low-rank + custom kernels): targets Whisper encoder bottlenecks, but relies on low-level kernel and approximation engineering rather than a simple drop-in method.
- **[BaldWhisper](./references/BaldWhisper-Faster-Whisper-with-Head-Shearing-and-Layer-Merging/meta/meta_info.txt)** (pruning/merging): changes the model structure (head shearing, layer merging) and is not a minimal post-hoc correction for naive truncation.

A simple but potentially powerful alternative is suggested by transformer interpretability work: intermediate representations may already contain the information needed for the final prediction, but in a different coordinate system. If this mismatch is mostly *representational* rather than *informational*, a small learned mapping might align intermediate encoder states to the distribution expected by the decoder.

### Key Insight and Hypothesis

**Key insight.** The tuned lens shows that for autoregressive transformers, a learned affine map can translate intermediate-layer residual streams into a space where the final unembedding yields accurate token predictions. More generally, many transformer layers can be viewed as iteratively refining a representation; earlier layers may encode most content but differ by an approximately linear transformation and scaling.

**Hypothesis.** For Whisper-style ASR, there exists an encoder layer L such that the intermediate encoder state h_L(x) already contains most of the information needed by the decoder, but is misaligned with the representation distribution of the final encoder state h_N(x). We hypothesize that learning a **tuned-lens-style affine translator** g_L(h)=A_L h + b_L (applied per token) to approximate h_N(x) from h_L(x) will:

1) significantly reduce word error rate (WER; lower is better) degradation compared to naive encoder truncation at the same L, and
2) enable meaningful end-to-end speedups in regimes where the encoder dominates latency (e.g., batched inference).

This could fail because later encoder layers may compute genuinely new non-linear features that are not recoverable from h_L using an affine map, or because encoder truncation provides little end-to-end speedup once real overheads (I/O, feature extraction, decoding) are accounted for.

---

## Proposed Approach

### Overview

We propose **Tuned-Encoder-Exit**: a post-hoc method to make fixed-depth Whisper encoder truncation usable by learning a lightweight affine translation layer that aligns intermediate encoder states to the final encoder representation space.

Given a pretrained Whisper model with encoder depth N and decoder D:

1) Choose an encoder truncation depth L < N based on a profiling target (speed).
2) Run only the first L encoder blocks to produce h_L.
3) Apply a learned affine translator g_L token-wise to produce \(\hat h_N = g_L(h_L)\).
4) Decode with the unchanged decoder D using \(\hat h_N\) as cross-attention memory.

This changes neither the decoder nor the encoder blocks; it only adds a small linear layer and enables truncation.

### Method Details

**Model notation.** Let \(E_{1:L}\) be the first L encoder layers, \(E_{1:N}\) the full encoder, and \(D\) the decoder. For an input audio chunk \(x\):

- Full encoder: \(h_N(x)=E_{1:N}(x)\)
- Truncated encoder: \(h_L(x)=E_{1:L}(x)\)

**Affine translator.** We learn parameters \(A_L \in \mathbb{R}^{d\times d}\), \(b_L \in \mathbb{R}^{d}\) where \(d\) is the model width (e.g., 1280 for Whisper large-v2):

- \(\hat h_N(x) = g_L(h_L(x)) = A_L h_L(x) + b_L\)

This is applied to each encoder token representation at each time step.

**Training objective (representation alignment).** We train only \(A_L, b_L\) with the full Whisper weights frozen. On a calibration set \(\mathcal{D}_{cal}\), we minimize an L2 alignment loss:

- \(\mathcal{L}_{repr} = \mathbb{E}_{x\sim\mathcal{D}_{cal}}\big[ \| g_L(h_L(x)) - h_N(x) \|_2^2 \big]\)

This objective directly targets the quantity consumed by decoder cross-attention (the final encoder hidden states). Because this does not require decoding text tokens during training, it is substantially cheaper than training against transcription loss.

**Optional logit distillation (analysis-only).** As an optional diagnostic, we can add a teacher-forced KL loss on decoder logits to check whether representation alignment transfers to output distributions:

- \(\mathcal{L}_{logit} = \mathbb{E}_{(x,y)}\big[ \mathrm{KL}( p_{full}(\cdot\mid x,y_{<t}) \;\|\; p_{trunc+g}(\cdot\mid x,y_{<t}) ) \big]\)

and train with \(\mathcal{L} = \mathcal{L}_{repr} + \lambda \mathcal{L}_{logit}\) for a fixed \(\lambda\) (no sweep). This is not required for the main decision.

**Profiling-guided layer selection.** To keep the verification decisive and avoid WER-based layer tuning, we will choose a single truncation depth L using only runtime profiling on a small audio subset:

- Measure end-to-end latency for encoder truncation at candidate depths (e.g., L in {8, 12, 16, 20, 24}) under the target serving regime (batch size, chunk size, greedy decoding).
- Select the **largest** L that yields at least a target speedup (e.g., **1.25** end-to-end) under naive truncation. This makes the method conservative: we only truncate as much as needed for the speed target.

### Key Innovations

- **Post-hoc, representation-level correction for Whisper encoder truncation.** Instead of retraining or redesigning the model, we learn a single affine translator to align intermediate encoder representations to the decoder-expected space.
- **A profiling gate that makes the speed claim testable.** The proposal explicitly measures whether the encoder is the dominant bottleneck in the chosen inference regime before spending effort on alignment.
- **A decisive negative result is still valuable.** If affine alignment does not improve WER vs naive truncation, it suggests that encoder later layers compute non-linear information that cannot be recovered by simple translation, motivating richer adapters or training-time approaches.

---

## Related Work

### Field Overview

ASR efficiency work spans (i) *model compression* (distillation, pruning, low-rank approximation), (ii) *decoding-time acceleration* (speculative decoding, early exit, streaming truncation), and (iii) *representation analysis and probing* (layer-wise decoding, logit/decoder lenses). For Whisper, several recent papers show meaningful speedups via distillation and architectural simplification, and others highlight that intermediate layers contain interpretable but imperfect predictions.

In parallel, early-exit and layer-skipping methods are well-studied in NLP and vision models, but most approaches require training-time modifications (LayerDrop, shallow-deep supervision) or multiple exits. The tuned lens provides evidence that intermediate transformer representations can be linearly translated into the final prediction space, suggesting a lightweight path to make truncation usable without retraining.

### Related Papers

- **[Whisper](https://arxiv.org/abs/2212.04356)**: Introduces Whisper, a robust encoder-decoder ASR model family that this proposal targets.
- **[Eliciting Latent Predictions from Transformers with the Tuned Lens](./references/Eliciting-Latent-Predictions-from-Transformers-with-the-Tuned-Lens/meta/meta_info.txt)**: Trains affine maps from intermediate residual streams to accurate next-token predictions, motivating affine alignment for truncation.
- **[DecoderLens](./references/DecoderLens-Layerwise-Interpretation-of-Encoder-Decoder-Transformers/meta/meta_info.txt)**: Studies layerwise decoding for encoder-decoder transformers and shows intermediate representations differ qualitatively from final outputs.
- **[Distil-Whisper](./references/Distil-Whisper-Robust-Knowledge-Distillation-via-Large-Scale-Pseudo-Labelling/meta/meta_info.txt)**: Distills Whisper into a smaller model with large reported latency gains, but requires large-scale training.
- **[distil-large-v2 model card](./references/distil-whisper-distil-large-v2-Hugging-Face/meta/meta_info.txt)**: Provides practical latency/WER summaries and deployment framing for distilled Whisper checkpoints.
- **[LiteASR](./references/LiteASR-Efficient-Automatic-Speech-Recognition-with-Low-Rank-Approximation/meta/meta_info.txt)**: Identifies encoder bottlenecks in Whisper and accelerates them via low-rank approximations and implementation optimizations.
- **[BaldWhisper](./references/BaldWhisper-Faster-Whisper-with-Head-Shearing-and-Layer-Merging/meta/meta_info.txt)**: Compresses Whisper by head shearing and layer merging, providing an alternative structural compression baseline.
- **[Whisper in Medusa's Ear](./references/Whisper-in-Medusas-Ear-Multi-head-Efficient-Decoding-for-Transformer-based-ASR/meta/meta_info.txt)**: Accelerates Whisper decoding via multi-head prediction, demonstrating complementary decoder-side speedups.
- **[SpecASR](../../papers/paper_summaries/SpecASR Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding.md)**: Studies speculative decoding for LLM-based ASR to reduce decoding latency.
- **[Model-free Speculative Decoding for Transformer-based ASR with Token Map Drafting](../../papers/paper_summaries/Model-free Speculative Decoding for Transformer-based ASR with Token Map Drafting.md)**: Proposes a speculative decoding approach tailored for ASR decoding.
- **[Simul-Whisper](./references/Simul-Whisper-Attention-Guided-Streaming-Whisper-with-Truncation-Detection/meta/meta_info.txt)**: Streaming Whisper with truncation detection, showing truncation policies matter for latency.
- **[DeCRED](./references/DeCRED-Decoder-Centric-Regularization-for-Encoder-Decoder-Based-Speech-Recognition/meta/meta_info.txt)**: Decoder-centric regularization for encoder-decoder ASR and discusses early exiting as a special case.
- **[When Less Is More: Diagnosing ASR Predictions in Sardinian via Layer-Wise Decoding](../../papers/paper_summaries/When Less Is More Diagnosing ASR Predictions in Sardinian via Layer-Wise Decoding.md)**: Layer-wise decoding analysis in ASR that highlights intermediate-layer prediction behavior.
- **[Beyond Transcription: Mechanistic Interpretability in ASR](../../papers/paper_summaries/Beyond Transcription Mechanistic Interpretability in ASR.md)**: Mechanistic analyses of ASR models, supporting the idea that intermediate representations contain structure.
- **[BranchyNet](https://arxiv.org/abs/1709.01686)**: Early exiting networks via side branches, a foundational dynamic inference idea.
- **[FastBERT](https://arxiv.org/abs/2004.02178)**: Early exit for BERT using self-distillation and confidence, showing the general viability of dynamic depth.
- **[DeeBERT](https://arxiv.org/abs/2004.04037)**: Early exiting BERT at intermediate layers with entropy-based stopping.
- **[LayerDrop](https://arxiv.org/abs/1909.11556)**: Trains transformers to be robust to layer removal, a training-time alternative to post-hoc truncation.
- **[BERT-of-Theseus](https://arxiv.org/abs/2002.02925)**: Compresses transformers by gradually replacing layers with smaller modules.
- **[Net2Net](https://arxiv.org/abs/1511.05641)**: Function-preserving network transformations that motivate post-hoc structural changes.
- **[Model Stitching](https://arxiv.org/abs/2006.12414)**: Studies connecting networks at intermediate layers, relevant to aligning representations across depths.
- **[Raghu et al. (SVCCA)](https://arxiv.org/abs/1708.05769)**: Canonicalizing and comparing representations, supporting the idea of cross-layer alignment.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| ASR model compression | Reduce parameter count and decoder cost via training | [Distil-Whisper](./references/Distil-Whisper-Robust-Knowledge-Distillation-via-Large-Scale-Pseudo-Labelling/meta/meta_info.txt), [BERT-of-Theseus](https://arxiv.org/abs/2002.02925) | LibriSpeech, FLEURS, long-form sets; WER + latency | Requires training and data; robustness can degrade |
| Encoder compute acceleration | Reduce encoder FLOPs / improve kernels | [LiteASR](./references/LiteASR-Efficient-Automatic-Speech-Recognition-with-Low-Rank-Approximation/meta/meta_info.txt) | Whisper encoder microbenchmarks; end-to-end latency | Engineering-heavy; not a simple drop-in change |
| Decoder-side acceleration | Reduce autoregressive decoding cost | [Whisper-Medusa](./references/Whisper-in-Medusas-Ear-Multi-head-Efficient-Decoding-for-Transformer-based-ASR/meta/meta_info.txt), [SpecASR](../../papers/paper_summaries/SpecASR Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding.md) | LibriSpeech; WER + speedup | Does not address encoder bottleneck regimes |
| Early exit / truncation | Stop computation early (fixed or adaptive) | [BranchyNet](https://arxiv.org/abs/1709.01686), [FastBERT](https://arxiv.org/abs/2004.02178) | GLUE; vision datasets; speed-accuracy tradeoff | Often needs training-time exits; mismatch for frozen downstream heads |
| Representation alignment / lenses | Map intermediate representations into a usable output space | [Tuned Lens](./references/Eliciting-Latent-Predictions-from-Transformers-with-the-Tuned-Lens/meta/meta_info.txt), [DecoderLens](./references/DecoderLens-Layerwise-Interpretation-of-Encoder-Decoder-Transformers/meta/meta_info.txt), [Model Stitching](https://arxiv.org/abs/2006.12414) | Interpretability studies; probe accuracy | Typically not used for deployment speedups |

### Closest Prior Work

- **[LiteASR](./references/LiteASR-Efficient-Automatic-Speech-Recognition-with-Low-Rank-Approximation/meta/meta_info.txt)**: Demonstrates that Whisper encoders can be a dominant latency bottleneck (especially under batching) and accelerates them using low-rank approximations and kernels. Our work differs by asking whether a *pure representation alignment* layer can make naive encoder truncation usable without approximation engineering.
- **[Distil-Whisper](./references/Distil-Whisper-Robust-Knowledge-Distillation-via-Large-Scale-Pseudo-Labelling/meta/meta_info.txt)**: Achieves large speedups by training a much smaller decoder, but requires large-scale pseudo-label training. Our method is post-hoc and only trains a small affine map, targeting deployment settings where retraining is not feasible.
- **[BaldWhisper](./references/BaldWhisper-Faster-Whisper-with-Head-Shearing-and-Layer-Merging/meta/meta_info.txt)**: Compresses Whisper by pruning and merging components. Our proposal does not attempt to redesign the architecture; it tests whether intermediate encoder states can be linearly aligned to enable truncation.
- **[Whisper in Medusa's Ear](./references/Whisper-in-Medusas-Ear-Multi-head-Efficient-Decoding-for-Transformer-based-ASR/meta/meta_info.txt)**: Provides decoder-side speedups via multi-head predictions. Our method is complementary and targets encoder bottlenecks.
- **[Eliciting Latent Predictions (Tuned Lens)](./references/Eliciting-Latent-Predictions-from-Transformers-with-the-Tuned-Lens/meta/meta_info.txt)**: Shows that affine translation of intermediate residual streams can recover accurate predictions in language modeling. We extend the idea to encoder representations in ASR, where the consumer is a frozen decoder cross-attention module.

**Novelty Kill Search Summary:** Searched for exact combinations such as tuned lens + Whisper, affine probe + encoder truncation + ASR, and Whisper + early exit + representation alignment, and checked the local proposal corpus for tuned lens. No close prior work using tuned-lens-style affine alignment to enable truncating Whisper encoders was found as of 2026-02-18.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [LiteASR](./references/LiteASR-Efficient-Automatic-Speech-Recognition-with-Low-Rank-Approximation/meta/meta_info.txt) | Accelerates Whisper encoders via low-rank approximation + kernels | Engineering-heavy; not just truncate | Add a tiny affine translator to make truncation usable | If truncation mismatch is mostly representational, affine alignment is sufficient |
| [Distil-Whisper](./references/Distil-Whisper-Robust-Knowledge-Distillation-via-Large-Scale-Pseudo-Labelling/meta/meta_info.txt) | Trains a smaller decoder and reports large speedups | Requires large-scale training data + compute | Train only ~1.6M parameters for alignment | Much cheaper; can be applied to frozen deployed models |
| [BaldWhisper](./references/BaldWhisper-Faster-Whisper-with-Head-Shearing-and-Layer-Merging/meta/meta_info.txt) | Compresses Whisper via pruning/merging | Structural change; may need tuning | Keep model intact; only truncate encoder + translate | Minimal intervention, easy to implement |
| [Whisper-Medusa](./references/Whisper-in-Medusas-Ear-Multi-head-Efficient-Decoding-for-Transformer-based-ASR/meta/meta_info.txt) | Decoder-side multi-head decoding speedups | Does not help encoder bottleneck | Target encoder depth, not decoder | Complementary; could combine if both bottlenecks matter |
| [Tuned Lens](./references/Eliciting-Latent-Predictions-from-Transformers-with-the-Tuned-Lens/meta/meta_info.txt) | Affine maps from intermediate LM layers to logits | Not a deployment method for seq2seq ASR | Apply affine translation to encoder states consumed by cross-attention | Cross-attention is sensitive to representation distribution; alignment may fix it |

---

## Experiments

### Experimental Setup

**Profiling gate (required before training alignment).** Measure encoder vs decoder wall-clock share for the chosen model and decoding setup on an A100.

- **Serving configuration (fixed for this proposal):** 30s audio chunks, greedy decoding, batch size **B=8** (chosen because LiteASR reports encoder-dominant latency in batched settings).
- If encoder share is <40% under this regime, encoder truncation is unlikely to yield >=1.2x end-to-end speedup; in that case we will treat the proposal as refuted for this deployment regime.

**Baseline Ladder (REQUIRED):** For ASR, there is no direct analog of "prompting" baselines. The closest analog of inference-time scaling is changing the decoding budget (e.g., greedy vs beam search). To keep the verification minimal and decisive, we fix **greedy decoding** for all main conditions and include one analysis-only decoding-budget reference.

**Main conditions (3 total):**

1) Full Whisper (no truncation)
2) Naive encoder truncation at depth L
3) Encoder truncation + tuned-lens-style affine translator (ours)

**Analysis-only reference (not part of the decision rule):** Full Whisper with beam search (higher compute) to contextualize the best achievable WER if extra decoding compute is allowed.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Whisper-large-v2 | ~1.55B | https://huggingface.co/openai/whisper-large-v2 | Open weights; widely used; reported LibriSpeech WER in Whisper-Medusa |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| LibriSpeech train-clean-100 | Train affine translator (representation alignment) | ~100 hours audio | https://www.openslr.org/12 | CC BY 4.0 |

**Resource Estimate**:

- **Compute budget**: 
  - Profiling + layer-depth timing sweep: <1 GPU-hour.
  - Training affine translator for 1 layer depth L on 1k-5k utterances: ~5-10 GPU-hours per seed (encoder forward for h_N and h_L + small backprop through translator). Run 3 seeds -> ~15-30 GPU-hours.
  - Evaluation on LibriSpeech test-clean/test-other (greedy decoding) for 3 methods (full, naive trunc, ours): expected <30 GPU-hours.
  - Total: <60 GPU-hours.
- **GPU memory**: Whisper-large-v2 bf16 inference fits on a single A100 80GB; training translator uses negligible additional parameters.
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| LibriSpeech | English read speech ASR benchmark; standard for Whisper evaluation | Word Error Rate (WER; lower is better) | test-clean, test-other | https://www.openslr.org/12 | https://github.com/huggingface/evaluate (WER) or jiwer + HuggingFace Transformers |

**Speed metrics (deployment-relevant).** Report:

- End-to-end **Real-Time Factor (RTF)** = seconds of processing / seconds of audio (lower is better).
- Encoder-only and decoder-only wall-clock share under the same decoding settings.

We will measure latency under a specified regime (batch size, chunk size, decoding config) and report the exact configuration to avoid cross-paper comparability issues.

### Main Results

#### Results Table

Baseline WER numbers for Whisper-large-v2 on LibriSpeech are copied from Whisper-Medusa Table II (same base model and dataset; see `./references/Whisper-in-Medusas-Ear-Multi-head-Efficient-Decoding-for-Transformer-based-ASR/sections/IV-A Main results.md`, Table II). Speed numbers are **TBD** and will be measured directly because cross-paper speed is not comparable.

| Method | Base Model | Benchmark | test-clean WER | test-other WER | RTF (lower is better) | Source | Notes |
|--------|------------|-----------|----------------|----------------|------------------------|--------|------|
| Full decoding | Whisper-large-v2 | LibriSpeech | 4.00 | 6.19 | **TBD** | Whisper-Medusa (Table II) | Greedy decoding; speed must be re-measured |
| Naive encoder truncation (depth L) | Whisper-large-v2 | LibriSpeech | **TBD** | **TBD** | **TBD** | - | Needs re-run |
| **Ours: truncation + affine translator** | Whisper-large-v2 | LibriSpeech | **TBD** | **TBD** | **TBD** | - | To be verified (3 seeds for translator training) |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Ours (full) | Truncation + learned affine translator | Best WER at the chosen L |
| No translator | Truncation only (naive) | Higher WER, showing translator addresses representation mismatch |
| (Optional) +logit distillation | Add fixed \(\lambda \mathcal{L}_{logit}\) | May further reduce WER; if no gain, representation alignment is sufficient |

### Experimental Rigor

**Variance & Reproducibility:**

- Encoder/decoder profiling and greedy decoding are deterministic.
- Translator training is stochastic; we will train with **3 random seeds** (e.g., `seeds=[0,1,2]`) and report mean +/- std WER.

**Validity & Controls:**

- **Control 1 (selection bias on L):** Choose L using only runtime profiling (not WER) to avoid cherry-picking depth on test performance.
- **Control 2 (speed-accuracy confound):** All three methods use identical decoding configuration (greedy, same max tokens, same chunking), so WER differences are attributable to encoder truncation and translation.
- **Sanity check:** Verify that the full-model run reproduces Whisper-Medusa's reported LibriSpeech WER within expected noise for the evaluation script.

### Analysis (Optional)

- **Representation alignment quality:** Report \(\|g_L(h_L) - h_N\|\) on held-out calibration data to test whether WER improvements correlate with representation-level matching.
- **Failure mode localization:** If WER degrades, analyze whether errors are concentrated in long utterances or acoustically difficult subsets (LibriSpeech test-other) to guide future adapters.

---

## Success Criteria

**Hypothesis** (directional -- what we expect):

- Under a deployment regime where the encoder is a major latency bottleneck (e.g., batched inference), truncating the Whisper encoder at depth L will provide a measurable end-to-end speedup, and the affine translator will recover a substantial fraction of the WER loss from naive truncation.

**Decision Rule** (concrete -- when to stop):

- **Proceed/Continue** if, for the selected depth L, we observe:
  - End-to-end speedup >= **1.2x** vs full model, and
  - \(\mathrm{WER}_{ours} \le \mathrm{WER}_{full} + 1.0\) on test-clean and \(\mathrm{WER}_{ours} \le \mathrm{WER}_{full} + 1.5\) on test-other, and
  - \(\mathrm{WER}_{ours}\) improves over naive truncation by at least **0.3** absolute WER on at least one split without worsening the other.

- **Pivot** if speedup is achieved but WER is outside thresholds; the next step would be to replace the affine translator with a small MLP adapter of similar parameter budget (not part of this proposal).

- **Refute** if either:
  - Profiling shows encoder share <40% under the intended serving regime (so encoder truncation cannot deliver >= **1.2x** end-to-end), or
  - The affine translator yields <0.3 absolute WER improvement over naive truncation at the chosen L (or consistently worsens WER).

---

## Impact Statement

If successful, this method would provide a lightweight, post-hoc way to accelerate Whisper-class ASR in batched serving settings by truncating the encoder while retaining most transcription quality, without requiring retraining or architectural changes. This would directly benefit practitioners serving ASR in real-time or high-throughput settings (call centers, meeting transcription, multimedia indexing) where encoder latency dominates cost.

---

## References

- [Whisper](https://arxiv.org/abs/2212.04356) - Radford et al., 2022
- [Eliciting Latent Predictions from Transformers with the Tuned Lens](./references/Eliciting-Latent-Predictions-from-Transformers-with-the-Tuned-Lens/meta/meta_info.txt)
- [DecoderLens: Layerwise Interpretation of Encoder-Decoder Transformers](./references/DecoderLens-Layerwise-Interpretation-of-Encoder-Decoder-Transformers/meta/meta_info.txt)
- [Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling](./references/Distil-Whisper-Robust-Knowledge-Distillation-via-Large-Scale-Pseudo-Labelling/meta/meta_info.txt)
- [distil-large-v2 model card](./references/distil-whisper-distil-large-v2-Hugging-Face/meta/meta_info.txt)
- [LiteASR: Efficient Automatic Speech Recognition with Low-Rank Approximation](./references/LiteASR-Efficient-Automatic-Speech-Recognition-with-Low-Rank-Approximation/meta/meta_info.txt)
- [BaldWhisper: Faster Whisper with Head Shearing and Layer Merging](./references/BaldWhisper-Faster-Whisper-with-Head-Shearing-and-Layer-Merging/meta/meta_info.txt)
- [Whisper in Medusa's Ear: Multi-head Efficient Decoding for Transformer-based ASR](./references/Whisper-in-Medusas-Ear-Multi-head-Efficient-Decoding-for-Transformer-based-ASR/meta/meta_info.txt)
- [SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding](../../papers/paper_summaries/SpecASR Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding.md)
- [Model-free Speculative Decoding for Transformer-based ASR with Token Map Drafting](../../papers/paper_summaries/Model-free Speculative Decoding for Transformer-based ASR with Token Map Drafting.md)
- [Simul-Whisper: Attention-Guided Streaming Whisper with Truncation Detection](./references/Simul-Whisper-Attention-Guided-Streaming-Whisper-with-Truncation-Detection/meta/meta_info.txt)
- [DeCRED: Decoder-Centric Regularization for Encoder-Decoder Based Speech Recognition](./references/DeCRED-Decoder-Centric-Regularization-for-Encoder-Decoder-Based-Speech-Recognition/meta/meta_info.txt)
- [OpenAI Whisper Benchmark Nvidia Tesla T4 A100 - Oliver Wehrens](./references/OpenAI-Whisper-Benchmark-Nvidia-Tesla-T4-A100---Oliver-Wehrens/meta/meta_info.txt)
- [BranchyNet](https://arxiv.org/abs/1709.01686)
- [FastBERT](https://arxiv.org/abs/2004.02178)
- [DeeBERT](https://arxiv.org/abs/2004.04037)
- [LayerDrop](https://arxiv.org/abs/1909.11556)
- [BERT-of-Theseus](https://arxiv.org/abs/2002.02925)
- [Net2Net](https://arxiv.org/abs/1511.05641)
- [Model Stitching](https://arxiv.org/abs/2006.12414)
- [SVCCA](https://arxiv.org/abs/1708.05769)
