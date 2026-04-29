# untitled

# Silence-Conditional Decoder Head Masking for Whisper Non-Speech Hallucinations (Training-Free)

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Automatic speech recognition (ASR) systems are widely deployed in voice assistants, meeting transcription, call centers, and accessibility tooling. Modern ASR models increasingly use large encoder-decoder Transformers (an audio encoder plus a text decoder) with a strong internal language modeling component (for example, Whisper), which improves average word error rate (WER; the fraction of words that are substituted, deleted, or inserted relative to a reference transcript; lower is better) under distribution shift but also introduces a failure mode: the system can generate fluent text even when the audio contains little or no speech.

This failure mode is not only a quality issue but also a safety and trust issue. **Careless Whisper** reports that, in Spring 2023, a non-trivial fraction of Whisper transcriptions contained entirely hallucinated phrases, and that a substantial portion of those hallucinations contained explicit harms (violence, false authority, or fabricated personal information) and disproportionately affected speakers with aphasia, who have longer non-vocal segments in their speech patterns.

Recent work suggests that Whisper's non-speech hallucination may be mechanistically localized. **Calm-Whisper** performs head-wise ablations in Whisper-large-v3 and finds that a small subset of decoder self-attention heads account for most hallucinations on a non-speech benchmark (UrbanSound8K, an environmental sound dataset that contains no speech). They show that masking heads {#1, #6, #11} reduces hallucination rate dramatically, but degrades WER on speech; and that selectively fine-tuning only those heads on non-speech audio reduces hallucinations while preserving WER.

### The Problem

Calm-Whisper indicates an attractive direction for mitigation: target a small part of the decoder that causes hallucination. However, their best-performing intervention (selective head fine-tuning) still requires training infrastructure and a non-speech dataset, which may be undesirable for many deployments.

A simple alternative is to always mask the hallucinatory heads at inference time, but Calm-Whisper reports this harms speech recognition quality (WER rises noticeably on LibriSpeech). The open question is whether we can obtain a better hallucination-vs-WER tradeoff without training.

Concretely:

- **[Calm-Whisper](./references/Calm-Whisper-Reduce-Whisper-Hallucination-On-Non-Speech-By-Calming-Crazy-Heads-Down/meta/meta_info.txt)** (decoder head analysis + selective head fine-tuning): shows that always masking decoder heads {1,6,11} reduces UrbanSound8K hallucination rate to 24.10% but increases LibriSpeech WER from 2.12% to 3.57% (test-clean) and 4.07% to 5.98% (test-other) (Section 3.2).
- **[Whisper](./references/Robust-Speech-Recognition-via-Large-Scale-Weak-Supervision/meta/meta_info.txt)** (large-scale weak supervision): includes decoding-time heuristics such as `no_speech_threshold`, `logprob_threshold`, and `compression_ratio_threshold` to skip low-confidence segments, but these do not modify the model's internal computation.

The practitioner need is a mitigation that is:
1) training-free,
2) easy to integrate into existing Whisper deployments, and
3) evaluated with fully automated metrics.

### Key Insight and Hypothesis

**Key insight:** If a small set of decoder heads are primarily responsible for hallucinating on non-speech audio, then it is likely that those heads are harmful mainly under a specific internal regime (low acoustic evidence / high language-model prior). Whisper already computes a per-segment signal related to this regime: the probability of the special no-speech token (informally, `p_no_speech`).

**Hypothesis:** Masking the hallucinatory decoder heads only when `p_no_speech` is high will preserve most of the hallucination reduction of always-masking, while recovering most of the WER loss on speech. This could fail if `p_no_speech` does not fire on hallucination cases (so the method rarely activates), or if it fires frequently on real speech (false positives), in which case WER will approach the always-mask baseline.

---

## Proposed Approach

### Overview

We propose **Silence-Conditional Head Masking (SCHM)**, a training-free inference-time modification for Whisper:

1. Compute `p_no_speech` for the audio clip using the unmodified model.
2. If `p_no_speech >= tau`, decode with a decoder self-attention head mask that disables the heads identified by Calm-Whisper as hallucinatory ({1,6,11}).
3. Otherwise, decode normally.

The method aims to improve the hallucination-vs-WER tradeoff relative to always masking.

### Method Details

**Base model.** `openai/whisper-large-v3`.

**Head masking interface.** HuggingFace Transformers supports a `decoder_head_mask` argument for Whisper models. We create a binary mask matrix `M` of shape `[n_decoder_layers, n_heads]` where `M[l,h]=0` for `h in {1,6,11}` and `M[l,h]=1` otherwise.

**Trigger computation.** For each input audio clip, compute `p_no_speech` from the unmasked model. Implementation options:

- Option 1 (preferred): run a 1-step forward pass at the start-of-transcript position and read the probability mass on the no-speech token.
- Option 2: run a short greedy decode for 1 token and reuse the computed logits.

**Conditional decoding.** If `p_no_speech >= tau`, run full greedy decoding with `decoder_head_mask=M`. Otherwise, run standard greedy decoding.

**Threshold choice.** Default `tau=0.6` to match Whisper's commonly used `no_speech_threshold` heuristic. We will also report results for `tau in {0.5, 0.6, 0.7}` as a small robustness table.

### Key Innovations

- **Training-free model-internal mitigation** for Whisper non-speech hallucination using decoder head masking, avoiding fine-tuning.
- **Mechanism-based conditional control**: only apply the head mask in the internal regime that plausibly corresponds to non-speech.
- **A decisive go/no-go diagnostic** that tests whether the trigger fires on hallucination cases before spending compute on full evaluation.

---

## Related Work

### Field Overview

Whisper hallucination has been studied from multiple angles: (i) harm analysis and correlations with non-vocal segments, (ii) evaluation methodology and hallucination-specific metrics beyond WER, (iii) training-time mitigations (distillation, robustness training), and (iv) inference-time mitigations (VAD preprocessing, decoding heuristics).

Separately, transformer interpretability and pruning work shows that specific attention heads can implement distinct behaviors, and that head-level ablations can identify small functional subsets. Calm-Whisper is an example of such an analysis applied to ASR hallucination.

### Related Papers

- **[Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](./references/Robust-Speech-Recognition-via-Large-Scale-Weak-Supervision/meta/meta_info.txt)**: Introduces Whisper and its decoding-time heuristics including a no-speech token.
- **[Careless Whisper: Speech-to-Text Hallucination Harms](./references/Careless-Whisper-Speech-to-Text-Hallucination-Harms/meta/meta_info.txt)**: Documents hallucination harms and links them to non-vocal duration and accessibility disparities.
- **[Calm-Whisper](./references/Calm-Whisper-Reduce-Whisper-Hallucination-On-Non-Speech-By-Calming-Crazy-Heads-Down/meta/meta_info.txt)**: Identifies hallucinatory decoder heads via head-wise masking and reduces hallucinations via selective head fine-tuning.
- **[Lost in Transcription, Found in Distribution Shift](./references/Lost-in-Transcription-Found-in-Distribution-Shift-Demystifying-Hallucination-in-Speech-Foundation-Models/meta/meta_info.txt)**: Introduces hallucination-focused metrics (HER) and studies how distribution shift affects hallucination.
- **[Listen Like a Teacher](./references/Listen-Like-a-Teacher-Mitigating-Whisper-Hallucinations-using-Adaptive-Layer-Attention-and-Knowledge-Distillation/meta/meta_info.txt)**: Reduces Whisper hallucinations under noise via adaptive layer attention and knowledge distillation (training-heavy).
- **[Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio](https://arxiv.org/abs/2501.11378)**: Studies Whisper hallucinations on non-speech and proposes filtering-style mitigations.
- **[WhisperX](https://arxiv.org/abs/2303.00747)**: Improves Whisper transcription with alignment and VAD-style preprocessing (pipeline-level mitigation).
- **[Distil-Whisper](https://arxiv.org/abs/2311.00416)**: Distills Whisper models and discusses filtering/selection effects that can reduce hallucinations.
- **[Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574)**: Shows that specific attention heads can drive specific behaviors; cited by Calm-Whisper as inspiration.
- **[Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)**: Classic analysis showing many attention heads are redundant and can be pruned.
- **[Conformer](https://arxiv.org/abs/2005.08100)**: A strong non-autoregressive ASR architecture often used as a robustness baseline.
- **[Connectionist Temporal Classification (CTC)](https://dl.acm.org/doi/10.1145/1143844.1143891)**: A non-autoregressive objective; CTC models are often more conservative on non-speech.
- **[OWSM-CTC](https://arxiv.org/abs/2402.12654)**: Reports robustness advantages of encoder-only / CTC-style speech foundation models and discusses hallucination-like failures.
- **[InterSpeech 2024/2025 work on Whisper hallucination mitigation](https://arxiv.org/abs/2511.14219)**: Training-time mitigation under noisy conditions.
- **[Outlier Reduction with Gated Attention for Improved Post-training Quantization in Large Seq2Seq Speech Foundation Models](https://arxiv.org/abs/2406.11022)**: Uses gating inside attention blocks (motivates head-level control, though for quantization).
- **[UrbanSound8K Dataset Paper](https://dl.acm.org/doi/10.1145/2647868.2655045)**: Defines the UrbanSound8K dataset used as a non-speech benchmark.
- **[SpecAugment](https://arxiv.org/abs/1904.08779)**: Data augmentation for ASR robustness (background context).
- **[Wav2Vec 2.0](https://arxiv.org/abs/2006.11477)**: Representative self-supervised speech pretraining baseline.
- **[HuBERT](https://arxiv.org/abs/2106.07447)**: Representative self-supervised speech pretraining baseline.
- **[Fast-Conformer / streaming ASR work](https://arxiv.org/abs/2006.03273)**: Context on deployment constraints that motivate inference-time mitigations.

(Additional Whisper-related works can be added if needed; the core closest-prior comparison is against Calm-Whisper and Whisper decoding heuristics.)

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Pipeline-level filtering | VAD + alignment + remove silence segments | WhisperX, practitioner heuristics | WER on ASR sets; ad-hoc hallucination checks | VAD errors can delete speech; adds extra components |
| Decode-heuristic skipping | Use `no_speech_threshold`, logprob, repetition/compression filters | Whisper | WER; skip rate | Heuristic, does not change internal LM behavior |
| Training-time mitigation | Fine-tune / distill to suppress hallucination | Calm-Whisper, Listen Like a Teacher, Distil-Whisper | UrbanSound8K hallucination rate; LibriSpeech WER | Requires training data/compute; may not be available in deployment |
| Head-level interpretability | Identify small functional subsets (heads) | Calm-Whisper, retrieval-head work | Mechanistic ablations | Often diagnostic; not always turned into an inference policy |
| **This work** | Conditional inference-time masking of problematic heads | (ours) | UrbanSound8K hallucination + LibriSpeech WER | Depends on trigger quality (`p_no_speech`) |

### Closest Prior Work

**Calm-Whisper** is the closest prior work. It (i) identifies a small set of hallucinatory decoder heads via head-wise masking, and (ii) reduces hallucination by selectively fine-tuning those heads on non-speech data. It reports that always masking heads {1,6,11} reduces hallucination rate but increases WER.

Our proposal is different in two ways:
1. It is **training-free** (no fine-tuning), and
2. It uses a **conditional policy**: only apply head masking when the model's own no-speech probability is high.

**Whisper decoding heuristics** (no_speech_threshold/logprob_threshold/compression_ratio_threshold) are also close, but they act only at the level of skipping segments rather than modifying decoder computation.

**Novelty Kill Search Summary:** Searched for combinations of "Whisper hallucination head masking", "conditional head masking Whisper", "no_speech_threshold head masking", and checked local finalized proposals for "Calm-Whisper", "UrbanSound8K", and "no_speech_threshold". No prior work proposing no-speech-triggered decoder head masking for hallucination mitigation was found as of 2026-02-23.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Calm-Whisper | Finds hallucinatory heads; fine-tunes them on non-speech | Requires training; masking always hurts WER | Use conditional masking with no training | Avoid WER loss by activating only on likely non-speech |
| Whisper heuristics | Skip segments based on no-speech prob and logprob | Heuristic; does not reduce internal hallucination tendency | Modify decoder computation when non-speech is likely | Suppress internal LM behavior that produces hallucination text |
| VAD preprocessing (WhisperX) | Remove silence via external VAD | VAD mistakes can delete speech; pipeline complexity | Use Whisper-internal signal (no-speech prob) | No extra model and directly targets hallucination mechanism |

---

## Experiments

### Experimental Setup

**Base model:** `openai/whisper-large-v3` (HuggingFace Transformers).

**Decoding:** greedy decoding (num_beams=1, temperature=0), no timestamps.

**Main conditions (3):**

- **A: Default Whisper**: unmodified decoding.
- **B: Always-mask**: decode with decoder self-attention heads {1,6,11} masked in every decoder layer.
- **C: SCHM (ours)**: compute `p_no_speech` using the unmasked model; if `p_no_speech >= tau` then decode with the head mask, else decode normally.

**Phase-1 go/no-go diagnostic (cheap):** Before running the full evaluation, compute the distribution of `p_no_speech` on UrbanSound8K and report the fraction of clips where A hallucinates but `p_no_speech > 0.5`. If this fraction is <30%, SCHM is unlikely to matter and the proposal should be refuted early.

**Analysis-only baselines (not part of the 3 main conditions):**
- Skip output if `p_no_speech >= tau` (no head mask) to test whether the trigger alone explains improvements.
- Vary tau in {0.5, 0.6, 0.7} for robustness.

**Training Data (if applicable):**

- No training data needed - inference only.

**Resource Estimate**:

- Total audio duration: UrbanSound8K ~10h, LibriSpeech test-clean+test-other ~6h.
- Whisper-large runs about ~10x faster than real-time on an A100 in public benchmarks, so one full pass is on the order of a few GPU-hours; 3 conditions plus overhead should be well under **100 A100 GPU-hours**.
- GPU memory: Whisper-large-v3 fits on 1x A100 80GB.
- API usage: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| UrbanSound8K | Environmental sound clips (no speech) | Hallucination rate = fraction of clips with non-empty transcript | full | https://huggingface.co/datasets/danavery/urbansound8K | custom script using Whisper decode + string length |
| LibriSpeech | English read-speech ASR benchmark | WER (lower is better) on test-clean and test-other | test-clean/test-other | https://huggingface.co/datasets/librispeech_asr | `jiwer` WER on references |

### Main Results

#### Comparability Rules (CRITICAL)

All A/B/C rows must use:
- same Whisper checkpoint
- same decoding settings
- same preprocessing (resample to 16kHz)

#### Results Table

Published baselines below are from Calm-Whisper (Section 3.2) and are provided for context; verification will re-run A/B under the exact codepath used for C.

| Method | Base Model | UrbanSound8K hallucination rate | LibriSpeech WER (clean) | LibriSpeech WER (other) | Source | Notes |
|---|---|---:|---:|---:|---|---|
| A: Default Whisper | Whisper-large-v3 | 99.97% | 2.12% | 4.07% | Calm-Whisper Sec 3.2 | 1 run; decoding config may differ |
| B: Always-mask heads {1,6,11} | Whisper-large-v3 | 24.10% | 3.57% | 5.98% | Calm-Whisper Sec 3.2 | 1 run; decoding config may differ |
| **C: SCHM (ours)** | Whisper-large-v3 | **TBD** | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

No additional ablations are required beyond the tau robustness table reported under analysis-only baselines.

### Experimental Rigor

- **Determinism / seeds**: Greedy decoding should be deterministic; run once. If any nondeterminism is observed, run 3 seeds and report mean+/-std.
- **Sanity check 1**: reproduce Calm-Whisper's always-mask directionality (hallucination drops, WER rises) before interpreting SCHM.
- **Sanity check 2**: report the Phase-1 diagnostic on `p_no_speech` so failures can be attributed to trigger quality rather than head masking implementation.

---

## Success Criteria

**Hypothesis:** SCHM achieves a strictly better hallucination-vs-WER tradeoff than always masking by activating head masking mainly on non-speech clips.

**Decision Rule:**

- **Refute early (trigger failure)** if, on UrbanSound8K, fewer than 30% of baseline-hallucinating clips have `p_no_speech > 0.5`.
- **Proceed** if SCHM Pareto-dominates always-mask:
  - UrbanSound8K hallucination(C) <= hallucination(B) + 2.0 percentage points, AND
  - LibriSpeech WER(C) <= WER(B) - 0.3 absolute on both test-clean and test-other.
- **Refute** if SCHM does not improve WER over always-mask while keeping hallucination close.

---

## Impact Statement

If successful, SCHM provides a training-free patch that reduces non-speech hallucinations in Whisper deployments with minimal impact on speech recognition quality. This can improve trust and safety for transcription systems used in high-stakes or accessibility-sensitive settings where non-speech segments and long pauses are common.

---

## References

- [Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down](./references/Calm-Whisper-Reduce-Whisper-Hallucination-On-Non-Speech-By-Calming-Crazy-Heads-Down/meta/meta_info.txt) - Wang et al., 2025
- [Robust Speech Recognition via Large-Scale Weak Supervision](./references/Robust-Speech-Recognition-via-Large-Scale-Weak-Supervision/meta/meta_info.txt) - Radford et al., 2022
- [Careless Whisper: Speech-to-Text Hallucination Harms](./references/Careless-Whisper-Speech-to-Text-Hallucination-Harms/meta/meta_info.txt) - Koenecke et al., 2024
- [Lost in Transcription, Found in Distribution Shift: Demystifying Hallucination in Speech Foundation Models](./references/Lost-in-Transcription-Found-in-Distribution-Shift-Demystifying-Hallucination-in-Speech-Foundation-Models/meta/meta_info.txt) - Atwany et al., 2025
- [Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation](./references/Listen-Like-a-Teacher-Mitigating-Whisper-Hallucinations-using-Adaptive-Layer-Attention-and-Knowledge-Distillation/meta/meta_info.txt) - Tripathi et al., 2025
- [Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio](https://arxiv.org/abs/2501.11378) - Baranski et al., 2025
- [WhisperX](https://arxiv.org/abs/2303.00747) - Bain et al., 2023
- [Distil-Whisper](https://arxiv.org/abs/2311.00416) - Gandhi et al., 2023
- [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574) - Wu et al., 2024
- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) - Michel et al., 2019
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) - Gulati et al., 2020
- [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data](https://dl.acm.org/doi/10.1145/1143844.1143891) - Graves et al., 2006
- [OWSM-CTC: An Open Encoder-Only Speech Foundation Model](https://arxiv.org/abs/2402.12654) - (authors), 2024
- [Outlier Reduction with Gated Attention for Improved Post-training Quantization](https://arxiv.org/abs/2406.11022) - (authors), 2024
- [A Dataset and Taxonomy for Urban Sound Research (UrbanSound8K)](https://dl.acm.org/doi/10.1145/2647868.2655045) - Salamon et al., 2014
- [SpecAugment](https://arxiv.org/abs/1904.08779) - Park et al., 2019
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) - Baevski et al., 2020
- [HuBERT](https://arxiv.org/abs/2106.07447) - Hsu et al., 2021
- [Fast Conformer](https://arxiv.org/abs/2006.03273) - Gulati et al., 2020
