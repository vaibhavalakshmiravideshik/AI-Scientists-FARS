# untitled

# Text-Length-Coupled Audio Stopping to Reduce Long-Form Speech Truncation in Omni-LLMs

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

End-to-end “omni” language models that both *understand* speech and *generate* speech (instead of using a separate automatic speech recognition (ASR) → LLM → text-to-speech (TTS) pipeline) are increasingly used as voice assistants and interactive agents. Examples include Qwen2.5-Omni, Qwen3-Omni, VITA, LLaMA-Omni(2), Moshi, and OpenOmni. These systems must generate spoken responses that can be long (tutoring explanations, step-by-step instructions, narration, or multi-turn conversations).

A common design in open models is to represent speech as discrete “speech unit” tokens emitted at a high frame rate. For example, Ex-Omni reports approximately 12 speech tokens per second in its speech-unit stream [Ex-Omni A.4](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/sections/A.4%20Supplementary%20of%20Experiments%20Results%20and%20Analysis.md). High-rate token streams create long output sequences: a 60-second answer can require hundreds to thousands of audio tokens. This makes long-form speech generation sensitive to decode-time stopping rules such as an **end-of-sequence (EOS)** token and to configured maximum output lengths.

Ex-Omni evaluates **speech-text consistency** on VoiceBench’s CommonEval subset (200 real spoken information-seeking questions from CommonVoice) by transcribing generated speech with Whisper-large-v3 and computing **word error rate (WER; lower is better)** between the ASR transcript and the model’s generated text [Ex-Omni A.4](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/sections/A.4%20Supplementary%20of%20Experiments%20Results%20and%20Analysis.md). They observe that inconsistency grows with output duration and explicitly report a failure mode at 40–60 seconds where “textual responses continue while the generated speech is prematurely truncated.” This mismatch is practically harmful: users receive incomplete spoken content even when the model produces a full text answer.

### The Problem

We focus on **speech truncation / speech-text mismatch** in end-to-end omni models: the model generates a complete text response, but the synthesized speech stops early.

There are (at least) two distinct, easily-confused causes:

1. **Hard cap truncation**: speech stops because decoding hits a configured maximum audio-token/duration limit.
2. **Premature speech EOS**: speech stops because the speech generator emits an EOS token early, despite remaining headroom.

A trivial mitigation for (1) is “increase the max tokens”. However, if (2) is common, simply raising limits does not fix the mismatch: the model still chooses to stop speaking early. Current reports (e.g., Ex-Omni’s analysis) attribute the phenomenon to both limited autoregressive capacity of a small speech generator and token budget constraints [Ex-Omni A.4](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/sections/A.4%20Supplementary%20of%20Experiments%20Results%20and%20Analysis.md), but do not isolate which cause dominates on open, runnable models.

In practice, developers need a **training-free, decode-time fix** that can be applied to existing open omni models, and a simple automated evaluation that distinguishes “we forced longer audio” from “we actually improved alignment between spoken and textual content.”

### Key Insight and Hypothesis

**Key insight:** In multi-stream generation (text + speech), the two streams often use **independent stopping rules**. Even when the text stream continues, the speech stream can emit EOS early because its learned length prior is miscalibrated for long outputs.

**Hypothesis:** On an open end-to-end omni model (Qwen2.5-Omni), a non-trivial fraction of long-form speech-text failures are caused by **premature speech EOS even when the audio-token cap is not binding**. If so, a simple decode-time coordination rule—**prevent speech EOS until the generated speech length is consistent with the generated text length**—should reduce truncation and improve speech-text consistency without retraining.

This hypothesis could be wrong for two straightforward reasons:
- Most truncations are actually hard-cap hits (then raising the cap solves it, and EOS suppression is unnecessary).
- Preventing early EOS could force the model into degenerate filler/repetition, improving “coverage” while not improving (or worsening) true content consistency.

Our experiment includes controls to rule out both.

---

## Proposed Approach

### Overview

We propose **Text-Length-Coupled Audio Stopping (TLC-AS)**: a training-free decoding modification that delays the **speech EOS** token until the speech stream has produced at least a minimum number of audio tokens consistent with the (simultaneously generated) text.

We evaluate TLC-AS with a 3-condition, inference-only experiment on Qwen2.5-Omni:
- **C0 (default)**: model’s default generation.
- **C1 (cap-only)**: increase the audio-token budget so the cap should not bind for ≤60s outputs.
- **C2 (TLC-AS)**: same as C1, plus EOS suppression until a text-conditioned minimum audio length is reached.

The key decision is whether **C2 improves over C1** (not just over C0). If C1 already fixes the issue, TLC-AS is unnecessary.

### Method Details

**Inputs/outputs.** For each audio prompt, the model generates:
- a text response (used as the “reference text”), and
- a speech waveform.

**Metric backbone (same as Ex-Omni’s analysis).** We transcribe the generated speech with Whisper-large-v3 and compute **word error rate (WER; lower is better)** between the ASR transcript and the model’s generated text [Ex-Omni A.4](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/sections/A.4%20Supplementary%20of%20Experiments%20Results%20and%20Analysis.md).

**Text-conditioned minimum speech length.** Let:
- `w(text_so_far)` = number of whitespace-separated words in the text generated so far,
- `wps` = estimated words-per-second of the model’s speaking rate,
- `r` = speech-token frame rate in tokens/sec for the model’s speech codec.

We define a *time* floor and convert to a *token* floor:

- `min_seconds(text_so_far) = w(text_so_far) / wps`
- `min_audio_tokens(text_so_far) = ceil(min_seconds(text_so_far) * r)`

**wps calibration.** To avoid hand-tuning, estimate `wps` from C1 outputs on a small calibration subset (e.g., 20 prompts): compute `(ASR_word_count / audio_duration_seconds)` from Whisper transcripts and take the median as `wps`. Fix this value for all runs (C0/C1/C2).

**EOS suppression rule (C2).** During speech decoding, if the top-1 candidate is the speech EOS token and `num_audio_tokens < min_audio_tokens(text_so_far)`, set the EOS logit to `-∞` (equivalently, mask it) and continue decoding.

Implementation note: In practice this is a small change in the speech decoder’s generation loop (e.g., a custom logits processor or stopping criteria in the Talker/speech-token generation step). It does not require changing model weights or training.

**Degeneracy guardrail.** TLC-AS can be harmful if it forces repetitive filler. We therefore also track repetition/diversity on the ASR transcript, and treat severe repetition as a failure mode (see Success Criteria).

### Key Innovations

- **A minimal, training-free cross-stream stop coordination rule** for end-to-end speech LMs: couple speech stopping to text length, rather than letting the speech stream stop independently.
- **A controlled evaluation protocol** that distinguishes: (i) fixing hard-cap truncation by raising limits (C1) vs (ii) fixing premature EOS (C2).
- **An automated failure-mode audit** (cap-hit vs early EOS) that can be applied to other omni models.

---

## Related Work

### Field Overview

End-to-end speech generation in “omni” models typically relies on discrete speech token streams (neural codec tokens or learned unit representations) and a speech decoder that autoregressively emits these tokens conditioned on text and multimodal context. This design enables streaming and low latency, but it also creates a long-sequence generation problem because audio tokens arrive at much higher rates than text tokens.

Long-horizon speech generation has been addressed primarily through architectural changes (e.g., chunk-based parallel decoding in MGM-Omni) or improved streaming synthesis modules (e.g., chunk-aware flow matching in CosyVoice 2). In contrast, the specific **decode-time mismatch between a model’s own generated text and its generated speech** has received less targeted attention, despite explicit observations of truncation in recent omni models (Ex-Omni).

### Related Papers

- **[Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/meta/meta_info.txt)**: Introduces speech+3D-face generation and reports speech-text inconsistency and truncation at 40–60s on CommonEval.
- **[Qwen3-Omni Technical Report](./references/Qwen3-Omni-Technical-Report/meta/meta_info.txt)**: Describes a Thinker–Talker omni model with RVQ (residual vector quantization) speech tokens and streaming speech synthesis.
- **[Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)**: Early open end-to-end omni model with Thinker–Talker design and streaming speech generation.
- **[VoiceBench: Benchmarking LLM-Based Voice Assistants](https://arxiv.org/abs/2410.17196)**: Provides the VoiceBench benchmark including the CommonEval subset used in Ex-Omni’s speech-text analysis.
- **[LLaMA-Omni: Seamless Speech Interaction with Large Language Models](https://arxiv.org/abs/2409.06666)**: End-to-end speech interaction with simultaneous text+speech generation and streaming design.
- **[LLaMA-Omni 2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis](https://arxiv.org/abs/2505.02625)**: Improves speech generation quality/latency tradeoffs and supports multiple model sizes.
- **[OpenOmni: Advancing Open-Source Omnimodal LLMs with Progressive Multimodal Alignment](https://arxiv.org/abs/2501.04561)**: Progressive alignment for omni models and real-time speech generation.
- **[VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](https://arxiv.org/abs/2501.01957)**: End-to-end omni model with a learned codec and AR/NAR speech decoding.
- **[Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)**: Speech-to-speech dialogue with discrete codec tokens and full-duplex interaction.
- **[OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](https://arxiv.org/abs/2410.17799)**: Full-duplex spoken dialogue model trained with interleaved speech/text streams; highlights training and decoding trade-offs between early stopping and speech-text alignment.
- **[MGM-Omni: Scaling Omni LLMs to Personalized Long-Horizon Speech](https://arxiv.org/abs/2509.25131)**: Focuses on long-horizon speech generation and introduces chunk-based parallel decoding to address the text–speech token-rate mismatch.
- **[SpeechSSM: Long-Form Speech Generation with Spoken Language Models](https://arxiv.org/abs/2412.18603)**: Uses hybrid state-space models to generate coherent multi-minute speech with bounded memory/compute per step, showing an architectural path to long-form stability.
- **[URO-Bench: Towards Comprehensive Evaluation for End-to-End Spoken Dialogue Models](https://arxiv.org/abs/2502.17810)**: A comprehensive speech-to-speech benchmark that complements VoiceBench with multilingual, multi-round, and paralinguistic evaluations.
- **[SpeechGPT](https://arxiv.org/abs/2305.11000)**: Early speech-enabled LLM using discrete speech units and cross-modal instruction tuning.
- **[SpeechGPT-Gen](https://arxiv.org/abs/2401.13527)**: Improves speech generation efficiency via chain-of-information generation (semantic then acoustic).
- **[SpeechTokenizer](https://arxiv.org/abs/2308.16692)**: Unified speech tokenization (semantic+acoustic) for speech language models.
- **[GLM-4-Voice](https://arxiv.org/abs/2412.02612)**: Open speech model with discrete tokenization and a streaming decoder.
- **[CosyVoice 2](https://arxiv.org/abs/2412.10117)**: Streaming TTS with chunk-aware causal flow matching and strong content consistency.
- **[Seed-TTS](https://arxiv.org/abs/2406.02430)**: Large-scale high-quality speech generation with strong WER and speaker similarity.
- **[VALL-E](https://arxiv.org/abs/2301.02111)**: Neural codec language modeling for zero-shot TTS.
- **[VALL-E 2](https://arxiv.org/abs/2406.05370)**: Improves stability via repetition-aware sampling and grouped code modeling.
- **[AudioLM](https://arxiv.org/abs/2209.03143)**: Hierarchical audio token modeling (semantic/coarse/fine) for coherent long audio generation.
- **[EnCodec](https://arxiv.org/abs/2210.13438)**: High-fidelity neural audio codec widely used for discrete audio tokens.
- **[SoundStream](https://arxiv.org/abs/2107.03312)**: Early end-to-end neural audio codec with residual vector quantization.
- **[BigVGAN](https://arxiv.org/abs/2206.04658)**: Neural vocoder with strong fidelity and robustness.
- **[SeamlessM4T](https://arxiv.org/abs/2308.11596)**: Multimodal translation including speech output; highlights streaming and long-sequence speech concerns.
- **[UniTalker](https://arxiv.org/abs/2408.00762)**: Large-scale audio-driven 3D facial animation and the A2F-Bench benchmark (relevant to Ex-Omni’s facial-motion side).
- **[Audio2Face-3D](https://arxiv.org/abs/2508.16401)**: Audio-driven facial animation used as a teacher/reference model in Ex-Omni.
- **[FaceFormer](https://arxiv.org/abs/2112.05329)**: Transformer model for audio-driven 3D facial animation.
- **[CodeTalker](https://arxiv.org/abs/2301.02379)**: Discrete motion prior for speech-driven 3D facial animation.
- **[Shanks: Simultaneous Hearing and Thinking for Spoken Language Models](https://arxiv.org/abs/2510.06917)**: Streaming spoken LMs, illustrating decode-time constraints in real-time speech interaction.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| End-to-end omni LMs (speech in/out) | Unified model that maps speech input to text+speech output, often with Thinker–Talker separation | Qwen2.5-Omni, Qwen3-Omni, VITA-1.5, LLaMA-Omni(2), OpenOmni, Moshi | VoiceBench, OmniBench-style suites | Multi-stream length mismatch; long speech can hit caps or stop early |
| Long-horizon speech generation | Architectural changes to handle long speech sequences and token-rate mismatch | MGM-Omni, Moshi | Long-TTS-Eval (MGM-Omni), dialogue evals | Often requires training; may not address decode-time EOS mismatch |
| Discrete speech tokenization / codecs | Represent speech as discrete tokens for LM-style modeling | EnCodec, SoundStream, SpeechTokenizer, Mimi (Moshi) | Codec reconstruction, ASR/WER, MOS | High token rates cause long sequences; decoding stability issues |
| Speech-text consistency evaluation | Compare generated speech content to intended text content | Ex-Omni (Table 9), VoiceBench | Whisper-ASR + WER/CER; human evals | WER can miss semantic drift if ASR is wrong; needs guardrails |
| Audio-driven facial animation | Generate facial motion from speech/audio; relevant as Ex-Omni couples speech+face | UniTalker, Audio2Face-3D, FaceFormer, CodeTalker | A2F-Bench, LVE, user studies | Facial motion is non-unique; evaluation is hard |

### Closest Prior Work

- **Ex-Omni** [Ex-Omni meta](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/meta/meta_info.txt): Provides the clearest published evidence of long-duration speech-text mismatch and attributes it partly to token budget constraints; it does not propose a decode-time fix or isolate premature EOS vs cap.
- **Qwen2.5-Omni** (technical report) [arXiv:2503.20215](https://arxiv.org/abs/2503.20215): A runnable open omni baseline with streaming speech generation, suitable for testing decode-time interventions; it does not discuss speech-text mismatch as a first-class evaluation target.
- **MGM-Omni** [arXiv:2509.25131](https://arxiv.org/abs/2509.25131): Targets long-horizon speech with chunk-based parallel decoding; this is mainly an architectural/training solution rather than a simple inference-time retrofit for existing checkpoints.
- **SpeechSSM** [arXiv:2412.18603](https://arxiv.org/abs/2412.18603): Demonstrates coherent multi-minute speech generation via hybrid attention/state-space modeling; relevant as evidence that architectural changes can mitigate long-form degeneration.
- **OmniFlatten** [arXiv:2410.17799](https://arxiv.org/abs/2410.17799): A speech-text dialogue model trained on interleaved streams; relevant as an alternative that changes training to manage stopping/latency trade-offs rather than applying a decoding patch.
- **LLaMA-Omni2** [arXiv:2505.02625](https://arxiv.org/abs/2505.02625): Emphasizes streaming synthesis and quality/latency trade-offs; does not isolate the specific truncation where text continues but speech stops.
- **CosyVoice 2** [arXiv:2412.10117](https://arxiv.org/abs/2412.10117): Improves streaming TTS quality and stability; our focus is not building a new TTS model but fixing mismatch inside an end-to-end omni LM.
- **URO-Bench** [arXiv:2502.17810](https://arxiv.org/abs/2502.17810): Provides broader speech-to-speech evaluation beyond VoiceBench (multilingual, multi-round, paralinguistics); useful for future generalization checks.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Ex-Omni | Reports speech-text inconsistency and truncation at long durations | No targeted fix; unclear if premature EOS vs cap | Add decode-time stop coordination for speech EOS | If premature EOS is common, coordination reduces truncation without retraining |
| Qwen2.5-Omni | Open omni model with streaming speech generation | No explicit consistency/stopping alignment mechanism | Add TLC-AS as a small decoding patch | Minimal engineering change; can be deployed immediately |
| MGM-Omni | Architectural/training changes for 10+ minute speech | Not a patch for existing models; more costly | Training-free decoding only | Lower barrier; can validate quickly on existing checkpoints |
| SpeechSSM | Hybrid attention/state-space long-form speech generation (minutes) | Different model family (not an omni assistant); not a retrofit | Target the truncation failure with a decode-time rule | If mismatch is mostly EOS-driven, a patch is cheaper than retraining |
| OmniFlatten | Interleaved speech/text training for full-duplex dialogue | Requires (re)training and proprietary-scale data | Inference-only stopping coordination | Works on existing checkpoints without new training |
| LLaMA-Omni2 | Streaming speech decoder improvements | Does not target speech EOS mismatch explicitly | Provide a general decode-time rule | Applies across model families using discrete speech tokens |
| CosyVoice 2 | Strong streaming TTS | Not end-to-end omni; doesn’t solve text-vs-speech mismatch in omni LMs | Measure and fix mismatch directly | Targets the end-to-end assistant failure mode |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen/Qwen2.5-Omni-7B | 7B | https://huggingface.co/Qwen/Qwen2.5-Omni-7B | Primary runnable end-to-end omni model for verification |
| openai/whisper-large-v3 | 1.55B | https://huggingface.co/openai/whisper-large-v3 | ASR used to compute WER between generated speech and text |

**Training Data (if applicable):**

- No training data needed - inference only.

**Resource Estimate**:
- **Compute budget**: ~50–200 GPU-hours (dominated by generating up to 60s speech for 200 prompts × 3 conditions, plus ASR). This should fit within the 768 GPU-hour cap.
- **GPU memory**: Qwen2.5-Omni-7B is expected to fit on a single A100 80GB for audio-only generation; Whisper-large-v3 fits on a single A100.
- **API usage**: None required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| VoiceBench / CommonEval | 200 real spoken information-seeking questions (audio prompts) used for voice-assistant evaluation | WER (ASR vs text), coverage, early-stop rate, repetition/diversity | test (200) | https://huggingface.co/datasets/hlt-lab/voicebench (subset `commoneval`) | https://github.com/MatthewCYM/VoiceBench (data loader); custom script for WER/coverage |

**Metrics (definitions):**
- **WER**: Word Error Rate between Whisper transcript of generated audio and the model’s generated text (lower is better).
- **Coverage**: `#words(ASR transcript) / #words(model text)` (a ratio typically near 1; can exceed 1 if ASR inserts extra words); values well below 1 often indicate truncated speech.
- **Early-stop rate**: fraction of prompts with Coverage < 0.8.
- **Repetition/diversity guardrail**: distinct-2 on the ASR transcript (unique bigrams / total bigrams; higher means more lexical diversity); used to detect forced filler.

**Bucketed analysis without selection bias:**
- Define long-response buckets using the **C1 text length** (e.g., 0–80 words, 80–160 words, 160–240 words; exact cutoffs chosen to yield a reasonable number of samples per bucket). Report WER and early-stop rate per bucket for C0/C1/C2 on the *same set of prompts*.
- Separately report audio duration distributions for each condition, but do not use realized durations to select the evaluation subset.

### Main Results

#### Published evidence (motivation)
Ex-Omni reports speech-text WER (%) on CommonEval (audio segments up to 60s) using Whisper-V3-Large, stratified by the realized audio duration bucket (0–20s, 20–40s, 40–60s) [Ex-Omni A.4](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/sections/A.4%20Supplementary%20of%20Experiments%20Results%20and%20Analysis.md):
- Qwen2.5-Omni: 0–20s 9.19, 20–40s 5.51, 40–60s 14.35.
- Ex-Omni: 0–20s 12.87, 20–40s 6.48, 40–60s 10.66.

(Table 9 also reports an overall 0–60 summary column; we treat that column as reported in the paper but do not rely on it for bucketed comparisons.)

These numbers motivate the phenomenon; the verification runs below will recompute all metrics under controlled decoding settings.

#### Results Table

| Method | Base Model | Benchmark | WER (all) | WER (long text bucket by C1 text length) | Early-stop rate (Coverage<0.8) | Source | Notes |
|--------|------------|-----------|-----------|------------------------------------|-------------------------------|--------|------|
| C0: default decoding | Qwen2.5-Omni-7B | VoiceBench/CommonEval | **TBD** | **TBD** | **TBD** | - | Needs re-run |
| C1: cap-only (raise audio-token budget) | Qwen2.5-Omni-7B | VoiceBench/CommonEval | **TBD** | **TBD** | **TBD** | - | Needs re-run |
| C2: TLC-AS (cap-only + text-length-coupled EOS suppression) | Qwen2.5-Omni-7B | VoiceBench/CommonEval | **TBD** | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| C2 (full) | Text-length-coupled EOS suppression | Best WER / lowest early-stop on long bucket |
| C2 with constant min length | Replace text-conditioned floor with a fixed min_audio_tokens | If coupling matters, fixed floor should either under-fix (too small) or cause repetition (too large) |

### Analysis (Optional)

- **Failure-mode audit**: Under C1, measure what fraction of long-bucket failures are “cap hits” (audio duration within 1s of the configured max) vs “early EOS” (stops far below the max). TLC-AS is expected to help mainly in the early-EOS cases.

---

## Success Criteria

**Criterion 1: TLC-AS improves over cap-only on long outputs**
- Hypothesis: For prompts whose **C1 text length** falls in a long bucket (e.g., 160–240 words), C2 reduces median WER relative to C1 and reduces early-stop rate.
- Validation: Compare C2 vs C1 on the fixed long-bucket subset (bucket assignment by C1 text length).

**Criterion 2: TLC-AS does not induce degenerate filler**
- Hypothesis: C2 does not substantially reduce ASR transcript diversity compared to C1.
- Validation: Require distinct-2(C2) ≥ 0.95 × distinct-2(C1) on the long-bucket subset, where **distinct-2 is the ratio of unique bigrams to total bigrams in the ASR transcript**.

**Criterion 3: Mechanism check (premature EOS exists)**
- Hypothesis: Under C1 (high token budget), a non-trivial fraction of outputs still stop early (Coverage < 0.8) without being near the cap.
- Validation: Report the early-EOS fraction (Coverage<0.8 AND not within 1s of the cap). If this fraction is **< 10%** on the long text bucket, then EOS suppression is unlikely to matter and the proposal should be refuted (or pivoted to chunked generation).

---

## Impact Statement

If successful, TLC-AS provides a small, deployable decoding patch that improves long-form spoken responses in open omni models without retraining. This directly benefits developers of voice assistants and multimodal agents by reducing cases where users hear incomplete answers even when the model produces a complete text response.

---

## References

- [Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/meta/meta_info.txt) - Zhang et al., 2026
- [Qwen3-Omni Technical Report](./references/Qwen3-Omni-Technical-Report/meta/meta_info.txt) - Xu et al., 2025
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) - Xu et al., 2025
- [VoiceBench: Benchmarking LLM-Based Voice Assistants](https://arxiv.org/abs/2410.17196) - Yin et al., 2024
- [LLaMA-Omni: Seamless Speech Interaction with Large Language Models](https://arxiv.org/abs/2409.06666) - ICTNLP, 2024
- [LLaMA-Omni 2](https://arxiv.org/abs/2505.02625) - ICTNLP, 2025
- [OpenOmni](https://arxiv.org/abs/2501.04561) - Luo et al., 2025
- [VITA-1.5](https://arxiv.org/abs/2501.01957) - VITA-MLLM, 2025
- [Moshi](https://arxiv.org/abs/2410.00037) - Kyutai, 2024
- [MGM-Omni](https://arxiv.org/abs/2509.25131) - Wang et al., 2025
- [SpeechGPT](https://arxiv.org/abs/2305.11000) - Zhang et al., 2023
- [SpeechGPT-Gen](https://arxiv.org/abs/2401.13527) - Zhang et al., 2024
- [SpeechTokenizer](https://arxiv.org/abs/2308.16692) - Zhang et al., 2023
- [GLM-4-Voice](https://arxiv.org/abs/2412.02612) - Zeng et al., 2024
- [CosyVoice 2](https://arxiv.org/abs/2412.10117) - Du et al., 2024
- [Seed-TTS](https://arxiv.org/abs/2406.02430) - Anastassiou et al., 2024
- [VALL-E](https://arxiv.org/abs/2301.02111) - Wang et al., 2023
- [VALL-E 2](https://arxiv.org/abs/2406.05370) - Chen et al., 2024
- [AudioLM](https://arxiv.org/abs/2209.03143) - Borsos et al., 2022
- [EnCodec](https://arxiv.org/abs/2210.13438) - Défossez et al., 2022
- [SoundStream](https://arxiv.org/abs/2107.03312) - Zeghidour et al., 2021
- [BigVGAN](https://arxiv.org/abs/2206.04658) - Lee et al., 2022
- [SeamlessM4T](https://arxiv.org/abs/2308.11596) - Meta AI, 2023
- [UniTalker](https://arxiv.org/abs/2408.00762) - Fan et al., 2024
- [Audio2Face-3D](https://arxiv.org/abs/2508.16401) - Chung et al., 2025
- [FaceFormer](https://arxiv.org/abs/2112.05329) - Fan et al., 2022
- [CodeTalker](https://arxiv.org/abs/2301.02379) - Xing et al., 2023
- [Shanks](https://arxiv.org/abs/2510.06917) - Chiang et al., 2025
