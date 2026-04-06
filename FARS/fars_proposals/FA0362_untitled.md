# untitled

# ScaffoldSwap: Are Discrete Speech Units Necessary as a Temporal Scaffold for Audio→3D Facial Animation?

## Scope and Constraints

- **Paper type**: short ablation study

## Introduction

### Context and Motivation

Audio-driven 3D facial animation maps a speech waveform to a time series of facial motion (e.g., 3D mesh vertices or ARKit blendshape coefficients (52 facial expression parameters used in Apple’s ARKit)). It is a core component of interactive avatars and omni-modal LLM systems that speak and animate a character.

A central design choice is the **speech representation** used to condition the face decoder. Common options include (i) continuous self-supervised speech features (wav2vec2/WavLM/HuBERT embeddings), (ii) discrete *motion* priors (VQ codebooks over facial motion), and (iii) more recently, **discrete speech unit token streams** produced by speech tokenizers/codecs.

### The Problem

Ex-Omni proposes using **discrete speech units** as an explicit “temporal scaffold” for facial animation generation: unit embeddings are resampled to the video frame rate to form frame-level queries, and semantic information is injected via token-as-query gated fusion (TQGF) (Ex-Omni, Sec. “Joint Speech and 3D Facial Animation Generation”). However, it remains unclear whether discretization is uniquely helpful, or whether a simpler and more interpretable scaffold—**phoneme IDs with explicit timing (durations/boundaries)**—can provide equivalent temporal structure.

This is decision-changing for real-time avatar/agentic systems: many production pipelines already use **phoneme/viseme-driven** lip-sync (e.g., Meta/Oculus LipSync and commercial tools like Speech Graphics / JALI), while recent OLLM work proposes adding a speech-tokenizer/codec dependency (discrete units). If phoneme+timing matches units, systems can keep simpler timing scaffolds; if units win, it justifies the added complexity.

### Key Insight and Hypothesis

**Key insight:** In frame-aligned audio→face models, the main benefit of “unit scaffolding” may be providing a dense, frame-aligned alignment grid rather than discretization itself. A phoneme scaffold with explicit timing expanded to the same temporal resolution could match this benefit.

**Hypothesis:** On BIWI (a standard speech→3D face benchmark), under a fixed UniTalker-style decoder and matched prosody features (F0, energy), **phoneme+timing conditioning will match discrete-unit conditioning on LVE (Lip Vertex Error: average over frames of the maximal L2 error among lip vertices; lower is better)**. The outcome is uncertain because discrete units may encode sub-phoneme coarticulation cues (within-phoneme articulation dynamics) that forced-aligned phonemes cannot represent.

Main confound: transcript/alignment noise. We control for it by prioritizing BIWI (a controlled read-speech corpus) and treating ASR→alignment as a diagnostic fallback if transcripts are missing.

---

## Proposed Approach

### Overview

We run a controlled **representation swap**: keep the same audio→face decoder, optimizer, training budget, and split, and change only the speech conditioning representation.

### Method Details

**Dataset / target:** BIWI (3D mesh vertices at 25 fps; UniTalker Tab.1). We follow UniTalker and compress vertex motion with **PCA (L=512)** (UniTalker Sec. 9.3).

**Shared decoder:** UniTalker-Base-style audio→mesh model trained on BIWI only (single-dataset head). Hyperparameters follow UniTalker: Adam, lr=1e-4; train for 100 epochs (UniTalker Sec. 4.2). Only the **audio frontend** changes.

**Shared prosody controls:** extract **F0** and **energy** at 50 Hz (20 ms hop) and append to all conditions.

**Conditions (all produce frame-aligned features, then resample to 25 fps with the same rule):**

| Condition | Speech scaffold | Notes |
|---|---|---|
| A (SSL) | WavLM-base-plus embeddings | Standard strong baseline used by UniTalker-Base |
| B (Units) | HuBERT embeddings → k-means unit IDs (e.g., K=200) → unit embeddings | Tests “discrete units as scaffold” without Ex-Omni code |
| C (Phoneme+timing) | MFA forced alignment → phoneme ID per 50 Hz frame + within-phoneme position p∈[0,1] + phoneme duration d | Tests explicit timing scaffold |

### Key Innovations

- **Representation-isolated** test of Ex-Omni’s scaffold premise under a fixed decoder.
- **Information symmetry control**: prosody features are provided to all conditions.
- **Decision-oriented outcome**: either discrete units are uniquely beneficial, or explicit phoneme timing is sufficient.

---

## Related Work

### Field Overview

Speech→3D face work varies by (i) motion representation (vertices vs blendshapes), (ii) decoder family (transformer/RNN/diffusion), and (iii) speech representation (handcrafted, continuous SSL, timing-aware, discrete tokens). Our proposal isolates axis (iii) while holding (i–ii) fixed.

### Key References

- **[Ex-Omni](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/meta/meta_info.txt)**: Introduces discrete speech units as a temporal scaffold (ARKit-52).
- **[UniTalker](./references/UniTalker-Scaling-up-Audio-Driven-3D-Facial-Animation-through-A-Unified-Model/meta/meta_info.txt)**: Strong audio→3D face baseline; reports BIWI + VOCASET metrics and provides an execution substrate.
- **[FaceFormer](./references/FaceFormer-Speech-Driven-3D-Facial-Animation-with-Transformers/meta/meta_info.txt)**: Canonical transformer audio→mesh baseline (SSL speech features).
- **[Content+Style Aware](https://arxiv.org/abs/2408.07005)**: Uses MFA-derived phoneme durations (FastSpeech2-style length regulation) for 3D face.
- **[Audio2Face-3D](https://arxiv.org/abs/2508.16401)**: Production-oriented audio→face model that combines SSL features with phoneme/viseme supervision.
- **[VOCA](https://arxiv.org/abs/1905.03079)**: Introduces VOCASET (a standard audio→3D face benchmark).

### Closest Prior Work

- **Ex-Omni** motivates discrete speech units as temporal scaffolding but does not ablate against phoneme+timing.
- **UniTalker** provides strong BIWI baselines and an execution substrate but does not compare unit tokens vs phoneme timing.
- **Content+Style Aware** shows phoneme-duration conditioning is viable, but does not compare to unit tokens under matched decoder.

**Novelty Kill Search Summary:** Queried “discrete speech units vs phoneme duration facial animation”, “HuBERT units blendshape ablation”, and OpenReview submissions (2025–2026). No prior work was found that isolates the scaffold representation (units vs phoneme+timing vs SSL) under a fixed decoder on 3D face benchmarks as of 2026-02-27.

### Comparison Table

| Related work | What it does | Key limitation | What we change |
|---|---|---|---|
| Ex-Omni | discrete speech units scaffold + semantic fusion | no scaffold swap ablation | swap scaffold only |
| UniTalker | strong unified audio→face model | no unit vs phoneme timing test | compare 3 scaffolds |
| FaceFormer | SSL-only conditioning | no discrete/timing variants | add units + timing |
| Content+Style Aware | phoneme-duration conditioning | no unit baseline | add discrete units |

---

## Experiments

### Experimental Setup

**Core runs (decisive):** 3 conditions (A/B/C) × 3 seeds on **BIWI** *and* 3 conditions × 3 seeds on **VOCASET** (both are standard speech→3D face benchmarks; UniTalker evaluates both in Tab.2). This is the minimal breadth needed to make the result portable beyond a single dataset.

**Baselines:** Prompting / best-of-N are not applicable because this is supervised audio→motion prediction evaluated by geometric error metrics.

Baseline ladder for this setting:
1) trivial constant-motion baseline; 2) handcrafted audio features (MFCC/log-mel) + small temporal model; 3) strong SSL baseline (A); 4) literature reference numbers (UniTalker Table 2).

**Base Models / Tools:**

| Component | Version | Link |
|---|---|---|
| WavLM-base-plus | microsoft/wavlm-base-plus | https://huggingface.co/microsoft/wavlm-base-plus |
| HuBERT base | facebook/hubert-base-ls960 | https://huggingface.co/facebook/hubert-base-ls960 |
| MFA | Montreal Forced Aligner | https://montreal-forced-aligner.readthedocs.io/ |

**Training Data:**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| BIWI B3D(AC)^2 | train/test audio→mesh (D0) | 0.33h (UniTalker Tab.1) | https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html | research-only; no redistribution (BIWI EULA PDF) |
| VOCASET (VOCA) | train/test audio→mesh (D1) | 0.56h (UniTalker Tab.1) | https://voca.is.tue.mpg.de/ (signup) | research-only (VOCA license) |

**Resource Estimate (order-of-magnitude):**
- UniTalker reports 100 epochs ≈ 2 days on 1×V100 for UniTalker-L-[D0–D7] (18.53h total; UniTalker Sec. 4.2). Our two datasets are tiny by comparison (BIWI 0.33h, VOCASET 0.56h; UniTalker Tab.1), so training should be much cheaper.
- Budget: **≤150 A100 GPU-hours total** for 18 main runs (BIWI+VOCASET, 3 conditions × 3 seeds × 2 datasets) + 3 ablation runs (C w/o timing on BIWI).

### Benchmarks and Metrics

- **BIWI-Test-A**: standard BIWI split used by FaceFormer and UniTalker for LVE reporting.
- **Primary metric: LVE** (lip vertex error; average over frames of maximal L2 error among lip vertices; lower is better).
- Secondary diagnostics (optional): MVE/UFVE/FDD as in UniTalker.

### Main Results

All published baseline numbers below are copied from **UniTalker Table 2** (proposal-local raw artifact: `./references/UniTalker-.../sections/4.3 Comparison with Prior Works.md`).

| Method | Base model / audio repr | Benchmark | LVE (↓) | Source | Notes |
|---|---|---|---:|---|---|
| FaceFormer | wav2vec2-base-960h | BIWI-Test-A | 4.9836×1e-4 | UniTalker Tab.2 | (1 run) |
| CodeTalker | wav2vec2-base-960h | BIWI-Test-A | 4.7914×1e-4 | UniTalker Tab.2 | (1 run) |
| SelfTalk | wav2vec2-large-xlsr-53-English | BIWI-Test-A | 4.2485×1e-4 | UniTalker Tab.2 | (1 run) |
| FaceDiffuser | hubert-base-ls960 | BIWI-Test-A | 4.2985×1e-4 | UniTalker Tab.2 | (1 run) |
| UniTalker-B-[D0] | WavLM-base-plus | BIWI-Test-A | 4.3681×1e-4 | UniTalker Tab.2 | (1 run) |
| UniTalker-L-[D0–D7] | wav2vec2-xlsr-53 | BIWI-Test-A | 3.8587×1e-4 | UniTalker Tab.2 | (1 run) |
| **A: SSL+prosody** | UniTalker-Base decoder + WavLM | BIWI-Test-A | TBD (mean±std) | - | 3 seeds |
| **B: Units+prosody** | UniTalker-Base decoder + HuBERT-kmeans | BIWI-Test-A | TBD (mean±std) | - | 3 seeds |
| **C: Phoneme+timing+prosody** | UniTalker-Base decoder + MFA phonemes | BIWI-Test-A | TBD (mean±std) | - | 3 seeds |
| FaceFormer | wav2vec2-base-960h | VOCA-Test | 1.1696×1e-5 m² | UniTalker Tab.2 | (1 run; VOCA LVE uses different unit) |
| CodeTalker | wav2vec2-base-960h | VOCA-Test | 1.1182×1e-5 m² | UniTalker Tab.2 | (1 run) |
| SelfTalk | wav2vec2-large-xlsr-53-English | VOCA-Test | 0.9626×1e-5 m² | UniTalker Tab.2 | (1 run) |
| FaceDiffuser | hubert-base-ls960 | VOCA-Test | 0.9684×1e-5 m² | UniTalker Tab.2 | (1 run) |
| UniTalker-B-[D1] | WavLM-base-plus | VOCA-Test | 0.9381×1e-5 m² | UniTalker Tab.2 | (1 run) |
| UniTalker-B-[D0–D7] | mixed | VOCA-Test | 0.8136×1e-5 m² | UniTalker Tab.2 | (1 run) |
| UniTalker-L-[D0–D7] | wav2vec2-xlsr-53 | VOCA-Test | 0.8303×1e-5 m² | UniTalker Tab.2 | (1 run) |
| **A: SSL+prosody** | UniTalker-Base decoder + WavLM | VOCA-Test | TBD (mean±std) | - | 3 seeds |
| **B: Units+prosody** | UniTalker-Base decoder + HuBERT-kmeans | VOCA-Test | TBD (mean±std) | - | 3 seeds |
| **C: Phoneme+timing+prosody** | UniTalker-Base decoder + MFA phonemes | VOCA-Test | TBD (mean±std) | - | 3 seeds |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| C (full) | phoneme ID + (p, d) + prosody | competitive with B |
| C w/o timing | drop (p, d) | worse LVE if explicit timing drives gains |
| B vs HuBERT-continuous | replace k-means unit IDs with continuous HuBERT features | if B ≈ HuBERT-cont, discretization itself is not the driver |

### Experimental Rigor

- **Seeds**: `seeds=[42, 123, 456]`.
- **Fair comparison**: identical decoder, optimizer, training steps, and prosody features.
- **Sanity check**: time-shuffle audio features within each utterance for condition A → LVE should degrade.
- **Leakage**: enforce BIWI official split; do not mix speaker or sentence subsets across train/test.

---

## Success Criteria

**Hypothesis (directional):** C (phoneme+timing) matches B (discrete units) on LVE; both improve over A (continuous SSL) when prosody is controlled.

**Decision Rule (concrete):**
- Compute mean±std over 3 seeds **separately on BIWI and VOCASET**.
- **Proceed (timing sufficient):** If on *both* datasets, C is within 1 std of B (and both are not worse than A), conclude explicit phoneme timing is sufficient as a scaffold at this scale.
- **Refute (units add info):** If on *either* dataset C underperforms B by >1 std consistently, conclude discrete-unit-style representations provide useful sub-phoneme cues beyond explicit timing for this decoder family.
- **Diagnose discretization vs HuBERT:** If B ≈ HuBERT-continuous, conclude *discretization* is not the driver (it’s the underlying SSL representation).
- **Pivot (alignment confound):** If C is unstable and correlates with alignment confidence, restrict to high-confidence alignments or replace MFA with provided transcripts/phonemes.

---

## Impact Statement

If explicit phoneme timing matches discrete unit scaffolds, avatar/OLLM pipelines can avoid speech-tokenizer/codec dependencies and use simpler ASR+alignment scaffolds with clearer interpretability. If discrete units are strictly better, the result supports Ex-Omni’s design choice and motivates better speech-unit tokenizers for embodied systems.

---

## References

- [Ex-Omni](./references/Ex-Omni-Enabling-3D-Facial-Animation-Generation-for-Omni-modal-Large-Language-Models/meta/meta_info.txt)
- [UniTalker](./references/UniTalker-Scaling-up-Audio-Driven-3D-Facial-Animation-through-A-Unified-Model/meta/meta_info.txt)
- [FaceFormer](./references/FaceFormer-Speech-Driven-3D-Facial-Animation-with-Transformers/meta/meta_info.txt)
- [VOCA](https://arxiv.org/abs/1905.03079)
- [MeshTalk](https://arxiv.org/abs/2104.08223)
- [CodeTalker](https://arxiv.org/abs/2301.02379)
- [SelfTalk](https://arxiv.org/abs/2306.10799)
- [FaceDiffuser](https://arxiv.org/abs/2309.11306)
- [EmoTalk](https://arxiv.org/abs/2303.11089)
- [Content+Style Aware](https://arxiv.org/abs/2408.07005)
- [Audio2Face-3D](https://arxiv.org/abs/2508.16401)
- [BIWI corpus](https://doi.org/10.1109/TMM.2010.2048114)
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477)
- [WavLM](https://arxiv.org/abs/2110.13900)
- [HuBERT](https://arxiv.org/abs/2106.07447)
- [FastSpeech 2](https://arxiv.org/abs/2006.04558)
- [EnCodec](https://arxiv.org/abs/2210.13438)
- [DeepSpeech](https://arxiv.org/abs/1412.5567)
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
- [Learning audio-driven viseme dynamics](https://arxiv.org/abs/2301.06059)
