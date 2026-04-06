# untitled

# Step-Down Bridge Guidance Scheduling for Dual-CFG in MOVA Video–Audio Diffusion

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, CVPR, ICCV (or similar top AI conferences)

## Introduction

### Context and Motivation

High-quality video generation is quickly becoming commoditized, but **video without synchronized audio** remains far from what users consider “finished” content. Many open systems therefore rely on **cascaded pipelines** (generate video, then generate audio conditioned on that video), which can accumulate errors and prevent bidirectional interactions between modalities during generation.

Recent open models have started to tackle **joint video–audio generation** directly. **[MOVA](./references/MOVA-Towards-Scalable-and-Synchronized-Video-Audio-Generation/meta/meta_info.txt)** is a prominent example: it combines a large video diffusion transformer with a text-to-audio diffusion model, and couples them through cross-modal “Bridge” layers to produce synchronized video and audio. MOVA further proposes **dual classifier-free guidance (dual CFG)** as an inference-time knob to strengthen cross-modal synchronization.

However, MOVA’s own ablations show that this knob reveals a fundamental deployment pain point: as cross-modal guidance becomes stronger, **synchronization improves** (better DeSync and lip-sync metrics), but **speech quality and instruction-following degrade** (worse DNSMOS and higher cpCER), and human preference slightly drops under dual CFG. This implies that “turn the sync knob up” is not a free win for real-world audio–video creators, especially for speech-heavy content.

### The Problem

MOVA derives dual CFG by viewing joint generation as having two conditioning sources: the **text prompt** \(c_T\) and **cross-modal Bridge conditioning** \(c_B\). In the paper’s Eq. (6), the guided velocity (or noise prediction) is:

\[
\tilde v_\theta
= v_\theta(z_t,\varnothing,\varnothing)
+ s_B\,[v_\theta(z_t,\varnothing,c_B)-v_\theta(z_t,\varnothing,\varnothing)]
+ s_T\,[v_\theta(z_t,c_T,c_B)-v_\theta(z_t,\varnothing,c_B)]
\]

where \(s_B\) scales cross-modal alignment guidance and \(s_T\) scales text guidance.

In MOVA Table 5 (360p), increasing \(s_B\) reduces DeSync and improves lip-sync confidence (LSE-C), but degrades DNSMOS and cpCER. MOVA attributes this to **“conditional interference”**: emphasizing synchronization geometry during multi-branch sampling reduces sensitivity to the text instruction (e.g., speaker tags), hurting what is said and how natural it sounds.

What is missing today is a **minimal, training-free inference rule** that reshapes this trade-off: can we keep most of the synchronization gain while recovering instruction-following for speech?

### Key Insight and Hypothesis

**Key insight**: In diffusion-style generation, **different timesteps play different roles**: early/high-noise steps set coarse global structure (here: timing and cross-modal alignment), while late/low-noise steps refine fine details (here: phonetics, timbre, and adherence to speaker-tag instructions). If conditional interference is primarily a **late-step** phenomenon, then using a constant \(s_B\) is unnecessarily harmful.

**Hypothesis**: A **timestep-dependent bridge guidance schedule** \(s_B(t)\) that is **high early** and **low late** will improve speech instruction-following (lower WER on Verse-Bench speech prompts) while retaining most audio–video synchronization gains (low AV-A / high LSE-C). The hypothesis could be wrong if (i) the interference happens primarily in early steps, (ii) cross-modal alignment requires strong guidance all the way to the end, or (iii) the public benchmark metrics are not sensitive to the speech degradation observed in MOVA’s cpCER/DNSMOS.

---

## Proposed Approach

### Overview

We propose **Step-Down Bridge Guidance**, a training-free modification to MOVA dual CFG:

- Keep text guidance \(s_T\) fixed (e.g., the default CFG scale used in MOVA code).
- Replace constant \(s_B\) with a simple **two-phase schedule** that turns down bridge guidance in the second half of denoising.
- Use a **time-reversal control (Step-Up)** to isolate whether the benefit comes from temporal allocation (when guidance is applied) rather than average magnitude.

### Method Details

#### 1) Implement dual CFG branches (Eq. 6)

The open-source MOVA pipeline currently implements only standard (text-only) CFG. Verification will implement dual CFG by computing, at each inference step \(k\), the three required model evaluations:

- \(v_{00}(k)=v_\theta(z_t,\varnothing,\varnothing)\): null text conditioning and Bridge interactions disabled.
- \(v_{0B}(k)=v_\theta(z_t,\varnothing,c_B)\): null text conditioning and Bridge interactions enabled.
- \(v_{TB}(k)=v_\theta(z_t,c_T,c_B)\): full text conditioning and Bridge interactions enabled.

Then compute the guided prediction using Eq. (6) with a fixed \(s_T\) and a scheduled \(s_B(k)\).

Implementation sketch (no code; for the verifier):
- Extend `mova/diffusion/pipelines/pipeline_mova.py` to support a `cfg_mode="dual"` path instead of raising `NotImplementedError`.
- Add a mechanism to disable Bridge injection for the \(v_{00}\) branch (e.g., pass a `condition_scale=0` into the bridge module, or temporarily bypass `dual_tower_bridge.should_interact`).
- Use the existing “empty prompt” embeddings as \(\varnothing\) for text-unconditional branches.

#### 2) Define a single, non-tuned schedule and its control

Fix total inference steps to **K=25** (as in the MOVA repo’s SGLang example). Let \(k\in\{0,\dots,K-1\}\) be the denoising step index, where \(k=0\) corresponds to highest noise.

- **Constant baseline**: \(s_B(k)=3.5\) for all \(k\) (this matches the paper’s strong-sync setting).
- **Step-Down (ours)**: \(s_B(k)=3.5\) for \(k < K/2\), and \(s_B(k)=1.0\) for \(k \ge K/2\).
- **Step-Up (control)**: \(s_B(k)=1.0\) for \(k < K/2\), and \(s_B(k)=3.5\) for \(k \ge K/2\).

Step-Down and Step-Up have identical average bridge guidance but opposite temporal allocation.

### Key Innovations

1. **Timing-is-the-hypothesis test for dual CFG**: The Step-Up control is the key scientific lever—it tests whether “when bridge guidance is applied” matters, rather than simply reducing its magnitude.
2. **Training-free trade-off reshaping in a new domain**: CFG scheduling is well-studied in text-to-image diffusion, but here the trade-off is **cross-modal synchronization vs speech instruction-following** in a joint video–audio generator.
3. **Reproducible dual-CFG implementation for MOVA**: Making Eq. (6) executable in the public MOVA codebase enables systematic future work on multi-condition guidance in multimodal diffusion.

---

## Related Work

### Field Overview

**Joint audio–video generation** spans cascaded pipelines and end-to-end joint models. Cascaded approaches (e.g., generate video then audio) can achieve strong temporal alignment but cannot let audio influence the video trajectory. Joint models use dual-stream architectures with cross-modal attention or fusion modules, but scaling them is difficult and often reveals new controllability trade-offs.

**Classifier-free guidance (CFG)** is the dominant inference-time control mechanism for conditional diffusion models, improving prompt adherence at the cost of diversity and sometimes fidelity. Recent work shows that **time-varying guidance schedules** can improve quality–diversity trade-offs in text-to-image diffusion, but these studies typically focus on single-condition CFG. MOVA’s dual CFG introduces a second condition source (cross-modal Bridge interactions), creating a new trade-off surface that has not been systematically optimized.

### Related Papers

- **[MOVA: Towards Scalable and Synchronized Video-Audio Generation](./references/MOVA-Towards-Scalable-and-Synchronized-Video-Audio-Generation/meta/meta_info.txt)**: Introduces dual-tower joint video–audio diffusion and dual CFG; reports the sync vs speech/instruction-following trade-off we target.
- **[InstructPix2Pix](https://arxiv.org/abs/2211.09800)**: Popularized dual conditioning guidance in diffusion-style image editing, inspiring multi-condition CFG formulations.
- **[Analysis of Classifier-Free Guidance Weight Schedulers](https://arxiv.org/abs/2404.13040)**: Shows that time-varying CFG weights can improve text-to-image generation, motivating scheduling rather than constant guidance.
- **[Rethinking Classifier-Free Guidance for Diffusion Models](https://arxiv.org/abs/2407.02687)**: Analyzes CFG variants and timestep guidance ideas; supports the broader theme that guidance should be timestep-aware.
- **[Navigating with Annealing Guidance Scale in Diffusion Space](https://arxiv.org/abs/2506.24108)**: Proposes learned/annealed guidance schedules, demonstrating non-constant guidance benefits.
- **[Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models](https://arxiv.org/abs/2509.22007)**: Provides a stage-wise explanation for why guidance strength should vary across denoising.
- **[Dynamic Classifier-Free Diffusion Guidance via Online Feedback](https://arxiv.org/abs/2509.16131)**: Uses feedback to pick timestep-wise guidance, showing schedules can be prompt-dependent.
- **[Adaptive Projected Guidance (APG)](https://openreview.net/forum?id=3eb92e3beef0e1f9f260c292855e9518dda3084f)**: Improves CFG at high guidance scales by modifying update geometry; complementary to scheduling.
- **[Synchformer](https://arxiv.org/abs/2310.13339)**: Provides the synchronization predictor used by Verse-Bench (AV-A) and MOVA (DeSync).
- **[SyncNet / “Out of Time”](https://arxiv.org/abs/1609.03467)**: Classic lip-sync evaluation model used for LSE-style metrics.
- **[Verse-Bench](https://huggingface.co/datasets/dorni/Verse-Bench)**: Public benchmark and dataset for joint audio–video generation used for our evaluation.
- **[UniVerse-1](https://arxiv.org/abs/2509.06155)**: An open joint audio–video model introducing Verse-Bench; a key baseline family.
- **[LTX-2](https://arxiv.org/abs/2602.02464)**: Large-scale joint audio–video foundation model cited by MOVA; strong synchronous baseline.
- **[Ovi](https://arxiv.org/abs/2507.08123)**: Dual-tower audio–video model with cross-modal fusion.
- **[MMAudio](https://arxiv.org/abs/2502.07517)**: Cascaded video-to-audio system used as a baseline family and to motivate joint modeling.
- **[ImageBind](https://arxiv.org/abs/2305.05665)**: Cross-modal embedding used for semantic audio–video alignment metrics (IB-Score).
- **[Meta AudioBox Aesthetics](https://arxiv.org/abs/2501.02881)**: Audio quality assessment tool used in MOVA’s data filtering and audio evaluation.
- **[AudioLDM](https://arxiv.org/abs/2301.12503)**: Latent diffusion for text-to-audio; representative audio diffusion baseline.
- **[AudioLDM2](https://arxiv.org/abs/2308.05734)**: Stronger text-to-audio diffusion baseline used in MOVA audio-tower comparisons.
- **[Stable Audio Open](https://arxiv.org/abs/2407.14304)**: Modern text-to-audio generation; relevant for audio quality comparisons.
- **[MMDisCo / Discriminator-Guided Cooperative Diffusion](https://arxiv.org/abs/2405.17842)**: A different approach to joint generation by aligning pretrained models with guidance signals.
- **[Effective Adaptation of Audio and Video Diffusion Models for Joint Generation](https://arxiv.org/abs/2409.17550)**: Studies how to combine pretrained audio and video diffusion models; relevant joint-generation baseline family.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Joint audio–video diffusion models | Generate audio and video together with cross-modal fusion | MOVA, UniVerse-1, LTX-2, Ovi | Verse-Bench; DeSync/AV-A; LSE; human pref | Hard to balance sync vs fidelity; costly inference |
| Cascaded pipelines (V→A) | Video first, audio conditioned on video | MMAudio; WAN+MMAudio | DeSync/AV-A; audio quality metrics | No audio→video influence; compounding errors |
| CFG + multi-condition guidance | Combine conditional and unconditional branches to steer samples | CFG; InstructPix2Pix dual CFG; MOVA dual CFG | Quality vs diversity; adherence metrics | Constant guidance causes artifacts/trade-offs |
| Guidance scheduling | Vary guidance strength across timesteps | CFG schedulers (2404.13040); Annealing guidance (2506.24108); Stage-wise CFG (2509.22007) | T2I metrics (FID/CLIP); human pref | Mostly single-condition; unclear for multimodal trade-offs |

### Closest Prior Work

1. **MOVA (dual CFG)**: Defines the two-scale guidance formulation (Eq. 6) and shows the sync–speech trade-off as \(s_B\) increases. It does not study timestep-dependent \(s_B(t)\) or whether temporal allocation matters.
2. **CFG scheduling (e.g., 2404.13040; 2509.22007)**: Shows that constant guidance is suboptimal in diffusion models and motivates timestep schedules. These works focus on text-to-image diffusion and single-condition CFG, not cross-modal guidance vs instruction-following.
3. **Dual conditioning guidance (InstructPix2Pix)**: Uses multiple conditioning signals in diffusion image editing. It does not study audio–video synchronization or speech instruction-following trade-offs.
4. **Universal lip-sync diffusion (e.g., OmniSync)**: Uses diffusion for lip-sync and introduces spatiotemporal guidance schedules. It targets lip-sync alignment specifically, not joint audio–video generation with a Bridge-induced condition competing with text instructions.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MOVA dual CFG | Constant \(s_B\) controls sync vs speech trade-off | Uses constant bridge guidance; does not test timing | Replace constant \(s_B\) with a fixed step schedule; add step-up control | If interference is late-step, step-down improves WER while preserving sync |
| CFG scheduler analysis (2404.13040) | Shows varying guidance across timesteps helps T2I | Single-condition setting | Apply scheduling to multi-condition dual CFG | Multi-condition schedules can reshape sync–speech trade-off |
| InstructPix2Pix | Dual conditioning guidance for image editing | Not a multimodal sync problem | Use dual-CFG structure but schedule only the cross-modal component | isolates cross-modal vs text interference |
| OmniSync / DS-CFG | Spatiotemporal guidance for lip-sync | Not joint audio–video generation | Focus on bridge-guidance timing for joint video–audio diffusion | Keeps broader joint-generation capability while improving speech adherence |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| MOVA-360p | 32B total (18B active) | https://huggingface.co/OpenMOSS-Team/MOVA-360p | Use official MOVA inference; implement dual CFG Eq. (6) in code |

**Training Data (if applicable):**

No training data needed — **inference only**.

**Other Resources (if applicable):**

- Verse-Bench dataset: https://huggingface.co/datasets/dorni/Verse-Bench
- Verse-Bench evaluation code: https://github.com/Dorniwang/Verse-Bench

**Resource Estimate**:

- **Compute budget**: Inference-only.
  - MOVA repo reports **9.0 s/step** on H100 (360p, 8 seconds) for one denoising step under component-wise offload (README table). With **K=25**, this is ~225 s/sample (not counting dual-CFG’s extra NFE). Dual CFG Eq. (6) uses **NFE=3**, so estimate ~1.5× over NFE=2 CFG: **~340 s/sample**.
  - For 100 Verse-Bench set3 samples × 3 conditions ≈ 300 samples → ~28 hours on a single H100-equivalent GPU. On A100s, expect a small constant-factor slowdown; still well within the 768 GPU-hour budget.
- **GPU memory**: repo suggests 48GB VRAM with CPU offload for 360p; A100 80GB should suffice.
- **API usage**: none.

**Infrastructure constraints**:
- No search APIs needed.
- No human evaluation required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Verse-Bench set3 (speech-heavy) | TED-style speech prompts with transcripts | **AV-A** (↓), **LSE-C** (↑), **WER** (↓) | set3 | https://huggingface.co/datasets/dorni/Verse-Bench | https://github.com/Dorniwang/Verse-Bench (`calculate_metrics.py`) |

**Evaluation Scripts:**
- Use the official Verse-Bench evaluation implementation (Synchformer for AV-A, SyncNet for LSE-C, and built-in WER calculation).

### Main Results

#### Baselines (published evidence)

MOVA reports Verse-Bench results for different constant \(s_B\) values (Table 5). These numbers motivate the trade-off and serve as evidence-based baselines.

| Method | Description | Settings | Results | Reference |
|---|---|---|---|---|
| MOVA text-only CFG (equivalently \(s_B=1.0\)) | No explicit bridge amplification; prioritizes text semantics | MOVA-360p; Verse-Bench; \(s_B=1.0\) | DNSMOS=3.797, DeSync=0.475, LSE-C=6.278, cpCER=0.177 | MOVA Table 5 in [Ablation Study](./references/MOVA-Towards-Scalable-and-Synchronized-Video-Audio-Generation/sections/Ablation%20Study.md) |
| MOVA dual CFG constant | Strong constant bridge guidance (best sync in Table 5 ablation) | MOVA-360p; Verse-Bench; \(s_B=3.5\) | DNSMOS=3.674, DeSync=0.351, LSE-C=7.800, cpCER=0.247 | MOVA Table 5 in [Ablation Study](./references/MOVA-Towards-Scalable-and-Synchronized-Video-Audio-Generation/sections/Ablation%20Study.md) |

**Note**: Verse-Bench public evaluation code reports **AV-A** (Synchformer-based desynchronization proxy) and **WER**, not DNSMOS/cpCER. In verification, we will report (AV-A, LSE-C, WER) as the primary metrics. DeSync/LSE-C from Table 5 are included as published evidence that stronger \(s_B\) improves synchronization but hurts speech-related metrics.

#### Results Table (to be verified)

(All results are **TBD** and will be filled by running the verification experiments under the Verse-Bench protocol.)

| Method | Description | Settings | Results | Reference |
|---|---|---|---|---|
| Dual CFG constant | Constant bridge guidance | MOVA-360p; K=25; \(s_B(k)=3.5\) | AV-A=**TBD**, LSE-C=**TBD**, WER=**TBD** | Needs re-run |
| Dual CFG Step-Up (control) | Time-reversal control for temporal allocation | MOVA-360p; K=25; \(s_B(k)=1.0\rightarrow 3.5\) at k=K/2 | AV-A=**TBD**, LSE-C=**TBD**, WER=**TBD** | Needs re-run |
| **Dual CFG Step-Down (ours)** | Proposed schedule | MOVA-360p; K=25; \(s_B(k)=3.5\rightarrow 1.0\) at k=K/2 | AV-A=**TBD**, LSE-C=**TBD**, WER=**TBD** | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Step-Up | Reverse temporal allocation of \(s_B\) | If late-step interference is real, Step-Up should be worse on WER than Step-Down |
| Constant \(s_B=3.5\) | No schedule | Establishes baseline sync strength; may harm WER |

### Analysis (Optional)

- Log per-step norms of the bridge term \(\|v_{0B}-v_{00}\|\) and text term \(\|v_{TB}-v_{0B}\|\) to test whether Step-Down reduces late-step bridge dominance.

---

## Success Criteria

**Criterion 1: Timing matters (core claim)**
- Hypothesis: Step-Down Pareto-dominates Step-Up on (AV-A, WER).
- Validation: On Verse-Bench set3, Step-Down is considered better if it improves at least one metric by a non-trivial margin while not worsening the other:
  - Either AV-A(step-down) ≤ AV-A(step-up) − 0.03 with WER(step-down) ≤ WER(step-up) + 0.01, or
  - WER(step-down) ≤ WER(step-up) − 0.03 with AV-A(step-down) ≤ AV-A(step-up) + 0.01.
  - If both metrics differ by ≤0.01 between step-down and step-up, treat as null.

**Criterion 2: Preserve synchronization while improving speech adherence (practical value)**
- Hypothesis: Step-Down reduces WER relative to constant \(s_B=3.5\) while not substantially hurting synchronization.
- Validation: WER(step-down) < WER(constant-3.5) and AV-A(step-down) ≤ AV-A(constant-3.5) + 0.02.

---

## Impact Statement

If successful, this provides a simple, training-free knob for deploying joint video–audio generators: creators can preserve synchronization benefits of strong cross-modal guidance while recovering speech instruction-following, improving usability for dialogue-heavy content. More broadly, it suggests that **multi-condition guidance in multimodal diffusion should be scheduled**, not constant.

---

## References

- [MOVA: Towards Scalable and Synchronized Video-Audio Generation](./references/MOVA-Towards-Scalable-and-Synchronized-Video-Audio-Generation/meta/meta_info.txt) - Yu et al., 2026
- [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800) - Brooks et al., 2023
- [Analysis of Classifier-Free Guidance Weight Schedulers](https://arxiv.org/abs/2404.13040) - Wang et al., 2024
- [Rethinking Classifier-Free Guidance for Diffusion Models](https://arxiv.org/abs/2407.02687) - 2024
- [Navigating with Annealing Guidance Scale in Diffusion Space](https://arxiv.org/abs/2506.24108) - Yehezkel et al., 2025
- [Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models](https://arxiv.org/abs/2509.22007) - Jin et al., 2025
- [Dynamic Classifier-Free Diffusion Guidance via Online Feedback](https://arxiv.org/abs/2509.16131) - 2025
- [Adaptive Projected Guidance (APG)](https://openreview.net/forum?id=3eb92e3beef0e1f9f260c292855e9518dda3084f) - 2025
- [Synchformer](https://arxiv.org/abs/2310.13339) - Iashin et al., 2024
- [Out of Time: Automated Lip Sync in the Wild (SyncNet)](https://arxiv.org/abs/1609.03467) - Son & Zisserman, 2016
- [Verse-Bench dataset](https://huggingface.co/datasets/dorni/Verse-Bench) - Wang et al., 2025
- [Verse-Bench evaluation code](https://github.com/Dorniwang/Verse-Bench) - 2025
- [UniVerse-1: Unified Audio-Video Generation via Stitching of Experts](https://arxiv.org/abs/2509.06155) - Wang et al., 2025
- [Synchformer: Efficient synchronization from sparse cues](https://arxiv.org/abs/2310.13339) - Iashin et al., 2024
- [ImageBind](https://arxiv.org/abs/2305.05665) - Girdhar et al., 2023
- [MMAudio](https://arxiv.org/abs/2502.07517) - Cheng et al., 2025
- [Meta AudioBox Aesthetics](https://arxiv.org/abs/2501.02881) - Tjandra et al., 2025
- [AudioLDM](https://arxiv.org/abs/2301.12503) - Liu et al., 2023
- [AudioLDM2](https://arxiv.org/abs/2308.05734) - 2023
- [Stable Audio Open](https://arxiv.org/abs/2407.14304) - 2024
- [Discriminator-Guided Cooperative Diffusion (MMDisCo)](https://arxiv.org/abs/2405.17842) - 2024
- [Effective Adaptation of Audio and Video Diffusion Models for Joint Generation](https://arxiv.org/abs/2409.17550) - 2024

(Additional baselines such as LTX-2 and Ovi are cited via the MOVA paper’s reference list.)
