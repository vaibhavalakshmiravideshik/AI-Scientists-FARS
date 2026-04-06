# untitled

# Partial Round-Trip Stability for Fast Seed Selection in Diffusion Models

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Text-to-image diffusion models (e.g., Stable Diffusion and SDXL) are widely used to generate images from a text prompt by starting from random Gaussian noise and iteratively denoising. A practical pain point is **seed sensitivity**: for the same prompt, different random seeds can produce very different image quality and prompt fidelity. Users often handle this by brute-force **best-of-N sampling** (generate many candidates, pick one), which scales inference cost linearly with N.

Recent work shows that this variance is not merely stochastic noise: some initial noises are systematically “better” than others. This has motivated test-time scaling methods that search the noise/seed space more intelligently. For example, **TTSnap** (Yu et al., 2025) prunes candidate trajectories early using **noise-aware reward models** trained by self-distillation, enabling more exploration under a fixed budget. However, many approaches still add substantial overhead (extra reward models, VAE decoding for intermediate verification, or full round trips) and can be too expensive to use routinely.

### The Problem

A representative recent method is **noise inversion stability** from **Not All Noises Are Created Equally** (Qi et al., 2024), which scores a seed by (i) generating a sample via **deterministic diffusion sampling** using **Denoising Diffusion Implicit Models (DDIM; Song et al., 2020)** and (ii) applying **DDIM inversion** (a deterministic approximate inverse mapping from a generated image back to its initial latent noise; Hertz et al., 2022), then computing the cosine similarity between the original noise and the inverted noise. In their experiments, simply choosing the *most stable* versus *least stable* noise improves **Human Preference Score v2 (HPSv2; a learned model that predicts human preference for images given a prompt, where higher is better)** by **+0.2246** on Pick-a-Pic (27.2688→27.4934) and **+0.2889** on DrawBench (28.1377→28.4266) using SDXL-turbo (Table 1 in their paper). However, scoring a candidate seed requires a **full denoise→invert round trip**.

This creates a compute bottleneck: if a model uses T denoising steps, full stability scoring costs roughly **2T denoiser evaluations per candidate seed**. In text-to-image generation, models are typically run with **classifier-free guidance (CFG)**, which improves text alignment by combining conditional and unconditional predictions (often implemented as either one batched UNet call or two separate UNet calls per step). This cost can dominate any benefit from selecting better seeds, especially when screening dozens to hundreds of candidates.

The goal of this proposal is to reduce the cost of inversion-stability seed selection while preserving its quality gains.

### Key Insight and Hypothesis

**Key insight:** if a seed is “good” because its denoising trajectory lies in a stable basin of the model’s denoise↔invert dynamics, then the deviation from perfect round-trip invertibility should appear early, when the sampler takes its largest effective steps and numerical/score-estimation mismatch is amplified.

**Hypothesis:** a **k-step partial round-trip stability score**

\[ s_k(z_T) = \cos(z_T, \tilde{z}_T) \]\

computed by denoising only k steps from the initial latent noise \(z_T\) to \(z_{T-k}\), then inverting back k steps to \(\tilde{z}_T\), is a faithful proxy for the full stability score \(s_{full}\) (which uses k=T). If true, we can screen many more seeds under the same compute budget and approach the quality of full inversion-stability selection.

This could fail if (i) the stability signal only becomes discriminative at mid/late timesteps when semantic structure forms, or (ii) the partial round-trip signal mainly captures numerical artifacts unrelated to perceptual quality.

---

## Proposed Approach

### Overview

Given a text prompt and a diffusion model sampled with a deterministic scheduler (DDIM), we propose to select the seed that maximizes a **partial round-trip stability** score:

1. Sample K candidate seeds \(\{z_T^{(i)}\}_{i=1}^K\) in latent space.
2. For each candidate, run k DDIM denoising steps to obtain \(z_{T-k}^{(i)}\).
3. Starting from \(z_{T-k}^{(i)}\), run k DDIM inversion steps to obtain \(\tilde{z}_T^{(i)}\).
4. Score the candidate by \(s_k^{(i)} = \cos(z_T^{(i)}, \tilde{z}_T^{(i)})\).
5. Select \(i^* = \arg\max_i s_k^{(i)}\) and finish denoising only this candidate for the remaining \(T-k\) steps.

This replaces the full 2T-step round trip used by Qi et al. with a 2k-step round trip. In the main experiments we will **pre-register k=2** as the primary configuration, since it is small enough to be much cheaper than full stability but large enough to include at least one non-trivial forward and inverse step.

### Method Details

**Partial round-trip scoring (latent space):**
- We operate entirely in the latent space of the diffusion model (for latent diffusion models, this is the VAE latent). We do not decode intermediate latents to pixels.
- We use a deterministic scheduler (DDIM) so that denoising and inversion are well-defined.

**Similarity metric:**
- Primary: cosine similarity between flattened latents \(z_T\) and \(\tilde{z}_T\).
- (Optional ablation) alternative similarity metrics (L2/MSE) if cosine is too concentrated.

**Compute accounting (UNet-call units):**
- We count one “denoiser evaluation” as one UNet forward call for one scheduler step. In common implementations of classifier-free guidance (CFG), conditional and unconditional predictions are concatenated into a single batch, so CFG is still one UNet forward call per step (higher batch size). If an implementation uses two separate UNet calls for CFG, multiply counts below by ~2.
- For K candidates:
  - Full stability scoring (Qi et al.): \(\approx 2T\cdot K\) denoiser evaluations (T-step denoise + T-step DDIM inversion for each seed).
  - Partial scoring (ours): \(\approx 2k\cdot K + (T-k)\) denoiser evaluations (k-step denoise + k-step inversion per seed, then finish only the selected seed for the remaining \(T-k\) steps).

**Cheap baseline (no inversion):** to test whether inversion adds real signal beyond any early-step statistic, we include a direction-consistency score computed from the model’s predicted noise directions during early denoising:

\[
 d_k = \frac{1}{k}\sum_{j=0}^{k-1} \cos\big(\epsilon_\theta(z_{T-j}, t_{T-j}),\ \epsilon_\theta(z_{T-j-1}, t_{T-j-1})\big)
\]

This uses comparable early denoising compute but avoids inversion.

### Key Innovations

1. **A multi-fidelity view of inversion stability**: treat full denoise→invert stability as an expensive “oracle” score and propose a low-cost partial round-trip approximation.
2. **A decisive separation of proxy fidelity vs budget benefit**: we explicitly test (i) whether \(s_k\) predicts \(s_{full}\) within a fixed candidate pool and (ii) whether \(s_k\) improves compute-normalized selection quality.
3. **No extra training or external verifier for screening**: the selection rule uses only the diffusion model’s own denoiser evaluations and deterministic inversion.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) inference-time scaling for diffusion models via candidate generation and selection, (ii) methods that optimize or select initial noise/seed to improve text-to-image alignment and quality, and (iii) inversion-based analyses of diffusion trajectories.

A key axis is **how candidates are scored**:
- **External reward / preference models** (e.g., CLIP-based or learned human preference models) typically require decoding candidates to pixel space and are often trained on clean images; they can be expensive or unreliable when applied at intermediate noisy timesteps.
- **Noise-aware reward models for early pruning** (e.g., TTSnap’s NARF) train reward models to score intermediate denoising states, enabling multi-stage pruning but requiring additional training and intermediate decoding.
- **Intrinsic signals** extracted during denoising (attention patterns, internal activations, reversibility/invertibility) can screen candidates earlier and cheaper without training extra verifiers.

Another axis is **whether the method requires training**:
- Training-free seed selection/optimization methods (e.g., InitNO, ADSS) modify inference-time computation but keep the diffusion model fixed.
- Training-based approaches (e.g., NPNet “golden noise”) amortize test-time cost by learning a noise prior, but require data and training.

### Related Papers

- **[Not All Noises Are Created Equally: Diffusion Noise Selection and Optimization](./references/Not-All-Noises-Are-Created-Equally-Diffusion-Noise-Selection-and-Optimization/meta/meta_info.txt)**: Introduces full denoise→invert noise inversion stability for noise selection/optimization; strong quality gains but high compute.
- **[TTSnap: Test-Time Scaling of Diffusion Models via Noise-Aware Pruning](./references/TTSnap-Test-Time-Scaling-of-Diffusion-Models-via-Noise-Aware-Pruning/meta/meta_info.txt)**: Trains noise-aware reward models (NARF) via self-distillation to prune candidate trajectories at intermediate timesteps, enabling best-of-N scaling with fewer full denoising runs.
- **[Golden Noise for Diffusion Models: A Learning Framework](./references/Golden-Noise-for-Diffusion-Models-A-Learning-Framework/meta/meta_info.txt)**: Learns a network (NPNet) to map random noise to “golden noise” with low inference overhead but requires dataset construction and training.
- **[Good Seed Makes a Good Crop: Discovering Secret Seeds in Text-to-Image Diffusion Models](./references/Good-Seed-Makes-a-Good-Crop-Discovering-Secret-Seeds-in-Text-to-Image-Diffusion-Models/meta/meta_info.txt)**: Empirical study showing seeds have systematic effects; motivates seed selection as a first-class control.
- **[InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization](./references/InitNO-Boosting-Text-to-Image-Diffusion-Models-via-Initial-Noise-Optimization/meta/meta_info.txt)**: Training-free initial-noise optimization using attention-derived validity scores; improves prompt fidelity but adds iterative optimization cost.
- **[ADSS: Boosting Text-to-Image Diffusion Models via Attention-Driven Seed Selection](./references/ADSS-Attention-Driven-Seed-Selection/meta/meta_info.txt)**: Training-free seed ranking via early cross-attention concentration to core tokens.
- **[Diffusion Rejection Sampling](./references/Diffusion-Rejection-Sampling/meta/meta_info.txt)**: Uses a learned time-dependent discriminator to reject/redo samples at intermediate timesteps; improves FID but adds 2–3× NFE and requires discriminator training.
- **[Zigzag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection](./references/Zigzag-Diffusion-Sampling-Diffusion-Models-Can-Self-Improve-via-Self-Reflection/meta/meta_info.txt)**: Alternates denoising and inversion steps to improve a single sample (self-reflection); highlights usefulness of denoise↔invert operations but does not address seed ranking.
- **[Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing (ELECT)](./references/Early-Timestep-Zero-Shot-Candidate-Selection-for-Instruction-Guided-Image-Editing/meta/meta_info.txt)**: Early-timestep candidate selection for diffusion-based editing using a **Tweedie posterior-mean estimate** (an approximate denoised image computed from a noisy latent and the model’s predicted noise) plus a background inconsistency score; requires a source image and is not directly a text-to-image seed-ranking method.
- **[Stable Diffusion](https://arxiv.org/abs/2112.10752)**: Latent diffusion backbone used by many seed-selection methods.
- **[SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)**: Stronger latent diffusion backbone (relevant base model).
- **[Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042)**: Distillation approach underlying SDXL-turbo; affects inversion/trajectory dynamics.
- **[Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)**: Deterministic sampling procedure for diffusion models; also enables a practical approximate inversion used in round-trip stability scoring.
- **[DPM-Solver++](https://arxiv.org/abs/2211.01095)**: Widely used fast ODE solver sampler for diffusion models; relevant when discussing deterministic sampling.
- **[Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)**: Introduces DDIM inversion formulation used in many inversion-based analyses.
- **[Null-text inversion for editing real images using guided diffusion models](https://arxiv.org/abs/2211.09794)**: Inversion method showing inversion quality and guidance interact.
- **[Attend-and-Excite](https://arxiv.org/abs/2304.11846)**: Training-free optimization over attention signals to improve compositional prompt fidelity.
- **[A-STAR: Test-time Attention Segregation and Retention for Text-to-image Synthesis](https://arxiv.org/abs/2306.14544)**: Training-free test-time attention segregation/retention to improve compositional prompts; representative of attention-based inference-time interventions.
- **[Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis](https://arxiv.org/abs/2212.05032)**: Incorporates linguistic structure into guidance via attention manipulation to improve compositional generation; another training-free compositional-control method.
- **[Repaint](https://arxiv.org/abs/2201.09865)**: Resampling-based editing method related to forward/backward diffusion cycles.
- **[ImageReward](https://arxiv.org/abs/2304.05977)**: Learned human preference model for text-to-image, used as an automated metric.
- **[Pick-a-Pic](https://arxiv.org/abs/2305.01569)**: Open dataset of user preferences and the basis for PickScore.
- **[HPS v2](https://arxiv.org/abs/2306.09341)**: Human Preference Score model used in multiple diffusion evaluation works.
- **[GenEval](https://arxiv.org/abs/2310.09612)**: Object-focused evaluation of prompt-image alignment (useful for alignment-focused slices like counting).

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Full round-trip stability | Score seeds by full denoise→invert consistency | Qi et al. 2024 | DrawBench, Pick-a-Pic; HPSv2/PickScore/ImageReward | ~2T denoiser evals per seed; expensive screening | 
| Noise-aware reward pruning | Train a reward model to score intermediate denoising states and prune candidates early | TTSnap 2025 | ImageReward / PickScore / HPS prompts; omega metric | Requires training noise-aware reward models and decoding intermediate images |
| Learned noise priors | Train a small model to generate better initial noise | Golden Noise / NPNet 2024 | Pick-a-Pic, DrawBench, GenEval | Requires dataset construction + training; model-specific |
| Attention-based screening | Use early attention maps to rank seeds | InitNO 2024, ADSS 2026 | compositional prompt sets, DrawBench/Pick-a-Pic | Requires access to attention internals; may be prompt-type specific |
| Rejection/refinement | Reject bad partial trajectories via learned critic/discriminator | DiffRS 2024 | CIFAR/ImageNet/SD | Requires discriminator training; 2–3× NFE |
| Denoise↔invert self-reflection | Use inversion steps to improve a single trajectory | Zigzag 2024 | Pick-a-Pic/DrawBench/GenEval | Improves one sample; not designed for seed ranking |
| Early candidate selection (editing) | Use early-timestep predictors + task-specific scores | ELECT 2025 | PIE-Bench / MagicBrush | Uses source image; not directly applicable to text-to-image |

### Closest Prior Work

- **Qi et al. (2024)**: Defines full inversion stability \(s_{full}=\cos(z_T, z_T')\) where \(z_T'\) is obtained by denoise-to-\(z_0\) and DDIM inversion back to \(z_T\). Provides measurable quality gains (e.g., Table 1) but requires a full round-trip per candidate seed.
- **TTSnap (Yu et al., 2025)**: Performs multi-stage pruning by scoring intermediate denoising estimates with **noise-aware reward models** trained via self-distillation (NARF). This reduces the number of full denoising runs but requires extra training and intermediate latent decoding for reward evaluation.
- **ADSS (ICLR 2026 submission)**: Uses early cross-attention concentration to core tokens to rank seeds. It is cheap but depends on attention map extraction and prompt token heuristics.
- **InitNO (2024)**: Optimizes initial noise using attention-derived losses; improves compositional fidelity but adds iterative optimization steps.

This proposal differs from **Qi et al.** by approximating their stability score with a truncated k-step round trip, and differs from **TTSnap** by providing a **training-free, latent-space** screening rule that does not require learning a noise-aware reward model or decoding intermediate images. We also explicitly test whether inversion adds value beyond a compute-matched intrinsic baseline (direction-consistency).

**Novelty Kill Search Summary:** Searched for combinations of “partial inversion stability diffusion”, “round-trip stability score diffusion seed selection”, “k-step denoise invert diffusion seed ranking”, and checked for overlap with known seed-selection methods including Qi et al. 2024, TTSnap 2025, Golden Noise 2024, InitNO 2024, ADSS (OpenReview LJVXhUOJWK), ELECT 2025, DiffRS 2024, and Zigzag 2024. No prior work explicitly proposing **k-step denoise→invert cosine stability as a proxy for full inversion-stability seed ranking** was found as of 2026-02-23.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Qi et al. 2024 (full stability) | Full denoise→invert stability scoring for seed selection | ~2T denoiser evals per seed | Use k≪T round trip | Captures most of stability signal with far less compute |
| TTSnap 2025 | Multi-stage pruning using noise-aware reward models (NARF) on intermediate estimates | Requires training a noise-aware reward model and decoding intermediate images | Training-free latent-space proxy | Avoids extra reward-model training and intermediate decoding overhead |
| ADSS 2026 | Attention-based early seed ranking | Requires attention internals + token heuristics | Use scheduler-level latent dynamics only | Works even when attention maps are not easily exposed; prompt-agnostic score |
| InitNO 2024 | Iterative noise optimization using attention scores | ~2×+ inference overhead and optimization instability | No optimization; only scoring + selection | More predictable compute and simpler implementation |
| Golden Noise 2024 | Train NPNet to generate better noise | Needs training data and training | Training-free | Immediate plug-and-play for any model with DDIM inversion |
| DiffRS 2024 | Discriminator-based rejection over timesteps | Requires discriminator training; 2–3× NFE | No discriminator | Uses intrinsic reversibility signal without training |

---

## Experiments

### Experimental Setup

**Task:** text-to-image generation seed selection.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| SDXL | ~2.6B (U-Net) | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 | Used with DDIM sampling |
| SDXL-turbo (optional) | distilled | https://huggingface.co/stabilityai/sdxl-turbo | Few-step model; inversion behavior may differ |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| N/A | Inference-only | - | - | - |

**Other Resources (if applicable):**
- Reward / metric models: PickScore, HPSv2, ImageReward (open-source checkpoints).

**Baseline Ladder (REQUIRED):**
- **No selection (single sample)**: standard SDXL sampling with one random seed.
- **Best-of-N (inference-time scaling) with reward-model selection**: generate N full samples and select the best by an automated reward/quality model (e.g., PickScore or HPSv2).
- **Compute-matched cheap intrinsic baseline**: early direction-consistency score \(d_k\) (no inversion).
- **Closest existing method**: full inversion-stability selection \(s_{full}\) (Qi et al., 2024).

**Resource Estimate (evidence-based)**:
- **UNet-step time reference**: Hugging Face reports SDXL at 1024×1024 on an A100 80GB takes **3.8s for 25 steps** (\(~0.15s\) per denoising step, as a rough upper bound; https://huggingface.co/blog/lcm_lora).
- **Planned verification compute** (default N=100 prompts, 3 runs over candidate pools):
  - **Exp A (proxy fidelity)**: compute \(s_{full}\) and \(s_k\) on K=32 seeds per prompt with T=10, k=2 → \(\approx 3\times 100\times 32\times (2T+2k) \approx 230{,}000\) UNet-step evaluations.
  - **Exp B (compute-matched quality at B=80 UNet steps per prompt; ≈ best-of-8 with T=10)**: \(\approx 3\times 100\times 80 = 24{,}000\) UNet-step evaluations.
  - Total UNet-step evaluations \(\lesssim 260{,}000\) → \(\lesssim 11\) GPU-hours at 0.15s/step, plus reward-model scoring overhead.
- **Compute budget**: **≤ 30 GPU-hours total** (conservative, includes reward-model scoring and overhead).
- **GPU memory**: fits on 1×A100 80GB; score many seeds via micro-batching (e.g., 4–8 seeds per forward) to avoid OOM.
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Pick-a-Pic prompts | User-written prompts for text-to-image (T2I) preference learning | PickScore (CLIP-based preference predictor), HPSv2 (human-preference reward model), ImageReward (human preference RM) | test subset (e.g., first 100 prompts) | https://huggingface.co/datasets/pickapic/pickapic_v2 | PickScore: https://github.com/yuvalkirstain/PickScore ; HPSv2: https://github.com/tgxs002/HPSv2 ; ImageReward: https://github.com/zai-org/ImageReward |
| DrawBench prompts (optional) | 200 diverse prompts designed to stress compositionality, counting, spatial relations, rare words, and text rendering | PickScore, HPSv2, ImageReward | full prompt list | https://huggingface.co/datasets/sayakpaul/drawbench | same as above |

### Main Results

#### Comparability Rules (CRITICAL)

All methods below must use:
- the same base model and scheduler (DDIM),
- the same number of denoising steps T,
- the same prompt set,
- the same candidate pool size K (when comparing selection rules),
- the same metric model checkpoints.

#### Results Table

| Method | Base Model | Benchmark | PickScore (mean±std) | HPSv2 (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Random seed (no selection) | SDXL | Pick-a-Pic | **TBD** | **TBD** | - | To be measured |
| Direction-consistency selection (d_k) | SDXL | Pick-a-Pic | **TBD** | **TBD** | - | Cheap intrinsic baseline |
| Full inversion-stability selection (s_full) | SDXL | Pick-a-Pic | **TBD** | **TBD** | Qi et al. 2024 (method) | Needs re-run in our exact setting (Qi et al. report SDXL-turbo numbers in Table 1) |
| **Ours: partial round-trip stability (s_k)** | SDXL | Pick-a-Pic | **TBD** | **TBD** | - | k≪T |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (full) | k-step denoise + k-step inversion, cosine similarity | Best compute-normalized quality |
| w/o inversion (direction-consistency) | Replace s_k with d_k | If similar, inversion is unnecessary |
| Similarity metric swap | cosine → L2/MSE | If cosine saturates, L2 may be more discriminative |

### Experimental Rigor

**Variance & Reproducibility:**
- Use ≥3 independent runs over candidate-pool sampling (different random draws of K seeds per prompt), report mean±std over these runs.
- Fix prompt list and metric model checkpoints.

**Confounders & controls:**
- **“More candidates wins” confound**: separate a fixed-pool proxy-fidelity experiment (same K, compare s_k vs s_full) from a compute-budget experiment (same UNet-step budget, varying K).
- **Metric mismatch**: report both correlation to s_full and downstream reward; refute if s_k does not predict s_full even if it sometimes improves reward.

---

## Success Criteria

**Hypothesis**:
- Partial round-trip stability \(s_k\) (k small) has high rank correlation with full stability \(s_{full}\) within a fixed candidate pool, and selecting by \(s_k\) yields nearly the same reward as selecting by \(s_{full}\).

**Decision Rule**:
- **Proceed** if:
  1) Pooled Spearman correlation between \(s_k\) and \(s_{full}\) is ≥ 0.8 *and* overlap@1 between argmax selections is ≥ 0.5, and
  2) Under at least one compute budget point, \(s_k\)-selection achieves ≥90% of the reward gain of \(s_{full}\)-selection *and* beats the direction-consistency baseline \(d_k\) by a margin outside std across ≥3 runs.
- **Pivot** if correlation is borderline (0.5–0.8) but selection regret is low; try larger k or mid-timestep round trips.
- **Refute** if correlation < 0.5 or overlap@1 < 0.3, or if \(s_k\) does not beat the direction-consistency baseline.

---

## Impact Statement

If successful, this provides a simple, training-free way to make inversion-stability seed selection usable at practical budgets, enabling better quality under fixed inference-time compute for text-to-image generation pipelines.

---

## References

- [Not All Noises Are Created Equally: Diffusion Noise Selection and Optimization](./references/Not-All-Noises-Are-Created-Equally-Diffusion-Noise-Selection-and-Optimization/meta/meta_info.txt) - Qi et al., 2024
- [TTSnap: Test-Time Scaling of Diffusion Models via Noise-Aware Pruning](./references/TTSnap-Test-Time-Scaling-of-Diffusion-Models-via-Noise-Aware-Pruning/meta/meta_info.txt) - Yu et al., 2025
- [Golden Noise for Diffusion Models: A Learning Framework](./references/Golden-Noise-for-Diffusion-Models-A-Learning-Framework/meta/meta_info.txt) - Zhou et al., 2024
- [InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization](./references/InitNO-Boosting-Text-to-Image-Diffusion-Models-via-Initial-Noise-Optimization/meta/meta_info.txt) - Guo et al., 2024
- [ADSS: Attention-Driven Seed Selection](./references/ADSS-Attention-Driven-Seed-Selection/meta/meta_info.txt) - Zhang et al., 2026 (submission)
- [Diffusion Rejection Sampling](./references/Diffusion-Rejection-Sampling/meta/meta_info.txt) - Na et al., 2024
- [Zigzag Diffusion Sampling](./references/Zigzag-Diffusion-Sampling-Diffusion-Models-Can-Self-Improve-via-Self-Reflection/meta/meta_info.txt) - Bai et al., 2024
- [ELECT](./references/Early-Timestep-Zero-Shot-Candidate-Selection-for-Instruction-Guided-Image-Editing/meta/meta_info.txt) - Kim et al., 2025
- [Good Seed Makes a Good Crop](./references/Good-Seed-Makes-a-Good-Crop-Discovering-Secret-Seeds-in-Text-to-Image-Diffusion-Models/meta/meta_info.txt) - Xu and Shi, 2024
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) - Rombach et al., 2022
- [SDXL](https://arxiv.org/abs/2307.01952) - Podell et al., 2023
- [DDIM](https://arxiv.org/abs/2010.02502) - Song et al., 2020
- [DPM-Solver++](https://arxiv.org/abs/2211.01095) - Lu et al., 2022
- [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) - Sauer et al., 2023
- [ImageReward](https://arxiv.org/abs/2304.05977) - Xu et al., 2023
- [Pick-a-Pic](https://arxiv.org/abs/2305.01569) - Kirstain et al., 2023
- [HPS v2](https://arxiv.org/abs/2306.09341) - Wu et al., 2023
- [GenEval](https://arxiv.org/abs/2310.09612) - Ghosh et al., 2023
- [A-STAR: Test-time Attention Segregation and Retention for Text-to-image Synthesis](https://arxiv.org/abs/2306.14544) - Agarwal et al., 2023
- [Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis](https://arxiv.org/abs/2212.05032) - Feng et al., 2022
- [Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626) - Hertz et al., 2022
- [Null-text inversion for editing real images using guided diffusion models](https://arxiv.org/abs/2211.09794) - Mokady et al., 2022
- [Attend-and-Excite](https://arxiv.org/abs/2304.11846) - Chefer et al., 2023
- [Repaint](https://arxiv.org/abs/2201.09865) - Lugmayr et al., 2022
