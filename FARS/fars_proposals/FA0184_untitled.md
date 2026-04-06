# untitled

# Velocity-Forecast Sampling for Flow-Matching Token Heads in Autoregressive Image Generation

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Autoregressive (AR) image generation models produce an image by generating a sequence of image tokens one at a time, analogous to text generation in language models. Recent AR systems can achieve high image quality by generating **continuous** (real-valued) image tokens and then decoding them to pixels with a VAE-style tokenizer, instead of relying on discrete vector-quantized codebooks.

**[NextStep-1](./references/NextStep-1-Toward-Autoregressive-Image-Generation-with-Continuous-Tokens-at-Scale/meta/meta_info.txt)** is a representative example: it uses a large causal Transformer backbone (initialized from Qwen2.5-14B) to autoregressively produce image tokens, and a lightweight **flow matching (FM) head** to sample each continuous token. NextStep-1 reports strong text-to-image alignment (GenEval **0.63** overall; Table 2 in [“Image–Text Alignment”](./references/NextStep-1-Toward-Autoregressive-Image-Generation-with-Continuous-Tokens-at-Scale/sections/Image–Text%20Alignment..md); GenEval is a 553-prompt object-focused text-to-image alignment benchmark) but also highlights a practical bottleneck: **sequential decoding latency**.

In its per-token latency decomposition (Table 9 in [“Inference Latency of Sequential Decoding”](./references/NextStep-1-Toward-Autoregressive-Image-Generation-with-Continuous-Tokens-at-Scale/sections/Inference%20Latency%20of%20Sequential%20Decoding..md)), last-token latency on H100 is **7.20 ms** (LLM decoder) + **0.40 ms** (LM head) + **3.40 ms** (FM head) for a 256-token image — i.e., the FM head is ~31% of the per-token cost.

This breakdown matters for feasibility: if non-FM components cost ~7.6 ms/token, an FM-head-only speedup of *s* yields end-to-end speedup
\(\text{speedup}(s)=11.0/(7.6+3.4/s)\). Achieving ≥1.2× requires \(s\gtrsim2.2\), so any training-free method must cut FM-head work by **more than ~2×** (or be paired with backbone acceleration).

If AR image generation is to be competitive in interactive settings, it likely needs inference acceleration methods that do **not** require re-training a 14B backbone or distilling a new sampler. In diffusion and flow-matching models, many acceleration approaches require additional training (e.g., distillation to few-step sampling) or introduce approximate caching schemes. A training-free method that reduces redundant computation while keeping benchmark scores stable would be directly useful to practitioners deploying AR image generators.

### The Problem

In NextStep-1’s released implementation, the FM head samples each image token using an iterative solver with **K** sampling steps (e.g., `num_sampling_steps=28` in the provided eval code). Concretely, `FlowMatchingHead.sample()` repeatedly evaluates a neural network to predict a time-dependent velocity field and then updates the latent state using an ODE/SDE-style step rule (see `inference/nextstep_model.py` in the NextStep repository).

This creates a nested sequential loop:

- Outer loop: generate **T** image tokens autoregressively (e.g., 256 tokens for a 256×256 image).
- Inner loop (per token): run **K** FM sampling steps (e.g., K=28) that each call the FM-head network.

Even if the FM head is much smaller than the Transformer backbone, it is invoked **T×K** times per image. NextStep’s analysis indicates the FM head is a non-trivial fraction of per-token latency (Table 9). A naive way to reduce FM-head compute is to reduce K (fewer sampling steps), but this can degrade image quality and alignment.

Recent work **FlowCast** shows that flow-matching models often have locally smooth velocity fields, enabling “speculative” trajectory forecasting with verification to skip redundant steps without retraining. However, it is unclear whether this idea transfers to NextStep’s *patch-wise, per-token* FM head under strong **classifier-free guidance (CFG)** (a guidance scale that mixes conditional and unconditional predictions and can amplify non-smooth dynamics), and whether the FM head is large enough for “parallel verification” to yield wall-clock gains.

### Key Insight and Hypothesis

**Key insight**: In NextStep’s FM sampler, many consecutive solver steps may be redundant because the predicted velocity field changes slowly over short time intervals. If so, we can reuse a previously computed velocity for multiple solver steps and only refresh it when a cheap verification check indicates drift.

**Hypothesis**: A velocity-forecast sampler that reuses FM-head velocities for multiple steps with verification can reduce the number of FM-head network evaluations by ~2× (or more) on many tokens, yielding an **end-to-end speedup ≥1.2×** for 256×256 generation, while keeping **GenEval** within a small tolerance of the baseline.

Why this could be wrong:
- Under strong CFG, the velocity field may change quickly, causing frequent verification failures and little speedup.
- The FM head is relatively small (a 12-layer MLP), so batching verification computations may not provide meaningful wall-clock gains.

---

## Proposed Approach

### Overview

We propose **Velocity-Forecast Sampling (VFS)**: a training-free modification to NextStep’s `FlowMatchingHead.sample()` that replaces per-step velocity recomputation with **speculative segments**.

Each segment uses the current velocity prediction to advance several solver steps “for free” (no additional FM-head forward passes), then verifies whether the velocity field remained stable. If stable, we accept the segment. If not, we fall back to the original per-step sampler for that segment.

### Method Details

#### Background: NextStep’s flow-matching head sampler

In the public NextStep implementation, a single sampling step (from timestep \(t_i\) to \(t_j\)):
1. Evaluates the FM-head network to obtain a velocity \(v\).
2. Forms \(x_0 = x - v\,t\) and \(x_1 = x + v\,(1-t)\).
3. Updates \(x\) using either an ODE or SDE-style rule (the code supports `sde_type ∈ {"ode", "sde", "cps"}`).

The key property we exploit is that the expensive operation is the network call producing \(v(x,t)\).

#### VFS algorithm (segment-wise speculative sampling)

Let \(t_0 < t_1 < \dots < t_K\) be the solver timesteps used inside `FlowMatchingHead.sample()`.

We introduce two hyperparameters:
- **Segment length** \(r\) (e.g., 4): number of solver steps per speculative segment.
- **Relative velocity drift threshold** \(\varepsilon\): a scale-normalized drift metric.

Algorithm sketch (within a single token sampling call):

1. **Anchor velocity**: compute \(v_{m} = v(x_{m}, t_{m})\) once at the current segment start \(m\).
2. **Draft segment**: for \(k=m,\dots,m+r-1\), advance \(x\) from \(t_k\) to \(t_{k+1}\) using the **same update rule as NextStep**, but with the velocity fixed to \(v_m\) instead of recomputing \(v(x_k,t_k)\) at each step.
3. **Verify**: compute \(v_{m+r} = v(\hat{x}_{m+r}, t_{m+r})\) once at the segment end, and compute velocity drift using mean-squared error (MSE), following FlowCast:
   \[
   e = \mathrm{MSE}(v_{m+r}, v_m) = \frac{1}{d}\lVert v_{m+r} - v_m \rVert_2^2,
   \]
   where \(d\) is the velocity dimension.
4. **Accept / reject**:
   - If \(e \le \varepsilon\): accept the speculative segment, set \(x_{m+r} \leftarrow \hat{x}_{m+r}\), and reuse \(v_{m+r}\) as the next segment’s anchor.
   - If \(e > \varepsilon\): reject the segment and re-run the original NextStep sampler for these \(r\) steps (per-step velocity recomputation).

This is a speculative execution pattern similar to FlowCast’s “draft then verify” idea, but specialized to NextStep’s per-token FM head. The key intended benefit is a reduction in FM-head network calls from \(K\) to roughly \(K/r\) in segments that are accepted.

**Scope choice**: We apply VFS in `sde_type="ode"` mode (deterministic solver). This avoids confounding due to per-step stochastic noise when a rejected segment must be recomputed, and matches FlowCast’s theoretical setting (velocity smoothness along a deterministic trajectory).

### Key Innovations

1. **Intra-token speculative sampling for per-token FM heads**: Prior speculative flow methods focus on accelerating *single* full-image denoisers over ~50 steps. We apply speculation inside an autoregressive model’s per-token FM head, where the solver is invoked **T×K** times per image (nested loop).
2. **Non-trivial transfer regime**: Unlike full-image FM, the per-token FM head is (i) conditioned on an evolving AR Transformer hidden state, (ii) run on a much smaller latent (a single token / patch) with shorter effective trajectories, and (iii) typically uses strong CFG. These factors can make the velocity field *less smooth*, so it is not obvious that FlowCast-style reuse will have a high acceptance rate.
3. **Velocity-space verification aligned with FM**: We verify segments using velocity-field consistency (MSE between the anchor velocity and the velocity re-evaluated at the segment end), which directly probes whether the learned flow is locally constant.
4. **Decision-changing negative result is possible**: If VFS yields little speedup, it suggests FM-head acceleration is not the limiting factor and that inference work should target the AR backbone (e.g., multi-token prediction/speculative decoding for image-token LLMs).

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) autoregressive image generation with discrete or continuous tokenizations, (ii) flow matching / diffusion-style continuous samplers, and (iii) inference acceleration via speculative execution or adaptive computation.

On the modeling side, recent multimodal AR systems (e.g., Emu3, Show-o, Janus-Pro, NextStep-1) unify text and image generation by representing images as sequences of tokens and applying next-token prediction. NextStep-1 differs from discrete-token AR models by using **continuous** tokens sampled by a lightweight flow-matching head.

On the efficiency side, diffusion and flow-matching samplers have a large literature on reducing the number of solver steps (distillation to few-step sampling, improved ODE solvers, caching, and trajectory rectification). Separately, speculative decoding in LLMs accelerates autoregressive generation by drafting multiple tokens and verifying them. Recent work explores transferring speculative ideas to visual AR and to continuous distributions.

### Related Papers

- **[NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale](./references/NextStep-1-Toward-Autoregressive-Image-Generation-with-Continuous-Tokens-at-Scale/meta/meta_info.txt)**: AR image generator with a flow-matching head for continuous tokens; provides the target system and latency breakdown.
- **[FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](./references/FlowCast-Trajectory-Forecasting-for-Scalable-Zero-Cost-Speculative-Flow-Matching/meta/meta_info.txt)**: Training-free speculative acceleration for flow matching using constant-velocity forecasting and MSE verification.
- **[Continuous Speculative Decoding for Autoregressive Image Generation](./references/Continuous-Speculative-Decoding-for-Autoregressive-Image-Generation/meta/meta_info.txt)**: Extends speculative decoding to continuous AR image generation with diffusion distributions via draft+verify and acceptance-rejection sampling.
- **[LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding](./references/LANTERN-Accelerating-Visual-Autoregressive-Models-with-Relaxed-Speculative-Decoding/meta/meta_info.txt)**: Speculative decoding for discrete-token visual AR using latent-neighborhood acceptance relaxation.
- **[Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)**: Continuous-valued AR image generation (MAR); establishes continuous-token AR baseline family.
- **[Fluid: Scaling Autoregressive Text-to-Image Generative Models with Continuous Tokens](https://arxiv.org/abs/2410.13863)**: Continuous-token AR text-to-image generation; related architecture family.
- **[Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869)**: Unified multimodal next-token prediction with discrete visual tokens; strong AR baseline.
- **[Show-o: One Single Transformer to Unite Multimodal Understanding and Generation](https://arxiv.org/abs/2408.12528)**: Unified multimodal AR model; related to mixed text-image sequences.
- **[Janus-Pro: Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2501.17811)**: Scaling multimodal AR generation and understanding.
- **[Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039)**: Hybrid next-token prediction and diffusion for images.
- **[Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion](https://arxiv.org/abs/2407.01392)**: Connects next-token prediction and diffusion-style training; conceptual bridge.
- **[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)**: Foundational flow matching formulation and training objective.
- **[DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)**: Solver-based diffusion acceleration baseline family.
- **[DPM-Solver++](https://arxiv.org/abs/2211.01095)**: Improved fast solver family (commonly used as diffusion acceleration baseline).
- **[InstaFlow](https://arxiv.org/abs/2309.06380)**: Distillation-style few-step generation for flow/diffusion models.
- **[PeRFlow: Piecewise Rectified Flow](https://arxiv.org/abs/2405.07510)**: Trajectory rectification enabling few-step generation with retraining.
- **[TeaCache](https://arxiv.org/abs/2411.19108)**: Caching-based diffusion acceleration via reusing intermediate representations.
- **[Speculative Decoding](https://arxiv.org/abs/2211.17192)**: Draft-and-verify decoding for LLMs; foundational speculative framework.
- **[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)**: Multi-head drafting for LLM speculative decoding.
- **[EAGLE-2](https://arxiv.org/abs/2406.16858)**: Learned draft heads for speculative decoding; relevant to AR backbone acceleration.
- **[Better & Faster Large Language Models via Multi-Token Prediction](https://arxiv.org/abs/2404.19737)**: Multi-token prediction as an alternative AR acceleration path.
- **[LlamaGen](https://arxiv.org/abs/2406.06525)**: Discrete-token AR image generation model; reference point for visual speculative decoding work.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Continuous-token AR image generation | AR Transformer generates continuous image tokens; small diffusion/FM head samples each token | NextStep-1, MAR (2406.11838), Fluid (2410.13863) | GenEval, DPG-Bench, GenAI-Bench | Sequential decoding latency; per-token sampling overhead |
| Solver / step-count reduction | Use fewer solver steps or better integrators | DPM-Solver, DPM-Solver++ | FID/CLIP-based eval, GenEval | Often degrades quality or needs careful tuning |
| Retraining / distillation for few-step | Train a new fast sampler | InstaFlow, PeRFlow | FID/CLIPIQA, GenEval | Requires retraining; may reduce fidelity |
| Caching / reuse | Reuse intermediate states across steps | TeaCache | FID/CLIPIQA, GenEval | Approximation error; may be model-specific |
| Speculative execution for continuous flows | Forecast multiple future steps and verify | FlowCast | GenEval, CLIPIQA, VBench | Assumes velocity smoothness; requires parallel compute |
| Speculative decoding for AR tokens | Draft tokens and verify with target | Speculative Decoding, Medusa, EAGLE; LANTERN (vision) | Text benchmarks; COCO-FID for vision AR | Needs good draft model/heads; vision token ambiguity |

### Closest Prior Work

1. **FlowCast** proposes constant-velocity drafting for flow-matching trajectories and verifies drafts using an MSE threshold. It is evaluated on diffusion/FMs that generate a full image latent over tens of timesteps. **Our proposal differs** by targeting **NextStep’s per-token FM head**, which is invoked thousands of times per image and is conditioned on an AR Transformer state; this setting may violate FlowCast’s smoothness assumptions and changes the wall-clock tradeoffs because the denoiser is a small MLP.

2. **Continuous Speculative Decoding (CSpD)** provides a lossless speculative decoding algorithm for continuous AR image generation with diffusion distributions, using a separate draft model and acceptance-rejection sampling. **Our proposal differs** by requiring **no draft model** and focusing on accelerating the *inner sampler* (FM head) rather than the AR backbone, accepting an approximate sampler as long as benchmark metrics remain stable.

3. **LANTERN** accelerates discrete-token visual AR by relaxing acceptance using latent-neighborhood interchangeability, trading off distributional exactness. **Our proposal differs** by operating in **continuous token space** and using **velocity-field consistency** as the verification signal.

**Novelty Kill Search Summary:** We searched for prior work combining “FlowCast / velocity forecasting” with “NextStep (continuous-token AR) / patch-wise flow matching head” and for “speculative decoding flow matching head” across arXiv and GitHub keywords (e.g., “FlowCast NextStep”, “velocity reuse flow matching sampler”, “autoregressive image generation flow matching acceleration”, “continuous speculative decoding flow matching head”). As of **2026-02-20**, we did not find prior work explicitly applying FlowCast-style speculative velocity reuse inside per-token flow-matching heads of AR image generators.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| FlowCast | Forecasts FM trajectory with constant velocity + verification | Tested on full-image FM denoisers; unclear transfer to per-token FM head | Apply speculation inside NextStep’s per-token FM head | Removes redundant FM-head calls in a bottlenecked nested loop |
| CSpD | Lossless speculative decoding for continuous AR image gen with diffusion | Requires a draft model + complex resampling | No draft model; accelerate inner FM sampler | Lower engineering + compute overhead |
| LANTERN | Relaxed speculative decoding for discrete-token visual AR | Discrete-token assumption; distribution distortion | Continuous-token velocity consistency verification | Directly matches FM formulation |
| Step-count reduction | Fewer solver steps (e.g., 10-step) | Quality degradation | Speculation with verification | Better speed–quality tradeoff than fixed truncation |

---

## Experiments

### Experimental Setup

**Goal**: measure whether VFS improves the **speed–quality tradeoff** for NextStep-style continuous-token AR generation.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| stepfun-ai/NextStep-1.1 | ~14B backbone + FM head | https://huggingface.co/stepfun-ai/NextStep-1.1 | Apache-2.0 per repo; use bf16 on A100-80GB |
| stepfun-ai/NextStep-1-f8ch16-Tokenizer | VAE tokenizer | https://huggingface.co/stepfun-ai/NextStep-1.1 (linked assets) | Used by NextStep inference code |

**Training Data (if applicable):**

No training. Inference-time modification only.

**Baseline Ladder (REQUIRED):**
- **Baseline (standard sampling)**: NextStep sampler, `num_sampling_steps=28`, `sde_type="ode"`.
- **Simple efficiency baseline**: reduce solver steps, `num_sampling_steps=10`, `sde_type="ode"`.
- **Closest existing method family**: speculative velocity forecasting for FM (FlowCast) — our method is a specialization/transfer to NextStep’s per-token FM head.

**Implementation details**:
- Modify `FlowMatchingHead.sample()` in `inference/nextstep_model.py` to add VFS mode.
- Keep the timestep schedule identical to baseline.
- Use deterministic ODE update (`sde_type="ode"`) for all conditions.

**Seeds / variance plan**:
- Use `seeds=[42, 123, 456, 789, 2026]` for image generation. Report mean±std across seeds.
- If std is large enough that a 0.01 GenEval delta is not resolvable, switch decision rule to a relative threshold: "VFS gives back ≤50% of the 28-step→10-step GenEval drop" (compute-stable criterion).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| GenEval | Text-to-image alignment benchmark with 553 prompts, evaluated via object detection + attribute checks | GenEval Overall (↑) and per-category scores (↑) | Official prompt set | https://github.com/djghosh13/geneval | Official GenEval eval script |

**Efficiency metrics**:
- Wall-clock **seconds per image** for GenEval generation (batch size 1).
- (Optional) throughput at batch size 8 if it does not require engineering changes.
- **FM-head share sanity check**: instrument time spent in (i) AR backbone forward, (ii) FM head forward, (iii) other overhead (tokenizer / decoding) to confirm the Table 9 bottleneck proportions hold on the verification GPU.

### Main Results

#### Results Table

**Published reference points (from FlowCast, full-image FM models; GenEval overall ↑):**

| Model | Full steps | Full-step GenEval | FlowCast speedup (same quality) | Source |
|---|---:|---:|---:|---|
| BAGEL | 50 | 0.78 | 2.5× (FlowCast-50 keeps 0.78) | FlowCast Table 1 in [RESULTS](./references/FlowCast-Trajectory-Forecasting-for-Scalable-Zero-Cost-Speculative-Flow-Matching/sections/RESULTS.md) |
| FLUX | 50 | 0.65 | 2.4× (FlowCast-50 keeps 0.65) | FlowCast Table 1 in [RESULTS](./references/FlowCast-Trajectory-Forecasting-for-Scalable-Zero-Cost-Speculative-Flow-Matching/sections/RESULTS.md) |

**Target experiment table (to be filled by verification runs):**

| Method | Base Model | Benchmark | GenEval Overall (mean±std) | Sec / image (mean±std) | Extra diagnostics | Source | Notes |
|---|---|---|---:|---:|---|---|---|
| Baseline sampler (28-step ODE) | NextStep-1.1 | GenEval | **TBD** | **TBD** | - | this work | Must reproduce NextStep-1 paper’s ballpark (0.63 on NextStep-1; Table 2) |
| Step-reduced sampler (10-step ODE) | NextStep-1.1 | GenEval | **TBD** | **TBD** | - | this work | Simple speed baseline |
| **Ours: VFS (segment r=4, ε=0.07)** | NextStep-1.1 | GenEval | **TBD** | **TBD** | acceptance rate, avg FM calls/token | this work | Expect fewer FM net calls |

**Parameter justification**:
- We set \(\varepsilon=0.07\) as a starting point consistent with FlowCast’s GenEval study (they use MSE thresholds to trade off speed/quality; see FlowCast Table 1 + Appendix ablations).
- We fix segment length \(r=4\) as a conservative reuse window to target FM-head speedup \(s\gtrsim2.2\) (per Table 9 feasibility math).

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| VFS (full) | Forecast + verification + fallback | Best speed–quality tradeoff |
| Forecast-only (no verification) | Always accept segments | Worse GenEval or visible artifacts → verification is necessary |

### Experimental Rigor

**Confounders and controls**:
1. **Sampler mismatch**: All methods use `sde_type="ode"` and the same timestep schedule; only velocity reuse differs.
2. **Prompt sensitivity**: Use the full GenEval prompt set and report **per-category** scores (object occurrence, spatial positioning, color binding, counting).
3. **Caching / compilation effects**: Warm up once before timing; report timing after warmup for all methods.
4. **CFG robustness (diagnostic, not a main condition)**: on a fixed 50-prompt subset, run baseline vs VFS with `cfg ∈ {3.0, 7.5, 12.0}` (1 seed) to check whether acceptance collapses at high guidance.

**Mechanism diagnostics (required logging)**:
- Segment **acceptance rate** (% accepted segments) and average **FM-head forward calls per token**.
- Distribution of verification drift values `e` (to interpret whether ε is too strict/loose).

**Pre-registered sanity gates**:
- **Gate 0 (profiling upper bound)**: On the same 50-prompt subset, instrument baseline wall-clock time spent in (i) AR backbone forward, (ii) FM head forward, (iii) other overhead (tokenizer / decoding). Let `f_FM` be the FM-head time fraction. Theoretical max end-to-end speedup from *any* FM-only optimization is `S_max = 1/(1 - f_FM)`. If `S_max < 1.15`, **refute early** (FM head too small for meaningful system-level gains). Otherwise define the main-test speed target `S_target = min(1.20, 0.95 · S_max)`.
- Verify that the 28-step baseline reproduces a GenEval score in the same ballpark as NextStep-1’s published result (**0.63** on NextStep-1; Table 2), noting that we use NextStep-1.1 weights so exact equality is not expected. If baseline is off by >0.05 absolute, treat as an evaluation setup bug.

### Resource Estimate

- **Compute budget**: Expected ≤ 120 GPU-hours total.
  - Generation: 553 prompts × 3 main methods × 5 seeds = 8,295 images. If 3–10 sec/image on A100-80GB, this is ~7–23 GPU-hours of pure generation.
  - GenEval scoring: object detection + attribute checks; expected to be comparable to (or less than) generation.
  - Diagnostics: profiling + CFG sweep on a 50-prompt subset adds <2 GPU-hours.
- **GPU memory**: 1×A100 80GB should fit NextStep-1.1 in bf16.
- **API usage**: None.

---

## Success Criteria

**Hypothesis**: VFS reduces FM-head network evaluations enough to yield a measurable end-to-end speedup, while verification prevents quality regressions compared to fixed-step truncation.

**Decision Rule**:
- **Proceed**: VFS achieves **≥ S_target** speedup (defined in Gate 0; upper-bounded by 1.20×) vs the 28-step baseline **and** GenEval Overall drops by **≤0.01** absolute, across 5 seeds.
- **Pivot**: If speedup ≥ S_target but GenEval drop is 0.01–0.03, try a smaller segment length (r=2) without changing ε.
- **Refute**: If speedup < 1.10× (or <0.9·S_target) or GenEval drop >0.03, or if forecast-only performs similarly to VFS (indicating verification is unnecessary and the method reduces to an unchecked step-reduction heuristic).

---

## Impact Statement

If successful, this provides a training-free way to reduce inference latency for continuous-token autoregressive image generators like NextStep, improving usability in interactive generation and evaluation loops. If unsuccessful, it yields a clear negative result indicating that FM-head acceleration is not the main bottleneck and that future efficiency work should target the autoregressive Transformer decoder.

---

## References

- [NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale](./references/NextStep-1-Toward-Autoregressive-Image-Generation-with-Continuous-Tokens-at-Scale/meta/meta_info.txt)
- [FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](./references/FlowCast-Trajectory-Forecasting-for-Scalable-Zero-Cost-Speculative-Flow-Matching/meta/meta_info.txt)
- [Continuous Speculative Decoding for Autoregressive Image Generation](./references/Continuous-Speculative-Decoding-for-Autoregressive-Image-Generation/meta/meta_info.txt)
- [LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding](./references/LANTERN-Accelerating-Visual-Autoregressive-Models-with-Relaxed-Speculative-Decoding/meta/meta_info.txt)
- [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)
- [Fluid: Scaling Autoregressive Text-to-Image Generative Models with Continuous Tokens](https://arxiv.org/abs/2410.13863)
- [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869)
- [Show-o](https://arxiv.org/abs/2408.12528)
- [Janus-Pro](https://arxiv.org/abs/2501.17811)
- [Transfusion](https://arxiv.org/abs/2408.11039)
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [DPM-Solver](https://arxiv.org/abs/2206.00927)
- [DPM-Solver++](https://arxiv.org/abs/2211.01095)
- [InstaFlow](https://arxiv.org/abs/2309.06380)
- [PeRFlow](https://arxiv.org/abs/2405.07510)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [EAGLE-2](https://arxiv.org/abs/2406.16858)
- [Better & Faster Large Language Models via Multi-Token Prediction](https://arxiv.org/abs/2404.19737)
- [LlamaGen](https://arxiv.org/abs/2406.06525)
