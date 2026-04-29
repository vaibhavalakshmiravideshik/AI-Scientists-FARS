# untitled

# Post-hoc Top-p Expert Routing: Retrofitting Dynamic Expert Count onto Pretrained MoE LLMs

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Mixture-of-Experts (MoE) language models scale model capacity by keeping many expert feed-forward networks (FFNs) but activating only a small subset of experts per token. This makes it possible to deploy models with very large total parameter counts at the per-token compute cost of much smaller dense models (e.g., top-2 routing in Mixtral; top-8 routing in Step 3.5 Flash).

In practice, MoE deployment is still expensive: inference cost scales roughly linearly with the number of activated experts per token (plus dispatch/communication overhead). As a result, most deployed MoE checkpoints expose a single integer knob **top-k** that fixes the number of experts used for every token. This is a coarse control surface: some tokens may be “easy” and need fewer experts, while others may be “hard” and benefit from more.

Recent work shows multiple ways to make expert usage adaptive, but they typically require **training-time changes** (training an MoE with dynamic routing) or **test-time training / optimization** (updating router logits during inference). A practical open question is whether we can obtain useful test-time compute control for **already-trained, fixed-top-k MoEs** with a minimal, deployment-friendly modification.

### The Problem

**Fixed top-k MoE inference is a blunt instrument.** If we reduce k globally (e.g., from top-8 to top-4), we reduce cost but can hurt accuracy and perplexity. If we keep k high everywhere, we pay unnecessary compute on tokens where the router is already highly confident about its top expert(s).

Existing approaches that introduce more adaptive MoE behavior leave a gap for **inference-only retrofits**:

- **[Harder Tasks Need More Experts: Dynamic Routing in MoE Models](./references/Harder-Tasks-Need-More-Experts-Dynamic-Routing-in-MoE-Models/meta/meta_info.txt)** trains MoEs with confidence-threshold (**top-p**, i.e., nucleus-style selection: choose the smallest set of experts whose cumulative probability exceeds threshold p) routing and additional losses to encourage dynamic expert counts. It does not answer whether the same routing rule works *post hoc* on a model trained with fixed top-k.
- **[Ada-K Routing](./references/Ada-K-Routing-Boosting-the-Efficiency-of-MoE-based-LLMs/meta/meta_info.txt)** learns a lightweight per-token allocator (trained with RL) to choose the number of active experts; unlike our post-hoc method, it requires training an additional module.
- **[Rewiring Experts on the Fly](./references/Rewiring-Experts-on-the-Fly-Continuous-Rerouting-for-Better-Online-Adaptation-in-Mixture-of-Expert-models/meta/meta_info.txt)** improves MoE performance via test-time gradient updates to router logits using self-supervision on the current context. This requires extra backward passes and optimization steps during inference.
- **[Dynamic Experts Search (DES)](./references/DYNAMIC-EXPERTS-SEARCH-ENHANCING-REASONING-IN-MIXTURE-OF-EXPERTS-LLMS-AT-TEST-TIME/meta/meta_info.txt)** treats expert-count as a search dimension for test-time scaling, but relies on verifier scoring and search overhead.

We focus on the simplest remaining question: **does a fixed-top-k router’s probability distribution contain a usable confidence signal for variable expert-count routing at inference time, without retraining or optimization?**

### Key Insight and Hypothesis

**Key insight:** Even when an MoE is trained with a fixed top-k, the router is still optimized to rank experts for each token, and its softmax distribution may encode a proxy for how many experts are “plausible” for that token. If this proxy is meaningful, we can dynamically allocate experts by selecting the smallest number of experts whose cumulative router probability exceeds a threshold (top-p routing), thereby spending less compute on confident tokens while preserving compute on uncertain tokens.

**Hypothesis:** On a pretrained fixed-top-k MoE LLM (Qwen3-30B-A3B, default top-8), post-hoc top-p routing can match the average compute of a static reduced baseline (fixed top-4) while achieving better quality (lower perplexity and/or higher GSM8K (grade-school math word problems) accuracy), improving the quality–compute tradeoff curve without any retraining.

This hypothesis could fail for a simple reason: fixed-top-k training may not incentivize probability calibration, so router “confidence” (entropy / probability mass concentration) may be uninformative. In that case, post-hoc top-p should behave similarly to any other uninformed variable-k rule and fail to beat the static reduced baseline.

---

## Proposed Approach

### Overview

We propose a **post-hoc dynamic expert-count routing rule** for pretrained MoE LLMs that were trained with a fixed expert count per token.

Given router probabilities over experts for each token at an MoE layer, we select a variable number of experts per token using **top-p routing**: choose the smallest k such that the cumulative probability mass of the top-k experts exceeds a threshold p. We then compute only those k experts and combine their outputs using router weights renormalized over the selected experts.

We calibrate p **only to match a target average experts/token budget** on an unlabeled calibration slice, not to tune downstream accuracy.

### Method Details

**Setting.** Consider an MoE layer with E routed experts. For a token hidden state h, the router produces logits z \in \mathbb{R}^E and probabilities \pi = \mathrm{softmax}(z).

**Baseline fixed-top-k routing.** A standard MoE uses the k0 highest-probability experts I = \mathrm{TopK}(\pi, k0) and computes:

- dispatch to experts in I,
- expert outputs f_e(h),
- output y = \sum_{e \in I} w_e f_e(h), where w = \mathrm{softmax}(z_I).

**Post-hoc top-p routing (ours).** Let the experts be sorted by probability: \pi_{(1)} \ge \pi_{(2)} \ge \cdots.

We choose:

- k(h) = min{k : \sum_{j=1}^k \pi_{(j)} \ge p}, with 1 \le k(h) \le k0.
- I(h) = {(1),\ldots,(k(h))}.
- Renormalize weights over I(h): w_e = \pi_e / \sum_{e'\in I(h)} \pi_{e'}.
- Compute only experts in I(h) and combine as y = \sum_{e \in I(h)} w_e f_e(h).

**Budget calibration.** To compare fairly against a static reduced-compute baseline (e.g., fixed top-4), we pick p by monotone search on a small unlabeled text slice to satisfy:

- E_h[k(h)] \approx k_target (within ±1%), where k_target is the average experts/token under fixed top-k_target.

**Implementation note (for verifiers):** We can implement this efficiently by first computing the model’s existing top-k0 expert indices and probabilities, then taking a prefix of that top-k0 list based on cumulative probability mass.

### Key Innovations

- **Inference-only retrofit:** Unlike dynamic-routing MoE training methods, we apply variable expert counts *post hoc* to an existing checkpoint trained with fixed top-k.
- **Compute-matched comparison against the “obvious” baseline:** We explicitly compare dynamic routing to the strongest simple alternative: reducing top-k everywhere.
- **Decision-changing negative result is possible:** If router confidence is not usable post hoc, the result directly informs practitioners that dynamic expert-count routing likely needs training-time support or test-time optimization.

---

## Related Work

### Field Overview

MoE routing research spans (i) foundational sparse gating and load balancing, (ii) improved routing objectives and router–expert co-design, (iii) dynamic-capacity MoEs that vary computation during training/inference, (iv) test-time scaling methods that exploit MoE-specific degrees of freedom, and (v) test-time adaptation and post-hoc routing control.

Our proposal sits at the intersection of **test-time scaling** and **MoE routing**: we test whether a dynamic expert-count policy can be obtained “for free” from an existing fixed-top-k router, without verifiers (as in DES) and without test-time training (as in router-optimization methods).

### Related Papers

- **[Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538)**: Introduces modern sparse MoE gating and conditional computation.
- **[GShard](https://arxiv.org/abs/2006.16668)**: Scales MoE training with sharded experts and top-k routing.
- **[Switch Transformers](https://arxiv.org/abs/2101.03961)**: Popularizes top-1 routing with auxiliary load-balancing losses.
- **[GLaM](https://arxiv.org/abs/2112.06905)**: Large-scale MoE LLM showing strong quality/efficiency trade-offs.
- **[Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)**: Inverts routing so experts select tokens, improving load balancing.
- **[Mixtral 8x7B](https://arxiv.org/abs/2401.04088)**: A widely deployed open MoE with fixed top-2 routing.
- **[DBRX](https://arxiv.org/abs/2403.12320)**: An open MoE with larger expert counts and fixed top-k routing.
- **[Ada-K Routing](./references/Ada-K-Routing-Boosting-the-Efficiency-of-MoE-based-LLMs/meta/meta_info.txt)**: Trains a lightweight allocator module to choose per-token expert counts, reporting large FLOPs/speed gains; closest prior on dynamic expert allocation but not training-free.
- **[DeepSeek-V2](https://arxiv.org/abs/2405.04434)**: Modern MoE LLM with strong performance and routing/load-balancing design.
- **[DeepSeek-V3](https://arxiv.org/abs/2412.19437)**: Reports auxiliary-loss-free load balancing and MoE systems optimizations.
- **[Qwen1.5-MoE](https://arxiv.org/abs/2404.14219)**: Open MoE family documenting expert routing and deployment trade-offs.
- **[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)**: Describes Qwen3 models and training/evaluation context.
- **[Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)**: Frontier-level MoE agent model; highlights routing confidence monitoring.
- **[Harder Tasks Need More Experts](./references/Harder-Tasks-Need-More-Experts-Dynamic-Routing-in-MoE-Models/meta/meta_info.txt)**: Trains MoEs with confidence-threshold (top-p) routing and shows improved efficiency.
- **[Uni-MoE-2.0-Omni](https://arxiv.org/abs/2511.12609)**: Uses trained-in top-p routing + null experts for dynamic-capacity MoE.
- **[UniMoE-Audio](https://arxiv.org/abs/2510.13344)**: Uses trained-in top-p routing and null experts for dynamic compute in audio generation.
- **[LongCat-Flash](https://arxiv.org/abs/2509.01322)**: Uses zero-computation experts and PID-controlled expert bias to regulate active compute.
- **[MoE++](https://arxiv.org/abs/2406.06266)**: Introduces zero-computation experts to improve MoE efficiency.
- **[Shortcut-Connected Expert Parallelism](https://arxiv.org/abs/2406.16544)**: Overlaps MoE communication with compute to reduce latency.
- **[Dynamic Experts Search](./references/DYNAMIC-EXPERTS-SEARCH-ENHANCING-REASONING-IN-MIXTURE-OF-EXPERTS-LLMS-AT-TEST-TIME/meta/meta_info.txt)**: Treats expert-count as a test-time scaling dimension but relies on verifiers and search.
- **[Rewiring Experts on the Fly](./references/Rewiring-Experts-on-the-Fly-Continuous-Rerouting-for-Better-Online-Adaptation-in-Mixture-of-Expert-models/meta/meta_info.txt)**: Test-time router-logit optimization using self-supervision on context.
- **[Rollout Routing Replay (R3)](https://arxiv.org/abs/2510.11370)**: Stabilizes MoE RL by replaying inference-time routing masks during training.
- **[Not Eliminate but Aggregate: Post-Hoc Control over MoE to Address Shortcut Shifts](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00701/124836/Not-Eliminate-but-Aggregate-Post-Hoc-Control-over)**: Studies post-hoc mixture control for robustness under distribution shift.
- **[ConSol](https://arxiv.org/abs/2501.04674)**: Uses sequential probability ratio testing to allocate test-time samples adaptively (non-MoE, but related compute allocation framing).

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Fixed-top-k MoE routing | Use a fixed k experts per token | Sparsely-Gated MoE; GShard; Switch; Mixtral; DBRX | LM perplexity; downstream accuracy | Blunt compute control; may waste compute |
| Trained-in dynamic expert count | Train MoE so k varies by confidence / top-p | Harder Tasks Need More Experts; UniMoE-Audio; Uni-MoE-2.0-Omni | QA/reasoning suites; domain benchmarks | Requires retraining; may depend on auxiliary losses |
| Test-time routing optimization | Update routing weights/logits at inference | Rewiring Experts on the Fly; R2-T2 (test-time rerouting) | Code/math; multimodal | Extra inference overhead; implementation complexity |
| Test-time search over expert configs | Search expert-count/paths at inference | Dynamic Experts Search | Reasoning benchmarks | Needs verifiers/PRMs; search overhead |
| System-level compute control | Control active compute via architectural knobs | LongCat-Flash; MoE++ | Throughput/latency + task scores | Typically trained-in / architecture-specific |

### Closest Prior Work

1. **[Harder Tasks Need More Experts](./references/Harder-Tasks-Need-More-Experts-Dynamic-Routing-in-MoE-Models/meta/meta_info.txt)**: Introduces confidence-threshold (top-p) routing to dynamically vary the number of experts per token, and trains the MoE with additional losses so this dynamic policy is effective. **Limitation for our question:** it does not test whether top-p routing works as a post-hoc retrofit on a fixed-top-k checkpoint.

2. **[Rewiring Experts on the Fly](./references/Rewiring-Experts-on-the-Fly-Continuous-Rerouting-for-Better-Online-Adaptation-in-Mixture-of-Expert-models/meta/meta_info.txt)**: Improves MoE performance by optimizing additive router-logit offsets via test-time backprop on the current context. **Limitation for our question:** it requires extra optimization steps and does not target compute reduction via changing expert count.

3. **[Dynamic Experts Search](./references/DYNAMIC-EXPERTS-SEARCH-ENHANCING-REASONING-IN-MIXTURE-OF-EXPERTS-LLMS-AT-TEST-TIME/meta/meta_info.txt)**: Explores varying expert counts as a test-time scaling axis via verifier-scored search over trajectories. **Limitation for our question:** depends on verifiers and search; does not offer a minimal, inference-only routing rule.

4. **[Step 3.5 Flash](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt)**: Reports routing confidence monitoring as an MoE-specific stability signal during RL training. **Limitation for our question:** it does not investigate using routing confidence/entropy to control expert counts at inference.

**Novelty Kill Search Summary:** We searched for the exact combination “post-hoc / inference-only top-p routing for pretrained fixed-top-k MoE LLMs” using web queries including: “inference-only dynamic routing MoE top-p”, “post-hoc routing threshold mixture of experts”, “router entropy top-k inference mixture of experts”, and “top-p routing Mixtral inference”, and we searched local drafts for “top-p routing / nucleus routing / adaptive top-k”. As of 2026-02-18, we found trained-in dynamic routing (Huang et al., 2024) and test-time router optimization (Su et al., 2025), but no prior work that clearly demonstrates a **training-free post-hoc top-p retrofit** on a fixed-top-k trained MoE checkpoint.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Harder Tasks Need More Experts (2024) | Trains MoE with top-p routing + losses | Requires retraining; unclear if post-hoc works | Apply top-p post hoc to fixed-top-k checkpoint | If router confidence is usable, we get dynamic routing “for free” |
| Ada-K Routing (ICLR 2025) | Trains a lightweight allocator (RL) to choose k per token | Requires training an extra module | Remove all training; use router probabilities directly | If it works, provides a simpler deployment knob than allocator training |
| Rewiring Experts on the Fly (2025) | Test-time gradient updates to router logits | Extra optimization overhead; not compute-focused | No updates; only change expert-count rule | Lower engineering and runtime overhead |
| Dynamic Experts Search (2025) | Search over expert-count trajectories using verifiers | Needs verifiers and search overhead | Single-pass routing rule | Lower cost; applies broadly without extra models |
| LongCat-Flash (2025) | Trained-in dynamic compute + PID control | Architecture-specific; not a retrofit | Retrofit on existing checkpoint | Applicable to any MoE exposing router probabilities |

---

## Experiments

### Experimental Setup

**Core evaluation principle:** Compare three inference-only routing policies under the same base model and prompting, and report quality vs average activated experts/token.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen3-30B-A3B | 30.5B total / ~3.3B active (MoE) | https://huggingface.co/Qwen/Qwen3-30B-A3B | Default top-8 routing; should fit on 1×A100-80GB in bf16 or 2×A100 for speed |

**Baseline Ladder (REQUIRED):**
- **Static compute control (obvious baseline):** fixed top-4 routing everywhere.
- **Default inference:** fixed top-8 routing (model default).
- **Closest method family:** dynamic routing via top-p, but applied post hoc without training (ours).

**Routing Conditions (main 3):**
- **A (Default):** fixed top-8 routing.
- **B (Static-reduced):** fixed top-4 routing.
- **C (Ours / Post-hoc top-p):** variable k per token using cumulative probability threshold p (cap k≤8). Choose p to match B’s average experts/token on an unlabeled calibration slice.

**Sanity / instrumentation (not extra conditions):**
- Log per-layer and overall average experts/token, router entropy, and any expert-capacity overflow/drop rate (should be ~0 with sufficient capacity factor).

**Resource Estimate** (order-of-magnitude; verifier should measure precisely during implementation):
- **Compute budget**: Inference-only. Expect O(10–30) GPU-hours on 1×A100 for WikiText perplexity (on a small subset) + GSM8K pass@1 across 3 conditions; comfortably ≤768 GPU-hours.
- **GPU memory**: aim for 1×A100-80GB bf16; fall back to 2×A100 tensor parallel if needed.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|------------------|
| WikiText (subset) | Standard language modeling corpus for perplexity evaluation | Perplexity (exp(mean NLL)) | test | https://huggingface.co/datasets/Salesforce/wikitext | lm-eval-harness or custom NLL loop |
| GSM8K | Grade-school math word problems with numeric answers | Exact-match accuracy (answer parsing) | test | https://huggingface.co/datasets/openai/gsm8k | lm-eval-harness or custom parser |

### Main Results

| Method | Base Model | Benchmark | Metric 1 (mean±std) | Metric 2 (mean±std) | Source | Notes |
|--------|------------|-----------|----------------------|----------------------|--------|-------|
| A: Fixed top-8 | Qwen3-30B-A3B | WikiText | **TBD** | Avg experts/token: **8.0** | - | To be measured |
| B: Fixed top-4 | Qwen3-30B-A3B | WikiText | **TBD** | Avg experts/token: **4.0** | - | To be measured |
| C: Post-hoc top-p (compute-matched) | Qwen3-30B-A3B | WikiText | **TBD** | Avg experts/token: **≈4.0** | - | p chosen without labels |
| A: Fixed top-8 | Qwen3-30B-A3B | GSM8K | **TBD** | Avg experts/token: **8.0** | - | To be measured |
| B: Fixed top-4 | Qwen3-30B-A3B | GSM8K | **TBD** | Avg experts/token: **4.0** | - | To be measured |
| C: Post-hoc top-p (compute-matched) | Qwen3-30B-A3B | GSM8K | **TBD** | Avg experts/token: **≈4.0** | - | p chosen without labels |

### Experimental Rigor

- **Determinism / seeds**: Use greedy decoding for GSM8K to minimize variance. Perplexity evaluation is deterministic. If any nondeterminism remains (e.g., kernel nondeterminism), run 3 seeds and report mean±std.
- **Confounders and controls**:
  1. **Capacity overflow**: set capacity factor high enough that dropped-token rate is ~0 and report it.
  2. **Prompt sensitivity**: use identical prompts across A/B/C.
  3. **Compute mismatch**: choose p solely to match B’s average experts/token on an unlabeled calibration slice.

---

## Success Criteria

**Hypothesis** (directional): Post-hoc top-p routing will allocate fewer experts to confident tokens and more to uncertain tokens, yielding better quality than fixed top-4 at the same average expert budget.

**Decision Rule** (concrete):
- **Proceed** if, at matched compute (C vs B), C improves WikiText perplexity by ≥0.05 and/or GSM8K exact-match accuracy by ≥0.5 pp, and stays within ≤0.5 pp of the top-8 baseline on GSM8K.
- **Pivot** if top-4 is too close to top-8 on a small pilot slice (e.g., Δppl < 0.05): repeat with a more aggressive static baseline (top-2) while keeping the 3-condition structure.
- **Refute** if C does not outperform B at matched compute (within noise), suggesting router confidence is not a useful post-hoc signal for expert-count control.

---

## Impact Statement

If successful, this provides a training-free deployment knob for existing MoE checkpoints: practitioners can reduce inference cost by dynamically varying expert count per token rather than choosing a single fixed top-k. If unsuccessful, it provides a clear guideline that dynamic expert-count routing likely requires training-time support or test-time optimization, preventing wasted engineering effort on post-hoc heuristics.

---

## References

- [Step 3.5 Flash Open Frontier-Level Intelligence with 11B Active Parameters](./references/Step-3.5-Flash-Open-Frontier-Level-Intelligence-with-11B-Active-Parameters-StepFun-Team/meta/meta_info.txt) - StepFun Team, 2026
- [Harder Tasks Need More Experts: Dynamic Routing in MoE Models](./references/Harder-Tasks-Need-More-Experts-Dynamic-Routing-in-MoE-Models/meta/meta_info.txt) - Huang et al., 2024
- [Rewiring Experts on the Fly: Continuous Rerouting for Better Online Adaptation in Mixture-of-Expert models](./references/Rewiring-Experts-on-the-Fly-Continuous-Rerouting-for-Better-Online-Adaptation-in-Mixture-of-Expert-models/meta/meta_info.txt) - Su et al., 2025
- [Dynamic Experts Search: Enhancing Reasoning in Mixture-of-Experts LLMs at Test Time](./references/DYNAMIC-EXPERTS-SEARCH-ENHANCING-REASONING-IN-MIXTURE-OF-EXPERTS-LLMS-AT-TEST-TIME/meta/meta_info.txt) - 2025
