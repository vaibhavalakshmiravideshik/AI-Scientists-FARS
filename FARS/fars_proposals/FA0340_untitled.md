# untitled

# Projected Self-Recall Error for Faster LoLA Sparse-Cache Updates

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Serving large language models (LLMs) at long context lengths is often bottlenecked by attention costs. Standard transformer attention requires storing all past key/value (KV) vectors and computing attention over them, which makes memory and compute scale with context length.

Linear attention is an alternative that replaces the softmax attention kernel with a feature-map approximation (e.g., \(\exp(q^\top k) \approx \phi(q)^\top\phi(k)\)). This allows attention over arbitrarily long contexts with constant memory by summarizing the entire history into a fixed-size recurrent state \(H_t\). However, linear attention suffers from **memory collisions**: different keys interfere in the low-rank state, corrupting key\(\to\)value associations and causing failures on long-context retrieval and state-tracking tasks.

LoLA (Low-Rank Linear Attention with Sparse Caching) is a recent inference-time method that improves associative recall in hybrid linear-attention models by maintaining (i) a local sliding-window KV cache, (ii) a recurrent linear-attention state, and (iii) a **sparse global KV cache** that stores “difficult-to-remember” KV pairs in full rank \(\,\) [LoLA](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt). LoLA identifies difficult KV pairs using the **self-recall error (SRE)**: the squared \(\ell_2\) difference between a value \(v\) and the value predicted by recalling \(v\) from the current linear state using its own key \(k\). On long-context evaluation with RULER (a synthetic benchmark suite of 13 tasks such as multi-query retrieval and variable tracking) [RULER](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt), LoLA yields large gains over a strong LoLCATs baseline [LoLCATs](./references/LoLCATs-On-Low-Rank-Linearizing-of-Large-Language-Models/meta/meta_info.txt).

### The Problem

LoLA’s sparse cache requires periodically **scoring** an eligible set of KV pairs with SRE and keeping the top-\(\lambda\) (largest-error) pairs in the sparse cache. In the LoLA paper’s chunkwise inference description, “increasing the chunk or sliding window size reduces the number of cache updates, thus decreasing the overhead costs from sparse caching” [LoLA Sliding Window](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/Sliding%20Window.md), implying that SRE scoring is a meaningful overhead in some regimes.

A key practical question is whether we can **reduce sparse-cache update cost** without changing LoLA’s attention computation or the underlying linear-attention kernel.

A naive approach (e.g., reparameterizing the feature map \(\phi\)) can change the effective attention kernel and harm language modeling quality. For example, scaling \(\phi(q)\) and \(\phi(k)\) by a shared diagonal matrix changes the effective kernel to \(\phi(q)^\top S^2 \phi(k)\), which is not semantics-preserving. This proposal focuses on an intervention that leaves the attention computation unchanged and only approximates the **ranking** used for cache selection.

### Key Insight and Hypothesis

LoLA’s SRE for a KV pair \((k,v)\) is a squared norm of a residual vector:
\[
\mathrm{SRE}(k,v\mid H_t,s_t)=\left\|\underbrace{\frac{\phi(k)^\top H_t}{\phi(k)^\top s_t}}_{\hat v\,(k)}-v\right\|_2^2.
\]
If the distribution of SRE values is heavy-tailed (a small number of KV pairs have much larger residuals due to collisions), then the **top-\(\lambda\) set** should be robust to small multiplicative perturbations of residual norms.

A Johnson–Lindenstrauss (JL) random projection approximately preserves \(\ell_2\) norms: for a residual vector \(e\in\mathbb{R}^{d_v}\), \(\|eR\|_2^2\approx\|e\|_2^2\) for a suitable random matrix \(R\in\mathbb{R}^{d_v\times r}\) with \(r\ll d_v\). Therefore, scoring KV pairs by a **projected residual norm** should preserve the top-\(\lambda\) selection well enough to maintain LoLA’s long-context accuracy.

**Hypothesis:** Maintaining an auxiliary projected linear-attention state \(H^{(R)}_t = H_t R\) and using **projected self-recall error**
\[
\mathrm{SRE}_R(k,v)=\left\|\frac{\phi(k)^\top H^{(R)}_t}{\phi(k)^\top s_t} - vR\right\|_2^2
\]
for cache selection will preserve most of LoLA’s accuracy gains on RULER while reducing cache-update time (and enabling a larger \(\lambda\) at fixed update cost).

We could be wrong if (i) SRE values are not well-separated at the top, so small distortions cause many rank inversions, or (ii) cache-update time is not a material portion of end-to-end inference time, making the practical impact small. Our experiments explicitly measure top-\(\lambda\) overlap and cache-update time share.

---

## Proposed Approach

### Overview

We propose a drop-in modification to LoLA’s cache-update step: replace full-dimensional SRE scoring with **projected SRE scoring** using a fixed random projection over the value dimension. Importantly, this does **not** change LoLA’s attention computation (Eq. 14 in the LoLA paper) and does not modify \(\phi\), queries, keys, or values used by the attention output. Only the cache selection rule changes.

### Method Details

**Baseline LoLA recap.** LoLA maintains a linear-attention state \(H_t\in\mathbb{R}^{D\times d_v}\) and normalization \(s_t\in\mathbb{R}^D\), plus a sliding-window KV cache (size \(\eta\)) and a sparse global KV cache (size \(\lambda\)). At each cache update (per token in generation or per chunk in prefill), it scores an eligible set \(E_t\) and keeps the top-\(\lambda\) pairs in the sparse cache \(G_t\) [LoLA Sliding Window](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/Sliding%20Window.md).

**Projected SRE.** Choose a projection matrix \(R\in\mathbb{R}^{d_v\times r}\) with \(r\ll d_v\). We will use either:
- Dense Gaussian \(R_{ij}\sim\mathcal{N}(0,1/r)\), or
- Achlioptas sparse \(\{-1,0,+1\}\) projection scaled by \(1/\sqrt{r}\)

Both are standard JL-style constructions.

Maintain an auxiliary projected state:
- \(H^{(R)}_t = H_t R\in\mathbb{R}^{D\times r}\)

When LoLA integrates a KV pair \((k,v)\) into the linear state, it updates:
- \(H_t \leftarrow H_t + \phi(k) v^\top\)
- \(H^{(R)}_t \leftarrow H^{(R)}_t + \phi(k) (vR)^\top\)
- \(s_t \leftarrow s_t + \phi(k)\)

For scoring a candidate \((k,v)\), compute:
- \(\alpha = \phi(k)^\top s_t\) (scalar)
- \(\hat v_R = (\phi(k)^\top H^{(R)}_t)/\alpha\in\mathbb{R}^r\)
- \(v_R = vR\in\mathbb{R}^r\)
- \(\mathrm{SRE}_R(k,v)=\|\hat v_R - v_R\|_2^2\)

**Complexity intuition.** Full SRE scoring requires computing \(\phi(k)^\top H_t\in\mathbb{R}^{d_v}\) per scored pair. Projected SRE scoring replaces this with \(\phi(k)^\top H^{(R)}_t\in\mathbb{R}^{r}\), reducing the dominant dot-product cost from \(O(D\,d_v)\) to \(O(D\,r)\) per scored pair. The extra memory is storing \(H^{(R)}_t\) (constant w.r.t. context length).

### Key Innovations

- **Semantics-preserving approximation of LoLA’s cache-selection metric:** we approximate the SRE norms by projecting the residual vector, without changing LoLA’s attention computation or feature map \(\phi\).
- **A practical compute/accuracy tradeoff knob \(r\):** \(r\) controls cache-update cost and selection fidelity, enabling an explicit Pareto curve for long-context accuracy vs. cache-update overhead.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) **subquadratic attention** (linear attention, state space models, and sparse attention), (ii) **KV cache management** for long-context inference, and (iii) **random projection / sketching** methods that preserve norms and rankings. Linear attention reduces attention complexity but suffers memory collisions due to low-rank state capacity, motivating hybrid methods that add local full attention or sparse global attention. Separately, sparse attention and KV-cache compression methods attempt to keep a limited set of tokens while preserving model quality. Random projections and sketches provide data-oblivious ways to approximate distances and norms in high dimensions; this proposal uses them to approximate LoLA’s cache-selection metric.

### Related Papers

- **[LoLA: Low-Rank Linear Attention with Sparse Caching](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt)**: Introduces SRE-based sparse caching for hybrid linear attention; our method approximates SRE for faster cache updates.
- **[LoLCATs: On Low-Rank Linearizing of Large Language Models](./references/LoLCATs-On-Low-Rank-Linearizing-of-Large-Language-Models/meta/meta_info.txt)**: Distills transformers into hybrid linear+sliding-window attention; serves as the base model for LoLA and for our experiments.
- **[RULER: What’s the Real Context Size of Your Long-Context Language Models?](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt)**: A synthetic long-context benchmark suite; we evaluate on its 4K setting.
- **[Linear Transformers](https://arxiv.org/abs/2006.16236)**: Formulates attention as linear in sequence length via kernel feature maps; foundational for linear attention.
- **[Performer / FAVOR+](https://arxiv.org/abs/2009.14794)**: Uses random features to approximate softmax attention; related in using randomized projections, but targets attention computation rather than cache selection.
- **[Random Feature Attention (RFA)](https://arxiv.org/abs/2103.02143)**: Another random-feature softmax approximation method; conceptually adjacent.
- **[Reformer](https://arxiv.org/abs/2001.04451)**: Uses LSH to sparsify attention via hashing; an alternative randomized approach to reduce attention cost.
- **[Longformer](https://arxiv.org/abs/2004.05150)**: Sliding-window sparse attention for long documents.
- **[BigBird](https://arxiv.org/abs/2007.14062)**: Sparse attention patterns with theoretical guarantees.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)**: IO-aware exact attention used in LoLA’s chunkwise implementation.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**: Further kernel improvements; relevant for fair runtime baselines.
- **[H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048)**: Selects “important” tokens for KV caching using attention-based heavy hitters; LoLA shows such metrics fail for tasks needing full-context state tracking.
- **[Loki: Low-rank Keys for Efficient Sparse Attention](https://arxiv.org/abs/2406.02528)**: Projects keys to speed sparse attention; conceptually similar but query-dependent and not SRE-based.
- **[Native Sparse Attention (NSA)](https://arxiv.org/abs/2402.16778)**: A modern sparse attention mechanism with learned/projected structures.
- **[StreamingLLM](https://arxiv.org/abs/2309.17453)**: KV cache retention via attention sinks; a different cache-selection heuristic.
- **[Get More with Less: Synthesizing Recurrence with KV Cache Compression](https://arxiv.org/abs/2402.09398)**: Combines recurrence with KV cache compression; related motivation.
- **[Simple Linear Attention Language Models (Based)](https://arxiv.org/abs/2402.03372)**: Studies recall/throughput tradeoffs for linear attention LMs.
- **[Mamba](https://arxiv.org/abs/2312.00752)**: State space model alternative to attention.
- **[RetNet](https://arxiv.org/abs/2307.08621)**: Retentive network with efficient recurrence and attention-like behavior.
- **[RWKV](https://arxiv.org/abs/2305.13048)**: Hybrid RNN/attention-like models for efficient inference.
- **[Atlas: Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/abs/2505.08243)**: Test-time memorization via learned associative maps; LoLA suggests such models may reduce cache needs.
- **[Johnson–Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)**: Theoretical foundation for norm-preserving random projections used in this proposal.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Linear attention via kernel features | Replace softmax kernel with \(\phi(q)^\top\phi(k)\) to make attention linear-time | Linear Transformers; Performer; RFA | LM eval; long-context retrieval | Limited memory capacity / collisions; approximation error |
| Distillation into subquadratic models | Convert pretrained transformers into linear/hybrid models via distillation / finetuning | LoLCATs; Mamba in the Llama | LM eval; long-context tasks | Often trained on shorter contexts; long-context generalization gaps |
| Sparse attention / token selection | Attend to a subset of tokens via patterns or data-dependent selection | Longformer; BigBird; Reformer; Loki; NSA | LongBench; L-Eval; RULER | Token-importance heuristics can fail for tasks requiring full context |
| KV cache management / compression | Keep/compress a subset of KV cache for inference efficiency | H2O; StreamingLLM; KV compression methods | Long-context QA; serving throughput | Heuristics may be brittle; can discard needed info |
| Collision-aware hybrid memory systems | Combine recurrence with local/global full attention to mitigate collisions | LoLA; Based; Atlas | RULER; Needle-in-haystack | Extra overhead for cache updates / gating / selection |

### Closest Prior Work

1. **LoLA** [LoLA](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt): Introduces SRE-based sparse caching and shows SRE is crucial via ablations; however, it computes SRE in the full value dimension, and does not study approximations of SRE scoring.
2. **LoLCATs** [LoLCATs](./references/LoLCATs-On-Low-Rank-Linearizing-of-Large-Language-Models/meta/meta_info.txt): Provides the base hybrid linear+sliding-window architecture that LoLA augments; it does not address collision-aware sparse caching.
3. **H2O heavy hitters** [H2O](https://arxiv.org/abs/2306.14048): Selects KV pairs using attention-weight-based metrics; LoLA argues such metrics are inappropriate when “unimportant” tokens must still be remembered via the linear state.
4. **Random-feature attention (Performer/RFA)** [Performer](https://arxiv.org/abs/2009.14794), [RFA](https://arxiv.org/abs/2103.02143): Uses randomized projections to approximate the softmax kernel itself; in contrast, we keep the attention computation fixed and approximate only LoLA’s cache-selection metric.

**Novelty Kill Search Summary:** On 2026-02-27, searched for combinations of (LoLA OR “self-recall error” OR SRE) with (random projection OR Johnson–Lindenstrauss OR sketch OR CountSketch), and checked local KB + web search. No prior work combining sketching/projection with LoLA’s SRE-based cache selection was found. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LoLA | SRE-based sparse caching for hybrid linear attention | Full-dimensional SRE scoring cost during cache updates | Approximate SRE norms by projected residuals using \(H^{(R)}\) | JL projection preserves norms; top-\(\lambda\) selection should be stable |
| LoLCATs | Distills transformers into hybrid linear+sliding-window attention | Collisions degrade long-context recall | Add LoLA-style sparse caching (as in LoLA) with cheaper scoring | Keeps LoLA’s recall gains with lower update overhead |
| H2O | Keeps heavy-hitter tokens by attention weights | Doesn’t handle collision-specific difficulty; fails on state tracking | Keep SRE objective but approximate efficiently | Preserves collision signal while reducing compute |
| Performer/RFA | Approximate softmax attention with random features | Changes attention computation / approximation properties | Keep attention fixed; approximate only cache scoring | Avoids kernel changes that can harm LM quality |

---

## Experiments

### Experimental Setup

**Goal:** Test whether projected SRE can replace full SRE for LoLA cache selection without losing most of LoLA’s long-context accuracy gains, and whether it reduces cache-update overhead.

**Implementation plan:** Use the open LoLCATs codebase [LoLCATs](./references/LoLCATs-On-Low-Rank-Linearizing-of-Large-Language-Models/meta/meta_info.txt) as the base hybrid attention model, and implement LoLA cache update equations from the LoLA paper [LoLA Sliding Window](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/Sliding%20Window.md). Add projected-state maintenance \(H^{(R)}\) and projected SRE scoring as described above.

**Baseline Ladder (REQUIRED):**
- **Prompting / protocol baseline (fixed):** RULER is evaluated with the official prompt templates and answer-prefix protocol; all methods use identical prompts and greedy decoding to remain comparable with published RULER/LoLA numbers.
- **Level 0 (simple):** LoLCATs-8B+ with a much larger sliding window (\(\eta=896,\lambda=0\)), matching LoLA’s total cache footprint reported in LoLA Table 2.
- **Level 1 (closest method):** Full LoLA with SRE scoring (\(\eta=128,\lambda=768\)).
- **Cost-matched baseline (critical control):** Full LoLA but with reduced sparse cache size \(\lambda=384\) (same \(\eta\)), to test whether “exact scoring on fewer cached pairs” is better than “approximate scoring on more cached pairs”.

**Inference-time scaling baseline note:** Best-of-\(N\) decoding is not part of the standard RULER protocol and would break comparability with published baselines. As a lightweight sanity check, we will optionally run best-of-8 sampling on a small subset (e.g., 50 examples of MQ at 4K) to verify that sampling does not close the gap between LoLCATs+ and LoLA-style methods.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| hazyresearch/lolcats-llama-3.1-8b-distill | 8B | https://huggingface.co/hazyresearch/lolcats-llama-3.1-8b-distill | Base hybrid linear+sliding-window model used by LoLA paper |

**Training Data (if applicable):**

No training data needed (inference-only modification).

**Other Resources (if applicable):**
- RULER benchmark code and data: https://github.com/hsiehjackson/RULER

**Resource Estimate**:
- **Compute budget**: Inference-only. RULER uses 13 tasks × 500 examples at 4K length by default [RULER Task configurations](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/sections/Task%20configurations.md). For 3 conditions (full LoLA, proj-SRE, and LoLA \(\lambda=384\)), total work is ~19,500 examples of 4K prefill + short decode. This should fit comfortably within the 768 GPU-hour budget; we estimate \(<100\) GPU-hours on A100s with batching.
- **Profiling first (required):** Before running full RULER, measure cache-update time share on a small subset (e.g., 100 examples of MQ @4K). If cache updates account for \(<10\%\) of wall-clock, we treat the main value proposition as “enable larger \(\lambda\) under a fixed update budget” rather than end-to-end speedup.
- **GPU memory**: Single A100 80GB should fit 8B inference; up to 8 GPUs can be used for throughput (RULER paper used 8×A100 for evaluation).
- **API usage**: None.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| RULER @ 4K | Synthetic long-context suite (13 tasks: retrieval, tracing, aggregation, QA) | Accuracy (recall-based exact-match) per task + average | test (generated) | https://github.com/hsiehjackson/RULER | Official repo |

**Primary metrics:**
- RULER average accuracy and key subtasks where LoLA shows large gains (MQ, VT, MV).

**Selection-fidelity metrics (for cache updates):**
- **Jaccard@\(\lambda\)** overlap between the sparse cache sets selected by full SRE vs projected SRE at each update, averaged over updates.
- **Precision@\(\lambda\)** of projected selection relative to full SRE selection.

**Efficiency metrics:**
- Cache-update wall-clock time per update (per chunk) and its fraction of total forward time.
- End-to-end throughput (tokens/sec) and time-to-first-token (TTFT) for 4K prefill.

### Main Results

#### Results Table

Published baselines (from LoLA Table 2) at 4K context:

| Method | Base Model | Benchmark | Metric (higher is better) | Source | Notes |
|--------|------------|-----------|---------------------------|--------|------|
| LoLCATs-8B+ (\(\eta=896,\lambda=0\)) | LoLCATs-8B | RULER (extended) @4K | Avg=6.7; MQ=3.3; VT=0.7 (accuracy %) | [LoLA Table 2](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/ASSOCIATIVE%20RECALL.md) | Published result (1 run) |
| LoLA-8B (\(\eta=128,\lambda=768\)) | LoLCATs-8B | RULER (extended) @4K | Avg=45.2; MQ=67.6; VT=85.2 (accuracy %) | [LoLA Table 2](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/ASSOCIATIVE%20RECALL.md) | Published result (1 run) |
| LoLA-8B (\(\eta=128,\lambda=384\)) | LoLCATs-8B | RULER (extended) @4K | Avg=**TBD**; MQ=**TBD**; VT=**TBD** | - | Needs re-run (not reported in paper) |
| **Ours: Proj-SRE LoLA (\(\eta=128,\lambda=768,r=64\))** | LoLCATs-8B | RULER (extended) @4K | Avg=**TBD**; MQ=**TBD**; VT=**TBD** (mean±std over 3 projection seeds) | - | To be verified |

#### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Proj-SRE (r=32) | Smaller projection dimension | Faster updates; possibly worse cache overlap / accuracy |
| Proj-SRE (r=128) | Larger projection dimension | Higher overlap; smaller speedup |

### Experimental Rigor

**Variance & Reproducibility:**
- The main source of stochasticity is the projection matrix \(R\). Run projected-SRE experiments with **3 random projection seeds** (e.g., `seeds=[0,1,2]`) and report mean ± std.
- Full LoLA baselines are deterministic under greedy decoding; run once and report as a single number.

**Validity & Controls:**
- **Compute confound**: Ensure all compared methods use the same \(\eta\) and chunking; treat \(\lambda=384\) as the explicit cost-matched baseline.
- **Implementation bug confound**: Include a sanity check that setting \(R\) to an identity-like projection (\(r=d_v\)) reproduces full SRE selections on a small batch.
- **Data leakage**: RULER is synthetic and generated by the benchmark; contamination risk is lower than web corpora, but we follow the official evaluation scripts.

---

## Success Criteria

**Hypothesis** (directional): Projected SRE preserves the identity of “difficult-to-remember” KV pairs (high top-\(\lambda\) overlap), so long-context accuracy remains close to full LoLA while cache-update overhead decreases.

**Decision Rule** (concrete):
- **Proceed/Continue** if:
  - Proj-SRE LoLA achieves **≥90% of LoLA’s lift** over LoLCATs-8B+ on RULER Avg @4K (i.e., \(\text{Avg}_{\text{proj}}-\text{Avg}_{\text{lolcats+}} \ge 0.9\cdot(\text{Avg}_{\text{lola}}-\text{Avg}_{\text{lolcats+}})\)), and
  - Proj-SRE reduces **cache-update time per update by ≥1.5×** vs full SRE (or enables \(\lambda=768\) at a cache-update cost comparable to full SRE with \(\lambda=384\)).
- **Pivot** if accuracy drops below the 90% lift threshold but cache-update gains are large: try larger \(r\) (e.g., 128) or a structured/sparse JL transform.
- **Refute** if Proj-SRE LoLA underperforms the cost-matched baseline (full SRE with \(\lambda=384\)) on RULER Avg, or if cache-update time is \(<10\%\) of end-to-end inference time such that improvements are negligible.

---

## Impact Statement

If projected SRE works, practitioners deploying hybrid linear-attention LLMs for long-context inference can either (i) reduce latency by lowering the cost of collision-aware cache updates, or (ii) allocate a larger sparse cache (higher \(\lambda\)) under the same compute budget, improving long-context recall and state tracking.

---

## References

- [LoLA: Low-Rank Linear Attention with Sparse Caching](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt)
- [LoLCATs: On Low-Rank Linearizing of Large Language Models](./references/LoLCATs-On-Low-Rank-Linearizing-of-Large-Language-Models/meta/meta_info.txt)
- [RULER: What’s the Real Context Size of Your Long-Context Language Models?](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt)
- [Linear Transformers](https://arxiv.org/abs/2006.16236)
- [Performer / FAVOR+](https://arxiv.org/abs/2009.14794)
- [Random Feature Attention](https://arxiv.org/abs/2103.02143)
- [Reformer](https://arxiv.org/abs/2001.04451)
- [Longformer](https://arxiv.org/abs/2004.05150)
- [BigBird](https://arxiv.org/abs/2007.14062)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048)
- [Loki](https://arxiv.org/abs/2406.02528)
- [Native Sparse Attention](https://arxiv.org/abs/2402.16778)
- [StreamingLLM](https://arxiv.org/abs/2309.17453)
- [Get More with Less: Synthesizing Recurrence with KV Cache Compression](https://arxiv.org/abs/2402.09398)
- [Simple Linear Attention Language Models (Based)](https://arxiv.org/abs/2402.03372)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [RetNet](https://arxiv.org/abs/2307.08621)
- [RWKV](https://arxiv.org/abs/2305.13048)
- [Atlas](https://arxiv.org/abs/2505.08243)
- [Johnson–Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
