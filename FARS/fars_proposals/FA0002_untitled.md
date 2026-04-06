# untitled

# Adaptive SRE-Mass Cache Sizing for LoLA-Style Hybrid Linear Attention

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Transformers achieve strong in-context recall by caching all past key-value (KV) pairs and performing full-rank softmax attention over them, but the KV cache grows linearly with sequence length and becomes a dominant deployment bottleneck for long contexts. A large line of work seeks **subquadratic** alternatives—linear attention, state space models (SSMs), and hybrids—that keep inference cost bounded or linear in sequence length.

A promising recent direction is **post-training linearization / distillation**: rather than pretraining new efficient architectures from scratch, convert a strong pretrained transformer into a subquadratic model using a small amount of training. For example, **LoLCATs** linearizes Llama-3.1 into a hybrid **sliding-window softmax + linear attention** model using only ~40M tokens ([LoLCATs](./references/LoLCATs-On-Low-Rank-Linearizing-of-Large-Language-Models/meta/meta_info.txt)). However, pure linear attention components have limited capacity due to **interference / collisions** in their low-rank recurrent state.

**LoLA** addresses this by adding a small full-rank **sparse cache** on top of a sliding-window+linear attention base model ([LoLA](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt)). LoLA uses a **self-recall error (SRE)** score to identify KV pairs that the linear state cannot faithfully store; it keeps the top-\(\lambda\) highest-error pairs in a sparse cache while sending the rest into the linear state. This yields large gains on long-context benchmarks like RULER (e.g., S-NIAH-1@4K: 0.6%→97.4% in LoLA Table 1).

### The Problem

LoLA’s sparse cache size \(\lambda\) is a **fixed hyperparameter**. In practice, the “difficulty” of KV pairs (as measured by SRE) is not constant across time: some cache-update steps may have many “hard” pairs while others have few.

A fixed \(\lambda\) forces a conservative choice sized for worst-case difficulty, which can waste memory and compute on easier steps:

- **Memory**: the sparse cache stores full-rank keys/values.
- **Compute**: full-rank attention over the sparse cache adds per-step overhead.

This matters because the primary motivation for LoLA-like hybrids is to make long-context inference cheaper and more scalable; if we can reduce the typical sparse-cache size while preserving quality, it can directly translate to larger batch sizes or longer contexts under the same hardware budget.

Concretely, LoLA’s strongest 4K setting on the *extended* RULER suite uses a **small** sliding window \,\(\eta=128\) but a **much larger** sparse cache \,\(\lambda=768\) (Table 2 in **[LoLA](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/ASSOCIATIVE%20RECALL.md)**). This means the extra full-rank attention introduced by the sparse cache attends over **6× more tokens** than the local sliding window, so \(\lambda\) is a first-order knob for both **softmax compute** and **KV storage** during inference.

LoLCATs motivates hybrid linear attention largely through *serving* efficiency: on Llama 3 8B, it reports closing the zero-shot LM-Eval gap while achieving **3× higher throughput** and supporting **64× larger batch sizes** than FlashAttention-2 by maintaining a fixed-size KV state (see **[LoLCATs](./references/LoLCATs-On-Low-Rank-Linearizing-of-Large-Language-Models/sections/1%20Introduction.md)**). A large fixed sparse cache partially erodes this advantage by reintroducing a sizeable full-rank KV buffer; reducing the **average** sparse-cache size is therefore directly tied to whether LoLA-like hybrids preserve their headline efficiency gains in practice.

Importantly, this is not addressed by transformer KV-cache eviction work (H2O/Scissorhands/SnapKV/PyramidKV/etc.), because those methods typically decide *which* tokens to keep under a fixed budget for **softmax transformers**; LoLA’s sparse cache is a different object: it is an explicit “escape hatch” for KV pairs that would otherwise corrupt a **linear attention associative state**.

### Key Insight and Hypothesis

**Key insight**: LoLA already computes an SRE score for each candidate KV pair at each cache update, and its collision analysis notes that “difficult-to-memorize pairs are evident, illustrated as bright columns” in the SRE heatmap (Figure 3 discussion in **[LoLA](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/UNDERSTANDING%20MEMORY%20COLLISIONS.md)**). This suggests the per-update SRE distribution is often **concentrated** on a small subset of pairs. If the SRE scores are heavy-tailed within the eligible set \(E_t\), then we should be able to keep a *variable* number \(\lambda_t\) of pairs per update—large only when needed—while reducing the *average* sparse-cache size.

**Hypothesis**: A pre-registered, training-free rule that chooses \(\lambda_t\) to capture a fixed fraction of the total SRE “mass” at each update will maintain nearly the same accuracy as fixed-\(\lambda\) LoLA on long-context tasks, while reducing the steady-state average sparse-cache size by a substantial margin.

We could be wrong if SRE is **diffuse** (many moderately-hard pairs) on tasks requiring global dependence (e.g., Variable Tracking), causing \(\lambda_t\) to saturate at \(\lambda_{\max}\) most of the time and producing little/no savings.

---

## Proposed Approach

### Overview

We modify LoLA’s sparse-cache update rule to use a **variable sparse-cache size** \(\lambda_t\) per cache update, determined by a fixed (pre-registered) threshold on the cumulative SRE mass.

The method is **training-free** and intended as a drop-in replacement for LoLA’s cache update logic. It does not alter the base LoLCATs model weights.

### Method Details

#### Background: LoLA sparse caching
LoLA maintains three stores:
1. **Sliding window** (size \(\eta\)) with full-rank softmax attention.
2. **Sparse cache** \(G_t\) (max size \(\lambda\)) with full-rank softmax attention.
3. **Linear attention state** \((H_t, s_t)\) updated by accumulating “easy” pairs.

At each cache update step \(t\), define eligible pairs
\[ E_t = G_{t-1} \cup \{(k_i,v_i) \text{ evicted from sliding window}\}. \]
LoLA computes **self-recall error** for each eligible pair:
\[ \text{SRE}(k,v\mid H_t,s_t)=\left\| \frac{\phi(k)^\top H_t}{\phi(k)^\top s_t} - v \right\|_2^2. \]
Fixed-\(\lambda\) LoLA retains the top-\(\lambda\) pairs by SRE in the sparse cache.

#### Our adaptive cache sizing rule (SRE-adaptive-mass)
At each update step \(t\):
1. Compute \(s_i = \text{SRE}(k_i,v_i\mid H_t,s_t)\) for all \(i\in E_t\).
2. Sort indices by decreasing \(s_i\).
3. Choose the *smallest* prefix \(G_t\) such that
\[ \sum_{i\in G_t} s_i \ge p \cdot \sum_{j\in E_t} s_j. \]
4. Set \(\lambda_t=|G_t|\), clamped to \([\lambda_{\min},\lambda_{\max}]\).
5. Send the remaining pairs \(E_t \setminus G_t\) into the linear attention state update (same as LoLA).

We pre-register \(p=0.9\), \(\lambda_{\min}=32\), and \(\lambda_{\max}=768\) (matching LoLA’s fixed-\(\lambda\) baseline) and do **no tuning on the evaluation tasks**.

#### Ablation: attention-score adaptive sizing (attention-adaptive-mass)
To test whether “adaptive sizing” alone is sufficient (vs. SRE being the key signal), we run the same adaptive-mass rule but replace \(s_i\) with a query-dependent attention proxy score (H2O-like), e.g. the sum of within-chunk attention weights received by key \(k_i\).

### Key Innovations

- **Variable sparse-cache size for LoLA**: replaces fixed \(\lambda\) with \(\lambda_t\) chosen from the observed per-update difficulty distribution.
- **Mass-based selection**: uses the SRE mass fraction \(p\) rather than a hard threshold on SRE values, making the rule scale-free across layers/models.
- **Decisive attribution**: a minimal ablation swaps the ranking signal while keeping the adaptive rule identical.

---

## Related Work

### Field Overview

This proposal sits at the intersection of: (i) **efficient sequence models** (linear attention, SSMs, and hybrids), (ii) **post-training conversion / distillation** of pretrained transformers into subquadratic models, and (iii) **KV-cache compression/eviction** methods for transformer inference.

LoLA is closest to our setting: it explicitly targets the failure mode of **memory collisions** (interference) in linear attention’s low-rank state by routing a small set of problematic KV pairs to a full-rank sparse cache. Many transformer KV-cache methods optimize a different tradeoff: which tokens to keep when *truncating* or *compressing* a transformer’s KV cache. Those methods may use attention statistics, layer/head allocation, reconstruction objectives, or learned policies; but they do not directly address collision management inside a linear attention associative state.

### Related Papers

(Each paper: one sentence summary + relevance.)

- **[LoLA](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt)**: Introduces SRE-scored sparse caching on top of sliding-window+linear attention; our proposal modifies its fixed-\(\lambda\) cache update.
- **[LoLCATs](./references/LoLCATs-On-Low-Rank-Linearizing-of-Large-Language-Models/meta/meta_info.txt)**: Provides a practical base hybrid sliding-window+linear attention model distilled from Llama; we use it as the base model.
- **[RULER](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt)**: Long-context benchmark with retrieval + multi-hop tracing tasks; we evaluate on its Variable Tracking and retrieval tasks.
- **[Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods](https://arxiv.org/abs/2510.05901)**: Shows hybrid conversions can ignore their linear branch; motivates careful evaluation of hybrid mechanisms.
- **[STILL](https://arxiv.org/abs/2602.02180)**: Content-aware token routing for hybrid softmax/linear attention; related alternative way to allocate full-rank compute.
- **[Hedgehog](https://arxiv.org/abs/2402.04347)**: Learnable linear attention feature maps trained to mimic softmax; relevant as an alternative linear attention backbone.
- **[Liger](https://arxiv.org/abs/2503.01496)**: Linearizes LLMs into gated recurrent structures without extra parameters; alternative base architecture.
- **[SUPRA](https://arxiv.org/abs/2405.06640)**: Converts transformers into linear recurrent models with additional uptraining; another post-training linearization baseline family.
- **[BASED](https://arxiv.org/abs/2402.18668)**: An efficient hybrid (Taylor linear attention + small sliding window) analyzed via recall–throughput tradeoffs; conceptually aligned with LoLCATs-style hybrids.
- **[Mamba](https://arxiv.org/abs/2312.00752)**: Selective SSM with linear-time inference; a major alternative to attention.
- **[Mamba-2 / SSD](https://arxiv.org/abs/2405.21060)**: Shows SSM–attention duality and introduces efficient chunkwise algorithms; relevant for chunked implementations.
- **[RetNet](https://arxiv.org/abs/2307.08621)**: Retention-based architecture offering parallel training + recurrent inference.
- **[Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484)**: Makes DeltaNet-style online regression parallelizable; related to collision/interference control via different update rules.
- **[Gated Delta Networks](https://arxiv.org/abs/2412.06464)**: Combines gating and delta updates, improving long-context associative recall in linear models.
- **[Infini-attention](https://arxiv.org/abs/2404.07143)**: Adds compressive linear memory to transformers for effectively infinite context; related “compressive memory” perspective.
- **[H2O](https://arxiv.org/abs/2306.14048)**: Heavy-hitter eviction for transformer KV caches; provides a canonical attention-score-based token-importance baseline.
- **[Scissorhands](https://arxiv.org/abs/2305.17118)**: KV eviction based on persistence of importance; related inference-time token selection.
- **[StreamingLLM](https://arxiv.org/abs/2309.17453)**: Attention-sink based streaming KV cache; contrasts with our goal (retain accuracy on global-dependence tasks).
- **[SnapKV](https://arxiv.org/abs/2404.14469)**: Prefill-time attention profiling for KV compression; related query-aware selection in transformers.
- **[PyramidKV](https://arxiv.org/abs/2406.02069)**: Layer-wise varying KV budgets for transformer cache compression.
- **[FastGen](https://arxiv.org/abs/2310.01801)**: Per-head adaptive KV compression based on head patterns.
- **[SqueezeAttention](https://arxiv.org/abs/2404.04793)**: Allocates KV budgets across layers based on measured layer importance.
- **[Dynamic Memory Compression (DMC)](https://arxiv.org/abs/2403.09636)**: Learns variable-length transformer KV caches via append/accumulate decisions.
- **[Cross-Layer Attention (CLA)](https://arxiv.org/abs/2405.12981)**: Reduces KV cache by sharing KV across layers.
- **[CAOTE](https://arxiv.org/abs/2504.14051)**: Token eviction score based on attention output error (value-aware), strengthening attention-score baselines.
- **[LESS](https://arxiv.org/abs/2402.09398)**: Combines sparse eviction with a low-rank side cache to preserve information; conceptually adjacent hybridization.
- **[KVzip](https://arxiv.org/abs/2505.23416)**: Query-agnostic KV compression via context reconstruction; similar “calibration” spirit but transformer-specific.
- **[Compactor](https://arxiv.org/abs/2507.08143)**: Query-agnostic KV compression with context-calibrated compression rate; closest in spirit to “adaptive budget”, but for transformers not LoLA.
- **[Expected Attention](https://arxiv.org/abs/2510.00636)**: Predicts future attention distribution to compress KV cache without materializing attention matrices.
- **[Token Sparse Attention](https://arxiv.org/abs/2602.03216)**: Reversible token-level sparsification compatible with FlashAttention.
- **[On the Limits of Learned Importance Scoring for KV Cache Compression](https://arxiv.org/abs/2601.14279)**: Negative result: learned query-agnostic token importance can fail vs simple heuristics, motivating our rule to rely on an architecture-native signal (SRE) instead.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Hybrid linear+softmax attention (post-training) | Keep local full-rank attention and approximate the rest with linear attention | LoLCATs, STILL, Liger | RULER, LongBench, LM Eval | Linear component capacity limits; component imbalance |
| Collision/interference mitigation in linear states | Identify “hard” pairs that corrupt low-rank states and treat them specially | LoLA, DeltaNet, Gated DeltaNet | RULER (NIAH, VT), MQAR | Hyperparameter sensitivity; greedy selection |
| Transformer KV eviction/compression | Reduce transformer KV cache by selecting/compressing tokens/heads/layers | H2O, Scissorhands, SnapKV, PyramidKV, SqueezeAttention, DMC, CAOTE, Compactor | LongBench, RULER, PPL | Often query-dependent; may fail in multi-query / reuse scenarios |
| Sparse+low-rank decomposition | Combine sparse retention with low-rank residual memory | LESS | LM, summarization | Requires training; approximation error |

### Closest Prior Work

1) **LoLA** ([OpenReview](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt))
- What it does: training-free sparse caching on top of sliding-window + linear attention; scores candidate KV pairs using SRE and keeps the top-\(\lambda\) in a sparse cache.
- Key limitation: sparse-cache size \(\lambda\) is fixed, even though per-update difficulty can vary.
- Why we differ: we keep LoLA’s SRE signal and cache mechanics but change the selection objective to a per-update **mass-capture** rule that yields a variable \(\lambda_t\).

2) **Compactor** ([arXiv:2507.08143](https://arxiv.org/abs/2507.08143))
- What it does: query-agnostic transformer KV cache compression with a context-calibrated compression rate.
- Key limitation: transformer-specific; does not address collision control in linear attention states.
- Why we differ: we adapt a budget inside LoLA’s sparse cache, using an architecture-native collision signal (SRE).

3) **DMC** ([arXiv:2403.09636](https://arxiv.org/abs/2403.09636))
- What it does: learns variable-length transformer KV caches via append/accumulate decisions.
- Key limitation: requires additional training and changes model behavior; transformer KV cache only.
- Why we differ: training-free, and targets a different cache (LoLA sparse cache) and failure mode (linear-state interference).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LoLA | Fixed-\(\lambda\) SRE-based sparse cache for linear attention | Over-provisions cache on easy updates | Replace fixed \(\lambda\) with per-update \(\lambda_t\) from SRE mass | Keeps enough “hard” pairs when needed, shrinks cache otherwise |
| Compactor | Chooses a compression rate for transformer KV cache (query-agnostic) | Not about linear-state collisions | Use SRE (collision metric) and adapt budget at each update | Architecture-native difficulty signal may yield reliable savings |
| H2O / Scissorhands | Evict tokens based on attention scores/persistence | Tokens are discarded (not stored in low-rank state) | Use attention-score proxy only as ablation | Expect weaker because it ignores linear-state collision behavior |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| LoLCATs Llama-3.1 | 8B | https://huggingface.co/hazyresearch/lolcats-llama-3.1-8b-distill | Base hybrid sliding-window+linear attention model |
| LoLA augmentation | 8B | ./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt | LoLA itself is training-free; implement from paper pseudocode |

**Training Data (if applicable):**
- No training data needed — inference-only modification of LoLA cache update.

**Other Resources:**
- RULER benchmark scripts: https://github.com/hsiehjackson/RULER
- LM Evaluation Harness (RULER task group): https://github.com/EleutherAI/lm-evaluation-harness

**Resource Estimate**:
- **Compute budget**: ~10–50 GPU-hours total (3 conditions × 2 tasks at 4K; single A100 80GB, batch size tuned for throughput).
- **GPU memory**: 1×A100 80GB sufficient for 8B inference with 4K context.
- **API usage**: none.

**Pre-analysis (to validate the core assumption before the full run):**
- Run the fixed-\(\lambda\) LoLA baseline on a small subset (e.g., 50 examples each from VT and MQ at 4K) while logging per-update eligible-set SRE scores \(\{s_i\}_{i\in E_t}\).
- Compute and report a concentration summary (Lorenz curve + Gini coefficient) and the implied \(\lambda_t\) sequence for our fixed \(p\) under the SRE-adaptive-mass rule.
- **Early-stop rule** (to avoid wasting compute if the idea is wrong): if the implied steady-state \(\mathbb{E}_t[\lambda_t]\) is \(\ge 0.9\times\lambda_{\max}\) *or* the implied \(\lambda_t\) saturates at \(\lambda_{\max}\) for \(\ge 80\%\) of updates on both tasks, we stop and report a negative result (“SRE is not sufficiently concentrated for adaptive sizing to help in this setting”).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| RULER (VT, MQ) | Long-context tracing + retrieval tasks | Accuracy | test | https://github.com/hsiehjackson/RULER | lm-eval-harness `ruler` tasks |

### Main Results

We will report both **task accuracy** and **average sparse-cache size**.

#### Results Table (baseline numbers copied from LoLA paper)

| Method | Description | Settings | Results | Reference |
|---|---|---|---|---|
| LoLCATs-8B+ | Hybrid sliding-window softmax + linear attention with **extended** sliding window; no sparse cache | RULER (extended tasks), context=4K; LoLCATs-8B+ with \(\eta=896,\lambda=0\) | MQ=3.3, VT=0.7; avg \(|G_t|\)=0 | **[LoLA Table 2](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/ASSOCIATIVE%20RECALL.md)** |
| LoLA (fixed) | Adds SRE-scored sparse cache of fixed size \(\lambda\) on top of LoLCATs | RULER (extended tasks), context=4K; LoLA-8B with \(\eta=128,\lambda=768\) | MQ=67.6, VT=85.2; avg \(|G_t|\)=768 | **[LoLA Table 2](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/sections/ASSOCIATIVE%20RECALL.md)** |
| **Ours: SRE-adaptive-mass** | Variable sparse-cache size \(\lambda_t\) to capture a fixed fraction \(p\) of SRE mass per update | RULER (extended tasks), context=4K; \(\eta=128\); \(p=0.9\); \(\lambda_{\min}=32\); \(\lambda_{\max}=768\) | MQ=**TBD**, VT=**TBD**; steady-state avg \(|G_t|\)=**TBD** | This work (to be verified) |
| Ablation: attention-adaptive-mass | Same adaptive-mass rule but rank by an attention-score proxy instead of SRE | Same as above, but score = attention proxy | MQ=**TBD**, VT=**TBD**; steady-state avg \(|G_t|\)=**TBD** | This work (to be verified) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| SRE-adaptive-mass (full) | Variable \(\lambda_t\) from SRE mass | Best accuracy–cache tradeoff |
| attention-adaptive-mass | Same rule; rank by attention-score proxy | Worse tradeoff if SRE is essential |

### Analysis (Optional)

- **SRE concentration analysis**: report Lorenz curve / Gini coefficient of \(\{s_i\}_{i\in E_t}\) across updates to quantify “heavy-tailedness” and relate it to observed \(\lambda_t\) distribution.

---

## Success Criteria

**Criterion 1: Efficiency without quality loss**
- Hypothesis: SRE-adaptive-mass achieves comparable accuracy to fixed-\(\lambda\) LoLA on a diffuse-dependence task (VT) and a retrieval-heavy task (MQ) while using a smaller steady-state average sparse-cache size.
- Validation: With \(p,\lambda_{\min},\lambda_{\max}\) fixed *a priori*, SRE-adaptive-mass must (i) stay within **≤1 absolute accuracy point** of fixed-\(\lambda\) LoLA on **both** tasks, and (ii) reduce steady-state \(\mathbb{E}_t[|G_t|]\) by **≥25%**.

**Criterion 2: Signal matters**
- Hypothesis: Using a non-collision-aware signal (attention-score proxy) for the same adaptive sizing rule yields a worse accuracy–cache tradeoff.
- Validation: At the same \(\mathbb{E}_t[|G_t|]\) as SRE-adaptive-mass, attention-adaptive-mass shows a larger accuracy drop; or to match LoLA accuracy, it requires substantially larger \(\mathbb{E}_t[|G_t|]\) than SRE-adaptive-mass.

---

## Impact Statement

If successful, this work provides a simple, training-free way to reduce the typical cost of LoLA-style hybrids, making subquadratic long-context inference more practical (larger batch sizes or longer contexts at fixed hardware). It would also suggest a general design principle for hybrid linear attention: **allocate full-rank resources in proportion to measured collision risk, not by a fixed budget**.

---

## References

- [LoLA: Low-Rank Linear Attention with Sparse Caching](./references/LOLA-LOW-RANK-LINEAR-ATTENTION-WITH-SPARSE-CACHING/meta/meta_info.txt) - McDermott et al., 2026 (under review)
- [LoLCATs: On Low-Rank Linearizing of Large Language Models](./references/LoLCATs-On-Low-Rank-Linearizing-of-Large-Language-Models/meta/meta_info.txt) - Zhang et al., 2024
- [RULER: What’s the Real Context Size of Your Long-Context Language Models?](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt) - Hsieh et al., 2024
- [Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores](https://arxiv.org/abs/2507.08143) - Chari & Van Durme, 2025
- [Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference](https://arxiv.org/abs/2403.09636) - Rae et al., 2024
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) - Zhang et al., 2023
- [Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression](https://arxiv.org/abs/2305.17118) - Liu et al., 2023
- [CAOTE: KV Cache Selection for LLMs via Attention Output Error-Based Token Eviction](https://arxiv.org/abs/2504.14051) - Qualcomm AI Research, 2025
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) - Li et al., 2024
- [PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling](https://arxiv.org/abs/2406.02069) - Zhang et al., 2024
- [FastGen: Adaptive KV Cache Compression for LLMs](https://arxiv.org/abs/2310.01801) - Ge et al., 2023
- [SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimization](https://arxiv.org/abs/2404.04793) - Lang et al., 2024
- [Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](https://arxiv.org/abs/2602.03216) - Jo et al., 2026
- [On the Limits of Learned Importance Scoring for KV Cache Compression](https://arxiv.org/abs/2601.14279) - Steele, 2026