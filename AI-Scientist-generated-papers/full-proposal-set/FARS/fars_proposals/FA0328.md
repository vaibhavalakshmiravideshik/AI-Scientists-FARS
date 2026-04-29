# untitled

# Debiased One-Pass Attention Sorting for Long-Context QA via Per-Prompt Position-Bias Estimation

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Long-context language models are increasingly used in retrieval-augmented generation (RAG) and multi-document question answering (QA), where a model is given a question plus many retrieved documents and must answer by extracting a short span from the relevant document. A persistent limitation is **positional bias** in long contexts: models tend to over-attend to documents near the beginning or end of the prompt and under-use documents in the middle, even when those middle documents contain the answer.

A simple, training-free mitigation is **Attention Sorting**: run one decode step, measure how much attention the model assigns to each document, then reorder documents so high-attention documents appear near the end of the prompt and repeat this procedure a few times before generating the final answer (**[Attention Sorting](./references/Attention Sorting Combats Recency Bias In Long Context Language Models/meta/meta_info.txt)**). Attention Sorting often improves long-context QA accuracy, but it can require **2–5 re-sorting iterations** for models with strong recency bias, and each iteration requires an expensive long-context prefill pass.

### The Problem

For long-context inputs (e.g., 16K–32K+ tokens), the **prefill** stage (encoding the full prompt to produce the first output token) dominates latency and cost. Attention Sorting’s iterative design increases prefill passes from 1 (vanilla generation) to **k+1** (k sorting iterations plus the final answer generation). For k=5, this is a ~6× prefill multiplier.

A complementary line of work, **Found-in-the-Middle (FITM)**, models attention to a document as the sum of a document’s relevance and a position-dependent bias term, and uses dummy-document subtraction to estimate and remove that bias (**[FITM](./references/Found in the Middle Calibrating Positional Attention Bias Improves Long Context Utilization/meta/meta_info.txt)**). FITM improves mid-sequence utilization but explicitly notes an **extra O(K) forward-pass overhead** to calibrate bias at each position (“Computational overhead.” section).

The practical gap is a training-free method that (i) preserves Attention Sorting’s ability to identify a relevant document from many distractors using internal attention signals, but (ii) avoids repeated iterations by removing position bias **without extra forward passes**.

### Key Insight and Hypothesis

**Hypothesis.** Attention Sorting needs multiple iterations mainly because the **raw per-document first-token attention scores are confounded by position bias**. In early iterations, a relevant document that starts far from the end may not receive enough raw attention to be placed at the end in a single sort; repeated sorting gradually moves it toward high-attention positions.

We hypothesize that if we **explicitly estimate and subtract a position-bias curve** from per-document attention scores, then a **single** sorting pass will be sufficient to match the accuracy of multi-iteration (k=5) Attention Sorting on long-context extractive QA.

This can fail if (i) the bias curve is too content- or query-dependent to be estimated robustly, (ii) the “bias” and “relevance” signals are entangled so subtraction removes useful signal, or (iii) iterative sorting is correcting additional effects beyond position bias (e.g., attention dilution).

---

## Proposed Approach

### Overview

We propose **Debiased One-Pass Attention Sorting**, an inference-time procedure:

1. Run one decode step to obtain first-token attention weights (as in Attention Sorting).
2. Compute a **document-level attention mass** for each document.
3. Estimate a **per-prompt position-bias function** from (mostly distractor) documents.
4. Subtract this bias to produce a **debiased relevance score** per document.
5. Reorder documents once (highest debiased score last) and generate the final answer.

The method uses the same number of prefill passes as k=1 Attention Sorting (two total: one for attention measurement, one for the final answer), but aims to match k=5 accuracy.

### Method Details

#### Setup and notation

A prompt contains an instruction, a list of K documents \(d_1,\dots,d_K\) (each occupying a known token span \(S_i\)), and a question. Let \(y_1\) be the first generated answer token. For transformer layer \(\ell\) and head \(h\), the attention weights from \(y_1\) to input tokens form \(A^{\ell,h}\in\mathbb{R}^{1\times L}\) (query length 1, key length L).

#### Per-document attention mass (raw score)

We compute the raw attention mass for document \(i\):

\[
 a_i = \sum_{\ell=1}^{L_{\text{layers}}}\sum_{h=1}^{H}\sum_{t\in S_i} A^{\ell,h}_{t}.
\]

This is the same signal used by Attention Sorting (aggregated across layers/heads and summed over document tokens).

#### Per-prompt bias estimation (no extra forward passes)

Let \(p_i\in\{1,\dots,K\}\) be the document’s position index in the prompt. We estimate a bias curve \(\hat b(p)\) from the set of (position, attention) pairs \((p_i,a_i)\) within the *same prompt*, under the assumption that most documents are distractors.

A concrete robust estimator:

1. **Trim high-attention outliers**: remove the top \(\alpha\) fraction of documents by \(a_i\) (default \(\alpha=0.05\)).
2. **Bin by position**: split positions into B equal-width bins (default B=20) and compute the median \(a_i\) within each bin.
3. **Interpolate**: linearly interpolate (or fit a low-degree polynomial) over bins to obtain \(\hat b(p)\) for all positions.

We then define the **debiased document score**:

\[
 s_i = a_i - \hat b(p_i).
\]

(Optionally, divide \(a_i\) and \(s_i\) by document length to reduce length bias; we treat this as an ablation.)

#### One-pass debiased sorting

We reorder documents by \(s_i\) and place the highest-score document last (end of the document list), then generate the answer from the reordered prompt.

#### Why this should reduce iterations

If \(a_i = \text{rel}(d_i) + \text{bias}(p_i) + \epsilon\) (FITM’s modeling assumption) and most documents have near-zero relevance, then the trimmed medians across documents should approximate \(\text{bias}(p)\). Subtracting \(\hat b(p)\) should therefore increase the ranking quality of the truly relevant document, enabling a single sort to place it near the end.

### Key Innovations

1. **Mechanistic reframing of Attention Sorting**: iterative sorting is treated as an implicit bias-correction procedure; we test whether explicit correction can remove the need for iterations.
2. **Per-prompt bias estimation without extra forward passes**: the bias curve is estimated from the attention-vs-position pattern within the prompt, relying on the “many distractors” regime common in RAG.
3. **Decisive verification via rank-lift gate + compute-matched comparison**: we include an explicit Phase-0 gate and a prefill-pass compute accounting.

---

## Related Work

### Field Overview

Work on long-context LLMs has documented that accuracy often degrades when relevant information is placed in the middle of long inputs (“lost in the middle”). Approaches to mitigate this include (i) extending context windows via position-encoding modifications or additional long-context training, (ii) training-free attention interventions that reweight or recalibrate attention to reduce positional artifacts, and (iii) input-level strategies such as document reordering, compression, or permutation/aggregation.

Our proposal focuses on a specific operational bottleneck: **multi-pass inference-time document reordering** methods (e.g., iterative Attention Sorting) can be effective but are expensive because each pass requires a long-context prefill.

### Related Papers

- **[Attention Sorting](./references/Attention Sorting Combats Recency Bias In Long Context Language Models/meta/meta_info.txt)**: Introduces iterative document reordering by first-token attention to combat recency bias in long-context extractive QA.
- **[Lost in the Middle](https://arxiv.org/abs/2307.03172)**: Characterizes U-shaped positional sensitivity in long-context QA and motivates mitigation methods.
- **[Found in the Middle](./references/Found in the Middle Calibrating Positional Attention Bias Improves Long Context Utilization/meta/meta_info.txt)**: Models attention as relevance plus positional bias and calibrates attention via dummy-document subtraction (with O(K) overhead).
- **[Attention Basin / AttnRank](https://arxiv.org/abs/2508.05128)**: Profiles a model’s intrinsic attention distribution over structured items and reorders retrieved items to align relevance with high-attention positions.
- **[Permutation Self-Consistency](https://arxiv.org/abs/2310.07712)**: Mitigates positional bias in LLM listwise ranking by repeated input permutations and Kemeny-Young aggregation.
- **[Gold Panning](https://arxiv.org/abs/2510.09770)**: Exploits position bias as a diagnostic signal using calibrated position profiles to reduce the number of multi-document queries.
- **[LLM-RankFusion](https://arxiv.org/abs/2406.00231)**: Reduces order and transitive inconsistencies in LLM ranking using calibration and rank aggregation.
- **[In-Context Reranking (ICR)](https://arxiv.org/abs/2410.02642)**: Uses attention aggregation for efficient reranking and proposes a content-free calibration query to subtract intrinsic attention biases.
- **[ReAttn](https://arxiv.org/abs/2602.19969)**: Improves attention-based reranking via post-hoc IDF-style weighting and entropy-based regularization.
- **[Contrastive Retrieval Heads (CoRe)](https://arxiv.org/abs/2510.02219)**: Identifies attention heads that are most discriminative for retrieval-style reranking via a contrastive metric.
- **[Adaptive Repetition for Position Bias](https://arxiv.org/abs/2507.17788)**: Reduces repeated-query cost in LLM ranking by adaptively stopping repetitions.
- **[LongLLMLingua](https://arxiv.org/abs/2310.06839)**: Training-free prompt compression and document reordering to mitigate lost-in-the-middle in RAG.
- **[ReFilter](https://arxiv.org/abs/2602.12709)**: Improves RAG robustness by learning token-level gates to suppress irrelevant retrieved content.
- **[RULER](https://arxiv.org/abs/2404.06654)**: Provides synthetic long-context benchmarks for retrieval and reasoning.
- **[LongBench](https://arxiv.org/abs/2308.14508)**: Benchmarks long-context understanding across multiple task types and domains.
- **[YaRN](https://arxiv.org/abs/2309.00071)**: Extends RoPE context windows efficiently and enables 64K+ contexts.
- **[Position Interpolation](https://arxiv.org/abs/2306.15595)**: Extends RoPE models to longer contexts via rescaled position indices.
- **[LongRoPE](https://arxiv.org/abs/2402.13753)**: Learns RoPE scaling to extend context length.
- **[StreamingLLM](https://arxiv.org/abs/2309.17453)**: Studies attention sinks and streaming inference, relevant to long-context attention pathologies.
- **[Where is the Answer? Positional Bias in Knowledge Extraction](https://arxiv.org/abs/2402.12170)**: Studies positional bias effects in extracting information from long documents.
- **[Can We Instruct LLMs to Compensate for Position Bias?](https://aclanthology.org/2024.findings-emnlp.732/)**: Evaluates prompt-based strategies for reducing position bias.
- **[Uncovering the Role of Initial Saliency in U-Shaped Attention Bias](https://arxiv.org/abs/2512.13109)**: Analyzes mechanistic sources of U-shaped attention bias and proposes training-free interventions.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Iterative attention-based reordering | Use first-step attention to reorder documents, optionally iteratively | Attention Sorting | SynthWiki | Can require multiple expensive prefill passes |
| Positional-bias calibration | Model attention as relevance + bias(position) and remove bias | FITM, initial-saliency methods | NaturalQuestions, SynthWiki | Often needs extra forward passes or careful calibration |
| Attention-based reranking (IR) | Use attention aggregation as a relevance signal for reranking | ICR, ReAttn, CoRe | BEIR, multi-hop QA | Usually focuses on reranking, not end-to-end long-context QA |
| Permutation / aggregation | Average out or exploit positional bias via multiple permutations | PSC, Gold Panning, RankFusion | TREC-DL, simulations | Multiple LLM calls; cost grows with #permutations |
| Prompt compression / placement | Compress/reorder prompt to fit token budget and leverage position preferences | LongLLMLingua | LongBench, narrative QA | Often requires heuristics; may drop needed evidence |
| Training-based long-context extension | Modify RoPE/training to increase context length | YaRN, Position Interp, LongRoPE | Perplexity + long-context tasks | Does not directly eliminate positional bias |

### Closest Prior Work

- **Attention Sorting**: Uses raw first-token attention to reorder documents; requires multiple iterations because a relevant document far from the end may not move enough in a single pass. Our work replaces raw attention with debiased attention to eliminate iterations.
- **Found in the Middle**: Provides an explicit relevance+bias model and a dummy-document subtraction technique; however it introduces extra O(K) forward passes to calibrate per position. Our work keeps the “bias subtraction” concept but estimates bias from distractors *within the same prompt* without extra passes.
- **Attention Basin / AttnRank**: Learns an attention-position profile and reorders items so high-relevance items are placed in high-attention positions. Unlike AttnRank, we do not assume an external retriever provides relevance ordering; we instead aim to improve the *attention-derived* relevance ordering itself.
- **ICR**: Subtracts attention scores from a content-free calibration query (two forward passes) to reduce intrinsic attention biases for reranking. Our approach is calibration-free at the prompt level (no second query) and specifically targets eliminating the need for multiple sorting iterations in long-context QA.
- **Gold Panning / PSC**: Reduce position bias via multiple permutations and aggregation or belief updates. Our approach is a single-pass (k=1) alternative specialized to the “one relevant doc + many distractors” regime.

**Novelty Kill Search Summary:** Searched for combinations of “attention sorting positional bias calibration”, “bias-corrected attention sorting”, “one-pass attention sorting”, and “Found in the Middle attention sorting”, and scanned local finalized and draft proposals for “Attention Sorting” / “2310.01427”. No prior work explicitly proposing **per-prompt bias subtraction to replace multi-iteration Attention Sorting** was found as of 2026-02-26 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Attention Sorting | Iteratively reorder by raw first-token attention | Needs k>1 expensive prefill passes | Replace raw attention with debiased scores to use k=1 | Debiasing makes one pass sufficient if iterations mainly correct position bias |
| FITM | Dummy-document subtraction to remove position bias | Extra O(K) forward passes | Estimate bias from distractor docs in the same prompt | No extra passes; prompt-adaptive bias estimate |
| ICR | Uses content-free query to calibrate attention scores | Requires extra forward pass; not designed for iterative sorting | No calibration query; per-prompt robust bias fit | Lower overhead; specialized to multi-doc QA sorting |
| AttnRank | Reorders based on external relevance ranking + attention profile | Depends on retriever; doesn’t fix attention-derived relevance ordering | Improve attention-derived relevance ranking directly | Works even when only attention provides relevance signal |
| PSC / Gold Panning | Multiple permutations + aggregation/bandits | Multiple LLM calls | Single-pass alternative | Lower latency/cost when “many distractors” assumption holds |

---

## Experiments

### Experimental Setup

We evaluate whether debiasing removes the need for iterative sorting.

**Primary benchmark:** SynthWiki (from Attention Sorting; 990 extractive QA instances over synthetic biographies, designed to minimize pretraining contamination). We use the official dataset and evaluation code from https://github.com/adamlerer/synthwiki.

**Base model (primary):** togethercomputer/LLaMA-2-7B-32K-Instruct (https://huggingface.co/togethercomputer/LLaMA-2-7B-32K-Instruct) as used in Attention Sorting.

**(Optional second model if budget allows):** NousResearch/YaRN-Llama-2-7b-64k (https://huggingface.co/NousResearch/Yarn-Llama-2-7b-64k) to test a stronger-recency-bias model.

**Context construction:** Follow Attention Sorting’s setup: construct contexts of ~30K tokens by sampling distractor documents until the next sample would exceed the max length, then shuffle document order so the gold document is at a random position (described in Attention Sorting Section 3).

**Baseline Ladder (REQUIRED):**
- **Level 1 (prompting):** No sorting with the official SynthWiki prompt template; include one additional prompt variant that explicitly requests copying an exact substring from the context.
- **Level 4 (inference-time scaling, task-specific):** k=5 Attention Sorting (strongest published baseline family for this benchmark).
- **Level 5 (closest method family):** k=5 Attention Sorting is the closest known method for this exact SynthWiki setting.

**Main conditions (3) + one ablation:**
- **A) No sorting**: vanilla generation.
- **B) Attention Sorting k=5**: iterative sorting baseline.
- **C) Debiased one-pass sorting (ours)**: k=1 with per-prompt bias subtraction.
- **Ablation) Uncalibrated one-pass sorting**: k=1 Attention Sorting to isolate the effect of debiasing.

**Implementation note (attention extraction):** For the first decode step, run the model with attention outputs enabled for query_len=1 so attention tensors are O(L) rather than O(L^2) to store. If FlashAttention prevents attention output, use an eager-attention fallback for the first-step pass and keep the same attention implementation across all conditions that require attention (k=1 and k=5).

**Resource Estimate**:
- **Compute budget**: 60–200 A100-80GB GPU-hours total.
  - Rationale: k=5 requires 6 prefill passes per example; k=1 methods require 2. We plan to start with a 200-example subset (random seed) for 3 shuffle seeds, then scale up toward the full 990 if time permits.
- **GPU memory**: 1×A100-80GB should be sufficient for 7B at 32K with paged KV cache; attention output at query_len=1 adds moderate overhead.
- **External APIs**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| SynthWiki | Synthetic long-context extractive QA with one gold document + many distractors | Exact-match accuracy (string containment after quote removal), gold-doc rank by attention | test | https://github.com/adamlerer/synthwiki | repo eval scripts + our sorting module |

### Main Results

**Primary table (to be filled by verification):**

| Method | Base Model | Benchmark | Exact-match acc (mean±std) | Mean prefill passes / query | Source | Notes |
|---|---|---|---|---:|---|---|
| No sorting | LLaMA-2-7B-32K-Instruct | SynthWiki@30K | **TBD** | 1 | - | To be reproduced |
| Attention Sorting (k=1) | LLaMA-2-7B-32K-Instruct | SynthWiki@30K | **TBD** | 2 | - | Ablation |
| **Debiased one-pass sorting (ours, k=1)** | LLaMA-2-7B-32K-Instruct | SynthWiki@30K | **TBD** | 2 | - | Per-prompt \(\hat b(p)\) |
| Attention Sorting (k=5) | LLaMA-2-7B-32K-Instruct | SynthWiki@30K | **TBD** | 6 | - | Strongest baseline |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours w/ different trim fraction | Change \(\alpha\) (e.g., 0.01 vs 0.10) | If bias estimation is robust, results are stable across \(\alpha\) |
| Ours w/ offline bias curve | Replace per-prompt \(\hat b\) with a global \(b_{\text{offline}}\) | Per-prompt should win if bias is prompt-dependent |

### Experimental Rigor

**Variance & Reproducibility:**
- Use **3 random seeds** for (i) distractor sampling and (ii) document shuffling within each prompt.
- Use greedy decoding for deterministic generation; report mean±std across seeds.

**Validity & Controls:**
- **Phase-0 gate (rank-lift):** On a small subset (e.g., 50 prompts), measure whether debiasing improves the gold-document rank (lower is better) when ranking by \(a_i\) vs \(s_i\). If median rank does not improve and win/tie rate is <60%, stop early.
- **Prompt sensitivity confound:** keep the exact prompt template fixed across conditions; only reorder document blocks.
- **Attention backend confound:** ensure all methods that use attention extraction use the same attention implementation for the attention pass.

**Sanity checks:**
- Random reordering should not reliably improve accuracy over no sorting.
- k=5 Attention Sorting should reproduce a non-trivial gain over no sorting (otherwise the regime is uninformative for this question).

---

## Success Criteria

**Hypothesis:** Debiased one-pass sorting will substantially outperform uncalibrated k=1 sorting and will match (or nearly match) k=5 Attention Sorting on long-context SynthWiki.

**Decision Rule:**

1. **Precondition (regime check):** Proceed only if Attention Sorting k=5 improves over no sorting by **≥3.0 accuracy points** on SynthWiki@30K (same model, same prompt).
2. **Phase-0 rank-lift gate:** On 50 prompts, proceed only if debiased ranking improves gold-doc rank vs raw-attention ranking (median improvement > 0 and win/tie rate ≥60%). Refute if not.
3. **Main success (accuracy + efficiency):** On SynthWiki@30K (≥200 prompts, 3 seeds), proceed if:
   - Debiased k=1 accuracy is within **1–2 points** of k=5 accuracy, and
   - Debiased k=1 wins or ties vs k=5 on **≥80%** of prompts (paired comparison), and
   - Debiased k=1 improves over uncalibrated k=1 by a margin outside the std range.
4. **Refute:** If debiased k=1 is statistically indistinguishable from uncalibrated k=1, or if it is **≥3 points** worse than k=5 while k=5 clears the precondition.
5. **Pivot:** If debiased k=1 improves over uncalibrated k=1 but remains 2–3 points behind k=5, try a minimal hybrid (k=2 debiased iterations) to test whether residual error is due to imperfect bias estimation rather than the one-pass hypothesis.

---

## Impact Statement

If successful, this work provides a training-free way to retain most of Attention Sorting’s long-context QA gains while reducing the number of long-context prefill passes from 6 to 2 (for k=5), which is directly relevant to RAG latency and serving cost. If it fails, the negative result is still decision-relevant: it would suggest that iterative Attention Sorting is correcting more than a simple per-position attention bias, guiding future work toward other mechanisms (e.g., attention dilution or multi-step retrieval dynamics).

---

## References

- [Attention Sorting Combats Recency Bias In Long Context Language Models](./references/Attention Sorting Combats Recency Bias In Long Context Language Models/meta/meta_info.txt) - Peysakhovich & Lerer, 2023
- [Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization](./references/Found in the Middle Calibrating Positional Attention Bias Improves Long Context Utilization/meta/meta_info.txt) - Hsieh et al., 2024
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - Liu et al., 2023
- [Attention in Large Language Models Yields Efficient Zero-shot Re-rankers](https://arxiv.org/abs/2410.02642) - Chen et al., 2024/2025
- [Attention Basin: Why Contextual Position Matters in Large Language Models](https://arxiv.org/abs/2508.05128) - Yi et al., 2025
- [ReAttn: Improving Attention-based Re-ranking via Attention Re-weighting](https://arxiv.org/abs/2602.19969) - 2026
- [Contrastive Retrieval Heads Improve Attention-Based Re-Ranking](https://arxiv.org/abs/2510.02219) - 2025
- [Permutation Self-Consistency Improves Listwise Ranking in Large Language Models](https://arxiv.org/abs/2310.07712) - Tang et al., 2023
- [Gold Panning: Turning Positional Bias into Signal for Multi-Document LLM Reasoning](https://arxiv.org/abs/2510.09770) - Byerly & Khashabi, 2025
- [LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking](https://arxiv.org/abs/2406.00231) - Zeng et al., 2024
- [Adaptive Repetition for Mitigating Position Bias in LLM-Based Ranking](https://arxiv.org/abs/2507.17788) - Vardasbi et al., 2025
- [LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression](https://arxiv.org/abs/2310.06839) - Jiang et al., 2023
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) - Peng et al., 2023
- [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595) - Chen et al., 2023
- [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) - Ding et al., 2024
- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) - Xiao et al., 2023
- [Where is the Answer? Investigating Positional Bias in Language Models’ Knowledge Extraction](https://arxiv.org/abs/2402.12170) - 2024
- [Can We Instruct LLMs to Compensate for Position Bias?](https://aclanthology.org/2024.findings-emnlp.732/) - 2024
- [Uncovering the Role of Initial Saliency in U-Shaped Attention Bias](https://arxiv.org/abs/2512.13109) - 2025
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) - Touvron et al., 2023
