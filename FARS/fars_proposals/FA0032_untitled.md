# untitled

# Risk-Controlled Early Stopping for Long-Context Memory Agents

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models increasingly face inputs that are far longer than a single model context window, such as long reports, codebases, or many retrieved documents. A growing line of work addresses this by turning a language model into a **memory agent**: the agent reads the document in chunks, repeatedly updates a bounded “working memory” (a fixed token budget inside the model context), and only produces the final answer from that memory. This makes inference cost scale roughly linearly with document length and enables processing of extremely long contexts.

However, these memory agents can still be expensive at inference time because they may process hundreds of chunks for a single query. Recent agents therefore introduce **early stopping** policies (stop reading once enough evidence is in memory) to reduce cost. For example, InfMem reports substantial efficiency–quality gains from learned early stopping (e.g., +7.73 to +11.80 accuracy points with 3.3× to 5.1× latency reduction across backbones; see InfMem `sections/Early stopping.md`).

Yet learned stopping policies still leave deployment knobs (e.g., InfMem’s 1-stop vs 3-stop) without an explicit, user-chosen degradation budget, and adapting them to a new risk tolerance or a new data distribution generally requires retraining or heuristic retuning. This motivates a **post-hoc, training-free calibration wrapper** that can be applied to any existing memory agent (including ones with public checkpoints today, such as MemAgent).

A practical deployment problem remains: early stopping introduces a **stop aggressiveness hyperparameter** (or a learned stop policy) whose safety implications are hard to reason about. In many applications, the key operational question is not “what stop rule maximizes average score?”, but:

> **How do we stop early while guaranteeing we do not break more than an ε fraction of the full-read system’s correct answers?**

This proposal aims to convert early stopping from ad-hoc tuning into an explicit, distribution-free calibration problem: given a small calibration set and a user-chosen risk budget (ε, δ), automatically pick the least conservative stopping setting that provably respects that budget.

### The Problem

We focus on **MemAgent** (a public, reproducible long-context memory agent) and its long-context QA evaluation setting constructed from HotpotQA and the **RULER** long-context QA synthesis methodology (embed gold evidence paragraphs into a large in-domain distractor “haystack” at controlled lengths).

- **MemAgent** reads a long document chunk-by-chunk, overwriting a fixed-size memory state each step, and answers from the final memory. It achieves strong accuracy even at very long contexts but requires many steps at inference time.  
  - **[MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent](./references/MemAgent-Reshaping-Long-Context-LLM-with-Multi-Conv-RL-based-Memory-Agent/meta/meta_info.txt)**.

- **InfMem** adds explicit control (PreThink–Retrieve–Write) plus early stopping and reports large accuracy and efficiency improvements over MemAgent, but its stop behavior is learned via training-time signals (e.g., identifying the earliest step where a verifier can already answer correctly from memory), and its final stopping policy choices (e.g., “1-stop” vs “3-stop”) are still a deployment knob rather than a user-facing guarantee.  
  - **[InfMem: Learning System-2 Memory Control for Long-Context Agent](./references/InfMem-Learning-System-2-Memory-Control-for-Long-Context-Agent/meta/meta_info.txt)**.
  - **Reproducibility note:** as of this writing, InfMem releases code and data but not an inference-ready model checkpoint; we therefore run verification experiments on MemAgent (public checkpoints) while using InfMem’s reported efficiency–accuracy frontier as a reference point.

Meanwhile, the conformal prediction / risk control literature provides post-hoc procedures that choose thresholds with **finite-sample, distribution-free guarantees** (under explicit assumptions). These methods have been applied to early-exit neural networks and to chain-of-thought early stopping, but (to our knowledge) not to **memory-agent stopping in chunk-scanning loops**, where “reading more can hurt” is plausible due to memory overwrite and distractor assimilation.

Crucially, this setting is not just “another early-exit application”: if memory overwrite makes the degradation risk **non-monotone** in the stopping knob, then standard CRC/UCB-style guarantees (which require monotonicity) can fail—making the applicability of risk control itself an open empirical question.

### Key Insight and Hypothesis

**Key insight.** Memory-agent early stopping can be framed as selecting a **single monotone “conservativeness” knob** that trades off compute and a clearly defined degradation risk relative to a full-read baseline. If we can find a knob where this degradation risk is empirically non-increasing as the knob becomes more conservative, we can apply conformal risk control to pick the least conservative setting that satisfies a user-chosen ε.

**Hypothesis.** For MemAgent-style long-context QA, the risk

- r_i(k) = 1[ c_full(i)=1 ∧ c_stop(i,k)=0 ]

(where c_full is correctness of full-read MemAgent and c_stop is correctness of the early-stopped MemAgent under knob value k) is approximately non-increasing as k becomes more conservative for a practical stopping rule based on **answer stability**. Under this approximate monotonicity, an **Upper Confidence Bound (UCB)** risk-control calibration rule will select a k that satisfies the user-specified ε on held-out data while still yielding meaningful chunk-count (and wall-clock) reduction.

This could fail for two main reasons: (i) the risk is strongly non-monotone in k due to memory overwrite dynamics, invalidating CRC/UCB-style guarantees; or (ii) the risk-controlled k is so conservative that early stopping rarely triggers, giving negligible speedup.

---

## Proposed Approach

### Overview

We propose **RC-MemStop**, a calibration wrapper for early stopping in memory agents.

- RC-MemStop does **not** retrain MemAgent.
- It introduces a **discrete stop-rule knob k** and selects k from a small candidate set using conformal risk control.
- It provides a user-facing guarantee: with probability ≥ 1−δ over the calibration set draw, early stopping breaks at most an ε fraction of the full-read system’s correct answers.

### Method Details

#### Base setting: MemAgent chunk-scanning
We use the public RL-MemAgent checkpoints and evaluation pipeline.

- Each instance i consists of a question q_i, a synthesized long document split into T chunks, and a ground-truth answer y_i.
- Full-read MemAgent processes all chunks 1..T, producing a final answer Â_full(i) and correctness c_full(i)=1[Â_full(i)=y_i].

#### Answer-stability early stopping rule (the knob)
We define an intermediate “draft answer” after each chunk step t:

1. Run MemAgent’s normal memory update to obtain memory state m_t.
2. Run the MemAgent answer-generation module conditioned on (q_i, m_t) with deterministic decoding and a small max_new_tokens cap (e.g., 32–64) to obtain a draft answer string a_t.

Define a normalization function Normalize(·) consistent with the MemAgent evaluation verifier (e.g., stripping punctuation/articles/case).

For a chosen integer k (k≥1), define the stopping time:

- t_stop(k) = min{ t : Normalize(a_{t−k+1}) = … = Normalize(a_t) }.

If the condition never holds, set t_stop(k)=T.

The early-stopped prediction is Â_stop(i,k)=a_{t_stop(k)} and correctness c_stop(i,k)=1[Â_stop(i,k)=y_i].

This yields a monotone “conservativeness” knob: larger k requires longer stability, and thus (typically) stops later.

#### Risk definition (“do not break full-read successes”)
We control the risk of converting a full-read success into a failure.

1. Run **full-read** MemAgent once to compute c_full(i)=1[Â_full(i)=y_i].
2. Restrict attention to the subset of full-read successes:
   - D_succ = { i : c_full(i)=1 }.
3. Define the per-instance loss on this conditional distribution:
   - ℓ_i(k) = 1[ c_stop(i,k)=0 ] for i∈D_succ.

The target risk is the *fraction of full-read successes broken by early stopping*:

- R(k) = E[ ℓ_i(k) | i ∈ D_succ ].

#### Calibration: Naive vs risk-controlled selection
Let K be a small discrete set, e.g., K={1,2,3,4,5,6,8,10}.

- **Naive empirical tuning**: choose  
  - k_emp = min{ k∈K : R̂_cal(k) ≤ ε }, where R̂_cal(k) = (1/|D_cal,succ|)∑_{i∈D_cal,succ} ℓ_i(k).

- **UCB risk-controlled tuning (primary, monotone-risk regime)**: use the distribution-free UCB rule from risk-control literature to compute an upper bound R̂⁺_cal(k) such that with probability ≥1−δ, R(k) ≤ R̂⁺_cal(k) for all k∈K. Then select
  - k_UCB = min{ k∈K : R̂⁺_cal(k′) ≤ ε for all k′ ≥ k }.

We will compute R̂⁺ using the Waudby-Smith–Ramdas betting bound (a tight finite-sample concentration bound for bounded losses) as implemented in RC-EENN (Fast yet Safe).

- **Learn-then-Test (LTT) fallback (non-monotone-risk regime)**: if the monotonicity diagnostic fails, we will use Learn-then-Test to select a k without assuming monotonicity.
  - For each k∈K, test H0: R(k) > ε using a one-sided binomial test on the calibration successes (count failures among D_cal,succ).
  - Apply a multiple-testing correction (e.g., Holm-Bonferroni) at level δ and select the least-conservative k among those that pass.

#### Go/no-go monotonicity diagnostic
CRC/UCB selection implicitly relies on risk being non-increasing with conservativeness. We will empirically plot R̂_cal(k) versus k and treat strong violations (e.g., multiple large upward jumps) as evidence that the assumption does not hold for this stop rule.

### Key Innovations

- **When does risk control even apply to memory agents?** We make the *applicability* of CRC/UCB a testable claim by measuring whether broken-success risk is monotone in a natural conservativeness knob. This is a qualitatively new failure mode vs. diffusion-LM or CNN early exit because memory overwrite can make “more compute” harmful.
- **A deployment-facing guarantee for memory-agent stopping**: cast early stopping in chunk-scanning memory agents as calibration to satisfy a user-chosen ε guarantee relative to full-read behavior.
- **A practical monotone knob for memory-agent early stopping**: answer-stability k-stop, designed to be low overhead and easy to implement in existing memory-agent codebases.
- **A decisive diagnostic for applicability**: explicitly test whether the broken-success risk is monotone in k; if not, fall back to Learn-then-Test (LTT) which does not assume monotonicity.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) long-context processing via memory agents and retrieval, (ii) adaptive inference / early exiting, and (iii) distribution-free calibration and risk control.

Memory agents like MemAgent and InfMem provide a bounded-cost alternative to extending transformer context windows, and recent work shows early stopping can yield large efficiency gains. Separately, risk control methods such as conformal risk control (CRC), Upper Confidence Bounds (UCB), and Learn-then-Test (LTT) provide post-hoc calibration procedures that can turn heuristic thresholds into user-facing statistical guarantees.

### Related Papers

- **[MemAgent](./references/MemAgent-Reshaping-Long-Context-LLM-with-Multi-Conv-RL-based-Memory-Agent/meta/meta_info.txt)**: RL-trained chunk-scanning memory agent that extrapolates to very long contexts; baseline system we calibrate.
- **[InfMem](./references/InfMem-Learning-System-2-Memory-Control-for-Long-Context-Agent/meta/meta_info.txt)**: Adds explicit control + early stopping and shows large speed/accuracy gains; motivates stop policies but does not provide distribution-free calibration.
- **[GRU-Mem](https://arxiv.org/abs/2602.10560)**: Adds update/exit gates to memory agents; highlights that stopping decisions are a first-class capability.
- **[RULER](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt)**: Long-context benchmark/synthesis framework used to construct QA-with-distractors settings.
- **[HotpotQA](https://arxiv.org/abs/1809.09600)**: Multi-hop QA dataset used as a seed corpus for long-context QA synthesis.
- **[RAG (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)**: Retrieval-augmented generation; a standard baseline family for long-context QA.
- **[YaRN](https://arxiv.org/abs/2309.00071)**: Train-free RoPE scaling method; a common long-context baseline referenced by memory-agent papers.
- **[Qwen2.5-1M technical report](https://arxiv.org/abs/2501.15383)**: Example of long-context window extension with a large advertised context length but potential performance cliffs.
- **[QwenLong-L1](https://arxiv.org/abs/2505.17667)**: Long-context reasoning model trained with RL; used as a baseline in MemAgent.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**: RL-trained reasoning model; used as a baseline family and illustrates test-time compute scaling.
- **[Fast yet Safe](./references/Fast-yet-Safe-Early-Exiting-with-Risk-Control/meta/meta_info.txt)**: Risk control for early-exit neural networks; provides CRC/UCB/LTT machinery we adapt.
- **[Conformal Thinking](./references/Conformal-Thinking-Risk-Control-for-Reasoning-on-a-Compute-Budget/meta/meta_info.txt)**: CRC/UCB-calibrated early stopping for chain-of-thought reasoning under compute budgets.
- **[Conformal Risk Control](./references/Conformal-Risk-Control/meta/meta_info.txt)**: Foundational CRC framework for expectation risk control of monotone losses.
- **[Learn then Test (LTT)](./references/Learn-then-Test-Calibrating-Predictive-Algorithms-to-Achieve-Risk-Control/meta/meta_info.txt)**: Risk control via multiple hypothesis testing; important alternative when monotonicity fails.
- **[Safe & Efficient ICL via Risk Control](./references/Safe-and-Efficient-In-Context-Learning-via-Risk-Control/meta/meta_info.txt)**: Applies LTT-style risk control to early exiting in ICL, emphasizing “overthinking” / harmful context regimes.
- **[CALM](https://arxiv.org/abs/2207.07061)**: Early-exit language modeling with LTT-based guarantees; canonical LLM early-exit baseline.
- **[BranchyNet](https://arxiv.org/abs/1709.01686)**: Early-exit networks; foundational architecture for adaptive computation.
- **[MSDNet](https://arxiv.org/abs/1810.06731)**: Multi-scale early-exit network; widely used in early-exit literature.
- **[Shallow-Deep Networks / overthinking](https://arxiv.org/abs/1810.10189)**: Shows deeper computation can hurt; conceptually analogous to memory overwrite degradation.
- **[Early Stopping Chain-of-thoughts in LLMs](../../papers/paper_summaries/Early%20Stopping%20Chain-of-thoughts%20in%20Large%20Language%20Models.md)**: Heuristic early stopping for CoT traces; contrasts with guarantee-driven calibration.
- **[Stop When Enough](../../papers/paper_summaries/Stop%20When%20Enough%20Adaptive%20Early-Stopping%20for%20Chain-of-Thought%20Reasoning.md)**: Adaptive early stopping for CoT; another non-memory-agent stopping baseline.
- **[Lost in the Noise](../../papers/paper_summaries/Lost%20in%20the%20Noise%20How%20Reasoning%20Models%20Fail%20with%20Contextual%20Distractors.md)**: Shows contextual distractors can induce inverse scaling with more reasoning; motivates explicit safety constraints when reducing compute.
- **[CONFLARE](https://arxiv.org/abs/2404.04287)**: Conformal methods for retrieval coverage in RAG; related conformal calibration idea but different decision (retrieve vs stop reading).

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Memory agents for long contexts | Process documents in chunks with bounded memory tokens | MemAgent, InfMem, GRU-Mem | RULER QA subsets, LongBench-style QA | Many-step inference cost; memory overwrite can lose info |
| Early stopping / adaptive compute (LMs) | Exit early based on confidence/stability | CALM, early-exit networks (BranchyNet/MSDNet) | CNN/DM, WMT, SQuAD; ImageNet | Threshold tuning; guarantees often relative to full model |
| Risk control / conformal calibration | Post-hoc threshold selection with finite-sample guarantees | CRC, UCB risk control, LTT | Many tasks incl. NLP and vision | Monotonicity assumptions (CRC/UCB) can fail; LTT can be conservative |
| Risk-controlled stopping for reasoning | Stop reasoning under compute budgets with guarantees | Conformal Thinking | AIME/GPQA etc | Focuses on CoT token steps, not chunk-reading memory loops |

### Closest Prior Work

- **MemAgent**: Provides an open, strong memory-agent baseline with public checkpoints and evaluation scripts, but does not include a principled early stopping guarantee; by design it processes the full chunk stream.
- **InfMem**: Demonstrates the value of early stopping for memory agents and reports a strong efficiency–accuracy frontier, but its early-stop behavior is learned via training-time signals and does not provide a post-hoc, user-chosen risk guarantee.
- **Fast yet Safe**: Provides the closest risk-control machinery for early exit thresholds (CRC/UCB/LTT) and discusses overthinking, but does not study long-context memory-agent reading loops.
- **Conformal Thinking**: Provides a similar “risk as a deployment knob” framing for CoT stopping, but uses a different computational axis (token-length reasoning traces rather than chunk-scanning memory updates).
- **CALM**: Introduces LTT-based guarantees for early exit in language modeling, but does not address long-context memory-agent architectures.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MemAgent | Chunk-scanning memory agent for ultra-long QA | No early stop; no safe deployment knob | Add calibrated early stopping wrapper | Reduces inference cost with explicit “break ≤ε successes” target |
| InfMem | Learned control + early stop | Requires training-time stop supervision; no distribution-free calibration | Post-hoc calibration on top of fixed agent | Makes stop aggressiveness deployable without retraining |
| Fast yet Safe | CRC/UCB/LTT for early-exit nets | Not evaluated on memory agents | Apply to chunk-scanning stop knob | Tests whether risk control assumptions hold in overwrite-based loops |
| Conformal Thinking | Risk-controlled CoT stopping | Not a memory-agent setting | Change compute axis to memory-agent steps | Extends “risk as a knob” to long-context reading agents |
| CALM | LTT-based early exit for LMs | Not long-context reading/memory updates | New stopping signal + evaluation regime | Addresses a different deployment bottleneck (many chunk steps) |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| BytedTsinghua-SIA/RL-MemoryAgent-7B | 7B | https://huggingface.co/BytedTsinghua-SIA/RL-MemoryAgent-7B | Primary reproducible target |
| BytedTsinghua-SIA/RL-MemoryAgent-14B | 14B | https://huggingface.co/BytedTsinghua-SIA/RL-MemoryAgent-14B | Optional (if budget allows) |

**Training Data (if applicable):**

No training required (inference + calibration only).

**Other Resources (if applicable):**

- MemAgent code + eval pipeline: https://github.com/BytedTsinghua-SIA/MemAgent

**Resource Estimate**:

- **Compute budget**: Target ≤ 768 A100 GPU-hours. Main cost is MemAgent inference on long contexts. A rough proxy is total generated memory tokens: each chunk step generates ~1024 memory tokens under ~7K-token context. For 128 examples at 448K (≈90 steps) and 896K (≈180 steps), this is on the order of ~35M generated tokens, which should be feasible with multi-GPU parallelism (e.g., 8×A100 for a couple of days). The intermediate-answer calls add comparatively small overhead (short outputs on ~2K-token context).
- **GPU memory**: RL-MemoryAgent-7B/14B bf16 inference should fit within 80GB GPUs; MemAgent repo supports vLLM-based serving.
- **API usage**: None required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| MemAgent long-context HotpotQA (RULER-style synthesis) | HotpotQA questions embedded into long distractor docs at controlled lengths | (i) Accuracy (exact match under MemAgent test verifier); (ii) **Broken-success risk** R(k)=P[c_stop(i,k)=0 \mid c_full(i)=1] | HotpotQA-val-derived synthesized test sets at multiple lengths | MemAgent repo data scripts | https://github.com/BytedTsinghua-SIA/MemAgent (taskutils/memory_data + taskutils/memory_eval) |

**Evaluation Scripts:**

- Use MemAgent’s provided evaluation pipeline (`taskutils/memory_eval`). Implement RC-MemStop by adding: (i) a per-step intermediate answer generation call from the current memory, (ii) computation of t_stop(k) for a fixed k, (iii) a calibration script to select k_UCB, and (iv) a driver to run early-stopped inference on the test split.

### Main Results

#### Results Table

Baseline numbers below are copied from MemAgent’s Table 2 (same benchmark family and evaluation protocol; see `./references/MemAgent-Reshaping-Long-Context-LLM-with-Multi-Conv-RL-based-Memory-Agent/sections/4.3 Main Results.md`). We also report **InfMem’s published efficiency–accuracy frontier** to contextualize what “competitive” early stopping looks like in this problem family, even though InfMem checkpoints are not yet released.

| Method | Base Model | Context length | Accuracy / Perf. | Broken-success risk @ ε | Avg steps (chunks) / Time | Speedup vs full | Source | Notes |
|--------|------------|----------------|------------------|--------------------------|---------------------------|-----------------|--------|------|
| Full read | RL-MemoryAgent-7B | 448K | 74.22 (acc %) | 0.00 (by definition) | T chunks | 1.0× | MemAgent (Table 2) | Processes all chunks |
| Full read | RL-MemoryAgent-7B | 896K | 76.56 (acc %) | 0.00 (by definition) | T chunks | 1.0× | MemAgent (Table 2) | Processes all chunks |
| **InfMem 3-stop (reported)** | Qwen2.5-7B | up to 1M | +7.73 pts vs MemAgent | N/A | **3.3× faster** | 3.3× | InfMem (Early stopping) | Learned policy; no ε guarantee |
| **InfMem 1-stop (reported)** | Qwen2.5-7B | up to 1M | (lower than 3-stop) | N/A | **5.1× faster** | 5.1× | InfMem (Early stopping) | More aggressive; lower perf |
| Naive early stop (k_emp) | RL-MemoryAgent-7B | 448K/896K | **TBD** | **TBD** | **TBD** | **TBD** | - | Calibrate k by empirical risk only |
| Risk-controlled early stop (k_UCB) | RL-MemoryAgent-7B | 448K/896K | **TBD** | ≤ ε (target) | **TBD** | **TBD** | - | UCB-calibrated k |
| Risk-controlled early stop (k_LTT, fallback) | RL-MemoryAgent-7B | 448K/896K | **TBD** | ≤ ε (target) | **TBD** | **TBD** | - | For non-monotone risk regimes |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| UCB vs empirical | k_UCB vs k_emp | UCB is more conservative but more reliable (risk violations rarer) |
| LTT fallback | k_LTT when monotonicity fails | Provides ε control without monotonicity, but may be more conservative |
| Calibration split variance | 3 random D_cal/D_test splits | Check stability of selected k and realized risk/speedup |
| k grid sensitivity | Different candidate sets K | Coarser grids may reduce achievable speedup at a fixed ε |

### Analysis (Optional)

- **Monotonicity plot**: empirical broken-success risk R̂(k) vs k (computed on D_succ). If this is strongly non-monotone, CRC/UCB is invalid and we switch to LTT.
- **Stop-time distribution**: histogram of t_stop(k_*) / T across instances (k_* is the calibrated knob).
- **“Overwriting overthinking” cases**: instances where early stopping improves over full read (c_full=0, c_stop=1), indicating that additional chunks sometimes harm.

---

## Success Criteria

**Criterion 1: Risk control works when assumptions hold**
- Hypothesis: For at least one ε in {0.05, 0.10}, the UCB-calibrated k_UCB achieves empirical test risk (fraction of broken full-read successes) ≤ ε on held-out test.
- Validation: Compute R̂_test(k_UCB) and compare to ε.

**Criterion 2: Non-trivial compute reduction under guarantees**
- Hypothesis: k_UCB does not collapse to full read (i.e., stopping almost never triggers).
- Validation: Achieve ≥1.5× speedup in average processed chunks at ε=0.10 on at least one long length (≥448K).

**Refutation conditions**
- If R̂(k) is strongly non-monotone in k on the calibration sweep, CRC/UCB guarantees are not applicable to this stop rule.
- If k_UCB yields speedup <1.2× even at ε=0.10, the approach is not practically useful for MemAgent-style loops.

---

## Impact Statement

If successful, RC-MemStop would let practitioners deploy early stopping in long-context memory agents by selecting a stop aggressiveness parameter from a small calibration set with a user-chosen error budget, replacing per-task heuristic tuning with a distribution-free guarantee on how often early stopping breaks previously-correct answers.

---

## References

- [MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent](./references/MemAgent-Reshaping-Long-Context-LLM-with-Multi-Conv-RL-based-Memory-Agent/meta/meta_info.txt) - Yu et al., 2025
- [InfMem: Learning System-2 Memory Control for Long-Context Agent](./references/InfMem-Learning-System-2-Memory-Control-for-Long-Context-Agent/meta/meta_info.txt) - Wang et al., 2026
- [RULER: What’s the Real Context Size of Your Long-Context Language Models?](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt) - Hsieh et al., 2024
- [Conformal Thinking: Risk Control for Reasoning on a Compute Budget](./references/Conformal-Thinking-Risk-Control-for-Reasoning-on-a-Compute-Budget/meta/meta_info.txt) - Wang et al., 2026
- [Fast yet Safe: Early-Exiting with Risk Control](./references/Fast-yet-Safe-Early-Exiting-with-Risk-Control/meta/meta_info.txt) - Jazbec et al., 2024
- [Conformal Risk Control](./references/Conformal-Risk-Control/meta/meta_info.txt) - Angelopoulos et al., 2022
- [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](./references/Learn-then-Test-Calibrating-Predictive-Algorithms-to-Achieve-Risk-Control/meta/meta_info.txt) - Angelopoulos et al., 2021
- [Safe and Efficient In-Context Learning via Risk Control](./references/Safe-and-Efficient-In-Context-Learning-via-Risk-Control/meta/meta_info.txt) - arXiv:2510.02480
- [GRU-Mem: When to Memorize and When to Stop](https://arxiv.org/abs/2602.10560) - Sheng et al., 2026
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Lewis et al., 2020
- [HotpotQA: A Dataset for Diverse Explainable Multi-Hop Question Answering](https://arxiv.org/abs/1809.09600) - Yang et al., 2018
- [Qwen2.5-1M Technical Report](https://arxiv.org/abs/2501.15383) - Qwen Team, 2025
- [QwenLong-L1](https://arxiv.org/abs/2505.17667) - Wan et al., 2025
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - Guo et al., 2025
- [Confident Adaptive Language Modeling (CALM)](https://arxiv.org/abs/2207.07061) - Schuster et al., 2022
- [BranchyNet](https://arxiv.org/abs/1709.01686) - Teerapittayanon et al., 2016
- [MSDNet](https://arxiv.org/abs/1810.06731) - Huang et al., 2018
- [Shallow-Deep Networks: Understanding and Mitigating Network Overthinking](https://arxiv.org/abs/1810.10189) - Kaya et al., 2019
- [Early Stopping Chain-of-thoughts in Large Language Models](../../papers/paper_summaries/Early%20Stopping%20Chain-of-thoughts%20in%20Large%20Language%20Models.md)
- [Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning](../../papers/paper_summaries/Stop%20When%20Enough%20Adaptive%20Early-Stopping%20for%20Chain-of-Thought%20Reasoning.md)
- [Lost in the Noise: How Reasoning Models Fail with Contextual Distractors](../../papers/paper_summaries/Lost%20in%20the%20Noise%20How%20Reasoning%20Models%20Fail%20with%20Contextual%20Distractors.md)
- [CONFLARE: CONFormal LArge language model REtrieval](https://arxiv.org/abs/2404.04287)
