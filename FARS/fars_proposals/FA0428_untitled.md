# untitled

# Confidence-Based Memory Eviction for Value-Aware Agent Memory

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation
Large language model (LLM) agents increasingly use **external memory** to persist information across long interactions and repeated tasks. A common pattern is *retrieve-then-prompt*: store past examples or trajectories in a database, retrieve a small subset relevant to the current query, and include them in the LLM prompt.

A key practical issue is that long-lived agents accumulate memories that are (i) redundant, (ii) outdated, or (iii) actively harmful (e.g., mislabeled demonstrations, incorrect past tool traces). Empirical work shows that naïvely adding everything to memory can cause **error propagation** through “experience-following” behavior, while strategic deletion can simultaneously improve performance and reduce memory size. For example, **How Memory Management Impacts LLM Agents** reports that, with a strict evaluator, **history-based deletion** can improve performance while reducing memory size (e.g., EHRAgent accuracy **38.89 → 42.65** while memory size **1012 → 784**, Table 2; see `./references/How-Memory-Management-Impacts-LLM-Agents-An-Empirical-Study-of-Experience-Following-Behavior/sections/5.2 Strategic Memory Deletion Improves the Agent Performance.md`).

Recently, **MemRL** introduced *value-aware retrieval*: instead of retrieving memories purely by embedding similarity, it learns a per-memory **utility** (a Q-value) from automated task feedback and uses it to re-rank retrieved candidates (see `./references/MemRL-MEMRL-SELF-EVOLVING-AGENTS-VIA-RUNTIME-REINFORCEMENT-LEARNING-ON-EPISODIC-MEMORY/sections/MEMRL.md`). MemRL structures memory as a growing set of intent–experience–utility triplets (Eq. 4; see `./references/MemRL-MEMRL-SELF-EVOLVING-AGENTS-VIA-RUNTIME-REINFORCEMENT-LEARNING-ON-EPISODIC-MEMORY/sections/MEMORY STRUCTURE THE INTENT-EXPERIENCE-UTILITY TRIPLET.md`), but it does not specify an **eviction** rule when memory must be capped. Its sensitivity analysis also suggests that retrieving more items can add noise in complex reasoning tasks (see `./references/MemRL-MEMRL-SELF-EVOLVING-AGENTS-VIA-RUNTIME-REINFORCEMENT-LEARNING-ON-EPISODIC-MEMORY/sections/SENSITIVITY.md`).

In deployment, capacity management is not optional: even if low-utility memories are rarely retrieved, they still consume storage, increase retrieval latency, and can crowd out useful memories when memory is capped.

### The Problem
We study **memory eviction under capacity constraints** for *value-aware memory agents*.

Setting: an agent faces a stream of classification queries (e.g., intent detection). It maintains an **active memory** of size at most **M** containing labeled examples (demonstrations). On each query, it retrieves one demonstration from memory and prompts an LLM to output a class label. After observing the ground-truth label, it updates an estimated utility for the retrieved memory item and writes the new labeled example to memory (possibly causing an eviction if memory is full).

A natural baseline is to evict the memory item with the lowest **point estimate** of utility (“evict-lowest-Q”). The issue is that point estimates confound **low utility** with **high uncertainty**: a rarely-retrieved but potentially important memory (e.g., representing a rare class) may have little evidence and thus a low estimated utility, causing it to be evicted prematurely. Conversely, truly harmful memories should be removed aggressively.

We want an eviction policy that is:
- **Conservative** (does not delete rare-but-useful memories without evidence),
- **Automatable** (no human judgments),
- **Minimal** (a small change to existing value-aware retrieval).

### Key Insight and Hypothesis
**Key insight:** Memory eviction is a statistical decision problem. When memory utility is learned online from binary success/failure signals, a posterior distribution over each memory’s utility can be maintained cheaply. Eviction should depend on a *conservative* estimate of “how good this memory could still plausibly be,” not just its posterior mean.

**Hypothesis:** Under a fixed memory budget, **confidence-based eviction** (evict the memory item with the lowest *upper* credible bound on utility) improves late-stage online accuracy and reduces the fraction of harmful memories in the active bank compared to point-estimate eviction, because it (i) retains uncertain-but-possibly-useful memories (often rare-class exemplars) and (ii) removes memories that are confidently unhelpful.

Why this might fail: (i) retrieval may not revisit items enough for confidence bounds to differ from means; (ii) errors may be dominated by base-model classification rather than memory; (iii) confidence-based eviction may keep too many uncertain-but-actually-bad memories and reduce responsiveness.

---

## Proposed Approach

### Overview
We propose **Confidence-Based Memory Eviction (CBME)** for value-aware agent memory under a capacity constraint.

Each memory item is a labeled example `m_i=(x_i, y_i)` with a learned estimate of how often it helps when retrieved. When memory capacity is reached, the agent evicts the item that is most confidently unhelpful.

### Method Details

**Streaming evaluation loop (Test-Time Learning / TTL classification).**
We follow the Test-Time Learning (TTL) framing from MemoryAgentBench: the agent processes a stream of labeled examples at inference time, retrieves from an external memory, and updates that memory online.

For each timestep `t=1..T` with query `(x_t, y_t)`: 
1. **Retrieve one memory item** `m_r` from the current active memory `M_t`.
2. Prompt the LLM with `(m_r, x_t)` and force a strict single-label output.
3. Compute reward `r_t = 1[\hat y_t == y_t]`.
4. Update the retrieved memory’s utility statistics.
5. Insert the new labeled example `(x_t, y_t)` into memory (with optional controlled corruption for stress testing), evicting one item if `|M_t| = M`.

**Value-aware retrieval (MemRL-inspired, simplified).**
Let `sim(x, x_i)` be embedding cosine similarity and `Q_hat_i` be a scalar utility estimate for memory `i`. Select the retrieved item by:

`score_i = (1-λ) * zscore(sim(x_t, x_i)) + λ * zscore(Q_hat_i)`

and retrieve `argmax_i score_i`. (Z-score normalization follows MemRL’s stability practice.)

**Utility model: Beta-Binomial posterior.**
Maintain per-memory counts `(s_i, f_i)` where `s_i` is the number of times retrieving `m_i` led to a correct answer and `f_i` is the number of times it led to an incorrect answer.

Model each memory’s success probability as:
- Prior: `p_i ~ Beta(1,1)`
- Posterior: `p_i | data ~ Beta(1+s_i, 1+f_i)`

Define:
- Point estimate baseline: `Q_hat_i = E[p_i] = (1+s_i)/(2+s_i+f_i)`
- Upper confidence bound: `UCB_i = Quantile_0.95(p_i)`

**CBME eviction rule (ours).**
When memory is full and a new item must be inserted:
- Evict `argmin_i UCB_i` (optionally restricted to items with `n_i=s_i+f_i ≥ n_min` to avoid arbitrary evictions among never-used items).

Intuition: an item with low posterior mean but high uncertainty can still have a high `UCB_i` and should not be evicted before it is tested; an item with both low mean and low `UCB_i` is confidently unhelpful.

### Key Innovations
- **Uncertainty-aware eviction for value-aware retrieval agents**: applies a simple Bayesian confidence bound to memory eviction, targeting the “rare-but-useful vs confidently-bad” ambiguity that point-estimate eviction cannot distinguish.
- **A minimal, fully automated verification harness**: online TTL classification provides exact-match rewards and allows controlled stress tests (capacity pressure + label noise) without any human evaluation.

---

## Related Work

### Field Overview
Agent memory systems vary along (i) what they store (facts, dialogues, trajectories), (ii) how they retrieve (BM25, dense, hybrid, value-aware), and (iii) how they manage lifecycle operations (write, update, delete). MemoryAgentBench frames this as competencies including accurate retrieval, test-time learning, long-range understanding, and conflict resolution (see `./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/sections/4.2 Overall Performance Comparison.md`). On its TTL-MCC benchmark, simple retrieval baselines like **BM25 (75.4)** and **Contriever (70.6)** can outperform several “memory agents” (e.g., **Mem0 3.4**, **Self-RAG 11.6**) under the paper’s evaluation setting (Table 2; same file).

A separate line studies memory **management policies** (selective write / delete) and shows that deleting can be beneficial when agents are prone to experience-following and error propagation (How Memory Management Impacts LLM Agents, arXiv:2505.16067). Another line treats memory and context management as a learnable control problem (e.g., MemAct/DCPO, MemSearcher/GRPO), but these typically learn memory editing policies end-to-end rather than isolating a small, interpretable eviction rule.

### Related Papers
(At least 20; key papers are local when available.)

- **[MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory](./references/MemRL-MEMRL-SELF-EVOLVING-AGENTS-VIA-RUNTIME-REINFORCEMENT-LEARNING-ON-EPISODIC-MEMORY/meta/meta_info.txt)**: Introduces value-aware retrieval by learning per-memory utilities (Q-values) from feedback; does not specify pruning/capacity management.
- **[How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior](./references/How-Memory-Management-Impacts-LLM-Agents-An-Empirical-Study-of-Experience-Following-Behavior/meta/meta_info.txt)**: Systematic study of memory addition/deletion; shows deletion can improve performance while shrinking memory.
- **[Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions (MemoryAgentBench)](./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/meta/meta_info.txt)**: Benchmark suite for memory agents including TTL-MCC; provides standard baselines and failure analysis.
- **[Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)**: Production-oriented CRUD memory (ADD/UPDATE/DELETE/NOOP) pipeline; not value-aware RL.
- **[MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)**: Agent framework with explicit external memory tiers; focuses on swapping context, not utility learning.
- **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)**: Canonical RAG; retrieval based on relevance rather than learned per-item utility.
- **[Self-RAG](https://arxiv.org/abs/2310.11511)**: Self-reflective retrieval/generation; not a capacity-management method.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Uses verbal feedback to improve agents; often appends experiences rather than managing retention.
- **[A-Mem: Agentic Memory for LLM Agents](https://openreview.net/forum?id=FiM0M8gcct)**: Agentic memory organization (Zettelkasten-like linking) and updates.
- **[AGENT KB: Leveraging Cross-Domain Experience for Agentic Problem Solving](https://openreview.net/pdf?id=QCLXVOMkl4)**: Cross-framework experience KB; mentions learned-utility eviction but not a focused study of uncertainty-aware retirement.
- **[MemSearcher](https://arxiv.org/abs/2511.02805)**: Trains a search agent to maintain compact memory with GRPO; focuses on working-memory compaction, not episodic eviction.
- **[Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2510.12635)**: Treats memory edits as actions and trains with DCPO; end-to-end learned memory management.
- **[HippoRAG](https://arxiv.org/abs/2405.14831)** and **[HippoRAG 2](https://openreview.net/forum?id=LWH8yn4HS2)**: Non-parametric memory via graph + PPR; emphasizes associative recall.
- **[RULER](https://arxiv.org/abs/2404.06654)**: Long-context retrieval evaluation; motivates measuring retrieval robustness.
- **[HELMET](https://arxiv.org/abs/2410.02694)**: Long-context evaluation protocol; includes test-time learning evaluation ideas.
- **[In-context Learning with Long-Context Models: An In-Depth Exploration](https://arxiv.org/abs/2405.00200)**: TTL-style ICL analysis; motivates evaluation designs.
- **[LoCoMo](https://arxiv.org/abs/2402.17753)**: Long-term conversational memory benchmark; highlights failures in very long interactive histories.
- **[LongMemEval](https://arxiv.org/abs/2410.10813)**: Long interactive memory benchmark.
- **[Memento: Fine-tuning LLM Agents without Fine-tuning LLMs](https://arxiv.org/abs/2508.16153)**: Non-parametric agent adaptation; different from utility-based memory.
- **[Mem^p: Exploring Agent Procedural Memory](https://arxiv.org/abs/2508.06433)**: Studies procedural memory build/retrieve/update and notes open questions about when to deprecate obsolete memories.
- **[Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework](https://arxiv.org/abs/2508.16629)**: Jointly optimizes retrieval/utilization/storage under the “memory cycle effect”; mostly focuses on retrieval/utilization rather than explicit eviction.
- **[Auer et al., 2002 (UCB1)](https://dl.acm.org/doi/10.1145/1008731.1008739)**: Classic upper-confidence-bound algorithm for exploration under uncertainty; conceptually motivates confidence-based retention.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Similarity-only retrieval (RAG/ICL) | Retrieve by lexical/dense similarity; keep-all or heuristic eviction | RAG, BM25/Contriever baselines in MemoryAgentBench | MemoryAgentBench TTL-MCC; long-context suites | Cannot distinguish harmful vs helpful memories beyond similarity |
| Value-aware retrieval | Learn per-memory utility and re-rank retrieval | MemRL | MemRL benchmarks; MemoryAgentBench-style TTL | Often lacks explicit capacity management; credit assignment can be hard |
| Heuristic write/delete policies | Add selectively and delete periodically/history-based | How Memory Management Impacts LLM Agents | Domain agents (EHR/driving/IoT) | Requires an evaluator; may be domain-specific |
| Learned memory managers | Treat memory edits as actions; train end-to-end | Memory as Action (MemAct), MemSearcher | QA/search tasks | More complex training; less interpretable |
| **Uncertainty-aware eviction (ours)** | Evict items that are *confidently* low-utility | This proposal | TTL classification under memory cap | Needs enough repeated retrieval to estimate uncertainty |

### Closest Prior Work
1. **MemRL**: learns utilities for value-aware retrieval and updates them online, but does not propose a concrete pruning/eviction policy for bounded memory.
2. **How Memory Management Impacts LLM Agents**: shows deletion helps and proposes history-based deletion, but does not use value-aware retrieval/Q-values as the retrieval primitive.
3. **Mem0**: provides production CRUD memory decisions but is not framed as utility learning with online statistical confidence.
4. **AGENT KB**: mentions utility-based eviction in a cross-framework KB, but does not isolate uncertainty-aware eviction nor test it in a controlled, capacity-limited online setting.

**Novelty Kill Search Summary:** Searched locally across all finalized proposals and all agents’ in-progress drafts for “MemRL + (prune/retire/quarantine/delete/evict)” and found no existing proposal targeting MemRL-style deletion/retirement. Ran multiple web queries for “MemRL memory pruning/eviction/capacity management” and “Q-value/utility-guided memory pruning for LLM agents”; no explicit MemRL follow-up paper implementing Q-guided retirement was found as of 2026-03-02 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| MemRL | Utility-aware re-ranking of retrieved memories | No explicit eviction; memory can grow / crowd out under caps | Add confidence-based eviction under a fixed memory budget | Removes confidently bad memories while keeping uncertain (rare) ones longer |
| How Memory Management Impacts | Studies selective addition + deletion heuristics | Not value-aware retrieval; deletion signals can be evaluator-dependent | Use value-aware retrieval + automated correctness reward | Eviction uses only automated success/failure, no extra judge |
| Mem0 | CRUD memory updates for production assistants | Not utility learning; policies not confidence-calibrated | Add Bayesian uncertainty to eviction | More robust under sparse evidence |
| Learned memory managers (MemAct/MemSearcher) | End-to-end RL for memory edits | Training complexity; not minimal/interpretable | Post-hoc drop-in eviction rule | Easier to adopt, audit, and verify |

---

## Experiments

### Experimental Setup

**Benchmarks.** We use TTL-style intent classification streams based on MemoryAgentBench’s TTL-MCC datasets:
- **BANKING77** (77 intent labels)
- **CLINC150** (150 intent labels)

(These are explicitly listed as TTL-MCC datasets in MemoryAgentBench; see `./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/sections/Datasets for Test-Time Learning (TTL).md`.)

**Stream construction (verification-ready).** For each dataset:
- Sample a stream of `T` labeled utterances `(x_t, y_t)` (e.g., `T=500`) with a fixed RNG seed.
- Each step is both a query (agent must predict `y_t`) and then a memory write (store `(x_t, y_t)` after reward is computed).
- Controlled corruption: when writing to memory, flip the stored label with probability `p` (e.g., `p=0.1`) to create harmful memories.

**Baseline Ladder (REQUIRED):**
- Level 0: Majority-class baseline (no LLM).
- Level 1: Zero-shot LLM classification (no retrieved demo).
- Level 2: Similarity-only retrieval with FIFO eviction under budget `M`.
- Level 3: Inference-time scaling baseline: self-consistency over `N` decodes for zero-shot (e.g., `N=5`) with majority vote (optional; include only if it meaningfully changes accuracy).
- Level 4: Value-aware retrieval + **evict-lowest posterior mean** `Q_hat`.
- Level 5: Value-aware retrieval + **CBME evict-lowest-UCB** (ours).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3.1-8B-Instruct (or similar) | ~8B | https://huggingface.co/meta-llama | Local inference; constrain output to label ID |
| Qwen2.5-7B-Instruct (optional second model) | ~7B | https://huggingface.co/Qwen | Improves generalizability; same prompts |

**Other Resources:**
- MemoryAgentBench dataset: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
- MemoryAgentBench code: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- BANKING77: https://huggingface.co/datasets/banking77
- CLINC150: https://huggingface.co/datasets/clinc_oos

**Resource Estimate (order-of-magnitude).**
Assume `T=500`, `M=200`, two datasets.
- LLM calls per step: 1 (single-label output). Majority baseline uses 0.
- Total calls per method per dataset per seed: `T`.
- If evaluating 4 LLM-based methods (zero-shot, similarity, evict-mean, evict-UCB): `2 datasets * 500 * 4 = 4000` calls per seed.
- `3 seeds` → `12k` calls per model.
- With 2 base models → `24k` total calls.
Given short prompts (one demo + query + label list) and 7–8B models, this should fit comfortably within the 768 A100 GPU-hour budget.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| BANKING77 stream | Online intent classification under memory cap | (1) online accuracy, (2) last-100 accuracy, (3) poisoned@1 retrieval rate, (4) memory size and poison fraction | test-derived stream | HF links above | Custom wrapper over MemoryAgentBench/HF |
| CLINC150 stream | Same as above | Same metrics | test-derived stream | HF links above | Custom wrapper over MemoryAgentBench/HF |

### Main Results

We report (i) **online accuracy** over the full stream, (ii) **last-100 accuracy** as a late-stage metric, and (iii) **Poisoned@1**, defined as the fraction of queries where the single retrieved memory item is a mislabeled (poisoned) example.

#### Results Table

| Method | Base Model | Benchmark | Online Acc (mean±std) | Last-100 Acc (mean±std) | Poisoned@1 (mean±std) | Active Mem Size | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| Majority | N/A | BANKING77 | TBD | TBD | N/A | 0 | - | no LLM |
| Zero-shot | 7–8B instruct | BANKING77 | TBD | TBD | N/A | 0 | - | no retrieval |
| Similarity + FIFO | 7–8B instruct | BANKING77 | TBD | TBD | TBD | M | - | memory cap |
| Value-aware + evict-mean | 7–8B instruct | BANKING77 | TBD | TBD | TBD | M | - | point utility |
| **Value-aware + evict-UCB (CBME)** | 7–8B instruct | BANKING77 | TBD | TBD | TBD | M | - | ours |
| Majority | N/A | CLINC150 | TBD | TBD | N/A | 0 | - | no LLM |
| Zero-shot | 7–8B instruct | CLINC150 | TBD | TBD | N/A | 0 | - | no retrieval |
| Similarity + FIFO | 7–8B instruct | CLINC150 | TBD | TBD | TBD | M | - | memory cap |
| Value-aware + evict-mean | 7–8B instruct | CLINC150 | TBD | TBD | TBD | M | - | point utility |
| **Value-aware + evict-UCB (CBME)** | 7–8B instruct | CLINC150 | TBD | TBD | TBD | M | - | ours |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| CBME (full) | 95% UCB eviction | Best late-stage acc and lower poison fraction |
| CBME with 90% UCB | Change confidence level | Similar trend; too low may behave like mean |
| CBME w/o n_min guard | Allow evict among never-used items | Potential instability early; quantify |

### Experimental Rigor

- **Seeds**: `seeds=[42, 123, 456]` controlling stream sampling + label-corruption RNG.
- **Decoding**: greedy decoding with strict label-ID output; deterministic when model/inference stack supports it.
- **Sanity checks**:
  - With `p=0`, CBME should not underperform evict-mean by more than noise.
  - With large `p` (e.g., 0.3), similarity-only should degrade and CBME should reduce poison fraction.
- **Confounders and controls**:
  - *Prompt sensitivity*: keep prompts identical across methods.
  - *Compute mismatch*: all value-aware methods use identical LLM call budget (1 per step).
  - *Data leakage*: pretrained models may have seen these datasets, but comparisons are within-model across memory policies.

---

## Success Criteria

**Hypothesis** (directional): CBME (evict-lowest-UCB) improves late-stage accuracy and reduces poisoned-memory retrieval relative to evict-lowest-mean under the same memory budget.

**Decision Rule** (concrete):
- **Proceed** if CBME improves **Last-100 accuracy** by ≥3 absolute points over evict-mean on **both** BANKING77 and CLINC150 (mean over 3 seeds) for at least one base model, *and* does not increase Poisoned@1.
- **Pivot** if CBME reduces Poisoned@1 but does not improve accuracy: increase stream length `T` (more evidence per memory) or lower memory cap `M` (more pressure) and re-test.
- **Refute** if CBME fails to outperform evict-mean beyond overlapping std on both datasets for both base models, or if it consistently worsens accuracy (suggesting uncertainty-aware eviction is not beneficial in this regime).

---

## Impact Statement

If successful, CBME provides a drop-in, interpretable memory eviction policy for value-aware retrieval agents operating under realistic memory budgets. Practitioners deploying long-lived assistants (customer support, enterprise copilots, tool-using agents) could replace heuristic eviction or point-estimate utility eviction with confidence-based eviction to reduce long-term degradation from harmful memories while retaining coverage of rare but important cases.

---

## References

- [MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory](./references/MemRL-MEMRL-SELF-EVOLVING-AGENTS-VIA-RUNTIME-REINFORCEMENT-LEARNING-ON-EPISODIC-MEMORY/meta/meta_info.txt) - 2026
- [How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior](./references/How-Memory-Management-Impacts-LLM-Agents-An-Empirical-Study-of-Experience-Following-Behavior/meta/meta_info.txt) - 2025
- [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions (MemoryAgentBench)](./references/Evaluating-Memory-in-LLM-Agents-via-Incremental-Multi-Turn-Interactions/meta/meta_info.txt) - 2026
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) - 2025
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) - 2023
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - 2020
- [Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) - 2023
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - 2023
- [A-Mem: Agentic Memory for LLM Agents](https://openreview.net/forum?id=FiM0M8gcct) - 2025
- [AGENT KB: Leveraging Cross-Domain Experience for Agentic Problem Solving](https://openreview.net/pdf?id=QCLXVOMkl4) - 2026
- [MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning](https://arxiv.org/abs/2511.02805) - 2025
- [Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2510.12635) - 2025
- [HippoRAG](https://arxiv.org/abs/2405.14831) - 2024
- [From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2)](https://openreview.net/forum?id=LWH8yn4HS2) - 2025
- [RULER: What’s the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) - 2024
- [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly](https://arxiv.org/abs/2410.02694) - 2024
- [In-context Learning with Long-Context Models: An In-Depth Exploration](https://arxiv.org/abs/2405.00200) - 2024
- [Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo)](https://arxiv.org/abs/2402.17753) - 2024
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) - 2024
- [Memento: Fine-tuning LLM Agents without Fine-tuning LLMs](https://arxiv.org/abs/2508.16153) - 2025
- [Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework](https://arxiv.org/abs/2508.16629) - 2025
- [Auer et al., 2002: Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)](https://dl.acm.org/doi/10.1145/1008731.1008739) - 2002
