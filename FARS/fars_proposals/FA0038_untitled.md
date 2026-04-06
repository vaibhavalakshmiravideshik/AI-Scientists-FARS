# untitled

# Citation-Consistent Voting for Permutation-Robust Retrieval-Augmented Generation

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Retrieval-augmented generation (RAG) systems improve factual question answering by conditioning a language model on a small set of retrieved documents (e.g., Top-5 passages from Wikipedia) rather than relying only on parametric memory. This design is widely used in assistants and enterprise search because it can incorporate up-to-date knowledge, provide some form of provenance, and be updated without retraining the generator.

However, modern RAG deployments rarely have a single stable “context”. The same query can be served with slightly different document **orders** due to retriever updates, caching, load-balancing, or indexing changes. If the generator is sensitive to the ordering of the retrieved set, then the system can become unreliable even when retrieval is correct.

Recent work has shown that this order sensitivity is not a minor prompting artifact: **Stable-RAG** identifies “retrieval-permutation-induced hallucinations,” where reordering the *same* Top-5 passages can cause the model to output different answers, including confident wrong answers, even in short contexts (Top-5 is typically <1k tokens) [Stable-RAG](https://arxiv.org/abs/2601.02993). Stable-RAG addresses this via training with **Direct Preference Optimization (DPO)** to make the generator invariant to permutations, but training is often infeasible for deployed systems that depend on a fixed base model.

### The Problem

A natural training-free alternative is inference-time scaling: run the generator on multiple document-order permutations and aggregate the outputs. **MoI** (“Inference Scaling for Bridging Retrieval and Augmented Generation”) explicitly discusses a train-free baseline that uses **Mixture-of-Agents (MoA)** / self-consistency aggregation over **permuted retrieved orders** [MoI](https://arxiv.org/abs/2412.10684), building on self-consistency for chain-of-thought reasoning [Self-Consistency](https://openreview.net/forum?id=1PL1NIMMrw) and mixture-of-agents style aggregation [MoA](https://arxiv.org/abs/2406.04692). In MoI’s QA experiments with LLaMA-3-8B, self-consistency improves **HotpotQA EM** from **48.54** (random ordering) to **51.72** (Table 1 in MoI, §4.2), indicating that permutation ensembling can be a non-trivial accuracy knob.

But simple **answer-frequency voting** is a weak signal in multi-document QA:

- It treats two outputs as different if they differ lexically, even if they are semantically identical.
- More importantly, it ignores *evidence*. A wrong answer can be frequent if the model repeatedly follows the same spurious reasoning path under many permutations.

At the same time, RAG systems increasingly require **citations / provenance** (doc IDs, quotes) for auditability. Yet citation behavior is itself unreliable: models can produce unfaithful or fabricated citations, and “correctness is not faithfulness” is a known issue in RAG attributions.

This proposal asks:

> When we already spend K inference calls to sample multiple document orders, can we use **cross-permutation agreement on the cited underlying document** as a better selection signal than answer-frequency voting?

### Key Insight and Hypothesis

**Key insight:** In permutation-sensitive RAG, hallucinated answers often arise from unstable or weakly grounded use of the retrieved set. If we force each run to provide a minimal, machine-checkable citation (a document index plus a verbatim quote), then the *same* correct answer should tend to be supported by the *same underlying document* across permutations, while hallucinated answers should show lower citation validity and/or lower agreement on the cited underlying document.

**Hypothesis:** Given K permuted RAG runs over the same retrieved set, selecting the answer with the strongest **citation-consistency score** (frequency of evidence-valid runs that cite the same underlying document) will improve QA accuracy and reduce answer instability compared to standard MoA majority voting at the same K, without fine-tuning.

The most likely failure mode is that the model is **consistently wrong**: it may repeatedly produce the same incorrect answer while also repeatedly citing the same (wrong) document. The experiment includes diagnostics (evidence-validity rate and citation-agreement rate) to determine whether this failure mode dominates.

---

## Proposed Approach

### Overview

We propose a training-free aggregation rule for permutation ensembles in multi-document RAG:

1. Retrieve Top-N documents (N=5) for a query.
2. Sample K permutations of these N documents.
3. For each permutation, run the generator with a prompt that requires a strict JSON output containing:
   - a short answer span,
   - a document index (1..N) within the *presented* order,
   - a short verbatim quote copied from that document.
4. Filter out runs whose quote is not an exact substring of the cited document (and/or where the answer is not contained in the quote).
5. Aggregate remaining runs by **underlying document agreement** (mapping the cited index back to the original retrieved document ID, independent of permutation).

### Method Details

#### Per-permutation generation format
For each permutation, we prompt the model to output:

```json
{
  "answer": "<short answer string>",
  "evidence_doc_idx": <integer 1..5>,
  "evidence_quote": "<verbatim substring from that document, <= 200 chars>"
}
```

To keep evaluation automatic and robust to formatting errors, we enforce:
- JSON-only output (no additional text).
- `evidence_doc_idx` must be in [1..5].
- `evidence_quote` length capped (e.g., 200 characters).

#### Evidence validity checks (fully automated)
Given the retrieved documents’ raw text:
- **Quote validity**: `evidence_quote` must be an exact substring of the selected document’s text.
- **Answer-in-quote**: normalized `answer` must be a substring of normalized `evidence_quote`.

These checks ensure the model cannot cite nonexistent text and that the cited quote directly contains the answer.

#### Mapping citations across permutations
Because the retrieval set is fixed for a query, we assign each document a stable **underlying document ID** (e.g., its position in the original retriever order or a hash of the document text).

For a run with permutation \(\pi\) and `evidence_doc_idx = j`, we map to the underlying document ID:
\(\mathrm{docID}(r) = \mathrm{docID}_{\text{underlying}}(\pi[j])\).

#### Citation-consistent voting
Let \(a\) be a normalized answer string.

- Let \(R(a)\) be the set of runs producing answer \(a\) that pass evidence validity.
- Let \(\mathrm{doc}^*(a)\) be the modal underlying document ID among \(R(a)\).
- Define:

\[
\mathrm{score}(a) = \#\{ r \in R(a) : \mathrm{docID}(r) = \mathrm{doc}^*(a) \}.
\]

We output \(\arg\max_a \mathrm{score}(a)\).

Tie-breakers (deterministic):
1) higher raw frequency of \(a\) across all K runs;
2) higher average token log-probability of the `answer` under the generating run (if available from the inference backend).

#### Relationship to existing baselines
- **MoA/self-consistency** uses answer-frequency voting across permutations.
- Our method uses the same K runs, but replaces the aggregation rule with an evidence-aware, permutation-invariant signal (underlying docID agreement).

### Key Innovations

- **Permutation-invariant evidence agreement**: use agreement over **underlying cited documents** (not just answer strings) as the voting signal.
- **Machine-checkable citations**: enforce a strict, automatically verifiable quote constraint, enabling fully automated evaluation and reducing reliance on LLM-as-a-judge.
- **Training-free**: no fine-tuning of retriever or generator; only inference-time aggregation.

---

## Related Work

### Field Overview

RAG combines retrieval (BM25/DPR-style dense retrieval) with an LLM reader/generator to answer knowledge-intensive questions [RAG](https://arxiv.org/abs/2005.11401), [DPR](https://arxiv.org/abs/2004.04906). Multi-document RAG introduces additional failure modes beyond retrieval recall: distractor documents can degrade performance even at fixed total context length, indicating that document segmentation and distractor profiles matter [More-Documents-Same-Length](https://arxiv.org/abs/2503.04388). Separately, long-context models show strong positional biases (“lost in the middle”), motivating both positional encoding interventions and document re-ordering methods [Lost-in-the-Middle](https://doi.org/10.1162/tacl_a_00638).

Stable-RAG shows that even in short contexts (Top-5), merely permuting the order of retrieved documents can induce qualitatively different reasoning trajectories and wrong answers, motivating explicit permutation robustness [Stable-RAG](https://arxiv.org/abs/2601.02993). Training-based approaches can reduce such sensitivity (e.g., Stable-RAG via DPO), but training is often infeasible in production. Inference-time scaling and aggregation (self-consistency, mixture-of-agents) provides a complementary training-free knob, but typical aggregation uses answer-frequency voting and does not explicitly incorporate evidence agreement.

Finally, citation and attribution are becoming first-class requirements for RAG systems, but correctness and faithfulness can diverge, and citation hallucination remains common. This motivates selectors that use **mechanically verifiable** provenance signals rather than post-hoc rationalized citations.

### Related Papers

- **[Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](https://arxiv.org/abs/2601.02993)**: Identifies permutation-induced hallucinations in Top-5 RAG and mitigates them via hidden-state clustering + DPO (training-based).
- **[Inference Scaling for Bridging Retrieval and Augmented Generation (MoI)](https://arxiv.org/abs/2412.10684)**: Uses multiple passage-order interventions to debias position and includes MoA/self-consistency aggregation baselines over permutations.
- **[Self-Consistency Improves Chain of Thought Reasoning](https://openreview.net/forum?id=1PL1NIMMrw)**: Introduces self-consistency (sample multiple reasoning paths, majority vote) as an inference-time reliability method.
- **[Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)**: Studies parallel multi-agent generation and aggregation to improve output quality.
- **[Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models](https://arxiv.org/abs/2310.07712)**: Uses shuffling + aggregation to reduce positional bias in LLM-based ranking (not QA answer selection).
- **[Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards (Con-RAG)](https://arxiv.org/abs/2510.04392)**: Trains generators to be consistent across paraphrased queries using group similarity rewards (training-based).
- **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)**: Foundational RAG framework for conditioning generation on retrieved documents.
- **[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)**: Dense retrieval baseline widely used for open-domain QA.
- **[Contriever: Unsupervised Dense Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118)**: Unsupervised dense retriever used in many RAG pipelines.
- **[Lost in the Middle: How Language Models Use Long Contexts](https://doi.org/10.1162/tacl_a_00638)**: Establishes positional bias (“lost in the middle”) in long contexts.
- **[MEGA-RAG: a retrieval-augmented generation framework with multi-source evidence retrieval and semantic-evidential alignment evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12540348/)**: Proposes a semantic-evidential alignment score (SEAE) to detect divergence across candidate answers; closest on “evidence alignment as a selection signal”, but in a domain RAG pipeline (public health) and not tied to document-order permutation ensembles.
- **[Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization](https://arxiv.org/abs/2406.16008)**: Calibrates positional attention bias in long contexts to improve retrieval and downstream QA when evidence is placed in the middle.
- **[Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding (Ms-PoE)](https://arxiv.org/abs/2403.04797)**: Proposes multi-scale positional encoding as a plug-and-play method to mitigate lost-in-the-middle positional bias without fine-tuning.
- **[Position Bias Mitigates Position Bias: Inter-Position Knowledge Distillation (Pos2Distill)](https://aclanthology.org/2025.emnlp-main.78/)**: Mitigates position bias via distillation across positions.
- **[Making Retrieval-Augmented Language Models Robust to Irrelevant Context (RetRobust)](https://arxiv.org/abs/2310.01558)**: Studies how irrelevant retrieved passages harm RAG and proposes training-time robustness to noisy retrieval.
- **[Enhancing Noise Robustness of Retrieval-Augmented LMs with Adversarial Training (RAAT)](https://aclanthology.org/2024.acl-long.540/)**: Uses adversarial training to improve robustness to noisy retrieval.
- **[ATM: Adversarial Tuning Multi-Agent System Makes a Robust RAG](https://aclanthology.org/2024.emnlp-main.610/)**: Multi-agent adversarial tuning to improve RAG robustness.
- **[More Documents, Same Length: Isolating the Challenge of Multiple Documents in RAG](https://arxiv.org/abs/2503.04388)**: Shows multi-document clutter can degrade performance even at fixed token budget.
- **[CITE-WHILE-YOU-GENERATE: Training-Free Evidence Attribution for Multimodal Clinical Summarization](https://arxiv.org/abs/2601.16397)**: Uses attention-based attribution during generation to produce training-free fine-grained citations.
- **[Correctness is not Faithfulness in RAG Attributions](https://arxiv.org/abs/2412.18004)**: Argues citation correctness can be high while citation faithfulness is low due to post-rationalization, motivating stricter provenance signals.
- **[Guided Decoding and Its Critical Role in Retrieval-Augmented Generation](https://arxiv.org/abs/2509.06631)**: Studies guided decoding backends for schema/format constraints in RAG, relevant to enforcing strict JSON outputs.
- **[Ranked Voting based Self-Consistency of Large Language Models](https://arxiv.org/abs/2505.10772)**: Improves self-consistency by using ranked voting rules (IRV/Borda/MRR) instead of simple majority.
- **[Ranking Free RAG: Replacing Re-ranking with Selection in RAG for Sensitive Domains (METEORA)](https://arxiv.org/abs/2505.16014)**: Uses rationale-driven evidence selection and verification; closest in spirit on “evidence-aware selection”, but does not study document-order permutation ensembling.
- **[GOLD PANNING: Iterative Bayesian Signal Anchoring for Many-Document Needle-in-Haystack Reasoning](https://arxiv.org/abs/2510.09770)**: Uses iterative Bayesian belief tracking and reordering to exploit positional diagnosticity in long contexts; operates on reordering/reranking rather than answer aggregation.


### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Training-time permutation robustness | Train model to be invariant to document permutations | Stable-RAG | NQ / TriviaQA / HotpotQA; SubEM/F1; PSR (Perturbation Success Rate) | Requires fine-tuning; training cost |
| Inference-time permutation aggregation | Run multiple orderings and aggregate outputs | MoI (MoA baseline), self-consistency | HotpotQA EM, MS MARCO ROUGE-L, etc. | Aggregation signal often weak (answer-frequency), can be costly |
| Position-bias mitigation | Modify positional encoding/attention to reduce lost-in-the-middle | Lost-in-the-middle; Ms-PoE; Pos2Distill | Long-context tasks; QA | Does not directly target permutation-induced hallucinations |
| Citation grounding / evaluation | Make outputs auditable with citations; evaluate faithfulness | CITE-WHILE-YOU-GENERATE; correctness-vs-faithfulness | citation precision/recall; groundedness | Citations can be unfaithful; evaluation is hard |

### Closest Prior Work

1. **Stable-RAG (2026)**: Addresses the same failure mode (permutation-induced hallucinations) but uses training (DPO) after clustering hidden states across permutations. Our approach targets the same instability but stays training-free and uses citation agreement as the selection signal.

2. **MoI (2024)**: Uses multiple permutations as “interventions” to estimate positional bias and rerank passages; it also defines a train-free MoA/self-consistency baseline that aggregates answers by consistency. Our work is complementary: we do not attempt reranking, but instead improve the aggregation rule by incorporating verifiable evidence agreement.

3. **Self-Consistency (2023)** and **Ranked-voting self-consistency (2025)**: These focus on reasoning-path sampling and answer voting (and improved voting rules), but they do not condition the vote on cross-run evidence agreement, and they are not designed for document-order permutations in RAG.

4. **Citation-grounding methods**: These aim to improve single-run attribution or evaluate faithfulness, but they do not use **cross-permutation citation agreement** as a selector under fixed retrieved sets.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Stable-RAG | Training-based permutation robustness via hidden-state clustering + DPO | Requires fine-tuning and preference data | No training; inference-time aggregation only | Lower barrier to deployment; can be applied to frozen models |
| MoA/self-consistency (as discussed in MoI) | Permute docs; majority vote on answers | Uses answer frequency only; ignores evidence | Weight votes by verifiable citation agreement | Evidence agreement should correlate with grounded correctness |
| MoI | Uses permutations to estimate position bias and rerank contexts | Goal is reranking, not permutation-hallucination selection | Keep retrieval fixed; focus on answer selection | Directly targets output instability rather than retrieval ordering |
| Ranked voting SC | Improves voting using ranked ballots | Still ignores evidence grounding in RAG | Use doc-ID agreement + quote validity | Grounds aggregation in retrieved evidence |

---

## Experiments

### Experimental Setup

**Pilot + early-stop plan (to make verification decisive):**
1. **Pilot** on a random subset of **200 NQ queries** with fixed Top-5 retrieval to measure:
   - JSON parse success rate
   - quote-validity rate (exact substring)
   - baseline MoA (majority vote) SubEM/F1
2. **Early-stop gate:** if quote-validity rate < **50%** (with a reasonable prompt), or if our method fails to beat MoA on the pilot, stop and **refute** (we do **not** proceed with a weakened selector in this proposal; that would be a different paper).
3. If the pilot passes, run the full NQ test (primary), then optionally TriviaQA/HotpotQA.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen/Qwen3-8B | 8B | https://huggingface.co/Qwen/Qwen3-8B | Matches Stable-RAG backbone family; fits in single A100 80GB |
| Qwen/Qwen2.5-7B-Instruct (optional) | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Optional generalization check |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| N/A | Inference only | - | - | - |

**Other Resources (if applicable):**
- Stable-RAG evaluation code + any released fixed Top-5 retrieval outputs (GitHub: https://github.com/zqc1023/Stable-RAG).
- Wikipedia corpus (if retrieval must be reproduced).

**Resource Estimate (evidence-based back-of-the-envelope)**:
- Let \(Q\) be #queries evaluated, \(K\) permutations per query (default K=5), and \(T\) total tokens processed per run (prompt+context prefill + generation).
- Total tokens \(\approx Q \times K \times T\).
- Example (order-of-magnitude): if \(Q=3{,}000\) (NQ-scale), \(K=5\), \(T\approx 800\) tokens/run, then total \(\approx 12\)M tokens.
- On a single A100-80GB with vLLM-like serving, this is typically on the order of **tens of GPU-hours** for one dataset; running all 3 datasets could be **O(100 GPU-hours)**. We will **run NQ first** (primary), then expand if promising.
- **GPU memory**: 1×A100 80GB per model replica for Qwen3-8B.
- **API usage**: none required.

### Benchmarks and Metrics

We follow Stable-RAG’s evaluation and report **Substring Exact Match (SubEM)** (prediction is counted correct if the gold answer appears as a substring) and token-level **F1**.

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| NaturalQuestions (NQ) **(primary)** | Open-domain QA with short answers from Wikipedia | SubEM, F1 | test | https://aclanthology.org/Q19-1026/ | Stable-RAG eval / standard scripts |
| TriviaQA (optional) | Open-domain QA | SubEM, F1 | test | https://aclanthology.org/P17-1147/ | Stable-RAG eval / standard scripts |
| HotpotQA (optional) | Multi-hop QA | SubEM, F1 | test | https://aclanthology.org/D18-1259/ | Stable-RAG eval / standard scripts |

**Evaluation Scripts:**
- Prefer using Stable-RAG’s released code and evaluation harness for SubEM/F1 to match their protocol.
- If Stable-RAG code is incomplete, use standard open-domain QA evaluation scripts; document prompt format changes.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | SubEM | F1 | Source | Notes |
|--------|------------|-----------|------:|---:|--------|------|
| Vanilla RAG (K=1, standard prompt) | Qwen3-8B | NQ (Contriever Top-5) | 44.65 | 45.34 | Stable-RAG Table 2 (§5.2) | Community baseline; **not** JSON-constrained |
| MoA majority vote over answers (K=5 permutations) | Qwen3-8B | NQ | **TBD** | **TBD** | - | Our reimplementation (same K as ours) |
| MoA + quote-validity filter, then majority vote | Qwen3-8B | NQ | **TBD** | **TBD** | - | Control: isolates gains from enforcing verifiable quotes |
| **Ours: citation-consistent voting (K=5)** | Qwen3-8B | NQ | **TBD** | **TBD** | - | Evidence-validity + underlying docID agreement |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Majority vote after filtering evidence-invalid runs | Remove docID agreement; keep only evidence-valid outputs | If close to full method, gains come mainly from forcing valid quotes |
| K sweep (K=1,3,5,7) | Vary number of permutations | Diminishing returns; K=3 captures most gain if hypothesis holds |
| Remove quote requirement (docID only) | Prompt returns doc idx without quote substring check | If performance similar, substring verification may be unnecessary |

### Analysis (Optional)

- **Citation validity rate**: fraction of runs passing the exact substring checks; if low, method is bottlenecked by citation compliance.
- **Underlying-doc agreement vs accuracy**: correlation between doc agreement and correctness to test whether the signal is meaningful.
- **Answer instability**: number of unique normalized answers across permutations per query, and whether our selector reduces wrong-answer selection on high-instability queries.

---

## Success Criteria

**Criterion 1: Evidence agreement improves selection beyond frequency voting**
- Hypothesis: At fixed K (e.g., K=5), citation-consistent voting yields higher SubEM/F1 than MoA majority voting.
- **Decision rule (primary):** On NaturalQuestions test with fixed Top-5 retrieval, our method beats MoA by **≥ +1.0 SubEM** (absolute) *and* beats the “filter invalid citations then majority vote” ablation by **≥ +0.5 SubEM**. If either margin is not met, we consider the core claim **not supported** (refute or re-scope to “quote filtering” only).
- **Uncertainty / variance plan:** run **2 random seeds** for permutation sampling (different permutation draws) and report mean±std; if the sign of improvement flips across seeds, treat as failure.

**Criterion 2: The method reduces permutation-induced instability in practice**
- Hypothesis: Queries with high answer disagreement across permutations benefit more from evidence-aware aggregation.
- Validation: Performance gains concentrate on the subset of queries with many unique answers across permutations, and diagnostics show higher underlying-doc agreement for correct predictions than for incorrect ones.

---

## Impact Statement

If validated, this work provides a simple, training-free way to make multi-document RAG systems more reliable under unavoidable document-order variation. It would change engineering practice for citation-heavy assistants by suggesting that, when spending multiple inference calls, aggregation should prefer answers with **stable, verifiable provenance** rather than answers that are merely frequent.

---

## References

- [Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](https://arxiv.org/abs/2601.02993) - Zhang et al., 2026
- [Inference Scaling for Bridging Retrieval and Augmented Generation](https://arxiv.org/abs/2412.10684) - Lee et al., 2024
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://openreview.net/forum?id=1PL1NIMMrw) - Wang et al., 2023
- [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692) - Wang et al., 2024
- [Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models](https://arxiv.org/abs/2310.07712) - Tang et al., 2024
- [Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards](https://arxiv.org/abs/2510.04392) - Hamman et al., 2025
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Lewis et al., 2020
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) - Karpukhin et al., 2020
- [HotpotQA: A Dataset for Diverse Explainable Multi-Hop Question Answering](https://aclanthology.org/D18-1259/) - Yang et al., 2018
- [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://aclanthology.org/P17-1147/) - Joshi et al., 2017
- [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/) - Kwiatkowski et al., 2019
