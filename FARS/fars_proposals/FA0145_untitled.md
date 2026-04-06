# untitled

# Disagreement-Gated KV Cache Reuse for Shuffle-Robust LLM Judges

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Many LLM systems generate multiple candidate answers (e.g., from multiple agents, prompts, or decoding seeds) and then use an LLM judge to select the best candidate. This “multi-candidate judging” pattern is increasingly used in code generation (unit-test based selection), reasoning, and multi-agent collaboration.

A major practical bottleneck is **prefill latency**: judges typically re-encode a long prompt containing the user query and all candidate responses. Recent work on **KV cache reuse** aims to accelerate inference by reusing key/value tensors for repeated or overlapping text segments, avoiding recomputation of the full prefill.

However, **judge-side** KV reuse is not behavior-preserving. *When KV Cache Reuse Fails in Multi-Agent Systems* (arXiv:2601.08343) shows that common reuse strategies can substantially change which candidate the judge selects, even when end-task accuracy looks similar. This is a reliability issue for evaluation, auditing, and production multi-agent systems.

### The Problem

The paper introduces **Judge Consistency Rate (JCR)**: the fraction of examples where a KV-reuse judge selects the same candidate as a **dense-prefill** judge under the same candidate ordering (and for shuffle, using the same permutation and mapping indices back).

Under candidate-order perturbation (**shuffle**), JCR drops to near-random choice for several reuse methods. For example, on HumanEval with Progressive Refinement (N=4 candidates, Llama-3.2-3B-Instruct), KVCOMM’s JCR drops from **57.14 → 21.74** under shuffle, and Naïve reuse drops from **64.60 → 23.60** (Table 1; see `./references/When-KV-Cache-Reuse-Fails-in-Multi-Agent-Systems-Cross-Candidate-Interaction-is-Crucial-for-LLM-Judges/sections/Overview judge decisions are not invariant under KV reuse..md`). This suggests that judge decisions depend on delicate **cross-candidate interactions** that current reuse methods disrupt.

A natural deployment response is **risk-aware fallback**: use KV reuse when it is “safe”, and fall back to dense prefill otherwise. The same paper suggests detecting “universally safe” instances with a classifier (AUC≈0.82), but the feature set and training protocol are not fully specified, and it requires building a supervised gate.

### Key Insight and Hypothesis

**Key insight:** for a fixed candidate set, if two different KV-reuse approximations induce different hidden-state perturbations, then their **disagreement** can act as a cheap instability signal. In particular:

- **Naïve stitched reuse** (RoPE re-index + stitching execution-side candidate KV chunks) and **KVCOMM** (anchor-offset correction + partial reuse) implement different approximations to judge-side cache construction.
- If the dense judge’s winner has a large “margin” (one candidate clearly best), both approximations will often still pick the same winner.
- If the winner margin is small, small perturbations flip rankings, so the approximations are more likely to **disagree**.

**Hypothesis:** On shuffled judge prompts, conditioning reuse on **agreement(Naïve, KVCOMM)** yields a high-precision “safe reuse” detector (precision ≥ 0.90), enabling an **agreement-gated dense fallback** that materially increases shuffle-JCR while keeping non-trivial reuse coverage.

This could be wrong if Naïve and KVCOMM fail in strongly correlated ways (agreeing on wrong winners), or if agreement is too rare (coverage too low to matter).

---

## Proposed Approach

### Overview

We propose **Disagreement-Gated Judge KV Reuse (DG-JKR)**:

1. Run two fast judge passes on the same candidate set:
   - Naïve reuse judge
   - KVCOMM reuse judge
2. If both reuse judges select the same winning candidate, **accept** that winner (reuse).
3. Otherwise, **fallback** to a dense-prefill judge pass and output the dense winner.

This is training-free and uses only signals already produced by running existing reuse methods.

### Method Details

Let the dense-prefill judge pick winner \(i^*\), and the two reuse judges pick \(\hat{i}_{\text{naive}}\) and \(\hat{i}_{\text{kvcomm}}\).

Define:
- Disagreement rate: \(d = \Pr[\hat{i}_{\text{naive}} \neq \hat{i}_{\text{kvcomm}}]\).
- Conditional precision on agreement: \(p = \Pr[i^* = \hat{i}_{\text{naive}}\mid \hat{i}_{\text{naive}} = \hat{i}_{\text{kvcomm}}]\) (equivalently the agreed winner).

If DG-JKR falls back to dense on disagreement, the expected JCR of DG-JKR is:
\[
\text{JCR}_{\text{gated}} = d\cdot 1 + (1-d)\cdot p.
\]

**Runtime model:** if Naïve and KVCOMM are run sequentially, expected cost is approximately
\(T_{\text{naive}} + T_{\text{kvcomm}} + d\,T_{\text{dense}}\). If they are run in parallel, cost is \(\max(T_{\text{naive}},T_{\text{kvcomm}}) + d\,T_{\text{dense}}\). We will report both regimes and measure wall-clock.

### Key Innovations

1. **Training-free safety gating for judge-side KV reuse**: uses algorithmic disagreement as a conservative safety signal, avoiding a learned classifier.
2. **Predictiveness-first evaluation**: pre-registers precision/coverage thresholds (p, d) as the decisive test before claiming speedups.
3. **Signal-isolation baseline**: includes a matched-rate **random gating** control to show disagreement is informative beyond “sometimes falling back”.

---

## Related Work

### Field Overview

**LLM-as-a-judge and multi-agent candidate selection.** Many systems use LLM judges to select among candidates; however, judge behavior can be sensitive to formatting and candidate ordering. The KV-reuse failure mode paper formalizes this as JCR and shows that cross-candidate attention is necessary for invariance.

**KV cache reuse / prefix caching.** A large body of work accelerates inference by caching and reusing KV tensors for repeated prefixes, retrieval contexts, or shared segments across agents. Cross-context reuse (same text under different prefixes) is harder and motivates methods like KVCOMM and related anchor-based approaches.

**Selective recomputation and token selection.** Methods such as CacheBlend/CacheClip recompute only “high-deviation” or “important” tokens to trade accuracy for speed. Our proposal instead treats **dense prefill as a fallback** and asks when reuse is safe.

### Related Papers

- **[When KV Cache Reuse Fails in Multi-Agent Systems](https://arxiv.org/abs/2601.08343)**: Introduces JCR and shows judge-side reuse breaks decision invariance due to disrupted cross-candidate interaction.
- **[KVComm](https://arxiv.org/abs/2510.12872)**: Training-free cross-context KV reuse via anchor-based offset correction.
- **[KVFlow](https://openreview.net/forum?id=5Iw1nDtYmT)**: Caching policies for multi-agent workflows (execution-side), providing a systems baseline.
- **[PromptCache](https://arxiv.org/abs/2311.04934)**: Prompt-structure-aware KV reuse for repeated prompts/prefixes.
- **[KVLink](https://arxiv.org/abs/2502.16002)**: Cross-prefix KV reuse via position adjustment/link tokens (RAG-style).
- **[CacheBlend](https://arxiv.org/abs/2405.16444)**: Selective KV recomputation based on deviation (RAG acceleration).
- **[CacheClip](https://arxiv.org/abs/2510.10129)**: Auxiliary-model-guided token selection for KV recomputation.
- **[TokenLake](https://arxiv.org/abs/2508.17219)**: Segment-level prefix-cache pooling for repeated segments.
- **[Segment-Level KV Cache Sharing (OpenReview)](https://openreview.net/forum?id=kgzBkyqg6Z)**: Segment-level KV sharing beyond exact-prefix reuse.
- **[H2O](https://arxiv.org/abs/2306.14048)**: Token-importance selection for KV budget allocation (streaming).
- **[SnapKV](https://arxiv.org/abs/2404.14469)**: Prefill-time attention profiling for KV compression.
- **[Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)**: Uses agreement across samples as a reliability signal (conceptually analogous to our agreement gate).

### Closest Prior Work

1. **When KV Cache Reuse Fails… (arXiv:2601.08343)**: Proposes risk-aware gating via a “universally safe” classifier (AUC≈0.82) but does not specify a training-free, deployment-ready gating mechanism.
2. **KVComm (arXiv:2510.12872)**: Strong baseline for cross-context reuse, but not designed to preserve cross-candidate judge interaction and has very low shuffle-JCR.
3. **KVFlow (OpenReview)**: Focuses on caching policies for agent workflows; does not study judge invariance or shuffle-robust selection.
4. **CacheBlend/CacheClip**: Selective recomputation for RAG; our setting is not RAG but multi-candidate judging, and our mechanism is conservative fallback rather than token recomputation.

**Novelty Kill Search Summary:** Searched for the exact combination “KV cache reuse + judge + disagreement/ensemble gating/dense fallback” and “JCR + gating”, including GitHub/arXiv/OpenReview. No prior work using **algorithmic disagreement between reuse methods** as a safety gate for multi-candidate judges was found as of 2026-02-18. Full query log is in `notes.md`.

---

## Experiments

### Experimental Setup

We use the authors’ public code release for arXiv:2601.08343:
- Repo: https://anonymous.4open.science/r/kv_reuse_fails-5B0C (ZIP: `https://anonymous.4open.science/api/repo/kv_reuse_fails-5B0C/zip`).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3.2-3B-Instruct | 3B | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct | Used in the paper’s main experiments |

**No training data needed** (inference + cache construction only).

**Key settings (match paper scripts):**
- Benchmark: HumanEval (164 coding problems with unit tests)
- Candidate generation regime: Progressive Refinement / FullConnected, N=4
- Judge ordering: shuffle (`KVCOMM_JUDGE_SHUFFLE=1`)
- Judge outputs include both reuse-selected ID and dense-selected ID (`--judge-compare-dense`).

**Baseline Ladder (for this objective):**
- Dense prefill judge (reference; JCR=100 by definition)
- KVCOMM reuse judge (strongest published reuse baseline in this setting)
- DG-JKR (ours): Naïve+KVCOMM agreement gate + dense fallback
- Sanity/controls: Naïve reuse (auxiliary), and matched-rate random gating

**Implementation note (critical for validity):** to isolate judge-side effects, the verifier should generate a candidate set once (dense) and run multiple judge passes on the **same candidate texts** (dense / Naïve / KVCOMM) to compute p and d. This matches the paper’s protocol (“disable execution-side reuse to ensure an identical candidate set across methods”).

**Resource Estimate:**
- Compute budget: ~10–50 GPU-hours on 1×A100-80GB (HumanEval + 3 judge passes; plus 3 seeds if needed).
- GPU memory: 1×A100-80GB sufficient for 3B inference.
- API usage: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| HumanEval | Code generation benchmark with unit tests | Pass@1 / solve rate; JCR; disagreement rate d; precision p; reuse coverage | test | https://github.com/openai/human-eval | In repo (`experiments/run_humaneval.py`) |

### Main Results

#### Results Table (published baselines + ours)

| Method | Base Model | Setting | Acc / Pass@1 | JCR (shuffle) | Reuse Rate | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| Dense Prefill | Llama-3.2-3B | Prog.Refine, N=4 | 31.68 | 100.00 | 0.00 | 2601.08343 Table 1 (`./references/When-KV-Cache-Reuse-Fails-in-Multi-Agent-Systems-Cross-Candidate-Interaction-is-Crucial-for-LLM-Judges/sections/Overview judge decisions are not invariant under KV reuse..md`) | shuffle |
| Naïve Reuse | Llama-3.2-3B | Prog.Refine, N=4 | 49.06 | 23.60 | 100.00 | 2601.08343 Table 1 (same path as above) | shuffle |
| KVCOMM | Llama-3.2-3B | Prog.Refine, N=4 | 49.69 | 21.74 | 44.84 | 2601.08343 Table 1 (same path as above) | shuffle |
| PAL-KV | Llama-3.2-3B | Prog.Refine, N=4 | 49.07 | 24.22 | 44.84 | 2601.08343 Table 1 (same path as above) | shuffle |
| **DG-JKR (ours)** | Llama-3.2-3B | Prog.Refine, N=4 | **TBD** | **TBD** | **TBD** | - | Uses Naïve+KVCOMM agreement; dense fallback on disagreement |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Random gating | Fallback to dense on random d-fraction of examples (matched to DG-JKR’s fallback rate) | Lower JCR than DG-JKR if disagreement is informative |

### Experimental Rigor

- Seeds: run ≥3 seeds unless decoding is made deterministic (e.g., temp=0). Report mean±std for p, d, and JCR_gated.
- Confounder control: ensure identical candidate set across judge methods (generate candidates once; rerun judge only).
- Sanity check: random gating baseline should be close to the JCR predicted by “fallback with no signal”.

---

## Success Criteria

**Hypothesis:** Agreement between Naïve and KVCOMM reuse judges is a high-precision indicator that reuse preserved the dense winner, and disagreement is concentrated on “unsafe” (tight-margin) instances.

**Decision Rule:**
- **Proceed** if on HumanEval+shuffle: precision \(p\ge 0.90\) and coverage \((1-d)\ge 0.30\), and DG-JKR’s JCR exceeds matched-rate random gating by a margin outside the seed std range.
- **Pivot** if \(0.85 \le p < 0.90\): report as marginal; suggest adding one extra cheap signal (e.g., a confidence/margin heuristic) as future work.
- **Refute** if \(p < 0.85\) or \((1-d) < 0.10\) (gate is non-informative or triggers fallback too often).

---

## Impact Statement

If DG-JKR works, practitioners deploying multi-agent systems can safely apply judge-side KV cache reuse by enabling reuse on “stable” instances and falling back to dense prefill only when necessary. This would improve the reliability of LLM-as-a-judge pipelines under candidate-order perturbations while preserving meaningful prefill speedups.

---

## References

- [When KV Cache Reuse Fails in Multi-Agent Systems](https://arxiv.org/abs/2601.08343) - Liang et al., 2026
- [KVComm](https://arxiv.org/abs/2510.12872) - Ye et al., 2025
- [KVFlow](https://openreview.net/forum?id=5Iw1nDtYmT) - Pan et al., 2025
- [PromptCache](https://arxiv.org/abs/2311.04934) - Gim et al., 2024
- [CacheBlend](https://arxiv.org/abs/2405.16444) - Yao et al., 2024
- [CacheClip](https://arxiv.org/abs/2510.10129) - (arXiv), 2025
- [KVLink](https://arxiv.org/abs/2502.16002) - Yang et al., 2025
- [DroidSpeak](https://arxiv.org/abs/2411.02820) - Liu et al., 2024
- [PrefillShare](https://arxiv.org/abs/2602.12029) - (arXiv), 2026
- [TokenLake](https://arxiv.org/abs/2508.17219) - (arXiv), 2025
- [Segment-Level KV Cache Sharing](https://openreview.net/forum?id=kgzBkyqg6Z) - (OpenReview), 2026
- [H2O](https://arxiv.org/abs/2306.14048) - Zhang et al., 2023
- [Scissorhands](https://arxiv.org/abs/2305.17118) - Liu et al., 2023
- [StreamingLLM](https://arxiv.org/abs/2309.17453) - Xiao et al., 2023
- [SnapKV](https://arxiv.org/abs/2404.14469) - (arXiv), 2024
- [PyramidKV](https://arxiv.org/abs/2406.02069) - (arXiv), 2024
- [Large Language Models Can Be Easily Distracted by Irrelevant Context](https://arxiv.org/abs/2302.00093) - Shi et al., 2023
- [SmallKV](https://openreview.net/forum?id=0BVrpXMr5Y) - Zhao et al., 2025
- [Speculative Prefill](https://openreview.net/forum?id=bzbuZ0ItBq) - Liu et al., 2025
- [Position: LLMs Need a Bayesian Meta-Reasoning Framework](https://openreview.net/forum?id=RrvhbxO2hd) - Yan et al., 2025
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2023
