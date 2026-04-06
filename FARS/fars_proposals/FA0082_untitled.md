# untitled

# Context Bagging: Greedy Multi-Context Voting for Robust Long-Context QA Under Hard Distractors

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Long-context language models are increasingly used in systems that must answer questions using large, noisy contexts: retrieval-augmented generation (RAG), tool-using agents that paste intermediate tool outputs, and assistants that carry long conversation histories. In these settings, the input context is rarely “clean”: it often contains irrelevant passages, conflicting evidence, or misleading text that is nevertheless topically similar to the question.

Recent evaluations suggest that this is a major reliability bottleneck. **NoisyBench**, a benchmark that evaluates robustness across 11 RAG/reasoning/alignment/tool-use datasets by injecting contextual distractors (random documents, random chat history, and task-specific hard negatives), shows that distractors can cause catastrophic accuracy drops (up to ~80%) ([Lost in the Noise](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt)). It also reports **inverse scaling under noise**: increasing test-time reasoning (more generated reasoning tokens) can *decrease* accuracy in the presence of hard distractors. Separately, a systematic study of **self-consistency** (sampling multiple answers and majority voting) finds that self-consistency provides minimal gains in long-context QA and can even degrade performance, consistent with the hypothesis that errors are correlated because the underlying context bias is shared across samples ([How Effective Is Self-Consistency for Long-Context Problems?](./references/How-Effective-Is-Self-Consistency-for-Long-Context-Problems/meta/meta_info.txt)).

These results raise a practical question for test-time scaling: when spending multiple inference calls for robustness, what should we randomize? Most inference-time scaling methods randomize the **decoding trajectory** (temperature sampling, search, best-of-N, self-consistency). But if failures in noisy long contexts are primarily caused by **context-induced bias** (position bias, distractor capture, correlated attention drift), then sampling multiple trajectories under the *same* contaminated context may not help.

### The Problem

We study long-context question answering with **hard distractors**: context segments that are semantically similar to the question but do not contain the correct answer, and that can mislead the model when placed in salient positions (especially near the end of the context). This setting is motivated by:

- **NoisyBench**: hard-negative distractors cause the largest robustness drops and induce inverse scaling with additional reasoning ([Lost in the Noise](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt)).
- **Reasoning distraction attacks**: injected “reasoning tasks” can hijack chain-of-thought and cause large accuracy drops even when the base task is unchanged ([Distractor Injection Attacks](./references/Distractor-Injection-Attacks-on-Large-Reasoning-Models-Characterization-and-Defense/meta/meta_info.txt)).
- **Long-context position bias**: models can overweight text near the beginning or end, making end-position distractors especially harmful ([RULER](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt); [How Effective Is Self-Consistency for Long-Context Problems?](./references/How-Effective-Is-Self-Consistency-for-Long-Context-Problems/meta/meta_info.txt)).

A practitioner baseline in many RAG-like systems is **deterministic context selection** (keep the top-m segments by a relevance score) to fit within the context window or reduce noise (e.g., sparse context selection in RAG) ([Sparse RAG](./references/Accelerating-Inference-of-Retrieval-Augmented-Generation-via-Sparse-Context-Selection/meta/meta_info.txt)). However, deterministic selection can be brittle: if a hard distractor ranks highly, it will be consistently included and can consistently bias the generation.

We want an inference-only method that (i) preserves strong, deterministic decoding (to avoid “more randomness” as an explanation), (ii) does not require fine-tuning, and (iii) yields a clear decision rule about whether multi-pass compute should be spent on **context diversity**.

### Key Insight and Hypothesis

**Key insight.** Many robustness failures under contextual distractors appear to be *context-driven and correlated* across samples (e.g., self-consistency not helping long-context QA; inverse scaling under noise). If so, spending K inference calls on K decoding samples under the same context is inefficient. Instead, we can spend K calls on K **slightly different contexts** (subsets and orders of context segments) and aggregate answers.

**Hypothesis.** At a fixed test-time budget of K model calls, **Context Bagging**—sampling multiple perturbed contexts and running **greedy decoding** on each, then majority voting the answers—improves accuracy on long-context QA with hard distractors compared to:
1) a strong deterministic context-selection baseline, and
2) a permutation-only ensemble that changes order but not the selected subset.

We could be wrong because (i) deterministic relevance filtering may already remove most harmful distractors, leaving no headroom; (ii) when a distractor is truly “hard,” it may cause the model to make the same wrong decision regardless of minor context perturbations; (iii) multi-hop QA may be fragile to subset sampling because removing bridging paragraphs can break the reasoning chain.

---

## Proposed Approach

### Overview

We propose **Context Bagging (CoBag)**: an inference-time ensembling method for question answering from long contexts that allocates test-time compute to **context diversity** rather than decoding diversity.

Given a question q and a pool of context segments \(\{p_i\}_{i=1}^L\) (e.g., retrieved passages, tool outputs, or paragraphs), CoBag:

1. Scores each segment with a lightweight relevance model (embedding similarity to q).
2. Creates K different m-segment contexts by sampling subsets (with a fixed “always-keep” core of top-ranked segments).
3. Runs the LLM with **greedy decoding** on each context.
4. Aggregates answers by majority vote.

This is conceptually analogous to randomized smoothing defenses (perturb input and aggregate) such as SmoothLLM and SemanticSmooth, but applied at the *context-segment level* rather than at the character/token level ([SmoothLLM](./references/SmoothLLM-Defending-Large-Language-Models-Against-Jailbreaking-Attacks/meta/meta_info.txt); [SemanticSmooth](./references/Defending-Large-Language-Models-against-Jailbreak-Attacks-via-Semantic-Smoothing/meta/meta_info.txt); [Certified Semantic Smoothing](./references/Provable-Defense-Framework-for-LLM-Jailbreaks-via-Noise-Augumented-Alignment/meta/meta_info.txt)).

### Method Details

#### A. Segment relevance scoring

Compute a relevance score \(s_i\) for each segment \(p_i\) using cosine similarity between an embedding of the question and an embedding of the segment:

- Embedding model: **BAAI/bge-m3** (an open text embedding model used for cosine-similarity retrieval; any comparable open embedding model is acceptable).
- Score: \(s_i = \cos(e(q), e(p_i))\).

This relevance score is used only to guide subset sampling; it does not require training.

#### B. Constructing K bagged contexts

Let \(m\) be the number of segments included in each context (fixed across methods). Let \(r\) be the number of “core” segments always included.

For each ensemble member \(k \in \{1,\dots,K\}\):

1. **Always include** the top-r segments by \(s_i\). (Default: \(r=4\).)
2. Sample the remaining \(m-r\) segments **without replacement** from the remaining pool with probabilities:

\[
\Pr(p_i \text{ chosen}) \propto \exp(s_i/\tau),
\]

where \(\tau\) is a fixed temperature controlling how peaked the sampling distribution is (default: \(\tau=1.0\)).
3. **Randomly permute** the selected m segments before prompting the LLM.

#### C. Greedy generation and answer voting

For each of the K contexts, run the base LLM with:

- **Greedy decoding** (temperature=0, i.e., always pick the highest-probability next token).
- A strict answer-only format (short answer string; no chain-of-thought required).

Aggregate the K answers with majority vote over normalized answer strings (lowercase, strip punctuation/articles/extra whitespace). If there is a tie, break ties deterministically by choosing the answer from the context with the highest mean relevance score \(\frac{1}{m}\sum s_i\) among tied answers.

#### D. Why greedy decoding is essential here

The main confound in multi-sample methods is that improvements can come from increased output entropy (“try more random answers”) rather than from the proposed mechanism. Greedy decoding removes this explanation: every diversity benefit must come from changing the **context**.

### Key Innovations

- **Compute reallocation for robustness**: use K calls to sample K contexts instead of K decoding trajectories under one context, motivated by evidence that long-context failures are correlated across samples.
- **Ablation-friendly design**: a permutation-only ensemble isolates position/order effects, so any additional gain from CoBag can be attributed to subset diversity.
- **Hard-distractor control**: the evaluation constructs distractors that survive top-r inclusion, preventing improvements that come from simply dropping distractors.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) long-context evaluation and position bias, (ii) robustness to contextual distractors and prompt injection, (iii) retrieval/context selection methods for RAG, and (iv) inference-time scaling via ensembling and voting.

Long-context benchmarks show that models often fail when relevant information is buried or when the context is large and heterogeneous, and these failures are not well-predicted by simple needle-in-a-haystack tests ([RULER](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt); [HELMET](./references/HELMET-How-to-Evaluate-Long-Context-Language-Models-Effectively-and-Thoroughly/meta/meta_info.txt)). Noisy-context benchmarks and attacks show that semantically plausible distractors can reliably mislead models and can worsen with additional “reasoning” compute ([Lost in the Noise](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt); [Distractor Injection Attacks](./references/Distractor-Injection-Attacks-on-Large-Reasoning-Models-Characterization-and-Defense/meta/meta_info.txt)).

Inference-time ensembling methods (self-consistency, permutation self-consistency, mixture-of-agents) can improve performance in some settings, but can fail in long-context QA due to correlated errors and shared context bias ([How Effective Is Self-Consistency for Long-Context Problems?](./references/How-Effective-Is-Self-Consistency-for-Long-Context-Problems/meta/meta_info.txt); [Permutation Self-Consistency](./references/Found-in-the-Middle-Permutation-Self-Consistency-Improves-Listwise-Ranking-in-Large-Language-Models/meta/meta_info.txt)). Separately, RAG work studies permutation sensitivity and proposes training-based stabilization (DPO) or architectural changes ([Stable-RAG](./references/Stable-RAG-Mitigating-Retrieval-Permutation-Induced-Hallucinations-in-Retrieval-Augmented-Generation/meta/meta_info.txt); [Sparse RAG](./references/Accelerating-Inference-of-Retrieval-Augmented-Generation-via-Sparse-Context-Selection/meta/meta_info.txt)). Our proposal asks whether a simpler, training-free alternative—context bagging with greedy decoding—can improve robustness under hard distractors.

### Related Papers

- **[Lost in the Noise: How Reasoning Models Fail with Contextual Distractors](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt)**: Introduces NoisyBench and shows large robustness drops and inverse scaling under contextual distractors.
- **[Distractor Injection Attacks on Large Reasoning Models](./references/Distractor-Injection-Attacks-on-Large-Reasoning-Models-Characterization-and-Defense/meta/meta_info.txt)**: Shows injected reasoning distractors can hijack reasoning models and that robustness may require explicit defenses.
- **[How Effective Is Self-Consistency for Long-Context Problems?](./references/How-Effective-Is-Self-Consistency-for-Long-Context-Problems/meta/meta_info.txt)**: Finds self-consistency provides minimal gains and can degrade long-context QA/TR due to correlated position-bias errors.
- **[SmoothLLM](./references/SmoothLLM-Defending-Large-Language-Models-Against-Jailbreaking-Attacks/meta/meta_info.txt)**: Uses randomized character perturbations and aggregation to defend against jailbreak suffixes; motivates input-perturb-and-vote as a robustness wrapper.
- **[SemanticSmooth](./references/Defending-Large-Language-Models-against-Jailbreak-Attacks-via-Semantic-Smoothing/meta/meta_info.txt)**: Applies semantic transformations plus voting for jailbreak defense with better utility than character noise.
- **[Certified Semantic Smoothing / NAAT](./references/Provable-Defense-Framework-for-LLM-Jailbreaks-via-Noise-Augumented-Alignment/meta/meta_info.txt)**: Provides a certified smoothing framework via randomized ablation and noise-augmented alignment tuning.
- **[Permutation Self-Consistency](./references/Found-in-the-Middle-Permutation-Self-Consistency-Improves-Listwise-Ranking-in-Large-Language-Models/meta/meta_info.txt)**: Reduces positional bias by shuffling list order and aggregating outputs; closest inference-time analog for order perturbations.
- **[Gold Panning](./references/Gold-Panning-Turning-Positional-Bias-into-Signal-for-Multi-Document-LLM-Reasoning/meta/meta_info.txt)**: Uses strategic context shuffling to identify relevant documents efficiently; highlights that permutations can be informative beyond averaging.
- **[Stable-RAG](./references/Stable-RAG-Mitigating-Retrieval-Permutation-Induced-Hallucinations-in-Retrieval-Augmented-Generation/meta/meta_info.txt)**: Studies retrieval-order sensitivity and uses hidden-state clustering + DPO to stabilize answers across permutations.
- **[Sparse RAG](./references/Accelerating-Inference-of-Retrieval-Augmented-Generation-via-Sparse-Context-Selection/meta/meta_info.txt)**: Selects a sparse subset of retrieved documents (KV-cache selection) to reduce compute and filter undesirable contexts.
- **[RULER](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt)**: A long-context benchmark suite showing failures beyond simple retrieval when contexts are long and complex.
- **[HELMET](./references/HELMET-How-to-Evaluate-Long-Context-Language-Models-Effectively-and-Thoroughly/meta/meta_info.txt)**: Argues for application-centric long-context evaluation and shows synthetic tasks poorly predict downstream performance.
- **[Search-R1](./references/Search-R1-Training-LLMs-to-Reason-and-Leverage-Search-Engines-with-Reinforcement-Learning/meta/meta_info.txt)**: Reports QA results on multi-hop datasets including MuSiQue and illustrates baseline difficulty for open 7B models.
- **[Lost in the Middle](https://arxiv.org/abs/2307.03172)**: Characterizes position bias in long contexts and motivates order-based perturbations.
- **[Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)**: Introduces self-consistency as an inference-time sampling-and-voting method for reasoning.
- **[Universal Self-Consistency](https://arxiv.org/abs/2405.02409)**: Improves aggregation in self-consistency via an LLM-based aggregator prompt.
- **[RAG](https://arxiv.org/abs/2005.11401)**: Introduces retrieval-augmented generation as a standard paradigm for knowledge-intensive QA.
- **[Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906)**: A widely used dense retriever that motivates paragraph-level retrieval and ranking pipelines.
- **[Fusion-in-Decoder (FiD)](https://arxiv.org/abs/2007.01282)**: Encodes multiple retrieved passages independently and fuses them in the decoder; a common multi-passage QA baseline.
- **[REPLUG](https://arxiv.org/abs/2301.12652)**: A plug-and-play method that combines LM predictions across retrieved documents, related to multi-context aggregation.
- **[TracLLM](https://arxiv.org/abs/2506.04202)**: Uses perturbation-based attribution to identify which context texts drive outputs, relevant to diagnosing distractor influence.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Long-context failure analysis | Measure and explain long-context failures (position bias, synthetic vs real) | [RULER](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt), [HELMET](./references/HELMET-How-to-Evaluate-Long-Context-Language-Models-Effectively-and-Thoroughly/meta/meta_info.txt), [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Long-context QA/summarization/retrieval | Diagnostics do not directly yield robust inference strategies |
| Noisy-context robustness | Evaluate robustness under contextual distractors and propose training-time fixes | [Lost in the Noise](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt), [Distractor Injection Attacks](./references/Distractor-Injection-Attacks-on-Large-Reasoning-Models-Characterization-and-Defense/meta/meta_info.txt) | NoisyBench; distractor injection across tasks | Strong methods often require fine-tuning or RL; inference-time baselines can fail |
| Context selection / compression | Reduce context length or filter irrelevant segments | [Sparse RAG](./references/Accelerating-Inference-of-Retrieval-Augmented-Generation-via-Sparse-Context-Selection/meta/meta_info.txt), [FiD](https://arxiv.org/abs/2007.01282) | RAG QA, summarization | Deterministic selection can be brittle under hard distractors |
| Permutation / ensemble at inference | Perturb order or sample multiple outputs and aggregate | [Permutation Self-Consistency](./references/Found-in-the-Middle-Permutation-Self-Consistency-Improves-Listwise-Ranking-in-Large-Language-Models/meta/meta_info.txt), [Gold Panning](./references/Gold-Panning-Turning-Positional-Bias-into-Signal-for-Multi-Document-LLM-Reasoning/meta/meta_info.txt), [Self-Consistency](https://arxiv.org/abs/2203.11171) | Ranking, multi-doc reasoning, QA | Can fail when errors are correlated (long-context self-consistency failure) |
| Randomized smoothing defenses | Perturb inputs and aggregate for robustness (often for safety) | [SmoothLLM](./references/SmoothLLM-Defending-Large-Language-Models-Against-Jailbreaking-Attacks/meta/meta_info.txt), [SemanticSmooth](./references/Defending-Large-Language-Models-against-Jailbreak-Attacks-via-Semantic-Smoothing/meta/meta_info.txt), [Certified Semantic Smoothing](./references/Provable-Defense-Framework-for-LLM-Jailbreaks-via-Noise-Augumented-Alignment/meta/meta_info.txt) | Jailbreak ASR, instruction-following | Mostly evaluated for prompt attacks, not long-context QA distractors |

### Closest Prior Work

- **How Effective Is Self-Consistency for Long-Context Problems?** ([paper](./references/How-Effective-Is-Self-Consistency-for-Long-Context-Problems/meta/meta_info.txt)): Evaluates self-consistency on long-context QA/TR and finds it does not mitigate position bias and can degrade performance; our proposal responds by changing what is randomized (context rather than decoding) and by using greedy decoding to avoid entropy confounds.
- **Lost in the Noise / NoisyBench** ([paper](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt)): Shows hard distractors are especially damaging and that more reasoning can hurt; our proposal tests an inference-time alternative that uses multiple calls to diversify context exposure rather than to generate longer reasoning traces.
- **Permutation Self-Consistency** ([paper](./references/Found-in-the-Middle-Permutation-Self-Consistency-Improves-Listwise-Ranking-in-Large-Language-Models/meta/meta_info.txt)): Uses random shuffles and aggregation to reduce positional bias, but is studied for ranking rather than answer robustness under hard distractors; we include a permutation-only voting baseline and test whether subset diversity adds further robustness.
- **Gold Panning** ([paper](./references/Gold-Panning-Turning-Positional-Bias-into-Signal-for-Multi-Document-LLM-Reasoning/meta/meta_info.txt)): Exploits position bias via strategic reordering for document identification efficiency; our proposal targets a different objective (answer correctness under distractors) and tests whether randomized subset sampling improves robustness at fixed query budget.
- **Sparse RAG** ([paper](./references/Accelerating-Inference-of-Retrieval-Augmented-Generation-via-Sparse-Context-Selection/meta/meta_info.txt)): Selects sparse contexts to improve efficiency and focus; our baseline is a deterministic top-m selector, and our contribution is to turn selection into a bagging ensemble for robustness under hard distractors.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| [Self-consistency (long-context study)](./references/How-Effective-Is-Self-Consistency-for-Long-Context-Problems/meta/meta_info.txt) | Samples multiple answers under the same context and votes | Errors remain correlated due to shared context bias; can degrade | Keep decoding deterministic; randomize context subsets/orders instead | If correlation is context-driven, context diversity reduces correlation |
| [Permutation self-consistency](./references/Found-in-the-Middle-Permutation-Self-Consistency-Improves-Listwise-Ranking-in-Large-Language-Models/meta/meta_info.txt) | Randomly shuffles list order and aggregates outputs | Not designed for hard distractors in QA; order-only perturbations may be insufficient | Add subset sampling (bagging) + evaluate on hard distractors | Subset diversity changes which distractors co-occur and where evidence appears |
| [Sparse RAG](./references/Accelerating-Inference-of-Retrieval-Augmented-Generation-via-Sparse-Context-Selection/meta/meta_info.txt) | Selects a sparse subset of documents/caches to reduce compute and improve focus | Deterministic selection brittle when distractor ranks highly | Sample multiple plausible sparse contexts and vote | Voting can recover when deterministic selection is unlucky |
| [Lost in the Noise](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt) | Benchmarks distractor robustness; proposes training-time RARE | Training required; inference-time scaling can worsen | Inference-only wrapper for robustness evaluation | If effective, offers immediate mitigation without retraining |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen/Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Open-weight; used in long-context SC study; fits on 1×A100 80GB |
| meta-llama/Llama-3.1-8B-Instruct (optional) | 8B | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | Optional generalization check; long context support |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| N/A | Inference only | - | - | - |

**Other Resources (if applicable):**
- MuSiQue dataset (paragraph-level multi-hop QA): https://huggingface.co/datasets/dgslibisey/MuSiQue
- Embedding model for relevance scoring: https://huggingface.co/BAAI/bge-m3

**Resource Estimate**:
- **Compute budget**: ~50–200 GPU-hours total (single A100-80GB), dominated by K-pass inference over 1–3k examples.
- **GPU memory**: 1×A100 80GB for 7B–8B inference (bf16/fp16).
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MuSiQue (validation) | Multi-hop question answering benchmark where each example provides a question and a set of 20 paragraphs (with labels for which paragraphs are supporting evidence) | Exact Match (EM; whether the normalized predicted answer matches any gold answer/alias exactly), token-level F1 (optional) | validation | https://huggingface.co/datasets/dgslibisey/MuSiQue | Standard QA normalization + alias matching |

**Noisy setting construction (hard distractor at end):**
- For each example, compute relevance scores \(s_i\) over its 20 paragraphs.
- Choose a **hard distractor** as the highest-scoring paragraph with `is_supporting=false` that does **not** contain the gold answer (or any alias) as a substring after normalization.
- Require that this distractor is ranked within the top-r by \(s_i\); otherwise drop the example (to ensure the distractor is included in all methods).
- Move this distractor to the **end** of the paragraph sequence to create a worst-case recency condition.

### Main Results

#### Results Table

(All results below are to be filled by verification; numbers from prior papers are not directly comparable because this proposal constructs a new noisy split.)

| Method | Base Model | Benchmark | EM | F1 | Source | Notes |
|---|---|---|---:|---:|---|---|
| Filtered-Single (top-m, original order) | Qwen2.5-7B | MuSiQue (noisy split) | **TBD** | **TBD** | - | Deterministic selection; greedy decoding |
| Permute-Vote (fixed subset, K perms) | Qwen2.5-7B | MuSiQue (noisy split) | **TBD** | **TBD** | - | Order diversity only; greedy per run; majority vote |
| **CoBag-Vote (ours)** | Qwen2.5-7B | MuSiQue (noisy split) | **TBD** | **TBD** | - | Subset+order diversity; greedy per run; majority vote |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| CoBag without order shuffling | Sample subsets but keep original order | If order drives most gains, this underperforms full CoBag |
| CoBag uniform subset sampling | Sample subsets uniformly (ignore relevance weights) | Underperforms relevance-weighted sampling due to dropped key evidence |
| Hyperparameter sensitivity (report-only) | Evaluate \(m\in\{8,12,16\}\), \(K\in\{3,5,7\}\) without selecting based on results | Diminishing returns in K; too-small m degrades multi-hop reasoning |

### Analysis (Optional)

- **Correlation diagnostic**: measure answer agreement across contexts and show whether CoBag reduces “same wrong answer” concentration compared to fixed-subset permutations.
- **Distractor influence**: report how often the predicted answer string appears only in the distractor paragraph (proxy for distractor capture).

---

## Success Criteria

**Criterion 1: Context diversity improves robustness at fixed K**
- Hypothesis: Under hard distractors, varying the context subset reduces correlated failures.
- Validation: On the noisy split, CoBag-Vote outperforms Permute-Vote (same K) and Filtered-Single, with consistent improvement across ≥3 random seeds for subset/permutation sampling.

**Criterion 2: The gain is not explained by dropping the distractor**
- Hypothesis: Improvements come from subset/order diversity while still including the hard distractor.
- Validation: By construction, the hard distractor is in the always-keep top-r set for all methods; report 100% distractor inclusion rate and still observe CoBag’s advantage over Permute-Vote.

---

## Impact Statement

If validated, Context Bagging provides a simple, training-free way to spend test-time compute for robustness in RAG and agent pipelines: instead of sampling multiple reasoning traces under a single noisy context, practitioners can sample multiple plausible contexts (subsets and orders) and vote. This would be immediately usable in deployments that already run multiple inference calls for reliability, and it would provide a concrete recommendation for mitigating inverse-scaling behavior under contextual distractors.

---

## References

- [Lost in the Noise: How Reasoning Models Fail with Contextual Distractors](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt) - Lee et al., 2026
- [How Effective Is Self-Consistency for Long-Context Problems?](./references/How-Effective-Is-Self-Consistency-for-Long-Context-Problems/meta/meta_info.txt) - 2024
- [SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks](./references/SmoothLLM-Defending-Large-Language-Models-Against-Jailbreaking-Attacks/meta/meta_info.txt) - Robey et al., 2023
- [Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing](./references/Defending-Large-Language-Models-against-Jailbreak-Attacks-via-Semantic-Smoothing/meta/meta_info.txt) - Ji et al., 2024
- [Provable Defense Framework for LLM Jailbreaks via Noise-Augumented Alignment](./references/Provable-Defense-Framework-for-LLM-Jailbreaks-via-Noise-Augumented-Alignment/meta/meta_info.txt) - Cheng et al., 2026
- [Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models](./references/Found-in-the-Middle-Permutation-Self-Consistency-Improves-Listwise-Ranking-in-Large-Language-Models/meta/meta_info.txt) - Tang et al., 2023
- [GOLD PANNING: Strategic Context Shuffling for Needle-in-Haystack Reasoning](./references/Gold-Panning-Turning-Positional-Bias-into-Signal-for-Multi-Document-LLM-Reasoning/meta/meta_info.txt) - Byerly & Khashabi, 2025
- [Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](./references/Stable-RAG-Mitigating-Retrieval-Permutation-Induced-Hallucinations-in-Retrieval-Augmented-Generation/meta/meta_info.txt) - Zhang et al., 2026
- [Distractor Injection Attacks on Large Reasoning Models: Characterization and Defense](./references/Distractor-Injection-Attacks-on-Large-Reasoning-Models-Characterization-and-Defense/meta/meta_info.txt) - Zhang et al., 2025
- [Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection](./references/Accelerating-Inference-of-Retrieval-Augmented-Generation-via-Sparse-Context-Selection/meta/meta_info.txt) - Zhu et al., 2024
- [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly](./references/HELMET-How-to-Evaluate-Long-Context-Language-Models-Effectively-and-Thoroughly/meta/meta_info.txt) - Yen et al., 2024
- [RULER: What’s the Real Context Size of Your Long-Context Language Models?](./references/RULER-Whats-the-Real-Context-Size-of-Your-Long-Context-Language-Models/meta/meta_info.txt) - Hsieh et al., 2024
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](./references/Search-R1-Training-LLMs-to-Reason-and-Leverage-Search-Engines-with-Reinforcement-Learning/meta/meta_info.txt) - Jin et al., 2025
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - Liu et al., 2024
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Universal Self-Consistency for Large Language Models](https://arxiv.org/abs/2405.02409) - Chen et al., 2024
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Lewis et al., 2020
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) - Karpukhin et al., 2020
- [Fusion-in-Decoder: Generating Answers by Fusing Retrieved Passages](https://arxiv.org/abs/2007.01282) - Izacard & Grave, 2021
- [REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/abs/2301.12652) - Shi et al., 2023
- [TracLLM: A Generic Framework for Attributing Long Context LLMs](https://arxiv.org/abs/2506.04202) - Wang et al., 2025
