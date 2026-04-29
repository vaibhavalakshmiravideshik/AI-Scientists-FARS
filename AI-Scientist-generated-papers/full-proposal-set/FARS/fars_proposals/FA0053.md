# untitled

# Draft De-anchoring via Logit Interpolation for Contextual Drag

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models are increasingly used in multi-step reasoning pipelines that expose the model to intermediate artifacts: prior attempts in iterative refinement, tool outputs in agent loops, and retrieved passages in retrieval-augmented generation (RAG). In these settings, the model must use context selectively: it should benefit from helpful context, but avoid being misled by erroneous context.

Recent evidence suggests this is a major unsolved reliability problem. **NoisyBench** shows that even strong reasoning models can lose a large fraction of their accuracy when the context contains irrelevant or misleading distractors, and that adding more reasoning tokens can worsen performance (“inverse scaling under noise”) ([Lost in the Noise](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt)). Separately, **Contextual Drag** shows that conditioning on a *failed* draft solution can bias subsequent generations toward structurally similar mistakes, causing 10–20% performance drops even when the prompt explicitly asks the model to verify the draft first ([Contextual Drag](./references/Contextual-Drag-How-Errors-in-the-Context-Affect-LLM-Reasoning/meta/meta_info.txt)).

A practical implication is that common “try again with your previous attempt in context” strategies can backfire: the model can recognize an error but still be pulled into nearby wrong reasoning patterns.

### The Problem

We focus on **contextual drag**: given a problem and a draft solution in context, the model is instructed to verify the draft and then produce a final answer. When the draft is incorrect, model performance often degrades substantially relative to solving from scratch.

Existing mitigation strategies in the contextual drag study rely on (i) multi-turn context denoising prompts that revise or filter the draft, or (ii) targeted supervised fine-tuning to teach a “fallback to clean-slate reasoning upon error detection” behavior ([Contextual Drag](./references/Contextual-Drag-How-Errors-in-the-Context-Affect-LLM-Reasoning/meta/meta_info.txt)). These approaches help but are either costly at inference (multi-turn pipelines) or require training data and fine-tuning.

In parallel, a separate line of work shows that **training-free logit combination** can change how strongly a model follows parts of its input:
- **Context Extrapolation / prior-aware decoding** uses logits from an original prompt and a weakened prompt to extrapolate away from “strong prior” failures (prompt injection, pattern-breaking tasks) ([Context Extrapolation](./references/Mitigating-the-Problem-of-Strong-Priors-in-LMs-with-Context-Extrapolation/meta/meta_info.txt)).
- **Selective Prompt Anchoring (SPA)** shows that comparing logits between a prompt and a version with a span masked approximates that span’s contextual contribution, enabling controllable amplification of prompt adherence ([SPA](./references/Selective-Prompt-Anchoring-for-Code-Generation/meta/meta_info.txt)).
- **Self-Anchor** adapts SPA-style steering to reasoning by anchoring attention to the question and intermediate plan steps during long generations ([Self-Anchor](./references/Self-Anchor-Large-Language-Model-Reasoning-via-Step-by-step-Attention-Alignment/meta/meta_info.txt)).

These methods suggest a simple question: can we use logit combination to **attenuate** the influence of an erroneous draft, without training and without multi-turn prompting?

### Key Insight and Hypothesis

**Key insight.** In contextual drag, the draft acts like an “anchor”: it contributes a systematic bias in next-token logits that can persist even when the model is instructed to verify and reject the draft. If we can estimate the model’s next-token logits when the draft content is removed, we can partially “pull” decoding toward that draft-free distribution to reduce draft-induced bias.

**Proposed hypothesis.** A fixed, training-free **logit interpolation** between (i) logits under the full draft-conditioned prompt and (ii) logits under a draft-redacted prompt will:
1) improve accuracy when the draft is wrong (by reducing anchoring to wrong reasoning patterns), while
2) preserving most of the benefit when the draft is correct (by not fully discarding the draft’s useful signal).

A critical confound for logit-mixing methods is that they can act like “more exploration” by increasing entropy. To avoid this, we will evaluate under **greedy decoding (temperature = 0)**, where increased entropy cannot improve performance via sampling.

We could be wrong because (i) the effect may be driven mainly by a context-length / recency shift when redacting the draft (not draft-content suppression), or (ii) contextual drag may not be well-modeled as an additive draft contribution in logit space, so mixing yields negligible changes under greedy decoding.

---

## Proposed Approach

### Overview

We propose **Draft De-anchoring Decoding (D3)**, a training-free decoding method for settings where the input contains a potentially unreliable span (here: a draft solution).

Given:
- an **original prompt** \(O\) that includes the full draft, and
- a **draft-redacted prompt** \(W\) that replaces the draft with a short placeholder block,

we compute next-token logits from both prompts at each decoding step and interpolate:

\[
\ell^*_t = (1-\beta)\,\ell_{O,t} + \beta\,\ell_{W,t}, \quad \beta \in [0,1].
\]

We then select the next token by greedy decoding from \(\ell^*_t\). We **pre-register** \(\beta=0.5\) and do not tune it after seeing results.

### Method Details

**Prompt definitions.** We use the contextual-drag prompt format from Cheng et al. (2026): the prompt contains (i) the problem and (ii) a model-generated draft solution, and asks the model to (a) check whether the draft is correct and (b) produce a final solution. We focus on the **1F** setting, which provides exactly **one** draft in context (their **2F** setting provides two drafts).

- **Original prompt \(O\)**: the standard 1F prompt with the full draft text.
- **Redacted prompt \(W\)**: the same prompt, but the draft span is replaced by:

```
--beginning of the draft--
[DRAFT REDACTED]
--end of the draft--
```

This choice avoids injecting out-of-distribution special tokens (e.g., `<pad>` in the middle of a prompt). It does, however, shorten the context substantially; we treat this as a known limitation and include two explicit controls: (i) a **drop-draft** baseline that removes the draft entirely, and (ii) a **length-matched filler** control that replaces the draft span with neutral text of the *same token length* (to isolate draft-content effects from position/recency effects).

**Decoding procedure.** Maintain two **key–value (KV) caches** (the standard transformer decoding cache that stores past attention keys/values so we do not recompute them every step), one for \(O\) and one for \(W\). At each decoding step \(t\):
1. Run a forward step on \(O\) to get \(\ell_{O,t}\).
2. Run a forward step on \(W\) to get \(\ell_{W,t}\).
3. Compute \(\ell^*_t\) by interpolation with \(\beta=0.5\).
4. Choose \(y_t = \arg\max \ell^*_t\) (greedy) and append \(y_t\) to both sequences.

**Why greedy decoding matters here.** Under greedy decoding, changing entropy or temperature cannot improve results via sampling diversity; any improvement requires a genuine change in the argmax sequence induced by mixing \(\ell_O\) and \(\ell_W\).

### Key Innovations

- **A minimal, training-free mitigation for contextual drag** that uses only two forward passes per decoding step and no multi-turn prompting.
- **Paired correct-draft vs wrong-draft evaluation** to explicitly measure the robustness–utilization trade-off (being robust to wrong context without discarding helpful context).
- **Greedy-decoding evaluation to rule out exploration confounds** for logit-mixing methods.

---

## Related Work

### Field Overview

This proposal connects three lines of work:

1) **Context pathologies in reasoning and agents.** NoisyBench and contextual drag show that models can be misled by irrelevant or erroneous context even when explicitly instructed to ignore it, and that additional reasoning compute can sometimes worsen outcomes under noise ([Lost in the Noise](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt), [Contextual Drag](./references/Contextual-Drag-How-Errors-in-the-Context-Affect-LLM-Reasoning/meta/meta_info.txt)).

2) **Training-free logit combination for steering and grounding.** Methods such as classifier-free guidance (CFG) and context-aware decoding (CAD) combine logits from conditional vs “weakened/unconditional” contexts to improve prompt adherence or factuality ([CFG](./references/Stay-on-topic-with-Classifier-Free-Guidance/meta/meta_info.txt), [CAD](./references/Context-aware-Decoding-Reduces-Hallucination-in-Query-focused-Summarization/meta/meta_info.txt)). Context extrapolation applies a similar idea to mitigate “strong priors” ([Context Extrapolation](./references/Mitigating-the-Problem-of-Strong-Priors-in-LMs-with-Context-Extrapolation/meta/meta_info.txt)).

3) **Anchoring / attention-steering during long generation.** SPA and Self-Anchor show that attention can drift away from important prompt components, and that logit-based approximations can be used to steer attention without training ([SPA](./references/Selective-Prompt-Anchoring-for-Code-Generation/meta/meta_info.txt), [Self-Anchor](./references/Self-Anchor-Large-Language-Model-Reasoning-via-Step-by-step-Attention-Alignment/meta/meta_info.txt)).

Our proposal tests a simplification: rather than learning when to reset (training) or running multi-turn denoising prompts, we apply a fixed logit interpolation toward a draft-redacted prompt and evaluate whether it reliably reduces wrong-draft anchoring while preserving correct-draft benefit.

### Related Papers

- **[Contextual Drag](./references/Contextual-Drag-How-Errors-in-the-Context-Affect-LLM-Reasoning/meta/meta_info.txt)**: Introduces contextual drag and shows large drops when conditioning on incorrect drafts even with explicit verification.
- **[Lost in the Noise (NoisyBench)](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt)**: Benchmark showing severe degradation under realistic distractors; highlights inverse scaling under noisy context.
- **[Context Extrapolation](./references/Mitigating-the-Problem-of-Strong-Priors-in-LMs-with-Context-Extrapolation/meta/meta_info.txt)**: Uses two prompts and logit extrapolation to mitigate strong priors / prompt injection.
- **[Selective Prompt Anchoring (SPA)](./references/Selective-Prompt-Anchoring-for-Code-Generation/meta/meta_info.txt)**: Logit-difference approximation for span contribution; improves code generation by amplifying instruction tokens.
- **[Self-Anchor](./references/Self-Anchor-Large-Language-Model-Reasoning-via-Step-by-step-Attention-Alignment/meta/meta_info.txt)**: Uses SPA-style steering + planning to maintain attention alignment in long reasoning.
- **[Context-aware Decoding (CAD)](./references/Context-aware-Decoding-Reduces-Hallucination-in-Query-focused-Summarization/meta/meta_info.txt)**: PMI-based decoding using logits with/without context; reduces hallucination in summarization.
- **[Classifier-Free Guidance for language](./references/Stay-on-topic-with-Classifier-Free-Guidance/meta/meta_info.txt)**: Adapts CFG to language generation for prompt adherence via conditional/unconditional logit mixing.
- **[CoRect](./references/CoRect-Context-Aware-Logit-Contrast-for-Hidden-State-Rectification-to-Resolve-Knowledge-Conflicts/meta/meta_info.txt)**: Uses contextual vs non-contextual logit contrast to localize and rectify “parametric suppression” in RAG.
- **[DoLa](https://arxiv.org/abs/2309.03883)**: Contrasts logits across transformer layers to reduce hallucinations without retraining.
- **[DExperts](https://arxiv.org/abs/2105.03023)**: Combines expert/anti-expert logits to steer generation (e.g., away from toxicity).
- **[Contrastive Search](https://arxiv.org/abs/2210.14140)**: Uses contrastive decoding objectives for coherent open-ended generation.
- **[Activation Addition](https://arxiv.org/abs/2308.10248)**: Adds activation directions from different prompts to steer generation.
- **[PASTA](https://arxiv.org/abs/2305.17952)**: Attention-head steering to improve prompt adherence (manual anchor selection).
- **[Re-Reading / RE2](https://arxiv.org/abs/2404.07143)**: Improves reasoning by re-reading or revisiting the prompt during generation.
- **[Plan-and-Solve](https://arxiv.org/abs/2305.04091)**: Uses prompted planning to improve multi-step reasoning.
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)**: Samples multiple reasoning paths and aggregates answers.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Iterative refinement with feedback; susceptible to contextual drag when prior attempts are wrong.
- **[Self-Refine](https://arxiv.org/abs/2303.17651)**: Iterative self-feedback improvement; can be harmed by wrong intermediate context.
- **[Inverse Scaling Dataset](https://arxiv.org/abs/2306.09479)**: Benchmarks where larger models do worse, including strong-prior failures.
- **[TruthfulQA](https://arxiv.org/abs/2109.07958)**: Benchmark exposing truthfulness failures driven by model priors.
- **[Lost in the Middle](https://arxiv.org/abs/2307.03172)**: Shows positional bias and context underuse in long contexts.
- **[ROME](https://arxiv.org/abs/2202.05262)**: Rank-one model editing; requires target labels and modifies weights.
- **[MEMIT](https://arxiv.org/abs/2210.07229)**: Large-scale model editing; supervised and weight-modifying.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Context pathologies | Measure failures under noisy / erroneous context | NoisyBench; Contextual Drag | NoisyBench; AIME/GPQA/CruxEval; Game of 24 | Often descriptive; mitigations can be costly |
| Output-level logit mixing | Combine logits from conditional vs weakened context | CFG; CAD; Context Extrapolation | LAMBADA; summarization; inverse scaling tasks | Usually 2× inference cost; hyperparameter sensitivity |
| Span-level anchoring control | Estimate span contribution via masking and manipulate it | SPA; Self-Anchor | HumanEval/MBPP; GSM8K/MATH/BBH | Masking can change context geometry; needs careful controls |
| Hidden-state interventions | Modify internal representations to preserve evidence | CoRect; DoLa | RAG QA/summarization | More invasive; harder to port across architectures |

### Closest Prior Work

- **Contextual Drag** ([paper](./references/Contextual-Drag-How-Errors-in-the-Context-Affect-LLM-Reasoning/meta/meta_info.txt)) documents the failure mode and evaluates mitigations via multi-turn denoising and supervised fine-tuning, but does not explore single-pass logit combination methods.

- **Context Extrapolation** ([paper](./references/Mitigating-the-Problem-of-Strong-Priors-in-LMs-with-Context-Extrapolation/meta/meta_info.txt)) uses logit extrapolation with weakened prompts to mitigate strong priors (e.g., prompt injection), but does not study erroneous drafts-in-context or the robustness–utilization trade-off with correct vs incorrect context.

- **SPA / Self-Anchor** ([SPA](./references/Selective-Prompt-Anchoring-for-Code-Generation/meta/meta_info.txt), [Self-Anchor](./references/Self-Anchor-Large-Language-Model-Reasoning-via-Step-by-step-Attention-Alignment/meta/meta_info.txt)) use masking-based logit differences to amplify useful prompt parts, but do not test the complementary question: whether mixing toward masked/redacted logits can *suppress* harmful context anchors in iterative reasoning settings.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Contextual Drag | Measures wrong-draft anchoring; mitigations via denoising prompts or SFT fallback | Multi-turn or training required; no single-pass logit method | Add a training-free logit interpolation that partially attenuates draft influence | If contextual drag is partly an anchoring effect, attenuation should recover wrong-draft performance |
| Context Extrapolation (PAD) | Logit extrapolation away from weakened prompts for strong priors | Not evaluated on draft-in-context; α tuning | Use fixed β interpolation between full-draft and redacted-draft logits | Simpler, targeted to draft anchoring; greedy decoding for interpretability |
| SPA / Self-Anchor | Uses masked-span logits to amplify prompt adherence | Focuses on amplifying helpful spans, not suppressing harmful ones | Use redacted-span logits as a “draft-free” reference to attenuate anchoring | Tests whether the same approximation helps robustness to erroneous context |
| CAD / CFG | Two-pass logit mixing for faithfulness/adherence | Often evaluated with sampling; can be confounded by diversity | Evaluate under greedy; focus on wrong-draft vs correct-draft trade-off | Greedy eliminates exploration confound; paired evaluation makes trade-off explicit |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen/Qwen3-8B (or Qwen/Qwen2.5-7B-Instruct as fallback) | 8B / 7B | https://huggingface.co/Qwen/Qwen3-8B | Open-weight target model for contextual-drag evaluation |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| N/A | Inference only | N/A | N/A | N/A |

**Other Resources (if applicable):**
- **Game of 24 puzzle set**: use the standard 1,362-instance set commonly used in Tree-of-Thoughts style evaluations if available; otherwise generate puzzles by enumerating 4-number multisets from {1,…,13} and filtering to solvable instances with a deterministic solver.
- **Deterministic verifier**: a Game-of-24 solver that verifies whether a proposed expression uses the numbers exactly once and evaluates to 24.

**Resource Estimate**:
- **Compute budget**: 10–80 GPU-hours total.
  - Draft generation: sampling the target model multiple times per puzzle until at least one correct and one incorrect draft are obtained (capped retries).
  - Evaluation: greedy decoding for 4 conditions (A, B, B2, C); condition C uses two forward passes per decoding step.
- **GPU memory**: ≤ 80GB for 7B–8B models.
- **API usage**: optional (not required); all experiments can run locally.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| Game of 24 (paired drafts) | Arithmetic puzzle: given four integers, construct an expression using each number exactly once with +, −, ×, ÷ such that the result is 24 | Accuracy (expression valid and equals 24); (optional) Tree Edit Distance to draft | test | (see setup) | Custom: parse expression + verify with solver |

**Draft construction for paired evaluation (fully automated):**
- For each puzzle, generate K draft solutions with the target model under a “draft solution” template (temperature 0.6, top-p 0.95), parse the final expression, and label as correct/incorrect via solver.
- Keep puzzles that yield at least one correct and one incorrect draft within a capped number of attempts; store one of each as the paired draft set.\n- For the **length-matched filler** condition (B2), replace the draft span with repeated neutral text (e.g., "This draft has been removed."), truncated/padded *after tokenization* to match the token length of the original draft exactly.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Accuracy (wrong draft) | Accuracy (correct draft) | Source | Notes |
|--------|------------|-----------|-------------------------|--------------------------|--------|------|
| 1F baseline (A) | Qwen3-8B | Game of 24 (paired) | **TBD** | **TBD** | To be verified | Greedy decoding; full draft in prompt |
| Drop-draft (B) | Qwen3-8B | Game of 24 (paired) | **TBD** | **TBD** | To be verified | Greedy; draft removed entirely |\n| Length-matched filler (B2) | Qwen3-8B | Game of 24 (paired) | **TBD** | **TBD** | To be verified | Greedy; replace the draft span with neutral filler text tokenized to the same length as the original draft |
| **D3 (ours)** (C) | Qwen3-8B | Game of 24 (paired) | **TBD** | **TBD** | To be verified | Greedy; \(\ell^*=(1-\beta)\ell_O+\beta\ell_W\), \(\beta=0.5\) |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| β sensitivity (report-only) | Evaluate β ∈ {0.25, 0.5, 0.75} (no selection) | If method is real, a moderate β should be best; extreme β behaves like baseline A or baseline B |
| Tree Edit Distance (TED) analysis (optional) | Compute **tree edit distance** between the output expression tree and the draft expression tree (higher means less structural copying) | In wrong-draft setting, D3 should increase TED vs 1F while improving accuracy |

### Analysis (Optional)

- **Where does D3 change decisions?** Count tokens where argmax under \(\ell_O\) differs from argmax under \(\ell^*\); correlate with correctness changes.
- **Failure stratification**: analyze by draft “closeness” (e.g., draft evaluates near 24 vs far), to see if anchoring is strongest for near-miss drafts.

---

## Success Criteria

**Criterion 1: Wrong-draft robustness without losing correct-draft utility**
- Hypothesis: D3 reduces wrong-draft anchoring.
- Validation: Under greedy decoding on the paired Game-of-24 set, D3 achieves **≥ +5** absolute accuracy improvement over 1F baseline on wrong-draft instances, while losing **≤ 1** point on correct-draft instances.

**Criterion 2: Net benefit under mixed-quality drafts**
- Hypothesis: A fixed β can improve expected performance in iterative pipelines where drafts are sometimes wrong.
- Validation: D3 improves accuracy on a 50/50 mixture of correct- and wrong-draft instances relative to (A) 1F, (B) drop-draft, and (B2) length-matched filler baselines.

---

## Impact Statement

If this works, practitioners building iterative refinement loops, multi-agent solvers, or tool-using agents can add a simple, training-free decoding wrapper that reduces error propagation from incorrect intermediate artifacts while retaining gains from helpful intermediate artifacts.

---

## References

- [Contextual Drag: How Errors in the Context Affect LLM Reasoning](./references/Contextual-Drag-How-Errors-in-the-Context-Affect-LLM-Reasoning/meta/meta_info.txt) — Cheng et al., 2026
- [Lost in the Noise: How Reasoning Models Fail with Contextual Distractors](./references/Lost-in-the-Noise-How-Reasoning-Models-Fail-with-Contextual-Distractors/meta/meta_info.txt) — Clark et al., 2025
- [Mitigating the Problem of Strong Priors in LMs with Context Extrapolation](./references/Mitigating-the-Problem-of-Strong-Priors-in-LMs-with-Context-Extrapolation/meta/meta_info.txt) — Douglas et al., 2024
- [Selective Prompt Anchoring for Code Generation](./references/Selective-Prompt-Anchoring-for-Code-Generation/meta/meta_info.txt) — Tian & Zhang, 2024
- [Self-Anchor: Large Language Model Reasoning via Step-by-step Attention Alignment](./references/Self-Anchor-Large-Language-Model-Reasoning-via-Step-by-step-Attention-Alignment/meta/meta_info.txt) — Zhang et al., 2025
- [Context-aware Decoding Reduces Hallucination in Query-focused Summarization](./references/Context-aware-Decoding-Reduces-Hallucination-in-Query-focused-Summarization/meta/meta_info.txt) — Xu, 2023
- [Stay on topic with Classifier-Free Guidance](./references/Stay-on-topic-with-Classifier-Free-Guidance/meta/meta_info.txt) — Sanchez et al., 2023
- [CoRect: Context-Aware Logit Contrast for Hidden State Rectification to Resolve Knowledge Conflicts](./references/CoRect-Context-Aware-Logit-Contrast-for-Hidden-State-Rectification-to-Resolve-Knowledge-Conflicts/meta/meta_info.txt) — Ma et al., 2026
