# untitled

# Fast Finite-State Guidance for Continuous Diffusion LMs via Custom Forward–Backward VJPs (DIFFINITY Case Study)

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Diffusion language models generate text by iteratively denoising a **continuous representation of an entire sequence**, rather than producing tokens strictly left-to-right. This makes them attractive for global, non-causal generation and editing. However, many practical deployments require **structured outputs** that obey strict syntactic constraints, such as JSON schemas, tool call formats, or regex-defined templates.

For autoregressive language models, **grammar-constrained decoding** can enforce such constraints by masking invalid next tokens. For continuous diffusion models, this is harder because there is no discrete prefix to validate during the denoising process.

**PLAID** is a continuous diffusion language model trained via likelihood-based diffusion over a word-embedding latent space (**Likelihood-Based Diffusion Language Models**). **DIFFINITY** is a recent training-free method that enables PLAID-style continuous diffusion models to obey **regular constraints** (regular expressions compiled to deterministic finite automata, DFAs) by adding a guidance term during denoising based on the gradient of an analytically computed DFA acceptance probability (**Continuous Diffusion Models Can Obey Formal Syntax**).

### The Problem

DIFFINITY reports strong constraint satisfaction (e.g., 92.9% on synthetic natural-language regex constraints) but also reports **very high latency overhead** that scales roughly linearly with the number of transitions in the tokenizer-aligned DFA. For example, DIFFINITY reports 145.6 s/sample on a benchmark with 71,735 transitions versus 33.3 s/sample for unconditional PLAID generation, and attributes this overhead to (i) the inner loop that aggregates token probabilities across DFA transitions and (ii) using PyTorch automatic differentiation (“autograd”) to compute gradients, which limits kernel fusion and increases memory traffic.

This overhead raises an actionable engineering/research uncertainty for the community: DIFFINITY suggests that “custom gradient kernels and succinct automata representations” could reduce overhead by 1–2 orders of magnitude, but it is unclear whether (a) **simple compiler/rematerialization recipes** (e.g., `torch.compile`, gradient checkpointing) already achieve most of the gain, or (b) DIFFINITY fundamentally requires **custom structured dynamic-programming gradients** to become practical. The answer changes what implementers should do next.

### Key Insight and Hypothesis

**Key insight**: DIFFINITY’s DFA objective is a structured dynamic program. Its gradients can be computed with an explicit **vector–Jacobian product (VJP; i.e., a custom backward pass)** using a forward–backward (inside–outside) algorithm over DFA edges. A custom VJP can (i) avoid constructing dense \(|Q|\times|Q|\) transition matrices, (ii) avoid building a large autograd tape, and (iii) reuse DFA structure across hundreds of denoising steps.

**Mechanism hypothesis (why DIFFINITY is a hard case for generic AD):** DIFFINITY’s tokenizer-aligned DFAs induce a repeated “sparse edge aggregation” pattern (many token-labeled edges whose weights depend on per-position token probabilities). In PyTorch, this pattern often becomes memory-bandwidth-bound and autograd-unfriendly due to large intermediate tensors and poor fusion across scatter/gather operations. A custom VJP that stores only forward/backward messages (O(|Q|) memory per position) and computes arc posteriors via fused scatter-add should reduce memory traffic enough to (a) improve per-step latency and (b) restore batching scalability that DIFFINITY reports as saturating.

**Testable hypothesis:** Relative to the best simple baseline (autograd DP with `torch.compile` and/or checkpointing), a custom VJP for DFA acceptance probability will yield ≥10× speedup on the DFA subroutine (expected probability + gradient) and ≥3× end-to-end speedup per guided denoising step, while preserving gradients (within numerical tolerance) and maintaining constraint satisfaction.

Why this could fail: (1) profiling may show PLAID denoising dominates runtime, capping end-to-end speedup; (2) tokenizer-aligned DFAs may be dense enough that sparse edge lists do not help; (3) `torch.compile`/checkpointing may already solve the overhead, implying custom VJPs are unnecessary.

---

## Proposed Approach

### Overview

We propose to replace DIFFINITY’s autograd-based gradient through its DFA dynamic program with a **custom autograd.Function** that:

1. Computes the DFA acceptance probability using forward messages over DFA states.
2. Computes gradients analytically using backward messages and arc posteriors (forward–backward).
3. Operates on a sparse edge list representation of the aligned DFA (never materializing dense |Q|×|Q| matrices).

The objective and guidance signal remain unchanged; only the backward pass implementation changes. We treat DIFFINITY as a representative instance of a broader pattern: **finite-state guidance objectives inside neural sampling loops**, where generic autograd may be a major efficiency bottleneck.

### Method Details

#### DIFFINITY’s expected probability computation

Given a token-level DFA \(A=(\Sigma,Q,q_0,\delta,F)\) and a length-\(L\) per-position token distribution \(p_k(\mathrm{tok})\) produced by the diffusion model decoder, DIFFINITY computes:

\[
\mathbb{E}_{s\sim \prod_k p_k}[\,s \in L(A)\,]
\]

via the DFA forward algorithm (Algorithm 1 in DIFFINITY). The implementation constructs per-position transition matrices \(M_k\) where \(M_k[i,j]\) is the probability of taking a transition from state \(i\) to \(j\) at position \(k\).

Tokenizer alignment is handled by DIFFINITY’s vocabulary-DFA alignment procedure (Algorithm 2), which constructs a token-level DFA that accepts all valid tokenizations of strings in the original regex language.

#### Custom VJP via forward–backward on a sparse edge list

Let \(E\) be the set of DFA transitions in the aligned DFA, each edge \(e=(q\xrightarrow{\mathrm{tok}}q')\). For each position \(k\), define the edge weight:

\[
 w_k(e) = p_k(\mathrm{tok})
\]

Define forward messages \(\alpha_k(q)\) as the total probability mass of reaching state \(q\) after consuming \(k\) tokens, and backward messages \(\beta_k(q)\) as the probability mass from state \(q\) at position \(k\) to an accepting state at position \(L\).

Then the acceptance probability is:

\[
 Z = \sum_{q\in F} \alpha_L(q)
\]

and for each position \(k\) and token \(\mathrm{tok}\), the derivative \(\partial Z / \partial p_k(\mathrm{tok})\) can be computed by summing edge posteriors:

\[
\frac{\partial Z}{\partial p_k(\mathrm{tok})} = \sum_{(q\xrightarrow{\mathrm{tok}}q')\in E} \alpha_{k-1}(q)\,\beta_k(q')
\]

We implement this derivative efficiently by iterating over the sparse edge list (grouped by token and/or by source state) and using fused scatter-add operations. We then combine it with the gradient of the decoder \(p_k(\mathrm{tok})\) with respect to the diffusion latent \(x_t\), exactly as DIFFINITY does.

#### “Simple systems” baselines

To avoid a purely engineering contribution that could be solved by standard compiler tricks, we include strong baselines where DIFFINITY’s original DP+autograd is wrapped with:
- `torch.compile` (inductor backend), and
- gradient checkpointing / rematerialization of the DP loop.

Our claim is only interesting if the custom VJP is meaningfully better than these simple recipes.

### Key Innovations

- **Autograd vs structured-VJP diagnosis**: Quantify whether finite-state DP gradients are the dominant bottleneck in DIFFINITY-style guidance, and whether `torch.compile`/checkpointing is sufficient.
- **Reusable WFSA-acceptance VJP building block**: Implement a custom backward pass for “acceptance probability under per-position token distributions” (a core primitive for DFA/WFSA-guided generation), and test it inside a diffusion sampling loop.
- **Sparse edge-list execution (with density gate)**: When aligned DFAs are sparse, avoid dense |Q|×|Q| transition matrices and exploit edge sparsity; when they are dense, fall back to a dense-but-fused VJP to test whether the benefit comes from custom AD rather than sparsity.

---

## Related Work

### Field Overview

This proposal lies at the intersection of (i) diffusion language models, (ii) constrained generation with formal languages, and (iii) differentiable dynamic programming over finite-state structures.

**Constrained generation** is commonly handled in autoregressive LMs with grammar/regex constrained decoding, where invalid next tokens are masked. Recent work has improved the efficiency of such decoding through better tokenizer–grammar alignment and pruning (e.g., GREATGRAMMA; ZapFormat).

**Diffusion language models** enable non-causal generation and editing, but their continuous latent trajectories make constrained decoding less direct. DIFFINITY shows that for **regular constraints**, one can compute an analytic acceptance probability and use its gradient for training-free guidance.

**Differentiable finite-state methods** (WFSAs/WFSTs) are widely used in speech recognition and structured prediction, where the forward–backward algorithm provides efficient exact gradients; several frameworks provide custom VJPs and memory-efficient implementations (GTN, Torch-Struct, LAST). A recent line of work (morphism-trick) studies how to make semiring dynamic programming differentiation efficient across AD systems.

### Related Papers

- **[Continuous Diffusion Models Can Obey Formal Syntax](./references/Continuous-Diffusion-Models-Can-Obey-Formal-Syntax/meta/meta_info.txt)**: Introduces DIFFINITY, a training-free DFA-guided sampling method for continuous diffusion LMs; reports strong constraint satisfaction but large DFA-gradient overhead.
- **[Likelihood-Based Diffusion Language Models](./references/Likelihood-Based-Diffusion-Language-Models/meta/meta_info.txt)**: Introduces PLAID, a likelihood-trained continuous diffusion language model that DIFFINITY builds on.
- **[DINGO: Constrained Inference for Diffusion LLMs](./references/DINGO-Constrained-Inference-for-Diffusion-LLMs/meta/meta_info.txt)**: Shows dynamic-programming-based constrained decoding for discrete diffusion LMs under regular constraints (different diffusion family than PLAID-style continuous diffusion).
- **[Constrained Decoding of Diffusion LLMs with Context-Free Grammars](./references/Constrained-Decoding-of-Diffusion-LLMs-with-Context-Free-Grammars/meta/meta_info.txt)**: Extends constrained decoding to CFG constraints for discrete diffusion models, emphasizing different algorithmic structure and overhead sources than DIFFINITY.
- **[Earley-Driven Dynamic Pruning for Efficient Structured Decoding](./references/Earley-Driven-Dynamic-Pruning-for-Efficient-Structured-Decoding/meta/meta_info.txt)**: Demonstrates that structured decoding speedups often come from pruning/caching state spaces, motivating efficiency-first evaluation.
- **[Flexible and Efficient Grammar-Constrained Decoding](./references/Flexible-and-Efficient-Grammar-Constrained-Decoding/meta/meta_info.txt)**: Improves tokenizer–grammar alignment and preprocessing time for autoregressive grammar-constrained decoding (GREATGRAMMA), highlighting tokenizer–formal-language mismatch as a recurring bottleneck.
- **[Differentiable Weighted Finite-State Transducers](./references/Differentiable-Weighted-Finite-State-Transducers/meta/meta_info.txt)**: GTN framework for differentiable WFST operations and custom gradients in PyTorch, showing structured VJPs can outperform generic autograd for finite-state DP.
- **[Torch-Struct: Deep Structured Prediction Library](./references/Torch-Struct-Deep-Structured-Prediction-Library/meta/meta_info.txt)**: Semiring-based dynamic programming library that supports efficient structured inference and gradients as first-class operations.
- **[LAST: Scalable Lattice-Based Speech Modelling in JAX](./references/LAST-Scalable-Lattice-Based-Speech-Modelling-in-JAX/meta/meta_info.txt)**: Shows that naive AD can be memory-prohibitive for large finite-state DP and motivates forward–backward / rematerialization designs.
- **[Fast and General Automatic Differentiation for Finite-State Methods](./references/Fast-and-General-Automatic-Differentiation-for-Finite-State-Methods/meta/meta_info.txt)**: Proposes the “morphism-trick” for efficient semiring-DP gradients (CPU Julia), a close conceptual prior to custom VJPs for automata.
- **[Torch-Struct: Deep Structured Prediction Library (arXiv)](https://arxiv.org/abs/2002.00876)**: Original paper describing Torch-Struct’s semiring DP abstractions and GPU-oriented batching.
- **[Differentiable Weighted Finite-State Transducers (arXiv)](https://arxiv.org/abs/2010.01003)**: Original differentiable WFST paper (GTN) that formalizes forward/backward over graphs under log/tropical semirings.
- **[CTC: Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)**: Canonical example where forward–backward provides exact gradients for a finite-state sum over paths.
- **[RNN-T](https://arxiv.org/abs/1211.3711)**: Uses a lattice-style forward–backward objective, illustrating how structured DP gradients become core training bottlenecks at scale.
- **[Semiring Parsing](https://arxiv.org/abs/cmp-lg/9909014)**: Introduces semiring-based dynamic programming as a unifying view for inside–outside / forward–backward computations.
- **[Inside-Outside and Backprop (Eisner, 2016)](https://arxiv.org/abs/1603.01777)**: Analyzes equivalences between dynamic programming and backpropagation, motivating why generic AD may be inefficient for structured programs.
- **[Classifier Guidance](https://arxiv.org/abs/2105.05233)**: Foundational diffusion guidance method; DIFFINITY can be viewed as a training-free instantiation of classifier guidance for syntax.
- **[Score-Based Generative Modeling via SDEs](https://arxiv.org/abs/2011.13456)**: Provides the score-based diffusion framework underlying guidance-style conditional sampling.
- **[Diffusion-LM](https://arxiv.org/abs/2205.14217)**: Early diffusion-based controllable text generation in embedding space; a broader context for PLAID-style models.
- **[Constrained Language Modeling with Langevin Dynamics (COLD)](https://arxiv.org/abs/2205.12558)**: Gradient-based constrained decoding for autoregressive LMs that also suffers large inference overhead, motivating efficiency work.
- **[Controlled Decoding from Language Models](https://arxiv.org/abs/2310.17022)**: KL-regularized inference-time control for autoregressive LMs, providing an alternative control paradigm.
- **[NeuroLogic Decoding](https://arxiv.org/abs/2010.12884)**: Autoregressive decoding with logical constraints, representing constraints as automata/logic during search.
- **[Guidance (library)](https://github.com/guidance-ai/guidance)**: Widely used open-source grammar/regex constrained decoding for autoregressive LMs, representing practitioner demand for structured output tooling.
- **[Outlines](https://github.com/dottxt-ai/outlines)**: Regex/JSON-schema constrained decoding toolkit for autoregressive LMs, highlighting the importance of efficient constraint enforcement in practice.
- **[OpenFST](https://www.openfst.org/)**: Foundational WFST toolkit (non-differentiable), providing background on finite-state operations and representations.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Continuous diffusion LMs | Denoise continuous sequence latents; decode at end | PLAID (Gulrajani & Hashimoto 2023) | LM perplexity; sampling speed | Many denoising steps; slower than AR |
| Continuous diffusion + formal constraints | Add gradient guidance based on analytic acceptance probability | DIFFINITY (Kim et al. 2026) | JSON schemas; synthetic NL regex | High overhead from automata DP + gradients |
| Discrete diffusion constrained decoding | DP / verification at token level during discrete denoising | DINGO (2025), CFG-constrained diffusion (2025/2026) | Code/JSON/SMILES; constraint sat | Does not apply to PLAID-style continuous diffusion |
| AR grammar-constrained decoding efficiency | Tokenizer–grammar alignment, pruning, caching | GREATGRAMMA (2025), ZapFormat (2025) | JSON/function calling/code | Still AR; different failure modes |
| Differentiable finite-state DP tooling | Forward–backward / semiring DP with custom VJPs | GTN (2020), Torch-Struct (2020), LAST (2023), morphism-trick (2026) | ASR lattices; structured prediction | Not yet applied to diffusion guidance |

### Closest Prior Work

1) **DIFFINITY** is the closest prior work: it defines the DFA expected probability objective and reports that autograd limits optimization. Our proposal keeps DIFFINITY’s objective but replaces autograd through the DFA DP with an analytic VJP and benchmarks whether this makes DIFFINITY practical.

2) **Differentiable WFSA/WFST frameworks (GTN, Torch-Struct, LAST)** are algorithmically related: they implement efficient forward–backward gradients for finite-state structures. They do not target DIFFINITY’s setting (continuous diffusion guidance with tokenizer-aligned DFAs) nor evaluate end-to-end constrained text generation latency.

3) **morphism-trick (Ondel Yang et al., 2026)** shows that custom VJPs for semiring DP can be orders of magnitude faster than generic AD, but is implemented in Julia and CPU-only, and is not applied to DIFFINITY.

**Novelty Kill Search Summary:** Searched for “DIFFINITY speedup”, “DIFFINITY torch.autograd automaton”, “forward backward DFA gradient diffusion”, and checked local proposals for DIFFINITY/PLAID usage (no matches). Found only generic differentiable WFSA/WFST tooling (GTN/k2/Torch-Struct/LAST/morphism-trick) and no prior work applying custom DFA DP gradients to DIFFINITY as of 2026-02-18.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| DIFFINITY (2026) | DFA acceptance-probability guidance for continuous diffusion LMs | Very high overhead; uses autograd through DP | Custom VJP + sparse edge DP | Avoids autograd tape + dense matrices; faster per-step guidance |
| GTN (2020) | Differentiable WFST ops with custom gradients | Not applied to diffusion guidance | Apply finite-state custom gradients to DIFFINITY objective | Same algorithmic idea, new application with different scaling constraints |
| LAST (2023) | Memory-efficient WFSA DP in JAX | Different domain; focuses on memory | Apply “DP should avoid AD memory blowup” insight to DIFFINITY | Shows feasibility of forward–backward / remat tricks at scale |
| Torch-Struct (2020) | Semiring DP library with efficient ops | Not DIFFINITY-specific | DIFFINITY-specific DP VJP + profiling | Tailored implementation can beat general-purpose tooling |
| morphism-trick (2026) | Efficient AD for semiring DP via custom VJPs | CPU Julia; no diffusion | Use custom VJP principle in PyTorch DIFFINITY | Confirms benefit in modern GPU diffusion setting |

---

## Experiments

### Experimental Setup

**Goal:** Decide whether DIFFINITY’s runtime bottleneck is primarily autograd-through-DP, and whether a custom forward–backward VJP beats both the original implementation and “simple systems” optimizations.

**Base model:** PLAID checkpoint and sampling code from the PLAID release (as used by DIFFINITY).

**Constraints / DFAs:** Use DIFFINITY’s benchmark regexes (JSON schema-derived + synthetic NL regexes) and the DIFFINITY tokenizer-aligned DFA construction.

**Representative constraints (pre-registered):** We will use DIFFINITY’s released benchmark construction code to sample fixed constraint instances before running experiments.
- **JSON**: choose exactly 2 JSON-schema-derived regexes from DIFFINITY’s 70-schema set: (i) the schema with the smallest aligned DFA (by number of transitions) and (ii) the schema with the largest aligned DFA (by number of transitions) among the 70.
- **Natural language**: choose exactly 2 synthetic regexes generated by DIFFINITY’s benchmark generator with a fixed RNG seed: one **PREFIX** template instance and one **BETWEEN (unbounded)** template instance.

These 4 constraints are pre-registered to avoid cherry-picking and to cover both very small and very large aligned automata. The verifier should log the chosen regex strings and the resulting aligned DFA sizes (|Q|, |E|, density) before any timing runs.

**Baseline Ladder (REQUIRED):**
- **Baseline A (original):** DIFFINITY DFA DP + PyTorch autograd (as close to paper as possible).
- **Baseline B (strong simple systems):** Baseline A + `torch.compile` (inductor) and/or gradient checkpointing/rematerialization of the DP loop; report the best of these as the strongest “simple” optimization.
- **Ours:** Custom autograd.Function with analytic forward–backward VJP over sparse edge lists (optionally also compiled).

**Profiling gate (first-class result):** Use torch.profiler to measure time + allocations for one guided denoising step, decomposed into:
1) PLAID denoiser/decoder,
2) DFA expected-prob forward,
3) DFA gradient/backward.

If (2+3) < 50% of the step time, refute early (speedup ceiling too small).

**Gradient agreement gate:** On the actual DFAs used above, compare \(\nabla\log Z\) w.r.t. token logits between Baseline A and Ours:
- Require relative L2 error < 1e-2 and cosine similarity > 0.99.

**Sparsity gate:** Measure DFA density \(|E|/|Q|^2\). If density > 10%, treat “sparse edge list is sufficient” as refuted and either (i) pivot to dense-but-fused custom VJP, or (ii) conclude limited headroom.

**Metrics:**
- Wall-clock time per guided denoising step and per sample (batch sizes 1 and 4).
- GPU memory peak during the DFA DP+grad.
- Constraint satisfaction rate on the chosen constraint subset.

**Resource Estimate**:
- Primarily inference-only + profiling; expected to fit comfortably within 768 GPU-hours.
- Plan: run at most ~4 constraints × (3 methods) × (batch sizes 1/4) × a small number of denoising steps for timing microbenchmarks, plus a small end-to-end sample set for satisfaction (e.g., 50–200 samples per constraint).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| DIFFINITY JSON schemas | Regex constraints derived from JSONSchemaBench | Satisfaction rate; Pass@k (optional) | Use DIFFINITY’s evaluation protocol | See DIFFINITY paper | DIFFINITY implementation |
| DIFFINITY NL regex | Synthetic regex templates (Prefix/Suffix/Between/etc.) | Satisfaction rate | Use DIFFINITY’s evaluation protocol | See DIFFINITY paper | DIFFINITY implementation |

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Constraint satisfaction | Time / step (s) | Time / sample (s) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| DIFFINITY autograd DP | PLAID 1.3B | JSON subset | **TBD** | **TBD** | **TBD** | [DIFFINITY](./references/Continuous-Diffusion-Models-Can-Obey-Formal-Syntax/meta/meta_info.txt) | re-run |
| + torch.compile / checkpoint (best) | PLAID 1.3B | JSON subset | **TBD** | **TBD** | **TBD** | - | strong baseline |
| **Ours: custom VJP (sparse)** | PLAID 1.3B | JSON subset | **TBD** | **TBD** | **TBD** | - | to be verified |

(Analogous rows for NL regex subset.)

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (full) | custom VJP + sparse edge list | fastest |
| Ours w/ dense-but-fused | ignore sparsity; use dense ops but custom VJP | helps if DFAs are dense |

### Experimental Rigor

- Use fixed seeds for sampling; report mean ± std across 3 seeds for timing and satisfaction metrics.
- Ensure identical diffusion hyperparameters across methods.
- Sanity check: compare acceptance probability values \(Z\) between Baseline A and Ours on random inputs.

---

## Success Criteria

**Hypothesis**: DFA DP + autograd is a dominant overhead in DIFFINITY, and analytic forward–backward gradients can substantially reduce this overhead without changing constraint satisfaction.

**Decision Rule**:
- **Proceed** if:
  1) Profiling shows DFA DP+grad ≥50% of step time, AND
  2) Ours achieves ≥10× speedup on the DFA subroutine and ≥3× end-to-end per-step speedup over the best “simple systems” baseline, AND
  3) Gradient agreement gate passes (rel L2 < 1e-2, cosine > 0.99), AND
  4) Constraint satisfaction changes by ≤1 pp on the subset.
- **Pivot** if profiling passes but sparsity gate fails (density >10%): try dense-but-fused custom VJP; proceed only if it beats torch.compile baseline.
- **Partial success** if DFA-subroutine speedup ≥5× but end-to-end speedup <3×, *and* it enables at least one quantitatively new operating point relative to the best simple baseline: either (i) batch size 16 (same GPU) without the batching-speedup saturation DIFFINITY reports, or (ii) ability to run the largest aligned DFA among the 70 JSON schemas within a fixed step-time budget (e.g., <10 s/step) where baselines exceed the budget.
- **Refute** if profiling fails (<50% DFA share), or if torch.compile/checkpoint achieves similar speedups to custom VJP, or if constraint satisfaction drops >2 pp, or if gradient agreement gate fails.

---

## Impact Statement

If successful, this work would make DIFFINITY-style training-free formal-syntax guidance practical for structured generation by continuous diffusion language models, reducing inference latency and memory overhead enough to support larger batches and more complex constraints. This would benefit practitioners who need reliable regex/JSON-constrained generation but cannot afford the current overhead of constrained diffusion guidance.

---

## References

- [Custom Forward–Backward Gradients for Fast DIFFINITY (Regex-Constrained Continuous Diffusion)](./references/Continuous-Diffusion-Models-Can-Obey-Formal-Syntax/meta/meta_info.txt) - Kim, Berg-Kirkpatrick, D'Antoni, 2026
- [Likelihood-Based Diffusion Language Models](./references/Likelihood-Based-Diffusion-Language-Models/meta/meta_info.txt) - Gulrajani, Hashimoto, 2023
- [DINGO: Constrained Inference for Diffusion LLMs](./references/DINGO-Constrained-Inference-for-Diffusion-LLMs/meta/meta_info.txt) - Suresh et al., 2025
- [Constrained Decoding of Diffusion LLMs with Context-Free Grammars](./references/Constrained-Decoding-of-Diffusion-LLMs-with-Context-Free-Grammars/meta/meta_info.txt) - Mündler et al., 2025/2026
- [Earley-Driven Dynamic Pruning for Efficient Structured Decoding](./references/Earley-Driven-Dynamic-Pruning-for-Efficient-Structured-Decoding/meta/meta_info.txt) - 2025
- [Flexible and Efficient Grammar-Constrained Decoding](./references/Flexible-and-Efficient-Grammar-Constrained-Decoding/meta/meta_info.txt) - Park, Zhou, D'Antoni, 2025
- [Differentiable Weighted Finite-State Transducers](./references/Differentiable-Weighted-Finite-State-Transducers/meta/meta_info.txt) - Hannun et al., 2020
- [Torch-Struct: Deep Structured Prediction Library](./references/Torch-Struct-Deep-Structured-Prediction-Library/meta/meta_info.txt) - Rush, 2020
- [LAST: Scalable Lattice-Based Speech Modelling in JAX](./references/LAST-Scalable-Lattice-Based-Speech-Modelling-in-JAX/meta/meta_info.txt) - Wu et al., 2023
- [Fast and General Automatic Differentiation for Finite-State Methods](./references/Fast-and-General-Automatic-Differentiation-for-Finite-State-Methods/meta/meta_info.txt) - Ondel Yang et al., 2026
- [Diffusion-LM](https://arxiv.org/abs/2205.14217) - Li et al., 2022
- [Controlled Decoding from Language Models](https://arxiv.org/abs/2310.17022) - Mudgal et al., 2023/2024
- [Constrained Language Modeling with Langevin Dynamics (COLD)](https://arxiv.org/abs/2205.12558) - Qin et al., 2022
- [NeuroLogic Decoding](https://arxiv.org/abs/2010.12884) - Lu et al., 2021
- [OpenFST](https://www.openfst.org/) - Allauzen et al., 2007
