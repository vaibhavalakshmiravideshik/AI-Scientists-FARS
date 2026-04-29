# untitled

# Progress-Guarded LAVE: Lexer-Ignored Stall Filtering for Reliable CFG-Constrained Diffusion Decoding

## Scope and Constraints
- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Setting**: Inference-only modification to CFG-constrained decoding for diffusion language models (no training)
- **Automation**: Fully automated evaluation (CFG parsing + unit tests; no human labels)
- **Compute budget**: ≤768 GPU-hours (expected ≤5 GPU-hours)

## Introduction

### Context and Motivation
Diffusion language models (dLLMs) generate by iteratively filling masked positions in a fixed-length token canvas. Many deployments require *structured generation* where outputs must satisfy a strict syntax (e.g., valid JSON, compilable code).

### The Problem
**LAVE (Lookahead-then-Verify)** provides *reliable* CFG-constrained decoding for dLLMs: a proposed token update is accepted only if at least one sampled lookahead completion yields a CFG-extendable intermediate state, checked via an Earley parser ([LAVE](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/meta/meta_info.txt)). On CPP-Bench (HumanEval-CPP, 164 C++ tasks), LAVE improves LLaDA-8B syntactic@1 from 74.2% (NO-CD) to 90.9% (Table 1; [Implementation Details](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/sections/Implementation%20Details.md)).

LAVE reports a specific residual failure mode: some generations produce repetitive patterns (often whitespace/newlines) until hitting the max length, and **all invalid outputs hit the length cap** ([Remaining Syntax Errors](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/sections/Understanding%20the%20Remaining%20Syntax%20Errors.md)). This suggests that “extendable under the grammar” is not sufficient to prevent “no-progress” loops.

### Key Insight and Hypothesis
Many practical grammars use lexer ignore rules (e.g., `%ignore WS` and comment patterns). Under such grammars, arbitrarily long ignored substrings can remain CFG-extendable, so an extendability-only acceptance test can keep accepting token updates that consume length budget without increasing non-ignored terminals.

**Hypothesis**: Adding a lightweight **lexer-progress stall guard** to LAVE will reduce max-length truncation loops (lower `hit_max_len_rate`) and improve syntactic@1 with negligible overhead, by reusing LAVE’s cached verified witness string.

---

## Proposed Approach

### Overview
**Progress-Guarded LAVE (PG-LAVE)** keeps LAVE unchanged except for an early rejection rule that blocks **ignored-only** proposals once lexical progress has stalled.

### Method Details
- **Base algorithm (unchanged): LAVE.** Keep LAVE’s proposal step, lookahead verification, and cache-enhanced recovery with attempt budget `tau` ([Overview of LAVE](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/sections/Overview%20of%20LAVE.md)).
- **Progress signal from an existing LAVE artifact.** LAVE stores a mask-free constructive witness string `y_cache` (the last lookahead completion that passed verification). Let `Lex_G(·)` be the grammar lexer (ignore rules applied). Define `g(y_cache)=|Lex_G(y_cache)|` (count of non-ignored terminals). Maintain `stall_count`: consecutive **accepted** steps where `g` does not increase.
- **Ignored-only proposal test.** A proposed token update `t*` with decoded string `s=decode(t*)` is *ignored-only* if it matches the union of grammar ignore regexes (or, if needed, the conservative proxy regex `^\s+$`).
- **Stall guard (new).** Before running LAVE verification for proposal `t*`: if `is_ignored_only(t*)` and `stall_count≥H`, reject immediately and resample (consumes one of `tau` attempts).
- **Fallback (avoid deadlock).** If `tau` attempts are exhausted at a step, revert to standard LAVE: accept the last proposal that passes verification.
- **Control baseline (no lexer progress): naive cap.** Same fallback, but reject when consecutive accepted ignored-only steps ≥ `K`.
- **Pre-registered hyperparameters:** `H=32`, `K=32`.

### Key Innovations
- **Progress vs extendability**: targets a failure mode where reliable extendability checks still accept length-wasting updates.
- **Grammar-derived signal**: uses lexer output (non-ignored terminals) instead of tokenizer-id heuristics.
- **Mechanism isolation**: naive cap control tests whether progress tracking adds value beyond a simple heuristic.

---

## Related Work

### Field Overview
Autoregressive CFG-constrained decoding relies on incremental parsing/token masking (e.g., PICARD, Outlines, LMQL), which assumes left-to-right prefixes. For diffusion LMs, IG-CD introduced CFG constraints via language-intersection feasibility checks, while LAVE introduced reliable constraints via lookahead extendability checks. LAVE’s own analysis isolates residual max-length failures due to repetition, motivating an inference-only fix targeted to lexer-ignored stalls.

### Related Papers
- **[LAVE](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/meta/meta_info.txt)**: Reliable CFG constrained decoding for diffusion LMs.
- **[IG-CD](./references/Constrained-Decoding-of-Diffusion-LLMs-with-Context-Free-Grammars/meta/meta_info.txt)**: CFG constraints for diffusion LMs via CFG∩regular feasibility.
- **[DINGO](./references/DINGO-Constrained-Inference-for-Diffusion-LLMs/meta/meta_info.txt)**: Constrained diffusion decoding for regular-language constraints.
- **[Rainbow Padding](./references/Rainbow-Padding-Mitigating-Early-Termination-in-Instruction-Tuned-Diffusion-LLMs/meta/meta_info.txt)**: Training-free fix for diffusion early termination.
- **[Fast & Fluent dLLMs](./references/Fast-and-Fluent-Diffusion-Language-Models-via-Convolutional-Decoding-and-Rejective-Fine-tuning/meta/meta_info.txt)**: Diffusion decoding/training changes to reduce degeneration.
- **[RPG](./references/Rethinking-Repetition-Problems-of-LLMs-in-Code-Generation/meta/meta_info.txt)**: Grammar-based repetition penalization for AR code generation.
- **[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)**: Empirical study of degeneration/repetition.
- **[A Theoretical Analysis of the Repetition Problem](https://arxiv.org/abs/2012.14660)**: Theoretical analysis of repetition loops.
- **[Learning to Break the Loop](https://arxiv.org/abs/2206.02341)**: Mitigating repetition loops.
- **[Repetition Dropout](https://arxiv.org/abs/2305.04490)**: Training-time repetition mitigation.
- **[LLaDA](https://arxiv.org/abs/2502.09992)**: Open diffusion LM family used by LAVE.
- **[MDLM](https://arxiv.org/abs/2406.07524)**: Masked discrete diffusion LM recipe.
- **[MaskGIT](https://arxiv.org/abs/2202.04200)**: Iterative masked-token generation.
- **[Block Diffusion](https://arxiv.org/abs/2503.09573)**: Semi-autoregressive block diffusion decoding.
- **[Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047)**: Distribution distortion analysis for AR grammar masking.
- **[Flexible & Efficient Grammar-Constrained Decoding](https://arxiv.org/abs/2502.05111)**: Efficient preprocessing + online masking for AR CFG decoding.
- **[Outlines](https://arxiv.org/abs/2307.09702)**: Practical structured generation with regex/CFG constraints.
- **[PICARD](https://arxiv.org/abs/2109.05093)**: Incremental parsing for constrained decoding.
- **[LMQL](https://arxiv.org/abs/2212.06094)**: Query language for constrained LM programs.

### Taxonomy
| Family | Core idea | Representative papers | Typical eval | Limitation |
|---|---|---|---|---|
| AR grammar-constrained decoding | Mask invalid next tokens via incremental parsing | PICARD; Outlines; LMQL; Synchromesh | structured SQL/JSON/code | assumes left-to-right prefixes |
| Diffusion constrained decoding | Constrain masked canvases via feasibility/extendability | IG-CD; LAVE; DINGO | CPP/JSON/SMILES CFG tasks | can still degenerate (e.g., stalls) |
| Degeneration mitigation | Reduce early termination / repetition loops | Rainbow Padding; Fast & Fluent; repetition papers | open-ended + structured | often not CFG-ignore aware |

### Closest Prior Work
- **LAVE**: identifies max-length repetition as the main remaining failure, but no explicit stall-prevention rule.
- **IG-CD / DINGO**: do not target LAVE’s residual max-length failure mode.
- **RPG / repetition work**: mostly AR-focused, not reliable CFG-constrained diffusion decoding.

**Novelty Kill Search Summary:** Queried for “LAVE/lookahead-then-verify + progress/stall/whitespace loop/repetition penalty” and scanned local proposal indexes; no close prior work adding a lexer-progress stall guard to LAVE was found as of 2026-02-18 (full query log in `notes.md`).

### Comparison Table
| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LAVE (2026) | Reliable CFG constraints via lookahead extendability | Residual max-length repetition | Add lexer-progress stall guard | Blocks ignored-token stalls without changing core LAVE |
| IG-CD (2025) | CFG feasibility via language intersection | Not reliable under finite remaining length | N/A (different base) | Targets a post-LAVE failure mode |
| RPG (2025) | Grammar-aware repetition penalties (AR) | AR-only | Use LAVE witness + stall counter | Minimal patch in diffusion setting |
| Rainbow Padding (2025) | Prevents premature `<eos>` in dLLMs | Opposite failure mode | Complementary | Does not prevent ignored-token stalls |

---

## Experiments

### Experimental Setup
- **Codebase:** Start from LAVE’s official implementation (https://github.com/zhangyitonggg/CD4dLLM) and add PG-LAVE as a small change in the acceptance/re-proposal loop.
- **Base model:** LLaDA-8B-Instruct (8B), https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct.
- **Benchmark:** **CPP-Bench (HumanEval-CPP)**, 164 C++ synthesis tasks with unit tests (as in LAVE; [Benchmark](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/sections/Benchmark.md)).
- **Decoding hyperparameters (match LAVE):** `L=256`, `T=128`, `temp=0.2`, block size `32`, lookahead `N=10`, attempt budget `tau=5` ([Implementation Details](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/sections/Implementation%20Details.md)).
- **Main methods (3 conditions):** (1) LAVE, (2) LAVE + naive ignored-token cap (`K=32`), (3) PG-LAVE (`H=32`).
- **Baseline ladder checks (run if PG-LAVE improves over LAVE):** (i) prompt add-on “avoid repeated blank lines / trailing whitespace”, (ii) best-of-3 LAVE.
- **Resource estimate:** LAVE reports 9.30 s/instance for LLaDA-8B on CPP-Bench (Table 3; [RQ3](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/sections/RQ3%20How%20about%20the%20Runtime%20Overhead%20Introduced%20by%20LAVE.md)) → ~0.42 GPU-hours per run on one A100; 3 methods × 3 seeds ≈3.8 GPU-hours.

### Benchmarks and Metrics
| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| CPP-Bench (HumanEval-CPP) | 164 C++ tasks with unit tests | syntactic@1, functional@1, hit_max_len_rate, time | test | via LAVE repo; fallback: https://huggingface.co/datasets/zai-org/humaneval-x | LAVE repo |

Metric definitions: syntactic@1 (parseable under CFG; higher better), functional@1 (passes tests; higher), hit_max_len_rate (len==L; lower), time (seconds; lower).

### Main Results
Baseline numbers below are copied from **LAVE Tables 1–3** (Table 1 in [Implementation Details](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/sections/Implementation%20Details.md), Table 2 in [RQ2](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/sections/RQ2%20How%20Well%20Does%20LAVE%20Improve%20the%20Functional%20Correctness%20of%20dLLM%20Outputs.md), Table 3 in [RQ3](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/sections/RQ3%20How%20about%20the%20Runtime%20Overhead%20Introduced%20by%20LAVE.md)). LAVE reports averages over ≥4 runs but not standard deviations.

| Method | Base Model | Benchmark | syntactic@1 (%) | functional@1 (%) | time (s) | hit_max_len_rate | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| NO-CD | LLaDA-8B | CPP-Bench | 74.2 | 16.3 | 8.55 | N/A | LAVE | Avg ≥4 runs |
| IG-CD | LLaDA-8B | CPP-Bench | 86.1 | 17.8 | 9.66 | N/A | LAVE | Avg ≥4 runs |
| **LAVE** | LLaDA-8B | CPP-Bench | 90.9 | 19.5 | 9.30 | N/A | LAVE | Avg ≥4 runs |
| LAVE + naive cap (K=32) | LLaDA-8B | CPP-Bench | TBD | TBD | TBD | TBD | - | Run 3 seeds |
| **PG-LAVE (H=32)** | LLaDA-8B | CPP-Bench | TBD | TBD | TBD | TBD | - | Run 3 seeds |

### Ablation Studies
| Variant | What’s changed | Expected finding |
|---|---|---|
| Naive cap control | caps consecutive ignored-only accepts | may reduce truncation but is less targeted |
| PG-LAVE | lexer-progress stall detection + guard | reduces truncation with fewer side effects |

### Experimental Rigor
- **Seeds / variance:** `seeds=[42, 123, 456]`; report mean±std for all methods we run.
- **Sanity check (LAVE claim):** measure fraction of syntactic failures with length==`L`; LAVE claims this is ~100%.
- **Key confounder control:** if gains come only from rejecting whitespace, naive cap should match PG-LAVE.

---

## Success Criteria
**Hypothesis (directional):** PG-LAVE reduces max-length truncation loops and improves syntactic@1.

**Decision Rule:**
- **Proceed** if PG-LAVE reduces `hit_max_len_rate` by ≥30% relative (mean over 3 seeds) and improves syntactic@1 by ≥1.0 point vs LAVE, with time within +10%.
- **Pivot** if PG-LAVE and naive cap have overlapping std on both syntactic@1 and `hit_max_len_rate`.
- **Refute** if `hit_max_len_rate` reduction <10% and syntactic@1 gain <0.5 points.

---

## Impact Statement
If successful, PG-LAVE is an inference-only patch targeting LAVE’s reported dominant residual failure mode (repetition until max length) for grammar-constrained diffusion decoding, improving reliability in structured-generation pipelines.

---

## References
- [Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars](./references/Lookahead-then-Verify-Reliable-Constrained-Decoding-for-Diffusion-LLMs-under-Context-Free-Grammars/meta/meta_info.txt) - Zhang et al., 2026
- [Constrained Decoding of Diffusion LLMs with Context-Free Grammars](./references/Constrained-Decoding-of-Diffusion-LLMs-with-Context-Free-Grammars/meta/meta_info.txt) - Muendler et al., 2025
- [DINGO: Constrained Inference for Diffusion LLMs](./references/DINGO-Constrained-Inference-for-Diffusion-LLMs/meta/meta_info.txt) - Suresh et al., 2025
- [Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs](./references/Rainbow-Padding-Mitigating-Early-Termination-in-Instruction-Tuned-Diffusion-LLMs/meta/meta_info.txt) - Kim et al., 2025
- [Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](./references/Fast-and-Fluent-Diffusion-Language-Models-via-Convolutional-Decoding-and-Rejective-Fine-tuning/meta/meta_info.txt) - Seo et al., 2025
- [Rethinking Repetition Problems of LLMs in Code Generation](./references/Rethinking-Repetition-Problems-of-LLMs-in-Code-Generation/meta/meta_info.txt) - Dong et al., 2025
- LAVE code: https://github.com/zhangyitonggg/CD4dLLM
- LLaDA-8B-Instruct: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
