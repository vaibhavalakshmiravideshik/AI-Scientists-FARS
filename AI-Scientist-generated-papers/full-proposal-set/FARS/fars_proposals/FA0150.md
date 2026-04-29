# untitled

# ShallowPPL: Early-Exit Conditional Perplexity for Faster LongCodeZip-Style Code Context Compression

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Constraints**: fully automated evaluation; no human labeling/judging.

## Introduction

### Context and Motivation

Repository assistants and long-context code completion often require feeding thousands of tokens of repository context into a code language model, which increases prefill latency and cost. **Context compression** shortens the input while aiming to preserve the information needed for the downstream task.

**LongCodeZip** is a training-free, structure-aware compressor for code. It ranks candidate function chunks using an information-theoretic relevance score based on **conditional perplexity** (token negative log-likelihood; lower is better), then optionally prunes within selected functions. LongCodeZip reports non-trivial scoring overhead: on Long Code Completion with Qwen2.5-Coder-7B (A100-80G), compression takes **2.58s** while generation time drops **15.70s → 6.59s** (Table IX in `./references/LongCodeZip-Compress-Long-Context-for-Code-Language-Models/sections/V-D RQ4 Efficiency Analysis.md`).

### The Problem

LongCodeZip’s core primitive is **approximated mutual information (AMI)** for chunk relevance. For candidate chunk \(c\) and query/instruction \(q\), it uses
\[\mathrm{AMI}(c,q)=\mathrm{PPL}(q)-\mathrm{PPL}(q\mid c),\]
where \(\mathrm{PPL}(q\mid c)\) is the perplexity of \(q\) when \(c\) is included as context. Computing \(\mathrm{PPL}(q\mid c)\) requires a transformer forward pass over \([c;q]\); doing this for many candidate functions/blocks can dominate compression latency.

### Key Insight and Hypothesis

**Key insight:** for *ranking* candidate code chunks, we may not need final-layer token predictions. If intermediate representations already encode enough query-conditioned semantics, then computing an approximate negative log-likelihood at an intermediate layer could preserve the top-ranked set while reducing scoring compute.

**Hypothesis:** Early-exiting the scoring model at \(L=\lfloor N_{\text{layers}}/2 \rfloor\) and decoding logits with a lightweight **logit lens** (unembedding applied to layer-\(L\) hidden states) preserves LongCodeZip-level downstream quality while reducing compression wall-clock time.

**Why this could be wrong:** early layers may mostly reflect surface cues (identifiers/local syntax), causing ShallowPPL to collapse toward embedding-similarity retrieval on “implicit dependency” cases.

---

## Proposed Approach

### Overview

We propose **ShallowPPL**, a drop-in replacement for LongCodeZip’s AMI scoring that computes an **early-exit conditional perplexity** \(\mathrm{PPL}_L\) at a fixed intermediate layer and substitutes \(\mathrm{AMI}_L\) for \(\mathrm{AMI}\) in LongCodeZip’s coarse function selection and fine block pruning.

### Method Details

Let the scoring model have \(N\) transformer layers. For Qwen2.5-Coder-7B, \(N=28\); we pre-register \(L=14\).

For each candidate chunk \(c\) and query \(q\):
1. Run layers 1..\(L\) on \([c;q]\) to obtain hidden states \(h_L\).
2. Apply the model’s final normalization (if present) to reduce representational drift.
3. Decode approximate logits \(\ell_L = W_U h_L\) using the unembedding matrix \(W_U\) (logit lens).
4. Compute token negative log-likelihood on \(q\) tokens only to obtain \(\mathrm{PPL}_L(q\mid c)\) and \(\mathrm{PPL}_L(q)\).
5. Define \(\mathrm{AMI}_L(c,q)=\mathrm{PPL}_L(q)-\mathrm{PPL}_L(q\mid c)\) and use it everywhere LongCodeZip uses AMI.

### Key Innovations

- **Early-exit AMI for code-function ranking**: applies intermediate-layer decoding to conditional-perplexity chunk ranking.
- **Lexical-collapse diagnostic**: tests whether ShallowPPL tracks AMI-style selection beyond embedding similarity.

---

## Related Work

### Field Overview

Long-context code systems often combine retrieval or compression with generation. Embedding-based retrieval is fast but can miss implicit dependencies; likelihood/MI-based chunk scoring can capture model-specific relevance but is expensive. Separately, early/intermediate-layer methods (heads, probes, intermediate retrieval) show that shallow signals can guide long-context efficiency. We test whether intermediate-layer decoding is sufficient for LongCodeZip’s AMI ranking.

### Related Papers

- **[LongCodeZip](./references/LongCodeZip-Compress-Long-Context-for-Code-Language-Models/meta/meta_info.txt)**: AMI-based function ranking plus intra-function pruning for code context compression.
- **[RepoQA](./references/RepoQA-Evaluating-Long-Context-Code-Understanding/meta/meta_info.txt)**: Evaluates long-context repository understanding via “needle function” identification.
- **[EHPC](./references/Efficient-Prompt-Compression-with-Evaluator-Heads-for-Long-Context-Transformer-Inference/meta/meta_info.txt)**: Training-free prompt compression using early-layer evaluator heads.
- **[ILRe](./references/ILRe-Intermediate-Layer-Retrieval-for-Context-Compression-in-Causal-Language-Models/meta/meta_info.txt)**: Intermediate-layer retrieval then rerun full model on reconstructed context.
- **[Probe and Skip](./references/Probe-and-Skip-Self-Predictive-Token-Skipping-for-Efficient-Long-Context-LLM-Inference/meta/meta_info.txt)**: Self-predictive probing for efficient long-context inference.
- **[Tuned Lens](../../papers/paper_summaries/Eliciting Latent Predictions from Transformers with the Tuned Lens.md)**: Reliable intermediate-layer decoding via learned affine translators.
- **[Semantic Hub Hypothesis](../../papers/paper_summaries/The Semantic Hub Hypothesis Language Models Share Semantic Representations Across Languages and Modalities.md)**: Suggests intermediate layers encode semantically aligned representations across modalities (including code), supporting the plausibility of mid-layer relevance signals.
- **[LLMLingua](https://arxiv.org/abs/2310.05736)**: Likelihood-guided prompt compression for faster inference.
- **[LongLLMLingua](https://arxiv.org/abs/2310.06839)**: Instruction-aware contrastive perplexity for long-context compression.
- **[LLMLingua-2](https://arxiv.org/abs/2403.12968)**: Distilled discriminative prompt compressor.
- **[Selective Context](https://arxiv.org/abs/2310.06201)**: Self-information token pruning for context compression.
- **[Gisting](../../papers/paper_summaries/Long Context In-Context Compression by Getting to the Gist of Gisting.md)**: Learns compact “gist” representations for long contexts.
- **[Compactor](https://arxiv.org/abs/2507.08143)**: Query-agnostic KV-cache compression calibrated with likelihood signals.
- **[KVzip](https://arxiv.org/abs/2505.23416)**: Query-agnostic KV-cache compression via context reconstruction.
- **[Expected Attention](https://arxiv.org/abs/2510.00636)**: Predicts future attention distributions for KV-cache compression.
- **[RepoCoder](https://aclanthology.org/2023.emnlp-main.152.pdf)**: Repository-level code completion via iterative retrieval and generation.
- **[RepoGenix](https://dl.acm.org/doi/10.1145/3691620.3695331)**: Dual-context repository-level code completion.
- **[RLCoder](https://arxiv.org/abs/2407.19487)**: Reinforcement learning for repository-level code completion.
- **[cAST](../../papers/paper_summaries/cAST Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree.md)**: AST-based structural chunking for code RAG.
- **[DietCode](https://dl.acm.org/doi/10.1145/3540250.3549094)**: Code simplification for pre-trained code models.
- **[SlimCode](https://dl.acm.org/doi/10.1145/3643753)**: Model-agnostic code simplification for large language models.

### Taxonomy

| Family | Core idea | Representative papers | Typical evaluation | Limitation |
|---|---|---|---|---|
| MI / likelihood-based code compression | Rank chunks by conditional likelihood reduction | LongCodeZip, LongLLMLingua | Long code completion, RepoQA | Scoring overhead |
| Embedding-based retrieval for code | Retrieve chunks by embedding similarity | RepoCoder, cAST | Repo QA / repo completion | Misses implicit deps |
| Early/intermediate-layer efficiency signals | Use shallow signals for salience | EHPC, ILRe, Probe-and-Skip | Long-context benchmarks | May lose semantics |
| KV/prefill compression | Compress KV or prefill compute | Compactor, KVzip, Expected Attention | Long-context LM inference | Often query-agnostic |

### Closest Prior Work

- **LongCodeZip**: Defines AMI scoring; does not evaluate early-exit approximations.
- **Tuned Lens / logit lens**: Decode intermediate predictions, but not used for conditional-perplexity *chunk ranking*.
- **EHPC/ILRe/Probe-and-Skip**: Early/intermediate signals for compression/inference, but not AMI-style code-function ranking.

**Novelty Kill Search Summary:** (2026-02-19) Web+local queries: “early-exit perplexity reranking”, “intermediate layer conditional perplexity retrieval”, “logit lens perplexity early exit”, “LongCodeZip acceleration”. Found Tuned Lens and generic early-exit work, but no direct application to LongCodeZip-style conditional-perplexity code chunk ranking.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours might win |
|---|---|---|---|---|
| LongCodeZip | Full-depth AMI scoring + pruning | Scoring overhead | Use AMI\(_L\) from early-exit | Similar ranking at lower cost |
| EHPC | Early heads score token importance | Not chunk-level AMI | Apply early-exit to AMI | Minimal pipeline change |
| ILRe | Intermediate-layer retrieval + reconstruct | Different pipeline | Only change scoring depth | Lower engineering burden |
| UniXCoder-style RAG | Embedding similarity ranking | Lexical/semantic gaps | Use shallow NLL signal | Avoid lexical collapse |

---

## Experiments

### Experimental Setup

**Implementation target:** official LongCodeZip repo (https://github.com/YerbaPage/LongCodeZip). Implement a new scoring backend that early-exits at layer \(L\) and computes logit-lens NLL for \(q\).

**Main conditions (3):**
1. **LongCodeZip (Full AMI)**.
2. **ShallowPPL (ours)**: AMI → AMI\(_{L=14}\).
3. **RAG (Function Chunking)**: UniXCoder embedding-similarity ranking (LongCodeZip baseline family).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-Coder-7B (Instruct) | 7B | https://huggingface.co/Qwen | Match LongCodeZip setting |
| UniXCoder-base | ~110M | https://huggingface.co/microsoft/unixcoder-base | Retrieval baseline |

**Resource Estimate:**
- **Compute budget**: ≤12 GPU-hours on 1×A100-80GB.
- **GPU memory**: ≤80GB.
- **API usage**: none.

### Benchmarks and Metrics

| Benchmark | What it evaluates | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Long Code Completion | Completion under long code context (LongCodeZip dataset) | ES (edit similarity; higher better), EM (exact match; higher better) | test (500) | LongCodeZip repo | LongCodeZip repo |
| RepoQA (LongCodeZip variant) | Needle-function identification from long repo context (includes Go; 600 tests/60 repos) | AvgAcc (accuracy; higher better) | test | LongCodeZip repo | LongCodeZip repo |

**SOTA note:** canonical RepoQA has a public leaderboard, but LongCodeZip uses a modified variant (includes Go), so we treat LongCodeZip as the closest comparable published baseline for this proposal.

### Main Results

We will report quality and (separately) compression/end-to-end wall-clock. Published accuracy baselines below are copied from the raw LongCodeZip table; std is not reported.

| Method | Base Model | Benchmark | Metric (mean±std) | Source | Notes |
|---|---|---|---|---|---|
| No Compression | Qwen2.5-Coder-7B | Long Code Completion | ES=56.36, EM=31.80 (1 run) | `./references/LongCodeZip-Compress-Long-Context-for-Code-Language-Models/sections/V-A RQ1 Effectiveness on Code Compression.md` | Table II |
| RAG (Function Chunking) | Qwen2.5-Coder-7B | Long Code Completion | ES=52.79, EM=26.00 (1 run) | same as above | Table II |
| LongCodeZip (Full AMI) | Qwen2.5-Coder-7B | Long Code Completion | ES=57.55, EM=32.40 (1 run) | same as above | Table II |
| No Compression | Qwen2.5-Coder-7B | RepoQA (LCZ variant) | AvgAcc=86.0 (1 run) | same as above | Table IV (includes Go) |
| RAG (Function Chunking) | Qwen2.5-Coder-7B | RepoQA (LCZ variant) | AvgAcc=54.3 (1 run) | same as above | Table IV |
| LongLLMLingua | Qwen2.5-Coder-7B | RepoQA (LCZ variant) | AvgAcc=71.3 (1 run) | same as above | Table IV |
| LongCodeZip (Full AMI) | Qwen2.5-Coder-7B | RepoQA (LCZ variant) | AvgAcc=87.2 (1 run) | same as above | Table IV |
| **ShallowPPL (Ours)** | Qwen2.5-Coder-7B | Both | **TBD (mean±std over 3 runs)** | - | To be verified |

### Ablation Studies

Only if ShallowPPL passes the success criteria, run one follow-up ablation:

| Variant | What’s changed | Expected finding |
|---|---|---|
| ShallowPPL (coarse-only) | AMI\(_L\) for coarse selection; full AMI for fine pruning | If coarse dominates cost, most speedup remains |

### Experimental Rigor

- **Variance**: use deterministic decoding; report mean±std over 3 repeats for wall-clock. If any component is stochastic, fix `seeds=[0,1,2]` and report mean±std for quality.
- **Sanity check**: random function ranking should be near chance (as LongCodeZip reports for “Random” / “No Context” baselines).
- **Validity threats / controls**:
  1. **Lexical-collapse confound**: compare ShallowPPL to embedding RAG on a disagreement slice.
  2. **Logit-lens instability**: apply final norm; if unstable, treat tuned-lens translators as a separate follow-up (not required for the decisive test).
  3. **Dataset mismatch**: evaluate on the LongCodeZip-provided RepoQA variant for comparability.

---

## Success Criteria

**Hypothesis (directional):** ShallowPPL will match Full AMI accuracy closely while reducing compression time.

**Decision Rule (concrete):**
- **Proceed** if ShallowPPL achieves all of:
  1) **≥1.5× compression-time speedup** vs Full AMI on at least one benchmark, and
  2) **≤1.0 absolute point** drop on the primary metric (Completion ES; RepoQA AvgAcc), and
  3) On the disagreement slice (\(\mathrm{Jaccard}(S_{\text{AMI}},S_{\text{RAG}})\le 0.2\)), \(\mathrm{Jaccard}(S_{\text{Shallow}},S_{\text{AMI}}) \ge \mathrm{Jaccard}(S_{\text{Shallow}},S_{\text{RAG}})+0.10\).
- **Pivot** if quality holds but speedup <1.5× (apply ShallowPPL only to fine stage or increase \(L\)).
- **Refute** if ShallowPPL quality approaches the RAG baseline or the diagnostic indicates lexical collapse.

---

## Impact Statement

If ShallowPPL works, LongCodeZip-style semantic compression becomes cheaper to deploy for long-context code tasks, improving time-to-first-token and reducing GPU cost. If it fails, it suggests AMI’s relevance signal requires deeper computation, guiding future compressors toward different fast relevance signals.

---

## References

- LongCodeZip: `./references/LongCodeZip-Compress-Long-Context-for-Code-Language-Models/meta/meta_info.txt`
- RepoQA: `./references/RepoQA-Evaluating-Long-Context-Code-Understanding/meta/meta_info.txt`
- Tuned Lens: `../../papers/paper_summaries/Eliciting Latent Predictions from Transformers with the Tuned Lens.md`
- Additional citations are listed inline in the Related Work section.
