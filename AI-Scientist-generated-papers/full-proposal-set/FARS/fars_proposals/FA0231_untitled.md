# untitled

# Draft-and-Continue Self-Consistency: Two-Stage Branch Budgeting without Verifiers

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Many state-of-the-art language model (LM) reasoning systems improve accuracy by spending more test-time compute: sampling multiple chain-of-thought (CoT) solutions, searching over reasoning branches, and/or iteratively revising candidate solutions. This trend is especially visible in research-level math systems (e.g., DeepMind's Aletheia agent in *Towards Autonomous Mathematics Research*), where solving a single problem may involve many parallel attempts and multiple rounds of verification and revision (https://arxiv.org/abs/2602.10177).

However, test-time scaling is expensive. In the common "best-of-N" / self-consistency paradigm, we generate N full-length solutions and then select or vote at the end. This wastes compute because many branches become unpromising early (e.g., they quickly converge to a minority answer), yet we still pay for finishing their full solutions.

A practical question for 2026 is whether we can keep most of the gains of test-time scaling while reducing cost using simple, deployment-friendly allocation rules that do not require training a process reward model (PRM) or a verifier.

### The Problem

Most adaptive test-time compute methods focus on *when to stop sampling more full solutions* (early stopping) or *how to prune redundant full solutions* (similarity pruning). Fewer methods isolate a simpler and complementary decision: **after seeing only short "draft" attempts, which branches should receive the remaining compute budget to finish and verify the solution?**

This decision matters in solver-verifier / generate-verify-revise loops (like Aletheia), but it is also relevant in plain self-consistency for math reasoning. If we can cheaply identify which answers are likely to be correct from short drafts, we can reallocate tokens away from dead-end branches and towards deeper verification of the most plausible answers.

The main risk is the "confidently wrong" failure mode: if the early majority answer is wrong, pruning can remove the only correct minority branch and hurt accuracy.

### Key Insight and Hypothesis

**Key insight**: Even without a learned verifier, short draft attempts often contain an implicit signal of which answer is most plausible. A minimal signal is **answer frequency** across drafts (a coarse self-consistency vote computed very early).

**Mechanism hypothesis (why this might be non-trivially true):** For many math problems, the final answer is determined by an early discrete "commit point" in the reasoning (e.g., which substitution/case split is chosen). Once a branch makes that early commitment, continuing it mostly (i) completes algebra and (ii) checks arithmetic. Thus, an interim answer emitted early is a noisy but informative indicator of which branch has selected a correct high-level path. Spending more continuation tokens on the early-majority answer(s) increases the chance that at least one correct-high-level branch survives long enough to self-correct local mistakes.

This hypothesis implies a sharp prediction: DCS should help on problems where the stage-1 majority is already correct, and hurt (unless hedged) on problems where the stage-1 majority is wrong.

**Main hypothesis**: Under a fixed total generation-token budget per problem, a two-stage schedule that (i) samples many short drafts to estimate an early vote distribution, then (ii) continues only a small number of branches from the top-voted answers, will improve exact-match accuracy on competition math compared to uniform self-consistency at the same token budget.

This could fail if (a) interim answers at `T_draft` are mostly uninformative (late-answer problems), or (b) majority-wrong cases are frequent enough that pruning dominates.

---

## Proposed Approach

### Overview

We propose **Draft-and-Continue Self-Consistency (DCS)**, a training-free, two-stage branch budgeting rule:

1. **Draft stage**: Sample B independent draft solutions with a small token budget `T_draft`.
2. **Early vote**: Extract each draft's interim answer and compute a vote histogram over answers.
3. **Continuation allocation**: Select a small subset of branches (k << B) to continue, prioritizing branches belonging to the top-voted answers (with an explicit diversity hedge to reduce majority-wrong failures).
4. **Continue stage**: Continue only the selected branches for an additional `T_cont` tokens (so the per-continued-branch maximum is `T_full = T_draft + T_cont`).
5. **Final aggregation**: Extract final answers from the continued branches and select by majority vote (same aggregator as standard self-consistency).

**Token budget matching.** DCS uses approximately:

`Tokens_DCS = B*T_draft + k*T_cont`.

A compute-matched uniform self-consistency baseline uses:

`Tokens_SC = N_full*T_full`,

where `N_full = floor(Tokens_DCS / T_full)`.

Concrete default hyperparameters for the main experiment:
- `T_full = 2048`, `T_draft = 512`, `T_cont = 1536`
- `B = 11`, `k = 3`  -> `Tokens_DCS = 10240`
- `N_full = 5`       -> `Tokens_SC = 10240`

(We choose a larger `T_full` than 1024 to reduce truncation of longer CoT traces; a sensitivity check at `T_full=4096` is optional.)

### Method Details

#### Prompting and answer extraction

We use explicit tags so answer extraction is fully automated:

- Draft stage prompt requests: short reasoning and an **`INTERIM_ANSWER:`** line.
- Continue stage prompt requests: continue from the draft, verify/correct if needed, and output **`FINAL_ANSWER:`**.

**Prompting control:** To avoid a prompt-format confound, we use the *same* tags in all conditions. In Uniform SC, each full sample is prompted to include both `INTERIM_ANSWER:` (somewhere before the end) and `FINAL_ANSWER:` (at the end). We then ignore `INTERIM_ANSWER:` for Uniform SC scoring and only use `FINAL_ANSWER:` for evaluation. This keeps output-format constraints matched across conditions.

We evaluate with standard MATH answer normalization (e.g., extracting `\\boxed{...}` when present; otherwise parsing `FINAL_ANSWER:`). All conditions use the same final-answer formatting requirement and the same answer-extraction code.

For the **interim vote**, we parse the draft's `INTERIM_ANSWER:`. Drafts without a parseable interim answer are assigned to a special `NULL` bucket and are **not** eligible for continuation unless all drafts are unparseable (in that case, DCS falls back to continuing k arbitrary drafts, which reduces to uniform truncated sampling). We treat final unparseable answers as incorrect.

#### Continuation allocation rule (core mechanism)

Let `a_i` be the interim answer extracted from draft `i`.

- Compute vote counts `c(a) = #{ i : a_i = a }`.
- Let `a1` and `a2` be the top-1 and top-2 answers by count.

We allocate continuation slots with a simple hedge:

- If there is no `a2` (all drafts agree or only one parseable answer exists), continue k branches from `a1`.
- Otherwise, continue `k-1` branches from `a1` and `1` branch from `a2`.

Branches within an answer cluster are chosen arbitrarily (e.g., earliest sampled). This keeps scoring cost at effectively zero (only string parsing).

#### Why continuation can help

Uniform self-consistency spends most of its budget on finishing full solutions for every sampled branch. DCS instead spends a small budget to measure early consensus, then uses the remaining budget to deepen verification (more tokens) only where the drafts suggest it is most useful.

---

## Related Work

### Field Overview

**Test-time scaling for reasoning.** Sampling multiple CoT solutions and aggregating by majority vote (self-consistency) is a standard baseline for improving reasoning without additional training (https://arxiv.org/abs/2203.11171). Many recent systems push this further with search and verification, including solver-verifier loops and multi-branch exploration for math and code (https://arxiv.org/abs/2602.10177; https://arxiv.org/abs/2305.20050).

**Adaptive sampling and early stopping.** Adaptive-Consistency and follow-up work reduce cost by stopping sampling once the answer vote is statistically stable, but they still operate on full solutions and do not allocate a remaining token budget to *continuations* of selected branches (https://arxiv.org/abs/2305.11860; https://arxiv.org/abs/2511.02603).

**Pruning, prefix reuse, and tree search.** Other lines improve efficiency by pruning redundant paths (e.g., similarity pruning in Slim-SC) or reusing high-confidence prefixes (Path-Consistency), and PRM-guided tree search methods cluster or diversify branches using intermediate checkpoints (https://arxiv.org/abs/2509.13990; https://arxiv.org/abs/2409.01281; https://arxiv.org/abs/2505.17829). These methods are often more complex and/or require PRMs.

DCS targets a narrower question: **is a very cheap early vote signal sufficient to support a coarse two-stage continuation schedule that improves accuracy per token without any learned verifier?**

### Related Papers

- **[Towards Autonomous Mathematics Research](https://arxiv.org/abs/2602.10177)**: Introduces Aletheia (generator-verifier-reviser) and highlights that inference-time scaling can be compute-hungry.
- **[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)**: Core majority-vote baseline for test-time scaling in reasoning.
- **[Let's Sample Step by Step: Adaptive-Consistency](https://arxiv.org/abs/2305.11860)**: Stops sampling when majority is statistically stable; complementary to continuation budgeting.
- **[Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning](https://arxiv.org/abs/2401.10480)**: Early stopping variants for self-consistency (stops sampling when window vote entropy collapses).
- **[CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency](https://arxiv.org/abs/2511.02603)**: Bayesian aggregation + early stopping; demonstrates benefits of confidence signals derived from token probabilities for adaptive sampling.
- **[Optimal Self-Consistency / Blend-ASC](https://arxiv.org/abs/2511.12309)**: Theoretically grounded adaptive self-consistency with dynamic sample allocation; strong deployable baseline family for efficient SC.
- **[A Single Revision Step Improves Token-Efficient LLM Reasoning (PACER)](https://arxiv.org/abs/2602.02828)**: Training-free consensus-packet + short revision step that matches 256-sample majority voting with fewer tokens; closest in spirit (two-phase) but uses explicit peer-revision rather than continuation allocation.
- **[Reasoning-Aware Self-Consistency (RASC)](https://arxiv.org/abs/2408.17017)**: Uses lightweight features to score reasoning paths and stop earlier.
- **[Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency](https://arxiv.org/abs/2509.13990)**: Online pruning of redundant reasoning chains via inter-chain similarity.
- **[Path-Consistency: Prefix Enhancement for Efficient Inference in LLMs](https://arxiv.org/abs/2409.01281)**: Reuses high-confidence prefixes to guide later sampling while maintaining diversity.
- **[Stepwise Reasoning Checkpoint Analysis (SRCA)](https://arxiv.org/abs/2505.17829)**: Uses intermediate answer checkpoints, answer clustering, and PRM-guided search/augmentation.
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)**: Search over reasoning branches with evaluation-guided expansion.
- **[Graph of Thoughts](https://arxiv.org/abs/2308.09687)**: Generalizes ToT with graph-based reasoning state transitions.
- **[Language Agent Tree Search (LATS)](https://arxiv.org/abs/2310.04406)**: MCTS-style agent search with self-evaluation signals.
- **[Scaling Test-time Compute for LLM Agents](https://arxiv.org/abs/2506.12928)**: Studies agentic test-time scaling and discusses diversity-aware tree search variants (e.g., DVTS) as a baseline family.
- **[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)**: Process reward models and verifier-guided selection for math reasoning.
- **[CarBoN: Calibrated Best-of-N Sampling Improves Test-time Reasoning](https://arxiv.org/abs/2510.15674)**: Two-phase exploration/exploitation with PRM-scored calibration.
- **[Instance-Adaptive Inference-Time Scaling with Calibrated Process Reward Models](https://openreview.net/forum?id=FGlsbGp1Bc)**: Uses calibrated PRMs to allocate per-instance sampling budgets.
- **[EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling](https://arxiv.org/abs/2510.11170)**: Uses token-level entropy to branch only at high-uncertainty points.
- **[Conformal Thinking: Risk Control for Reasoning on a Compute Budget](https://arxiv.org/abs/2602.03814)**: Distribution-free risk control for early stopping in reasoning under compute budgets.
- **[FEval-TTC: Fair Evaluation Protocol for Test-Time Compute](https://arxiv.org/abs/2511.01203)**: Cached-response framework and cost models for TTC evaluation.
- **[Anytime Verified Agents (AVA)](https://openreview.net/forum?id=JMDCMf7mlF)**: Budget-aware controller allocating compute across sampling/search/verification.
- **[Strategic Scaling of Test-Time Compute: A Bandit Learning Approach](https://openreview.net/forum?id=0mNnINd2z5)**: Bandit formulation for allocating sampling across queries.
- **[Do Not Waste Your Rollouts: Recycling Search Experience](https://arxiv.org/abs/2601.21684)**: Reuses intermediate insights across rollout batches to improve scaling efficiency.
- **[SPIRIT: Stepwise Perplexity-Guided Refinement for CoT](https://arxiv.org/abs/2502.13260)**: Coarse-to-fine pruning/refinement of reasoning steps.
- **[Pruning the Unsurprising (ASAP)](https://arxiv.org/abs/2508.05988)**: CoT compression for code reasoning via surprisal, targeting inference efficiency.
- **[Reasoning at the Right Length: Adaptive Budget Forcing (ABF)](https://openreview.net/forum?id=ieBgxTG7Mt)**: Adapts reasoning length online using confidence signals.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Uniform sampling | Fixed N full solutions then vote/select | Self-Consistency | GSM8K, MATH-500, AIME | Wastes compute on easy instances and dead branches |
| Adaptive sample count | Stop sampling early when confident | Adaptive-Consistency, ESC, CGES | GSM8K, MATH-500, HumanEval | Mostly decides "how many full samples", not how to allocate continuation tokens |
| Prune redundant chains | Drop similar/full chains during SC | Slim-SC | GPQA, AIME | Often requires embeddings/similarity; may prune late |
| Prefix reuse | Extract prefix from confident branch and guide new samples | Path-Consistency | GSM8K, HumanEval | Does not allocate continuation budget across existing branches |
| Search-based scaling | Tree/MCTS/beam-like branching with scoring | ToT, LATS, DVTS, SRCA | MATH-500, OlympiadBench | Often PRM- or judge-dependent; more complex to deploy |
| Two-phase distribution shaping | Use exploration samples to adjust generation distribution | CarBoN | MATH-500, AIME | Requires PRM; per-instance optimization overhead |
| Multi-dimension budget controllers | Allocate compute across sampling/search/verification/tools | AVA, Strategic Scaling | GSM8K, HotpotQA, HumanEval | Heavier machinery; may require calibrated uncertainty |
| CoT compression / length control | Shorten reasoning traces | SPIRIT, ASAP, ABF | MATH-500, code benchmarks | Changes reasoning format; may trade off interpretability |

### Closest Prior Work

1) **Adaptive-Consistency (Aggarwal et al., 2023)**
- What it does: sequentially samples full solutions and stops once the vote is confident.
- Key limitation for our question: does not support a "draft then continue" schedule where early information is used to decide which branches to finish.
- Why different: DCS explicitly tests whether short drafts can support reallocating *tokens within a problem* from many branches to a few continuations.

2) **Slim-SC (Hong et al., 2025)**
- What it does: prunes redundant full reasoning chains online using thought embedding similarity.
- Key limitation: pruning decisions are based on similarity, not early answer votes; and chains are often already long when pruned.
- Why different: DCS uses an earlier, cheaper signal (interim answer frequency) and stops generation early for pruned branches.

3) **Path-Consistency (Zhu et al., 2024)**
- What it does: extracts a prefix from a high-confidence branch and reuses it to guide subsequent sampling.
- Key limitation: focuses on prefix reuse rather than allocating a fixed remaining budget to continuations of selected branches.
- Why different: DCS keeps branches independent, but allocates continuation compute based on an early vote.

4) **SRCA (Wang et al., 2025)**
- What it does: injects intermediate answer checkpoints, clusters paths, and uses a PRM for diversity-aware search and candidate augmentation.
- Key limitation: PRM-dependent and more complex (multi-step checkpoints, clustering, beam-like maintenance).
- Why different: DCS asks whether a much simpler two-stage rule (no PRM, one early checkpoint) can recover some of the benefits.

5) **CarBoN (Tang et al., 2025)**
- What it does: uses PRM-scored exploration samples to fit per-instance calibration parameters (delta, temperature) and improve best-of-N.
- Key limitation: requires a PRM and per-instance optimization.
- Why different: DCS avoids PRMs entirely and treats early votes as the only signal.

**Novelty Kill Search Summary:** Searched for combinations of "successive halving chain-of-thought", "draft then continue self-consistency", "partial answer voting continue reasoning", "hyperband LLM reasoning", and checked local KB summaries for pruning/prefix-reuse/self-consistency variants (Slim-SC, Path-Consistency, SRCA, CGES, Adaptive-Consistency, CarBoN, IAS) as of 2026-02-22. Found related pruning/prefix/search methods but no exact match for a two-stage self-consistency schedule that uses only interim answer frequency (no PRM/verifier/embeddings) to decide which branches to continue.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Self-Consistency | N full solutions then majority vote | Fixed budget wastes compute | Two-stage draft-then-continue | More breadth early, more depth only where needed |
| Adaptive-Consistency | Stops sampling when vote is confident | No continuation; still full-solution samples | Continue a small set of branches | Uses remaining budget to deepen verification |
| Slim-SC | Similarity prune redundant chains | Requires embeddings; may prune late | Prune after short drafts via answer votes | Earlier pruning reduces wasted tokens |
| Path-Consistency | Reuse high-confidence prefixes | Not a continuation budget policy | Continue selected branches, keep others independent | Tests whether early votes are sufficient without prefix sharing |
| SRCA | Intermediate checkpoints + PRM search | PRM-dependent and complex | Remove PRM; single early checkpoint | Simpler, training-free, easier to deploy |

---

## Experiments

### Experimental Setup

**Task setting:** Automatically graded competition math with exact-match answers.

**Baseline Ladder (REQUIRED):**
1. **Greedy CoT (k=1)**: single sample, `max_new_tokens=T_full`.
2. **Uniform self-consistency (SC)**: sample `N_full` full solutions, vote at the end.
3. **CGES-LNS (baseline)**: confidence-guided early stopping with token-level length-normalized scoring (LNS arithmetic mean), per CGES (Aghazadeh et al., 2025). Uses the same total *call budget* as SC but adaptively stops earlier on easy items.
4. **DCS (ours)**: sample `B` drafts then continue `k` branches, matched total generation-token budget.

(We include CGES-LNS as the strongest deployable baseline that also uses a training-free signal derived from model token probabilities.)

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Qwen2.5-Math-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct | Chosen to avoid ceiling effects seen on very-strong distilled models; also matches the base model family used for MATH500 in CGES (Sec. 4.1) |

**Training Data:**
- None (inference-only).

**Decoding / sampling:**
- Temperature = 0.8, top_p = 0.95 (same across conditions)
- Seeds = [42, 123, 456]
- `T_full=2048`, `T_draft=512`, `T_cont=1536` (primary)

**Resource Estimate:**
- Approx tokens per problem: `Tokens_DCS = 10240` and `Tokens_SC = 10240`.
- For MATH-500 and 3 seeds: ~`500 * 10240 * 3 ~= 15.4M` generated tokens per method.
- 4 methods (Greedy, Uniform SC, CGES-LNS, DCS) -> ~62M generated tokens total.
- Expected to fit within **<= 250 A100 GPU-hours** using batched vLLM inference for a 7B model (larger `T_full` and an extra baseline).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| MATH-500 | 500 competition-style math problems with exact answers | Accuracy (exact match), avg generated tokens | test | https://huggingface.co/datasets/Hendrycks/competition_math | https://github.com/EleutherAI/lm-evaluation-harness (math task) |

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Accuracy (mean +- std) | Avg generated tokens | Source | Notes |
|---|---|---|---:|---:|---|---|
| Greedy CoT (k=1) | Qwen2.5-Math-7B-Instruct | MATH-500 | TBD | TBD | - | Sanity run (should underperform SC) |
| Uniform SC (N_full=5, T_full=2048) | Qwen2.5-Math-7B-Instruct | MATH-500 | TBD | TBD | - | Primary baseline |
| CGES-LNS (B=5 calls max, adaptive early stop) | Qwen2.5-Math-7B-Instruct | MATH-500 | TBD | TBD | CGES Alg.2 + LNS arith | Implement confidence C_t via mean token probability; stop when max posterior >= gamma |
| **DCS (B=11, k=3, T_draft=512, T_cont=1536)** | Qwen2.5-Math-7B-Instruct | MATH-500 | TBD | TBD | - | To be verified |

### Ablation Studies

| Variant | What changes | Expected finding |
|---|---|---|
| DCS (majority-only) | Continue k branches from top-1 interim answer only (no top-2 hedge) | Worse on majority-wrong subset; may be slightly better on majority-correct subset |
| DCS (proportional hedge) | Allocate continuation slots proportional to interim vote counts over top-3 answers (at fixed k) | More robust on majority-wrong cases; may slightly reduce gains on easy cases |
| Uniform truncated (no vote) | Sample B drafts with T_draft, then continue k random branches (ignores votes) | Matches DCS tokens but removes the allocation mechanism; should underperform DCS if votes are informative |

### Experimental Rigor

**Variance & Reproducibility:**
- Use sampling seeds = [42, 123, 456] for each method.
- Report mean +- std across seeds.

**Validity & Controls (top confounders):**
1. **Token-budget mismatch**: DCS and Uniform SC are matched by construction (`Tokens_DCS = Tokens_SC`). For CGES-LNS, we treat `N_full*T_full` as a per-question *cap* (max calls = `N_full`) and report **actual** avg calls and avg generated tokens.
2. **Prompt distribution shift**: All conditions use the same answer-format tags (`INTERIM_ANSWER:` and `FINAL_ANSWER:`). In Uniform SC and CGES-LNS, we explicitly request an `INTERIM_ANSWER` after roughly `T_draft` tokens before continuing, so that the "early answer" appears at a comparable point in the trace as in DCS.
3. **Answer extraction errors**: Use the same answer normalization code for all methods; report fraction of unparseable outputs.

**Sanity checks:**
- Verify that Uniform SC improves over Greedy CoT for this model on MATH-500 (directional check).

**Data leakage note:** MATH-style problems may overlap with pretraining. This does not invalidate the comparison because all conditions use the same base model and differ only in inference allocation.

### Analysis (Optional)

- **Stratified failure-mode analysis**: Partition problems by whether the stage-1 majority interim answer is correct, and report DCS vs Uniform SC accuracy in each stratum. This tests the core risk: majority-wrong amplification.

---

## Success Criteria

**Hypothesis** (directional): DCS improves accuracy at a fixed generation-token budget by reallocating continuation tokens towards the most plausible answers.

**Decision Rule** (concrete):
- **Proceed** if DCS improves MATH-500 accuracy by **>= 3.0 percentage points** over Uniform SC at matched tokens, *or* if DCS matches Uniform SC within 1.0 pp while using **>= 25% fewer generated tokens** (measured empirically via avg tokens), and the direction is consistent across 3 seeds.
- **Pivot** if DCS helps on the majority-correct stratum but hurts substantially on majority-wrong instances; try a more conservative hedge (allocate continuation proportional to vote counts over the top-3 answers at fixed k).
- **Refute** if DCS underperforms Uniform SC by >= 1.0 pp, or if DCS is clearly dominated by CGES-LNS at the same (or lower) token budget.

---

## Impact Statement

If successful, DCS provides a minimal, training-free primitive for making test-time scaling more cost-effective in math reasoning and solver-verifier pipelines: spend a small budget to estimate early consensus, then spend the remaining budget only where it is most useful. This could reduce the cost of best-of-N style inference for practitioners who cannot afford PRM training or heavy search frameworks.

---

## References

- Towards Autonomous Mathematics Research - Feng et al., 2026. https://arxiv.org/abs/2602.10177
- Self-Consistency Improves Chain of Thought Reasoning in Language Models - Wang et al., 2022. https://arxiv.org/abs/2203.11171
- Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs - Aggarwal et al., 2023. https://arxiv.org/abs/2305.11860
- CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency - Aghazadeh et al., 2025. https://arxiv.org/abs/2511.02603
- Reasoning-Aware Self-Consistency (RASC) - Wan et al., 2024. https://arxiv.org/abs/2408.17017
- Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency - Hong et al., 2025. https://arxiv.org/abs/2509.13990
- Path-Consistency: Prefix Enhancement for Efficient Inference in LLMs - Zhu et al., 2024. https://arxiv.org/abs/2409.01281
- Stepwise Reasoning Checkpoint Analysis - Wang et al., 2025. https://arxiv.org/abs/2505.17829
- Tree of Thoughts - Yao et al., 2023. https://arxiv.org/abs/2305.10601
- Graph of Thoughts - Lei et al., 2023. https://arxiv.org/abs/2308.09687
- Language Agent Tree Search (LATS) - (authors not listed here), 2023. https://arxiv.org/abs/2310.04406
- CarBoN: Calibrated Best-of-N Sampling Improves Test-time Reasoning - Tang et al., 2025. https://arxiv.org/abs/2510.15674
- Instance-Adaptive Inference-Time Scaling with Calibrated Process Reward Models - Park et al., (year). https://openreview.net/forum?id=FGlsbGp1Bc
- EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling - Scalena et al., 2025. https://arxiv.org/abs/2510.11170
- FEval-TTC: Fair Evaluation Protocol for Test-Time Compute - Rumiantsev et al., 2025. https://arxiv.org/abs/2511.01203
- Anytime Verified Agents: Adaptive Compute Allocation for Reliable LLM Reasoning under Budget - (authors not listed here; OpenReview submission), 2026. https://openreview.net/forum?id=JMDCMf7mlF
- Strategic Scaling of Test-Time Compute: A Bandit Learning Approach - (authors not listed here; OpenReview submission), 2026. https://openreview.net/forum?id=0mNnINd2z5
- Do Not Waste Your Rollouts: Recycling Search Experience for Efficient Test-Time Scaling - Wang et al., 2026. https://arxiv.org/abs/2601.21684
- SPIRIT: Stepwise Perplexity-Guided Refinement for CoT - Cui et al., 2025. https://arxiv.org/abs/2502.13260
- Pruning the Unsurprising: Efficient Code Reasoning via First-Token Surprisal - Zeng et al., 2025. https://arxiv.org/abs/2508.05988
- Reasoning at the Right Length: Adaptive Budget Forcing - (authors not listed here; OpenReview submission), 2026. https://openreview.net/forum?id=ieBgxTG7Mt
