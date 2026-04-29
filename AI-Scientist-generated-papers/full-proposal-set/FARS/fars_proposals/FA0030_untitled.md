# untitled

# Answerability-Gain Rewards for Evidence-Label-Free GRU-Mem Gating

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used for question answering and agentic workflows that require reasoning over very long documents (hundreds of thousands to millions of tokens). Because standard Transformer attention has high compute/memory cost with sequence length, many systems process a long context incrementally and maintain a compact *textual memory* that is updated as the model reads the document chunk-by-chunk.

**[MemAgent](./references/MemAgent-Reshaping-Long-Context-LLM-with-Multi-Conv-RL-based-Memory-Agent/meta/meta_info.txt)** is a representative approach: it uses the same LLM in two prompt roles. A **memory agent** reads one chunk at a time and writes a bounded textual memory (e.g., a fixed token budget summary), and an **answer agent** produces the final answer from the final memory. This recurrent workflow keeps per-step compute bounded and empirically supports extreme length extrapolation, but it can suffer from **memory pollution/saturation** (updating on distractors until the memory buffer reaches its maximum size) and **wasted computation** (processing chunks after all relevant evidence has been read).

**[GRU-Mem](./references/When-to-Memorize-and-When-to-Stop-Gated-Recurrent-Memory-for-Long-Context-Reasoning_1/meta/meta_info.txt)** addresses these issues by adding two *text-controlled gates* to the recurrent workflow: an **update gate** decides whether a proposed memory update should be applied, and an **exit gate** decides whether to stop scanning further chunks and answer early. In a controlled setting where the last evidence occurs early, GRU-Mem reports large efficiency gains (e.g., Table 2 shows ~4× lower inference time than MemAgent at similar accuracy).

### The Problem

GRU-Mem’s training objective depends on **supervision that is often unavailable outside synthetic benchmarks**:

1. **Evidence-chunk labels for the update gate**. GRU-Mem uses a per-step reward that requires knowing whether each chunk contains evidence (rewarding `<check>yes</check>` on evidence chunks and `<check>no</check>` otherwise).
2. **The last-evidence position for the exit gate**. GRU-Mem uses a trajectory-level exit reward based on the index of the last evidence chunk (`t_last_evidence`).

These labels are easy to obtain in fully synthetic tasks (e.g., Needle-in-a-Haystack), but they are rarely available for realistic long-context QA corpora and many downstream “read a long file and answer” settings. Requiring them limits where gated recurrent memory can be trained and deployed.

A practitioner who wants to add GRU-Mem-style gating to a recurrent memory workflow today typically faces a choice:
- either restrict training to synthetic tasks where evidence positions are known,
- or invest in bespoke labeling pipelines / heuristics to approximate evidence chunks and last-evidence indices.

This proposal targets a concrete, decision-changing question: **can we train useful update/exit gates without any evidence-position supervision**, using only standard QA supervision (question, long context, ground-truth answer)?

### Key Insight and Hypothesis

**Key insight:** even when evidence-position labels are unavailable, we can create *dense learning signals* by measuring whether a proposed memory update makes the correct answer *more predictable*. Concretely, we can compute an **answerability gain** signal, defined as the change in teacher-forced log-likelihood of the ground-truth answer when using the candidate memory instead of the previous memory.

**Hypothesis:** Replacing GRU-Mem’s evidence-label rewards (`r_update`, `r_exit`) with an **answerability-gain reward** plus explicit per-step/per-update costs will learn gates that (i) skip distractor chunks and (ii) stop scanning once additional chunks cease to improve answerability, yielding an accuracy–efficiency trade-off close to gold-supervised GRU-Mem and better than outcome-only training.

**Relation to IGPO and novelty claim:** IGPO (Information Gain-based Policy Optimization) proposes using the marginal increase in the log-likelihood of the correct answer as a turn-level intrinsic reward for training multi-turn LLM agents. Our setting is a different control problem: *chunk-level* memory **accept/reject** decisions and **early stopping** in a recurrent long-context workflow. We focus on learning gates on top of an existing memory-writing policy (starting from MemAgent/RL-MemoryAgent), and we propose a **gate-only fine-tuning** implementation (optimize only the `<check>` / `<next>` decisions, optionally masking the RL loss on `<update>` spans) so the policy cannot trivially “game” the reward by changing the memory-writing behavior.

Why this could fail: answerability gain can be **delayed** in multi-hop QA (early evidence may not increase answer likelihood until later evidence appears), which could cause the update gate to under-update. The proposed experiments explicitly include multi-hop QA and a distractor setting to test for such failure modes.

---

## Proposed Approach

### Overview

We propose an *evidence-label-free* reward design for training GRU-Mem’s update/exit gates in a MemAgent-style recurrent workflow:

- The memory agent reads chunk-by-chunk and outputs (a) an update decision, (b) a candidate memory, and (c) an exit decision.
- We compute a **dense intrinsic reward** from the *change in teacher-forced answer log-likelihood* caused by applying the candidate memory.
- We add explicit **compute costs** (penalize processing more chunks; penalize applying updates) so that early exit emerges without any `t_last_evidence` supervision.

### Method Details

#### Recurrent workflow with gates
Let the long context be split into chunks \(C_1,\dots,C_T\). The memory agent maintains a textual memory \(M_{t}\) (bounded length). At each step \(t\), the memory agent produces:

- **Update decision** \(U_t \in \{0,1\}\) via `<check>yes/no</check>`
- **Candidate memory** \(\tilde{M}_t\) via `<update>...</update>`
- **Exit decision** \(E_t \in \{0,1\}\) via `<next>end/continue</next>`

If \(U_t=1\), we set \(M_t \leftarrow \tilde{M}_t\); otherwise \(M_t \leftarrow M_{t-1}\). If \(E_t=1\), we stop and answer using the current memory.

#### Answerability-gain reward (label-free)
We assume standard QA supervision \((Q, C, A_{gt})\), where \(A_{gt}\) is the ground-truth answer string (or one of multiple acceptable strings).

We define the *teacher-forced* log-likelihood of the ground-truth answer under the answer-agent prompt:
\[
\mathcal{L}(M) = \log p_\theta(A_{gt}\mid Q, M).
\]
Teacher forcing here means we score the probability of the known answer tokens by conditioning on the ground-truth previous answer tokens, i.e., a forward-only scoring pass rather than free generation.
We compute an **answerability gain** for the candidate memory at step \(t\):
\[
\Delta_t = \mathcal{L}(\tilde{M}_t) - \mathcal{L}(M_{t-1}).
\]

We use \(\Delta_t\) as a dense reward for the update decision:
\[
 r^{\text{gain}}_{t} = U_t \cdot (\Delta_t - \mu),
\]
where \(\mu\ge 0\) is a per-update cost / minimum-gain threshold. This makes harmful updates (\(\Delta_t < \mu\)) explicitly negative-reward events and encourages updating only when the candidate memory measurably improves answerability.

#### Compute / efficiency shaping (label-free early exit)
We penalize reading more chunks with a per-step cost:
\[
 r^{\text{step}}_{t} = -\lambda,
\]
where \(\lambda>0\) encourages the exit gate to terminate once additional chunks provide low marginal value.

#### Terminal rewards and format reward
We include:
- **Outcome reward** \(r^{\text{outcome}}\): exact-match / equivalence-based QA correctness.
- **Format reward** \(r^{\text{format}}\): strict parseability of the required tags (as in GRU-Mem).

The total trajectory reward is:
\[
 r_{\text{traj}} = r^{\text{outcome}} + r^{\text{format}} + \sum_{t=1}^{t_{exit}} \left(r^{\text{gain}}_t + r^{\text{step}}_t\right).
\]

#### Policy optimization (GRPO) and implementation details

We train with **GRPO (Group Relative Policy Optimization)**, a PPO-style policy gradient method that samples multiple rollouts per prompt and uses group-relative advantages (no separate value/critic model). GRU-Mem and MemAgent both use GRPO-style objectives.

**Gate-only fine-tuning (anti-reward-hacking).** A naive IGPO-style application would backpropagate through the entire memory-agent output, including the `<update>...</update>` span. This gives the policy a trivial way to increase \(\mathcal{L}(M)\): learn to write the answer string into memory, rather than learning *when to accept* a memory update.

To isolate the gating problem, we propose a **gate-only optimization** variant:
- Freeze the base memory-writing behavior by starting from `BytedTsinghua-SIA/RL-MemoryAgent-7B`.
- Apply the GRPO loss only to the **gate decisions** (the tokens that realize `<check>yes/no</check>` and `<next>continue/end</next>`), and mask the RL loss on the `<update>...</update>` tokens (optionally also mask `<think>`).
- This makes the action space closer to a binary accept/reject + stop/continue controller on top of an existing memory writer.

**Efficient computation of \(\Delta_t\).** Computing \(\Delta_t\) as written suggests two teacher-forced scoring passes per step. We can reduce overhead with caching:
- Maintain a cached score \(S_{t-1}=\mathcal{L}(M_{t-1})\).
- When the policy chooses \(U_t=1\), compute \(\mathcal{L}(\tilde{M}_t)\) with a single teacher-forced forward pass and set \(\Delta_t=\mathcal{L}(\tilde{M}_t)-S_{t-1}\).
- If \(U_t=1\), update the cache \(S_t\leftarrow\mathcal{L}(\tilde{M}_t)\); if \(U_t=0\), set \(\Delta_t=0\) and reuse \(S_t\leftarrow S_{t-1}\).

This makes the incremental cost of answerability shaping **at most one extra forward-only scoring pass** on steps where the agent updates, rather than two passes on every step. We will report the wall-clock overhead of this scoring relative to rollout generation in the final results.

### Key Innovations

1. **Evidence-label-free gating objective**: trains update/exit gates without evidence chunk labels or `t_last_evidence` supervision.
2. **Gate-only IGPO-style reward shaping**: applies answerability-gain rewards to *accept/reject* and *stop/continue* decisions while masking RL loss on memory-writing tokens to prevent reward hacking (e.g., writing the answer into memory).
3. **Compute-aware answerability scoring**: computes \(\Delta_t\) with caching so the overhead is at most one extra teacher-forced scoring pass on steps that actually update.
4. **Targeted robustness evaluation**: introduces a distractor-answer variant to diagnose when likelihood-based rewards spuriously prefer non-evidential chunks that contain the answer string.

---

## Related Work

### Field Overview

This proposal sits at the intersection of (i) long-context modeling, (ii) memory-augmented / recurrent long-context agents, and (iii) reinforcement learning with verifiable or intrinsic rewards.

**Long-context modeling** includes architectural approaches that extend Transformer context length (e.g., sparse attention or improved attention kernels) as well as *semi-parametric* methods that retrieve from external memory. In contrast, **recurrent memory agents** keep the model’s context window small and process the document sequentially, updating a compact textual memory.

**Reward design for long-horizon RL** is a key bottleneck for training recurrent workflows. Sparse terminal rewards (answer correctness only) can lead to poor credit assignment. Dense intrinsic rewards (e.g., information gain) have been proposed for multi-turn agents, but have not been systematically tested as a replacement for *evidence-position supervision* in gated recurrent memory.

### Related Papers

- **[When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning](./references/When-to-Memorize-and-When-to-Stop-Gated-Recurrent-Memory-for-Long-Context-Reasoning_1/meta/meta_info.txt)**: Introduces update/exit gates for recurrent memory, but trains them using evidence-position supervision.
- **[MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent](./references/MemAgent-Reshaping-Long-Context-LLM-with-Multi-Conv-RL-based-Memory-Agent/meta/meta_info.txt)**: Trains a recurrent memory agent with outcome-only verifiable rewards, but has no gating and can waste compute by scanning all chunks.
- **[RULER: What’s the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)**: Provides configurable synthetic long-context tasks (including QA with known golden paragraphs) suitable for automated gating evaluation.
- **[HotpotQA](https://arxiv.org/abs/1809.09600)**: Multi-hop QA dataset commonly used to synthesize long-context QA with known supporting paragraphs.
- **[SQuAD](https://arxiv.org/abs/1606.05250)**: Extractive QA dataset often used in long-context QA evaluations.
- **[DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)**: Introduces GRPO, a critic-free policy optimization method widely used for reasoning and RLVR.
- **[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)**: Canonical policy-gradient method underlying many PPO/GRPO-style optimizers.
- **[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)**: Describes practical stabilizations for GRPO/PPO-style RL on LLMs (clipping, sampling, token-level losses).
- **[Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)](https://arxiv.org/abs/2503.20783)**: Analyzes biases in GRPO-style objectives and proposes fixes; useful for stable training in our setting.
- **[GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL](<../../papers/paper_summaries/GDPO Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization.md>)**: Shows GRPO-style normalization can collapse signals in multi-reward settings and proposes decoupled normalization; relevant because our reward combines outcome, format, and efficiency terms.
- **[Information Gain-based Policy Optimization (IGPO)](<../../papers/paper_summaries/Information Gain-based Policy Optimization A Simple and Effective Approach for Multi-Turn LLM Agents.md>)**: Uses marginal improvement in ground-truth answer probability as an intrinsic per-turn reward for multi-turn agents; closest in reward form, but does not study gate-only optimization or chunk-level early stopping in recurrent long-context memory workflows.
- **[Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents (ReMemR1)](https://arxiv.org/abs/2509.23040)**: Adds dense step rewards for memory agents and a callback mechanism, but still relies on ground-truth answer strings/entities and does not address evidence-label-free gate supervision.
- **[Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)**: Retrieval-augmented seq2seq models for QA; a strong non-recurrent baseline family for knowledge-intensive QA.
- **[Fusion-in-Decoder (FiD)](https://arxiv.org/abs/2007.01282)**: Fuses multiple retrieved passages in the decoder; representative of multi-passage QA baselines.
- **[RETRO](https://arxiv.org/abs/2112.04426)**: Retrieval-enhanced language modeling for long contexts; contrasts with recurrent memory agents.
- **[Transformer-XL](https://arxiv.org/abs/1901.02860)**: Segment-level recurrence for Transformers; foundational for long-context recurrence ideas.
- **[Longformer](https://arxiv.org/abs/2004.05150)**: Sparse attention for long documents; representative architectural baseline direction.
- **[Confident Adaptive Language Modeling (CALM)](https://arxiv.org/abs/2207.07061)**: Early exiting for generation using confidence/calibration; conceptually related to exit gating.
- **[ConsistentEE](https://arxiv.org/abs/2312.11882)**: Reinforcement-learning formulation for early exit to align training and inference objectives.
- **[PABEE](https://arxiv.org/abs/2006.04152)**: Patience-based early exiting for BERT; classical early-exit baseline.
- **[DeeBERT](https://arxiv.org/abs/2004.12993)**: Dynamic early exiting with entropy thresholds; classical early-exit baseline.
- **[FastBERT](https://arxiv.org/abs/2004.14202)**: Self-distillation for early exit; another classical baseline.
- **[BERxiT](https://aclanthology.org/2021.eacl-main.8/)**: Early exiting with improved fine-tuning and learning-to-exit module.
- **[Adaptive Computation Time](https://arxiv.org/abs/1603.08983)**: Classic halting mechanism; conceptual precursor to learned stopping.
- **[Toolformer](https://arxiv.org/abs/2302.04761)**: Self-supervised tool-use filtering using loss improvements; analogous “loss-improvement as usefulness” principle.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: Interleaved reasoning and acting in multi-step agents; motivates dense rewards/credit assignment in long-horizon workflows.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Recurrent memory agents | Process long context chunk-by-chunk with bounded memory | MemAgent, GRU-Mem, ReMemR1 | RULER QA, long-context HotpotQA | Credit assignment; memory overwrite errors |
| Dense intrinsic rewards | Provide per-step reward via likelihood / information gain | IGPO, ReMemR1, Toolformer (filtering principle) | Multi-turn agent benchmarks; QA | May be delayed/noisy on multi-hop problems |
| Early exiting / halting | Learn to stop computation early based on confidence or RL | CALM, ConsistentEE, PABEE, DeeBERT, BERxiT | GLUE, generation tasks | Confidence miscalibration; training–inference mismatch |
| Retrieval-augmented QA | Retrieve passages and fuse for answering | RAG, FiD, RETRO | Natural Questions, TriviaQA | Retrieval latency; needs index/corpus |
| Long-context Transformer variants | Modify attention to handle long sequences | Transformer-XL, Longformer | Long-document tasks | Still expensive at extreme lengths |

### Closest Prior Work

1. **GRU-Mem**: Demonstrates that explicit update/exit gating can improve efficiency, but requires evidence-chunk labels and `t_last_evidence` to train the gates.
2. **MemAgent**: Shows outcome-only RL can train a recurrent memory agent, but lacks selective updating and early exit, leading to unnecessary computation and potential memory pollution.
3. **IGPO**: Proposes information/answerability gain as a dense intrinsic reward for multi-turn agents; we adapt this idea specifically to *memory update* and *early exit* decisions in a long-context recurrent workflow.
4. **ReMemR1**: Adds step-level rewards for memory agents and a callback mechanism, but does not address training gates without evidence labels and targets a different architectural extension (revisitable memory).
5. **ConsistentEE / CALM**: Address early exit, but primarily in the “exit Transformer layers/tokens” setting rather than “exit chunk-by-chunk document scanning” with an explicit memory state.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| GRU-Mem | Learns update/exit gates for recurrent memory | Needs evidence labels and `t_last_evidence` | Replace gate supervision with answerability gain + costs | Removes a major supervision bottleneck while keeping gating benefits |
| MemAgent | Outcome-only RL for recurrent memory updates | Always updates; no early exit | Add update/exit gating trained label-free | Should reduce memory pollution and wasted scanning |
| IGPO | Dense per-turn reward via ground-truth answer likelihood gain for multi-turn agents | Does not isolate gating; actions include full tool-use / text generation; no chunk-level stop controller | Use gain to train a gate-only accept/reject + stop/continue controller on top of a fixed memory writer | Avoids reward hacking and targets efficiency of long-document scanning |
| ReMemR1 | Revisitable memory + step rewards | Different goal (callback), still uses answer-derived signals | Focus on gating supervision without evidence labels | Lower complexity; isolates the gating bottleneck |
| ConsistentEE/CALM | Early exit for model layers/tokens | Different exit granularity; no memory state | Early exit at chunk level using memory-conditioned answerability | Directly optimizes long-document scanning cost |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| RL-MemoryAgent-7B | 7B | https://huggingface.co/BytedTsinghua-SIA/RL-MemoryAgent-7B | Starting point for recurrent memory workflow (MemAgent-style prompts) |
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Optional reference model / debugging baseline |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| BytedTsinghua-SIA/hotpotqa | RL training for recurrent workflow + gates | ~32,768 filtered train + dev | https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa | CC-BY-SA-4.0 |

**Other Resources (if applicable):**
- RULER benchmark generation + evaluation code: https://github.com/NVIDIA/RULER
- MemAgent training framework (verl-based): https://github.com/BytedTsinghua-SIA/MemAgent

**Resource Estimate**:
- **Compute budget**: ≤ 768 GPU-hours total.
  - Anchor points from literature: Dr. GRPO reports ~27 hours on 8×A100 for 7B GRPO-style RL on math reasoning (≈216 GPU-hours) ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)). ReMemR1 reports much larger runs (e.g., 80 hours on 32×H800 for 7B; ≈2560 GPU-hours) ([arXiv:2509.23040](https://arxiv.org/abs/2509.23040)), indicating that full-scale long-context RL can be expensive.
  - **Verification plan is intentionally downscaled and gate-only**: LoRA fine-tune only the gate tokens on top of `RL-MemoryAgent-7B` for a small number of RL updates (e.g., 200–400 optimizer steps) on a subset of the training set (e.g., 2k–5k samples) with training contexts ≈28K tokens / 200 docs.
  - **Planned runs and budget** (sequential):
    - Run A (Outcome-only + cost gates): up to 8×A100 × 20h = **160 GPU-hours**
    - Run B (Ours: gate-only answerability gain + cost): up to 8×A100 × 20h = **160 GPU-hours**
    - Optional ablation Run C (naive/unmasked gain, expected to reward-hack): cap at 8×A100 × 5h = **40 GPU-hours** with aggressive early stop
    - Evaluation (RULER + custom variant generation + inference): ≤ **120 GPU-hours**
    - **Total planned**: 160 + 160 + 40 + 120 = **480 GPU-hours** (≤768), leaving margin for reruns or longer training if learning is slow.
  - **Early stopping rule** (to protect budget): stop a run if after 25% of steps the policy is degenerate (e.g., updates >90% of chunks and never exits, or exits in the first 5% of chunks on >90% of examples), measured on a held-out dev subset.
- **GPU memory**: 7B + LoRA should fit on 1–2×A100 80GB for training with FSDP; rollout generation can use tensor parallelism if needed.
- **API usage**: none required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| RULER QA (HotpotQA / SQuAD variants) | RULER is a configurable synthetic long-context benchmark; its QA tasks insert gold paragraphs from HotpotQA/SQuAD into many distractor paragraphs to test long-context evidence retrieval and question answering | Accuracy (%; higher is better), EM (exact match; higher is better), Avg processed chunks (lower is better) | test | https://github.com/NVIDIA/RULER | Official RULER repo |
| Unbalanced “evidence in top 20%” variant | Same as above but the last evidence is forced to occur in the first 20% of chunks to stress early exit | Same as above | test | via RULER config | Official RULER repo + small config change |
| Distractor-answer variant (custom) | Same as above but add non-evidential paragraphs containing the answer string to test reward hacking / false updates | Same as above + update precision/recall (against gold evidence positions; higher is better) | test | derived from RULER | Small patch to RULER generator |

**Evaluation Scripts:**
- Use official RULER evaluation pipeline; adapt prompts to match GRU-Mem tag format.

### Main Results

#### Results Table

| Method | Base Model | Benchmark | Accuracy / EM | Avg processed chunks (↓) | Source | Notes |
|--------|------------|-----------|--------------|---------------------------|--------|------|
| MemAgent | Qwen2.5-7B (paper setting) | Evidence top-20%, 112K–896K | 79.69/171.65s (112K), 78.91/358.60s (224K), 78.12/804.23s (448K), 80.47/1691.93s (896K) | N/A | GRU-Mem Table 2 ([source section](./references/When-to-Memorize-and-When-to-Stop-Gated-Recurrent-Memory-for-Long-Context-Reasoning_1/sections/(RQ2) Study of Gating Mechanisms.md)) | Published result reports wall-clock time (seconds; lower is better) instead of chunk counts |
| GRU-Mem (gold-supervised) | Qwen2.5-7B (paper setting) | Evidence top-20%, 112K–896K | 78.91/60.81s (112K), 82.03/111.67s (224K), 80.47/213.04s (448K), 78.12/454.72s (896K) | N/A | GRU-Mem Table 2 (same as above) | Published result; trained with evidence labels + `t_last_evidence` supervision |
| Gold-supervised GRU-Mem (re-impl) | RL-MemoryAgent-7B | RULER QA (top-20%) | **TBD** | **TBD** | - | Optional: re-run if evidence labels can be derived from RULER generator for this setting; otherwise we treat published GRU-Mem Table 2 as an upper-bound reference but not directly comparable in model initialization |
| RL-MemoryAgent-7B (no gates) | RL-MemoryAgent-7B | RULER QA (top-20%) | **TBD** | Full scan (all chunks) | - | Inference-only baseline: shows cost of no early exit and no selective updates |
| IGPO-style gain (unmasked; expected to reward-hack) | RL-MemoryAgent-7B | RULER QA (top-20%) | **TBD** | **TBD** | - | Ablation: apply Δ-based reward but do NOT mask RL loss on `<update>` tokens; tests whether the policy learns degenerate “write answer into memory” behaviors |
| Outcome-only + cost (gate-only) | RL-MemoryAgent-7B | RULER QA (top-20%) | **TBD** | **TBD** | - | Baseline for isolating value of answerability-gain shaping; optimizes only `<check>`/`<next>` tokens with reward = outcome + format − λ·steps − μ·updates |
| **Ours: answerability gain + cost (gate-only)** | RL-MemoryAgent-7B | RULER QA (top-20%) | **TBD** | **TBD** | - | Reward adds Δ_t computed by cached teacher-forced scoring; RL loss masked on `<update>` spans to avoid answer-in-memory reward hacking |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| Outcome-only + cost (baseline) | Remove answerability-gain term \(\Delta_t\) | Worse accuracy–efficiency trade-off than ours if \(\Delta_t\) provides real credit assignment |
| Gold-supervised (upper bound) | Replace \(\Delta_t\) with evidence-label rewards | Best gating quality; quantifies remaining gap |

### Analysis (Optional)

- **Gate quality vs evidence labels**: measure update precision/recall against known evidence chunks and exit error \(t_{exit} - t_{last\_evidence}\) (labels used only for evaluation).
- **Distractor robustness**: measure how often the model updates on answer-string distractors and whether accuracy degrades.

---

## Success Criteria

**Criterion 1: Label-free gating learns non-degenerate behavior**
- Hypothesis: the learned policy does not collapse to “always update + never exit” or “exit immediately”.
- Validation: on RULER QA, avg processed chunks is substantially below full scanning, while accuracy remains well above chance and above the outcome-only+cost baseline.

**Criterion 2: Answerability gain adds value beyond compute penalties**
- Hypothesis: adding \(\Delta_t\) improves the accuracy–efficiency frontier compared to outcome-only+cost.
- Validation: at similar avg processed chunks (matched by tuning \(\lambda,\mu\) on a small dev set), our method yields higher accuracy than outcome-only+cost.

**Criterion 3: Closes much of the gap to gold-supervised gates**
- Hypothesis: evidence-label-free training recovers most of the benefit of gold-supervised GRU-Mem.
- Validation: at the gold-supervised method’s avg processed chunk budget, our accuracy is within a small margin (e.g., ~1–2 points) on the top-20% evidence setting.

---

## Impact Statement

If successful, this work removes a key supervision bottleneck for training efficient long-context recurrent memory systems: developers could train GRU-Mem-style update/exit gating on any QA dataset that has answers, without needing evidence-position labels. This would make recurrent memory agents more practical for deployment in document QA and long-context assistants where scanning cost is high and evidence annotations are not available.

---

## References

- When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning (Sheng et al., 2026) - [meta](./references/When-to-Memorize-and-When-to-Stop-Gated-Recurrent-Memory-for-Long-Context-Reasoning_1/meta/meta_info.txt)
- MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent (Yu et al., 2025) - [meta](./references/MemAgent-Reshaping-Long-Context-LLM-with-Multi-Conv-RL-based-Memory-Agent/meta/meta_info.txt)
- RULER: What’s the Real Context Size of Your Long-Context Language Models? (Hsieh et al., 2024) - https://arxiv.org/abs/2404.06654
- HotpotQA (Yang et al., 2018) - https://arxiv.org/abs/1809.09600
- SQuAD (Rajpurkar et al., 2016) - https://arxiv.org/abs/1606.05250
- DeepSeekMath (GRPO) (Shao et al., 2024) - https://arxiv.org/abs/2402.03300
- Proximal Policy Optimization (Schulman et al., 2017) - https://arxiv.org/abs/1707.06347
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale (Zhu et al., 2025) - https://arxiv.org/abs/2503.14476
- Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) (Liu et al., 2025) - https://arxiv.org/abs/2503.20783
- Information Gain-based Policy Optimization (Wang et al., 2025) - https://arxiv.org/abs/2510.14967
- Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents (Shi et al., 2025) - https://arxiv.org/abs/2509.23040
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020) - https://arxiv.org/abs/2005.11401
- Fusion-in-Decoder: Efficient Open-Domain Question Answering (Izacard & Grave, 2020) - https://arxiv.org/abs/2007.01282
- Improving language models by retrieving from trillions of tokens (RETRO) (Borgeaud et al., 2021) - https://arxiv.org/abs/2112.04426
- Transformer-XL (Dai et al., 2019) - https://arxiv.org/abs/1901.02860
- Longformer (Beltagy et al., 2020) - https://arxiv.org/abs/2004.05150
- Confident Adaptive Language Modeling (Schuster et al., 2022) - https://arxiv.org/abs/2207.07061
- ConsistentEE (Liu et al., 2023) - https://arxiv.org/abs/2312.11882
- BERT Loses Patience (PABEE) (Zhou et al., 2020) - https://arxiv.org/abs/2006.04152
- DeeBERT (Xin et al., 2020) - https://arxiv.org/abs/2004.12993
- FastBERT (Liu et al., 2020) - https://arxiv.org/abs/2004.14202
- BERxiT (Xin et al., 2021) - https://aclanthology.org/2021.eacl-main.8/
- Adaptive Computation Time (Graves, 2016) - https://arxiv.org/abs/1603.08983
- Toolformer (Schick et al., 2023) - https://arxiv.org/abs/2302.04761
- ReAct (Yao et al., 2022) - https://arxiv.org/abs/2210.03629
