# untitled

# Query-OOD Escalation: Selective High-k Consensus for Efficient Memory-Poisoning Defense

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

LLM-based agents increasingly use **retrieval-based memory**: they store past interactions or external documents in a database, retrieve the top-k most similar entries for each query, and include them as in-context evidence. This improves capability but creates a persistent attack surface: an adversary can inject a small number of malicious memory entries that later get retrieved and steer the agent’s behavior.

Recent attacks show this can be practical. **AgentPoison** backdoors retrieval-based agents by inserting a few poisoned “demonstrations” into the memory/knowledge base and optimizing a trigger so that triggered queries retrieve the poison with high probability while preserving benign utility (**[AgentPoison](./references/AgentPoison-Red-teaming-LLM-Agents-via-Poisoning-Memory-or-Knowledge-Bases/meta/meta_info.txt)**). **MINJA** shows a stricter threat model where the attacker only interacts via normal user queries, yet can still induce the agent to store malicious records with high injection and attack success (**[MINJA](./references/A-Practical-Memory-Injection-Attack-against-LLM-Agents/meta/meta_info.txt)**).

A-MemGuard is a state-of-the-art defense for agent memory poisoning. It sits between retrieval and action: for each query, it retrieves multiple memories, generates multiple **structured reasoning paths** (entity–relation trajectories) conditioned on each retrieved item, and filters items whose paths deviate from the benign consensus. It also stores detected failure patterns as “lessons” to prevent self-reinforcing error cycles (**[A-MemGuard](./references/A-MemGuard-A-Proactive-Defense-Framework-for-LLM-based-Agent-Memory/meta/meta_info.txt)**).

### The Problem

The strongest current retrieval-time defenses are **computationally expensive** because they run multiple LLM calls per retrieved item. This creates a deployment dilemma:

- **Using a larger retrieval set (higher top-k) can make consensus validation stronger**, but cost scales roughly linearly with k.
- **Attack traffic is typically rare** in many deployments (most queries are benign), so always running a high-k defense can impose large overhead for little marginal benefit.

A-MemGuard itself exhibits a strong top-k effect in its sensitivity analysis: increasing the number of retrieved memories strengthens consensus and reduces attack success (A-MemGuard Table 6; **top-k=4 → top-k=8** improves ASR-t from **36.17** to **4.25** in their reported setting) (**[A-MemGuard top-k sensitivity](./references/A-MemGuard-A-Proactive-Defense-Framework-for-LLM-based-Agent-Memory/sections/5.7 Hyperparameter Sensitivity Analysis.md)**).

**Important caveat (baseline source):** A-MemGuard’s Table 6 sensitivity numbers are reported for their **EHRAgent** setting (Sec. 5.6), not ReAct-StrategyQA. Therefore, this proposal treats “high-k improves robustness” as a *plausible cross-task trend* and **explicitly re-runs k=4 vs k=8 on ReAct-StrategyQA** as part of verification.

This suggests that “run higher k” is a powerful robustness knob, but it is unclear whether we can get the robustness benefits **without paying the cost on every query**.

This proposal asks a concrete, deployment-facing question:

> Can we use a cheap, training-free **query-embedding OOD (out-of-distribution) score** to selectively escalate A-MemGuard’s retrieval top-k only on suspicious queries, achieving near-high-k robustness at near-low-k average cost?

**Metrics note:** We evaluate security using AgentPoison’s multi-stage *attack success rates* (ASR) at retrieval (ASR-r), action generation (ASR-a), and end-to-end task compromise (ASR-t), where lower is better; and benign utility via task accuracy (ACC), where higher is better.

### Key Insight and Hypothesis

**Key insight.** AgentPoison’s trigger optimization explicitly includes a **uniqueness objective** that pushes triggered query embeddings away from benign query cluster centers while keeping triggered queries compact (AgentPoison §3.3.1; see Fig. 2) (**[AgentPoison](./references/AgentPoison-Red-teaming-LLM-Agents-via-Poisoning-Memory-or-Knowledge-Bases/meta/meta_info.txt)**). This implies a potentially exploitable geometry: a triggered query can be **OOD relative to the benign query distribution**, yet still retrieve poison because poisoned keys are co-located with the triggered query in the same embedding region.

**Hypothesis.** In AgentPoison-style direct poisoning on ReAct-StrategyQA, a simple OOD score computed on the retriever’s query embedding (e.g., cosine distance to benign centroid or Mahalanobis distance) can separate benign queries from triggered queries well enough to:

1) escalate A-MemGuard’s top-k only on a small fraction of benign queries (low false positives), and
2) achieve attack robustness close to always using high-k (low false negatives),

thereby improving the robustness–cost trade-off compared to fixed-k defenses.

This could fail if (i) triggered queries are not reliably OOD (the attack remains in-distribution while being retrievable), (ii) any separability comes from superficial text artifacts rather than embedding geometry, or (iii) an adaptive attacker can reduce detectability by changing AgentPoison’s uniqueness weight with minimal loss in attack success.

---

## Proposed Approach

### Overview

We propose **Query-OOD Escalation (QOE)**: a wrapper around A-MemGuard that adaptively sets the number of retrieved memories used for consensus validation.

- Default mode (benign): run A-MemGuard with low compute, **top-k = k_low**.
- Escalation mode (suspicious): if the query is flagged as OOD, run A-MemGuard with stronger consensus, **top-k = k_high**.

### Method Details

#### Setting and notation

- Retriever embedding function: \(E(\cdot)\) (e.g., **DPR = Dense Passage Retrieval** question encoder).
- Query embedding: \(z_q = E(q)\in\mathbb{R}^d\).
- Benign query embedding distribution (estimated on a dev set): mean \(\mu\) and covariance \(\Sigma\).

#### OOD scores (training-free)

We pre-register evaluating OOD scores in increasing complexity, choosing the simplest score that meets the detection-quality gate:

1) **Centroid cosine distance**: \(s_{\text{cos}}(q)=1-\cos(z_q,\mu)\).
2) **Retrieval confidence margin (optional)**: \(s_{\text{margin}}(q)=\text{sim}(q,k_1)-\text{sim}(q,k_2)\) for top-1 vs top-2 retrieved keys.
3) **Mahalanobis distance**: \(s_{\text{mah}}(q)=\sqrt{(z_q-\mu)^\top \Sigma^{-1}(z_q-\mu)}\) (standard outlier score under a Gaussian fit to benign embeddings).

We choose a threshold \(\tau\) by calibrating on benign dev queries to target an escalation budget (e.g., ≤5% of benign queries escalated).

#### Adaptive top-k policy

Let \(k_{low}=4\) and \(k_{high}=8\) (values chosen to match A-MemGuard defaults and the strongest setting in their sensitivity analysis).

For each query \(q\):

- If \(s(q) \le \tau\): run A-MemGuard with \(k=k_{low}\).
- If \(s(q) > \tau\): run A-MemGuard with \(k=k_{high}\).

No model fine-tuning is required; this is an inference-time policy.

### Key Innovations

- **Selective consensus escalation for agent memory security**: treat retrieval top-k (consensus size) as a security knob and activate high-k only when needed.
- **Mechanism claim tied to attack design**: exploit AgentPoison’s explicit uniqueness objective as a testable geometric hypothesis.
- **Pre-registered detection-quality gate**: if query-OOD separability is weak, the proposal is refuted early without running expensive full-agent evaluations.

---

## Related Work

### Field Overview

**Memory poisoning attacks and defenses.** LLM agents with persistent memory are vulnerable to poisoning via direct writes (attacker injects into the memory store) and indirect writes (attacker tricks the agent into storing malicious content). AgentPoison demonstrates training-free backdoor triggers optimized in embedding space for retrieval hijacking, while MINJA shows a query-only attacker can induce malicious memory records without privileged access. Defenses include retrieval-time validation (A-MemGuard), verify-before-commit pipelines for tool streams (VIGIL), and trust-scoring / sanitization approaches that often face calibration failures.

**Retrieval-stage defenses in RAG.** A separate literature studies corpus poisoning in RAG systems and proposes retriever-side defenses such as partition-and-vote (RAGPart) and similarity-shift token masking (RAGMask). These methods operate on retrieved documents rather than on the query, and do not target agent-memory consensus validators.

**Adaptive retrieval and gating.** Training-free retrieval gating for efficiency (e.g., uncertainty-based retrieval decisions) shows that lightweight signals computed before retrieval can drive large cost reductions, motivating analogous “selective escalation” designs in security settings.

### Related Papers

- **[A-MemGuard](./references/A-MemGuard-A-Proactive-Defense-Framework-for-LLM-based-Agent-Memory/meta/meta_info.txt)**: Consensus-based validation + lesson memory for defending agent memory; provides baseline ASR/ACC and top-k sensitivity.
- **[AgentPoison](./references/AgentPoison-Red-teaming-LLM-Agents-via-Poisoning-Memory-or-Knowledge-Bases/meta/meta_info.txt)**: Training-free backdoor attack that optimizes retrieval triggers via uniqueness+compactness in embedding space.
- **[MINJA / Memory Injection Attacks](./references/A-Practical-Memory-Injection-Attack-against-LLM-Agents/meta/meta_info.txt)**: Query-only memory injection via bridging steps + progressive shortening.
- **[Memory Poisoning Attack and Defense on Memory Based LLM-Agents](https://arxiv.org/abs/2601.05504)**: Shows realistic initial memory states can drastically change ASR; highlights calibration failures of trust scoring.
- **[Agent Security Bench (ASB)](./references/Agent-Security-Bench-(ASB)-Formalizing-and-Benchmarking-Attacks-and-Defenses-in-LLM-based-Agents/meta/meta_info.txt)**: Benchmark spanning multiple agent attack stages including memory poisoning; reports defense limitations.
- **[VIGIL](./references/VIGIL-Defending-LLM-Agents-Against-Tool-Stream-Injection-via-Verify-Before-Commit/meta/meta_info.txt)**: Verify-before-commit defense for tool stream injection; conceptually related to memory-layer checks.
- **[RAGPart & RAGMask](./references/RAGPart-&-RAGMask-Retrieval-Stage-Defenses-Against-Corpus-Poisoning-in-Retrieval-Augmented-Generation/meta/meta_info.txt)**: Retrieval-stage defenses against corpus poisoning in RAG via partitioning/voting and token masking.
- **[TARG](https://arxiv.org/abs/2511.09803)**: Training-free adaptive retrieval gating for efficient RAG using uncertainty signals (efficiency analogue).
- **[Bias Injection Attacks on RAG Databases and Sanitization Defenses](https://arxiv.org/abs/2512.00804)**: Uses embedding-geometry/statistical tests (KL/PCA/Mahalanobis) to detect adversarial passages in RAG.
- **[Detecting language model attacks with perplexity](https://arxiv.org/abs/2308.14132)**: Perplexity-based filtering baseline used in A-MemGuard.
- **[Certifying LLM safety against adversarial prompting](https://arxiv.org/abs/2309.02705)**: Classifier-style baseline used in A-MemGuard.
- **[BadChain](https://arxiv.org/abs/2401.12242)**: Backdoor chain-of-thought prompting; related backdoor family.
- **[PoisonedRAG](https://arxiv.org/abs/2402.07867)**: Systematic study of poisoning attacks on RAG corpora.
- **[Poisoning retrieval corpora by injecting adversarial passages](https://arxiv.org/abs/2310.19156)**: Early corpus poisoning for dense retrievers.
- **[PR-Attack](https://arxiv.org/abs/2504.07717)**: Coordinated prompt+RAG poisoning via bilevel optimization.
- **[PoisonArena](https://arxiv.org/abs/2505.12574)**: Multi-attacker regimes; motivates robustness beyond single-attacker settings.
- **[Secure Retrieval-Augmented Generation against Poisoning Attacks](https://arxiv.org/abs/2510.25025)**: RAG defense-in-depth perspective.
- **[RAGForensics](https://arxiv.org/abs/2504.21668)**: Traceback/identification of poisoned texts for RAG.
- **[RAGuard](https://arxiv.org/abs/2509.20324)**: Detection of poisoned retrieval contexts via similarity/perplexity anomalies.
- **[A Survey on Backdoor Threats in Large Language Models](https://arxiv.org/abs/2309.06055)**: Survey of backdoor attacks/defenses including representation-based anomaly detection.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Memory poisoning attacks | Inject malicious records into agent memory/KB to steer future behavior | AgentPoison; MINJA | ReAct-StrategyQA; EHRAgent; Webshop; MMLU | Success depends on retrieval + memory state |
| Consensus validation defenses | Validate retrieved memories by comparing multiple reasoning paths | A-MemGuard | ASR-r/ASR-a/ASR-t + benign ACC | Compute cost grows with top-k |
| Retrieval-stage corpus defenses | Modify retrieval to reduce poison influence before generation | RAGPart; RAGMask | NQ; FIQA; poisoning suites | Different threat model (corpus poisoning vs agent memory) |
| Adaptive retrieval gating (efficiency) | Decide when/how much to retrieve based on uncertainty | TARG | NQ/TriviaQA/PopQA | Not designed for adversaries |
| Statistical anomaly detection in embedding space | Detect adversarial texts/passages via geometry | BiasDef; RAGuard | RAG poisoning benchmarks | Often targets passages, not query triggers |

### Closest Prior Work

- **A-MemGuard**: Closest defense target; shows top-k is a strong robustness knob, but does not propose *selective* escalation based on query-level signals.
- **AgentPoison**: Closest attacker; explicitly optimizes triggered queries to be unique and compact in embedding space, motivating a query-OOD hypothesis.
- **RAGPart/RAGMask**: Closest “retrieval-stage defenses” family; operates on retrieved documents, not on query-trigger detection or selective consensus escalation.
- **TARG**: Closest “training-free gating” idea, but aims at efficiency for benign QA rather than robustness against adversarial triggers.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| A-MemGuard | Consensus validation over top-k memories | High compute; high-k is costly | Add query-OOD gate to adapt k per query | Achieve high-k robustness at low-k average cost |
| RAGPart/RAGMask | Retriever-side defenses against corpus poisoning | Different surface; no agent-memory consensus stage | Focus on query-trigger detection for consensus defenses | Addresses a cost bottleneck specific to consensus validators |
| TARG | Training-free retrieval gating for efficiency | Not adversarial; no security metrics | Apply gating to *security escalation* not retrieval/no-retrieval | Security-utility-cost trade-off is deployment-critical |
| BiasDef / RAGuard | Embedding-geometry anomaly detection for poisoned passages | Targets passages; may require extra processing | Use query embedding OOD as a trigger detector | Cheaper than passage-level analysis; directly tied to AgentPoison geometry |

---

## Experiments

### Experimental Setup

**Goal:** Test whether query-embedding OOD reliably detects AgentPoison triggers and enables selective high-k A-MemGuard execution with lower average cost.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Llama-3.1 Instruct (or closest available Llama-3-8B instruct checkpoint) | ~8B | https://huggingface.co/meta-llama | Local inference with vLLM for all agent/defense prompts |

**Retriever / embeddings:**

| Component | Choice | Download Link | Notes |
|---|---|---|---|
| Dense retriever | DPR question encoder | https://huggingface.co/facebook/dpr-question_encoder-single-nq-base | Matches AgentPoison/A-MemGuard DPR setting |

**Benchmarks / attack code:**
- A-MemGuard code: https://github.com/TangciuYueng/AMemGuard
- AgentPoison code: https://github.com/BillChan226/AgentPoison
- StrategyQA dataset / evidence: https://allenai.org/data/strategyqa

**Resource Estimate**:
- **Compute budget**: 50–200 GPU-hours (Llama-8B inference dominated by A-MemGuard path generation; depends on evaluation subset size). Detection-gate stage is cheap (embedding computations).
- **GPU memory**: 1×A100 80GB sufficient for Llama-8B inference with vLLM.
- **API usage**: None required.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| ReAct-StrategyQA under AgentPoison | Knowledge-intensive QA agent with retrieval; AgentPoison inserts poisoned demonstrations and trigger queries | ASR-r, ASR-a, ASR-t (lower is safer); ACC on benign (higher is better); escalation rate; AUROC/FPR@TPR for detector | test (+ benign dev for calibration) | https://allenai.org/data/strategyqa | Use AgentPoison + A-MemGuard repos |

### Main Results

**Published baseline context (A-MemGuard Table 1; ReAct-StrategyQA, LLaMA-3-8B + DPR):**

| Method | Base Model | Benchmark | ASR-r ↓ | ASR-a ↓ | ASR-t ↓ | ACC ↑ | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| No Defense | LLaMA-3-8B | ReAct-StrategyQA | 37.50 | 40.74 | 48.14 | N/A | [A-MemGuard Table 1](./references/A-MemGuard-A-Proactive-Defense-Framework-for-LLM-based-Agent-Memory/sections/5.2 Effectiveness at defending against direct injection methods.md) | Published |
| A-MemGuard (top-k=4) | LLaMA-3-8B | ReAct-StrategyQA | 0.00 | 0.00 | 42.85 | N/A | [A-MemGuard Table 1](./references/A-MemGuard-A-Proactive-Defense-Framework-for-LLM-based-Agent-Memory/sections/5.2 Effectiveness at defending against direct injection methods.md) | Published |

To be verified (this proposal; exact values TBD):

| Method | Base Model | Benchmark | ASR-r ↓ | ASR-a ↓ | ASR-t ↓ | ACC ↑ | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| A-MemGuard (top-k=4, re-run) | LLaMA-3-8B | ReAct-StrategyQA | TBD | TBD | TBD | TBD | - | Re-run in open-source stack |
| A-MemGuard (top-k=8) | LLaMA-3-8B | ReAct-StrategyQA | TBD | TBD | TBD | TBD | - | High-k robustness baseline |
| **QOE (ours; OOD→k=8 else k=4)** | LLaMA-3-8B | ReAct-StrategyQA | TBD | TBD | TBD | TBD | - | To be verified |

**Pre-registered detection gate (must pass before running full agent eval):**
- Compute AUROC of the chosen OOD score on benign vs triggered queries.
- If AUROC < **0.80** OR FPR@99%TPR > **30%**, stop and treat the proposal as unsupported.

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| QOE (full) | OOD-gated k selection | Best robustness–cost trade-off |
| QOE with text-level gate (diagnostic) | Replace embedding OOD with text anomaly score | If similar, the “embedding geometry” mechanism claim is weaker |
| **Adaptive attacker (required)** | Re-run the detection gate after re-optimizing triggers with **reduced AgentPoison uniqueness-loss weight** (e.g., 0.5×), keeping other settings fixed | If detectability collapses while ASR stays high, QOE is brittle; if ASR drops, QOE is robust to this adaptation |

### Analysis (Optional)

- Report escalation rate on benign dev/test queries and attacked queries.
- Plot ASR vs escalation budget (sweep τ) to show the Pareto frontier.

---

## Success Criteria

**Criterion 1: Query-trigger separability exists**
- Hypothesis: AgentPoison-triggered queries are separable from benign queries by a simple embedding-space OOD score.
- Validation: AUROC ≥ 0.80 with an operating point achieving ≥99% TPR and ≤30% FPR on held-out data.

**Criterion 2: Selective escalation improves robustness–cost trade-off**
- Hypothesis: QOE matches (or approaches) the robustness of always-high-k while escalating on only a small fraction of benign queries.
- Validation: QOE’s ASR-t is directionally close to the fixed high-k baseline, while benign escalation rate remains low (e.g., ≤5–10%) and benign ACC does not degrade materially vs low-k.

**Criterion 3 (required): Not trivially evaded by weakening uniqueness loss**
- Hypothesis: If an attacker reduces the AgentPoison uniqueness-loss weight (e.g., 0.5×), then either (a) triggers remain detectable by the OOD score (gate still passes), or (b) attack success degrades enough that QOE is unnecessary.
- Validation: Under reduced uniqueness-loss weight, we require **either** (i) the detection gate still passes (AUROC ≥ 0.80 and FPR@99%TPR ≤ 30%), **or** (ii) ASR-t drops substantially relative to default AgentPoison (e.g., ≥20 points absolute). If neither holds (high ASR with low detectability), treat QOE as brittle and reject.

---

## Impact Statement

If QOE works, developers of memory-augmented LLM agents can deploy strong consensus-based memory poisoning defenses with substantially lower average inference overhead by running the expensive high-k validator only on suspicious queries. This would make retrieval-time memory defenses more practical to ship in real agent frameworks where attack traffic is sparse but high-impact.

---

## References

- [A-MemGuard: A Proactive Defense Framework for LLM-based Agent Memory](./references/A-MemGuard-A-Proactive-Defense-Framework-for-LLM-based-Agent-Memory/meta/meta_info.txt) - Wei et al., 2025
- [AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases](./references/AgentPoison-Red-teaming-LLM-Agents-via-Poisoning-Memory-or-Knowledge-Bases/meta/meta_info.txt) - Chen et al., 2024
- [Memory Injection Attacks on LLM Agents via Query-Only Interaction (MINJA)](./references/A-Practical-Memory-Injection-Attack-against-LLM-Agents/meta/meta_info.txt) - Dong et al., 2025
- [Memory Poisoning Attack and Defense on Memory Based LLM-Agents](https://arxiv.org/abs/2601.05504) - Bhatnagar, 2025
- [Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents](./references/Agent-Security-Bench-(ASB)-Formalizing-and-Benchmarking-Attacks-and-Defenses-in-LLM-based-Agents/meta/meta_info.txt) - Zhang et al., 2024
- [VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit](./references/VIGIL-Defending-LLM-Agents-Against-Tool-Stream-Injection-via-Verify-Before-Commit/meta/meta_info.txt) - Lin et al., 2026
- [RAGPart & RAGMask: Retrieval-Stage Defenses Against Corpus Poisoning in Retrieval-Augmented Generation](./references/RAGPart-&-RAGMask-Retrieval-Stage-Defenses-Against-Corpus-Poisoning-in-Retrieval-Augmented-Generation/meta/meta_info.txt) - Pathmanathan et al., 2025
- [TARG: Training-Free Adaptive Retrieval Gating for Efficient RAG](https://arxiv.org/abs/2511.09803) - Wang et al., 2025
- [Bias Injection Attacks on RAG Databases and Sanitization Defenses](https://arxiv.org/abs/2512.00804) - (authors unknown in this context), 2025
- [Detecting language model attacks with perplexity](https://arxiv.org/abs/2308.14132) - Alon & Kamfonas, 2023
- [Certifying LLM safety against adversarial prompting](https://arxiv.org/abs/2309.02705) - Kumar et al., 2023
- [BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models](https://arxiv.org/abs/2401.12242) - Xiang et al., 2024
- [PoisonedRAG](https://arxiv.org/abs/2402.07867) - Zhong et al., 2024
- [Poisoning retrieval corpora by injecting adversarial passages](https://arxiv.org/abs/2310.19156) - Zhong et al., 2023
- [PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation](https://arxiv.org/abs/2504.07717) - (see arXiv), 2025
- [PoisonArena](https://arxiv.org/abs/2505.12574) - (see arXiv), 2025
- [Secure Retrieval-Augmented Generation against Poisoning Attacks](https://arxiv.org/abs/2510.25025) - (see arXiv), 2025
- [RAGForensics: Traceback of Poisoning Attacks to Retrieval-Augmented Generation](https://arxiv.org/abs/2504.21668) - (see arXiv), 2025
- [RAGuard](https://arxiv.org/abs/2509.20324) - (see arXiv), 2025
- [A Survey on Backdoor Threats in Large Language Models](https://arxiv.org/abs/2309.06055) - (see arXiv), 2023
