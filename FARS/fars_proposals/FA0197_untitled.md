# untitled

# Persistent Template-Merge Poisoning in LLM Log Parsers

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences); security / software-engineering workshops also suitable

## Introduction

### Context and Motivation

System and security logs are a core telemetry source for reliability and security monitoring (e.g., anomaly detection, root-cause analysis, incident response). Many downstream analytics pipelines require **structured logs**: each raw log line is mapped to a **log template** (the constant event pattern) plus parameters (the variable fields).

Recent work shows that large language models (LLMs) can parse diverse log formats with fewer hand-crafted rules than classical log parsers. However, directly applying an LLM to every log line is too expensive at production scale, so practical systems use **stateful accelerators** such as template caches and online template revision. For example, **LILAC** uses an LLM with few-shot in-context learning (ICL) plus an **adaptive parsing cache** that stores templates in a tree; when a log line misses the cache, the LLM generates a new template and the cache may be *updated by merging templates*.

These design choices implicitly treat log text as benign data. In practice, logs are often attacker-influenced (e.g., compromised hosts/services, untrusted tenant workloads, or adversarial inputs that get logged). Industry reports already document “log prompt poisoning” where adversarial log content misleads LLM-based security analysis. But the security implications of **stateful log-parsing caches** have not been systematically studied.

### The Problem

Stateful LLM log parsers maintain persistent internal state that influences future parsing decisions.

- **[LILAC: Log Parsing using LLMs with Adaptive Parsing Cache](./references/LILAC-Log-Parsing-using-LLMs-with-Adaptive-Parsing-Cache/meta/meta_info.txt)**: When cache matching fails, LILAC compares a newly generated template \(T_a\) to “relevant” cached templates \(\{T_i\}\). If a **longest common subsequence (LCS)**-based similarity exceeds a threshold (e.g., \(\tau=0.8\) in their implementation), it *merges* by replacing differing tokens with wildcards `<*>` (see LILAC cache updating).
- **[Log Parsing with Self-Generated In-Context Learning and Self-Correction (AdaParser)](./references/Log-Parsing-with-Self-Generated-In-Context-Learning-and-Self-Correction/meta/meta_info.txt)**: Maintains a trie template cache and a self-generated demonstration pool used for future ICL, which resembles an online memory that could be polluted.

The key concern is that these update rules can permanently broaden templates (more wildcards, fewer discriminative constants). Broad templates can then match multiple event types, causing persistent mis-parsing even after the attacker stops injecting logs.

This is not obviously solved by stronger prompting or inference-time scaling because the vulnerability is driven by **state update logic** (cache merge decisions), not just a single LLM response.

### Key Insight and Hypothesis

**Hypothesis**: In LILAC-style parsers, a small fraction of attacker-injected “near-collision” log lines early in the stream can trigger high-similarity template merges that replace constant tokens with wildcards, causing **persistent degradation** in parsing accuracy on later *clean* logs.

**Why this might be wrong**: (i) the dataset may lack near-collision templates so merges rarely occur, (ii) LLM outputs may be too dissimilar to exceed the merge threshold, or (iii) later clean logs might “repair” the cache via subsequent updates.

---

## Proposed Approach

### Overview

We propose an **attack + defense evaluation** for stateful LLM log parsers:

1. **Crafted template-merge poisoning attack**: automatically generate plausible log lines that are near-collisions of common early templates, aiming to force LILAC’s cache update to merge templates and increase wildcarding.
2. **Collision-aware merge defense**: before committing a merge, test whether the merged template would match multiple existing cached templates (i.e., increases ambiguity). If so, reject the merge and insert the new template separately.

The output is a minimal, fully automated security evaluation protocol that produces (a) a measurable accuracy drop on a clean suffix after poisoning, and (b) a small mitigation that restores accuracy.

### Method Details

#### Target mechanism: LILAC cache updating

LILAC’s cache update uses an LCS-based similarity:
\[
\mathrm{Sim}(T_1,T_2)=\frac{2\,|\mathrm{LCS}(L_1,L_2)|}{|L_1|+|L_2|}
\]
where \(L_1,L_2\) are tokenized templates. If \(\mathrm{Sim}(T_a,T_b)\ge \tau\) (\(\tau\approx 0.8\) in LILAC), LILAC merges by replacing differing tokens along the template path with `<*>`.

#### Attack: automated near-collision injection (gray-box)

Operational attacker model: attacker can inject additional log lines into a service’s log stream and has **gray-box** knowledge of the parser’s general behavior: they know (or can estimate) the merge threshold \(\tau\) and can observe a small prefix of benign logs from the target system (e.g., by being a tenant generating requests that get logged). They do *not* need direct access to the cache internals.

Attack generator (fully automatic given a stream prefix):

1. Take the first \(W\)% of the log stream (default \(W=10\%\)) as the attacker’s observation window.
2. Select \(m\) target log lines from distinct high-frequency templates in the window (default \(m=5\)).
3. For each target line, create \(k\) variants by replacing **one alphabetic token** with another token sampled from the observation window vocabulary (default \(k=4\)).
4. Inject all crafted lines into the first \(P\)% of the stream (default \(P=20\%\)).

With \(m=5\), \(k=4\), the poison budget is 20 injected lines. On a 2,000-line dataset, this is a 1% injection rate.

Persistence evaluation: compute accuracy on a late clean suffix (last 30% of the original logs) after a long intervening clean period (middle window) with no attacker injections.

#### Defense: collision-aware merge screening

When LILAC proposes merging \(T_a\) into \(T_b\):

1. Compute the merged template \(T_{merge}\) (the wildcarded version).
2. Estimate ambiguity: count how many existing leaf templates in the cache would match \(T_{merge}\) under LILAC’s cache matcher.
3. If the match count exceeds 1 (or increases beyond a small tolerance compared to \(T_b\)), reject the merge and insert \(T_a\) as a new template instead.

This defense targets merges that reduce discriminative power and create collisions.

### Key Innovations

- **Security framing for stateful log parsers**: evaluates a persistent, cache-level poisoning surface distinct from runtime prompt injection (e.g., Matryoshka avoids runtime LLM calls).
- **Mechanism-driven attack**: targets a specific state update rule (LCS similarity + wildcard merge) rather than generic “prompt injection”.
- **Minimal mitigation**: a collision-aware merge screening rule implementable as a small change in cache update logic.

---

## Related Work

### Field Overview

Log parsing has a long history with syntax-based parsers (e.g., Drain) and semantic/neural parsers. LLM-based log parsing emerged in late 2023–2024, with many methods relying on in-context learning, retrieval, caching, and template revision to reach competitive accuracy at lower LLM call counts. **SoK: System Log Parsing with Large Language Models** provides the first systematic taxonomy and benchmark of multiple open-source LLM parsing frameworks.

Security work relevant to this proposal spans: (i) prompt injection and untrusted input in LLM-based security automation, (ii) poisoning attacks on in-context learning, and (iii) integrity/privacy risks in other LLM caches (e.g., prompt caching and semantic response caching). However, none of these lines directly evaluate poisoning of **template caches with merge rules** in log parsing frameworks.

### Related Papers

(≥20 papers; one sentence each)

- **[SoK: System Log Parsing with Large Language Models: A Review](./references/SoK-LLM-based-Log-Parsing/meta/meta_info.txt)**: Surveys 29 LLM log parsing methods and benchmarks 7 frameworks, noting that caches can store incorrect templates but not studying adversarial poisoning.
- **[LILAC](./references/LILAC-Log-Parsing-using-LLMs-with-Adaptive-Parsing-Cache/meta/meta_info.txt)**: Introduces an adaptive template cache with LCS-based merging that is the core mechanism we evaluate under attack.
- **[AdaParser](./references/Log-Parsing-with-Self-Generated-In-Context-Learning-and-Self-Correction/meta/meta_info.txt)**: Uses a trie cache plus a self-generated demonstration pool and self-correction, suggesting additional stateful attack surfaces beyond merges.
- **[LogBatcher (Demonstration-Free)](https://doi.org/10.1145/3691620.3694994)**: Parses logs in batches with LLMs and heuristic template corrections; strong accuracy in SoK and a useful comparison point for robustness.
- **[OpenLogParser / LibreLog](https://doi.org/10.48550/arXiv.2408.01585)**: Uses open-source LLMs with ICL + caching, highlighting that stateful designs are common beyond proprietary APIs.
- **[DivLog](https://doi.org/10.1145/3597503.3639155)**: Uses prompt-enhanced ICL with diverse demonstration selection, similar to LILAC’s reliance on representative examples.
- **[SelfLog](https://ieeexplore.ieee.org/document/10771304/)**: Proposes a self-evolutionary log parsing pipeline retrieving historical logs/templates, another form of persistent state.
- **[LUNAR](https://doi.org/10.48550/arXiv.2406.07174)**: Combines unsupervised parsing with retrieval and caching, illustrating alternative stateful components.
- **[ECLIPSE](https://doi.org/10.48550/arXiv.2405.13548)**: Uses semantic entropy + LCS for cross-lingual industrial log parsing, showing LCS-like similarity rules are widespread.
- **[LogParser-LLM](https://doi.org/10.1145/3637528.3671810)**: KDD’24 framework for efficient LLM-based parsing, emphasizing practicality via caching.
- **[LogPrompt](https://ieeexplore.ieee.org/document/10556497/)**: Prompt-strategy-based online log analysis; SoK reports strict output markers can fail on reasoning models.
- **[Log Parsing: How Far Can ChatGPT Go?](https://doi.org/10.1109/ASE56229.2023.00206)**: Early evidence that LLMs can parse logs, motivating later cache-based systems.
- **[Log Parsing with Prompt-Based Few-Shot Learning (LogPPT)](https://doi.org/10.1109/ICSE48619.2023.00204)**: Few-shot prompting for log parsing, establishing a baseline prompt formulation family.
- **[LLMParser](https://doi.org/10.1145/3597503.3639150)**: Explores fine-tuning LLMs for parsing, representing an alternative to adding stateful wrapper logic.
- **[SuperLog](https://doi.org/10.48550/arXiv.2412.01377)**: Adapts LLMs to log analysis via domain knowledge and training, also avoiding some wrapper-state mechanisms.
- **[Lemur](https://doi.org/10.48550/arXiv.2402.18205)**: Uses entropy sampling and chain-of-thought merging, another template revision/merging design.
- **[HELP](https://doi.org/10.48550/arXiv.2408.08300)**: Uses hierarchical embeddings plus LCS-related ideas; relevant to similarity-driven grouping.
- **[LogHub](https://doi.org/10.1109/ISSRE59848.2023.00071)**: Standard log dataset collection used in most evaluations.
- **[Corrected LogHub guidelines](https://doi.org/10.1145/3510003.3510101)**: Provides corrected template labels and evaluation guidance used by SoK.
- **[LogHub-2.0 evaluation](https://doi.org/10.1145/3650212.3652123)**: Large-scale benchmark of log parsing techniques, motivating efficiency and caching.
- **[Drain](https://doi.org/10.1109/ICWS.2017.13)**: Widely used deterministic baseline parser for log templates.
- **[SPELL](https://doi.org/10.1109/ICDM.2016.0103)**: Streaming parsing using LCS-like matching, connecting to merge/similarity mechanisms.
- **[Brain](https://doi.org/10.1109/TSC.2023.3270566)**: Bidirectional parsing tree baseline used in SoK.
- **[ULP](https://doi.org/10.1109/ICSME55016.2022.00009)**: Efficient parser baseline used in SoK.
- **[AEL](https://doi.org/10.1109/QSIC.2008.50)**: Early baseline approach for abstracting execution logs.
- **[Evaluation study on log parsing](https://doi.org/10.1109/DSN.2016.66)**: Discusses evaluation pitfalls and preprocessing choices in log parsing.

Security / poisoning context:

- **[Semantic-Aware Parsing for Security Logs (Matryoshka)](./references/Semantic-Aware-Parsing-for-Security-Logs/meta/meta_info.txt)**: Uses LLMs to generate static parsers offline to avoid runtime prompt injection, providing a contrasting safer architecture.
- **[SecureCAI](./references/SecureCAI-Injection-Resilient-LLM-Assistants-for-Cybersecurity-Operations/meta/meta_info.txt)**: Studies prompt injection in SOC assistants and treats logs as an untrusted input channel.
- **[LPCI](./references/Logic-layer-Prompt-Control-Injection-LPCI-A-Novel-Security-Vulnerability-Class-in-Agentic-Systems/meta/meta_info.txt)**: Defines delayed/conditional payloads embedded in memory/tool outputs, closely analogous to persistent cache poisoning.
- **[ICLPoison](./references/Data-Poisoning-for-In-context-Learning/meta/meta_info.txt)**: Demonstrates poisoning attacks on in-context examples, motivating risks in self-generated demonstration pools.
- **[ICLAttack](./references/Universal-Vulnerabilities-in-Large-Language-Models-Backdoor-Attacks-for-In-context-Learning/meta/meta_info.txt)**: Shows universal backdoors for ICL without fine-tuning, relevant to state-based attacks.
- **[Auditing Prompt Caching in Language Model APIs](./references/Auditing-Prompt-Caching-in-Language-Model-APIs/meta/meta_info.txt)**: Shows that caching can introduce new security risks (timing/privacy side channels) in real deployments.
- **[VectorQ / vCache](./references/Adaptive-Semantic-Prompt-Caching-with-VectorQ/meta/meta_info.txt)**: Improves semantic caching correctness via adaptive thresholds, relevant as a “cache correctness” prior in another domain.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Classical syntax-based parsers | Fixed rules/trees over tokens | Drain, Spell, AEL, Brain, ULP | LogHub, LogHub-2.0 | Require log formats/config; brittle under drift |
| LLM direct parsing (no state) | One LLM call per log/batch | ChatGPT parsing, LogPrompt | LogHub | Too expensive; output instability |
| LLM + caching + revision | Persist templates; reduce calls; revise/merge templates | LILAC, LogBatcher, AdaParser, SelfLog | corrected LogHub, LogHub-2.0 | Stateful attack surface; benign assumptions |
| Offline generation (no runtime LLM) | LLM generates static parser offline | Matryoshka | SecurityLogs | Generation-time risks; less flexible at runtime |

### Closest Prior Work

1. **LILAC**: Defines the exact cache update mechanism we target (LCS similarity + wildcard merge). Our work adds an adversarial threat model, a poisoning attack, merge diagnostics, and a merge-screening defense.
2. **AdaParser**: Also uses a trie cache and a stateful self-generated demo pool, but focuses on accuracy improvements via self-correction rather than security/integrity.
3. **Matryoshka**: Explicitly motivates prompt injection risk and avoids runtime LLM calls; our work studies stateful runtime parsers that do use LLM calls and caches.

**Novelty Kill Search Summary:** Searched for “log parsing template cache poisoning”, “LILAC cache poisoning”, “adaptive parsing cache poisoning”, and “log parser prompt injection cache”, and checked recent surveys/benchmarks (SoK 2025; LogEval 2024). No prior work explicitly analyzing poisoning of LCS-merge template caches in LLM log parsers was found as of 2026-02-20.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LILAC | LLM+ICL log parsing with adaptive cache + merge | No adversarial threat model | Add poisoning attack + merge diagnostics + merge screening | Targets a concrete failure mode induced by the merge rule |
| AdaParser | Trie cache + SG-ICL + self-correction | No adversarial evaluation | Focus on cache integrity under adversarial logs | Persistent state is an attack surface; needs hardening |
| Matryoshka | Offline generation, runtime regex parsing | Different architecture; not cache poisoning | Contrast: shows a safer design point | Clarifies trade-off: runtime flexibility vs security |
| ICL poisoning works | Poison ICL demos/backdoors | Not log parsing; not template merges | Map poisoning concepts to log parsing caches | Prior that small poison budgets can have persistent effects |

---

## Experiments

### Experimental Setup

**Execution harness:** Use the SoK benchmark repository (integrates multiple parsers + metrics) to reduce engineering risk.
- Repo: https://github.com/ait-aecid/LLM-log-parsing

**Primary dataset (single-dataset verification):** corrected LogHub **BGL** dataset (2,000 logs).
- Download: https://github.com/logpai/loghub

**Primary metric:** FTA on a clean suffix (last 30% of original logs) after poisoning stops.
- Also report PA for interpretability.

**Target parser:** LILAC (supervised few-shot). Use \(n=4\) labeled templates (“LILAC-4”) to match SoK’s evaluation setting.

**Baseline Ladder (REQUIRED):**
- **Level 1 (prompting baseline, small subset)**: Stateless LLM parsing of individual logs with a fixed prompt (no cache), evaluated on 200 logs to bound feasibility.
- **Level 4 (inference scaling, small subset)**: Self-consistency for the stateless baseline (e.g., sample 5 outputs, select the one passing a template-matching verifier).
- **Level 5 (closest method)**: LILAC-4 (our target) and LogBatcher (strong stateful baseline in SoK).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| gpt-4o-mini | API | (available via platform) | Primary for LILAC calls; set temperature=0 |
| DeepSeek-V3.2 | API | (available via platform) | Optional replication with a non-OpenAI frontier model |
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Optional local model if API cost is a concern |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| N/A | Inference-only | - | - | - |

**Phase-0 gates (required; fast checks that must pass before full runs):**

- **G0a (dataset precondition)**: compute pairwise LCS similarity among ground-truth templates in candidate datasets; proceed only if high-similarity pairs exist (e.g., 95th percentile similarity \(>\tau-0.05\) for \(\tau=0.8\)).
- **G0b (attack-chain sanity + crafting specificity)**: warmup on first 200 logs; inject 10 crafted poison lines and 10 random token-substitution lines; require that crafted injection produces (i) ≥1 merge event and (ii) larger increase in wildcarding than random.
- **G0c (small-scale degradation check)**: evaluate FTA/PA on a small held-out clean set (e.g., 100 logs) before vs after warmup poisoning; proceed only if degradation is detectable.

**Main experiment conditions (3 conditions):**

- **C0 Clean**: run LILAC-4 on the original stream.
- **C1 Crafted poison**: inject crafted poison lines into the first 20% of the stream; run LILAC-4; evaluate on the clean suffix (poison removed from evaluation).
- **C2 Crafted poison + defense**: same as C1, but with collision-aware merge screening enabled.

**Seeds / runs:** 3 runs per condition. SoK notes that even with LLM temperature=0, outputs can be non-deterministic, so they repeat evaluations three times and report averages ([SoK Sec. 7.2.2 Randomness](./references/SoK-LLM-based-Log-Parsing/sections/7.2.2.%20Randomness.md)). Report mean ± std.

**Resource Estimate**:
- **Compute budget**: inference-only; dominated by LLM calls. If using API models: expect \(\mathcal{O}(10^2\text{–}10^3)\) calls for a 2k-log dataset depending on caching; if using local 7B model, expected < 20 GPU-hours total.
- **GPU memory**: none (API) or ≤1×A100-80GB (local 7B).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| corrected LogHub (BGL) | 2,000 system logs with corrected templates | FTA (primary), PA (secondary) | prefix poison / suffix eval | https://github.com/logpai/loghub | SoK eval code (LLM-log-parsing repo) |

- **PA (Parsing Accuracy)**: fraction of logs whose predicted template exactly matches ground truth.
- **FTA (F1-score of Template Accuracy)**: F1 over templates, less sensitive to class imbalance.

### Main Results

We will report a single results table for the primary dataset (BGL).

| Method | Base Model | Benchmark | FTA (mean±std) | PA (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Drain | N/A | corrected LogHub (BGL) | 0.22 *(1 run)* | 0.34 *(1 run)* | SoK Table 5 (Sec. 6.1.2) | Deterministic baseline; SoK reports per-dataset scores (no std) |
| LogBatcher | gpt-3.5-turbo-0125 | corrected LogHub (BGL) | 0.83 *(1 run)* | 0.94 *(1 run)* | SoK Table 5 (Sec. 6.1.2) | Strong LLM log-parsing baseline; base model differs from our main runs (comparability note) |
| LILAC-4 (clean) | gpt-3.5-turbo-0125 | corrected LogHub (BGL) | 0.89 *(1 run)* | 0.96 *(1 run)* | SoK Table 5 (Sec. 6.1.2) | Reference point for expected clean performance; base model differs from our main runs (comparability note) |
| **Ours: LILAC-4 (C0 Clean)** | gpt-4o-mini | corrected LogHub (BGL) | **TBD** | **TBD** | - | To be verified (3 seeds) |
| **Ours: LILAC-4 (C1 Poison)** | gpt-4o-mini | corrected LogHub (BGL) | **TBD** | **TBD** | - | Evaluate on clean suffix only |
| **Ours: LILAC-4 (C2 Poison+Defense)** | gpt-4o-mini | corrected LogHub (BGL) | **TBD** | **TBD** | - | Collision-aware merge screening |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Random poison (gate; not main) | Replace token with random string not drawn from window vocabulary | Should cause fewer merges and smaller degradation than crafted poison |
| No-merge (optional) | Disable merge updates entirely | If poisoning works via merges, degradation should shrink (but clean accuracy may drop) |

### Experimental Rigor

- **Variance & Reproducibility**: run 3 seeds (random sampling of labeled templates for LILAC-4) and report mean±std; set LLM temperature=0.
- **Confounders + controls**:
  1. *Any noise hurts parsing* → random-poison gate checks that crafted \> random.
  2. *Effect is not merge-driven* → log merge events and wildcard growth; optional no-merge ablation.
  3. *Evaluation protocol mismatch* → use SoK’s evaluation scripts and document split construction (poison prefix, clean suffix).
- **Sanity checks**: verify that clean C0 reproduces reasonable accuracy (within a few points of SoK under a similar model class); if not, treat results as inconclusive.
- **Data leakage**: pretrained LLMs may have seen LogHub; we focus on *within-model relative degradation* (C1 vs C0) to reduce contamination risk.

---

## Success Criteria

**Hypothesis** (directional): Crafted early poison lines will increase merge-driven wildcarding and reduce FTA/PA on a late clean suffix; the collision-aware merge defense will prevent most of the degradation.

**Decision Rule** (concrete):
- **Proceed**: C1 reduces suffix FTA by ≥5 percentage points vs C0 (outside std over 3 runs) and C2 recovers ≥80% of that drop while keeping C0-vs-C2 clean performance within 1 percentage point.
- **Pivot**: If G0a fails on BGL, switch to Thunderbird (or another high-template-diversity dataset) while keeping poison rate ≤1%.
- **Refute**: If G0b–G0c pass but C1 effect is within noise and diagnostics show minimal merge/wildcard change, conclude LILAC’s merge rule is not practically poisonable under this attacker model.

---

## Impact Statement

If confirmed, this work provides a concrete, automated security test for stateful LLM log parsers and a small mitigation that could be adopted by practitioners deploying LLM-based parsing in security and reliability pipelines. A negative result would also be decision-changing, suggesting that LILAC-style caches are more robust than expected under realistic injection budgets.

---

## References

- [SoK: System Log Parsing with Large Language Models: A Review](./references/SoK-LLM-based-Log-Parsing/meta/meta_info.txt) - Beck et al., 2025
- [LILAC: Log Parsing using LLMs with Adaptive Parsing Cache](./references/LILAC-Log-Parsing-using-LLMs-with-Adaptive-Parsing-Cache/meta/meta_info.txt) - Jiang et al., 2024
- [Log Parsing with Self-Generated In-Context Learning and Self-Correction (AdaParser)](./references/Log-Parsing-with-Self-Generated-In-Context-Learning-and-Self-Correction/meta/meta_info.txt) - Wu et al., 2024
- [Semantic-Aware Parsing for Security Logs (Matryoshka)](./references/Semantic-Aware-Parsing-for-Security-Logs/meta/meta_info.txt) - Piet et al., 2025
- [SecureCAI](./references/SecureCAI-Injection-Resilient-LLM-Assistants-for-Cybersecurity-Operations/meta/meta_info.txt) - 2026
- [LPCI](./references/Logic-layer-Prompt-Control-Injection-LPCI-A-Novel-Security-Vulnerability-Class-in-Agentic-Systems/meta/meta_info.txt) - 2025
- [ICLPoison](./references/Data-Poisoning-for-In-context-Learning/meta/meta_info.txt) - 2024
- [ICLAttack](./references/Universal-Vulnerabilities-in-Large-Language-Models-Backdoor-Attacks-for-In-context-Learning/meta/meta_info.txt) - 2024
- [Auditing Prompt Caching in Language Model APIs](./references/Auditing-Prompt-Caching-in-Language-Model-APIs/meta/meta_info.txt) - Gu et al., 2025
- [VectorQ / vCache](./references/Adaptive-Semantic-Prompt-Caching-with-VectorQ/meta/meta_info.txt) - Schroeder et al., 2025
- Drain (He et al., 2017) - https://doi.org/10.1109/ICWS.2017.13
- Spell (Du & Li, 2016) - https://doi.org/10.1109/ICDM.2016.0103
- Corrected LogHub guidelines (Khan et al., 2022) - https://doi.org/10.1145/3510003.3510101
- LogHub dataset (Zhu et al., 2023) - https://doi.org/10.1109/ISSRE59848.2023.00071
- LogHub-2.0 evaluation (Jiang et al., 2024) - https://doi.org/10.1145/3650212.3652123
