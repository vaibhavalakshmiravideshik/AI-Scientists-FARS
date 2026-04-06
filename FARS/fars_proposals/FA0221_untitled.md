# untitled

# Persistent Demo-Pool Poisoning in Online LLM Log Parsers with Auto-Generated In-Context Examples

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Modern software systems generate large volumes of logs that are used for reliability monitoring and security operations (e.g., anomaly detection, incident triage, and root-cause analysis). Many downstream pipelines require **structured logs**, where each raw log line is mapped to a **template** (the constant event pattern) and parameters (variable fields). Because manual parsing rules do not scale and log formats evolve with frequent software changes, recent work has explored using large language models (LLMs) to parse logs.

A practical LLM-based parser cannot call an LLM for every log line at production scale, so most proposed systems introduce **persistent state** to reduce LLM calls and stabilize outputs. Examples include template caches (e.g., trees that match frequent templates) and retrieval-based in-context learning (ICL) that selects a small set of demonstrations for each query. These design choices make log parsing efficient enough for real deployments, but they also create a security-relevant question: **what happens when the input log stream is attacker-influenced?** Logs often contain attacker-controlled substrings even without host compromise (e.g., request paths, HTTP headers, usernames, error strings, or untrusted tenant identifiers), and these substrings can flow into LLM prompts.

### The Problem

This proposal studies a specific persistent-state mechanism that is increasingly used in LLM log parsers: **online, model-generated demonstration pools** for ICL.

- **[Log Parsing with Self-Generated In-Context Learning and Self-Correction (AdaParser)](./references/Log-Parsing-with-Self-Generated-In-Context-Learning-and-Self-Correction/meta/meta_info.txt)** maintains a **candidate set** of (log message, template) pairs that is used as an ICL demonstration pool. For each new log message, it retrieves the top-*k* most similar prior logs as demonstrations using **LCS similarity** (Longest Common Subsequence over tokenized logs), queries an LLM, and then **adds the new (log, generated template) pair into the candidate set** for future queries (Sec. 3.3). AdaParser calls this online update loop **Self-Generated In-Context Learning (SG-ICL)**. AdaParser also includes a template corrector that enforces (i) template↔log regex match and (ii) a limited heuristic to prevent over-wildcarding after key tokens such as “Exception”, “failed”, and “interrupted” (Sec. 3.4).

The candidate set update rule creates an **online pseudo-labeling loop**: once a (log, template) pair enters the pool, it can influence future LLM outputs through ICL prompts. Unlike standard retrieval-augmented generation (RAG) poisoning, the attacker does not need write access to an external knowledge base; they only need to inject a small number of log lines into the stream such that the system stores and later retrieves harmful demonstrations.

The core open question is whether AdaParser’s internal safeguards (regex verification + limited over-wildcarding correction) are sufficient to prevent persistent corruption of the demonstration pool under realistic, low-budget injection.

### Key Insight and Hypothesis

**Hypothesis:** Even with AdaParser’s template corrector, a small attacker-controlled prefix (e.g., 20 injected log lines in a 2k-log stream) can cause **persistent degradation** in parsing quality on later clean logs by polluting the SG-ICL candidate set. Concretely, poisoned demonstrations can become **high-similarity candidates** under LCS similarity (because they share generic boilerplate tokens with many logs), and their overly generic template labels can cause the LLM to output overly generic templates for many later queries. Because AdaParser’s candidate set grows monotonically and lacks a forgetting or revalidation mechanism, these effects may not self-correct even after long clean periods.

**Why this might be wrong:** (i) the template corrector may prevent the poisoned templates from being overly generic, (ii) poisoned logs may not be retrieved often enough to matter once the candidate set grows with clean examples, or (iii) LLM behavior may be robust to a small fraction of misleading demonstrations (or best-of-*N* sampling may eliminate the effect).

---

## Proposed Approach

### Overview

We propose a minimal, fully automated robustness evaluation for **demo-pool poisoning** in online ICL-based log parsers. The contribution is an **attack-only** experiment with strong controls that isolates whether persistence is real:

- **C0 Clean stream**: AdaParser run on a clean stream.
- **C1 Targeted poison**: inject a small number of crafted log lines early, then evaluate on a late clean window.
- **C2 Random-noise control**: inject the same number of non-targeted (non-optimized) log lines early, constructed to still pass AdaParser’s internal checks and enter the candidate set.

The key outcome is whether C1 causes a much larger late-window accuracy drop than C2, along with mechanism diagnostics (retrieval contamination rate and template wildcarding).

### Method Details

#### Target system: AdaParser’s online demonstration pool

AdaParser’s SG-ICL loop (Sec. 3.3) does the following at each step *t*:
1. Retrieve top-*k* demonstrations by LCS similarity between the query log and candidate logs.
2. Build an ICL prompt with *k* demonstrations (default *k*=3; Sec. 4.5).
3. Query an LLM to generate a template.
4. Apply a template corrector (Sec. 3.4) and then **insert (log, corrected template)** into the candidate set.

Because the candidate set is used as an ICL memory, inserting a small number of “high-similarity, misleading-label” examples can create long-lived corruption.

#### Attack construction (C1: targeted poison)

Attacker capability (realistic for production): the attacker can cause some requests/events to be logged (e.g., through a public-facing service) but cannot read or modify the parser’s internal state. The attacker can observe a short prefix of benign logs (e.g., from their own interactions).

Given the first **200** benign logs (attacker observation window), we craft **N=20** injected logs by a simple, reproducible algorithm designed to (i) maximize similarity to many future logs while (ii) producing templates that are internally consistent and likely to pass AdaParser’s corrector:

1. Tokenize each log by whitespace and punctuation (matching AdaParser’s LCS tokenization).
2. Compute token document frequency on the 200-log window; define “rare” tokens as those in the bottom 20% by frequency (high IDF).
3. Select the **5 most frequent** raw log messages in the window (exact string frequency).
4. For each selected log message, create 4 variants by:
   - replacing one rare alphabetic token with a high-frequency alphabetic token from the window, and
   - replacing one additional alphabetic token with a high-entropy identifier-like string (e.g., `id=8F3A1C2D`) to encourage the LLM to treat more tokens as variables.
   - avoid inserting the key tokens used by AdaParser’s over-wildcarding correction (“Exception”, “failed”, “interrupted”) so that broad templates are less likely to be corrected (Sec. 3.4.2).

These injected logs are placed uniformly within the first 200 logs. They are excluded from evaluation (we evaluate only on clean logs).

#### Control construction (C2: random-noise injection that still enters the pool)

To rule out the trivial explanation “any extra logs hurt,” we construct a control injection set with the same size and placement budget but without targeting high-similarity patterns:

- Sample 20 logs uniformly at random from the first 200 logs.
- Apply the same two-token perturbation operator (one rare-token replacement + one high-entropy identifier insertion), using random replacement targets rather than high-frequency tokens.
- Apply the same constraint of avoiding AdaParser’s key-token list.

This control is designed so that the injected logs still look format-consistent, pass AdaParser’s template↔log matching correction, and are inserted into the candidate set, but should not become high-frequency retrieval items.

#### Mechanism instrumentation (fully automated)

In addition to standard parsing metrics, we log:

- **Retrieval contamination rate**: fraction of evaluation queries whose top-*k* demonstrations include ≥1 injected demo (poisoned or random-noise).
- **Template wildcard fraction**: fraction of tokens in the predicted template that are wildcards (e.g., `<*>`), tracked over time.

### Key Innovations

- **Attack surface specific to online demo pools**: differs from classic prompt injection and corpus poisoning because the system **writes its own retrieval corpus** from its outputs and the attacker only controls the input stream.
- **Persistence-focused evaluation**: measures whether errors persist after long clean periods (late-window evaluation) rather than only immediate impact.
- **Mechanism-linked diagnostics**: ties accuracy drops to retrieval contamination and template broadening, reducing the risk of ambiguous negative/positive outcomes.

---

## Related Work

### Field Overview

LLM-based log parsing emerged in late 2023–2024 and commonly combines in-context learning, retrieval, caching, and template revision to achieve acceptable accuracy and cost. The **SoK on LLM-based log parsing** emphasizes that persistent state (caching, retrieval, revision) is a dominant design pattern and also documents reproducibility and evaluation inconsistencies across implementations.

Separately, security research has shown that LLM systems can be influenced by attacker-controlled text via prompt injection and via poisoning of retrieved context. However, most prior work assumes an attacker can directly write into a retrieval corpus (e.g., a RAG document store). Online log parsers with model-generated demo pools create a different channel: the attacker can induce the system to store harmful demonstrations by sending ordinary requests that get logged.

### Related Papers

- **[Log Parsing with Self-Generated In-Context Learning and Self-Correction (AdaParser)](./references/Log-Parsing-with-Self-Generated-In-Context-Learning-and-Self-Correction/meta/meta_info.txt)**: Online SG-ICL demo pool + self-correction; primary target system.
- **[LILAC: Log Parsing using LLMs with Adaptive Parsing Cache](./references/LILAC-Log-Parsing-using-LLMs-with-Adaptive-Parsing-Cache/meta/meta_info.txt)**: Introduces ICL + caching for log parsing; illustrates how persistent state is used for efficiency.
- **[SoK: System Log Parsing with Large Language Models: A Review](./references/SoK-LLM-based-Log-Parsing/meta/meta_info.txt)**: Taxonomy and benchmark of LLM log parsers; highlights that caching/revision are common and under-evaluated under adversarial inputs.
- **[MicLog: Towards Accurate and Efficient LLM-based Log Parsing via Progressive Meta In-Context Learning](https://arxiv.org/abs/2601.07005)**: Recent log parser using progressive meta-ICL + multi-level caches; motivates that persistent-state log parsers remain an active line.
- **[LogEval: A Comprehensive Benchmark Suite for Large Language Models in Log Analysis](https://arxiv.org/abs/2407.01896)**: Benchmark suite for LLM log analysis tasks including log parsing; provides broader evaluation context.
- **[Demonstration-Free: Towards More Practical Log Parsing with Large Language Models (LogBatcher)](https://doi.org/10.1145/3691620.3694994)**: Batch-oriented log parsing that reduces reliance on demonstrations.
- **[DivLog: Log Parsing with Prompt Enhanced In-Context Learning](https://doi.org/10.1145/3597503.3639155)**: Improves log parsing via demonstration selection and prompt design.
- **[LibreLog: Accurate and Efficient Unsupervised Log Parsing Using Open-Source Large Language Models](https://arxiv.org/abs/2408.01585)**: Unsupervised LLM log parsing with open models.
- **[LUNAR: Unsupervised LLM-based Log Parsing](https://arxiv.org/abs/2406.07174)**: Unsupervised log parsing using semantic retrieval and caching.
- **[LogPrompt: Prompt Engineering Towards Zero-shot and Interpretable Log Analysis](https://arxiv.org/abs/2308.07610)**: Studies prompt-based approaches for log analysis and parsing.
- **[LLMParser: An Exploratory Study on Using Large Language Models for Log Parsing](https://doi.org/10.1145/3597503.3639150)**: Systematic study of LLM prompting/fine-tuning for log parsing.
- **[LibreLog / OpenLogParser (Ma et al., 2024)](https://arxiv.org/abs/2408.01585)**: Unsupervised log parsing with open-source LLMs and caching; representative of open implementations and retrieval-based parsing.
- **[Self-Evolutionary Group-wise Log Parsing Based on Large Language Model (SelfLog)](https://ieeexplore.ieee.org/document/10771304/)**: Group-wise parsing with an evolutionary update loop; another form of online adaptation.
- **[HELP: Hierarchical Embeddings-based Log Parsing](https://arxiv.org/abs/2408.08300)**: Retrieval-based parsing using hierarchical embeddings.
- **[ECLIPSE: Semantic Entropy-LCS for Cross-Lingual Industrial Log Parsing](https://arxiv.org/abs/2405.13548)**: Uses LCS-style matching with entropy signals; related to similarity-based retrieval.
- **[Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging](https://arxiv.org/abs/2402.18205)**: Uses entropy sampling and template merging; shows continued interest in state updates.
- **[LogRules: Enhancing Log Analysis Capability of Large Language Models through Rules](https://aclanthology.org/2025.findings-naacl.28.pdf)**: Rule-based knowledge extraction for log analysis; alternative to online demo pools.
- **[Matryoshka: Semantic-Aware Parsing for Security Logs](https://arxiv.org/abs/2506.17512)**: Offline parser generation to avoid runtime LLM exposure to untrusted logs.
- **[KELP: Robust Online Log Parsing Through Evolutionary Grouping Trees](https://arxiv.org/abs/2601.00633)**: Online log parsing under evolution; provides streaming evaluation ideas.
- **[Drain: An Online Log Parsing Approach with Fixed Depth Tree](https://doi.org/10.1109/ICWS.2017.13)**: Standard classical online parser baseline.
- **[SPELL: Streaming Parsing of System Event Logs](https://doi.org/10.1109/ICDM.2016.0103)**: Early streaming parser using LCS-like ideas.
- **[Data Poisoning for In-context Learning (ICLPoison)](./references/Data-Poisoning-for-In-context-Learning/meta/meta_info.txt)**: Shows that ICL is vulnerable to poisoned demonstrations in NLP tasks; motivates analogous risks in log parsing.
- **[Universal Vulnerabilities in Large Language Models: Backdoor Attacks for In-context Learning](https://arxiv.org/abs/2401.05949)**: Demonstrates backdoor-style attacks that activate via ICL demonstrations.
- **[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)**: Adversarial suffix attacks; related to attacker-controlled strings influencing LLM behavior.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| LLM log parsing with ICL + caching | Use few-shot prompts plus persistent caches to reduce cost | LILAC, AdaParser, LUNAR | LogHub-2.0, corrected LogHub | Assumes benign logs; persistent state can amplify errors |
| Template revision / merging | Update templates online to improve consistency | LILAC, Lemur | LogHub variants | Update rules can create overly broad templates |
| Demonstration selection / retrieval | Choose similar examples to guide parsing | DivLog, HELP, AdaParser | LogHub-2.0 | Similarity metrics can be brittle or biased |
| ICL poisoning / context attacks | Poison demonstrations or context to degrade performance | ICLPoison, adversarial suffix work | NLP classification; RAG settings | Limited coverage of online “system writes its own demos” loops |

### Closest Prior Work

- **AdaParser** (SG-ICL + corrector): stores model-generated demonstrations online; this proposal tests whether its safeguards prevent persistent poisoning under small injection budgets.
- **LILAC** (ICL + cache + template merging): closest in log parsing but focuses on template caches and merge rules; it does not study poisoning of an online demonstration pool built from model outputs.
- **ICLPoison**: demonstrates that poisoning ICL demonstrations can degrade performance, but in offline NLP classification tasks rather than online log parsing pipelines with internal verifiers.
- **SoK LLM log parsing**: surveys and benchmarks parsers, but does not evaluate adversarial log streams or persistence under poisoning.

**Novelty Kill Search Summary:** Searched for combinations of “AdaParser poisoning”, “log parsing SG-ICL poisoning”, “demonstration pool poisoning log parsing”, and checked for “in-context learning poisoning” applied to logs. Also searched local proposal drafts for (AdaParser|MicLog|SG-ICL|demo pool|ICL poisoning). No prior work explicitly evaluating persistent poisoning of AdaParser-style online demonstration pools was found as of 2026-02-21 (query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| AdaParser | Online demo pool + corrector for log parsing | No adversarial evaluation of demo-pool updates | Evaluate targeted vs random stream injection | Reveals whether safeguards prevent persistent corruption |
| LILAC | Caching + template revision for efficiency | State updates studied for accuracy/cost, not security | Treat log stream as attacker-influenced | Persistent-state robustness becomes measurable |
| ICLPoison | Poison demonstrations for ICL in NLP tasks | Not an online parser; no internal verifiers | Apply poisoning framing to online demo pools | Identifies a new practical attack surface |
| SoK log parsing | Taxonomy + benchmark harness | No adversarial/persistence tests | Add adversarial stream protocol | Standardizes a missing evaluation dimension |

---

## Experiments

### Experimental Setup

**Task:** log template parsing under an online stream with (optional) injected log lines.

**Target implementation:** AdaParser reference implementation (MIT): https://github.com/wuyifan18/AdaParser

**Dataset:** LogHub-2.0 (14 real-world log systems with labeled ground-truth templates; the AdaParser benchmark slice uses 2k logs per system). AdaParser expects LogHub-2.0 from Zenodo (record 8275861) under `full_dataset/`.

**Prompt / parameters:** Use AdaParser defaults: first 20% as available logs; candidate sampling size 32; demonstrations per query *k*=3 (AdaParser Sec. 4.5).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| gpt-4o-mini (or gpt-3.5-turbo-0125) | API | (available via our API pool) | Match AdaParser’s API-based setting; temperature=0; seed=0 (Sec. 4.5) |

No fine-tuning is required; experiments are inference-only.

**Resource Estimate**:
- **Compute budget**: GPU-hours ≈ 0 (API inference). If a local open model is substituted, expected ≤50 GPU-hours for 3 conditions × 3 seeds on 2k logs.
- **GPU memory**: none (API) or ≤1×A100-80GB (local 7B–14B inference).
- **API usage**: dominated by LLM calls on cache misses. Upper bound: O(#unique templates) per 2k logs; budget for ≤5,000 calls total (3 conditions × 3 seeds).

#### Phase-0 gates (fast checks before full runs)

- **G0 (near-collision precondition):** compute pairwise LCS similarity among ground-truth templates on each LogHub-2.0 dataset and pick a dataset with high near-collision density (e.g., 95th percentile similarity ≥0.75). If BGL fails, try Thunderbird.
- **G1 (insertion sanity):** verify that injected logs are accepted into the candidate set (candidate set size increases by N after the injected window).

**Main experiment conditions (3 conditions):**

For each dataset (BGL and Thunderbird), run the same 3 conditions:

- **C0 Clean:** run AdaParser on the original 2k-log stream.
- **C1 Targeted poison:** inject N=20 crafted logs into the first 200 logs; run AdaParser; evaluate on the last 500 logs of the original stream (excluding injected logs).
- **C2 Random-noise control:** inject N=20 random (non-targeted) logs using the control procedure; run AdaParser; evaluate on the last 500 clean logs.

**Seeds / runs:** 3 seeds per condition (randomness can come from candidate sampling and any residual LLM nondeterminism). Report mean ± std. AdaParser reports repeating random experiments 5 times (Sec. 4.5); we use 3 for budget.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| LogHub-2.0 (two datasets: BGL + Thunderbird) | 2k labeled logs per system for template parsing | **FTA** (primary), **PA** (secondary); plus retrieval contamination rate | last-500 evaluation window | https://zenodo.org/record/8275861 | AdaParser `benchmark/evaluation/` |

- **FTA (F1 score of Template Accuracy):** template-level metric computed as an F1 over correctly identified templates; a template is correct iff (i) its assigned logs share the same ground-truth template and (ii) the template tokens exactly match the ground truth (AdaParser Sec. 4.3.4).
- **PA (Parsing Accuracy):** per-log metric: the fraction of logs that are *fully* correctly parsed (all template and variable tokens correctly identified) (AdaParser Sec. 4.3.3).

### Main Results

#### Results Table

Baseline reference numbers from AdaParser Table 1 (Sec. 5.1.1) (full-dataset evaluation; not directly comparable to our late-window metric):
- BGL: Drain PA 40.7 / FTA 19.3; LILAC PA 97.5 / FTA 78.0; AdaParser PA 98.3 / FTA 84.9.
- Thunderbird: Drain PA 21.6 / FTA 7.1; LILAC PA 52.7 / FTA 56.0; AdaParser PA 72.6 / FTA 68.0.

Our main comparisons are C1 vs C0 and C2 vs C0 under the same late-window evaluation protocol.

| Method | Base Model | Benchmark | Late-window FTA (mean±std) | Late-window PA (mean±std) | Contamination rate (mean±std) | Source | Notes |
|---|---|---|---:|---:|---:|---|---|
| Drain | N/A | LogHub-2.0 (BGL) | 19.3 *(1 run)* | 40.7 *(1 run)* | N/A | AdaParser Table 1 | Full-dataset evaluation, not late-window |
| LILAC | gpt-3.5-turbo-0125 | LogHub-2.0 (BGL) | 78.0 *(1 run)* | 97.5 *(1 run)* | N/A | AdaParser Table 1 | Full-dataset evaluation, not late-window |
| AdaParser (clean ref) | gpt-3.5-turbo-0125 | LogHub-2.0 (BGL) | 84.9 *(1 run)* | 98.3 *(1 run)* | N/A | AdaParser Table 1 | Full-dataset evaluation, not late-window |
| **C0 Clean (ours)** | gpt-4o-mini | LogHub-2.0 (BGL) | **TBD** | **TBD** | **TBD** | - | Evaluate on last 500 logs |
| **C1 Targeted poison (ours)** | gpt-4o-mini | LogHub-2.0 (BGL) | **TBD** | **TBD** | **TBD** | - | Inject N=20 in first 200 logs |
| **C2 Random-noise control (ours)** | gpt-4o-mini | LogHub-2.0 (BGL) | **TBD** | **TBD** | **TBD** | - | Same N=20 budget |
| **C0 Clean (ours)** | gpt-4o-mini | LogHub-2.0 (Thunderbird) | **TBD** | **TBD** | **TBD** | - | Evaluate on last 500 logs |
| **C1 Targeted poison (ours)** | gpt-4o-mini | LogHub-2.0 (Thunderbird) | **TBD** | **TBD** | **TBD** | - | Inject N=20 in first 200 logs |
| **C2 Random-noise control (ours)** | gpt-4o-mini | LogHub-2.0 (Thunderbird) | **TBD** | **TBD** | **TBD** | - | Same N=20 budget |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Poison w/ SG-ICL updates disabled (diagnostic) | Do not insert newly parsed (log, template) into candidate set after the injected window | If persistence is due to demo-pool accumulation, late-window drop should shrink |
| Poison w/ larger k (diagnostic) | Increase demonstrations per query from 3 → 5 | If attack is demo-driven, larger k increases contamination and damage |

### Experimental Rigor

**Variance & Reproducibility:**
- Run 3 seeds per condition with temperature=0.
- Log all injected lines and random seeds.

**Validity & Controls:**
- **Confound: “any extra logs hurt.”** Controlled by C2 random-noise injection with the same budget and insertion operator.
- **Confound: injected logs never enter the pool.** Controlled by G1 and by logging candidate set membership.
- **Confound: effect is not retrieval-driven.** Controlled by reporting contamination rate and by the SG-ICL-disable diagnostic.

---

## Success Criteria

**Hypothesis** (directional — what you expect):
Targeted poisoning will cause a substantially larger late-window drop in FTA/PA than the random-noise control, and this drop will co-occur with a high retrieval contamination rate (many evaluation queries include poisoned demonstrations).

**Decision Rule** (concrete — when to stop):
- **Proceed**: C1 reduces late-window FTA by a margin outside the C0 vs C1 std range across 3 seeds (practically ≥5 FTA points) and C1 contamination rate is substantially higher than C2 (practically ≥2×).
- **Pivot**: If C1≈C2 on BGL but G0 indicates another dataset has higher near-collision density, switch to that dataset (e.g., Thunderbird) and repeat.
- **Refute**: If C1 and C2 both have negligible late-window degradation vs C0 and contamination rates remain low (<15%), conclude AdaParser’s SG-ICL demo pool is not practically poisonable under this attacker model and budget.

---

## Impact Statement

If the hypothesis holds, SRE and security teams deploying LLM-based log parsers would need to treat online demo pools as a persistent attack surface, similar to other long-lived caches, and adopt explicit monitoring/controls for what gets stored and retrieved. A negative result would also be decision-relevant: it would suggest that AdaParser’s internal verification and retrieval design are more robust than expected to low-budget log-stream manipulation.

---

## References

- [Log Parsing with Self-Generated In-Context Learning and Self-Correction](./references/Log-Parsing-with-Self-Generated-In-Context-Learning-and-Self-Correction/meta/meta_info.txt) - Wu et al., 2024
- [LILAC: Log Parsing using LLMs with Adaptive Parsing Cache](./references/LILAC-Log-Parsing-using-LLMs-with-Adaptive-Parsing-Cache/meta/meta_info.txt) - Jiang et al., 2024
- [SoK: System Log Parsing with Large Language Models: A Review](./references/SoK-LLM-based-Log-Parsing/meta/meta_info.txt) - Beck et al., 2025
- [Data Poisoning for In-context Learning](./references/Data-Poisoning-for-In-context-Learning/meta/meta_info.txt) - He et al., 2024
- [MicLog: Towards Accurate and Efficient LLM-based Log Parsing via Progressive Meta In-Context Learning](https://arxiv.org/abs/2601.07005) - Yu et al., 2026 (arXiv:2601.07005)
