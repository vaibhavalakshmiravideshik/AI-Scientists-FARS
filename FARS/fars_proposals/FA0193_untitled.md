# untitled

# Patch, Don't Rewrite: Post-Drift Rule Updates for LogRules-Style LLM Log Parsers

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Log parsing converts raw log lines (free-form text emitted by software systems) into structured event templates with wildcards (e.g., replacing variable fields like IDs and timestamps with `<*>`). Accurate log parsing is a prerequisite for many IT operations (ITOps) workflows such as anomaly detection, incident triage, and root cause analysis, because it enables aggregation and statistical analysis over stable event types.

A persistent operational challenge is **log template drift**: after software upgrades, configuration changes, or new deployment environments, the surface form of log statements changes (new fields, renamed keys, delimiter changes, reordered tokens). Drift can silently break parsers, causing downstream monitoring pipelines to mis-aggregate events or miss incidents. This is especially problematic for online parsers used in practice (e.g., [Drain](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf)), which often require preprocessing rules that can become stale.

Recently, several papers have explored using large language models (LLMs) for log parsing. These methods typically rely on in-context demonstrations (e.g., [LogDiv](https://arxiv.org/abs/2307.09950), [LILAC](./references/LILAC-Log-Parsing-using-LLMs-with-Adaptive-Parsing-Cache/meta/meta_info.txt)) or batching/caching (e.g., [LogBatcher](./references/Stronger,-Cheaper-and-Demonstration-Free-Log-Parsing-with-LLMs/meta/meta_info.txt)) to reduce cost. LogRules introduces a different design point: it uses a larger LLM to induce an explicit natural-language **rule repository** and then applies those rules when parsing with a smaller model, achieving strong accuracy on LogHub-style benchmarks.

### The Problem

While explicit rule repositories are attractive for controllability and cost, they introduce a new lifecycle question: **what is the right update policy when drift occurs?** LogRules induces its rule repository once and does not study post-deployment refresh. In a production setting, a naïve response to drift is to “regenerate the whole rule set” from a small batch of post-drift examples. However, this global rewrite can be risky: limited post-drift evidence may cause the LLM to inadvertently change stable rules, creating regressions on templates that did not drift.

At the same time, there is prior work on drift-robust *template* maintenance (e.g., online clustering like Drain, and KELP’s evolutionary grouping trees), and LLM parsers with adaptive caches (LILAC, LogBatcher). These approaches update templates or clusters, but do not address the **staleness and maintenance of an explicit natural-language rule repository** as the primary object.

### Key Insight and Hypothesis

We hypothesize that, under a fixed post-drift update budget, **conservative additive updates** to an existing rule repository (generate a small set of drift-specific “delta rules” and prepend them) will yield better post-drift parsing accuracy than a full rule rewrite.

**Mechanism (why it might work):** This is the “conservative update vs full retraining” intuition from continual learning, applied to natural-language rule repositories. Post-drift labeled data is typically small relative to the pre-drift evidence that shaped the original repository. A global rewrite forces the LLM to re-specify *all* rules under limited evidence, which can introduce wording/priority changes that degrade stable-template parsing. In contrast, a patch keeps pre-drift rules unchanged and uses the update budget only to add rules that handle the new drifted formats.

We could be wrong if (i) the induced rules are already generic enough that drift does not localize to a small delta, or (ii) a carefully prompted rewrite can preserve stable rules as well as patching, making both policies equivalent.

---

## Proposed Approach

### Overview

We propose and evaluate a **post-deployment rule update policy** for LogRules-style parsers:

- Maintain a pre-existing ordered rule repository \(R_0\) (a fixed pre-deployment artifact).
- When drift occurs at time \(t_d\), collect a small labeled calibration set \(K_{update}\) from post-drift logs.
- Use a fixed update budget to either:
  - **Patch**: generate a small set of delta rules \(\Delta\) and prepend them to \(R_0\), or
  - **Rewrite**: regenerate a complete rule list \(R_{rewrite}\) using \(R_0\) and \(K_{update}\).

The core research question is whether patching improves **post-drift FGA** (F1 of grouping accuracy) relative to rewriting when both have the same access to artifacts and the same update budget.

### Method Details

**Rule-repository representation.** We follow LogRules in representing rules as short natural-language statements embedded in the log-parsing prompt. The parsing model is instructed to output a template string using `<*>` for variable fields.

**Pre-drift repository \(R_0\).** For reproducibility, \(R_0\) is generated once from a pre-drift calibration set \(K_0\) of labeled (log, template) pairs using a single LLM call. The repository has a fixed size \(N\) rules (we will use \(N=50\) as a default, and cap the total rule text to a fixed token budget to keep prompts comparable). \(R_0\) is ordered as produced by the LLM (or optionally re-ranked by rule-usage frequency on \(K_0\), matching LogRules’ ranking concept).

**Post-drift update budget.** We define the post-drift update budget \(B\) as:
- a labeled post-drift calibration set of size \(|K_{update}|\) (default \(|K_{update}|=200\)), and
- **one** LLM call that produces updated rules (Patch or Rewrite), with the same maximum output length and the same required repository size constraint (\(N\) rules for Rewrite; \(m\) rules for Patch).

**Update policies (3 conditions).**

1) **No update**: parse post-drift logs using \(R_0\).

2) **Rewrite (global refresh baseline)**: one-shot rewrite of the full repository.
- Input: \(R_0\) and \(K_{update}\).
- Output: a new ordered rule list \(R_{rewrite}\).
- Prompt constraint: the model is instructed to preserve stable rules unless contradicted by \(K_{update}\), but it must output a complete repository (not only a delta).

3) **Patch (ours)**: generate drift-specific delta rules without rewriting existing rules.
- Input: \(R_0\) and \(K_{update}\).
- Output: \(\Delta\), a small list of rules (e.g., \(m=5\) to \(10\)).
- Final repository: \(R_{patch} = \Delta\; +\; R_0\), with optional truncation to a fixed rule-token budget (applied equally to Patch and Rewrite).

**Fairness controls.**
- Same induction model for Patch and Rewrite.
- Same \(K_{update}\) and prompt length limits.
- Same parsing model and decoding settings.
- Same maximum rule-token budget in the parsing prompt.

### Key Innovations

- **Problem reframing**: treat the *rule repository* of LogRules-style parsers as a maintained artifact subject to drift, analogous to a model update policy, rather than a one-time static prompt component.
- **Minimal, controlled comparison**: define Patch vs Rewrite with equal access (both see \(R_0\)) and equal update budget (one LLM call), avoiding common confounds in “refresh vs patch” comparisons.
- **Drift-focused evaluation protocol**: measure post-drift accuracy on both **stable templates** and **drifted templates** to detect regressions caused by global rewriting.

---

## Related Work

### Field Overview

Classical log parsers largely rely on heuristics or clustering over tokenized messages, often assuming relatively stable formats. Online parsers (e.g., Drain) can adapt incrementally but can suffer under high-cardinality variables or when drift changes tokenization assumptions. More recent work has introduced LLM-based log parsing, typically as **in-context learning (ICL)** with demonstrations, sometimes augmented with caching and batching to reduce cost.

LogRules is a key recent step that introduces explicit rule induction: rather than relying only on demonstrations, it induces a reusable natural-language rule repository and uses rule-based prompting to guide smaller models. This raises a new systems question that is not yet well studied: how to maintain the rule repository when log formats evolve.

KELP argues that standard LogHub evaluation suffers from “ground truth leakage” due to regex-based annotation artifacts, and proposes a “zero-bias” synthetic protocol based on template extraction and high-entropy variable injection. This protocol is well suited for drift studies because it cleanly separates template structure from domain-specific variable formatting.

### Related Papers

- **[LogRules](./references/LogRules-Enhancing-Log-Analysis-Capability-of-Large-Language-Models-through-Rules/meta/meta_info.txt)**: Induces an explicit natural-language rule repository (via GPT-4o-mini) and uses rule-based prompting to improve LLM log analysis; does not study post-deployment drift refresh.
- **[KELP](./references/KELP-Robust-Online-Log-Parsing-Through-Evolutionary-Grouping-Trees/meta/meta_info.txt)**: Proposes a drift-robust online parser and introduces a “zero-bias” synthetic evaluation protocol that avoids LogHub ground-truth leakage.
- **[LILAC](./references/LILAC-Log-Parsing-using-LLMs-with-Adaptive-Parsing-Cache/meta/meta_info.txt)**: Uses LLM parsing with an adaptive cache of templates to reduce cost and improve consistency; updates templates rather than an explicit rule repository.
- **[LogParser-LLM](./references/LogParser-LLM-Advancing-Efficient-Log-Parsing-with-Large-Language-Models/meta/meta_info.txt)**: Studies efficiency-oriented LLM log parsing (prompting/engineering); does not focus on drift maintenance of rule repositories.
- **[LogBatcher](./references/Stronger,-Cheaper-and-Demonstration-Free-Log-Parsing-with-LLMs/meta/meta_info.txt)**: Demonstration-free LLM log parsing via clustering + caching + batching, emphasizing cost reduction and robustness to demo selection.
- **[LogDiv / Prompting for Automatic Log Template Extraction](https://arxiv.org/abs/2307.09950)**: GPT-3 in-context log template extraction using DPP-based diverse sampling and kNN example selection; illustrates demonstration-dependent LLM parsing and motivates update policies under drift.
- **[Drain](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf)**: A widely used online log parser using a fixed-depth parse tree; can suffer template explosion under high-cardinality data and relies on domain-specific preprocessing in many deployments.
- **[Spell](https://users.cs.utah.edu/~lifeifei/papers/spell-tkde19.pdf)**: Streaming log parsing using longest common subsequence (LCS) matching; sensitive to format changes and computational cost of matching.
- **[IPLoM](https://dl.acm.org/doi/10.1145/1557019.1557154)**: Iterative partitioning log mining (KDD 2009); a classic non-LLM baseline.
- **[LogMine](https://dl.acm.org/doi/10.1145/2983323.2983358)**: Clustering-based log pattern recognition and template extraction; illustrates trade-offs between clustering quality and scalability.
- **[Logram](https://arxiv.org/abs/2001.03038)**: Efficient parsing using n-gram dictionaries; referenced by LogRules as a baseline family.
- **[Brain](https://ieeexplore.ieee.org/document/10109145/)**: Tree-based log parsing using a bidirectional parallel tree; used as an open-source baseline in LogRules.
- **[Tools and Benchmarks for Automated Log Parsing (LogPai/LogHub)](https://arxiv.org/abs/1811.03509)**: Releases the LogPai toolkit and LogHub benchmark suite for log parsing; defines widely used evaluation protocols.
- **[A Large-Scale Evaluation for Log Parsing Techniques: How Far Are We? (LogHub-2.0)](https://arxiv.org/abs/2308.10828)**: Introduces LogHub-2.0 and emphasizes template-level metrics (e.g., FGA/FTA) under dataset imbalance; motivates drift-relevant evaluation beyond LogHub-2k.
- **[AEL (Abstracting Execution Logs to Execution Events)](http://www.cse.yorku.ca/~zmjiang/publications/QSIC2008.pdf)**: An early heuristics-based log abstraction method that uses anonymization + binning; a common baseline family in log parsing studies.
- **[MoLFI](https://dl.acm.org/doi/10.1145/3196321.3196340)**: Search-based (multi-objective) identification of log message formats; illustrates pre-LLM optimization approaches.
- **[SHISO](https://doi.org/10.1109/SCC.2013.73)**: Incremental mining of system log formats for streaming parsing; an early online parser baseline.
- **[DeepLog](https://users.cs.utah.edu/~lifeifei/papers/deeplog.pdf)**: LSTM-based log anomaly detection; illustrates why stable parsing is important for downstream monitoring.
- **[LogAnomaly](https://www.ijcai.org/proceedings/2019/658)**: Anomaly detection using semantic representations of templates; highlights issues caused by new/unseen templates between retrainings.
- **[LogPPT](https://arxiv.org/abs/2302.07435)**: Prompt-based few-shot log parsing with a pretrained encoder (RoBERTa); an early semantic/prompt baseline used in later LLM log parsing work.
- **[An Evaluation of Log Parsing with ChatGPT](https://arxiv.org/abs/2306.01590)**: Early study showing few-shot prompting can substantially improve ChatGPT’s log parsing, highlighting prompt sensitivity.
- **[LogShrink](https://arxiv.org/abs/2309.09479)**: Log compression leveraging commonality/variability; illustrates that log streams change structure and distributions over time.
- **[AdaParser: Log Parsing Using LLMs with Self-Generated In-Context Learning and Self-Correction](https://arxiv.org/abs/2406.03376)**: Uses self-generated demonstrations and self-correction with a cache/tree to handle scarce history and evolving logs; updates templates rather than rule repositories.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Heuristic / tree-based parsers | Hand-designed token matching and parse trees | Drain, Spell, IPLoM | LogHub-style datasets; GA/PA/FGA/FTA | Can break under drift and high-cardinality variables |
| Clustering / mining parsers | Cluster logs and mine templates | LogMine, MoLFI, SHISO | LogHub-style datasets | Often slower; quality depends on similarity metric |
| LLM ICL + cache/batch | Use LLM prompting with demonstrations, caching, batching | LogDiv, LILAC, LogBatcher | LogHub-2k / LogHub-2.0; GA/PA/FTA/ED; cost | Prompt sensitivity; drift can invalidate caches/demos |
| Explicit rule repositories | Induce reusable natural-language rules and apply them during parsing | LogRules | LogHub-2k; GA/PA/ED | Rule repo is static; no drift refresh policy |
| Drift-robust online parsers | Maintain evolving grouping structures under streaming | KELP (EGT), Drain (online) | Synthetic protocols; throughput + FGA/PA | Not designed for LLM rule repositories |

### Closest Prior Work

**LogRules** creates a rule repository once via induction and then uses it during deduction. It does not define how to refresh rules under drift, nor does it measure regressions caused by re-induction.

**LILAC** and **LogBatcher** maintain caches of templates to reduce LLM calls and improve consistency. They update caches when matching fails, but they do not maintain an explicit natural-language rule repository; their “update” is about templates/examples rather than rules.

**KELP** focuses on drift-robust online parsing and evaluation methodology. It does not use an LLM nor address rule repository updates, but it provides a strong benchmark protocol for drift.

**Novelty Kill Search Summary:** Searched for the exact combination of “LogRules rule update drift”, “rule repository refresh log parsing”, “LLM log parsing rule induction online update”, and “template drift + rule-based prompting” (web + local KB) and checked recent LLM log parsing papers (LogDiv/LILAC/LogBatcher/LogParser-LLM). No prior work explicitly evaluating **patch vs rewrite policies for natural-language rule repositories** in LLM log parsers was found as of 2026-02-20 (query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| LogRules | Induce static rule repo; rule-based prompting | No post-drift update policy | Define and test update policies | Drift requires updates; patching may avoid regressions |
| LILAC | ICL log parser with adaptive template cache | Updates templates, not rule repos | Focus on rule repository maintenance | Rules are cheaper to apply than per-case demos |
| LogBatcher | Demo-free batching + cache | No explicit rule repository | Separate “rules” from “templates” | Rules can generalize across many templates |
| KELP | Online drift-robust clustering; zero-bias protocol | Not LLM/rule-based | Use its protocol to evaluate rule updates | Protocol isolates true structural generalization |
| Drain | Online parse tree | Template explosion; heuristic fragility | Use as classical context baseline | Highlights need for robust drift handling |

---

## Experiments

### Experimental Setup

**Task:** log parsing (template induction) under a single drift event.

**Benchmark construction (based on KELP’s zero-bias protocol):**
1. **Template extraction:** start from Apache and Linux templates (as in KELP), erase original variables into placeholder slots.
2. **High-entropy injection:** generate synthetic streams by filling variable slots with random high-cardinality strings.
3. **Drift injection:** at time \(t_d\), select a fixed fraction of templates (default 30%) and apply one of three drift operators to their *static tokens*:
   - **Key rename:** replace a frequent static key token (e.g., `uid=`) with a new token (e.g., `user_id=`).
   - **Delimiter change:** change punctuation that separates fields (e.g., `;` to `|` or whitespace to `,`).
   - **Field insertion:** insert a new static field marker at a fixed position (e.g., `src=` before an existing variable).

We will pre-register drift severity by using the same operator mix across seeds (e.g., 1/3 each) and ensuring that drifted templates remain semantically aligned with the original (only surface form changes). After \(t_d\), logs from these templates follow the drifted format; other templates remain unchanged.

**Update protocol:**
- Build \(R_0\) from pre-drift calibration \(K_0\) (e.g., 500 labeled pairs).
- After drift, build \(K_{update}\) from post-drift logs (e.g., 200 labeled pairs).
- Run the three methods (No update / Rewrite / Patch) and evaluate on the post-drift stream.

**Primary metric:** **FGA** (F1 score of grouping accuracy; higher is better), computed on the post-drift stream.

**Slice metrics (diagnostics):** FGA on (i) stable templates after drift and (ii) drifted templates after drift.

**Baseline Ladder (REQUIRED):**
- **Zero-shot LLM log parsing (no rules)**: parse with the deduction model using only a task description (no \(R_0\)); this tests whether rules are actually necessary.
- **No update (\(R_0\) only)**: establishes that drift causes measurable degradation.
- **Rewrite (global refresh)**: a practitioner baseline for “refresh rules after drift”.
- **Patch (ours)**: proposed update policy.

To keep the main experiment to three update policies, the “zero-shot no-rules” baseline will be run once (deterministic decoding) and reported as a sanity/baseline row.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| Qwen2.5-7B-Instruct | 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Used for deduction (local inference via vLLM) |
| gpt-4o-mini | API | (available in our API model pool) | Used for rule induction/update (Patch/Rewrite) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| LogHub (Apache, Linux) | Source templates for synthetic protocol | N/A | https://github.com/logpai/loghub | See LogHub license |

No model fine-tuning is required; experiments are inference-only.

**Other Resources (if applicable):**
- KELP synthetic protocol description: used to implement dataset construction and evaluation constraints.

**Resource Estimate**:
- **Compute budget**: ≤ 50 A100 GPU-hours (dominated by running a 7B model over ~2k–10k logs for 3 methods × 3 drift seeds).
- **GPU memory**: ≤ 20GB (7B inference)
- **API usage**: a small number of calls (\(\le 5\) total) to gpt-4o-mini for \(R_0\) induction and post-drift Patch/Rewrite.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| KELP-style synthetic drift stream (Apache+Linux templates) | Synthetic log stream with high-entropy variables and an injected drift event | FGA (primary), GA, PA; plus stable/drifted FGA slices | time-split at \(t_d\) | https://github.com/logpai/loghub | Use LogHub/LogPai metric implementations; compute clusters from predicted templates |

### Main Results

#### Results Table

| Method | Base Model | Benchmark | FGA (mean±std) | Stable-slice FGA (mean±std) | Drifted-slice FGA (mean±std) | Source | Notes |
|--------|------------|-----------|----------------|-----------------------------|------------------------------|--------|-------|
| No update (R0) | Qwen2.5-7B | Synthetic drift stream | **TBD** | **TBD** | **TBD** | - | Should drop on drifted slice |
| Rewrite (global refresh) | Qwen2.5-7B | Synthetic drift stream | **TBD** | **TBD** | **TBD** | - | 1 LLM call update |
| **Patch (ours)** | Qwen2.5-7B | Synthetic drift stream | **TBD** | **TBD** | **TBD** | - | 1 LLM call update |

### Ablation Studies

| Variant | What's changed | Expected finding |
|---------|----------------|------------------|
| Patch w/o access to \(R_0\) | Generate \(\Delta\) using only \(K_{update}\) | Worse stable-slice performance; tests whether \(R_0\) reuse drives gains |
| Rewrite best-of-3 (optional) | Sample 3 rewrites and pick best on \(K_{update}\) | If Patch still wins, reduces “rewrite implemented poorly” concern |

### Experimental Rigor

**Variance & Reproducibility:**
- Use deterministic decoding for deduction (temperature 0) and run **3 drift seeds** (different drifted-template subsets and random string instantiations). Report mean ± std across seeds.

**Validity & Controls:**
- **Confounder: unequal artifact access.** Controlled by giving both Patch and Rewrite access to \(R_0\) and the same \(K_{update}\).
- **Confounder: prompt/compute mismatch.** Controlled by using the same parsing prompt (except rule list) and the same rule-token budget.
- **Sanity check:** Verify that No update suffers a clear drop on the drifted-template slice (otherwise drift is too weak and the experiment is not informative).

---

## Success Criteria

**Hypothesis** (directional — what you expect):
Patching (adding a small set of drift-specific delta rules while keeping \(R_0\) unchanged) should yield higher overall post-drift FGA than rewriting the entire repository, primarily by avoiding regressions on the stable-template slice.

**Decision Rule** (concrete — when to stop):
- **Continue/Proceed**: Patch improves overall post-drift FGA over Rewrite by a margin outside the std range across 3 drift seeds (practically, ≥2 FGA points) **and** Patch is not worse than Rewrite on drifted-slice FGA by >1 point.
- **Pivot**: If Patch ≈ Rewrite overall but Patch preserves stable-slice FGA better, explore hybrid policies (rewrite only low-confidence rules, or patch + re-rank).
- **Refute**: If Patch ≤ Rewrite on overall post-drift FGA (within noise), abandon the claim that conservative patching is a better default update policy.

---

## Impact Statement

If successful, this work provides a concrete, low-overhead maintenance policy for rule-based LLM log parsers: when drift occurs, spend scarce update budget on additive drift-specific rules rather than rewriting the whole rule base. This could reduce regressions and operational risk for SRE teams using LLM-assisted parsing in continuous deployment environments.

---

## References

- [LogRules](./references/LogRules-Enhancing-Log-Analysis-Capability-of-Large-Language-Models-through-Rules/meta/meta_info.txt)
- [KELP](./references/KELP-Robust-Online-Log-Parsing-Through-Evolutionary-Grouping-Trees/meta/meta_info.txt)
- [LILAC](./references/LILAC-Log-Parsing-using-LLMs-with-Adaptive-Parsing-Cache/meta/meta_info.txt)
- [LogParser-LLM](./references/LogParser-LLM-Advancing-Efficient-Log-Parsing-with-Large-Language-Models/meta/meta_info.txt)
- [Stronger, Cheaper and Demonstration-Free Log Parsing with LLMs (LogBatcher)](./references/Stronger,-Cheaper-and-Demonstration-Free-Log-Parsing-with-LLMs/meta/meta_info.txt)
- Additional related work links are listed in the Related Work section.