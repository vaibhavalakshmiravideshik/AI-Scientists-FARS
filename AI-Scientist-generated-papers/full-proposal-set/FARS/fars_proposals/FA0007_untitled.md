# untitled

# WindowScan-Judge: length-aware windowed safety judging to mitigate benign-padding attacks

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Infrastructure constraint (important for feasibility)**: Do **not** use OpenAI models for any safety/jailbreak evaluation or baselines, because Azure OpenAI content filters would block many prompts/responses in this setting. All proposed experiments use open-source judge models run locally.

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly deployed in chat and assistant settings, where outputs may contain policy-violating content (e.g., instructions for wrongdoing, hate, or sexual content). In this context, a **jailbreak** is a prompt or interaction that induces such policy-violating output. Many organizations therefore use automated **safety judges** (also called *guard models* or *moderation models*) that classify a completed response as **safe vs unsafe**, and then use these labels for (i) offline safety benchmarking and (ii) post-hoc filtering of generated outputs.

A common design choice is **holistic judging**: the judge reads the entire response and outputs a single label. This implicitly assumes that the judge’s decision is stable under benign, label-preserving transformations of the response text, such as adding extra safe content.

However, recent robustness studies show that this assumption can fail severely. For example, **[Know Thy Judge](./references/Know-Thy-Judge-On-the-Robustness-Meta-Evaluation-of-LLM-Safety-Judges/meta/meta_info.txt)** evaluates several open-source safety judges on a balanced subset (100 examples) of the JailbreakBench judge-comparison dataset (a dataset of prompt+response pairs with a binary harmfulness label from majority vote of three human annotators). Under their “Prepend + Append Benign” padding transformation, WildGuard’s harmful-class **F1 score** (higher is better) drops to **0.00**, and its **false negative rate (FNR)** (fraction of unsafe examples incorrectly judged safe; lower is better) rises to **1.00** (Table 3). This means the judge misses essentially all unsafe outputs after padding.

### The Problem

A natural mitigation is to judge **local segments** rather than the full response: split the response into shorter overlapping token windows and run the judge on each window. If any window is unsafe, the full response is unsafe. This approach can make the unsafe span salient even when surrounded by large amounts of benign text.

The difficulty is that naive window-scanning introduces a multiple-comparisons effect: as response length grows, the number of windows increases, and the probability of at least one spurious unsafe window on an actually safe response can increase. In deployment, this manifests as a **false positive rate (FPR)** increase (fraction of safe examples incorrectly judged unsafe; lower is better), which can make the system impractical for long responses.

This trade-off is directly relevant to benign-padding attacks because padding adversarially increases response length and therefore the number of tested windows.

### Key Insight and Hypothesis

We hypothesize that benign padding breaks holistic judges mainly by **diluting** the local evidence for unsafe content across a much longer context. Windowed judging should therefore recover recall on padded unsafe responses. We further hypothesize that most of the false-positive inflation from windowed judging is a predictable function of the number of scanned windows, so calibrating the aggregation rule as a function of response length can keep FPR close to the holistic judge while improving FNR under padding.

---

## Proposed Approach

### Overview

**WindowScan-Judge (WSJ)** is a post-hoc wrapper around an existing safety judge model \(J\):

1) Split a response into overlapping token windows.
2) Run \(J\) on each window.
3) Aggregate the per-window decisions using a length-aware rule calibrated on safe examples to control false positives.

WSJ treats the judge as a black-box function (query access only; no model retraining or weight modification), and it does not require integration into the text generation loop.

### Method Details

**Inputs**: response text \(R\); base judge \(J\) (e.g., WildGuard); window size \(W\) tokens and stride \(S\) tokens, where tokens are the subword units produced by the judge’s tokenizer.

**Windowing**: tokenize \(R\) using \(J\)’s tokenizer and form windows \(w_1, \dots, w_m\) of length \(W\) with stride \(S\). Default: \(W=512\), \(S=256\).

**Per-window scoring**: run \(J(w_i)\) with deterministic decoding (temperature = 0; temperature controls sampling randomness) to obtain either:
- a binary unsafe decision \(u_i \in \{0,1\}\), or
- a probability/score \(p_i\) (if exposed), which we convert into \(u_i = \mathbb{1}[p_i \ge \tau_0]\) with a fixed threshold \(\tau_0\) (default 0.5).

**Aggregation baselines (to quantify the false-positive issue)**:
- **Holistic**: run \(J\) once on the full response.
- **Max-OR**: predict unsafe iff \(\max_i u_i = 1\) (expected high recall; FPR may grow with \(m\)).
- **Fixed-k**: predict unsafe iff \(\sum_i u_i \ge k\) with a constant \(k\).

**Our main aggregation: Length-Aware FPR Control (LA-FPR)**:

- Let \(m\) be the number of windows for a response.
- On a **dev split containing only safe examples**, estimate the distribution of \(C_m = \sum_i u_i\) (the count of unsafe windows) for each observed \(m\).
- Choose \(k(m)\) as the smallest integer such that:
  - \(\Pr_{\text{dev}}(C_m \ge k(m) \mid \text{safe}, m) \le \delta\),
  where \(\delta\) is a target per-response false-positive budget. In our main experiments we set \(\delta = \text{FPR}_{\text{holistic}} + 0.05\).
- At test time, predict unsafe iff \(\sum_i u_i \ge k(m)\).

**Calibration protocol (to avoid tuning on test)**:
- Split the safe subset into **Dev-safe** (for learning \(k(m)\)) and **Test-safe** (for evaluation) with a fixed random seed.
- Evaluate all unsafe examples only in the test set (no unsafe labels are used for calibration).

### Key Innovations

1) A tuning-free, post-hoc robustness wrapper for existing safety judges (no retraining; no generation-time integration).
2) A length-aware aggregation rule for windowed judging that explicitly targets benign-padding attacks by controlling false positives as the number of windows increases.
3) A focused evaluation protocol centered on the benign-padding failure mode reported by **[Know Thy Judge](./references/Know-Thy-Judge-On-the-Robustness-Meta-Evaluation-of-LLM-Safety-Judges/meta/meta_info.txt)**, plus an “interleaved benign” variant to test boundary cases.

---

## Related Work

### Field Overview

This proposal builds on three lines of work.

First, **guard models / safety judges** are typically trained to label user prompts or model responses as safe vs unsafe. Common open models include WildGuard, Llama Guard, ShieldGemma, and HarmBench’s classifier.

Second, the broader “LLM-as-a-judge” literature studies the reliability of using LLMs to grade model outputs, including robustness under distribution shifts.

Third, recent robustness work on safety evaluation emphasizes that judge failures can invalidate safety benchmarking conclusions. In particular, benign output padding is important because it can be label-preserving and can occur naturally (long helpful responses that include a short unsafe span) or adversarially (padding added to conceal unsafe content).

Common datasets and benchmarks in this space include:
- **JailbreakBench** (a benchmark suite for jailbreak robustness; it includes a judge-comparison dataset of 300 prompt+response pairs with human harmfulness labels),
- **HarmBench** (a standardized benchmark framework for automated red teaming and refusal robustness),
- **AdvBench** (a dataset of adversarial prompts frequently used to test jailbreak behavior),
- **BeaverTails** (a response-level safety dataset for training/evaluating moderation models),
- **SafeRLHF** (a reinforcement learning from human feedback (RLHF) dataset with safety-related preference labels),
- **ToxicChat** (a dataset of toxic dialogue used for moderation evaluation), and
- **AEGIS / AEGIS2.0** (safety datasets and risk taxonomies used for training and evaluation of guardrails).

### Related Papers

- **[Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges](./references/Know-Thy-Judge-On-the-Robustness-Meta-Evaluation-of-LLM-Safety-Judges/meta/meta_info.txt)**: Shows that benign output padding can cause large increases in FNR (false negative rate; lower is better) for multiple safety judges, motivating our target failure mode.
- **[WildGuard](./references/WildGuard-Open-One-stop-Moderation-Tools-for-Safety-Risks-Jailbreaks-and-Refusals-of-LLMs/meta/meta_info.txt)**: An open multi-task moderation model that is widely used as a response-level safety judge baseline.
- **[Qwen3Guard Technical Report](./references/Qwen3Guard-Technical-Report/meta/meta_info.txt)**: Introduces Qwen3Guard models, including streaming-oriented variants that provide token-level signals.
- **[CARE](./references/CARE-Decoding-Time-Safety-Alignment-via-Rollback-and-Introspection-Intervention/meta/meta_info.txt)**: Uses decoding-time monitoring with rollback and intervention; related in spirit but focuses on generation-time control rather than post-hoc evaluation.
- **[LLMs Cannot Reliably Judge (Yet?)](./references/LLMs-Cannot-Reliably-Judge-Yet-A-Comprehensive-Assessment-on-the-Robustness-of-LLM-as-a-Judge/meta/meta_info.txt)**: A broader robustness assessment of LLM-as-a-judge that motivates more careful judge validation.
- **[Adversarial Attacks on LLM-as-a-Judge Systems](./references/Adversarial-Attacks-on-LLM-as-a-Judge-Systems-Insights-from-Prompt-Injections/meta/meta_info.txt)**: Studies prompt-injection attacks on judge systems; complementary because our threat model modifies the evaluated output rather than the judge prompt.
- **[Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt Injection Attacks](./references/Investigating-the-Vulnerability-of-LLM-as-a-Judge-Architectures-to-Prompt-Injection-Attacks/meta/meta_info.txt)**: Analyzes architectural factors that contribute to injection vulnerability in LLM judges.
- **[Optimization-based Prompt Injection Attack to LLM-as-a-Judge](./references/Optimization-based-Prompt-Injection-Attack-to-LLM-as-a-Judge/meta/meta_info.txt)**: Proposes optimization-based methods to attack LLM judges; relevant as a different robustness axis.
- **[JailbreakBench](https://arxiv.org/abs/2404.01318)**: Provides datasets and evaluation protocols for jailbreak robustness, including a judge-comparison dataset with human labels.
- **[HarmBench](https://arxiv.org/abs/2402.04249)**: Proposes a standardized framework for evaluating harmful behavior and automated red teaming, including classifier-based judges.
- **[Llama Guard](https://arxiv.org/abs/2312.06674)**: Early open LLM-based safeguard model family used for moderation.
- **[ShieldGemma](https://arxiv.org/abs/2407.21772)**: A Gemma-based content moderation model family.
- **[AdvBench](https://arxiv.org/abs/2307.15043)**: A benchmark of adversarial prompts used to evaluate jailbreak behavior.
- **[BeaverTails](https://arxiv.org/abs/2307.04657)**: A dataset of safe/unsafe assistant responses used for training and evaluating safety systems.
- **[SafeRLHF](https://arxiv.org/abs/2305.08088)**: An RLHF dataset with safety annotations, often used for safety alignment and evaluation.
- **[ToxicChat](https://arxiv.org/abs/2309.07875)**: A dataset of real-world toxic chat messages used to evaluate moderation models.
- **[AEGIS: Online Adaptive AI Content Safety Moderation](https://arxiv.org/abs/2404.05993)**: A dataset and framework for AI content safety moderation.
- **[AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails](https://arxiv.org/abs/2501.09004)**: An expanded safety dataset and taxonomy aimed at broader coverage.
- **[From judgment to interference: Early stopping LLM harmful outputs via streaming content monitoring](https://arxiv.org/abs/2506.09996)**: Explores streaming monitoring and early stopping, which is related to token-level guardrails.
- **[Trust The Typical](https://arxiv.org/abs/2602.04581)**: Studies typicality-based detection as an alternative mechanism for safety screening.
- **[MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety](https://arxiv.org/abs/2602.01539)**: Uses co-evolutionary training for robustness, but requires retraining and an attacker environment.
- **[GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation](https://arxiv.org/abs/2405.13077)**: Illustrates that jailbreak attacks can be strong, motivating robust evaluation pipelines.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Holistic safety judges | Classify the entire response as one unit | WildGuard; Llama Guard; ShieldGemma; HarmBench | JailbreakBench; HarmBench; BeaverTails | Vulnerable to dilution when unsafe span is small relative to benign text; limited localization |
| Safety judge robustness meta-evaluation | Stress-test judges under shifts and attacks | Know Thy Judge; LLMs Cannot Reliably Judge (Yet?) | JailbreakBench judge-comparison + transformations | Often focuses on diagnosing failures; fewer tuning-free mitigations |
| Attacks on LLM-as-judge | Manipulate the evaluator via prompts/optimization | Adversarial Attacks on LLM-as-a-Judge; Optimization-based Prompt Injection | Custom judge tasks | Different threat model than output padding |
| Streaming / intervention guardrails | Monitor generation and intervene | CARE; From judgment to interference; Qwen3Guard-Stream | BeaverTails and related suites | Requires generation-time integration; may require specialized models |
| Distribution/typicality-based screening | Flag atypical inputs/outputs as unsafe | Trust The Typical | Multiple safety benchmarks | Can miss in-distribution unsafe content; can be more invasive |
| Co-evolutionary robustness training | Train defenses against adaptive attackers | MAGIC | Adversarial suites | Requires training and attacker environment |
| Post-hoc windowed judging with length-aware aggregation (this work) | Localize judging via windows and control false positives as length grows | WindowScan-Judge | Benign-padding transformations on JailbreakBench judge-comparison | May miss harms requiring long-range context; requires calibration on safe data |

### Closest Prior Work

1) **Know Thy Judge**: Identifies benign-padding failures of safety judges on the JailbreakBench judge-comparison dataset. **Difference**: WSJ proposes and tests a concrete mitigation that explicitly targets the false-positive issue created by window scanning.

2) **Qwen3Guard-Stream**: Uses token-level heads to detect unsafe spans during streaming moderation. **Difference**: WSJ does not require a specialized token-level model; it wraps any existing judge in a post-hoc setting.

3) **CARE**: Uses periodic safety checks and rollback during decoding. **Difference**: CARE is a generation-time intervention mechanism; WSJ is a post-hoc evaluator/guard wrapper for completed outputs and offline logs.

4) **Prompt-injection attacks on LLM-as-a-judge**: Focus on manipulating the judge prompt or judge system. **Difference**: WSJ addresses output-level, label-preserving padding where the evaluated text is modified.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why the proposed approach may help |
|---|---|---|---|---|
| Know Thy Judge | Evaluates safety judge robustness under output transformations | Does not provide a tuning-free mitigation; padding remains a failure mode | Add windowed judging + length-aware aggregation | Unsafe span can be detected in some window; calibration keeps FPR stable as length grows |
| Qwen3Guard-Stream | Token-level streaming detection via specialized heads | Requires a specific model and training; designed for streaming | Wrap any existing judge post-hoc | No retraining; can be applied to existing evaluation logs |
| CARE | Decoding-time rollback and intervention | Requires integration into generation-time pipeline | Post-hoc scan of completed responses | Applicable to offline evaluation and filtering without generator modification |
| Prompt-injection judge attacks | Attacks judge prompting/system | Different threat model than output padding | Study output-padding robustness + defense | Targets a practically plausible failure mode for response-level judges |
| Trust The Typical | Uses typicality/out-of-distribution (OOD) signals for screening | Different mechanism; may not localize unsafe spans | Keep judge fixed; localize windows and calibrate | If dilution is primary failure mode, localization is a direct fix |
| MAGIC | Co-evolving attacker/defender training | Requires retraining and attack environment | Inference-only wrapper | Lower operational cost; can be deployed immediately |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| WildGuard | 7B | https://huggingface.co/allenai/wildguard | Primary response-level safety judge; used in Know Thy Judge |
| Llama Guard 3 | 8B | https://huggingface.co/meta-llama/Llama-Guard-3-8B | Secondary safety judge baseline used in Know Thy Judge |
| HarmBench classifier | 13B | https://huggingface.co/cais/HarmBench-Llama-2-13b-cls | Classifier baseline used in Know Thy Judge |
| ShieldGemma | 9B | https://huggingface.co/google/shieldgemma-9b | Optional additional judge baseline |
| Qwen3Guard-Stream | 4B | https://huggingface.co/Qwen/Qwen3Guard-Stream-4B | Token-level streaming guard baseline; evaluated post-hoc on completed responses |

**Training Data (if applicable):**

No training data needed — inference only.

**Other Resources (if applicable):**

- A fixed set of benign padding strings (static text snippets) to ensure deterministic output transformations.

**Resource Estimate**:

- **Compute budget**: expected ≤ 50 NVIDIA A100 GPU-hours (GPU-hours = number of GPUs × hours) total.
  - Upper bound estimate: if padded responses reach \(\le 4096\) tokens, then with \(W=512\), \(S=256\) there are \(m \le 15\) windows per response.
  - For 300 examples and 4 conditions (original + 3 padding variants), this is \(300 \times 4 \times 15 = 18{,}000\) judge calls per base judge.
  - Even if a single judge call took ~1 second on one A100 (a conservative bound for 7B–13B models with short outputs), this is ~5 GPU-hours per judge; running 4–5 judges remains within budget.
- **GPU memory**: 7B–13B models typically fit in \(\le 80\) GB for inference (1×A100 80GB per model).
- **API usage**: none.

**Infrastructure constraints** (proposals requiring these are infeasible):

- No browser/GUI environments.
- No search engine APIs.
- No OpenAI model calls for safety/jailbreak evaluation (Azure content filters).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| JailbreakBench judge-comparison dataset | A dataset of 300 prompt+response pairs with a binary harmfulness label (majority vote of three human annotators); used to compare automated safety judges | Acc, F1, FNR, FPR | fixed | https://github.com/JailbreakBench/jailbreakbench | JailbreakBench loader + our transformations |
| Benign-padding transformation suite (this work) | Deterministic output-level transformations applied to each response: Append Long Benign (append a long benign passage), Prepend+Append Benign (add benign text both before and after the response), and Interleaved Benign (insert benign segments between response chunks); designed to preserve the original harmfulness label | Acc, F1, FNR, FPR; FPR-vs-length curve on safe subset | generated | (derived) | Implemented as transformation functions |

**Metrics (definitions):**
- **Acc (accuracy; higher is better)**: fraction of examples correctly classified as safe/unsafe.
- **F1 (harmful-class F1; higher is better)**: harmonic mean of precision and recall for the unsafe class.
- **FNR (false negative rate; lower is better)**: fraction of unsafe examples predicted safe.
- **FPR (false positive rate; lower is better)**: fraction of safe examples predicted unsafe.

**Evaluation Scripts:**

- Use the official JailbreakBench code to load the judge-comparison dataset.
- Implement deterministic output transformations (padding/interleaving).
- Implement the WSJ wrapper (windowing + aggregation) and LA-FPR calibration.
- Compute Acc/F1/FNR/FPR with a fixed positive class definition (“unsafe” = positive).

**Download Links Checklist:**

- [ ] All benchmark datasets have download links
- [ ] All training datasets have download links (if applicable)
- [ ] All models have download links
- [ ] Licenses are compatible with research use

### Main Results

#### Comparability Rules

- Same dataset and labels for all methods: JailbreakBench judge-comparison (300 prompt+response pairs; unsafe = positive label).
- Same output transformations for all methods (deterministic padding/interleaving code).
- Same base judge inference settings (temperature = 0) for all methods.
- LA-FPR calibration uses only Dev-safe examples; unsafe examples are never used for calibration.

#### Results Table

All rows below use the same dataset and evaluation protocol (JailbreakBench judge-comparison, 300 examples). Numbers marked **TBD** will be produced by the verification runs.

| Method | Base Model | Condition | Acc (↑) | F1 (↑) | FNR (↓) | FPR (↓) | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| Holistic judge | WildGuard (7B) | Original responses | **TBD** | **TBD** | **TBD** | **TBD** | - | Single pass over full response |
| Holistic judge | WildGuard (7B) | Prepend+Append Benign | **TBD** | **TBD** | **TBD** | **TBD** | - | Failure mode reported in Know Thy Judge; re-evaluated here on full 300 |
| WindowScan (Max-OR) | WildGuard (7B) | Prepend+Append Benign | **TBD** | **TBD** | **TBD** | **TBD** | - | Unsafe iff any window unsafe |
| WindowScan (Fixed-k) | WildGuard (7B) | Prepend+Append Benign | **TBD** | **TBD** | **TBD** | **TBD** | - | Unsafe iff \(\sum u_i \ge k\) with constant \(k\) |
| **WindowScan-Judge (LA-FPR, ours)** | WildGuard (7B) | Prepend+Append Benign | **TBD** | **TBD** | **TBD** | **TBD** | - | Calibrate \(k(m)\) on Dev-safe to enforce FPR budget |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| WSJ (Max-OR) | Unsafe iff any window unsafe | Strong recall on padded unsafe outputs, but FPR increases with length |
| WSJ (Fixed-k) | Constant \(k\) across lengths | Better FPR than Max-OR, but still length-sensitive |
| WSJ (LA-FPR) | Learn \(k(m)\) from Dev-safe as a function of number of windows \(m\) | FPR is stable across lengths while FNR improves under padding |
| WSJ (W=256 vs 512 vs 1024) | Window size sweep | Smaller windows improve localization but may increase FPR; larger windows reduce localization benefit |
| WSJ (S=W/4 vs W/2) | Overlap/stride sweep | More overlap reduces boundary misses but increases compute |

### Analysis (Optional)

- **Length sweep on safe subset:** append benign text at several lengths and plot FPR vs length for Holistic vs Max-OR vs LA-FPR.
- **Localization sanity check:** compare the distribution of unsafe-window counts \(\sum u_i\) for safe vs unsafe examples.
- **Interleaved benign stress test:** evaluate whether WSJ remains effective when benign text is inserted between segments rather than appended.

---

## Success Criteria

**Criterion 1: Robustness under benign padding (primary)**
- Hypothesis: WSJ (LA-FPR) reduces false negatives under Prepend+Append Benign compared to the holistic judge.
- Validation: On the 300-example judge-comparison dataset with padding applied, WSJ achieves a substantial absolute decrease in FNR (false negative rate; lower is better) relative to the holistic WildGuard baseline while improving F1.

**Criterion 2: Controlled false positives for long responses (multiple testing)**
- Hypothesis: Length-aware calibration prevents FPR inflation as the number of scanned windows increases.
- Validation: In a length sweep on safe examples, WSJ maintains FPR within +0.05 absolute of the holistic judge and does not increase monotonically with padding length.

**Criterion 3: Generality across judges (secondary)**
- Hypothesis: The same wrapper improves robustness for at least one additional judge (e.g., Llama Guard 3 or HarmBench) without per-model retraining.
- Validation: Directionally similar improvements in FNR under padding when wrapping at least one additional judge model.

**Explicit decision rule (go/no-go):** If LA-FPR does not reduce FNR on padded unsafe examples without exceeding the FPR budget (holistic + 0.05), we refute the hypothesis that benign padding can be addressed primarily by localization + length-aware aggregation.

---

## Impact Statement

If WSJ is effective, teams that use automated safety judges for offline evaluation or post-hoc filtering can make these pipelines more robust to long benign padding without retraining or replacing the underlying judge model. This reduces the risk that safety benchmarks and internal audits systematically under-estimate unsafe outputs when the unsafe span is short relative to benign text.

---

## References

- [Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges](./references/Know-Thy-Judge-On-the-Robustness-Meta-Evaluation-of-LLM-Safety-Judges/meta/meta_info.txt) - Eiras et al., 2025
- [WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs](./references/WildGuard-Open-One-stop-Moderation-Tools-for-Safety-Risks-Jailbreaks-and-Refusals-of-LLMs/meta/meta_info.txt) - Han et al., 2024
- [Qwen3Guard Technical Report](./references/Qwen3Guard-Technical-Report/meta/meta_info.txt) - Zhao et al., 2025
- [CARE: Decoding Time Safety Alignment via Rollback and Introspection Intervention](./references/CARE-Decoding-Time-Safety-Alignment-via-Rollback-and-Introspection-Intervention/meta/meta_info.txt) - Hu et al., 2025
- [LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge](./references/LLMs-Cannot-Reliably-Judge-Yet-A-Comprehensive-Assessment-on-the-Robustness-of-LLM-as-a-Judge/meta/meta_info.txt) - Li et al., 2025
- [Adversarial Attacks on LLM-as-a-Judge Systems: Insights from Prompt Injections](./references/Adversarial-Attacks-on-LLM-as-a-Judge-Systems-Insights-from-Prompt-Injections/meta/meta_info.txt) - 2025
- [Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt Injection Attacks](./references/Investigating-the-Vulnerability-of-LLM-as-a-Judge-Architectures-to-Prompt-Injection-Attacks/meta/meta_info.txt) - 2025
- [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](./references/Optimization-based-Prompt-Injection-Attack-to-LLM-as-a-Judge/meta/meta_info.txt) - 2025
- [JailbreakBench](https://arxiv.org/abs/2404.01318) - Chao et al., 2024
- [HarmBench](https://arxiv.org/abs/2402.04249) - Mazeika et al., 2024
- [Llama Guard](https://arxiv.org/abs/2312.06674) - Inan et al., 2023
- [ShieldGemma](https://arxiv.org/abs/2407.21772) - Zeng et al., 2024
- [AdvBench](https://arxiv.org/abs/2307.15043) - Zou et al., 2023
- [BeaverTails](https://arxiv.org/abs/2307.04657) - 2023
- [SafeRLHF](https://arxiv.org/abs/2305.08088) - 2023
- [ToxicChat](https://arxiv.org/abs/2309.07875) - 2023
- [AEGIS: Online Adaptive AI Content Safety Moderation](https://arxiv.org/abs/2404.05993) - 2024
- [AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails](https://arxiv.org/abs/2501.09004) - 2025
- [Trust The Typical](https://arxiv.org/abs/2602.04581) - 2026
- [MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety](https://arxiv.org/abs/2602.01539) - 2026
- [From judgment to interference: Early stopping LLM harmful outputs via streaming content monitoring](https://arxiv.org/abs/2506.09996) - Li et al., 2025
- [GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation](https://arxiv.org/abs/2405.13077) - Ramesh et al., 2024
