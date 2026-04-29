# untitled

# OCR-Anchor Reranking: Inference-Time Proxy Verification for Reliable PDF-to-Markdown Conversion

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Vision-language models (VLMs) are increasingly used to convert PDFs and document images into machine-readable formats such as plain text, Markdown, or HTML. This conversion is a common first step in retrieval-augmented generation (RAG) and document analytics pipelines. Compared to classical optical character recognition (OCR) pipelines, these models can better handle long-range layout dependencies (multi-column reading order, section hierarchy) and can emit structured outputs (tables, equations, lists). However, modern VLM-based OCR is also often **stochastic** at inference time (temperature sampling, long generations where early token choices steer later structure), and small decoding differences can cause large downstream failures (e.g., missing a table row, permuting reading order, or hallucinating header/footer text).

A standard reliability trick in generative modeling is **best-of-N** inference: generate multiple candidates and select one. In document OCR/parsing, best-of-N is appealing because it requires no retraining and can be applied to any base model. The practical bottleneck is **candidate selection**: the model’s own log-likelihood is not necessarily well calibrated to visual correctness, and directly comparing long structured strings is hard.

### The Problem

Given a base VLM document parser and a fixed sampling budget N, we want a **training-free** method to select the best candidate that correlates with downstream correctness. For document conversion, “correctness” is not just textual similarity: it includes omission of spurious content (headers/footers), reading order, table structure, and equation fidelity.

Recent work addresses parts of this but leaves a gap for inference-time candidate selection:
- **[olmOCR 2](./references/olmOCR-2-Unit-Test-Rewards-for-Document-OCR/meta/meta_info.txt)** improves document OCR mainly via training (RL with verifiable unit-test rewards), but does not study inference-time candidate reranking beyond its decoding heuristics.
- **[POINTS-Reader](./references/POINTS-Reader-Distillation-Free-Adaptation-of-Vision-Language-Models-for-Document-Conversion/meta/meta_info.txt)** uses PaddleOCR-based F1 filtering to curate self-labeled data during iterative self-improvement, but does not study best-of-N inference-time selection.
- **[Consensus Entropy](./references/Consensus-Entropy-Harnessing-Multi-VLM-Agreement-for-Self-Verifying-and-Self-Improving-OCR/meta/meta_info.txt)** proposes multi-VLM agreement as a training-free OCR uncertainty metric and routing signal, but requires running multiple VLMs and does not use classical OCR engines as independent verifiers.

This motivates a concrete question: **Can a cheap, independent OCR engine provide a useful proxy verification signal to select among VLM candidates at inference time?**

### Key Insight and Hypothesis

**Key insight:** Classical OCR engines (e.g., PaddleOCR) can often extract a small set of **high-confidence tokens** (e.g., medium-length alphanumeric words) even when they fail on global structure. These high-confidence tokens act as “anchors” that are visually grounded and largely independent of a VLM’s language prior.

**Hypothesis:** For a fixed candidate budget N, selecting the VLM candidate that maximizes **coverage of OCR-derived high-confidence anchors** improves end-task correctness on unit-test-based document parsing benchmarks, compared to selecting candidates by the VLM’s own score (log-likelihood).

**Why this could be wrong:**
1. OCR anchors may be unreliable on the hardest pages (old scans, tiny text), yielding little signal.
2. Anchors may over-emphasize visually present but undesired content (headers/footers), harming “absence” unit tests.
3. Model self-score may already correlate well with correctness once decoding and prompts are tuned.

---

## Proposed Approach

### Overview

We propose **OCR-Anchor Reranking**, a training-free inference-time method:

1. Run a classical OCR engine once on the input page to extract **high-confidence anchor tokens**.
2. Generate N candidate Markdown/text outputs from a base VLM document parser.
3. Score each candidate by how many anchors it contains (anchor coverage).
4. Select the highest-coverage candidate.

The method is designed to be a lightweight reliability layer: it uses external OCR only as a **proxy verifier**, not as the main extraction system.

### Method Details

**External OCR anchor extraction (PaddleOCR):**
- Run PaddleOCR on the input page image.
- From each recognized span, extract tokens using regex `[A-Za-z0-9]+`.
- Keep tokens with `len(token) ≥ 3` and `alnum_ratio(token) ≥ 0.8`.
- Keep only tokens whose parent span confidence ≥ **τ = 0.90** (PaddleOCR reports a per-span confidence score in [0,1]).
- Exclude tokens whose bounding-box center is in the top/bottom margins (default: y < 0.08·H or y > 0.92·H) to reduce header/footer anchoring.
- Deduplicate tokens (lowercased) and keep the top **K = 50** by confidence.

**Candidate generation (base VLM):**
- Generate **N = 8** candidates per page with the base model’s recommended decoding settings. For olmOCR, this includes **dynamic temperature scaling** (start at temperature 0.1 and increase up to 0.8 if the model fails to emit an end-of-sequence token), which is intended to get the benefits of low-temperature decoding while avoiding repetition loops (olmOCR 2 paper).

**Candidate scoring and selection:**
- Tokenize the candidate output with the same regex and lowercase.
- Define **Coverage(candidate) = |anchors ∩ candidate_tokens| / |anchors|**.
- Select the candidate with maximum coverage; ties are broken deterministically by smallest candidate index.

### Key Innovations

1. **OCR-as-proxy-verifier for VLM reranking**: use a classical OCR engine only to extract a sparse set of high-confidence visual anchors, rather than to produce the full output.
2. **Training-free reliability knob**: improves best-of-N selection without fine-tuning, reward modeling, or multi-VLM ensembles.
3. **Mechanism-targeted evaluation**: measures whether anchors help specifically on structure-sensitive unit tests (tables, reading order), rather than only on string similarity.

---

## Related Work

### Field Overview

Document parsing spans (i) classical OCR pipelines that detect text boxes and recognize strings, (ii) end-to-end VLMs that directly generate structured outputs from document images, and (iii) hybrid approaches that separate layout detection from content recognition to reduce high-resolution compute. Recent progress also includes reinforcement learning with verifiable rewards (RLVR) for document parsing and training-free self-verification/routing methods.

Our proposal sits at the intersection of **inference-time scaling** (best-of-N) and **programmatic verification**: instead of training with verifiable rewards, we use a lightweight verifier (external OCR anchors) at inference time to select a candidate.

### Related Papers

- **[olmOCR 2: Unit Test Rewards for Document OCR](./references/olmOCR-2-Unit-Test-Rewards-for-Document-OCR/meta/meta_info.txt)**: RLVR with unit-test rewards for PDF-to-text/markdown OCR; provides olmOCR-Bench and strong open baselines.
- **[Infinity-Parser: Layout-Aware Reinforcement Learning for Scanned Document Parsing](./references/Infinity-Parser-Layout-Aware-Reinforcement-Learning-for-Scanned-Document-Parsing/meta/meta_info.txt)**: GRPO + composite layout rewards for scanned document parsing.
- **[PaddleOCR-VL](./references/PaddleOCR-VL-Boosting-Multilingual-Document-Parsing-via-a-0.9B-Ultra-Compact-Vision-Language-Model/meta/meta_info.txt)**: compact two-stage VLM pipeline achieving strong OmniDocBench and olmOCR-Bench results.
- **[DeepSeek-OCR](./references/DeepSeek-OCR-Contexts-Optical-Compression/meta/meta_info.txt)**: optical token compression for long-document OCR.
- **[POINTS-Reader](./references/POINTS-Reader-Distillation-Free-Adaptation-of-Vision-Language-Models-for-Document-Conversion/meta/meta_info.txt)**: iterative self-improvement for document conversion; uses PaddleOCR text F1 filtering during training data curation.
- **[Consensus Entropy](./references/Consensus-Entropy-Harnessing-Multi-VLM-Agreement-for-Self-Verifying-and-Self-Improving-OCR/meta/meta_info.txt)**: training-free multi-VLM agreement metric for OCR reliability and routing.
- **[OmniDocBench](./references/OmniDocBench-Benchmarking-Diverse-PDF-Document-Parsing-with-Comprehensive-Annotations/meta/meta_info.txt)**: diverse PDF parsing benchmark with multi-module evaluation.
- **[Nougat](https://arxiv.org/abs/2308.13418)**: neural OCR for scientific documents (PDF-to-markup) with a focus on academic papers.
- **[GOT-OCR 2.0](https://arxiv.org/abs/2409.01704)**: end-to-end OCR model aiming at unified OCR theory.
- **[Donut](https://arxiv.org/abs/2111.15664)**: OCR-free document understanding via vision encoder + text decoder.
- **[pix2struct](https://arxiv.org/abs/2210.03347)**: image-to-text model for structured generation tasks including document-like inputs.
- **[LayoutLM](https://arxiv.org/abs/1912.13318)**: joint text-layout pretraining for document understanding.
- **[LayoutLMv2](https://arxiv.org/abs/2012.14740)**: improved multimodal document pretraining with visual features.
- **[LayoutLMv3](https://arxiv.org/abs/2204.08387)**: unified text-image masking for document representation.
- **[TrOCR](https://arxiv.org/abs/2109.10282)**: transformer-based OCR with encoder-decoder pretraining.
- **[DocOwl 1.5](https://arxiv.org/abs/2403.12895)**: OCR-free document understanding with unified structure learning.
- **[TextMonkey](https://arxiv.org/abs/2403.04473)**: OCR-free document model with token compression and shifted-window attention.
- **[CC-OCR](https://arxiv.org/abs/2412.02210)**: OCR benchmark for evaluating multimodal models across multiple OCR-centric tracks.
- **[OCRBench v2](https://arxiv.org/abs/2501.00321)**: large-scale benchmark for OCR capabilities of multimodal models, including structured parsing and reasoning tasks.
- **[Seeing is Believing? Mitigating OCR Hallucinations in MLLMs](https://arxiv.org/abs/2506.20168)**: GRPO training with rewards to reduce OCR hallucination under degraded visuals.
- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)**: self-consistency-based hallucination detection (general LLM setting; motivates selection signals).
- **[Model soups](https://arxiv.org/abs/2203.05482)**: weight-averaging for robustness; used by olmOCR 2.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Classical OCR pipelines | detect boxes + recognize text, then post-process layout | PaddleOCR (tool), Tesseract (tool) | task-specific metrics; end-to-end downstream | brittle on complex layout; weak global structure |
| End-to-end VLM parsing | generate long structured output directly from page image | Donut, Nougat, GOT-OCR | OmniDocBench, olmOCR-Bench | expensive at high resolution; stochastic outputs |
| Coarse-to-fine parsing | layout→crop→recognize to reduce compute | MinerU2.5, PaddleOCR-VL | OmniDocBench, olmOCR-Bench | errors propagate from layout stage |
| RLVR for OCR | train with verifiable, outcome-based rewards | olmOCR 2, Infinity-Parser | olmOCR-Bench, OmniDocBench | training cost; reward coverage gaps |
| Inference-time self-verification | use agreement/uncertainty to route/ensemble at inference | Consensus Entropy | OCRBench, custom OCR QA | requires multiple VLMs; not tailored to structure |
| **Proxy-verifier reranking (ours)** | best-of-N + external OCR anchors as selection signal | (this proposal) | olmOCR-Bench | anchors may fail on degraded scans; may bias to visible but undesired text |

### Closest Prior Work

1. **olmOCR 2** trains a document OCR VLM with RLVR and reports strong olmOCR-Bench performance, but does not study inference-time reranking signals beyond decoding heuristics (e.g., dynamic temperature scaling) and does not propose external tool-based selection.
2. **POINTS-Reader** uses PaddleOCR to filter self-labeled text examples during training, but it is a training-time data filtering mechanism, not an inference-time selection rule over multiple candidates.
3. **Consensus Entropy** provides a training-free selection/routing metric based on multi-VLM agreement, but it uses multiple VLMs and semantic embeddings rather than an external OCR tool as a proxy verifier.
4. **Multi-engine OCR fusion** (prior OCR literature) merges outputs from multiple OCR engines via voting/consensus, but typically does not address VLM-generated structured Markdown outputs nor selection among candidates from a single generator.

**Novelty Kill Search Summary:** We searched for direct combinations of best-of-N document OCR reranking with external OCR engines (queries including “best-of-n OCR reranking external OCR verifier PaddleOCR”, “best-of OCR reranking document parsing VLM”, and “consensus decoding OCR engine agreement rerank”), and reviewed the closest families (POINTS-Reader filtering; Consensus Entropy routing). As of **2026-02-17**, we did not find prior work that uses a sparse set of high-confidence classical-OCR anchors as a proxy verifier to rerank best-of-N candidates from a VLM document parser on unit-test-based benchmarks.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| olmOCR 2 | RLVR training with unit-test rewards | needs training; no inference-time reranking signal studied | inference-only reranking | avoids training cost; can be applied to any base model |
| POINTS-Reader | training-time PaddleOCR F1 filtering | not an inference-time selector | use OCR as proxy verifier for best-of-N selection | directly targets inference reliability |
| Consensus Entropy | multi-VLM agreement for OCR routing | requires multiple VLMs; not tool-based | single VLM + external OCR anchors | cheaper than multi-VLM; visually grounded anchors |
| Multi-engine OCR fusion | voting across OCR engines | not designed for VLM Markdown outputs | rerank VLM candidates instead of merging OCR outputs | leverages VLM structure while using OCR for grounding |

---

## Experiments

### Experimental Setup

**Task / benchmark:** PDF-to-Markdown (or PDF-to-text) conversion evaluated by **olmOCR-Bench**, a unit-test-based OCR benchmark (1402 PDFs, 7010 tests) that scores outputs by the fraction of programmatic checks passed (text presence/absence, reading order, tables, and math rendering) ([olmOCR 2](./references/olmOCR-2-Unit-Test-Rewards-for-Document-OCR/meta/meta_info.txt)).

**Baseline Ladder (REQUIRED):**
- **Prompting baseline:** Use the official olmOCR inference prompt and decoding settings (as in the olmOCR repo / paper).
- **Inference-time scaling baseline:** Best-of-N with the base model using a standard selection rule (self-score by log-likelihood).
- **Closest existing method family:** Training-free self-verification via multi-model agreement (Consensus Entropy) is the closest known inference-time reliability approach; we include it as a contextual baseline discussion (and optional experiment if compute allows).

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| olmOCR-2-7B-1025 | 7B | https://huggingface.co/allenai/olmOCR-2-7B-1025 | Base parser for all conditions |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---|---|---:|---|---|
| N/A | Inference-only | - | - | - |

**Other Resources (if applicable):**
- PaddleOCR engine: https://github.com/PaddlePaddle/PaddleOCR
- olmOCR evaluation code: https://github.com/allenai/olmocr

**Resource Estimate** (conservative, for full olmOCR-Bench):
- **VLM inference**: 1402 pages × N=8 candidates × 3 seeds ≈ 33.6k generations. Using reported high-throughput inference for olmOCR-style models on H100s (AI2 blog reports thousands of output tokens/sec for olmOCR 2), this should be well within **≤200 GPU-hours** on A100s with batching; likely much less.
- **External OCR**: PaddleOCR run once per page per seed (or cached across seeds) → negligible compared to VLM inference; can run on GPU or CPU.
- **Total**: expected **≤250 GPU-hours**, within the 768 GPU-hour cap.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| olmOCR-Bench | Unit-test-based PDF-to-text/Markdown OCR benchmark | Unit-test pass rate (0–1; higher is better) | test | https://huggingface.co/datasets/allenai/olmOCR-bench | https://github.com/allenai/olmocr (bench / eval scripts) |

### Main Results

We will report mean±std across 3 sampling seeds.

| Method | Base Model | Benchmark | Unit-test pass rate (mean±std) | Source | Notes |
|---|---|---|---:|---|---|
| olmOCR 2 (reported) | olmOCR-2-7B-1025 | olmOCR-Bench | **82.4±1.1** | olmOCR 2 (Table 3) | Reported with olmOCR decoding heuristics |
| N=1 (ours re-run) | olmOCR-2-7B-1025 | olmOCR-Bench | **TBD** | - | Single-sample baseline under our exact pipeline |
| Best-of-8 self-score | olmOCR-2-7B-1025 | olmOCR-Bench | **TBD** | - | Select by length-normalized log-probability |
| **Best-of-8 OCR-anchor coverage (ours)** | olmOCR-2-7B-1025 | olmOCR-Bench | **TBD** | - | Select by OCR-anchor coverage |

(We will also report per-category scores from olmOCR-Bench: Tables, Multi-column, Old scans, etc.)

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Format-only selection | Select best-of-8 by simple markdown validity heuristic | If this matches ours, gains are formatting-driven rather than OCR-driven |
| No margin filtering | Include top/bottom anchors | Expected to hurt Headers/Footers category if anchoring pulls in boilerplate |

### Experimental Rigor

**Variance & Reproducibility:**
- Sampling introduces randomness → run **3 seeds** for best-of-8 conditions and report mean±std.
- Cache the same 8 candidates per page per seed for both selection rules to enable paired comparisons.

**Confounders & controls:**
1. **Oracle headroom too small**: compute oracle@8 (best candidate by unit-test score) on a validation subset; if oracle gain over N=1 is <2 points, selection cannot help and we stop early.
2. **Proxy tautology**: anchors are sparse and do not encode table/math structure; report category-wise gains to test whether improvements extend beyond plain-text presence.
3. **Header/footer bias**: margin filtering is pre-registered; report Headers/Footers category separately.

**Sanity checks:**
- Random selection among the 8 candidates should underperform both selection rules.
- If anchor set is empty for a page, fall back deterministically to candidate 0; report the empty-anchor rate.

**Data leakage:**
- We do not train any model; we only compare inference-time selection rules on the same base model. Any benchmark contamination should affect all conditions similarly.

---

## Success Criteria

**Hypothesis**: OCR-anchor coverage provides a more visually grounded selection signal than model self-score, improving unit-test correctness at fixed N.

**Decision Rule**:
- **Proceed**: On olmOCR-Bench, OCR-anchor selection improves unit-test pass rate by **≥2.0 points** over self-score selection, with non-overlapping std across 3 sampling seeds, and achieves ≥50% of oracle@8 gain.
- **Pivot**: If improvements concentrate only on text-presence tests but not on structure-related categories (tables/multi-column), try a stricter anchor definition (higher τ / smaller K) or integrate a single structure diagnostic (format validity) as a secondary criterion.
- **Refute**: If OCR-anchor selection does not beat self-score by ≥1.0 point (or is within noise), or if it systematically hurts key categories (e.g., Headers/Footers), abandon OCR-anchor reranking for this benchmark.

---

## Impact Statement

If successful, this work provides an immediately deployable reliability layer for VLM-based document OCR/parsing: practitioners can keep their existing base model and sampling budget, but replace self-score reranking with a cheap external proxy verifier derived from classical OCR anchors. This could reduce silent document conversion failures in pipelines that ingest PDFs into downstream RAG or analytics systems.

---

## References

- [olmOCR 2: Unit Test Rewards for Document OCR](./references/olmOCR-2-Unit-Test-Rewards-for-Document-OCR/meta/meta_info.txt) - Poznanski et al., 2025
- [POINTS-Reader: Distillation-Free Adaptation of Vision-Language Models for Document Conversion](./references/POINTS-Reader-Distillation-Free-Adaptation-of-Vision-Language-Models-for-Document-Conversion/meta/meta_info.txt) - Liu et al., 2024
- [Consensus Entropy: Harnessing Multi-VLM Agreement for Self-Verifying and Self-Improving OCR](./references/Consensus-Entropy-Harnessing-Multi-VLM-Agreement-for-Self-Verifying-and-Self-Improving-OCR/meta/meta_info.txt) - Zhang et al., 2025
- [OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations](./references/OmniDocBench-Benchmarking-Diverse-PDF-Document-Parsing-with-Comprehensive-Annotations/meta/meta_info.txt) - Ouyang et al., 2024
- [Infinity-Parser: Layout-Aware Reinforcement Learning for Scanned Document Parsing](./references/Infinity-Parser-Layout-Aware-Reinforcement-Learning-for-Scanned-Document-Parsing/meta/meta_info.txt) - Wang et al., 2025
- [PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model](./references/PaddleOCR-VL-Boosting-Multilingual-Document-Parsing-via-a-0.9B-Ultra-Compact-Vision-Language-Model/meta/meta_info.txt) - Cui et al., 2025
- [DeepSeek-OCR: Contexts Optical Compression](./references/DeepSeek-OCR-Contexts-Optical-Compression/meta/meta_info.txt) - Wei et al., 2025
- [Donut: Document Understanding Transformer](https://arxiv.org/abs/2111.15664) - Kim et al., 2021
- [pix2struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347) - Lee et al., 2022
- [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) - Xu et al., 2019
- [LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding](https://arxiv.org/abs/2012.14740) - Xu et al., 2021
- [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387) - Huang et al., 2022
- [TrOCR: Transformer-based OCR with Pre-trained Models](https://arxiv.org/abs/2109.10282) - Li et al., 2021
- [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418) - Blecher et al., 2023
- [GOT-OCR 2.0 / General OCR Theory](https://arxiv.org/abs/2409.01704) - Wei et al., 2024
- [CC-OCR: A Comprehensive and Challenging OCR Benchmark for Evaluating Large Multimodal Models](https://arxiv.org/abs/2412.02210) - Yang et al., 2024
- [OCRBench](https://arxiv.org/abs/2312.12346) - Liu et al., 2023
- [Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](https://arxiv.org/abs/2506.20168) - He et al., 2025
- [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896) - Manakul et al., 2023
- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482) - Wortsman et al., 2022
