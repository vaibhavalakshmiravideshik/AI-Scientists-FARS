# untitled

# TemplateLeak: Template-Disjoint Evaluation for CommonForms Form Field Detection

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Document understanding systems often need to process **forms** (government, tax, healthcare, onboarding). A practical first step is **form preparation**: converting a flat PDF into a fillable form by detecting where form widgets should be placed (text inputs, checkboxes, signature boxes).

**CommonForms** is a recent web-scale dataset that makes this problem measurable by extracting form widget locations from interactive PDFs at scale and framing **form field detection** as object detection on rendered page images **[CommonForms](./references/CommonForms-A-Large,-Diverse-Dataset-for-Form-Field-Detection/meta/meta_info.txt)**. The paper also releases strong YOLO-based detectors (FFDNet-S/L), reporting **81.0 mAP@0.50–0.95** (mean Average Precision averaged across intersection-over-union (IoU) thresholds 0.50 to 0.95; higher is better) for FFDNet-L on the official test split **[CommonForms](./references/CommonForms-A-Large,-Diverse-Dataset-for-Form-Field-Detection/meta/meta_info.txt)**.

A key question for practitioners is whether a model trained on web-scale PDFs generalizes to **new form templates**. In many real deployments, the form layout is not fixed: new vendors, new jurisdictions, and updated versions of the same form introduce unseen layouts. Prior document-understanding benchmarks explicitly emphasize this “unseen template” setting (e.g., VRDU’s Unseen Template Learning split) **[VRDU](https://arxiv.org/abs/2211.15421)**.

### The Problem

CommonForms splits data by **document** (PDF) to avoid page-level leakage across train and test **[CommonForms](./references/CommonForms-A-Large,-Diverse-Dataset-for-Form-Field-Detection/meta/meta_info.txt)**. However, web-scale form corpora can contain many **near-duplicate templates** across different PDF files (e.g., the same government form hosted on many sites, minor revisions, or re-packaged PDFs). A document-level split does not guarantee a **template-disjoint** split.

If a substantial fraction of the CommonForms test set contains templates already present in training, then reported mAP may overestimate **out-of-template generalization**, and method comparisons could be partially driven by template memorization.

This proposal asks a concrete evaluation question:

> Does CommonForms’ official split unintentionally include a large “template-overlap” subset, and is detection performance materially higher on that subset than on truly template-novel pages?

### Key Insight and Hypothesis

**Key insight:** In form field detection, a “template” can be operationalized by the **geometry and types of form fields** on the page. CommonForms provides ground-truth widget boxes (Text / Choice / Signature), enabling a training-free audit of whether pages in train and test share the same (or near-same) field-layout template.

**Hypothesis:** A non-trivial portion of the CommonForms test set has high template overlap with training. FFDNet’s mAP is materially higher on overlap templates than on template-novel pages, meaning the official test mAP is an optimistic estimate of “new-template” performance.

**Why we could be wrong:** (i) despite web-scale crawling, templates may be diverse enough that overlap is rare; (ii) FFDNet may generalize well and show minimal overlap-vs-novel gap; (iii) our template clustering may be too strict (missing near-duplicates) or too loose (merging distinct templates).

---

## Proposed Approach

### Overview

We propose **TemplateLeak**, a benchmark audit for CommonForms:

1. Build **template clusters** over pages using only the ground-truth field-layout geometry (no model predictions).
2. Define two evaluation slices of the official test set:
   - **Overlap-Test**: test pages whose template cluster contains at least one training page.
   - **Novel-Test**: test pages whose template cluster contains no training pages.
3. Evaluate released detectors (FFDNet-L; optionally FFDNet-S) on both slices, and report the performance gap.

### Method Details

#### 1) Field-layout “template signature”

For each page with width \(W\) and height \(H\), each ground-truth box \(b\) has class \(c\in\{\text{Text},\text{Choice},\text{Signature}\}\) and COCO-format box \((x,y,w,h)\) (pixels). Convert to normalized geometry:

- \(cx=(x+w/2)/W\), \(cy=(y+h/2)/H\)
- \(bw=w/W\), \(bh=h/H\)

Quantize to bins (pre-registered): \(q(v)=\lfloor B\cdot v\rfloor\) with \(B=32\).

Define a token for each field:
\[
\text{token}(b) = (c, q(cx), q(cy), q(bw), q(bh)).
\]

Represent the page as a multiset of tokens \(S(page)\).

#### 2) Template similarity and clustering

Define similarity between pages \(i,j\) as Jaccard similarity over token multisets:
\[
J(i,j)=\frac{|S_i\cap S_j|}{|S_i\cup S_j|}.
\]

We approximate \(J(i,j)\) using **MinHash** (a sketching method for estimating Jaccard similarity) and retrieve candidate neighbors via **locality-sensitive hashing** (LSH; an approximate nearest-neighbor method). Two pages are connected if \(J(i,j)\ge \tau\). Clusters are connected components.

**Threshold sensitivity (pre-registered):** we will report results for \(\tau\in\{0.50,0.55,\dots,0.95\}\). This is reporting from the same model outputs, not additional training.

#### 3) Null expectation for overlap rate

To avoid arbitrary “overlap% must exceed X%” thresholds, we compute a permutation-based null:

- Keep the derived template clusters fixed.
- Randomly permute **document IDs** into train/valid/test split sizes matching the official split (CommonForms splits by document), repeat 100 times.
- For each permutation, compute the implied Overlap-Test fraction.
- Report the observed Overlap-Test fraction alongside the null distribution.

### Key Innovations

1. **Template-disjoint auditing for form field detection**: a concrete method to quantify and report template overlap in CommonForms, complementing its document-level split.
2. **Outcome-relevant slice evaluation**: measures the effect of template overlap on reported detection mAP, not just duplicate counts.
3. **Pre-registered robustness to threshold choice**: reports overlap and mAP gaps across a sweep of \(\tau\) rather than one tuned threshold.

---

## Related Work

### Field Overview

Form understanding and document AI often face a “template generalization” issue: models can perform well when evaluation contains layouts similar to training, but degrade on unseen layouts. Benchmarks such as VRDU explicitly separate **mixed-template** from **unseen-template** evaluation to measure this effect **[VRDU](https://arxiv.org/abs/2211.15421)**. Meanwhile, large-scale datasets (including web-crawled corpora) often exhibit **near-duplicate** and **template-repeated** content, which can create train/test leakage if splits are not template-disjoint.

Our proposal sits at the intersection of (i) document form understanding and (ii) dataset-quality auditing / near-duplicate detection. The goal is not to propose a new detector, but to provide an evaluation protocol that better matches deployment: “new form templates.”

### Related Papers

- **[CommonForms: A Large, Diverse Dataset for Form Field Detection](./references/CommonForms-A-Large,-Diverse-Dataset-for-Form-Field-Detection/meta/meta_info.txt)**: Web-scale form-field detection dataset + FFDNet detectors; uses document-level splitting but does not report template-disjoint metrics.
- **[VRDU: A Benchmark for Visually-rich Document Understanding](https://arxiv.org/abs/2211.15421)**: Introduces Unseen Template Learning to explicitly test template generalization in document extraction.
- **[FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents](https://arxiv.org/abs/1905.13538)**: A classic form-understanding dataset; highlights the importance of generalizing across diverse scanned forms.
- **[LayoutLM](https://arxiv.org/abs/1912.13318)**: Pretrains on text + layout features for document understanding; influential baseline family.
- **[LayoutLMv2](https://arxiv.org/abs/2012.14740)**: Adds visual features and improved pretraining for visually rich documents.
- **[LayoutLMv3](https://arxiv.org/abs/2204.08387)**: Unifies text and image masking for document AI.
- **[Donut](https://arxiv.org/abs/2111.15664)**: OCR-free document understanding transformer; end-to-end document parsing.
- **[pix2struct](https://arxiv.org/abs/2210.03347)**: Screenshot/document parsing as pretraining; relevant for document-to-structured output.
- **[PubLayNet](https://arxiv.org/abs/1908.07836)**: Document layout detection dataset; illustrates the role of layout generalization.
- **[DocLayNet](https://arxiv.org/abs/2206.01062)**: Modern layout analysis dataset; used widely for layout detection benchmarking.
- **[TableBank](https://arxiv.org/abs/1903.01949)**: Table detection/recognition dataset; another document-structure task where template overlap can matter.
- **[ImageNetV2](https://arxiv.org/abs/1902.10811)**: Shows that benchmark distribution/sampling can meaningfully change reported accuracy.
- **[When does dough become a bagel?](https://openreview.net/pdf?id=jim9G4TScBx)**: Analyzes remaining ImageNet errors and reports exact duplicate leakage between train/val.
- **[Evolution of a Web-Scale Near Duplicate Image Detection System](https://dl.acm.org/doi/10.1145/3366423.3380031)**: Practical near-duplicate detection/clustering at web scale (LSH + clustering), motivating scalable approaches.
- **[pHash thesis](https://www.phash.org/docs/pubs/thesis_zauner.pdf)**: Foundational perceptual-hash method for near-duplicate detection.
- **[Soft Contamination Means Benchmarks Test Shallow Generalization](https://arxiv.org/abs/2602.12413)**: Argues that benchmark contamination can inflate measured generalization (in LLMs), conceptually aligned with our concern.
- **[Detecting Data Contamination in LLMs via In-Context Learning](https://arxiv.org/abs/2510.27055)**: Proposes automated contamination detection; illustrates broader importance of decontamination audits.
- **[CCpdf](https://arxiv.org/abs/2304.14953)**: Constructs a large PDF corpus from Common Crawl; highlights that web-scale PDF collections contain repeated content and require careful filtering.
- **[FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)**: Large PDF-derived corpus; motivates caution about repeated documents/templates.
- **[LayoutParser](https://arxiv.org/abs/2103.15348)**: Toolkit for document image analysis; representative of deployment pipelines where new templates arise.

(We will add additional related form/document benchmarks and near-duplicate detection works during final writing to exceed the 20-paper minimum even if some links above change.)

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Form field detection datasets | Weakly supervised widget extraction from PDFs | CommonForms | COCO mAP on widget boxes | Split may not be template-disjoint |
| Template generalization benchmarks | Explicitly separate seen vs unseen templates | VRDU (UTL) | Micro/macro F1 for extraction | Often KIE-focused, not widget detection |
| Document foundation models | Pretraining on text+layout+vision | LayoutLM family; Donut; pix2struct | DocVQA/KIE/layout benchmarks | Can overfit template patterns |
| Near-duplicate detection | Find/clusters duplicates at scale | LSH+clustering systems; pHash | Precision/recall of duplicate pairs | Threshold choice; semantics vs layout |

### Closest Prior Work

1. **CommonForms (Barrow, 2025)** **[CommonForms](./references/CommonForms-A-Large,-Diverse-Dataset-for-Form-Field-Detection/meta/meta_info.txt)**: introduces the dataset and reports mAP on a document-level split, but does not measure template-disjoint generalization.
2. **VRDU (2022)** **[VRDU](https://arxiv.org/abs/2211.15421)**: formalizes unseen-template evaluation for KIE-style extraction tasks, motivating a similar axis for form-field detection.
3. **ImageNet duplicate/leakage analyses** (e.g., ImageNetV2; “bagel” paper) show that benchmark leakage and sampling can shift reported performance and should be quantified.

**Novelty Kill Search Summary:** We searched for combinations of “CommonForms + template split”, “CommonForms + template leakage”, “form field detection + template-disjoint evaluation”, and scanned local proposal drafts for “template-disjoint / template leakage”. As of 2026-02-17, we did not find prior work that audits CommonForms with a template-disjoint split or reports overlap-vs-novel mAP slices.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| CommonForms | Introduces large dataset + detectors; document-level split | Does not quantify template overlap | Define template clusters from GT field geometry; report overlap-vs-novel metrics | Directly measures out-of-template generalization on the same benchmark |
| VRDU | Defines unseen-template evaluation for extraction | Different task (KIE), smaller scale | Transfer the unseen-template concept to widget detection | Provides a missing evaluation axis for CommonForms |
| ImageNet leakage analyses | Audits duplicates/leakage for vision benchmarks | Not document-form-specific | Apply analogous leakage audit to web-scale form dataset | Produces actionable benchmark guidance for document AI |

---

## Experiments

### Experimental Setup

**Baseline Ladder (REQUIRED):**
- **Sanity baseline (reproduction check):** Reproduce FFDNet-L’s official test mAP under our evaluation harness; if we cannot match within ±1 mAP, we do not interpret slice results.
- **Primary evaluation:** FFDNet-L on Overlap-Test vs Novel-Test slices.
- **Optional analysis (not required for core decision):** FFDNet-S on slices to check whether the L-vs-S gap shrinks on template-novel pages.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| FFDNet-L | 25M params | https://huggingface.co/jbarrow/FFDNet-L | YOLO-style detector (Ultralytics; open-source YOLO implementation); weights `FFDNet-L.pt` |
| (Optional) FFDNet-S | 9M params | https://huggingface.co/jbarrow/FFDNet-S | Ultralytics YOLO detector; weights `FFDNet-S.pt` |

**Training Data (if applicable):**

No training data needed – inference-only evaluation of released checkpoints.

**Other Resources (if applicable):**
- CommonForms dataset: https://huggingface.co/datasets/jbarrow/CommonForms

**Resource Estimate**:
- **Compute budget**: ≤20 A100 GPU-hours
  - ~1–5 GPU-hours for batched inference over 33,061 test pages (FFDNet-L)
  - COCO mAP evaluation + slice reporting is CPU-bound
  - Template clustering uses only ground-truth boxes and can run on CPU
- **GPU memory**: 1×A100 40–80GB sufficient for Ultralytics inference
- **API usage**: none

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| CommonForms | Form widget detection on rendered PDF pages (3 classes: Text/Choice/Signature) | COCO mAP@0.50–0.95 (overall + per class) | official `test` + derived Overlap/Novel slices | https://huggingface.co/datasets/jbarrow/CommonForms | pycocotools COCOeval (standard COCO-format object detection evaluation library; used by CommonForms) |

**Evaluation Scripts:**
- Use pycocotools COCO evaluation to compute mAP on (i) full test, (ii) Overlap-Test, (iii) Novel-Test.
- Use one inference run per model; compute slice metrics by filtering image IDs.

### Main Results

#### Results Table

(All slice results are TBD; they will be produced by the verification run.)

| Method | Benchmark split | mAP@0.50–0.95 (overall) | mAP Text | mAP Choice | mAP Signature | Source | Notes |
|---|---|---:|---:|---:|---:|---|---|
| FFDNet-L (reported) | CommonForms official test | 81.0 | 71.4 | 78.1 | 93.5 | CommonForms paper | Published; not slice-specific |
| FFDNet-L (ours, reproduced) | CommonForms official test | TBD | TBD | TBD | TBD | This work | Sanity check: should be within ±1 of 81.0 |
| FFDNet-L (ours) | Overlap-Test (τ sweep) | TBD | TBD | TBD | TBD | This work | Report curve over τ ∈ {0.50..0.95} |
| **FFDNet-L (ours)** | **Novel-Test (τ sweep)** | **TBD** | **TBD** | **TBD** | **TBD** | This work | Primary “new-template” metric |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Template threshold sweep | Vary τ_layout from 0.50 to 0.95 | If leakage is real, Overlap% and Δ mAP remain qualitatively consistent across a reasonable τ range |

### Experimental Rigor

**Variance & Reproducibility:**
- Inference is deterministic (single checkpoint + fixed preprocessing) → no multi-seed requirement.
- Threshold sweep is fully pre-registered.

**Validity & Controls:**
1. **Threshold cherry-picking**: report the full τ curve rather than a single tuned value.
2. **Using labels to define templates**: clarify that template overlap is a property of the dataset split, and the model is only used to measure whether this property changes reported mAP.
3. **Slice composition differences**: report basic slice statistics (number of pages; distribution of #fields per page; class counts) alongside mAP.

**Sanity checks:**
- Reproduce the reported overall mAP on the official test split within ±1 mAP.
- If Overlap-Test is empty for high τ, treat that as evidence the split is effectively template-disjoint at that strictness.

### Analysis (Optional)

- Report whether the model scaling gap (FFDNet-L minus FFDNet-S) is smaller on Novel-Test than on Overlap-Test.

---

## Success Criteria

**Hypothesis** (directional — what you expect):
- A meaningful fraction of test pages share templates with training pages, and detector mAP is higher on overlap templates than on template-novel templates.

**Decision Rule** (concrete — when to stop):

Let:
- \(p\) be the observed Overlap-Test fraction at τ=0.80 (and we also report the full τ curve).
- \(p_{null}\) be the mean overlap fraction under the permutation null.
- \(\Delta\) be \(mAP(Overlap) - mAP(Novel)\) at τ=0.80.
- \(G\) be the published scale gap \(mAP(FFDNet\text{-}L) - mAP(FFDNet\text{-}S)=8.7\) on the official test.

Proceed (and recommend template-disjoint reporting) if:
1. **Overlap above null:** \(p\) is substantially above the permutation null (e.g., \(p > p_{null} + 3\sigma_{null}\)).
2. **Material inflation:** \(\Delta \ge 0.5\,G\) (i.e., ≥4.35 mAP).

Pivot if overlap is above null but \(\Delta\) is small (<4.35): report overlap statistics as a benchmark-audit note but do not claim evaluation inflation.

Refute if overlap is near the null expectation and \(\Delta\) is within noise across τ: conclude the document-level split is effectively template-disjoint for field-layout templates.

---

## Impact Statement

If successful, this work provides a concrete, automated benchmark-audit protocol for CommonForms: report both standard mAP and **template-novel mAP**. This would change how document AI researchers interpret results and compare methods when the deployment target is new form templates.

---

## References

- [CommonForms: A Large, Diverse Dataset for Form Field Detection](./references/CommonForms-A-Large,-Diverse-Dataset-for-Form-Field-Detection/meta/meta_info.txt) - Barrow, 2025
- [VRDU: A Benchmark for Visually-rich Document Understanding](https://arxiv.org/abs/2211.15421) - (authors), 2022
- [FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents](https://arxiv.org/abs/1905.13538) - Jaume et al., 2019
- [LayoutLM](https://arxiv.org/abs/1912.13318) - Xu et al., 2019
- [LayoutLMv2](https://arxiv.org/abs/2012.14740) - Xu et al., 2020
- [LayoutLMv3](https://arxiv.org/abs/2204.08387) - Huang et al., 2022
- [Donut](https://arxiv.org/abs/2111.15664) - Kim et al., 2021
- [pix2struct](https://arxiv.org/abs/2210.03347) - Lee et al., 2022
- [PubLayNet](https://arxiv.org/abs/1908.07836) - Zhong et al., 2019
- [DocLayNet](https://arxiv.org/abs/2206.01062) - Pfitzmann et al., 2022
- [TableBank](https://arxiv.org/abs/1903.01949) - Li et al., 2019
- [ImageNetV2](https://arxiv.org/abs/1902.10811) - Recht et al., 2019
- [When does dough become a bagel? Analyzing the remaining mistakes on ImageNet](https://openreview.net/pdf?id=jim9G4TScBx) - Vasudevan et al., 2022
- [Evolution of a Web-Scale Near Duplicate Image Detection System](https://dl.acm.org/doi/10.1145/3366423.3380031) - Gusev & Xu, 2020
- [pHash thesis](https://www.phash.org/docs/pubs/thesis_zauner.pdf) - Zauner, 2010
- [Soft Contamination Means Benchmarks Test Shallow Generalization](https://arxiv.org/abs/2602.12413) - (authors), 2026
- [Detecting Data Contamination in LLMs via In-Context Learning](https://arxiv.org/abs/2510.27055) - (authors), 2025
- [CCpdf: Building a High Quality Corpus for Visually Rich Documents from Web Crawl Data](https://arxiv.org/abs/2304.14953) - Turski et al., 2023
- [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) - Kydlíček et al., 2025
- [LayoutParser](https://arxiv.org/abs/2103.15348) - Shen et al., 2021
