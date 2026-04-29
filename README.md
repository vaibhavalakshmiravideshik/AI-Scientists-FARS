# FARS Paper Generation Benchmark

## Overview

This repository is a curated research artifact collection for the comparative study of AI-generated scientific papers derived from **FARS (Fully Automated Research System)** proposal documents. It brings together the source proposal corpus, paper outputs produced by multiple AI scientist systems, and reviewer-side evaluation artifacts used in downstream comparative analysis.

The repository is intended to support publication-facing documentation and artifact inspection for the pilot and full experimental collections associated with this project.

At a high level, the workflow represented here is:

1. FARS proposal documents are constructed from FARS (Fully Automated Research System) repositories.
2. Multiple AI scientist systems generate papers from those proposal documents.
3. Reviewer-side outputs are collected and organized for comparative evaluation.
4. Pilot-set and full-set artifacts are archived in a consistent, navigable structure.

An important methodological distinction in this repository concerns the Sakana-generated papers. In the **pilot set**, Sakana v1 and Sakana v2 were run in a multi-idea setting: for each proposal, three paper drafts corresponding to three generated ideas were produced and then merged by an LLM into a single final paper artifact. In the **full set**, Sakana v1 and Sakana v2 were run in a single-idea setting, producing one paper per proposal without the pilot-stage multi-draft merging procedure. This distinction is important for interpreting the pilot and full collections, and the two architecture figures below summarize the corresponding workflows.

<p align="center">
  <img src="./readme_assets/pilot-set-architecture.png" alt="Pilot-set Sakana architecture" width="47%" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./readme_assets/full-set-architecture.png" alt="Full-set Sakana architecture" width="47%" />
</p>

---

## What this repository contains

This repository contains three principal artifact groups.

| Artifact group | Description | Primary location |
|---|---|---|
| FARS source material | Proposal documents and FARS paper artifacts that serve as the source/reference corpus | [`AI-Scientist-generated-papers/pilot-paper-set/FARS/`](./AI-Scientist-generated-papers/pilot-paper-set/FARS/) and [`AI-Scientist-generated-papers/full-proposal-set/FARS/`](./AI-Scientist-generated-papers/full-proposal-set/FARS/) |
| AI-scientist-generated papers | Papers generated from FARS proposals by multiple automated scientific writing systems | [`AI-Scientist-generated-papers/`](./AI-Scientist-generated-papers/) |
| Reviewer and analysis artifacts | Reviewer-side outputs, score tables, JSON exports, and pilot-set comparison artifacts | [`AI-Reviewer-outputs/`](./AI-Reviewer-outputs/) |

---

## AI scientist systems represented in this repository

The repository compares generated paper outputs from four AI scientist systems against the FARS paper set.

| System | Reference |
|---|---|
| FARS | **Fully Automated Research System (FARS)** proposal-to-paper source set represented directly in this repository |
| Cycle Researcher | [Researcher](https://github.com/zhu-minjun/Researcher) |
| Data-to-Paper | [Data-to-Paper](https://github.com/Technion-Kishony-lab/data-to-paper) |
| Sakana v1 | [Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist) |
| Sakana v2 | [Sakana AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) |

These names are used as the canonical folder labels throughout the repository.

---

## Naming convention

The repository uses a single, publication-oriented naming convention to keep the pilot and full collections parallel and easy to navigate.

### Folder naming

- Folder names identify the **AI scientist system**.
- The same canonical system names are used in both the pilot and full collections.
- The full collection does **not** use a redundant `_full` suffix.

Examples:

- `AI-Scientist-generated-papers/pilot-paper-set/Cycle Researcher/`
- `AI-Scientist-generated-papers/full-proposal-set/Cycle Researcher/`

### File naming

Generated paper files use the convention:

```text
FAxxxx.pdf
```

Examples:

- `FA0001.pdf`
- `FA0042.pdf`
- `FA0080.pdf`

This design is intentional. The folder encodes the system identity, while the filename encodes the proposal/paper identifier. This avoids redundant naming while preserving a stable and concise artifact path.

Examples:

- `AI-Scientist-generated-papers/pilot-paper-set/Sakana v1/FA0001.pdf`
- `AI-Scientist-generated-papers/full-proposal-set/Data-to-Paper/FA0042.pdf`

---

## GRAIL reviewer

Reviewer-side analysis in this repository was organized through the **GRAIL** reviewer workflow. GRAIL is a multi-agent scientific review system designed to accept manuscript PDFs and return structured peer-review style outputs for comparative evaluation.

**Figure 1.** GRAIL reviewer interface used to organize and inspect reviewer-side evaluation outputs for the paper comparison workflow.

![GRAIL reviewer interface](./readme_assets/GRAIL_interface.png)

For the pilot-set evaluations archived here, GRAIL aggregates independent reviewer outputs from multiple frontier models, including:

- **GPT-5.4**
- **Gemini 3.1 Pro**
- **Claude Opus 4.6**

The structured score exports associated with these reviews include model-level scores as well as a synthesis-level assessment. The principal review dimensions archived in this repository are:

- **Quality** — evaluates whether the submission is technically sound and whether its claims are adequately supported by theory, experiments, or analysis.
- **Clarity** — evaluates whether the manuscript is clearly written, well organized, and sufficiently specified for expert readers to understand and reproduce the work.
- **Significance** — evaluates the likely impact of the contribution and whether the results are likely to matter to the relevant research community.
- **Originality** — evaluates the novelty of the ideas, methods, or findings and the extent to which the work is differentiated from prior literature.
- **Overall** — summarizes the reviewer’s holistic assessment of the submission after considering the individual dimensions together.
- **Confidence** — indicates the reviewer model’s confidence in its own assessment, given the evidence available in the manuscript.

In addition to numeric scores, GRAIL returns written review text summarizing strengths, weaknesses, questions for authors, limitations, and a multi-model consensus assessment. The repository also stores reviewer-side metadata such as submission identifiers, completion status, model availability or failure information, and synthesis outputs.

Reviewer-related artifacts in this repository are stored under:

- [`AI-Reviewer-outputs/`](./AI-Reviewer-outputs/)

---

## Repository organization

### `AI-Scientist-generated-papers/`

This directory contains the paper corpora and is divided into two collections.

#### `pilot-paper-set/`

This is the smaller pilot comparison collection.

| Folder | Description | Count |
|---|---|---:|
| [`Cycle Researcher/`](./AI-Scientist-generated-papers/pilot-paper-set/Cycle%20Researcher/) | Pilot-set papers generated by Cycle Researcher | 15 papers |
| [`Data-to-Paper/`](./AI-Scientist-generated-papers/pilot-paper-set/Data-to-Paper/) | Pilot-set papers generated by Data-to-Paper | 15 papers |
| [`FARS/`](./AI-Scientist-generated-papers/pilot-paper-set/FARS/) | Pilot-set FARS proposals and corresponding FARS papers | 15 proposals + 15 papers |
| [`Sakana v1/`](./AI-Scientist-generated-papers/pilot-paper-set/Sakana%20v1/) | Pilot-set papers generated by Sakana v1 | 15 papers |
| [`Sakana v2/`](./AI-Scientist-generated-papers/pilot-paper-set/Sakana%20v2/) | Pilot-set papers generated by Sakana v2 | 15 papers |

#### `full-proposal-set/`

This is the larger proposal-linked collection.

| Folder | Description | Count |
|---|---|---:|
| [`Cycle Researcher/`](./AI-Scientist-generated-papers/full-proposal-set/Cycle%20Researcher/) | Full-collection papers generated by Cycle Researcher | 167 papers |
| [`Data-to-Paper/`](./AI-Scientist-generated-papers/full-proposal-set/Data-to-Paper/) | Full-collection papers generated by Data-to-Paper | 33 papers |
| [`FARS/`](./AI-Scientist-generated-papers/full-proposal-set/FARS/) | FARS proposal corpus and associated FARS papers for the full collection | 189 proposals + 167 papers |
| [`Sakana v1/`](./AI-Scientist-generated-papers/full-proposal-set/Sakana%20v1/) | Full-collection papers generated by Sakana v1 | 167 papers |
| [`Sakana v2/`](./AI-Scientist-generated-papers/full-proposal-set/Sakana%20v2/) | Full-collection papers generated by Sakana v2 | 167 papers |

The smaller Data-to-Paper count is expected. That pipeline depends more directly on dataset availability than the other systems, so only a subset of FARS proposals could be converted into paper outputs through that workflow.

### `AI-Reviewer-outputs/`

This directory contains reviewer-side outputs and evaluation artifacts.

| Folder | Description |
|---|---|
| [`Results_pilot_set/`](./AI-Reviewer-outputs/Results_pilot_set/) | Pilot-set reviewer outputs, score tables, JSON exports, summary analyses, and comparison figures |
| [`Results_full_set/`](./AI-Reviewer-outputs/Results_full_set/) | Reserved destination for future reviewer outputs and analyses associated with the full collection |

---

## Pilot-set analysis artifacts

Pilot-set evaluation outputs are stored in:

- [`AI-Reviewer-outputs/Results_pilot_set/`](./AI-Reviewer-outputs/Results_pilot_set/)

Key files include:

| File | Description |
|---|---|
| [`results_scores.csv`](./AI-Reviewer-outputs/Results_pilot_set/results_scores.csv) | Structured per-paper reviewer score table |
| [`results.jsonl`](./AI-Reviewer-outputs/Results_pilot_set/results.jsonl) | Long-form consensus review outputs |
| [`submission_ids.csv`](./AI-Reviewer-outputs/Results_pilot_set/submission_ids.csv) | Submission and score tracking table |
| [`analysis_ai_scientists_comparison/`](./AI-Reviewer-outputs/Results_pilot_set/analysis_ai_scientists_comparison/) | Summary statistics, comparison plots, tables, and downstream analysis artifacts |

At present, the populated reviewer outputs are specific to the pilot collection. The full-collection reviewer folder has been created but is not yet populated with corresponding analysis files.

---

## Quick navigation

### Pilot paper collection

- [`AI-Scientist-generated-papers/pilot-paper-set/`](./AI-Scientist-generated-papers/pilot-paper-set/)

### Full paper collection

- [`AI-Scientist-generated-papers/full-proposal-set/`](./AI-Scientist-generated-papers/full-proposal-set/)

### Full FARS source material

- [`AI-Scientist-generated-papers/full-proposal-set/FARS/fars_proposals/`](./AI-Scientist-generated-papers/full-proposal-set/FARS/fars_proposals/)
- [`AI-Scientist-generated-papers/full-proposal-set/FARS/fars_papers/`](./AI-Scientist-generated-papers/full-proposal-set/FARS/fars_papers/)

### Reviewer outputs

- [`AI-Reviewer-outputs/Results_pilot_set/`](./AI-Reviewer-outputs/Results_pilot_set/)
- [`AI-Reviewer-outputs/Results_full_set/`](./AI-Reviewer-outputs/Results_full_set/)

---

## References

- **FARS**: [Analemma — FARS](https://analemma.ai/fars/)
- **FARS overview**: [Introducing FARS](https://analemma.ai/blog/introducing-fars/)
- **Sakana v1**: [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)
- **Sakana v2**: [The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](https://arxiv.org/abs/2504.08066)
- **Cycle Researcher**: [Researcher: Iterative Research Idea Generation over Scientific Literature with Large Language Models](https://arxiv.org/abs/2411.00816)
- **Data-to-Paper**: [Data-to-Paper repository](https://github.com/Technion-Kishony-lab/data-to-paper)

---

## Summary

This repository provides a standardized, publication-facing organization of FARS-derived proposal documents, AI-generated paper outputs, and reviewer-side evaluation artifacts.

- The directory structure is parallel across pilot and full collections.
- Generated paper filenames follow a single stable convention: `FAxxxx.pdf`.
- System identities are encoded at the folder level rather than repeated inside filenames.
- Reviewer outputs are separated cleanly from the paper corpora.

The result is a repository structure intended to be legible to readers, collaborators, and reviewers associated with formal conference submissions.
