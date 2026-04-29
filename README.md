# AI-Scientists-FARS

## Overview

This repository is a curated research artifact collection for the comparative study of AI-generated scientific papers derived from FARS proposal documents. It brings together the source proposal corpus, paper outputs produced by multiple AI scientist systems, and reviewer-side evaluation artifacts used in downstream comparative analysis.

The repository is intended to support publication-facing documentation and artifact inspection for the pilot and full experimental collections associated with this project.

At a high level, the workflow represented here is:

1. FARS proposal documents are constructed from FARS repositories.
2. Multiple AI scientist systems generate papers from those proposal documents.
3. Reviewer-side outputs are collected and organized for comparative evaluation.
4. Pilot-set and full-set artifacts are archived in a consistent, navigable structure.

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
| FARS | FARS-based proposal-to-paper source set represented directly in this repository |
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

Reviewer-side analysis in this repository was organized through the **GRAIL** reviewer workflow. GRAIL was used to manage paper-level review runs, collect structured evaluator outputs, and produce consensus-style review text for comparative assessment of the generated manuscripts.

For the pilot-set evaluations archived here, GRAIL aggregates outputs from multiple reviewer models, including:

- **GPT-5.4**
- **Gemini 3.1 Pro**
- **Claude Opus 4.6**

The structured score exports associated with these reviews include model-level and synthesis-level assessments. As reflected in the archived score tables, the principal scoring dimensions include:

- **overall**
- **quality**
- **clarity**
- **significance**
- **originality**
- **confidence**

The repository also stores reviewer-side metadata such as submission identifiers, completion status, model availability/failure information, consensus review text, and synthesis outputs.

Reviewer-related artifacts in this repository are stored under:

- [`AI-Reviewer-outputs/`](./AI-Reviewer-outputs/)

**Figure.** GRAIL reviewer interface used to organize and inspect reviewer-side evaluation outputs for the paper comparison workflow.

![GRAIL reviewer interface](./grail_reviewer-modified-Picsart-AiImageEnhancer.png)

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

## Corpus statistics

The repository contains both a pilot comparison set and a larger full collection. Counts differ across systems because not every proposal was propagated through every downstream generation pipeline.

| Artifact category | Count |
|---|---:|
| Pilot-set FARS proposals | 15 |
| Pilot-set papers per AI scientist system | 15 each |
| Full-collection FARS proposals currently archived in this repository | 189 |
| Full-collection FARS papers | 167 |
| Full-collection Cycle Researcher papers | 167 |
| Full-collection Sakana v1 papers | 167 |
| Full-collection Sakana v2 papers | 167 |
| Full-collection Data-to-Paper papers | 33 |

The primary reasons for count differences are:

- the pilot collection is a deliberately smaller comparison subset;
- the full collection is a broader proposal-linked corpus;
- not every FARS proposal propagated into every downstream artifact type;
- Data-to-Paper has a smaller full-collection footprint because dataset availability is a practical constraint for that pipeline.

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

## Summary

This repository provides a standardized, publication-facing organization of FARS-derived proposal documents, AI-generated paper outputs, and reviewer-side evaluation artifacts.

- The directory structure is parallel across pilot and full collections.
- Generated paper filenames follow a single stable convention: `FAxxxx.pdf`.
- System identities are encoded at the folder level rather than repeated inside filenames.
- Reviewer outputs are separated cleanly from the paper corpora.

The result is a repository structure intended to be legible to readers, collaborators, and reviewers associated with formal conference submissions.
