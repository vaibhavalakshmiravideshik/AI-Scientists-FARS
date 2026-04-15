# AI Scientists Comparison Analysis

## Overview
This analysis compares AI reviewer scores across 5 AI Scientist systems:
- Cycle Researcher
- Data-to-Paper
- FARS
- Sakana v1
- Sakana v2

Total papers analyzed: 75 (15 papers per system)

## Data Source
- `/workspace/AI-Scientists-FARS/AI-Scientist-generated-papers/Results/results_scores.csv`

## Overall Scores Summary

| AI Scientist | n | GPT-5.4 Mean | Gemini Mean | Claude Mean | Synthesis Mean |
|---|---|---|---|---|---|
| Cycle Researcher | 15 | 2.00 | 1.00 | 1.00 | 1.00 |
| Data-to-Paper | 15 | 1.87 | 1.07 | 1.00 | 1.07 |
| FARS | 15 | 2.14 | 2.47 | 2.47 | 2.27 |
| Sakana v1 | 15 | 1.87 | 1.00 | 1.00 | 1.00 |
| Sakana v2 | 15 | 1.67 | 1.00 | 1.00 | 1.00 |

## Key Findings

### Average Overall Scores by Model:

- **gpt 5.4**:
  - Highest: FARS (2.14)
  - Lowest: Sakana v2 (1.67)
  - Range: 0.48

- **gemini 3.1 pro**:
  - Highest: FARS (2.47)
  - Lowest: Cycle Researcher (1.00)
  - Range: 1.47

- **claude opus 4.6**:
  - Highest: FARS (2.47)
  - Lowest: Cycle Researcher (1.00)
  - Range: 1.47

- **synthesis**:
  - Highest: FARS (2.27)
  - Lowest: Cycle Researcher (1.00)
  - Range: 1.27

## Statistical Tests

### Pairwise Comparisons (GPT-5.4 Overall)

Mann-Whitney U tests comparing all pairs of AI Scientists.
Holm correction applied for multiple comparisons.

| Group 1 | Group 2 | Mean Diff | Raw p | Corrected p | Cliff's δ |
|---|---|---|---|---|---|
| Cycle Researcher | Data-to-Paper | +0.13 | 0.1641 | 0.8975 | +0.133 |
| Cycle Researcher | FARS | -0.14 | 0.1496 | 0.8975 | -0.143 |
| Cycle Researcher | Sakana v1 | +0.13 | 0.1641 | 0.8975 | +0.133 |
| Cycle Researcher | Sakana v2 | +0.33 | 0.0175 | 0.1574 | +0.333 |
| Data-to-Paper | FARS | -0.28 | 0.0536 | 0.4289 | -0.257 |
| Data-to-Paper | Sakana v1 | +0.00 | 1.0000 | 1.0000 | +0.000 |
| Data-to-Paper | Sakana v2 | +0.20 | 0.2132 | 0.8975 | +0.200 |
| FARS | Sakana v1 | +0.28 | 0.0536 | 0.4289 | +0.257 |
| FARS | Sakana v2 | +0.48 | 0.0094 | 0.0936 | +0.429 |
| Sakana v1 | Sakana v2 | +0.20 | 0.2132 | 0.8975 | +0.200 |

### Interpretation

- **Mean Diff**: Positive means Group 1 scores higher than Group 2
- **Cliff's δ**: Effect size (-1 to +1). Positive = Group 1 tends to score higher
  - |δ| < 0.147: negligible
  - |δ| < 0.33: small
  - |δ| < 0.474: medium
  - |δ| ≥ 0.474: large

## Files Generated

- `/workspace/AI-Scientists-FARS/AI-Scientist-generated-papers/Results/analysis_ai_scientists_comparison/descriptive_stats_by_ai_scientist.csv` - Full descriptive statistics
- `/workspace/AI-Scientists-FARS/AI-Scientist-generated-papers/Results/analysis_ai_scientists_comparison/overall_scores_summary.csv` - Summary of overall scores
- `/workspace/AI-Scientists-FARS/AI-Scientist-generated-papers/Results/analysis_ai_scientists_comparison/pairwise_comparisons_gpt54.csv` - Statistical test results
- `/workspace/AI-Scientists-FARS/AI-Scientist-generated-papers/Results/analysis_ai_scientists_comparison/ai_scientists_comparison_plots.png` - Visualization