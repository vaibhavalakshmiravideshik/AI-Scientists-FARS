# untitled

# INLP-Projected Linear Probing for Cross-Subject EEG Foundation Model Transfer

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR (or similar top AI conferences)

## Introduction

### Context and Motivation

Electroencephalography (EEG) is widely used for non-invasive brain-computer interfaces (BCIs), but EEG decoders often generalize poorly to a new subject due to large inter-subject variability. Recently, **EEG foundation models (EEG-FMs)** have been proposed: neural encoders pre-trained on large heterogeneous EEG corpora and adapted to downstream BCI tasks.

A practical deployment hope is **frozen-encoder transfer**: compute embeddings with a pretrained encoder and train a small classifier head, which is cheaper and easier to reuse than full fine-tuning. However, recent EEG-FM benchmarks suggest this regime underperforms end-to-end fine-tuning, limiting the usefulness of EEG-FMs as reusable feature extractors.

### The Problem

The benchmark in **[EEG Foundation Models: Progresses, Benchmarking, and Open Problems](./references/EEG-Foundation-Models-Progresses-Benchmarking-and-Open-Problems/meta/meta_info.txt)** evaluates 12 EEG-FMs across 13 datasets under a **leave-one-subject-out (LOSO)** protocol (train on multiple subjects, test on a held-out subject), and compares full fine-tuning to **linear probing** (freeze the encoder; train only a linear head). On **BNCI2014001** (BCI Competition IV-2a motor imagery; 9 subjects; 4 classes; 22 channels), **CBraMod (a 4M-parameter EEG-FM)** achieves **53.03±0.22%** LOSO accuracy with full fine-tuning but only **41.45±0.50%** with linear probing (Table XV in `sections/B. Comparison of Different Fine-tuning Ratios.md`).

This large gap motivates a concrete question: can we improve frozen-encoder LOSO transfer without updating the encoder?

### Key Insight and Hypothesis

A plausible contributor to weak frozen transfer is **subject-identity leakage**: frozen embeddings may encode subject-specific directions that a linear head exploits on training subjects but that do not transfer to a new subject.

**Key insight**: BrainAlign reports that in CBraMod-derived representations, subject identity can be reduced to near-chance by a post-hoc procedure (CCA followed by INLP) with minimal change to their main task metrics (**[BrainAlign](./references/BrainAlign-Leveraging-EEG-Foundation-Models-for-Symmetric-Interpretable-Alignment-with-Visual-Representations/meta/meta_info.txt)**, Table 4). This suggests subject identity is (at least partly) contained in a low-rank linear subspace.

**Hypothesis**: Removing the linearly decodable subject-identity subspace from frozen EEG-FM embeddings using **iterative nullspace projection (INLP)** will improve LOSO linear-probe accuracy compared to a strong frozen baseline and a rank-matched PCA control.

The outcome is genuinely uncertain: (i) identity may be entangled with label-relevant signal, so removal could hurt; (ii) frozen embeddings may simply not be linearly separable for the task, so identity removal may not help.

---

## Proposed Approach

### Overview

For each LOSO fold:
1. (Optional but default) apply **Euclidean Alignment (EA)** to each subject’s raw EEG as a strong standard preprocessing baseline.
2. Extract frozen embeddings from an EEG-FM encoder (primary: CBraMod).
3. Fit INLP on training-subject embeddings to remove subject identity.
4. Train the downstream linear head on projected embeddings; evaluate on the held-out subject.

### Method Details

**Euclidean Alignment (EA)**: EA (He & Wu 2020) aligns per-subject trial covariances to a common reference. The EEG-FM benchmark reports EA improves generalization for most models on BNCI2014001 (Fig. 11). We include EA to ensure any gains from INLP are beyond a widely used, strong transfer baseline.

**INLP (iterative nullspace projection)**: INLP (Ravfogel et al., 2020) iteratively trains a linear classifier to predict a protected attribute (here: subject ID) and projects features onto the classifier’s nullspace until the attribute is no longer linearly predictable.

Concretely, given training-subject embeddings \(Z\in\mathbb{R}^{N\times d}\) and subject labels \(s\in\{1,\dots,S\}\), repeat up to 10 times:
- fit a multinomial linear classifier \(g_t\) to predict \(s\) from \(Z_t\),
- compute an orthogonal projection \(P_t\) onto the nullspace of \(g_t\)’s weight matrix,
- update \(Z_{t+1}=Z_t P_t\).

Stop early when subject-ID accuracy on a held-out split of training-subject data is close to chance (<=1.25×1/S). The final projection is \(P=\prod_t P_t\). Importantly, **INLP is trained only on training subjects**; the held-out subject is never used to fit \(P\).

**Control: PCA-k removal**: To test whether improvements are just generic dimensionality reduction, we remove the top \(k\) principal components where \(k\) matches the rank removed by INLP.

### Key Innovations

- Apply INLP as a practical post-hoc feature edit for EEG-FM transfer (not only as analysis).
- Use a rank-matched PCA control to distinguish targeted identity removal from generic regularization.

---

## Related Work

### Field Overview

EEG transfer learning often uses alignment methods (Euclidean or Riemannian) to reduce subject/domain shift, but these do not explicitly remove all linearly decodable subject identity from learned embeddings. Separately, representation editing methods such as INLP remove a targeted attribute by construction, providing a direct intervention to test whether an attribute (subject identity) causally harms transfer.

Recent EEG-FM benchmarks show that pretraining does not guarantee transferable frozen features, motivating methods that improve frozen transfer without full fine-tuning.

### Related Papers

- **[EEG-FM Benchmark](./references/EEG-Foundation-Models-Progresses-Benchmarking-and-Open-Problems/meta/meta_info.txt)**: Quantifies full fine-tuning vs linear probing gaps across many EEG-FMs.
- **[Are EEG Foundation Models Worth It?](./references/Are-EEG-Foundation-Models-Worth-It-Comparative-Evaluation-with-Traditional-Decoders-in-Diverse-BCI-Tasks/meta/meta_info.txt)**: Shows frozen transfer can be weak and sensitive to head design.
- **[CBraMod](./references/CBraMod-A-Criss-Cross-Brain-Foundation-Model-for-EEG-Decoding/meta/meta_info.txt)**: EEG-FM architecture used as the primary frozen encoder.
- **[BrainAlign](./references/BrainAlign-Leveraging-EEG-Foundation-Models-for-Symmetric-Interpretable-Alignment-with-Visual-Representations/meta/meta_info.txt)**: Uses CCA+INLP to separate subject-dependent vs subject-agnostic components (analysis setting).
- **[Revisiting Euclidean Alignment](./references/Revisiting-Euclidean-Alignment-for-Transfer-Learning-in-EEG-Based-Brain-Computer-Interfaces/meta/meta_info.txt)**: Survey of EA and related alignment methods for EEG transfer.
- **[EA (He & Wu 2020)](https://ieeexplore.ieee.org/document/9094063)**: Introduces Euclidean Alignment for BCI transfer.
- **[Riemannian Alignment](https://ieeexplore.ieee.org/document/8335016)**: Aligns EEG covariance structure on SPD manifolds for domain/subject transfer.
- **[CORAL](https://arxiv.org/abs/1607.01719)**: Feature covariance alignment for unsupervised domain adaptation.
- **[INLP / Null it out](https://aclanthology.org/2020.acl-main.647/)**: Iteratively removes linearly decodable protected attributes via nullspace projection.
- **[Bolukbasi et al. 2016](https://arxiv.org/abs/1607.06520)**: Early projection-based debiasing motivating targeted subspace removal.
- **[Gonen & Goldberg 2019](https://arxiv.org/abs/2004.03644)**: Shows bias can persist after single-direction projection, motivating iterative removal.
- **[DANN](https://arxiv.org/abs/1505.07818)**: Adversarially learns domain-invariant representations (training-based alternative).
- **[Domain Separation Networks](https://arxiv.org/abs/1608.06019)**: Separates shared vs private domain subspaces (conceptually related to identity vs task subspaces).
- **[EEGNet](https://arxiv.org/abs/1611.08024)**: Common compact CNN baseline for EEG decoding.
- **[Deep ConvNet / Shallow ConvNet](https://arxiv.org/abs/1703.05051)**: Standard ConvNet EEG decoders used widely for motor imagery.
- **[BENDR](https://arxiv.org/abs/2101.10819)**: Self-supervised EEG representation learning baseline.
- **[BIOT](https://arxiv.org/abs/2307.13786)**: Biosignal transformer baseline compared in EEG-FM benchmarks.
- **[LaBraM](https://arxiv.org/abs/2405.18765)**: EEG foundation model baseline.
- **[EEGPT / BrainGPT](https://arxiv.org/abs/2410.19779)**: Autoregressive pretraining for generalist EEG representations.
- **[MIRepNet](https://arxiv.org/abs/2507.01038)**: MI-focused pretraining showing smaller but persistent linear-probing gaps.

### Taxonomy

| Family | Core idea | Representative papers | Typical evaluation | Key limitation |
|---|---|---|---|---|
| EEG alignment | Align subject distributions (input/feature) | EA, Riemannian alignment, CORAL | LOSO accuracy / kappa | Not targeted to remove all identity leakage |
| EEG foundation models | Large-scale EEG pretraining | CBraMod, LaBraM, BIOT, EEGPT | Multi-dataset LOSO + few-shot | Frozen transfer often weak |
| Attribute removal | Remove targeted attributes from embeddings | INLP, projection debiasing | Leakage vs task performance | May remove useful signal if entangled |

### Closest Prior Work

- **BrainAlign (CCA+INLP)** shows subject-ID can be reduced in a CCA space with minimal task change, but does not test improving LOSO classification on standard BCI datasets.
- **EA / alignment methods** improve transfer but do not explicitly remove residual linearly decodable subject identity in learned embeddings.
- **Adversarial subject-invariant learning (DANN/DSN-style)** requires training and is not a post-hoc edit of frozen embeddings.

**Novelty Kill Search Summary:** Searched for "INLP nullspace projection EEG", "iterative nullspace projection BCI", "Ravfogel INLP EEG subject identity", and "subject-invariant EEG INLP", and checked this repo’s finalized proposals and all agents’ draft `proposal.md` files for INLP/nullspace/Ravfogel keywords. No prior work applying INLP as a method to improve EEG-FM linear probing was found as of 2026-02-26. Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| BrainAlign (CCA+INLP) | INLP used for analysis in an EEG-vision alignment task | Not evaluated on BCI LOSO decoding | Apply INLP directly to EEG-FM embeddings for LOSO MI | Tests whether identity removal improves frozen transfer |
| EA | Aligns EEG statistics across subjects | May leave identity directions in learned embeddings | Add embedding-space identity removal | EA reduces input shift; INLP removes residual linear identity |
| PCA removal | Removes high-variance directions | Not targeted to identity | Use INLP + rank-matched PCA control | Should beat PCA when identity is not just top-variance |

---

## Experiments

### Experimental Setup

**Primary benchmark:** BNCI2014001 (BCI Competition IV-2a motor imagery; 9 subjects; 4-class classification; 22-channel EEG).

**Protocol:** LOSO (9 folds). For each fold, learn INLP projection using only the 8 training subjects, then apply it to both train and test embeddings.

**Baseline ladder (adapted):** Prompting and inference-time sampling baselines are not applicable (deterministic supervised classifiers). Instead, we ladder from strong classical/transfer baselines to the proposed embedding edit:
- Frozen EEG-FM baseline: EA + CBraMod embeddings + linear head.
- Rank-matched control: EA + PCA-k + linear head.
- **Ours**: EA + INLP + linear head.
- Contextual upper bound: full fine-tuning numbers from the benchmark paper.

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| CBraMod | 4.0M params | https://huggingface.co/weighting666/CBraMod | Primary frozen encoder |

**Data:** BNCI2014001 via MOABB: https://github.com/NeuroTechX/moabb

**Evaluation code:** Prefer reusing the published pipeline: https://github.com/Dingkun0817/EEG-FM-Benchmark

**Resource Estimate:** <=20 GPU-hours (cache embeddings once; linear probes/heads are cheap).

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| BNCI2014001 | 4-class motor imagery EEG classification | Accuracy (%; higher is better) | LOSO | MOABB | EEG-FM-Benchmark / MOABB |

### Main Results

| Method | Base Model | Benchmark | Accuracy (mean±std) | Source | Notes |
|---|---|---|---:|---|---|
| Linear probing (reported) | CBraMod | BNCI2014001 LOSO | 41.45±0.50 | [Liu et al. 2026](./references/EEG-Foundation-Models-Progresses-Benchmarking-and-Open-Problems/meta/meta_info.txt) | Table XV in raw sections |
| Full fine-tuning (reported) | CBraMod | BNCI2014001 LOSO | 53.03±0.22 | [Liu et al. 2026](./references/EEG-Foundation-Models-Progresses-Benchmarking-and-Open-Problems/meta/meta_info.txt) | Table XV in raw sections |
| EA + linear head | CBraMod | BNCI2014001 LOSO | **TBD** | - | Verification run |
| EA + PCA-k + linear head | CBraMod | BNCI2014001 LOSO | **TBD** | - | k matched to INLP rank |
| **EA + INLP + linear head (Ours)** | CBraMod | BNCI2014001 LOSO | **TBD** | - | Verification run |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| INLP early stopping | Stop before reaching chance leakage | If identity is partially label-relevant, early stop may win |
| Mean-subspace removal | Project out between-subject mean subspace (BrainAlign baseline) | Helps if identity is mostly mean shift; otherwise weaker than INLP |

### Experimental Rigor

- **Seeds**: `seeds=[42, 123, 456]`.
- **Leakage diagnostic**: report subject-ID linear-probe accuracy pre/post INLP on held-out training-subject split (chance = 1/S).
- **Over-removal check**: report within-subject accuracy change after applying the learned projection.
- **Sanity check**: randomize subject labels before INLP; should not improve task accuracy.

---

## Success Criteria

**Hypothesis**: INLP removal of subject identity from frozen EEG-FM embeddings improves cross-subject linear probing by removing subject-specific nuisance directions that do not generalize.

**Decision Rule**:
- **Proceed** if on BNCI2014001 LOSO, INLP improves accuracy over EA baseline by a margin outside EA’s std (target: >=+2 pp) and exceeds PCA-k by >=+1 pp (3 seeds).
- **Pivot** if INLP ≈ PCA-k > EA: treat gains as generic dimension reduction and simplify to PCA.
- **Refute** if INLP <= max(EA, PCA-k), or if within-subject accuracy drops substantially after INLP.

---

## Impact Statement

If successful, this provides a cheap, fully automated way to improve frozen-encoder EEG-FM transfer, reducing reliance on expensive end-to-end fine-tuning and improving practical reuse of pretrained EEG encoders for cross-subject BCI decoding.

---

## References

(See **Related Papers** above for additional citations and links.)

- [EEG-FM Benchmark](./references/EEG-Foundation-Models-Progresses-Benchmarking-and-Open-Problems/meta/meta_info.txt)
- [BrainAlign](./references/BrainAlign-Leveraging-EEG-Foundation-Models-for-Symmetric-Interpretable-Alignment-with-Visual-Representations/meta/meta_info.txt)
- [CBraMod](./references/CBraMod-A-Criss-Cross-Brain-Foundation-Model-for-EEG-Decoding/meta/meta_info.txt)
- [INLP / Null it out](https://aclanthology.org/2020.acl-main.647/)
- [EA (He & Wu 2020)](https://ieeexplore.ieee.org/document/9094063)
