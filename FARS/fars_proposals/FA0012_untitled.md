# untitled

# Delta-Map Belief Updates for Stable Spatial Revision in Theory of Space

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Core constraint**: Fully automated evaluation (no human-in-the-loop)
- **Infrastructure constraint**: No interactive simulator required; use the released **offline** Theory-of-Space dataset (pre-rendered images + metadata)
- **Compute constraint**: ≤768 A100 GPU-hours (expected **0 GPU-hours**; API-only)

## Introduction

### Context and Motivation

Embodied agents (robots, navigation assistants, and computer-use systems) must act under partial observability: the agent cannot see the whole world state at once, so it must build and maintain an internal **spatial belief** (a map-like hypothesis about object locations and orientations) from a sequence of observations. Recent multimodal foundation models (vision-language models, VLMs) can answer many spatial questions when given the relevant views, but it is less clear whether they can **construct, maintain, and revise** a coherent spatial belief over time.

**Theory of Space (ToS)** is a recent benchmark that isolates this challenge by asking models to explore a multi-room environment, then evaluating (i) how well their induced belief supports downstream spatial queries and (ii) how accurate their **explicitly probed cognitive map** is at each step. ToS further tests belief revision with a **false-belief paradigm**: after exploration, the environment changes (k=4 objects moved or rotated), and the agent must re-explore and update its belief. Empirically, ToS finds severe failures in vision-based settings: even strong models show low final map correctness and large belief inertia under changes (e.g., in vision-world false-belief, GPT-5.2 has positional inertia 68.9% and orientation inertia 34.7%—**lower is better**; Table 7 in the ToS paper).

### The Problem

ToS attributes these failures to perception bottlenecks (especially orientation) and to **belief instability**, where previously correct information is overwritten over time. However, ToS’s cognitive-map probing also creates a practical systems question: the benchmark repeatedly asks an LLM/VLM to output a *complete* global JSON cognitive map from a growing interaction history.

In many LLM systems, repeatedly regenerating a large structured state is error-prone: the model may inadvertently mutate fields unrelated to the current observation (“copy noise”), which can appear as belief drift. In addition, when observations conflict with a prior belief (false belief), the model may not reliably apply a consistent conflict resolution rule, leading to **belief inertia**.

If a large portion of ToS “belief drift/inertia” comes from the *interface* by which the belief is externalized (full regeneration from long context), then a simple state-management intervention—treating the map as an external state and updating it incrementally—could substantially improve ToS map correctness and false-belief revision without changing the underlying model.

### Key Insight and Hypothesis

**Key insight**: For structured beliefs like ToS’s cognitive map, the model should not need to rewrite the entire belief every step. Most steps only provide evidence about a small subset of objects (those currently visible). If we provide the previous belief explicitly and require the model to update only the parts supported by current evidence, we may reduce accidental overwrites and make belief revision under conflicts more reliable.

**Hypothesis**: On Theory-of-Space (vision world), providing the previous cognitive map as explicit context and enforcing a “preserve unless evidenced / overwrite on contradiction” update rule will (i) increase final cognitive-map correctness and (ii) reduce false-belief inertia. A further hypothesis is that asking for **delta updates** (a small JSON object describing only changed entries) reduces transcription errors beyond the same rule with full-map regeneration.

Why this could be wrong:
- Drift/inertia may be dominated by perception errors (wrong object identity/orientation) rather than map-regeneration errors.
- A “preserve unless evidenced” rule could increase apparent stability while hurting correction of earlier mistakes, potentially worsening final correctness.

---

## Proposed Approach

### Overview

We propose a **belief-update interface** for Theory-of-Space cognitive maps that treats the global map as an external state `M_{t}` and updates it at each step using the latest observation `O_t`.

We test three conditions that differ only in how the model is prompted to update the map:

- **A (Scratch regeneration baseline)**: original ToS probing style—at each step, the model outputs a full global map from the interaction history.
- **B (Rule-based full regeneration)**: the model is given the previous map `M_{t-1}` and the new observation `O_t`, and must output a full updated map `M_t`, with explicit rules to preserve unchanged entries and overwrite contradicted entries.
- **C (Delta-map updates)**: the model is given `M_{t-1}` and `O_t`, and must output a compact JSON “delta” describing only the objects whose states should change; the evaluator applies the delta to form `M_t`.

### Method Details

**Belief state (global map) schema.** We use the ToS global cognitive map format:

```json
{
  "agent": {"position": [x,y], "facing": "north|south|east|west"},
  "obj_name": {"position": [x,y], "facing": "north|south|east|west"},
  "gate_name": {"position": [x,y], "facing": "north|south|east|west"}
}
```

(We follow ToS’s rule: include only observed objects; facing is required only for entities with a facing direction.)

**Condition A (scratch regeneration).** For each timestep `t`, provide the ToS observation history up to `t` (the same history ToS uses for probing) and ask the model to output the full global map.

**Condition B (full regeneration with explicit update rules).** For each timestep `t`, provide:
1) previous map `M_{t-1}` (verbatim JSON),
2) current observation `O_t` (image + text observation in ToS format),
3) update rules:
   - Preserve: copy all entries from `M_{t-1}` unchanged unless the current observation provides evidence they should change.
   - Evidence restriction: only update objects that are visible in `O_t` (or newly observed).
   - Conflict resolution: if `O_t` contradicts an entry for a visible object, overwrite that object’s state to be consistent with `O_t`.

Then ask for the full updated `M_t`.

**Condition C (delta-map updates).** Same inputs and rules as B, but the model outputs a compact delta JSON:

```json
{
  "updates": {
    "obj_name": {"position": [x,y], "facing": "east"},
    "obj2": {"position": [x,y]}
  }
}
```

The evaluator computes `M_t = Apply(M_{t-1}, delta)` by replacing only the keys in `updates`.

**Why delta might help beyond B.** Even if B has the right rule, generating a large JSON repeatedly can still introduce transcription errors (missing keys, accidental edits). Delta outputs reduce output length and isolate the model’s generation to the minimal set of updates.

**False-belief revision (dynamic update).** To test belief inertia, we use ToS’s released `falsebelief_exp.json` per scene (k=4 changed objects) and the corresponding post-change images (`*_fbexp.png`). We run the same three conditions during re-exploration (using a fixed proxy trajectory; see Experiments) and measure:
- identification F1 of changed objects,
- inertia metrics as defined in ToS (positional alignment + orientation inertia).

### Key Innovations

- A **verification-first causal test** for whether ToS belief drift/inertia is partly an artifact of full-map regeneration from long context, by comparing scratch regeneration vs explicit-map update.
- A simple, implementation-friendly **delta-map update interface** for cognitive-map probing that is model-agnostic (prompt-only; no fine-tuning).
- A focused evaluation on **dynamic belief revision** (false belief) where improvements cannot be explained by “never changing old entries,” because the task requires overwriting obsolete beliefs.

---

## Related Work

### Field Overview

**Spatial reasoning benchmarks for VLMs and embodied agents.** Many benchmarks test spatial reasoning from static images or short clips, but fewer isolate long-horizon belief construction under partial observability. ToS contributes by probing explicit cognitive maps and measuring belief revision under environment shifts.

**Explicit spatial representations (cognitive maps, scene graphs).** A common strategy to improve spatial reasoning is to convert perception into structured intermediate representations—scene graphs, maps, or coordinate lists—then condition a VLM/LLM on this representation. This can improve spatial queries but introduces new failure modes in representation generation, maintenance, and revision.

**Incremental structured-state updates.** Outside spatial reasoning, recent work shows that LLMs often struggle to repeatedly regenerate large structured objects, motivating incremental update formats (diffs, patches, constrained edits). This proposal applies that insight to spatial belief probing.

### Related Papers

- **[Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?](./references/THEORY%20OF%20SPACE%20CAN%20FOUNDATION%20MODELS%20CON-STRUCT%20SPATIAL%20BELIEFS%20THROUGH%20ACTIVE%20EXPLORATION/meta/meta_info.txt)**: Introduces ToS, cognitive-map probing, and the belief inertia metric we target.
- **[Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](https://arxiv.org/abs/2412.14171)**: Studies spatial memory and recall in MLLMs; motivates explicit belief representations.
- **[SpatialVLM](https://arxiv.org/abs/2401.12168)**: Enhances VLMs with spatial reasoning capabilities; representative of spatially focused VLM adaptation.
- **[SpatialRGPT](https://arxiv.org/abs/2406.01584)**: Grounded spatial reasoning with region-centric prompting; highlights grounding limitations.
- **[EmbodiedBench](https://arxiv.org/abs/2502.09560)**: Benchmarks embodied decision making for multimodal LLM agents.
- **[Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making](https://arxiv.org/abs/2410.07166)**: Evaluates LLMs in embodied settings; complements ToS’s belief-centric focus.
- **[Embodied Question Answering](https://arxiv.org/abs/1711.11543)**: Classic task-driven exploration benchmark; contrasts with ToS’s task-agnostic exploration.
- **[IQA: Visual Question Answering in Interactive Environments](https://arxiv.org/abs/1712.03316)**: Early interactive VQA benchmark; related environment-query framing.
- **[ALFRED](https://arxiv.org/abs/1912.01734)**: Instruction following in interactive environments; emphasizes long-horizon interaction.
- **[TEACh](https://arxiv.org/abs/2110.00534)**: Task-driven embodied agents with dialogue; highlights language grounding needs.
- **[EXCALIBUR](https://arxiv.org/abs/2303.07342)**: Encourages embodied exploration; relates to ToS’s exploration efficiency axis.
- **[Reverie](https://arxiv.org/abs/1904.10151)**: Remote embodied referring expression; spatial grounding under partial observability.
- **[Seeing from Another Perspective: Evaluating Multi-view Understanding in MLLMs](https://arxiv.org/abs/2504.15280)**: Multi-view spatial evaluation; complements ToS with passive multi-view settings.
- **[MMSI-Bench](https://arxiv.org/abs/2505.23764)**: Multi-image spatial intelligence benchmark; broader coverage of spatial skills.
- **[3DSRBench](https://arxiv.org/abs/2412.07825)**: 3D spatial reasoning benchmark; focuses on 3D geometry reasoning.
- **[InternSpatial](https://arxiv.org/abs/2506.18385)**: Large dataset for spatial reasoning in VLMs; data-centric angle.
- **[What’s “Up” with Vision-Language Models?](https://arxiv.org/abs/2310.19785)**: Diagnoses spatial failure modes of VLMs; motivates targeted interventions.
- **[SpatialTree](https://arxiv.org/abs/2512.20617)**: Capability-centric taxonomy of spatial abilities; situates ToS at agentic competence.
- **[Ego3D-VLM / Ego3D-Bench](https://arxiv.org/abs/2509.06266)**: Uses textual/JSON cognitive maps to improve ego-centric multi-view spatial reasoning.
- **[3DThinker](https://arxiv.org/abs/2510.18632)**: Distills 3D latent representations to improve spatial reasoning; a model-internal alternative to explicit maps.
- **[JSON Whisperer: Efficient JSON Editing with LLMs](https://arxiv.org/abs/2510.04717)**: Uses diff/patch-style edits for structured outputs; provides tools and motivation for incremental updates.
- **[GraphPad: Inference-Time 3D Scene Graph Updates for Embodied Question Answering](https://arxiv.org/abs/2506.01174)**: Updates structured scene graphs online; conceptually similar but different benchmark and modality.
- **[CoSPlan: Corrective Sequential Planning via Scene Graph Incremental Updates](https://arxiv.org/abs/2512.10342)**: Uses incremental scene-graph updates for sequential planning; supports the incremental-update framing.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Belief-centric embodied evaluation | Evaluate active exploration + belief construction + revision | ToS, EXCALIBUR, EmbodiedBench | ToS, embodied QA tasks | Conflates exploration vs belief maintenance; belief externalization may be lossy |
| Explicit spatial representations | Convert perception to structured maps/graphs | Ego3D-VLM, GraphPad | Ego3D-Bench, OpenEQA | Errors in representation can dominate; maintaining consistency over time is hard |
| Spatial VLM adaptation | Train/augment VLMs for spatial reasoning | SpatialVLM, SpatialRGPT, 3DThinker | Multiple spatial benchmarks | Often targets static questions, not belief revision |
| Incremental structured editing | Avoid full regeneration; output diffs/patches | JSON Whisperer | JSON editing tasks | Output validity and conflict handling remain challenges |

### Closest Prior Work

1) **Theory of Space (ToS)**: Defines cognitive-map probing and belief inertia metrics. It reports belief drift and inertia but does not propose interventions beyond analysis.

2) **Ego3D-VLM**: Builds explicit cognitive maps to help spatial reasoning, but focuses on constructing maps from multi-view perception modules rather than maintaining/revising a belief under changes.

3) **GraphPad / CoSPlan**: Maintain/update structured scene representations over time, but they are evaluated on embodied QA or planning tasks rather than ToS’s belief probing and false-belief revision.

4) **JSON Whisperer**: Studies incremental structured edits for JSON documents, but not in embodied spatial belief settings.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| ToS | Benchmarks belief construction + revision; probes full-map JSON | No intervention; probing requires full regeneration | Add explicit belief-update protocol (external state) | Removes regeneration noise; enforces conflict revision |
| Ego3D-VLM | Uses cognitive maps to improve spatial QA | Not about belief revision; maps built per query | Apply incremental belief updates across steps | Targets drift/inertia over time |
| GraphPad | Updates 3D scene graph via APIs | Different task; heavier perception stack | Apply update interface to ToS map probing | ToS offers direct stability/inertia metrics |
| JSON Whisperer | JSON diff/patch editing with LLMs | Not spatial; no inertia metric | Use delta updates for cognitive maps | Smaller outputs reduce transcription errors |

---

## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Gemini-3 Pro | - | https://ai.google.dev/ | Used in the ToS paper; strong VLM baseline |

(Verification may optionally also run GPT-5.2 as a secondary check, but the decisive experiment uses a single model.)

**Training Data (if applicable):**

No training data needed — **inference only**.

**Other Resources (if applicable):**

- ToS code (release branch) and evaluation scripts: **[GitHub - mll-lab-nu/Theory-of-Space](./references/GitHub%20-%20mll-lab-nu%20Theory-of-Space/meta/meta_info.txt)**
- ToS offline dataset (100 runs, includes false-belief images): **[MLL-Lab/tos-data](./references/MLL-Lab%20tos-data%20%C2%B7%20Datasets%20at%20Hugging%20Face/meta/meta_info.txt)**

**Resource Estimate**:

- **GPU budget**: 0 GPU-hours
- **API calls (rough)**:
  - Choose `N=25` scenes (run00–run24) for verification.
  - Use passive **SCOUT** proxy trajectories (ToS’s scripted baseline: 360° sweep + room-visitation) with length ≈ 9–12 steps per scene.
  - Cognitive-map update calls: ~`N * steps * 3 conditions` ≈ 25 * 12 * 3 = 900 calls.
  - False-belief revision calls: similar order (another ~900).
  - Total: ~1.8k multimodal calls.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Theory-of-Space (vision world) cognitive map probing | Build a global map from sequential partial views | Final map correctness (pos/dir/facing composite), perception/local↔global (optional) | run00–run24 | [tos-data](./references/MLL-Lab%20tos-data%20%C2%B7%20Datasets%20at%20Hugging%20Face/meta/meta_info.txt) | ToS repo (`scripts/SpatialGym/`) with prompt modifications |
| Theory-of-Space false-belief revision (vision world) | After k=4 object changes, revise belief | **Identification F1** (which objects changed), **belief inertia** (pos/ori; ↓ better) per ToS | run00–run24 | same | same |

**Evaluation Scripts:**
- Use ToS’s released pipeline (`scripts/SpatialGym/spatial_run.py`) and modify only the cognitive-map probing prompts and the state-passing logic for conditions B/C.

### Main Results

**Decision rule (verification):**

- Primary: On the false-belief revision task, **B** reduces belief inertia and improves identification F1 vs **A**.
- Secondary (format effect): **C** improves over **B** on the same metrics (or matches B with fewer invalid outputs / fewer retries).
- Refute: If **B** does not improve final map correctness and inertia vs **A**.

#### Results Table

| Method | Base Model | Benchmark | Final map correctness (↑) | False-belief F1 (↑) | Pos inertia (↓) | Ori inertia (↓) | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| Scratch regeneration (A) | Gemini-3 Pro | ToS vision (N=25) | **TBD** | **TBD** | **TBD** | **TBD** | - | Needs re-run (subset + prompt differs from paper reporting) |
| Full regen + preserve/overwrite rules (B) | Gemini-3 Pro | ToS vision (N=25) | **TBD** | **TBD** | **TBD** | **TBD** | - | Proposed |
| **Delta updates (C)** | Gemini-3 Pro | ToS vision (N=25) | **TBD** | **TBD** | **TBD** | **TBD** | - | Proposed |

Reference numbers from the ToS paper (full setting; not directly comparable to N=25 subset): Gemini-3 Pro vision-world final correctness 52.1% and false-belief positional inertia 51.1% / orientation inertia 14.4% (Tables 5 and 7 in the ToS paper).

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B vs A | Adds explicit state + preserve/overwrite update rule | Improves correctness and reduces inertia if regeneration noise / inconsistent conflict handling matters |
| C vs B | Changes output format to delta | If transcription errors matter, C ≥ B on correctness/inertia and/or requires fewer retries |

### Analysis (Optional)

- **Edit magnitude**: average number of objects changed per step under B/C; if it is small, it supports the “sparse evidence” premise.
- **Failure stratification**: split by objects with facing direction vs not; hypothesis: gains are larger for orientation (where drift/inertia are worst).

---

## Success Criteria

**Criterion 1: Belief revision improves under false belief**
- Hypothesis: Providing explicit prior map and conflict-based overwriting (B) reduces ToS belief inertia and improves identification of changed objects vs scratch regeneration (A).
- Validation: Directional improvement on inertia and identification F1 across scenes, with paired bootstrap confidence interval excluding 0 for the mean difference.

**Criterion 2: Delta outputs reduce structured-state transcription errors**
- Hypothesis: Delta updates (C) match or improve over B while producing fewer invalid JSON outputs / fewer retries.
- Validation: C ≥ B on correctness/inertia, and retry rate decreases.

---

## Impact Statement

If successful, this work provides a simple, model-agnostic way to make spatial-belief maintenance more reliable: treat the belief as an external structured state and update it incrementally under explicit evidence and conflict rules. This can improve the reliability of LLM/VLM-based agents that must maintain revisable spatial beliefs, and it clarifies how much of ToS’s observed belief drift/inertia is attributable to state externalization rather than purely perception.

---

## References

- [Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?](./references/THEORY%20OF%20SPACE%20CAN%20FOUNDATION%20MODELS%20CON-STRUCT%20SPATIAL%20BELIEFS%20THROUGH%20ACTIVE%20EXPLORATION/meta/meta_info.txt) - Zhang et al., 2026
- [GitHub - mll-lab-nu/Theory-of-Space](./references/GitHub%20-%20mll-lab-nu%20Theory-of-Space/meta/meta_info.txt) - Code, 2026
- [MLL-Lab/tos-data · Datasets at Hugging Face](./references/MLL-Lab%20tos-data%20%C2%B7%20Datasets%20at%20Hugging%20Face/meta/meta_info.txt) - Dataset, 2026
- [JSON Whisperer: Efficient JSON Editing with LLMs](https://arxiv.org/abs/2510.04717) - Duanis et al., 2025
- [GraphPad: Inference-Time 3D Scene Graph Updates for Embodied Question Answering](https://arxiv.org/abs/2506.01174) - 2025
- [CoSPlan: Corrective Sequential Planning via Scene Graph Incremental Updates](https://arxiv.org/abs/2512.10342) - 2025
