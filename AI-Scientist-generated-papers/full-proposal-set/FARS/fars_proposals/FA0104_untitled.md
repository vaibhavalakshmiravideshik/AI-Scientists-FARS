# untitled

# Search-Anchored Hybrid Rollouts for WebShop Text World Models

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language models (LLMs) are increasingly used as interactive agents: they observe an environment described in text, choose an action (often also text), and receive the next observation. A common way to make such agents safer and more sample-efficient is to train or use a **world model**: a model that predicts the next observation and reward (or success signal) given the interaction history and a candidate action.

Recent work has shown that LLMs can be trained into **text-based world models** by supervised fine-tuning on environment interaction trajectories. **From Word to World** (Word2World) trains Qwen2.5-7B / Llama3.1-8B world models for several text environments (including WebShop) and evaluates them not only on single-step next-state accuracy but also on long-horizon rollouts via the **Consistency Ratio** (CR). CR is defined as CR = W2R / Real, where Real is the task success rate in the real environment and W2R (“world-to-real”) is the success rate when we replay an action sequence generated inside the world model back in the real environment. Higher CR means better rollout fidelity, and CR = 1 corresponds to perfect transfer (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt`).

However, a key limitation is that rollout consistency degrades substantially in more open-ended environments. In Word2World’s Table 2, the Qwen2.5-7B world model has average WebShop CR = 0.67, and for the GPT-4o acting agent specifically Real = 29.36%, W2R = 16.51%, CR = 0.56 (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/5.2 Rollout Consistency.md`). Low CR means that imagined trajectories are often not executable in the real environment, limiting how much practitioners can rely on world-model rollouts for planning, verification, or synthetic data generation.

WebShop is a simulated e-commerce website environment with 1.18M products and 12,087 instructions split into 10,587 / 1,000 / 500 train/dev/test (`./references/WebShop-Towards-Scalable-Real-World-Web-Interaction-with-Grounded-Language-Agents/meta/meta_info.txt`). WebShop uses a deterministic BM25-based retrieval system for search (Pyserini) and exposes a high-level action `search[query]` that returns a ranked list of products (`./references/WebShop-Towards-Scalable-Real-World-Web-Interaction-with-Grounded-Language-Agents/sections/3.2 Environment Implementation.md`).

### The Problem

Word2World provides a concrete clue for why WebShop rollouts drift: it states that WebShop consistency is “typically below 80%” primarily because of “diverse search results that the world model struggles to simulate accurately,” and that “when the rollout is initialized with real search results the consistency with GPT-4o increases dramatically from 56% to nearly 100%” (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/Consistency Across Environments.md`).

This suggests that WebShop rollouts may have a concentrated error source: the **search-results observation** returned after `search[...]`. If the world model is inaccurate mainly on search-result pages (high entropy, long-tail product lists), then downstream drift may be dominated by this single failure mode because early search results condition the rest of the trajectory.

The practical question is: **Can we recover near-perfect rollout transfer in WebShop by grounding only the search-result observations, while keeping the rest of the rollout simulated?** If so, world-model rollouts would become more reliable for planning and training without requiring full real-environment interaction.

### Key Insight and Hypothesis

**Key insight:** WebShop observations can be decomposed into (i) relatively structured, action-conditioned state updates (e.g., navigating pages, selecting options), and (ii) a high-entropy retrieval channel (the ranked product list returned by `search[...]`). A supervised world model trained on trajectories may learn (i) well but still fail on (ii) because retrieval outputs depend on a large product index and subtle IR details.

**Hypothesis:** If we replace only the world model’s predicted search-result observations with the real (or cached) WebShop search results during imagination rollouts, then the **Consistency Ratio** (CR) will increase compared to Word2World’s “initialize with real search results” baseline. The improvement should be concentrated in episodes with multiple search actions, where compounding search-result errors would otherwise accumulate.

**Why this could be wrong:** (i) initializing with real search results may already remove most drift (CR ≈ 1), leaving no headroom; (ii) most WebShop episodes may use only one search, making “anchor every search” equivalent to “anchor initialization”; (iii) drift may come from non-search parts of the environment (e.g., product detail text or option-selection dynamics) or from action-policy mismatch rather than observation mismatch.

---

## Proposed Approach

### Overview

We propose **search-anchored hybrid rollouts** for text-based world models in WebShop. During a simulated rollout inside the world model, whenever the acting agent chooses an action of the form `search[query]`, we bypass the world model’s next-state prediction and instead fetch the search results from the real WebShop environment (or from a deterministic cache keyed by the query). For all other actions, the rollout remains purely within the learned world model.

This is a minimal intervention: it does not change the world model weights, does not change the acting agent, and does not introduce any external web search APIs. It only uses WebShop’s own deterministic local search index.

### Method Details

**Base components.** We follow Word2World’s evaluation pipeline (`./references/GitHub-X1AOX1A-Word2World/meta/meta_info.txt`):

- **Real**: run an acting agent in the real WebShop environment and record success rate.
- **WM**: run the same acting agent against a learned world model (Qwen2.5-7B or Llama3.1-8B checkpoint) and record success rate.
- **W2R**: replay the action trace collected in WM back in the real WebShop environment and record success rate.
- **CR**: compute CR = W2R / Real (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/Metrics.md`).

**Search-anchored rollout wrapper.** During WM interaction, we intercept the agent action at each step:

- If action is not `search[...]`: forward the (history, action) pair to the world model and use its predicted next observation.
- If action is `search[query]`:
  - Query the WebShop environment’s search function to obtain the search-results observation that would be shown in the real environment.
  - Substitute this observation as the next observation in the WM rollout.

To keep the procedure deterministic and efficient:
- We build a cache mapping `query -> serialized results page text` from the WebShop index once (or cache on demand as queries appear).
- We treat search results as an **exogenous observation channel** rather than part of the learned world dynamics.

### Key Innovations

- A minimal, mechanism-driven grounding intervention for text world models: **anchor only the retrieval channel** (search results) while leaving all other steps simulated.
- A diagnostic framing for open-ended text environments: isolate whether rollout drift is driven primarily by retrieval outputs versus other parts of the transition model.

---

## Related Work

### Field Overview

This proposal builds on work in (i) LLM-based world models for interactive environments, (ii) web-agent benchmarks and model-based planning for web interaction, and (iii) grounding / retrieval for reducing hallucination in simulated trajectories. A recurring observation is that LLM simulators can be accurate locally but drift over multi-step rollouts, especially in settings with large action spaces or high-entropy observations.

### Related Papers

(Proposal-local references are used when available; otherwise arXiv/OpenReview URLs.)

- **[From Word to World](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt)**: Trains LLM-based text world models and introduces the CR metric; reports low WebShop CR and notes search-result diversity as a primary drift source.
- **[Word2World GitHub repo](./references/GitHub-X1AOX1A-Word2World/meta/meta_info.txt)**: Provides evaluation scripts for Real/WM/W2R/CR across environments including WebShop.
- **[WebShop](./references/WebShop-Towards-Scalable-Real-World-Web-Interaction-with-Grounded-Language-Agents/meta/meta_info.txt)**: Defines the WebShop benchmark and its deterministic BM25 (Pyserini) search interface.
- **[WebDreamer](./references/Is-Your-LLM-Secretly-a-World-Model-of-the-Internet-Model-Based-Planning-for-Web-Agents/meta/meta_info.txt)**: Uses LLM simulation for short-horizon web planning; highlights that long-horizon simulation degrades, but does not study partial observation anchoring in WebShop.
- **[R-WoM](./references/R-WoM-Retrieval-augmented-World-Model-For-Computer-use-Agents/meta/meta_info.txt)**: Grounds computer-use world models with retrieved tutorials; focuses on WebArena/OSWorld and does not isolate search-result anchoring within WebShop rollouts.
- **[DynaWeb](./references/DynaWeb/meta/meta_info.txt)**: Uses a learned web world model for imagination-driven training of web agents (WebArena/WebVoyager), emphasizing compounding-error limits and the need for stabilizing real-data interleaving.

Additional related work (external links; not all scraped locally):

- **[World Models](https://arxiv.org/abs/1803.10122)**: Foundational work on learning compact generative models for planning.
- **[Dreamer](https://arxiv.org/abs/1912.01603)**: Latent world models for model-based RL; illustrates error compounding over rollouts.
- **[MuZero](https://arxiv.org/abs/1911.08265)**: Planning with learned dynamics without explicit observation modeling.
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)**: Inference-time search over reasoning trajectories; relevant as a strong inference-scaling baseline for LLM decision making.
- **[ReAct](https://arxiv.org/abs/2210.03629)**: A common interactive agent prompting framework.
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)**: Multi-sample aggregation for reasoning; relevant as a generic inference-time scaling baseline.
- **[WebArena](https://arxiv.org/abs/2307.13854)**: A realistic web-agent benchmark (requires browser execution; not feasible here).
- **[Mind2Web](https://arxiv.org/abs/2306.06070)**: Large-scale web interaction dataset; motivates distribution shift issues.
- **[VisualWebArena](https://arxiv.org/abs/2401.13649)**: Multimodal web benchmark used by WebDreamer.
- **[AgentGym](https://arxiv.org/abs/2406.04151)**: A unified suite of environments (including WebShop) for training and evaluating LLM agents.
- **[Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992)**: Early work explicitly connecting LLM reasoning to world modeling.
- **[Tree Search for Language Model Agents](https://arxiv.org/abs/2403.14589)**: Studies tree search for LLM agents; motivates simulation vs real-interaction trade-offs.
- **[TEXT2WORLD](https://arxiv.org/abs/2502.13092)**: Benchmark for generating executable symbolic world models from text.
- **[Making LLMs into World Models with Precondition and Effect Knowledge](https://arxiv.org/abs/2409.12278)**: Adds explicit precondition/effect modeling for more reliable simulation.
- **[RLVR-World](https://arxiv.org/abs/2505.13934)**: Trains world models with verifiable rewards.
- **[Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/)**: Evaluates LLM simulators and discusses compounding error in text environments.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Text world models (SFT) | Train LLM to predict next observations from trajectories | Word2World | ALFWorld, SciWorld, TextWorld, WebShop; CR=W2R/Real | Drift in open-ended envs (e.g., WebShop) |
| Web planning via simulation | Use LLM simulation to score candidate actions | WebDreamer | VWA, Mind2Web-live | Simulation horizon limited; hallucinations |
| Retrieval-grounded world models | Use external retrieval (tutorials/docs) to reduce hallucination | R-WoM | WebArena, OSWorld | Needs tutorial corpora; retrieval quality |
| Web MBRL training | Use learned web world model to generate imagined data for RL | DynaWeb | WebArena, WebVoyager | Needs real-data interleaving; compounding errors |

### Closest Prior Work

**Word2World.** Word2World is the closest prior work because it defines the CR metric and reports that WebShop consistency is limited by search-result diversity. It also reports that initializing rollouts with real search results can raise GPT‑4o WebShop consistency from 56% to nearly 100%. However, it does not isolate whether anchoring only the first search is sufficient, nor whether anchoring later searches yields additional gains.

**WebShop.** The WebShop benchmark defines the deterministic search interface (`search[query]`) and explicitly notes that agents may need to perform multiple searches and query reformulation.

**WebDreamer / R-WoM / DynaWeb.** These works use simulation and/or retrieval grounding to improve web agents, but they do not test the minimal intervention proposed here: *inject real/cached WebShop search results into otherwise simulated rollouts* and measure CR.

**Novelty Kill Search Summary:** Searched for combinations of “WebShop world model real search results anchoring,” “anchor search results rollout simulation,” “hybrid rollout real observation injection world model,” and checked the local KB for “search-anchored” / “anchored rollout” occurrences. No prior work directly testing “anchor search observations only” for WebShop CR was found as of 2026-02-16 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Word2World | Trains text world models; evaluates CR; notes init search anchoring helps | Does not isolate the drift source beyond a qualitative claim | Anchor **every** search observation | If drift is mainly due to search results, repeated anchoring prevents compounding search errors |
| WebDreamer | Simulates web transitions for planning | Simulation drifts with horizon; not WebShop CR | Replace only retrieval observations | More targeted grounding with minimal extra real interaction |
| R-WoM | Uses tutorials to ground imagination | Requires tutorial corpora; different benchmarks | Use the environment’s own search index as grounding | WebShop search is deterministic and directly addresses the identified drift source |
| DynaWeb | Trains agents from imagined web rollouts | Needs heavy training + real-data mixing | Inference-only grounding for rollouts | Improves rollout transfer without retraining the world model |

---

## Experiments

### Experimental Setup

**Core setting.** Use Word2World’s WebShop evaluation harness and pretrained world model checkpoints (`./references/GitHub-X1AOX1A-Word2World/sections/README.md.md`). Use the WebShop environment’s deterministic search index (BM25/Pyserini) for anchored search observations.

**Acting agent.** Use a fixed acting agent model across all conditions. Default: **gemini-2.5-flash** (API model; no local GPU cost). Alternate: **gpt-4o** if needed for direct comparability with Word2World Table 2.

**World model.** Use the released WebShop world model checkpoint, default: `X1AOX1A/WorldModel-Webshop-Qwen2.5-7B` (as used in Table 2). Optionally repeat with `X1AOX1A/WorldModel-Webshop-Llama3.1-8B` as a robustness check if budget allows.

**Splits.** Use the WebShop test split (500 instructions) as defined by the WebShop benchmark (`./references/WebShop-Towards-Scalable-Real-World-Web-Interaction-with-Grounded-Language-Agents/sections/5.1 Setup and task verification.md`).

**Main conditions (≤3).**

1. **Pure WM rollout (baseline).** Agent interacts with the learned world model for the entire episode.
2. **First-search anchored rollout (control).** For the first occurrence of a `search[...]` action in the episode (whenever it happens), replace the world model’s predicted search-results observation with real/cached WebShop search results. All later search results are generated by the world model.
3. **All-search anchored rollout (ours).** For every `search[...]` action during WM interaction, replace the predicted search results with real/cached WebShop search results.

**Metrics.** For each condition, compute:
- Real / WM / W2R success rates and **CR = W2R / Real** (Word2World definition).
- Number of `search[...]` actions per episode; report CR stratified by search count.

**Pre-check gate (early stop).** Run conditions (1) and (2) first and compute:
- CR_init = CR(condition 2)
- p_multi = fraction of episodes with >=2 `search[...]` actions

Proceed to condition (3) only if:
- **CR_init < 0.95** (enough headroom to improve beyond first-search anchoring), and
- **N_multi >= 100** episodes with >=2 `search[...]` actions (i.e., p_multi >= 0.20 on the 500-task WebShop test set).

Note: Word2World reports that first-search anchoring raises GPT‑4o consistency “from 56% to nearly 100%” but does not provide the exact anchored CR value in the paper; we therefore treat CR_init as a measured quantity in this pre-check. The N_multi threshold is motivated by the original WebShop analysis, which reports that 74.8% of human expert trajectories contain only one query (so 25.2% have >=2 queries) on the 500 test tasks (`./references/WebShop-Towards-Scalable-Real-World-Web-Interaction-with-Grounded-Language-Agents/sections/Results with at Choice oracle..md`).

If the gate fails, record the result as an informative negative: first-search anchoring already saturates consistency (no headroom) and/or multi-search is too rare for all-search anchoring to matter at aggregate level.

**Baseline ladder (minimum).**
- **Closest existing method baseline**: Word2World pure WM rollout numbers (Table 2) as a reference point; we will re-run the baselines in our harness for the chosen acting agent.
- **Inference-time scaling baseline (agent-side)**: best-of-N action sampling for the acting agent in WM (e.g., N=4) while keeping the world model unchanged; this tests whether gains can be matched by spending more inference compute on action selection rather than anchoring observations.
- **Prompting baseline (agent-side)**: a stronger agent prompt for WebShop (e.g., ReAct-style with explicit search/refinement instructions) in the pure WM setting.

Note: Unlike typical ML tasks, there is no direct “zero-shot prompting” baseline for the world model itself because the world model is a fine-tuned simulator, not the acting agent. Our prompting and best-of-N baselines apply to the *acting agent* under a fixed world model.

**Seeds / variance.** The environment is deterministic given an action sequence, but the acting agent may be stochastic (temperature sampling) and the world model may be stochastic if sampled. We will run each condition with **3 random seeds** controlling agent/world-model sampling. If the chosen agent is run deterministically (temperature 0), we will state this explicitly and treat the run as deterministic.

**Key confounders and controls.**
- **Compute mismatch**: anchoring uses extra real-environment queries; we will report the number of real search calls per episode and optionally match a compute budget by allowing the baseline to query the world model multiple times (best-of-N) at each action.
- **Caching artifacts**: to ensure no leakage, the cache key is only the literal query string, and cached results must match the environment’s deterministic search output.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| WebShop | Text-based shopping tasks with `search[...]` and `choose[...]` actions | Real, WM, W2R, CR | test (500) | https://github.com/princeton-nlp/WebShop | Word2World scripts (`./references/GitHub-X1AOX1A-Word2World/sections/README.md.md`) |

### Main Results

(Published baselines are included; new results to be filled by verification.)

| Method | Base Model | Benchmark | CR ↑ (mean±std) | Real ↑ (mean±std) | W2R ↑ (mean±std) | Source | Notes |
|---|---|---|---|---|---|---|---|
| Pure WM rollout (published) | Qwen2.5-7B WorldModel + GPT‑4o agent | WebShop | 0.56 (1 run) | 29.36% (1 run) | 16.51% (1 run) | Word2World Table 2 (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/5.2 Rollout Consistency.md`) | WM=17.43% |
| Pure WM rollout (published) | Qwen2.5-7B WorldModel + gemini-2.5-flash agent | WebShop | 0.73 (1 run) | 25.00% (1 run) | 18.35% (1 run) | Word2World Table 2 (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/5.2 Rollout Consistency.md`) | WM=21.10% |
| Pure WM rollout (published, avg) | Qwen2.5-7B WorldModel + 7 agents (GPT-4o-mini, GPT‑4o, GPT‑4-turbo, GPT‑4.1, GPT‑5, gemini-2.5-flash, claude-sonnet-4.5) | WebShop | 0.67 (1 run) | 30.17% (1 run) | 20.13% (1 run) | Word2World Table 2 (`./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/sections/5.2 Rollout Consistency.md`) | WM=21.79% |
| Pure WM rollout (re-run, matched harness) | Qwen2.5-7B WorldModel + chosen acting agent | WebShop | TBD | TBD | TBD | - | Should reproduce the corresponding Word2World baseline when using the same agent + prompt |
| First-search anchored (control) | Qwen2.5-7B WorldModel + chosen acting agent | WebShop | TBD | TBD | TBD | - | Anchor only the first `search[...]` observation |
| **All-search anchored (ours)** | Qwen2.5-7B WorldModel + chosen acting agent | WebShop | TBD | TBD | TBD | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Ours (all-search anchored) | Anchor all searches | Best CR if search is dominant drift source |
| Init-only anchored | Anchor only first search | Helps, but less than anchoring all searches when multiple searches occur |

### Experimental Rigor

- **Seeds**: 3 seeds per condition unless determinism is justified.
- **Sanity check**: verify that the cached search results exactly match the real WebShop search output for a random subset of queries.
- **Data leakage note**: WebShop may appear in some LLM pretraining; the core comparison is between WM variants under the same agent, so this primarily affects absolute Real rates rather than the CR deltas.

---

## Success Criteria

**Hypothesis (directional).** Anchoring only search observations will increase WebShop rollout consistency compared to init-only anchoring, especially on episodes that perform multiple searches.

**Decision Rule (concrete).**
- **Proceed** if CR(all-search anchored) > CR(init-only anchored) by a statistically distinguishable margin across 3 seeds (bootstrap CI over tasks excludes 0 for the mean difference) and the improvement is larger for episodes with more searches.
- **Refute** if CR(init-only anchored) >= 0.95 (no headroom) or if CR(all-search anchored) − CR(init-only anchored) is within noise and does not increase with search count.
- **Pivot** if CR does not improve but analysis shows drift originates from non-search observations (e.g., item pages); a follow-up would be to anchor a different observation type.

---

## Impact Statement

If successful, this work would provide a simple way for practitioners to make world-model rollouts reliable in WebShop-like environments: treat retrieval outputs as an external observation channel and only ground those steps, instead of abandoning rollouts entirely or requiring full real-environment interaction. If unsuccessful, the result still clarifies that WebShop drift is not primarily caused by search-result simulation, narrowing where future world-model improvements should focus.

---

## References

- [From Word to World: Can Large Language Models be Implicit Text-based World Models?](./references/From-Word-to-World-Can-Large-Language-Models-be-Implicit-Text-based-World-Models/meta/meta_info.txt) - Li et al., 2025
- [GitHub - X1AOX1A Word2World](./references/GitHub-X1AOX1A-Word2World/meta/meta_info.txt) - Li et al., 2025
- [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](./references/WebShop-Towards-Scalable-Real-World-Web-Interaction-with-Grounded-Language-Agents/meta/meta_info.txt) - Yao et al., 2022
- [Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents](./references/Is-Your-LLM-Secretly-a-World-Model-of-the-Internet-Model-Based-Planning-for-Web-Agents/meta/meta_info.txt) - Gu et al., 2025
- [R-WoM: Retrieval-augmented World Model For Computer-use Agents](./references/R-WoM-Retrieval-augmented-World-Model-For-Computer-use-Agents/meta/meta_info.txt) - Mei et al., 2025
- [DynaWeb](./references/DynaWeb/meta/meta_info.txt) - Ding et al., 2025
- [World Models](https://arxiv.org/abs/1803.10122) - Ha and Schmidhuber, 2018
- [Dreamer](https://arxiv.org/abs/1912.01603) - Hafner et al., 2019
- [MuZero](https://arxiv.org/abs/1911.08265) - Schrittwieser et al., 2019
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) - Yao et al., 2023
- [ReAct](https://arxiv.org/abs/2210.03629) - Yao et al., 2022
- [Self-Consistency](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [WebArena](https://arxiv.org/abs/2307.13854) - Zhou et al., 2023
- [Mind2Web](https://arxiv.org/abs/2306.06070) - Deng et al., 2023
- [VisualWebArena](https://arxiv.org/abs/2401.13649) - Koh et al., 2024
- [AgentGym](https://arxiv.org/abs/2406.04151) - Huang et al., 2024
- [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992) - Hao et al., 2023
- [Tree Search for Language Model Agents](https://arxiv.org/abs/2403.14589) - Koh et al., 2024
- [TEXT2WORLD](https://arxiv.org/abs/2502.13092) - Hu et al., 2025
- [Making Large Language Models into World Models with Precondition and Effect Knowledge](https://arxiv.org/abs/2409.12278) - Li et al., 2024
- [RLVR-World](https://arxiv.org/abs/2505.13934) - Chen et al., 2025
- [Can Language Models Serve as Text-Based World Simulators?](https://aclanthology.org/2024.acl-short.1/) - Wang et al., 2024
