# untitled

# Fit Cards for Agentic Marketplace Search: Query-Conditioned Structured Metadata to Reduce Welfare Loss at Large Consideration Sets

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language model (LLM) agents are increasingly used as decision-making proxies for users in economic tasks such as product discovery, negotiation, and purchasing. In many deployed systems, an agent’s first step is *discovery*: it issues a search request to a marketplace (or an agent directory) and decides which candidates to contact. This makes the **search result payload** (what information is shown per candidate at search time) a platform design choice that can affect downstream outcomes.

**Magentic Marketplace** is an open-source environment for studying two-sided agentic markets, where “assistant agents” represent consumers and “service agents” represent businesses. In the Mexican restaurant domain at scale 100 customers / 300 businesses, Magentic Marketplace reports a striking *consideration-set scaling penalty* (the “paradox of choice”): increasing the number of search results shown from 3 to 100 reduces **welfare** by −4.3% (GPT-4o), −65.4% (Sonnet-4), and −44% (GPT-5 with minimal reasoning) (Section 5.2 in `./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/sections/5.2 Consideration Set Size.md`). Here welfare is defined as the **sum of customer utilities across completed transactions** (Section 4.2).

The same paper finds that **first-proposal bias** is severe and nearly universal: assistant agents tend to accept the *first* proposal they receive (60–100% selection rates) rather than waiting to compare later proposals (Section 5.4 in `./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/sections/5.4 Agent Behavior Biases.md`). This implies that early discovery mistakes can cascade: contacting poor-fit businesses early can produce early low-utility proposals that get accepted.

Human-facing search and recommender systems often address choice overload by improving “choice architecture”: presenting concise summaries, organizing options, and showing structured attributes that help users quickly eliminate poor options. For LLM agents, the dominant failure modes may be different (context window growth, brittle comparison, anchoring), but the same high-level idea applies: **increase decision-relevant information at search time without restricting the number of options shown**.

### The Problem

In Magentic Marketplace, businesses and customers are generated with structured fields (items/services, prices, amenities/attributes), and then rendered into natural-language descriptions for agents (Section 4.1 in `./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/sections/4.1 Data Generation.md`). However, the search API (`/action search`) primarily returns business identifiers (Table 1 in `./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/sections/3.2 Implementation Overview.md`), so assistants must rely on unstructured snippets and additional interaction to infer which businesses are good fits.

This creates an interface mismatch:

- The platform has structured catalogs that directly answer key selection questions (which requested items exist? do amenities match? what is the price for the requested bundle?),
- but assistants must infer these signals from many unstructured snippets and then spend additional interactions contacting businesses to verify details.

At large consideration set sizes (e.g., 100 results), this mismatch can induce the failure modes hypothesized in Magentic Marketplace (Section 5.2): assistants contact more poor-fit businesses, context grows, and early low-utility proposals become more likely to be accepted due to first-proposal bias.

The core research question is:

**Can a platform-side change to the search result payload (query-conditioned structured metadata per result) recover a substantial fraction of the welfare lost when scaling from N=3 to N=100, without changing the search algorithm or ranking?**

### Key Insight and Hypothesis

**Key insight:** The consideration-set scaling penalty is partly an *information interface* problem. The marketplace can compute query-conditioned fit signals deterministically from its structured catalog and the customer’s structured request, but this information is not surfaced at search time.

**Hypothesis:** Adding a fixed-budget, query-conditioned structured “fit card” to each search result will improve early contact selection (higher true-fit / oracle-utility among contacted businesses), reduce irrelevant context growth, and recover a large fraction of welfare lost when moving from N=3 to N=100.

This could fail if the scaling penalty is dominated by proposal-stage anchoring rather than discovery-stage errors, if assistants can already extract equivalent signals from free-form text with strong prompting, or if the key limitation is insufficient exploration rather than lack of information.

---

## Proposed Approach

### Overview

We propose **Fit Cards**: platform-computed, query-conditioned structured metadata attached to each search result. A fit card summarizes how well a business matches the customer’s request according to the platform’s structured catalog, in a compact, fixed schema.

This is explicitly a **platform-side metadata augmentation** intervention:

- The search algorithm and ranking are unchanged (lexical search remains lexical search).
- The only change is what per-result information the platform returns.

### Method Details

**Injection point:** Extend the `/action search` response schema in Magentic Marketplace (Table 1 in `./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/sections/3.2 Implementation Overview.md`). The response should remain backward compatible (keep `results: [agent_name]`) and add an optional `result_cards` mapping keyed by `agent_name`.

Example schema:

```json
{
  "results": ["agent_17", "agent_92"],
  "result_cards": {
    "agent_17": {
      "items_hit": 2,
      "amenities_hit": 1,
      "est_total_price": 23.50
    }
  }
}
```

**Fit card fields (Mexican domain):**
- `items_hit`: count of requested menu items present in business catalog
- `amenities_hit`: count of required amenities satisfied
- `est_total_price`: sum of prices for requested items that exist (else `null`)

These fields are computed deterministically from the business record and the customer’s structured request (Section 4.1). In Magentic Marketplace the platform can associate the calling assistant (via `api_token`) with its customer profile.

**Token-budget control (explicit):** To avoid confounding improvements with “shorter context”, we cap the per-result payload to a fixed budget **L = 40 tokens** across all conditions.
- Baseline payload: show the business description truncated to the first 40 tokens.
- Fit-card payload: render the 3 fields in a fixed template that fits within 40 tokens.

This control does not equalize “informativeness per token” (fit cards are intentionally information-dense). The goal is to ensure gains are not trivially explained by reducing context length.

### Key Innovations

1. **Search payload as a mechanism-design lever for LLM agents:** Treat the search result payload (not just ranking) as a controllable design variable that can shift welfare outcomes in agentic markets.
2. **Query-conditioned, platform-trusted metadata:** Use structured catalog information already available to the marketplace to compute decision-relevant features, reducing reliance on brittle free-form descriptions.
3. **Mechanism test using welfare:** Evaluate a metadata intervention using end-to-end welfare in a market simulation, and test the mechanism via contacted-fit diagnostics.

---

## Related Work

### Field Overview

This proposal connects three literatures: (i) agentic market environments that make economic outcomes measurable under controlled interventions, (ii) choice overload and choice architecture in search and recommender systems, and (iii) the role of metadata/structured interfaces in steering agent behavior.

Agentic marketplace environments such as Magentic Marketplace and other economic testbeds highlight that multi-agent economic settings exhibit failure modes (biases, manipulation, anchoring) not captured by single-agent benchmarks. Choice overload work establishes that increasing the number of available options can reduce decision quality, and motivates interventions that make option comparison easier. Finally, recent work on tool/agent ecosystems shows that metadata can strongly influence agent behavior, both constructively (protocol standardization) and adversarially (metadata attacks), suggesting that metadata is a first-class control surface.

A key confound for this proposal is separating **discovery-stage** improvements (better contact selection) from **proposal-stage** improvements (mitigating first-proposal anchoring). A complementary class of interventions changes the transaction protocol to reduce first-proposal bias (e.g., delaying acceptance or batching payments). A related finalized proposal in this repository (“quote-batched-payment-proposal-bias”) explores such protocol constraints; our work instead targets discovery-stage errors and includes strong agent-side prompting and budget baselines to test whether platform metadata adds value beyond prompting/compute.

### Related Papers

- **[Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets](./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/meta/meta_info.txt)**: Introduces a two-sided agentic marketplace simulator and documents the N=3→N=100 welfare drop and first-proposal bias that motivate this proposal.
- **[The Agentic Economy](https://arxiv.org/abs/2505.15799)**: Argues that two-sided agentic markets will reshape economic activity, motivating market-design research for agentic platforms.
- **[Virtual Agent Economies](https://arxiv.org/abs/2509.10147)**: Surveys risks and dynamics of economies populated by agents.
- **[What Is Your AI Agent Buying?](https://arxiv.org/abs/2508.02630)**: Studies agentic e-commerce behavior and highlights evaluation and design implications.
- **[FaMA: LLM-Empowered Agentic Assistant for C2C Marketplace](https://arxiv.org/abs/2509.03890)**: Builds a marketplace assistant for buyer/seller workflows, illustrating deployment-relevant agent behaviors.
- **[Project Vend](https://www.anthropic.com/research/project-vend-1)**: Case study of an LLM agent operating a shop, emphasizing practical failure modes and control needs.
- **[NegotiationArena](https://arxiv.org/abs/2402.05863)**: Platform for studying LLM negotiation behavior, related to proposal-stage dynamics.
- **[Deal or No Deal?](https://arxiv.org/abs/1706.05125)**: Early end-to-end negotiation dialogue study, foundational for understanding offer dynamics.
- **[Decoupling Strategy and Generation in Negotiation Dialogues](https://arxiv.org/abs/1808.09637)**: Separates negotiation planning from language generation, relevant to mitigating anchoring.
- **[Auction Arena](https://arxiv.org/abs/2310.05746)**: Economic environment benchmark for strategic planning and execution.
- **[GLEE](https://arxiv.org/abs/2410.05254)**: Framework/benchmark for language-based economic environments.
- **[Economics Arena for LLMs](https://arxiv.org/abs/2401.01735)**: Evaluation suite for LLM economic behaviors.
- **[The AI Economist](https://arxiv.org/abs/2004.13332)**: RL-based economic simulation environment.
- **[ABIDES-Economist](https://arxiv.org/abs/2402.09563)**: Agent-based simulation framework for economic systems.
- **[LLM Economist](https://arxiv.org/abs/2507.15815)**: Large-population generative simulacra for mechanism design.
- **[Algorithmic Collusion by LLMs](https://arxiv.org/abs/2404.00806)**: Shows emergent multi-agent market risks.
- **[When More Is Less: The Paradox of Choice in Search Engine Use](https://dl.acm.org/doi/10.1145/1571941.1572030)**: Human study showing choice overload in search result selection.
- **[Understanding Choice Overload in Recommender Systems](https://dl.acm.org/doi/10.1145/1864708.1864724)**: RecSys study of recommendation set size and choice difficulty/satisfaction.
- **[The Choice Overload Effect in Online Recommender Systems](https://pubsonline.informs.org/doi/10.1287/msom.2022.0659)**: Large-scale field evidence of an inverted-U effect of set size on engagement.
- **[The Amplification Paradox in Recommender Systems](https://arxiv.org/abs/2302.11225)**: Agent-based framing relating recommendation exposure and user utility.
- **[Fairness and Diversity in Recommender Systems: A Survey](https://arxiv.org/abs/2307.04644)**: Survey covering slate design, exposure, and diversity objectives relevant to option-set design.
- **[Diffusion Model for Slate Recommendation](https://arxiv.org/abs/2408.06883)**: Modern method family for slate generation.
- **[Attractive Metadata Attack](https://arxiv.org/abs/2508.02110)**: Demonstrates that tool/agent metadata strongly steers LLM agent behavior, motivating metadata-centric interventions.
- **[Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)**: Illustrates the trend toward structured interfaces and schemas in agent ecosystems.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Agentic marketplace environments | End-to-end markets with multi-agent discovery/negotiation/payment | Magentic Marketplace; GLEE; Economics Arena | Welfare/utility, bias measures, robustness | Often synthetic data; limited real-world externalities |
| Choice overload & choice architecture | Set size affects decision quality; improve option presentation | Oulasvirta 2009; Bollen 2010; Long 2024 | Human studies; click/purchase behavior | Human-centric; not tailored to LLM context constraints |
| Slate recommendation | Optimize sets (not single items), trading off diversity/utility | DMSR; fairness/diversity survey | Ranking metrics, engagement | Rarely evaluated for LLM-agent welfare |
| Metadata as agent control surface | Structured metadata/protocols steer agent actions | Attractive Metadata Attack; MCP | Tool-use benchmarks; security metrics | Often adversarial rather than constructive |

### Closest Prior Work

1. **Magentic Marketplace**: Quantifies the consideration-set size penalty and first-proposal bias, but does not test platform-side search payload interventions.
2. **Choice overload in search/recommenders**: Motivates “better summaries and attributes” as a solution, but evaluates humans rather than LLM agents and does not measure welfare.
3. **Metadata steering / attacks**: Shows that metadata influences agent decisions, but focuses on security risks rather than welfare-improving platform design.
4. **Protocol-level mitigations of first-proposal bias (delayed acceptance / batched proposals)**: Complementary approach targeting proposal-stage anchoring rather than discovery-stage selection.

**Novelty Kill Search Summary:** We searched for direct prior work combining “Magentic Marketplace” + “consideration set size” + “structured result cards / query-conditioned metadata” and for “LLM agent paradox of choice search results” (queries logged in `notes.md`). No work was found that tests platform-side query-conditioned per-result fit metadata as a mitigation for Magentic Marketplace’s N=3→N=100 welfare drop as of 2026-02-21.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Magentic Marketplace | Measures welfare and biases in agentic markets | Search payload is essentially unstructured text | Add query-conditioned structured fit metadata to each result | Better early contact selection under first-proposal bias |
| Choice overload (search/recsys) | Studies overload and UI effects for humans | Not evaluated for LLM agents; not welfare-based | Apply choice-architecture idea to LLM-agent marketplace search payload | LLM-specific failure modes make metadata particularly valuable |
| Protocol-level delayed acceptance | Forces assistants to consider multiple proposals | Does not improve who gets contacted first | Fit cards improve discovery-stage selection | Orthogonal levers: discovery quality vs acceptance timing |
| Metadata attack work | Shows metadata steers agent behavior (adversarially) | Security-centric; not welfare-centric | Use trusted platform-computed metadata | Improves decisions without relying on seller-provided text |

---

## Experiments

### Experimental Setup

**Environment:** Magentic Marketplace (`./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/`).

**Benchmark instance:** Mexican domain at 100 customers / 300 businesses (dataset directory `data/mexican_100_300` exists in the upstream repo).

**Core knob for consideration set size:** Use `magentic-marketplace run ... --search-algorithm lexical --search-bandwidth N` (docs list `--search-bandwidth`, default 10). We interpret `search-bandwidth` as the number of results returned to the assistant per search call/page, matching the paper’s framing of “providing 3 vs 100 search results in the initial consideration set” (Section 5.2).

**Base model:** `claude-sonnet-4` (API), because the Magentic Marketplace paper reports a large N=3→N=100 welfare drop for Sonnet-4 in Mexican 100–300 (−65.4%). The upstream repo’s `.env` configuration uses a single `LLM_MODEL`, so we assume assistants and service agents share the same LLM.

**Baseline Ladder (REQUIRED):**
- **Level 0 (simple engineering baseline):** Reduce consideration set size (N=3).
- **Level 1 (status quo):** N=100 with baseline search payload.
- **Level 2 (prompting baseline):** N=100 baseline payload with an assistant prompt that explicitly extracts fit-relevant signals from the description snippet and compares candidates.
- **Level 4 (inference-time scaling baseline):** N=100 baseline payload with higher interaction budget and an explicit exploration constraint before paying.
- **Level 5 (ours):** N=100 with platform-computed fit cards.

**Main conditions (3):**
- **A: N=3 baseline payload** (`--search-bandwidth 3`): per-result payload is the business description truncated to **L=40 tokens**.
- **B: N=100 baseline payload** (`--search-bandwidth 100`): per-result payload is the business description truncated to **L=40 tokens**.
- **C: N=100 Fit Cards (ours)** (`--search-bandwidth 100`): per-result payload is a fit card rendered in a fixed template capped at **L=40 tokens**.

**Ablation Studies (2):**
- **B-prompt ("just prompt it" baseline):** same as B, but the assistant prompt is modified to (i) explicitly extract `items_hit`, `amenities_hit`, and an estimated total price from each description snippet, (ii) score candidates, and (iii) only contact top-ranked businesses.
- **B-budget (inference-time scaling baseline):** same as B-prompt, but increase `--customer-max-steps` (e.g., 200) and require contacting ≥M businesses and waiting for ≥K proposals before paying, unless a hard timeout triggers.

**Seeds / variance:** Run all methods with ≥3 independent runs (seeds), reporting mean ± std. If feasible, match the paper’s 5-run protocol.

**Primary outcome:** Consumer welfare, defined as the sum of utilities over completed transactions (Section 4.2 in `./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/sections/4.2 Evaluation.md`), where utility is `U = V * F − P` (fit is all-or-nothing).

**Mechanism diagnostics (automatically computed from ground truth):**
- Contacted-fit rate: fraction of contacted businesses that satisfy all required items and amenities.
- Oracle-utility among contacted businesses: best achievable utility if the agent chose the best contacted business.
- # businesses contacted; # proposals received; time-to-first-proposal.

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| `claude-sonnet-4` | API | https://www.anthropic.com | Chosen because Magentic Marketplace reports a large N=3→N=100 welfare drop for Sonnet-4 (−65.4%) |

**Training Data (if applicable):**

| Dataset | Purpose | Size | Download Link | License |
|---------|---------|------|---------------|---------|
| N/A | Inference-only | - | - | - |

No training data needed – inference only.

**Other Resources (if applicable):**
- Magentic Marketplace code + datasets: https://github.com/microsoft/multi-agent-marketplace

**Resource Estimate**:
- **Compute budget**: 0 GPU-hours (API-based inference; no training).
- **GPU memory**: N/A.
- **API usage**: Upper bound if every customer takes `customer_max_steps` steps:
  - Main conditions: 100 customers × 100 steps × 3 conditions × 3 seeds ≈ 90,000 agent steps.
  - Ablations: 100 customers × 200 steps × 2 ablations × 3 seeds ≈ 120,000 agent steps.
  - Total upper bound ≈ 210,000 agent steps/LLM calls.
  - Actual calls should be lower due to early stopping after a successful transaction.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Magentic Marketplace Mexican 100–300 | Synthetic two-sided marketplace (100 customers, 300 restaurants) | Welfare (sum utility; higher is better), welfare drop %, welfare recovery ratio, contacted-fit diagnostics | N/A (simulation) | https://github.com/microsoft/multi-agent-marketplace | `magentic-marketplace analyze <exp>` |

### Main Results

#### Comparability Rules (CRITICAL)

- Same dataset (Mexican 100–300) and same model across methods.
- Same search algorithm (lexical) and same search ranking.
- Same per-result token budget (L=40) for search payload across all conditions.

#### Results Table

| Method | Base Model | Benchmark | Welfare drop from N=3→N=100 (mean±std) | Welfare recovery (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| N=100 baseline (paper reference) | Sonnet-4 | Mexican 100–300 | −65.4% (std not reported) | N/A | `./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/sections/5.2 Consideration Set Size.md` | Reported with default environment settings |
| **B: N=100 baseline payload** | `claude-sonnet-4` | Mexican 100–300 | **TBD** | 0.0 | - | To be verified (token-budget controlled) |
| **B-prompt** | `claude-sonnet-4` | Mexican 100–300 | **TBD** | **TBD** | - | Prompt-only baseline |
| **B-budget** | `claude-sonnet-4` | Mexican 100–300 | **TBD** | **TBD** | - | Inference-time scaling baseline |
| **C: N=100 Fit Cards (ours)** | `claude-sonnet-4` | Mexican 100–300 | **TBD** | **TBD** | - | To be verified |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B-prompt | Same payload as baseline, but prompt explicitly extracts and compares fit signals | If B-prompt ≈ C, fit cards add little beyond prompting |
| B-budget | More steps + enforced exploration before paying | If B-budget ≈ C, the main fix is more inference budget rather than metadata |

### Experimental Rigor

**Variance & Reproducibility:**
- Run each method with ≥3 independent runs (seeds) and report mean ± std.
- Record `LLM_TEMPERATURE`, max tokens, and any RNG seeds used in the environment.

**Validity & Controls:**
- **Primary confound (payload length):** controlled by enforcing identical per-result token budget (L=40) across A/B/C.
- **Primary confound (ranking changes):** controlled by keeping lexical ranking fixed between B and C.
- **Sanity check:** verify that the baseline paradox-of-choice effect exists for the chosen model (B has lower welfare than A).
- **Data leakage:** the dataset is synthetic and generated in 2025; it is unlikely to be in model training data, but we do not rely on this for correctness.

**Mechanism interpretation (discovery vs proposal stage):**
- If Fit Cards improves contacted-fit rate but does not improve welfare, this supports the hypothesis that proposal-stage anchoring dominates; the natural follow-up is to combine discovery improvements (fit cards) with protocol-level delayed-acceptance mechanisms that reduce first-proposal bias.

### Analysis (Optional)

- Test the mechanism by correlating contacted-fit rate and oracle-utility-within-contacted-set with welfare across conditions.

---

## Success Criteria

**Hypothesis** (directional — what you expect):
Fit cards reduce the welfare drop at N=100 by improving early contact selection, thereby mitigating first-proposal anchoring and context growth.

**Decision Rule** (concrete — when to stop):
- **Continue/Proceed** if all of the following hold:
  - Welfare recovery for Fit Cards is ≥ 0.5 (mean over seeds),
  - `W_C > W_B` in at least 2/3 seeds,
  - `W_C > W_{B-prompt}` (fit cards beat strong prompting),
  - `W_C ≥ W_{B-budget}` (metadata is at least as good as “spend more inference budget”).
- **Pivot** if Fit Cards improves contacted-fit rate but welfare does not improve over B: interpret as “proposal-stage bias dominates discovery-stage errors” and pivot toward combining fit cards with a protocol-level delayed-acceptance mechanism.
- **Refute** if recovery < 0.2 or if `W_C ≤ W_{B-prompt}` (no evidence that platform-computed metadata matters beyond prompting).

---

## Impact Statement

If successful, this work would suggest that agentic marketplaces should treat **search result payload design** as a first-class mechanism-design tool: platform-computed, query-conditioned structured metadata can improve welfare at scale without restricting option set size or relying on brittle prompt engineering.

---

## References

- [Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets](./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/meta/meta_info.txt) - Bansal et al., 2025
- [The Agentic Economy](https://arxiv.org/abs/2505.15799) - Rothschild et al., 2025
- [Virtual Agent Economies](https://arxiv.org/abs/2509.10147) - Tomasev et al., 2025
- [What Is Your AI Agent Buying?](https://arxiv.org/abs/2508.02630) - Allouah et al., 2025
- [FaMA: LLM-Empowered Agentic Assistant for C2C Marketplace](https://arxiv.org/abs/2509.03890) - Yan et al., 2025
- [Project Vend](https://www.anthropic.com/research/project-vend-1) - Anthropic, 2024
- [NegotiationArena](https://arxiv.org/abs/2402.05863) - Bianchi et al., 2024
- [Deal or No Deal?](https://arxiv.org/abs/1706.05125) - Lewis et al., 2017
- [Decoupling Strategy and Generation in Negotiation Dialogues](https://arxiv.org/abs/1808.09637) - He et al., 2018
- [Auction Arena](https://arxiv.org/abs/2310.05746) - Richardson et al., 2023
- [GLEE](https://arxiv.org/abs/2410.05254) - Madmon et al., 2024
- [Economics Arena](https://arxiv.org/abs/2401.01735) - Sui et al., 2024
- [The AI Economist](https://arxiv.org/abs/2004.13332) - Zheng et al., 2020
- [ABIDES-Economist](https://arxiv.org/abs/2402.09563) - Dwarakanath et al., 2024
- [LLM Economist](https://arxiv.org/abs/2507.15815) - Karten et al., 2025
- [Algorithmic Collusion by LLMs](https://arxiv.org/abs/2404.00806) - Gonczarowski et al., 2024
- [When More Is Less: The Paradox of Choice in Search Engine Use](https://dl.acm.org/doi/10.1145/1571941.1572030) - Oulasvirta et al., 2009
- [Understanding Choice Overload in Recommender Systems](https://dl.acm.org/doi/10.1145/1864708.1864724) - Bollen et al., 2010
- [The Choice Overload Effect in Online Recommender Systems](https://pubsonline.informs.org/doi/10.1287/msom.2022.0659) - Long et al., 2024
- [The Amplification Paradox in Recommender Systems](https://arxiv.org/abs/2302.11225) - Ribeiro et al., 2023
- [Fairness and Diversity in Recommender Systems: A Survey](https://arxiv.org/abs/2307.04644) - Zhao et al., 2025
- [Diffusion Model for Slate Recommendation](https://arxiv.org/abs/2408.06883) - 2024
- [Attractive Metadata Attack](https://arxiv.org/abs/2508.02110) - Hu et al., 2025
- [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) - Anthropic, 2024
