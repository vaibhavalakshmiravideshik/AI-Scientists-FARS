# untitled

# Escrowed Batch Reveal: Testing Whether Sequential Proposal Visibility Causes First-Proposal Bias in Agentic Marketplaces

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language model (LLM) agents are increasingly deployed as intermediaries for economic actions: searching for services, negotiating, and committing to purchases or payments. When both sides of a marketplace are represented by autonomous agents (consumer assistants and service providers), small behavioral biases in how agents process information can scale into market-level distortions.

**Magentic Marketplace** is an open-source two-sided agentic marketplace environment that simulates a full transaction loop (search → messaging → structured order proposals → payment) and measures automated outcomes such as consumer welfare. A central finding is a severe and nearly universal **proposal-order bias**: across many models, the customer agent selects the **first** proposal it receives 60–100% of the time, while later proposals are almost never selected (often 0–7%). This creates a 10–30× advantage for response speed over price/quality, shifting incentives toward latency arms races.

A key open question for marketplace designers is whether this bias is primarily:
1) an **agent-internal anchoring/satisficing tendency** that persists even when all proposals are available, or
2) a consequence of **sequential information visibility** (the agent sees one proposal arrive earlier, and commits before it has a comparison set).

These imply different interventions. If (2) dominates, then marketplace protocol changes that delay *visibility* (not necessarily payment) could mitigate first-mover advantage without requiring model retraining.

### The Problem

Existing interventions in this repo’s proposal portfolio focus on changing what the agent is *allowed to do* (e.g., **quote-batched payment** gates payment until K proposals exist). However, gating payment does not isolate whether the bias is caused by **seeing** proposals sequentially.

In Magentic’s customer agent implementation, proposals are revealed to the LLM through the **message-fetch (“check_messages”) transcript**: each time the agent checks messages, newly received `order_proposal` messages are appended to an event history and rendered into the next LLM prompt. Even though timestamps are not shown, the sequential appearance of proposals in the transcript provides an implicit “arrival order” signal.

The research question is:

**Does sequential visibility of proposals cause first-proposal bias even when premature payment is prevented? Concretely, if we gate payment until K proposals (QuoteBatch) but still stream proposals sequentially, does the customer still disproportionately choose the earliest-arriving proposal, and does batch-revealing proposals eliminate this residual bias?**

### Key Insight and Hypothesis

**Key insight:** Even if the marketplace prevents *premature payment* (e.g., by requiring the customer to collect **K** proposals before paying), the customer agent still observes proposals **sequentially** through its `check_messages` transcript. This can create a primacy anchor: the first-seen proposal becomes a reference point that later proposals are compared against weakly, so the agent may still favor the earliest-arriving proposal even after all K proposals are available.

**Hypothesis:** In Magentic’s proposal-bias setting under a quote-batched payment constraint (payment blocked until **K=3** proposals have been received), an **escrowed batch reveal** protocol (buffering `order_proposal` messages and revealing them simultaneously in shuffled order) will reduce the probability that the **earliest-arriving** proposal is ultimately paid compared to payment gating alone, beyond what strong prompting and lightweight inference-time scaling achieve.

This could be wrong if (i) payment gating already eliminates first-proposal bias (so there is no residual effect to remove), (ii) bias simply shifts to the first *revealed* proposal within the batch (a generic list-position effect), or (iii) Magentic’s runtime already returns multiple proposals per `fetch_messages` call in practice, so sequential visibility is not the binding mechanism.

---

## Proposed Approach

### Overview

We propose **Escrowed Batch Reveal (EBR)**, a customer-side wrapper around message fetching, used in conjunction with a **quote-batched payment guardrail (HardGate)** to isolate visibility effects:

- The customer still calls `check_messages` as usual.
- Any incoming `order_proposal` messages are **buffered** in an escrow list.
- Until the escrow size reaches **K**, the `check_messages` result returned to the LLM contains **no `order_proposal` messages** (it may still contain other message types).
- When escrow reaches **K**, the next `check_messages` result returns **all K proposals at once**, in **shuffled order**, with timestamps already excluded by Magentic’s prompt renderer.

In the main experiment, conditions **B** and **C** both gate payment until ≥K proposals have been received (HardGate); the only difference is whether proposals are **streamed sequentially** (B) or **batch-revealed** (C). This makes the B→C difference attributable to visibility rather than premature commitment.

### Method Details

**Target environment:** Magentic Marketplace position/proposal bias experiments.

**K:** Default K=3 (matching the built-in proposal-bias experiment design).

**Implementation hook (verification-relevant):** implement EBR by **overriding `CustomerAgent.fetch_messages()`** (defined in `BaseSimpleMarketplaceAgent`) for the customer agent used in the experiment.

Why this is the right hook:
- In Magentic, the *LLM-visible transcript* is built from `_event_history`, and `CustomerAgent._execute_customer_action` appends the raw `fetch_messages()` return value (`FetchMessagesResponse`) into `_event_history` for rendering.
- Therefore, buffering only inside `proposal_storage` is insufficient to hide proposals: proposals must be removed from the `FetchMessagesResponse.messages` list **before** `_event_history.append((action, fetch_response))`.

Concrete path evidence (from the public Magentic repo):
- Customer agent calls `fetch_response = await self.fetch_messages(); self._event_history.append((action, fetch_response))` in `packages/magentic-marketplace/src/magentic_marketplace/marketplace/agents/customer/agent.py`.
- Prompt rendering for `check_messages` iterates over `FetchMessagesResponse.messages` and explicitly excludes timestamp-like fields such as `expiry_time` (see `.../customer/prompts.py`).

Implementation sketch:
- Maintain `self._ebr_buffer: list[ReceivedMessage]` and `self._ebr_K`.
- Call `super().fetch_messages()` to advance the internal `last_fetch_index` / `_seen_message_indexes` cursor.
- Split new messages into proposals vs non-proposals. Always return non-proposals.
- Append proposals into `_ebr_buffer`. If `len(_ebr_buffer) < K`, return `FetchMessagesResponse(messages=non_proposals, has_more=False)`.
- If `len(_ebr_buffer) >= K`, return `FetchMessagesResponse(messages=non_proposals + shuffle(_ebr_buffer[:K]), has_more=False)` and pop those K proposals from the buffer.

This makes proposal visibility batched while keeping the underlying server-side message arrival and DB logs unchanged.

### Key Innovations

1. **Mechanism isolation (visibility vs action gating):** tests whether *sequential proposal visibility* is causal, distinct from payment-gating interventions.
2. **Protocol-level, training-free intervention:** minimal engineering; no model changes required.
3. **Decision-relevant outcome:** determines whether marketplace designers should treat batch reveal as a first-line mechanism to reduce latency arms races.

---

## Related Work

### Field Overview

**Agentic marketplaces and economic simulations.** Recent benchmarks study LLM agents in economic settings including negotiations and two-sided markets. Magentic Marketplace operationalizes a realistic transaction lifecycle with automated welfare metrics and exposes emergent biases and vulnerabilities.

**Order/anchoring biases in LLM decision-making.** Separately, work on cognitive-bias-like behavior and on sequential decision procedures suggests that the order in which information appears in an LLM’s context can strongly influence outcomes.

**Mechanism design inspirations.** In traditional markets, batching (e.g., call auctions) is used to reduce timing advantages. Agentic marketplaces provide a new setting where similar protocol choices can be evaluated automatically.

### Related Papers

- **[Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets](../../papers/paper_summaries/Magentic%20Marketplace%20An%20Open-Source%20Environment%20for%20Studying%20Agentic%20Markets.md)**: Two-sided agentic marketplace benchmark; reports extreme first-proposal bias.
- **[MAGENTIC MARKETPLACE AN OPEN-SOURCE ENVIRONMENT FOR STUDYING AGENTIC MARKETS (Anonymous)](../../papers/paper_summaries/MAGENTIC%20MARKETPLACE%20AN%20OPEN-SOURCE%20ENVI-RONMENT%20FOR%20STUDYING%20AGENTIC%20MARKETS%20Anonymous.md)**: OpenReview version of Magentic Marketplace.
- **[The Agentic Economy](https://arxiv.org/abs/2505.15799)**: Conceptual framing for two-sided agentic markets.
- **[Virtual Agent Economies](../../papers/paper_summaries/Virtual%20Agent%20Economies.md)**: Position paper on agent economies and market infrastructure.
- **[Large Language Models as Simulated Economic Agents (Homo Silicus)](https://dl.acm.org/doi/10.1145/3670865.3673566)**: Studies LLM behavior in economic settings.
- **[NegotiationArena](https://arxiv.org/abs/2402.05863)**: Negotiation benchmark for LLM agents.
- **[Economics Arena for Large Language Models](https://arxiv.org/abs/2401.01735)**: Benchmark for economic reasoning and decision-making.
- **[Put Your Money Where Your Mouth Is: Auction Arena](https://arxiv.org/abs/2310.05746)**: Evaluates strategic planning/execution of LLM agents in auctions.
- **[Language Models as Auction Participants](https://openreview.net/pdf/665ce75262e3c2f8a35c3dc83267ceb9788e2242.pdf)**: Experimental-econ style evaluation of LMs in auctions.
- **[Algorithmic collusion by large language models](https://arxiv.org/abs/2404.00806)**: Studies collusion risks in LLM-mediated markets.
- **[Vertical tacit collusion in AI-mediated markets](https://arxiv.org/abs/2601.03061)**: Shows super-additive consumer harm when multiple market distortions interact.
- **[Fit Cards for Agentic Marketplace Search](../marketplace-search-fit-cards-scaling/proposal.md)**: Platform-side search payload augmentation to recover welfare at large consideration sets.
- **[Quote-Batched Payment](../quote-batched-payment-proposal-bias/proposal.md)**: Protocol constraint that blocks payment until K proposals exist.
- **[Buffered Checkout Commit](../buffered-checkout-commit/proposal.md)**: Context-neutral commit pattern for irreversible actions in agents.
- **[Anchors in the Machine: Behavioral and Attributional Evidence of Anchoring Bias in LLMs](https://arxiv.org/abs/2511.05766)**: Provides behavioral and attributional evidence that LLM decisions can be anchored by early information, motivating structural debiasing beyond instructions.
- **[Anchoring Bias in Large Language Models: An Experimental Study](https://arxiv.org/abs/2412.06593)**: Controlled experiments showing anchoring persists under common debiasing prompts (e.g., CoT, reflection), supporting protocol-level interventions.
- **[Fragile Preferences: A Deep Dive Into Order Effects in Large Language Models](https://arxiv.org/abs/2506.14092)**: Systematic study of order effects (primacy/recency/centrality) in LLM choice tasks; relevant for understanding bias under batched-but-ordered proposal presentation.
- **[Serial Position Effects of Large Language Models](https://arxiv.org/abs/2406.15981)**: Documents primacy/recency effects in LLMs and partial mitigation via prompting; relevant to sequential vs simultaneous proposal exposure.
- **[The High-Frequency Trading Arms Race: Frequent Batch Auctions as a Market Design Response](https://ericbudish.org/publication/the-high-frequency-trading-arms-race-frequent-batch-auctions-as-a-market-design-response/)**: Classic market-design argument that batching reduces latency arms races; conceptual motivation for batching proposals in agentic markets.
- **[G-Eval](https://arxiv.org/abs/2303.16634)**: LLM-based evaluation framing; relevant as evaluation infrastructure.
- **[Generative Agents](https://arxiv.org/abs/2304.03442)**: Simulation of agent behaviors; contextual foundation.
- **[GLEE: A unified framework for language-based economic environments](https://arxiv.org/abs/2410.05254)**: Economic environments suite.
- **[The AI Economist](https://arxiv.org/abs/2004.13332)**: Classic multi-agent economic simulation framework.
- **[ABIDES-Economist](https://arxiv.org/abs/2402.09563)**: Agent-based market simulator (microstructure inspiration).

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Two-sided agentic marketplaces | Full lifecycle (search→negotiation→transaction) with automated welfare | Magentic Marketplace | Welfare, bias rates | Synthetic domains; short horizons |
| Protocol/mechanism interventions | Change action availability or information visibility | QuoteBatch, Fit Cards, Buffered Commit | Market-level metrics | Hard to isolate mechanisms |
| Auction/negotiation arenas | Economic games for LLM agents | Auction Arena, NegotiationArena, Economics Arena | Task success, revenue | Often one-sided or simplified |

### Closest Prior Work

1. **Magentic Marketplace**: identifies proposal bias but does not isolate whether sequential visibility is causal.
2. **Quote-Batched Payment**: tests payment gating (action availability) but does not test whether sequential proposal visibility alone is sufficient to cause the bias.

**Novelty Kill Search Summary:** Searched for “Magentic Marketplace proposal bias mitigation protocol”, “agentic marketplace first proposal bias batching proposals”, “simultaneous reveal proposals LLM agent”, and “commit-reveal proposals LLM agent” (WebSearch + AnnaResearch). No prior work found that evaluates **batch/simultaneous reveal of order proposals** as a protocol intervention for Magentic-style proposal bias as of 2026-03-01. Full query log in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Magentic Marketplace | Measures proposal bias | Does not isolate causality | Hide proposals until batch | Separates sequential-visibility effects from intrinsic anchoring |
| Quote-Batched Payment | Blocks payment until K proposals | Still reveals proposals sequentially | Change visibility, not action set | Tests whether visibility alone drives bias |
| Fit Cards | Improves search payload | Targets discovery, not proposal stage | Focus on proposal reveal protocol | Addresses latency arms race directly |

---

## Experiments

### Experimental Setup

**Benchmark / environment:** Magentic Marketplace proposal-bias scenarios `data/position_bias/contractors_{first,second,third}` executed via `experiments/position/run_n_experiments.py`.

**Conditions (3 total; main experiment; K=3):**
- **A. SoftWait (prompt-only; sequential reveal):** modify only the customer system prompt to require waiting for K proposals and explicitly comparing before paying (no code enforcement).
- **B. QuoteBatch / HardGate (payment gated; sequential reveal):** enforce “no payment until ≥K proposals received” by blocking `pay` messages until `proposal_storage.count_proposals() >= K` (use a constant non-informative failure string such as `ACTION_UNAVAILABLE`). Use the same SoftWait prompt as (A).
- **C. QuoteBatch + Escrowed Batch Reveal (ours):** same HardGate as (B), plus override `CustomerAgent.fetch_messages()` to buffer `order_proposal` messages until K collected, then reveal all K in one `check_messages` result in shuffled order.

(Informal glossary: **SoftWait** = prompt-only instruction to wait for K proposals (no enforcement); **HardGate** = code-level payment blocking until K proposals exist; **EBR** = code-level batching/shuffling of proposal *visibility* via `fetch_messages()`.)

**Baseline Ladder (minimum):**
- **Trivial baseline:** offline random-choice among the received proposals at the end of a completed run (expected ~33% earliest-arrival chosen if 3 proposals arrive and the agent is unbiased).
- **Prompting baseline:** SoftWait (Condition A).
- **Inference-time scaling baseline (lightweight):** SoftWait + increased decision compute at payment time: sample the final `send_messages` action **N=5 times** and choose the pay action that selects the lowest-price proposal among the proposals already visible.
- **Closest existing method:** QuoteBatch / HardGate (Condition B) — an action-gating protocol constraint that enforces a quote-collection phase but still streams proposals sequentially.

**Base models (API; do not consume local GPU budget):**

| Model | Notes |
|---|---|
| `gemini-2.5-flash` | Strong, cost-effective model; included in Magentic’s paper experiments |
| `claude-sonnet-4-5` | Second strong model for confirmation if budget allows |

(If API budget is constrained, run the decisive experiment on a single model first, then confirm on the second.)

**Training Data (if applicable):**

No training data needed - inference only.

**Important controls:**
- Run across `contractors_first`, `contractors_second`, `contractors_third` so each business profile occupies each arrival position across the suite (mitigates the confound that the fastest responder is intrinsically better).

**Resource Estimate**:
- **Compute budget**: ~0 GPU-hours (API inference).
- **API usage (order-of-magnitude):** For this 1-customer/3-business setting, expect ~20–60 LLM calls per run. With 3 conditions × 1–2 models × (3 folders × 10 repetitions) ≈ 90–180 runs, total calls ≈ 1.8k–10.8k.
- **Wall-clock:** dominated by API latency; expected to finish within hours.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Magentic proposal-bias (contractors) | 1 customer interacts with 3 contractors; measure proposal-order effects | (1) Earliest-arrival chosen rate, (2) Rank histogram, (3) Utility conditional on completion, (4) Completion rate | N/A | https://github.com/microsoft/multi-agent-marketplace | `experiments/position/generate_proposal_data.py` + sqlite export |

### Main Results

#### Comparability Rules (CRITICAL)

- Same dataset folders (`contractors_{first,second,third}`), same `customer_max_steps`, same search settings.
- Same base model and temperature across conditions.
- Only differences: (i) payment gating off vs on (A vs B/C) and (ii) proposal visibility streaming vs batched (B vs C). B and C share the same SoftWait prompt to avoid prompt confounds.

#### Results Table

| Method | Base Model | Benchmark | Earliest-arrival chosen (mean±std) | Completion rate (mean±std) | Source | Notes |
|---|---|---|---:|---:|---|---|
| Magentic reported (range) | mixed | proposal bias | 0.60–1.00 | N/A | Magentic Sec. 5.4 | Range across models/scenarios |
| **A: SoftWait (prompt-only)** | gemini-2.5-flash | contractors_{first,second,third} | **TBD** | **TBD** | - | Sequential reveal |
| **B: QuoteBatch / HardGate** | gemini-2.5-flash | contractors_{first,second,third} | **TBD** | **TBD** | - | Payment gated, sequential reveal |
| **C: QuoteBatch + EBR** | gemini-2.5-flash | contractors_{first,second,third} | **TBD** | **TBD** | - | Payment gated, batched reveal |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| Reveal-order bias check (analysis-only) | In C, compute which *reveal position* (1/2/3 in the shuffled batch) was paid (i.e., whether the **first/second/third presented** proposal in the batch was chosen) | If bias shifts from arrival-order to presentation-order, earliest-arrival rate drops but **first-presented selection** remains high |

### Experimental Rigor

**Variance & reproducibility:**
- Run each condition with **10 independent repetitions** per scenario (API sampling is not seedable; repetitions serve as empirical variance estimates).
- Fix decoding settings (e.g., `LLM_TEMPERATURE=0.7`) and other environment variables across all conditions.

**Validity & controls:**
- **Quality confound (fast responder better):** mitigated by evaluating across the 3 rotated scenarios.
- **Prompt leakage of timing:** timestamps are not rendered in prompts (customer prompt renderer excludes time fields); arrival order enters mainly through sequential message visibility.
- **Batching already happens:** gate analysis step: verify baseline earliest-arrival rate is high (>0.5) for the chosen model; otherwise refute/pivot (no bias to fix).
- **Reveal timing:** in Condition C, reveal the proposal batch immediately when the escrow buffer first reaches K (no additional fixed delay), so the B→C difference isolates visibility/presentation rather than time-pressure heuristics.

---

## Success Criteria

**Hypothesis (directional):** Escrowed batch reveal reduces earliest-arrival chosen rate beyond prompt-only SoftWait.

**Decision Rule (concrete):**
- **Proceed/accept claim:** For at least one strong model, `EarliestArrival(C) ≤ EarliestArrival(B) − 0.20` averaged across the three contractor scenarios, and completion rate drop is ≤10 percentage points vs B.
- **Pivot:** If `EarliestArrival(C) < EarliestArrival(B)` but the paid proposal is almost always the *first revealed in the batch* (bias shifts from arrival-order to presentation-order), pivot to adding a forced comparison step at pay time.
- **Refute:** If `EarliestArrival(C) ≥ EarliestArrival(B) − 0.10` (within noise), conclude that after payment is gated, sequential visibility is not a dominant remaining driver in this setting; residual bias is likely intrinsic or list-position-driven.

---

## Impact Statement

If successful, this provides a concrete design rule for agentic marketplaces: **do not stream competing proposals to consumer agents; batch and reveal simultaneously** to prevent latency from dominating competition. If unsuccessful, the negative result is still decision-changing: it suggests that proposal bias is primarily an agent-internal satisficing tendency that requires stronger interventions (e.g., enforced comparison or action gating), not just message-visibility changes.

---

## References

- [Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets](../../papers/paper_summaries/Magentic%20Marketplace%20An%20Open-Source%20Environment%20for%20Studying%20Agentic%20Markets.md) - Bansal et al., 2025
- [MAGENTIC MARKETPLACE AN OPEN-SOURCE ENVIRONMENT FOR STUDYING AGENTIC MARKETS (Anonymous)](../../papers/paper_summaries/MAGENTIC%20MARKETPLACE%20AN%20OPEN-SOURCE%20ENVI-RONMENT%20FOR%20STUDYING%20AGENTIC%20MARKETS%20Anonymous.md) - OpenReview, ICLR 2026 submission
- [The Agentic Economy](https://arxiv.org/abs/2505.15799) - Rothschild et al., 2025
- [Virtual Agent Economies](../../papers/paper_summaries/Virtual%20Agent%20Economies.md) - Tomasev et al., 2025
- [Large language models as simulated economic agents](https://dl.acm.org/doi/10.1145/3670865.3673566) - Filippas et al., 2024
- [NegotiationArena](https://arxiv.org/abs/2402.05863) - Bianchi et al., 2024
- [Economics Arena for Large Language Models](https://arxiv.org/abs/2401.01735) - Sui et al., 2024
- [Put Your Money Where Your Mouth Is: Auction Arena](https://arxiv.org/abs/2310.05746) - Richardson et al., 2023
- [Language Models as Auction Participants](https://openreview.net/pdf/665ce75262e3c2f8a35c3dc83267ceb9788e2242.pdf) - ICLR 2026
- [Algorithmic collusion by large language models](https://arxiv.org/abs/2404.00806) - Gonczarowski et al., 2024
- [Vertical tacit collusion in AI-mediated markets](https://arxiv.org/abs/2601.03061) - Affonso et al., 2026
- [The AI Economist](https://arxiv.org/abs/2004.13332) - Zheng et al., 2020
- [ABIDES-Economist](https://arxiv.org/abs/2402.09563) - Dwarakanath et al., 2024
- [GLEE](https://arxiv.org/abs/2410.05254) - Madmon et al., 2024
- [Generative Agents](https://arxiv.org/abs/2304.03442) - Park et al., 2023
- [Fit Cards for Agentic Marketplace Search](../marketplace-search-fit-cards-scaling/proposal.md) - finalized proposal (this repo)
- [Quote-Batched Payment](../quote-batched-payment-proposal-bias/proposal.md) - finalized proposal (this repo)
- [Buffered Checkout Commit](../buffered-checkout-commit/proposal.md) - finalized proposal (this repo)

- [Anchors in the Machine: Behavioral and Attributional Evidence of Anchoring Bias in LLMs](https://arxiv.org/abs/2511.05766) - 2025
- [Anchoring Bias in Large Language Models: An Experimental Study](https://arxiv.org/abs/2412.06593) - Lou & Sun, 2024
- [Fragile Preferences: A Deep Dive Into Order Effects in Large Language Models](https://arxiv.org/abs/2506.14092) - Yin et al., 2025
- [Serial Position Effects of Large Language Models](https://arxiv.org/abs/2406.15981) - Guo & Vosoughi, 2024
- [The High-Frequency Trading Arms Race: Frequent Batch Auctions as a Market Design Response](https://ericbudish.org/publication/the-high-frequency-trading-arms-race-frequent-batch-auctions-as-a-market-design-response/) - Budish et al., 2015