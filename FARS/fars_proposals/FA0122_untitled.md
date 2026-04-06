# untitled

# Quote-Batched Payment: A Protocol Constraint to Reduce First-Proposal Bias in Agentic Marketplaces

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Large language model (LLM) agents are increasingly used to execute economic actions on behalf of users, such as searching for services, negotiating details, and completing purchases. This creates a new class of “agentic marketplaces”, where automated buyer agents interact with automated seller agents through a platform protocol.

A key risk in such markets is that even small behavioral biases in buyer agents can translate into systemic market distortions. For example, if buyers strongly prefer offers that arrive first, competition shifts from price/quality to response latency, incentivizing sellers to invest in speed rather than service quality. This is analogous to well-studied problems in market microstructure where timing advantages can dominate fundamental value.

Magentic Marketplace is an open-source simulation environment (released by Microsoft Research in 2025) for studying two-sided agentic markets (consumer assistants and service agents) under controlled protocols and objective utility metrics. It reports a striking behavioral phenomenon: across multiple models, **the first proposal received is selected at rates between 60–100%, while third proposals are selected at near-zero rates** (often 0–7%), yielding a reported **10–30× first-mover advantage** in acceptance probability (a ratio of selection probabilities; e.g., 60% vs 2% ≈ 30×), despite the customer agent prompt explicitly recommending “wait for proposals” and “compare all proposals” before paying.

### The Problem

**Problem-order bias in agentic markets.** In Magentic Marketplace’s “proposal bias” experiment, the buyer’s payment action can be taken at any time, including immediately after the first acceptable proposal arrives. This creates a natural stopping point for the agent: once a feasible proposal appears, paying ends the need to reason about additional offers.

While this behavior is often described as “anchoring”, there are at least two distinct mechanisms:

1. **Premature commitment under partial information**: the agent commits because the environment allows an irreversible action (payment) before a comparison set is available.
2. **Anchoring given full information**: even after seeing multiple proposals, the first proposal disproportionately influences the final decision.

These mechanisms imply different interventions. If premature commitment is dominant, a protocol-level constraint that delays payment availability should largely remove first-proposal advantages. If anchoring dominates, delaying payment may not help.

Prompting alone is an especially important baseline here: Magentic’s customer prompt already instructs the agent to wait and compare, yet the bias remains severe. This suggests that purely instruction-based fixes may be insufficient and that protocol constraints are a plausible lever for marketplace designers.

### Key Insight and Hypothesis

**Key insight:** Proposal-order bias may be driven less by “irrational preference for the first offer” and more by **the availability of an irreversible “pay” action before the agent has accumulated a minimally adequate comparison set**. In other words, the market protocol may be implicitly rewarding early termination.

**Hypothesis:** If the marketplace enforces a short “quote collection” phase—by blocking payment until the customer has received at least **K** order proposals—then the probability that the customer pays the first-arriving proposal will drop substantially (relative to both the default agent and a prompt-only “wait for K quotes” baseline).

This hypothesis is genuinely uncertain because the bias could persist even after all proposals arrive (pure anchoring), in which case delaying payment will not reduce first-proposal selection rates.

---

## Proposed Approach

### Overview

We propose a **quote-batched payment protocol** for agentic marketplaces:

- The marketplace (or client SDK) defines a parameter **K** (minimum number of proposals to collect).
- The buyer agent is prevented from completing the payment action until at least **K** proposals have been received.

Operationally, we implement this as a lightweight client-side guardrail in Magentic Marketplace’s customer agent: “pay” messages are intercepted and rejected until the proposal storage contains at least K proposals.

### Method Details

**Setting:** Magentic Marketplace customer agents send payments as a message type (`payment`) inside the `send_messages` action. The customer agent stores incoming `order_proposal` messages in an `OrderProposalStorage`.

**Hard-gate mechanism (QuoteBatch):**

- Add a configurable `min_quotes = K` and `enforce_min_quotes = True` to the customer agent (read from environment variables or passed at initialization).
- When executing a `send_messages` action, before sending any `pay` message:
  - If `proposal_storage.count_proposals() < K`, do **not** send the payment.
  - Return a **constant, non-informative failure string** (e.g., `ACTION_UNAVAILABLE`) as the per-payment result, so the event-history log does not leak extra guidance beyond “not allowed yet”.

**Soft prompt baseline (SoftWait):**

- Modify only the customer’s system prompt to explicitly state: “Do not pay until you have received at least K order proposals; if fewer than K proposals have arrived, continue checking messages and/or contacting additional businesses.”
- Do not enforce this constraint in code.

**Why a constant error string matters:** In Magentic, failed pay attempts are written into the agent’s future prompt via the “Action Trajectory” (e.g., `Message failed to send: <error>`). If the error message contains rich content, it becomes an additional prompt intervention. Using a constant placeholder reduces this confound.

### Key Innovations

- **Protocol-level intervention rather than model fine-tuning**: we test whether a marketplace rule can mitigate bias without changing the underlying LLM.
- **Mechanism isolation via a 3-condition design**: baseline vs prompt-only waiting vs enforced waiting distinguishes “can the agent self-regulate?” from “must the environment enforce?”
- **A minimal, reusable implementation hook**: in Magentic Marketplace, payment gating is implementable by intercepting pay messages inside the customer agent, without changing server APIs.

---

## Related Work

### Field Overview

**Agentic marketplaces and economic simulations.** Recent work has introduced platforms to study LLM agents as economic actors, including negotiation arenas, auctions, and market simulations. Magentic Marketplace provides an end-to-end protocol (search → message → proposal → payment) with controlled data generation and automatic utility metrics, enabling the study of emergent market distortions.

**Biases in sequential decision-making by LLMs.** A growing literature documents cognitive-bias-like patterns in LLM decisions, including anchoring effects where early information disproportionately influences final judgments. In multi-agent marketplaces, such biases can be amplified by protocol timing (when decisions become irreversible) and by context-window dynamics.

**Mechanism design and protocol constraints.** In economics and market microstructure, protocol choices (e.g., continuous vs batch auctions) can change incentives and reduce timing-based arms races. For agentic markets, protocol design is under-explored compared to agent training; Magentic explicitly calls for iterative market-mechanism experimentation.

### Related Papers

- **[Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets](./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/meta/meta_info.txt)**: Introduces a two-sided LLM marketplace simulator and reports severe first-proposal bias (60–100% first-proposal selection).
- **[Vertical tacit collusion in AI-mediated markets](./references/Vertical-tacit-collusion-in-AI-mediated-markets/meta/meta_info.txt)**: Shows how marketplace biases (e.g., position bias) can be strategically exploited and analyzes debiasing interventions.
- **[BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](./references/BiasBusters-Uncovering-and-Mitigating-Tool-Selection-Bias-in-Large-Language-Models/meta/meta_info.txt)**: Studies position bias in tool marketplaces and mitigates it via filtering + uniform sampling.
- **[Anchors in the Machine: Behavioral and Attributional Evidence of Anchoring Bias in LLMs](./references/Anchors-in-the-Machine-Behavioral-and-Attributional-Evidence-of-Anchoring-Bias-in-LLMs/meta/meta_info.txt)**: Provides controlled measurements and attribution analyses of anchoring-like effects in LLMs.
- **[What Is Your AI Agent Buying?](./references/What-Is-Your-AI-Agent-Buying-Evaluation,-Implications,-and-Emerging-Questions-for-Agentic-E-Commerce/meta/meta_info.txt)**: Evaluates agentic e-commerce behaviors and highlights emerging failure modes and implications.
- **[AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions](./references/AgenticPay-A-Multi-Agent-LLM-Negotiation-System-for-Buyer-Seller-Transactions/meta/meta_info.txt)**: Benchmarks negotiation in buyer–seller settings, complementary to marketplace-protocol studies.
- **[LLM-Deliberation](https://arxiv.org/abs/2305.19118)**: Uses interactive negotiation games to evaluate LLM decision-making in multi-agent settings.
- **[Cooperation, competition, and maliciousness: LLM-stakeholders interactive negotiation](https://arxiv.org/abs/2402.01918)**: Studies strategic multi-agent negotiation with cooperative and adversarial behaviors.
- **[NegotiationArena](https://arxiv.org/abs/2402.05863)**: Provides a platform and analysis for LLM negotiation capabilities.
- **[Put Your Money Where Your Mouth Is: Evaluating strategic planning and execution of LLM agents in an auction arena](https://arxiv.org/abs/2310.05746)**: Evaluates LLM agents in auctions where timing/strategy matter.
- **[Algorithmic collusion by large language models](https://arxiv.org/abs/2404.00806)**: Demonstrates risks of strategic behavior and collusion in LLM-mediated markets.
- **[STEER: Assessing the economic rationality of large language models](https://arxiv.org/abs/2402.09552)**: Evaluates whether LLMs behave consistently with economic rationality criteria.
- **[Economics Arena for Large Language Models](https://arxiv.org/abs/2401.01735)**: Provides a benchmark suite for economic behaviors of LLM agents.
- **[GLEE: A unified framework and benchmark for language-based economic environments](https://arxiv.org/abs/2410.05254)**: Benchmarks language-based economic environments for agent evaluation.
- **[ABIDES-Economist](https://arxiv.org/abs/2402.09563)**: Uses agent-based simulation for economic systems, highlighting the value of protocol and market-structure choices.
- **[The AI Economist](https://arxiv.org/abs/2004.13332)**: Uses multi-agent simulation and mechanism design (tax policies) to shape economic outcomes.
- **[Generative Agents](https://arxiv.org/abs/2304.03442)**: Shows how LLM agents can produce realistic multi-agent dynamics, motivating careful protocol design.
- **[Homo Silicus / LLMs as simulated economic agents](https://dl.acm.org/doi/10.1145/3670865.3673566)**: Discusses what can be learned from LLM economic simulacra and their limitations.
- **[The agentic economy](https://arxiv.org/abs/2505.15799)**: Frames economic impacts of agentic systems, motivating robust marketplace design.
- **[Virtual agent economies](https://arxiv.org/abs/2509.10147)**: Surveys and frames risks/opportunities of AI-driven economies.
- **[Qwen3 technical report](https://arxiv.org/abs/2505.09388)**: Provides details on an open model family used in many agentic evaluations.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Agentic marketplace simulators | End-to-end protocols for search, communication, proposals, and transactions | Magentic Marketplace; GLEE | Market utility, welfare, bias metrics | Emergent behaviors can depend on protocol details |
| LLM economic behavior benchmarks | Evaluate negotiation/auction/rationality behaviors | NegotiationArena; STEER; Economics Arena | Task success, utility, strategy metrics | Often simplified protocols; limited mechanism interventions |
| Bias measurement in LLM decisions | Controlled tasks to measure anchoring/position/ordering effects | Anchors in the Machine; BiasBusters | Selection-rate skew, bias scores | Mostly evaluates, fewer protocol-level fixes |
| Market mechanism design (protocol constraints) | Change protocol rules to alter incentives and reduce timing races | Call/batch auction literature; AI Economist | Welfare, efficiency, fairness | Not widely applied to LLM agent marketplaces |

### Closest Prior Work

**Magentic Marketplace** directly measures proposal-order bias and argues it can shift competition toward latency, but does not test protocol-level mitigations beyond highlighting the issue.

**BiasBusters** addresses position bias in tool selection by altering the selection mechanism (filtering + uniform sampling), but focuses on a tool list rather than a transactional market protocol with irreversible payments.

**Anchors in the Machine** provides evidence that LLMs exhibit anchoring-like behavior in controlled tasks, but does not test marketplace protocol interventions.

**Vertical tacit collusion in AI-mediated markets** analyzes strategic exploitation of agent biases and shows debiasing can alter market power, but does not study payment timing constraints for buyers.

**Novelty Kill Search Summary:** Searched for “Magentic Marketplace proposal bias mitigation”, “first-proposal bias LLM agent mitigation”, “batch auction agentic markets”, “cooling-off period LLM buyer agent”, and checked for papers combining quote batching with LLM marketplaces. No close prior work enforcing “pay only after K quotes” for LLM buyer agents was found as of 2026-02-17 (full query log in `notes.md`).

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| Magentic Marketplace | Measures proposal-order bias in an agentic market | Does not test protocol mitigations | Add a payment-timing constraint + prompt-vs-enforcement comparison | Directly tests whether protocol rules reduce first-mover advantage |
| BiasBusters | Mitigates position bias in tool selection | Not a transaction protocol; no irreversible payment stage | Apply a protocol constraint at the irreversible action boundary | Timing constraints target the hypothesized premature-commitment mechanism |
| Anchors in the Machine | Measures anchoring in LLM decisions | Not in interactive market setting; no mitigation | Use market protocol as mitigation lever | Provides a realistic test of whether anchoring vs commitment drives bias |
| Vertical tacit collusion | Studies exploitation of agent biases by sellers | Focuses on ranking/position; not buyer payment timing | Constrain buyer commitment timing | Reduces incentives for seller latency manipulation |

---

## Experiments

### Experimental Setup

**Environment:** Magentic Marketplace open-source repository (client/server simulation with postgres logging and sqlite export).

**Task / Dataset:** Use the existing proposal-bias data folder with a single customer and three businesses:

- `data/position_bias/contractors_first/` (1 customer, 3 businesses)

This setting matches Magentic’s proposal-bias measurement setup and allows direct measurement of the rank (by arrival time) of the accepted proposal.

**Conditions (3 total; main experiment):**

- **A. Baseline:** Unmodified customer agent and prompt (as in Magentic).
- **B. SoftWait:** Add explicit prompt constraint “do not pay until ≥K proposals” (K=3), but do not enforce.
- **C. QuoteBatch (HardGate):** Same SoftWait prompt, plus enforcement: block pay until ≥K proposals are stored.

**Baseline Ladder (REQUIRED):**

- **Trivial baseline:** Randomly choose among the received proposals at the end of the run (offline; expected ~33% for each rank if 3 proposals arrive).
- **Prompting baseline:** SoftWait (Condition B).
- **Inference-time scaling baseline (lightweight):** SoftWait with increased deliberation budget at the pay decision by sampling the final `send_messages` action **N=5 times** and selecting the pay action that chooses the lowest-priced available proposal (if multiple pay choices appear). This tests whether additional compute at decision time can substitute for protocol enforcement.
- **Closest existing method:** None directly targets payment timing in agentic marketplaces; Magentic Marketplace is the closest benchmarked prior.

**Base Models:** (API-based; do not consume local GPU budget)

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| `gemini-2.5-flash` | proprietary | (API; see `available_models.md`) | Included in Magentic’s experiments |
| `claude-sonnet-4-5` | proprietary | (API; see `available_models.md`) | Included in Magentic’s experiments |

(If API budget is constrained, run the decisive experiment on a single model first, then confirm on the second.)

**Training Data (if applicable):**

No training data needed – inference only.

**Resource Estimate**:

- **Compute budget**: ~0 GPU-hours (API inference only)
- **API usage (order-of-magnitude):** For this 1-customer/3-business setting, expect ~20–60 LLM calls per run (customer decisions + business replies). With 3 conditions × 2 models × 10 runs ≈ 60 runs, total calls ≈ 1.2k–3.6k (within typical API experimentation budgets).
- **Wall-clock:** dominated by API latency; expected to finish within hours.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| Magentic proposal-bias scenario (`position_bias/contractors_first`) | One customer interacts with three businesses; measure proposal-order effects | (1) First-proposal selection rate, (2) Rank distribution, (3) Completion rate, (4) Steps-to-payment | N/A | https://github.com/microsoft/multi-agent-marketplace | `experiments/position/generate_proposal_data.py` (minor extension to tag protocol condition in filenames) |

**Primary metric:**

- **First-proposal selection rate**: fraction of runs where the paid proposal has `chosen_proposal_rank = 1` (ranked by proposal arrival time from sqlite logs).

**Secondary metrics:**

- **Rank distribution** over {1,2,3}
- **Completion rate**: fraction of runs with a payment made before `customer_max_steps`
- **Overhead**: mean number of customer steps and LLM calls per completed transaction

### Main Results

#### Results Table

| Method | Base Model | Benchmark | First-proposal chosen (mean±std) | 3rd-proposal chosen (mean±std) | Source | Notes |
|--------|------------|-----------|----------------------------------|--------------------------------|--------|-------|
| Baseline (Magentic, reported) | mixed | Magentic proposal-bias | 60–100% | near-zero (often 0–7%) | [Magentic](./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/meta/meta_info.txt) | Reported as ranges in Sec. 5.4 |
| **Baseline (A)** | `gemini-2.5-flash` | contractors_first | **TBD** | **TBD** | - | To be verified |
| SoftWait (B) | `gemini-2.5-flash` | contractors_first | **TBD** | **TBD** | - | Prompt-only |
| QuoteBatch / HardGate (C) | `gemini-2.5-flash` | contractors_first | **TBD** | **TBD** | - | Enforced K=3 |
| **Baseline (A)** | `claude-sonnet-4-5` | contractors_first | **TBD** | **TBD** | - | To be verified |
| SoftWait (B) | `claude-sonnet-4-5` | contractors_first | **TBD** | **TBD** | - | Prompt-only |
| QuoteBatch / HardGate (C) | `claude-sonnet-4-5` | contractors_first | **TBD** | **TBD** | - | Enforced K=3 |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| QuoteBatch (K=2) | Reduce required proposals from 3 → 2 | Tests sensitivity: smaller K should still reduce bias but less strongly |
| QuoteBatch (informative error) | Replace constant `ACTION_UNAVAILABLE` with descriptive error | Should not be needed; if it improves results, the “error as reminder” confound is important |

### Experimental Rigor

**Variance & Reproducibility:**

- Run each condition with **10 independent repetitions** per model (API sampling is not seedable; repetitions serve as empirical variance estimates).
- Fix `LLM_TEMPERATURE` (e.g., 0.7) and other environment variables across all conditions.

**Validity & Controls:**

- **Confound 1 (prompt differences):** QuoteBatch (C) uses the same SoftWait prompt as (B); the only additional change is enforcement.
- **Confound 2 (block message as extra instruction):** Use a constant non-informative error string (`ACTION_UNAVAILABLE`); optionally ablate with an informative error to quantify this channel.
- **Confound 3 (proposal non-arrival):** Track completion rate; if completion drops materially, reduce K or add a fallback “pay allowed after max_wait_steps” and report it as a pivot.

**Sanity checks:**

- Random-choice baseline yields ~33% for each rank when 3 proposals arrive.
- Reproduce the qualitative Magentic claim that baseline agents strongly prefer the first-arriving proposal.

### Analysis (Optional)

- **Attempted premature pay frequency:** how often the agent tries to pay before K proposals in QuoteBatch.
- **Behavioral trace inspection:** whether the agent explicitly compares proposals in text before paying (simple keyword-based analysis; no human labels).

---

## Success Criteria

**Hypothesis** (directional — what you expect):

- SoftWait will reduce first-proposal selection slightly or not at all (since the base prompt already advises waiting).
- QuoteBatch will substantially reduce first-proposal selection and increase selection of later proposals, indicating that premature commitment is a major driver.

**Decision Rule** (concrete — when to stop):

- **Continue/Proceed:** QuoteBatch reduces first-proposal selection rate by **≥20 percentage points** vs SoftWait on at least one model, and does not reduce completion rate by more than **5 pp**.
- **Pivot:** If SoftWait matches QuoteBatch (difference <10 pp), the bias is largely prompt-fixable; pivot to cheaper prompt templates or to anchoring-focused interventions (e.g., shuffling proposal presentation once K is reached).
- **Refute:** If QuoteBatch shows <10 pp improvement vs SoftWait (or worsens completion materially), then delaying payment is not the dominant mechanism for proposal bias in this setting.

---

## Impact Statement

If successful, this work provides a concrete, low-engineering-cost rule that marketplace designers can adopt: **delay irreversible actions until a minimal comparison set exists**. This would reduce incentives for sellers to compete primarily on latency and could improve fairness and market efficiency in real agent-mediated service marketplaces.

---

## References

- [Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets](./references/Magentic-Marketplace-An-Open-Source-Environment-for-Studying-Agentic-Markets/meta/meta_info.txt) - Microsoft, 2025
- [Vertical tacit collusion in AI-mediated markets](./references/Vertical-tacit-collusion-in-AI-mediated-markets/meta/meta_info.txt) - Affonso et al., 2026
- [BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](./references/BiasBusters-Uncovering-and-Mitigating-Tool-Selection-Bias-in-Large-Language-Models/meta/meta_info.txt) - 2025
- [Anchors in the Machine: Behavioral and Attributional Evidence of Anchoring Bias in LLMs](./references/Anchors-in-the-Machine-Behavioral-and-Attributional-Evidence-of-Anchoring-Bias-in-LLMs/meta/meta_info.txt) - 2025
- [What Is Your AI Agent Buying?](./references/What-Is-Your-AI-Agent-Buying-Evaluation,-Implications,-and-Emerging-Questions-for-Agentic-E-Commerce/meta/meta_info.txt) - Allouah et al., 2025
- [AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions](./references/AgenticPay-A-Multi-Agent-LLM-Negotiation-System-for-Buyer-Seller-Transactions/meta/meta_info.txt) - 2026
- [LLM-Deliberation](https://arxiv.org/abs/2305.19118) - Abdelnabi et al., 2023
- [Cooperation, competition, and maliciousness: LLM-stakeholders interactive negotiation](https://arxiv.org/abs/2402.01918) - Abdelnabi et al., 2024
- [NegotiationArena](https://arxiv.org/abs/2402.05863) - Bianchi et al., 2024
- [Put Your Money Where Your Mouth Is: Auction Arena](https://arxiv.org/abs/2310.05746) - Richardson et al., 2023
- [Algorithmic collusion by large language models](https://arxiv.org/abs/2404.00806) - Gonczarowski et al., 2024
- [STEER](https://arxiv.org/abs/2402.09552) - Raman et al., 2024
- [Economics Arena for Large Language Models](https://arxiv.org/abs/2401.01735) - Sui et al., 2024
- [GLEE](https://arxiv.org/abs/2410.05254) - Madmon et al., 2024
- [ABIDES-Economist](https://arxiv.org/abs/2402.09563) - Dwarakanath et al., 2024
- [The AI Economist](https://arxiv.org/abs/2004.13332) - Zheng et al., 2020
- [Generative Agents](https://arxiv.org/abs/2304.03442) - Park et al., 2023
- [Homo Silicus / LLMs as simulated economic agents](https://dl.acm.org/doi/10.1145/3670865.3673566) - Filippas et al., 2024
- [The agentic economy](https://arxiv.org/abs/2505.15799) - Rothschild et al., 2025
- [Virtual agent economies](https://arxiv.org/abs/2509.10147) - Tomasev et al., 2025
- [Qwen3 technical report](https://arxiv.org/abs/2505.09388) - Yang et al., 2025
