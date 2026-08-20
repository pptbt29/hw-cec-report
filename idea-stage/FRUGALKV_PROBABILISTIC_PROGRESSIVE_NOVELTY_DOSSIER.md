# Novelty dossier: probability-conditioned progressive KV preparation

## Proposed mechanism

For each completed chatbot session, estimate the conditional probability that its next request will arrive within future intervals. At each background scheduling opportunity, choose one unit action: transfer the next KV block batch to node $j$, recompute the next KV block batch on node $j$, or wait. A high return probability should justify more preparation; a low probability should justify less preparation or waiting. Actions must be coordinated across sessions under inter-site bandwidth, background accelerator, and HBM constraints, with the primary objective of maximizing global TTFT SLO-goodput and a secondary objective of minimizing resource use.

The current FrugalKV draft instead uses return probability implicitly: a return-risk window activates a session; probability contributes to a complete plan's expected global goodput; the controller admits complete SLO-sufficient plans; and the runtime executes admitted plans in block batches. It does not yet explicitly define a probability-conditioned block-action policy.

## Core claims to evaluate

1. Conditional return-time probability determines how many KV blocks should be prepared now for a multi-turn chatbot session.
2. The online action chooses among cross-node KV transfer, target-node KV recomputation, and waiting at block-batch granularity.
3. The value of a unit action includes the continuation value of completing a residual SLO-sufficient plan, rather than only its immediate cache-hit benefit.
4. Multiple sessions are coordinated under global bandwidth, background accelerator, and HBM scarcity to maximize TTFT SLO-goodput, then minimize resource cost.
5. The same probability- and SLO-based value is used for proactive preparation and idle-KV eviction.

## Candidate prior work

1. **SYMPHONY: Enabling Compute-Memory Disaggregation in LLM Serving Systems**, NSDI 2026; arXiv:2412.16434. Advisory-driven proactive cross-node KV migration and opportunistic prefetching before a future request, with priority-based and cooperative memory management. It does not explicitly optimize probability-conditioned preparation amount or a global resource-minimal SLO-goodput objective.
2. **KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows**, NeurIPS 2025; arXiv:2507.07400. Uses steps-to-execution to drive fine-grained eviction and prefetches KV for the next workflow step. It assumes a known workflow schedule rather than chatbot return-time uncertainty.
3. **Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management (PBKV)**, arXiv:2605.06472 preprint. Predicts multi-step agent-invocation probability distributions; aggregates per-cache-node reuse probability; uses that value for eviction; and greedily prefetches cache nodes one by one under a GPU-space and PCIe-bandwidth budget. This is the closest overlap with probability-conditioned unit KV actions.
4. **Efficient Serving of LLM Applications with Probabilistic Demand Modeling (Hermes)**, arXiv:2506.14851 preprint. Uses probabilistic demand graphs for scheduling and backend prewarming, but not KV block transfer/recomputation choices.
5. **ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management**, arXiv:2601.21473 preprint. Uses predicted invocation distance to drive proactive prefetch and priority eviction for agent memory, but not chatbot return-time distributions or SLO-goodput/resource-minimal cross-node KV preparation.
6. **Leveraging the Power of Prediction: Predictive Service Placement for Latency-Sensitive Mobile Edge Computing**, IEEE TWC 2020; arXiv:2006.09710. Uses predicted future information and two-timescale Lyapunov optimization for latency-sensitive service placement under a long-term migration-cost budget. This is conceptually close to prediction-driven, cost-constrained edge placement, although it is not about LLM KV cache.

## Questions for reviewer

1. Is the probability-conditioned unit-action idea novel by itself?
2. Which prior work is closest, especially PBKV?
3. Does the combination of chatbot conditional return-time distributions, transfer/recompute/wait decisions, residual-plan continuation value, TTFT SLO feasibility, and global CEC resource coordination constitute a defensible systems contribution?
4. What claims should be avoided, and what narrow differentiator remains?
