# Independent Novelty Jury Trace: KV Scheduling + Request Scheduling + Batching

> Input read directly by reviewer: `/Users/dazzysy/workspace/project/hw-cec-report/idea-stage/KV_TASK_BATCHING_OPPORTUNITY_REVIEW_20260724_001415.md`  
> Reviewer context: fresh, adversarial, no executor summary  
> Model: `gpt-5.6-sol`, reasoning effort `xhigh`  
> Date: 2026-07-24

## Candidate scores

Scores use $1$ as very weak and $5$ as very strong.

| Candidate | Importance | Novelty | Defensibility | Feasibility | Evidence readiness | Verdict |
|---|---:|---:|---:|---:|---:|---|
| CEC KV-ready batch closed-loop orchestration | 5 | 3 | 2 | 3 | 2 | Only candidate with top systems upside, but highly conditional |
| Pooling--fragmentation phase diagram | 4 | 3 | 3 | 5 | 1 | Prioritize as a measurement question |
| Batch option value metric | 2 | 2 | 1 | 5 | 1 | Analysis tool only |
| Cohort-level KV placement | 3 | 2 | 1 | 2 | 1 | Excessive oracle/prediction risk |
| KV restoration coflow | 2 | 1 | 1 | 3 | 2 | Direct method transfer |
| KV affinity versus batch compatibility | 2 | 1 | 1 | 4 | 1 | Core intuition substantially covered |
| Batch-size KV eviction externality | 3 | 1 | 1 | 4 | 2 | Crowded local caching/batching space |
| TTFT optimization creates decode bursts | 3 | 2 | 2 | 4 | 2 | Measurable but overlapped and not current priority |

## Highest-risk prior-art combination

The most likely reviewer reduction is:

> Mooncake or ThunderServe for global KV/network-aware orchestration, plus Strata for loading-aware batch formation, plus Libra for cross-instance KV transfer and two-level SLO scheduling.

The only defensible remaining claim would be that shared-link KV flows mutually change the ready set, causing a structural failure of any composition of per-request routing and an independent local batcher.

## Kill cases

1. KV transfer has no economic region:

$$
t^{\mathrm{transfer}}
\geq
\min
\left\{
t^{\mathrm{recompute}},
t^{\mathrm{wait\mbox{-}local}}
\right\}.
$$

2. Link contention is weak or transfer is completely hidden, so ready time does not change batch membership.
3. Iteration-level continuous batching, chunked prefill and a strong Strata-style local batcher absorb the ready-time differences, reducing the gap to a joint oracle below about $5\%$.

## Category and missing evidence

Current category: **reasonable combination close to engineering integration**, not an established new problem structure.

The indispensable missing evidence is a stable and causal oracle gap between the strongest network-aware decomposed baseline and a batch-level joint oracle, using real serving profiles, real arrival/session traces and realistic CEC bandwidth traces. The gap must disappear when shared contention or real continuous batching is removed.

## Decision

**NO-GO for a full top-systems paper now. Conditional GO only for a two-week falsification prototype.**

Reconsider only if a non-extreme realistic region shows a stable $15\%$--$20\%$ SLO-goodput gap. Terminate if the gap is broadly below $5\%$ or exists only where KV transfer itself is uneconomic.
