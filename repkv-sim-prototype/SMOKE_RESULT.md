# Default Smoke Result

This is a logic check, not paper evidence. The default command uses five random seeds, four nodes, forty multi-turn sessions, a two-second TTFT SLO, and noisy next-turn predictions.

| Policy | SLO-goodput (req/s) | SLO attainment | P99 TTFT (s) | Transfer blocks / SLO success | Recomputed blocks / SLO success | HBM block·s / SLO success | Unused preparation |
|---|---:|---:|---:|---:|---:|---:|---:|
| On-demand | 0.5400 | 0.5997 | 6.8107 | 6.6177 | 19.9757 | 1067.0483 | 0.0000 |
| Eager-full | 0.6436 | 0.7138 | 4.9694 | 28.5466 | 5.2821 | 961.8729 | 0.7779 |
| RepKV | 0.7409 | 0.8210 | 4.5673 | 16.4036 | 3.1026 | 782.9180 | 0.6528 |

The result confirms that the prototype produces the intended qualitative comparison: proactive preparation can improve SLO-goodput, while full preparation consumes more network capacity. It also exposes a weakness rather than hiding it: 65.28% of RepKV's prepared blocks are unused in this setting. The next experiment should therefore sweep prediction error, preparation horizon, and HBM capacity before treating the policy as promising.

