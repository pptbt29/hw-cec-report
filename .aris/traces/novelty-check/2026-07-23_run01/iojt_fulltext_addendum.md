# IoT-J 2026 full-text addendum

**Paper**：Co-optimizing Request Scheduling and KV Caching for Edge LLM Serving  
**DOI**：10.1109/JIOT.2026.3709703  
**Full text received**：2026-07-23  
**Detailed analysis**：`idea-stage/PAPER_DEEP_ANALYSIS_COOPT_EDGE_LLM.md`

## Claim-chart correction

| Dimension | Full-text finding |
|---|---|
| Multi-node edge request scheduling | Yes |
| Block-level prefix KV | Yes |
| Local KV retention/eviction | Yes |
| Temporal request/cache coupling | Yes |
| CPU RAM and GPU HBM capacity | Yes |
| Cloud fallback | Yes |
| Explicit cross-node KV fetch/migration | No; the model explicitly assumes cross-node fetching is generally slower than recomputation |
| Proactive remote KV preparation | No |
| Multiple prepared destinations / late binding | No |
| Per-link bandwidth or shared backhaul | No |
| Foreground/background network contention | No |
| Explicit queue dynamics / service rates | No |
| Explicit per-request SLO constraint | No; the paper says it could be added |
| Main optimized metric | Weighted transmission, storage, local fetch, prefill and cloud cost |
| Main reported performance | Average TTFT and failure rate |

## Novelty impact

The paper directly invalidates any claim of being first to co-optimize request scheduling and KV caching for edge LLM serving. It does not directly cover the narrower proposed mechanism of transferring missing session KV before future arrivals to create late-bound, path-qualified execution options under shared CEC backhaul.

The residual mechanism remains a conditional gap, not a confirmed contribution. It may still be viewed as a natural combination of ReSK-style scheduling/retention, SYMPHONY-style prefetch, Mooncake/Llumnix-style KV movement, and classical CEC network-resource allocation.
