# One-Page Idea Summary

## Working Title

**Move the Request or Move the State? Prediction-Guided Routing and Staged KV Replication for Mobile Edge LLM Serving**

## Core Research Question

When a mobile user's request reaches a new edge node while its session KV cache remains elsewhere, should the system forward the lightweight request to the state, move the heavyweight state toward the user, or progressively prepare both options before handoff?

## Draft Abstract (147 words, 10 sentences)

Mobile edge LLMs reduce access latency but may leave session KV caches behind as users move. We study joint request routing and KV preparation under uncertain mobility. KV caches can be gigabytes while requests are megabytes, making blind state migration inefficient. A correct trade-off can reduce handoff stalls without wasting bandwidth or GPU memory. We formulate long-term mobile LLM serving over execution-node selection, probabilistic KV placement, and residual recovery under SLA and resource constraints. We derive a break-even boundary between state-sticky routing and state movement. Our KV manager progressively replicates immutable prefix blocks using multi-step mobility predictions and spare bandwidth. At handoff, it overlaps residual synchronization with recomputation, while the router selects the node using predicted recovery costs. We will compare against greedy, reactive-migration, and oracle baselines across mobility, workload, and network traces. **[Results placeholder: report P99 latency, SLA violations, handoff stall, KV traffic, and wasted replication.]**

## Introduction Logic

**P1 -- Context.** Edge-hosted LLMs can provide low-latency interactive services, but user mobility changes the ingress node during a multi-turn session while the accumulated KV state remains at previous compute nodes.

**P2 -- Fundamental asymmetry.** A request and response are often KB--MB scale, whereas a long-context KV cache can be GB scale; therefore, following the user with KV is not automatically better than forwarding future requests to the KV-resident node.

**P3 -- Why routing alone is insufficient.** The execution node chosen for the current request determines where new KV blocks grow, so a locally optimal route can increase future forwarding, recovery, and memory costs; output length and unknown session duration further obscure the long-term consequence.

**P4 -- Why reactive migration is insufficient.** Waiting until handoff to transfer the full KV places a large state movement on the critical path, while immediately replicating it to every likely node wastes bandwidth and memory when mobility predictions are uncertain.

**P5 -- Key insight.** Transformer KV is largely append-only: stable prefix blocks can be progressively copied in the background as a handoff becomes more likely, leaving only a small tail to synchronize or recompute when the user actually moves.

**P6 -- Approach overview.** We couple a state-aware long-term router with a mobility-aware KV manager through a per-node recovery-cost interface. The KV manager estimates and reduces each candidate node's residual recovery cost through staged replication and hybrid synchronization/recomputation; the router selects the execution node using these costs, queue states, SLA constraints, output-length uncertainty, and future mobility.

**P7 -- Contributions.** The paper contributes (1) a stochastic break-even model characterizing when request forwarding, background state preparation, or execution migration is preferable; (2) a multi-step prediction-guided KV manager that controls destination, block set, replication rate, and retention under bandwidth/memory constraints; and (3) a long-term state-aware router that exploits the resulting recovery-cost landscape while avoiding infeasible and short-sighted actions.

## Technical Skeleton

For candidate execution node \(j\), the KV manager exposes

\[
\Phi_k(j)=\left(T_k^{\mathrm{restore},*}(j),D_k^{\mathrm{sync},*}(j),C_k^{\mathrm{rec},*}(j),\Delta M_k(j),\chi_k^{\mathrm{feasible}}(j)\right).
\]

The router chooses only the execution node,

\[
J_k=\pi_{\mathrm{route}}(s_k,\Phi_k(1),\ldots,\Phi_k(N)),
\]

while the KV manager periodically chooses background placement variables \(x_{t,j,b}\), representing how much bandwidth is assigned to copying block \(b\) toward node \(j\). At handoff, it minimizes the residual critical path:

\[
T_k^{\mathrm{restore},*}(j)=
\min_x\left[
\max\{T_k^{\mathrm{final\ sync}}(j,x),T_k^{\mathrm{recompute}}(j,1-x)\}
+T_k^{\mathrm{bubble}}
\right].
\]

Execution should move from a KV-resident node to candidate \(j\) when

\[
C_k^{\mathrm{background}}(j)+C_k^{\mathrm{residual}}(j)
<
\mathbb E\!\left[\sum_{h\ge1}\gamma^{h-1}\Delta C_{k+h}^{\mathrm{remote}}(j)\right].
\]

This break-even surface changes with KV size, spare bandwidth, recompute throughput, handoff lead time, queue difference, session continuation probability, prediction confidence, memory pressure, and replica eviction cost.

## Paper Outline and Planned Figures

1. **Introduction:** problem, asymmetry, insight, design and contributions. *Fig. 1:* motivating timeline comparing request forwarding, reactive KV migration, and staged pre-copy.
2. **Background and Motivation:** KV-size measurements, edge mobility scenario, limitations of routing-only and migration-only approaches. *Fig. 2:* measured break-even phase diagram over KV size, bandwidth and future dwell time.
3. **System Model and Break-even Analysis:** nodes, mobility process, append-only versioned KV blocks, unknown output/session lengths, cost and constraints. *Fig. 3:* state-sticky versus state-moving decision regions.
4. **System Design:** foreground routing, background placement, block directory, recovery-cost estimator and feedback loop. *Fig. 4:* end-to-end architecture.
5. **Long-term State-aware Router:** state/action design, feasibility masking, long-term value learning or prediction-assisted online policy, and fallback. *Fig. 5:* router/KV-manager interaction over multiple requests.
6. **Prediction-guided KV Manager:** multi-step mobility probabilities, staged block replication, bandwidth allocation, replica retention/eviction, and residual hybrid recovery. *Fig. 6:* block-level pre-copy and final sync/recompute pipeline.
7. **Implementation:** integration with an LLM serving runtime, directory consistency, telemetry, background workers and failure handling.
8. **Evaluation:** end-to-end results, microbenchmarks, ablations, prediction robustness and overhead. *Fig. 7+:* latency, traffic, waste, hit rate and sensitivity results.
9. **Related Work, Discussion and Conclusion:** distributed KV systems, predictive prefetching, mobile edge inference, limitations and VLM/VLA extensions.

## Evaluation Plan

**Testbed.** Three identical-model inference nodes connected by two 100 Gbps links and one 25 Gbps link; vary effective bandwidth, RTT, congestion, GPU load and memory pressure. Use multiple model/context configurations to cover sub-GB to tens-of-GB KV states.

**Workloads.** Multi-turn chat sessions with controlled and trace-derived prompt/output lengths; synthetic Markov and trajectory-based mobility; varied handoff lead time, dwell time, session continuation probability, request interval and prediction error.

**Baselines.** Nearest-node routing; KV-resident/state-sticky routing; current-cost Greedy; reactive full sync; pure recompute; immediate full pre-copy; threshold pre-copy; routing with an ideal recovery oracle; and a full mobility/length oracle.

**Primary metrics.** Mean/P95/P99 E2E latency, TTFT, SLA violation rate, handoff blocking time and supported concurrency under SLA.

**Explanatory metrics.** KV bytes transferred, residual KV at handoff, prefix/block hit rate, sync/recompute overlap, wasted speculative traffic, replica memory, eviction externality, route switching frequency and controller overhead.

**Ablations.** Remove long-term routing, staged pre-copy, multi-step prediction, hybrid recovery, continuation prediction, eviction cost and action masking individually; sweep mobility accuracy, bandwidth, KV growth rate and background budget.

**Results to fill before submission.** The abstract's final sentence must report measured improvements over the strongest non-oracle baseline, preferably including one tail-latency/SLA result, one handoff result, and one resource-efficiency result rather than only average latency.
