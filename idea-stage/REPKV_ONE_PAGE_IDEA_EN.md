# RepKV: One-Page Research Idea

**Working title:** *RepKV: Resource-Efficient Proactive KV Management for SLO-Aware Collaborative Edge LLM Serving*

**One-sentence idea:** RepKV prepares the necessary KV at a small number of backup nodes before the next request arrives, and removes the least useful idle KV when memory is full. Its goal is to serve more requests within their TTFT limits using less network, compute, and memory.

## Abstract

> Edge sites jointly serve multi-turn LLM requests with limited compute, memory, and network capacity. Later requests reuse KV from earlier turns, so they usually return to the node holding it. As KV grows, moving a request only after that node becomes busy may be too late because rebuilding its history elsewhere can exceed the TTFT limit. Preparing KV earlier can help, but unnecessary work wastes resources and may remove other useful KV from memory. We study how to maximize requests meeting TTFT limits with minimal resource use. RepKV estimates when a session will return and the cost of its next prompt. It calculates the least transfer or recomputation needed to prepare one useful alternative node. RepKV then chooses among competing sessions and removes the least useful idle KV when memory is full. We will evaluate RepKV on eight Ascend 910B NPUs with chatbot traces, several routers, and prediction errors.

## 1. Draft Introduction

### 1.1 Context

Collaborative Edge Computing (CEC) lets nearby edge sites serve requests together. Compared with a large datacenter, these sites have less compute, accelerator memory, and inter-site bandwidth, so a traffic burst can quickly create a long queue at one site.

Multi-turn LLM requests store intermediate results from earlier turns in a KV cache. Reusing this KV avoids recomputing the full conversation history. If the next request moves to a node without the KV, the system must transfer it or recompute the history. Both operations take longer as the conversation grows.

Routers therefore tend to send the next request back to the node holding its KV. This works well while that node is idle, but a long session becomes increasingly dependent on it. If the node later becomes busy, moving the request elsewhere may already be too late.

### 1.2 Problem

We study three connected problems.

1. **Moving a request after it arrives can be too late.** If the node holding the KV has a long queue, an idle node may still fail to help because transferring or recomputing a long history can exceed the TTFT limit. The router has not simply chosen the wrong node; it has no node that can serve the request on time.
2. **The right amount of early preparation is unknown.** The system does not know whether a session will continue, when the next request will arrive, or how long its new prompt will be. Preparing too early or too much wastes resources, while preparing too late or too little does not help the request finish on time.
3. **Sessions compete for the same resources.** Preparing KV for one session can fill a node's memory and force the system to remove another session's KV. Improving one session alone can therefore reduce the total number of requests served on time.

RepKV asks: **Under limited resources, which sessions should receive early KV preparation, how much should be prepared, and which idle KV should be removed when memory is full?** We call the number of requests per second that meet their TTFT limits SLO-goodput.

### 1.3 Challenges

- **When to prepare:** predict whether and when the next request will arrive, and estimate the cost of its new prompt.
- **What to prepare:** choose between transferring KV, recomputing history, and waiting, and determine how much work is actually needed.
- **How to coordinate:** allocate shared resources across sessions and make preparation and removal follow the same system-wide goal.

### 1.4 Boundaries of Existing Work

| Direction and representative systems | What they provide | Remaining gap |
|---|---|---|
| **KV-aware routing:** [Preble](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html), [DualMap](https://openreview.net/forum?id=zCadrJ32Xn) | Route an arrived request using existing KV and current load. | They do not prepare missing KV at other nodes before the next request arrives. |
| **Post-arrival migration and restoration:** [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao), [CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost), [Cake](https://proceedings.mlr.press/v267/jin25d.html), [HCache](https://2025.eurosys.org/accepted-papers.html) | Move queued or running requests, or make KV loading faster. | They start after the request needs service, so restoring a long history can still miss its deadline. |
| **Pre-arrival KV preparation:** [SYMPHONY](https://www.usenix.org/conference/nsdi26/presentation/agarwal) | Use an early user or agent signal to prefetch KV to a selected node. | It does not decide how much to prepare for each session under global resource competition. |
| **Workflow-based KV prediction:** [KVFlow](https://papers.neurips.cc/paper_files/paper/2025/hash/b7971d31a7d5eb0f1eed2f8f6f368195-Abstract-Conference.html), [PBKV](https://arxiv.org/abs/2605.06472) (preprint), [Pythia](https://arxiv.org/abs/2604.25899) (preprint) | Use an agent call graph or call probability to prefetch and remove KV. | They mainly target agents with known workflows, while RepKV targets multi-turn chats without a call graph and shared edge resources. |

Existing work has studied routing, KV restoration, early prefetching, and agent workflow prediction. RepKV asks how to use the least preparation work to serve more multi-turn chat requests on time when many sessions share limited resources.

### 1.5 Approach Overview

RepKV works in three steps.

1. **Predict the next request.** After the current request finishes, RepKV estimates whether and when the session will return and how much work its next prompt will require. It starts preparation only when the next request is likely to arrive soon.
2. **Calculate the necessary preparation.** RepKV checks the nodes allowed by the router and estimates how much KV each node still needs to return the first token on time. It may transfer KV, recompute history, or combine both, but it does not prepare every node.
3. **Allocate shared resources.** RepKV compares how many on-time requests each preparation may add with the network, compute, and memory it consumes. When memory is full, it removes the idle KV that has the smallest effect on the system-wide result.

RepKV does not replace the router. It reports how much KV each node already has and how much recovery remains. The existing router still chooses the final node after the request arrives, so RepKV can work with different routing policies.

### 1.6 Expected Contributions

1. **A resource-limited proactive KV problem.** The goal is not to copy more KV, but to improve system-wide SLO-goodput under explicit resource limits. Preparation and removal follow the same goal.
2. **A necessary-preparation method.** RepKV transfers or recomputes only KV that is expected to help a future request finish on time.
3. **Coordination across sessions.** RepKV gives priority to preparations that provide more SLO-goodput per unit resource and removes idle KV with the smallest expected harm.
4. **A router-independent prototype.** We will implement RepKV on an Ascend 910B cluster and test whether it improves SLO-goodput while reducing network, background compute, and memory use.

---

> The remaining sections are technical notes for the formulation, algorithm, and evaluation. They are not part of the one-page summary or the draft Introduction.

## 2. Problem Formulation

### 2.1 Scope

CEC nodes expose independent queues, accelerators, memory, and optional lower storage tiers. RepKV manages sessions that have completed their current turn and are waiting for a possible next request. KV belonging to an active request is never removed by RepKV.

The router chooses the final execution node after request arrival. RepKV changes only the pre-arrival KV state at candidate nodes, thereby changing which routing options are likely to meet the SLO.

### 2.2 Main Notation

| Symbol | Meaning |
|---|---|
| $T,t,\Delta$ | Evaluation horizon, current time, and short control interval |
| $s,n,r$ | Waiting session, router-allowed node, and arrived request |
| $\mathcal R_T$ | Requests arriving within horizon $T$ |
| $D_r,D_s$ | TTFT SLO of request $r$ and the next request of session $s$ |
| $X_s,\tau_s$ | Next-turn interval and time already elapsed since the current turn completed |
| $q_s(t,\Delta)$ | Probability that session $s$ returns in $[t,t+\Delta]$, conditioned on not having returned by $t$ |
| $\mathcal N_s$ | Nodes that the router allows for session $s$ |
| $p,\mathcal P_s$ | One complete plan and the candidate plan set for session $s$ |
| $x_{s,p,t}$ | Binary decision to select plan $p$ at time $t$ |
| $\widehat T_n^{queue}$ | Estimated queueing time at node $n$ |
| $\widehat T_{s,n}^{recover}(p)$ | Remaining historical-state recovery time after plan $p$ |
| $\widehat T_s^{new}$ | Estimated prefill time for the new prompt |
| $B_{s,p},U_{s,p},H_{s,p}$ | Transfer bytes, background accelerator time, and prepared-KV HBM residency of plan $p$ |
| $\Delta G_{s,p,t}$ | Expected global SLO-goodput gain of plan $p$ |
| $\lambda_B,\lambda_U,\lambda_H$ | Weights converting normalized resource use into SLO-goodput cost |
| $G_T(\pi),J_T(\pi)$ | Global SLO-goodput and the combined objective under policy $\pi$ |
| $e,S_e$ | An evictable idle-KV block batch and the HBM it releases |

### 2.3 Next-Return Estimation

RepKV fits a lightweight survival model to historical inter-turn intervals. Let $F_c(x)$ be the next-turn interval distribution for a session class $c$, where classes may use application type, turn count, and context length. For a session that has waited $\tau_s$ without returning:

$$
q_s(t,\Delta)
=
\Pr(X_s\le\tau_s+\Delta\mid X_s>\tau_s)
=
\frac{F_c(\tau_s+\Delta)-F_c(\tau_s)}
{1-F_c(\tau_s)}.
$$

This model estimates a conditional return probability, not a guaranteed arrival time. An advisory updates the relevant probability window when available.

### 2.4 Global Objective

SLO-goodput is the number of requests per second that produce their first token within the SLO:

$$
G_T(\pi)
=
\frac{1}{T}
\sum_{r\in\mathcal R_T}
\mathbf 1\{TTFT_r(\pi)\le D_r\}.
$$

To prevent small SLO gains from consuming all spare resources, RepKV directly penalizes normalized resource use:

$$
J_T(\pi)
=
G_T(\pi)
-\lambda_B\frac{B_T(\pi)}{B^{ref}}
-\lambda_U\frac{U_T(\pi)}{U^{ref}}
-\lambda_H\frac{H_T(\pi)}{H^{ref}}.
$$

RepKV maximizes $J_T(\pi)$ subject to instantaneous bandwidth and background-accelerator quotas, hard HBM capacity, and protection of active-request KV. Evaluation reports $G_T$, transfer bytes, accelerator time, and HBM residency separately so that a weighted score cannot hide resource regressions.

### 2.5 Complexity

Consider a restricted case with one node and one shared resource. Each session either consumes $w_s$ resource units to obtain expected SLO gain $v_s$, or waits. Selecting sessions under total budget $\mathcal B$ is exactly 0–1 knapsack. The restricted offline problem is NP-hard, so the full finite-horizon problem with multiple nodes, resources, recovery methods, blocks, and eviction is also NP-hard.

## 3. Method

RepKV follows one chronological path: record state, predict return, construct complete plans, coordinate plans globally, execute them in batches, reclaim HBM when necessary, and publish readiness to the router.

### 3.1 Record State at Request Completion

When a turn completes, RepKV records the latest KV length, replicas and storage tiers, turn count, context length, and historical inter-turn intervals. It does not immediately migrate the KV because the next turn may arrive much later or never arrive.

### 3.2 Activate Preparation Near the Return Window

The predictor continuously updates $q_s(t,\Delta)$. A session enters the global preparation queue when it reaches its predicted return window and preparation must begin soon to finish before the likely arrival. Request completion creates the prediction; the approaching return window triggers preparation.

The scheduler reruns when a session enters the queue, a block batch completes, or background resources become available. It reconsiders only active candidates rather than globally rearranging every KV object.

### 3.3 Construct Complete SLO-Feasible Plans

For node $n$, RepKV estimates:

$$
\widehat{TTFT}_{s,n}(p)
=
\widehat T_n^{queue}
+\widehat T_{s,n}^{recover}(p)
+\widehat T_s^{new}.
$$

Starting from the node's longest correct KV prefix, a node-level solver considers transferring the next block batch, exactly recomputing it, or using transfer and recomputation for different contiguous ranges. It finds the lowest normalized-cost path satisfying:

$$
\widehat{TTFT}_{s,n}(p)\le D_s.
$$

The result is a complete useful plan, such as “transfer 40 blocks to A” or “transfer 20 and recompute 10 blocks at B.” A partial action that cannot eventually cross the SLO boundary is not counted as a new routing option. Nodes that remain infeasible even after full restoration produce no plan.

For each session, RepKV considers waiting, preparing one node, or preparing at most two nodes. The router may expose all available nodes; the cap applies only to how many additional nodes RepKV prepares for one session.

### 3.4 Coordinate Sessions Globally

Plans from all active sessions compete in one decision:

$$
\max_x
\sum_s\sum_{p\in\mathcal P_s}
x_{s,p,t}
\left[
\Delta G_{s,p,t}
-\lambda_B\frac{B_{s,p}}{B^{ref}}
-\lambda_U\frac{U_{s,p}}{U^{ref}}
-\lambda_H\frac{H_{s,p}}{H^{ref}}
\right],
$$

subject to at most one selected plan per session and the current bandwidth, background-accelerator, and per-node HBM limits. $\Delta G_{s,p,t}$ includes the return probability, the change in SLO success probability, and the value of additional routing options.

The online controller decomposes the NP-hard decision using resource prices: a scarce resource receives a higher price, each session scores only its small plan set, and the controller resolves plans that conflict on the same link or node. Small experiments compare this approximation against an offline oracle with full future information.

### 3.5 Execute Selected Plans Progressively

The controller admits complete plans, while the runtime executes one block batch at a time. It prioritizes plans with the smallest preparation slack—time until the return window minus remaining preparation time—and breaks ties by net value.

If A needs 40 blocks to become feasible, transferring the first 10 does not yet count as an SLO-feasible option. After each batch, RepKV updates the return probability, queue, link state, and remaining completion time. It stops further batches if the plan can no longer finish on time or loses positive net value.

### 3.6 Reclaim HBM Under Pressure

Before admitting prepared KV, RepKV checks whether active KV, growth headroom, idle KV, and admitted preparation would exceed a high watermark. If so, it reclaims idle KV until reaching a low watermark.

For each evictable tail batch $e$, RepKV computes:

$$
Score(e)
=
\frac{\Delta G_e^{evict}+C_e^{evict}}{S_e},
$$

where $\Delta G_e^{evict}$ is expected global SLO-goodput loss, $C_e^{evict}$ is the cost of moving or deleting the batch, and $S_e$ is released HBM. Lower-score batches are reclaimed first. RepKV demotes them to CPU memory or local storage when useful; otherwise, it removes the KV while retaining source tokens for later recomputation.

Each node maintains a lazy priority queue of idle KV. Scores are updated when KV becomes idle, a session enters its return window, return probability changes materially, or state placement changes. Active prefill and decode KV never enter this queue.

### 3.7 Compose with Existing Routers

RepKV publishes the valid KV prefix, storage tier, and estimated remaining recovery time at every candidate node. The router then applies its own SLO, queue, and load policy without needing to know whether KV came from transfer, recomputation, or demotion.

RepKV therefore decides **how to change KV state before arrival**, while the router decides **where an arrived request executes**. The two components share current node, queue, and KV-readiness metadata, but not an internal scoring rule.

## 4. Evaluation Plan

### 4.1 Testbed and Models

- **Hardware:** eight Ascend 910B NPUs, primarily as eight single-NPU model replicas. CEC results require at least two physical servers; the final paper will report server placement, NUMA topology, native link bandwidth, and RTT.
- **Runtime:** vLLM Ascend or an equivalent Ascend serving runtime, extended with KV-block transfer and exact background recomputation. Every baseline uses the same data plane.
- **Models:** Qwen2.5-7B-Instruct as the primary model and another single-910B 7B–8B model for replication, with 4K, 16K, and 32K contexts.
- **Network:** native bandwidth $B_0$ plus $0.25B_0$ and $0.5B_0$ limits; native, 10 ms, and 30 ms RTT.

### 4.2 Workloads and Prediction

- Use real multi-turn ShareGPT or WildChat conversations, preserving context growth and output lengths.
- Replay BurstGPT arrival times; when full session timestamps are unavailable, combine real conversations with parameterized inter-turn intervals.
- Use log-normal inter-turn intervals with medians $\{5,15,60\}$ s and geometric standard deviations $\{1.5,3\}$, plus Gamma arrivals with $CV\in\{0.5,1,2\}$.
- Train the return estimator with chronological train/validation/test splits and inject early, late, and absent returns.
- Use MetaGPT or a public workflow trace only as a supplementary Agent experiment.

### 4.3 SLO, Load, and Resource Settings

- Measure unloaded P99 TTFT with local HBM KV for each input-length bucket, $T_{99}^{hit}(l)$, then use $\{1.5,2,3\}T_{99}^{hit}(l)$ as strict, medium, and relaxed SLOs.
- Find the router-only saturation rate $\lambda_0$ and replay $\{0.6,0.8,1.0,1.2\}\lambda_0$.
- Allocate $\{40,60,80\}\%$ of HBM remaining after weights and runtime reservation to managed KV.
- Evaluate bandwidth-, accelerator-, and HBM-constrained regimes by changing both hard budgets and normalized resource weights.
- Prepare at most one extra node by default and compare against a two-extra-node configuration.

### 4.4 Baselines

All baselines share the router, predictor, KV block size, and transfer–recomputation data plane.

1. **On-demand:** restore KV only after request arrival.
2. **Eager-full:** prepare complete KV at the predicted target when the same return window begins.
3. **SYMPHONY-style:** use the same return signal, preselect one target by current load, and apply opportunistic staging with layer-priority reclamation.

The primary router is minimum-predicted-TTFT. Plugin experiments pair RepKV with Preble-style and DualMap-style routers. Cake, HCache, or [CacheFlow](https://arxiv.org/abs/2604.25080) (preprint) can provide a common recovery data path rather than serving as control-policy baselines.

### 4.5 Metrics and Comparisons

The main metrics are:

1. **SLO-goodput:** requests per second whose first token meets the TTFT SLO.
2. **Resource per SLO-success:** inter-site GB, background NPU-seconds, and prepared-KV HBM GB-seconds.
3. **P99 TTFT:** tail latency under the same offered load.
4. **Unused preparation ratio:** prepared bytes evicted before use or never selected by the router.

Main results compare SLO-goodput under equal resource budgets and resource use at matched SLO-goodput. The key dimensions are offered load, bandwidth/HBM scarcity, session length, and return-prediction error.

### 4.6 Ablations and Supporting Experiments

1. Replace the SLO-sufficient node planner with full-KV preparation under the same global controller.
2. Cross global versus independent per-session selection with SLO-value versus LRU eviction in a $2\times2$ study.

Supporting experiments report return-window coverage, available preparation lead time, early preparation, false returns, oracle gap, controller latency, block-batch sensitivity, HBM watermark sensitivity, and confidence intervals from at least three repeated runs.

## 5. Paper Structure

| Section | Content | Main figures |
|---|---|---:|
| Introduction | Context, three-step problem, three challenges, four related directions, three design ideas, contributions | One teaser |
| Background and Motivation | CEC resources, KV lifecycle, restoration paths, router–KV interface | One timeline |
| Problem Formulation | Global SLO–resource objective, constraints, NP-hard special case | One decision example |
| Design | Return prediction, node plans, global control, progressive execution, HBM reclamation | Architecture and workflow |
| Implementation | Ascend runtime, KV block interface, transfer and recomputation data paths | Implementation diagram |
| Evaluation | End-to-end results, resource efficiency, ablations, prediction robustness, router composition, oracle gap | Six to eight plots |
| Related Work and Limitations | Routing, restoration, proactive staging, workflow prediction | No main figure |

## 6. Preconditions Before Implementation

1. Confirm how the eight Ascend 910B NPUs are distributed across physical servers and measure the actual inter-server data path.
2. Hold the predictor, router, and restoration data plane fixed when comparing KV control policies.
3. Use a small oracle-gap study to determine whether global coordination consistently outperforms independent per-session decisions before implementing a complex online controller.
4. Recheck concurrent work on proactive KV preparation, probabilistic prediction, and CEC KV migration before submission.
