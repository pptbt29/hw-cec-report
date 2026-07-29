# Directed novelty search log

## Candidate under review

**StateWeaver / Multi-Source Executable-State Assembly**

Core claim under test:

> A CEC request becomes executable only after multiple geographically distributed, version-compatible state artifacts complete an AND-join; each artifact has OR alternatives such as transfer, suffix synchronization, exact recomputation, or equivalent local generation; worker placement and shared-link scheduling must therefore be selected jointly.

## Query families

### Batch/task scheduling overlap

- `edge LLM continuous batching task scheduling distributed`
- `batch-ready option scheduling inference`
- `commit-ready batch scheduling distributed inference`
- `candidate batch global scheduler worker inference`
- `edge batching opportunity task scheduling`

### Stateful placement and multi-source assembly

- `LLM multi-source state transfer adapter KV input scheduling`
- `LLM state assembly coflow scheduling`
- `edge LLM multiple state sources scheduling`
- `KV cache adapter model weights coflow scheduling`
- `executable state closure task scheduling`

### Hierarchical CEC and adjacent abstractions

- `hierarchical edge scheduling stale telemetry inference`
- `stateful capability lease scheduling edge inference`
- `resource offers batching scheduler global local`
- `multimodal workflow distributed KV model sharing`

## Formal closest works found

| Work | Venue | Direct overlap | Residual distinction |
|---|---|---|---|
| SHEPHERD | NSDI 2023 | Two-level planning/serving, candidate batch generation, model-specific batching | Does not assemble multiple distributed generative-state artifacts |
| Adaptive Scheduling for Edge-Assisted DNN Serving | IEEE MASS 2023 | Scheduling creates edge batching opportunity | One-shot DNN requests, not state assembly |
| Joint Batching and Scheduling for High-Throughput Multiuser Edge AI | IEEE TWC 2024 | Joint task-batch association, batch start and radio resources | Static one-shot batches |
| PPipe | USENIX ATC 2025 | Pooled pipelines, resource reservation, GPU/NIC-aware adaptive batching | Sequential DNN pipeline, not AND-OR state readiness |
| Chameleon | MICRO 2025 | Adapter caching and scheduling; adapter/KV/HBM interaction | No cross-artifact distributed closure or CEC link scheduling |
| Preble | ICLR 2025 | KV reuse and load-aware routing | Primarily a single cache-locality dimension |
| ReSK | IEEE IoT-J 2026 | Edge request scheduling and local KV caching | No cross-node multi-artifact assembly |
| LMetric | OSDI 2026 | KV-aware and batch-size-aware request routing | No AND-OR artifact plan |
| Libra | NSDI 2026 | Global request splitting, local SLO batches, chunked KV transfer | No multi-source observation/adapter/KV closure |
| JITServe | NSDI 2026 | Imprecise request information and batch composition | No CEC state-assembly plan |
| Varys | SIGCOMM 2014 | Coflow scheduling | Fixed flow groups/endpoints; no alternate state realization or worker choice |
| CLARINET | OSDI 2016 | WAN-aware plan selection | Generic analytics plans; no versioned generative-state closure |
| Sonic | USENIX ATC 2021 | Workflow data-passing plans | Generic serverless data passing |

## Preprint collision risk

*Omni-Flow: A Unified Workflow Orchestration and Distributed KV Cache Sharing Framework for Multimodal Inference* (arXiv:2606.31093) is a direct 2026 concurrent-risk item. It discusses multimodal workflow orchestration, intermediate transfer, and distributed KV/model sharing. It is not used as core evidence because it is a preprint, but a full mechanism-by-mechanism comparison is mandatory before submission.

## Adversarial result

The originally preferred candidate, commit-ready batch options, was rejected after SHEPHERD and PPipe were found. Its remaining delta was largely an extension of existing batch/path scheduling to continuous batching and multi-tier links.

No searched formal paper was found that directly combines all four of the following as the central scheduling abstraction:

1. multiple required state artifacts with AND-join readiness;
2. alternative exact realization actions for each artifact;
3. joint worker placement;
4. contention-aware scheduling over hierarchical CEC links.

This is **not proof of novelty**. The residual claim remains conditional because generic DAG placement, WAN plan selection and coflow scheduling can express large parts of the problem, and because the real workload may be dominated by a single artifact.

## Decision

Status: **conditional pass to kill-test, not paper-ready**.

Required empirical disproof attempts:

- measure whether multi-source non-dominated requests are common;
- compare against target-first + optimal per-request plan/coflow;
- show the joint gap disappears when state is co-located or the shared link is removed;
- stop if a scalar/vector-price router closes the gap.

