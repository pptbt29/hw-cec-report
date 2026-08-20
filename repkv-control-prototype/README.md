# RepKV Control Prototype

> **PROTOTYPE — throwaway control-plane model.** It does not contain real KV tensors, a vLLM/Ascend block allocator, RDMA, or NPU execution.

## Question

This prototype asks whether RepKV's original control meaning is internally coherent: under causal next-turn predictions and shared HBM, transfer, host-restore, and background-compute limits, can a controller admit complete plans that prepare only the smallest valid KV prefix needed to create an additional TTFT-SLO-feasible routing option?

## What is implemented

- causally generated return probability and arrival-window advisories;
- a strict valid-prefix model for every session/node KV replica;
- HBM and local-host tiers, with tail demotion under pressure;
- strict transfer-source coverage instead of a Boolean “some source exists” check;
- contiguous hybrid plans using local restore, remote transfer, and exact recomputation;
- hard TTFT-SLO feasibility and background completion-time checks;
- cluster routing-option value, rather than valuing only the single current best node;
- scarcity-adjusted global candidate scoring under shared per-tick resources;
- progressive batches with capacity reservation, completion, revalidation, and cancellation;
- one consistent expected-SLO-loss signal for preparation and HBM reclamation;
- router-facing valid-prefix and remaining-recovery snapshots;
- runtime invariants preventing invalid prefixes and HBM overcommit.

## Run

Batch comparison:

```bash
python3 run.py
```

Interactive state trace:

```bash
python3 trace.py
```

Non-interactive smoke trace:

```bash
python3 trace.py --steps 40 --no-ansi
```

## Policy comparison

- `on_demand`: no pre-arrival work;
- `eager_full`: progressively build one complete backup using the same data-plane assumptions;
- `repkv`: admit only a hard-SLO-useful minimum-prefix plan and coordinate it by expected global value and resource scarcity.

## Remaining system boundary

The rates are parameters, foreground recovery contention is summarized as latency, and the queue predictor is deliberately simple. Real-system claims still require replacing the abstract data plane with measured vLLM Ascend/MindIE KV layouts, block-table updates, compatible model metadata, asynchronous copy streams, network transport, failure handling, and interference measurements on multiple physical Ascend servers.
