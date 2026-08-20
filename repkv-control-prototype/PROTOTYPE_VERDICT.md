# Prototype Verdict

## Question answered

Can the original RepKV control path be represented without future-information leakage, invalid partial KV sources, softening an impossible SLO into a nominally useful plan, or exceeding HBM while background work is in flight?

## Current answer

Yes at the control-model level. The prototype can construct a complete minimum-prefix plan, execute it progressively, re-evaluate it every control interval, expose the resulting state to a router, and reclaim HBM using the same expected SLO-loss signal. Runtime invariants held in the default three-seed smoke run and in a tighter-SLO pressure trace that exercised transfer, recomputation, demotion, two prepared routing options, and batch completion.

The initial three-seed run produced only a modest RepKV trend: SLO attainment was `0.9502` for RepKV, `0.9422` for on-demand recovery, and `0.9327` for eager full preparation. These values are a smoke result, not paper evidence. They show that the state machine runs and that eager copying can lose under HBM pressure; they do not establish statistical significance or hardware benefit.

## Decision retained

The algorithm should operate on strict valid prefixes and admit only complete hard-SLO-feasible plans. A complete plan may contain contiguous local-restore, remote-transfer, and exact-recompute ranges, while execution remains batch-progressive. Preparation and eviction should share a cluster-level expected SLO-success signal so that creating a second viable routing option has value even when one node is already viable.

## Still outside the prototype

Absolute performance, foreground/background interference, actual Ascend KV layout compatibility, block-table mutation, asynchronous transport, host-pinned allocation, failures, and multi-server correctness remain unimplemented. Those require a measured runtime data plane rather than additional simulator detail.
