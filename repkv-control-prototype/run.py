#!/usr/bin/env python3
"""Reproduce the main configuration of `PROTOTYPE_VERDICT.md` section 6.1.

Defaults place the run in the zone where fetching KV from a peer still fits the
deadline but recomputing it does not, which is where the policies separate. Use
`--ttft-slo` to move across the two thresholds reported there.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from repkv_sim import Config, HardwareSpec, ResourceModel, aggregate, run_experiment

GiB = 2**30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the causal RepKV control-plane prototype.")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--sessions", type=int, default=2000, help="size of the session script pool")
    parser.add_argument(
        "--concurrent-sessions",
        type=int,
        default=90,
        help="live sessions held by replenishment; 0 starts the whole pool at once",
    )
    parser.add_argument("--horizon", type=float, default=900.0)
    parser.add_argument("--ttft-slo", type=float, default=3.0)
    parser.add_argument("--context-tokens", type=int, default=32000)
    parser.add_argument("--host-gib", type=float, default=256.0, help="DRAM tier given to KV per node")
    parser.add_argument("--network-gbytes", type=float, default=12.5, help="inter-node bandwidth")
    parser.add_argument(
        "--hbm-blocks",
        type=int,
        default=0,
        help="override the HBM capacity derived from the hardware sheet",
    )
    parser.add_argument("--think-scale", type=float, default=1.0)
    parser.add_argument(
        "--min-return-probability",
        type=float,
        default=0.05,
        help="raise above 1 to refuse every preparation candidate (reclaim-only ablation)",
    )
    parser.add_argument(
        "--lru-eviction",
        action="store_true",
        help="rank HBM victims by last-used time instead of SLO loss (prepare-only ablation)",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resources = ResourceModel(
        hardware=HardwareSpec(
            decode_context_tokens=args.context_tokens,
            host_capacity_bytes=args.host_gib * GiB,
            network_bytes_s=args.network_gbytes * 1e9,
        )
    )
    cfg = Config(
        nodes=args.nodes,
        sessions=args.sessions,
        concurrent_sessions=args.concurrent_sessions,
        horizon_s=args.horizon,
        ttft_slo_s=args.ttft_slo,
        resources=resources,
        hbm_blocks_override=args.hbm_blocks,
        think_scale=args.think_scale,
        first_prompt_tokens=args.context_tokens,
        min_return_probability=args.min_return_probability,
        value_based_eviction=not args.lru_eviction,
    )
    block_tokens = cfg.block_tokens
    blocks = args.context_tokens / block_tokens
    budget = args.ttft_slo - cfg.follow_prompt_tokens / (resources.prefill_blocks_s * block_tokens)
    rebuild = blocks / resources.recompute_blocks_s
    fetch = blocks / resources.transfer_blocks_s
    restore = blocks / resources.restore_blocks_s
    if budget < restore:
        zone = "nothing fits"
    elif rebuild <= budget:
        zone = "cheap rebuild"
    elif fetch <= budget:
        zone = "cheap fetch"
    else:
        zone = "must be local"
    print(
        f"block={resources.model.block_bytes / 2**20:.2f} MiB  hbm={cfg.hbm_blocks} blk  "
        f"host={cfg.host_blocks} blk  slots={cfg.decode_slots}"
    )
    print(
        f"context={args.context_tokens} tok  budget={budget:.2f}s  "
        f"rebuild={rebuild:.2f}s  fetch={fetch:.2f}s  restore={restore:.2f}s  [{zone}]\n"
    )

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = aggregate(run_experiment(cfg, seeds, args.output))
    columns = (
        "policy",
        "requests",
        "continuation_attainment",
        "slo_attainment",
        "p99_ttft_s",
        "transfer_blocks_per_success",
        "restore_blocks_per_success",
        "recompute_blocks_per_success",
        "hbm_block_seconds_per_success",
        "unused_preparation_ratio",
        "prep_accept_rate",
        "prep_skip_already_feasible_rate",
        "prep_skip_nonpositive_rate",
        "foreground_transfer_blocks",
        "deferred_admissions",
        "exhausted_replenishments",
    )
    widths = {
        column: max(
            len(column),
            *(
                len(f"{row[column]:.4f}") if isinstance(row[column], float) else len(str(row[column]))
                for row in rows
            ),
        )
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        print(
            "  ".join(
                (f"{row[column]:.4f}" if isinstance(row[column], float) else str(row[column])).ljust(
                    widths[column]
                )
                for column in columns
            )
        )
    print(f"\nDetailed CSV files: {args.output.resolve()}")


if __name__ == "__main__":
    main()
