#!/usr/bin/env python3
"""Reproduce the main configuration of `PROTOTYPE_VERDICT.md` section 6.1.

Defaults place the run in the zone where fetching KV from a peer still fits the
deadline but recomputing it does not, which is where the policies separate. Use
`--ttft-slo` to move across the two thresholds reported there.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from repkv_sim import Config, HardwareSpec, ModelSpec, ResourceModel, aggregate, run_experiment

GiB = 2**30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the causal RepKV control-plane prototype.")
    parser.add_argument(
        "--profile",
        choices=("dc", "edge"),
        default="dc",
        help="dc: 70B/100GbE/32K; edge: 7B/1GbE/1500 (PROTOTYPE_VERDICT §6)",
    )
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--sessions", type=int, default=None, help="size of the session script pool")
    parser.add_argument(
        "--concurrent-sessions",
        type=int,
        default=None,
        help="live sessions held by replenishment; 0 starts the whole pool at once",
    )
    parser.add_argument("--horizon", type=float, default=900.0)
    parser.add_argument("--ttft-slo", type=float, default=3.0)
    parser.add_argument("--context-tokens", type=int, default=None)
    parser.add_argument("--host-gib", type=float, default=None, help="DRAM tier given to KV per node")
    parser.add_argument("--network-gbytes", type=float, default=None, help="inter-node bandwidth")
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


def _profile_defaults(profile: str) -> dict[str, float | int]:
    if profile == "edge":
        return {
            "nodes": 4,
            "sessions": 2000,
            "concurrent_sessions": 82,
            "context_tokens": 1500,
            "host_gib": 32.0,
            "network_gbytes": 0.125,
            "follow_prompt_tokens": 300,
        }
    return {
        "nodes": 4,
        "sessions": 2000,
        "concurrent_sessions": 90,
        "context_tokens": 32000,
        "host_gib": 256.0,
        "network_gbytes": 12.5,
        "follow_prompt_tokens": 3000,
    }


def main() -> None:
    args = parse_args()
    defaults = _profile_defaults(args.profile)
    nodes = defaults["nodes"] if args.nodes is None else args.nodes
    sessions = defaults["sessions"] if args.sessions is None else args.sessions
    concurrent = (
        defaults["concurrent_sessions"] if args.concurrent_sessions is None else args.concurrent_sessions
    )
    context_tokens = defaults["context_tokens"] if args.context_tokens is None else args.context_tokens
    host_gib = defaults["host_gib"] if args.host_gib is None else args.host_gib
    network_gbytes = defaults["network_gbytes"] if args.network_gbytes is None else args.network_gbytes
    follow_prompt_tokens = int(defaults["follow_prompt_tokens"])
    if args.profile == "edge":
        model = ModelSpec(layers=32, kv_heads=8, head_dim=128, parameters=7.0e9)
        hardware = HardwareSpec(
            devices=1,
            device_flops=2.0e13,
            achieved_utilisation=0.35,
            hbm_capacity_bytes=24 * GiB,
            network_bytes_s=network_gbytes * 1e9,
            host_capacity_bytes=host_gib * GiB,
            decode_batch=8,
            decode_context_tokens=context_tokens,
        )
    else:
        model = ModelSpec()
        hardware = HardwareSpec(
            decode_context_tokens=context_tokens,
            host_capacity_bytes=host_gib * GiB,
            network_bytes_s=network_gbytes * 1e9,
        )
    resources = ResourceModel(model=model, hardware=hardware)
    cfg = Config(
        nodes=int(nodes),
        sessions=int(sessions),
        concurrent_sessions=int(concurrent),
        horizon_s=args.horizon,
        ttft_slo_s=args.ttft_slo,
        resources=resources,
        hbm_blocks_override=args.hbm_blocks,
        think_scale=args.think_scale,
        first_prompt_tokens=int(context_tokens),
        follow_prompt_tokens=follow_prompt_tokens,
        min_return_probability=args.min_return_probability,
        value_based_eviction=not args.lru_eviction,
    )
    block_tokens = cfg.block_tokens
    blocks = context_tokens / block_tokens
    budget = args.ttft_slo - cfg.follow_prompt_tokens / (resources.prefill_blocks_s * block_tokens)
    rebuild = blocks / resources.recompute_blocks_s
    fetch = blocks / resources.transfer_blocks_s
    restore = blocks / resources.restore_blocks_s
    cut = resources.thresholds(budget)
    context = int(context_tokens)
    if context <= cut["c1_tokens"]:
        zone = "both remote actions"
    elif context <= cut["c2_tokens"]:
        zone = "recompute only" if cut["recompute_tokens"] >= cut["transfer_tokens"] else "transfer only"
    elif context <= cut["restore_tokens"]:
        zone = "must be local"
    else:
        zone = "hbm hit only"
    print(
        f"block={resources.model.block_bytes / 2**20:.2f} MiB  hbm={cfg.hbm_blocks} blk  "
        f"host={cfg.host_blocks} blk  slots={cfg.decode_slots}"
    )
    print(
        f"context={context} tok  budget={budget:.2f}s  "
        f"rebuild={rebuild:.2f}s  fetch={fetch:.2f}s  restore={restore:.2f}s"
    )
    print(
        f"C_recompute={cut['recompute_tokens']:.0f}  C_transfer={cut['transfer_tokens']:.0f}  "
        f"C1={cut['c1_tokens']:.0f}  C2={cut['c2_tokens']:.0f}  "
        f"C_restore={cut['restore_tokens']:.0f}  [{zone}]\n"
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
        "ttft_residual_mean_s",
        "ttft_residual_mae_s",
        "queue_residual_mean_s",
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
