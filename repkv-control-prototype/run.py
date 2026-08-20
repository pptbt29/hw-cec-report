#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from repkv_sim import Config, aggregate, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the causal RepKV control-plane prototype.")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--sessions", type=int, default=48)
    parser.add_argument("--horizon", type=float, default=240.0)
    parser.add_argument("--hbm-blocks", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(nodes=args.nodes, sessions=args.sessions, horizon_s=args.horizon, hbm_blocks=args.hbm_blocks)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = aggregate(run_experiment(cfg, seeds, args.output))
    columns = (
        "policy",
        "slo_goodput_rps",
        "slo_attainment",
        "p99_ttft_s",
        "transfer_blocks_per_success",
        "restore_blocks_per_success",
        "recompute_blocks_per_success",
        "hbm_block_seconds_per_success",
        "unused_preparation_ratio",
    )
    widths = {column: max(len(column), *(len(f"{row[column]:.4f}") if isinstance(row[column], float) else len(str(row[column])) for row in rows)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        print("  ".join((f"{row[column]:.4f}" if isinstance(row[column], float) else str(row[column])).ljust(widths[column]) for column in columns))
    print(f"\nDetailed CSV files: {args.output.resolve()}")


if __name__ == "__main__":
    main()
