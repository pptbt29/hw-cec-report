#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from repkv_sim.simulator import Config, aggregate, run_experiment


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the RepKV discrete-event simulation prototype.")
    p.add_argument("--seeds", default="1,2,3", help="Comma-separated random seeds")
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--sessions", type=int, default=40)
    p.add_argument("--horizon", type=float, default=220.0)
    p.add_argument("--hbm-blocks", type=int, default=180)
    p.add_argument("--ttft-slo", type=float, default=2.0)
    p.add_argument("--prediction-error", type=float, default=5.0)
    p.add_argument("--output", default="outputs")
    return p


def main() -> None:
    args = parser().parse_args()
    cfg = Config(
        nodes=args.nodes,
        sessions=args.sessions,
        horizon_s=args.horizon,
        hbm_blocks=args.hbm_blocks,
        ttft_slo_s=args.ttft_slo,
        prediction_error_s=args.prediction_error,
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    output_dir = Path(__file__).parent / args.output
    rows = aggregate(run_experiment(cfg, seeds, output_dir))

    columns = [
        ("policy", 13),
        ("slo_goodput_rps", 14),
        ("slo_attainment", 14),
        ("p99_ttft_s", 11),
        ("transfer_blocks_per_success", 17),
        ("recompute_blocks_per_success", 18),
        ("hbm_block_seconds_per_success", 16),
        ("unused_preparation_ratio", 14),
    ]
    print("\nRepKV prototype — mean across seeds")
    print(" ".join(name.ljust(width) for name, width in columns))
    print(" ".join("-" * width for _, width in columns))
    for row in rows:
        values = []
        for name, width in columns:
            value = row[name]
            rendered = str(value) if isinstance(value, str) else f"{value:.4f}"
            values.append(rendered.ljust(width))
        print(" ".join(values))
    print(f"\nRaw CSV files: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
