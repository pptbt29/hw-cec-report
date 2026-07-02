"""Sensitivity sweep for greedy vs long-term routing.

Runs the same model/workload under request-level and session-level mobility,
then varies the long-term discount/penalty weight. This makes it easy to see
when long-term state placement pays off and when greedy current-latency routing
is the better choice.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Iterable, List

from .compute_simulator import get_hardware
from .config import default_config, load_config
from .data_generator import DataGenerator
from .large_model import get_model
from .router import Policy, simulate_trace


def _parse_gammas(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _print_row(
    granularity: str,
    policy: str,
    gamma: str,
    metrics: dict,
    greedy: dict | None = None,
) -> None:
    avg_delta = ""
    p99_delta = ""
    if greedy is not None:
        avg_delta = f"{metrics['avg_e2e_ms'] - greedy['avg_e2e_ms']:+.2f}"
        p99_delta = f"{metrics['p99_ttft_ms'] - greedy['p99_ttft_ms']:+.2f}"
    print(
        f"{granularity:<8} {policy:<13} {gamma:>5}"
        f" {metrics['avg_e2e_ms']:9.2f} {avg_delta:>9}"
        f" {metrics['p99_ttft_ms']:9.2f} {p99_delta:>9}"
        f" {metrics['cross_node_ratio'] * 100:8.1f}"
        f" {metrics['migrate_count']:6d} {metrics['recompute_count']:7d}"
        f" {metrics['migrate_bytes_mb']:9.1f}"
    )


def run_sweep(
    config_path: str | None,
    model_name: str,
    granularities: Iterable[str],
    gammas: Iterable[float],
    token_id_bytes: int,
    request_overhead_bytes: int,
    response_overhead_bytes: int,
    visual_bytes_per_token: int,
) -> None:
    base = load_config(config_path) if config_path else default_config()
    base.apply()
    hw = base.hardware or get_hardware("A800T-A2")
    model = get_model(model_name)

    print(
        "mobility policy        gamma   avg_e2e  d_avg_g"
        "  p99_ttft  d_p99_g   xnode%   migr  recomp     migMB"
    )
    for granularity in granularities:
        experiment = deepcopy(base)
        experiment.workload.mobility_granularity = granularity
        experiment.apply()
        requests = DataGenerator(experiment.workload).generate()
        cluster = experiment.cluster

        greedy = simulate_trace(
            Policy.GREEDY,
            requests,
            model,
            hw,
            experiment.new_network(),
            num_nodes=cluster.num_nodes,
            staleness_ms=cluster.staleness_ms,
            kv_capacity_bytes=cluster.kv_capacity_bytes,
            activation_reserve_bytes=cluster.activation_reserve_bytes,
            token_id_bytes=token_id_bytes,
            request_overhead_bytes=request_overhead_bytes,
            response_overhead_bytes=response_overhead_bytes,
            visual_bytes_per_token=visual_bytes_per_token,
        )
        _print_row(granularity, "greedy", "-", greedy)

        for gamma in gammas:
            lt = simulate_trace(
                Policy.LONG_TERM,
                requests,
                model,
                hw,
                experiment.new_network(),
                num_nodes=cluster.num_nodes,
                staleness_ms=cluster.staleness_ms,
                gamma=gamma,
                kv_capacity_bytes=cluster.kv_capacity_bytes,
                activation_reserve_bytes=cluster.activation_reserve_bytes,
                token_id_bytes=token_id_bytes,
                request_overhead_bytes=request_overhead_bytes,
                response_overhead_bytes=response_overhead_bytes,
                visual_bytes_per_token=visual_bytes_per_token,
            )
            _print_row(granularity, "long_term", f"{gamma:.2g}", lt, greedy)

        kv = simulate_trace(
            Policy.LONG_TERM_KV,
            requests,
            model,
            hw,
            experiment.new_network(),
            num_nodes=cluster.num_nodes,
            staleness_ms=cluster.staleness_ms,
            gamma=max(gammas),
            kv_capacity_bytes=cluster.kv_capacity_bytes,
            activation_reserve_bytes=cluster.activation_reserve_bytes,
            token_id_bytes=token_id_bytes,
            request_overhead_bytes=request_overhead_bytes,
            response_overhead_bytes=response_overhead_bytes,
            visual_bytes_per_token=visual_bytes_per_token,
        )
        _print_row(granularity, "long_term_kv", f"{max(gammas):.2g}", kv, greedy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model", default="CodeLlama34B")
    parser.add_argument(
        "--mobility-granularity",
        default="request,session,markov",
        help="Comma-separated list: request,session,markov",
    )
    parser.add_argument("--gammas", default="0.1,0.3,0.5,0.9")
    parser.add_argument("--token-id-bytes", type=int, default=4)
    parser.add_argument("--request-overhead-bytes", type=int, default=4096)
    parser.add_argument("--response-overhead-bytes", type=int, default=4096)
    parser.add_argument("--visual-bytes-per-token", type=int, default=0)
    parser.add_argument("--request-bytes-per-token", type=int)
    parser.add_argument("--response-bytes-per-token", type=int)
    args = parser.parse_args()

    granularities = [
        g.strip() for g in args.mobility_granularity.split(",") if g.strip()
    ]
    invalid = [g for g in granularities if g not in ("request", "session", "markov")]
    if invalid:
        raise ValueError(f"invalid mobility granularity: {invalid}")
    token_id_bytes = args.request_bytes_per_token or args.token_id_bytes
    request_overhead = args.request_overhead_bytes
    response_overhead = args.response_overhead_bytes
    if args.response_bytes_per_token:
        # Backward compatible approximation for old sweep scripts.
        response_overhead = args.response_bytes_per_token
    run_sweep(
        args.config,
        args.model,
        granularities,
        _parse_gammas(args.gammas),
        token_id_bytes,
        request_overhead,
        response_overhead,
        args.visual_bytes_per_token,
    )


if __name__ == "__main__":
    main()
