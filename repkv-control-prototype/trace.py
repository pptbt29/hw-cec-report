#!/usr/bin/env python3
"""Interactive state trace for the RepKV logic prototype."""

from __future__ import annotations

import argparse
import json

from repkv_sim import Config, Simulator, generate_workload


BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
CLEAR = "\x1b[2J\x1b[H"


def render(simulator: Simulator, clear: bool = True) -> None:
    if clear:
        print(CLEAR, end="")
    print(f"{BOLD}RepKV control trace — PROTOTYPE{RESET}")
    print(f"{DIM}time={simulator.now:.1f}s policy={simulator.policy}{RESET}\n")
    print(f"{BOLD}Nodes{RESET}")
    for node in simulator.nodes:
        replicas = ", ".join(
            f"s{sid}:h{replica.hbm_prefix}/host{replica.host_prefix}/prep{replica.prepared_blocks}"
            for sid, replica in sorted(node.replicas.items())
            if replica.hbm_prefix or replica.host_prefix
        ) or "-"
        print(
            f"n{node.nid}: committed={node.committed}/{node.capacity} busy_until={node.busy_until:.2f} "
            f"active={dict(sorted(node.active_reservations.items()))} replicas=[{replicas}]"
        )
    print(f"\n{BOLD}In-flight batches{RESET}")
    if simulator.inflight:
        for batch in sorted(simulator.inflight, key=lambda item: item.complete_at):
            print(f"s{batch.sid}->n{batch.nid} {batch.method}[{batch.start},{batch.end}) completes={batch.complete_at:.2f}")
    else:
        print("-")
    print(f"\n{BOLD}Last events{RESET}")
    print("\n".join(simulator.last_events[-8:]) or "-")
    print(f"\n{BOLD}Top complete plans{RESET}")
    for candidate in sorted(simulator.last_candidates, key=lambda item: (-item.score, item.sid))[:5]:
        actions = "+".join(f"{segment.method}:{segment.blocks}" for segment in candidate.plan.segments)
        print(
            f"s{candidate.sid}->n{candidate.nid} target={candidate.plan.target_prefix} "
            f"actions={actions} gain={candidate.expected_gain:.4f} net={candidate.net_value:.4f} slack={candidate.slack_s:.2f}"
        )
    if not simulator.last_candidates:
        print("-")
    print(f"\n{BOLD}Router-published readiness{RESET}")
    for row in simulator.readiness_snapshot()[:12]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"\n{BOLD}[n]{RESET} next tick  {BOLD}[r]{RESET} run to next arrival  {BOLD}[q]{RESET} quit")


def main() -> None:
    global BOLD, DIM, RESET, CLEAR
    parser = argparse.ArgumentParser(description="Drive one RepKV state trace.")
    parser.add_argument("--steps", type=int, help="non-interactive number of ticks")
    parser.add_argument("--no-ansi", action="store_true")
    args = parser.parse_args()
    if args.no_ansi:
        BOLD = DIM = RESET = CLEAR = ""
    cfg = Config(nodes=3, sessions=4, horizon_s=100.0, hbm_blocks=70)
    simulator = Simulator(cfg, generate_workload(cfg, seed=7), "repkv", seed=7)
    if args.steps is not None:
        for _ in range(args.steps):
            simulator.step()
        render(simulator, clear=not args.no_ansi)
        return
    while True:
        render(simulator)
        command = input("> ").strip().lower()
        if command == "q":
            break
        if command == "r":
            current_requests = simulator.metrics.requests
            while simulator.metrics.requests == current_requests and simulator.now < cfg.horizon_s:
                simulator.step()
        else:
            simulator.step()


if __name__ == "__main__":
    main()
