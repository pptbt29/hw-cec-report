"""Microbenchmark for the Dueling DQN forward path described in Section 6.4.

The benchmark measures a batch-size-one float32 forward pass for a
92 -> 128 -> 128 shared MLP with one value output and nine advantage outputs.
It uses NumPy only, fixes BLAS thread counts to one, and reports percentile
latencies together with the analytical parameter and MAC counts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from .control_plane import AgentObservationSchema


INPUT_DIM = AgentObservationSchema().observation_dim
HIDDEN_DIM = 128
ACTION_DIM = 9


def parameter_count() -> int:
    """Return the number of float32 trainable parameters."""

    weights = (
        INPUT_DIM * HIDDEN_DIM
        + HIDDEN_DIM * HIDDEN_DIM
        + HIDDEN_DIM
        + HIDDEN_DIM * ACTION_DIM
    )
    biases = HIDDEN_DIM + HIDDEN_DIM + 1 + ACTION_DIM
    return weights + biases


def mac_count() -> int:
    """Return multiply-accumulate operations in one batch-size-one pass."""

    return (
        INPUT_DIM * HIDDEN_DIM
        + HIDDEN_DIM * HIDDEN_DIM
        + HIDDEN_DIM
        + HIDDEN_DIM * ACTION_DIM
    )


def percentile(sorted_values: List[float], quantile: float) -> float:
    """Return a nearest-rank percentile from an already sorted sample."""

    if not sorted_values:
        raise ValueError("at least one latency sample is required")
    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]


class DuelingDQNForward:
    """Minimal NumPy implementation of the report's inference graph."""

    def __init__(self, seed: int = 20260729) -> None:
        rng = np.random.default_rng(seed)
        scale = np.float32(0.02)
        self.w1 = (rng.standard_normal((INPUT_DIM, HIDDEN_DIM)) * scale).astype(
            np.float32
        )
        self.b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.w2 = (rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)) * scale).astype(
            np.float32
        )
        self.b2 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.w_value = (
            rng.standard_normal((HIDDEN_DIM, 1)) * scale
        ).astype(np.float32)
        self.b_value = np.zeros(1, dtype=np.float32)
        self.w_advantage = (
            rng.standard_normal((HIDDEN_DIM, ACTION_DIM)) * scale
        ).astype(np.float32)
        self.b_advantage = np.zeros(ACTION_DIM, dtype=np.float32)
        self.observation = rng.random(INPUT_DIM, dtype=np.float32)
        self.action_mask = np.array(
            [True, False, False, False, True, True, True, False, True],
            dtype=np.bool_,
        )

    def forward(self) -> np.ndarray:
        hidden1 = np.maximum(
            self.observation @ self.w1 + self.b1,
            np.float32(0.0),
        )
        hidden2 = np.maximum(
            hidden1 @ self.w2 + self.b2,
            np.float32(0.0),
        )
        value = hidden2 @ self.w_value + self.b_value
        advantage = hidden2 @ self.w_advantage + self.b_advantage
        q_values = value + advantage - advantage.mean(dtype=np.float32)
        return np.where(self.action_mask, q_values, np.float32(np.inf))


def benchmark(warmup: int, iterations: int) -> Dict[str, object]:
    model = DuelingDQNForward()
    for _ in range(warmup):
        model.forward()

    latencies_us: List[float] = []
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        model.forward()
        end_ns = time.perf_counter_ns()
        latencies_us.append((end_ns - start_ns) / 1_000.0)

    latencies_us.sort()
    params = parameter_count()
    macs = mac_count()
    return {
        "benchmark": {
            "implementation": "NumPy float32 CPU, one BLAS thread",
            "batch_size": 1,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "input_dim": INPUT_DIM,
            "hidden_dims": [HIDDEN_DIM, HIDDEN_DIM],
            "value_outputs": 1,
            "advantage_outputs": ACTION_DIM,
            "parameter_count": params,
            "parameter_bytes_float32": params * 4,
            "macs_per_forward": macs,
            "approx_flops_per_forward": macs * 2,
        },
        "latency_us": {
            "mean": sum(latencies_us) / len(latencies_us),
            "p50": percentile(latencies_us, 0.50),
            "p95": percentile(latencies_us, 0.95),
            "p99": percentile(latencies_us, 0.99),
            "minimum": latencies_us[0],
            "maximum": latencies_us[-1],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")

    result = benchmark(args.warmup, args.iterations)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
