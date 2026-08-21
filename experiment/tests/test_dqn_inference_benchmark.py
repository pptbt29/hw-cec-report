from __future__ import annotations

import unittest

import numpy as np

from experiment.sim.dqn_inference_benchmark import (
    ACTION_DIM,
    DuelingDQNForward,
    benchmark,
    mac_count,
    parameter_count,
)


class DQNInferenceBenchmarkTest(unittest.TestCase):
    def test_analytical_counts(self) -> None:
        self.assertEqual(parameter_count(), 29_706)
        self.assertEqual(mac_count(), 29_440)

    def test_forward_shape_and_mask(self) -> None:
        model = DuelingDQNForward()
        q_values = model.forward()
        self.assertEqual(q_values.shape, (ACTION_DIM,))
        self.assertTrue(np.isfinite(q_values[model.action_mask]).all())
        self.assertTrue(np.isinf(q_values[~model.action_mask]).all())

    def test_benchmark_returns_ordered_percentiles(self) -> None:
        result = benchmark(warmup=2, iterations=20)
        latency = result["latency_us"]
        self.assertLessEqual(latency["minimum"], latency["p50"])
        self.assertLessEqual(latency["p50"], latency["p95"])
        self.assertLessEqual(latency["p95"], latency["p99"])
        self.assertLessEqual(latency["p99"], latency["maximum"])


if __name__ == "__main__":
    unittest.main()
