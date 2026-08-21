#!/usr/bin/env python3
"""Fluid queues are calibrated from residuals; unique volatile copies are not treated as certain."""
from __future__ import annotations

import unittest

from repkv_sim.simulator import ResidualTracker


class ResidualTrackerAdjusts(unittest.TestCase):
    def test_underestimation_raises_the_point_estimate(self) -> None:
        tracker = ResidualTracker(alpha=0.5, prior_std=0.1)
        for _ in range(8):
            tracker.update(0.1, 0.6)
        adjusted = tracker.adjust(0.1, z=0.0)
        self.assertGreater(adjusted, 0.35)
        self.assertGreater(tracker.mean, 0.3)

    def test_adjust_never_goes_negative(self) -> None:
        tracker = ResidualTracker(alpha=0.5, prior_std=0.1)
        tracker.update(1.0, 0.0)
        self.assertEqual(tracker.adjust(0.0, z=1.0), 0.0)

    def test_cold_start_leaves_the_fluid_estimate(self) -> None:
        tracker = ResidualTracker(alpha=0.05, prior_std=0.25)
        self.assertAlmostEqual(tracker.adjust(0.1, z=1.0), 0.1)


if __name__ == "__main__":
    unittest.main()
