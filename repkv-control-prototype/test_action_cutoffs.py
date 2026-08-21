#!/usr/bin/env python3
"""C1 and C2 are ordered action lengths; recompute and transfer can swap."""
from __future__ import annotations

import unittest

from repkv_sim.resources import HardwareSpec, ResourceModel


class OrderedRemoteCutoffs(unittest.TestCase):
    def test_c1_is_always_the_shorter_action(self) -> None:
        for network_bytes_s in (12.5e9, 0.125e9, 2.5e7):
            model = ResourceModel(hardware=HardwareSpec(network_bytes_s=network_bytes_s))
            cut = model.thresholds(2.61)
            self.assertAlmostEqual(cut["c1_tokens"], min(cut["recompute_tokens"], cut["transfer_tokens"]))
            self.assertAlmostEqual(cut["c2_tokens"], max(cut["recompute_tokens"], cut["transfer_tokens"]))
            self.assertLessEqual(cut["c1_tokens"], cut["c2_tokens"])

    def test_fast_link_puts_recompute_at_c1(self) -> None:
        cut = ResourceModel().thresholds(2.61)
        self.assertLess(cut["recompute_tokens"], cut["transfer_tokens"])
        self.assertAlmostEqual(cut["c1_tokens"], cut["recompute_tokens"])
        self.assertAlmostEqual(cut["c2_tokens"], cut["transfer_tokens"])

    def test_slow_link_swaps_transfer_onto_c1(self) -> None:
        cut = ResourceModel(hardware=HardwareSpec(network_bytes_s=2.5e7)).thresholds(1.40)
        self.assertGreater(cut["recompute_tokens"], cut["transfer_tokens"])
        self.assertAlmostEqual(cut["c1_tokens"], cut["transfer_tokens"])
        self.assertAlmostEqual(cut["c2_tokens"], cut["recompute_tokens"])
