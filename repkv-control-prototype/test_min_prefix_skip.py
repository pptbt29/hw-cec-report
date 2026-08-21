#!/usr/bin/env python3
"""Minimum-prefix preparation skips work that full-prefix preparation still does."""
from __future__ import annotations

import unittest

from repkv_sim.planner import (
    PrefixAvailability,
    cheapest_preparation,
    decide_minimum_slo_preparation,
)
from repkv_sim.simulator import Config, KVReplica, Simulator, generate_workload


class MinimumSloVetoes(unittest.TestCase):
    def setUp(self) -> None:
        self.fg = {"restore": 100.0, "transfer": 50.0, "recompute": 5.0}
        self.bg = dict(self.fg)
        self.costs = {"restore": 1.0, "transfer": 1.0, "recompute": 2.0}

    def _decide(self, availability: PrefixAvailability, queue_s: float = 0.0) -> tuple:
        return decide_minimum_slo_preparation(
            context_blocks=40,
            availability=availability,
            queue_s=queue_s,
            prompt_s=0.1,
            slo_s=1.0,
            time_limit_s=30.0,
            foreground_rates=self.fg,
            background_rates=self.bg,
            method_costs=self.costs,
        )

    def test_skips_when_host_restore_already_meets_slo(self) -> None:
        availability = PrefixAvailability(local_hbm=0, local_host=40, remote_hbm=0)
        plan, reason = self._decide(availability)
        self.assertEqual(reason, "already_feasible")
        self.assertIsNone(plan)
        full = cheapest_preparation(0, 40, availability, 30.0, self.bg, self.costs)
        self.assertIsNotNone(full)
        self.assertEqual(full.target_prefix, 40)
        self.assertTrue(full.segments)

    def test_prepares_when_host_copy_is_not_durable(self) -> None:
        availability = PrefixAvailability(local_hbm=0, local_host=40, remote_hbm=0)
        plan, reason = decide_minimum_slo_preparation(
            context_blocks=40,
            availability=availability,
            queue_s=0.0,
            prompt_s=0.1,
            slo_s=1.0,
            time_limit_s=30.0,
            foreground_rates=self.fg,
            background_rates=self.bg,
            method_costs=self.costs,
            copy_persist=0.3,
            already_feasible_probability=0.9,
        )
        self.assertEqual(reason, "ok")
        self.assertIsNotNone(plan)
        self.assertGreater(plan.target_prefix, 0)

    def test_skips_when_remote_transfer_already_meets_slo(self) -> None:
        plan, reason = self._decide(PrefixAvailability(0, 0, 40))
        self.assertEqual(reason, "already_feasible")
        self.assertIsNone(plan)

    def test_skips_when_queue_alone_exceeds_budget(self) -> None:
        plan, reason = self._decide(PrefixAvailability(0, 0, 0), queue_s=2.0)
        self.assertEqual(reason, "queue_exceeds_budget")
        self.assertIsNone(plan)

    def test_plans_when_nothing_currently_fits(self) -> None:
        slow = {"restore": 1.0, "transfer": 1.0, "recompute": 0.1}
        plan, reason = decide_minimum_slo_preparation(
            context_blocks=40,
            availability=PrefixAvailability(0, 0, 40),
            queue_s=0.0,
            prompt_s=0.1,
            slo_s=1.0,
            time_limit_s=100.0,
            foreground_rates=slow,
            background_rates=slow,
            method_costs=self.costs,
        )
        self.assertEqual(reason, "ok")
        self.assertIsNotNone(plan)
        self.assertGreater(plan.target_prefix, 0)


class SkipIsNotEviction(unittest.TestCase):
    """Same LRU recycling; only the preparation rule changes."""

    def setUp(self) -> None:
        self.cfg = Config(
            nodes=2,
            sessions=16,
            concurrent_sessions=0,
            horizon_s=40.0,
            ttft_slo_s=0.08,
            first_prompt_tokens=128,
            follow_prompt_tokens=128,
            output_tokens=64,
            size_sigma=0.05,
            value_based_eviction=False,
        )
        self.workload = generate_workload(self.cfg, seed=3)
        self.sid = next(
            sid
            for sid in sorted(self.workload.session_start)
            if sum(1 for turn in self.workload.turns if turn.sid == sid) >= 2
        )
        self.context = self.cfg.block_batch

    def _idle(self, policy: str) -> Simulator:
        sim = Simulator(self.cfg, self.workload, policy, seed=3)
        sim.now = 8.0
        state = sim.sessions[self.sid]
        state.completed_turn = 0
        state.context_blocks = self.context
        state.idle_since = 0.0
        state.active_until = 0.0
        state.active_node = None
        sim.nodes[0].replicas[self.sid] = KVReplica(
            hbm_prefix=0, host_prefix=self.context, last_used=0.0
        )
        sim._invalidate()
        restore = self.context / sim.cfg.foreground_rates["restore"]
        recompute = self.context / sim.cfg.foreground_rates["recompute"]
        prompt = sim._prediction(self.sid).predicted_prompt_blocks / sim.cfg.prefill_blocks_s
        self.assertLess(restore + prompt, sim.cfg.ttft_slo_s)
        self.assertGreater(recompute + prompt, sim.cfg.ttft_slo_s)
        return sim

    def test_unique_host_copy_is_prepared_eager_still_copies(self) -> None:
        repkv = self._idle("repkv")
        eager = self._idle("eager_full")
        repkv_own = [candidate for candidate in repkv._collect_candidates() if candidate.sid == self.sid]
        eager_own = [candidate for candidate in eager._collect_candidates() if candidate.sid == self.sid]
        self.assertTrue(repkv_own)
        self.assertGreater(repkv.metrics.prep_accepted, 0)
        self.assertTrue(eager_own)
        self.assertGreater(eager.metrics.prep_accepted, 0)
        self.assertEqual(eager.metrics.prep_skip_already_feasible, 0)

    def test_redundant_host_copies_still_skip(self) -> None:
        repkv = self._idle("repkv")
        context = self.context
        repkv.nodes[1].replicas[self.sid] = KVReplica(
            hbm_prefix=0, host_prefix=context, last_used=0.0
        )
        repkv._invalidate()
        own = [candidate for candidate in repkv._collect_candidates() if candidate.sid == self.sid]
        self.assertEqual(own, [])
        self.assertGreater(repkv.metrics.prep_skip_already_feasible, 0)


if __name__ == "__main__":
    unittest.main()
