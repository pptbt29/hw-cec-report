#!/usr/bin/env python3
"""Eviction must price a prefix against every node's copy, not this GPU alone."""
from __future__ import annotations

import unittest

from repkv_sim.simulator import Config, KVReplica, Simulator, generate_workload


def _idle_session(workload, min_turns: int = 2) -> int:
    for sid in sorted(workload.session_start):
        turns = sum(1 for turn in workload.turns if turn.sid == sid)
        if turns >= min_turns:
            return sid
    raise AssertionError("workload has no multi-turn session")


class EvictionValuesGlobalCopies(unittest.TestCase):
    def setUp(self) -> None:
        # Deadline sits between a local HBM hit and a host restore of one batch,
        # so only a resident HBM copy is predicted-SLO-feasible.
        self.cfg = Config(
            nodes=2,
            sessions=16,
            concurrent_sessions=0,
            horizon_s=40.0,
            ttft_slo_s=0.045,
            first_prompt_tokens=128,
            follow_prompt_tokens=128,
            output_tokens=64,
            size_sigma=0.05,
        )
        self.workload = generate_workload(self.cfg, seed=3)
        self.sid = _idle_session(self.workload)
        self.context = self.cfg.block_batch
        self.sim = Simulator(self.cfg, self.workload, "repkv", seed=3)
        self.sim.now = 8.0
        state = self.sim.sessions[self.sid]
        state.completed_turn = 0
        state.context_blocks = self.context
        state.idle_since = 0.0
        state.active_until = 0.0
        state.active_node = None
        self.sim._invalidate()

    def _put(self, nid: int, hbm: int, host: int | None = None) -> None:
        self.sim.nodes[nid].replicas[self.sid] = KVReplica(
            hbm_prefix=hbm, host_prefix=hbm if host is None else host, last_used=0.0
        )

    def test_hit_fits_restore_does_not(self) -> None:
        prediction = self.sim._prediction(self.sid)
        self.assertIsNotNone(prediction)
        prompt = prediction.predicted_prompt_blocks / self.cfg.prefill_blocks_s
        restore = self.context / self.cfg.foreground_rates["restore"]
        self.assertLess(prompt, self.cfg.ttft_slo_s)
        self.assertGreater(restore + prompt, self.cfg.ttft_slo_s)

    def test_spare_replica_is_cheaper_than_the_only_copy(self) -> None:
        self._put(0, self.context)
        unique = self.sim._eviction_score(self.sid, 0)
        self._put(1, self.context)
        self.sim._invalidate()
        spare = self.sim._eviction_score(self.sid, 0)
        self.assertGreater(unique, spare)
        self.assertGreater(unique / spare, 5.0)

    def test_peer_dram_copy_that_still_misses_does_not_make_local_hbm_spare(self) -> None:
        """In the must-hit zone a peer's DRAM copy cannot cover the deadline."""
        self._put(0, self.context)
        self._put(1, 0, host=self.context)
        self.sim._invalidate()
        with_peer_dram = self.sim._eviction_score(self.sid, 0)
        del self.sim.nodes[1].replicas[self.sid]
        self.sim._invalidate()
        unique = self.sim._eviction_score(self.sid, 0)
        self.assertAlmostEqual(with_peer_dram, unique, places=4)


if __name__ == "__main__":
    unittest.main()
