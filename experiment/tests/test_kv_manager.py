from dataclasses import replace
from types import SimpleNamespace
import unittest

from sim.compute_simulator import get_hardware
from sim.data_generator import Request
from sim.kv_cache import make_blocks
from sim.kv_manager import KVManagerConfig, ProactiveKVManager
from sim.large_model import get_model
from sim.network import NetworkSimulator, default_topology
from sim.node import build_cluster
from sim.router import Policy, simulate_trace


class ProactiveKVManagerTest(unittest.TestCase):
    def setUp(self):
        self.model = get_model("CodeLlama34B")
        self.cluster = build_cluster(
            self.model,
            get_hardware("A800T-A2"),
            NetworkSimulator(default_topology()),
            kv_capacity_bytes=20e9,
        )
        self.blocks = make_blocks(
            self.model,
            session_id="s1",
            prefix_id="CodeLlama34B:s1",
            num_tokens=160,
            owner=0,
        )
        self.cluster.node(0).kv_store.insert(self.blocks, 0.0)
        for block in self.blocks:
            self.cluster.kv.register(0, block)
        self.request = SimpleNamespace(
            session_id="s1",
            prefix_id="CodeLlama34B:s1",
            entry_node=0,
            mobility_active=True,
            mobility_ratio=0.2,
            expected_session_turns=6.0,
            turn_index=1,
            expected_interarrival_ms=1000.0,
        )

    def test_probability_weighted_prefix_replication(self):
        manager = ProactiveKVManager(
            self.model,
            self.cluster,
            KVManagerConfig(
                base_replication_factor=1.0,
                max_replication_fraction=0.25,
                continuation_reference_turns=4.0,
            ),
        )
        result = manager.schedule_after_request(self.request, 0, 0.0)

        self.assertEqual(result.scheduled_tasks, 2)
        self.assertEqual(result.scheduled_blocks, 2)
        first_hash = self.blocks[0].block_hash
        self.assertNotIn(1, self.cluster.kv.locate(first_hash))

        manager.advance(1e9)
        self.assertIn(1, self.cluster.kv.locate(first_hash))
        self.assertIn(2, self.cluster.kv.locate(first_hash))
        self.assertEqual(self.blocks[0].owner, 0)

    def test_current_ingress_is_prioritized_when_execution_stays_at_old_owner(self):
        self.request.entry_node = 1
        manager = ProactiveKVManager(
            self.model,
            self.cluster,
            KVManagerConfig(
                base_replication_factor=1.0,
                max_replication_fraction=0.25,
                continuation_reference_turns=4.0,
            ),
        )

        self.assertAlmostEqual(
            manager._transition_probability(self.request, 1), 0.8
        )
        self.assertAlmostEqual(
            manager._transition_probability(self.request, 2), 0.1
        )
        result = manager.schedule_after_request(
            self.request,
            source_node=0,
            t_now=0.0,
        )
        self.assertIn(1, result.by_destination)
        self.assertIn(2, result.by_destination)
        self.assertGreater(
            result.by_destination[1],
            result.by_destination[2],
        )

    def test_no_replication_without_remaining_turns(self):
        self.request.turn_index = 5
        manager = ProactiveKVManager(
            self.model, self.cluster, KVManagerConfig()
        )
        result = manager.schedule_after_request(self.request, 0, 0.0)
        self.assertEqual(result.scheduled_bytes, 0)
        self.assertEqual(manager.stats["skipped_no_continuation"], 1)

    def test_simulation_activity_uses_oracle_remaining_requests(self):
        # The configured mean says the session should already have ended, but
        # the generated trace contains one real future request.  Oracle-mode
        # simulation must follow the trace rather than the distribution mean.
        self.request.expected_session_turns = 1.0
        future = SimpleNamespace(turn_index=self.request.turn_index + 1)
        manager = ProactiveKVManager(
            self.model,
            self.cluster,
            KVManagerConfig(continuation_reference_turns=1.0),
            oracle_session_requests={"s1": [self.request, future]},
        )
        result = manager.schedule_after_request(self.request, 0, 0.0)
        self.assertGreater(result.scheduled_bytes, 0)

    def test_no_replication_before_mobility_phase(self):
        self.request.mobility_active = False
        manager = ProactiveKVManager(
            self.model, self.cluster, KVManagerConfig()
        )
        result = manager.schedule_after_request(self.request, 0, 0.0)
        self.assertEqual(result.scheduled_tasks, 0)
        self.assertEqual(manager.stats["skipped_inactive"], 1)

    def test_oracle_targets_actual_future_ingress_before_mobility(self):
        self.request.mobility_active = False
        future = SimpleNamespace(
            turn_index=self.request.turn_index + 1,
            entry_node=2,
            arrival_ms=1000.0,
        )
        manager = ProactiveKVManager(
            self.model,
            self.cluster,
            KVManagerConfig(),
            oracle_session_requests={"s1": [self.request, future]},
            oracle_placement=True,
        )
        result = manager.schedule_after_request(self.request, 0, 0.0)
        self.assertEqual(result.scheduled_tasks, 1)
        self.assertEqual(set(result.by_destination), {2})

    def test_oracle_commits_completed_blocks_progressively(self):
        future = SimpleNamespace(
            turn_index=self.request.turn_index + 1,
            entry_node=1,
            arrival_ms=1000.0,
        )
        manager = ProactiveKVManager(
            self.model,
            self.cluster,
            KVManagerConfig(),
            oracle_session_requests={"s1": [self.request, future]},
            oracle_placement=True,
        )
        result = manager.schedule_after_request(self.request, 0, 0.0)
        self.assertEqual(result.scheduled_tasks, 1)
        task = manager._tasks[0]
        midpoint = task.start_ms + task.latency_ms + (
            task.completion_ms - task.start_ms - task.latency_ms
        ) / 2.0
        manager.advance(midpoint)
        ready = sum(
            1 for block in self.blocks
            if 1 in self.cluster.kv.locate(block.block_hash)
        )
        self.assertGreater(ready, 0)
        self.assertLess(ready, len(self.blocks))

    def test_policy_boundary_and_block_level_switches(self):
        request = Request(
            request_id=0,
            session_id="active-session",
            model_name=self.model.name,
            model_type="LLM",
            arrival_ms=0.0,
            entry_node=0,
            priority="normal",
            group_name="test",
            sla_ms=1e9,
            prompt_text_tokens=160,
            visual_tokens=0,
            state_tokens=0,
            input_tokens=160,
            output_len=16,
            prefix_id=f"{self.model.name}:active-session",
            prefix_tokens=0,
            is_session_first=True,
            turn_index=0,
            expected_session_turns=6.0,
            home_node=0,
            mobility_active=True,
            mobility_ratio=0.2,
            expected_interarrival_ms=1000.0,
        )
        future = replace(
            request,
            request_id=1,
            arrival_ms=1000.0,
            turn_index=1,
            is_session_first=False,
            prefix_tokens=request.input_tokens + request.output_len,
        )
        config = KVManagerConfig(enabled=True)
        common = dict(
            requests=[request, future],
            model=self.model,
            hardware=get_hardware("A800T-A2"),
            num_nodes=3,
            kv_capacity_bytes=20e9,
            kv_manager_config=config,
        )

        passive = simulate_trace(
            Policy.LONG_TERM,
            network=NetworkSimulator(default_topology()),
            long_term_block_level_kv=False,
            **common,
        )
        passive_block = simulate_trace(
            Policy.LONG_TERM,
            network=NetworkSimulator(default_topology()),
            long_term_block_level_kv=True,
            **common,
        )
        proactive = simulate_trace(
            Policy.LONG_TERM_KV,
            network=NetworkSimulator(default_topology()),
            long_term_kv_block_level_kv=True,
            **common,
        )
        greedy_proactive = simulate_trace(
            Policy.GREEDY_KV,
            network=NetworkSimulator(default_topology()),
            long_term_kv_block_level_kv=True,
            **common,
        )
        oracle = simulate_trace(
            Policy.ORACLE_KV,
            network=NetworkSimulator(default_topology()),
            long_term_kv_block_level_kv=True,
            **common,
        )
        oracle_prefetch = simulate_trace(
            Policy.ORACLE_PREFETCH,
            network=NetworkSimulator(default_topology()),
            long_term_kv_block_level_kv=True,
            **common,
        )

        self.assertFalse(passive["block_level_kv_enabled"])
        self.assertFalse(passive["proactive_kv_enabled"])
        self.assertEqual(passive["placement_scheduled_tasks"], 0)
        self.assertTrue(passive_block["block_level_kv_enabled"])
        self.assertFalse(passive_block["proactive_kv_enabled"])
        self.assertEqual(passive_block["placement_scheduled_tasks"], 0)
        self.assertTrue(proactive["block_level_kv_enabled"])
        self.assertTrue(proactive["proactive_kv_enabled"])
        self.assertGreater(proactive["placement_scheduled_tasks"], 0)
        self.assertEqual(greedy_proactive["policy"], "greedy_kv")
        self.assertTrue(greedy_proactive["block_level_kv_enabled"])
        self.assertTrue(greedy_proactive["proactive_kv_enabled"])
        self.assertGreater(
            greedy_proactive["placement_scheduled_tasks"], 0
        )
        self.assertEqual(
            greedy_proactive["future_driven_action_count"], 0
        )
        self.assertTrue(oracle["oracle_kv_placement"])
        self.assertTrue(oracle["proactive_kv_enabled"])
        self.assertTrue(oracle_prefetch["oracle_prefetch"])
        self.assertTrue(oracle_prefetch["proactive_kv_enabled"])


if __name__ == "__main__":
    unittest.main()
