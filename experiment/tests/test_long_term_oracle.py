import unittest

from sim.compute_simulator import get_hardware
from sim.data_generator import Request
from sim.large_model import get_model
from sim.network import NetworkSimulator, default_topology
from sim.node import build_cluster
from sim.router import Action, Policy, Router, StateMode


class OracleLongTermHeuristicTest(unittest.TestCase):
    def setUp(self):
        self.model = get_model("CodeLlama34B")
        self.cluster = build_cluster(
            self.model,
            get_hardware("A800T-A2"),
            NetworkSimulator(default_topology()),
            kv_capacity_bytes=80e9,
        )

    def request(
        self,
        turn,
        *,
        arrival_ms,
        entry_node=0,
        prefix_tokens=1024,
        input_tokens=128,
        output_len=16,
        expected_turns=2.0,
    ):
        return Request(
            request_id=turn,
            session_id="s1",
            model_name=self.model.name,
            model_type="LLM",
            arrival_ms=arrival_ms,
            entry_node=entry_node,
            priority="normal",
            group_name="test",
            sla_ms=1e12,
            prompt_text_tokens=input_tokens,
            visual_tokens=0,
            state_tokens=0,
            input_tokens=input_tokens,
            output_len=output_len,
            prefix_id=f"{self.model.name}:s1",
            prefix_tokens=prefix_tokens,
            is_session_first=turn == 0,
            turn_index=turn,
            expected_session_turns=expected_turns,
        )

    def test_uses_actual_remaining_requests_not_expected_session_mean(self):
        current = self.request(5, arrival_ms=0.0, expected_turns=2.0)
        future = self.request(6, arrival_ms=10_000.0, expected_turns=2.0)
        router = Router(
            self.model,
            self.cluster,
            Policy.LONG_TERM,
            oracle_session_requests={"s1": [current, future]},
        )
        self.assertEqual(router._known_future_requests(current), [future])

    def test_future_value_discounts_each_known_future_request(self):
        current = self.request(0, arrival_ms=0.0, prefix_tokens=0)
        future_1 = self.request(1, arrival_ms=1_000_000.0, prefix_tokens=512)
        future_2 = self.request(2, arrival_ms=2_000_000.0, prefix_tokens=768)
        gamma = 0.5
        router = Router(
            self.model,
            self.cluster,
            Policy.LONG_TERM,
            gamma=gamma,
            oracle_session_requests={"s1": [current, future_1, future_2]},
        )
        current_cost = router._cost(current, Action(0, StateMode.FRESH))

        def local_service_cost(request):
            compute = self.cluster.node(0).compute
            return (
                compute.estimate_incremental_prefill(
                    request.prefix_tokens,
                    request.input_tokens,
                ).prefill_ms
                + compute.estimate_amortized_decode(
                    max(request.output_len - 1, 0),
                    request.prefix_tokens + request.input_tokens,
                    32,
                ).total_ms
            )

        expected = local_service_cost(future_1) + gamma * local_service_cost(
            future_2
        )
        self.assertAlmostEqual(
            router._future_value(current, current_cost), expected, places=8
        )

    def test_future_recompute_prices_the_complete_historical_prefix(self):
        current = self.request(0, arrival_ms=0.0, prefix_tokens=0)
        future = self.request(
            1,
            arrival_ms=10_000.0,
            entry_node=1,
            prefix_tokens=8192,
            input_tokens=32,
            output_len=8,
        )
        router = Router(
            self.model,
            self.cluster,
            Policy.LONG_TERM,
            oracle_session_requests={"s1": [current, future]},
        )
        calls = []
        for node in self.cluster.nodes.values():
            original = node.compute.recompute_time_ms

            def record(tokens, original=original):
                calls.append(tokens)
                return original(tokens)

            node.compute.recompute_time_ms = record

        current_cost = router._cost(current, Action(0, StateMode.FRESH))
        router._future_value(current, current_cost)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {future.prefix_tokens})
        self.assertNotIn(future.input_tokens + future.output_len, calls)

    def test_greedy_rollout_replays_all_remaining_turns_without_discount(self):
        current = self.request(
            0, arrival_ms=0.0, prefix_tokens=0,
            input_tokens=128, output_len=16,
        )
        future_1 = self.request(
            1, arrival_ms=1_000_000.0, prefix_tokens=144,
            input_tokens=32, output_len=8,
        )
        future_2 = self.request(
            2, arrival_ms=2_000_000.0, prefix_tokens=184,
            input_tokens=24, output_len=4,
        )
        router = Router(
            self.model,
            self.cluster,
            Policy.GREEDY_ROLLOUT,
            gamma=0.1,
            oracle_session_requests={
                "s1": [current, future_1, future_2],
            },
        )
        current_cost = router._cost(current, Action(0, StateMode.FRESH))

        def local_service_cost(request):
            compute = self.cluster.node(0).compute
            return (
                compute.estimate_incremental_prefill(
                    request.prefix_tokens,
                    request.input_tokens,
                ).prefill_ms
                + compute.estimate_amortized_decode(
                    max(request.output_len - 1, 0),
                    request.prefix_tokens + request.input_tokens,
                    32,
                ).total_ms
            )

        expected = local_service_cost(future_1) + local_service_cost(future_2)
        self.assertAlmostEqual(
            router._greedy_rollout_value(current, current_cost),
            expected,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
