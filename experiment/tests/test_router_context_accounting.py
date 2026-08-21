import unittest
from dataclasses import replace

from sim.compute_simulator import get_hardware
from sim.data_generator import Request
from sim.kv_cache import make_blocks
from sim.large_model import get_model
from sim.network import NetworkSimulator, default_topology
from sim.node import build_cluster
from sim.router import Action, Policy, Router, StateMode


class RouterContextAccountingTest(unittest.TestCase):
    def setUp(self):
        self.model = get_model("CodeLlama34B")
        self.cluster = build_cluster(
            self.model,
            get_hardware("A800T-A2"),
            NetworkSimulator(default_topology()),
            kv_capacity_bytes=80e9,
        )
        self.router = Router(self.model, self.cluster, Policy.LONG_TERM)

    def request(self, *, first=False):
        return Request(
            request_id=0,
            session_id="s1",
            model_name=self.model.name,
            model_type="LLM",
            arrival_ms=0.0,
            entry_node=0,
            priority="normal",
            group_name="test",
            sla_ms=1e9,
            prompt_text_tokens=128,
            visual_tokens=0,
            state_tokens=0,
            input_tokens=128,
            output_len=16,
            prefix_id=f"{self.model.name}:s1",
            prefix_tokens=1024,
            is_session_first=first,
            turn_index=0 if first else 1,
            expected_session_turns=8.0,
        )

    def test_cached_prefix_does_not_remove_current_turn_prefill(self):
        request = self.request()
        cost = self.router._cost(
            request,
            Action(0, StateMode.LOCAL, src_node=0, hit_tokens=1024),
        )
        expected_prefill = (
            self.cluster.node(0).compute.estimate_incremental_prefill(1024, 128)
        )
        expected_decode = self.cluster.node(0).compute.estimate_amortized_decode(
            gen_tokens=15, ctx_len=1024 + 128, batch_size=32
        )
        self.assertAlmostEqual(cost.t_prefill_ms, expected_prefill.prefill_ms)
        self.assertAlmostEqual(cost.t_decode_ms, expected_decode.total_ms)

    def test_first_output_token_is_produced_by_prefill(self):
        request = replace(self.request(), output_len=1)
        cost = self.router._cost(
            request,
            Action(0, StateMode.LOCAL, src_node=0, hit_tokens=1024),
        )
        self.assertEqual(cost.t_decode_ms, 0.0)

    def test_multimodal_decode_excludes_vision_encoder_weights(self):
        model = get_model("Qwen2-VL-7B-Instruct")
        self.assertEqual(
            model.decoder_weight_bytes(),
            int(model.num_params * model.dtype_bytes),
        )
        self.assertLess(model.decoder_weight_bytes(), model.total_weight_bytes())

    def test_fresh_request_prefills_initial_prefix_and_current_input(self):
        request = self.request(first=True)
        cost = self.router._cost(
            request, Action(0, StateMode.FRESH, hit_tokens=0)
        )
        expected = self.cluster.node(0).compute.estimate_prefill(1024 + 128)
        self.assertAlmostEqual(cost.t_prefill_ms, expected.prefill_ms)

    def test_prefill_batching_only_reduces_queue_service_demand(self):
        request = self.request()
        cost = self.router._cost(
            request,
            Action(0, StateMode.LOCAL, src_node=0, hit_tokens=1024),
        )
        compute = self.cluster.node(0).compute
        standalone = compute.estimate_incremental_prefill(
            request.prefix_tokens,
            request.input_tokens,
        )
        batched = compute.estimate_incremental_prefill_service(
            request.prefix_tokens,
            request.input_tokens,
            batch_size=self.cluster.node(0).prefill_batch_size,
        )
        self.assertAlmostEqual(cost.t_prefill_ms, standalone.prefill_ms)
        self.assertAlmostEqual(
            cost.queue_prefill_service_ms,
            batched.prefill_ms,
        )
        self.assertLessEqual(batched.prefill_ms, standalone.prefill_ms)

    def test_batching_does_not_divide_compute_work(self):
        compute = self.cluster.node(0).compute
        prompt_tokens = 8192
        batched = compute.estimate_prefill_service(prompt_tokens, batch_size=4)
        compute_floor_ms = (
            self.model.prefill_flops(prompt_tokens)
            / compute.hw.effective_compute()
            * 1000.0
        )
        self.assertGreaterEqual(batched.prefill_ms, compute_floor_ms)

    def test_missing_all_historical_kv_requires_full_prefix_recompute(self):
        request = self.request()
        actions = self.router._enumerate(
            request, self.router._prefix_hashes(request)
        )
        self.assertEqual(len(actions), len(self.cluster.node_ids()))
        self.assertTrue(all(a.mode == StateMode.RECOMPUTE for a in actions))
        self.assertTrue(all(a.hit_tokens == 1024 for a in actions))

    def test_recompute_only_covers_suffix_missing_after_local_prefix(self):
        request = self.request()
        blocks = make_blocks(
            self.model,
            request.session_id,
            request.prefix_id,
            request.prefix_tokens,
            owner=0,
        )
        self.cluster.node(0).kv_store.insert(blocks, 0.0)
        for block in blocks:
            self.cluster.kv.register(0, block)

        half = len(blocks) // 2
        self.cluster.node(1).kv_store.insert(blocks[:half], 0.0)
        for block in blocks[:half]:
            self.cluster.kv.register(1, block)

        actions = self.router._enumerate(
            request, self.router._prefix_hashes(request)
        )
        recompute = next(
            action for action in actions
            if action.exec_node == 1 and action.mode == StateMode.RECOMPUTE
        )
        self.assertEqual(recompute.hit_tokens, request.prefix_tokens // 2)

        cost = self.router._cost(request, recompute)
        compute = self.cluster.node(1).compute
        expected = compute.estimate_incremental_prefill(
            request.prefix_tokens // 2,
            request.prefix_tokens // 2,
        )
        full = compute.recompute_time_ms(request.prefix_tokens)
        self.assertAlmostEqual(cost.t_state_ms, expected.prefill_ms)
        self.assertLess(cost.t_state_ms, full)

    def test_incremental_prefill_keeps_cross_attention_work(self):
        compute = self.cluster.node(0).compute
        cached = 512
        residual = 512
        incremental = compute.estimate_incremental_prefill(cached, residual)
        suffix_from_zero = compute.estimate_prefill(residual)
        full = compute.estimate_prefill(cached + residual)
        self.assertGreater(incremental.flops, suffix_from_zero.flops)
        self.assertLess(incremental.flops, full.flops)

    def test_non_block_migration_charges_the_complete_prefix(self):
        request = self.request()
        blocks = make_blocks(
            self.model,
            request.session_id,
            request.prefix_id,
            request.prefix_tokens,
            owner=0,
        )
        self.cluster.node(0).kv_store.insert(blocks, 0.0)
        for block in blocks:
            self.cluster.kv.register(0, block)

        hashes = self.router._prefix_hashes(request)
        action = self.router._migrate_action(
            hashes,
            dst=1,
            located_tokens=request.prefix_tokens,
        )
        decision = self.router._cost(request, action)
        self.router.commit(request, decision, 0.0)

        self.assertFalse(self.router.block_level_kv)
        self.assertEqual(
            self.cluster.kv.stats["migrate_bytes"],
            action.migrate_bytes,
        )

    def test_closed_form_decode_matches_stepwise_model(self):
        compute = self.cluster.node(0).compute
        for ctx_len, gen_tokens, batch_size in (
            (1, 1, 1),
            (1152, 16, 1),
            (32768, 257, 1),
            (4096, 64, 8),
        ):
            expected_ms = 0.0
            last_ms = 0.0
            memory_steps = 0
            for g in range(gen_tokens):
                length = ctx_len + g
                t_comp = (
                    batch_size * self.model.decode_flops_per_token(length)
                    / compute.hw.effective_compute()
                )
                t_mem = (
                    self.model.decoder_weight_bytes()
                    + batch_size * self.model.kv_bytes_per_token() * length
                ) / compute.hw.effective_bandwidth()
                if t_mem >= t_comp:
                    memory_steps += 1
                last_ms = (
                    max(t_comp, t_mem) * 1000.0
                    + compute.hw.fixed_overhead_ms
                )
                expected_ms += last_ms

            result = compute.estimate_decode(gen_tokens, ctx_len, batch_size)
            self.assertAlmostEqual(result.total_ms, expected_ms, places=8)
            self.assertAlmostEqual(result.step_ms, last_ms, places=8)
            expected_bound = (
                "memory" if memory_steps >= gen_tokens / 2 else "compute"
            )
            self.assertEqual(result.bound, expected_bound)


if __name__ == "__main__":
    unittest.main()
