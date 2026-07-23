import unittest

from sim.compute_simulator import get_hardware
from sim.data_generator import Request
from sim.large_model import get_model
from sim.network import NetworkSimulator, default_topology
from sim.router import Policy, simulate_trace


class AdmissionControlTest(unittest.TestCase):
    def request(self, request_id, *, sla_ms):
        model = get_model("CodeLlama34B")
        return Request(
            request_id=request_id, session_id=f"s{request_id}",
            model_name=model.name, model_type="LLM",
            arrival_ms=float(request_id), entry_node=0, priority="high",
            group_name="admission-test", sla_ms=sla_ms,
            prompt_text_tokens=4096, visual_tokens=0, state_tokens=0,
            input_tokens=4096, output_len=16,
            prefix_id=f"{model.name}:s{request_id}", prefix_tokens=0,
            is_session_first=True, turn_index=0, expected_session_turns=1.0,
        )

    def test_intrinsically_infeasible_request_is_rejected_and_excluded(self):
        model = get_model("CodeLlama34B")
        result = simulate_trace(
            Policy.GREEDY,
            [self.request(0, sla_ms=1.0), self.request(1, sla_ms=1e9)],
            model, get_hardware("A800T-A2"),
            NetworkSimulator(default_topology()), collect_records=True,
        )
        self.assertEqual(result["offered_requests"], 2)
        self.assertEqual(result["accepted_requests"], 1)
        self.assertEqual(result["num_requests"], 1)
        self.assertEqual(result["admission_rejected_count"], 1)
        self.assertAlmostEqual(result["admission_rejected_ratio"], 0.5)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(len(result["rejection_records"]), 1)
        self.assertEqual(result["sla_violation_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
