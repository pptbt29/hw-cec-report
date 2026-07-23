import unittest

from sim.compute_simulator import get_hardware
from sim.large_model import get_model
from sim.node import ServingNode


class SLASlackQueueTest(unittest.TestCase):
    def setUp(self):
        self.node = ServingNode(
            0,
            get_model("CodeLlama34B"),
            get_hardware("A800T-A2"),
        )

    def test_waiting_jobs_are_ranked_by_latest_start(self):
        # The first job is already running and therefore cannot be preempted.
        self.node.add_load(prefill_ms=10.0, latest_start_ms=100.0)
        self.node.add_load(prefill_ms=20.0, latest_start_ms=200.0)
        self.node.add_load(prefill_ms=5.0, latest_start_ms=50.0)

        state = self.node.state()
        total, prefill, recompute, decode = state.queue_before(60.0)
        self.assertAlmostEqual(total, 15.0)
        self.assertAlmostEqual(prefill, 15.0)
        self.assertEqual(recompute, 0.0)
        self.assertEqual(decode, 0.0)

        # A loose-deadline request waits behind every existing job.
        self.assertAlmostEqual(state.queue_before(300.0)[0], 35.0)

    def test_non_preemptive_head_then_earliest_deadline(self):
        self.node.add_load(prefill_ms=10.0, latest_start_ms=100.0)
        self.node.add_load(prefill_ms=20.0, latest_start_ms=200.0)
        self.node.add_load(prefill_ms=5.0, latest_start_ms=50.0)

        self.node.advance_to(10.0, 0.0)
        state = self.node.state()
        self.assertEqual(state.running_queue, (5.0, 0.0, 0.0))

        self.node.advance_to(15.0, 10.0)
        state = self.node.state()
        self.assertEqual(state.running_queue, (20.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
