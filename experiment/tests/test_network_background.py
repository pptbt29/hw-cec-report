import unittest

from sim.network import NetworkSimulator, default_topology


class BackgroundBandwidthReservationTest(unittest.TestCase):
    def test_background_fraction_reserves_only_its_configured_share(self):
        net = NetworkSimulator(default_topology())
        payload = 1e9
        baseline = net.transfer_time_ms(0, 1, payload, contention=True)
        background = net.start_transfer(
            0,
            1,
            payload,
            0.0,
            background_fraction=0.1,
        )
        with_background = net.transfer_time_ms(
            0,
            1,
            payload,
            contention=True,
        )
        baseline_data = baseline - 0.05
        background_data = with_background - 0.05
        self.assertAlmostEqual(
            background_data / baseline_data,
            1.0 / 0.9,
            places=8,
        )
        net.finish_transfer(background, 1000.0)
        restored = net.transfer_time_ms(0, 1, payload, contention=True)
        self.assertAlmostEqual(restored, baseline)


if __name__ == "__main__":
    unittest.main()
