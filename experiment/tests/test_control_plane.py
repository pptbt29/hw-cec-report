import unittest

from sim.control_plane import (
    AgentObservationSchema,
    ControlPlaneConfig,
    MetadataControlPlaneMeter,
    project_control_plane_scale,
)
from sim.network import default_topology


class AgentObservationSchemaTest(unittest.TestCase):
    def test_dimensions_are_fixed(self):
        schema = AgentObservationSchema()
        config = ControlPlaneConfig()
        self.assertEqual(schema.request_features, 19)
        self.assertEqual(schema.mobility_phase_features, 1)
        self.assertEqual(schema.local_feature_dim, 20)
        self.assertEqual(schema.node_features_per_node, 17)
        self.assertEqual(schema.central_feature_dim, 72)
        self.assertEqual(schema.observation_dim, 92)
        self.assertEqual(schema.action_mask_dim, 9)
        self.assertEqual(len(set(schema.local_feature_names)), 19)
        self.assertEqual(schema.central_continuous_feature_dim, 66)
        self.assertEqual(schema.central_flag_feature_dim, 6)
        self.assertEqual(schema.central_compact_bytes, 270)
        self.assertEqual(schema.central_float32_bytes, 288)
        self.assertGreaterEqual(
            config.node_report_payload_bytes,
            schema.node_features_per_node * 4 + 24,
        )
        self.assertGreaterEqual(
            config.link_report_payload_bytes,
            schema.link_features_per_link * 4 + 16,
        )
        self.assertEqual(
            config.snapshot_response_payload_bytes,
            schema.central_feature_dim * 4
            + config.snapshot_response_metadata_bytes,
        )

    def test_full_mesh_scale_projection_matches_three_node_profile(self):
        row = project_control_plane_scale(
            num_nodes=3,
            num_links=3,
            request_rate_rps=28.083333333333332,
        )
        self.assertEqual(row["central_feature_dim"], 72)
        self.assertEqual(row["central_compact_bytes"], 270)
        self.assertEqual(row["snapshot_response_wire_bytes"], 464)
        self.assertEqual(row["bytes_per_pull"], 640)
        self.assertAlmostEqual(
            row["periodic_report_generated_mbps"],
            5.760,
        )
        self.assertAlmostEqual(row["pull_generated_mbps"], 0.1437866667)

    def test_full_mesh_projection_grows_with_node_count(self):
        small = project_control_plane_scale(
            num_nodes=3,
            num_links=3,
            request_rate_rps=100,
        )
        large = project_control_plane_scale(
            num_nodes=10,
            num_links=45,
            request_rate_rps=100,
        )
        self.assertGreater(
            large["periodic_report_generated_mbps"],
            small["periodic_report_generated_mbps"],
        )
        self.assertGreater(
            large["snapshot_response_wire_bytes"],
            small["snapshot_response_wire_bytes"],
        )

    def test_latest_complete_replica_exposes_reuse_only(self):
        self.assertEqual(
            AgentObservationSchema.node_action_mask(
                has_history=True,
                complete_latest_version=True,
                sync_feasible=True,
                recompute_feasible=True,
            ),
            (True, False, False),
        )

    def test_incomplete_replica_cannot_reuse(self):
        self.assertEqual(
            AgentObservationSchema.node_action_mask(
                has_history=True,
                complete_latest_version=False,
                sync_feasible=True,
                recompute_feasible=True,
            ),
            (False, True, True),
        )

    def test_first_request_uses_recompute_slot_as_fresh_prefill(self):
        self.assertEqual(
            AgentObservationSchema.node_action_mask(
                has_history=False,
                complete_latest_version=False,
                sync_feasible=False,
                recompute_feasible=True,
            ),
            (False, False, True),
        )


class MetadataControlPlaneMeterTest(unittest.TestCase):
    def test_local_pull_generates_messages_without_network_bytes(self):
        meter = MetadataControlPlaneMeter(default_topology())
        meter.record_snapshot_pull(1)
        summary = meter.summary(1000.0, 1)

        self.assertEqual(
            summary["per_request_pulls"]["generated_bytes"],
            176 + 464,
        )
        self.assertEqual(
            summary["per_request_pulls"]["link_carried_bytes"],
            0,
        )

    def test_remote_pull_is_accounted_on_the_correct_link(self):
        meter = MetadataControlPlaneMeter(default_topology())
        meter.record_snapshot_pull(0)
        summary = meter.summary(1000.0, 1)

        wire_bytes = 176 + 464
        self.assertEqual(
            summary["per_request_pulls"]["link_carried_bytes"],
            wire_bytes,
        )
        self.assertEqual(
            summary["links"]["A-B-100G"]["bytes"],
            wire_bytes,
        )
        self.assertEqual(summary["links"]["B-C-100G"]["bytes"], 0)
        self.assertEqual(summary["links"]["A-C-25G"]["bytes"], 0)
        expected_ms = (
            2 * 0.05
            + (176 + 464) / (100e9 / 8.0 * 0.9) * 1000.0
        )
        self.assertAlmostEqual(
            summary["per_request_pulls"]["network_latency_ms"]["mean"],
            expected_ms,
        )

    def test_periodic_reports_have_exact_counts(self):
        config = ControlPlaneConfig(
            node_report_interval_ms=10.0,
            link_report_interval_ms=20.0,
        )
        meter = MetadataControlPlaneMeter(default_topology(), config)
        meter.account_periodic_reports(100.0)
        summary = meter.summary(100.0, 0)

        self.assertEqual(
            summary["categories"]["node_reports"]["messages"],
            3 * 10,
        )
        self.assertEqual(
            summary["categories"]["link_reports"]["messages"],
            3 * 5,
        )
        # Collector-local reports are generated but do not consume links.
        self.assertLess(
            summary["collection"]["link_carried_bytes"],
            summary["collection"]["generated_bytes"],
        )

    def test_only_committed_kv_ranges_are_counted(self):
        meter = MetadataControlPlaneMeter(default_topology())
        meter.record_kv_range_update(2, range_count=3)
        summary = meter.summary(1000.0, 0)

        self.assertEqual(
            summary["categories"]["kv_range_updates"]["messages"],
            3,
        )
        self.assertEqual(
            summary["categories"]["kv_range_updates"]["generated_bytes"],
            3 * 208,
        )
        self.assertEqual(
            summary["links"]["B-C-100G"]["bytes"],
            3 * 208,
        )


if __name__ == "__main__":
    unittest.main()
