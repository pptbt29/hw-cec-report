"""Control-plane metadata accounting for the centralized state collector.

The meter is intentionally independent from request routing and data-plane
simulation.  It counts metadata bytes only; it does not model state age,
in-flight KV placement, or add artificial latency to request execution.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .network import LinkSpec, NetworkTopology


@dataclass(frozen=True)
class AgentObservationSchema:
    """Fixed-size observation consumed by the long-term routing agent."""

    num_nodes: int = 3
    num_links: int = 3
    local_feature_names: Tuple[str, ...] = (
        "entry_node_0",
        "entry_node_1",
        "entry_node_2",
        "input_tokens_norm",
        "context_tokens_norm",
        "request_bytes_norm",
        "sla_ms_norm",
        "output_length_bin_0",
        "output_length_bin_1",
        "output_length_bin_2",
        "output_length_bin_3",
        "output_length_bin_4",
        "output_length_bin_5",
        "output_length_bin_6",
        "output_length_bin_7",
        "continuation_h1",
        "continuation_h2",
        "continuation_h4",
        "continuation_h8",
    )
    phase_feature_names: Tuple[str, ...] = ("mobility_active",)
    node_feature_names: Tuple[str, ...] = (
        "running_prefill_ms_norm",
        "running_recompute_ms_norm",
        "running_decode_ms_norm",
        "waiting_overdue_prefill_ms_norm",
        "waiting_overdue_recompute_ms_norm",
        "waiting_overdue_decode_ms_norm",
        "waiting_slack_0_50_prefill_ms_norm",
        "waiting_slack_0_50_recompute_ms_norm",
        "waiting_slack_0_50_decode_ms_norm",
        "waiting_slack_50_200_prefill_ms_norm",
        "waiting_slack_50_200_recompute_ms_norm",
        "waiting_slack_50_200_decode_ms_norm",
        "waiting_slack_over_200_prefill_ms_norm",
        "waiting_slack_over_200_recompute_ms_norm",
        "waiting_slack_over_200_decode_ms_norm",
        "free_kv_memory_ratio",
        "recent_p99_ttft_ms_norm",
    )
    link_feature_names: Tuple[str, ...] = (
        "effective_available_bandwidth_norm",
        "rtt_ms_norm",
        "link_available",
    )
    kv_feature_names: Tuple[str, ...] = (
        "complete_latest_version",
        "contiguous_prefix_ratio",
        "residual_sync_ms_norm",
        "missing_suffix_recompute_ms_norm",
    )
    action_names: Tuple[str, ...] = ("reuse", "sync", "recompute")

    @property
    def request_features(self) -> int:
        return len(self.local_feature_names)

    @property
    def mobility_phase_features(self) -> int:
        return len(self.phase_feature_names)

    @property
    def node_features_per_node(self) -> int:
        return len(self.node_feature_names)

    @property
    def link_features_per_link(self) -> int:
        return len(self.link_feature_names)

    @property
    def kv_features_per_node(self) -> int:
        return len(self.kv_feature_names)

    @property
    def actions_per_node(self) -> int:
        return len(self.action_names)

    @property
    def local_feature_dim(self) -> int:
        return self.request_features + self.mobility_phase_features

    @property
    def central_feature_dim(self) -> int:
        return (
            self.num_nodes * self.node_features_per_node
            + self.num_links * self.link_features_per_link
            + self.num_nodes * self.kv_features_per_node
        )

    @property
    def observation_dim(self) -> int:
        return self.local_feature_dim + self.central_feature_dim

    @property
    def action_mask_dim(self) -> int:
        return self.num_nodes * self.actions_per_node

    @property
    def central_continuous_feature_dim(self) -> int:
        """Continuous values encoded as float32 on the wire."""
        return (
            self.num_nodes * 17
            + self.num_links * 2
            + self.num_nodes * 3
        )

    @property
    def central_flag_feature_dim(self) -> int:
        """Availability and version flags encoded as uint8 on the wire."""
        return self.num_nodes + self.num_links

    @property
    def central_compact_bytes(self) -> int:
        """Minimum field bytes before identifiers and message framing."""
        return (
            self.central_continuous_feature_dim * 4
            + self.central_flag_feature_dim
        )

    @property
    def central_float32_bytes(self) -> int:
        """Bytes after the Router casts the central state to float32."""
        return self.central_feature_dim * 4

    @staticmethod
    def node_action_mask(
        *,
        has_history: bool,
        complete_latest_version: bool,
        sync_feasible: bool,
        recompute_feasible: bool,
    ) -> Tuple[bool, bool, bool]:
        """Return the ``reuse/sync/recompute`` mask for one node.

        For the first request, the recompute slot represents fresh prefill.
        Multiple nodes may simultaneously hold a complete latest-version
        replica; every such node exposes reuse and suppresses dominated
        sync/recompute alternatives.
        """
        if not has_history:
            return (False, False, recompute_feasible)
        if complete_latest_version:
            return (True, False, False)
        return (False, sync_feasible, recompute_feasible)


@dataclass(frozen=True)
class ControlPlaneConfig:
    """Wire-size and reporting assumptions used by the traffic model.

    Sizes include a conservative 128-byte allowance for framing, RPC headers,
    and transport overhead.  KV updates carry a compressed contiguous block
    range rather than a complete block directory.  The payload layouts are:

    * node report: 68 B state + 24 B IDs/timestamp + 4 B alignment;
    * link report: 9 B state + 20 B IDs/timestamp + 3 B alignment;
    * KV range update: 76 B fixed record + 4 B alignment;
    * snapshot query: 48 B request/session metadata;
    * snapshot response: 48 B metadata + float32 state vector.
    """

    collector_node: int = 1
    node_report_interval_ms: float = 1.0
    link_report_interval_ms: float = 10.0
    framing_overhead_bytes: int = 128
    node_report_payload_bytes: int = 96
    link_report_payload_bytes: int = 32
    kv_range_update_payload_bytes: int = 80
    snapshot_query_payload_bytes: int = 48
    snapshot_response_metadata_bytes: int = 48
    snapshot_response_payload_bytes: int = 336

    @property
    def node_report_wire_bytes(self) -> int:
        return self.framing_overhead_bytes + self.node_report_payload_bytes

    @property
    def link_report_wire_bytes(self) -> int:
        return self.framing_overhead_bytes + self.link_report_payload_bytes

    @property
    def kv_range_update_wire_bytes(self) -> int:
        return (
            self.framing_overhead_bytes
            + self.kv_range_update_payload_bytes
        )

    @property
    def snapshot_query_wire_bytes(self) -> int:
        return (
            self.framing_overhead_bytes
            + self.snapshot_query_payload_bytes
        )

    @property
    def snapshot_response_wire_bytes(self) -> int:
        return (
            self.framing_overhead_bytes
            + self.snapshot_response_payload_bytes
        )


def project_control_plane_scale(
    *,
    num_nodes: int,
    num_links: int,
    request_rate_rps: float,
    config: ControlPlaneConfig | None = None,
) -> Dict[str, float]:
    """Project logical control traffic for a larger fixed-size topology.

    The response contains every node/link summary in float32 plus a fixed
    request/session metadata allowance.  The projection excludes KV range
    events because their rate depends on the placement policy.
    """
    if num_nodes <= 0 or num_links < 0 or request_rate_rps < 0:
        raise ValueError("invalid topology size or request rate")
    cfg = config or ControlPlaneConfig()
    schema = AgentObservationSchema(
        num_nodes=num_nodes,
        num_links=num_links,
    )
    response_payload_bytes = (
        cfg.snapshot_response_metadata_bytes
        + schema.central_float32_bytes
    )
    response_wire_bytes = (
        cfg.framing_overhead_bytes + response_payload_bytes
    )
    pull_wire_bytes = cfg.snapshot_query_wire_bytes + response_wire_bytes
    periodic_generated_mbps = 8.0 * (
        num_nodes
        * cfg.node_report_wire_bytes
        * 1000.0
        / cfg.node_report_interval_ms
        + num_links
        * cfg.link_report_wire_bytes
        * 1000.0
        / cfg.link_report_interval_ms
    ) / 1e6
    return {
        "num_nodes": num_nodes,
        "num_links": num_links,
        "central_feature_dim": schema.central_feature_dim,
        "central_compact_bytes": schema.central_compact_bytes,
        "central_float32_bytes": schema.central_float32_bytes,
        "snapshot_response_wire_bytes": response_wire_bytes,
        "bytes_per_pull": pull_wire_bytes,
        "periodic_report_generated_mbps": periodic_generated_mbps,
        "pull_generated_mbps": (
            8.0 * request_rate_rps * pull_wire_bytes / 1e6
        ),
        "periodic_report_messages_per_second": (
            num_nodes * 1000.0 / cfg.node_report_interval_ms
            + num_links * 1000.0 / cfg.link_report_interval_ms
        ),
    }


class MetadataControlPlaneMeter:
    """Count generated and physically carried metadata traffic."""

    CATEGORIES = (
        "node_reports",
        "link_reports",
        "kv_range_updates",
        "snapshot_queries",
        "snapshot_responses",
    )

    def __init__(
        self,
        topology: NetworkTopology,
        config: ControlPlaneConfig | None = None,
    ):
        self.topology = topology
        self.config = config or ControlPlaneConfig()
        if not 0 <= self.config.collector_node < topology.num_nodes:
            raise ValueError("collector_node is outside the topology")
        self._category: Dict[str, Dict[str, float]] = {
            name: {
                "messages": 0,
                "generated_bytes": 0,
                "link_carried_bytes": 0,
            }
            for name in self.CATEGORIES
        }
        self._by_link: Dict[str, Dict[str, float]] = {
            self._link_label(spec): {
                "bytes": 0,
                "capacity_bps": spec.bandwidth_bps,
            }
            for spec in topology.all_links()
        }
        self._snapshot_pull_network_ms: List[float] = []

    @staticmethod
    def _link_label(spec: LinkSpec) -> str:
        return spec.name or f"{spec.key()[0]}-{spec.key()[1]}"

    def _record(
        self,
        category: str,
        src: int,
        dst: int,
        wire_bytes: int,
        count: int = 1,
    ) -> None:
        if category not in self._category:
            raise KeyError(f"unknown metadata category {category!r}")
        if count <= 0 or wire_bytes <= 0:
            return
        generated = int(wire_bytes) * int(count)
        row = self._category[category]
        row["messages"] += int(count)
        row["generated_bytes"] += generated
        hops = self.topology.path(src, dst)
        row["link_carried_bytes"] += generated * len(hops)
        for hop in hops:
            self._by_link[self._link_label(hop)]["bytes"] += generated

    def _closest_endpoint_to_collector(self, spec: LinkSpec) -> int:
        collector = self.config.collector_node

        def path_latency(node: int) -> float:
            return sum(
                hop.latency_ms
                for hop in self.topology.path(node, collector)
            )

        candidates = (spec.src, spec.dst)
        return min(candidates, key=lambda node: (path_latency(node), node))

    @staticmethod
    def _report_count(duration_ms: float, interval_ms: float) -> int:
        if duration_ms <= 0 or interval_ms <= 0:
            return 0
        return int(math.floor(duration_ms / interval_ms))

    def account_periodic_reports(
        self,
        duration_ms: float,
        node_ids: Iterable[int] | None = None,
    ) -> None:
        """Account node and link telemetry over one experiment window."""
        cfg = self.config
        collector = cfg.collector_node
        node_count = self._report_count(
            duration_ms, cfg.node_report_interval_ms
        )
        for node_id in (
            range(self.topology.num_nodes) if node_ids is None else node_ids
        ):
            self._record(
                "node_reports",
                int(node_id),
                collector,
                cfg.node_report_wire_bytes,
                node_count,
            )

        link_count = self._report_count(
            duration_ms, cfg.link_report_interval_ms
        )
        for spec in self.topology.all_links():
            reporter = self._closest_endpoint_to_collector(spec)
            self._record(
                "link_reports",
                reporter,
                collector,
                cfg.link_report_wire_bytes,
                link_count,
            )

    def record_kv_range_update(
        self,
        source_node: int,
        range_count: int = 1,
    ) -> None:
        """Record committed contiguous KV range updates only."""
        self._record(
            "kv_range_updates",
            source_node,
            self.config.collector_node,
            self.config.kv_range_update_wire_bytes,
            range_count,
        )

    def record_snapshot_pull(self, requester_node: int) -> None:
        """Record one request-triggered query and snapshot response."""
        collector = self.config.collector_node
        self._record(
            "snapshot_queries",
            requester_node,
            collector,
            self.config.snapshot_query_wire_bytes,
        )
        self._record(
            "snapshot_responses",
            collector,
            requester_node,
            self.config.snapshot_response_wire_bytes,
        )
        self._snapshot_pull_network_ms.append(
            self._one_way_network_ms(
                requester_node,
                collector,
                self.config.snapshot_query_wire_bytes,
            )
            + self._one_way_network_ms(
                collector,
                requester_node,
                self.config.snapshot_response_wire_bytes,
            )
        )

    def _one_way_network_ms(
        self,
        src: int,
        dst: int,
        wire_bytes: int,
    ) -> float:
        hops = self.topology.path(src, dst)
        if not hops:
            return 0.0
        latency_ms = sum(hop.latency_ms for hop in hops)
        bottleneck_bytes_per_second = min(
            hop.effective_bandwidth() for hop in hops
        )
        return (
            latency_ms
            + wire_bytes / bottleneck_bytes_per_second * 1000.0
        )

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = int(round(
            percentile / 100.0 * (len(ordered) - 1)
        ))
        return float(ordered[position])

    def summary(self, duration_ms: float, request_count: int) -> Dict:
        seconds = max(float(duration_ms) / 1000.0, 1e-12)
        categories = {}
        for name, raw in self._category.items():
            generated_bps = raw["generated_bytes"] * 8.0 / seconds
            link_carried_bps = raw["link_carried_bytes"] * 8.0 / seconds
            categories[name] = {
                **raw,
                "generated_mbps": generated_bps / 1e6,
                "link_carried_mbps": link_carried_bps / 1e6,
            }

        collection_names = (
            "node_reports",
            "link_reports",
            "kv_range_updates",
        )
        pull_names = ("snapshot_queries", "snapshot_responses")

        def group(names: Tuple[str, ...]) -> Dict[str, float]:
            generated = sum(
                self._category[name]["generated_bytes"] for name in names
            )
            link_carried = sum(
                self._category[name]["link_carried_bytes"] for name in names
            )
            return {
                "generated_bytes": generated,
                "link_carried_bytes": link_carried,
                "generated_mbps": generated * 8.0 / seconds / 1e6,
                "link_carried_mbps": (
                    link_carried * 8.0 / seconds / 1e6
                ),
            }

        collection = group(collection_names)
        pulls = group(pull_names)
        pulls["network_latency_ms"] = {
            "mean": (
                statistics.fmean(self._snapshot_pull_network_ms)
                if self._snapshot_pull_network_ms else 0.0
            ),
            "p50": self._percentile(
                self._snapshot_pull_network_ms, 50
            ),
            "p95": self._percentile(
                self._snapshot_pull_network_ms, 95
            ),
            "p99": self._percentile(
                self._snapshot_pull_network_ms, 99
            ),
        }
        total = group(self.CATEGORIES)
        links = {}
        for label, raw in self._by_link.items():
            throughput_bps = raw["bytes"] * 8.0 / seconds
            links[label] = {
                **raw,
                "throughput_mbps": throughput_bps / 1e6,
                "capacity_fraction": (
                    throughput_bps / raw["capacity_bps"]
                    if raw["capacity_bps"] > 0 else 0.0
                ),
            }
        return {
            "duration_ms": duration_ms,
            "request_count": int(request_count),
            "request_rate_rps": request_count / seconds,
            "collection": collection,
            "per_request_pulls": pulls,
            "total": total,
            "categories": categories,
            "links": links,
        }
