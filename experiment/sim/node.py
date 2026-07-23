"""Serving node and shared state directory.

A ServingNode loads one large-model instance and forms a serving service
(queue + compute simulator + local KV store). GlobalStateDirectory aggregates
per-node snapshots, the network and the KV directory into a (possibly stale)
view consumed by the per-node routers.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .compute_simulator import ComputeSimulator, HardwareSpec
from .kv_cache import GlobalKVDirectory, KVCacheStore
from .large_model import ModelSpec
from .network import NetworkSimulator


@dataclass
class NodeState:
    node_id: int
    estimated_queue_ms: float
    queue_prefill_ms: float
    queue_recompute_ms: float
    queue_decode_ms: float
    kv_used_bytes: float
    kv_capacity_bytes: float
    mem_free_bytes: float
    recent_p99_ttft_ms: float
    sla_slack_scheduling: bool = True
    running_queue: Optional[Tuple[float, float, float]] = None
    waiting_queues: Tuple[Tuple[float, float, float, float], ...] = ()

    def queue_before(
        self,
        latest_start_ms: float,
    ) -> Tuple[float, float, float, float]:
        """Return queued work that executes before a new slack-ranked job.

        The currently running job is non-preemptible.  Waiting jobs use
        earliest latest-start-time first; existing jobs win ties.
        """
        if not self.sla_slack_scheduling:
            return (
                self.estimated_queue_ms,
                self.queue_prefill_ms,
                self.queue_recompute_ms,
                self.queue_decode_ms,
            )
        prefill = recompute = decode = 0.0
        if self.running_queue is not None:
            prefill, recompute, decode = self.running_queue
        for deadline, p_ms, r_ms, d_ms in self.waiting_queues:
            if deadline <= latest_start_ms:
                prefill += p_ms
                recompute += r_ms
                decode += d_ms
        return prefill + recompute + decode, prefill, recompute, decode


@dataclass
class _QueueJob:
    latest_start_ms: float
    sequence: int
    prefill_ms: float
    recompute_ms: float
    decode_ms: float

    def total_ms(self) -> float:
        return self.prefill_ms + self.recompute_ms + self.decode_ms

    def consume(self, elapsed_ms: float) -> float:
        """Consume service proportionally and return unused elapsed time."""
        total = self.total_ms()
        if total <= 0.0:
            return elapsed_ms
        used = min(max(elapsed_ms, 0.0), total)
        ratio = max(total - used, 0.0) / total
        self.prefill_ms *= ratio
        self.recompute_ms *= ratio
        self.decode_ms *= ratio
        return max(elapsed_ms - used, 0.0)


class ServingNode:
    def __init__(
        self,
        node_id: int,
        model: ModelSpec,
        hardware: HardwareSpec,
        kv_capacity_bytes: Optional[float] = None,
        activation_reserve_bytes: float = 64e9,
        prefill_batch_size: int = 4,
        sla_slack_scheduling: bool = True,
    ):
        self.node_id = node_id
        self.model = model
        self.compute = ComputeSimulator(model, hardware)
        total = hardware.total_memory()
        weights = model.total_weight_bytes()
        if kv_capacity_bytes is None:
            kv_capacity_bytes = max(total - weights - activation_reserve_bytes, 0.0)
        self.kv_store = KVCacheStore(node_id, model, kv_capacity_bytes)
        self.mem_total = total
        self.mem_weights = weights
        self.activation_reserve = activation_reserve_bytes
        self.prefill_batch_size = max(int(prefill_batch_size), 1)
        self.sla_slack_scheduling = bool(sla_slack_scheduling)

        self.assigned_load_ms = 0.0
        self._queue_components = {
            "prefill": 0.0,
            "recompute": 0.0,
            "decode": 0.0,
        }
        self._running_job: Optional[_QueueJob] = None
        self._waiting_jobs: List[Tuple[float, int, _QueueJob]] = []
        self._queue_sequence = 0
        self._ttft_samples: List[float] = []
        self.served = 0

    def estimated_queue_ms(self) -> float:
        return max(self.assigned_load_ms, 0.0)

    def mem_free_bytes(self) -> float:
        return self.kv_store.free_bytes()

    def recent_p99_ttft_ms(self) -> float:
        if not self._ttft_samples:
            return 0.0
        s = sorted(self._ttft_samples[-200:])
        k = min(len(s) - 1, int(round(0.99 * (len(s) - 1))))
        return s[k]

    def state(self) -> NodeState:
        running = None
        if self._running_job is not None:
            running = (
                self._running_job.prefill_ms,
                self._running_job.recompute_ms,
                self._running_job.decode_ms,
            )
        waiting = tuple(
            (
                job.latest_start_ms,
                job.prefill_ms,
                job.recompute_ms,
                job.decode_ms,
            )
            for _, _, job in self._waiting_jobs
        )
        return NodeState(
            node_id=self.node_id,
            estimated_queue_ms=self.estimated_queue_ms(),
            queue_prefill_ms=self._queue_components["prefill"],
            queue_recompute_ms=self._queue_components["recompute"],
            queue_decode_ms=self._queue_components["decode"],
            kv_used_bytes=self.kv_store.used_bytes(),
            kv_capacity_bytes=self.kv_store.capacity_bytes,
            mem_free_bytes=self.mem_free_bytes(),
            recent_p99_ttft_ms=self.recent_p99_ttft_ms(),
            sla_slack_scheduling=self.sla_slack_scheduling,
            running_queue=running,
            waiting_queues=waiting,
        )

    def add_load(
        self,
        prefill_ms: float = 0.0,
        recompute_ms: float = 0.0,
        decode_ms: float = 0.0,
        latest_start_ms: float = float("inf"),
    ) -> None:
        job = _QueueJob(
            latest_start_ms=float(latest_start_ms),
            sequence=self._queue_sequence,
            prefill_ms=max(prefill_ms, 0.0),
            recompute_ms=max(recompute_ms, 0.0),
            decode_ms=max(decode_ms, 0.0),
        )
        self._queue_sequence += 1
        if job.total_ms() <= 0.0:
            return
        if self._running_job is None:
            self._running_job = job
        else:
            heapq.heappush(
                self._waiting_jobs,
                (job.latest_start_ms, job.sequence, job),
            )
        self._sync_queue_components()

    def _sync_queue_components(self) -> None:
        components = {"prefill": 0.0, "recompute": 0.0, "decode": 0.0}
        jobs = []
        if self._running_job is not None:
            jobs.append(self._running_job)
        jobs.extend(job for _, _, job in self._waiting_jobs)
        for job in jobs:
            components["prefill"] += job.prefill_ms
            components["recompute"] += job.recompute_ms
            components["decode"] += job.decode_ms
        self._queue_components = components
        self.assigned_load_ms = sum(components.values())

    def advance_to(self, t_now: float, prev_t: float) -> None:
        """Drain the queue by elapsed wall-clock time."""
        elapsed = max(t_now - prev_t, 0.0)
        while elapsed > 0.0 and self._running_job is not None:
            elapsed = self._running_job.consume(elapsed)
            if self._running_job.total_ms() > 1e-12:
                break
            self._running_job = None
            if self._waiting_jobs:
                _, _, self._running_job = heapq.heappop(self._waiting_jobs)
        self._sync_queue_components()

    def record_ttft(self, ttft_ms: float) -> None:
        self._ttft_samples.append(ttft_ms)
        self.served += 1


class GlobalStateDirectory:
    def __init__(
        self,
        nodes: Dict[int, ServingNode],
        kv_directory: GlobalKVDirectory,
        network: NetworkSimulator,
        staleness_ms: float = 0.0,
    ):
        self.nodes = nodes
        self.kv = kv_directory
        self.net = network
        self.staleness_ms = staleness_ms
        self._snapshot: Dict[int, NodeState] = {}
        self._snapshot_t = -1e18
        self.refresh(t_now=0.0, force=True)

    def refresh(self, t_now: float, force: bool = False) -> None:
        """Periodically sync node states into the shared snapshot."""
        if force or (t_now - self._snapshot_t) >= self.staleness_ms:
            self._snapshot = {i: n.state() for i, n in self.nodes.items()}
            self._snapshot_t = t_now

    def snapshot(self) -> Dict[int, NodeState]:
        """Return the last synced view (may lag real state by staleness_ms)."""
        return self._snapshot

    def node(self, node_id: int) -> ServingNode:
        return self.nodes[node_id]

    def node_ids(self) -> List[int]:
        return sorted(self.nodes)


def build_cluster(
    model: ModelSpec,
    hardware: HardwareSpec,
    network: NetworkSimulator,
    num_nodes: int = 3,
    staleness_ms: float = 0.0,
    kv_capacity_bytes: Optional[float] = None,
    activation_reserve_bytes: float = 64e9,
    prefill_batch_size: int = 4,
    sla_slack_scheduling: bool = True,
) -> GlobalStateDirectory:
    nodes = {
        i: ServingNode(
            i, model, hardware,
            kv_capacity_bytes=kv_capacity_bytes,
            activation_reserve_bytes=activation_reserve_bytes,
            prefill_batch_size=prefill_batch_size,
            sla_slack_scheduling=sla_slack_scheduling,
        )
        for i in range(num_nodes)
    }
    kv_dir = GlobalKVDirectory(num_nodes=num_nodes)
    return GlobalStateDirectory(nodes, kv_dir, network, staleness_ms=staleness_ms)


if __name__ == "__main__":
    from .compute_simulator import get_hardware
    from .large_model import get_model
    from .network import NetworkSimulator, default_topology

    net = NetworkSimulator(default_topology())
    cluster = build_cluster(get_model("CodeLlama34B"), get_hardware("A800T-A2"), net)
    for i, st in cluster.snapshot().items():
        print(
            f"node {i}: queue={st.estimated_queue_ms:.1f}ms "
            f"kv_cap={st.kv_capacity_bytes/1e9:.1f}GB free={st.mem_free_bytes/1e9:.1f}GB"
        )
