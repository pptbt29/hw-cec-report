"""Network (edge) simulator.

Models node-to-node links as shared, contendable resources: bandwidth,
latency, multi-hop paths, fair-share contention and per-link utilisation.
Used to price KV migration / request forwarding / multimodal input transfer
and to report 100G vs 25G link utilisation.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LinkSpec:
    src: int
    dst: int
    bandwidth_bps: float
    latency_ms: float = 0.1
    name: str = ""
    link_efficiency: float = 0.9

    def effective_bandwidth(self) -> float:
        """Achievable bytes/s after protocol/RDMA overhead."""
        return self.bandwidth_bps / 8.0 * self.link_efficiency

    def key(self) -> Tuple[int, int]:
        return (min(self.src, self.dst), max(self.src, self.dst))


@dataclass
class Flow:
    flow_id: int
    hop_keys: List[Tuple[int, int]]
    bottleneck_key: Tuple[int, int]
    num_bytes: float
    start_ms: float


class _LinkState:
    def __init__(self, spec: LinkSpec):
        self.spec = spec
        self.active_flows = 0
        self.peak_concurrency = 0
        self.total_bytes = 0.0
        self.total_busy_ms = 0.0


class NetworkTopology:
    def __init__(self, num_nodes: int, links: List[LinkSpec]):
        self.num_nodes = num_nodes
        self._links: Dict[Tuple[int, int], LinkSpec] = {}
        self._adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        for spec in links:
            k = spec.key()
            self._links[k] = spec
            self._adj[spec.src].append(spec.dst)
            self._adj[spec.dst].append(spec.src)

    def has_link(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self._links

    def link(self, a: int, b: int) -> LinkSpec:
        k = (min(a, b), max(a, b))
        if k not in self._links:
            raise KeyError(f"no direct link between {a} and {b}")
        return self._links[k]

    def all_links(self) -> List[LinkSpec]:
        return list(self._links.values())

    def path(self, src: int, dst: int) -> List[LinkSpec]:
        """Use a declared direct link, otherwise find a multi-hop path."""
        if src == dst:
            return []
        if self.has_link(src, dst):
            return [self.link(src, dst)]
        dist = {src: 0.0}
        prev: Dict[int, Tuple[int, LinkSpec]] = {}
        pq: List[Tuple[float, int]] = [(0.0, src)]
        visited = set()
        while pq:
            d, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == dst:
                break
            for nb in self._adj[node]:
                spec = self.link(node, nb)
                nd = d + spec.latency_ms
                if nd < dist.get(nb, float("inf")):
                    dist[nb] = nd
                    prev[nb] = (node, spec)
                    heapq.heappush(pq, (nd, nb))
        if dst not in prev and dst != src:
            raise KeyError(f"no path between {src} and {dst}")
        hops: List[LinkSpec] = []
        cur = dst
        while cur != src:
            node, spec = prev[cur]
            hops.append(spec)
            cur = node
        hops.reverse()
        return hops


class NetworkSimulator:
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self._state: Dict[Tuple[int, int], _LinkState] = {
            spec.key(): _LinkState(spec) for spec in topology.all_links()
        }
        self._flow_ids = itertools.count()

    def transfer_time_ms(
        self, src: int, dst: int, num_bytes: float, contention: bool = True
    ) -> float:
        """Predicted transfer time with no side effects (for cost estimation)."""
        if num_bytes <= 0 or src == dst:
            return 0.0
        hops = self.topology.path(src, dst)
        latency = sum(h.latency_ms for h in hops)
        min_bw = float("inf")
        for h in hops:
            bw = h.effective_bandwidth()
            if contention:
                flows = self._state[h.key()].active_flows
                bw = bw / max(flows + 1, 1)
            min_bw = min(min_bw, bw)
        if min_bw == float("inf") or min_bw <= 0:
            return latency
        return latency + num_bytes / min_bw * 1000.0

    def start_transfer(self, src: int, dst: int, num_bytes: float, t_now: float) -> Flow:
        """Begin a transfer along the shortest path, updating link occupancy."""
        hops = self.topology.path(src, dst)
        hop_keys = [h.key() for h in hops]
        bottleneck = min(hops, key=lambda h: h.effective_bandwidth()) if hops else None
        bottleneck_key = bottleneck.key() if bottleneck else (src, dst)
        for k in hop_keys:
            st = self._state[k]
            st.active_flows += 1
            st.peak_concurrency = max(st.peak_concurrency, st.active_flows)
        return Flow(next(self._flow_ids), hop_keys, bottleneck_key, num_bytes, t_now)

    def finish_transfer(self, flow: Flow, t_now: Optional[float] = None) -> None:
        for k in flow.hop_keys:
            st = self._state.get(k)
            if st is None:
                continue
            st.active_flows = max(st.active_flows - 1, 0)
        bottleneck = self._state.get(flow.bottleneck_key)
        if bottleneck is None:
            return
        bottleneck.total_bytes += flow.num_bytes
        if t_now is not None:
            bottleneck.total_busy_ms += max(t_now - flow.start_ms, 0.0)

    def link_utilization(self, window_ms: float) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        window_ms = max(window_ms, 1e-6)
        for k, st in self._state.items():
            label = st.spec.name or f"{k[0]}-{k[1]}"
            out[label] = {
                "bandwidth_gbps": st.spec.bandwidth_bps / 1e9,
                "total_bytes": st.total_bytes,
                "busy_ms": st.total_busy_ms,
                "utilization": min(st.total_busy_ms / window_ms, 1.0),
                "throughput_bps": st.total_bytes * 8.0 / (window_ms / 1000.0),
                "peak_concurrency": st.peak_concurrency,
            }
        return out

    def reset_stats(self) -> None:
        for st in self._state.values():
            st.active_flows = 0
            st.peak_concurrency = 0
            st.total_bytes = 0.0
            st.total_busy_ms = 0.0


def default_topology(num_nodes: int = 3) -> NetworkTopology:
    """Three A800T-A2 nodes: two 100G direct links plus one 25G RDMA link."""
    links = [
        LinkSpec(0, 1, 100e9, latency_ms=0.05, name="A-B-100G"),
        LinkSpec(1, 2, 100e9, latency_ms=0.05, name="B-C-100G"),
        LinkSpec(0, 2, 25e9, latency_ms=0.2, name="A-C-25G"),
    ]
    return NetworkTopology(num_nodes=num_nodes, links=links)


if __name__ == "__main__":
    net = NetworkSimulator(default_topology())
    payload = 200 * 1024 * 1024  # 200 MB KV blob
    print("== single-flow transfer time (200 MB) ==")
    for (a, b) in [(0, 1), (0, 2), (1, 2)]:
        ms = net.transfer_time_ms(a, b, payload, contention=False)
        link = net.topology.link(a, b)
        print(f"  {link.name:<10} {a}->{b}: {ms:7.2f} ms")

    print("\n== contention on A-C-25G ==")
    f1 = net.start_transfer(0, 2, payload, t_now=0.0)
    f2 = net.start_transfer(0, 2, payload, t_now=0.0)
    shared = net.transfer_time_ms(0, 2, payload, contention=True)
    print(f"  with 2 active flows, predicted: {shared:7.2f} ms")
    net.finish_transfer(f1, t_now=shared)
    net.finish_transfer(f2, t_now=shared)
    print("  utilization:", {k: round(v["utilization"], 3)
                              for k, v in net.link_utilization(shared).items()})
