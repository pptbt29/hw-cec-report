"""Lightweight proactive KV placement for the request-level simulator.

The manager is deliberately policy-light: after a request commits its new KV,
it copies a probability- and continuation-weighted fraction of the missing
contiguous prefix to candidate entry nodes.  Transfers run in the background;
only completed blocks become visible to routers through the global directory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .kv_cache import KVBlock
from .large_model import ModelSpec
from .network import Flow
from .node import GlobalStateDirectory


@dataclass
class KVManagerConfig:
    enabled: bool = True
    base_replication_factor: float = 2.0
    max_replication_fraction: float = 0.50
    background_bandwidth_fraction: float = 0.20
    continuation_reference_turns: float = 4.0
    min_window_ms: float = 1.0
    max_window_ms: float = 5000.0


@dataclass
class KVPlacementTask:
    session_id: str
    src: int
    dst: int
    blocks: List[KVBlock]
    num_bytes: int
    start_ms: float
    completion_ms: float
    flow: Flow
    rate_bytes_s: float
    latency_ms: float
    committed_blocks: int = 0


@dataclass
class PlacementResult:
    scheduled_bytes: int = 0
    scheduled_blocks: int = 0
    scheduled_tasks: int = 0
    by_destination: Dict[int, int] = field(default_factory=dict)


class ProactiveKVManager:
    """Request-completion-triggered incremental KV replication."""

    def __init__(
        self,
        model: ModelSpec,
        directory: GlobalStateDirectory,
        config: KVManagerConfig,
        oracle_session_requests: Optional[Dict[str, List]] = None,
        oracle_placement: bool = False,
    ):
        self.model = model
        self.dir = directory
        self.config = config
        self.oracle_session_requests = oracle_session_requests or {}
        self.oracle_placement = oracle_placement
        self._tasks: List[KVPlacementTask] = []
        self._inflight: Set[Tuple[int, str]] = set()
        self._reserved_bytes: Dict[int, int] = {
            node: 0 for node in directory.node_ids()
        }
        self.stats: Dict[str, float] = {
            "scheduled_tasks": 0,
            "completed_tasks": 0,
            "scheduled_blocks": 0,
            "completed_blocks": 0,
            "scheduled_bytes": 0,
            "completed_bytes": 0,
            "skipped_inactive": 0,
            "skipped_no_continuation": 0,
            "skipped_no_resource": 0,
        }

    def _activity(self, request) -> float:
        rows = self.oracle_session_requests.get(request.session_id)
        if rows is not None:
            # The simulation intentionally evaluates the mechanism without
            # continuation-prediction error: the generated trace reveals the
            # exact number of requests still remaining in this session.
            remaining = sum(
                row.turn_index > request.turn_index for row in rows
            )
        else:
            # Fallback retained for isolated component tests and external use.
            expected_turns = max(float(request.expected_session_turns), 0.0)
            remaining = max(
                expected_turns - float(request.turn_index) - 1.0,
                0.0,
            )
        reference = max(self.config.continuation_reference_turns, 1e-9)
        return min(remaining / reference, 1.0)

    def _transition_probability(self, request, dst: int) -> float:
        if not getattr(request, "mobility_active", False):
            return 0.0
        nodes = self.dir.node_ids()
        if len(nodes) <= 1:
            return 0.0
        move_probability = max(
            0.0, min(float(getattr(request, "mobility_ratio", 0.0)), 1.0)
        )
        # ``request.entry_node`` is the current Markov state.  The next
        # request stays there with probability 1-p_move; each other node gets
        # an equal share of p_move.  This matters when execution remains at an
        # old KV owner: the current ingress must then be the highest-priority
        # placement destination rather than being excluded.
        if dst == request.entry_node:
            return 1.0 - move_probability
        return move_probability / (len(nodes) - 1)

    def _oracle_target(
        self,
        request,
        source_node: int,
    ) -> Optional[Tuple[int, float]]:
        """Return the next actual ingress different from the KV source."""
        rows = self.oracle_session_requests.get(request.session_id, [])
        for future in rows:
            if future.turn_index <= request.turn_index:
                continue
            if future.entry_node != source_node:
                return future.entry_node, float(future.arrival_ms)
        return None

    def _background_rate_bytes_s(self, src: int, dst: int) -> float:
        hops = self.dir.net.topology.path(src, dst)
        if not hops:
            return 0.0
        hop_keys = {hop.key() for hop in hops}
        if any(hop_keys.intersection(task.flow.hop_keys) for task in self._tasks):
            return 0.0
        bottleneck = min(h.effective_bandwidth() for h in hops)
        return bottleneck * max(
            0.0, min(self.config.background_bandwidth_fraction, 1.0)
        )

    def _candidate_blocks(
        self,
        request,
        dst: int,
        alpha: float,
        byte_budget: float,
    ) -> List[KVBlock]:
        blocks = self.dir.kv.blocks_for_session(
            request.session_id, request.prefix_id, self.model.name
        )
        if not blocks:
            return []

        ready = 0
        for block in blocks:
            if dst in self.dir.kv.locate(block.block_hash):
                ready += 1
            else:
                break

        missing = blocks[ready:]
        if not missing or (dst, missing[0].block_hash) in self._inflight:
            return []

        target_count = max(1, int(math.ceil(alpha * len(missing))))
        selected: List[KVBlock] = []
        used = 0
        for block in missing[:target_count]:
            if (dst, block.block_hash) in self._inflight:
                break
            if used + block.size_bytes > byte_budget:
                break
            selected.append(block)
            used += block.size_bytes
        return selected

    def schedule_after_request(
        self,
        request,
        source_node: int,
        t_now: float,
    ) -> PlacementResult:
        result = PlacementResult()
        if not self.config.enabled:
            return result
        candidates: List[Tuple[int, float, float]] = []
        if self.oracle_placement:
            target = self._oracle_target(request, source_node)
            if target is None:
                self.stats["skipped_no_continuation"] += 1
                return result
            dst, deadline_ms = target
            window_ms = max(deadline_ms - t_now, 0.0)
            if window_ms <= 0.0:
                self.stats["skipped_no_resource"] += 1
                return result
            candidates.append((dst, 1.0, window_ms))
        else:
            if not getattr(request, "mobility_active", False):
                self.stats["skipped_inactive"] += 1
                return result
            activity = self._activity(request)
            if activity <= 0.0:
                self.stats["skipped_no_continuation"] += 1
                return result
            window_ms = min(
                max(float(getattr(request, "expected_interarrival_ms", 0.0)),
                    self.config.min_window_ms),
                self.config.max_window_ms,
            )
            for dst in self.dir.node_ids():
                probability = self._transition_probability(request, dst)
                if probability <= 0.0 or dst == source_node:
                    continue
                alpha = min(
                    max(
                        self.config.base_replication_factor
                        * probability
                        * activity,
                        0.0,
                    ),
                    self.config.max_replication_fraction,
                )
                candidates.append((dst, alpha, window_ms))

            # Schedule the most likely next ingress first.  Overlapping paths
            # share one bounded background budget, so node-id order must not
            # decide which placement receives that budget.
            candidates.sort(key=lambda item: item[1], reverse=True)

        for dst, alpha, window_ms in candidates:
            if alpha <= 0.0:
                continue

            rate = self._background_rate_bytes_s(source_node, dst)
            free = max(
                self.dir.node(dst).kv_store.free_bytes()
                - self._reserved_bytes[dst],
                0.0,
            )
            latency_ms = sum(
                hop.latency_ms
                for hop in self.dir.net.topology.path(source_node, dst)
            )
            transferable_ms = max(window_ms - latency_ms, 0.0)
            budget = min(rate * transferable_ms / 1000.0, free)
            blocks = self._candidate_blocks(request, dst, alpha, budget)
            if not blocks:
                self.stats["skipped_no_resource"] += 1
                continue

            num_bytes = sum(block.size_bytes for block in blocks)
            if num_bytes <= 0 or rate <= 0.0:
                self.stats["skipped_no_resource"] += 1
                continue
            transfer_ms = latency_ms + num_bytes / rate * 1000.0
            flow = self.dir.net.start_transfer(
                source_node,
                dst,
                num_bytes,
                t_now,
                background_fraction=self.config.background_bandwidth_fraction,
            )
            task = KVPlacementTask(
                session_id=request.session_id,
                src=source_node,
                dst=dst,
                blocks=blocks,
                num_bytes=num_bytes,
                start_ms=t_now,
                completion_ms=t_now + transfer_ms,
                flow=flow,
                rate_bytes_s=rate,
                latency_ms=latency_ms,
            )
            self._tasks.append(task)
            self._reserved_bytes[dst] += num_bytes
            for block in blocks:
                self._inflight.add((dst, block.block_hash))

            result.scheduled_bytes += num_bytes
            result.scheduled_blocks += len(blocks)
            result.scheduled_tasks += 1
            result.by_destination[dst] = num_bytes
            self.stats["scheduled_tasks"] += 1
            self.stats["scheduled_blocks"] += len(blocks)
            self.stats["scheduled_bytes"] += num_bytes
        return result

    def advance(self, t_now: float) -> int:
        """Commit every fully received block by ``t_now``."""
        completed_bytes = 0
        pending: List[KVPlacementTask] = []
        for task in self._tasks:
            data_ms = max(t_now - task.start_ms - task.latency_ms, 0.0)
            received_bytes = min(
                task.num_bytes,
                task.rate_bytes_s * data_ms / 1000.0,
            )
            cumulative = 0
            completed_block_count = 0
            for block in task.blocks:
                cumulative += block.size_bytes
                if cumulative <= received_bytes + 1e-9:
                    completed_block_count += 1
                else:
                    break
            new_blocks = task.blocks[
                task.committed_blocks:completed_block_count
            ]
            store = self.dir.node(task.dst).kv_store
            evicted = store.insert(new_blocks, min(t_now, task.completion_ms))
            for block_hash in evicted:
                self.dir.kv.unregister(task.dst, block_hash)
            committed_blocks = []
            for block in new_blocks:
                if store.contains(block.block_hash):
                    self.dir.kv.register(task.dst, block)
                    committed_blocks.append(block)
                else:
                    self.dir.kv.unregister(task.dst, block.block_hash)
                self._inflight.discard((task.dst, block.block_hash))
            transferred_block_bytes = sum(
                block.size_bytes for block in new_blocks
            )
            self._reserved_bytes[task.dst] = max(
                self._reserved_bytes[task.dst] - transferred_block_bytes,
                0,
            )
            committed_bytes = sum(
                block.size_bytes for block in committed_blocks
            )
            completed_bytes += committed_bytes
            self.stats["completed_blocks"] += len(committed_blocks)
            self.stats["completed_bytes"] += committed_bytes
            task.committed_blocks = completed_block_count
            if task.committed_blocks >= len(task.blocks):
                self.dir.net.finish_transfer(task.flow, task.completion_ms)
                self.stats["completed_tasks"] += 1
            else:
                pending.append(task)
        self._tasks = pending
        return completed_bytes

    def pending_bytes(self) -> int:
        return sum(
            sum(block.size_bytes for block in task.blocks[task.committed_blocks:])
            for task in self._tasks
        )
