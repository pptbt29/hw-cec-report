"""Block-level KV cache management.

Organises KV cache into fixed-size blocks with prefix-chained hashes, a
capacity-bounded per-node store with LRU eviction, and a global directory that
tracks block replicas and prices local / migrate / recompute actions.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .large_model import ModelSpec
from .network import NetworkSimulator


def _hash(*parts: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def block_hashes(
    model_name: str,
    model_version: str,
    token_ids: List[int],
    block_size: int,
) -> List[str]:
    """Prefix-chained block hashes for an explicit token sequence."""
    hashes: List[str] = []
    prev = ""
    for i in range(0, len(token_ids), block_size):
        chunk = token_ids[i : i + block_size]
        chunk_repr = ",".join(map(str, chunk))
        if i == 0:
            prev = _hash(model_name, model_version, chunk_repr)
        else:
            prev = _hash(prev, chunk_repr)
        hashes.append(prev)
    return hashes


def block_hashes_for_len(
    model_name: str,
    model_version: str,
    prefix_id: str,
    num_tokens: int,
    block_size: int,
) -> List[str]:
    """Prefix-chained block hashes derived from a (prefix_id, length).

    The simulator has no real token ids, so blocks are identified by
    ``prefix_id`` plus block index; the chaining keeps prefix sensitivity so
    two requests sharing a ``prefix_id`` share their leading blocks.
    """
    n_blocks = (num_tokens + block_size - 1) // block_size
    hashes: List[str] = []
    prev = ""
    for i in range(n_blocks):
        if i == 0:
            prev = _hash(model_name, model_version, prefix_id, "blk0")
        else:
            prev = _hash(prev, f"blk{i}")
        hashes.append(prev)
    return hashes


@dataclass
class KVBlock:
    block_hash: str
    session_id: str
    prefix_id: str
    model_name: str
    model_version: str
    block_index: int
    num_tokens: int
    size_bytes: int
    owner: int
    replicas: Set[int] = field(default_factory=set)
    version: int = 1
    last_access_ms: float = 0.0
    hit_count: int = 0


def make_blocks(
    model: ModelSpec,
    session_id: str,
    prefix_id: str,
    num_tokens: int,
    owner: int,
    model_version: str = "v1",
    t_now: float = 0.0,
) -> List[KVBlock]:
    bs = model.kv_block_size
    hashes = block_hashes_for_len(model.name, model_version, prefix_id, num_tokens, bs)
    per_block_bytes = bs * model.kv_bytes_per_token()
    blocks: List[KVBlock] = []
    remaining = num_tokens
    for i, h in enumerate(hashes):
        tok = min(bs, remaining)
        remaining -= tok
        blocks.append(
            KVBlock(
                block_hash=h,
                session_id=session_id,
                prefix_id=prefix_id,
                model_name=model.name,
                model_version=model_version,
                block_index=i,
                num_tokens=tok,
                size_bytes=int(tok / bs * per_block_bytes) if bs else 0,
                owner=owner,
                replicas={owner},
                last_access_ms=t_now,
            )
        )
    return blocks


class KVCacheStore:
    def __init__(self, node_id: int, model: ModelSpec, capacity_bytes: float):
        self.node_id = node_id
        self.model = model
        self.capacity_bytes = float(capacity_bytes)
        self._blocks: "OrderedDict[str, KVBlock]" = OrderedDict()
        self._used = 0
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "inserts": 0}

    def used_bytes(self) -> int:
        return self._used

    def free_bytes(self) -> float:
        return self.capacity_bytes - self._used

    def contains(self, block_hash: str) -> bool:
        return block_hash in self._blocks

    def touch(self, block_hash: str, t_now: float) -> None:
        blk = self._blocks.get(block_hash)
        if blk is not None:
            blk.last_access_ms = t_now
            blk.hit_count += 1
            self._blocks.move_to_end(block_hash)

    def match_prefix(self, hashes: List[str]) -> int:
        """Number of leading blocks present locally (longest prefix)."""
        count = 0
        for h in hashes:
            if h in self._blocks:
                count += 1
            else:
                break
        if count > 0:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1
        return count

    def evict(self, need_bytes: float, t_now: float, pinned: Optional[Set[str]] = None) -> List[str]:
        pinned = pinned or set()
        evicted: List[str] = []
        # prefer blocks that already have replicas elsewhere, then LRU order
        for h in list(self._blocks.keys()):
            if self.free_bytes() >= need_bytes:
                break
            if h in pinned:
                continue
            blk = self._blocks[h]
            del self._blocks[h]
            self._used -= blk.size_bytes
            blk.replicas.discard(self.node_id)
            evicted.append(h)
            self.stats["evictions"] += 1
        return evicted

    def insert(self, blocks: List[KVBlock], t_now: float, pinned: Optional[Set[str]] = None) -> List[str]:
        evicted: List[str] = []
        for blk in blocks:
            if blk.block_hash in self._blocks:
                self.touch(blk.block_hash, t_now)
                continue
            if self.free_bytes() < blk.size_bytes:
                evicted += self.evict(blk.size_bytes, t_now, pinned)
            self._blocks[blk.block_hash] = blk
            self._blocks.move_to_end(blk.block_hash)
            self._used += blk.size_bytes
            blk.replicas.add(self.node_id)
            blk.last_access_ms = t_now
            self.stats["inserts"] += 1
        return evicted


@dataclass
class MigrationPlan:
    dst: int
    src: Optional[int]
    missing_hashes: List[str]
    bytes_to_move: int
    transfer_ms: float

    @property
    def is_local(self) -> bool:
        return not self.missing_hashes


class GlobalKVDirectory:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self._locations: Dict[str, Set[int]] = {}
        self._meta: Dict[str, KVBlock] = {}
        self.stats = {
            "migrate_bytes": 0,
            "recompute_count": 0,
            "owner_switch_count": 0,
            "prefix_lookups": 0,
            "prefix_local_hit_blocks": 0,
            "prefix_remote_hit_blocks": 0,
        }

    def register(self, node: int, block: KVBlock) -> None:
        self._locations.setdefault(block.block_hash, set()).add(node)
        self._meta[block.block_hash] = block

    def unregister(self, node: int, block_hash: str) -> None:
        nodes = self._locations.get(block_hash)
        if nodes:
            nodes.discard(node)
            if not nodes:
                self._locations.pop(block_hash, None)
                self._meta.pop(block_hash, None)

    def locate(self, block_hash: str) -> Set[int]:
        return set(self._locations.get(block_hash, set()))

    def longest_prefix(self, hashes: List[str], node: int) -> Tuple[int, int]:
        """Return (local_hit_blocks, remote_hit_blocks) for a prefix list."""
        local_hit = 0
        remote_hit = 0
        for h in hashes:
            nodes = self._locations.get(h)
            if not nodes:
                break
            if node in nodes:
                local_hit += 1
            else:
                remote_hit += 1
        self.stats["prefix_lookups"] += 1
        self.stats["prefix_local_hit_blocks"] += local_hit
        self.stats["prefix_remote_hit_blocks"] += remote_hit
        return local_hit, remote_hit

    def set_owner(self, block_hash: str, node: int) -> None:
        blk = self._meta.get(block_hash)
        if blk is None:
            return
        if blk.owner != node:
            self.stats["owner_switch_count"] += 1
            blk.owner = node
        self._locations.setdefault(block_hash, set()).add(node)
        blk.replicas.add(node)

    def plan_migration(
        self, hashes: List[str], dst: int, net: NetworkSimulator
    ) -> MigrationPlan:
        """Pick the cheapest source for blocks missing at ``dst``."""
        missing = [h for h in hashes if dst not in self._locations.get(h, set())]
        if not missing:
            return MigrationPlan(dst, None, [], 0, 0.0)

        bytes_by_src: Dict[int, int] = {}
        unreachable = 0
        for h in missing:
            srcs = self._locations.get(h, set())
            if not srcs:
                unreachable += 1
                continue
            blk = self._meta[h]
            # tentatively attribute the block to each candidate source
            for s in srcs:
                bytes_by_src[s] = bytes_by_src.get(s, 0) + blk.size_bytes

        if not bytes_by_src:
            return MigrationPlan(dst, None, missing, 0, float("inf"))

        best_src = None
        best_ms = float("inf")
        total_bytes = sum(self._meta[h].size_bytes for h in missing if h in self._meta)
        for src in bytes_by_src:
            ms = net.transfer_time_ms(src, dst, total_bytes, contention=True)
            if ms < best_ms:
                best_ms = ms
                best_src = src
        return MigrationPlan(dst, best_src, missing, int(total_bytes), best_ms)

    def commit_migration(self, plan: MigrationPlan, switch_owner: bool = True) -> None:
        for h in plan.missing_hashes:
            self._locations.setdefault(h, set()).add(plan.dst)
            blk = self._meta.get(h)
            if blk is not None:
                blk.replicas.add(plan.dst)
        self.stats["migrate_bytes"] += plan.bytes_to_move
        if switch_owner:
            for h in plan.missing_hashes:
                self.set_owner(h, plan.dst)

    def note_recompute(self) -> None:
        self.stats["recompute_count"] += 1


if __name__ == "__main__":
    from .large_model import get_model
    from .network import NetworkSimulator, default_topology

    model = get_model("CodeLlama34B")
    net = NetworkSimulator(default_topology())
    directory = GlobalKVDirectory(num_nodes=3)

    # session with 2048-token context, owned by node 0 (A)
    blocks = make_blocks(model, session_id="s1", prefix_id="CodeLlama34B:s1",
                         num_tokens=2048, owner=0)
    store_a = KVCacheStore(0, model, capacity_bytes=20e9)
    store_a.insert(blocks, t_now=0.0)
    for b in blocks:
        directory.register(0, b)

    hashes = [b.block_hash for b in blocks]
    print(f"context blocks={len(blocks)} bytes={sum(b.size_bytes for b in blocks)/1e6:.1f} MB")

    # user moves: request now enters node 2 (C, via 25G). Plan migration A->C.
    plan_c = directory.plan_migration(hashes, dst=2, net=net)
    print(f"migrate A->C: missing={len(plan_c.missing_hashes)} "
          f"bytes={plan_c.bytes_to_move/1e6:.1f}MB src={plan_c.src} "
          f"transfer={plan_c.transfer_ms:.2f}ms")

    # vs migrate to node 1 (B, via 100G)
    plan_b = directory.plan_migration(hashes, dst=1, net=net)
    print(f"migrate A->B: bytes={plan_b.bytes_to_move/1e6:.1f}MB "
          f"transfer={plan_b.transfer_ms:.2f}ms")

    local_hit, remote_hit = directory.longest_prefix(hashes, node=0)
    print(f"prefix at node A: local_hit={local_hit} remote_hit={remote_hit}")
