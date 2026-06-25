"""Per-node request router.

Enumerates local/migrate/recompute/fresh actions, prices them with the
compute / network / kv modules, filters by SLA and memory, and selects an
action under one of four policies (nearest, greedy, long-term cost,
long-term cost + low-cost KV management).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .kv_cache import block_hashes_for_len, make_blocks
from .large_model import ModelSpec
from .node import GlobalStateDirectory


class Policy(str, Enum):
    NEAREST = "nearest"
    GREEDY = "greedy"
    LONG_TERM = "long_term"
    LONG_TERM_KV = "long_term_kv"


class StateMode(str, Enum):
    FRESH = "fresh"
    LOCAL = "local"
    MIGRATE = "migrate"
    RECOMPUTE = "recompute"


@dataclass
class Action:
    exec_node: int
    mode: StateMode
    src_node: Optional[int] = None
    hit_tokens: int = 0
    migrate_bytes: int = 0


@dataclass
class ActionCost:
    action: Action
    t_network_ms: float
    t_queue_ms: float
    t_state_ms: float
    t_prefill_ms: float
    ttft_ms: float
    e2e_ms: float
    new_kv_bytes: float
    feasible: bool
    reason: str = ""
    q_value: float = 0.0


class Router:
    def __init__(
        self,
        model: ModelSpec,
        directory: GlobalStateDirectory,
        policy: Policy = Policy.GREEDY,
        gamma: float = 0.9,
        sla_margin_ms: float = 20.0,
        expected_session_turns: int = 4,
        model_version: str = "v1",
    ):
        self.model = model
        self.dir = directory
        self.policy = policy
        self.gamma = gamma
        self.sla_margin_ms = sla_margin_ms
        self.expected_session_turns = expected_session_turns
        self.model_version = model_version
        self.block_level_kv = policy == Policy.LONG_TERM_KV

    # ----- prefix analysis -------------------------------------------------
    def _prefix_hashes(self, request) -> List[str]:
        if request.prefix_tokens <= 0 or request.is_session_first:
            return []
        return block_hashes_for_len(
            self.model.name,
            self.model_version,
            request.prefix_id,
            request.prefix_tokens,
            self.model.kv_block_size,
        )

    def _prefix_stats(self, hashes: List[str], node: int):
        """(located_blocks anywhere, contiguous local blocks at node)."""
        kv = self.dir.kv
        located = 0
        for h in hashes:
            if kv.locate(h):
                located += 1
            else:
                break
        contiguous_local = 0
        for h in hashes:
            if node in kv.locate(h):
                contiguous_local += 1
            else:
                break
        return located, contiguous_local

    def _owner_of(self, hashes: List[str], located: int) -> Optional[int]:
        if located <= 0:
            return None
        nodes = self.dir.kv.locate(hashes[0])
        return min(nodes) if nodes else None

    # ----- action enumeration ---------------------------------------------
    def _enumerate(self, request, hashes: List[str]) -> List[Action]:
        bs = self.model.kv_block_size
        actions: List[Action] = []
        located, _ = self._prefix_stats(hashes, request.entry_node)
        located_tokens = min(located * bs, request.prefix_tokens)

        for node in self.dir.node_ids():
            if located == 0:
                actions.append(Action(node, StateMode.FRESH, hit_tokens=0))
                continue
            _, local_blocks = self._prefix_stats(hashes, node)
            if local_blocks >= located:
                actions.append(
                    Action(node, StateMode.LOCAL, src_node=node, hit_tokens=located_tokens)
                )
            else:
                actions.append(self._migrate_action(hashes[:located], node, located_tokens))
                actions.append(
                    Action(node, StateMode.RECOMPUTE, hit_tokens=located_tokens)
                )
        return actions

    def _migrate_action(self, located_hashes: List[str], dst: int, located_tokens: int) -> Action:
        if self.block_level_kv:
            plan = self.dir.kv.plan_migration(located_hashes, dst, self.dir.net)
            return Action(
                dst, StateMode.MIGRATE, src_node=plan.src,
                hit_tokens=located_tokens, migrate_bytes=plan.bytes_to_move,
            )
        # non-block-level: move the whole located prefix from a single owner
        src = self._owner_of(located_hashes, len(located_hashes))
        bytes_full = self.model.kv_bytes_for_tokens(located_tokens)
        return Action(
            dst, StateMode.MIGRATE, src_node=src,
            hit_tokens=located_tokens, migrate_bytes=int(bytes_full),
        )

    # ----- cost model ------------------------------------------------------
    def _cost(self, request, action: Action) -> ActionCost:
        snap = self.dir.snapshot()
        node_state = snap[action.exec_node]
        node = self.dir.node(action.exec_node)
        compute = node.compute
        net = self.dir.net

        input_bytes = request.input_tokens * self.model.dtype_bytes
        if action.exec_node == request.entry_node:
            t_network = 0.0
        else:
            t_network = net.transfer_time_ms(
                request.entry_node, action.exec_node, input_bytes, contention=True
            )

        t_queue = node_state.estimated_queue_ms

        if action.mode == StateMode.MIGRATE:
            if action.src_node is None:
                t_state = float("inf")
            else:
                t_state = net.transfer_time_ms(
                    action.src_node, action.exec_node, action.migrate_bytes, contention=True
                )
        elif action.mode == StateMode.RECOMPUTE:
            t_state = compute.recompute_time_ms(action.hit_tokens)
        else:  # LOCAL / FRESH
            t_state = 0.0

        new_prefill_tokens = max(request.input_tokens - action.hit_tokens, 0)
        t_prefill = compute.estimate_prefill(new_prefill_tokens).prefill_ms

        ttft = t_network + t_queue + t_state + t_prefill
        decode = compute.estimate_decode(
            gen_tokens=request.output_len, ctx_len=request.input_tokens, batch_size=1
        ).total_ms
        e2e = ttft + decode

        new_kv = self.model.kv_bytes_for_tokens(request.input_tokens + request.output_len)

        feasible, reason = self._feasible(request, action, ttft, new_kv, node_state)
        return ActionCost(
            action=action, t_network_ms=t_network, t_queue_ms=t_queue,
            t_state_ms=t_state, t_prefill_ms=t_prefill, ttft_ms=ttft, e2e_ms=e2e,
            new_kv_bytes=new_kv, feasible=feasible, reason=reason,
        )

    def _feasible(self, request, action, ttft, new_kv, node_state):
        if ttft + self.sla_margin_ms > request.sla_ms:
            return False, "sla"
        if new_kv > node_state.mem_free_bytes:
            return False, "memory"
        return True, ""

    # ----- future value (long-term) ---------------------------------------
    def _future_value(self, request, cost: ActionCost) -> float:
        remaining = max(self.expected_session_turns - request.turn_index - 1, 0)
        if remaining == 0:
            return 0.0
        future_entry = request.entry_node if request.mobility_switched else request.home_node
        if cost.action.exec_node == future_entry:
            return 0.0
        future_state_bytes = self.model.kv_bytes_for_tokens(
            request.input_tokens + request.output_len
        )
        penalty = self.dir.net.transfer_time_ms(
            cost.action.exec_node, future_entry, future_state_bytes, contention=False
        )
        return remaining * penalty

    # ----- selection -------------------------------------------------------
    def route(self, request) -> ActionCost:
        hashes = self._prefix_hashes(request)
        actions = self._enumerate(request, hashes)
        costs = [self._cost(request, a) for a in actions]

        if self.policy == Policy.NEAREST:
            return self._select_nearest(request, costs)

        feasible = [c for c in costs if c.feasible]
        pool = feasible if feasible else costs  # if none feasible, still pick least-bad

        if self.policy == Policy.GREEDY:
            best = min(pool, key=lambda c: c.e2e_ms)
            best.q_value = best.e2e_ms
            return best

        # LONG_TERM / LONG_TERM_KV
        for c in pool:
            c.q_value = c.e2e_ms + self.gamma * self._future_value(request, c)
        return min(pool, key=lambda c: c.q_value)

    def _select_nearest(self, request, costs: List[ActionCost]) -> ActionCost:
        at_entry = [c for c in costs if c.action.exec_node == request.entry_node]
        feasible = [c for c in at_entry if c.feasible]
        pool = feasible if feasible else at_entry
        best = min(pool, key=lambda c: c.e2e_ms)
        best.q_value = best.e2e_ms
        return best

    # ----- commit side effects ---------------------------------------------
    def commit(self, request, decision: ActionCost, t_now: float) -> None:
        exec_node = decision.action.exec_node
        node = self.dir.node(exec_node)
        kv = self.dir.kv
        net = self.dir.net

        if decision.action.mode == StateMode.MIGRATE and decision.action.src_node is not None:
            flow = net.start_transfer(
                decision.action.src_node, exec_node, decision.action.migrate_bytes, t_now
            )
            net.finish_transfer(flow, t_now + decision.t_state_ms)
            located_hashes = self._prefix_hashes(request)
            located, _ = self._prefix_stats(located_hashes, request.entry_node)
            plan = kv.plan_migration(located_hashes[:located], exec_node, net)
            kv.commit_migration(plan, switch_owner=True)
        elif decision.action.mode == StateMode.RECOMPUTE:
            kv.note_recompute()

        # the grown session context now resides on exec_node
        context_tokens = request.input_tokens + request.output_len
        blocks = make_blocks(
            self.model, request.session_id, request.prefix_id, context_tokens,
            owner=exec_node, model_version=self.model_version, t_now=t_now,
        )
        node.kv_store.insert(blocks, t_now)
        for b in blocks:
            kv.register(exec_node, b)
            kv.set_owner(b.block_hash, exec_node)

        node.add_load(decision.t_prefill_ms + (decision.t_state_ms
                      if decision.action.mode == StateMode.RECOMPUTE else 0.0))
        node.record_ttft(decision.ttft_ms)


def simulate_trace(
    policy: Policy,
    requests: List,
    model: ModelSpec,
    hardware,
    network,
    num_nodes: int = 3,
    staleness_ms: float = 0.0,
    gamma: float = 0.9,
    sla_margin_ms: float = 20.0,
    collect_records: bool = False,
) -> Dict:
    """Replay a request trace under a policy and return aggregate metrics.

    When ``collect_records`` is set, also returns per-request records and link
    utilisation, suitable for building a metrics dashboard.
    """
    from .node import build_cluster

    network.reset_stats()
    cluster = build_cluster(model, hardware, network, num_nodes, staleness_ms)
    routers = {
        i: Router(model, cluster, policy, gamma=gamma, sla_margin_ms=sla_margin_ms)
        for i in range(num_nodes)
    }

    reqs = sorted(
        [r for r in requests if r.model_name == model.name], key=lambda r: r.arrival_ms
    )
    prev_t = 0.0
    e2e_list: List[float] = []
    ttft_list: List[float] = []
    sla_viol = 0
    infeasible = 0
    cross_node = 0
    mode_counts = {m.value: 0 for m in StateMode}
    records: List[Dict] = []
    last_t = 0.0

    for r in reqs:
        for n in cluster.nodes.values():
            n.advance_to(r.arrival_ms, prev_t)
        prev_t = r.arrival_ms
        cluster.refresh(r.arrival_ms, force=True)

        router = routers[r.entry_node]
        decision = router.route(r)
        is_infeasible = not decision.feasible
        is_sla = decision.ttft_ms + sla_margin_ms > r.sla_ms
        is_cross = decision.action.exec_node != r.entry_node
        if is_infeasible:
            infeasible += 1
        if is_sla:
            sla_viol += 1
        if is_cross:
            cross_node += 1
        mode_counts[decision.action.mode.value] += 1

        router.commit(r, decision, r.arrival_ms)
        e2e_list.append(decision.e2e_ms)
        ttft_list.append(decision.ttft_ms)
        last_t = r.arrival_ms

        if collect_records:
            records.append({
                "t": round(r.arrival_ms, 2),
                "ttft": round(decision.ttft_ms, 3),
                "e2e": round(decision.e2e_ms, 3),
                "mode": decision.action.mode.value,
                "entry": r.entry_node,
                "exec": decision.action.exec_node,
                "cross": int(is_cross),
                "sla_violation": int(is_sla),
                "infeasible": int(is_infeasible),
                "priority": r.priority,
                "moved": int(r.mobility_switched),
            })

    def pct(v, p):
        if not v:
            return 0.0
        s = sorted(v)
        return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]

    n = max(len(reqs), 1)
    result = {
        "policy": policy.value,
        "model": model.name,
        "num_requests": len(reqs),
        "avg_e2e_ms": sum(e2e_list) / n,
        "p95_e2e_ms": pct(e2e_list, 95),
        "p50_ttft_ms": pct(ttft_list, 50),
        "p95_ttft_ms": pct(ttft_list, 95),
        "p99_ttft_ms": pct(ttft_list, 99),
        "sla_violation_ratio": sla_viol / n,
        "infeasible_ratio": infeasible / n,
        "cross_node_ratio": cross_node / n,
        "local_count": mode_counts["local"],
        "fresh_count": mode_counts["fresh"],
        "migrate_count": mode_counts["migrate"],
        "recompute_count": mode_counts["recompute"],
        "owner_switch_count": cluster.kv.stats["owner_switch_count"],
        "migrate_bytes_mb": cluster.kv.stats["migrate_bytes"] / 1e6,
    }
    if collect_records:
        window = max(last_t, 1.0)
        result["records"] = records
        result["link_utilization"] = network.link_utilization(window)
        result["duration_ms"] = window
    return result


if __name__ == "__main__":
    from .compute_simulator import get_hardware
    from .data_generator import DataGenerator, WorkloadConfig
    from .large_model import get_model
    from .network import NetworkSimulator, default_topology

    model = get_model("CodeLlama34B")
    hw = get_hardware("A800T-A2")
    requests = DataGenerator(WorkloadConfig.default_experiment()).generate()

    print(f"{'policy':<14}{'avg_e2e':>9}{'p99_ttft':>9}{'sla%':>7}"
          f"{'xnode%':>8}{'migr':>6}{'recomp':>7}{'ownsw':>7}{'migMB':>8}")
    for pol in Policy:
        net = NetworkSimulator(default_topology())
        m = simulate_trace(pol, requests, model, hw, net)
        print(
            f"{m['policy']:<14}{m['avg_e2e_ms']:9.1f}{m['p99_ttft_ms']:9.1f}"
            f"{m['sla_violation_ratio']*100:7.1f}{m['cross_node_ratio']*100:8.1f}"
            f"{m['migrate_count']:6d}{m['recompute_count']:7d}"
            f"{m['owner_switch_count']:7d}{m['migrate_bytes_mb']:8.1f}"
        )
