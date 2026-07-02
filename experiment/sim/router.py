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
    t_queue_prefill_ms: float
    t_queue_recompute_ms: float
    t_queue_decode_ms: float
    t_state_ms: float
    t_prefill_ms: float
    t_decode_ms: float
    t_return_ms: float
    ttft_ms: float
    e2e_ms: float
    new_kv_bytes: float
    feasible: bool
    reason: str = ""
    q_value: float = 0.0
    future_cost_ms: float = 0.0
    selection_reason: str = ""


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
        token_id_bytes: int = 4,
        request_overhead_bytes: int = 4096,
        response_overhead_bytes: int = 4096,
        visual_bytes_per_token: int = 0,
    ):
        self.model = model
        self.dir = directory
        self.policy = policy
        self.gamma = gamma
        self.sla_margin_ms = sla_margin_ms
        self.expected_session_turns = expected_session_turns
        self.model_version = model_version
        self.token_id_bytes = token_id_bytes
        self.request_overhead_bytes = request_overhead_bytes
        self.response_overhead_bytes = response_overhead_bytes
        self.visual_bytes_per_token = visual_bytes_per_token
        self.block_level_kv = policy == Policy.LONG_TERM_KV

    def _request_payload_bytes(self, request) -> int:
        token_bytes = request.input_tokens * self.token_id_bytes
        visual_bytes = request.visual_tokens * self.visual_bytes_per_token
        return int(token_bytes + visual_bytes + self.request_overhead_bytes)

    def _response_payload_bytes(self, output_tokens: int) -> int:
        return int(output_tokens * self.token_id_bytes + self.response_overhead_bytes)

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

        input_bytes = self._request_payload_bytes(request)
        if action.exec_node == request.entry_node:
            t_network = 0.0
            t_first_token_return = 0.0
            t_return = 0.0
        else:
            t_network = net.transfer_time_ms(
                request.entry_node, action.exec_node, input_bytes, contention=True
            )
            t_first_token_return = net.transfer_time_ms(
                action.exec_node,
                request.entry_node,
                self._response_payload_bytes(1),
                contention=True,
            )
            t_return = net.transfer_time_ms(
                action.exec_node,
                request.entry_node,
                self._response_payload_bytes(request.output_len),
                contention=True,
            )

        t_queue = node_state.estimated_queue_ms
        t_queue_prefill = node_state.queue_prefill_ms
        t_queue_recompute = node_state.queue_recompute_ms
        t_queue_decode = node_state.queue_decode_ms

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

        ttft = t_network + t_queue + t_state + t_prefill + t_first_token_return
        decode = compute.estimate_decode(
            gen_tokens=request.output_len, ctx_len=request.input_tokens, batch_size=1
        ).total_ms
        e2e = ttft + decode + max(t_return - t_first_token_return, 0.0)

        new_kv = self.model.kv_bytes_for_tokens(request.input_tokens + request.output_len)

        feasible, reason = self._feasible(request, action, ttft, new_kv, node_state)
        return ActionCost(
            action=action, t_network_ms=t_network, t_queue_ms=t_queue,
            t_queue_prefill_ms=t_queue_prefill,
            t_queue_recompute_ms=t_queue_recompute,
            t_queue_decode_ms=t_queue_decode,
            t_state_ms=t_state, t_prefill_ms=t_prefill, t_decode_ms=decode,
            t_return_ms=t_return,
            ttft_ms=ttft, e2e_ms=e2e,
            new_kv_bytes=new_kv, feasible=feasible, reason=reason,
        )

    def _feasible(self, request, action, ttft, new_kv, node_state):
        blockers = []
        if ttft + self.sla_margin_ms > request.sla_ms:
            blockers.append("sla")
        if new_kv > node_state.mem_free_bytes:
            blockers.append("memory")
        return not blockers, "+".join(blockers)

    def _annotate_selection(
        self,
        request,
        selected: ActionCost,
        costs: List[ActionCost],
    ) -> ActionCost:
        if selected.action.mode != StateMode.MIGRATE:
            return selected

        at_entry = [
            cost for cost in costs
            if cost.action.exec_node == request.entry_node
        ]
        if not any(cost.feasible for cost in at_entry):
            sla_blocks_all = all("sla" in cost.reason.split("+") for cost in at_entry)
            memory_blocks_all = all(
                "memory" in cost.reason.split("+") for cost in at_entry
            )
            if sla_blocks_all and memory_blocks_all:
                selected.selection_reason = "entry_sla_and_memory"
            elif sla_blocks_all:
                selected.selection_reason = "entry_sla"
            elif memory_blocks_all:
                selected.selection_reason = "entry_memory"
            else:
                selected.selection_reason = "entry_mixed_constraints"
            return selected

        feasible = [cost for cost in costs if cost.feasible]
        pool = feasible if feasible else costs
        immediate_best = min(cost.e2e_ms for cost in pool)
        if (
            self.policy in (Policy.LONG_TERM, Policy.LONG_TERM_KV)
            and selected.e2e_ms > immediate_best + 1e-9
        ):
            selected.selection_reason = "future_cost"
        else:
            selected.selection_reason = "immediate_cost"
        return selected

    # ----- future value (long-term) ---------------------------------------
    def _remaining_turn_weights(self, request) -> List[float]:
        expected_turns = float(
            getattr(request, "expected_session_turns", self.expected_session_turns)
        )
        remaining = max(expected_turns - request.turn_index - 1, 0.0)
        whole = int(remaining)
        weights = [1.0] * whole
        fraction = remaining - whole
        if fraction > 1e-9:
            weights.append(fraction)
        return weights

    def _future_entry_distributions(
        self,
        request,
        turn_weights: List[float],
    ) -> List[Dict[int, float]]:
        nodes = self.dir.node_ids()
        n = len(nodes)
        if n <= 1:
            return [{nodes[0]: 1.0} for _ in turn_weights]

        mode = getattr(request, "mobility_granularity", "session")
        move_probability = max(
            0.0, min(1.0, float(getattr(request, "mobility_ratio", 0.0)))
        )
        interval = max(
            float(getattr(request, "expected_interarrival_ms", 0.0)), 0.0
        )
        mobility_start = float(
            getattr(request, "mobility_start_ms", float("inf"))
        )

        def active_at(step: int) -> bool:
            return (
                getattr(request, "mobility_active", False)
                or request.arrival_ms + (step + 1) * interval >= mobility_start
            )

        if mode == "request":
            distributions = []
            for step in range(len(turn_weights)):
                if not active_at(step):
                    distributions.append({request.home_node: 1.0})
                    continue
                dist = {
                    node: move_probability / (n - 1)
                    for node in nodes if node != request.home_node
                }
                dist[request.home_node] = 1.0 - move_probability
                distributions.append(dist)
            return distributions

        if mode == "session":
            if getattr(request, "mobility_active", False):
                return [{request.entry_node: 1.0} for _ in turn_weights]
            distributions = []
            for step in range(len(turn_weights)):
                if not active_at(step):
                    distributions.append({request.home_node: 1.0})
                    continue
                dist = {
                    node: move_probability / (n - 1)
                    for node in nodes if node != request.home_node
                }
                dist[request.home_node] = 1.0 - move_probability
                distributions.append(dist)
            return distributions

        # Markov state is (current entry, turns already resident there).
        residency = max(
            int(getattr(request, "mobility_residency_turns", 0)), 0
        )
        states = {
            (
                request.entry_node,
                int(getattr(request, "mobility_residency_age", 0)),
            ): 1.0
        }
        distributions = []
        for step in range(len(turn_weights)):
            next_states: Dict[tuple, float] = {}
            is_active = active_at(step)
            for (node, age), probability in states.items():
                if is_active and age >= residency:
                    stay = (node, age + 1)
                    next_states[stay] = (
                        next_states.get(stay, 0.0)
                        + probability * (1.0 - move_probability)
                    )
                    for dst in nodes:
                        if dst == node:
                            continue
                        moved = (dst, 1)
                        next_states[moved] = (
                            next_states.get(moved, 0.0)
                            + probability * move_probability / (n - 1)
                        )
                else:
                    stayed = (node, age + 1)
                    next_states[stayed] = (
                        next_states.get(stayed, 0.0) + probability
                    )
            states = next_states
            node_dist: Dict[int, float] = {}
            for (node, _), probability in states.items():
                node_dist[node] = node_dist.get(node, 0.0) + probability
            distributions.append(node_dist)
        return distributions

    def _future_service_cost(
        self,
        request,
        exec_node: int,
        distributions: List[Dict[int, float]],
        turn_weights: List[float],
    ) -> float:
        request_bytes = self._request_payload_bytes(request)
        response_bytes = self._response_payload_bytes(request.output_len)
        total = 0.0
        for weight, distribution in zip(turn_weights, distributions):
            for entry, probability in distribution.items():
                if entry == exec_node:
                    continue
                total += weight * probability * (
                    self.dir.net.transfer_time_ms(
                        entry, exec_node, request_bytes, contention=False
                    )
                    + self.dir.net.transfer_time_ms(
                        exec_node, entry, response_bytes, contention=False
                    )
                )
        return total

    def _future_value(self, request, cost: ActionCost) -> float:
        turn_weights = self._remaining_turn_weights(request)
        if not turn_weights:
            return 0.0

        # Session mobility reveals the new destination only when the first
        # request arrives there. Before that observation, choosing one of the
        # symmetric candidate nodes is speculative and should not pull the
        # placement away from the immediate-cost optimum.
        if (
            getattr(request, "mobility_granularity", "session") == "session"
            and not getattr(request, "mobility_active", False)
        ):
            return 0.0

        distributions = self._future_entry_distributions(request, turn_weights)
        current_placement = cost.action.exec_node
        state_bytes = self.model.kv_bytes_for_tokens(
            request.input_tokens + request.output_len
        )

        # Option 1: leave KV at the current placement and forward future RPCs.
        best = self._future_service_cost(
            request, current_placement, distributions, turn_weights
        )

        # Option 2: relocate state once to another node, then serve from there.
        for dst in self.dir.node_ids():
            if dst == current_placement:
                continue
            migrate_once = self.dir.net.transfer_time_ms(
                current_placement, dst, state_bytes, contention=False
            )
            recompute_once = self.dir.node(dst).compute.recompute_time_ms(
                request.input_tokens + request.output_len
            )
            relocation = min(migrate_once, recompute_once)
            future = relocation + self._future_service_cost(
                request, dst, distributions, turn_weights
            )
            best = min(best, future)
        return best

    # ----- selection -------------------------------------------------------
    def route(self, request) -> ActionCost:
        hashes = self._prefix_hashes(request)
        actions = self._enumerate(request, hashes)
        costs = [self._cost(request, a) for a in actions]

        if self.policy == Policy.NEAREST:
            return self._annotate_selection(
                request, self._select_nearest(request, costs), costs
            )

        feasible = [c for c in costs if c.feasible]
        pool = feasible if feasible else costs  # if none feasible, still pick least-bad

        if self.policy == Policy.GREEDY:
            best = min(pool, key=lambda c: c.e2e_ms)
            best.q_value = best.e2e_ms
            return self._annotate_selection(request, best, costs)

        # LONG_TERM / LONG_TERM_KV
        for c in pool:
            c.future_cost_ms = self._future_value(request, c)
            c.q_value = c.e2e_ms + self.gamma * c.future_cost_ms
        return self._annotate_selection(
            request, min(pool, key=lambda c: c.q_value), costs
        )

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

        node.add_load(
            prefill_ms=decision.t_prefill_ms,
            recompute_ms=(
                decision.t_state_ms
                if decision.action.mode == StateMode.RECOMPUTE
                else 0.0
            ),
            # Decode is assumed to be absorbed by continuous batching rather
            # than serialized in the admission queue.
            decode_ms=0.0,
        )
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
    kv_capacity_bytes: Optional[float] = None,
    activation_reserve_bytes: float = 4e9,
    token_id_bytes: int = 4,
    request_overhead_bytes: int = 4096,
    response_overhead_bytes: int = 4096,
    visual_bytes_per_token: int = 0,
) -> Dict:
    """Replay a request trace under a policy and return aggregate metrics.

    When ``collect_records`` is set, also returns per-request records and link
    utilisation, suitable for building a metrics dashboard.
    """
    from .node import build_cluster

    network.reset_stats()
    cluster = build_cluster(
        model, hardware, network, num_nodes, staleness_ms,
        kv_capacity_bytes=kv_capacity_bytes,
        activation_reserve_bytes=activation_reserve_bytes,
    )
    routers = {
        i: Router(
            model,
            cluster,
            policy,
            gamma=gamma,
            sla_margin_ms=sla_margin_ms,
            token_id_bytes=token_id_bytes,
            request_overhead_bytes=request_overhead_bytes,
            response_overhead_bytes=response_overhead_bytes,
            visual_bytes_per_token=visual_bytes_per_token,
        )
        for i in range(num_nodes)
    }

    reqs = sorted(
        [r for r in requests if r.model_name == model.name], key=lambda r: r.arrival_ms
    )
    prev_t = 0.0
    e2e_list: List[float] = []
    ttft_list: List[float] = []
    future_cost_list: List[float] = []
    q_value_list: List[float] = []
    component_totals = {
        "request_network": 0.0,
        "queue_prefill": 0.0,
        "queue_recompute": 0.0,
        "queue_decode": 0.0,
        "migration": 0.0,
        "recompute": 0.0,
        "prefill": 0.0,
        "decode": 0.0,
        "response_network": 0.0,
    }
    sla_viol = 0
    infeasible = 0
    cross_node = 0
    mode_counts = {m.value: 0 for m in StateMode}
    migrate_reason_counts: Dict[str, int] = {}
    migrate_reason_bytes: Dict[str, int] = {}
    records: List[Dict] = []
    queue_by_node = {
        i: {
            "requests": 0,
            "queue_ms": 0.0,
            "queue_prefill_ms": 0.0,
            "queue_recompute_ms": 0.0,
            "queue_decode_ms": 0.0,
        }
        for i in range(num_nodes)
    }
    last_t = 0.0
    offload_cost_ms = 0.0  # cumulative network + state-transfer cost

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
        node_queue = queue_by_node[decision.action.exec_node]
        node_queue["requests"] += 1
        node_queue["queue_ms"] += decision.t_queue_ms
        node_queue["queue_prefill_ms"] += decision.t_queue_prefill_ms
        node_queue["queue_recompute_ms"] += decision.t_queue_recompute_ms
        node_queue["queue_decode_ms"] += decision.t_queue_decode_ms

        migrate_bytes_before = cluster.kv.stats["migrate_bytes"]
        router.commit(r, decision, r.arrival_ms)
        request_migrate_bytes = (
            cluster.kv.stats["migrate_bytes"] - migrate_bytes_before
        )
        if decision.action.mode == StateMode.MIGRATE:
            reason = decision.selection_reason or "unclassified"
            migrate_reason_counts[reason] = migrate_reason_counts.get(reason, 0) + 1
            migrate_reason_bytes[reason] = (
                migrate_reason_bytes.get(reason, 0) + request_migrate_bytes
            )
        e2e_list.append(decision.e2e_ms)
        ttft_list.append(decision.ttft_ms)
        future_cost_list.append(decision.future_cost_ms)
        q_value_list.append(decision.q_value)
        state_cost = decision.t_state_ms if decision.t_state_ms != float("inf") else 0.0
        component_totals["request_network"] += decision.t_network_ms
        component_totals["queue_prefill"] += decision.t_queue_prefill_ms
        component_totals["queue_recompute"] += decision.t_queue_recompute_ms
        component_totals["queue_decode"] += decision.t_queue_decode_ms
        state_component = (
            "migration"
            if decision.action.mode == StateMode.MIGRATE
            else "recompute"
        )
        if decision.action.mode in (StateMode.MIGRATE, StateMode.RECOMPUTE):
            component_totals[state_component] += state_cost
        component_totals["prefill"] += decision.t_prefill_ms
        component_totals["decode"] += decision.t_decode_ms
        component_totals["response_network"] += decision.t_return_ms
        offload_cost_ms += (
            decision.t_network_ms + decision.t_queue_ms
            + state_cost + decision.t_return_ms
        )
        last_t = r.arrival_ms

        if collect_records:
            records.append({
                "t": round(r.arrival_ms, 2),
                "ttft": round(decision.ttft_ms, 3),
                "e2e": round(decision.e2e_ms, 3),
                "predicted_future_cost": round(decision.future_cost_ms, 3),
                "q_value": round(decision.q_value, 3),
                "t_network": round(decision.t_network_ms, 3),
                "t_queue": round(decision.t_queue_ms, 3),
                "t_queue_prefill": round(decision.t_queue_prefill_ms, 3),
                "t_queue_recompute": round(decision.t_queue_recompute_ms, 3),
                "t_queue_decode": round(decision.t_queue_decode_ms, 3),
                "t_state": round(decision.t_state_ms, 3),
                "t_prefill": round(decision.t_prefill_ms, 3),
                "t_decode": round(decision.t_decode_ms, 3),
                "t_return": round(decision.t_return_ms, 3),
                "mode": decision.action.mode.value,
                "migrate_reason": decision.selection_reason,
                "migrate_bytes": request_migrate_bytes,
                "entry": r.entry_node,
                "exec": decision.action.exec_node,
                "cross": int(is_cross),
                "sla_violation": int(is_sla),
                "infeasible": int(is_infeasible),
                "priority": r.priority,
                "group_name": r.group_name,
                "moved": int(r.mobility_switched),
                "mobility_transition": int(r.mobility_transitioned),
            })

    def pct(v, p):
        if not v:
            return 0.0
        s = sorted(v)
        return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]

    n = max(len(reqs), 1)
    avg_components = {
        f"avg_{name}_ms": total / n
        for name, total in component_totals.items()
    }
    result = {
        "policy": policy.value,
        "model": model.name,
        "num_requests": len(reqs),
        "avg_e2e_ms": sum(e2e_list) / n,
        "avg_predicted_future_cost_ms": sum(future_cost_list) / n,
        "avg_q_value_ms": sum(q_value_list) / n,
        "p95_e2e_ms": pct(e2e_list, 95),
        "p50_ttft_ms": pct(ttft_list, 50),
        "p95_ttft_ms": pct(ttft_list, 95),
        "p99_ttft_ms": pct(ttft_list, 99),
        "sla_violation_ratio": sla_viol / n,
        "infeasible_ratio": infeasible / n,
        "cross_node_ratio": cross_node / n,
        "mobility_transition_count": sum(
            int(r.mobility_transitioned) for r in reqs
        ),
        "local_count": mode_counts["local"],
        "fresh_count": mode_counts["fresh"],
        "migrate_count": mode_counts["migrate"],
        "migrate_reason_counts": migrate_reason_counts,
        "migrate_reason_bytes_mb": {
            reason: value / 1e6
            for reason, value in migrate_reason_bytes.items()
        },
        "recompute_count": mode_counts["recompute"],
        "owner_switch_count": cluster.kv.stats["owner_switch_count"],
        "migrate_bytes_mb": cluster.kv.stats["migrate_bytes"] / 1e6,
        "offload_cost_ms": offload_cost_ms,
        **avg_components,
    }
    result["avg_state_ms"] = (
        result["avg_migration_ms"] + result["avg_recompute_ms"]
    )
    result["avg_queue_ms"] = (
        result["avg_queue_prefill_ms"]
        + result["avg_queue_recompute_ms"]
        + result["avg_queue_decode_ms"]
    )
    result["avg_e2e_component_sum_ms"] = sum(avg_components.values())
    result["queue_by_exec_node"] = {
        str(node): {
            "requests": values["requests"],
            "request_ratio": values["requests"] / n,
            "avg_queue_ms": values["queue_ms"] / max(values["requests"], 1),
            "avg_queue_prefill_ms": (
                values["queue_prefill_ms"] / max(values["requests"], 1)
            ),
            "avg_queue_recompute_ms": (
                values["queue_recompute_ms"] / max(values["requests"], 1)
            ),
            "avg_queue_decode_ms": (
                values["queue_decode_ms"] / max(values["requests"], 1)
            ),
        }
        for node, values in queue_by_node.items()
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
