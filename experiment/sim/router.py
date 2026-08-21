"""Per-node request router.

Enumerates local/migrate/recompute/fresh actions, prices them with the
compute / network / kv modules, filters by SLA and memory, and selects an
action under nearest, greedy, long-term, proactive-KV, or oracle-placement
upper-bound policies.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, List, Optional

from .kv_cache import KVBlock, MigrationPlan, block_hashes_for_len, make_blocks
from .kv_manager import KVManagerConfig, ProactiveKVManager
from .large_model import ModelSpec
from .node import GlobalStateDirectory


class Policy(str, Enum):
    NEAREST = "nearest"
    GREEDY = "greedy"
    GREEDY_KV = "greedy_kv"
    LONG_TERM = "long_term"
    LONG_TERM_KV = "long_term_kv"
    GREEDY_ROLLOUT = "greedy_rollout"
    ORACLE_PREFETCH = "oracle_prefetch"
    ORACLE_KV = "oracle_kv"


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
    queue_prefill_service_ms: float
    queue_recompute_service_ms: float
    queue_latest_start_ms: float
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
    future_driven: bool = False
    entry_ready_ratio: float = 0.0
    entry_residual_bytes: float = 0.0
    entry_sync_ms: float = 0.0
    entry_recompute_ms: float = 0.0
    entry_recovery_ms: float = 0.0
    entry_recovery_mode: str = "none"
    forwarding_premium_ms: float = 0.0
    kv_owner_queue_ms: float = 0.0
    ideal_entry_queue_ms: float = 0.0
    kv_owner_e2e_ms: float = 0.0
    ideal_entry_e2e_ms: float = 0.0
    complete_kv_owner_queue_ms: float = 0.0
    complete_kv_owner_e2e_ms: float = 0.0
    breakpoint_turns: Optional[float] = None
    remaining_turns: int = 0
    remaining_entry_turns: int = 0
    breakpoint_region: str = "no_history"


class Router:
    def __init__(
        self,
        model: ModelSpec,
        directory: GlobalStateDirectory,
        policy: Policy = Policy.GREEDY,
        gamma: float = 1.0,
        decode_batch_size: int = 32,
        sla_margin_ms: float = 20.0,
        expected_session_turns: int = 4,
        model_version: str = "v1",
        token_id_bytes: int = 4,
        request_overhead_bytes: int = 4096,
        response_overhead_bytes: int = 4096,
        visual_bytes_per_token: int = 0,
        block_level_kv: Optional[bool] = None,
        oracle_session_requests: Optional[Dict[str, List]] = None,
        state_recovery_scale: float = 1.0,
    ):
        self.model = model
        self.dir = directory
        self.policy = policy
        self.gamma = gamma
        self.decode_batch_size = max(int(decode_batch_size), 1)
        self.sla_margin_ms = sla_margin_ms
        self.expected_session_turns = expected_session_turns
        self.model_version = model_version
        self.token_id_bytes = token_id_bytes
        self.request_overhead_bytes = request_overhead_bytes
        self.response_overhead_bytes = response_overhead_bytes
        self.visual_bytes_per_token = visual_bytes_per_token
        self.oracle_session_requests = oracle_session_requests or {}
        self.state_recovery_scale = max(float(state_recovery_scale), 0.0)
        self.block_level_kv = (
            policy in (
                Policy.GREEDY_KV,
                Policy.LONG_TERM_KV,
                Policy.ORACLE_PREFETCH,
                Policy.ORACLE_KV,
            )
            if block_level_kv is None
            else block_level_kv
        )

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
        # Whole-prefix recovery needs one source that actually contains every
        # block.  Looking only at the first block is invalid once old or
        # proactive replicas exist on several nodes.
        complete_sources = self.dir.kv.locate(hashes[0])
        for block_hash in hashes[1:located]:
            complete_sources &= self.dir.kv.locate(block_hash)
            if not complete_sources:
                return None
        return min(complete_sources) if complete_sources else None

    # ----- action enumeration ---------------------------------------------
    def _enumerate(self, request, hashes: List[str]) -> List[Action]:
        bs = self.model.kv_block_size
        actions: List[Action] = []
        located, _ = self._prefix_stats(hashes, request.entry_node)
        located_tokens = min(located * bs, request.prefix_tokens)

        for node in self.dir.node_ids():
            if not hashes:
                # First request: no reusable session KV exists.  The initial
                # shared/system prefix is part of this request's full prefill.
                actions.append(Action(node, StateMode.FRESH, hit_tokens=0))
                continue
            _, local_blocks = self._prefix_stats(hashes, node)
            local_tokens = min(
                local_blocks * bs,
                request.prefix_tokens,
            )
            residual_tokens = max(request.prefix_tokens - local_tokens, 0)
            if located < len(hashes):
                # Some historical blocks are no longer recoverable from any
                # node.  A target that still owns a contiguous prefix can use
                # it as attention history and prefill only the missing suffix.
                actions.append(
                    Action(
                        node,
                        StateMode.RECOMPUTE,
                        hit_tokens=residual_tokens,
                    )
                )
                continue
            if local_blocks >= located:
                actions.append(
                    Action(node, StateMode.LOCAL, src_node=node, hit_tokens=located_tokens)
                )
            else:
                actions.append(self._migrate_action(hashes[:located], node, located_tokens))
                actions.append(
                    Action(node, StateMode.RECOMPUTE, hit_tokens=residual_tokens)
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

        if action.mode == StateMode.MIGRATE:
            if action.src_node is None:
                t_state = float("inf")
            else:
                t_state = net.transfer_time_ms(
                    action.src_node, action.exec_node, action.migrate_bytes, contention=True
                )
        elif action.mode == StateMode.RECOMPUTE:
            cached_prefix_tokens = max(
                request.prefix_tokens - action.hit_tokens,
                0,
            )
            t_state = compute.estimate_incremental_prefill(
                cached_prefix_tokens,
                action.hit_tokens,
            ).prefill_ms
        else:  # LOCAL / FRESH
            t_state = 0.0

        if action.mode in (StateMode.MIGRATE, StateMode.RECOMPUTE):
            t_state *= self.state_recovery_scale

        # ``hit_tokens`` describes historical prefix KV; it cannot eliminate
        # prefill for the newly arrived input tokens.  A FRESH first request
        # must additionally prefill its initial shared/system prefix.
        if action.mode == StateMode.FRESH:
            new_prefill_tokens = request.prefix_tokens + request.input_tokens
            t_prefill = compute.estimate_prefill(new_prefill_tokens).prefill_ms
            queue_prefill_service = compute.estimate_prefill_service(
                new_prefill_tokens,
                batch_size=node.prefill_batch_size,
            ).prefill_ms
        else:
            new_prefill_tokens = request.input_tokens
            t_prefill = compute.estimate_incremental_prefill(
                request.prefix_tokens,
                new_prefill_tokens,
            ).prefill_ms
            queue_prefill_service = (
                compute.estimate_incremental_prefill_service(
                    request.prefix_tokens,
                    new_prefill_tokens,
                    batch_size=node.prefill_batch_size,
                ).prefill_ms
            )
        queue_recompute_service = (
            compute.estimate_incremental_prefill_service(
                max(request.prefix_tokens - action.hit_tokens, 0),
                action.hit_tokens,
                batch_size=node.prefill_batch_size,
            ).prefill_ms
            if action.mode == StateMode.RECOMPUTE
            else 0.0
        )
        queue_recompute_service *= self.state_recovery_scale

        base_ttft = t_network + t_state + t_prefill + t_first_token_return
        queue_latest_start = (
            request.arrival_ms
            + request.sla_ms
            - self.sla_margin_ms
            - base_ttft
        )
        (
            t_queue,
            t_queue_prefill,
            t_queue_recompute,
            t_queue_decode,
        ) = node_state.queue_before(queue_latest_start)
        ttft = base_ttft + t_queue
        decode = compute.estimate_amortized_decode(
            # Prefill produces the logits for the first output token.  Only
            # the remaining output tokens require autoregressive decode steps.
            gen_tokens=max(request.output_len - 1, 0),
            ctx_len=request.prefix_tokens + request.input_tokens,
            batch_size=self.decode_batch_size,
        ).total_ms
        e2e = ttft + decode + max(t_return - t_first_token_return, 0.0)

        appended_kv = self.model.kv_bytes_for_tokens(
            request.input_tokens + request.output_len
        )
        if action.mode == StateMode.LOCAL:
            new_kv = appended_kv
        elif action.mode == StateMode.MIGRATE:
            # Network bytes and newly occupied memory are different when the
            # target already retains an older prefix.  Whole-prefix mode still
            # transfers the full prefix, but existing local blocks do not
            # consume memory twice.
            hashes = self._prefix_hashes(request)
            _, local_blocks = self._prefix_stats(hashes, action.exec_node)
            local_tokens = min(
                local_blocks * self.model.kv_block_size,
                request.prefix_tokens,
            )
            local_bytes = self.model.kv_bytes_for_tokens(local_tokens)
            new_kv = max(action.migrate_bytes - local_bytes, 0) + appended_kv
        elif action.mode == StateMode.RECOMPUTE:
            # Existing contiguous prefix blocks remain reusable.  Recompute
            # only materializes the missing historical suffix, then appends
            # KV generated by the current request.
            new_kv = self.model.kv_bytes_for_tokens(
                action.hit_tokens + request.input_tokens + request.output_len
            )
        else:  # FRESH builds the complete initial context at the target.
            new_kv = self.model.kv_bytes_for_tokens(
                request.prefix_tokens + request.input_tokens + request.output_len
            )

        feasible, reason = self._feasible(request, action, ttft, new_kv, node_state)
        return ActionCost(
            action=action, t_network_ms=t_network, t_queue_ms=t_queue,
            t_queue_prefill_ms=t_queue_prefill,
            t_queue_recompute_ms=t_queue_recompute,
            t_queue_decode_ms=t_queue_decode,
            t_state_ms=t_state, t_prefill_ms=t_prefill, t_decode_ms=decode,
            queue_prefill_service_ms=queue_prefill_service,
            queue_recompute_service_ms=queue_recompute_service,
            queue_latest_start_ms=queue_latest_start,
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

        if not any(cost.feasible for cost in costs):
            selected.selection_reason = "global_no_feasible_action"
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
            self.policy in (
                Policy.LONG_TERM,
                Policy.LONG_TERM_KV,
                Policy.GREEDY_ROLLOUT,
                Policy.ORACLE_PREFETCH,
                Policy.ORACLE_KV,
            )
            and selected.e2e_ms > immediate_best + 1e-9
        ):
            selected.selection_reason = "future_cost"
        else:
            selected.selection_reason = "immediate_cost"
        return selected

    def _attach_breakpoint_diagnostics(
        self,
        request,
        selected: ActionCost,
        costs: List[ActionCost],
        hashes: List[str],
    ) -> ActionCost:
        """Classify the current request by its state-locality breakpoint.

        The comparison is deliberately local and interpretable: residual
        recovery at the ingress is compared with the per-turn cost premium of
        forwarding to a node that already has the complete prefix.  The real
        remaining session length comes from the generated trace.
        """
        future_requests = self._known_future_requests(request)
        selected.remaining_turns = 1 + len(future_requests)
        selected.remaining_entry_turns = 1 + sum(
            future.entry_node == request.entry_node
            for future in future_requests
        )
        if not hashes:
            selected.entry_ready_ratio = 1.0
            return selected

        # Trade-off diagnostic independent of the request ingress: identify
        # the best alternative node that can reuse the complete current KV.
        # A selected MIGRATE/RECOMPUTE action can then be compared directly
        # with waiting at this complete-state owner.
        complete_owner_costs = [
            candidate for candidate in costs
            if candidate.action.mode == StateMode.LOCAL
        ]
        feasible_owners = [
            candidate for candidate in complete_owner_costs
            if candidate.feasible
        ]
        owner_pool = feasible_owners or complete_owner_costs
        if owner_pool:
            complete_owner = min(
                owner_pool, key=lambda candidate: candidate.e2e_ms
            )
            selected.complete_kv_owner_queue_ms = complete_owner.t_queue_ms
            selected.complete_kv_owner_e2e_ms = complete_owner.e2e_ms

        _, local_blocks = self._prefix_stats(hashes, request.entry_node)
        selected.entry_ready_ratio = local_blocks / max(len(hashes), 1)
        if local_blocks >= len(hashes):
            # The ingress itself is the complete-KV owner.  Preserve its
            # counterfactual queue/E2E before returning so cross-node migrate
            # decisions can be compared with staying at the congested owner.
            entry_local = [
                candidate for candidate in costs
                if candidate.action.exec_node == request.entry_node
                and candidate.action.mode == StateMode.LOCAL
            ]
            if entry_local:
                owner_cost = min(entry_local, key=lambda candidate: candidate.e2e_ms)
                selected.kv_owner_queue_ms = owner_cost.t_queue_ms
                selected.kv_owner_e2e_ms = owner_cost.e2e_ms
                selected.ideal_entry_queue_ms = owner_cost.t_queue_ms
                selected.ideal_entry_e2e_ms = owner_cost.e2e_ms
            selected.breakpoint_region = "already_ready"
            return selected

        entry_costs = [
            candidate for candidate in costs
            if candidate.action.exec_node == request.entry_node
        ]
        entry_pool = [candidate for candidate in entry_costs if candidate.feasible]
        if not entry_pool:
            entry_pool = entry_costs
        if not entry_pool:
            return selected
        entry_best = min(entry_pool, key=lambda candidate: candidate.e2e_ms)

        migrate_to_entry = [
            candidate for candidate in entry_costs
            if candidate.action.mode == StateMode.MIGRATE
        ]
        if migrate_to_entry:
            best_sync = min(
                migrate_to_entry, key=lambda candidate: candidate.t_state_ms
            )
            selected.entry_residual_bytes = best_sync.action.migrate_bytes
            selected.entry_sync_ms = best_sync.t_state_ms
        recompute_at_entry = [
            candidate for candidate in entry_costs
            if candidate.action.mode == StateMode.RECOMPUTE
        ]
        if recompute_at_entry:
            selected.entry_recompute_ms = min(
                candidate.t_state_ms for candidate in recompute_at_entry
            )

        ideal_entry = self._cost(
            request,
            Action(
                request.entry_node,
                StateMode.LOCAL,
                src_node=request.entry_node,
                hit_tokens=request.prefix_tokens,
            ),
        )
        selected.entry_recovery_ms = max(
            entry_best.e2e_ms - ideal_entry.e2e_ms,
            0.0,
        )
        selected.entry_recovery_mode = entry_best.action.mode.value
        selected.ideal_entry_queue_ms = ideal_entry.t_queue_ms
        selected.ideal_entry_e2e_ms = ideal_entry.e2e_ms

        remote_local = [
            candidate for candidate in costs
            if candidate.action.mode == StateMode.LOCAL
            and candidate.action.exec_node != request.entry_node
        ]
        if not remote_local:
            selected.breakpoint_region = "no_remote_owner"
            return selected
        forward_best = min(remote_local, key=lambda candidate: candidate.e2e_ms)
        selected.kv_owner_queue_ms = forward_best.t_queue_ms
        selected.kv_owner_e2e_ms = forward_best.e2e_ms
        selected.forwarding_premium_ms = (
            forward_best.e2e_ms - ideal_entry.e2e_ms
        )
        if selected.forwarding_premium_ms <= 0.0:
            selected.breakpoint_region = "no_local_advantage"
            return selected

        selected.breakpoint_turns = (
            selected.entry_recovery_ms / selected.forwarding_premium_ms
        )
        if selected.breakpoint_turns <= 1.0:
            selected.breakpoint_region = "greedy_switch"
        elif selected.breakpoint_turns < selected.remaining_entry_turns:
            selected.breakpoint_region = "long_term_gap"
        else:
            selected.breakpoint_region = "not_amortizable"
        return selected

    def _finalize_selection(
        self,
        request,
        selected: ActionCost,
        costs: List[ActionCost],
        hashes: List[str],
    ) -> ActionCost:
        selected = self._annotate_selection(request, selected, costs)
        return self._attach_breakpoint_diagnostics(
            request, selected, costs, hashes
        )

    # ----- oracle-informed future value (long-term heuristic) -------------
    def _known_future_requests(self, request) -> List:
        """Return the actual remaining requests of this session in the trace.

        The experiment deliberately removes predictor error: session
        continuation, future ingress, input length and response length are all
        read from the generated trace.  This is an oracle-informed heuristic,
        not a learned value function.
        """
        rows = self.oracle_session_requests.get(request.session_id, [])
        return [row for row in rows if row.turn_index > request.turn_index]

    def _future_value(self, request, cost: ActionCost) -> float:
        future_requests = self._known_future_requests(request)
        if not future_requests:
            return 0.0

        nodes = self.dir.node_ids()
        snapshot = self.dir.snapshot()
        net = self.dir.net
        # V_h(owner) is the minimum discounted E2E cost from future request h
        # onward when the latest complete KV prefix initially resides at owner.
        next_value = {owner: 0.0 for owner in nodes}
        current_owner = cost.action.exec_node
        current_added_load = (
            cost.queue_prefill_service_ms
            + cost.queue_recompute_service_ms
        )

        for future in reversed(future_requests):
            elapsed = max(future.arrival_ms - request.arrival_ms, 0.0)
            request_bytes = self._request_payload_bytes(future)
            response_bytes = self._response_payload_bytes(future.output_len)
            state_bytes = self.model.kv_bytes_for_tokens(future.prefix_tokens)
            current_value: Dict[int, float] = {}

            for owner in nodes:
                best = float("inf")
                for exec_node in nodes:
                    compute = self.dir.node(exec_node).compute
                    request_network = 0.0
                    response_network = 0.0
                    if exec_node != future.entry_node:
                        request_network = net.transfer_time_ms(
                            future.entry_node,
                            exec_node,
                            request_bytes,
                            contention=False,
                        )
                        response_network = net.transfer_time_ms(
                            exec_node,
                            future.entry_node,
                            response_bytes,
                            contention=False,
                        )

                    recovery = 0.0
                    if exec_node != owner and future.prefix_tokens > 0:
                        migrate = net.transfer_time_ms(
                            owner,
                            exec_node,
                            state_bytes,
                            contention=False,
                        )
                        # Rebuilding state at another node requires the entire
                        # historical prefix, not only the newest turn.
                        recompute = compute.recompute_time_ms(
                            future.prefix_tokens
                        )
                        recovery = min(migrate, recompute)

                    queue_now = snapshot[exec_node].estimated_queue_ms
                    if exec_node == current_owner:
                        queue_now += current_added_load
                    queue_at_arrival = max(queue_now - elapsed, 0.0)
                    prefill = compute.estimate_incremental_prefill(
                        future.prefix_tokens,
                        future.input_tokens,
                    ).prefill_ms
                    decode = compute.estimate_amortized_decode(
                        gen_tokens=max(future.output_len - 1, 0),
                        ctx_len=future.prefix_tokens + future.input_tokens,
                        batch_size=self.decode_batch_size,
                    ).total_ms
                    immediate = (
                        request_network
                        + queue_at_arrival
                        + recovery
                        + prefill
                        + decode
                        + response_network
                    )
                    candidate = immediate + self.gamma * next_value[exec_node]
                    best = min(best, candidate)
                current_value[owner] = best
            next_value = current_value

        return next_value[current_owner]

    def _greedy_rollout_value(self, request, cost: ActionCost) -> float:
        """Replay the rest of this session after one candidate action.

        The current candidate is committed to a private copy of the simulator
        state.  Every actual remaining request of the same session is then
        routed by the ordinary myopic Greedy policy and committed through the
        normal simulator path.  The returned value is the undiscounted sum of
        those future E2E latencies.

        This is a one-step policy-rollout heuristic, not a global oracle: it
        does not replay requests from other sessions and it does not schedule
        proactive KV placement inside the private rollout.
        """
        future_requests = self._known_future_requests(request)
        if not future_requests:
            return 0.0

        rollout_dir = self._lightweight_rollout_directory(request)
        rollout_router = Router(
            self.model,
            rollout_dir,
            Policy.GREEDY,
            gamma=1.0,
            decode_batch_size=self.decode_batch_size,
            sla_margin_ms=self.sla_margin_ms,
            expected_session_turns=self.expected_session_turns,
            model_version=self.model_version,
            token_id_bytes=self.token_id_bytes,
            request_overhead_bytes=self.request_overhead_bytes,
            response_overhead_bytes=self.response_overhead_bytes,
            visual_bytes_per_token=self.visual_bytes_per_token,
            block_level_kv=self.block_level_kv,
            oracle_session_requests=self.oracle_session_requests,
        )

        rollout_router.commit(request, cost, request.arrival_ms)
        previous_arrival = request.arrival_ms
        total = 0.0
        for future in future_requests:
            for node in rollout_dir.nodes.values():
                node.advance_to(future.arrival_ms, previous_arrival)
            rollout_dir.refresh(future.arrival_ms, force=True)
            future_cost = rollout_router.route(future)
            total += future_cost.e2e_ms
            rollout_router.commit(future, future_cost, future.arrival_ms)
            previous_arrival = future.arrival_ms
        return total

    def _lightweight_rollout_directory(self, request) -> GlobalStateDirectory:
        """Clone only state needed by a single-session routing rollout.

        Copying the complete global KV directory for every candidate action is
        quadratic in the number of sessions and quickly exhausts memory.  A
        rollout only needs the current session's block metadata.  KV occupied
        by other sessions is represented by one evictable placeholder per
        node so that the cloned stores start with the same free capacity.
        """
        from .node import build_cluster
        from .network import NetworkSimulator

        original_nodes = self.dir.nodes
        first_node = original_nodes[min(original_nodes)]
        rollout_network = NetworkSimulator(deepcopy(self.dir.net.topology))
        for key, original_state in self.dir.net._state.items():
            clone_state = rollout_network._state[key]
            clone_state.active_flows = original_state.active_flows
            clone_state.peak_concurrency = original_state.peak_concurrency
            clone_state.total_bytes = original_state.total_bytes
            clone_state.total_busy_ms = original_state.total_busy_ms
            clone_state.background_fraction = original_state.background_fraction
        rollout_dir = build_cluster(
            self.model,
            first_node.compute.hw,
            rollout_network,
            num_nodes=len(original_nodes),
            staleness_ms=self.dir.staleness_ms,
            kv_capacity_bytes=first_node.kv_store.capacity_bytes,
            activation_reserve_bytes=first_node.activation_reserve,
            prefill_batch_size=first_node.prefill_batch_size,
            sla_slack_scheduling=first_node.sla_slack_scheduling,
        )

        session_blocks = self.dir.kv.blocks_for_session(
            request.session_id,
            request.prefix_id,
            self.model.name,
        )
        cloned_blocks = {
            block.block_hash: deepcopy(block) for block in session_blocks
        }
        for block_hash, block in cloned_blocks.items():
            block.replicas = self.dir.kv.locate(block_hash)

        for node_id, original in original_nodes.items():
            clone = rollout_dir.node(node_id)
            clone.kv_store.capacity_bytes = original.kv_store.capacity_bytes
            local_blocks = [
                cloned_blocks[block.block_hash]
                for block in session_blocks
                if node_id in self.dir.kv.locate(block.block_hash)
            ]
            local_bytes = sum(block.size_bytes for block in local_blocks)
            other_bytes = max(original.kv_store.used_bytes() - local_bytes, 0)
            if other_bytes > 0:
                placeholder = KVBlock(
                    block_hash=f"__rollout_other_{node_id}",
                    session_id="__other_sessions__",
                    prefix_id="__other_sessions__",
                    model_name=self.model.name,
                    model_version=self.model_version,
                    block_index=0,
                    num_tokens=0,
                    size_bytes=int(other_bytes),
                    owner=node_id,
                    replicas={node_id},
                    last_access_ms=-1e18,
                )
                clone.kv_store.insert([placeholder], request.arrival_ms)
            clone.kv_store.insert(local_blocks, request.arrival_ms)
            for block in local_blocks:
                rollout_dir.kv.register(node_id, block)

            clone.assigned_load_ms = original.assigned_load_ms
            clone._queue_components = dict(original._queue_components)
            clone._running_job = deepcopy(original._running_job)
            clone._waiting_jobs = deepcopy(original._waiting_jobs)
            clone._queue_sequence = original._queue_sequence
            clone._ttft_samples = list(original._ttft_samples)
            clone.served = original.served

        rollout_dir.kv.stats = dict(self.dir.kv.stats)
        rollout_dir.refresh(request.arrival_ms, force=True)
        return rollout_dir

    # ----- selection -------------------------------------------------------
    def route(self, request) -> ActionCost:
        hashes = self._prefix_hashes(request)
        actions = self._enumerate(request, hashes)
        costs = [self._cost(request, a) for a in actions]

        if self.policy == Policy.NEAREST:
            return self._finalize_selection(
                request, self._select_nearest(request, costs), costs, hashes
            )

        feasible = [c for c in costs if c.feasible]
        pool = feasible if feasible else costs  # if none feasible, still pick least-bad

        if self.policy in (Policy.GREEDY, Policy.GREEDY_KV):
            best = min(pool, key=lambda c: c.e2e_ms)
            best.q_value = best.e2e_ms
            return self._finalize_selection(request, best, costs, hashes)

        if self.policy == Policy.GREEDY_ROLLOUT:
            for c in pool:
                c.future_cost_ms = self._greedy_rollout_value(request, c)
                # The generated trace has a finite, known session boundary;
                # use its undiscounted cumulative E2E objective.
                c.q_value = c.e2e_ms + c.future_cost_ms
            selected = min(pool, key=lambda c: c.q_value)
            immediate_best = min(pool, key=lambda c: c.e2e_ms)
            selected.future_driven = selected is not immediate_best
            return self._finalize_selection(
                request, selected, costs, hashes
            )

        # LONG_TERM / LONG_TERM_KV and oracle-placement variants
        future_cache: Dict[tuple, float] = {}
        for c in pool:
            added_load = (
                c.queue_prefill_service_ms
                + c.queue_recompute_service_ms
            )
            # Future DP depends on the resulting KV owner and the computation
            # added to that owner's queue, not on whether the current prefix
            # arrived via LOCAL or MIGRATE.  Reuse identical subproblems.
            cache_key = (c.action.exec_node, added_load)
            if cache_key not in future_cache:
                future_cache[cache_key] = self._future_value(request, c)
            c.future_cost_ms = future_cache[cache_key]
            c.q_value = c.e2e_ms + self.gamma * c.future_cost_ms
        selected = min(pool, key=lambda c: c.q_value)
        immediate_best = min(pool, key=lambda c: c.e2e_ms)
        selected.future_driven = selected is not immediate_best
        return self._finalize_selection(request, selected, costs, hashes)

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
            if self.block_level_kv:
                plan = kv.plan_migration(located_hashes[:located], exec_node, net)
            else:
                # The non-block-level ablation transfers and charges the whole
                # prefix even if the destination retains some old blocks.
                plan = MigrationPlan(
                    dst=exec_node,
                    src=decision.action.src_node,
                    missing_hashes=located_hashes[:located],
                    bytes_to_move=decision.action.migrate_bytes,
                    transfer_ms=decision.t_state_ms,
                )
            kv.commit_migration(plan, switch_owner=True)
        elif decision.action.mode == StateMode.RECOMPUTE:
            kv.note_recompute()

        # the grown session context now resides on exec_node
        context_tokens = (
            request.prefix_tokens + request.input_tokens + request.output_len
        )
        blocks = make_blocks(
            self.model, request.session_id, request.prefix_id, context_tokens,
            owner=exec_node, model_version=self.model_version, t_now=t_now,
        )
        evicted = node.kv_store.insert(blocks, t_now)
        for block_hash in evicted:
            kv.unregister(exec_node, block_hash)
        for b in blocks:
            if node.kv_store.contains(b.block_hash):
                kv.register(exec_node, b)
                kv.set_owner(b.block_hash, exec_node)
            else:
                kv.unregister(exec_node, b.block_hash)

        node.add_load(
            prefill_ms=decision.queue_prefill_service_ms,
            recompute_ms=decision.queue_recompute_service_ms,
            # Decode is assumed to be absorbed by continuous batching rather
            # than serialized in the admission queue.
            decode_ms=0.0,
            latest_start_ms=decision.queue_latest_start_ms,
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
    gamma: float = 1.0,
    decode_batch_size: int = 32,
    sla_margin_ms: float = 20.0,
    reject_intrinsically_infeasible: bool = True,
    collect_records: bool = False,
    kv_capacity_bytes: Optional[float] = None,
    activation_reserve_bytes: float = 64e9,
    prefill_batch_size: int = 4,
    sla_slack_scheduling: bool = True,
    token_id_bytes: int = 4,
    request_overhead_bytes: int = 4096,
    response_overhead_bytes: int = 4096,
    visual_bytes_per_token: int = 0,
    kv_manager_config: Optional[KVManagerConfig] = None,
    long_term_block_level_kv: bool = False,
    long_term_kv_block_level_kv: bool = True,
    state_recovery_scale: float = 1.0,
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
        prefill_batch_size=prefill_batch_size,
        sla_slack_scheduling=sla_slack_scheduling,
    )
    configured_manager = kv_manager_config or KVManagerConfig(enabled=False)
    # Proactive placement belongs to the heuristic and oracle-KV variants.
    # Passive block-level recovery remains independently configurable.
    manager_config = replace(
        configured_manager,
        enabled=(
            configured_manager.enabled
            and policy in (
                Policy.GREEDY_KV,
                Policy.LONG_TERM_KV,
                Policy.ORACLE_PREFETCH,
            )
        ),
    )
    if policy == Policy.LONG_TERM:
        block_level_kv = long_term_block_level_kv
    elif policy in (
        Policy.GREEDY_KV,
        Policy.LONG_TERM_KV,
        Policy.ORACLE_PREFETCH,
        Policy.ORACLE_KV,
    ):
        block_level_kv = long_term_kv_block_level_kv
    else:
        block_level_kv = False
    reqs = sorted(
        [r for r in requests if r.model_name == model.name], key=lambda r: r.arrival_ms
    )
    oracle_session_requests: Dict[str, List] = {}
    for req in reqs:
        oracle_session_requests.setdefault(req.session_id, []).append(req)
    for rows in oracle_session_requests.values():
        rows.sort(key=lambda req: req.turn_index)
    kv_manager = ProactiveKVManager(
        model,
        cluster,
        manager_config,
        oracle_session_requests=oracle_session_requests,
        oracle_placement=(policy == Policy.ORACLE_PREFETCH),
    )
    routers = {
        i: Router(
            model,
            cluster,
            policy,
            gamma=gamma,
            decode_batch_size=decode_batch_size,
            sla_margin_ms=sla_margin_ms,
            token_id_bytes=token_id_bytes,
            request_overhead_bytes=request_overhead_bytes,
            response_overhead_bytes=response_overhead_bytes,
            visual_bytes_per_token=visual_bytes_per_token,
            block_level_kv=block_level_kv,
            oracle_session_requests=oracle_session_requests,
            state_recovery_scale=state_recovery_scale,
        )
        for i in range(num_nodes)
    }
    prev_t = 0.0
    e2e_list: List[float] = []
    ttft_list: List[float] = []
    future_cost_list: List[float] = []
    q_value_list: List[float] = []
    future_driven_action_count = 0
    breakpoint_region_counts: Dict[str, int] = {}
    breakpoint_ready_ratios: List[float] = []
    breakpoint_residual_bytes: List[float] = []
    breakpoint_recovery_ms: List[float] = []
    breakpoint_sync_ms: List[float] = []
    breakpoint_recompute_ms: List[float] = []
    breakpoint_recovery_mode_counts: Dict[str, int] = {}
    breakpoint_forwarding_premiums: List[float] = []
    breakpoint_turn_values: List[float] = []
    breakpoint_gap_entry_choices = 0
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
    rejection_records: List[Dict] = []
    accepted_requests: List = []
    admission_rejected = 0
    admission_rejected_by_group: Dict[str, int] = {}
    admission_rejected_by_priority: Dict[str, int] = {}
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
    placement_scheduled_bytes = 0
    oracle_prepared_bytes = 0
    oracle_prepared_blocks = 0
    oracle_prepared_tasks = 0

    def prepare_oracle_entry(request) -> tuple[int, int]:
        """Materialise an ideal, zero-cost complete prefix at the ingress.

        This deliberately ignores bandwidth and capacity.  It is an
        optimistic KV-readiness bound, not an executable placement policy.
        """
        if request.prefix_tokens <= 0 or request.is_session_first:
            return 0, 0
        hashes = block_hashes_for_len(
            model.name,
            "v1",
            request.prefix_id,
            request.prefix_tokens,
            model.kv_block_size,
        )
        metadata = {
            block.block_hash: block
            for block in cluster.kv.blocks_for_session(
                request.session_id,
                request.prefix_id,
                model.name,
            )
        }
        prepared_bytes = 0
        prepared_blocks = 0
        for block_hash in hashes:
            block = metadata.get(block_hash)
            if block is None:
                break
            if request.entry_node in cluster.kv.locate(block_hash):
                continue
            cluster.kv.register(request.entry_node, block)
            prepared_bytes += block.size_bytes
            prepared_blocks += 1
        return prepared_bytes, prepared_blocks

    for r in reqs:
        completed_before_request = kv_manager.advance(r.arrival_ms)
        for n in cluster.nodes.values():
            n.advance_to(r.arrival_ms, prev_t)
        prev_t = r.arrival_ms
        cluster.refresh(r.arrival_ms, force=True)

        # Optimistic admission lower bound: complete historical KV is assumed
        # to be locally ready at the ingress, with zero queueing, forwarding,
        # KV recovery, and return-network delay.  The incremental prefill for
        # the current input remains irreducible for TTFT.
        ideal_ttft_ms = cluster.node(r.entry_node).compute.estimate_incremental_prefill(
            r.prefix_tokens, r.input_tokens
        ).prefill_ms
        if (
            reject_intrinsically_infeasible
            and ideal_ttft_ms > r.sla_ms
        ):
            admission_rejected += 1
            admission_rejected_by_group[r.group_name] = (
                admission_rejected_by_group.get(r.group_name, 0) + 1
            )
            admission_rejected_by_priority[r.priority] = (
                admission_rejected_by_priority.get(r.priority, 0) + 1
            )
            rejection_records.append({
                "t": round(r.arrival_ms, 2),
                "request_id": r.request_id,
                "session_id": r.session_id,
                "turn_index": r.turn_index,
                "group_name": r.group_name,
                "priority": r.priority,
                "sla_ms": r.sla_ms,
                "ideal_ttft_lower_bound_ms": round(ideal_ttft_ms, 3),
                "prefix_tokens": r.prefix_tokens,
                "input_tokens": r.input_tokens,
                "reason": "ideal_ttft_lower_bound_exceeds_sla",
            })
            continue

        if policy == Policy.ORACLE_KV:
            prepared_bytes, prepared_blocks = prepare_oracle_entry(r)
            oracle_prepared_bytes += prepared_bytes
            oracle_prepared_blocks += prepared_blocks
            oracle_prepared_tasks += int(prepared_blocks > 0)

        router = routers[r.entry_node]
        decision = router.route(r)
        accepted_requests.append(r)
        is_infeasible = not decision.feasible
        is_sla = decision.ttft_ms + sla_margin_ms > r.sla_ms
        is_cross = decision.action.exec_node != r.entry_node
        if is_infeasible:
            infeasible += 1
        if is_sla:
            sla_viol += 1
        if is_cross:
            cross_node += 1
        if decision.future_driven:
            future_driven_action_count += 1
        region = decision.breakpoint_region
        breakpoint_region_counts[region] = (
            breakpoint_region_counts.get(region, 0) + 1
        )
        if region != "no_history":
            breakpoint_ready_ratios.append(decision.entry_ready_ratio)
        if region not in ("no_history", "already_ready", "no_remote_owner"):
            breakpoint_residual_bytes.append(decision.entry_residual_bytes)
            breakpoint_recovery_ms.append(decision.entry_recovery_ms)
            if decision.entry_sync_ms > 0.0:
                breakpoint_sync_ms.append(decision.entry_sync_ms)
            if decision.entry_recompute_ms > 0.0:
                breakpoint_recompute_ms.append(decision.entry_recompute_ms)
            breakpoint_recovery_mode_counts[decision.entry_recovery_mode] = (
                breakpoint_recovery_mode_counts.get(
                    decision.entry_recovery_mode, 0
                ) + 1
            )
            if decision.forwarding_premium_ms > 0.0:
                breakpoint_forwarding_premiums.append(
                    decision.forwarding_premium_ms
                )
            if decision.breakpoint_turns is not None:
                breakpoint_turn_values.append(decision.breakpoint_turns)
        if (
            region == "long_term_gap"
            and decision.action.exec_node == r.entry_node
        ):
            breakpoint_gap_entry_choices += 1
        mode_counts[decision.action.mode.value] += 1
        node_queue = queue_by_node[decision.action.exec_node]
        node_queue["requests"] += 1
        node_queue["queue_ms"] += decision.t_queue_ms
        node_queue["queue_prefill_ms"] += decision.t_queue_prefill_ms
        node_queue["queue_recompute_ms"] += decision.t_queue_recompute_ms
        node_queue["queue_decode_ms"] += decision.t_queue_decode_ms

        migrate_bytes_before = cluster.kv.stats["migrate_bytes"]
        router.commit(r, decision, r.arrival_ms)
        placement = kv_manager.schedule_after_request(
            r, decision.action.exec_node, r.arrival_ms
        )
        placement_scheduled_bytes += placement.scheduled_bytes
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
                "future_driven": int(decision.future_driven),
                "entry_ready_ratio": round(decision.entry_ready_ratio, 6),
                "entry_residual_bytes": int(decision.entry_residual_bytes),
                "entry_recovery_ms": round(decision.entry_recovery_ms, 6),
                "entry_sync_ms": round(decision.entry_sync_ms, 6),
                "entry_recompute_ms": round(decision.entry_recompute_ms, 6),
                "entry_recovery_mode": decision.entry_recovery_mode,
                "forwarding_premium_ms": round(
                    decision.forwarding_premium_ms, 6
                ),
                "kv_owner_queue_ms": round(decision.kv_owner_queue_ms, 6),
                "ideal_entry_queue_ms": round(decision.ideal_entry_queue_ms, 6),
                "kv_owner_e2e_ms": round(decision.kv_owner_e2e_ms, 6),
                "ideal_entry_e2e_ms": round(decision.ideal_entry_e2e_ms, 6),
                "complete_kv_owner_queue_ms": round(
                    decision.complete_kv_owner_queue_ms, 6
                ),
                "complete_kv_owner_e2e_ms": round(
                    decision.complete_kv_owner_e2e_ms, 6
                ),
                "breakpoint_turns": (
                    round(decision.breakpoint_turns, 6)
                    if decision.breakpoint_turns is not None
                    else None
                ),
                "remaining_turns": decision.remaining_turns,
                "remaining_entry_turns": decision.remaining_entry_turns,
                "breakpoint_region": decision.breakpoint_region,
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
                "placement_scheduled_bytes": placement.scheduled_bytes,
                "placement_completed_bytes": completed_before_request,
            })

    def pct(v, p):
        if not v:
            return 0.0
        s = sorted(v)
        return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]

    accepted_n = len(accepted_requests)
    offered_n = len(reqs)
    n = max(accepted_n, 1)
    avg_components = {
        f"avg_{name}_ms": total / n
        for name, total in component_totals.items()
    }
    result = {
        "policy": policy.value,
        "model": model.name,
        "block_level_kv_enabled": block_level_kv,
        "proactive_kv_enabled": (
            manager_config.enabled or policy == Policy.ORACLE_KV
        ),
        "oracle_prefetch": policy == Policy.ORACLE_PREFETCH,
        "oracle_kv_placement": policy == Policy.ORACLE_KV,
        "num_requests": accepted_n,
        "offered_requests": offered_n,
        "accepted_requests": accepted_n,
        "admission_rejected_count": admission_rejected,
        "admission_rejected_ratio": admission_rejected / max(offered_n, 1),
        "admission_rejected_by_group": admission_rejected_by_group,
        "admission_rejected_by_priority": admission_rejected_by_priority,
        "avg_e2e_ms": sum(e2e_list) / n,
        "avg_predicted_future_cost_ms": sum(future_cost_list) / n,
        "avg_q_value_ms": sum(q_value_list) / n,
        "future_driven_action_count": future_driven_action_count,
        "future_driven_action_ratio": future_driven_action_count / n,
        "breakpoint_region_counts": breakpoint_region_counts,
        "avg_entry_ready_ratio": (
            sum(breakpoint_ready_ratios) / len(breakpoint_ready_ratios)
            if breakpoint_ready_ratios else 0.0
        ),
        "avg_entry_residual_mb": (
            sum(breakpoint_residual_bytes)
            / max(len(breakpoint_residual_bytes), 1)
            / 1e6
        ),
        "avg_entry_recovery_ms": (
            sum(breakpoint_recovery_ms)
            / max(len(breakpoint_recovery_ms), 1)
        ),
        "avg_entry_sync_ms": (
            sum(breakpoint_sync_ms) / max(len(breakpoint_sync_ms), 1)
        ),
        "avg_entry_recompute_ms": (
            sum(breakpoint_recompute_ms)
            / max(len(breakpoint_recompute_ms), 1)
        ),
        "entry_recovery_mode_counts": breakpoint_recovery_mode_counts,
        "avg_forwarding_premium_ms": (
            sum(breakpoint_forwarding_premiums)
            / max(len(breakpoint_forwarding_premiums), 1)
        ),
        "avg_breakpoint_turns": (
            sum(breakpoint_turn_values)
            / max(len(breakpoint_turn_values), 1)
        ),
        "breakpoint_eligible_count": sum(
            breakpoint_region_counts.get(region, 0)
            for region in (
                "greedy_switch", "long_term_gap", "not_amortizable"
            )
        ),
        "breakpoint_gap_ratio": (
            breakpoint_region_counts.get("long_term_gap", 0)
            / max(
                sum(
                    breakpoint_region_counts.get(region, 0)
                    for region in (
                        "greedy_switch", "long_term_gap", "not_amortizable"
                    )
                ),
                1,
            )
        ),
        "breakpoint_gap_entry_choice_ratio": (
            breakpoint_gap_entry_choices
            / max(breakpoint_region_counts.get("long_term_gap", 0), 1)
        ),
        "p95_e2e_ms": pct(e2e_list, 95),
        "p50_ttft_ms": pct(ttft_list, 50),
        "p95_ttft_ms": pct(ttft_list, 95),
        "p99_ttft_ms": pct(ttft_list, 99),
        "sla_violation_ratio": sla_viol / n,
        "infeasible_ratio": infeasible / n,
        "cross_node_ratio": cross_node / n,
        "mobility_transition_count": sum(
            int(r.mobility_transitioned) for r in accepted_requests
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
        "placement_scheduled_bytes_mb": (
            placement_scheduled_bytes + oracle_prepared_bytes
        ) / 1e6,
        "placement_completed_bytes_mb": (
            kv_manager.stats["completed_bytes"] + oracle_prepared_bytes
        ) / 1e6,
        "oracle_prepared_bytes_mb": oracle_prepared_bytes / 1e6,
        "placement_pending_bytes_mb": kv_manager.pending_bytes() / 1e6,
        "placement_scheduled_tasks": int(
            kv_manager.stats["scheduled_tasks"] + oracle_prepared_tasks
        ),
        "placement_completed_tasks": int(
            kv_manager.stats["completed_tasks"] + oracle_prepared_tasks
        ),
        "placement_completed_blocks": int(
            kv_manager.stats["completed_blocks"] + oracle_prepared_blocks
        ),
        "placement_activity_skips": int(
            kv_manager.stats["skipped_no_continuation"]
        ),
        "offload_cost_ms": offload_cost_ms,
        **avg_components,
    }
    result["conditional_migration_ms"] = (
        component_totals["migration"] / max(mode_counts["migrate"], 1)
    )
    result["conditional_recompute_ms"] = (
        component_totals["recompute"] / max(mode_counts["recompute"], 1)
    )
    result["avg_state_ms"] = (
        result["avg_migration_ms"] + result["avg_recompute_ms"]
    )
    result["total_kv_transfer_bytes_mb"] = (
        result["migrate_bytes_mb"] + result["placement_completed_bytes_mb"]
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
        result["rejection_records"] = rejection_records
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
