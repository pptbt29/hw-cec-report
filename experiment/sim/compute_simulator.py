"""Roofline-based GPU/NPU compute simulator.

Estimates prefill / decode time, memory footprint and KV transfer cost for a
given model on a given hardware node. Used by the discrete-event scheduler to
predict TTFT, end-to-end latency and the local/migrate/recompute action costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .large_model import ModelSpec


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    num_devices: int
    peak_flops_per_device: float       # BF16 FLOPS
    mem_bandwidth_per_device: float    # bytes/s
    mem_capacity_per_device: float     # bytes
    compute_efficiency: float = 0.5    # MFU for prefill
    bandwidth_efficiency: float = 0.7  # achievable HBM bandwidth fraction
    interconnect_bandwidth: float = 400e9  # intra-node link, bytes/s
    fixed_overhead_ms: float = 0.2

    def effective_compute(self) -> float:
        return self.num_devices * self.peak_flops_per_device * self.compute_efficiency

    def effective_bandwidth(self) -> float:
        return (
            self.num_devices
            * self.mem_bandwidth_per_device
            * self.bandwidth_efficiency
        )

    def total_memory(self) -> float:
        return self.num_devices * self.mem_capacity_per_device


@dataclass
class PrefillResult:
    prefill_ms: float
    flops: float
    bytes: float
    bound: str  # "compute" or "memory"


@dataclass
class DecodeResult:
    step_ms: float
    total_ms: float
    throughput_tokens_per_s: float
    bound: str


class ComputeSimulator:
    def __init__(self, model: ModelSpec, hw: HardwareSpec):
        self.model = model
        self.hw = hw

    def estimate_prefill(
        self, prompt_tokens: int, batch_tokens: Optional[int] = None
    ) -> PrefillResult:
        """Time to prefill ``prompt_tokens`` for one request.

        ``batch_tokens`` (>= prompt_tokens) amortises weight reads across a
        batched prefill; defaults to prompt_tokens (single request).
        """
        prompt_tokens = max(int(prompt_tokens), 0)
        if prompt_tokens == 0:
            return PrefillResult(self.hw.fixed_overhead_ms, 0.0, 0.0, "compute")
        batch_tokens = max(batch_tokens or prompt_tokens, prompt_tokens)

        flops = self.model.prefill_flops(prompt_tokens)
        t_comp = flops / self.hw.effective_compute()

        bytes_read = self.model.total_weight_bytes() + self.model.kv_bytes_for_tokens(
            prompt_tokens
        )
        t_mem = bytes_read / self.hw.effective_bandwidth()

        bound = "compute" if t_comp >= t_mem else "memory"
        ms = max(t_comp, t_mem) * 1000.0 + self.hw.fixed_overhead_ms
        return PrefillResult(ms, flops, float(bytes_read), bound)

    def estimate_prefill_service(
        self,
        prompt_tokens: int,
        batch_size: int = 1,
    ) -> PrefillResult:
        """Equivalent per-request queue work under ideal prefill batching.

        Batching does not remove the request's FLOPs.  It only amortises the
        model-weight read and fixed launch overhead across requests in the
        same batch.  The returned time is therefore device service demand,
        not the request's standalone prefill latency.
        """
        prompt_tokens = max(int(prompt_tokens), 0)
        batch_size = max(int(batch_size), 1)
        if prompt_tokens == 0:
            return PrefillResult(
                self.hw.fixed_overhead_ms / batch_size,
                0.0,
                0.0,
                "compute",
            )
        flops = self.model.prefill_flops(prompt_tokens)
        t_comp = flops / self.hw.effective_compute()
        bytes_read = (
            self.model.total_weight_bytes() / batch_size
            + self.model.kv_bytes_for_tokens(prompt_tokens)
        )
        t_mem = bytes_read / self.hw.effective_bandwidth()
        bound = "compute" if t_comp >= t_mem else "memory"
        ms = (
            max(t_comp, t_mem) * 1000.0
            + self.hw.fixed_overhead_ms / batch_size
        )
        return PrefillResult(ms, flops, float(bytes_read), bound)

    def estimate_decode(
        self, gen_tokens: int, ctx_len: int, batch_size: int = 1
    ) -> DecodeResult:
        """Time to generate ``gen_tokens`` tokens given starting context length.

        Context length grows by one each step; cost is integrated step by step.
        """
        gen_tokens = max(int(gen_tokens), 0)
        ctx_len = max(int(ctx_len), 1)
        batch_size = max(int(batch_size), 1)
        if gen_tokens == 0:
            return DecodeResult(0.0, 0.0, 0.0, "memory")

        # Both compute and memory time are affine in L=ctx_len+g.  Their
        # maximum therefore changes branch at most once, so the exact sum can
        # be evaluated in O(1) instead of looping over every generated token.
        compute_bw = self.hw.effective_compute()
        memory_bw = self.hw.effective_bandwidth()
        comp_a = batch_size * 2.0 * self.model.num_params / compute_bw
        comp_b = (
            batch_size * 4.0 * self.model.num_layers * self.model.hidden_size
            / compute_bw
        )
        # Vision-encoder weights are consumed while encoding the visual input,
        # not reread by every autoregressive decoder step.
        mem_a = self.model.decoder_weight_bytes() / memory_bw
        mem_b = batch_size * self.model.kv_bytes_per_token() / memory_bw

        d0 = (comp_a - mem_a) + (comp_b - mem_b) * ctx_len
        d_step = comp_b - mem_b

        def affine_sum(a: float, b: float, start: int, end: int) -> float:
            if end < start:
                return 0.0
            count = end - start + 1
            return count * a + b * (start + end) * count / 2.0

        # Sum memory time for all steps, then add only the positive part of
        # compute-minus-memory.  Strict positivity preserves the old
        # ``t_mem >= t_comp`` tie classification.
        memory_sum = affine_sum(
            mem_a + mem_b * ctx_len, mem_b, 0, gen_tokens - 1
        )
        positive_start = gen_tokens
        positive_end = -1
        if abs(d_step) < 1e-30:
            if d0 > 0.0:
                positive_start, positive_end = 0, gen_tokens - 1
        elif d_step > 0.0:
            positive_start = max(0, int(math.floor(-d0 / d_step)) + 1)
            positive_end = gen_tokens - 1
        else:
            positive_start = 0
            positive_end = min(
                gen_tokens - 1,
                int(math.ceil(d0 / (-d_step))) - 1,
            )
        positive_start = min(max(positive_start, 0), gen_tokens)
        positive_end = min(max(positive_end, -1), gen_tokens - 1)
        compute_dominant_steps = max(positive_end - positive_start + 1, 0)
        excess_compute = affine_sum(
            d0, d_step, positive_start, positive_end
        )
        total_ms = (
            (memory_sum + excess_compute) * 1000.0
            + gen_tokens * self.hw.fixed_overhead_ms
        )

        last_L = ctx_len + gen_tokens - 1
        last_comp = comp_a + comp_b * last_L
        last_mem = mem_a + mem_b * last_L
        last_step_ms = max(last_comp, last_mem) * 1000.0 + self.hw.fixed_overhead_ms
        mem_steps = gen_tokens - compute_dominant_steps

        bound = "memory" if mem_steps >= gen_tokens / 2 else "compute"
        # throughput counts all sequences in the batch
        throughput = (
            batch_size * gen_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0
        )
        return DecodeResult(last_step_ms, total_ms, throughput, bound)

    def estimate_amortized_decode(
        self, gen_tokens: int, ctx_len: int, batch_size: int = 32
    ) -> DecodeResult:
        """Average per-request decode service time under a fixed full batch.

        This is a lightweight throughput-equivalent estimate: the time of a
        homogeneous batch iteration is divided equally among its sequences.
        It intentionally does not model dynamic continuous batching.
        """
        batch_size = max(int(batch_size), 1)
        batched = self.estimate_decode(gen_tokens, ctx_len, batch_size)
        return DecodeResult(
            step_ms=batched.step_ms / batch_size,
            total_ms=batched.total_ms / batch_size,
            throughput_tokens_per_s=batched.throughput_tokens_per_s,
            bound=batched.bound,
        )

    def kv_cache_bytes(self, num_tokens: int) -> int:
        return self.model.kv_bytes_for_tokens(num_tokens)

    def weight_bytes(self) -> int:
        return self.model.total_weight_bytes()

    def memory_usage(
        self,
        resident_tokens: int,
        batch_tokens: int = 0,
        mem_reserve: float = 0.0,
        activation_factor: float = 2.0,
    ) -> Dict[str, float]:
        """Predict device memory state given resident KV tokens and active batch."""
        mem_weights = self.model.total_weight_bytes()
        mem_kv = self.model.kv_bytes_for_tokens(resident_tokens)
        mem_act = (
            activation_factor
            * max(batch_tokens, 0)
            * self.model.hidden_size
            * self.model.dtype_bytes
        )
        mem_used = mem_weights + mem_kv + mem_act
        total = self.hw.total_memory()
        return {
            "weights": float(mem_weights),
            "kv": float(mem_kv),
            "activation": float(mem_act),
            "used": float(mem_used),
            "total": float(total),
            "free": float(total - mem_used - mem_reserve),
        }

    def kv_transfer_time_ms(
        self, num_tokens: int, link_bps: float, latency_ms: float = 0.0
    ) -> float:
        """Time to move KV for ``num_tokens`` over a link (migrate action)."""
        if num_tokens <= 0 or link_bps <= 0:
            return latency_ms
        bytes_to_move = self.model.kv_bytes_for_tokens(num_tokens)
        return bytes_to_move / link_bps * 1000.0 + latency_ms

    def recompute_time_ms(self, prefix_tokens: int) -> float:
        """Time to rebuild KV when no historical prefix is reusable."""
        return self.estimate_incremental_prefill(0, prefix_tokens).prefill_ms

    def estimate_incremental_prefill(
        self,
        cached_prefix_tokens: int,
        new_tokens: int,
    ) -> PrefillResult:
        """Prefill only a missing suffix while attending to cached prefix KV.

        The suffix does not repeat the prefix's transformer work, but its
        attention still observes that prefix.  Consequently its FLOPs are the
        difference between full-prefix FLOPs before and after the suffix, not
        simply ``prefill_flops(new_tokens)``.
        """
        cached = max(int(cached_prefix_tokens), 0)
        new = max(int(new_tokens), 0)
        if new == 0:
            return PrefillResult(self.hw.fixed_overhead_ms, 0.0, 0.0, "compute")
        total = cached + new
        flops = max(
            self.model.prefill_flops(total)
            - self.model.prefill_flops(cached),
            0.0,
        )
        t_comp = flops / self.hw.effective_compute()
        bytes_read = (
            self.model.total_weight_bytes()
            + self.model.kv_bytes_for_tokens(total)
        )
        t_mem = bytes_read / self.hw.effective_bandwidth()
        bound = "compute" if t_comp >= t_mem else "memory"
        ms = max(t_comp, t_mem) * 1000.0 + self.hw.fixed_overhead_ms
        return PrefillResult(ms, flops, float(bytes_read), bound)

    def estimate_incremental_prefill_service(
        self,
        cached_prefix_tokens: int,
        new_tokens: int,
        batch_size: int = 1,
    ) -> PrefillResult:
        """Queue service demand for incremental suffix prefill."""
        cached = max(int(cached_prefix_tokens), 0)
        new = max(int(new_tokens), 0)
        batch_size = max(int(batch_size), 1)
        if new == 0:
            return PrefillResult(
                self.hw.fixed_overhead_ms / batch_size,
                0.0,
                0.0,
                "compute",
            )
        total = cached + new
        flops = max(
            self.model.prefill_flops(total)
            - self.model.prefill_flops(cached),
            0.0,
        )
        t_comp = flops / self.hw.effective_compute()
        bytes_read = (
            self.model.total_weight_bytes() / batch_size
            + self.model.kv_bytes_for_tokens(total)
        )
        t_mem = bytes_read / self.hw.effective_bandwidth()
        bound = "compute" if t_comp >= t_mem else "memory"
        ms = (
            max(t_comp, t_mem) * 1000.0
            + self.hw.fixed_overhead_ms / batch_size
        )
        return PrefillResult(ms, flops, float(bytes_read), bound)

    def estimate_prefill_batch(
        self, prompt_tokens_list: Iterable[int], max_batch_tokens: Optional[int] = None
    ) -> PrefillResult:
        tokens = [max(int(t), 0) for t in prompt_tokens_list]
        total = sum(tokens)
        if max_batch_tokens is not None:
            total = min(total, max_batch_tokens)
        flops = sum(self.model.prefill_flops(t) for t in tokens)
        t_comp = flops / self.hw.effective_compute()
        bytes_read = self.model.total_weight_bytes() + self.model.kv_bytes_for_tokens(
            total
        )
        t_mem = bytes_read / self.hw.effective_bandwidth()
        bound = "compute" if t_comp >= t_mem else "memory"
        ms = max(t_comp, t_mem) * 1000.0 + self.hw.fixed_overhead_ms
        return PrefillResult(ms, flops, float(bytes_read), bound)


_A800T_A2 = HardwareSpec(
    name="A800T-A2",
    num_devices=8,
    peak_flops_per_device=376e12,
    mem_bandwidth_per_device=1.6e12,
    mem_capacity_per_device=64e9,
    compute_efficiency=0.4,
    bandwidth_efficiency=0.6,
    interconnect_bandwidth=400e9,
    fixed_overhead_ms=0.2,
)


HARDWARE_REGISTRY: Dict[str, HardwareSpec] = {_A800T_A2.name: _A800T_A2}


def get_hardware(name: str) -> HardwareSpec:
    if name not in HARDWARE_REGISTRY:
        raise KeyError(
            f"unknown hardware {name!r}; available: {sorted(HARDWARE_REGISTRY)}"
        )
    return HARDWARE_REGISTRY[name]


if __name__ == "__main__":
    from .large_model import list_models, get_model

    hw = get_hardware("A800T-A2")
    print(
        f"hardware={hw.name}  compute={hw.effective_compute()/1e12:.0f} TFLOP/s  "
        f"bw={hw.effective_bandwidth()/1e12:.2f} TB/s  mem={hw.total_memory()/1e9:.0f} GB"
    )
    for name in list_models():
        sim = ComputeSimulator(get_model(name), hw)
        pf = sim.estimate_prefill(1024)
        dec = sim.estimate_decode(gen_tokens=128, ctx_len=1024, batch_size=8)
        print(
            f"{name:<24} prefill(1024)={pf.prefill_ms:7.2f} ms [{pf.bound}]  "
            f"decode(128tok,b8)={dec.total_ms:8.2f} ms [{dec.bound}]  "
            f"thrpt={dec.throughput_tokens_per_s:8.0f} tok/s"
        )
