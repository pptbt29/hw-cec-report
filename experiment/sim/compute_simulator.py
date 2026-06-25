"""Roofline-based GPU/NPU compute simulator.

Estimates prefill / decode time, memory footprint and KV transfer cost for a
given model on a given hardware node. Used by the discrete-event scheduler to
predict TTFT, end-to-end latency and the local/migrate/recompute action costs.
"""

from __future__ import annotations

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

        total_ms = 0.0
        last_step_ms = 0.0
        mem_steps = 0
        for g in range(gen_tokens):
            L = ctx_len + g
            flops = batch_size * self.model.decode_flops_per_token(L)
            t_comp = flops / self.hw.effective_compute()
            bytes_read = (
                self.model.total_weight_bytes()
                + batch_size * self.model.kv_bytes_per_token() * L
            )
            t_mem = bytes_read / self.hw.effective_bandwidth()
            if t_mem >= t_comp:
                mem_steps += 1
            step_ms = max(t_comp, t_mem) * 1000.0 + self.hw.fixed_overhead_ms
            total_ms += step_ms
            last_step_ms = step_ms

        bound = "memory" if mem_steps >= gen_tokens / 2 else "compute"
        # throughput counts all sequences in the batch
        throughput = (
            batch_size * gen_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0
        )
        return DecodeResult(last_step_ms, total_ms, throughput, bound)

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
        """Time to rebuild KV by re-prefilling a reusable prefix (recompute action)."""
        return self.estimate_prefill(prefix_tokens).prefill_ms

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
    compute_efficiency=0.5,
    bandwidth_efficiency=0.7,
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
