"""Physical derivation of block size, operation rates and HBM capacity.

Every rate the control loop uses is derived here from a model shape and a
hardware sheet instead of being stated directly in blocks per second. The
purpose is not absolute fidelity: the numbers below are a mock-up of a 7B-class
dense decoder on one 910B-class device and are not measured. The purpose is
that the *ratios* between recompute, remote transfer, host restore, prefill and
decode follow from bytes and FLOPs rather than from three hand-picked
constants, because those ratios are what every preparation and reclamation
decision turns on.

Two consequences of the physical form are worth stating up front, because they
differ sharply from a hand-set rate sheet:

- Recompute is far more expensive than moving the same KV. Rebuilding one token
  of KV costs a full forward pass over the model, while moving it costs one
  KV-cache read. At the mock numbers below recompute is roughly ten times
  slower than a remote transfer and sixteen times slower than a host restore.
- Decode, not prefill, dominates how long a request occupies a device. Decoding
  one token requires reading every weight, so per-sequence decode throughput is
  set by HBM bandwidth divided by weight bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class ModelSpec:
    """Shape of the served model and of one KV block.

    Defaults describe a 7B-class dense decoder with grouped-query attention in
    bfloat16, served as a single replica. `block_tokens` is the KV paging
    granularity, matching the block size a paged-attention runtime would use.
    """

    layers: int = 80
    kv_heads: int = 8
    head_dim: int = 128
    parameters: float = 70.0e9
    dtype_bytes: int = 2
    block_tokens: int = 16

    @cached_property
    def kv_bytes_per_token(self) -> int:
        """Keys and values for every layer of one token."""
        return self.layers * 2 * self.kv_heads * self.head_dim * self.dtype_bytes

    @cached_property
    def block_bytes(self) -> int:
        return self.block_tokens * self.kv_bytes_per_token

    @cached_property
    def weight_bytes(self) -> float:
        return self.parameters * self.dtype_bytes

    @cached_property
    def prefill_flops_per_token(self) -> float:
        """Multiply-accumulate cost of one forward pass, counted as 2 FLOPs per
        parameter. Attention over the context is not included, so long-context
        prefill is optimistic here."""
        return 2.0 * self.parameters


@dataclass(frozen=True)
class HardwareSpec:
    """Per-node device and interconnect capacities.

    Defaults are a mock-up of one 910B-class accelerator: peak bfloat16 dense
    throughput with a separate achieved-utilisation factor, HBM bandwidth and
    capacity, one 100 GbE port for inter-node KV movement, and a PCIe 4.0 x16
    path to host DRAM.
    """

    devices: int = 8
    device_flops: float = 3.2e14
    achieved_utilisation: float = 0.42
    hbm_bytes_s: float = 1.6e12
    hbm_capacity_bytes: float = 64 * 2**30
    network_bytes_s: float = 12.5e9
    host_link_bytes_s: float = 20.0e9
    host_capacity_bytes: float = 1.5 * 2**40
    decode_batch: int = 16
    decode_context_tokens: int = 30000

    @cached_property
    def effective_flops(self) -> float:
        """Aggregate compute of the tensor-parallel group serving one replica."""
        return self.devices * self.device_flops * self.achieved_utilisation

    @cached_property
    def aggregate_hbm_bytes_s(self) -> float:
        return self.devices * self.hbm_bytes_s

    @cached_property
    def aggregate_hbm_capacity_bytes(self) -> float:
        return self.devices * self.hbm_capacity_bytes


@dataclass(frozen=True)
class ResourceModel:
    """Rates in blocks per second, derived from a model and a hardware sheet."""

    model: ModelSpec = ModelSpec()
    hardware: HardwareSpec = HardwareSpec()

    @cached_property
    def prefill_blocks_s(self) -> float:
        tokens_s = self.hardware.effective_flops / self.model.prefill_flops_per_token
        return tokens_s / self.model.block_tokens

    @cached_property
    def recompute_blocks_s(self) -> float:
        """Rebuilding KV is a prefill over the same tokens."""
        return self.prefill_blocks_s

    @cached_property
    def decode_slots(self) -> int:
        """Sequences a node can decode concurrently.

        The configured batch size is an upper bound, but KV memory is the real
        one: every sequence in the batch keeps its whole context resident, so a
        node cannot run more concurrent sequences than its HBM holds contexts.
        At long context this bound is what limits a node, not the batch setting.
        """
        context_blocks = max(1, self.hardware.decode_context_tokens // self.model.block_tokens)
        by_memory = max(1, self.hbm_blocks // context_blocks)
        return min(self.hardware.decode_batch, by_memory)

    @cached_property
    def decode_step_seconds(self) -> float:
        """Wall time of one decode step of a full batch.

        A step reads every weight once regardless of batch size, plus the KV of
        each sequence in the batch. Serving one sequence at a time would spend
        almost all HBM bandwidth on weight reads, which is why engines batch.
        The step time is taken at a stated operating point: `decode_slots`
        sequences of `decode_context_tokens` each.
        """
        hardware = self.hardware
        kv_bytes = self.decode_slots * self.model.kv_bytes_per_token * hardware.decode_context_tokens
        step_bytes = self.model.weight_bytes + kv_bytes
        return step_bytes / hardware.aggregate_hbm_bytes_s

    @cached_property
    def decode_blocks_s(self) -> float:
        """Per-sequence decode rate. Every sequence in the batch advances one
        token per step, so this is one token per step time."""
        return 1.0 / (self.decode_step_seconds * self.model.block_tokens)

    @cached_property
    def transfer_blocks_s(self) -> float:
        return self.hardware.network_bytes_s / self.model.block_bytes

    @cached_property
    def restore_blocks_s(self) -> float:
        return self.hardware.host_link_bytes_s / self.model.block_bytes

    @cached_property
    def hbm_blocks(self) -> int:
        """KV blocks that fit beside the weights, leaving room for activations.

        The weights are sharded once across the group, so they are subtracted
        once from the aggregate capacity rather than per device.
        """
        capacity = self.hardware.aggregate_hbm_capacity_bytes
        spare = capacity - self.model.weight_bytes - 0.08 * capacity
        return max(1, int(spare // self.model.block_bytes))

    @cached_property
    def host_blocks(self) -> int:
        """KV blocks the node's DRAM tier can hold.

        Finite, and that matters: if the host tier never forgot, a restore
        would always be available on the node that last served a session and no
        recovery would ever need recomputation, which would make KV placement
        irrelevant by construction.
        """
        return max(1, int(self.hardware.host_capacity_bytes // self.model.block_bytes))

    @cached_property
    def rates(self) -> dict[str, float]:
        """Rate of each recovery method, keyed as the planner expects."""
        return {
            "transfer": self.transfer_blocks_s,
            "restore": self.restore_blocks_s,
            "recompute": self.recompute_blocks_s,
        }

    @cached_property
    def resource_of_method(self) -> dict[str, str]:
        """Which serial resource each recovery method occupies."""
        return {"transfer": "link", "restore": "host", "recompute": "compute"}

    def describe(self) -> dict[str, float]:
        return {
            "kv_bytes_per_token": self.model.kv_bytes_per_token,
            "block_bytes": self.model.block_bytes,
            "weight_bytes": self.model.weight_bytes,
            "prefill_blocks_s": self.prefill_blocks_s,
            "decode_blocks_s": self.decode_blocks_s,
            "transfer_blocks_s": self.transfer_blocks_s,
            "restore_blocks_s": self.restore_blocks_s,
            "recompute_blocks_s": self.recompute_blocks_s,
            "hbm_blocks": self.hbm_blocks,
        }
