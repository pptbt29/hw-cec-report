"""Large-model specifications shared by the compute simulator, data generator,
network and KV cache modules.

This module is the single source of truth for model architecture, derived
quantities (KV cache size, prefill/decode FLOPs) and modality input composition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional


class ModelType(str, Enum):
    LLM = "LLM"
    VLM = "VLM"
    VLA = "VLA"


@dataclass(frozen=True)
class LengthDistributionSpec:
    """Declarative default output-length distribution for a model.

    The actual sampling lives in the data generator; this only records the
    distribution family and parameters so a model can advertise sane defaults.
    """

    kind: str = "lognormal"  # one of: fixed, normal, lognormal
    mean: float = 128.0
    std: float = 64.0
    minimum: int = 1
    maximum: int = 4096


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: ModelType

    # transformer backbone
    num_params: float
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    dtype_bytes: int = 2  # bf16/fp16

    # explicit weight footprint; defaults to num_params * dtype_bytes
    weight_bytes: Optional[int] = None

    # vision encoder (VLM / VLA)
    vision_params: float = 0.0
    patch_size: int = 14
    spatial_merge: int = 1
    tokens_per_image: Optional[int] = None  # fixed token count (VLA style)

    # business metadata
    default_output_dist: LengthDistributionSpec = field(
        default_factory=LengthDistributionSpec
    )
    default_sla_ms: float = 500.0
    kv_block_size: int = 16  # tokens per KV block

    def total_weight_bytes(self) -> int:
        if self.weight_bytes is not None:
            return int(self.weight_bytes)
        return int((self.num_params + self.vision_params) * self.dtype_bytes)

    def kv_bytes_per_token(self) -> int:
        """K and V across all layers for a single token."""
        return int(
            2 * self.num_kv_heads * self.head_dim * self.dtype_bytes * self.num_layers
        )

    def prefill_flops(self, num_tokens: int) -> float:
        """Forward FLOPs to process ``num_tokens`` prompt tokens at once."""
        if num_tokens <= 0:
            return 0.0
        flops_linear = 2.0 * self.num_params * num_tokens
        flops_attention = 4.0 * self.num_layers * (num_tokens ** 2) * self.hidden_size
        return flops_linear + flops_attention

    def decode_flops_per_token(self, ctx_len: int) -> float:
        """FLOPs to generate one token given current context length."""
        ctx_len = max(ctx_len, 1)
        flops_linear = 2.0 * self.num_params
        flops_attention = 4.0 * self.num_layers * ctx_len * self.hidden_size
        return flops_linear + flops_attention

    def decode_bytes_per_token(self, ctx_len: int) -> int:
        """Bytes read to generate one token (weights + KV of full context)."""
        ctx_len = max(ctx_len, 1)
        return self.total_weight_bytes() + self.kv_bytes_per_token() * ctx_len

    def visual_tokens(self, width: int = 0, height: int = 0, num_frames: int = 1) -> int:
        """Number of visual tokens contributed by image/video input."""
        if self.model_type == ModelType.LLM:
            return 0
        if self.tokens_per_image is not None:
            return int(self.tokens_per_image) * max(num_frames, 1)
        effective_patch = self.patch_size * max(self.spatial_merge, 1)
        if width <= 0 or height <= 0 or effective_patch <= 0:
            return 0
        cols = math.ceil(width / effective_patch)
        rows = math.ceil(height / effective_patch)
        return cols * rows * max(num_frames, 1)

    def input_tokens(
        self,
        text_tokens: int,
        image_width: int = 0,
        image_height: int = 0,
        num_frames: int = 1,
        state_tokens: int = 0,
    ) -> int:
        """Total prompt token count combining all modalities."""
        visual = self.visual_tokens(image_width, image_height, num_frames)
        return int(text_tokens) + int(visual) + int(state_tokens)

    def kv_blocks_for_tokens(self, num_tokens: int) -> int:
        return math.ceil(max(num_tokens, 0) / self.kv_block_size)

    def kv_bytes_for_tokens(self, num_tokens: int) -> int:
        """KV footprint rounded up to whole blocks (matches KV manager granularity)."""
        blocks = self.kv_blocks_for_tokens(num_tokens)
        return blocks * self.kv_block_size * self.kv_bytes_per_token()

    def with_overrides(self, **kwargs) -> "ModelSpec":
        return replace(self, **kwargs)


_CODELLAMA_34B = ModelSpec(
    name="CodeLlama34B",
    model_type=ModelType.LLM,
    num_params=33.7e9,
    num_layers=48,
    hidden_size=8192,
    num_attention_heads=64,
    num_kv_heads=8,
    head_dim=128,
    intermediate_size=22016,
    vocab_size=32016,
    default_output_dist=LengthDistributionSpec(
        kind="lognormal", mean=256.0, std=200.0, minimum=8, maximum=2048
    ),
    default_sla_ms=500.0,
)

_QWEN2_VL_7B = ModelSpec(
    name="Qwen2-VL-7B-Instruct",
    model_type=ModelType.VLM,
    num_params=7.6e9,
    num_layers=28,
    hidden_size=3584,
    num_attention_heads=28,
    num_kv_heads=4,
    head_dim=128,
    intermediate_size=18944,
    vocab_size=152064,
    vision_params=0.675e9,
    patch_size=14,
    spatial_merge=2,
    default_output_dist=LengthDistributionSpec(
        kind="lognormal", mean=128.0, std=96.0, minimum=4, maximum=1024
    ),
    default_sla_ms=500.0,
)

_OPENVLA_7B = ModelSpec(
    name="OpenVLA-7B",
    model_type=ModelType.VLA,
    num_params=7.5e9,
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_kv_heads=32,
    head_dim=128,
    intermediate_size=11008,
    vocab_size=32064,
    vision_params=0.4e9,
    tokens_per_image=256,
    default_output_dist=LengthDistributionSpec(
        kind="normal", mean=7.0, std=1.0, minimum=1, maximum=16
    ),
    default_sla_ms=200.0,
)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    m.name: m for m in (_CODELLAMA_34B, _QWEN2_VL_7B, _OPENVLA_7B)
}


def get_model(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY)


def register_model(spec: ModelSpec, overwrite: bool = False) -> None:
    if spec.name in MODEL_REGISTRY and not overwrite:
        raise ValueError(f"model {spec.name!r} already registered")
    MODEL_REGISTRY[spec.name] = spec


if __name__ == "__main__":
    for name in list_models():
        m = get_model(name)
        kv_kb = m.kv_bytes_per_token() / 1024
        pf = m.prefill_flops(1024) / 1e12
        print(
            f"{name:<24} type={m.model_type.value:<3} "
            f"KV/token={kv_kb:6.1f} KB  "
            f"prefill(1024tok)={pf:6.2f} TFLOP  "
            f"weights={m.total_weight_bytes()/1e9:5.1f} GB"
        )
