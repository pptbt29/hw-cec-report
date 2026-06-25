"""Simulation components for the long-term cost-aware offloading experiment.

Modules
-------
- ``large_model``: model specs for LLM / VLM / VLA (single source of truth).
- ``compute_simulator``: roofline-based GPU/NPU prefill/decode/memory model.
- ``data_generator``: reproducible workload trace generator.
- ``network``: node-to-node link (edge) bandwidth/latency/contention model.
- ``kv_cache``: block-level KV cache store, prefix directory and migration.
- ``node``: serving node + shared (stale) state directory.
- ``router``: per-node action enumeration, constraint filtering and policies.
"""

from .large_model import (
    LengthDistributionSpec,
    ModelType,
    ModelSpec,
    MODEL_REGISTRY,
    get_model,
    list_models,
    register_model,
)
from .compute_simulator import (
    HardwareSpec,
    ComputeSimulator,
    PrefillResult,
    DecodeResult,
    HARDWARE_REGISTRY,
    get_hardware,
)
from .data_generator import (
    LengthDistribution,
    Request,
    Session,
    WorkloadGroup,
    WorkloadConfig,
    DataGenerator,
)

from .network import (
    LinkSpec,
    NetworkSimulator,
    NetworkTopology,
    default_topology,
)
from .kv_cache import (
    KVBlock,
    KVCacheStore,
    GlobalKVDirectory,
    MigrationPlan,
)
from .node import (
    NodeState,
    ServingNode,
    GlobalStateDirectory,
    build_cluster,
)
from .router import (
    Policy,
    StateMode,
    Action,
    ActionCost,
    Router,
    simulate_trace,
)

__all__ = [
    "LengthDistributionSpec",
    "ModelType",
    "ModelSpec",
    "MODEL_REGISTRY",
    "get_model",
    "list_models",
    "register_model",
    "HardwareSpec",
    "ComputeSimulator",
    "PrefillResult",
    "DecodeResult",
    "HARDWARE_REGISTRY",
    "get_hardware",
    "LengthDistribution",
    "Request",
    "Session",
    "WorkloadGroup",
    "WorkloadConfig",
    "DataGenerator",
    "LinkSpec",
    "NetworkSimulator",
    "NetworkTopology",
    "default_topology",
    "KVBlock",
    "KVCacheStore",
    "GlobalKVDirectory",
    "MigrationPlan",
    "NodeState",
    "ServingNode",
    "GlobalStateDirectory",
    "build_cluster",
    "Policy",
    "StateMode",
    "Action",
    "ActionCost",
    "Router",
    "simulate_trace",
]
