"""Hand-editable experiment configuration.

Bundles hardware, model, network and workload (request-generation) settings
into a single JSON file plus a loader. Edit ``configs/default.json`` (or pass
``--config <path>`` to the dashboard) to reconfigure the whole experiment
without touching code.

Only the Python standard library is used; JSON is chosen over YAML to avoid a
third-party dependency. Field meanings are documented in
``docs/config_design.md``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .compute_simulator import HARDWARE_REGISTRY, HardwareSpec, get_hardware
from .data_generator import (
    LengthDistribution,
    WorkloadConfig,
    WorkloadGroup,
)
from .large_model import (
    LengthDistributionSpec,
    MODEL_REGISTRY,
    ModelSpec,
    ModelType,
    get_model,
    register_model,
)
from .network import LinkSpec, NetworkSimulator, NetworkTopology


@dataclass
class ClusterConfig:
    num_nodes: int = 3
    staleness_ms: float = 0.0
    kv_capacity_bytes: Optional[float] = None      # None -> auto from HBM
    activation_reserve_bytes: float = 4e9


@dataclass
class RouterConfig:
    gamma: float = 0.9
    sla_margin_ms: float = 20.0
    token_id_bytes: int = 4
    request_overhead_bytes: int = 4096
    response_overhead_bytes: int = 4096
    visual_bytes_per_token: int = 0


@dataclass
class ExperimentConfig:
    hardware: HardwareSpec
    models: List[ModelSpec]
    links: List[LinkSpec]
    workload: WorkloadConfig
    cluster: ClusterConfig
    policies: List[str]
    router: RouterConfig

    # -- runtime helpers ----------------------------------------------------
    def apply(self) -> "ExperimentConfig":
        """Register configured models/hardware so all lookups use them."""
        for m in self.models:
            register_model(m, overwrite=True)
        HARDWARE_REGISTRY[self.hardware.name] = self.hardware
        self.workload.num_nodes = self.cluster.num_nodes
        return self

    def topology(self) -> NetworkTopology:
        return NetworkTopology(self.cluster.num_nodes, self.links)

    def new_network(self) -> NetworkSimulator:
        return NetworkSimulator(self.topology())

    def workload_model_names(self) -> List[str]:
        names: List[str] = []
        seen = set()
        for g in self.workload.groups:
            if g.model_name not in seen:
                seen.add(g.model_name)
                names.append(g.model_name)
        return names


# ----------------------------------------------------------------------------
# serialisation helpers
# ----------------------------------------------------------------------------
def _dist_to_dict(d: LengthDistribution) -> Dict:
    return {"kind": d.kind, "mean": d.mean, "std": d.std,
            "minimum": d.minimum, "maximum": d.maximum}


def _dist_from_dict(d: Dict) -> LengthDistribution:
    return LengthDistribution(
        kind=d.get("kind", "lognormal"),
        mean=d.get("mean", 128.0),
        std=d.get("std", 64.0),
        minimum=d.get("minimum", 1),
        maximum=d.get("maximum", 4096),
    )


def _hardware_to_dict(h: HardwareSpec) -> Dict:
    return {
        "name": h.name,
        "num_devices": h.num_devices,
        "peak_flops_per_device": h.peak_flops_per_device,
        "mem_bandwidth_per_device": h.mem_bandwidth_per_device,
        "mem_capacity_per_device": h.mem_capacity_per_device,
        "compute_efficiency": h.compute_efficiency,
        "bandwidth_efficiency": h.bandwidth_efficiency,
        "interconnect_bandwidth": h.interconnect_bandwidth,
        "fixed_overhead_ms": h.fixed_overhead_ms,
    }


def _hardware_from_dict(d: Dict) -> HardwareSpec:
    return HardwareSpec(**d)


def _model_to_dict(m: ModelSpec) -> Dict:
    return {
        "name": m.name,
        "model_type": m.model_type.value,
        "num_params": m.num_params,
        "num_layers": m.num_layers,
        "hidden_size": m.hidden_size,
        "num_attention_heads": m.num_attention_heads,
        "num_kv_heads": m.num_kv_heads,
        "head_dim": m.head_dim,
        "intermediate_size": m.intermediate_size,
        "vocab_size": m.vocab_size,
        "dtype_bytes": m.dtype_bytes,
        "weight_bytes": m.weight_bytes,
        "vision_params": m.vision_params,
        "patch_size": m.patch_size,
        "spatial_merge": m.spatial_merge,
        "tokens_per_image": m.tokens_per_image,
        "default_output_dist": {
            "kind": m.default_output_dist.kind,
            "mean": m.default_output_dist.mean,
            "std": m.default_output_dist.std,
            "minimum": m.default_output_dist.minimum,
            "maximum": m.default_output_dist.maximum,
        },
        "default_sla_ms": m.default_sla_ms,
        "kv_block_size": m.kv_block_size,
    }


def _model_from_dict(d: Dict) -> ModelSpec:
    """Build a ModelSpec; known names start from the registry + overrides."""
    overrides = {k: v for k, v in d.items()
                 if k not in ("name", "model_type", "default_output_dist")}
    if "model_type" in d:
        overrides["model_type"] = ModelType(d["model_type"])
    if "default_output_dist" in d:
        overrides["default_output_dist"] = LengthDistributionSpec(
            **d["default_output_dist"]
        )

    if d["name"] in MODEL_REGISTRY:
        return get_model(d["name"]).with_overrides(**overrides)
    # brand new model: requires the full field set
    return ModelSpec(name=d["name"], **overrides)


def _link_to_dict(lk: LinkSpec) -> Dict:
    return {"src": lk.src, "dst": lk.dst, "bandwidth_bps": lk.bandwidth_bps,
            "latency_ms": lk.latency_ms, "name": lk.name,
            "link_efficiency": lk.link_efficiency}


def _link_from_dict(d: Dict) -> LinkSpec:
    return LinkSpec(
        src=d["src"], dst=d["dst"], bandwidth_bps=d["bandwidth_bps"],
        latency_ms=d.get("latency_ms", 0.1), name=d.get("name", ""),
        link_efficiency=d.get("link_efficiency", 0.9),
    )


def _group_to_dict(g: WorkloadGroup) -> Dict:
    return {
        "model_name": g.model_name,
        "name": g.name,
        "entry_mode": g.entry_mode,
        "concurrency": g.concurrency,
        "entry_concurrency": list(g.entry_concurrency) if g.entry_concurrency else None,
        "entry_ratios": list(g.entry_ratios) if g.entry_ratios else None,
        "sla_ms": g.sla_ms,
        "arrival_rate": g.arrival_rate,
        "prompt_dist": _dist_to_dict(g.prompt_dist),
        "output_dist": _dist_to_dict(g.output_dist) if g.output_dist else None,
        "turns_mean": g.turns_mean,
        "turns_min": g.turns_min,
        "turns_max": g.turns_max,
        "image_size": list(g.image_size),
        "num_frames": g.num_frames,
        "shared_prefix_tokens": g.shared_prefix_tokens,
        "history_growth": g.history_growth,
    }


def _group_from_dict(d: Dict) -> WorkloadGroup:
    img = d.get("image_size", [0, 0])
    group_name = d.get("name", d.get("priority", "default"))
    return WorkloadGroup(
        model_name=d["model_name"],
        name=group_name,
        entry_mode=d.get(
            "entry_mode",
            "ratios" if d.get("entry_ratios") else "counts",
        ),
        priority=d.get("priority", group_name),
        concurrency=d.get("concurrency", 24),
        entry_concurrency=d.get("entry_concurrency"),
        entry_ratios=d.get("entry_ratios"),
        sla_ms=d.get("sla_ms"),
        arrival_rate=d.get("arrival_rate"),
        prompt_dist=_dist_from_dict(d.get("prompt_dist", {})),
        output_dist=_dist_from_dict(d["output_dist"]) if d.get("output_dist") else None,
        turns_mean=d.get("turns_mean", 4.0),
        turns_min=d.get("turns_min", 1),
        turns_max=d.get("turns_max", 12),
        image_size=(img[0], img[1]),
        num_frames=d.get("num_frames", 1),
        shared_prefix_tokens=d.get("shared_prefix_tokens", 0),
        history_growth=d.get("history_growth", 0.6),
    )


# ----------------------------------------------------------------------------
# top-level (de)serialisation
# ----------------------------------------------------------------------------
def to_dict(cfg: ExperimentConfig) -> Dict:
    w = cfg.workload
    return {
        "cluster": {
            "num_nodes": cfg.cluster.num_nodes,
            "staleness_ms": cfg.cluster.staleness_ms,
            "kv_capacity_bytes": cfg.cluster.kv_capacity_bytes,
            "activation_reserve_bytes": cfg.cluster.activation_reserve_bytes,
        },
        "router": {
            "gamma": cfg.router.gamma,
            "sla_margin_ms": cfg.router.sla_margin_ms,
            "token_id_bytes": cfg.router.token_id_bytes,
            "request_overhead_bytes": cfg.router.request_overhead_bytes,
            "response_overhead_bytes": cfg.router.response_overhead_bytes,
            "visual_bytes_per_token": cfg.router.visual_bytes_per_token,
        },
        "policies": list(cfg.policies),
        "hardware": _hardware_to_dict(cfg.hardware),
        "models": [_model_to_dict(m) for m in cfg.models],
        "network": {"links": [_link_to_dict(lk) for lk in cfg.links]},
        "workload": {
            "duration_ms": w.duration_ms,
            "session_start_spread_frac": w.session_start_spread_frac,
            "seed": w.seed,
            "mobility_start_frac": w.mobility_start_frac,
            "mobility_ratio": w.mobility_ratio,
            "mobility_granularity": w.mobility_granularity,
            "mobility_residency_turns": w.mobility_residency_turns,
            "groups": [_group_to_dict(g) for g in w.groups],
        },
    }


def from_dict(d: Dict) -> ExperimentConfig:
    cl = d.get("cluster", {})
    cluster = ClusterConfig(
        num_nodes=cl.get("num_nodes", 3),
        staleness_ms=cl.get("staleness_ms", 0.0),
        kv_capacity_bytes=cl.get("kv_capacity_bytes"),
        activation_reserve_bytes=cl.get("activation_reserve_bytes", 4e9),
    )
    rt = d.get("router", {})
    router = RouterConfig(
        gamma=rt.get("gamma", 0.9),
        sla_margin_ms=rt.get("sla_margin_ms", 20.0),
        token_id_bytes=rt.get("token_id_bytes", rt.get("request_bytes_per_token", 4)),
        request_overhead_bytes=rt.get("request_overhead_bytes", 4096),
        response_overhead_bytes=rt.get("response_overhead_bytes", 4096),
        visual_bytes_per_token=rt.get("visual_bytes_per_token", 0),
    )
    hardware = _hardware_from_dict(d["hardware"])
    models = [_model_from_dict(m) for m in d["models"]]
    links = [_link_from_dict(lk) for lk in d["network"]["links"]]
    w = d["workload"]
    workload = WorkloadConfig(
        groups=[_group_from_dict(g) for g in w["groups"]],
        num_nodes=cluster.num_nodes,
        duration_ms=w.get("duration_ms", 60000.0),
        session_start_spread_frac=w.get("session_start_spread_frac", 0.8),
        mobility_start_frac=w.get("mobility_start_frac", 0.5),
        mobility_ratio=w.get("mobility_ratio", 0.2),
        mobility_granularity=w.get("mobility_granularity", "request"),
        mobility_residency_turns=w.get("mobility_residency_turns", 2),
        seed=w.get("seed", 0),
    )
    policies = d.get("policies",
                     ["nearest", "greedy", "long_term", "long_term_kv"])
    return ExperimentConfig(
        hardware=hardware, models=models, links=links,
        workload=workload, cluster=cluster, policies=policies, router=router,
    )


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return from_dict(json.load(fh))


def save_config(cfg: ExperimentConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_dict(cfg), fh, ensure_ascii=False, indent=2)


def default_config() -> ExperimentConfig:
    """Built-in defaults equivalent to the hard-coded experiment setup."""
    from .network import default_topology

    hw = get_hardware("A800T-A2")
    models = list(MODEL_REGISTRY.values())
    links = default_topology(3).all_links()
    workload = WorkloadConfig.default_experiment()
    cluster = ClusterConfig(num_nodes=3, staleness_ms=0.0)
    router = RouterConfig()
    policies = ["nearest", "greedy", "long_term", "long_term_kv"]
    return ExperimentConfig(
        hardware=hw, models=models, links=links,
        workload=workload, cluster=cluster, policies=policies, router=router,
    )


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "default.json"
)


if __name__ == "__main__":
    # regenerate the default config file
    cfg = default_config()
    save_config(cfg, DEFAULT_CONFIG_PATH)
    print(f"wrote default config -> {DEFAULT_CONFIG_PATH}")
    # round-trip check
    again = load_config(DEFAULT_CONFIG_PATH)
    again.apply()
    print(f"models={[m.name for m in again.models]}")
    print(f"hardware={again.hardware.name} nodes={again.cluster.num_nodes} "
          f"links={len(again.links)} groups={len(again.workload.groups)} "
          f"policies={again.policies}")
