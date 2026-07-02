"""Metrics dashboard generator.

Runs the four offloading policies across one or more models on the same
workload trace, collects per-request records and aggregate metrics, and emits
a single self-contained interactive HTML dashboard (no third-party deps).

Usage:
    python -m sim.dashboard                      # default config -> output/dashboard.html
    python -m sim.dashboard --open               # also open in browser
    python -m sim.dashboard --config my.json     # use a hand-edited config file
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .config import ExperimentConfig, default_config, load_config
from .data_generator import DataGenerator, Request
from .large_model import get_model
from .network import LinkSpec
from .router import Policy, simulate_trace

_DEFAULT_NODE_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _infer_node_names(num_nodes: int, links: List[LinkSpec]) -> List[str]:
    """Derive human-readable node tags from link names (e.g. A-B-100G -> A, B)."""
    names: List[Optional[str]] = [None] * num_nodes
    for lk in links:
        if not lk.name:
            continue
        parts = lk.name.split("-")
        if len(parts) < 2:
            continue
        for node_id, tag in ((lk.src, parts[0]), (lk.dst, parts[1])):
            if 0 <= node_id < num_nodes and names[node_id] is None:
                names[node_id] = tag
    fallback = _DEFAULT_NODE_NAMES + [str(i) for i in range(len(_DEFAULT_NODE_NAMES), num_nodes)]
    return [
        names[i] if names[i] else (fallback[i] if i < len(fallback) else str(i))
        for i in range(num_nodes)
    ]


def _node_connectivity(node_id: int, names: List[str], links: List[LinkSpec]) -> str:
    parts: List[str] = []
    for lk in links:
        peer_id: Optional[int] = None
        if lk.src == node_id:
            peer_id = lk.dst
        elif lk.dst == node_id:
            peer_id = lk.src
        if peer_id is None or peer_id >= len(names):
            continue
        bw_g = lk.bandwidth_bps / 1e9
        bw = f"{bw_g:.0f}G" if bw_g >= 1 else f"{lk.bandwidth_bps / 1e6:.0f}M"
        parts.append(f"↔{names[peer_id]} {bw}")
    return " · ".join(parts) if parts else "推理实例 + 请求入口"


def _build_node_labels(experiment: ExperimentConfig) -> List[Dict]:
    n = experiment.cluster.num_nodes
    custom = getattr(experiment.cluster, "node_names", None)
    names = list(custom[:n]) if custom and len(custom) >= n else _infer_node_names(n, experiment.links)
    return [
        {
            "id": i,
            "name": names[i],
            "label": f"{names[i]}（节点{i}）",
            "role": "入口客户端 / 推理实例",
            "connectivity": _node_connectivity(i, names, experiment.links),
        }
        for i in range(n)
    ]


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def _request_dict(req: Request, tag: str = "") -> Dict:
    d = {
        "request_id": req.request_id,
        "session_id": req.session_id,
        "model_name": req.model_name,
        "model_type": req.model_type,
        "arrival_ms": round(req.arrival_ms, 2),
        "arrival_s": round(req.arrival_ms / 1000.0, 2),
        "entry_node": req.entry_node,
        "home_node": req.home_node,
        "group_name": req.group_name,
        "sla_ms": req.sla_ms,
        "input_tokens": req.input_tokens,
        "output_len": req.output_len,
        "prefix_tokens": req.prefix_tokens,
        "is_session_first": req.is_session_first,
        "turn_index": req.turn_index,
        "mobility_switched": req.mobility_switched,
        "visual_tokens": req.visual_tokens,
        "state_tokens": req.state_tokens,
    }
    if tag:
        d["tag"] = tag
    return d


def _workload_stats(requests: List[Request], duration_ms: float, mobility_start_frac: float) -> Dict:
    if not requests:
        return {"num_requests": 0}

    by_model: Dict[str, int] = {}
    by_group: Dict[str, int] = {}
    by_entry: Dict[str, int] = {}
    by_home: Dict[str, int] = {}
    prompts = [float(r.input_tokens) for r in requests]
    outputs = [float(r.output_len) for r in requests]
    prefixes = [float(r.prefix_tokens) for r in requests]
    turns = [float(r.turn_index) for r in requests]
    session_turns: Dict[str, int] = {}
    switch_after = duration_ms * mobility_start_frac

    for r in requests:
        by_model[r.model_name] = by_model.get(r.model_name, 0) + 1
        by_group[r.group_name] = by_group.get(r.group_name, 0) + 1
        by_entry[str(r.entry_node)] = by_entry.get(str(r.entry_node), 0) + 1
        by_home[str(r.home_node)] = by_home.get(str(r.home_node), 0) + 1
        session_turns[r.session_id] = max(session_turns.get(r.session_id, 0), r.turn_index + 1)

    moved = sum(1 for r in requests if r.mobility_switched)
    after_mob = sum(1 for r in requests if r.arrival_ms >= switch_after)
    sessions = len(session_turns)
    turns_per_session = list(session_turns.values())

    return {
        "num_requests": len(requests),
        "num_sessions": sessions,
        "by_model": by_model,
        "by_group": by_group,
        "by_entry_node": by_entry,
        "by_home_node": by_home,
        "input_tokens_avg": round(sum(prompts) / len(prompts), 1),
        "input_tokens_p50": round(_pct(prompts, 50), 0),
        "input_tokens_p95": round(_pct(prompts, 95), 0),
        "output_len_avg": round(sum(outputs) / len(outputs), 1),
        "output_len_p50": round(_pct(outputs, 50), 0),
        "output_len_p95": round(_pct(outputs, 95), 0),
        "prefix_tokens_avg": round(sum(prefixes) / len(prefixes), 1),
        "prefix_tokens_p95": round(_pct(prefixes, 95), 0),
        "turn_index_avg": round(sum(turns) / len(turns), 2),
        "turns_per_session_avg": round(sum(turns_per_session) / len(turns_per_session), 2),
        "turns_per_session_p95": round(_pct([float(x) for x in turns_per_session], 95), 0),
        "first_turn_ratio": round(sum(1 for r in requests if r.is_session_first) / len(requests), 4),
        "mobility_switched_count": moved,
        "mobility_switched_ratio": round(moved / len(requests), 4),
        "after_mobility_start_count": after_mob,
        "after_mobility_start_ratio": round(after_mob / len(requests), 4),
        "total_kv_tokens": int(sum(r.input_tokens + r.output_len for r in requests)),
    }


def _histogram(values: List[float], nb: int = 20) -> Dict:
    if not values:
        return {"bins": [], "avg": 0.0, "p95": 0.0}
    lo, hi = min(values), max(values)
    avg = round(sum(values) / len(values), 1)
    p95 = round(_pct(values, 95), 0)
    if lo == hi:
        return {
            "bins": [{"label": str(int(lo)), "lo": lo, "hi": hi, "count": len(values)}],
            "avg": avg,
            "p95": p95,
        }
    width = (hi - lo) / nb
    counts = [0] * nb
    for v in values:
        idx = min(nb - 1, int((v - lo) / width) if hi > lo else 0)
        counts[idx] += 1
    bins = []
    for i in range(nb):
        blo = lo + i * width
        bhi = lo + (i + 1) * width if i < nb - 1 else hi
        if i == nb - 1:
            label = f"{int(round(blo))}+"
        else:
            label = f"{int(round(blo))}-{int(round(bhi))}"
        bins.append({
            "label": label,
            "lo": round(blo, 1),
            "hi": round(bhi, 1),
            "count": counts[i],
        })
    return {"bins": bins, "avg": avg, "p95": p95}


def _histogram_discrete(values: List[int]) -> Dict:
    if not values:
        return {"bins": [], "avg": 0.0, "p95": 0.0}
    floats = [float(v) for v in values]
    counts: Dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    bins = [
        {"label": str(k), "lo": k, "hi": k, "count": counts[k]}
        for k in sorted(counts.keys())
    ]
    return {
        "bins": bins,
        "avg": round(sum(floats) / len(floats), 2),
        "p95": round(_pct(floats, 95), 0),
    }


def _time_arrival_series(
    requests: List[Request],
    duration_ms: float,
    num_nodes: int,
    nb: int = 40,
) -> Dict:
    series = {str(i): [0] * nb for i in range(num_nodes)}
    if not requests or duration_ms <= 0:
        return {"nb": nb, "duration_ms": duration_ms, "series": series}
    for req in requests:
        idx = min(nb - 1, int(req.arrival_ms / duration_ms * nb))
        key = str(req.entry_node)
        if key in series:
            series[key][idx] += 1
    return {"nb": nb, "duration_ms": duration_ms, "series": series}


def _exec_time_series(records: List[Dict], duration_ms: float, num_nodes: int, nb: int = 40) -> Dict:
    series = {str(i): [0] * nb for i in range(num_nodes)}
    if not records or duration_ms <= 0:
        return {"nb": nb, "duration_ms": duration_ms, "series": series}
    for rec in records:
        idx = min(nb - 1, int(rec["t"] / duration_ms * nb))
        key = str(rec["exec"])
        if key in series:
            series[key][idx] += 1
    return {"nb": nb, "duration_ms": duration_ms, "series": series}


def _workload_distributions(
    model_reqs: List[Request],
    duration_ms: float,
    num_nodes: int,
) -> Dict:
    if not model_reqs:
        return {}

    session_req_counts: Dict[str, int] = {}
    for req in model_reqs:
        session_req_counts[req.session_id] = session_req_counts.get(req.session_id, 0) + 1

    return {
        "session_requests": _histogram_discrete(list(session_req_counts.values())),
        "input_tokens": _histogram([float(r.input_tokens) for r in model_reqs]),
        "output_tokens": _histogram([float(r.output_len) for r in model_reqs]),
        "entry_over_time": _time_arrival_series(model_reqs, duration_ms, num_nodes),
        "exec_over_time": {},
    }


def _pick_samples(requests: List[Request], limit: int = 16) -> List[Dict]:
    if not requests:
        return []

    samples: List[Dict] = []
    seen: set = set()

    def add(req: Request, tag: str) -> None:
        if req.request_id in seen or len(samples) >= limit:
            return
        samples.append(_request_dict(req, tag))
        seen.add(req.request_id)

    for req in requests[:5]:
        add(req, "前序请求")

    for req in requests:
        if req.mobility_switched:
            add(req, "移动后入口变化")
        if len([s for s in samples if s.get("tag") == "移动后入口变化"]) >= 5:
            break

    by_session: Dict[str, List[Request]] = {}
    for req in requests:
        by_session.setdefault(req.session_id, []).append(req)
    multi = max(by_session.values(), key=len)
    if len(multi) > 1:
        for req in multi[:4]:
            add(req, "同 session 多轮")

    return samples[:limit]


def _build_workload_payload(
    experiment: ExperimentConfig,
    requests: List[Request],
) -> Dict:
    wl = experiment.workload
    cluster = experiment.cluster
    groups = []
    for g in wl.groups:
        model = get_model(g.model_name)
        groups.append({
            "model_name": g.model_name,
            "name": g.name,
            "entry_mode": g.entry_mode,
            "concurrency": g.concurrency,
            "entry_concurrency": list(g.entry_concurrency) if g.entry_concurrency else None,
            "entry_ratios": list(g.entry_ratios) if g.entry_ratios else None,
            "sla_ms": g.sla_ms if g.sla_ms is not None else model.default_sla_ms,
            "turns_mean": g.turns_mean,
            "shared_prefix_tokens": g.shared_prefix_tokens,
            "image_size": list(g.image_size),
        })

    by_model: Dict[str, Dict] = {}
    for model_name in experiment.workload_model_names():
        model_reqs = [r for r in requests if r.model_name == model_name]
        by_model[model_name] = {
            "summary": _workload_stats(model_reqs, wl.duration_ms, wl.mobility_start_frac),
            "distributions": _workload_distributions(
                model_reqs, wl.duration_ms, cluster.num_nodes,
            ),
            "samples": _pick_samples(model_reqs),
        }

    return {
        "seed": wl.seed,
        "duration_ms": wl.duration_ms,
        "mobility_ratio": wl.mobility_ratio,
        "mobility_start_frac": wl.mobility_start_frac,
        "mobility_granularity": wl.mobility_granularity,
        "total_concurrency": sum(g.concurrency for g in wl.groups),
        "groups": groups,
        "summary": _workload_stats(requests, wl.duration_ms, wl.mobility_start_frac),
        "by_model": by_model,
    }


def run_experiments(experiment: Optional[ExperimentConfig] = None) -> Dict:
    """Run all configured policies for each model on one shared trace.

    ``experiment`` is an :class:`ExperimentConfig`; when omitted the built-in
    defaults are used. Edit ``configs/default.json`` (or pass a loaded config)
    to reconfigure hardware / models / network / workload / cluster.
    """
    experiment = experiment or default_config()
    experiment.apply()

    cluster = experiment.cluster
    hw = experiment.hardware
    requests = DataGenerator(experiment.workload).generate()
    policies = [Policy(p) for p in experiment.policies]

    data: Dict = {
        "meta": {
            "hardware": hw.name,
            "num_nodes": cluster.num_nodes,
            "node_labels": _build_node_labels(experiment),
            "staleness_ms": cluster.staleness_ms,
            "duration_ms": experiment.workload.duration_ms,
            "mobility_ratio": experiment.workload.mobility_ratio,
            "mobility_start_frac": experiment.workload.mobility_start_frac,
            "session_start_spread_frac": (
                experiment.workload.session_start_spread_frac
            ),
            "mobility_granularity": experiment.workload.mobility_granularity,
            "mobility_residency_turns": experiment.workload.mobility_residency_turns,
            "total_concurrency": sum(g.concurrency for g in experiment.workload.groups),
            "seed": experiment.workload.seed,
            "router_gamma": experiment.router.gamma,
            "total_requests": len(requests),
            "policies": experiment.policies,
        },
        "models": {},
    }

    for model_name in experiment.workload_model_names():
        model = get_model(model_name)
        per_policy = {}
        for pol in policies:
            net = experiment.new_network()
            res = simulate_trace(
                pol, requests, model, hw, net,
                num_nodes=cluster.num_nodes, staleness_ms=cluster.staleness_ms,
                gamma=experiment.router.gamma,
                sla_margin_ms=experiment.router.sla_margin_ms,
                collect_records=True,
                kv_capacity_bytes=cluster.kv_capacity_bytes,
                activation_reserve_bytes=cluster.activation_reserve_bytes,
                token_id_bytes=experiment.router.token_id_bytes,
                request_overhead_bytes=experiment.router.request_overhead_bytes,
                response_overhead_bytes=experiment.router.response_overhead_bytes,
                visual_bytes_per_token=experiment.router.visual_bytes_per_token,
            )
            per_policy[pol.value] = res
        data["models"][model_name] = per_policy

    data["workload"] = _build_workload_payload(experiment, requests)
    wl = experiment.workload
    for model_name in experiment.workload_model_names():
        dist = data["workload"]["by_model"][model_name]["distributions"]
        for pol in ("nearest", "greedy", "long_term"):
            records = data["models"][model_name].get(pol, {}).get("records", [])
            dist["exec_over_time"][pol] = _exec_time_series(
                records, wl.duration_ms, cluster.num_nodes,
            )
    return data


def export_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def render_html(data: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False)
    html = _HTML_TEMPLATE.replace("__DATA__", payload)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CEC-LLM 卸载模拟 Metrics Dashboard</title>
<style>
  :root,[data-theme="dark"]{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --border:#30363d;
    --fg:#e6edf3; --muted:#8b949e; --grid:#21262d;
    --nearest:#8b949e; --greedy:#f0883e; --long_term:#3fb950; --long_term_kv:#58a6ff;
    --good:#3fb950; --bad:#f85149; --warn:#d29922;
    --tooltip-bg:#1c2230; --tooltip-border:#484f58;
    --bar-label-on:#ffffff;
    --bar-label-out:#8b949e;
  }
  [data-theme="light"]{
    --bg:#f6f8fa; --panel:#ffffff; --panel2:#f0f3f6; --border:#d0d7de;
    --fg:#1f2328; --muted:#656d76; --grid:#eaeef2;
    --nearest:#656d76; --greedy:#bc4c00; --long_term:#1a7f37; --long_term_kv:#0969da;
    --good:#1a7f37; --bad:#cf222e; --warn:#9a6700;
    --tooltip-bg:#ffffff; --tooltip-border:#d0d7de;
    --bar-label-on:#ffffff;
    --bar-label-out:#656d76;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;}
  header{padding:20px 28px;border-bottom:1px solid var(--border);
    background:linear-gradient(180deg,var(--panel),var(--bg));
    display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap}
  header .head-main{flex:1;min-width:240px}
  header h1{margin:0;font-size:20px;font-weight:650}
  header .sub{color:var(--muted);font-size:13px;margin-top:6px}
  .theme-btn{padding:6px 12px;border:1px solid var(--border);border-radius:8px;
    background:var(--panel2);color:var(--fg);cursor:pointer;font-size:12px;white-space:nowrap}
  .theme-btn:hover{border-color:#58a6ff}
  .wrap{padding:20px 28px;max-width:1280px;margin:0 auto}
  .tabs{display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap}
  .tab{padding:7px 16px;border:1px solid var(--border);border-radius:8px;
    background:var(--panel);color:var(--muted);cursor:pointer;font-size:13px;transition:.15s}
  .tab:hover{color:var(--fg);border-color:#484f58}
  .tab.active{background:var(--panel2);color:var(--fg);border-color:#58a6ff}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 18px;font-size:12px;color:var(--muted)}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .dot{width:11px;height:11px;border-radius:3px;display:inline-block}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:22px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
  .card .k{color:var(--muted);font-size:12px}
  .card .v{font-size:22px;font-weight:650;margin-top:6px}
  .card .d{font-size:12px;margin-top:4px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:900px){.grid2{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin-bottom:18px}
  .panel h3{margin:0 0 4px;font-size:15px}
  .panel .hint{color:var(--muted);font-size:12px;margin-bottom:12px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--grid)}
  th:first-child,td:first-child{text-align:left}
  thead th{color:var(--muted);font-weight:600;border-bottom:1px solid var(--border)}
  tbody tr:hover{background:var(--panel2)}
  .best{background:rgba(63,185,80,.16);color:var(--fg);font-weight:650}
  [data-theme="light"] .best{background:#dafbe1}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .chart-wrap{position:relative}
  .axis{stroke:var(--grid);stroke-width:1}
  .axlab{fill:var(--muted);font-size:10px}
  .barval-out{fill:var(--bar-label-out);font-size:10px;font-weight:400}
  .barval-in{fill:var(--bar-label-on);font-size:10px;font-weight:400}
  text{font-family:inherit}
  .chart-tip{position:fixed;z-index:1000;pointer-events:none;display:none;
    padding:8px 10px;border-radius:8px;font-size:12px;line-height:1.45;
    background:var(--tooltip-bg);border:1px solid var(--tooltip-border);
    color:var(--fg);box-shadow:0 8px 24px rgba(0,0,0,.18);max-width:280px}
  .chart-tip .tip-title{font-weight:650;margin-bottom:4px;color:var(--muted);font-size:11px}
  .chart-tip .tip-row{display:flex;align-items:center;gap:8px;margin-top:3px}
  .chart-tip .tip-dot{width:8px;height:8px;border-radius:2px;flex-shrink:0}
  .panel-collapsible{padding:0}
  .collapse-head{width:100%;display:flex;align-items:center;justify-content:space-between;gap:12px;
    padding:16px 18px;background:none;border:none;color:inherit;cursor:pointer;font:inherit;text-align:left}
  .collapse-head:hover{background:var(--panel2)}
  .collapse-title-wrap{flex:1;min-width:0}
  .collapse-title-wrap h3{margin:0 0 4px;font-size:15px}
  .collapse-title-wrap .hint{margin:0}
  .collapse-chevron{color:var(--muted);font-size:14px;flex-shrink:0;transition:transform .2s}
  .collapse-head[aria-expanded="true"] .collapse-chevron{transform:rotate(90deg)}
  .collapse-body{padding:0 18px 20px;border-top:1px solid var(--grid)}
  #workloadContent{padding-top:2px}
  .subsec{margin:20px 0 10px;font-size:13px;font-weight:650;color:var(--fg)}
  #workloadContent .subsec{margin-top:40px;margin-bottom:10px}
  #workloadContent .subsec:first-of-type{margin-top:12px}
  #workloadContent .hint{margin:12px 0 16px;line-height:1.55}
  #workloadContent table{margin-bottom:18px}
  #workloadContent table th,#workloadContent table td{padding:10px 10px}
  #workloadContent .stats-grid{margin:0 0 16px}
  #workloadContent #workloadDists{margin-bottom:6px}
  .stats-grid{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
  .stat-item{
    width:136px;height:68px;box-sizing:border-box;
    background:var(--panel2);border:1px solid var(--border);border-radius:8px;
    padding:10px 12px;display:flex;flex-direction:column;justify-content:space-between;
    flex:0 0 auto}
  .stat-item .k{color:var(--muted);font-size:11px;line-height:1.35}
  .stat-item .v{font-size:15px;font-weight:600;line-height:1.2;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .dist-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:14px 0 8px}
  @media(max-width:760px){.dist-grid{grid-template-columns:1fr}}
  .dist-block{min-width:0}
  .dist-stats{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
  .dist-stats .stat-item{width:120px;height:60px;padding:8px 10px}
  .dist-stats .stat-item .v{font-size:14px}
  .pie-grid{display:flex;gap:18px;margin:8px 0 16px;align-items:flex-start;flex-wrap:wrap}
  .pie-block{display:flex;flex-direction:column;align-items:flex-start;min-width:0}
  .pie-block .pie-title{font-size:12px;font-weight:650;margin-bottom:6px;color:var(--fg)}
  .node-legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin:4px 0 10px;font-size:12px;color:var(--muted);line-height:1.45}
  .node-legend span{display:inline-flex;align-items:center;gap:6px}
  .node-legend .node-meta{color:var(--fg);font-weight:600}
  .chart-block-title{font-size:12px;font-weight:650;margin:0 0 6px;color:var(--fg)}
  .pie-slice-n{font-size:6px;font-weight:600;fill:var(--bar-label-on)}
  .pie-slice-p{font-size:5px;fill:var(--bar-label-on);opacity:0.92}
  .time-line-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:8px 0 20px}
  @media(max-width:760px){.time-line-grid{grid-template-columns:1fr}}
  .node-line-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:8px}
  @media(max-width:900px){.node-line-grid{grid-template-columns:1fr}}
  .policy-block{margin:0 0 22px}
  .policy-block h4{margin:0 0 10px;font-size:12px;font-weight:650;color:var(--muted)}
  .phase-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
  .phase-chart{min-width:0;border-top:1px solid var(--grid);padding-top:10px}
  .phase-chart h4{margin:0 0 4px;font-size:12px;font-weight:650}
  @media(max-width:760px){.phase-grid{grid-template-columns:1fr}}
  .time-sec-title{margin:0 0 10px;font-size:13px;font-weight:650;color:var(--fg)}
  .sample-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:8px;margin-top:4px;margin-bottom:8px}
  .sample-table{width:100%;border-collapse:collapse;font-size:12px;min-width:960px}
  .sample-table th,.sample-table td{padding:10px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
  .sample-table th:first-child,.sample-table td:first-child{text-align:left}
  .sample-table thead th{position:sticky;top:0;background:var(--panel2);color:var(--muted);font-weight:600}
  .sample-table tbody tr:hover{background:var(--panel2)}
  .tag-pill{display:inline-block;padding:1px 6px;border-radius:999px;font-size:11px;background:var(--panel2);color:var(--muted)}
  .tag-pill.moved{color:var(--warn)}
  footer{color:var(--muted);font-size:12px;padding:20px 28px;border-top:1px solid var(--border)}
</style>
</head>
<body>
<header>
  <div class="head-main">
    <h1>CEC-LLM 推理卸载 · 模拟 Metrics Dashboard</h1>
    <div class="sub" id="meta"></div>
  </div>
  <button type="button" class="theme-btn" id="themeToggle" title="切换明暗主题">🌙 暗色</button>
</header>
<div class="chart-tip" id="chartTip"></div>
<div class="wrap">
  <div class="tabs" id="modelTabs"></div>
  <div class="legend" id="legend"></div>
  <div class="cards" id="cards"></div>

  <div class="panel panel-collapsible">
    <button type="button" class="collapse-head" id="workloadToggle" aria-expanded="false" aria-controls="workloadPanel">
      <div class="collapse-title-wrap">
        <h3>请求轨迹数据</h3>
        <p class="hint">展开查看工作负载配置、统计摘要与请求样例（同一条可复现轨迹，按当前模型 Tab 过滤）。</p>
      </div>
      <span class="collapse-chevron" aria-hidden="true">▸</span>
    </button>
    <div class="collapse-body" id="workloadPanel" hidden>
      <div id="workloadContent"></div>
    </div>
  </div>

  <div class="panel">
    <h3>策略汇总对比</h3>
    <div class="hint">同一条请求轨迹下四种策略的关键指标；每列最优值高亮（绿色）。</div>
    <div id="summaryTable"></div>
  </div>

  <div class="panel">
    <h3>Avg E2E 延迟分拆</h3>
    <div class="hint">各项均为单请求平均耗时，分项合计应与 Avg E2E 一致；网络拆为请求转发和响应回传，状态获取拆为 KV 迁移和 KV 重算。</div>
    <div id="latencyBreakdown"></div>
  </div>

  <div class="panel">
    <h3>排队来源拆分</h3>
    <div class="hint">按策略和实际执行节点拆分请求占比及平均 backlog。当前模型假设 Decode 由 continuous batching 吸收，不进入串行 admission queue。</div>
    <div id="queueBreakdown"></div>
  </div>

  <div class="panel">
    <h3>E2E 各阶段耗时分布</h3>
    <div class="hint">每个 phase 单独一张累计分布曲线（ECDF），每张图内比较四种路由策略。横轴为该 phase 耗时，纵轴为累计请求比例；横轴使用对数映射以保留小于 1 ms 的差异。</div>
    <div id="latencyDistribution"></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h3>关键指标对比（柱状）</h3>
      <div class="hint">越低越好：P99 TTFT、跨节点比例、迁移字节、owner 切换。</div>
      <div id="barCharts"></div>
    </div>
    <div class="panel">
      <h3>状态获取动作分布</h3>
      <div class="hint">local 命中 / migrate 迁移 / recompute 重算 / fresh 新建。</div>
      <div id="actionDist"></div>
    </div>
  </div>

  <div class="panel">
    <h3>Migrate 原因拆分</h3>
    <div class="hint">硬约束类表示入口节点没有可执行动作；即时成本表示当前 E2E 对比选择迁移；FutureCost 表示 long-term 为降低未来成本主动牺牲当前 E2E。</div>
    <div id="migrateReasons"></div>
  </div>

  <div class="panel">
    <h3>P99 TTFT 时间序列</h3>
    <div class="hint">按到达时间分桶的滚动 P99 TTFT；竖线为用户移动起始时刻。</div>
    <div id="ttftSeries"></div>
  </div>

  <div class="panel">
    <h3>累计跨节点请求（状态黏附曲线）</h3>
    <div class="hint">移动后曲线越陡，说明请求越被吸到远端节点（黏附越严重）。</div>
    <div id="stickiness"></div>
  </div>

  <div class="panel">
    <h3>累计卸载相关成本对比（几种 routing 方法）</h3>
    <div class="hint">累计 (网络传输 + 排队 + 状态获取) 时间，即卸载决策直接影响的成本（不含固定 prefill/decode）。
      竖线为用户移动时刻。注意：greedy 把小请求转发到 KV 所在节点，即时成本最低，但代价是状态黏附与负载集中
      （见上方“黏附曲线”“跨节点比例”）；long-term 牺牲一部分前期迁移成本以降低黏附与负载热点。</div>
    <div id="cumCost"></div>
  </div>

  <div class="panel">
    <h3>链路利用率</h3>
    <div class="hint">100G 直连 vs 25G 跨主机 RDMA 在各策略下的占用与传输字节。</div>
    <div id="linkUtil"></div>
  </div>
</div>
<footer>由 <code>sim/dashboard.py</code> 离线生成 · 纯标准库 · 数据内联，可直接分享此 HTML。</footer>

<script>
const DATA = __DATA__;
const POLICIES = ["nearest","greedy","long_term","long_term_kv"];
const PLABEL = {nearest:"Nearest(基线)",greedy:"Greedy",long_term:"Long-term",long_term_kv:"Long-term+KV"};
const PCOLOR = {nearest:"#8b949e",greedy:"#f0883e",long_term:"#3fb950",long_term_kv:"#58a6ff"};
const NODE_COLORS=["#58a6ff","#3fb950","#f0883e"];
const PRIORITY_COLORS={high:"#f85149",normal:"#8b949e"};
const MIGRATE_REASON_LABEL={
  entry_sla:"入口 SLA",entry_memory:"入口 Memory",
  entry_sla_and_memory:"入口 SLA+Memory",
  entry_mixed_constraints:"入口混合约束",
  immediate_cost:"即时成本",future_cost:"FutureCost",
  unclassified:"未分类"
};
const MIGRATE_REASON_COLOR={
  entry_sla:"#f85149",entry_memory:"#d29922",
  entry_sla_and_memory:"#a371f7",entry_mixed_constraints:"#db6d28",
  immediate_cost:"#58a6ff",future_cost:"#3fb950",unclassified:"#8b949e"
};
let currentModel = Object.keys(DATA.models)[0];
const chartTip = document.getElementById("chartTip");

function initTheme(){
  const saved = localStorage.getItem("cec-dashboard-theme") || "dark";
  applyTheme(saved);
  document.getElementById("themeToggle").onclick = ()=>{
    const next = document.documentElement.getAttribute("data-theme")==="light" ? "dark" : "light";
    applyTheme(next);
    localStorage.setItem("cec-dashboard-theme", next);
  };
}
function applyTheme(mode){
  document.documentElement.setAttribute("data-theme", mode);
  const btn = document.getElementById("themeToggle");
  btn.textContent = mode==="light" ? "☀️ 亮色" : "🌙 暗色";
}

function showTip(html, x, y){
  chartTip.innerHTML = html;
  chartTip.style.display = "block";
  const pad = 12, rect = chartTip.getBoundingClientRect();
  let left = x + pad, top = y + pad;
  if(left + rect.width > window.innerWidth - 8) left = x - rect.width - pad;
  if(top + rect.height > window.innerHeight - 8) top = y - rect.height - pad;
  chartTip.style.left = Math.max(8, left) + "px";
  chartTip.style.top = Math.max(8, top) + "px";
}
function hideTip(){ chartTip.style.display = "none"; }

function attachBarHover(rect, html){
  rect.style.cursor = "default";
  rect.addEventListener("mouseenter", ev=>showTip(html, ev.clientX, ev.clientY));
  rect.addEventListener("mousemove", ev=>showTip(html, ev.clientX, ev.clientY));
  rect.addEventListener("mouseleave", hideTip);
}

const meta = DATA.meta;
const NODE_INFO=meta.node_labels||[];
function nodeInfo(id){
  const n=Number(id);
  return NODE_INFO.find(x=>x.id===n)||{id:n,name:String(n),label:"节点"+n,role:"",connectivity:""};
}
function NODE_LABEL(n){return nodeInfo(n).label;}
function NODE_NAME(n){return nodeInfo(n).name;}
function nodeLegendHtml(){
  if(!NODE_INFO.length) return "";
  return `<div class="node-legend">`+NODE_INFO.map(x=>
    `<span><i class="dot" style="background:${NODE_COLORS[x.id%3]}"></i>`+
    `<span class="node-meta">${x.label}</span> · ${x.connectivity}</span>`
  ).join("")+`</div>`;
}
document.getElementById("meta").innerHTML =
  `硬件 ${meta.hardware} · ${meta.num_nodes} 节点 · 时长 ${(meta.duration_ms/1000).toFixed(0)}s · `+
  `移动比例 ${(meta.mobility_ratio*100).toFixed(0)}% (起于 ${(meta.mobility_start_frac*100).toFixed(0)}%) · `+
  `${meta.mobility_granularity}`+
  `${meta.mobility_granularity==="markov"?" / 驻留 "+meta.mobility_residency_turns+" 轮":""} · `+
  `总并发用户 ${meta.total_concurrency} · `+
  `seed ${meta.seed} · gamma ${meta.router_gamma} · 请求总数 ${meta.total_requests} · staleness ${meta.staleness_ms}ms`;

function tabs(){
  const el=document.getElementById("modelTabs"); el.innerHTML="";
  Object.keys(DATA.models).forEach(m=>{
    const d=document.createElement("div");
    d.className="tab"+(m===currentModel?" active":"");
    d.textContent=m; d.onclick=()=>{currentModel=m;render();}; el.appendChild(d);
  });
}
function legend(){
  document.getElementById("legend").innerHTML = POLICIES.map(p=>
    `<span><i class="dot" style="background:${PCOLOR[p]}"></i>${PLABEL[p]}</span>`).join("");
}

function fmt(x,d=1){return (typeof x==="number"&&isFinite(x))?x.toFixed(d):x;}
function el(tag,attrs,children){
  const e=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const k in attrs) e.setAttribute(k,attrs[k]);
  (children||[]).forEach(c=>e.appendChild(c)); return e;
}
function svg(w,h){return el("svg",{viewBox:`0 0 ${w} ${h}`});}
function txt(x,y,s,cls,anchor){const t=el("text",{x,y,class:cls||"axlab","text-anchor":anchor||"start"});t.textContent=s;return t;}

function cards(){
  const M=DATA.models[currentModel];
  const base=M.nearest, lt=M.long_term_kv;
  const reqs=base.num_requests;
  const e2eImp=(base.avg_e2e_ms-lt.avg_e2e_ms)/base.avg_e2e_ms*100;
  const xnodeImp=(base.cross_node_ratio-lt.cross_node_ratio)*100;
  const migImp=(base.migrate_bytes_mb-lt.migrate_bytes_mb)/Math.max(base.migrate_bytes_mb,1e-9)*100;
  const c=[
    ["请求数(该模型)",reqs,"",""],
    ["P99 TTFT (Greedy)",fmt(M.greedy.p99_ttft_ms)+" ms","",""],
    ["P99 TTFT (LT+KV)",fmt(lt.p99_ttft_ms)+" ms","",""],
    ["跨节点↓ vs 基线",fmt(xnodeImp,1)+" pp",xnodeImp>0?"good":"bad","黏附改善"],
    ["迁移字节↓ vs 基线",fmt(migImp,0)+"%",migImp>0?"good":"bad","状态搬运减少"],
    ["owner 切换 (LT+KV)",lt.owner_switch_count,"",""],
  ];
  document.getElementById("cards").innerHTML = c.map(x=>{
    const col=x[2]==="good"?"var(--good)":x[2]==="bad"?"var(--bad)":"var(--fg)";
    return `<div class="card"><div class="k">${x[0]}</div>`+
      `<div class="v" style="color:${col}">${x[1]}</div>`+
      `<div class="d" style="color:var(--muted)">${x[3]}</div></div>`;
  }).join("");
}

function summaryTable(){
  const M=DATA.models[currentModel];
  const rows=[
    ["avg E2E (ms)","avg_e2e_ms",1,"min"],
    ["预测 FutureCost (ms)","avg_predicted_future_cost_ms",3,"none"],
    ["P50 TTFT (ms)","p50_ttft_ms",1,"min"],
    ["P95 TTFT (ms)","p95_ttft_ms",1,"min"],
    ["P99 TTFT (ms)","p99_ttft_ms",1,"min"],
    ["SLA 违约率 (%)","sla_violation_ratio",2,"min",100],
    ["不可执行率 (%)","infeasible_ratio",2,"min",100],
    ["跨节点比例 (%)","cross_node_ratio",1,"min",100],
    ["用户入口迁移次数","mobility_transition_count",0,"min"],
    ["migrate 次数","migrate_count",0,"min"],
    ["recompute 次数","recompute_count",0,"min"],
    ["owner 切换","owner_switch_count",0,"min"],
    ["迁移字节 (MB)","migrate_bytes_mb",1,"min"],
    ["累计卸载成本 (ms)","offload_cost_ms",0,"min"],
  ];
  let h="<table><thead><tr><th>指标</th>"+POLICIES.map(p=>`<th>${PLABEL[p]}</th>`).join("")+"</tr></thead><tbody>";
  rows.forEach(r=>{
    const vals=POLICIES.map(p=>M[p][r[1]]*(r[4]||1));
    const best=r[3]==="none"?null:Math.min(...vals);
    h+=`<tr><td>${r[0]}</td>`+vals.map(v=>{
      const cls=best!==null&&Math.abs(v-best)<1e-9?"best":"";
      return `<td class="${cls}">${fmt(v,r[2])}</td>`;
    }).join("")+"</tr>";
  });
  h+="</tbody></table>";
  document.getElementById("summaryTable").innerHTML=h;
}

function latencyBreakdown(){
  const M=DATA.models[currentModel];
  const columns=[
    ["Avg E2E","avg_e2e_ms"],
    ["请求转发","avg_request_network_ms"],
    ["排队·Prefill","avg_queue_prefill_ms"],
    ["排队·重算","avg_queue_recompute_ms"],
    ["排队·Decode","avg_queue_decode_ms"],
    ["KV 迁移","avg_migration_ms"],
    ["KV 重算","avg_recompute_ms"],
    ["Prefill","avg_prefill_ms"],
    ["Decode","avg_decode_ms"],
    ["响应回传","avg_response_network_ms"],
    ["分项合计","avg_e2e_component_sum_ms"],
  ];
  const best={};
  columns.forEach(c=>{best[c[1]]=Math.min(...POLICIES.map(p=>M[p][c[1]]));});
  let h="<div class='sample-wrap'><table class='sample-table'><thead><tr><th>策略</th>"+
    columns.map(c=>`<th>${c[0]} (ms)</th>`).join("")+"</tr></thead><tbody>";
  POLICIES.forEach(p=>{
    h+=`<tr><td>${PLABEL[p]}</td>`+columns.map(c=>{
      const v=M[p][c[1]];
      const cls=Math.abs(v-best[c[1]])<1e-9?"best":"";
      return `<td class="${cls}">${fmt(v,2)}</td>`;
    }).join("")+"</tr>";
  });
  h+="</tbody></table></div>";
  document.getElementById("latencyBreakdown").innerHTML=h;
}

function queueBreakdown(){
  const M=DATA.models[currentModel];
  let h="<div class='sample-wrap'><table class='sample-table'><thead><tr><th>策略</th><th>执行节点</th>"+
    "<th>请求占比</th><th>总排队 (ms)</th><th>Prefill backlog</th><th>Recompute backlog</th><th>Decode backlog</th>"+
    "</tr></thead><tbody>";
  POLICIES.forEach(p=>{
    const per=M[p].queue_by_exec_node||{};
    Object.entries(per).forEach(([node,q])=>{
      h+=`<tr><td>${PLABEL[p]}</td><td>${NODE_LABEL(node)}</td>`+
        `<td>${fmt(q.request_ratio*100,1)}%</td><td>${fmt(q.avg_queue_ms,3)}</td>`+
        `<td>${fmt(q.avg_queue_prefill_ms,3)}</td><td>${fmt(q.avg_queue_recompute_ms,3)}</td>`+
        `<td>${fmt(q.avg_queue_decode_ms,3)}</td></tr>`;
    });
  });
  h+="</tbody></table></div>";
  document.getElementById("queueBreakdown").innerHTML=h;
}

function latencyDistribution(){
  const M=DATA.models[currentModel];
  const stages=[
    ["请求转发",r=>r.t_network||0],
    ["排队 · Prefill backlog",r=>r.t_queue_prefill||0],
    ["排队 · Recompute backlog",r=>r.t_queue_recompute||0],
    ["排队 · Decode backlog",r=>r.t_queue_decode||0],
    ["KV迁移",r=>r.mode==="migrate"?(r.t_state||0):0],
    ["KV重算",r=>r.mode==="recompute"?(r.t_state||0):0],
    ["Prefill",r=>r.t_prefill||0],
    ["Decode",r=>r.t_decode||0],
    ["响应回传",r=>r.t_return||0],
  ];
  const box=document.getElementById("latencyDistribution");
  box.innerHTML=""; box.className="phase-grid";
  stages.forEach(stage=>{
    const sorted={};
    POLICIES.forEach(p=>{
      sorted[p]=(M[p].records||[]).map(stage[1]).filter(Number.isFinite).sort((a,b)=>a-b);
    });
    const maxValue=Math.max(...POLICIES.map(p=>sorted[p][sorted[p].length-1]||0),.001);
    const W=480,H=250,L=48,R=14,T=12,B=38,plotW=W-L-R,plotH=H-T-B;
    const logMax=Math.log10(1+maxValue/.001);
    const X=v=>L+Math.log10(1+Math.max(v,0)/.001)/logMax*plotW;
    const Y=p=>T+plotH-(p*plotH);
    const s=svg(W,H);
    [0,.25,.5,.75,1].forEach(p=>{
      const y=Y(p);
      s.appendChild(el("line",{x1:L,y1:y,x2:W-R,y2:y,class:"axis"}));
      s.appendChild(txt(L-6,y+3,`${Math.round(p*100)}%`,"axlab","end"));
    });
    [0,.01,.1,1,10,100,1000,10000].filter(v=>v<=maxValue).forEach(v=>{
      const x=X(v);
      s.appendChild(el("line",{x1:x,y1:T,x2:x,y2:T+plotH,class:"axis"}));
      s.appendChild(txt(x,H-15,v===0?"0":String(v),"axlab","middle"));
    });
    POLICIES.forEach(p=>{
      const values=sorted[p];
      if(!values.length) return;
      const points=[];
      for(let i=0;i<=100;i++){
        const q=i/100,idx=Math.min(values.length-1,Math.floor(q*(values.length-1)));
        points.push([X(values[idx]),Y(q)]);
      }
      const path=points.map((pt,i)=>`${i?"L":"M"}${pt[0].toFixed(2)},${pt[1].toFixed(2)}`).join(" ");
      s.appendChild(el("path",{d:path,fill:"none",stroke:PCOLOR[p],"stroke-width":2}));
    });
    const hit=el("rect",{x:L,y:T,width:plotW,height:plotH,fill:"transparent"});
    hit.addEventListener("mousemove",ev=>{
      const rect=s.getBoundingClientRect();
      const rel=(ev.clientX-rect.left)/rect.width*W;
      const log=Math.max(0,Math.min(logMax,(rel-L)/plotW*logMax));
      const value=.001*(Math.pow(10,log)-1);
      let html=`<div class="tip-title">${stage[0]} · ${fmt(value,3)} ms</div>`;
      POLICIES.forEach(p=>{
        const values=sorted[p]; let count=0;
        while(count<values.length&&values[count]<=value) count++;
        html+=`<div class="tip-row"><i class="tip-dot" style="background:${PCOLOR[p]}"></i>`+
          `<span>${PLABEL[p]}: <b>${fmt(count/Math.max(values.length,1)*100,1)}%</b></span></div>`;
      });
      showTip(html,ev.clientX,ev.clientY);
    });
    hit.addEventListener("mouseleave",hideTip);
    s.appendChild(hit);
    const wrap=document.createElement("div");
    wrap.className="phase-chart";
    wrap.innerHTML=`<h4>${stage[0]}</h4>`;
    wrap.appendChild(s); box.appendChild(wrap);
  });
}

function barLabel(s, label, pad, y, barH, bw){
  s.appendChild(txt(pad+bw+5, y+barH-3, label, "barval-out", "start"));
}

function barCharts(){
  const M=DATA.models[currentModel];
  const metrics=[
    ["P99 TTFT (ms)","p99_ttft_ms",1],
    ["跨节点比例 (%)","cross_node_ratio",100],
    ["迁移字节 (MB)","migrate_bytes_mb",1],
    ["owner 切换","owner_switch_count",1],
  ];
  const W=560,titleH=20,barH=15,gap=5,groupGap=16,pad=120,rightReserve=50;
  const barMax=W-pad-rightReserve;
  const rowH=titleH+POLICIES.length*(barH+gap)+groupGap;
  const s=svg(W, metrics.length*rowH+8);
  metrics.forEach((m,mi)=>{
    const y0=mi*rowH+6;
    const title=el("text",{x:0,y:y0+12,class:"axlab",fill:"var(--fg)","font-size":"12"});
    title.textContent=m[0]; s.appendChild(title);
    const vals=POLICIES.map(p=>M[p][m[1]]*m[2]);
    const mx=Math.max(...vals,1e-9);
    POLICIES.forEach((p,pi)=>{
      const v=vals[pi], y=y0+titleH+pi*(barH+gap);
      const bw=Math.max(barMax*v/mx, 1);
      s.appendChild(el("rect",{x:pad,y,width:bw,height:barH,rx:3,fill:PCOLOR[p]}));
      s.appendChild(txt(pad-6,y+barH-3,PLABEL[p],"axlab","end"));
      const label=fmt(v, m[2]===100?1:(m[1]==="migrate_bytes_mb"?0:1));
      barLabel(s, label, pad, y, barH, bw);
    });
  });
  const box=document.getElementById("barCharts"); box.innerHTML=""; box.appendChild(s);
}

function actionDist(){
  const M=DATA.models[currentModel];
  const modes=[["local","#3fb950"],["migrate","#f0883e"],["recompute","#d29922"],["fresh","#8b949e"]];
  const W=560,rowH=46,pad=120;
  const barMax=W-pad-10;
  const s=svg(W,POLICIES.length*rowH+30);
  POLICIES.forEach((p,pi)=>{
    const r=M[p];
    const total=modes.reduce((a,m)=>a+(r[m[0]+"_count"]||0),0)||1;
    const y=pi*rowH+10; let x=pad;
    s.appendChild(txt(pad-6,y+18,PLABEL[p],"axlab","end"));
    modes.forEach(m=>{
      const c=r[m[0]+"_count"]||0, w=barMax*c/total;
      if(w>0.5){
        const rect=el("rect",{x,y:y+6,width:w,height:20,fill:m[1],rx:2});
        s.appendChild(rect);
        const pct=(c/total*100).toFixed(1);
        const tipHtml=`<div class="tip-title">${PLABEL[p]}</div>`+
          `<div class="tip-row"><i class="tip-dot" style="background:${m[1]}"></i>`+
          `<span>${m[0]}: <b>${c}</b> (${pct}%)</span></div>`;
        attachBarHover(rect, tipHtml);
        if(w>=32){
          s.appendChild(txt(x+w/2,y+20,String(c),"barval-in","middle"));
        }
      }
      x+=w;
    });
  });
  let lx=pad,ly=POLICIES.length*rowH+18;
  modes.forEach(m=>{
    s.appendChild(el("rect",{x:lx,y:ly-9,width:10,height:10,fill:m[1],rx:2}));
    s.appendChild(txt(lx+14,ly,m[0],"axlab")); lx+=100;
  });
  const box=document.getElementById("actionDist"); box.innerHTML=""; box.appendChild(s);
}

function bucketize(records,key,nbuckets,dur,agg){
  const b=Array.from({length:nbuckets},()=>[]);
  records.forEach(r=>{
    let i=Math.min(nbuckets-1,Math.floor(r.t/dur*nbuckets));
    if(i<0)i=0; b[i].push(r[key]);
  });
  return b.map(arr=>{
    if(!arr.length) return null;
    if(agg==="p99"){const s=[...arr].sort((a,b)=>a-b);return s[Math.min(s.length-1,Math.round(0.99*(s.length-1)))];}
    return arr.reduce((a,c)=>a+c,0)/arr.length;
  });
}

function lineChart(boxId,seriesByPolicy,dur,yLabel){
  const W=1180,H=300,L=88,R=18,T=14,B=34,labelCol=28;
  const s=svg(W,H);
  let mx=0;
  for(const p of POLICIES) seriesByPolicy[p].forEach(v=>{if(v!=null&&v>mx)mx=v;});
  mx=mx||1;
  const n=seriesByPolicy[POLICIES[0]].length;
  const plotW=W-L-R, plotH=H-T-B;
  const X=i=>L+plotW*(i/(Math.max(n-1,1)));
  const Y=v=>T+plotH*(1-v/mx);
  for(let g=0;g<=4;g++){
    const yy=T+plotH*g/4, val=mx*(1-g/4);
    s.appendChild(el("line",{x1:L,y1:yy,x2:W-R,y2:yy,class:"axis"}));
    s.appendChild(txt(L-10,yy+3,fmt(val, mx<10?2:0),"axlab","end"));
  }
  const mob=meta.mobility_start_frac;
  const mx2=L+plotW*mob;
  s.appendChild(el("line",{x1:mx2,y1:T,x2:mx2,y2:H-B,stroke:"#f85149","stroke-dasharray":"4 4","stroke-width":1}));
  s.appendChild(txt(mx2+4,T+10,"用户移动 →","axlab"));
  for(let g=0;g<=5;g++){
    const xx=L+plotW*g/5;
    s.appendChild(txt(xx,H-B+16,((dur/1000)*g/5).toFixed(0)+"s","axlab","middle"));
  }
  const yMid=T+plotH/2;
  const yLab=el("text",{x:labelCol,y:yMid,class:"axlab","text-anchor":"middle",
    transform:`rotate(-90,${labelCol},${yMid})`});
  yLab.textContent=yLabel; s.appendChild(yLab);
  const cross=el("line",{x1:L,y1:T,x2:L,y2:H-B,stroke:"var(--muted)",
    "stroke-width":1,"stroke-dasharray":"3 3",opacity:0,"pointer-events":"none"});
  s.appendChild(cross);
  const dots={};
  POLICIES.forEach(p=>{
    dots[p]=el("circle",{cx:L,cy:T,r:4,fill:PCOLOR[p],stroke:"var(--bg)","stroke-width":1.5,
      opacity:0,"pointer-events":"none"});
    s.appendChild(dots[p]);
  });
  POLICIES.forEach(p=>{
    const ser=seriesByPolicy[p]; let d=""; let started=false;
    ser.forEach((v,i)=>{ if(v==null)return; d+=(started?"L":"M")+X(i)+" "+Y(v)+" "; started=true; });
    s.appendChild(el("path",{d,fill:"none",stroke:PCOLOR[p],"stroke-width":2,"stroke-linejoin":"round","pointer-events":"none"}));
  });
  const hit=el("rect",{x:L,y:T,width:plotW,height:plotH,fill:"transparent",cursor:"crosshair"});
  s.appendChild(hit);
  hit.addEventListener("mousemove", ev=>{
    const svgRect=s.getBoundingClientRect();
    const relX=(ev.clientX-svgRect.left)/svgRect.width*W;
    let idx=Math.round((relX-L)/(plotW/Math.max(n-1,1)));
    idx=Math.max(0, Math.min(n-1, idx));
    const cx=X(idx);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("opacity", 0.55);
    const tSec=(dur/1000)*idx/Math.max(n-1,1);
    let html=`<div class="tip-title">${tSec.toFixed(1)}s</div>`;
    POLICIES.forEach(p=>{
      const v=seriesByPolicy[p][idx];
      dots[p].setAttribute("cx", cx);
      if(v==null){
        dots[p].setAttribute("opacity", 0);
        html+=`<div class="tip-row"><i class="tip-dot" style="background:${PCOLOR[p]}"></i>`+
          `<span>${PLABEL[p]}: —</span></div>`;
      }else{
        dots[p].setAttribute("cy", Y(v)); dots[p].setAttribute("opacity", 1);
        html+=`<div class="tip-row"><i class="tip-dot" style="background:${PCOLOR[p]}"></i>`+
          `<span>${PLABEL[p]}: <b>${fmt(v, mx<10?2:1)}</b></span></div>`;
      }
    });
    showTip(html, ev.clientX, ev.clientY);
  });
  hit.addEventListener("mouseleave", ()=>{
    hideTip();
    cross.setAttribute("opacity", 0);
    POLICIES.forEach(p=>dots[p].setAttribute("opacity", 0));
  });
  const box=document.getElementById(boxId);
  box.innerHTML=""; box.className="chart-wrap"; box.appendChild(s);
}

function ttftSeries(){
  const M=DATA.models[currentModel], dur=meta.duration_ms, nb=40;
  const series={};
  POLICIES.forEach(p=>{series[p]=bucketize(M[p].records,"ttft",nb,dur,"p99");});
  lineChart("ttftSeries",series,dur,"ms");
}

function stickiness(){
  const M=DATA.models[currentModel], dur=meta.duration_ms, nb=60;
  const series={};
  POLICIES.forEach(p=>{
    const recs=[...M[p].records].sort((a,b)=>a.t-b.t);
    let cum=0; const arr=Array(nb).fill(null);
    // cumulative cross-node count sampled into buckets
    const cumAt=[]; recs.forEach(r=>{cum+=r.cross; cumAt.push([r.t,cum]);});
    for(let i=0;i<nb;i++){
      const t=dur*(i+1)/nb;
      let v=0; for(const [tt,cc] of cumAt){ if(tt<=t) v=cc; else break; }
      arr[i]=v;
    }
    series[p]=arr;
  });
  lineChart("stickiness",series,dur,"累计跨节点请求数");
}

function cumulativeCost(){
  const M=DATA.models[currentModel], dur=meta.duration_ms, nb=60;
  const series={};
  POLICIES.forEach(p=>{
    const recs=[...M[p].records].sort((a,b)=>a.t-b.t);
    let cum=0; const cumAt=[];
    recs.forEach(r=>{cum+=(r.t_network||0)+(r.t_queue||0)+(r.t_state||0); cumAt.push([r.t,cum]);});
    const arr=Array(nb).fill(null);
    for(let i=0;i<nb;i++){
      const t=dur*(i+1)/nb; let v=0;
      for(const [tt,cc] of cumAt){ if(tt<=t) v=cc; else break; }
      arr[i]=v;
    }
    series[p]=arr;
  });
  lineChart("cumCost",series,dur,"累计 ms");
}

function linkUtil(){
  const M=DATA.models[currentModel];
  const links=Object.keys(M.nearest.link_utilization||{});
  const box=document.getElementById("linkUtil");
  if(!links.length){
    box.innerHTML="<span style='color:var(--muted)'>无链路数据</span>";
    return;
  }
  const W=560,titleH=20,barH=15,gap=5,groupGap=16,pad=120,rightReserve=50;
  const barMax=W-pad-rightReserve;
  const rowH=titleH+POLICIES.length*(barH+gap)+groupGap;
  const s=svg(W, links.length*rowH+8);
  links.forEach((lk,li)=>{
    const y0=li*rowH+6;
    const title=el("text",{x:0,y:y0+12,class:"axlab",fill:"var(--fg)","font-size":"12"});
    title.textContent=lk;
    s.appendChild(title);
    const mx=Math.max(...POLICIES.map(p=>(M[p].link_utilization[lk]||{}).total_bytes||0),1);
    POLICIES.forEach((p,pi)=>{
      const u=M[p].link_utilization[lk]||{};
      const bytes=u.total_bytes||0;
      const y=y0+titleH+pi*(barH+gap);
      const bw=Math.max(barMax*bytes/mx, 1);
      s.appendChild(el("rect",{x:pad,y,width:bw,height:barH,rx:3,fill:PCOLOR[p]}));
      s.appendChild(txt(pad-6,y+barH-3,PLABEL[p],"axlab","end"));
      barLabel(s, (bytes/1e6).toFixed(0)+"MB", pad, y, barH, bw);
    });
  });
  box.innerHTML="";
  box.appendChild(s);
}

function migrateReasons(){
  const M=DATA.models[currentModel];
  const reasons=Object.keys(MIGRATE_REASON_LABEL);
  const W=760,rowH=42,L=130,R=20,barW=W-L-R;
  const s=svg(W,POLICIES.length*rowH+16);
  POLICIES.forEach((p,pi)=>{
    const counts=M[p].migrate_reason_counts||{};
    const total=Object.values(counts).reduce((a,b)=>a+b,0);
    let x=L,y=pi*rowH+8;
    s.appendChild(txt(L-8,y+17,PLABEL[p],"axlab","end"));
    reasons.forEach(reason=>{
      const count=counts[reason]||0;
      if(!count||!total) return;
      const width=barW*count/total;
      const rect=el("rect",{x,y,width,height:22,fill:MIGRATE_REASON_COLOR[reason],rx:2});
      s.appendChild(rect);
      attachBarHover(rect,`<div class="tip-title">${PLABEL[p]} · ${MIGRATE_REASON_LABEL[reason]}</div>`+
        `<div>次数 <b>${count}</b> / ${total}</div>`+
        `<div>迁移 ${fmt((M[p].migrate_reason_bytes_mb||{})[reason]||0,2)} MB</div>`);
      if(width>34) s.appendChild(txt(x+width/2,y+15,String(count),"barval-in","middle"));
      x+=width;
    });
    if(!total) s.appendChild(txt(L,y+17,"无 migrate","axlab","start"));
  });
  const rows=[];
  POLICIES.forEach(p=>{
    const counts=M[p].migrate_reason_counts||{};
    const bytes=M[p].migrate_reason_bytes_mb||{};
    Object.keys(counts).forEach(reason=>{
      if(counts[reason]) rows.push(
        `<tr><td>${PLABEL[p]}</td><td>${MIGRATE_REASON_LABEL[reason]||reason}</td>`+
        `<td>${counts[reason]}</td><td>${fmt(bytes[reason]||0,2)}</td></tr>`
      );
    });
  });
  const box=document.getElementById("migrateReasons");
  box.innerHTML=""; box.appendChild(s);
  box.insertAdjacentHTML("beforeend",`<div class="sample-wrap"><table class="sample-table">`+
    `<thead><tr><th>策略</th><th>原因</th><th>次数</th><th>迁移 MB</th></tr></thead>`+
    `<tbody>${rows.join("")||"<tr><td colspan='4'>无 migrate</td></tr>"}</tbody></table></div>`);
}

function initWorkloadCollapse(){
  const btn=document.getElementById("workloadToggle");
  const panel=document.getElementById("workloadPanel");
  const key="cec-dashboard-workload-open";
  const open=localStorage.getItem(key)==="1";
  btn.setAttribute("aria-expanded", open?"true":"false");
  panel.hidden=!open;
  btn.onclick=()=>{
    const next=btn.getAttribute("aria-expanded")!=="true";
    btn.setAttribute("aria-expanded", next?"true":"false");
    panel.hidden=!next;
    localStorage.setItem(key, next?"1":"0");
  };
}

function statCard(k,v){
  return `<div class="stat-item"><div class="k">${k}</div><div class="v">${v}</div></div>`;
}

function workloadHistSvg(dist, title){
  const W=520,padL=28,padR=8,padT=22,padB=34,barGap=2;
  const bins=dist.bins||[];
  const nb=bins.length||1;
  const plotW=W-padL-padR, plotH=108;
  const H=padT+plotH+padB;
  const s=svg(W,H);
  const titleEl=el("text",{x:0,y:12,class:"axlab",fill:"var(--fg)","font-size":"12"});
  titleEl.textContent=title;
  s.appendChild(titleEl);
  const mx=Math.max(...bins.map(b=>b.count),1);
  const bw=plotW/nb;
  const labelStep=nb<=10?1:Math.ceil(nb/8);
  bins.forEach((b,i)=>{
    const h=plotH*b.count/mx;
    const x=padL+i*bw;
    const y=padT+plotH-h;
    s.appendChild(el("rect",{x:x+barGap/2,y,width:Math.max(bw-barGap,1),height:Math.max(h,0),rx:2,fill:"#58a6ff"}));
    if(i%labelStep===0 || i===nb-1){
      s.appendChild(txt(x+bw/2, padT+plotH+14, b.label, "axlab", "middle"));
    }
  });
  s.appendChild(el("line",{x1:padL,y1:padT+plotH,x2:W-padR,y2:padT+plotH,class:"axis"}));
  return s;
}

function buildDistGrid(dists){
  const grid=document.createElement("div");
  grid.className="dist-grid";
  const specs=[
    [dists.session_requests, "Session 请求数", "条/session"],
    [dists.input_tokens, "请求输入 tokens（当前轮次）", "tokens"],
    [dists.output_tokens, "请求输出 tokens", "tokens"],
  ];
  specs.forEach(([dist,title,unit])=>{
    if(!dist || !dist.bins || !dist.bins.length) return;
    const block=document.createElement("div");
    block.className="dist-block";
    block.appendChild(workloadHistSvg(dist, title));
    const stats=document.createElement("div");
    stats.className="dist-stats";
    const suffix=unit?" "+unit:"";
    stats.innerHTML=
      statCard("平均", dist.avg+suffix)+
      statCard("P95", dist.p95+suffix);
    block.appendChild(stats);
    grid.appendChild(block);
  });
  return grid;
}

function workloadLineChart(data, mobStartFrac, mode, chartTitle){
  const W=520,L=32,R=8,T=chartTitle?24:8,B=32,plotH=100;
  const nb=data.nb,dur=data.duration_ms;
  const plotW=W-L-R,H=T+plotH+B;
  const s=svg(W,H);
  if(chartTitle){
    const titleEl=el("text",{x:0,y:12,class:"axlab",fill:"var(--fg)","font-size":"12","font-weight":"650"});
    titleEl.textContent=chartTitle;
    s.appendChild(titleEl);
  }
  let lines=[];
  if(mode==="total"){
    const nodes=Object.keys(data.series||{}).sort((a,b)=>Number(a)-Number(b));
    const vals=Array.from({length:nb},(_,i)=>nodes.reduce((sum,n)=>sum+(data.series[n][i]||0),0));
    lines=[{key:"total",vals,color:"#8b949e"}];
  }else{
    const ni=Number(mode);
    const key=String(mode);
    lines=[{key,vals:data.series[key]||[],color:NODE_COLORS[ni%3]}];
  }
  let mx=1;
  lines.forEach(ln=>ln.vals.forEach(v=>{if(v>mx)mx=v;}));
  const X=i=>L+plotW*(i/Math.max(nb-1,1));
  const Y=v=>T+plotH*(1-v/mx);
  for(let g=0;g<=3;g++){
    const yy=T+plotH*g/3;
    s.appendChild(el("line",{x1:L,y1:yy,x2:W-R,y2:yy,class:"axis"}));
    s.appendChild(txt(L-6,yy+3,fmt(mx*(1-g/3),0),"axlab","end"));
  }
  s.appendChild(el("line",{x1:L,y1:T+plotH,x2:W-R,y2:T+plotH,class:"axis"}));
  const mobX=L+plotW*mobStartFrac;
  s.appendChild(el("line",{x1:mobX,y1:T,x2:mobX,y2:T+plotH,stroke:"#f85149","stroke-dasharray":"4 4","stroke-width":1}));
  lines.forEach(ln=>{
    let d="", started=false;
    ln.vals.forEach((v,i)=>{d+=(started?"L":"M")+X(i)+" "+Y(v)+" "; started=true;});
    s.appendChild(el("path",{d,fill:"none",stroke:ln.color,"stroke-width":2,"stroke-linejoin":"round"}));
  });
  for(let g=0;g<=4;g++){
    const xx=L+plotW*g/4;
    s.appendChild(txt(xx,T+plotH+16,((dur/1000)*g/4).toFixed(0)+"s","axlab","middle"));
  }
  return s;
}

function buildEntryLineCharts(entryData, mobStartFrac){
  const wrap=document.createElement("div");
  const title=document.createElement("div");
  title.className="time-sec-title";
  title.textContent="入口到达（折线）";
  wrap.appendChild(title);
  const grid=document.createElement("div");
  grid.className="time-line-grid";
  const nodes=Object.keys(entryData.series||{}).sort((a,b)=>Number(a)-Number(b));
  nodes.forEach(n=>{
    const b=document.createElement("div");
    b.className="dist-block";
    b.appendChild(workloadLineChart(entryData, mobStartFrac, n, NODE_LABEL(n)));
    grid.appendChild(b);
  });
  const tb=document.createElement("div");
  tb.className="dist-block";
  tb.appendChild(workloadLineChart(entryData, mobStartFrac, "total", "全部入口合计"));
  grid.appendChild(tb);
  wrap.appendChild(grid);
  return wrap;
}

function buildExecLineCharts(execByPolicy, mobStartFrac){
  const wrap=document.createElement("div");
  const title=document.createElement("div");
  title.className="time-sec-title";
  title.textContent="实际处理实例（折线 · 各节点处理请求数）";
  wrap.appendChild(title);
  const policies=[["nearest","Nearest(基线)"],["greedy","Greedy"],["long_term","Long-term"]];
  policies.forEach(([key,label])=>{
    const data=execByPolicy[key];
    if(!data||!data.series) return;
    const sec=document.createElement("div");
    sec.className="policy-block";
    const h=document.createElement("h4");
    h.textContent=label;
    sec.appendChild(h);
    const grid=document.createElement("div");
    grid.className="node-line-grid";
    Object.keys(data.series).sort((a,b)=>Number(a)-Number(b)).forEach(n=>{
      const b=document.createElement("div");
      b.className="dist-block";
      b.appendChild(workloadLineChart(data, mobStartFrac, n, NODE_LABEL(n)));
      grid.appendChild(b);
    });
    sec.appendChild(grid);
    wrap.appendChild(sec);
  });
  return wrap;
}

function buildWorkloadTimeCharts(dists, mobStartFrac){
  const wrap=document.createElement("div");
  if(dists.entry_over_time){
    wrap.appendChild(buildEntryLineCharts(dists.entry_over_time, mobStartFrac));
  }
  if(dists.exec_over_time&&Object.keys(dists.exec_over_time).length){
    wrap.appendChild(buildExecLineCharts(dists.exec_over_time, mobStartFrac));
  }
  return wrap;
}

function pieChartSvg(counts, colorForKey){
  const W=120,H=104,cx=60,cy=52,r=38;
  const s=svg(W,H);
  const keys=Object.keys(counts||{}).sort((a,b)=>Number(a)-Number(b));
  const total=keys.reduce((a,k)=>a+(counts[k]||0),0)||1;
  let ang=-Math.PI/2;
  keys.forEach(k=>{
    const v=counts[k]||0;
    if(v<=0) return;
    const slice=2*Math.PI*v/total;
    const mid=ang+slice/2;
    const x1=cx+r*Math.cos(ang), y1=cy+r*Math.sin(ang);
    ang+=slice;
    const x2=cx+r*Math.cos(ang), y2=cy+r*Math.sin(ang);
    const large=slice>Math.PI?1:0;
    const d=`M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`;
    s.appendChild(el("path",{d,fill:colorForKey(k),stroke:"var(--panel)","stroke-width":1.2}));
    if(slice>=0.12){
      const lx=cx+r*0.58*Math.cos(mid), ly=cy+r*0.58*Math.sin(mid);
      const pct=(100*v/total).toFixed(1);
      const name=NODE_NAME(k);
      s.appendChild(txt(lx,ly-5,name,"pie-slice-n","middle"));
      s.appendChild(txt(lx,ly,String(v),"pie-slice-p","middle"));
      s.appendChild(txt(lx,ly+5,pct+"%","pie-slice-p","middle"));
    }
  });
  const wrap=document.createElement("div");
  wrap.className="pie-block";
  wrap.appendChild(s);
  return wrap;
}

function buildWorkloadPies(summary){
  const entry=summary.by_entry_node||{};
  if(!Object.keys(entry).length) return null;
  const grid=document.createElement("div");
  grid.className="pie-grid";
  grid.appendChild(pieChartSvg(entry,k=>NODE_COLORS[Number(k)%3]));
  return grid;
}

function renderWorkload(){
  const wl=DATA.workload;
  const modelWl=wl.by_model[currentModel]||{summary:{},samples:[]};
  const s=modelWl.summary||{};
  const g=(wl.groups||[]).filter(x=>x.model_name===currentModel);

  let groupsHtml="";
  const byGroup=s.by_group||{};
  const totalReqs=s.num_requests||0;
  if(g.length){
    groupsHtml=`<div class="subsec">负载分组配置（${currentModel}）</div>`+
      `<table><thead><tr><th>配置组</th><th>SLA (ms)</th><th>共享 prefix</th><th>入口配置</th>`+
      `<th>请求数</th><th>占比</th><th>平均轮数</th><th>图像尺寸</th></tr></thead><tbody>`+
      g.map(x=>{
        const cnt=byGroup[x.name];
        const pct=(cnt!=null&&totalReqs)?(100*cnt/totalReqs).toFixed(1)+"%":"—";
        const entry=x.entry_mode==="ratios"?`${x.entry_ratios?.join(":")||"—"} / total ${x.concurrency}`:(x.entry_concurrency?.join(",")||"—");
        return `<tr><td>${x.name}</td><td>${x.sla_ms}</td><td>${x.shared_prefix_tokens}</td>`+
          `<td>${entry}</td><td>${cnt??"—"}</td><td>${pct}</td><td>${x.turns_mean}</td>`+
          `<td>${x.image_size[0]?x.image_size[0]+"×"+x.image_size[1]:"—"}</td></tr>`;
      }).join("")+
      `</tbody></table>`;
  }

  const stats=[
    ["请求数", s.num_requests??"—"],
    ["Session 数", s.num_sessions??"—"],
    ["首轮占比", s.first_turn_ratio!=null?(s.first_turn_ratio*100).toFixed(1)+"%":"—"],
    ["移动窗口后请求", s.after_mobility_start_ratio!=null?(s.after_mobility_start_ratio*100).toFixed(1)+"%":"—"],
    ["移动切换数", s.mobility_switched_count??"—"],
    ["移动切换占比", s.mobility_switched_ratio!=null?(s.mobility_switched_ratio*100).toFixed(1)+"%":"—"],
    ["累计 KV tokens", s.total_kv_tokens??"—"],
  ];

  const samples=modelWl.samples||[];
  const sampleRows=samples.map(r=>{
    const moved=r.mobility_switched?`<span class="tag-pill moved">是</span>`:"否";
    return `<tr>`+
      `<td>${r.tag||""}</td>`+
      `<td>#${r.request_id}</td>`+
      `<td>${r.session_id}</td>`+
      `<td>${r.arrival_s}s</td>`+
      `<td>${NODE_LABEL(r.entry_node)}</td>`+
      `<td>${NODE_LABEL(r.home_node)}</td>`+
      `<td>${moved}</td>`+
      `<td>${r.group_name}</td>`+
      `<td>${r.sla_ms}</td>`+
      `<td>${r.turn_index}</td>`+
      `<td>${r.input_tokens}</td>`+
      `<td>${r.output_len}</td>`+
      `<td>${r.prefix_tokens}</td>`+
      `<td>${r.is_session_first?"是":"否"}</td>`+
      `</tr>`;
  }).join("");

  document.getElementById("workloadContent").innerHTML=
    `<div class="hint">轨迹 seed=${wl.seed} · 时长 ${(wl.duration_ms/1000).toFixed(0)}s · `+
    `移动 ${(wl.mobility_ratio*100).toFixed(0)}% (${wl.mobility_granularity}, 起始于 ${(wl.mobility_start_frac*100).toFixed(0)}%)</div>`+
    `<div class="subsec">统计摘要（${currentModel}）</div>`+
    `<div class="stats-grid">${stats.map(x=>statCard(x[0],x[1])).join("")}</div>`+
    groupsHtml+
    `<div class="subsec">请求样例（${samples.length} 条，含前序 / 移动 / 多轮 session）</div>`+
    `<div class="sample-wrap"><table class="sample-table"><thead><tr>`+
    `<th>标签</th><th>ID</th><th>Session</th><th>到达</th><th>入口</th><th>归属</th>`+
    `<th>移动</th><th>配置组</th><th>SLA</th><th>轮次</th><th>输入</th><th>输出</th><th>Prefix</th><th>首轮</th>`+
    `</tr></thead><tbody>${sampleRows||`<tr><td colspan="14" style="text-align:center;color:var(--muted)">无样例</td></tr>`}</tbody></table></div>`+
    `<div class="subsec">分布（${currentModel}）</div>`+
    `<div id="workloadDists"></div>`+
    `<div class="subsec">入口节点占比（${currentModel}）</div>`+
    `<div class="hint">节点编号与拓扑对应：每个节点既是请求入口客户端，也是推理实例。`+
    `Session 初始 <code>home_node</code> 随机分配；移动窗口后部分请求改从其他入口进入。</div>`+
    `<div id="workloadNodeLegend"></div>`+
    `<div id="workloadPies"></div>`+
    `<div class="subsec">到达与处理时序（${currentModel}）</div>`+
    `<div class="hint">按到达时间分桶；竖线为移动起始时刻。入口为轨迹原始数据；处理实例为各策略下 exec 节点统计。`+
    `折线标题即入口/执行节点：<b>${NODE_INFO.map(x=>x.label).join(" · ")}</b>。</div>`+
    `<div id="workloadTimeCharts"></div>`;

  const distBox=document.getElementById("workloadDists");
  distBox.innerHTML="";
  const dists=modelWl.distributions||{};
  if(dists.session_requests || dists.input_tokens || dists.output_tokens){
    distBox.appendChild(buildDistGrid(dists));
  }else{
    distBox.innerHTML=`<span class="hint">无分布数据</span>`;
  }

  const timeBox=document.getElementById("workloadTimeCharts");
  timeBox.innerHTML="";
  if(dists.entry_over_time || (dists.exec_over_time&&Object.keys(dists.exec_over_time).length)){
    timeBox.appendChild(buildWorkloadTimeCharts(dists, wl.mobility_start_frac));
  }

  const pieBox=document.getElementById("workloadPies");
  pieBox.innerHTML="";
  const legendBox=document.getElementById("workloadNodeLegend");
  if(legendBox) legendBox.innerHTML=nodeLegendHtml();
  const pies=buildWorkloadPies(s);
  if(pies) pieBox.appendChild(pies);
}

function render(){
  tabs(); legend(); cards(); renderWorkload(); summaryTable(); latencyBreakdown(); queueBreakdown(); barCharts();
  latencyDistribution(); actionDist(); migrateReasons(); ttftSeries(); stickiness(); cumulativeCost(); linkUtil();
}
initTheme();
initWorkloadCollapse();
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--mobility-granularity", choices=("request", "session", "markov"))
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    html_path = os.path.join(out_dir, "dashboard.html")
    json_path = os.path.join(out_dir, "metrics.json")

    experiment = None
    if args.config:
        print(f"loading config: {args.config}")
        experiment = load_config(args.config)
    if args.mobility_granularity:
        experiment = experiment or default_config()
        experiment.workload.mobility_granularity = args.mobility_granularity

    print("running experiments (policies x models on shared trace)...")
    data = run_experiments(experiment)
    export_json(data, json_path)
    render_html(data, html_path)

    for model_name, per in data["models"].items():
        print(f"\n[{model_name}]")
        print(f"  {'policy':<14}{'p99_ttft':>9}{'xnode%':>8}{'migrMB':>8}{'ownsw':>7}")
        for pol, m in per.items():
            print(f"  {pol:<14}{m['p99_ttft_ms']:9.1f}"
                  f"{m['cross_node_ratio']*100:8.1f}{m['migrate_bytes_mb']:8.1f}"
                  f"{m['owner_switch_count']:7d}")

    print(f"\nJSON  -> {json_path}")
    print(f"HTML  -> {html_path}")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(html_path))
