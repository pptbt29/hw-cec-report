"""Run the customer acceptance workload at baseline and 150% concurrency.

The strict comparison is Nearest at the contractual baseline concurrency
against Long-term+KV at 1.5x concurrency.  All five executable policies are
also run at both loads so that load scaling and routing-policy gains remain
separable.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
from typing import Dict, Iterable, List

from .config import DEFAULT_CONFIG_PATH, load_config
from .dashboard import export_json, render_html, run_experiments


POLICIES = ["nearest", "greedy", "greedy_kv", "long_term", "long_term_kv"]
SEEDS = [0, 1, 2, 3, 4]
BASELINE_CONCURRENCY = {
    ("CodeLlama34B", "high"): 24,
    ("CodeLlama34B", "normal"): 96,
    ("Qwen2-VL-7B-Instruct", "default"): 24,
}
TARGET_CONCURRENCY = {
    ("CodeLlama34B", "high"): 36,
    ("CodeLlama34B", "normal"): 144,
    ("Qwen2-VL-7B-Instruct", "default"): 36,
}
ENTRY_RATIOS = [1.0, 1.0, 2.0]


def percentile(values: Iterable[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
    return float(values[index])


def build_config(seed: int, target: bool):
    cfg = copy.deepcopy(load_config(DEFAULT_CONFIG_PATH))
    cfg.models = [
        model for model in cfg.models
        if model.name in ("CodeLlama34B", "Qwen2-VL-7B-Instruct")
    ]
    cfg.workload.groups = [
        group for group in cfg.workload.groups
        if group.model_name in ("CodeLlama34B", "Qwen2-VL-7B-Instruct")
    ]
    cfg.workload.seed = seed
    cfg.workload.mobility_start_frac = 0.5
    cfg.workload.mobility_ratio = 0.2
    cfg.workload.mobility_granularity = "request"
    cfg.router.gamma = 1.0
    cfg.policies = list(POLICIES)
    concurrency = TARGET_CONCURRENCY if target else BASELINE_CONCURRENCY
    for group in cfg.workload.groups:
        group.entry_mode = "ratios"
        group.entry_concurrency = None
        group.entry_ratios = list(ENTRY_RATIOS)
        group.concurrency = concurrency[(group.model_name, group.name)]
    return cfg


def group_metrics(result: Dict) -> Dict[str, Dict]:
    groups: Dict[str, Dict] = {}
    for record in result.get("records", []):
        group = groups.setdefault(record["group_name"], {
            "accepted": 0, "ttft": [], "e2e": [], "violations": 0,
        })
        group["accepted"] += 1
        group["ttft"].append(float(record["ttft"]))
        group["e2e"].append(float(record["e2e"]))
        group["violations"] += int(record["sla_violation"])
    rejected: Dict[str, int] = {}
    for record in result.get("rejection_records", []):
        name = record["group_name"]
        rejected[name] = rejected.get(name, 0) + 1
    summary = {}
    for name in sorted(set(groups) | set(rejected)):
        group = groups.get(name, {
            "accepted": 0, "ttft": [], "e2e": [], "violations": 0,
        })
        accepted = group["accepted"]
        rejection_count = rejected.get(name, 0)
        summary[name] = {
            "accepted": accepted,
            "rejected": rejection_count,
            "offered": accepted + rejection_count,
            "avg_e2e_ms": (
                sum(group["e2e"]) / accepted if accepted else 0.0
            ),
            "p99_ttft_ms": percentile(group["ttft"], 99),
            "sla_violation_ratio": (
                group["violations"] / accepted if accepted else 0.0
            ),
            "admission_rejected_ratio": (
                rejection_count / max(accepted + rejection_count, 1)
            ),
        }
    return summary


def compact_run(data: Dict) -> Dict:
    compact = {"meta": data["meta"], "models": {}}
    for model_name, policies in data["models"].items():
        compact["models"][model_name] = {}
        for policy in POLICIES:
            result = policies[policy]
            compact["models"][model_name][policy] = {
                key: result[key]
                for key in (
                    "offered_requests", "accepted_requests",
                    "admission_rejected_count", "admission_rejected_ratio",
                    "avg_e2e_ms", "p50_ttft_ms", "p99_ttft_ms",
                    "sla_violation_ratio",
                    "avg_queue_ms", "avg_state_ms", "migrate_bytes_mb",
                    "placement_completed_bytes_mb",
                )
            }
            compact["models"][model_name][policy]["groups"] = group_metrics(
                result
            )
    return compact


def mean_ci95(values: List[float]) -> Dict[str, float]:
    mean = statistics.fmean(values)
    # Student-t critical value for n=5.  The runner defaults to five seeds.
    half = 0.0
    if len(values) > 1:
        critical = 2.776 if len(values) == 5 else 1.96
        half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci95_half": half, "min": min(values), "max": max(values)}


def aggregate(runs: List[Dict]) -> Dict:
    output: Dict = {"models": {}}
    model_names = runs[0]["models"]
    for model_name in model_names:
        output["models"][model_name] = {}
        for policy in POLICIES:
            rows = [run["models"][model_name][policy] for run in runs]
            policy_out = {}
            for key in (
                "avg_e2e_ms", "p50_ttft_ms", "p99_ttft_ms",
                "sla_violation_ratio",
                "admission_rejected_ratio", "avg_queue_ms", "avg_state_ms",
            ):
                policy_out[key] = mean_ci95([float(row[key]) for row in rows])
            policy_out["groups"] = {}
            for group_name in rows[0]["groups"]:
                policy_out["groups"][group_name] = {}
                for key in (
                    "avg_e2e_ms", "p99_ttft_ms", "sla_violation_ratio",
                    "admission_rejected_ratio",
                ):
                    policy_out["groups"][group_name][key] = mean_ci95([
                        float(row["groups"][group_name][key]) for row in rows
                    ])
            output["models"][model_name][policy] = policy_out
    return output


def markdown_report(baseline: Dict, target: Dict, seed_count: int) -> str:
    lines = [
        "# 请求调度算法验收实验",
        "",
        "严格验收比较：Nearest@基线并发 vs Long-term+KV@150%并发；"
        f"数值为 {seed_count} 个随机种子的均值，括号内为 95% CI 半宽。",
        "",
        "| 模型 | 基线 Nearest avg E2E | 150% LT+KV avg E2E | 降幅 | 目标 |",
        "|---|---:|---:|---:|---|",
    ]
    for model in baseline["models"]:
        b = baseline["models"][model]["nearest"]["avg_e2e_ms"]
        t = target["models"][model]["long_term_kv"]["avg_e2e_ms"]
        reduction = (b["mean"] - t["mean"]) / b["mean"]
        passed = reduction >= 0.20
        lines.append(
            f"| {model} | {b['mean']:.2f} ± {b['ci95_half']:.2f} ms | "
            f"{t['mean']:.2f} ± {t['ci95_half']:.2f} ms | "
            f"{reduction * 100:.2f}% | {'PASS' if passed else 'FAIL'} |"
        )
    lines += ["", "## 150% 并发下同负载策略比较", ""]
    for model in target["models"]:
        lines += [
            f"### {model}", "",
            "| 策略 | avg E2E (ms) | P99 TTFT (ms) | 已接纳 SLA 违约率 | 准入拒绝率 |",
            "|---|---:|---:|---:|---:|",
        ]
        for policy in POLICIES:
            row = target["models"][model][policy]
            lines.append(
                f"| {policy} | {row['avg_e2e_ms']['mean']:.2f} | "
                f"{row['p99_ttft_ms']['mean']:.2f} | "
                f"{row['sla_violation_ratio']['mean'] * 100:.2f}% | "
                f"{row['admission_rejected_ratio']['mean'] * 100:.2f}% |"
            )
        lines.append("")
    lines += ["## 150% 并发下分业务 SLA", ""]
    group_sla = {
        ("CodeLlama34B", "high"): 150.0,
        ("CodeLlama34B", "normal"): 500.0,
        ("Qwen2-VL-7B-Instruct", "default"): 500.0,
    }
    lines += [
        "| 模型/业务 | SLA | LT+KV P99 TTFT 均值 | 各 seed 最差 P99 | 判定 |",
        "|---|---:|---:|---:|---|",
    ]
    for (model, group), sla in group_sla.items():
        p99 = target["models"][model]["long_term_kv"]["groups"][group]["p99_ttft_ms"]
        lines.append(
            f"| {model}/{group} | {sla:.0f} ms | {p99['mean']:.2f} ms | "
            f"{p99['max']:.2f} ms | {'PASS' if p99['max'] <= sla else 'FAIL'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "output", "acceptance"
    )
    os.makedirs(out_dir, exist_ok=True)
    scenario_runs = {"baseline": [], "target_150pct": []}
    for seed in seeds:
        for scenario, target in (("baseline", False), ("target_150pct", True)):
            print(f"running {scenario}, seed={seed}", flush=True)
            data = run_experiments(build_config(seed, target), args.workers)
            scenario_runs[scenario].append(compact_run(data))
            if seed == seeds[0]:
                export_json(data, os.path.join(out_dir, f"{scenario}_seed{seed}.json"))
                render_html(data, os.path.join(out_dir, f"{scenario}_seed{seed}.html"))
    aggregates = {
        scenario: aggregate(runs) for scenario, runs in scenario_runs.items()
    }
    output = {
        "seeds": seeds,
        "baseline_concurrency": {
            "CodeLlama34B/high": 24, "CodeLlama34B/normal": 96,
            "Qwen2-VL-7B-Instruct/default": 24,
        },
        "target_concurrency": {
            "CodeLlama34B/high": 36, "CodeLlama34B/normal": 144,
            "Qwen2-VL-7B-Instruct/default": 36,
        },
        "entry_ratios": ENTRY_RATIOS,
        "runs": scenario_runs,
        "aggregate": aggregates,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    report = markdown_report(
        aggregates["baseline"], aggregates["target_150pct"], len(seeds)
    )
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
