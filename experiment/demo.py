"""End-to-end demo wiring all simulation components together.

Run: python demo.py
"""

from sim import (
    ComputeSimulator,
    DataGenerator,
    GlobalKVDirectory,
    KVCacheStore,
    NetworkSimulator,
    Policy,
    WorkloadConfig,
    default_topology,
    get_hardware,
    get_model,
    list_models,
    simulate_trace,
)
from sim.kv_cache import make_blocks


def show_large_models() -> None:
    print("== large models ==")
    for name in list_models():
        m = get_model(name)
        print(
            f"{name:<24} type={m.model_type.value:<3} layers={m.num_layers:<3} "
            f"kv_heads={m.num_kv_heads:<3} KV/token={m.kv_bytes_per_token()/1024:6.1f}KB "
            f"out_dist={m.default_output_dist.kind}"
        )


def show_compute() -> None:
    print("\n== compute simulator (A800T-A2) ==")
    hw = get_hardware("A800T-A2")
    for name in list_models():
        sim = ComputeSimulator(get_model(name), hw)
        pf = sim.estimate_prefill(1024)
        dec = sim.estimate_decode(gen_tokens=64, ctx_len=1024, batch_size=8)
        mem = sim.memory_usage(resident_tokens=200_000, batch_tokens=2048)
        print(
            f"{name:<24} prefill(1024)={pf.prefill_ms:6.1f}ms[{pf.bound}] "
            f"decode(64,b8)={dec.total_ms:7.1f}ms[{dec.bound}] "
            f"mem_used={mem['used']/1e9:5.1f}GB free={mem['free']/1e9:5.1f}GB"
        )


def show_network() -> None:
    print("\n== network edges (3-node topology) ==")
    net = NetworkSimulator(default_topology())
    payload = 200 * 1024 * 1024
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        ms = net.transfer_time_ms(a, b, payload, contention=False)
        print(f"  {net.topology.link(a, b).name:<10} 200MB: {ms:7.2f} ms")


def show_offloading_decision() -> None:
    print("\n== local / migrate / recompute decision (moved request) ==")
    hw = get_hardware("A800T-A2")
    net = NetworkSimulator(default_topology())
    directory = GlobalKVDirectory(num_nodes=3)
    model = get_model("CodeLlama34B")
    sim = ComputeSimulator(model, hw)

    # session context owned by node A (0)
    ctx_tokens = 2048
    blocks = make_blocks(model, "s1", "CodeLlama34B:s1", ctx_tokens, owner=0)
    store_a = KVCacheStore(0, model, capacity_bytes=20e9)
    store_a.insert(blocks, t_now=0.0)
    for b in blocks:
        directory.register(0, b)
    hashes = [b.block_hash for b in blocks]

    recompute_ms = sim.recompute_time_ms(ctx_tokens)
    print(f"  context={ctx_tokens} tok, KV owner=A. request entry moved.")
    print(f"  recompute prefix @ any node : {recompute_ms:7.2f} ms")
    for dst, label in [(0, "A(local)"), (1, "B(100G)"), (2, "C(25G)")]:
        plan = directory.plan_migration(hashes, dst=dst, net=net)
        if plan.is_local:
            print(f"  dst={label:<9} local hit, T_state ~= 0 ms")
        else:
            best = min(recompute_ms, plan.transfer_ms)
            choice = "migrate" if plan.transfer_ms <= recompute_ms else "recompute"
            print(
                f"  dst={label:<9} migrate={plan.transfer_ms:7.2f}ms "
                f"(src={plan.src}, {plan.bytes_to_move/1e6:.0f}MB) -> choose {choice}"
            )


def show_workload() -> None:
    print("\n== data generator ==")
    gen = DataGenerator(WorkloadConfig.default_experiment())
    requests = gen.generate()
    s = gen.summary(requests)
    print(
        f"  {s['num_requests']} requests / {s['num_sessions']} sessions, "
        f"mobility={s['mobility_ratio']:.1%}, "
        f"avg_in={s['input_tokens_avg']:.0f} avg_out={s['output_len_avg']:.0f}"
    )


def show_router_policies() -> None:
    print("\n== router: 4 policies on the same CodeLlama34B trace ==")
    model = get_model("CodeLlama34B")
    hw = get_hardware("A800T-A2")
    requests = DataGenerator(WorkloadConfig.default_experiment()).generate()
    print(f"  {'policy':<14}{'p99_ttft':>9}{'xnode%':>8}{'migr':>6}"
          f"{'recomp':>7}{'ownsw':>7}{'migMB':>9}")
    for pol in Policy:
        net = NetworkSimulator(default_topology())
        m = simulate_trace(pol, requests, model, hw, net)
        print(
            f"  {m['policy']:<14}{m['p99_ttft_ms']:9.1f}{m['cross_node_ratio']*100:8.1f}"
            f"{m['migrate_count']:6d}{m['recompute_count']:7d}"
            f"{m['owner_switch_count']:7d}{m['migrate_bytes_mb']:9.1f}"
        )
    print("  (greedy 把请求吸到 KV 所在的远端节点 -> 高 xnode、状态黏附;")
    print("   long_term 更早把 KV 迁向未来入口 -> 更低 xnode、更少迁移字节)")


def show_dashboard() -> None:
    import os

    from sim import export_json, render_html, run_experiments

    print("\n== metrics dashboard ==")
    data = run_experiments()
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    html = render_html(data, os.path.join(out_dir, "dashboard.html"))
    export_json(data, os.path.join(out_dir, "metrics.json"))
    print(f"  generated -> {html}")
    print("  open it in a browser, or run: python -m sim.dashboard --open")


if __name__ == "__main__":
    show_large_models()
    show_compute()
    show_network()
    show_offloading_decision()
    show_workload()
    show_router_policies()
    show_dashboard()
