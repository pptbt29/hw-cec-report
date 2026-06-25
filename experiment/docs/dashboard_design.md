# Metrics 看板设计文档

## 1. 设计目标

实验是**离线回放**式的（同一条轨迹回放多种策略），因此选择“跑完即出一份可分享的指标看板”，
而非在线实时监控。看板需要：

- 在**同一条请求轨迹**上回放四种策略（覆盖所有模型），保证可比；
- 既给**聚合指标**（汇总表、对比柱），也给**时间维度**信息（TTFT 时间序列、黏附曲线）；
- 产物**自包含、无第三方依赖**：数据内联进单个 HTML，可直接发给他人或在报告中引用；
- 同时导出 `metrics.json` 便于二次分析。

## 2. 数据采集（`router.simulate_trace(collect_records=True)`）

在原有聚合指标基础上扩展（向后兼容）：

- 分位数：`p50/p95/p99 TTFT`、`avg/p95 E2E`；
- 动作计数：`local/fresh/migrate/recompute_count`；
- KV 统计：`owner_switch_count`、`migrate_bytes_mb`；
- `link_utilization`：各链路传输字节/占用/并发（来自 `NetworkSimulator.link_utilization`）；
- `records`：逐请求 `{t, ttft, e2e, mode, entry, exec, cross, sla_violation, infeasible, priority, moved}`。

## 3. 实验编排（`dashboard.run_experiments`）

```
requests = DataGenerator(config).generate()      # 一条共享轨迹
for model in 出现的模型:
    for policy in Policy:                         # 每策略独立 network，避免利用率串扰
        simulate_trace(policy, requests, model, hw, net, collect_records=True)
```

输出结构：

```
{
  "meta": {hardware, num_nodes, staleness_ms, duration_ms, mobility_*, total_requests},
  "models": { <model>: { <policy>: <metrics+records+link_utilization> } }
}
```

## 4. 渲染（`dashboard.render_html`）

把上面的 dict 用 `json.dumps` 内联进 HTML 模板的 `const DATA = ...`，前端纯 JS + 内联 SVG 绘制，
不引入任何外部库/字体/网络请求。包含可视化：

| 区块 | 内容 | 解读 |
| --- | --- | --- |
| KPI 卡片 | 请求数、Greedy/LT+KV 的 P99 TTFT、跨节点改善、迁移字节改善、owner 切换 | 一眼看收益 |
| 汇总对比表 | 11 项指标 × 4 策略，逐列高亮最优 | 全面对比 |
| 关键指标柱状 | P99 TTFT / 跨节点% / 迁移字节 / owner 切换 | 越低越好 |
| 动作分布 | local/migrate/recompute/fresh 堆叠条 | 策略行为差异 |
| P99 TTFT 时间序列 | 按到达时间分桶的滚动 P99，标注移动时刻 | 服务质量随时间 |
| 累计跨节点曲线 | 各策略累计跨节点请求数 | **状态黏附**：Greedy 移动后更陡 |
| 链路利用率 | 100G/25G 各策略传输字节 | 迁移落在哪条链路 |

顶部 model tab 切换模型；颜色固定映射 nearest/greedy/long_term/long_term_kv。

## 5. 使用

```bash
python -m sim.dashboard            # -> output/dashboard.html + metrics.json
python -m sim.dashboard --open     # 生成后用默认浏览器打开
```

`demo.py` 末尾也会生成一次看板。`output/` 已加入 `.gitignore`（生成物不入库）。

## 6. 局限与扩展

- 当前是**离线回放后**的看板，不是在线实时监控；若需实时，可让事件循环周期性 append 到 JSON 并配合自动刷新页面。
- 时间序列按到达时间分桶近似，不建模并发 decode 的真实时刻；与 `simulate_trace` 的简化一致。
