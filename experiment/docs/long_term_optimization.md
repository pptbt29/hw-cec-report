# Long-term Router 与 KV Cache 管理联合优化方案

## 1. 目标与核心结论

本文说明：在什么条件下、通过哪些 Router 和 KV Cache 管理策略，可以让 Long-term 稳定且显著
优于 Greedy。

Greedy 对每个请求选择当前预测 E2E 最小的动作：

```text
a_greedy = argmin_a current_cost(s, a)
```

Long-term 允许当前请求承担少量额外成本，以换取后续请求的累计收益：

```text
a_long_term = argmin_a [
    current_cost(s, a)
    + gamma * expected_future_cost(s', a)
]
```

Long-term 要真正优于 Greedy，必须同时满足：

```text
未来状态可预测
× 后续收益可持续
× 一次性本地化成本足够低
> 预测误差 + 当前额外成本
```

对应的 break-even 条件为：

```text
一次性迁移/重算/复制成本
<
未来多轮请求的 [
    远程通信节省
  + 排队节省
  + KV 命中和重算节省
  + 淘汰风险降低
  + SLA 违约损失降低
]
```

因此，Long-term 不能只优化 Router，也不能只优化 KV Cache Manager：

- Router 决定“是否值得提前投入、状态应该放在哪里”。
- KV Cache Manager 决定“怎样以最低成本实现该状态布局”。

## 2. 当前场景为什么难以显著胜过 Greedy

当前三节点 LLM 模拟具有以下特征：

- 请求和响应 payload 较小，转发通信通常只有 0.01-0.1 ms。
- Decode 占完整 E2E 的绝大部分，而且不同 Router 的 Decode 基本相同。
- KV 状态较大，完整迁移可能比多轮 RPC 通信更贵。
- 迁移会保留源节点副本；节点少、显存充足时，KV 很快扩散到多个节点。
- Greedy 本身会在即时成本划算时迁移，因此并非完全不管理 KV。
- 当前没有真实的后台预迁移、continuous batching 调度和未来队列预测。

在该场景中，Long-term 即使降低了跨节点比例，也可能被迁移和排队成本抵消。完整 Avg E2E
又会被固定 Decode 稀释，因此只能看到非常小的差异。

这不是 Long-term 理论失效，而是当前工作负载中可供长期优化利用的空间有限。

## 3. Router 层优化

### 3.1 优化目标

Router 不应只优化完整 Avg E2E，而应显式优化可被路由影响的成本：

```text
routing_cost =
    request_network
  + queue
  + state_acquisition
  + incremental_prefill
  + response_network
  + sla_penalty
  + eviction_and_recompute_risk
```

完整 E2E 仍然保留为最终用户指标，但策略决策和归因应重点使用 routing_cost、TTFT 和 SLA。

### 3.2 未来请求建模

Long-term 至少需要以下 session 状态：

```text
当前入口和 home
移动模式及其概率
Markov 驻留年龄和驻留阈值
配置组期望剩余轮数
预计下一轮到达时间
当前 KV block 布局
各节点排队和显存压力
链路带宽、RTT 和 contention
```

未来入口应按概率分布预测，而不是固定成某一个节点：

- `request`：每轮相对 home 独立抖动。
- `session`：一旦迁移，后续请求持续从新入口进入。
- `markov`：根据当前入口、驻留年龄、驻留阈值和迁移概率逐轮传播。

### 3.3 正确计算 FutureCost

对候选执行节点 `e`，先计算继续远端服务的成本：

```text
KeepCost(e) =
    sum_k E_entry[
        request_network(entry_k, e)
      + response_network(e, entry_k)
    ]
```

再计算未来只进行一次状态本地化的成本：

```text
RelocateCost(e, d) =
    min(
        migrate_once(e, d),
        recompute_once(d),
        hybrid_state_acquisition(e, d)
    )
  + KeepCost(d)
```

最终：

```text
FutureCost(e) = min(
    KeepCost(e),
    min_d RelocateCost(e, d)
)
```

完整 KV 迁移只能计算一次，不能乘以 remaining。remaining 应来自配置组的 `turns_mean`
或在线 session 生存模型，而不是全局固定值。

### 3.4 未来队列预测

LLM 请求/响应通信很小时，未来排队是 Long-term 最可能形成明显收益的来源。

每个节点需要维护短视窗负载预测：

```text
queue_next(node) =
    current_backlog
  + expected_arrival_work
  + sticky_sessions_after_placement
  - expected_service_capacity
```

其中 `expected_arrival_work` 应按 Prefill、Recompute 和 Decode batch demand 分开预测。Router 的
状态布局决策也会改变未来请求的粘附方向，因此必须把“本次放置导致的未来流量”纳入预测。

不能简单把当前 queue 乘以 remaining。更合理的实现包括：

- 基于未来时间桶的 workload forecast。
- 基于 EWMA 的每模型到达率和 service demand。
- 基于 continuous batching 的可服务 token 数，而不是单请求串行时长。
- 对预测误差增加 safety margin。

### 3.5 不确定性与自适应回退

Long-term 不应该在预测收益很小时强行改变 Greedy 决策。建议增加置信门：

```text
predicted_gain =
    greedy_future_cost
  - long_term_future_cost
  - extra_current_cost

仅当 predicted_gain > uncertainty_margin 时采用 Long-term，
否则回退 Greedy。
```

`uncertainty_margin` 可由历史预测误差、移动模式和剩余轮数决定：

- request 抖动、短 session：margin 较高，更接近 Greedy。
- session/markov 长期驻留：margin 较低，允许提前布局。
- mobility predictor 置信度低：禁止定向预迁移，只允许低成本复制。

### 3.6 主动预迁移

仅在请求到达时迁移，迁移耗时会直接进入该请求 E2E。更强的 Long-term 需要后台动作：

- 在上一轮 Decode 时预取未来需要的 blocks。
- 利用节点和链路空闲窗口迁移。
- 对后台流量限速，避免影响在线 RPC。
- 用户真正移动后只同步增长的 KV tail。
- 预测置信度不足时复制而不是切换 owner。

理想情况下：

```text
visible_prefetch_cost =
max(prefetch_time - available_bubble, 0)
```

当迁移被 bubble 完全覆盖时，Long-term 可以同时避免首次请求迁移和后续远程通信。

## 4. KV Cache Manager 层优化

### 4.1 Block 级动作空间

KV Manager 应把每个 block 的动作扩展为：

```text
KEEP
REPLICATE
MOVE_OWNER
MIGRATE
RECOMPUTE
EVICT
PREFETCH
```

Router 输出目标状态布局和时限，KV Manager 负责生成最低成本执行计划。

### 4.2 副本价值与准入

当前迁移后长期保留源副本，三节点环境很容易让所有节点都有 KV，既降低 Long-term 的区分度，
也可能在真实有限显存中挤压其他 session。

每个副本应计算净价值：

```text
replica_value(block, node) =
    expected_future_hits
    * (
        remote_network_saved
      + recompute_saved
      + sla_penalty_saved
    )
  - copy_cost
  - memory_opportunity_cost
  - eviction_risk_to_other_blocks
```

只有价值为正的副本才能准入。低价值副本应及时淘汰，热共享 prefix 可以保留多个副本，
session 私有 tail 则限制复制因子。

### 4.3 成本感知淘汰

淘汰不能只使用 LRU。建议淘汰分数综合：

```text
eviction_score =
    recency
  + expected_reuse
  + recompute_cost
  + remote_restore_cost
  + replica_scarcity
  - block_size_penalty
```

原则：

- 只有一个副本且重算昂贵的 block 更难淘汰。
- 多副本、低复用、可从 100G 邻居恢复的 block 优先淘汰。
- 共享 prefix 与私有 tail 使用不同保留策略。

### 4.4 拓扑感知与多源迁移

迁移计划需要按 block 选择源节点，而不是整段 KV 使用同一源：

```text
block_1: A -> C
block_2: B -> C
block_3: local at C
```

支持多链路并行后：

```text
total_migration_time = max(per_link_transfer_time)
```

规划时还需考虑：

- 100G/25G 链路差异。
- 当前链路 contention。
- 在线 RPC 的带宽优先级。
- 源节点读取带宽和目标节点写入带宽。

### 4.5 每 block 迁移与重算混合

当前状态获取是整段 MIGRATE 或整段 RECOMPUTE 二选一。更合理的是每个 block 独立选择：

```text
cost(block) = min(
    migrate_from_best_replica(block),
    recompute(block)
)
```

最终计划可以是：

```text
本地命中 blocks
+ 100G 迁移 blocks
+ 25G 不划算而重算的 blocks
```

如果迁移和重算可并行：

```text
state_ready_time =
max(migration_branch_time, recompute_branch_time)
```

### 4.6 增量同步与版本管理

用户移动前完成预复制后，只需同步上一轮新增的 KV tail：

```text
full_context = stable_prefix + growing_tail
```

KV Directory 需要维护：

- block version。
- owner 与 replicas。
- 已同步到哪个 token/block。
- 正在迁移的 flow。
- 过期副本与并发写入状态。

稳定 prefix 只复制一次，后续仅增量同步 tail。

## 5. Router 与 KV Manager 的接口

建议把 Router 和 KV Manager 的职责明确分开。

Router 输出：

```text
target_exec_node
target_state_nodes
required_blocks
deadline
predicted_residency
mobility_confidence
max_background_bandwidth
```

KV Manager 返回多个候选计划：

```text
visible_latency
background_latency
bytes_to_move
blocks_to_recompute
memory_delta
evictions_caused
replica_value_delta
completion_probability
```

Router 最终比较的是完整计划，而不是只比较一个粗粒度 migrate 字节数。

## 6. 不同模型的主要收益来源

| 场景 | Long-term 主要收益 | 关键前提 |
| --- | --- | --- |
| LLM 短 session、低通信 | 很有限，Greedy 通常更合适 | 使用置信门回退 |
| LLM 长 session、负载不均 | 未来排队、SLA、KV 淘汰 | 准确 queue forecast |
| LLM 跨区域/慢链路 | 多轮 RPC 通信节省 | 移动持续且可预测 |
| VLM | 图像/视频输入转发节省 | 使用真实视觉 payload |
| VLA | 高频视觉输入、连续控制延迟 | 高频稳定驻留、后台同步 |
| 热共享 prefix | 多 session 复用和重算节省 | prefix-aware 副本管理 |

当前 `visual_bytes_per_token=0` 时，VLM/VLA 不会体现真实媒体通信成本。实验必须配置压缩后
图片、视频或传感器 payload，才能验证这类收益。

## 7. 指标与归因

### 7.1 主指标

- Avg routing-sensitive E2E。
- P95/P99 TTFT。
- SLA violation ratio。
- 完整 Avg/P95 E2E。

### 7.2 状态管理指标

- migrate 次数及原因。
- 条件迁移耗时：仅对 MIGRATE 请求统计 Avg/P50/P95。
- 迁移 MB、每次迁移 MB。
- recompute 次数和耗时。
- owner 切换与 replica 数量分布。
- prefix local/remote hit ratio。
- eviction 数量和被迫重算成本。
- 后台迁移被 bubble 覆盖的比例。

### 7.3 Long-term 预测质量

每次决策记录：

```text
predicted_future_cost
predicted_gain_vs_greedy
mobility_prediction
queue_prediction
selected_plan
realized_cost_over_horizon
```

重点观察：

```text
prediction_error =
realized_future_cost - predicted_future_cost
```

若 FutureCost 触发大量迁移但 realized gain 为负，应自动提高 uncertainty margin 或降低 gamma。

## 8. 实验矩阵

至少扫描：

```text
mobility mode: request / session / markov
mobility ratio: 0.0 / 0.2 / 0.4 / 0.8
residency: 1 / 3 / 5 / 10 / 20
session turns: 2 / 4 / 8 / 16 / 32
nodes: 3 / 8 / 16
KV capacity: 低 / 中 / 高
links: 100G / 25G / 跨区域 RTT
visual payload: 0 / 256KB / 1MB / 4MB
load: 低 / SLA 临界 / 过载
seed: 至少 5-10 个
```

对比策略：

```text
Nearest
Greedy
Long-term Router only
Long-term + block migration
Long-term + replica/eviction manager
Long-term + proactive prefetch
Long-term + queue forecast
```

## 9. 成功标准

不能只挑选单个 seed 或单个指标。建议定义：

- session/markov 场景下，至少 5 个 seed 的平均 routing cost 优于 Greedy 10%。
- P99 TTFT 至少改善 10%，且 SLA 违约率不恶化。
- request 抖动场景通过置信门回退，与 Greedy 的差距控制在 1% 内。
- FutureCost 触发的迁移中，realized gain 为正的比例超过 70%。
- 后台迁移至少 50% 被 bubble/idle time 覆盖。
- 限定显存下，淘汰导致的重算成本低于 Greedy。

对于 Decode 占比极高的 LLM，完整 Avg E2E 改善可能小于 1%。此时应同时报告 routing cost 和
P99 TTFT；若业务要求完整 Avg E2E 显著下降，则需要真实排队压力、慢链路、大 payload 或更高效
的推理执行层配合。

## 10. 分阶段实施路线

### Phase 1：可观测性与可靠回退

- 记录 predicted gain、realized gain 和预测误差。
- 增加条件迁移耗时、replica 分布和 eviction 指标。
- 加入 uncertainty margin，收益不明确时回退 Greedy。

### Phase 2：KV Manager 基础优化

- 副本价值准入和复制因子限制。
- 成本感知淘汰。
- 每 block 最优源选择和多源并行迁移。
- 每 block migrate/recompute 混合计划。

### Phase 3：主动预取

- 后台迁移任务队列。
- Decode bubble/idle window 覆盖。
- 增量同步 KV tail。
- 带宽限速和在线流量优先级。

### Phase 4：未来队列预测

- 模型/配置组级到达率预测。
- Continuous batching service capacity 模型。
- 将状态放置导致的粘附流量纳入未来队列。
- 预测误差在线校准。

### Phase 5：联合优化

- Router 比较 KV Manager 返回的完整执行计划。
- 基于实际 trace 训练或校准移动、剩余轮数和队列模型。
- 在真实集群数据上验证模拟结论。

## 11. 最终建议

当前优先级应是：

```text
1. 置信门与 realized gain 对账
2. 副本准入/淘汰，避免三节点副本快速饱和
3. 每 block 混合状态获取与多源迁移
4. 后台预迁移和增量 tail 同步
5. 未来队列预测
```

预期结果不是让 Long-term 在所有场景都强于 Greedy，而是：

- 在不可预测、短 session 场景自动退化为 Greedy。
- 在可预测、长期驻留、慢链路、大 payload 或负载热点场景稳定获得显著收益。
- 能明确解释每次提前迁移为何发生，以及最终是否真正回本。
