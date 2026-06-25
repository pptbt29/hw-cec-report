# 系统流程文档（基于现有代码）

本文档基于 `experiment/sim/` 下已实现的代码，描述从初始化、工作负载生成、请求路由到状态反馈的**完整数据流与调用链**。每一步都标注对应的类/函数，便于对照源码阅读。

## 0. 模块与文件

| 文件 | 关键类/函数 | 职责 |
| --- | --- | --- |
| `large_model.py` | `ModelType`、`ModelSpec`、`get_model`、`MODEL_REGISTRY` | 大模型结构/KV/FLOPs/输入构成（单一事实来源） |
| `compute_simulator.py` | `HardwareSpec`、`ComputeSimulator`、`get_hardware` | roofline 估算 prefill/decode/显存/迁移/重算 |
| `network.py` | `LinkSpec`、`NetworkTopology`、`NetworkSimulator`、`default_topology` | 链路带宽/时延/争用/利用率 |
| `kv_cache.py` | `block_hashes_for_len`、`make_blocks`、`KVCacheStore`、`GlobalKVDirectory`、`MigrationPlan` | block 级 KV 存储、目录、迁移规划 |
| `data_generator.py` | `LengthDistribution`、`WorkloadConfig`、`DataGenerator`、`Request` | 生成可复现请求轨迹 |
| `node.py` | `ServingNode`、`NodeState`、`GlobalStateDirectory`、`build_cluster` | 节点服务 + 共享状态目录（含 staleness） |
| `router.py` | `Policy`、`StateMode`、`Action`、`ActionCost`、`Router`、`simulate_trace` | 动作枚举/成本/约束/四策略/回放评估 |

调用主入口：`router.simulate_trace(policy, requests, model, hardware, network, ...)`。

## 1. 初始化流程

```
get_model("CodeLlama34B")           # large_model: 取 ModelSpec
get_hardware("A800T-A2")            # compute_simulator: 取 HardwareSpec
NetworkSimulator(default_topology())# network: A-B 100G, A/B-C 25G
build_cluster(model, hw, net, num_nodes=3, staleness_ms)   # node
```

`build_cluster`（`node.py`）做三件事：

1. 为每个节点创建 `ServingNode(i, model, hw)`。其构造函数内：
   - 建 `ComputeSimulator(model, hw)`；
   - 计算 KV 容量 `kv_capacity = total_memory - weights - activation_reserve`，建 `KVCacheStore(i, model, kv_capacity)`；
   - 初始化 `assigned_load_ms=0`、TTFT 样本表。
2. 建 `GlobalKVDirectory(num_nodes)`（block 副本目录 + 统计）。
3. 建 `GlobalStateDirectory(nodes, kv_dir, net, staleness_ms)`，并 `refresh(force=True)` 拍下首张快照。

`simulate_trace` 随后为每个节点建一个 `Router(model, cluster, policy, ...)`，即**每节点一个 router**。

## 2. 工作负载生成流程（`DataGenerator.generate`）

```
WorkloadConfig.default_experiment()    # 3 个分组: 高优 CodeLlama / 普通 CodeLlama / Qwen2-VL
DataGenerator(config).generate()
```

`generate()` 的步骤：

1. 用 `random.Random(seed)` 固定随机源（保证四策略复用同一轨迹）。
2. 对每个 `WorkloadGroup`：
   - 取 `ModelSpec`，确定 `sla`、`out_dist`（缺省取 `model.default_output_dist`）、到达率 `rate`、session 数 `= concurrency`。
   - 对每个 session：采样 `num_turns`、`home_node`、`prefix_id`、`session_start`，初始化 `carried_tokens = shared_prefix_tokens`。
   - 对每一轮 `turn`：
     - `prompt_text` ← `prompt_dist.sample`；
     - `visual_tokens` ← `model.visual_tokens(image_size, num_frames)`（VLM/VLA）；
     - `state_tokens` ← VLA 为 8，否则 0；
     - `input_tokens` ← `model.input_tokens(...)` 合成；
     - `output_len` ← `out_dist.sample`（**预设生成长度**）；
     - `prefix_tokens = carried_tokens`（会话历史累加，turn 越大前缀越长）；
     - `arrival_ms` ← 首轮 = session_start，之后按 `expovariate(rate)` 累加（Poisson）；
     - 追加一个 `Request`。
     - 更新 `carried_tokens += history_growth*(input_tokens+output_len)`。
3. `_apply_mobility`：`duration*mobility_start_frac` 之后到达的请求，按 `mobility_ratio` 把 `entry_node` 改为非 `home_node`，标记 `mobility_switched=True`。
4. 按 `arrival_ms` 排序，赋 `request_id`，返回 `List[Request]`。

产物 `Request` 关键字段：`arrival_ms, entry_node, home_node, model_name, model_type, priority, sla_ms, input_tokens, output_len, prefix_id, prefix_tokens, is_session_first, turn_index, mobility_switched`。

## 3. 主回放循环（`simulate_trace`）

```
network.reset_stats()
cluster = build_cluster(...)
routers = {i: Router(model, cluster, policy, ...) for i in nodes}
reqs = sorted(只取 model_name 匹配的请求, by arrival_ms)
prev_t = 0
for r in reqs:
    for n in nodes: n.advance_to(r.arrival_ms, prev_t)   # 队列随时间排空
    prev_t = r.arrival_ms
    cluster.refresh(r.arrival_ms, force=True)             # 同步共享快照
    router = routers[r.entry_node]                        # 入口节点的 router
    decision = router.route(r)                            # 选动作（见第 4 节）
    统计 infeasible / sla_viol / cross_node / migrate / recompute
    router.commit(r, decision, r.arrival_ms)             # 落地副作用（见第 5 节）
    记录 e2e / ttft
return 指标 dict
```

要点：

- **时间推进**：`ServingNode.advance_to(t_now, prev_t)` 用 `assigned_load_ms -= (t_now-prev_t)` 模拟节点在请求间隔内消化排队（最低到 0）。
- **状态同步**：`GlobalStateDirectory.refresh` 把各节点 `state()` 拍成快照；当 `staleness_ms>0` 时 router 读到的是**可能过期**的视图。
- **分布式路由**：用 `r.entry_node` 选 router，体现“从哪进入由哪决策”。

## 4. 单请求路由流程（`Router.route`）

`route(request)` 四步：`_prefix_hashes → _enumerate → _cost(每个动作) → 策略选择`。

### 4.1 前缀分析（`_prefix_hashes` / `_prefix_stats`）

- `_prefix_hashes`：若 `prefix_tokens>0` 且非 `is_session_first`，调用
  `block_hashes_for_len(model.name, version, prefix_id, prefix_tokens, kv_block_size)`
  得到**前缀链式 hash 列表**；否则返回空（首请求/无可复用 KV）。
- `_prefix_stats(hashes, node)`：返回
  - `located`：从头连续、在**任意节点**有副本的 block 数（`GlobalKVDirectory.locate`）；
  - `contiguous_local`：从头连续、在**该节点**有副本的 block 数。

### 4.2 动作枚举（`_enumerate`）

`located_tokens = min(located*block_size, prefix_tokens)`。对每个节点 `node`：

- `located == 0` → 生成 `FRESH`（全量 prefill）。
- 否则取 `contiguous_local`：
  - `local_blocks >= located` → `LOCAL`（该节点已有完整可复用前缀，`hit_tokens=located_tokens`）；
  - 否则 → 同时生成 `MIGRATE`（`_migrate_action`）和 `RECOMPUTE`。

`_migrate_action(located_hashes, dst, located_tokens)` 按策略分叉：

- `LONG_TERM_KV`（`block_level_kv=True`）→ `GlobalKVDirectory.plan_migration` 只传 dst 缺失 block、选最优源，`migrate_bytes=plan.bytes_to_move, src=plan.src`。
- 其他策略 → 从单一 owner（`_owner_of`）整段搬运，`migrate_bytes = kv_bytes_for_tokens(located_tokens)`。

### 4.3 成本计算（`_cost`）

对每个动作产出 `ActionCost`：

| 项 | 计算 |
| --- | --- |
| `t_network` | `exec==entry` 取 0，否则 `net.transfer_time_ms(entry, exec, input_tokens*dtype_bytes, contention=True)` |
| `t_queue` | 快照里 exec 节点的 `estimated_queue_ms` |
| `t_state` | MIGRATE：`net.transfer_time_ms(src, exec, migrate_bytes)`；RECOMPUTE：`compute.recompute_time_ms(hit_tokens)`；LOCAL/FRESH：0 |
| `t_prefill` | `compute.estimate_prefill(input_tokens - hit_tokens).prefill_ms` |
| `ttft` | `t_network + t_queue + t_state + t_prefill` |
| `e2e` | `ttft + compute.estimate_decode(output_len, ctx_len=input_tokens, batch_size=1).total_ms` |
| `new_kv` | `model.kv_bytes_for_tokens(input_tokens + output_len)` |

### 4.4 约束过滤（`_feasible`）

- SLA：`ttft + sla_margin_ms <= sla_ms`，否则 `reason="sla"`；
- 显存：`new_kv <= node_state.mem_free_bytes`，否则 `reason="memory"`。

### 4.5 策略选择

- `NEAREST`（`_select_nearest`）：只在 `exec==entry` 的动作里选 `e2e` 最小（不看其他节点）。
- `GREEDY`：可行集（无可行则全集）里选 `e2e` 最小。
- `LONG_TERM` / `LONG_TERM_KV`：对每个候选算 `q = e2e + gamma*_future_value`，选 `q` 最小。

`_future_value`（长期项，防黏附）：

```
remaining    = max(expected_session_turns - turn_index - 1, 0)
future_entry = entry_node if mobility_switched else home_node
若 exec == future_entry → 0
否则 penalty = net.transfer_time_ms(exec, future_entry, future_state_bytes, contention=False)
返回 remaining * penalty
```

即：把 KV 放在“未来入口”节点则未来零远程成本；放在远离未来入口处则每个后续请求都被罚，长期策略据此**更早迁移/重算到新入口**。

## 5. 提交与状态反馈（`Router.commit`）

`commit(request, decision, t_now)` 执行副作用，形成闭环：

1. 若动作是 `MIGRATE` 且有源：
   - `net.start_transfer(src, exec, migrate_bytes, t_now)` → `net.finish_transfer(flow, t_now+t_state)`，累加链路利用率；
   - `kv.plan_migration(located_hashes, exec, net)` → `kv.commit_migration(plan, switch_owner=True)`：把缺失 block 副本登记到 exec 并切 owner（累加 `migrate_bytes`、`owner_switch_count`）。
2. 若动作是 `RECOMPUTE`：`kv.note_recompute()`（累加 `recompute_count`）。
3. **会话状态落到 exec 节点**：`context_tokens = input_tokens + output_len`，
   `make_blocks(...)` 生成该会话增长后的全部 block →
   `node.kv_store.insert(blocks)`（容量不足触发 LRU 淘汰）→
   对每个 block `kv.register(exec, b)` 且 `kv.set_owner(b, exec)`。
   这一步使后续同会话请求的前缀定位到 exec，正是状态黏附/迁移的来源。
4. 负载与指标：`node.add_load(t_prefill + (recompute 时加 t_state))`、`node.record_ttft(ttft)`。

## 6. 指标统计（`simulate_trace` 返回）

逐请求累加并最终汇总：

- `avg_e2e_ms`、`p99_ttft_ms`（注：e2e 被 batch=1 的 decode 主导，区分力弱）；
- `sla_violation_ratio`、`infeasible_ratio`；
- `cross_node_ratio`（exec≠entry 占比）；
- `migrate_count`、`recompute_count`；
- `owner_switch_count`、`migrate_bytes_mb`（取自 `GlobalKVDirectory.stats`）。

## 7. 单请求端到端走查（示例）

以一个已移动的 CodeLlama34B 连续会话请求为例（KV owner=A，请求入口移动到 C）：

```
route(r):
  hashes = _prefix_hashes(r)                # 前缀 block hash
  located, _ = _prefix_stats(hashes, C)     # located>0（KV 在 A）
  _enumerate → A:LOCAL, B:MIGRATE/RECOMPUTE, C:MIGRATE/RECOMPUTE
  _cost 各动作：
    A:LOCAL      t_network=entry->A, t_state=0,      t_prefill=prefill(new)
    C:MIGRATE    t_state=transfer(A->C,25G,大)        ← 慢链路，贵
    C:RECOMPUTE  t_state=recompute(prefix)            ← 可能比 25G 迁移便宜
  GREEDY: 选 e2e 最小 → 常落在 A:LOCAL（把请求吸回 A）→ cross_node, 状态黏附
  LONG_TERM: q=e2e+γ·future_value，future_entry=C(已移动)
             A:LOCAL 的 future_value 大（未来都要 A->C）→ 倾向 C:RECOMPUTE/MIGRATE
             → 把 KV 迁向 C，后续请求本地命中
commit(r, decision):
  迁移则 net.start/finish_transfer + kv.commit_migration（切 owner 到 exec）
  make_blocks(context) 写入 exec.kv_store 并 register/set_owner
  add_load + record_ttft
```

这解释了 `python -m sim.router` 的对比结果：greedy 的 `cross_node%`/`migrate_bytes` 偏高（黏附），long-term 显著下降。

## 8. 调用关系总览

```
simulate_trace
 ├─ build_cluster ─ ServingNode(ComputeSimulator, KVCacheStore) ─ GlobalKVDirectory ─ GlobalStateDirectory
 ├─ for r in trace:
 │   ├─ ServingNode.advance_to            (排空队列)
 │   ├─ GlobalStateDirectory.refresh      (同步快照)
 │   ├─ Router.route
 │   │   ├─ _prefix_hashes  → kv_cache.block_hashes_for_len
 │   │   ├─ _enumerate      → _prefix_stats(GlobalKVDirectory.locate) / _migrate_action(plan_migration)
 │   │   ├─ _cost           → ComputeSimulator.estimate_prefill/estimate_decode/recompute_time_ms
 │   │   │                    NetworkSimulator.transfer_time_ms
 │   │   ├─ _feasible       (SLA + memory)
 │   │   └─ 策略选择        (_future_value for long-term)
 │   └─ Router.commit
 │       ├─ NetworkSimulator.start/finish_transfer
 │       ├─ GlobalKVDirectory.commit_migration / note_recompute / register / set_owner
 │       ├─ kv_cache.make_blocks → KVCacheStore.insert
 │       └─ ServingNode.add_load / record_ttft
 └─ 汇总指标
```

各模块单独运行（`python -m sim.<module>`）也都带 `__main__` 自检；`demo.py` 串联全部组件并打印四策略对比。
