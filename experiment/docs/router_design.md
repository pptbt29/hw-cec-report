# Router（路由器）设计文档

## 1. 设计目标

Router 是分布式卸载方案的决策核心。每个节点有一个 router，请求从哪个节点进入，就由该节点的 router 决定：

- 请求在哪个节点执行（exec_node）；
- 该请求复用的历史 KV/prefix 如何获取（local / migrate / recompute / fresh）；
- 在满足 **P99 TTFT SLA** 与 **GPU memory** 约束的前提下，按所选策略最小化成本。

Router 不改变实例内部的 batching/token scheduling，只做请求级路由与会话级状态决策。它把
`large_model`（结构）、`compute_simulator`（时间/显存）、`network`（链路）、`kv_cache`（block 状态）
四个模块组合成一个可比较的成本函数，并实现四类策略，对应实验报告第 4 节。

## 2. 输入：从哪里读状态

Router 从**共享元数据目录** `GlobalStateDirectory` 读取（可能略旧的）快照：

- 每个节点的 `NodeState`：预计排队时间、KV 已用/容量、空闲显存、近期 P99 TTFT。
- `NetworkSimulator`：节点间链路传输时间（含争用）。
- `GlobalKVDirectory`：该请求 prefix 对应的 block 在哪些节点有副本。

快照存在 staleness，因此约束过滤里加入安全裕量 `Δ`（`sla_margin_ms`）。

## 3. 动作空间

设请求 `r` 的可复用前缀对应 block hash 列表为 `P`（由 `prefix_id` + `prefix_tokens` 经
`kv_cache.block_hashes_for_len` 得到）。对每个候选执行节点 `i`，查 `GlobalKVDirectory.longest_prefix(P, i)`
得到本地命中块数 `local_hit` 与远端命中块数 `remote_hit`：

| 条件 | 动作 `mode` | 含义 |
| --- | --- | --- |
| `P` 为空（首请求/无可复用 KV） | `FRESH` | 在 `i` 上对全部输入 prefill |
| `local_hit == len(P)` | `LOCAL` | `i` 已有完整前缀，直接复用 |
| 否则（前缀在别处或部分缺失） | `MIGRATE` | 把缺失 block 迁到 `i` 后执行 |
| 否则 | `RECOMPUTE` | 在 `i` 上重算前缀 KV 后执行 |

动作集合：`FRESH/LOCAL` 各节点至多一个；`MIGRATE/RECOMPUTE` 对每个非完整命中节点各一个。
单 KV owner、3 节点时，动作总数约 `2N-1`，与实验报告第 5 节一致。

## 4. 成本模型

即时成本 `c_t(s,a) = T_network + T_queue + T_state + T_inference`：

- **T_network**：把请求输入从 `entry_node` 传到 `exec_node`。`exec==entry` 时为 0；否则
  `network.transfer_time_ms(entry, exec, input_bytes)`，`input_bytes ≈ input_tokens * dtype_bytes`
  （VLM/VLA 因视觉 token 多而更大，体现多模态输入传输代价）。
- **T_queue**：`NodeState.estimated_queue_ms`（exec_node 当前预计排队）。
- **T_state**：
  - `LOCAL/FRESH`：0；
  - `MIGRATE`：`GlobalKVDirectory.plan_migration(P, exec, net).transfer_ms`（块级、只传缺失、选最优源）；
  - `RECOMPUTE`：`compute.recompute_time_ms(reusable_prefix_tokens)`。
- **T_inference（prefill，决定 TTFT）**：
  - `LOCAL/MIGRATE`：只 prefill 非前缀部分 `prefill(input_tokens - hit_tokens)`；
  - `RECOMPUTE/FRESH`：`prefill(input_tokens)`（重算把前缀并入整段 prefill）。

由此：

```
TTFT_hat = T_network + T_queue + T_state + T_prefill
E2E_hat  = TTFT_hat + decode_total(output_len, ctx_len)
```

> 关键权衡天然体现：`MIGRATE` 付出传输但省掉前缀 prefill；`RECOMPUTE` 付出整段 prefill 但不占链路。
> 前缀大、链路慢 → recompute 更划算；前缀大、链路快 → migrate 更划算。

## 5. 约束过滤

可行动作集合：

```
A_t^F = { a : TTFT_hat(a) + Δ <= SLA_r,  KV_used(exec) + new_kv(r) <= KV_capacity(exec) }
```

- SLA：用 P99 口径的 TTFT 预测加裕量 `Δ` 补偿快照滞后与预测误差。
- memory：请求完整上下文新增 KV 必须放得下，否则触发 KV 淘汰或剔除该动作。
- 若 `A_t^F` 为空：记为 SLA/memory 不可执行请求（admission control / 降级 / 拒绝），计入指标。

## 6. 四类策略

| 策略 | 选择规则 | block 级 KV |
| --- | --- | --- |
| `NEAREST`（基线） | 固定在 `entry_node` 执行（命中则 LOCAL，否则 FRESH/MIGRATE/RECOMPUTE 取较优），不看其他节点 | 否 |
| `GREEDY` | 在 `A_t^F` 中选 `E2E_hat` 最小的动作 | 否（migrate 整段从 owner 传） |
| `LONG_TERM` | 在 `A_t^F` 中选 `Q(s,a)=c_t(s,a)+γ·V_next(a)` 最小的动作 | 否 |
| `LONG_TERM_KV` | 同 `LONG_TERM`，但 migrate 用 `plan_migration`（块级、只传缺失、最优源、增量同步） | 是 |

### 6.1 未来价值 V_next（长期策略）

Greedy 只看当前请求，易产生**状态黏附**：请求被转到远端节点后，KV 留在远端，后续请求持续被吸到远端。
长期策略加入一项估计“把 KV 落在 `exec_node` 对未来请求的影响”：

```
remaining = max(group.turns_mean - turn_index - 1, 0)
P_k(entry) = 第 k 个未来请求的入口概率

serve(node) =
    Σ_k E_{entry~P_k}[request_network(entry,node) + response_network(node,entry)]

relocate(exec,dst) =
    min(one_time_migrate(exec,dst), one_time_recompute(dst)) + serve(dst)

V_next(a) = min(serve(exec), min_dst relocate(exec,dst))
```

- `request` 按 home 抖动概率生成每轮入口分布；
- `session` 在移动后保持当前入口，移动前按一次迁移概率预测；
- `markov` 按当前入口、已驻留轮数、驻留阈值和迁移概率逐轮传播入口分布；
- KV 搬迁或重算只计一次，不再乘以 remaining。

于是长期策略会在用户移动后**更早迁移/重算到新入口节点**，避免黏附；而 Greedy 因 `γ·V_next` 缺席会继续黏在旧 owner。`γ`（默认 0.9）控制对未来的重视程度。

当前 `V_next` 仍是有限视野近似而非完整集群动态规划：它使用配置组的 `turns_mean`，尚未预测未来
集群排队演化，并以当前请求长度估计未来 payload/state；迁移、重算和网络传输仍按串行相加，尚未
模拟“部分迁移 + 部分重算”或并行覆盖 bubble。

每个候选动作的状态获取方式互斥：完整 LOCAL、MIGRATE 或 RECOMPUTE 三选一。`LONG_TERM_KV`
的“部分”仅指目标节点已有 block 不重复迁移，并不把剩余 block 再拆成迁移与重算并行执行。

对最终选中的 MIGRATE 记录 `selection_reason`：入口所有动作被 SLA、memory、两者或混合约束阻塞时
归入对应硬约束；入口仍有可执行动作时，Greedy/即时最优归为 `immediate_cost`，long-term 为降低
FutureCost 而接受更高当前 E2E 时归为 `future_cost`。指标同时按原因聚合次数和实际迁移字节。

排队 backlog 按前序 Prefill、Recompute、Decode 三类记录。当前默认假设 Decode 由 continuous
batching 吸收，不串行进入 admission queue，因此 `queue_decode_ms` 为 0；总排队目前主要来自
前序 Prefill，若未来接入真实 batch scheduler 再替换这一假设。

### 6.2 低成本 KV 管理（LONG_TERM_KV）

与 LONG_TERM 的唯一区别在 migrate 成本：

- `LONG_TERM`：把整段前缀从单一 owner 传到 exec（`bytes = prefix_tokens 的 KV`）。
- `LONG_TERM_KV`：`plan_migration` 只传 exec 缺失的 block，并从**传输成本最低的副本源**取数，支持
  100G 域内增量复制。因此在已有部分副本、或存在 100G 近源时迁移更便宜，迁移字节更少。

这把报告中“block-level KV 索引、节点间直接传输、增量状态同步”的收益落成可度量差异。

## 7. 决策提交与反馈

`route(request)` 返回所选 `ActionCost`；`commit(request, decision, t_now)` 执行副作用：

1. 在 exec_node 的 `KVCacheStore` 写入该请求新生成的 KV block；
2. 更新 `GlobalKVDirectory`：注册新 block；migrate 时 `commit_migration` 增加副本并按策略切 owner（含旧副本短 TTL）；recompute 时 `note_recompute`；
3. 更新 exec_node 的 `estimated_queue_ms` 与近期 TTFT；
4. 累加链路利用率（`network.start/finish_transfer`）与指标。

由此形成“请求分类 → 节点过滤 → 短期/长期选择 → 状态迁移 → 反馈”的闭环。

## 8. 接口设计

```python
class Policy(Enum): NEAREST, GREEDY, LONG_TERM, LONG_TERM_KV
class StateMode(Enum): FRESH, LOCAL, MIGRATE, RECOMPUTE

@dataclass
class Action:
    exec_node: int; mode: StateMode
    src_node: Optional[int]; hit_tokens: int; migrate_bytes: int

@dataclass
class ActionCost:
    action: Action
    t_network_ms; t_queue_ms; t_state_ms; t_prefill_ms
    ttft_ms; e2e_ms; new_kv_bytes; feasible; reason; q_value

class Router:
    def __init__(self, model, directory, policy=Policy.GREEDY,
                 gamma=0.9, sla_margin_ms=20.0, expected_session_turns=4)
    def route(self, request) -> ActionCost
    def commit(self, request, decision, t_now) -> None
    # 内部: _enumerate_actions / _cost / _feasible / _select / _future_value
```

## 9. 与其他模块的关系

- `large_model`：结构、KV block 大小与字节数。
- `compute_simulator`：prefill/decode 时间、显存、recompute 成本。
- `network`：T_network 与 migrate 传输定价、链路利用率。
- `kv_cache`：prefix 命中、`plan_migration`、owner/副本管理。
- `node`：`ServingNode` 提供队列/显存/KV 状态，`GlobalStateDirectory` 提供（含 staleness 的）全网快照。
- `data_generator`：提供请求轨迹（entry_node、prefix、mobility）。
