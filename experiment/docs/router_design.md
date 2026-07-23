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
- `RECOMPUTE`：复用目标节点已有的连续 prefix KV，仅对缺失历史后缀执行增量 prefill，再处理当前输入。
- `FRESH`：对初始上下文和当前输入执行完整 prefill。

由此：

```
TTFT_hat = T_network + T_queue + T_state + T_prefill
E2E_hat  = TTFT_hat + decode_total(output_len, ctx_len)
```

> 关键权衡天然体现：`MIGRATE` 传输缺失 KV；`RECOMPUTE` 对同一缺失后缀执行增量 prefill。两者都利用目标节点已经持有的连续 prefix。
> 前缀大、链路慢 → recompute 更划算；前缀大、链路快 → migrate 更划算。

## 5. 约束过滤

可行动作集合：

```
A_t^F = { a : TTFT_hat(a) + Δ <= SLA_r,  KV_used(exec) + new_kv(r) <= KV_capacity(exec) }
```

- SLA：用 P99 口径的 TTFT 预测加裕量 `Δ` 补偿快照滞后与预测误差。
- memory：请求完整上下文新增 KV 必须放得下，否则触发 KV 淘汰或剔除该动作。
- 若 `A_t^F` 为空：记为 SLA/memory 不可执行请求（admission control / 降级 / 拒绝），计入指标。

## 6. 五类策略

| 策略 | 选择规则 | block 级 KV |
| --- | --- | --- |
| `NEAREST`（基线） | 固定在 `entry_node` 执行（命中则 LOCAL，否则 FRESH/MIGRATE/RECOMPUTE 取较优），不看其他节点 | 否 |
| `GREEDY` | 在 `A_t^F` 中选 `E2E_hat` 最小的动作 | 否（migrate 整段从 owner 传） |
| `GREEDY_KV` | 与 Greedy 相同的即时选择规则，启用主动 KV placement | 是 |
| `LONG_TERM` | 在 `A_t^F` 中选 `Q(s,a)=c_t(s,a)+γ·V_next(a)` 最小的动作；仅被动恢复 KV | 默认否，可配置 |
| `LONG_TERM_KV` | 同一长期路由逻辑，并启用块级恢复与后台主动 KV placement | 默认是，可配置 |

### 6.1 未来价值 V_next（长期策略）

Greedy 只看当前请求。模拟中的长期策略不训练模型，而是直接读取已生成 trace 中该 session 的真实
剩余请求、入口、输入长度和输出长度，进行有限时域动态规划。设未来请求到达前最新完整 KV 位于
`owner`，则：

```
V_h(owner) = min_exec [
    E2E_h(owner, exec) + gamma * V_{h+1}(exec)
]
Q_k(a) = E2E_k(a) + gamma * V_{k+1}(a.exec_node)
```

- `E2E_h` 使用第 `h` 个真实请求的入口、input、output 和完整历史 prefix；
- `exec != owner` 时比较整段 prefix 同步与完整 prefix recompute，取较小者；
- 每个未来请求都可重新选择执行节点，执行节点成为下一状态的完整 KV owner；
- 折扣因子逐个未来请求施加，而不是对聚合 FutureCost 只乘一次。

这种设置消除了 response-length 与 session-continuation 的预测误差，用于观察长期机制在已知 session
未来 realization 下的行为；它不是训练得到的 Q 网络，也不代表完整集群未来状态全知。未来其他 session
的具体路由和排队演化仍未知，代码只将当前队列快照按真实未来到达间隔排空，并计入当前候选动作新增的
计算负载。

迁移、重算和网络传输仍按串行相加，尚未模拟“部分迁移 + 部分重算”或并行覆盖 bubble。

每个候选动作的状态获取方式互斥：完整 LOCAL、MIGRATE 或 RECOMPUTE 三选一。启用 block-level
机制时，“部分”仅指目标节点已有 block 不重复迁移，并不把剩余 block 再拆成迁移与重算并行执行。

对最终选中的 MIGRATE 记录 `selection_reason`：入口所有动作被 SLA、memory、两者或混合约束阻塞时
归入对应硬约束；入口仍有可执行动作时，Greedy/即时最优归为 `immediate_cost`，long-term 为降低
FutureCost 而接受更高当前 E2E 时归为 `future_cost`。指标同时按原因聚合次数和实际迁移字节。

排队 backlog 按前序 Prefill、Recompute、Decode 三类记录。当前默认假设 Decode 由 continuous
batching 吸收，不串行进入 admission queue，因此 `queue_decode_ms` 为 0；总排队目前主要来自
前序 Prefill，若未来接入真实 batch scheduler 再替换这一假设。

### 6.2 KV Manager（GREEDY_KV / LONG_TERM_KV）

默认配置下，两种长期策略的区别为：

- `LONG_TERM`：把整段前缀从单一 owner 被动同步到 exec，不运行后台 placement。
- `LONG_TERM_KV`：`plan_migration` 只同步 exec 缺失的 block，并从传输成本最低的副本源取数；请求
  完成后，KV Manager 还会按移动概率和 session 活跃度后台预放置连续 KV blocks。

配置项 `router.long_term_block_level_kv` 和 `router.long_term_kv_block_level_kv` 可分别控制两种策略
是否采用 block-level 被动同步。默认前者关闭、后者开启，使 `LONG_TERM_KV - LONG_TERM` 表示 KV
Manager 的总体增益；将前者打开后，可单独分离 block-level 被动同步与后台主动 placement 的贡献。

## 7. 决策提交与反馈

`route(request)` 返回所选 `ActionCost`；`commit(request, decision, t_now)` 执行副作用：

1. 在 exec_node 的 `KVCacheStore` 写入该请求新生成的 KV block；
2. 更新 `GlobalKVDirectory`：注册新 block；migrate 时 `commit_migration` 增加副本并按策略切 owner（含旧副本短 TTL）；recompute 时 `note_recompute`；
3. 更新 exec_node 的 `estimated_queue_ms` 与近期 TTFT；
4. 累加链路利用率（`network.start/finish_transfer`）与指标。

由此形成“请求分类 → 节点过滤 → 短期/长期选择 → 状态迁移 → 反馈”的闭环。

## 8. 接口设计

```python
class Policy(Enum): NEAREST, GREEDY, GREEDY_KV, LONG_TERM, LONG_TERM_KV
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
                 gamma=1.0, sla_margin_ms=20.0, expected_session_turns=4)
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
