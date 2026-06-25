# 整体架构设计文档

## 1. 你的理解是否正确

你提出的心智模型基本正确，这里做一处关键澄清，并补全两个细节：

> “每个计算单元模拟器加载一个大模型形成一个服务；一个总节点搜集全网信息；每个节点有一个 router 做任务分配；data generator 不断向 router 生成任务。”

- ✅ **每个计算节点加载一个大模型 → 形成一个推理服务**：正确。每个节点 = 队列 + 计算模拟器（绑定一个 `ModelSpec`）+ 本地 KV store。
- 🔶 **“一个总节点搜集全网信息”**：更准确地说，是一个**逻辑上的共享元数据目录**（`GlobalStateDirectory`），它聚合各节点上报的负载、显存、网络、prefix/KV 目录。它**不是一个上帝视角的中央调度器**，而是“周期性快照 + 存在同步延迟（staleness）”的元数据服务，这与报告 6.4“节点状态只能以周期性快照形式获得，路由决策需要为上报延迟预留安全裕量”一致。
- ✅ **每个节点有一个 router**：正确。采用**分布式路由**——请求从哪个节点进入，就由那个节点的 router 决策（读取共享元数据目录，可能略微过期）。这对应报告 6.5“三个节点分别作为请求入口并执行本地路由”。
- 🔶 **“data generator 不断向 router 生成任务”**：实现上拆成两步以保证可复现：`DataGenerator` **离线生成一条请求轨迹**（带 `arrival_ms`），再由**事件驱动器/Client** 按到达时间把请求注入到对应入口节点的 router。效果等价于“不断生成任务”，但四种策略可以在**同一条轨迹**上公平对比。

结论：分布式 router + 共享元数据目录（带 staleness），而不是单一全局调度器。

## 2. 总体架构图

```
                 ┌──────────────────────────────────────────────┐
                 │      共享元数据目录 GlobalStateDirectory          │
                 │  节点负载/队列/显存 · 网络状态 · prefix/KV 目录     │
                 │      周期性同步快照（含 staleness 延迟）            │
                 └──────────────────────────────────────────────┘
                  ▲上报 │读取(可能过期)   ▲          ▲
       ┌──────────┴─────┴───┐  ┌─────────┴────────┐ ┌┴─────────────────┐
       │   ServingNode A     │  │   ServingNode B   │ │   ServingNode C   │
       │  Router             │  │  Router           │ │  Router           │
       │  WaitQueue          │  │  WaitQueue        │ │  WaitQueue        │
       │  ComputeSimulator   │  │  ComputeSimulator │ │  ComputeSimulator │
       │  (一个大模型实例)     │  │  (同一大模型)      │ │  (同一大模型)      │
       │  KVCacheStore       │  │  KVCacheStore     │ │  KVCacheStore     │
       └──────┬──────────────┘  └────────┬─────────┘ └────────┬─────────┘
              │ A-B 100Gbps              │                    │
              └──────────────────────────┘  C 25Gbps RDMA     │
              └─────────────────────── NetworkSimulator(边) ───┘
        ▲请求注入(arrival_ms)        ▲                     ▲
   ┌────┴─────┐               ┌──────┴────┐         ┌──────┴────┐
   │ Client A │               │ Client B  │         │ Client C  │
   └──────────┘               └───────────┘         └───────────┘
                  ▲ 离线生成请求轨迹
            ┌─────┴───────┐
            │ DataGenerator│
            └─────────────┘
```

## 3. 组件职责

| 组件 | 模块 | 职责 |
| --- | --- | --- |
| 大模型规格 | `large_model.py` | LLM/VLM/VLA 结构、KV 字节、prefill/decode FLOPs、输入构成（共享事实来源） |
| 计算模拟器 | `compute_simulator.py` | roofline 估算 prefill/decode 时间、显存、迁移/重算成本 |
| 数据生成器 | `data_generator.py` | 离线生成 session→request 轨迹（含移动性、prefix 复用） |
| 网络（边） | `network.py` | 节点间链路带宽/时延/争用，传输时间与利用率 |
| KV 缓存 | `kv_cache.py` | block 级 KV 存储、prefix 目录、命中/迁移/重算/淘汰 |
| 计算节点 | `node.py`（后续） | 队列 + 计算 + KV store 组合成一个推理服务 |
| 元数据目录 | `GlobalStateDirectory`（后续） | 聚合全网状态快照，供 router 读取（带 staleness） |
| 路由器 | `Router`（后续） | 枚举动作、SLA/memory 过滤、四类策略选择 |
| 事件驱动 | `Simulator`（后续） | 离散事件循环，回放轨迹、推进时钟、统计指标 |

本次交付：`large_model`（重命名）、`network`、`kv_cache` 三个模块及设计文档；`node/Router/Simulator` 在架构文档中定义清楚，作为下一步实现。

## 4. 一次请求的完整生命周期

1. `DataGenerator` 预生成轨迹；`Client` 在 `arrival_ms` 把请求投递到 `entry_node` 的 `Router`（`request_arrival` 事件）。
2. `Router` 从 `GlobalStateDirectory` 读取（可能过期的）各节点负载、显存、网络、KV 目录。
3. `Router` 通过 `kv_cache` 的 prefix 目录定位该 session 历史 KV 的 owner 节点，枚举动作集合：`(o,local)`、`(i,migrate)`、`(i,recompute)`。
4. 对每个动作用 `compute_simulator` + `network` 预测 TTFT、端到端时延、显存增量、状态获取成本，做 **SLA 与 memory 约束过滤**。
5. 按策略（就近 / Greedy / 长期成本 / 长期成本+低成本 KV 管理）从可行动作中选择，必要时经 `network` 触发 KV block 迁移。
6. 目标节点入队、prefill、decode；`kv_cache` 写入新 block、更新 owner/副本；`network` 累加链路利用率。
7. `request_finish` 更新节点状态、prefix/session 目录、时延与成本统计，反馈给后续路由决策。

## 5. 为什么这样划分

- **大模型规格独立**：硬件、数据、KV 三处对同一模型必须用一致参数，集中定义避免漂移。
- **网络作为独立“边”模拟器**：100 Gbps 与 25 Gbps 链路差异是实验核心变量；把链路建成共享、可争用的资源，才能体现迁移收益随带宽变化、以及跨主机链路的拥塞。
- **KV 作为 block 级独立模块**：local/migrate/recompute 决策、prefix 命中、增量同步都围绕 block 展开；独立模块让“只传缺失 block”“多副本选最优源”“LRU 淘汰”可被精确建模与度量。
- **分布式 router + 共享目录**：贴近真实系统，并能复现“状态过期导致的误判”和“状态黏附”。
