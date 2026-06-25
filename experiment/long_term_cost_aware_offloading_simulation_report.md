# 面向三节点 CEC-LLM 推理的长期成本感知请求卸载模拟实验报告

## 1. 实验目的

本实验用于验证第 6.4 节提出的“长期成本感知请求卸载方案”在项目模拟场景中的有效性。实验关注三个相同模型推理节点组成的 CEC 推理集群，在开启 prefix cache、存在非对称互联网络和用户移动的条件下，请求调度算法能否在满足 P99 TTFT SLA 与 GPU memory 约束的前提下降低长期服务成本。

传统就近推理或最小时延路由通常只根据当前请求选择执行节点。当用户移动后，请求入口可能与历史 KV cache 所在节点分离。如果系统仍然只优化当前请求时延，可能持续访问旧节点或长期黏附在远端节点，产生累积跨节点通信成本。本实验希望模拟这种状态黏附现象，并比较长期成本感知策略是否能通过 KV cache 迁移、重算或本地执行选择降低长期代价。

## 2. 实验场景

实验集群由 3 台 A800T-A2 机器组成，每台机器部署一个相同模型的推理实例，并开启 prefix cache。每个节点同时作为请求入口和可选计算节点。

节点互联关系如下：

- Node A 与 Node B 之间通过 100 Gbps 高速链路直连。
- 第三个节点 Node C 通过 25 Gbps 跨主机 RDMA 链路接入。
- 三个节点均维护本地请求队列、GPU memory 状态、KV cache 容量和 prefix cache 元数据。

请求客户端设置如下：

- 三个请求客户端分别对应三个入口节点。
- 测试前 50% 时间内，请求默认发送给就近推理实例。
- 测试进行到总时长的 50% 后，20% 的请求改为发送给另外两个实例，用于模拟用户移动后入口节点变化。
- 移动后，请求入口可能与历史 KV cache owner 不一致，从而触发 local、migrate、recompute 三类状态处理动作。

模型与负载设置如下：

- CodeLlama34B 的高优先级请求：P99 TTFT SLA 为 150 ms，总并发数为 24。
- CodeLlama34B 的普通请求：P99 TTFT SLA 为 500 ms，总并发数为 96。
- Qwen2-VL-7B-Instruct 请求：P99 TTFT SLA 为 500 ms，总并发数为 24。
- 基线配置为就近推理，即请求默认发送给最近推理实例直接处理，不做负载均衡。
- 优化目标是在基线基础上使并发能力提升 50%，平均端到端时延下降 20%。

## 3. 待验证问题

本实验重点回答以下问题：

- 在用户移动导致请求入口与历史 KV cache owner 分离后，Greedy 最小时延路由是否会产生状态黏附和长期跨节点通信成本。
- 在 SLA 与 GPU memory 约束下，长期成本感知策略能否选择更合理的 local、migrate 或 recompute 动作。
- 100 Gbps 高速链路与 25 Gbps 跨主机 RDMA 链路的差异是否会影响 KV cache 迁移收益。
- block-level KV cache 管理是否能够降低状态迁移成本，提高 prefix/KV 复用率。
- 对 CodeLlama34B 与 Qwen2-VL-7B-Instruct，不同 SLA 和输入规模是否会导致不同的卸载策略偏好。

## 4. 算法方案

实验比较四类调度策略。

第一类是就近推理基线。请求始终发送到入口节点对应的本地推理实例。该策略不考虑其他节点负载、历史 KV cache 位置和网络拓扑，作为项目默认 baseline。

第二类是 Greedy 最小时延路由。入口节点根据同步得到的全局状态枚举可行动作，并在满足 SLA 和 memory 约束的动作中选择当前请求预计端到端时延最低的动作。其即时成本为：

```text
c_t(s_t, a)
= T_network(a)
+ T_queue(a)
+ T_state(a)
+ T_inference(a)
```

其中 `T_state(a)` 根据动作类型分别表示本地 KV 读取、KV 迁移或 KV 重算时间。该策略可以缓解局部节点过载，但可能因为单次迁移或重算成本较高而持续选择远端已有 KV 的节点。

第三类是长期成本感知路由。该策略在 Greedy 的即时成本基础上引入下一状态价值：

```text
Q(s_t, a)
= c_t(s_t, a)
+ gamma * E[V(s_{t+1}) | s_t, a]
```

系统状态 `s_t` 包括当前请求入口、各节点队列与显存状态、网络状态、用户移动趋势、KV cache 位置、KV cache 大小和版本。路由器在满足 SLA 与 memory 约束的可行动作集合内，选择长期累计成本最低的动作。该策略对应第 6.4 节中的有限时域滚动规划。

第四类是长期成本感知路由与低成本 KV 管理联合方案。该策略在长期成本路由基础上进一步引入 block-level KV 索引、节点间直接传输和增量状态同步。它不只决定请求在哪个节点执行，还决定实际发生 state transfer 时从哪个副本传输、传输哪些 KV blocks、何时切换 KV owner，以及旧副本保留多久用于回退。

## 5. 可行动作建模

对于首次请求或不存在历史 KV cache 的请求，动作仅为选择 Node A、Node B 或 Node C 之一执行。

对于已有 KV cache 的 session，设当前有效 KV 位于节点 `o_t`，系统共有 `N=3` 个节点，则动作集合为：

```text
A_t =
{(o_t, local)}
union
{(i, migrate), (i, recompute) | i != o_t}
```

在单一 KV owner 的情况下，动作总数为 `2N - 1 = 5`：

- 在已有 KV 的节点直接执行。
- 将 KV 迁移到另一个节点后执行。
- 在另一个节点重新计算 KV 后执行。

如果某些 KV blocks 已经存在多个有效副本，则具有本地副本的节点可以直接执行，KV 管理模块从已有副本中选择传输成本最低的迁移源节点。

## 6. 约束过滤

每个候选动作必须同时满足 TTFT SLA 和 GPU memory 约束。对于请求 `r_t` 和动作 `a`，模拟器预测：

- 请求入口到目标节点的网络传输时间。
- 目标节点预计排队时间。
- KV 本地读取、迁移或重算时间。
- prefill 时间。
- decode 时间。
- 请求执行过程中的新增 KV cache 和显存占用。

可行动作集合定义为：

```text
A_t^F = {
  a in A_t:
  T_TTFT_hat(r_t, a) + Delta_t(a) <= SLA_r_t,
  M_hat(r_t, a) + M_reserve_i <= M_free_i
}
```

其中 `Delta_t(a)` 用于补偿执行时间预测误差和元数据同步滞后。若可行动作集合为空，该请求不进入正常执行流程，并记录为 SLA/memory 不可行请求。

## 7. KV Cache 管理

KV cache 按固定大小 block 组织，并为每个 block 维护以下元数据：

- model ID 与 model version。
- adapter ID。
- session ID。
- prefix hash。
- block hash。
- block index。
- KV block 大小。
- 当前 owner 节点。
- 有效副本列表。
- 最近访问时间。
- 状态版本。

当动作选择 migrate 时，目标节点先查询本地已有 prefix/KV blocks，仅从源节点传输缺失 blocks。对于 Node A 与 Node B 之间的 100 Gbps 链路，模拟器允许更积极的 KV 迁移或复制；对于涉及 Node C 的 25 Gbps 跨主机链路，迁移策略需要更高的长期复用收益才能触发。

迁移过程采用分块传输和增量同步。稳定历史 blocks 优先迁移，新生成的 KV blocks 以增量方式同步。目标节点完成校验后更新 KV owner，旧节点保留短 TTL 作为回退副本。

## 8. 模拟器设计

模拟器采用离散事件方式实现。核心事件包括：

- request_arrival：请求到达入口节点。
- route_decision：入口节点枚举动作并执行 SLA/memory 过滤。
- state_transfer_start：KV block 迁移开始。
- state_transfer_end：KV block 迁移完成。
- prefill_start 与 prefill_end。
- decode_start 与 decode_end。
- request_finish：请求完成并更新状态。
- metadata_sync：节点间同步负载、显存、网络和 KV 元数据。
- mobility_switch：测试过半后触发部分请求入口切换。

每个节点维护以下状态：

- waiting queue。
- running requests。
- estimated queueing delay。
- GPU memory used/free。
- KV cache used/free。
- local prefix/KV directory。
- recent P50/P95/P99 TTFT。
- recent end-to-end latency。

每个 session 维护以下状态：

- session ID。
- current entry node。
- current KV owner。
- KV size。
- prefix hash。
- request count。
- expected remaining requests。
- mobility phase。
- last access time。

## 9. 工作负载建模

CodeLlama34B 请求分为高优先级和普通优先级。高优先级请求使用 150 ms P99 TTFT SLA，普通请求使用 500 ms P99 TTFT SLA。两类请求共享同一组推理实例，但在 admission、排队和路由时采用不同 SLA 约束。

Qwen2-VL-7B-Instruct 请求使用 500 ms P99 TTFT SLA。由于多模态请求可能包含图像输入，模拟器应提高输入传输成本在路由决策中的权重，避免大规模输入频繁经过 25 Gbps 跨主机链路。

请求到达过程采用可配置分布：

- 基础版本使用 Poisson arrival。
- 每个客户端按目标并发数生成 session 请求。
- 每个 session 包含多个连续请求，用于模拟 prefix/KV 复用。
- 测试前半段请求入口等于初始就近节点。
- 测试后半段随机选择 20% session 或 request 切换到其他入口节点。

## 10. 评价指标

实验记录以下指标：

- P50/P95/P99 TTFT。
- 平均端到端时延。
- SLA violation ratio。
- 因显存不足导致的不可执行请求比例。
- 满足 SLA 的请求吞吐。
- 系统可支撑并发数。
- GPU memory 峰值占用。
- KV cache 峰值占用。
- Prefix/KV cache hit ratio。
- KV migration bytes。
- KV recomputation count。
- Cross-node request ratio。
- Cross-node state transfer ratio。
- 100 Gbps 链路利用率。
- 25 Gbps RDMA 链路利用率。
- Session owner switch count。
- 用户移动后的累计端到端时延。
- 用户移动后的跨节点通信时延。
- 请求转发次数。
- Greedy 状态黏附次数。
- Long-term routing 相对 Greedy 的累计成本下降。

主要对比目标为：

- 相比就近推理基线，并发用户规格提升 50%。
- 相比就近推理基线，平均端到端时延下降 20%。
- 在 P99 TTFT SLA 约束下，减少用户移动后的跨节点远程访问成本。

## 11. 实验流程

实验分为九个阶段。

第一，初始化三个节点、网络拓扑、模型配置、SLA 约束和 prefix cache 目录。

第二，生成包含 CodeLlama34B 高优先级请求、CodeLlama34B 普通请求和 Qwen2-VL-7B-Instruct 请求的混合工作负载。

第三，运行静态就近卸载基线，入口节点直接执行请求，记录 SLA、时延、吞吐、显存和 KV cache 指标。

第四，运行 Greedy 最小时延路由，观察其在用户移动后是否产生状态黏附，以及是否增加长期跨节点通信。

第五，运行长期成本感知路由，使用有限时域滚动规划估计未来请求入口、节点负载和 KV 增长，并比较其累计成本。

第六，运行长期成本感知路由与低成本 KV 管理联合方案，在长期策略基础上加入 block-level KV 索引、节点间直接传输和增量状态同步，并比较其相对纯长期路由的状态传输量、KV 命中率和服务中断风险。

第七，从 session 维度评估长期优化效果，统计用户移动后的累计端到端时延、跨节点通信时延、请求转发次数、KV 迁移与重算次数、状态传输量、prefix/KV 命中率以及 session 的节点切换次数。

第八，进行消融实验。移除移动性预测，验证未来入口信息对长期决策的作用；仅考虑当前 session 的未来成本，验证全局节点负载和显存压力的影响；关闭 block-level KV 管理和增量同步，评估低成本状态迁移机制的贡献。

第九，对不同参数做敏感性分析，包括 20% 移动请求比例、100 Gbps 与 25 Gbps 链路差距、KV block 大小、session 长度、SLA 阈值和显存保护水位。各组实验采用相同请求轨迹重复运行，并报告平均值、P95/P99 指标及波动范围。

## 12. 预期结果

预期就近推理基线在测试后半段会出现负载不均衡和 KV owner 与入口节点分离问题。当入口节点负载升高或历史 KV 位于远端时，部分请求的 TTFT 和端到端时延会恶化。

Greedy 最小时延路由预计能缓解短期负载不均衡，并降低部分请求的当前时延。但在连续 session 中，它可能持续选择已有 KV 的远端节点，导致 KV owner 长期停留在非入口节点，形成状态黏附。

长期成本感知路由预计在以下方面优于 Greedy：

- 在用户移动后更早迁移或重算高复用 KV，减少长期远程访问。
- 更少使用 25 Gbps 跨主机链路迁移大规模 KV。
- 对 100 Gbps 直连节点之间的状态迁移更积极。
- 提高 prefix/KV 复用率，同时降低不必要的跨节点传输。
- 在相近 SLA violation ratio 下获得更低平均端到端时延和长期累计成本。

## 13. 后续实现计划

第一步实现离散事件模拟器，支持三节点拓扑、请求到达、队列、prefill/decode 时间和 KV cache 元数据。

第二步实现三类策略：就近推理、Greedy 最小时延路由、长期成本感知路由。

第三步加入 block-level KV cache 管理、KV migration、KV recomputation 和 prefix hit 模型。

第四步根据模拟结果绘制 TTFT 分布、端到端时延、SLA 违约率、KV 迁移量和长期累计成本曲线。

第五步将实验结论反馈到报告第 6.4 节和实验评估章节，形成“方案设计—模拟验证—指标分析”的闭环。
