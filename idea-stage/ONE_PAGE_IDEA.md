# One-Page Paper Idea：RoutableKV

**暂定标题**：*RoutableKV: Activating Stranded Capacity for Long-Session LLM Inference across the Cloud–Edge Continuum*

**标题备选**：*Prepare to Offload: Unlocking Stranded Edge Capacity for Long-Session LLM Inference*

**论文类型**：CEC systems paper

**目标 venue 范围**：以 ACM/IEEE SEC 为首选，会议方向包括 IEEE ICDCS 和 ACM Middleware，期刊方向包括 IEEE TCC、IEEE TPDS 和 FGCS。

> **待验证的一句话主张**：RoutableKV 旨在端、接入边缘、区域边缘和云端组成的异构 CEC 系统中，以“使一条候选执行路径满足 SLO 所缺少的最小 KV 状态”为调度单位，在共享回传链路和有限 HBM 下，将被 session state 阻塞的远端空闲算力转化为实际可卸载的服务容量。

## 1. One-page summary

### 1.1 研究场景

Cloud–Edge–Client Collaborative Computing（云边端协同计算，CEC）通过客户端、接入边缘、区域边缘和中心云组成分层推理资源池。客户端请求从固定的接入边缘进入；接入边缘提供低访问时延但算力和 HBM 有限，区域边缘具有更强的共享能力，云端拥有更高计算容量但服务路径更长。不同层级的 GPU 性能、显存容量、排队状态和模型服务能力并不相同，层级间链路还具有非对称带宽、RTT、抖动和共享拥塞。客户端负责请求产生、接入和业务 SLO，端侧执行不在本文范围；核心卸载决策发生在兼容的 edge–cloud workers 之间。

边缘请求可能呈现地域性、突发性和负载偏斜，其出现频率和持续时间由 RQ1 使用真实或公开 trace 验证。本地接入边缘可能在短时间内形成长队列，而邻近边缘、区域边缘或云端仍有空闲计算资源。对于无状态任务，Router 可以直接把请求卸载到这些节点；对于长 session LLM，请求还依赖此前累积的 KV cache。最新 KV 通常保留在先前执行节点，其他节点只保存旧 prefix 或完全没有状态。远端 GPU 因而可能“物理空闲但对该 session 不可用”：若在请求到达后再恢复状态，TTFT 将超过 SLO。

本文将这种现象定义为 **state-induced stranded capacity**。它表示某个兼容节点具有足够的计算能力，却因 session state、服务路径或显存条件不满足而无法在目标时延内接管请求。CEC 的研究问题不是简单寻找最短队列，而是判断哪些分散算力值得通过状态准备被激活，以及有限回传带宽和 HBM 应优先激活哪些 session–path 组合。

### 1.2 问题定义

系统包含多个同时活跃的长 session。每个 session 具有固定 ingress、当前 context version、不断增长的 KV state、TTFT SLO 和节点 eligibility mask。候选节点必须部署兼容的同模型副本；模型权重、tokenizer、位置编码、精度和 KV layout 必须允许精确复用。模型放置、跨模型转换和回答质量选择不属于本文的决策范围。

令 $K_s(t)$ 表示 session $s$ 在时刻 $t$ 的完整 KV block 数，$x_{s,j}(t)$ 表示节点 $j$ 已持有的、覆盖所有 Transformer layers 的最新连续 token-prefix blocks。任意稀疏 blocks 不能被视为可执行状态；partial prefix 本身也不能直接用于 decode，节点必须在执行前通过 suffix transfer 或 exact recomputation 补齐缺失部分。

定义候选执行方案 $q=(j,p)$，其中 $j$ 是目标 worker，$p$ 同时指定 ingress 到 worker 的请求路径、状态源到 worker 的恢复路径，以及首 token 返回 ingress 的路径。给定已准备 prefix $x$，方案 $q$ 的预测 TTFT 为：

$$
T_{s,q}(x,t)
=
T_q^{req}(t)
+
T_j^{wait}(t)
+
T_{s,q}^{suffix}(K_s(t)-x,t)
+
T_{s,j}^{first}(t)
+
T_q^{ret}(t).
$$

其中，$T_{s,q}^{suffix}$ 是请求到达后恢复剩余 suffix 的精确传输或重算时间，$T_{s,j}^{first}$ 只包含当前增量输入的 prefill 和生成首 token 的计算，$T_q^{ret}$ 只包含首 token 返回 ingress 的时间。路径时延按照为 foreground 保留的服务包络估计，避免 readiness threshold 随后台调度器自身的瞬时带宽分配循环变化。所有分布均来自在线校准的节点和链路 profile，而不是未来真值。

使候选执行方案获得 SLO 服务资格的最小状态量为：

$$
x_{s,q}^{SLO}(t)
=
\min
\left\{
x \in \left\{0,1,\ldots,K_s(t)\right\}
\mid
P\left(T_{s,q}(x,t)\le D_s\right)\ge 1-\epsilon
\right\}.
$$

如果该集合为空，则令 $x_{s,q}^{SLO}(t)=+\infty$，并从 preparation candidates 中剔除方案 $q$。达到有限 threshold 并不表示 partial prefix 可以直接执行，而是表示“已准备 prefix 加请求关键路径上的 residual exact suffix recovery”共同满足 TTFT SLO。对应的 routeability deficit 为：

$$
d_{s,q}(t)
=
\max
\left\{
0,
x_{s,q}^{SLO}(t)-x_{s,j}(t)
\right\}.
$$

当 deficit 被完整填补时，候选执行方案才成为新的 SLO-feasible offload option。低于该边界的准备仍可能减少恢复时间，但不会新增可满足 SLO 的服务容量。论文的算法贡献不来自边界公式本身，而来自：在多 session、多瓶颈路径和有限 HBM 的条件下，如何识别具有足够有效期的边界，并集中完成能够真正激活容量的 deficit。

主目标是在固定 offered trace、admission rule、回传预算和 HBM 预算下最大化 SLO-goodput：

$$
G_{SLO}
=
\frac{1}{T_m}
\sum_{r\in R} I_r.
$$

其中，当 offered request $r$ 在 deadline $D_r$ 内完成时 $I_r=1$，否则 $I_r=0$；拒绝、丢弃和超时请求均计为失败。P99 TTFT、drop ratio、per-class fairness、background bytes、HBM-time 和 foreground interference 同时报告，防止通过丢弃困难请求或额外消耗资源提高主指标。

### 1.3 RoutableKV

RoutableKV 采用两个时间尺度，但只围绕一个控制原语组织系统：**path-qualified KV readiness**。

1. **Hierarchical telemetry plane** 收集每个节点的排队工作量、实测服务速率、可用 HBM 和模型能力，以及每条端边、边边和边云路径的 RTT、残余带宽、共享队列和前台预留。异构节点上不能只比较 queue length，必须估计请求级 remaining work。
2. **Versioned KV directory** 记录每个 session 在各节点的最高连续有效 prefix、版本、占用量和 in-flight transfer。旧副本可以通过增量 suffix 更新继续使用，但版本或布局不兼容的副本立即失效。
3. **Readiness estimator** 根据实测 transfer、recompute、queue 和 compute profile，估计每个 session–candidate path 的 $x_{s,q}^{SLO}$、置信区间和有效截止时间。队列、链路或 session version 越过 guard 后，qualification 立即过期。
4. **Deficit scheduler** 把每个 preparation job 表示为 session、目标节点、源节点与路径、缺失 prefix、完成 deadline、HBM demand 和预期新增服务容量。在 per-link backhaul 和 per-node HBM 约束下，它优先完成能够在失效前跨过边界的 deficit，支持抢占、取消和 foreground priority，而不是把资源平均分散到所有 partial replicas。
5. **Online Router** 只消费当前 readiness、队列、路径和 eligibility。请求优先进入已 qualified 且风险最低的节点；没有安全远端路径时，系统回退到 state-owner execution、reactive suffix transfer 或 recomputation。

在线系统允许使用 session age、已完成轮数、历史 inter-turn interval、当前 context length、当前请求的真实输入长度、实时 queue 和 link telemetry，以及由历史数据校准的 continuation、下一轮到达时间和 state growth 分布。真实剩余轮数、未来请求长度、未来输出长度、未来负载和未来链路状态只用于 clairvoyant Oracle。

### 1.4 核心主张

> **在具有固定 ingress、异构分层算力、共享受限 backhaul 和有限 HBM 的 CEC 系统中，按 routeability deficit 集中准备 KV，能够在相同资源预算下，比 state-sticky、reactive recovery、full prefetch、advisory-driven prefetch 和逐字节 marginal-utility allocation 获得更高的 SLO-goodput。**

该主张要求同时证明三个事实：

1. 真实或可信 trace 中，state-induced stranded capacity 具有足够高的出现频率和 SLO 影响；
2. 负载与 route qualification 的有效期长于状态准备时间，使非 Oracle 的后台准备可以及时完成；
3. 多 session 竞争共享路径时，完成少量有价值 deficit 的收益严格优于将相同字节和 HBM-time 按连续边际效用摊开。

## 2. Abstract draft：严格 10 句

> Cloud–edge–client collaborative computing (CEC) can serve interactive large language models close to users while pooling accelerators across access edges, regional edges, and the cloud. Long-running sessions weaken this pooling because their key–value caches grow across turns and remain attached to prior execution nodes. When an access edge becomes overloaded, remote spare capacity may be unusable for that session because transferring or rebuilding the missing state over constrained backhaul would violate the request latency objective. State-sticky routing preserves queue hotspots, reactive recovery puts state reconstruction on the critical path, and full replication may consume scarce bandwidth and accelerator memory at targets that still cannot meet the latency objective. We present RoutableKV, a CEC serving system that prepares state according to the remote capacity it makes usable rather than the cache bytes it places. For every session and compatible execution path, RoutableKV estimates the minimum prefetched prefix for which the remaining suffix can be recovered while the path still meets a time-to-first-token service-level objective (SLO). A budgeted scheduler concentrates shared backhaul and memory on readiness deficits that can finish before expiry and add SLO-feasible service capacity. A lightweight online router consumes measured readiness, queue, path, and capability information and falls back to state-owner execution, on-demand recovery, or recomputation when no prepared path is qualified. We evaluate RoutableKV on a heterogeneous multi-tier prototype and trace-driven workloads against state-sticky, load-only, cache-aware, full-prefetch, marginal-utility, and advisory-driven baselines under matched bandwidth and memory budgets. [Final-result placeholder] RoutableKV increases the rate of offered requests meeting the SLO by X%, reduces 99th-percentile time to first token by Y%, and cuts unused prepared bytes by Z%, showing that path-qualified state readiness can turn stranded edge capacity into usable serving capacity.

十句话依次承担：CEC 场景、状态障碍、问题后果、已有方法缺口、系统总览、核心抽象、后台调度、在线路由、评测设计和定量结论。最后一句必须在实验完成后替换为真实结果。

## 3. Introduction：段落逻辑与主题句

### P1：CEC 需要跨层资源池化

**Under geographically skewed bursts, cloud–edge inference requires usable capacity across tiers to meet stringent latency objectives.**

介绍客户端、接入边缘、区域边缘和云端构成的分层资源池。强调 CEC 的价值不仅是把模型放在近端，而是在地域负载突发时利用跨站点容量维持服务质量。段末引出“物理存在的 GPU 不一定构成可用容量”。

### P2：长 session 将卸载变成 computation–state co-optimization

**Long-running sessions make CEC resource pooling stateful because every routing decision leaves a growing KV cache that constrains subsequent placement.**

说明普通任务卸载只需考虑输入、计算和 deadline，而多轮 LLM 请求还依赖持续增长的历史 KV。一次路由不仅决定本轮执行位置，还改变后续 state locality。段末把问题从 request placement 推进到 computation–state co-optimization。

### P3：CEC 特征使状态错位成为一阶成本

**Heterogeneous accelerators and constrained asymmetric backhaul turn misplaced session state into stranded serving capacity.**

说明接入边缘、区域边缘和云端在 service rate、HBM、排队和路径成本上的差异。远端节点即使空闲，也可能因状态恢复和链路竞争无法满足 TTFT。段末给出 state-induced stranded capacity 的定义。

### P4：现有策略没有直接优化“可卸载容量”

**Existing routing and state-preparation policies do not directly expose path qualification as the state-manager/router interface under shared CEC links.**

按机制组织相关工作：state-sticky 固化热点，load-only 忽略状态，reactive transfer/recompute 占用关键路径，full prefetch 浪费回传和 HBM，cache-aware routing 只能利用已经存在的副本。明确研究缺口不是简单组合 Router 与 KV Manager，而是缺少以新增 SLO-feasible path 为单位的资源分配。

### P5：关键洞察是 routeability threshold

**KV preparation becomes capacity-creating when it helps a candidate path cross the SLO-feasibility boundary.**

定义 $x_{s,q}^{SLO}$ 和 deficit。解释低于边界的进度可能降低残余恢复，却没有新增服务容量；跨过边界后，Router 才获得一个新的卸载动作。段末将资源分配问题转化为多 session、多路径、带 deadline 的 deficit completion。

### P6：RoutableKV 将状态准备与在线路由连接起来

**RoutableKV makes path-qualified readiness the interface between background state management and online request routing.**

概述 telemetry、KV directory、readiness estimator、deficit scheduler 和 Router。强调 foreground 优先、过期失效、fallback 和非 Oracle 信息边界。该段只解释系统设计，不展开实现细节。

### P7：贡献与结果预告

**RoutableKV is designed to determine when state blocks CEC capacity and whether resource-aware readiness improves end-to-end service under realistic contention.**

列出三项可检验贡献：

1. 刻画 CEC 中 state-induced stranded capacity 的机会频率、持续时间和 phase map；
2. 提出 path-qualified readiness 与多 session、per-link、HBM-constrained deficit scheduler；
3. 在多层异构原型和校准 replay 上验证 SLO-goodput、尾延迟、资源效率、鲁棒性和控制开销。

段末仅预告真实主结果，不重复 Abstract，也不写无法由实验支撑的“首次”或“最优”。

## 4. 文章结构与图表预算

以下按 12 页 systems paper 设计。

| 章节 | 本章唯一中心句 | 页数 | 图表 |
|---|---|---:|---|
| 1. Introduction | Session state 使分散算力不能自动成为 CEC 服务容量。 | 1.25 | Fig. 1 |
| 2. Motivation and Characterization | State、queue、heterogeneity 和 shared backhaul 共同决定远端容量是否可用。 | 1.50 | Fig. 2, Table 1 |
| 3. Model and Design Goals | 目标是在固定资源预算下最大化 offered-load SLO-goodput。 | 0.75 | 无 |
| 4. RoutableKV Design | 系统优先完成能在有效期内打开新卸载路径的最小 deficit。 | 3.00 | Fig. 3, Fig. 4 |
| 5. Implementation | 版本化 block data plane 和在线 telemetry 支撑可抢占的渐进准备。 | 1.00 | Table 2 |
| 6. Evaluation | 所有收益必须在相同 backhaul、HBM-time 和 offered trace 下成立。 | 3.50 | Fig. 5, Fig. 6 |
| 7. Related Work | 与 cache-aware routing、distributed KV、prefetch 和 CEC offloading 按机制边界比较。 | 0.75 | 无 |
| 8. Conclusion | 总结状态就绪如何决定 CEC 算力的实际可用性。 | 0.25 | 无 |

**总预算：6 张主图和 2 张表。**

1. **Fig. 1 — Motivation and thesis**：采用 before/after 两半构图，只突出 fixed ingress、local queue、shared backhaul、remote compatible worker、current prefix、residual suffix recovery 和 SLO-feasible set 的变化。Sticky、reactive full recovery 和 blind full prefetch 仅作为灰色短注释，不各自展开完整流程。
2. **Fig. 2 — Measured CEC phase map**：以 queue gap、backhaul、state size 和异构 service rate 为轴，标出 local、sticky、reactive transfer、recompute 和 proactive prepare 的可行区域。
3. **Fig. 3 — Routeability and validity**：展示 valid-prefix blocks 与 TTFT quantile 的关系，并标出 $x_{s,q}^{SLO}$、residual suffix recovery、估计区间、qualification lifetime 和失效 guard。
4. **Fig. 4 — System architecture**：展示 ingress、层级 telemetry、KV directory、readiness estimator、deficit scheduler、Router 和共享数据面闭环。
5. **Fig. 5 — Main Pareto result**：固定 HBM 下比较 SLO-goodput–background bytes Pareto，并同时标注 P99 TTFT。
6. **Fig. 6 — CEC causality and robustness**：对比 full CEC、flat fabric、homogeneous nodes、dedicated backhaul、stateless control，并展示预测误差和遥测延迟。
7. **Table 1 — Testbed and workload**：节点、模型、KV layout、HBM quota、链路、session 和 SLO 配置。
8. **Table 2 — Baseline fidelity and overhead**：各 baseline 的 routing、transfer、recompute、prefetch 能力，以及 RoutableKV 的 decision latency 和 telemetry overhead。

## 5. 实验设计

### 5.1 原型与校准 replay

最小真实拓扑包含两个固定 ingress/access edges、一个 regional edge 和一个 cloud worker。客户端作为 workload generator 和接入端；主实验只在部署兼容同模型副本的节点间迁移 KV。节点使用不同的计算能力和 HBM quota，网络通过真实或受控共享瓶颈构造非对称端边、边边和边云路径。Foreground request forwarding、reactive recovery 和 background preparation 必须竞争同一链路与 PCIe/HBM 资源，不能只在代价函数中增加一个静态传输时间。

原型测得不同模型、节点、context length 和 prefix fraction 下的 prefill、decode、transfer、recompute、queue 和 interference profile。经原型校准的离散事件 replay 再扩展到更多 access edges、regional edges 和数百至数千个 sessions；若 replay 无法复现原型上的 P50/P95 TTFT、queue time 和 link utilization，模拟结果不用于支撑主要 claim。

### 5.2 Workloads 与变量

- **模型**：至少两个 KV-per-token 和 compute profile 不同的长对话 LLM；所有 eligible tiers 使用兼容权重、精度、tokenizer、位置编码和 KV layout。
- **会话**：公开或真实多轮会话 trace 提供 context growth、inter-turn interval 和 session continuation；合成 trace 只用于独立控制状态大小与到达过程。
- **负载**：覆盖低负载、非极端的局部过载和全局过载；重点构造本地热点与远端 spare capacity 同时存在的区间。
- **网络**：覆盖不同层级、非对称路径、共享瓶颈、抖动和突发拥塞；高速 full-bisection fabric 作为非 CEC control。
- **资源**：扫描 HBM quota、background bandwidth、session 数量、context length、节点 service-rate ratio 和 telemetry staleness。
- **SLO**：在实验前按应用或服务等级固定，不能根据结果选择阈值。

### 5.3 Baselines

1. ingress-local；
2. state-owner sticky execution；
3. nearest-edge/cloud spillover；
4. load-only routing + reactive restore；
5. cache/load-aware routing + measured transfer-or-recompute；
6. top-1 full prefetch；
7. equal-share 或 load-proportional partial preparation；
8. marginal-utility/water-filling + deadline-aware allocation；
9. 机制完整的 advisory-driven prefetch scheduler，包括错误 advisory、多个候选 target、partial loading 和优先级；
10. RoutableKV + 与 baseline 相同的 Greedy Router；
11. clairvoyant Oracle，仅作为上界。

所有方法必须共享同一 KV data plane、offered trace、admission rule、HBM quota、foreground priority 和 background-byte budget。

### 5.4 Research questions

| RQ | 核心问题 | 关键实验 | 主要指标 |
|---|---|---|---|
| RQ1 | State 是否在真实 CEC 参数下阻塞远端容量，routeability threshold 是否稳定？ | stateful/stateless control，并扫描 prefix、context、queue、bandwidth、RTT 和节点能力 | blocked-capacity ratio、boundary error、false qualification、qualification lifetime |
| RQ2 | RoutableKV 是否以更少资源提升端到端服务容量？ | 在不同 offered load 下配对比较，并对齐 background bytes 与 HBM-time | SLO-goodput、P95/P99 TTFT、bytes per SLO saved、unused bytes、fairness |
| RQ3 | CEC 特征是否真正改变策略？ | flatten topology、homogenize compute/HBM、dedicate backhaul、remove ingress locality | 相对增益、动作排序、phase-map shift |
| RQ4 | 非 Oracle 条件下是否仍有效？ | continuation、load、state growth、link estimate 和 telemetry delay 扰动 | SLO-goodput、wasted preparation、regret to Oracle |
| RQ5 | 控制器能否扩展并稳定运行？ | 扩展 session、node、path 和 tenant 数量；结果汇总到 Table 2 | decision latency、metadata traffic、oscillation、foreground slowdown |

### 5.5 必要消融

- 去掉 routeability threshold，改为逐字节 marginal utility；
- 去掉集中完成，改为按概率或负载均匀摊薄；
- 去掉 per-path shared-bottleneck awareness；
- 去掉 versioned stale prefix，只允许 binary full hit/miss；
- 去掉 expiry、guard 和任务取消；
- 去掉 HBM-time opportunity cost；
- 用 Oracle future 替换在线 estimator，量化信息差距；
- 在相同 Manager 下比较 Greedy Router 与短窗口 Router，判断复杂 Router 是否产生独立收益。

## 6. CEC 因果检验

论文必须以因果消融证明“edge”改变了问题，而不是给普通集群调度更换部署名称。

| 对照环境 | 保持不变 | 被移除的 CEC 特征 | 预期需要观察的变化 |
|---|---|---|---|
| Full CEC | offered trace、总算力、总 HBM | 无 | RoutableKV 在共享多层瓶颈下选择性激活远端容量 |
| Flat fabric | offered trace、总算力、总 HBM | 层级路径和 per-link contention | preparation target、优先级和相对收益明显变化 |
| Homogeneous workers | topology、offered trace、总算力 | service-rate 与 HBM 异构 | path qualification 和 offload 层级发生变化 |
| Dedicated backhaul | topology、offered trace、节点 | 前后台共享拥塞 | full prefetch 的代价下降，RoutableKV 优势缩小 |
| Stateless control | arrival、compute demand、节点和链路 | 跨轮 KV dependency | state-induced stranded capacity 消失 |
| No local burst/skew | 总 offered load、节点和链路 | 地域负载偏斜 | 跨层容量激活需求显著下降 |

如果 flatten topology、消除节点异构和解除 shared-backhaul constraint 后，本文的动作排序与相对收益基本不变，则实验只能支持通用 distributed LLM serving 结论，不能支持 CEC-specific claim。

## 7. 新颖性边界

| 相邻方向 | 已经解决的部分 | RoutableKV 必须额外证明的部分 |
|---|---|---|
| [Preble](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html) | 根据已有 KV locality 与负载进行请求路由 | 主动改变状态分布，并在相同资源下新增 SLO-feasible execution paths |
| [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) | 全局 KV 存储、cache-aware routing、transfer/recompute 与热点复制 | 受 CEC per-path bottleneck 和有限 HBM 约束的、off-critical-path capacity activation |
| [SYMPHONY](https://www.usenix.org/conference/nsdi26/presentation/agarwal) | 基于不可靠、多目标 advisory 的 proactive KV loading、partial loading 和优先级调度 | 在相同 advisory、multi-target 和 partial-loading 能力下，证明 CEC per-link bottleneck 与 qualification-bound completion 能改变资源效率 Pareto |
| [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao) | 负载驱动的请求与 KV live migration | 请求间隙中的渐进准备、过期控制及 CEC backhaul/HBM 联合预算 |
| [CachedAttention](https://arxiv.org/abs/2403.19708) | 多轮会话的持久化分层 KV、layer-wise preload 和异步保存 | 跨节点准备如何在共享路径预算下新增 SLO-feasible serving capacity |
| [Low-Latency Edge LLM Handover](https://arxiv.org/abs/2603.28018) | handover 触发的 KV transfer、token prefill 和 backhaul scheduling | 固定 ingress 下由 load skew 驱动的多层 offload，以及多 session 容量激活 |
| [Tail-Optimized Caching for LLM Inference](https://papers.nips.cc/paper_files/paper/2025/hash/f05fe8b796dcbd67bc7bb1ea89df1793-Abstract-Conference.html) | 以尾延迟阈值决定保留多少缓存 | 跨节点主动准备如何改变 Router 的可行动作和远端服务容量 |

“minimum prefix”“Router + KV Manager”或“联合优化”本身不构成新颖性。可发表的机制差异必须体现在多路径共享瓶颈下的 completion-aware allocation，并在相同 bytes 和 HBM-time 上超过机制完整的 marginal-utility 与 advisory-driven baselines。

## 8. Falsification gates

1. **Problem-existence gate**：真实或可信 CEC workload 中，远端算力因 state recovery 无法满足 SLO 的比例及其造成的 SLO loss 必须具有工程意义；否则停止该系统主线。
2. **Timescale gate**：load persistence 和 qualification lifetime 必须显著长于 preparation completion time；否则 proactive readiness 不可部署。
3. **Runtime-semantics gate**：真实 runtime 必须支持正确的连续 prefix 增量准备、suffix 恢复、版本检查和结果一致性；否则 progressive readiness 不成立。
4. **Boundary-structure gate**：若 partial preparation 的价值近似平滑，强 marginal-utility baseline 与 RoutableKV 位于同一 Pareto 前沿，则 routeability 只是 break-even 的重新命名。
5. **CEC-causality gate**：full CEC 环境中的动作和收益必须与 homogeneous flat fabric 明显不同；否则不能声称 CEC-specific contribution。
6. **Non-Oracle gate**：未知 continuation、请求长度、负载和链路条件下仍需保持稳定收益；真实未来只能作为上界。
7. **Interference gate**：后台准备造成的前台链路、PCIe 和 HBM 干扰全部计入后，端到端收益仍必须存在。
8. **Strong-baseline gate**：在相同 offered trace、background bytes 和 HBM-time 下，RoutableKV 必须在至少两个非极端负载区间超过最强非 Oracle baseline。建议预注册至少 10% 的 SLO-goodput 工程阈值，并要求 paired 95% confidence interval 不跨 0。
9. **Fairness gate**：收益不能来自更多 drop、饿死长 session 或牺牲特定 tenant；必须同时报告分组 SLO 达成率。

## 9. 预期论文结论

若上述 gates 通过，论文能够支持的最小结论是：

> 在 CEC 中，分散算力只有在模型兼容性、服务路径、队列、KV readiness 和 HBM 同时满足业务 SLO 时才构成真实服务容量。通过在共享 backhaul 下选择性完成 path-specific KV deficit，RoutableKV 可以比反应式恢复和无差别预取更高效地激活跨层计算资源。

该结论把 CEC 特征直接写入问题可行域和系统机制，并以可否证的端到端实验验证，而不是将 edge 仅作为部署背景。
