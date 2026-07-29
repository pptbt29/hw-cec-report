# CEC 生成式 AI 任务调度：候选 Idea 评审档案

> 用途：供独立评审者在不知道执行者偏好的情况下，选择最值得继续验证的一个研究问题。  
> 状态：候选生成与机械去重稿；本文件不作预排序，不宣称任何候选已经通过新颖性审查。

## 1. 研究边界

目标是在 Computing Edge Continuum（CEC）中形成一篇以**任务调度**为中心的新论文。这里的 CEC 至少包含 access edge、metro/regional edge 与 cloud 中的两层，并具有以下客观特征：

- 算力、显存、模型状态与链路容量异构；
- 请求从地理分散的 ingress 到达，跨层传输共享瓶颈链路；
- 控制面可能是分层的，遥测存在延迟；
- 目标负载是自回归 LLM/VLM/VLA 推理，具有在线到达、未知输出长度、逐 token continuous batching 和增长中的持久状态；
- 论文不能假设未来请求、未来输出长度、未来 session 数量或用户移动轨迹已知；
- mobility 不是强制变量；只有它产生不可被普通负载波动替代的因果机制时才纳入；
- SafeRL、Lyapunov、DRL 或整数规划只能是求解手段，不能单独构成新颖性。

可接受的论文贡献应至少包含下列两项：

1. 一个在真实参数范围内可复现、现有调度抽象无法解释或处理的系统现象；
2. 一个由该现象直接导出的新调度抽象或机制；
3. 一个可落地的在线系统原型和强基线实验；
4. 一个有明确假设和边界的分析结果。

## 2. 已确认的正式发表重合边界

以下工作用于划定候选不能重复声称的贡献。

| 已有方向 | 正式代表工作 | 已覆盖内容 | 仍不能据此直接排除的窄缺口 |
|---|---|---|---|
| MEC 中联合 batching 与 task scheduling | [Cang et al., IEEE TWC 2024](https://doi.org/10.1109/TWC.2024.3404811) | 异步到达、多用户 edge AI、task-batch association、batch start time 与无线资源联合优化 | 自回归请求的逐 token 动态 batch、增长 KV 和 placement 对未来服务率的反馈 |
| Edge LLM 的 request scheduling 与 KV caching | [ReSK, IEEE IoT-J 2026](https://doi.org/10.1109/JIOT.2026.3709703) | 本地 KV 保留与请求放置联合优化 | 跨节点状态流、逐 iteration batch 演化和端到端 decode SLO |
| Edge LLM pipeline 与 batching | [IEEE TCOM 2026](https://doi.org/10.1109/TCOMM.2026.3686718) | 离线 batching、模型划分和带宽分配 | 在线未知长度、persistent session state 与动态服务域 |
| Edge-assisted inference 的 offloading 与 batch scheduling | [FGCS 2025](https://doi.org/10.1016/j.future.2025.108030) | 一次性 DNN 推理的卸载、压缩和 batch scheduling | 自回归 continuous batching 的状态依赖服务率 |
| Edge DNN 的 batching-opportunity-aware scheduling | [IEEE MASS 2023](https://doi.org/10.1109/MASS58611.2023.00071) | 通过请求调度与端边协同提升同模型/共享层请求的 batching opportunity | 多节点 CEC 中逐 token batch 演化与增长状态，但“调度创造 batching opportunity”本身已不新 |
| 商用 5G MEC 的 SLO-aware resource management | [SMEC, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/zhang-xiao) | RAN 与 edge server 解耦的 deadline-aware scheduling，并实测 coordination delay 与 resource contention | 生成式推理状态和 continuous-batch 外部性，但“CEC 协调延迟 + SLO 调度”本身已不新 |
| 分布式 LLM 的 KV-aware routing | [Preble, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html) | prefix reuse 与负载感知路由 | CEC 分层链路和可变服务域，但“KV-aware routing”本身已不新 |
| 分布式 KV 池和 SLO 调度 | [Mooncake, FAST 2025](https://www.usenix.org/conference/fast25/presentation/qin) | disaggregated KV pool、prefill/decode 调度 | 低速多层 CEC 下的任务池化边界 |
| KV load、cache 与 batch formation | [Strata, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang) | hierarchical KV load/cache、balanced batch formation | 不能再宽泛声称首次联合 KV 和 batching |
| 全局 request splitting、local SLO batching 与 KV transfer | [Libra, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/ruan-libra) | 全局与局部协调、chunked KV transfer | 不能再宽泛声称首次联合 routing、batching 和 KV transfer |
| Session KV storage、request ordering 与 continuous batching | [Bidaw, FAST 2026](https://www.usenix.org/conference/fast26/presentation/hu-shipeng) | session KV 持久化、request ordering、continuous batching | 单一集群之外的 CEC 调度边界仍需专门证明 |
| Commodity GPU cluster 的 macro-instance 与在线伸缩 | [EcoServe, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/du) | macro-instance、adaptive routing、mitosis scaling | 地理分层 CEC 的路径时延—请求池化边界，但“弹性实例组”本身已不新 |
| 未知请求信息下的 SLO-aware batch composition | [JITServe, NSDI 2026 technical sessions](https://www.usenix.org/conference/nsdi26/technical-sessions) | 逐步修正未知请求信息并联合 batch composition 最大化 goodput | CEC 跨节点任务放置，但“未知输出长度 + SLO batching”本身已不新 |
| Edge KV migration/reuse | [ICDCS 2026 technical program](https://icdcs2026.icdcs.org/program/main-technical-sessions/) | 正式接收了 collaborative edge KV migration 和 elastic edge KV reuse 工作 | 纯“边缘 KV 迁移”已是高重合方向 |

由此，以下宽泛表述不构成候选的新颖性：

- “首次联合任务调度、KV cache 和 batching”；
- “使用 RL/Lyapunov 解决边缘 LLM 卸载”；
- “根据 KV 命中率和负载选择节点”；
- “移动后迁移 KV 降低延迟”；
- “增加带宽 penalty 以满足长期 SLO”。

## 3. 机械去重后的候选池

### C1. Elastic Locality Domains：动态决定请求应被池化到多大的服务域

**问题。** CEC 中将请求留在最近节点可以降低路径延迟，却会把稀疏、突发的请求拆散到多个小队列，降低 continuous-batch occupancy，并把 session KV、prefix、model 或 adapter 状态分散开。把请求集中到 regional pool 会增加网络延迟，却可能显著提高 GPU 服务率和状态复用。每个请求的放置因而改变其他请求未来可获得的 batch opportunity，服务率是调度决策的内生变量。

**候选机制。**

- 慢时间尺度：根据 ingress 相关性、链路时延、batch density 和状态碎片化，动态合并或拆分 locality domain；
- 快时间尺度：在域内或相邻域之间，按 queue、state locality、deadline 与预计 batch opportunity 路由；
- 现有 vLLM/SGLang continuous batching 作为执行器，不重新发明 kernel。

**最小可证伪实验。**

1. 构造 iteration-level continuous-batch simulator；
2. 扫描 arrival sparsity、burst correlation、edge-to-regional RTT、模型大小和 session 长度；
3. 检查 SLO-goodput 对 active locality-domain 数量是否存在稳定的内部最优点；
4. 比较 nearest-edge、global JSQ、KV-aware routing、固定 regional pooling、全局 oracle 与在线 elastic-domain controller。

**最接近工作。** TWC 2024 的静态 task-batch association；Preble、Strata、Libra 的分布式 LLM routing/batching；CEC service placement 文献。

**主要风险。** 如果在真实 profile 下网络路径成本始终压倒 batching 收益，或所有 edge 都能稳定形成饱和 batch，则该现象不重要；“服务域动态伸缩”也可能被认为是 autoscaling/service placement 的实例，必须用内生服务率和 persistent-state fragmentation 证明本质差异。

**去重键。** `elastic-locality-domain`

### C2. Batch-Ready Execution Options：把全局 placement 与本地 continuous batching 作为不可分解决策

**问题。** 常见架构先由全局 router 给请求选节点，再由各节点独立组 batch。对于自回归推理，节点服务时间不是请求属性之和，而取决于同一 iteration 共存的请求集合、prefill/decode 混合、KV 占用和未知完成时刻。先路由后 batching 可能选择出每个请求局部看来合理、整体却无法形成高效 batch 的解。

**候选机制。** 局部 worker 不上报单一 queue length，而生成少量可提交的 batch-ready option，例如“某 cohort 在时间窗内以某 state action 进入 batch”；全局调度器在共享链路约束下选择兼容 options。

**最小可证伪实验。** 在小规模精确 oracle 上测量“全局 router + 最优 local batcher”相对 joint oracle 的 SLO-goodput gap；若 gap 稳定大于工程噪声，再实现 option generation 与在线选择。

**最接近工作。** Libra、Strata、LMetric、FastServe，以及 TWC 2024 joint batching。

**主要风险。** 可能只是将现有联合优化改写为 column generation；如果 global-local decomposition gap 很小，候选失败。

**去重键。** `batch-ready-option`

### C3. Stateful Capability Leases：过期遥测下的分层可提交调度

**问题。** 在跨自治域 CEC 中，中心 router 得到的 queue、HBM 和 KV 目录可能已经过期。普通状态同步会产生 false placement、重复状态拉取和跨 ingress 竞争。

**候选机制。** 局部集群签发短时 stateful capability lease，承诺兼容 model/KV、HBM 和启动时间窗；上层只在 leases 之间 admission，提交后用 token 防止重复消费，过期时仅做 suffix reconciliation。

**最小可证伪实验。** 扫描 5–200 ms 控制 RTT、遥测周期、突发程度和域数量，比较完美信息 oracle、周期性集中调度、聚合负载分层调度和 lease controller。

**最接近工作。** [Oakestra, USENIX ATC 2023](https://www.usenix.org/conference/atc23/presentation/bartolomeo)、[Omega, EuroSys 2013](https://research.google/pubs/omega-flexible-scalable-schedulers-for-large-compute-clusters/)、[RTSFaaS, USENIX ATC 2025](https://www.usenix.org/conference/atc25/presentation/zhao-jianjun)。

**主要风险。** 若测试规模只有少数节点，集中控制已经足够；若 lease 仅是在普通 reservation 中附加 KV 字段，贡献不足。

**去重键。** `stateful-capability-lease`

### C4. Executable-State Closure：调度可执行状态闭包而非单项 cache hit

**问题。** 一个请求可能需要基础模型、LoRA adapter、KV 版本、视觉编码器或模态特征同时就绪。独立优化各类 cache 可能留下大量“占用资源但不能组合成一次执行”的 orphan state。

**候选机制。** 用 AND/OR 状态依赖图表示必要状态及传输、加载、重算等替代动作；调度器优先物化能够完成请求的最小 executable closure。

**最小可证伪实验。** 多模型、多 adapter、多 session 场景中，先测 orphan-state ratio 和独立缓存策略到 closure oracle 的 gap，再决定是否做原型。

**最接近工作。** [GRAPHENE, OSDI 2016](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/grandl_graphene)、[ServerlessLLM, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/fu)、[dLoRA, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)、S-LoRA。

**主要风险。** 如果部署中基础模型永久驻留、adapter 很小且只有 KV 稀缺，则闭包退化为普通 KV-aware routing；需要真实 workload 证明 orphan state 不是人为构造。

**去重键。** `executable-state-closure`

### C5. Multi-Source State Assembly：把请求启动建模为多源状态 AND-join

**问题。** 原始图像可能在 access edge，KV suffix 在旧节点，adapter 在租户域，基础模型在 regional edge。执行节点的启动时间由多个必要状态流中最慢者决定，且这些流竞争共享链路。

**候选机制。** 为候选 worker 生成 state-assembly plan，将传输、重算或本地编码表示为 OR 分支，把所有必要子流表示成 assembly coflow，并联合选择执行节点、组装计划和逐链路次序。

**最小可证伪实验。** 真实测量 model、adapter、KV、视觉特征和原始视频的大小以及传输/重算时间，比较 scalar network cost、独立 artifact 调度、固定 placement + coflow 与 joint oracle。

**最接近工作。** [Varys, SIGCOMM 2014](https://conferences.sigcomm.org/sigcomm/2014/program.php)、[CLARINET, OSDI 2016](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/viswanathan)、[Sonic, USENIX ATC 2021](https://www.usenix.org/system/files/atc21-mahgoub.pdf)。

**主要风险。** 容易被评价为“coflow 应用于 LLM”；若一个最大 artifact 始终支配决策，多源建模没有实际价值。

**去重键。** `multi-source-state-assembly`

### C6. Representation-Aware Stage Scheduling：VLM/VLA 中传原始输入、表示还是 KV

**问题。** CEC 中 VLM/VLA 可以在 access、regional 或 cloud 切分视觉编码、prefill、decode/action head。不同阶段生成的 representation 大小、可复用性、隐私属性和 freshness 不同；固定 split point 或只比较 FLOPs 与输入字节可能选错路径。

**候选机制。** 在线联合选择 stage fusion/fission、representation materialization、execution node 和 per-link transfer，并复用共享视觉特征或 prefix。

**最小可证伪实验。** 对 2–3 个 VLM/VLA 做真实 profile，寻找 stage boundary 随链路和 batch occupancy 改变而发生策略反转的区域，再与固定 split、cloud-only、edge-only 和 latency-greedy 比较。

**最接近工作。** split inference、pipeline parallelism、CLARINET 式 plan selection，以及 TCOM 2026 edge LLM pipeline。

**主要风险。** stage scheduling 与 model partitioning 已很拥挤；如果没有生成式状态复用或动态 batch 带来的新结构，容易沦为已有 split inference。

**去重键。** `representation-stage-scheduling`

### C7. Component-Wise Freshness Scheduling：VLA/VLM 状态的组件级有效期

**问题。** 语言指令、长期地图、视觉场景、proprioception 和短期 action plan 的变化速度不同。最快的 stale-state inference 可能产生过期动作，因此 latency SLO 不等于任务成功。

**候选机制。** 离线标定各状态组件的 age/drift 与 action deviation 关系；在线选择 refresh subset、执行位置和 fallback，本地 safety guard 拒绝超出 validity envelope 的动作。

**最小可证伪实验。** 在 OpenVLA 或 DiffusionVLA 与仿真/真实机器人任务中，控制场景变化、网络抖动和 regional load，比较完整刷新、永久复用、统一 AoI threshold 与组件级策略。

**最接近工作。** AoI-aware MEC offloading、deadline-aware edge dispatch、VLA state reuse。

**主要风险。** 可能只是“AoI 应用于 VLA”；KV 跨视觉观测复用是否安全首先是模型实证问题，且真实机器人验证成本高。

**去重键。** `component-freshness`

### C8. Privacy-Typed State Transformation：把隐私边界变成可选择的执行计划

**问题。** 原始视频、embedding、KV 和 adapter 具有不同迁移权限。allowed/forbidden placement 无法利用“在本地编码或脱敏后即可跨域”的转换路径。

**候选机制。** 为状态附加 policy type 与 provenance，为节点和转换标注信任等级、延迟、字节与精度损失，联合选择 policy-valid transformation、execution node 和链路。

**最小可证伪实验。** 工业视频问答或机器人视觉任务中，比较 local-only、固定 split、TEE-only、privacy-oblivious upper bound 和 policy-aware planner。

**最接近工作。** [Gaia, NSDI 2017](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/hsieh)、[Ryoan, OSDI 2016](https://www.usenix.org/conference/osdi16/program)、split inference。

**主要风险。** 安全模型、可信执行和模型切分会扩大范围，容易偏离任务调度主线。

**去重键。** `privacy-typed-transform`

### C9. Cohort Multicast State Setup：利用 CEC 层级拓扑共享模型或 adapter 的分发

**问题。** 空间相关 burst 可能使多个 access edges 同时需要同一模型、视觉编码器或 adapter。独立 unicast warming 在共享上游链路重复传输，状态 setup cost 对目的节点集合不是可加的。

**候选机制。** 按模型、adapter、deadline 和路径重叠形成 cohort，联合决定等待窗口、regional root、分发树和物化层级；私有 KV 仍独立传输。

**最小可证伪实验。** 树状 CEC、空间相关 burst 和多 adapter 场景，比较 LRU、独立 unicast prefetch、固定 regional cache、CDN placement、cohort multicast 与 oracle。

**最接近工作。** hierarchical CDN/multicast、ServerlessLLM、dLoRA。

**主要风险。** 需要真实空间相关需求；overlay multicast 的工程收益可能有限，且主状态不是长 session KV。

**去重键。** `cohort-multicast-setup`

### C10. Stateful Hedging：只对可恢复状态做有界冗余执行

**问题。** CEC 尾延迟和节点波动可能使单路径任务违约，但直接复制长 session 请求会重复 KV、占用大量 HBM 和计算。冗余价值取决于可复用 prefix、取消点和共享链路。

**候选机制。** 根据 state reconstructability 与 deadline，仅复制 prefill、部分 decode 或轻量 shadow state，并设计快速取消与状态收敛。

**最小可证伪实验。** 扫描 failure/straggler rate、session KV 大小和网络相关性，比较无 hedging、全请求复制、经典 delayed hedging 和 state-aware bounded hedging。

**最接近工作。** tail-tolerant distributed systems、speculative inference、LLM preemption/state offload。

**主要风险。** 额外资源成本可能使收益只存在于罕见故障区；与现有 hedged requests 的差异需要清楚的状态成本模型。

**去重键。** `stateful-hedging`

### C11. Calibrated SLO Scheduling under Unknown Generative Work

**问题。** 在线调度不知道输出长度和未来 arrival。点预测可能系统性低估长尾工作量，并在共享 CEC 链路与 GPU 上连锁违约。

**候选机制。** 使用分布校准或 conformal bound 估计剩余 decode 工作量，将 admission、placement 和迁移动作限制在可审计的 violation budget 中；不依赖 future oracle。

**最小可证伪实验。** 在 workload shift、模型变化和非平稳到达下，比较点预测、分位数预测、Lyapunov/queue-based、distributionally robust 和 calibrated scheduler。

**最接近工作。** uncertainty-aware scheduling、chance-constrained MEC、remaining-length prediction 和 SLO admission。

**主要风险。** 容易变成把 conformal prediction 套到 scheduling；若 uncertainty 不改变最优动作，贡献不成立。

**去重键。** `calibrated-unknown-work`

## 4. 独立评审任务

评审者必须逐个评估 C1–C11，不得因为某个候选实现困难而不读。请按以下维度各给 1–10 分，并给出证据化理由：

1. **问题重要性**：在现实 CEC 参数范围内是否会造成足够大的 SLO-goodput、成本或可用性损失；
2. **结构新颖性**：是否引入现有 task offloading、batch scheduling、KV-aware routing 无法等价表达的新决策耦合；
3. **与正式先行工作的可区分性**：是否能用一句精确 claim 与最近顶会/顶刊工作划界；
4. **可证伪性**：是否存在 2–6 周内能决定 go/no-go 的最小实验；
5. **完整论文潜力**：是否能形成 characterization、method 与 system/evaluation 三个相互支撑的贡献；
6. **目标适配**：是否适合 CEC/MEC 顶级期刊/会议，并有机会扩展到系统/网络会议。

请输出：

- 一个首选候选和一个备选候选；
- 首选候选最强的拒稿理由；
- 首选候选必须满足的三个 go/no-go 条件；
- 建议删除或收缩的非必要变量；
- 与最近正式工作最精确的 one-sentence novelty delta；
- 若首选候选需要与另一个候选合并，只允许在两者共享同一根因时合并，并说明合并后为什么不是简单堆砌。
