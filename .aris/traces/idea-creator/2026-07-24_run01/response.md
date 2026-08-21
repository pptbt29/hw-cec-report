# Independent jury response

## Initial verdict

- **首选：C2 Batch-Ready Execution Options**
- **备选：C5 Multi-Source State Assembly**
- **判决：C2 仅值得进入严格 go/no 验证，现有材料不足以形成可投稿的新颖性主张。**

六维分数顺序为：问题重要性 / 结构新颖性 / 正式工作可区分性 / 可证伪性 / 完整论文潜力 / 目标适配。

| 候选 | 重要性 | 新颖性 | 可区分性 | 可证伪性 | 论文潜力 | 目标适配 | 核心意见 |
|---|---:|---:|---:|---:|---:|---:|---|
| C1 | 8 | 5 | 5 | 9 | 8 | 9 | MASS 2023 已通过调度提升 edge batching opportunity，EcoServe 已有 macro-instance、adaptive routing 和 scaling；必须证明 domain formation 改变了逐请求代价无法表达的未来服务曲线。 |
| C2 | 7 | 6 | 6 | 9 | 8 | 9 | 全局 placement 与 set-dependent batch service 可能不可分，但 TWC 2024、Libra 和 JITServe 已覆盖相邻空间；剩余新意只能是可提交 cohort option。 |
| C3 | 7 | 5 | 5 | 8 | 7 | 8 | 过期遥测是真问题，但 leases、reservation 和 optimistic concurrency 已成熟；给 lease 增加 KV/HBM 字段不够。 |
| C4 | 6 | 7 | 6 | 8 | 7 | 7 | executable closure 较新，但模型常驻、adapter 较小时会退化，必须证明真实 orphan state。 |
| C5 | 6 | 7 | 7 | 8 | 8 | 9 | AND-join 的最大完成时间和 transfer/recompute/local-encode 的 OR 分支形成非平凡计划空间；风险是场景人为或单一 artifact 支配。 |
| C6 | 7 | 5 | 5 | 8 | 7 | 8 | stage partition 和 split inference 拥挤，必须证明 continuous batching 或 representation reuse 改变最优边界。 |
| C7 | 8 | 6 | 5 | 6 | 7 | 6 | VLA freshness 重要，但易被视为 AoI 应用，且有效期曲线任务依赖强。 |
| C8 | 7 | 6 | 6 | 7 | 7 | 7 | typed information-flow 和 policy-valid planning 基础深厚，安全范围容易淹没调度主线。 |
| C9 | 5 | 5 | 5 | 9 | 6 | 8 | 接近 CDN/multicast、batching window 和 cache placement，依赖空间相关 burst。 |
| C10 | 6 | 7 | 6 | 8 | 7 | 7 | state-aware partial hedge 有细化，但受益区可能窄，且经典 hedging 理论接近。 |
| C11 | 9 | 4 | 4 | 9 | 7 | 8 | 问题重要，但 JITServe 已处理 imprecise request information 和 SLO-aware batch composition。 |

### 对 C2 的最强攻击

1. C2 可能只是 column generation 的系统包装。若 option 只是把本地可行 batch 枚举成 columns，再让全局 set packing，这不是新结构。
2. 任意 joint oracle 都比先 placement 后 batching 拥有更大动作空间；只展示 oracle gap 近乎同义反复。必须在信息、预测能力和求解预算完全匹配时证明逐请求接口仍不能表达关键交互。
3. option 可能因 continuous-batch 状态逐 token 改变而快速失效；CEC 控制 RTT 和 reservation waste 可能抵消收益。
4. 真正收益可能只存在于狭窄中负载区；低负载没有 cohort，高负载会自然形成饱和 batch。
5. 强 per-request router 若读取 marginal service curve、link dual price 和 KV/HBM shadow price，可能已逼近 options。

### C2 的三个严格 go/no 条件

1. 在至少两个真实模型/执行器 profile、两个共享瓶颈 CEC 拓扑和三类到达过程中，非预知未来的 joint oracle 相对信息匹配的最佳 separated oracle，SLO-goodput 中位提升至少 10%，且至少一半配置不低于 15%。
2. 最强逐请求 vector-price router 仍不能将 gap 缩小到 3% 内；batch service 线性化后收益下降至少 70%，移除共享链路后下降至少 30%。
3. 在线 top-$K$ option controller 回收至少 70% oracle gap，相对 Libra-like、JITServe-like 和强 batch-aware router 净提升至少 10%，同时 option 失效率和未使用 reservation 均低于 5%。

### C2 应收缩的变量

删除 request splitting、运行中迁移、locality-domain scaling、stale telemetry 独立贡献、输出长度预测、VLM/VLA stage、adapter、privacy、mobility、energy、fairness、RL 和 Lyapunov。固定同一基础模型、同构 GPU、两层 CEC、单一 SLO-goodput、admission-time placement，KV 仅作为 feasibility/affinity。

### 初始 one-sentence delta

相较于 Libra 由全局调度器选择逐请求 token split、再由实例局部形成 SLO batch，以及 JITServe 在不精确请求信息下优化 batch composition，C2 唯一可守的增量是让 worker 暴露带短时提交保证的完整 cohort option，并由全局在多跳 CEC 共享链路与 HBM 约束下选择兼容 option 集。

## Follow-up verdict after SHEPHERD and PPipe

**改变：C2 不再作为首选，改选 C5。**

[SHEPHERD](https://www.usenix.org/conference/nsdi23/presentation/zhang-hong) 已把完整 batch 作为调度决策变量，并在线生成各 GPU 的 candidate batch；[PPipe](https://www.usenix.org/conference/atc25/presentation/kong) 又覆盖了基于 GPU/NIC 未来可用性和资源预留的 batch-size、pipeline/path 联合选择。结合 Libra 与 JITServe 后，C2 剩余的“CEC 上 commit-ready autoregressive batch option”主要是既有 batch-level scheduling 向 continuous batching 和多层链路的迁移，结构新颖性不足，除非先发现既有 batch/path 表达无法表示的新现象。

新首选 **C5 Multi-Source State Assembly**。其最小可守 claim 是：

> 执行启动由分散必需状态的 AND-join 决定，同时每项状态具有传输、重算或本地编码的 OR 替代路径，并与执行节点及共享链路次序联合选择。

这不等价于 SHEPHERD 的单 GPU batch selection，也不等价于 PPipe 的单源顺序 pipeline probing。

## Independence limitation

本次评审使用 fresh-context 的 `gpt-5.6-sol`，未获得异构模型家族评审。结论只作为 advisory evidence；执行者在新增 SHEPHERD、PPipe、Chameleon 和 Omni-Flow 证据后再次进行了定向审计。

