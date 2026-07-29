# One-Page Paper Idea：RoutePrep

**暂定标题**：*RoutePrep: Late-Binding KV Preparation for SLO-Aware Stateful LLM Serving across the Computing Edge Continuum*

**论文类型**：CEC systems paper + online control

**目标 venue 范围**：优先面向 edge/distributed systems 与 networking venue；具体选择在完成 direct-risk 查新和 Phase-0 kill test 后确定。

**一句话贡献主张**：

> RoutePrep 利用不确定的 pre-arrival advisory，在共享 CEC backhaul 和有限 HBM 下准备少量可撤销、可验证的 SLO-feasible execution options，但不提前绑定执行节点；真实请求到达后，系统根据已经实现的队列、链路和状态就绪度选择 prepared option、state owner 或 exact-recovery fallback。

**当前证据状态**：这是待验证的 paper hypothesis，不是已经成立的贡献。现有模拟只说明 naive proactive preparation 可能变差，尚未证明 late-binding recourse 相对最优 single-target preparation 存在显著收益。

## 1. One-page summary

### 1.1 场景与问题

Cloud–Edge–Client Collaborative Computing（CEC）通过接入边缘、区域边缘和云端组成异构资源池，以吸收地域性突发负载并满足交互式 LLM 的时延目标。对于无状态请求，Router 可以把任务发往当前较空闲的兼容节点；对于多轮 LLM session，下一轮请求还依赖此前生成、不断增长并具有版本约束的 KV cache。远端 worker 即使拥有空闲 GPU，也可能因为缺少最新 KV、跨站点链路拥塞或 HBM 不足而无法在 TTFT SLO 内开始执行。

用户开始输入、Agent 上游步骤启动等 advisory 为请求到达前准备 KV 提供了时间窗口，但 advisory 只提供概率性信息。系统不知道该提示是否一定转化为请求、请求准确到达时间、未来输入输出长度，以及未来的队列、链路和 HBM 状态。[SYMPHONY](https://www.usenix.org/conference/nsdi26/presentation/agarwal) 在 advisory 到达时预选一个执行节点，再向该节点 staging KV；如果 advisory 与请求到达之间的系统状态发生变化，目标节点可能已不再是最佳执行位置，而准备好的状态可能无法被使用。

本文把这一损失定义为 **early-binding regret**：系统在不确定性尚未实现时提前承诺唯一执行节点，导致请求到达时失去更好的路由动作。把完整 KV 复制到多个节点可以保留更多选择，但会消耗共享 backhaul、HBM-byte·time，并挤占活动推理。因此，CEC 中需要解决的问题不是“预测唯一最佳节点并尽量搬完 KV”，而是：

> 在有限准备预算下，应为哪些 session 创建哪些未来执行选项，以及请求到达后应如何行使这些选项，才能最大化长期 SLO-goodput？

### 1.2 核心抽象：Preparation Creates Route Options

在 advisory 时刻 $\tau_i$，系统只观察信息集 $\mathcal I_i(\tau_i)$，包括：

- session、模型与 KV version；
- 当前 state owner、各节点连续有效的 exact-KV boundary；
- 当前 queue、HBM 和逐链路负载；
- 校准后的 advisory 成真概率与 lead-time 分布；
- 由已完成请求得到的输入、输出和执行时间分布。

系统不知道真实 future queue、link state、输入输出长度、剩余 session 轮数或用户轨迹。令 $\ell$ 表示 serving engine 可以正确恢复和继续执行的合法 layer/block boundary。若为 session $i$ 在候选 worker $w$ 准备到边界 $\ell$，则请求到达后的 TTFT 风险由以下部分组成：

$$
T_{i,w}(\ell)
=
T_{i,w}^{\mathrm{queue}}
+
T_{i,w}^{\mathrm{remainingKV}}(\ell)
+
T_{i,w}^{\mathrm{first}}.
$$

RoutePrep 为候选 worker 计算最小 routeability boundary：

$$
\ell_{i,w}^{\min}
=
\min
\left\{
\ell:
\Pr
\left(
T_{i,w}(\ell) \le D_i^{\mathrm{TTFT}}
\mid
\mathcal I_i(\tau_i)
\right)
\ge 1-\epsilon
\right\}.
$$

达到 $\ell_{i,w}^{\min}$ 不表示 partial KV 可以脱离剩余状态直接执行，而表示“已准备状态 + 请求到达后的 exact residual recovery”在给定风险水平下可以满足 SLO。未达到合法边界的准备可以降低剩余恢复时间，但不一定创造新的 SLO-feasible route。

请求到达时 $t_i$，系统观察实现状态 $\xi_i(t_i)$，重新计算可行选项集合：

$$
\mathcal F_i(a_i,\xi_i)
=
\left\{
w:
T_{i,w}^{\mathrm{queue}}(\xi_i)
+
T_{i,w}^{\mathrm{remainingKV}}(a_i,\xi_i)
+
T_{i,w}^{\mathrm{first}}(\xi_i)
\le D_i^{\mathrm{TTFT}}
\right\},
$$

其中 $a_i$ 是 advisory 阶段完成的 preparation action。RoutePrep 再从 $\mathcal F_i$、state owner、on-demand exact transfer 和 recomputation fallback 中选择真实执行动作。

### 1.3 RoutePrep 的三个控制阶段

1. **Routeability-certified option generation**  
   对每个 session–worker–path 组合，计算最小合法准备边界、完成 deadline、置信水平、剩余恢复方式和预期有效期；不可在 SLO 内完成的候选不进入 option set。

2. **Budgeted option selection**  
   在逐链路 backhaul、节点 HBM、foreground priority 和 speculation-waste budget 下，从 no-op、owner-sticky、single-option、少量 multi-option、full preparation 中选择动作。优化单位是“新增 SLO-feasible route 的概率价值”，不是已复制的 KV 字节数。

3. **Arrival-time recourse**  
   请求到达后不机械遵循 advisory 阶段的 target，而根据实现的 queue、link、HBM 和 KV readiness 选择 prepared option；如果全部 option 失效，则回退到 owner、reactive transfer 或 exact recomputation。

长期目标是在相同 offered trace 和 admission rule 下最大化 SLO-goodput：

$$
\max_{\pi}
\liminf_{T\rightarrow\infty}
\frac{1}{T}
\sum_{r:t_r\le T}
\mathbb E_{\pi}
\left[
\mathbb I
\left(
T_r^{\mathrm{TTFT}}\le D_r^{\mathrm{TTFT}},
\;
T_r^{\mathrm{TPOT}}\le D_r^{\mathrm{TPOT}}
\right)
\right],
$$

同时满足逐链路 background traffic、speculative HBM-byte·time 和 unused preparation 的长期预算，以及每个时刻的物理容量和 foreground inference 优先约束。Lyapunov 或 online primal-dual 可以作为实现预算控制的方法，但不构成论文的新颖性。

### 1.4 核心可证伪主张

> 当 advisory window 内的 queue/link/HBM drift 足以改变最佳执行节点，而完整 top-$k$ replication 又受到共享 backhaul 和 HBM 限制时，RoutePrep 的 option preparation 与 late-binding recourse 能在相同资源和信息预算下，比 single-target early binding、arrival-only routing 和 full replication 获得更高的 SLO-goodput。

该主张要求同时证明：

1. early-binding regret 在真实或可信 trace 中频繁出现，并显著影响 SLO；
2. preparation 实际能够在 advisory lead time 内完成到合法 routeability boundary；
3. multi-option recourse 相对最优 single-target policy 存在不可忽略的 oracle gap；
4. 在线 RoutePrep 能在无 future oracle 的条件下收回大部分 gap；
5. 收益确实来自 CEC 的异构节点和共享逐链路约束，而不是额外复制资源。

## 2. Abstract draft：严格 10 句、140 words

1. Cloud–edge–continuum platforms pool distributed accelerators for latency-sensitive, multi-turn LLM serving.  
2. Growing key–value caches nevertheless bind sessions to prior workers and strand remote capacity.  
3. Advisory-driven prefetching can move state before a request arrives, but existing schemes commit early to one target.  
4. Queue, link, and memory conditions may then change, wasting preparation and causing SLO violations.  
5. We formulate speculative route preparation under uncertain arrivals and shared edge resources.  
6. RoutePrep creates cancellable execution options by preparing exact cache boundaries needed for SLO-feasible recovery.  
7. Its scheduler prices backhaul, memory, and speculation waste while protecting foreground inference.  
8. At arrival, RoutePrep selects a prepared option, the owner, or exact recovery using realized state.  
9. We evaluate RoutePrep on a heterogeneous prototype and trace-driven workloads against early-binding, reactive, replication, and SLO-aware baselines.  
10. **[Result placeholder]** RoutePrep improves SLO-goodput by X%, reduces P99 TTFT by Y%, and cuts unused preparation by Z%.

十句话依次承担：场景、一般问题、已有机会与限制、问题影响、技术问题定义、方法一、方法二、方法三、评测方法和结果。第 10 句必须在实验完成后用预注册主结果替换，当前不能写入估计数字。

## 3. Introduction 草稿：段落逻辑与 Topic Sentence

### P1：CEC 的价值是跨站点弹性，而不只是近端执行

**A geographically distributed accelerator pool helps only when an arriving request can use remote capacity before its deadline.**

介绍接入边缘、区域边缘和云端的异构资源池，以及地域突发负载下利用远端 spare capacity 的必要性。段末指出“物理空闲 GPU”不等于“当前请求可用的容量”。

### P2：多轮 LLM 将 task routing 变成 stateful routing

**Multi-turn LLM sessions make remote execution stateful because each request depends on a growing, versioned KV cache left by prior turns.**

解释 KV cache 的规模、版本和位置依赖。普通卸载只需传输入并选择算力节点，而 session request 必须同时获得兼容模型和最新 exact state。段末引出 state locality、排队与网络之间的耦合。

### P3：Advisory 创造准备窗口，也引入提前承诺风险

**Pre-arrival advisories expose time for state movement but do not reveal the execution node that will remain best when the request arrives.**

说明用户输入事件或 Agent workflow signal 能提供 lead time，但 future queue、link、HBM、输入输出长度和 advisory 成真与否都未知。给出 single-target early binding 的失败时间线，并定义 early-binding regret 与 stranded preparation。

### P4：已有工作覆盖了组成部分，但没有证明 late-binding option preparation 的价值

**Existing systems separately provide proactive KV staging, SLO-aware routing, or joint cache–routing control, but these mechanisms do not establish whether speculative preparation should create route options rather than commit to one route.**

按问题族定位相关工作：[SYMPHONY](https://www.usenix.org/conference/nsdi26/presentation/agarwal) 覆盖 advisory 和跨节点 staging，[KVFlow](https://papers.neurips.cc/paper_files/paper/2025/hash/b7971d31a7d5eb0f1eed2f8f6f368195-Abstract-Conference.html) 覆盖 workflow-aware future prefetch，[Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) 和 [DualMap](https://iclr.cc/virtual/2026/poster/10006475) 覆盖请求到达后的 SLO/cache/load-aware routing，[Randomization Boosts KV Caching, Learning Balances Query Load](https://iclr.cc/virtual/2026/poster/10009529) 已统一建模 KV eviction 与 query routing。段末把剩余问题限定为 uncertain advisory 下的 preparation recourse，而不是声称首次 KV placement、SLO routing 或 cache–routing 建模。

### P5：核心洞察是准备“选择权”，而不是预测唯一 winner

**The value of speculative KV preparation lies in the future SLO-feasible routes it creates, not in the cache bytes it copies.**

解释 routeability boundary 和 option value。准备不足时某些字节可能无法新增可满足 SLO 的节点；跨过合法边界后，Router 才获得新的执行动作。该阈值和多节点共享资源使 per-byte greedy、blind full replication 和 target-first decomposition 可能做出不同决策。

### P6：RoutePrep 用两阶段 recourse 连接 KV Manager 与 Router

**RoutePrep prepares a small, budgeted option set before arrival and binds the request only after uncertainty is realized.**

概述 option generator、resource-priced selector、arrival-time recourse 和 fallback。强调 preparation 可撤销、active inference 优先、状态版本一致，以及所有在线决策只能使用当前信息与校准分布。

### P7：贡献必须由问题刻画、结构 gap、系统机制和实测结果共同支撑

**RoutePrep is useful only if late binding recovers a material SLO gap that stronger single-target and replication baselines cannot close.**

Introduction 末尾列出四项待验证贡献：

1. 首次系统刻画 advisory-driven stateful serving 中的 early-binding regret、best-worker flip 和 stranded preparation；“首次”只能在 ICDCS direct-risk 全文核验后保留；
2. 提出 preparation-created route options 的两阶段 CEC 控制模型和 routeability certificate；
3. 设计利用合法 readiness threshold 与 option complementarity 的在线选择算法，而非泛用优化器替换；
4. 在真实多层原型与 trace replay 上，按相同 offered load、backhaul bytes 和 HBM-byte·time 验证 SLO-goodput、尾延迟和浪费。

段末只预告真实主结果，并明确适用条件与停止条件。

## 4. Claims–Evidence Matrix

| Claim | 所需证据 | 当前状态 | 计划位置 |
|---|---|---|---|
| C1：advisory 到 arrival 之间经常发生 best-worker flip，并导致 single-target preparation 浪费或违约 | 真实/可信 advisory trace、状态漂移分解、Fig. 2 | 未验证 | §2 |
| C2：相同资源下，multi-option recourse oracle 显著优于最优 single-target early-binding oracle | 小规模精确 oracle、配对 trace、Fig. 3 | 未验证，核心 kill test | §2–3 |
| C3：RoutePrep 能在不使用 future truth 的条件下收回大部分 oracle gap | 原型与 replay 主结果、Fig. 5 | 未实现 | §6 |
| C4：收益由 CEC heterogeneity 与 shared per-link contention 产生 | flat fabric、homogeneous worker、dedicated-link 对照，Fig. 6 | 未验证 | §6 |

当前项目中的 `proactive_only` 在并发 24 和 36 时分别使平均 TTFT 变差约 0.76% 和 0.86%，只能作为“naive action selection 会失败”的初步证据。该模拟仍读取真实剩余 session request 数，不能支撑 C1–C4。

## 5. 文章章节与图表预算

按 10–12 页双栏 systems paper 骨架规划；若目标会议只有 8 页并包含 references，应压缩 §2、§5 和 §7，而不是删除核心 oracle 或资源公平性实验。

| 章节 | 本章唯一中心句 | 预计页数 | 图表 |
|---|---|---:|---|
| 1. Introduction | 提前绑定使 KV preparation 失去请求到达时的路由选择权。 | 1.25 | Fig. 1 |
| 2. Motivation and Characterization | 必须先证明 best-worker drift、准备时间窗和 early-binding regret 同时存在。 | 1.50 | Fig. 2 |
| 3. Model and Design Goals | Preparation action 改变的是到达后的 SLO-feasible route set。 | 1.00 | Table 1 |
| 4. RoutePrep Design | 系统生成、选择并维护少量可撤销 route options，再在 arrival 时 recourse。 | 2.50 | Fig. 3–4 |
| 5. Implementation | 版本化 exact-KV data plane 支持合法边界、取消、foreground priority 和 fallback。 | 1.00 | Table 2 |
| 6. Evaluation | 收益必须在相同信息、offered trace、字节和 HBM-time 预算下成立。 | 3.50 | Fig. 5–6 |
| 7. Related Work and Limitations | 按 preparation timing、binding timing 和 state movement 范围划定贡献边界。 | 0.75 | 无 |
| 8. Conclusion | 总结 route-option preparation 的成立条件，而不泛化到所有 CEC 负载。 | 0.25 | 无 |

**主文预算：6 张图、2 张表。**

1. **Fig. 1 — Hero timeline**：左侧展示 SYMPHONY 式 `advisory → bind w1 → stage → w1 overloaded at arrival`；右侧展示 `advisory → prepare {w1,w2} options → observe state → choose w2`。图中必须同时标注相同 preparation budget 和未使用状态，避免把收益误画成额外复制资源。
2. **Fig. 2 — Problem existence**：展示 advisory lead time、best-worker flip rate、SLO violation decomposition、stranded bytes 与状态大小/负载/链路的关系。
3. **Fig. 3 — Routeability certificate**：以合法 KV boundary 为横轴、TTFT quantile 为纵轴，标出最低可行边界、过期时间和 queue/link drift 后的边界移动。
4. **Fig. 4 — System architecture**：展示 advisory plane、versioned KV directory、option generator、budget controller、Router 和 foreground/background 共享数据面。
5. **Fig. 5 — Main result**：在 matched backhaul 和 HBM-time 下比较 SLO-goodput–resource Pareto，同时报告 P99 TTFT 和 unused preparation。
6. **Fig. 6 — Causality and robustness**：包含 flat/hierarchical topology、homogeneous/heterogeneous worker、advisory precision/lead time、telemetry staleness 和 HBM budget 的多面板消融。
7. **Table 1 — Model and action space**：符号、状态、可观测信息、不可观测 future truth、action 和约束。
8. **Table 2 — Baseline fidelity and overhead**：逐项说明 baseline 是否支持 advisory、single/multi-target、partial/full state、arrival recourse、SLO routing，以及决策和数据面开销。

## 6. 实验设计

### 6.1 Phase 0：先做问题存在性与 oracle kill test

在实现完整系统前，先用实测 transfer/compute profile 和事件驱动模拟回答：

1. advisory 是否有用：advisory-aware offline oracle 相对 arrival-only 最优路由是否有稳定收益；
2. recourse 是否有用：在完全相同的 preparation bytes、HBM-time 和信息条件下，multi-option recourse oracle 是否优于最优 single-target oracle；
3. joint control 是否必要：joint preparation+routing oracle 是否优于 `SYMPHONY + optimal SLO target selector`；
4. partial readiness 是否真实有用：合法 boundary preparation 是否优于 full top-$k$ replication 和 per-byte marginal utility；
5. CEC 是否是一阶因素：把拓扑改成 flat fabric、节点改成 homogeneous、链路改成 dedicated 后，收益是否显著消失。

建议的继续标准不是论文结论，而是工程立项阈值：

- recourse oracle 相对 single-target oracle 的 SLO-goodput 中位提升至少约 10%，配对置信区间下界为正；
- joint oracle 相对最强分解策略至少约 10%；
- `SYMPHONY + SLO selector` 若与 recourse oracle 差距低于 3%–5%，停止该核心主张；
- 若只有极端低带宽、极端 HBM 或人为高漂移才能产生 gap，停止 CEC 普适性主张；
- 在完成 [ICDCS 2026 直接相关论文](https://icdcs2026.icdcs.org/program/main-technical-sessions/)全文核验前，不进入“首次”类贡献表述。

### 6.2 原型与 replay

最小原型包括两个 access edges、一个 regional edge 和一个 cloud worker。所有 eligible workers 部署兼容模型、tokenizer、精度和 KV layout，但具有不同 GPU service rate 和 HBM quota。端边、边边和边云路径具有不同 RTT、带宽和共享瓶颈；foreground request、reactive recovery 与 background preparation 必须竞争同一真实或受控链路，不能只在目标函数中加入静态传输 penalty。

原型负责测量：

- 不同 context、合法 KV boundary 和目标节点下的 transfer/recompute latency；
- queueing、prefill、decode、PCIe 和 network interference；
- background preparation 对 active inference 的影响；
- cancellation、version invalidation、fallback 和 route decision overhead。

经原型校准的离散事件 replay 扩展到更多 edge sites 和 sessions。只有当 replay 能复现实机 P50/P95 TTFT、queue time、link utilization 和 waste，模拟结果才能支撑主 claim。

### 6.3 Workloads 与信息边界

- **Sessions**：多轮对话为主；Agent workflow 作为具有更明确 advisory 的补充场景，而不是唯一场景。
- **Advisory**：优先采集真实前端 typing/start、tool-call 或 workflow events；无法获得时，必须把合成 lead time、precision、recall 和 calibration 明确标注并做全范围扫描。
- **Models**：至少两个 KV-per-token、prefill/decode profile 不同的模型；VLM/VLA 只有在实测显示状态或传输占比改变 phase boundary 时才加入。
- **Loads**：低负载、本地热点但全局有 spare capacity、全局过载三类；主要结论不能只来自极端过载。
- **Uncertainty**：扫描 advisory precision/recall、lead time、arrival jitter、输入输出长度误差、queue/link drift 和 telemetry delay。
- **Oracle boundary**：future request、future output length、future queue/link 和真实剩余 session 轮数只能供离线 oracle 使用。

### 6.4 强基线

1. state-owner sticky；
2. load-only routing + reactive exact recovery；
3. cache/load/SLO-aware arrival router；
4. 原始 SYMPHONY 式 single-target even-load preparation；
5. SYMPHONY + 最优可实现 SLO target selector；
6. top-$k$ full replication；
7. equal-share 或 load-proportional partial preparation；
8. per-byte marginal-utility / water-filling；
9. KVFlow/Pythia-style next-state preparation，在其适用 workload 中复现相同信息能力；
10. single-target offline oracle；
11. multi-option recourse offline oracle；
12. RoutePrep。

所有方法共享相同的 KV data plane、offered trace、admission rule、foreground priority、background bytes、HBM-byte·time、可观测信息和模型 profile。

### 6.5 Research Questions

| RQ | 问题 | 主要指标/图 |
|---|---|---|
| RQ0 | early-binding regret 是否真实存在且足够大？ | flip rate、stranded bytes、违约分解，Fig. 2 |
| RQ1 | preparation、single-target 与 recourse 的 oracle gap 各有多大？ | SLO-goodput、P99 TTFT、resource-matched gap |
| RQ2 | RoutePrep 是否能在线收回 oracle gap？ | main Pareto、gap recovery，Fig. 5 |
| RQ3 | 哪些 action 与组件真正产生收益？ | option count、certificate、late binding、fallback 消融 |
| RQ4 | 对错误 advisory 和陈旧 telemetry 是否稳健？ | precision/recall、lead time、staleness，Fig. 6 |
| RQ5 | 收益是否确实来自 CEC？ | topology/node/link counterfactual，Fig. 6 |
| RQ6 | 控制和数据面代价是否可接受？ | decision latency、control bytes、interference、waste |

### 6.6 指标

主指标：

- offered-load SLO-goodput；拒绝、丢弃和超时均计为失败；
- P50/P95/P99 TTFT，并按 queue、remaining-KV、prefill 和 network 分解；
- TPOT SLO violation，确认 preparation 没有通过干扰 decode 转移成本。

资源和正确性指标：

- preparation bytes、unused bytes、cancelled bytes；
- speculative HBM-byte·time、KV version invalidation；
- per-link utilization、foreground interference；
- option hit rate、fallback rate、best-worker flip rate；
- decision latency、telemetry/control overhead；
- 按 session length、state size 和 ingress class 分层的公平性。

## 7. 新颖性边界与明确不做

RoutePrep 不声称首次提出 advisory、KV prefetch、partial/layer-wise loading、SLO-aware routing、cache–routing 联合建模、多候选路由或 CEC KV migration。SafeRL、Lyapunov、CMDP、MILP 和移动性也不是贡献；移动性最多是造成路径或 ingress 不确定性的一类 workload。

当前最重要的 direct risk 是 ICDCS 2026 的 *Efficient KV Cache Migration for Geo-Distributed LLM Inference in Collaborative Edge Computing*，以及 arXiv-only 的 Pythia 等并发工作。若 direct-risk 全文已经覆盖 uncertain advisory、multi-target preparation、arrival-time route choice 和 SLO/resource budgets，应停止或重新定义 RoutePrep。

本文第一版不同时研究模型 placement、KV 压缩、有损状态、batch formation、PD request splitting、隐私、能耗和多模型质量选择。这些因素可以作为兼容性讨论或未来工作，但不能用来堆叠贡献。

## 8. Go / Pivot / Stop

- **Go**：问题存在性、recourse oracle gap、在线 gap recovery 和 CEC causality 均通过，且强基线无法用相同信息和资源追平。
- **Pivot**：preparation 有价值但 late binding gap 很小，则改为更窄的 SLO-aware SYMPHONY scheduler/engineering paper，不再声称 route-option abstraction。
- **Stop**：advisory lead time 不足、KV 不是主要违约来源、best worker 很少改变、强 single-target policy 已接近 oracle，或 direct-risk 工作已经完整覆盖。

这份 one-page 的中心不是证明 RoutePrep 必然有效，而是把一篇可被尽早证伪的论文主张写清楚。
