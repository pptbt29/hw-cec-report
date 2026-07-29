# One-Page Paper Idea：RoutePrep

**暂定标题**：*RoutePrep: Late-Binding KV Preparation for SLO-Aware Stateful LLM Serving across the Computing Edge Continuum*

**论文类型**：CEC systems paper + online control

**一句话主张**：

> RoutePrep 在不可靠 advisory 到达后，不提前绑定唯一执行节点，而是在共享 CEC 路径与有限 HBM 下准备少量非绑定的执行选项；真实请求到达后，系统再根据更新后的队列、链路和状态就绪度选择 prepared option、state owner 或 exact-recovery fallback。

**证据状态**：这是一个待验证、可停止的 paper hypothesis。当前没有证据证明 late-binding recourse 显著优于最强 single-target 或组合基线。

## 1. One-page summary

### Problem

CEC 通过接入边缘、区域边缘和云端池化异构 GPU，以吸收地域突发负载并满足交互式 LLM 的时延目标。多轮 session 却具有不断增长、带版本约束的 KV cache；远端 worker 即使空闲，也可能因为缺少最新状态、共享路径拥塞或 HBM 不足而无法在 TTFT SLO 内接管请求。

用户开始输入、Agent 上游步骤启动等 advisory 提供了请求到达前的准备窗口，但系统不知道提示是否成真、准确到达时间、未来输入输出长度，以及未来 queue、path 和 HBM 状态。[SYMPHONY](https://www.usenix.org/conference/nsdi26/presentation/agarwal) 式 single-target staging 在 advisory 阶段选择一个 worker 并准备 KV；若到达前系统状态改变，原目标可能不再合适。完整 top-$k$ replication 可以保留选择权，却会消耗共享 backhaul、HBM-byte·time，并干扰活动推理。

本文研究的不是“如何更准确地预测唯一最佳 worker”，而是：

> 在有限准备预算下，哪些 session–worker–path 选项值得被提前变为 SLO-feasible，以及请求到达后如何行使这些选项？

### Core idea

一个 route option 记为 $q=(w,p)$，其中 $w$ 是兼容 worker，$p$ 指定 ingress request、KV recovery 和首 token 返回所经过的分层路径。CEC 的异构性通过逐跳共享瓶颈、路径不对称、worker service rate 和 HBM 显式进入模型，而不是只给 worker 增加一个标量网络 penalty。

RoutePrep 只准备已经提交、不可变且版本一致的 session prefix。准备边界 $\ell$ 表示“跨所有 Transformer layers 均完整存在的连续 token-block prefix”，而不是任意 KV 百分比或单独若干 layers。目标节点仍需通过 exact suffix transfer 或由相同历史 tokens 进行语义等价的 exact recomputation 补齐剩余状态；并发 version delta 会使旧 qualification 失效或要求追加同步。

在 advisory 时刻 $\tau_i$，RoutePrep 使用当前信息和校准分布判断候选 $q$ 在准备到边界 $\ell$ 后满足 TTFT SLO 的条件概率：

$$
P_{i,q}(\ell)
=
\Pr
\left[
T_{i,p}^{\mathrm{request}}
+
T_{i,w}^{\mathrm{queue}}
+
T_{i,q}^{\mathrm{remainingKV}}(\ell)
+
T_{i,w}^{\mathrm{first}}
+
T_{i,p}^{\mathrm{return}}
\le D_i^{\mathrm{TTFT}}
\mid
\mathcal I_i(\tau_i)
\right].
$$

若存在合法边界，则定义最小 risk-qualified boundary：

$$
\ell_{i,q}^{\min}
=
\min
\left\{
\ell:
P_{i,q}(\ell)\ge 1-\epsilon
\right\};
$$

不存在时记为 $\bot$。这只是基于预测分布的 risk qualification，不是确定性 certificate。请求到达后，系统用更新的信息 $\mathcal I_i(t_i)$ 重新计算条件概率，只在仍满足风险阈值的 options 中 late-bind；其余情况回退到 owner、reactive exact transfer 或 recomputation。

RoutePrep 的控制链为：

$$
\text{generate legal options}
\rightarrow
\text{select options under per-link/HBM budgets}
\rightarrow
\text{observe arrival-time state}
\rightarrow
\text{route or fallback}.
$$

主目标是在相同 offered load 和 admission rule 下最大化长期 SLO-goodput，并约束逐链路 background traffic、speculative HBM-byte·time、unused preparation 和 foreground interference。Lyapunov、online primal-dual 或 stochastic programming只是候选求解方法，不是贡献。

### Falsifiable claim

> 当 advisory window 内的 queue/path/HBM drift 经常改变最佳 route option，而完整复制又受逐链路和 HBM 约束时，两阶段 non-anticipating option preparation 与 late-binding recourse 能在相同信息和资源上限下，比 single-target stochastic preparation 和最强 prefetch–routing 组合获得更高的 SLO-goodput。

该 claim 只有在以下证据全部成立时才能保留：

1. early-binding regret、best-option flip 和 stranded preparation 在非极端 CEC workload 中真实存在；
2. advisory lead time 足以完成有用的跨层 token-prefix boundary；
3. two-stage recourse optimum 显著优于 non-anticipating single-target optimum；
4. RoutePrep 显著优于“multi-target partial preparation + arrival-time SLO router”等最强组合基线；
5. CEC heterogeneity 与 shared-path contention 对该 gap 具有显著放大作用。

## 2. Abstract draft：严格 10 句、143 words

1. Cloud–edge–continuum platforms pool distributed accelerators for latency-sensitive, multi-turn LLM serving.  
2. Growing key–value caches create state-locality costs that can make remote capacity miss request deadlines.  
3. Single-target advisory staging can move state before arrival but commits early to one worker.  
4. Queue, path, and memory conditions may then change, wasting preparation and causing SLO violations.  
5. We formulate speculative route preparation under uncertain arrivals and shared edge resources.  
6. RoutePrep creates evictable, nonbinding execution options by preparing exact cross-layer token-prefix boundaries for SLO-feasible recovery.  
7. Its scheduler prices path bandwidth, memory, and speculation waste while protecting foreground inference.  
8. At arrival, RoutePrep selects a prepared option, the owner, or semantics-preserving exact recovery using updated state.  
9. We evaluate RoutePrep on a heterogeneous CEC prototype and trace-driven workloads against early-binding, composite prefetch-routing, replication, and oracle baselines.  
10. **[Result placeholder]** RoutePrep improves SLO-goodput by X%, reduces P99 TTFT by Y%, and cuts unused preparation by Z%.

第 10 句必须由预注册主实验结果替换，当前不能填入估计数字。

## 3. Introduction：每段只表达一个意思

| 段落 | Topic Sentence | 本段任务 |
|---|---|---|
| P1：CEC opportunity | **A distributed accelerator pool helps only when an arriving request can use remote capacity before its deadline.** | 说明 CEC 价值是跨站点弹性，并区分 physical spare capacity 与 usable capacity。 |
| P2：Stateful obstacle | **Multi-turn LLM sessions make remote execution stateful because every request depends on a growing, versioned KV cache.** | 说明 KV locality、queue、path 和 HBM 的耦合。 |
| P3：Early-binding regret | **Pre-arrival advisories expose time for state movement but not the worker that will remain best at arrival.** | 给出状态漂移时间线，定义 best-option flip、stranded preparation 和 early-binding regret。 |
| P4：Prior-work boundary | **The unresolved question is whether speculative preparation should create route options rather than commit to one route.** | 准确定位 SYMPHONY/KVFlow 的 proactive preparation、Mooncake/DualMap 的 arrival routing 和已有 cache–routing modeling；不声称首次 KV placement 或 SLO routing。 |
| P5：Insight | **Preparation is valuable when it creates an additional SLO-feasible route, not merely when it copies more KV bytes.** | 引出跨层 token-prefix boundary、risk qualification 和 option value。 |
| P6：System | **RoutePrep prepares a budgeted option set before arrival and binds the request only after uncertainty is updated.** | 概述 option generation、budgeted selection、late-binding recourse、versioning 和 fallback。 |
| P7：Evidence | **RoutePrep is useful only if it beats optimized single-target and composite prefetch-routing policies under matched resources.** | 列出 problem characterization、two-stage model、structured policy、CEC prototype 四项待验证贡献，并预告真实结果。 |

## 4. 文章结构与图表

按 10–12 页双栏 systems paper 规划；若目标 venue 只有 8 页且 references 计入页数，应压缩背景与实现描述，不能删除 oracle 和 matched-resource 对比。

| 章节 | 内容 | 图表 |
|---|---|---|
| 1. Introduction | 问题、洞察、贡献和结果预告 | Fig. 1 |
| 2. Motivation and Characterization | lead time、best-option flip、违约归因和 early-binding regret | Fig. 2 |
| 3. Model and Design Goals | 两阶段信息结构、$(w,p,\ell)$ action、SLO-goodput 与预算 | Table 1 |
| 4. RoutePrep Design | risk qualification、option selection、arrival recourse | Fig. 3–4 |
| 5. Implementation | versioned token blocks、取消/驱逐、foreground priority、fallback | Table 2 |
| 6. Evaluation | 主结果、oracle gap、组合基线、鲁棒性和 CEC interaction | Fig. 5–6 |
| 7. Related Work and Limitations | 按 preparation/binding timing 与资源范围划界 | 无 |
| 8. Conclusion | 总结成立条件与适用边界 | 无 |

**主文预算：6 张图、2 张表。**

- **Fig. 1 Hero figure**：相同资源预算下对比 `single-target early binding` 与 `option preparation + arrival recourse`；必须显示未使用 bytes，避免把收益画成额外复制。
- **Fig. 2 Problem existence**：lead-time distribution、best-option flip、stranded bytes 和 SLO violation decomposition。
- **Fig. 3 Qualification**：跨层 token-prefix boundary 与 TTFT risk 的关系、$\ell_{i,q}^{\min}$、$\bot$ 和 qualification expiry。
- **Fig. 4 Architecture**：advisory plane、versioned directory、option generator、budget controller、Router 与共享路径。
- **Fig. 5 Main Pareto**：matched backhaul/HBM-time 下的 SLO-goodput、P99 TTFT 和 waste。
- **Fig. 6 CEC interaction/robustness**：hierarchical vs. flat topology、heterogeneous vs. homogeneous workers、shared vs. dedicated paths，以及 advisory/telemetry 误差。

## 5. 实验设计

### Phase 0：先判断问题是否值得研究

在小规模场景树或离散事件模拟中，使用相同预测分布、可观测信息和资源上限比较：

1. **Arrival-only optimum**：没有 pre-arrival action；
2. **Non-anticipating single-target stochastic optimum**：advisory 时只能选择一个目标，不能读取 future realization；
3. **Two-stage non-anticipating recourse optimum**：advisory 时准备 options，到达后根据更新信息选择；
4. **Clairvoyant perfect-information oracle**：准备时知道 future realization，只作为上述方法的共同上界。

不能把 clairvoyant single-target oracle 与 non-clairvoyant recourse 比较；前者已知道最终最佳节点，原则上不应被后者击败。

最强可实现组合基线必须包括：

- SYMPHONY staging + arrival-time reroute/fallback；
- optimized multi-target partial preparation + Mooncake/DualMap-style cache/load/SLO router；
- KVFlow-style future-state prefetch + location choice；
- full top-$k$ replication；
- per-byte marginal utility；
- owner-sticky、reactive exact recovery 和普通 arrival router。

建议的工程继续门槛：

- recourse optimum 相对 non-anticipating single-target optimum 的 SLO-goodput 中位提升约 10% 以上，且 paired confidence interval 下界为正；
- RoutePrep 相对最强组合基线仍有约 10% 净收益；若差距低于 3%–5%，停止 route-option abstraction；
- advisory lead time 不足、KV 不是主要违约来源或 best option 很少改变时停止；
- 在读完 [ICDCS 2026 直接相关论文](https://icdcs2026.icdcs.org/program/main-technical-sessions/)全文前，不使用“首次”类表述。

### Prototype and replay

最小原型包含两个 access edges、一个 regional edge 和一个 cloud worker；eligible workers 使用兼容模型、tokenizer、精度和 KV layout，但具有不同 service rate 与 HBM quota。端边、边边和边云路径具有非对称 RTT、逐跳共享瓶颈和突发拥塞。Foreground request、reactive recovery 与 background preparation 必须竞争同一链路和 PCIe/HBM 资源。

多轮对话是主 workload，Agent workflow 是 advisory 更明确的补充场景。真实前端/Agent events 优先；合成 advisory 必须明确报告 precision、recall、calibration 和 lead-time distribution。所有在线方法禁止读取 future request、future output length、future queue/link、真实剩余 session 轮数或 mobility trajectory。

### Research questions and metrics

| RQ | 要回答的问题 | 核心指标 |
|---|---|---|
| RQ0 | early-binding regret 是否真实且足够大？ | best-option flip、stranded bytes、违约分解 |
| RQ1 | late-binding recourse 是否存在结构性 oracle gap？ | resource-matched SLO-goodput gap |
| RQ2 | RoutePrep 是否能收回 gap 并胜过组合基线？ | gap recovery、P99 TTFT、SLO-goodput |
| RQ3 | 哪些机制产生收益？ | option count、qualification、late binding、fallback 消融 |
| RQ4 | 对 advisory 和 telemetry 误差是否稳健？ | precision/recall、lead time、staleness |
| RQ5 | CEC 因素是否显著放大收益？ | topology × heterogeneity × shared-path interaction |
| RQ6 | 控制与数据面代价是否可接受？ | decision latency、bytes、HBM-byte·time、foreground interference |

Flat fabric 或 homogeneous worker 不必让收益完全消失；合理主张是 CEC 的层级异构和共享路径对 recourse gap 产生显著、可量化的 interaction effect。

## 6. 新颖性边界与审稿结论

RoutePrep 不声称首次提出 advisory、KV prefetch、partial loading、SLO-aware routing、cache–routing 联合建模、多候选路由或 CEC KV migration。SafeRL、Lyapunov、CMDP、MILP 和移动性不是贡献。第一版也不同时研究模型 placement、KV 压缩、batch formation、PD splitting、隐私或有损状态。

独立审稿式复核对初稿给出 **Weak Reject（约 6/10）**：逻辑流和实验充分性较好，但新颖性尚未由证据建立，且初稿的 oracle 与 partial-boundary 定义不严谨。本版已经修正这两项，并加入最强组合基线；在 Phase-0 通过前，RoutePrep 仍应视为 **conditional idea，而不是 ready-to-write paper**。
