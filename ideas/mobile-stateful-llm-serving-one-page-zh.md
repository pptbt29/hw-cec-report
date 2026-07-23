# One-Page Idea Summary（中文版）

## 暂定标题

**Mobility-Aware Long-Term Request Routing and Predictive KV Cache Placement for Edge LLM Serving**

## 核心主张

移动 LLM serving 中，request routing 与 KV placement 不能独立优化：当前执行节点决定新增 KV 的位置，后台 KV 放置又改变未来各节点的状态就绪程度。本文以双时间尺度闭环统一二者：前台 Router 决定 request 在哪里执行，后台 KV manager 决定向哪些候选节点预复制哪些 KV blocks 以及投入多少资源。

## 中文摘要草稿（10 句）

将 LLM 部署在靠近用户的边缘节点可以降低交互式服务的访问时延。当移动用户切换接入节点时，多轮 session 的请求入口随之改变，但历史 KV cache 可能仍保留在原计算节点。继续将请求转发至原节点只需传输轻量输入与输出，却会在后续交互中反复承担跨节点通信和远端排队开销。将 KV 提前放置到潜在执行节点可以改善后续状态本地性，但预测错误或过度复制会消耗大量带宽与显存。因此，本文将该问题建模为移动、输出长度和 session 持续性不确定条件下，受 SLA、带宽和显存约束的长期请求路由与 KV 副本放置联合优化。我们提出一个双时间尺度框架，协同前台请求执行决策与后台 KV 状态准备。前台长期 Router 综合未来接入分布、节点负载和 KV 状态位置，为每个到达请求选择执行节点。后台 KV manager 根据多步移动预测和资源状态，分阶段向潜在执行节点放置可复用 KV，并在用户切换时完成残余状态恢复。我们将在多节点 LLM serving 原型上，使用不同模型、上下文、移动轨迹、网络条件和预测误差，与代表性的请求路由、被动 KV 恢复、启发式预放置策略及 Oracle 上界进行比较。**【结果占位：填写相对最强非 Oracle 基线的 P99 时延、SLA violation、handoff 阻塞和 KV 资源开销改进。】**

## Introduction 的学术结构

**P1 背景与问题。** 从 edge-hosted LLM 的低时延价值切入，限定具有多轮 KV 状态的移动 session；用用户从 $A$ 移动到 $B$、KV 留在 $A$ 的场景提出问题：应把 request 发回 state，还是提前在未来节点准备 state？

**P2 现有工作与缺口。** 用两类相关工作定位本文：现有 request routing 能依据链路、队列、SLA 和缓存位置选择执行节点，但通常不控制未来 KV 如何放置；distributed KV、prefetch 和 context migration 能准备或恢复状态，但通常不联合决定移动 session 的长期执行位置。该段用代表性文献和简洁对比明确缺口，当前版本不再设置独立 Related Work 章节。

**P3 测量证据与研究动机。** Profiling 展示不同模型/context 的 KV 大小、25/100 Gbps 下的 KV 传输时间，以及不同 RTT、排队差和未来驻留时间下 forwarding 与 placement 的 crossover，用数据检验两个假设：短 session 和大 KV 时保持状态本地性更合理；若在切换前逐步复制不可变 prefix，在线恢复量可由完整 KV 降为 residual tail。由此得到本文动机：系统既不能始终迁移 KV，也不能始终把请求发回原节点，而应联合决定当前执行位置和未来状态准备。Break-even 只用于解释测量和确定实验区域，不列为主要贡献。

**P4 技术挑战。** （1）未来入口、驻留时间、session continuation 和输出长度均不确定，Router 必须评估当前执行位置的长期影响；（2）KV manager 必须在多个目标节点、连续 blocks、带宽、显存和误预测代价之间分配资源；（3）Router 改变新 KV 的生成位置，而 placement 又改变后续恢复成本，两个控制环相互耦合。

**P5 方案概述。** 系统为每个候选节点维护 KV readiness profile，包括已就绪连续 prefix、残余同步量、预测恢复时间、显存需求和可行性。前台 Router 基于该 profile、移动/长度预测及节点负载选择执行节点；后台 KV manager 根据移动概率与 Router 估计的未来执行需求、预计切换前可用准备时间和资源余量分阶段建立副本，并随预测更新调整复制目标和进度。实验完成后，在本段末用少量数字预告 P99 E2E latency、handoff blocking time、KV transfer/wasted replication 和 SLA capacity 的主要结果。

**P6 贡献。** 按“统一问题与框架、两个核心算法、原型与实验验证”总结，不把 break-even、KV-local 策略、block directory 或 residual recovery 单独包装成贡献。

## 方法骨架

系统在 request arrival 和 background placement 两个时间尺度上最小化 session 的折扣长期服务成本，同时满足逐请求 SLA、节点显存和后台链路预算。KV manager 向 Router 暴露候选节点状态：

$$
\Phi_k(j)=\bigl(L_k^{\mathrm{ready}}(j),D_k^{\mathrm{residual}}(j),
\widehat T_k^{\mathrm{restore}}(j),M_{k,j}^{\mathrm{available}},
\chi_k^{\mathrm{feasible}}(j)\bigr),
\qquad
J_k=\pi_{\mathrm{route}}\left(s_k,\{\Phi_k(j)\}_{j\in\mathcal N}\right).
$$

其中：

- $k$：当前 session 中到达的第 $k$ 个 request；实际 session 长度事先未知。
- $j$：候选执行节点索引；$\mathcal N$ 为当前系统中的候选节点集合。
- $s_k$：Router 在 request $k$ 到达时观测到的全局状态，包括当前/预测入口、输出长度分布、节点队列、链路、显存和 KV 目录状态。
- $\Phi_k(j)$：KV manager 为节点 $j$ 生成的 KV readiness profile，即在该节点执行 request $k$ 前所需的状态准备信息。
- $L_k^{\mathrm{ready}}(j)$：节点 $j$ 已经持有且版本一致的最长连续 KV prefix 所覆盖的 token 数；只有连续 prefix 可以直接复用。
- $D_k^{\mathrm{residual}}(j)$：在节点 $j$ 执行前仍需补齐的 KV 字节数，等于当前请求所需 KV 与该节点可复用 KV 的差集大小。
- $\widehat T_k^{\mathrm{restore}}(j)$：按照当前恢复策略，通过增量同步或重算补齐 residual KV 的预测关键路径时间。
- $M_{k,j}^{\mathrm{available}}$：扣除模型、运行时和安全裕量后，节点 $j$ 当前可用于 request 及其 KV 的显存余量。
- $\chi_k^{\mathrm{feasible}}(j)\in\{0,1\}$：节点 $j$ 的硬可行性指示量；只有模型/adapter 版本匹配、节点和链路可用、且预计满足 SLA 与显存约束时取 1。
- $\pi_{\mathrm{route}}$：长期 Router 的决策策略；$J_k$ 是该策略为 request $k$ 选择的实际执行节点。

**长期 Router。** 动作只包含执行节点 $J_k$。状态包括当前/预测入口、输出长度分布、节点负载、链路、显存和 $\Phi_k(j)$；不满足 SLA、显存、模型或 KV 可用性的节点由 action mask 排除。暂定使用 prediction-assisted、action-masked Double DQN 近似长期 action value，实际算法需要通过与 Myopic Greedy 和 Oracle 的差距证明长期建模的必要性。

**预测驱动 KV placement。** KV manager 在每个后台周期为候选“节点 $j$--block $b$”计算预期净收益：未来在 $j$ 执行的概率乘以可避免的在线恢复代价，再减去传输、存储、驱逐和误预测成本。可写为：

$$
U_{t,j,b}
=p_{t,j}^{\mathrm{exec}}\Delta C_{t,j,b}^{\mathrm{restore}}
-C_{t,j,b}^{\mathrm{transfer}}
-C_{t,j,b}^{\mathrm{storage}}
-C_{t,j,b}^{\mathrm{waste}}.
$$

其中：

- $t$：后台 placement decision epoch，与按 request 编号的 $k$ 不同；一个 request 间隔内可以运行多个 epoch。
- $b$：KV block 索引；$D_b$ 为该 block 的字节数。
- $p_{t,j}^{\mathrm{exec}}$：在当前预测时域内，未来 request 实际在节点 $j$ 执行并复用该 session KV 的概率，由多步移动预测、session continuation 和 Router 的执行偏好共同估计。
- $\Delta C_{t,j,b}^{\mathrm{restore}}$：若提前把 block $b$ 放到节点 $j$，预计能够避免的在线同步、重算或 SLA violation 成本。
- $C_{t,j,b}^{\mathrm{transfer}}$：后台读取并跨节点传输 block $b$ 的网络与系统成本。
- $C_{t,j,b}^{\mathrm{storage}}$：副本在节点 $j$ 占用显存或其他缓存层产生的保留和潜在驱逐成本。
- $C_{t,j,b}^{\mathrm{waste}}$：预测错误、session 提前结束或副本未被使用时产生的无效复制成本。
- $U_{t,j,b}$：该 block-target placement 的预期净收益；只有正收益候选才进入复制调度。

上述 $\Delta C$ 和各个 $C$ 项必须通过统一权重归一化为同一标量成本单位，不能直接将秒、字节数和显存容量相加。

对进入调度的候选，$z_{t,j,b}\in\{0,1\}$ 表示本 epoch 是否选择该复制任务，$r_{t,j,b}\ge0$ 表示为其分配的后台带宽。它们满足：

$$
\sum_{j,b}r_{t,j,b}\le BW_t^{\mathrm{bg}},
\qquad
\sum_b D_b z_{t,j,b}\le M_{t,j}^{\mathrm{placement}},\quad \forall j,
\qquad
0\le r_{t,j,b}\le z_{t,j,b}BW_t^{\mathrm{bg}}.
$$

其中 $BW_t^{\mathrm{bg}}$ 是 epoch $t$ 可用于 KV 预复制的后台带宽预算，$M_{t,j}^{\mathrm{placement}}$ 是节点 $j$ 可用于新 KV 副本的容量预算。实际复制量由 $r_{t,j,b}$ 与 epoch 时长共同决定；调度器优先选择单位资源收益高、并能扩展最长连续 prefix 的 blocks，而不是任意离散 blocks。

**执行基础。** Block directory、版本检查和副本生命周期管理保证预复制结果可用；源副本在预测确认前保留。真实切换时先复用已准备 prefix，再对 residual tail 做 final incremental sync；若该节点无法满足 SLA，Router 继续选择旧节点或使用 recompute fallback。纯同步、纯重算和可选 hybrid recovery 属于恢复实现，不与端到端策略并列为主要算法。

## 主要贡献

1. **双时间尺度协同框架：** 统一建模移动 session 的前台 request routing 与后台 KV preparation，刻画执行位置、KV 增长位置和节点 readiness 的闭环关系。
2. **长期状态感知 Router：** 联合多步移动预测、输出长度、节点负载和 residual recovery cost，在硬约束下优化 session 长期成本。
3. **预测驱动 KV placement：** 联合决定目标节点、连续 blocks、复制进度和资源预算，并控制误预测造成的无效流量与显存占用。

Block directory、版本检查、增量同步和 recompute fallback 是执行基础；KV-Local Routing 是基线；break-even 是动机分析；复杂 sync/recompute hybrid 只有在 profiling 证明 residual recovery 仍是主瓶颈时才加入。

## 论文结构

1. **Introduction：** 背景与问题、现有工作与缺口、测量证据与动机、技术挑战、方案概述和贡献。
2. **System Model and Problem Formulation：** 系统实体、两个时间尺度、状态、动作、长期目标与 SLA/带宽/显存约束。
3. **Design Overview：** 双时间尺度架构、控制流程以及 Router 与 KV manager 的职责边界和信息接口。
4. **Long-Term Request Routing：** 状态构造、移动/长度预测、action mask、长期价值学习和在线决策。
5. **Predictive KV Cache Placement：** block utility、受约束放置、增量复制、副本保留与回收。
6. **Implementation：** runtime 集成、block directory、版本一致性、后台传输和 residual recovery。
7. **Evaluation：** 端到端比较、Router 与 placement 组件实验、消融、鲁棒性和系统开销。
8. **Conclusion：** 总结适用条件、主要收益和当前局限。

## 实验与 Baseline 定义

原型使用三个同模型推理节点和 25/100 Gbps 链路，覆盖不同模型/context、RTT/拥塞、GPU 负载、显存压力、移动轨迹、切换前可用准备时间、驻留时间和预测误差。为避免把不同层级的方法混为一谈，实验分三组比较：

**Router 对比（固定同一个 KV manager）：**

- **Nearest-Ingress Routing：** 请求始终在当前接入节点执行，不考虑 KV 位置；若缺少 KV，则使用统一的按需恢复机制。
- **KV-Local Routing：** 请求始终发往持有最长可用连续 KV prefix 的节点，优先避免状态恢复，不考虑额外 RTT 和未来负载。
- **Myopic Greedy Routing：** 在可行节点中选择当前预测 E2E latency 最小者，但不考虑本轮执行对未来 KV 位置的影响。
- **Proposed Long-Term Router：** 使用本文的长期 action value 选择执行节点。
- **Trajectory Oracle Router：** 已知未来入口、输出长度和系统轨迹，用作非因果上界，不参与实际部署比较。

**KV placement 对比（统一固定为 Myopic Greedy Router）：**

- **Reactive-Only：** handoff 前不建立新副本，request 到达缺 KV 节点后才执行按需同步或重算。
- **Eager Full Pre-Copy：** 始终将当前完整 KV 尽快复制到预测概率最高的下一节点，不考虑置信度变化和资源机会成本。
- **Probability-Threshold Pre-Copy：** 只有目标节点移动概率超过固定阈值时才开始复制，并使用固定后台带宽。
- **Proposed Predictive Placement：** 使用本文的 block utility 和资源约束，分阶段选择目标、blocks 和复制量。
- **Mobility Oracle Placement：** 已知真实 handoff 节点和时刻，给出预放置上界。

**端到端与模块消融：** 使用一个 $2\times2$ 因子实验，同时控制 Router 和 KV placement 两个变量：

| 组合 | Router | KV placement | 含义 |
|---|---|---|---|
| A | Myopic Greedy | Reactive-Only | 两个 proposed 模块均关闭的共同参照 |
| B | Proposed Long-Term Router | Reactive-Only | 只启用长期 Router，不做提前 KV placement |
| C | Myopic Greedy | Proposed Predictive Placement | 只启用提前 KV placement，Router 仍为短视 Greedy |
| D | Proposed Long-Term Router | Proposed Predictive Placement | 完整方案 |

对应比较关系为：

- **B 与 D 比较：** 在 Router 固定为 Proposed Long-Term Router 时，衡量提前 KV placement 带来的增益。
- **C 与 D 比较：** 在 KV manager 固定为 Proposed Predictive Placement 时，衡量长期 Router 相对 Myopic Greedy 的增益。
- **A 与 B 比较：** 衡量没有提前 placement 时，长期 Router 本身是否仍然有效。
- **A 与 C 比较：** 衡量搭配普通 Greedy Router 时，预测式 placement 本身是否仍然有效。

前两项是以完整方案 D 为中心的直接消融，正好分别回答“去掉提前 placement 会损失多少”和“把长期 Router 换成 Greedy 会损失多少”。后两项用于检查两个模块是否只在彼此存在时才有效。若以越小越好的统一成本 $J$ 为指标，还可以计算交互项：

$$
\Delta_{\mathrm{interaction}}=J_D-J_B-J_C+J_A.
$$

当 $\Delta_{\mathrm{interaction}}<0$ 时，完整方案 D 的成本低于两个模块独立收益的线性叠加，可以认为存在正向协同；若接近 0，则两者收益基本可加；若大于 0，则两个模块可能存在资源竞争或功能重叠。

Pure Sync、Pure Recompute 和 Hybrid Recovery 只比较 residual KV 的恢复时间、传输量和计算量，属于 recovery microbenchmark，不是完整系统 baseline，也不进入上述 $2\times2$ 消融表。

主要指标为平均/P95/P99 E2E latency、SLA violation、handoff blocking time 和 SLA 下并发能力；解释性指标为 KV transfer、residual size、prefix hit rate、wasted replication、显存副本和控制器开销。消融进一步移除多步预测、资源/误预测代价和 action mask，并扫描网络、移动和预测条件。
