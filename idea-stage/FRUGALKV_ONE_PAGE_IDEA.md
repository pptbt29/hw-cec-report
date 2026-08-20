# FrugalKV：One-Page Research Idea

**暂定标题**：*FrugalKV: Resource-Efficient Proactive KV Preparation for SLO-Aware Collaborative Edge LLM Serving*

**一句话 Idea**：在多轮请求到来前，FrugalKV 只准备足以增加 SLO 可行路由选择的 KV，并让多个 session 在统一的带宽、后台 NPU 和 HBM 限额下竞争资源，从而以尽量少的资源取得尽量高的全局 SLO-goodput。

## Abstract

> Collaborative edge sites share scarce accelerators, HBM, and links for multi-turn LLM serving. Growing session KV caches increasingly restrict later requests to their state-holding nodes. When such a node becomes busy, restoring KV elsewhere after arrival may violate the TTFT SLO. Eager multi-node preparation instead wastes resources and can evict useful KV. We study proactive KV preparation and eviction that maximize global SLO-goodput minus bandwidth, NPU, and HBM costs. FrugalKV estimates each completed session's return probability. For each router-approved node, it finds the lowest-cost transfer–recomputation plan that makes the next request SLO-feasible. A global controller selects competing plans and evicts idle KV using the same SLO value. We will implement FrugalKV on eight Ascend 910B NPUs and evaluate multi-turn chatbot traces under controlled prediction errors. We expect higher SLO-goodput with less bandwidth, NPU time, and HBM than reactive recovery and eager preparation.

## 1. Introduction 草稿

### 1.1 背景

Collaborative Edge Computing（CEC）让多个边缘站点共同服务局部突发请求，但每个站点的计算、HBM 和站点间带宽都比较有限。多轮 LLM session 会不断产生并复用 KV cache；session 越长，历史 KV 越大，在没有该 KV 的节点上重新执行历史 prefill 或跨站传输 KV 的时间也越长。FrugalKV 的协作池由运行相同模型版本、tokenizer、位置编码和 adapter 的节点组成，因此节点之间的 KV 可以正确复用；节点的计算速度、剩余 HBM 和网络条件可以不同。

### 1.2 问题：为什么当前路由选择会越来越少？

这个问题由连续的三步形成，而不是三个彼此独立的问题。

1. **KV affinity 逐渐增强。** Cache-aware router 通常优先选择已经保存 session KV 的节点，因为其他节点需要迁移或重算历史 KV。随着历史变长，其他节点的恢复时间也变长，因此能在 TTFT SLO 内接管请求的节点可能越来越少。
2. **请求到达后再处理可能已经太晚。** 下一轮请求到达时，router 可以继续选择持有 KV 的忙节点，也可以选择空闲节点并迁移或重算 KV。但如果原节点排队很长，而其他节点又来不及恢复历史 KV，router 此时已经没有满足 SLO 的选择。问题不一定是 router 选错了节点，而是系统没有提前准备可用节点。
3. **提前准备全部 KV 又太贵。** 把每个 session 的完整 KV 提前复制到多个节点会占用站点间带宽、后台 NPU 和 HBM。部分用户会很晚返回、不再返回，或最终被路由到别的节点，这些准备都会浪费资源，还可能拖慢其他真实请求。

因此，本文研究的问题是：**在多个 session 竞争有限资源时，如何在请求到达前准备和回收 KV，使全局更多请求满足 TTFT SLO，同时避免为很小的 goodput 增益消耗大量资源。** 负载均衡和缓存命中只是中间手段，最终目标是全局 SLO-goodput。

### 1.3 挑战

第一，系统不知道 session 是否返回以及何时返回，准备过早会延长无效驻留时间，准备过晚又无法隐藏恢复时间。第二，一个 session 的准备会占用其他 session 需要的带宽和后台 NPU；当准备任务使目标节点接近 HBM 上限时，还可能迫使系统淘汰其他 session 的 KV，改变它们未来的恢复成本和 SLO。因此，多个 session 不能各自独立地最大化成功概率。第三，准备和淘汰必须使用同一个价值标准：刚准备好的 KV 如果被错误淘汰，准备成本会白费；只保护准备 KV 又可能挤压正在运行的请求。

### 1.4 现有工作的边界

- **KV-aware routing**：[Preble](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html) 和 [DualMap](https://openreview.net/forum?id=zCadrJ32Xn) 根据已有缓存与队列选择节点。DualMap 还能把尚未执行的排队请求改派到备用候选节点；目标节点复用其本地已有前缀，并重算缺失的历史 KV，但不会把源节点已经生成的 KV 随请求一起传过去。它处理已经形成的排队热点，不在未来 chatbot 请求到达前主动准备 session KV。
- **运行时迁移与按需恢复**：[Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao) 迁移正在执行的请求及其 KV；[CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost)、[Cake](https://proceedings.mlr.press/v267/jin25d.html) 和 [HCache](https://2025.eurosys.org/accepted-papers.html) 加速已经到达或已经排队请求的 KV 恢复。这些工作改善“如何恢复”，但不能提前创造新的 SLO 可行节点。
- **请求到达前的 KV 准备**：[SYMPHONY](https://www.usenix.org/conference/nsdi26/presentation/agarwal) 利用用户输入或 Agent 工作流的 advisory，提前选择目标节点并机会式预取 KV，是最直接的基线。它没有把全局 SLO-goodput 收益与带宽、后台计算和 HBM 代价放入同一个显式优化目标。
- **Agent 专用预测管理**：[KVFlow](https://papers.neurips.cc/paper_files/paper/2025/hash/b7971d31a7d5eb0f1eed2f8f6f368195-Abstract-Conference.html) 根据 Agent 执行图做预取和淘汰；[TokenCake](https://arxiv.org/abs/2510.18586)、[Pythia](https://arxiv.org/abs/2604.25899) 和 [PBKV](https://arxiv.org/abs/2605.06472) 也利用工具调用或工作流结构，后三篇目前按 preprint 处理。它们是 Agent 补充实验的对照，但不能直接替代没有显式工作流图的多轮 chatbot 主场景。

### 1.5 方法概览

FrugalKV 按以下步骤运行：

1. **发现近期 session。** 根据每个已完成 session 的轮次间隔历史，估计下一轮请求的返回窗口和返回概率。
2. **读取候选状态。** 从现有 router 获取模型兼容的候选节点，并读取各节点的队列、KV 位置、可用 HBM、链路和后台 NPU 状态。
3. **生成节点方案。** 对每个候选节点计算迁移、重算或两者混合的方案，只保留能够满足 TTFT SLO 且资源代价较低的方案。
4. **协调多个 session。** 用“预计新增的全局 SLO-goodput 减去带宽、后台 NPU 和 HBM 代价”的统一价值比较所有方案，在资源上限内选择当前要执行的方案。
5. **分批执行准备。** KV 运行时按 block batch 执行所选方案，并在每批完成后根据最新返回概率、队列和链路状态决定继续执行还是等待。
6. **按需回收 HBM。** 当节点接近 HBM 高水位时，只检查空闲 session KV，优先回收单位 HBM 对全局 SLO-goodput 影响最小的 block。
7. **发布 KV 就绪状态。** FrugalKV 将每个节点的有效 KV 长度、存储层级和剩余恢复时间提供给 router；请求到达后，router 仍按自己的策略选择最终执行节点。

### 1.6 预期贡献

1. **统一的资源—SLO 建模。** 将全局 SLO-goodput 收益与带宽、后台 NPU 和 HBM 代价直接放入同一个目标函数，在资源硬约束下求解，而不是先后求解两个优化问题；并证明其有限时域离线版本是 NP-hard。
2. **SLO 充分且低资源消耗的准备方法。** 对每个候选节点联合考虑已有 KV、队列、传输和重算，生成足以使节点满足 SLO 且资源代价较低的准备方案；只有当另一方案的预计 SLO 收益不低且资源消耗不高时，才删除被支配方案。无法满足 SLO 的节点不消耗准备资源。
3. **面向全局 goodput 的准备—回收协同。** 先把每个 session 的大动作空间压缩成少量可执行方案，再用全局资源紧张程度协调 session 竞争；HBM 回收使用同一套 SLO 价值，避免准备和淘汰互相抵消。
4. **与路由策略算法解耦的 KV 控制层。** 在 Ascend 910B 上实现独立的 KV 准备与回收模块。该模块不依赖 router 的内部打分规则或历史路由动作，router 也不需要知道 KV 是通过迁移、重算还是回收形成的；双方只通过候选节点、当前队列和 KV 就绪元数据交互，因此同一 KV 策略可以叠加在多种 router 上。

## 2. 问题定义

### 2.1 目标与符号

**系统、请求和时间：**

| 符号 | 含义 |
|---|---|
| $T$ | 统计全局收益和资源消耗的时间窗口长度 |
| $t$ | 当前调度时刻 |
| $\Delta$ | 一次返回概率估计或后台调度所覆盖的短时间段 |
| $\mathcal R_T$ | 在时间窗口 $[0,T]$ 内到达的全部请求集合 |
| $r$ | 一个请求，$r\in\mathcal R_T$ |
| $s$ | 请求所属的多轮 session |
| $j$ | 一个模型兼容的候选节点 |
| $\mathcal J_s$ | router 为 session $s$ 提供的候选节点集合 |
| $D_r$ | 请求 $r$ 的 TTFT SLO |
| $D_s$ | session $s$ 下一轮请求采用的 TTFT SLO |
| $TTFT_r(\pi)$ | 策略 $\pi$ 下请求 $r$ 的实际 TTFT |
| $\mathbf 1\{\cdot\}$ | 条件成立时为 1，否则为 0 的指示函数 |

**返回预测和准备方案：**

| 符号 | 含义 |
|---|---|
| $T_s^{ret}$ | session $s$ 下一轮请求的返回时间 |
| $q_s(t,\Delta)$ | 已知 session 在 $t$ 前尚未返回时，它在 $[t,t+\Delta]$ 内返回的条件概率 |
| $p$ | 一个完整 KV 准备方案 |
| $\mathcal P_{s,j}$ | 为 session $s$ 在节点 $j$ 生成的候选准备方案集合 |
| $\widehat{TTFT}_{s,j}$ | session $s$ 的下一轮请求在节点 $j$ 执行时的预计 TTFT |
| $\Delta G_s(p;t)$ | 在时刻 $t$ 执行方案 $p$ 带来的预计全局 SLO-goodput 增量 |
| $V_s(p;t)$ | 方案 $p$ 的预计 SLO-goodput 增量扣除资源代价后的价值 |
| $B(p)$ | 方案 $p$ 消耗的跨站传输字节数 |
| $U(p)$ | 方案 $p$ 消耗的后台 NPU 时间 |
| $H(p)$ | 方案 $p$ 产生的准备 KV 的 HBM 占用时间 |

**全局资源和目标：**

| 符号 | 含义 |
|---|---|
| $\pi$ | KV 准备、分批执行和回收策略 |
| $\pi^*$ | 在安全策略集合中使全局目标最大的策略 |
| $\Pi_{safe}$ | 满足活动 KV 保护、HBM 硬容量和后台资源上限的策略集合 |
| $G_T(\pi)$ | 时间窗口 $T$ 内的全局 SLO-goodput |
| $B_T(\pi)$ | 时间窗口内的跨站传输总字节数 |
| $U_T(\pi)$ | 时间窗口内的后台重算 NPU 时间 |
| $H_T(\pi)$ | 时间窗口内准备 KV 的 HBM 占用量，单位为 GB·s |
| $B_t,U_t,H_{j,t}$ | 时刻 $t$ 的链路速率、后台 NPU 占用和节点 $j$ 的 KV HBM 占用 |
| $\bar B_t,\bar U_t,\bar H_j$ | 对应资源在时刻 $t$ 的可用上限 |
| $\alpha_B,\alpha_N,\alpha_H$ | 将三类资源代价转换为 SLO-goodput 代价的权重 |
| $C_T(\pi)$ | 时间窗口内经过权重换算的平均资源代价 |
| $J_T(\pi)$ | 同时考虑 SLO-goodput 和资源代价的全局目标值 |

**时延估计、回收和实验参数：**

| 符号 | 含义 |
|---|---|
| $\widehat T_j^{queue}$ | 请求在节点 $j$ 的预计排队时间 |
| $\widehat T_{s,j}^{remain}$ | 请求到达后，session $s$ 在节点 $j$ 仍需承担的历史 KV 恢复时间 |
| $\widehat T_s^{new}$ | session $s$ 下一轮新增 prompt 的预计 prefill 时间 |
| $e$ | 一个可回收的空闲 KV 尾部 block |
| $L(e)$ | 回收 block $e$ 造成的预计 SLO-goodput 损失 |
| $M(e)$ | 将 block $e$ 下移到较慢存储的代价；直接删除时为 0 |
| $S(e)$ | 回收 block $e$ 释放的 HBM 容量 |
| $E(e)$ | 回收 block $e$ 的单位 HBM 综合损失 |
| $B_0$ | 实验平台测得的原生站点间带宽 |
| $RTT$ | 站点间往返时延 |
| $\lambda_0$ | router-only 系统的饱和请求率 |
| $CV$ | 请求到达间隔的变异系数 |
| $l$ | 请求输入长度所在区间 |
| $T_{99}^{hit}(l)$ | 长度区间 $l$ 中，KV 已在本地 HBM 时的空载 P99 TTFT |

**复杂度证明使用的符号：**$i$ 表示受限问题中的一个 session，$w_i$ 是准备该 session 的资源消耗，$v_i$ 是其预期 SLO-goodput 收益，$\mathcal B$ 是总资源预算。

在时长为 $T$ 的统计窗口内，SLO-goodput 定义为：

$$
G_T(\pi)
=
\frac{1}{T}
\sum_{r\in\mathcal R_T}
\mathbf 1\{TTFT_r(\pi)\le D_r\}.
$$

尚未在统计截止时间前产生首 token 的请求按 SLO 失败计入，避免策略通过无限排队提高指标。三类资源直接进入同一个平均代价：

$$
C_T(\pi)
=
\frac{1}{T}
\left[
\alpha_B B_T(\pi)
+
\alpha_N U_T(\pi)
+
\alpha_H H_T(\pi)
\right].
$$

FrugalKV 使用一个统一目标：

$$
J_T(\pi)=G_T(\pi)-C_T(\pi),
$$

$$
\pi^*
\in
\arg\max_{\pi\in\Pi_{safe}}J_T(\pi).
$$

$\Pi_{safe}$ 同时要求 $B_t\le\bar B_t$、$U_t\le\bar U_t$ 和 $H_{j,t}\le\bar H_j$。$\alpha_B$、$\alpha_N$ 和 $\alpha_H$ 表示系统愿意用多少 SLO-goodput 换取对应资源；它们由部署环境确定，带宽紧张、后台计算紧张和 HBM 紧张的配置分别提高对应权重。$J_T$ 用于控制器选取动作和计算小规模最优差距；端到端实验分别报告 $G_T$、$B_T$、$U_T$ 和 $H_T$，避免权重掩盖某一类资源的真实变化。

### 2.2 为什么是 NP-hard？

只考虑一个节点、一类资源、没有淘汰，并让 session $i$ 只有“花费 $w_i$ 完成准备，获得 $v_i$ 预期 SLO-goodput”或“等待”两个动作。令该特例中的资源权重为 0，在总预算 $\mathcal B$ 下选择 session，使总收益最大，正好是 0–1 knapsack。该限制问题已经是 NP-hard，所以包含多节点、迁移与重算、部分 block、资源代价和淘汰的完整离线问题也是 NP-hard。

这里能写的是“**有限时域离线问题是 NP-hard**”，不能把一般的在线随机控制问题直接写成 NP-complete。完整归约见 [FRUGALKV_NP_HARDNESS_DERIVATION.md](/Users/dazzysy/.codex/worktrees/d093/hw-cec-report/idea-stage/FRUGALKV_NP_HARDNESS_DERIVATION.md)。

## 3. FrugalKV 设计

### 3.1 何时触发？

准备和回收使用两个简单、不同的触发条件。

- **准备触发**：一个已完成 session 进入下一轮请求的预计返回窗口。请求完成时只记录 KV 和历史轮次间隔，不立即迁移；周期采样也不会扫描并重排所有 KV。
- **回收触发**：某节点的“活动请求 KV、近期增长预留、空闲 KV 和已接纳准备任务”之和预计超过 HBM 高水位。系统回收到低水位后停止。

系统在 session 尚未返回的条件下持续更新近期返回概率：

$$
q_s(t,\Delta)
=
\Pr\left(T_s^{ret}\in[t,t+\Delta]\mid T_s^{ret}>t\right).
$$

返回概率越高，完成该 session 准备方案的预期 SLO 收益越高；准备多少 KV 仍由节点跨过 SLO 门槛所需的最小状态决定。外部 advisory 可以缩短返回窗口；历史信息不足时，系统使用保守的低返回概率，使方案自然排在其他候选之后。

### 3.2 每个候选节点先生成 SLO 充分且低资源消耗的方案

对 session $s$ 和候选节点 $j$，系统估计：

$$
\widehat{TTFT}_{s,j}
=
\widehat T^{queue}_j
+
\widehat T^{remain}_{s,j}
+
\widehat T^{new}_{s},
$$

其中三项分别是节点排队时间、请求到达后仍需完成的历史 KV 恢复时间，以及本轮新增 prompt 的 prefill 时间。系统枚举迁移、重算和两者混合的分界点，形成方案集合 $\mathcal P_{s,j}$，并保留使 $\widehat{TTFT}_{s,j}\le D_s$ 的方案。若方案甲的预计 SLO-goodput 增量不低于方案乙，而且 $B(p)$、$U(p)$ 和 $H(p)$ 都不高于方案乙，系统才删除方案乙。剩余方案再交给全局控制器根据当前资源代价进行选择。

这里的候选是一个**足以跨过 SLO 门槛的完整方案**，不是单独的“迁移 10 个 block”动作。例如 A 需要准备 40 个 block、B 需要 20 个 block，那么它们是两个完整候选方案；如果 A 只迁 10 个 block 仍然不能在剩余时间内完成其余准备，A 不会被误算成 SLO 可行。方案执行时可以分批传输，但完成前不计为已经创造了一个可行节点。

### 3.3 先压缩单 session 的选择，再协调所有 session

对每个被唤醒的 session，局部求解器只保留少量候选：等待、准备一个节点，或最多准备两个节点。节点数上限不是 router 只能看两个节点；router 可以先提供所有模型兼容候选，局部求解器再删除无法满足 SLO 或在资源和时延上都明显更差的方案。

多个 session 随后一起竞争资源。对方案 $p$，控制器估计它在返回概率 $q_s(t,\Delta)$ 下带来的预计全局 SLO-goodput 增量 $\Delta G_s(p;t)$，并计算：

$$
V_s(p;t)
=
\Delta G_s(p;t)
-\alpha_B B(p)
-\alpha_N U(p)
-\alpha_H H(p).
$$

带宽紧张的部署提高 $\alpha_B$，后台计算紧张的部署提高 $\alpha_N$，HBM 紧张的部署提高 $\alpha_H$；当期剩余容量仍作为不能突破的硬约束。每个 session 只比较自己的少量方案，但所有 session 使用同一组部署权重和全局资源约束，因此一个 session 占用 HBM 并可能引发其他 session KV 回收的影响也会进入当前决策。

控制器选择目标值 $J_T$ 增量为正且满足硬资源上限的方案。该分解避免枚举“所有 session × 所有节点 × 所有 block”的联合动作；它是在线近似，小规模实验将与离线最优解比较其目标值差距。

### 3.4 分批执行与重新判断

控制器选中的是完整准备方案，KV 运行时再按固定大小的 block batch 执行。多项方案同时等待时，优先执行当前 $V_s(p;t)$ 较高且最接近完成的方案。每批完成后，系统更新返回概率、剩余时间、链路和节点队列：若完整方案仍有正价值并能按时完成，就继续；若已经来不及或价值转负，则停止继续投入。这样，返回概率较高的 session 通常更快获得后续 block，但一个不足以满足 SLO 的小批次不会被误当成最终收益。

### 3.5 HBM 回收

HBM 压力出现时，只检查该节点上的**空闲 session KV**。每个可回收尾部 block 计算：

$$
E(e)
=
\frac{L(e)+M(e)}{S(e)}.
$$

$E(e)$ 越小，表示每释放一单位 HBM 造成的损失越小，因此越先回收。分数在高水位触发时计算；session 返回概率、KV 位置或队列状态明显变化时，只更新受影响的条目。回收一个 block 时，有容量的下层存储接收该 block；保留价值极低时可以删除 KV，但仍保留原始 token 历史，以便未来重算。正在 prefill 或 decode 的活动 KV 永远不进入后台回收队列。

### 3.6 如何与现有 router 组合？

FrugalKV 与 router 是**算法解耦、状态互通**的关系。FrugalKV 需要 router 提供模型兼容候选节点及当前队列状态，但不依赖 router 的内部打分公式或最近的路由动作序列；它只决定这些节点应提前具备多少 KV，以及在 HBM 压力下回收哪些空闲 KV。FrugalKV 向 router 发布节点已有的有效 KV 长度、KV 所处存储层级和请求到达后仍需承担的恢复时间。router 根据当前请求、队列和这些 KV 元数据选择最终节点，不需要知道 KV 是通过迁移、重算还是回收形成的。

实验因此对每种 router 成对比较“原 router”和“原 router + FrugalKV”。

只要 router 能读取 KV 就绪程度或预计恢复时间，FrugalKV 就能作为独立 KV 插件与该路由策略组合；完全忽略 KV 状态的黑盒 router 不在本文接口范围内。

## 4. 正式实验设计

实验围绕五项方法主张展开：全局 SLO-goodput、达到相同 goodput 所需的资源、准备—回收协同、预测误差下的稳定性，以及对不同 router 的增益。

### 4.1 Testbed 与模型

- **硬件**：8 张 Ascend 910B NPU。主配置使用 8 个单卡模型副本；若物理机器布局允许，再测试 4 个双卡副本。
- **物理前提**：CEC 的跨站结论至少需要两台物理服务器。若 8 张卡全部位于同一台服务器，只能完成单机控制面和 HBM 实验，不能仅凭流量整形声称已经验证跨站 CEC。
- **运行时**：优先使用 vLLM Ascend，并在其 block manager 外增加 FrugalKV 控制层；传输采用实机支持的 TCP/RDMA/HCCL 路径，最终选择必须根据网卡和拓扑微基准确定。
- **模型**：Qwen2.5-7B-Instruct 为主模型，Llama-3.1-8B-Instruct 为跨模型复现。两者都属于 7B–8B 单卡可部署模型。上下文长度测试 4K、16K 和 32K。
- **网络**：先测量实际可用带宽 $B_0$ 和 RTT。主实验使用原生网络，并通过真实链路限速测试 $0.25B_0$、$0.5B_0$ 和 $B_0$；RTT 测试原生、10 ms 和 30 ms。网卡规格确认后再把绝对 Gbps 写入论文。

[vLLM Ascend 官方文档](https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen2.5-7B.html)已经给出 Ascend 910B 和 Qwen2.5-7B 的支持路径；Qwen2.5-7B BF16 可单卡部署。实际软件版本必须在实验前固定并报告。

### 4.2 Workload 与没有完整数据时的生成方法

主 workload 是**多轮 chatbot**，Agent 只做补充。

1. **真实会话内容与长度**：使用 ShareGPT 或 WildChat 的真实多轮消息，保留每轮 prompt 增长和输出长度。
2. **全局到达强度**：回放 BurstGPT 的真实到达时间，并将其与 ShareGPT 或 WildChat 的多轮 session 内容组合。
3. **轮次间隔**：若拿不到同时包含 session ID、逐轮时间戳和 token 长度的公开 trace，则将真实会话内容与轮次间隔模型组合后在真实系统上回放。默认使用 log-normal 分布，测试中位数 $\{5,15,60\}$ 秒和几何标准差 $\{1.5,3\}$；同时测试 $\{0,10,30,50\}\%$ 的“预测返回但实际未返回”。
4. **突发程度**：若不用 BurstGPT 原始到达，则用 Gamma 到达间隔并测试变异系数 $CV\in\{0.5,1,2\}$。$CV=1$ 对应 Poisson，$CV>1$ 表示更强突发。
5. **Agent 补充**：使用 MetaGPT 或公开 Agent workflow trace，报告显式工作流信息能够带来的额外增益。

所有参数化数据标记为“合成时间轴上的真实系统回放”。主结果同时给出真实到达回放和参数敏感性。

### 4.3 SLO、负载和资源超参

- **TTFT SLO**：先测每个模型和输入长度区间在“空载、KV 已在本地 HBM”时的 P99 TTFT，记为 $T^{hit}_{99}(l)$。主 SLO 设为 $2T^{hit}_{99}(l)$，并测试严格、中等、宽松三档 $\{1.5,2,3\}T^{hit}_{99}(l)$；正文同时报告对应的绝对毫秒数。
- **Offered load**：先找到 router-only 系统的饱和请求率 $\lambda_0$，测试 $\{0.6,0.8,1.0,1.2\}\lambda_0$。
- **KV HBM 预算**：扣除模型权重和运行时预留后，将剩余 HBM 的 $\{40,60,80\}\%$ 用作可管理 KV 空间。
- **资源权重**：先将带宽、后台 NPU 和 HBM 消耗分别除以各自预算，再设置 $\alpha_B$、$\alpha_N$ 和 $\alpha_H$。主配置对三种归一化资源使用相同基础权重，并分别提高其中一种资源的权重，构造带宽紧张、后台计算紧张和 HBM 紧张三类 CEC 环境。
- **准备副本上限**：主实验最多准备 1 个额外节点，并与最多 2 个额外节点比较；敏感性实验继续增加候选节点数。

### 4.4 Baselines

主实验只保留三类能够回答核心问题的 baseline。所有方法使用相同的 router、KV block 大小、返回时间预测和迁移—重算数据面。

1. **On-demand**：请求到达后，router 才在原节点排队、跨站迁移或目标节点重算之间选择，用于衡量主动准备的整体价值。
2. **Eager-full**：session 进入相同返回窗口后，在预测目标节点准备完整 KV，用于衡量“SLO 充分且低资源消耗的方案”相对完整准备节省多少资源。
3. **SYMPHONY-style**：使用相同返回信号，按当前负载选择一个目标节点，并采用机会式预取和原论文的层优先级回收，用于比较最接近的请求到达前 KV 准备机制。论文将明确标注这是对 SYMPHONY 控制策略的重实现。

恢复数据面在所有方法之间固定；Cake、HCache 或 CacheFlow 类恢复路径作为共同底层能力。主实验使用 minimum-predicted-TTFT router；插件实验再使用 DualMap-style two-choice router，对两种 router 分别比较“原版”与“原版 + FrugalKV”。前者能在全体候选节点中按预计完成时间选择，后者把每个前缀限制在两个候选节点，二者足以检验 FrugalKV 能否适配两种不同的路由空间。小规模离线最优解用于报告在线算法的最优差距。

### 4.5 Metrics 与比较维度

正文使用三组与论文目标直接对应的指标：

1. **SLO-goodput**：每秒产生首 token 且 TTFT 满足 SLO 的请求数，是首要效果指标。
2. **达到相同 SLO-goodput 时的资源消耗**：分别报告跨站传输 GB、后台重算 NPU·s 和准备 KV 的 HBM GB·s，不把三项只合成一个分数。
3. **P99 TTFT**：确认 goodput 提升没有掩盖严重的尾部时延退化。

预测鲁棒性实验同时报告未被请求使用的准备字节比例，用来解释错误预测造成的资源浪费；控制面实验报告单次决策时延，用来验证在线运行能力。

主结果覆盖 offered load 和资源紧张程度，分别给出“固定资源预算比较 SLO-goodput”和“固定 SLO-goodput 比较三类资源消耗”。随后测试返回预测误差和 router 类型。RTT、上下文长度、session 间隔、block batch 大小和额外准备节点数放入敏感性实验。

### 4.6 Ablation

正文安排两个消融实验：

1. **方案生成消融**：在同一个全局控制器下，用完整 KV 准备替换“SLO 充分且低资源消耗的方案”，比较相同 SLO-goodput 下的带宽、后台 NPU 和 HBM。
2. **协调—回收二维消融**：构造“全局协调 / 单 session 贪心”与“SLO 价值回收 / LRU”组成的 $2\times2$ 四种配置，同时观察 SLO-goodput、资源消耗以及准备完成后又被淘汰的 KV 比例。该实验分别揭示全局资源竞争和准备—回收协同的作用。

### 4.7 预测器与其他必要实验

返回时间预测采用按时间顺序划分的训练、验证和测试数据。线上使用分桶经验分布或生存概率估计 $q_s(t,\Delta)$，并报告返回窗口覆盖率、可用准备提前量、过早准备时间和 false-return 比例。

鲁棒性实验分别注入早到、晚到、未返回和提示缺失，测量 SLO-goodput、三类资源消耗和未使用准备字节的变化；perfect prediction 结果作为预测误差为零时的性能上界。小规模请求集合使用完整未来信息求解离线最优方案，并报告在线控制器的目标值差距。控制面实验测量不同 session 数量下的决策时延；敏感性实验覆盖 block batch 大小和 HBM 高低水位。每个配置使用同一到达序列完成至少 3 次重复实验并报告置信区间。

### 4.8 Claim–evidence 对照

| 论文主张 | 核心证据 |
|---|---|
| 资源受限时提高全局 SLO 服务能力 | 不同负载和资源预算下的 SLO-goodput 与 P99 TTFT |
| 相同 SLO-goodput 下消耗更少资源 | 跨站传输 GB、后台 NPU·s 和 HBM GB·s |
| SLO 充分且低资源方案优于完整准备 | “完整准备 / 本文方案”消融及对应三类资源量 |
| 全局协调与价值回收需要协同 | “全局 / 单 session”与“本文回收 / LRU”的 $2\times2$ 消融 |
| 能与不同 router 组合 | minimum-predicted-TTFT 与 DualMap-style router 的 before/after 结果 |
| 在线算法的近似质量可控 | 小规模离线最优解与在线结果的 $J_T$ 差距 |

## 5. 论文结构与图表

| 章节 | 核心内容 | 建议图表 |
|---|---|---:|
| 1. Introduction | 背景、三步问题链、挑战、相关工作、方法与贡献 | 1 张 teaser |
| 2. Background | CEC 资源、KV 生命周期、router 与恢复数据面 | 1 张系统时间线 |
| 3. Problem Formulation | 统一资源—SLO 目标、约束和 NP-hard 证明 | 1 张符号表 |
| 4. Design | 触发、候选生成、全局协调、分批执行和回收 | 2 张算法图 |
| 5. Implementation | Ascend 910B 原型、router 接口和 KV 数据面 | 1 张架构图 |
| 6. Evaluation | 主结果、资源曲线、消融、鲁棒性、插件矩阵和 oracle gap | 6–8 张图表 |
| 7. Related Work & Limitations | 与主动准备、路由、恢复数据面和 Agent 预测管理的边界 | 无主图 |

## 6. 当前最重要的现实约束

1. 必须确认 8 张 Ascend 910B 分布在几台物理服务器以及网卡带宽；否则跨站 CEC 结论的实验条件不成立。
2. 必须用同一预测器和同一恢复数据面比较 FrugalKV、SYMPHONY-style 和完整预取，才能把收益归因于控制策略。
3. FrugalKV 需要相对使用相同预测信号和恢复数据面的 SYMPHONY-style 基线，在相近 goodput 下稳定降低至少一种关键资源，同时避免其他资源出现抵消性增长；这是支撑方法价值的最低证据。
4. Agent 预取论文数量增长很快；投稿前必须重新检查 SYMPHONY、KVFlow、Pythia、PBKV、TokenCake 以及直接面向 CEC KV 迁移的新工作。
