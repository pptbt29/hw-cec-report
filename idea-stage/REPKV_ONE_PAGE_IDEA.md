# RepKV：One-Page Research Idea

**暂定标题**：*RepKV: Resource-Efficient Proactive KV Management for SLO-Aware Collaborative Edge LLM Serving*

**名称含义**：RepKV = **Resource-Efficient Proactive KV** Management。

**一句话 Idea**：RepKV 在下一轮请求到达前，为少量备用节点准备必要的 KV，并在内存不足时回收最不影响请求按时完成的空闲 KV。目标是在有限带宽、算力和内存下，让尽可能多的请求在规定时间内返回首 token（满足 TTFT SLO）。

## Abstract

> Edge sites jointly serve multi-turn LLM requests with limited compute, memory, and network capacity. Later requests reuse KV from earlier turns, so they usually return to the node holding it. As KV grows, moving a request only after that node becomes busy may be too late because rebuilding its history elsewhere can exceed the TTFT limit. Preparing KV earlier can help, but unnecessary work wastes resources and may remove other useful KV from memory. We study how to maximize requests meeting TTFT limits with minimal resource use. RepKV estimates when a session will return and the cost of its next prompt. It calculates the least transfer or recomputation needed to prepare one useful alternative node. RepKV then chooses among competing sessions and removes the least useful idle KV when memory is full. We will evaluate RepKV on eight Ascend 910B NPUs with chatbot traces, several routers, and prediction errors.

## 1. Introduction 草稿

### 1.1 背景

Collaborative Edge Computing（CEC）让多个相邻边缘站点共同处理请求。相比大型数据中心，边缘站点的算力、加速卡内存和站点间带宽更有限，突发请求也更容易让单个站点排起长队。

多轮 LLM 请求还会保存此前对话的中间计算结果，即 KV cache。下一轮请求复用这些 KV，就不必重新计算全部历史；如果换到没有这些 KV 的节点，系统就必须传输 KV，或者重新计算历史内容。对话越长，需要恢复的 KV 越多，耗时也越长。

因此，路由器通常会把下一轮请求继续发给保存其 KV 的节点。这在节点空闲时很有效，但也使一个长 session 越来越依赖该节点。一旦这个节点变忙，再把请求转移到其他节点可能已经来不及。

### 1.2 问题

本文关注三个连续的问题：

1. **请求到达后再换节点可能太晚。** 当保存 KV 的节点排队严重时，其他节点即使空闲，也可能因为传输或重算历史 KV 耗时过长，无法在 TTFT SLO 内返回首 token。此时路由器并不是“选错了节点”，而是已经没有能够按时处理请求的节点。
2. **提前准备多少并不确定。** 系统不知道一个 session 是否还会继续、下一轮何时到达，也不知道新 prompt 的长度。准备过早或过多会浪费带宽、算力和内存；准备过晚或过少，又不能帮助请求按时完成。
3. **不同 session 会争抢同一批资源。** 为一个 session 准备 KV，可能占满目标节点的内存，迫使系统删除另一个 session 的 KV。只考虑单个 session 的收益，可能反而降低整个系统按时完成的请求数。

因此，RepKV 要解决的是：**在有限资源下，提前为哪些 session 准备多少 KV，以及内存不足时回收哪些 KV，才能让尽可能多的请求满足 TTFT SLO。** 本文把单位时间内满足 TTFT SLO 的请求数称为 SLO-goodput。

### 1.3 挑战

- **何时准备：** 既要预测下一轮请求是否会来、何时来，也要估计新 prompt 会带来多少计算。
- **准备什么：** 系统需要在传输 KV、重新计算历史内容和暂不准备之间选择，并确定实际需要准备多少。
- **如何协调：** 多个 session 同时竞争资源时，准备和回收必须服务于同一个全局目标，不能各自做局部决定。

### 1.4 现有工作的边界

| 方向与代表工作 | 已经解决什么 | 仍未覆盖本文问题的部分 |
|---|---|---|
| **KV 感知路由**：[Preble](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html)、[DualMap](https://openreview.net/forum?id=zCadrJ32Xn) | 请求到达后，根据节点已有的 KV 和当前负载选择执行节点 | 不会在下一轮请求到达前补齐其他节点缺少的 KV |
| **请求到达后的迁移与恢复**：[Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao)、[CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost)、[Cake](https://proceedings.mlr.press/v267/jin25d.html)、[HCache](https://2025.eurosys.org/accepted-papers.html) | 迁移正在排队或执行的请求，或者加快 KV 的读取和恢复 | 开始处理问题时，请求已经到达，长 KV 的恢复仍可能错过 SLO |
| **请求到达前的 KV 准备**：[SYMPHONY](https://www.usenix.org/conference/nsdi26/presentation/agarwal) | 根据用户输入或 Agent 调用等提前信号，把 KV 预取到预选节点 | 没有回答多个 session 竞争资源时，应为谁准备、准备多少，以及怎样以较少资源提高全局 SLO-goodput |
| **根据工作流预测 KV 需求**：[KVFlow](https://papers.neurips.cc/paper_files/paper/2025/hash/b7971d31a7d5eb0f1eed2f8f6f368195-Abstract-Conference.html)、[PBKV](https://arxiv.org/abs/2605.06472)（preprint）、[Pythia](https://arxiv.org/abs/2604.25899)（preprint） | 根据 Agent 调用图或调用概率预取和淘汰 KV | 主要面向已知工作流程的 Agent；本文关注没有调用图的多轮对话和边缘站点间的资源竞争 |

现有工作已经分别研究了路由、KV 恢复、提前预取和 Agent 工作流预测。RepKV 关注它们尚未共同回答的问题：**当多个多轮对话共享有限资源时，怎样用尽量少的准备工作，提高整个系统按时完成的请求数。**

### 1.5 方法概览

RepKV 按以下顺序工作：

1. **预测下一轮请求。** 当前请求结束后，系统根据历史轮次间隔估计这个 session 是否会继续、可能何时继续，以及下一轮 prompt 的计算量。只有当下一轮可能即将到来时，系统才考虑提前准备。
2. **计算必要的准备量。** 系统逐个检查路由器允许使用的节点，估算该节点还缺多少 KV，才能在规定时间内返回首 token。恢复方式可以是传输 KV、重新计算历史内容，或两者结合。系统不会把所有节点都准备好，而是保留少量真正有用的选择。
3. **在所有 session 之间分配资源。** 系统比较各项准备能够增加多少按时完成的请求，以及要消耗多少带宽、算力和内存，再决定先执行哪些准备。当内存不足时，系统优先回收对全局结果影响最小的空闲 KV。

RepKV 不负责最终路由。它只告诉现有路由器“每个节点已经有多少 KV、还需要多少恢复时间”；请求到达后，路由器仍按自己的规则选择节点。因此，同一套 RepKV 可以与不同路由方法组合。

### 1.6 预期贡献

1. **提出资源受限的主动 KV 管理问题。** 目标不是复制更多 KV，而是在有限资源下提高整个系统的 SLO-goodput；准备和回收由同一个目标决定。
2. **设计按需准备方法。** RepKV 根据下一轮请求的预计到达时间和计算量，只传输或重算能够产生实际 SLO 收益的 KV。
3. **设计跨 session 的全局控制方法。** 当多个 session 竞争资源时，系统优先执行单位资源带来更多 SLO-goodput 的准备，并回收影响最小的空闲 KV。
4. **实现可与现有路由器组合的原型。** 在 Ascend 910B 集群上验证 RepKV 能否同时提高 SLO-goodput 并降低带宽、后台计算和内存消耗。

---

> 以下内容是问题建模、算法和实验的技术备忘，不属于前面的 one-page summary，也不应直接塞进 Introduction。

## 2. 问题定义

### 2.1 系统范围

系统包含多个可以协同处理同一类 LLM 请求的 CEC 节点。每个节点拥有独立的请求队列、加速卡内存和可选的 CPU 内存或本地存储；节点之间可以传输 KV。RepKV 只处理已经完成当前轮、正在等待下一轮请求的 session KV。正在执行的请求所使用的 KV 不会被后台回收。

一个 session 的下一轮请求到达后，路由器决定其最终执行节点。RepKV 不改变路由器的目标，只在请求到达前改变各节点已经保存的 KV，从而让路由器拥有更多能够按时处理请求的选择。

### 2.2 符号

| 符号 | 含义 |
|---|---|
| $T$ | 统计全局收益与资源消耗的有限时间窗口 |
| $t$ | 当前时刻 |
| $\Delta$ | 一次预测和后台调度覆盖的短时间段 |
| $s$ | 一个已经完成当前轮、等待下一轮请求的 session |
| $n$ | 路由器允许使用的一个节点 |
| $r$ | 时间窗口 $T$ 内到达的一个真实请求 |
| $\mathcal R_T$ | 时间窗口 $T$ 内到达的全部真实请求 |
| $D_r$ | 请求 $r$ 的 TTFT SLO |
| $D_s$ | session $s$ 的下一轮请求采用的 TTFT SLO |
| $TTFT_r(\pi)$ | 策略 $\pi$ 下请求 $r$ 的实际 TTFT |
| $\mathbf 1\{\cdot\}$ | 条件成立时为 1、否则为 0 的指示函数 |
| $X_s$ | session $s$ 从当前请求完成到下一轮请求到达的间隔；若不再返回则视为右删失样本 |
| $\tau_s$ | 当前时刻距离 session $s$ 上一轮完成已经过去的时间 |
| $c$ | 按应用类型、轮数和上下文长度划分的 session 类别 |
| $F_c(x)$ | 类别 $c$ 的下一轮请求间隔分布 |
| $q_s(t,\Delta)$ | 已知 session $s$ 到时刻 $t$ 尚未返回时，它在 $[t,t+\Delta]$ 内返回的条件概率 |
| $\mathcal N_s$ | 路由器允许 session $s$ 使用的节点集合 |
| $p$ | 为 session $s$ 生成的一个完整准备方案，可包含等待或为至多两个节点准备 KV |
| $\mathcal P_s$ | session $s$ 的完整候选准备方案集合 |
| $x_{s,p,t}$ | 时刻 $t$ 是否选择方案 $p$ 的二元变量 |
| $\widehat T^{queue}_n$ | 新请求在节点 $n$ 上的预计排队时间 |
| $\widehat T^{recover}_{s,n}(p)$ | 执行方案 $p$ 后，请求到达节点 $n$ 仍需承担的历史 KV 恢复时间 |
| $\widehat T^{new}_s$ | 下一轮新增 prompt 的预计 prefill 时间 |
| $\widehat{TTFT}_{s,n}(p)$ | 方案 $p$ 下，下一轮请求在节点 $n$ 的预计 TTFT |
| $\Delta G_{s,p,t}$ | 时刻 $t$ 选择方案 $p$ 预计增加的全局 SLO-goodput |
| $B_{s,p}$ | 方案 $p$ 需要的跨站传输字节数 |
| $U_{s,p}$ | 方案 $p$ 需要的后台 NPU 时间 |
| $M_{s,p,n}$ | 方案 $p$ 在节点 $n$ 新增的瞬时 HBM 占用 |
| $H_{s,p}$ | 方案 $p$ 产生的准备 KV 驻留量，按 GB·s 统计 |
| $B_T(\pi),U_T(\pi),H_T(\pi)$ | 策略 $\pi$ 在窗口 $T$ 内产生的三类资源总消耗 |
| $\bar B_t,\bar U_t,\bar H_n$ | 时刻 $t$ 可用于后台准备的带宽、NPU 和节点 $n$ 的 KV HBM 上限 |
| $B^{ref},U^{ref},H^{ref}$ | 用于归一化三类资源消耗的参考预算 |
| $\lambda_B,\lambda_U,\lambda_H$ | 把归一化资源消耗换算为 SLO-goodput 代价的权重 |
| $\pi$ | 一个 KV 准备、分批执行和回收策略 |
| $\Pi_{safe}$ | 不突破资源硬上限且不回收活动 KV 的安全策略集合 |
| $G_T(\pi)$ | 策略 $\pi$ 在窗口 $T$ 内实现的全局 SLO-goodput |
| $J_T(\pi)$ | 同时考虑 SLO-goodput 和资源代价的全局目标 |
| $e$ | 节点上一个可回收的空闲 KV block batch |
| $\Delta G^{evict}_{e}$ | 回收 $e$ 预计造成的全局 SLO-goodput 损失 |
| $C^{evict}_{e}$ | 下移或删除 $e$ 的资源代价，已换算为 SLO-goodput 单位 |
| $S_e$ | 回收 $e$ 释放的 HBM 容量 |
| $\mathcal B$ | NP-hard 特例中的总资源预算 |

### 2.3 返回概率需要怎样的预测模型？

RepKV 使用历史轮次间隔训练一个轻量的生存概率模型。令 $F_c(x)$ 表示某类 session 的下一轮间隔分布，其中类别 $c$ 可以由应用类型、当前轮数和上下文长度分桶得到。对于已经等待 $\tau_s$、但尚未返回的 session：

$$
q_s(t,\Delta)
=
\Pr(X_s\le \tau_s+\Delta\mid X_s>\tau_s)
=
\frac{F_c(\tau_s+\Delta)-F_c(\tau_s)}{1-F_c(\tau_s)}.
$$

该模型回答的是“已经等了这么久以后，下一小段时间内返回的可能性还有多大”，而不是给出一个必然准确的到达时刻。系统按时间顺序更新经验分布；外部 advisory 可以提高相应时间段的返回概率，但不是 RepKV 运行的必要条件。

### 2.4 全局目标

全局 SLO-goodput 定义为单位时间内按时产生首 token 的请求数：

$$
G_T(\pi)
=
\frac{1}{T}
\sum_{r\in\mathcal R_T}
\mathbf 1\{TTFT_r(\pi)\le D_r\}.
$$

为了避免系统总是消耗尽可能多的资源来换取很小的 SLO 提升，RepKV 把三类资源归一化后直接加入同一个目标：

$$
J_T(\pi)
=
G_T(\pi)
-\lambda_B\frac{B_T(\pi)}{B^{ref}}
-\lambda_U\frac{U_T(\pi)}{U^{ref}}
-\lambda_H\frac{H_T(\pi)}{H^{ref}}.
$$

其中 $B^{ref}$、$U^{ref}$ 和 $H^{ref}$ 是部署环境给定的参考预算。系统求解：

$$
\pi^*\in\arg\max_{\pi\in\Pi_{safe}}J_T(\pi),
$$

其中安全策略集合 $\Pi_{safe}$ 要求后台准备不能突破实时带宽与后台 NPU 配额，节点 KV 占用不能超过 HBM 硬容量，而且活动请求 KV 不可被后台回收。论文会分别报告 SLO-goodput 和三类资源量，而不是只报告经过权重合并的 $J_T$。

### 2.5 为什么该问题是 NP-hard？

考虑一个受限特例：只有一个候选节点和一种共享资源；每个 session 只有“消耗 $w_s$ 完成准备并获得期望 SLO 收益 $v_s$”或“等待”两个选择；没有回收、路由变化和时间耦合。在总资源预算 $\mathcal B$ 下选择一组 session，使总收益最大，正好是 0–1 knapsack。该特例已经是 NP-hard，因此包含多节点、迁移、重算、部分 block、三类资源和回收动作的有限时域离线问题也是 NP-hard。

## 3. 具体方法

RepKV 按照“记录状态 → 预测返回 → 生成完整方案 → 全局选择 → 分批执行 → 必要时回收 → 发布状态”的顺序运行。

### 3.1 请求完成后只记录状态

一个 session 完成当前请求后，RepKV 记录：最新 KV 长度、KV 在各节点和存储层级中的位置、当前轮次、上下文长度以及本轮与历史轮次的时间间隔。此时系统不立即迁移 KV，因为下一轮请求可能很久以后才到达，也可能不再到达。

### 3.2 进入预测返回窗口时触发准备

预测器持续计算 $q_s(t,\Delta)$。当时间进入预测出的下一轮请求到达窗口，而且候选方案从现在开始执行才可能在窗口内完成时，session $s$ 被加入全局准备队列。换句话说：**当前请求完成只产生预测；真正的准备由“下一轮请求可能即将到达”触发。**

后台调度器在有 session 进入准备队列、一个 block batch 执行完成或后台资源重新空闲时运行。它只重新考虑当前准备队列中的 session，不周期性重排系统中的全部 KV。

### 3.3 计算每个备用节点需要准备多少 KV

RepKV 从路由器获取可用节点集合 $\mathcal N_s$，并读取这些节点当前的队列和 KV 状态。对于节点 $n$，预计 TTFT 为：

$$
\widehat{TTFT}_{s,n}(p)
=
\widehat T^{queue}_n
+\widehat T^{recover}_{s,n}(p)
+\widehat T^{new}_s.
$$

节点级求解器从当前已有的连续有效 KV 前缀出发，按 block batch 枚举三种恢复动作：

- 从已有副本向节点 $n$ 迁移下一批 KV；
- 在节点 $n$ 的后台 NPU 上精确重算下一批历史 KV；
- 对不同连续区间分别使用迁移和重算。

求解器寻找使 $\widehat{TTFT}_{s,n}(p)\le D_s$ 的最低归一化资源成本路径。这里输出的是一个**能够使节点跨过 SLO 门槛的完整方案**，例如“向 A 迁移 40 个 block”，或“向 B 迁移 20 个 block 并重算 10 个 block”，不是一个无法单独产生 SLO 收益的零散动作。若任何恢复方式都无法使节点满足 SLO，该节点不会形成候选方案。

随后，RepKV 为 session $s$ 保留少量选择：暂不准备、准备一个节点，或者准备两个节点。只有当第二个节点能明显增加请求按时完成的机会时，系统才会考虑准备两个节点。路由器仍可检查全部可用节点，但 RepKV 不需要同时尝试所有节点上的所有 KV 操作。

### 3.4 全局选择哪些 session 的方案

所有进入准备队列的 session 一起竞争资源。方案 $p$ 的预计收益包含：session 在当前窗口内返回的概率、准备前后的 SLO 成功概率变化，以及增加一个或两个可行路由节点对 router 的价值。控制器选择：

$$
\max_{x}
\sum_s\sum_{p\in\mathcal P_s}
x_{s,p,t}
\left[
\Delta G_{s,p,t}
-\lambda_B\frac{B_{s,p}}{B^{ref}}
-\lambda_U\frac{U_{s,p}}{U^{ref}}
-\lambda_H\frac{H_{s,p}}{H^{ref}}
\right],
$$

满足：

$$
\sum_{p\in\mathcal P_s}x_{s,p,t}\le 1,
$$

以及当前时段的带宽、后台 NPU 和各节点 HBM 上限。也就是说，每个 session 至多选择一个完整方案，但所有 session 在同一个目标和同一组资源约束下比较。方案是否执行由预计 SLO 收益、当前资源代价和硬容量共同决定。

完整整数问题是 NP-hard。在线实现使用资源价格进行分解：某类资源越紧张，它在上式中的实时价格越高；每个 session 只需在自己的少量完整方案中计算净价值，控制器再解决同一节点和链路上的冲突。该方法以全局目标为方向进行在线近似；小规模实验将与具有完整未来信息的离线最优解比较差距。

### 3.5 按完整方案分批执行

被选中的是完整方案，KV 运行时再把它拆成多个 block batch 执行。调度器优先执行“距离预计返回窗口最近且剩余准备时间最长”的方案；若准备时限相同，再选择净价值更高的方案。每一批的动作已经由节点级方案确定，因此可能是迁移，也可能是重算。

例如，A 需要迁移 40 个 block 才能满足 SLO，B 需要迁移 20 个 block。系统不会因为先向 A 迁移了 10 个 block 就把 A 计作新的可行节点；只有完整方案达到门槛后，才计算该路由选择已经建立。每个 batch 完成后，系统更新返回概率、剩余准备时间、队列和资源状态；若方案已经来不及完成或净价值变为负值，后续 batch 不再执行。

### 3.6 HBM 压力下回收空闲 KV

准备方案在进入目标节点前必须通过 HBM 容量检查。若节点的“活动请求 KV + 活动请求增长预留 + 空闲 KV + 已接纳准备 KV”将超过高水位，RepKV 才启动该节点的回收过程，并在占用降到低水位后停止。

回收只考虑没有正在执行请求的 session KV。对每个可回收的尾部 block batch $e$，系统计算：

$$
Score(e)
=
\frac{\Delta G^{evict}_e+C^{evict}_e}{S_e}.
$$

分数表示“每释放一单位 HBM，需要牺牲多少预计 SLO-goodput 并付出多少下移代价”。分数低的 block 先被回收：若 CPU 内存或本地存储有空间，就把它下移；若其未来使用价值很低，则删除 KV，但保留原始 token 以便未来重算。

节点为所有空闲 KV 维护一个回收优先队列。KV 新进入空闲状态、session 进入返回窗口、返回概率跨过分桶边界或 KV 位置变化时，系统更新受影响条目的分数；HBM 触发高水位时，再校验队首分数并逐项回收。系统不在每个时刻重新计算节点内所有 KV，也不会淘汰正在 prefill 或 decode 的活动 KV。

### 3.7 将 KV 就绪状态交给 router

准备或回收完成后，RepKV 向 router 发布：每个候选节点已经拥有多少连续有效 KV、KV 位于 HBM 还是较慢存储，以及请求到达后预计还需要多少恢复时间。router 仍根据自己的 SLO、队列和负载规则选择最终节点，不需要知道 KV 是由迁移、重算还是回收形成的。

二者的边界是：

- RepKV 决定**请求到达前怎样改变 KV 状态**；
- router 决定**请求到达后最终去哪个节点执行**；
- 两者通过当前候选节点、队列和 KV 就绪元数据交互，而不是共享同一套内部打分公式。

## 4. 正式实验设计

实验不再证明“问题是否存在”，而是直接检验 RepKV 在问题成立时是否提高全局 SLO 服务能力、是否减少资源消耗，以及各机制是否真的必要。

### 4.1 Testbed 与模型

- **硬件**：8 张 Ascend 910B NPU。主配置使用 8 个单卡模型副本；若物理拓扑允许，再加入 4 个双卡副本配置。
- **物理部署**：CEC 主结果至少使用两台物理服务器。正式实验前记录 NPU 分布、NUMA 拓扑、原生网卡带宽和 RTT；若 8 张卡位于同一服务器，则该配置只能用于单机 HBM 与控制面实验。
- **运行时**：基于 vLLM Ascend 或等价 Ascend serving runtime 实现 KV block 接口、后台迁移和精确重算。所有 baseline 共享相同的数据面实现。
- **模型**：以 Qwen2.5-7B-Instruct 为主模型，以另一个可在单张 910B 上运行的 7B–8B 模型做复现；测试 4K、16K 和 32K 上下文。无需把 70B 作为主配置。
- **网络**：先测实际带宽 $B_0$ 和 RTT；主实验使用原生网络，并测试 $0.25B_0$、$0.5B_0$ 和 $B_0$ 三档带宽，以及原生、10 ms 和 30 ms RTT。

### 4.2 Workload 与预测数据

主 workload 是多轮 chatbot，Agent 只作为补充。

1. 使用 ShareGPT 或 WildChat 的真实多轮内容，保留每轮上下文增长和输出长度。
2. 使用 BurstGPT 的真实到达时间；如果无法与 session 内容直接对应，则在相同请求序列上组合多轮内容与合成轮次间隔。
3. 轮次间隔默认使用 log-normal 分布，中位数取 $\{5,15,60\}$ 秒，几何标准差取 $\{1.5,3\}$；同时使用 Gamma 到达间隔测试 $CV\in\{0.5,1,2\}$ 的突发程度。
4. 预测器按时间顺序划分训练、验证和测试集，估计条件返回概率 $q_s(t,\Delta)$。实验分别注入早到、晚到和不再返回，检验预测误差下的退化。
5. Agent 补充实验使用 MetaGPT 或公开 workflow trace，检验显式 advisory 是否能进一步改善准备时机，但不把 Agent 设为主场景。

### 4.3 SLO、负载与资源设置

- **TTFT SLO**：先测模型在 KV 已位于本地 HBM、系统空载时，各输入长度区间的 P99 TTFT，记为 $T_{99}^{hit}(l)$。使用 $\{1.5,2,3\}T_{99}^{hit}(l)$ 表示严格、中等和宽松 SLO，并同时报告绝对毫秒数。
- **Offered load**：找到 router-only 系统的饱和请求率 $\lambda_0$，测试 $\{0.6,0.8,1.0,1.2\}\lambda_0$。
- **KV HBM 预算**：扣除模型权重和运行时预留后，使用剩余 HBM 的 $\{40,60,80\}\%$ 作为可管理 KV 空间。
- **资源环境**：分别构造带宽紧张、后台 NPU 紧张和 HBM 紧张三种配置。目标函数先按各自预算归一化，再提高相应资源权重。
- **额外准备节点数**：主配置最多准备 1 个额外节点，并与最多 2 个额外节点比较。

### 4.4 Baselines

主实验保留三个直接回答核心问题的 baseline；它们使用同一个 router、返回预测器、block大小和迁移—重算数据面。

1. **On-demand**：请求到达后才在原节点排队、跨站迁移和目标节点重算之间选择，衡量主动准备的整体价值。
2. **Eager-full**：session 进入相同返回窗口后，在预测目标节点准备完整 KV，衡量 RepKV 的 SLO 充分方案节省了多少资源。
3. **SYMPHONY-style**：使用相同返回信号，按当前负载预选一个目标节点，并采用机会式预取和层优先级回收，比较最接近的请求到达前 KV 准备策略。论文明确标注该项为控制策略重实现。

恢复数据面作为所有方法共享的基础设施，不把 [Cake](https://proceedings.mlr.press/v267/jin25d.html)、[HCache](https://2025.eurosys.org/accepted-papers.html) 或 [CacheFlow](https://arxiv.org/abs/2604.25080)（preprint）与控制策略混为同类 baseline。主 router 使用 minimum-predicted-TTFT；插件实验再选择 Preble-style 和 DualMap-style router，分别比较“原 router”和“原 router + RepKV”。小规模工作负载另外与离线最优解比较。

### 4.5 Metrics 与比较维度

正文使用四个必要指标：

1. **SLO-goodput**：每秒产生首 token 且 TTFT 满足 SLO 的请求数，作为首要效果指标。
2. **每个 SLO 成功请求的资源消耗**：分别报告跨站传输 GB、后台重算 NPU·s 和准备 KV 的 HBM GB·s，直接体现资源效率。
3. **P99 TTFT**：确认 goodput 提升没有掩盖严重的尾部时延退化。
4. **未被使用的准备比例**：统计在使用前被回收或最终未被路由请求使用的准备字节，解释预测错误造成的浪费。

主结果从两个方向展示价值：在相同资源预算下比较 SLO-goodput；在达到相近 SLO-goodput 时比较三类资源消耗。核心变化维度是 offered load、带宽/HBM 预算、session 长度和返回预测误差。

### 4.6 消融实验

正文保留两个消融。

1. **SLO 充分方案的作用**：把节点级求解器替换为完整 KV 准备，在同一个全局控制器下比较 SLO-goodput 和三类资源消耗。
2. **全局协调与一致回收的作用**：使用“全局控制 / 每个 session 独立贪心”与“本文 SLO 价值回收 / LRU”组成 $2\times2$ 配置，比较 SLO-goodput、每成功请求资源量，以及准备完成后又被回收的 KV 比例。

### 4.7 预测器、最优差距与实现开销

- 报告预测返回窗口覆盖率、平均可用准备时间、过早准备时间和未返回比例。
- 分别注入早到、晚到、未返回和 advisory 缺失，观察 SLO-goodput与资源消耗如何变化。
- 将无预测误差的结果作为性能上界；主结果使用时间顺序训练得到的在线预测。
- 在小规模 session 集合上使用完整未来信息求解离线最优方案，报告在线控制器的 $J_T$ 差距。
- 报告不同 session 数量下的控制决策时延，以及 block batch 大小和 HBM 高低水位的敏感性。
- 每个配置使用相同到达序列完成至少 3 次重复实验，并报告置信区间。

### 4.8 Claim–Evidence 对照

| 论文主张 | 必需证据 |
|---|---|
| RepKV 在有限 CEC 资源下提高全局 SLO 服务能力 | 不同负载和资源预算下的 SLO-goodput与 P99 TTFT |
| RepKV 用更少资源达到相近 SLO-goodput | 每个成功请求的传输 GB、后台 NPU·s 和 HBM GB·s |
| SLO 充分准备优于完整主动准备 | “本文方案 / 完整准备”消融 |
| 全局协调和一致回收避免 session 之间互相伤害 | “全局 / 单 session”与“本文回收 / LRU”的 $2\times2$ 消融 |
| RepKV 能叠加到不同 router | minimum-TTFT、Preble-style 和 DualMap-style router 的 before/after 结果 |
| 在线近似没有严重偏离全局目标 | 小规模离线最优解与在线 $J_T$ 的差距 |

## 5. 论文结构与图表

| 章节 | 主要内容 | 建议图表 |
|---|---|---:|
| Abstract | 背景、问题、核心方法和评估方式 | 无 |
| 1. Introduction | 连贯背景、三步问题链、三个挑战、相关工作边界、三个方法亮点和贡献 | 1 张 teaser：反应式恢复、完整预复制与 RepKV 对比 |
| 2. Background & Motivation | CEC 资源、KV 生命周期、恢复路径、router 与 KV 管理层接口 | 1 张请求时间线 |
| 3. Problem Formulation | 全局 SLO—资源目标、约束和 NP-hard 特例 | 1 张简化决策示例 |
| 4. Design | 返回预测、节点方案、全局控制、分批执行和 HBM 回收 | 1 张系统架构图 + 1 张决策流程图 |
| 5. Implementation | Ascend 910B 原型、KV block接口、迁移和重算数据面 | 1 张实现架构图 |
| 6. Evaluation | 端到端结果、资源效率、两项消融、预测鲁棒性、router 插件和最优差距 | 6–8 张图表 |
| 7. Related Work & Limitations | KV-aware routing、按需恢复、主动准备与预测式管理 | 无主图 |

## 6. 实施前必须确认的条件

1. 确认 8 张 Ascend 910B 分布在几台物理服务器，并测出实际网卡带宽、RTT 和跨节点传输路径。
2. 固定同一返回预测器、同一 router 和同一迁移—重算数据面，再比较不同 KV 控制策略，避免把预测或底层恢复优化误算成 RepKV 收益。
3. 先通过小规模 oracle gap 检查“全局协调”相对单 session 贪心是否存在稳定差距，再决定在线控制器需要多复杂。
4. 投稿前重新核查主动 KV 准备、概率预测和 CEC KV 迁移方向的最新并发工作，并据全文调整新颖性表述与 baseline。
