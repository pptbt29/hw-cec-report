# FrugalKV：One-Page Research Idea

**暂定标题**：*FrugalKV: Resource-Minimal Proactive KV Preparation for SLO-Efficient Collaborative Edge LLM Serving*

**一句话主张**：FrugalKV 不追求“能搬多少 KV 就搬多少”，而是在接近最高全局 SLO-goodput 的方案中，选择跨站带宽、后台 GPU 时间和 HBM 占用最少的 KV 准备与回收方案。

## Abstract

> Collaborative edge computing pools accelerators to absorb local demand bursts. Multi-turn LLM requests remain tied to nodes holding their session KV cache. Restoring KV only after arrival can violate time-to-first-token SLOs. Eagerly replicating complete KV states wastes scarce bandwidth, GPU time, and HBM, potentially harming other requests. We study resource-minimal proactive KV preparation and eviction while preserving near-maximal global SLO-goodput. FrugalKV predicts completed sessions' return windows and activates preparation only when return risk becomes relevant. A rolling controller compares next-batch transfer, recomputation, and waiting actions using their global goodput and resource effects. It retains near-best goodput actions and executes the least-cost one, while projected HBM pressure triggers loss-aware idle-cache eviction. We will evaluate a router-compatible prototype using multi-turn traces, heterogeneous edge nodes, routers, and prediction errors. We expect FrugalKV to approach oracle goodput with lower bandwidth, GPU, and HBM consumption than reactive migration, full prefetching, and cache-only policies.

该 Abstract 共 10 句，依次是：背景、一般问题、两句动机、技术问题、三句方法、评估和预期结果。最后一句是研究假设，不是已经取得的实验结果。

## 1. Introduction 的六段逻辑

**第 1 段：场景。** Collaborative Edge Computing（CEC）可让不同边缘节点协同承接局部突发负载，但多轮 LLM session 携带跨请求复用的 KV cache，因此空闲 GPU 未必能在 TTFT SLO 内直接接管请求。

**第 2 段：矛盾。** 请求到达后再恢复 KV 容易超时；请求完成后立即复制完整 KV 又会消耗有限的跨站带宽、后台 GPU 和 HBM，而用户可能晚到或不再返回。

**第 3 段：现有工作缺口。** Llumnix 主要在请求运行后迁移执行状态，DualMap 主要在请求到达后进行 KV-aware routing，SYMPHONY 利用 advisory 提前选择节点并分层预取；它们分别解决反应式迁移、路由和主动准备，但没有直接优化“接近最高全局 SLO-goodput 时最少需要准备和保留多少 KV”。

**第 4 段：挑战。** 某个 session 的准备可以提高其下一轮达标概率，却会占用其他 session 需要的链路、GPU、HBM，并可能改变后续路由和队列；同时，返回时间预测可能早、晚或完全错误。

**第 5 段：方法。** FrugalKV 使用两层目标：先保持近最高的全局 SLO-goodput，再在近似等效方案中最小化资源；每个 session 被唤醒后只决定下一批 KV 是迁移、重算还是等待，HBM 压力则单独触发与同一目标一致的空闲 KV 回收。

**第 6 段：贡献与证据。** 论文贡献是面向全局目标的资源最小化 KV 生命周期控制及其可扩展在线近似，验证则通过真实 trace、多种宿主路由器、预测误差和离线 oracle 检查其 goodput、资源效率与最优差距。

## 2. 问题定义

SLO 是固定的 TTFT 门限。系统要提高的是单位时间内满足门限的请求数，即 SLO-goodput，而不是“提高 SLO”。给定不由本文设计的宿主路由器 $\rho$，KV 策略 $\pi$ 的长期 SLO-goodput 定义为：

$$
G(\pi;\rho)
=
\liminf_{T\rightarrow\infty}
\frac{1}{T}
\mathbb E
\left[
\sum_{r:A_r\le T}
\mathbf 1\{TTFT_r(\pi,\rho)\le D_r\}
\right].
$$

$A_r$ 是请求 $r$ 的到达时间，$D_r$ 是其 TTFT SLO。资源成本定义为：

$$
C(\pi)
=
\beta_B\bar B(\pi)
+\beta_G\bar U(\pi)
+\beta_M\bar H(\pi),
$$

其中，$\bar B$、$\bar U$、$\bar H$ 分别是长期平均跨站传输量、后台 GPU 时间和 HBM 驻留量；$\beta_B$、$\beta_G$、$\beta_M$ 将三类资源归一化到可比较成本。由于三类资源可以互相替代，论文还应报告完整 Pareto 曲线，不能只报告一个人为权重下的结果。

只最大化 $G$ 会鼓励系统使用所有仍可用资源，单独最大化 $G/C$ 又可能偏向“几乎不用资源、但 goodput 很低”的策略。因此采用两层目标：

$$
G^*
=
\max_{\pi\in\Pi_{\mathrm{safe}}}G(\pi;\rho),
$$

$$
\pi_\epsilon^*
\in
\arg\min_{\pi\in\Pi_{\mathrm{safe}}}C(\pi)
\quad
\mathrm{s.t.}
\quad
G(\pi;\rho)\ge(1-\epsilon)G^*.
$$

$\Pi_{\mathrm{safe}}$ 是不抢占前台请求、且不超过 HBM 硬容量的策略集合。$\epsilon=0$ 表示在精确最高 goodput 的策略中选择最省资源者；实践中使用很小的 $\epsilon>0$，避免为统计误差范围内的微小 goodput 增益付出大量资源。资源预算仍是安全上限，而不是应该被用满的目标。

## 3. FrugalKV 方法

### 3.1 Session 唤醒，而非周期性全局重排

request 完成时，系统只登记最新 KV 版本，并用历史轮次间隔估计下一轮返回分布。session 接近预计返回窗口时才被唤醒；没有可靠历史信息或外部提示时，系统不主动跨站准备。advisory 可以修正返回窗口，但不是 FrugalKV 的必要输入，也不是本文贡献。

### 3.2 先保 goodput，再选最省的下一步

被唤醒的 session 每次只比较三个基础动作：等待、向候选节点迁移下一批 KV、在候选节点重算下一批 KV。有限时域求解器同时估计动作序列对全局 goodput 的贡献 $Q_s^G(a)$ 和资源成本 $Q_s^C(a)$；goodput 贡献需要扣除该动作占用共享资源、改变节点队列后对其他请求造成的预计损失。

设当前动作中的最高预测 goodput 价值为 $Q_{s,\max}^G$，先保留：

$$
\mathcal A_s^\eta
=
\{a:Q_s^G(a)\ge Q_{s,\max}^G-\eta\},
$$

再选择：

$$
a_s^*
=
\arg\min_{a\in\mathcal A_s^\eta}Q_s^C(a).
$$

$\eta$ 是局部近似容差。若等待与迁移具有近似相同的 goodput，系统选择等待；只有准备能够带来不可忽略的全局收益时才消耗资源。一次迁移 10 个 block 即使尚不足以满足 SLO，也可因其后续价值被选择；若剩余准备已来不及完成且部分 KV 无法降低到达后的恢复时间，则不启动第一批。

每个 block 执行完成后重新求解，因此预测或资源状态变化不会锁死一个完整迁移计划。全局 $\epsilon$ 与局部 $\eta$ 并不严格等价；本文只声称这是可扩展在线近似，并通过小规模离线联合 oracle 测量最优差距。

### 3.3 HBM 压力触发价值一致的回收

正常准备阶段不主动删除其他 session 的 KV。只有节点的“活动 KV + 短期增长预留 + 空闲 KV + 已批准准备空间”预计超过安全容量时，才启动回收。每个空闲 session 提交一个可执行的尾部 KV 回收动作，并按照：

$$
E(e)
=
\frac{\Delta G_{\mathrm{loss}}(e)+C_{\mathrm{move}}(e)}{M_{\mathrm{free}}(e)}
$$

排序。$\Delta G_{\mathrm{loss}}(e)$ 是下移或删除导致的预计全局 goodput 损失，$C_{\mathrm{move}}(e)$ 是下移成本，$M_{\mathrm{free}}(e)$ 是释放的 HBM；分数越小越先回收。正在 prefill 或 decode 的活动 KV 不进入该队列。

### 3.4 与现有路由器组合

FrugalKV 不决定真实请求最终路由到哪里。它只向宿主路由器提供每个候选节点已经准备的 KV 数量、所在存储层和剩余恢复时间。因此实验应逐一比较“原路由器”与“原路由器 + FrugalKV”，并承认完全忽略 KV 状态的路由器未必能利用准备收益。

## 4. 预期贡献

1. **主要贡献：资源最小化的全局 SLO-goodput KV 控制。** 首先保持近最高的全局 SLO-goodput，再寻找跨站带宽、后台 GPU 和 HBM 消耗最小的准备方案，避免“为了保证 SLO 而总是使用最多资源”。
2. **支撑贡献：统一价值下的准备与回收。** session 侧按未来全局 goodput 价值选择下一批迁移、重算或等待；节点侧按单位 HBM 回收造成的全局 goodput 损失淘汰空闲 KV，使两个 phase 服务于同一目标。

不作为贡献的内容包括：新的返回预测模型、新的 request routing policy、新的 KV 传输或压缩内核、公平调度，以及硬 SLO 保证。

## 5. 论文结构与图表

| 章节 | 内容 | 图表 |
|---|---|---:|
| 1. Introduction | 六段问题—缺口—方法—贡献逻辑 | 1 张 teaser |
| 2. Background & Motivation | 到达后恢复与完整预取的矛盾；资源用量与 goodput 饱和现象 | 2 张动机图 |
| 3. Problem Formulation | 两层目标、资源定义、在线近似边界 | 1 张符号表 |
| 4. Design | session 唤醒、滚动动作、全局影响估计、HBM 回收、router adapter | 2 张图 |
| 5. Implementation | vLLM/LMCache 类数据面接入和控制面开销 | 1 张表 |
| 6. Evaluation | 主结果、Pareto 前沿、预测鲁棒性、消融和 oracle gap | 5–6 张图表 |
| 7. Related Work & Limitations | 机制边界和并发工作风险 | 无主图 |

## 6. 最小但足够的实验设计

**实验 A：问题是否存在。** 在 3–8 个异构 CEC 节点上改变带宽、RTT、GPU 速度和 HBM，测量按需恢复造成的 SLO miss，以及完整预取在 goodput 已饱和后继续增加的传输、GPU 和 HBM 消耗。

**实验 B：核心主张。** 使用带 session ID 和真实轮次时间戳的 chatbot/agent trace，对比 arrival-time restore、完整预取、SYMPHONY-style single-target staging、reactive migration、cache-only eviction 和“主动准备 + LRU”。核心证据是：在相同 SLO-goodput 下资源更少，或者在相同资源下 SLO-goodput 更高；同时给出 goodput–bandwidth、goodput–GPU 和 goodput–HBM Pareto 曲线。

**实验 C：必要性与鲁棒性。** 去掉两层选择、未来价值、其他请求损失或价值一致回收，并注入系统性早到、晚到、false return 和 missed return。小规模离线 oracle 用于报告 goodput gap 和 resource gap，不作为主实验输入。

主要指标是 SLO-goodput、SLO attainment、P95/P99 TTFT、跨站字节、后台 GPU·s、准备 KV 的 HBM GB·s、无效准备比例和控制面开销。主实验必须采用时间切分的历史预测，不能使用 oracle 未来信息。

## 7. 相关工作与新颖性风险

- [Llumnix, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/sun-biao) 提供运行中请求和 KV 的低停顿迁移；FrugalKV 关注下一轮请求到达前、资源最小化的准备与回收。
- [DualMap, ICLR 2026](https://openreview.net/forum?id=zCadrJ32Xn) 在两个候选实例内平衡 cache affinity、负载和 TTFT SLO，但不主动跨节点准备已有 session KV，可作为宿主路由器。
- [SYMPHONY, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/agarwal) 已覆盖 advisory 驱动的目标选择、跨节点分层预取和机会式 HBM 管理，是最接近且最危险的已有工作。FrugalKV 必须实验证明“两层资源最小化目标 + 多 session 全局影响 + 价值一致回收”不是简单参数调优。
- [CachedAttention, ATC 2024](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) 和 [InferCept, ICML 2024](https://proceedings.mlr.press/v235/abhyankar24a.html) 提供分层 KV 存储与恢复机制，可作为数据面而非竞争性的控制目标。

开工前的关键否决实验是：若 FrugalKV 相比“SYMPHONY-style staging + SLO target selector + LRU”不能在相同 goodput 下显著降低资源，或者相比简单的全局收益/成本优先队列没有稳定优势，那么当前方法不足以形成独立论文贡献。
