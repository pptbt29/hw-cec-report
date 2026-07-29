# CEC 中 KV 调度、请求调度与 Batching 联合优化的研究机会审计

> 版本：2026-07-24  
> 结论性质：idea-stage go/no-go 审计，不是新颖性已被证明的声明  
> 证据标准：优先采用 CEC/MEC、系统、网络和机器学习领域的正式顶刊顶会论文；arXiv-only 工作仅作为并发风险，不与正式论文等权。

## 1. 结论

“把 KV 调度、请求调度和 batching 融合起来”本身已经不新。2023--2026 年的正式顶会工作已经覆盖以下组合：

- 单实例内的 KV 内存管理、请求准入和 continuous batching；
- 分层 KV 加载与 cache-loading-aware batch formation；
- 分布式 KV-aware routing 与本地 batch formation；
- 跨实例 KV 传输、请求切分和 SLO-aware batching；
- KV swap、抢占式请求调度与迭代级 batch formation。

因此，不能将论文的贡献写成“首次联合优化三者”，也不能通过换成 SafeRL、Lyapunov optimization 或新的多目标函数获得新颖性。

仍然存在一个有条件的 CEC 研究机会：

> 在低带宽、时变且存在共享瓶颈的多层 CEC 中，跨节点 KV 的传输、重算或保留决定请求何时进入 batch；batch 的成员和启动时间又决定能够隐藏多少 KV 传输、占用多少 HBM，以及后续请求的排队和状态可用性。论文可以研究这一跨节点、跨时间尺度的双向闭环，而不是把全局路由器和各节点的本地 batcher 串联起来。

这个方向目前只能给出“值得验证”，不能直接给出“值得写论文”。它成立的必要条件是：在真实 CEC 参数范围内，强解耦基线相对联合 oracle 存在稳定且足够大的性能缺口。如果 continuous batching 已经吸收了大部分网络与状态差异，或者迁移 KV 在目标链路上几乎从不经济，则应终止该方向。

## 2. 顶会工作已经覆盖到哪里

| 工作 | 正式出处 | 已覆盖的核心机制 | 对本文想法的直接阻塞 |
|---|---|---|---|
| [vLLM](https://doi.org/10.1145/3600006.3613165) | SOSP 2023 | PagedAttention、KV sharing、动态内存分配和请求准入 | “KV 管理 + continuous batching”不是新贡献 |
| [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao) | OSDI 2024 | 请求及其运行中内存状态的跨实例 live migration | “迁移 KV 来动态负载均衡”不是新贡献 |
| [SGLang](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) | NeurIPS 2024 | RadixAttention、prefix-aware scheduling、cache eviction 与 continuous batching | 单 runtime 内 cache-aware request ordering 已被覆盖 |
| [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) | OSDI 2024 | Chunked prefill、stall-free hybrid batches | Prefill/decode 的 batch 组成和干扰控制已有强基线 |
| [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) | FAST 2025，Best Paper | 分布式 KVCache、P/D 集群、全局 SLO-aware scheduler | 分布式 KV 数据面与全局请求调度已被覆盖 |
| [Preble](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html) | ICLR 2025 | 分布式 prefix KV reuse、全局 locality/load-aware routing、本地调度 | “全局 KV-aware routing + 本地 batching”已被覆盖 |
| [ThunderServe](https://proceedings.mlsys.org/paper_files/paper/2025/hash/c2a0e26dd9ee7d57e92bb1c24b39659a-Abstract-Conference.html) | MLSys 2025 | 异构 GPU、网络带宽感知部署和轻量在线重调度 | 单纯强调网络与算力异构不是 CEC 独有贡献 |
| [NEO](https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html) | MLSys 2025 | KV/attention CPU offload、load-aware scheduling、非对称 sub-batches | 本地 KV 层级与 batch size 的联合控制已有先例 |
| [Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang) | OSDI 2026 | HBM/CPU/SSD KV、cache-loading-aware scheduler、balanced batch formation | “用 batch 隐藏 KV 加载”已被正面覆盖 |
| [LMetric](https://www.usenix.org/conference/osdi26/presentation/zhang-dingyan) | OSDI 2026 | 用新增 prefill token 与当前 batch size 联合衡量 locality 和负载 | 简单地给路由 score 加 batch size 不足以构成贡献 |
| [Libra](https://www.usenix.org/conference/nsdi26/presentation/ruan-libra) | NSDI 2026 | 请求 token 级切分、全局调度、本地 SLO-aware batch、分块 KV 传输 | “请求迁移/切分 + KV 传输 + batch composition”已有高度重叠 |
| [FastServe](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang) | NSDI 2026 | 迭代级抢占、GPU/Host 状态 swap、优先级 batch formation | 单实例内部的 swap、任务次序和 batch 联合调度已覆盖 |
| [Bidaw](https://www.usenix.org/conference/fast26/presentation/hu-shipeng) | FAST 2026 | 多轮 session KV、分层存储、KV-aware request ordering、continuous batching | 长 session 的本地状态恢复与 batching 已覆盖 |

两个 CEC 方向的正式工作也分别覆盖了部分问题：

- *Co-optimizing Request Scheduling and KV Caching for Edge LLM Serving*，IEEE Internet of Things Journal 2026，联合边缘请求放置与 KV 保留，但明确不处理跨节点 KV 拉取，重点是 prefill/TTFT，未联合建模真实 continuous batching。
- *Edge Inference for Large Language Models With Pipeline Parallelism and Batching*，IEEE Transactions on Communications 2026，联合带宽、模型划分与离线 batch 调度，但不处理持久 KV 的放置、复用或迁移。

这两篇论文说明 CEC 社区认可“边缘 LLM 状态管理”和“边缘 batching”两个问题，但不能据此声称三者闭环尚无相关工作。真正的 novelty blockers 主要来自 OSDI、NSDI、FAST、SOSP、EuroSys、MLSys、ICLR 和 NeurIPS 的系统工作。

## 3. 为什么 CEC 可能产生不同的问题

### 3.1 不是固定网络时延，而是状态就绪与 batch formation 的内生耦合

设请求为 $r$，候选执行节点为 $j$，请求在节点 $j$ 的状态就绪时间为

$$
t_{rj}^{\mathrm{ready}}
=
\min
\left\{
t_{rj}^{\mathrm{local}},
t_{rj}^{\mathrm{transfer}},
t_{rj}^{\mathrm{recompute}}
\right\}.
$$

如果链路不存在竞争，可以把 $t_{rj}^{\mathrm{transfer}}$ 近似成一个固定附加时延，此时 CEC 只是普通分布式调度的背景替换。真正有区别的情形是

$$
t_{rj}^{\mathrm{transfer}}
=
f\left(
s_r,
\mathcal{F}_{\ell},
b_{\ell}(t),
\mathcal{P}_{rj}
\right),
$$

其中 $s_r$ 是待传输 KV 的大小，$\mathcal{F}_{\ell}$ 是共享链路 $\ell$ 上并发的状态流集合，$b_{\ell}(t)$ 是时变可用带宽，$\mathcal{P}_{rj}$ 是跨层路径。此时一个请求的状态准备会改变其他请求的就绪时间。

### 3.2 Batch 的价值是集合价值，不是单请求 score 的和

令 $B$ 表示一个候选 batch，$T_j(B)$ 表示它在节点 $j$ 的执行时间。现代 ragged attention 与 continuous batching 减少了传统 padding 问题，但没有消除集合级非线性：

$$
T_j(B)
\neq
\sum_{r \in B} T_j(\{r\}).
$$

Batch 的可行性至少取决于所有成员的状态是否及时就绪、HBM 是否足够以及 SLO 是否仍有余量：

$$
\max_{r \in B} t_{rj}^{\mathrm{ready}} \leq t_B^{\mathrm{start}},
$$

$$
M_j^{\mathrm{model}}
+ M_j^{\mathrm{active}}(B)
+ M_j^{\mathrm{cached}}
\leq M_j^{\mathrm{HBM}},
$$

$$
\widehat{C}_r(B,j) \leq d_r,\quad \forall r \in B.
$$

其中 $t_B^{\mathrm{start}}$ 是 batch 启动时刻，$\widehat{C}_r(B,j)$ 是请求 $r$ 的预测完成时刻，$d_r$ 是其 SLO deadline。

### 3.3 闭环来自五个方向

1. 路由决定 KV 是本地命中、跨节点传输还是重新 prefill。
2. KV 的完成时间决定请求的 batch-ready time。
3. Batch membership 决定请求的排队和执行时间，并改变其他请求的 TBT。
4. Active batch 占用 HBM，可能驱逐空闲 session KV，改变未来 locality。
5. KV 流量与请求输入、模型数据或其他 CEC 业务共享链路，使状态准备具有跨请求外部性。

只要其中第 2、4、5 点在实验中很弱，联合控制器就可能退化成复杂但无必要的工程拼接。

## 4. 推荐的论文问题

### 4.1 暂定题目

**BatchEdge: Contention-Aware KV-Ready Batch Orchestration across the Computing Edge Continuum**

中文概括：面向计算边缘连续体的链路竞争感知 KV 就绪与批次编排。

### 4.2 核心研究问题

在 access edge、aggregation edge 和 regional edge 构成的多层 CEC 中，如何在线联合决定：

- 请求在哪个节点执行；
- 远端 KV 应传输、部分传输、重算还是等待本地资源；
- 共享链路上的 KV 准备顺序和带宽份额；
- 哪些已到达或即将就绪的请求进入下一轮 prefill/decode batch；
- 何时启动 batch，以及是否用现有计算窗口隐藏状态传输；

从而最大化 SLO goodput，并控制 P99 TTFT、P99 TBT、链路流量和 HBM 压力。

这里的核心贡献必须是“batch-level state readiness”这一决策抽象及其在线系统机制，而不是目标函数中同时出现五类成本。

### 4.3 建议的系统边界

第一篇论文不应同时加入移动性、模型并行、LoRA adapter placement、能耗、碳排放和用户行为预测。建议固定以下边界：

- 同一模型或少量模型副本；
- 多层异构 GPU 节点；
- 多轮 session 产生持久 KV；
- 链路带宽低于数据中心 RDMA，且存在共享瓶颈与时变可用容量；
- 请求在线到达，输入长度在到达时可知，输出长度仅可估计；
- 后续 session 请求和未来到达不可知；
- 使用现有 continuous batching 执行器，不重新设计 attention kernel。

移动性只在后续实验确认它显著改变候选节点集合、链路路径或 KV 可达性时再加入；否则它不是必要变量。

## 5. 建议的机制

### 5.1 两个时间尺度

- 慢时间尺度：根据观测到的 session reuse、HBM 压力和拓扑，决定 KV 保留、降层、复制或删除。
- 快时间尺度：根据当前队列、状态位置、链路竞争和 SLO slack，联合做 routing、transfer/recompute 选择与 batch admission。

慢时间尺度不需要准确知道“还剩多少轮请求”；它只能基于在线估计和不确定性做有限准备，并需要显式统计浪费的复制和预取。

### 5.2 Batch option，而不是独立 request score

每个候选 batch option 可写为

$$
o =
\left(
j,
\tau,
B,
\boldsymbol{u},
\boldsymbol{z},
\boldsymbol{q}
\right),
$$

其中 $j$ 是执行节点，$\tau$ 是启动窗口，$B$ 是请求集合，$\boldsymbol{u}$ 是每个请求的执行 token/chunk 配额，$\boldsymbol{z}$ 是本地命中、传输、重算等状态动作，$\boldsymbol{q}$ 是链路资源需求。

一个 option 只有在状态就绪、HBM、链路容量和 SLO 约束同时满足时才是可行的。控制器在滚动时域内从有限候选 option 中选择互不冲突的组合：

$$
\max_{\boldsymbol{x}}
\sum_{o \in \mathcal{O}}
V_o x_o,
$$

$$
\sum_{o:r \in B_o} x_o \leq 1,\quad \forall r,
$$

$$
\sum_{o} q_{o\ell t}x_o \leq b_{\ell}(t),\quad \forall \ell,t,
$$

$$
x_o \in \{0,1\}.
$$

$V_o$ 不是简单的 KV hit value，而应反映 SLO-feasible tokens、batch occupancy、状态准备成本和外部性。系统实现不必在线求解完整整数规划；可采用受约束的候选生成、primal-dual 选择或低开销 greedy，并用小规模 oracle 衡量最优性缺口。

### 5.3 需要避免的算法包装

- SafeRL 不能证明问题重要，也不能自动保证未见负载下的 SLO。
- Lyapunov optimization 适合平均队列、平均带宽或长期能耗约束，但 P99 TTFT、P99 TBT 和单请求 deadline 不是平均约束，仍需显式 admission、deadline 或 chance constraint。
- Column generation、matching 或 primal-dual 只是实现工具。只有在 batch 集合价值和链路竞争造成现有分解失效后，它们才有论文意义。

## 6. 候选贡献的优先级

| 候选点 | 初步判断 | 原因 |
|---|---|---|
| CEC 中 KV-ready batch 的闭环编排 | 首选，但需 go/no-go 验证 | 与 Mooncake 的全局数据面、Strata 的本地 loading-aware batching 形成明确交叉边界 |
| Pooling--fragmentation phase diagram | 强测量贡献或首选问题的核心 finding | 可回答“节点更多是否反而造成 KV 和 batch 碎片化”，但单独作为算法论文偏弱 |
| Batch option value 指标 | 可作为诊断与机制组件 | 比 KV hit ratio 更接近 batch-level SLO capacity，但单独容易被批评为指标重命名 |
| Cohort-level KV placement | 备选 | 具有集合价值，但依赖未来共同到达预测，容易被质疑 oracle 假设和复制浪费 |
| KV restoration coflow | 机制组件，不宜做主贡献 | Coflow scheduling 的方法迁移过于直接 |
| KV affinity 与 batch compatibility | 暂不建议 | Ragged attention 和 continuous batching 已削弱传统长度兼容性问题，需先证明仍有强干扰 |
| 大 batch 的 KV eviction 外部性 | 暂不建议单独成文 | 与本地 memory-aware batching、Strata、eLLM 等重叠较大 |
| TTFT 优化导致 decode burst | 独立备选问题 | 可能真实，但与 P/D disaggregation 和 traffic shaping 文献重叠，需要单独审计 |

最好的论文叙事不是列出八个优化变量，而是由两个可复现的 finding 驱动：

1. 在某个真实 CEC 参数区间，解耦的“KV-aware global router + local continuous batcher”发生稳定的策略排序反转或显著 oracle gap。
2. 该差距来自 batch-ready time、共享链路竞争和 batch 集合价值的闭环，而不是预测更准或计算预算更多。

## 7. 最小验证实验

### 7.0 当前模拟器不能直接回答这个问题

项目现有计算模拟器已经提供 batch-aware 的 roofline 估算接口，但当前路由实验的 decode 路径使用固定满批近似：先按固定 `decode_batch_size` 计算一个同质 batch，再把总时间平均分摊给请求；它不维护请求逐轮加入、退出、状态变为 ready、prefill chunk 插入和 HBM 驱逐等事件。因此，它能测算给定 batch size 下的服务需求，不能验证“路由与真实 continuous batching 是否发生策略排序反转”。

若直接在现有固定满批模型上比较联合策略，得到的收益很可能只是目标函数自洽，而不是系统效果。第一步应新增迭代级 batch state machine，而不是先实现复杂 KV 策略。

### 7.1 第一阶段：两周内完成的事件驱动模拟

模拟器必须使用真实 serving profile，而不是给每个请求设固定服务时间。至少从 vLLM 或 SGLang 测得：

- 不同 prompt/context 长度与 batch token 数下的 prefill 时间；
- 不同 active sequence 数和 context 长度下的每 token decode 时间；
- HBM 中每 token KV 占用；
- GPU--host、节点间不同带宽下的 KV 加载时间；
- chunked prefill 与 continuous batching 的调度行为。

CEC 设置至少包括：

- 3--8 个 access/regional 节点；
- 0.5、1、5、10、25、100 Gbps 链路；
- 单瓶颈与多跳共享路径；
- 同构和异构 GPU；
- 稀疏、稳态和突发到达；
- 不同 session reuse 间隔、context 长度和 HBM 配额。

### 7.2 必须比较的基线

1. Sticky routing + local continuous batching。
2. Join-shortest-queue + local batching。
3. LMetric 类的 KV locality/load score。
4. Preble 类的 global KV-aware routing + local scheduling。
5. Mooncake 类的 transfer/queue/recompute completion-time routing。
6. ThunderServe 类的异构网络和部署/重调度策略。
7. Libra 类的 active request split/migration，若系统允许迁移运行中请求。
8. Strata 类的 local KV-loading-aware batch formation。
9. CEC 正式工作中的 request/KV co-optimization 与 edge batching。
10. 小规模 batch-level joint oracle。

不能只与 JSQ、LRU、random、nearest-edge 比较。

### 7.3 主要指标

- SLO goodput，作为第一主指标；
- P50/P95/P99 TTFT；
- P50/P95/P99 TBT 或 TPOT；
- batch occupancy 和每轮有效 token 数；
- KV 传输字节、链路利用率及等待时间；
- 无效预取/复制字节；
- HBM 使用、驱逐与重算量；
- 调度开销。

## 8. Go/No-Go 判据

### Go

只有同时满足以下条件才继续：

1. 在合理而非极端的 CEC 参数区域，最强解耦基线与 batch-level oracle 的 SLO goodput 差距稳定达到约 $15\%$--$20\%$ 或更高。
2. 联合在线方法能关闭大部分 oracle gap，而不是只在单一 trace 上好看。
3. 固定带宽、无链路竞争或禁用真实 continuous batching 后，优势显著下降，从而证明贡献确实来自所声称的闭环。
4. 不需要未来输出长度、剩余 session 轮数、未来请求或用户轨迹的 oracle 信息。
5. 调度器的运行时间显著小于一个调度周期。

### No-Go

出现任一情况就应降级或终止：

- 在广泛参数范围内，强基线与 oracle 的差距小于约 $5\%$；
- 只有在极低带宽下联合方法才有优势，但该区域内 KV 迁移始终劣于重算或 sticky routing；
- 优势主要来自准确预测未来请求、输出长度或移动轨迹；
- local continuous batching 已经吸收了状态就绪差异；
- 算法只是将 Mooncake、Preble 和 Strata 串联，消融后不存在新的双向反馈机制。

若算法收益较小但发现清晰、可复现，可转为一篇 CEC measurement/characterization paper，重点报告 pooling--fragmentation 相变、迁移经济区间和解耦策略失效边界。

## 9. 最危险的审稿意见

### “这是 Mooncake + Strata/Preble 的工程拼接”

回应不能是“我们使用了新的优化器”。必须通过反例、消融和机制分析证明：任何独立 per-request routing score 加任一本地 batcher，都无法表达只有一组请求共同放置和共同准备状态时才产生的 batch option value。

### “CEC 上迁移 KV 根本不经济”

先做 break-even 分析：

$$
t^{\mathrm{transfer}}
<
\min
\left\{
t^{\mathrm{wait-local}},
t^{\mathrm{recompute}}
\right\}.
$$

如果该不等式在目标模型、上下文和链路范围内几乎从不成立，就不要研究 KV migration；转而研究 request steering、selective recompute 或 regional KV pooling。

### “收益来自 oracle 未来信息”

必须区分：

- oracle：仅用于量化上界；
- online estimator：只使用到当前时刻的到达、队列、KV 位置、链路和可测运行状态；
- robustness evaluation：输出长度预测误差、reuse 预测误差、带宽预测误差和突发到达。

## 10. 当前建议

现在不应直接写完整 paper，也不应先决定 SafeRL 或 Lyapunov。下一步只做一个 falsification prototype：

1. 加入真实 continuous batching 和 profile-based service curves；
2. 实现 LMetric、Preble/Mooncake 风格的强解耦基线；
3. 实现小规模 batch-option oracle；
4. 扫描 CEC 带宽、共享竞争、异构性和 session locality；
5. 观察是否存在稳定的 oracle gap 与策略排序反转。

若 gap 存在，再设计低开销在线算法；若 gap 不存在，就应明确否定“三者联合优化是必要的”这一假设。

## 11. 证据可信度说明

本轮使用的核心论据来自正式顶会/顶刊页面和用户提供的正式论文。自动题名核验工具因外部索引服务瞬时失败将全部条目标记为 `verify_pending`，因此没有用该自动结果支撑任何存在性结论；关键论文的正式出处由 USENIX、ICLR Proceedings、ACM/DOI 或 IEEE 正式版本逐项确认。低质量 handover preprint 未被用于支撑问题重要性或方法有效性。

## 12. 独立敌对评审

一名未接收执行者摘要的独立 reviewer 直接读取本报告后，将 BatchEdge 判定为：

> 当前是“合理组合，且靠近工程拼接”，还不是已经成立的“新问题结构”。

其评分中，CEC KV-ready batch 闭环编排的 importance 为 $5/5$，novelty 为 $3/5$，defensibility 为 $2/5$，evidence readiness 为 $2/5$。第二候选 pooling--fragmentation phase diagram 的可行性较高，但当前 evidence readiness 仅为 $1/5$。

独立评审的正式判决是：**完整顶会系统论文暂时 NO-GO；只对预算封顶的两周 falsification prototype 给 conditional GO。** 唯一能改变判决的证据，是在真实 serving profile、现实 session/到达 trace 和 CEC 带宽轨迹下，强网络感知解耦基线相对 joint oracle 存在稳定且由共享链路竞争因果导致的 $15\%$--$20\%$ SLO-goodput gap。

完整评审 trace 保存在 `.aris/traces/idea-creator/20260724_kv_batching_fresh_jury.md`。
