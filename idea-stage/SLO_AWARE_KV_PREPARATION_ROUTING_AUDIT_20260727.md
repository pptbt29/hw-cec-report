# CEC 中 SLO-Aware KV Preparation 与 LLM Request Routing：机制边界、查新与立题审计

**检索与审计截止日期**：2026-07-27  
**研究对象**：在未来 stateful LLM request 尚未真正到达时，利用不可靠 advisory 提前准备 session KV；请求到达后再选择执行节点，以提高 CEC 中的长期 SLO-goodput。  
**结论口径**：正式顶会顶刊优先；arXiv-only 工作只作为并发风险；“没有找到完整交集”不等于“组合本身具有新颖性”。

## 固定开场分析

### 快速结论

该方向不是没有研究空间，但空间已经非常窄。以下宽泛命题均已不成立：

- 首次联合 KV cache 与 request routing；
- 首次以 SLO 为目标做 KV-aware routing；
- 首次在真实 request 前主动准备 KV；
- 首次在 edge/CEC 中迁移 KV 以缓解负载不均；
- 用 Lyapunov、CMDP、SafeRL 或 MILP 取代 SYMPHONY 的启发式即可构成主要新颖性。

截至检索日期，尚未核实到一篇正式论文同时完整覆盖：

1. 不确定 advisory 驱动的 future-session KV preparation；
2. 请求到达后的 late-binding SLO-aware routing；
3. 多个候选执行节点之间的 recourse；
4. 共享 CEC backhaul、异构算力和 HBM；
5. 长期 SLO、投机带宽或 HBM-byte·time 预算。

但这个“交集暂未被一篇论文完整占满”只能支持继续做低成本 kill test，不能直接支持立题。独立对抗审稿给出的当前评分是：

| 维度 | 当前评分 | 原因 |
|---|---:|---|
| 问题重要性 | 7/10 | 状态位置、排队、网络与 SLO 的冲突真实存在 |
| 当前新颖性 | 3/10 | 主要组成部分已被 SYMPHONY、KVFlow、Mooncake、DualMap 等分别覆盖 |
| 当前成熟度 | 3/10 | 缺少真实 advisory trace、direct-risk 全文和 recourse oracle gap |

因此当前决策应是：

> **Pivot for kill-test，暂不进入完整系统和算法实现。**

只有当“多候选 late-binding recourse”相对 SYMPHONY 式单目标 early binding 在相同信息和资源预算下存在稳定、显著的 oracle gap，才值得继续。

## 1. 三篇核心论文的事实边界

### 1.1 SYMPHONY 到底做了什么

[SYMPHONY（NSDI 2026）](https://www.usenix.org/conference/nsdi26/presentation/agarwal)的基本顺序是：

```text
advisory 到达
    ↓
全局调度器按简单的跨节点均匀分配策略预选执行节点
    ↓
目标节点提前获取并提升该 session 的 KV 状态
    ↓
真实请求到达
    ↓
请求被路由到 advisory 阶段已经选定的节点
```

论文只把默认策略描述为 “distributes requests evenly across nodes”。示例中选择的是活动请求较少的节点，但目标节点仍有活动请求，因此不能说它要求“完全空闲节点”；也不能把它严格表述为 round-robin 或某个有明确定义的 `argmin` 公式。

更重要的是，SYMPHONY 没有联合计算：

- KV 大小及完成迁移所需时间；
- 链路当前和未来占用；
- 目标节点排队完成时间；
- prompt/output 长度不确定性；
- false-positive advisory 的浪费；
- TTFT/TPOT 违约概率；
- 是否应继续在 owner 上执行、迁移、部分准备或重算。

所以它确实是机制驱动、启发式的系统，而不是目标驱动的联合控制器。

### 1.2 SYMPHONY 是否把全部 KV 搬进目标节点

不能简单回答“是”或“否”，必须区分三层含义。

#### 跨节点状态获取

目标 node manager 会从论文记录的当前 KV 位置取回 session 状态。在 HBM 充足的示例里，完整 session KV 可在请求到达前出现在目标节点的 GPU、host memory 和 disk。

#### 目标节点的完整低层副本

SYMPHONY 始终让最慢层保留完整副本。因此目标节点的 DRAM、SSD 或远程低层可以保存完整状态，这为后续驱逐和恢复提供正确性基础。

#### GPU HBM 中的准备

HBM 中不保证完整预放置：

- 有空闲 HBM 时，系统机会性地填充；
- 多个 advisory 竞争时，优先准备不同 session 的较低 Transformer layers；
- 高层 KV 可以等真实请求到达后再加载；
- HBM 紧张时，先驱逐高层 KV，再按 session LRU 处理；
- 活动推理可随时覆盖投机预取 block。

准确结论是：

> SYMPHONY 会把 session 状态导向预选节点，但“目标节点持有完整慢层副本”和“GPU 已完整装入 KV”是两回事；后者明确允许只是部分、按层且可撤销的准备。

### 1.3 SYMPHONY 是否同时存在后台与前台迁移

存在三条不同数据路径。

| 路径 | 触发时间 | 是否位于真实请求路径 | 功能 |
|---|---|---:|---|
| Advisory-triggered staging | 真实请求前 | 理想情况下否 | 从当前 owner 向预选节点提前准备状态 |
| Background persistence | 新 KV 生成期间 | 否 | 后台线程持续把新增 KV 写入低层完整副本 |
| Execution-time layer loading | 请求执行时 | 是，但可与计算重叠 | 从 host 向 GPU 逐层恢复未预取或已驱逐的 KV |

因此“SYMPHONY 已经把 KV 迁移完全移出 critical path”也是过强表述。更准确的是：它尽量用 advisory 隐藏迁移；准备不完整、准备被驱逐或 advisory 太晚时，仍有 execution-time restoration，只是通过逐层异步加载减少可见阻塞。

### 1.4 ReSK 的 proactive 不是 future KV pre-placement

用户的判断正确。[Co-optimizing Request Scheduling and KV Caching for Edge LLM Serving（IEEE IoT-J 2026）](https://doi.org/10.1109/JIOT.2026.3709703)中的 action 是：

- $x_{n,u}^{t}$：将时隙 $t$ 已经到达的请求 $u$ 路由到节点 $n$；
- $y_{n,b}^{t}$：在节点 $n$ 的本地 CPU RAM 保留或淘汰已有 KV block $b$。

其状态转移约束规定，一个 block 在当前节点存在，只能因为：

1. 上一时隙已经在该节点保留；或
2. 上一时隙有真实请求在该节点执行并生成了它。

论文还显式假设跨 edge 节点传 KV 往往慢于重算，因此不允许跨节点获取、迁移或复制 KV。CPU RAM 到本地 GPU 的 fetch 是当前请求确定执行位置后的按需恢复，不是未来请求的提前放置。

需要修正的只有一点：

> 不能说 ReSK 完全没有长期模型。它写出了跨时隙累计成本问题，但实际在线 ReSK 将其分解为逐时隙问题，不知道未来、没有 lookahead/value function，也没有显式 SLO 约束。因此它具有“累计目标的形式”，但在线控制仍然很短视。

### 1.5 Libra 与 future-session KV preparation 无关

[Libra（NSDI 2026）](https://www.usenix.org/conference/nsdi26/presentation/ruan-libra)处理的是已经到达的单个请求。它将请求 token 序列切成有因果依赖的 micro-requests：

$$
r^{\alpha}=[1,s], \qquad r^{\beta}=[s+1,L].
$$

$r^{\beta}$ 必须等待 $r^{\alpha}$ 产生 KV 和生成状态后才能继续，因此它不是把同一请求复制到多个节点并行完成，而是跨请求流水化和实例间负载平衡。其 KV transfer 是同一请求两个连续执行段之间的依赖传输。

所以 Libra：

- 有 SLO-goodput 目标；
- 有跨实例 KV 传输；
- 有 request partitioning 和 SLO-aware batching；
- 没有跨轮 session KV 预放置；
- 没有 advisory；
- 没有未来请求路由。

## 2. 启发式节点选择与系统建模各有什么优劣

### 2.1 SYMPHONY 式启发式的优势

简单策略并不等于低质量。它有四个真实优势：

1. **决策开销低**：advisory 到达后可立即选择较轻载节点；
2. **对模型误差不敏感**：不依赖输出长度、准确到达时间或复杂排队模型；
3. **数据平面可自我保护**：投机 HBM 可被活动请求回收，错误 advisory 主要损失性能而不破坏正确性；
4. **容易部署与解释**：系统贡献集中在 advisory、分层预取和 cooperative memory management。

如果所有候选 worker 同构、链路相近、advisory lead time 足以完成完整准备，简单均衡本来就可能接近最优。此时再写复杂优化器只会增加估计误差和控制开销。

### 2.2 建模的潜在优势

系统建模真正有价值的条件是，不同 action 的机会成本显著不同。advisory $i$ 对 worker $w$ 的准备价值可以抽象为：

$$
V_{i,w}
=
p_i \Delta G_{i,w}
- \lambda_B B_{i,w}
- \lambda_H H_{i,w}
- \lambda_C C_{i,w}
- \lambda_W W_{i,w},
$$

其中：

- $p_i$ 是 advisory 转化为真实请求的概率；
- $\Delta G_{i,w}$ 是准备后增加的 SLO-goodput 或降低的违约风险；
- $B_{i,w}$ 是 backhaul 占用；
- $H_{i,w}$ 是 HBM-byte·time；
- $C_{i,w}$ 是对活动推理、其他 preparation 和 batch 的干扰；
- $W_{i,w}$ 是错误或未使用准备的浪费；
- $\lambda$ 是当前资源 shadow price。

相比“选择较空节点后尽量预取”，这样的控制器至少可以表达：

- 当前不准备；
- 继续 owner-sticky；
- 只准备满足启动所需的连续 layers/blocks；
- 准备一个候选节点；
- 为少数候选节点创建可撤销 option；
- 迁移、继续远程恢复或 exact recomputation；
- 在带宽/HBM 紧张时优先保护真正能跨过 SLO 边界的请求。

### 2.3 建模的风险

建模不是自动改进，至少存在五类失败方式：

1. 输出长度、到达时间和队列预测误差可能超过启发式本身的次优损失；
2. binary SLO indicator 会让小的 latency estimation error 导致 action 跳变；
3. 复杂控制器可能过度拟合 synthetic trace；
4. 多目标 penalty 的权重很容易被调参后选择；
5. 求解器开销和陈旧遥测可能使理论最优动作在执行时已经过期。

因此正确的研究问题不是：

> 如何用优化模型代替 SYMPHONY 的 heuristic？

而是：

> 是否存在一个可测量、不可被简单 route-first 或 transfer-first 策略消除的决策耦合，并且结构化在线算法能在不使用未来 oracle 的情况下稳定收回该 gap？

## 3. 文献版图：哪些组成部分已经存在

### 3.1 正式发表或正式接收工作

| 工作 | Future signal | Pre-arrival KV preparation | Request routing | 显式 SLO | Edge/geo/长期约束 | 对当前 idea 的影响 | 自动存在性核验 |
|---|---:|---:|---:|---:|---:|---|---|
| SYMPHONY, NSDI 2026 | 是 | 跨节点 | 是，单目标 early binding | 调度器无显式目标 | 否 | 覆盖 advisory + staging + routing 主线 | `verified` |
| KVFlow, NeurIPS 2025 | workflow | CPU→GPU | 下一 agent 调度 | 否 | 否 | 覆盖 workflow-aware proactive prefetch | `[UNVERIFIED]`；官方 proceedings 已查 |
| Mooncake, FAST 2025 | 否 | 否；请求到达后取 KV | 是 | 是 | 集群内 | 覆盖 SLO-aware route + KV transfer | `verified` |
| DualMap, ICLR 2026 | 否 | 否 | 是，两个候选 | 是 | 否 | 覆盖 cache/load/SLO 与多候选 current routing | `verified` |
| Randomization Boosts KV Caching..., ICLR 2026 | 否 | 否 | 是 | 否 | 在线累计负载 | 直接否定“首次统一 cache eviction + routing 建模” | `verified` |
| ReSK, IEEE IoT-J 2026 | 否 | 否 | 是 | 否 | edge 累计成本 | 覆盖 edge routing + local KV retention | `[UNVERIFIED]`；DOI 与本地全文已查 |
| CachedAttention, ATC 2024 | 已进入等待队列 | 本地分层预加载 | 否 | 否 | 否 | 说明 queue-aware prefetch 已存在，但请求已经到达 | `[UNVERIFIED]`；USENIX 页面已查 |
| SkyWalker, EuroSys 2026 | 否 | 不迁 KV | 跨地域 | latency/cost | geo network | 覆盖跨地域 KV-locality-aware routing | `[UNVERIFIED]`；EuroSys 页面与 DOI 已查 |
| SuperInfer, MLSys 2026 | 当前活动请求 | 本地 GPU↔CPU | 本地 rotary scheduling | 是 | 单机 Superchip | 覆盖 SLO-aware KV action，但不是 future session | `[UNVERIFIED]`；MLSys 页面已查 |
| ICDCS 2026 CEC KV migration | 待全文确认 | 待全文确认 | 待全文确认 | 待全文确认 | 明确 CEC/geo | 标题直接重合，是当前 hard blocker | `[UNVERIFIED]`；仅官方议程可查 |

对应的一手来源包括：

- [SYMPHONY 官方页面](https://www.usenix.org/conference/nsdi26/presentation/agarwal)；
- [KVFlow NeurIPS 2025 proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/b7971d31a7d5eb0f1eed2f8f6f368195-Abstract-Conference.html)；
- [Mooncake FAST 2025](https://www.usenix.org/conference/fast25/presentation/qin)；
- [CachedAttention ATC 2024](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost)；
- [DualMap ICLR 2026](https://iclr.cc/virtual/2026/poster/10006475)；
- [Randomization Boosts KV Caching... ICLR 2026](https://iclr.cc/virtual/2026/poster/10009529)；
- [Preble ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html)；
- [Libra NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/ruan-libra)；
- [SkyWalker EuroSys 2026](https://2026.eurosys.org/papers.html)；
- [SuperInfer MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/07fd64f9316f40193c6a4d87d8afa011-Abstract-Conference.html)；
- [ICDCS 2026 官方议程](https://icdcs2026.icdcs.org/program/main-technical-sessions/)。

论文存在性自动核验脚本对 23 个候选进行了检查：6 个为 `verified`，17 个因外部服务 transient failure 为 `verify_pending`。上表中自动核验 pending 的正式论文均另外用官方 venue 页面或本地正式 PDF 检查；这不把 pending 当成论文不存在的证据。

### 3.2 2025–2026 并发预印本风险

这些工作不能与正式顶会顶刊等权支撑研究共识，但会影响新颖性：

| 工作 | 已覆盖内容 | 风险 | 自动存在性核验 |
|---|---|---|---|
| [Pythia](https://arxiv.org/abs/2604.25899) | workflow predictability、主动 cache staging/precompute、scheduler、autoscaling | 最直接；显著压缩“利用未来信息准备 KV” | `[UNVERIFIED]`；arXiv 页面已查 |
| [Continuum](https://arxiv.org/abs/2511.02230) | tool duration prediction、KV TTL/pinning、program-level scheduling | 覆盖 future reuse + KV retention + scheduling | `verified` |
| [TokenCake](https://arxiv.org/abs/2510.18586) | tool-call 期间 proactive offload 与 predictive upload | 覆盖预测式 KV 上传/卸载 | `[UNVERIFIED]`；arXiv 页面已查 |
| [Talaria](https://arxiv.org/abs/2607.17181) | session-aware placement/admission、KV locality、soft reservation | 覆盖 likely-return reservation 与 session 连续性 | `[UNVERIFIED]`；arXiv 页面已查 |
| [SMetric](https://arxiv.org/abs/2607.08565) | session-centric routing、local/global KV tier 与负载平衡 | 覆盖长 session 的 locality/load routing | `[UNVERIFIED]`；arXiv 页面已查 |
| [Robust KV Cache Management](https://arxiv.org/abs/2607.16892) | 输出长度不确定性、DRO、prefix placement、routing、SLO | 压缩“不确定性 + SLO + placement/routing”的建模新意 | `verified` |

所以第 4 点应改写为：

> 相同目标、相同方法和系统建模分别都已有先例；暂未看到的是它们在“不确定 future-session advisory + 跨站点可撤销 preparation + arrival-time recourse + 长期 CEC 预算”上的完整统一。

这比“没有系统性建模”更准确，也更难做。

## 4. 当前项目模拟结果给出的反证

项目现有的 KV manager 质量消融已经说明，主动准备并不天然有效。该实验：

- 对每个 `(concurrency, seed)` 重放相同请求 trace；
- 区分 `block_only`、`proactive_only`、`weak/current/strong` preparation；
- 采用 Qwen2-VL-7B-Instruct 和 5 个 seeds；
- 结果位于 [summary.json](../outputs/kv-manager-quality/summary.json)。

相对完全关闭 KV manager：

| 并发 | `proactive_only` TTFT 改善 | `proactive_only` SLO 违约绝对下降 | `strong` TTFT 改善 | `strong` SLO 违约绝对下降 |
|---:|---:|---:|---:|---:|
| 16 | +0.16% | +0.10 个百分点 | +5.86% | +2.30 个百分点 |
| 24 | -0.76% | -0.90 个百分点 | +8.83% | +3.02 个百分点 |
| 36 | -0.86% | -0.12 个百分点 | +12.74% | +5.67 个百分点 |

这里的负值表示策略变差。该结果支持两个互相制约的判断：

1. naive proactive policy 确实会做错，存在 action selection 问题；
2. 收益对 block-level recovery、复制强度和带宽份额高度敏感，不能先验假定某个复杂控制器一定有效。

而且这个模拟对 proactive policy 仍然过于有利。当前实现明确读取真实 trace，知道 session 还剩多少请求：

```python
# The simulation intentionally evaluates the mechanism without
# continuation-prediction error: the generated trace reveals the
# exact number of requests still remaining in this session.
```

对应代码见 [kv_manager.py](../experiment/sim/kv_manager.py) 第 88–105 行。它尚未完整建模：

- advisory false positive、miss 和 calibration；
- 未使用副本的 HBM-byte·time；
- preparation 挤占其他 session 后造成的违约；
- advisory 到请求到达之间的 queue/link drift；
- 单目标 early binding 与多候选 recourse 的独立 oracle gap。

因此现有结果只能说明“策略敏感性真实存在”，不能作为新 paper idea 已经成立的证据。

## 5. 仍可能成立的最小 paper idea

### 5.1 暂定题目

*RoutePrep: SLO-Aware Speculative KV Preparation with Late-Binding Recourse across the Computing Edge Continuum*

### 5.2 一句话 idea

> 在 advisory 阶段不立即承诺唯一执行节点，而把有限 backhaul/HBM 投入到一个或少量可撤销、可继续完成的 exact-KV preparation options；真实请求到达后，根据实现的队列、链路和状态就绪度，只在仍能满足 SLO 的 option、owner 或 recomputation fallback 中选择执行位置。

这里真正可能有新意的不是：

- SLO；
- KV placement；
- 多候选；
- Lyapunov；
- SafeRL；
- CEC 名称。

唯一可能防守的结构是：

> **两阶段 recourse：准备阶段购买未来路由选择权，到达阶段再行使选择权。**

### 5.3 为什么它不同于 SYMPHONY

SYMPHONY 在 advisory 时已经选定目标节点，随后准备 KV，真实请求再被送到该节点：

$$
\text{route first} \rightarrow \text{stage state} \rightarrow \text{execute}.
$$

RoutePrep 的候选结构是：

$$
\text{prepare route options}
\rightarrow
\text{observe realized state}
\rightarrow
\text{select route or fallback}.
$$

如果 advisory 与真实请求之间 queue、link 或 HBM 状态基本不变，则 late binding 没有价值，论文应停止。只有状态漂移足以频繁改变最优 worker，同时完整 top-$k$ 复制又因资源成本过高时，recourse 才可能构成新问题。

## 6. 建议的系统模型

### 6.1 非 Oracle 信息边界

在 advisory 时刻 $\tau_i$，系统只知道：

- session/model/version ID；
- 当前 KV owner、连续有效 prefix 和大小；
- 当前队列、HBM 与逐链路占用；
- advisory 是否成真的校准概率 $\hat p_i$；
- 到达时间区间或条件分布；
- 已完成请求提供的历史统计。

系统不知道：

- 真实剩余 session request 数；
- 当前 advisory 是否一定转化为请求；
- 真实 future input/output length；
- future queue、link state；
- future mobility trajectory。

### 6.2 Preparation action

对 advisory $i$、候选 worker $w$ 和合法的连续 layer/block 截止点 $\ell$，定义：

$$
x_{i,w,\ell} \in \{0,1\},
$$

表示是否为该候选准备到可恢复中间点 $\ell$。另外分配逐时隙带宽：

$$
b_{i,w,e}(t) \ge 0,
$$

其中 $e$ 是 CEC 路径上的链路。

合法 action 只允许：

- `NO-OP`；
- `OWNER-STICKY`；
- exact contiguous block/layer staging；
- 完整 staging；
- 可证明正确的 exact recomputation fallback。

不能把任意“复制百分比”当作有意义的 partial state。部分准备必须对应可被 serving engine 使用的恢复边界。

### 6.3 Arrival-time recourse

真实请求到达后，令：

$$
y_{i,w} \in \{0,1\},
$$

表示是否选择 worker $w$。还可选择 owner 或 recomputation fallback，并满足：

$$
\sum_{w} y_{i,w} + y_{i,\mathrm{owner}} + y_{i,\mathrm{recompute}} = 1.
$$

对 worker $w$，实际 TTFT 近似为：

$$
L_{i,w}
=
T_{i,w}^{\mathrm{ingress}}
+ Q_{i,w}
+ \max\left(0, T_{i,w}^{\mathrm{residual}} - T_{i,w}^{\mathrm{overlap}}\right)
+ T_{i,w}^{\mathrm{prefill}}.
$$

这里只把真正影响首 token 的阶段放入 TTFT；TPOT/TBT 需要单独约束，不能把整个 decode 时间错误地塞入 TTFT。

### 6.4 长期目标与约束

更合适的主目标是 SLO-goodput，而不是 latency 加权和：

$$
\max
\liminf_{T\rightarrow\infty}
\frac{1}{T}
\mathbb{E}
\left[
\sum_{i:t_i\le T}
\omega_i
\mathbf{1}\left\{L_i\le D_i\right\}
\right].
$$

投机资源最好作为显式长期预算，而不只是可任意调权的 penalty：

$$
\limsup_{T\rightarrow\infty}
\frac{1}{T}
\sum_{t=1}^{T}
B_t^{\mathrm{spec}}
\le \bar B,
$$

$$
\limsup_{T\rightarrow\infty}
\frac{1}{T}
\sum_{t=1}^{T}
H_t^{\mathrm{spec}}
\le \bar H,
$$

$$
\limsup_{T\rightarrow\infty}
\frac{1}{T}
\sum_{t=1}^{T}
W_t^{\mathrm{unused}}
\le \bar W.
$$

另外必须满足每条链路瞬时容量、worker HBM、KV 版本兼容和活动推理优先级约束。

## 7. 方法应该怎么做

### 7.1 第一阶段：先做 option generator，不做 RL

对每个候选 worker，基于当前 latency quantile profile 求出最小合法 preparation boundary：

$$
\ell_{i,w}^{*}
=
\min
\left\{
\ell:
\widehat{\Pr}
\left[
L_{i,w}(\ell)\le D_i
\right]
\ge 1-\epsilon_i
\right\}.
$$

如果不存在可行 $\ell$，该 worker 不应进入 option set。这个步骤必须使用 calibration/coverage test 验证，不能只报告预测均方误差。

### 7.2 第二阶段：资源价格下的 option selection

为每个 option 计算“新增 SLO-feasible route 的概率收益”减去资源 shadow price：

$$
I_{i,w}
=
\hat p_i \Delta G_{i,w}
- \sum_e \lambda_e B_{i,w,e}
- \mu_w H_{i,w}
- \nu W_{i,w}.
$$

然后在逐链路和 HBM 约束下选择少量 options。若需要长期预算，可使用 online primal-dual 或 Lyapunov virtual queues 更新 $\lambda_e$、$\mu_w$ 和 $\nu$。

但必须保持准确表述：

> Lyapunov/primal-dual 是求解手段，不是论文的新颖性。论文必须先证明 option-level value 与普通 per-byte utility 不等价。

### 7.3 第三阶段：arrival-time recourse

请求到达后重新估计实际 queue、residual restore 和 SLO slack：

1. 在已准备 candidates 中筛选当前仍 SLO-feasible 的 worker；
2. 若多个可行，选择最低机会成本者；
3. 若均不可行，比较 owner-sticky、on-demand recovery 与 exact recomputation；
4. 未被使用的投机状态进入正常缓存管理，浪费必须记账。

### 7.4 为什么不建议先用 SafeRL

SafeRL 当前不是好起点：

- 缺少真实、安全、覆盖充分的训练 trace；
- combinatorial action space 大且动态变化；
- SLO 和容量约束需要可审计的硬处理；
- offline RL 容易利用日志中不可用的隐含未来信息；
- 与标准 primal-dual/MPC 相比，失败原因更难解释；
- 即使有效，也容易被评价为 optimizer swap。

只有在结构化非学习策略已建立、真实 workload 存在明显非平稳性且 RL 能在相同信息边界下显著改善时，才值得作为后续方法或 baseline。

## 8. 必须通过的实验与 kill tests

### 8.1 先解除两个 blocker

1. 获取并精读 ICDCS 2026 的 *Efficient KV Cache Migration for Geo-Distributed LLM Inference in Collaborative Edge Computing* 全文；
2. 完成 Pythia、KVFlow 和 SYMPHONY 的逐 action、逐约束、逐信息边界 claim chart。

若 ICDCS 论文已经覆盖“预测 future arrival + 多候选 preparation + 到达后选择 + SLO”，本方向应直接停止或大幅换题。

### 8.2 四个独立 oracle gap

| 要验证的价值 | 比较 | Go 条件 | No-Go |
|---|---|---:|---|
| Advisory value | arrival-only 最优路由 vs advisory-aware offline oracle | SLO-goodput 稳定提高至少 10% | 小于 5% |
| Joint value | route-first + optimal staging vs joint routing/staging oracle | 至少 10–15% | 小于 5% |
| Recourse value | 最优单目标 early binding vs 多候选 recourse oracle | 至少 10% 且同 bytes/HBM | 小于 5% |
| Online value | 在线方法 vs 同信息、同预算 offline recourse oracle | 收回至少 70% oracle gap | 仅理想预测有效 |

这些阈值是项目立项门槛，不是统计定理。必须报告 paired confidence interval 和绝对 SLO 完成数。

### 8.3 强基线

1. no preparation；
2. owner-sticky；
3. load-only routing；
4. Preble/LMetric/DualMap-style current-request router；
5. ReSK-style current routing + local retention；
6. SYMPHONY 原始 single-target preparation；
7. SYMPHONY + SLO-aware target selector；
8. Pythia-style prepare-first + affinity routing；
9. full top-$k$ replication；
10. per-byte marginal utility preparation；
11. offline single-target oracle；
12. offline recourse oracle。

所有方法必须共享：

- 相同请求 trace；
- 相同 advisory 信息；
- 相同 HBM、带宽和 background priority；
- 相同 serving profiles；
- 相同失败、取消和副本回收机制。

### 8.4 问题存在性指标

必须先报告：

- advisory precision、recall、校准曲线和 lead-time CDF；
- $T_{\mathrm{lead}} \ge T_{\mathrm{remaining\ transfer}}$ 的请求比例；
- SLO 违约中 queue、state restoration、prefill 和 decode 的归因；
- advisory 到真实请求之间最优 worker 发生变化的比例；
- 单目标准备完成后因 queue/HBM drift 失效的比例；
- preparation bytes、HBM-byte·s、unused bytes、取消率和能耗；
- background preparation 对 foreground TTFT/TBT 的干扰；
- CEC 拓扑扁平化、链路独立化和 worker 同构化后的收益变化。

### 8.5 立即停止条件

以下任一项出现时，不应再用更复杂算法继续包装：

- 最优单目标 preparation 与多候选 recourse 的 gap 小于 5%；
- `SYMPHONY + SLO selector` 与 proposed method 差异小于 3–5%；
- edge 间 KV transfer 在现实链路下几乎总是慢于 exact recomputation；
- 80% 以上违约由 decode 或无法通过状态准备改变的算力排队主导；
- 只有使用真实剩余 request 数、future output length 或 future mobility 才有收益；
- flattened datacenter topology 与 CEC topology 的收益相同；
- generic two-stage stochastic optimizer 在相同开销下已经复现全部收益。

## 9. 对五个问题的最终回答

### 问题 1

SYMPHONY 确实采用简单 request-level load balancing，而没有联合 SLO、KV、链路和排队建模。它不是把全部 KV 无条件塞进空闲 GPU；目标慢层可保有完整副本，HBM 则是容量感知、按层、机会性和可撤销的。系统同时具有请求前 staging、后台低层持久化和请求执行中的逐层恢复。

### 问题 2

确认：ReSK 的 proactive 不是 future-request KV pre-placement。它对已到达请求做路由，对本地已有 KV 做 retain/evict，并明确排除跨 edge KV fetching。

### 问题 3

确认：Libra 是 request-internal causal partitioning + SLO-aware batching；其 KV transfer 服务于同一请求的前后 micro-request，不是跨轮 session KV 准备，也不是 atomic request 选择单一执行节点。

### 问题 4

该判断需要改写。已有：

- 相同目标：Mooncake、DualMap、SuperInfer 等；
- 相同方法：SYMPHONY、KVFlow、CachedAttention 等；
- 系统建模：ReSK、ICLR 2026 的 unified KV eviction/routing 等；
- 长期或不确定性控制：ReSK 的累计目标以及近期 DRO/agent-serving 预印本。

尚未被正式工作明确完整覆盖的是上述要素在不确定 future-session CEC preparation 和 arrival-time recourse 上的交集。

### 问题 5

CEC 中仍有条件性的工作空间，但不能以 “SLO-aware KV placement & routing” 为题直接立项。最窄且仍可能防守的命题是：

> **在不确定 advisory 和共享 CEC backhaul 下，通过可恢复 partial preparation 创建可撤销的 SLO-feasible execution options，并在真实请求到达后进行 late-binding recourse。**

当前只建议执行 claim-chart 和 oracle-gap 实验，不建议先上 SafeRL、Lyapunov 完整系统或论文写作。若 recourse oracle 对单目标 oracle 没有稳定的两位数收益，这个方向应停止。

## 10. 最终决策

**当前状态：Pivot，not Go。**

推荐的下一步只有三件：

1. 拿到 ICDCS 2026 直接相关论文全文；
2. 在现有 simulator 中加入 advisory event、single-target oracle、multi-option recourse oracle、false positive 和 HBM-byte·time 记账；
3. 先画出 `recourse oracle − single-target oracle` 随 lead time、queue drift、带宽和 HBM 的相图。

只有出现连续、现实且预注册的收益区域，才进入完整系统模型与在线算法。否则保留现有 StateWeaver/reactive assembly 方向或重新选择问题，不再增加方法复杂度。
