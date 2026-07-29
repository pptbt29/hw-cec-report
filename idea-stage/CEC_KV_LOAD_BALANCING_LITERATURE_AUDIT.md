# CEC 中 KV-aware LLM 负载均衡：文献与新颖性审计

> **2026-07-27 更新**：SYMPHONY、ReSK、Libra 的机制边界以及 ICLR/NeurIPS/ICDCS 2026 新工作已在[专项审计](./SLO_AWARE_KV_PREPARATION_ROUTING_AUDIT_20260727.md)中重新核对。新结论更保守：宽泛的 “SLO-aware KV placement + routing” 已不具备新颖性，只保留“不确定 advisory 下的可撤销 partial preparation 与 arrival-time recourse”作为待 kill-test 的窄候选；在取得 ICDCS 2026 直接相关全文并验证 oracle gap 前，当前决策为 `Pivot, not Go`。

**检索截止日期**：2026-07-23  
**审计对象**：通过迁移、复制、远程获取或主动准备 KV cache，解除长 session 的状态粘性，利用 CEC 中的远端空闲算力并提高 SLO 达成率。  
**结论口径**：只要已有论文覆盖宽泛机制，就不把“换到 CEC 场景”视为自动恢复新颖性；只有 CEC 约束真正引出已有系统无法处理的新决策结构时，才可能构成独立贡献。

## 1. Executive verdict

结论不是“这个方向没人做”，而是：

> **“通过 KV cache migration/prefetch/placement 改善分布式 LLM 的负载均衡和 SLO”已经被多篇论文直接研究；“在 edge 中联合 KV caching 与 request scheduling”也已有高度重合工作。**

因此，以下主张已经不能成立：

- 首次发现 KV locality 与负载均衡之间的冲突；
- 首次迁移 KV cache 以解除请求粘性；
- 首次联合 KV placement 与 request routing；
- 首次在真实请求到达前预取 KV 以支持负载均衡；
- 首次以 SLO 为目标决定 KV 与请求的跨节点放置；
- 仅仅因为节点属于 access edge、regional edge 和 cloud，便声称问题是新的。

当前仍未发现一篇已取得公开全文的论文同时覆盖：

1. 固定 ingress 的多层异构 CEC；
2. 前台请求、反应式恢复和后台 KV preparation 竞争同一组逐链路 backhaul 容量；
3. 在下一请求到达前、不确定最终目的节点时，为同一 session 创建多个候选执行路径；
4. 只补齐使某条完整路径跨越概率 TTFT-SLO 边界的最小 exact-prefix KV deficit；
5. 在有限 target HBM、版本增长、过期和错误预测下，以新增 SLO-feasible options 或 SLO-goodput 为目标分配资源。

但这只是一个**窄、未证实且高风险的残余交集**。它处于 Mooncake、Llumnix、SYMPHONY、DualMap、Libra、Online Context Caching、edge handover 和经典 CEC 状态迁移的交叉处，不能仅靠重新命名 `routeability` 成为贡献。

**2026-07-23 全文核验更新**：已经取得并精读 IoT-J 2026 的 *Co-optimizing Request Scheduling and KV Caching for Edge LLM Serving*。全文确认它联合优化当前请求路由与本地 KV retention，但明确假设 edge 间 KV fetching 通常慢于 recomputation，因此不允许跨节点获取、迁移或主动复制 KV。它封死了“首次在 edge 联合 scheduling 与 KV caching”的宽泛表述，但没有直接覆盖 shared-backhaul 下的 proactive remote KV preparation、late binding 或 path-qualified SLO option creation。

定性新颖性风险评分：

| 命题 | 当前新颖性 | 判断 |
|---|---:|---|
| KV migration for load balancing/SLO | 1–2/10 | 已直接做过 |
| KV-aware routing + load balancing | 1–2/10 | 已形成密集文献 |
| 到达前 proactive KV preparation | 2–3/10 | SYMPHONY 已高度重合 |
| Edge LLM 中联合 KV caching 与 scheduling | 2–3/10 | 已有 INFOCOM 2025 与 IoT-J 2026 高风险先例 |
| CEC shared-backhaul 下的 late-bound、path-qualified minimum-deficit option creation | 4–5/10 | 尚未找到完整等价先例，但容易被视为已有机制的自然组合 |

这些分数是投稿风险评估，不是数学度量。

## 2. 为什么用户的直觉成立

LLM inference 的确具有传统无状态负载均衡没有的约束：一个运行中或多轮 session 的 KV cache 随 token 数增长，并占用大量显存；若目标节点没有可复用状态，就必须传输 KV、传输 token 后重新 prefill，或放弃迁移。于是目标节点的空闲计算能力与其对某个 session 的实际可用性不是同一件事。

但这一事实已经被系统领域明确识别，并沿着四条路线发展：

1. **Cache-aware routing**：优先把请求送到已有 prefix KV 且负载合适的节点；
2. **Live request/KV migration**：运行中把请求及其状态迁往其他实例；
3. **Distributed KV store 与 remote fetch**：把 KV 从单节点 HBM 解耦出来，由调度器按需获取；
4. **Proactive prefetch/replication**：在真实请求前或热点形成时提前移动 KV，使未来路由不再被原状态位置完全限制。

所以，“KV 重导致负载均衡困难”是一个真实且重要的问题，但已经不是未被发现的问题。

## 3. 最接近工作的具体覆盖

### 3.1 已经直接覆盖宽泛命题的工作

| 工作 | 已完成的核心机制 | 对当前方向的直接影响 |
|---|---|---|
| [Llumnix, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/sun-biao) | 运行时跨实例重调度请求，并以 live migration 搬运请求及其内存状态；目标包括负载均衡、碎片、优先级和 SLO | 直接否定“首次通过 KV/request migration 做动态负载均衡” |
| [Preble, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html) | 分布式调度联合优化 KV reuse 与 computation load balancing | 直接否定“首次联合 KV locality 和负载均衡路由” |
| [MELL, INFOCOM 2025](https://doi.org/10.1109/INFOCOM55648.2025.11044533) | 在多 GPU 间管理和迁移 KV cache，以缓解显存与负载不均 | 直接覆盖 load-driven KV migration |
| [Mooncake, FAST 2025](https://www.usenix.org/conference/fast25/presentation/qin) | 全局 KV cache、跨节点 KV 获取、cache load balancing，以及满足 latency SLO 的全局调度 | 已联合状态位置、传输、计算负载和 SLO |
| [Online Context Caching, INFOCOM 2025](https://doi.org/10.1109/INFOCOM55648.2025.11044599) | 在线联合决定分布式 KV placement 与 request scheduling，显式处理跨时隙耦合和实例负载均衡 | 直接否定“KV placement + request scheduling 尚无人联合建模” |
| [SYMPHONY, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/agarwal) | 从用户交互或 agent 结构取得不可靠 advisory hints，在真实请求前预取 KV，以解除状态位置与 request-level load balancing 的耦合 | 当前对“主动准备 KV 后再负载均衡”最危险的系统先例 |
| [BanaServe, 2026](https://arxiv.org/abs/2510.13223) | global KV store、attention-level KV migration 和动态负载再均衡 | 再次否定“通过细粒度 KV movement 解锁空闲节点”是新机制 |
| [Libra, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/ruan-libra) | 以 SLO-goodput 为目标动态再均衡负载，在任意 token boundary 切分请求，并用 chunked KV transfer 支持跨实例执行 | 覆盖“partial KV transfer + dynamic load balancing + SLO”的宽泛组合 |

这组工作已经足以回答第一个问题：**有，而且不是只有零散一两篇。**

### 3.2 不迁移 KV，但已把 cache affinity、负载和 SLO 联合起来的工作

| 工作 | 核心机制 | 当前方法必须超过的部分 |
|---|---|---|
| [DualMap, ICLR 2026](https://openreview.net/forum?id=zCadrJ32Xn) | 为请求提供两个候选实例，在 cache affinity 与负载间选择，并对热点排队请求再均衡 | 多候选、cache/load trade-off 和 TTFT SLO 已不新 |
| [Randomization Boosts KV Caching, Learning Balances Query Load, ICLR 2026](https://openreview.net/forum?id=R7fv5NWfMm) | 用统一在线模型联合 KV eviction 与 query load balancing | 不能再声称首次统一建模 |
| [LMetric, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/zhang-dingyan) | 用“新增 prefill token 数 × 当前 batch size”的简单分数兼顾 KV locality 与负载 | 复杂控制器必须证明逐链路 CEC 约束确实需要更复杂方法 |
| [QUARTZ, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1888/) | 用保守 TTFT quantile、backlog 和 router-visible signals 做 SLO-aware worker selection 与 admission | 概率 TTFT certificate 本身也不能单独作为新颖点 |
| [SkyWalker, EuroSys 2026](https://doi.org/10.1145/3767295.3769353) | 跨地域 LLM load balancing，考虑网络时延、KV locality、容量与 selective pushing | 固定本地入口、远端空闲算力和网络感知路由本身也已有先例 |

这些工作说明：即使不主动移动 KV，Router 一侧的主要抽象也已经很成熟。新方法不能只把 queue length、cache hit 和 RTT 换成一个更长的加权目标。

### 3.3 Edge/CEC 中的直接相邻工作

| 工作 | 已覆盖内容 | 尚可区分之处 |
|---|---|---|
| [Co-optimizing Request Scheduling and KV Caching for Edge LLM Serving](./PAPER_DEEP_ANALYSIS_COOPT_EDGE_LLM.md), IEEE IoT-J 2026 | 联合当前 request scheduling 与本地 block-level KV retention；KV 只能由本节点此前保留或本地执行请求生成；以 transmission、storage、local CPU-to-GPU fetch、prefill 和 cloud fallback 加权成本为目标 | 明确不做跨节点 KV fetching/migration；不建模 per-link backhaul、后台 preparation、late binding、真实 queue dynamics 或显式 per-request SLO |
| [Low-Latency Edge LLM Handover, arXiv 2026](https://arxiv.org/abs/2603.28018) | 在受限 edge backhaul 上联合选择 KV transfer 与 token prefill，并为多用户调度链路以降低 handover delay | mobility-triggered、两 BS handover；不是固定 ingress 下由负载偏斜触发的多层卸载 |
| [Serving Long-Context LLMs at the Mobile Edge, ToN 2026](https://arxiv.org/abs/2501.14205) | 联合模型 caching 与 inference offloading，面向 mobile edge 的 long-context serving | 主要管理模型与卸载，不等价于 session-specific KV option creation |
| [Follow Me at the Edge](https://arxiv.org/abs/1809.05239) 及传统 MEC 状态迁移 | 已研究服务状态迁移、移动预测、长期迁移预算和 Lyapunov 在线控制 | 把传统 state 改名为 KV、再加入 Lyapunov penalty 不构成独立贡献 |

该全文把两类问题清楚地区分开：

- **ReSK 已解决**：当前请求应路由到哪里，以及各节点本地生成的 prefix KV 应保留还是淘汰；
- **仍未解决**：未来请求到达前，是否应经受限共享 backhaul 把缺失 session KV 主动准备到一个或多个远端候选节点。

后者是真实差异，但仍可能被审稿人视为 ReSK、SYMPHONY、Mooncake 和经典 CEC 链路调度的自然组合。

## 4. 已解决到什么程度

### 4.1 已基本饱和的部分

以下问题已经有成熟或快速拥挤的文献：

- cache affinity 与 load balancing 的冲突；
- cache-aware worker selection；
- 运行中 request/KV live migration；
- 远程 KV fetch 与全局 KV pool；
- 热 prefix 的迁移或复制；
- 到达前的 KV prefetch；
- 多候选实例与 SLO-aware routing；
- 分布式 KV placement 与 request scheduling 的联合在线优化；
- partial/chunked KV transfer 支持的动态负载再均衡；
- 跨地域网络时延、容量与 KV locality 的联合路由。

如果论文仍以这些内容作为主要贡献，审稿人可以分别用 Preble、Llumnix、Mooncake、SYMPHONY、DualMap、Libra 和 SkyWalker 逐项否定。

### 4.2 可能尚未完全覆盖的最窄问题

仍有机会的不是“KV Manager + Router”，而是：

> **在共享逐链路 backhaul 和有限 HBM 的 CEC 中，在未来目标节点尚未确定时，只完成能让完整 worker–path 跨过概率 TTFT-SLO 边界的最小 exact-prefix KV deficit，从而创造可在请求到达后再选择的执行选项。**

这一表述包含四个不能被拆掉的条件：

1. **Late binding**  
   SYMPHONY 式方法通常在 advisory 时选择目标节点；这里准备的是多个 option，真实请求到达后再根据实际 queue 和 link state 选路。

2. **Path-qualified readiness**  
   readiness 不是“目标 GPU 有多少 KV”，而是完整 ingress-to-worker、state-source-to-worker、return path、worker queue、compute profile 和 HBM 共同决定的 SLO 可行性。

3. **SLO-boundary completion**  
   不是持续最大化每字节收益或盲目复制全部状态，而是优先完成能够新增一个 SLO-feasible option 的 deficit。

4. **Shared-link coupling**  
   多个 preparation jobs 的路径可能共享中间 bottleneck；相同端到端带宽但不同链路重叠应导致不同决策。若一个标量 transfer-time penalty 已足够，CEC-specific 方法就没有必要。

## 5. 为什么这个窄缺口仍然可能不够新

从第一性原理看，`minimum deficit` 只是对 latency model 求一个达到 deadline 的最小状态量。这个阈值本身并不复杂，也很可能被评价为显然。

真正的研究难点必须来自以下耦合，而不是公式命名：

- 一份 KV preparation 对多个未来路由动作具有 option value；
- 多路径共享链路，使 job value 不是独立可加；
- session KV 在准备期间继续增长，需要 prefix/version consistency；
- background transfer 会干扰 foreground request 和 reactive recovery；
- 未来到达时间、输入长度、输出长度、queue 和 link state 均未知；
- edge backhaul 较慢时，传 KV 可能始终劣于传 token 后重算；
- 多准备目的地会增加浪费，late binding 只有在状态变化足够大时才有净收益。

如果算法最后只是：

1. 估计每个候选节点的 TTFT；
2. 把未达到 SLO 的差额换算成 KV bytes；
3. 采用 knapsack、Lyapunov 或 SafeRL 分配带宽；

那么审稿人很可能认为这是现有 KV scheduler 和经典 CEC resource allocation 的直接组合，而不是新的系统机制。

## 6. 是否应该现在另写完整系统模型和方法文档

**不应该。**

用户给出的条件是“如果这个方向没有工作做过，再另写系统建模和方法”。现在的查新已经明确证明该前提不成立。若仍继续写一份以“无人做过”为出发点的完整方法文档，会把时间投入在一个已经错误的 novelty premise 上。

本轮因此只生成这份文献与新颖性审计，没有生成第二份完整系统建模文档。继续建模需要先通过以下门槛。

## 7. 继续建模前的 Go/No-Go gates

### Gate 1：取得并拆解最危险全文

IoT-J 2026 的全文 claim chart 已完成，结果见 [专项精读](./PAPER_DEEP_ANALYSIS_COOPT_EDGE_LLM.md)。还需完整阅读并逐项编码：

1. SYMPHONY；
2. Mooncake；
3. Libra；
4. Online Context Caching；
5. SkyWalker。

必须确认它们是否处理：

- 显式跨节点 KV transfer；
- per-link shared backhaul；
- foreground/background contention；
- target HBM；
- non-Oracle arrival/length/load uncertainty；
- multiple destination options；
- exact-prefix/version semantics；
- SLO-boundary completion。

任何一篇若同时覆盖大部分条件，都应重新评估甚至停止该方向。

### Gate 2：先做 kill-test simulation

在完整建模前，先用最小离散事件模拟比较：

1. state-sticky；
2. load-only + reactive transfer/recompute；
3. Preble/LMetric-style cache-aware routing；
4. SYMPHONY-style single-target early-binding prefetch；
5. DualMap-style two-candidate routing；
6. full replication；
7. per-byte marginal-utility prefetch；
8. proposed minimum-deficit late binding；
9. oracle。

所有方法共享同一 data plane、offered trace、bandwidth、HBM-time、foreground priority 和 admission rule。

必须扫描：

- KV transfer time 与 token recomputation time 的比值；
- advisory lead time；
- advisory 到真实请求之间 queue/link state 的变化；
- 路径重叠和共享瓶颈强度；
- HBM 容量；
- context size 与 KV growth；
- continuation、input length 和 output length prediction error；
- 候选节点数量与异构程度。

### Gate 3：需要出现连续而现实的收益区域

只有同时观察到以下现象才应 Go：

- 全量复制因 shared backhaul/HBM 浪费而劣化；
- 单目标 early binding 因未来 queue/link 变化而频繁选错；
- 逐字节 marginal utility 因摊薄资源而不能及时完成有价值 option；
- minimum-deficit completion 在多个连续、现实参数区间提高 SLO-goodput；
- flatten topology、dedicate backhaul 或 homogenize workers 后，相对收益显著缩小；
- 计算全部 foreground interference 和错误准备后，收益仍存在。

以下任一情况应 No-Go：

- 只在“预测极差、同时带宽和 HBM 又大量空闲”的窄点获益；
- scalar network penalty 与 per-link model 的结果几乎相同；
- transfer KV 在目标 CEC 带宽下几乎总是劣于 token recomputation；
- 强 marginal-utility baseline 与 proposed method 位于同一 Pareto frontier；
- 移除 CEC 层级、异构和共享链路后，算法收益基本不变；
- 收益主要来自更多 drop、牺牲长 session 或额外资源。

## 8. 对论文定位的最终建议

当前不应把论文定位成：

> KV cache migration for load balancing in CEC.

如果 gates 通过，才可以把它收缩为：

> Late-bound creation of path-qualified execution options under shared CEC backhaul.

即便如此，主贡献也不能是 `routeability` 名词、threshold 公式、SafeRL 或 Lyapunov。可发表性必须来自一个由 CEC 逐链路耦合和 session-version semantics 导出的新控制结构，以及对最强组合基线的可重复端到端优势。

## 9. 文献核验说明

- 正式 venue 论文优先使用 USENIX、ICLR/OpenReview、ACL Anthology、IEEE DOI/DBLP 等一手或权威索引。
- arXiv-only 工作只作为预印本证据，不与正式论文等量表述。
- IEEE IoT-J 2026 作者版本全文已由用户提供并完成逐页精读；其不包含跨节点 KV fetching/migration、per-link backhaul、后台 preparation、late binding 或显式 SLO qualification。
- ARIS 论文存在性核验脚本已对候选 arXiv 列表执行，但服务返回 transient failure / `verify_pending`；本报告没有用该失败结果证明论文存在或不存在。
- 本报告只能支持“截至检索日期未发现完整等价公开先例”，不能证明绝对不存在未公开、刚接收或索引尚未更新的工作。

## 10. 独立对抗审稿

在主报告完成后，另一个不继承本轮讨论、只读取 claim dossier 的独立审稿代理进行了反向检查。其结论为：

> **No-Go for full modeling；Conditional Go for novelty kill-test only。**

它特别指出：

1. “未发现一篇论文同时列出所有约束”不是 non-obviousness 证据；
2. minimum-deficit 很可能只是对 SLO threshold 的显然逆解；
3. 以“新增 feasible option”为指标再证明集中完成优于摊薄资源，可能构成循环论证；
4. late binding 必须在相同 bytes、HBM-time、data plane 和预测信息下显示净收益；
5. 逐链路 CEC 模型必须用反例或结构性结果证明不可约化为标量 network penalty；
6. 背景 preparation 自身引起的拥塞必须完整计入。

该审稿意见与本报告的最终决策一致：**现在停止扩展完整方法，只允许继续剩余高风险全文 claim chart、结构性反例和低成本 kill-test。**
