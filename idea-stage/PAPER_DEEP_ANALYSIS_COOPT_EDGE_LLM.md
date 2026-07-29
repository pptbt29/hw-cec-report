# 论文精读：Co-optimizing Request Scheduling and KV Caching for Edge LLM Serving

**论文**：Xishuo Li, Wei Jiao, Junyi He, Shan Zhang, *Co-optimizing Request Scheduling and KV Caching for Edge LLM Serving*  
**状态**：IEEE Internet of Things Journal 已接收作者版本，DOI `10.1109/JIOT.2026.3709703`  
**本地全文**：[Co-optimizing_Request_Scheduling_and_KV_Caching_for_Edge_LLM_Serving.pdf](/Users/dazzysy/Documents/readings/edge3/Co-optimizing_Request_Scheduling_and_KV_Caching_for_Edge_LLM_Serving.pdf)  
**精读日期**：2026-07-23

## 固定开场分析

## 一、快速判断

这篇论文研究的是：

> **在多个资源受限的 edge LLM 节点之间，联合决定当前请求送到哪里，以及各节点应保留哪些本地生成的 prefix KV blocks，从而降低传输、CPU cache storage、CPU-to-GPU loading、prefill computation 和 cloud fallback 的加权成本。**

它属于 **online request scheduling + local KV retention co-optimization**，不是 **cross-node KV migration / proactive remote KV preparation**。

具体决策对象是：

```text
当前时隙到达的 request -> edge node 或 cloud
本节点已经存在或由本节点上一时隙请求生成的 KV block -> 保留或淘汰
```

最关键的判断是：

> 这篇论文已经占据了“Edge LLM 中联合 request scheduling 与 KV caching”的宽泛问题，但没有研究“在请求到达前将 session KV 主动移动到其他节点，以创造新的 SLO-feasible offload option”。

因此，它不会直接否定 RoutableKV 最窄的跨节点 preparation 机制，但会否定以下宽泛表述：

- 首次在 edge 联合 request scheduling 和 KV caching；
- 首次用 block-level KV state 改善 edge load balancing；
- 首次考虑请求放置与未来 KV locality 的时间耦合；
- 首次在 edge 上建立多时隙在线 KV/request 联合优化。

---

## 二、它在什么场景下解决什么问题？

### 2.1 系统场景

论文考虑离散时隙运行的多节点 edge LLM serving 系统：

- 系统包含多个异构 edge nodes 和一个远端 cloud fallback；
- 每个请求从某个 origin edge node 到达；
- 当前请求可以被调度到任意 edge node，资源不足时发送到 cloud；
- 每个 edge node 运行兼容的 LLM serving process；
- GPU HBM 用于模型、激活和当前请求的 active KV；
- CPU RAM 用于保留已经生成但当前不活跃的 prefix KV blocks；
- 请求的 prompt 被切分为固定大小 KV blocks，实验中每 block 为 16 tokens；
- KV prefix 具有严格依赖关系：后续 block 可复用的前提是此前所有 prefix blocks 均存在；
- controller 每 5 秒读取请求、KV Radix Tree、GPU 占用和系统负载，并产生下一轮调度与保留决策。

论文实验并不是真正由多个地域 edge servers 构成的 CEC testbed。它使用一台 Ubuntu 22.04 服务器上的四张 RTX 4090：

- 三张 GPU 模拟三个 edge nodes；
- 一张 GPU 模拟 cloud；
- 每张 GPU 运行独立 SGLang process；
- edge-to-edge latency 设置为 30 ms；
- edge-to-cloud latency 设置为 300 ms。

也就是说，实验验证的是**单机多进程上的逻辑 edge 拓扑**，不是具有真实带宽瓶颈、排队和链路竞争的多站点网络。

### 2.2 请求与 KV 状态

请求 $u$ 的输入长度为 $L_u$，按每 block 包含 $A$ 个 tokens 切成 $K_u$ 个 blocks。一个 KV block 不仅由 token 内容决定，还由此前完整上下文决定，因此相同 token 出现在不同 prefix 后不能直接复用。

论文只允许一个 KV block 通过两种方式出现在节点 $n$：

1. 它在上一时隙已经保存在节点 $n$；
2. 上一时隙有一个包含该 block 的请求在节点 $n$ 上执行，从而本地生成该 KV。

论文在 PDF 第 4 页明确说明：即使相同 KV 位于其他 edge node 或 cloud，跨节点取回在其假设下通常比重算更慢，因此**不允许把其他节点的 KV 直接迁移过来**。

这是理解该论文与 RoutableKV 差异的决定性事实。

### 2.3 决策变量

| 符号 | 含义 |
|---|---|
| $x_{n,u}^{t}$ | 时隙 $t$ 是否把请求 $u$ 调度到节点 $n$ |
| $y_{n,b}^{t}$ | 时隙 $t$ 开始时，节点 $n$ 的 CPU RAM 是否保留 KV block $b$ |
| $z_{n,u,b}^{t}$ | 请求 $u$ 是否在节点 $n$ 执行并命中本地 block $b$；用于线性化 $x_{n,u}^{t}y_{n,b}^{t}$ |
| $C_n^{GPU}$ | 节点 $n$ 的 GPU HBM 容量 |
| $C_n^{RAM}$ | 节点 $n$ 用于 inactive KV 的 CPU RAM 容量 |
| $S_{n,t}$ | 时隙 $t$ 节点 $n$ 已被模型、激活和 active KV 占用的 GPU memory |
| $P_b$ | block $b$ 依赖的全部 prefix blocks |

### 2.4 目标函数

论文把总目标写成五类成本的加权和：

$$
\min
\sum_t
\left(
w_1 C_t^{trans}
+
w_2 C_t^{stor}
+
w_3 C_t^{fetch}
+
w_4 C_t^{pre}
+
w_5 C_t^{clu}
\right).
$$

其中：

- $C_t^{trans}$：把 prompt tokens 从 origin node 转发到执行节点的成本；
- $C_t^{stor}$：在 edge CPU RAM 中保留 KV blocks 的成本；
- $C_t^{fetch}$：把命中的 KV 从本地 CPU RAM 加载到本地 GPU HBM 的成本；
- $C_t^{pre}$：没有本地缓存覆盖的 prompt 部分所需的 prefill computation cost；
- $C_t^{clu}$：edge 无法处理时使用 cloud 的成本。

注意：这里的 transmission 传输的是**请求 tokens**，不是跨节点 KV cache。

### 2.5 关键约束

- 每个请求必须选择一个 edge node 或 cloud；
- KV block 只能保留在此前已经拥有或刚刚生成它的节点；
- 后续 block 被保留时，其依赖的所有 prefix blocks 也必须保留；
- 每个节点的 CPU KV storage 不能超过 $C_n^{RAM}$；
- 当前时隙被调度到节点的请求所需 GPU memory 不能超过剩余 HBM；
- GPU 已占用量 $S_{n,t}$ 被当作外生观测值，不由优化器预测或控制。

论文没有显式约束：

- per-link bandwidth；
- network queue；
- 多条流共享 backhaul；
- background KV transfer；
- 请求等待队列演化；
- per-request TTFT deadline；
- SLO violation probability；
- 多候选节点的预准备；
- growing session KV 的跨节点版本一致性。

---

## 三、其他相关工作做了什么？

| 方向 | 相关工作能力 | ReSK 的位置 |
|---|---|---|
| Prefix-aware routing | Preble 等根据已有 prefix locality 与节点负载路由当前请求 | ReSK 进一步决定哪些本地 KV blocks 跨时隙保留 |
| Distributed context caching | Online Context Caching 联合 cache placement 与 request scheduling | ReSK 去掉静态 prefix catalog 假设，使用实际请求产生的 arbitrary blocks |
| KV hierarchy | CachedAttention、LMCache 管理 GPU/CPU/storage 间 KV | ReSK 主要管理每个 edge node 的 CPU RAM retention |
| Cross-node KV movement | Llumnix、Mooncake、SYMPHONY、MELL 等迁移、远程获取或预取 KV | ReSK 不执行此类动作，甚至在模型中主动排除 cross-node fetching |
| Edge service caching | 传统 MEC 联合 service/content caching 与 workload scheduling | ReSK 将对象换成具有 prefix dependency、只能由推理生成的 KV blocks |

论文真实的新意不是发明 KV-aware routing，而是：

> 在 edge LLM 中，将当前请求放置、由请求生成的 KV state、未来本地 prefix reuse 和本地 KV retention 放进同一个 block-level、多时隙在线模型。

---

## 四、这是新问题，还是以前没有解决好？

不是完全新问题，而是已有 scheduling、prefix caching 和 edge resource allocation 在一个更接近现代 LLM runtime 的状态模型下重新联合。

| 子问题 | 此前状态 | 论文新增内容 |
|---|---|---|
| KV-aware request routing | 已有 Preble 等工作 | 加入本地 cache retention decision |
| Cache placement + routing | 已有 Online Context Caching | 去掉预定义静态 cache catalog，使用实际 token-prefix blocks |
| Edge request scheduling | 已有大量 MEC 工作 | 引入 LLM KV prefix dependency 和 prefill reuse |
| Cross-node KV transfer | 已有 Llumnix、Mooncake、MELL 等 | 本文未做 |
| Proactive remote preparation | 已有 SYMPHONY 等 | 本文未做 |
| Shared-backhaul scheduling | Edge handover 等开始研究 | 本文未做 |

所以，该论文是一个有明确场景价值的组合创新，但不是对所有 edge KV scheduling 问题的完整覆盖。

---

## 五、作者的工作动机

### 5.1 Reactive eviction 的时间短视

LRU 只依据最近访问历史淘汰 KV，可能在 block 即将再次被请求前将其删除。由于 KV 被删除后需要重新 prefill，当前 eviction 会影响未来请求成本。

### 5.2 Cache-affinity routing 的空间短视

总是把请求送往 prefix match 最大的节点，可能不断强化某些节点的 KV 热度与请求负载，造成资源争用；而请求在何处执行又决定未来 KV 在何处产生。

### 5.3 作者的核心设计原则

```text
把“当前 request placement”与“下一时隙仍保留哪些本地 KV blocks”联合决定，
用当前真实状态做在线单时隙优化，并通过历史状态把决策串联起来。
```

---

## 六、作者具体如何解决？

整体流程为：

```text
收集当前请求、Radix Tree、GPU memory 与负载
-> 构造当前时隙 LP relaxation
-> 线性 randomized rounding
-> 得到 request placement 与 KV retention
-> repair 容量违规
-> 调度当前请求并锁定/解锁 Radix Tree blocks
-> 下一时隙重复
```

### 6.1 从联合 INLP 到单时隙 ILP

原始问题 P1 同时包含：

- binary request assignment；
- binary KV retention；
- 跨时隙 cache feasibility；
- request placement 与 local cache hit 的乘积项。

作者固定上一时隙决策，把当前时隙问题写成 P2；随后引入 $z_{n,u,b}^{t}$ 线性化乘积，得到 ILP。

### 6.2 LP relaxation 与 randomized rounding

作者先把 binary variables 放松到 $[0,1]$，用 OR-Tools 求 fractional optimum。然后构造若干整数候选解，并按从 fractional values 导出的概率随机选择。

理论上：

- request assignment、cache lineage 和 prefix dependency 在每个 realization 中严格满足；
- CPU/GPU capacity 等约束最初只在期望或高概率 constant-factor violation 意义下成立；
- 期望目标等于 fractional optimum；
- 作者给出目标和容量违反的概率界。

### 6.3 Multi-slot ReSK

ReSK 并不预测未来请求。它在每个时隙：

1. 读取上一时隙的 request placement 和 KV retention；
2. 只求解当前时隙；
3. 实施当前调度和 eviction；
4. 将结果作为下一时隙输入。

它之所以被称为 multi-slot，是因为 cache state 通过 retention 约束跨时隙传递，而不是因为求解了未来窗口或使用了 demand prediction。

### 6.4 Feasibility repair

由于 randomized rounding 可能超出 CPU/GPU capacity，作者增加 post-processing：

- CPU RAM 超限时，按 marginal contribution 淘汰低收益 blocks；
- GPU memory 超限时，把高成本请求重新调度到有空间的邻近节点；
- 没有 edge 可容纳时，发送到 cloud。

---

## 七、解决效果如何？

### 7.1 实验设置

| 项目 | 设置 |
|---|---|
| 硬件 | 单机 4 × RTX 4090，250 GB CPU RAM |
| 逻辑拓扑 | 3 个 edge SGLang processes + 1 个 cloud process |
| 网络 | edge-edge 30 ms；edge-cloud 300 ms |
| 模型 | LLaMA3-8B、Qwen3-8B |
| 数据 | HotpotQA、OASST1 |
| Block size | 16 tokens |
| Scheduling interval | 5 seconds |
| Baselines | Round Robin、Cache Affinity + LRU、Lowest GPU Utilization + LRU |

### 7.2 主要结果

- Qwen3-8B + HotpotQA：平均 TTFT 从 RR 的 0.477 s 降至 0.219 s，下降 54.02%；
- Qwen3-8B + OASST1：平均 TTFT 最多下降 40.90%；
- LLaMA3-8B + HotpotQA：平均 TTFT 下降 47.79%；
- LLaMA3-8B + OASST1：平均 TTFT 下降 33.48%；
- 控制器处理 8、24、40 个 requests 时，运行时间分别约为 10.91、54.56、97.23 ms；
- 中高并发下，ReSK 的 failure rate 低于三个简单 baselines。

这些结果能够支持：

> 联合当前请求放置和本地 KV retention，优于只看 load、只看 cache affinity 或 round robin。

它们不能支持：

> ReSK 优于当前最先进的 distributed LLM schedulers，或者已解决真实 CEC shared-backhaul 下的 KV movement。

---

## 八、需要谨慎看待的地方

### 8.1 它没有做 KV migration

这是与 RoutableKV 最重要的边界。ReSK 的 cache state 只能在本节点被保留，或由调度到本节点的真实请求顺便生成。它不能在 request gap 中主动把 session KV 迁到远端节点。

### 8.2 “Edge network”只被压缩为标量成本

论文的 transmission cost 是 prompt size 乘以 origin-target unit cost。它没有：

- per-link capacity；
- shared bottleneck；
- network congestion；
- multiple transfer paths；
- foreground/background competition；
- KV bytes transfer。

所以它无法回答“有限 backhaul 应优先准备哪些 remote KV states”。

### 8.3 Load balancing 模型缺少显式 queue dynamics

论文讨论 queue、GPU utilization 和 resource contention，但 P1/P2 中没有明确的 queue evolution、remaining work、node service rate 或 waiting-time constraint。负载的主要硬限制是 GPU memory admission，TTFT 则通过传输、local fetch 和 prefill cost 近似。

因此，它解决的是 cost-aware placement，不是严格 queueing/SLO load balancing。

### 8.4 没有真正的 SLO constraint

论文说明模型“可以扩展”加入 per-request deadline，但实际 P1 没有 deadline、tail probability 或 SLO constraint。实验报告平均 TTFT，而不是 offered-load SLO-goodput 或 P99 SLO attainment。

### 8.5 平均 TTFT 存在 survivor bias

作者在 PDF 第 13 页明确承认：高并发时，长 prompt requests 更容易失败，留下的短请求使平均 TTFT 反而低于中等并发。

这意味着平均 TTFT 不能单独衡量系统是否改善。至少需要同时固定 offered load 并报告：

- 所有 offered requests 的 SLO-goodput；
- failure/drop ratio；
- prompt-length 分组成功率；
- P95/P99 TTFT；
- fairness。

这反而支持 RoutableKV one-page 中采用 offered-load SLO-goodput 的设计。

### 8.6 Baselines 不足以支撑强 SOTA 结论

实验只比较 RR、cache affinity 和 lowest-utilization 三个简单策略。论文正文已经引用 Preble 和 Online Context Caching，却没有实现或比较它们；也没有比较 Mooncake、Llumnix、MELL、SYMPHONY 或强组合 baseline。

因此，“优于现有方法”只能理解为优于所选简单 baselines，不能理解为优于 SOTA distributed LLM serving。

### 8.7 理论保证的实际意义有限

需要区分三个层次：

1. 原 randomized solution 的 CPU/GPU constraints 只在期望或 constant-factor violation 意义下成立；
2. Section VII 的 repair 才保证实际可行；
3. 论文没有重新证明 repair 后仍保持此前的 expected objective equality 或 approximation bound。

此外，多时隙所谓 constant factor 包含 instance-dependent 的 relaxed/full optimum ratio，其数值可能很松。理论结果可以说明算法具有可分析结构，但不能直接解释为严格接近真实系统全局最优。

### 8.8 测试规模与“large-scale edge”表述不匹配

系统只有四张同机 RTX 4090，最多展示每 5 秒 40 个 requests。没有：

- 多机真实链路；
- edge 数量扩展；
- network bandwidth sweep；
- CPU RAM/HBM sweep；
- 节点计算异构 sweep；
- slot length sensitivity；
- objective weights sensitivity；
- foreground interference；
- 多 tenant fairness。

因此，论文证明了一个小规模 prototype 上的收益，但没有充分验证 general CEC scalability。

### 8.9 Prefill cost 是简化 proxy

其 computation model 用 block 数和 prefix hit 近似 FFN/attention cost，没有建模 continuous batching、chunked prefill、kernel efficiency、decode interference 和真实 queueing。这适合作为优化 surrogate，但不能直接等同于端到端 TTFT。

---

## 九、它与 RoutableKV 的逐项关系

| 维度 | ReSK | RoutableKV 当前设想 | 是否直接重合 |
|---|---|---|---|
| 场景 | 多 edge + cloud fallback | access/regional/cloud CEC | 部分重合 |
| 当前请求路由 | 是 | 是 | 重合 |
| Block-level prefix | 是 | 是 | 重合 |
| 本地 KV retention/eviction | 是 | 需要作为基础能力 | 重合 |
| 跨节点 KV migration | 明确不做 | 核心动作 | 不重合 |
| 请求到达前 preparation | 不做 | 核心动作 | 不重合 |
| 多个 prepared destinations | 不做 | late-binding options | 不重合 |
| Shared per-link backhaul | 不建模 | 核心 CEC 约束 | 不重合 |
| Foreground/background contention | 不建模 | 必须计入 | 不重合 |
| 显式 queue/service rate | 不完整 | 必须建模 | 不重合 |
| Per-request SLO qualification | 仅称可扩展 | 核心定义 | 不重合 |
| 主目标 | weighted average cost | offered-load SLO-goodput | 不重合 |
| Non-Oracle future | 无未来知识，但只响应当前请求 | 概率 session continuation 与资格有效期 | 部分重合 |
| Growing session version | 本地 prefix lineage | 跨节点增量版本一致性 | 部分重合 |

最准确的研究边界是：

> ReSK 决定“当前请求送到哪里，以及由真实请求在各节点产生的 KV 应保留多久”；RoutableKV 拟决定“在未来请求尚未到达、最终节点尚未确定时，应主动把哪些缺失 KV 状态准备到哪些候选路径上”。

这个差异是真实的，但仍不自动等于新颖。审稿人可以把 RoutableKV 看成：

```text
ReSK-style request scheduling and retention
+ SYMPHONY-style proactive prefetch
+ Llumnix/Mooncake-style cross-node KV movement
+ CEC per-link resource allocation
```

因此必须证明 shared-link coupling、late binding 和 SLO-boundary completion 共同产生不可由上述模块直接组合得到的新策略结构。

---

## 十、对当前 paper idea 的影响

### 10.1 被这篇论文彻底封死的表述

- “Edge LLM 尚未联合考虑 request scheduling 和 KV cache placement。”
- “现有 edge 方法只做模型 caching 或无状态 offloading。”
- “首次在 edge 上考虑 block-level KV reuse 与 load balancing。”
- “首次建立多时隙 request/KV 联合在线优化。”

### 10.2 仍然可以保留的研究问题

- cross-node session KV preparation，而不是 local retention；
- background KV bytes 与 foreground requests 共享逐链路 bottleneck；
- transfer、recompute、no-preparation 三种动作的联合选择；
- 多候选目的节点的 late binding；
- path-specific probabilistic TTFT qualification；
- growing/versioned KV 的增量准备与过期；
- offered-load SLO-goodput，而不是只优化平均成本或平均 TTFT。

### 10.3 更新后的 Go/No-Go

这篇全文解决了最大文献 pending，但没有让 RoutableKV 自动变成 Go。

当前建议仍然是：

- **No-Go for full system implementation**；
- **Go for a small falsification simulation and a structural counterexample**。

在进入完整建模前，至少要证明：

1. 若只采用 ReSK 式“路由当前请求 + 保留本地 KV”，CEC load skew 下仍存在显著 state-induced stranded capacity；
2. 主动跨节点准备相对 reactive recompute/transfer 具有现实可行的时间窗口；
3. per-link overlap 会改变 preparation ordering，不能被 origin-target scalar cost 替代；
4. multi-target late binding 在相同 bytes 和 HBM-time 下优于 single-target early binding；
5. minimum-deficit completion 的收益不是由人为定义“option count”指标循环制造，而能提高真实 offered-load SLO-goodput。

## 十一、最终结论

这篇论文面向多 edge node LLM serving，解决当前请求调度与本地 prefix KV retention 的联合在线优化；其主要贡献是 block-level spatiotemporal model、LP randomized rounding 和 SGLang prototype。它证明了简单 cache affinity 或 load-only routing 会遗漏 request placement 对未来本地 KV locality 的影响。

但它不移动 KV，不调度 backhaul，不提前准备替代节点，也没有真实 per-request SLO 或 queueing model。其平均 TTFT 结果还受到失败请求 survivor bias、简单 baselines 和单机逻辑 edge testbed 的限制。

用于论文对比表的概括：

> **ReSK 联合调度当前 edge LLM 请求并保留本地生成的 prefix KV blocks，以降低传输、存储和 prefill 成本；它覆盖 edge request/KV co-optimization 的宽泛框架，但显式排除跨节点 KV fetching，因此没有解决 shared-backhaul 下的 proactive remote KV preparation、late binding 和 SLO-qualified capacity activation。**
