# Novelty-check dossier：CEC 中通过 KV 状态移动缓解 LLM 负载不均

**检索截止日期**：2026-07-23  
**被审计想法**：`idea-stage/ONE_PAGE_IDEA.md` 中的 RoutableKV  
**审计口径**：先判断宽泛方向是否已有先例，再判断 CEC-specific 交集是否仍有可辩护的新问题。未取得全文的论文不用于证明“没有覆盖”。

## 1. 待核查核心主张

### C1：问题存在

长 session 的 KV state 会把请求粘在既有节点；在局部过载而远端仍有空闲算力时，状态恢复成本可能使远端算力无法在 SLO 内被使用。

检索式：

1. `"LLM serving" KV cache load imbalance migration SLO`
2. `"stateful LLM" request migration load balancing KV`
3. `"long session" KV cache remote spare capacity inference`
4. 近年限定：`2024 2025 2026 LLM KV cache load balancing migration`

### C2：宽泛机制

迁移、复制、远程获取或预取 KV，可以解除 cache locality 与 load balancing 的耦合，并改善 TTFT、尾延迟或 SLO-goodput。

检索式：

1. `"KV cache migration" load balancing LLM inference`
2. `"proactive KV prefetch" request-level load balancing`
3. `"distributed KV cache" SLO-aware request scheduling`
4. 近年限定：`2025 2026 KV migration load-aware LLM serving`

### C3：CEC-specific 机制

在固定 ingress、多层异构 CEC 拓扑和共享逐链路 backhaul 下，为多个 session–worker–path 组合主动准备最小 KV deficit，使原本不满足 TTFT SLO 的候选路径变为可用，并把真实请求的最终目标选择延迟到请求到达后。

检索式：

1. `"edge LLM" KV cache request scheduling backhaul`
2. `"cloud edge continuum" LLM KV migration load balancing`
3. `"edge LLM serving" proactive KV cache placement SLO`
4. `"multi-region LLM serving" KV locality load balancing network`
5. 近年限定：`2026 edge LLM KV caching scheduling TTFT`

### C4：结构性算法主张

在 per-link bandwidth、target HBM、deadline、版本增长和前后台竞争约束下，集中完成能够越过 SLO feasibility boundary 的 deficit，可能比 full prefetch、单目标 advisory prefetch 和逐字节 marginal utility 分配创造更多 late-bound execution options。

检索式：

1. `"minimum KV prefix" SLO feasible route LLM`
2. `"partial KV transfer" shared bandwidth HBM scheduling`
3. `"late binding" KV prefetch multiple candidate workers`
4. `"SLO boundary" KV cache placement routing`
5. 近年限定：`2026 path-aware partial KV cache preparation`

## 2. 一手来源确认的直接先例

| 工作 | 已确认的覆盖范围 | 对当前主张的影响 |
|---|---|---|
| Llumnix, OSDI 2024 | 运行时跨实例重调度请求及其内存状态，用于负载均衡、隔离、碎片治理、优先级和 SLO | 直接否定“首次以 KV/request migration 做负载均衡” |
| Preble, ICLR 2025 | 分布式调度联合优化 KV reuse 与 computation load balancing | 直接否定“首次联合 cache locality 与 load balancing” |
| Mooncake, FAST 2025 | 全局 KV cache、SLO-aware scheduler、跨节点 KV 获取以及热点缓存迁移/复制 | 直接否定“首次联合 KV placement/transfer、路由与 SLO” |
| Online Context Caching, INFOCOM 2025 | 在线联合决定分布式 KV placement 与请求 scheduling，显式处理跨时隙耦合与负载均衡 | 直接否定“分布式 KV placement + request scheduling 为空白” |
| SYMPHONY, NSDI 2026 | 在真实请求前利用不可靠 advisory signal 预取 KV，解除 state location 与细粒度 request-level load balancing 的耦合 | 几乎覆盖“到达前 KV preparation 以激活负载均衡” |
| DualMap, ICLR 2026 | 两个候选节点、KV affinity、hotspot rebalancing 与 TTFT-SLO-aware routing | 覆盖“多候选 + cache/load routing”，但不主动准备 KV |
| Randomization Boosts KV Caching, ICLR 2026 | KV eviction 与 query load balancing 的统一在线模型 | 否定“首次统一建模 KV 与负载均衡” |
| LMetric, OSDI 2026 | 用新增 prefill tokens 与当前 batch size 的乘积兼顾 KV locality 和负载 | 构成必须击败的强简单基线 |
| Libra, NSDI 2026 | SLO-goodput、动态负载再均衡、任意 token boundary 的请求切分和 chunked KV transfer | 覆盖“partial KV transfer + SLO load balancing”的宽泛表述 |
| Low-Latency Edge LLM Handover, arXiv 2026 | Edge、受限 backhaul、partial KV transfer 与 token recomputation 的联合选择和多用户链路调度 | 若以 mobility 为核心，将高度重合；固定 ingress 负载卸载仍不同 |

## 3. 最大风险条目

### 3.1 Co-optimizing Request Scheduling and KV Caching for Edge LLM Serving

- 作者：Xishuo Li, Wei Jiao, Junyi He, Shan Zhang
- 期刊：IEEE Internet of Things Journal, 2026
- DOI：`10.1109/JIOT.2026.3709703`
- 当前可见摘要覆盖：多节点 edge LLM、block-level KV caching、request scheduling、负载不均、资源争用、任意长度 prefix、在线算法、TTFT 和理论保证。
- 风险：它已经足以否定“首次在 edge 联合 KV caching 与 request scheduling”。
- 未决：尚未取得可核验全文，不能确认是否显式处理跨节点 KV transfer、per-link shared backhaul、foreground/background contention、HBM、late binding 和 SLO-boundary completion。
- 处理原则：在全文获得前，不能以“该文没有覆盖这些点”支撑新颖性。

### 3.2 近期但边界尚需完整全文核验的工作

- SkyWalker, EuroSys 2026：跨地域 LLM load balancing，联合网络时延、KV locality、容量和 selective pushing。
- QUARTZ, Findings of ACL 2026：基于候选 worker 的 TTFT quantile、cache hit、queue 和 KV pressure 做 SLO routing。
- PBKV, arXiv 2026：预测未来 agent invocation，并进行保守 eviction 和 proactive prefetch。
- OrbitFlow, PVLDB 2026：SLO-aware、per-request、细粒度 KV layer placement 与在线重配置。
- PPD, arXiv 2026：面向 multi-turn session，在带宽拥塞下动态决定下一轮 prefill/decode 放置。
- DynoPipe, ISCA 2026：题名显示为 heterogeneous edge–cloud LLM serving，但未取得全文。
- Connex, SIGCOMM 2026：题名显示为 dynamic LLM serving 的 endpoint mobility primitives，但未取得全文。

这些条目至少说明 2026 年相关空间非常拥挤；其中任一全文都可能进一步压缩残余缺口。

## 4. 当前重合判断

| 主张 | 结论 | 证据强度 |
|---|---|---|
| C1：KV state 会造成 load imbalance / state stickiness | 已有工作明确确认 | 高 |
| C2：通过 KV migration/prefetch/placement 改善负载与 SLO | 已直接做过 | 高 |
| C3：固定 ingress、逐链路受限的多层 CEC 中创建 SLO-feasible route options | 未发现一篇公开全文同时覆盖全部条件 | 中 |
| C4：minimum-deficit completion 比 per-byte utility 更好 | 未发现直接先例，但单独看可能只是显然的阈值逆解 | 低至中 |

## 5. 暂定 novelty verdict

1. 宽泛方向的原创性：**2/10**。不能再声称“KV migration for load balancing/SLO”。
2. 当前 RoutableKV 整体的原创性：**4–5/10**。它位于 Llumnix、Mooncake、SYMPHONY、DualMap、Libra、Edge Handover 和传统 CEC 资源分配的交叉处。
3. 最窄残余机制：**CEC shared-backhaul 下的 late-bound, path-qualified, minimum-deficit option creation**。
4. 该残余机制目前不是已确立贡献，只是待证假设。它必须证明：
   - CEC 的共享逐链路瓶颈会改变最优准备顺序，不能被单个 transfer-time penalty 替代；
   - late binding 的价值高于多准备副本的额外带宽/HBM成本；
   - KV transfer、token recomputation 与不准备三种动作被统一比较；
   - 非 Oracle 的概率 qualification 在预测误差下仍可靠；
   - 相同 data plane、bytes、HBM-time 和 offered load 下超过机制完整的组合基线。

## 6. 条件判断

用户给出的后续条件是：“如果这个方向没有工作做过，再另写系统建模和具体方法。”

该条件按宽泛方向解释时明确不成立。因此本轮应：

1. 产出文献与新颖性审计文档；
2. 不产出一份把该方向当作空白的完整方法文档；
3. 把获得 IoT-J 2026 全文和完成 kill-test simulation 设为继续建模的前置门槛。

## 7. 论文存在性核验记录

执行了 ARIS `verify_papers.py` 对 arXiv 候选列表的批量核验。核验服务在本轮返回 transient failure / `verify_pending`，因此不把该工具当作确认依据。正式 venue 条目优先采用 USENIX、ICLR、ACL Anthology、IEEE DOI/DBLP 等一手或权威索引；arXiv 条目采用 arXiv 原始页面。无法取得一手全文的论文均明确标记为 pending，而不是据此推断“不重合”。

## 8. 供独立审稿人的问题

1. 上述资料是否足以否定“KV migration for load balancing/SLO 是空白”？
2. 残余的 CEC-specific 交集是否构成独立研究问题，还是现有工作的自然组合？
3. 哪一项最可能被审稿人认为 obvious？
4. 在什么证据出现前，不应继续完整系统建模？
