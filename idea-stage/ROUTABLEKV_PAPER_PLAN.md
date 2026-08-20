# RoutableKV Paper Development Plan

本文件维护 Introduction 逻辑、论文目录、图表预算和详细实验设计。Abstract、研究问题、数学模型、触发机制和研究贡献见 [ONE_PAGE_IDEA.md](ONE_PAGE_IDEA.md)。

## 1. Introduction Logic

### P1 — CEC 可缓解局部负载，但多轮状态限制远端资源的及时使用

说明 CEC 如何利用其他站点的空闲算力缓解局部突发负载，以及 KV 位置、站点异构和共享链路为什么会延迟这种协作。明确站点拥有的 GPU、HBM 与站点间带宽均有限。

### P2 — 多轮会话把可用算力变成状态依赖资源

解释 KV 如何跨轮增长。通过一个定量例子展示：远端节点有空闲 GPU，但请求到达后才恢复 KV 会违反 TTFT SLO。

### P3 — 无差别提前复制会把状态瓶颈转化为资源竞争

说明多个会话同时准备时会竞争链路、后台 GPU 和 HBM；用户可能迟迟不返回，使完整副本长期占用资源。

### P4 — 现有工作的三个缺口

1. 到达时路由和运行时迁移只能利用或调整已经到达的请求；
2. advisory 预取和访问预测没有在多会话、多资源竞争下选择达到 SLO 的最小预测准备方案；
3. 通用淘汰策略没有计算为准备方案腾出空间会损失多少宿主路由器下的 SLO 成功请求。

### P5 — 核心 insight：准备和回收应通过同一个路由后结果耦合

对每条迁移、重算或混合路径，只准备到候选节点能够满足 SLO；HBM 足够时不运行回收，HBM 不足时只接受收益高于最小回收损失的准备方案。

### P6 — 问题建模

先给出长期 SLO-goodput 目标，再说明未知未来使其无法在线直接求解。随后提出预计返回窗口上的多会话资源分配问题及 HBM 回收子问题；用联合返回场景评估共享队列下的预计 goodput，并用 block 时限可行性检查保证所选任务能在各自窗口前完成。

### P7 — 系统设计

概述 Return-Window Manager、Preparation Planner、Preparation Admission、On-Demand Reclaimer、Router Adapter 和 Foreground Resource Guard。明确请求完成只登记状态并建立延迟任务；返回窗口决定会话何时进入候选池，block 完成或前台资源释放只唤醒候选池内的重新调度；advisory 只修正窗口，回收是 HBM 不足时的条件子过程。

### P8 — 贡献与结果预告

列出状态受限容量刻画、窗口级建模、达到 SLO 的最小预测准备与最小损失回收、可接入现有路由器的运行时。段末只使用实验实际支持的数字。

## 2. Paper Structure and Page Budget

以下按 12 页双栏系统论文设计。

| 章节 | 页数 | 中心内容 |
|---|---:|---|
| 1. Introduction | 1.5 | 问题、缺口、insight、贡献与主结果 |
| 2. Background and Motivation | 1.2 | KV 生命周期、CEC 资源条件、状态受限容量测量 |
| 3. Model and Design Goals | 1.2 | 返回窗口、宿主路由器接口、联合场景目标与有时限的窗口级问题 |
| 4. RoutableKV Design | 3.0 | 准备方案生成、准入、按需回收、Router Adapter、资源保护 |
| 5. Implementation | 0.8 | KV block、版本检查、存储层、运行时接口和求解开销 |
| 6. Evaluation | 3.5 | 端到端、资源边界、消融、可插拔性、鲁棒性与扩展性 |
| 7. Related Work | 0.6 | 路由迁移、主动准备、分层缓存和回收 |
| 8. Conclusion | 0.2 | 问题、方法和实际结论 |

**总计：12.0 页。**

## 3. Figure and Table Budget

1. **Fig. 1 — Problem and insight**  
   展示本地过载、远端空闲但缺少 KV、达到 SLO 的最小预测准备，以及错误回收如何撤销该执行选择。

2. **Fig. 2 — Characterization**  
   展示 KV 缺口、链路、队列、节点速度和 HBM 如何共同造成状态受限容量。

3. **Fig. 3 — Deferred preparation timeline and formulation**  
   时间线上展示请求完成后登记状态、预计返回窗口、可选 advisory 修正、block 调度边界和按需 HBM 回收，并标出何时进行多会话准备选择。

4. **Fig. 4 — Architecture**  
   展示宿主路由器、Router Adapter、Preparation Planner、Admission、Reclaimer、KV runtime 和资源监测。

5. **Fig. 5 — Main results**  
   在相同后台资源预算下比较 SLO-goodput、P99 TTFT 和资源消耗。

6. **Fig. 6 — Ablation and robustness**  
   比较立即完整复制、窗口触发完整复制、达到 SLO 的最小预测准备、独立 LRU、最小损失回收、联合准入，以及预测和监测误差。

7. **Table 1 — Testbed and workloads**  
   节点、GPU、模型、KV block、存储层、链路、会话和 SLO 设置。

8. **Table 2 — Capability and overhead comparison**  
   各方法是否支持 advisory、迁移、重算、部分准备、按需回收和 router adapter，并报告控制开销。

## 4. Evaluation Design

### 4.1 Prototype and Replay

- 至少三个可协作边缘站点，区域站点或云端仅作为补充对照；
- 节点使用兼容模型，但具有不同 GPU 速度和 HBM 配额；
- 真实或受控网络呈现不同 RTT、非对称带宽和共享拥塞；
- 后台准备、下移、请求到达后的恢复和前台流量竞争同一真实数据路径；
- 用原型测量校准离散事件回放，并用 P50/P95 TTFT、排队和链路利用率检验回放误差。

### 4.2 Workloads and Variables

- 至少两个 KV 大小和计算特征不同的长对话模型；
- 公开或真实多轮日志提供上下文增长、轮次间隔和继续会话概率；
- 低负载、局部过载和全局过载三个区域；
- 后台带宽、后台 GPU 配额、HBM 安全水位、上下文长度、节点速度差、返回窗口误差和 advisory 可用性；
- SLO 在实验前按照应用类别确定，不根据结果调整。

### 4.3 Baselines

1. 请求到达后的按需恢复；
2. 请求完成后立即进行完整准备；
3. 预计返回窗口触发的完整准备；
4. 完整 KV advisory 预取；
5. SYMPHONY 式分层、逐层准备；
6. 仅按返回概率、KV 大小或即时 TTFT 降幅排序；
7. LRU、TTL、访问概率、未来调用距离和恢复成本回收；
8. 达到 SLO 的最小预测准备 + LRU；
9. 新回收器但不主动准备；
10. 完整 RoutableKV；
11. 知道真实未来请求、队列和网络状态的理想上界。

Llumnix、DualMap、KVRouting 和 SYMPHONY 的原生系统进入端到端系统比较；对其机制做适配时，应明确标注适配边界。

### 4.4 Two Evaluation Modes

**机制隔离实验**：所有 KV 管理方法接入同一个 predicted-TTFT 参考路由器，并对齐接纳规则、KV runtime、资源配额和前台优先级。

**可插拔性实验**：分别将 RoutableKV 接入 least-loaded、cache-aware 和 predicted-TTFT 路由器，验证收益是否依赖某一个特定路由规则。完全忽略 KV 和恢复成本的路由器只作为负面边界，不用于证明可插拔收益。

### 4.5 Metrics

- SLO-goodput、SLO attainment、P50/P95/P99 TTFT；
- 后台传输量、GPU 时间和 HBM GB·s；
- 无效准备量、下移和删除次数、恢复成本；
- 准备收益与回收损失预测误差；
- 对 Prefill、Decode、按需恢复和通信的干扰；
- 控制器求解时间、状态更新开销和公平性。

### 4.6 Required Ablations

- 请求完成后立即准备、预计返回窗口触发和显式 advisory；
- 完整 KV 与达到 SLO 的最小预测准备；
- 只迁移、只重算与混合路径；
- 保留全部候选与 Pareto 候选过滤；
- 不考虑回收损失的准备准入；
- 达到 SLO 的最小预测准备 + LRU；
- 最小损失回收但不主动准备；
- 不同 HBM guard watermark；
- 仅内部返回预测、显式 advisory 修正及二者结合；
- 状态采样间隔和状态变化阈值；
- 不同宿主路由器。

## 5. Related Work and Open Risks

### 5.1 Related Work Placement

Related Work 按五组简要组织：

1. Llumnix、DualMap：请求到达时或运行中的路由与迁移；
2. SYMPHONY：advisory 触发的主动 KV 准备；
3. KVCache Cache in the Wild、Bidaw：KV 复用时间刻画和访问预测；
4. CachedAttention、InferCept、Mooncake：分层 KV 存储与按需恢复；
5. KVFlow、KVRouting：工作流预取、路由与回收。

每组只说明其主要动作，以及 RoutableKV 增加的窗口级多会话资源分配、达到 SLO 的最小预测准备或按需回收维度。

### 5.2 Open Risks

- 多资源准备方案可能通常退化为完整复制，无法形成有意义的非支配选择；
- 宿主路由器如果不读取 KV 或恢复成本，准备好的节点可能不会被使用；
- ICDCS 2026 的 *Efficient KV Cache Migration for Geo-Distributed LLM Inference in Collaborative Edge Computing* 可能与跨站准备机制重叠，正式全文公开后需要重新核验。

### 5.3 Falsification Criteria

- 状态受限容量在可信负载中不显著；
- 达到 SLO 的最小预测准备量几乎只有 0 或完整 KV；
- 资源非支配方案通常只有一个完整复制方案；
- HBM 回收很少发生或不影响 SLO-goodput；
- 独立 LRU/访问预测/恢复成本策略达到相同结果；
- 后台干扰计入后没有端到端收益；
- 收益仅存在于一个人为选择的路由器、SLO 或极端负载点。

## 6. Literature Audit Links

- [CEC_KV_LOAD_BALANCING_LITERATURE_AUDIT.md](CEC_KV_LOAD_BALANCING_LITERATURE_AUDIT.md)
- [SLO_AWARE_KV_PREPARATION_ROUTING_AUDIT_20260727.md](SLO_AWARE_KV_PREPARATION_ROUTING_AUDIT_20260727.md)
- [ROUTEPREP_ONE_PAGE_IDEA.md](ROUTEPREP_ONE_PAGE_IDEA.md)
