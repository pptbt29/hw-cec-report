# Novelty-check dossier：移动性与 CEC 是否为 KV-aware LLM serving 增加独立研究空间

**检索截止日期**：2026-07-23  
**审计对象**：用户移动后，CEC 是否应继续把 session 锚定在旧 worker，或迁移、部分准备、重算 KV，并据此形成区别于通用 KV-aware load balancing 的论文问题。  
**审计原则**：移动性只有在改变可行动作、信息结构或约束耦合时才是研究变量；仅改变 workload trace 或 path-cost 数值不构成独立贡献。

## 1. 待核查主张与检索式

### C1：问题是否客观存在

移动用户切换接入点时，如果计算状态位于原 edge，继续转发、迁移 KV 和 token-prefill 重建之间可能存在 SLO 级别的权衡。

检索式：

1. `"edge LLM" handover "KV cache" transfer prefill`
2. `"LLM" mobility handover context migration edge server`
3. `"mobile user" "KV cache" LLM edge inference`
4. `site:arxiv.org mobility edge LLM handover KV cache transfer prefill 2025 2026`

### C2：移动预测与主动状态同步是否已有先例

预测下一接入点、提前迁移状态、在 handover 时补齐增量，以及在长期迁移预算下做在线控制，是否已经属于成熟 MEC 结构。

检索式：

1. `"proactive" "KV cache" mobility edge LLM handover`
2. `"context migration" mobile edge LLM multi-turn`
3. `"EdgeWarp" edge service migration state synchronization paper 2025`
4. `"Follow Me at the Edge" service migration Lyapunov edge computing paper`
5. `site:ieeexplore.ieee.org MEC service anchoring user mobility forwarding vs migration edge server`

### C3：CEC 是否产生不同于传统 follow-me migration 的新结构

无线接入位置、执行节点和 KV 所在节点可否独立；多层 CEC、共享逐链路 backhaul、worker queue/HBM 和 token-reconstructibility 是否产生已有固定状态迁移模型不能表达的动作。

检索式：

1. `site:arxiv.org edge LLM mobility session anchoring follow user compute placement KV cache`
2. `site:dl.acm.org stateful edge application anchor service migration handover old edge forwarding`
3. `"joint handover and service migration" MEC 2023 IoT Journal`
4. `"Coordinated Container Migration and Base Station Handover" DOI abstract`
5. `"multi-destination" KV cache prefetch edge LLM mobility`

### C4：VLM/VLA 的持续输入是否扩大该空间

流式视觉、视频或传感器输入是否使“持续转发输入到旧 worker”与“一次迁移 session state”处于相同量级，同时保留可跨周期复用的 LLM/VLA 状态。

检索式：

1. `site:arxiv.org VLA edge inference offloading mobile robot KV cache 2025 2026`
2. `site:arxiv.org VLM edge inference mobility user handover state migration`
3. `site:dl.acm.org mobile edge vision language model inference offloading mobility`
4. `site:ieeexplore.ieee.org multimodal large language model edge inference mobility task offloading`
5. `"VLA-Cache" adaptive token caching robotic manipulation`

## 2. 一手来源确认的事实

| 工作 | 已确认机制 | 对当前命题的约束 |
|---|---|---|
| *Low-Latency Edge LLM Handover via Joint KV Cache Transfer and Token Prefill*, arXiv:2603.28018 | 已知 source/target BS 和 handover 时刻；handover 后联合选择 token prefill 长度与多用户共享 backhaul 的 KV 传输调度；目标为最坏用户 handover delay | 直接覆盖“移动 Edge LLM + KV/recompute + shared backhaul”；论文还把 soft handover/pre-computation 列为未来工作 |
| *Warping the Edge: Where Instant Mobility in 5G Meets Stateful Applications*, SEC 2025 | 基于 RNI 提前预测 target BS；两阶段同步低更新状态与 handover 时残余状态；使用应用时延预算；错误预测时退回 reactive migration | 直接覆盖“移动预测 + 主动状态复制 + 增量追平 + SLO-aware handover” |
| *Follow Me at the Edge*, JSAC 2018 | 未知移动下，在长期迁移预算约束内，用 Lyapunov 在线决定服务放置 | 否定“mobility + migration cost + Lyapunov”具有独立新颖性 |
| *Coordinated Container Migration and Base Station Handover*, GLOBECOM 2020 | 独立选择 BS handover 与 container migration 的目标和触发时刻，并通过 delta checkpoint 降低 downtime | 否定“首次解耦无线 handover 与计算迁移”的宽泛表述 |
| SYMPHONY, NSDI 2026 | 不可靠 future/advisory hint 下提前准备 KV，并用于 request-level load balancing | “预测下一事件后提前加载 KV”在去掉 mobility 后已有高度重合 |
| VLA-Cache, 2025 | 比较连续视觉输入，选择变化小的 visual tokens，并跨步骤复用其 KV | 证明部分 VLA 可跨帧复用状态，但该复用是选择性的，不能据此假定所有 VLA 都有 exact、append-only session KV |

## 3. 量级检查

arXiv:2603.28018 对 Qwen2.5-7B-Instruct 给出每 token KV 为 458,752 bits；3,072-token context 的 KV 为 176 MB。该文在 2 Gbit/s backhaul 下观察到 pure KV handover 相对联合 KV/prefill 的延迟劣势，并在 12 个 UE 共享资源时报告约 0.95 s 的联合方案最坏 handover delay。

以 176 MB 为例，仅比较同一 backhaul 上的字节量：

- 若持续转发流量为 25 Mbit/s，累计流量达到 176 MB 约需 56 s；
- 若为 100 Mbit/s，约需 14 s；
- 这尚未计入迁移 stall、错误目的地复制、HBM-time 和状态在迁移期间继续增长，因此只是偏向迁移的下界检查。

该计算说明两点：

1. 文本对话的后续 prompt/token 流量通常不足以快速摊销大 KV，`pin-and-tunnel` 可能长期占优；
2. 持续视频或传感器流可能进入动作切换区，但“大输入”不自动推出“迁 KV 更优”，必须同时测量 state size、输入码率、剩余驻留时间和可复用状态语义。

## 4. 第一性原理判断

移动性具有独立决策价值的必要条件不是“用户位置发生变化”，而是：

1. 新接入点使旧执行锚点不再满足 SLO，或持续 forwarding 明显占用共享 backhaul；
2. 至少一个新执行选项在迁移、部分准备或重算后能够满足 SLO；
3. 准备能在 handover lead time 或新区域 dwell time 内完成并被后续流量摊销；
4. 真实移动信息会改变动作，而不只是重复 queue/link telemetry 已经给出的结论。

若无线 handover 后仍可经新 BS 将请求 tunnel 到旧/区域 worker，并继续满足 SLO，则移动性通常只改变 path cost。此时通用 path-aware KV Router 已能表达问题，不需要独立 mobility 方法。

## 5. 剩余候选命题

### 5.1 最强候选：Anchor or Follow

在动态 physical ingress 下，独立决定：

- `pin-and-tunnel`：执行与 KV 留在旧 worker；
- `regional anchor`：把 session 锚定在覆盖多个 access cells 的区域节点；
- `follow`：迁移完整或增量 KV；
- `rebuild`：传 token 后 prefill；
- `prepare-and-late-bind`：在 handover 前为少量候选路径准备最小状态，真实位置与负载揭晓后再选择。

该命题只有在利用以下 LLM-specific 结构时才可能超过传统 state-migration versus path-stretch：

- session-private、随 token 增长的状态；
- token 是 KV 的紧凑、可重算表示；
- exact-prefix、版本差和部分准备；
- GPU queue/HBM 与共享 backhaul 的联合可行性；
- 多层模型/并行配置兼容性。

### 5.2 高重合候选：不确定多目标 soft handover

在 target cell、handover time、session continuation 和目标 load 均未知时，对多个候选目标做 growing-KV preparation 与最终 delta catch-up。

风险很高：EdgeWarp 已覆盖预测与两阶段同步，arXiv:2603.28018 已覆盖 KV/prefill 和共享 backhaul，SYMPHONY 已覆盖不可靠 hint 下的主动 KV 准备。除非能证明 joint uncertainty 或逐链路竞争产生不可约的新结构，否则容易被判为自然组合。

### 5.3 较弱候选

- `A→B→A` 返回时保留 stale KV 并做 delta catch-up：容易退化为 TTL/return prediction；
- crowd mobility 下的 stateful load wave：容易退化为普通 spatial load forecasting + state migration；
- 单纯使用 mobility predictor、SafeRL 或 Lyapunov：求解器或预测器不是问题贡献。

## 6. 必须通过的 kill tests

1. **两站点单用户相图**：用真实 runtime 测得 state size、prefill、输入流量和 backhaul；现实连续区间内必须出现 `pin`、`follow`、`regional anchor/rebuild` 的动作切换。若一个动作始终支配，问题不成立。
2. **Mobility × load-skew 因果分解**：做无移动无失衡、仅移动、仅失衡、二者同时存在的 $2\times2$ 实验。若收益只来自 load-only，论文仍是通用负载均衡。
3. **Identity/trajectory shuffle**：保持各 edge 的 aggregate arrival、queue 和 link trace 不变，只打乱 session 与轨迹绑定。若收益不变，移动身份没有额外信息价值。
4. **强 anchor baseline**：必须包含新 BS tunnel 到旧 worker、区域 edge 锚定和全局 KV store；不能把“handover 后必须迁 KV”写成默认事实。
5. **VLA state-semantics gate**：先证明目标 VLA 跨控制周期存在可精确或质量可控复用的状态；若每周期重建或低级控制必须本地，停止该方向。

## 7. 暂定判决

- 对普通文本长对话：移动性作为主变量的前景低，当前应保留 fixed-ingress RoutableKV。
- 对持续视频 VLM/VLA：存在条件性研究空间，但更像一篇独立的 mobility-aware state-continuity 论文，而不是当前论文的附加 workload。
- 最有希望的命题不是“移动后迁 KV”，而是“动态 ingress 下，何时只切网络路径，何时切换 inference state”。
- 当前状态：**Conditional Go for measurement/kill-test，No-Go for full method modeling**。

## 8. 论文存在性核验

运行 `verify_papers.py` 核验 arXiv:2603.28018、2412.10927、1809.05239、2009.05682、2501.09383 和 2502.02175：

- arXiv:2603.28018：`verified`；
- 其余五项：因 transient failure 返回 `verify_pending`。

`verify_pending` 不用于推断论文不存在。其存在性与关键机制已分别通过 arXiv 原始页面、原始 PDF 或正式 DOI 页面人工核对。

## 9. 供独立审稿人的问题

1. 该问题在什么场景是真实的一阶变量，而非 workload 修饰？
2. `Anchor or Follow` 是否只是传统 state migration versus path stretch 换成 KV？
3. 能留下的最窄非显然命题是什么？
4. 哪些反例会直接杀死该方向？
5. 在现有 EdgeWarp、Edge LLM Handover、SYMPHONY 和传统 MEC migration 文献下，应给出怎样的 Go/No-Go？
