# Independent adversarial novelty review

**Reviewer configuration**：独立上下文；`gpt-5.6-sol`，`xhigh`；未编辑项目文件。  
**总体判定**：**No-Go for full paper；Conditional Go for measurement and kill-test only。**  
**新颖性风险**：**8.5/10**，其中 10 表示与已有工作重合风险最高。

## 1. 问题真实性

问题真实，但只在以下条件同时成立时重要：

- 用户切换后仍有足够长的剩余驻留或 session 时长；
- 旧计算节点产生显著 path stretch、排队或持续 backhaul 流量；
- 新节点具有兼容模型及足够算力/HBM；
- KV 迁移或重算能够在 SLO 内完成；
- 状态增长速度低于增量同步能力，或系统能通过窗口化/压缩使迁移收敛。

“VLM/VLA 输入大，因此迁移一定更划算”不成立。视觉流同时可能增加 tunnel 流量和待同步 KV；真正需要测量的是端侧压缩或视觉编码后穿过 CEC backhaul 的实际流量。

## 2. 与已有问题的关系

当前抽象仍高度接近经典问题：

> 保持服务在旧节点并持续支付 path-stretch 成本，或支付一次状态恢复/迁移成本把服务移近用户。

已有工作分别覆盖长期 path stretch–迁移成本、接入与计算迁移协同、预测式动态状态预同步、KV transfer/token reconstruction、不可靠 hint 下的 KV prefetch，以及多目标服务复制。

KV 的 append-only、exact-prefix、token-recomputable 和 block-transfer 性质会改变成本与实现，但仅更换状态类型不足以证明新问题。必须证明这些性质导出了通用 state-migration 模型无法表达或强组合基线无法实现的新决策结构。

## 3. 最窄可保留命题

最可信的是一个发现型命题：

> 在移动的连续多模态自回归服务中，持续视觉输入的 path-stretch 流量与 growing、recomputable KV 的恢复债务同时演化，导致 text 与 streaming VLM/VLA 出现可测量的相反最优驻留策略；系统利用该 crossover，在相同预测和资源预算下，于真实请求/链路状态揭晓后，在旧计算锚点和多个 exact-prefix-ready 候选节点之间 late-bind。

仍需证明：

- crossover 不是简单的 `state_size / traffic_rate` 阈值；
- late binding 在相同复制字节、HBM-time 和预测信息下优于单目标预取；
- KV growth、连续 prefix 与 transfer/recompute 并行产生了新的结构，而非换皮 MDP/Lyapunov。

## 4. 三个最强 kill tests

1. **现实 phase diagram**：扫描真实 backhaul 输入率、KV 增长率、同步带宽、窗口/压缩、驻留时间和 RTT。若只有未压缩视觉输入支持迁移，问题价值被杀死。
2. **通用迁移模型等价性**：把 KV 恢复折算为有效迁移成本，运行普通 MEC MDP/Lyapunov/阈值策略。若性能接近，KV 没有产生新控制结构。
3. **公平强组合基线**：组合 EdgeWarp 两阶段同步、Edge LLM Handover 的 transfer/prefill、SYMPHONY 的 advisory prefetch 与 sticky fallback；所有方法共享预测、带宽、HBM 和一致性约束。若多候选收益来自额外副本、更多资源或遗漏 wasted replication/foreground interference，则 No-Go。

## 5. 不可声称

- 首次研究移动 Edge LLM handover；
- 首次解耦无线 handover 与计算/状态位置；
- 首次联合 KV transfer 与 token prefill/recompute；
- 首次预测式主动同步动态状态；
- 首次联合接入切换和计算迁移；
- 首次做长期时延–迁移成本优化；
- 首次用不可靠 hint 预取 KV 并负载均衡；
- 首次多候选预准备或服务复制；
- 首次跨帧复用 VLA visual-token KV；
- 文本输入输出总是小；
- VLM/VLA 流式输入必然使迁移优于 tunneling；
- “未发现一篇同时包含全部动作”即可证明非显然性。

## 6. 最终判决

**问题值得做量级测量和 kill-test，但尚不值得按当前统一机制立项。只有实验证明 growing multimodal KV 与持续 path stretch 产生了通用迁移模型解释不了的稳定新现象，才进入完整建模。**
