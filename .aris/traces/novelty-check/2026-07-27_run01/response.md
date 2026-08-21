**结论：当前应判定为“组合性很强，先 pivot；未证明前不宜立题”。**  
分数：新颖性 **3/10**，重要性 **7/10**，成熟度 **3/10**。若 ICDCS’26 的论文已经包含“预测未来到达 + 多 edge 迁移候选 + 到达后选择”，则直接 **stop**；若其仅做已到达请求的单目标迁移，才有条件继续。

1. **结构性新问题？**  
按当前描述，尚未成立。它是“advisory 驱动预迁移/预取 + 多候选复制 + 到达后 SLO 路由 + 在线资源定价”的拼接。  
真正可能成立的核心只能是：**请求尚未存在时，系统为有误差的 advisory 购买可撤销的空间状态准备选项；真实请求出现后才行使该选项，并且该选择同时受共享 backhaul、HBM、实时队列和 deadline 约束。**  
但“多候选”本身不是结构贡献；在该表述下就是把 single target 扩展成 top-$k$，很容易被审稿人视作自然工程升级。

2. **最强 kill objection**

- **SYMPHONY 是一号杀手**：其 advisory、跨节点 staging 与预选目标已覆盖大部分主线。将 even-load 换成 SLO、将单目标换成候选集合、再套 primal-dual，不足以构成系统新问题。
- **KVFlow 是二号杀手**：它已针对分支 workflow，在有限并发预取预算下预取所有可能下一 agent，并在未准备好时做状态感知调度。这与“不确定 tool/workflow signal → 多候选准备 → 实现后选择”在决策结构上高度同构；仅把 CPU–GPU 换成 edge–edge 不能自动产生新颖性。
- **CachedAttention / Mooncake / DualMap 等已拆走其余部件**：异步预加载、队列感知放置、到达后 SLO 路由、KV 传输均已有。因此“端到端整合”只能是工程贡献，且必须显示简单拼装会系统性失败。
- **理论包装不救新颖性**：two-stage online primal-dual / Lyapunov + shadow price 是标准工具。除非迁移可分阶段、准备可撤销、共享资源与 SLO 可行性之间存在新且必要的结构定理，否则只是在已知问题上换求解器。
- **ICDCS’26 是硬风险**：官方程序确认已有 *Efficient KV Cache Migration for Geo-Distributed LLM Inference in Collaborative Edge Computing*。全文未获得前，不能声称 edge KV migration 的问题定义或机制新颖；应按“可能直接重合”处理，而不是按未知利好处理。

3. **什么主张才可能提高新颖性**

将论文限定为以下可证伪命题，而不是“更聪明地预迁移”：

> 在带误差且有 lead-time 的 advisory 下，预先把 session KV 的**可完成比例/迁移进度**分配到未绑定的候选边缘节点；真实到达后，只在“剩余传输 + 实时排队 + 推理”仍可满足 SLO 的候选中行使准备选项。该问题的最优动作可能是 no-op、owner-sticky、单候选部分准备或多候选部分准备，且这四类动作不能由固定目标、top-$k$ 复制或到达后路由替代。

必要条件：

- “partial staging”必须是真正有用的可恢复中间态，而非“复制完整 KV 的百分比”这种人为连续化。
- 给出一个严格反例/定理：任意固定目标、贪心 top-$k$ 或 arrival-only router 在共享链路/HBM 下会损失常数量级 SLO-goodput；而 recourse option policy 不会。
- 算法贡献应针对该特殊结构，例如可行性单调性、部分迁移的阈值/索引性质，或可校准 advice 下的 regret/资源违约界；泛用 primal-dual 不足。

4. **必须通过的 kill tests**

**问题存在性：**

- 测 advisory 的 precision、recall、校准曲线、lead-time 分布；特别报告“可用 lead-time ≥ 剩余 KV 传输时间”的比例。否则预准备物理上来不及。
- 分解 SLO 违约来源：排队、KV 传输、prefill、decode。若 decode 或算力排队主导，KV 准备几乎没有价值。
- 测空间异质性：候选 edge 之间的链路/队列差异必须足以改变 SLO 可行性；否则 owner-sticky 已接近最优。
- 用真实或保守合成的移动性、对话和 tool/workflow trace；不能只用“advisory 准确且请求随后到达”的友好 trace。
- 报告 wasted staging 的 HBM-byte·s、backhaul-byte、能耗及其挤占其他会话的 SLO 损失，而非只报目标会话收益。

**oracle-gap：至少三组不可缺：**

| 要验证的价值 | 对照 oracle / baseline | 若 gap 小于噪声，结论 |
|---|---|---|
| advisory 是否有用 | arrival-only 最优路由 vs 知道 advisory 的离线最优 | 问题不存在 |
| recourse 是否有用 | 单一不可撤销 target 的 advisory oracle vs 多候选 recourse oracle | 核心机制不存在 |
| 在线算法是否有用 | 你的在线策略 vs 同信息、同预算的离线 recourse oracle | 算法不成立 |
| 预测鲁棒性 | perfect / calibrated / miscalibrated / adversarial advice | 只能做理想化 demo |

此外必须对比：no-op、owner-sticky、固定单目标 staging、完整 top-$k$ 复制、partial-only、arrival-time SLO router，以及“SYMPHONY + 最小改动的 SLO selector”。若最后一个已追平，论文应停止。

**最终 verdict：Pivot。**  
先把贡献缩到“带不确定 advice 的、可撤销 partial KV readiness 与 SLO recourse”的独立问题；先跑上述 oracle-gap。若 `multi-candidate recourse oracle − single-target oracle` 很小，或 ICDCS 论文已覆盖预测式多目标迁移，立即 stop，不要用更复杂的在线算法掩盖不存在的贡献。
