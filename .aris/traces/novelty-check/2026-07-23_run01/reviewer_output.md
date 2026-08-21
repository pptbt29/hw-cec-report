# Independent adversarial novelty review

**Reviewer configuration**：独立上下文，仅允许读取 `dossier.md`；`gpt-5.6-sol`，`xhigh`；未编辑项目文件。  
**总体判定**：**No-Go for full modeling；Conditional Go for novelty kill-test only。**

## 1. 宽泛方向是否已做过

是，且不是边缘重合。Llumnix、Preble、Mooncake、SYMPHONY、Libra 和 Edge Handover 等工作的冗余覆盖足以否定“通过 KV migration、prefetch 或 placement 缓解负载不均并改善 SLO 是空白方向”。C1 和 C2 不能作为 novelty claims。

## 2. 残余 CEC 交集是否独立

当前更像已有构件的自然组合和部署域特化，而不是已经成立的独立问题。固定 ingress、多层异构 CEC 和共享逐链路 backhaul 只有在产生已有模型无法表达的决策结构、并导致定性不同的最优策略时，才可能成为贡献；仅增加链路容量约束和状态变量不够。

## 3. 最明显或最容易被判 obvious 的部分

1. KV state 造成粘性和负载不均：已经做过。
2. 移动、复制、获取或预取 KV 以激活远端算力：已经做过。
3. minimum-deficit completion：给定 SLO threshold 后反解最小 KV 量，容易被视为普通阈值逆解，再套 deadline scheduling、covering、knapsack 或 admission control。
4. 相对 per-byte marginal utility 的优势可能由“新增 feasible option”这一评价定义机械诱导；若只以弱基线对比，会形成 straw-man。
5. late-bound multi-candidate preparation 目前只是 proactive prefetch、多候选 routing 和两阶段资源配置的组合。

## 4. 继续完整建模前的硬门槛

1. 取得 IoT-J 2026、SYMPHONY、Libra、Edge Handover、SkyWalker、OrbitFlow、PPD 等高风险全文并完成逐项 claim chart；摘要未出现某机制不能作为“未覆盖”的证据。
2. 构造严格反例或小型定理，证明把网络压缩成单一 transfer-time penalty 或端到端带宽会做出错误选择，而 per-link shared constraint 会改变准备顺序或可行集合。
3. 先做最小 kill-test simulation；在 homogeneous/fat-network 中优势应消失，在共享链路、空间负载错配和足够可预测性同时存在时才出现。
4. 在相同 bytes、HBM-time、offered load、data plane 和预测信息下，比较 single-target prefetch、multi-target late binding、reactive migration、recompute 和 no preparation，证明 late binding 有净价值。
5. 用真实 SLO-goodput、TTFT quantiles、bandwidth、HBM-time、wasted preparation 和 foreground interference 消除阈值目标的循环论证。
6. 在 arrival、target、KV growth 和 network prediction error 下报告 false qualification 与 wasted-option 两类代价。

任何关键门槛失败都应终止方向，而不是增加更多约束或算法复杂度。

## 5. dossier 中必须降级的表述

- “未发现一篇同时覆盖全部条件”只是检索结果，不是 non-obviousness 证明。
- IoT-J 只有摘要时，未知维度必须保持 pending。
- DynoPipe、Connex 等题名级线索不能证明覆盖或不覆盖。
- 固定 ingress 与 mobility 不同只是场景差异，尚未证明技术结构不同。
- C4 还没有定理或实验支持。
- 4–5/10 只能作为内部风险评分，不能当正式证据。
- `verify_pending` 不能确认 arXiv 候选的存在性和内容。
- late binding 需要明确其可获得的非 Oracle 信息和准备提前量。
- background preparation 的 self-induced congestion 必须完整计入。

## 6. 最终建议

不要继续以“CEC 中首次通过 KV 状态移动解决 LLM 负载不均”为题。只有全文核验和 kill-test 证明“逐链路共享拥塞与 late binding 产生不可约化的新决策结构”后，才可收缩为 shared-backhaul 下、受预算和预测误差约束的 path-qualified KV option creation；否则应判定为既有 KV-aware load balancing、proactive prefetch、partial migration 和 CEC resource scheduling 的自然组合。
