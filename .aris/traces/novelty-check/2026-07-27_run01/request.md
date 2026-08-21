你是独立、挑剔的系统论文审稿人。请只做 novelty/importance kill review，不迎合。

候选 idea：面向 edge continuum/CEC 的 stateful LLM serving。系统在不确定 advisory（用户开始输入、tool/workflow signal）到来时，不立即把未来请求绑定到单节点，而选择 no-op / owner-sticky / 将 session KV 部分或完整 stage 到一个或少量候选 edge workers，并分配共享 backhaul/HBM 的投机预算；真实请求到达后根据已实现队列与链路状态，在 prepared candidates/owner/recompute fallback 中做 recourse routing。目标是长期 SLO-goodput，约束 speculative bandwidth/HBM/wasted staging budget。拟用 two-stage online primal-dual 或 Lyapunov + shadow prices，不用 SafeRL。

已知正式 prior work：SYMPHONY NSDI 2026（advisory + cross-node staging + simple even-load，single preselected target，无显式 SLO objective），ReSK IEEE IoT-J 2026（edge current-arrival routing + local KV retention cumulative cost，无 migration/future action/SLO），DualMap ICLR 2026（SLO-aware cache/load routing after request arrival），Randomization Boosts KV Caching... ICLR 2026（unified cache eviction + routing model），KVFlow NeurIPS 2025（workflow-aware proactive CPU-GPU prefetch），CachedAttention ATC 2024（queue-aware local prefetch），Mooncake FAST 2025（arrived-request SLO-aware routing/KV transfer），SkyWalker EuroSys 2026（cross-region cache/locality routing no KV migration），Libra NSDI 2026（within-request microrequest split/SLO batching），SuperInfer MLSys 2026（SLO-aware local KV rotation），OrbitFlow PVLDB 2026（SLO-aware local KV tier placement），以及预印本 Pythia、Continuum、TokenCake、Talaria、SMetric、Robust KV uncertainty。直接 blocker：ICDCS 2026 官方议程包含 *Efficient KV Cache Migration for Geo-Distributed LLM Inference in Collaborative Edge Computing*，但全文暂不可得。

请回答：

1. 候选是否包含可防守的新结构问题，还是只是组合已知组件？
2. 最强 novelty-killing related work / objection 是什么？
3. 什么精确 claim、action 或 algorithm 才能抬高新颖性？
4. 必须做哪些 oracle-gap 和 problem-existence kill tests？
5. 给出 novelty、importance、readiness 的 1–10 分及 go/pivot/stop 结论。

请用中文简洁回答并给出明确理由。
