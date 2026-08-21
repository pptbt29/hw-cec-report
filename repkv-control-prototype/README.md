# RepKV 控制面原型

> **原型边界：**本项目是用于检验 RepKV 控制逻辑的离散时间模拟器，不包含真实 KV 张量、vLLM/Ascend block allocator、RDMA、CANN 或 NPU 推理执行。

## 研究问题

本原型检验以下控制逻辑是否自洽：在下一轮请求只能概率预测、多个 session 共享 HBM、跨节点带宽、本地恢复带宽和后台计算资源的条件下，控制器能否只准备使一个候选节点达到 TTFT SLO 所需的最小有效 KV 前缀，从而以有限资源增加可路由节点。

## 已实现机制

- 不使用真实未来到达时间的合成 session 返回预测；
- 按预测返回时刻构造的未来排队估计，而不是只读当前 `busy_until`；
- 把 TTFT 预测误差分为各节点共同项和节点独立项的集群可路由概率；
- 每个 session、节点和存储层级上的严格连续有效前缀；
- HBM 与本地 host 两级 KV 状态，以及 HBM 压力下的尾部降级；
- 对远端传输来源逐区间验证，不再以“任意位置存在部分 KV”代替来源正确性；
- 由本地恢复、远端传输和精确重算组成的连续混合计划；
- 硬 TTFT SLO 可行性检查和预测窗口内完成性检查；
- 面向集群可路由性的收益，而非只评估当前最优节点；
- 根据共享资源稀缺程度调整候选计划成本；
- 带容量预留、完成验证和取消机制的渐进式 batch 执行；
- 准备和 HBM 回收共用预期 SLO 成功概率变化；
- 面向 router 的有效前缀与剩余恢复时间快照；
- 防止无效前缀和 HBM 超额承诺的运行时不变量。

详细设计见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。原型结论和证据边界见 [PROTOTYPE_VERDICT.md](PROTOTYPE_VERDICT.md)。

## 运行方法

批量比较三种策略：

```bash
python3 run.py
```

交互式观察状态变化：

```bash
python3 trace.py
```

非交互 smoke trace：

```bash
python3 trace.py --steps 40 --no-ansi
```

## 比较策略

- `on_demand`：请求到达前不准备 KV；
- `eager_full`：使用相同恢复机制，渐进式构造一个完整备用副本；
- `repkv`：只接纳能够新增硬 SLO 可行节点的最小前缀计划，并根据全局预期收益和资源稀缺程度协调执行。

## 文档语言

本项目后续新增和更新的 Markdown 文档统一使用中文。代码标识符、命令、标准系统名和无法准确翻译的术语可以保留英文。

## 尚未覆盖的系统边界

当前速率均为参数，前台恢复竞争只折算为延迟，队列与返回预测也经过简化。真实系统结论仍需接入 vLLM Ascend 或 MindIE 的实际 KV 布局、block table 更新、模型兼容元数据、异步复制流、跨机传输、失败处理，并在多个物理 Ascend 节点上测量前后台干扰。

控制面本身也还有一个未解决的问题：默认 workload 的排队压力太低，`repkv` 的实际准备量只有几十个 block，因此当前数值不能用来判断预放置是否有效。原因和证据见 [PROTOTYPE_VERDICT.md](PROTOTYPE_VERDICT.md)。
