# RepKV 控制面原型

> **原型边界：**本项目是用于检验 RepKV 控制逻辑的离散时间模拟器，不包含真实 KV 张量、vLLM/Ascend block allocator、RDMA、CANN 或 NPU 推理执行。速率由模型形状和硬件指标推导，不是实测吞吐。

## 研究问题

本原型检验以下控制逻辑是否自洽：在下一轮请求只能概率预测、多个 session 共享 HBM、跨节点带宽、本地恢复带宽和后台计算资源的条件下，控制器能否只准备使一个候选节点达到 TTFT SLO 所需的最小有效 KV 前缀，从而以有限资源增加可路由节点。

实验报告给出两个由物理参数决定的门槛，并在门槛两侧分开测量预放置与 SLO 感知回收。全部实验的对照表、关键结论和两个手段的适用条件见 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)；逐格数字与实现细节见 [PROTOTYPE_VERDICT.md](PROTOTYPE_VERDICT.md)。

## 已实现机制

- 由模型形状与硬件指标推导 block 字节数、prefill/重算/传输/restore 速率、HBM 容量和 decode 槽位数，不直接给定 blocks/s；
- 每个节点三条串行资源时间线（设备计算、网卡、host 链路）加一组 decode 槽位；前台恢复与后台准备占用同一条时间线；
- closed-loop 到达：下一轮按上一轮的完成时刻加思考时间排定，一个 session 不会同时有两个在途请求；
- 并发 session 补充，使负载可以稳定在给定并发度而不是随脚本耗尽而衰减；
- 类条件返回概率 $q_s(t,\Delta)$，含"可能不再返回"的质量，同时驱动准备和回收；
- 按预测返回时刻构造的未来排队估计，decode 按槽位投影，恢复与排队取最大值；
- 把 TTFT 预测误差分为各节点共同项和节点独立项的集群可路由概率，两个尺度随预测排队增长；
- 每个 session、节点和存储层级上的严格连续有效前缀；
- 有限容量的 HBM 与本地 DRAM 两级 KV 状态；DRAM 层按最近使用时间淘汰；
- 对端 HBM 与对端 DRAM 均可作为传输来源，两者同为网络速率；
- 由本地恢复、远端传输和精确重算组成的连续混合计划，方法代价为 $1/R_m$；
- 硬 TTFT SLO 可行性检查（二分求解最小前缀）和预测窗口内完成性检查；
- 面向集群可路由性的收益，准备为它挤掉的降级付费；
- 根据共享资源稀缺程度调整候选计划成本；
- 带容量预留、完成验证和取消机制的渐进式 batch 执行；
- 回收分数按集群副本计价：第一项是集群可路由概率的变化；若另一节点在本卡丢掉这块 HBM 之后仍能满足预测 SLO，则本卡副本视为冗余，不收重建代价并压低降级地板；
- 面向 router 的有效前缀与剩余恢复时间快照；
- 防止无效前缀和 HBM 超额承诺的运行时不变量。

详细设计见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 运行方法

默认复现报告第 6.1 节的数据中心主配置：4 节点、存活并发 90、上下文 32000 token、SLO 3.0 秒。该配置位于取回可行、重算不可行区间。

```bash
python3 run.py
```

用截止时间跨过两个门槛（固定负载，只改 SLO）：

```bash
python3 run.py --ttft-slo 0.8
python3 run.py --ttft-slo 8.0
```

准备/回收消融。只看输出中的 `repkv` 行：`--lru-eviction` 关闭按价值回收，`--min-return-probability 1.1` 关闭全部准备。

```bash
python3 run.py --lru-eviction
python3 run.py --min-return-probability 1.1
```

缩短思考时间或改链路、DRAM 层：

```bash
python3 run.py --think-scale 0.33 --network-gbytes 0.125 --host-gib 32
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

- `on_demand`：请求到达前不准备 KV，按 LRU 回收；
- `eager_full`：使用相同恢复机制，渐进式构造一个完整备用副本，按 LRU 回收；
- `repkv`：只接纳能够新增硬 SLO 可行节点的最小前缀计划，并按预期 SLO 损失回收。

准备与回收可以用开关分开，两者都关闭时与 `on_demand` 逐 seed 逐位相同。

三种策略共享同一份 session 脚本，但到达时刻由各自的完成时刻决定，因此完成的请求数不同。比较时必须同时看请求数、后续轮 attainment 和 goodput。首轮没有先前 KV 可供放置，实验口径用 `continuation_attainment`。

## 文档语言

本项目后续新增和更新的 Markdown 文档统一使用中文。代码标识符、命令、标准系统名和无法准确翻译的术语可以保留英文。

## 尚未覆盖的系统边界

硬件表、达成率和链路带宽均为 mock 值。两个门槛对网络带宽和每 token KV 字节数线性敏感，真实部署的区间边界须用实测重算。队列预测仍是流体近似，节点同质，workload 没有会话迁移。真实系统结论仍需接入实际 KV 布局、block table 更新、异步复制流和跨机故障处理。

结果范围和证据边界见 [PROTOTYPE_VERDICT.md](PROTOTYPE_VERDICT.md)。
