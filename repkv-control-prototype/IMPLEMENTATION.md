# RepKV 控制面原型实现说明

## 1. 实现定位

当前项目是离散时间控制面模拟器。它回答的是“RepKV 的状态、计划和在线决策是否能以因果一致、资源有界的方式运行”，不是“真实 Ascend 集群能达到多少吞吐”。

模拟器把真实 KV 张量压缩为连续 block 前缀，把节点推理负载压缩为 `busy_until`，把网络、本地恢复和后台计算压缩为 blocks/s。该抽象足以检查计划正确性、资源竞争和策略相对趋势，但不能推导绝对硬件性能。

代码分为四部分：

| 文件 | 职责 |
|---|---|
| `repkv_sim/planner.py` | 纯函数形式的前缀区间划分、恢复规划和最小 SLO 前缀求解 |
| `repkv_sim/simulator.py` | workload、节点和 session 状态、router、RepKV controller、回收及指标 |
| `run.py` | 多 seed 批量比较和 CSV 输出 |
| `trace.py` | 逐 tick 展示完整状态的交互式终端界面 |

## 2. 默认系统参数

默认配置为：

| 参数 | 数值 | 含义 |
|---|---:|---|
| `nodes` | 4 | 节点数 |
| `sessions` | 48 | 合成 session 数 |
| `horizon_s` | 240 s | 单个 seed 的模拟时长 |
| `step_s` | 0.5 s | 控制周期 |
| `hbm_blocks` | 180 | 每节点 HBM block 容量 |
| `ttft_slo_s` | 2.0 s | TTFT SLO |
| `prefill_blocks_s` | 24 | 新 prompt prefill 速率 |
| `decode_blocks_s` | 10 | output decode 速率 |
| `transfer_blocks_s` | 20 | 前台远端恢复速率 |
| `host_restore_blocks_s` | 28 | 前台本地 host 恢复速率 |
| `recompute_blocks_s` | 12 | 前台精确重算速率 |
| `preparation_horizon_s` | 18 s | 请求前准备窗口 |
| `block_batch` | 4 | 单次后台 action 最大 block 数 |
| `high_watermark` | 0.92 | HBM 回收触发水位 |
| `low_watermark` | 0.82 | HBM 回收目标水位 |
| `arrival_forecast` | True | 是否用预测到达构造未来排队估计 |
| `shared_uncertainty_s` | 0.45 s | 各节点共同承受的 TTFT 预测误差 |
| `queue_uncertainty_s` | 0.25 s | 各节点独立的排队估计误差 |
| `assumed_output_blocks` | 4.0 | output 长度估计的初值 |

后台资源只允许使用前台标称速率的一部分：

$$
r_m^{bg}=\alpha_m r_m.
$$

默认有效后台速率为：

| 方法 | 后台比例 | 有效速率 |
|---|---:|---:|
| 本地 host 恢复 | 0.45 | 12.6 blocks/s |
| 远端传输 | 0.45 | 9 blocks/s |
| 精确重算 | 0.30 | 3.6 blocks/s |

这些值不是硬件测量结果，只是用于形成资源冲突的实验参数。

## 3. Workload 模型

### 3.1 Session 类别

每个 session 以相同概率属于三类之一：

| 类别 | 轮间隔中位数 | 初始返回概率 |
|---|---:|---:|
| fast | 12 s | 0.88 |
| normal | 22 s | 0.78 |
| slow | 40 s | 0.66 |

### 3.2 首轮到达

每个固定 session 独立抽取一次首轮到达时间：

$$
A_{s,0}\sim \operatorname{Exponential}\left(\frac{1}{28}\right).
$$

这不是固定 QPS 的稳态 Poisson 流。系统只创建 48 个 session，不会持续补充新 session，因此首轮到达强度随时间下降。

### 3.3 多轮返回

类别 $c$ 的实际轮间隔为：

$$
\Delta A_{s,k}
=
\operatorname{clip}
\left(
m_c\exp(Z),6,75
\right),
\qquad
Z\sim\mathcal N(0,0.55^2).
$$

第 $k$ 轮后的返回概率为：

$$
q_{s,k}=\max(0.18,q_{s,0}-0.045k).
$$

下一轮是否真实出现由 Bernoulli 抽样决定。每个 session 的上限轮数在 3–8 之间随机选择，但 session 可以提前结束。

第一轮 prompt 为 8–16 blocks 均匀分布，后续 prompt 为 3–9 blocks，output 为 2–7 blocks。历史 context 是此前所有 prompt 和 output 的累计长度。

### 3.4 下一轮预测

预测间隔与实际间隔只共享 session 类别，不共享随机噪声：

$$
\widehat{\Delta A}_{s,k}
=
m_c\exp(\widehat Z),
\qquad
\widehat Z\sim\mathcal N(0,0.28^2).
$$

因此 controller 不读取真实未来到达时间。当前预测器仍是合成模型，不是从真实 trace 训练得到的模型。

## 4. 状态模型

### 4.1 KVReplica

每个 session 在每个节点上的 KV 状态包含：

- `hbm_prefix`：HBM 中从 block 0 开始连续正确的前缀长度；
- `host_prefix`：本地 host tier 中连续正确的前缀长度；
- `prepared_blocks`：当前 HBM 前缀中由后台准备产生且尚未被请求消费的 block 数；
- `last_used`：最近使用时间。

状态只表示前缀 $[0,L)$，不允许以一个计数表示任意离散 block 集合。这保证恢复区间具有明确来源。

当前实现满足：

$$
0\le L_{hbm}\le L_{host}.
$$

需要注意，代码在请求完成或后台准备完成时会同步扩大 `host_prefix`，没有单独计算写入 host tier 的带宽和容量，因此 host 副本成本被低估。

### 4.2 Node

节点维护：

- `busy_until`：现有推理工作预计完成时间；
- `replicas`：各 session 的 KV 前缀；
- `active_reservations`：活动请求完成其 context 所需的 HBM 预留；
- `background_reserved`：在途后台 batch 已预留但尚未写入的 HBM。

节点总承诺容量为：

$$
C_n^{committed}
=
C_n^{hbm}
+C_n^{active-extra}
+C_n^{background-reserved}.
$$

每个 tick 都检查：

$$
C_n^{committed}\le C_n.
$$

### 4.3 SessionState

session 状态包括已完成轮次、最新完整 context 长度、当前活动请求结束时间和执行节点。活动请求在其执行节点上的 KV 不允许被回收。

## 5. 每个控制周期的事件顺序

每个 0.5 秒 tick 按以下顺序执行：

1. 完成到期的后台 batch；
2. 完成到期的前台请求；
3. 处理当前时刻真实到达的请求；
4. 对尚未到达的预测请求执行后台准备决策；
5. 累计 HBM block-seconds；
6. 检查状态不变量。

真实请求先于同一 tick 的后台决策处理，因此后台 controller 不能在看到该 tick 到达请求后反向改变其 KV 命中状态。

## 6. 请求到达后的 Router

当前 router 是所有策略共用的理想化 minimum-TTFT router，不是 Preble、DualMap 或其他现有 router。

对每个容量可行节点 $n$，计算：

$$
TTFT_{s,n}
=
Q_n
+T_{s,n}^{recover}
+T_s^{prefill}.
$$

其中：

$$
Q_n=\max(0,busy\_until_n-t),
$$

router 使用的是当前真实 `busy_until`，不使用第 8.0 节的排队预测。

$$
T_s^{prefill}=\frac{B_s^{prompt}}{24}.
$$

恢复器把缺失区间按本地 host 和远端 HBM 的覆盖边界拆分。每个区间按以下可用集合选择速率最高的方法：

- 本地 host 覆盖该区间时可以 restore；
- 任一远端 HBM 连续前缀覆盖该区间时可以 transfer；
- source token 始终存在，因此可以 recompute。

router 选择 TTFT 最小的节点，平局时选择节点编号较小者，并为请求输出后的完整 context 预留 HBM。

这个 router 使用模拟器内部的准确 `busy_until` 和准确 KV 前缀，不包含元数据延迟、预测误差或前台恢复之间的链路竞争，因此结果偏理想化。

## 7. RepKV 完整计划生成

### 7.1 候选 session

只有满足以下条件的 session 才进入准备候选集：

- 已经完成至少一轮；
- 当前没有活动请求；
- 预测返回时刻位于未来 18 秒内。

即：

$$
0<\widehat t_s-t\le18.
$$

### 7.2 最小 SLO 前缀

设 session 的完整 context 为 $H_s$，节点当前 HBM 前缀为 $L_{s,n}$，计划准备到 $X_{s,n}$。准备后还需恢复区间 $[X_{s,n},H_s)$。

计划必须满足硬约束：

$$
Q_n
+T_s^{prompt}
+T_{s,n}^{recover}(X_{s,n})
\le D_s.
$$

其中 $Q_n$ 取第 8.0 节在预测返回时刻的排队估计，不是当前 `busy_until`。

计划器从 $L_{s,n}+1$ 开始逐 block 增加目标前缀，返回第一个满足约束的 $X_{s,n}$。如果节点当前已经满足 SLO，则不为该节点准备；如果 $Q_n+T_s^{prompt}>D_s$，即使完整 KV 在本地也无法满足 SLO，也不生成计划。

### 7.3 混合恢复 action

准备区间 $[L_{s,n},X_{s,n})$ 根据来源覆盖边界划分为连续区间。每个区间可选择：

- `restore`：本地 host tier 覆盖；
- `transfer`：其他节点 HBM 覆盖；
- `recompute`：利用 source token 精确重算。

计划器枚举区间级方法组合，只保留总后台执行时间不超过预测剩余时间的组合，并按单位成本选择最低成本方案。默认单位成本为：

$$
c_{restore}=0.65,
\quad
c_{transfer}=1.0,
\quad
c_{recompute}=1.35.
$$

当前完成时间按各连续区间时间求和，未模拟多个资源的流水并行。

## 8. 全局 action 决策

### 8.0 预测到达时刻的排队估计

`busy_until` 只反映节点已经接纳的工作，而准备决策针对的是数秒之后的预测返回时刻。若直接用 `busy_until` 作为 $\widehat T^{queue}_n$，控制器看不到"KV 所在节点在请求到达前变忙"这一风险，因此不会产生任何准备动作。

控制器改为在每个 tick 构造一次排队预测。所有处于空闲等待的 session 按预测返回时刻升序插入一条影子时间线；插入 session $s$ 之前记录的各节点投影值即为 $s$ 使用的 $\widehat T^{queue}_n$，因此每个 session 的排队估计只包含比它更早返回的其他 session，不包含自身。节点归属按与 router 相同的最小预测 TTFT 规则确定，写入的占用量为：

$$
q_s\left(
T_{s,n}^{recover}
+T_s^{prefill}
+\frac{\widehat B^{output}}{r^{decode}}
\right).
$$

其中 $\widehat B^{output}$ 是对已完成请求 output 长度的指数滑动平均，初值为 `assumed_output_blocks`，因此不读取未来 output 长度。

预测返回时刻已经过去的 session 不写入占用。当前预测器是单点预测，对这类 session 不再给出到达时刻，若把它们的全部服务需求记在当前时刻，会使所有节点的排队估计显著偏高。实测在默认配置下这类 session 占空闲 session 的 65%，一旦计入会把最坏节点的排队估计抬到约 5 秒，超过 2 秒的 TTFT SLO。因此当前排队预测是一个已知的偏低估计，这一偏差需要由条件返回概率模型而不是由排队模型消除。

router 在请求真实到达时仍使用准确的 `busy_until`，排队预测只影响控制器。

### 8.1 集群路由成功概率

节点预测 TTFT 被转换为平滑成功概率。TTFT 预测误差分为两部分：预测返回时刻和预测 prompt 长度的误差同时作用于所有节点，排队估计误差按节点独立。因此至少存在一个可行节点的概率为：

$$
P_s^{cluster}
=
\mathbb E_{Z}
\left[
1-\prod_n
\left(
1-
\frac{1}
{1+\exp((\widehat{TTFT}_{s,n}+\sigma_0 Z-D_s)/\sigma_n)}
\right)
\right],
$$

其中 $\sigma_0=0.45$ 为共同误差尺度，$\sigma_n=0.25$ 为节点独立误差尺度，$Z$ 为标准正态，按五个等概率分层的条件均值做数值积分。

早期版本把各节点成功事件当作完全独立，即 $P_s^{cluster}=1-\prod_n(1-p_{s,n})$。该假设在本模型中明显不成立：当一个 session 的 KV 前缀在所有节点都缺失时，各节点需要恢复的是同一段历史，TTFT 主要由同一个恢复量决定而不是由各自队列决定。完全独立假设会把四个恰好位于 SLO 边界的节点合成为 0.94 的集群成功概率，从而系统性低估准备价值。当前形式在共同误差较大时收敛到接近单节点成功概率，仍保留第二个可行节点在排队误差方向上的价值。

$\sigma_0$ 和 $\sigma_n$ 是控制器参数，不是实测的预测误差分布。

### 8.2 计划收益

为 session $s$ 在节点 $n$ 准备后的收益为：

$$
G_{s,n}
=
q_s
\left(
P_{s,n}^{after}
-P_s^{before}
\right).
$$

因此第二个可行节点仍可增加集群成功概率，而不是因为当前已有一个较好节点就被视为无价值；但在共同误差占主导时，这部分增量会明显小于把各节点当作独立时的取值。

### 8.3 资源稀缺价格

对方法 $m$，所有原始候选计划形成归一化需求：

$$
d_m
=
\sum_{s,n}
\frac{B_{s,n,m}}
{r_m^{bg}(\widehat t_s-t)}.
$$

价格为：

$$
\pi_m
=
1+\max(0,d_m-1)^2.
$$

HBM 驻留成本近似为：

$$
C_{s,n}^{hbm}
=
\frac{
(X_{s,n}-L_{s,n})(\widehat t_s-t)
}
{C_n\times18}.
$$

当集群 HBM 使用率超过 82% 时，该成本继续增大。总净收益为：

$$
V_{s,n}
=
G_{s,n}
-0.22C_{s,n}^{resource}.
$$

只有 $V_{s,n}>0$ 的候选计划可以执行。

### 8.4 排序和本周期选择

完整计划的 preparation slack 为：

$$
Slack_{s,n}
=
\widehat t_s-t-T_{s,n}^{prepare}.
$$

候选计划实际按以下顺序排序：

1. slack 更小的计划优先；
2. slack 相同时，净收益与资源成本之比更高者优先；
3. 再按 session ID 和节点 ID 确定顺序。

因此当前实现是“正净收益过滤 + 最小 slack 优先 + 收益成本比”，不是全局目标的精确优化解。

每个 tick 对同一 session 最多调度一个 batch，RepKV 最多在两个节点上保留该 session 的后台准备状态。默认每 tick 的整数预算为：restore 6 blocks、transfer 4 blocks、recompute 1 block。

## 9. 渐进执行与重新规划

controller 先生成完整目标前缀和完整方法区间，但每个 tick 只调度第一个连续区间中的一个 batch。batch 最大为 4 blocks，完成时间为：

$$
t^{complete}=t+\frac{B^{batch}}{r_m^{bg}}.
$$

调度时先预留目标节点 HBM。batch 完成时重新检查：

- 目标 HBM 前缀仍等于 batch 起点；
- restore 的本地 host 前缀仍覆盖 batch 终点；
- transfer 至少有一个远端 HBM 前缀仍覆盖 batch 终点。

任一条件失败则取消 batch；成功则把目标 HBM 前缀扩展到 batch 终点。

每个控制周期重新生成完整计划，因此这是 receding-horizon controller。它没有为完整计划预留未来所有带宽和计算资源，可能出现当前 batch 完成但后续 batch 因资源竞争无法在预测时间前完成的情况。

## 10. HBM 回收

新请求预留或后台 batch 可能触发容量回收。节点使用率超过 92% 时，目标是降至 82%。不被选为 victim 的只有两类：在该节点上有活动请求的 session，以及触发本次回收的 session 自身。

其他 session 正在进行的后台准备不受保护。若某个 session 的 HBM 前缀在其 batch 在途期间被降级，该 batch 完成时会因起点前缀不再匹配而被取消。这是 idea 中"准备完成后又被回收"这一现象在当前实现中的真实来源，不是被排除的情况。

RepKV 对一个尾部 batch 计算：

$$
Score_e
=
\frac{
q_s(P_s^{before}-P_{s,e}^{after})
+0.002B_e
}
{B_e}.
$$

分数越低，表示每释放一个 HBM block 导致的预期路由损失越小，因此越先降级。降级只缩短 `hbm_prefix`，原前缀保留在 `host_prefix` 中。

`on_demand` 和 `eager_full` 不使用该预期损失分数，而按 `last_used` 执行 LRU 降级。

当前 host tier 没有容量限制，降级也没有显式带宽和时延，因此高压场景下可能低估回收成本。

## 11. 三种策略的公平口径

三种策略在同一个 seed 上使用完全相同的请求 trace、router、前台恢复机制、节点容量和服务速率。

| 策略 | 请求前行为 |
|---|---|
| `on_demand` | 不执行后台准备 |
| `eager_full` | 渐进式构造一个完整备用前缀，不使用 RepKV 收益过滤 |
| `repkv` | 只准备能够新增硬 SLO 可行路由选项的最小前缀 |

这种比较控制了数据面假设，但共享的理想 router 和简化恢复模型也会共同影响三种策略的外部有效性。

## 12. 指标

单个 seed 的 SLO attainment 为：

$$
A_T
=
\frac{N_{success}}{N_{requests}}.
$$

SLO goodput 为：

$$
G_T
=
\frac{N_{success}}{T}.
$$

还统计：

- P99 TTFT；
- 每个成功请求对应的 transfer、restore 和 recompute blocks；
- 每个成功请求对应的 HBM block-seconds；
- 未使用准备比例；
- 降级 blocks、取消 batches 和重叠轮次。

`aggregate()` 对各 seed 的指标做简单宏平均。由于不同 seed 的请求数不同，宏平均 SLO attainment 不等于把所有请求合并后的请求级加权 attainment。正式报告必须明确采用哪一种统计口径。

## 13. 已知实现限制

当前结果不能直接支撑真实系统结论，原因包括：

1. workload 是非稳态合成 session 流，不是固定 QPS，也不是公开 trace；默认 4 节点 48 session 配置下节点利用率很低，几乎不产生排队竞争；
2. predictor 是按预设类别生成的合成单点预测器，没有条件返回概率、训练、校准和线上推理开销；
3. router 使用准确队列和准确 KV 元数据；
4. 前台恢复只增加 TTFT，不占用共享网络或计算队列；
5. host tier 容量、写入成本和降级成本被忽略，且降级后的前缀永久保留在 host tier，之后可以按 28 blocks/s 免费 restore，因此 HBM 竞争的代价被系统性低估；
6. 排队预测忽略预测返回时刻已经过去的 session，是已知的偏低估计；
7. 完整计划不预留未来资源；
8. block 没有对应真实 token 数、层、dtype、张量地址和 block table；
9. 没有模拟 RDMA、CANN stream、NPU kernel、NUMA 和跨机故障；
10. 默认速率和成本权重没有由 Ascend 910B 测量校准；
11. 在 2 节点、长 context 的配置下 `_arrive` 的容量预检与实际预留会发生冲突并触发 `node capacity exceeded` 断言。该问题在不执行任何准备的 `on_demand` 策略上同样出现，属于容量预留逻辑本身的缺陷，尚未修复。

## 14. 下一阶段实现顺序

当前最主要的控制面缺陷是准备与回收在时间上脱耦。实测在 4 节点 96 session 配置下，71% 的 SLO 失败请求在到达前 18 秒内曾经在某个节点持有完整 HBM 前缀，即 KV 原本存在、被回收、到达时再付一次完整 restore。控制器无法用准备动作弥补，因为在准备窗口内前缀仍然驻留，看不出需要准备；等到前缀被回收后，session 往往已经过了预测返回时刻，不再进入准备队列。idea 要求准备与回收由同一个目标决定，当前实现只共享打分公式，不共享时间轴。

下一阶段不应继续无限增加模拟器机制，而应依次完成：

1. 用固定 offered QPS 和真实多轮会话 trace 替换当前非稳态 workload，使节点利用率进入会产生排队竞争的区间；
2. 用 Ascend 实测 prefill、decode、host restore 和跨机传输数据校准参数；
3. 在真实 runtime 中验证 KV block 导出、连续前缀传输和导入正确性；
4. 建立 router metadata 接口并测量状态延迟；
5. 测量前后台并发干扰，替换当前固定资源比例；
6. 最后再验证 RepKV controller 的 goodput、资源成本和失败模式。
