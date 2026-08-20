# RepKV 控制面原型实现说明

## 1. 实现定位

当前项目是离散时间控制面模拟器。它回答的是"RepKV 的状态、计划和在线决策是否能以因果一致、资源有界的方式运行"，不是"真实 Ascend 集群能达到多少吞吐"。

模拟器把真实 KV 张量压缩为连续 block 前缀，把节点推理负载压缩为 `busy_until`，把网络、本地恢复和后台计算压缩为 blocks/s。该抽象足以检查计划正确性、资源竞争和策略相对趋势，但不能推导绝对硬件性能。

代码分为五部分：

| 文件 | 职责 |
|---|---|
| `repkv_sim/planner.py` | 纯函数形式的前缀区间划分、恢复规划、最小 SLO 前缀求解和可路由概率 |
| `repkv_sim/predictor.py` | 类条件返回概率模型：窗口内返回概率和条件中位剩余等待 |
| `repkv_sim/simulator.py` | workload、节点和 session 状态、router、RepKV controller、回收及指标 |
| `run.py` | 多 seed 批量比较和 CSV 输出 |
| `trace.py` | 逐 tick 展示完整状态的交互式终端界面 |

## 2. 默认系统参数

| 参数 | 数值 | 含义 |
|---|---:|---|
| `nodes` | 4 | 节点数 |
| `sessions` | 600 | session 脚本池大小 |
| `concurrent_sessions` | 0 | 并发存活 session 数；0 表示池内全部按各自起始时间进入 |
| `horizon_s` | 240 s | 单个 seed 的模拟时长 |
| `step_s` | 0.5 s | 控制周期 |
| `hbm_blocks` | 180 | 每节点 HBM block 容量 |
| `ttft_slo_s` | 2.0 s | TTFT SLO |
| `prefill_blocks_s` | 24 | 新 prompt prefill 速率 |
| `decode_blocks_s` | 10 | output decode 速率 |
| `transfer_blocks_s` | 20 | 前台远端恢复速率 |
| `host_restore_blocks_s` | 28 | 前台本地 host 恢复速率 |
| `recompute_blocks_s` | 12 | 前台精确重算速率 |
| `preparation_horizon_s` | 18 s | 返回概率与准备时限使用的窗口 $\Delta$ |
| `block_batch` | 4 | 单次后台 action 最大 block 数 |
| `high_watermark` | 0.92 | HBM 回收触发水位 |
| `low_watermark` | 0.82 | HBM 回收目标水位 |
| `arrival_forecast` | True | 是否用预测到达构造未来排队估计 |
| `shared_uncertainty_s` | 0.45 s | 各节点共同承受的 TTFT 预测误差 |
| `queue_uncertainty_s` | 0.25 s | 各节点独立的排队估计误差 |
| `assumed_output_blocks` | 4.0 | output 长度估计的初值 |
| `think_sigma` | 0.55 | 真实思考时间的对数正态尺度 |
| `think_scale` | 1.0 | 思考时间整体缩放，用于调节负载 |
| `predictor_sigma_scale` | 1.0 | 预测器使用的尺度相对真实值的比例 |
| `min_return_probability` | 0.05 | 进入准备候选集所需的最低窗口返回概率 |

`run.py` 的默认实验配置与 `Config` 默认值不同：它使用 `sessions=600`、`concurrent_sessions=40`、`hbm_blocks=520`，因为 `Config` 默认值不产生排队竞争（见第 13 节第 1 条）。

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

| 类别 | 思考时间中位数 | 初始继续概率 |
|---|---:|---:|
| fast | 12 s | 0.88 |
| normal | 22 s | 0.78 |
| slow | 40 s | 0.66 |

### 3.2 closed-loop 到达

第 $k+1$ 轮的到达时刻由第 $k$ 轮的**完成时刻**决定：

$$
A_{s,k+1}=C_{s,k}+\Theta_{s,k+1}.
$$

其中 $C_{s,k}$ 是第 $k$ 轮请求返回的时刻，$\Theta_{s,k+1}$ 是该轮之后的思考时间。用户在收到上一轮回答之前不会发出下一轮请求，因此一个 session 不会同时存在两个在途请求。

思考时间的分布为：

$$
\Theta_{s,k}
=
\operatorname{clip}
\left(
m_c\exp(Z),6,75
\right),
\qquad
Z\sim\mathcal N(0,0.55^2).
$$

早期实现按 $A_{s,k+1}=A_{s,k}+\Theta$ 排定到达，即从上一轮的**到达**时刻起算，忽略服务耗时。这会生成物理上不可能的到达序列：负载较高时下一轮可能在上一轮返回之前到达。由此产生的两个并发请求会共用 `active_reservations` 中同一个以 session 为键的条目，先完成的请求释放该条目后，后完成的请求把未经预留的前缀写入 HBM，触发容量断言。

由于到达时刻依赖完成时刻，不同策略的到达序列不再完全相同（见第 11 节）。

### 3.3 轮数与右删失

第 $k$ 轮之后继续的概率为：

$$
p_{c,k}=\max(0.18,p_{c,0}-0.045k).
$$

是否继续由 Bernoulli 抽样决定，每个 session 的上限轮数在 3–8 之间随机选择。因此部分 session 提前结束，对预测器构成右删失样本。

第一轮 prompt 为 8–16 blocks 均匀分布，后续 prompt 为 3–9 blocks，output 为 2–7 blocks。历史 context 是此前所有 prompt 和 output 的累计长度。

### 3.4 并发补充

当 `concurrent_sessions` 大于 0 时，模拟器先放入该数量的 session，并在一个 session 的脚本结束时立即从池中放入下一个未使用的 session。

不做补充时，workload 的总工作量由 `sessions` 与每个 session 的轮数上限固定。此时缩短思考时间只会把同样的请求提前，不能提高稳态负载：实测把 `think_scale` 从 1.0 降到 0.25，48 session 的请求总数始终是 153，节点利用率停在 16% 附近。加入补充后，负载由存活并发数决定，利用率可覆盖 47% 到 84%。

### 3.5 预测器可见的信息

预测器只接收 session 类别对应的分布参数（思考时间中位数、尺度、继续概率、上下界），不接收任何实际抽样值。它不知道每个 session 的轮数上限，因此"不再返回"这一事件始终是预测误差的来源。

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

排队中的每个请求都按其完整 context 预留 HBM，因此并发度提高时大量 HBM 被尚未执行的请求占据。这是当前实现能承载的并发上限的主要来源，而不是一个已验证的建模选择。

### 4.3 SessionState

session 状态包括已完成轮次、最新完整 context 长度、当前活动请求结束时间、执行节点，以及最近一次请求完成的时刻 `idle_since`。活动请求在其执行节点上的 KV 不允许被回收。

`idle_since` 给出预测器需要的已等待时长：

$$
\tau_s=t-idle\_since_s.
$$

## 5. 每个控制周期的事件顺序

每个 0.5 秒 tick 按以下顺序执行：

1. 完成到期的后台 batch；
2. 完成到期的前台请求，并按完成时刻排定该 session 的下一轮或补充新 session；
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
\qquad
T_s^{prefill}=\frac{B_s^{prompt}}{24}.
$$

router 使用的是当前真实 `busy_until`，不使用第 8.1 节的排队预测。

恢复器把缺失区间按本地 host 和远端 HBM 的覆盖边界拆分。每个区间按以下可用集合选择速率最高的方法：

- 本地 host 覆盖该区间时可以 restore；
- 任一远端 HBM 连续前缀覆盖该区间时可以 transfer；
- source token 始终存在，因此可以 recompute。

router 选择 TTFT 最小的节点，平局时选择节点编号较小者，并为请求输出后的完整 context 预留 HBM。

这个 router 使用模拟器内部的准确 `busy_until` 和准确 KV 前缀，不包含元数据延迟、预测误差或前台恢复之间的链路竞争，因此结果偏理想化。

## 7. 返回预测

### 7.1 窗口返回概率

对已经等待 $\tau_s$ 且尚未返回的 session，窗口 $\Delta$ 内返回的概率为：

$$
q_s(t,\Delta)
=
\frac{p_{c,k}\left(F_c(\tau_s+\Delta)-F_c(\tau_s)\right)}
{1-p_{c,k}F_c(\tau_s)}.
$$

其中 $F_c$ 是类别 $c$ 的思考时间分布，$p_{c,k}$ 是第 $k$ 轮之后继续的概率。分子分母中的 $p_{c,k}$ 表示"可能不再返回"这一质量：等待越久既说明返回越近，也说明可能根本不会返回，两种效应同时体现在该式中。

### 7.2 用于评估的时刻

节点队列和 TTFT 需要一个具体时刻。当前使用条件中位剩余等待：

$$
\widehat t_s=t+\delta_s,
\qquad
F_c(\tau_s+\delta_s)=F_c(\tau_s)+\frac{1-F_c(\tau_s)}{2}.
$$

$\delta_s$ 按 CDF 二分求解，恒非负，因此 $\widehat t_s$ 不会落在过去。等待时间超过中位思考时间的 session 被判定为"即将返回"，而不是"错过了某个时刻"。

早期实现使用单点预测 $\widehat t_s=A_{s,k}+\widehat\Theta$，并在两处把"当前时刻已超过 $\widehat t_s$"当作硬开关：准备侧以 `time_left <= 0` 跳过该 session，回收侧对该 session 返回固定最低分 0.0001。默认配置下任意时刻有 65% 的空闲 session 处于该状态，两处硬开关把同一条信息解释为相反的含义——准备侧视为不再相关，回收侧视为没有价值。实测该分数低于任何真实边际成本（最便宜的一刀为 0.00148），因此这些 session 的前缀会被优先剥空：占降级事件的 41.9%，但占降级 block 的 67.7%，平均每次被取走 19 个 block，而按时 session 只有 6.6 个。改用 $q_s(t,\Delta)$ 之后，两侧使用同一个概率，该偏差消失（第 12 节）。

## 8. RepKV 完整计划生成

### 8.1 预测到达时刻的排队估计

`busy_until` 只反映节点已经接纳的工作，而准备决策针对的是数秒之后的预测返回时刻。若直接用 `busy_until` 作为 $\widehat T^{queue}_n$，控制器看不到"KV 所在节点在请求到达前变忙"这一风险，因此不会产生任何准备动作。

控制器每个 tick 构造一次排队预测。所有处于空闲等待的 session 按 $\widehat t_s$ 升序插入一条影子时间线；插入 session $s$ 之前记录的各节点投影值即为 $s$ 使用的 $\widehat T^{queue}_n$，因此每个 session 的排队估计只包含比它更早返回的其他 session，不包含自身。节点归属按与 router 相同的最小预测 TTFT 规则确定，写入的占用量为：

$$
q_s\left(
T_{s,n}^{recover}
+T_s^{prefill}
+\frac{\widehat B^{output}}{r^{decode}}
\right).
$$

其中 $\widehat B^{output}$ 是对已完成请求 output 长度的指数滑动平均，初值为 `assumed_output_blocks`，因此不读取未来 output 长度。

router 在请求真实到达时仍使用准确的 `busy_until`，排队预测只影响控制器。

### 8.2 候选 session

只有满足以下条件的 session 才进入准备候选集：

- 已经完成至少一轮且当前没有活动请求；
- 脚本中仍存在下一轮，因此存在预测；
- $q_s(t,\Delta)\ge$ `min_return_probability`。

### 8.3 最小 SLO 前缀

设 session 的完整 context 为 $H_s$，节点当前 HBM 前缀为 $L_{s,n}$，计划准备到 $X_{s,n}$。准备后还需恢复区间 $[X_{s,n},H_s)$。计划必须满足硬约束：

$$
Q_n
+T_s^{prompt}
+T_{s,n}^{recover}(X_{s,n})
\le D_s.
$$

其中 $Q_n$ 取第 8.1 节在 $\widehat t_s$ 的排队估计，不是当前 `busy_until`。

计划器从 $L_{s,n}+1$ 开始逐 block 增加目标前缀，返回第一个满足约束的 $X_{s,n}$。如果节点当前已经满足 SLO，则不为该节点准备；如果 $Q_n+T_s^{prompt}>D_s$，即使完整 KV 在本地也无法满足 SLO，也不生成计划。

### 8.4 混合恢复 action

准备区间 $[L_{s,n},X_{s,n})$ 根据来源覆盖边界划分为连续区间。每个区间可选择 `restore`（本地 host tier 覆盖）、`transfer`（其他节点 HBM 覆盖）或 `recompute`（利用 source token 精确重算）。

计划器枚举区间级方法组合，只保留总后台执行时间不超过 $\widehat t_s-t$ 的组合，并按单位成本选择最低成本方案。默认单位成本为：

$$
c_{restore}=0.65,
\quad
c_{transfer}=1.0,
\quad
c_{recompute}=1.35.
$$

这些是固定常量，不是第 9.3 节的实时稀缺价格，因此计划内部的成本排序与全局选择使用的价格并不是同一套。

当前完成时间按各连续区间时间求和，未模拟多个资源的流水并行。

## 9. 全局 action 决策

### 9.1 集群可路由概率

TTFT 预测误差分为两部分：预测返回时刻和预测 prompt 长度的误差同时作用于所有节点，排队估计误差按节点独立。因此至少存在一个可行节点的概率为：

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

早期版本把各节点成功事件当作完全独立。该假设在本模型中明显不成立：当一个 session 的 KV 前缀在所有节点都缺失时，各节点需要恢复的是同一段历史，TTFT 主要由同一个恢复量决定而不是由各自队列决定。完全独立假设会把四个恰好位于 SLO 边界的节点合成为 0.94 的集群成功概率，从而系统性低估准备价值。

$\sigma_0$ 和 $\sigma_n$ 是控制器参数，不是实测的预测误差分布。

### 9.2 计划收益

为 session $s$ 在节点 $n$ 准备后的收益为：

$$
G_{s,n}
=
q_s(t,\Delta)
\left(
P_{s,n}^{after}
-P_s^{before}
\right).
$$

因此第二个可行节点仍可增加集群可路由概率；但在共同误差占主导时，这部分增量会明显小于把各节点当作独立时的取值。

### 9.3 资源稀缺价格

对方法 $m$，所有原始候选计划形成归一化需求，价格随需求超过 1 后二次增长：

$$
d_m
=
\sum_{s,n}
\frac{B_{s,n,m}}
{r_m^{bg}(\widehat t_s-t)},
\qquad
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
{C_n\Delta}.
$$

当集群 HBM 使用率超过 82% 时，该成本继续增大。总净收益为：

$$
V_{s,n}
=
G_{s,n}
-0.22C_{s,n}^{resource}.
$$

只有 $V_{s,n}>0$ 的候选计划可以执行。三类资源在这里被合并为单一标量并共用 0.22 的权重，方法之间的差异只体现在归一化分母和价格 $\pi_m$ 上，没有实现相互独立的 $\lambda_B$、$\lambda_U$、$\lambda_H$。

### 9.4 排序和本周期选择

完整计划的 preparation slack 为：

$$
Slack_{s,n}
=
\widehat t_s-t-T_{s,n}^{prepare}.
$$

候选计划按 slack 升序、净收益与资源成本之比降序、再按 session ID 和节点 ID 排序。因此当前实现是"正净收益过滤 + 最小 slack 优先 + 收益成本比"，不是全局目标的精确优化解。

每个 tick 对同一 session 最多调度一个 batch，RepKV 最多在两个节点上保留该 session 的后台准备状态。默认每 tick 的整数预算为：restore 6 blocks、transfer 4 blocks、recompute 1 block。

## 10. 渐进执行与重新规划

controller 先生成完整目标前缀和完整方法区间，但每个 tick 只调度第一个连续区间中的一个 batch。batch 最大为 4 blocks，完成时间为：

$$
t^{complete}=t+\frac{B^{batch}}{r_m^{bg}}.
$$

调度时先预留目标节点 HBM。batch 完成时重新检查目标 HBM 前缀仍等于 batch 起点、restore 的本地 host 前缀仍覆盖 batch 终点、transfer 至少有一个远端 HBM 前缀仍覆盖 batch 终点。任一条件失败则取消 batch。

每个控制周期重新生成完整计划，因此这是 receding-horizon controller。方案对象不持久化，"方案来不及完成或净价值转负则停止后续 batch"是靠每个 tick 重新规划隐式实现的。代价是没有达到 SLO 门槛的半程前缀会留在 HBM 中，当前没有回滚机制。它也没有为完整计划预留未来所有带宽和计算资源。

## 11. HBM 回收

新请求预留或后台 batch 可能触发容量回收。节点使用率超过 92% 时，目标是降至 82%。不被选为 victim 的只有两类：在该节点上有活动请求的 session，以及触发本次回收的 session 自身。

其他 session 正在进行的后台准备不受保护。若某个 session 的 HBM 前缀在其 batch 在途期间被降级，该 batch 完成时会因起点前缀不再匹配而被取消。

RepKV 对一个尾部 batch 计算：

$$
Score_e
=
\frac{
q_s(t,\Delta)\left(P_s^{before}-P_{s,e}^{after}\right)
+0.002B_e
}
{B_e}.
$$

分数越低，表示每释放一个 HBM block 导致的预期路由损失越小，因此越先降级。脚本中已无下一轮的 session 没有预测，分数为 0，可以自由回收。降级只缩短 `hbm_prefix`，原前缀保留在 `host_prefix` 中。

该分数与第 9.2 节的准备收益是同一个状态函数。实测把 40 个 block 的前缀从零建起来的收益为 0.24087，而按每次 4 个 block 从 40 削到 0 的十次损失之和为 0.23665，差值来自 $\max(0,\cdot)$ 的截断。因此在不同时刻分别计算不构成口径不一致。该分数对前缀长度是强凸的：40 到 0 的逐步损失依次为 0.0015、0.0028、0.0049、0.0085、0.0140、0.0218、0.0313、0.0415、0.0510、0.0594，最外层一刀比最内层便宜约 40 倍。

`on_demand` 和 `eager_full` 不使用该预期损失分数，而按 `last_used` 执行 LRU 降级。

实现上没有维护回收优先队列，而是每次回收都对节点内所有 replica 重新求最小分数。当前 host tier 没有容量限制，降级也没有显式带宽和时延，而且降级后的前缀永久保留在 host tier，因此高压场景下会低估回收成本。也没有实现"删除 KV 但保留 token"的分支。

分数中的 $0.002B_e$ 只代表降级操作本身的开销，不包含在 $\widehat t_s$ 之前重新准备该区间的代价。因此回收侧不会为"稍后需要重新搬回来"付费。

## 12. 三种策略的公平口径

三种策略在同一个 seed 上使用同一份 session 脚本：prompt 长度、output 长度、思考时间和继续与否的抽样完全相同。

由于到达时刻由完成时刻决定，服务更快的策略会更早获得下一轮，因此三种策略的到达序列不再逐位相同，完成的请求总数也不同。比较必须同时报告请求数、attainment 和 goodput：只看 attainment 会让完成更少请求、承受更低负载的策略显得更好。

| 策略 | 请求前行为 | 回收 |
|---|---|---|
| `on_demand` | 不执行后台准备 | LRU |
| `eager_full` | 渐进式构造一个完整备用前缀，不使用 RepKV 收益过滤 | LRU |
| `repkv` | 只准备能够新增硬 SLO 可行路由选项的最小前缀 | 预期 SLO 损失分数 |

准备策略和回收策略在这里是绑定的，因此当前配置无法单独隔离两者的贡献；$2\times2$ 消融需要额外的策略组合。

## 13. 指标

单个 seed 的 SLO attainment 与 goodput 为：

$$
A_T=\frac{N_{success}}{N_{requests}},
\qquad
G_T=\frac{N_{success}}{T}.
$$

还统计 P99 TTFT、每个成功请求对应的 transfer/restore/recompute blocks、每个成功请求对应的 HBM block-seconds、未使用准备比例、降级 blocks、取消 batches 和重叠轮次。

`aggregate()` 对各 seed 的指标做简单宏平均。由于不同 seed 的请求数不同，宏平均 attainment 不等于把所有请求合并后的请求级加权 attainment。

## 14. 已知实现限制

1. workload 是合成的，不是公开 trace；`Config` 默认值（4 节点、48 session、不做补充）节点利用率约 21%，不产生排队竞争，任何预放置策略在该配置上都无法被区分；
2. 预测器接收真实的类别参数，只是不知道抽样值和轮数上限；没有训练、校准和线上推理开销；
3. router 使用准确队列和准确 KV 元数据，也没有 Preble-style 或 DualMap-style 变体，因此"可叠加到不同 router"未被验证；
4. `readiness_snapshot()` 只被 `trace.py` 调用，router 实际直接读内部状态，元数据接口没有延迟或陈旧；
5. 前台恢复只增加 TTFT，不占用共享网络或计算队列；
6. host tier 容量、写入成本和降级成本被忽略，且降级后的前缀永久保留在 host tier，因此 HBM 竞争的代价被系统性低估；
7. 回收分数不含重新准备的代价，准备与回收仍未在资源账目上闭环；
8. 排队中的请求按完整 context 预留 HBM，这限制了可模拟的并发度，且不是经过验证的建模选择；
9. 完整计划不预留未来资源，未达门槛的半程前缀没有回滚；
10. block 没有对应真实 token 数、层、dtype、张量地址和 block table；
11. 没有模拟 RDMA、CANN stream、NPU kernel、NUMA 和跨机故障；
12. 默认速率和成本权重没有由 Ascend 910B 测量校准。

## 15. 下一阶段实现顺序

1. 在回收分数中加入在 $\widehat t_s$ 之前重新准备的预计代价，使准备与回收在资源账目上闭环；需要先确定"确实需要重建"的概率如何估计；
2. 用公开多轮会话 trace 替换合成脚本，并用实测思考时间分布替换对数正态假设；
3. 用 Ascend 实测 prefill、decode、host restore 和跨机传输数据校准参数；
4. 在真实 runtime 中验证 KV block 导出、连续前缀传输和导入正确性；
5. 建立 router metadata 接口并测量状态延迟；
6. 测量前后台并发干扰，替换当前固定资源比例；
7. 增加 $2\times2$ 消融所需的策略组合，分离准备与回收的贡献。
