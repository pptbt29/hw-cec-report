# RepKV 控制面原型实现说明

## 1. 实现定位

当前项目是离散时间控制面模拟器。它回答的是"RepKV 的状态、计划和在线决策是否能以因果一致、资源有界的方式运行"，不是"真实 Ascend 集群能达到多少吞吐"。

模拟器把真实 KV 张量压缩为连续 block 前缀，把节点压缩为三条串行资源时间线加一组 decode 槽位。所有速率由模型形状和硬件指标推导，不直接给定 blocks/s，因为三条恢复路径之间的比例关系是每个放置决策的依据。该抽象足以检查计划正确性、资源竞争和策略相对趋势，但不能推导绝对硬件性能。

代码分为六部分：

| 文件 | 职责 |
|---|---|
| `repkv_sim/resources.py` | 由模型形状与硬件指标推导 block 字节数、各操作速率、显存容量与 decode 槽位数 |
| `repkv_sim/planner.py` | 纯函数形式的前缀区间划分、恢复规划、最小 SLO 前缀求解和可路由概率 |
| `repkv_sim/predictor.py` | 类条件返回概率模型：窗口内返回概率和条件中位剩余等待 |
| `repkv_sim/simulator.py` | workload、节点和 session 状态、router、RepKV controller、回收及指标 |
| `run.py` | 多 seed 批量比较和 CSV 输出 |
| `trace.py` | 逐 tick 展示完整状态的交互式终端界面 |

## 2. 资源推导与默认参数

### 2.1 模型形状

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `layers` | 80 | 层数 |
| `kv_heads` | 8 | KV 头数（分组查询注意力）|
| `head_dim` | 128 | 头维度 |
| `parameters` | 70e9 | 参数量 |
| `dtype_bytes` | 2 | 权重元素字节数 |
| `kv_dtype_bytes` | 2 | KV 元素字节数，与权重分开以表达 KV 量化 |
| `block_tokens` | 16 | 每 block 的 token 数 |

每 token 的 KV 字节数为 $b = L\cdot2\cdot H_{kv}\cdot d_{head}\cdot s_{kv}$，默认 320 KiB；一个 block 为 5.00 MiB；权重为 130 GiB。

### 2.2 硬件指标

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `devices` | 8 | 张量并行组的设备数 |
| `device_flops` | 3.2e14 | 单设备峰值算力 |
| `achieved_utilisation` | 0.42 | prefill 的达成率 |
| `hbm_bytes_s` | 1.6e12 | 单设备显存带宽 |
| `hbm_capacity_bytes` | 64 GiB | 单设备显存容量 |
| `network_bytes_s` | 12.5e9 | 节点间可用带宽 |
| `host_link_bytes_s` | 20.0e9 | 到 host DRAM 的带宽 |
| `host_capacity_bytes` | 1.5 TiB | DRAM 层给 KV 的容量 |
| `decode_batch` | 16 | 批大小上限 |
| `decode_context_tokens` | 30000 | 标定 decode 速率与槽位数的工作点 |
| `cache_reserve_fraction` | 0.35 | 从运行批次中留给前缀缓存的显存比例 |

### 2.3 推导出的速率与容量

$$R_{\text{prefill}} = R_{\text{recompute}} = \frac{F_{\text{eff}}}{2P\,t_{blk}},\qquad
R_{\text{transfer}} = \frac{B_{\text{net}}}{b\,t_{blk}},\qquad
R_{\text{restore}} = \frac{B_{\text{host}}}{b\,t_{blk}}$$

重算与 prefill 同速率，因为重建 KV 就是对同一段 token 重跑一次前向。默认值下每 token 折算：prefill 与重算 7680/s，传输 38147/s，restore 61035/s。也就是说重算比搬运同一段 KV 慢 10 到 16 倍，这个比例由字节数与 FLOPs 决定而不是由参数设定。

decode 每序列速率由一次 step 的读取量给出：一次 step 读一遍全部权重再读批内每条序列的 KV，批内每条序列推进一个 token。默认值下 43 tok/s，step 23.2 ms。

显存容量为总容量减去权重与激活预留后能装的 block 数，默认每节点 69766 block，即 1.12M token。槽位数为 $\min(n_{\text{batch}},\ \lfloor(1-\rho)H / (C_{\text{op}}/t_{blk})\rfloor)$：批内每条序列都要驻留完整上下文，因此显存而不是批大小设置才是长上下文下的真实上限。$\rho=0$ 的部署会让运行批次吃掉全部显存，此时任何放置策略在物理上无效。

方法代价定义为每 block 消耗的资源秒 $1/R_m$，因此规划器最小化的是前台要争的那个量，而不是人为的偏好顺序。recompute 与跨节点 transfer 是两个对等动作。各自在 $\beta$ 内仍能使 TTFT 满足 SLO 的最长 context 为 $C_m=\beta R_m t_{blk}$，再按长短排序：

$$C_1=\min(C_{\text{recompute}},C_{\text{transfer}}),\qquad
C_2=\max(C_{\text{recompute}},C_{\text{transfer}}).$$

$C_1\le C_2$ 恒成立。网快时 $C_2$ 是 transfer；网慢到 transfer 比 recompute 还慢时两者对调，中间区间仍然存在，只是剩下的动作变成 recompute。本机 restore 是第三个动作，有自己的 $C_{\text{restore}}$。`ResourceModel.thresholds` 同时返回各动作长度和排好序的 $C_1$、$C_2$。

### 2.4 控制与 workload 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `nodes` | 4 | 节点数 |
| `sessions` | 48 | session 脚本池大小 |
| `concurrent_sessions` | 0 | 并发存活 session 数；0 表示池内全部按各自起始时间进入 |
| `horizon_s` | 900 s | 单个 seed 的模拟时长 |
| `step_s` | 1.0 s | 控制周期 |
| `ttft_slo_s` | 3.0 s | TTFT SLO |
| `hbm_blocks_override` | 0 | 非零时覆盖推导出的显存容量 |
| `preparation_horizon_s` | 60 s | 返回概率与准备时限使用的窗口 $\Delta$ |
| `block_batch` | 128 | 单次后台 action 最大 block 数 |
| `high_watermark` | 0.92 | 显存回收触发水位 |
| `low_watermark` | 0.82 | 显存回收目标水位 |
| `arrival_forecast` | True | 是否用预测到达构造未来排队估计 |
| `shared_uncertainty_s` | 0.45 s | 各节点共同承受的 TTFT 预测误差下限 |
| `queue_uncertainty_s` | 0.25 s | 各节点独立的排队估计误差下限 |
| `shared_uncertainty_scale` | 0.5 | 共同误差随各节点预测排队均值增长的系数 |
| `queue_uncertainty_scale` | 0.8 | 独立误差随该节点预测排队增长的系数 |
| `reprepare_cost_weight` | 1.0 | 回收分数中重建代价项的权重；0 表示关闭 |
| `spare_replica_factor` | 0.01 | 另一节点仍能满足预测 SLO 时，降级地板的折扣；1 表示不折扣 |
| `displacement_cost_weight` | 1.0 | 准备为它挤掉的降级付费的权重；0 表示关闭 |
| `value_based_eviction` | True | 回收受害者按价值排序；False 退回 LRU，用于消融 |
| `first_prompt_tokens` | 4000 | 首轮 prompt 的 token 中位数 |
| `follow_prompt_tokens` | 3000 | 后续轮 prompt 的 token 中位数 |
| `output_tokens` | 1500 | output 的 token 中位数 |
| `size_sigma` | 0.30 | 尺寸的对数正态尺度 |
| `think_sigma` | 0.55 | 真实思考时间的对数正态尺度 |
| `think_scale` | 1.0 | 思考时间整体缩放，用于调节负载 |
| `think_floor_s` / `think_ceiling_s` | 15 / 900 s | 思考时间截断 |
| `predictor_sigma_scale` | 1.0 | 预测器使用的尺度相对真实值的比例 |
| `min_return_probability` | 0.05 | 进入准备候选集所需的最低窗口返回概率 |

`run.py` 的默认实验配置对应 `PROTOTYPE_VERDICT.md` 第 6.1 节的数据中心主配置：4 节点、存活并发 90、上下文 32000 token、SLO 3.0 秒、DRAM 层 256 GiB。该配置位于 $C_1<C<C_2$ 且较快动作是搬迁的区间（报告里称取回可行、重算不可行），三种策略在此分开。`--ttft-slo` 可以跨过两个门槛，`--network-gbytes` 与 `--host-gib` 可以移动各动作的可行长度从而让 $C_1$、$C_2$ 对调。`--lru-eviction` 把回收退回最近使用时间，`--min-return-probability` 提到 1 以上则关闭全部准备，两者合起来构成准备/回收的 $2\times2$ 消融。不做 session 补充时不产生排队竞争（见第 14 节第 1 条）。

后台工作没有固定的速率比例，它与前台以相同的物理速率运行并占用同一条资源时间线，只在该资源于当前控制周期内仍有空闲时才被派发。共享由竞争决定而不是由常数决定。

## 3. Workload 模型

### 3.1 Session 类别

每个 session 以相同概率属于三类之一：

| 类别 | 思考时间中位数 | 初始继续概率 |
|---|---:|---:|
| fast | 45 s | 0.88 |
| normal | 120 s | 0.78 |
| slow | 300 s | 0.66 |

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
m_c\exp(Z),\ \underline{\Theta},\ \overline{\Theta}
\right),
\qquad
Z\sim\mathcal N(0,0.55^2).
$$

其中 $\underline{\Theta}$ 与 $\overline{\Theta}$ 为 `think_floor_s` 与 `think_ceiling_s`，默认 15 秒与 900 秒，并随 `think_scale` 一起缩放。

早期实现按 $A_{s,k+1}=A_{s,k}+\Theta$ 排定到达，即从上一轮的**到达**时刻起算，忽略服务耗时。这会生成物理上不可能的到达序列：负载较高时下一轮可能在上一轮返回之前到达。由此产生的两个并发请求会共用 `active_reservations` 中同一个以 session 为键的条目，先完成的请求释放该条目后，后完成的请求把未经预留的前缀写入 HBM，触发容量断言。

由于到达时刻依赖完成时刻，不同策略的到达序列不再完全相同（见第 11 节）。

### 3.3 轮数与右删失

第 $k$ 轮之后继续的概率为：

$$
p_{c,k}=\max(0.18,p_{c,0}-0.045k).
$$

是否继续由 Bernoulli 抽样决定，每个 session 的上限轮数在 3–8 之间随机选择。因此部分 session 提前结束，对预测器构成右删失样本。

第一轮 prompt、后续 prompt 和 output 均按 token 的对数正态分布抽样，中位数分别为 `first_prompt_tokens`、`follow_prompt_tokens` 和 `output_tokens`，尺度为 `size_sigma`，再向上取整到 block。`run.py` 把首轮中位数设为 `--context-tokens`，使后续轮要恢复的历史长度落在所报告的门槛附近。历史 context 是此前所有 prompt 和 output 的累计长度。

### 3.4 并发补充

当 `concurrent_sessions` 大于 0 时，模拟器先放入该数量的 session，并在一个 session 的脚本结束时立即从池中放入下一个未使用的 session。

不做补充时，workload 的总工作量由 `sessions` 与每个 session 的轮数上限固定。此时缩短思考时间只会把同样的请求提前，不能提高稳态负载。加入补充后，负载由存活并发数决定，物理资源模型下的直接约束是 decode 槽位而不是一条串行设备队列。

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

HBM 降级时会先把被削掉的前缀写入 DRAM 层，因此降级刚完成时通常仍有 $L_{hbm}\le L_{host}$。DRAM 层容量有限，写入时按最近使用时间淘汰其他 session 的前缀，淘汰后 $L_{host}$ 可以短于 $L_{hbm}$。这一点必须建模：若该层永不遗忘，则最后服务过该 session 的节点上 restore 永远可用，恢复永远廉价，KV 放置在构造上就不可能有影响。写入该层的带宽仍未计入，因此 host 副本成本被低估。

### 4.2 Node

节点维护：

- `until`：三条串行资源各自的空闲时刻。设备计算承载 prefill 与重算，网卡承载远端传输，host 链路承载从 DRAM 的 restore。前台与后台占用同一条时间线；
- `decode_finish`：各已占用 decode 槽位的完成时刻。连续批处理下已 prefill 的请求加入运行批次，批内每条序列每步推进一个 token，因此 decode 不串行在设备时间线上而是占用固定数量的槽位；
- `replicas`：各 session 的 KV 前缀；
- `active_reservations`：活动请求完成其 context 所需的 HBM 预留；
- `background_reserved`：在途后台 batch 已预留但尚未写入的 HBM。

`busy_until` 是 `until["compute"]` 的别名，表示新排队请求可以开始 prefill 的时刻。

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

每个 1.0 秒 tick 按以下顺序执行：

1. 完成到期的后台 batch；
2. 完成到期的前台请求，并按完成时刻排定该 session 的下一轮或补充新 session；
3. 处理当前时刻真实到达的请求；
4. 对尚未到达的预测请求执行后台准备决策；
5. 累计 HBM block-seconds；
6. 检查状态不变量。

真实请求先于同一 tick 的后台决策处理，因此后台 controller 不能在看到该 tick 到达请求后反向改变其 KV 命中状态。

## 6. 请求到达后的 Router

当前 router 是所有策略共用的理想化 minimum-TTFT router，不是 Preble、DualMap 或其他现有 router。

对每个容量可行节点 $n$，恢复区间依次占用各自的资源，随后 prefill 需要设备与一个空闲 decode 槽位：

$$
TTFT_{s,n}
=
\max\left(T_{s,n}^{recover},\ T_n^{compute},\ T_n^{slot}\right)
+\frac{B_s^{prompt}}{R_{\text{prefill}}}.
$$

恢复与设备排队取最大值而不是相加：只用网卡或 host 链路的恢复可以在设备为别的请求做 prefill 时并行进行。$T_{s,n}^{recover}$ 本身是各区间在其资源上排队后的完成时刻，因此后台准备占用的带宽会推迟前台恢复。

恢复器把缺失区间按本地 DRAM 和对端可达前缀的覆盖边界拆分。每个区间按以下可用集合选择速率最高的方法：

- 本地 DRAM 覆盖该区间时可以 restore；
- 任一对端的 HBM 或 DRAM 连续前缀覆盖该区间时可以 transfer。两者归为同一速率，因为两条路径都要离开对端、穿过网络、落到本地 HBM，而网络是其中较窄的一段；
- source token 始终存在，因此可以 recompute。

router 选择 TTFT 最小的节点，平局时选择节点编号较小者，并为请求输出后的完整 context 预留 HBM。若没有任何节点在回收全部可回收前缀后仍能容纳该 context，请求推迟到下一个控制周期重试，等待时间计入其 TTFT。

这个 router 使用模拟器内部准确的资源时间线和准确的 KV 前缀，不包含元数据延迟或预测误差，因此结果偏理想化。

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

早期实现使用单点预测 $\widehat t_s=A_{s,k}+\widehat\Theta$，并在两处把"当前时刻已超过 $\widehat t_s$"当作硬开关：准备侧以 `time_left <= 0` 跳过该 session，回收侧对该 session 返回固定最低分 0.0001。默认配置下任意时刻有 65% 的空闲 session 处于该状态，两处硬开关把同一条信息解释为相反的含义——准备侧视为不再相关，回收侧视为没有价值。实测该分数低于任何真实边际成本（最便宜的一刀为 0.00148），因此这些 session 的前缀会被优先剥空：占降级事件的 41.9%，但占降级 block 的 67.7%，平均每次被取走 19 个 block，而 SLO 达标 session 只有 6.6 个。改用 $q_s(t,\Delta)$ 之后，两侧使用同一个概率，该偏差消失（见 `PROTOTYPE_VERDICT.md` 第 11 节）。

## 8. RepKV 完整计划生成

### 8.1 预测到达时刻的排队估计

`busy_until` 只反映节点已经接纳的工作，而准备决策针对的是数秒之后的预测返回时刻。若直接用 `busy_until` 作为 $\widehat T^{queue}_n$，控制器看不到"KV 所在节点在请求到达前变忙"这一风险，因此不会产生任何准备动作。

控制器每个 tick 构造一次排队预测。所有处于空闲等待的 session 按 $\widehat t_s$ 升序插入一条影子时间线；插入 session $s$ 之前记录的各节点投影值即为 $s$ 使用的 $\widehat T^{queue}_n$，因此每个 session 的排队估计只包含比它更早返回的其他 session，不包含自身。节点归属按与 router 相同的最小预测 TTFT 规则确定。写入设备时间线的占用量只含该节点上的重算与 prefill：

$$
q_s\left(T_{s,n}^{recompute}+T_s^{prefill}\right).
$$

decode 不写入设备时间线，而是按概率占用一个槽位，占用时长为 $\widehat B^{output}/R_{\text{decode}}$。节点在 $\widehat t_s$ 的排队取设备空闲时刻与最早空闲槽位的最大值，与前台 TTFT 的重叠形式一致。$\widehat B^{output}$ 是对已完成请求 output 长度的指数滑动平均，初值为 `assumed_output_blocks`（即 output 中位数除以每 block token 数），因此不读取未来 output 长度。

对端 DRAM 与对端 HBM 一样计入远端可达前缀。HBM 降级后副本往往只留在 DRAM，若预测侧看不到该副本，就会把一次可取回的恢复判成缺失。

router 在请求真实到达时仍使用准确的 `busy_until`，排队预测只影响控制器。

流体队列会与到达时刻接近 $\widehat t_s$ 的请求成对，用 EWMA 记下残差。点估计只加回正残差（流体低估）；负残差通常是请求早于条件中位等待到达，不写回。`queue_exceeds_budget` 只用残差均值，集群可路由概率用均值加一倍散度。冷启动样本不足时不加这项。

DRAM 与对端副本没有预留。判定「当前布局已经能使 TTFT 满足 SLO」时，把 restore/transfer 路径按副本仍在的概率与「只靠本地 HBM、其余 recompute」混合。占用低于低水位且同一 session 在两台以上机器上有完整副本时，该概率为 1，行为与只看当场恢复相同；唯一副本或所在层已经超过低水位时，概率下降，最小前缀会把前缀钉进本地 HBM，集群概率也会给第二份 durable 副本正增益。

流体形式本身仍是均值近似，残差校正修的是偏差而不是把队列换成分布。网卡和 host 链路上的前台排队仍未写入影子时间线，只通过 TTFT 残差间接看到。

### 8.2 候选 session

只有满足以下条件的 session 才进入准备候选集：

- 已经完成至少一轮且当前没有活动请求；
- 脚本中仍存在下一轮，因此存在预测；
- $q_s(t,\Delta)\ge$ `min_return_probability`。

### 8.3 最小 SLO 前缀

设 session 的完整 context 为 $H_s$，节点当前 HBM 前缀为 $L_{s,n}$，计划准备到 $X_{s,n}$。准备后还需恢复区间 $[X_{s,n},H_s)$。计划必须满足硬约束：

$$
\max\left(Q_n,\ T_{s,n}^{recover}(X_{s,n})\right)
+T_s^{prompt}
\le D_s.
$$

恢复与设备排队取最大值：只用网卡或 host 链路的恢复可以与设备上其他请求的 prefill 重叠。$Q_n$ 取第 8.1 节在 $\widehat t_s$ 的排队估计，不是当前 `busy_until`。

剩余恢复时间对目标前缀单调不增，因此可行性对目标前缀单调，最小可行 $X_{s,n}$ 用二分求得。当场恢复把 restore/transfer 当成一定还在，会把唯一 DRAM 副本写成已经满足 SLO。当前实现把这条路径按副本仍在的概率 $p$ 与「只靠本地 HBM」混合：混合成功概率不低于 `already_feasible_probability`（默认 $0.9$）才放弃。$Q_n+T_s^{prompt}>D_s$ 时，完整 KV 在本地也无法满足 SLO，仍不生成计划；该条否决使用校正后的排队均值。最小可行目标若仍无法在 $\widehat t_s-t$ 内建完，则更大的目标同样来不及，计划器直接放弃。

`eager_full` 不经过这两条否决。只要 think time 内还能建完，它就把整段准备到另一台，净收益近似为 $1-0.01C$，几乎总为正。最小前缀在混合成功概率已经够高、或预测排队已经单独超过 SLO 时放弃；生成的计划还要再过 $V_{s,n}>0$ 的过滤。$P_s^{cluster}$ 对 volatile 与 durable 路径做同样的混合，唯一副本或高占用下再准备一份 durable 前缀会有正增益。

### 8.4 混合恢复 action

准备区间 $[L_{s,n},X_{s,n})$ 根据来源覆盖边界划分为连续区间。每个区间可选择 `restore`（本地 host tier 覆盖）、`transfer`（其他节点 HBM 或 DRAM 覆盖）或 `recompute`（利用 source token 精确重算）。

计划器枚举区间级方法组合，只保留总后台执行时间不超过 $\widehat t_s-t$ 的组合，并按单位成本选择最低成本方案。单位成本是该方法每 block 占用的资源秒：

$$
c_m=\frac{1}{R_m}.
$$

因此计划内部最小化的量与前台要争的资源秒是同一个，而不是一组与第 9.3 节无关的固定偏好。当前完成时间仍按各连续区间时间求和，未模拟多个资源的流水并行。

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
{1+\exp((\widehat{TTFT}_{s,n}+\sigma_{shared} Z-D_s)/\sigma_n)}
\right)
\right],
$$

其中 $Z$ 为标准正态，按五个等概率分层的条件均值做数值积分。

两个误差尺度随预测排队增长，而不是常数：

$$
\sigma_{shared}=0.45+0.5\bar Q,
\qquad
\sigma_n=0.25+0.8Q_n,
$$

其中 $Q_n$ 为该节点在 $\widehat t_s$ 处校正后的预测排队，$\bar Q$ 为各节点 $Q_n$ 的均值。每个节点的成功概率再按副本仍在的概率混合 volatile TTFT 与 durable TTFT。观测到的 TTFT 残差标准差若更大，会抬高 $\sigma_{shared}$ 与 $\sigma_n$ 的下限。理由是第 8.1 节的排队估计是流体量，它的可信度随排队本身下降：集群实际出现的负载量偏离均值多少属于共同误差，这些负载落在哪个节点上属于节点独立误差，两者的绝对幅度都随预测排队增长。取常数会让控制器在流体队列很小时最自信，而那恰是该估计最不可靠的区间。

早期版本把各节点成功事件当作完全独立。该假设在本模型中明显不成立：当一个 session 的 KV 前缀在所有节点都缺失时，各节点需要恢复的是同一段历史，TTFT 主要由同一个恢复量决定而不是由各自队列决定。完全独立假设会把四个恰好位于 SLO 边界的节点合成为 0.94 的集群成功概率，从而系统性低估准备价值。

这四个系数是控制器参数，不是实测的预测误差分布。随排队放大改善的是分辨力而不是偏差。在线残差把观测到的 (实际 $-$ 预测) 加回点估计，并抬高 $\sigma$ 的下限；流体形式本身仍是均值近似。

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
{R_m(\widehat t_s-t)},
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

当集群 HBM 使用率超过 82% 时，该成本继续增大。准备越过回收触发水位时，还要为它挤掉的降级付费，价格取该节点上最便宜受害者的回收分数，计费深度是到高水位的缺口而不是到满容量。总净收益为：

$$
V_{s,n}
=
G_{s,n}
-C_{s,n}^{displace}
-0.22C_{s,n}^{resource}.
$$

只有 $V_{s,n}>0$ 的候选计划可以执行。空动作的价值为 0，因此这条过滤本应保证不比不准备更差。三类资源在这里被合并为单一标量并共用 0.22 的权重，方法之间的差异只体现在归一化分母和价格 $\pi_m$ 上，没有实现相互独立的 $\lambda_B$、$\lambda_U$、$\lambda_H$。$G_{s,n}$ 用软成功概率，硬 SLO 已经不可行时「更接近截止」仍可能得到正增益；DRAM 溢出和前台抢网也没有进 $C^{displace}$。所以 `repkv` 在若干格子里仍低于 `on_demand`。

`eager_full` 不使用该净收益，而用 $1-0.01C$。几乎总为正，空动作进不去候选集。它是「总是准备完整前缀」的对照，不是覆盖全部工况的控制器。

### 9.4 排序和本周期选择

完整计划的 preparation slack 为：

$$
Slack_{s,n}
=
\widehat t_s-t-T_{s,n}^{prepare}.
$$

候选计划按 slack 升序、净收益与资源成本之比降序、再按 session ID 和节点 ID 排序。因此当前实现是"正净收益过滤 + 最小 slack 优先 + 收益成本比"，不是全局目标的精确优化解。

每个 tick 对同一 session 最多调度一个 batch，RepKV 最多在两个节点上保留该 session 的后台准备状态。后台没有按方法给出的整数预算。一个 batch 只能占用当前控制周期内仍空闲的那条资源，占用后与前台一样独占到完成；派发上限为 `block_batch`（默认 128 个 block）。

## 10. 渐进执行与重新规划

controller 先生成完整目标前缀和完整方法区间，但每个 tick 只调度第一个连续区间中的一个 batch。batch 最大为 `block_batch` 个 block，完成时间为：

$$
t^{complete}=t+\frac{B^{batch}}{R_m}.
$$

后台与前台使用同一组物理速率。调度时先预留目标节点 HBM。batch 完成时重新检查目标 HBM 前缀仍等于 batch 起点、restore 的本地 host 前缀仍覆盖 batch 终点、transfer 至少有一个远端 HBM 或 DRAM 前缀仍覆盖 batch 终点。任一条件失败则取消 batch。

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
+0.002\,\rho_e B_e
+\lambda q_s(t,\Delta)\,I_e\dfrac{B_e}{R_{\text{restore}}\Delta}
}
{B_e}.
$$

$P_s^{before}$ 与 $P_{s,e}^{after}$ 是集群可路由概率，不是本节点单独完成的概率。假设降级状态与 `_ensure_capacity` 一致：先把当前 HBM 前缀写入本节点 DRAM，再缩短 HBM，并且对端可达前缀随这次缩短一起更新，因此其他节点不能继续把正在丢掉的 HBM 当作取回来源。

$I_e=0$、$\rho_e=$ `spare_replica_factor`（默认 0.01）当且仅当缩短之后仍有**其他**节点的预测 TTFT 不超过 $D_s$。此时这块前缀是冗余副本：集群已经有一条不依赖它的可行路径，不必在本节点重建，降级地板也压低，使它排在任何集群仍需要的副本之前。$I_e=1$、$\rho_e=1$ 表示没有这样的外部可行节点，重建按本节点一次 restore 计费。

没有重建项时，控制器可以先为一次准备付费，再在同一个窗口内把**唯一**副本回收而不产生任何记账；并且当概率项在高负载下趋于饱和、$P^{before}$ 与 $P^{after}$ 都接近 1 时，前两项对所有候选几乎相同，排序失去梯度而退化为按 session ID 选择。重建项只加在唯一副本上，避免把这一地板同时垫高已经有外部可行路径的冗余副本。

分数越低，表示每释放一个 HBM block 的预期代价越小，因此越先降级。脚本中已无下一轮的 session 没有预测，分数为 0，可以自由回收。降级只缩短 `hbm_prefix`，原前缀保留在 `host_prefix` 中。

该分数与第 9.2 节的准备收益是同一个状态函数。实测把 40 个 block 的前缀从零建起来的收益为 0.24087，而按每次 4 个 block 从 40 削到 0 的十次损失之和为 0.23665，差值来自 $\max(0,\cdot)$ 的截断。因此在不同时刻分别计算不构成口径不一致。该分数对前缀长度是强凸的：40 到 0 的逐步损失依次为 0.0015、0.0028、0.0049、0.0085、0.0140、0.0218、0.0313、0.0415、0.0510、0.0594，最外层一刀比最内层便宜约 40 倍。

`on_demand` 和 `eager_full` 不使用该分数，而按 `last_used` 执行 LRU 降级。`value_based_eviction=False` 使 `repkv` 也退回 LRU，用于把回收侧的贡献从准备侧分离出来。`min_return_probability` 提到 1 以上则关闭全部准备。两开关合起来构成 $2\times2$ 消融，结果见 `PROTOTYPE_VERDICT.md`：只做回收在数据中心取回可行区间稳定为正，只做准备不能单独稳定产生收益，必须本地区间内完整准备强于最小前缀准备。

上述凸性是该分数与 LRU 行为差异的来源。LRU 的排序键是 `last_used`，一个 session 被选中后每次重新求最小值仍然选中它，因此它会被连续取 batch 直到削空；按价值排序则在削掉最外层后该 session 的边际损失立刻上升，下一次最小值转移到别的 session。结果是恢复量分布从"少数请求被剥空"变为"多数请求各损失一点"，对阈值型 SLO 更有利。

实现上没有维护回收优先队列，而是每次回收都对节点内所有 replica 重新求最小分数。每个候选都要求两次 $P_s^{cluster}$，这是模拟器的主要开销来源。DRAM 层有容量上限并按最近使用时间淘汰，但降级写入不占用 host 链路，因此 host 副本成本仍被低估，重建项的量级也没有实测支撑。也没有实现"删除 KV 但保留 token"的分支。

## 12. 三种策略的公平口径

三种策略在同一个 seed 上使用同一份 session 脚本：prompt 长度、output 长度、思考时间和继续与否的抽样完全相同。

由于到达时刻由完成时刻决定，服务更快的策略会更早获得下一轮，因此三种策略的到达序列不再逐位相同，完成的请求总数也不同。比较必须同时报告请求数、attainment 和 goodput：只看 attainment 会让完成更少请求、承受更低负载的策略显得更好。

| 策略 | 请求前行为 | 回收 |
|---|---|---|
| `on_demand` | 不执行后台准备 | LRU |
| `eager_full` | 渐进式构造一个完整备用前缀，不使用 RepKV 收益过滤 | LRU |
| `repkv` | 只准备能够新增硬 SLO 可行路由选项的最小前缀 | 预期 SLO 损失分数 |

准备与回收可以用两个开关分开：`value_based_eviction=False` 把 `repkv` 的回收退回 LRU，`min_return_probability>1` 拒绝全部准备候选。两者都关闭时与 `on_demand` 逐 seed 逐位相同。

## 13. 指标

单个 seed 的 SLO attainment 与 goodput 为：

$$
A_T=\frac{N_{success}}{N_{requests}},
\qquad
G_T=\frac{N_{success}}{T}.
$$

还统计 continuation SLO attainment、P99 TTFT、每个成功请求对应的 transfer/restore/recompute blocks、每个成功请求对应的 HBM block-seconds、未使用准备比例、降级 blocks、取消 batches、重叠轮次、推迟接纳次数和脚本池耗尽次数。另外单独报告准备准入：对每个空闲 `(session, node)` 是否生成计划、是否因「预测 TTFT 已满足 SLO」或「预测排队超过 SLO」放弃、是否被净收益过滤掉，以及真正动手时目标前缀占整段的比例、被下一轮用掉的准备量和到达时走前台 transfer 的量。完整准备与最小前缀的差距首先出现在这些准入计数上，而不是回收计数上。本实验的 SLO 约束在 TTFT，不约束 TPOT。

attainment 的实验口径以后续轮为准：首轮没有先前 KV 可供放置，其 TTFT 由整段 prompt 的 prefill 决定，计入会淹没被检验的量。`run.py` 因此同时打印 `continuation_attainment` 与 `slo_attainment`。

`exhausted_replenishments` 必须为 0 才能认为该次运行达到了所要求的负载。它非零表示 `sessions` 脚本池在 horizon 结束前用完，此后存活并发逐步衰减，负载和 attainment 都不再对应设定值。提高 `concurrent_sessions` 而不同步提高 `sessions` 很容易触发：`sessions=48` 配 `concurrent_sessions=40` 会在 240 秒内耗尽 39 次，请求数从 493 掉到 158，attainment 虚高到 1.0000。

`aggregate()` 对各 seed 的指标做简单宏平均。由于不同 seed 的请求数不同，宏平均 attainment 不等于把所有请求合并后的请求级加权 attainment。

## 14. 已知实现限制

1. workload 是合成的，不是公开 trace；`concurrent_sessions=0` 时不做 session 补充，不产生排队竞争，任何预放置策略在该配置上都无法被区分；`sessions` 与 `concurrent_sessions` 相互耦合，脚本池不足时运行会被静默截断，只能靠 `exhausted_replenishments` 事后发现；
2. 预测器接收真实的类别参数，只是不知道抽样值和轮数上限；没有训练、校准和线上推理开销；
3. router 使用准确的资源时间线和准确的 KV 元数据，也没有 Preble-style 或 DualMap-style 变体，因此"可叠加到不同 router"未被验证；
4. `readiness_snapshot()` 只被 `trace.py` 调用，router 实际直接读内部状态，元数据接口没有延迟或陈旧；
5. 前台恢复占用对应的计算、网络或 host 时间线，与后台争夺同一条资源，但恢复区间内部仍按串行求和，没有流水并行；
6. DRAM 层有容量上限，但写入不占用 host 链路，DRAM 淘汰按最近使用时间而不是按价值，因此 host 副本成本仍被低估；
7. 排队中的请求按完整 context 预留 HBM，这限制了可模拟的并发度，且不是经过验证的建模选择；
8. 完整计划不预留未来资源，未达门槛的半程前缀没有回滚；计划按整段计算净收益，每周期只执行一个 batch，决策单位与记账单位不一致；
9. block 没有对应真实 token 数之外的张量地址和 block table；
10. 没有模拟 RDMA、CANN stream、NPU kernel、NUMA 和跨机故障；
11. `device_flops`、`achieved_utilisation`、`network_bytes_s`、`host_capacity_bytes`、`decode_batch`、`cache_reserve_fraction` 均为 mock 值，没有由实测校准；两个门槛对 $B_{\text{net}}$ 和 $b$ 线性敏感；
12. 第 8.1 节的排队预测仍是流体均值。在线残差校正偏差，volatile/durable 混合处理会消失的副本；网卡与 host 链路排队仍未写入影子时间线，成功概率仍是 sigmoid 而不是硬 SLO 指示；
13. 节点是同质的。算力或带宽异构会使每个动作的可行长度成为节点属性，$C_1$ 与 $C_2$ 随节点排序，当前没有实现；
14. workload 没有会话迁移，慢链路下亲和被强制打破这一情形没有建模；
15. decode 速率在一个标定工作点上求得后作为常数使用，不随实际批内上下文长度变化。
16. 若干实验格子低于 `on_demand`。空动作在动作集里；`eager_full` 的 $1-0.01C$ 几乎总是准备，`repkv` 的软成功概率和未计价的 DRAM 溢出、前台抢网仍会让净收益为正。

## 15. 下一阶段实现顺序

1. 用实测的网络带宽、host 带宽、prefill 与 decode 吞吐替换 mock 硬件表，重新定位两个门槛。结论的结构不依赖这些数值，但适用区间的边界完全依赖；
2. 实现节点异构，使每个动作的可行长度成为节点属性，$C_1$ 与 $C_2$ 随节点排序，并把放置与路由合为一个决策；
3. 在 workload 中加入会话迁移，检验亲和被迫打破时预放置是否成为唯一手段；
4. 把第 8.1 节的流体排队换成队列长度分布，并把网卡、host 链路写入影子时间线。在线残差只校正偏差，不能代替分布；硬 SLO 指示仍未替换 sigmoid；
5. 使计划的决策单位与记账单位一致：在接纳计划时预留其全部所需资源，使声称的收益可实现；
6. 重新审视最小前缀准备的定位。物理参数下它在取回可行区间弱于完整准备；在必须本地区间一旦动手则目标长度接近整段，但「预测 TTFT 已满足 SLO」的否决使它经常不准备。需要给出它确有优势的条件，或将其降为次要贡献；
7. 用公开多轮会话 trace 替换合成脚本，并用实测思考时间分布替换对数正态假设；
8. 在真实 runtime 中验证 KV block 导出、连续前缀传输和导入正确性，并测量前后台并发干扰；
9. 使准备和回收相对空动作成立：硬 SLO 不可行时增益为 0，DRAM 溢出按删除计价，后台不得抢前台唯一链路。实现不应低于 `on_demand`。
