# AMCoEdge 强化学习模型结构核查

本文直接依据 `/Users/dazzysy/workspace/project/AMCoEdge` 中的源码，核查 AMCoEdge 及其两个强化学习基线的状态、动作、网络结构和训练方式。Stable Diffusion 目录中的生成模型不属于任务卸载策略，因此不纳入本次核查。

## 1. 总体结论

- AMCoEdge 中与任务卸载有关的强化学习网络共有三种：主方法 `AdaDQN`、单服务器选择基线 `DRLCoEdge/DQN` 和固定 Top-$k$ 服务器选择基线 `SMCoEdge/TopkDQN`。
- 三者都采用 **TensorFlow 1.x 全连接 DQN**，连续数值状态直接输入 MLP；隐藏层使用 ReLU，输出层直接产生每个离散动作的 Q 值。源码没有 embedding、卷积、循环网络、注意力、BatchNorm 或 Dropout。
- AMCoEdge 所谓“adaptive”主要体现在动作空间允许选择任意非空服务器子集，而不是采用了特殊的自适应神经网络。它仍然是标准 DQN，不是 Double DQN，也不是 Dueling DQN。
- AMCoEdge 的状态只有 6 维，因此 20 个神经元的窄网络已经足够小。该结构可以证明“连续调度状态可经归一化后直接输入 MLP”，但不能据此把当前报告中的 56 维状态也机械压缩到 20 个神经元。

## 2. 状态与多 Agent 组织方式

三个方法使用相同的状态：

$$
s_{b,n,t}
=
\left[
D_{b,n,t},
Q_{1,t},\ldots,Q_{5,t}
\right],
$$

其中 $D_{b,n,t}$ 是当前任务大小，$Q_{j,t}$ 是第 $j$ 个边缘服务器的处理队列工作量。环境将特征数定义为 $1+N_{\mathrm{BS}}$；实验设置 $N_{\mathrm{BS}}=5$，因此输入维度为 6。状态在主循环中通过 `np.hstack` 直接拼接，没有额外离散化或 embedding。[environment.py:32](/Users/dazzysy/workspace/project/AMCoEdge/environment.py:32) [main.py:17](/Users/dazzysy/workspace/project/AMCoEdge/main.py:17) [main.py:25](/Users/dazzysy/workspace/project/AMCoEdge/main.py:25)

系统为每个接入基站分别创建一个 DQN 实例，共 5 个 Agent；这些 Agent 读取相同结构的全局队列向量，但各自独立初始化、存储经验并更新参数。当前接入基站身份没有放入状态，而是由“哪个 Agent 进行决策”隐式表示。[main.py:57](/Users/dazzysy/workspace/project/AMCoEdge/main.py:57) [main.py:76](/Users/dazzysy/workspace/project/AMCoEdge/main.py:76) [main.py:77](/Users/dazzysy/workspace/project/AMCoEdge/main.py:77)

## 3. 三种 DQN 的精确结构

| 方法 | 状态维度 | 动作含义 | Q 值数量 | 单个在线网络结构 | 单网参数量 |
|---|---:|---|---:|---|---:|
| AMCoEdge `AdaDQN` | 6 | 5 个服务器的任意非空子集 | $2^5-1=31$ | $6\rightarrow20\rightarrow20\rightarrow20\rightarrow31$ | 1,631 |
| DRLCoEdge | 6 | 选择 1 个服务器 | 5 | $6\rightarrow20\rightarrow20\rightarrow5$ | 665 |
| SMCoEdge Top-$k$ | 6 | 固定选择 3 个服务器 | $\binom{5}{3}=10$ | $6\rightarrow20\rightarrow20\rightarrow10$ | 770 |

### 3.1 AMCoEdge `AdaDQN`

动作由 5 位二进制向量表示，每一位表示是否选择对应服务器。代码枚举大小为 1 至 5 的全部服务器组合，共产生 31 个动作；输出层因此包含 31 个 Q 值。[AdaDQN.py:36](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:36) [AdaDQN.py:46](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:46)

在线网络和目标网络均为三层隐藏 MLP：

- 输入层：6；
- 隐藏层：20、20、20，均使用 ReLU；
- 输出层：31，线性输出。

对应实现位于 [AdaDQN.py:68](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:68) 至 [AdaDQN.py:97](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:97)。权重使用均值 0、标准差 0.3 的正态分布初始化，偏置初始化为 0.1。[AdaDQN.py:107](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:107) [AdaDQN.py:111](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:111)

### 3.2 DRLCoEdge

该基线只选择一个服务器，因此动作数等于服务器数 5。[DQN.py:35](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/DRLCoEdge/DQN.py:35)

在线网络和目标网络均为：

- 输入层：6；
- 隐藏层：20、20，均使用 ReLU；
- 输出层：5，线性输出。

网络实现见 [DQN.py:59](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/DRLCoEdge/DQN.py:59) 至 [DQN.py:83](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/DRLCoEdge/DQN.py:83)。权重正态初始化的标准差为 0.5，偏置为 0.1。[DQN.py:93](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/DRLCoEdge/DQN.py:93) [DQN.py:97](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/DRLCoEdge/DQN.py:97)

### 3.3 SMCoEdge Top-$k$ DQN

该基线固定选择 $k=3$ 个服务器，5 个服务器对应 10 种组合动作。[TopkDQN.py:37](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/SMCoEdge/TopkDQN.py:37) [TopkDQN.py:39](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/SMCoEdge/TopkDQN.py:39) [main.py:58](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/SMCoEdge/main.py:58)

在线网络和目标网络均为：

- 输入层：6；
- 隐藏层：20、20，均使用 ReLU；
- 输出层：10，线性输出。

网络实现见 [TopkDQN.py:63](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/SMCoEdge/TopkDQN.py:63) 至 [TopkDQN.py:87](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/SMCoEdge/TopkDQN.py:87)。权重初始化标准差为 0.5，偏置为 0.1。[TopkDQN.py:97](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/SMCoEdge/TopkDQN.py:97) [TopkDQN.py:101](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/SMCoEdge/TopkDQN.py:101)

## 4. 在线网络、目标网络与 TD 目标

三种实现都维护结构相同的在线网络和目标网络，并通过硬拷贝更新目标网络参数。[AdaDQN.py:51](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:51) [AdaDQN.py:54](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:54) [AdaDQN.py:128](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:128)

TD 目标使用目标网络直接取最大 Q 值：

$$
y=r+\gamma\max_{a'}Q_{\mathrm{target}}(s',a').
$$

源码先计算 `np.max(q_next, axis=1)`，再更新已执行动作对应的目标值。[AdaDQN.py:195](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:195) [AdaDQN.py:203](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:203)

因此该实现属于 **vanilla DQN**。它没有使用在线网络选择下一动作、目标网络评估该动作的 Double DQN 目标，也没有价值流与优势流组成的 Dueling head。

训练使用全动作 Q 向量的均方误差和 Adam 优化器。[AdaDQN.py:118](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:118) [AdaDQN.py:123](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:123)

## 5. Replay 与训练超参数

### 5.1 实际运行配置

| 配置 | AMCoEdge | DRLCoEdge | SMCoEdge |
|---|---:|---:|---:|
| 学习率 | 0.001 | 0.001 | 0.001 |
| 折扣因子 $\gamma$ | 0.9 | 0.9 | 0.9 |
| 最终贪心概率 | 0.99 | 0.99 | 0.99 |
| 每次增量 | 0.001 | 0.001 | 0.001 |
| Batch size | 32 | 32 | 32 |
| Replay 容量 | 500 | 1,024 | 1,024 |
| 目标网络硬更新间隔 | 100 次学习 | 200 次学习 | 200 次学习 |
| Episode 数 | 500 | 400 | 300 |

AMCoEdge 的实际配置见 [main.py:57](/Users/dazzysy/workspace/project/AMCoEdge/main.py:57) 至 [main.py:67](/Users/dazzysy/workspace/project/AMCoEdge/main.py:67) 和 [main.py:80](/Users/dazzysy/workspace/project/AMCoEdge/main.py:80) 至 [main.py:88](/Users/dazzysy/workspace/project/AMCoEdge/main.py:88)。DRLCoEdge 的配置见 [main.py:53](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/DRLCoEdge/main.py:53) 至 [main.py:80](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/DRLCoEdge/main.py:80)。SMCoEdge 的配置见 [main.py:57](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/SMCoEdge/main.py:57) 至 [main.py:87](/Users/dazzysy/workspace/project/AMCoEdge/Baselines/SMCoEdge/main.py:87)。

### 5.2 Replay 和更新过程

- Replay buffer 是覆盖旧样本的环形数组；AMCoEdge 每条记录包含当前状态、5 位动作、奖励和下一状态，形状为 $500\times18$。[AdaDQN.py:48](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:48) [AdaDQN.py:133](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:133)
- Mini-batch 从已有 replay 中均匀随机采样，`np.random.choice` 未关闭重复采样。[AdaDQN.py:170](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:170)
- 学习在全局 `action_step` 超过 200 后开始，此后每 10 个 time slot 触发一次，每次让 5 个 Agent 各更新一次。[main.py:43](/Users/dazzysy/workspace/project/AMCoEdge/main.py:43) [main.py:44](/Users/dazzysy/workspace/project/AMCoEdge/main.py:44)
- `epsilon` 在该代码中表示选择贪心动作的概率。它从 0 开始，每次学习增加 0.001，最大为 0.99；其余概率选择随机动作。[AdaDQN.py:27](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:27) [AdaDQN.py:31](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:31) [AdaDQN.py:146](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:146) [AdaDQN.py:219](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:219)
- Transition 中没有终止标志，TD 目标也没有针对 session 或 episode 结束进行截断。[AdaDQN.py:133](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:133) [AdaDQN.py:203](/Users/dazzysy/workspace/project/AMCoEdge/AdaDQN.py:203)

## 6. 对当前报告中 DQN 结构的启示

AMCoEdge 提供的最可靠参考是：**由任务规模、队列、网络和资源状态组成的连续数值特征，可以经过尺度归一化后直接输入轻量 MLP。** 没有必要仅因输入来源不同而为每个数值字段单独构造 embedding。

但 AMCoEdge 的网络不能直接作为当前 Router 的最终配置：

- AMCoEdge 只有 6 维输入，当前报告有 56 维输入，状态之间的交互关系明显更多；
- AMCoEdge 没有 KV 位置、恢复方式、长度分布和 session continuation 等特征；
- AMCoEdge 采用 vanilla DQN，当前报告面对相近动作成本时，Double DQN 可减少最大值估计偏差；
- AMCoEdge 没有不可行动作过滤，而当前 Router 必须根据 KV 完整性、显存和节点可用性使用动作掩码。

因此，当前报告采用的 **56 维输入、两层 128 单元 ReLU 主干、Dueling head、9 个动作输出和动作掩码**仍是合理的首版设计。它只有约 2.5 万个参数，规模依然远小于通常意义上的深度模型。AMCoEdge 更适合作为“小型全连接调度网络”的实现依据，而不是逐层照搬的模板。

## 7. 当前 Router 的建议定稿结构

结合当前三节点系统的状态和动作定义，首版 Router 为每种模型训练一套共享策略参数，并可在三个入口节点分别部署相同的推理副本。当前入口节点已经包含在 56 维输入中，因此三个入口可以共享同一套参数，无需像 AMCoEdge 那样为每个入口分别训练一个 Agent。

建议网络固定为：

```text
56-dimensional float32 input
  -> Linear(56, 128) + ReLU
  -> Linear(128, 128) + ReLU
  -> value head: Linear(128, 1)
  -> advantage head: Linear(128, 9)
```

Dueling head 输出 9 个动作的长期成本；系统在网络输出后应用 9 个动作可用标志，并选择可行动作中成本最小的动作。在线网络负责选择下一动作，目标网络负责评价该动作，从而形成 Double DQN。该单网共有 25,098 个参数；在线网络和目标网络合计 50,196 个 float32 参数，权重合计约 196 KiB。

输入处理保持为：

- 当前入口使用 one-hot；
- 概率、比例和布尔状态保留在 $[0,1]$；
- token 数、请求字节数和排队工作量使用对数归一化；
- 带宽和显存按照系统上限归一化；
- RTT、KV 恢复时间等时延按照请求 SLA 归一化。

首版训练配置可采用：

| 项目 | 建议值 |
|---|---:|
| 损失函数 | Huber loss |
| 优化器 | Adam |
| 学习率 | $3\times10^{-4}$ |
| Batch size | 128 |
| Replay 容量 | $10^5$ |
| 训练预热 | 5,000 个 transition |
| 目标网络更新 | 每 1,000 个梯度步硬更新 |
| 折扣因子 | $\gamma=0.99$ |
| 随机探索概率 | 从 1.0 逐步降至 0.05 |
| 梯度范数上限 | 10 |

Replay 至少保存

$$
(s_k,a_k,c_k,s_{k+1},d_k,m_k,m_{k+1}),
$$

其中 $d_k$ 表示当前 session 是否结束，$m_k$ 和 $m_{k+1}$ 分别是当前状态和下一状态的动作可用标志。由于网络预测的是长期成本，动作选择使用 $\arg\min$；探索阶段同样只在当前可行动作中随机采样。Double DQN 的训练目标为

$$
z_k
=
c_k
+\gamma(1-d_k)
Q_{\bar\theta}
\left(
s_{k+1},
\arg\min_{a\in\mathcal A_{k+1}^{F}}
Q_\theta(s_{k+1},a)
\right).
$$

以上超参数是首版实现配置，而不是由 AMCoEdge 直接推出的最优值。正式实验至少应比较 64 与 128 两种隐藏层宽度、普通 Double DQN 与 Dueling Double DQN，以及 $\gamma\in\{0.95,0.99,1.0\}$。
