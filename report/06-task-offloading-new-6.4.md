# 6.4 面向用户移动性的长期会话感知 LLM 请求卸载方案

## 6.4.1 问题背景与研究目标

在协同边缘计算（Collaborative Edge Computing，CEC）系统中，多个边缘节点可以共同承载大语言模型（Large Language Model，LLM）推理服务。移动用户通过当前位置附近的边缘节点接入系统，但同一用户的多轮会话可能持续较长时间，后续请求的接入节点会随着用户移动而改变。与此同时，LLM 多轮推理产生的 KV cache、prefix cache 和会话上下文通常保留在先前执行请求的节点上。因此，用户的当前接入节点、实际推理节点和已有 KV cache 所在节点可能彼此不同。

普通计算任务卸载主要决定一个具有给定输入数据量、计算量和 deadline 的任务应在哪个节点执行。对于多轮 LLM 会话，一次请求的执行位置还会改变后续可复用 KV cache 的位置。当前选择在远端节点继续执行，可能减少本轮 KV 迁移开销，却使后续请求长期承担跨节点转发成本；当前将 KV cache 复制到新节点，虽然会增加本轮 TTFT 和网络传输量，但如果用户后续继续停留在新区域，则可能显著降低整个 session 的累计服务成本。

此外，KV cache 的位置不能简单表示为唯一的 owner。实际 KV 管理系统可能采用 copy-and-retain、move-and-delete、增量同步、TTL 保留或成本感知驱逐等策略。一次 KV 复制完成后，源节点上的旧 blocks 不一定立即删除；但由于新一轮推理会在执行节点产生新增 KV，源节点保留的副本可能只对应较旧的会话版本。后续请求返回源节点时，系统可以只同步缺失的增量 blocks，而不必重新传输完整 KV 或执行完整 prefill。

本节研究如下问题：

> 在一个多节点 CEC 系统中，给定一个具有已知交互 steps 的 LLM session，在用户未来接入位置不确定的情况下，如何联合决定每个 step 的推理节点以及 KV cache 的复用、复制、增量同步、删除或重算方式，在满足 TTFT SLA 和 GPU memory 约束的前提下，最小化整个 session 的期望累计服务成本。

本问题的核心耦合关系是：

\[
\text{当前卸载动作}
\longrightarrow
\text{新的 KV 分布与版本}
\longrightarrow
\text{未来移动状态下的服务成本}.
\]

用户移动性不是作为一个附加特征简单输入路由器，而是通过当前入口到计算节点的通信成本，以及未来入口节点的概率转移，显式进入长期优化目标。

---

## 6.4.2 系统场景与基本假设

考虑由 \(N\) 个边缘推理节点组成的 CEC 系统：

\[
\mathcal N=\{1,2,\ldots,N\}.
\]

本项目中 \(N=3\)。三个节点部署相同版本的 LLM，并同时作为用户请求的接入节点和候选计算节点。节点之间通过异构链路互联，其中两条链路为 100 Gbps 高速链路，另一条为 25 Gbps 跨主机 RDMA 链路。节点间链路具有不同的 RTT、有效带宽和设备间数据拷贝成本。

为隔离用户移动性的作用，第一阶段采用以下假设。

1. 系统首先考虑单用户、单 session；多用户资源竞争作为后续扩展。
2. 一个 session 包含 \(K\) 个顺序到达的交互 step。
3. 每个 step 的到达时间、输入 token 数、输出 token 数和 KV 增量在实验前已知。
4. 所有节点上的模型版本相同，单个 step 在各节点上的 prefill 和 decode 成本可通过 profiling 预先获得。
5. 节点背景负载、剩余显存和链路状态由给定 trace 提供，在当前基础模型中不作为随机变量。
6. 唯一重点建模的随机因素是用户移动，即后续请求将从哪个节点接入。
7. 基础模型假设用户只在相邻两个 step 之间移动，一个 step 执行期间接入节点保持不变；请求执行期间发生 handover 的情况在扩展模型中讨论。
8. KV cache 采用 block-level 管理，每个 block 具有 session ID、模型版本、token 区间、block hash 和状态版本。

令第 \(k\) 个 step 的任务描述为：

\[
r_k=(\tau_k,p_k,y_k,\Delta q_k),
\]

其中：

- \(\tau_k\) 为第 \(k\) 个 step 的到达时间；
- \(p_k\) 为该 step 新增的 prompt token 数；
- \(y_k\) 为该 step 的输出 token 数；
- \(\Delta q_k\) 为该 step 执行后新增的 KV cache 大小；
- \(\Delta\tau_k=\tau_{k+1}-\tau_k\) 为相邻两个 step 之间的时间间隔。

由于整个 step 序列已知，任务相关变量可由 step 索引 \(k\) 唯一确定。系统决策的主要不确定性来自未来用户入口位置。

---

## 6.4.3 用户移动性建模

### （1）接入节点状态

对于请求卸载，系统真正关心的不是用户的精确经纬度，而是用户在每个 step 从哪个边缘节点接入。因此，定义：

\[
U_k\in\mathcal N
\]

为第 \(k\) 个 step 的用户入口节点。

用户设备与入口节点之间的无线接入开销可以单独测量或合并到入口传输成本中。CEC 路由器重点优化入口节点、计算节点和 KV 节点之间的跨边缘通信。

### （2）离散 Markov 移动模型

当相邻 step 的时间间隔近似相同时，可以使用离散 Markov chain：

\[
P_{ij}
=
\Pr(U_{k+1}=j\mid U_k=i),
\]

并构造移动转移矩阵：

\[
\mathbf P=
\begin{bmatrix}
P_{11}&\cdots&P_{1N}\\
\vdots&\ddots&\vdots\\
P_{N1}&\cdots&P_{NN}
\end{bmatrix}.
\]

\(P_{ii}\) 表示用户下一轮继续停留在节点 \(i\) 的概率，\(P_{ij}\) 表示用户从节点 \(i\) 的覆盖区域移动到节点 \(j\) 的概率。

### （3）连续时间与半马尔可夫移动模型

LLM 多轮交互的相邻请求间隔通常并不固定。用户在两次请求之间等待得越久，移动到其他区域的概率通常越高。对于不同的 \(\Delta\tau_k\)，可以使用连续时间 Markov chain：

\[
\mathbf P(\Delta\tau_k)
=
\exp(\mathbf Q\Delta\tau_k),
\]

其中 \(\mathbf Q\) 为移动过程的生成矩阵。

如果需要更真实地描述用户在一个区域内的驻留时间，可以进一步使用 semi-Markov 模型：

- \(P_{ij}\) 表示用户离开节点 \(i\) 后前往节点 \(j\) 的概率；
- \(F_i(d)\) 表示用户在节点 \(i\) 的驻留时间分布；
- 驻留时间可以由 Weibull、Log-normal 分布或真实移动 trace 拟合。

### （4）预测式移动分布

如果系统部署 Digital Twin、LSTM、Transformer 或其他轨迹预测器，预测器应输出未来入口节点的概率分布：

\[
b_k(h,n)
=
\Pr(U_{k+h}=n\mid U_{1:k},\Delta\tau_{k:k+h-1}),
\]

而不是只输出一个确定位置。该概率分布用于计算不同未来移动分支下的期望成本，并可以在每个新请求到达后更新。

---

## 6.4.4 版本化多副本 KV cache 状态

### （1）为什么不能只建模唯一 KV owner

如果一次 KV 迁移采用 copy-and-retain 策略，目标节点获得 KV 副本后，源节点不会立即删除已有 blocks。但是，后续推理在目标节点产生的新 KV blocks 不会自动出现在源节点。因此，各节点可能同时保留同一 session 的不同版本。

例如，在第 \(k\) 个 step 开始前，节点 A 保存完整版本 \(v_{k-1}\)。系统将该版本复制到节点 B，并在 B 执行第 \(k\) 个 step。执行完成后：

- 节点 B 保存完整版本 \(v_k\)；
- 节点 A 仍保存版本 \(v_{k-1}\)；
- 如果用户下一轮返回 A，系统只需从 B 向 A 同步第 \(k\) 个 step 新增的 blocks，即可将 A 更新到 \(v_k\)。

因此，单一变量 \(O_k\) 无法准确描述 KV 状态。

### （2）Block-level placement

定义：

\[
X_{k,n,b}\in\{0,1\},
\]

表示在第 \(k\) 个 step 到达前，节点 \(n\) 是否保存 KV block \(b\)。

令 \(\mathcal B_k\) 为当前 session 截至第 \(k-1\) 个 step 已产生的全部 KV blocks，则节点 \(n\) 已有的 blocks 为：

\[
\mathcal B_{k,n}
=
\{b\in\mathcal B_k:X_{k,n,b}=1\}.
\]

系统 KV 状态可写为：

\[
\mathcal C_k
=
\{(n,v_{k,n},\mathcal B_{k,n})\mid n\in\mathcal N\},
\]

其中 \(v_{k,n}\) 为节点 \(n\) 当前保存的最高连续会话版本。

### （3）缺失状态大小

如果准备在节点 \(j\) 执行第 \(k\) 个 step，则该节点缺失的 KV blocks 为：

\[
\mathcal M_{k,j}
=
\mathcal B_k\setminus\mathcal B_{k,j},
\]

缺失状态的数据量为：

\[
Q_{k,j}^{\mathrm{miss}}
=
\sum_{b\in\mathcal M_{k,j}}q_b.
\]

这使系统能够区分：

- 节点没有任何 KV，需要复制完整状态；
- 节点保留旧版本，只需增量同步；
- 节点已有完整最新版本，可以直接复用。

### （4）KV 状态转移

KV 状态转移统一表示为：

\[
\mathcal C_{k+1}
=
G(
\mathcal C_k,
a_k,
\Delta\mathcal B_k,
\psi_k^{\mathrm{cache}}
),
\]

其中：

- \(a_k\) 为当前卸载和状态操作；
- \(\Delta\mathcal B_k\) 为第 \(k\) 个 step 新产生的 KV blocks；
- \(\psi_k^{\mathrm{cache}}\) 为 KV 管理策略。

\(\psi_k^{\mathrm{cache}}\) 可以实现：

- copy-and-retain：复制缺失 blocks，保留源节点已有副本；
- move-and-delete：复制完成后删除源副本；
- incremental-sync：只同步目标节点缺失的 blocks；
- replicate-after-execution：执行后将新增 blocks 同步到一个或多个节点；
- TTL retention：旧副本保留固定时间；
- cost-aware eviction：根据显存压力、复用概率和重建成本驱逐 blocks。

本节基础方案采用 copy-and-retain 与增量同步，并在显存不足时调用成本感知驱逐策略。

---

## 6.4.5 联合卸载与 KV 管理动作

令 \(J_k\in\mathcal N\) 为第 \(k\) 个 step 的计算节点。一次决策包含计算节点选择和 KV 状态处理方式选择：

\[
a_k=(J_k,Z_k).
\]

状态操作集合为：

\[
Z_k\in
\{
\mathrm{reuse},
\mathrm{forward},
\mathrm{copy\mbox{-}retain},
\mathrm{move\mbox{-}delete},
\mathrm{incremental\mbox{-}sync},
\mathrm{recompute}
\}.
\]

各动作含义如下。

1. **reuse**：计算节点已经拥有完整最新 KV，直接执行请求。
2. **forward**：入口节点将请求转发到拥有完整最新 KV 的节点执行，不移动 KV。
3. **copy-retain**：将目标节点缺失的 KV blocks 复制到目标节点，源节点保留已有 blocks。
4. **move-delete**：完成复制和状态校验后删除源节点相应 blocks。
5. **incremental-sync**：目标节点已有旧版本 KV，只传输版本差异对应的 blocks。
6. **recompute**：目标节点根据原始 context 执行完整 prefill，重建 KV。

动作必须满足状态一致性约束。例如，只有目标节点拥有完整最新版本后才能开始基于历史 KV 的增量 prefill；如果选择 recompute，则无需传输历史 KV，但必须能够获取完整 token context。

---

## 6.4.6 即时服务成本

### （1）入口到计算节点的请求传输

用户第 \(k\) 个请求从入口节点 \(U_k\) 到达。若计算节点为 \(J_k\)，请求转发成本为：

\[
T_k^{\mathrm{request}}(U_k,J_k)
=
RTT_{U_k,J_k}
+
\frac{D_k^{\mathrm{input}}}{BW_{U_k,J_k}}.
\]

当 \(U_k=J_k\) 时，不产生 inter-edge 请求转发成本，但仍包含用户到入口节点的无线接入开销。

生成结果返回用户的成本为：

\[
T_k^{\mathrm{return}}(J_k,U_k)
=
RTT_{J_k,U_k}
+
\frac{D_k^{\mathrm{output}}}{BW_{J_k,U_k}}.
\]

### （2）KV 状态准备成本

如果目标节点已有完整 KV：

\[
T_k^{\mathrm{state}}(a_k)=0.
\]

如果目标节点缺少部分 blocks，并从源节点 \(s\) 增量同步：

\[
T_k^{\mathrm{state}}(a_k)
=
RTT_{s,J_k}
+
\frac{Q_{k,J_k}^{\mathrm{miss}}}{BW_{s,J_k}}
+
T_{s,J_k}^{\mathrm{copy}}.
\]

如果目标节点没有可复用 KV 且选择 recompute：

\[
T_k^{\mathrm{state}}(a_k)
=
T_k^{\mathrm{full\mbox{-}prefill}}(J_k).
\]

当多个节点持有目标 blocks 时，可以针对每个 block 选择传输成本最低的源节点，或者优先选择拥有最长连续 prefix 的节点，以降低连接建立和设备拷贝开销。

### （3）TTFT 与端到端时延

第 \(k\) 个请求的 TTFT 为：

\[
\begin{aligned}
T_k^{\mathrm{TTFT}}(s_k,a_k)
=\;&
T_k^{\mathrm{request}}(U_k,J_k)
+T_k^{\mathrm{queue}}(J_k)\\
&+T_k^{\mathrm{state}}(a_k)
+T_k^{\mathrm{incremental\mbox{-}prefill}}(J_k).
\end{aligned}
\]

端到端时延为：

\[
T_k^{\mathrm{E2E}}(s_k,a_k)
=
T_k^{\mathrm{TTFT}}
+
T_k^{\mathrm{decode}}(J_k,y_k)
+
T_k^{\mathrm{return}}(J_k,U_k).
\]

### （4）综合即时成本

\[
\begin{aligned}
c_k(s_k,a_k)
=\;&
w_1T_k^{\mathrm{TTFT}}
+w_2T_k^{\mathrm{E2E}}
+w_3D_k^{\mathrm{inter\mbox{-}edge}}\\
&+w_4D_k^{\mathrm{KV}}
+w_5E_k
+w_6C_k^{\mathrm{eviction}}.
\end{aligned}
\]

其中：

- \(D_k^{\mathrm{inter\mbox{-}edge}}\) 为请求和结果的跨节点传输量；
- \(D_k^{\mathrm{KV}}\) 为 KV blocks 的传输量；
- \(E_k\) 为通信和推理能耗；
- \(C_k^{\mathrm{eviction}}\) 为驱逐已有 KV 对未来重建成本造成的估计损失。

---

## 6.4.7 SLA 与资源约束

对每个动作，需要满足目标节点显存约束：

\[
M_k^{\mathrm{model}}
+M_k^{\mathrm{request}}
+M_k^{\mathrm{KV}}(a_k)
+M^{\mathrm{reserve}}
\le
M_{J_k}^{\mathrm{free}}.
\]

在线可行性过滤使用单请求预测上界：

\[
\widehat T_k^{\mathrm{TTFT}}(s_k,a_k)
+\Delta_k(a_k)
\le
SLA_k,
\]

其中 \(\Delta_k(a_k)\) 为模型误差和元数据滞后的安全裕量。

如果 TTFT 预测本身是随机变量，可以使用机会约束：

\[
\Pr\left(
T_k^{\mathrm{TTFT}}(s_k,a_k)
\le SLA_k
\right)
\ge1-\epsilon.
\]

P99 TTFT 是跨请求统计指标，不直接等同于单请求可行性条件。系统在在线阶段使用预测上界或机会约束过滤动作，在实验阶段统计总体 P99 TTFT 和 SLA violation ratio。

可行动作集合定义为：

\[
\mathcal A_k^F(s_k)
=
\left\{
a\in\mathcal A_k(s_k):
\text{memory、SLA、版本一致性和链路约束均满足}
\right\}.
\]

---

## 6.4.8 完整长期优化问题

系统状态定义为：

\[
s_k=
\left(
k,
U_k,
\mathcal C_k,
\mathbf L_k,
\mathbf M_k,
\mathbf W_k
\right),
\]

其中：

- \(k\) 为当前 session step；
- \(U_k\) 为当前入口节点；
- \(\mathcal C_k\) 为版本化多副本 KV 状态；
- \(\mathbf L_k\) 为各节点负载；
- \(\mathbf M_k\) 为各节点剩余显存；
- \(\mathbf W_k\) 为节点间链路状态。

在基础模型中，\(\mathbf L_k\)、\(\mathbf M_k\) 和 \(\mathbf W_k\) 由已知 trace 给定，随机状态只有 \(U_k\)。

优化目标为：

\[
\begin{aligned}
\min_{\pi}\quad
&
\mathbb E_{U_{1:K}\sim P}
\left[
\sum_{k=1}^{K}
\gamma^{k-1}
c_k(s_k,a_k)
\right]\\
\mathrm{s.t.}\quad
&
a_k=\pi_k(s_k),\\
&
a_k\in\mathcal A_k^F(s_k),\\
&
U_{k+1}\sim
P(\cdot\mid U_k,\Delta\tau_k),\\
&
\mathcal C_{k+1}
=
G(
\mathcal C_k,
a_k,
\Delta\mathcal B_k,
\psi_k^{\mathrm{cache}}
).
\end{aligned}
\]

将用户移动轨迹的概率展开后，目标函数为：

\[
\begin{aligned}
\min_{\pi}
\sum_{u_1,\ldots,u_K}
&
\left[
p_0(u_1)
\prod_{k=1}^{K-1}
P_{u_k u_{k+1}}(\Delta\tau_k)
\right]\\
&\cdot
\left[
\sum_{k=1}^{K}
\gamma^{k-1}
c_k(s_k,a_k)
\right].
\end{aligned}
\]

该表达式清楚地说明，用户移动性通过两条路径影响优化结果：

1. 已实现的当前入口 \(U_k\) 改变请求到不同计算节点的即时通信成本；
2. 未来入口转移概率 \(P_{u_ku_{k+1}}\) 改变不同 KV 分布状态的期望长期价值。

---

## 6.4.9 有限时域动态规划

由于 session 的 steps 已知、节点数量较少，可以使用有限时域动态规划求解，而不必在基础模型中直接引入 reinforcement learning。

Bellman 方程为：

\[
\begin{aligned}
V_k(u,\mathcal C)
=
\min_{a\in\mathcal A_k^F(u,\mathcal C)}
\Bigg[
&
c_k(u,\mathcal C,a)\\
&+
\gamma
\sum_{u'\in\mathcal N}
P_{uu'}(\Delta\tau_k)
V_{k+1}
\left(
u',
G(\mathcal C,a,\Delta\mathcal B_k,\psi_k^{\mathrm{cache}})
\right)
\Bigg].
\end{aligned}
\]

终止条件为：

\[
V_{K+1}(u,\mathcal C)=0.
\]

这里，用户移动性显式体现在：

\[
\sum_{u'\in\mathcal N}
P_{uu'}(\Delta\tau_k)V_{k+1}(u',\cdot).
\]

当前动作决定下一阶段 KV cache 的分布和版本，移动模型决定下一阶段请求从哪个入口到达。优化器比较的是不同“用户入口—KV 分布”组合的未来最小成本，而不是给每次移动机械地增加固定的跨节点费用。

对于三个节点，如果限制每个 session 最多保留两个 KV 副本，并将 KV 状态压缩为各节点的最高连续版本，则状态和动作规模仍然较小，可以通过枚举获得最优策略。该最优解可以作为后续 RL 或启发式算法的 oracle baseline。

---

## 6.4.10 在线滚动优化

离线动态规划需要完整的移动转移模型。在线运行时采用 receding-horizon model predictive control（MPC）：

1. 第 \(k\) 个请求到达，系统观察真实入口 \(U_k\) 和当前 KV 分布 \(\mathcal C_k\)；
2. 移动预测器输出未来 \(H\) 个 steps 的入口概率；
3. 后台同步节点负载、显存和链路状态；
4. 枚举当前可行动作以及预测窗口内的未来动作；
5. 使用期望累计成本或风险感知成本评价候选方案；
6. 只执行当前 step 的最优动作；
7. 请求完成后更新 KV blocks、版本、实际 TTFT 和传输成本；
8. 下一请求到达后更新移动预测并重新规划。

窗口内目标为：

\[
\min_{a_{k:k+H-1}}
\mathbb E
\left[
\sum_{h=0}^{H-1}
\gamma^h
c_{k+h}(s_{k+h},a_{k+h})
+
\gamma^H
\widehat V(s_{k+H})
\right],
\]

其中 \(\widehat V(s_{k+H})\) 为窗口末端价值估计。对于当前三节点实验，可以先令 \(\widehat V=0\)，并通过增加 \(H\) 观察规划窗口长度的影响。

如果移动预测存在较大误差，可以进一步使用：

- chance-constrained MPC；
- CVaR 风险成本；
- distributionally robust optimization；
- top-\(m\) 移动场景树裁剪。

---

## 6.4.11 示例：保留旧 KV 副本如何影响长期决策

考虑三个节点 A、B、C。第 2 个 step 到达前：

- 用户从 B 接入；
- A 保存第 1 个 step 后的完整 KV，大小为 8 GB，版本为 \(v_1\)；
- B 和 C 没有该 session 的 KV；
- 第 2 个 step 会新增 2 GB KV；
- copy-and-retain 策略保留源节点已有 blocks；
- 节点显存足够，不发生自动驱逐。

移动模型预测第 3 个 step 的入口概率：

\[
\Pr(U_3=B)=0.70,\quad
\Pr(U_3=A)=0.25,\quad
\Pr(U_3=C)=0.05.
\]

方案一将 A 的 8 GB KV 复制到 B，并在 B 执行第 2 个 step。设当前成本为 6 个单位。执行完成后：

\[
\mathcal C_3
=
\{
(A,v_1,8\text{ GB}),
(B,v_2,10\text{ GB})
\}.
\]

如果下一轮用户继续在 B，B 可以直接复用最新 KV，状态准备成本为 0。如果用户返回 A，A 只缺少第 2 个 step 新增的 2 GB blocks，可以从 B 增量同步；设成本为 2。如果用户移动到 C，系统在转发请求、复制完整 KV 和重算之间重新选择，设最小成本为 10。因此：

\[
Q_2^{\mathrm{copy}}
=
6
+0.70\times0
+0.25\times2
+0.05\times10
=7.
\]

方案二不复制 KV，而是将 B 的请求转发至 A，并继续在 A 执行。设当前成本为 8。执行完成后，A 保存完整 \(v_2\)，B 和 C 没有 KV。如果下一轮用户在 B，最小服务成本为 9；如果用户返回 A，状态准备成本为 0；如果用户移动到 C，最小服务成本为 11。因此：

\[
Q_2^{\mathrm{stay}}
=
8
+0.70\times9
+0.25\times0
+0.05\times11
=14.85.
\]

在该移动分布下，复制到 B 并保留 A 的旧 prefix 更优。

如果系统使用 move-and-delete 策略，A 的 8 GB 旧副本会在复制完成后被删除。用户返回 A 时便无法只同步新增的 2 GB，而需要重新传输完整 KV、将请求转发到 B 或执行完整重算。由此可见，KV 删除策略会改变状态转移函数 \(G\)，并进一步改变长期最优卸载策略。

---

## 6.4.12 请求执行期间发生移动的扩展

基础模型假设用户只在两个 steps 之间移动。对于长输出请求，decode 可能持续较长时间，用户可能在单个 step 完成前发生 handover。此时需要增加剩余驻留时间：

\[
D_k^{\mathrm{remain}},
\]

并将动作可行性写为：

\[
\Pr\left(
D_k^{\mathrm{remain}}
\ge
T_k^{\mathrm{service}}(s_k,a_k)
\right)
\ge1-\epsilon.
\]

如果一个动作预计无法在当前驻留时间内完成，可以选择：

- 将请求卸载到用户预计前往的下一个节点；
- 在当前节点完成 prefill，并将 KV 发送到下一个节点执行 decode；
- 在相邻节点保留冗余 KV 副本；
- 允许当前节点继续执行，并将生成结果跨节点流式转发给新的入口节点。

该扩展将用户移动性进一步引入单个请求内部，但会增加 handover、流式生成和增量 KV 同步的建模复杂度，适合作为基础 session-level 模型之后的进一步研究。

---

## 6.4.13 算法流程

完整在线流程如下。

1. **状态同步**：各节点同步队列、显存、链路和 block-level KV 元数据。
2. **入口观测**：请求到达后确认当前用户入口节点。
3. **移动预测**：输出未来若干 steps 的入口概率分布。
4. **动作生成**：枚举候选计算节点和 reuse、forward、copy-retain、incremental-sync、move-delete、recompute 动作。
5. **可行性过滤**：删除违反 TTFT、memory、版本一致性和链路约束的动作。
6. **长期成本评估**：使用 DP 或 MPC 计算移动场景树下的期望累计成本。
7. **动作执行**：完成请求转发、KV 准备和 LLM 推理。
8. **状态更新**：记录新增 KV blocks、各节点版本、实际时延和传输量。
9. **副本管理**：根据 TTL、显存压力和未来复用价值保留或驱逐旧 blocks。
10. **滚动重规划**：下一 step 到达后重新执行上述过程。

---

## 6.4.14 对比方法与实验设计

### （1）移动场景

实验至少包含以下移动模式。

1. **Static**：用户始终从同一节点接入。
2. **Single handover**：session 运行到固定比例后切换一次入口。
3. **Markov roaming**：用户按照给定转移矩阵在节点间随机移动。
4. **Route mobility**：用户按照 A→B→C 等预设路径移动。
5. **Return mobility**：用户按照 A→B→A 等模式返回旧节点，用于验证保留旧 KV 副本的价值。
6. **Variable dwell time**：使用 semi-Markov 模型生成不同驻留时间。
7. **Prediction error**：控制预测准确率、分布熵和错误目的节点比例。

### （2）对比算法

1. **Entry-local**：始终在当前入口节点执行；缺失 KV 时复制或重算。
2. **Latest-KV-node**：始终将请求转发到拥有最新完整 KV 的节点。
3. **Greedy**：选择当前 step 即时成本最低的可行动作。
4. **No-prediction MPC**：只使用当前入口，不使用未来移动分布。
5. **Mobility-aware DP/MPC**：使用预测移动分布最小化累计成本。
6. **Move-delete**：移动 KV 后立即删除源副本。
7. **Copy-retain**：保留源节点旧版本，并支持增量同步。
8. **Oracle trajectory**：已知真实未来入口序列，作为性能上界。

### （3）评价指标

请求级指标包括：

- 平均、P95 和 P99 TTFT；
- 平均端到端时延；
- TPOT/TBT；
- SLA violation ratio；
- 请求转发次数和跨节点传输量。

Session 级指标包括：

- session 累计服务成本；
- session 累计端到端时延；
- KV 完整复制次数；
- KV 增量同步次数；
- KV 总传输量；
- 完整 prefill 重算次数；
- prefix/KV block 命中率；
- 无效副本比例；
- 因错误移动预测造成的无效传输量；
- KV 副本平均驻留时间；
- 显存占用和 eviction 次数。

### （4）消融实验

1. 移除移动预测，只使用当前入口；
2. 将概率分布替换为单点位置预测；
3. 禁用旧副本保留；
4. 禁用增量同步，只允许完整复制；
5. 禁用 recompute；
6. 改变规划窗口 \(H\)；
7. 改变 session 长度和 KV 增长速度；
8. 改变 25/100 Gbps 链路组合；
9. 改变节点显存水位和 cache TTL；
10. 改变移动预测误差和用户返回概率。

---

## 6.4.15 模型边界与后续扩展

本节模型有意固定任务 steps 和请求规模，将用户移动性作为主要随机因素，以便清晰验证长期路由与 KV 状态管理之间的关系。该基础模型可以沿以下方向扩展。

1. **未知输出长度**：将输出 token 数和 KV 增长建成随机变量。
2. **多用户竞争**：联合考虑多个 session 对 GPU、显存和链路的竞争。
3. **请求期间 handover**：允许用户在 prefill 或 decode 期间改变入口。
4. **Prefill/decode 分离**：分别选择 prefill 和 decode 节点，并优化 KV handoff。
5. **多模型选择**：联合选择本地 SLM、边缘 LLM 和云端 LLM。
6. **预测—决策联合训练**：让长期卸载 regret 反向优化移动预测模型。
7. **大规模求解**：当节点、用户和副本数量增加后，使用 approximate dynamic programming、actor-critic 或 multi-agent reinforcement learning。

基础模型首先回答一个清晰问题：

> 当 session 的未来 steps 已知、用户未来入口存在概率不确定性时，系统应在什么条件下承担当前 KV 复制或同步成本，以换取后续多个 steps 更低的请求转发和状态准备成本。

该问题同时体现了 LLM 多轮会话的状态性、KV cache 的版本化多副本特征以及用户移动带来的长期决策耦合，是普通无状态 task offloading 模型无法直接覆盖的问题。
