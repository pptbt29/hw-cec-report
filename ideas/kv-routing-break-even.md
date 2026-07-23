# KV 状态恢复与请求路由 Break-even 思考

## 1. 问题背景

在移动 CEC 场景中，用户的新请求可能从新入口节点到达，而该 session 的历史 KV cache 仍保留在旧计算节点。此时系统至少有三类选择：

1. 将轻量 request 转发到旧节点，并复用旧节点已有的 KV；
2. 将缺失 KV blocks 同步到新节点，再在新节点继续执行；
3. 在新节点根据完整 context 重算缺失 KV。

一个重要的数量级事实是：单次 request/response 通常为 KB 到数 MB，而长上下文 KV cache 可以达到数 GB 乃至数十 GB。因此，不能默认“用户移动后 KV 也应该跟随用户迁移”。更合理的默认策略通常是 **state-sticky routing**：优先将轻量请求转发回 KV 所在节点，只有当未来累计收益能够覆盖状态恢复成本时，才将执行位置迁移到新节点。

本方案的核心价值不应表述为“让 KV 跟着用户移动”，而应表述为：

> 将轻量 request forwarding、重量级 KV synchronization、目标节点 recomputation 和未来 KV reuse 放入同一个长期成本模型，显式判断何时保持状态黏附、何时同步增量状态、何时重算更划算。

## 2. KV 规模的数量级

对于采用标准 attention cache 的模型，每个 token 的 KV 大小可近似为

\[
D_{\mathrm{KV/token}}
=2L n_{\mathrm{kv}}d_{\mathrm{head}}b,
\]

其中，\(L\) 是 Transformer 层数，\(n_{\mathrm{kv}}\) 是 KV head 数量，\(d_{\mathrm{head}}\) 是每个 head 的维度，\(b\) 是单个元素的字节数，系数 2 分别对应 Key 和 Value。

典型数量级如下：

| 模型配置示例 | 每 token KV | 32K context | 128K context |
|---|---:|---:|---:|
| 8B GQA | 约 128 KiB | 约 4 GiB | 约 16 GiB |
| 70B GQA | 约 320 KiB | 约 10 GiB | 约 40 GiB |
| 传统 MHA 7B | 约 512 KiB | 约 16 GiB | 约 64 GiB |

因此，GB 级 KV 并非极端情况。模型结构、KV quantization、block 稀疏性和实际连续 prefix 长度会改变具体数值，但不会改变“KV 通常远大于单次请求负载”这一基本关系。

## 3. 全量同步的基础成本

项目场景包含 100 Gbps 和 25 Gbps 链路。若按约 80% 的有效利用率估算：

- 100 Gbps 链路的有效吞吐约为 10 GB/s；
- 25 Gbps 链路的有效吞吐约为 2.5 GB/s。

只考虑网络传输的乐观同步时间为：

| KV 同步量 | 100 Gbps | 25 Gbps |
|---:|---:|---:|
| 1 GB | 0.1 s | 0.4 s |
| 5 GB | 0.5 s | 2 s |
| 10 GB | 1 s | 4 s |
| 30 GB | 3 s | 12 s |

真实开销通常更高，因为还可能包括链路竞争、源端读取、目标端写入、PCIe/DMA、显存带宽争用、版本检查和执行 barrier。

## 4. 两类不同的 Break-even

### 4.1 按网络字节数计算

若每个后续请求因转发到旧节点产生额外流量 \(D_{\mathrm{req}}\)，完整同步量为 \(D_{\mathrm{KV}}\)，则仅按网络字节数计算的 break-even 请求数约为

\[
N_{\mathrm{BE}}^{\mathrm{bytes}}
\approx
\frac{D_{\mathrm{KV}}}{D_{\mathrm{req}}}.
\]

若每轮额外转发 1--5 MB，则 10 GB KV 需要约 2,000--10,000 个后续请求才能在流量上摊平。普通 session 很难达到这一长度。

因此，如果迁移收益只是“以后不用转发几 MB 的 request”，全量 KV 迁移通常不划算。

### 4.2 按端到端时延或综合成本计算

KV 迁移也可能避免远端 RTT、旧节点排队、跨域成本或 SLA violation。若每个后续请求在旧节点执行会额外产生 \(\Delta C_{\mathrm{remote}}\)，状态迁移的一次性成本为 \(C_{\mathrm{move}}\)，则

\[
N_{\mathrm{BE}}
\approx
\frac{C_{\mathrm{move}}}{\Delta C_{\mathrm{remote}}}.
\]

以 10 GB KV 为例，假设在 25 Gbps 链路上同步约需 4 s：

| 每轮远端额外时延 | Break-even 请求数 |
|---:|---:|
| 20 ms | 约 200 轮 |
| 50 ms | 约 80 轮 |
| 100 ms | 约 40 轮 |
| 500 ms | 约 8 轮 |

由此可见，全量迁移只有在旧节点明显拥塞、远端路径显著增加时延、SLA 风险较高，或预计 session 仍有很多轮时才可能成立。

## 5. Sync 与 Recompute 的混合恢复

缺失 KV 不必全部同步或全部重算。KV manager 可以将缺失集合划分为

\[
\mathcal M_k
=
\mathcal M_k^{\mathrm{sync}}
\cup
\mathcal M_k^{\mathrm{rec}},
\qquad
\mathcal M_k^{\mathrm{sync}}
\cap
\mathcal M_k^{\mathrm{rec}}
=\varnothing.
\]

网络同步和 GPU recompute 若使用相对独立的资源，可以流水并行。理想恢复时间应近似写为

\[
T_k^{\mathrm{hybrid}}
\approx
T_k^{\mathrm{startup}}
+
\max\left\{
T_k^{\mathrm{sync}}(\mathcal M_k^{\mathrm{sync}}),
T_k^{\mathrm{rec}}(\mathcal M_k^{\mathrm{rec}})
\right\}
+
T_k^{\mathrm{merge}},
\]

而不是简单相加 \(T_k^{\mathrm{sync}}+T_k^{\mathrm{rec}}\)。其中 startup 和 merge 分别表示目录查询、连接建立、版本检查、block 注册和执行 barrier 等流水线启动/排空 bubble。

如果完整同步时间为 \(T_s\)，完整重算时间为 \(T_r\)，同步比例为 \(x\)，并假设成本随处理比例近似线性，则

\[
T_{\mathrm{hybrid}}(x)
=
\max\{xT_s,(1-x)T_r\}.
\]

理想最优划分满足两条流水线的结束时间相等：

\[
x^*T_s=(1-x^*)T_r,
\qquad
x^*=\frac{T_r}{T_s+T_r},
\]

从而

\[
T_{\mathrm{hybrid}}^*
=
\frac{T_sT_r}{T_s+T_r}.
\]

例如，完整同步需要 4 s，完整重算需要 2 s，则理想同步比例为 \(1/3\)，混合恢复时间约为 1.33 s，低于全同步的 4 s 和全重算的 2 s。

## 6. 混合恢复如何改变路由 Breakpoint

采用全量同步时，切换到新节点所需的未来请求数约为

\[
H_{\mathrm{sync}}^*
=
\frac{T_s}{\Delta C_{\mathrm{remote}}}.
\]

采用理想混合恢复后，阈值变为

\[
H_{\mathrm{hybrid}}^*
=
\frac{T_sT_r}
{(T_s+T_r)\Delta C_{\mathrm{remote}}}.
\]

两者的比值为

\[
\frac{H_{\mathrm{hybrid}}^*}{H_{\mathrm{sync}}^*}
=
\frac{T_r}{T_s+T_r}.
\]

继续使用 \(T_s=4\) s、\(T_r=2\) s、每轮远端额外时延为 50 ms 的例子：

- 全量同步需要约 \(4/0.05=80\) 个后续请求才能 break-even；
- 混合恢复需要约 \(1.33/0.05\approx27\) 个后续请求。

因此，混合恢复会扩大“切换到新节点”的可行区域。但 breakpoint 仍然不是固定轮数，而是以下状态变量共同决定的一张动态边界：

\[
(D_k^{\mathrm{miss}},BW_{s,j},R_j^{\mathrm{rec}},
L_j,\Pr[\text{session continues}],
\Delta C_{\mathrm{remote}})
\longrightarrow
\{\text{旧节点复用},\text{新节点恢复}\}.
\]

总体变化趋势如下：

- 缺失 KV 越少，越容易切换到新节点；
- 网络越快，越容易切换到新节点；
- 目标节点 recompute 越快，越容易切换到新节点；
- 通信与计算重叠越充分，所需未来轮数越少；
- 旧节点越拥塞或远端 RTT 越高，越容易切换到新节点；
- session continuation probability 越低，越应保持 state-sticky；
- PCIe、显存带宽和 GPU 计算竞争越严重，混合恢复收益越小；
- 若恢复过程可以被当前请求的其他阶段或后台预取完全隐藏，临界未来请求数可能接近 1。

## 7. Session 长度不确定时的概率阈值

设每轮结束后 session 继续的概率近似为常数 \(q\)，则未来请求数期望为

\[
\mathbb E[H]=\frac{q}{1-q}.
\]

若每轮迁移后的平均收益为 \(\Delta C_{\mathrm{remote}}\)，状态恢复成本为 \(C_{\mathrm{restore}}\)，则迁移条件为

\[
\frac{q}{1-q}\Delta C_{\mathrm{remote}}
>
C_{\mathrm{restore}},
\]

对应 continuation probability breakpoint 为

\[
q^*
=
\frac{C_{\mathrm{restore}}}
{C_{\mathrm{restore}}+\Delta C_{\mathrm{remote}}}.
\]

以上述 50 ms 每轮收益为例：

- 全同步成本为 4 s 时，\(q^*\approx98.8\%\)；
- 混合恢复成本为 1.33 s 时，\(q^*\approx96.4\%\)。

混合恢复降低了迁移所要求的 session 连续性，但当每轮远端额外代价很小时，系统仍需要对 session 会持续较长时间有较高置信度。

## 8. 更完整的混合恢复成本

理想的最大值模型只描述关键路径时延。实际决策还需要同时考虑同步字节数、重算计算量、资源竞争和缓存外部性。对目标节点 \(j\)，可定义

\[
\begin{aligned}
C_k^{\mathrm{hybrid}}(j,x)
=\;&w_t
\max\left\{
xT_k^{\mathrm{sync}}(j),
(1-x)T_k^{\mathrm{rec}}(j)
\right\}\\
&+w_bxD_k^{\mathrm{miss}}
+w_cC_k^{\mathrm{rec}}(1-x)
+C_k^{\mathrm{bubble}}
+C_k^{\mathrm{contention}}
+C_k^{\mathrm{eviction}}.
\end{aligned}
\]

KV manager 或 per-action estimator 可以先求解

\[
C_k^{\mathrm{restore},*}(j)
=
\min_{0\le x\le1}
C_k^{\mathrm{hybrid}}(j,x),
\]

路由器再比较该一次性恢复成本和未来保持远端执行的累计成本：

\[
C_k^{\mathrm{restore},*}(j)
<
\sum_{h=1}^{H}
p_h\Delta C_{k+h}^{\mathrm{remote}}(j).
\]

这一定义可以兼容未知 session 长度、移动预测和未来负载变化。

## 9. 实现与建模限制

混合恢复在真实系统中存在以下限制：

1. **连续 prefix 依赖。** KV 重算具有因果依赖，不能将任意散乱 token blocks 完全独立地重算。通常需要从已有连续 prefix、checkpoint 或可恢复边界开始。
2. **网络与 GPU 并非完全独立。** KV 写入会争用 PCIe、DMA、GPU memory bandwidth 和显存容量，因此实际恢复时间通常高于理想的最大值。
3. **关键 block barrier。** 当前 attention 所需的连续 KV 未就绪时，prefill/decode 仍会停顿，应优先恢复关键路径上的 blocks。
4. **同步和重算的选择不是只看字节数。** 对于重算代价高但传输较小的 blocks 应优先同步；对于传输代价高但可快速生成的 blocks，可以优先重算。
5. **副本会产生外部性。** 新 KV 副本占用显存并可能驱逐其他 session 的热点 blocks，因此迁移收益需要扣除 cache eviction cost。

## 10. 对上层动作空间的影响

当前上层动作仍可保持为

\[
a_k=(J_k,Z_k),
\qquad
Z_k\in\{\mathrm{reuse},\mathrm{sync},\mathrm{recompute}\}.
\]

不必让 DQN 直接决定每个 block 的恢复方式。可以将上层 `sync` 更准确地解释为“由 KV manager 管理的状态恢复”：

- `reuse`：目标节点已具有所需连续 KV，无状态恢复；
- `sync`：允许 KV manager 对缺失 blocks 执行纯同步或同步与重算的混合恢复；
- `recompute`：不依赖远端 KV 传输，在目标节点执行纯重算。

per-action estimator 对每个目标节点返回优化后的 \(C_k^{\mathrm{restore},*}(j)\)、实际同步字节数和关键路径恢复时间。DQN 只需判断优化后的恢复成本是否低于未来维持状态黏附的累计代价，而不需要扩张到 block-level action space。

这种设计保持了固定且较小的 DQN 动作空间，同时将复杂的 block 划分和流水执行留给更适合处理底层约束的 KV manager。

## 11. 方法亮点的重新定位

基于上述分析，较稳妥的 sale points 是：

1. **State-sticky by default。** 方法显式认识到 KV 通常比请求负载大几个数量级，不会盲目让 KV 跟随用户迁移。
2. **Long-term break-even routing。** 路由决策比较一次性状态恢复成本和未来多轮远端执行成本，而不是只优化当前请求。
3. **Block-level selective recovery。** 只处理目标节点缺失的 blocks，避免完整 KV 的无效复制。
4. **Communication-computation overlap。** 同步与重算可被组织成并行流水，使恢复时延由关键路径而不是两项之和决定。
5. **Hierarchical decision。** DQN 决定计算节点和高层 KV 获取方式，KV manager 决定 block 划分、同步源、流水和副本管理。
6. **Prediction changes action value。** 移动预测、输出长度预测和 session continuation probability 不仅估计当前时延和显存，也改变未来 KV 增量、状态位置以及路由 breakpoint。

一句话总结为：

> 本方案不是主动推动重量级 KV 迁移，而是以 state-sticky routing 为基线，联合估计轻量请求转发、增量 KV 同步、目标节点重算及其并行重叠的长期成本，只在预期多轮收益超过最优状态恢复成本时切换计算节点。

## 12. 可验证的实验假设

后续实验可以围绕以下假设展开：

1. 在低 RTT、旧节点低负载和短 session 下，state-sticky routing 应优于 KV migration。
2. 随旧节点排队时延增加，路由 breakpoint 会快速向较短 session 移动。
3. 增量缺失比例降低时，混合恢复的收益应明显提高。
4. 25 Gbps 链路下，全量同步通常不划算，但 block-level hybrid recovery 可能在较长 session 或高排队差场景达到 break-even。
5. 100 Gbps 链路下，迁移可行区域扩大，但仍取决于 KV 大小和 continuation probability。
6. 相比纯同步和纯重算，混合恢复应降低关键路径恢复时间；实际收益与网络/GPU/显存带宽竞争程度相关。
7. 相比只按即时成本路由，长期策略应减少错误迁移和反复迁移，并降低单位 session 的 KV 传输量。
8. DQN 的主要收益应出现在动态负载、移动和未知 session 长度共同存在的场景；静态简单场景下 Greedy 可能已经接近最优。

