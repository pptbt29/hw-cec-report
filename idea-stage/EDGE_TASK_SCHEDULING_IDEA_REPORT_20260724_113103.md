# One-Page Paper Idea：StateWeaver

**暂定标题**：*StateWeaver: Executable-State Assembly Scheduling for Stateful Multimodal Inference across the Computing Edge Continuum*

**当前判断**：这是当前候选中最值得优先验证的方向，但还不是可以直接开写的 paper claim。它必须先通过第 9 节的三个 kill tests；否则应停止，而不是继续添加 RL、mobility 或更多约束。

## 1. 一句话 Idea

在 CEC 中，调度器不再把请求看成“输入数据 + 计算量”，而把它看成一个需要由多个位置共同补齐的**可执行状态闭包**：只有通过显式兼容性检查的模型环境、LoRA adapter、最新连续 KV prefix 和当前多模态 observation/representation 全部就绪，请求才能启动。StateWeaver 联合选择执行节点、每项状态的获取或重建方式，以及共享链路上的状态组装次序，以最大化 SLO-goodput。

## 2. 为什么这是 CEC 原生问题

一个长 session 的新请求到达 access edge 时，完成本轮推理所需的对象可能天然分散：

- 当前图像、视频或传感器 observation 位于 ingress/access edge；
- 最新 session KV 位于上一轮执行节点；
- 租户 adapter 可能位于另一个 GPU、host memory 或 regional store；
- 可执行的基础模型和空闲 GPU 位于若干 access/regional/cloud worker。

普通 task offloading 先选执行节点，再把输入传输时间、KV 恢复时间和排队时间加成一个标量。这个模型忽略了三个结构：

1. **AND-join**：所有必要状态都完成后才能启动，ready time 由最晚完成的必要分支决定；
2. **OR realization**：同一状态可能通过已有副本传输、增量 suffix transfer、exact recomputation 或等价的本地 representation generation 获得；
3. **shared-link externality**：不同请求和不同状态流竞争同一 CEC backhaul，逐请求选择的局部最优来源可能共同堵塞同一瓶颈。

因此，正确的调度对象不是 request、cache block 或单条 flow，而是一个能够使请求真正可执行的 **state-assembly plan**。

## 3. 第一性原理建模

令 $\mathcal{A}_r$ 表示请求 $r$ 的必要状态集合，$\mathcal{W}_r$ 表示兼容 worker 集合。对状态 $a \in \mathcal{A}_r$、目标 worker $w \in \mathcal{W}_r$，集合 $\mathcal{P}_{r,a,w}$ 包含所有语义等价的实现方式，例如 local hit、transfer、suffix transfer 或 exact recomputation。

一个 assembly plan $p$ 同时确定目标 worker、每项状态的实现方式及其路径。状态 $a$ 的完成时刻为 $C_{r,a}(p)$，则请求的执行就绪时刻为

$$
R_r(p) = \max_{a \in \mathcal{A}_r} C_{r,a}(p).
$$

共享链路存在时，$C_{r,a}(p)$ 不是独立的 size/bandwidth 计算结果，而由所有已接纳 assembly flows 的联合网络调度决定。请求完成时刻可写为

$$
F_r(p) = R_r(p) + Q_{r,w}(p) + T_{r,w}^{\mathrm{infer}}(p),
$$

其中 $Q_{r,w}(p)$ 是执行节点排队时间，$T_{r,w}^{\mathrm{infer}}(p)$ 使用真实 serving profile 估计。系统目标是

$$
\max \sum_r \omega_r y_r,
$$

$$
y_r = \mathbf{1}\left[F_r(p)-t_r^{\mathrm{arr}} \le D_r\right],
$$

并满足逐链路带宽、GPU/HBM、状态版本和兼容性约束。在线算法只能观测当前队列、状态目录和链路占用，不知道未来 arrival、输出长度或 session 数量。

生成式 session 还具有状态转移。请求在 worker $w$ 完成后产生下一版本状态

$$
K_{s,v+1} = U\left(K_{s,v}, I_r, O_r\right),
$$

且最新版本的 owner 变为 $w$。因此当前 assembly decision 会改变下一轮的状态拓扑，但算法不能读取未来请求来为当前决策作弊。这个“消费一个版本化闭包并产生下一版本”的在线状态语义，是 StateWeaver 相对一次性 DAG job 最需要验证的差异；它本身仍不足以自动构成新颖性。

## 4. 关键 sanity check：量级可能重合，但尚未证明场景普遍存在

当前项目模型配置给出的 KV footprint 为：

| 模型 | KV/token | 1K-token KV | 4K-token KV |
|---|---:|---:|---:|
| CodeLlama-34B | 192 KiB | 192 MiB | 768 MiB |
| Qwen2-VL-7B | 56 KiB | 56 MiB | 224 MiB |
| OpenVLA-7B 配置 | 512 KiB | 512 MiB | 2 GiB |

[Chameleon（MICRO 2025）](https://doi.org/10.1145/3725843.3756083) 报告 rank-32 Llama-7B adapter 约为 64 MB，而 rank-128 adapter 可达到数百 MB；这说明 adapter 与中短 context KV 的量级可能重合。当前多模态输入、adapter 和 KV 因而**可能**共同影响 readiness，而不是必然由单一状态支配。

这只是 plausibility check，不是问题重要性的证据。压缩后的图像或短视频可能远小于 KV，adapter 也可能长期命中；即使两个对象都非本地，也不代表二者都会改变调度动作。如果真实 workload 中一个 artifact 长期占 state-ready time 的 80% 以上，或者只改变任一其他 artifact 都不会改变最优 worker、plan 或 SLO 结果，StateWeaver 会退化为普通 KV transfer 或 adapter caching，论文方向应终止。

## 5. 系统设计

StateWeaver 只围绕一个控制原语组织系统：**executable-state assembly plan**。

1. **Versioned State Catalog**  
   记录 KV 的最高连续有效 prefix、adapter 版本、representation 类型、来源、大小、兼容 worker、观测时间和 in-flight 状态。兼容谓词至少检查 model/adapter hash、tokenizer、position encoding、dtype、KV layout、历史 token 和 prefix version；谓词失败的 transfer/recompute 分支不进入候选计划。目录更新延迟、版本失配和控制流量必须计入评估。

2. **Minimal Closure Generator**  
   对每个候选 worker 生成少量最小闭包方案。第一版只允许语义精确的动作：local hit、exact transfer、增量 suffix transfer、exact recomputation 和数值兼容的本地 encoder execution，不引入有损压缩或模型质量选择。每一种 OR 分支都必须通过 continuation-logit/token equivalence test；未经验证的近似重建不能算作可行 plan。

3. **Assembly-Aware Admission and Placement**  
   根据 deadline slack、GPU/HBM 可行性和逐链路 shadow price，联合选择 request、worker 与 assembly plan。算法优化的是完成一个闭包所增加的 SLO-goodput，而不是缓存字节数或单条 flow 完成时间。

4. **Assembly Coflow Data Plane**  
   同一请求的必要状态流以 AND-join coflow 执行；不同请求在共享瓶颈上进行 deadline-aware 排序。资源 reservation 具有短时有效期，失败时回退到 state owner、on-demand recovery 或 recomputation。过期 reservation、取消传输和 catalog reconciliation 的 CPU、链路和时延开销不能视为零。

5. **Unmodified Local Serving Engine**  
   worker 内继续使用 vLLM/SGLang 的 continuous batching。本文不声称新的 batching 算法；本地 batch 状态只用于估计 GPU 启动和 HBM feasibility。

## 6. 与最近工作的精确边界

- [ReSK（IEEE IoT-J 2026）](https://doi.org/10.1109/JIOT.2026.3709703)、[Preble（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html) 和 [LMetric（OSDI 2026）](https://www.usenix.org/conference/osdi26/presentation/zhang-dingyan) 联合考虑 KV/cache locality 与负载，但没有把多个必要 artifact 的 AND-join readiness 和替代实现计划作为调度对象。
- [Chameleon（MICRO 2025）](https://doi.org/10.1145/3725843.3756083) 与 [dLoRA（OSDI 2024）](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang) 处理 adapter caching、batching 或迁移，但不联合组装 ingress observation、session KV 和 adapter。
- [Varys（SIGCOMM 2014）](https://conferences.sigcomm.org/sigcomm/2014/program.php) 调度给定 endpoints 的 coflows；[CLARINET（OSDI 2016）](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/viswanathan) 和 [Sonic（USENIX ATC 2021）](https://www.usenix.org/conference/atc21/presentation/mahgoub) 选择 WAN/workflow data-passing plan。通用 AND-OR DAG placement 已能表达 max-ready、替代分支、worker 和容量约束，因此“能写出该优化问题”不是新抽象。StateWeaver 必须证明版本化状态消费/产生、在线目录不确定性或 generative-state 专用求解结构带来通用 planner 不能等价复现的正确性或性能收益。
- [SHEPHERD（NSDI 2023）](https://www.usenix.org/conference/nsdi23/presentation/zhang-hong)、[PPipe（USENIX ATC 2025）](https://www.usenix.org/conference/atc25/presentation/kong)、Libra 和 JITServe 已显著覆盖 batch-level scheduling，因此 batching 不能作为本文的新颖性来源。
- *Omni-Flow*（arXiv:2606.31093）是 2026 年出现的 multimodal workflow、distributed KV/model sharing preprint，属于阻断性并发风险。完成逐机制、逐约束、逐算法和逐实验比较前，不得使用 “first” 或“此前工作没有组合这些元素”的表述；不能因为它尚未正式发表而忽略 novelty collision。

**待验证的最小技术主张**：

> StateWeaver treats generative inference placement as online executable-state assembly: it jointly selects a worker and an AND-OR realization plan for geographically distributed, version-compatible request state, while scheduling the resulting assembly coflows over shared hierarchical links and preserving session-state transitions.

只有当通用 CLARINET/Varys/Sonic-style joint planner、Chameleon-style state-aware router 和 Omni-Flow 均不能等价复现该机制或收益时，这个技术主张才可能上升为新颖性 claim。

## 7. 建议的论文贡献

1. **Characterization**：系统量化 CEC 中 multi-source state、orphan state、single-artifact dominance 和 scalar-cost ranking error 的出现条件；
2. **Abstraction and model**：提出 executable-state closure 与 state-assembly plan，形式化 AND-join、OR realization、版本兼容和逐链路竞争；
3. **Online system**：实现 assembly-aware placement/admission、deadline-aware assembly coflow 和 versioned data plane；
4. **Evaluation**：在真实多节点原型和 trace-driven workload 中证明收益来自 multi-source assembly，而不是更多带宽、未来信息或更宽松资源预算。

## 8. 最小实验

### 8.1 Phase 0：2–4 周 kill test

- **模型**：至少一个多轮 VLM + 多 adapter workload；一个 text-only LLM 作为预期退化的 negative control；
- **拓扑**：ingress/access、previous-state owner、candidate edge worker、regional store 四类角色；至少两条流共享一个受限 backhaul；
- **真实 profile**：adapter load、KV suffix transfer、history recomputation、representation generation、prefill/decode 和 HBM；
- **预注册输入**：在实验前固定 workload/trace 来源、artifact placement 生成规则、topology、SLO 和负载区间，禁止看到结果后再挑“多源”参数；
- **请求信息**：所有在线方法仅观察当前状态，不读取未来 EOS、future request 或 future mobility；
- **逐 artifact 因果实验**：对同一 request/topology 固定总字节、容量和其他状态，只改变一个 artifact 的来源或 realization，检查最优 worker、瓶颈链路和 SLO 是否改变；使用全因子交互项或 Shapley attribution 区分“共同出现”与“必须联合优化”；
- **小规模 oracle**：枚举 worker 和 state realization plan，求 joint optimum；与信息完全匹配的 strongest decomposition 和 generic joint planner 比较。

### 8.2 强基线

1. state-owner sticky：把新 observation 发往 KV owner；
2. nearest/load-only offloading；
3. Preble/LMetric-like KV-and-load-aware routing；
4. ReSK-like request placement + local KV caching；
5. target-first + optimal coflow：先按精确标量代价选 worker，再最优调度状态流；
6. per-request CLARINET-style plan + shared-link scheduler；
7. generic joint AND-OR DAG placement + coflow optimizer：允许使用相同状态 branches、worker、版本 feasibility 和当前全局信息；
8. joint offline oracle。

所有方法共享相同状态副本、模型、HBM、链路、在线信息和 local serving engine。

### 8.3 主要指标

- SLO-goodput、P95/P99 TTFT 和 end-to-end latency；
- state-ready time、assembly-coflow completion time；
- bytes per SLO saved、recompute FLOPs per SLO saved；
- orphan-state ratio、single-artifact dominance ratio；
- artifact 的主效应与交互效应、assembly time 对违约 TTFT 的归因；
- catalog age、version mismatch、control bytes、plan decision latency、reservation waste 和 fallback rate。

## 9. 三个严格 go/no 条件

以下三项必须全部满足。

1. **场景存在性与因果性**  
   在预注册的至少两个非极端 CEC 拓扑和两个真实或公开来源 workload 中，至少 20% 的 offered requests 需要两个或更多非本地必要状态；其中至少一半不存在占 state-ready time 80% 以上的单一 artifact，且至少两个 artifact 的匹配干预分别会改变最优 worker、plan 或 SLO outcome。Assembly time 还必须解释至少 30% 的尾部 TTFT 或 SLO violations。若只有人为 mobility、刻意打散副本或极端低带宽才能满足，停止。

2. **联合决策必要性**  
   在预注册的中负载区间，joint oracle 相对“target-first + optimal per-request plan/coflow”的 SLO-goodput 中位提升至少 15%，且 95% paired confidence interval 的下界为正。保持请求、总状态字节、容量和 SLO 分布不变，只通过 co-location 或独立链路移除交互后，多源与共享瓶颈的交互收益应下降至少 70%。若 gap 小于 5%，强 scalar/vector-price baseline 可将 gap 压到 3% 内，或 generic joint planner 在同等在线预算下与 StateWeaver 的差距小于 3%，停止“新调度抽象”主张。

3. **在线可落地性**  
   在线 StateWeaver 至少回收 70% 的 oracle gap，并相对最强正式工作风格基线获得至少 10% 的净 SLO-goodput；跨独立 seeds、拓扑和 SLO 点报告配对置信区间及绝对违约数。决策时间低于 scheduling epoch 的 10%，计入 catalog staleness、控制流量和回退后，过期 reservation 与未使用传输仍低于 5%。若只能由理想全局目录、零成本 reservation 或离线未来信息获得收益，停止。

## 10. 明确不做什么

- mobility 只作为可选 workload，不是机制前提；
- 不预测真实剩余 session requests、未来输出长度或未来用户轨迹；
- 不把 SafeRL、Lyapunov、DRL、MILP 或 column generation 当作贡献；
- 不把 KV migration、task scheduling 与 batching 的三项拼接称为新颖性；
- 第一版不做有损 representation、模型精度选择、隐私、能耗或 fairness；
- 不做 proactive preparation；先证明当前请求的 reactive assembly 已有不可忽略的 joint gap。

## 11. Abstract 草稿（10 句）

1. Cloud–edge–continuum platforms can place generative inference close to data sources while pooling heterogeneous accelerators across access and regional sites.  
2. Stateful multimodal requests, however, become executable only when their current observations, compatible adapters, and version-consistent session states are simultaneously available at a worker.  
3. These artifacts may reside at different sites and may be obtained through transfer, incremental synchronization, or exact recomputation.  
4. Existing offloading and cache-aware routing methods reduce these choices to per-request scalar costs and therefore ignore both AND-join readiness and contention among assembly flows on shared hierarchical links.  
5. We formulate executable-state assembly, in which worker placement and alternative state-realization plans are selected jointly under link, GPU-memory, and latency constraints.  
6. We present StateWeaver, an online CEC scheduler that maintains a versioned state catalog and generates a small set of compatible assembly plans for each request.  
7. StateWeaver admits plans according to their SLO-goodput gain per bottleneck resource and executes the selected state transfers as deadline-aware assembly coflows.  
8. The design requires neither future request knowledge nor output-length or mobility oracles and leaves the local continuous-batching engine unchanged.  
9. We evaluate StateWeaver on a heterogeneous multi-tier prototype and trace-driven long-session multimodal workloads against state-sticky, cache-aware, target-first, and coflow-aware baselines under matched resource budgets.  
10. **[Result placeholder]** StateWeaver improves SLO-goodput by $X\%$ and reduces state-ready tail latency by $Y\%$, demonstrating that executable-state assembly is a necessary scheduling abstraction when generative state is geographically fragmented.

## 12. Introduction 段落逻辑

1. **P1 — CEC opportunity**：CEC 同时提供 proximity 和 accelerator pooling，但任务调度必须跨异构节点和层级链路。
2. **P2 — New obstacle**：长 session multimodal inference 的执行依赖多个增长或变化的状态对象，而非单一输入。
3. **P3 — Why existing abstraction fails**：用结构性反例说明 additive offloading cost、KV-only routing 和 target-first coflow 的策略排序错误。
4. **P4 — Core insight**：只有完成最小 executable-state closure 才会创造真实可用的执行容量。
5. **P5 — System**：概述 StateWeaver 的 catalog、plan generator、joint admission 和 assembly coflow。
6. **P6 — Contributions**：依次总结 characterization、abstraction/system 和 evaluation，不声称算法名称本身是贡献。

## 13. 章节与图表预算

| 章节 | 内容 | 主要图表 |
|---|---|---|
| 1. Introduction | 问题、失败案例、洞察与贡献 | Fig. 1 |
| 2. Background and Motivation | CEC、KV/adapter/observation 量级与实测反例 | Fig. 2–3 |
| 3. Model and Problem | AND-OR closure、coflow、SLO objective | Table 1 |
| 4. StateWeaver Design | catalog、plan generation、online selection | Fig. 4 |
| 5. Data Plane and Prototype | versioning、reservation、transfer/recompute | Fig. 5 |
| 6. Evaluation | RQ、主结果、敏感性、消融和开销 | Fig. 6–10，Table 2–3 |
| 7. Related Work | 按机制划界 | 无 |
| 8. Discussion and Limitations | dominance、compatibility、failure 和适用边界 | 无 |

建议总计 10 张主图、3 张表。若核心 joint gap 不能用 Fig. 2–3 清楚证明，后续系统实现没有必要。

## 14. 投稿定位

- **CEC/MEC 主定位**：IEEE TMC、IEEE TSC、INFOCOM、ICDCS；需要真实多层链路、在线算法和原型，而非仅做仿真。
- **更普适的系统定位**：若能证明 executable-state assembly 适用于 edge、federated GPU 和 geo-distributed serving，并实现通用 data plane，可考虑 EuroSys、NSDI 或 MLSys。
- **现实下限**：只有 MILP/DRL + synthetic simulation，不足以支撑上述定位；此时不应把问题包装成系统 paper。
