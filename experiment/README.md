# 长期成本感知请求卸载模拟实验

本目录实现第 6.4 节“长期成本感知请求卸载方案”的离散事件模拟实验底层组件，
对应实验报告 `long_term_cost_aware_offloading_simulation_report.md`。

## 目录结构

```
experiment/
├── long_term_cost_aware_offloading_simulation_report.md   # 实验总报告
├── docs/
│   ├── architecture_design.md         # 整体架构（节点/路由/元数据/事件）
│   ├── large_model_design.md          # 大模型规格设计（LLM/VLM/VLA）
│   ├── compute_simulator_design.md    # GPU/NPU 计算模拟器设计
│   ├── network_design.md              # 网络（边）模拟设计
│   ├── data_generator_design.md       # 数据生成器设计
│   ├── kv_cache_design.md             # block 级 KV cache 管理设计
│   └── router_design.md               # 路由器设计（动作/成本/约束/四策略）
├── sim/
│   ├── __init__.py
│   ├── large_model.py                 # 大模型规格（单一事实来源）
│   ├── compute_simulator.py           # roofline 计算/访存/显存模型
│   ├── network.py                     # 链路带宽/时延/争用/利用率
│   ├── kv_cache.py                    # block 级 KV store + 全局目录 + 迁移规划
│   ├── data_generator.py              # 可复现工作负载轨迹生成
│   ├── node.py                        # ServingNode + 共享状态目录（含 staleness）
│   ├── router.py                      # 路由器：动作枚举+约束过滤+四策略+回放评估
│   ├── dashboard.py                   # 跑全部策略并生成自包含 HTML metrics 看板
│   └── config.py                      # 可手编 JSON 配置的加载/保存/默认
├── configs/
│   └── default.json                   # 节点/模型/网络/工作负载统一配置（手编）
├── output/                            # 生成物：dashboard.html / metrics.json（git 忽略）
├── demo.py                            # 端到端演示（含四策略对比 + 生成看板）
└── README.md
```

## 整体架构（简述）

- 每个 **ServingNode** 加载一个大模型实例，构成一个推理服务（队列 + 计算模拟器 + 本地 KV store）。
- 每个节点有一个 **Router**，做请求路由与 local/migrate/recompute 决策（分布式路由，非单一全局调度器）。
- 一个**共享元数据目录**（`GlobalKVDirectory` 等）聚合全网负载、显存、网络、prefix/KV 状态，周期性同步、含 staleness。
- **DataGenerator** 离线生成请求轨迹，事件驱动器按 `arrival_ms` 把请求注入到入口节点的 router。

完整说明见 `docs/architecture_design.md`。

## 模块依赖关系

```
large_model ──(结构/KV/FLOPs/block)──▶ compute_simulator ─┐
     │                                                     ├─▶ node ──▶ router ──▶ 四策略决策
     ├──(输入构成/输出分布)──▶ data_generator ──(轨迹)──────┤            ▲
     └──(block 大小/KV 字节)──▶ kv_cache ◀──(迁移定价)── network ────────┘
```

## 预置配置

- 硬件：`A800T-A2`（8×Ascend 910B 级，BF16 ≈376 TFLOPS/卡，HBM ≈1.6 TB/s、64 GB/卡）。
- 模型：`CodeLlama34B`（LLM）、`Qwen2-VL-7B-Instruct`（VLM）、`OpenVLA-7B`（VLA）。
- 网络：A–B / B–C 为 100 Gbps 直连，A–C 为 25 Gbps 跨主机 RDMA。
- 负载：按接入点配置并发（单点 1–256），默认含高优/普通 CodeLlama34B、Qwen2-VL 与固定长度 OpenVLA。

## 运行

仅依赖 Python 3 标准库，无需第三方包。

```bash
cd experiment
python demo.py                  # 端到端演示（含 migrate vs recompute、四策略对比）
python -m sim.large_model       # 大模型规格一览
python -m sim.compute_simulator # 各模型 prefill/decode 估算
python -m sim.network           # 链路传输时间与争用
python -m sim.kv_cache          # block 级 KV 迁移规划
python -m sim.data_generator    # 工作负载摘要
python -m sim.node              # 集群节点状态一览
python -m sim.router            # 四策略在同一轨迹上的指标对比
python -m sim.dashboard         # 跑全部策略并生成 output/dashboard.html
python -m sim.dashboard --open  # 生成后在浏览器打开
python -m sim.dashboard --config configs/default.json  # 用手编配置跑
python -m sim.config            # 重新生成 configs/default.json
```

## 统一配置文件

节点、模型、网络、请求生成四类配置全部抽到 `configs/default.json`，改实验不必动代码：

- `cluster`：节点数、状态同步 staleness、KV 容量、激活预留；
- `policies`：参与对比的策略子集；
- `hardware`：单卡算力/带宽/显存与效率系数；
- `models`：LLM/VLM/VLA 结构与 KV/SLA（已知模型可只写覆盖字段）；
- `network.links`：链路带宽/时延（100G 直连、25G 跨主机）；
- `workload`：时长、种子、移动性、各分组并发/SLA/长度分布。

字段含义见 `docs/config_design.md`。Long-term 策略的 Router 与 KV Cache
联合优化方案见
[Long-term Router 与 KV Cache 管理联合优化方案](docs/long_term_optimization.md)。用法：

```bash
python -m sim.dashboard --config configs/default.json
```

VLM/VLA 的长期放置对照场景使用同构算力和既有三节点拓扑，仅放大多模态
payload、会话轮数与移动后的驻留时间。`8192 bytes/visual token` 表示入口
完成视觉编码后转发 BF16 hidden embedding；它不是原始图片的每 token 大小：

```bash
python -m sim.dashboard --config configs/multimodal_long_term.json
```

或代码内 `load_config(path)` 后修改再 `run_experiments(cfg)`。

## Metrics 看板

交互式控制台推荐用法：

```bash
python -m sim.dashboard_server --open
```

它会先打开参数配置页，展示完整实验 JSON 与常用参数快捷项；点击“开始模拟”后才运行实验，
完成后展示摘要表，并提供跳转按钮打开 `metrics.json` 与完整 `dashboard.html`。
快捷项覆盖 cluster、mobility、router gamma/SLA margin、token id bytes、request/response 固定开销，以及
Data Generator 的分组并发、到达率、SLA、轮数和输入/输出长度分布。

`python -m sim.dashboard` 会在同一条请求轨迹上回放四种策略（覆盖所有模型），
产出两份文件到 `output/`：

- `dashboard.html`：**自包含**交互式看板（数据内联、纯前端、无依赖，可直接分享）。包含
  KPI 卡片、策略汇总对比表（逐列高亮最优）、每阶段独立的 E2E ECDF 分布曲线、排队来源拆分、
  关键指标柱状图、状态获取动作分布、
  **P99 TTFT 时间序列**、**累计跨节点请求（状态黏附）曲线**（标注用户移动时刻）、链路利用率。
- `metrics.json`：结构化指标 + 逐请求记录，便于二次分析。

看板里最直观的是“黏附曲线”：用户移动后 Greedy 曲线明显更陡（请求被吸到远端节点），
long-term 策略更平缓——直接对应实验报告要验证的现象。

可用 `--mobility-granularity request|session|markov` 切换移动语义。`request` 表示移动窗口后逐请求随机切换入口；
`session` 表示一个会话移动到固定新入口，后续请求持续从该入口进入，更适合验证“用户从 A 迁到 B 后
长期驻留”的场景；`markov` 表示在当前入口驻留指定轮数后，再按概率迁往其他入口。

策略边界可用敏感性扫描查看：

```bash
python -m sim.policy_sweep
python -m sim.policy_sweep --mobility-granularity session --gammas 0.1,0.3,0.5,0.9
```

`demo.py` 展示两处核心权衡：
1. 同一份 2048-token KV，100G 链路上 migrate 比 recompute 便宜，25G 上 recompute 更划算；
2. 四策略对比中，greedy 把请求吸到 KV 所在远端节点形成**状态黏附**（高 cross-node、高迁移字节），
   long-term 更早把 KV 迁向未来入口，cross-node 与迁移字节显著下降。

> 注：演示里 `avg_e2e` 被 decode（batch=1）主导，真正区分策略的是 p99 TTFT、cross-node、迁移字节与 owner 切换。

## 后续

router 已落地（`node.py` + `router.py`，含四策略与回放评估 `simulate_trace`）。
下一步可实现完整事件循环 `simulator.py`（continuous batching、并发 decode、显存压力下的 SLA 违约、
消融与敏感性实验），详见 `docs/architecture_design.md` 与实验报告第 11 节。
