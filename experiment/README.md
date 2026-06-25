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
│   └── router.py                      # 路由器：动作枚举+约束过滤+四策略+回放评估
├── demo.py                            # 端到端演示（含四策略对比）
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
- 网络：A–B 100 Gbps 直连，C 经 25 Gbps 跨主机 RDMA 接入。
- 负载：高优 CodeLlama34B(并发24,SLA150ms)、普通 CodeLlama34B(并发96,SLA500ms)、Qwen2-VL(并发24,SLA500ms)。

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
