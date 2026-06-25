# 实验配置文件设计文档

## 1. 目标

把**节点配置、模型配置、请求生成配置、网络配置**全部抽到一个可手编的 JSON 文件
（`configs/default.json`），改实验不必动代码。loader 在 `sim/config.py`，纯标准库
（选 JSON 而非 YAML 以避免第三方依赖；字段含义见本文）。

用法：

```bash
python -m sim.config                          # 重新生成 configs/default.json
python -m sim.dashboard --config configs/default.json   # 用指定配置跑实验
python -m sim.dashboard                       # 不传则用内置默认（等价于 default.json）
```

代码内：

```python
from sim import load_config, run_experiments
cfg = load_config("configs/default.json")
cfg.workload.mobility_ratio = 0.5      # 任意修改
data = run_experiments(cfg)
```

## 2. 顶层结构

```json
{
  "cluster":  { ... 节点/集群配置 ... },
  "policies": ["nearest","greedy","long_term","long_term_kv"],
  "hardware": { ... 硬件配置 ... },
  "models":   [ { ... 模型配置 ... }, ... ],
  "network":  { "links": [ { ... 链路配置 ... }, ... ] },
  "workload": { ... 请求生成配置 ... }
}
```

加载时 `ExperimentConfig.apply()` 会把 `models` 与 `hardware` 注册进全局注册表，
使数据生成器、计算模拟器、路由器全部使用配置中的规格（保证一致）。

## 3. cluster（节点/集群配置）

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `num_nodes` | 节点数（须与 network.links 覆盖的节点一致） | 3 |
| `staleness_ms` | 共享状态目录同步周期；>0 时 router 读到的状态可能滞后 | 0 |
| `kv_capacity_bytes` | 每节点 KV 容量；`null` 表示按 `HBM - 权重 - 激活预留` 自动推算 | null |
| `activation_reserve_bytes` | 为激活值预留的显存 | 4e9 |

> 改 `num_nodes` 必须同时在 `network.links` 里补齐新节点的链路，否则路由找不到路径会报错。

## 4. policies（参与对比的策略）

字符串数组，取值：`nearest` / `greedy` / `long_term` / `long_term_kv`。
可只保留子集（例如只比较 `greedy` 与 `long_term_kv`）。

## 5. hardware（硬件配置 → `HardwareSpec`）

| 字段 | 含义 |
| --- | --- |
| `name` | 硬件名（如 A800T-A2） |
| `num_devices` | 单节点加速卡数 |
| `peak_flops_per_device` | 单卡 BF16 峰值算力（FLOPS，支持 `3.76e14`） |
| `mem_bandwidth_per_device` | 单卡 HBM 带宽（B/s） |
| `mem_capacity_per_device` | 单卡 HBM 容量（B） |
| `compute_efficiency` | prefill 计算利用率 MFU（0~1） |
| `bandwidth_efficiency` | decode 带宽利用率（0~1） |
| `interconnect_bandwidth` | 卡间互联带宽（B/s） |
| `fixed_overhead_ms` | 每次内核启动固定开销 |

## 6. models（模型配置 → `ModelSpec`）

数组，每个元素描述一个大模型。**已在注册表中的模型名**（CodeLlama34B /
Qwen2-VL-7B-Instruct / OpenVLA-7B）可只写要覆盖的字段，其余沿用内置默认；
**全新模型**需提供完整字段。

| 字段 | 含义 |
| --- | --- |
| `name` | 模型名（数据/工作负载据此引用） |
| `model_type` | `LLM` / `VLM` / `VLA` |
| `num_params` | 参数量（prefill 计算量） |
| `num_layers`/`hidden_size`/`num_attention_heads`/`num_kv_heads`/`head_dim` | Transformer 结构（KV 体积、FLOPs） |
| `intermediate_size`/`vocab_size`/`dtype_bytes` | FFN/词表/精度字节 |
| `weight_bytes` | 权重总字节；`null` 则按 `num_params*dtype_bytes` |
| `vision_params`/`patch_size`/`spatial_merge`/`tokens_per_image` | 视觉编码器（VLM/VLA） |
| `default_output_dist` | 默认输出长度分布 `{kind,mean,std,minimum,maximum}` |
| `default_sla_ms` | 默认 P99 TTFT SLA |
| `kv_block_size` | KV block 的 token 粒度（与 KV 管理对齐） |

## 7. network（网络/链路配置 → `LinkSpec` 列表）

`network.links` 是无向边数组：

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `src`/`dst` | 两端节点 id（0-based） | — |
| `bandwidth_bps` | 链路带宽（bits/s，支持 `100e9`） | — |
| `latency_ms` | 单向时延 | 0.1 |
| `name` | 标签（如 A-B-100G，用于看板显示） | "" |
| `link_efficiency` | 有效带宽系数（协议/RDMA 开销） | 0.9 |

默认拓扑：A–B 100Gbps 直连，A–C / B–C 各 25Gbps 跨主机 RDMA。

## 8. workload（请求生成配置 → `WorkloadConfig`）

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `duration_ms` | 实验总时长 | 60000 |
| `seed` | 随机种子（同种子→同轨迹，四策略公平对比） | 0 |
| `mobility_start_frac` | 用户移动起始时刻占比 | 0.5 |
| `mobility_ratio` | 移动后切换入口的请求比例 | 0.2 |
| `mobility_granularity` | `request`（逐请求）或 `session`（整会话）切换 | request |
| `groups` | 工作负载分组数组（见下） | 3 组 |

每个 `group`（`WorkloadGroup`）：

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `model_name` | 引用的模型名 | — |
| `priority` | `high` / `normal` | normal |
| `concurrency` | 并发数（= 生成的 session 数） | 24 |
| `sla_ms` | 覆盖模型默认 SLA；`null` 用模型默认 | null |
| `arrival_rate` | 到达率（请求/秒）；`null` 由并发÷时长推导 | null |
| `prompt_dist` | prompt 长度分布 `{kind,mean,std,minimum,maximum}` | — |
| `output_dist` | 输出长度分布；`null` 用模型默认 | null |
| `turns_mean`/`turns_min`/`turns_max` | 每会话轮数分布 | 4/1/12 |
| `image_size` | `[宽,高]`（VLM 视觉 token） | [0,0] |
| `num_frames` | 帧数（VLM/VLA） | 1 |
| `shared_prefix_tokens` | 组内共享 system prompt 长度 | 0 |
| `history_growth` | 每轮并入前缀的历史比例 | 0.6 |

`kind` 取值：`fixed` / `normal` / `lognormal`。

## 9. 常见改配示例

- **加大移动强度**：`workload.mobility_ratio = 0.4`。
- **只比较两种策略**：`policies = ["greedy","long_term_kv"]`。
- **收紧高优 SLA**：把 high 分组的 `sla_ms` 改小（如 100）。
- **限制显存看不可执行率**：把 `cluster.kv_capacity_bytes` 设为较小值（如 2e10）。
- **启用 VLA**：在 `workload.groups` 增加一个 `model_name="OpenVLA-7B"` 的分组。
- **改网络对比**：把 `B-C-25G` 的 `bandwidth_bps` 调到 100e9，看跨主机链路提速的影响。

## 10. 与代码的关系

`sim/config.py` 提供 `load_config / save_config / default_config / from_dict / to_dict`
与 `ExperimentConfig`（含 `apply()` 注册、`new_network()` 构造）。`dashboard.run_experiments`
接受 `ExperimentConfig`，据此生成轨迹、构建集群、回放所有策略并产出看板。
