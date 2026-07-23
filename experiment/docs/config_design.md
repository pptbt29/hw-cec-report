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
  "kv_manager": { ... 后台 KV 预放置配置 ... },
  "policies": ["nearest","greedy","greedy_kv","long_term","long_term_kv"],
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
| `activation_reserve_bytes` | 为激活值、通信 workspace 和运行时碎片预留的整机显存 | 64e9 |
| `prefill_batch_size` | Prefill continuous batching 的目标批大小；仅摊销权重读取和固定启动开销 | 4 |
| `sla_slack_scheduling` | 启用非抢占式 SLA-slack 队列；运行中的任务不被打断，等待任务按最晚可开始时间升序调度 | true |

> 改 `num_nodes` 必须同时在 `network.links` 里补齐新节点的链路，否则路由找不到路径会报错。

`router.decode_batch_size` 默认为 32。当前模拟不维护动态 continuous-batching 状态，直接将固定满批的一次 decode 迭代成本按 batch size 摊分，作为单请求平均 decode 服务时间。

## 4. policies（参与对比的策略）

字符串数组支持：`nearest` / `greedy` / `greedy_kv` / `long_term` / `long_term_kv` /
`greedy_rollout` / `oracle_prefetch` / `oracle_kv`。`greedy_rollout` 对当前动作进行
单步分支，并让同一 session 的全部真实剩余请求采用 Greedy 路由，以未折扣累计 E2E
评价当前动作；它用于敏感性分析，默认不加入主实验。
可只保留子集（例如只比较 `greedy` 与 `long_term_kv`）。

- `long_term`：长期 Router + 被动 KV 恢复，不运行后台主动 placement；block-level 机制默认关闭。
- `greedy_kv`：Greedy 即时路由 + block-level KV 管理和后台主动 placement。
- `long_term_kv`：使用同一长期路由逻辑，并启用 block-level KV 管理和后台主动 placement。

因此，`kv_manager.enabled=true` 会为 `greedy_kv` 和 `long_term_kv` 启用相同主动预放置；它不会改变
`long_term` 的被动 KV 管理语义。

## 5. kv_manager（后台增量 KV 预放置）

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `enabled` | 是否启用请求完成后触发的后台 KV 预放置 | true |
| `base_replication_factor` | 动态复制比例的基础系数 \(\alpha_0\) | 2.0 |
| `max_replication_fraction` | 单个目标节点单轮最大复制比例 | 0.50 |
| `background_bandwidth_fraction` | 每条链路允许后台 KV 任务使用的总带宽比例 | 0.20 |
| `continuation_reference_turns` | session 剩余请求数归一化参考值 | 4.0 |
| `min_window_ms` / `max_window_ms` | 单轮后台复制窗口的上下界 | 1 / 5000 |

第 \(k\) 个请求完成后，目标节点 \(j\) 的复制比例为

```text
alpha(k,j) = clip(alpha0 * p(entry,j) * session_activity, 0, alpha_max)
```

实际复制量进一步受后台时间窗、链路预算和目标 KV 容量约束。同一路径同一时刻只运行一个后台任务，因此所有 session 的叠加流量不会突破配置的链路后台份额。

## 6. hardware（硬件配置 → `HardwareSpec`）

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

## 7. models（模型配置 → `ModelSpec`）

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

## 8. network（网络/链路配置 → `LinkSpec` 列表）

`network.links` 是无向边数组：

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `src`/`dst` | 两端节点 id（0-based） | — |
| `bandwidth_bps` | 链路带宽（bits/s，支持 `100e9`） | — |
| `latency_ms` | 单向时延 | 0.1 |
| `name` | 标签（如 A-B-100G，用于看板显示） | "" |
| `link_efficiency` | 有效带宽系数（协议/RDMA 开销） | 0.9 |

默认拓扑：A–B 100Gbps 直连，A–C / B–C 各 25Gbps 跨主机 RDMA。

## 9. workload（请求生成配置 → `WorkloadConfig`）

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `duration_ms` | 实验总时长 | 60000 |
| `session_start_spread_frac` | session 起始时间铺开的范围；0.8 表示在前 80% 实验时长内启动 | 0.8 |
| `seed` | 随机种子（同种子→同轨迹，四策略公平对比） | 0 |
| `mobility_start_frac` | 用户移动起始时刻占比 | 0.5 |
| `mobility_ratio` | 移动后切换入口的请求比例 | 0.2 |
| `mobility_granularity` | `request`（逐请求抖动）、`session`（整会话一次迁移）或 `markov`（驻留后继续迁移） | markov |
| `mobility_residency_turns` | `markov` 模式下在当前入口至少驻留的会话轮数 | 0 |
| `groups` | 工作负载分组数组（见下） | 4 组 |

`request` 模式用于模拟短时、随机的入口抖动；`session` 模式会为被选中的会话固定一个非 home
新入口，移动窗口后的后续请求都从该入口进入，用于模拟用户从 A 持续迁移到 B 的场景。也就是说，
`session` 模式下移动比例是“每个 session 被选中迁移的概率”，被选中后不会在后续每轮继续按相同概率二次跳转。
`markov` 模式维护每个 session 的当前入口，在当前入口至少驻留 `mobility_residency_turns` 轮，再按
`mobility_ratio` 概率迁到另一个入口。

## 9.1 router（路由代价配置）

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `gamma` | long-term future value 权重；当前有限 session 启发式不折扣后续请求 | 1.0 |
| `sla_margin_ms` | SLA 判断安全裕量 | 20 |
| `long_term_block_level_kv` | 是否为 `long_term` 启用 block-level 缺失块同步；用于单独测量 block 机制增益 | false |
| `long_term_kv_block_level_kv` | 是否为 `long_term_kv` 启用 block-level 缺失块同步 | true |
| `token_id_bytes` | 请求/响应 token id 的字节数；输入输出 token 使用同一值 | 4 |
| `request_overhead_bytes` | 跨节点转发请求时的固定 RPC/协议开销 | 4096 |
| `response_overhead_bytes` | 跨节点返回响应时的固定 RPC/协议开销 | 4096 |
| `visual_bytes_per_token` | VLM/VLA 每个视觉 token 对应的压缩图片/视频输入 payload；设为 0 可模拟已在入口侧编码的视觉 token | 512 B |

每个 `group`（`WorkloadGroup`）：

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `model_name` | 引用的模型名 | — |
| `name` | 模型下的配置组名；可命名为 high/normal 以兼容优先级语义，也可使用任意业务名 | default |
| `entry_mode` | `counts`（具体数量）或 `ratios`（比例 + 组总并发） | counts |
| `concurrency` | `ratios` 模式的组总并发；`counts` 模式下由各入口数量求和 | 24 |
| `entry_concurrency` | `counts` 模式下每个接入点的并发用户数，单点 1–256 | null |
| `entry_ratios` | `ratios` 模式下各入口相对比例，如 `[1,2,1]` | null |
| `sla_ms` | 覆盖模型默认 SLA；`null` 用模型默认 | null |
| `arrival_rate` | 到达率（请求/秒）；`null` 由并发÷时长推导 | null |
| `inter_turn_mean_ms` | 同一 session 相邻请求的平均间隔；与组级到达率分开建模 | null（兼容旧逻辑） |
| `prompt_dist` | prompt 长度分布 `{kind,mean,std,minimum,maximum}` | — |
| `output_dist` | 输出长度分布；`null` 用模型默认 | null |
| `turns_dist` | session 请求数分布 `{kind,mean,std,minimum,maximum}`；配置后优先于旧式三个字段 | null |
| `turns_mean`/`turns_min`/`turns_max` | 兼容旧配置的截断正态轮数参数 | 4/1/12 |
| `image_size` | `[宽,高]`（VLM 视觉 token） | [0,0] |
| `num_frames` | 帧数（VLM/VLA） | 1 |
| `shared_prefix_tokens` | 组内共享 system prompt 长度 | 0 |
| `history_growth` | 每轮输入与输出并入下一轮 prefix 的比例；1 表示完整保留 | 1.0 |

`kind` 取值：`fixed` / `normal` / `lognormal`。

默认 CodeLlama workload 使用 coding-agent-oriented profile：公共 system prompt 在三个同模型节点上预计算并常驻，不计入各 session 的动态 prefix；每轮新增输入采用
均值 512、标准差 512、范围 32--4096 的截断 lognormal 分布；输出采用均值 384、标准差
384、范围 16--4096 的截断 lognormal 分布。高优与普通 session 的轮数分别采用均值 8/12 的
截断 lognormal 分布，历史保留比例默认为 1.0。该配置用于覆盖长状态迁移场景，不代表普通聊天
服务的总体均值。

## 10. 常见改配示例

- **加大移动强度**：`workload.mobility_ratio = 0.4`。
- **只比较两种策略**：`policies = ["greedy","long_term_kv"]`。
- **收紧高优 SLA**：把 high 分组的 `sla_ms` 改小（如 100）。
- **限制显存看不可执行率**：把 `cluster.kv_capacity_bytes` 设为较小值（如 2e10）。
- **VLA 固定长度**：默认 OpenVLA 分组使用 fixed prompt/output；也可把该组的 fixed 均值改成目标动作长度。
- **改网络对比**：默认 A-B/B-C 为 100G、A-C 为 25G；可调整任一 `bandwidth_bps` 做链路消融。

## 11. 与代码的关系

`sim/config.py` 提供 `load_config / save_config / default_config / from_dict / to_dict`
与 `ExperimentConfig`（含 `apply()` 注册、`new_network()` 构造）。`dashboard.run_experiments`
接受 `ExperimentConfig`，据此生成轨迹、构建集群、回放所有策略并产出看板。
