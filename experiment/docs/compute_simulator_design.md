# GPU/NPU 计算模拟器设计文档

## 1. 设计目标

计算模拟器是整个卸载实验的“执行时间与显存”内核。它的职责不是逐算子精确仿真，而是在离散事件模拟器中为每个请求快速、可解释地估算：

- **prefill 时间**（决定 TTFT 的主体）。
- **decode 时间**（每 token TPOT 与端到端时延）。
- **显存占用**（权重 + KV cache + 激活），用于 GPU memory 约束过滤。
- **KV 迁移 / 重算时间**（配合调度器的 local/migrate/recompute 决策）。

模拟器必须做到：参数可配置、对 LLM/VLM/VLA 三类模型一致、对 A800T-A2（Ascend 910B 级）这类 NPU 与通用 GPU 都适用。因此采用 **roofline（计算-访存上界）模型** 而非查表，使其在不同硬件、不同 batch、不同上下文长度下都能给出合理趋势。

## 2. 为什么用 roofline 模型

LLM 推理的两个阶段具有截然不同的瓶颈：

- **prefill**：一次处理整段 prompt，矩阵乘规模大，通常 **计算受限（compute-bound）**。
- **decode**：每步只产生 1 个 token，但要把全部权重和已有 KV 从 HBM 读出，通常 **访存受限（memory-bound）**。

roofline 模型用一句话统一两者：

```
time = max(compute_flops / effective_compute, access_bytes / effective_bandwidth)
```

其中 `effective_*` 由峰值能力乘以利用率（MFU / 带宽利用率）得到。这样既能反映“prefill 看算力、decode 看带宽”的本质，又只需要少量硬件参数。

## 3. 硬件规格（HardwareSpec）

描述单个推理节点的算力与显存能力，字段如下：

```
name                  硬件名（如 A800T-A2）
num_devices           节点内加速卡数量（如 8）
peak_flops_per_device 单卡峰值算力（BF16 FLOPS）
mem_bandwidth_per_device  单卡 HBM 带宽（B/s）
mem_capacity_per_device   单卡 HBM 容量（B）
compute_efficiency    计算利用率 MFU（prefill 有效算力系数，如 0.5）
bandwidth_efficiency  带宽利用率（decode 有效带宽系数，如 0.7）
interconnect_bandwidth  卡间互联带宽（B/s，张量并行时折损）
fixed_overhead_ms     每次内核启动/调度固定开销
```

派生有效能力（聚合整节点）：

```
effective_compute   = num_devices * peak_flops_per_device * compute_efficiency
effective_bandwidth = num_devices * mem_bandwidth_per_device * bandwidth_efficiency
total_memory        = num_devices * mem_capacity_per_device
```

### 3.1 预置硬件：A800T-A2

以 Ascend 910B 级为参考量级（可在配置中覆盖）：

- num_devices = 8
- peak_flops_per_device ≈ 376 TFLOPS（BF16）
- mem_bandwidth_per_device ≈ 1.6 TB/s
- mem_capacity_per_device ≈ 64 GB
- compute_efficiency = 0.5，bandwidth_efficiency = 0.7

> 这些值用于相对趋势分析与策略对比，不追求绝对精度；实验中关心的是不同卸载策略在同一硬件模型下的差异。

## 4. 计算模型

模型结构量全部来自 `large_model.ModelSpec`，硬件能力来自 `HardwareSpec`，计算模拟器只负责把两者套进 roofline。

### 4.1 prefill

对一个 batch，设其总 prompt token 数为 `P`（多请求合并）：

```
flops  = sum_r model.prefill_flops(P_r)          # 逐请求求和
t_comp = flops / effective_compute
bytes  = weight_bytes + model.kv_bytes_for_tokens(P) # 写入新 KV
t_mem  = bytes / effective_bandwidth
t_prefill = max(t_comp, t_mem) + fixed_overhead
```

prefill 通常 `t_comp` 占主导。TTFT ≈ 排队时间 + 状态获取时间 + t_prefill。

### 4.2 decode

decode 以 batch 形式逐步生成。设 batch 内有 `B` 个序列、平均上下文长度 `L`、本步生成 1 token：

```
flops  = B * model.decode_flops_per_token(L)
t_comp = flops / effective_compute
bytes  = weight_bytes + B * model.kv_bytes_per_token() * L  # 权重读一次，KV 按序列
t_mem  = bytes / effective_bandwidth
t_step = max(t_comp, t_mem) + fixed_overhead
```

单请求生成 `G` 个 token 的 decode 时间 ≈ `G * t_step`（上下文随步数增长，可按平均或逐步累加）。decode 几乎总是 `t_mem` 主导，正是 batching 能摊薄权重读取、提高吞吐的原因。

### 4.3 端到端时延

```
TTFT      = T_queue + T_state + t_prefill
E2E       = TTFT + decode_total
decode_total = sum_{g=1..G} t_step(L0 + g)
```

`T_state` 由调度动作决定：local（≈0，仅本地读取）、migrate（KV 传输时间）、recompute（等价于对已有前缀再做一次 prefill）。

## 5. 显存模型

GPU memory 约束过滤需要预测动作执行后的显存占用：

```
mem_weights   = model.total_weight_bytes()         # 常驻
mem_kv        = sum over resident sessions of kv_bytes_for_tokens(ctx_len)
mem_activation= activation_factor * batch_tokens * hidden_size * dtype_bytes
mem_used      = mem_weights + mem_kv + mem_activation
mem_free      = total_memory - mem_used - mem_reserve
```

接口提供：给定请求的新增 token 数，返回新增 KV 显存；据此判断 `M_hat + M_reserve <= M_free`。

## 6. KV 迁移与重算成本

配合调度器的 local/migrate/recompute：

- **local**：`T_state ≈ 0`（命中本地 KV），仅需块查表。
- **migrate**：只传输目标节点缺失的 blocks：
  ```
  bytes_to_move = missing_blocks * kv_block_size * kv_bytes_per_token
  T_migrate = bytes_to_move / link_bandwidth + link_latency
  ```
  link_bandwidth 由网络拓扑给出（100 Gbps 直连 / 25 Gbps 跨主机）。
- **recompute**：在目标节点对可复用前缀重新 prefill：
  ```
  T_recompute = prefill_time(reusable_prefix_tokens)
  ```

模拟器对外暴露 `kv_transfer_time(num_tokens, link_bandwidth_bps, latency_ms)` 与 `recompute_time(prefix_tokens)`，由网络/KV 管理模块组合调用。

## 7. Batching 模型

为反映服务系统的连续批处理（continuous batching），模拟器支持：

- `estimate_prefill_batch(requests)`：合并多请求 prompt，受 `max_batch_tokens` 限制。
- `estimate_decode_step(batch_size, avg_ctx_len)`：返回一步 decode 时间与吞吐。
- 输出 `throughput_tokens_per_s`，供吞吐/并发指标统计。

batch 越大，decode 阶段权重读取被摊薄越充分，单 token 时延下降、吞吐上升，但 TTFT 可能因排队增加。这一权衡正是第五章 batching 与第六章 offloading 联合作用的体现。

## 8. 接口设计

```python
@dataclass
class HardwareSpec:
    name: str
    num_devices: int
    peak_flops_per_device: float
    mem_bandwidth_per_device: float
    mem_capacity_per_device: float
    compute_efficiency: float = 0.5
    bandwidth_efficiency: float = 0.7
    interconnect_bandwidth: float = 400e9
    fixed_overhead_ms: float = 0.2
    def effective_compute(self) -> float
    def effective_bandwidth(self) -> float
    def total_memory(self) -> float

@dataclass
class PrefillResult:  prefill_ms; flops; bytes; bound  # "compute"/"memory"
@dataclass
class DecodeResult:   step_ms; total_ms; throughput_tokens_per_s; bound

class ComputeSimulator:
    def __init__(self, model: ModelSpec, hw: HardwareSpec)
    def estimate_prefill(self, prompt_tokens, batch_tokens=None) -> PrefillResult
    def estimate_decode(self, gen_tokens, ctx_len, batch_size=1) -> DecodeResult
    def kv_cache_bytes(self, num_tokens) -> int
    def weight_bytes(self) -> int
    def memory_usage(self, resident_tokens, batch_tokens=0) -> dict
    def kv_transfer_time_ms(self, num_tokens, link_bps, latency_ms=0.0) -> float
    def recompute_time_ms(self, prefix_tokens) -> float

HARDWARE_REGISTRY: Dict[str, HardwareSpec]   # 预置 A800T-A2
```

## 9. 与其他模块的关系

- 输入依赖 `large_model.ModelSpec`（结构/KV/FLOPs）。
- 被离散事件模拟器在 `prefill_start/end`、`decode_*`、`state_transfer_*` 事件中调用。
- `kv_transfer_time_ms` / `memory_usage` 直接服务于约束过滤与 migrate/recompute 成本估算。

## 10. 校验与局限

- **校验**：可用已知模型在已知硬件上的公开吞吐/TTFT 量级做粗校准，调节 `compute_efficiency` 与 `bandwidth_efficiency`。
- **局限**：不建模算子融合、pipeline 气泡、调度抖动、张量并行通信细节；这些以效率系数和固定开销近似。对“策略对比”足够，对“绝对性能预测”不足。
