# 网络（边）模拟设计文档

## 1. 设计目标

网络模块把节点之间的**链路（边）**显式建模为一种**共享、可争用的资源**。它服务于卸载实验中最核心的一组变量：100 Gbps 高速直连与 25 Gbps 跨主机 RDMA 之间的差异如何影响 KV 迁移、请求转发和多模态输入传输的成本。

模块需要回答三类问题：

- 在节点 `src → dst` 之间传输 `bytes` 字节需要多久？（传输时间）
- 当多条传输同时使用同一条链路时，带宽如何分摊？（争用）
- 各链路在一段时间内被占用多少？（利用率，用于报告 100G/25G 链路利用率指标）

设计原则：足够刻画“带宽差异 + 时延 + 争用”的趋势，不追求 packet 级仿真。

## 2. 拓扑与边

网络是一个带权图：**节点为顶点，链路为边**。每条边 `LinkSpec`：

```
src, dst            两端节点 id（无向边，双向可用）
bandwidth_bps       链路带宽（bits/s 输入，内部转 bytes/s）
latency_ms          单向传播/建立时延
name                标签（如 "A-B-100G"、"A-C-25G"）
```

实验默认拓扑（对齐报告第 2 节）：

```
Node A ──100Gbps── Node B
   \                 /
   25Gbps        25Gbps      （Node C 经 25Gbps 跨主机 RDMA 接入 A-B 高速域）
     \             /
        Node C
```

- A–B：100 Gbps，低时延（如 0.05 ms）。
- A–C、B–C：25 Gbps，较高时延（如 0.2 ms）。

> 也可配置为 C 仅与某一节点直连、其余走多跳；模块支持最短路（按时间）多跳传输，但默认三节点全连通。

## 3. 传输时间模型

单条传输（无争用）的时间：

```
transfer_ms = latency_ms + bytes / effective_bandwidth * 1000
```

`effective_bandwidth = bandwidth_bps/8 * link_efficiency`，`link_efficiency` 反映协议/RDMA 开销（默认 0.9）。

多跳路径（若需要）取各边时延之和 + 受最小带宽边约束：

```
path_ms = sum(latency_e) + bytes / min(effective_bandwidth_e) * 1000
```

## 4. 链路争用模型

真实迁移会与其他传输竞争带宽。提供两种精度：

### 4.1 静态份额（默认，给路由器做成本预测）

路由器预测动作成本时使用“当前活跃流数”估计可得带宽：

```
share_bandwidth = effective_bandwidth / max(active_flows + 1, 1)
predict_ms = latency_ms + bytes / share_bandwidth * 1000
```

这让 router 在链路繁忙时自动提高迁移成本估计，倾向于避开拥塞链路。

### 4.2 离散事件占用（仿真执行时）

仿真真正执行一次传输时，按时间窗记录链路占用：

- `reserve(link, bytes, t_start)`：登记一条流，返回预计完成时间。
- 多条并发流按 **max-min fair share** 平分带宽（简化：按当前并发数等分）。
- `release(link, flow)`：流结束，释放份额。
- 累计 `busy_bytes` 与 `busy_ms`，用于计算链路利用率。

> 实现上提供 `transfer_time_ms()`（预测，无副作用）与 `start_transfer()/finish_transfer()`（执行，更新利用率）两套接口，分别给“路由决策”和“事件执行”使用。

## 5. 利用率与指标

每条边维护累计统计，供报告指标使用：

```
total_bytes        累计传输字节
total_busy_ms      累计占用时间
peak_concurrency   峰值并发流数
utilization(window_ms) = total_busy_ms / window_ms     # 近似占用率
throughput(window_ms)  = total_bytes / (window_ms/1000)
```

模块汇总输出 `link_utilization` 字典（按 100G / 25G 分别报告），对应报告评价指标中的“100 Gbps 链路利用率 / 25 Gbps RDMA 链路利用率 / cross-node state transfer ratio”。

## 6. 与 KV / 计算 / 路由的关系

- **KV 迁移**：`kv_cache` 计算需传输的缺失 block 字节数，调用 `network.transfer_time_ms(src, dst, bytes)` 得到 migrate 成本。
- **多模态输入**：VLM/VLA 请求被转发到非入口节点时，输入数据也走链路；模块对图像等大输入提高传输权重，避免频繁经 25G。
- **请求转发**：跨节点执行时，请求/结果的控制面与数据面传输都计入链路占用。
- **路由器**：在 SLA 约束过滤中，用 `transfer_time_ms`（静态份额）预测 `T_network(a)`；执行选定动作时用 `start_transfer/finish_transfer` 更新真实利用率。

## 7. 接口设计

```python
@dataclass(frozen=True)
class LinkSpec:
    src: int; dst: int; bandwidth_bps: float
    latency_ms: float = 0.1; name: str = ""; link_efficiency: float = 0.9
    def effective_bandwidth(self) -> float   # bytes/s

class NetworkTopology:
    def __init__(self, num_nodes: int, links: List[LinkSpec])
    def link(self, a, b) -> LinkSpec
    def path(self, src, dst) -> List[LinkSpec]   # 最短(按时间)路径

class NetworkSimulator:
    def __init__(self, topology: NetworkTopology)
    # 预测（无副作用，供路由成本估计）
    def transfer_time_ms(self, src, dst, num_bytes, contention=True) -> float
    # 执行（更新利用率，供事件循环）
    def start_transfer(self, src, dst, num_bytes, t_now) -> "Flow"
    def finish_transfer(self, flow) -> None
    def link_utilization(self, window_ms) -> Dict[str, float]
    def reset_stats(self) -> None

def default_topology(num_nodes=3) -> NetworkTopology   # A-B/B-C 100G, A-C 25G
```

## 8. 局限

- 不建模丢包、重传、拥塞控制曲线、交换机缓冲；争用用等分/份额近似。
- 多跳仅取最小带宽边与时延和，不建模逐跳排队。
- 对“策略对比下不同链路的相对迁移成本与利用率”足够；对绝对网络性能预测不足。
