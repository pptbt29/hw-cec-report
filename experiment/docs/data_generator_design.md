# 数据生成器（Data Generator）设计文档

## 1. 设计目标

数据生成器为离散事件模拟器产生可复现的**请求工作负载轨迹（workload trace）**。它必须同时满足实验场景的多项要求：

- 支持 **LLM / VLM / VLA** 三种大模型类别，按各自的输入模态构成 token。
- 由于不是真实 decode，每个请求需要**预设生成长度**，且生成长度来自**可配置的概率分布**（固定 / 正态 / 对数正态）。
- 组织为 **session（会话）→ request（请求）** 两级结构，使连续请求共享 prefix，从而能模拟 prefix/KV cache 复用。
- 建模**用户移动**：测试过半后，部分 session/request 的入口节点发生切换，制造 KV owner 与入口分离。
- 携带 **SLA、优先级、到达时间、prefix hash** 等调度所需字段。
- 完全由随机种子控制，保证四种策略在**同一条轨迹**上对比。

## 2. 为什么需要“预设生成长度分布”

真实系统中输出长度由模型自行决定，模拟实验无法真正解码，因此必须为每个请求预先采样一个生成长度 `output_len`。三类模型的输出特征不同：

- **LLM**（代码/对话）：长尾分布，少数请求很长，建议 **lognormal**。
- **VLM**（图文问答/描述）：中等长度，建议 **lognormal 或截断正态**。
- **VLA**（动作）：短且近确定（如 7 自由度离散动作），建议 **固定值或窄正态**。

输出长度直接决定 decode 步数、KV cache 增长、端到端时延与显存压力，是实验中区分策略的关键变量。分布参数默认来自 `ModelSpec.default_output_dist`，可在 `WorkloadConfig` 中覆盖。

## 3. 长度分布抽象（LengthDistribution）

```
kind     fixed | normal | lognormal
mean     期望（token 数）
std      标准差
minimum  下界裁剪
maximum  上界裁剪
```

采样规则：

- `fixed`：恒返回 `round(mean)`，再裁剪到 [min, max]。
- `normal`：从 N(mean, std) 采样，裁剪。
- `lognormal`：以 mean/std 反解底层正态参数 (mu, sigma)，采样后裁剪，得到右偏长尾。

同一抽象用于 **prompt 长度** 与 **output 长度**，两者各持一份参数。

## 4. 请求与会话数据结构

### 4.1 Request

```
request_id        全局唯一
session_id        所属会话
model_name        模型名（决定类别与结构）
model_type        LLM / VLM / VLA
arrival_ms        到达时间（绝对毫秒）
entry_node        请求入口节点 id（受移动影响）
priority          high | normal
sla_ms            P99 TTFT SLA

# token 构成
prompt_text_tokens  文本 token 数
visual_tokens       视觉 token 数（VLM/VLA）
state_tokens        状态 token 数（VLA）
input_tokens        合计输入 token（= 上面之和，由大模型规格计算）
output_len          预设生成长度（采样自分布）

# 复用与状态
prefix_id           可复用前缀标识（同 session/同模板共享）
prefix_tokens       可复用前缀 token 数（用于 prefix hit / recompute）
is_session_first    是否会话首请求
turn_index          会话内第几轮
```

### 4.2 Session

```
session_id        会话 id
model_name        会话绑定的模型
num_turns         会话请求数（多轮）
home_node         初始就近入口节点
mobility_switched 是否在后半段被切换入口
prefix_id         会话共享前缀（system prompt / 模板）
prefix_tokens     共享前缀长度
```

会话内每一轮请求的 prompt 在前缀基础上增长（历史对话累加），使 `prefix_tokens` 随 `turn_index` 单调增加，从而真实反映 KV 复用收益随会话变长而增大。

## 5. 工作负载配置（WorkloadConfig）

按实验报告第 2、9 节固定的混合负载组织。每个“负载分组（WorkloadGroup）”描述一类请求：

```
WorkloadGroup:
  model_name        如 CodeLlama34B
  priority          high | normal
  concurrency       目标并发数（如高优 24 / 普通 96 / VLM 24）
  sla_ms            覆盖默认 SLA（如高优 150ms）
  arrival_rate      到达率（Poisson，请求/秒），或由并发数推导
  prompt_dist       prompt 长度分布
  output_dist       输出长度分布（缺省取模型默认）
  turns_dist        每会话轮数分布
  image_size        VLM 图像分辨率 (w, h)
  num_frames        VLA/VLM 帧数
  shared_prefix_tokens  组内共享 system prompt 长度
```

顶层配置：

```
WorkloadConfig:
  groups: List[WorkloadGroup]
  num_nodes: 3
  duration_ms          实验总时长
  mobility_start_frac  入口切换起始时刻（0.5）
  mobility_ratio       切换比例（0.2）
  seed                 随机种子
```

### 5.1 实验默认负载（对齐报告）

| 分组 | 模型 | 优先级 | 并发 | P99 TTFT SLA |
| --- | --- | --- | --- | --- |
| G1 | CodeLlama34B | high | 24 | 150 ms |
| G2 | CodeLlama34B | normal | 96 | 500 ms |
| G3 | Qwen2-VL-7B-Instruct | normal | 24 | 500 ms |

VLA 作为可选第四分组提供（短输出、固定视觉 token），用于扩展实验。

## 6. 生成流程

```
1. 按 seed 初始化 RNG。
2. 对每个 WorkloadGroup：
   a. 根据 concurrency / arrival_rate 计算需要的 session 数。
   b. 为每个 session 采样 num_turns、home_node、shared prefix。
   c. 为每一轮请求：
      - 采样 prompt_text_tokens（prompt_dist）。
      - 若 VLM/VLA：按 image_size/num_frames 由 ModelSpec.visual_tokens 计算视觉 token。
      - 由 ModelSpec.input_tokens 合成 input_tokens。
      - 采样 output_len（output_dist 或模型默认）。
      - 设定 arrival_ms（Poisson 过程在会话起点 + 轮间隔）。
      - 设定 prefix_id/prefix_tokens（会话共享前缀 + 历史累加）。
3. 应用移动性：duration*mobility_start_frac 之后到达的请求，
   按 mobility_ratio 随机改写 entry_node 为非 home_node。
4. 按 arrival_ms 全局排序，分配 request_id。
5. 输出 Request 列表，可序列化为 JSONL trace。
```

## 7. 移动性建模

- 前半段（`t < duration * mobility_start_frac`）：`entry_node = home_node`（就近）。
- 后半段：以 `mobility_ratio` 概率把请求入口改为另外两个节点之一（均匀选择）。
- 切换发生在 **request 粒度**（默认）或 **session 粒度**（可配置），以分别模拟“偶发漫游”和“持续迁移”。
- 被切换的请求其 `prefix_id` 不变，使 KV owner（历史节点）与入口节点分离——这正是触发 local/migrate/recompute 决策的根因。

## 8. 接口设计

```python
@dataclass
class LengthDistribution:
    kind: str; mean: float; std: float; minimum: int; maximum: int
    def sample(self, rng) -> int
    @classmethod
    def from_spec(cls, spec) -> "LengthDistribution"

@dataclass
class Request: ...      # 见第 4.1 节
@dataclass
class Session: ...      # 见第 4.2 节

@dataclass
class WorkloadGroup: ...
@dataclass
class WorkloadConfig:
    groups: List[WorkloadGroup]
    num_nodes: int = 3
    duration_ms: float = 60000.0
    mobility_start_frac: float = 0.5
    mobility_ratio: float = 0.2
    seed: int = 0
    @staticmethod
    def default_experiment() -> "WorkloadConfig"

class DataGenerator:
    def __init__(self, config: WorkloadConfig)
    def generate(self) -> List[Request]
    def summary(self, requests) -> dict
    def to_jsonl(self, requests, path) -> None
```

## 9. 可复现性与统计

- 全程使用单一 `random.Random(seed)`，四种策略复用同一 trace 保证公平对比。
- `summary()` 输出：请求总数、各模型/优先级数量、平均/分位 prompt 与 output 长度、移动请求比例、平均会话轮数、预计总 KV token 量，供 sanity check。
- `to_jsonl()` 落盘，便于离线分析与多次实验复跑。

## 10. 与其他模块的关系

- 依赖 `large_model.ModelSpec` 计算视觉 token 与输入合成、获取默认输出分布与 SLA。
- 输出的 `Request` 被离散事件模拟器消费：`arrival_ms` 驱动 `request_arrival` 事件，`input_tokens`/`output_len` 交给计算模拟器估算时间，`entry_node`/`prefix_id` 交给调度器做路由与 KV 决策。
