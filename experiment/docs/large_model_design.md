# 大模型（Large Model）设计文档

## 1. 设计目标

大模型模块为整个卸载模拟实验提供统一的“模型规格”抽象。它是硬件、数据、网络、KV 四个模块共享的**单一事实来源**，同时满足：

- 为**计算模拟器**提供估算 prefill/decode 时间和显存所需的全部结构参数（层数、隐藏维度、注意力头、GQA 分组、FFN 维度、KV cache 每 token 字节数等）。
- 为**数据生成器**提供每类模型的输入构成方式（纯文本、图文、视觉-语言-动作）以及默认的输入/输出长度分布。
- 为 **KV 缓存模块**提供 KV block 粒度（`kv_block_size`）与单 token KV 字节数。
- 为**调度器/约束过滤**提供默认 SLA、优先级等业务侧元数据。

实验固定三类大模型：LLM（CodeLlama34B）、VLM（Qwen2-VL-7B-Instruct）、VLA（OpenVLA-7B 架构）。模块以注册表方式管理，便于扩展。

## 2. 模型类别

用 `ModelType` 区分三类大模型：

| 类别 | 含义 | 输入模态 | 输出特征 |
| --- | --- | --- | --- |
| `LLM` | 纯文本解码模型 | text prompt | 输出较长、长度方差大 |
| `VLM` | 视觉-语言模型 | text + image | 输出中等、依赖视觉 token |
| `VLA` | 视觉-语言-动作模型 | image + state + instruction | 输出短且接近确定（动作 chunk） |

三类大模型的关键差异：

1. **输入 token 构成**。LLM 仅文本 token；VLM 额外引入视觉 token（数量随图像分辨率动态变化）；VLA 引入固定数量的视觉 token、机器人状态 token 与指令 token。
2. **输出长度分布**。LLM 长尾、方差大；VLM 中等；VLA 短且近确定（如 7 自由度离散动作）。
3. **KV cache 体积**。由层数、KV 头数、head_dim 和精度共同决定，差异显著，直接影响 migrate/recompute 成本。

## 3. 模型结构规格

每个模型用 `ModelSpec` 描述，核心字段：

```
name                模型名
model_type          LLM / VLM / VLA
num_params          总参数量（用于 prefill 计算量估算）
num_layers          解码层数
hidden_size         隐藏维度 d_model
num_attention_heads 注意力头数
num_kv_heads        KV 头数（GQA；等于 heads 时为 MHA）
head_dim            单头维度
intermediate_size   FFN 中间维度
vocab_size          词表大小
dtype_bytes         权重/KV 精度字节数（bf16 = 2）
weight_bytes        权重总字节数（可显式覆盖）
```

视觉字段（VLM/VLA）：`vision_params`、`patch_size`、`spatial_merge`、`tokens_per_image`。
业务字段：`default_output_dist`、`default_sla_ms`、`kv_block_size`。

### 3.1 三个预置模型参数

**CodeLlama34B（LLM，Llama2-34B 架构）**：layers=48，hidden=8192，heads=64，kv_heads=8（GQA），head_dim=128，intermediate=22016，vocab=32016，params≈33.7e9，bf16。

**Qwen2-VL-7B-Instruct（VLM）**：layers=28，hidden=3584，heads=28，kv_heads=4，head_dim=128，intermediate=18944，vocab=152064，params≈7.6e9；视觉 ViT，patch=14，spatial_merge=2（每合并 token 覆盖 28×28 像素）。

**OpenVLA-7B（VLA，Llama2-7B 骨干 + DINOv2/SigLIP 视觉）**：layers=32，hidden=4096，heads=32，kv_heads=32（MHA），head_dim=128，intermediate=11008，vocab=32064，params≈7.5e9；固定 tokens_per_image=256；动作输出 7 自由度离散 token。

> VLA 采用 MHA 且层数多，单 token KV 体积显著高于 Qwen2-VL，使 migrate/recompute 决策在三类模型间表现不同偏好，符合实验待验证问题。

## 4. 关键派生量

### 4.1 单 token KV cache 字节数

```
kv_bytes_per_token = 2 * num_kv_heads * head_dim * dtype_bytes * num_layers
```

系数 2 表示 K 和 V。用于迁移/重算成本、显存占用、KV 容量约束、block 大小换算。

### 4.2 prefill 计算量（FLOPs）

```
prefill_flops = 2 * num_params * num_tokens
              + 4 * num_layers * num_tokens^2 * hidden_size   # 注意力 QK^T + AV
```

### 4.3 decode 单 token 计算量与访存量

```
decode_flops_per_token = 2 * num_params + 4 * num_layers * ctx_len * hidden_size
decode_bytes_per_token = weight_bytes + kv_bytes_per_token * ctx_len
```

### 4.4 输入 token 构成

- LLM：`input_tokens = prompt_text_tokens`
- VLM：`input_tokens = prompt_text_tokens + visual_tokens`，
  `visual_tokens = ceil(W/(patch*merge)) * ceil(H/(patch*merge))`
- VLA：`input_tokens = tokens_per_image*num_frames + instruction_tokens + state_tokens`

### 4.5 KV block 换算（供 KV 模块复用）

```
kv_blocks_for_tokens = ceil(num_tokens / kv_block_size)
kv_bytes_for_tokens  = kv_blocks_for_tokens * kv_block_size * kv_bytes_per_token
```

## 5. 输出长度分布

每个模型携带默认输出长度分布（详见数据生成器设计文档的 `LengthDistribution`）：LLM 建议 lognormal；VLM 建议 lognormal/截断正态；VLA 建议固定或窄正态（如 7±1）。模型只声明默认值，实际采样在数据生成器完成。

## 6. 接口设计

```python
class ModelType(Enum): LLM, VLM, VLA

@dataclass(frozen=True)
class ModelSpec:
    # 结构 + 视觉 + 业务字段（见第 3 节）
    def kv_bytes_per_token(self) -> int
    def prefill_flops(self, num_tokens) -> float
    def decode_flops_per_token(self, ctx_len) -> float
    def decode_bytes_per_token(self, ctx_len) -> int
    def visual_tokens(self, width, height, num_frames=1) -> int
    def input_tokens(self, text_tokens, **modality) -> int
    def kv_blocks_for_tokens(self, num_tokens) -> int
    def kv_bytes_for_tokens(self, num_tokens) -> int

MODEL_REGISTRY: Dict[str, ModelSpec]   # 预置三模型
def get_model(name) -> ModelSpec
def list_models() -> List[str]
```

## 7. 与其他模块的关系

- **计算模拟器**：调用 `prefill_flops` / `decode_*` / `kv_bytes_per_token` / `total_weight_bytes`。
- **数据生成器**：调用 `input_tokens` / `visual_tokens` 与 `default_output_dist`。
- **KV 缓存**：调用 `kv_block_size` / `kv_bytes_per_token` / `kv_bytes_for_tokens` 组织 block。
- **调度器**：读取 `default_sla_ms` 等用于约束过滤。

大模型规格是四者共享的“单一事实来源”，避免多处对同一模型给出不一致参数。
