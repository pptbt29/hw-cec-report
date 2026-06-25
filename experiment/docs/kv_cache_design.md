# KV Cache 管理与 Block 设计文档

## 1. 设计目标

KV cache 模块决定卸载实验中 local / migrate / recompute 三类动作的成本，是“状态黏附”和“长期成本”能否被刻画的关键。它需要做到：

- 把 KV cache 组织成**固定大小的 block**，支持 **block 级 hash 与最长前缀匹配**，使请求即使只命中部分历史前缀也能复用。
- 在每个节点维护一个**容量受限的 KV store**（带淘汰策略），反映 GPU memory 约束。
- 维护一个跨节点的**全局 prefix/KV 目录**，记录每个 block 在哪些节点有有效副本（owner + replica 列表）。
- 支持 **migrate（只传缺失 block）**、**recompute（重算前缀）**、**local（命中本地）** 三种状态获取方式，并给出各自的字节数/时间。
- 为指标统计提供：prefix/KV 命中率、迁移字节数、重算次数、owner 切换次数、容量峰值。

## 2. 为什么用 block

如果对整个 prefix 只生成一个 hash，任何细微差异都会导致整体 miss，且迁移必须整体搬运。**block 化**带来三个好处：

1. **部分复用**：相同 system prompt / 公共文档前缀对应的前若干 block 可被多请求共享。
2. **增量迁移**：迁移时只传目标节点缺失的 block；A↔B 100G 可激进复制，A/B↔C 25G 只传高复用 block。
3. **增量同步**：会话增长时新生成的 block 增量同步，旧 block 可保留短 TTL 做回退。

block 粒度由大模型规格的 `kv_block_size`（默认 16 token）给出，单 block 字节数 = `kv_block_size * kv_bytes_per_token`。

## 3. Block 与 hash 设计

### 3.1 序列 → block 切分

一个 session 的 token 序列按 `kv_block_size` 切成有序 block：block `i` 覆盖 token `[i*B, (i+1)*B)`。

### 3.2 block hash（前缀链式 hash）

为支持“最长前缀匹配”，block hash 必须包含其**全部祖先前缀**，即链式 hash：

```
block_hash[0] = H(model_id, model_version, adapter_id, tokens[0:B])
block_hash[i] = H(block_hash[i-1], tokens[i*B:(i+1)*B])
```

这样两个请求只要共享相同的前缀 token 与相同模型，前若干 block 的 hash 必然相同，可命中同一批 block；一旦某个 block 内容不同，其后所有 block hash 都不同（前缀敏感），符合自回归注意力对前缀的依赖。

### 3.3 block 元数据（KVBlock）

```
block_hash          链式 hash（主键）
session_id          所属会话（私有状态）/ 或 shared 表示公共前缀
prefix_id           可复用前缀标识
model_name          模型名（区分不同模型的 KV）
model_version       模型/adapter 版本
block_index         在序列中的序号
num_tokens          实际 token 数（末块可能不满）
size_bytes          字节数
owner               当前主副本节点
replicas            有效副本节点集合
version             状态版本（增量同步用）
last_access_ms      最近访问时间（LRU）
hit_count           命中次数
```

## 4. 每节点 KV Store

`KVCacheStore`（绑定 node id 与 ModelSpec）：

- 容量 `capacity_bytes`（由计算模拟器的显存预算折算给 KV 的部分）。
- `blocks: Dict[block_hash, KVBlock]` 本地驻留 block。
- 写入 `insert(blocks)`：占用容量，超限触发淘汰。
- 查询 `match_prefix(hashes)`：返回本地命中的最长前缀 block 数。
- 淘汰 `evict()`：默认 **LRU**，跳过仍被活跃 session 引用的 block；优先淘汰已有他处副本的 block（淘汰代价低）。
- 统计：used_bytes、free_bytes、evictions、hit/miss。

## 5. 全局 KV 目录

`GlobalKVDirectory` 聚合所有节点的 block 位置（对应架构中的共享元数据目录的 KV 部分）：

- `locate(block_hash) -> set(nodes)`：该 block 有有效副本的节点集合。
- `longest_prefix(hashes, node) -> (local_hit_blocks, remote_hit_blocks)`：给定一串前缀 block hash，返回本地命中数与可从远端获取的命中数。
- `register(node, block)` / `unregister(node, block_hash)`：副本增删。
- `set_owner(block_hash, node)`：owner 切换（迁移完成后），记录 `owner_switch_count`。
- 注意：目录是**周期性同步**的，路由器读取时可能略旧（staleness），与架构文档一致。

## 6. 三类状态获取动作的成本

给定请求的可复用前缀对应 block hash 列表 `P`，目标执行节点 `dst`，历史 owner `o`：

### 6.1 local（dst 已有完整前缀）

```
missing = P - blocks_on(dst)
若 missing 为空 → T_state ≈ 0（仅查表 + 读本地）
```

### 6.2 migrate（从已有副本节点传缺失 block 到 dst）

```
missing_blocks = P 中 dst 缺失、但在某源节点 src 存在的 block
bytes_to_move  = sum(size_bytes of missing_blocks)
src*           = argmin_src network.transfer_time_ms(src, dst, bytes_from_src)  # 选最优源
T_migrate      = network.transfer_time_ms(src*, dst, bytes_to_move)
迁移完成后：dst 加入 replicas；按策略 set_owner(dst) 并对旧副本设 TTL
```

> 多副本时从“传输成本最低的源”取数；100G 域内优先。这正是“低成本 KV 管理”相对纯长期路由的增量价值。

### 6.3 recompute（在 dst 重算前缀 KV）

```
reusable_prefix_tokens = len(P) 命中的 token 数
T_recompute = compute_simulator.recompute_time_ms(reusable_prefix_tokens)
              # 等价于对前缀再做一次 prefill
```

路由器在可行动作集合内比较 `T_migrate(各源)` 与 `T_recompute`：前缀大、链路慢时 recompute 往往更划算；前缀小、链路快时 migrate 更划算。

## 7. 与显存约束的关系

- KV store 容量来自计算模拟器 `memory_usage` 中分配给 KV 的部分：`kv_capacity = total_memory - weights - activation_reserve`。
- 动作执行后新增 KV（prefill 写入 + decode 增长）必须满足 `kv_used + new_kv <= kv_capacity`，否则该动作在约束过滤阶段被剔除，或触发淘汰。
- 这把“显存不足导致不可执行请求”这一指标落到 KV store 上。

## 8. 接口设计

```python
@dataclass
class KVBlock:
    block_hash: str; session_id: str; prefix_id: str
    model_name: str; model_version: str
    block_index: int; num_tokens: int; size_bytes: int
    owner: int; replicas: Set[int]; version: int
    last_access_ms: float; hit_count: int

def block_hashes(model_name, model_version, token_ids, block_size) -> List[str]
def block_hashes_for_len(model_name, model_version, prefix_id, num_tokens, block_size) -> List[str]

class KVCacheStore:               # 每节点一个
    def __init__(self, node_id, model: ModelSpec, capacity_bytes)
    def insert(self, blocks: List[KVBlock], t_now) -> int          # 返回淘汰数
    def contains(self, block_hash) -> bool
    def match_prefix(self, hashes) -> int                          # 本地最长前缀命中块数
    def used_bytes(self) -> int; def free_bytes(self) -> int
    def evict(self, need_bytes, t_now) -> List[str]
    stats: dict

class GlobalKVDirectory:          # 共享元数据目录的 KV 部分
    def __init__(self, num_nodes)
    def register(self, node, block) ; def unregister(self, node, block_hash)
    def locate(self, block_hash) -> Set[int]
    def longest_prefix(self, hashes, node) -> Tuple[int, int]      # (local_hit, remote_hit)
    def set_owner(self, block_hash, node)
    def plan_migration(self, hashes, dst, net) -> "MigrationPlan"  # 选最优源、算字节与时间
    stats: dict   # hit_ratio, migrate_bytes, recompute_count, owner_switch_count

@dataclass
class MigrationPlan:
    dst: int; src: Optional[int]
    missing_hashes: List[str]; bytes_to_move: int; transfer_ms: float
```

## 9. 与其他模块的关系

- 依赖 `large_model.ModelSpec`：`kv_block_size`、`kv_bytes_per_token`、block 字节数。
- 依赖 `network.NetworkSimulator`：`plan_migration` 用其 `transfer_time_ms` 选源并定价。
- 依赖 `compute_simulator`：recompute 成本 = 前缀 prefill 时间；KV 容量 = 显存预算折算。
- 服务 `Router`：提供 local/migrate/recompute 成本与命中信息，构造可行动作集合与长期成本。
