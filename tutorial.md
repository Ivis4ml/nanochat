# Nanochat Tutorial

> **最小化、全栈式 ChatGPT 克隆实现 —— 用 $100 打造你自己的 LLM**

---

## 📚 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [架构设计](#架构设计)
4. [训练流程](#训练流程)
5. [核心组件](#核心组件)
6. [Karpathy 的设计哲学](#karpathy-的设计哲学)
7. [常用命令](#常用命令)
8. [调优与定制](#调优与定制)

---

## 概述

### 什么是 Nanochat？

Nanochat 是一个**完整的端到端 LLM 实现**，涵盖从分词器训练到 Web 界面的所有环节：

- **代码量**: ~8,000 行（核心库仅 3,004 行）
- **训练成本**: $100 (4小时) 到 $800 (33小时)
- **训练硬件**: 8×H100 GPU
- **模型规模**: 561M - 1.9B 参数
- **性能**: 可超越 GPT-2 (2019)

### 核心特性

```
✅ 完整训练流程    从零到部署的全流程
✅ 清晰可读代码    无复杂抽象，易于理解
✅ 现代架构改进    RoPE, RMSNorm, MQA, Muon优化器
✅ 工具使用能力    Python执行、计算器
✅ 分布式训练      多GPU支持，梯度累积
✅ Web界面        ChatGPT风格的交互界面
```

### 项目定位

- **不是**: 复杂的可配置框架
- **而是**: 一个清晰、可黑客的强基线（strong baseline）
- **目标**: 教育、研究、快速实验

---

## 快速开始

### 最简单方式：Speedrun ($100 / 4小时)

```bash
# 1. 克隆仓库
git clone https://github.com/karpathy/nanochat
cd nanochat

# 2. 启动训练（推荐在 screen 会话中）
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# 3. 等待 4 小时...

# 4. 训练完成后，启动 Web 界面
source .venv/bin/activate
python -m scripts.chat_web

# 5. 访问 http://<your-ip>:8000
```

### 查看训练报告

训练完成后会生成 `report.md`：

```bash
cat report.md
```

典型输出：

```markdown
| Metric          | BASE   | MID    | SFT    | RL     |
|-----------------|--------|--------|--------|--------|
| CORE            | 0.2219 | -      | -      | -      |
| GSM8K           | -      | 0.0250 | 0.0455 | 0.0758 |
| HumanEval       | -      | 0.0671 | 0.0854 | -      |
| MMLU            | -      | 0.3111 | 0.3151 | -      |

Total wall clock time: 3h51m
```

---

## 架构设计

### 文件结构总览

```
nanochat/
├── nanochat/              # 核心库 (3004行)
│   ├── gpt.py            # GPT Transformer模型 (307行)
│   ├── engine.py         # KV Cache推理引擎 (371行)
│   ├── tokenizer.py      # BPE分词器 (398行)
│   ├── muon.py           # Muon优化器 (187行)
│   ├── adamw.py          # 分布式AdamW (76行)
│   └── ...               # 其他工具模块
│
├── scripts/              # 训练/推理脚本 (11个)
│   ├── base_train.py     # 预训练
│   ├── mid_train.py      # 中间训练
│   ├── chat_sft.py       # 监督微调
│   ├── chat_rl.py        # 强化学习
│   └── chat_web.py       # Web服务
│
├── tasks/                # 评估任务 (8个)
│   ├── gsm8k.py          # 数学问题
│   ├── humaneval.py      # 代码生成
│   ├── mmlu.py           # 多学科问答
│   └── ...
│
├── rustbpe/              # Rust实现的快速BPE
├── speedrun.sh           # $100 训练脚本
└── run1000.sh            # $800 训练脚本
```

### 模型架构 (nanochat/gpt.py)

#### GPTConfig

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024      # 上下文窗口
    vocab_size: int = 50304       # 词汇表大小
    n_layer: int = 12             # Transformer层数
    n_head: int = 6               # Query头数量
    n_kv_head: int = 6            # KV头数量 (MQA支持)
    n_embd: int = 768             # 嵌入维度
```

#### 核心创新

1. **Rotary Embeddings (RoPE)**
   - 无需学习的位置编码
   - 相对位置信息，支持更长序列

2. **RMSNorm**
   ```python
   def norm(x):
       # 纯函数式，无可学习参数
       return F.rms_norm(x, (x.size(-1),))
   ```

3. **QK Normalization**
   - 训练稳定性提升
   - 防止注意力分数爆炸

4. **ReLU² Activation**
   ```python
   # MLP中使用 relu().square()
   F.relu(x).square()
   ```

5. **Multi-Query Attention (MQA)**
   - 减少KV Cache大小
   - 加速推理

6. **Logits Softcapping**
   - 防止损失峰值
   - 梯度稳定

7. **Untied Weights**
   - Token Embedding ≠ LM Head
   - 更灵活的参数学习

### 推理引擎 (nanochat/engine.py)

#### KV Cache 机制

```python
class KVCache:
    """高效自回归生成的关键"""
    - 缓存每层的Key/Value张量
    - 支持动态扩展
    - 批量预填充
    - 位置跟踪
```

**为什么需要 KV Cache？**

- **无缓存**: O(T²) 复杂度，每次重新计算所有token
- **有缓存**: O(T) 复杂度，只计算新token

#### 工具集成

```python
# 支持两种工具：
1. Calculator Tool   # 安全的数学计算
2. Python REPL       # 沙盒化代码执行
   - 内存限制: 256MB
   - 超时限制: 10秒
   - 危险函数黑名单
```

### 优化器设计

#### 双优化器策略

```python
# 1. DistAdamW - 用于嵌入层
embedding_params → AdamW (lr=0.2)

# 2. Muon - 用于2D矩阵参数
attention_weights → Muon (lr=0.02)
mlp_weights → Muon (lr=0.02)

# 3. 输出层特殊学习率
lm_head → AdamW (lr=0.004, 低50倍)
```

**Muon 优化器特点**：
- 基于动量的正交化优化
- Newton-Schulz迭代
- BFloat16稳定计算
- 适合大型矩阵参数

---

## 训练流程

### 完整Pipeline

```
1. 分词器训练 (Tokenizer)
   ↓
2. 基础预训练 (Base Pretraining)
   ↓
3. 中间训练 (Midtraining)
   ↓
4. 监督微调 (Supervised Fine-Tuning)
   ↓
5. 强化学习 (Reinforcement Learning) [可选]
   ↓
6. 推理与评估
```

### 1. 分词器训练

```bash
# 训练BPE分词器
python -m scripts.tok_train --max_chars=2000000000

# 参数：
# - 词汇表大小: 2^16 = 65,536
# - 训练数据: ~2B字符
# - 使用Rust实现加速
```

**特殊Token**:

```python
<|bos|>              # 序列开始
<|user_start|>       # 用户消息开始
<|user_end|>         # 用户消息结束
<|assistant_start|>  # 助手回复开始
<|assistant_end|>    # 助手回复结束
<|python_start|>     # Python代码开始
<|python_end|>       # Python代码结束
<|output_start|>     # 工具输出开始
<|output_end|>       # 工具输出结束
```

### 2. 基础预训练

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- --depth=20 --run=myrun
```

**关键参数**:

```python
depth = 20                    # 模型深度 (d20 = 561M参数)
max_seq_len = 2048            # 上下文长度
target_param_data_ratio = 20  # Chinchilla缩放定律
device_batch_size = 32        # 每GPU批量大小
total_batch_size = 524_288    # 总批量大小

# 学习率设置
embedding_lr = 0.2
matrix_lr = 0.02
unembedding_lr = 0.004
```

**数据集**: FineWeb-Edu-100B
- 1822个Parquet分片
- 每个分片 ~250M字符
- 自动下载，带重试机制

**Chinchilla 计算**:

```python
# d20 模型
params = 561M
tokens_needed = params × 20 = 11.2B tokens
chars_needed = tokens × 4.8 = 54B chars
shards_needed = chars / 250M = 216 shards
# 实际下载 240 shards (留余量)
```

### 3. 中间训练 (Midtraining)

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.mid_train -- --run=myrun
```

**目的**: 教模型理解特殊token和工具使用

**训练任务混合**:
- GSM8K (数学问题 + 计算器工具)
- MMLU (多学科问答)
- SmolTalk (对话数据)
- 身份对话 (个性化)

### 4. 监督微调 (SFT)

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft -- --run=myrun
```

**特点**:
- 单轮训练（1 epoch）
- 任务混合训练
- 对话行为学习
- 格式规范化

### 5. 强化学习 (可选)

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_rl -- --run=myrun
```

**GRPO 风格训练**:
- 基于采样的rollout生成
- 奖励计算（任务成功与否）
- 优势归一化
- 在线策略学习（无需参考模型）

---

## 核心组件

### 1. 分词器 (tokenizer.py)

**双实现策略**:

```python
# 训练 & 慢速推理
HuggingFace Tokenizers → Python实现

# 快速推理
RustBPE + Tiktoken → Rust实现 (10-100x加速)
```

### 2. 数据加载器 (dataloader.py)

```python
class TokenizingDistributedDataLoader:
    """流式数据加载"""
    - 从Parquet文件流式加载
    - 即时分词
    - 分布式批处理 (DDP感知)
    - 可配置批量大小
```

### 3. 评估系统

#### CORE 评估 (core_eval.py)

```python
# DCLM论文的评估指标
支持任务类型：
- Multiple-choice (MMLU, ARC)
- Schema-based (HellaSwag)
- Language modeling (LAMBADA, SQuAD)

评分方式：
- 基于概率的选择
- Few-shot模板
```

#### Bits Per Byte (loss_eval.py)

```python
# 与分词器无关的评估指标
def bits_per_byte(loss, token_bytes):
    """
    loss: 模型的交叉熵损失
    token_bytes: 每个token对应的字节数
    返回: 每字节的比特数
    """
    return loss / math.log(2) / token_bytes.mean()
```

### 4. 任务系统 (tasks/)

#### 基础Task抽象

```python
class Task:
    eval_type: str  # 'generative' 或 'categorical'

    def num_examples(self) -> int:
        """数据集大小"""

    def get_example(self, idx: int) -> dict:
        """返回对话字典"""

    def evaluate(self, problem, completion) -> bool:
        """判断答案正确性"""
```

#### 任务组合

```python
# 混合多个任务（随机打乱）
TaskMixture([GSM8K(), MMLU(), SmolTalk()])

# 顺序训练
TaskSequence([Task1(), Task2(), Task3()])
```

### 5. 检查点管理 (checkpoint_manager.py)

```python
# 每个检查点包含3个文件:
model.pth       # 模型权重
optimizer.pth   # 优化器状态
metadata.json   # 超参数和元信息

# 支持功能：
- 处理 torch.compile 包装器
- 设备转换 (bfloat16 ↔ float32)
- Meta设备初始化（节省内存）
- 可恢复训练
```

### 6. 代码执行 (execution.py)

```python
# 沙盒化Python执行
class CodeExecutor:
    timeout: int = 10          # 超时限制 (秒)
    max_memory: int = 256      # 内存限制 (MB)

    # 危险函数黑名单
    blacklist = [
        'eval', 'exec', '__import__',
        'open', 'input', 'compile', ...
    ]
```

**安全措施**:
- 进程隔离（fork-based）
- 资源限制
- 函数黑名单
- 超时保护

---

## Karpathy 的设计哲学

### 核心原则

#### 1. **清晰胜于配置**

```python
# ❌ 不要这样：
config = ComplexConfigObject(
    model=ModelConfig(...),
    training=TrainingConfig(...),
    optimizer=OptimizerConfig(...)
)

# ✅ 而是这样：
# 全局变量 + 类型检查
depth = 20
learning_rate = 0.02
max_seq_len = 2048

# CLI覆盖：
# python script.py --depth=24 --learning_rate=0.01
```

#### 2. **极简主义**

> "No config monsters, no dependency hell. You can literally read every line and know what's going on."

- 核心库仅 3,004 行
- 无复杂工厂模式
- 扁平依赖树
- 每一行都可理解

#### 3. **手写代码**

> "The code was basically entirely hand-written (with tab autocomplete)"

Karpathy 发现：
- Claude/Codex 代理对此项目"不太有用"
- 项目可能"偏离数据分布"
- 人工编写能确保代码质量

#### 4. **强基线，非框架**

```
nanochat 是：
✅ 教育工具
✅ 研究起点
✅ 可复制的基线
✅ 最大化可fork性

nanochat 不是：
❌ 生产框架
❌ 配置怪兽
❌ 通用工具库
```

### 实用建议

#### ⚠️ 不要在个人数据上微调

> "Fine-tuning on personal data can end up generating 'slop'"

**更好的方法**:
- 使用 RAG (检索增强生成)
- 工具如 NotebookLM
- 保持模型通用性

#### 💰 成本-性能权衡

```
$100  (4h)   → 幼儿园水平 (有趣但幼稚)
$300  (12h)  → 略超GPT-2
$800  (33h)  → 可解决简单数学/代码
$1000+ (42h) → 更连贯，实用性增强
```

#### 🔬 仍有优化空间

> "It's by no means finished, tuned or optimized (actually I think there's likely quite a bit of low-hanging fruit)"

**潜在改进**:
- 超参数调优
- 数据混合优化
- 架构微调
- 训练配方改进

#### 📊 可扩展为研究工具

> "It also has potential to grow into a research harness, or a benchmark, similar to nanoGPT before it."

---

## 常用命令

### 训练相关

```bash
# 1. Speedrun ($100, 4小时)
bash speedrun.sh

# 2. 更大模型 ($800, 33小时)
bash run1000.sh

# 3. 自定义训练
# 预训练
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=24 \
    --device_batch_size=16 \
    --run=my_experiment

# 中间训练
torchrun --standalone --nproc_per_node=8 \
    -m scripts.mid_train -- \
    --device_batch_size=16

# 监督微调
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft

# 强化学习
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_rl
```

### 评估相关

```bash
# CORE评估 (预训练)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_eval

# 损失评估
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_loss

# 聊天任务评估
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_eval -- -i sft

# 仅评估特定任务
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_eval -- -i sft -a GSM8K
```

### 推理相关

```bash
# CLI 聊天 (交互式)
python -m scripts.chat_cli

# CLI 聊天 (单次查询)
python -m scripts.chat_cli -p "Why is the sky blue?"

# Web 界面
python -m scripts.chat_web
# 然后访问 http://<your-ip>:8000
```

### 工具相关

```bash
# 训练分词器
python -m scripts.tok_train --max_chars=2000000000

# 评估分词器
python -m scripts.tok_eval

# 下载数据
python -m nanochat.dataset -n 240  # 下载240个分片

# 生成报告
python -m nanochat.report generate
```

---

## 调优与定制

### 内存不足 (OOM) 解决

```bash
# 减少 device_batch_size（代码会自动补偿）
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=16  # 从32降到16

# 如果还是OOM，继续降低
--device_batch_size=8
--device_batch_size=4
--device_batch_size=2
```

**机制**: 代码自动增加梯度累积步数，将并行计算转为顺序计算。

### 单GPU训练

```bash
# 省略 torchrun，直接运行
python -m scripts.base_train -- --depth=20

# 注意：
# - 结果与多GPU基本相同
# - 但速度慢8倍
```

### 定制模型大小

```bash
# Chinchilla 计算
# 参数量 = depth × config

# d20: 561M 参数
# d26: 1.1B 参数
# d32: 1.9B 参数

# 调整depth
--depth=26

# 相应调整数据量
# tokens = params × 20
# chars = tokens × 4.8
# shards = chars / 250M

# d26 需要 ~450 shards
python -m nanochat.dataset -n 450
```

### 自定义个性化

参考官方指南：
- [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139)

**步骤**:
1. 生成合成身份对话数据
2. 混入midtraining数据
3. 混入SFT数据

### 添加新能力

参考官方指南：
- [Guide: counting r in strawberry](https://github.com/karpathy/nanochat/discussions/164)

**步骤**:
1. 创建新Task类
2. 准备训练数据
3. 混入训练流程

### 使用 WandB 日志

```bash
# 1. 登录 wandb
wandb login

# 2. 设置环境变量运行
WANDB_RUN=my_experiment bash speedrun.sh

# 3. 在 wandb 网站查看训练曲线
```

### CPU/MPS 运行

```bash
# 参考 dev/runcpu.sh
# 主要调整：
# - 更小的模型
# - 更少的迭代
# - 更小的批量大小

# 示例
python -m scripts.base_train -- \
    --depth=6 \
    --device_batch_size=4 \
    --max_iters=100
```

---

## 快速索引

### 关键文件速查

| 功能 | 文件 | 行数 |
|------|------|------|
| 模型定义 | `nanochat/gpt.py` | 307 |
| 推理引擎 | `nanochat/engine.py` | 371 |
| 分词器 | `nanochat/tokenizer.py` | 398 |
| Muon优化器 | `nanochat/muon.py` | 187 |
| AdamW优化器 | `nanochat/adamw.py` | 76 |
| 预训练脚本 | `scripts/base_train.py` | ~17KB |
| Web界面 | `scripts/chat_web.py` | - |

### 超参数速查

| 参数 | d20 (561M) | d26 (1.1B) | d32 (1.9B) |
|------|------------|------------|------------|
| Depth | 20 | 26 | 32 |
| Tokens | 11B | 22B | 38B |
| Data Shards | 240 | 450 | 800 |
| Device Batch | 32 | 16 | 16 |
| Training Time | 4h | 12h | 33h |
| Cost | $100 | $300 | $800 |

### 评估指标速查

| 指标 | 说明 | 用途 |
|------|------|------|
| CORE | DCLM基准 | 预训练质量 |
| BPB | Bits per byte | 分词器无关评估 |
| GSM8K | 数学问题 | 推理能力 |
| HumanEval | 代码生成 | 编程能力 |
| MMLU | 多学科问答 | 知识广度 |
| ARC | 科学问题 | 常识推理 |

---

## 总结

### Nanochat 的价值

1. **教育价值**: 完整展示现代LLM训练流程
2. **研究价值**: 清晰的基线，易于实验
3. **实用价值**: 可在合理预算内训练自己的模型
4. **代码质量**: 每一行都可读、可理解

### 适合人群

- ✅ LLM研究者
- ✅ 机器学习学生
- ✅ 想理解LLM内部原理的开发者
- ✅ 需要定制化小模型的项目

### 进一步学习

- **原仓库**: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)
- **讨论区**: GitHub Discussions
  - [介绍帖](https://github.com/karpathy/nanochat/discussions/1)
  - [个性化指南](https://github.com/karpathy/nanochat/discussions/139)
  - [能力扩展指南](https://github.com/karpathy/nanochat/discussions/164)
- **前作**: [nanoGPT](https://github.com/karpathy/nanoGPT) (仅预训练)
- **在线实例**: [nanochat.karpathy.ai](https://nanochat.karpathy.ai/)

### 引用

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

---

**🚀 现在就开始你的 LLM 训练之旅吧！**
