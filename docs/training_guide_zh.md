# Nanochat 完整训练指南

> **在 8×H100 GPU 上从零开始训练你自己的 ChatGPT 克隆版**

这是一份超级详细的端到端训练指南，涵盖从环境配置到最终模型评估的每一个步骤。

## 📋 目录

- [前置要求](#前置要求)
- [环境配置](#环境配置)
- [理解训练流程](#理解训练流程)
- [阶段 0：初始设置](#阶段-0初始设置)
- [阶段 1：Tokenizer 训练](#阶段-1tokenizer-训练)
- [阶段 2：基础模型预训练](#阶段-2基础模型预训练)
- [阶段 3：中间训练](#阶段-3中间训练)
- [阶段 4：监督微调](#阶段-4监督微调)
- [阶段 5：强化学习（可选）](#阶段-5强化学习可选)
- [阶段 6：推理与评估](#阶段-6推理与评估)
- [监控与调试](#监控与调试)
- [成本与时间估算](#成本与时间估算)
- [故障排除](#故障排除)

---

## 前置要求

### 硬件要求

你的机器配置完美符合要求！

```
8× NVIDIA H100 80GB HBM3
- 总显存: 640 GB
- CUDA 版本: 12.4
- 驱动版本: 550.144.03
```

**可以训练的模型规模：**
- ✅ d20 模型 (561M 参数) - $100 / 4小时
- ✅ d26 模型 (1.1B 参数) - $300 / 12小时
- ✅ d32 模型 (1.9B 参数) - $800 / 33小时

### 软件要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.10+
- **CUDA**: 12.4 (已安装)
- **存储空间**:
  - d20: 50 GB
  - d26: 150 GB
  - d32: 300 GB
- **网络**: 稳定的网络连接用于下载数据

### 成本估算

假设 GPU 租金为 **$3/GPU/小时** (8 个 GPU)：
- **d20**: ~$100 (4小时)
- **d26**: ~$300 (12小时)
- **d32**: ~$800 (33小时)

---

## 环境配置

### 步骤 1: 克隆代码仓库

```bash
# 克隆 nanochat
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 确认在正确的目录
pwd  # 应该显示: /path/to/nanochat
ls   # 应该看到: README.md, nanochat/, scripts/, speedrun.sh 等
```

### 步骤 2: 安装 uv 包管理器

```bash
# 安装 uv (Python 包管理器)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 添加到 PATH (如需要)
source $HOME/.cargo/env

# 验证安装
uv --version
```

### 步骤 3: 创建虚拟环境

```bash
# 创建虚拟环境
uv venv

# 激活
source .venv/bin/activate

# 安装依赖 (GPU 版本)
uv sync --extra gpu

# 验证 PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU 数量: {torch.cuda.device_count()}')"
```

**预期输出:**
```
PyTorch: 2.x.x
CUDA 可用: True
GPU 数量: 8
```

### 步骤 4: 安装 Rust (用于 Tokenizer)

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 添加到 PATH
source "$HOME/.cargo/env"

# 验证安装
rustc --version
cargo --version
```

### 步骤 5: 编译 Rust Tokenizer

```bash
# 编译 rustbpe tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 这会编译 Rust 代码并将其安装为 Python 模块
# 大约需要 1-2 分钟
```

### 步骤 6: 设置目录结构

```bash
# 设置工件存储目录
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 这个目录将存储:
# - 下载的数据分片
# - 训练好的 tokenizer
# - 模型检查点
# - 训练日志

# 添加到 shell 配置文件以持久化
echo 'export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"' >> ~/.bashrc
```

### 步骤 7: (可选) 设置 Weights & Biases

```bash
# 安装 wandb
pip install wandb

# 登录 (从 https://wandb.ai 获取 API key)
wandb login

# 训练时设置 WANDB_RUN 环境变量:
# export WANDB_RUN=我的实验名称
```

---

## 理解训练流程

在开始之前，让我们先理解完整的训练流程：

```
┌─────────────────────────────────────────────────────────────┐
│                     训练流程总览                             │
└─────────────────────────────────────────────────────────────┘

1. TOKENIZER 训练 (~30分钟)
   ├─ 下载: ~2GB 文本数据 (8个分片)
   ├─ 训练: BPE tokenizer，词汇表大小=65536
   └─ 输出: tokenizer.pkl, token_bytes.pt

2. 基础模型预训练 (~2-3小时，d20)
   ├─ 下载: ~24GB 更多数据 (总共240个分片)
   ├─ 训练: 561M 参数的 Transformer，540亿字符
   ├─ 评估: CORE 指标，验证集损失
   └─ 输出: base_model.pth, base_optimizer.pth

3. 中间训练 (~30分钟)
   ├─ 下载: 合成对话数据 (~2MB)
   ├─ 训练: 教授特殊 token、工具使用、对话
   ├─ 评估: 任务混合 (GSM8K, MMLU, 等)
   └─ 输出: mid_model.pth, mid_optimizer.pth

4. 监督微调 (~20分钟)
   ├─ 训练: 在对话任务上训练一轮
   ├─ 评估: 所有聊天任务
   └─ 输出: sft_model.pth, sft_optimizer.pth

5. 强化学习 (可选, ~20分钟)
   ├─ 训练: 在 GSM8K 上使用 GRPO
   ├─ 评估: GSM8K 准确率
   └─ 输出: rl_model.pth, rl_optimizer.pth

6. 推理与 Web 服务
   ├─ 生成: 采样补全
   ├─ 报告: 完整的成绩单 (report.md)
   └─ 服务: Web UI，端口 8000

总时间 (d20): ~4小时
总成本: ~$100
```

---

## 阶段 0: 初始设置

### 创建训练会话

推荐使用 `screen` 或 `tmux`，因为训练需要数小时：

```bash
# 方案 1: 使用 screen
screen -S nanochat-training

# 方案 2: 使用 tmux
tmux new -s nanochat-training

# 稍后可以分离/重新连接:
# screen: Ctrl+A 然后 D 分离, screen -r 重新连接
# tmux: Ctrl+B 然后 D 分离, tmux attach 重新连接
```

### 初始化报告

```bash
# 重置报告并写入系统信息
python -m nanochat.report reset

# 这会创建 report/ 目录并写入:
# - GPU 信息
# - 系统规格
# - Git commit hash
# - 时间戳
```

### 选择模型大小

编辑训练脚本或设置参数：

```bash
# d20 (561M 参数, $100, 4小时) - 首次训练推荐
export MODEL_DEPTH=20
export DEVICE_BATCH_SIZE=32
export NUM_SHARDS=240

# d26 (1.1B 参数, $300, 12小时)
# export MODEL_DEPTH=26
# export DEVICE_BATCH_SIZE=16
# export NUM_SHARDS=450

# d32 (1.9B 参数, $800, 33小时)
# export MODEL_DEPTH=32
# export DEVICE_BATCH_SIZE=16
# export NUM_SHARDS=800
```

---

## 阶段 1: Tokenizer 训练

**时间**: ~30分钟
**数据**: ~2 GB

### 步骤 1.1: 下载初始训练数据

```bash
# 下载前8个分片 (~2GB) 用于 tokenizer 训练
python -m nanochat.dataset -n 8

# 正在发生什么:
# - 从 HuggingFace 下载 Parquet 文件
# - 数据源: FineWeb-Edu-100B 数据集
# - 每个分片: ~100MB 压缩后, ~250M 字符
# - 位置: $NANOCHAT_BASE_DIR/data/

# 监控进度:
ls -lh $NANOCHAT_BASE_DIR/data/*.parquet | wc -l  # 应该显示 8
```

**预期输出:**
```
正在下载分片 0/8...
正在下载分片 1/8...
...
正在下载分片 7/8...
所有分片下载成功。
```

### 步骤 1.2: 启动后台下载

在 tokenizer 训练时，在后台下载剩余数据：

```bash
# 下载总共 240 个分片用于 d20 (或 450 用于 d26, 800 用于 d32)
python -m nanochat.dataset -n $NUM_SHARDS &
DATASET_PID=$!

# 保存 PID 以便稍后检查
echo $DATASET_PID > /tmp/dataset_download.pid

# 在另一个终端监控进度:
# watch -n 10 "ls $NANOCHAT_BASE_DIR/data/*.parquet | wc -l"
```

### 步骤 1.3: 训练 Tokenizer

```bash
# 在 20 亿字符上训练 BPE tokenizer
python -m scripts.tok_train --max_chars=2000000000

# 参数:
# --max_chars: 训练字符数 (推荐 2B)
# --vocab_size: 65536 (默认, 2^16)
# --doc_cap: 10000 (每个文档最大字符数)

# 训练过程:
# 1. 从 parquet 文件流式读取文本
# 2. 使用正则模式分割文本
# 3. 执行 BPE 合并
# 4. 创建 65536 token 的词汇表
```

**预期输出:**
```
max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 65,536
正在处理序列...
进度: 10%...
进度: 50%...
进度: 100%
训练时间: 1234.56秒
tokenizer 已保存到 ~/.cache/nanochat/tokenizer/tokenizer.pkl
```

### 步骤 1.4: 评估 Tokenizer

```bash
# 测试 tokenizer 并计算压缩率
python -m scripts.tok_eval

# 这会评估:
# - 压缩率 (每 token 的字符数)
# - 编码/解码正确性
# - 特殊 token 处理
```

**预期输出:**
```
Tokenizer 加载成功
词汇表大小: 65536
特殊 token: 9
平均压缩率: 4.8 字符/token
测试通过: ✓
```

### 步骤 1.5: 验证数据下载

```bash
# 检查后台下载是否完成
wait $DATASET_PID

# 验证分片数量
SHARD_COUNT=$(ls $NANOCHAT_BASE_DIR/data/*.parquet | wc -l)
echo "已下载 $SHARD_COUNT 个分片"

# 对于 d20，应该是 240
if [ $SHARD_COUNT -ge $NUM_SHARDS ]; then
    echo "✓ 所有数据下载完成！"
else
    echo "⚠ 仍在下载... ($SHARD_COUNT/$NUM_SHARDS)"
fi
```

---

## 阶段 2: 基础模型预训练

**时间**: ~2-3小时 (d20), ~8小时 (d26), ~22小时 (d32)
**成本**: 总预算中的约 $70

这是计算最密集的阶段！

### 步骤 2.1: 理解数学计算

```
模型: d20 (depth=20)
参数量: 561M
训练 token 数: 561M × 20 = 11.2B (Chinchilla 缩放法则)
字符数: 11.2B × 4.8 = 54B 字符
数据分片: 54B / 250M = 216 分片 (我们下载 240 以确保安全)

训练配置:
- 序列长度: 2048 tokens
- 每 GPU 批量: 32
- 总批量: 524,288 tokens
- 梯度累积: 自动
- 学习率调度: 带预热的余弦
```

### 步骤 2.2: 开始基础训练

```bash
# 训练基础模型 (d20 示例)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=32 \
    --run=my_d20_model

# 如果使用 wandb:
# --run=my_d20_model 会记录到 https://wandb.ai/你的用户名/nanochat/runs/my_d20_model

# 参数说明:
# --depth=20: 模型大小 (20层 = 561M 参数)
# --device_batch_size=32: 每个 GPU 的批量大小
# --run: 实验名称 (用于 wandb)
```

**训练脚本会:**
1. 从零初始化模型
2. 设置优化器 (权重用 Muon，嵌入用 AdamW)
3. 流式加载数据
4. 训练计算好的迭代次数
5. 周期性保存检查点
6. 周期性评估

### 步骤 2.3: 监控训练进度

打开另一个终端并监控：

```bash
# 观察 GPU 利用率
watch -n 1 nvidia-smi

# 预期:
# - 所有 8 个 GPU 利用率 90-100%
# - 显存: d20 每 GPU ~40-50GB
# - 功耗: 每 GPU ~500-600W

# 查看训练日志
tail -f speedrun.log  # 如果使用 screen 日志记录

# 观察损失下降
# 初始损失: ~11-12
# 最终损失: ~3.5-4.0 (d20)
```

### 步骤 2.4: 检查点

检查点周期性保存到:
```
$NANOCHAT_BASE_DIR/checkpoints/base/
├── model.pth           # 模型权重 (d20 约 2.2GB)
├── optimizer.pth       # 优化器状态 (d20 约 6.6GB)
└── metadata.json       # 超参数，迭代计数
```

**从中断恢复:**
```bash
# 训练会自动从最新检查点恢复
# 只需重新运行相同的命令:
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=32 \
    --run=my_d20_model
```

### 步骤 2.5: 评估基础模型

```bash
# 训练完成后，在验证集上评估
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_loss

# 计算:
# - 验证损失
# - 每字节比特数 (BPB)
# - 生成示例文本

# 在 CORE 基准上评估
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_eval

# CORE 任务:
# - MMLU (多学科问答)
# - HellaSwag (常识推理)
# - ARC (科学问题)
# - PIQA, 等
```

**d20 的预期 CORE 分数:** ~0.22 (略低于 GPT-2)

---

## 阶段 3: 中间训练

**时间**: ~30分钟
**目的**: 教模型对话结构和工具使用

### 步骤 3.1: 下载对话数据

```bash
# 下载合成对话数据
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 这个文件 (~2MB) 包含:
# - 合成对话
# - 展示正确格式
# - 教授模型个性
```

### 步骤 3.2: 检查对话格式

```bash
# 查看示例对话
head -n 1 $NANOCHAT_BASE_DIR/identity_conversations.jsonl | jq .

# 格式:
# {
#   "messages": [
#     {"role": "user", "content": "你好！"},
#     {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
#   ]
# }
```

### 步骤 3.3: 运行中间训练

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.mid_train -- \
    --device_batch_size=32 \
    --run=my_d20_model

# 训练内容:
# - 身份对话
# - GSM8K (带计算器工具的数学)
# - MMLU (多选题)
# - SmolTalk (对话数据)
# - SpellingBee (字符计数)

# 教授内容:
# - 特殊 token 使用 (<|user_start|>, 等)
# - 工具调用 (计算器, Python)
# - 多选题格式
```

### 步骤 3.4: 评估中间训练

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_eval -- -i mid

# 评估任务:
# - ARC-Easy, ARC-Challenge
# - GSM8K (带工具使用)
# - HumanEval (代码生成)
# - MMLU
# - ChatCORE
```

**预期结果 (d20):**
```
ARC-Challenge: ~0.28
ARC-Easy: ~0.35
GSM8K: ~0.02-0.03
HumanEval: ~0.06-0.07
MMLU: ~0.31
```

---

## 阶段 4: 监督微调

**时间**: ~20分钟
**目的**: 提升对话能力

### 步骤 4.1: 运行 SFT

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft -- \
    --run=my_d20_model

# 单轮训练:
# - 任务混合 (与中间训练相同)
# - 专注于提升准确率
```

### 步骤 4.2: 评估 SFT

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_eval -- -i sft

# 应该看到改进:
# - GSM8K: ~0.04-0.05 (+2-3%)
# - HumanEval: ~0.08-0.09 (+2%)
# - 其他: 略有提升
```

---

## 阶段 5: 强化学习（可选）

**时间**: ~20分钟
**目的**: 通过奖励优化进一步提升 GSM8K

### 步骤 5.1: 运行 RL 训练

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_rl -- \
    --run=my_d20_model

# GRPO (Group Relative Policy Optimization):
# 1. 采样多个补全
# 2. 评估哪些是正确的
# 3. 使用正确性作为奖励
# 4. 更新策略以偏好正确答案
```

### 步骤 5.2: 评估 RL

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_eval -- -i rl -a GSM8K

# 预期改进:
# GSM8K: ~0.07-0.08 (比 SFT 提升 3-4%)
```

---

## 阶段 6: 推理与评估

### 步骤 6.1: 生成最终报告

```bash
# 编译所有评估结果
python -m nanochat.report generate

# 这会创建 report.md，包含:
# - 系统信息
# - 训练超参数
# - 所有评估结果
# - 训练时间和成本
# - 代码统计

# 查看报告
cat report.md

# 复制到当前目录便于访问
cp $NANOCHAT_BASE_DIR/report/report.md ./report.md
```

### 步骤 6.2: 测试命令行聊天

```bash
# 通过命令行与模型聊天
python -m scripts.chat_cli

# 交互模式 - 输入并按回车
# 输入 'exit' 或 'quit' 退出

# 单次查询模式:
python -m scripts.chat_cli -p "为什么天空是蓝色的？"
python -m scripts.chat_cli -p "写一首关于 AI 的俳句"
python -m scripts.chat_cli -p "计算 123 × 456"
```

### 步骤 6.3: 启动 Web 界面

```bash
# 启动 web 服务器
python -m scripts.chat_web

# 默认运行在: http://0.0.0.0:8000
```

**访问 UI:**

如果在本地运行:
```
http://localhost:8000
```

如果在远程服务器运行 (例如 Lambda Labs):
```
http://你的服务器IP:8000
```

**功能:**
- ChatGPT 风格界面
- 流式响应
- 对话历史
- 工具使用 (计算器, Python)

### 步骤 6.4: 生成示例

```bash
# 生成样本并评估
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_loss

# 你的模型可以做什么:

# ✓ 回答问题
"问: 法国的首都是什么？
 答: 法国的首都是巴黎。"

# ✓ 写故事
"从前，有一个年轻的程序员..."

# ✓ 简单数学 (带计算器)
"问: 15 × 23 是多少？
 答: <<15*23=345>> 答案是 345。"

# ⚠ 局限性 (d20 模型):
# - 频繁产生幻觉
# - 推理能力有限
# - 像个"幼儿园小朋友"
```

---

## 监控与调试

### 实时监控

```bash
# 终端 1: 训练
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# 终端 2: GPU 监控
watch -n 1 nvidia-smi

# 终端 3: 损失跟踪
tail -f speedrun.log | grep "loss"

# 终端 4: 磁盘使用
watch -n 60 "df -h | grep -E '(Filesystem|cache)'"
```

### 需要关注的关键指标

**预训练期间:**
- **损失**: 应该从 ~11 降到 ~3.5-4.0
- **学习率**: 预热然后衰减
- **GPU 利用率**: 应该 90-100%
- **Tokens/秒**: ~50,000-100,000 (因模型大小而异)

**中间训练/SFT 期间:**
- **损失**: 更低，~1-2
- **任务准确率**: 检查评估分数

### 检查点与恢复

```bash
# 检查点保存在:
$NANOCHAT_BASE_DIR/checkpoints/

# 结构:
checkpoints/
├── base/
│   ├── model.pth
│   ├── optimizer.pth
│   └── metadata.json
├── mid/
├── sft/
└── rl/

# 中断后恢复训练:
# 只需重新运行相同命令 - 会自动恢复！

# 手动指定检查点:
# 编辑脚本或使用 --resume 标志（如果可用）
```

### Wandb 仪表板

如果设置了 Weights & Biases:

```bash
# 训练期间，访问:
# https://wandb.ai/你的用户名/nanochat

# 可以监控:
# - 损失曲线
# - 学习率调度
# - GPU 利用率
# - 评估指标
# - 生成样本
```

---

## 成本与时间估算

### d20 模型 (561M 参数) - "$100 模型"

**训练时间:** 总共约 4 小时

| 阶段 | 时间 | 成本 | 说明 |
|------|------|------|------|
| Tokenizer | 30分钟 | $12 | 一次性，可复用 |
| 基础预训练 | 2.5小时 | $60 | 最昂贵的阶段 |
| 中间训练 | 20分钟 | $8 | 快速适应 |
| SFT | 15分钟 | $6 | 单轮 |
| RL (可选) | 15分钟 | $6 | GSM8K 改进 |
| 评估 | 20分钟 | $8 | 所有基准 |
| **总计** | **~4小时** | **~$100** | |

**性能:**
- CORE: ~0.22 (低于 GPT-2)
- GSM8K: ~0.07 (有 RL)
- 适合: 学习、实验

### d26 模型 (1.1B 参数) - "$300 模型"

**训练时间:** 总共约 12 小时

| 阶段 | 时间 | 成本 |
|------|------|------|
| Tokenizer | 30分钟 | $12 |
| 基础预训练 | 8小时 | $192 |
| 中间训练 | 1小时 | $24 |
| SFT | 30分钟 | $12 |
| RL (可选) | 30分钟 | $12 |
| 评估 | 1小时 | $24 |
| **总计** | **~12小时** | **~$276** |

**性能:**
- CORE: ~0.28 (GPT-2 水平)
- GSM8K: ~0.15-0.20
- 适合: 认真的实验

### d32 模型 (1.9B 参数) - "$800 模型"

**训练时间:** 总共约 33 小时

| 阶段 | 时间 | 成本 |
|------|------|------|
| Tokenizer | 30分钟 | $12 |
| 基础预训练 | 22小时 | $528 |
| 中间训练 | 2小时 | $48 |
| SFT | 1小时 | $24 |
| RL (可选) | 1小时 | $24 |
| 评估 | 2小时 | $48 |
| **总计** | **~33小时** | **~$684** |

**性能:**
- CORE: ~0.32-0.35 (超过 GPT-2)
- GSM8K: ~0.25-0.30
- 适合: 预算内的最佳结果

---

## 故障排除

### 常见问题与解决方案

#### 问题 1: 内存不足 (OOM)

**症状:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**
```bash
# 减少 device_batch_size
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=16  # 原来是 32

# 或更低: 8, 4, 2

# 代码会自动增加梯度累积
# 以保持相同的有效批量大小
```

#### 问题 2: 数据下载缓慢

**症状:**
- 数据集下载时间很长
- 连接超时

**解决方案:**
```bash
# 1. 检查网络连接
ping huggingface.co

# 2. 手动下载分片
python -c "
from nanochat.dataset import download_shard
for i in range(240):
    download_shard(i)
    print(f'已下载分片 {i}/240')
"

# 3. 使用更少的分片 (训练更多轮)
# 可接受但学习稍慢
```

#### 问题 3: NCCL 错误 (多 GPU)

**症状:**
```
NCCL error: unhandled system error
```

**解决方案:**
```bash
# 设置 NCCL 调试标志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 重新运行训练
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# 如果仍然存在，尝试单 GPU:
python -m scripts.base_train -- --depth=20
# (会慢 8 倍，但有助于隔离问题)
```

#### 问题 4: 检查点损坏

**症状:**
- 训练无法恢复
- "找不到检查点" 错误

**解决方案:**
```bash
# 检查检查点目录
ls -lh $NANOCHAT_BASE_DIR/checkpoints/base/

# 如果损坏，删除并重新开始
rm -rf $NANOCHAT_BASE_DIR/checkpoints/base/

# 训练将从头开始
```

#### 问题 5: 模型性能差

**症状:**
- 模型生成胡言乱语
- 准确率很低

**可能原因:**
1. **数据不足**: 验证分片数量
   ```bash
   ls $NANOCHAT_BASE_DIR/data/*.parquet | wc -l
   ```

2. **训练未完成**: 检查迭代计数
   ```bash
   cat $NANOCHAT_BASE_DIR/checkpoints/base/metadata.json | jq .iteration
   ```

3. **加载错误的检查点**: 检查检查点路径

4. **Tokenizer 问题**: 验证 tokenizer 工作正常
   ```bash
   python -m scripts.tok_eval
   ```

### 获取帮助

1. **检查日志**: 总是先检查训练日志
2. **GitHub Issues**: https://github.com/karpathy/nanochat/issues
3. **讨论区**: https://github.com/karpathy/nanochat/discussions
4. **DeepWiki**: https://deepwiki.com/karpathy/nanochat

---

## 快速参考命令

### 一键训练（首次推荐）

```bash
# speedrun 脚本会为你完成所有事情
bash speedrun.sh

# 或使用 screen:
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# 使用 wandb:
WANDB_RUN=my_run bash speedrun.sh
```

### 手动分步执行

```bash
# 1. 设置
uv venv && source .venv/bin/activate
uv sync --extra gpu
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 2. Tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 240 &
python -m scripts.tok_train --max_chars=2000000000

# 3. 基础模型
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# 4. 中间训练
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# 5. SFT
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 6. RL (可选)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K

# 7. 报告
python -m nanochat.report generate

# 8. 推理
python -m scripts.chat_web
```

---

## 下一步

完成首次 d20 训练后：

1. **尝试不同提示**: 试试不同的问题和任务
2. **分析报告**: 理解优势和劣势
3. **扩大规模**: 尝试 d26 或 d32 获得更好性能
4. **定制**: 添加你自己的训练数据 (参见讨论区)
5. **贡献**: 分享你的发现、改进或问题

---

## 额外资源

- **主仓库**: https://github.com/karpathy/nanochat
- **英文教程**: [docs/training_guide_en.md](training_guide_en.md)
- **纯 Python Tokenizer**: [docs/pure_python_tokenizer.md](pure_python_tokenizer.md)
- **定制指南**: https://github.com/karpathy/nanochat/discussions/139
- **添加能力**: https://github.com/karpathy/nanochat/discussions/164

---

**祝训练顺利！🚀**

如果你成功训练了模型，考虑在 GitHub Discussions 中分享你的经验！
