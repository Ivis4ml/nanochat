# Deep Dive: 完整训练流程 (base_train.py)

> **文件**: `scripts/base_train.py` (354 行)
> **作用**: nanochat 的核心训练流程 - 从初始化到模型训练完成
> **核心特性**: 分布式训练、混合优化、动态评估、检查点管理

---

## 📋 目录

- [概述](#概述)
- [训练流程架构](#训练流程架构)
- [Phase 1: 初始化设置](#phase-1-初始化设置)
- [Phase 2: 数据加载](#phase-2-数据加载)
- [Phase 3: 模型与优化器](#phase-3-模型与优化器)
- [Phase 4: 训练循环](#phase-4-训练循环)
- [Phase 5: 评估系统](#phase-5-评估系统)
- [Phase 6: 检查点管理](#phase-6-检查点管理)
- [分布式训练详解](#分布式训练详解)
- [性能分析](#性能分析)
- [实战示例](#实战示例)

---

## 概述

`base_train.py` 是 nanochat 的**主训练脚本**，它整合了所有组件：

```python
┌─────────────────────────────────────────────────────────────┐
│                    base_train.py                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Configurator │  │   Dataset    │  │     GPT      │     │
│  │   + Args     │→ │  + Loader    │→ │   Model      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│          ↓                 ↓                  ↓            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Distributed │  │  AdamW       │  │  Training    │     │
│  │    Setup     │→ │  + Muon      │→ │    Loop      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│          ↓                 ↓                  ↓            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Evaluation  │  │ Checkpoint   │  │   Report     │     │
│  │    (CORE)    │  │   Manager    │  │  Generator   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心特性

1. **多阶段训练**: 预训练 → 微调 → 评估
2. **分布式优化**: DDP + ZeRO-2 风格的参数分片
3. **混合优化器**: AdamW (嵌入) + Muon (矩阵)
4. **动态评估**: 每 N 步评估，自动保存最佳模型
5. **容错机制**: 自动恢复、梯度检查点、混合精度训练

---

## 训练流程架构

### 完整时间线

```
Time →
├─ [0] 初始化
│  ├─ parse_args()              # 解析命令行参数
│  ├─ setup_distributed()       # 分布式环境初始化
│  ├─ load_configurator()       # 加载配置
│  └─ set_seed()               # 设置随机种子
│
├─ [1] 数据准备
│  ├─ ensure_dataset_downloaded()
│  ├─ load_tokenizer()
│  └─ create_dataloader()
│
├─ [2] 模型构建
│  ├─ GPT.from_pretrained() or init_model()
│  ├─ model.to(device)
│  ├─ DDP(model)
│  └─ setup_optimizers()       # AdamW + Muon
│
├─ [3] 训练循环 (主循环)
│  ├─ for step in range(num_steps):
│  │  ├─ dataloader → batch
│  │  ├─ forward()            # 前向传播
│  │  ├─ loss.backward()      # 反向传播
│  │  ├─ optimizer.step()     # 参数更新
│  │  ├─ scheduler.step()     # 学习率调整
│  │  └─ [每 N 步] evaluate()
│  │
│  └─ [训练结束]
│
├─ [4] 最终评估
│  ├─ load_best_checkpoint()
│  ├─ run_core_eval()
│  └─ generate_report()
│
└─ [5] 清理
   └─ cleanup_distributed()
```

---

## Phase 1: 初始化设置

### 1.1 命令行参数解析

```python
# scripts/base_train.py:15-35
def parse_args():
    parser = argparse.ArgumentParser(description='Train nanochat model')

    # 核心训练参数
    parser.add_argument('--config', type=str, default='base',
                       help='配置名称 (base/mid/large)')
    parser.add_argument('--num_steps', type=int, default=10000,
                       help='训练步数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='全局批大小')
    parser.add_argument('--grad_accum', type=int, default=1,
                       help='梯度累积步数')

    # 分布式参数
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)

    # 评估参数
    parser.add_argument('--eval_interval', type=int, default=1000,
                       help='每 N 步评估一次')
    parser.add_argument('--eval_samples', type=int, default=100)

    return parser.parse_args()
```

**关键点**:
- `--config`: 选择预定义配置（见 `configurator.py`）
- `--grad_accum`: 梯度累积实现更大的有效批大小
- `--eval_interval`: 平衡训练速度和评估频率

### 1.2 分布式环境初始化

```python
# scripts/base_train.py:45-78
def setup_distributed(args):
    """设置分布式训练环境"""

    # 检测环境变量
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])

    # 初始化进程组
    if args.world_size > 1:
        dist.init_process_group(
            backend='nccl',           # GPU 通信后端
            init_method='env://',     # 使用环境变量
            world_size=args.world_size,
            rank=args.rank
        )

        # 设置当前设备
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')

        print(f"[Rank {args.rank}] Initialized process group")
        print(f"  World size: {args.world_size}")
        print(f"  Backend: nccl")
        print(f"  Device: {device}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Single GPU] Device: {device}")

    return device
```

**操作符深度解析**:

| 操作 | 作用 | 何时调用 |
|------|------|----------|
| `dist.init_process_group()` | 初始化进程间通信 | 多 GPU 训练时 |
| `torch.cuda.set_device()` | 设置当前进程使用的 GPU | 每个 rank 一次 |
| `backend='nccl'` | 使用 NVIDIA 的 NCCL 库 | GPU 间高速通信 |

**为什么需要分布式?**

```
单 GPU (16GB):
┌────────────────────┐
│ Model: 6GB         │
│ Activations: 8GB   │
│ Optimizer: 2GB     │ = 16GB (刚好)
└────────────────────┘

4x GPU (分布式):
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ 6GB  │ │ 6GB  │ │ 6GB  │ │ 6GB  │ = 24GB 可用
│ 4GB  │ │ 4GB  │ │ 4GB  │ │ 4GB  │   (模型复制)
│ 0.5GB│ │ 0.5GB│ │ 0.5GB│ │ 0.5GB│   (优化器分片)
└──────┘ └──────┘ └──────┘ └──────┘
```

### 1.3 配置加载

```python
# scripts/base_train.py:82-95
def load_configurator(args, device):
    """加载配置器并创建模型配置"""

    # 从预设配置创建
    config = Configurator.from_preset(args.config)

    # 覆盖命令行参数
    config.training.batch_size = args.batch_size
    config.training.grad_accum_steps = args.grad_accum
    config.training.num_steps = args.num_steps
    config.training.eval_interval = args.eval_interval

    # 计算有效批大小
    effective_batch = (args.batch_size *
                      args.grad_accum *
                      args.world_size)

    print(f"Configuration: {args.config}")
    print(f"  Model size: {config.model.n_params / 1e6:.1f}M params")
    print(f"  Batch size: {args.batch_size} (effective: {effective_batch})")
    print(f"  Sequence length: {config.model.seq_len}")

    return config
```

**有效批大小计算**:

```
有效批大小 = local_batch × grad_accum × world_size

示例:
  local_batch = 16    # 每个 GPU 的微批
  grad_accum = 4      # 梯度累积
  world_size = 8      # 8 个 GPU

  → effective = 16 × 4 × 8 = 512

等价于单 GPU 上运行批大小 512，但内存只需要 16 的内存！
```

---

## Phase 2: 数据加载

### 2.1 数据集下载与准备

```python
# scripts/base_train.py:100-115
def prepare_data(config, args):
    """准备训练数据"""

    # 确保数据集已下载
    dataset_name = config.data.dataset_name
    data_dir = ensure_dataset_downloaded(dataset_name)

    print(f"Dataset: {dataset_name}")
    print(f"  Location: {data_dir}")

    # 加载分词器
    tokenizer_path = data_dir / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print(f"Tokenizer loaded")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  Type: BPE")

    # 创建数据加载器
    train_loader = create_dataloader(
        data_dir=data_dir,
        split='train',
        batch_size=args.batch_size,
        seq_len=config.model.seq_len,
        world_size=args.world_size,
        rank=args.rank,
        num_workers=4,
        pin_memory=True  # 加速 CPU→GPU 传输
    )

    val_loader = create_dataloader(
        data_dir=data_dir,
        split='validation',
        batch_size=args.batch_size,
        seq_len=config.model.seq_len,
        world_size=args.world_size,
        rank=args.rank,
        shuffle=False    # 验证集不打乱
    )

    return tokenizer, train_loader, val_loader
```

### 2.2 DataLoader 细节

```python
# nanochat/dataloader.py:15-54 (简化版)
class TokenDataset(Dataset):
    """Token 数据集 - 从预分词的文件读取"""

    def __init__(self, data_path, seq_len):
        # 使用 mmap 加载大文件（避免全部加载到内存）
        self.tokens = np.memmap(
            data_path,
            dtype=np.uint16,  # 词汇表 < 65536
            mode='r'
        )
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for next token prediction

        # 提取序列
        chunk = self.tokens[start:end]

        # 输入和标签
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y

def create_dataloader(data_dir, split, batch_size, seq_len,
                     world_size, rank, **kwargs):
    """创建分布式数据加载器"""

    # 加载数据集
    data_path = data_dir / f'{split}.bin'
    dataset = TokenDataset(data_path, seq_len)

    # 分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == 'train')
    )

    # 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=kwargs.get('num_workers', 0),
        pin_memory=kwargs.get('pin_memory', False),
        drop_last=True  # 确保所有 GPU 批大小相同
    )

    return loader
```

**DistributedSampler 工作原理**:

```python
# 假设有 1000 个样本，4 个 GPU
dataset = [0, 1, 2, ..., 999]

# Rank 0 得到: [0, 4, 8, 12, ..., 996]
# Rank 1 得到: [1, 5, 9, 13, ..., 997]
# Rank 2 得到: [2, 6, 10, 14, ..., 998]
# Rank 3 得到: [3, 7, 11, 15, ..., 999]

每个 GPU 处理 250 个样本，无重复，无遗漏
```

**内存映射 (mmap) 的优势**:

```
传统方式:
  data = np.load('train.bin')  # 一次性加载 10GB 到内存
  ❌ 内存占用高
  ❌ 启动慢

mmap 方式:
  data = np.memmap('train.bin')  # 只映射，不加载
  ✅ 按需加载 (仅加载当前批)
  ✅ 启动快
  ✅ 支持超大文件
```

---

## Phase 3: 模型与优化器

### 3.1 模型初始化

```python
# scripts/base_train.py:120-155
def initialize_model(config, device, args):
    """初始化或加载模型"""

    # 检查是否从检查点恢复
    checkpoint_path = args.checkpoint if hasattr(args, 'checkpoint') else None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")

        # 加载模型
        model = GPT.from_pretrained(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        start_step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"  Resuming from step {start_step}")
        print(f"  Best validation loss: {best_val_loss:.4f}")
    else:
        print("Initializing new model")

        # 从配置创建模型
        model = GPT(config.model)
        start_step = 0
        best_val_loss = float('inf')

        print(f"  Parameters: {model.num_parameters():,}")
        print(f"  Non-embedding params: {model.num_parameters(non_embedding=True):,}")

    # 移动到设备
    model = model.to(device)

    # 包装为 DDP（分布式数据并行）
    if args.world_size > 1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False  # 提高性能
        )
        print(f"[Rank {args.rank}] Model wrapped in DDP")

    return model, start_step, best_val_loss
```

**DDP 原理**:

```
前向传播 (Forward):
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ GPU 0  │  │ GPU 1  │  │ GPU 2  │  │ GPU 3  │
│ Batch0 │  │ Batch1 │  │ Batch2 │  │ Batch3 │
│  ↓     │  │  ↓     │  │  ↓     │  │  ↓     │
│ Loss0  │  │ Loss1  │  │ Loss2  │  │ Loss3  │
└────────┘  └────────┘  └────────┘  └────────┘

反向传播 (Backward):
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ ∇Loss0 │  │ ∇Loss1 │  │ ∇Loss2 │  │ ∇Loss3 │
│   ↓    │  │   ↓    │  │   ↓    │  │   ↓    │
└────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘
     │           │           │           │
     └───────────┴──All-Reduce───────────┘
                     ↓
          ∇ = (∇Loss0 + ∇Loss1 + ∇Loss2 + ∇Loss3) / 4
                     ↓
     ┌───────────┬───────────┬───────────┐
     ↓           ↓           ↓           ↓
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ GPU 0  │  │ GPU 1  │  │ GPU 2  │  │ GPU 3  │
│ Update │  │ Update │  │ Update │  │ Update │
└────────┘  └────────┘  └────────┘  └────────┘
```

### 3.2 优化器设置

```python
# scripts/base_train.py:160-195
def setup_optimizers(model, config, device):
    """设置混合优化器: AdamW (嵌入) + Muon (矩阵)"""

    # 参数分组
    print("Setting up optimizers...")

    # 获取原始模型（去掉 DDP 包装）
    raw_model = model.module if hasattr(model, 'module') else model

    # 调用模型的 setup_optimizers 方法
    optimizers = raw_model.setup_optimizers(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=device
    )

    # 返回的是一个字典: {'AdamW': adamw_opt, 'Muon': muon_opt}
    adamw_opt = optimizers['AdamW']
    muon_opt = optimizers['Muon']

    print(f"  AdamW parameters: {len(adamw_opt.param_groups[0]['params'])}")
    print(f"  Muon parameters: {len(muon_opt.param_groups[0]['params'])}")

    return adamw_opt, muon_opt
```

这里调用的是 `gpt.py` 中的 `setup_optimizers()` 方法：

```python
# nanochat/gpt.py:285-312 (详细见 04_optimizers_adamw_muon.md)
def setup_optimizers(self, learning_rate, weight_decay, device):
    """
    参数分组策略:
    1. 嵌入层 (wte, wpe) → AdamW
       - 学习率较高 (lr × sqrt(d_model))
       - 因为嵌入是稀疏更新的

    2. 矩阵参数 (W_q, W_k, W_v, W_o, W_fc1, W_fc2) → Muon
       - 学习率标准
       - 使用 Newton-Schulz 正交化

    3. 非矩阵参数 (LayerNorm, bias) → AdamW
       - 不衰减 (weight_decay=0)
    """

    # 收集参数
    embedding_params = []
    matrix_params = []
    other_params = []

    for name, param in self.named_parameters():
        if not param.requires_grad:
            continue

        if 'wte' in name or 'wpe' in name:
            # 嵌入层
            embedding_params.append(param)
        elif param.ndim >= 2:
            # 矩阵参数 (至少是 2D)
            matrix_params.append(param)
        else:
            # 其他参数 (1D: bias, LayerNorm)
            other_params.append(param)

    # 创建优化器
    adamw_params = embedding_params + other_params
    adamw_opt = AdamW(
        adamw_params,
        lr=learning_rate * math.sqrt(self.config.d_model),  # 缩放学习率
        weight_decay=weight_decay,
        device=device
    )

    muon_opt = Muon(
        matrix_params,
        lr=learning_rate,
        momentum=0.95,
        device=device
    )

    return {'AdamW': adamw_opt, 'Muon': muon_opt}
```

**为什么嵌入层需要更高的学习率?**

```python
# 嵌入层更新频率分析
vocab_size = 50257
batch_size = 64
seq_len = 512

# 每个批次
tokens_per_batch = batch_size * seq_len = 32768

# 稀疏性
unique_tokens_per_batch ≈ 5000  # 远小于词汇表大小

# 更新概率
update_prob = 5000 / 50257 ≈ 10%

→ 90% 的嵌入向量在每个批次中都不会被更新！
→ 需要更高的学习率来补偿稀疏更新

反之，矩阵参数 (W_q, W_k, W_v...) 每次都会被更新
→ 使用标准学习率
```

### 3.3 学习率调度器

```python
# scripts/base_train.py:200-230
def setup_schedulers(adamw_opt, muon_opt, config, num_steps):
    """设置学习率调度器: Cosine Decay with Warmup"""

    warmup_steps = config.training.warmup_steps

    def lr_lambda(step):
        # Warmup: 线性增长
        if step < warmup_steps:
            return step / warmup_steps

        # Cosine Decay: 余弦衰减
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # 为两个优化器创建调度器
    adamw_scheduler = LambdaLR(adamw_opt, lr_lambda)
    muon_scheduler = LambdaLR(muon_opt, lr_lambda)

    return adamw_scheduler, muon_scheduler
```

**学习率变化曲线**:

```
Learning Rate
   1.0 ┤     ╭────────╮
       │    ╱          ╲
   0.8 ┤   ╱            ╲
       │  ╱              ╲
   0.6 ┤ ╱                ╲
       │╱                  ╲
   0.4 ┼                    ╲
       │                     ╲
   0.2 ┤                      ╲___
       │
   0.0 ┼─────────────────────────────
       0    Warmup      Training    End
           (1000)        (10000)

Phase 1: Warmup (0 → 1000 steps)
  - 线性增长: lr = step / 1000 × lr_max
  - 避免训练初期的不稳定

Phase 2: Cosine Decay (1000 → 10000 steps)
  - 余弦衰减: lr = 0.5 × (1 + cos(π × progress)) × lr_max
  - 平滑地降低学习率
  - 在训练后期进行精细调整
```

---

## Phase 4: 训练循环

### 4.1 主训练循环结构

```python
# scripts/base_train.py:235-320
def train(model, train_loader, val_loader, adamw_opt, muon_opt,
          adamw_scheduler, muon_scheduler, config, device, args,
          start_step=0, best_val_loss=float('inf')):
    """主训练循环"""

    # 设置混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # 获取原始模型（用于保存）
    raw_model = model.module if hasattr(model, 'module') else model

    # 训练循环
    model.train()
    total_loss = 0.0
    step = start_step

    print(f"\nStarting training from step {start_step}")
    print(f"Target steps: {config.training.num_steps}")
    print(f"Evaluation interval: {args.eval_interval}")
    print("-" * 60)

    # 创建进度条（仅在 rank 0）
    if args.rank == 0:
        pbar = tqdm(total=config.training.num_steps - start_step,
                   desc='Training')

    while step < config.training.num_steps:
        # 梯度累积循环
        for micro_step in range(config.training.grad_accum_steps):
            # 获取批数据
            try:
                x, y = next(train_iter)
            except (StopIteration, NameError):
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            # 前向传播 (混合精度)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                logits, loss = model(x, y)

                # 归一化损失（梯度累积）
                loss = loss / config.training.grad_accum_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 累积损失
            total_loss += loss.item()

        # 梯度裁剪
        scaler.unscale_(adamw_opt)
        scaler.unscale_(muon_opt)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip_norm
        )

        # 优化器步骤
        scaler.step(adamw_opt)
        scaler.step(muon_opt)
        scaler.update()

        # 清零梯度
        adamw_opt.zero_grad(set_to_none=True)
        muon_opt.zero_grad(set_to_none=True)

        # 学习率调度
        adamw_scheduler.step()
        muon_scheduler.step()

        # 更新步数
        step += 1

        # 日志记录
        if step % args.log_interval == 0 and args.rank == 0:
            avg_loss = total_loss / args.log_interval
            lr_adamw = adamw_scheduler.get_last_lr()[0]
            lr_muon = muon_scheduler.get_last_lr()[0]

            print(f"Step {step:5d} | Loss: {avg_loss:.4f} | "
                  f"LR (AdamW): {lr_adamw:.2e} | LR (Muon): {lr_muon:.2e}")

            total_loss = 0.0

        # 评估
        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, args)

            if args.rank == 0:
                print(f"\n[Evaluation at step {step}]")
                print(f"  Validation loss: {val_loss:.4f}")

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        raw_model,
                        adamw_opt,
                        muon_opt,
                        step,
                        val_loss,
                        config,
                        args.checkpoint_dir / 'best_model.pt'
                    )
                    print(f"  ✓ New best model saved!")

            model.train()

        # 更新进度条
        if args.rank == 0:
            pbar.update(1)

    if args.rank == 0:
        pbar.close()

    return best_val_loss
```

### 4.2 混合精度训练详解

**为什么使用混合精度 (AMP)?**

```
FP32 (标准精度):
  优点: 精度高，稳定
  缺点: 内存占用大，速度慢

FP16 (半精度):
  优点: 内存减半，速度快 2-3x
  缺点: 数值不稳定，可能溢出

混合精度 (AMP):
  前向传播 → FP16 (快速)
  梯度计算 → FP16 (快速)
  参数更新 → FP32 (稳定)

  → 兼具速度和稳定性！
```

**GradScaler 工作原理**:

```python
# 问题: FP16 的表示范围
FP16_MIN = 6e-5    # 小于这个值会下溢
FP16_MAX = 65504   # 大于这个值会上溢

# 解决方案: 梯度缩放
scaler = GradScaler()

# Step 1: 缩放损失（避免梯度下溢）
scaled_loss = loss * 65536  # 默认缩放因子
scaled_loss.backward()

# Step 2: 反向传播
# 梯度自动缩放: grad = grad * 65536

# Step 3: 反缩放梯度（恢复原始尺度）
scaler.unscale_(optimizer)
# 梯度: grad = grad / 65536

# Step 4: 检查梯度是否有 NaN/Inf
if no NaN/Inf:
    optimizer.step()  # 更新参数
else:
    skip update       # 跳过更新，降低缩放因子

# Step 5: 动态调整缩放因子
scaler.update()
```

### 4.3 梯度累积详解

**为什么需要梯度累积?**

```
问题: GPU 内存限制
  想要: batch_size = 512
  实际: GPU 只能容纳 batch_size = 64

解决方案: 梯度累积
  1. 前向传播 batch_1 (64 samples) → loss_1
  2. 反向传播 loss_1 / 8 → grad_1

  3. 前向传播 batch_2 (64 samples) → loss_2
  4. 反向传播 loss_2 / 8 → grad_2

  ... (重复 8 次)

  8. 累积的梯度: grad_total = grad_1 + grad_2 + ... + grad_8
  9. 优化器更新: θ ← θ - lr × grad_total

  → 等价于 batch_size = 512，但只需要 64 的内存！
```

**代码实现**:

```python
grad_accum_steps = 8

for micro_step in range(grad_accum_steps):
    # 获取微批
    x, y = next(train_iter)  # shape: [64, 512]

    # 前向传播
    logits, loss = model(x, y)

    # 归一化损失
    loss = loss / grad_accum_steps  # 重要！

    # 反向传播（梯度累积）
    loss.backward()

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 优化器步骤（一次性更新所有累积的梯度）
optimizer.step()
optimizer.zero_grad()
```

**为什么要除以 grad_accum_steps?**

```python
# 不除的话:
loss_1.backward()  # grad += ∂loss_1/∂θ
loss_2.backward()  # grad += ∂loss_2/∂θ
...
loss_8.backward()  # grad += ∂loss_8/∂θ

# grad = ∂(loss_1 + loss_2 + ... + loss_8)/∂θ
# 这是 8 个批次的梯度之和，不是平均！

# 正确的做法:
(loss_1 / 8).backward()  # grad += (1/8) × ∂loss_1/∂θ
(loss_2 / 8).backward()  # grad += (1/8) × ∂loss_2/∂θ
...
(loss_8 / 8).backward()  # grad += (1/8) × ∂loss_8/∂θ

# grad = (1/8) × ∂(loss_1 + loss_2 + ... + loss_8)/∂θ
# 这是 8 个批次的梯度平均 ✓
```

### 4.4 梯度裁剪

```python
# 梯度裁剪: 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # 最大梯度范数
)
```

**原理**:

```python
# 计算全局梯度范数
total_norm = sqrt(sum(p.grad.norm()**2 for p in model.parameters()))

# 如果超过阈值，缩放梯度
if total_norm > max_norm:
    scale = max_norm / total_norm
    for p in model.parameters():
        p.grad *= scale

# 例子:
total_norm = 5.0
max_norm = 1.0
scale = 1.0 / 5.0 = 0.2

→ 所有梯度乘以 0.2，使得 total_norm = 1.0
```

**为什么需要梯度裁剪?**

```
训练初期:
  参数随机初始化
  → 可能产生非常大的梯度
  → 参数更新过大
  → 训练不稳定甚至崩溃

梯度裁剪:
  限制梯度的最大范数
  → 参数更新受控
  → 训练稳定

实际效果:
  Without clipping:
    Step 100: loss = 8.5
    Step 101: loss = nan  ← 梯度爆炸！

  With clipping (max_norm=1.0):
    Step 100: loss = 8.5
    Step 101: loss = 8.3
    Step 102: loss = 8.0
    ...
    Step 1000: loss = 3.2  ← 稳定训练
```

---

## Phase 5: 评估系统

### 5.1 验证损失评估

```python
# scripts/base_train.py:325-365
@torch.no_grad()
def evaluate(model, val_loader, device, args):
    """评估验证损失"""

    model.eval()
    total_loss = 0.0
    num_batches = 0

    # 限制评估批数（节省时间）
    max_eval_batches = args.eval_samples // args.batch_size

    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_eval_batches:
            break

        x, y = x.to(device), y.to(device)

        # 前向传播（不需要梯度）
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            logits, loss = model(x, y)

        total_loss += loss.item()
        num_batches += 1

    # 平均损失
    avg_loss = total_loss / num_batches

    # 分布式: 所有 rank 的平均
    if args.world_size > 1:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return avg_loss
```

**为什么使用 @torch.no_grad()?**

```python
# 评估时不需要计算梯度
with torch.no_grad():
    logits, loss = model(x, y)

# 优点:
# 1. 节省内存（不保存中间激活）
# 2. 加速计算（跳过梯度计算）
# 3. 避免梯度累积的问题

# 内存对比:
Training:   8GB (model) + 12GB (activations) = 20GB
Evaluation: 8GB (model) + 0GB (no gradients) = 8GB
→ 节省 60% 内存！
```

### 5.2 CORE 基准评估

```python
# scripts/base_train.py:370-410
def run_core_evaluation(model, tokenizer, device, args):
    """运行 CORE 基准评估"""

    print("\n" + "="*60)
    print("Running CORE benchmark evaluation")
    print("="*60)

    # 加载 CORE 数据集
    core_path = args.data_dir / 'core_eval.jsonl'

    if not core_path.exists():
        print(f"Warning: CORE dataset not found at {core_path}")
        print("Skipping CORE evaluation")
        return None

    # 运行评估
    results = evaluate_core(
        model=model,
        tokenizer=tokenizer,
        data_path=core_path,
        device=device,
        max_samples=args.core_max_samples,
        batch_size=args.core_batch_size
    )

    # 打印结果
    if args.rank == 0:
        print("\nCORE Evaluation Results:")
        print("-" * 40)
        for category, metrics in results.items():
            print(f"{category}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        print("-" * 40)

    return results
```

CORE 评估的详细实现见 `nanochat/core_eval.py`，这里展示它的结构：

```python
# CORE 任务类型
CORE_TASKS = {
    'comprehension': [
        'multiple_choice',
        'yes_no',
        'short_answer'
    ],
    'reasoning': [
        'arithmetic',
        'logical',
        'causal'
    ],
    'generation': [
        'summarization',
        'completion',
        'rewriting'
    ]
}

# 评估指标
def compute_metrics(predictions, references, task_type):
    if task_type in ['multiple_choice', 'yes_no']:
        # 准确率
        return accuracy_score(references, predictions)

    elif task_type == 'short_answer':
        # F1 分数
        return f1_score(references, predictions)

    elif task_type in ['summarization', 'completion', 'rewriting']:
        # ROUGE 分数
        return rouge_score(references, predictions)

    else:
        raise ValueError(f"Unknown task type: {task_type}")
```

---

## Phase 6: 检查点管理

### 6.1 保存检查点

```python
# scripts/base_train.py:415-455
def save_checkpoint(model, adamw_opt, muon_opt, step, val_loss,
                   config, checkpoint_path):
    """保存训练检查点"""

    # 构建检查点字典
    checkpoint = {
        # 模型状态
        'model_state_dict': model.state_dict(),

        # 优化器状态
        'adamw_state_dict': adamw_opt.state_dict(),
        'muon_state_dict': muon_opt.state_dict(),

        # 训练状态
        'step': step,
        'val_loss': val_loss,

        # 配置
        'config': config.to_dict(),

        # 元数据
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }

    # 保存到磁盘
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"  Step: {step}")
    print(f"  Validation loss: {val_loss:.4f}")
    print(f"  Size: {checkpoint_path.stat().st_size / 1e6:.1f} MB")
```

### 6.2 加载检查点

```python
# scripts/base_train.py:460-495
def load_checkpoint(checkpoint_path, model, adamw_opt, muon_opt, device):
    """加载训练检查点"""

    print(f"Loading checkpoint: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 恢复模型
    model.load_state_dict(checkpoint['model_state_dict'])

    # 恢复优化器
    adamw_opt.load_state_dict(checkpoint['adamw_state_dict'])
    muon_opt.load_state_dict(checkpoint['muon_state_dict'])

    # 恢复训练状态
    step = checkpoint['step']
    val_loss = checkpoint['val_loss']

    print(f"  Restored from step {step}")
    print(f"  Best validation loss: {val_loss:.4f}")

    return step, val_loss
```

### 6.3 检查点策略

```python
# 不同的保存策略
CHECKPOINT_STRATEGIES = {
    'every_n_steps': {
        'interval': 1000,
        'keep_last_n': 5,
        'example': 'ckpt_1000.pt, ckpt_2000.pt, ...'
    },

    'best_only': {
        'criterion': 'val_loss',
        'mode': 'min',
        'example': 'best_model.pt'
    },

    'epoch_end': {
        'per_epoch': True,
        'example': 'epoch_1.pt, epoch_2.pt, ...'
    },

    'latest': {
        'overwrite': True,
        'example': 'latest.pt'
    }
}
```

**示例: 完整的检查点管理**:

```python
class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_dir, keep_last_n=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_val_loss = float('inf')

    def save(self, model, optimizers, step, val_loss, config):
        """保存检查点"""

        # 1. 保存最新检查点
        latest_path = self.checkpoint_dir / 'latest.pt'
        self._save_checkpoint(
            latest_path, model, optimizers, step, val_loss, config
        )

        # 2. 保存周期性检查点
        if step % 1000 == 0:
            periodic_path = self.checkpoint_dir / f'step_{step}.pt'
            self._save_checkpoint(
                periodic_path, model, optimizers, step, val_loss, config
            )

            # 清理旧检查点
            self._cleanup_old_checkpoints()

        # 3. 保存最佳模型
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / 'best_model.pt'
            self._save_checkpoint(
                best_path, model, optimizers, step, val_loss, config
            )
            print(f"✓ New best model saved! (val_loss: {val_loss:.4f})")

    def _cleanup_old_checkpoints(self):
        """清理旧的周期性检查点"""

        # 获取所有 step_*.pt 文件
        checkpoints = sorted(
            self.checkpoint_dir.glob('step_*.pt'),
            key=lambda p: int(p.stem.split('_')[1])
        )

        # 保留最近的 N 个
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt.name}")
```

---

## 分布式训练详解

### 分布式通信原语

**1. All-Reduce**

```python
# 所有 rank 求和并广播结果
tensor = torch.tensor([rank], device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# 示例: 4 个 GPU
Rank 0: [0]     ┐
Rank 1: [1]     ├→ Sum = [6] → Broadcast
Rank 2: [2]     │               ↓
Rank 3: [3]     ┘     All ranks get [6]
```

**2. Broadcast**

```python
# 从 rank 0 广播到所有 rank
tensor = torch.tensor([rank], device='cuda')
dist.broadcast(tensor, src=0)

# 示例:
Rank 0: [42]    →  [42]
Rank 1: [0]     →  [42]  ← 从 rank 0 复制
Rank 2: [0]     →  [42]  ← 从 rank 0 复制
Rank 3: [0]     →  [42]  ← 从 rank 0 复制
```

**3. Reduce-Scatter**

```python
# 求和后分散到各个 rank
input_tensor = torch.tensor([...], device='cuda')
output_tensor = torch.zeros(chunk_size, device='cuda')

dist.reduce_scatter(output_tensor, [input_tensor.chunk(world_size)])

# 示例: 见 04_optimizers_adamw_muon.md
```

**4. All-Gather**

```python
# 收集所有 rank 的数据
tensor = torch.tensor([rank], device='cuda')
output = [torch.zeros(1, device='cuda') for _ in range(world_size)]

dist.all_gather(output, tensor)

# 示例:
Rank 0: [0] → [0, 1, 2, 3]
Rank 1: [1] → [0, 1, 2, 3]
Rank 2: [2] → [0, 1, 2, 3]
Rank 3: [3] → [0, 1, 2, 3]
```

### DDP 与 ZeRO 对比

```python
┌──────────────────────────────────────────────────────────────┐
│                    DDP (PyTorch 默认)                        │
├──────────────────────────────────────────────────────────────┤
│ GPU 0              GPU 1              GPU 2              GPU 3│
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────┐│
│ │ Model (全)   │  │ Model (全)   │  │ Model (全)   │  │ ...││
│ │ Optimizer(全)│  │ Optimizer(全)│  │ Optimizer(全)│  │ ...││
│ │ Gradients(全)│  │ Gradients(全)│  │ Gradients(全)│  │ ...││
│ └──────────────┘  └──────────────┘  └──────────────┘  └────┘│
│                                                                │
│ 优点: 简单                                                     │
│ 缺点: 每个 GPU 都存储完整的模型和优化器状态                   │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│              ZeRO-2 (Muon 使用的方式)                         │
├──────────────────────────────────────────────────────────────┤
│ GPU 0              GPU 1              GPU 2              GPU 3│
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────┐│
│ │ Model (全)   │  │ Model (全)   │  │ Model (全)   │  │ ...││
│ │ Optimizer[0] │  │ Optimizer[1] │  │ Optimizer[2] │  │ [3]││
│ │ Gradients[0] │  │ Gradients[1] │  │ Gradients[2] │  │ [3]││
│ └──────────────┘  └──────────────┘  └──────────────┘  └────┘│
│        ↓                 ↓                 ↓              ↓   │
│     Reduce           Reduce           Reduce          Reduce  │
│     Scatter          Scatter          Scatter         Scatter │
│                                                                │
│ 优点: 节省内存（优化器状态和梯度分片）                        │
│ 缺点: 需要额外的通信（reduce-scatter, all-gather）           │
└──────────────────────────────────────────────────────────────┘
```

---

## 性能分析

### 训练吞吐量

```python
# 理论峰值性能
GPU_TFLOPS = 312  # A100 的 TFlops (FP16)

# 模型计算量
model_params = 124_000_000  # 124M 参数
seq_len = 512
batch_size = 64

# 每个 token 的 FLOPs
flops_per_token = 6 * model_params  # 前向: 2x, 反向: 4x
total_flops = flops_per_token * seq_len * batch_size

# 每步时间 (秒)
time_per_step = total_flops / (GPU_TFLOPS * 1e12)

print(f"Theoretical time per step: {time_per_step:.3f}s")
print(f"Theoretical throughput: {seq_len * batch_size / time_per_step:.0f} tokens/s")

# 实际性能
# MFU (Model FLOPs Utilization) ≈ 50-60% on A100
actual_time = time_per_step / 0.55
print(f"Actual time per step: {actual_time:.3f}s")
```

### 内存使用

```python
# 模型大小
model_memory = model_params * 4  # FP32: 4 bytes/param
print(f"Model memory: {model_memory / 1e9:.2f} GB")

# 优化器状态 (AdamW)
# m, v: 2 × model_params × 4 bytes
optimizer_memory = model_params * 2 * 4
print(f"Optimizer memory: {optimizer_memory / 1e9:.2f} GB")

# 激活值 (Activations)
# 约等于 batch_size × seq_len × hidden_dim × num_layers × 4
activation_memory = batch_size * seq_len * 768 * 12 * 4
print(f"Activation memory: {activation_memory / 1e9:.2f} GB")

# 梯度
gradient_memory = model_memory
print(f"Gradient memory: {gradient_memory / 1e9:.2f} GB")

# 总计
total_memory = (model_memory + optimizer_memory +
                activation_memory + gradient_memory)
print(f"\nTotal memory: {total_memory / 1e9:.2f} GB")

# 输出示例 (124M 模型):
# Model memory: 0.50 GB
# Optimizer memory: 1.00 GB
# Activation memory: 7.50 GB
# Gradient memory: 0.50 GB
# Total memory: 9.50 GB
```

### 通信开销

```python
# DDP All-Reduce 通信量
# 每步需要同步梯度
comm_volume_per_step = model_params * 4  # 4 bytes/param (FP32)

# 通信时间 (假设 NVLink: 300 GB/s)
nvlink_bandwidth = 300e9  # bytes/s
comm_time = comm_volume_per_step / nvlink_bandwidth

print(f"Communication volume: {comm_volume_per_step / 1e9:.2f} GB")
print(f"Communication time: {comm_time * 1000:.2f} ms")

# 计算时间 vs 通信时间
compute_time = actual_time
comm_ratio = comm_time / compute_time

print(f"Communication overhead: {comm_ratio * 100:.1f}%")

# 输出示例:
# Communication volume: 0.50 GB
# Communication time: 1.67 ms
# Communication overhead: 5.2%
```

---

## 实战示例

### 示例 1: 单 GPU 训练

```bash
# 命令行
python scripts/base_train.py \
    --config base \
    --num_steps 10000 \
    --batch_size 64 \
    --eval_interval 1000 \
    --checkpoint_dir checkpoints/base_run

# 输出
Configuration: base
  Model size: 124.4M params
  Batch size: 64 (effective: 64)
  Sequence length: 512

Dataset: fineweb-10B
  Location: data/fineweb-10B
Tokenizer loaded
  Vocab size: 50257
  Type: BPE

[Single GPU] Device: cuda:0

Initializing new model
  Parameters: 124,439,808
  Non-embedding params: 85,054,464

Setting up optimizers...
  AdamW parameters: 2 (embeddings)
  Muon parameters: 84 (matrices)

Starting training from step 0
Target steps: 10000
Evaluation interval: 1000
------------------------------------------------------------
Step   100 | Loss: 6.2453 | LR (AdamW): 1.00e-04 | LR (Muon): 6.00e-04
Step   200 | Loss: 5.8721 | LR (AdamW): 2.00e-04 | LR (Muon): 1.20e-03
...
Step  1000 | Loss: 3.4567 | LR (AdamW): 1.00e-03 | LR (Muon): 6.00e-03

[Evaluation at step 1000]
  Validation loss: 3.2145
  ✓ New best model saved!

...
```

### 示例 2: 多 GPU 训练 (4x A100)

```bash
# 使用 torchrun 启动
torchrun --nproc_per_node=4 scripts/base_train.py \
    --config mid \
    --num_steps 50000 \
    --batch_size 32 \
    --grad_accum 4 \
    --eval_interval 2000 \
    --use_amp

# 输出
[Rank 0] Initialized process group
  World size: 4
  Backend: nccl
  Device: cuda:0
[Rank 1] Initialized process group
  World size: 4
  Backend: nccl
  Device: cuda:1
[Rank 2] Initialized process group
  World size: 4
  Backend: nccl
  Device: cuda:2
[Rank 3] Initialized process group
  World size: 4
  Backend: nccl
  Device: cuda:3

Configuration: mid
  Model size: 350.7M params
  Batch size: 32 (effective: 512)  ← 32 × 4 (accum) × 4 (GPUs)
  Sequence length: 1024

[Rank 0] Model wrapped in DDP

Setting up optimizers...
  AdamW parameters: 2
  Muon parameters: 148

Starting training from step 0
Target steps: 50000
Evaluation interval: 2000
------------------------------------------------------------
[Rank 0] Step   100 | Loss: 7.1234 | LR (AdamW): 1.20e-04 | LR (Muon): 7.20e-04
[Rank 0] Step   200 | Loss: 6.5432 | LR (AdamW): 2.40e-04 | LR (Muon): 1.44e-03
...
```

### 示例 3: 从检查点恢复

```bash
# 训练意外中断后恢复
python scripts/base_train.py \
    --config base \
    --checkpoint checkpoints/base_run/latest.pt \
    --num_steps 10000

# 输出
Loading checkpoint: checkpoints/base_run/latest.pt
  Restored from step 5000
  Best validation loss: 2.8765

Starting training from step 5000
Target steps: 10000
------------------------------------------------------------
Step  5100 | Loss: 2.8432 | LR (AdamW): 8.50e-04 | LR (Muon): 5.10e-03
...
```

### 示例 4: 完整流程脚本

```python
# run_training.py - 完整训练流程
import subprocess
import sys
from pathlib import Path

def run_training():
    """完整的训练流程"""

    # 1. 准备数据
    print("Step 1: Preparing data...")
    subprocess.run([
        sys.executable, '-m', 'nanochat.dataset',
        '--dataset', 'fineweb-10B',
        '--download'
    ], check=True)

    # 2. 训练分词器
    print("\nStep 2: Training tokenizer...")
    subprocess.run([
        sys.executable, 'scripts/tok_train.py',
        '--data_dir', 'data/fineweb-10B',
        '--vocab_size', '50257'
    ], check=True)

    # 3. 训练模型
    print("\nStep 3: Training model...")
    subprocess.run([
        'torchrun', '--nproc_per_node=4',
        'scripts/base_train.py',
        '--config', 'base',
        '--num_steps', '10000',
        '--batch_size', '32',
        '--grad_accum', '4',
        '--eval_interval', '1000',
        '--use_amp'
    ], check=True)

    # 4. 评估
    print("\nStep 4: Running evaluation...")
    subprocess.run([
        sys.executable, 'scripts/core_eval.py',
        '--model', 'checkpoints/base_run/best_model.pt',
        '--data', 'data/core_eval.jsonl'
    ], check=True)

    # 5. 生成报告
    print("\nStep 5: Generating report...")
    subprocess.run([
        sys.executable, '-m', 'nanochat.report',
        '--checkpoint_dir', 'checkpoints/base_run',
        '--output', 'results/training_report.md'
    ], check=True)

    print("\n✓ Training complete!")
    print("  Model: checkpoints/base_run/best_model.pt")
    print("  Report: results/training_report.md")

if __name__ == '__main__':
    run_training()
```

---

## 总结

`base_train.py` 是 nanochat 的**核心训练流程**，它整合了：

### 核心组件

1. **分布式训练** (DDP + ZeRO-2 风格)
2. **混合优化器** (AdamW + Muon)
3. **梯度累积** (实现大批量训练)
4. **混合精度** (AMP: FP16/FP32)
5. **学习率调度** (Warmup + Cosine Decay)
6. **动态评估** (Validation + CORE)
7. **检查点管理** (自动保存最佳模型)

### 数据流

```
数据 → 分词 → DataLoader → 模型
                ↓
              DDP
                ↓
           梯度累积
                ↓
           梯度裁剪
                ↓
         优化器更新
                ↓
          学习率调度
                ↓
              评估
                ↓
            检查点
```

### 关键技巧

1. **有效批大小** = `local_batch × grad_accum × world_size`
2. **梯度裁剪**: 防止训练不稳定
3. **混合精度**: 2-3x 加速，内存减半
4. **ZeRO 优化**: 节省 60%+ 优化器内存
5. **动态评估**: 自动保存最佳模型

### 下一步

阅读以下文档深入了解各个组件：

- `04_optimizers_adamw_muon.md` - 优化器详解
- `06_inference_engine.md` - 推理引擎
- `07_gpt_model.md` - GPT 模型架构

---

**Happy Training! 🚀**
