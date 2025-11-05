# Deep Dive: common.py - 通用工具函数

> **文件**: `nanochat/common.py` (196 行)
> **作用**: 提供分布式训练、日志、文件下载等通用工具
> **核心功能**: DDP 支持、彩色日志、并发安全的文件下载

---

## 📋 目录

- [概述](#概述)
- [模块结构](#模块结构)
- [彩色日志系统](#彩色日志系统)
- [目录管理](#目录管理)
- [并发安全的文件下载](#并发安全的文件下载)
- [分布式训练工具](#分布式训练工具)
- [设备管理](#设备管理)
- [计算环境初始化](#计算环境初始化)
- [Wandb 集成](#wandb-集成)
- [实战示例](#实战示例)

---

## 概述

`common.py` 是 nanochat 的**工具箱**，提供了训练过程中需要的各种基础设施：

### 核心功能

```python
📊 日志系统
├─ ColoredFormatter: 彩色日志
└─ setup_default_logging(): 默认日志配置

📁 文件管理
├─ get_base_dir(): 获取缓存目录
└─ download_file_with_lock(): 并发安全下载

🖥️ 分布式训练
├─ is_ddp(): 检测 DDP 环境
├─ get_dist_info(): 获取 rank 信息
├─ print0(): 只在 rank 0 打印
└─ compute_init(): 初始化 DDP

🔧 设备管理
├─ autodetect_device_type(): 自动检测设备
└─ compute_cleanup(): 清理资源

📈 实验追踪
└─ DummyWandb: Wandb 替代品
```

---

## 模块结构

### 依赖关系

```python
import os                # 环境变量、文件操作
import re                # 正则表达式（日志高亮）
import logging           # 日志系统
import fcntl             # 文件锁（Linux）
import urllib.request    # 文件下载
import torch             # PyTorch
import torch.distributed as dist  # 分布式训练
```

### 模块初始化

```python
# 文件加载时自动执行
setup_default_logging()  # 配置日志
logger = logging.getLogger(__name__)  # 创建 logger
```

**这意味着**：只要导入 `common.py`，日志系统就已经配置好了！

---

## 彩色日志系统

### `ColoredFormatter` 类

#### 设计目的

让终端输出更易读：

```python
# ❌ 没有颜色:
2025-01-15 10:30:45 - nanochat - INFO - Training loss: 3.456
2025-01-15 10:30:46 - nanochat - WARNING - High memory usage: 78.5%
2025-01-15 10:30:47 - nanochat - ERROR - CUDA out of memory

# ✅ 有颜色:
2025-01-15 10:30:45 - nanochat - INFO - Training loss: 3.456     (绿色加粗 INFO)
2025-01-15 10:30:46 - nanochat - WARNING - High memory usage: 78.5%  (黄色加粗 WARNING)
2025-01-15 10:30:47 - nanochat - ERROR - CUDA out of memory     (红色加粗 ERROR)
```

#### ANSI 颜色码

```python
class ColoredFormatter(logging.Formatter):
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan (青色)
        'INFO': '\033[32m',     # Green (绿色)
        'WARNING': '\033[33m',  # Yellow (黄色)
        'ERROR': '\033[31m',    # Red (红色)
        'CRITICAL': '\033[35m', # Magenta (品红)
    }
    RESET = '\033[0m'   # 重置颜色
    BOLD = '\033[1m'    # 加粗
```

**ANSI 转义序列原理:**

```python
# 格式: \033[<code>m
# 033 是 ESC 字符的八进制
# [<code>m 是颜色/样式代码

print('\033[31m红色文本\033[0m')  # 红色
print('\033[32m绿色文本\033[0m')  # 绿色
print('\033[1m加粗文本\033[0m')   # 加粗
print('\033[1m\033[31m红色加粗\033[0m')  # 红色 + 加粗
```

#### `format()` 方法

```python
def format(self, record):
    # 步骤 1: 给日志级别添加颜色
    levelname = record.levelname
    if levelname in self.COLORS:
        record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

    # 步骤 2: 格式化消息
    message = super().format(record)

    # 步骤 3: 高亮 INFO 消息中的特殊内容
    if levelname == 'INFO':
        # 高亮数字和单位
        message = re.sub(
            r'(\d+\.?\d*\s*(?:GB|MB|%|docs))',
            rf'{self.BOLD}\1{self.RESET}',
            message
        )
        # 高亮 "Shard X"
        message = re.sub(
            r'(Shard \d+)',
            rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}',
            message
        )

    return message
```

**工作流程:**

```python
# 原始日志:
logger.info("Downloaded 250 MB from Shard 5")

# 步骤 1: 级别着色
# "INFO" → 绿色加粗 "INFO"

# 步骤 2: 格式化
# "2025-01-15 - nanochat - INFO - Downloaded 250 MB from Shard 5"

# 步骤 3: 内容高亮
# "250 MB" → 加粗
# "Shard 5" → 绿色加粗

# 最终输出: (想象成彩色)
# 2025-01-15 - nanochat - [绿色加粗]INFO[/] - Downloaded [加粗]250 MB[/] from [绿色加粗]Shard 5[/]
```

#### 正则表达式解析

```python
# 匹配数字和单位
r'(\d+\.?\d*\s*(?:GB|MB|%|docs))'

# 分解:
\d+          # 一个或多个数字
\.?          # 可选的小数点
\d*          # 零个或多个数字（小数部分）
\s*          # 可选的空格
(?:GB|MB|%|docs)  # 非捕获组：GB 或 MB 或 % 或 docs

# 匹配示例:
"250 MB"     ✓
"3.5GB"      ✓
"78.5%"      ✓
"1000 docs"  ✓
```

```python
# 匹配 "Shard X"
r'(Shard \d+)'

# 匹配示例:
"Shard 5"    ✓
"Shard 123"  ✓
```

### `setup_default_logging()`

```python
def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )
```

**配置内容:**

```python
1. StreamHandler: 输出到终端（stdout）
2. ColoredFormatter: 使用自定义格式化器
3. 日志格式: "时间 - 模块名 - 级别 - 消息"
4. 日志级别: INFO（不显示 DEBUG）
```

**使用示例:**

```python
from nanochat.common import logger

logger.debug("这条不会显示")     # DEBUG < INFO
logger.info("开始训练")          # ✓ 绿色
logger.warning("内存使用率高")    # ✓ 黄色
logger.error("CUDA OOM")        # ✓ 红色
```

---

## 目录管理

### `get_base_dir()`

#### 作用

确定 nanochat 存储所有中间文件的位置。

#### 目录结构

```bash
~/.cache/nanochat/               # 默认基础目录
├── data/                        # 数据集分片
│   ├── shard_0.parquet
│   ├── shard_1.parquet
│   └── ...
├── tokenizer/                   # 分词器
│   ├── tokenizer.pkl
│   └── token_bytes.pt
├── checkpoints/                 # 模型检查点
│   ├── base/
│   ├── mid/
│   └── sft/
└── report/                      # 训练报告
    └── report.md
```

#### 源码解析

```python
def get_base_dir():
    # 优先使用环境变量
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        # 默认使用 ~/.cache/nanochat
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")

    # 确保目录存在
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir
```

**关键点:**

1. **环境变量优先**
   ```bash
   export NANOCHAT_BASE_DIR="/mnt/ssd/nanochat"
   # 现在会使用 /mnt/ssd/nanochat 而非 ~/.cache/nanochat
   ```

2. **自动创建目录**
   ```python
   os.makedirs(nanochat_dir, exist_ok=True)
   # exist_ok=True: 目录存在也不报错
   ```

3. **跨平台支持**
   ```python
   os.path.expanduser("~")
   # Linux/Mac: /home/username
   # Windows: C:\Users\username
   ```

#### 使用场景

```python
from nanochat.common import get_base_dir

# 保存模型
base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "checkpoints", "base")
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))

# 下载数据
data_dir = os.path.join(base_dir, "data")
shard_path = os.path.join(data_dir, f"shard_{i}.parquet")
```

---

## 并发安全的文件下载

### `download_file_with_lock()`

#### 问题背景

**多进程训练时的并发问题:**

```python
# 8 个 GPU，8 个进程同时运行
# 都需要下载同一个文件

# ❌ 没有锁:
GPU 0: 开始下载 data.parquet...
GPU 1: 开始下载 data.parquet...  # 重复下载！
GPU 2: 开始下载 data.parquet...  # 重复下载！
...
# 结果: 8 次重复下载，浪费带宽和时间

# ✅ 有锁:
GPU 0: 获取锁 → 开始下载 data.parquet...
GPU 1: 等待锁...
GPU 2: 等待锁...
...
GPU 0: 下载完成 → 释放锁
GPU 1: 获取锁 → 检查文件 → 已存在 → 返回
GPU 2: 获取锁 → 检查文件 → 已存在 → 返回
# 结果: 只下载 1 次！
```

#### 文件锁原理

```python
import fcntl  # File Control

# 排他锁（Exclusive Lock）
fcntl.flock(file_descriptor, fcntl.LOCK_EX)
# LOCK_EX: Exclusive lock
# 同一时间只有一个进程能持有锁
# 其他进程会阻塞（block）直到锁被释放
```

**操作系统级别的锁:**

```
进程 A                进程 B                进程 C
  |                     |                     |
  V                     V                     V
flock(LOCK_EX)      flock(LOCK_EX)      flock(LOCK_EX)
  |                     |                     |
获得锁 ✓               阻塞...              阻塞...
  |                     |                     |
下载文件                等待                 等待
  |                     |                     |
释放锁                  |                     |
  X                     |                     |
                    获得锁 ✓                 阻塞...
                        |                     |
                    文件已存在 → 返回          等待
                        X                     |
                                          获得锁 ✓
                                              |
                                          文件已存在 → 返回
                                              X
```

#### 源码详解

```python
def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    url: 下载地址
    filename: 保存的文件名（相对于 base_dir）
    postprocess_fn: 下载后的处理函数（可选）
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"  # 锁文件路径
```

**步骤 1: 快速检查（无锁）**

```python
    if os.path.exists(file_path):
        return file_path
    # 如果文件已存在，直接返回
    # 避免不必要的锁操作
```

**为什么需要这个检查?**

```python
# 第二次运行训练时
# 文件已经存在
# 如果没有这个检查，每个进程都要获取锁
# 即使只是为了检查文件存在
```

**步骤 2: 获取文件锁**

```python
    with open(lock_path, 'w') as lock_file:
        # 获取排他锁（阻塞直到获得）
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
```

**`with` 语句的作用:**

```python
# 等价于:
lock_file = open(lock_path, 'w')
try:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    # ... 下载代码 ...
finally:
    lock_file.close()  # 自动释放锁！
```

**步骤 3: 双重检查（Double-Check）**

```python
        # 重新检查文件是否存在
        if os.path.exists(file_path):
            return file_path
```

**为什么需要双重检查?**

```python
# 时间线:
# t0: 进程 A 通过步骤 1（文件不存在）
# t1: 进程 B 通过步骤 1（文件不存在）
# t2: 进程 A 获取锁，下载文件
# t3: 进程 A 释放锁
# t4: 进程 B 获取锁，这时文件已存在！
#     如果没有这个检查，进程 B 会重复下载
```

**步骤 4: 下载文件**

```python
        # 下载内容
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()  # bytes

        # 写入本地文件
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")
```

**步骤 5: 后处理（可选）**

```python
        # 运行后处理函数
        if postprocess_fn is not None:
            postprocess_fn(file_path)
```

**使用示例:**

```python
def extract_and_validate(file_path):
    """解压并验证文件"""
    print(f"Extracting {file_path}...")
    # 解压逻辑...
    print(f"Validating {file_path}...")
    # 验证逻辑...

# 下载并自动处理
download_file_with_lock(
    "https://example.com/data.tar.gz",
    "data/data.tar.gz",
    postprocess_fn=extract_and_validate
)
```

**步骤 6: 清理锁文件**

```python
    # with 块结束，锁自动释放

    # 清理锁文件
    try:
        os.remove(lock_path)
    except OSError:
        pass  # 如果已被其他进程删除，忽略错误

    return file_path
```

**为什么用 try-except?**

```python
# 可能的情况:
# 进程 A 删除了 lock_path
# 进程 B 也尝试删除 lock_path
# → FileNotFoundError
# 但这不是问题，可以忽略
```

#### 完整执行流程

```python
# 8 个进程并发下载

┌────────────────────────────────────────────────────────┐
│ 进程 0                                                 │
├────────────────────────────────────────────────────────┤
│ 1. 检查文件 → 不存在                                   │
│ 2. 打开锁文件                                          │
│ 3. 获取 LOCK_EX → 成功 ✓                              │
│ 4. 双重检查 → 不存在                                   │
│ 5. 下载文件... (1分钟)                                 │
│ 6. 写入文件                                            │
│ 7. 后处理                                              │
│ 8. 释放锁                                              │
│ 9. 删除锁文件                                          │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ 进程 1-7                                               │
├────────────────────────────────────────────────────────┤
│ 1. 检查文件 → 不存在                                   │
│ 2. 打开锁文件                                          │
│ 3. 尝试获取 LOCK_EX → 阻塞... (等待进程 0)            │
│    ... 等待 1 分钟 ...                                 │
│ 4. 获取 LOCK_EX → 成功 ✓ (进程 0 释放后)              │
│ 5. 双重检查 → 文件已存在 ✓                            │
│ 6. 直接返回                                            │
└────────────────────────────────────────────────────────┘
```

---

## 分布式训练工具

### `print0()`

#### 作用

分布式训练时，只让 rank 0 进程打印输出。

#### 源码

```python
def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)
```

#### 为什么需要?

```python
# 8 个 GPU 训练时
# 如果每个进程都打印:

# ❌ 没有 print0:
[GPU 0] Epoch 1, loss: 3.456
[GPU 1] Epoch 1, loss: 3.456
[GPU 2] Epoch 1, loss: 3.456
[GPU 3] Epoch 1, loss: 3.456
[GPU 4] Epoch 1, loss: 3.456
[GPU 5] Epoch 1, loss: 3.456
[GPU 6] Epoch 1, loss: 3.456
[GPU 7] Epoch 1, loss: 3.456
# 8 行重复！

# ✅ 使用 print0:
[GPU 0] Epoch 1, loss: 3.456
# 只有一行
```

#### 环境变量 `RANK`

```python
# torchrun 自动设置
# export RANK=0  # 主进程
# export RANK=1  # 第 2 个进程
# export RANK=2  # 第 3 个进程
# ...

# 如果不在 DDP 环境（单 GPU）:
# RANK 不存在 → 默认 0 → 正常打印
```

### `print_banner()`

```python
def print_banner():
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)
```

**使用:**

```python
from nanochat.common import print_banner

print_banner()
# 显示酷炫的 ASCII logo
# 只在 rank 0 显示
```

### `is_ddp()`

```python
def is_ddp():
    return int(os.environ.get('RANK', -1)) != -1
```

**检查是否在分布式环境:**

```python
if is_ddp():
    print("Running in distributed mode")
else:
    print("Running in single-GPU mode")
```

### `get_dist_info()`

```python
def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
```

**返回值:**

```python
is_distributed, rank, local_rank, world_size = get_dist_info()

# 单 GPU:
False, 0, 0, 1

# 8 GPU (进程 0):
True, 0, 0, 8

# 8 GPU (进程 5):
True, 5, 5, 8
```

**术语解释:**

- **RANK**: 全局进程 ID (0 到 world_size-1)
- **LOCAL_RANK**: 单机内 GPU ID (0 到 7)
- **WORLD_SIZE**: 总进程数（= GPU 数）

---

## 设备管理

### `autodetect_device_type()`

```python
def autodetect_device_type():
    # 优先级: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type
```

**设备类型:**

- **CUDA**: NVIDIA GPU (最快)
- **MPS**: Apple Silicon GPU (M1/M2/M3)
- **CPU**: 中央处理器 (最慢，兜底)

**使用:**

```python
device_type = autodetect_device_type()
# 在 8×H100 上: "cuda"
# 在 MacBook M2 上: "mps"
# 在无 GPU 机器上: "cpu"

model = model.to(device_type)
```

---

## 计算环境初始化

### `compute_init()`

这是**最重要的函数**，初始化整个训练环境。

#### 函数签名

```python
def compute_init(device_type="cuda"):
    """
    返回: ddp, ddp_rank, ddp_local_rank, ddp_world_size, device
    """
```

#### 步骤 1: 设备验证

```python
assert device_type in ["cuda", "mps", "cpu"], "Invalid device type"

if device_type == "cuda":
    assert torch.cuda.is_available(), \
        "PyTorch not configured for CUDA but device_type is 'cuda'"

if device_type == "mps":
    assert torch.backends.mps.is_available(), \
        "PyTorch not configured for MPS but device_type is 'mps'"
```

**防止配置错误:**

```python
# 如果用户指定 device_type="cuda"
# 但 PyTorch 没有 CUDA 支持
# → 立即报错，避免后续神秘错误
```

#### 步骤 2: 可重复性

```python
torch.manual_seed(42)
if device_type == "cuda":
    torch.cuda.manual_seed(42)
```

**为什么设置种子?**

```python
# 确保实验可重复
# 相同的种子 → 相同的随机数 → 相同的初始化 → 相同的结果

# 例如:
torch.manual_seed(42)
w1 = torch.randn(10, 10)  # 权重初始化

torch.manual_seed(42)  # 重新设置
w2 = torch.randn(10, 10)  # 相同的权重！

assert torch.equal(w1, w2)  # ✓
```

#### 步骤 3: 精度设置

```python
if device_type == "cuda":
    torch.set_float32_matmul_precision("high")
```

**什么是 TF32?**

```python
# FP32 (Float32): 标准 32 位浮点
# TF32 (TensorFloat-32): NVIDIA Ampere+ 的特殊格式
# - 精度: 与 FP32 相近（10 位尾数 vs 23 位）
# - 速度: 与 FP16 相近（8x 加速）
# - 最佳平衡！

# "high" = 使用 TF32 进行矩阵乘法
# 在 H100 上可获得显著加速
```

#### 步骤 4: 分布式设置

```python
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

if ddp and device_type == "cuda":
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
else:
    device = torch.device(device_type)
```

**DDP 初始化详解:**

```python
# 1. 设置设备
device = torch.device("cuda", ddp_local_rank)
# 进程 0 → GPU 0
# 进程 5 → GPU 5

# 2. 设为默认
torch.cuda.set_device(device)
# 之后 .cuda() 会自动使用这个 GPU

# 3. 初始化进程组
dist.init_process_group(backend="nccl", device_id=device)
# backend="nccl": NVIDIA Collective Communication Library
# 最快的 GPU 间通信库

# 4. 同步所有进程
dist.barrier()
# 等待所有进程都完成初始化
# 类似线程的 barrier
```

**进程同步:**

```
进程 0: init_process_group() → barrier() → [等待...]
进程 1: init_process_group() → barrier() → [等待...]
进程 2: [慢一点...] → init_process_group() → barrier()
进程 3: init_process_group() → barrier() → [等待...]
...
所有进程到达 barrier → 继续执行 ✓
```

#### 步骤 5: 日志和返回

```python
if ddp_rank == 0:
    logger.info(f"Distributed world size: {ddp_world_size}")

return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device
```

#### 完整使用示例

```python
from nanochat.common import compute_init, compute_cleanup

# 初始化
ddp, rank, local_rank, world_size, device = compute_init(device_type="cuda")

print(f"Rank {rank}/{world_size} on device {device}")

# 创建模型
model = GPT(config).to(device)

if ddp:
    model = DDP(model, device_ids=[local_rank])

# 训练...

# 清理
compute_cleanup()
```

### `compute_cleanup()`

```python
def compute_cleanup():
    if is_ddp():
        dist.destroy_process_group()
```

**为什么需要清理?**

```python
# 释放 NCCL 资源
# 关闭进程间通信
# 防止资源泄漏

# 通常在训练脚本末尾调用:
try:
    train()
finally:
    compute_cleanup()  # 确保总是执行
```

---

## Wandb 集成

### `DummyWandb` 类

```python
class DummyWandb:
    """不想用 wandb 但保持相同接口"""
    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass  # 什么都不做

    def finish(self):
        pass  # 什么都不做
```

**使用场景:**

```python
# 训练脚本中
try:
    import wandb
    run = wandb.init(project="nanochat")
except ImportError:
    from nanochat.common import DummyWandb
    run = DummyWandb()

# 现在可以安全调用
run.log({"loss": 3.5})  # 有 wandb 时记录，没有时忽略
run.finish()
```

**优点:**

- 无需到处写 `if wandb_available:`
- 代码更简洁
- 开发时可以不装 wandb

---

## 实战示例

### 示例 1: 基础训练脚本

```python
from nanochat.common import compute_init, compute_cleanup, print0

# 初始化
ddp, rank, local_rank, world_size, device = compute_init("cuda")

print0(f"Starting training on {world_size} GPUs")

# 模型
model = GPT(config).to(device)
if ddp:
    model = DDP(model, device_ids=[local_rank])

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        # 训练代码...
        pass

    # 只在 rank 0 打印
    print0(f"Epoch {epoch} complete")

# 清理
compute_cleanup()
```

### 示例 2: 下载数据集

```python
from nanochat.common import download_file_with_lock

def decompress(file_path):
    import tarfile
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=os.path.dirname(file_path))

# 多进程安全下载
data_path = download_file_with_lock(
    url="https://example.com/dataset.tar.gz",
    filename="data/dataset.tar.gz",
    postprocess_fn=decompress
)

# 只下载一次，所有进程共享
```

### 示例 3: 彩色日志

```python
from nanochat.common import logger

# 不同级别的日志
logger.debug("详细调试信息")  # 不显示（INFO 级别）
logger.info("Downloaded 250 MB from Shard 5")  # 绿色，数字加粗
logger.warning("Memory usage at 85%")  # 黄色
logger.error("CUDA out of memory")  # 红色
logger.critical("Fatal error!")  # 品红
```

---

## 总结

### 核心功能

| 功能 | 函数 | 用途 |
|------|------|------|
| **日志** | `ColoredFormatter` | 彩色日志 |
| **目录** | `get_base_dir()` | 统一缓存目录 |
| **下载** | `download_file_with_lock()` | 并发安全下载 |
| **打印** | `print0()` | DDP 环境打印 |
| **分布式** | `get_dist_info()` | 获取 rank 信息 |
| **初始化** | `compute_init()` | 完整环境初始化 |
| **清理** | `compute_cleanup()` | 资源清理 |

### 设计亮点

1. **文件锁机制** - 防止多进程重复下载
2. **双重检查** - 最小化锁竞争
3. **彩色日志** - 提升终端输出可读性
4. **DDP 友好** - 处理多进程打印
5. **设备自动检测** - 支持 CUDA/MPS/CPU
6. **TF32 加速** - H100 上的性能优化

### 关键洞察

```python
# common.py 是 nanochat 的"胶水"
# 连接各个组件，处理底层细节
# 让上层代码更简洁

# 典型模式:
from nanochat.common import compute_init, print0

# 一行初始化所有环境
ddp, rank, local_rank, world_size, device = compute_init()

# 安全打印
print0("Training started")
```

---

**下一篇**: [dataset.py - 数据集下载与管理](03_dataset.md)

---

## 思考题

1. 为什么需要双重检查（double-check）模式？
2. `fcntl.flock` 是进程级锁还是线程级锁？
3. 如何在不支持 `fcntl` 的 Windows 上实现文件锁？
4. TF32 会损失精度吗？什么时候不应该用？
