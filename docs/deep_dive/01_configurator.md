# Deep Dive: configurator.py - 配置系统

> **文件**: `nanochat/configurator.py` (57 行)
> **作用**: 极简配置系统，支持配置文件和命令行参数覆盖
> **设计哲学**: "Poor Man's Configurator" - 避免复杂的配置对象

---

## 📋 目录

- [概述](#概述)
- [设计理念](#设计理念)
- [工作原理](#工作原理)
- [使用方式](#使用方式)
- [源码详解](#源码详解)
- [优缺点分析](#优缺点分析)
- [实际应用示例](#实际应用示例)

---

## 概述

`configurator.py` 是 nanochat 项目的配置系统核心。它采用了一种**非常规但极简**的设计：

- ❌ **不是** Python 模块
- ❌ **不是** 配置类或对象
- ✅ **而是** 直接修改调用脚本的全局变量

### 核心功能

```python
# 支持两种配置方式：

# 1. 配置文件
python train.py config/my_config.py

# 2. 命令行参数
python train.py --batch_size=32 --learning_rate=0.001

# 3. 组合使用
python train.py config/my_config.py --batch_size=32
```

---

## 设计理念

### Karpathy 的原话

```python
"""
Poor Man's Configurator. Probably a terrible idea.

I just really dislike configuration complexity and having to
prepend config. to every single variable. If someone comes up
with a better simple Python solution I am all ears.
"""
```

### 为什么这样设计？

#### ❌ 传统配置方式的问题

```python
# 方式 1: 配置对象（太啰嗦）
config = Config()
batch_size = config.batch_size
learning_rate = config.learning_rate
device_batch_size = config.device_batch_size
# 每次都要加 config. 前缀！

# 方式 2: argparse（太复杂）
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
# ... 需要声明每个参数
args = parser.parse_args()

# 方式 3: 配置文件库（太重）
import yaml/json/toml
config = yaml.load(...)
# 又要学新的语法
```

#### ✅ Nanochat 的方式（极简）

```python
# 在训练脚本中直接定义全局变量
batch_size = 32
learning_rate = 0.001
device_batch_size = 16

# 导入 configurator
exec(open('nanochat/configurator.py').read())

# 现在直接使用变量，无需前缀！
print(f"Batch size: {batch_size}")
model = Model(learning_rate=learning_rate)
```

---

## 工作原理

### 核心机制：修改 `globals()`

```python
# configurator.py 通过修改调用脚本的 globals() 来工作

# 步骤 1: 训练脚本定义默认值
# train.py:
batch_size = 32
learning_rate = 0.001

# 步骤 2: 执行 configurator
exec(open('nanochat/configurator.py').read())

# 步骤 3: configurator 修改 globals()
globals()['batch_size'] = 64  # 覆盖！

# 步骤 4: 训练脚本继续使用变量
# 现在 batch_size = 64（已被覆盖）
```

### 执行流程图

```
┌─────────────────────────────────────────────────────────┐
│ 1. 训练脚本开始                                          │
│    batch_size = 32 (默认值)                              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 执行 configurator.py                                  │
│    exec(open('configurator.py').read())                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 解析命令行参数                                        │
│    sys.argv = ['train.py', 'config.py', '--batch_size=64']│
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 执行配置文件 (如果有)                                 │
│    exec(open('config.py').read())                        │
│    → 修改 globals()                                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 5. 覆盖命令行参数                                        │
│    globals()['batch_size'] = 64                          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 6. 训练脚本继续                                          │
│    print(batch_size)  # 输出: 64                         │
└─────────────────────────────────────────────────────────┘
```

---

## 使用方式

### 方式 1: 仅使用默认值

```python
# train.py
batch_size = 32
learning_rate = 0.001

exec(open('nanochat/configurator.py').read())

print(f"Batch size: {batch_size}")  # 输出: 32
```

```bash
python train.py
# 使用所有默认值
```

### 方式 2: 命令行参数覆盖

```bash
python train.py --batch_size=64 --learning_rate=0.01
```

**发生了什么:**
1. `batch_size` 从 32 → 64
2. `learning_rate` 从 0.001 → 0.01
3. 类型自动检查（必须匹配）

### 方式 3: 配置文件

```python
# config/large_model.py
batch_size = 128
learning_rate = 0.02
max_iters = 10000
```

```bash
python train.py config/large_model.py
```

**发生了什么:**
1. 执行 `config/large_model.py`
2. 其中的赋值语句修改 `globals()`
3. 所有变量被覆盖

### 方式 4: 组合使用（推荐）

```bash
# 先加载配置文件，再用命令行微调
python train.py config/large_model.py --batch_size=256

# 执行顺序:
# 1. 默认值: batch_size = 32
# 2. 配置文件: batch_size = 128
# 3. 命令行: batch_size = 256 (最终值)
```

---

## 源码详解

### 第 1 部分：`print0` 工具函数

```python
def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)
```

**作用**: 分布式训练时只让 rank 0 进程打印

**为什么需要?**
```python
# 多 GPU 训练时，每个 GPU 都会运行相同代码
# 如果都打印，会有 8 份相同输出！

# ❌ 没有 print0:
# [GPU 0] Batch size: 32
# [GPU 1] Batch size: 32
# [GPU 2] Batch size: 32
# ... (8 行重复)

# ✅ 使用 print0:
# [GPU 0] Batch size: 32
# (其他 GPU 不打印)
```

**实现细节:**
- `RANK` 环境变量由 `torchrun` 设置
- Rank 0 是"主"进程，负责打印和日志
- 其他进程跳过打印

---

### 第 2 部分：参数解析循环

```python
for arg in sys.argv[1:]:  # 跳过脚本名本身
```

**`sys.argv` 是什么?**

```python
# 命令: python train.py config.py --batch_size=32
# sys.argv = ['train.py', 'config.py', '--batch_size=32']
# sys.argv[1:] = ['config.py', '--batch_size=32']
```

---

### 第 3 部分：处理配置文件

```python
if '=' not in arg:
    # 没有 '=' → 是配置文件
    assert not arg.startswith('--')
    config_file = arg
    print0(f"Overriding config with {config_file}:")
    with open(config_file) as f:
        print0(f.read())  # 打印配置文件内容
    exec(open(config_file).read())  # 执行配置文件！
```

**关键点:**

#### 1. 判断是配置文件
```python
if '=' not in arg:
    # 'config.py' → 配置文件
    # '--batch_size=32' → 命令行参数（有 '='）
```

#### 2. 安全检查
```python
assert not arg.startswith('--')
# 确保配置文件不以 -- 开头
# 避免混淆
```

#### 3. 打印配置内容
```python
with open(config_file) as f:
    print0(f.read())
# 让用户看到将要应用的配置
```

#### 4. **核心：执行配置文件**
```python
exec(open(config_file).read())
```

**这一行非常关键！**

```python
# 假设 config.py 内容:
# batch_size = 128
# learning_rate = 0.02

# exec() 会执行这些语句
# 等价于:
batch_size = 128      # 直接在当前作用域执行！
learning_rate = 0.02

# 因为 exec() 在 configurator.py 中运行
# 而 configurator.py 通过 exec() 在 train.py 的作用域运行
# 所以最终修改了 train.py 的 globals()
```

---

### 第 4 部分：处理命令行参数

```python
else:
    # 有 '=' → 是命令行参数
    assert arg.startswith('--')
    key, val = arg.split('=')
    key = key[2:]  # 去掉 '--'
```

**示例:**
```python
# arg = '--batch_size=32'
# key = 'batch_size'
# val = '32'
```

#### 步骤 4.1: 类型转换

```python
if key in globals():
    try:
        # 尝试智能转换类型
        attempt = literal_eval(val)
    except (SyntaxError, ValueError):
        # 转换失败，当作字符串
        attempt = val
```

**`literal_eval` 的作用:**

```python
from ast import literal_eval

# 自动识别类型
literal_eval('32')     # → 32 (int)
literal_eval('0.001')  # → 0.001 (float)
literal_eval('True')   # → True (bool)
literal_eval('[1,2]')  # → [1, 2] (list)
literal_eval('hello')  # → SyntaxError (当作字符串)
```

**为什么不用 `eval()`?**

```python
# eval() 可以执行任意代码，不安全！
eval('__import__("os").system("rm -rf /")')  # 💀 危险！

# literal_eval() 只解析字面值，安全
literal_eval('__import__("os")')  # ✓ ValueError
```

#### 步骤 4.2: 类型检查

```python
if globals()[key] is not None:
    attempt_type = type(attempt)
    default_type = type(globals()[key])
    assert attempt_type == default_type, \
        f"Type mismatch: {attempt_type} != {default_type}"
```

**作用**: 确保类型匹配

**示例:**
```python
# train.py 中定义:
batch_size = 32  # int

# 命令行:
# ✓ --batch_size=64  → 64 (int) → OK
# ✗ --batch_size=hello → 'hello' (str) → AssertionError!
```

**为什么要检查?**

```python
# 防止意外错误
batch_size = 32

# 如果不检查，可能:
batch_size = '64'  # 字符串！
data = data[:batch_size]  # TypeError: slice indices must be integers
```

#### 步骤 4.3: 覆盖值

```python
print0(f"Overriding: {key} = {attempt}")
globals()[key] = attempt
```

**实际效果:**

```python
# 训练脚本中:
batch_size = 32

# configurator 执行后:
# globals()['batch_size'] = 64

# 训练脚本继续时:
print(batch_size)  # 输出: 64 (已被修改！)
```

#### 步骤 4.4: 错误处理

```python
else:
    raise ValueError(f"Unknown config key: {key}")
```

**作用**: 防止拼写错误

```python
# train.py 定义:
batch_size = 32

# ✓ python train.py --batch_size=64    → OK
# ✗ python train.py --batch_szie=64    → ValueError!
#                           ↑ 拼写错误
```

---

## 优缺点分析

### ✅ 优点

#### 1. **极简**
```python
# 只需 57 行代码
# 无需外部依赖
# 无需学习新语法
```

#### 2. **零前缀**
```python
# ✅ Nanochat:
print(f"Batch: {batch_size}")

# ❌ 传统方式:
print(f"Batch: {config.batch_size}")
#                 ^^^^^^^ 烦人的前缀
```

#### 3. **类型安全**
```python
# 自动类型检查
# 防止类型错误
batch_size = 32
# --batch_size=hello → AssertionError
```

#### 4. **灵活组合**
```python
# 配置文件 + 命令行
python train.py config.py --batch_size=64

# 覆盖顺序清晰：
# 默认值 → 配置文件 → 命令行
```

#### 5. **调试友好**
```python
# 自动打印所有覆盖
# Overriding: batch_size = 64
# Overriding: learning_rate = 0.01
```

---

### ❌ 缺点（Karpathy 承认的）

#### 1. **"Probably a terrible idea"**

**为什么?**

```python
# 使用 exec() 和修改 globals()
# 这在 Python 社区被认为是"黑魔法"
# 不符合常规最佳实践
```

#### 2. **不是真正的 Python 模块**

```python
# ❌ 不能这样导入:
from nanochat import configurator

# ✓ 必须这样执行:
exec(open('nanochat/configurator.py').read())
```

#### 3. **IDE 支持差**

```python
# IDE 可能不知道变量被修改了
batch_size = 32
exec(open('configurator.py').read())
# IDE 仍然认为 batch_size = 32
# 但实际可能是 64
```

#### 4. **全局变量污染**

```python
# 所有配置都是全局变量
# 可能与其他代码冲突
batch_size = 32  # 我的变量
exec(...)
# batch_size 可能被意外修改
```

#### 5. **不适合复杂配置**

```python
# 只适合简单的 key=value
# 不适合嵌套配置:
model:
  layers: 12
  hidden: 768
```

---

### 🤔 为什么 Karpathy 仍然用它？

```python
"""
I just really dislike configuration complexity
"""
```

**权衡取舍:**
- ✅ **简单性** > 最佳实践
- ✅ **易读性** > IDE 支持
- ✅ **快速迭代** > 严格工程

**适用场景:**
- 研究代码（不是生产系统）
- 个人项目
- 教育用途
- 快速原型

---

## 实际应用示例

### 示例 1: 基础训练脚本

```python
# train.py

# 1. 定义默认配置
batch_size = 32
learning_rate = 0.001
max_iters = 1000
device = 'cuda'

# 2. 应用配置系统
exec(open('nanochat/configurator.py').read())

# 3. 使用配置（无前缀！）
print(f"Training with batch_size={batch_size}, lr={learning_rate}")

for i in range(max_iters):
    # 训练逻辑
    pass
```

**运行:**
```bash
# 使用默认值
python train.py

# 覆盖部分参数
python train.py --batch_size=64 --learning_rate=0.01

# 使用配置文件
python train.py config/fast_train.py
```

---

### 示例 2: 配置文件

```python
# config/large_model.py

# 简单的 Python 赋值
batch_size = 128
learning_rate = 0.02
max_iters = 5000

# 可以有逻辑
if device == 'cuda':
    batch_size *= 2  # GPU 可以用更大 batch

# 可以有注释
# This config is for large models
```

**使用:**
```bash
python train.py config/large_model.py
```

---

### 示例 3: Nanochat 实际使用

```python
# scripts/base_train.py

# 定义所有超参数
depth = 20                     # 模型深度
max_seq_len = 2048             # 序列长度
device_batch_size = 32         # 每 GPU 批量
total_batch_size = 524288      # 总批量
embedding_lr = 0.2             # 嵌入层学习率
matrix_lr = 0.02               # 矩阵层学习率
run = "dummy"                  # Wandb 运行名

# 应用配置
exec(open('nanochat/configurator.py').read())

# 现在可以直接使用这些变量
model = GPT(GPTConfig(n_layer=depth, sequence_len=max_seq_len))
optimizer = Muon(model.parameters(), lr=matrix_lr)
```

**运行:**
```bash
# d20 模型
torchrun -m scripts.base_train -- --depth=20

# d26 模型（需要调整 batch size）
torchrun -m scripts.base_train -- --depth=26 --device_batch_size=16

# 带 wandb
torchrun -m scripts.base_train -- --depth=20 --run=my_experiment
```

---

## 核心洞察

### 1. **设计权衡**

```
简单性 vs 最佳实践
│
├─ 传统方式: 遵循最佳实践，但复杂
│  └─ argparse, dataclass, OOP 等
│
└─ Nanochat: 打破常规，追求简单
   └─ exec(), globals(), 全局变量
```

### 2. **适用场景**

**✅ 适合:**
- 研究代码
- 快速原型
- 教育项目
- 个人项目

**❌ 不适合:**
- 生产系统
- 大型团队
- 复杂配置
- 需要严格类型系统的项目

### 3. **Karpathy 的哲学**

```python
"""
Clarity matters more than convention.
Simplicity is worth the cost of unconventionality.
"""
```

---

## 总结

### 核心机制
```python
1. 训练脚本定义全局变量（默认值）
2. 执行 configurator.py
3. configurator 修改 globals()
   - 先执行配置文件（如果有）
   - 再应用命令行参数
4. 训练脚本继续，使用修改后的值
```

### 关键特点
- ✅ 极简（57 行）
- ✅ 零前缀
- ✅ 类型安全
- ✅ 灵活组合
- ⚠️ 非常规设计

### 学习要点
- **理解**: `exec()` 和 `globals()` 的工作原理
- **欣赏**: 简单性优于复杂性的设计理念
- **权衡**: 知道何时使用/不使用这种方式

---

**下一篇**: [common.py - 通用工具函数详解](02_common.md)

---

## 附录：完整示例

```python
# example_train.py

# ========== 配置部分 ==========
# 模型
depth = 12
hidden_dim = 768
n_heads = 12

# 训练
batch_size = 32
learning_rate = 0.001
max_iters = 1000
warmup_iters = 100

# 系统
device = 'cuda'
dtype = 'bfloat16'
seed = 42

# ========== 应用配置 ==========
exec(open('nanochat/configurator.py').read())

# ========== 训练代码 ==========
print(f"Training with:")
print(f"  Model: depth={depth}, hidden={hidden_dim}")
print(f"  Batch: {batch_size}, LR: {learning_rate}")

# ... 训练逻辑 ...
```

**运行示例:**
```bash
# 1. 默认配置
python example_train.py

# 2. 调整 batch size
python example_train.py --batch_size=64

# 3. 使用配置文件
python example_train.py config/small.py

# 4. 组合
python example_train.py config/small.py --learning_rate=0.01
```

---

**思考题:**

1. 为什么 `literal_eval` 比 `eval` 更安全？
2. 配置文件执行顺序为什么重要？
3. 如果你要设计配置系统，会选择这种方式吗？为什么？

**下一步**: 阅读 `common.py`，了解 nanochat 的通用工具函数。
