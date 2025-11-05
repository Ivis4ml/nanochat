# Deep Dive: dataset.py - 数据集下载与管理

> **文件**: `nanochat/dataset.py` (128 行)
> **作用**: 管理 FineWeb-Edu-100B 数据集的下载和迭代
> **核心功能**: 按需下载、重试机制、分布式友好的数据迭代

---

## 📋 目录

- [概述](#概述)
- [数据集介绍](#数据集介绍)
- [全局配置](#全局配置)
- [核心函数](#核心函数)
- [下载机制](#下载机制)
- [数据迭代](#数据迭代)
- [命令行接口](#命令行接口)
- [实战示例](#实战示例)

---

## 概述

`dataset.py` 是 nanochat 的**数据源管理器**，负责：

### 核心功能

```python
📦 数据集管理
├─ FineWeb-Edu-100B: 1823 个 Parquet 分片
├─ 按需下载: 需要时才下载
└─ 自动重试: 网络失败时重试

🔄 数据迭代
├─ 流式加载: 不需要全部加载到内存
├─ 分布式友好: 支持 DDP 的数据分割
└─ Train/Val 分割: 最后一个分片是验证集

⚙️ 工具函数
├─ list_parquet_files(): 列出所有分片
├─ parquets_iter_batched(): 批量迭代器
└─ download_single_file(): 下载单个文件
```

---

## 数据集介绍

### FineWeb-Edu-100B

**官方链接**: https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle

#### 数据集规格

```python
名称: FineWeb-Edu-100B (shuffled version)
来源: HuggingFace CommonCrawl 教育内容
总大小: ~100 billion tokens
分片数: 1823 个 Parquet 文件
每个分片: ~100 MB (压缩后)
          ~250 million 字符 (解压后)
格式: Apache Parquet
```

#### 为什么选择这个数据集？

```python
✅ 高质量: 经过教育内容过滤
✅ 大规模: 足够训练 GPT-2 级别模型
✅ 开源: 免费可用
✅ 已打乱: 随机顺序，训练更好
✅ Parquet 格式: 高效存储和读取
```

#### Parquet 文件格式

```python
# Parquet 是列式存储格式
# 优点:
1. 压缩率高 (比 JSON/CSV 小 5-10x)
2. 读取快速 (只读需要的列)
3. 支持复杂类型
4. 内置元数据

# 结构:
Parquet 文件
├─ Row Group 1 (1000 行)
│  └─ Column 'text': ["doc1", "doc2", ...]
├─ Row Group 2 (1000 行)
│  └─ Column 'text': ["doc1001", "doc1002", ...]
└─ Row Group N
```

**为什么用 Row Groups?**

```python
# 不用 Row Groups:
# 需要读取整个文件到内存

# 使用 Row Groups:
# 可以逐个 Row Group 读取
# 内存友好！

# 例如:
pf = pq.ParquetFile("shard_00000.parquet")
for rg_idx in range(pf.num_row_groups):
    rg = pf.read_row_group(rg_idx)  # 只读一个 Row Group
    process(rg)  # 处理完就释放内存
```

---

## 全局配置

### 数据集配置

```python
# URL base
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"

# 最大分片索引
MAX_SHARD = 1822  # 共 1823 个文件 (0-1822)

# 文件命名规则
index_to_filename = lambda index: f"shard_{index:05d}.parquet"

# 示例:
index_to_filename(0)     # → "shard_00000.parquet"
index_to_filename(5)     # → "shard_00005.parquet"
index_to_filename(1822)  # → "shard_01822.parquet"
```

**为什么用 5 位数字?**

```python
# 最大索引: 1822
# 需要至少 4 位: 0000-1822
# 使用 5 位留余量: 00000-01822

# 好处:
1. 字符串排序正确
   "shard_00005.parquet" < "shard_00010.parquet"  # ✓
   vs
   "shard_5.parquet" > "shard_10.parquet"  # ✗ (字符串排序错误)

2. 文件名对齐整齐
   shard_00000.parquet
   shard_00001.parquet
   shard_00002.parquet
   ...
   shard_01822.parquet
```

### 目录配置

```python
base_dir = get_base_dir()  # ~/.cache/nanochat 或自定义
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# 目录结构:
~/.cache/nanochat/
└── base_data/              # 数据集目录
    ├── shard_00000.parquet
    ├── shard_00001.parquet
    ├── ...
    └── shard_01822.parquet
```

**为什么不用 `data/` 而用 `base_data/`?**

```python
# nanochat 可能有多个数据集:
base_data/      # 预训练数据
mid_data/       # 中间训练数据（未来）
sft_data/       # SFT 数据（未来）

# 明确命名，避免混淆
```

---

## 核心函数

### 1. `list_parquet_files()`

#### 函数签名

```python
def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
```

#### 源码解析

```python
def list_parquet_files(data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir

    # 列出所有 .parquet 文件
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])

    # 转换为完整路径
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths
```

**关键点:**

1. **过滤 `.tmp` 文件**
   ```python
   not f.endswith('.tmp')
   # 下载中的临时文件不算
   ```

2. **排序**
   ```python
   sorted([...])
   # 确保顺序: shard_00000, shard_00001, ...
   # 重要！训练时需要按顺序遍历
   ```

3. **完整路径**
   ```python
   os.path.join(data_dir, f)
   # 返回绝对路径，方便其他函数使用
   ```

#### 使用示例

```python
from nanochat.dataset import list_parquet_files

# 获取所有分片
files = list_parquet_files()
print(f"Found {len(files)} shards")

# 示例输出:
# Found 240 shards
# ['/home/user/.cache/nanochat/base_data/shard_00000.parquet',
#  '/home/user/.cache/nanochat/base_data/shard_00001.parquet',
#  ...]
```

---

### 2. `parquets_iter_batched()`

#### 函数签名

```python
def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups.
    - split: "train" or "val"
    - start/step: for DDP (e.g. start=rank, step=world_size)
    """
```

#### 为什么需要这个函数？

**问题**: Parquet 文件很大，不能一次性全部加载

```python
# ❌ 不好的方式:
all_data = []
for file in files:
    all_data.extend(read_entire_file(file))  # 占用大量内存！

# ✅ 好的方式:
for batch in parquets_iter_batched("train"):
    process(batch)  # 处理一批就释放
```

#### 源码解析

**步骤 1: 验证参数**

```python
assert split in ["train", "val"], "split must be 'train' or 'val'"
```

**步骤 2: 选择文件**

```python
parquet_paths = list_parquet_files()
parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
```

**Train/Val 分割策略:**

```python
# 假设有 240 个分片:
all_files = [shard_00000, shard_00001, ..., shard_00239]

# Train split:
train_files = [shard_00000, shard_00001, ..., shard_00238]  # 前 239 个
# [:-1] 切片：除了最后一个

# Val split:
val_files = [shard_00239]  # 只有最后一个
# [-1:] 切片：只要最后一个
```

**为什么这样分割?**

```python
# 1. 简单
# 不需要复杂的分割逻辑

# 2. 高效
# Val 只需要一个分片，评估速度快

# 3. 固定
# 每次训练使用相同的验证集，结果可比较

# 4. 数据量合理
# 1 个分片 ≈ 250M 字符 ≈ 足够评估
```

**步骤 3: 迭代文件和 Row Groups**

```python
for filepath in parquet_paths:
    pf = pq.ParquetFile(filepath)
    for rg_idx in range(start, pf.num_row_groups, step):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column('text').to_pylist()
        yield texts
```

**逐行解析:**

```python
# 1. 遍历每个文件
for filepath in parquet_paths:
    # 2. 打开 Parquet 文件（不读取数据）
    pf = pq.ParquetFile(filepath)

    # 3. 遍历 Row Groups
    for rg_idx in range(start, pf.num_row_groups, step):
        # start=0, step=1: 遍历所有 (0, 1, 2, 3, ...)
        # start=0, step=8: 只取 (0, 8, 16, 24, ...) ← DDP rank 0
        # start=5, step=8: 只取 (5, 13, 21, 29, ...) ← DDP rank 5

        # 4. 读取一个 Row Group
        rg = pf.read_row_group(rg_idx)

        # 5. 提取 'text' 列
        texts = rg.column('text').to_pylist()
        # texts = ["doc1", "doc2", "doc3", ...]

        # 6. 返回这一批
        yield texts
```

#### DDP 支持详解

**为什么需要 `start` 和 `step`?**

```python
# 8 GPU 分布式训练
# 每个 GPU 需要不同的数据

# GPU 0: start=0, step=8
# 读取 Row Group: 0, 8, 16, 24, ...

# GPU 1: start=1, step=8
# 读取 Row Group: 1, 9, 17, 25, ...

# GPU 5: start=5, step=8
# 读取 Row Group: 5, 13, 21, 29, ...

# 结果: 每个 GPU 读取不同的数据，无重叠！
```

**使用示例:**

```python
from nanochat.common import get_dist_info
from nanochat.dataset import parquets_iter_batched

# 获取 DDP 信息
ddp, rank, local_rank, world_size = get_dist_info()

# 创建迭代器
data_iter = parquets_iter_batched(
    split="train",
    start=rank,        # 每个 rank 不同的起点
    step=world_size    # 步长 = 总进程数
)

# 迭代数据
for batch in data_iter:
    # batch = ["doc1", "doc2", ...]
    for doc in batch:
        # 处理每个文档
        process(doc)
```

#### 内存效率

```python
# 假设一个 Row Group 有 1000 个文档
# 每个文档平均 5000 字符
# 总内存: 1000 × 5000 = 5M 字符 ≈ 5 MB

# 迭代 240 个分片 × 100 Row Groups
# 但每次只占用 ~5 MB 内存！

# 相比一次性加载:
# 240 × 100 × 5 MB = 120 GB 内存 ❌
# vs
# 流式读取: ~5 MB 内存 ✓
```

---

## 下载机制

### `download_single_file()`

这是**最核心**的函数，处理单个文件的下载，包含完整的错误处理。

#### 函数签名

```python
def download_single_file(index):
    """ Downloads a single file index, with some backoff """
```

#### 完整流程

```
┌─────────────────────────────────────────────────────────┐
│ 1. 检查文件是否已存在                                    │
│    → 存在: 跳过                                          │
│    → 不存在: 继续                                        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 构建 URL                                             │
│    BASE_URL/shard_00005.parquet                         │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 尝试下载 (最多 5 次)                                 │
│    ├─ 流式下载到 .tmp 文件                              │
│    ├─ 成功: 重命名 .tmp → .parquet                      │
│    └─ 失败: 指数退避后重试                              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 返回结果                                             │
│    True: 成功 / False: 失败                             │
└─────────────────────────────────────────────────────────┘
```

#### 源码详解

**步骤 1: 检查已存在**

```python
filename = index_to_filename(index)  # "shard_00005.parquet"
filepath = os.path.join(DATA_DIR, filename)

if os.path.exists(filepath):
    print(f"Skipping {filepath} (already exists)")
    return True
```

**为什么需要这个检查?**

```python
# 场景 1: 重新运行训练
# 数据已经下载过了，跳过

# 场景 2: 下载被中断
# 重新运行，继续下载剩余文件

# 场景 3: 并行下载
# 多个进程同时下载不同文件，互不干扰
```

**步骤 2: 构建 URL**

```python
url = f"{BASE_URL}/{filename}"
print(f"Downloading {filename}...")

# 示例:
# url = "https://huggingface.co/.../shard_00005.parquet"
```

**步骤 3: 重试循环**

```python
max_attempts = 5
for attempt in range(1, max_attempts + 1):
    try:
        # 尝试下载
        ...
        return True  # 成功
    except (requests.RequestException, IOError) as e:
        # 失败处理
        ...
        if attempt < max_attempts:
            # 重试
            ...
        else:
            # 放弃
            return False
```

**为什么重试 5 次?**

```python
# 网络不稳定的原因:
1. 临时连接失败
2. 服务器过载
3. 超时
4. 部分数据损坏

# 重试策略:
# 尝试 1: 立即重试
# 尝试 2: 等 2 秒
# 尝试 3: 等 4 秒
# 尝试 4: 等 8 秒
# 尝试 5: 等 16 秒
# 总等待: 2 + 4 + 8 + 16 = 30 秒

# 5 次是经验值:
# - 太少: 临时故障无法恢复
# - 太多: 浪费时间
```

**步骤 4: 流式下载**

```python
response = requests.get(url, stream=True, timeout=30)
response.raise_for_status()

# stream=True: 不一次性读取全部
# 而是流式读取，内存友好
```

**什么是流式下载?**

```python
# ❌ 非流式:
response = requests.get(url)
content = response.content  # 一次性读取 100 MB 到内存
write_to_file(content)

# ✅ 流式:
response = requests.get(url, stream=True)
for chunk in response.iter_content(chunk_size=1MB):
    write_to_file(chunk)  # 每次只处理 1 MB

# 优点:
1. 内存占用小
2. 可以显示进度
3. 可以提前中止
```

**步骤 5: 写入临时文件**

```python
temp_path = filepath + f".tmp"  # shard_00005.parquet.tmp
with open(temp_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
        if chunk:
            f.write(chunk)
```

**为什么先写临时文件?**

```python
# 如果直接写最终文件:
# 1. 下载到一半
# 2. 网络断开
# 3. 文件不完整但存在
# 4. 下次运行会跳过（以为下载完了）

# 使用临时文件:
# 1. 下载到一半
# 2. 网络断开
# 3. .tmp 文件存在，但最终文件不存在
# 4. 下次运行会重新下载 ✓
```

**步骤 6: 原子重命名**

```python
os.rename(temp_path, filepath)
# shard_00005.parquet.tmp → shard_00005.parquet
```

**为什么用 `rename`?**

```python
# os.rename() 是原子操作
# 要么成功，要么失败
# 不会出现"半个文件"的情况

# 时间线:
# t0: shard_00005.parquet.tmp 存在
# t1: os.rename()
# t2: shard_00005.parquet 存在，.tmp 消失

# 没有中间状态！
```

**步骤 7: 错误处理**

```python
except (requests.RequestException, IOError) as e:
    print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")

    # 清理损坏的文件
    for path in [filepath + f".tmp", filepath]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass  # 删除失败也不报错
```

**为什么要清理?**

```python
# 损坏的文件可能:
1. 不完整（只下载了一部分）
2. 格式错误（传输中损坏）
3. 占用磁盘空间

# 清理后:
# 下次重试从零开始，确保完整性
```

**步骤 8: 指数退避**

```python
if attempt < max_attempts:
    wait_time = 2 ** attempt  # 指数退避
    print(f"Waiting {wait_time} seconds before retry...")
    time.sleep(wait_time)
```

**指数退避 (Exponential Backoff):**

```python
# 尝试次数 → 等待时间
attempt 1: 2^1 = 2 秒
attempt 2: 2^2 = 4 秒
attempt 3: 2^3 = 8 秒
attempt 4: 2^4 = 16 秒

# 为什么指数增长?
1. 临时故障: 快速恢复
2. 持续故障: 避免过度重试
3. 服务器压力: 给服务器喘息时间
```

**完整示例:**

```python
# 下载失败的时间线:

t0:   尝试 1 → 失败 (网络抖动)
t2:   等待 2 秒
t2:   尝试 2 → 失败 (仍然有问题)
t6:   等待 4 秒
t6:   尝试 3 → 成功 ✓

# 如果没有指数退避:
t0:   尝试 1 → 失败
t2:   尝试 2 → 失败 (间隔太短，问题未解决)
t4:   尝试 3 → 失败
...
# 大量无效重试
```

---

## 命令行接口

### 使用方式

```bash
# 作为模块运行
python -m nanochat.dataset [选项]
```

### 参数

```python
parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
parser.add_argument("-n", "--num-files", type=int, default=-1,
                    help="Number of shards to download (default: -1), -1 = disable")
parser.add_argument("-w", "--num-workers", type=int, default=4,
                    help="Number of parallel download workers (default: 4)")
```

**参数详解:**

#### `-n, --num-files`

```bash
# 默认: -1 (不下载)
python -m nanochat.dataset

# 下载 8 个分片
python -m nanochat.dataset -n 8

# 下载所有 1823 个分片
python -m nanochat.dataset -n 1823
# 或
python -m nanochat.dataset -n -1  # 会自动转换为 1823
```

**为什么默认是 -1?**

```python
# 避免意外下载
# 用户必须明确指定要下载多少

# 安全机制:
if args.num_files == -1:
    num = MAX_SHARD + 1  # 下载所有
else:
    num = min(args.num_files, MAX_SHARD + 1)  # 限制最大值
```

#### `-w, --num-workers`

```bash
# 默认: 4 个并行下载
python -m nanochat.dataset -n 240

# 使用 8 个并行下载（更快）
python -m nanochat.dataset -n 240 -w 8

# 使用 1 个下载（串行，慢但稳定）
python -m nanochat.dataset -n 240 -w 1
```

**为什么并行下载?**

```python
# 单线程下载:
# 下载 240 个文件，每个 30 秒
# 总时间: 240 × 30 = 7200 秒 = 2 小时

# 4 线程并行:
# 总时间: 7200 / 4 = 1800 秒 = 30 分钟

# 8 线程并行:
# 总时间: 7200 / 8 = 900 秒 = 15 分钟
```

**为什么默认 4?**

```python
# 权衡:
# - 太少 (1-2): 慢
# - 适中 (4-8): 快且不过载服务器
# - 太多 (16+): 服务器可能限流
```

### 并行下载实现

```python
ids_to_download = list(range(num))  # [0, 1, 2, ..., 239]
print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")

with Pool(processes=args.num_workers) as pool:
    results = pool.map(download_single_file, ids_to_download)
```

**`multiprocessing.Pool` 工作原理:**

```
主进程
  ↓
创建进程池 (4 个 worker)
  ↓
分配任务: [0, 1, 2, ..., 239]
  ↓
┌────────────────────────────────────────┐
│ Worker 0: download(0), download(4), ...│
│ Worker 1: download(1), download(5), ...│
│ Worker 2: download(2), download(6), ...│
│ Worker 3: download(3), download(7), ...│
└────────────────────────────────────────┘
  ↓
等待所有完成
  ↓
收集结果: [True, True, False, ...]
```

**为什么用进程而非线程?**

```python
# Python 的 GIL (Global Interpreter Lock)
# 同一时间只有一个线程执行 Python 代码

# 多线程: 不能真正并行（I/O 密集可以）
# 多进程: 真正并行 ✓

# 对于下载任务:
# I/O 密集（等待网络），线程也可以
# 但进程更稳定（隔离性好）
```

### 结果统计

```python
successful = sum(1 for success in results if success)
print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")

# 示例输出:
# Done! Downloaded: 238/240 shards to ~/.cache/nanochat/base_data
# (2 个下载失败)
```

---

## 实战示例

### 示例 1: 下载训练数据

```bash
# d20 模型需要 240 个分片
python -m nanochat.dataset -n 240 -w 8

# 输出:
# Downloading 240 shards using 8 workers...
# Target directory: /home/user/.cache/nanochat/base_data
#
# Downloading shard_00000.parquet...
# Downloading shard_00001.parquet...
# ...
# Successfully downloaded shard_00000.parquet
# ...
# Done! Downloaded: 240/240 shards to /home/user/.cache/nanochat/base_data
```

### 示例 2: 流式迭代数据

```python
from nanochat.dataset import parquets_iter_batched

# 迭代训练数据
for batch in parquets_iter_batched(split="train"):
    # batch = ["doc1", "doc2", "doc3", ...]
    print(f"Processing {len(batch)} documents")
    for doc in batch:
        # 处理每个文档
        tokens = tokenize(doc)
        train(tokens)

# 验证集
for batch in parquets_iter_batched(split="val"):
    evaluate(batch)
```

### 示例 3: 分布式训练

```python
from nanochat.common import get_dist_info
from nanochat.dataset import parquets_iter_batched

# 获取 rank 信息
ddp, rank, local_rank, world_size = get_dist_info()

# 每个 rank 读取不同的数据
data_iter = parquets_iter_batched(
    split="train",
    start=rank,
    step=world_size
)

print(f"Rank {rank} starting iteration...")
for batch in data_iter:
    # 这个 rank 独有的数据
    process(batch)
```

### 示例 4: 检查下载进度

```python
from nanochat.dataset import list_parquet_files, MAX_SHARD

# 检查已下载的文件
downloaded = list_parquet_files()
total_needed = 240  # d20 需要的数量

print(f"Progress: {len(downloaded)}/{total_needed}")
print(f"Remaining: {total_needed - len(downloaded)}")

# 示例输出:
# Progress: 150/240
# Remaining: 90
```

### 示例 5: 手动下载特定分片

```python
from nanochat.dataset import download_single_file

# 下载单个分片
success = download_single_file(5)
if success:
    print("Downloaded shard 5")
else:
    print("Failed to download shard 5")

# 下载一批特定分片
for i in [10, 20, 30, 40, 50]:
    download_single_file(i)
```

---

## 核心设计思想

### 1. 按需下载 (Lazy Loading)

```python
# ✅ nanochat 方式:
# 1. 训练时才下载
# 2. 只下载需要的数量
# 3. 下载和训练可以并行

# ❌ 传统方式:
# 1. 预先下载全部数据
# 2. 占用大量磁盘空间
# 3. 下载完才能训练
```

### 2. 流式处理 (Streaming)

```python
# ✅ 使用 Row Groups:
# 每次只加载一小批数据
# 内存占用: ~5 MB

# ❌ 一次性加载:
# 加载整个数据集
# 内存占用: ~120 GB
```

### 3. 容错性 (Fault Tolerance)

```python
# 多层容错:
1. 重试机制 (5 次)
2. 指数退避 (避免过度重试)
3. 临时文件 (避免损坏)
4. 原子重命名 (确保完整性)
5. 清理损坏文件 (重新开始)
```

### 4. 分布式友好 (DDP Support)

```python
# start/step 参数:
# 每个 rank 读取不同的 Row Groups
# 无数据重叠，无需额外同步
```

### 5. 并行下载 (Parallel Download)

```python
# multiprocessing.Pool:
# 多进程并行下载
# 显著减少下载时间
```

---

## 数据流程图

```
┌─────────────────────────────────────────────────────────┐
│                  训练启动                                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 检查本地数据                                             │
│ list_parquet_files()                                    │
└─────────────────────────────────────────────────────────┘
                         ↓
                    数据足够？
                    ↙         ↘
                  是           否
                  ↓            ↓
┌─────────────────┐   ┌──────────────────────┐
│ 开始训练        │   │ 下载数据             │
└─────────────────┘   │ python -m dataset -n │
         ↑            └──────────────────────┘
         │                      ↓
         │            ┌──────────────────────┐
         │            │ 并行下载             │
         │            │ (multiprocessing)    │
         │            └──────────────────────┘
         │                      ↓
         │            ┌──────────────────────┐
         │            │ 重试 + 容错          │
         │            └──────────────────────┘
         │                      ↓
         └──────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 流式迭代数据                                             │
│ parquets_iter_batched()                                 │
│ ├─ 文件 1                                               │
│ │  ├─ Row Group 0                                       │
│ │  ├─ Row Group 1                                       │
│ │  └─ ...                                               │
│ ├─ 文件 2                                               │
│ └─ ...                                                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 训练模型                                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 总结

### 核心功能

| 功能 | 函数 | 用途 |
|------|------|------|
| **列出文件** | `list_parquet_files()` | 获取已下载的分片 |
| **迭代数据** | `parquets_iter_batched()` | 流式读取训练数据 |
| **下载文件** | `download_single_file()` | 下载单个分片 |
| **批量下载** | 命令行 `-n -w` | 并行下载多个分片 |

### 设计亮点

1. **按需下载** - 节省存储空间
2. **流式处理** - 节省内存
3. **容错机制** - 网络不稳定时仍能成功
4. **并行下载** - 减少等待时间
5. **DDP 友好** - 支持多 GPU 训练
6. **原子操作** - 保证数据完整性

### 关键数字

```python
数据集: FineWeb-Edu-100B
分片数: 1823 个
每个分片: ~100 MB (压缩), ~250M 字符
总大小: ~180 GB (压缩)

d20 需要: 240 个分片 ≈ 24 GB
d26 需要: 450 个分片 ≈ 45 GB
d32 需要: 800 个分片 ≈ 80 GB

重试: 最多 5 次
退避: 2^attempt 秒
并行: 4 个进程
```

---

**下一篇**: [tokenizer.py - BPE 分词器实现](04_tokenizer.md)

---

## 思考题

1. 为什么用临时文件 + 原子重命名，而不是直接写最终文件？
2. 指数退避的好处是什么？为什么不用固定间隔重试？
3. 如何计算某个模型大小需要下载多少个分片？
4. `parquets_iter_batched` 的 `start/step` 参数如何实现 DDP 数据分割？