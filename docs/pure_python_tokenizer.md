# Pure Python BPE Tokenizer

一个**完全用 Python 实现**的 Byte Pair Encoding (BPE) 分词器，用于教育和学习目的。

## 概述

nanochat 项目中的生产环境使用 Rust + tiktoken 来实现高性能的 BPE 分词。但为了学习和理解 BPE 算法的工作原理，我们提供了这个**纯 Python 实现**。

### 为什么需要纯 Python 版本？

- **📚 教育价值**: 代码清晰易读，每一步都有注释
- **🔍 可调试**: 可以单步调试，观察算法每一步的行为
- **🛠️ 可修改**: 易于修改和实验不同的想法
- **⚡ 无依赖**: 不需要编译 Rust 或安装复杂的库（只需 `regex` 包）

### 性能对比

| 实现方式 | 训练速度 | 编码速度 | 适用场景 |
|---------|---------|---------|---------|
| **Pure Python** | 慢 (~10-100x) | 慢 (~10-100x) | 学习、实验、小数据集 |
| **Rust + tiktoken** | 快 | 快 | 生产环境、大规模训练 |

## 核心概念

### 什么是 BPE？

Byte Pair Encoding (字节对编码) 是一种数据压缩和分词算法：

1. **初始化**: 从 256 个字节级别的 token 开始
2. **统计**: 找到文本中最频繁出现的字节对（byte pair）
3. **合并**: 将最频繁的 pair 合并成一个新 token
4. **重复**: 重复步骤 2-3，直到达到目标词汇表大小

### 示例

```
原始文本: "hello world"

Step 1: 字节级表示
h e l l o   w o r l d
104 101 108 108 111 32 119 111 114 108 100

Step 2: 找到最频繁的 pair，假设是 (108, 108) "ll"
合并后: 104 101 [256] 111 32 119 111 114 108 100
        h   e   [ll]  o       w   o   r   l   d

Step 3: 继续找下一个最频繁的 pair...
```

## 使用方法

### 安装依赖

```bash
pip install regex
```

### 快速开始

```python
from nanochat.pure_python_tokenizer import train_tokenizer

# 准备训练数据
training_texts = [
    "Hello, world!",
    "Machine learning is fascinating.",
    "Python programming is powerful.",
    # ... 更多文本
]

# 训练分词器
tokenizer = train_tokenizer(
    iter(training_texts),
    vocab_size=8192,  # 目标词汇表大小
    verbose=True
)

# 编码文本
text = "Hello, world!"
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")

# 解码
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")

# 保存
tokenizer.save("my_tokenizer")

# 加载
from nanochat.pure_python_tokenizer import PurePythonBPETokenizer
loaded_tokenizer = PurePythonBPETokenizer.load("my_tokenizer")
```

### 详细示例

参考 `examples/tokenizer_demo.py` 查看完整示例，包括：

1. **基础使用** - 训练、编码、解码
2. **详细可视化** - 查看每个 token 的细节
3. **特殊 token** - 使用 `<|bos|>` 等特殊标记
4. **保存和加载** - 持久化分词器
5. **性能对比** - 与 Rust 版本对比
6. **压缩分析** - 分析压缩比例

运行示例：

```bash
cd nanochat
python examples/tokenizer_demo.py
```

## API 文档

### `PurePythonBPETokenizer`

#### 初始化

```python
tokenizer = PurePythonBPETokenizer()
```

#### 训练

```python
tokenizer.train(
    text_iterator,  # Iterator[str]: 文本迭代器
    vocab_size,     # int: 目标词汇表大小
    verbose=True    # bool: 是否打印进度
)
```

**参数说明**:
- `text_iterator`: 产生文本字符串的迭代器
- `vocab_size`: 词汇表大小（必须 ≥ 256 + 特殊token数量）
- `verbose`: 是否打印训练进度

**训练过程**:
1. 使用正则表达式分割文本
2. 统计 chunk 频率
3. 执行 BPE 合并（vocab_size - 256 - num_special 次）
4. 添加特殊 token

#### 编码

```python
token_ids = tokenizer.encode(
    text,                    # str: 要编码的文本
    add_special_tokens=False # bool: 是否添加 <|bos|>
)
```

**返回**: `List[int]` - token ID 列表

#### 解码

```python
text = tokenizer.decode(token_ids)  # List[int] -> str
```

**返回**: `str` - 解码后的文本

#### 特殊 token

```python
# 获取特殊 token ID
bos_id = tokenizer.encode_special("<|bos|>")

# 获取所有特殊 token
special_tokens = tokenizer.get_special_tokens()
```

#### 保存和加载

```python
# 保存
tokenizer.save("path/to/save/dir")

# 加载
tokenizer = PurePythonBPETokenizer.load("path/to/save/dir")
```

#### 工具方法

```python
# 获取词汇表大小
vocab_size = tokenizer.get_vocab_size()

# 可视化编码过程（调试用）
tokenizer.visualize_encoding("Hello, world!")
```

## 算法详解

### 1. 文本预处理

使用 GPT-4 风格的正则表达式分割文本：

```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

这个模式：
- 保留缩写词 (I'm, you're)
- 单独处理数字（1-2位数字为一组）
- 处理标点符号
- 处理空白字符

### 2. BPE 训练算法

```python
伪代码：

初始化:
  vocab = {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}
  merges = {}
  next_token_id = 256

循环 (vocab_size - 256 - num_special 次):
  1. 统计所有文本块中的字节对频率
     pair_counts = Counter()
     for each chunk:
       for each adjacent pair (a, b):
         pair_counts[(a, b)] += chunk_count

  2. 找到最频繁的 pair
     best_pair = max(pair_counts, key=pair_counts.get)

  3. 记录这次合并
     merges[best_pair] = next_token_id
     vocab[next_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

  4. 在所有 chunk 中应用这次合并
     for each chunk:
       merge best_pair into next_token_id

  5. next_token_id += 1
```

### 3. 编码过程

```python
def encode(text):
  1. 使用正则分割文本成 chunks
     chunks = regex.findall(SPLIT_PATTERN, text)

  2. 对每个 chunk:
     a. 转换为字节序列: [104, 101, 108, 108, 111] for "hello"
     b. 迭代应用 merges:
        - 找到序列中存在且最早训练的 pair
        - 用对应的 merged_token_id 替换
        - 重复直到无法继续合并

  3. 连接所有 chunk 的 token IDs
```

### 4. 解码过程

```python
def decode(token_ids):
  1. 查找每个 token_id 对应的字节序列
     bytes_list = [vocab[tid] for tid in token_ids]

  2. 连接所有字节
     result_bytes = b''.join(bytes_list)

  3. 解码为 UTF-8 字符串
     return result_bytes.decode('utf-8')
```

## 性能优化建议

纯 Python 实现主要用于学习，但如果需要提速：

### 1. 使用 NumPy

```python
import numpy as np

# 用 NumPy 数组替代 Python list
ids = np.array(ids, dtype=np.int32)
```

### 2. 缓存正则匹配

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def split_text(text):
    return regex.findall(SPLIT_PATTERN, text)
```

### 3. 并行处理

```python
from multiprocessing import Pool

def process_chunk(chunk):
    # 并行处理多个 chunk
    return encode_chunk(chunk)

with Pool() as pool:
    results = pool.map(process_chunk, chunks)
```

### 4. 升级到生产版本

对于大规模应用，使用：
- **训练**: `nanochat.tokenizer.RustBPETokenizer`
- **推理**: `tiktoken`

## 与 nanochat Rust 版本的区别

| 特性 | Pure Python | Rust + tiktoken |
|-----|-------------|-----------------|
| **训练速度** | ~100 docs/sec | ~10,000 docs/sec |
| **编码速度** | ~1,000 tokens/sec | ~100,000 tokens/sec |
| **内存使用** | 较高 | 优化的 |
| **并行支持** | 需手动实现 | 内置 Rayon |
| **依赖** | 仅 `regex` | Rust 工具链 |
| **代码行数** | ~400 行 | ~500 行 (Rust) |
| **可读性** | 极高 ⭐⭐⭐⭐⭐ | 中等 ⭐⭐⭐ |

## 常见问题

### Q1: 为什么训练很慢？

**A**: 纯 Python 实现未优化性能。对于大规模训练，使用 Rust 版本。

### Q2: 可以用于生产吗？

**A**: 不建议。这是**教育工具**。生产环境使用 `RustBPETokenizer`。

### Q3: 词汇表大小如何选择？

**A**: 常见选择：
- **小模型**: 8K - 16K
- **中等模型**: 32K - 64K
- **大模型**: 64K - 100K (nanochat 使用 65536 = 2^16)

### Q4: 为什么结果与 Rust 版本不完全一致？

**A**: 可能的原因：
- 训练数据顺序不同
- 随机性（如果有的话）
- Tie-breaking 策略差异

### Q5: 如何添加更多特殊 token？

**A**: 修改 `SPECIAL_TOKENS` 列表：

```python
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    # 添加你的特殊 token
    "<|my_token|>",
]
```

## 进阶话题

### BPE 的局限性

1. **语言偏见**: 对训练语料库中的语言更有效
2. **罕见词**: 罕见词可能被过度分割
3. **数字处理**: 大数字会被分成多个 token
4. **格式敏感**: 空格、大小写敏感

### 改进方向

- **Unigram LM**: 基于概率的分词
- **WordPiece**: Google 的变体
- **SentencePiece**: 支持更多语言
- **BPE-dropout**: 训练时随机丢弃合并

### 实验建议

1. **修改正则模式**: 尝试不同的文本分割策略
2. **词汇表大小**: 观察不同大小对压缩率的影响
3. **训练数据**: 使用不同领域的文本
4. **特殊 token**: 添加任务特定的标记

## 资源链接

- **原始论文**: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- **GPT-2 Tokenizer**: [OpenAI tiktoken](https://github.com/openai/tiktoken)
- **HuggingFace 实现**: [tokenizers library](https://github.com/huggingface/tokenizers)
- **nanochat 文档**: [tutorial.md](../tutorial.md)

## 总结

纯 Python BPE tokenizer 是理解分词算法的**最佳入口**：

✅ **清晰**: 每一行代码都易于理解
✅ **完整**: 实现了完整的训练和推理流程
✅ **教育**: 适合学习和实验
✅ **可扩展**: 易于修改和添加新功能

但记住：**生产环境请使用优化的 Rust 版本！** 🚀

---

**Happy tokenizing! 🎉**
