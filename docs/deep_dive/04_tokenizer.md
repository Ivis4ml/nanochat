# nanochat Deep Dive 02: tokenizer.py - BPE 分词器实现

## 概述

`tokenizer.py` 是 nanochat 的核心组件之一，实现了 GPT-4 风格的 BPE（Byte Pair Encoding）分词器。该文件提供了两种不同的实现方案，每种都有其独特的优势和使用场景。

## 目录

1. [什么是 BPE 分词器？](#什么是-bpe-分词器)
2. [特殊 Token 系统](#特殊-token-系统)
3. [分割模式（Split Pattern）](#分割模式split-pattern)
4. [两种实现方案](#两种实现方案)
   - [HuggingFace Tokenizer](#huggingface-tokenizer)
   - [RustBPE + tiktoken 组合](#rustbpe--tiktoken-组合)
5. [对话渲染机制](#对话渲染机制)
6. [工具函数](#工具函数)

---

## 什么是 BPE 分词器？

BPE（Byte Pair Encoding）是一种数据压缩算法，在 NLP 中被用作 subword tokenization 方法。它的核心思想是：

1. 从单个字节（或字符）开始
2. 迭代地合并最频繁出现的字节对
3. 重复此过程直到达到目标词汇表大小

这种方法的优势：
- **处理未知词**：通过 subword 分解，可以表示任何单词
- **平衡词汇表大小**：不会像字符级那样序列太长，也不会像词级那样词汇表太大
- **多语言友好**：可以有效处理不同语言

## 特殊 Token 系统

nanochat 定义了一组特殊的控制 token，用于标记文档边界和对话结构：

```python
SPECIAL_TOKENS = [
    "<|bos|>",              # Beginning of Sequence - 文档开始标记
    "<|user_start|>",       # 用户消息开始
    "<|user_end|>",         # 用户消息结束
    "<|assistant_start|>",  # 助手消息开始
    "<|assistant_end|>",    # 助手消息结束
    "<|python_start|>",     # Python 工具调用开始
    "<|python_end|>",       # Python 工具调用结束
    "<|output_start|>",     # Python 输出开始
    "<|output_end|>",       # Python 输出结束
]
```

### 设计思路

1. **`<|bos|>` (Beginning of Sequence)**：
   - 每个文档都以此 token 开头
   - 帮助模型识别新文档的开始
   - 历史上也被称为 `<|endoftext|>`（来自 GPT-2/3）

2. **对话结构 tokens**：
   - 用于在微调阶段渲染对话格式
   - 清晰区分用户和助手的消息边界
   - 支持工具调用（Python REPL）的集成

3. **工具调用 tokens**：
   - `<|python_start|>` / `<|python_end|>`：标记助手调用 Python 代码
   - `<|output_start|>` / `<|output_end|>`：标记 Python 执行结果

## 分割模式（Split Pattern）

```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

这是一个复杂的正则表达式，用于在 BPE 训练前将文本分割成更小的单元。让我们逐部分分析：

### 模式解析

1. **`'(?i:[sdmt]|ll|ve|re)`**：
   - 匹配英语缩写：'s, 'S, 'd, 'D, 'm, 'M, 't, 'T, 'll, 'LL, 've, 'VE, 're, 'RE
   - 例如："don't" → ["don", "'t"]

2. **`[^\r\n\p{L}\p{N}]?+\p{L}+`**：
   - 匹配可选的非字母数字字符 + 一个或多个字母
   - 处理单词（可能带前导符号）

3. **`\p{N}{1,2}`**：
   - 匹配 1-2 位数字
   - **重要差异**：GPT-4 使用 `\p{N}{1,3}`（1-3 位）
   - 作者的考虑：对于小词汇表，3位数字可能"浪费"了太多 token

4. **` ?[^\s\p{L}\p{N}]++[\r\n]*`**：
   - 匹配可选空格 + 非空白/字母/数字的字符 + 可选换行
   - 处理标点符号和特殊字符

5. **`\s*[\r\n]`**：
   - 匹配可选空白 + 换行符

6. **`\s+(?!\S)`** 和 **`\s+`**：
   - 匹配空白序列

### 为什么需要分割？

BPE 算法需要在"合理的边界"上工作。如果不预先分割：
- 数字可能会被合并成很长的 token
- 跨越单词边界的合并可能不合理
- 训练效率会降低

## 两种实现方案

nanochat 提供了两种分词器实现，满足不同的需求：

### HuggingFace Tokenizer

**位置**：`HuggingFaceTokenizer` 类（第 39-148 行）

#### 特点

- **统一接口**：同时支持训练和推理
- **完整功能**：基于 HuggingFace `tokenizers` 库
- **易于理解**：代码相对直观

#### 架构组件

```python
class HuggingFaceTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # HuggingFace Tokenizer 对象
```

##### 1. 模型配置（Training）

```python
tokenizer = HFTokenizer(BPE(
    byte_fallback=True,  # 关键！允许回退到字节级编码
    unk_token=None,      # 不使用未知 token
    fuse_unk=False,
))
```

##### 2. 预处理器（Pre-tokenizer）

```python
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    # 1. 使用 GPT-4 风格的正则表达式分割
    pre_tokenizers.Split(
        pattern=gpt4_split_regex,
        behavior="isolated",
        invert=False
    ),
    # 2. 字节级编码
    pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False
    )
])
```

**工作流程**：
1. 正则表达式将文本分割成 chunks
2. ByteLevel 将每个 chunk 转换为字节序列
3. BPE 在这些字节序列上进行合并

##### 3. 解码器（Decoder）

```python
tokenizer.decoder = decoders.ByteLevel()
```

将 token IDs 解码回文本时，需要逆向 ByteLevel 编码。

##### 4. 训练器（Trainer）

```python
trainer = BpeTrainer(
    vocab_size=vocab_size,
    show_progress=True,
    min_frequency=0,  # 不设置最小频率限制
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=SPECIAL_TOKENS,
)
```

#### 核心方法

**训练**：
```python
@classmethod
def train_from_iterator(cls, text_iterator, vocab_size):
    # 配置分词器组件
    # 从文本迭代器训练
    tokenizer.train_from_iterator(text_iterator, trainer)
    return cls(tokenizer)
```

**编码**：
```python
def encode(self, text, prepend=None, append=None):
    # 支持添加前缀/后缀 token（如 BOS）
    ids = self.tokenizer.encode(text, add_special_tokens=False).ids
    if prepend is not None:
        ids.insert(0, prepend_id)
    if append is not None:
        ids.append(append_id)
    return ids
```

**解码**：
```python
def decode(self, ids):
    return self.tokenizer.decode(ids, skip_special_tokens=False)
```

**保存/加载**：
```python
def save(self, tokenizer_dir):
    # 保存为 tokenizer.json
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    self.tokenizer.save(tokenizer_path)

@classmethod
def from_directory(cls, tokenizer_dir):
    # 从 tokenizer.json 加载
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    tokenizer = HFTokenizer.from_file(tokenizer_path)
    return cls(tokenizer)
```

#### 优缺点

**优点**：
- 代码简洁，易于理解和调试
- 统一的训练和推理接口
- 丰富的 HuggingFace 生态支持

**缺点**：
- 推理速度相对较慢
- 配置复杂，容易出错
- 文档注释中提到"really confusing"

---

### RustBPE + tiktoken 组合

**位置**：`RustBPETokenizer` 类（第 155-378 行）

#### 特点

- **高性能训练**：使用 Rust 实现的 `rustbpe` 进行训练
- **高效推理**：使用 OpenAI 的 `tiktoken` 进行推理
- **最佳实践**：分离训练和推理，各自优化

#### 架构设计

```python
class RustBPETokenizer:
    def __init__(self, enc, bos_token):
        self.enc = enc  # tiktoken Encoding 对象
        self.bos_token_id = self.encode_special(bos_token)
```

#### 训练流程（rustbpe）

```python
@classmethod
def train_from_iterator(cls, text_iterator, vocab_size):
    # 1. 使用 rustbpe 训练
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(
        text_iterator,
        vocab_size_no_special,
        pattern=SPLIT_PATTERN
    )

    # 2. 提取训练结果
    pattern = tokenizer.get_pattern()
    mergeable_ranks_list = tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}

    # 3. 构造 tiktoken Encoding
    tokens_offset = len(mergeable_ranks)
    special_tokens = {
        name: tokens_offset + i
        for i, name in enumerate(SPECIAL_TOKENS)
    }
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    return cls(enc, "<|bos|>")
```

**关键点**：
1. **分离词汇空间**：
   - 前 N 个 token：BPE 训练的普通 token
   - 后 9 个 token：特殊 token
   - `vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)`

2. **无缝转换**：
   - rustbpe 训练后提取 `mergeable_ranks`（合并规则）
   - 使用相同规则构造 tiktoken Encoding
   - 保证训练和推理的一致性

#### 推理流程（tiktoken）

**编码**：
```python
def encode(self, text, prepend=None, append=None, num_threads=8):
    if isinstance(text, str):
        ids = self.enc.encode_ordinary(text)
    elif isinstance(text, list):
        # 批量编码，多线程加速
        ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)

    # 添加前缀/后缀
    if prepend is not None:
        ids.insert(0, prepend_id)
    if append is not None:
        ids.append(append_id)

    return ids
```

**特殊 token 编码**（带缓存）：
```python
@lru_cache(maxsize=32)
def encode_special(self, text):
    return self.enc.encode_single_token(text)
```

**解码**：
```python
def decode(self, ids):
    return self.enc.decode(ids)
```

#### 加载预训练模型

```python
@classmethod
def from_pretrained(cls, tiktoken_name):
    # 加载 OpenAI 的预训练分词器（如 "gpt2", "cl100k_base"）
    enc = tiktoken.get_encoding(tiktoken_name)
    # 注意：OpenAI 使用 "<|endoftext|>" 而非 "<|bos|>"
    return cls(enc, "<|endoftext|>")
```

**历史命名混淆**：
- OpenAI/tiktoken 使用 `<|endoftext|>`
- 但这个 token 通常用于文档**开始**（不是结束！）
- nanochat 重命名为 `<|bos|>`（更准确的语义）

#### 持久化

```python
def save(self, tokenizer_dir):
    # 使用 pickle 保存 tiktoken Encoding 对象
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(self.enc, f)

@classmethod
def from_directory(cls, tokenizer_dir):
    # 从 pickle 加载
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pickle_path, "rb") as f:
        enc = pickle.load(f)
    return cls(enc, "<|bos|>")
```

#### 优缺点

**优点**：
- **极快的推理速度**：tiktoken 是 C++ 实现，高度优化
- **批量处理**：支持多线程批量编码
- **生产级质量**：tiktoken 被 OpenAI 在生产环境使用
- **清晰分离**：训练和推理关注点分离

**缺点**：
- 依赖两个库（rustbpe 和 tiktoken）
- 需要理解两个系统的交互
- 保存格式是 pickle（不如 JSON 可读）

---

## 对话渲染机制

`render_conversation()` 方法是 nanochat 的核心功能之一，将对话转换为模型可训练的 token 序列。

### 函数签名

```python
def render_conversation(self, conversation, max_tokens=2048):
    """
    返回：
    - ids: list[int] - token ID 序列
    - mask: list[int] - 训练掩码（1=训练，0=不训练）
    """
```

### 输入格式

```python
conversation = {
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        # ...
    ]
}
```

### 处理流程

#### 1. 初始化

```python
ids, mask = [], []

def add_tokens(token_ids, mask_val):
    """辅助函数：添加 token 和对应的掩码值"""
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    ids.extend(token_ids)
    mask.extend([mask_val] * len(token_ids))
```

#### 2. 处理 System 消息

```python
if conversation["messages"][0]["role"] == "system":
    # 将 system 消息合并到第一个 user 消息
    conversation = copy.deepcopy(conversation)
    messages = conversation["messages"]
    assert messages[1]["role"] == "user"
    messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
    messages = messages[1:]
```

**设计考虑**：
- GPT 模型通常不直接支持 system role
- 将 system 消息作为 user 消息的前缀
- 使用 `\n\n` 分隔

#### 3. 获取特殊 token IDs

```python
bos = self.get_bos_token_id()
user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")
```

#### 4. 添加 BOS token

```python
add_tokens(bos, 0)  # mask=0，不训练 BOS
```

#### 5. 遍历消息

```python
for i, message in enumerate(messages):
    # 验证消息顺序
    must_be_from = "user" if i % 2 == 0 else "assistant"
    assert message["role"] == must_be_from

    content = message["content"]
```

**严格的顺序检查**：
- 偶数位置必须是 user
- 奇数位置必须是 assistant
- 防止格式错误

#### 6. 渲染 User 消息

```python
if message["role"] == "user":
    assert isinstance(content, str)
    value_ids = self.encode(content)
    add_tokens(user_start, 0)  # 不训练
    add_tokens(value_ids, 0)   # 不训练用户输入
    add_tokens(user_end, 0)    # 不训练
```

**掩码策略**：
- User 消息的所有 token 都是 `mask=0`
- 模型不需要学习生成用户输入

#### 7. 渲染 Assistant 消息

**简单文本**：
```python
elif message["role"] == "assistant":
    add_tokens(assistant_start, 0)
    if isinstance(content, str):
        value_ids = self.encode(content)
        add_tokens(value_ids, 1)  # mask=1，训练！
    add_tokens(assistant_end, 1)
```

**复杂结构（工具调用）**：
```python
elif isinstance(content, list):
    for part in content:
        value_ids = self.encode(part["text"])
        if part["type"] == "text":
            add_tokens(value_ids, 1)  # 训练
        elif part["type"] == "python":
            add_tokens(python_start, 1)
            add_tokens(value_ids, 1)  # 训练 Python 代码
            add_tokens(python_end, 1)
        elif part["type"] == "python_output":
            add_tokens(output_start, 0)
            add_tokens(value_ids, 0)  # 不训练输出（来自 Python）
            add_tokens(output_end, 0)
```

**掩码策略详解**：
- `assistant_start`：`mask=0`（控制 token）
- 文本内容：`mask=1`（训练生成文本）
- Python 代码：`mask=1`（训练生成代码）
- Python 输出：`mask=0`（运行时生成，不训练）
- `assistant_end`：`mask=1`（训练结束标记）

#### 8. 截断

```python
ids = ids[:max_tokens]
mask = mask[:max_tokens]
return ids, mask
```

防止序列过长导致 OOM。

### 可视化工具

```python
def visualize_tokenization(self, ids, mask, with_token_id=False):
    """调试工具：可视化 token 和掩码"""
    RED = '\033[91m'    # mask=0
    GREEN = '\033[92m'  # mask=1
    RESET = '\033[0m'

    tokens = []
    for token_id, mask_val in zip(ids, mask):
        token_str = self.decode([token_id])
        color = GREEN if mask_val == 1 else RED
        tokens.append(f"{color}{token_str}{RESET}")
        if with_token_id:
            tokens.append(f"({token_id})")

    return '|'.join(tokens)
```

**输出示例**：
```
<|bos|>|<|user_start|>|Hello|!|<|user_end|>|<|assistant_start|>|Hi| there|!|<|assistant_end|>
  红色                                             绿色（训练这些 token）
```

### 强化学习渲染

```python
def render_for_completion(self, conversation):
    """用于强化学习：准备让模型补全的上下文"""
    # 1. 移除最后的 assistant 消息
    conversation = copy.deepcopy(conversation)
    messages = conversation["messages"]
    assert messages[-1]["role"] == "assistant"
    messages.pop()

    # 2. 正常渲染（不包括最后的回复）
    ids, mask = self.render_conversation(conversation)

    # 3. 添加 assistant_start token（触发补全）
    assistant_start = self.encode_special("<|assistant_start|>")
    ids.append(assistant_start)

    return ids
```

**使用场景**：
- RLHF（Reinforcement Learning from Human Feedback）
- 模型需要生成完整的 assistant 回复
- 提供"提示"但不提供答案

---

## 工具函数

### get_tokenizer()

```python
def get_tokenizer():
    """nanochat 的默认分词器加载函数"""
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)
```

**设计决策**：
- 默认使用 `RustBPETokenizer`（高性能）
- 从 `{base_dir}/tokenizer/` 目录加载
- 简化了其他模块的导入

### get_token_bytes()

```python
def get_token_bytes(device="cpu"):
    """加载 token 的字节表示（用于模型初始化）"""
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")

    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
```

**用途**：
- 用于 GPT 模型的词嵌入层初始化
- 提供每个 token 的原始字节表示
- 支持跨设备加载（CPU/GPU）

---

## 总体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     tokenizer.py                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │         特殊 Token 定义（SPECIAL_TOKENS）          │     │
│  │  <|bos|>, <|user_start|>, <|assistant_start|>, ...│     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │     分割模式（SPLIT_PATTERN - GPT-4 风格正则）     │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  ┌──────────────────────┐  ┌─────────────────────────┐    │
│  │ HuggingFaceTokenizer │  │  RustBPETokenizer       │    │
│  ├──────────────────────┤  ├─────────────────────────┤    │
│  │ 训练：HF Tokenizers  │  │ 训练：rustbpe (Rust)    │    │
│  │ 推理：HF Tokenizers  │  │ 推理：tiktoken (C++)    │    │
│  │                      │  │                         │    │
│  │ 优点：                │  │ 优点：                  │    │
│  │ - 统一接口            │  │ - 极快推理速度          │    │
│  │ - 易于理解            │  │ - 批量+多线程           │    │
│  │                      │  │ - 生产级质量            │    │
│  │ 缺点：                │  │                         │    │
│  │ - 推理较慢            │  │ 缺点：                  │    │
│  │ - 配置复杂            │  │ - 依赖两个库            │    │
│  └──────────────────────┘  └─────────────────────────┘    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │          对话渲染（render_conversation）           │     │
│  │                                                    │     │
│  │  输入：对话 JSON → 输出：token IDs + 训练掩码       │     │
│  │                                                    │     │
│  │  <|bos|> <|user_start|> ... <|user_end|>         │     │
│  │  <|assistant_start|> ... <|assistant_end|>       │     │
│  │                                                    │     │
│  │  掩码策略：                                         │     │
│  │  - User 消息：mask=0（不训练）                     │     │
│  │  - Assistant 消息：mask=1（训练）                  │     │
│  │  - Python 输出：mask=0（运行时生成）               │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │              工具函数                               │     │
│  │  - get_tokenizer(): 加载默认分词器                 │     │
│  │  - get_token_bytes(): 加载 token 字节表示         │     │
│  │  - visualize_tokenization(): 调试可视化            │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 设计亮点

### 1. 双实现策略

提供两种分词器实现，允许用户根据需求选择：
- **开发/实验**：使用 HuggingFace（易于调试）
- **生产部署**：使用 RustBPE + tiktoken（高性能）

### 2. 清晰的掩码机制

`render_conversation()` 的掩码设计精妙：
- 只训练 assistant 生成的内容
- 跳过用户输入和工具输出
- 防止模型学习不应学习的部分

### 3. 工具调用支持

原生支持 Python REPL 工具调用：
- 特殊 token 标记代码和输出边界
- 区分训练目标（代码 vs 输出）
- 为未来扩展更多工具做准备

### 4. 性能优化

- `encode_special()` 使用 `@lru_cache` 缓存
- 批量编码支持多线程
- tiktoken 的 C++ 实现提供极速推理

### 5. 健壮性检查

- 严格验证消息顺序（user/assistant 交替）
- 断言检查防止格式错误
- 明确的错误消息

---

## 使用示例

### 训练自定义分词器

```python
from nanochat.tokenizer import RustBPETokenizer

# 准备文本数据
def text_iterator():
    with open("data.txt", "r") as f:
        for line in f:
            yield line.strip()

# 训练
tokenizer = RustBPETokenizer.train_from_iterator(
    text_iterator(),
    vocab_size=4096
)

# 保存
tokenizer.save("my_tokenizer")
```

### 加载和使用

```python
from nanochat.tokenizer import get_tokenizer

# 加载
tokenizer = get_tokenizer()

# 编码
text = "Hello, world!"
ids = tokenizer.encode(text, prepend="<|bos|>")
print(f"Token IDs: {ids}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# 解码
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")
```

### 渲染对话

```python
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
}

ids, mask = tokenizer.render_conversation(conversation)

# 可视化
viz = tokenizer.visualize_tokenization(ids, mask)
print(viz)
```

### 使用 OpenAI 预训练模型

```python
from nanochat.tokenizer import RustBPETokenizer

# 加载 GPT-2 分词器
tokenizer = RustBPETokenizer.from_pretrained("gpt2")

# 直接使用
ids = tokenizer.encode("Hello, GPT!")
```

---

## 常见问题

### Q1: 为什么默认使用 RustBPE 而不是 HuggingFace？

**A**: 性能。tiktoken 的推理速度远快于 HuggingFace Tokenizers，在大规模训练中差异显著。

### Q2: 为什么使用 `\p{N}{1,2}` 而不是 GPT-4 的 `\p{N}{1,3}`？

**A**: 作者猜测对于小词汇表（如 4096），3位数字会"浪费"token 空间。但这还未充分验证（见代码注释 TODO）。

### Q3: `<|bos|>` 和 `<|endoftext|>` 有什么区别？

**A**: 语义上没有区别，都用于标记文档开始。OpenAI 历史上使用 `<|endoftext|>`（命名令人困惑），nanochat 重命名为 `<|bos|>` 以更清晰。

### Q4: 为什么 Python 输出的 mask 是 0？

**A**: 因为 Python 输出是运行时由解释器生成的，不是模型生成的。训练时不应该让模型学习"生成"这些输出。

### Q5: 可以训练支持其他语言的分词器吗？

**A**: 可以！BPE 是语言无关的。只需使用目标语言的文本数据训练即可。Split pattern 可能需要调整以适应不同语言的特点。

---

## 与其他组件的关系

```
tokenizer.py
    ↓ 提供分词功能
dataset.py
    ↓ 生成 token 序列
dataloader.py
    ↓ 批量加载数据
engine.py
    ↓ 训练引擎
gpt.py (模型定义)
```

**关键连接点**：
- `dataset.py` 调用 `tokenizer.render_conversation()` 处理对话
- `gpt.py` 使用 `get_token_bytes()` 初始化嵌入层
- `dataloader.py` 处理 tokenizer 输出的 IDs 和 masks

---

## 性能考虑

### 训练阶段

- **rustbpe**：Rust 实现，训练速度快
- **并行化**：可以在多个进程中并行处理文本数据
- **内存效率**：流式处理，不需要一次性加载所有数据

### 推理阶段

- **tiktoken**：C++ 实现，极速
- **批量编码**：`encode_ordinary_batch()` 支持多线程
- **缓存**：`encode_special()` 使用 LRU 缓存避免重复计算

### 内存占用

```python
# 词汇表大小 vs 内存
vocab_size = 4096    # 小型：~几 MB
vocab_size = 32000   # 中型：~几十 MB
vocab_size = 100000  # 大型：~百 MB
```

主要内存占用来自：
1. `mergeable_ranks` 字典（token → rank）
2. 逆向映射（rank → token）
3. 预编译的正则表达式

---

## 总结

`tokenizer.py` 是 nanochat 的基础设施，提供了：

1. **GPT-4 风格的 BPE 分词**：兼容现代 LLM 的分词标准
2. **双实现方案**：灵活性和性能的平衡
3. **对话渲染机制**：将对话转换为训练数据，带智能掩码
4. **工具调用支持**：原生支持 Python REPL 等工具
5. **生产级质量**：使用 OpenAI 的 tiktoken，久经考验

理解这个模块是理解 nanochat 整个训练流程的关键第一步。接下来的 deep dive 文章将探讨如何使用这些 token 进行数据加载和模型训练。

---

## 延伸阅读

- [GPT-2 论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [BPE 原始论文](https://arxiv.org/abs/1508.07909)
- [tiktoken GitHub](https://github.com/openai/tiktoken)
- [HuggingFace Tokenizers 文档](https://huggingface.co/docs/tokenizers)
- [Unicode 正则表达式](https://www.regular-expressions.info/unicode.html)

---

**下一篇预告**：`dataset.py` - 数据集加载与处理
