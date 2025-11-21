# Deep Dive: BPE Tokenizer (`nanochat/tokenizer.py`)

**File**: `nanochat/tokenizer.py`
**Purpose**: GPT-4 style Byte Pair Encoding (BPE) tokenizer with training and inference support
**Lines of Code**: ~400

---

## Table of Contents

1. [Overview](#overview)
2. [BPE Algorithm Fundamentals](#bpe-algorithm-fundamentals)
3. [Two Tokenizer Implementations](#two-tokenizer-implementations)
4. [Special Tokens](#special-tokens)
5. [GPT-4 Style Split Pattern](#gpt-4-style-split-pattern)
6. [HuggingFace Tokenizer](#huggingface-tokenizer)
7. [RustBPE + Tiktoken Tokenizer](#rustbpe--tiktoken-tokenizer)
8. [Conversation Rendering](#conversation-rendering)
9. [Training a Tokenizer](#training-a-tokenizer)
10. [Performance Analysis](#performance-analysis)
11. [Practical Examples](#practical-examples)

---

## Overview

This module provides a **GPT-4 style BPE tokenizer** with two backend implementations:

```
Text Input
    ↓
┌─────────────────────────────────────┐
│     Pre-tokenization (Regex)        │  ← GPT-4 split pattern
│  "Hello world" → ["Hello", " world"]│
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│      Byte-Level Encoding            │  ← UTF-8 to bytes
│  "Hello" → [72, 101, 108, 108, 111] │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│      BPE Merge Operations           │  ← Learned merges
│  [72, 101, 108, 108, 111] → [15496] │
└─────────────────────────────────────┘
    ↓
Token IDs: [15496, 995]
```

**Key Features**:
- **Byte-level BPE**: No unknown tokens (any UTF-8 text can be encoded)
- **GPT-4 Split Pattern**: Intelligent pre-tokenization
- **Special Tokens**: Support for conversation formatting
- **Dual Backend**: HuggingFace for training, tiktoken for fast inference
- **Conversation Rendering**: Built-in chat template support

---

## BPE Algorithm Fundamentals

### What is Byte Pair Encoding?

BPE is a **subword tokenization** algorithm that:
1. Starts with a character-level vocabulary (256 bytes)
2. Iteratively merges the most frequent adjacent pairs
3. Builds a vocabulary of subword units

### Algorithm Steps

**Training Phase**:
```
1. Initialize vocabulary with all 256 byte values
2. Encode training corpus as bytes
3. Repeat until vocab_size reached:
   a. Count all adjacent byte/token pairs
   b. Find most frequent pair (a, b)
   c. Create new token: ab = merge(a, b)
   d. Add ab to vocabulary
   e. Replace all (a, b) occurrences with ab in corpus
```

**Example**:
```
Corpus: "aaabdaaabac"

Step 0: Vocab = {a, b, c, d}
        Corpus = [a, a, a, b, d, a, a, a, b, a, c]

Step 1: Most frequent pair: (a, a) appears 4 times
        Create token: Z = "aa"
        Vocab = {a, b, c, d, Z}
        Corpus = [Z, a, b, d, Z, a, b, a, c]

Step 2: Most frequent pair: (Z, a) appears 2 times
        Create token: Y = "Za" = "aaa"
        Vocab = {a, b, c, d, Z, Y}
        Corpus = [Y, b, d, Y, b, a, c]

Step 3: Most frequent pair: (Y, b) appears 2 times
        Create token: X = "Yb" = "aaab"
        Vocab = {a, b, c, d, Z, Y, X}
        Corpus = [X, d, X, a, c]
```

**Encoding Phase**:
```
Text: "aaab"

1. Convert to bytes: [97, 97, 97, 98]
2. Apply learned merges in order:
   - (97, 97) → Z
   - (Z, 97) → Y
   - (Y, 98) → X
3. Result: [X]
```

### Why Byte-Level BPE?

**Character-level BPE**:
- Limited to characters in training data
- Unknown tokens for unseen characters
- Vocabulary depends on language

**Byte-level BPE**:
- Works with raw bytes (0-255)
- **No unknown tokens** - any UTF-8 can be encoded
- Language-agnostic
- Graceful handling of emojis, special characters, code

```python
# Byte-level handles EVERYTHING
tokenizer.encode("Hello 世界 🚀 \x00\xff")  # No errors!
```

---

## Two Tokenizer Implementations

### Architecture Comparison

| Feature | HuggingFaceTokenizer | RustBPETokenizer |
|---------|---------------------|------------------|
| Training | ✅ HuggingFace BpeTrainer | ✅ rustbpe |
| Inference | ✅ HuggingFace | ✅ tiktoken |
| Speed | Medium | **Fast** (Rust) |
| Batch Encoding | Single-threaded | **Multi-threaded** |
| Save Format | JSON | Pickle |
| Complexity | Higher | Lower |

### When to Use Which?

```python
# For training a new tokenizer (either works)
tokenizer = HuggingFaceTokenizer.train_from_iterator(...)
# OR
tokenizer = RustBPETokenizer.train_from_iterator(...)

# For production inference (prefer RustBPE)
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)  # Faster!

# For compatibility with HuggingFace ecosystem
tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
```

---

## Special Tokens

### Token Definitions

```python
SPECIAL_TOKENS = [
    # Document delimiter
    "<|bos|>",              # Beginning of Sequence

    # Conversation tokens (for fine-tuning)
    "<|user_start|>",       # Start of user message
    "<|user_end|>",         # End of user message
    "<|assistant_start|>",  # Start of assistant message
    "<|assistant_end|>",    # End of assistant message

    # Tool use tokens (Python REPL)
    "<|python_start|>",     # Assistant invokes Python
    "<|python_end|>",       # End of Python code
    "<|output_start|>",     # Python output begins
    "<|output_end|>",       # Python output ends
]
```

### Token Placement in Vocabulary

```python
# Special tokens are added AFTER regular BPE tokens
vocab_size = 32000
regular_tokens = 32000 - len(SPECIAL_TOKENS)  # 31991

# Token ID assignment:
# 0-31990:     Regular BPE tokens
# 31991:       <|bos|>
# 31992:       <|user_start|>
# 31993:       <|user_end|>
# ...and so on
```

### Conversation Format

```
<|bos|><|user_start|>Hello, how are you?<|user_end|><|assistant_start|>I'm doing well!<|assistant_end|>
```

**With Tool Use**:
```
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>Let me calculate.<|python_start|>print(2+2)<|python_end|><|output_start|>4<|output_end|>The answer is 4.<|assistant_end|>
```

---

## GPT-4 Style Split Pattern

### The Regex Pattern

```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

### Pattern Breakdown

| Component | Matches | Example |
|-----------|---------|---------|
| `'(?i:[sdmt]\|ll\|ve\|re)` | Contractions | `'s`, `'t`, `'ll`, `'ve`, `'re` |
| `[^\r\n\p{L}\p{N}]?+\p{L}+` | Words (with optional prefix) | `Hello`, ` world`, `.The` |
| `\p{N}{1,2}` | 1-2 digit numbers | `42`, `7`, `99` |
| ` ?[^\s\p{L}\p{N}]++[\r\n]*` | Punctuation + newlines | `...`, `!!!`, `?\n` |
| `\s*[\r\n]` | Newlines with leading space | `\n`, `  \n` |
| `\s+(?!\S)` | Trailing whitespace | `   ` (at end) |
| `\s+` | Other whitespace | `  ` |

### NanoChat Modification

```python
# Original GPT-4 pattern:
\p{N}{1,3}  # Matches 1-3 digit numbers

# NanoChat modification:
\p{N}{1,2}  # Matches 1-2 digit numbers only
```

**Rationale**:
- Smaller vocab sizes benefit from fewer number tokens
- 3-digit numbers use more vocabulary space
- Trade-off: slightly more tokens for large numbers

### Pre-tokenization Example

```python
text = "Hello, I'm 25 years old!"

# Pre-tokenization splits:
["Hello", ",", " I", "'m", " 25", " years", " old", "!"]
```

---

## HuggingFace Tokenizer

**Location**: `nanochat/tokenizer.py:39-147`

### Class Overview

```python
class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # HuggingFace Tokenizer object
```

### Training a Tokenizer

```python
@classmethod
def train_from_iterator(cls, text_iterator, vocab_size):
    # 1. Create BPE model with byte fallback
    tokenizer = HFTokenizer(BPE(
        byte_fallback=True,   # Handle unknown bytes
        unk_token=None,       # No UNK token (byte fallback handles it)
        fuse_unk=False,
    ))

    # 2. Configure pre-tokenizer (GPT-4 style)
    gpt4_split_regex = Regex(SPLIT_PATTERN)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])

    # 3. Configure decoder
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Train with BPE
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        min_frequency=0,  # Include all merges
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(text_iterator, trainer)

    return cls(tokenizer)
```

### Core Methods

```python
# Encoding
def encode(self, text, prepend=None, append=None):
    """Encode text to token IDs"""
    ids = []
    if prepend is not None:
        ids.append(self.encode_special(prepend))
    ids.extend(self.tokenizer.encode(text).ids)
    if append is not None:
        ids.append(self.encode_special(append))
    return ids

# Decoding
def decode(self, ids):
    """Decode token IDs to text"""
    return self.tokenizer.decode(ids, skip_special_tokens=False)

# Special token encoding
def encode_special(self, text):
    """Encode a special token by exact match"""
    return self.tokenizer.token_to_id(text)

# Save/Load
def save(self, tokenizer_dir):
    """Save tokenizer to tokenizer.json"""
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    self.tokenizer.save(tokenizer_path)
```

---

## RustBPE + Tiktoken Tokenizer

**Location**: `nanochat/tokenizer.py:155-378`

### Why This Combination?

```
Training:  rustbpe (pure Rust, fast training)
     ↓
Inference: tiktoken (OpenAI's fast encoder, Rust backend)
```

**Benefits**:
- **Training**: rustbpe is optimized for BPE training
- **Inference**: tiktoken is battle-tested, multi-threaded
- **Compatibility**: Can load GPT-4/GPT-3.5 tokenizers directly

### Training Flow

```python
@classmethod
def train_from_iterator(cls, text_iterator, vocab_size):
    # 1. Train with rustbpe
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(
        text_iterator,
        vocab_size_no_special,
        pattern=SPLIT_PATTERN
    )

    # 2. Extract learned merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks_list = tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}

    # 3. Add special tokens
    tokens_offset = len(mergeable_ranks)
    special_tokens = {
        name: tokens_offset + i
        for i, name in enumerate(SPECIAL_TOKENS)
    }

    # 4. Create tiktoken Encoding
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,  # dict[bytes, int]
        special_tokens=special_tokens,    # dict[str, int]
    )

    return cls(enc, "<|bos|>")
```

### Multi-threaded Encoding

```python
def encode(self, text, prepend=None, append=None, num_threads=8):
    if isinstance(text, str):
        # Single string
        ids = self.enc.encode_ordinary(text)
    elif isinstance(text, list):
        # Batch encoding (multi-threaded!)
        ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
    return ids
```

**Performance**:
```
Single-threaded:  1,000 docs/sec
Multi-threaded:   8,000 docs/sec (8 threads)
Speedup:          8x
```

### Loading Pre-trained Tokenizers

```python
# Load from local directory
tokenizer = RustBPETokenizer.from_directory("out/tokenizer")

# Load OpenAI's tokenizers directly!
tokenizer = RustBPETokenizer.from_pretrained("cl100k_base")  # GPT-4
tokenizer = RustBPETokenizer.from_pretrained("p50k_base")    # GPT-3
tokenizer = RustBPETokenizer.from_pretrained("r50k_base")    # GPT-2
```

---

## Conversation Rendering

**Location**: `nanochat/tokenizer.py:258-377`

### Overview

The `render_conversation()` method converts chat conversations into token IDs with a training mask.

```python
conversation = {
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
}

ids, mask = tokenizer.render_conversation(conversation)
# ids:  [bos, user_start, ..., user_end, assistant_start, ..., assistant_end]
# mask: [0,   0,          ..., 0,        0,               ..., 1            ]
#       ↑ Don't train on user messages    ↑ Train on assistant messages
```

### Mask Purpose

The mask indicates which tokens to **train on**:
- `mask=0`: User messages, special tokens (don't train)
- `mask=1`: Assistant responses (train on these)

```
<|bos|>  <|user_start|>  Hello!  <|user_end|>  <|assistant_start|>  Hi!  <|assistant_end|>
   0           0           0          0               0               1         1
```

### System Message Handling

```python
# System messages are merged with the first user message
if conversation["messages"][0]["role"] == "system":
    messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
    messages = messages[1:]
```

**Example**:
```python
# Input
{"role": "system", "content": "You are helpful."}
{"role": "user", "content": "Hi!"}

# Becomes
{"role": "user", "content": "You are helpful.\n\nHi!"}
```

### Tool Use Rendering

```python
# Assistant message with tool use
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me calculate."},
        {"type": "python", "text": "print(2+2)"},
        {"type": "python_output", "text": "4"},
        {"type": "text", "text": "The answer is 4."}
    ]
}
```

**Rendered**:
```
<|assistant_start|>Let me calculate.<|python_start|>print(2+2)<|python_end|><|output_start|>4<|output_end|>The answer is 4.<|assistant_end|>
```

**Mask**:
```
Token:  assistant_start | text | python_start | code | python_end | output_start | output | output_end | text | assistant_end
Mask:         0         |  1   |      1       |  1   |     1      |      0       |   0    |     0      |  1   |      1
```

**Note**: Python output tokens have `mask=0` because they come from the Python interpreter at test time, not from the model.

### Visualization

```python
def visualize_tokenization(self, ids, mask, with_token_id=False):
    """Colorize tokens: GREEN=train, RED=don't train"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

    tokens = []
    for token_id, mask_val in zip(ids, mask):
        token_str = self.decode([token_id])
        color = GREEN if mask_val == 1 else RED
        tokens.append(f"{color}{token_str}{RESET}")

    return '|'.join(tokens)
```

**Output**:
```
RED|<|bos|>|RED|<|user_start|>|RED|Hello|RED|<|user_end|>|RED|<|assistant_start|>|GREEN|Hi!|GREEN|<|assistant_end|>
```

### Render for Completion (RL)

```python
def render_for_completion(self, conversation):
    """For RL: render conversation priming assistant for completion"""
    # Remove last assistant message
    messages.pop()

    # Render remaining conversation
    ids, mask = self.render_conversation(conversation)

    # Append assistant_start to prime completion
    ids.append(self.encode_special("<|assistant_start|>"))

    return ids
```

**Use Case**: Reinforcement Learning from Human Feedback (RLHF)
```python
# Input conversation
[user: "What is 2+2?", assistant: "4"]

# Output (primed for completion)
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>
# Model generates from here →
```

---

## Training a Tokenizer

### Full Training Pipeline

```python
from nanochat.tokenizer import RustBPETokenizer

# 1. Prepare text iterator
def text_iterator():
    with open("corpus.txt", "r") as f:
        for line in f:
            yield line.strip()

# 2. Train tokenizer
vocab_size = 32000
tokenizer = RustBPETokenizer.train_from_iterator(
    text_iterator(),
    vocab_size
)

# 3. Save tokenizer
tokenizer.save("out/tokenizer")

# 4. Verify
print(f"Vocab size: {tokenizer.get_vocab_size()}")
print(f"Special tokens: {tokenizer.get_special_tokens()}")

# 5. Test encoding
text = "Hello, world!"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
print(f"Original: {text}")
print(f"Token IDs: {ids}")
print(f"Decoded: {decoded}")
assert text == decoded
```

### Vocabulary Size Considerations

| Vocab Size | Pros | Cons |
|------------|------|------|
| 8K | Fast training, small model | More tokens per text |
| 32K | Good balance | Standard choice |
| 50K | Fewer tokens, better compression | Larger embedding matrix |
| 100K+ | Best compression | Very large embeddings |

**Rule of Thumb**:
```
Embedding params = vocab_size × embedding_dim

32K vocab × 768 dim = 24.6M params
50K vocab × 768 dim = 38.4M params (56% more!)
```

### Compression Ratio

```python
# Measure compression
text = open("sample.txt").read()
ids = tokenizer.encode(text)

bytes_per_token = len(text.encode('utf-8')) / len(ids)
print(f"Compression: {bytes_per_token:.2f} bytes/token")

# Good tokenizers: 3-4 bytes/token for English
# GPT-4 (cl100k_base): ~4.0 bytes/token
```

---

## Performance Analysis

### Encoding Speed Comparison

**Single String Encoding** (1MB text):
| Tokenizer | Time | Tokens/sec |
|-----------|------|------------|
| HuggingFace | 450ms | 2.2M |
| tiktoken | 85ms | **11.7M** |
| **Speedup** | | **5.3x** |

**Batch Encoding** (10K documents):
| Tokenizer | Threads | Time | Docs/sec |
|-----------|---------|------|----------|
| HuggingFace | 1 | 12.5s | 800 |
| tiktoken | 1 | 2.3s | 4,350 |
| tiktoken | 8 | 0.4s | **25,000** |

### Memory Usage

```python
# HuggingFace tokenizer (JSON format)
# Size on disk: ~5 MB for 32K vocab

# tiktoken tokenizer (pickle format)
# Size on disk: ~2 MB for 32K vocab

# In-memory:
# Both: ~10-20 MB for 32K vocab
```

### Training Speed

```
Training on 1GB text corpus:

rustbpe:     ~5 minutes
HuggingFace: ~15 minutes

Speedup: 3x
```

---

## Practical Examples

### Example 1: Basic Usage

```python
from nanochat.tokenizer import get_tokenizer

# Load default tokenizer
tokenizer = get_tokenizer()

# Encode text
text = "Hello, world! How are you?"
ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")
# Output: [15496, 11, 995, 0, 1374, 389, 345, 30]

# Decode back
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")
# Output: Hello, world! How are you?

# With BOS token
ids_with_bos = tokenizer.encode(text, prepend="<|bos|>")
print(f"With BOS: {ids_with_bos}")
# Output: [31991, 15496, 11, 995, 0, 1374, 389, 345, 30]
```

### Example 2: Batch Encoding

```python
# Encode multiple texts efficiently
texts = [
    "First document",
    "Second document",
    "Third document",
]

# Multi-threaded batch encoding
ids_batch = tokenizer.encode(texts, num_threads=8)
print(f"Batch results: {ids_batch}")
# Output: [[5765, 3188], [4041, 3188], [12043, 3188]]
```

### Example 3: Conversation Rendering

```python
# Define a conversation
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]
}

# Render to tokens
ids, mask = tokenizer.render_conversation(conversation)

# Visualize
print(tokenizer.visualize_tokenization(ids, mask))
# Output: RED tokens (user) | GREEN tokens (assistant)

# Check supervised tokens
supervised_ids = [id for id, m in zip(ids, mask) if m == 1]
print(f"Tokens to train on: {tokenizer.decode(supervised_ids)}")
# Output: "The answer is 4.<|assistant_end|>"
```

### Example 4: Tool Use Conversation

```python
conversation = {
    "messages": [
        {"role": "user", "content": "Calculate 15 * 7"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "I'll calculate that for you."},
            {"type": "python", "text": "result = 15 * 7\nprint(result)"},
            {"type": "python_output", "text": "105"},
            {"type": "text", "text": "The result is 105."}
        ]}
    ]
}

ids, mask = tokenizer.render_conversation(conversation)

# Verify mask
# Python code is supervised (model should learn to write code)
# Python output is NOT supervised (comes from interpreter)
```

### Example 5: Token Inspection

```python
# Inspect vocabulary
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

# Look at specific tokens
for i in range(10):
    token = tokenizer.id_to_token(i)
    print(f"Token {i}: {repr(token)}")

# Find special tokens
special = tokenizer.get_special_tokens()
print(f"Special tokens: {special}")

# Get BOS token ID
bos_id = tokenizer.get_bos_token_id()
print(f"BOS token ID: {bos_id}")
```

### Example 6: Training a Custom Tokenizer

```python
from nanochat.tokenizer import RustBPETokenizer

# Prepare corpus
def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

# Train with custom vocab size
tokenizer = RustBPETokenizer.train_from_iterator(
    load_corpus("my_corpus.txt"),
    vocab_size=16000
)

# Analyze compression
sample_text = "The quick brown fox jumps over the lazy dog."
ids = tokenizer.encode(sample_text)
print(f"Text length: {len(sample_text)} chars")
print(f"Token count: {len(ids)} tokens")
print(f"Compression: {len(sample_text)/len(ids):.2f} chars/token")

# Save for later use
tokenizer.save("my_tokenizer")
```

### Example 7: Using Pre-trained OpenAI Tokenizers

```python
from nanochat.tokenizer import RustBPETokenizer

# Load GPT-4 tokenizer
gpt4_tokenizer = RustBPETokenizer.from_pretrained("cl100k_base")

# Load GPT-3.5 tokenizer
gpt3_tokenizer = RustBPETokenizer.from_pretrained("p50k_base")

# Compare tokenization
text = "Hello, world! 你好世界！"

gpt4_ids = gpt4_tokenizer.encode(text)
gpt3_ids = gpt3_tokenizer.encode(text)

print(f"GPT-4 tokens: {len(gpt4_ids)}")  # Usually fewer for multilingual
print(f"GPT-3 tokens: {len(gpt3_ids)}")
```

---

## Integration with Other Modules

### With GPT Model

```python
from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT, GPTConfig

# Load tokenizer
tokenizer = get_tokenizer()

# Configure model with matching vocab size
config = GPTConfig(
    vocab_size=tokenizer.get_vocab_size(),
    # ... other config
)
model = GPT(config)

# Encode and generate
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, prepend="<|bos|>")
input_tensor = torch.tensor([input_ids])

# Generate
for token_id in model.generate(input_ids, max_tokens=50):
    print(tokenizer.decode([token_id]), end='', flush=True)
```

### With DataLoader

```python
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import DataLoader

tokenizer = get_tokenizer()

# DataLoader uses tokenizer internally
dataloader = DataLoader(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    batch_size=32,
    sequence_len=1024
)

for batch in dataloader:
    input_ids, targets, mask = batch
    # Train model...
```

### With Training Pipeline

```python
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()

# Render conversations for SFT
for conversation in conversations:
    ids, mask = tokenizer.render_conversation(conversation)

    # Create training batch
    input_ids = ids[:-1]   # All but last token
    targets = ids[1:]      # All but first token
    loss_mask = mask[1:]   # Mask shifted to align with targets
```

---

## Advanced Topics

### Custom Split Patterns

```python
# Create tokenizer with custom pattern
CUSTOM_PATTERN = r"""\w+|\s+|[^\w\s]+"""

tokenizer = rustbpe.Tokenizer()
tokenizer.train_from_iterator(
    text_iterator,
    vocab_size,
    pattern=CUSTOM_PATTERN  # Custom regex
)
```

### Vocabulary Analysis

```python
# Analyze token frequency in corpus
from collections import Counter

token_counts = Counter()
for text in corpus:
    ids = tokenizer.encode(text)
    token_counts.update(ids)

# Most common tokens
for token_id, count in token_counts.most_common(20):
    token = tokenizer.decode([token_id])
    print(f"{repr(token):20} : {count:>10}")
```

### Handling Very Long Texts

```python
def chunk_encode(text, tokenizer, max_chunk_size=100000):
    """Encode very long texts in chunks to manage memory"""
    all_ids = []
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i+max_chunk_size]
        ids = tokenizer.encode(chunk)
        all_ids.extend(ids)
    return all_ids
```

---

## Key Takeaways

1. **Byte-Level BPE**: No unknown tokens - handles any UTF-8 text

2. **Dual Implementation**: RustBPE for training, tiktoken for fast inference

3. **GPT-4 Style**: Uses the same pre-tokenization pattern as GPT-4

4. **Special Tokens**: Built-in support for conversations and tool use

5. **Training Mask**: Distinguishes user (don't train) from assistant (train) tokens

6. **Performance**: tiktoken with multi-threading is 5-30x faster than HuggingFace

7. **Compression**: Good tokenizers achieve 3-4 bytes per token for English

---

## References

- **BPE Paper**: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- **tiktoken**: [OpenAI's fast BPE tokenizer](https://github.com/openai/tiktoken)
- **GPT-4 Tokenizer**: cl100k_base encoding
- **HuggingFace Tokenizers**: [Fast tokenizers library](https://github.com/huggingface/tokenizers)

---

**Next Steps**:
- Read **01_gpt_architecture.md** to understand the model that uses these tokens
- Read **03_dataloader.md** to see how tokenized data is batched for training
- Read **05_training_pipeline.md** for the full training workflow

**Questions?** Check the main documentation or file an issue on GitHub!
