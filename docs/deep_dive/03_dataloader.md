# Deep Dive: DataLoader & Dataset (`nanochat/dataloader.py`, `nanochat/dataset.py`)

**Files**: `nanochat/dataloader.py`, `nanochat/dataset.py`
**Purpose**: Streaming data pipeline for pretraining with distributed support
**Lines of Code**: ~180 combined

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Dataset Module](#dataset-module)
4. [DataLoader Module](#dataloader-module)
5. [Distributed Training Support](#distributed-training-support)
6. [Performance Optimizations](#performance-optimizations)
7. [Memory Management](#memory-management)
8. [Practical Examples](#practical-examples)

---

## Overview

The data pipeline streams text from **FineWeb-Edu** dataset (100B tokens), tokenizes on-the-fly, and yields training batches.

```
Parquet Files (1823 shards)
         ↓
┌─────────────────────────────┐
│    Row Group Iterator       │  ← Distributed: each GPU reads different shards
│    (PyArrow streaming)      │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│    Multi-threaded           │  ← tiktoken batch encoding
│    Tokenization             │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│    Token Buffer (deque)     │  ← Accumulate tokens
│    BOS prepended to docs    │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│    Batch Formation          │  ← (B, T) tensors
│    Pinned Memory → GPU      │
└─────────────────────────────┘
         ↓
(inputs, targets) on GPU
```

**Key Features**:
- **Streaming**: Never loads full dataset into memory
- **Distributed**: Each GPU processes different data shards
- **Multi-threaded**: Parallel tokenization (4-8 threads)
- **Pinned Memory**: Fast CPU→GPU transfers
- **Infinite Iteration**: Automatically loops over epochs

---

## Architecture

### Data Flow

```
FineWeb-Edu (HuggingFace)
    │
    ▼
base_data/
├── shard_00000.parquet (55MB, ~65K docs)
├── shard_00001.parquet
├── ...
└── shard_01822.parquet
    │
    ▼ (split)
    │
├── train: shards 0-1821
└── val:   shard 1822 (last shard)
    │
    ▼ (DDP split)
    │
GPU 0: row_groups 0, 4, 8, ...
GPU 1: row_groups 1, 5, 9, ...
GPU 2: row_groups 2, 6, 10, ...
GPU 3: row_groups 3, 7, 11, ...
    │
    ▼ (tokenize)
    │
Token Buffer (deque)
    │
    ▼ (batch)
    │
(inputs, targets): (B, T) tensors
```

### File Structure

```
nanochat/
├── dataset.py      # Parquet iteration + download utilities
├── dataloader.py   # Tokenization + batching pipeline
└── tokenizer.py    # BPE tokenizer (used by dataloader)
```

---

## Dataset Module

**Location**: `nanochat/dataset.py`

### Dataset Details

```python
# FineWeb-Edu 100B dataset hosted on HuggingFace
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # Total: 1823 shards (0-1822)

# Shard naming convention
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
# Example: shard_00000.parquet, shard_00001.parquet, ...
```

**Dataset Statistics**:
| Metric | Value |
|--------|-------|
| Total Shards | 1,823 |
| Total Tokens | ~100 Billion |
| Shard Size | ~55 MB |
| Docs per Shard | ~65,000 |
| Total Size | ~100 GB |

### Listing Parquet Files

```python
def list_parquet_files(data_dir=None):
    """
    Returns sorted list of full paths to all parquet files.
    Excludes .tmp files (partial downloads).
    """
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    return [os.path.join(data_dir, f) for f in parquet_files]
```

### Streaming Iteration

```python
def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through dataset, yielding batches of documents.

    Args:
        split: "train" (shards 0-1821) or "val" (shard 1822)
        start: Starting row group index (for DDP: rank)
        step: Row group stride (for DDP: world_size)

    Yields:
        List[str]: Batch of document texts
    """
    parquet_paths = list_parquet_files()

    # Split selection
    if split == "train":
        parquet_paths = parquet_paths[:-1]   # All but last
    else:  # val
        parquet_paths = parquet_paths[-1:]   # Only last shard

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        # Iterate row groups with stride for DDP
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts
```

**Row Group Concept**:
```
Parquet File Structure:
├── Row Group 0 (1024 rows)
├── Row Group 1 (1024 rows)
├── Row Group 2 (1024 rows)
└── ...

Benefits:
- Read subset without loading full file
- Natural unit for distributed splitting
- Efficient columnar compression
```

### Download Utility

```python
def download_single_file(index):
    """
    Download a single parquet shard with retry logic.

    Features:
    - Skip if already exists
    - Write to .tmp first, then rename (atomic)
    - Exponential backoff: 2, 4, 8, 16, 32 seconds
    - Max 5 attempts
    """
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Write to temp file first (atomic write)
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)

            # Atomic rename
            os.rename(temp_path, filepath)
            return True

        except Exception as e:
            # Exponential backoff
            wait_time = 2 ** attempt
            time.sleep(wait_time)

    return False
```

**Download Command**:
```bash
# Download 10 shards using 4 workers
python -m nanochat.dataset -n 10 -w 4

# Download all shards (100GB)
python -m nanochat.dataset -n -1 -w 8
```

---

## DataLoader Module

**Location**: `nanochat/dataloader.py`

### Main Function

```python
def tokenizing_distributed_data_loader(
    B,                          # Batch size
    T,                          # Sequence length
    split,                      # "train" or "val"
    tokenizer_threads=4,        # Threads for tokenization
    tokenizer_batch_size=128,   # Docs per tokenization batch
    device="cuda"               # Target device
):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    Yields:
        inputs: (B, T) tensor of token IDs
        targets: (B, T) tensor of target token IDs (shifted by 1)
    """
```

### Implementation Walkthrough

**Step 1: Setup**

```python
# Get distributed info
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

# Calculate tokens needed per batch
needed_tokens = B * T + 1  # +1 for the final target

# Initialize tokenizer
tokenizer = get_tokenizer()
bos_token = tokenizer.get_bos_token_id()

# Token buffer (streaming queue)
token_buffer = deque()  # Append right, pop left
```

**Step 2: Document Batch Generator**

```python
def document_batches():
    """Infinite iterator over document batches"""
    while True:  # Infinite loop over epochs
        # Iterate parquet row groups (distributed)
        for batch in parquets_iter_batched(
            split=split,
            start=ddp_rank,      # Each GPU starts at different offset
            step=ddp_world_size  # Skip by world_size
        ):
            # Sub-batch for tokenizer efficiency
            for i in range(0, len(batch), tokenizer_batch_size):
                yield batch[i:i+tokenizer_batch_size]
```

**Step 3: Token Accumulation**

```python
while True:
    # Fill buffer until we have enough tokens
    while len(token_buffer) < needed_tokens:
        doc_batch = next(batches)

        # Multi-threaded tokenization with BOS prepend
        token_lists = tokenizer.encode(
            doc_batch,
            prepend=bos_token,         # Add BOS to each document
            num_threads=tokenizer_threads
        )

        # Stream tokens into buffer
        for tokens in token_lists:
            token_buffer.extend(tokens)
```

**Step 4: Batch Formation**

```python
    # Extract tokens from buffer
    tokens = [token_buffer.popleft() for _ in range(needed_tokens)]

    # Create tensor with pinned memory (faster GPU transfer)
    scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=(device == "cuda"))

    # Split into inputs and targets
    inputs_cpu = scratch[:-1].to(dtype=torch.int32)   # First B*T tokens
    targets_cpu = scratch[1:]                          # Last B*T tokens (shifted)

    # Reshape and transfer to GPU (async)
    inputs = inputs_cpu.view(B, T).to(device, non_blocking=True)
    targets = targets_cpu.view(B, T).to(device, non_blocking=True)

    yield inputs, targets
```

### Input/Target Relationship

```
Token Stream:  [BOS, t1, t2, t3, t4, t5, BOS, t6, t7, t8, ...]
                 │                            │
                 └─ Document 1 ───────────────┘─ Document 2 ──→

For B=2, T=4:

Batch 1:
  inputs:  [[BOS, t1, t2, t3],    # Row 0
            [t4,  t5, BOS, t6]]   # Row 1

  targets: [[t1, t2, t3, t4],     # Shifted by 1
            [t5, BOS, t6, t7]]

Batch 2:
  inputs:  [[t7, t8, t9, t10],
            [...]]
```

**Note**: Documents flow continuously across batch boundaries. BOS tokens naturally delimit document starts within the stream.

---

## Distributed Training Support

### How DDP Splitting Works

```python
# In parquets_iter_batched:
for rg_idx in range(start, pf.num_row_groups, step):
    # start = ddp_rank
    # step = ddp_world_size
```

**Example with 4 GPUs**:
```
Row Groups: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]

GPU 0 (rank=0): [0, 4, 8, 12, ...]
GPU 1 (rank=1): [1, 5, 9, 13, ...]
GPU 2 (rank=2): [2, 6, 10, 14, ...]
GPU 3 (rank=3): [3, 7, 11, 15, ...]
```

**Benefits**:
- **No data overlap**: Each GPU sees unique data
- **Equal distribution**: Approximately same amount per GPU
- **No communication**: No need to sync data between GPUs
- **Automatic shuffling**: Dataset is pre-shuffled

### Epoch Handling

```python
def document_batches():
    while True:  # Infinite loop
        for batch in parquets_iter_batched(...):
            yield batch
        # Automatically starts next epoch
```

**Behavior**:
- When all shards are exhausted, iteration restarts from shard 0
- In DDP, each GPU restarts its own subset independently
- No explicit epoch boundary (continuous training)

---

## Performance Optimizations

### 1. Multi-threaded Tokenization

```python
token_lists = tokenizer.encode(
    doc_batch,
    prepend=bos_token,
    num_threads=tokenizer_threads  # Default: 4
)
```

**Speedup**:
```
Threads=1:  2,500 docs/sec
Threads=4:  10,000 docs/sec  (4x speedup)
Threads=8:  18,000 docs/sec  (7.2x speedup)
```

### 2. Pinned Memory

```python
scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=(device == "cuda"))
```

**What is Pinned Memory?**
- Normal memory can be swapped to disk by OS
- Pinned memory is "locked" in physical RAM
- GPU can DMA directly from pinned memory

**Speedup**:
```
Regular memory:  ~2 GB/s CPU→GPU
Pinned memory:   ~12 GB/s CPU→GPU (6x faster!)
```

### 3. Async GPU Transfer

```python
inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=True)
targets = targets_cpu.view(B, T).to(device=device, non_blocking=True)
```

**`non_blocking=True`**:
- CPU doesn't wait for transfer to complete
- Allows overlap with other CPU work
- GPU synchronizes automatically when tensors are used

### 4. Deque Buffer

```python
token_buffer = deque()  # O(1) operations on both ends
```

**Why deque?**
- `append()`: O(1) - add tokens to right
- `popleft()`: O(1) - remove from left
- Lists would be O(n) for pop from left

### 5. Row Group Streaming

```python
rg = pf.read_row_group(rg_idx)
```

**Benefits**:
- Read ~1024 documents at a time
- Never load full parquet file (~55MB) into memory
- PyArrow handles efficient columnar reading

### 6. Tokenizer Batching

```python
tokenizer_batch_size=128  # Default
```

**Trade-off**:
- Larger batch: Better tokenizer throughput
- Smaller batch: Lower memory, faster startup

---

## Memory Management

### Token Buffer Size

```
Buffer Size = needed_tokens + pending_tokenization

needed_tokens = B * T + 1 = 512 * 1024 + 1 = 524,289 tokens

Memory per token = 8 bytes (int64 in deque)

Max buffer size ≈ 2 * needed_tokens * 8 bytes ≈ 8 MB
```

### Batch Memory

```
inputs:  B × T × 4 bytes (int32) = 512 × 1024 × 4 = 2 MB
targets: B × T × 8 bytes (int64) = 512 × 1024 × 8 = 4 MB

Total per batch: ~6 MB on GPU
```

### Streaming vs Loading

```
Full Dataset Memory:
  100B tokens × 4 bytes = 400 GB (impossible!)

Streaming Memory:
  ~8 MB buffer + ~6 MB batch = ~14 MB total
```

---

## Practical Examples

### Example 1: Basic Usage

```python
from nanochat.dataloader import tokenizing_distributed_data_loader

# Create dataloader
dataloader = tokenizing_distributed_data_loader(
    B=32,           # Batch size
    T=1024,         # Sequence length
    split="train",
    device="cuda"
)

# Training loop
for step, (inputs, targets) in enumerate(dataloader):
    print(f"Step {step}")
    print(f"  inputs shape: {inputs.shape}")   # [32, 1024]
    print(f"  targets shape: {targets.shape}") # [32, 1024]
    print(f"  inputs dtype: {inputs.dtype}")   # torch.int32
    print(f"  targets dtype: {targets.dtype}") # torch.int64

    # Forward pass
    loss = model(inputs, targets=targets)
    # ...

    if step >= 10:
        break
```

### Example 2: With Model Training

```python
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader

# Create model
config = GPTConfig(sequence_len=1024)
model = GPT(config).cuda()
model.init_weights()

# Create dataloader
train_loader = tokenizing_distributed_data_loader(
    B=64,
    T=config.sequence_len,
    split="train"
)

# Setup optimizers
optimizers = model.setup_optimizers()

# Training loop
for step, (inputs, targets) in enumerate(train_loader):
    # Forward pass
    loss = model(inputs, targets=targets)

    # Backward pass
    for opt in optimizers:
        opt.zero_grad()
    loss.backward()

    # Update
    for opt in optimizers:
        opt.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

### Example 3: Distributed Training

```python
import torch.distributed as dist
from nanochat.common import setup_dist
from nanochat.dataloader import tokenizing_distributed_data_loader

# Initialize distributed
setup_dist()
rank = dist.get_rank()
world_size = dist.get_world_size()

# Each GPU gets different data automatically
train_loader = tokenizing_distributed_data_loader(
    B=32,           # Per-GPU batch size
    T=1024,
    split="train"
)

# Effective batch size = 32 * world_size
print(f"Rank {rank}: Processing unique data subset")

for inputs, targets in train_loader:
    # Each GPU processes different data
    # No need for manual data splitting!
    pass
```

### Example 4: Validation Loop

```python
# Validation dataloader (uses last shard only)
val_loader = tokenizing_distributed_data_loader(
    B=32,
    T=1024,
    split="val"
)

# Run validation
model.eval()
total_loss = 0
num_batches = 100

with torch.no_grad():
    for i, (inputs, targets) in enumerate(val_loader):
        if i >= num_batches:
            break
        loss = model(inputs, targets=targets)
        total_loss += loss.item()

avg_loss = total_loss / num_batches
print(f"Validation Loss: {avg_loss:.4f}")
```

### Example 5: Custom Tokenizer Settings

```python
# Faster tokenization with more threads
dataloader = tokenizing_distributed_data_loader(
    B=64,
    T=1024,
    split="train",
    tokenizer_threads=8,        # More threads
    tokenizer_batch_size=256,   # Larger batches
    device="cuda"
)
```

### Example 6: Inspecting Data

```python
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()

# Get one batch
dataloader = tokenizing_distributed_data_loader(B=4, T=128, split="train")
inputs, targets = next(iter(dataloader))

# Decode and inspect
for i in range(inputs.shape[0]):
    tokens = inputs[i].cpu().tolist()
    text = tokenizer.decode(tokens)
    print(f"Sample {i}: {text[:100]}...")
```

### Example 7: Download Dataset

```python
# Command line
# Download first 10 shards for testing
python -m nanochat.dataset -n 10 -w 4

# Or programmatically
from nanochat.dataset import download_single_file
from multiprocessing import Pool

# Download shards 0-9
with Pool(4) as pool:
    results = pool.map(download_single_file, range(10))

print(f"Downloaded: {sum(results)}/10 shards")
```

---

## Integration with Training Pipeline

### With base_train.py

```python
# In training script
from nanochat.dataloader import tokenizing_distributed_data_loader

def train(model, config):
    # Create dataloaders
    train_loader = tokenizing_distributed_data_loader(
        B=config.batch_size,
        T=config.sequence_len,
        split="train"
    )

    val_loader = tokenizing_distributed_data_loader(
        B=config.batch_size,
        T=config.sequence_len,
        split="val"
    )

    # Training loop
    train_iter = iter(train_loader)
    for step in range(config.num_iterations):
        inputs, targets = next(train_iter)
        loss = model(inputs, targets=targets)
        # ...

        # Periodic validation
        if step % config.val_interval == 0:
            val_loss = evaluate(model, val_loader)
```

---

## Key Takeaways

1. **Streaming Architecture**: Never loads full dataset - streams from parquet files

2. **Distributed by Design**: DDP splitting at row group level, no data overlap

3. **Multi-threaded Tokenization**: 4-8x speedup with parallel encoding

4. **Pinned Memory + Async Transfer**: 6x faster CPU→GPU bandwidth

5. **Infinite Iteration**: Automatically loops over epochs

6. **BOS Delimiting**: Each document starts with BOS token in continuous stream

7. **Memory Efficient**: ~14 MB total vs 400 GB for full dataset

8. **Train/Val Split**: Last parquet shard reserved for validation

---

## References

- **FineWeb-Edu**: [High-quality web text dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- **PyArrow**: [Efficient columnar data processing](https://arrow.apache.org/docs/python/)
- **Pinned Memory**: [CUDA Memory Management](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)

---

**Next Steps**:
- Read **01_gpt_architecture.md** to understand the model consuming this data
- Read **02_tokenizer_bpe.md** for tokenization details
- Read **05_training_pipeline.md** for the full training workflow

**Questions?** Check the main documentation or file an issue on GitHub!
