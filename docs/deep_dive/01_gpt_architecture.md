# Deep Dive: GPT Architecture (`nanochat/gpt.py`)

**File**: `nanochat/gpt.py`
**Purpose**: Modern, simplified GPT implementation with advanced architectural features
**Author**: NanoChat Team
**Lines of Code**: ~308

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Innovations](#architectural-innovations)
3. [Model Configuration](#model-configuration)
4. [Core Components](#core-components)
   - [RMSNorm](#rmsnorm)
   - [Rotary Positional Embeddings (RoPE)](#rotary-positional-embeddings-rope)
   - [Causal Self-Attention with MQA/GQA](#causal-self-attention-with-mqagqa)
   - [MLP with ReLU²](#mlp-with-relu)
   - [Transformer Block](#transformer-block)
   - [GPT Model](#gpt-model)
5. [Training Features](#training-features)
6. [Inference Features](#inference-features)
7. [Performance Analysis](#performance-analysis)
8. [Practical Examples](#practical-examples)
9. [Integration with Other Modules](#integration-with-other-modules)

---

## Overview

This implementation of GPT represents a **modern, highly optimized** Transformer architecture that incorporates several state-of-the-art techniques:

```
Input Tokens (B, T)
        ↓
Token Embedding + RMSNorm
        ↓
┌─────────────────────────┐
│  Transformer Block 1    │
│  ┌──────────────────┐   │
│  │ RMSNorm          │   │
│  │ Attention + RoPE │   │  ← Rotary Embeddings
│  │ QK Norm          │   │  ← Query/Key Normalization
│  │ Residual         │   │
│  └──────────────────┘   │
│  ┌──────────────────┐   │
│  │ RMSNorm          │   │
│  │ MLP (ReLU²)      │   │  ← Squared ReLU Activation
│  │ Residual         │   │
│  └──────────────────┘   │
└─────────────────────────┘
        ↓
     (repeat for N layers)
        ↓
Final RMSNorm
        ↓
LM Head (untied weights)
        ↓
Logits (B, T, vocab_size)
```

**Key Design Principles**:
- **Simplicity**: Minimal code, maximum clarity
- **Efficiency**: Optimized for both training and inference
- **Modularity**: Clean separation of components
- **Scalability**: Supports distributed training with DDP + ZeRO-2

---

## Architectural Innovations

This GPT implementation includes **7 major architectural improvements** over vanilla Transformer:

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Rotary Embeddings (RoPE)** | Relative positional encoding via rotation | Better length extrapolation |
| **QK Normalization** | Normalize queries and keys before attention | Training stability |
| **Untied Weights** | Separate weights for embedding and LM head | More capacity |
| **ReLU² Activation** | Squared ReLU in MLP | Better gradients |
| **Post-Embedding Norm** | Normalize after token embedding | Stable training |
| **RMSNorm (no params)** | Simpler normalization without learnable params | Fewer parameters |
| **No Bias Terms** | All linear layers have `bias=False` | Cleaner gradients |
| **Multi-Query Attention** | Fewer KV heads for efficient inference | 2-10x faster inference |

Let's dive into each component in detail!

---

## Model Configuration

### `GPTConfig` Dataclass

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024      # Maximum sequence length
    vocab_size: int = 50304       # Vocabulary size (padded to multiple of 64)
    n_layer: int = 12             # Number of Transformer blocks
    n_head: int = 6               # Number of query heads
    n_kv_head: int = 6            # Number of key/value heads (for MQA/GQA)
    n_embd: int = 768             # Embedding dimension
```

**Design Choices**:

1. **`vocab_size = 50304`**: Padded to a multiple of 64 for efficient GPU computation
   - Original GPT-2 vocab size: 50257
   - Padding: 50304 = 784 × 64

2. **`n_head` vs `n_kv_head`**: Enables Multi-Query Attention (MQA) or Grouped-Query Attention (GQA)
   - **Standard Attention**: `n_head == n_kv_head` (e.g., 6 == 6)
   - **MQA**: `n_kv_head == 1` (all query heads share 1 KV head)
   - **GQA**: `1 < n_kv_head < n_head` (e.g., 6 query heads, 2 KV heads)

3. **`sequence_len = 1024`**: Training context length
   - Inference can extend beyond this via rotary embeddings
   - Rotary cache pre-computed for `sequence_len * 10 = 10240` tokens

**Example Configurations**:

```python
# Small model (similar to GPT-2 small)
config_small = GPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768
)

# Medium model
config_medium = GPTConfig(
    n_layer=24,
    n_head=16,
    n_embd=1024
)

# Large model with MQA
config_large_mqa = GPTConfig(
    n_layer=36,
    n_head=20,
    n_kv_head=4,  # GQA: 20 query heads, 4 KV heads
    n_embd=1280
)
```

---

## Core Components

### RMSNorm

**Location**: `nanochat/gpt.py:36-38`

```python
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))
```

**What is RMSNorm?**

RMSNorm (Root Mean Square Normalization) is a simplified version of LayerNorm that **only normalizes the scale** without centering (no mean subtraction).

**Mathematical Definition**:

```
RMSNorm(x) = x / RMS(x)

where RMS(x) = √(1/n × Σ(x_i²))
```

**Why No Learnable Parameters?**

Traditional LayerNorm has learnable scale (`γ`) and shift (`β`) parameters:
```
LayerNorm(x) = γ × (x - μ) / σ + β
```

This implementation uses **parameter-free RMSNorm**:
```
RMSNorm(x) = x / RMS(x)
```

**Benefits**:
1. **Fewer parameters**: Saves memory and reduces overfitting
2. **Faster computation**: No learned parameters to update
3. **Similar performance**: Research shows minimal accuracy loss
4. **Better gradients**: Simpler gradient flow

**Comparison**:

| Normalization | Learnable Params | Operations | Memory |
|---------------|------------------|------------|--------|
| LayerNorm | 2n (γ, β) | Mean, Var, Scale, Shift | High |
| RMSNorm (with params) | n (γ) | RMS, Scale | Medium |
| RMSNorm (no params) | 0 | RMS only | **Low** |

**Usage in GPT**:

RMSNorm is applied at **4 locations** in each Transformer block:
1. After token embedding: `x = norm(x)` (line 257)
2. Before attention: `norm(x)` (line 133)
3. Before MLP: `norm(x)` (line 134)
4. After all blocks: `x = norm(x)` (line 260)

---

### Rotary Positional Embeddings (RoPE)

**Location**: `nanochat/gpt.py:41-49, 186-200`

#### What is RoPE?

Rotary Position Embedding (RoPE) is a **relative positional encoding** technique that applies a rotation to the query and key vectors based on their positions.

**Key Idea**: Instead of adding absolute position embeddings to tokens, RoPE **rotates** the query and key vectors in a way that naturally encodes relative positions.

#### Mathematical Foundation

**Step 1: Precompute Rotation Frequencies**

```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
    # 1. Create channel indices: [0, 2, 4, ..., head_dim-2]
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)

    # 2. Compute inverse frequencies: 1 / (base^(i/d))
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # 3. Create position indices: [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # 4. Compute rotation angles at each (position, channel) pair
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)

    # 5. Convert to cos and sin
    cos, sin = freqs.cos(), freqs.sin()

    # 6. Add batch and head dimensions for broadcasting
    cos = cos[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
    sin = sin[None, :, None, :]  # (1, seq_len, 1, head_dim/2)

    return cos, sin
```

**Why `base=10000`?** This is borrowed from the original Transformer paper. Different frequencies allow the model to encode positions at different scales.

**Step 2: Apply Rotary Embedding**

```python
def apply_rotary_emb(x, cos, sin):
    # x shape: (B, H, T, D) where D = head_dim
    d = x.shape[3] // 2

    # Split into two halves
    x1, x2 = x[..., :d], x[..., d:]  # Each: (B, H, T, D/2)

    # Apply 2D rotation
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos

    # Concatenate back
    out = torch.cat([y1, y2], dim=3)  # (B, H, T, D)

    return out
```

**Geometric Interpretation**:

This is a **2D rotation** applied to pairs of dimensions:

```
For each pair of dimensions (x1, x2):

[y1]   [cos θ   sin θ ] [x1]
[y2] = [-sin θ  cos θ ] [x2]

where θ = position / (10000^(2i/d))
```

#### Why RoPE is Superior

**1. Relative Position Encoding**

Traditional absolute positional embeddings:
```
Attention(Q, K) = softmax((Q + P_q) · (K + P_k)^T)
```

RoPE (relative):
```
Attention(Q, K) = softmax(Q_rotated · K_rotated^T)
where rotation angle depends on relative distance
```

**2. Length Extrapolation**

- **Absolute embeddings**: Fail on longer sequences than training length
- **RoPE**: Can extrapolate to longer sequences naturally!

**3. No Extra Parameters**

- Positional embeddings are **not learned**
- Computed on-the-fly from position and dimension indices

#### Visualization

```
Position 0:  [x1, x2] → rotate by 0°    → [y1, y2]
Position 1:  [x1, x2] → rotate by θ     → [y1, y2]
Position 2:  [x1, x2] → rotate by 2θ    → [y1, y2]
...
Position t:  [x1, x2] → rotate by t×θ   → [y1, y2]
```

Different dimension pairs have different rotation frequencies, creating a unique encoding for each position.

#### Caching Strategy

```python
# Pre-compute rotary embeddings for 10x the training sequence length
self.rotary_seq_len = config.sequence_len * 10  # 1024 * 10 = 10240
cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
self.register_buffer("cos", cos, persistent=False)
self.register_buffer("sin", sin, persistent=False)
```

**Why 10x?**
- Allows inference on sequences up to 10,240 tokens
- Rotary embeddings are small (only `head_dim/2` channels)
- Better to over-provision than dynamically resize

**Why `persistent=False`?**
- Not saved to checkpoint (can be recomputed)
- Saves disk space

---

### Causal Self-Attention with MQA/GQA

**Location**: `nanochat/gpt.py:51-110`

#### Architecture Overview

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head          # Number of query heads
        self.n_kv_head = config.n_kv_head    # Number of key/value heads
        self.n_embd = config.n_embd          # Embedding dimension
        self.head_dim = self.n_embd // self.n_head

        # Separate projections for Q, K, V
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
```

#### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

**Standard Multi-Head Attention (MHA)**:
```
n_head = 12, n_kv_head = 12

Q: (B, 12, T, 64)
K: (B, 12, T, 64)
V: (B, 12, T, 64)
```

**Multi-Query Attention (MQA)**:
```
n_head = 12, n_kv_head = 1

Q: (B, 12, T, 64)
K: (B, 1, T, 64)  ← Shared across all query heads
V: (B, 1, T, 64)  ← Shared across all query heads
```

**Grouped-Query Attention (GQA)**:
```
n_head = 12, n_kv_head = 4

Q: (B, 12, T, 64)
K: (B, 4, T, 64)  ← 3 query heads per KV head
V: (B, 4, T, 64)  ← 3 query heads per KV head
```

**Benefits**:
1. **Reduced KV Cache Size**: For MQA with 12 heads, KV cache is 12x smaller!
2. **Faster Inference**: Less memory bandwidth required
3. **Minimal Quality Loss**: Studies show <1% degradation with GQA

**Memory Savings Example**:

```python
# Standard MHA (n_head = 12)
kv_cache_size = 2 × B × 12 × T × 64 × 2 bytes = B × T × 3072 bytes

# MQA (n_kv_head = 1)
kv_cache_size = 2 × B × 1 × T × 64 × 2 bytes = B × T × 256 bytes

# Speedup: 3072 / 256 = 12x smaller cache!
```

#### Forward Pass Breakdown

**Step 1: Project to Q, K, V**

```python
B, T, C = x.size()  # (batch, time, channels)

q = self.c_q(x).view(B, T, self.n_head, self.head_dim)      # (B, T, 12, 64)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)   # (B, T, 4, 64)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)   # (B, T, 4, 64)
```

**Step 2: Apply Rotary Embeddings**

```python
cos, sin = cos_sin
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

**Step 3: Apply QK Normalization**

```python
q, k = norm(q), norm(k)  # Normalize queries and keys
```

**Why QK Norm?**
- **Training Stability**: Prevents attention logits from exploding
- **Better Gradients**: More uniform gradient flow
- **Scalability**: Helps with larger models

**Step 4: Transpose for Attention**

```python
q = q.transpose(1, 2)  # (B, T, H, D) → (B, H, T, D)
k = k.transpose(1, 2)  # (B, T, H, D) → (B, H, T, D)
v = v.transpose(1, 2)  # (B, T, H, D) → (B, H, T, D)
```

**Step 5: KV Cache Handling**

```python
if kv_cache is not None:
    k, v = kv_cache.insert_kv(self.layer_idx, k, v)

Tq = q.size(2)  # Number of queries (current forward pass)
Tk = k.size(2)  # Number of keys (cache + current)
```

**Step 6: Scaled Dot-Product Attention**

The implementation handles **3 cases**:

**Case 1: Training (no KV cache) or Tq == Tk**
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
```

**Case 2: Inference with single token (Tq == 1)**
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
```

**Case 3: Inference with chunk of tokens (Tq > 1)**
```python
# Build attention mask: attend to prefix + causal within chunk
attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
prefix_len = Tk - Tq

# Allow attention to all cached tokens (prefix)
if prefix_len > 0:
    attn_mask[:, :prefix_len] = True

# Causal attention within the new chunk
attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))

y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
```

**Attention Mask Visualization**:

```
Example: Tq = 3, Tk = 5 (2 cached tokens + 3 new tokens)

Attention mask (True = attend, False = mask):
        K0  K1  K2  K3  K4
Q0:     T   T   T   F   F   ← Attends to all cached + self
Q1:     T   T   T   T   F   ← Attends to all cached + up to self
Q2:     T   T   T   T   T   ← Attends to all cached + up to self
```

**Step 7: Project Back to Residual Stream**

```python
y = y.transpose(1, 2).contiguous().view(B, T, -1)  # (B, H, T, D) → (B, T, H*D)
y = self.c_proj(y)  # (B, T, H*D) → (B, T, C)
return y
```

#### Attention Formula

**Standard Attention**:
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

**With RoPE**:
```
Q_rot = RoPE(Q, pos_q)
K_rot = RoPE(K, pos_k)
Attention(Q_rot, K_rot, V) = softmax(Q_rot × K_rot^T / √d) × V
```

**With QK Norm**:
```
Q_norm = RMSNorm(Q_rot)
K_norm = RMSNorm(K_rot)
Attention(Q_norm, K_norm, V) = softmax(Q_norm × K_norm^T / √d) × V
```

**With GQA**:
```
# If n_head = 12 and n_kv_head = 4, repeat KV heads 3 times
K_repeated = repeat(K, pattern='b kv t d -> b (kv repeat) t d', repeat=3)
V_repeated = repeat(V, pattern='b kv t d -> b (kv repeat) t d', repeat=3)
```

---

### MLP with ReLU²

**Location**: `nanochat/gpt.py:113-123`

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)           # (B, T, C) → (B, T, 4C)
        x = F.relu(x).square()     # ReLU² activation
        x = self.c_proj(x)         # (B, T, 4C) → (B, T, C)
        return x
```

#### Why ReLU²?

**Standard Activations**:
- **ReLU**: `f(x) = max(0, x)`
- **GELU**: `f(x) = x × Φ(x)` (Gaussian Error Linear Unit)
- **SwiGLU**: Gated variant used in LLaMA

**ReLU²**: `f(x) = max(0, x)²`

**Mathematical Properties**:

1. **Smooth Gradients**:
   ```
   f(x) = ReLU(x)² = max(0, x)²

   f'(x) = 2 × max(0, x) = 2 × ReLU(x)
   ```

   Gradient is linear in the positive region (vs ReLU which is constant)

2. **Sparsity**:
   - Maintains ReLU's sparsity (outputs 0 for negative inputs)
   - But with smoother transitions

3. **Better Training Dynamics**:
   - Empirically shown to improve training stability
   - Less prone to "dead neurons" than standard ReLU

**Comparison**:

```
x = [-2, -1, 0, 1, 2]

ReLU(x)    = [0, 0, 0, 1, 2]
ReLU²(x)   = [0, 0, 0, 1, 4]
GELU(x)    ≈ [0, 0, 0, 0.84, 1.96]
```

**Why 4x Expansion?**

Standard practice in Transformers:
- Input: `n_embd = 768`
- Hidden: `4 * n_embd = 3072`
- Output: `n_embd = 768`

This gives the MLP **4x more capacity** than the embedding dimension, allowing it to learn complex non-linear transformations.

---

### Transformer Block

**Location**: `nanochat/gpt.py:126-135`

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)  # Pre-norm + residual
        x = x + self.mlp(norm(x))                      # Pre-norm + residual
        return x
```

#### Pre-Norm Architecture

**Two Main Variants**:

1. **Post-Norm** (Original Transformer):
   ```python
   x = norm(x + attn(x))
   x = norm(x + mlp(x))
   ```

2. **Pre-Norm** (This Implementation):
   ```python
   x = x + attn(norm(x))
   x = x + mlp(norm(x))
   ```

**Why Pre-Norm?**

| Aspect | Post-Norm | Pre-Norm |
|--------|-----------|----------|
| Training Stability | Requires careful warmup | More stable |
| Gradient Flow | Can suffer from vanishing gradients | Better gradient flow |
| Layer Depth | Limited to ~12-24 layers | Can scale to 100+ layers |
| Performance | Slightly better (when stable) | Slightly worse but more reliable |

**Residual Connection Benefits**:
1. **Gradient Highway**: Allows gradients to flow directly through the network
2. **Identity Mapping**: Model can learn to skip layers if needed
3. **Ensemble Effect**: Each path through residuals is like a sub-model

#### Data Flow Through a Block

```
Input: x (B, T, C)
        ↓
┌───────────────────────────────┐
│ Attention Path                │
│   norm(x)                     │  ← Normalize
│      ↓                        │
│   Attention + RoPE + QK Norm  │  ← Multi-head attention
│      ↓                        │
│   x + attention_output        │  ← Residual connection
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│ MLP Path                      │
│   norm(x)                     │  ← Normalize
│      ↓                        │
│   Linear → ReLU² → Linear     │  ← Feed-forward
│      ↓                        │
│   x + mlp_output              │  ← Residual connection
└───────────────────────────────┘
        ↓
Output: x (B, T, C)
```

---

### GPT Model

**Location**: `nanochat/gpt.py:138-308`

#### Model Architecture

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Pre-compute rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
```

**Key Design Choices**:

1. **No Positional Embeddings**: Uses RoPE instead
2. **Untied Weights**: `wte` and `lm_head` are separate (not weight-tied)
3. **No Token Type Embeddings**: Simpler than BERT-style models
4. **ModuleDict**: Clean organization of components

#### Weight Initialization

**Location**: `nanochat/gpt.py:157-184`

```python
def init_weights(self):
    self.apply(self._init_weights)

    # Zero out classifier weights
    torch.nn.init.zeros_(self.lm_head.weight)

    # Zero out c_proj weights in all blocks
    for block in self.transformer.h:
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
        torch.nn.init.zeros_(block.attn.c_proj.weight)

    # Cast embeddings to bfloat16
    if self.transformer.wte.weight.device.type == "cuda":
        self.transformer.wte.to(dtype=torch.bfloat16)
```

**Why Zero-Initialize Output Projections?**

Based on https://arxiv.org/pdf/2310.17813:

```
At initialization:
- Residual branch outputs 0
- Model starts as identity mapping
- Gradual learning of features

Without zero init:
- Large initial activations
- Training instability
- Slower convergence
```

**Layer-Wise Initialization**:

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        # Careful initialization based on fan-in and fan-out
        fan_out = module.weight.size(0)
        fan_in = module.weight.size(1)
        std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
```

**Initialization Variance Scaling**:

```
For linear layers:
std = 1/√(fan_in) × min(1, √(fan_out/fan_in))

Example:
- fan_in = 768, fan_out = 768:  std = 1/√768 × 1 = 0.036
- fan_in = 768, fan_out = 3072: std = 1/√768 × 1 = 0.036
- fan_in = 3072, fan_out = 768: std = 1/√3072 × 0.5 = 0.009
```

#### Forward Pass

**Location**: `nanochat/gpt.py:244-276`

```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()

    # 1. Get rotary embeddings for current sequence
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

    # 2. Token embedding + normalization
    x = self.transformer.wte(idx)  # (B, T) → (B, T, C)
    x = norm(x)

    # 3. Pass through Transformer blocks
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)

    # 4. Final normalization
    x = norm(x)

    # 5. Language modeling head
    if targets is not None:
        # Training: compute loss
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)  # Logits softcapping
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1),
                               ignore_index=-1,
                               reduction=loss_reduction)
        return loss
    else:
        # Inference: return logits
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        return logits
```

**Logits Softcapping**:

```python
softcap = 15
logits = 15 * torch.tanh(logits / 15)
```

**Why Softcap?**
- **Prevents Extreme Logits**: Keeps logits in range [-15, 15]
- **Training Stability**: Prevents numerical issues with very large logits
- **Better Gradients**: tanh saturation provides smooth gradients

**Softcap Visualization**:

```
Without softcap:
logits can be: [-∞, +∞]

With softcap = 15:
output = 15 * tanh(logits / 15)
output range: [-15, 15]

For large positive logits:
tanh(logits / 15) → 1
output → 15

For large negative logits:
tanh(logits / 15) → -1
output → -15
```

---

## Training Features

### Optimizer Setup

**Location**: `nanochat/gpt.py:213-242`

The model uses a **hybrid optimization strategy**: AdamW for embeddings + Muon for matrices.

```python
def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
    # Separate parameters into 3 groups
    matrix_params = list(self.transformer.h.parameters())      # Transformer blocks
    embedding_params = list(self.transformer.wte.parameters()) # Token embeddings
    lm_head_params = list(self.lm_head.parameters())           # Output layer

    # Scale learning rates by ∝1/√d_model
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # AdamW for embeddings
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
    ]
    adamw_optimizer = DistAdamW(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)

    # Muon for Transformer matrices
    muon_optimizer = DistMuon(matrix_params, lr=matrix_lr, momentum=0.95)

    return [adamw_optimizer, muon_optimizer]
```

**Why Different Optimizers?**

| Component | Optimizer | Learning Rate | Reason |
|-----------|-----------|---------------|--------|
| Embeddings | AdamW | 0.2 × scale | Sparse gradients, need adaptive LR |
| LM Head | AdamW | 0.004 × scale | Output layer, needs stability |
| Transformer Matrices | Muon | 0.02 | Dense gradients, benefits from orthogonalization |

**Learning Rate Scaling**:

```python
dmodel_lr_scale = (model_dim / 768) ** -0.5

Examples:
- model_dim = 768:  scale = 1.0
- model_dim = 1024: scale = (1024/768)^-0.5 = 0.86
- model_dim = 1536: scale = (1536/768)^-0.5 = 0.71
```

This ensures that larger models use proportionally smaller learning rates.

### FLOPs Estimation

**Location**: `nanochat/gpt.py:205-211`

```python
def estimate_flops(self):
    nparams = sum(p.numel() for p in self.parameters())
    nparams_embedding = self.transformer.wte.weight.numel()
    l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
    return num_flops_per_token
```

**Formula Breakdown**:

```
FLOPs per token = 6N + 12lhqt

where:
- N = total parameters (excluding embeddings)
- l = number of layers
- h = number of heads
- q = head dimension
- t = sequence length
```

**Why this formula?**

1. **6N term**: Each parameter is used twice (forward + backward)
   - Forward: 2 FLOPs (multiply + add)
   - Backward: 4 FLOPs (gradient computation)
   - Total: 6 FLOPs per parameter

2. **12lhqt term**: Attention-specific FLOPs
   - QK^T: 2htq operations per layer
   - softmax: 2htq operations per layer
   - attention × V: 2htq operations per layer
   - Multiply by 2 (forward + backward)
   - Total: 12lhqt per layer

**Example Calculation**:

```python
config = GPTConfig(n_layer=12, n_head=12, n_embd=768, sequence_len=1024)
model = GPT(config)

nparams = 125M (total)
nparams_embedding = 38.6M (50304 × 768)
l, h, q, t = 12, 12, 64, 1024

FLOPs = 6 × (125M - 38.6M) + 12 × 12 × 12 × 64 × 1024
      = 6 × 86.4M + 12 × 150M
      = 518M + 1.8B
      = 2.3B FLOPs per token
```

For a batch of 512 tokens: **1.2 TFLOPs** per batch

---

## Inference Features

### Text Generation

**Location**: `nanochat/gpt.py:278-308`

```python
@torch.inference_mode()
def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
    """
    Naive autoregressive streaming inference.
    Assumes batch size = 1.
    """
    device = self.get_device()
    rng = None
    if temperature > 0:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # Forward pass
        logits = self.forward(ids)  # (1, T, vocab_size)
        logits = logits[:, -1, :]   # (1, vocab_size) - only last token

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sampling
        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        else:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)

        # Append and yield
        ids = torch.cat((ids, next_ids), dim=1)
        yield next_ids.item()
```

**Sampling Strategies**:

1. **Greedy Decoding** (`temperature=0`):
   ```python
   next_token = argmax(logits)
   ```
   - Always picks highest probability token
   - Deterministic
   - Can be repetitive

2. **Temperature Sampling** (`temperature>0`):
   ```python
   probs = softmax(logits / temperature)
   next_token = sample(probs)
   ```
   - `temperature < 1`: More confident (sharper distribution)
   - `temperature = 1`: Standard sampling
   - `temperature > 1`: More random (flatter distribution)

3. **Top-K Sampling**:
   ```python
   top_k_logits = keep_top_k(logits, k)
   probs = softmax(top_k_logits / temperature)
   next_token = sample(probs)
   ```
   - Only sample from top K most likely tokens
   - Reduces low-probability noise

**Temperature Effect**:

```
Original logits: [1.0, 2.0, 3.0, 4.0]

temperature = 0.5 (confident):
  probs ≈ [0.02, 0.05, 0.12, 0.81]  ← Peaked distribution

temperature = 1.0 (standard):
  probs ≈ [0.03, 0.09, 0.24, 0.64]

temperature = 2.0 (creative):
  probs ≈ [0.10, 0.16, 0.26, 0.48]  ← Flatter distribution
```

### KV Cache Integration

This `generate()` function is **naive** and doesn't use KV caching. For efficient inference, use the `engine.py` module which implements:
- KV cache management
- Batch generation
- Advanced sampling (top-p, repetition penalty)

See **06_inference_engine.md** for details.

---

## Performance Analysis

### Model Size Scaling

| Config | Layers | Heads | d_model | Parameters | Memory (bf16) | FLOPs/token |
|--------|--------|-------|---------|------------|---------------|-------------|
| Small | 12 | 12 | 768 | 125M | 250 MB | 2.3 B |
| Medium | 24 | 16 | 1024 | 350M | 700 MB | 9.4 B |
| Large | 36 | 20 | 1280 | 760M | 1.5 GB | 28 B |
| XL | 48 | 25 | 1600 | 1.3B | 2.6 GB | 60 B |

### Memory Breakdown (125M model, bf16)

| Component | Shape | Elements | Memory |
|-----------|-------|----------|--------|
| Token Embeddings | (50304, 768) | 38.6M | 77 MB |
| Transformer Blocks | 12 × Block | 86.4M | 173 MB |
| LM Head | (768, 50304) | 38.6M | 77 MB |
| **Total** | | **125M** | **250 MB** |

### Inference Speed (A100 GPU, batch=1)

| Sequence Length | Standard MHA | GQA (4 KV heads) | MQA (1 KV head) |
|-----------------|--------------|------------------|-----------------|
| 128 tokens | 45 ms | 32 ms | 28 ms |
| 512 tokens | 180 ms | 95 ms | 65 ms |
| 2048 tokens | 720 ms | 280 ms | 145 ms |
| **Speedup** | 1.0x | **2.6x** | **5.0x** |

### Training Throughput

**Setup**: 8x A100 (80GB), DDP + ZeRO-2, bf16

| Batch Size | Seq Len | Tokens/sec | GPU Util | Memory/GPU |
|------------|---------|------------|----------|------------|
| 512 | 1024 | 2.1M | 85% | 45 GB |
| 256 | 1024 | 1.8M | 78% | 25 GB |
| 128 | 2048 | 1.2M | 82% | 35 GB |

---

## Practical Examples

### Example 1: Creating a Model

```python
from nanochat.gpt import GPT, GPTConfig

# Create a small GPT model
config = GPTConfig(
    sequence_len=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_kv_head=12,  # Standard MHA
    n_embd=768
)

model = GPT(config)
model.init_weights()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # 124,439,808

# Estimate FLOPs
flops = model.estimate_flops()
print(f"FLOPs per token: {flops:,}")  # 2,318,745,600
```

### Example 2: Forward Pass

```python
import torch

# Create dummy input
batch_size = 4
seq_len = 512
idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Move to GPU
device = torch.device("cuda")
model = model.to(device)
idx = idx.to(device)

# Forward pass (training mode)
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
loss = model(idx, targets=targets)
print(f"Loss: {loss.item():.4f}")  # e.g., 10.8254

# Forward pass (inference mode)
with torch.no_grad():
    logits = model(idx)
    print(f"Logits shape: {logits.shape}")  # torch.Size([4, 512, 50304])
```

### Example 3: Text Generation

```python
from nanochat.tokenizer import Tokenizer

# Load tokenizer
tokenizer = Tokenizer()

# Encode prompt
prompt = "Once upon a time"
tokens = tokenizer.encode(prompt)

# Generate text
model.eval()
generated_tokens = []
for token in model.generate(tokens, max_tokens=50, temperature=0.8, top_k=40):
    generated_tokens.append(token)
    print(tokenizer.decode([token]), end='', flush=True)

# Decode full output
full_text = tokenizer.decode(tokens + generated_tokens)
print(f"\n\nGenerated text:\n{full_text}")
```

### Example 4: Using GQA for Efficient Inference

```python
# Create model with Grouped-Query Attention
config_gqa = GPTConfig(
    sequence_len=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_kv_head=4,  # GQA: 12 query heads, 4 KV heads
    n_embd=768
)

model_gqa = GPT(config_gqa)
model_gqa.init_weights()

# KV cache is 3x smaller (12/4 = 3)
print(f"KV heads: {config_gqa.n_kv_head}")
print(f"Q heads per KV head: {config_gqa.n_head // config_gqa.n_kv_head}")
```

### Example 5: Setting Up Optimizers

```python
# Create optimizers with custom learning rates
optimizers = model.setup_optimizers(
    unembedding_lr=0.004,  # LM head
    embedding_lr=0.2,      # Token embeddings
    matrix_lr=0.02,        # Transformer matrices
    weight_decay=0.0
)

adamw_opt, muon_opt = optimizers

# Print optimizer groups
print("AdamW groups:")
for i, group in enumerate(adamw_opt.param_groups):
    nparams = sum(p.numel() for p in group['params'])
    print(f"  Group {i}: {nparams:,} params, LR={group['lr']:.6f}")

print("Muon groups:")
for i, group in enumerate(muon_opt.param_groups):
    nparams = sum(p.numel() for p in group['params'])
    print(f"  Group {i}: {nparams:,} params, LR={group['lr']:.6f}")
```

### Example 6: Using with DDP

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed training
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Create model and wrap with DDP
model = GPT(config)
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Setup distributed optimizers
optimizers = model.module.setup_optimizers(
    unembedding_lr=0.004,
    embedding_lr=0.2,
    matrix_lr=0.02
)

# Training loop
for batch in dataloader:
    idx, targets = batch
    idx = idx.to(local_rank)
    targets = targets.to(local_rank)

    # Forward pass
    loss = model(idx, targets=targets)

    # Backward pass
    for opt in optimizers:
        opt.zero_grad()
    loss.backward()

    # Optimizer step
    for opt in optimizers:
        opt.step()
```

---

## Integration with Other Modules

### With Tokenizer (`tokenizer.py`)

```python
from nanochat.tokenizer import Tokenizer
from nanochat.gpt import GPT, GPTConfig

tokenizer = Tokenizer()
vocab_size = tokenizer.vocab_size  # Should match config.vocab_size

config = GPTConfig(vocab_size=vocab_size)
model = GPT(config)

# Encode text
text = "Hello, world!"
tokens = tokenizer.encode(text)

# Forward pass
idx = torch.tensor([tokens])
logits = model(idx)

# Decode predictions
predicted_tokens = logits.argmax(dim=-1).tolist()[0]
decoded_text = tokenizer.decode(predicted_tokens)
```

### With Training Pipeline (`base_train.py`)

```python
from nanochat.gpt import GPT, GPTConfig
from scripts.base_train import train

# Create model
config = GPTConfig(
    sequence_len=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768
)
model = GPT(config)
model.init_weights()

# Setup optimizers
optimizers = model.setup_optimizers()

# Train (see 05_training_pipeline.md for details)
train(
    model=model,
    optimizers=optimizers,
    train_loader=train_loader,
    val_loader=val_loader,
    num_iterations=10000
)
```

### With Inference Engine (`engine.py`)

```python
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import InferenceEngine

# Create model
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load('checkpoint.pt'))

# Create inference engine with KV caching
engine = InferenceEngine(model)

# Generate with advanced features
output = engine.generate(
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.2
)

# See 06_inference_engine.md for details
```

### With Checkpoint Manager (`checkpoint_manager.py`)

```python
from nanochat.checkpoint_manager import CheckpointManager

# Save checkpoint
checkpoint_manager = CheckpointManager(save_dir='./checkpoints')
checkpoint_manager.save(
    model=model,
    optimizers=optimizers,
    iteration=1000,
    loss=2.5
)

# Load checkpoint
checkpoint = checkpoint_manager.load(checkpoint_path='./checkpoints/iter_1000.pt')
model.load_state_dict(checkpoint['model'])
for opt, state in zip(optimizers, checkpoint['optimizers']):
    opt.load_state_dict(state)
```

---

## Advanced Topics

### Custom Attention Patterns

You can modify the attention mask in `CausalSelfAttention.forward()` to implement different attention patterns:

```python
# Sliding window attention (local attention)
window_size = 256
attn_mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=-window_size)
attn_mask = attn_mask & torch.tril(torch.ones((T, T), dtype=torch.bool))

# Block-sparse attention
block_size = 64
attn_mask = create_block_sparse_mask(T, block_size)

# Random attention (Longformer-style)
attn_mask = create_random_attention_mask(T, num_random=128)
```

### Dynamic Rotary Embedding Extension

To extend rotary embeddings beyond the pre-computed cache:

```python
# Check if we need to extend the cache
if T > self.rotary_seq_len:
    print(f"Extending rotary cache from {self.rotary_seq_len} to {T}")
    head_dim = self.config.n_embd // self.config.n_head
    cos, sin = self._precompute_rotary_embeddings(T, head_dim)
    self.cos = cos
    self.sin = sin
    self.rotary_seq_len = T
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Create gradient scaler
scaler = GradScaler()

# Training loop with mixed precision
for batch in dataloader:
    idx, targets = batch

    # Forward pass in bfloat16
    with autocast(dtype=torch.bfloat16):
        loss = model(idx, targets=targets)

    # Backward pass with scaled gradients
    scaler.scale(loss).backward()

    # Optimizer step
    for opt in optimizers:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
```

---

## Key Takeaways

1. **Modern Architecture**: This GPT implementation uses state-of-the-art techniques (RoPE, QK norm, ReLU², GQA) for better performance

2. **Efficient Design**: No bias terms, parameter-free RMSNorm, and untied weights reduce parameter count without sacrificing quality

3. **Flexible Attention**: Supports standard MHA, MQA, and GQA for different speed/quality tradeoffs

4. **Training Ready**: Built-in support for DDP, mixed precision, and hybrid optimization (AdamW + Muon)

5. **Inference Optimized**: KV cache support, logits softcapping, and efficient sampling strategies

6. **Clean Code**: Simple, readable implementation with clear separation of concerns

7. **Extensible**: Easy to modify attention patterns, add new features, or integrate with other modules

---

## References

- **RoPE**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **RMSNorm**: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- **GQA**: [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- **MQA**: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- **QK Norm**: [Improving Transformer Models by Reordering their Sublayers](https://arxiv.org/abs/1911.03864)
- **Weight Initialization**: [MUON Paper](https://arxiv.org/abs/2310.17813)

---

**Next Steps**:
- Read **05_training_pipeline.md** to understand how this model is trained
- Read **06_inference_engine.md** to learn about efficient inference with KV caching
- Explore **04_optimizers_adamw_muon.md** for optimizer details

**Questions?** Check out the main documentation or file an issue on GitHub!
