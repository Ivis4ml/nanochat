# Deep Dive: Checkpoint Manager (`nanochat/checkpoint_manager.py`)

**File**: `nanochat/checkpoint_manager.py`
**Purpose**: Save and load model/optimizer/state checkpoints
**Lines of Code**: ~153

---

## Table of Contents

1. [Overview](#overview)
2. [Checkpoint Structure](#checkpoint-structure)
3. [Saving Checkpoints](#saving-checkpoints)
4. [Loading Checkpoints](#loading-checkpoints)
5. [Building Models](#building-models)
6. [Auto-Discovery](#auto-discovery)
7. [NanoChat Directory Structure](#nanochat-directory-structure)
8. [Practical Examples](#practical-examples)

---

## Overview

The checkpoint manager handles **three-part checkpoints**:
1. **Model weights** (`model_XXXXXX.pt`)
2. **Optimizer states** (`optim_XXXXXX.pt`)
3. **Metadata** (`meta_XXXXXX.json`)

```
Checkpoint Directory Structure:
checkpoints/
├── d12/                        # Model tag (d12 = 12 layers)
│   ├── model_000000.pt         # Initial checkpoint
│   ├── optim_000000.pt
│   ├── meta_000000.json
│   ├── model_001000.pt         # Step 1000
│   ├── optim_001000.pt
│   ├── meta_001000.json
│   └── ...
└── d24/                        # Larger model
    ├── model_000000.pt
    └── ...
```

**Key Features**:
- **Separate files**: Model, optimizer, metadata stored separately
- **Step-indexed**: Easy to identify and resume from any step
- **JSON metadata**: Human-readable training config and stats
- **Auto-discovery**: Find largest model and latest step automatically
- **torch.compile compatible**: Handles `_orig_mod.` prefix stripping

---

## Checkpoint Structure

### Model File (`model_XXXXXX.pt`)

```python
# Contains: model.state_dict()
model_data = {
    'transformer.wte.weight': tensor(...),           # Token embeddings
    'transformer.h.0.attn.c_q.weight': tensor(...),  # Layer 0 attention Q
    'transformer.h.0.attn.c_k.weight': tensor(...),  # Layer 0 attention K
    'transformer.h.0.attn.c_v.weight': tensor(...),  # Layer 0 attention V
    'transformer.h.0.attn.c_proj.weight': tensor(...),
    'transformer.h.0.mlp.c_fc.weight': tensor(...),  # Layer 0 MLP
    'transformer.h.0.mlp.c_proj.weight': tensor(...),
    # ... more layers ...
    'lm_head.weight': tensor(...),                   # Output projection
}
```

**Size Example** (125M model, bfloat16):
```
Total params: 125M
Size on disk: ~250 MB (bfloat16)
```

### Optimizer File (`optim_XXXXXX.pt`)

```python
# Contains: list of optimizer.state_dict()
optimizer_data = [
    {  # AdamW optimizer
        'state': {
            0: {'exp_avg': tensor(...), 'exp_avg_sq': tensor(...), 'step': 1000},
            1: {'exp_avg': tensor(...), 'exp_avg_sq': tensor(...), 'step': 1000},
            # ... per-parameter states ...
        },
        'param_groups': [
            {'lr': 0.004, 'betas': (0.8, 0.95), 'eps': 1e-10, ...},
            {'lr': 0.2, 'betas': (0.8, 0.95), 'eps': 1e-10, ...},
        ]
    },
    {  # Muon optimizer
        'state': {...},
        'param_groups': [...]
    }
]
```

**Size**: Roughly 2-3x model size (momentum buffers)

### Metadata File (`meta_XXXXXX.json`)

```json
{
  "model_config": {
    "sequence_len": 1024,
    "vocab_size": 32768,
    "n_layer": 12,
    "n_head": 12,
    "n_kv_head": 12,
    "n_embd": 768
  },
  "step": 1000,
  "train_loss": 3.245,
  "val_loss": 3.312,
  "learning_rate": 0.0006,
  "tokens_processed": 524288000,
  "timestamp": "2024-01-15T10:30:00"
}
```

**Benefits**:
- Human-readable
- Easy to inspect without loading PyTorch
- Contains all info needed to rebuild model

---

## Saving Checkpoints

**Location**: `nanochat/checkpoint_manager.py:23-39`

```python
def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    """
    Save a complete checkpoint (model + optimizer + metadata).

    Args:
        checkpoint_dir: Directory to save to
        step: Current training step (used in filename)
        model_data: model.state_dict()
        optimizer_data: List of optimizer.state_dict() or None
        meta_data: Dict of training metadata
    """
    # Safety: only rank 0 should save
    assert int(os.environ.get('RANK', 0)) == 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_data, model_path)
    log0(f"Saved model file to: {model_path}")

    # Save optimizer state (optional)
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        torch.save(optimizer_data, optimizer_path)
        log0(f"Saved optimizer file to: {optimizer_path}")

    # Save metadata as JSON
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    log0(f"Saved metadata file to: {meta_path}")
```

### Filename Format

```
model_{step:06d}.pt   →  model_001000.pt (step 1000)
optim_{step:06d}.pt   →  optim_001000.pt
meta_{step:06d}.json  →  meta_001000.json

Zero-padded to 6 digits for proper sorting:
  model_000001.pt
  model_000010.pt
  model_000100.pt
  model_001000.pt
```

### Usage Pattern

```python
# In training loop
if step % save_interval == 0 and rank == 0:
    save_checkpoint(
        checkpoint_dir=f"checkpoints/{model_tag}",
        step=step,
        model_data=model.state_dict(),
        optimizer_data=[opt.state_dict() for opt in optimizers],
        meta_data={
            "model_config": config.__dict__,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
    )
```

---

## Loading Checkpoints

**Location**: `nanochat/checkpoint_manager.py:42-55`

```python
def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    """
    Load a checkpoint from disk.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        step: Training step to load
        device: Device to load tensors to
        load_optimizer: Whether to load optimizer states

    Returns:
        model_data: Model state dict
        optimizer_data: Optimizer state dict (or None)
        meta_data: Metadata dict
    """
    # Load model weights
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)

    # Load optimizer (optional)
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)

    # Load metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    return model_data, optimizer_data, meta_data
```

### Device Handling

```python
# Load to specific device
model_data = torch.load(path, map_location='cuda:0')

# Load to CPU first, then move
model_data = torch.load(path, map_location='cpu')
model.load_state_dict(model_data)
model = model.to('cuda')
```

---

## Building Models

**Location**: `nanochat/checkpoint_manager.py:58-94`

```python
def build_model(checkpoint_dir, step, device, phase):
    """
    Build a model from checkpoint.

    Args:
        checkpoint_dir: Directory with checkpoint files
        step: Step to load
        device: Target device
        phase: "train" or "eval"

    Returns:
        model: Loaded GPT model
        tokenizer: Loaded tokenizer
        meta_data: Training metadata
    """
    # Load checkpoint data
    model_data, _, meta_data = load_checkpoint(
        checkpoint_dir, step, device, load_optimizer=False
    )

    # CPU inference: convert bfloat16 to float32
    if device.type == "cpu":
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

    # Fix torch.compile key prefix
    model_data = {
        k.removeprefix("_orig_mod."): v
        for k, v in model_data.items()
    }

    # Build model from config
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)

    # Create model on meta device (no memory allocation)
    with torch.device("meta"):
        model = GPT(model_config)

    # Move to real device and initialize
    model.to_empty(device=device)
    model.init_weights()  # Initialize rotary embeddings

    # Load weights
    model.load_state_dict(model_data, strict=True, assign=True)

    # Set training mode
    if phase == "eval":
        model.eval()
    else:
        model.train()

    # Load tokenizer and verify compatibility
    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]

    return model, tokenizer, meta_data
```

### torch.compile Compatibility

```python
# Problem: torch.compile prepends "_orig_mod." to all keys
compiled_keys = ["_orig_mod.transformer.wte.weight", ...]

# Solution: Strip the prefix
model_data = {
    k.removeprefix("_orig_mod."): v
    for k, v in model_data.items()
}
# Result: ["transformer.wte.weight", ...]
```

### Meta Device Pattern

```python
# Create model without allocating memory
with torch.device("meta"):
    model = GPT(config)  # Tensors on "meta" device (no memory)

# Move to real device
model.to_empty(device="cuda")  # Allocate empty tensors

# Initialize and load
model.init_weights()
model.load_state_dict(model_data, assign=True)
```

**Benefits**:
- Faster initialization for large models
- Lower peak memory usage
- Clean separation of creation and loading

---

## Auto-Discovery

### Find Largest Model

**Location**: `nanochat/checkpoint_manager.py:97-114`

```python
def find_largest_model(checkpoint_dir):
    """
    Find the largest model by layer count (d<number> convention).

    Directory structure:
    checkpoints/
    ├── d12/  → 12 layers
    ├── d24/  → 24 layers
    └── d36/  → 36 layers (selected)
    """
    model_tags = [f for f in os.listdir(checkpoint_dir)
                  if os.path.isdir(os.path.join(checkpoint_dir, f))]

    # Try to parse d<number> format
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))

    if candidates:
        # Sort by depth descending, return largest
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # Fallback: most recently modified
    model_tags.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )
    return model_tags[0]
```

### Find Last Step

**Location**: `nanochat/checkpoint_manager.py:117-123`

```python
def find_last_step(checkpoint_dir):
    """
    Find the highest step number in a checkpoint directory.

    Files: model_000000.pt, model_001000.pt, model_002000.pt
    Returns: 2000
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Extract step numbers and find max
    last_step = int(max(
        os.path.basename(f).split("_")[-1].split(".")[0]
        for f in checkpoint_files
    ))

    return last_step
```

---

## NanoChat Directory Structure

### Checkpoint Directories

```python
# Source → Directory mapping
model_dir = {
    "base": "base_checkpoints",      # Pretraining
    "mid": "mid_checkpoints",        # Mid-training
    "sft": "chatsft_checkpoints",    # Supervised fine-tuning
    "rl": "chatrl_checkpoints",      # Reinforcement learning
}
```

### Full Directory Layout

```
nanochat/
├── base_data/                  # Training data
│   └── shard_*.parquet
├── tokenizer/                  # Tokenizer files
│   └── tokenizer.pkl
├── base_checkpoints/           # Pretraining checkpoints
│   ├── d12/
│   │   ├── model_001000.pt
│   │   ├── optim_001000.pt
│   │   └── meta_001000.json
│   └── d24/
├── mid_checkpoints/            # Mid-training (optional)
├── chatsft_checkpoints/        # SFT checkpoints
│   └── d12/
└── chatrl_checkpoints/         # RL checkpoints
    └── d12/
```

### Convenience Functions

**Location**: `nanochat/checkpoint_manager.py:128-152`

```python
def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    """
    Load model with auto-discovery of tag and step.
    """
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing: {model_tag}")

    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)

    if step is None:
        step = find_last_step(checkpoint_dir)

    log0(f"Loading model from {checkpoint_dir} with step {step}")
    return build_model(checkpoint_dir, step, device, phase)


def load_model(source, *args, **kwargs):
    """
    Load model by source name.

    Args:
        source: "base", "mid", "sft", or "rl"
        *args, **kwargs: Passed to load_model_from_dir
    """
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]

    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
```

---

## Practical Examples

### Example 1: Save Checkpoint During Training

```python
from nanochat.checkpoint_manager import save_checkpoint

# In training loop
for step, (inputs, targets) in enumerate(dataloader):
    loss = model(inputs, targets=targets)
    loss.backward()

    for opt in optimizers:
        opt.step()
        opt.zero_grad()

    # Save every 1000 steps
    if step % 1000 == 0 and step > 0:
        save_checkpoint(
            checkpoint_dir="checkpoints/d12",
            step=step,
            model_data=model.state_dict(),
            optimizer_data=[opt.state_dict() for opt in optimizers],
            meta_data={
                "model_config": {
                    "n_layer": 12,
                    "n_head": 12,
                    "n_embd": 768,
                    "vocab_size": 32768,
                    "sequence_len": 1024,
                },
                "step": step,
                "train_loss": loss.item(),
            }
        )
```

### Example 2: Load Model for Inference

```python
from nanochat.checkpoint_manager import load_model

# Load latest base model (auto-discovery)
model, tokenizer, meta = load_model(
    source="base",
    device=torch.device("cuda"),
    phase="eval"
)

# Generate text
prompt = "Once upon a time"
tokens = tokenizer.encode(prompt, prepend="<|bos|>")
for token in model.generate(tokens, max_tokens=100):
    print(tokenizer.decode([token]), end='', flush=True)
```

### Example 3: Resume Training

```python
from nanochat.checkpoint_manager import load_checkpoint

# Load checkpoint with optimizer state
model_data, optimizer_data, meta_data = load_checkpoint(
    checkpoint_dir="checkpoints/d12",
    step=5000,
    device=torch.device("cuda"),
    load_optimizer=True
)

# Rebuild model
model = GPT(GPTConfig(**meta_data["model_config"]))
model.load_state_dict(model_data)

# Rebuild optimizers
optimizers = model.setup_optimizers()
for opt, opt_state in zip(optimizers, optimizer_data):
    opt.load_state_dict(opt_state)

# Resume from step 5000
start_step = meta_data["step"]
print(f"Resuming training from step {start_step}")
```

### Example 4: Load Specific Checkpoint

```python
from nanochat.checkpoint_manager import load_model_from_dir

# Load specific model and step
model, tokenizer, meta = load_model_from_dir(
    checkpoints_dir="base_checkpoints",
    device=torch.device("cuda"),
    phase="eval",
    model_tag="d24",   # Specific model size
    step=10000         # Specific step
)
```

### Example 5: Find Available Checkpoints

```python
from nanochat.checkpoint_manager import find_largest_model, find_last_step
import os

checkpoints_dir = "base_checkpoints"

# List all model tags
model_tags = [f for f in os.listdir(checkpoints_dir)
              if os.path.isdir(os.path.join(checkpoints_dir, f))]
print(f"Available models: {model_tags}")

# Find largest
largest = find_largest_model(checkpoints_dir)
print(f"Largest model: {largest}")

# Find last step for each
for tag in model_tags:
    checkpoint_dir = os.path.join(checkpoints_dir, tag)
    try:
        last_step = find_last_step(checkpoint_dir)
        print(f"  {tag}: last step = {last_step}")
    except FileNotFoundError:
        print(f"  {tag}: no checkpoints")
```

### Example 6: Load SFT Model for Chat

```python
from nanochat.checkpoint_manager import load_model

# Load supervised fine-tuned model
model, tokenizer, meta = load_model(
    source="sft",
    device=torch.device("cuda"),
    phase="eval"
)

# Chat format
conversation = {
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ]
}

# Render for completion
ids = tokenizer.render_for_completion(conversation)

# Generate response
for token in model.generate(ids, max_tokens=200, temperature=0.7):
    print(tokenizer.decode([token]), end='', flush=True)
```

### Example 7: Distributed Save (Rank 0 Only)

```python
import torch.distributed as dist
from nanochat.checkpoint_manager import save_checkpoint

# Only rank 0 saves
if dist.get_rank() == 0:
    # Get model state (from DDP wrapper)
    model_state = model.module.state_dict()

    # Gather optimizer states
    opt_states = [opt.state_dict() for opt in optimizers]

    save_checkpoint(
        checkpoint_dir="checkpoints/d12",
        step=step,
        model_data=model_state,
        optimizer_data=opt_states,
        meta_data={...}
    )

# Sync all processes after save
dist.barrier()
```

---

## Key Takeaways

1. **Three-Part Checkpoints**: Model, optimizer, metadata stored separately

2. **Step-Indexed**: Zero-padded step numbers for proper sorting

3. **JSON Metadata**: Human-readable config and training stats

4. **Auto-Discovery**: Find largest model and latest step automatically

5. **torch.compile Compatible**: Handles `_orig_mod.` prefix stripping

6. **CPU Inference**: Auto-converts bfloat16 to float32 on CPU

7. **Meta Device**: Efficient model creation without memory allocation

8. **Rank 0 Only**: Saves only from rank 0 to avoid corruption

---

## References

- **PyTorch Checkpointing**: [Save and Load Guide](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- **torch.compile**: [Compilation Guide](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- **Meta Device**: [Meta Tensors](https://pytorch.org/docs/stable/meta.html)

---

**Next Steps**:
- Read **01_gpt_architecture.md** for model architecture
- Read **05_training_pipeline.md** for how checkpoints fit into training

**Questions?** Check the main documentation or file an issue on GitHub!
