# Complete Training Guide for Nanochat

> **Step-by-step guide to training your own ChatGPT clone on 8×H100 GPUs**

This comprehensive guide will walk you through the entire process of training nanochat from scratch, covering everything from environment setup to final model evaluation.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Understanding the Training Pipeline](#understanding-the-training-pipeline)
- [Phase 0: Initial Setup](#phase-0-initial-setup)
- [Phase 1: Tokenizer Training](#phase-1-tokenizer-training)
- [Phase 2: Base Model Pretraining](#phase-2-base-model-pretraining)
- [Phase 3: Midtraining](#phase-3-midtraining)
- [Phase 4: Supervised Fine-Tuning](#phase-4-supervised-fine-tuning)
- [Phase 5: Reinforcement Learning (Optional)](#phase-5-reinforcement-learning-optional)
- [Phase 6: Inference and Evaluation](#phase-6-inference-and-evaluation)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Cost and Time Estimates](#cost-and-time-estimates)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

You have exactly what you need! Your machine specs:

```
8× NVIDIA H100 80GB HBM3
- Total VRAM: 640 GB
- CUDA Version: 12.4
- Driver: 550.144.03
```

**Ideal for:**
- ✅ d20 model (561M params) - $100 / 4 hours
- ✅ d26 model (1.1B params) - $300 / 12 hours
- ✅ d32 model (1.9B params) - $800 / 33 hours

### Software Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10+
- **CUDA**: 12.4 (already installed)
- **Storage**:
  - 50 GB for d20 ($100 tier)
  - 150 GB for d26 ($300 tier)
  - 300 GB for d32 ($800 tier)
- **Internet**: Stable connection for downloading data

### Cost Estimate

Assuming GPU rental at **$3/GPU/hour** (8 GPUs):
- **d20**: ~$100 (4 hours)
- **d26**: ~$300 (12 hours)
- **d32**: ~$800 (33 hours)

---

## Environment Setup

### Step 1: Clone the Repository

```bash
# Clone nanochat
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Verify you're in the right place
pwd  # Should show: /path/to/nanochat
ls   # Should see: README.md, nanochat/, scripts/, speedrun.sh, etc.
```

### Step 2: Install uv Package Manager

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if needed)
source $HOME/.cargo/env

# Verify installation
uv --version
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install dependencies (GPU version)
uv sync --extra gpu

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA available: True
GPU count: 8
```

### Step 4: Install Rust (for Tokenizer)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add to PATH
source "$HOME/.cargo/env"

# Verify installation
rustc --version
cargo --version
```

### Step 5: Build Rust Tokenizer

```bash
# Build the rustbpe tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# This compiles the Rust code and installs it as a Python module
# Takes ~1-2 minutes
```

### Step 6: Set Up Directory Structure

```bash
# Set base directory for artifacts
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# This will store:
# - Downloaded data shards
# - Trained tokenizer
# - Model checkpoints
# - Training logs

# Add to your shell profile for persistence
echo 'export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"' >> ~/.bashrc
```

### Step 7: (Optional) Set Up Weights & Biases

```bash
# Install wandb
pip install wandb

# Login (get API key from https://wandb.ai)
wandb login

# During training, set WANDB_RUN environment variable:
# export WANDB_RUN=my_experiment_name
```

---

## Understanding the Training Pipeline

Before we start, let's understand the full pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

1. TOKENIZER TRAINING (~30 min)
   ├─ Download: ~2GB of text data (8 shards)
   ├─ Train: BPE tokenizer with vocab_size=65536
   └─ Output: tokenizer.pkl, token_bytes.pt

2. BASE MODEL PRETRAINING (~2-3 hours for d20)
   ├─ Download: ~24GB more data (240 shards total)
   ├─ Train: 561M parameter Transformer on 54B chars
   ├─ Evaluate: CORE metric, validation loss
   └─ Output: base_model.pth, base_optimizer.pth

3. MIDTRAINING (~30 min)
   ├─ Download: Synthetic conversation data (~2MB)
   ├─ Train: Teach special tokens, tool use, conversations
   ├─ Evaluate: Task mixture (GSM8K, MMLU, etc.)
   └─ Output: mid_model.pth, mid_optimizer.pth

4. SUPERVISED FINE-TUNING (~20 min)
   ├─ Train: Single epoch on conversation tasks
   ├─ Evaluate: All chat tasks
   └─ Output: sft_model.pth, sft_optimizer.pth

5. REINFORCEMENT LEARNING (Optional, ~20 min)
   ├─ Train: GRPO on GSM8K
   ├─ Evaluate: GSM8K accuracy
   └─ Output: rl_model.pth, rl_optimizer.pth

6. INFERENCE & WEB SERVING
   ├─ Generate: Sample completions
   ├─ Report: Complete report card (report.md)
   └─ Serve: Web UI on port 8000

Total time (d20): ~4 hours
Total cost: ~$100
```

---

## Phase 0: Initial Setup

### Create a Training Session

It's recommended to use `screen` or `tmux` since training takes hours:

```bash
# Option 1: Using screen
screen -S nanochat-training

# Option 2: Using tmux
tmux new -s nanochat-training

# You can detach/reattach later:
# screen: Ctrl+A then D to detach, screen -r to reattach
# tmux: Ctrl+B then D to detach, tmux attach to reattach
```

### Initialize Report

```bash
# Reset report and write system info
python -m nanochat.report reset

# This creates report/ directory and writes:
# - GPU information
# - System specs
# - Git commit hash
# - Timestamp
```

### Choose Your Model Size

Edit the training script or set parameters:

```bash
# For d20 (561M params, $100, 4 hours) - RECOMMENDED FOR FIRST TIME
export MODEL_DEPTH=20
export DEVICE_BATCH_SIZE=32
export NUM_SHARDS=240

# For d26 (1.1B params, $300, 12 hours)
# export MODEL_DEPTH=26
# export DEVICE_BATCH_SIZE=16
# export NUM_SHARDS=450

# For d32 (1.9B params, $800, 33 hours)
# export MODEL_DEPTH=32
# export DEVICE_BATCH_SIZE=16
# export NUM_SHARDS=800
```

---

## Phase 1: Tokenizer Training

**Time**: ~30 minutes
**Data**: ~2 GB

### Step 1.1: Download Initial Training Data

```bash
# Download first 8 shards (~2GB) for tokenizer training
python -m nanochat.dataset -n 8

# What's happening:
# - Downloads Parquet files from HuggingFace
# - Source: FineWeb-Edu-100B dataset
# - Each shard: ~100MB compressed, ~250M characters
# - Location: $NANOCHAT_BASE_DIR/data/

# Monitor progress:
ls -lh $NANOCHAT_BASE_DIR/data/*.parquet | wc -l  # Should show 8
```

**Expected output:**
```
Downloading shard 0/8...
Downloading shard 1/8...
...
Downloading shard 7/8...
All shards downloaded successfully.
```

### Step 1.2: Start Background Download

While tokenizer trains, download remaining data in background:

```bash
# Download 240 shards total for d20 (or 450 for d26, 800 for d32)
python -m nanochat.dataset -n $NUM_SHARDS &
DATASET_PID=$!

# Save PID to check later
echo $DATASET_PID > /tmp/dataset_download.pid

# Monitor progress in another terminal:
# watch -n 10 "ls $NANOCHAT_BASE_DIR/data/*.parquet | wc -l"
```

### Step 1.3: Train Tokenizer

```bash
# Train BPE tokenizer on 2B characters
python -m scripts.tok_train --max_chars=2000000000

# Parameters:
# --max_chars: Characters to train on (2B recommended)
# --vocab_size: 65536 (default, 2^16)
# --doc_cap: 10000 (max chars per document)

# Training process:
# 1. Streams text from parquet files
# 2. Splits text using regex pattern
# 3. Performs BPE merges
# 4. Creates 65536 token vocabulary
```

**Expected output:**
```
max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 65,536
Processing sequences from iterator...
Progress: 10%...
Progress: 50%...
Progress: 100%
Training time: 1234.56s
Saved tokenizer to ~/.cache/nanochat/tokenizer/tokenizer.pkl
```

### Step 1.4: Evaluate Tokenizer

```bash
# Test tokenizer and compute compression ratio
python -m scripts.tok_eval

# This evaluates:
# - Compression ratio (chars per token)
# - Encoding/decoding correctness
# - Special token handling
```

**Expected output:**
```
Tokenizer loaded successfully
Vocab size: 65536
Special tokens: 9
Average compression: 4.8 chars/token
Test passed: ✓
```

### Step 1.5: Verify Data Download

```bash
# Check if background download finished
wait $DATASET_PID

# Verify shard count
SHARD_COUNT=$(ls $NANOCHAT_BASE_DIR/data/*.parquet | wc -l)
echo "Downloaded $SHARD_COUNT shards"

# For d20, should be 240
if [ $SHARD_COUNT -ge $NUM_SHARDS ]; then
    echo "✓ All data downloaded!"
else
    echo "⚠ Still downloading... ($SHARD_COUNT/$NUM_SHARDS)"
fi
```

---

## Phase 2: Base Model Pretraining

**Time**: ~2-3 hours (d20), ~8 hours (d26), ~22 hours (d32)
**Cost**: ~$70 of the total budget

This is the most compute-intensive phase!

### Step 2.1: Understanding the Math

```
Model: d20 (depth=20)
Parameters: 561M
Training tokens: 561M × 20 = 11.2B (Chinchilla scaling)
Characters: 11.2B × 4.8 = 54B chars
Data shards: 54B / 250M = 216 shards (we download 240 for safety)

Training config:
- Sequence length: 2048 tokens
- Device batch size: 32 per GPU
- Total batch size: 524,288 tokens
- Gradient accumulation: Automatic
- Learning rate schedule: Cosine with warmup
```

### Step 2.2: Start Base Training

```bash
# Train base model (d20 example)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=32 \
    --run=my_d20_model

# If using wandb:
# --run=my_d20_model will log to https://wandb.ai/your-username/nanochat/runs/my_d20_model

# Parameters explained:
# --depth=20: Model size (20 layers = 561M params)
# --device_batch_size=32: Batch size per GPU
# --run: Name for this experiment (for wandb)
```

**Training script does:**
1. Initializes model from scratch
2. Sets up optimizers (Muon for weights, AdamW for embeddings)
3. Loads data in streaming fashion
4. Trains for calculated number of iterations
5. Saves checkpoints periodically
6. Evaluates periodically

### Step 2.3: Monitor Training Progress

Open another terminal and monitor:

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Expected:
# - All 8 GPUs at 90-100% utilization
# - Memory: ~40-50GB per GPU for d20
# - Power: ~500-600W per GPU

# View training logs
tail -f speedrun.log  # if using screen logging

# Watch loss decrease
# Initial loss: ~11-12
# Final loss: ~3.5-4.0 (for d20)
```

### Step 2.4: Checkpoints

Checkpoints are saved periodically to:
```
$NANOCHAT_BASE_DIR/checkpoints/base/
├── model.pth           # Model weights (~2.2GB for d20)
├── optimizer.pth       # Optimizer state (~6.6GB for d20)
└── metadata.json       # Hyperparameters, iteration count
```

**Recovery from interruption:**
```bash
# Training automatically resumes from latest checkpoint
# Just re-run the same command:
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=32 \
    --run=my_d20_model
```

### Step 2.5: Evaluate Base Model

```bash
# After training completes, evaluate on validation set
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_loss

# Computes:
# - Validation loss
# - Bits per byte (BPB)
# - Generates sample text

# Evaluate on CORE benchmark
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_eval

# CORE tasks:
# - MMLU (multi-subject QA)
# - HellaSwag (commonsense reasoning)
# - ARC (science questions)
# - PIQA, etc.
```

**Expected CORE score for d20:** ~0.22 (slightly below GPT-2)

---

## Phase 3: Midtraining

**Time**: ~30 minutes
**Purpose**: Teach the model conversation structure and tool use

### Step 3.1: Download Conversation Data

```bash
# Download synthetic conversation data
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# This file (~2MB) contains:
# - Synthetic conversations
# - Demonstrates proper format
# - Teaches model personality
```

### Step 3.2: Inspect Conversation Format

```bash
# Look at example conversations
head -n 1 $NANOCHAT_BASE_DIR/identity_conversations.jsonl | jq .

# Format:
# {
#   "messages": [
#     {"role": "user", "content": "Hello!"},
#     {"role": "assistant", "content": "Hi! How can I help?"}
#   ]
# }
```

### Step 3.3: Run Midtraining

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.mid_train -- \
    --device_batch_size=32 \
    --run=my_d20_model

# This trains on:
# - Identity conversations
# - GSM8K (math with calculator tool)
# - MMLU (multiple choice)
# - SmolTalk (conversational data)
# - SpellingBee (character counting)

# Training teaches:
# - Special token usage (<|user_start|>, etc.)
# - Tool calling (calculator, Python)
# - Multiple-choice formatting
```

### Step 3.4: Evaluate Midtraining

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_eval -- -i mid

# Evaluates on:
# - ARC-Easy, ARC-Challenge
# - GSM8K (with tool use)
# - HumanEval (code generation)
# - MMLU
# - ChatCORE
```

**Expected results (d20):**
```
ARC-Challenge: ~0.28
ARC-Easy: ~0.35
GSM8K: ~0.02-0.03
HumanEval: ~0.06-0.07
MMLU: ~0.31
```

---

## Phase 4: Supervised Fine-Tuning

**Time**: ~20 minutes
**Purpose**: Improve conversational abilities

### Step 4.1: Run SFT

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft -- \
    --run=my_d20_model

# Single epoch training on:
# - Task mixture (same as midtraining)
# - Focus on improving accuracy
```

### Step 4.2: Evaluate SFT

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_eval -- -i sft

# Should see improvements:
# - GSM8K: ~0.04-0.05 (+2-3%)
# - HumanEval: ~0.08-0.09 (+2%)
# - Others: slight improvements
```

---

## Phase 5: Reinforcement Learning (Optional)

**Time**: ~20 minutes
**Purpose**: Further improve GSM8K via reward optimization

### Step 5.1: Run RL Training

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_rl -- \
    --run=my_d20_model

# GRPO (Group Relative Policy Optimization):
# 1. Samples multiple completions
# 2. Evaluates which are correct
# 3. Uses correctness as reward
# 4. Updates policy to favor correct answers
```

### Step 5.2: Evaluate RL

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_eval -- -i rl -a GSM8K

# Expected improvement:
# GSM8K: ~0.07-0.08 (+3-4% over SFT)
```

---

## Phase 6: Inference and Evaluation

### Step 6.1: Generate Final Report

```bash
# Compile all evaluation results
python -m nanochat.report generate

# This creates report.md with:
# - System information
# - Training hyperparameters
# - All evaluation results
# - Training time and cost
# - Code statistics

# View the report
cat report.md

# Copy to current directory for easy access
cp $NANOCHAT_BASE_DIR/report/report.md ./report.md
```

### Step 6.2: Test CLI Chat

```bash
# Chat with your model via command line
python -m scripts.chat_cli

# Interactive mode - type and press Enter
# Type 'exit' or 'quit' to stop

# Single query mode:
python -m scripts.chat_cli -p "Why is the sky blue?"
python -m scripts.chat_cli -p "Write a haiku about AI"
python -m scripts.chat_cli -p "Calculate 123 * 456"
```

### Step 6.3: Start Web Interface

```bash
# Start web server
python -m scripts.chat_web

# By default runs on: http://0.0.0.0:8000
```

**Access the UI:**

If running locally:
```
http://localhost:8000
```

If running on remote server (e.g., Lambda Labs):
```
http://YOUR_SERVER_IP:8000
```

**Features:**
- ChatGPT-like interface
- Streaming responses
- Conversation history
- Tool use (calculator, Python)

### Step 6.4: Sample Generations

```bash
# Generate samples and evaluate
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_loss

# Examples of what your model can do:

# ✓ Answer questions
"Q: What is the capital of France?
 A: The capital of France is Paris."

# ✓ Write stories
"Once upon a time, there was a young programmer..."

# ✓ Simple math (with calculator)
"Q: What is 15 * 23?
 A: <<15*23=345>> The answer is 345."

# ⚠ Limitations (d20 model):
# - Hallucinates frequently
# - Limited reasoning
# - Like a "kindergartener"
```

---

## Monitoring and Debugging

### Real-time Monitoring

```bash
# Terminal 1: Training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Terminal 2: GPU monitoring
watch -n 1 nvidia-smi

# Terminal 3: Loss tracking
tail -f speedrun.log | grep "loss"

# Terminal 4: Disk usage
watch -n 60 "df -h | grep -E '(Filesystem|cache)'"
```

### Key Metrics to Watch

**During pretraining:**
- **Loss**: Should decrease from ~11 to ~3.5-4.0
- **Learning rate**: Warms up, then decays
- **GPU utilization**: Should be 90-100%
- **Tokens/sec**: ~50,000-100,000 (varies by model size)

**During midtraining/SFT:**
- **Loss**: Lower, ~1-2
- **Task accuracy**: Check eval scores

### Checkpoints and Resuming

```bash
# Checkpoints saved at:
$NANOCHAT_BASE_DIR/checkpoints/

# Structure:
checkpoints/
├── base/
│   ├── model.pth
│   ├── optimizer.pth
│   └── metadata.json
├── mid/
├── sft/
└── rl/

# To resume training after interruption:
# Just re-run the same command - it auto-resumes!

# To manually specify checkpoint:
# Edit the script or use --resume flag if available
```

### Wandb Dashboard

If you set up Weights & Biases:

```bash
# During training, view at:
# https://wandb.ai/your-username/nanochat

# You can monitor:
# - Loss curves
# - Learning rate schedule
# - GPU utilization
# - Evaluation metrics
# - Sample generations
```

---

## Cost and Time Estimates

### d20 Model (561M params) - "The $100 Model"

**Training time:** ~4 hours total

| Phase | Time | Cost | Notes |
|-------|------|------|-------|
| Tokenizer | 30 min | $12 | One-time, reusable |
| Base pretraining | 2.5 hrs | $60 | Most expensive phase |
| Midtraining | 20 min | $8 | Quick adaptation |
| SFT | 15 min | $6 | Single epoch |
| RL (optional) | 15 min | $6 | GSM8K improvement |
| Evaluation | 20 min | $8 | All benchmarks |
| **Total** | **~4 hrs** | **~$100** | |

**Performance:**
- CORE: ~0.22 (below GPT-2)
- GSM8K: ~0.07 with RL
- Good for: Learning, experimentation

### d26 Model (1.1B params) - "The $300 Model"

**Training time:** ~12 hours total

| Phase | Time | Cost |
|-------|------|------|
| Tokenizer | 30 min | $12 |
| Base pretraining | 8 hrs | $192 |
| Midtraining | 1 hr | $24 |
| SFT | 30 min | $12 |
| RL (optional) | 30 min | $12 |
| Evaluation | 1 hr | $24 |
| **Total** | **~12 hrs** | **~$276** |

**Performance:**
- CORE: ~0.28 (GPT-2 level)
- GSM8K: ~0.15-0.20
- Good for: Serious experiments

### d32 Model (1.9B params) - "The $800 Model"

**Training time:** ~33 hours total

| Phase | Time | Cost |
|-------|------|------|
| Tokenizer | 30 min | $12 |
| Base pretraining | 22 hrs | $528 |
| Midtraining | 2 hrs | $48 |
| SFT | 1 hr | $24 |
| RL (optional) | 1 hr | $24 |
| Evaluation | 2 hrs | $48 |
| **Total** | **~33 hrs** | **~$684** |

**Performance:**
- CORE: ~0.32-0.35 (beyond GPT-2)
- GSM8K: ~0.25-0.30
- Good for: Best results on budget

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce device_batch_size
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=16  # Was 32

# Or even lower: 8, 4, 2

# Code automatically increases gradient accumulation
# to maintain same effective batch size
```

#### Issue 2: Slow Data Download

**Symptoms:**
- Dataset download taking very long
- Connection timeouts

**Solutions:**
```bash
# 1. Check internet connection
ping huggingface.co

# 2. Download shards manually
python -c "
from nanochat.dataset import download_shard
for i in range(240):
    download_shard(i)
    print(f'Downloaded shard {i}/240')
"

# 3. Use fewer shards (trains more epochs)
# Acceptable but slightly slower learning
```

#### Issue 3: NCCL Errors (Multi-GPU)

**Symptoms:**
```
NCCL error: unhandled system error
```

**Solutions:**
```bash
# Set NCCL debug flag
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Re-run training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# If persists, try single GPU first:
python -m scripts.base_train -- --depth=20
# (Will be 8x slower but helps isolate issue)
```

#### Issue 4: Checkpoint Corruption

**Symptoms:**
- Training fails to resume
- "checkpoint not found" errors

**Solutions:**
```bash
# Check checkpoint directory
ls -lh $NANOCHAT_BASE_DIR/checkpoints/base/

# If corrupted, remove and restart
rm -rf $NANOCHAT_BASE_DIR/checkpoints/base/

# Training will start from scratch
```

#### Issue 5: Poor Model Performance

**Symptoms:**
- Model generates nonsense
- Very low accuracy

**Possible causes:**
1. **Not enough data**: Verify shard count
   ```bash
   ls $NANOCHAT_BASE_DIR/data/*.parquet | wc -l
   ```

2. **Training not finished**: Check iteration count
   ```bash
   cat $NANOCHAT_BASE_DIR/checkpoints/base/metadata.json | jq .iteration
   ```

3. **Wrong checkpoint loaded**: Check checkpoint paths

4. **Tokenizer issue**: Verify tokenizer works
   ```bash
   python -m scripts.tok_eval
   ```

### Getting Help

1. **Check logs**: Always check training logs first
2. **GitHub Issues**: https://github.com/karpathy/nanochat/issues
3. **Discussions**: https://github.com/karpathy/nanochat/discussions
4. **DeepWiki**: https://deepwiki.com/karpathy/nanochat

---

## Quick Reference Commands

### One-Command Training (Recommended First Time)

```bash
# The speedrun script does everything for you
bash speedrun.sh

# Or with screen:
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# With wandb:
WANDB_RUN=my_run bash speedrun.sh
```

### Manual Step-by-Step

```bash
# 1. Setup
uv venv && source .venv/bin/activate
uv sync --extra gpu
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 2. Tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 240 &
python -m scripts.tok_train --max_chars=2000000000

# 3. Base model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# 4. Midtraining
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# 5. SFT
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 6. RL (optional)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K

# 7. Report
python -m nanochat.report generate

# 8. Inference
python -m scripts.chat_web
```

---

## Next Steps

After completing your first d20 training:

1. **Experiment with prompts**: Try different questions and tasks
2. **Analyze the report**: Understand strengths and weaknesses
3. **Scale up**: Try d26 or d32 for better performance
4. **Customize**: Add your own training data (see Discussions)
5. **Contribute**: Share your findings, improvements, or issues

---

## Additional Resources

- **Main Repo**: https://github.com/karpathy/nanochat
- **Tutorial (CN)**: [docs/training_guide_zh.md](training_guide_zh.md)
- **Pure Python Tokenizer**: [docs/pure_python_tokenizer.md](pure_python_tokenizer.md)
- **Customization Guide**: https://github.com/karpathy/nanochat/discussions/139
- **Adding Abilities**: https://github.com/karpathy/nanochat/discussions/164

---

**Happy training! 🚀**

If you successfully train a model, consider sharing your experience in GitHub Discussions!
