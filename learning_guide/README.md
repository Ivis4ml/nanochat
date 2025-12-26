# Nanochat Learning Guide

A step-by-step tutorial series for understanding and building a complete LLM from scratch. This guide covers the entire pipeline from tokenization to production serving, designed to run on Google Colab with a single A100 GPU (80GB).

## Overview

This learning guide is part of the **nanochat** project - a minimal, hackable implementation of a ChatGPT-like language model. The goal is to train, evaluate, and serve a complete LLM pipeline end-to-end, producing a 1.9B parameter model competitive with GPT-2 for ~$100-$1000 in compute.

## Prerequisites

- Basic Python programming
- Familiarity with PyTorch fundamentals
- Understanding of neural network basics
- Access to Google Colab (A100 80GB recommended)

## Tutorial Structure

The tutorials follow a **bottom-up** approach, building from fundamentals to advanced topics:

### Part 1: Foundations

| Module | Notebook | Topics Covered |
|--------|----------|----------------|
| 0 | `00_environment_setup.ipynb` | Colab setup, dependencies, GPU verification |
| 1 | `01_bpe_tokenizer.ipynb` | BPE algorithm, RustBPE training, tiktoken inference |
| 2 | `02_gpt_architecture.ipynb` | Transformer architecture, RoPE, MQA, QK-Norm |

### Part 2: Training Pipeline

| Module | Notebook | Topics Covered |
|--------|----------|----------------|
| 3 | `03_data_pipeline.ipynb` | FinWeb dataset, distributed data loading |
| 4 | `04_pretraining.ipynb` | Base model training, Muon optimizer, DDP |
| 5 | `05_midtraining.ipynb` | Conversation format, special tokens |
| 6 | `06_sft.ipynb` | Supervised fine-tuning on chat data |

### Part 3: Inference & Optimization

| Module | Notebook | Topics Covered |
|--------|----------|----------------|
| 7 | `07_inference_engine.ipynb` | KV Cache, efficient generation, tool use |
| 8 | `08_serving.ipynb` | FastAPI server, web UI, OpenAI API compatibility |

### Part 4: Evaluation

| Module | Notebook | Topics Covered |
|--------|----------|----------------|
| 9 | `09_evaluation.ipynb` | CORE benchmark, ARC, MMLU, GSM8K, HumanEval |

### Part 5: Advanced Topics (Integration with Mini-SGLang)

| Module | Notebook | Topics Covered |
|--------|----------|----------------|
| 10 | `10_paged_attention.ipynb` | Memory-efficient attention, page tables |
| 11 | `11_scheduling.ipynb` | Continuous batching, chunked prefill |
| 12 | `12_radix_cache.ipynb` | Prefix caching, LRU eviction |

## Quick Start

### Option 1: Run on Google Colab

1. Open the first notebook in Colab
2. Select Runtime → Change runtime type → A100 GPU
3. Follow the instructions in each notebook

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/Ivis4ml/nanochat.git
cd nanochat

# Install dependencies
pip install -e .

# Start Jupyter
jupyter notebook learning_guide/
```

## Learning Path

### Beginner Path (4-6 hours)
Start with notebooks 00-02 to understand the fundamentals:
- Environment setup and verification
- How tokenization works
- Transformer architecture deep dive

### Intermediate Path (8-12 hours)
Continue with notebooks 03-08 for the full training pipeline:
- Data processing and loading
- Pre-training and fine-tuning
- Inference optimization

### Advanced Path (4-6 hours)
Complete notebooks 09-12 for production-level understanding:
- Comprehensive evaluation
- SGLang-style optimizations
- Production deployment

## Hardware Requirements

| Stage | Minimum GPU | Recommended GPU |
|-------|-------------|-----------------|
| Notebooks 00-02 | T4 (16GB) | A100 (40GB) |
| Notebooks 03-06 | A100 (40GB) | A100 (80GB) |
| Notebooks 07-12 | T4 (16GB) | A100 (40GB) |

## Code Philosophy

This guide inherits nanochat's philosophy:
- **Minimal**: ~7,800 lines of Python across 45 files
- **Readable**: Clean, well-documented code
- **Hackable**: Easy to modify and experiment with
- **Educational**: Learn by building, not just reading

## Related Projects

- [nanochat](https://github.com/Ivis4ml/nanochat) - The main project
- [mini-sglang](https://github.com/Ivis4ml/mini-sglang) - Inference optimization reference

## Contributing

We welcome contributions! Please see the main nanochat repository for guidelines.

## License

MIT License - see the main repository for details.
