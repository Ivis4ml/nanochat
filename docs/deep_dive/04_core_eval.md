# Deep Dive: CORE Evaluation (`nanochat/core_eval.py`)

**File**: `nanochat/core_eval.py`
**Purpose**: Evaluate models using the CORE metric from DCLM paper
**Reference**: [DCLM Paper](https://arxiv.org/abs/2406.11794)
**Lines of Code**: ~263

---

## Table of Contents

1. [Overview](#overview)
2. [CORE Metric](#core-metric)
3. [Task Types](#task-types)
4. [Prompt Rendering](#prompt-rendering)
5. [Sequence Batching](#sequence-batching)
6. [Model Evaluation](#model-evaluation)
7. [Distributed Evaluation](#distributed-evaluation)
8. [Practical Examples](#practical-examples)

---

## Overview

CORE (Common Reasoning Evaluation) is a **standardized evaluation metric** for comparing language models, introduced in the DCLM (DataComp-LM) paper.

```
Evaluation Pipeline:
                                        ┌─────────────────────┐
Dataset Item                            │  Few-shot Examples  │
    │                                   └──────────┬──────────┘
    ▼                                              │
┌─────────────────────────────────────────────────────────────┐
│              Prompt Rendering (Jinja2)                       │
│  "Context: {fewshot}... Question: {item} Answer: {choice}"  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              Tokenization + Batching                         │
│  Identify continuation span for loss calculation            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Forward Pass                              │
│  Get per-token losses and predictions                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              Scoring                                         │
│  MC/Schema: lowest avg loss wins                            │
│  LM: exact match on continuation tokens                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Accuracy (0-100%)
```

---

## CORE Metric

### What is CORE?

CORE aggregates performance across multiple evaluation tasks to produce a **single comparable score** for language models.

**Included Tasks** (typical):
- **HellaSwag**: Commonsense reasoning (sentence completion)
- **PIQA**: Physical intuition QA
- **ARC-Easy/Challenge**: Science QA
- **WinoGrande**: Pronoun resolution
- **LAMBADA**: Word prediction
- **SQuAD**: Reading comprehension (extractive QA)

### Task Metadata

```python
task_meta = {
    'task_type': 'multiple_choice',    # or 'schema', 'language_modeling'
    'num_fewshot': 5,                   # Number of few-shot examples
    'continuation_delimiter': '\n'      # Separator between context and answer
}
```

---

## Task Types

### 1. Multiple Choice (`multiple_choice`)

**Format**: Same context, different answer choices

```
Question: What is the capital of France?
A) London
B) Paris      ← correct
C) Berlin
D) Madrid
```

**Evaluation**: Compute loss for each choice, select lowest loss

```python
item = {
    'query': "What is the capital of France?",
    'choices': ["London", "Paris", "Berlin", "Madrid"],
    'gold': 1  # Index of correct answer
}
```

### 2. Schema (`schema`)

**Format**: Different contexts, same continuation

```
Context A: The trophy doesn't fit in the suitcase because it is too big.
Context B: The trophy doesn't fit in the suitcase because it is too small.
Continuation: The trophy    ← which context makes more sense?
```

**Evaluation**: Compute loss for continuation given each context

```python
item = {
    'context_options': [
        "The trophy doesn't fit... it is too big.",
        "The trophy doesn't fit... it is too small."
    ],
    'continuation': "The trophy",
    'gold': 0  # Index of correct context
}
```

### 3. Language Modeling (`language_modeling`)

**Format**: Exact token prediction

```
Context: The quick brown fox jumps over the lazy
Target: dog
```

**Evaluation**: Exact match on predicted tokens

```python
item = {
    'context': "The quick brown fox jumps over the lazy",
    'continuation': "dog"
}
```

---

## Prompt Rendering

### Multiple Choice Prompts

**Location**: `nanochat/core_eval.py:17-33`

```python
def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
```

**Example Output** (2-shot):
```
What is 2+2?
4

What is 3+3?
6

What is the capital of France?
Paris
```

### Schema Prompts

**Location**: `nanochat/core_eval.py:36-53`

```python
def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render prompts for schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
```

### Language Modeling Prompts

**Location**: `nanochat/core_eval.py:56-83`

```python
def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render prompt for language modeling task"""
    # Returns TWO prompts: without and with continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]
```

**Purpose**:
- `prompt_without`: Used to find where continuation starts
- `prompt_with`: Full prompt for evaluation

---

## Sequence Batching

### Finding Common Spans

**Location**: `nanochat/core_eval.py:86-101`

```python
def find_common_length(token_sequences, direction='left'):
    """Find length of common prefix or suffix across sequences"""
    min_len = min(len(seq) for seq in token_sequences)

    indices = {
        'left': range(min_len),          # prefix: 0, 1, 2, ...
        'right': range(-1, -min_len-1, -1)  # suffix: -1, -2, -3, ...
    }[direction]

    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i  # First differing position

    return min_len
```

**Example**:
```python
# Multiple Choice: common prefix
tokens = [
    [1, 2, 3, 100],  # "Question... A"
    [1, 2, 3, 200],  # "Question... B"
    [1, 2, 3, 300],  # "Question... C"
]
find_common_length(tokens, 'left')  # Returns 3 (prefix length)

# Schema: common suffix
tokens = [
    [100, 5, 6, 7],  # "Context A... continuation"
    [200, 5, 6, 7],  # "Context B... continuation"
]
find_common_length(tokens, 'right')  # Returns 3 (suffix length)
```

### Multiple Choice Batching

```python
def batch_sequences_mc(tokenizer, prompts):
    """For MC: contexts are same, continuations differ"""
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())

    # Find where the common prefix ends (answer starts)
    answer_start_idx = find_common_length(tokens, direction='left')

    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]

    return tokens, start_indices, end_indices
```

**Visualization**:
```
Prompt 1: [BOS Question tokens... A tokens]
                              ↑
Prompt 2: [BOS Question tokens... B tokens]
                              ↑
Prompt 3: [BOS Question tokens... C tokens]
                              ↑
                        answer_start_idx
```

### Schema Batching

```python
def batch_sequences_schema(tokenizer, prompts):
    """For Schema: contexts vary, continuation is same"""
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())

    # Find common suffix (continuation)
    suffix_length = find_common_length(tokens, direction='right')

    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]

    return tokens, start_indices, end_indices
```

**Visualization**:
```
Prompt 1: [BOS Context A tokens... continuation tokens]
                                 ↑                    ↑
Prompt 2: [BOS Context B tokens... continuation tokens]
                                 ↑                    ↑
                           start_idx              end_idx
```

### Language Modeling Batching

```python
def batch_sequences_lm(tokenizer, prompts):
    """For LM: compare prompt with and without continuation"""
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens

    start_idx = len(tokens_without)  # Where continuation starts
    end_idx = len(tokens_with)        # End of full prompt

    # Only need the full prompt (batch size 1)
    return [tokens_with], [start_idx], [end_idx]
```

**Visualization**:
```
tokens_without: [BOS Context tokens...]
                                      ↑ start_idx

tokens_with:    [BOS Context tokens... continuation tokens]
                                      ↑                    ↑
                                 start_idx             end_idx
```

---

## Model Evaluation

### Forward Pass

**Location**: `nanochat/core_eval.py:144-164`

```python
@torch.no_grad()
def forward_model(model, input_ids):
    """
    Get per-token losses and predictions.

    Args:
        model: Language model
        input_ids: (B, T) tensor of token IDs

    Returns:
        losses: (B, T) tensor of cross-entropy losses
        predictions: (B, T) tensor of argmax predictions
    """
    batch_size, seq_len = input_ids.size()

    # Forward pass
    outputs = model(input_ids)  # (B, T, vocab_size)

    # Autoregressive targets (shifted left by 1)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)

    # Per-position cross-entropy loss
    losses = F.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)

    # Last position has no target (set to nan)
    losses[:, -1] = float('nan')

    # Argmax predictions
    predictions = outputs.argmax(dim=-1)

    return losses, predictions
```

**Loss Calculation**:
```
Input:   [BOS, t1, t2, t3, t4]
Target:  [t1,  t2, t3, t4, ???]  ← rolled left
Loss:    [L0,  L1, L2, L3, nan]

Li = -log P(target_i | input_{0:i})
```

### Single Example Evaluation

**Location**: `nanochat/core_eval.py:167-241`

```python
@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, return True if correct"""

    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']

    # 1. Sample few-shot examples (deterministic by idx)
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)  # Reproducible
        available = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # 2. Render prompts based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(...)
        tokens, start_idxs, end_idxs = batch_sequences_mc(...)
    elif task_type == 'schema':
        prompts = render_prompts_schema(...)
        tokens, start_idxs, end_idxs = batch_sequences_schema(...)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(...)
        tokens, start_idxs, end_idxs = batch_sequences_lm(...)

    # 3. Handle sequence length limits
    if hasattr(model, 'max_seq_len') and model.max_seq_len:
        # Truncate from left, adjust indices
        ...

    # 4. Stack and forward
    input_ids = stack_sequences(tokens, pad_token_id)
    losses, predictions = forward_model(model, input_ids)

    # 5. Score based on task type
    if task_type == 'language_modeling':
        # Exact match on continuation tokens
        predicted = predictions[0, si-1:ei-1]
        actual = input_ids[0, si:ei]
        is_correct = torch.all(predicted == actual).item()

    elif task_type in ['multiple_choice', 'schema']:
        # Lowest average loss wins
        mean_losses = [
            losses[i, si-1:ei-1].mean().item()
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))
        ]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = (pred_idx == item['gold'])

    return is_correct
```

### Scoring Methods

**Multiple Choice / Schema**:
```
For each option i:
  mean_loss[i] = mean(losses[i, start:end])

prediction = argmin(mean_loss)
correct = (prediction == gold_label)
```

**Language Modeling**:
```
For continuation span [start:end]:
  predicted_tokens = argmax(logits)[start-1:end-1]
  actual_tokens = input_ids[start:end]

correct = (predicted_tokens == actual_tokens)
```

**Index Offset Explanation**:
```
Position:    0    1    2    3    4
Input:      BOS  t1   t2   t3   t4
                           ↑    ↑
                       start  end

Prediction at position i predicts token at position i+1:
predictions[2] predicts input[3]

So for span [3:5], we check predictions[2:4] vs input[3:5]
```

---

## Distributed Evaluation

**Location**: `nanochat/core_eval.py:244-262`

```python
def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    Evaluate task across all examples with distributed support.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Results tensor
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)

    # Distribute examples across ranks
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, ...)
        correct[idx] = float(is_correct)

    # Sync across processes
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    # Compute accuracy
    mean_correct = correct.mean().item()
    return mean_correct
```

**Distribution Pattern**:
```
Dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]

GPU 0 (rank=0): [0, 4, 8, 12, ...]
GPU 1 (rank=1): [1, 5, 9, 13, ...]
GPU 2 (rank=2): [2, 6, 10, 14, ...]
GPU 3 (rank=3): [3, 7, 11, 15, ...]

After all_reduce:
All GPUs have complete results tensor
```

---

## Practical Examples

### Example 1: Basic Evaluation

```python
from nanochat.core_eval import evaluate_task
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# Load model and tokenizer
config = GPTConfig()
model = GPT(config).cuda()
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

tokenizer = get_tokenizer()

# Load dataset
data = [
    {'query': "What is 2+2?", 'choices': ["3", "4", "5"], 'gold': 1},
    {'query': "Capital of France?", 'choices': ["London", "Paris"], 'gold': 1},
    # ...
]

# Task metadata
task_meta = {
    'task_type': 'multiple_choice',
    'num_fewshot': 5,
    'continuation_delimiter': '\n'
}

# Evaluate
accuracy = evaluate_task(model, tokenizer, data, 'cuda', task_meta)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

### Example 2: Evaluate Single Example

```python
from nanochat.core_eval import evaluate_example

# Single item evaluation
item = {
    'query': "The Earth orbits the",
    'choices': ["Moon", "Sun", "Mars"],
    'gold': 1  # Sun
}

is_correct = evaluate_example(
    idx=0,
    model=model,
    tokenizer=tokenizer,
    data=[item],
    device='cuda',
    task_meta={
        'task_type': 'multiple_choice',
        'num_fewshot': 0,
        'continuation_delimiter': ' '
    }
)

print(f"Correct: {is_correct}")
```

### Example 3: Language Modeling Task

```python
# LAMBADA-style task
data = [
    {
        'context': "She opened the door and saw her best friend standing there with a huge",
        'continuation': " smile"
    },
    # ...
]

task_meta = {
    'task_type': 'language_modeling',
    'num_fewshot': 0,
    'continuation_delimiter': ''
}

accuracy = evaluate_task(model, tokenizer, data, 'cuda', task_meta)
print(f"LAMBADA accuracy: {accuracy * 100:.2f}%")
```

### Example 4: Schema Task (WinoGrande)

```python
# WinoGrande-style task
data = [
    {
        'context_options': [
            "The trophy doesn't fit in the suitcase because the trophy is too big.",
            "The trophy doesn't fit in the suitcase because the suitcase is too big."
        ],
        'continuation': " So we need a bigger suitcase.",
        'gold': 0
    },
    # ...
]

task_meta = {
    'task_type': 'schema',
    'num_fewshot': 5,
    'continuation_delimiter': ''
}

accuracy = evaluate_task(model, tokenizer, data, 'cuda', task_meta)
```

### Example 5: Distributed Evaluation

```python
# Run with: torchrun --nproc_per_node=4 eval_script.py

import torch.distributed as dist

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
local_rank = int(os.environ['LOCAL_RANK'])

torch.cuda.set_device(local_rank)

model = GPT(config).cuda()
model.load_state_dict(torch.load('checkpoint.pt'))

# Each GPU evaluates subset, results are synced
accuracy = evaluate_task(
    model, tokenizer, data,
    device=f'cuda:{local_rank}',
    task_meta=task_meta
)

if rank == 0:
    print(f"Accuracy: {accuracy * 100:.2f}%")
```

### Example 6: Custom Few-shot Examples

```python
from nanochat.core_eval import render_prompts_mc

item = {
    'query': "What color is grass?",
    'choices': ["blue", "green", "red"],
    'gold': 1
}

fewshot = [
    {'query': "What color is the sky?", 'choices': ["blue", "red"], 'gold': 0},
    {'query': "What color is blood?", 'choices': ["green", "red"], 'gold': 1},
]

prompts = render_prompts_mc(
    item,
    continuation_delimiter='\nAnswer: ',
    fewshot_examples=fewshot
)

for i, prompt in enumerate(prompts):
    print(f"=== Choice {i} ===")
    print(prompt)
    print()
```

**Output**:
```
=== Choice 0 ===
What color is the sky?
Answer: blue

What color is blood?
Answer: red

What color is grass?
Answer: blue

=== Choice 1 ===
What color is the sky?
Answer: blue

What color is blood?
Answer: red

What color is grass?
Answer: green

=== Choice 2 ===
...
```

---

## Key Takeaways

1. **Three Task Types**: Multiple choice, schema, language modeling

2. **Loss-Based Scoring**: MC/Schema select option with lowest average loss

3. **Exact Match**: Language modeling requires perfect token prediction

4. **Few-Shot Support**: Configurable number of in-context examples

5. **Distributed**: Automatic example distribution across GPUs

6. **Reproducible**: Deterministic few-shot sampling by example index

7. **Length Handling**: Truncation support for models with max sequence length

---

## References

- **DCLM Paper**: [DataComp-LM: In Search of the Next Generation of Training Data](https://arxiv.org/abs/2406.11794)
- **HellaSwag**: [Zellers et al., 2019](https://arxiv.org/abs/1905.07830)
- **LAMBADA**: [Paperno et al., 2016](https://arxiv.org/abs/1606.06031)
- **WinoGrande**: [Sakaguchi et al., 2020](https://arxiv.org/abs/1907.10641)

---

**Next Steps**:
- Read **01_gpt_architecture.md** to understand the model being evaluated
- Read **05_training_pipeline.md** for how evaluation fits into training

**Questions?** Check the main documentation or file an issue on GitHub!
