# Fine-Tuning Methodology for Domain-Specific LLMs

## Overview

This guide provides step-by-step instructions for fine-tuning Large Language Models on domain-specific emergency response data using **LoRA (Low-Rank Adaptation)** - the recommended approach for most practitioners.

**Expected Results**:
- Training time: 6-24 hours on RTX 4090 / 2-8 hours on A100
- Final model size: Base model + 100-500MB LoRA adapter
- Performance: 20-40% improvement on domain benchmarks vs. base model
- Cost: $50-200 (cloud GPU) or free (local GPU)

---

## Quick Reference

| Training Method | GPU Required | Dataset Size | Training Time | Cost | Quality |
|-----------------|--------------|--------------|---------------|------|---------|
| **LoRA** (Recommended) | RTX 4090 (24GB) | 1K-10K | 6-24h | $50-200 | ★★★★☆ |
| QLoRA | RTX 3090 (16GB) | 1K-10K | 12-48h | $100-300 | ★★★☆☆ |
| Full Fine-Tuning | A100 (40GB+) | 10K+ | 2-7 days | $500-2K | ★★★★★ |
| Short LLM (sLLM) | RTX 4090 | 500-5K | 2-6h | $20-50 | ★★★☆☆ |

---

## Prerequisites

Before starting, ensure you have:
- [x] Training data prepared (see `../data_collection/README.md`)
- [x] Development environment set up (see `../tools/setup_environment.sh`)
- [x] GPU with sufficient VRAM (16GB minimum)
- [x] Selected base model (Llama 3.1, Mistral, Qwen2, etc.)

---

## LoRA Fine-Tuning (Recommended Method)

### What is LoRA?

**LoRA (Low-Rank Adaptation)** injects trainable low-rank matrices into transformer layers instead of updating all model parameters. This allows:
- **10-100x fewer trainable parameters** (e.g., 20M vs 7B)
- **Much lower GPU memory requirements** (24GB vs 80GB)
- **Faster training** (hours vs days)
- **Smaller output files** (100MB vs 14GB)
- **Preserves base model knowledge** (less forgetting)

**Trade-off**: Slightly lower performance than full fine-tuning (~5-10% worse) but still excellent for most use cases.

---

## Step-by-Step LoRA Training

### Step 1: Choose Your Base Model

**For Greek Emergency Response**, recommended base models:

| Model | Size | Strengths | Download |
|-------|------|-----------|----------|
| **Llama 3.1 8B Instruct** | 8B | Best overall, strong reasoning | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| **Mistral 7B Instruct v0.3** | 7B | Fast inference, good quality | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Qwen2-7B-Instruct** | 7B | Multilingual (Greek support) | `Qwen/Qwen2-7B-Instruct` |
| **Gemma 2 9B** | 9B | Latest from Google, strong | `google/gemma-2-9b-it` |

**Note**: Llama 3.1 8B is the recommended starting point (best quality/speed trade-off).

---

### Step 2: Install Training Framework

We recommend **Axolotl** for its ease of use and excellent LoRA support:

```bash
# Install Axolotl
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install packaging ninja
pip install -e '.[flash-attn,deepspeed]'

# Verify installation
axolotl version
```

**Alternative**: LLaMA Factory (more GUI-friendly) or Unsloth (faster training).

---

### Step 3: Prepare Your Dataset

Your data should be in **JSONL format** with one of these structures:

#### ChatML Format (Recommended)
```jsonl
{"messages": [{"role": "system", "content": "You are a Greek fire commander..."}, {"role": "user", "content": "How do I..."}, {"role": "assistant", "content": "First, you should..."}]}
{"messages": [{"role": "system", "content": "You are a Greek fire commander..."}, {"role": "user", "content": "What about..."}, {"role": "assistant", "content": "In that case..."}]}
```

#### Alpaca Format
```jsonl
{"instruction": "You are a Greek fire commander. How do I...", "input": "", "output": "First, you should..."}
{"instruction": "You are a Greek fire commander. What about...", "input": "", "output": "In that case..."}
```

**Convert your data** if needed:
```bash
python ../tools/data_formatter.py \
  --input raw_data.json \
  --output formatted_data.jsonl \
  --format chatml
```

**Split into train/validation**:
```bash
python ../tools/split_dataset.py \
  --input formatted_data.jsonl \
  --train_output train.jsonl \
  --val_output val.jsonl \
  --split_ratio 0.9
```

---

### Step 4: Configure Training Parameters

Create a config file `configs/firefighter_lora_7b.yml`:

```yaml
# Base model
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

# Dataset
datasets:
  - path: ../data_collection/dataset_templates/firefighter_train.jsonl
    type: chat_template  # Use model's built-in chat template

val_set_size: 0.1  # 10% validation split

# LoRA configuration
adapter: lora
lora_r: 32  # Rank (higher = more capacity but slower)
lora_alpha: 64  # Alpha (typically 2x rank)
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training hyperparameters
sequence_len: 2048  # Max context length
batch_size: 2  # Per GPU (reduce if OOM)
gradient_accumulation_steps: 8  # Effective batch size = 2 * 8 = 16
micro_batch_size: 2
num_epochs: 3
learning_rate: 2e-4
lr_scheduler: cosine
warmup_steps: 100
optimizer: adamw_torch
weight_decay: 0.01

# Memory optimization
gradient_checkpointing: true
bf16: true  # Use bfloat16 (requires Ampere GPU or newer)
fp16: false
tf32: true

# Logging and saving
output_dir: ./outputs/firefighter-lora-llama3.1-8b
logging_steps: 10
eval_steps: 100
save_steps: 500
save_total_limit: 3

# Evaluation
eval_batch_size: 4
evals_per_epoch: 4

# Special tokens (adjust for your model)
special_tokens:
  pad_token: <|end_of_text|>
```

**Key Parameters Explained**:
- **lora_r**: Rank of LoRA matrices. Higher = more capacity. Range: 8-64. Start with 32.
- **lora_alpha**: Scaling factor. Rule of thumb: `alpha = 2 * r`
- **batch_size**: Larger = faster but more memory. Reduce if you get CUDA OOM errors.
- **learning_rate**: 2e-4 is standard for LoRA. Can try 1e-4 (conservative) or 5e-4 (aggressive).
- **num_epochs**: 3 is typical. More may overfit on small datasets.

---

### Step 5: Run Training

```bash
accelerate launch -m axolotl.cli.train configs/firefighter_lora_7b.yml
```

**Monitor training**:
- Watch for decreasing loss (should go from ~2.5 → ~0.5 for well-fit model)
- Check validation loss doesn't increase (sign of overfitting)
- Training on 1,000 examples, 3 epochs: ~6-12 hours on RTX 4090

**Expected output**:
```
[2025-11-13 10:23:45] Epoch 1/3, Step 100/375, Loss: 1.234, LR: 1.8e-4
[2025-11-13 10:45:12] Epoch 1/3, Step 200/375, Loss: 0.876, LR: 2.0e-4
[2025-11-13 11:12:33] Validation loss: 0.823
...
[2025-11-13 16:34:22] Epoch 3/3, Step 375/375, Loss: 0.432
[2025-11-13 16:35:10] Final validation loss: 0.456
[2025-11-13 16:35:45] Training complete! Model saved to ./outputs/firefighter-lora-llama3.1-8b
```

---

### Step 6: Merge LoRA Adapter (Optional)

**Option A: Keep Separate** (Recommended for experimentation)
- Base model: 14GB (Llama 3.1 8B)
- LoRA adapter: 200MB
- Load both at inference time

**Option B: Merge into Single Model** (Recommended for deployment)
```bash
python -m axolotl.cli.merge_lora \
  --config configs/firefighter_lora_7b.yml \
  --lora_model_dir ./outputs/firefighter-lora-llama3.1-8b \
  --output_dir ./merged_models/firefighter-llama3.1-8b
```

Merged model is a standard Hugging Face model (14.2GB) that can be used anywhere.

---

### Step 7: Test Your Model

Quick inference test:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_path = "./outputs/firefighter-lora-llama3.1-8b"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    load_in_8bit=True,  # Quantize to fit in memory
    device_map="auto"
)

# Load LoRA adapter
from peft import PeftModel
model = PeftModel.from_pretrained(model, model_path)

# Test
messages = [
    {"role": "system", "content": "You are Pyragos Ioanna Michaelidou, a Greek fire commander."},
    {"role": "user", "content": "A wildfire is approaching a village. Wind is 40 km/h. What do I do?"}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## Troubleshooting Common Issues

### CUDA Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `batch_size` (try 1)
2. Enable `gradient_checkpointing: true`
3. Reduce `sequence_len` (try 1024)
4. Use QLoRA instead (4-bit quantization)
5. Use smaller base model (7B instead of 13B)

---

### Training Loss Not Decreasing

**Symptoms**: Loss stays constant or increases

**Solutions**:
1. Check data formatting (ensure it matches model's chat template)
2. Reduce learning rate (try 1e-4)
3. Increase `lora_r` rank (try 64)
4. Check for data quality issues (duplicates, errors)
5. Ensure sufficient dataset size (500+ examples minimum)

---

### Model Outputs Nonsense

**Symptoms**: Gibberish or repetitive text

**Solutions**:
1. Overtraining: Reduce epochs to 1-2
2. Too high learning rate: Try 1e-4 or 5e-5
3. Data format mismatch: Verify chat template
4. Check tokenizer special tokens configuration

---

### Validation Loss Increasing (Overfitting)

**Symptoms**: Training loss decreases but validation loss increases

**Solutions**:
1. Stop training early (use fewer epochs)
2. Increase `lora_dropout` to 0.1
3. Add more training data
4. Reduce model capacity (`lora_r` to 16)
5. Use weight decay (`weight_decay: 0.01`)

---

## Hyperparameter Tuning Guide

### Conservative (Safe but Slower Learning)
```yaml
learning_rate: 1e-4
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
num_epochs: 2
```

### Balanced (Recommended Default)
```yaml
learning_rate: 2e-4
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
num_epochs: 3
```

### Aggressive (Fast Learning, Risk of Overfitting)
```yaml
learning_rate: 5e-4
lora_r: 64
lora_alpha: 128
lora_dropout: 0.0
num_epochs: 5
```

---

## Alternative Training Frameworks

### LLaMA Factory (GUI-friendly)

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e .

# Launch web UI
llamafactory-cli webui
```

Advantages: Easy-to-use web interface, good for beginners

---

### Unsloth (2x Faster Training)

```bash
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Train (same as standard Hugging Face Trainer)
```

Advantages: 2x faster than Axolotl, memory efficient

---

## Catastrophic Forgetting Mitigation

### What is Catastrophic Forgetting?

**Catastrophic forgetting** occurs when a model fine-tuned on a narrow domain loses its general knowledge and capabilities. For example, a firefighter-specialized model might:
- Forget basic math, logic, or common sense reasoning
- Lose multilingual abilities
- Decline on general benchmarks (MMLU, BBH, TruthfulQA)
- Only respond well to emergency-related queries

**Why This Matters**: Emergency response AI must maintain general intelligence (e.g., understanding weather reports, calculating distances, reasoning about resource allocation) while gaining domain expertise.

---

### Detection: Measuring Forgetting

**Before and after fine-tuning**, evaluate on general benchmarks:

```python
from transformers import pipeline
from datasets import load_dataset

def measure_general_performance(model_path):
    """Evaluate model on general knowledge tasks."""

    # Test on MMLU (Massive Multitask Language Understanding)
    mmlu_dataset = load_dataset("cais/mmlu", "all", split="test[:100]")

    # Test on BBH (Big-Bench Hard)
    bbh_dataset = load_dataset("lukaemon/bbh", "boolean_expressions", split="test[:100]")

    # Test on common sense reasoning
    commonsense_dataset = load_dataset("commonsense_qa", split="validation[:100]")

    results = {
        "mmlu_score": evaluate_mmlu(model_path, mmlu_dataset),
        "bbh_score": evaluate_bbh(model_path, bbh_dataset),
        "commonsense_score": evaluate_commonsense(model_path, commonsense_dataset)
    }

    return results

# Before fine-tuning
base_scores = measure_general_performance("meta-llama/Meta-Llama-3.1-8B-Instruct")

# After fine-tuning
finetuned_scores = measure_general_performance("./outputs/firefighter-lora-llama3.1-8b")

# Check for catastrophic forgetting
for task, score in finetuned_scores.items():
    base_score = base_scores[task]
    degradation = ((base_score - score) / base_score) * 100

    if degradation > 5.0:
        print(f"⚠️ WARNING: {task} degraded by {degradation:.1f}%")
        print(f"   Base: {base_score:.1f}% → Fine-tuned: {score:.1f}%")
    else:
        print(f"✓ {task}: {base_score:.1f}% → {score:.1f}% ({degradation:+.1f}%)")
```

**Acceptable Degradation**: <5% on general benchmarks is acceptable. >10% indicates severe catastrophic forgetting.

---

### Mitigation Strategy 1: Mix General and Domain Data (Recommended)

**Approach**: Include general instruction-following data alongside domain-specific data during fine-tuning.

**Ratio**: 80% domain-specific + 20% general (adjust based on forgetting severity)

**Implementation**:

```python
import json
import random
from datasets import load_dataset

def create_mixed_dataset(domain_path, output_path, general_ratio=0.2):
    """Mix domain data with general instruction data."""

    # Load domain-specific data
    with open(domain_path, 'r') as f:
        domain_data = [json.loads(line) for line in f]

    # Load general instruction data (high-quality sources)
    general_sources = [
        load_dataset("Open-Orca/OpenOrca", split="train[:10000]"),
        load_dataset("timdettmers/openassistant-guanaco", split="train[:5000]"),
        load_dataset("Intel/orca_dpo_pairs", split="train[:5000]")
    ]

    # Convert general data to same format as domain data
    general_data = []
    for dataset in general_sources:
        for example in dataset:
            general_data.append({
                "messages": [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["response"]}
                ]
            })

    # Calculate target counts
    num_domain = len(domain_data)
    num_general = int(num_domain * general_ratio / (1 - general_ratio))

    # Sample general data
    general_sample = random.sample(general_data, min(num_general, len(general_data)))

    # Combine and shuffle
    mixed_data = domain_data + general_sample
    random.shuffle(mixed_data)

    # Save
    with open(output_path, 'w') as f:
        for example in mixed_data:
            f.write(json.dumps(example) + '\n')

    print(f"Created mixed dataset:")
    print(f"  Domain examples: {num_domain} ({100 * (1 - general_ratio):.0f}%)")
    print(f"  General examples: {len(general_sample)} ({100 * len(general_sample) / len(mixed_data):.0f}%)")
    print(f"  Total: {len(mixed_data)} examples")

# Usage
create_mixed_dataset(
    domain_path="../data_collection/dataset_templates/firefighter_train.jsonl",
    output_path="../data_collection/dataset_templates/firefighter_mixed_train.jsonl",
    general_ratio=0.2  # 20% general, 80% domain
)
```

**Update your training config**:
```yaml
datasets:
  - path: ../data_collection/dataset_templates/firefighter_mixed_train.jsonl
    type: chat_template
```

**Expected Results**:
- Domain performance: Same or slightly better (domain data still dominant)
- General performance: Maintained within 2-3% of base model
- Training time: +20-30% (more data to process)

---

### Mitigation Strategy 2: Replay Buffer Technique

**Approach**: Periodically sample examples from general datasets during training to "remind" the model of general knowledge.

**Implementation** (using Axolotl's built-in dataset mixing):

```yaml
# configs/firefighter_lora_with_replay.yml

datasets:
  # Primary domain data (80% weight)
  - path: ../data_collection/dataset_templates/firefighter_train.jsonl
    type: chat_template
    weight: 0.8

  # General knowledge replay buffer (20% weight)
  - path: hf://datasets/Open-Orca/OpenOrca
    type: chat_template
    weight: 0.15
    max_samples: 5000  # Limit to prevent overwhelming domain data

  - path: hf://datasets/timdettmers/openassistant-guanaco
    type: sharegpt
    weight: 0.05
    max_samples: 2000
```

**Advantages**:
- No manual dataset creation needed
- Automatic balancing via weights
- Can adjust ratios easily

**Disadvantages**:
- Requires Hugging Face datasets access
- Slower training (more data)

---

### Mitigation Strategy 3: Regularization Techniques

**Elastic Weight Consolidation (EWC)**: Penalize changes to weights important for general tasks.

**Implementation** (advanced):

```python
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

class EWCLoRAModel:
    """LoRA with Elastic Weight Consolidation."""

    def __init__(self, base_model, fisher_matrix, old_params, ewc_lambda=0.4):
        self.model = base_model
        self.fisher = fisher_matrix  # Importance of each parameter
        self.old_params = old_params  # Base model parameters
        self.ewc_lambda = ewc_lambda  # Regularization strength

    def ewc_loss(self):
        """Calculate EWC penalty for parameter changes."""
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                # Penalize changes to important parameters
                loss += (self.fisher[name] * (param - self.old_params[name]).pow(2)).sum()
        return self.ewc_lambda * loss

    def training_step(self, batch):
        """Add EWC loss to normal training loss."""
        outputs = self.model(**batch)
        loss = outputs.loss + self.ewc_loss()
        return loss
```

**Note**: EWC is complex and may not be necessary for LoRA (which already has implicit regularization). Consider only for severe forgetting cases.

---

### Mitigation Strategy 4: Lower Learning Rate for LoRA

**Approach**: Use a smaller learning rate to make gentler updates that preserve more base knowledge.

```yaml
# More conservative fine-tuning (less forgetting)
learning_rate: 1e-4  # Instead of 2e-4
lora_dropout: 0.1    # Higher dropout = more regularization
num_epochs: 2        # Fewer epochs = less specialization
```

**Trade-off**: May require more epochs to achieve same domain performance.

---

### Recommended Mitigation Pipeline

**For most practitioners**, use this combination:

1. **Mix 20% general data** (Strategy 1) - Easy to implement, highly effective
2. **Lower learning rate to 1e-4** (Strategy 4) - Simple config change
3. **Evaluate on general benchmarks** (Detection) - Verify no severe degradation
4. **If forgetting >5%**: Increase general data to 30-40%

**Implementation Checklist**:
- [x] Create mixed dataset with 80/20 domain/general ratio
- [x] Update training config to use mixed dataset
- [x] Reduce learning rate to 1e-4 or 5e-5
- [x] Evaluate base model on MMLU, BBH, CommonsenseQA before training
- [x] Evaluate fine-tuned model on same benchmarks after training
- [x] Verify degradation <5% on all general benchmarks
- [x] If degradation >5%, retrain with 30% general data

---

### Example: Before/After Forgetting Mitigation

**Scenario**: Firefighter LLM trained on 2,500 domain examples

| Metric | Base Model | Fine-tuned (No Mitigation) | Fine-tuned (With Mitigation) |
|--------|------------|----------------------------|------------------------------|
| **Domain Accuracy** | 58.2% | 81.7% | 80.3% |
| **MMLU (General)** | 62.4% | 48.1% ⚠️ (-23%) | 60.8% ✓ (-2.6%) |
| **BBH (Reasoning)** | 54.3% | 39.2% ⚠️ (-28%) | 53.1% ✓ (-2.2%) |
| **CommonsenseQA** | 71.2% | 62.4% ⚠️ (-12%) | 69.8% ✓ (-2.0%) |

**Without mitigation**: Model gained +23.5% domain accuracy but lost -21% general performance (catastrophic forgetting)

**With mitigation** (20% general data mix + lower LR): Model gained +22.1% domain accuracy while maintaining general performance (-2.3% average, acceptable)

**Cost**: +2 hours training time, -1.4% domain accuracy

**Verdict**: Mitigation is worth the trade-off for production deployment.

---

### When Can You Skip Mitigation?

**Skip catastrophic forgetting mitigation if**:
1. Your model will ONLY be used for narrow domain queries (e.g., dedicated HAZMAT lookup tool)
2. You're building a Short LLM (sLLM) intentionally limited to one task
3. You have a separate general-purpose model available for non-domain queries

**Always use mitigation if**:
1. Model will interact with end-users in open-ended conversations
2. Model needs to reason about general scenarios (weather, logistics, calculations)
3. Model will be evaluated by human experts (who expect general competence)
4. Model will be used in safety-critical contexts (must maintain reasoning ability)

---

## Learning Rate Scheduling

### Why Use Learning Rate Scheduling?

**Problem**: Fixed learning rate may be:
- Too high initially (unstable training, overshooting)
- Too low later (slow convergence, underfitting)

**Solution**: Gradually adjust learning rate during training for optimal convergence.

---

### Recommended Schedulers for LoRA

#### 1. Cosine Annealing (Recommended)

**How it works**: Learning rate smoothly decreases from initial value to near-zero following a cosine curve.

```yaml
# In your Axolotl config
lr_scheduler: cosine
learning_rate: 2e-4
warmup_steps: 100
num_epochs: 3
```

**Visualization**:
```
LR
 │
2e-4│     ╭───╮
    │    ╱     ╲
    │   ╱       ╲
    │  ╱         ╲___
0   │─────────────────→ Steps
      Warmup  Cosine Decay
```

**Advantages**:
- Smooth, stable training
- Works well for most LoRA tasks
- Good default choice

**Use when**: You want reliable, stable training (90% of cases)

---

#### 2. Linear Warmup + Constant

**How it works**: LR increases linearly for warmup steps, then stays constant.

```yaml
lr_scheduler: constant_with_warmup
learning_rate: 2e-4
warmup_steps: 100
```

**Advantages**:
- Simple and predictable
- Good for small datasets

**Use when**: You have <1,000 training examples

---

#### 3. Linear Decay

**How it works**: LR decreases linearly from initial value to zero.

```yaml
lr_scheduler: linear
learning_rate: 2e-4
warmup_steps: 100
```

**Advantages**:
- Gentle learning rate reduction
- Good for large datasets

**Use when**: You have >10,000 training examples or training for many epochs

---

#### 4. Cosine with Restarts (Advanced)

**How it works**: LR follows cosine decay, then restarts periodically to escape local minima.

```yaml
lr_scheduler: cosine_with_restarts
learning_rate: 2e-4
warmup_steps: 100
lr_scheduler_kwargs:
  num_cycles: 3  # Number of restarts
```

**Visualization**:
```
LR
 │
2e-4│  ╭─╮   ╭─╮   ╭─╮
    │ ╱  ╲  ╱  ╲  ╱  ╲
    │╱    ╲╱    ╲╱    ╲___
0   │─────────────────────→ Steps
      Restart Restart Restart
```

**Advantages**:
- Can escape local minima
- Often achieves better final loss

**Disadvantages**:
- Less stable training
- Harder to tune

**Use when**: You're stuck at high validation loss with other schedulers

---

### Warmup Steps Explained

**Warmup** = Gradual increase from very small LR to target LR at the start of training.

**Why?** Prevents exploding gradients and unstable training in early steps when model is far from optimal.

**How many warmup steps?**
- Small datasets (<1K examples): 50-100 steps
- Medium datasets (1K-10K): 100-500 steps
- Large datasets (>10K): 500-1000 steps

**Rule of thumb**: 5-10% of total training steps

```python
total_steps = (num_examples / batch_size / gradient_accumulation_steps) * num_epochs
warmup_steps = int(0.05 * total_steps)  # 5% of total
```

**Example**:
- 2,500 examples, batch_size=2, gradient_accumulation_steps=8, epochs=3
- Total steps = (2500/2/8) * 3 = 469 steps
- Warmup = 0.05 * 469 ≈ **25 steps**

---

### Complete LR Scheduling Example

```yaml
# configs/firefighter_lora_with_scheduling.yml

# Learning rate configuration
learning_rate: 2e-4
lr_scheduler: cosine
warmup_steps: 100
lr_scheduler_kwargs:
  min_lr: 1e-5  # Don't go below this LR

# Alternative: Cosine with restarts
# lr_scheduler: cosine_with_restarts
# lr_scheduler_kwargs:
#   num_cycles: 2
#   min_lr: 1e-5

# Monitor LR during training
logging_steps: 10  # Log LR every 10 steps
```

**View LR during training**:
```bash
# Training output shows current LR
[2025-11-13 10:23:45] Epoch 1/3, Step 100/375, Loss: 1.234, LR: 1.8e-4
[2025-11-13 10:45:12] Epoch 1/3, Step 200/375, Loss: 0.876, LR: 1.5e-4
[2025-11-13 11:12:33] Epoch 2/3, Step 250/375, Loss: 0.654, LR: 1.2e-4
```

---

### Early Stopping (Prevent Overfitting)

**Problem**: Training for too many epochs causes overfitting (validation loss increases).

**Solution**: Automatically stop training when validation loss stops improving.

**Implementation** (using Hugging Face Transformers callbacks):

```python
from transformers import TrainerCallback, EarlyStoppingCallback

# Add to your training script
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # Stop if no improvement for 3 eval cycles
    early_stopping_threshold=0.01  # Minimum improvement to count as "better"
)

# In Axolotl config
callbacks:
  - type: early_stopping
    patience: 3
    threshold: 0.01
```

**How it works**:
1. Evaluate validation loss every `eval_steps` (e.g., every 100 steps)
2. If validation loss doesn't improve by >0.01 for 3 consecutive evaluations, stop training
3. Save best checkpoint (lowest validation loss)

**Example**:
```
Eval 1: val_loss = 0.823 (saved as best)
Eval 2: val_loss = 0.651 (saved as best, improved by 0.172)
Eval 3: val_loss = 0.589 (saved as best, improved by 0.062)
Eval 4: val_loss = 0.598 (no improvement, patience = 1/3)
Eval 5: val_loss = 0.612 (no improvement, patience = 2/3)
Eval 6: val_loss = 0.627 (no improvement, patience = 3/3)
→ Training stopped! Best checkpoint: Eval 3 (val_loss = 0.589)
```

**Benefits**:
- Prevents overfitting automatically
- Saves time (don't train unnecessary epochs)
- Ensures best model is used (not last model)

---

### Learning Rate Finder (Advanced)

**Use case**: Not sure what learning rate to use? Run LR finder to discover optimal range.

```python
from transformers import Trainer
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

def find_lr(model, train_dataset, start_lr=1e-7, end_lr=1e-1, num_steps=100):
    """Find optimal learning rate using LR range test."""

    lrs = []
    losses = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)

    # Exponentially increase LR from start_lr to end_lr
    lr_lambda = lambda step: (end_lr / start_lr) ** (step / num_steps)
    scheduler = LambdaLR(optimizer, lr_lambda)

    for step in range(num_steps):
        batch = next(iter(train_loader))
        outputs = model(**batch)
        loss = outputs.loss

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Plot results
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('LR Finder: Choose LR where loss decreases fastest')
    plt.savefig('lr_finder.png')

    # Find optimal LR (steepest descent point)
    gradients = np.gradient(losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]

    print(f"Suggested learning rate: {optimal_lr:.2e}")
    return optimal_lr

# Usage
optimal_lr = find_lr(model, train_dataset)
# → Suggested learning rate: 2.3e-4
```

**Interpretation**:
- **Flat region (left)**: LR too low, learning is slow
- **Steep descent**: Optimal LR range (choose middle of this region)
- **Increasing loss (right)**: LR too high, training unstable

**Use this LR** in your config:
```yaml
learning_rate: 2.3e-4  # From LR finder
```

---

### Recommended LR Configurations by Use Case

#### Small Dataset (<1K examples)
```yaml
learning_rate: 1e-4
lr_scheduler: constant_with_warmup
warmup_steps: 50
num_epochs: 3-5
```

#### Medium Dataset (1K-10K examples) - RECOMMENDED
```yaml
learning_rate: 2e-4
lr_scheduler: cosine
warmup_steps: 100
num_epochs: 3
```

#### Large Dataset (>10K examples)
```yaml
learning_rate: 3e-4
lr_scheduler: cosine
warmup_steps: 500
num_epochs: 2-3
```

#### Aggressive (Fastest Training, Higher Risk)
```yaml
learning_rate: 5e-4
lr_scheduler: cosine_with_restarts
lr_scheduler_kwargs:
  num_cycles: 2
warmup_steps: 100
num_epochs: 5
```

---

## Next Steps

After training:
1. **Evaluate**: See `../evaluation/README.md` for benchmarking
2. **Check for forgetting**: Test on general benchmarks (MMLU, BBH)
3. **Optimize**: Quantize to GGUF format (see `../tools/model_quantizer.py`)
4. **Deploy**: Set up inference server (see `../DEPLOYMENT.md`)
5. **Iterate**: Collect more data, retrain with improved dataset

---

**Generated**: 2025-11-13
**Version**: 1.2
