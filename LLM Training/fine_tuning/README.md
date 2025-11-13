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

## Next Steps

After training:
1. **Evaluate**: See `../evaluation/README.md` for benchmarking
2. **Optimize**: Quantize to GGUF format (see `../tools/model_quantizer.py`)
3. **Deploy**: Set up inference server (see `../DEPLOYMENT.md`)
4. **Iterate**: Collect more data, retrain with improved dataset

---

**Generated**: 2025-11-13
**Version**: 1.0
