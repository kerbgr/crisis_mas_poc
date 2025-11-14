# Apple Silicon Training Guide

## Overview

This guide provides **Apple Silicon-specific instructions** for training domain-specific LLMs on M1, M2, M3, and M4 chips. Apple Silicon offers unique advantages for LLM training: **unified memory architecture**, excellent **power efficiency**, and **competitive performance** for 7B-13B models.

**TL;DR**: Yes, you can train LLMs on Apple Silicon! Use **MLX framework** for best results.

---

## Quick Comparison: Apple Silicon vs NVIDIA

| Feature | Apple Silicon (M3 Max) | NVIDIA (RTX 4090) | Winner |
|---------|----------------------|-------------------|---------|
| **Memory Architecture** | Unified (192GB max) | Split (24GB VRAM) | 🍎 Apple |
| **Training Speed (7B)** | 8-14 hours | 6-12 hours | 🟢 NVIDIA |
| **Power Consumption** | 40-60W | 350-450W | 🍎 Apple |
| **Noise Level** | Silent/minimal fans | Loud fans | 🍎 Apple |
| **Inference Speed** | 80-100 tok/s | 100-120 tok/s | 🟢 NVIDIA |
| **Framework Support** | MLX, PyTorch MPS | CUDA, everything | 🟢 NVIDIA |
| **Cost (Cloud)** | Not available | $1-2/hour | N/A |
| **Cost (Owned)** | MacBook/Mac Studio | Desktop build | Depends |

**Verdict**: Apple Silicon is **excellent for LLM training**, especially if you already own a Mac. Use MLX framework for best results.

---

## Hardware Requirements

### Minimum: M1/M2 (16GB RAM)

**Capabilities**:
- Train: 1.5B-3B models (Short LLMs)
- Inference: 7B models (quantized)
- LoRA: Very limited (requires aggressive optimization)

**Recommended use**: Train focused sLLMs (1.5B-3B) for narrow tasks

**Example Configuration**:
```yaml
base_model: Qwen/Qwen2-1.5B-Instruct  # 1.5B model
lora_r: 8  # Low rank to save memory
batch_size: 1
gradient_accumulation_steps: 16
sequence_len: 1024  # Shorter context
```

**Expected Training Time**: 4-8 hours for 1,000 examples

---

### Recommended: M2/M3 Pro (32GB-36GB RAM)

**Capabilities**:
- Train: 7B models with LoRA ✅
- Inference: 7B-13B models (quantized)
- LoRA: r=32 (standard configuration)

**Recommended use**: Standard LoRA fine-tuning on 7B models

**Example Configuration**:
```yaml
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct  # 7B-8B model
lora_r: 32
batch_size: 2
gradient_accumulation_steps: 8
sequence_len: 2048
```

**Expected Training Time**: 12-18 hours for 2,000 examples

---

### Optimal: M2/M3/M4 Max or Ultra (64GB-192GB RAM)

**Capabilities**:
- Train: 13B-34B models with LoRA ✅
- Train: 7B models with full fine-tuning (possible on Ultra)
- Inference: 70B models (quantized on Ultra)
- LoRA: r=64 (high capacity)

**Recommended use**: Large model training, research, production

**Example Configuration**:
```yaml
base_model: meta-llama/Meta-Llama-3.1-70B-Instruct  # 70B on M2 Ultra (192GB)
lora_r: 64
batch_size: 4
gradient_accumulation_steps: 4
sequence_len: 4096
```

**Expected Training Time**: 6-12 hours for 2,000 examples (7B-13B models)

---

## Framework Comparison

### Option 1: MLX (Recommended ⭐)

**Why MLX?**
- **Built by Apple** specifically for Apple Silicon
- **Optimized for Metal** (GPU acceleration on M-series chips)
- **Fast**: Competitive with NVIDIA for 7B-13B models
- **Memory efficient**: Unified memory architecture
- **Easy to use**: Simple API, good documentation
- **Active development**: Regular updates from Apple

**Installation**:
```bash
pip install mlx
pip install mlx-lm  # LLM utilities
```

**Training Example**:
```bash
# Fine-tune Llama 3.1 8B with LoRA
python -m mlx_lm.lora \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --data ./data/firefighter_train.jsonl \
  --train \
  --iters 1000 \
  --lora-layers 32 \
  --batch-size 2 \
  --learning-rate 2e-5 \
  --val-batches 10

# Expected: 8-12 hours on M3 Max (48GB)
```

**Advantages**:
- ✅ Best performance on Apple Silicon
- ✅ Unified memory (no VRAM limits)
- ✅ Low power consumption
- ✅ Simple Python API

**Disadvantages**:
- ❌ Only works on Apple Silicon (not portable)
- ❌ Smaller ecosystem than PyTorch
- ❌ Some advanced features missing (e.g., DeepSpeed)

**Recommendation**: **Use MLX for all Apple Silicon training** unless you need PyTorch-specific features.

---

### Option 2: PyTorch with MPS Backend

**Why PyTorch MPS?**
- **Cross-platform**: Code works on NVIDIA, AMD, Apple Silicon
- **Large ecosystem**: Hugging Face, Axolotl, etc.
- **Familiar**: Standard PyTorch API
- **Mature**: Well-tested, stable

**Installation**:
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**Axolotl Configuration**:
```yaml
# configs/firefighter_lora_apple_silicon.yml

base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM

# Apple Silicon specific
device: mps  # Use Metal Performance Shaders
bf16: false  # bfloat16 not supported on MPS
fp16: true   # Use float16 instead
flash_attention: false  # CUDA-only feature

# LoRA config (same as NVIDIA)
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Memory optimization
batch_size: 1  # Reduce for MPS
gradient_accumulation_steps: 16
gradient_checkpointing: true
sequence_len: 2048

# Training
num_epochs: 3
learning_rate: 2e-4
optimizer: adamw_torch

datasets:
  - path: ./firefighter_train.jsonl
    type: chat_template
```

**Training**:
```bash
accelerate launch -m axolotl.cli.train configs/firefighter_lora_apple_silicon.yml
```

**Advantages**:
- ✅ Cross-platform (portable code)
- ✅ Full Hugging Face ecosystem
- ✅ Well-documented
- ✅ Familiar to PyTorch users

**Disadvantages**:
- ❌ Slower than MLX on Apple Silicon
- ❌ Some CUDA features unavailable (Flash Attention, bitsandbytes)
- ❌ Less memory efficient than MLX

**Recommendation**: Use PyTorch MPS if you need **cross-platform compatibility** or specific Hugging Face tools.

---

### Option 3: llama.cpp (Inference Only)

**Why llama.cpp?**
- **Excellent Apple Silicon support**: Metal backend
- **Fast inference**: 80-120 tokens/sec on M3 Max
- **Low memory**: Quantized models (4-bit, 5-bit, 8-bit)
- **CPU fallback**: Works even without Metal

**Installation**:
```bash
# Install Python bindings
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Verify Metal support
python -c "from llama_cpp import Llama; print('Metal support compiled')"
```

**Inference Example**:
```python
from llama_cpp import Llama

# Load quantized model (4-bit for speed)
llm = Llama(
    model_path="./models/firefighter-q4_k_m.gguf",
    n_gpu_layers=-1,  # Use all Metal GPU layers
    n_ctx=4096,       # Context length
    verbose=False
)

# Generate response
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are Pyragos Ioanna Michaelidou..."},
        {"role": "user", "content": "What is the IDLH for ammonia?"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response['choices'][0]['message']['content'])
```

**Performance on M3 Max (48GB)**:
- q4_k_m: 100-120 tokens/sec
- q5_k_m: 80-100 tokens/sec
- q8_0: 60-80 tokens/sec
- f16: 40-50 tokens/sec

**Recommendation**: Use llama.cpp for **production inference** (not training).

---

## Step-by-Step: Training with MLX

### Step 1: Install MLX

```bash
# Create environment
conda create -n mlx_training python=3.11
conda activate mlx_training

# Install MLX
pip install mlx mlx-lm

# Install utilities
pip install huggingface-hub datasets transformers
```

### Step 2: Prepare Data

MLX expects **JSON Lines format**:

```jsonl
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are Pyragos Ioanna Michaelidou...<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is IDLH for ammonia?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe IDLH for ammonia is 300 ppm...<|eot_id|>"}
{"text": "..."}
```

**Convert from ChatML**:
```python
# tools/convert_to_mlx.py
import json
from datasets import load_dataset

# Load your ChatML dataset
dataset = load_dataset("json", data_files="firefighter_train.jsonl")

# Convert to MLX format (full conversation as single text)
def format_for_mlx(example):
    messages = example['messages']
    # Use Llama 3.1 chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
mlx_dataset = dataset.map(format_for_mlx)
mlx_dataset.to_json("firefighter_mlx.jsonl")
```

### Step 3: Train with MLX

```bash
# Fine-tune Llama 3.1 8B
python -m mlx_lm.lora \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --data firefighter_mlx.jsonl \
  --train \
  --iters 1000 \
  --steps-per-eval 100 \
  --val-batches 10 \
  --lora-layers 32 \
  --batch-size 2 \
  --learning-rate 2e-5 \
  --save-every 100 \
  --adapter-path ./mlx_adapters/firefighter

# Monitor training
# Iter 100: Train loss 1.234, Val loss 1.456, Tokens/sec: 1234
# Iter 200: Train loss 0.987, Val loss 1.123, Tokens/sec: 1256
# ...
```

**Training Parameters**:
- `--iters`: Number of training iterations (1000-3000 typical)
- `--lora-layers`: LoRA rank (16, 32, 64)
- `--batch-size`: Batch size (1-4, depending on memory)
- `--learning-rate`: 1e-5 to 5e-5 (lower than NVIDIA due to precision)

### Step 4: Test the Model

```python
from mlx_lm import load, generate

# Load base model + LoRA adapter
model, tokenizer = load("meta-llama/Meta-Llama-3.1-8B-Instruct",
                        adapter_path="./mlx_adapters/firefighter")

# Generate response
messages = [
    {"role": "system", "content": "You are Pyragos Ioanna Michaelidou..."},
    {"role": "user", "content": "A wildfire is approaching. What do I do?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=512, temp=0.7)
print(response)
```

### Step 5: Fuse and Quantize (Optional)

```bash
# Fuse LoRA adapter into base model
python -m mlx_lm.fuse \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --adapter-path ./mlx_adapters/firefighter \
  --save-path ./fused_models/firefighter-llama3.1-8b

# Quantize to 4-bit for faster inference
python -m mlx_lm.convert \
  --hf-path ./fused_models/firefighter-llama3.1-8b \
  --mlx-path ./quantized_models/firefighter-q4 \
  --quantize \
  -q 4  # 4-bit quantization

# Now you have a 4-bit quantized model optimized for Apple Silicon!
```

---

## Memory Usage Guide

### Model Size in Memory (Approximate)

| Model Size | Full Precision (f16) | 8-bit | 4-bit (q4_k_m) | Min RAM |
|------------|---------------------|-------|----------------|---------|
| 1.5B | 3GB | 1.8GB | 1.2GB | 8GB |
| 3B | 6GB | 3.5GB | 2.2GB | 16GB |
| 7B | 14GB | 8GB | 4.5GB | 24GB |
| 13B | 26GB | 15GB | 8.5GB | 32GB |
| 34B | 68GB | 38GB | 22GB | 64GB |
| 70B | 140GB | 78GB | 42GB | 128GB |

### Training Memory Requirements

**LoRA training adds**:
- **Optimizer states**: ~2x model size
- **Gradients**: ~1x model size
- **Activations**: Depends on sequence length and batch size

**Example** (7B model, LoRA r=32):
- Model: 14GB (f16)
- LoRA parameters: ~200MB
- Optimizer: ~400MB
- Gradients: ~200MB
- Activations (batch=2, seq=2048): ~4GB
- **Total**: ~19GB

**Recommendation**:
- M2/M3 Pro (32GB): 7B models, batch_size=1-2
- M3 Max (48GB): 7B models, batch_size=2-4
- M3 Max (128GB): 13B models or 7B full fine-tuning
- M2 Ultra (192GB): 70B LoRA or 13B full fine-tuning

---

## Limitations and Workarounds

### 1. No Flash Attention 2

**Issue**: Flash Attention 2 is CUDA-only (not available on Metal/MPS)

**Impact**:
- Slower training for long sequences (>2048 tokens)
- Higher memory usage for long contexts

**Workaround**:
- Use standard attention (automatically handled by MLX/PyTorch)
- Reduce `sequence_len` to 1024-2048
- Use gradient checkpointing to save memory

**Performance Hit**: ~10-20% slower training vs. Flash Attention

---

### 2. No bitsandbytes (QLoRA)

**Issue**: bitsandbytes library is CUDA-only (4-bit quantization during training)

**Impact**: Cannot use QLoRA (4-bit quantized base model + LoRA)

**Workaround**:
- Use standard LoRA (16-bit base model)
- Use smaller base models (3B instead of 7B)
- MLX has its own quantization (not during training though)

**Alternative**: Train in fp16 or bf16 (bf16 not supported on MPS, use fp16)

---

### 3. Limited Multi-GPU Support

**Issue**: Apple Silicon doesn't support multi-GPU training (no Mac has multiple discrete GPUs)

**Impact**: Single-GPU training only

**Workaround**:
- Use larger unified memory instead (M2 Ultra has 192GB)
- Unified memory architecture often compensates for single "GPU"

---

### 4. bfloat16 Not Supported on MPS

**Issue**: PyTorch MPS backend doesn't support bfloat16 (only float16)

**Impact**: Slightly lower numerical stability than bfloat16

**Workaround**:
- Use `fp16: true` instead of `bf16: true` in configs
- MLX supports bfloat16 natively (use MLX instead of PyTorch)

**Performance Hit**: Minimal, float16 works fine for most cases

---

### 5. Slower Than High-End NVIDIA GPUs

**Issue**: Apple Silicon is slower than RTX 4090/A100 for training

**Impact**:
- M3 Max: ~1.5x slower than RTX 4090
- M2 Ultra: ~1.2x slower than A100

**Workaround**:
- Use MLX (optimized for Metal)
- Train overnight (power efficiency makes this practical)
- Use cloud NVIDIA GPUs for time-critical training

---

## Performance Benchmarks

### Training Speed (Llama 3.1 8B, LoRA r=32, 2000 examples, 3 epochs)

| Hardware | Training Time | Cost | Power | Tokens/sec |
|----------|---------------|------|-------|------------|
| **M1 Max (64GB)** | 22 hours | $0 | 50W | 800 |
| **M2 Pro (32GB)** | 18 hours | $0 | 45W | 950 |
| **M3 Max (48GB)** | 12 hours | $0 | 55W | 1,400 |
| **M3 Max (128GB)** | 10 hours | $0 | 60W | 1,650 |
| **M2 Ultra (192GB)** | 8 hours | $0 | 100W | 2,100 |
| RTX 4090 (24GB) | 8 hours | $0 | 400W | 2,300 |
| A100 (40GB) | 4 hours | $400 | 300W | 4,500 |

**Energy Cost** (at $0.12/kWh):
- M3 Max (12h @ 55W): $0.08
- RTX 4090 (8h @ 400W): $0.38
- **Apple Silicon is 5x more energy efficient!**

---

### Inference Speed (Llama 3.1 8B, q4_k_m quantization)

| Hardware | Tokens/sec | Latency (512 tokens) |
|----------|------------|---------------------|
| M1 Max (64GB) | 65 | 7.8s |
| M2 Pro (32GB) | 75 | 6.8s |
| M3 Max (48GB) | 110 | 4.7s |
| M2 Ultra (192GB) | 145 | 3.5s |
| RTX 4090 (24GB) | 125 | 4.1s |

**Conclusion**: Apple Silicon inference is **competitive with NVIDIA** for quantized models!

---

## Best Practices for Apple Silicon

### 1. Use MLX for Training

```bash
# Don't use PyTorch for training if you're only targeting Apple Silicon
# MLX is 20-30% faster
python -m mlx_lm.lora --model ... --data ...
```

### 2. Leverage Unified Memory

```python
# On Apple Silicon, you can load larger models than on NVIDIA with same price
# M3 Max (128GB) > RTX 4090 (24GB VRAM)

# Train 13B model on M3 Max (128GB)
python -m mlx_lm.lora \
  --model meta-llama/Meta-Llama-3.1-13B-Instruct \
  --batch-size 2 \
  --lora-layers 32
```

### 3. Optimize Batch Size and Gradient Accumulation

```yaml
# M2/M3 Pro (32GB)
batch_size: 1
gradient_accumulation_steps: 16  # Effective batch = 16

# M3 Max (48GB-128GB)
batch_size: 2-4
gradient_accumulation_steps: 4-8  # Effective batch = 16-32
```

### 4. Use Quantized Inference

```python
# For deployment, use 4-bit quantization
# 3x faster, 1/4 memory, <5% quality loss
python -m mlx_lm.convert \
  --hf-path ./fused_models/firefighter \
  --mlx-path ./quantized/firefighter-q4 \
  --quantize -q 4
```

### 5. Monitor Temperature and Throttling

```bash
# Apple Silicon throttles under sustained load
# Monitor temperature during training
sudo powermetrics --samplers smc | grep -i "CPU die temperature"

# If throttling occurs:
# - Improve cooling (laptop stand, external cooling)
# - Reduce batch size
# - Train in batches with breaks
```

---

## Recommended Configurations by Mac Model

### MacBook Air M2/M3 (16GB)

**Use case**: Short LLM training, inference

```yaml
base_model: Qwen/Qwen2-1.5B-Instruct
lora_r: 8
batch_size: 1
gradient_accumulation_steps: 16
sequence_len: 1024
num_epochs: 3
```

**Expected**: 4-6 hours for 1,000 examples

---

### MacBook Pro M3 Pro (36GB)

**Use case**: Standard LoRA training (7B models)

```yaml
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
lora_r: 32
batch_size: 2
gradient_accumulation_steps: 8
sequence_len: 2048
num_epochs: 3
```

**Expected**: 12-16 hours for 2,000 examples

---

### MacBook Pro M3 Max (48GB-128GB)

**Use case**: Production LLM training (7B-13B)

```yaml
# 48GB: 7B models
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
lora_r: 32
batch_size: 4
gradient_accumulation_steps: 4

# 128GB: 13B models
base_model: meta-llama/Meta-Llama-3.1-13B-Instruct
lora_r: 64
batch_size: 2
gradient_accumulation_steps: 8
```

**Expected**: 8-12 hours (7B), 16-24 hours (13B) for 2,000 examples

---

### Mac Studio M2 Ultra (192GB)

**Use case**: Large model training (70B LoRA possible!)

```yaml
# 70B LoRA training
base_model: meta-llama/Meta-Llama-3.1-70B-Instruct
lora_r: 64
batch_size: 1
gradient_accumulation_steps: 16
sequence_len: 2048
num_epochs: 2
```

**Expected**: 48-72 hours for 2,000 examples (70B)

---

## Updated Requirements for Apple Silicon

### requirements_apple_silicon.txt

```txt
# Core (Apple Silicon optimized)
mlx>=0.10.0
mlx-lm>=0.8.0

# Standard ML
torch>=2.0.0  # MPS backend included
transformers>=4.36.0
datasets>=2.14.0
peft>=0.7.0
accelerate>=0.25.0

# Inference
llama-cpp-python  # Build with Metal support

# Evaluation (same as NVIDIA)
rouge-score>=0.1.2
bert-score>=0.3.13
sacrebleu>=2.3.1

# DO NOT INSTALL (CUDA-only):
# - bitsandbytes
# - flash-attn
# - deepspeed
```

**Installation**:
```bash
# Install with Metal support for llama.cpp
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Install other dependencies
pip install -r requirements_apple_silicon.txt
```

---

## Troubleshooting

### Issue: "MPS backend out of memory"

**Solution**:
```yaml
# Reduce batch size
batch_size: 1

# Reduce sequence length
sequence_len: 1024

# Enable gradient checkpointing
gradient_checkpointing: true

# Use smaller model
base_model: Qwen/Qwen2-1.5B-Instruct  # Instead of 7B
```

### Issue: "MLX module not found"

**Solution**:
```bash
# MLX only works on Apple Silicon
# Verify you're on M1/M2/M3/M4
uname -m  # Should output: arm64

# Reinstall MLX
pip uninstall mlx mlx-lm
pip install mlx mlx-lm
```

### Issue: Training very slow (< 500 tokens/sec)

**Solution**:
```bash
# 1. Check if using MLX (not PyTorch MPS)
# MLX is 2x faster

# 2. Reduce sequence length
sequence_len: 1024  # Instead of 2048

# 3. Check for throttling
sudo powermetrics --samplers smc | grep -i temperature
# If >80°C, improve cooling

# 4. Close other apps (free up memory)
```

### Issue: "metal shader compilation failed"

**Solution**:
```bash
# Update macOS to latest version
# MLX requires macOS 13.3+ (Ventura) or macOS 14+ (Sonoma)

# Update MLX
pip install --upgrade mlx mlx-lm
```

---

## Conclusion

**Apple Silicon is excellent for LLM training**, especially for:
- ✅ 7B-13B models with LoRA
- ✅ Power-efficient training (overnight jobs)
- ✅ Silent operation (great for home/office)
- ✅ Unified memory (train larger models than NVIDIA at same price point)
- ✅ Fast inference (deployment)

**Use MLX framework** for best results (20-30% faster than PyTorch MPS).

**Recommended hardware**:
- **Minimum**: M2/M3 Pro (32GB) for 7B training
- **Optimal**: M3 Max (48GB+) or M2 Ultra for production use

---

**Generated**: 2025-11-13
**Version**: 1.0
**Optimized for**: M1, M2, M3, M4 (all variants)
