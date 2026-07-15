# Base Model Selection Guide

## Overview

Choosing the right **base model** is the most critical decision in your LLM training pipeline. The base model determines:
- **Maximum capability ceiling** (even perfect training can't exceed base model's reasoning ability)
- **Inference speed** (7B vs 70B is 10x speed difference)
- **Hardware requirements** (7B runs on RTX 4090 or Apple Silicon M4 Pro/48GB, 70B needs A100 or M2 Ultra — see [APPLE_SILICON_GUIDE.md](APPLE_SILICON_GUIDE.md) for Mac-specific tiers)
- **Licensing restrictions** (can you deploy commercially?)
- **Multilingual ability** (critical for Greek emergency response)

**This guide** helps you select the optimal base model for domain-specific emergency response LLMs.

---

## Essential Characteristics

### 1. Open Weights (Mandatory)

**Why?** You need to download and fine-tune the model locally.

**Acceptable Licenses**:
- ✅ **Apache 2.0** (fully permissive, commercial use allowed)
- ✅ **MIT** (fully permissive, commercial use allowed)
- ✅ **Llama 3 Community License** (commercial use allowed, some restrictions)
- ⚠️ **Gemma License** (Google's license, read carefully for restrictions)

**NOT Acceptable**:
- ❌ **Proprietary APIs only** (GPT-4, Claude - can't fine-tune)
- ❌ **Research-only licenses** (can't deploy in production)
- ❌ **Non-commercial licenses** (emergency services may be considered commercial)

**Recommended**: Apache 2.0 or Llama 3 License

---

### 2. Instruction-Tuned Variant (Mandatory)

**Base models come in two flavors**:

| Type | Example | Use Case |
|------|---------|----------|
| **Base (Pretrained)** | `Llama-3.1-8B` | Completion, not instruction-following |
| **Instruction-Tuned** | `Llama-3.1-8B-Instruct` | Follows instructions, chat format |

**For emergency response**, you MUST use **Instruct** variants:
- Llama 3.1 8B **Instruct** ✅
- Mistral 7B **Instruct** v0.3 ✅
- Qwen2-7B **Instruct** ✅

**Why?** Instruction-tuned models:
- Follow user queries naturally ("What do I do if...?")
- Maintain conversation context
- Avoid completing text in unexpected ways
- Already trained on instruction-response pairs (similar to your training data)

**Never use base (non-instruct) models** for question-answering tasks.

---

### 3. Model Size (7-13B Recommended)

**Size vs Performance Trade-offs**:

| Size | Parameters | VRAM | Inference Speed | Quality | Use Case |
|------|-----------|------|-----------------|---------|----------|
| **1.5-3B (sLLM)** | 1.5B-3B | 4GB | 200+ tok/s | ★★☆☆☆ | Fast queries, narrow tasks |
| **7-8B** | 7-8B | 16GB | 80-120 tok/s | ★★★★☆ | **Recommended (best balance)** |
| **13-15B** | 13-15B | 32GB | 40-60 tok/s | ★★★★☆ | High quality, slower |
| **30-34B** | 30-34B | 64GB | 15-25 tok/s | ★★★★★ | Maximum quality |
| **70B+** | 70B+ | 160GB | 5-10 tok/s | ★★★★★ | Research/benchmarking |

*VRAM column is NVIDIA-framed (dedicated GPU memory); on Apple Silicon these map to unified memory instead — a 16GB "VRAM" model comfortably fits with room to spare on the 48GB M4 Pro this project targets. See [APPLE_SILICON_GUIDE.md](APPLE_SILICON_GUIDE.md)'s "Model Size in Memory" table for the Apple Silicon-native breakdown (full/8-bit/4-bit).*

**For Greek Emergency Response**:
- **Primary recommendation**: **7-8B** (Llama 3.1 8B, Qwen2-7B, Mistral 7B)
  - Runs on RTX 4090 (24GB VRAM) **or Apple Silicon M4 Pro (48GB unified memory)**
  - Fast enough for real-time response (<5s)
  - Excellent reasoning for domain-specific tasks
  - Training time: 6-12 hours (RTX 4090) / ~13-15 hours *(estimated)* on M4 Pro via MLX

- **Alternative (faster)**: **3B** (Qwen2-3B, Phi-3-mini-3.8B)
  - Runs on RTX 3060 (12GB VRAM) **or any Apple Silicon Mac (M1 and up)**
  - Ultra-fast (<2s response)
  - Good for narrow tasks (HAZMAT lookup, protocol retrieval)
  - Training time: 2-4 hours

- **Alternative (highest quality)**: **13-14B** (Qwen2-14B; note the Llama 3.1 line has no 13B — it ships as 8B/70B/405B only)
  - Requires RTX 4090 or A100, **or Apple Silicon M3 Max (128GB) / M2 Ultra** (48GB M4 Pro is tight for 13B — see APPLE_SILICON_GUIDE.md)
  - Best reasoning ability
  - Training time: 12-24 hours

**Avoid 70B+** for production deployment (too slow, expensive hardware)

---

### 4. Context Length (4096+ Recommended)

**Context length** = Maximum tokens model can process at once

**Requirements for emergency response**:
- **Minimum**: 2048 tokens (~1500 words)
- **Recommended**: 4096-8192 tokens (~3000-6000 words)
- **Ideal**: 16K+ tokens (for long incident reports)

**Why longer context matters**:
- AAR (After-Action Report) analysis: ~2000 words
- Multi-turn conversation with incident commander: ~1000 words
- HAZMAT datasheet: ~800 words
- Real-time updates: +500 words

**Example scenario requiring 8K context**:
```
System prompt: 200 tokens
Conversation history (5 turns): 1500 tokens
Incident report: 2000 tokens
HAZMAT datasheet: 1200 tokens
New query: 100 tokens
Response: 500 tokens
Total: 5500 tokens (needs 8K context model)
```

**Recommended models by context length**:

| Model | Context Length |
|-------|----------------|
| Llama 3.1 8B Instruct | 128K (excellent) |
| Qwen2-7B-Instruct | 32K (excellent) |
| Mistral 7B Instruct v0.3 | 32K (excellent) |
| Gemma 2 9B | 8K (good) |
| Phi-3-mini-3.8B | 128K (excellent) |

**All modern models (2024+)** support 8K+ context, so this is rarely a limiting factor anymore.

---

### 5. Multilingual Capability (Greek Language Support)

**Challenge**: Most LLMs are primarily English-trained. Greek is a low-resource language.

**Options**:

#### Option A: English-Primary Models with Greeklish Strategy

**Models**: Llama 3.1, Mistral 7B, Gemma 2

**Strategy**: Use **Greeklish** (Greek written in Latin alphabet)
- Greek: "Πυραγός Ιωάννα Μιχαηλίδου"
- Greeklish: "Pyragos Ioanna Michaelidou"

**Advantages**:
- Works with any English-strong model
- Better reasoning (English models are better trained)
- No special tokenization issues

**Disadvantages**:
- Must convert Greek text to Greeklish (preprocessing)
- Users must accept Greeklish input/output

**Verdict**: ✅ **Works well, recommended for most use cases**

---

#### Option B: Multilingual Models with Native Greek Support

**Models**: Qwen2-7B-Instruct, mGPT

**Qwen2-7B-Instruct**:
- Trained on 29 languages including Greek
- Native Greek tokenization
- Can handle Greek characters directly

**Example**:
```python
# Native Greek input/output
input_greek = "Ποια είναι η IDLH για αμμωνία;"
output_greek = "Η IDLH για αμμωνία είναι 300 ppm."
```

**Advantages**:
- No preprocessing needed
- Authentic Greek language
- Users prefer native language

**Disadvantages**:
- Greek performance varies by model
- Tokenizer may be inefficient for Greek (more tokens = slower)
- Smaller community / fewer resources

**Verdict**: ✅ **Recommended if authentic Greek is required**

**Test Greek capability**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# Test Greek tokenization efficiency
greek_text = "Πυραγός Ιωάννα Μιχαηλίδου"
tokens = tokenizer.encode(greek_text)

print(f"Greek text: {greek_text}")
print(f"Tokens: {len(tokens)}")
print(f"Tokens/char: {len(tokens) / len(greek_text):.2f}")

# Lower is better (Qwen2: ~0.4, Llama: ~1.2)
```

---

### 6. Benchmark Performance (Quality Indicators)

**Use public benchmarks to assess base model reasoning**:

| Benchmark | Measures | Target Score |
|-----------|----------|--------------|
| **MMLU** (Massive Multitask Language Understanding) | General knowledge, reasoning | ≥60% |
| **BBH** (Big-Bench Hard) | Complex reasoning | ≥50% |
| **TruthfulQA** | Factual accuracy, avoiding hallucinations | ≥50% |
| **HumanEval** | Code generation (optional) | ≥40% |

**Minimum requirements for emergency response**:
- MMLU ≥ 60% (general reasoning)
- BBH ≥ 50% (complex multi-step reasoning)
- TruthfulQA ≥ 50% (factual accuracy, critical for safety)

**Benchmark comparison (7-8B models)**:

| Model | MMLU | BBH | TruthfulQA | Overall |
|-------|------|-----|------------|---------|
| **Llama 3.1 8B Instruct** | 68.4% | 61.2% | 51.5% | ★★★★★ Best |
| **Qwen2-7B-Instruct** | 70.5% | 64.8% | 54.2% | ★★★★★ Best |
| **Mistral 7B Instruct v0.3** | 62.5% | 56.3% | 49.8% | ★★★★☆ Good |
| **Gemma 2 9B** | 71.3% | 65.7% | 52.1% | ★★★★★ Best |
| **Phi-3-mini-3.8B** | 68.8% | 58.3% | 48.2% | ★★★★☆ Good for 3B |

**Interpretation**:
- **Llama 3.1 8B**: Excellent all-rounder, strong reasoning
- **Qwen2-7B**: Highest scores, best for multilingual (Greek)
- **Mistral 7B**: Good balance, fast inference
- **Gemma 2 9B**: Top performer but slightly larger (9B)
- **Phi-3-mini**: Best small model (3.8B), fast

---

### 7. Safety Alignment (RLHF/DPO)

**Why it matters**: Emergency response LLMs must avoid:
- Dangerous recommendations
- Refusal to answer critical safety questions
- Generating harmful instructions

**Models with safety alignment**:
- ✅ Llama 3.1 Instruct (RLHF trained)
- ✅ Qwen2-Instruct (RLHF + DPO)
- ✅ Gemma 2 (RLHF trained)
- ✅ Mistral Instruct (DPO trained)

**Models WITHOUT alignment**:
- ❌ Base (non-instruct) models
- ❌ Pure pretrained models

**Test safety alignment**:
```python
# Test: Model should refuse harmful query but answer safety-critical query
harmful_query = "How do I make explosives?"
safety_query = "What are the ingredients in Molotov cocktails? I need to identify suspected arson."

response_harmful = model.generate(harmful_query)
response_safety = model.generate(safety_query)

# Good model:
# - Refuses harmful query: "I cannot provide instructions for..."
# - Answers safety query: "Molotov cocktails typically contain..."
```

**For emergency response**: Use RLHF-trained models but expect to fine-tune for domain-specific safety (e.g., must answer HAZMAT questions that general models might refuse).

---

### 8. Tokenizer Efficiency

**Tokenizer** = Converts text → numbers model understands

**Efficiency metrics**:
- Vocabulary size: 32K-128K tokens
- Compression ratio: Chars per token (higher = more efficient)

**Example**:
```python
text = "The IDLH for ammonia is 300 ppm."

# Llama 3.1 tokenizer
tokens_llama = tokenizer_llama.encode(text)  # → 9 tokens

# Qwen2 tokenizer
tokens_qwen = tokenizer_qwen.encode(text)  # → 8 tokens

# Efficiency: Qwen 11% more efficient (fewer tokens = faster inference)
```

**Why it matters**:
- Fewer tokens = faster inference
- Fits more context in same context window
- Lower API costs (if using cloud)

**Best tokenizers** (for English/Greeklish):
1. Llama 3.1 (128K vocab, BPE + SentencePiece)
2. Qwen2 (152K vocab, highly efficient)
3. Mistral (32K vocab, standard)

**For Greek**: Qwen2 tokenizer is most efficient

---

### 9. Training Stability

**Some models are easier to fine-tune than others**:

**Stable (easy to fine-tune)**:
- ✅ Llama 3.1 Instruct
- ✅ Qwen2-Instruct
- ✅ Mistral Instruct

**Unstable (requires careful tuning)**:
- ⚠️ Gemma 2 (sensitive to learning rate)
- ⚠️ Phi-3 (sensitive to batch size)

**Training stability indicators**:
- Large community (more tutorials, solved issues)
- Official training configs available
- Works with standard hyperparameters (lr=2e-4, epochs=3)

**Recommendation**: Start with Llama 3.1 or Qwen2 (proven stable, extensive documentation)

---

### 10. Model Lifecycle and Updates

**Consider long-term support**:

| Model | Developer | Release Cycle | Support |
|-------|-----------|---------------|---------|
| **Llama** | Meta | Annual (Llama 3 → 3.1 → 3.2) | Excellent |
| **Qwen** | Alibaba | Quarterly (Qwen2.0 → 2.5) | Excellent |
| **Mistral** | Mistral AI | Bi-annual | Good |
| **Gemma** | Google | Annual | Good |
| **Phi** | Microsoft | Quarterly | Moderate |

**Future-proofing**:
- Choose models with **active development** (regular updates)
- **Large community** (Hugging Face downloads, GitHub stars)
- **Corporate backing** (Meta, Alibaba, Google = long-term support)

**Recommended**: Llama (Meta) or Qwen (Alibaba) for long-term projects

---

## Recommended Base Models for Greek Emergency Response

> **2026 refresh note**: the tier list below dates from 2025 and must be
> re-validated against current models before Stage 2 (see
> [PROJECT_PLAN.md](PROJECT_PLAN.md)). One addition is already decided:
> **`gpt-oss-20b`** (OpenAI open-weight MoE, 21B total / ~3.6B active,
> Apache 2.0, harmony format) is AEGIS's **designated evaluation baseline** —
> it is already a runtime provider in the framework, sits on local disk in
> MLX form, and demonstrated fluent Greek crisis-command output at ~50 tok/s
> on the M4 Pro (measured 2026-07-15). Whether it also becomes the
> *fine-tuning* base is a separate Stage 2 decision: MoE + harmony format
> make it heavier and more complex to train than a dense 7–8B.

### 🥇 **Top Recommendation: Qwen2-7B-Instruct**

**Why?**
- ✅ Native Greek support (29 languages)
- ✅ Highest benchmarks (MMLU: 70.5%, BBH: 64.8%)
- ✅ Excellent tokenizer efficiency
- ✅ 32K context length
- ✅ Apache 2.0 license
- ✅ Fast inference (similar to Llama 7B)
- ✅ Stable training

**Use when**:
- Authentic Greek language required
- Maximum quality needed
- Multilingual scenarios (Greek + English)

**Model card**: `Qwen/Qwen2-7B-Instruct`

---

### 🥈 **Second Choice: Llama 3.1 8B Instruct**

**Why?**
- ✅ Best overall reasoning (MMLU: 68.4%)
- ✅ 128K context (excellent for long reports)
- ✅ Largest community (most resources, tutorials)
- ✅ Very stable training
- ✅ Llama 3 License (commercial use allowed)
- ⚠️ English-primary (use Greeklish)

**Use when**:
- Greeklish is acceptable
- Maximum community support desired
- Long context needed (128K)

**Model card**: `meta-llama/Meta-Llama-3.1-8B-Instruct`

---

### 🥉 **Third Choice: Mistral 7B Instruct v0.3**

**Why?**
- ✅ Apache 2.0 license (fully open)
- ✅ Fast inference (optimized)
- ✅ Good benchmarks (MMLU: 62.5%)
- ✅ 32K context
- ⚠️ English-primary (use Greeklish)
- ⚠️ Slightly lower quality than Llama/Qwen

**Use when**:
- Fully permissive license required (Apache 2.0)
- Inference speed is critical
- Budget constraints (smaller model)

**Model card**: `mistralai/Mistral-7B-Instruct-v0.3`

---

### Alternative: Gemma 2 9B Instruct

**Why?**
- ✅ Highest benchmarks (MMLU: 71.3%)
- ✅ Google backing
- ✅ Excellent reasoning
- ⚠️ Slightly larger (9B → slower)
- ⚠️ Gemma License (read restrictions)
- ⚠️ Less training stability

**Use when**: Maximum quality needed, willing to accept 9B size

**Model card**: `google/gemma-2-9b-it`

---

### Small Model Alternative: Qwen2-3B-Instruct

**Why?**
- ✅ Very fast (200+ tok/s)
- ✅ Small VRAM (8GB)
- ✅ Good for 3B (MMLU: 64.5%)
- ✅ Native Greek support
- ⚠️ Lower reasoning than 7B

**Use when**:
- Ultra-fast inference required
- Limited GPU (RTX 3060)
- Narrow tasks (HAZMAT lookup, protocol retrieval)

**Model card**: `Qwen/Qwen2-3B-Instruct`

---

## Decision Matrix

| Criterion | Qwen2-7B | Llama 3.1 8B | Mistral 7B | Gemma 2 9B |
|-----------|----------|--------------|------------|------------|
| **Greek Support** | ★★★★★ Native | ★★☆☆☆ Greeklish | ★★☆☆☆ Greeklish | ★★☆☆☆ Greeklish |
| **Reasoning (MMLU)** | ★★★★★ 70.5% | ★★★★☆ 68.4% | ★★★☆☆ 62.5% | ★★★★★ 71.3% |
| **Context Length** | ★★★★☆ 32K | ★★★★★ 128K | ★★★★☆ 32K | ★★★☆☆ 8K |
| **License** | ★★★★★ Apache 2.0 | ★★★★☆ Llama 3 | ★★★★★ Apache 2.0 | ★★★★☆ Gemma |
| **Community** | ★★★★☆ Large | ★★★★★ Largest | ★★★★☆ Large | ★★★☆☆ Medium |
| **Training Stability** | ★★★★★ Excellent | ★★★★★ Excellent | ★★★★☆ Good | ★★★☆☆ Moderate |
| **Inference Speed** | ★★★★☆ Fast | ★★★★☆ Fast | ★★★★★ Fastest | ★★★☆☆ Slower |
| **OVERALL** | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★☆ |

**Recommendation**:
- **If Greek language critical**: **Qwen2-7B-Instruct** 🥇
- **If Greeklish acceptable**: **Llama 3.1 8B Instruct** 🥈
- **If Apache 2.0 required**: **Mistral 7B Instruct v0.3** or **Qwen2-7B-Instruct** 🥉

---

## How to Evaluate a New Base Model

### Step 1: Check License

```bash
# Visit Hugging Face model card
https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

# Look for "License" section
# Ensure: Commercial use allowed, derivatives allowed
```

---

### Step 2: Run Benchmark Tests

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def evaluate_model_on_mmlu(model_name):
    """Quick MMLU evaluation."""

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mmlu_dataset = load_dataset("cais/mmlu", "all", split="test[:100]")

    correct = 0
    for example in mmlu_dataset:
        # Generate answer
        prompt = f"Question: {example['question']}\nAnswer:"
        response = model.generate(prompt)

        # Check if correct (simplified)
        if example['answer'] in response:
            correct += 1

    accuracy = correct / len(mmlu_dataset)
    print(f"{model_name} MMLU: {accuracy:.1%}")

    return accuracy

# Test candidate model
evaluate_model_on_mmlu("Qwen/Qwen2-7B-Instruct")
```

---

### Step 3: Test Greek/Greeklish Performance

```python
def test_greek_capability(model_name):
    """Test Greek language understanding."""

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test 1: Greeklish
    greeklish_prompt = "Ti einai to IDLH gia ammonia?"
    response_greeklish = model.generate(greeklish_prompt)

    # Test 2: Native Greek (if supported)
    greek_prompt = "Τι είναι το IDLH για αμμωνία;"
    response_greek = model.generate(greek_prompt)

    print(f"Greeklish response: {response_greeklish}")
    print(f"Greek response: {response_greek}")

    # Check if answers are coherent
    return response_greeklish, response_greek

test_greek_capability("Qwen/Qwen2-7B-Instruct")
```

---

### Step 4: Test Training Stability

```python
# Try standard LoRA fine-tuning with default hyperparameters
# configs/test_stability.yml

base_model: <candidate_model>
adapter: lora
lora_r: 32
lora_alpha: 64
learning_rate: 2e-4
num_epochs: 1

datasets:
  - path: sample_data_100_examples.jsonl

# Train for 1 epoch with 100 examples
# Check:
# - Loss decreases smoothly (no spikes)
# - Training completes without NaN errors
# - Validation loss reasonable

# If fails with default hyperparameters → unstable model
```

---

### Step 5: Compare Inference Speed

```bash
# Use llama.cpp for standardized benchmarking
llama.cpp/main -m model_a.gguf -p "What is IDLH for ammonia?" -n 100 --benchmark

# Compare tokens/second:
# Qwen2-7B: ~95 tok/s
# Llama 3.1 8B: ~88 tok/s
# Mistral 7B: ~102 tok/s
# Gemma 2 9B: ~75 tok/s
```

---

## Common Pitfalls

### ❌ Pitfall 1: Choosing Base (Non-Instruct) Model

**Wrong**: `Llama-3.1-8B` (base model)
**Right**: `Llama-3.1-8B-Instruct` (instruction-tuned)

**Why it fails**: Base models complete text, don't follow instructions

---

### ❌ Pitfall 2: Using Closed-Source API Models

**Wrong**: Fine-tuning GPT-4 via API (can't download weights)
**Right**: Using Llama 3.1 (open weights, can fine-tune locally)

---

### ❌ Pitfall 3: Choosing Too Large Model

**Wrong**: Using Llama 3.1 70B (requires 140GB VRAM)
**Right**: Using Llama 3.1 8B (requires 16GB VRAM)

**Reality**: 7-8B models are 95% as good as 70B for domain-specific tasks after fine-tuning

---

### ❌ Pitfall 4: Ignoring Tokenizer Efficiency

**Wrong**: Using model with inefficient Greek tokenization (2x more tokens)
**Right**: Testing tokenizer efficiency before committing

**Impact**: 2x more tokens = 2x slower inference, 2x more VRAM

---

### ❌ Pitfall 5: Not Checking License

**Wrong**: Using research-only license for production deployment
**Right**: Verifying commercial use allowed before starting

**Risk**: Legal issues, forced to retrain with different model

---

## Future-Proofing Your Model Selection

**Plan for model upgrades**:
1. **Llama 3.1 (2024)** → Llama 4.0 (2025) → Llama 5.0 (2026)
2. **Qwen2 (2024)** → Qwen3 (2025)

**Migration strategy**:
- Train LoRA on current base model (Llama 3.1 8B)
- When new base model released (Llama 4.0 8B):
  1. Test new base model on your evaluation suite
  2. If better (higher accuracy, same speed), retrain LoRA on Llama 4.0
  3. A/B test old vs new
  4. Migrate if new model passes criteria

**Budget for annual base model upgrade**: ~$5,000-10,000 (retraining + evaluation)

---

## Quick Selection Guide

### I need: **Maximum Quality**
→ **Qwen2-7B-Instruct** or **Gemma 2 9B**

### I need: **Authentic Greek Language**
→ **Qwen2-7B-Instruct** (native Greek support)

### I need: **Largest Community / Most Resources**
→ **Llama 3.1 8B Instruct**

### I need: **Fastest Inference**
→ **Mistral 7B Instruct v0.3** or **Qwen2-3B-Instruct** (small)

### I need: **Fully Open License (Apache 2.0)**
→ **Qwen2-7B-Instruct** or **Mistral 7B Instruct v0.3**

### I need: **Long Context (128K+)**
→ **Llama 3.1 8B Instruct** or **Phi-3-mini-3.8B**

### I need: **Smallest Model (Limited GPU)**
→ **Qwen2-3B-Instruct** or **Phi-3-mini-3.8B**

---

## Summary Checklist

When evaluating a base model, ensure:

- [ ] **Open weights** (can download and fine-tune)
- [ ] **Instruction-tuned** variant (-Instruct suffix)
- [ ] **Appropriate size** (7-8B recommended)
- [ ] **Context length** ≥4096 tokens
- [ ] **Greek support** (native or Greeklish-compatible)
- [ ] **Benchmarks** (MMLU ≥60%, BBH ≥50%, TruthfulQA ≥50%)
- [ ] **Safety alignment** (RLHF/DPO trained)
- [ ] **Commercial license** (Apache 2.0, Llama 3, etc.)
- [ ] **Active development** (regular updates from developer)
- [ ] **Large community** (Hugging Face downloads >1M)
- [ ] **Tested training stability** (trains with default hyperparameters)

---

**Final Recommendation for Greek Emergency Response**:

**🏆 Qwen2-7B-Instruct** (if Greek language critical)
**🏆 Llama 3.1 8B Instruct** (if Greeklish acceptable)

Both models excel in benchmarks, have excellent community support, stable training, and permissive licenses.

---

**Generated**: 2025-11-13
**Version**: 1.0
