# LLM Training for Domain-Specific Emergency Response Experts

## Overview

This guide provides a **zero-to-hero methodology** for training local Large Language Models (LLMs) with domain-specific expertise from emergency response professionals such as firefighters, police officers, paramedics, and other first responders.

By the end of this guide, you will have:
- A trained domain-specific LLM with expert knowledge
- OR a Short LLM (sLLM) optimized for focused use cases
- An evaluation framework to measure domain expertise quality
- Deployment-ready inference setup

---

## Table of Contents

1. [Introduction](#introduction)
2. [Zero-to-Hero Roadmap](#zero-to-hero-roadmap)
3. [Prerequisites](#prerequisites)
4. [Training Approaches](#training-approaches)
5. [Folder Structure](#folder-structure)
6. [Quick Start](#quick-start)
7. [Success Metrics](#success-metrics)

---

## Introduction

### Why Domain-Specific LLMs?

General-purpose LLMs lack the **specialized knowledge**, **operational procedures**, and **decision-making patterns** of expert emergency responders. Training domain-specific models provides:

- **Accurate domain terminology** (e.g., firefighting tactics, police procedures)
- **Contextual decision-making** (e.g., triage protocols, incident command)
- **Regulatory compliance** (e.g., Greek emergency response laws, EU directives)
- **Cultural context** (e.g., Greek crisis management structure)
- **Reduced hallucinations** on technical topics

### Use Cases for Greek Emergency Response

This methodology is designed for the **Greek Emergency Response Multi-Agent System** with focus on:

1. **Pyragos (Fire Commander)** - Wildfire tactics, structure fire operations, HAZMAT
2. **Taxiarchos (Police Commander)** - Crowd control, evacuation coordination, security
3. **EKAB Physician** - Medical triage, mass casualty protocols, toxicology
4. **Coast Guard Officer** - Maritime rescue, coastal evacuation
5. **Civil Protection Coordinator** - Resource allocation, multi-agency coordination

---

## Zero-to-Hero Roadmap

### Phase 1: Foundation (Week 1-2)
- **Goal**: Understand LLM training basics and set up infrastructure
- **Tasks**:
  - Set up training environment (GPU, storage, dependencies)
  - Choose base model (Llama 3, Mistral, Qwen, etc.)
  - Understand fine-tuning vs. pre-training vs. RAG
- **Output**: Working development environment

### Phase 2: Data Collection (Week 3-6)
- **Goal**: Gather high-quality domain-specific training data
- **Tasks**:
  - Interview domain experts (firefighters, police, medics)
  - Collect standard operating procedures (SOPs)
  - Gather incident reports, case studies, training materials
  - Create question-answer pairs for supervised fine-tuning
- **Output**: 1,000-10,000 training examples (depending on approach)

### Phase 3: Data Preparation (Week 7-8)
- **Goal**: Format and validate training data
- **Tasks**:
  - Convert data to instruction-tuning format (ChatML, Alpaca, etc.)
  - Split into train/validation/test sets (80/10/10)
  - Quality control and deduplication
  - Anonymize sensitive information
- **Output**: Clean, formatted datasets ready for training

### Phase 4: Model Training (Week 9-12)
- **Goal**: Fine-tune base model on domain data
- **Tasks**:
  - Choose training method (LoRA, QLoRA, full fine-tuning)
  - Configure hyperparameters (learning rate, batch size, epochs)
  - Run training with monitoring (loss curves, perplexity)
  - Evaluate checkpoints on validation set
- **Output**: Trained domain-specific model checkpoint

### Phase 5: Evaluation (Week 13-14)
- **Goal**: Validate domain expertise quality
- **Tasks**:
  - Run domain-specific benchmark tests
  - Expert human evaluation (Turing test with real firefighters/police)
  - Compare against base model and GPT-4
  - Measure factual accuracy, safety, bias
- **Output**: Evaluation report with metrics

### Phase 6: Deployment (Week 15-16)
- **Goal**: Deploy model for production use
- **Tasks**:
  - Optimize inference (quantization, GGUF conversion)
  - Set up LM Studio / Ollama / vLLM server
  - Integrate with crisis management system
  - Create API endpoints for multi-agent system
- **Output**: Production-ready deployment

---

## Prerequisites

### Hardware Requirements

**Minimum (for LoRA fine-tuning 7B model):**
- GPU: NVIDIA RTX 3090 / 4090 (24GB VRAM)
- RAM: 32GB system RAM
- Storage: 500GB SSD

**Recommended (for full fine-tuning 7B-13B models):**
- GPU: NVIDIA A100 (40GB/80GB) or H100
- RAM: 64GB+ system RAM
- Storage: 1TB+ NVMe SSD

**Cloud Alternatives:**
- Lambda Labs (A100 rentals)
- RunPod (H100 rentals)
- Google Colab Pro+ (A100 access)
- AWS SageMaker

### Software Requirements

- **Python**: 3.10+
- **CUDA**: 11.8+ or 12.1+
- **PyTorch**: 2.0+
- **Transformers**: 4.36+
- **Training frameworks**: Axolotl, LLaMA Factory, Unsloth, or TRL
- **Inference**: LM Studio, Ollama, vLLM, or llama.cpp

See `tools/requirements.txt` for complete dependencies.

### Knowledge Requirements

- **Basic Python programming**
- **Understanding of command-line tools**
- **Familiarity with machine learning concepts** (optional but helpful)
- **Domain expertise access** (firefighters, police officers, etc.)

---

## Training Approaches

### 1. Full Fine-Tuning
**Best for**: Maximum performance, large datasets (10K+ examples)

**Pros:**
- Highest quality domain adaptation
- Full model parameter updates
- Best for complex reasoning tasks

**Cons:**
- Requires high-end GPU (A100/H100)
- Expensive compute costs
- Risk of catastrophic forgetting

**Estimated Time**: 3-7 days on A100
**Estimated Cost**: $500-2,000 (cloud GPU rental)

---

### 2. LoRA (Low-Rank Adaptation)
**Best for**: Most practitioners, moderate datasets (1K-10K examples)

**Pros:**
- Efficient training (RTX 4090 sufficient)
- Fast iteration cycles
- Preserves base model knowledge
- Small adapter files (100-500MB)

**Cons:**
- Slightly lower performance than full fine-tuning
- Limited capacity for new knowledge

**Estimated Time**: 6-24 hours on RTX 4090
**Estimated Cost**: $50-200 (cloud GPU rental) or free (local GPU)

**Recommended for Greek Emergency Response System** ✅

---

### 3. QLoRA (Quantized LoRA)
**Best for**: Limited GPU memory (16GB VRAM)

**Pros:**
- Trains 7B models on consumer GPUs (RTX 3090)
- Very memory efficient (4-bit quantization)
- Similar performance to LoRA

**Cons:**
- Slower training than LoRA
- Requires careful hyperparameter tuning

**Estimated Time**: 12-48 hours on RTX 3090
**Estimated Cost**: $100-300 or free (local GPU)

---

### 4. Short LLM (sLLM) Training
**Best for**: Focused tasks (e.g., triage only, evacuation only)

**Approach:**
- Start with smaller base model (1B-3B params): Phi-3, Qwen2-1.5B, StableLM-2
- Fine-tune on narrow domain (e.g., only fire tactics)
- Optimize for low-latency inference

**Pros:**
- Fast inference (<100ms)
- Runs on CPU or small GPU
- Highly focused expertise
- Easy deployment

**Cons:**
- Limited general knowledge
- Poor performance outside narrow domain
- May require multiple sLLMs for different tasks

**Estimated Time**: 2-6 hours on RTX 4090
**Estimated Cost**: $20-50 or free (local GPU)

**Use case**: Deploy separate sLLMs for each Greek expert agent (Pyragos, Taxiarchos, etc.)

---

### 5. RAG (Retrieval-Augmented Generation)
**Best for**: When you have documents but limited training budget

**Approach:**
- Keep base model unchanged
- Build vector database of domain documents (SOPs, manuals, regulations)
- Retrieve relevant context at inference time

**Pros:**
- No training required
- Easy to update knowledge (just add documents)
- Works with any LLM (GPT-4, Claude, local models)

**Cons:**
- Doesn't capture expert reasoning patterns
- Slower inference (retrieval overhead)
- Quality depends on document coverage

**Not a training method, but a valid alternative** ℹ️

---

## Folder Structure

```
LLM Training/
├── README.md                          # This file
├── data_collection/
│   ├── README.md                      # Data collection methodology
│   ├── expert_interview_guide.md     # Interview templates
│   ├── sop_extraction_guide.md       # How to extract SOPs
│   └── dataset_templates/
│       ├── firefighter_qa.jsonl      # Example Q&A pairs
│       ├── police_qa.jsonl
│       └── medical_qa.jsonl
├── fine_tuning/
│   ├── README.md                      # Fine-tuning guide
│   ├── lora_training_guide.md        # LoRA training walkthrough
│   ├── full_finetuning_guide.md      # Full fine-tuning walkthrough
│   ├── slm_training_guide.md         # Short LLM training
│   └── configs/
│       ├── axolotl_lora_7b.yml       # Axolotl config for 7B LoRA
│       ├── llama_factory_config.yaml # LLaMA Factory config
│       └── unsloth_config.py         # Unsloth training script
├── evaluation/
│   ├── README.md                      # Evaluation methodology
│   ├── domain_benchmarks.md          # Domain-specific tests
│   ├── human_evaluation_protocol.md  # Expert evaluation guide
│   └── scripts/
│       ├── run_benchmark.py           # Automated benchmark runner
│       └── calculate_metrics.py       # Metric computation
├── examples/
│   ├── firefighter_example/
│   │   ├── README.md                  # Firefighter LLM training example
│   │   ├── dataset_sample.jsonl       # Sample training data (50 examples)
│   │   ├── training_log.txt           # Example training output
│   │   └── evaluation_results.json    # Benchmark results
│   ├── police_example/
│   │   ├── README.md                  # Police LLM training example
│   │   ├── dataset_sample.jsonl
│   │   ├── training_log.txt
│   │   └── evaluation_results.json
│   └── ekab_medical_example/
│       ├── README.md                  # EKAB medical LLM example
│       ├── dataset_sample.jsonl
│       ├── training_log.txt
│       └── evaluation_results.json
├── tools/
│   ├── requirements.txt               # Python dependencies
│   ├── setup_environment.sh           # Environment setup script
│   ├── data_formatter.py              # Convert data to training format
│   ├── model_quantizer.py             # Quantize trained models (GGUF)
│   └── deployment_server.py           # Inference API server
└── DEPLOYMENT.md                      # Deployment and inference guide
```

---

## Quick Start

### Option 1: LoRA Fine-Tuning (Recommended)

```bash
# 1. Set up environment
cd "LLM Training/tools"
bash setup_environment.sh

# 2. Prepare your data (see data_collection/README.md)
python data_formatter.py \
  --input raw_expert_data/ \
  --output formatted_data/ \
  --format chatml

# 3. Train with Axolotl (see fine_tuning/lora_training_guide.md)
cd ../fine_tuning
axolotl train configs/axolotl_lora_7b.yml

# 4. Evaluate (see evaluation/README.md)
cd ../evaluation
python scripts/run_benchmark.py \
  --model ../fine_tuning/outputs/lora-out \
  --benchmark firefighter_tactics

# 5. Deploy (see DEPLOYMENT.md)
cd ../tools
python deployment_server.py \
  --model ../fine_tuning/outputs/lora-out \
  --port 8000
```

### Option 2: Short LLM (sLLM) for Focused Task

```bash
# Train a 1.5B model for fire tactics only
cd "LLM Training/fine_tuning"
python train_slm.py \
  --base_model Qwen/Qwen2-1.5B-Instruct \
  --data ../data_collection/dataset_templates/firefighter_qa.jsonl \
  --output fire_tactics_slm \
  --epochs 3

# Quantize for CPU inference
cd ../tools
python model_quantizer.py \
  --model ../fine_tuning/fire_tactics_slm \
  --output fire_tactics_q4.gguf \
  --quantization q4_k_m
```

---

## Success Metrics

### Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Domain Accuracy** | >85% | Benchmark question correctness |
| **Factual Consistency** | >90% | Fact-checking against SOPs |
| **Response Time** | <2s | Inference latency (7B model) |
| **Training Loss** | <0.5 | Final validation loss |
| **Perplexity** | <10 | Domain-specific test set |

### Qualitative Metrics

- **Expert Validation**: 3+ domain experts rate responses 4/5 or higher
- **Turing Test**: Can experts distinguish model from human expert?
- **Safety**: Zero critical errors in high-stakes scenarios (e.g., wrong HAZMAT protocol)
- **Bias**: Fair treatment across demographics, regions, situations

### Comparison Baseline

Compare your trained model against:
1. **Base model** (before fine-tuning)
2. **GPT-4** (commercial baseline)
3. **Human expert** (gold standard)

**Goal**: Your model should outperform base model significantly and approach GPT-4 performance on domain tasks.

---

## Next Steps

1. **Start with data collection**: Read `data_collection/README.md`
2. **Review examples**: Check `examples/firefighter_example/` for end-to-end walkthrough
3. **Set up tools**: Run `tools/setup_environment.sh`
4. **Choose training approach**: LoRA for most users, sLLM for focused tasks

---

## Additional Resources

### Training Frameworks
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl (Recommended)
- **LLaMA Factory**: https://github.com/hiyouga/LLaMA-Factory
- **Unsloth**: https://github.com/unslothai/unsloth (2x faster LoRA)
- **TRL (Transformers Reinforcement Learning)**: https://github.com/huggingface/trl

### Base Models for Greek Language
- **Llama 3.1 8B Instruct**: Best general purpose
- **Mistral 7B v0.3**: Strong reasoning
- **Qwen2-7B-Instruct**: Multilingual (includes Greek)
- **mGPT-13B**: Multilingual (60+ languages)
- **GreekBERT**: Greek-specific (encoder only, not for generation)

### Datasets
- **OpenHermes 2.5**: General instruction-following
- **WizardLM**: Complex reasoning tasks
- **Orca 2**: Step-by-step explanations
- **Your domain data**: 80% of training should be domain-specific

---

## Support

For questions about this methodology:
1. Review the detailed guides in each subfolder
2. Check the examples for concrete implementations
3. Consult the evaluation methodology for quality assurance

---

**Generated**: 2025-11-13
**System**: Crisis Management Multi-Agent System (Greek Emergency Response Edition)
**Version**: 1.0
**Authors**: Crisis MAS POC Team
