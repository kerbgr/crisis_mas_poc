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

This methodology is designed for the **Greek Emergency Response Multi-Agent System** featuring **14 total agents** (13 specialized expert agents + 1 orchestrator):

**Core Expert Agents (3):**
1. **Meteorologist** - Weather forecasting, atmospheric analysis, wildfire weather prediction
2. **Medical Expert (EKAB)** - Pre-hospital emergency medicine, triage protocols
3. **Logistics Coordinator** - Emergency supply chain, resource allocation

**Emergency Response Command Structure (10 additional agents):**
4. **Civil Protection Director** - National emergency coordination, multi-agency command
5. **Environmental Expert** - Environmental impact assessment, ecosystem restoration
6. **PSAP Commander** - Emergency dispatch operations, 112 system coordination
7. **On-Scene Police Commander** - Tactical law enforcement, scene security
8. **Regional Police Commander** - Strategic police coordination, mutual aid
9. **On-Scene Fire Brigade Commander (Pyragos)** - Tactical firefighting, HAZMAT response
10. **Regional Fire Brigade Commander (Taxiarchos)** - Strategic fire operations, wildfire campaigns
11. **Medical Infrastructure Director** - Hospital capacity planning, medical surge operations
12. **On-Scene Coast Guard Commander** - Maritime search and rescue, coastal evacuation
13. **National Coast Guard Director** - Strategic maritime security, inter-regional SAR

**Orchestrator:**
14. **System Coordinator** - Multi-agent decision aggregation using Evidential Reasoning (ER) and Graph Attention Networks (GAT)

For complete agent profiles and architecture details, see [../agents/agent_profiles.json](../agents/agent_profiles.json) and [../agents/AGENT_DEVELOPMENT_GUIDE.md](../agents/AGENT_DEVELOPMENT_GUIDE.md).

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

**Apple Silicon (local, no cloud GPU needed):**
- MacBook Pro **M4 Pro (48GB)** — this project's reference local config for LoRA fine-tuning (7B-8B models), see [APPLE_SILICON_GUIDE.md](APPLE_SILICON_GUIDE.md)
- Minimum: M2/M3 Pro (32GB) for 7B LoRA
- Optimal: M3 Max (48GB+) or M2 Ultra for larger models / full fine-tuning
- Full hardware tiers, benchmarks, and MLX/PyTorch-MPS setup: **[APPLE_SILICON_GUIDE.md](APPLE_SILICON_GUIDE.md)**

### Software Requirements

- **Python**: 3.10+
- **CUDA**: 11.8+ or 12.1+ (NVIDIA only — not needed on Apple Silicon, see [APPLE_SILICON_GUIDE.md](APPLE_SILICON_GUIDE.md))
- **PyTorch**: 2.0+ (includes MPS backend for Apple Silicon)
- **Transformers**: 4.36+
- **Training frameworks**: Axolotl, LLaMA Factory, Unsloth, or TRL (NVIDIA); **MLX** or PyTorch-MPS (Apple Silicon)
- **Inference**: LM Studio, Ollama, vLLM, or llama.cpp (llama.cpp and LM Studio/Ollama both run natively on Apple Silicon via Metal)

See `tools/requirements.txt` for complete dependencies (NVIDIA-oriented) or `APPLE_SILICON_GUIDE.md`'s `requirements_apple_silicon.txt` snippet for the Apple Silicon equivalent.

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
**Apple Silicon**: Not practical at this model size on Mac hardware — use full fine-tuning only on NVIDIA cloud GPUs, then deploy the resulting model locally on Apple Silicon for inference.

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
**Apple Silicon**: ~13-15 hours on M4 Pro (48GB) via MLX, free (local) — this is the approach used in `examples/firefighter_example/`. See [APPLE_SILICON_GUIDE.md](APPLE_SILICON_GUIDE.md).

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
**Apple Silicon**: QLoRA needs `bitsandbytes`, which is CUDA-only — not available on Apple Silicon. Use standard LoRA instead (see above) or a smaller base model; see APPLE_SILICON_GUIDE.md's "No bitsandbytes (QLoRA)" limitation.

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
**Apple Silicon**: Comfortably runs on any Apple Silicon Mac (M1 and up) given the small model size — M4 Pro (48GB) has plenty of headroom for 1B-3B models with fast iteration.

**Use case**: Deploy separate sLLMs for each of the 13 specialized Greek expert agents (see system architecture with 14 total agents including orchestrator)

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
│       ├── meteorologist_qa.jsonl    # Weather/environmental specialist Q&A
│       ├── medical_expert_qa.jsonl   # EKAB emergency medicine Q&A
│       ├── logistics_qa.jsonl        # Supply chain management Q&A
│       ├── fire_onscene_qa.jsonl     # Tactical firefighting Q&A
│       ├── fire_regional_qa.jsonl    # Strategic fire operations Q&A
│       ├── police_onscene_qa.jsonl   # Tactical law enforcement Q&A
│       ├── police_regional_qa.jsonl  # Strategic police coordination Q&A
│       ├── coastguard_onscene_qa.jsonl   # Maritime SAR Q&A
│       ├── coastguard_national_qa.jsonl  # Strategic maritime security Q&A
│       ├── medical_infrastructure_qa.jsonl # Hospital capacity planning Q&A
│       ├── psap_commander_qa.jsonl   # Emergency dispatch Q&A
│       ├── public_safety_qa.jsonl    # Multi-agency coordination Q&A
│       └── environmental_qa.jsonl    # Environmental assessment Q&A
│       # Note: 13 agent-specific templates (orchestrator uses aggregation logic, not Q&A training)
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
│   ├── meteorologist_example/
│   │   ├── README.md                  # Meteorologist LLM training example
│   │   ├── dataset_sample.jsonl       # Sample training data (50 examples)
│   │   ├── training_log.txt           # Example training output
│   │   └── evaluation_results.json    # Benchmark results
│   ├── fire_onscene_example/
│   │   ├── README.md                  # Fire Brigade (tactical) LLM training example
│   │   ├── dataset_sample.jsonl
│   │   ├── training_log.txt
│   │   └── evaluation_results.json
│   ├── medical_expert_example/
│   │   ├── README.md                  # EKAB medical LLM example
│   │   ├── dataset_sample.jsonl
│   │   ├── training_log.txt
│   │   └── evaluation_results.json
│   └── orchestrator_example/
│       ├── README.md                  # Orchestrator training (decision aggregation)
│       ├── aggregation_training.jsonl # Multi-agent consensus examples
│       ├── training_log.txt
│       └── evaluation_results.json
│   # Full examples for all 14 agents available in subdirectories
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

### System Usage Examples

After training your domain-specific LLMs, integrate them into the multi-agent system:

**Default Mode (Backward Compatibility - 3 Core Experts):**
```bash
# Uses meteorologist, logistics, and medical experts
python ../main.py --scenario flood_scenario.json
```

**Automatic Expert Selection (Recommended):**
```bash
# System auto-selects 3-11 experts based on scenario metadata
python ../main.py --scenario flood_scenario.json --expert-selection auto

# With verbose logging to see selection reasoning
python ../main.py --scenario flood_scenario.json --expert-selection auto --verbose
```

**Manual Expert Selection:**
```bash
# Specify exactly which experts to engage
python ../main.py --scenario flood_scenario.json \
  --agents fire_onscene_01 fire_regional_01 agent_meteorologist

# Use all 13 expert agents
python ../main.py --scenario flood_scenario.json --agents all
```

**With Aggregation Method Selection:**
```bash
# Use Graph Attention Network (GAT) instead of Evidential Reasoning (ER)
python ../main.py --scenario flood_scenario.json \
  --expert-selection auto \
  --aggregation GAT \
  --verbose
```

See [../README.md](../README.md) for complete usage documentation.

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

- **Expert Validation**: Each of the 13 specialized domain experts should rate responses ≥4/5 (minimum 11/13 agents meeting threshold for system-wide acceptance)
- **Multi-Agent Consensus**: Orchestrator achieves ≥85% consensus level across expert agents (target: 92% based on academic evaluation results)
- **Turing Test**: Can domain professionals distinguish trained agent from human expert in their specialty?
- **Safety**: Zero critical errors in high-stakes scenarios (e.g., wrong HAZMAT protocol, incorrect triage classification)
- **Bias**: Fair treatment across demographics, regions, crisis types
- **Orchestrator Performance**: Decision aggregation quality measured via Decision Quality Score (DQS) target ≥80%

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
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl (Recommended for multi-agent systems)
- **LLaMA Factory**: https://github.com/hiyouga/LLaMA-Factory (User-friendly GUI)
- **Unsloth**: https://github.com/unslothai/unsloth (2x faster LoRA training)
- **TRL (Transformers Reinforcement Learning)**: https://github.com/huggingface/trl (Advanced RLHF)

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

## Belief Aggregation Methods: ER vs GAT

The orchestrator agent (14th agent) aggregates beliefs from the 13 expert agents using one of two methods:

### Overview

| Aspect | Evidential Reasoning (ER) | Graph Attention Network (GAT) |
|--------|---------------------------|-------------------------------|
| **Type** | Classical (Dempster-Shafer) | Neural Network (Deep Learning) |
| **Complexity** | Simple weighted averaging | Multi-head attention mechanism |
| **Training** | No training required | Pre-trained on crisis scenarios |
| **Weights** | Fixed or confidence-based | Dynamic, context-aware, learned |
| **Interpretability** | Fully transparent | Attention weights provide insight |
| **Processing** | ~38.9s (13 agents) | ~41.2s (13 agents) |

### Performance Comparison (Academic Evaluation Results)

Based on 75 evaluation runs across 3 Greek crisis scenarios with 13 expert agents:

| Metric | ER (Classical) | GAT (Neural) | Improvement |
|--------|----------------|--------------|-------------|
| **Decision Quality Score** | 86.1% | 86.3% | +0.2% (not significant) |
| **Consensus Level** | 68.2% | **71.0%** | **+2.8%** ✓ |
| **Decision Confidence** | 80.4% | **81.9%** | **+1.5%** ✓ |
| **Processing Time** | 38.9s | 41.2s | +2.3s (6% slower) |
| **Convergence Rate** | 94.7% | 96.0% | +1.3% |

**Key Finding**: GAT provides significantly better consensus (+2.8%, p<0.01) and confidence (+1.5%, p<0.05) with minimal computational overhead, while maintaining equivalent decision quality.

### SWOT Analysis

#### Evidential Reasoning (ER)

**Strengths:**
- ✅ Mathematically rigorous (Dempster-Shafer theory, 1976)
- ✅ Fully transparent and auditable
- ✅ No training data required
- ✅ Proven in high-stakes domains
- ✅ Slightly faster processing (~6%)
- ✅ Works with any number of agents (3-13+)
- ✅ Deterministic output (same inputs = same results)

**Weaknesses:**
- ⚠️ Fixed weighting schemes (doesn't adapt to context)
- ⚠️ Can't learn from historical performance patterns
- ⚠️ Lower consensus levels (68.2% vs 71.0%)
- ⚠️ Treats all crisis types identically
- ⚠️ No implicit uncertainty calibration

**Opportunities:**
- 🔹 Combine with dynamic weighting based on reliability tracking
- 🔹 Extend to full Dempster-Shafer combination (more complex)
- 🔹 Add scenario-specific weight adjustments

**Threats:**
- 🔻 May underperform in novel crisis scenarios
- 🔻 Can't capture non-linear expert interactions
- 🔻 Limited ability to smooth extreme outlier opinions

---

#### Graph Attention Network (GAT)

**Strengths:**
- ✅ Context-aware adaptive weighting
- ✅ Learns from historical crisis patterns
- ✅ Higher consensus (+2.8% absolute improvement)
- ✅ Better confidence calibration (+1.5%)
- ✅ Attention weights provide interpretability
- ✅ Smooths extreme opinions more effectively
- ✅ Can capture non-linear expert relationships

**Weaknesses:**
- ⚠️ Requires pre-training on crisis scenarios
- ⚠️ ~6% slower processing (41.2s vs 38.9s)
- ⚠️ More complex implementation
- ⚠️ Potential overfitting to training scenarios
- ⚠️ Requires expertise to tune hyperparameters

**Opportunities:**
- 🔹 Continual learning from new crisis events
- 🔹 Transfer learning to new crisis types
- 🔹 Integration with reliability tracker for meta-learning
- 🔹 Explainable AI visualizations of attention patterns

**Threats:**
- 🔻 May not generalize to completely novel crisis types
- 🔻 Requires computational resources for inference
- 🔻 Black-box perception (despite attention visualization)
- 🔻 Dependency on training data quality

---

### When to Choose Each Method

**Use Evidential Reasoning (ER) when:**
- Full transparency and auditability are critical
- You have limited computational resources
- You need deterministic, repeatable results
- You're deploying in regulated environments requiring explainability
- You have few historical scenarios for GAT training
- Fastest possible processing is required

**Use Graph Attention Network (GAT) when:**
- You have historical crisis data for training
- Consensus quality is paramount
- You can tolerate +6% processing overhead
- You want adaptive, context-aware aggregation
- You need the system to learn from past performance
- You have diverse crisis scenarios requiring different expert weightings

**Recommended Default: GAT** for operational deployments where the +2.8% consensus improvement and +1.5% confidence boost justify the minimal computational overhead. Fall back to ER for regulatory/compliance scenarios requiring full mathematical transparency.

### How to Switch Between Methods

```bash
# Use Evidential Reasoning (classical, default)
python main.py --scenario flood.json --aggregation ER

# Use Graph Attention Network (neural, recommended)
python main.py --scenario flood.json --aggregation GAT

# With verbose mode to see aggregation details
python main.py --scenario flood.json --aggregation GAT --verbose
```

**System Default**: ER (for backward compatibility). Recommend using `--aggregation GAT` for best performance.

### Technical Implementation

- **ER Implementation**: [../decision_framework/evidential_reasoning.py](../decision_framework/evidential_reasoning.py)
- **GAT Implementation**: [../decision_framework/gat_aggregator.py](../decision_framework/gat_aggregator.py)
- **Coordinator Logic**: [../agents/coordinator_agent.py](../agents/coordinator_agent.py) (lines 241-468)
- **Comparative Evaluation**: [../academic_paper.md](../academic_paper.md) (Section 6.2)

---

## Academic Foundation & References

This LLM training methodology is part of a comprehensive research framework. For mathematical foundations, system architecture, and theoretical background, see:

### Core Documentation
- **Academic Paper**: [../academic_paper.md](../academic_paper.md) - Full research paper with literature review
- **System Architecture**: [../README.md](../README.md) - Multi-agent system overview
- **Agent Profiles**: [../agents/agent_profiles.json](../agents/agent_profiles.json) - All 14 agent specifications
- **Evaluation Framework**: [../evaluation/EVALUATION_METHODOLOGY.md](../evaluation/EVALUATION_METHODOLOGY.md) - Metrics and formulas
- **Formula Verification**: [../evaluation/FORMULA_VERIFICATION.md](../evaluation/FORMULA_VERIFICATION.md) - Mathematical validation

### Key Mathematical Methods (Validated)
- **Evidential Reasoning**: Dempster-Shafer theory for belief aggregation (Shafer, 1976)
- **Graph Attention Networks**: Neural attention for dynamic expert weighting (Veličković et al., 2018)
- **TOPSIS**: Multi-criteria decision analysis (Hwang & Yoon, 1981)
- **Consensus Metrics**: Cosine similarity for agent agreement measurement
- **Historical Reliability**: Exponential moving average with dynamic weighting

### System Performance Benchmarks (from Academic Evaluation)
- **Consensus Level**: 92% (GAT aggregation method)
- **Decision Quality Score**: 84.7% average across 3 crisis scenarios
- **Processing Time**: 12.4s (3-agent core), 35-45s (13-agent full deployment)
- **GAT Improvement over ER**: +2.8% consensus, +1.5% confidence

---

## Support

For questions about this methodology:
1. Review the detailed guides in each subfolder
2. Check the examples for concrete implementations
3. Consult the evaluation methodology for quality assurance
4. See [../academic_paper.md](../academic_paper.md) for theoretical foundation

---

**Generated**: 2025-11-13
**Updated**: 2025-12-25 (Agent count verification and mathematical validation)
**System**: Crisis Management Multi-Agent System (Greek Emergency Response Edition)
**Version**: 1.1
**Author**: Vasileios Kazoukas
**Institution**: Technical University of Crete - Military Academy
