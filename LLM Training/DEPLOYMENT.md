# Deployment and Inference Guide

## Overview

This guide covers deploying your trained domain-specific LLM for production use in the **Greek Emergency Response Multi-Agent System**.

**Deployment Options**:
1. **LM Studio** (Desktop GUI, easiest)
2. **Ollama** (CLI, lightweight)
3. **vLLM** (High-performance server, production)
4. **Custom Python API** (Integration with existing system)

---

## Quick Start: LM Studio (Recommended for Testing)

### Step 1: Convert Model to GGUF

```bash
# Install llama.cpp Python bindings
pip install llama-cpp-python

# Convert LoRA-merged model to GGUF
python tools/model_quantizer.py \
  --model ./fine_tuning/merged_models/firefighter-llama3.1-8b \
  --output ./deployed_models/firefighter-q4.gguf \
  --quantization q4_k_m

# q4_k_m = 4-bit quantization, medium quality (good speed/quality balance)
# Alternatives: q5_k_m (higher quality), q8_0 (near-full quality), f16 (full precision)
```

### Step 2: Load in LM Studio

1. Download **LM Studio** from https://lmstudio.ai
2. Open LM Studio → **Local Models**
3. Click **Import Model** → Select `firefighter-q4.gguf`
4. Configure **System Prompt**:
   ```
   You are Pyragos Ioanna Michaelidou, an experienced Greek fire commander with 15 years of service in the Hellenic Fire Corps.
   ```
5. Click **Start Server** (default: http://localhost:1234)

### Step 3: Test API

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are Pyragos Ioanna Michaelidou..."},
      {"role": "user", "content": "What is the IDLH for ammonia?"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

---

## Production Deployment: vLLM Server

### Why vLLM?

- **Fast**: PagedAttention optimization, 10x faster than Hugging Face
- **Scalable**: Batched inference, handles multiple requests concurrently
- **Compatible**: OpenAI API format
- **Production-ready**: Used by major AI companies

### Installation

```bash
pip install vllm

# Verify GPU support
python -c "import vllm; print(vllm.__version__)"
```

### Launch Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./fine_tuning/merged_models/firefighter-llama3.1-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096

# Options:
# --tensor-parallel-size: Number of GPUs (1 = single GPU)
# --dtype: bfloat16 (faster) or float16
# --max-model-len: Max context length (reduce if OOM)
```

### Test vLLM Server

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require real key
)

response = client.chat.completions.create(
    model="firefighter-llama3.1-8b",
    messages=[
        {"role": "system", "content": "You are Pyragos Ioanna Michaelidou..."},
        {"role": "user", "content": "A wildfire is approaching. What do I do?"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

---

## Integration with Crisis MAS

### Option 1: LMStudioClient Wrapper (Existing)

Your system already has `llm_integration/lmstudio_client.py`:

```python
from llm_integration.lmstudio_client import LMStudioClient

# Point to your deployed model
client = LMStudioClient(
    base_url="http://localhost:1234/v1",  # or vLLM server
    model_name="firefighter-llama3.1-8b"
)

# Use in expert agent
from agents.expert_agent import ExpertAgent

firefighter = ExpertAgent(
    name="Pyragos Ioanna Michaelidou",
    role="Fire Tactical Commander",
    expertise="firefighting",
    llm_client=client
)

assessment = firefighter.assess_scenario(scenario_data)
```

### Option 2: Direct Integration

```python
# agents/expert_agent.py (add new method)

def _load_fine_tuned_model(self, model_path):
    """Load fine-tuned domain-specific model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        load_in_8bit=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)

    return model, tokenizer
```

---

## Inference Optimization

### Quantization for Speed

| Quantization | Model Size | Speed | Quality | Use Case |
|--------------|------------|-------|---------|----------|
| **f16** | 14.3GB | 1x | 100% | Development/testing |
| **q8_0** | 7.6GB | 1.5x | 99% | High-quality production |
| **q5_k_m** | 5.2GB | 2x | 97% | Balanced production |
| **q4_k_m** | 4.1GB | 3x | 95% | Fast production (recommended) |
| **q3_k_m** | 3.3GB | 4x | 90% | Mobile/edge devices |

**Recommendation**: Use `q4_k_m` for production (95% quality, 3x faster)

### Hardware Requirements

**Minimum**:
- CPU: Intel i5 or AMD Ryzen 5 (for CPU inference)
- RAM: 16GB
- GPU: RTX 3060 12GB (for GPU inference)

**Recommended**:
- CPU: Intel i7/i9 or AMD Ryzen 7/9
- RAM: 32GB
- GPU: RTX 4090 24GB or A100

**Performance**:
- **CPU (q4_k_m)**: 5-10 tokens/sec (slow but works)
- **RTX 4090 (q4_k_m)**: 80-120 tokens/sec (excellent)
- **A100 (q4_k_m)**: 150-200 tokens/sec (production)

---

## Multi-Agent Deployment Strategy

### Approach A: Single Merged Model (Simple)

- Train one model on **all** domain data (firefighter + police + medical)
- Use different system prompts to activate different "personalities"
- **Pros**: Simple deployment, single server
- **Cons**: Less specialized, may confuse domains

```python
# Single model, different system prompts
firefighter_prompt = "You are Pyragos Ioanna Michaelidou, fire commander..."
police_prompt = "You are Taxiarchos Nikos Konstantinou, police commander..."

firefighter = ExpertAgent(llm_client=client, system_prompt=firefighter_prompt)
police = ExpertAgent(llm_client=client, system_prompt=police_prompt)
```

---

### Approach B: Separate LoRA Adapters (Recommended)

- Train separate LoRA for each domain
- Load different adapter per agent
- **Pros**: Highly specialized, excellent quality
- **Cons**: More complex deployment

```python
from peft import PeftModel

# Load base model once
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Load different LoRA adapters
firefighter_model = PeftModel.from_pretrained(base_model, "./lora/firefighter")
police_model = PeftModel.from_pretrained(base_model, "./lora/police")
medical_model = PeftModel.from_pretrained(base_model, "./lora/medical")
```

**Memory efficient** with adapter swapping:
```python
# Swap adapters dynamically
from peft import set_peft_model_state_dict

model.set_adapter("firefighter")  # Switch to firefighter expertise
model.set_adapter("police")       # Switch to police expertise
```

---

### Approach C: Short LLMs per Agent (Fast but Limited)

- Train 1.5B-3B models per agent
- Ultra-fast inference (<100ms)
- **Pros**: Very fast, runs on CPU, easy scaling
- **Cons**: Limited reasoning, narrow expertise

```python
# 13 separate sLLMs (one per Greek expert)
firefighter_slm = load_slm("./slm/pyragos-qwen2-1.5b")
police_slm = load_slm("./slm/taxiarchos-qwen2-1.5b")
medical_slm = load_slm("./slm/ekab-physician-qwen2-1.5b")
# ... 10 more

# Fast parallel inference
with ThreadPoolExecutor(max_workers=13) as executor:
    assessments = executor.map(lambda agent: agent.assess(scenario), all_agents)
```

---

## API Server Example

### Custom Flask API

```python
# tools/deployment_server.py

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)

# Load model at startup
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    load_in_8bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./lora/firefighter")

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    messages = data["messages"]

    # Generate response
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids,
        max_new_tokens=data.get("max_tokens", 512),
        temperature=data.get("temperature", 0.7),
        do_sample=True
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({
        "choices": [{"message": {"role": "assistant", "content": response_text}}]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

**Run server**:
```bash
python tools/deployment_server.py
```

---

## Monitoring and Logging

### Log All Requests

```python
import logging

logging.basicConfig(
    filename='llm_requests.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    logging.info(f"Request: {data['messages']}")

    response = generate_response(data)

    logging.info(f"Response: {response}")
    return response
```

### Track Performance Metrics

```python
import time

def track_inference_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logging.info(f"Inference time: {duration:.2f}s")
        return result
    return wrapper

@track_inference_time
def generate_response(data):
    # ... inference code
```

---

## Security Considerations

### 1. Input Validation

```python
def validate_input(messages):
    # Check message length
    if len(messages) > 20:
        raise ValueError("Too many messages")

    # Check token count
    total_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)
    if total_tokens > 4000:
        raise ValueError("Input too long")

    # Check for malicious content (optional)
    for message in messages:
        if contains_sql_injection(message["content"]):
            raise ValueError("Invalid input detected")
```

### 2. Rate Limiting

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour", "10 per minute"]
)

@app.route("/v1/chat/completions", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    # ... inference code
```

### 3. Authentication

```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('Authorization')
        if not api_key or not is_valid_key(api_key):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route("/v1/chat/completions", methods=["POST"])
@require_api_key
def chat():
    # ... inference code
```

---

## Troubleshooting

### Issue: Slow Inference

**Solutions**:
1. Use quantized model (q4_k_m)
2. Reduce max_tokens
3. Use vLLM instead of standard transformers
4. Enable Flash Attention 2

### Issue: Out of Memory

**Solutions**:
1. Use load_in_8bit=True or load_in_4bit=True
2. Reduce max_model_len
3. Use smaller batch size
4. Use CPU offloading: device_map="auto"

### Issue: Model Not Responding Correctly

**Solutions**:
1. Check system prompt formatting
2. Verify chat template matches training format
3. Check temperature (too high = random, too low = repetitive)
4. Review training logs for convergence

---

## Production Checklist

Before deploying to production:

- [ ] Model evaluated and meets quality targets (>80% accuracy)
- [ ] Quantized for target hardware (q4_k_m or q5_k_m)
- [ ] API endpoint tested with load testing
- [ ] Logging and monitoring configured
- [ ] Rate limiting and authentication enabled
- [ ] Input validation implemented
- [ ] Backup model available (rollback plan)
- [ ] Expert validation completed
- [ ] Documentation for operators written

---

## Next Steps

1. **Start with LM Studio** for local testing
2. **Move to vLLM** for production deployment
3. **Monitor performance** and collect feedback
4. **Iterate**: Retrain with new data every 3-6 months

---

**Generated**: 2025-11-13
**Version**: 1.0
