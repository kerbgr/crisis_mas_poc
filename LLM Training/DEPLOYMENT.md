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

## A/B Testing Framework

### Why A/B Testing for LLM Deployment?

**Problem**: You've fine-tuned a new model, but is it actually **better** than your current production model?

**Risks of Direct Replacement**:
- New model may perform worse on real-world queries (despite good benchmark scores)
- May introduce new failure modes not caught during evaluation
- User experience may degrade without you knowing
- Safety issues may only appear with production traffic

**Solution**: A/B testing allows you to:
- Compare new model (B) against current model (A) with real traffic
- Make data-driven decisions on model promotion
- Gradually roll out changes with minimal risk
- Quickly rollback if problems detected

---

### A/B Testing Strategies

#### Strategy 1: Traffic Splitting (Simple A/B Test)

**How it works**: Randomly assign incoming requests to Model A (current) or Model B (new) with specified ratio.

**Implementation**:

```python
# tools/ab_testing_server.py

import random
from flask import Flask, request, jsonify
import time
import json

app = Flask(__name__)

# Load both models
model_a = load_model("./models/firefighter-v1.0-production")  # Current
model_b = load_model("./models/firefighter-v1.1-candidate")   # New

# A/B test configuration
AB_TEST_CONFIG = {
    "model_a_weight": 0.8,  # 80% traffic to A (current model)
    "model_b_weight": 0.2,  # 20% traffic to B (new model)
    "enabled": True
}

# Metrics storage
metrics = {
    "model_a": {"requests": 0, "total_latency": 0, "errors": 0},
    "model_b": {"requests": 0, "total_latency": 0, "errors": 0}
}

def select_model():
    """Randomly select model based on weights."""
    if not AB_TEST_CONFIG["enabled"]:
        return model_a, "model_a"

    rand = random.random()
    if rand < AB_TEST_CONFIG["model_a_weight"]:
        return model_a, "model_a"
    else:
        return model_b, "model_b"

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    request_id = str(time.time())

    # Select model for this request
    model, model_name = select_model()

    # Track metrics
    start_time = time.time()

    try:
        # Generate response
        response = model.generate(data["messages"])

        latency = time.time() - start_time

        # Log for analysis
        log_ab_test_result(
            request_id=request_id,
            model_name=model_name,
            input=data["messages"],
            output=response,
            latency=latency,
            error=None
        )

        # Update metrics
        metrics[model_name]["requests"] += 1
        metrics[model_name]["total_latency"] += latency

        return jsonify({"choices": [{"message": {"content": response}}]})

    except Exception as e:
        latency = time.time() - start_time

        # Log error
        log_ab_test_result(
            request_id=request_id,
            model_name=model_name,
            input=data["messages"],
            output=None,
            latency=latency,
            error=str(e)
        )

        metrics[model_name]["errors"] += 1

        # Fallback to model A on error
        if model_name == "model_b":
            response = model_a.generate(data["messages"])
            return jsonify({"choices": [{"message": {"content": response}}]})
        else:
            return jsonify({"error": str(e)}), 500

def log_ab_test_result(request_id, model_name, input, output, latency, error):
    """Log every A/B test result for later analysis."""
    log_entry = {
        "timestamp": time.time(),
        "request_id": request_id,
        "model": model_name,
        "input": input,
        "output": output,
        "latency_ms": latency * 1000,
        "error": error
    }

    with open("ab_test_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """View current A/B test metrics."""
    return jsonify({
        "model_a": {
            "requests": metrics["model_a"]["requests"],
            "avg_latency_ms": (metrics["model_a"]["total_latency"] / metrics["model_a"]["requests"] * 1000) if metrics["model_a"]["requests"] > 0 else 0,
            "error_rate": metrics["model_a"]["errors"] / metrics["model_a"]["requests"] if metrics["model_a"]["requests"] > 0 else 0
        },
        "model_b": {
            "requests": metrics["model_b"]["requests"],
            "avg_latency_ms": (metrics["model_b"]["total_latency"] / metrics["model_b"]["requests"] * 1000) if metrics["model_b"]["requests"] > 0 else 0,
            "error_rate": metrics["model_b"]["errors"] / metrics["model_b"]["requests"] if metrics["model_b"]["requests"] > 0 else 0
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

**Usage**:
```bash
# Start A/B testing server
python tools/ab_testing_server.py

# Check metrics
curl http://localhost:8000/metrics
```

**Expected output**:
```json
{
  "model_a": {
    "requests": 1247,
    "avg_latency_ms": 342.5,
    "error_rate": 0.012
  },
  "model_b": {
    "requests": 318,
    "avg_latency_ms": 298.3,
    "error_rate": 0.019
  }
}
```

---

#### Strategy 2: Shadow Deployment (Zero-Risk Testing)

**How it works**: Send all traffic to Model A (production), but also run Model B in parallel. Log both outputs for comparison without affecting users.

**Advantages**:
- Zero user impact (users only see Model A)
- Compare both models on same queries
- Test Model B under real load before committing

**Implementation**:

```python
import threading

@app.route("/v1/chat/completions", methods=["POST"])
def chat_with_shadow():
    data = request.json
    request_id = str(time.time())

    # Primary: Model A (production)
    start_time = time.time()
    response_a = model_a.generate(data["messages"])
    latency_a = time.time() - start_time

    # Shadow: Model B (run in background, don't wait)
    def shadow_inference():
        start_time = time.time()
        try:
            response_b = model_b.generate(data["messages"])
            latency_b = time.time() - start_time

            # Log for comparison
            log_shadow_comparison(
                request_id=request_id,
                input=data["messages"],
                response_a=response_a,
                response_b=response_b,
                latency_a=latency_a,
                latency_b=latency_b
            )
        except Exception as e:
            log_shadow_error(request_id, str(e))

    # Run shadow model in background thread
    threading.Thread(target=shadow_inference).start()

    # Return Model A response immediately
    return jsonify({"choices": [{"message": {"content": response_a}}]})

def log_shadow_comparison(request_id, input, response_a, response_b, latency_a, latency_b):
    """Compare shadow model output to production."""
    log_entry = {
        "request_id": request_id,
        "input": input,
        "model_a": {"response": response_a, "latency_ms": latency_a * 1000},
        "model_b": {"response": response_b, "latency_ms": latency_b * 1000},
        "outputs_match": response_a.strip() == response_b.strip()
    }

    with open("shadow_comparison.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

**Analysis**:
```python
# tools/analyze_shadow_results.py

import json

def analyze_shadow_logs(log_file="shadow_comparison.jsonl"):
    """Analyze shadow deployment results."""

    with open(log_file, "r") as f:
        logs = [json.loads(line) for line in f]

    total = len(logs)
    outputs_match = sum(1 for log in logs if log["outputs_match"])
    avg_latency_a = sum(log["model_a"]["latency_ms"] for log in logs) / total
    avg_latency_b = sum(log["model_b"]["latency_ms"] for log in logs) / total

    print(f"Shadow Deployment Results ({total} requests):")
    print(f"  Outputs match: {outputs_match}/{total} ({100*outputs_match/total:.1f}%)")
    print(f"  Model A latency: {avg_latency_a:.1f}ms")
    print(f"  Model B latency: {avg_latency_b:.1f}ms")
    print(f"  Latency improvement: {100*(avg_latency_a - avg_latency_b)/avg_latency_a:+.1f}%")

    # Sample disagreements
    disagreements = [log for log in logs if not log["outputs_match"]][:5]
    print(f"\nSample disagreements:")
    for log in disagreements:
        print(f"\nInput: {log['input']}")
        print(f"Model A: {log['model_a']['response'][:100]}...")
        print(f"Model B: {log['model_b']['response'][:100]}...")

analyze_shadow_logs()
```

---

#### Strategy 3: Canary Deployment (Gradual Rollout)

**How it works**: Start with 1% traffic to Model B. If metrics look good, gradually increase to 5%, 10%, 25%, 50%, 100%.

**Timeline**: 1% → 5% → 10% → 25% → 50% → 100% over 2-4 weeks

**Implementation**:

```python
# Canary rollout schedule
CANARY_SCHEDULE = [
    {"date": "2025-11-13", "model_b_weight": 0.01},  # Day 1: 1%
    {"date": "2025-11-15", "model_b_weight": 0.05},  # Day 3: 5%
    {"date": "2025-11-18", "model_b_weight": 0.10},  # Day 6: 10%
    {"date": "2025-11-22", "model_b_weight": 0.25},  # Day 10: 25%
    {"date": "2025-11-27", "model_b_weight": 0.50},  # Day 15: 50%
    {"date": "2025-12-04", "model_b_weight": 1.00},  # Day 22: 100%
]

def get_canary_weight():
    """Get current traffic weight for Model B based on schedule."""
    from datetime import date

    today = str(date.today())

    for schedule in reversed(CANARY_SCHEDULE):
        if today >= schedule["date"]:
            return schedule["model_b_weight"]

    return 0.0  # No canary yet

# In your server
AB_TEST_CONFIG["model_b_weight"] = get_canary_weight()
AB_TEST_CONFIG["model_a_weight"] = 1.0 - get_canary_weight()
```

**Automatic Rollback**:
```python
def check_health_and_rollback():
    """Automatically rollback if Model B performs poorly."""

    if metrics["model_b"]["requests"] < 50:
        return  # Not enough data yet

    error_rate_a = metrics["model_a"]["errors"] / metrics["model_a"]["requests"]
    error_rate_b = metrics["model_b"]["errors"] / metrics["model_b"]["requests"]

    # Rollback if Model B error rate >2x Model A
    if error_rate_b > 2 * error_rate_a:
        print(f"⚠️ ROLLBACK: Model B error rate too high ({error_rate_b:.1%} vs {error_rate_a:.1%})")
        AB_TEST_CONFIG["model_b_weight"] = 0.0
        AB_TEST_CONFIG["model_a_weight"] = 1.0
        send_alert("Model B rolled back due to high error rate")

# Run health check every 5 minutes
import schedule
schedule.every(5).minutes.do(check_health_and_rollback)
```

---

### Metrics to Track

#### 1. Accuracy Metrics

**Manual evaluation** on sample of requests:

```python
# tools/ab_test_evaluation.py

import json
import random

def sample_for_evaluation(log_file, num_samples=100):
    """Sample random requests for expert evaluation."""

    with open(log_file, "r") as f:
        logs = [json.loads(line) for line in f]

    # Sample equally from both models
    model_a_logs = [log for log in logs if log["model"] == "model_a"]
    model_b_logs = [log for log in logs if log["model"] == "model_b"]

    sample_a = random.sample(model_a_logs, num_samples // 2)
    sample_b = random.sample(model_b_logs, num_samples // 2)

    # Create evaluation sheet (CSV)
    with open("ab_test_evaluation.csv", "w") as f:
        f.write("request_id,model,input,output,accuracy_score,safety_score,notes\n")

        for log in sample_a + sample_b:
            f.write(f"{log['request_id']},{log['model']},{log['input']},{log['output']},,,,\n")

    print(f"Created evaluation sheet with {num_samples} samples")
    print("Send ab_test_evaluation.csv to domain experts for scoring")

def analyze_evaluation_results(eval_file="ab_test_evaluation_completed.csv"):
    """Analyze expert ratings."""

    import pandas as pd

    df = pd.read_csv(eval_file)

    # Compare average scores
    results = df.groupby("model").agg({
        "accuracy_score": ["mean", "std"],
        "safety_score": ["mean", "std"]
    })

    print("Expert Evaluation Results:")
    print(results)

    # Statistical significance test
    from scipy import stats

    model_a_scores = df[df["model"] == "model_a"]["accuracy_score"]
    model_b_scores = df[df["model"] == "model_b"]["accuracy_score"]

    t_stat, p_value = stats.ttest_ind(model_a_scores, model_b_scores)

    print(f"\nStatistical Significance:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        if model_b_scores.mean() > model_a_scores.mean():
            print("  ✓ Model B is SIGNIFICANTLY BETTER (p < 0.05)")
        else:
            print("  ✗ Model B is SIGNIFICANTLY WORSE (p < 0.05)")
    else:
        print("  ≈ No significant difference (p >= 0.05)")

sample_for_evaluation("ab_test_logs.jsonl", num_samples=100)
```

#### 2. Latency Metrics

Track p50, p95, p99 latency:

```python
import numpy as np

def calculate_latency_percentiles(model_name):
    """Calculate latency percentiles for a model."""

    with open("ab_test_logs.jsonl", "r") as f:
        logs = [json.loads(line) for line in f if json.loads(line)["model"] == model_name]

    latencies = [log["latency_ms"] for log in logs if log["error"] is None]

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"{model_name} Latency:")
    print(f"  p50: {p50:.1f}ms")
    print(f"  p95: {p95:.1f}ms")
    print(f"  p99: {p99:.1f}ms")

    return {"p50": p50, "p95": p95, "p99": p99}
```

#### 3. Safety Metrics

Track safety-critical failures:

```python
def detect_safety_failures(output_text):
    """Detect potential safety issues in model output."""

    safety_issues = []

    # Check for hallucinated HAZMAT data
    if "IDLH" in output_text or "ppm" in output_text:
        # Validate against known database
        if not validate_hazmat_data(output_text):
            safety_issues.append("hallucinated_hazmat_data")

    # Check for unsafe recommendations
    unsafe_keywords = ["do not evacuate", "ignore warning", "not dangerous"]
    if any(kw in output_text.lower() for kw in unsafe_keywords):
        safety_issues.append("potentially_unsafe_recommendation")

    # Check for overconfidence on uncertain answer
    if "I'm certain" in output_text or "definitely" in output_text:
        if not is_well_supported(output_text):
            safety_issues.append("overconfident_uncertain_answer")

    return safety_issues

# Track safety failure rate
model_a_safety = sum(len(detect_safety_failures(log["output"])) for log in logs_a) / len(logs_a)
model_b_safety = sum(len(detect_safety_failures(log["output"])) for log in logs_b) / len(logs_b)

print(f"Safety failure rate:")
print(f"  Model A: {model_a_safety:.2%}")
print(f"  Model B: {model_b_safety:.2%}")
```

---

### Decision Criteria: When to Promote Model B

**Promote Model B to production if ALL criteria met**:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Accuracy** | Model B ≥ Model A + 2% | Significant improvement |
| **Safety** | Model B safety failures ≤ Model A | No regressions |
| **Latency p95** | Model B ≤ Model A + 50ms | Acceptable slowdown |
| **Error rate** | Model B ≤ Model A | No increase in crashes |
| **Expert rating** | Model B ≥ 4.0/5.0 | Expert approval |
| **Sample size** | ≥200 requests per model | Statistical significance |
| **p-value** | p < 0.05 | Statistically significant |

**Example decision matrix**:

```python
def should_promote_model_b(metrics_a, metrics_b, expert_ratings_a, expert_ratings_b):
    """Decide whether to promote Model B based on decision criteria."""

    checks = {}

    # 1. Accuracy improvement
    accuracy_diff = metrics_b["accuracy"] - metrics_a["accuracy"]
    checks["accuracy"] = accuracy_diff >= 0.02

    # 2. Safety (no regressions)
    checks["safety"] = metrics_b["safety_failures"] <= metrics_a["safety_failures"]

    # 3. Latency (acceptable slowdown)
    latency_diff = metrics_b["latency_p95"] - metrics_a["latency_p95"]
    checks["latency"] = latency_diff <= 50

    # 4. Error rate
    checks["errors"] = metrics_b["error_rate"] <= metrics_a["error_rate"]

    # 5. Expert rating
    checks["expert_rating"] = expert_ratings_b["mean"] >= 4.0

    # 6. Sample size
    checks["sample_size"] = metrics_b["requests"] >= 200

    # 7. Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(expert_ratings_a["scores"], expert_ratings_b["scores"])
    checks["significance"] = p_value < 0.05

    # Print results
    print("Promotion Decision Criteria:")
    for criterion, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")

    # Decision
    if all(checks.values()):
        print("\n✓ PROMOTE Model B to production")
        return True
    else:
        print("\n✗ DO NOT promote Model B (criteria not met)")
        return False
```

---

### Complete A/B Testing Workflow

**Week 1: Shadow Deployment**
1. Deploy Model B in shadow mode (100% traffic to A, shadow to B)
2. Collect 500+ shadow comparisons
3. Analyze disagreements and latency
4. Fix any critical issues found

**Week 2-3: Canary Deployment**
1. Start canary at 1% traffic
2. Monitor error rate, latency, safety failures
3. If stable after 48 hours, increase to 5%
4. Continue gradual increase: 10% → 25% → 50%

**Week 4: Evaluation**
1. Collect 200+ requests per model
2. Sample 100 requests for expert evaluation
3. Run statistical significance tests
4. Make promotion decision

**Week 5: Rollout or Rollback**
- If criteria met: Promote Model B to 100%
- If criteria not met: Rollback to 100% Model A, investigate issues

---

### Monitoring Dashboard Example

```python
# tools/ab_test_dashboard.py

from flask import Flask, render_template
import json

app = Flask(__name__)

@app.route("/dashboard")
def dashboard():
    """Real-time A/B testing dashboard."""

    # Load metrics
    with open("ab_test_logs.jsonl", "r") as f:
        logs = [json.loads(line) for line in f]

    model_a_logs = [log for log in logs if log["model"] == "model_a"]
    model_b_logs = [log for log in logs if log["model"] == "model_b"]

    # Calculate metrics
    metrics = {
        "model_a": {
            "requests": len(model_a_logs),
            "avg_latency": sum(log["latency_ms"] for log in model_a_logs) / len(model_a_logs),
            "error_rate": sum(1 for log in model_a_logs if log["error"]) / len(model_a_logs),
        },
        "model_b": {
            "requests": len(model_b_logs),
            "avg_latency": sum(log["latency_ms"] for log in model_b_logs) / len(model_b_logs),
            "error_rate": sum(1 for log in model_b_logs if log["error"]) / len(model_b_logs),
        }
    }

    return render_template("ab_test_dashboard.html", metrics=metrics)

if __name__ == "__main__":
    app.run(port=5000)
```

**Access dashboard**: http://localhost:5000/dashboard

---

### Rollback Plan

**Automatic rollback triggers**:
1. Model B error rate >2x Model A
2. Model B p95 latency >500ms
3. Model B safety failure rate >5%
4. Manual override by operator

**Rollback procedure**:
```python
def rollback_to_model_a():
    """Immediately rollback to Model A."""

    print("⚠️ ROLLING BACK TO MODEL A")

    # Set traffic to 100% Model A
    AB_TEST_CONFIG["model_a_weight"] = 1.0
    AB_TEST_CONFIG["model_b_weight"] = 0.0
    AB_TEST_CONFIG["enabled"] = False

    # Send alerts
    send_alert("Model B rolled back to Model A")

    # Log rollback
    with open("rollback_log.txt", "a") as f:
        f.write(f"{time.time()}: Rolled back to Model A\n")

# Manual rollback via API
@app.route("/rollback", methods=["POST"])
def rollback():
    rollback_to_model_a()
    return jsonify({"status": "rolled back to Model A"})
```

**Trigger rollback**:
```bash
curl -X POST http://localhost:8000/rollback
```

---

## Production Checklist

Before deploying to production:

- [ ] Model evaluated and meets quality targets (>80% accuracy)
- [ ] Quantized for target hardware (q4_k_m or q5_k_m)
- [ ] API endpoint tested with load testing
- [ ] Logging and monitoring configured
- [ ] Rate limiting and authentication enabled
- [ ] Input validation implemented
- [ ] **A/B testing plan defined** (shadow → canary → full rollout)
- [ ] **Decision criteria established** (accuracy, safety, latency thresholds)
- [ ] **Rollback procedure tested** (automatic + manual triggers)
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
