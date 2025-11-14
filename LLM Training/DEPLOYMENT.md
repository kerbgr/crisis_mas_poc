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

## Production Monitoring and Drift Detection

### Why Production Monitoring is Critical

**Problem**: Models degrade over time in production due to:
- **Concept drift**: Real-world patterns change (new fire suppression techniques, updated regulations)
- **Data drift**: Input distribution changes (different types of emergencies, seasonal patterns)
- **Model staleness**: Training data becomes outdated
- **Performance degradation**: Accuracy drops without you knowing

**Without monitoring**:
- Model quietly degrades from 85% → 70% accuracy over months
- Users lose trust in the system
- Dangerous recommendations go undetected
- You only find out when someone complains

**With monitoring**:
- Detect degradation within hours/days
- Trigger automatic retraining
- Alert operators to issues
- Maintain user trust

---

### Real-Time Performance Monitoring

#### 1. Response Quality Tracking

**Metrics to monitor**:

```python
# tools/production_monitor.py

import time
import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class ResponseMetrics:
    """Track metrics for each model response."""
    timestamp: float
    request_id: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    user_feedback: int  # -1 (thumbs down), 0 (no feedback), 1 (thumbs up)
    safety_flags: List[str]  # ["hallucination", "unsafe_recommendation", etc.]
    confidence_score: float  # Model's self-reported confidence

class ProductionMonitor:
    """Real-time monitoring for production LLM."""

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.recent_metrics = deque(maxlen=window_size)

        # Baseline metrics (from initial deployment)
        self.baseline = {
            "avg_latency_ms": 300.0,
            "thumbs_up_rate": 0.75,
            "safety_failure_rate": 0.02,
            "avg_confidence": 0.80
        }

        # Alert thresholds
        self.thresholds = {
            "latency_p95_ms": 800,  # Alert if p95 latency > 800ms
            "thumbs_down_rate": 0.20,  # Alert if >20% thumbs down
            "safety_failure_rate": 0.05,  # Alert if >5% safety failures
            "confidence_drop": 0.15  # Alert if confidence drops >15%
        }

    def log_response(self, metrics: ResponseMetrics):
        """Log a model response for monitoring."""
        self.recent_metrics.append(metrics)

        # Check for immediate issues
        self._check_alerts(metrics)

        # Periodic drift detection (every 100 requests)
        if len(self.recent_metrics) % 100 == 0:
            self._detect_drift()

    def _check_alerts(self, metrics: ResponseMetrics):
        """Check for immediate issues requiring alerts."""

        # Alert 1: High latency
        if metrics.latency_ms > self.thresholds["latency_p95_ms"]:
            self._send_alert(
                severity="warning",
                message=f"High latency detected: {metrics.latency_ms:.0f}ms",
                request_id=metrics.request_id
            )

        # Alert 2: Safety flags
        if len(metrics.safety_flags) > 0:
            self._send_alert(
                severity="critical",
                message=f"Safety issue: {metrics.safety_flags}",
                request_id=metrics.request_id
            )

        # Alert 3: Low confidence on critical query
        if metrics.confidence_score < 0.5:
            self._send_alert(
                severity="info",
                message=f"Low confidence response: {metrics.confidence_score:.2f}",
                request_id=metrics.request_id
            )

    def _detect_drift(self):
        """Detect performance drift over time."""

        if len(self.recent_metrics) < 100:
            return  # Not enough data

        current_metrics = self._calculate_current_metrics()

        # Compare to baseline
        issues = []

        # Check latency drift
        if current_metrics["avg_latency_ms"] > self.baseline["avg_latency_ms"] * 1.5:
            issues.append(f"Latency increased by {100*(current_metrics['avg_latency_ms']/self.baseline['avg_latency_ms']-1):.0f}%")

        # Check user satisfaction drift
        if current_metrics["thumbs_up_rate"] < self.baseline["thumbs_up_rate"] - 0.10:
            issues.append(f"User satisfaction dropped to {current_metrics['thumbs_up_rate']:.1%}")

        # Check safety drift
        if current_metrics["safety_failure_rate"] > self.baseline["safety_failure_rate"] * 2:
            issues.append(f"Safety failures increased to {current_metrics['safety_failure_rate']:.1%}")

        # Check confidence drift
        if current_metrics["avg_confidence"] < self.baseline["avg_confidence"] - self.thresholds["confidence_drop"]:
            issues.append(f"Model confidence dropped to {current_metrics['avg_confidence']:.2f}")

        if issues:
            self._send_alert(
                severity="warning",
                message="Performance drift detected:\n" + "\n".join(f"  - {issue}" for issue in issues)
            )

    def _calculate_current_metrics(self) -> Dict:
        """Calculate metrics over recent window."""

        recent = list(self.recent_metrics)[-self.window_size:]

        return {
            "avg_latency_ms": np.mean([m.latency_ms for m in recent]),
            "p95_latency_ms": np.percentile([m.latency_ms for m in recent], 95),
            "thumbs_up_rate": sum(1 for m in recent if m.user_feedback == 1) / len(recent),
            "thumbs_down_rate": sum(1 for m in recent if m.user_feedback == -1) / len(recent),
            "safety_failure_rate": sum(1 for m in recent if len(m.safety_flags) > 0) / len(recent),
            "avg_confidence": np.mean([m.confidence_score for m in recent])
        }

    def _send_alert(self, severity: str, message: str, request_id: str = None):
        """Send alert to monitoring system."""
        alert = {
            "timestamp": time.time(),
            "severity": severity,
            "message": message,
            "request_id": request_id
        }

        # Log to file
        with open("production_alerts.jsonl", "a") as f:
            f.write(json.dumps(alert) + "\n")

        # Send to monitoring service (Slack, PagerDuty, etc.)
        if severity == "critical":
            self._send_to_pagerduty(alert)
        else:
            self._send_to_slack(alert)

        print(f"[{severity.upper()}] {message}")

    def _send_to_slack(self, alert: Dict):
        """Send alert to Slack channel."""
        # Placeholder - integrate with your Slack webhook
        pass

    def _send_to_pagerduty(self, alert: Dict):
        """Send critical alert to PagerDuty."""
        # Placeholder - integrate with PagerDuty API
        pass

    def get_dashboard_metrics(self) -> Dict:
        """Get current metrics for monitoring dashboard."""

        if len(self.recent_metrics) == 0:
            return {}

        current = self._calculate_current_metrics()

        # Calculate trends (last 100 vs previous 100)
        if len(self.recent_metrics) >= 200:
            recent_100 = list(self.recent_metrics)[-100:]
            previous_100 = list(self.recent_metrics)[-200:-100]

            recent_latency = np.mean([m.latency_ms for m in recent_100])
            previous_latency = np.mean([m.latency_ms for m in previous_100])
            latency_trend = "↑" if recent_latency > previous_latency * 1.1 else "↓" if recent_latency < previous_latency * 0.9 else "→"

            recent_satisfaction = sum(1 for m in recent_100 if m.user_feedback == 1) / len(recent_100)
            previous_satisfaction = sum(1 for m in previous_100 if m.user_feedback == 1) / len(previous_100)
            satisfaction_trend = "↑" if recent_satisfaction > previous_satisfaction * 1.1 else "↓" if recent_satisfaction < previous_satisfaction * 0.9 else "→"
        else:
            latency_trend = "→"
            satisfaction_trend = "→"

        return {
            "current": current,
            "baseline": self.baseline,
            "trends": {
                "latency": latency_trend,
                "satisfaction": satisfaction_trend
            },
            "total_requests": len(self.recent_metrics)
        }

# Usage in production server
monitor = ProductionMonitor(window_size=1000)

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    request_id = str(time.time())

    start_time = time.time()

    # Generate response
    response = model.generate(data["messages"])

    latency_ms = (time.time() - start_time) * 1000

    # Extract confidence (if model provides it)
    confidence = extract_confidence(response)

    # Detect safety issues
    safety_flags = detect_safety_issues(response)

    # Log metrics
    metrics = ResponseMetrics(
        timestamp=time.time(),
        request_id=request_id,
        latency_ms=latency_ms,
        input_tokens=len(tokenizer.encode(str(data["messages"]))),
        output_tokens=len(tokenizer.encode(response)),
        user_feedback=0,  # Updated later via feedback endpoint
        safety_flags=safety_flags,
        confidence_score=confidence
    )

    monitor.log_response(metrics)

    return jsonify({"choices": [{"message": {"content": response}}]})

# Feedback endpoint
@app.route("/feedback", methods=["POST"])
def feedback():
    """Collect user feedback (thumbs up/down)."""
    data = request.json
    request_id = data["request_id"]
    feedback_value = data["feedback"]  # 1 or -1

    # Update metrics for this request
    for metrics in monitor.recent_metrics:
        if metrics.request_id == request_id:
            metrics.user_feedback = feedback_value
            break

    return jsonify({"status": "feedback recorded"})
```

---

#### 2. Concept Drift Detection

**What is concept drift**: The relationship between inputs and outputs changes over time.

**Example in emergency response**:
- **2023**: "Evacuate buildings during wildfire" → Correct answer
- **2024**: New regulation requires "shelter in place if fire >5km away" → Old answer now partially wrong

**Detection methods**:

```python
class ConceptDriftDetector:
    """Detect when model's learned concepts become outdated."""

    def __init__(self):
        self.reference_set = self._load_reference_set()
        self.drift_threshold = 0.15  # Alert if accuracy drops >15%

    def _load_reference_set(self):
        """Load gold-standard test set for drift detection."""
        # This is your evaluation benchmark from training
        with open("reference_test_set.json", "r") as f:
            return json.load(f)

    def check_drift_monthly(self, model):
        """Run drift detection monthly on reference set."""

        print("Running concept drift detection...")

        correct = 0
        total = len(self.reference_set)

        for item in self.reference_set:
            response = model.generate(item["question"])

            # Check if answer is correct (fuzzy match)
            if self._is_correct(response, item["correct_answer"]):
                correct += 1

        current_accuracy = correct / total
        baseline_accuracy = 0.85  # Your accuracy at deployment time

        accuracy_drop = baseline_accuracy - current_accuracy

        print(f"Reference Set Accuracy: {current_accuracy:.1%}")
        print(f"Baseline Accuracy: {baseline_accuracy:.1%}")
        print(f"Accuracy Drop: {accuracy_drop:+.1%}")

        if accuracy_drop > self.drift_threshold:
            self._send_drift_alert(
                current_accuracy=current_accuracy,
                baseline_accuracy=baseline_accuracy,
                accuracy_drop=accuracy_drop
            )

            return True  # Drift detected

        return False  # No drift

    def _is_correct(self, response: str, correct_answer: str) -> bool:
        """Check if response matches correct answer (fuzzy)."""
        from difflib import SequenceMatcher

        similarity = SequenceMatcher(None, response.lower(), correct_answer.lower()).ratio()
        return similarity > 0.8

    def _send_drift_alert(self, current_accuracy, baseline_accuracy, accuracy_drop):
        """Alert that concept drift detected."""
        alert = {
            "type": "concept_drift",
            "severity": "critical",
            "message": f"Model accuracy dropped from {baseline_accuracy:.1%} to {current_accuracy:.1%}",
            "recommendation": "Retrain model with updated data",
            "timestamp": time.time()
        }

        with open("drift_alerts.jsonl", "a") as f:
            f.write(json.dumps(alert) + "\n")

        # Send to monitoring
        send_to_pagerduty(alert)

# Schedule monthly drift detection
import schedule

drift_detector = ConceptDriftDetector()

def run_drift_detection():
    drift_detected = drift_detector.check_drift_monthly(model)
    if drift_detected:
        print("⚠️ CONCEPT DRIFT DETECTED - Retraining recommended")

# Run on first day of each month
schedule.every().month.at("00:00").do(run_drift_detection)
```

---

#### 3. Data Drift Detection

**What is data drift**: The distribution of input queries changes over time.

**Example**:
- **Training**: 60% wildfire, 30% structure fire, 10% HAZMAT
- **Production (Summer)**: 80% wildfire, 15% structure fire, 5% HAZMAT
- **Production (Winter)**: 20% wildfire, 70% structure fire, 10% HAZMAT

**Detection using embedding distance**:

```python
class DataDriftDetector:
    """Detect when input distribution changes."""

    def __init__(self, model):
        self.model = model
        self.tokenizer = tokenizer
        self.reference_embeddings = self._compute_reference_embeddings()
        self.drift_threshold = 0.25  # Alert if distance > 0.25

    def _compute_reference_embeddings(self):
        """Compute embeddings for training data distribution."""
        # Sample 1000 examples from training data
        with open("training_sample.json", "r") as f:
            training_data = json.load(f)

        embeddings = []
        for item in training_data[:1000]:
            emb = self._get_embedding(item["input"])
            embeddings.append(emb)

        # Average embedding (centroid)
        reference_centroid = np.mean(embeddings, axis=0)

        return {"centroid": reference_centroid, "embeddings": embeddings}

    def _get_embedding(self, text: str):
        """Get embedding for text using model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state, mean pooling
            embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]
        return embedding

    def check_drift_weekly(self, recent_queries: List[str]):
        """Check if recent queries have drifted from training distribution."""

        print("Running data drift detection...")

        # Get embeddings for recent queries
        recent_embeddings = [self._get_embedding(q) for q in recent_queries]
        recent_centroid = np.mean(recent_embeddings, axis=0)

        # Calculate distance between centroids (cosine distance)
        from scipy.spatial.distance import cosine

        distance = cosine(self.reference_embeddings["centroid"], recent_centroid)

        print(f"Data Drift Distance: {distance:.3f}")

        if distance > self.drift_threshold:
            self._send_drift_alert(distance)
            return True

        return False

    def _send_drift_alert(self, distance):
        """Alert that data drift detected."""
        alert = {
            "type": "data_drift",
            "severity": "warning",
            "message": f"Input distribution has drifted (distance: {distance:.3f})",
            "recommendation": "Analyze recent queries for new patterns. Consider collecting new training data.",
            "timestamp": time.time()
        }

        with open("drift_alerts.jsonl", "a") as f:
            f.write(json.dumps(alert) + "\n")

        send_to_slack(alert)

# Run weekly
data_drift_detector = DataDriftDetector(model)

def check_data_drift():
    # Get last 500 queries
    with open("llm_requests.log", "r") as f:
        recent_queries = [json.loads(line)["messages"][-1]["content"] for line in f.readlines()[-500:]]

    drift_detected = data_drift_detector.check_drift_weekly(recent_queries)
    if drift_detected:
        print("⚠️ DATA DRIFT DETECTED - Input distribution has changed")

schedule.every().monday.at("00:00").do(check_data_drift)
```

---

#### 4. Automated Monitoring Dashboard

**Real-time dashboard for monitoring model health**:

```python
# tools/monitoring_dashboard.py

from flask import Flask, render_template, jsonify
import json
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route("/dashboard")
def dashboard():
    """Production monitoring dashboard."""
    return render_template("monitor.html")

@app.route("/api/metrics")
def get_metrics():
    """API endpoint for dashboard data."""

    metrics = monitor.get_dashboard_metrics()

    # Add drift status
    metrics["drift_status"] = {
        "concept_drift": check_concept_drift_status(),
        "data_drift": check_data_drift_status()
    }

    # Add recent alerts
    metrics["recent_alerts"] = get_recent_alerts(limit=10)

    return jsonify(metrics)

@app.route("/api/timeseries")
def get_timeseries():
    """Get time-series data for charts."""

    # Load metrics from last 24 hours
    with open("llm_requests.log", "r") as f:
        logs = [json.loads(line) for line in f]

    cutoff_time = time.time() - 86400  # 24 hours ago
    recent_logs = [log for log in logs if log["timestamp"] > cutoff_time]

    # Aggregate by hour
    hourly_metrics = {}

    for log in recent_logs:
        hour = datetime.fromtimestamp(log["timestamp"]).strftime("%Y-%m-%d %H:00")

        if hour not in hourly_metrics:
            hourly_metrics[hour] = {
                "requests": 0,
                "total_latency": 0,
                "errors": 0,
                "thumbs_up": 0,
                "thumbs_down": 0
            }

        hourly_metrics[hour]["requests"] += 1
        hourly_metrics[hour]["total_latency"] += log.get("latency_ms", 0)
        if log.get("error"):
            hourly_metrics[hour]["errors"] += 1
        if log.get("feedback") == 1:
            hourly_metrics[hour]["thumbs_up"] += 1
        elif log.get("feedback") == -1:
            hourly_metrics[hour]["thumbs_down"] += 1

    # Format for chart
    timeseries = {
        "timestamps": sorted(hourly_metrics.keys()),
        "requests_per_hour": [hourly_metrics[h]["requests"] for h in sorted(hourly_metrics.keys())],
        "avg_latency_ms": [hourly_metrics[h]["total_latency"] / hourly_metrics[h]["requests"] if hourly_metrics[h]["requests"] > 0 else 0 for h in sorted(hourly_metrics.keys())],
        "error_rate": [hourly_metrics[h]["errors"] / hourly_metrics[h]["requests"] if hourly_metrics[h]["requests"] > 0 else 0 for h in sorted(hourly_metrics.keys())],
        "satisfaction_rate": [hourly_metrics[h]["thumbs_up"] / (hourly_metrics[h]["thumbs_up"] + hourly_metrics[h]["thumbs_down"]) if (hourly_metrics[h]["thumbs_up"] + hourly_metrics[h]["thumbs_down"]) > 0 else 0 for h in sorted(hourly_metrics.keys())]
    }

    return jsonify(timeseries)

def check_concept_drift_status():
    """Check if concept drift alert exists in last 30 days."""
    # Check drift_alerts.jsonl
    try:
        with open("drift_alerts.jsonl", "r") as f:
            alerts = [json.loads(line) for line in f]

        cutoff = time.time() - (30 * 86400)  # 30 days
        recent_concept_drift = any(
            a for a in alerts
            if a["type"] == "concept_drift" and a["timestamp"] > cutoff
        )

        return "DRIFTED" if recent_concept_drift else "STABLE"
    except FileNotFoundError:
        return "UNKNOWN"

def check_data_drift_status():
    """Check if data drift alert exists in last 7 days."""
    try:
        with open("drift_alerts.jsonl", "r") as f:
            alerts = [json.loads(line) for line in f]

        cutoff = time.time() - (7 * 86400)  # 7 days
        recent_data_drift = any(
            a for a in alerts
            if a["type"] == "data_drift" and a["timestamp"] > cutoff
        )

        return "DRIFTED" if recent_data_drift else "STABLE"
    except FileNotFoundError:
        return "UNKNOWN"

def get_recent_alerts(limit=10):
    """Get most recent alerts."""
    try:
        with open("production_alerts.jsonl", "r") as f:
            alerts = [json.loads(line) for line in f]

        # Sort by timestamp, most recent first
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)

        return alerts[:limit]
    except FileNotFoundError:
        return []

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

**Dashboard HTML Template** (`templates/monitor.html`):

```html
<!DOCTYPE html>
<html>
<head>
    <title>LLM Production Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 36px; font-weight: bold; margin: 10px 0; }
        .metric-label { color: #666; font-size: 14px; }
        .metric-trend { font-size: 18px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .alert { padding: 15px; margin: 10px 0; border-radius: 4px; }
        .alert-critical { background: #ffebee; border-left: 4px solid #f44336; }
        .alert-warning { background: #fff3e0; border-left: 4px solid #ff9800; }
        .alert-info { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .status-stable { color: #4caf50; }
        .status-drifted { color: #f44336; }
    </style>
</head>
<body>
    <h1>LLM Production Monitor</h1>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Requests (24h)</div>
            <div class="metric-value" id="total-requests">-</div>
            <div class="metric-trend" id="request-trend">-</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Avg Latency (ms)</div>
            <div class="metric-value" id="avg-latency">-</div>
            <div class="metric-trend" id="latency-trend">-</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">User Satisfaction</div>
            <div class="metric-value" id="satisfaction-rate">-</div>
            <div class="metric-trend" id="satisfaction-trend">-</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Error Rate</div>
            <div class="metric-value" id="error-rate">-</div>
            <div class="metric-trend">-</div>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Concept Drift Status</div>
            <div class="metric-value" id="concept-drift-status">-</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Data Drift Status</div>
            <div class="metric-value" id="data-drift-status">-</div>
        </div>
    </div>

    <div class="chart-container">
        <h2>Requests per Hour (24h)</h2>
        <canvas id="requests-chart"></canvas>
    </div>

    <div class="chart-container">
        <h2>Average Latency (24h)</h2>
        <canvas id="latency-chart"></canvas>
    </div>

    <div class="chart-container">
        <h2>User Satisfaction Rate (24h)</h2>
        <canvas id="satisfaction-chart"></canvas>
    </div>

    <div class="chart-container">
        <h2>Recent Alerts</h2>
        <div id="alerts-container"></div>
    </div>

    <script>
        // Fetch and update metrics every 10 seconds
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update metric cards
                    document.getElementById('total-requests').textContent = data.total_requests || '-';
                    document.getElementById('avg-latency').textContent =
                        data.current?.avg_latency_ms?.toFixed(0) || '-';
                    document.getElementById('satisfaction-rate').textContent =
                        ((data.current?.thumbs_up_rate || 0) * 100).toFixed(0) + '%';
                    document.getElementById('error-rate').textContent =
                        ((data.current?.safety_failure_rate || 0) * 100).toFixed(1) + '%';

                    // Update trends
                    document.getElementById('latency-trend').textContent = data.trends?.latency || '-';
                    document.getElementById('satisfaction-trend').textContent = data.trends?.satisfaction || '-';

                    // Update drift status
                    const conceptDriftEl = document.getElementById('concept-drift-status');
                    conceptDriftEl.textContent = data.drift_status?.concept_drift || 'UNKNOWN';
                    conceptDriftEl.className = 'metric-value ' +
                        (data.drift_status?.concept_drift === 'STABLE' ? 'status-stable' : 'status-drifted');

                    const dataDriftEl = document.getElementById('data-drift-status');
                    dataDriftEl.textContent = data.drift_status?.data_drift || 'UNKNOWN';
                    dataDriftEl.className = 'metric-value ' +
                        (data.drift_status?.data_drift === 'STABLE' ? 'status-stable' : 'status-drifted');

                    // Update alerts
                    const alertsContainer = document.getElementById('alerts-container');
                    alertsContainer.innerHTML = '';
                    (data.recent_alerts || []).forEach(alert => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = 'alert alert-' + alert.severity;
                        alertDiv.innerHTML = `
                            <strong>${alert.severity.toUpperCase()}</strong>: ${alert.message}
                            <br><small>${new Date(alert.timestamp * 1000).toLocaleString()}</small>
                        `;
                        alertsContainer.appendChild(alertDiv);
                    });
                });

            // Fetch time-series data for charts
            fetch('/api/timeseries')
                .then(response => response.json())
                .then(data => {
                    updateCharts(data);
                });
        }

        function updateCharts(data) {
            // Implementation of Chart.js charts
            // (chart code here)
        }

        // Update dashboard every 10 seconds
        updateDashboard();
        setInterval(updateDashboard, 10000);
    </script>
</body>
</html>
```

**Access dashboard**: http://localhost:5000/dashboard

---

#### 5. Alerting Best Practices

**Alert Severity Levels**:

| Severity | Trigger | Response Time | Example |
|----------|---------|---------------|---------|
| **Critical** | Safety failure, model crash | Immediate (PagerDuty) | Hallucinated HAZMAT data, unsafe recommendation |
| **Warning** | Performance degradation | 1 hour | Latency >2x baseline, accuracy drop >10% |
| **Info** | Minor issues | 24 hours | Low confidence response, typo detected |

**Alert Fatigue Prevention**:

```python
class AlertManager:
    """Prevent alert fatigue with smart grouping and throttling."""

    def __init__(self):
        self.alert_history = {}
        self.throttle_window = 3600  # 1 hour

    def should_send_alert(self, alert_type: str) -> bool:
        """Throttle repeated alerts."""

        last_alert_time = self.alert_history.get(alert_type, 0)
        current_time = time.time()

        if current_time - last_alert_time < self.throttle_window:
            return False  # Alert sent recently, throttle

        self.alert_history[alert_type] = current_time
        return True

    def group_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Group similar alerts together."""

        grouped = {}

        for alert in alerts:
            key = (alert["severity"], alert["type"])

            if key not in grouped:
                grouped[key] = {
                    "severity": alert["severity"],
                    "type": alert["type"],
                    "count": 0,
                    "first_occurrence": alert["timestamp"],
                    "messages": []
                }

            grouped[key]["count"] += 1
            grouped[key]["messages"].append(alert["message"])

        # Format grouped alerts
        result = []
        for (severity, alert_type), data in grouped.items():
            if data["count"] > 1:
                message = f"{data['count']} {alert_type} alerts (first: {data['messages'][0]})"
            else:
                message = data["messages"][0]

            result.append({
                "severity": severity,
                "type": alert_type,
                "message": message,
                "count": data["count"]
            })

        return result
```

---

### Monitoring Checklist

**Daily**:
- [ ] Check dashboard for any critical alerts
- [ ] Review user feedback (thumbs up/down ratio)
- [ ] Check p95 latency < 800ms
- [ ] Verify error rate < 2%

**Weekly**:
- [ ] Run data drift detection
- [ ] Review sample of model responses (10-20 random)
- [ ] Check for new failure modes
- [ ] Update reference test set if needed

**Monthly**:
- [ ] Run concept drift detection on reference set
- [ ] Analyze long-term trends (latency, satisfaction)
- [ ] Expert evaluation of 50 random responses
- [ ] Review and update alert thresholds

**Quarterly**:
- [ ] Full model evaluation on benchmark
- [ ] Compare to baseline metrics from deployment
- [ ] Decision: Retrain or continue monitoring
- [ ] Update documentation with findings

---

### Retraining Decision Criteria

**Trigger retraining if**:

| Criterion | Threshold | Severity |
|-----------|-----------|----------|
| **Concept drift detected** | Accuracy drop >15% | Critical |
| **User satisfaction** | Thumbs up rate <60% | Critical |
| **Safety failures** | Rate >5% | Critical |
| **Data drift** | Sustained >4 weeks | High |
| **New procedures** | Major regulation change | High |
| **Time since training** | >6 months | Medium |

**Retraining workflow**:

```bash
# 1. Collect new training data (include recent production queries)
python data_collection/collect_recent_production_data.py \
  --output new_training_data.json

# 2. Merge with existing training data
python data_collection/merge_datasets.py \
  --existing training_data.json \
  --new new_training_data.json \
  --output training_data_v2.json

# 3. Retrain model
python fine_tuning/finetune.py \
  --data training_data_v2.json \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output firefighter-llama3.1-8b-v2

# 4. Evaluate new model
python evaluation/run_benchmark.py \
  --model firefighter-llama3.1-8b-v2 \
  --benchmark firefighter

# 5. A/B test new model vs production
# (See A/B Testing Framework section)
```

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

### Model Preparation
- [ ] Model evaluated and meets quality targets (>80% accuracy)
- [ ] Quantized for target hardware (q4_k_m or q5_k_m)
- [ ] API endpoint tested with load testing
- [ ] Expert validation completed

### Security and Validation
- [ ] Rate limiting and authentication enabled
- [ ] Input validation implemented
- [ ] Security audit completed

### Monitoring and Alerting
- [ ] **Production monitoring dashboard deployed** (real-time metrics)
- [ ] **Drift detection configured** (concept drift + data drift)
- [ ] **Alert thresholds configured** (latency, safety, satisfaction)
- [ ] **Logging and monitoring configured** (requests, responses, feedback)
- [ ] **Baseline metrics recorded** (accuracy, latency, satisfaction at deployment)
- [ ] **Reference test set prepared** (for monthly drift detection)

### Deployment Strategy
- [ ] **A/B testing plan defined** (shadow → canary → full rollout)
- [ ] **Decision criteria established** (accuracy, safety, latency thresholds)
- [ ] **Rollback procedure tested** (automatic + manual triggers)
- [ ] Backup model available (rollback plan)

### Documentation
- [ ] **Monitoring runbook** (how to interpret dashboard, respond to alerts)
- [ ] **Retraining procedure documented** (when and how to retrain)
- [ ] Documentation for operators written
- [ ] Incident response plan created

---

## Next Steps

1. **Start with LM Studio** for local testing
2. **Move to vLLM** for production deployment
3. **Deploy monitoring dashboard** and configure alerts
4. **Monitor performance** continuously (daily/weekly/monthly checks)
5. **Iterate**: Retrain when drift detected or every 3-6 months

---

**Generated**: 2025-11-14
**Version**: 1.1
