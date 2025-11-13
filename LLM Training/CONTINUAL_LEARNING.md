# Continual Learning Strategy for Domain-Specific LLMs

## Overview

**Continual Learning** is the process of continuously updating and improving your deployed LLM over time as new data, feedback, and requirements emerge.

**Why Continual Learning is Critical**:
- **Domain knowledge evolves**: New protocols, regulations, equipment, tactics
- **Model drift**: Performance degrades as real-world distribution shifts
- **Feedback accumulates**: Users identify errors, edge cases, missing knowledge
- **New scenarios emerge**: Novel crisis types not in original training data

**Without continual learning**: Your model becomes outdated within 6-12 months, accuracy drops 10-20%, user trust erodes.

**With continual learning**: Model stays current, maintains >80% accuracy, adapts to new requirements.

---

## When to Update Your Model

### Scheduled Updates (Recommended)

**Quarterly Updates** (Every 3 months):
- Collect feedback from production usage
- Add 200-500 new training examples
- Retrain and deploy updated model
- **Effort**: 2-3 days per quarter

**Bi-Annual Updates** (Every 6 months):
- Major model improvements
- Incorporate regulatory changes
- Add new domain knowledge
- **Effort**: 1-2 weeks per update

**Annual Full Retraining**:
- Complete dataset refresh
- Consider new base model (Llama 3.2 → 4.0)
- Architectural improvements
- **Effort**: 2-4 weeks

---

### Trigger-Based Updates (As Needed)

**Immediate Update Required** if:
1. **Critical safety issue** (model gives dangerous advice)
2. **Regulatory change** (new HAZMAT protocols, EU AI Act update)
3. **Major accuracy drop** (>10% decrease in production metrics)
4. **New crisis type** (never seen before, e.g., novel chemical leak)

**Planned Update Required** if:
1. **Moderate accuracy drop** (5-10% decrease)
2. **Accumulated feedback** (>500 user corrections)
3. **New equipment/tactics** (new firefighting foam, updated evacuation procedures)
4. **Seasonal changes** (wildfire season → prepare for new scenarios)

---

## Data Collection for Continual Learning

### 1. Production Feedback Loop

**Collect from deployed model**:

```python
# tools/production_feedback.py

import json
from datetime import datetime

class FeedbackCollector:
    """Collect user feedback on model predictions."""

    def __init__(self, log_path="production_feedback.jsonl"):
        self.log_path = log_path

    def log_prediction(self, request_id, input, output, metadata):
        """Log every production prediction for potential training data."""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "input": input,
            "output": output,
            "metadata": metadata,
            "user_feedback": None  # Filled in later if user provides feedback
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def add_user_feedback(self, request_id, rating, correction=None, notes=None):
        """Add user feedback to logged prediction."""

        # Load all logs
        with open(self.log_path, "r") as f:
            logs = [json.loads(line) for line in f]

        # Find matching request
        for log in logs:
            if log["request_id"] == request_id:
                log["user_feedback"] = {
                    "rating": rating,  # 1-5 scale
                    "correction": correction,  # Corrected output if wrong
                    "notes": notes,
                    "timestamp": datetime.now().isoformat()
                }

        # Rewrite logs
        with open(self.log_path, "w") as f:
            for log in logs:
                f.write(json.dumps(log) + "\n")

    def extract_training_data(self, min_rating=4, output_file="continual_learning_data.jsonl"):
        """Extract high-quality examples for retraining."""

        with open(self.log_path, "r") as f:
            logs = [json.loads(line) for line in f]

        # Filter for high-quality feedback
        good_examples = []
        corrected_examples = []

        for log in logs:
            if log["user_feedback"] is None:
                continue

            rating = log["user_feedback"]["rating"]

            # High-rated examples (rating >= 4): Keep as-is
            if rating >= min_rating:
                good_examples.append({
                    "messages": [
                        {"role": "user", "content": log["input"]},
                        {"role": "assistant", "content": log["output"]}
                    ]
                })

            # Low-rated with correction: Use corrected version
            elif rating <= 2 and log["user_feedback"]["correction"]:
                corrected_examples.append({
                    "messages": [
                        {"role": "user", "content": log["input"]},
                        {"role": "assistant", "content": log["user_feedback"]["correction"]}
                    ]
                })

        # Combine
        all_examples = good_examples + corrected_examples

        # Save
        with open(output_file, "w") as f:
            for example in all_examples:
                f.write(json.dumps(example) + "\n")

        print(f"Extracted {len(all_examples)} training examples:")
        print(f"  High-quality: {len(good_examples)}")
        print(f"  Corrected: {len(corrected_examples)}")

        return all_examples

# Usage
collector = FeedbackCollector()

# During inference
collector.log_prediction(
    request_id="req_12345",
    input="What is IDLH for ammonia?",
    output="The IDLH for ammonia is 300 ppm.",
    metadata={"model_version": "v1.0", "latency_ms": 342}
)

# User provides feedback
collector.add_user_feedback(
    request_id="req_12345",
    rating=5,
    notes="Perfect answer"
)

# Extract for retraining
collector.extract_training_data()
```

---

### 2. Expert-in-the-Loop Annotation

**Schedule regular annotation sessions**:

```python
# tools/expert_annotation.py

def sample_for_annotation(num_samples=50):
    """Sample production queries for expert annotation."""

    with open("production_feedback.jsonl", "r") as f:
        logs = [json.loads(line) for line in f]

    # Sample diverse queries (not just errors)
    # Stratified sampling: errors, low-confidence, random
    errors = [log for log in logs if log["user_feedback"] and log["user_feedback"]["rating"] <= 2]
    low_conf = [log for log in logs if log["metadata"].get("confidence", 1.0) < 0.7]
    random_sample = random.sample([log for log in logs if log not in errors and log not in low_conf], 20)

    samples = errors[:15] + low_conf[:15] + random_sample[:20]

    # Create annotation sheet
    with open("expert_annotation_batch.csv", "w") as f:
        f.write("request_id,input,model_output,expert_output,accuracy,safety,notes\n")

        for log in samples:
            f.write(f"{log['request_id']},{log['input']},{log['output']},,,,\n")

    print(f"Created annotation batch with {len(samples)} samples")
    print("Send to domain expert for annotation")

# Schedule: Every 2 weeks, annotate 50 examples
# After 3 months: 50 * 6 = 300 new annotated examples
```

---

### 3. Synthetic Data Generation for New Scenarios

**Generate training data for new situations**:

```python
# tools/synthetic_data_generator.py

from openai import OpenAI

def generate_synthetic_examples(new_scenario_description, num_examples=50):
    """Generate synthetic training data for new scenario type."""

    client = OpenAI()

    examples = []

    prompt = f"""Generate {num_examples} realistic training examples for a Greek emergency response expert dealing with this new scenario:

{new_scenario_description}

Format each example as:
User: [realistic question from incident commander]
Assistant: [expert response with specific actions, protocols, safety considerations]

Make questions diverse: tactical decisions, resource allocation, safety protocols, coordination."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    # Parse generated examples (simplified)
    generated_text = response.choices[0].message.content

    # Convert to training format
    # ... (parsing logic)

    return examples

# Example: New wildfire scenario not in original training data
new_examples = generate_synthetic_examples(
    new_scenario_description="""
    A new type of wildfire scenario has emerged: urban-wildland interface fires in Greek island resorts during peak tourist season (July-August).

    Key challenges:
    - Thousands of tourists unfamiliar with Greek language
    - Limited evacuation routes (island geography)
    - Historical buildings that can't use standard suppression
    - Strong meltemi winds (30-50 km/h gusts)
    - Coordination with coast guard for sea evacuation

    Generate realistic incident commander questions and expert firefighter responses.
    """,
    num_examples=50
)
```

---

### 4. Real-World Incident Review

**After-Action Reports (AARs) from actual incidents**:

```python
# After real incident, expert reviews model performance
def review_incident_performance(incident_id):
    """Review model recommendations during real incident."""

    # Load all model outputs during incident
    with open(f"incidents/{incident_id}/model_logs.jsonl", "r") as f:
        model_outputs = [json.loads(line) for line in f]

    # Expert annotates each recommendation
    for output in model_outputs:
        print(f"\nQuery: {output['input']}")
        print(f"Model recommended: {output['output']}")

        expert_rating = input("Rating (1-5): ")
        what_actually_happened = input("What actually happened: ")
        should_have_recommended = input("What should model have recommended: ")

        # Save for retraining
        if int(expert_rating) <= 3:
            # Model was wrong, add correction to training data
            save_correction(
                input=output['input'],
                wrong_output=output['output'],
                correct_output=should_have_recommended,
                context=what_actually_happened
            )

# Real incidents are the BEST training data source
# Even 5-10 real incident reviews > 1000 synthetic examples
```

---

## Incremental Training Strategies

### Strategy 1: Continued Fine-Tuning (Recommended)

**Approach**: Continue training existing LoRA adapter on new data

**When to use**: Quarterly updates with 200-500 new examples

**Implementation**:

```yaml
# configs/continual_learning_v1.1.yml

# Start from previous LoRA adapter
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
adapter: lora
load_in_8bit: true

# Load previous adapter weights
resume_from_checkpoint: ./outputs/firefighter-v1.0-lora

# New training data only (incremental)
datasets:
  - path: ../data_collection/continual_learning_data_q1_2025.jsonl
    type: chat_template

# Conservative hyperparameters (avoid forgetting)
learning_rate: 1e-5  # Lower than initial training (was 2e-4)
num_epochs: 1        # Fewer epochs
lora_dropout: 0.1    # Higher dropout for regularization

# Mix with general data to prevent forgetting
# (20% general, 80% new domain data)
```

**Train**:
```bash
accelerate launch -m axolotl.cli.train configs/continual_learning_v1.1.yml
```

**Advantages**:
- Fast (1-2 hours)
- Preserves previous knowledge
- Low compute cost

**Disadvantages**:
- May not fully adapt to major distribution shifts
- Accumulated adapters can grow large over time

---

### Strategy 2: Merge and Retrain (Periodic Reset)

**Approach**: Merge LoRA into base model, then train new LoRA from scratch on ALL data

**When to use**: Annual updates or after 3-4 incremental updates

**Implementation**:

```bash
# Step 1: Merge previous LoRA into base model
python -m axolotl.cli.merge_lora \
  --config configs/firefighter-v1.4-lora.yml \
  --lora_model_dir ./outputs/firefighter-v1.4-lora \
  --output_dir ./merged_models/firefighter-v1.4-merged

# Step 2: Use merged model as new base
# configs/annual_retrain_v2.0.yml
base_model: ./merged_models/firefighter-v1.4-merged  # Previous merged model

# Train NEW LoRA on COMPLETE dataset (original + all incremental)
datasets:
  - path: ../data_collection/complete_dataset_2025.jsonl  # All data
    type: chat_template

# Standard training hyperparameters
learning_rate: 2e-4
num_epochs: 3
```

**Advantages**:
- Fresh start, no accumulated drift
- Can leverage better base model (e.g., Llama 4.0)
- Clean architecture

**Disadvantages**:
- Slower (6-12 hours)
- Higher compute cost
- Need to maintain complete dataset

---

### Strategy 3: Knowledge Distillation (Advanced)

**Approach**: Train new model to mimic both old model (preserve knowledge) and new data (add knowledge)

```python
# Advanced: Distill from v1.0 to v1.1
def distillation_loss(student_model, teacher_model, batch):
    """Combine standard loss with distillation from old model."""

    # Standard loss on new data
    student_output = student_model(**batch)
    standard_loss = student_output.loss

    # Distillation loss: Match teacher predictions on old data
    with torch.no_grad():
        teacher_logits = teacher_model(**batch).logits

    student_logits = student_output.logits

    # KL divergence between student and teacher
    distill_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Combine losses
    total_loss = 0.7 * standard_loss + 0.3 * distill_loss

    return total_loss
```

**Use when**: Major model upgrade but must preserve exact behavior on critical queries

---

## Regression Testing

**Problem**: New model version might break capabilities that worked in v1.0

**Solution**: Maintain regression test suite

### Create Regression Test Suite

```python
# tools/regression_tests.py

import json

class RegressionTestSuite:
    """Test new model version against critical test cases."""

    def __init__(self, test_suite_path="regression_tests.jsonl"):
        self.test_suite_path = test_suite_path
        self.tests = []

    def add_critical_test(self, input, expected_output, test_id, criticality="high"):
        """Add test case that new model MUST pass."""

        test = {
            "test_id": test_id,
            "input": input,
            "expected_output": expected_output,
            "criticality": criticality,  # "critical", "high", "medium", "low"
            "tags": []
        }

        self.tests.append(test)

        with open(self.test_suite_path, "a") as f:
            f.write(json.dumps(test) + "\n")

    def run_regression_tests(self, model_v1, model_v2):
        """Compare v1 and v2 on regression test suite."""

        results = {
            "v1_pass": 0,
            "v2_pass": 0,
            "v1_fail": 0,
            "v2_fail": 0,
            "regressions": []  # v1 passed but v2 failed
        }

        for test in self.tests:
            # Test v1
            v1_output = model_v1.generate(test["input"])
            v1_correct = self._check_correctness(v1_output, test["expected_output"])

            # Test v2
            v2_output = model_v2.generate(test["input"])
            v2_correct = self._check_correctness(v2_output, test["expected_output"])

            if v1_correct:
                results["v1_pass"] += 1
            else:
                results["v1_fail"] += 1

            if v2_correct:
                results["v2_pass"] += 1
            else:
                results["v2_fail"] += 1

            # Regression detected: v1 passed, v2 failed
            if v1_correct and not v2_correct:
                results["regressions"].append({
                    "test_id": test["test_id"],
                    "input": test["input"],
                    "expected": test["expected_output"],
                    "v1_output": v1_output,
                    "v2_output": v2_output,
                    "criticality": test["criticality"]
                })

        return results

    def _check_correctness(self, output, expected):
        """Check if output matches expected (fuzzy matching)."""
        # Simple implementation: key facts present
        # Advanced: Use LLM-as-judge
        return expected.lower() in output.lower()

# Build regression test suite from production successes
suite = RegressionTestSuite()

# Add critical test cases
suite.add_critical_test(
    input="What is the IDLH for ammonia?",
    expected_output="300 ppm",
    test_id="hazmat_001",
    criticality="critical"
)

suite.add_critical_test(
    input="Wind is 40 km/h toward the village. Wildfire 2km away. Evacuate?",
    expected_output="immediate evacuation",
    test_id="wildfire_001",
    criticality="critical"
)

# ... add 100-200 critical tests

# Before deploying v1.1, run regression tests
results = suite.run_regression_tests(model_v1_0, model_v1_1)

print(f"Regression test results:")
print(f"  v1.0: {results['v1_pass']}/{len(suite.tests)} passed")
print(f"  v1.1: {results['v2_pass']}/{len(suite.tests)} passed")
print(f"  Regressions detected: {len(results['regressions'])}")

if len(results['regressions']) > 0:
    print("\n⚠️ WARNING: Regressions detected!")
    for reg in results['regressions']:
        if reg['criticality'] == 'critical':
            print(f"  CRITICAL: {reg['test_id']}")

# Decision: Only deploy if zero critical regressions
if any(r['criticality'] == 'critical' for r in results['regressions']):
    print("\n✗ DO NOT DEPLOY v1.1 (critical regressions)")
else:
    print("\n✓ Safe to deploy v1.1")
```

---

## Model Versioning

### Semantic Versioning for Models

**Format**: `MAJOR.MINOR.PATCH`

**MAJOR** (1.0 → 2.0): Breaking changes
- New base model (Llama 3.1 → Llama 4.0)
- Complete retraining on new data distribution
- Changed output format

**MINOR** (1.0 → 1.1): New features, improvements
- Added new domain knowledge (new HAZMAT protocols)
- Significant accuracy improvement (+10%)
- Quarterly updates with new training data

**PATCH** (1.0 → 1.0.1): Bug fixes
- Fixed hallucination on specific query
- Corrected factual error
- Safety patch

### Track Model Lineage

```python
# models/model_registry.json

{
  "models": [
    {
      "version": "1.0.0",
      "release_date": "2025-01-15",
      "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
      "training_data": "firefighter_dataset_v1.0.jsonl (2,847 examples)",
      "training_duration_hours": 11.2,
      "accuracy": 81.7,
      "notes": "Initial release"
    },
    {
      "version": "1.1.0",
      "release_date": "2025-04-15",
      "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
      "training_data": "firefighter_dataset_v1.1.jsonl (3,214 examples)",
      "parent_model": "1.0.0",
      "training_type": "incremental (continued LoRA)",
      "training_duration_hours": 2.3,
      "accuracy": 84.2,
      "improvements": [
        "Added 367 examples from Q1 2025 production feedback",
        "Fixed ammonia IDLH hallucination",
        "Added new wildfire evacuation protocols"
      ],
      "regressions": 0
    }
  ]
}
```

---

## Update Schedule and Cost Planning

### Quarterly Update Budget

**Q1 2025 Update (Example)**:

| Task | Hours | Cost |
|------|-------|------|
| **Data Collection** | 16h | $2,400 |
| - Review 200 production logs | 4h | $600 |
| - Expert annotation (50 samples) | 6h | $900 |
| - AAR from 2 real incidents | 4h | $600 |
| - Synthetic data generation | 2h | $300 |
| **Model Training** | 4h | $200 |
| - Prepare dataset | 1h | $50 |
| - Configure training | 1h | $50 |
| - Run training (RTX 4090) | 2h | $100 |
| **Evaluation** | 8h | $1,200 |
| - Regression tests | 2h | $300 |
| - Expert evaluation (100 samples) | 4h | $600 |
| - A/B testing setup | 2h | $300 |
| **Deployment** | 4h | $600 |
| - Shadow deployment | 1h | $150 |
| - Monitor for 1 week | 2h | $300 |
| - Production rollout | 1h | $150 |
| **TOTAL** | **32h** | **$4,400** |

**Annual Cost**: 4 quarters × $4,400 = **$17,600/year**

**ROI**: Maintains 80%+ accuracy (vs 60% without updates), prevents safety incidents, user trust

---

## Decision Framework: To Update or Not?

### Calculate Update Value

```python
def should_update_model(current_metrics, new_data_available):
    """Decide if model update is worthwhile."""

    # 1. Performance degradation
    accuracy_drop = 82.0 - current_metrics["accuracy"]  # Was 82%, now?

    # 2. New data quality
    high_quality_examples = sum(1 for ex in new_data_available if ex["expert_verified"])

    # 3. Cost of update
    update_cost_usd = 4400  # Quarterly update

    # 4. Cost of NOT updating (estimated from degradation)
    # Assume: 1% accuracy drop = $1000/quarter in incident response errors
    cost_of_degradation = accuracy_drop * 1000

    # Decision
    if accuracy_drop > 5.0:
        print(f"✓ UPDATE REQUIRED: Accuracy dropped {accuracy_drop:.1f}%")
        print(f"   Cost of degradation: ${cost_of_degradation:.0f} > Update cost: ${update_cost_usd}")
        return True

    elif high_quality_examples >= 200:
        print(f"✓ UPDATE RECOMMENDED: {high_quality_examples} new examples available")
        print(f"   Expected improvement: +2-5% accuracy")
        return True

    else:
        print(f"✗ UPDATE NOT NEEDED YET")
        print(f"   Accuracy drop: {accuracy_drop:.1f}% (threshold: 5%)")
        print(f"   New data: {high_quality_examples} examples (threshold: 200)")
        return False

# Usage
should_update = should_update_model(
    current_metrics={"accuracy": 79.2, "safety_failures": 0.03},
    new_data_available=load_new_data()
)
```

---

## Complete Continual Learning Workflow

### Month 1: Data Collection
- Week 1-4: Collect production feedback (target: 200 examples)
- Week 2, 4: Expert annotation sessions (50 examples each)
- Ongoing: Log all predictions for later review

### Month 2: Model Training
- Week 1: Prepare dataset (merge production + annotated + any AARs)
- Week 2: Train v1.1 using continued fine-tuning
- Week 3: Run regression tests, expert evaluation
- Week 4: A/B testing (shadow deployment)

### Month 3: Deployment & Monitoring
- Week 1-2: Canary deployment (1% → 10% → 50%)
- Week 3: Full deployment if metrics good
- Week 4: Monitor, collect feedback for next quarter

**Repeat every quarter**

---

## Checklist: Before Deploying Updated Model

- [ ] **New training data collected** (minimum 200 examples)
- [ ] **Data versioned** (SHA-256 hash, provenance tracked)
- [ ] **Model trained** with continued fine-tuning or full retrain
- [ ] **Regression tests passed** (zero critical regressions)
- [ ] **Accuracy maintained or improved** (≥v1.0 accuracy)
- [ ] **Safety metrics stable** (no increase in failure rate)
- [ ] **Expert validation** (4.0+ rating on sample)
- [ ] **A/B testing completed** (shadow → canary → 50%)
- [ ] **Model versioned** (semantic versioning, lineage tracked)
- [ ] **Rollback plan tested** (can revert to v1.0 immediately)
- [ ] **Documentation updated** (changelog, known issues)

---

## Tools and Automation

### Automated Feedback Pipeline

```bash
# cron job: Every Monday at 9am
0 9 * * 1 python tools/weekly_feedback_extraction.py

# tools/weekly_feedback_extraction.py
def weekly_feedback_extraction():
    collector = FeedbackCollector()

    # Extract high-quality examples from last week
    new_examples = collector.extract_training_data(min_rating=4)

    # Check if ready for update
    if len(new_examples) >= 200:
        send_alert("Ready for quarterly model update: 200+ examples collected")

weekly_feedback_extraction()
```

---

## Key Takeaways

1. **Plan for continual learning from day one** - Not an afterthought
2. **Quarterly updates are realistic** for most teams (32 hours, $4,400)
3. **Production feedback is gold** - Better than synthetic data
4. **Regression testing is mandatory** - Prevent breaking critical capabilities
5. **Version everything** - Models, datasets, configs (reproducibility)
6. **Incremental training works** - Don't need to retrain from scratch every time
7. **Cost of NOT updating** often exceeds cost of updating

---

**Generated**: 2025-11-13
**Version**: 1.0
