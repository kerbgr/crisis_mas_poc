# Data Collection Methodology for Domain-Specific LLM Training

## Overview

High-quality training data is the **most critical factor** for successful domain-specific LLM fine-tuning. This guide provides a systematic methodology for collecting, curating, and formatting expert knowledge from emergency response professionals.

**Goal**: Collect 1,000-10,000 high-quality examples that capture:
- Domain terminology and concepts
- Decision-making patterns
- Standard operating procedures (SOPs)
- Real-world scenarios and case studies
- Safety protocols and regulations

---

## Data Quality Principles

### The 3 Pillars of Quality Training Data

1. **Accuracy** - Information must be factually correct and verified by experts
2. **Diversity** - Cover wide range of scenarios, difficulty levels, and decision types
3. **Relevance** - Focus on knowledge that improves agent performance in crisis scenarios

### Quality over Quantity

- **100 excellent examples > 1,000 mediocre examples**
- Each example should teach the model something specific
- Avoid redundancy and near-duplicates
- Prioritize complex reasoning over simple facts

---

## Data Collection Methods

### Method 1: Expert Interviews (Highest Quality)

**Estimated Output**: 50-200 Q&A pairs per 2-hour interview

**Process**:
1. Schedule 2-hour session with domain expert (firefighter, police officer, paramedic)
2. Record audio (with permission) + take notes
3. Ask scenario-based questions (see `expert_interview_guide.md`)
4. Follow up on interesting responses with "why?" and "how?"
5. Transcribe and convert to Q&A format

**Example Questions for Firefighter**:
```
Q: "A wildfire is approaching a village with 500 residents. Wind speed is 40 km/h,
temperature is 38°C, and you have 2 fire trucks and 8 firefighters. The nearest backup
is 45 minutes away. What do you do?"

A: "In this scenario, with limited resources and high wind speed, evacuation takes
priority over fire suppression. I would immediately: 1) Order full evacuation using
both trucks for transport, prioritizing elderly and children. 2) Set up a defensive
perimeter at the village edge if time permits. 3) Request air support (Canadair) and
additional ground units. 4) Identify safe evacuation routes away from wind direction.
5) Establish a rally point 5km upwind. The 40 km/h wind makes direct attack too
dangerous with only 8 personnel..."
```

**Advantages**:
- Captures expert reasoning and decision-making patterns
- Natural language, realistic scenarios
- Can probe for edge cases and rare situations

**Time Investment**: 3-4 hours per session (including transcription)

---

### Method 2: Standard Operating Procedures (SOPs)

**Estimated Output**: 100-500 Q&A pairs per SOP document

**Process**:
1. Obtain official SOP documents from agencies:
   - Hellenic Fire Corps operational manuals
   - ELAS (Greek Police) tactical procedures
   - EKAB medical protocols
   - Civil Protection coordination guidelines
2. Extract procedural knowledge
3. Convert to instruction format: "How do I do X?" → "Follow these steps..."

**Example from EKAB Triage Protocol**:
```
Q: "Walk me through the START triage method for a mass casualty incident with 30+ victims."

A: "The START (Simple Triage and Rapid Treatment) method classifies victims in 60 seconds or less:

**RED (Immediate)**:
- Respirations: >30/min after airway opening
- Perfusion: Capillary refill >2 seconds
- Mental status: Unable to follow simple commands

**YELLOW (Delayed)**:
- Respirations: <30/min
- Perfusion: Good (cap refill <2s)
- Mental status: Alert but with injuries

**GREEN (Minor)**:
- Walking wounded
- Can follow commands

**BLACK (Deceased/Expectant)**:
- No respirations after airway positioning

Process: 1) Assess respirations, 2) Check perfusion, 3) Evaluate mental status. Tag and move to next victim in <60 seconds."
```

**Advantages**:
- Authoritative, verified information
- Comprehensive coverage of standard procedures
- Reduces liability (official protocols)

**Challenges**:
- SOPs may be confidential or restricted
- Need permission from agencies
- Dry, procedural language (less natural)

---

### Method 3: Incident After-Action Reports (AARs)

**Estimated Output**: 20-100 Q&A pairs per AAR

**Process**:
1. Obtain anonymized incident reports from:
   - Fire department incident reviews
   - Police after-action reports
   - Hospital mass casualty debriefs
2. Extract lessons learned, decisions made, outcomes
3. Convert to "What would you do in X situation?" format

**Example from 2021 Evia Fire AAR**:
```
Q: "During the 2021 Evia wildfire, some villages were successfully defended while others
were evacuated. What factors determine whether to defend or evacuate?"

A: "The decision to defend versus evacuate depends on:

**Defend if**:
- Adequate firefighting resources available (3:1 responder-to-structure ratio)
- Defensible space around structures (30m+ clearance)
- Water supply accessible (hydrants, swimming pools, tankers)
- Fire intensity manageable (<4,000 kW/m flame length)
- Escape routes remain open

**Evacuate if**:
- Insufficient resources or extreme fire behavior
- Structures lack defensible space
- Residents unprepared (no evacuation plan)
- Smoke inhalation risk high
- Escape routes threatened

In 2021 Evia, Limni was defended (coastal access, good water supply) while Gouves was
evacuated early (surrounded by dense forest, single-lane access road). This proactive
evacuation saved lives when the fire crowned unexpectedly."
```

**Advantages**:
- Real-world scenarios with actual outcomes
- Captures complexity and uncertainty
- Lessons learned from mistakes

**Challenges**:
- Sensitive information (requires anonymization)
- May be difficult to access
- Limited availability

---

### Method 4: Training Manuals and Textbooks

**Estimated Output**: 200-1,000 Q&A pairs per textbook

**Process**:
1. Identify authoritative texts:
   - Fire Engineering textbooks
   - Police academy training materials
   - Emergency medicine textbooks (e.g., Tintinalli's)
2. Extract key concepts, definitions, procedures
3. Convert to Q&A or explanation format

**Example from Fire Behavior Textbook**:
```
Q: "Explain the fire triangle and how it applies to wildfire suppression."

A: "The fire triangle consists of three elements required for combustion:

1. **Fuel**: Vegetation (grass, shrubs, trees)
2. **Oxygen**: Atmosphere (21% O2)
3. **Heat**: Ignition source (lightning, embers, human activity)

**Suppression strategies target one or more elements**:

- **Remove fuel**: Create firebreaks, controlled burns, vegetation clearing
- **Remove oxygen**: Smother with foam or dirt (limited in wildfire context)
- **Remove heat**: Water or retardant to cool below ignition temperature (most common)

In practice, wildfire suppression combines fuel removal (bulldozers creating breaks)
and heat removal (aerial water drops). The most effective tactic is often removing
fuel ahead of the fire rather than directly attacking the flame front, especially
in extreme fire weather."
```

**Advantages**:
- Comprehensive foundational knowledge
- Well-organized, systematic coverage
- No confidentiality concerns

**Challenges**:
- May lack practical context
- Generic (not specific to Greek operations)

---

### Method 5: Simulated Scenarios (AI-Assisted)

**Estimated Output**: Unlimited (with human review)

**Process**:
1. Use GPT-4 or Claude to generate scenario-based questions
2. Have domain experts answer or review/edit AI-generated answers
3. Validate for accuracy and realism

**Prompt Example**:
```
Generate 10 realistic emergency scenarios for a Greek fire commander responding to:
- Structure fires in urban Athens
- Wildfires in Mediterranean climate
- HAZMAT incidents at industrial sites
- Vehicle extrication on highways

For each scenario, include: location, resources available, complicating factors,
and time pressure. Format as questions requiring tactical decisions.
```

**Advantages**:
- Scalable data generation
- Can target specific gaps in dataset
- Fast iteration

**Challenges**:
- Requires expert validation (AI can hallucinate)
- May generate unrealistic scenarios
- Risk of biasing model toward AI-generated patterns

**Recommendation**: Use for initial dataset creation, then validate with 2+ real experts

---

## Data Formats

### Instruction-Tuning Format (Recommended)

Most modern LLMs expect data in **instruction format** with three components:

1. **Instruction** (or System): Task description or role
2. **Input** (or User): The question or scenario
3. **Output** (or Assistant): The expert response

#### ChatML Format (OpenAI, Mistral, many open models)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an experienced Greek fire commander (Pyragos) with 15 years of experience in wildfire suppression and urban search and rescue."
    },
    {
      "role": "user",
      "content": "A wildfire is threatening a village of 500 people. Wind is 40 km/h, temperature 38°C. I have 2 trucks and 8 firefighters. Backup arrives in 45 min. What do I do?"
    },
    {
      "role": "assistant",
      "content": "Immediate evacuation is the priority. With only 8 personnel and high wind speed (40 km/h), direct fire attack is too dangerous. Actions: 1) Order full evacuation using both trucks, prioritize elderly and children. 2) Request Canadair air support immediately. 3) Establish safe rally point 5km upwind from village. 4) If time permits before evacuation, create minimal defensive perimeter at village edge. 5) Ensure all escape routes monitored. Do NOT attempt to fight the fire directly with insufficient resources - life safety first."
    }
  ]
}
```

#### Alpaca Format (Meta Llama, many LoRA tutorials)

```json
{
  "instruction": "You are an experienced Greek fire commander. A wildfire is threatening a village of 500 people. Wind is 40 km/h, temperature 38°C. You have 2 trucks and 8 firefighters. Backup arrives in 45 min.",
  "input": "",
  "output": "Immediate evacuation is the priority. With only 8 personnel and high wind speed (40 km/h), direct fire attack is too dangerous..."
}
```

#### ShareGPT Format (Used by many datasets)

```json
{
  "conversations": [
    {"from": "system", "value": "You are a Greek fire commander..."},
    {"from": "human", "value": "A wildfire is threatening a village..."},
    {"from": "gpt", "value": "Immediate evacuation is the priority..."}
  ]
}
```

**Tool Provided**: Use `../tools/data_formatter.py` to convert between formats

---

## Dataset Structure

### Recommended Split

- **Training set**: 80% (e.g., 8,000 examples)
- **Validation set**: 10% (e.g., 1,000 examples)
- **Test set**: 10% (e.g., 1,000 examples)

**Important**: Test set should contain held-out scenarios NEVER seen during training

### Coverage Checklist

Ensure your dataset covers:

#### For Firefighter LLM:
- ✅ Structure fires (residential, commercial, high-rise)
- ✅ Wildfires (grass, brush, forest, interface)
- ✅ Vehicle fires and extrication
- ✅ HAZMAT incidents (chemicals, gas leaks)
- ✅ Search and rescue (collapsed structures, confined spaces)
- ✅ Incident command and resource coordination
- ✅ Safety protocols and risk assessment

#### For Police LLM:
- ✅ Crowd control and riot management
- ✅ Evacuation coordination
- ✅ Traffic management during emergencies
- ✅ Security perimeters and access control
- ✅ Victim assistance and family reunification
- ✅ Multi-agency coordination with fire/EMS
- ✅ Evidence preservation at disaster scenes

#### For Medical LLM (EKAB):
- ✅ Triage (START, SALT, JumpSTART for pediatrics)
- ✅ Mass casualty incidents (MCIs)
- ✅ Toxicology and HAZMAT medical response
- ✅ Trauma care and stabilization
- ✅ Burn treatment
- ✅ Heat injuries and dehydration
- ✅ Hospital coordination and patient transport

---

## Quality Control

### Validation Checklist

Before adding an example to your dataset, verify:

- [ ] **Factually accurate** (cross-check with 2+ sources or experts)
- [ ] **Complete answer** (not truncated or vague)
- [ ] **Realistic scenario** (based on real conditions, not hypothetical extremes)
- [ ] **Appropriate detail level** (not too simple, not excessively technical)
- [ ] **Free of sensitive info** (no names, locations of ongoing ops, classified tactics)
- [ ] **Grammatically correct** (especially if non-native English/Greek)
- [ ] **Consistent terminology** (use standard terms throughout dataset)

### Common Issues to Avoid

1. **Hallucinated tactics**: AI-generated data with fake procedures
2. **Contradictory information**: Different experts saying opposite things
3. **Outdated protocols**: SOPs that have changed
4. **Overly simplistic**: "Just call 112" without explaining decision-making
5. **Dangerous advice**: Incorrect safety information
6. **Bias**: All scenarios assume unlimited resources or perfect conditions

### Expert Review Process

1. **Self-review**: Creator checks for obvious errors
2. **Peer review**: Another team member validates
3. **Expert review**: Domain expert (firefighter/police/medic) approves subset (10-20%)
4. **Test evaluation**: Model trained on data is tested by experts

---

## Ethical and Legal Considerations

### Anonymization

Remove or anonymize:
- Personal names (victims, responders)
- Specific addresses (use "residential neighborhood" instead of "123 Main St")
- Dates of specific incidents (use "summer 2021" instead of "August 5, 2021")
- Identifying details (license plates, case numbers)

### Permissions

Obtain written permission to use:
- Interview transcripts
- Incident reports
- Training materials
- Photographs or diagrams

### Sensitive Information

DO NOT include:
- Classified tactical procedures
- Vulnerabilities in infrastructure
- Ongoing investigation details
- Personal medical information
- Security weaknesses

---

## Data Versioning and Provenance Tracking

### Why Data Versioning Matters

**Critical for**:
- **Reproducibility**: Re-train exact same model later
- **Regulatory compliance**: EU AI Act requires data provenance
- **Debugging**: Identify which data caused model issues
- **Auditing**: Trace model behavior to training data source
- **Liability protection**: Prove data quality in legal disputes

**Problem without versioning**:
```
Training run #1: accuracy 82%
Training run #2 (one month later): accuracy 75%

Question: What changed?
Answer: Unknown - no way to reproduce training run #1
```

**Solution**: Version control for datasets (like Git for code)

---

### Dataset Versioning System

**Implementation**:

```python
# tools/data_versioning.py

import hashlib
import json
from datetime import datetime
from pathlib import Path

class DatasetVersion:
    """Track dataset versions with cryptographic hashing."""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.metadata = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "dataset_hash": self._compute_hash(),
            "num_examples": self._count_examples(),
            "sources": [],
            "contributors": [],
            "validation_status": "pending",
            "splits": {}
        }

    def _compute_hash(self):
        """Compute SHA-256 hash of entire dataset."""
        hasher = hashlib.sha256()

        with open(self.dataset_path, 'rb') as f:
            # Read in chunks for large files
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()

    def _count_examples(self):
        """Count number of examples in dataset."""
        count = 0
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def log_source(self, source_type, source_id, expert_id, expert_credentials, date):
        """Track data provenance (where data came from)."""
        self.metadata["sources"].append({
            "type": source_type,  # "interview", "sop", "aar", "manual", "ai_generated"
            "id": source_id,
            "expert": {
                "id": expert_id,
                "name": "REDACTED",  # Don't store PII in logs
                "credentials": expert_credentials,  # "Pyragos, 15 years experience"
                "verified": True
            },
            "collection_date": date,
            "timestamp": datetime.now().isoformat()
        })

    def log_validation(self, validator_id, quality_score, issues_found):
        """Track validation results."""
        self.metadata["validation"] = {
            "validator_id": validator_id,
            "quality_score": quality_score,  # 1-5
            "issues_found": issues_found,
            "timestamp": datetime.now().isoformat(),
            "status": "approved" if quality_score >= 4 else "needs_revision"
        }

    def create_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        """Create reproducible train/val/test splits."""
        import random
        random.seed(seed)

        # Load all examples
        with open(self.dataset_path, 'r') as f:
            examples = [line.strip() for line in f if line.strip()]

        # Shuffle with fixed seed
        random.shuffle(examples)

        # Calculate split sizes
        n = len(examples)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        # Create splits
        train = examples[:train_size]
        val = examples[train_size:train_size + val_size]
        test = examples[train_size + val_size:]

        # Save splits
        split_dir = self.dataset_path.parent / "splits" / self.metadata["version"]
        split_dir.mkdir(parents=True, exist_ok=True)

        self._save_split(split_dir / "train.jsonl", train)
        self._save_split(split_dir / "val.jsonl", val)
        self._save_split(split_dir / "test.jsonl", test)

        # Record split hashes
        self.metadata["splits"] = {
            "train": {"size": len(train), "hash": self._hash_list(train)},
            "val": {"size": len(val), "hash": self._hash_list(val)},
            "test": {"size": len(test), "hash": self._hash_list(test)},
            "seed": seed
        }

    def _save_split(self, path, examples):
        """Save split to file."""
        with open(path, 'w') as f:
            for example in examples:
                f.write(example + '\n')

    def _hash_list(self, items):
        """Hash a list of strings."""
        hasher = hashlib.sha256()
        for item in items:
            hasher.update(item.encode('utf-8'))
        return hasher.hexdigest()

    def save_metadata(self):
        """Save metadata to JSON."""
        metadata_path = self.dataset_path.parent / f"metadata_v{self.metadata['version']}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✅ Metadata saved: {metadata_path}")
        print(f"📊 Dataset hash: {self.metadata['dataset_hash']}")
        print(f"📝 Examples: {self.metadata['num_examples']}")

    def verify_integrity(self):
        """Verify dataset hasn't been corrupted."""
        current_hash = self._compute_hash()

        if current_hash == self.metadata["dataset_hash"]:
            print("✅ Dataset integrity verified (hash matches)")
            return True
        else:
            print("❌ WARNING: Dataset hash mismatch!")
            print(f"   Expected: {self.metadata['dataset_hash']}")
            print(f"   Actual:   {current_hash}")
            return False
```

**Usage**:

```python
# Create dataset version
dataset = DatasetVersion("./firefighter_train.jsonl")

# Log data sources
dataset.log_source(
    source_type="interview",
    source_id="INT-001",
    expert_id="EXP-042",
    expert_credentials="Pyragos, Hellenic Fire Corps, 15 years",
    date="2024-10-15"
)

dataset.log_source(
    source_type="sop",
    source_id="SOP-HFC-2024",
    expert_id="Official",
    expert_credentials="Hellenic Fire Corps Official Manual",
    date="2024-01-01"
)

# Log validation
dataset.log_validation(
    validator_id="VAL-001",
    quality_score=4.5,
    issues_found=["2 typos corrected", "1 factual error fixed"]
)

# Create reproducible splits
dataset.create_splits(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

# Save metadata
dataset.save_metadata()

# Later: verify integrity
dataset.verify_integrity()
```

---

### Semantic Versioning for Datasets

Follow **SemVer** (major.minor.patch):

**MAJOR.MINOR.PATCH**

- **MAJOR**: Breaking changes (incompatible format, major content changes)
  - Example: `1.0.0` → `2.0.0` (changed from Alpaca to ChatML format)

- **MINOR**: New features (added new examples, new categories)
  - Example: `1.2.0` → `1.3.0` (added 500 HAZMAT examples)

- **PATCH**: Bug fixes (corrected errors, typos)
  - Example: `1.2.1` → `1.2.2` (fixed 10 factual errors)

**Example Version History**:
```
v1.0.0 - Initial release (2,000 examples, firefighting only)
v1.1.0 - Added police operations (500 examples)
v1.1.1 - Fixed IDLH value errors (12 corrections)
v1.2.0 - Added HAZMAT scenarios (300 examples)
v2.0.0 - Migrated to ChatML format (breaking change)
v2.1.0 - Added Greek language examples (200 examples)
```

---

### Provenance Tracking

**Track every example's journey**:

```json
{
  "example_id": "FIR-001-042",
  "content": {
    "question": "What is the IDLH for ammonia?",
    "answer": "The IDLH for ammonia (NH3) is 300 ppm..."
  },
  "provenance": {
    "source_type": "expert_interview",
    "source_id": "INT-005",
    "expert": {
      "id": "EXP-042",
      "credentials": "Pyragos, 15 years, HAZMAT certified",
      "verification": "Verified by 2 additional experts"
    },
    "collection_date": "2024-10-20",
    "validation": {
      "validator_1": {"approved": true, "score": 5},
      "validator_2": {"approved": true, "score": 4}
    },
    "modifications": [
      {
        "date": "2024-10-25",
        "type": "typo_fix",
        "description": "Fixed 'amonia' → 'ammonia'"
      }
    ],
    "inclusion_status": "included_in_training"
  }
}
```

---

### Regulatory Compliance

**EU AI Act Article 10 (Data Governance)** requires:

✅ **Data provenance**: Know where each example came from
✅ **Quality checks**: Documented validation process
✅ **Expert verification**: Credentials of contributors
✅ **Error tracking**: Log all corrections and modifications
✅ **Bias audits**: Test for demographic/geographic bias

**Implementation**:

```python
# Compliance report
def generate_compliance_report(dataset):
    """Generate EU AI Act Article 10 compliance report."""

    report = {
        "dataset_id": dataset.metadata["dataset_hash"][:16],
        "version": dataset.metadata["version"],
        "total_examples": dataset.metadata["num_examples"],

        "data_sources": {
            "expert_interviews": sum(1 for s in dataset.metadata["sources"] if s["type"] == "interview"),
            "official_sops": sum(1 for s in dataset.metadata["sources"] if s["type"] == "sop"),
            "after_action_reports": sum(1 for s in dataset.metadata["sources"] if s["type"] == "aar"),
            "validated_ai_generated": sum(1 for s in dataset.metadata["sources"] if s["type"] == "ai_generated")
        },

        "expert_verification": {
            "unique_experts": len(set(s["expert"]["id"] for s in dataset.metadata["sources"])),
            "all_verified": all(s["expert"]["verified"] for s in dataset.metadata["sources"]),
            "credentials_documented": True
        },

        "quality_assurance": {
            "validation_score": dataset.metadata.get("validation", {}).get("quality_score", 0),
            "issues_resolved": len(dataset.metadata.get("validation", {}).get("issues_found", [])),
            "approved": dataset.metadata.get("validation", {}).get("status") == "approved"
        },

        "bias_audit": {
            "conducted": "bias_audit" in dataset.metadata,
            "results": dataset.metadata.get("bias_audit", {})
        },

        "traceability": {
            "reproducible_splits": "splits" in dataset.metadata,
            "seed_documented": dataset.metadata.get("splits", {}).get("seed"),
            "hash_verified": dataset.verify_integrity()
        }
    }

    return report
```

---

### Best Practices

1. **Version Every Release**:
   ```bash
   # Before training
   python tools/data_versioning.py --dataset firefighter_train.jsonl --version 2.1.0
   ```

2. **Tag in Git**:
   ```bash
   git tag dataset-v2.1.0
   git push --tags
   ```

3. **Document Changes**:
   ```markdown
   # CHANGELOG.md

   ## [2.1.0] - 2024-11-15
   ### Added
   - 200 Greek language examples
   - 50 HAZMAT ammonia leak scenarios

   ### Fixed
   - Corrected IDLH values for 3 chemicals
   - Fixed typos in 12 examples
   ```

4. **Archive Old Versions**:
   ```bash
   # Keep copies of all versions
   datasets/
   ├── v1.0.0/
   │   ├── firefighter_train.jsonl
   │   └── metadata_v1.0.0.json
   ├── v2.0.0/
   │   ├── firefighter_train.jsonl
   │   └── metadata_v2.0.0.json
   └── v2.1.0/  # Current
       ├── firefighter_train.jsonl
       └── metadata_v2.1.0.json
   ```

5. **Link Dataset to Model**:
   ```python
   # model_metadata.json
   {
       "model_version": "2.1.0",
       "training_dataset": {
           "version": "2.1.0",
           "hash": "a3f5b9c...",
           "path": "datasets/v2.1.0/firefighter_train.jsonl"
       }
   }
   ```

---

## Dataset Templates

See the `dataset_templates/` folder for starter examples:

- `firefighter_qa.jsonl` - 50 example Q&A pairs for fire operations
- `police_qa.jsonl` - 50 example Q&A pairs for police coordination
- `medical_qa.jsonl` - 50 example Q&A pairs for emergency medicine

These can be used as:
1. **Templates**: Follow the format and style for your own data
2. **Seed data**: Start training with these, then expand
3. **Reference**: See quality expectations

---

## Tools and Scripts

### data_formatter.py

Convert raw text to training format:

```bash
python ../tools/data_formatter.py \
  --input raw_interviews.txt \
  --output formatted_dataset.jsonl \
  --format chatml \
  --system_prompt "You are an experienced Greek fire commander..."
```

### dataset_validator.py

Check dataset for common issues:

```bash
python ../tools/dataset_validator.py \
  --dataset firefighter_train.jsonl \
  --check_duplicates \
  --check_length \
  --check_format
```

---

## Estimated Timeline

| Task | Time Required |
|------|---------------|
| Expert interviews (5 sessions) | 20 hours |
| SOP extraction and formatting | 40 hours |
| AAR analysis and conversion | 20 hours |
| AI-assisted generation + review | 30 hours |
| Quality control and validation | 20 hours |
| **Total** | **130 hours (3-4 weeks)** |

---

## Success Criteria

You have a good dataset when:
- ✅ 1,000+ high-quality examples (minimum for LoRA)
- ✅ Covers all major scenario types
- ✅ Validated by 2+ domain experts
- ✅ Diverse difficulty levels (basic to complex)
- ✅ Consistent formatting (no parse errors)
- ✅ Train/val/test splits properly separated
- ✅ Zero sensitive/classified information

---

## Next Steps

After completing data collection:
1. Review `../fine_tuning/README.md` for training instructions
2. Start with a small pilot (100-200 examples) to validate approach
3. Iterate based on initial evaluation results
4. Scale up to full dataset

---

**Generated**: 2025-11-13
**Version**: 1.0
