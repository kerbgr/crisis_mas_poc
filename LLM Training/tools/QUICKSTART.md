# Quick Start Guide - LLM Training Tools

Get started with the LLM Training Tools in 5 minutes!

## Step 1: Extract Code from Methodology (1 minute)

All code is embedded in the methodology documents. Extract it automatically:

```bash
cd "LLM Training/tools"

# Extract all Python code from methodology
python scripts/extract_code_from_methodology.py --all

# This will create:
# - data_collection/calculate_inter_rater_reliability.py
# - data_collection/resolve_disagreements.py
# - evaluation/fairness_tester.py
# - deployment/production_monitor.py
# - deployment/drift_detector.py
# - deployment/monitoring_dashboard.py
# - And ~20 more files
```

## Step 2: Install Package (2 minutes)

```bash
# Option A: Install in development mode (recommended)
pip install -e .

# Option B: Install dependencies only
pip install -r requirements.txt

# Verify installation
python -c "import data_collection, evaluation, deployment; print('✅ Import successful!')"
```

## Step 3: Run Your First Tool (2 minutes)

### Example 1: Calculate Data Quality Metrics

```bash
# Using command-line tool
llm-quality --input ../data_collection/sample_data.json --output metrics.json

# Or use Python directly
python -c "
from data_collection.calculate_quality_metrics import calculate_metrics
metrics = calculate_metrics('sample_data.json')
print(f'Quality Score: {metrics[\"avg_quality\"]:.2f}/5.0')
"
```

### Example 2: Test Model Fairness

```python
from evaluation.fairness_tester import FairnessTester

# Load your model (example)
# model = load_model("./my-firefighter-model")

# Run comprehensive fairness tests
tester = FairnessTester(model)
results = tester.run_full_suite()

# Results show pass/fail for each test:
# ✅ Geographic Fairness: PASS
# ✅ Resource Adaptation: PASS
# ❌ Language Fairness: FAIL (needs improvement)
```

### Example 3: Start Production Monitoring

```bash
# Start monitoring dashboard
llm-dashboard --port 5000

# Open browser to http://localhost:5000/dashboard
# See real-time metrics, drift alerts, user feedback
```

## Common Workflows

### Workflow 1: Data Collection Pipeline

```bash
# 1. Collect expert data
llm-collect --interviews_dir ./interviews --output raw_data.json

# 2. Clean and validate
llm-clean --input raw_data.json --output clean_data.json

# 3. Remove duplicates
llm-dedup --input clean_data.json --output dedup_data.json

# 4. Calculate quality metrics
llm-quality --input dedup_data.json --output metrics.json

# 5. Check inter-rater reliability
python -m data_collection.calculate_inter_rater_reliability \
  --expert1 expert1_ratings.csv \
  --expert2 expert2_ratings.csv

# 6. Split dataset
python -m data_collection.split_data \
  --input dedup_data.json \
  --train 0.8 --val 0.1 --test 0.1
```

### Workflow 2: Model Evaluation

```bash
# 1. Run benchmarks
python -m evaluation.run_benchmark \
  --model ./my-model \
  --benchmark firefighter \
  --output results.json

# 2. Test fairness
python -m evaluation.fairness_tester \
  --model ./my-model \
  --output fairness_report.json

# 3. Check calibration
python -m evaluation.calibration \
  --model ./my-model \
  --validation_data val_data.json

# 4. Robustness testing
python -m evaluation.robustness \
  --model ./my-model \
  --output robustness_report.json
```

### Workflow 3: Production Deployment

```bash
# 1. Start A/B testing server (shadow deployment)
python -m deployment.ab_testing_server \
  --model-a ./current-model \
  --model-b ./new-model \
  --mode shadow \
  --port 8000

# 2. Start monitoring dashboard (separate terminal)
llm-dashboard --port 5000

# 3. Monitor drift (runs weekly)
python -m deployment.drift_detector \
  --model ./production-model \
  --reference_data test_set.json \
  --schedule weekly
```

## Jupyter Notebooks (Interactive Tutorials)

```bash
cd notebooks/

# Install Jupyter if needed
pip install jupyter

# Launch Jupyter
jupyter notebook

# Open:
# 01_data_collection_pipeline.ipynb  - Full data workflow
# 02_quality_metrics.ipynb           - Quality analysis
# 03_fairness_testing.ipynb          - Bias detection
# 04_production_monitoring.ipynb     - Deployment monitoring
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'data_collection'"

**Solution**: Install the package first:
```bash
cd "LLM Training/tools"
pip install -e .
```

### Issue: "Code files not found in data_collection/"

**Solution**: Extract code from methodology:
```bash
python scripts/extract_code_from_methodology.py --all
```

### Issue: "ImportError: transformers not found"

**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

## Next Steps

1. ✅ **Read the full methodology**: Start with `../data_collection/README.md`
2. ✅ **Run the example notebooks**: See `notebooks/` directory
3. ✅ **Explore TOOLS_INVENTORY.md**: Complete catalog of all 26 tools
4. ✅ **Check the tests**: Run `pytest tests/ -v` to see examples

## Getting Help

- **Documentation**: See parent `LLM Training/` directory READMEs
- **Code Examples**: All tools have docstrings and usage examples
- **Issues**: Check TOOLS_INVENTORY.md for known issues
- **Support**: contact@crisis-mas.org

Happy Training! 🚀
