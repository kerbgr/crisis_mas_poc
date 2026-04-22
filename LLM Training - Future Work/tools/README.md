# LLM Training Tools

Production-ready toolkit for domain-specific LLM training, evaluation, and deployment monitoring.

## Features

- **Data Collection**: Automate expert interview processing, quality checks, and dataset versioning
- **Quality Assurance**: Inter-rater reliability, conflict resolution, deduplication
- **Fairness Testing**: Comprehensive bias detection across demographics, geography, language
- **Production Monitoring**: Real-time drift detection, alerting, and performance tracking
- **A/B Testing**: Safe deployment with canary rollouts and automatic rollback

## Installation

```bash
# Clone the repository
cd "LLM Training/tools"

# Install in development mode
pip install -e .

# Or install from PyPI (when published)
pip install llm-training-tools
```

## Quick Start

### 1. Data Collection Pipeline

```bash
# Collect and process expert data
llm-collect --interviews_dir ./interviews --output raw_data.json

# Clean and validate
llm-clean --input raw_data.json --output clean_data.json

# Remove duplicates
llm-dedup --input clean_data.json --output training_data.json

# Calculate quality metrics
llm-quality --input training_data.json --output metrics.json
```

### 2. Fairness Testing

```python
from evaluation.fairness_tester import FairnessTester

# Load your model
model = load_model("./firefighter-model")

# Run comprehensive fairness tests
tester = FairnessTester(model)
results = tester.run_full_suite()
```

### 3. Production Monitoring

```bash
# Start monitoring dashboard
llm-dashboard --port 5000

# In your Flask app
from deployment.production_monitor import ProductionMonitor

monitor = ProductionMonitor(window_size=1000)
# Log each request
monitor.log_response(metrics)
```

## Package Structure

```
tools/├── data_collection/      # Data processing and QA tools
│   ├── collect_expert_data.py
│   ├── clean_data.py
│   ├── deduplicate.py
│   ├── calculate_quality_metrics.py
│   └── calculate_inter_rater_reliability.py
│
├── evaluation/          # Model evaluation tools
│   ├── fairness_tester.py
│   ├── calibration.py
│   └── robustness.py
│
├── deployment/          # Production monitoring
│   ├── production_monitor.py
│   ├── drift_detector.py
│   ├── ab_testing_server.py
│   └── monitoring_dashboard.py
│
├── notebooks/           # Interactive tutorials
│   ├── 01_data_collection_pipeline.ipynb
│   ├── 02_quality_metrics.ipynb
│   ├── 03_fairness_testing.ipynb
│   └── 04_production_monitoring.ipynb
│
├── tests/               # Unit tests
│   ├── test_data_collection.py
│   ├── test_evaluation.py
│   └── test_deployment.py
│
└── scripts/             # Shell scripts
    └── run_full_pipeline.sh
```

## Documentation

Full documentation available in the parent `LLM Training/` directory:

- [Data Collection Methodology](../data_collection/README.md)
- [Evaluation Methodology](../evaluation/README.md)
- [Deployment Guide](../DEPLOYMENT.md)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_fairness.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use these tools in your research, please cite:

```bibtex
@software{llm_training_tools2025,
  title = {LLM Training Tools: Production-Ready Toolkit for Domain-Specific LLMs},
  author = {kerbgr},
  year = {2025},
  url = {https://github.com/kerbgr/crisis_mas_poc}
}
```

## Support

- Documentation: [Full Methodology](../README.md)
- Issues: [GitHub Issues](https://github.com/yourusername/crisis-mas-llm-training/issues)
- Email: contact@crisis-mas.org
