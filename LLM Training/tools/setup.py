"""
Setup script for LLM Training Tools package.

This package provides production-ready tools for domain-specific LLM training,
including data collection, quality assurance, evaluation, and deployment monitoring.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-training-tools",
    version="1.0.0",
    author="Crisis MAS Team",
    author_email="contact@crisis-mas.org",
    description="Production-ready tools for domain-specific LLM training and deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crisis-mas-llm-training",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llm-collect=data_collection.collect_expert_data:main",
            "llm-clean=data_collection.clean_data:main",
            "llm-dedup=data_collection.deduplicate:main",
            "llm-quality=data_collection.calculate_quality_metrics:main",
            "llm-fairness=evaluation.fairness_tester:main",
            "llm-monitor=deployment.production_monitor:main",
            "llm-dashboard=deployment.monitoring_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
)
