#!/bin/zsh
# Stage 0 training run -- verified against mlx-lm 0.31.3 CLI on 2026-07-15.
# NOTE the CLI differences vs the 2025 examples in APPLE_SILICON_GUIDE.md:
#   --data takes a DIRECTORY containing train.jsonl/valid.jsonl (not a file)
#   --lora-layers was renamed to --num-layers
#   invocation is `python -m mlx_lm lora` (subcommand), not `python -m mlx_lm.lora`
set -euo pipefail
cd "$(dirname "$0")/.."

./.venv/bin/python stage0/make_seed_dataset.py

/usr/bin/time ./.venv/bin/python -m mlx_lm lora \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --train \
  --data stage0/data \
  --iters 100 \
  --batch-size 2 \
  --num-layers 8 \
  --learning-rate 1e-4 \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --mask-prompt \
  --adapter-path stage0/adapters \
  --seed 42 2>&1 | tee stage0/training_log.txt

./.venv/bin/python stage0/test_adapter.py
