#!/bin/zsh
# Serve the Stage 0 model through an OpenAI-compatible API for AEGIS.
#
# KNOWN ISSUE (verified 2026-07-15, mlx-lm 0.31.3): `mlx_lm server
# --adapter-path` did NOT apply the adapter to responses in our testing,
# even when requests use "model": "default_model" (compared at temperature 0
# against direct `load(model, adapter_path=...)` generation, which does
# apply it). Until that's resolved upstream, FUSE the adapter into the model
# first and serve the fused model -- that path is unambiguous.
set -euo pipefail
cd "$(dirname "$0")/.."

FUSED=stage0/fused_model

if [ ! -d "$FUSED" ]; then
  ./.venv/bin/python -m mlx_lm fuse \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --adapter-path stage0/adapters \
    --save-path "$FUSED"
fi

# AEGIS integration: point llm_integration/lmstudio_client.py at
# base_url="http://localhost:18080/v1" -- same OpenAI-compatible protocol
# as LM Studio. Requests may use any "model" name once serving a local path.
exec ./.venv/bin/python -m mlx_lm server \
  --model "$FUSED" \
  --port 18080
