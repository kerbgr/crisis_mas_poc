# local_training/ — the executed pipeline (not documentation)

Everything in this directory has **actually been run** on the project's reference
hardware (MacBook Pro M4 Pro, 48GB, macOS 26.5) — unlike the parent folder's
methodology documents, which describe targets. Stage status lives in
[../PROJECT_PLAN.md](../PROJECT_PLAN.md).

## Environment

```bash
# One-time setup (Python 3.12 via anaconda; .venv is gitignored)
/opt/anaconda3/bin/python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt   # mlx 0.32.0, mlx-lm 0.31.3 (pinned as verified)
```

## Stage 0 — pipeline smoke test (EXECUTED 2026-07-15)

Proves the full loop on local hardware: synthetic seed data → MLX QLoRA →
adapter → fuse → OpenAI-compatible serving → AEGIS-ready endpoint →
deployment safety checker. **The model produced here is NOT deployable** —
its dataset is rule-generated and expert-unvalidated by design.

```bash
./stage0/train.sh    # dataset gen + training + before/after adapter test
./stage0/serve.sh    # fuse (once) + serve at http://localhost:18080/v1
```

### Measured results (M4 Pro 48GB, first real numbers for this project)

| Metric | Value |
|--------|-------|
| Model | mlx-community/Qwen2.5-0.5B-Instruct-4bit (MLX-native QLoRA) |
| Trainable params | 1.47M (0.297%) |
| 100 iters, batch 2, 8 layers, --mask-prompt | **~66s train wall time** (71s incl. model download) |
| Throughput | **~1,300–1,470 tokens/sec** |
| Peak memory | **1.71 GB** (of 48GB) |
| Val loss | 3.648 → **0.133** |
| Behavior | Base model rambles; adapted model applies evacuation-first doctrine correctly on **held-out** parameter combinations (see `stage0/adapter_test_output.json`) |

### Findings that changed the documentation (verified, not assumed)

1. **mlx-lm CLI drifted from the 2025 guide**: invocation is `python -m mlx_lm lora`
   (subcommand), `--data` takes a **directory** with `train.jsonl`/`valid.jsonl`,
   `--lora-layers` → `--num-layers`. Chat-format `{"messages": [...]}` JSONL is
   accepted directly — no text-template conversion needed.
2. **QLoRA works on Apple Silicon via MLX** (training directly on a 4-bit model).
   The "no QLoRA on Apple Silicon" limitation in APPLE_SILICON_GUIDE.md applies
   only to the CUDA/bitsandbytes route. `--fine-tune-type full` and `dora` also exist.
3. **`mlx_lm server --adapter-path` did not apply the adapter** in mlx-lm 0.31.3
   (verified at temperature 0 against direct `load(..., adapter_path=...)`, which
   does apply it; requests must also use `"model": "default_model"` to route to
   the CLI-specified model at all). **Workaround, verified working**: fuse first
   (`mlx_lm fuse`), then serve the fused model — `serve.sh` does exactly this.
4. **The safety net works end-to-end**: the Stage-0 model confidently answered a
   wrong IDLH value for ammonia ("10 ppm" — it cross-wired the chlorine figure;
   classic tiny-model fact scrambling), and
   `tools/deployment/detect_safety_failures.py` flagged it as
   `hallucinated_hazmat_data`. This is the concrete argument for keeping the
   evaluation/monitoring layer in front of any fine-tuned model.

### AEGIS integration

`serve.sh` exposes the same OpenAI-compatible protocol as LM Studio. Point the
existing client at it — no AEGIS code changes needed:

```python
from llm_integration.lmstudio_client import LMStudioClient
client = LMStudioClient(base_url="http://localhost:18080/v1", model_name="default_model")
```

## Artifacts in stage0/

- `make_seed_dataset.py` — rule-based generator (doctrine distilled from the 5
  hand-written seed examples; constants consistent with
  `tools/deployment/detect_safety_failures.py`)
- `data/` — 108 train / 12 valid / 8 test (synthetic, unvalidated)
- `adapters/` — trained LoRA weights
- `fused_model/` — adapter fused into base, ready to serve
- `training_log.txt` — the real training log (this project's first)
- `adapter_test_output.json` — base-vs-tuned comparison on a held-out scenario
