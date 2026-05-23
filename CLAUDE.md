# Crisis MAS PoC — CLAUDE.md

## What this is
Master's thesis PoC: a Multi-Agent System for crisis management decision support. Python 3, Pydantic v2, no web framework.

**Run:** `python main.py [--scenario flood_scenario|forest_fire_evia|ammonia_leak_elefsina] [--aggregation-method ER|GAT] [--agents all|auto] [--compare-methods]`
**Venv:** `venv/` — activate before running. Results land in `results/`.

## Key architecture

`CoordinatorAgent.make_final_decision()` runs a 6-step pipeline:
1. Collect expert assessments (parallel via `ThreadPoolExecutor`)
2. Aggregate beliefs — **ER** (Dempster-Shafer) or **GAT** (graph attention network)
3. Score alternatives — MCDA (`mcda_engine.py`)
4. Check consensus (`consensus_model.py`)
5. Conflict resolution if consensus < threshold (default 0.75)
6. Final decision — 60% belief weight + 40% MCDA score

**Aggregation methods:** `ER` (default), `GAT`, `GAT_TRAINED`. Set via `--aggregation-method` or `aggregation_method=` in constructor.

## Critical model quirks

- `AgentAssessment` (`models/data_models.py:146`) — Pydantic with `extra="allow"`. Has a dict-like interface; set arbitrary attrs directly. Do NOT reconstruct it.
- `BeliefDistribution` — beliefs must sum to ~1.0 (±0.01 tolerance); values in [0.0, 1.0].
- GAT reads `assessment.get('reliability_score', 0.8)` at `gat_aggregator.py:423` — inject before calling GAT.

## ReliabilityTracker
- Persistence: `results/reliability/{agent_id}_reliability.json`
- Loaded/saved via `base_agent.py` wrappers
- Recorded in coordinator after each assessment; updated post-decision using `recommended_alternative` as consensus ground truth
- Dynamic weights injected into GAT via `assessment['reliability_score']`

## Directory map
```
agents/          CoordinatorAgent, ExpertAgent, BaseAgent, ReliabilityTracker
decision_framework/  evidential_reasoning, gat_aggregator, mcda_engine, consensus_model
llm_integration/     claude_client, lmstudio_client, openai_client, prompt_templates
models/          data_models (BeliefDistribution, AgentAssessment, ...)
scenarios/       JSON scenario files + loader/selector
utils/           config.py, validation.py
results/         all output (JSON, plots, reliability/)
```

## LLM providers
`--llm-provider lmstudio` runs fully local/free (LM Studio at localhost:1234). Use this when conserving API costs.

## Vision Subsystem (Multimodal Pre-Assessment Layer)

Two optional agents run **before Step 1** in `make_final_decision()` as a pre-filter / context enrichment pass.

### GeospatialContextAgent (`agents/geospatial_agent.py`)
- Fetches a 256×256 OSM tile PNG for the scenario's `location.coordinates`
- Sends tile to a local Ollama vision model (llama3.2-vision, moondream, etc.)
- Returns terrain classification: `island | coastal_mainland | inland`
- Filters ineligible agents — e.g. coast guard excluded for inland flood in continental Greece
- **Fallback**: deterministic bounding-box rules for Greek geography (no Ollama required)
- Result stored in `decision['geospatial_context']`

### CameraFeedAgent (`agents/camera_feed_agent.py`)
- Reads image URL, RTSP stream, or local file; samples a frame via OpenCV
- Sends frame to vision model; returns structured JSON situational report
- Modes: `general`, `tsunami` (harbor wave detection), `crowd`, `fire`
- Scenario JSON can declare `camera_feeds: [{url, mode, label}]` for automatic ingestion
- Output injected into scenario context string before expert agents see it
- Bronze-level agent (`LEVEL = "bronze"`)

### VisionClient (`llm_integration/vision_client.py`)
- Thin wrapper for Ollama's OpenAI-compatible API at `localhost:11434/v1`
- Base64-encodes image bytes; sends as `image_url` content block
- Gracefully returns `None` when Ollama is not running (pipeline continues unaffected)

**Local setup:** `ollama pull llama3.2-vision` (11B, ~7 GB) or `ollama pull moondream` (1.9B, ~1.8 GB, faster)
**Dependencies:** `requests` (already present); `opencv-python` for RTSP (`pip install opencv-python`)
**Inject into coordinator:**
```python
from agents.geospatial_agent import GeospatialContextAgent
from agents.camera_feed_agent import CameraFeedAgent
coordinator = CoordinatorAgent(..., vision_agent=GeospatialContextAgent(), camera_agent=CameraFeedAgent())
```

## Scenarios
`flood_scenario`, `forest_fire_evia`, `ammonia_leak_elefsina`, `santorini_volcanic_seismic` — JSON files in `scenarios/`.
Santorini scenario declares `camera_feeds` for automatic CameraFeedAgent ingestion.
