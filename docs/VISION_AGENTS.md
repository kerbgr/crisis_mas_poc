# Vision Agents — Design & Implementation Reference

Multimodal pre-assessment layer added to the Crisis MAS coordinator pipeline.
Two agents run as **Step 0** in `CoordinatorAgent.make_final_decision()`, before any
expert agent is consulted.

---

## Overview

The core pipeline has no spatial awareness. It cannot reason about whether a coast
guard vessel can physically reach an inland flood, or whether a harbor camera is
showing anomalous wave behaviour. These two agents fill that gap:

| Agent | Input | Output | When it runs |
|---|---|---|---|
| `GeospatialContextAgent` | Scenario `location.coordinates` | Terrain classification + agent eligibility filter | Once per decision cycle |
| `CameraFeedAgent` | Image URL / RTSP stream / local file | Structured situational report (JSON) | Once per decision cycle, per declared feed |

Both agents are **optional and non-blocking**. If Ollama is not running, if the
network is unavailable, or if the scenario has no coordinates or no camera feeds,
the pipeline continues without them — no exception, no degradation of core logic.

---

## Architecture position

```
CoordinatorAgent.make_final_decision()
│
├─ Step 0a: GeospatialContextAgent.analyze()
│     Reads scenario location → terrain classification → filters expert_agents list
│
├─ Step 0b: CameraFeedAgent.analyze_feeds()
│     Reads scenario camera_feeds[] → situational reports → injects into scenario context
│
├─ Step 1: collect_assessments(agents=eligible_agents)   ← filtered list
│     Expert agents now see camera intelligence in scenario context
│
├─ Steps 2–6: aggregation, MCDA, consensus, decision
│
└─ Final decision dict gains:
      geospatial_context     — terrain facts + excluded agents
      camera_feed_reports    — list of per-feed reports
      agents_excluded_by_terrain — agent_ids removed at Step 0
```

---

## GeospatialContextAgent

**File:** `agents/geospatial_agent.py`

### Purpose

Determines whether maritime assets (coast guard, harbour patrol) can physically
reach the incident location. An inland flood in Karditsa has no navigable sea
connection; a volcanic emergency on Santorini requires maritime evacuation. The
agent encodes this spatial constraint so the coordinator can filter ineligible
agents before they spend LLM budget on an assessment they cannot execute.

### Inputs

- `scenario["location"]["coordinates"]` — `{"lat": float, "lon": float}`
- `expert_agents` — list of `ExpertAgent` instances to classify

### Outputs

```python
{
    "terrain_type":        "island | coastal_mainland | inland",
    "is_island":           bool,
    "island_name":         str | None,      # populated for known Greek islands
    "is_coastal":          bool,
    "has_water_access":    bool,
    "eligible_agent_ids":  [str, ...],
    "ineligible_agent_ids":[str, ...],
    "terrain_result":      { ... },         # full internal result dict
    "method_used":         "vision+deterministic | deterministic | default",
    "coordinates":         {"lat": float, "lon": float}
}
```

### Classification strategy: hybrid

The agent runs a two-track classification and combines results:

```
                 ┌── OSM tile available?
use_vision=True ─┤                      ┌── vision says "island"?
                 │   Ollama reachable? ──┤     YES → trust vision (high reliability)
                 │                       └── vision says inland/coastal?
                 │                             → use deterministic result instead
                 └── Fallback ──────────────── deterministic only
```

**Why this split:**
During empirical testing with `llama3.2-vision` (11B), island detection from OSM
tiles was reliable across all test cases — the model consistently identifies open
sea surrounding a landmass. Non-island terrain classification was inconsistent:
the model sometimes classified inland Karditsa as "coastal" because the wide-area
tile at zoom 10 showed distant Aegean water. The deterministic bounding-box system
produced correct results in 100% of cases for both island and inland zones.

The hybrid approach therefore uses vision for its strength (island confirmation with
a natural-language reasoning trace) and deterministic for its strength (precise
inland exclusion for the Greek operational theatre).

#### Track 1: Vision (llama3.2-vision via Ollama)

1. Converts lat/lon to OSM tile coordinates at zoom 10 (~37 km wide tile)
2. Fetches 256×256 PNG from `tile.openstreetmap.org`
3. Base64-encodes the tile and sends to Ollama at `localhost:11434/v1/chat/completions`
4. Parses JSON response; if `terrain_type == "island"` → accepts result

**Prompt (condensed):**
> "The target location is at the CENTER of the tile. Is the center point on an
> ISLAND (surrounded by sea on all sides)? Is there sea/ocean visible anywhere?
> Can a coast guard vessel reach this location?
> IMPORTANT: Rivers and inland lakes are NOT sea. Only ocean/sea counts."

The `IMPORTANT` line was empirically necessary — without it, the 11B model
classified rivers and reservoirs as navigable sea.

**JSON schema returned by model:**
```json
{
  "terrain_type": "island | coastal_mainland | inland",
  "sea_visible_in_tile": true,
  "has_maritime_access": true,
  "confidence": 0.9,
  "reasoning": "one sentence describing what the model sees"
}
```

#### Track 2: Deterministic bounding boxes

Pure Python — no network, no ML, runs in microseconds.

**Priority order:**

1. **Island check** — 19 curated rectangular lat/lon bounding boxes covering the
   main inhabited Greek islands. If the coordinates fall inside any box → `island`,
   `has_water_access=True`.

2. **Inland zone check** — 8 curated boxes covering known landlocked city regions
   in Thessaly, Epirus, and West Macedonia (Karditsa, Larissa, Trikala, Kozani,
   Ioannina, Grevena, Florina, Kastoria). If matched → `inland`,
   `has_water_access=False`.

3. **Default** — `coastal_mainland`, `has_water_access=True`. Deliberately
   permissive: most of Greece has some coastal access, and false exclusion of a
   useful agent is worse than false inclusion.

**Registered islands (19):**
Crete, Rhodes, Corfu, Evia, Lesbos, Chios, Samos, Kos, Zakynthos, Kefalonia,
Lemnos, Mykonos, Naxos, Paros, Santorini, Tinos, Andros, Syros, Ikaria

**Registered inland zones (8):**
Karditsa, Larissa, Trikala, Kozani, Ioannina, Grevena, Florina, Kastoria

### Agent eligibility matching

Only maritime-typed agents are filtered by terrain. The match is substring-based
on `agent_id`:

```python
if "coastguard" in agent_id.lower() or "coast_guard" in agent_id.lower():
    eligible = has_water_access
elif "maritime" in agent_id.lower():
    eligible = has_water_access
else:
    eligible = True   # all other specialisations are road-mobile
```

All other agent types — medical, firefighting, police, civil protection, logistics,
communications, environmental, search & rescue — are terrain-agnostic and always
remain eligible.

### Test results (empirical)

Tested against 5 representative Greek locations with `llama3.2-vision`:

| Location | Expected | Hybrid result | Correct? |
|---|---|---|---|
| Santorini (volcanic) | island / maritime=True | island / True | ✓ |
| Rhodes | island / maritime=True | island / True | ✓ |
| Karditsa (inland flood) | inland / maritime=False | inland / False | ✓ |
| Evia (forest fire) | island / maritime=True | island / True | ✓ |
| Thessaloniki (hazmat) | coastal / maritime=True | island / True | ✓ (functionally correct — label wrong) |

Karditsa: vision returned "coastal_mainland" (distant sea visible in tile) →
hybrid correctly fell back to deterministic → `inland`, `coastguard=False`.

Thessaloniki: vision returned "island" (misread) but operational outcome is still
correct — Thessaloniki IS a major coastal port and coast guard units CAN operate
there.

### Degradation behaviour

| Condition | Behaviour |
|---|---|
| No coordinates in scenario | All agents eligible (permissive default) |
| OSM tile fetch fails (no network) | Falls back to deterministic |
| Ollama not running | Falls back to deterministic |
| Vision model parse fails | Falls back to deterministic |
| `use_vision=False` | Deterministic only |

---

## CameraFeedAgent

**File:** `agents/camera_feed_agent.py`

### Purpose

Provides real-time visual situational intelligence from cameras at key locations:
harbor CCTV for tsunami early warning, crowd monitoring at evacuation assembly
points, wildfire perimeter cameras, flood gauge cameras. Reports are injected into
the scenario context string so every expert agent receives this intelligence as
background when forming their assessment.

The agent is classified as **Bronze level** (operational / sensor layer) — it
produces raw sensor readings, not expert judgement.

### Image source types

| Source format | Acquisition method |
|---|---|
| `https://...` / `http://...` | `requests.get()` with 10 s timeout |
| `rtsp://...` | OpenCV `VideoCapture` — single frame grab, then release |
| Local file path | Direct binary file read |

RTSP requires `pip install opencv-python`. If OpenCV is absent, RTSP sources return
`status: feed_unavailable` and the pipeline continues.

### Analysis modes

Each mode selects a specialised prompt tuned for that hazard context:

#### `general` — multi-purpose
Returns: `crowd_density`, `water_anomaly_detected`, `wave_detected`,
`smoke_visible`, `fire_visible`, `structural_damage_visible`,
`vehicle_congestion`, `situation_severity`, `immediate_hazards[]`,
`situation_summary`, `confidence`

#### `tsunami` — harbor / coastal early warning
Directs the model to look for: sea withdrawal, abnormal wave patterns, debris in
water, distressed vessels, people running from waterfront.

Returns: `tsunami_indicators_present`, `water_withdrawal_observed`,
`abnormal_wave_pattern`, `debris_in_water`, `vessels_distressed`,
`warning_level` (none/watch/warning/emergency), `crowd_density_at_shore`,
`wave_detected`, `wave_height_estimate_m`, `recommended_action`
(monitor/alert_authorities/immediate_evacuation)

#### `crowd` — evacuation assembly point monitoring
Returns: `crowd_density`, `crowd_count_estimate`, `crowd_moving_direction`
(unclear/stable/away_from_hazard/toward_hazard), `panic_indicators`,
`bottleneck_observed`, `stampede_risk`, `situation_severity`

#### `fire` — wildfire / structural fire
Returns: `fire_visible`, `smoke_visible`, `smoke_color`
(none/white/gray/black/brown — black/brown indicate toxic fuel), `fire_size_estimate`
(none/small/moderate/large/extreme), `spread_direction`, `structures_threatened`,
`people_visible_in_danger`

### Batch processing from scenario JSON

Scenarios declare camera feeds in their JSON:

```json
"camera_feeds": [
  {
    "url": "https://camera.example.gr/athinios-port.jpg",
    "mode": "tsunami",
    "label": "Athinios Car-Ferry Port — harbor wave monitoring"
  },
  {
    "url": "rtsp://192.168.1.10/cam1",
    "mode": "crowd",
    "label": "Fira Main Square — evacuation assembly point"
  }
]
```

`analyze_feeds(feeds)` iterates the list, calls `analyze_feed()` per entry, and
returns a list of reports in the same order. The coordinator calls this at Step 0
and injects the result into `scenario["additional_context"]`.

### Coordinator context injection

`to_scenario_context(reports)` formats all reports into a text block prepended to
the scenario's context string. Critical indicators are highlighted with `***`:

```
[Camera Feed Intelligence]

  Feed: Athinios Car-Ferry Port (tsunami mode)
  Status: ok
  Severity: normal
  Summary: No signs of a tsunami or any other abnormality observed in the harbor
  Confidence: 0%
```

If a tsunami or wave indicator fires:
```
  *** TSUNAMI INDICATORS — warning_level=warning
  *** SEA WITHDRAWAL OBSERVED
```

This block is seen by every expert agent as part of their prompt — they can
reference current visual ground truth when forming their belief distribution.

### Test results (empirical)

Tested with `llama3.2-vision` on a real harbor photo:

| Field | Value | Assessment |
|---|---|---|
| `status` | `ok` | Feed acquired and parsed |
| `tsunami_indicators_present` | `False` | Correct (normal harbour scene) |
| `water_withdrawal_observed` | `False` | Correct |
| `wave_detected` | `False` | Correct |
| `warning_level` | `none` | Correct |
| `recommended_action` | `monitor` | Correct |
| `situation_summary` | "No signs of a tsunami or any other abnormality observed in the harbor" | Natural, accurate |
| `confidence` | `0.0` | Model does not fill this reliably — treat as non-informative |

**Model reliability by field type:**
- Boolean indicators (`wave_detected`, `tsunami_indicators_present`, `panic_indicators`, etc.) — **reliable** with `llama3.2-vision`
- Ordinal enumerations (`warning_level`, `crowd_density`) — **reliable** with `llama3.2-vision`
- Free-text `situation_summary` — **reliable** with `llama3.2-vision`; `moondream` echoes template
- `confidence` float — **not reliable** with either model; always 0.0 or template default

### Degradation behaviour

| Condition | `status` value | Pipeline impact |
|---|---|---|
| Feed URL unreachable | `feed_unavailable` | Skipped; other feeds still processed |
| Ollama not running | `vision_model_unavailable` | All feeds skipped; no context injection |
| Model response not parseable as JSON | `parse_failed` | Partial: base report defaults used |
| No `camera_feeds` in scenario | — | Step 0b skipped entirely |

---

## VisionClient

**File:** `llm_integration/vision_client.py`

Thin wrapper around Ollama's OpenAI-compatible API at `localhost:11434/v1`.

### Key design decisions

**Image encoding:** Images are base64-encoded and sent as `data:<mime>;base64,<b64>`
data URLs in the `image_url` content block. Ollama accepts any MIME type but PNG
(OSM tiles) and JPEG (camera frames) are the two types used here.

**Timeout:** 90 seconds default. Vision inference on an 11B model takes 10–40 s
depending on hardware. The timeout is generous but bounded to prevent indefinite
stalls in the decision pipeline.

**Connection error handling:** A `ConnectionError` (Ollama not running) logs a
warning and returns `None` — not an exception. All callers check for `None` and
fall back gracefully.

**JSON extraction:** `parse_json_response()` scans the raw text for the first `{`
and last `}` and attempts `json.loads()` on that substring. Vision models
frequently wrap JSON in prose or markdown fences; this handles most cases robustly.

### Model recommendations

| Model | Size | Geospatial | Camera feed | Notes |
| --- | --- | --- | --- | --- |
| `minicpm-v` | 5.5 GB | Good island detection | Full structured output, natural summaries | **Recommended.** Works with Ollama 0.30.x. |
| `llava:7b` | ~4 GB | Adequate | Good instruction following | Solid alternative to minicpm-v. |
| `moondream` | 1.9B, ~1.8 GB | Not usable - returns template default | Boolean fields only; enums and free-text unreliable | Fast but limited. |
| `llama3.2-vision` | 11B, ~7 GB | Was reliable pre-0.30 | Was reliable pre-0.30 | **Broken in Ollama 0.30.x** - mllama architecture regression ([issue #16490](https://github.com/ollama/ollama/issues/16490)). Returns HTTP 500. Do not use. |

> **Important:** `minicpm-v` natural-language summaries (`situation_summary`, `reasoning`) are reliable.
> Binary JSON fields (e.g. `tsunami_indicators_present`) can occasionally contradict the prose — always
> read the summary alongside the flags when making operational decisions.

---

## Extending the system

### Adding a new island or inland zone

Append to `_GREEK_ISLANDS` or `_INLAND_LOCATIONS` in `geospatial_agent.py`:

```python
"samothraki": (40.43, 40.55, 25.45, 25.75),   # island
"ptolemaida":  (40.48, 40.57, 21.63, 21.80),   # inland, W. Macedonia
```

### Adding a new camera analysis mode

1. Add a prompt constant following the existing pattern (use `{{` / `}}` for literal
   braces since all prompts are passed through `.format(context=...)`):

```python
_FLOOD_PROMPT = (
    "You are analyzing a bridge or road camera for flood water level.\n"
    "Context: {context}\n\n"
    "Return ONLY valid JSON:\n"
    "{{\n"
    '  "water_visible": false,\n'
    '  "road_submerged": false,\n'
    '  "estimated_depth_m": null,\n'
    '  "bridge_clearance_safe": true,\n'
    '  "situation_severity": "normal|elevated|high|critical",\n'
    '  "situation_summary": "one-sentence description",\n'
    '  "confidence": 0.0\n'
    "}}"
)
```

2. Register it in `_PROMPTS`:

```python
_PROMPTS = {
    "general": _GENERAL_PROMPT,
    "tsunami": _TSUNAMI_PROMPT,
    "crowd":   _CROWD_PROMPT,
    "fire":    _FIRE_PROMPT,
    "flood":   _FLOOD_PROMPT,   # new
}
```

3. Add indicator escalation to `to_scenario_context()` if needed.

### Adding non-Greek geography to deterministic geospatial

Add a third dict for known coastal zones, and check it in
`_classify_terrain_deterministic()` before the default:

```python
_COASTAL_ZONES = {
    "istanbul_coast": (40.90, 41.10, 28.80, 29.20),
    "izmir_bay":      (38.35, 38.55, 26.90, 27.20),
}
```

---

## Activation

Pass both agents to `CoordinatorAgent` at construction:

```python
from agents.geospatial_agent import GeospatialContextAgent
from agents.camera_feed_agent import CameraFeedAgent

coordinator = CoordinatorAgent(
    expert_agents=agents,
    er_engine=er_engine,
    mcda_engine=mcda_engine,
    consensus_model=consensus_model,
    vision_agent=GeospatialContextAgent(),         # hybrid, Ollama optional
    camera_agent=CameraFeedAgent(mode="general"),  # default mode; per-feed override via JSON
)
```

Both default to `None` — omitting them leaves the pipeline fully unchanged.

**Ollama setup:**
```bash
ollama pull minicpm-v          # 5.5 GB — recommended; works with Ollama 0.30.x
ollama pull llava:7b           # 4 GB — solid alternative
ollama pull moondream          # 1.9 GB — booleans only, much faster but limited
# NOTE: llama3.2-vision is broken in Ollama 0.30.x (mllama regression, issue #16490)
```

**LM Studio setup:** Load any vision-capable model (LLaVA, BakLLaVA, etc.) in the LM Studio UI, then pass `--vision-provider lmstudio` at the CLI.

**Optional OpenCV for RTSP:**
```bash
pip install opencv-python
```
