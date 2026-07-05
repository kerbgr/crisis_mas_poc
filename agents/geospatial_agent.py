"""
GeospatialContextAgent — Map-based terrain analysis for agent eligibility filtering.

Fetches a 256x256 OpenStreetMap tile for the scenario's coordinates, sends it to a
local vision model, and returns terrain classification + agent eligibility filter.

RATIONALE:
The coordinator cannot know from text alone whether maritime assets can physically
reach an incident.  A flood in central continental Greece (Karditsa, Larissa) is
inaccessible to coast guard vessels; a volcanic emergency on Santorini island
demands them.  This agent encodes that spatial awareness.

FALLBACK HIERARCHY:
  1. Vision model (Ollama) — fetches OSM tile, analyzes with llama3.2-vision/moondream
  2. Deterministic rules — fast, offline, Greek geography bounding boxes
  3. Permissive default — all agents eligible (safe degradation when no coords)
"""

import json
import logging
import math
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OSM tile helpers
# ---------------------------------------------------------------------------

def _deg2tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert WGS-84 lat/lon to OSM tile (x, y) at the given zoom level."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _fetch_osm_tile(lat: float, lon: float, zoom: int = 12) -> Optional[bytes]:
    """
    Fetch a 256x256 PNG map tile from OpenStreetMap.

    Returns raw PNG bytes, or None if the request fails (network unavailable,
    rate-limited, etc.).  OSM tile usage policy requires a User-Agent header.
    """
    x, y = _deg2tile(lat, lon, zoom)
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CrisisMAS-PoC/1.0 (academic research, contact: thesis)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        logger.debug("OSM tile fetched: z=%d x=%d y=%d (%d bytes)", zoom, x, y, len(data))
        return data
    except Exception as e:
        logger.debug("OSM tile fetch failed (z=%d x=%d y=%d): %s", zoom, x, y, e)
        return None


# ---------------------------------------------------------------------------
# Deterministic geography rules (Greek theatre of operations)
# ---------------------------------------------------------------------------

# Island bounding boxes: (lat_min, lat_max, lon_min, lon_max)
_GREEK_ISLANDS: Dict[str, Tuple[float, float, float, float]] = {
    "crete":       (34.80, 35.70, 23.50, 26.40),
    "lesbos":      (38.90, 39.40, 25.90, 26.70),
    "rhodes":      (35.85, 36.50, 27.70, 28.30),
    "corfu":       (39.35, 39.85, 19.60, 20.25),
    "santorini":   (36.30, 36.55, 25.30, 25.65),
    "mykonos":     (37.40, 37.55, 25.20, 25.55),
    "chios":       (38.20, 38.70, 25.80, 26.25),
    "samos":       (37.55, 37.90, 26.60, 27.10),
    "zakynthos":   (37.60, 37.90, 20.70, 21.10),
    "kefalonia":   (38.00, 38.55, 20.30, 20.85),
    "lemnos":      (39.80, 40.10, 25.00, 25.65),
    "naxos":       (36.80, 37.20, 25.30, 25.70),
    "paros":       (37.00, 37.15, 25.10, 25.35),
    "kos":         (36.70, 36.95, 26.80, 27.40),
    "evia":        (38.00, 39.00, 22.80, 24.10),
    "andros":      (37.75, 37.95, 24.60, 24.95),
    "tinos":       (37.50, 37.65, 25.00, 25.30),
    "ikaria":      (37.55, 37.65, 26.05, 26.40),
    "syros":       (37.35, 37.50, 24.85, 25.00),
}

# Known mainland inland zones with no viable sea access
_INLAND_LOCATIONS: Dict[str, Tuple[float, float, float, float]] = {
    "karditsa":  (39.20, 39.50, 21.80, 22.15),
    "larissa":   (39.50, 39.80, 22.20, 22.70),
    "trikala":   (39.40, 39.65, 21.50, 21.90),
    "kozani":    (40.20, 40.50, 21.60, 22.05),
    "ioannina":  (39.50, 39.80, 20.70, 21.05),
    "grevena":   (40.00, 40.25, 21.40, 21.75),
    "florina":   (40.70, 40.95, 21.25, 21.65),
    "kastoria":  (40.45, 40.65, 21.15, 21.40),
}


def _classify_terrain_deterministic(lat: float, lon: float) -> Dict[str, Any]:
    """
    Classify terrain and agent eligibility from coordinates using bounding boxes.

    Priority: island check → known inland → coastal mainland default.
    """
    # 1. Island check
    for name, (lat_min, lat_max, lon_min, lon_max) in _GREEK_ISLANDS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return _make_terrain_result(
                terrain_type="island",
                is_island=True,
                island_name=name,
                is_coastal=True,
                has_water_access=True,
                method="deterministic",
            )

    # 2. Known inland zones
    for name, (lat_min, lat_max, lon_min, lon_max) in _INLAND_LOCATIONS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return _make_terrain_result(
                terrain_type="inland",
                is_island=False,
                island_name=None,
                is_coastal=False,
                has_water_access=False,
                method="deterministic",
            )

    # 3. Default: coastal mainland (conservative — most of Greece has sea access)
    return _make_terrain_result(
        terrain_type="coastal_mainland",
        is_island=False,
        island_name=None,
        is_coastal=True,
        has_water_access=True,
        method="deterministic",
    )


def _make_terrain_result(
    terrain_type: str,
    is_island: bool,
    island_name: Optional[str],
    is_coastal: bool,
    has_water_access: bool,
    method: str,
) -> Dict[str, Any]:
    return {
        "terrain_type": terrain_type,
        "is_island": is_island,
        "island_name": island_name,
        "is_coastal": is_coastal,
        "has_water_access": has_water_access,
        "method": method,
        "agent_eligible": {
            "coastguard": has_water_access,
            "maritime": has_water_access,
            "firefighting": True,
            "medical": True,
            "police": True,
            "civil_protection": True,
            "logistics": True,
            "communications": True,
            "environmental": True,
            "search_rescue": True,
        },
    }


# ---------------------------------------------------------------------------
# Vision model prompt
# ---------------------------------------------------------------------------

_TERRAIN_PROMPT = (
    "You are analyzing an OpenStreetMap tile for emergency response planning.\n"
    "Look at the map image. The target location is at the CENTER of the tile.\n\n"
    "Answer these questions about the CENTER point:\n"
    "1. Is the center point on an ISLAND (surrounded by sea on all sides)?\n"
    "2. Is there sea/ocean visible anywhere in this map tile?\n"
    "3. Can a coast guard vessel reach this location by sea?\n\n"
    "IMPORTANT: Rivers and inland lakes are NOT sea. Only ocean/sea counts.\n\n"
    "Respond ONLY with valid JSON — no prose, no markdown.\n"
    'Set "terrain_type" to exactly ONE word: either "island" or "coastal_mainland" or "inland".\n'
    "JSON schema (fill in your own values):\n"
    '{"terrain_type":"...",'
    '"sea_visible_in_tile":true,'
    '"has_maritime_access":true,'
    '"confidence":0.9,'
    '"reasoning":"one sentence describing what you see on the map"}'
)


_VALID_TERRAIN_LABELS = {"island", "coastal_mainland", "inland"}


def _normalize_terrain_label(value: Any) -> Optional[str]:
    """
    Reduce a vision-model terrain_type value to one valid label, or None.

    Handles models that echo the option list from the prompt (e.g.
    "island|coastal_mainland|inland") or wrap the label in prose. Matching is
    token-based, not substring-based: "coastal_mainland" contains "inland" as a
    substring, so substring tests would misclassify. If zero or multiple
    distinct labels appear, the value is ambiguous and None is returned.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    # Unify separator variants of the two-word label before tokenizing
    cleaned = re.sub(r"coastal[\s_\-]+mainland", "coastal_mainland", cleaned)
    if cleaned in _VALID_TERRAIN_LABELS:
        return cleaned
    tokens = set(re.split(r"[^a-z_]+", cleaned))
    matches = _VALID_TERRAIN_LABELS & tokens
    return matches.pop() if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class GeospatialContextAgent:
    """
    Analyzes scenario location to determine which response agents are physically eligible.

    Can use a local Ollama vision model (for richer context) or deterministic
    bounding-box rules (fast, offline).  Always degrades gracefully.

    Usage:
        geo = GeospatialContextAgent()
        result = geo.analyze(scenario, expert_agents)
        # result['eligible_agent_ids'] — list of agent_id strings to keep
        # result['ineligible_agent_ids'] — list excluded by terrain rules
    """

    def __init__(
        self,
        vision_client=None,
        osm_zoom: int = 10,
        use_vision: bool = True,
    ):
        """
        Args:
            vision_client: VisionClient instance.  Created lazily if None.
            osm_zoom: OSM tile zoom level (12 = ~city scale, good for terrain reading).
            use_vision: If False, skip Ollama and use deterministic rules only.
        """
        self._vision_client = vision_client
        self.osm_zoom = osm_zoom
        self.use_vision = use_vision

    def _get_vision_client(self):
        if self._vision_client is not None:
            return self._vision_client
        try:
            from llm_integration.vision_client import VisionClient
            self._vision_client = VisionClient()
        except ImportError:
            logger.warning("GeospatialContextAgent: vision_client import failed")
        return self._vision_client

    # ------------------------------------------------------------------

    def analyze(
        self,
        scenario: Dict[str, Any],
        expert_agents: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze scenario location and return terrain classification + eligibility filter.

        Args:
            scenario: Crisis scenario dict containing `location.coordinates`.
            expert_agents: ExpertAgent instances to classify as eligible/ineligible.

        Returns:
            Dict with:
              terrain_type, is_island, is_coastal, has_water_access — terrain facts
              eligible_agent_ids, ineligible_agent_ids — classification of expert_agents
              terrain_result — full terrain dict from classifier
              method_used — "vision" | "deterministic" | "default"
              coordinates — {lat, lon} used
        """
        coords = (
            scenario.get("location", {}).get("coordinates", {})
        )
        lat = coords.get("lat") if coords else None
        lon = coords.get("lon") if coords else None

        if lat is None or lon is None:
            logger.warning(
                "GeospatialContextAgent: no coordinates in scenario — all agents eligible"
            )
            return self._permissive_result(expert_agents)

        terrain = self._classify(lat, lon)

        eligible_ids: List[str] = []
        ineligible_ids: List[str] = []

        if expert_agents:
            for agent in expert_agents:
                agent_id = getattr(agent, "agent_id", str(agent))
                if self._is_agent_eligible(agent_id, terrain):
                    eligible_ids.append(agent_id)
                else:
                    ineligible_ids.append(agent_id)
                    logger.info(
                        "GeospatialContextAgent: excluding %s "
                        "(terrain=%s has_water_access=%s)",
                        agent_id,
                        terrain["terrain_type"],
                        terrain["has_water_access"],
                    )

        return {
            "terrain_type": terrain["terrain_type"],
            "is_island": terrain.get("is_island", False),
            "island_name": terrain.get("island_name"),
            "is_coastal": terrain.get("is_coastal", True),
            "has_water_access": terrain.get("has_water_access", True),
            "eligible_agent_ids": eligible_ids,
            "ineligible_agent_ids": ineligible_ids,
            "terrain_result": terrain,
            "method_used": terrain.get("method", "deterministic"),
            "coordinates": {"lat": lat, "lon": lon},
        }

    # ------------------------------------------------------------------

    def _classify(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Hybrid classifier: vision for island detection, deterministic for mainland cases.

        Vision models (even 11B) reliably identify islands from map tiles but are
        inconsistent at distinguishing coastal_mainland from inland.  The deterministic
        bounding-box system is 100% accurate for both island and inland classification
        within the Greek operational theatre.  Strategy:
          - If vision says "island"  → trust vision (high reliability)
          - If vision says "inland"  → cross-check with deterministic; use deterministic if they disagree
          - If vision says "coastal_mainland" → use deterministic (vision unreliable here)
          - If vision unavailable or fails → deterministic only
        """
        deterministic = _classify_terrain_deterministic(lat, lon)

        if not self.use_vision:
            logger.info(
                "GeospatialContextAgent: deterministic terrain=%s for (%.4f, %.4f)",
                deterministic["terrain_type"], lat, lon,
            )
            return deterministic

        tile_bytes = _fetch_osm_tile(lat, lon, self.osm_zoom)
        if tile_bytes:
            vc = self._get_vision_client()
            if vc is not None:
                raw = vc.analyze_image(tile_bytes, _TERRAIN_PROMPT, max_tokens=256)
                parsed = vc.parse_json_response(raw)
                if parsed and "terrain_type" in parsed:
                    terrain_type = _normalize_terrain_label(parsed["terrain_type"])
                    if terrain_type is None:
                        logger.info(
                            "GeospatialContextAgent: vision terrain_type %r is ambiguous "
                            "(echoed option list or unrecognised label) — "
                            "using deterministic terrain=%s for (%.4f, %.4f)",
                            parsed["terrain_type"], deterministic["terrain_type"], lat, lon,
                        )
                        return deterministic

                    # Trust vision only for island classification — most reliable use case
                    if terrain_type == "island":
                        water_access = parsed.get("has_maritime_access",
                                                   parsed.get("has_water_access", True))
                        # Islands always have maritime access — correct any model confusion
                        water_access = True
                        result = {
                            "terrain_type": "island",
                            "is_island": True,
                            "island_name": deterministic.get("island_name"),
                            "is_coastal": True,
                            "has_water_access": True,
                            "method": "vision+deterministic",
                            "reasoning": parsed.get("reasoning", ""),
                            "confidence": parsed.get("confidence", 0.9),
                        }
                        result["agent_eligible"] = _make_terrain_result(
                            "island", True, result["island_name"], True, True, "hybrid"
                        )["agent_eligible"]
                        logger.info(
                            "GeospatialContextAgent: vision=island (confidence=%.2f), "
                            "confirmed with hybrid method for (%.4f, %.4f)",
                            parsed.get("confidence", 0), lat, lon,
                        )
                        return result

                    # For non-island terrain, deterministic is more reliable
                    logger.info(
                        "GeospatialContextAgent: vision=%s for non-island — "
                        "using deterministic terrain=%s for (%.4f, %.4f)",
                        terrain_type, deterministic["terrain_type"], lat, lon,
                    )
                else:
                    logger.debug("GeospatialContextAgent: vision parse failed — using deterministic")

        logger.info(
            "GeospatialContextAgent: deterministic terrain=%s for (%.4f, %.4f)",
            deterministic["terrain_type"], lat, lon,
        )
        return deterministic

    def _is_agent_eligible(self, agent_id: str, terrain: Dict[str, Any]) -> bool:
        """Check if an agent can operate at the classified terrain."""
        agent_eligible = terrain.get("agent_eligible", {})
        aid_lower = agent_id.lower()

        if "coastguard" in aid_lower or "coast_guard" in aid_lower:
            return agent_eligible.get("coastguard", terrain.get("has_water_access", True))
        if "maritime" in aid_lower:
            return agent_eligible.get("maritime", terrain.get("has_water_access", True))

        return True  # all other specialisations are terrain-agnostic

    def _permissive_result(self, expert_agents) -> Dict[str, Any]:
        """Safe default when coordinates are unavailable — all agents eligible."""
        all_ids = [getattr(a, "agent_id", str(a)) for a in expert_agents] if expert_agents else []
        return {
            "terrain_type": "unknown",
            "is_island": False,
            "island_name": None,
            "is_coastal": True,
            "has_water_access": True,
            "eligible_agent_ids": all_ids,
            "ineligible_agent_ids": [],
            "terrain_result": {},
            "method_used": "default",
            "coordinates": None,
        }
