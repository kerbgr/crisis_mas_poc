"""
CameraFeedAgent — Bronze-level visual situation awareness from camera feeds.

Analyzes images (static URLs or RTSP video frames) using a local vision model and
produces structured situational reports that the coordinator injects into scenario
context before expert agents evaluate the crisis.

PRIMARY USE CASES:
- Santorini harbor cameras: wave anomaly / tsunami indicator detection
- Evacuation assembly points: crowd density monitoring
- Wildfire perimeter cameras: smoke/fire spread assessment
- Flood sensor cameras: water level estimation from bridge/road views

ARCHITECTURE ROLE:
Bronze (operational) agent — raw sensor intelligence.  Not an ExpertAgent subclass;
used by CoordinatorAgent as a pre-processing step (Step 0) to enrich scene context.
"""

import base64
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_GENERAL_PROMPT = (
    "You are an emergency management analyst reviewing a surveillance camera image.\n"
    "Context: {context}\n\n"
    "Analyze the image and return ONLY valid JSON:\n"
    "{{\n"
    '  "crowd_density": "none|low|medium|high|critical",\n'
    '  "crowd_count_estimate": 0,\n'
    '  "water_anomaly_detected": false,\n'
    '  "wave_detected": false,\n'
    '  "wave_height_estimate_m": null,\n'
    '  "smoke_visible": false,\n'
    '  "fire_visible": false,\n'
    '  "structural_damage_visible": false,\n'
    '  "vehicle_congestion": "none|low|medium|high",\n'
    '  "situation_severity": "normal|elevated|high|critical",\n'
    '  "immediate_hazards": [],\n'
    '  "situation_summary": "one-sentence description",\n'
    '  "confidence": 0.0\n'
    "}}"
)

_TSUNAMI_PROMPT = (
    "You are analyzing a coastal/harbor surveillance camera for tsunami early-warning signs.\n"
    "Look specifically for:\n"
    "- Sea withdrawal: water receding rapidly from the shoreline\n"
    "- Abnormal wave patterns, unusual turbulence, or wall of water approaching\n"
    "- Debris floating in harbor water\n"
    "- Vessels swinging wildly, listing, or breaking moorings\n"
    "- People running away from the waterfront\n\n"
    "Return ONLY valid JSON:\n"
    "{{\n"
    '  "tsunami_indicators_present": false,\n'
    '  "water_withdrawal_observed": false,\n'
    '  "abnormal_wave_pattern": false,\n'
    '  "debris_in_water": false,\n'
    '  "vessels_distressed": false,\n'
    '  "warning_level": "none|watch|warning|emergency",\n'
    '  "crowd_density_at_shore": "none|low|medium|high|critical",\n'
    '  "wave_detected": false,\n'
    '  "wave_height_estimate_m": null,\n'
    '  "situation_summary": "one-sentence description",\n'
    '  "confidence": 0.0,\n'
    '  "recommended_action": "monitor|alert_authorities|immediate_evacuation"\n'
    "}}"
)

_CROWD_PROMPT = (
    "You are analyzing a surveillance camera for crowd monitoring during an emergency.\n"
    "Context: {context}\n\n"
    "Assess crowd size, movement, and safety indicators.\n"
    "Return ONLY valid JSON:\n"
    "{{\n"
    '  "crowd_density": "none|low|medium|high|critical",\n'
    '  "crowd_count_estimate": 0,\n'
    '  "crowd_moving_direction": "unclear|stable|away_from_hazard|toward_hazard",\n'
    '  "panic_indicators": false,\n'
    '  "bottleneck_observed": false,\n'
    '  "stampede_risk": false,\n'
    '  "situation_severity": "normal|elevated|high|critical",\n'
    '  "situation_summary": "one-sentence description",\n'
    '  "confidence": 0.0\n'
    "}}"
)

_FIRE_PROMPT = (
    "You are analyzing a camera image for wildfire or structural fire assessment.\n"
    "Context: {context}\n\n"
    "Return ONLY valid JSON:\n"
    "{{\n"
    '  "fire_visible": false,\n'
    '  "smoke_visible": false,\n'
    '  "smoke_color": "none|white|gray|black|brown",\n'
    '  "fire_size_estimate": "none|small|moderate|large|extreme",\n'
    '  "spread_direction": "unknown",\n'
    '  "structures_threatened": false,\n'
    '  "people_visible_in_danger": false,\n'
    '  "situation_severity": "normal|elevated|high|critical",\n'
    '  "situation_summary": "one-sentence description",\n'
    '  "confidence": 0.0\n'
    "}}"
)

_PROMPTS = {
    "general": _GENERAL_PROMPT,
    "tsunami": _TSUNAMI_PROMPT,
    "crowd": _CROWD_PROMPT,
    "fire": _FIRE_PROMPT,
}


# ---------------------------------------------------------------------------
# Image acquisition helpers
# ---------------------------------------------------------------------------

def _fetch_image_bytes(source: str) -> Optional[bytes]:
    """
    Acquire image bytes from URL, RTSP stream, or local file.

    Tries OpenCV for RTSP; requests for HTTP(S); direct file read otherwise.
    Returns None on any failure — caller handles graceful degradation.
    """
    if source.startswith("rtsp://"):
        return _grab_rtsp_frame(source)
    if source.startswith(("http://", "https://")):
        return _fetch_http_image(source)
    return _read_local_file(source)


def _fetch_http_image(url: str) -> Optional[bytes]:
    try:
        import requests
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.warning("CameraFeedAgent: HTTP fetch failed for %s — %s", url, e)
        return None


def _grab_rtsp_frame(rtsp_url: str) -> Optional[bytes]:
    """Grab a single frame from an RTSP stream via OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.warning("CameraFeedAgent: cannot open RTSP stream %s", rtsp_url)
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            logger.warning("CameraFeedAgent: no frame received from %s", rtsp_url)
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()
    except ImportError:
        logger.warning(
            "CameraFeedAgent: opencv-python not installed — cannot read RTSP stream. "
            "Install with: pip install opencv-python"
        )
        return None
    except Exception as e:
        logger.warning("CameraFeedAgent: RTSP frame grab failed — %s", e)
        return None


def _read_local_file(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning("CameraFeedAgent: cannot read file %s — %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class CameraFeedAgent:
    """
    Bronze-level agent: analyzes camera images / RTSP feeds for situational awareness.

    Produces structured JSON reports that the coordinator injects into scene context
    before expert agents evaluate the scenario.  Designed to fail safely — always
    returns a dict even when the vision model or feed is unavailable.

    Modes:
        general  — multi-purpose situational report
        tsunami  — harbor/coastal wave anomaly and withdrawal detection
        crowd    — density, movement, panic indicators
        fire     — smoke color, spread, structures at risk
    """

    AGENT_ID = "camera_feed_bronze"
    LEVEL = "bronze"

    def __init__(
        self,
        vision_client=None,
        mode: str = "general",
    ):
        """
        Args:
            vision_client: VisionClient instance.  Created lazily if None.
            mode: Default analysis mode ("general" | "tsunami" | "crowd" | "fire").
        """
        self._vision_client = vision_client
        self.mode = mode
        self.last_report: Optional[Dict[str, Any]] = None

    def _get_vision_client(self):
        if self._vision_client is not None:
            return self._vision_client
        try:
            from llm_integration.vision_client import VisionClient
            self._vision_client = VisionClient()
        except ImportError:
            logger.warning("CameraFeedAgent: VisionClient import failed")
        return self._vision_client

    # ------------------------------------------------------------------

    def analyze_feed(
        self,
        source: str,
        context: str = "",
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a camera feed and return a structured situational report.

        Args:
            source: Image URL, RTSP stream URL, or local file path.
            context: Brief description of what is being monitored (injected into prompt).
            mode: Override default mode for this call.

        Returns:
            Dict containing the situation report.  Always returns a dict.
        """
        effective_mode = mode or self.mode
        report = self._base_report(source, effective_mode)

        image_bytes = _fetch_image_bytes(source)
        if image_bytes is None:
            report["status"] = "feed_unavailable"
            report["situation_summary"] = "Camera feed could not be accessed"
            self.last_report = report
            return report

        vc = self._get_vision_client()
        if vc is None:
            report["status"] = "vision_model_unavailable"
            report["situation_summary"] = "Vision model (Ollama) not running"
            self.last_report = report
            return report

        prompt = _PROMPTS.get(effective_mode, _GENERAL_PROMPT).format(
            context=context or "emergency monitoring camera"
        )
        raw = vc.analyze_image(image_bytes, prompt, max_tokens=512, mime_type="image/jpeg")
        parsed = vc.parse_json_response(raw)

        if parsed:
            report.update(parsed)
            report["status"] = "ok"
        else:
            report["status"] = "parse_failed"
            report["situation_summary"] = "Vision analysis complete but response parsing failed"
            report["raw_response"] = raw

        self.last_report = report
        logger.info(
            "CameraFeedAgent: %s analysis complete — severity=%s confidence=%.2f",
            effective_mode,
            report.get("situation_severity", "unknown"),
            float(report.get("confidence", 0)),
        )
        return report

    def analyze_feeds(
        self,
        feeds: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Analyze a list of feed descriptors from scenario JSON.

        Each descriptor: {"url": "...", "mode": "tsunami", "label": "Athinios Harbor"}

        Returns list of reports in the same order.
        """
        results = []
        for feed in feeds:
            url = feed.get("url", "")
            mode = feed.get("mode", self.mode)
            label = feed.get("label", url)
            if not url:
                continue
            report = self.analyze_feed(url, context=label, mode=mode)
            report["label"] = label
            results.append(report)
        return results

    # ------------------------------------------------------------------

    def to_scenario_context(self, reports: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Format camera feed reports as text suitable for injection into LLM prompts.

        Args:
            reports: List of reports.  Uses [self.last_report] if None.

        Returns:
            Human-readable situation summary string, or "" if no reports.
        """
        if reports is None:
            if self.last_report is None:
                return ""
            reports = [self.last_report]

        if not reports:
            return ""

        lines = ["[Camera Feed Intelligence]"]
        for r in reports:
            label = r.get("label") or r.get("source", "camera")
            lines.append(f"\n  Feed: {label} ({r.get('mode', 'general')} mode)")
            lines.append(f"  Status: {r.get('status', 'unknown')}")
            lines.append(f"  Severity: {r.get('situation_severity', 'unknown')}")
            lines.append(f"  Summary: {r.get('situation_summary', 'N/A')}")

            # Highlight critical indicators
            if r.get("tsunami_indicators_present"):
                lines.append(f"  *** TSUNAMI INDICATORS — warning_level={r.get('warning_level', '?')}")
            if r.get("wave_detected"):
                h = r.get("wave_height_estimate_m")
                lines.append(f"  *** WAVE DETECTED — est. height {h}m" if h else "  *** WAVE DETECTED")
            if r.get("water_withdrawal_observed"):
                lines.append("  *** SEA WITHDRAWAL OBSERVED")
            if r.get("smoke_visible") or r.get("fire_visible"):
                size = r.get("fire_size_estimate", "unknown")
                lines.append(f"  *** FIRE/SMOKE VISIBLE — size: {size}")
            if r.get("crowd_density") in ("high", "critical"):
                lines.append(f"  *** CROWD DENSITY: {r.get('crowd_density', '').upper()}")
            if r.get("panic_indicators"):
                lines.append("  *** PANIC INDICATORS OBSERVED")
            if r.get("stampede_risk"):
                lines.append("  *** STAMPEDE RISK")

            lines.append(f"  Confidence: {float(r.get('confidence', 0)):.0%}")

        return "\n".join(lines)

    def _base_report(self, source: str, mode: str) -> Dict[str, Any]:
        return {
            "agent_id": self.AGENT_ID,
            "level": self.LEVEL,
            "mode": mode,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            # safe defaults — normal/unknown state
            "crowd_density": "unknown",
            "water_anomaly_detected": False,
            "wave_detected": False,
            "smoke_visible": False,
            "fire_visible": False,
            "situation_severity": "unknown",
            "situation_summary": "Analysis pending",
            "confidence": 0.0,
        }
