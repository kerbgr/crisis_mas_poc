"""
VisionClient — Ollama multimodal inference for geospatial and camera-feed analysis.

Uses Ollama's OpenAI-compatible API at localhost:11434/v1 with vision-capable models
(minicpm-v, llava, moondream, etc.).  Falls back gracefully when Ollama is
not running so the rest of the pipeline is unaffected.

Setup:
    ollama pull minicpm-v          # 5.5 GB, strong structured-output support
    ollama pull llava:7b           # 4 GB, good instruction following
    ollama pull moondream          # 1.9 GB, fast but limited JSON output

Known issues:
    llama3.2-vision is broken in Ollama 0.30.x (mllama architecture regression,
    tracked at github.com/ollama/ollama/issues/16490).
"""

import base64
import json
import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "minicpm-v:latest"
DEFAULT_TIMEOUT = 90   # vision inference is slower than text
WARMUP_TIMEOUT = 120   # seconds to wait for model to load cold


class VisionClient:
    """
    Thin wrapper around Ollama's OpenAI-compatible vision API.

    Sends images as base64-encoded data URLs in the `image_url` content block
    format accepted by Ollama's /v1/chat/completions endpoint.

    Warmup: Ollama loads vision models lazily. Call warmup() (or pass
    auto_warmup=True) to block until the model is in memory before
    the first real inference request.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        auto_warmup: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._ollama_root = self.base_url.removesuffix("/v1").rstrip("/")
        self._warmed_up = False

        if auto_warmup:
            self.warmup()

    # ------------------------------------------------------------------
    # Availability and warmup (3-tier, provider-agnostic)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Tier 1 — server reachable. Works for any OpenAI-compat provider."""
        try:
            r = requests.get(f"{self.base_url}/models", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def is_serving(self) -> bool:
        """Tier 2 — at least one model is loaded and ready to serve."""
        try:
            r = requests.get(f"{self.base_url}/models", timeout=5)
            if r.status_code == 200:
                return len(r.json().get("data", [])) > 0
        except Exception:
            pass
        return False

    def _is_ollama(self) -> bool:
        """Detect Ollama by probing its native /api/ps endpoint."""
        try:
            r = requests.get(f"{self._ollama_root}/api/ps", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def is_model_loaded(self) -> bool:
        """Tier 3 (Ollama only) — specific model is resident in RAM."""
        try:
            r = requests.get(f"{self._ollama_root}/api/ps", timeout=5)
            if r.status_code == 200:
                loaded = [m.get("name", "") for m in r.json().get("models", [])]
                return any(self.model in name or name in self.model for name in loaded)
        except Exception:
            pass
        return False

    def warmup(self, timeout: int = WARMUP_TIMEOUT) -> bool:
        """
        Ensure the vision model is ready before the first inference call.

        Three tiers — each builds on the previous:
          Tier 1 (all providers): server is reachable (GET /v1/models).
          Tier 2 (all providers): at least one model is loaded and listed.
          Tier 3 (Ollama only):   this specific model is resident in RAM
                                  (GET /api/ps); trigger loading if not.

        LM Studio satisfies tiers 1+2 as soon as the user loads a model in
        the UI — no tier-3 polling needed.  Ollama loads lazily on first
        request, so tier 3 fires a background trigger and polls /api/ps.

        Returns:
            True if model is ready, False on timeout or server unavailable.
        """
        if self._warmed_up:
            return True

        # --- Tier 1: server reachable ---
        if not self.is_available():
            logger.warning(
                "VisionClient: server not reachable at %s — warmup skipped",
                self.base_url,
            )
            return False

        # --- Tier 2: model(s) serving ---
        if self.is_serving():
            # For non-Ollama providers (LM Studio etc.) this is sufficient.
            if not self._is_ollama():
                logger.info("VisionClient: server ready at %s", self.base_url)
                self._warmed_up = True
                return True
        else:
            # No model loaded yet — wait for tier 2 before proceeding to tier 3
            logger.info(
                "VisionClient: waiting for a model to be served at %s (timeout=%ds)",
                self.base_url, timeout,
            )
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self.is_serving():
                    break
                time.sleep(3)
            else:
                logger.warning("VisionClient: no model became available within %ds", timeout)
                return False

        # --- Tier 3: Ollama — specific model in RAM ---
        if self.is_model_loaded():
            self._warmed_up = True
            return True

        logger.info(
            "VisionClient: loading %s into Ollama memory (timeout=%ds)",
            self.model, timeout,
        )
        deadline = time.time() + timeout

        # Fire a background request to trigger model loading
        import threading
        def _trigger():
            try:
                requests.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "stream": False,
                    },
                    timeout=timeout,
                )
            except Exception:
                pass
        threading.Thread(target=_trigger, daemon=True).start()

        while time.time() < deadline:
            if self.is_model_loaded():
                logger.info("VisionClient: %s is ready", self.model)
                self._warmed_up = True
                return True
            time.sleep(3)

        logger.warning(
            "VisionClient: %s did not load within %ds — first inference may be slow",
            self.model, timeout,
        )
        return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        max_tokens: int = 512,
        mime_type: str = "image/png",
    ) -> Optional[str]:
        """
        Send image bytes + text prompt to the vision model.

        Args:
            image_bytes: Raw image bytes (PNG, JPEG, etc.)
            prompt: Instruction for the model
            max_tokens: Maximum response tokens
            mime_type: MIME type for the base64 data URL

        Returns:
            Model response text, or None if Ollama is unavailable.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False,
        }
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            logger.warning(
                "VisionClient: Ollama not reachable at %s — skipping vision analysis",
                self.base_url,
            )
            return None
        except requests.exceptions.Timeout:
            logger.warning(
                "VisionClient: request timed out after %ds", self.timeout
            )
            return None
        except Exception as e:
            logger.warning("VisionClient: request failed — %s", e)
            return None

    def analyze_image_url(
        self,
        url: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """Fetch image from HTTP URL then analyze it."""
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
            return self.analyze_image(r.content, prompt, max_tokens, mime_type=content_type)
        except Exception as e:
            logger.warning("VisionClient: failed to fetch image from %s — %s", url, e)
            return None

    def parse_json_response(self, response_text: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Extract and parse the first JSON object found in a model response.

        Vision models sometimes wrap JSON in prose — this strips the surrounding text.
        """
        if not response_text:
            return None
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            return json.loads(response_text[start:end])
        except json.JSONDecodeError:
            logger.debug("VisionClient: JSON parse failed on: %s", response_text[start:end][:200])
            return None
