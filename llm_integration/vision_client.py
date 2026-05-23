"""
VisionClient — Ollama multimodal inference for geospatial and camera-feed analysis.

Uses Ollama's OpenAI-compatible API at localhost:11434/v1 with vision-capable models
(llama3.2-vision, moondream, minicpm-v, etc.).  Falls back gracefully when Ollama is
not running so the rest of the pipeline is unaffected.

Setup:
    ollama pull llama3.2-vision   # 11B, ~7 GB, higher quality
    ollama pull moondream         # 1.9B, ~1.8 GB, much faster
"""

import base64
import json
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2-vision"
DEFAULT_TIMEOUT = 90  # vision inference is slower than text


class VisionClient:
    """
    Thin wrapper around Ollama's OpenAI-compatible vision API.

    Sends images as base64-encoded data URLs in the `image_url` content block
    format accepted by Ollama's /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            r = requests.get(f"{self.base_url}/models", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

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
