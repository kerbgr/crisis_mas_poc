"""
OllamaClient — Ollama as a full text LLM provider.

Ollama runs local models via an OpenAI-compatible API at localhost:11434.
Unlike LM Studio, Ollama requires the model name in every request and loads
models lazily (first request triggers loading, which can take 30-120 seconds).

Setup:
    ollama pull <model>          # e.g. qwen3.5:35b, llama3.1:8b, mistral:7b
    # Ollama starts automatically on macOS via launchd

Usage:
    client = OllamaClient(model="qwen3.5:35b")
    client.warmup()              # wait for model to load before first real request
    response = client.generate_assessment(prompt)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests

from models.data_models import LLMResponse

try:
    from openai import OpenAI, APIError, APIConnectionError
except ImportError:
    raise ImportError(
        "OpenAI package required for Ollama compatibility. "
        "Install with: pip install openai"
    )

# Re-use JSON parsing and validation from LMStudioClient
from llm_integration.lmstudio_client import LMStudioClient

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3.5:35b"
WARMUP_TIMEOUT = 180  # seconds — large models need time


class OllamaClient(LMStudioClient):
    """
    LLM client for Ollama's OpenAI-compatible text API.

    Subclasses LMStudioClient to reuse JSON parsing, retry logic, and
    assessment generation. Key differences from LMStudio:
    - Explicit model name required per request
    - Models load lazily — warmup() blocks until the model is ready
    - Uses /api/ps (native Ollama endpoint) to check load status
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        """
        Args:
            model: Ollama model name including tag (e.g. "qwen3.5:35b").
            base_url: Ollama OpenAI-compat base URL (default: localhost:11434/v1).
        """
        # Derive the native Ollama API root (strips the /v1 suffix)
        self._ollama_root = base_url.rstrip("/").removesuffix("/v1").rstrip("/")

        super().__init__(base_url=base_url, api_key="ollama", model=model)
        logger.info("OllamaClient initialized (model=%s, endpoint=%s)", model, base_url)

    # ------------------------------------------------------------------
    # Model loading / warmup (3-tier, Ollama-specific tier 3)
    # ------------------------------------------------------------------

    def is_model_loaded(self) -> bool:
        """Tier 3 — specific model is resident in Ollama RAM (/api/ps)."""
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
        Ensure the model is ready before the first real request.

        Mirrors the 3-tier logic in VisionClient:
          Tier 1 (inherited from LMStudioClient): server reachable.
          Tier 2 (inherited from LMStudioClient): a model is being served.
          Tier 3 (Ollama-specific): this model is resident in RAM;
                 trigger loading and poll /api/ps if not.

        Returns:
            True if model is ready, False on timeout or server unavailable.
        """
        # Tiers 1 + 2 via parent
        if not super().warmup(timeout=min(timeout, 30)):
            return False

        # Tier 3: model in RAM
        if self.is_model_loaded():
            logger.info("OllamaClient: model %s already in memory", self.model)
            return True

        logger.info(
            "OllamaClient: loading %s into memory (timeout=%ds) — large models may take a while",
            self.model, timeout,
        )
        deadline = time.time() + timeout

        import threading
        def _trigger():
            try:
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
            except Exception:
                pass
        threading.Thread(target=_trigger, daemon=True).start()

        while time.time() < deadline:
            if self.is_model_loaded():
                logger.info("OllamaClient: model %s is ready", self.model)
                return True
            time.sleep(3)

        logger.warning(
            "OllamaClient: model %s did not load within %ds — first call may be slow",
            self.model, timeout,
        )
        return False

    def list_available_models(self) -> List[str]:
        """Return names of all models pulled in Ollama."""
        try:
            r = requests.get(f"{self._ollama_root}/api/tags", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    def is_available(self) -> bool:
        """Return True if Ollama server is reachable."""
        try:
            r = requests.get(f"{self._ollama_root}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
