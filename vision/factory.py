"""vision.factory - resolve a VisionModelClient from Settings."""
from __future__ import annotations

import logging

from config import Settings
from vision.base import VisionModelClient

log = logging.getLogger(__name__)


def build_vision_client(settings: Settings) -> VisionModelClient | None:
    """Return a client for the configured provider, or None.

    Unknown providers / missing API key → returns None and logs clearly.
    """
    provider = (settings.provider or "").lower()
    if provider in ("", "none", "off"):
        log.info("Vision provider disabled (MODEL_PROVIDER=%r)", provider)
        return None

    if not settings.api_key:
        log.warning(
            "Vision provider %r configured but API key missing. "
            "Vision source will be skipped.", provider,
        )
        return None

    # ----- Groq (Qwen VL) -----
    if provider == "groq":
        from vision.providers.groq_client import GroqVisionClient
        return GroqVisionClient(
            api_key=settings.api_key,
            model_name=settings.model_name,
            timeout_s=settings.vision_timeout_s,
            max_retries=settings.vision_max_retries,
        )

    # ----- Gemini (kept for future use) -----
    if provider == "gemini":
        from vision.providers.gemini_client import GeminiVisionClient
        return GeminiVisionClient(
            api_key=settings.api_key,
            model_name=settings.model_name,
            timeout_s=settings.vision_timeout_s,
            max_retries=settings.vision_max_retries,
        )

    # ----- Future provider slots -----
    # if provider == "qwen":
    #     from vision.providers.qwen_client import QwenVisionClient
    #     return QwenVisionClient(api_key=settings.api_key,
    #                             model_name=settings.model_name)

    log.error("Unknown MODEL_PROVIDER=%r — vision disabled", provider)
    return None