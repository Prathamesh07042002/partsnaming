# env loading + redacted summary
"""config.py - centralized environment configuration.

Loads .env from project root (if present) and exposes the settings the
extraction pipeline needs.  Never import this from libraries that need to
be dependency-free (e.g. pdf_extractor.py) — keep it at the pipeline level.

Supported providers:  groq | gemini | none
API keys read (in order, per provider):
  groq   → GROQ_API_KEY   (fallback: OPENAI_API_KEY)
  gemini → GEMINI_API_KEY (fallback: GOOGLE_API_KEY)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _pick_api_key(provider: str) -> str:
    """Select the API key matching the configured provider."""
    if provider == "groq":
        return _get("GROQ_API_KEY") or _get("OPENAI_API_KEY")
    if provider == "gemini":
        return _get("GEMINI_API_KEY") or _get("GOOGLE_API_KEY")
    return ""


@dataclass(frozen=True)
class Settings:
    provider: str
    model_name: str
    api_key: str
    extraction_mode: str              # vector_first | vision_first | hybrid
    vision_dpi: int
    vision_timeout_s: int
    vision_max_retries: int
    vector_min_spans: int
    merge_spatial_radius_pt: float
    merge_text_similarity: float
    log_level: str
    input_dir: str

    @property
    def provider_configured(self) -> bool:
        return self.provider not in ("", "none", "off")

    @property
    def vision_enabled(self) -> bool:
        return self.provider_configured and bool(self.api_key)

    def redacted_summary(self) -> dict:
        """ Safe-to-log summary — API key NEVER included."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "extraction_mode": self.extraction_mode,
            "vision_dpi": self.vision_dpi,
            "vision_timeout_s": self.vision_timeout_s,
            "vector_min_spans": self.vector_min_spans,
            "vision_enabled": self.vision_enabled,
            "api_key_present": bool(self.api_key),
        }


def load_settings() -> Settings:
    provider = _get("MODEL_PROVIDER", "groq").lower()

    # Default model names per provider (env override wins).
    if provider == "groq":
        # Any Qwen VL model string supported on Groq Cloud.
        # Override via MODEL_NAME in .env — e.g. "qwen/qwen3-vl-32b-instruct"
        # or whatever Groq exposes.
        default_model = "qwen-2.5-vl"
    elif provider == "gemini":
        default_model = "gemini-2.0-flash"
    else:
        default_model = ""

    return Settings(
        provider=provider,
        model_name=_get("MODEL_NAME", default_model),
        api_key=_pick_api_key(provider),
        extraction_mode=_get("EXTRACTION_MODE", "hybrid").lower(),
        vision_dpi=_get_int("VISION_DPI", 150),
        vision_timeout_s=_get_int("VISION_TIMEOUT", 120),
        vision_max_retries=_get_int("VISION_MAX_RETRIES", 2),
        vector_min_spans=_get_int("VECTOR_MIN_SPANS", 10),
        merge_spatial_radius_pt=_get_float("MERGE_SPATIAL_RADIUS_PT", 28.0),
        merge_text_similarity=_get_float("MERGE_TEXT_SIMILARITY", 0.82),
        log_level=_get("LOG_LEVEL", "INFO").upper(),
        input_dir=_get("INPUT_DIR", "sources"),
    )