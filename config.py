"""config.py — env loading + Settings."""
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
    try:
        v = _get(name)
        return int(v) if v else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        v = _get(name)
        return float(v) if v else default
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    v = _get(name).lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _pick_api_key(provider: str) -> str:
    if provider == "mistral":
        return _get("MISTRAL_API_KEY")
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
    extraction_mode: str
    vision_dpi: int
    vision_timeout_s: int
    vision_max_retries: int
    vector_min_spans: int
    merge_spatial_radius_pt: float
    merge_text_similarity: float
    log_level: str
    input_dir: str
    vision_padding: bool
    mistral_annotation: bool

    @property
    def provider_configured(self) -> bool:
        return self.provider not in ("", "none", "off")

    @property
    def vision_enabled(self) -> bool:
        return self.provider_configured and bool(self.api_key)

    def redacted_summary(self) -> dict:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "extraction_mode": self.extraction_mode,
            "vision_dpi": self.vision_dpi,
            "vision_timeout_s": self.vision_timeout_s,
            "vector_min_spans": self.vector_min_spans,
            "vision_enabled": self.vision_enabled,
            "vision_padding": self.vision_padding,
            "mistral_annotation": self.mistral_annotation,
            "api_key_present": bool(self.api_key),
        }


# def load_settings() -> Settings:
#     provider = _get("MODEL_PROVIDER", "mistral").lower()

#     if provider == "mistral":
#         default_model = "mistral-large-4"
#     elif provider == "groq":
#         default_model = "qwen-2.5-vl"
#     elif provider == "gemini":
#         default_model = "gemini-2.0-flash"
#     else:
#         default_model = ""

#     default_padding = provider == "gemini"
#     default_annotation = False

#     return Settings(
#         provider=provider,
#         model_name=_get("MODEL_NAME", default_model),
#         api_key=_pick_api_key(provider),
#         extraction_mode=_get("EXTRACTION_MODE", "vision_only").lower(),
#         vision_dpi=_get_int("VISION_DPI", 300),
#         vision_timeout_s=_get_int("VISION_TIMEOUT", 180),
#         vision_max_retries=_get_int("VISION_MAX_RETRIES", 2),
#         vector_min_spans=_get_int("VECTOR_MIN_SPANS", 10),
#         merge_spatial_radius_pt=_get_float("MERGE_SPATIAL_RADIUS_PT", 28.0),
#         merge_text_similarity=_get_float("MERGE_TEXT_SIMILARITY", 0.82),
#         log_level=_get("LOG_LEVEL", "INFO").upper(),
#         input_dir=_get("INPUT_DIR", "sources"),
#         vision_padding=_get_bool("VISION_PADDING", default_padding),
#         mistral_annotation=_get_bool("MISTRAL_ANNOTATION", default_annotation),
#     )
def load_settings() -> Settings:
    provider = _get("MODEL_PROVIDER", "mistral").lower()

    if provider == "mistral":
        default_model = "pixtral-12b-2409"   # safer, non-reasoning
    elif provider == "groq":
        default_model = "qwen-2.5-vl"
    elif provider == "gemini":
        default_model = "gemini-2.0-flash"
    else:
        default_model = ""

    default_padding = provider == "gemini"
    default_annotation = False

    return Settings(
        provider=provider,
        model_name=_get("MODEL_NAME", default_model),
        api_key=_pick_api_key(provider),
        extraction_mode=_get("EXTRACTION_MODE", "vision_only").lower(),
        vision_dpi=_get_int("VISION_DPI", 300),
        vision_timeout_s=_get_int("VISION_TIMEOUT", 180),
        vision_max_retries=_get_int("VISION_MAX_RETRIES", 2),
        vector_min_spans=_get_int("VECTOR_MIN_SPANS", 10),
        merge_spatial_radius_pt=_get_float("MERGE_SPATIAL_RADIUS_PT", 28.0),
        merge_text_similarity=_get_float("MERGE_TEXT_SIMILARITY", 0.82),
        log_level=_get("LOG_LEVEL", "INFO").upper(),
        input_dir=_get("INPUT_DIR", "sources"),
        vision_padding=_get_bool("VISION_PADDING", default_padding),
        mistral_annotation=_get_bool("MISTRAL_ANNOTATION", default_annotation),
    )