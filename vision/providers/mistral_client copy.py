"""MistralVLMClient — safe, no-reasoning, pure-OCR extraction.

Uses mistral-large-4 (natively multimodal). Every request explicitly sets
reasoning_effort="none" (with a `reasoning={"effort": "none"}` fallback)
so no thinking tokens are ever billed and the full completion budget goes
to the JSON output.

Coordinate math identical to GeminiVisionClient (the proven 96% path):
image is a padded square, model returns bbox_1000 [ymin, xmin, ymax, xmax],
parse reverses padding to page points.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time

from PIL import Image

from vision.base import VisionModelClient
from vision.providers.mistral_prompts import DRAWING_PROMPT

log = logging.getLogger(__name__)


# Models DEFINITELY wrong for this task (different endpoint / text-only).
_BLOCK_MODELS = (
    "mistral-ocr",
    "magistral",
    "codestral",
    "mistral-embed",
    "mistral-moderation",
)

_TB_TYPES = {
    "part_number", "material_description", "material_specification",
    "envelope", "mass", "surface_protection",
}

_TB_REGION_X_MIN = 0.55
_TB_REGION_Y_MIN = 0.60


def _resolve_mistral_class():
    candidates = [
        ("mistralai", "Mistral"),
        ("mistralai.client", "Mistral"),
        ("mistralai.client.sdk", "Mistral"),
        ("mistralai.sdk", "Mistral"),
        ("mistralai.client._client", "Mistral"),
    ]
    errors = []
    for mod_path, name in candidates:
        try:
            mod = __import__(mod_path, fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls
        except Exception as e:
            errors.append(f"{mod_path}: {type(e).__name__}: {e}")
    raise ImportError(
        "Mistral class not importable from installed mistralai SDK.\n"
        "Tried:\n  " + "\n  ".join(errors)
    )


class MistralVLMClient(VisionModelClient):
    provider_name = "mistral"

    def __init__(self, api_key: str, model_name: str,
                 timeout_s: int = 180, max_retries: int = 2,
                 use_annotation: bool = False):
        if not api_key:
            raise ValueError("Mistral API key missing (MISTRAL_API_KEY)")

        m = (model_name or "").lower()
        if not m:
            raise ValueError("MODEL_NAME is empty. Set mistral-large-4 in .env")

        if any(t in m for t in _BLOCK_MODELS):
            raise ValueError(
                f"MODEL_NAME={model_name!r} is not usable for drawing "
                f"extraction. Use mistral-large-4."
            )

        Mistral = _resolve_mistral_class()
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.use_annotation = use_annotation

    def extract_candidates(self, image, page_width_pt, page_height_pt,
                           page_num=0, context=None):
        ctx = context or {}
        pad_left    = int(ctx.get("pad_left", 0))
        pad_top     = int(ctx.get("pad_top", 0))
        square_side = int(ctx.get("square_side", image.size[0]))
        content_w   = int(ctx.get("content_w", image.size[0]))
        content_h   = int(ctx.get("content_h", image.size[1]))

        log.info(
            "Mistral VLM: model=%s img=%dx%d content=%dx%d pad=(%d,%d)",
            self.model_name,
            image.size[0], image.size[1],
            content_w, content_h, pad_left, pad_top,
        )

        data_url = self._pil_to_data_url(image)
        try:
            raw = self._call_model(data_url)
        except Exception as exc:
            log.error("Mistral VLM extraction failed: %s", exc)
            return []

        cands = self._parse(
            raw, page_width_pt, page_height_pt,
            content_w, content_h, square_side, pad_left, pad_top,
        )
        log.info("Mistral VLM: %d candidates", len(cands))
        return cands

    @staticmethod
    def _pil_to_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    # -----------------------------------------------------------------
    # Safe call: reasoning off, multi-shape fallback, cost-aware
    # -----------------------------------------------------------------
    def _call_model(self, data_url: str) -> str:
        # Each tuple is a full request variant. Order = cheapest / most
        # likely to succeed first. Every variant carries BOTH
        # reasoning_effort="none" and reasoning={"effort": "none"} so the
        # API disables thinking regardless of which knob it reads.
        base_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,  # JSON output fits comfortably; smaller
                                 # budget also caps accidental reasoning.
            "reasoning_effort": "none",
            "reasoning": {"effort": "none"},
        }

        attempts = [
            # 1: standard string URL + JSON mode
            (self._build_messages(data_url, True),
             {**base_params, "response_format": {"type": "json_object"}},
             "str-url+json+no-reason"),
            # 2: string URL, no JSON mode (some SDK versions reject both
            #    response_format and reasoning fields together)
            (self._build_messages(data_url, True),
             dict(base_params),
             "str-url-no-json+no-reason"),
            # 3: object URL + JSON mode
            (self._build_messages(data_url, False),
             {**base_params, "response_format": {"type": "json_object"}},
             "obj-url+json+no-reason"),
        ]

        max_attempts = min(len(attempts), self.max_retries + 2)
        last_err = None

        for idx, (messages, params, tag) in enumerate(attempts[:max_attempts]):
            try:
                log.info("Mistral VLM attempt %d [%s]", idx + 1, tag)
                resp = self.client.chat.complete(
                    model=self.model_name, messages=messages, **params,
                )
                content = self._extract_content(resp)

                if content.strip():
                    self._log_usage(tag, resp, ok=True)
                    return content

                self._log_empty_response(tag, resp)
                last_err = RuntimeError(f"Empty content [{tag}]")
            except Exception as exc:
                log.warning("Mistral VLM [%s] exception: %s", tag, exc)
                last_err = exc

        raise RuntimeError(
            f"Mistral VLM exhausted {max_attempts} attempts for "
            f"model={self.model_name}. Last: {last_err}"
        )

    @staticmethod
    def _build_messages(data_url: str, string_url: bool):
        img = data_url if string_url else {"url": data_url}
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": DRAWING_PROMPT},
                {"type": "image_url", "image_url": img},
            ],
        }]

    @staticmethod
    def _extract_content(resp) -> str:
        try:
            choice = resp.choices[0]
        except Exception:
            return ""
        msg = getattr(choice, "message", None)
        if msg is None:
            return ""
        content = getattr(msg, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for chunk in content:
                if isinstance(chunk, str):
                    parts.append(chunk)
                elif isinstance(chunk, dict):
                    parts.append(chunk.get("text", "") or "")
                else:
                    t = getattr(chunk, "text", None)
                    if t:
                        parts.append(str(t))
                    else:
                        try:
                            parts.append(
                                chunk.model_dump().get("text", "") or ""
                            )
                        except Exception:
                            pass
            return "".join(parts)
        if isinstance(content, dict):
            return content.get("text", "") or ""
        return str(content)

    @staticmethod
    def _reasoning_tokens(resp) -> int:
        try:
            usage = getattr(resp, "usage", None)
            if usage is None:
                return 0
            details = getattr(usage, "completion_tokens_details", None)
            if details is None:
                return 0
            if isinstance(details, dict):
                return int(details.get("reasoning_tokens", 0) or 0)
            return int(getattr(details, "reasoning_tokens", 0) or 0)
        except Exception:
            return 0

    @classmethod
    def _log_usage(cls, tag: str, resp, ok: bool):
        try:
            usage = getattr(resp, "usage", None)
            rt = cls._reasoning_tokens(resp)
            ct = getattr(usage, "completion_tokens", None) if usage else None
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            if rt > 0:
                log.warning(
                    "Mistral VLM [%s] ok=%s reasoning_tokens=%d "
                    "(expected 0 — reasoning was not disabled on this "
                    "endpoint!)",
                    tag, ok, rt,
                )
            else:
                log.info(
                    "Mistral VLM [%s] ok=%s prompt=%s completion=%s "
                    "reasoning=%d",
                    tag, ok, pt, ct, rt,
                )
        except Exception:
            pass

    @classmethod
    def _log_empty_response(cls, tag: str, resp):
        try:
            choice = resp.choices[0]
            cls._log_usage(tag, resp, ok=False)
            log.warning(
                "Mistral VLM [%s] EMPTY content: finish_reason=%r "
                "model=%r",
                tag,
                getattr(choice, "finish_reason", None),
                getattr(resp, "model", None),
            )
        except Exception as exc:
            log.warning("Mistral VLM [%s] empty, dump failed: %s",
                        tag, exc)

    @staticmethod
    def _parse(raw, page_w, page_h,
               content_w, content_h, square_side, pad_left, pad_top):
        raw = (raw or "").strip()
        if not raw:
            log.warning("Mistral VLM _parse: empty raw")
            return []

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            data = json.loads(raw)
        except Exception:
            log.warning("Mistral VLM non-JSON (%d chars): %r",
                        len(raw), raw[:300])
            return []

        sx = page_w / float(content_w) if content_w else 1.0
        sy = page_h / float(content_h) if content_h else 1.0

        out = []
        tb_warned = 0
        for it in data.get("candidates", []):
            text = (it.get("text") or "").strip()
            if not text or text in ("-", "\u2013", "\u2014"):
                continue

            b = it.get("bbox_1000")
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            try:
                ymin = max(0.0, min(1000.0, float(b[0])))
                xmin = max(0.0, min(1000.0, float(b[1])))
                ymax = max(0.0, min(1000.0, float(b[2])))
                xmax = max(0.0, min(1000.0, float(b[3])))
            except (TypeError, ValueError):
                continue

            sq_x1 = (xmin / 1000.0) * square_side
            sq_y1 = (ymin / 1000.0) * square_side
            sq_x2 = (xmax / 1000.0) * square_side
            sq_y2 = (ymax / 1000.0) * square_side

            img_x1 = max(0.0, min(float(content_w), sq_x1 - pad_left))
            img_y1 = max(0.0, min(float(content_h), sq_y1 - pad_top))
            img_x2 = max(0.0, min(float(content_w), sq_x2 - pad_left))
            img_y2 = max(0.0, min(float(content_h), sq_y2 - pad_top))

            if img_x2 < img_x1: img_x1, img_x2 = img_x2, img_x1
            if img_y2 < img_y1: img_y1, img_y2 = img_y2, img_y1
            if (img_x2 - img_x1) < 1 and (img_y2 - img_y1) < 1:
                continue

            x1 = img_x1 * sx
            y1 = img_y1 * sy
            x2 = img_x2 * sx
            y2 = img_y2 * sy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            typ = (it.get("type") or "feature").strip().lower()
            par = (it.get("parameter") or "Dimn").strip()

            if typ == "title_block":
                if par == "Part No.":
                    typ = "part_number"
                elif par == "Envelope":
                    typ = "envelope"
                elif par == "Final mass":
                    typ = "mass"
                elif par == "Surface Protection":
                    typ = "surface_protection"
                elif par in ("Thk", "Material Description"):
                    typ = "material_description"
                elif par in ("Grade", "Material Specification"):
                    typ = "material_specification"

            if typ in _TB_TYPES:
                nx = cx / page_w if page_w else 0.5
                ny = cy / page_h if page_h else 0.5
                if nx < _TB_REGION_X_MIN or ny < _TB_REGION_Y_MIN:
                    tb_warned += 1
                    log.debug(
                        "Mistral VLM: %s at (%.0f%%, %.0f%%) outside "
                        "title-block region — will re-anchor bubble",
                        typ, nx * 100, ny * 100,
                    )

            out.append({
                "text":          text,
                "type":          typ,
                "parameter":     par,
                "specification": (it.get("specification") or text).strip(),
                "tolerance":     (it.get("tolerance") or "").strip(),
                "bbox":          (x1, y1, x2, y2),
                "anchor_x":      cx,
                "anchor_y":      cy,
                "source":        "mistral",
            })

        if tb_warned:
            log.info("Mistral VLM: %d title-block candidates had bbox "
                     "outside bottom-right region", tb_warned)
        return out