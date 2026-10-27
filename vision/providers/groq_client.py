"""GroqVisionClient – Qwen VL models hosted on Groq Cloud.

Uses the OpenAI-compatible API at https://api.groq.com/openai/v1.
Coordinates are requested as 0-1000 normalized bboxes and mapped
linearly back to page points.

Thinking mode is disabled via `reasoning_effort="none"` so the model
does not burn tokens on internal reasoning loops.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import re
import time

from PIL import Image

from vision.base import VisionModelClient

log = logging.getLogger(__name__)

_CONSECUTIVE_FAILURE_LIMIT = int(os.environ.get("VISION_BREAKER_LIMIT", "4"))
_consecutive_failures = 0


_PROMPT = """\
You are a mechanical-engineering drawing inspector.
Return EVERY visible inspection characteristic as structured JSON.
INCLUDE: linear dimensions, tolerances, holes, radii, angles, part numbers, mass.
EXCLUDE: title/scale text, frame grids, boilerplate notes.

Return a single JSON object with key "candidates" (an array).
Each element MUST contain exactly:
  text          (string)
  type          (string: feature | material_description | material_specification | part_number | envelope | mass | surface_protection | note)
  parameter     (string: Dimn | Hole | Radius | Angle | Part No. | Thk | Envelope | Final mass | Surface Protection)
  specification (string, value without tolerance, e.g. "Ø20")
  tolerance     (string, e.g. "±0.1", or "" if none)
  bbox_1000     (array of exactly 4 numbers: [x_min, y_min, x_max, y_max])
  confidence    (number 0..1)

CRITICAL COORDINATE RULE:
bbox_1000 coordinates MUST be normalized to a 0-1000 scale where:
  [0, 0]        = top-left corner of the image
  [1000, 1000]  = bottom-right corner of the image
Do NOT return pixel coordinates. Do NOT return 0..1.
Only 0..1000.

Rules:
  - Ignore anything under a table's gridlines.
  - If in doubt, leave it out.
  - Output ONLY valid JSON. No prose. No markdown fences.
"""


class GroqVisionClient(VisionModelClient):
    provider_name = "groq"

    def __init__(self, api_key: str, model_name: str,
                 timeout_s: int = 120, max_retries: int = 3):
        if not api_key:
            raise ValueError("Groq API key missing (GROQ_API_KEY)")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI SDK missing. Install with: pip install openai"
            ) from exc
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=timeout_s,
        )
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def extract_candidates(self, image, page_width_pt, page_height_pt,
                           page_num=0, context=None):
        global _consecutive_failures
        if _consecutive_failures >= _CONSECUTIVE_FAILURE_LIMIT:
            log.warning("Groq circuit breaker open (%d failures) — skipping.",
                        _consecutive_failures)
            return []

        img_w, img_h = (image.size if hasattr(image, "size")
                        else (int(page_width_pt), int(page_height_pt)))

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call_model(image)
                _consecutive_failures = 0
                return self._parse(raw, page_width_pt, page_height_pt,
                                   img_w, img_h)
            except Exception as exc:
                last_err = exc
                log.warning("Groq call failed (attempt %d/%d): %s",
                            attempt + 1, self.max_retries + 1, exc)
                if attempt < self.max_retries:
                    time.sleep((2 ** attempt) * (0.7 + 0.6 * random.random()))

        _consecutive_failures += 1
        log.error("Groq extraction failed after %d attempts: %s",
                  self.max_retries + 1, last_err)
        return []

    @staticmethod
    def _pil_to_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _call_model(self, image: Image.Image) -> str:
        data_url = self._pil_to_data_url(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": data_url}},
                ],
            }
        ]

        # Disable Qwen thinking mode so we don't burn tokens on a
        # reasoning loop. Groq / OpenAI-compatible param.
        # If the model does not accept it, fall back to a plain call.
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
                extra_body={"reasoning_effort": "none"},
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "reasoning_effort" in msg or "extra" in msg or "unknown" in msg:
                log.warning("reasoning_effort not accepted by model — "
                            "retrying without it")
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
            else:
                raise

        content = ""
        try:
            content = resp.choices[0].message.content or ""
        except Exception:
            pass
        if not content:
            raise RuntimeError("Groq returned empty response")
        return content

    @staticmethod
    def _parse(raw: str, page_w: float, page_h: float,
               img_w: int, img_h: int) -> list[dict]:
        raw = raw.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            data = json.loads(raw)
        except Exception as exc:
            log.warning("Groq returned malformed JSON: %s", exc)
            return []

        candidates = data.get("candidates") or []
        out = []
        for it in candidates:
            b = it.get("bbox_1000") or it.get("bbox") or []
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            try:
                xmin = float(b[0])
                ymin = float(b[1])
                xmax = float(b[2])
                ymax = float(b[3])
            except (TypeError, ValueError):
                continue

            # Clamp to [0, 1000]
            xmin = max(0.0, min(1000.0, xmin))
            ymin = max(0.0, min(1000.0, ymin))
            xmax = max(0.0, min(1000.0, xmax))
            ymax = max(0.0, min(1000.0, ymax))

            # Normalize order
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            if ymax < ymin:
                ymin, ymax = ymax, ymin

            # Map 0-1000 → page points (STRICT linear mapping)
            x1 = (xmin / 1000.0) * page_w
            y1 = (ymin / 1000.0) * page_h
            x2 = (xmax / 1000.0) * page_w
            y2 = (ymax / 1000.0) * page_h

            # Reject degenerate boxes
            if (x2 - x1) < 1 and (y2 - y1) < 1:
                continue

            text = (it.get("text") or "").strip()
            if not text:
                continue

            out.append({
                "text": text,
                "type": (it.get("type") or "feature").strip().lower(),
                "parameter": (it.get("parameter") or "Dimn").strip(),
                "specification": (it.get("specification") or text).strip(),
                "tolerance": (it.get("tolerance") or "").strip(),
                "bbox": (x1, y1, x2, y2),
                "anchor_x": (x1 + x2) / 2.0,
                "anchor_y": (y1 + y2) / 2.0,
                "source": "groq",
                "confidence": float(it.get("confidence") or 0.5),
            })
        return out