"""GeminiVisionClient – [ymin, xmin, ymax, xmax] with exact unpad.

The image supplied by the pipeline is already a padded square.
Padding metadata arrives via `context`:
    pad_left, pad_top, content_w, content_h, square_side
so we can reverse the operation deterministically and map
normalized 0-1000 bboxes back to page points.
"""
from __future__ import annotations
import json
import logging
import random
import re
import time
from PIL import Image
from vision.base import VisionModelClient

log = logging.getLogger(__name__)


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string"},
                    "parameter": {"type": "string"},
                    "specification": {"type": "string"},
                    "tolerance": {"type": "string"},
                    "bbox_1000": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Exactly [ymin, xmin, ymax, xmax] mapped to 0-1000 scale"
                    }
                },
                "required": ["text", "bbox_1000"],
            },
        }
    },
    "required": ["candidates"],
}


def _build_prompt() -> str:
    return """\
TASK: You are a strict mechanical drawing inspector. Extract dimensions and the bottom-right Title Block data from the image. Output ONLY valid JSON.

1. DIMENSIONS TO EXTRACT (type: "feature")
You MUST find and extract EVERY dimension on the sheet, including:
- Normal numbers: "407", "379±0.5", "R15", "Ø13".
- Boxed Dimensions: You MUST extract numbers enclosed inside small rectangular boxes. Do NOT skip them.
- Rotated Text: You MUST extract numbers written vertically (reading bottom-to-top or top-to-bottom) along the side edges.
- Angles and Callouts: "103°", "2 HOLES Ø8".

2. TITLE BLOCK TO EXTRACT (Bottom-Right Table)
You MUST look at the bottom-right corner of the drawing and extract these exact fields:
- "part_number": Find the text near "Drg/Part No." or "Drawing No." (e.g., "2882 3154 01 10").
- "material_description": Find the value under "Material description, size, spec" (e.g., "3mm Thk - T SH HR-Fe410").
- "material_specification": Find the standard or grade near the material description (e.g., "TS23521").
- "envelope": Find the 3D dimensions near "Envelope Dimensions" (e.g., "409.00 x 65.00 x 27.15").
- "surface_protection": Find the value near "Surface protection as per standard" (e.g., "FeZn 8 Cr III Y").
- "mass": Find the decimal number near "Final mass in kg" (e.g., "0.553").
- "note": Find the vehicle model near "Product/Group No." (e.g., "LPT 1613").

3. WHAT TO IGNORE
- Do NOT extract grids of small numbers (Tolerance/Roughness tables).
- Do NOT extract signature rows, dates, or revision history.
- Do NOT extract frame border letters (A, B, C) and numbers (1, 2, 3).

4. BOUNDING BOX RULES (bbox_1000)
- You MUST return a tightly wrapped bounding box for each item: [ymin, xmin, ymax, xmax].
- The values MUST be mapped to a 0 to 1000 scale, where [0,0] is top-left and [1000,1000] is bottom-right.
- For rotated text, ensure the bounding box covers the entire vertical length of the text.

5. JSON OUTPUT FORMAT
You MUST output ONLY JSON, without any markdown formatting or prose. 
Separate the base dimension into "specification" and any tolerance into "tolerance". If no tolerance exists, use "".

{
  "candidates": [
    {
      "text": "379±0.5",
      "type": "feature",
      "parameter": "Dimn",
      "specification": "379",
      "tolerance": "±0.5",
      "bbox_1000": [250, 800, 270, 820],
      "confidence": 0.95
    },
    {
      "text": "2882 3154 01 10",
      "type": "part_number",
      "parameter": "Part No.",
      "specification": "2882 3154 01 10",
      "tolerance": "",
      "bbox_1000": [900, 700, 950, 850],
      "confidence": 0.98
    }
  ]
}

If absolutely nothing is found, output: {"candidates": []}
"""


class GeminiVisionClient(VisionModelClient):
    provider_name = "gemini"

    def __init__(self, api_key: str, model_name: str, timeout_s: int = 120, max_retries: int = 3):
        if not api_key:
            raise ValueError("Gemini API key missing")
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    def extract_candidates(self, image, page_width_pt, page_height_pt,
                           page_num=0, context=None):
        ctx = context or {}
        pad_left   = int(ctx.get("pad_left", 0))
        pad_top    = int(ctx.get("pad_top", 0))
        square_side = int(ctx.get("square_side", image.size[0]))
        content_w  = int(ctx.get("content_w", image.size[0]))
        content_h  = int(ctx.get("content_h", image.size[1]))

        log.info("Gemini pad-in: img=%dx%d content=%dx%d pad=(%d,%d) side=%d",
                 image.size[0], image.size[1],
                 content_w, content_h, pad_left, pad_top, square_side)

        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call_model(image)
                return self._parse(
                    raw, page_width_pt, page_height_pt,
                    content_w, content_h, square_side, pad_left, pad_top,
                )
            except Exception as exc:
                log.warning("Gemini call failed (attempt %d/%d): %s",
                            attempt + 1, self.max_retries + 1, exc)
                time.sleep(2)
        return []

    def _call_model(self, image: Image.Image) -> str:
        try:
            afc = self._types.AutomaticFunctionCallingConfig(disable=True)
            cfg = self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.1, automatic_function_calling=afc,
            )
        except Exception:
            cfg = self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA, temperature=0.1,
            )
        resp = self.client.models.generate_content(
            model=self.model_name, contents=[_build_prompt(), image], config=cfg
        )
        return getattr(resp, "text", "")

    @staticmethod
    def _parse(raw, page_w, page_h,
               content_w, content_h, square_side, pad_left, pad_top):
        raw = raw.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            data = json.loads(raw)
        except Exception:
            return []

        # Scale factors: content pixels → page points
        sx = page_w / float(content_w) if content_w else 1.0
        sy = page_h / float(content_h) if content_h else 1.0

        out = []
        for it in data.get("candidates", []):
            b = it.get("bbox_1000")
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue

            # Gemini native order: [ymin, xmin, ymax, xmax]
            try:
                ymin = max(0.0, min(1000.0, float(b[0])))
                xmin = max(0.0, min(1000.0, float(b[1])))
                ymax = max(0.0, min(1000.0, float(b[2])))
                xmax = max(0.0, min(1000.0, float(b[3])))
            except (TypeError, ValueError):
                continue

            # 0-1000 → padded square pixels
            sq_x1 = (xmin / 1000.0) * square_side
            sq_y1 = (ymin / 1000.0) * square_side
            sq_x2 = (xmax / 1000.0) * square_side
            sq_y2 = (ymax / 1000.0) * square_side

            # remove padding we added → content pixels
            img_x1 = sq_x1 - pad_left
            img_y1 = sq_y1 - pad_top
            img_x2 = sq_x2 - pad_left
            img_y2 = sq_y2 - pad_top

            # clamp to content rectangle
            img_x1 = max(0.0, min(float(content_w), img_x1))
            img_y1 = max(0.0, min(float(content_h), img_y1))
            img_x2 = max(0.0, min(float(content_w), img_x2))
            img_y2 = max(0.0, min(float(content_h), img_y2))

            if img_x2 < img_x1: img_x1, img_x2 = img_x2, img_x1
            if img_y2 < img_y1: img_y1, img_y2 = img_y2, img_y1
            if (img_x2 - img_x1) < 1 and (img_y2 - img_y1) < 1:
                continue

            # content pixels → page points
            x1 = img_x1 * sx
            y1 = img_y1 * sy
            x2 = img_x2 * sx
            y2 = img_y2 * sy

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
                "anchor_x": (x1 + x2) / 2,
                "anchor_y": (y1 + y2) / 2,
                "source": "gemini",
            })
        return out