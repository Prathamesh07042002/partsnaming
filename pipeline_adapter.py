"""pipeline_adapter.py - glue between the new extraction pipeline and the
existing bubbling engine in clockwise_numbering.py.

The bubble engine consumes `items` with pixel coordinates.  This module
converts unified candidates (page points) into that schema, reusing the
existing place_bubble / _dedupe / _spread helpers unchanged.

All pixel coordinates produced here are clamped into [0, w-1] / [0, h-1]
so the (read-only) validate_items boundary check cannot fail on anchors
that land exactly on the far edge of the page (e.g. vision anchor_norm=1.0).
"""
from __future__ import annotations

import logging
from pathlib import Path

import clockwise_numbering as cn
from config import Settings, load_settings
from extraction_pipeline import extract_for_page

log = logging.getLogger(__name__)


_ALLOWED_TYPES = {
    "material_description", "material_specification",
    "part_number", "envelope", "surface_protection",
    "feature", "mass", "note",
}


def _safe_type(t: str) -> str:
    t = (t or "").strip().lower()
    return t if t in _ALLOWED_TYPES else "feature"


def _clamp(v: float, lo: int, hi_inclusive: int) -> int:
    """Clamp to [lo, hi_inclusive]; hi_inclusive is a valid pixel index."""
    iv = int(round(v))
    if iv < lo:
        return lo
    if iv > hi_inclusive:
        return hi_inclusive
    return iv


def _sanitize_bbox(bbox, w: int, h: int):
    """Return (x1, y1, x2, y2) clamped inside the image, x1<=x2, y1<=y2."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w - 1)
    y2 = _clamp(y2, 0, h - 1)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # guarantee a non-zero box (validate_items wants >=0 anyway; this helps
    # place_bubble pick a sane slot)
    if x2 == x1:
        x2 = min(w - 1, x1 + 1)
    if y2 == y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def pipeline_draft(input_path: Path, page_num: int | None = None,
                   target_size: tuple[int, int] | None = None,
                   settings: Settings | None = None):
    """Extraction entrypoint that mirrors cn.pdf_draft's signature.

    Returns (items, (img_w, img_h)).
    """
    settings = settings or load_settings()
    result = extract_for_page(input_path, page_num, settings)
    page = result.page

    try:
        if target_size:
            img_w, img_h = int(target_size[0]), int(target_size[1])
        else:
            img_w, img_h = page.content_size

        if img_w < 2 or img_h < 2:
            raise ValueError(f"Image too small: {img_w}x{img_h}")

        sx = img_w / page.width_pt
        sy = img_h / page.height_pt

        items = []
        skipped = 0
        for c in result.candidates:
            # --- bbox: clamp into image rectangle ------------------------
            px1, py1, px2, py2 = _sanitize_bbox(
                (c["bbox"][0] * sx, c["bbox"][1] * sy,
                 c["bbox"][2] * sx, c["bbox"][3] * sy),
                img_w, img_h,
            )

            # --- anchor → target px, clamped so anchor never == w or h ---
            ax_pt = c.get("anchor_x")
            ay_pt = c.get("anchor_y")
            if ax_pt is None or ay_pt is None:
                target_px = None
            else:
                tx = _clamp(ax_pt * sx, 0, img_w - 1)
                ty = _clamp(ay_pt * sy, 0, img_h - 1)
                target_px = (tx, ty)

            bx, by, tx, ty = cn.place_bubble(
                (px1, py1, px2, py2), img_w, img_h,
                target_override=target_px,
            )

            # place_bubble already clamps the bubble; still guard the target
            tx = _clamp(tx, 0, img_w - 1)
            ty = _clamp(ty, 0, img_h - 1)
            bx = _clamp(bx, 0, img_w - 1)
            by = _clamp(by, 0, img_h - 1)

            items.append({
                "type":          _safe_type(c.get("type")),
                "parameter":     c.get("parameter", "Dimn"),
                "specification": c.get("text") or c.get("specification", ""),
                "tolerance":     c.get("tolerance", ""),
                "x": bx, "y": by,
                "target_x": tx, "target_y": ty,
                "source_x": px1, "source_y": py1,
                "source_x2": px2, "source_y2": py2,
                "notes": f"source={c.get('source', 'unknown')}",
            })

        log.info("Adapter: %d items (%d skipped during sanitise)",
                 len(items), skipped)

        items = cn._dedupe(items)
        cn._spread(items, min_distance=115.0, width=img_w, height=img_h)
        return items, (img_w, img_h)
    finally:
        page.close()