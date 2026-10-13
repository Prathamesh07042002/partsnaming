"""candidate_merger.py – deduplicate + merge candidates (TYP/REF aware)."""
from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

_SYMBOL_MAP = str.maketrans({
    "ϕ": "Ø", "φ": "Ø", "⌀": "Ø", "Φ": "Ø", "∅": "Ø",
    "º": "°", "×": "x",
})

_TB_MERGE_TYPES = {
    "material_description", "material_specification", "part_number",
    "envelope", "mass", "surface_protection",
}


def _normalize_text(t: str) -> str:
    t = (t or "").strip().upper().translate(_SYMBOL_MAP)
    # strip TYP/REF/typical markers
    t = re.sub(r"\(TYP\)|\(REF\)|\(TYPICAL\)", "", t)
    t = re.sub(r"\bTYP\b|\bREF\b|\bTYPICAL\b", "", t)
    t = re.sub(r"\s+", "", t)
    return t.replace(",", ".")


def _numeric_core(t: str) -> str:
    """Extract the leading numeric/symbol token, stripping TYP/REF/tolerance.

    "34(TYP)"     → "34"
    "R20(TYP)"    → "R20"
    "379±0.5"     → "379"
    "Ø10.5H11"    → "Ø10.5H11"
    """
    t = (t or "").strip().upper().translate(_SYMBOL_MAP)
    t = re.sub(r"\(TYP\)|\(REF\)|\(TYPICAL\)", "", t)
    t = re.sub(r"\bTYP\b|\bREF\b|\bTYPICAL\b", "", t)
    t = t.strip()
    m = re.match(r"^(.+?)\s*[±]\s*\d", t)
    if m:
        t = m.group(1)
    return re.sub(r"\s+", "", t).strip()


def _similar(a: str, b: str) -> float:
    a_n, b_n = _normalize_text(a), _normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 1.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def _center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _spatial_close(a, b, radius):
    ax, ay = _center(a["bbox"])
    bx, by = _center(b["bbox"])
    return math.hypot(ax - bx, ay - by) <= radius


def _bbox_overlap_ratio(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


def _source_rank(c):
    s = (c.get("source") or "vector").lower()
    return {"vector": 0, "pymupdf": 0, "gemini": 1, "ocr": 2}.get(s, 3)


def _combine(weaker: dict, stronger: dict) -> dict:
    out = dict(stronger)
    for k in ("tolerance", "parameter", "specification", "type"):
        if not out.get(k) and weaker.get(k):
            out[k] = weaker[k]
    out["merged_sources"] = sorted({
        *(stronger.get("merged_sources")
          or [stronger.get("source", "vector")]),
        *(weaker.get("merged_sources")
          or [weaker.get("source", "vision")]),
    })
    return out


def merge_candidates(vector_cands, vision_cands,
                     spatial_radius_pt: float = 60.0,
                     text_similarity: float = 0.82):
    """Four-tier merge.

    Tier A : spatial + fuzzy text
    Tier B : same TB type + exact text
    Tier C : bbox overlap + fuzzy text
    Tier D : bbox overlap + same numeric core  (catches "34" vs "34(TYP)")
    """
    vlist = list(vector_cands or [])
    glist = list(vision_cands or [])
    merged = list(vlist)
    added = duplicates = replaced = 0

    for g in glist:
        match_idx = -1

        # Tier A
        for i, v in enumerate(merged):
            if not _spatial_close(v, g, spatial_radius_pt):
                continue
            if _similar(v.get("text", ""), g.get("text", "")) >= text_similarity:
                match_idx = i
                break

        # Tier B — TB fields
        if match_idx < 0:
            g_type = (g.get("type") or "").lower()
            g_norm = _normalize_text(g.get("text", ""))
            if g_type in _TB_MERGE_TYPES and len(g_norm) >= 3:
                for i, v in enumerate(merged):
                    if (v.get("type") or "").lower() != g_type:
                        continue
                    if _normalize_text(v.get("text", "")) == g_norm:
                        match_idx = i
                        break

        # Tier C — bbox overlap + fuzzy text
        if match_idx < 0:
            for i, v in enumerate(merged):
                if _bbox_overlap_ratio(v["bbox"], g["bbox"]) < 0.40:
                    continue
                if _similar(v.get("text", ""), g.get("text", "")) >= text_similarity:
                    match_idx = i
                    break

        # Tier D — bbox overlap + identical numeric core
        if match_idx < 0:
            g_core = _numeric_core(g.get("text", ""))
            if g_core and len(g_core) >= 2:
                for i, v in enumerate(merged):
                    if _bbox_overlap_ratio(v["bbox"], g["bbox"]) < 0.30:
                        continue
                    v_core = _numeric_core(v.get("text", ""))
                    if v_core and v_core == g_core:
                        match_idx = i
                        break

        if match_idx < 0:
            merged.append(g)
            added += 1
            continue

        duplicates += 1
        existing = merged[match_idx]
        if _source_rank(g) < _source_rank(existing):
            merged[match_idx] = _combine(existing, g)
            replaced += 1
        else:
            merged[match_idx] = _combine(g, existing)

    stats = {
        "vector_in": len(vlist),
        "vision_in": len(glist),
        "merged_total": len(merged),
        "vision_added": added,
        "duplicates_removed": duplicates,
        "replaced_by_vision": replaced,
    }
    return merged, stats