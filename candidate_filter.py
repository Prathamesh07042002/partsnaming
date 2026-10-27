"""candidate_filter.py - unified post-source filter with density rejection."""
from __future__ import annotations

import math
import re
from collections import Counter


REJECT_KEYWORDS = (
    "DO NOT SCALE", "IF IN DOUBT", "FIRST ANGLE", "THIRD ANGLE",
    "ALL DIMS IN MM", "ALL DIMENSIONS IN MM", "UNLESS OTHERWISE",
    "DRG DRAWN TO", "DRAWING DRAWN TO", "ANGLE OF PROJECTION",
    "SUPPLIER DRG RELEASE", "NOMINAL DIMENSION",
    "TOLERANCE IN UM", "TOLERANCE IN MM",
    "TATA MOTORS", "TATA GROUP", "ERC - PUNE", "ERC- PUNE", "ERC-PUNE",
    "NEW RELEASE", "CLASSIFICATION OF CHARACTERISTICS",
    "SHEET NO", "ORIGINAL DRAWING",
    "REVISED AND REDRAWN", "DRAWING UPDATED", "FEEDBACK",
    "TRIVALENT", "FASTENERS",
    "PER LOT", "YEARLY", "SAMPLE SIZE", "CHECKING FREQUENCY",
    "DEVELOPMENT SAMPLE", "PILOT LOT", "PROTO TOOLING",
    "LAYOUT INSPECTION", "REGULAR PRODUCTION",
    "CUSTOMER NAME", "SUPPLIER NAME", "PART DESCRIPTION",
    "DRAWING REV", "CHECKING METHOD",
    "OPEN TOLERENCE", "OPEN TOLERANCE",
    "THIS DRAWING IS THE SOLE", "IT SHOULD NOT BE COPIED",
    "WITHOUT THE WRITTEN",
    "FOR BAR CODES", "FOR NUMBERING", "FOR LIST OF PARTS", "FOR ALL VEHICLE",
    "TO BE MARKED AS PER", "SHALL BE MARKED",
    "MATERIAL DESCRIPTION.SIZE",
    "SURFACE PROTECTION AS PER STANDARD", "TOLERANCE AS PER STANDARD",
    "REPLACES DRG", "REFERENCE DRG", "OPPOSITE HAND", "DRG/PART",
    "TACK WELD", "DEBURR",
    "PRODUCT QUALITY", "TRACEABILITY", "CLEANLINESS", "PACKAGING",
    "SURFACE FINISH", "EDGE CONDITIONS", "LEGENDS",
    "MICROMETRE", "VERNIER", "RADIUS GAUGE", "BEVEL",
    "WEIGHT MACHINE", "NOT MEASURABLE", "UNSPECIFIED",
    "LAST SECTION NAMED", "LAST VIEW NAMED", "LAST LINE NAMED",
    "UNDIMENSIONED", "DIMENSION IN THE DRAWING",
    "VIBRATION TEST", "SURFACE PROTECTION TO BE DONE", "SHARP EDGES",
    "FOR OTHER ROUGHNESS", "SURFACE ROUGHNESS", "MACHINING DEVIATION",
    "GENERAL TOLERANCE", "TOLERANCES OF STAMPINGS", "TOLERANCES ON",
    "OPEN DIMENSIONS", "PANEL FIXTURE",
    "SCALE:", "SCALE :",
    "DIMENSION NOMINAL", "DIMENSION IN M(", "DIMENSION IN M (",
    "NOMINAL ACCURACY", "SURFACE ROUGHNESS SYMBOLS",
)

REJECT_EXACT = {
    "MIC", "VC", "HG", "RG", "BP", "MT", "IF", "TC", "WT M", "NM",
    "BSH", "HMP", "RMP", "SKK", "NHH", "RSD", "GPP", "PNM", "NPY",
    "MVT", "MCT", "SRD", "RDC", "CHKD", "APPD", "SIGN", "DATE",
    "DRN", "CHK", "MOD", "ZONE", "QTY", "TYP", "REF", "MAX", "MIN",
    "NOM", "CRITICAL", "MAJOR", "MINOR", "NONE", "KEY INSTRUCTION",
    "A", "B", "C", "D", "E", "F", "G", "H",
    "A-A", "B-B", "C-C", "D-D",
}

RE_DATE     = re.compile(r"^\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?$"
                         r"|^\d{1,2}\.\d{1,2}\.\d{2,4}$"
                         r"|^\d{1,2}\.\d{4}$")
RE_REV_DATE = re.compile(r"^[A-Z]/?\d{1,2}\.\d{1,2}\.\d{2,4}$")
RE_VIEW_LABEL = re.compile(r"^(?:SECTION|VIEW|DETAIL)\s+[A-Z]"
                           r"(?:\s*-\s*[A-Z])?$", re.I)
RE_SCALE    = re.compile(r"^(?:SCALE|SCA)\s*[:=]?\s*[\d:./\s]+$", re.I)
RE_SINGLE_LD = re.compile(r"^[A-Z]\d?$")
RE_PCF      = re.compile(r"PCF[- ]?\d+", re.I)

RE_PURE_TOL  = re.compile(r"^[±]\s*\d{1,2}(?:\.\d+)?$")
RE_PURE_DEC  = re.compile(r"^\d{1,2}\.\d{1,2}$")
RE_PURE_INT  = re.compile(r"^\d{1,3}$")
RE_MULTI_NUM = re.compile(r"^(?:\d+(?:\.\d+)?\s+){2,}\d+(?:\.\d+)?$")
RE_TABLE_HDR = re.compile(
    r"^(?:"
    r"(?:up\s*to|upto(?:\s*&\s*incl\.?)?|over|from|to)\s+\d"
    r"|nominal(?:\s+accuracy)?"
    r"|dimension\s+(?:in|nominal)"
    r"|n\d+(?:\s+(?:hrd|grd|&|[a-z]+))*\s*$"
    r")",
    re.I,
)
RE_REAL_DIM = re.compile(
    r"[Ø⌀ϕφΦ∅]|^R\s*\d|°|º|[xX×]|HOLES?|TYP\b|REF\b"
)


def _looks_like_table_row(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if RE_TABLE_HDR.match(t):
        return True
    if RE_MULTI_NUM.match(t):
        tokens = t.split()
        if len(tokens) >= 3 and all(len(tok) <= 6 for tok in tokens):
            return True
    return False


def _is_boilerplate(text: str) -> bool:
    up = text.strip().upper()
    if not up:
        return True
    if up in REJECT_EXACT:
        return True
    if RE_DATE.match(up) or RE_REV_DATE.match(up):
        return True
    if RE_VIEW_LABEL.match(up) or RE_SCALE.match(up):
        return True
    if RE_SINGLE_LD.match(up):
        return True
    if RE_PCF.search(up):
        return True
    for kw in REJECT_KEYWORDS:
        if kw in up:
            return True
    if _looks_like_table_row(up):
        return True
    if not re.search(r"[A-Za-z0-9]", text):
        return True
    if len(text) > 120:
        return True
    return False


def detect_table_regions(spans, grid_pt: float = 40.0,
                         min_per_cell: int = 3):
    numeric_pts = []
    for s in spans:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        if (RE_PURE_TOL.match(t) or RE_PURE_DEC.match(t)
                or RE_PURE_INT.match(t) or RE_TABLE_HDR.match(t)
                or (RE_MULTI_NUM.match(t) and len(t) <= 30)):
            x1, y1, x2, y2 = s["bbox"]
            numeric_pts.append(((x1 + x2) / 2, (y1 + y2) / 2))

    if len(numeric_pts) < min_per_cell * 2:
        return []

    cells = Counter((int(cx // grid_pt), int(cy // grid_pt))
                    for cx, cy in numeric_pts)
    dense = {c for c, n in cells.items() if n >= min_per_cell}
    if not dense:
        return []

    visited: set[tuple] = set()
    regions = []
    for seed in dense:
        if seed in visited:
            continue
        queue = [seed]
        comp = set()
        while queue:
            c = queue.pop()
            if c in visited:
                continue
            visited.add(c)
            comp.add(c)
            gx, gy = c
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (gx + dx, gy + dy)
                    if nb in dense and nb not in visited:
                        queue.append(nb)
        xs = [c[0] for c in comp]
        ys = [c[1] for c in comp]
        regions.append((
            min(xs) * grid_pt, min(ys) * grid_pt,
            (max(xs) + 1) * grid_pt, (max(ys) + 1) * grid_pt,
        ))
    return regions


def _point_in_any(cx, cy, regions):
    for x1, y1, x2, y2 in regions:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


PROTECTED_TYPES = {
    "material_description", "material_specification",
    "part_number", "envelope", "mass", "surface_protection",
}


def filter_candidates(candidates, page_w: float, page_h: float,
                      table_regions=None):
    table_regions = table_regions or []
    kept = []
    stats = {"boilerplate": 0, "table_cell": 0,
             "out_of_page": 0, "kept": 0}

    for c in candidates:
        if c.get("type") in PROTECTED_TYPES:
            kept.append(c); stats["kept"] += 1
            continue
        text = (c.get("text") or "").strip()
        if not text or _is_boilerplate(text):
            stats["boilerplate"] += 1
            continue
        bx1, by1, bx2, by2 = c["bbox"]
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if not (0 <= cx <= page_w and 0 <= cy <= page_h):
            stats["out_of_page"] += 1
            continue
        if _point_in_any(cx, cy, table_regions):
            if not RE_REAL_DIM.search(text):
                stats["table_cell"] += 1
                continue
        kept.append(c); stats["kept"] += 1
    return kept, stats


def reject_dense_clusters(candidates, radius_pt: float = 30.0,
                          min_neighbors: int = 4,
                          log=None):
    feats = [c for c in candidates if c.get("type") == "feature"]
    out = []
    dropped = 0
    for c in candidates:
        if c.get("type") in PROTECTED_TYPES:
            out.append(c)
            continue
        if c.get("type") != "feature":
            out.append(c)
            continue
        bx1, by1, bx2, by2 = c["bbox"]
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        n = 0
        for o in feats:
            if o is c:
                continue
            ox1, oy1, ox2, oy2 = o["bbox"]
            ox, oy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            if abs(cx - ox) < radius_pt and abs(cy - oy) < radius_pt:
                n += 1
                if n >= min_neighbors:
                    break
        if n >= min_neighbors:
            dropped += 1
            continue
        out.append(c)
    if log and dropped:
        log.info("Cluster rejection: %d feature candidates dropped", dropped)
    return out


# ---------------------------------------------------------------------
# NEW: reject frame border letters/digits near page edges
# ---------------------------------------------------------------------

_FRAME_LETTERS = {"A", "B", "C", "D", "E", "F", "G", "H"}
_FRAME_DIGITS = {"1", "2", "3", "4", "5", "6", "7", "8"}


def reject_frame_borders(candidates, page_w, page_h, edge_pct=0.06):
    """Drop single-letter/digit candidates that sit at page edges.

    Tata drawings have A-H (left/right) and 1-8 (top/bottom) frame marks
    that models often extract as if they were dimensions. Real dimensions
    are never single letters or single 1-8 digits right at the page edge.
    """
    out = []
    dropped = 0
    for c in candidates:
        text = (c.get("text") or "").strip().upper()
        if len(text) == 1 and text:
            bx1, by1, bx2, by2 = c["bbox"]
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            nx = cx / page_w if page_w else 0.5
            ny = cy / page_h if page_h else 0.5
            near_x = nx < edge_pct or nx > (1 - edge_pct)
            near_y = ny < edge_pct or ny > (1 - edge_pct)
            if (text in _FRAME_LETTERS and near_x) or \
               (text in _FRAME_DIGITS and near_y):
                dropped += 1
                continue
        out.append(c)
    return out, dropped