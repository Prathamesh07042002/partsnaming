"""
pdf_extractor.py – native-PDF inspection characteristic extractor.

Strategy (no OCR as primary path):
  1. Extract text spans from PyMuPDF with full geometry (bbox, direction, font).
  2. Identify the title-block region from TB anchor labels and structural lines.
  3. Filter out boilerplate / layout elements using geometry + content rules.
  4. Classify surviving spans into typed characteristics.
  5. Merge fragmented same-row spans into logical annotations.
  6. Attach each annotation to its nearest relevant leader-line endpoint
     (arrowhead target) so bubbles land on the dimension, not a random centroid.
  7. Handle rotated/vertical text by normalising bboxes to axis-aligned rects.
  8. OCR FALLBACK: after native extraction, render embedded image blocks and
     drawing-region gaps to recover raster-only dimensions; merge results
     without duplicating native finds; apply strict 5-field TB filter for
     any OCR finds in the title-block region.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("pip install PyMuPDF") from exc


# =========================================================================
# Boilerplate reject lists  (unchanged from original)
# =========================================================================

REJECT_KEYWORDS = [
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
    "SHARADA", "DIVESH", "PRASHANT", "BHALERAO", "WALUNJ", "KALE", "ARVIND",
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
]

REJECT_EXACT = {
    "MIC", "VC", "HG", "RG", "BP", "MT", "IF", "TC", "WT M", "NM",
    "BSH", "HMP", "RMP", "SKK", "NHH", "RSD", "GPP", "PNM", "NPY",
    "MVT", "MCT", "SRD", "RDC", "CHKD", "APPD",
    "SIGN", "DATE", "DRN", "CHK", "MOD", "ZONE", "QTY",
    "TYP", "REF", "MAX", "MIN", "NOM",
    "CRITICAL", "MAJOR", "MINOR", "NONE", "KEY INSTRUCTION",
    "A", "B", "C", "D", "E", "F", "G", "H",
    "A-A", "B-B", "C-C", "D-D",
}

RE_DATE = re.compile(
    r"^\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?$"
    r"|^\d{1,2}\.\d{1,2}\.\d{2,4}$"
    r"|^\d{1,2}\.\d{4}$"
)
RE_REVISION_DATE = re.compile(r"^[A-Z]/?\d{1,2}\.\d{1,2}\.\d{2,4}$")
RE_BARE_MASS = re.compile(r"^0\.\d{2,4}$")
RE_SHEET_FORMAT = re.compile(r"^\d{4}\s+\d{4}\s+\d{2}\s+\d{2}$")
RE_PCF_CODE = re.compile(r"PCF[- ]?\d+", re.I)

RE_SINGLE_LETTER_DIGIT = re.compile(r"^[A-Z]\d?$")
RE_VIEW_LABEL = re.compile(r"^(?:SECTION|VIEW|DETAIL)\s+[A-Z](?:\s*-\s*[A-Z])?$", re.I)
RE_SCALE_LABEL = re.compile(r"^(?:SCALE|SCA)\s*[:=]?\s*[\d:./\s]+$", re.I)


def _looks_like_boilerplate(text: str) -> bool:
    up = text.strip().upper()
    if not up:
        return True
    if up in REJECT_EXACT:
        return True
    if RE_DATE.match(up):
        return True
    if RE_REVISION_DATE.match(up):
        return True
    if RE_VIEW_LABEL.match(up):
        return True
    if RE_SCALE_LABEL.match(up):
        return True
    if RE_SINGLE_LETTER_DIGIT.match(up):
        return True
    for kw in REJECT_KEYWORDS:
        if kw in up:
            return True
    if not re.search(r"[A-Za-z0-9]", text):
        return True
    if len(text) > 80:
        return True
    return False


def _looks_like_dimension(text: str) -> bool:
    """Contains a digit or engineering symbol (Ø ⌀ ϕ φ Φ ∅ ± ° º R±)."""
    return bool(re.search(r"[0-9Ø⌀ϕφΦ∅±°º]", text))


# =========================================================================
# Rotated-span normalisation
# =========================================================================

def _normalise_bbox(bbox, direction):
    """Return an axis-aligned bounding box even for rotated spans."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w < 2 or h < 2:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        length = max(w, h, 8)
        thickness = 8
        if w < 2:  # vertical text
            return (cx - thickness / 2, cy - length / 2,
                    cx + thickness / 2, cy + length / 2)
        else:  # horizontal but very thin
            return (cx - length / 2, cy - thickness / 2,
                    cx + length / 2, cy + thickness / 2)
    return bbox


# =========================================================================
# Fragment grouping (same-row merge for split annotations)
# =========================================================================

def _smart_join(parts):
    """Join text fragments — no space around engineering connectors."""
    out = ""
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        if i == 0:
            out = p
            continue
        if p[0] in ")].,;:°/±([":
            out += p
        elif out and out[-1] in "([/±ØR":
            out += p
        elif re.match(r"^[A-Z]\d+$", p) and out and out[-1].isdigit():
            out += p
        else:
            out += " " + p
    return re.sub(r"\s+", " ", out).strip()


def _group_fragmented_spans(spans, row_tol=4.0, gap_tol=10.0):
    """Merge adjacent same-row spans into one logical annotation."""
    if not spans:
        return []

    def _dir_bucket(d):
        dx, dy = d
        angle = round(math.degrees(math.atan2(-dy, dx)) % 360 / 90) * 90
        return int(angle % 360)

    ordered = sorted(spans, key=lambda s: (round(s["bbox"][1] / 3), s["bbox"][0]))
    used = [False] * len(ordered)
    groups = []

    for i, s in enumerate(ordered):
        if used[i]:
            continue
        bucket = [s]
        used[i] = True
        dir_i = _dir_bucket(s.get("direction", (1.0, 0.0)))
        changed = True
        while changed:
            changed = False
            for j in range(len(ordered)):
                if used[j]:
                    continue
                o = ordered[j]
                if _dir_bucket(o.get("direction", (1.0, 0.0))) != dir_i:
                    continue
                for m in bucket:
                    cy1 = (m["bbox"][1] + m["bbox"][3]) / 2
                    cy2 = (o["bbox"][1] + o["bbox"][3]) / 2
                    if abs(cy1 - cy2) > row_tol:
                        continue
                    gap_f = o["bbox"][0] - m["bbox"][2]
                    gap_b = m["bbox"][0] - o["bbox"][2]
                    if -3.0 <= gap_f <= gap_tol or -3.0 <= gap_b <= gap_tol:
                        bucket.append(o)
                        used[j] = True
                        changed = True
                        break

        bucket.sort(key=lambda x: x["bbox"][0])
        text = _smart_join([b["text"] for b in bucket])
        xs, ys = [], []
        for b in bucket:
            xs.extend([b["bbox"][0], b["bbox"][2]])
            ys.extend([b["bbox"][1], b["bbox"][3]])
        groups.append({
            "text": text,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "font_size": max(b.get("font_size", 0) for b in bucket),
            "direction": bucket[0].get("direction", (1.0, 0.0)),
            "page": bucket[0].get("page", 0),
        })
    return groups


# =========================================================================
# Classification regexps
# =========================================================================

RE_DIAMETER = re.compile(r"^[Ø⌀ϕφΦ∅]\s*\d", re.I)
RE_RADIUS = re.compile(r"^R\s*\d", re.I)
RE_ANGLE = re.compile(r"\d+(?:\.\d+)?\s*[°º]")
RE_HOLES = re.compile(r"^\d+\s*HOLES?\b", re.I)
RE_DIM_TOL = re.compile(r"^\(?\d+(?:\.\d+)?\)?\s*[±]\s*\d")
RE_PARTNUM_SPACED = re.compile(r"^\d{4}\s+\d{4}\s+\d{2}\s+\d{2}$")
RE_PARTNUM_BARE = re.compile(r"^\d{10,14}$")
RE_ENVELOPE = re.compile(r"\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?\s*[xX×]\s*\d+")
RE_MASS = re.compile(r"\d+(?:\.\d+)?\s*kg\b", re.I)
RE_MAT_DESC = re.compile(
    r"^\d+(?:\.\d+)?\s*mm\s*thk|^(?:thick\s+)?(?:sheet|plate)\s+\d", re.I
)
RE_MAT_SPEC = re.compile(r"^(?:T\s*SH|T\s*HR|TS\s*\d|IS\s*\d|DD\s*:)", re.I)

RE_PLAIN_NUMERIC = re.compile(r"^\(?\d{2,}(?:\.\d+)?\)?$|^\(?\d+\.\d+\)?$")


def _split_tolerance(text: str) -> tuple[str, str]:
    m = re.match(r"^(.+?)\s*([±]\s*\d+(?:\.\d+)?)$", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return text, ""


def classify_span(text: str) -> tuple[str | None, str, str]:
    """Return (type, parameter, tolerance) for a text span, or (None,'','')."""
    t = text.strip()
    if not t:
        return None, "", ""

    if RE_MASS.search(t):
        return "mass", "Final mass", ""
    if RE_PARTNUM_SPACED.match(t) or RE_PARTNUM_BARE.match(t):
        return "part_number", "Part No.", ""
    if RE_ENVELOPE.search(t):
        return "envelope", "Envelope", ""
    if RE_MAT_DESC.match(t.lower()):
        return "material_description", "Thk", ""
    if RE_MAT_SPEC.match(t):
        return "material_specification", "Grade", ""
    if "surface protection" in t.lower():
        return "surface_protection", "Surface Protection", ""

    if RE_RADIUS.match(t):
        return "feature", "Radius", ""
    if RE_DIAMETER.match(t):
        return "feature", "Hole", ""
    if RE_HOLES.match(t):
        return "feature", "Hole", ""
    if RE_ANGLE.search(t):
        return "feature", "Angle", ""
    if RE_DIM_TOL.match(t):
        _, tol = _split_tolerance(t)
        return "feature", "Dimn", tol
    if re.search(r"\d", t) and RE_PLAIN_NUMERIC.match(t.strip("() ")):
        return "feature", "Dimn", ""
    if re.search(r"\d", t) and len(t) >= 3:
        return "feature", "Dimn", ""
    return None, "", ""


# =========================================================================
# Title-block detection using both text anchors AND drawing lines
# =========================================================================

TB_LABELS = {
    "envelope":             re.compile(r"envelope\s*dim", re.I),
    "mass":                 re.compile(r"final\s*mass|mass\s*in\s*kg", re.I),
    "surface_protection":   re.compile(r"surface\s*protection", re.I),
    "drg_part":             re.compile(r"drg\s*/\s*part\s*no", re.I),
    "material_description": re.compile(r"material\s*description", re.I),
}

TB_ALLOWED_KINDS = {
    "material_description",
    "material_specification",
    "part_number",
    "envelope",
    "mass",
    "surface_protection",
}


# def _detect_title_block_region(grouped, pw, ph, drawings):
#     """Return (x1, y1, x2, y2) of the title block in PDF points."""
#     # --- Strategy 1: text anchors ---
#     tb_anchors = []
#     for g in grouped:
#         for pat in TB_LABELS.values():
#             if pat.search(g["text"]):
#                 tb_anchors.append(g["bbox"])
#                 break

#     if len(tb_anchors) >= 2:
#         tb_x1 = min(a[0] for a in tb_anchors) - 40
#         tb_y1 = min(a[1] for a in tb_anchors) - 40
#         return (max(0, tb_x1), max(0, tb_y1), pw, ph)

#     # --- Strategy 2: structural lines ---
#     long_h_ys = []
#     long_v_xs = []
#     for d in drawings:
#         for item in d.get("items", []):
#             if item[0] != "l":
#                 continue
#             p1, p2 = item[1], item[2]
#             dx = abs(p2.x - p1.x)
#             dy = abs(p2.y - p1.y)
#             length = math.hypot(dx, dy)
#             if length < pw * 0.25:
#                 continue
#             if dy < length * 0.05 and p1.y > ph * 0.55:
#                 long_h_ys.append((p1.y + p2.y) / 2)
#             if dx < length * 0.05 and min(p1.x, p2.x) > pw * 0.55:
#                 long_v_xs.append((p1.x + p2.x) / 2)

#     if long_h_ys and long_v_xs:
#         tb_y1 = min(long_h_ys) - 5
#         tb_x1 = min(long_v_xs) - 5
#         return (max(0, tb_x1), max(0, tb_y1), pw, ph)

#     if long_h_ys:
#         tb_y1 = min(long_h_ys) - 5
#         return (0, max(0, tb_y1), pw, ph)

#     # --- Strategy 3: fixed bottom-right fraction ---
#     if len(tb_anchors) == 1:
#         a = tb_anchors[0]
#         return (max(0, a[0] - 40), max(0, a[1] - 40), pw, ph)

#     tb_x1 = pw * 0.72
#     tb_y1 = ph * 0.78

#     # Force the Title Block to stay ONLY in the bottom-right corner (max 30% width, 25% height)
#     # This prevents the Title block region from eating up actual drawing features!
#     tb_x1 = max(tb_x1, pw * 0.70)
#     tb_y1 = max(tb_y1, ph * 0.75)
#     return (max(0, tb_x1), max(0, tb_y1), pw, ph)
def _detect_title_block_region(grouped, pw, ph, drawings):
    """Safely clamp the Title Block to the bottom-right corner."""
    return (max(0, pw * 0.65), max(0, ph * 0.70), pw, ph)

# =========================================================================
# Title-block field extraction  (strict 5-field filter)
# =========================================================================

def extract_title_block(grouped, tb_region):
    """Extract the 5 mandatory title-block characteristics.

    ONLY the fields in TB_ALLOWED_KINDS are returned.  Everything else
    inside the title block stays unbubbled.
    """
    x1r, y1r, x2r, y2r = tb_region

    def _in_tb(b):
        x1, y1, x2, y2 = b
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return x1r <= cx <= x2r and y1r <= cy <= y2r

    inside = [g for g in grouped if _in_tb(g["bbox"])]
    fields = []
    seen = set()

    # 2. Part number  (Drg / Part No.)
    for g in inside:
        t = g["text"].strip()
        if (re.match(r"^\d{4}\s+\d{4}\s+\d{2}\s+\d{2}$", t)
                or re.match(r"^\d{10,14}$", t)):
            fields.append({"kind": "part_number", "parameter": "Part No.",
                           "specification": t, "bbox": g["bbox"], "text": t})
            seen.add("part_number")
            break

    # 3. Envelope Dimensions
    for g in inside:
        t = g["text"].strip()
        if re.match(r"^\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?$", t):
            fields.append({"kind": "envelope", "parameter": "Envelope",
                           "specification": t, "bbox": g["bbox"], "text": t})
            seen.add("envelope")
            break

    # 1. Material description — size + specification + standard number
    for g in inside:
        t = g["text"].strip()
        if re.match(r"^\d+(?:\.\d+)?\s*mm\s*thk", t, re.I):
            fields.append({"kind": "material_description", "parameter": "Thk",
                           "specification": t, "bbox": g["bbox"], "text": t})
            seen.add("material_description")
            break
    if "material_description" not in seen:
        for g in inside:
            t = g["text"].strip()
            if re.match(r"^(?:thick\s+)?(?:sheet|plate)\s+\d", t, re.I):
                fields.append({"kind": "material_description", "parameter": "Thk",
                               "specification": t, "bbox": g["bbox"], "text": t})
                seen.add("material_description")
                break

    # 1b. Material specification (grade / standard)
    for g in inside:
        t = g["text"].strip()
        if (re.match(r"^(?:T\s*SH|T\s*HR|TS\s*\d|IS\s*\d|DD\s*:)", t, re.I)
                or re.search(r"\bTS\s*\d{4,}\b", t, re.I)):
            fields.append({"kind": "material_specification", "parameter": "Grade",
                           "specification": t, "bbox": g["bbox"], "text": t})
            seen.add("material_specification")
            break

    # 4. Final mass in kg
    mass_anchor = None
    for g in inside:
        if TB_LABELS["mass"].search(g["text"]):
            mass_anchor = g
            break
    if mass_anchor:
        lx1, ly1, lx2, ly2 = mass_anchor["bbox"]
        lcy = (ly1 + ly2) / 2
        best, best_d = None, 1e9
        for g in inside:
            if g is mass_anchor:
                continue
            t = g["text"].strip()
            if not re.match(r"^\d+(?:\.\d+)?(\s*kg)?$", t, re.I):
                continue
            sx1, sy1, sx2, sy2 = g["bbox"]
            scy = (sy1 + sy2) / 2
            d = abs(scy - lcy) * 3 + max(0, sx1 - lx2)
            if d < best_d:
                best_d, best = d, g
        if best:
            spec = best["text"].strip()
            if not spec.lower().endswith("kg"):
                spec += " kg"
            fields.append({"kind": "mass", "parameter": "Final mass",
                           "specification": spec, "bbox": best["bbox"], "text": spec})
            seen.add("mass")

    # Mass fallback: bare decimal in bottom-right quadrant
    if "mass" not in seen:
        for g in inside:
            t = g["text"].strip()
            if re.match(r"^0\.\d{2,4}$", t):
                gcx = (g["bbox"][0] + g["bbox"][2]) / 2
                gcy = (g["bbox"][1] + g["bbox"][3]) / 2
                if gcx > (x1r + x2r) / 2 and gcy > (y1r + y2r) / 2:
                    spec = t + " kg"
                    fields.append({"kind": "mass", "parameter": "Final mass",
                                   "specification": spec, "bbox": g["bbox"], "text": spec})
                    seen.add("mass")
                    break

    # 5. Surface protection as per standard
    for g in inside:
        if TB_LABELS["surface_protection"].search(g["text"]):
            after = re.sub(r".*?surface\s*protection", "", g["text"], flags=re.I).strip()
            after = re.sub(r"^[:\-–\s]+", "", after)
            if not after or after.lower() in ("as per standard", "-", ""):
                after = "As per standard"
            fields.append({"kind": "surface_protection",
                           "parameter": "Surface Protection",
                           "specification": after, "bbox": g["bbox"], "text": after})
            break

    out, seen2 = [], set()
    for f in fields:
        if f["kind"] in seen2:
            continue
        if f["kind"] not in TB_ALLOWED_KINDS:
            continue
        seen2.add(f["kind"])
        out.append(f)
    return out


# =========================================================================
# Drawing structure analysis — isolate table/border lines from dim lines
# =========================================================================

def _classify_drawing_lines(drawings, pw, ph):
    table_rects = []
    leader_segs = []
    long_lines = []

    page_diagonal = math.hypot(pw, ph)

    for d in drawings:
        for item in d.get("items", []):
            op = item[0]

            if op == "re":
                r = item[1]
                rw = abs(r.x1 - r.x0)
                rh = abs(r.y1 - r.y0)
                if rw > pw * 0.05 or rh > ph * 0.05:
                    table_rects.append((r.x0, r.y0, r.x1, r.y1))

            elif op == "l":
                p1, p2 = item[1], item[2]
                length = math.hypot(p2.x - p1.x, p2.y - p1.y)
                if length > page_diagonal * 0.5:
                    long_lines.append((p1.x, p1.y, p2.x, p2.y))
                elif length < page_diagonal * 0.3:
                    leader_segs.append((p1.x, p1.y, p2.x, p2.y,
                                        length, d.get("width") or 0))

    return table_rects, leader_segs, long_lines


def _build_exclusion_zones(table_rects, long_lines, pw, ph):
    zones = []
    for r in table_rects:
        zones.append(r)
    for x1, y1, x2, y2 in long_lines:
        if abs(y2 - y1) < abs(x2 - x1) * 0.1:
            yc = (y1 + y2) / 2
            zones.append((0, yc - 6, pw, yc + 6))
    return zones


def _pt_in_zone(cx, cy, zones):
    for x1, y1, x2, y2 in zones:
        if min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2):
            return True
    return False


# =========================================================================
# Arrowhead / leader-endpoint detection
# =========================================================================

def _collect_leader_endpoints(drawings, drawing_region):
    dx1, dy1, dx2, dy2 = drawing_region
    quads = []
    short_lines = []

    for d in drawings:
        for item in d.get("items", []):
            op = item[0]
            if op == "qu":
                q = item[1]
                cx = (q.ul.x + q.ur.x + q.lr.x + q.ll.x) / 4
                cy = (q.ul.y + q.ur.y + q.lr.y + q.ll.y) / 4
                if dx1 <= cx <= dx2 and dy1 <= cy <= dy2:
                    quads.append((cx, cy))
            elif op == "l":
                p1, p2 = item[1], item[2]
                L = math.hypot(p2.x - p1.x, p2.y - p1.y)
                if 1.5 <= L <= 14:
                    mx, my = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
                    if dx1 <= mx <= dx2 and dy1 <= my <= dy2:
                        short_lines.append((mx, my))

    def cluster(pts, radius=6.0):
        if not pts:
            return []
        used = [False] * len(pts)
        out = []
        for i in range(len(pts)):
            if used[i]:
                continue
            grp = [pts[i]]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(pts)):
                    if used[j]:
                        continue
                    for gx, gy in grp:
                        if abs(pts[j][0] - gx) < radius and abs(pts[j][1] - gy) < radius:
                            grp.append(pts[j])
                            used[j] = True
                            changed = True
                            break
            gx = sum(p[0] for p in grp) / len(grp)
            gy = sum(p[1] for p in grp) / len(grp)
            out.append((gx, gy, len(grp)))
        return out

    arrows = cluster(quads, radius=6.0)
    arrows = [(x, y) for x, y, n in arrows if n >= 2]
    if not arrows:
        sl_clusters = cluster(short_lines, radius=8.0)
        arrows = [(x, y) for x, y, n in sl_clusters if 2 <= n <= 12]

    return arrows


def _nearest_arrowhead(bbox, arrows, max_dist=80.0):
    if not arrows:
        return None
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    probe_pts = [
        (cx, cy),
        (x1, y1), (x2, y1), (x1, y2), (x2, y2),
        (x1, cy), (x2, cy), (cx, y1), (cx, y2),
    ]
    best_d, best_a = max_dist, None
    for ax, ay in arrows:
        for px, py in probe_pts:
            d = math.hypot(ax - px, ay - py)
            if d < best_d:
                best_d, best_a = d, (ax, ay)
    return best_a


# =========================================================================
# Span extraction (with normalised bboxes)
# =========================================================================

def extract_pdf_spans(pdf_path, page_num=None):
    doc = fitz.open(str(pdf_path))
    try:
        if page_num is None:
            page_num = len(doc) - 1
        if page_num < 0 or page_num >= len(doc):
            raise ValueError(f"Page {page_num} out of range (0..{len(doc) - 1})")
        page = doc[page_num]
        rect = page.rect
        raw = page.get_text("dict")
        spans = []
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                direction = tuple(line.get("dir", (1.0, 0.0)))
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    raw_bbox = tuple(span.get("bbox", (0, 0, 0, 0)))
                    norm_bbox = _normalise_bbox(raw_bbox, direction)
                    spans.append({
                        "text": text,
                        "bbox": norm_bbox,
                        "font_size": float(span.get("size", 0)),
                        "direction": direction,
                        "page": page_num,
                    })
        return spans, rect
    finally:
        doc.close()


# =========================================================================
# OCR FALLBACK — targeted raster-content recovery
# =========================================================================

_OCR_DPI = 200
_OCR_PAD = 6
_DEDUP_RADIUS = 30


def _ocr_image_region(pix_full, crop_pdf, page_rect, scale):
    try:
        import pytesseract
        from pytesseract import Output
        from PIL import Image
    except ImportError:
        return []

    x1p, y1p, x2p, y2p = crop_pdf
    x1p = max(0, x1p - _OCR_PAD)
    y1p = max(0, y1p - _OCR_PAD)
    x2p = min(page_rect.width,  x2p + _OCR_PAD)
    y2p = min(page_rect.height, y2p + _OCR_PAD)

    px1, py1 = int(x1p * scale), int(y1p * scale)
    px2, py2 = int(x2p * scale), int(y2p * scale)
    if px2 - px1 < 4 or py2 - py1 < 4:
        return []

    try:
        clip = fitz.IRect(px1, py1, px2, py2)
        sub_pix = pix_full.copy()
        img_bytes = pix_full.tobytes("png")
        pil_full = Image.frombytes("RGB", (pix_full.width, pix_full.height),
                                   pix_full.samples)
        pil_crop = pil_full.crop((px1, py1, px2, py2))
    except Exception:
        return []

    try:
        data = pytesseract.image_to_data(
            pil_crop,
            output_type=Output.DICT,
            config="--psm 11 --oem 3",
        )
    except Exception:
        return []

    results = []
    n = len(data["text"])
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        conf = int(data["conf"][i]) if data["conf"][i] != "" else -1
        if conf < 30:
            continue
        wx1 = data["left"][i]
        wy1 = data["top"][i]
        wx2 = wx1 + data["width"][i]
        wy2 = wy1 + data["height"][i]
        bx1_pdf = x1p + wx1 / scale
        by1_pdf = y1p + wy1 / scale
        bx2_pdf = x1p + wx2 / scale
        by2_pdf = y1p + wy2 / scale
        results.append({
            "text": word,
            "bbox_pdf": (bx1_pdf, by1_pdf, bx2_pdf, by2_pdf),
            "cx_pdf": (bx1_pdf + bx2_pdf) / 2,
            "cy_pdf": (by1_pdf + by2_pdf) / 2,
        })
    return results


def _ocr_lines_from_words(words):
    if not words:
        return []
    rows = []
    used = [False] * len(words)
    words_s = sorted(words, key=lambda w: (round(w["cy_pdf"] / 4), w["cx_pdf"]))
    for i, w in enumerate(words_s):
        if used[i]:
            continue
        row = [w]
        used[i] = True
        for j in range(i + 1, len(words_s)):
            if used[j]:
                continue
            if abs(words_s[j]["cy_pdf"] - w["cy_pdf"]) <= 4:
                row.append(words_s[j])
                used[j] = True
        row.sort(key=lambda x: x["cx_pdf"])
        text = " ".join(r["text"] for r in row).strip()
        xs = [r["bbox_pdf"][0] for r in row] + [r["bbox_pdf"][2] for r in row]
        ys = [r["bbox_pdf"][1] for r in row] + [r["bbox_pdf"][3] for r in row]
        merged_bbox = (min(xs), min(ys), max(xs), max(ys))
        cx = (merged_bbox[0] + merged_bbox[2]) / 2
        cy = (merged_bbox[1] + merged_bbox[3]) / 2
        rows.append({"text": text, "bbox_pdf": merged_bbox,
                     "cx_pdf": cx, "cy_pdf": cy})
    return rows


def _is_dup_of_native(cx, cy, text, native_mids, radius=_DEDUP_RADIUS):
    for nx, ny, ntxt in native_mids:
        dist = math.hypot(cx - nx, cy - ny)
        if dist < radius:
            return True
        if ntxt == text and dist < radius * 3:
            return True
    return False


def _ocr_fallback(page, page_rect, drawings, tb_region, drawing_region,
                  arrows, native_candidates):
    try:
        import pytesseract
    except ImportError:
        return []

    pw, ph = page_rect.width, page_rect.height
    scale = _OCR_DPI / 72.0

    try:
        pix_full = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    except Exception:
        return []

    native_mids = []
    for c in native_candidates:
        bx1, by1, bx2, by2 = c["bbox"]
        native_mids.append(
            ((bx1 + bx2) / 2, (by1 + by2) / 2, c["text"].strip())
        )

    tb_found_kinds = {c["type"] for c in native_candidates
                      if c["type"] in TB_ALLOWED_KINDS}

    new_cands = []

    def _make_cand(text, bbox_pdf, in_tb_area):
        t = text.strip()
        if not t or _looks_like_boilerplate(t):
            return None
        if not _looks_like_dimension(t):
            return None
        typ, par, tol = classify_span(t)
        if not typ:
            return None
        if in_tb_area and typ not in TB_ALLOWED_KINDS:
            return None
        if in_tb_area and typ in TB_ALLOWED_KINDS and typ in tb_found_kinds:
            return None
        bx1, by1, bx2, by2 = bbox_pdf
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        if _is_dup_of_native(cx, cy, t, native_mids):
            return None
        
        ax, ay = cx, cy
        
        return {
            "text": t,
            "bbox": bbox_pdf,
            "type": typ,
            "parameter": par,
            "tolerance": tol,
            "anchor_x": ax,
            "anchor_y": ay,
            "source": "ocr",
        }

    def _in_tb_region(bbox_pdf):
        bx1, by1, bx2, by2 = bbox_pdf
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        return (tb_region[0] <= cx <= tb_region[2]
                and tb_region[1] <= cy <= tb_region[3])

    try:
        raw_dict = page.get_text("dict")
    except Exception:
        raw_dict = {"blocks": []}

    for block in raw_dict.get("blocks", []):
        if block.get("type") != 1:
            continue
        bb = block.get("bbox", None)
        if bb is None:
            continue
        crop_pdf = (bb[0], bb[1], bb[2], bb[3])
        in_tb = _in_tb_region(crop_pdf)

        words = _ocr_image_region(pix_full, crop_pdf, page_rect, scale)
        lines = _ocr_lines_from_words(words)
        for ln in lines:
            cand = _make_cand(ln["text"], ln["bbox_pdf"], in_tb)
            if cand:
                new_cands.append(cand)
                native_mids.append((ln["cx_pdf"], ln["cy_pdf"], ln["text"].strip()))
                if cand["type"] in TB_ALLOWED_KINDS:
                    tb_found_kinds.add(cand["type"])

    all_mids = list(native_mids)
    gap_arrows = []
    for ax, ay in arrows:
        covered = False
        for mx, my, _ in all_mids:
            if math.hypot(ax - mx, ay - my) < _DEDUP_RADIUS:
                covered = True
                break
        if not covered:
            gap_arrows.append((ax, ay))

    arrow_pad = 50
    for ax, ay in gap_arrows:
        crop_pdf = (ax - arrow_pad, ay - arrow_pad,
                    ax + arrow_pad, ay + arrow_pad)
        if _in_tb_region(crop_pdf):
            continue
        words = _ocr_image_region(pix_full, crop_pdf, page_rect, scale)
        lines = _ocr_lines_from_words(words)
        for ln in lines:
            cand = _make_cand(ln["text"], ln["bbox_pdf"], False)
            if cand:
                new_cands.append(cand)
                all_mids.append((ln["cx_pdf"], ln["cy_pdf"], ln["text"].strip()))
                native_mids.append((ln["cx_pdf"], ln["cy_pdf"], ln["text"].strip()))

    try:
        draw_crop = (drawing_region[0], drawing_region[1],
                     drawing_region[2], drawing_region[3])
        words = _ocr_image_region(pix_full, draw_crop, page_rect, scale)
        lines = _ocr_lines_from_words(words)
        for ln in lines:
            cand = _make_cand(ln["text"], ln["bbox_pdf"], False)
            if cand:
                new_cands.append(cand)
                native_mids.append(
                    (ln["cx_pdf"], ln["cy_pdf"], ln["text"].strip()))
    except Exception:
        pass

    missing_tb = TB_ALLOWED_KINDS - tb_found_kinds
    if missing_tb:
        tb_crop = (tb_region[0], tb_region[1], tb_region[2], tb_region[3])
        words = _ocr_image_region(pix_full, tb_crop, page_rect, scale)
        lines = _ocr_lines_from_words(words)
        for ln in lines:
            cand = _make_cand(ln["text"], ln["bbox_pdf"], True)
            if cand and cand["type"] in missing_tb:
                new_cands.append(cand)
                native_mids.append((ln["cx_pdf"], ln["cy_pdf"], ln["text"].strip()))
                missing_tb.discard(cand["type"])
                tb_found_kinds.add(cand["type"])

    return new_cands


# =========================================================================
# Main extraction entry point
# =========================================================================

def extract_pdf_candidates(pdf_path, page_num=None, debug_png=None):
    """Extract inspection characteristics from a native PDF page.

    Returns (candidates, stats, page_rect).
    """
    spans, rect = extract_pdf_spans(pdf_path, page_num)
    pw, ph = rect.width, rect.height

    doc = fitz.open(str(pdf_path))
    try:
        _pn = (len(doc) - 1) if page_num is None else page_num
        page = doc[_pn]
        drawings = page.get_drawings()

        grouped = _group_fragmented_spans(spans)

        # --- Title block region ---
        tb_region = _detect_title_block_region(grouped, pw, ph, drawings)

        def _in_tb(bbox):
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            return (tb_region[0] <= cx <= tb_region[2]
                    and tb_region[1] <= cy <= tb_region[3])

        # --- Drawing exclusion zones ---
        table_rects, leader_segs, long_lines = _classify_drawing_lines(drawings, pw, ph)
        
        # Override exclusion zones to an empty list so tables are not suppressed
        excl_zones = [] 

        margin = max(pw, ph) * 0.02
        drawing_region = (
            margin, margin,
            min(pw - margin, tb_region[0]) if tb_region[0] < pw * 0.9 else pw - margin,
            ph - margin,
        )

        arrows = _collect_leader_endpoints(drawings, drawing_region)

        tb_fields = extract_title_block(grouped, tb_region)

        candidates = []
        stats = {
            "total": len(spans),
            "grouped": len(grouped),
            "tb_fields": len(tb_fields),
            "features": 0,
            "ocr_added": 0,
            "rejected_boilerplate": 0,
            "rejected_non_dim": 0,
            "rejected_layout": 0,
            "kept": 0,
            "boilerplate": 0,
            "no_dim": 0,
        }

        for f in tb_fields:
            bx1, by1, bx2, by2 = f["bbox"]
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            candidates.append({
                "text": f["text"],
                "bbox": f["bbox"],
                "type": f["kind"],
                "parameter": f["parameter"],
                "tolerance": "",
                "anchor_x": cx,
                "anchor_y": cy,
            })

        for g in grouped:
            if _in_tb(g["bbox"]):
                continue

            text = g["text"]
            x1, y1, x2, y2 = g["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if cx < pw * 0.03 or cy > ph * 0.98 or cy < ph * 0.01:
                stats["rejected_boilerplate"] += 1
                continue

            if RE_PCF_CODE.search(text):
                stats["rejected_boilerplate"] += 1
                continue

            if _looks_like_boilerplate(text):
                stats["rejected_boilerplate"] += 1
                continue

            if _pt_in_zone(cx, cy, excl_zones):
                stats["rejected_layout"] += 1
                continue

            if not _looks_like_dimension(text):
                typ, par, tol = classify_span(text)
                if not typ:
                    stats["rejected_non_dim"] += 1
                    continue
            else:
                typ, par, tol = classify_span(text)
                if not typ:
                    stats["rejected_non_dim"] += 1
                    continue

            typ, par, tol = classify_span(text)
            if not typ:
                stats["rejected_non_dim"] += 1
                continue

            # Point at the dimension text, not the arrowhead (arrowheads
            # on Tata drawings are dimension-line arrows sitting on the
            # shape outline, not leader-line arrowheads).
            ax, ay = cx, cy

            candidates.append({
                "text": text,
                "bbox": (x1, y1, x2, y2),
                "type": typ,
                "parameter": par,
                "tolerance": tol,
                "anchor_x": ax,
                "anchor_y": ay,
            })
            stats["features"] += 1

        tol_re = re.compile(r"^[±]\s*\d+(?:\.\d+)?$")
        tol_spans = [g for g in grouped
                     if tol_re.match(g["text"].strip())
                     and not _in_tb(g["bbox"])]
        for c in candidates:
            if c.get("tolerance"):
                continue
            if c["type"] != "feature":
                continue
            cx = (c["bbox"][0] + c["bbox"][2]) / 2
            cy = (c["bbox"][1] + c["bbox"][3]) / 2
            best, best_d = None, 30.0
            for t in tol_spans:
                tcx = (t["bbox"][0] + t["bbox"][2]) / 2
                tcy = (t["bbox"][1] + t["bbox"][3]) / 2
                d = math.hypot(cx - tcx, cy - tcy)
                if d < best_d:
                    best_d, best = d, t
            if best:
                c["tolerance"] = best["text"].strip()

        stats["kept"] = len(candidates)
        stats["boilerplate"] = stats["rejected_boilerplate"]
        stats["no_dim"] = stats["rejected_non_dim"]

        protected = {"material_description", "material_specification",
                     "part_number", "envelope", "mass", "surface_protection"}

        # def _filter_dense(cands, radius=28.0, min_neighbors=4):
        #     out = []
        #     feats = [c for c in cands if c["type"] == "feature"]
        #     for c in cands:
        #         if c["type"] in protected:
        #             out.append(c); continue
        #         cx = (c["bbox"][0] + c["bbox"][2]) / 2
        #         cy = (c["bbox"][1] + c["bbox"][3]) / 2
        #         n = 0
        #         for o in feats:
        #             if o is c:
        #                 continue
        #             ox = (o["bbox"][0] + o["bbox"][2]) / 2
        #             oy = (o["bbox"][1] + o["bbox"][3]) / 2
        #             if abs(cx - ox) < radius and abs(cy - oy) < radius:
        #                 n += 1
        #                 if n >= min_neighbors:
        #                     break
        #         if n < min_neighbors:
        #             out.append(c)
        #     return out
        def _filter_dense(cands, radius=28.0, min_neighbors=4):
            # No over-engineering: Let all candidates pass through so tables don't get deleted
            return cands

        before = len(candidates)
        candidates = _filter_dense(candidates)
        stats["features"] = max(0, stats["features"] - (before - len(candidates)))

        ocr_new = _ocr_fallback(
            page, rect, drawings, tb_region, drawing_region,
            arrows, candidates,
        )
        if ocr_new:
            candidates.extend(ocr_new)
            stats["ocr_added"] = len(ocr_new)
            stats["kept"] = len(candidates)

    finally:
        doc.close()

    if debug_png is None:
        try:
            debug_png = Path(pdf_path).with_suffix(".extract_debug.png")
        except Exception:
            debug_png = None
    if debug_png:
        try:
            render_debug_png(pdf_path, candidates, Path(debug_png), page_num)
        except Exception as e:
            print(f"  [debug png failed: {e}]")

    return candidates, stats, rect


# =========================================================================
# Debug PNG renderer
# =========================================================================

def render_debug_png(pdf_path, candidates, output_path, page_num=None, dpi=150):
    from PIL import Image, ImageDraw, ImageFont

    doc = fitz.open(str(pdf_path))
    try:
        if page_num is None:
            page_num = len(doc) - 1
        page = doc[page_num]
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()

    for i, cand in enumerate(candidates, 1):
        px1 = int(cand["bbox"][0] * scale)
        py1 = int(cand["bbox"][1] * scale)
        px2 = int(cand["bbox"][2] * scale)
        py2 = int(cand["bbox"][3] * scale)
        colour = (255, 140, 0) if cand.get("source") == "ocr" else (255, 0, 0)
        draw.rectangle([px1, py1, px2, py2], outline=colour, width=2)
        draw.text((px1, max(0, py1 - 34)),
                  f"{i}:{cand['text'][:18]}", fill=colour, font=font)

        ax = int(cand.get("anchor_x", (cand["bbox"][0] + cand["bbox"][2]) / 2) * scale)
        ay = int(cand.get("anchor_y", (cand["bbox"][1] + cand["bbox"][3]) / 2) * scale)
        draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), outline=(0, 0, 255), width=2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


# =========================================================================
# Standalone CLI
# =========================================================================

if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    p.add_argument("--page", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("pdf_extract_out"))
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    cands, stats, rect = extract_pdf_candidates(a.pdf, a.page)

    print(f"PDF: {a.pdf}")
    print(f"  Page size (pt): {rect.width:.1f} x {rect.height:.1f}")
    print(f"  Total native spans: {stats['total']}")
    print(f"  Grouped: {stats['grouped']}")
    print(f"  Title-block fields: {stats['tb_fields']}")
    print(f"  Drawing features: {stats['features']}")
    print(f"  OCR added: {stats['ocr_added']}")
    print(f"  Rejected boilerplate: {stats['boilerplate']}")
    print(f"  Rejected layout: {stats['rejected_layout']}")
    print(f"  Non-dimension: {stats['no_dim']}")
    print(f"  Final candidates: {stats['kept']}")
    print()
    for c in cands[:40]:
        src = " [OCR]" if c.get("source") == "ocr" else ""
        anchor = f"anchor=({c.get('anchor_x', '?'):.1f},{c.get('anchor_y', '?'):.1f})"
        print(f"  [{c['type']:22s}] {c['text']!r}  {anchor}{src}")

    (a.out / "extracted_candidates.json").write_text(
        json.dumps(cands, indent=2), encoding="utf-8"
    )
    render_debug_png(a.pdf, cands, a.out / "native_text_debug.png", a.page)
    print(f"\nWrote {a.out / 'extracted_candidates.json'}")
    print(f"Wrote {a.out / 'native_text_debug.png'}")