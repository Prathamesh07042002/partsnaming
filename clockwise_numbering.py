"""
clockwise_numbering.py – bubble placement, numbering, annotation and Excel output.

Changes vs original:
  * pdf_draft() uses the new anchor_x/anchor_y from pdf_extractor so bubbles
    land on the actual leader endpoint rather than the text bbox centre.
  * place_bubble() now takes an optional explicit target override so the
    caller can pass the arrowhead coordinate as the leader target.
  * _dedupe() compares on target position (arrowhead) rather than marker
    position so two different callouts that happen to get the same bubble
    slot are not silently dropped.
  * _spread() respects image boundaries with a harder clamp.
  * Everything else (number_items, annotate, write_excel, ocr_draft,
    normalise_items, validate_items, clockwise_angle) is preserved verbatim.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from openpyxl import load_workbook
from openpyxl.cell.cell import MergedCell

_ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

BLUE = (32, 91, 177)
BUBBLE_RADIUS = 55
BUBBLE_OFFSET = 30

# Junk filter used ONLY by the OCR fallback path.
RE_JUNK_OCR = re.compile(
    r"do not scale|if in doubt|first angle|third angle|"
    r"all dims\b|all dimensions|dimensions in mm|"
    r"drg drawn|drawing drawn|angle of projection|"
    r"supplier drg release|"
    r"nominal dimension|tolerance in um|"
    r"classification of characteristics|"
    r"tata motors|tata group|erc[-\s]*pune|"
    r"shall be marked|marked as per|"
    r"this drawing is the sole|it should not be copied|"
    r"without the written|"
    r"for all vehicle|for m1\b|for bar code|"
    r"sheet no\.|original drawing|size a[0-9]|"
    r"lpt\s*\d|"
    r"tack weld|^\s*at \d+ places?\s*$|"
    r"^\s*qty[.\s]|"
    r"hex\s+(?:weld\s+)?(?:nut|bolt)|"
    r"^section\s+[a-z]|"
    r"^view\s+[a-z]\s*$|"
    r"^detail\s+[a-z]\s*$",
    re.IGNORECASE,
)


# =========================================================================
# Bubble placement
# =========================================================================

def place_bubble(bbox, w, h, radius=BUBBLE_RADIUS, offset=BUBBLE_OFFSET,
                 target_override=None):
    """Compute (bubble_x, bubble_y, target_x, target_y) in pixel coords.

    target_override: if given, use it as (tx, ty) instead of bbox centre.
    The bubble is placed adjacent to the target if a target override is
    supplied; otherwise it falls back to the original 8-candidate scheme
    around the bbox.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if target_override is not None:
        tx, ty = int(target_override[0]), int(target_override[1])
    else:
        tx, ty = cx, cy

    # Eight candidate positions around the bbox (unchanged from original)
    candidates = [
        (x2 + offset + radius, cy),
        (x1 - offset - radius, cy),
        (cx, y1 - offset - radius),
        (cx, y2 + offset + radius),
        (x2 + offset + radius, y1 - offset - radius),
        (x1 - offset - radius, y1 - offset - radius),
        (x2 + offset + radius, y2 + offset + radius),
        (x1 - offset - radius, y2 + offset + radius),
    ]
    # When we have a real arrowhead target, prefer candidates that are
    # on the opposite side from the arrowhead (so the leader is visible).
    if target_override is not None:
        # Sort candidates by distance from target so the bubble sits near
        # the callout text but the leader still points at the arrowhead.
        candidates.sort(key=lambda c: math.hypot(c[0] - tx, c[1] - ty),
                        reverse=True)

    for bx, by in candidates:
        if radius + 5 <= bx <= w - radius - 5 and radius + 5 <= by <= h - radius - 5:
            return int(bx), int(by), tx, ty

    # Fallback: clamp to page
    return (
        int(max(radius + 5, min(w - radius - 5, x2 + offset + radius))),
        int(max(radius + 5, min(h - radius - 5, cy))),
        tx, ty,
    )


def _dedupe(items):
    """Remove duplicate items (same text at the same source location).

    Compares the ORIGINAL text bbox, NOT the arrowhead target — many
    dimensions legitimately share one leader line, so target-based
    dedupe would silently drop valid callouts.
    """
    out = []
    for it in items:
        sx = it.get("source_x", it["x"])
        sy = it.get("source_y", it["y"])
        dup = False
        for k in out:
            ksx = k.get("source_x", k["x"])
            ksy = k.get("source_y", k["y"])
            if (it["specification"] == k["specification"]
                    and abs(sx - ksx) < 20
                    and abs(sy - ksy) < 20):
                dup = True
                break
        if not dup:
            out.append(it)
    return out



def _spread(items, min_distance=115.0, width=0, height=0, iterations=200):
    """Push bubble centres apart so they do not overlap."""
    n = len(items)
    if n < 2:
        return
    for _ in range(iterations):
        moved = False
        for i in range(n):
            a = items[i]
            for j in range(i + 1, n):
                b = items[j]
                dx, dy = a["x"] - b["x"], a["y"] - b["y"]
                d = math.hypot(dx, dy)
                if d < min_distance:
                    if d < 1e-6:
                        dx, dy, d = 1.0, -1.0, math.hypot(1.0, 1.0)
                    push = (min_distance - d) / 2.0
                    ux, uy = dx / d, dy / d
                    a["x"] += ux * push
                    a["y"] += uy * push
                    b["x"] -= ux * push
                    b["y"] -= uy * push
                    moved = True
        if not moved:
            break
    if width and height:
        r = BUBBLE_RADIUS + 5
        for it in items:
            it["x"] = int(max(r, min(width - r, it["x"])))
            it["y"] = int(max(r, min(height - r, it["y"])))

def mask_bubbles(src_path, items, dst_path, pad=8):
    """Copy src → dst with every existing bubble erased to white.

    This is used before a second OCR pass on the annotated image so
    that the numbers drawn *inside* bubbles are not re-detected as
    dimensions (which caused double bubbles).
    """
    im = Image.open(src_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    r = BUBBLE_RADIUS + pad
    for it in items:
        x, y = int(it["x"]), int(it["y"])
        draw.ellipse((x - r, y - r, x + r, y + r), fill="white")
    im.save(dst_path)

# =========================================================================
# PDF-native draft  (main entry point called by process_pdfs.py)
# =========================================================================

def pdf_draft(pdf_path: Path, page_num: int | None = None, target_size=None):
    """Extract inspection items from a native PDF and return (items, img_size).

    Uses pdf_extractor.extract_pdf_candidates() which now returns
    anchor_x / anchor_y (the nearest arrowhead in PDF points) per candidate.
    """
    import pdf_extractor

    cands, stats, rect = pdf_extractor.extract_pdf_candidates(pdf_path, page_num)
    print(f"  Native PDF spans:     {stats['total']}")
    print(f"  Rejected boilerplate: {stats['boilerplate']}")
    print(f"  Rejected layout:      {stats.get('rejected_layout', 0)}")
    print(f"  Non-dimension:        {stats['no_dim']}")
    print(f"  Kept candidates:      {stats['kept']}")

    # Scale factors: PDF points → pixel coords of the rendered PNG
    if target_size:
        img_w, img_h = target_size
        sx = img_w / rect.width
        sy = img_h / rect.height
    else:
        sx = sy = 300 / 72.0
        img_w = int(rect.width * sx)
        img_h = int(rect.height * sy)

    items = []
    for c in cands:
        x1, y1, x2, y2 = c["bbox"]

        # Convert PDF-pt bbox to pixel bbox
        px1 = int(x1 * sx)
        py1 = int(y1 * sy)
        px2 = int(x2 * sx)
        py2 = int(y2 * sy)

        # Convert arrowhead anchor to pixel coords
        ax_pt = c.get("anchor_x")
        ay_pt = c.get("anchor_y")
        if ax_pt is not None and ay_pt is not None:
            target_px = (ax_pt * sx, ay_pt * sy)
        else:
            target_px = None

        bx, by, tx, ty = place_bubble(
            (px1, py1, px2, py2), img_w, img_h,
            target_override=target_px,
        )

        items.append({
            "type":          c["type"],
            "parameter":     c["parameter"],
            "specification": c["text"],
            "tolerance":     c.get("tolerance", ""),
            "x":  bx, "y":  by,
            "target_x": tx, "target_y": ty,
            # NEW: original text bbox in pixel coords — used for dedupe
            "source_x":  px1, "source_y":  py1,
            "source_x2": px2, "source_y2": py2,
            "notes": "Extracted from native PDF",
        })

    items = _dedupe(items)
    _spread(items, min_distance=115.0, width=img_w, height=img_h)
    return items, (img_w, img_h)


# =========================================================================
# OCR fallback --> kept for compatibility
# =========================================================================

def _bbox_back(angle, x1, y1, x2, y2, w, h):
    """Map bbox from a rotated image back to original coords."""
    if angle == 0:   return (x1, y1, x2, y2)
    if angle == 90:  return (w - y2, x1, w - y1, x2)
    if angle == 180: return (w - x2, h - y2, w - x1, h - y1)
    if angle == 270: return (y1, h - x2, y2, h - x1)
    return (x1, y1, x2, y2)


def ocr_draft(image_path: Path):
    """Multi-angle OCR (0/90/180/270) with symbol-friendly config."""
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise RuntimeError("pip install pytesseract") from exc

    import pdf_extractor

    with Image.open(image_path).convert("L") as img:
        w, h = img.size
        all_lines = []
        for angle in (0, 90, 180, 270):
            rotated = img.rotate(angle, expand=True) if angle else img
            try:
                data = pytesseract.image_to_data(
                    rotated, output_type=Output.DICT,
                    config="--psm 11 --oem 3",
                )
            except pytesseract.TesseractNotFoundError as exc:
                raise RuntimeError("Install Tesseract OCR.") from exc
            except Exception:
                continue

            groups: dict[tuple, list[int]] = defaultdict(list)
            for i, t in enumerate(data["text"]):
                if t.strip():
                    groups[(data["block_num"][i], data["par_num"][i],
                            data["line_num"][i])].append(i)

            for idxs in groups.values():
                text = " ".join(data["text"][i] for i in idxs).strip()
                if not text or RE_JUNK_OCR.search(text):
                    continue
                x1 = min(data["left"][i] for i in idxs)
                y1 = min(data["top"][i] for i in idxs)
                x2 = max(data["left"][i] + data["width"][i] for i in idxs)
                y2 = max(data["top"][i] + data["height"][i] for i in idxs)
                bx1, by1, bx2, by2 = _bbox_back(angle, x1, y1, x2, y2, w, h)
                all_lines.append((text, bx1, by1, bx2, by2))

    items = []
    for text, x1, y1, x2, y2 in all_lines:
        typ, par, tol = pdf_extractor.classify_span(text)
        if not typ:
            continue
        bx, by, tx, ty = place_bubble((x1, y1, x2, y2), w, h)
        items.append({
            "type": typ, "parameter": par, "specification": text,
            "tolerance": tol, "x": bx, "y": by,
            "target_x": tx, "target_y": ty,
            "source_x": x1, "source_y": y1,
            "source_x2": x2, "source_y2": y2,
            "notes": "Detected by OCR",
        })

    items = _dedupe(items)
    _spread(items, min_distance=115.0, width=w, height=h)
    return items


def merge_ocr_items(existing, ocr_new, exclusion_radius=30.0):
    """Add OCR items that aren't already covered by an existing bubble.

    An OCR hit is dropped if its centre lies within `exclusion_radius` px
    of any existing item's target (that's the leader endpoint / bubble).
    """
    merged = list(existing)
    for n in ocr_new:
        cx = n.get("source_x", n["x"])
        cy = n.get("source_y", n["y"])
        dup = False
        for e in merged:
            ex = e.get("target_x", e["x"])
            ey = e.get("target_y", e["y"])
            if math.hypot(cx - ex, cy - ey) < exclusion_radius:
                dup = True
                break
        if not dup:
            merged.append(n)
    return merged

# =========================================================================
# Clockwise ordering  
# =========================================================================

def clockwise_angle(x, y, cx, cy):
    return (math.atan2(x - cx, -(y - cy)) + 2 * math.pi) % (2 * math.pi)


def number_items(items):
    buckets: dict[str, list] = defaultdict(list)
    for it in items:
        buckets[it["type"]].append(it)

    feats = buckets["feature"]
    views: dict[str, list] = defaultdict(list)
    for f in feats:
        views[str(f.get("view", "main"))].append(f)

    ordered_feats = []
    for key in sorted(views, key=lambda k: sum(i["x"] for i in views[k]) / len(views[k])):
        view = views[key]
        cx = sum(i["x"] for i in view) / len(view)
        cy = sum(i["y"] for i in view) / len(view)
        ordered_feats.extend(
            sorted(view, key=lambda i: clockwise_angle(i["x"], i["y"], cx, cy))
        )

    ordered = (
        buckets["material_description"][:1]
        + buckets["material_specification"][:1]
        + buckets["part_number"][:1]
        + buckets["envelope"][:1]
        + buckets["surface_protection"][:1]
        + ordered_feats
        + buckets["mass"][:1]
        + buckets["note"][:1]
    )
    for s, item in enumerate(ordered, start=1):
        item["serial"] = s
    return ordered


# =========================================================================
# Normalise / validate items  
# =========================================================================

def normalise_items(raw):
    required = {"type", "parameter", "specification", "x", "y"}
    allowed = {"material_description", "material_specification",
               "part_number", "envelope", "surface_protection",
               "feature", "mass", "note"}
    out = []
    for i, it in enumerate(raw, 1):
        miss = required - it.keys()
        if miss:
            raise ValueError(f"Item {i} missing: {', '.join(sorted(miss))}")
        if it["type"] not in allowed:
            raise ValueError(f"Item {i} bad type: {it['type']}")
        r = dict(it)
        r.setdefault("tolerance", "")
        r.setdefault("notes", "")
        r.setdefault("target_x", r["x"])
        r.setdefault("target_y", r["y"])
        out.append(r)
    return out


def validate_items(items, image_path, allow_identical_marker_target=False):
    with Image.open(image_path) as im:
        w, h = im.size
    for it in items:
        for k, lim in (("x", w), ("target_x", w), ("y", h), ("target_y", h)):
            v = it[k]
            if not isinstance(v, (int, float)) or not 0 <= v <= lim:
                raise ValueError(f"{it['parameter']}: {k}={v!r} outside {w}x{h}")


# =========================================================================
# Excel output
# =========================================================================

def _sanitize(value):
    if isinstance(value, str):
        return _ILLEGAL_XLSX.sub("", value)
    return value


def safe_write(sheet, row, col, value):
    """Write a value; if the target is inside a merged range, write to the
    top-left cell of that range instead.  Formatting is never touched."""
    value = _sanitize(value)
    cell = sheet.cell(row=row, column=col)
    if isinstance(cell, MergedCell):
        for mr in sheet.merged_cells.ranges:
            if cell.coordinate in mr:
                sheet.cell(row=mr.min_row, column=mr.min_col).value = value
                return
    else:
        cell.value = value

_TOL_RE = re.compile(r"[±]\s*\d+(?:\.\d+)?")

def _tol_or_extract(spec, cur):
    if cur and str(cur).strip():
        return cur
    m = _TOL_RE.search(spec or "")
    return m.group(0) if m else ""

def write_excel(items, output_path, image_path=None):
    """Fill the AOI template. Only A-D written; formatting preserved.

    Header cells:
        C2  PART NO             (filename stem, fallback: extracted)
        C3  PART DESCRIPTION    (only if extractor produced one)
        C4  DRAWING NO          (same as C2)
        C5  DRAWING REV & DATE  (only if extractor produced one)
        N2  DATE:-DD.MM.YYYY

    Data table:
        Row 8   A8:D8 fixed headers (SR | PARAMETER | Specifications | Tolerance)
        Row 9   sub-headers (left untouched)
        Row 10+ one row per item; only columns A-D written.

    Columns E onward are never touched.  Template file itself is not
    modified — a copy is written to `output_path`.
    """
    tpl = Path(__file__).resolve().parent / "554743810147 AOI.xlsx"
    if not tpl.exists():
        raise FileNotFoundError(f"Template not found: {tpl}")

    wb = load_workbook(tpl)          # keeps every format / merge / formula
    sh = wb.worksheets[0]            # sheet 1: 554743810147AOI

    # ---------------- Header cells ----------------
    def _first_of(kind):
        for it in items:
            if it.get("type") == kind:
                v = str(it.get("specification", "")).strip()
                return v or None
        return None

    part_no = image_path.stem.split("-")[0] if image_path else ""
    if not part_no:
        part_no = _first_of("part_number") or ""

    safe_write(sh, 2, 3, part_no)                 # C2  PART NO
    safe_write(sh, 4, 3, part_no)                 # C4  DRAWING NO

    desc = _first_of("part_description")
    if desc:
        safe_write(sh, 3, 3, desc)                # C3

    rev = _first_of("drawing_rev") or _first_of("drawing_rev_date")
    if rev:
        safe_write(sh, 5, 3, rev)                 # C5

    safe_write(sh, 2, 14,
               f"DATE:-{datetime.date.today().strftime('%d.%m.%Y')}")  # N2

    # ---------------- Data table ----------------
    DATA_START = 10

    # Clear stale A–D values from row 10 down.  Upper bound covers both
    # the sheet's current max row and the rows we are about to write so
    # a previous longer run leaves no orphan rows.
    last_to_clear = max(sh.max_row, DATA_START + len(items) - 1)
    for r in range(DATA_START, last_to_clear + 1):
        for c in (1, 2, 3, 4):                    # A, B, C, D only
            cell = sh.cell(row=r, column=c)
            if not isinstance(cell, MergedCell):
                cell.value = None

    row = DATA_START
    for it in items:
        safe_write(sh, row, 1, it["serial"])                              # A
        safe_write(sh, row, 2, it["parameter"])                           # B
        safe_write(sh, row, 3, it["specification"])                       # C
        safe_write(sh, row, 4,
                _tol_or_extract(it["specification"],
                                it.get("tolerance", "")))   # ← nayi            # D
        row += 1

    wb.save(output_path)

# =========================================================================
# Annotation  
# =========================================================================

def annotate(image_path, items, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    radius = BUBBLE_RADIUS

    for item in items:
        x, y = int(item["x"]), int(item["y"])
        tx, ty = int(item["target_x"]), int(item["target_y"])

        dx, dy = tx - x, ty - y
        d = math.hypot(dx, dy) or 1.0
        if d > radius + 2:
            ex = x + dx / d * radius
            ey = y + dy / d * radius
            draw.line((ex, ey, tx, ty), fill=BLUE, width=2)
            draw.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill=BLUE)

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="white", outline=BLUE, width=3,
        )
        text = str(item["serial"])
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.text((x - tw / 2 - tb[0], y - th / 2 - tb[1]),
                  text, fill=BLUE, font=font)

    image.save(output_path)


# =========================================================================
# CLI  
# =========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path, nargs="?", default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--items", type=Path)
    g.add_argument("--ocr", action="store_true")
    g.add_argument("--pdf", type=Path, help="Native PDF (preferred over --ocr)")
    p.add_argument("--page", type=int, default=None, help="PDF page index (0-based)")
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    a = p.parse_args()

    a.output_dir.mkdir(parents=True, exist_ok=True)

    if a.items:
        raw = json.loads(a.items.read_text(encoding="utf-8"))
    elif a.pdf:
        raw, _ = pdf_draft(a.pdf, a.page)
    else:
        if not a.image:
            raise SystemExit("--ocr needs an image path as positional arg")
        raw = ocr_draft(a.image)

    items = number_items(normalise_items(raw))
    print(f"  After ordering: {len(items)} items")

    if a.image:
        validate_items(items, a.image, allow_identical_marker_target=True)
        annotate(a.image, items, a.output_dir / "annotated_drawing.png")
        write_excel(items, a.output_dir / "extracted_dimensions.xlsx", a.image)
    else:
        print("Note: PDF mode requires --output-dir; annotation is done by process_pdfs.py")

    if a.ocr:
        (a.output_dir / "ocr_review_items.json").write_text(
            json.dumps(items, indent=2), encoding="utf-8"
        )
    print(f"Done. Items: {len(items)}")


if __name__ == "__main__":
    main()