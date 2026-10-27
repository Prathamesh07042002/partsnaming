from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


BLUE = (32, 91, 177)
TITLE_TYPES = ("material_description", "material_specification")


def clockwise_angle(x: float, y: float, centre_x: float, centre_y: float) -> float:
    """Angle starting at 12 o'clock and increasing clockwise."""
    return (math.atan2(x - centre_x, -(y - centre_y)) + 2 * math.pi) % (2 * math.pi)


def number_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply title-block, clockwise, mass and final-note ordering rules."""
    descriptions = [item for item in items if item["type"] == "material_description"]
    specifications = [item for item in items if item["type"] == "material_specification"]
    features = [item for item in items if item["type"] == "feature"]
    masses = [item for item in items if item["type"] == "mass"]
    notes = [item for item in items if item["type"] == "note"]

    # Complete each view before moving to the next; views are ordered left-to-right.
    views: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in features:
        views[str(feature.get("view", "main"))].append(feature)

    sorted_views = sorted(views.values(), key=lambda view: sum(i["x"] for i in view) / len(view))
    ordered_features: list[dict[str, Any]] = []
    for view in sorted_views:
        if all("order" in item for item in view):
            ordered_features.extend(sorted(view, key=lambda item: item["order"]))
            continue
        centre_x = sum(item["x"] for item in view) / len(view)
        centre_y = sum(item["y"] for item in view) / len(view)
        ordered_features.extend(sorted(view, key=lambda item: clockwise_angle(item["x"], item["y"], centre_x, centre_y)))

    ordered = descriptions[:1] + specifications[:1] + ordered_features + masses[:1] + notes[:1]
    for serial, item in enumerate(ordered, start=1):
        item["serial"] = serial
    return ordered


def normalise_items(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {"type", "parameter", "specification", "x", "y"}
    allowed = {"material_description", "material_specification", "feature", "mass", "note"}
    items = []
    for index, item in enumerate(raw, start=1):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Item {index} is missing: {', '.join(sorted(missing))}")
        if item["type"] not in allowed:
            raise ValueError(f"Item {index} has unsupported type: {item['type']}")
        result = dict(item)
        result.setdefault("tolerance", "N/A")
        result.setdefault("notes", "")
        result.setdefault("target_x", result["x"])
        result.setdefault("target_y", result["y"])
        items.append(result)
    return items


def validate_items(items: list[dict[str, Any]], image_path: Path) -> None:
    """Reject invalid annotation coordinates before any output is created."""
    with Image.open(image_path) as image:
        width, height = image.size
    for item in items:
        for key, limit in (("x", width), ("target_x", width), ("y", height), ("target_y", height)):
            value = item[key]
            if not isinstance(value, (int, float)) or not 0 <= value < limit:
                raise ValueError(
                    f"{item['parameter']}: {key}={value!r} is outside image bounds "
                    f"({width} x {height}). Review the JSON marker coordinates."
                )
        if item["x"] == item["target_x"] and item["y"] == item["target_y"]:
            raise ValueError(
                f"{item['parameter']}: marker and leader target are identical. "
                "Place the marker in whitespace and target the exact callout."
            )


def ocr_draft(image_path: Path) -> list[dict[str, Any]]:
    """Produce a reviewable draft from OCR; do not use unreviewed values for QA."""
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise RuntimeError("Install OCR support first: python -m pip install pytesseract") from exc

    try:
        data = pytesseract.image_to_data(Image.open(image_path), output_type=Output.DICT, config="--psm 11")
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Install Tesseract OCR and add it to PATH before using --ocr.") from exc

    # Build OCR lines. Only dimension-like text becomes a feature candidate.
    lines: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, text in enumerate(data["text"]):
        if text.strip():
            lines[(data["block_num"][index], data["par_num"][index], data["line_num"][index])].append(index)

    result: list[dict[str, Any]] = []
    for indices in lines.values():
        text = " ".join(data["text"][i] for i in indices).strip()
        x = min(data["left"][i] for i in indices)
        y = min(data["top"][i] for i in indices)
        right = max(data["left"][i] + data["width"][i] for i in indices)
        bottom = max(data["top"][i] + data["height"][i] for i in indices)
        centre_x, centre_y = (x + right) // 2, (y + bottom) // 2
        lower = text.lower()
        item_type = None
        parameter = ""
        if "material description" in lower:
            item_type, parameter = "material_description", "Material / thickness"
        elif re.search(r"\b(note|general tolerance)\b", lower):
            item_type, parameter = "note", "General note"
        elif re.search(r"\b(final mass|mass|kg)\b", lower):
            item_type, parameter = "mass", "Final mass"
        elif re.search(r"(\d|\br\s*\d|dia|ø|phi|±|\+\s*\d|\-\s*\d)", lower):
            item_type, parameter = "feature", "OCR drawing callout"

        if item_type:
            result.append({
                "type": item_type, "parameter": parameter, "specification": text,
                "tolerance": "OCR review required", "x": centre_x, "y": centre_y,
                "target_x": centre_x, "target_y": centre_y, "notes": "Draft detected by OCR",
            })
    return result


def write_excel(items: list[dict[str, Any]], output_path: Path) -> None:
    """Create the formatted Excel inspection table requested for each drawing."""
    headers = ["Sr. No.", "Parameter", "Specification", "Tolerance", "Location / Notes"]
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Extracted Dimensions"
    sheet.append(headers)
    for item in items:
        sheet.append([item["serial"], item["parameter"], item["specification"], item["tolerance"], item["notes"]])

    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    widths = (11, 34, 48, 22, 32)
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    workbook.save(output_path)


def annotate(image_path: Path, items: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 24)
    radius = 24
    for item in items:
        x, y = item["x"], item["y"]
        draw.line((x, y, item["target_x"], item["target_y"]), fill=BLUE, width=3)
        # White fill ensures the leader cannot obscure the serial number.
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline=BLUE, width=4)
        text = str(item["serial"])
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text((x - (right - left) / 2, y - (bottom - top) / 2 - top), text, fill=BLUE, font=font)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clockwise engineering-drawing labels and table.")
    parser.add_argument("image", type=Path, help="Input drawing image")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--items", type=Path, help="Reviewed item JSON")
    source.add_argument("--ocr", action="store_true", help="Create a reviewable OCR draft")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(args.items.read_text(encoding="utf-8")) if args.items else ocr_draft(args.image)
    items = number_items(normalise_items(raw))
    validate_items(items, args.image)
    write_excel(items, args.output_dir / "extracted_dimensions.xlsx")
    annotate(args.image, items, args.output_dir / "annotated_drawing.png")
    if args.ocr:
        (args.output_dir / "ocr_review_items.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"Created {args.output_dir / 'annotated_drawing.png'}")
    print(f"Created {args.output_dir / 'extracted_dimensions.xlsx'}")


if __name__ == "__main__":
    main()
