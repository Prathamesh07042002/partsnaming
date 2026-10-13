"""Batch-process every PDF AND drawing image; input-agnostic extraction."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pymupdf as fitz
from PIL import Image

import clockwise_numbering as cn
from pipeline_adapter import pipeline_draft


POPPLER_CANDIDATES = [
    r"C:\Program Files\poppler-26.09.0\Library\bin",
    r"C:\Program Files\poppler\Library\bin",
    r"C:\poppler\Library\bin",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def find_poppler_tool(name: str):
    exe = shutil.which(name)
    if exe:
        return exe
    for folder in POPPLER_CANDIDATES:
        candidate = Path(folder) / f"{name}.exe"
        if candidate.exists():
            return str(candidate)
    return None


def get_page_count_fitz(pdf: Path) -> int:
    try:
        doc = fitz.open(str(pdf))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def render_page_fitz(pdf: Path, page_num: int, out_png: Path, dpi: int = 300) -> bool:
    try:
        doc = fitz.open(str(pdf))
        page = doc[page_num]
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        pix.save(str(out_png))
        doc.close()
        return True
    except Exception as e:
        print(f"  [render_page_fitz] {e}")
        return False


def render_source_image(path: Path, fitz_page: int, out_png: Path) -> bool:
    """PDF → 300 DPI PNG; image input → copied as-is."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return render_page_fitz(path, fitz_page, out_png, dpi=300)
    try:
        with Image.open(path) as im:
            im.convert("RGB").save(out_png)
        return True
    except Exception as e:
        print(f"  [render_source_image] {e}")
        return False


def process_one(path: Path, root: Path, page_override: int, force: bool):
    part = path.stem
    out_dir = root / "outputs" / part
    annotated = out_dir / f"{part}.png"
    excel = out_dir / f"{part}.xlsx"
    reviewed_path = root / "reviewed_items" / f"{part}.json"

    if not force and annotated.exists() and excel.exists():
        print(f"[SKIP]  {part}  (outputs exist)")
        return "skip"

    is_pdf = path.suffix.lower() == ".pdf"

    if is_pdf:
        total_pages = get_page_count_fitz(path)
        if page_override > 0:
            page_1based = page_override
        else:
            page_1based = total_pages if total_pages > 0 else 1
        fitz_page = max(0, page_1based - 1)
    else:
        page_1based = 1
        fitz_page = 0

    with tempfile.TemporaryDirectory(prefix="partsnaming-render-") as tmp:
        image = Path(tmp) / f"{part}.png"
        if not render_source_image(path, fitz_page, image):
            print(f"[FAIL]  {part}: render failed")
            return "fail"

        out_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image) as im:
            png_size = im.size

        if reviewed_path.exists():
            raw = json.loads(reviewed_path.read_text(encoding="utf-8"))
            mode_label = "items"
        else:
            try:
                raw, _ = pipeline_draft(
                    path,
                    fitz_page if is_pdf else None,
                    target_size=png_size,
                )
                mode_label = "pipeline"
            except Exception as e:
                print(f"[WARN] {part}: pipeline failed ({e}); falling back to OCR")
                raw = cn.ocr_draft(image)
                mode_label = "ocr"

        items = cn.number_items(cn.normalise_items(raw))
        cn.validate_items(items, image, allow_identical_marker_target=True)
        cn.write_excel(items, excel, image)
        cn.annotate(image, items, annotated)

        (out_dir / f"{part}.json").write_text(
            json.dumps(items, indent=2), encoding="utf-8"
        )

        print(f"[OK]    {part}  (page {page_1based}, {mode_label}, {len(items)} items)")
        return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--input-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "sources",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--only", type=str, default="",
                help="Process only this file stem (e.g. 550629507102)")
    args = parser.parse_args()

    root = args.input_dir.resolve()
    if not root.exists():
        print(f"Input folder not found: {root}", file=sys.stderr)
        return 2

    _EXCLUDE_TOKENS = (
        "extract_debug", "_masked", "annotated_", ".extract.",
        "native_text_debug", "overlay_debug",
    )

    def _is_source_file(p: Path) -> bool:
        if not p.is_file():
            return False
        if p.suffix.lower() not in ({".pdf"} | IMAGE_EXTS):
            return False
        low = p.name.lower()
        return not any(tok in low for tok in _EXCLUDE_TOKENS)

    files = sorted(p for p in root.iterdir() if _is_source_file(p))
    if args.only:
        files = [f for f in files if f.stem == args.only]
    if not files:
        print(f"No PDF/image files found in {root}")
        return 0
    print(f"Found {len(files)} input file(s)\n")

    stats = {"ok": 0, "skip": 0, "fail": 0}
    for f in files:
        stats[process_one(f, root, args.page, args.force)] += 1

    print()
    print("Summary")
    print(f"  Processed : {stats['ok']}")
    print(f"  Skipped   : {stats['skip']}")
    print(f"  Failed    : {stats['fail']}")
    return 1 if stats["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())