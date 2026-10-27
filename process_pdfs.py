"""Batch-process PDFs / images with graceful PyMuPDF fallback."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

# PyMuPDF is optional — Application Control may block its DLL.
try:
    import pymupdf as fitz
    _HAS_FITZ = True
except Exception as _e:
    fitz = None
    _HAS_FITZ = False
    print(f"[warn] PyMuPDF unavailable ({type(_e).__name__}) "
          f"— using pdf2image fallback")

import clockwise_numbering as cn
from pipeline_adapter import pipeline_draft
from input_normalizer import pick_best_page


POPPLER_CANDIDATES = [
    r"C:\Program Files\poppler-26.09.0\Library\bin",
    r"C:\Program Files\poppler\Library\bin",
    r"C:\poppler\Library\bin",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _find_poppler_path():
    if shutil.which("pdftoppm"):
        return None
    for folder in POPPLER_CANDIDATES:
        if Path(folder).exists():
            return folder
    return None


def get_page_count(pdf: Path) -> int:
    if _HAS_FITZ:
        try:
            doc = fitz.open(str(pdf))
            n = len(doc)
            doc.close()
            return n
        except Exception:
            pass
    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(
            str(pdf), poppler_path=_find_poppler_path(),
        )
        return int(info.get("Pages", 1))
    except Exception as e:
        print(f"  [page-count] {e}")
        return 0


def render_pdf_page_poppler(pdf: Path, page_num: int,
                             out_png: Path, dpi: int = 300) -> bool:
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(
            str(pdf), dpi=dpi,
            first_page=page_num + 1, last_page=page_num + 1,
            poppler_path=_find_poppler_path(),
        )
        if not images:
            return False
        images[0].convert("RGB").save(out_png)
        return True
    except Exception as e:
        print(f"  [render_poppler] {e}")
        return False


def render_source_image(path: Path, fitz_page: int, out_png: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if _HAS_FITZ:
            try:
                doc = fitz.open(str(path))
                page = doc[fitz_page]
                scale = 300 / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                                      alpha=False)
                pix.save(str(out_png))
                doc.close()
                return True
            except Exception as e:
                print(f"  [render_fitz] {e} — trying poppler")
        return render_pdf_page_poppler(path, fitz_page, out_png, dpi=300)
    try:
        with Image.open(path) as im:
            im.convert("RGB").save(out_png)
        return True
    except Exception as e:
        print(f"  [render_image] {e}")
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
        if page_override > 0:
            fitz_page = page_override - 1
        else:
            try:
                fitz_page = pick_best_page(path)
            except Exception as e:
                print(f"  [page-pick] {e} — using last page")
                total = get_page_count(path)
                fitz_page = max(0, total - 1) if total > 0 else 0
        page_1based = fitz_page + 1
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
                print(f"[WARN] {part}: pipeline failed ({e}); "
                      f"falling back to OCR")
                raw = cn.ocr_draft(image)
                mode_label = "ocr"

        items = cn.number_items(cn.normalise_items(raw))
        cn.validate_items(items, image, allow_identical_marker_target=True)
        cn.write_excel(items, excel, image)
        cn.annotate(image, items, annotated)

        (out_dir / f"{part}.json").write_text(
            json.dumps(items, indent=2), encoding="utf-8",
        )

        print(f"[OK]    {part}  (page {page_1based}, {mode_label}, "
              f"{len(items)} items)")
        return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path(__file__).resolve().parent / "sources",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--page", type=int, default=0,
                        help="Force 1-based page; 0 = auto-pick")
    parser.add_argument("--only", type=str, default="")
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