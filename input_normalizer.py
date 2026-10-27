"""input_normalizer.py — PDF/image → NormalizedPage.

PyMuPDF is tried first (works if not blocked). If PyMuPDF import fails
(Application Control policy, missing DLL, etc.), falls back to pdf2image
(uses Poppler's pdftoppm.exe — unaffected by DLL policies).

Padding is opt-in via `apply_padding` (env VISION_PADDING). Gemini needs
a padded square; Mistral does not.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

log = logging.getLogger(__name__)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_PAD_COLOR = (255, 255, 255)

# Same Poppler install folders that process_pdfs.py already uses.
_POPPLER_CANDIDATES = [
    r"C:\Program Files\poppler-26.09.0\Library\bin",
    r"C:\Program Files\poppler\Library\bin",
    r"C:\poppler\Library\bin",
]


def _find_poppler_path() -> str | None:
    if shutil.which("pdftoppm"):
        return None  # already on PATH
    for folder in _POPPLER_CANDIDATES:
        if Path(folder).exists():
            return folder
    return None


def _try_import_fitz():
    """Return pymupdf/fitz module, or None if blocked/missing."""
    try:
        import pymupdf as fitz
        return fitz
    except Exception:
        try:
            import fitz
            return fitz
        except Exception:
            return None


def _pad_to_square(img: Image.Image, enabled: bool = True):
    """Return (square, pad_left, pad_top, content_w, content_h, side).

    When `enabled=False`, returns the image as-is with zero padding
    metadata (Mistral / Groq case). When True, pads to a centered square
    (Gemini case).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size
    if not enabled:
        return img, 0, 0, W, H, max(W, H)
    side = max(W, H)
    if W == H:
        return img, 0, 0, W, H, side
    square = Image.new("RGB", (side, side), _PAD_COLOR)
    pad_left = (side - W) // 2
    pad_top = (side - H) // 2
    square.paste(img, (pad_left, pad_top))
    return square, pad_left, pad_top, W, H, side


@dataclass
class NormalizedPage:
    source_path: Path
    kind: str
    page_num: int
    page_count: int
    width_pt: float
    height_pt: float
    image: Image.Image
    image_dpi: int
    pdf_doc: object = None             # pymupdf Document, or None
    vector_span_count: int = 0
    has_vector_text: bool = False
    pad_left: int = 0
    pad_top: int = 0
    content_w: int = 0
    content_h: int = 0
    square_side: int = 0

    @property
    def content_size(self) -> tuple[int, int]:
        return (self.content_w or self.image.size[0],
                self.content_h or self.image.size[1])

    def close(self):
        if self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
            except Exception:
                pass
            self.pdf_doc = None


# ---------------------------------------------------------------------
# PyMuPDF path (preferred when available)
# ---------------------------------------------------------------------

def _count_text_spans_pymupdf(fitz, doc, page_num: int) -> int:
    try:
        page = doc[page_num]
        raw = page.get_text("dict")
        n = 0
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if (span.get("text") or "").strip():
                        n += 1
        return n
    except Exception:
        return 0


def _pick_best_page_pymupdf(fitz, path: Path) -> int:
    doc = fitz.open(str(path))
    try:
        best_page, best_score = 0, -1
        for i in range(len(doc)):
            p = doc[i]
            try:
                raw = p.get_text("dict")
            except Exception:
                raw = {"blocks": []}
            n_spans = 0
            for b in raw.get("blocks", []):
                if b.get("type") != 0:
                    continue
                for l in b.get("lines", []):
                    n_spans += len(l.get("spans", []))
            try:
                n_draw = len(p.get_drawings())
            except Exception:
                n_draw = 0
            try:
                n_img = len(p.get_images(full=True))
            except Exception:
                n_img = 0
            score = n_spans + 2 * n_draw + 5 * n_img
            if score > best_score:
                best_score, best_page = score, i
        return best_page
    finally:
        doc.close()


def _normalize_pdf_pymupdf(fitz, path, page_num, vision_dpi,
                            vector_min_spans, apply_padding):
    doc = fitz.open(str(path))
    if page_num is None:
        page_num = _pick_best_page_pymupdf(fitz, path)
    if page_num < 0 or page_num >= len(doc):
        doc.close()
        raise ValueError(f"Page {page_num} out of range (0..{len(doc)-1})")
    page = doc[page_num]
    rect = page.rect
    spans = _count_text_spans_pymupdf(fitz, doc, page_num)
    scale = vision_dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    raw = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    square, pl, pt, cw, ch, sd = _pad_to_square(raw, enabled=apply_padding)
    return NormalizedPage(
        source_path=path, kind="pdf", page_num=page_num,
        page_count=len(doc),
        width_pt=float(rect.width), height_pt=float(rect.height),
        image=square, image_dpi=vision_dpi, pdf_doc=doc,
        vector_span_count=spans,
        has_vector_text=spans >= vector_min_spans,
        pad_left=pl, pad_top=pt,
        content_w=cw, content_h=ch, square_side=sd,
    )


# ---------------------------------------------------------------------
# pdf2image / Poppler path (fallback when PyMuPDF is blocked)
# ---------------------------------------------------------------------

def _pick_best_page_poppler(path: Path, total_pages: int) -> int:
    """Try pdfplumber for smart scoring; otherwise pick the last page."""
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            best_page, best_score = 0, -1
            for i, p in enumerate(pdf.pages):
                try:
                    text = p.extract_text() or ""
                except Exception:
                    text = ""
                n_spans = len(text.split())
                try:
                    n_draw = len(p.curves) + len(p.lines) + len(p.rects)
                except Exception:
                    n_draw = 0
                try:
                    n_img = len(p.images)
                except Exception:
                    n_img = 0
                score = n_spans + 2 * n_draw + 5 * n_img
                if score > best_score:
                    best_score, best_page = score, i
            return best_page
    except Exception:
        return max(0, total_pages - 1)


def _normalize_pdf_poppler(path, page_num, vision_dpi, apply_padding):
    try:
        from pdf2image import convert_from_path, pdfinfo_from_path
    except ImportError as exc:
        raise ImportError(
            "pdf2image not installed. Run: pip install pdf2image"
        ) from exc

    poppler_path = _find_poppler_path()
    info = pdfinfo_from_path(str(path), poppler_path=poppler_path)
    total_pages = int(info.get("Pages", 1))

    if page_num is None:
        page_num = _pick_best_page_poppler(path, total_pages)
    if page_num < 0 or page_num >= total_pages:
        raise ValueError(f"Page {page_num} out of range (0..{total_pages-1})")

    images = convert_from_path(
        str(path), dpi=vision_dpi,
        first_page=page_num + 1, last_page=page_num + 1,
        poppler_path=poppler_path,
    )
    if not images:
        raise RuntimeError(
            f"poppler returned no image for page {page_num}"
        )
    raw = images[0].convert("RGB")
    W, H = raw.size
    width_pt = W * 72.0 / vision_dpi
    height_pt = H * 72.0 / vision_dpi

    square, pl, pt, cw, ch, sd = _pad_to_square(raw, enabled=apply_padding)
    return NormalizedPage(
        source_path=path, kind="pdf", page_num=page_num,
        page_count=total_pages,
        width_pt=width_pt, height_pt=height_pt,
        image=square, image_dpi=vision_dpi, pdf_doc=None,
        vector_span_count=0, has_vector_text=False,
        pad_left=pl, pad_top=pt,
        content_w=cw, content_h=ch, square_side=sd,
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def pick_best_page(path: Path) -> int:
    fitz = _try_import_fitz()
    if fitz is not None:
        try:
            return _pick_best_page_pymupdf(fitz, path)
        except Exception as exc:
            log.warning("pick_best_page pymupdf failed: %s", exc)
    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(str(path), poppler_path=_find_poppler_path())
        return _pick_best_page_poppler(path, int(info.get("Pages", 1)))
    except Exception as exc:
        log.warning("pick_best_page poppler failed: %s", exc)
    return 0


def normalize_pdf(path, page_num, vision_dpi, vector_min_spans,
                  apply_padding=False):
    fitz = _try_import_fitz()
    if fitz is not None:
        try:
            return _normalize_pdf_pymupdf(
                fitz, path, page_num, vision_dpi,
                vector_min_spans, apply_padding,
            )
        except Exception as exc:
            log.warning(
                "PyMuPDF normalize failed (%s) — falling back to pdf2image",
                exc,
            )
    log.info("Using pdf2image fallback for %s", path.name)
    return _normalize_pdf_poppler(
        path, page_num, vision_dpi, apply_padding,
    )


def normalize_image(path: Path, apply_padding: bool = False):
    img = Image.open(path)
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    W, H = img.size
    square, pl, pt, cw, ch, sd = _pad_to_square(img, enabled=apply_padding)
    return NormalizedPage(
        source_path=path, kind="image", page_num=0, page_count=1,
        width_pt=float(W), height_pt=float(H),
        image=square, image_dpi=72, pdf_doc=None,
        vector_span_count=0, has_vector_text=False,
        pad_left=pl, pad_top=pt,
        content_w=cw, content_h=ch, square_side=sd,
    )


def normalize_input(path: Path, page_num: int | None = None,
                    vision_dpi: int = 300, vector_min_spans: int = 10,
                    apply_padding: bool = False) -> NormalizedPage:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return normalize_pdf(path, page_num, vision_dpi,
                             vector_min_spans, apply_padding)
    if suffix in IMAGE_EXTS:
        return normalize_image(path, apply_padding)
    raise ValueError(f"Unsupported input format: {suffix}")