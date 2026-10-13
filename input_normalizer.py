"""input_normalizer.py – PDF/image → NormalizedPage with square padding.

The rendered raster is padded to a centered square at extraction time.
Gemini receives a perfect square, and the padding metadata is stored on
the page so the vision parser can reverse it exactly.

Coordinate contract:
  * page.width_pt / height_pt = PDF content size (points)
  * page.content_w / content_h = unpadded raster size (pixels)
  * page.pad_left / pad_top = padding offset (pixels)
  * page.square_side = padded canvas side (pixels)
  * page.image = the PADDED SQUARE (for vision only — not for display)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from PIL import Image, ImageOps


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_PAD_COLOR = (255, 255, 255)


def _pad_to_square(img: Image.Image):
    """Return (square, pad_left, pad_top, content_w, content_h, side)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size
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
    kind: str                          # "pdf" | "image"
    page_num: int
    page_count: int
    width_pt: float                    # PDF content width (points)
    height_pt: float                   # PDF content height (points)
    image: Image.Image                 # PADDED SQUARE (for vision)
    image_dpi: int
    pdf_doc: "fitz.Document | None" = None
    vector_span_count: int = 0
    has_vector_text: bool = False
    # --- padding metadata (exact) ---
    pad_left: int = 0                  # pixels padded on the left
    pad_top: int = 0                   # pixels padded on the top
    content_w: int = 0                 # unpadded content width (pixels)
    content_h: int = 0                 # unpadded content height (pixels)
    square_side: int = 0               # padded canvas side (pixels)

    @property
    def content_size(self) -> tuple[int, int]:
        """Unpadded raster size — use this for display scale factors."""
        return (self.content_w or self.image.size[0],
                self.content_h or self.image.size[1])

    def close(self):
        if self.pdf_doc is not None:
            try:
                self.pdf_doc.close()
            except Exception:
                pass
            self.pdf_doc = None


def _render_pdf_page(doc, page_num: int, dpi: int) -> Image.Image:
    page = doc[page_num]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _count_text_spans(doc, page_num: int) -> int:
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


def normalize_pdf(path: Path, page_num: int | None, vision_dpi: int,
                  vector_min_spans: int) -> NormalizedPage:
    doc = fitz.open(str(path))
    if page_num is None:
        page_num = len(doc) - 1
    if page_num < 0 or page_num >= len(doc):
        doc.close()
        raise ValueError(f"Page {page_num} out of range (0..{len(doc)-1})")
    page = doc[page_num]
    rect = page.rect
    spans = _count_text_spans(doc, page_num)
    raw = _render_pdf_page(doc, page_num, vision_dpi)
    square, pl, pt, cw, ch, sd = _pad_to_square(raw)
    return NormalizedPage(
        source_path=path,
        kind="pdf",
        page_num=page_num,
        page_count=len(doc),
        width_pt=float(rect.width),
        height_pt=float(rect.height),
        image=square,
        image_dpi=vision_dpi,
        pdf_doc=doc,
        vector_span_count=spans,
        has_vector_text=spans >= vector_min_spans,
        pad_left=pl, pad_top=pt,
        content_w=cw, content_h=ch, square_side=sd,
    )


def normalize_image(path: Path) -> NormalizedPage:
    img = Image.open(path)
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    W, H = img.size
    square, pl, pt, cw, ch, sd = _pad_to_square(img)
    return NormalizedPage(
        source_path=path,
        kind="image",
        page_num=0,
        page_count=1,
        width_pt=float(W),          # 1 pixel = 1 point for images
        height_pt=float(H),
        image=square,
        image_dpi=72,
        pdf_doc=None,
        vector_span_count=0,
        has_vector_text=False,
        pad_left=pl, pad_top=pt,
        content_w=cw, content_h=ch, square_side=sd,
    )


def normalize_input(path: Path, page_num: int | None = None,
                    vision_dpi: int = 150,
                    vector_min_spans: int = 10) -> NormalizedPage:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return normalize_pdf(path, page_num, vision_dpi, vector_min_spans)
    if suffix in IMAGE_EXTS:
        return normalize_image(path)
    raise ValueError(f"Unsupported input format: {suffix}")