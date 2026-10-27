"""inspect_pdf.py — non-destructive PyMuPDF diagnostic. No classification, no bubbling."""
import json
import sys
from pathlib import Path
import pymupdf as fitz


def dump(pdf_path, page_num=None, out_dir=None, dpi=150):
    doc = fitz.open(str(pdf_path))
    if page_num is None:
        page_num = len(doc) - 1
    page = doc[page_num]
    rect = page.rect

    spans = []
    raw_text = page.get_text("dict")
    for block in raw_text.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            direction = list(line.get("dir", [1.0, 0.0]))
            for span in line.get("spans", []):
                spans.append({
                    "text": span.get("text", ""),
                    "bbox": list(span.get("bbox", [0, 0, 0, 0])),
                    "size": float(span.get("size", 0)),
                    "font": span.get("font", ""),
                    "direction": direction,
                    "flags": int(span.get("flags", 0)),
                })

    drawings = page.get_drawings()
    prims = []
    for d in drawings:
        for it in d.get("items", []):
            op = it[0]
            if op == "l":
                p1, p2 = it[1], it[2]
                prims.append({"op": "l",
                              "p1": [round(p1.x, 1), round(p1.y, 1)],
                              "p2": [round(p2.x, 1), round(p2.y, 1)],
                              "stroke": d.get("stroke"),
                              "width": d.get("width")})
            elif op == "re":
                r = it[1]
                prims.append({"op": "re",
                              "rect": [round(r.x0, 1), round(r.y0, 1),
                                       round(r.x1, 1), round(r.y1, 1)],
                              "stroke": d.get("stroke"),
                              "fill": d.get("fill"),
                              "width": d.get("width")})
            elif op == "qu":
                q = it[1]
                prims.append({"op": "qu",
                              "pts": [[round(q.ul.x, 1), round(q.ul.y, 1)],
                                      [round(q.ur.x, 1), round(q.ur.y, 1)],
                                      [round(q.lr.x, 1), round(q.lr.y, 1)],
                                      [round(q.ll.x, 1), round(q.ll.y, 1)]],
                              "stroke": d.get("stroke"),
                              "fill": d.get("fill")})
            elif op == "c":
                pts = it[1:5]
                prims.append({"op": "c",
                              "pts": [[round(p.x, 1), round(p.y, 1)] for p in pts],
                              "stroke": d.get("stroke"),
                              "width": d.get("width")})

    # --- summary stats ---
    def count_by_op(p):
        c = {"l": 0, "re": 0, "qu": 0, "c": 0}
        for x in p:
            c[x["op"]] = c.get(x["op"], 0) + 1
        return c

    summary = {
        "pdf": str(pdf_path),
        "page": page_num,
        "page_size_pt": [round(rect.width, 1), round(rect.height, 1)],
        "span_count": len(spans),
        "drawing_path_count": len(drawings),
        "drawing_prim_count": len(prims),
        "prim_by_op": count_by_op(prims),
    }

    # --- render page + overlays ---
    from PIL import Image, ImageDraw, ImageFont
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()

    # drawings in green
    for x in prims:
        if x["op"] == "l":
            p1, p2 = x["p1"], x["p2"]
            draw.line([p1[0] * scale, p1[1] * scale,
                       p2[0] * scale, p2[1] * scale], fill=(0, 180, 0), width=1)
        elif x["op"] == "re":
            r = x["rect"]
            draw.rectangle([r[0] * scale, r[1] * scale,
                            r[2] * scale, r[3] * scale],
                           outline=(0, 180, 0), width=1)
        elif x["op"] == "qu":
            pts = [(p[0] * scale, p[1] * scale) for p in x["pts"]]
            draw.polygon(pts, outline=(0, 180, 0))

    # text boxes in red with index
    for i, s in enumerate(spans, 1):
        x1, y1, x2, y2 = [v * scale for v in s["bbox"]]
        draw.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=1)
        draw.text((x1, max(0, y1 - 12)), f"{i}", fill=(220, 0, 0), font=font)

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "native_debug.json").write_text(
            json.dumps({"summary": summary, "spans": spans}, indent=2),
            encoding="utf-8")
        (out_dir / "vector_debug.json").write_text(
            json.dumps({"summary": summary, "primitives": prims}, indent=2),
            encoding="utf-8")
        img.save(out_dir / "overlay_debug.png")

    return summary, out_dir


if __name__ == "__main__":
    pdf = Path(sys.argv[1])
    page = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out = Path("inspect_out") / pdf.stem
    summary, _ = dump(pdf, page, out)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote to {out}/")
    print(f"  native_debug.json   (all text spans)")
    print(f"  vector_debug.json   (all drawing primitives)")
    print(f"  overlay_debug.png   (page + text in red + vectors in green)")