"""Single-call Pixtral VLM diagnostic. Fails loudly with response dump."""
from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import traceback
from pathlib import Path

from config import load_settings


_POPPLER_CANDIDATES = [
    r"C:\Program Files\poppler-26.09.0\Library\bin",
    r"C:\Program Files\poppler\Library\bin",
    r"C:\poppler\Library\bin",
]


def _find_poppler_path():
    if shutil.which("pdftoppm"):
        return None
    for folder in _POPPLER_CANDIDATES:
        if Path(folder).exists():
            return folder
    return None


def render_page(pdf_path, page_idx, dpi=300):
    from pdf2image import convert_from_path
    images = convert_from_path(
        str(pdf_path), dpi=dpi,
        first_page=page_idx + 1, last_page=page_idx + 1,
        poppler_path=_find_poppler_path(),
    )
    if not images:
        raise RuntimeError(f"No page rendered for index {page_idx}")
    return images[0].convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("page", type=int, nargs="?", default=1)
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()

    pdf = Path(a.pdf).resolve()
    if not pdf.exists():
        print(f"File not found: {pdf}")
        return 1

    settings = load_settings()
    if not settings.api_key:
        print("MISTRAL_API_KEY missing in .env")
        return 1

    print(f"Provider : {settings.provider}")
    print(f"Model    : {settings.model_name}")
    print(f"Padding  : {settings.vision_padding}")
    print(f"File     : {pdf.name}")
    print(f"Page     : {a.page} (0-based)")
    print(f"DPI      : {a.dpi}")
    print()

    img = render_page(pdf, a.page, dpi=a.dpi)
    print(f"Rendered : {img.size[0]}x{img.size[1]}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/png;base64,{b64}"

    from vision.providers.mistral_client import (
        MistralVLMClient, _resolve_mistral_class,
    )

    try:
        client = MistralVLMClient(
            api_key=settings.api_key,
            model_name=settings.model_name,
            timeout_s=settings.vision_timeout_s,
            max_retries=0,
        )
    except ValueError as e:
        print(f"[SETUP ERROR] {e}")
        return 1

    print("Calling VLM (1 API credit) ...")
    raw = None
    resp_obj = None
    try:
        Mistral = _resolve_mistral_class()
        sdk = Mistral(api_key=settings.api_key)
        messages = client._build_messages(data_url, string_url=True)
        resp_obj = sdk.chat.complete(
            model=settings.model_name,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )
        raw = client._extract_content(resp_obj)
    except Exception as e:
        print(f"CALL FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    out = Path("mistral_test_out")
    out.mkdir(exist_ok=True)

    rawfile = out / f"{pdf.stem}_p{a.page}_vlm_raw.txt"
    rawfile.write_text(raw or "", encoding="utf-8")
    print(f"Raw content: {len(raw or '')} chars → {rawfile}")

    try:
        dump = (resp_obj.model_dump() if hasattr(resp_obj, "model_dump")
                else resp_obj.dict() if hasattr(resp_obj, "dict")
                else repr(resp_obj))
        dumpfile = out / f"{pdf.stem}_p{a.page}_vlm_resp.json"
        dumpfile.write_text(
            json.dumps(dump, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Full response → {dumpfile}")
    except Exception as e:
        print(f"(response dump failed: {e})")

    if not raw or not raw.strip():
        print()
        print("===== EMPTY CONTENT FROM API =====")
        try:
            c = resp_obj.choices[0]
            print(f"  finish_reason : {getattr(c, 'finish_reason', '?')}")
            print(f"  model         : {getattr(resp_obj, 'model', '?')}")
            print(f"  usage         : {getattr(resp_obj, 'usage', '?')}")
        except Exception:
            pass
        print()
        print("Likely causes:")
        print(f"  1. MODEL_NAME={settings.model_name!r} is text-only or")
        print("     does not have vision capability.")
        print("  2. This key has no access to that model.")
        return 1

    try:
        data = json.loads(raw)
    except Exception:
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not m:
            print("Content is not JSON. Head: %r" % raw[:300])
            return 1
        data = json.loads(m.group(0))

    cands = data.get("candidates", [])
    print(f"\nCandidates: {len(cands)}")
    texts = [c.get("text", "") for c in cands]
    joined = " ".join(texts)
    for probe in ("379", "409", "R15", "R10", "Ø8",
                  "2882", "TS23521", "0.553"):
        print(f"  contains {probe!r}: {probe in joined}")

    print()
    for i, c in enumerate(cands[:30]):
        print(f"  [{i:3d}] {c.get('type','?'):22s} "
              f"{c.get('text','')[:40]!r}  bbox={c.get('bbox_1000')}")
    if len(cands) > 30:
        print(f"  ... and {len(cands) - 30} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())