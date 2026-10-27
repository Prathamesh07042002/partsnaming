"""extraction_pipeline.py - orchestrator (vector + vision + merge)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import Settings, load_settings
from input_normalizer import NormalizedPage, normalize_input
from candidate_filter import (
    filter_candidates, detect_table_regions, reject_dense_clusters,
)
from candidate_merger import merge_candidates

log = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    candidates: list[dict]
    page: NormalizedPage
    stats: dict


def _vector_candidates(page):
    if page.kind != "pdf" or page.pdf_doc is None:
        return []
    try:
        import pdf_extractor
        cands, stats, _ = pdf_extractor.extract_pdf_candidates(
            page.source_path, page.page_num,
        )
        log.info("Vector source: %d candidates (spans=%d, kept=%d)",
                 len(cands), stats.get("total", 0), stats.get("kept", 0))
        for c in cands:
            c.setdefault("source", "vector")
        return cands
    except Exception as exc:
        log.exception("Vector extraction failed: %s", exc)
        return []


def _vision_candidates(page, settings):
    from vision.factory import build_vision_client
    client = build_vision_client(settings)
    if client is None:
        log.info("Vision source: skipped (no provider / no API key)")
        return []
    try:
        cands = client.extract_candidates(
            image=page.image,
            page_width_pt=page.width_pt,
            page_height_pt=page.height_pt,
            page_num=page.page_num,
            context={
                "pad_left":    page.pad_left,
                "pad_top":     page.pad_top,
                "content_w":   page.content_w,
                "content_h":   page.content_h,
                "square_side": page.square_side,
            },
        )
        log.info("Vision source (%s): %d candidates",
                 client.provider_name, len(cands))
        return cands
    except Exception as exc:
        log.exception("Vision extraction failed: %s", exc)
        return []


def extract_for_page(input_path: Path, page_num=None,
                     settings: Settings | None = None) -> ExtractionResult:
    settings = settings or load_settings()
    log.info("Input: %s", input_path)
    log.info("Extraction mode: %s | provider: %s (%s)",
             settings.extraction_mode, settings.provider, settings.model_name)

    page = normalize_input(
        input_path, page_num=page_num,
        vision_dpi=settings.vision_dpi,
        vector_min_spans=settings.vector_min_spans,
    )
    log.info(
        "Normalized: kind=%s page=%d size=%.0fx%.0fpt vector_spans=%d "
        "has_vector=%s",
        page.kind, page.page_num, page.width_pt, page.height_pt,
        page.vector_span_count, page.has_vector_text,
    )
    log.info("Padding: content=%dx%d → square=%d (offset %d, %d)",
             page.content_w, page.content_h, page.square_side,
             page.pad_left, page.pad_top)
    # mode = settings.extraction_mode
    # if mode not in ("vector_first", "vision_first", "hybrid"):
    #     log.warning("Unknown EXTRACTION_MODE=%r — defaulting to hybrid", mode)
    #     mode = "hybrid"
    mode = settings.extraction_mode
    if mode not in ("vector_first", "vision_first", "hybrid", "vision_only"):
        log.warning("Unknown EXTRACTION_MODE=%r — defaulting to hybrid", mode)
        mode = "hybrid"

    vector_cands, vision_cands = [], []

    # if mode == "vector_first":
    #     vector_cands = _vector_candidates(page)
    #     if page.kind == "image" or not vector_cands:
    #         vision_cands = _vision_candidates(page, settings)
    # elif mode == "vision_first":
    #     vision_cands = _vision_candidates(page, settings)
    #     if page.kind == "pdf":
    #         vector_cands = _vector_candidates(page)
    # else:
    #     vector_cands = _vector_candidates(page)
    #     vision_cands = _vision_candidates(page, settings)
    # if mode == "vision_only":
    #     # Skip vector entirely — pure Gemini run for diagnosis.
    #     log.info("vision_only: vector source disabled")
    #     vision_cands = _vision_candidates(page, settings)
    # elif mode == "vector_first":
    #     vector_cands = _vector_candidates(page)
    #     if page.kind == "image" or not vector_cands:
    #         vision_cands = _vision_candidates(page, settings)
    # elif mode == "vision_first":
    #     vision_cands = _vision_candidates(page, settings)
    #     if page.kind == "pdf":
    #         vector_cands = _vector_candidates(page)
    # else:
    #     vector_cands = _vector_candidates(page)
    #     vision_cands = _vision_candidates(page, settings)
    if mode == "vision_only":
        # Skip vector entirely — pure Gemini run for diagnosis.
        log.info("vision_only: vector source disabled")
        vision_cands = _vision_candidates(page, settings)
    elif mode == "vector_first":
        vector_cands = _vector_candidates(page)
        if page.kind == "image" or not vector_cands:
            vision_cands = _vision_candidates(page, settings)
    elif mode == "vision_first":
        vision_cands = _vision_candidates(page, settings)
        if page.kind == "pdf":
            vector_cands = _vector_candidates(page)
    else:
        vector_cands = _vector_candidates(page)
        vision_cands = _vision_candidates(page, settings)
    # --- table-region detection on raw spans -----------------------------
    try:
        from pdf_extractor import extract_pdf_spans
        if page.kind == "pdf":
            spans, _ = extract_pdf_spans(page.source_path, page.page_num)
            table_regions = detect_table_regions(spans)
        else:
            table_regions = []
    except Exception as exc:
        log.warning("Table-region detection failed: %s", exc)
        table_regions = []

    # --- filter both sources ---------------------------------------------
    vector_cands, vstats = filter_candidates(
        vector_cands, page.width_pt, page.height_pt, table_regions,
    )
    vision_cands, gstats = filter_candidates(
        vision_cands, page.width_pt, page.height_pt, table_regions,
    )
    log.info(
        "Filter: vector kept=%d (bp=%d tbl=%d) | vision kept=%d "
        "(bp=%d tbl=%d) | table_regions=%d",
        vstats["kept"], vstats["boilerplate"], vstats["table_cell"],
        gstats["kept"], gstats["boilerplate"], gstats["table_cell"],
        len(table_regions),
    )

    # --- NEW: cluster rejection on combined stream -----------------------
    combined = vector_cands + vision_cands
    before = len(combined)
    combined = reject_dense_clusters(combined, radius_pt=30.0,
                                     min_neighbors=4, log=log)
    if len(combined) < before:
        log.info("Cluster rejection removed %d candidates", before - len(combined))
        # split back (preserve order)
        vector_cands = [c for c in combined if c.get("source") == "vector"]
        vision_cands = [c for c in combined if c.get("source") != "vector"]

    # --- merge ------------------------------------------------------------
    merged, mstats = merge_candidates(
        vector_cands, vision_cands,
        spatial_radius_pt=settings.merge_spatial_radius_pt,
        text_similarity=settings.merge_text_similarity,
    )
    log.info(
        "Merge: vector=%d vision=%d → total=%d "
        "(vision_added=%d, dupes_removed=%d)",
        mstats["vector_in"], mstats["vision_in"], mstats["merged_total"],
        mstats["vision_added"], mstats["duplicates_removed"],
    )

    stats = {
        "input": str(input_path),
        "kind": page.kind,
        "page": page.page_num,
        "mode": mode,
        "provider": settings.provider,
        "model": settings.model_name,
        "vector_in": mstats["vector_in"],
        "vision_in": mstats["vision_in"],
        "merged_total": mstats["merged_total"],
        "duplicates_removed": mstats["duplicates_removed"],
        "vision_enabled": settings.vision_enabled,
    }
    return ExtractionResult(candidates=merged, page=page, stats=stats)