"""VisionModelClient - provider-independent vision model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class VisionModelClient(ABC):
    """Provider-independent vision extraction interface.

    Implementations MUST return a list of candidate dicts already in the
    unified candidate schema (page-point coordinates):

        {
            "text": str, "type": str, "parameter": str,
            "specification": str, "tolerance": str,
            "bbox": (x1, y1, x2, y2),
            "anchor_x": float, "anchor_y": float,
            "source": "<provider>", "confidence": float,
        }

    Implementations MUST NOT raise on normal failures — they return [].
    """

    provider_name: str = "unknown"

    @abstractmethod
    def extract_candidates(
        self,
        image: Image.Image,
        page_width_pt: float,
        page_height_pt: float,
        page_num: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[dict]:
        ...