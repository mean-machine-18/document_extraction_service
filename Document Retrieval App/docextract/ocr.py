"""Lightweight OCR used only for grounding (never for extraction).

RapidOCR runs PaddleOCR's detection + recognition models through ONNX Runtime on CPU:
no GPU contention with the VLM, ~15 MB of weights, no system packages needed.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from .normalize import number_candidates
from .pdf import Page, TextLine

# In a crop that should hold a number, these glyphs are almost always misread digits.
_DIGIT_CONFUSABLES = str.maketrans("lIi|!oO", "1111100")

_engine = None
_lock = threading.Lock()


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
    return _engine


def ocr_image(image: Image.Image) -> List[TextLine]:
    result, _ = _get_engine()(np.asarray(image.convert("RGB")))
    lines: List[TextLine] = []
    for box, text, _score in result or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append(TextLine(text=text, bbox=(min(xs), min(ys), max(xs), max(ys))))
    return lines


class CropReader:
    """Second-chance check on bounding-box crops.

    Page-level detection misses tiny isolated glyphs (a lone "1" in a qty column). For an
    item whose name was found, this re-reads the strips left and right of the name box
    with the recogniser only and returns the numbers it sees there.
    """

    def __init__(self, image: Image.Image):
        self.image = image.convert("RGB")
        self._cache: Dict[Tuple[float, ...], Set[float]] = {}

    def row_numbers(self, bbox: Tuple[float, float, float, float]) -> Set[float]:
        if bbox in self._cache:
            return self._cache[bbox]
        x0, y0, x1, y1 = bbox
        W, H = self.image.size
        h = max(y1 - y0, 1.0)
        top, bottom = max(0.0, y0 - 0.3 * h), min(float(H), y1 + 0.3 * h)
        boxes = []
        if x0 > h:
            boxes.append((max(0.0, x0 - 6 * h), top, x0, bottom))
        if W - x1 > h:
            boxes.append((x1, top, float(W), bottom))
        nums: Set[float] = set()
        for b in boxes:
            crop = np.asarray(self.image.crop(tuple(int(v) for v in b)))
            result, _ = _get_engine()(crop, use_det=False, use_cls=False, use_rec=True)
            text = " ".join(r[0] for r in result or []).translate(_DIGIT_CONFUSABLES)
            nums |= number_candidates(text)
        self._cache[bbox] = nums
        return nums


@dataclass
class PageEvidence:
    lines: List[TextLine]
    source: str  # "text_layer" | "ocr"
    reader: Optional[CropReader] = None


def page_evidence(page: Page) -> PageEvidence:
    """Text to ground against: the PDF text layer if present, otherwise OCR (+ crop re-reads)."""
    if page.has_text_layer:
        return PageEvidence(page.text_layer, "text_layer")
    return PageEvidence(ocr_image(page.image), "ocr", CropReader(page.image))


def warmup() -> None:
    ocr_image(Image.new("RGB", (64, 32), "white"))
