"""PDF validation and page rendering.

All bad-input handling for the PDF itself lives here and raises `InputError`, which the
API maps to a JSON error response with the right status code.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

MAX_RENDER_SIDE = 3000  # px; protects memory against poster-sized pages


class InputError(Exception):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class TextLine:
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1 in page-image pixels


@dataclass
class Page:
    index: int
    image: Image.Image
    text_layer: List[TextLine] = field(default_factory=list)

    @property
    def has_text_layer(self) -> bool:
        return sum(len(t.text) for t in self.text_layer) >= 20


def open_pdf(data: bytes, max_pages: int) -> fitz.Document:
    if not data:
        raise InputError("empty_file", "Uploaded file is empty.", 400)
    if b"%PDF" not in data[:1024]:
        raise InputError("not_a_pdf", "File is not a PDF (missing %PDF header).", 415)
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:  # PyMuPDF raises several types for broken files
        raise InputError("corrupt_pdf", f"PDF could not be parsed: {e}") from e
    if doc.needs_pass:
        doc.close()
        raise InputError("encrypted_pdf", "PDF is password-protected; upload an unlocked copy.")
    if doc.page_count == 0:
        doc.close()
        raise InputError("corrupt_pdf", "PDF has no pages.")
    if doc.page_count > max_pages:
        n = doc.page_count
        doc.close()
        raise InputError("too_many_pages", f"PDF has {n} pages; the limit is {max_pages}.", 413)
    return doc


def _text_layer(page: fitz.Page, zoom: float) -> List[TextLine]:
    lines = {}
    for x0, y0, x1, y1, word, block, line, _ in page.get_text("words"):
        key = (block, line)
        if key in lines:
            t, (a, b, c, d) = lines[key]
            lines[key] = (f"{t} {word}", (min(a, x0), min(b, y0), max(c, x1), max(d, y1)))
        else:
            lines[key] = (word, (x0, y0, x1, y1))
    return [TextLine(t, tuple(v * zoom for v in bb)) for t, bb in lines.values()]


def render_pages(doc: fitz.Document, dpi: int) -> List[Page]:
    pages: List[Page] = []
    for i, page in enumerate(doc):
        zoom = dpi / 72.0
        longest = max(page.rect.width, page.rect.height) * zoom
        if longest > MAX_RENDER_SIDE:
            zoom *= MAX_RENDER_SIDE / longest
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        except Exception as e:
            raise InputError("corrupt_pdf", f"Page {i + 1} could not be rendered: {e}") from e
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(Page(index=i, image=img, text_layer=_text_layer(page, zoom)))
    return pages


def load_pdf(data: bytes, max_pages: int, dpi: int) -> List[Page]:
    doc = open_pdf(data, max_pages)
    try:
        return render_pages(doc, dpi)
    finally:
        doc.close()


def images_to_pdf(images: List[Image.Image], dpi: Optional[int] = 200) -> bytes:
    """Build an image-only PDF (no text layer) - used to turn CORD images into test PDFs."""
    doc = fitz.open()
    for img in images:
        w, h = img.size
        page = doc.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)
        page.insert_image(page.rect, stream=buf.getvalue())
    out = doc.tobytes()
    doc.close()
    return out
