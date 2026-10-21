"""Grounding: check every extracted value against text that is really on the page.

Evidence comes from the PDF text layer or, for scanned/image PDFs, from OCR (see ocr.py).
The VLM never sees this text, so the check is independent of the model's own reading.

Rules (per page):
  * store_name / item names: fuzzy match (rapidfuzz) against an evidence line.
  * date: the ISO date must equal some date-looking string in the evidence.
  * numbers: some number token in the evidence must parse to the same value
    ("45.000", "45,000" and "Rp45000" all match 45000).
  * Small numbers (<= 2 digits, e.g. qty=1) are too common to verify page-wide, so they
    only count if they appear on the item's own row band. If OCR missed them there, the
    row is cropped beside the item name and re-read (ocr.CropReader).
Policy "null" (default) replaces every unverified value with null and drops line items
whose name cannot be found. Policy "flag" keeps the values and only marks them.

`sanity_checks` adds arithmetic checks on the merged result; they are reported in
`_checks` and do not change values (see README for why).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

from .normalize import find_dates, normalize_text, number_candidates
from .pdf import TextLine
from .schema import Extraction, CheckResult

VERIFIED, UNVERIFIED, NOT_FOUND = "verified", "unverified", "not_found"


@dataclass
class _Line:
    norm: str
    tokens: Set[str]
    numbers: Set[float]
    bbox: Tuple[float, float, float, float]


class Evidence:
    def __init__(self, lines: List[TextLine]):
        self.lines = [
            _Line(normalize_text(l.text), set(normalize_text(l.text).split()), number_candidates(l.text), l.bbox)
            for l in lines
        ]
        full = "\n".join(l.text for l in lines)
        self.numbers: Set[float] = set().union(*(l.numbers for l in self.lines)) if self.lines else set()
        self.dates = find_dates(full)

    def has_number(self, v: float) -> bool:
        return _num_in(v, self.numbers)

    def best_text_match(self, text: str) -> Tuple[float, Optional[_Line]]:
        q = normalize_text(text)
        if not q:
            return 0.0, None
        best, best_line = 0.0, None
        for line in self.lines:
            if len(q) < 4:  # short names: require an exact token, partial ratio is meaningless
                score = 100.0 if q in line.tokens or q == line.norm else 0.0
            else:
                score = fuzz.partial_ratio(q, line.norm) if len(line.norm) >= len(q) * 0.6 else fuzz.ratio(q, line.norm)
            if score > best:
                best, best_line = score, line
        return best, best_line

    def row_band_numbers(self, anchor: _Line) -> Set[float]:
        """Numbers on the anchor's row and the row below (unit price x qty lines)."""
        x0, y0, x1, y1 = anchor.bbox
        h = max(y1 - y0, 1.0)
        top, bottom = y0 - 0.5 * h, y1 + 1.6 * h
        out: Set[float] = set()
        for line in self.lines:
            cy = (line.bbox[1] + line.bbox[3]) / 2
            if top <= cy <= bottom:
                out |= line.numbers
        return out


def _num_in(v: float, pool: Set[float]) -> bool:
    return round(abs(v), 2) in pool


def _is_small(v: float) -> bool:
    return abs(v) < 100 and float(v).is_integer()


@dataclass
class PageGrounding:
    data: Extraction
    verification: Dict[str, str] = field(default_factory=dict)
    item_verification: List[Dict[str, str]] = field(default_factory=list)
    dropped_items: List[str] = field(default_factory=list)
    evidence_source: str = "ocr"


def ground_page(ext: Extraction, lines: List[TextLine], policy: str = "null",
                text_threshold: float = 85.0, crop_reader=None) -> PageGrounding:
    ev = Evidence(lines)
    keep_unverified = policy == "flag"
    out = ext.model_copy(deep=True)
    ver: Dict[str, str] = {}

    def settle(field_name: str, value, ok: bool):
        if value is None:
            ver[field_name] = NOT_FOUND
            return None
        ver[field_name] = VERIFIED if ok else UNVERIFIED
        return value if ok or keep_unverified else None

    if ext.store_name is not None:
        score, _ = ev.best_text_match(ext.store_name)
        out.store_name = settle("store_name", ext.store_name, score >= text_threshold)
    else:
        ver["store_name"] = NOT_FOUND
    out.date = settle("date", ext.date, ext.date in ev.dates if ext.date else False)
    for k in ("subtotal", "tax", "total"):
        v = getattr(ext, k)
        setattr(out, k, settle(k, v, v is not None and ev.has_number(v)))

    items, item_ver, dropped = [], [], []
    for it in ext.line_items:
        score, anchor = ev.best_text_match(it.name)
        name_ok = score >= text_threshold
        if not name_ok and not keep_unverified:
            dropped.append(it.name)
            continue
        band = ev.row_band_numbers(anchor) if (anchor is not None and name_ok) else set()
        new = it.model_copy()
        iv = {"name": VERIFIED if name_ok else UNVERIFIED}
        for k in ("qty", "unit_price", "amount"):
            v = getattr(it, k)
            if v is None:
                iv[k] = NOT_FOUND
                continue
            ok = _num_in(v, band) or (not _is_small(v) and ev.has_number(v))
            if not ok and crop_reader is not None and name_ok:
                ok = _num_in(v, crop_reader.row_numbers(anchor.bbox))
            iv[k] = VERIFIED if ok else UNVERIFIED
            if not ok and not keep_unverified:
                setattr(new, k, None)
        items.append(new)
        item_ver.append(iv)
    out.line_items = items
    return PageGrounding(out, ver, item_ver, dropped)


# --- arithmetic sanity checks -------------------------------------------------------------

def _tol(v: float) -> float:
    return max(1.0, 0.01 * abs(v))


def sanity_checks(ext: Extraction) -> List[CheckResult]:
    checks: List[CheckResult] = []

    full = [it for it in ext.line_items if None not in (it.qty, it.unit_price, it.amount)]
    if full:
        bad = [it.name for it in full if abs(it.qty * it.unit_price - it.amount) > _tol(it.amount)]
        checks.append(CheckResult(
            name="qty_x_unit_price_equals_amount",
            status="fail" if bad else "pass",
            detail=f"{len(full) - len(bad)}/{len(full)} items consistent" + (f"; mismatched: {bad[:5]}" if bad else ""),
        ))
    else:
        checks.append(CheckResult(name="qty_x_unit_price_equals_amount", status="skipped",
                                  detail="no item has qty, unit_price and amount"))

    amounts = [it.amount for it in ext.line_items if it.amount is not None]
    target_name, target = ("subtotal", ext.subtotal) if ext.subtotal is not None else ("total", ext.total)
    if amounts and target is not None and len(amounts) == len(ext.line_items):
        s = sum(amounts)
        ok = abs(s - target) <= _tol(target)
        checks.append(CheckResult(name="line_items_sum_to_subtotal", status="pass" if ok else "fail",
                                  detail=f"sum(amount)={s:g} vs {target_name}={target:g}"))
    else:
        checks.append(CheckResult(name="line_items_sum_to_subtotal", status="skipped",
                                  detail="missing item amounts or subtotal/total"))

    if ext.subtotal is not None and ext.total is not None:
        tax = ext.tax or 0.0
        if abs(ext.subtotal + tax - ext.total) <= _tol(ext.total):
            checks.append(CheckResult(name="subtotal_plus_tax_equals_total", status="pass",
                                      detail=f"{ext.subtotal:g} + {tax:g} = {ext.total:g}"))
        elif abs(ext.subtotal - ext.total) <= _tol(ext.total):
            checks.append(CheckResult(name="subtotal_plus_tax_equals_total", status="pass",
                                      detail="subtotal equals total (tax-inclusive pricing)"))
        else:
            checks.append(CheckResult(name="subtotal_plus_tax_equals_total", status="fail",
                                      detail=f"{ext.subtotal:g} + {tax:g} != {ext.total:g} "
                                             "(service charge / discount / rounding not in schema?)"))
    else:
        checks.append(CheckResult(name="subtotal_plus_tax_equals_total", status="skipped",
                                  detail="subtotal or total missing"))
    return checks
