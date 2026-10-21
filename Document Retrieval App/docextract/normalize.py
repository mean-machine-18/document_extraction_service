"""Value normalisation shared by ground-truth conversion, grounding and metrics.

Receipts in CORD are Indonesian: "60.000" and "60,000" both mean sixty thousand, while
"1.00" is a quantity of one. These helpers turn raw strings into comparable values.
"""
from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Set

from dateutil import parser as date_parser

_NUM_TOKEN = re.compile(r"[-+(]?\s*(?:\d[\d.,]*|[.,]\d+)")
_DIGIT_GROUPS = re.compile(r"\d+")


def _interpret(token: str) -> List[float]:
    """All plausible numeric readings of a token like '1.500', '45,000.00' or '12,5'."""
    neg = token.lstrip().startswith(("-", "("))
    body = re.sub(r"[^\d.,]", "", token)
    if re.fullmatch(r"[.,]\d{1,2}", body):  # ".5" -> 0.5
        return [float("0." + body[1:])]
    body = body.strip(".,")
    if not body:
        return []
    seps = [c for c in body if c in ".,"]
    readings: List[float] = []
    if not seps:
        readings.append(float(body))
    elif len(set(seps)) == 2:
        # Both separators present: the last one is the decimal mark.
        dec = body.rfind(".") if body.rfind(".") > body.rfind(",") else body.rfind(",")
        intpart = re.sub(r"[.,]", "", body[:dec])
        readings.append(float(f"{intpart or 0}.{body[dec + 1:] or 0}"))
    else:
        sep = seps[0]
        groups = body.split(sep)
        thousands_ok = all(len(g) == 3 for g in groups[1:]) and len(groups[0]) <= 3 and groups[0] != ""
        if thousands_ok:
            readings.append(float("".join(groups)))
        if len(groups) == 2:  # single separator can also be a decimal mark
            readings.append(float(f"{groups[0] or 0}.{groups[1]}"))
        if not readings:
            readings.append(float("".join(groups)))
    return [-r if neg else r for r in readings]


def parse_number(value) -> Optional[float]:
    """Best single numeric reading of a raw value; None if it has no digits."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        value = value[0] if value else None
        return parse_number(value)
    s = str(value)
    m = _NUM_TOKEN.search(s)
    if not m:
        return None
    readings = _interpret(m.group(0))
    if not readings:
        return None
    # "-Rp 34,363": the sign can precede the currency symbol
    if readings[0] > 0 and re.match(r"^\s*[-(]", s[:m.start()]):
        return -readings[0]
    return readings[0]


def number_candidates(text: str) -> Set[float]:
    """Every numeric reading of every number-like token in `text` (absolute values)."""
    out: Set[float] = set()
    for m in _NUM_TOKEN.finditer(text):
        for r in _interpret(m.group(0)):
            out.add(round(abs(r), 2))
    # OCR often splits "60 000" or drops a separator; also try digit runs joined pairwise.
    groups = _DIGIT_GROUPS.findall(text)
    for a, b in zip(groups, groups[1:]):
        if len(b) == 3:
            out.add(float(a + b))
    return out


def numbers_equal(a: Optional[float], b: Optional[float], rel: float = 0.0) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(0.01, rel * max(abs(a), abs(b)))


def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r"[^0-9A-Z]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_DATE_PATTERNS = [
    re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b"),  # 2019-03-21
    re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b"),  # 21/03/2019, 21-03-19
    re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{2,4})\b"),  # 21 Mar 2019
    re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{2,4})\b"),  # Mar 21, 2019
]


def _valid(d: date) -> bool:
    return 1990 <= d.year <= 2100


def normalize_date(value: Optional[str]) -> Optional[str]:
    """Parse a printed date into ISO YYYY-MM-DD (day-first, as on most non-US receipts)."""
    if not value:
        return None
    s = str(value).strip()
    m = _DATE_PATTERNS[0].search(s)
    try:
        if m:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        else:
            d = date_parser.parse(s, dayfirst=True, fuzzy=True).date()
    except (ValueError, OverflowError, TypeError, AttributeError):
        return None
    return d.isoformat() if _valid(d) else None


def find_dates(text: str) -> Set[str]:
    """ISO dates of every date-looking substring in OCR text."""
    out: Set[str] = set()
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            iso = normalize_date(m.group(0))
            if iso:
                out.add(iso)
            # Also accept the month-first reading for ambiguous numeric dates.
            if pat is _DATE_PATTERNS[1]:
                try:
                    yr = int(m.group(3))
                    yr = yr + 2000 if yr < 100 else yr
                    d = date(yr, int(m.group(1)), int(m.group(2)))
                    if _valid(d):
                        out.add(d.isoformat())
                except ValueError:
                    pass
    return out
