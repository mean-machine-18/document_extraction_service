"""CORD v2 ground-truth conversion into the assignment schema.

CORD's `gt_parse` labels menu items, sub-totals and totals. It does NOT label the store
name or the date (that information was excluded from the public release), so for CORD
those two fields are always null in the targets. See README "Dataset".
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .normalize import parse_number
from .pdf import TextLine

CORD_DATASET = "naver-clova-ix/cord-v2"


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _text(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, list):
        v = " ".join(str(x) for x in v if x)
    v = str(v).strip()
    return v or None


def _num(v: Any) -> Optional[float]:
    return parse_number(_text(v) if isinstance(v, list) else v)


def _item(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = _text(d.get("nm"))
    if not name:
        return None
    return {
        "name": name,
        "qty": _num(d.get("cnt")),
        "unit_price": _num(d.get("unitprice")),
        "amount": _num(d.get("price")),
    }


def gt_to_target(gt_parse: Dict[str, Any]) -> Dict[str, Any]:
    """Map one CORD `gt_parse` dict to the extraction schema (plain dict)."""
    items: List[Dict[str, Any]] = []
    for m in _as_list(gt_parse.get("menu")):
        if not isinstance(m, dict):
            continue
        it = _item(m)
        if it:
            items.append(it)
        # Modifiers ("+ coconut jelly 4,000") are printed rows that count towards the
        # subtotal, so they are kept as their own line items.
        for sub in _as_list(m.get("sub")):
            if isinstance(sub, dict):
                s = _item(sub)
                if s:
                    items.append(s)
    sub_total = gt_parse.get("sub_total") or {}
    total = gt_parse.get("total") or {}
    if isinstance(sub_total, list):
        sub_total = sub_total[0] if sub_total else {}
    if isinstance(total, list):
        total = total[0] if total else {}
    return {
        "store_name": None,
        "date": None,
        "line_items": items,
        "subtotal": _num(sub_total.get("subtotal_price")),
        "tax": _num(sub_total.get("tax_price")),
        "total": _num(total.get("total_price")),
    }


def _clean_numbers(obj: Any) -> Any:
    """29090.0 -> 29090 so targets look like what is printed on the receipt."""
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    if isinstance(obj, dict):
        return {k: _clean_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_numbers(v) for v in obj]
    return obj


def target_to_json(target: Dict[str, Any]) -> str:
    return json.dumps(_clean_numbers(target), ensure_ascii=False, separators=(",", ":"))


def parse_ground_truth(gt_json: str) -> Dict[str, Any]:
    return json.loads(gt_json)


def document_lines(gt: Dict[str, Any]) -> List[TextLine]:
    """Human-annotated text rows of a receipt with their boxes (an OCR-independent oracle)."""
    rows: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for line in gt.get("valid_line", []):
        for w in line.get("words", []):
            rows.setdefault((line.get("group_id", 0), w.get("row_id", 0)), []).append(w)
    out = []
    for words in rows.values():
        words.sort(key=lambda w: w["quad"]["x1"])
        xs = [w["quad"][k] for w in words for k in ("x1", "x2", "x3", "x4")]
        ys = [w["quad"][k] for w in words for k in ("y1", "y2", "y3", "y4")]
        out.append(TextLine(" ".join(w["text"] for w in words), (min(xs), min(ys), max(xs), max(ys))))
    return out


def summary_crop_y(gt: Dict[str, Any]) -> Optional[int]:
    """y coordinate that separates the menu block from the subtotal/total block.

    Used for "abstention" augmentation: cropping the image above this line removes all
    totals, and the target for subtotal/tax/total becomes null. Returns None when the
    two blocks overlap vertically (then the sample is not cropped).
    """
    menu_bottom, summary_top = -1, None
    for line in gt.get("valid_line", []):
        cat = line.get("category", "")
        ys = [min(w["quad"]["y1"], w["quad"]["y2"]) for w in line.get("words", [])]
        ye = [max(w["quad"]["y3"], w["quad"]["y4"]) for w in line.get("words", [])]
        if not ys:
            continue
        if cat.startswith("menu"):
            menu_bottom = max(menu_bottom, max(ye))
        elif cat.startswith(("sub_total", "total")):
            summary_top = min(ys) if summary_top is None else min(summary_top, min(ys))
    if summary_top is None or menu_bottom < 0 or summary_top - menu_bottom < 4:
        return None
    return (menu_bottom + summary_top) // 2
