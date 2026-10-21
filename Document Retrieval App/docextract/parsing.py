"""Turn raw model text into a schema-valid `Extraction`, no matter what the model emits.

Two outcomes are tracked separately so the evaluation can be honest about them:
  * `strict_valid`: the raw text was a JSON object that passed the schema as-is.
  * the returned `Extraction` is always valid (repairs + coercion + empty fallback),
    which is what guarantees a 100% JSON validity rate at the API.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .normalize import normalize_date, parse_number
from .schema import Extraction

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass
class ParseInfo:
    strict_valid: bool = False
    repaired: bool = False
    fallback_empty: bool = False
    errors: List[str] = field(default_factory=list)


def _close_truncated(s: str) -> Optional[Any]:
    """Recover a JSON object cut off by max_new_tokens: cut after the last complete
    object/array and append the missing closing brackets."""
    stack: List[str] = []
    in_str = esc = False
    cuts = []
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
    for cut, closers in reversed(cuts[-50:]):
        try:
            return json.loads(re.sub(r",\s*$", "", s[:cut]) + closers)
        except json.JSONDecodeError:
            continue
    return None


def _load_json(text: str, info: ParseInfo) -> Optional[Any]:
    s = _FENCE.sub("", text.strip())
    start = s.find("{")
    if start < 0:
        info.errors.append("no JSON object in output")
        return None
    end = s.rfind("}")
    candidate = s[start:end + 1] if end > start else s[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        info.errors.append(f"json: {e.msg}")
    info.repaired = True
    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)  # trailing commas
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return _close_truncated(s[start:])


def _coerce(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    store = obj.get("store_name")
    out["store_name"] = (str(store).strip() or None) if store is not None else None
    out["date"] = normalize_date(obj.get("date")) if obj.get("date") else None
    for k in ("subtotal", "tax", "total"):
        out[k] = parse_number(obj.get(k))
    items = []
    for it in obj.get("line_items") or []:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        if name is None or not str(name).strip():
            continue
        items.append({
            "name": str(name).strip(),
            "qty": parse_number(it.get("qty")),
            "unit_price": parse_number(it.get("unit_price")),
            "amount": parse_number(it.get("amount")),
        })
    out["line_items"] = items
    return out


def parse_model_output(text: str) -> "tuple[Extraction, ParseInfo]":
    info = ParseInfo()
    obj = _load_json(text, info)
    if not isinstance(obj, dict):
        info.fallback_empty = True
        info.errors.append("unparseable output, returning empty extraction")
        return Extraction(), info
    if not info.repaired:
        try:
            Extraction.model_validate(obj)
            # strict validity also requires no silent string->number coercion of numbers
            info.strict_valid = all(
                obj.get(k) is None or isinstance(obj.get(k), (int, float)) for k in ("subtotal", "tax", "total")
            )
        except ValidationError as e:
            info.errors.append(f"schema: {e.error_count()} errors")
    try:
        return Extraction.model_validate(_coerce(obj)), info
    except ValidationError as e:  # should not happen after coercion, but never fail the request
        info.errors.append(f"coerced schema: {e.error_count()} errors")
        info.fallback_empty = True
        return Extraction(), info
