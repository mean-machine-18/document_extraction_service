"""Evaluation metrics (assignment section 04).

Field-level P/R/F1: a predicted non-null value that equals the gold value is a TP; a
non-null prediction that is wrong (or where gold is null) is a FP; a gold value that was
not predicted correctly is a FN. A wrong value is therefore both a FP and a FN.

Line items are aligned to gold items with the Hungarian algorithm on name similarity
before per-field comparison, so item order does not matter.

Hallucination: a non-null predicted value is "hallucinated" when it cannot be found in
the document. The document text used here is CORD's human-annotated words (an oracle
that is independent of the OCR used by the grounding step).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from .grounding import UNVERIFIED, ground_page
from .normalize import normalize_text, numbers_equal
from .pdf import TextLine
from .schema import Extraction

SCALARS = ("store_name", "date", "subtotal", "tax", "total")
ITEM_KEYS = ("name", "qty", "unit_price", "amount")
ALL_FIELDS = SCALARS + tuple(f"line_items.{k}" for k in ITEM_KEYS)
# Fields the CORD oracle text can confirm (store name / date regions are not annotated).
HALLUCINATION_FIELDS = ("subtotal", "tax", "total") + tuple(f"line_items.{k}" for k in ITEM_KEYS)
ITEM_MATCH_MIN_SIM = 50.0


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return numbers_equal(float(a), float(b))
    return normalize_text(str(a)) == normalize_text(str(b))


def align_items(pred: List[dict], gold: List[dict]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Pairs (pred_idx, gold_idx); unmatched items appear with None on one side."""
    if not pred or not gold:
        return [(i, None) for i in range(len(pred))] + [(None, j) for j in range(len(gold))]
    sim = np.array([[fuzz.ratio(normalize_text(p["name"]), normalize_text(g["name"])) for g in gold] for p in pred])
    # small bonus for equal amounts helps with repeated / abbreviated names
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if p.get("amount") is not None and values_equal(p.get("amount"), g.get("amount")):
                sim[i, j] += 20
    rows, cols = linear_sum_assignment(-sim)
    pairs, used_p, used_g = [], set(), set()
    for i, j in zip(rows, cols):
        if sim[i, j] >= ITEM_MATCH_MIN_SIM:
            pairs.append((i, j))
            used_p.add(i)
            used_g.add(j)
    pairs += [(i, None) for i in range(len(pred)) if i not in used_p]
    pairs += [(None, j) for j in range(len(gold)) if j not in used_g]
    return pairs


class Counts:
    def __init__(self):
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.exact = defaultdict(int)
        self.docs = 0

    def add(self, field: str, pred, gold):
        if pred is not None and values_equal(pred, gold):
            self.tp[field] += 1
            return
        if pred is not None:
            self.fp[field] += 1
        if gold is not None:
            self.fn[field] += 1

    def prf(self) -> Dict[str, Dict[str, Optional[float]]]:
        out = {}
        for f in ALL_FIELDS + ("micro",):
            if f == "micro":
                tp, fp, fn = (sum(d.values()) for d in (self.tp, self.fp, self.fn))
            else:
                tp, fp, fn = self.tp[f], self.fp[f], self.fn[f]
            p = tp / (tp + fp) if tp + fp else None
            r = tp / (tp + fn) if tp + fn else None
            f1 = 2 * p * r / (p + r) if tp else (0.0 if fp + fn else None)
            out[f] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        return out

    def exact_match(self) -> Dict[str, Optional[float]]:
        """Share of documents where the field (or whole item list / whole document) is exactly right."""
        keys = SCALARS + ("line_items", "document")
        return {k: (self.exact[k] / self.docs if self.docs else None) for k in keys}


def score_document(c: Counts, pred: Extraction, gold: dict) -> None:
    c.docs += 1
    doc_ok = True
    for f in SCALARS:
        p, g = getattr(pred, f), gold.get(f)
        c.add(f, p, g)
        ok = values_equal(p, g)
        c.exact[f] += ok
        doc_ok &= ok
    pitems = [i.model_dump() for i in pred.line_items]
    gitems = gold.get("line_items", [])
    items_ok = len(pitems) == len(gitems)
    for pi, gi in align_items(pitems, gitems):
        p = pitems[pi] if pi is not None else {}
        g = gitems[gi] if gi is not None else {}
        for k in ITEM_KEYS:
            c.add(f"line_items.{k}", p.get(k), g.get(k))
            items_ok &= values_equal(p.get(k), g.get(k))
    c.exact["line_items"] += items_ok
    doc_ok &= items_ok
    c.exact["document"] += doc_ok


def hallucinations(pred: Extraction, oracle: List[TextLine]) -> Tuple[int, int, Dict[str, List[int]]]:
    """(# non-null values checked, # not found in the document, per-field [checked, missing])."""
    g = ground_page(pred, oracle, policy="flag")
    per = defaultdict(lambda: [0, 0])
    for f in ("subtotal", "tax", "total"):
        if getattr(pred, f) is not None:
            per[f][0] += 1
            per[f][1] += g.verification[f] == UNVERIFIED
    for iv in g.item_verification:
        for k in ITEM_KEYS:
            if iv[k] != "not_found":
                per[f"line_items.{k}"][0] += 1
                per[f"line_items.{k}"][1] += iv[k] == UNVERIFIED
    checked = sum(v[0] for v in per.values())
    missing = sum(v[1] for v in per.values())
    return checked, missing, dict(per)


def grounding_effect(raw: Extraction, final: Extraction, gold: dict) -> Dict[str, int]:
    """How many values grounding removed, split by whether they were actually correct."""
    out = {"removed_wrong": 0, "removed_correct": 0}
    for f in ("store_name", "date", "subtotal", "tax", "total"):
        r, fv = getattr(raw, f), getattr(final, f)
        if r is not None and fv is None:
            out["removed_correct" if values_equal(r, gold.get(f)) else "removed_wrong"] += 1
    raw_items = [i.model_dump() for i in raw.line_items]
    final_names = [i.name for i in final.line_items]
    gold_items = gold.get("line_items", [])
    for pi, gi in align_items(raw_items, gold_items):
        if pi is None:
            continue
        r = raw_items[pi]
        if r["name"] not in final_names:
            correct = gi is not None and values_equal(r["name"], gold_items[gi]["name"])
            out["removed_correct" if correct else "removed_wrong"] += 1
    return out


def percentile(xs: Iterable[float], q: float) -> Optional[float]:
    xs = list(xs)
    return float(np.percentile(xs, q)) if xs else None
