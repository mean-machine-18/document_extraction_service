"""Page images -> grounded, merged, schema-valid response.

Split into a GPU part (`run_model`, must be called while holding the GPU slot) and a CPU
part (`ground_and_merge`), so the API can do OCR and post-processing outside the slot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import Settings
from .grounding import NOT_FOUND, UNVERIFIED, VERIFIED, PageGrounding, ground_page, sanity_checks
from .model import Backend, Generation
from .parsing import ParseInfo, parse_model_output
from .ocr import PageEvidence
from .pdf import Page
from .schema import CheckResult, Extraction, ExtractionResponse


class InferenceTimeout(Exception):
    pass


@dataclass
class PageOutput:
    generation: Generation
    extraction: Extraction
    parse: ParseInfo


def run_model(backend: Backend, pages: List[Page], deadline: Optional[float]) -> List[PageOutput]:
    outputs = []
    for page in pages:
        gen = backend.generate(page.image, deadline=deadline)
        if gen.timed_out:
            raise InferenceTimeout(f"inference exceeded the job timeout on page {page.index + 1}")
        ext, info = parse_model_output(gen.text)
        outputs.append(PageOutput(gen, ext, info))
    return outputs


def _merge(groundings: List[PageGrounding]) -> "tuple[Extraction, Dict[str, str], List[str]]":
    merged = Extraction()
    ver: Dict[str, str] = {}

    def pick(field: str, prefer_last: bool):
        order = list(reversed(groundings)) if prefer_last else groundings
        for g in order:
            if getattr(g.data, field) is not None:
                setattr(merged, field, getattr(g.data, field))
                ver[field] = g.verification[field]
                return
        statuses = {g.verification.get(field, NOT_FOUND) for g in groundings}
        ver[field] = UNVERIFIED if UNVERIFIED in statuses else NOT_FOUND

    # Header fields come from the first page that has them, totals from the last page.
    pick("store_name", prefer_last=False)
    pick("date", prefer_last=False)
    for f in ("subtotal", "tax", "total"):
        pick(f, prefer_last=True)

    dropped: List[str] = []
    item_statuses = []
    for g in groundings:
        for item, iv in zip(g.data.line_items, g.item_verification):
            idx = len(merged.line_items)
            merged.line_items.append(item)
            for k, status in iv.items():
                ver[f"line_items[{idx}].{k}"] = status
            item_statuses.extend(iv.values())
        dropped.extend(g.dropped_items)
    if not merged.line_items and not dropped:
        ver["line_items"] = NOT_FOUND
    elif dropped or UNVERIFIED in item_statuses:
        ver["line_items"] = UNVERIFIED
    else:
        ver["line_items"] = VERIFIED
    return merged, ver, dropped


def ground_and_merge(outputs: List[PageOutput], evidence: List[PageEvidence], settings: Settings,
                     meta: Optional[dict] = None) -> ExtractionResponse:
    groundings = [
        ground_page(o.extraction, ev.lines, policy=settings.grounding_policy,
                    text_threshold=settings.text_match_threshold, crop_reader=ev.reader)
        for o, ev in zip(outputs, evidence)
    ]
    merged, ver, dropped = _merge(groundings)
    checks = sanity_checks(merged)
    if dropped:
        checks.append(CheckResult(name="unverified_line_items_removed", status="fail",
                                  detail=f"{len(dropped)} item(s) not found on the page: {dropped[:5]}"))
    if any(o.parse.fallback_empty for o in outputs):
        checks.append(CheckResult(name="model_output_parse", status="fail",
                                  detail="model output for at least one page was not valid JSON; page returned empty"))
    return ExtractionResponse(**merged.model_dump(), _verification=ver, _checks=checks, _meta=meta)
