"""Evaluate a model (zero-shot or with a LoRA adapter) on the held-out CORD v2 test split.

    python -m scripts.evaluate --name zeroshot
    python -m scripts.evaluate --name finetuned --adapter outputs/qwen25vl3b-cord-lora/adapter
    python -m scripts.evaluate --name finetuned_rot90 --adapter ... --perturb rotate90

Each run runs the same code path as the API (model -> parse -> OCR grounding -> merge ->
checks) and writes:
  results/<name>.jsonl         per-document predictions (raw and grounded)
  results/<name>.summary.json  all metrics, before and after grounding
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import time
from collections import defaultdict
from typing import List

from PIL import Image, ImageFilter
from tqdm import tqdm

from docextract.config import get_settings
from docextract.cord import (CORD_DATASET, document_lines, gt_to_target, parse_ground_truth, summary_crop_y,
                             target_to_json)
from docextract.metrics import Counts, grounding_effect, hallucinations, percentile, score_document
from docextract.model import Backend, Generation, load_backend
from docextract.ocr import page_evidence
from docextract.pdf import Page
from docextract.pipeline import ground_and_merge, run_model
from docextract.schema import Extraction, ExtractionResponse

PERTURBATIONS = ("none", "rotate5", "rotate90", "blur", "jpeg", "downscale", "multipage")


def perturb(img: Image.Image, kind: str, gt: dict) -> List[Image.Image]:
    """Stress tests for the failure-mode analysis in the feasibility report."""
    if kind == "rotate5":
        return [img.rotate(5, expand=True, fillcolor="white", resample=Image.BICUBIC)]
    if kind == "rotate90":
        return [img.rotate(90, expand=True)]
    if kind == "blur":  # poor scan / out-of-focus photo
        return [img.filter(ImageFilter.GaussianBlur(2.0))]
    if kind == "jpeg":  # heavy compression artefacts
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=10)
        return [Image.open(io.BytesIO(buf.getvalue())).convert("RGB")]
    if kind == "downscale":  # small fonts: 1/3 resolution, then back up
        w, h = img.size
        return [img.resize((w // 3, h // 3), Image.BILINEAR).resize((w, h), Image.BILINEAR)]
    if kind == "multipage":  # split the receipt into 2 pages between items and totals
        y = summary_crop_y(gt)
        h0 = gt["meta"]["image_size"]["height"] or img.height
        cut = int(y * img.height / h0) if y else img.height // 2
        return [img.crop((0, 0, img.width, cut)), img.crop((0, cut, img.width, img.height))]
    return [img]


class OracleBackend(Backend):
    """Answers with the gold JSON. Validates the metric code (raw F1 must be 100%) and
    measures what grounding alone costs a perfect model. CPU only."""

    name = "oracle"
    gold_json = ""

    def generate(self, image, deadline=None) -> Generation:
        return Generation(self.gold_json, 0.0)


def to_ext(resp: ExtractionResponse) -> Extraction:
    return Extraction(**resp.model_dump(include={"store_name", "date", "line_items", "subtotal", "tax", "total"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default="", help="LoRA adapter path or HF id; empty = zero-shot")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--perturb", default="none", choices=PERTURBATIONS)
    ap.add_argument("--max-visual-tokens", type=int, default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--oracle", action="store_true", help="use gold labels as model output (no GPU)")
    args = ap.parse_args()
    assert args.split != "train", "never evaluate on training data"

    from datasets import load_dataset

    settings = get_settings()
    if args.max_visual_tokens:
        settings = dataclasses.replace(settings, max_visual_tokens=args.max_visual_tokens)
    flag_settings = dataclasses.replace(settings, grounding_policy="flag")
    null_settings = dataclasses.replace(settings, grounding_policy="null")

    backend = OracleBackend() if args.oracle else load_backend(settings, adapter=args.adapter)
    ds = load_dataset(CORD_DATASET, split=args.split)
    if args.limit:
        ds = ds.select(range(args.limit))

    os.makedirs(args.out_dir, exist_ok=True)
    raw_c, final_c = Counts(), Counts()
    hall = {"raw": [0, 0], "grounded": [0, 0]}
    hall_fields = {"raw": defaultdict(lambda: [0, 0]), "grounded": defaultdict(lambda: [0, 0])}
    effect = defaultdict(int)
    page_lat, ocr_lat, doc_lat, out_tokens = [], [], [], []
    strict_valid = parse_fail = repaired = final_valid = timed_out = 0
    store_date_nonnull = 0
    backend.reset_peak_memory()

    with open(os.path.join(args.out_dir, f"{args.name}.jsonl"), "w") as fout:
        for i, row in enumerate(tqdm(ds, desc=args.name)):
            gt = parse_ground_truth(row["ground_truth"])
            gold = gt_to_target(gt["gt_parse"])
            images = perturb(row["image"].convert("RGB"), args.perturb, gt)
            pages = [Page(index=k, image=im) for k, im in enumerate(images)]
            if args.oracle:
                assert len(pages) == 1, "oracle mode supports single-page documents only"
                backend.gold_json = target_to_json(gold)

            t0 = time.monotonic()
            outputs = run_model(backend, pages, deadline=None)
            t1 = time.monotonic()
            evidence = [page_evidence(p) for p in pages]
            t2 = time.monotonic()
            raw = to_ext(ground_and_merge(outputs, evidence, flag_settings))  # merged, nothing removed
            final_resp = ground_and_merge(outputs, evidence, null_settings)
            final = to_ext(final_resp)
            doc_lat.append(time.monotonic() - t0)
            ocr_lat.append(t2 - t1)
            for o in outputs:
                page_lat.append(o.generation.seconds)
                out_tokens.append(o.generation.output_tokens)
                strict_valid += o.parse.strict_valid
                parse_fail += o.parse.fallback_empty
                repaired += o.parse.repaired
                timed_out += o.generation.timed_out
            try:
                ExtractionResponse.model_validate(json.loads(json.dumps(final_resp.to_json_dict())))
                final_valid += 1
            except Exception:
                pass

            score_document(raw_c, raw, gold)
            score_document(final_c, final, gold)
            # Oracle = annotated words + their boxes; independent of the (perturbed) image.
            oracle = document_lines(gt)
            for key, ext in (("raw", raw), ("grounded", final)):
                checked, missing, per = hallucinations(ext, oracle)
                hall[key][0] += checked
                hall[key][1] += missing
                for f, (c, m) in per.items():
                    hall_fields[key][f][0] += c
                    hall_fields[key][f][1] += m
            store_date_nonnull += (raw.store_name is not None) + (raw.date is not None)
            for k, v in grounding_effect(raw, final, gold).items():
                effect[k] += v

            fout.write(json.dumps({
                "idx": i, "gold": gold, "raw_text": [o.generation.text for o in outputs],
                "raw": raw.model_dump(), "grounded": final_resp.to_json_dict(),
                "page_seconds": [round(o.generation.seconds, 3) for o in outputs],
            }, ensure_ascii=False) + "\n")

    n_pages = len(page_lat)
    rate = lambda pair: (pair[1] / pair[0]) if pair[0] else None  # noqa: E731
    summary = {
        "name": args.name,
        "model": backend.name,
        "adapter": args.adapter or None,
        "split": args.split,
        "documents": len(ds),
        "pages": n_pages,
        "perturbation": args.perturb,
        "max_visual_tokens": settings.max_visual_tokens,
        "raw": {"prf": raw_c.prf(), "exact_match": raw_c.exact_match()},
        "grounded": {"prf": final_c.prf(), "exact_match": final_c.exact_match()},
        "json": {
            "strict_valid_rate": strict_valid / n_pages,
            "repaired_rate": repaired / n_pages,
            "unparseable_rate": parse_fail / n_pages,
            "service_valid_rate": final_valid / len(ds),
            "timed_out_pages": timed_out,
        },
        "hallucination": {
            "raw_rate": rate(hall["raw"]), "grounded_rate": rate(hall["grounded"]),
            "raw_counts": hall["raw"], "grounded_counts": hall["grounded"],
            "per_field_raw": {f: rate(v) for f, v in hall_fields["raw"].items()},
            "per_field_grounded": {f: rate(v) for f, v in hall_fields["grounded"].items()},
            "store_name_or_date_predicted_non_null": store_date_nonnull,
        },
        "grounding_effect": dict(effect),
        "latency": {
            "vlm_per_page_p50_s": percentile(page_lat, 50), "vlm_per_page_p95_s": percentile(page_lat, 95),
            "ocr_per_doc_p50_s": percentile(ocr_lat, 50),
            "end_to_end_per_doc_p50_s": percentile(doc_lat, 50), "end_to_end_per_doc_p95_s": percentile(doc_lat, 95),
            "output_tokens_p50": percentile(out_tokens, 50),
        },
        "peak_gpu_memory_mb": backend.peak_memory_mb(),
    }
    with open(os.path.join(args.out_dir, f"{args.name}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    m_raw, m_g = summary["raw"]["prf"]["micro"], summary["grounded"]["prf"]["micro"]
    print(json.dumps({"micro_f1_raw": m_raw["f1"], "micro_f1_grounded": m_g["f1"],
                      "hallucination": summary["hallucination"]["raw_rate"],
                      "hallucination_grounded": summary["hallucination"]["grounded_rate"],
                      "latency": summary["latency"], "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
                      "json": summary["json"]}, indent=2))


if __name__ == "__main__":
    main()
