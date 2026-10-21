"""Render reports/evaluation_report.md from the JSON files written by the eval scripts.

    python -m scripts.make_report --results results --train-metrics outputs/qwen25vl3b-cord-lora/train_metrics.json

Run naming convention (see notebooks/colab_train_eval.ipynb):
  zeroshot, finetuned            main comparison (same test split, same prompt, same pipeline)
  finetuned_<perturbation>       robustness runs (rotate5, rotate90, blur, jpeg, downscale, multipage)
  finetuned_vt<N>                visual-token budget ablation
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

from docextract.metrics import ALL_FIELDS


def f(x: Optional[float], pct: bool = True, nd: int = 1) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.{nd}f}%" if pct else f"{x:.{nd}f}"


def load(results: str) -> Dict[str, dict]:
    runs = {}
    for p in sorted(glob.glob(os.path.join(results, "*.summary.json"))):
        d = json.load(open(p))
        runs[d["name"]] = d
    return runs


def headline(runs: Dict[str, dict], names: List[str], gpu_price: float) -> List[str]:
    rows = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def row(label, fn):
        rows.append(f"| {label} | " + " | ".join(fn(runs[n]) for n in names) + " |")

    row("Micro F1, raw model output", lambda r: f(r["raw"]["prf"]["micro"]["f1"]))
    row("Micro F1, after grounding", lambda r: f(r["grounded"]["prf"]["micro"]["f1"]))
    row("Micro precision raw → grounded",
        lambda r: f"{f(r['raw']['prf']['micro']['precision'])} → {f(r['grounded']['prf']['micro']['precision'])}")
    row("Micro recall raw → grounded",
        lambda r: f"{f(r['raw']['prf']['micro']['recall'])} → {f(r['grounded']['prf']['micro']['recall'])}")
    row("Document exact match (grounded)", lambda r: f(r["grounded"]["exact_match"]["document"]))
    row("JSON valid as emitted by model", lambda r: f(r["json"]["strict_valid_rate"]))
    row("JSON unparseable (needs fallback)", lambda r: f(r["json"]["unparseable_rate"]))
    row("JSON valid at API (after repair)", lambda r: f(r["json"]["service_valid_rate"]))
    row("Hallucination rate, before grounding", lambda r: f(r["hallucination"]["raw_rate"]))
    row("Hallucination rate, after grounding", lambda r: f(r["hallucination"]["grounded_rate"]))
    row("Grounding removed wrong / correct values",
        lambda r: f"{r['grounding_effect'].get('removed_wrong', 0)} / {r['grounding_effect'].get('removed_correct', 0)}")
    row("VLM latency per page p50 / p95 (s)",
        lambda r: f"{f(r['latency']['vlm_per_page_p50_s'], False, 2)} / {f(r['latency']['vlm_per_page_p95_s'], False, 2)}")
    row("End-to-end per doc p50 / p95 (s)",
        lambda r: f"{f(r['latency']['end_to_end_per_doc_p50_s'], False, 2)} / "
                  f"{f(r['latency']['end_to_end_per_doc_p95_s'], False, 2)}")
    row("Peak GPU memory (GB)", lambda r: f((r["peak_gpu_memory_mb"] or 0) / 1024, False, 2))
    row(f"GPU cost / 1,000 pages @ ${gpu_price}/h (p50)",
        lambda r: f"${r['latency']['vlm_per_page_p50_s'] * 1000 / 3600 * gpu_price:.2f}")
    return rows


def per_field(run: dict) -> List[str]:
    rows = ["| Field | P raw | R raw | F1 raw | P grounded | R grounded | F1 grounded | Exact match (grounded) |",
            "|---|---|---|---|---|---|---|---|"]
    em = run["grounded"]["exact_match"]
    for fld in ALL_FIELDS + ("micro",):
        a, b = run["raw"]["prf"][fld], run["grounded"]["prf"][fld]
        key = fld if fld in em else ("line_items" if fld == "line_items.name" else None)
        rows.append(f"| {fld} | {f(a['precision'])} | {f(a['recall'])} | {f(a['f1'])} | {f(b['precision'])} | "
                    f"{f(b['recall'])} | {f(b['f1'])} | {f(em.get(key)) if key else ''} |")
    rows.append("\n_Exact match for `line_items.name` row = whole item list exactly right._")
    return rows


def hallucination_fields(runs: Dict[str, dict], names: List[str]) -> List[str]:
    fields = sorted({k for n in names for k in runs[n]["hallucination"]["per_field_raw"]})
    rows = ["| Field | " + " | ".join(f"{n} raw → grounded" for n in names) + " |", "|---|" + "---|" * len(names)]
    for fld in fields:
        cells = []
        for n in names:
            h = runs[n]["hallucination"]
            cells.append(f"{f(h['per_field_raw'].get(fld))} → {f(h['per_field_grounded'].get(fld))}")
        rows.append(f"| {fld} | " + " | ".join(cells) + " |")
    return rows


def robustness(runs: Dict[str, dict], base: str) -> List[str]:
    names = [base] + sorted(n for n in runs if n.startswith(base + "_") and "_vt" not in n)
    rows = ["| Condition | Micro F1 raw | Micro F1 grounded | Hallucination raw → grounded | total F1 | items amount F1 | p50 s/page |",
            "|---|---|---|---|---|---|---|"]
    for n in names:
        r = runs[n]
        rows.append(
            f"| {r['perturbation']} | {f(r['raw']['prf']['micro']['f1'])} | {f(r['grounded']['prf']['micro']['f1'])} | "
            f"{f(r['hallucination']['raw_rate'])} → {f(r['hallucination']['grounded_rate'])} | "
            f"{f(r['grounded']['prf']['total']['f1'])} | {f(r['grounded']['prf']['line_items.amount']['f1'])} | "
            f"{f(r['latency']['vlm_per_page_p50_s'], False, 2)} |")
    return rows


def resolution(runs: Dict[str, dict]) -> List[str]:
    names = sorted((n for n in runs if "_vt" in n), key=lambda n: runs[n]["max_visual_tokens"])
    rows = ["| Max visual tokens | ~Max pixels | Micro F1 (grounded) | p50 s/page | Peak GPU GB |", "|---|---|---|---|---|"]
    for n in names:
        r = runs[n]
        vt = r["max_visual_tokens"]
        rows.append(f"| {vt} | {vt * 784 / 1e6:.2f} MP | {f(r['grounded']['prf']['micro']['f1'])} | "
                    f"{f(r['latency']['vlm_per_page_p50_s'], False, 2)} | {f((r['peak_gpu_memory_mb'] or 0) / 1024, False, 2)} |")
    return rows


def audit(results: str) -> List[str]:
    rows = []
    paths = sorted(glob.glob(os.path.join(results, "grounding_audit*.json")))
    if not paths:
        return rows
    audits = [json.load(open(p)) for p in paths]
    audits.sort(key=lambda a: a["text_match_threshold"])
    rows += ["| Name-match threshold | Gold values verified | False rejection | Injected hallucinations caught |",
             "|---|---|---|---|"]
    for a in audits:
        o = a["overall"]
        rows.append(f"| {a['text_match_threshold']:g} | {f(o['gold_verified_rate'])} | {f(o['false_rejection_rate'])} | "
                    f"{f(o['hallucination_detection_rate'])} |")
    main = next((a for a in audits if a["text_match_threshold"] == 85), audits[0])
    rows += ["", f"Per field at threshold {main['text_match_threshold']:g} ({main['documents']} test receipts, RapidOCR evidence):", "",
             "| Field | Gold verified | Injected caught |", "|---|---|---|"]
    det = main["hallucination_detection_rate"]
    for fld, g in main["gold_verified_rate"].items():
        d = det.get(fld)
        rows.append(f"| {fld} | {f(g['rate'])} ({g['verified']}/{g['checked']}) | "
                    + (f"{f(d['rate'])} ({d['caught']}/{d['injected']})" if d else "–") + " |")
    return rows


def feasibility_numbers(runs: Dict[str, dict], gpu_price: float) -> List[str]:
    """Key-number table injected into reports/feasibility_report.md between AUTO markers."""
    names = [n for n in ("zeroshot", "finetuned") if n in runs]
    if not names:
        return ["_Not measured yet: run `notebooks/colab_train_eval.ipynb` on a T4, then "
                "`python -m scripts.make_report` fills this table._"]
    rows = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def row(label, fn):
        rows.append(f"| {label} | " + " | ".join(fn(runs[n]) for n in names) + " |")

    row("Micro F1 (after grounding)", lambda r: f(r["grounded"]["prf"]["micro"]["f1"]))
    row("Precision / recall (after grounding)",
        lambda r: f"{f(r['grounded']['prf']['micro']['precision'])} / {f(r['grounded']['prf']['micro']['recall'])}")
    row("Total F1 / line-item amount F1",
        lambda r: f"{f(r['grounded']['prf']['total']['f1'])} / {f(r['grounded']['prf']['line_items.amount']['f1'])}")
    row("Hallucination rate before → after grounding",
        lambda r: f"{f(r['hallucination']['raw_rate'])} → {f(r['hallucination']['grounded_rate'])}")
    row("JSON valid from model / at API", lambda r: f"{f(r['json']['strict_valid_rate'])} / {f(r['json']['service_valid_rate'])}")
    row("Latency per page p50 / p95", lambda r: f"{f(r['latency']['vlm_per_page_p50_s'], False, 1)} s / "
                                                f"{f(r['latency']['vlm_per_page_p95_s'], False, 1)} s")
    row("Peak GPU memory", lambda r: f"{f((r['peak_gpu_memory_mb'] or 0) / 1024, False, 1)} GB")
    row("Throughput, 1 GPU (p50)", lambda r: f"{3600 / r['latency']['vlm_per_page_p50_s']:.0f} pages/h")
    row(f"GPU cost per 1,000 pages (${gpu_price}/h)",
        lambda r: f"${r['latency']['vlm_per_page_p50_s'] * 1000 / 3600 * gpu_price:.2f} (p50), "
                  f"${r['latency']['vlm_per_page_p95_s'] * 1000 / 3600 * gpu_price:.2f} (p95)")
    return rows


def inject(path: str, marker: str, lines: List[str]) -> None:
    if not os.path.exists(path):
        return
    text = open(path).read()
    start, end = f"<!-- AUTO:{marker}:START -->", f"<!-- AUTO:{marker}:END -->"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        with open(path, "w") as fh:
            fh.write(head + start + "\n" + "\n".join(lines) + "\n" + end + tail)
        print(f"updated {path} [{marker}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--train-metrics", default="")
    ap.add_argument("--gpu-price", type=float, default=0.526, help="$/hour; default AWS g4dn.xlarge (T4) on-demand")
    ap.add_argument("--out", default="reports/evaluation_report.md")
    ap.add_argument("--feasibility", default="reports/feasibility_report.md")
    args = ap.parse_args()

    runs = load(args.results)
    md = ["# Evaluation report", "",
          "Generated by `python -m scripts.make_report` from the JSON files in `results/`. "
          "Test set: CORD v2 `test` split (100 receipts), never used for training or model selection.", ""]
    main_runs = [n for n in ("zeroshot", "finetuned") if n in runs]
    if main_runs:
        md += ["## Zero-shot vs fine-tuned", ""] + headline(runs, main_runs, args.gpu_price) + [""]
        for n in main_runs:
            md += [f"### Per-field results: {n}", ""] + per_field(runs[n]) + [""]
        md += ["### Hallucination rate per field (share of non-null values not in the document)", ""]
        md += hallucination_fields(runs, main_runs) + [""]
    else:
        md += ["## Zero-shot vs fine-tuned", "", "_No GPU evaluation results found in `results/` yet. "
               "Run the Colab notebook (`notebooks/colab_train_eval.ipynb`) to produce them._", ""]
    if "finetuned" in runs and any(n.startswith("finetuned_") and "_vt" not in n for n in runs):
        md += ["## Robustness (failure modes)", ""] + robustness(runs, "finetuned") + [""]
    if any("_vt" in n for n in runs):
        md += ["## Image resolution vs accuracy / memory", ""] + resolution(runs) + [""]
    if "oracle_upper_bound" in runs:
        r = runs["oracle_upper_bound"]
        md += ["## Cost of grounding for a perfect model (oracle run)", "",
               "`python -m scripts.evaluate --oracle` feeds the gold labels through the full pipeline in place of "
               "the VLM. Raw F1 must be 100% (validates the metric code); the grounded numbers show the recall that "
               "OCR-based grounding takes away even when every value is right, i.e. the ceiling for any model.", "",
               "| Field | Recall after grounding | Precision after grounding |", "|---|---|---|"]
        for fld in ALL_FIELDS + ("micro",):
            g = r["grounded"]["prf"][fld]
            if g["tp"] + g["fn"]:
                md.append(f"| {fld} | {f(g['recall'])} | {f(g['precision'])} |")
        md += ["", f"Raw micro F1 {f(r['raw']['prf']['micro']['f1'])} → grounded {f(r['grounded']['prf']['micro']['f1'])}; "
               f"document exact match after grounding {f(r['grounded']['exact_match']['document'])}. "
               f"Noise floor of the hallucination metric (gold values the oracle text cannot confirm): "
               f"{f(r['hallucination']['raw_rate'])}.", ""]
    a = audit(args.results)
    if a:
        md += ["## Grounding audit (no VLM involved)", "",
               "Gold values fed through the OCR grounding step measure how often grounding would wrongly null a "
               "correct value; synthetic hallucinations (perturbed numbers, item names from other receipts) measure "
               "how many invented values it catches. Produced by `python -m scripts.grounding_audit`.", ""] + a + [""]
    if args.train_metrics and os.path.exists(args.train_metrics):
        t = json.load(open(args.train_metrics))
        md += ["## Training run", "",
               f"- GPU: {t.get('gpu')}", f"- Runtime: {t['train_runtime_s'] / 60:.1f} min",
               f"- Final train loss: {t['train_loss']:.4f}; validation loss: {t['eval'].get('eval_loss', float('nan')):.4f}",
               f"- Peak GPU memory during training: {t['peak_gpu_memory_mb'] / 1024:.2f} GB",
               f"- Hyper-parameters: `{json.dumps({k: v for k, v in t['args'].items() if k not in ('push_to_hub',)})}`", ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(md))
    print(f"wrote {args.out}")
    inject(args.feasibility, "NUMBERS", feasibility_numbers(runs, args.gpu_price))


if __name__ == "__main__":
    main()
