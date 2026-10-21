"""Measure the grounding step on its own, without a VLM (runs on CPU).

1. False rejections: feed the GOLD values through OCR grounding. Every gold value is on
   the page, so anything marked "unverified" is a value the grounding would wrongly null.
2. Detection: inject synthetic hallucinations (numbers perturbed, item names taken from
   other receipts) and count how many grounding catches.

    python -m scripts.grounding_audit --split test --out results/grounding_audit.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from tqdm import tqdm

from docextract.cord import CORD_DATASET, gt_to_target, parse_ground_truth
from docextract.grounding import UNVERIFIED, VERIFIED, ground_page
from docextract.ocr import page_evidence
from docextract.pdf import Page
from docextract.schema import Extraction, LineItem


def perturb_number(v: float, rng: random.Random) -> float:
    """A plausible-looking wrong value (what a VLM misreading or inventing would produce)."""
    choice = rng.random()
    if choice < 0.4:
        return v + rng.choice([-1, 1]) * rng.choice([500, 1000, 2000, 5000])
    if choice < 0.7:  # one digit changed
        s = list(str(int(v)))
        i = rng.randrange(len(s))
        s[i] = str((int(s[i]) + rng.randint(1, 9)) % 10)
        return float("".join(s).lstrip("0") or "1")
    return round(v * rng.choice([1.1, 0.9, 1.5]), -2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=85.0)
    ap.add_argument("--out", default="results/grounding_audit.json")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(CORD_DATASET, split=args.split)
    if args.limit:
        ds = ds.select(range(args.limit))
    rng = random.Random(0)
    golds = [gt_to_target(parse_ground_truth(r["ground_truth"])["gt_parse"]) for r in ds]
    all_names = [it["name"] for g in golds for it in g["line_items"]]

    gold_stats = defaultdict(lambda: [0, 0])  # field -> [checked, verified]
    fake_stats = defaultdict(lambda: [0, 0])  # field -> [injected, caught]
    for row, gold in tqdm(zip(ds, golds), total=len(golds)):
        ev = page_evidence(Page(index=0, image=row["image"].convert("RGB")))
        lines = ev.lines

        def ground(ext: Extraction):
            return ground_page(ext, lines, policy="flag", text_threshold=args.threshold, crop_reader=ev.reader)

        g = ground(Extraction(**gold))
        for f in ("subtotal", "tax", "total"):
            if gold[f] is not None:
                gold_stats[f][0] += 1
                gold_stats[f][1] += g.verification[f] == VERIFIED
        for iv in g.item_verification:
            for k, status in iv.items():
                if status != "not_found":
                    gold_stats[f"line_items.{k}"][0] += 1
                    gold_stats[f"line_items.{k}"][1] += status == VERIFIED

        # synthetic hallucinations, one field at a time so they do not interact
        page_text = " ".join(l.text for l in lines).upper()
        for f in ("subtotal", "tax", "total"):
            if gold[f]:
                fake = perturb_number(gold[f], rng)
                fg = ground(Extraction(**dict(gold, **{f: fake})))
                fake_stats[f][0] += 1
                fake_stats[f][1] += fg.verification[f] == UNVERIFIED
        if gold["line_items"]:
            j = rng.randrange(len(gold["line_items"]))
            item = gold["line_items"][j]
            if item["amount"]:
                items = [dict(x) for x in gold["line_items"]]
                items[j]["amount"] = perturb_number(item["amount"], rng)
                fg = ground(Extraction(**dict(gold, line_items=items)))
                fake_stats["line_items.amount"][0] += 1
                fake_stats["line_items.amount"][1] += fg.item_verification[j]["amount"] == UNVERIFIED
            if item["qty"]:
                items = [dict(x) for x in gold["line_items"]]
                items[j]["qty"] = item["qty"] + rng.choice([1, 2, 3])
                fg = ground(Extraction(**dict(gold, line_items=items)))
                fake_stats["line_items.qty"][0] += 1
                fake_stats["line_items.qty"][1] += fg.item_verification[j]["qty"] == UNVERIFIED
            # an item name from another receipt that is not printed on this one
            candidates = [n for n in all_names if n.upper() not in page_text]
            if candidates:
                fake_item = LineItem(name=rng.choice(candidates), amount=None)
                fg = ground(Extraction(**dict(gold, line_items=[fake_item])))
                fake_stats["line_items.name"][0] += 1
                fake_stats["line_items.name"][1] += fg.item_verification[0]["name"] == UNVERIFIED

    report = {
        "split": args.split, "documents": len(golds), "text_match_threshold": args.threshold,
        "gold_verified_rate": {f: {"checked": c, "verified": v, "rate": v / c if c else None}
                               for f, (c, v) in sorted(gold_stats.items())},
        "hallucination_detection_rate": {f: {"injected": c, "caught": v, "rate": v / c if c else None}
                                         for f, (c, v) in sorted(fake_stats.items())},
    }
    tot_c = sum(c for c, _ in gold_stats.values())
    tot_v = sum(v for _, v in gold_stats.values())
    inj = sum(c for c, _ in fake_stats.values())
    caught = sum(v for _, v in fake_stats.values())
    report["overall"] = {"gold_verified_rate": tot_v / tot_c, "false_rejection_rate": 1 - tot_v / tot_c,
                         "hallucination_detection_rate": caught / inj}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
