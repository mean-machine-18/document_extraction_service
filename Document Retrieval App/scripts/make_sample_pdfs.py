"""Turn a few held-out CORD test receipts into PDFs for demos and the concurrency test.

    python -m scripts.make_sample_pdfs --n 3 --out samples

Writes image-only PDFs (no text layer, so the OCR grounding path is exercised) plus one
2-page PDF built from two receipts to show multi-page merging.
"""
from __future__ import annotations

import argparse
import os

from docextract.cord import CORD_DATASET
from docextract.pdf import images_to_pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="samples")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(CORD_DATASET, split="test").select(range(args.n))
    os.makedirs(args.out, exist_ok=True)
    images = [r["image"].convert("RGB") for r in ds]
    for i, img in enumerate(images):
        path = os.path.join(args.out, f"cord_test_{i:03d}.pdf")
        with open(path, "wb") as f:
            f.write(images_to_pdf([img]))
        print("wrote", path)
    if len(images) >= 2:
        path = os.path.join(args.out, "cord_test_2pages.pdf")
        with open(path, "wb") as f:
            f.write(images_to_pdf(images[:2]))
        print("wrote", path)


if __name__ == "__main__":
    main()
