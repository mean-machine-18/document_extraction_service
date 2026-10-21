# Feasibility report: is a small fine-tuned VLM good enough for production extraction?

**Setup.** Qwen2.5-VL-3B-Instruct with a QLoRA adapter trained on CORD v2 (800 receipts), served one page at a time on one GPU. Every value is checked against OCR/text-layer evidence before it is returned. Test set: the CORD v2 `test` split (100 receipts), which is never used for training.

## 1. Short answer

**Yes for single-page, receipt-like documents, provided three things are in place:** the grounding layer, a review queue for fields marked `unverified`, and the understanding that "no answer" is a normal outcome. **No** as an unsupervised source of truth for dense multi-page tables, or for high-volume work on a T4 with one-job-at-a-time serving.

What drives that answer:

- **Accuracy:** see the measured table below. The model alone is not the safety mechanism. Grounding is, and it was measured separately on all 100 test receipts:
  - It caught **97.6% of injected hallucinations** (perturbed totals and amounts, wrong quantities, item names that are not on the receipt).
  - It wrongly nulled **12.9% of correct values**. With a perfect model, grounded micro-recall would be 85.6% at 100% precision, and grounded F1 would be 92.3% (oracle run).
  - So the service fails *safe*: errors turn into `null`/`unverified`, not into wrong numbers.
- **Cost:** a 3B VLM fits a 16 GB T4 with room to spare. Even at T4 prices, GPU cost per 1,000 pages is around a dollar or two (formula in section 3). Latency is the real constraint: each page takes several seconds, because the model generates every JSON token one at a time.

## 2. Numbers

**Measured on T4.** This table is filled automatically by `python -m scripts.make_report` from `results/*.summary.json`:

<!-- AUTO:NUMBERS:START -->
_Not measured yet: run `notebooks/colab_train_eval.ipynb` on a T4, then `python -m scripts.make_report` fills this table._
<!-- AUTO:NUMBERS:END -->

**Measured on CPU (Apple M2 Pro).** These numbers do not need the VLM:

| | Value |
|---|---|
| Gold values confirmed by OCR grounding (name threshold 85) | 87.1% (false rejection 12.9%) |
| Injected hallucinations caught | 97.6% (amount 97.6%, qty 93.9%, total 98.9%, names 100%) |
| Grounded recall / precision with a perfect model (oracle) | 85.6% / 100% |
| Weakest field under grounding | `qty`: 68.4% of correct quantities confirmed. A lone "1" is often missed by OCR detection; the bounding-box crop re-read raised this from 56% |
| OCR evidence time per receipt (RapidOCR, CPU) | ≈0.4 s p50. Runs in parallel with the GPU job, so it adds no time to the critical path |

**Analytical estimate for T4 fp16, before measurement.** Use these only as a sanity check for the measured table:

- **Peak GPU memory, about 9–10.5 GB.** Language-model weights (3.1B params in fp16) take 6.2 GB. The ViT (0.67B params) runs in fp32 on T4, because the ViT overflows in fp16, and takes 2.7 GB. Activations at 1,024 visual tokens add about 1 GB. Setting `LOAD_IN_4BIT=1` brings this to about 4 GB.
- **Latency, about 8–12 s per page.** Decoding is memory-bound at roughly 35–50 ms per token. A receipt JSON is about 150–400 tokens. Prefill plus the vision encoder add about 1 s.

## 3. Cost per 1,000 pages

`cost = p50_seconds_per_page × 1000 / 3600 × GPU_$/h`. With one job at a time, one GPU handles 3600 / p50 pages per hour. At 10 s/page on an AWS g4dn.xlarge (T4, $0.526/h on-demand), that is **≈$1.46 per 1,000 pages** and 360 pages/h per GPU. OCR runs on the CPU cores of the same instance and costs nothing extra.

The serving policy is one job per GPU because the assignment requires it. In production, the same model behind vLLM with continuous batching would run 8–16 pages concurrently on an L4 or A10G. That would plausibly cut cost per page by 5× or more, at the price of losing the strict one-job guarantee.

## 4. Where it fails, and how each failure is measured

`scripts/evaluate.py --perturb …` re-runs the test set under each condition. The results are in `reports/evaluation_report.md` → *Robustness*.

| Failure mode | Mechanism | Measured by | Behaviour with grounding |
|---|---|---|---|
| **Dense tables / long receipts** | Output length grows with the number of rows, so latency grows linearly. Above `MAX_NEW_TOKENS` the JSON is truncated. Item alignment errors (merged or split rows) grow with density. | per-page latency vs. output tokens; `line_items.*` F1 | Truncated JSON is repaired back to the last complete item. Missing rows lower recall but do not produce wrong values. |
| **Poor scans** (blur, JPEG) | The VLM misreads digits, e.g. 8↔6 and 1↔7. OCR degrades too, so fewer values can be confirmed. | `blur`, `jpeg` | Recall drops; misread digits rarely match OCR evidence, so they come back as `unverified`/null. |
| **Rotation** | Small skew is fine: training used ±4° augmentation. At 90° or 180° the model was never trained and reading collapses. | `rotate5`, `rotate90` | Fix: detect orientation before inference (OCR angle classifier or Tesseract OSD) and rotate the page back. Not implemented. |
| **Small fonts** | The visual token budget caps resolution, at 1,024 tokens ≈ 0.8 MP. Thin glyphs disappear after the 28 px patch merge. | `downscale`; resolution ablation (`*_vt256 … _vt1536`) | Raise `MAX_VISUAL_TOKENS` for such documents. Cost is roughly linear in visual tokens for prefill and memory. |
| **Multi-page documents** | Each page is processed on its own, with no cross-page context. The header comes from the first page and the totals from the last. Items that continue across a page break can be split or duplicated. | `multipage` (each receipt split into 2 pages) | Sanity checks (items sum vs. subtotal) flag inconsistent merges. |
| **Fields CORD does not label** | CORD v2 has no store-name or date annotations, so the model is trained to return null for both and cannot be scored on them. | count of non-null `store_name`/`date` | Any value that is returned must still pass grounding. |
| **Field confusion** | The model may put the cash-paid amount in `total`. That number *is* on the page, so grounding passes it. | `total` precision | Partly caught by the `subtotal + tax = total` check (reported in `_checks`). Grounding checks that a value exists, not what role it plays. |

## 5. OCR + text-only model, briefly

This pipeline was not built. The grounding audit gives a bound on it: RapidOCR's page text contains 88–98% of the gold item names, amounts, tax, subtotals and totals, but only about 56% of quantities without crop re-reads. An OCR + LLM pipeline would inherit those misses as hard recall limits, and it would lose the layout reasoning (which number belongs to which row) that the VLM gets from the image. On the other hand, it would be cheaper per page on CPU-heavy hardware, and every value it returns would be grounded by construction. The VLM + OCR-verification design used here combines the VLM's reading with OCR's guarantee.

## 6. Recommendation

1. Ship for receipts and invoices of one or two pages, with `GROUNDING_POLICY=null` and a review UI driven by `_verification`.
2. Before broadening scope:
   - add orientation detection;
   - add a document-type check that sends dense tables to a table-specific path;
   - add labelled data for `store_name` and `date` (CORD lacks them; SROIE has both).
3. For volume, move from one-job-at-a-time HF `generate` to batched serving (vLLM) on an L4. Re-measure cost and latency with `scripts/evaluate.py`.
