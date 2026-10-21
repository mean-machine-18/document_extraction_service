# Receipt extraction with a fine-tuned small VLM

This service takes a PDF upload and returns structured JSON, validated against a schema, in the same HTTP response. A **Qwen2.5-VL-3B** model, fine-tuned with QLoRA on CORD v2, reads the page images. Before any value is returned, it is **checked against text that is actually on the page**: the PDF text layer, or OCR with bounding-box crop re-reads. Values that cannot be confirmed come back as `null` and are marked `unverified`.

```
PDF ──► validate ──► render pages (200 dpi) ──┬──► [GPU slot: one job at a time] Qwen2.5-VL-3B + LoRA ──► JSON parse/repair ──┐
                                              └──► evidence: PDF text layer or RapidOCR (CPU, in parallel) ─────────────────┤
                                                                                                                             ▼
                                         response ◄── merge pages ◄── arithmetic sanity checks ◄── grounding (null unverified values)
```

| Deliverable | Where |
|---|---|
| Service code | [`docextract/`](docextract/) |
| Training script (QLoRA) | [`training/train_lora.py`](training/train_lora.py) |
| Colab notebook (train + all GPU evaluations + concurrency run) | [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb) |
| LoRA adapter on Hugging Face | `https://huggingface.co/<hf-user>/qwen25vl3b-cord-lora` ← *pushed by the notebook* |
| Evaluation report, zero-shot vs. fine-tuned | [`reports/evaluation_report.md`](reports/evaluation_report.md) (generated) |
| Feasibility report (Task D) | [`reports/feasibility_report.md`](reports/feasibility_report.md) |
| Concurrency test script + captured output | [`scripts/concurrency_test.py`](scripts/concurrency_test.py), [`reports/concurrency_output*.txt`](reports/) |
| Dockerfile | [`Dockerfile`](Dockerfile) |

> **Status of the numbers.** This repository was built on a machine without a GPU. Everything that does not need the VLM has been run, and its real output is committed:
> - the 43 unit/API tests;
> - the grounding audit on all 100 CORD test receipts;
> - the oracle pipeline run;
> - the concurrency test, run against the mock backend.
>
> The zero-shot and fine-tuned evaluation, training, latency, GPU-memory and real-model concurrency numbers come from running [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb) on a free T4. That takes roughly 3–4 h. `scripts/make_report.py` then writes them into both reports automatically. No GPU number in this repository is made up: where none has been measured yet, the reports say so.

---

## 1. Quick start

### Local, CPU only (tests, API with the mock model, grounding audit)
```bash
make setup                      # python3 -m venv .venv && pip install -r requirements-dev.txt
make test                       # 43 tests: parsing, normalisation, grounding, API errors, queue, timeouts
make serve-mock                 # API on :8000 with a canned model answer, everything else real
make concurrency                # 5 parallel requests -> reports/concurrency_output.txt
make audit oracle report        # grounding audit + oracle run + reports (CPU, ~5 min)
```
The mock backend always answers with a fake "MOCK ITEM 10,000" receipt. On a real receipt, grounding therefore removes every value, which is a quick way to see the hallucination control at work.

### GPU (Colab / Kaggle T4, or any CUDA box)
Open `notebooks/colab_train_eval.ipynb`, set `REPO_URL` and `HF_ADAPTER_REPO`, then run all cells. To do the same by hand:
```bash
pip install -r requirements-train.txt
python -m scripts.evaluate --name zeroshot                                   # baseline
python -m training.train_lora --output-dir outputs/lora --push-to-hub <user>/qwen25vl3b-cord-lora
python -m scripts.evaluate --name finetuned --adapter outputs/lora/adapter
python -m scripts.make_report --train-metrics outputs/lora/train_metrics.json
ADAPTER=<user>/qwen25vl3b-cord-lora uvicorn docextract.service.app:app --port 8000 --workers 1
```

### Docker
```bash
docker build -t receipt-extractor .                         # add --build-arg PRELOAD_MODEL=1 to bake weights in
docker run --gpus all -p 8000:8000 -e ADAPTER=<user>/qwen25vl3b-cord-lora receipt-extractor
docker run -p 8000:8000 -e MODEL_BACKEND=mock receipt-extractor   # no GPU
curl -F file=@samples/cord_test_000.pdf localhost:8000/extract
```

---

## 2. API

`POST /extract`: multipart form with a field `file` holding the PDF. Add `?debug=true` to also get the raw model output and the OCR evidence in `_meta`.

Example response (CORD test receipt #2, produced by the pipeline in oracle mode):
```json
{
  "store_name": null,
  "date": null,
  "line_items": [
    {"name": "JASMINE MT ( L )",    "qty": null, "unit_price": null, "amount": 24000.0},
    {"name": "COCONUT JELLY ( L )", "qty": null, "unit_price": null, "amount": 4000.0}
  ],
  "subtotal": 28000.0, "tax": null, "total": 28000.0,
  "_verification": {
    "store_name": "not_found", "date": "not_found", "subtotal": "verified", "tax": "not_found", "total": "verified",
    "line_items[0].name": "verified", "line_items[0].qty": "unverified", "line_items[0].amount": "verified", "...": "...",
    "line_items": "unverified"
  },
  "_checks": [
    {"name": "line_items_sum_to_subtotal",     "status": "pass", "detail": "sum(amount)=28000 vs subtotal=28000"},
    {"name": "subtotal_plus_tax_equals_total", "status": "pass", "detail": "28000 + 0 = 28000"}
  ],
  "_meta": {"pages": 1, "evidence_source": ["ocr"], "queue_wait_ms": 0, "inference_ms": 0, "total_ms": 512}
}
```
In this example, the "1" in the quantity column was not found by OCR, so `qty` was nulled and marked `unverified`. This is the system choosing "no answer" over an unconfirmed answer (see the limitations in section 3.5).

**`_verification` values:**
- `verified`: found on the page.
- `unverified`: the model produced a value that could not be confirmed, so it was set to `null`. With `GROUNDING_POLICY=flag` the value is kept instead.
- `not_found`: the model reported that the field is not on the page.

`_checks` and `_meta` are additive diagnostics. The six schema fields and `_verification` follow the assignment schema exactly.

| Status | `error.code` | When |
|---|---|---|
| 400 | `empty_file` | empty upload |
| 415 | `not_a_pdf` | no `%PDF` header |
| 413 | `file_too_large` | over `MAX_FILE_MB` (checked from Content-Length first, then while streaming) |
| 413 | `too_many_pages` | over `MAX_PAGES` |
| 422 | `corrupt_pdf` | PyMuPDF cannot open or render it |
| 422 | `encrypted_pdf` | password-protected |
| 503 + `Retry-After` | `busy` | GPU busy and wait queue full |
| 503 + `Retry-After` | `queue_timeout` | waited longer than `QUEUE_TIMEOUT_S` |
| 504 | `inference_timeout` | generation exceeded `JOB_TIMEOUT_S` (stopped inside `generate`, so the GPU is actually freed) |
| 500 | `internal_error` | anything unexpected; still returned as JSON |

Other endpoints:
- `GET /healthz`: liveness.
- `GET /readyz`: model loaded.
- `GET /status`: running/waiting jobs and average job time.

Configuration is through environment variables, all in [`docextract/config.py`](docextract/config.py):

| Variable | Default | Controls |
|---|---|---|
| `BASE_MODEL`, `ADAPTER` | `Qwen/Qwen2.5-VL-3B-Instruct`, `""` | model; an empty adapter gives zero-shot |
| `MAX_VISUAL_TOKENS` | `1024` | resolution budget per page |
| `LOAD_IN_4BIT` | `0` | 4-bit serving |
| `MAX_FILE_MB`, `MAX_PAGES` | `20`, `5` | input limits |
| `MAX_QUEUE`, `QUEUE_TIMEOUT_S`, `JOB_TIMEOUT_S` | `3`, `60`, `90` | concurrency gate |
| `GROUNDING_POLICY` | `null` | `null` or `flag` |
| `TEXT_MATCH_THRESHOLD` | `85` | fuzzy name-match threshold |
| `MODEL_BACKEND` | `qwen` | `qwen` or `mock` |

---

## 3. Design decisions and trade-offs

### 3.1 Model: Qwen2.5-VL-3B-Instruct
| Candidate | Why not / why |
|---|---|
| Florence-2 (0.23/0.77B) | Very fast, but it is a seq2seq model tuned for captioning and OCR tasks. Long nested JSON output and following instructions are weak, and it would need a custom task head. |
| PaliGemma 2 (3B) | Strong OCR, but a fixed 224/448/896 input resolution. Tall receipts get squashed or need 896² (4,096 tokens), and there is little instruction tuning for structured output. |
| SmolVLM (256M–2.2B) | Smallest memory footprint, but noticeably weaker at dense text reading and digits. Digits are the core of this task. |
| **Qwen2.5-VL-3B** ✔ | **Native dynamic resolution:** the aspect ratio is kept and `max_pixels` sets an explicit accuracy/memory dial, which suits tall receipts. It has strong document and OCR pre-training, already produces JSON zero-shot (which makes the baseline meaningful), is supported by PEFT/bitsandbytes, and fits a T4 for both QLoRA training and fp16 inference. |

### 3.2 Dataset: CORD v2 (`naver-clova-ix/cord-v2`)
- **Splits:** train (800) is used for training. Validation (100) is used only for eval loss. Test (100) is used **only** in `scripts/evaluate.py` and the audit, and `evaluate.py` refuses `--split train`.
- **Label mapping** ([`docextract/cord.py`](docextract/cord.py)):
  - `menu.nm`/`cnt`/`unitprice`/`price` → `name`/`qty`/`unit_price`/`amount`.
  - `sub` (modifier) rows are kept as their own items, because they are printed and count towards the subtotal.
  - `sub_total.subtotal_price`/`tax_price` → `subtotal`/`tax`, and `total.total_price` → `total`.
  - Indonesian number formats (`60.000`, `Rp 29,090`, `-Rp 34,363`, `1.00`) are normalised, with unit tests.
- **Important gap:** CORD's public release does **not** label store name or date. In every target they are therefore `null`. The model learns to abstain on those fields, and they cannot be scored on CORD. SROIE has both but no line items, so it would need a multi-task setup. That is listed under future work rather than half-done.

### 3.3 Fine-tuning ([`training/`](training/))
- **QLoRA:**
  - NF4 4-bit base with double quantisation.
  - LoRA r=16, α=32, dropout 0.05.
  - LoRA is applied to the **language model's** q/k/v/o and MLP projections only. The ViT and merger are frozen and left unquantised: 800 receipts are too few to adapt the vision encoder, and the task is mostly about output format and reading conventions.
  - lr 2e-4 with cosine schedule, 2 epochs, batch 1 × grad-accum 8, gradient checkpointing.
  - Loss is computed only on the assistant answer.
- **T4 specifics:** a T4 has no bf16, so compute is fp16. The Qwen2.5-VL ViT overflows in fp16, so it is kept in fp32 (+1.3 GB). Checkpoints are saved every 50 steps and `--resume` continues after a Colab disconnect.
- **Abstention augmentation** (15% of training samples): the receipt is cropped above the totals block, using CORD's boxes, and the target sets `subtotal`/`tax`/`total` to `null`. This teaches the model that a value it cannot see is `null`, which is the assignment's top priority.
- **Robustness augmentation** (30% of samples): ±4° rotation, blur and contrast.
- **The same prompt** ([`docextract/prompts.py`](docextract/prompts.py)) is used for zero-shot, training and serving, so the comparison only measures the adapter.
- At serving time the adapter is merged into the fp16 base for speed. That differs slightly from the 4-bit base it was trained on. `LOAD_IN_4BIT=1` serves exactly as trained, with less memory but somewhat slower.

### 3.4 Image resolution
Qwen2.5-VL turns every 28×28 px area into one visual token, so `MAX_VISUAL_TOKENS` sets the pixel budget directly. Pages are rendered at 200 dpi, then downscaled by the processor to fit the budget with the aspect ratio kept.

| Max visual tokens | ≈ max pixels | A CORD receipt (≈864×1296) becomes | Effect |
|---|---|---|---|
| 256 | 0.2 MP | ~390×590 | small print becomes unreadable |
| 512 | 0.4 MP | ~550×830 | most text is fine; digits in faint prints suffer |
| **1024 (default)** | **0.8 MP** | **~780×1170** | close to native for CORD; prefill + ViT activations are still small on T4 |
| 1536 | 1.2 MP | native | marginal gain for receipts; more ViT memory and prefill time |

Memory and prefill time scale roughly linearly with visual tokens. Decoding time does not depend on them, and for long JSON it dominates. 1,024 tokens is therefore the point where extra resolution stops being cheap. The notebook measures F1, latency and memory at 256/512/1024/1536, and the results go into *Image resolution* in the evaluation report.

### 3.5 Structured output and hallucination control (Task B)
1. **Parse** ([`docextract/parsing.py`](docextract/parsing.py)):
   - Strip code fences and take the JSON object.
   - Fix trailing commas.
   - **Recover JSON truncated** at `MAX_NEW_TOKENS` by cutting after the last complete item.
   - Coerce numbers written as strings and dates into ISO format.
   - Validate with Pydantic ([`docextract/schema.py`](docextract/schema.py)). If nothing can be parsed, return the all-null object and add a `_checks` entry.

   The response is therefore always schema-valid (100% JSON validity at the API). The evaluation separately reports how often the **model itself** emitted valid JSON.
2. **Grounding** ([`docextract/grounding.py`](docextract/grounding.py)). The evidence is the PDF text layer if the page has one, otherwise RapidOCR (PaddleOCR models on ONNX/CPU, about 15 MB, no GPU contention).
   - **Numbers:** a number token in the evidence must parse to the same value. All plausible readings are tried, so `45.000`, `45,000` and `Rp45000` all match 45000. Matching is exact; there is no tolerance.
   - **Small numbers** (two digits or fewer, such as qty 1 or 2) occur everywhere on a page. They count only if they appear in the **item's own row band**: the name's box plus the row below it.
   - **Bounding-box crop re-read:** if OCR detection missed the digit, the strips left and right of the name box are cropped and run through the recogniser alone. Common digit confusions are mapped (l/I/|→1, O→0). This raised gold `qty` verification from 56% to 68%.
   - **Names:** fuzzy match against an evidence line with rapidfuzz `partial_ratio` ≥ 85. Names shorter than 4 characters need an exact token match.
   - **Date:** must equal a date-like string on the page. Both day-first and month-first readings are accepted.
   - **Policy:** with `null` (the default), an unverified value becomes `null` and an item whose name is not found is dropped and listed in `_checks`. With `flag`, values are kept and only marked.
3. **Sanity checks** ([`grounding.sanity_checks`](docextract/grounding.py)):
   - qty × unit_price = amount;
   - Σ amounts = subtotal (or total);
   - subtotal + tax = total, or subtotal = total for tax-inclusive receipts.

   These are **reported, not enforced**. A failed check says *something* is inconsistent but not *which* value. CORD receipts also legitimately break these identities with service charges, discounts and rounding, which the schema does not model. Nulling on a failed check would remove correct, grounded values.

**Grounding measured by itself** (CPU, all 100 test receipts, [`results/grounding_audit*.json`](results/)):

| Name threshold | Gold values confirmed | False rejection | Injected hallucinations caught |
|---|---|---|---|
| 75 | 88.6% | 11.4% | 97.6% |
| **85 (default)** | **87.1%** | **12.9%** | **97.6%** |
| 95 | 81.4% | 18.6% | 98.1% |

The **oracle run** (`--oracle` puts the gold labels through the full pipeline in place of the model) does two things:
- It validates the metric code: raw F1 is 100%.
- It shows the ceiling grounding imposes: **85.6% recall at 100% precision** (F1 92.3%). The weakest field is `qty` (68% confirmed), because OCR misses lone "1"s.

The default of 85 was chosen over 75: the synthetic name test (names from other receipts) is easy at any threshold, while real VLM name errors are near-misses, which a lower threshold would let through.

### 3.6 One job at a time (Task C)
The policy is **a small bounded FIFO queue, then fast rejection** ([`docextract/service/gpu_slot.py`](docextract/service/gpu_slot.py)):
- One job runs on the GPU. Up to `MAX_QUEUE=3` more requests wait in FIFO order.
- A request that arrives while the queue is full gets **503 with `Retry-After`** straight away. The `Retry-After` value is the queue length × a moving average of job time.
- A request that waits more than `QUEUE_TIMEOUT_S` also gets 503.
- A running job that exceeds `JOB_TIMEOUT_S` is stopped by a `StoppingCriteria` deadline inside `generate` and returns **504**. The slot is released only after the GPU has really stopped.

Why this policy:
- **Why queue at all:** this is a synchronous API. A client that waits a few seconds gets its answer in one round trip, which beats retry loops. Bursts of 2–4 requests are normal.
- **Why bounded:** with a page at p95 of several seconds, an unbounded queue would turn every request's latency into minutes and hit client and proxy timeouts. That fails *later* and wastes GPU time on results nobody is waiting for. The queue size × p95 should stay below a typical 60 s client timeout.
- **Why 503 and not 429:** 429 means *you* (one client) are sending too much. Here the *server* is temporarily at capacity regardless of who is calling, which is what 503 + `Retry-After` means. Load balancers and retry libraries also treat it correctly.

Implementation guarantees:
- **Exactly one uvicorn worker:** one process, so one model copy.
- GPU work goes to a **single-thread executor**. Even if a request coroutine is cancelled (the client disconnects), a second job cannot start on the GPU while the first is still running.
- **CPU work stays outside the slot.** PDF rendering and OCR start before the request waits for the GPU and run in parallel with it. Grounding runs after the slot is released. The GPU is held only for VLM inference.
- **The model is loaded once** in the FastAPI lifespan hook. OCR is warmed up at startup too.

Test output against the mock backend (3 s per job, `MAX_QUEUE=3`, [`reports/concurrency_output_mock.txt`](reports/concurrency_output_mock.txt)):
```
req status client_s queue_s gpu_start  gpu_end  result
  2    503     0.03       -         -        -  busy (Retry-After: 41)
  4    200     3.05    0.01      0.02     3.05
  3    200     6.10    3.02      3.05     6.10
  1    200     9.15    6.07      6.10     9.15
  0    200    12.20    9.12      9.15    12.20
GPU intervals overlap: False  ->  OK: one job at a time
```
The same script against the real model is run in the notebook. Its output is saved to `reports/concurrency_output.txt`.

---

## 4. Evaluation methodology ([`docextract/metrics.py`](docextract/metrics.py), [`scripts/evaluate.py`](scripts/evaluate.py))
- **Same code path as the API:** model → parse → OCR evidence → grounding → merge → checks. Each run records both the **raw** model output (merged, nothing removed) and the **grounded** output, so every metric is reported before and after grounding.
- **Field-level P/R/F1:**
  - A correct non-null prediction is a TP.
  - A wrong prediction counts as both FP and FN.
  - A gold value that was missed is a FN.
  - Numbers must match exactly, to within 0.01. Strings are compared after normalisation.
- **Line items** are matched to gold items with the Hungarian algorithm on name similarity (plus a bonus for equal amounts), so a model that lists items in a different order is not penalised.
- **Exact match:** per field, per whole item list, and per whole document.
- **JSON validity:**
  - *strict*: the model's text is a schema-valid object as emitted;
  - *unparseable*: needed the empty fallback;
  - *at API*: always 100% by construction, and checked.
- **Hallucination rate:** the share of non-null values **not found in the document**.
  - "The document" here means CORD's **human-annotated words and boxes**, an oracle independent of the OCR used by grounding, so the measure is not circular.
  - The noise floor, i.e. gold values the oracle cannot confirm, is 1.2%.
  - Store name and date are excluded, because CORD does not annotate those regions.
- **Latency** is p50/p95 per page of VLM time and end-to-end per document. **Peak GPU memory** is `torch.cuda.max_memory_allocated`.
- **Robustness:** `--perturb rotate5|rotate90|blur|jpeg|downscale|multipage`. `multipage` splits each receipt into two pages between the items and the totals, which exercises the page merge.

---

## 5. Repository layout
```
docextract/            service package
  config.py            all settings (env vars)
  schema.py            Pydantic output schema
  prompts.py           single prompt for zero-shot / training / serving
  cord.py              CORD -> schema conversion, oracle text, abstention crop
  normalize.py         number/date/text normalisation (Indonesian formats)
  pdf.py               PDF validation, rendering, text layer
  ocr.py               RapidOCR evidence + bounding-box crop re-reader
  parsing.py           model text -> valid Extraction (repair, truncation recovery)
  grounding.py         verification + arithmetic checks
  model.py             Qwen2.5-VL backend (+ deadline stopping), mock backend
  pipeline.py          per-page inference, grounding, multi-page merge
  metrics.py           P/R/F1, exact match, hallucination, item alignment
  service/app.py       FastAPI app;  service/gpu_slot.py  one-job gate
training/              QLoRA training (dataset, collator, trainer)
scripts/               evaluate, grounding_audit, concurrency_test, make_report, make_sample_pdfs
notebooks/             Colab notebook for all GPU work
tests/                 pytest suite (runs on CPU with the mock backend)
reports/               evaluation_report.md (generated), feasibility_report.md, concurrency outputs
results/               *.summary.json from evaluation runs (per-document .jsonl are git-ignored)
samples/               3 CORD test receipts as image-only PDFs + a 2-page PDF
```

---

## 6. Limitations (honest list)
- **No GPU numbers committed yet.** The zero-shot, fine-tuned, latency, memory and real-model concurrency figures appear only after the notebook has been run (see the status note at the top).
- **Store name and date are untrained**, because CORD does not label them. The fine-tuned model is expected to return `null` for both.
- **Grounding trades recall for safety.** Even a perfect model loses ~14% of correct values, mostly `qty` of 1 that OCR does not detect. `GROUNDING_POLICY=flag` keeps such values for a human to review.
- **Grounding checks existence, not role.** A total that is really the cash-paid amount is on the page, so it passes. The arithmetic checks catch only some of these cases.
- **No orientation correction.** Pages rotated 90° or 180° are not fixed before inference (measured by the `rotate90` run).
- **Pages are processed independently.** Items split across a page break can be split or duplicated.
- **Serving uses HF `generate`, with no batching or KV-cache reuse across requests.** It is correct for one-job-at-a-time but not cost-optimal (see the feasibility report).
- **Some untested combinations.** The package versions in `requirements*.txt` are the ones the code targets; the GPU path has not been run in this repository yet, so a version mismatch on a current Colab image is possible. The fp16 ViT workaround is also untested on non-T4 GPUs, where bf16 is used instead.

## 7. Use of external tools
- No paid LLM API is used for extraction, and none was used for data preparation. The labels are CORD's own.
- Open-source components: Qwen2.5-VL (Apache-2.0), CORD v2 (CC-BY-4.0), RapidOCR/PaddleOCR models, PyMuPDF, Hugging Face transformers/peft/bitsandbytes.
- An AI coding assistant was used while writing the code. All design choices are documented above.
