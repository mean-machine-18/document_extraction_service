"""FastAPI service: POST /extract takes a PDF and returns grounded JSON synchronously.

Request lifecycle
  1. size / type checks             (CPU, no slot needed)
  2. open + render PDF pages        (CPU threadpool)
  3. evidence text: text layer/OCR  (CPU threadpool, started in parallel with step 4)
  4. wait for the GPU slot -> VLM   (single GPU thread, one job at a time)
  5. grounding, merge, checks       (CPU)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ..config import get_settings
from ..model import load_backend
from ..ocr import page_evidence, warmup as ocr_warmup
from ..pdf import InputError, Page, load_pdf
from ..pipeline import InferenceTimeout, ground_and_merge, run_model
from .gpu_slot import GpuSlot, QueueFull, QueueTimeout

log = logging.getLogger("docextract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()
    app.state.backend = load_backend(settings)  # loaded exactly once per process
    app.state.slot = GpuSlot(settings.max_queue, settings.queue_timeout_s)
    await run_in_threadpool(ocr_warmup)
    log.info("model %s ready in %.1fs", app.state.backend.name, time.monotonic() - t0)
    yield


app = FastAPI(title="Receipt extraction service", version="1.0.0", lifespan=lifespan)


def _error(status: int, code: str, message: str, headers=None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}}, headers=headers)


@app.exception_handler(InputError)
async def _input_error(_: Request, e: InputError):
    return _error(e.status, e.code, e.message)


@app.exception_handler(Exception)
async def _unexpected(_: Request, e: Exception):
    log.exception("unhandled error")
    return _error(500, "internal_error", f"{type(e).__name__}: {e}")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request):
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        return _error(503, "not_ready", "model not loaded")
    return {"status": "ready", "backend": backend.name}


@app.get("/status")
async def status(request: Request):
    return request.app.state.slot.status()


async def _read_upload(request: Request, file: UploadFile) -> bytes:
    limit = settings.max_file_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit + 64 * 1024:
        raise InputError("file_too_large", f"Upload exceeds {settings.max_file_mb:g} MB.", 413)
    chunks, size = [], 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise InputError("file_too_large", f"Upload exceeds {settings.max_file_mb:g} MB.", 413)
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/extract")
async def extract(request: Request, file: UploadFile = File(...), debug: bool = False):
    rid = uuid.uuid4().hex[:8]
    t_start = time.monotonic()
    data = await _read_upload(request, file)
    pages: List[Page] = await run_in_threadpool(load_pdf, data, settings.max_pages, settings.render_dpi)

    # OCR is CPU work: start it now so it overlaps with waiting for / using the GPU.
    evidence_task = asyncio.ensure_future(
        asyncio.gather(*(run_in_threadpool(page_evidence, p) for p in pages)))

    slot: GpuSlot = request.app.state.slot
    backend = request.app.state.backend
    try:
        async with slot.acquire() as waited_s:
            gpu_start_wall = time.time()
            deadline = time.monotonic() + settings.job_timeout_s
            outputs = await slot.run(run_model, backend, pages, deadline)
            gpu_end_wall = time.time()
    except QueueFull:
        evidence_task.cancel()
        log.info("[%s] rejected: queue full", rid)
        return _error(503, "busy", "GPU busy and wait queue full; retry later.",
                      {"Retry-After": str(slot.estimated_wait_s)})
    except QueueTimeout:
        evidence_task.cancel()
        log.info("[%s] rejected: queue wait > %ss", rid, settings.queue_timeout_s)
        return _error(503, "queue_timeout", f"Waited more than {settings.queue_timeout_s:g}s for the GPU.",
                      {"Retry-After": str(slot.estimated_wait_s)})
    except InferenceTimeout as e:
        evidence_task.cancel()
        log.warning("[%s] inference timeout", rid)
        return _error(504, "inference_timeout", str(e))

    evidence = await evidence_task
    meta = {
        "request_id": rid,
        "backend": backend.name,
        "pages": len(pages),
        "evidence_source": [ev.source for ev in evidence],
        "queue_wait_ms": round(waited_s * 1000),
        "inference_ms": round((gpu_end_wall - gpu_start_wall) * 1000),
        "gpu_started_at": gpu_start_wall,
        "gpu_finished_at": gpu_end_wall,
    }
    if debug:
        meta["raw_model_output"] = [o.generation.text for o in outputs]
        meta["evidence_text"] = [[line.text for line in ev.lines] for ev in evidence]
    response = await run_in_threadpool(ground_and_merge, outputs, evidence, settings, meta)
    response.meta["total_ms"] = round((time.monotonic() - t_start) * 1000)
    log.info("[%s] ok pages=%d wait=%dms gpu=%dms total=%dms", rid, len(pages), meta["queue_wait_ms"],
             meta["inference_ms"], response.meta["total_ms"])
    return JSONResponse(response.to_json_dict())
