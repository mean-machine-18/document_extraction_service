"""Fire N simultaneous POST /extract requests and show what the server does with them.

    python -m scripts.concurrency_test --url http://localhost:8000 --pdf samples/cord_test_000.pdf \
        --n 5 --save reports/concurrency_output.txt

Prints one row per request (status, client latency, queue wait, GPU interval), a timeline
of GPU usage, and checks that no two GPU intervals overlap.
"""
from __future__ import annotations

import argparse
import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


def send(i: int, url: str, pdf: bytes, barrier: threading.Barrier, timeout: float) -> dict:
    barrier.wait()  # release all requests at the same instant
    t0 = time.time()
    try:
        r = httpx.post(f"{url}/extract", files={"file": (f"req{i}.pdf", pdf, "application/pdf")}, timeout=timeout)
        body = r.json()
        return {"i": i, "status": r.status_code, "sent": t0, "done": time.time(), "body": body,
                "retry_after": r.headers.get("retry-after")}
    except Exception as e:  # client-side timeout etc.
        return {"i": i, "status": f"ERR {type(e).__name__}", "sent": t0, "done": time.time(), "body": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    out = io.StringIO()

    def log(s=""):
        print(s)
        out.write(s + "\n")

    pdf = open(args.pdf, "rb").read()
    server = httpx.get(f"{args.url}/readyz", timeout=10).json()
    status = httpx.get(f"{args.url}/status", timeout=10).json()
    log(f"server: {server}  gate: {status}")
    log(f"sending {args.n} parallel requests with {args.pdf} ({len(pdf) / 1024:.0f} KB) at "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    barrier = threading.Barrier(args.n)
    with ThreadPoolExecutor(args.n) as pool:
        results = list(pool.map(lambda i: send(i, args.url, pdf, barrier, args.timeout), range(args.n)))

    t_ref = min(r["sent"] for r in results)
    log(f"{'req':>3} {'status':>6} {'client_s':>8} {'queue_s':>7} {'gpu_start':>9} {'gpu_end':>8}  result")
    spans = []
    for r in sorted(results, key=lambda r: r["done"]):
        meta = (r["body"] or {}).get("_meta") or {}
        if r["status"] == 200:
            gs, ge = meta["gpu_started_at"] - t_ref, meta["gpu_finished_at"] - t_ref
            spans.append((gs, ge, r["i"]))
            res = f"total={r['body'].get('total')} items={len(r['body'].get('line_items', []))}"
            log(f"{r['i']:>3} {r['status']:>6} {r['done'] - r['sent']:>8.2f} {meta['queue_wait_ms'] / 1000:>7.2f} "
                f"{gs:>9.2f} {ge:>8.2f}  {res}")
        else:
            err = (r["body"] or {}).get("error", {})
            log(f"{r['i']:>3} {r['status']!s:>6} {r['done'] - r['sent']:>8.2f} {'-':>7} {'-':>9} {'-':>8}  "
                f"{err.get('code')} (Retry-After: {r.get('retry_after')})")

    if spans:
        end = max(e for _, e, _ in spans)
        width = 60
        log("\nGPU timeline (each # = %.2fs)" % (end / width))
        for s, e, i in sorted(spans):
            a, b = int(s / end * width), max(int(e / end * width), int(s / end * width) + 1)
            log(f"req {i}: |{' ' * a}{'#' * (b - a)}{' ' * (width - b)}|")
    spans.sort()
    overlap = any(s2 < e1 - 1e-3 for (_, e1, _), (s2, _, _) in zip(spans, spans[1:]))
    codes = [r["status"] for r in results]
    log(f"\nsummary: {codes.count(200)} x 200, {codes.count(503)} x 503, {codes.count(504)} x 504, "
        f"other: {[c for c in codes if c not in (200, 503, 504)]}")
    log(f"GPU intervals overlap: {overlap}  ->  {'FAIL' if overlap else 'OK: one job at a time'}")

    if args.save:
        with open(args.save, "w") as f:
            f.write(out.getvalue())
    sys.exit(1 if overlap else 0)


if __name__ == "__main__":
    main()
