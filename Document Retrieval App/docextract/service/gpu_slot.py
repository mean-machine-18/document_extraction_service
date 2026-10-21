"""One-job-at-a-time GPU gate with a small bounded wait queue.

Policy (see README "Concurrency policy"):
  * 1 job runs on the GPU. Up to `max_queue` more requests may wait for it, FIFO.
  * A request that arrives when the queue is full gets 503 + Retry-After immediately.
  * A request that waits longer than `queue_timeout_s` gets 503 + Retry-After.
  * The running job has its own deadline (job timeout -> 504), enforced inside
    generation so the GPU is actually freed, not just abandoned.

GPU work is also submitted to a single-thread executor, so even if a request coroutine is
cancelled (client disconnect), a second job can never start on the GPU while the first is
still running in its thread.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Callable


class QueueFull(Exception):
    pass


class QueueTimeout(Exception):
    pass


class GpuSlot:
    def __init__(self, max_queue: int, queue_timeout_s: float):
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self._sem = asyncio.Semaphore(1)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self.running = 0
        self.waiting = 0
        self.completed = 0
        self._avg_job_s = 10.0  # updated with an EMA of observed job durations

    @property
    def estimated_wait_s(self) -> int:
        return int((self.waiting + self.running) * self._avg_job_s) + 1

    @asynccontextmanager
    async def acquire(self):
        if self.waiting >= self.max_queue and self.running:
            raise QueueFull()
        self.waiting += 1
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.queue_timeout_s)
        except asyncio.TimeoutError:
            raise QueueTimeout()
        finally:
            self.waiting -= 1
        self.running = 1
        started = time.monotonic()
        try:
            yield started - t0
        finally:
            self._avg_job_s = 0.7 * self._avg_job_s + 0.3 * (time.monotonic() - started)
            self.running = 0
            self.completed += 1
            self._sem.release()

    async def run(self, fn: Callable[..., Any], *args) -> Any:
        """Run blocking GPU work on the dedicated GPU thread (call inside `acquire`)."""
        loop = asyncio.get_running_loop()
        return await asyncio.shield(loop.run_in_executor(self._executor, fn, *args))

    def status(self) -> dict:
        return {"running": self.running, "waiting": self.waiting, "max_queue": self.max_queue,
                "completed": self.completed, "avg_job_s": round(self._avg_job_s, 2)}
