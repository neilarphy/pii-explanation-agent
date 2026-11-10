"""FIFO admission; the detached owner keeps its slot until real CPU/GPU completion."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from .errors import ServiceError


class Admission:
    def __init__(self, capacity=8, timeout=5.0):
        self.capacity = capacity
        self.timeout = timeout
        self.busy = False
        self.waiters = deque()

    async def acquire(self):
        if not self.busy:
            self.busy = True
            return
        if len(self.waiters) >= self.capacity:
            raise ServiceError("BUSY", 429, "admission", ["validation"])
        future = asyncio.get_running_loop().create_future()
        self.waiters.append(future)
        try:
            async with asyncio.timeout(self.timeout):
                await asyncio.shield(future)
        except BaseException as exc:
            if future in self.waiters:
                self.waiters.remove(future)
            elif future.done() and not future.cancelled():
                # Cancellation can race with a just-transferred slot.
                self.release()
            if isinstance(exc, TimeoutError):
                raise ServiceError("BUSY", 429, "admission", ["validation"]) from None
            raise

    def release(self):
        while self.waiters:
            future = self.waiters.popleft()
            if not future.done():
                future.set_result(None)
                return
        self.busy = False


class Runtime:
    def __init__(self, pipeline, executor: ThreadPoolExecutor, fatal, capacity=8, timeout=5.0):
        self.pipeline = pipeline
        self.executor = executor
        self.fatal = fatal
        self.ready = True
        self.admission = Admission(capacity, timeout)
        self.tasks = set()

    def check_ready(self):
        if not self.ready:
            raise ServiceError("NOT_READY", 503, "admission", ["validation"])

    async def run(self, text, request_id, started):
        self.check_ready()
        queued = perf_counter()
        await self.admission.acquire()
        queue_ms = (perf_counter() - queued) * 1000
        # No await between ownership transfer and creating the detached task.
        owner = asyncio.create_task(self._own(text, request_id, started, queue_ms))
        self.tasks.add(owner)
        owner.add_done_callback(self._finished)
        return await asyncio.shield(owner)

    def _finished(self, task):
        self.tasks.discard(task)
        if not task.cancelled():
            task.exception()  # Consume failures even when the client has disconnected.

    async def _own(self, text, request_id, started, queue_ms):
        try:
            self.check_ready()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor,
                self.pipeline.run,
                text,
                request_id,
                started,
                queue_ms,
            )
        except ServiceError as exc:
            if exc.code == "DETECTOR_UNAVAILABLE":
                self.ready = False
                asyncio.get_running_loop().call_later(1.0, self.fatal)
            raise
        finally:
            self.admission.release()

    async def close(self):
        self.ready = False
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
