import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import httpx
import pytest
from app.admission import Admission, Runtime
from app.errors import ServiceError
from app.main import create_app


async def eventually(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


def test_queue_capacity_timeout_cancellation_and_fifo():
    async def scenario():
        gate = Admission(timeout=0.05)
        await gate.acquire()
        tasks = [asyncio.create_task(gate.acquire()) for _ in range(8)]
        await eventually(lambda: len(gate.waiters) == 8)
        with pytest.raises(ServiceError, match="BUSY"):
            await gate.acquire()
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        assert len(gate.waiters) == 7
        gate.release()
        await tasks[1]
        assert gate.busy
        gate.release()
        await tasks[2]
        for task in tasks[3:]:
            with pytest.raises(ServiceError, match="BUSY"):
                await task
        gate.release()
        assert not gate.busy and not gate.waiters
        await gate.acquire()
        gate.release()

    asyncio.run(scenario())


def test_cancelled_running_request_keeps_slot_and_health_responsive():
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        running = 0
        max_running = 0
        calls = []

        class BlockingPipeline:
            def run(self, text, *args):
                nonlocal running, max_running
                running += 1
                max_running = max(max_running, running)
                calls.append(text)
                entered.set()
                assert release.wait(3)
                running -= 1
                return None

        executor = ThreadPoolExecutor(max_workers=1)
        runtime = Runtime(BlockingPipeline(), executor, lambda: None)
        try:
            first = asyncio.create_task(runtime.run("first", "id", perf_counter()))
            await eventually(entered.is_set)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert runtime.admission.busy
            second = asyncio.create_task(runtime.run("second", "id", perf_counter()))
            await eventually(lambda: len(runtime.admission.waiters) == 1)
            app = create_app()
            app.state.runtime = runtime
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                async with asyncio.timeout(0.2):
                    assert (await client.get("/health/live")).status_code == 200
                    assert (await client.get("/health/ready")).status_code == 200
            assert calls == ["first"]
            release.set()
            await second
            await runtime.close()
            assert max_running == 1 and calls == ["first", "second"]
            assert not runtime.admission.busy
        finally:
            release.set()
            executor.shutdown()

    asyncio.run(scenario())


def test_errors_release_actual_slot():
    async def scenario():
        class FailingPipeline:
            def run(self, *args):
                raise ServiceError("INTERNAL_ERROR", 500, "regex")

        executor = ThreadPoolExecutor(max_workers=1)
        runtime = Runtime(FailingPipeline(), executor, lambda: None)
        try:
            for _ in range(3):
                with pytest.raises(ServiceError):
                    await runtime.run("x", "id", perf_counter())
                assert not runtime.admission.busy
        finally:
            await runtime.close()
            executor.shutdown()

    asyncio.run(scenario())


def test_http_disconnect_removes_waiter_but_keeps_running_owner(profile):
    from app.observations import ModelRun
    from app.pipeline import Pipeline
    from app.regex_detector import RegexDetector
    from conftest import EmptyModel

    async def scenario():
        entered, release = threading.Event(), threading.Event()

        class BlockingModel(EmptyModel):
            def detect(self, text) -> ModelRun:
                entered.set()
                assert release.wait(3)
                return super().detect(text)

        app = create_app(lambda: Pipeline(BlockingModel(profile), RegexDetector(profile), profile))
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime

            def request():
                inbox = asyncio.Queue()
                inbox.put_nowait(
                    {"type": "http.request", "body": b'{"text":"synthetic"}', "more_body": False}
                )
                messages = []

                async def send(event):
                    messages.append(event)

                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "method": "POST",
                    "path": "/v1/detect",
                    "headers": [(b"content-type", b"application/json")],
                    "query_string": b"",
                    "server": ("test", 80),
                    "client": ("test", 1),
                    "scheme": "http",
                    "http_version": "1.1",
                }
                return asyncio.create_task(app(scope, inbox.get, send)), inbox, messages

            first, first_in, _ = request()
            try:
                await eventually(entered.is_set)
                second, second_in, _ = request()
                await eventually(lambda: len(runtime.admission.waiters) == 1)
                second_in.put_nowait({"type": "http.disconnect"})
                with pytest.raises(asyncio.CancelledError):
                    await second
                assert not runtime.admission.waiters
                first_in.put_nowait({"type": "http.disconnect"})
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert runtime.admission.busy and runtime.tasks
                release.set()
                await eventually(lambda: not runtime.admission.busy)
            finally:
                release.set()

    asyncio.run(scenario())


def test_http_busy_retry_header(profile):
    from app.observations import ModelRun
    from app.pipeline import Pipeline
    from app.regex_detector import RegexDetector
    from conftest import EmptyModel

    async def scenario():
        entered, release = threading.Event(), threading.Event()

        class BlockingModel(EmptyModel):
            def detect(self, text) -> ModelRun:
                entered.set()
                assert release.wait(3)
                return super().detect(text)

        app = create_app(lambda: Pipeline(BlockingModel(profile), RegexDetector(profile), profile))
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(client.post("/v1/detect", json={"text": "first"}))
                await eventually(entered.is_set)
                waiters = [
                    asyncio.create_task(client.post("/v1/detect", json={"text": "next"}))
                    for _ in range(8)
                ]
                try:
                    await eventually(lambda: len(runtime.admission.waiters) == 8)
                    response = await client.post("/v1/detect", json={"text": "excess"})
                    assert response.status_code == 429
                    assert response.headers["Retry-After"] == "1"
                    assert response.json()["error"] == {
                        "code": "BUSY",
                        "failed_stage": "admission",
                        "completed_stages": ["validation"],
                    }
                    waiters[0].cancel()
                    await asyncio.gather(waiters[0], return_exceptions=True)
                    assert len(runtime.admission.waiters) == 7
                    release.set()
                    results = await asyncio.gather(first, *waiters[1:])
                    assert all(r.status_code == 200 for r in results)
                finally:
                    release.set()
                    for task in (first, *waiters):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(first, *waiters, return_exceptions=True)

    asyncio.run(scenario())
