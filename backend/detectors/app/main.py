"""ASGI limits, uniform errors, responsive health and one model worker."""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from .admission import Runtime
from .config import MAX_BODY, MAX_TEXT, load_profile
from .errors import ConfigurationError, ServiceError
from .pipeline import Pipeline
from .regex_detector import RegexDetector
from .schemas import DetectRequest, DetectResponse, ErrorDetail, ErrorResponse

logger = logging.getLogger("pii")


class SafeLogFilter(logging.Filter):
    def filter(self, record):
        return record.name == "pii" and not record.exc_info and not record.stack_info


def configure_logging():
    handler = logging.StreamHandler()
    handler.addFilter(SafeLogFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    # Libraries may install their own handlers; force them through the safe filter.
    for name, item in logging.root.manager.loggerDict.items():
        if isinstance(item, logging.Logger) and name != "pii":
            item.handlers.clear()
            item.propagate = True
    logging.captureWarnings(True)


def error_response(request_id, error):
    body = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=error.code, failed_stage=error.stage, completed_stages=error.completed
        ),
    )
    headers = {"X-Request-ID": str(request_id), "Cache-Control": "no-store"}
    if error.status == 429:
        headers["Retry-After"] = "1"
    logger.info("request_id=%s stage=%s code=%s", request_id, error.stage, error.code)
    return JSONResponse(body.model_dump(mode="json"), status_code=error.status, headers=headers)


class HTTPBoundary:
    """Read bounded ASGI chunks; don't rely on Content-Length or buffer past the cap."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = str(uuid4())
        scope.setdefault("state", {}).update(request_id=request_id, started=perf_counter())
        response_started = False

        async def safe_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in (b"x-request-id", b"cache-control")
                ]
                headers += [(b"x-request-id", request_id.encode()), (b"cache-control", b"no-store")]
                message = {**message, "headers": headers}
            await send(message)

        try:
            if scope["path"] == "/v1/detect":
                media = (
                    dict(scope["headers"])
                    .get(b"content-type", b"")
                    .split(b";", 1)[0]
                    .strip()
                    .lower()
                )
                if media != b"application/json":
                    raise ServiceError("UNSUPPORTED_MEDIA_TYPE", 415, "validation")
            chunks, size = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > MAX_BODY:
                    raise ServiceError("INPUT_TOO_LARGE", 413, "validation")
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay, safe_send)
        except ServiceError as exc:
            if not response_started:
                await error_response(request_id, exc)(scope, receive, safe_send)
        except Exception:
            if not response_started:
                error = ServiceError(
                    "INTERNAL_ERROR",
                    500,
                    scope["state"].get("stage", "validation"),
                    scope["state"].get("completed", ()),
                )
                await error_response(request_id, error)(scope, receive, safe_send)


def initialize():
    # NER_LABELS and all regex compilation fail before loading weights.
    profile = load_profile()
    regex = RegexDetector(profile)
    from .gliner_detector import GlinerDetector

    model = GlinerDetector(profile, Path(os.environ.get("MODEL_PATH", "/models")))
    model.detect("Anna Smith, Анна Иванова: a@example.org")
    probe = regex.detect("a@example.org a@example.org")
    if [(o.start, o.end) for o in probe.observations if o.rule_id == "email_ascii_v1"] != [
        (0, 13),
        (14, 27),
    ]:
        raise ConfigurationError("regex: warmup failed")
    return Pipeline(model, regex, profile)


def create_app(pipeline_factory=None, fatal=None):
    factory = pipeline_factory or initialize
    fatal = fatal or (lambda: os._exit(1))

    @asynccontextmanager
    async def lifespan(application):
        configure_logging()
        application.state.runtime = None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pii-worker")
        try:
            try:
                pipeline = await asyncio.get_running_loop().run_in_executor(executor, factory)
            except Exception as exc:
                # No library traceback, configuration values or model data reach stderr.
                code = str(exc) if isinstance(exc, ConfigurationError) else "initialization failed"
                logger.error("stage=startup code=STARTUP_FAILED configuration=%s", code)
                raise RuntimeError("PII startup failed") from None
            # HF/torch can register handlers during their first import in the worker.
            configure_logging()
            runtime = Runtime(pipeline, executor, fatal)
            application.state.runtime = runtime
            yield
        finally:
            if application.state.runtime is not None:
                await application.state.runtime.close()
            executor.shutdown(wait=True, cancel_futures=False)

    application = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    application.add_middleware(HTTPBoundary)

    @application.exception_handler(ServiceError)
    async def service_error(request, exc):
        return error_response(request.state.request_id, exc)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return error_response(
            request.state.request_id, ServiceError("INVALID_REQUEST", 422, "validation")
        )

    @application.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error_response(
            request.state.request_id,
            ServiceError(
                "INVALID_REQUEST", 422 if exc.status_code == 400 else exc.status_code, "validation"
            ),
        )

    @application.get("/health/live")
    async def live():
        return {"status": "live"}

    @application.get("/health/ready")
    async def ready(request: Request):
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None or not runtime.ready:
            raise ServiceError("NOT_READY", 503, "startup")
        return {"status": "ready"}

    @application.post("/v1/detect", response_model=DetectResponse)
    async def detect(payload: DetectRequest, request: Request):
        if len(payload.text) > MAX_TEXT:
            raise ServiceError("INPUT_TOO_LARGE", 413, "validation")
        request.state.stage = "admission"
        request.state.completed = ["validation"]
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise ServiceError("NOT_READY", 503, "admission", ["validation"])
        job = asyncio.create_task(
            runtime.run(payload.text, request.state.request_id, request.state.started)
        )

        async def disconnected():
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    return

        watcher = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait((job, watcher), return_when=asyncio.FIRST_COMPLETED)
            if job not in done:
                job.cancel()
                with suppress(asyncio.CancelledError):
                    await job
                # The peer is gone: finish without inventing an HTTP error outside v1.
                raise asyncio.CancelledError
            response = await job
            request.state.stage = "merge"
            request.state.completed = ["validation", "admission", "model", "regex"]
            # Finish total on the event loop immediately before serialization.
            response.trace.timings_ms.total = (perf_counter() - request.state.started) * 1000
            logger.info(
                "request_id=%s stage=merge code=OK count=%d time_ms=%.3f",
                request.state.request_id,
                len(response.entities),
                response.trace.timings_ms.total,
            )
            return JSONResponse(response.model_dump(mode="json"))
        finally:
            watcher.cancel()
            if not job.done():
                job.cancel()
            await asyncio.gather(watcher, job, return_exceptions=True)

    return application


app = create_app()
