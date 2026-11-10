import asyncio
import json
import threading
from uuid import UUID

import pytest
from app.config import MAX_BODY, load_profile
from app.errors import AdapterContractError, ServiceError, TraceLimitError
from app.main import create_app
from app.observations import ModelObservation as M
from app.observations import Window
from app.pipeline import Pipeline
from app.regex_detector import RegexDetector
from conftest import EmptyModel, model_run
from fastapi.testclient import TestClient


@pytest.fixture
def client(profile):
    pipeline = Pipeline(EmptyModel(profile), RegexDetector(profile), profile)
    with TestClient(create_app(lambda: pipeline)) as client:
        yield client


def assert_error(response, status, code, stage="validation", completed=()):
    assert response.status_code == status, response.text
    value = response.json()
    UUID(value["request_id"])
    assert value == {
        "schema_version": "1.0",
        "request_id": response.headers["X-Request-ID"],
        "error": {"code": code, "failed_stage": stage, "completed_stages": list(completed)},
    }
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": None}, {"text": 1}, {"text": True}, {"text": []}, {"text": "a", "extra": 1}],
)
def test_invalid_request(client, payload):
    assert_error(client.post("/v1/detect", json=payload), 422, "INVALID_REQUEST")


@pytest.mark.parametrize(
    "body", [b"not json", b"{", b'{"text":"\\ud800"}', b'{"text":"\\udfff"}', b"\xff"]
)
def test_invalid_json_and_surrogates(client, body):
    assert_error(
        client.post("/v1/detect", content=body, headers={"Content-Type": "application/json"}),
        422,
        "INVALID_REQUEST",
    )


@pytest.mark.parametrize("media", ["", "text/plain", "application/xml", "application/problem+json"])
def test_content_type(client, media):
    assert_error(
        client.post("/v1/detect", content="{}", headers={"Content-Type": media}),
        415,
        "UNSUPPORTED_MEDIA_TYPE",
    )


def test_content_type_charset(client):
    response = client.post(
        "/v1/detect",
        content='{"text":"a@example.org"}',
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    assert response.json()["has_pii"]


@pytest.mark.parametrize("text", ["", " \r\n\t", "no matches"])
def test_blank_and_zero(client, text):
    response = client.post("/v1/detect", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert result["entities"] == [] and result["has_pii"] is False
    assert result["trace"]["candidates"] == []
    for detector in ["model", "regex"]:
        assert result["trace"][detector]["status"] == (
            "skipped_blank" if not text.strip() else "ok"
        )
    if not text.strip():
        assert result["trace"]["model"]["windows"] == []
    assert [r["matches"] for r in result["trace"]["regex"]["rules"]] == [0, 0, 0, 0]
    assert all(v >= 0 for v in result["trace"]["timings_ms"].values())


def test_no_text_normalization_or_request_id_trust(client):
    text = "🙂 Анна:\r\n a@example.org e\u0301\ue000"
    response = client.post("/v1/detect", json={"text": text}, headers={"X-Request-ID": "untrusted"})
    result = response.json()
    assert result["trace"]["text_length"] == len(text)
    assert result["request_id"] == response.headers["X-Request-ID"] != "untrusted"
    assert response.headers["Cache-Control"] == "no-store"
    for entity in result["entities"]:
        assert text[entity["start"] : entity["end"]] == entity["text"]
    assert "text" not in result


def test_size_limits_inclusive(client):
    assert client.post("/v1/detect", json={"text": "x" * 32768}).status_code == 200
    assert_error(client.post("/v1/detect", json={"text": "x" * 32769}), 413, "INPUT_TOO_LARGE")
    body = b'{"text":""}'
    exact = body + b" " * (MAX_BODY - len(body))
    assert (
        client.post(
            "/v1/detect", content=exact, headers={"Content-Type": "application/json"}
        ).status_code
        == 200
    )
    assert_error(
        client.post(
            "/v1/detect", content=exact + b" ", headers={"Content-Type": "application/json"}
        ),
        413,
        "INPUT_TOO_LARGE",
    )


def test_actual_chunks_stop_on_first_excess():
    async def scenario():
        app = create_app()
        consumed = 0
        sent = []

        async def receive():
            nonlocal consumed
            consumed += 1
            return {"type": "http.request", "body": b"x" * (MAX_BODY // 2 + 1), "more_body": True}

        async def send(event):
            sent.append(event)

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
        await app(scope, receive, send)
        assert consumed == 2
        assert sent[0]["status"] == 413
        assert json.loads(sent[1]["body"])["error"]["code"] == "INPUT_TOO_LARGE"

    asyncio.run(scenario())


def test_not_ready_even_blank():
    with TestClient(create_app(lambda: None)) as client:
        client.app.state.runtime.ready = False
        assert_error(
            client.post("/v1/detect", json={"text": ""}),
            503,
            "NOT_READY",
            "admission",
            ["validation"],
        )
        assert_error(client.get("/health/ready"), 503, "NOT_READY", "startup")
        assert client.get("/health/live").json() == {"status": "live"}


@pytest.mark.parametrize(
    ("stage", "error", "status", "code"),
    [
        ("model", AdapterContractError(), 500, "ADAPTER_CONTRACT_ERROR"),
        ("model", TraceLimitError("model"), 422, "TRACE_LIMIT_EXCEEDED"),
        ("model", ValueError("PRIVATE_SYNTHETIC_MARKER"), 500, "INTERNAL_ERROR"),
        ("regex", ValueError("PRIVATE_SYNTHETIC_MARKER"), 500, "INTERNAL_ERROR"),
    ],
)
def test_failure_has_no_partial_result(profile, stage, error, status, code, capfd):
    class Failing:
        def detect(self, text):
            raise error

    pipeline = Pipeline(
        Failing() if stage == "model" else EmptyModel(profile),
        Failing() if stage == "regex" else RegexDetector(profile),
        profile,
    )
    with TestClient(create_app(lambda: pipeline)) as client:
        response = client.post("/v1/detect", json={"text": "PRIVATE_SYNTHETIC_MARKER"})
    completed = ["validation", "admission"] + (["model"] if stage == "regex" else [])
    assert_error(response, status, code, stage, completed)
    captured = capfd.readouterr()
    assert "PRIVATE_SYNTHETIC_MARKER" not in captured.out + captured.err + response.text


def test_fatal_removes_readiness_and_requests_restart(profile):
    class Failing:
        def detect(self, text):
            raise ServiceError("DETECTOR_UNAVAILABLE", 503, "model")

    event = threading.Event()
    with TestClient(
        create_app(lambda: Pipeline(Failing(), RegexDetector(profile), profile), fatal=event.set)
    ) as client:
        assert_error(
            client.post("/v1/detect", json={"text": "a"}),
            503,
            "DETECTOR_UNAVAILABLE",
            "model",
            ["validation", "admission"],
        )
        assert client.get("/health/ready").status_code == 503
        assert event.wait(2)


def test_combined_trace_limit(profile):
    class Many:
        def detect(self, text):
            return model_run(
                profile,
                [M("person", "PERSON", "w1", 0, 0, 1, 0.9)] * 4096,
                [Window("w1", 0, len(text), 4096)],
            )

    with TestClient(
        create_app(lambda: Pipeline(Many(), RegexDetector(profile), profile))
    ) as client:
        assert_error(
            client.post("/v1/detect", json={"text": "a@b.co"}),
            422,
            "TRACE_LIMIT_EXCEEDED",
            "regex",
            ["validation", "admission", "model"],
        )


def test_labels_selected_even_on_blank(monkeypatch):
    monkeypatch.setenv("NER_LABELS", "person, passport")
    profile = load_profile()
    with TestClient(
        create_app(lambda: Pipeline(EmptyModel(profile), RegexDetector(profile), profile))
    ) as client:
        for text in ["", "a@example.org"]:
            result = client.post("/v1/detect", json={"text": text}).json()
            assert result["trace"]["model"]["labels"] == ["person", "passport"]
            assert [r["id"] for r in result["trace"]["regex"]["rules"]] == [
                r.id for r in profile.rules
            ]
            assert result["has_pii"] == bool(text)
