# Optional dependencies must be checked before importing the HTTP application.

import asyncio
import json
import threading
from uuid import UUID

import pytest

pytest.importorskip("fastapi", exc_type=ModuleNotFoundError)
pytest.importorskip("httpx", exc_type=ModuleNotFoundError)

from fastapi.testclient import TestClient

from backend.detectors.app.config import DEFAULT_PROFILE_PATH, MAX_BODY, ROOT, load_profile
from backend.detectors.app.errors import AdapterContractError, ServiceError, TraceLimitError
from backend.detectors.app.main import create_app
from backend.detectors.app.observations import ModelObservation as M
from backend.detectors.app.observations import Window
from backend.detectors.app.pipeline import Pipeline
from backend.detectors.app.regex_detector import RegexDetector
from backend.detectors.tests.helpers import EmptyModel, model_run


@pytest.fixture
def client(profile):
    pipeline = Pipeline(EmptyModel(profile), RegexDetector(profile), profile)
    with TestClient(create_app(lambda: pipeline)) as client:
        yield client


def test_external_profile_is_used_at_startup_and_reported(tmp_path, profile, monkeypatch):
    source = (
        DEFAULT_PROFILE_PATH.read_text()
        .replace("pii-ru-en-v3", "experiment-v1")
        .replace("policy_version = '2.0.0'", "policy_version = '2.1.0'")
        .replace("emit_threshold = 0.20", "emit_threshold = 0.30")
        .replace("accept_threshold = 0.50", "accept_threshold = 0.70")
        .replace("email_ascii_v1", "experiment_email_v1")
    )
    path = tmp_path / "experiment.toml"
    path.write_text(source)
    monkeypatch.setenv("PROFILE_PATH", str(path))
    monkeypatch.setenv("NER_LABELS", "person,passport")
    monkeypatch.setattr(
        "backend.detectors.app.gliner_detector.GlinerDetector",
        lambda selected, model_path: EmptyModel(selected),
    )
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post("/v1/detect", json={"text": "a@example.org"})
    assert response.status_code == 200
    result = response.json()
    trace = result["trace"]
    assert result["has_pii"] and result["entities"][0]["type"] == "EMAIL"
    assert trace["profile"] == "experiment-v1" and trace["policy_version"] == "2.1.0"
    assert trace["model"]["emit_threshold"] == 0.30
    assert trace["model"]["accept_threshold"] == 0.70
    assert trace["model"]["labels"] == ["person", "passport"]
    assert trace["regex"]["rules"][0]["id"] == "experiment_email_v1"


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


@pytest.mark.parametrize(
    "spoken",
    [
        "восемь девять один два три четыре пять шесть семь восемь девять",
        "восемь девятьсот двенадцать триста сорок пять шестьдесят семь восемьдесят девять",
    ],
)
def test_spoken_phone_http_contract(client, spoken):
    text = f"🙂 e\u0301\r\nТелефон: {spoken}."
    response = client.post("/v1/detect", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert result["has_pii"] and len(result["entities"]) == 1
    entity = result["entities"][0]
    assert entity["type"] == "PHONE" and entity["text"] == spoken
    assert entity["start"] == text.index(spoken)
    assert text[entity["start"] : entity["end"]] == spoken
    candidate = result["trace"]["candidates"][0]
    assert candidate["source"] == "regex" and candidate["rule_id"] == "phone_words_ru_v1"
    assert candidate["decision"] == "accepted" and candidate["reason"] == "rule_passed"
    assert candidate["entity_id"] == entity["id"]
    assert entity["candidate_ids"] == [candidate["id"]]
    assert next(
        rule for rule in result["trace"]["regex"]["rules"] if rule["id"] == "phone_words_ru_v1"
    ) == {"id": "phone_words_ru_v1", "matches": 1}


def test_invalid_spoken_phone_has_only_rejected_candidate(client):
    text = "Телефон: один два три четыре пять шесть"
    response = client.post("/v1/detect", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert not result["has_pii"] and not result["entities"]
    candidate = result["trace"]["candidates"][0]
    assert candidate["rule_id"] == "phone_words_ru_v1"
    assert candidate["decision"] == "rejected" and candidate["reason"] == "invalid_format"
    assert candidate["entity_id"] is None


def test_structured_entities_and_conflict_reasons_over_http(client):
    expected = {
        "PHONE": "+7 999 123-45-67",
        "INN": "123456789047",
        "SNILS": "112-233-445 95",
        "PASSPORT": "7518 492137",
        "DRIVER_LICENSE": "77 14 987654",
        "BIRTH_CERTIFICATE": "III-АГ 112233",
        "MILITARY_ID": "МЗ 0045678",
        "OMS_POLICY": "4111 1111 1111 1111",
        "DATE_OF_BIRTH": "29.02.2000",
        "ADDRESS": "г. Казань, пр-т Мира, д. 43, кв. 154",
    }
    text = "🙂 e\u0301\r\n" + "\n".join(
        [
            "Телефон: " + expected["PHONE"],
            "ИНН: " + expected["INN"],
            "СНИЛС: " + expected["SNILS"],
            "Паспорт: " + expected["PASSPORT"],
            "Водительское удостоверение: " + expected["DRIVER_LICENSE"],
            "Свидетельство о рождении: " + expected["BIRTH_CERTIFICATE"],
            "Военный билет: " + expected["MILITARY_ID"],
            "Полис ОМС: " + expected["OMS_POLICY"],
            "Дата рождения: " + expected["DATE_OF_BIRTH"],
            "Адрес: " + expected["ADDRESS"],
        ]
    )
    response = client.post("/v1/detect", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["has_pii"]
    assert {e["type"]: e["text"] for e in body["entities"]} == expected
    assert len(body["entities"]) == len(expected)
    assert body["trace"]["profile"] == "pii-ru-en-v3"
    assert body["trace"]["policy_version"] == "2.0.0"
    candidates = {c["id"]: c for c in body["trace"]["candidates"]}
    for entity in body["entities"]:
        assert text[entity["start"] : entity["end"]] == entity["text"]
        for candidate_id in entity["candidate_ids"]:
            assert candidates[candidate_id]["entity_id"] == entity["id"]
            assert candidates[candidate_id]["decision"] == "accepted"
    card = next(c for c in candidates.values() if c["type"] == "PAYMENT_CARD")
    assert card["decision"] == "rejected" and card["reason"] == "superseded_by_specific_rule"
    assert card["entity_id"] is None


def test_optional_ip_profile_over_http(client):
    text = "192.0.2.1 и 2001:db8::1"
    assert not client.post("/v1/detect", json={"text": text}).json()["has_pii"]
    profile = load_profile(ROOT / "config/profile-with-ip.toml")
    pipeline = Pipeline(EmptyModel(profile), RegexDetector(profile), profile)
    with TestClient(create_app(lambda: pipeline)) as ip_client:
        response = ip_client.post("/v1/detect", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert [(e["type"], e["text"]) for e in body["entities"]] == [
        ("IP_ADDRESS", "192.0.2.1"),
        ("IP_ADDRESS", "2001:db8::1"),
    ]
    assert body["trace"]["profile"] == "pii-ru-en-v3-ip"


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
    assert [r["matches"] for r in result["trace"]["regex"]["rules"]] == [0] * 15
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
