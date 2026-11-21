"""Cached inference must preserve scores, rejected candidates and scoring semantics."""

from dataclasses import replace

import pytest

pytest.importorskip("pydantic", exc_type=ModuleNotFoundError)

from backend.detectors.app.artifacts import MODEL_REVISION
from backend.detectors.app.observations import ModelObservation, Window
from backend.detectors.app.pipeline import build_response
from backend.detectors.app.regex_detector import RegexDetector
from backend.detectors.app.schemas import Timings
from backend.detectors.scripts.pii_bench import summarize
from backend.detectors.scripts.replay_regex import replay_record
from backend.detectors.tests.helpers import model_run


def cached_record(profile):
    text = "Паспортные данные: 7518 492137"
    start = text.index("7518")
    model = model_run(
        profile,
        [ModelObservation("national_id", "NATIONAL_ID", "w1", 0, start, len(text), 0.91)],
        [Window("w1", 0, len(text), 1)],
    )
    model = replace(model, metadata=replace(model.metadata, revision=MODEL_REVISION))
    body = build_response(
        text,
        model,
        RegexDetector(profile).detect(text),
        profile,
        "00000000-0000-0000-0000-000000000001",
        Timings(queue=0, model=1, regex=1, merge=1, total=3),
    ).model_dump(mode="json")
    return {
        "id": "cache-test",
        "domain": "synthetic",
        "split": "test",
        "text": text,
        "gold": [],
        "response": body,
        "client_ms": 5,
    }


def test_replay_reproduces_entities_and_all_candidates(profile):
    original = cached_record(profile)
    replayed = replay_record(original, profile, RegexDetector(profile))
    assert replayed["response"]["entities"] == original["response"]["entities"]
    assert (
        replayed["response"]["trace"]["candidates"] == original["response"]["trace"]["candidates"]
    )
    assert replayed["client_ms"] is None
    assert summarize([replayed])["latency_ms"] is None


def test_replay_reconsiders_model_candidate_rejected_by_original_policy(profile):
    original = cached_record(profile)
    model = next(c for c in original["response"]["trace"]["candidates"] if c["source"] == "model")
    assert model["decision"] == "rejected"
    selected = replace(profile, resolve_conflicts=False)
    replayed = replay_record(original, selected, RegexDetector(selected))
    model = next(c for c in replayed["response"]["trace"]["candidates"] if c["source"] == "model")
    assert model["decision"] == "accepted" and model["score"] == 0.91


@pytest.mark.parametrize(
    "change",
    [
        {"emit_threshold": 0.3},
        {"accept_threshold": 0.6},
        {"labels": ("person",)},
        {"public_types": {"national_id": "OTHER"}},
    ],
)
def test_replay_rejects_changed_model_configuration(profile, change):
    original = cached_record(profile)
    selected = replace(profile, **change)
    with pytest.raises(ValueError, match="differs"):
        replay_record(original, selected, RegexDetector(selected))


def test_replay_cannot_fill_missing_inference(profile):
    original = cached_record(profile)
    original["response"] = None
    with pytest.raises(ValueError, match="failed request"):
        replay_record(original, profile, RegexDetector(profile))
