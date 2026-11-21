import json
from dataclasses import replace
from pathlib import Path

import pytest
from app.errors import TraceLimitError
from app.observations import ModelObservation as M
from app.observations import RegexMetadata, RegexObservation, RegexRun, RuleCount, Window
from app.pipeline import build_response, merge_observations
from app.regex_detector import RegexDetector
from app.schemas import Timings
from conftest import model_run


def test_contract_fixture(profile):
    text = "🙂 Анна: a@example.org"
    model = model_run(
        profile,
        [
            M("first_name", "PERSON", "w1", 0, 2, 6, 0.91),
            M("email", "EMAIL", "w1", 0, 8, 21, 0.84),
        ],
        [Window("w1", 0, 21, 2)],
    )
    result = build_response(
        text,
        model,
        RegexDetector(profile).detect(text),
        profile,
        "f3c006d9-2bc7-43cc-b1e7-e1e46c6d60da",
        Timings(queue=0.1, model=30.5, regex=2.2, merge=0.3, total=33.4),
    )
    expected = json.loads((Path(__file__).parent / "fixtures/contract.json").read_text())
    assert result.model_dump(mode="json") == expected
    assert text[8:21] == result.entities[1].text == "a@example.org"


def test_threshold_rejected_source_does_not_veto(profile):
    text = "4111 1111 1111 1112"
    model = model_run(
        profile,
        [
            M("card_number", "PAYMENT_CARD", "w1", 0, 0, len(text), score)
            for score in [0.49, 0.5, 0.4999999999]
        ],
    )
    entities, candidates = merge_observations(text, model, RegexDetector(profile).detect(text), 0.5)
    assert len(entities) == 1
    assert [c.score for c in candidates[:3]] == [0.5, 0.4999999999, 0.49]
    assert [c.decision for c in candidates] == ["accepted", "rejected", "rejected", "rejected"]
    assert candidates[-1].reason == "checksum_failed"
    assert all(c.entity_id is None for c in candidates[1:])
    assert entities[0].candidate_ids == ["c1"]


def test_overlap_and_window_duplicates(profile):
    text = "John Smith!"
    model = model_run(
        profile,
        [
            M("person", "PERSON", "w1", 0, 0, 4, 0.9),
            M("person", "PERSON", "w1", 0, 0, 11, 0.9),
            M("address", "ADDRESS", "w1", 0, 0, 4, 0.9),
            M("first_name", "PERSON", "w2", 1, 0, 4, 0.8),
        ],
        [Window("w1", 0, 11, 3), Window("w2", 0, 11, 1)],
    )
    entities, candidates = merge_observations(text, model, RegexDetector(profile).detect(text), 0.5)
    assert [(e.start, e.end, e.type) for e in entities] == [
        (0, 4, "ADDRESS"),
        (0, 4, "PERSON"),
        (0, 11, "PERSON"),
    ]
    assert entities[1].candidate_ids == ["c2", "c3"]
    for entity in entities:
        for cid in entity.candidate_ids:
            assert next(c for c in candidates if c.id == cid).entity_id == entity.id


def test_only_rejected(profile):
    text = "4111 1111 1111 1112"
    model = model_run(profile, windows=[Window("w1", 0, len(text))])
    result = build_response(
        text,
        model,
        RegexDetector(profile).detect(text),
        profile,
        "00000000-0000-0000-0000-000000000001",
        Timings(queue=0, model=0, regex=0, merge=0, total=0),
    )
    assert not result.has_pii and not result.entities and len(result.trace.candidates) == 1


def test_custom_label(profile):
    profile = replace(profile, labels=("employee_id",))
    model = model_run(
        profile, [M("employee_id", profile.public_type("employee_id"), "w1", 0, 0, 3, 0.8)]
    )
    entities, candidates = merge_observations(
        "E12", model, RegexDetector(profile).detect("E12"), 0.5
    )
    assert entities[0].type == "EMPLOYEE_ID" and candidates[0].raw_label == "employee_id"


def test_merge_limit_combined(profile):
    model = model_run(profile, [M("person", "PERSON", "w1", 0, 0, 1, 0.9)] * 4096)
    regex = RegexRun((), RegexMetadata("ok", ()))
    entities, candidates = merge_observations("a", model, regex, 0.5)
    assert len(candidates) == 4096 and len(entities) == 1
    regex = RegexRun(
        (RegexObservation("r", "PERSON", 0, 0, 1, "rule_passed"),),
        RegexMetadata("ok", (RuleCount("r", 1),)),
    )
    with pytest.raises(TraceLimitError):
        merge_observations("a", model, regex, 0.5)
