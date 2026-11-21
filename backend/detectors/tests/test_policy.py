# The response contract requires the detector's optional CPU dependencies.

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("pydantic", exc_type=ModuleNotFoundError)

from backend.detectors.app.errors import TraceLimitError
from backend.detectors.app.observations import ModelObservation as M
from backend.detectors.app.observations import (
    RegexMetadata,
    RegexObservation,
    RegexRun,
    RuleCount,
    Window,
)
from backend.detectors.app.pipeline import build_response, merge_observations
from backend.detectors.app.regex_detector import RegexDetector
from backend.detectors.app.schemas import Timings
from backend.detectors.tests.helpers import model_run


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


def policy_response(profile, text, observations=()):
    return build_response(
        text,
        model_run(profile, observations, [Window("w1", 0, len(text), len(observations))]),
        RegexDetector(profile).detect(text),
        profile,
        "00000000-0000-0000-0000-000000000001",
        Timings(queue=0, model=0, regex=0, merge=0, total=0),
    )


def test_oms_supersedes_card_and_account_but_preserves_evidence(profile):
    value = "4111 1111 1111 1111"
    text = f"Полис ОМС: {value}"
    start = text.index(value)
    result = policy_response(
        profile,
        text,
        [
            M("card_number", "PAYMENT_CARD", "w1", 0, start, len(text), 0.99),
            M("bank_account", "BANK_ACCOUNT", "w1", 0, start, len(text), 0.90),
        ],
    )
    assert [e.type for e in result.entities] == ["OMS_POLICY"]
    assert result.entities[0].text == value
    rejected = [c for c in result.trace.candidates if c.decision == "rejected"]
    assert len(rejected) == 3  # Both model candidates and the Luhn-valid regex candidate.
    assert all(c.reason == "superseded_by_specific_rule" and c.entity_id is None for c in rejected)
    assert (
        next(c.score for c in rejected if c.source == "model" and c.type == "PAYMENT_CARD") == 0.99
    )
    assert next(r.matches for r in result.trace.regex.rules if r.id == "card_pan_v1") == 1


def test_contextual_rights_override_model_passport(profile):
    value = "77 14 номер 987654"
    text = f"Водительское удостоверение серии {value}"
    observation = M("passport", "PASSPORT", "w1", 0, text.index(value), len(text), 0.95)
    result = policy_response(profile, text, [observation])
    assert [(e.type, e.text) for e in result.entities] == [("DRIVER_LICENSE", value)]
    model = next(c for c in result.trace.candidates if c.source == "model")
    assert model.decision == "rejected" and model.reason == "superseded_by_specific_rule"


def test_specific_policy_can_be_disabled(profile):
    from dataclasses import replace

    value = "77 14 номер 987654"
    text = f"Водительское удостоверение серии {value}"
    observation = M("passport", "PASSPORT", "w1", 0, text.index(value), len(text), 0.95)
    result = policy_response(replace(profile, resolve_conflicts=False), text, [observation])
    assert {e.type for e in result.entities} == {"DRIVER_LICENSE", "PASSPORT"}


def test_military_number_does_not_remain_a_person_or_generic_id(profile):
    value = "МЗ, номер 0045678"
    text = f"Анна: военный билет, серия {value}"
    observations = [
        M("person", "PERSON", "w1", 0, 0, 4, 0.95),
        M("person", "PERSON", "w1", 0, text.index(value), len(text), 0.95),
        M("national_id", "NATIONAL_ID", "w1", 0, text.index("0045678"), len(text), 0.99),
    ]
    result = policy_response(profile, text, observations)
    assert {(e.type, e.text) for e in result.entities} == {
        ("PERSON", "Анна"),
        ("MILITARY_ID", value),
    }
    for entity in result.entities:
        for identifier in entity.candidate_ids:
            candidate = next(c for c in result.trace.candidates if c.id == identifier)
            assert candidate.decision == "accepted" and candidate.entity_id == entity.id


def test_rejected_snils_does_not_veto_model_id(profile):
    text = "СНИЛС: 112-233-445 94"
    value = "112-233-445 94"
    result = policy_response(
        profile,
        text,
        [
            M("national_id", "NATIONAL_ID", "w1", 0, text.index(value), len(text), 0.9),
        ],
    )
    assert [e.type for e in result.entities] == ["NATIONAL_ID"]
    snils = next(c for c in result.trace.candidates if c.type == "SNILS")
    assert snils.decision == "rejected" and snils.reason == "checksum_failed"


@pytest.mark.parametrize(
    "label", ["Номер заказа", "Номер договора", "БИК", "КПП", "Расчётный счёт"]
)
def test_non_payment_context_rejects_model_and_regex_cards(profile, label):
    value = "4111 1111 1111 1111"
    text = f"{label}: {value}"
    result = policy_response(
        profile,
        text,
        [
            M("card_number", "PAYMENT_CARD", "w1", 0, text.index(value), len(text), 0.99),
        ],
    )
    assert not result.has_pii and not result.entities
    assert all(
        c.reason == "context_mismatch" and c.entity_id is None for c in result.trace.candidates
    )


def test_normal_card_is_kept_after_an_order_in_the_same_document(profile):
    text = "Номер заказа: 42. Карта: 4111 1111 1111 1111"
    result = policy_response(profile, text)
    assert [e.type for e in result.entities] == ["PAYMENT_CARD"]


def test_specific_checksum_refines_model_national_id(profile):
    value = "123456789047"
    text = f"ИНН: {value}"
    result = policy_response(
        profile,
        text,
        [
            M("national_id", "NATIONAL_ID", "w1", 0, text.index(value), len(text), 0.99),
        ],
    )
    assert [(e.type, e.text) for e in result.entities] == [("INN", value)]


def test_explicit_plus_keeps_phone_despite_incidental_snils_checksum(profile):
    text = "+11223344595"
    result = policy_response(profile, text)
    assert [(e.type, e.text) for e in result.entities] == [("PHONE", text)]
    assert (
        next(c.reason for c in result.trace.candidates if c.type == "SNILS")
        == "superseded_by_specific_rule"
    )


def test_equal_strength_contextual_conflict_is_preserved(profile):
    from dataclasses import replace

    oms = next(r for r in profile.rules if r.id == "oms_policy_ru_v1")
    conflicting = replace(oms, id="ambiguous_card", type="PAYMENT_CARD")
    result = policy_response(replace(profile, rules=(oms, conflicting)), "ОМС: 4111 1111 1111 1111")
    assert {e.type for e in result.entities} == {"OMS_POLICY", "PAYMENT_CARD"}
