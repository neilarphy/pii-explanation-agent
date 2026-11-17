"""Проверки кодом (контрольные суммы, поиск по тексту), склейка находок, согласованность вердикта."""

import json

import pytest

from backend.agent.agent import explain_case
from backend.agent.cases import build_cases
from backend.agent.checks import identify, inn_ok, ogrn_ok, scan, snils_ok
from backend.schemas import (
    Candidate,
    Category,
    DetectionLog,
    Entity,
    Message,
    ModelConfig,
    Session,
    Verdict,
)


@pytest.mark.parametrize(
    ("check", "value", "ok"),
    [
        (inn_ok, "7707083893", True),  # ИНН организации
        (inn_ok, "7707083894", False),
        (inn_ok, "500100732259", True),  # ИНН физлица
        (inn_ok, "500100732258", False),
        (snils_ok, "11223344595", True),
        (snils_ok, "11223344596", False),
        (ogrn_ok, "1027700132195", True),  # ОГРН
        (ogrn_ok, "1027700132196", False),
        (ogrn_ok, "304500116000157", True),  # ОГРНИП
    ],
)
def test_checksums(check, value, ok):
    assert check(value) is ok


@pytest.mark.parametrize(
    ("text", "label", "category"),
    [
        ("4276 1234 5678 9014", "CARD", Category.PII),
        ("500100732259", "INN_PERSON", Category.PII),
        ("7707083893", "INN_COMPANY", Category.NONE),
        ("112-233-445 95", "SNILS", Category.PII),
        ("1027700132195", "OGRN", Category.NONE),
        ("4510 298713", "PASSPORT", Category.PII),
        ("ab12cd34ef56gh78ij90kl", "TOKEN", Category.SENSITIVE),
    ],
)
def test_identify(text, label, category):
    first = identify(text)[0]
    assert (first.label, first.category) == (label, category)


def test_scan_finds_numbers_that_pass_checks_only():
    text = "СНИЛС 112-233-445 95, заказ 123456789, сумма 25000, ИНН 500100732259"
    found = [(text[s:e], c.label) for s, e, c in scan(text)]
    assert found == [("112-233-445 95", "SNILS"), ("500100732259", "INN_PERSON")]


def _log(entities, candidates) -> DetectionLog:
    return DetectionLog(
        session_id="s",
        profile="pii-ru-en-v1",
        policy_version="1.0.0",
        model=ModelConfig(id="m", emit_threshold=0.2, accept_threshold=0.5, labels=["person"]),
        rules=[],
        entities=entities,
        candidates=candidates,
    )


def _cand(cid, start, end, text, eid, decision="accepted", score=0.9, label="person"):
    return Candidate(
        id=cid,
        turn=0,
        start=start,
        end=end,
        text=text,
        type="PERSON",
        source="model",
        decision=decision,
        entity_id=eid,
        score=score,
        raw_label=label,
        reason="model_threshold_passed" if decision == "accepted" else "below_model_threshold",
    )


def test_overlapping_entities_merge_and_scan_adds_unseen_numbers():
    text = "Анна Иванова, мой СНИЛС 112-233-445 95"
    session = Session(id="s", messages=[Message(turn=0, direction="in", text=text)])
    log = _log(
        entities=[
            Entity(
                id="s-t0-e1",
                turn=0,
                start=0,
                end=4,
                text="Анна",
                type="PERSON",
                candidate_ids=["s-t0-c1"],
            ),
            Entity(
                id="s-t0-e2",
                turn=0,
                start=0,
                end=12,
                text="Анна Иванова",
                type="PERSON",
                candidate_ids=["s-t0-c2"],
            ),
            Entity(
                id="s-t0-e3",
                turn=0,
                start=5,
                end=12,
                text="Иванова",
                type="PERSON",
                candidate_ids=["s-t0-c3"],
            ),
        ],
        candidates=[
            _cand("s-t0-c1", 0, 4, "Анна", "s-t0-e1", label="first_name"),
            _cand("s-t0-c2", 0, 12, "Анна Иванова", "s-t0-e2"),
            _cand("s-t0-c3", 5, 12, "Иванова", "s-t0-e3", label="last_name"),
            _cand("s-t0-c4", 18, 23, "СНИЛС", None, decision="rejected", score=0.3),
        ],
    )
    cases = build_cases(log, session)
    assert [(c.kind, c.text) for c in cases] == [
        ("entity", "Анна Иванова"),
        ("rejected", "СНИЛС"),
        ("scan", "112-233-445 95"),
    ]
    assert len(cases[0].candidates) == 3
    assert cases[2].type == "NATIONAL_ID"


def _answer(verdict, category):
    def complete(_messages):
        return json.dumps(
            {
                "verdict": verdict,
                "category": category,
                "confidence": 0.9,
                "explanation": "x",
                "citations": [],
                "evidence": [],
                "note": "",
                "recommendation": None,
            }
        )

    return complete


def test_verdict_follows_category():
    text = "ИНН организации 7707083893"
    session = Session(id="s", messages=[Message(turn=0, direction="in", text=text)])
    log = _log([], [])
    (case,) = build_cases(log, session)
    assert case.kind == "scan"
    # LLM сказала «пропуск», но категория none — значит маскировать не нужно
    assert explain_case(case, session, log, _answer("missed_pii", "none")) is None
    answer = explain_case(case, session, log, _answer("no_issue", "sensitive"))
    assert (answer.verdict, answer.category) == (Verdict.MISSED_PII, Category.SENSITIVE)


def test_sensitive_miss_is_not_downgraded_by_wrong_detector_type():
    text = "код с обратной стороны карты 673"
    session = Session(id="s", messages=[Message(turn=0, direction="in", text=text)])
    cand = Candidate(
        id="s-t0-c1",
        turn=0,
        start=29,
        end=32,
        text="673",
        type="PASSPORT",
        source="model",
        decision="rejected",
        reason="below_model_threshold",
        entity_id=None,
        score=0.3,
        raw_label="passport",
    )
    log = _log([], [cand])
    (case,) = build_cases(log, session)
    answer = explain_case(case, session, log, _answer("missed_pii", "sensitive"))
    assert answer.verdict == Verdict.MISSED_PII
