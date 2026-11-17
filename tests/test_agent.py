"""Агент без сети: вместо LLM — заглушка, отвечающая эталоном из docs/examples."""

import json
import re
from pathlib import Path

import pytest

from backend.agent.agent import explain_case, explain_session, parse_json
from backend.agent.cases import build_cases
from backend.agent.checks import format_check, resolve_citations
from backend.schemas import DetectionLog, Session, Verdict

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


@pytest.fixture(scope="module")
def session() -> Session:
    return Session.model_validate_json((EXAMPLES / "session.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def log() -> DetectionLog:
    return DetectionLog.model_validate_json((EXAMPLES / "detection_log.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def reference() -> dict[str | None, dict]:
    answers = json.loads((EXAMPLES / "agent_answers.json").read_text("utf-8"))
    return {a["entity_id"]: a for a in answers}


def fake_llm(reference: dict[str | None, dict], **override):
    """Отвечает эталоном для находки из промпта; цитаты — без офсетов, как от настоящей LLM."""

    def complete(messages: list[dict[str, str]]) -> str:
        case_id = re.search(r"case_id: (\S+)\)", messages[-1]["content"]).group(1)
        key = case_id if case_id in reference else None
        answer = dict(reference[key])
        answer["citations"] = [
            {"turn": c["turn"], "quote": c["quote"]} for c in answer["citations"]
        ]
        answer.pop("entity_id")
        answer.update(override)
        return "```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```"

    return complete


def test_cases_are_entities_plus_uncovered_rejected(log):
    cases = build_cases(log)
    assert [(c.id, c.kind) for c in cases] == [
        ("s-001-t0-e1", "entity"),
        ("s-001-t2-e1", "entity"),
        ("s-001-t2-e2", "entity"),
        ("s-001-t4-c1", "rejected"),
    ]
    assert [c.id for c in cases[1].candidates] == ["s-001-t2-c1", "s-001-t2-c2"]


def test_end_to_end_matches_reference(session, log, reference):
    results = explain_session(session, log, fake_llm(reference))
    for case, answer in results:
        want = reference[case.entity_id]
        assert answer.verdict == want["verdict"]
        assert [c.model_dump() for c in answer.citations] == want["citations"]


def test_prompt_contains_dialog_and_detector_facts(session, log, reference):
    seen = []

    def complete(messages):
        seen.append(messages[-1]["content"])
        return fake_llm(reference)(messages)

    explain_session(session, log, complete)
    card, *_, phone = seen
    assert "номер заказа 4276 1234 5678 9014" in card
    assert "rule_id=card_pan_v1" in card
    assert "проверки кодом: номер карты: Луна сходится" in card
    assert "детектор отклонил фрагмент" in phone
    assert "score=0.31" in phone


def test_made_up_quote_is_dropped(session):
    citations, dropped = resolve_citations(
        [{"turn": 2, "quote": "anna.petrova@mail.ru"}, {"turn": 2, "quote": "Анна Петрова"}],
        session,
    )
    assert [(c.start, c.end) for c in citations] == [(0, 20)]
    assert dropped == ["реплика 2: «Анна Петрова»"]


def test_rejected_non_sensitive_is_skipped(session, log, reference):
    case = build_cases(log)[-1]
    assert (
        explain_case(case, session, log, fake_llm(reference, verdict="no_issue", category="none"))
        is None
    )


def test_missed_pii_needs_format_check(session, log, reference):
    case = build_cases(log)[-1]
    bad = case.__class__(**{**case.__dict__, "text": "8 9 1 2"})
    answer = explain_case(bad, session, log, fake_llm(reference))
    assert answer.verdict == Verdict.UNCERTAIN
    assert "не подтверждён" in answer.note


@pytest.mark.parametrize(
    ("pii_type", "text", "ok"),
    [
        ("PHONE", "8 9 1 2 3 4 5 6 7 8 9 доб. 12", True),
        ("PHONE", "8 9 1 2", False),
        ("PAYMENT_CARD", "4276 1234 5678 9014", True),
        ("PAYMENT_CARD", "4276 1234 5678 9015", False),
        ("EMAIL", "anna.petrova@mail.ru", True),
        ("PERSON", "Соколову", None),
    ],
)
def test_format_check(pii_type, text, ok):
    assert format_check(pii_type, text) is ok


def test_parse_json_strips_fences():
    assert parse_json('```json\n{"verdict": "true_pii"}\n```') == {"verdict": "true_pii"}
