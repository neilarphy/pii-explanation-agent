"""Судья: разбор ответа, сообщение судье, согласие с человеком."""

import json
from pathlib import Path

from backend.agent.cases import build_cases
from backend.schemas import AgentAnswer, DetectionLog, Session
from evaluation.judge import cohen_kappa, judge_messages, parse_verdicts

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


def test_parse_verdicts_handles_na_and_strings():
    v = parse_verdicts(
        {
            "faithful": "1",
            "citations_ok": 0,
            "recommendation_ok": "n/a",
            "verdict_ok": True,
            "unsupported_claims": ["score 0.9"],
            "comment": "ok",
        }
    )
    assert (v.faithful, v.citations_ok, v.recommendation_ok, v.verdict_ok) == (1, 0, None, 1)
    assert v.unsupported_claims == ("score 0.9",)


def test_judge_sees_what_agent_saw_plus_answer():
    session = Session.model_validate_json((EXAMPLES / "session.json").read_text("utf-8"))
    log = DetectionLog.model_validate_json((EXAMPLES / "detection_log.json").read_text("utf-8"))
    answer = AgentAnswer.model_validate(
        json.loads((EXAMPLES / "agent_answers.json").read_text("utf-8"))[0]
    )
    case = build_cases(log, session)[0]
    system, user = judge_messages(case, session, log, answer)
    assert "faithful" in system["content"]
    assert "номер заказа 4276 1234 5678 9014" in user["content"]
    assert '"verdict": "false_positive"' in user["content"]
    assert "Инструкция, по которой работал агент" in user["content"]  # правила агента видны


def test_judge_does_not_see_evidence_added_by_code():
    session = Session.model_validate_json((EXAMPLES / "session.json").read_text("utf-8"))
    log = DetectionLog.model_validate_json((EXAMPLES / "detection_log.json").read_text("utf-8"))
    answer = AgentAnswer.model_validate(
        json.loads((EXAMPLES / "agent_answers.json").read_text("utf-8"))[0]
    )
    answer = answer.model_copy(
        update={"evidence": ["candidate=x-c9", "код подтвердил: телефон", "контекст: заказ"]}
    )
    _, user = judge_messages(build_cases(log, session)[0], session, log, answer)
    assert "candidate=x-c9" not in user["content"] and "код подтвердил" not in user["content"]
    assert "контекст: заказ" in user["content"]


def test_cohen_kappa():
    assert cohen_kappa([(1, 1), (0, 0), (1, 1), (0, 0)]) == 1.0
    assert cohen_kappa([(1, 0), (0, 1), (1, 0), (0, 1)]) == -1.0
