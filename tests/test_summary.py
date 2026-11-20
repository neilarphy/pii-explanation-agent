"""Сводка по сессиям: группировка правок по адресату и задача для Claude Code."""

from backend.agent.cases import Case
from backend.agent.summary import Finding, build_summary
from backend.schemas import AgentAnswer, Category, Recommendation, Summary, Verdict


def finding(session, case_id, text, verdict, rec=None, entity_id="e1", kind="entity"):
    case = Case(
        id=case_id,
        kind=kind,
        turn=0,
        start=0,
        end=len(text),
        text=text,
        type="PAYMENT_CARD",
        entity_id=entity_id,
        candidates=(),
    )
    answer = AgentAnswer(
        entity_id=entity_id,
        verdict=verdict,
        category=Category.NONE if verdict == Verdict.FALSE_POSITIVE else Category.PII,
        confidence=0.9,
        explanation="…",
        citations=[],
        evidence=[],
        note="Нужен человек" if verdict == Verdict.UNCERTAIN else "",
        recommendation=rec,
    )
    return Finding(session, case, answer)


ALLOW = Recommendation(
    kind="allowlist",
    target="card_pan_v1",
    change="не считать картой число после «заказ»",
    rationale="Луна сходится у части номеров заказов",
)


def test_errors_are_grouped_by_addressee_and_turned_into_a_task():
    findings = [
        finding("s1", "s1-t0-e1", "4276123456789014", Verdict.FALSE_POSITIVE, ALLOW),
        finding("s2", "s2-t0-e1", "5469123456789012", Verdict.FALSE_POSITIVE, ALLOW),
        finding("s3", "s3-t0-e1", "4111111111111111", Verdict.TRUE_PII),  # детектор прав
        finding("s4", "s4-t0-e1", "1234", Verdict.UNCERTAIN, entity_id=None, kind="rejected"),
    ]
    summary = build_summary(findings)
    Summary.model_validate(summary.model_dump())
    assert summary.session_ids == ["s1", "s2", "s3", "s4"]
    (issue,) = summary.issues
    assert issue.title == "card_pan_v1: ложные срабатывания PAYMENT_CARD" and issue.count == 2
    assert summary.recommendations[0].target == "card_pan_v1"
    assert "Тест детектора: «4276123456789014» не даёт PAYMENT_CARD" in summary.markdown
    assert "## Нужен человек (1)" in summary.markdown
