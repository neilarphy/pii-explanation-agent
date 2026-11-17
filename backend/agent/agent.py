"""Агент: находка → промпт → LLM → проверки кодом → AgentAnswer."""

import json

from backend.agent.cases import Case, build_cases
from backend.agent.checks import format_check, resolve_citations
from backend.agent.llm import Complete
from backend.agent.prompt import build_messages
from backend.schemas import AgentAnswer, DetectionLog, Session, Verdict


def parse_json(raw: str) -> dict:
    """Достаёт JSON-объект из ответа, даже если модель обернула его в ```json."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"В ответе LLM нет JSON: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])


def explain_case(
    case: Case, session: Session, log: DetectionLog, complete: Complete
) -> AgentAnswer | None:
    """Ответ по одной находке; None — отклонённый кандидат оказался не ПДн."""
    data = parse_json(complete(build_messages(case, session, log)))
    verdict = data.get("verdict")
    if verdict == "no_issue":
        if case.kind == "rejected":
            return None
        verdict = Verdict.UNCERTAIN  # для принятой сущности no_issue не бывает

    citations, dropped = resolve_citations(data.get("citations") or [], session)
    evidence = [f"candidate={c.id}" for c in case.candidates] + list(data.get("evidence") or [])
    note = data.get("note") or ""

    if dropped:
        evidence.append("отброшены недословные цитаты: " + "; ".join(dropped))
    if verdict == Verdict.MISSED_PII and format_check(case.type, case.text) is False:
        verdict = Verdict.UNCERTAIN
        note = (
            f"{note} Пропуск не подтверждён: формат {case.type} не прошёл проверку кодом.".strip()
        )

    return AgentAnswer(
        entity_id=case.entity_id,
        verdict=verdict,
        confidence=data.get("confidence", 0.5),
        explanation=data.get("explanation") or "",
        citations=citations,
        evidence=evidence,
        note=note,
        recommendation=data.get("recommendation"),
    )


def explain_session(
    session: Session, log: DetectionLog, complete: Complete
) -> list[tuple[Case, AgentAnswer | None]]:
    return [(case, explain_case(case, session, log, complete)) for case in build_cases(log)]
