"""Агент: находка → промпт → LLM → проверки кодом → AgentAnswer."""

import json

from backend.agent.cases import Case, build_cases
from backend.agent.checks import format_check, identify, resolve_citations
from backend.agent.llm import Complete
from backend.agent.prompt import build_messages
from backend.schemas import AgentAnswer, Category, DetectionLog, Session, Verdict


def parse_json(raw: str) -> dict:
    """Достаёт JSON-объект из ответа, даже если модель обернула его в ```json."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"В ответе LLM нет JSON: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])


def _consistent_verdict(case: Case, verdict: str, category: Category) -> str:
    """Вердикт следует из категории: маскировать нужно pii и sensitive.

    LLM иногда пишет «missed_pii» про реквизиты юрлица или «true_pii» про номер заказа —
    правим по категории; «uncertain» оставляем как есть.
    """
    if verdict == Verdict.UNCERTAIN:
        return verdict
    mask = category != Category.NONE
    if case.kind == "entity":
        return Verdict.TRUE_PII if mask else Verdict.FALSE_POSITIVE
    return Verdict.MISSED_PII if mask else "no_issue"


def explain_case(
    case: Case, session: Session, log: DetectionLog, complete: Complete
) -> AgentAnswer | None:
    """Ответ по одной находке; None — непринятый фрагмент оказался не тем, что надо маскировать."""
    data = parse_json(complete(build_messages(case, session, log)))
    try:
        category = Category(data.get("category"))
    except ValueError:
        category = (
            Category.NONE if data.get("verdict") in ("false_positive", "no_issue") else Category.PII
        )
    verdict = _consistent_verdict(case, data.get("verdict"), category)
    if verdict == "no_issue":
        return None

    citations, dropped = resolve_citations(data.get("citations") or [], session)
    evidence = [f"candidate={c.id}" for c in case.candidates] + list(data.get("evidence") or [])
    if case.kind == "scan":
        evidence.insert(0, "найдено кодом, детектор не отметил")
    note = data.get("note") or ""

    if dropped:
        evidence.append("отброшены недословные цитаты: " + "; ".join(dropped))
    # Формат проверяем только для ПДн и только если код не узнал значение сам: тип у
    # отклонённого кандидата бывает неверным (токен с типом PAYMENT_CARD), а у CVC и
    # токенов (sensitive) формата карты или документа нет вовсе.
    if (
        verdict == Verdict.MISSED_PII
        and category == Category.PII
        and not identify(case.text)
        and format_check(case.type, case.text) is False
    ):
        verdict = Verdict.UNCERTAIN
        note = (
            f"{note} Пропуск не подтверждён: формат {case.type} не прошёл проверку кодом.".strip()
        )

    return AgentAnswer(
        entity_id=case.entity_id,
        verdict=verdict,
        category=category,
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
    cases = build_cases(log, session)
    return [(case, explain_case(case, session, log, complete)) for case in cases]
