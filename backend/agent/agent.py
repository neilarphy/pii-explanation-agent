"""Агент: находка → промпт → LLM → проверки кодом → AgentAnswer."""

import json

from backend.agent.cases import Case, build_cases
from backend.agent.checks import format_check, identify, resolve_citations
from backend.agent.llm import Complete
from backend.agent.prompt import build_messages
from backend.schemas import AgentAnswer, Category, DetectionLog, Session, Verdict

# Строки evidence, которые дописывает код, а не модель: их проверяет код, судья их не видит.
CODE_EVIDENCE = (
    "найдено при чтении диалога",
    "candidate=",
    "найдено кодом",
    "отброшены недословные цитаты",
    "код подтвердил",
    "сбой агента",
    "каскад:",
)


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
    found = identify(case.text)
    return finalize(case, session, data, found[0].description if found else None)


def finalize(
    case: Case, session: Session, data: dict, confirmed: str | None, guard: bool = True
) -> AgentAnswer | None:
    """Ответ модели → AgentAnswer: вердикт по категории, цитаты по тексту, пропуск — по коду.

    confirmed — что код подтвердил о значении (сам фрагмент, целое значение вокруг него или
    склейка по репликам); None — ничего. Пропуск ПДн без такого подтверждения и с неверным
    форматом фрагмента понижается до uncertain. guard=False — без понижения: вердикт модели
    как есть (голая модель в сравнении режимов).
    """
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
    if case.kind == "llm":
        evidence.insert(0, "найдено при чтении диалога, детектор и код не отметили")
    note = data.get("note") or ""

    if dropped:
        evidence.append("отброшены недословные цитаты: " + "; ".join(dropped))
    # Формат проверяем только для ПДн и только если код не узнал значение сам: тип у
    # отклонённого кандидата бывает неверным (токен с типом PAYMENT_CARD), а у CVC и
    # токенов (sensitive) формата карты или документа нет вовсе.
    if (
        guard
        and verdict == Verdict.MISSED_PII
        and category == Category.PII
        and confirmed is None
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


def fail_open(case: Case, reason: str) -> AgentAnswer:
    """Сбой LLM: находка не теряется, а уходит человеку как возможные ПДн (чекпоинт 1).

    Решение детектора не меняется: принятое остаётся замаскированным, отклонённое — на проверку.
    """
    detector = (
        "детектор фрагмент принял, маскирование остаётся"
        if case.kind == "entity"
        else "детектор фрагмент не замаскировал, проверьте вручную"
    )
    return AgentAnswer(
        entity_id=case.entity_id,
        verdict=Verdict.UNCERTAIN,
        category=Category.PII,
        confidence=0.0,
        explanation=f"Агент не смог разобрать находку ({reason}). Это могут быть ПДн: {detector}.",
        citations=[],
        evidence=[f"candidate={c.id}" for c in case.candidates] + ["сбой агента, fail-open"],
        note="Нужен человек: ответа агента нет.",
    )


def failure_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:120]}"


def explain_session(
    session: Session, log: DetectionLog, complete: Complete
) -> list[tuple[Case, AgentAnswer | None]]:
    results = []
    for case in build_cases(log, session):
        try:
            answer = explain_case(case, session, log, complete)
        except Exception as exc:  # сбой LLM или ответа не должен терять находку
            answer = fail_open(case, failure_reason(exc))
        results.append((case, answer))
    return results
