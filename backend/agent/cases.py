"""Что агент разбирает в одной сессии: сущности детектора и отклонённых кандидатов."""

from dataclasses import dataclass
from typing import Literal

from backend.schemas import Candidate, DetectionLog


@dataclass(frozen=True)
class Case:
    """Одна находка для разбора.

    kind="entity" — сущность, которую детектор принял (вопрос: верно ли).
    kind="rejected" — кандидат, которого детектор отклонил (вопрос: не пропуск ли).
    """

    id: str
    kind: Literal["entity", "rejected"]
    turn: int
    start: int
    end: int
    text: str
    type: str
    entity_id: str | None
    candidates: tuple[Candidate, ...]


def _overlaps(a_turn: int, a_start: int, a_end: int, b_turn: int, b_start: int, b_end: int) -> bool:
    return a_turn == b_turn and a_start < b_end and b_start < a_end


def build_cases(log: DetectionLog) -> list[Case]:
    """Сущности + отклонённые кандидаты, которые не пересекаются ни с одной сущностью."""
    by_id = {c.id: c for c in log.candidates}
    cases = [
        Case(
            id=e.id,
            kind="entity",
            turn=e.turn,
            start=e.start,
            end=e.end,
            text=e.text,
            type=e.type,
            entity_id=e.id,
            candidates=tuple(by_id[cid] for cid in e.candidate_ids),
        )
        for e in log.entities
    ]

    rejected: dict[tuple[int, int, int], list[Candidate]] = {}
    for c in log.candidates:
        if c.decision != "rejected":
            continue
        if any(_overlaps(c.turn, c.start, c.end, e.turn, e.start, e.end) for e in log.entities):
            continue
        rejected.setdefault((c.turn, c.start, c.end), []).append(c)

    for group in rejected.values():
        first = group[0]
        cases.append(
            Case(
                id=first.id,
                kind="rejected",
                turn=first.turn,
                start=first.start,
                end=first.end,
                text=first.text,
                type=first.type,
                entity_id=None,
                candidates=tuple(group),
            )
        )
    return cases
