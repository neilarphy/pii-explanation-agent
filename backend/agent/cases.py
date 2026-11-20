"""Что агент разбирает в одной сессии.

- entity — сущность, которую детектор принял (вопрос: верно ли);
- rejected — кандидат, которого детектор отклонил (вопрос: не пропуск ли);
- scan — фрагмент, который детектор не заметил вовсе, а код нашёл по контрольной
  сумме или формату (вопрос: не пропуск ли);
- llm — фрагмент, который никто не заметил, а модель назвала, прочитав диалог целиком
  (`backend/agent/reader.py`, вопрос: не пропуск ли).

Пересекающиеся сущности одного типа («Анна», «Анна Иванова», «Иванова») склеиваются
в одну находку: агент разбирает имя один раз, а не три.
"""

from dataclasses import dataclass
from typing import Literal

from backend.agent.checks import scan, scan_wide
from backend.schemas import Candidate, DetectionLog, Entity, Session


@dataclass(frozen=True)
class Case:
    id: str
    kind: Literal["entity", "rejected", "scan", "llm"]
    turn: int
    start: int
    end: int
    text: str
    type: str
    entity_id: str | None
    candidates: tuple[Candidate, ...]


def _overlaps(a, b) -> bool:
    return a.turn == b.turn and a.start < b.end and b.start < a.end


def _clusters(items: list, same) -> list[list]:
    """Группы попарно связанных элементов: связь — пересечение спанов и same(a, b)."""
    groups: list[list] = []
    for item in items:
        linked = [g for g in groups if any(_overlaps(item, x) and same(item, x) for x in g)]
        merged = [item] + [x for g in linked for x in g]
        groups = [g for g in groups if all(g is not x for x in linked)] + [merged]
    return groups


def _entity_case(group: list[Entity], by_id: dict[str, Candidate]) -> Case:
    main = max(group, key=lambda e: (e.end - e.start, -e.start))
    candidate_ids = dict.fromkeys(cid for e in group for cid in e.candidate_ids)
    return Case(
        id=main.id,
        kind="entity",
        turn=main.turn,
        start=main.start,
        end=main.end,
        text=main.text,
        type=main.type,
        entity_id=main.id,
        candidates=tuple(by_id[cid] for cid in candidate_ids),
    )


def _rejected_case(group: list[Candidate]) -> Case:
    main = max(group, key=lambda c: (c.score or 0, c.end - c.start))
    return Case(
        id=main.id,
        kind="rejected",
        turn=main.turn,
        start=main.start,
        end=main.end,
        text=main.text,
        type=main.type,
        entity_id=None,
        candidates=tuple(sorted(group, key=lambda c: c.id)),
    )


def build_cases(log: DetectionLog, session: Session | None = None, wide: bool = True) -> list[Case]:
    """Склеенные сущности, отклонённые кандидаты вне сущностей и (если дана сессия) находки кода.

    wide — ещё и поиск кодом шире (`scan_wide`): номер по слову-документу рядом, числа словами,
    адрес без «д.». Включён с итерации 8: на redmadrobot находит 48% невидимых ПДн при одной лишней
    находке из 155, на диалогах лишних нет; wide=False — поиск как до итерации 8 (для сравнения).
    """
    by_id = {c.id: c for c in log.candidates}
    entity_groups = _clusters(list(log.entities), lambda a, b: a.type == b.type)
    cases = [_entity_case(g, by_id) for g in entity_groups]

    rejected = [
        c
        for c in log.candidates
        if c.decision == "rejected" and not any(_overlaps(c, e) for e in log.entities)
    ]
    cases += [_rejected_case(g) for g in _clusters(rejected, lambda a, b: True)]

    if session is not None:
        taken = list(log.entities) + list(log.candidates)
        for m in session.messages:
            found = [("s", hit) for hit in scan(m.text)]
            found += [("w", hit) for hit in scan_wide(m.text)] if wide else []
            numbers = {"s": 0, "w": 0}
            for prefix, (start, end, check) in found:
                numbers[prefix] += 1
                span = Case(
                    id=f"{session.id}-t{m.turn}-{prefix}{numbers[prefix]}",
                    kind="scan",
                    turn=m.turn,
                    start=start,
                    end=end,
                    text=m.text[start:end],
                    type=check.type,
                    entity_id=None,
                    candidates=(),
                )
                if not any(_overlaps(span, t) for t in taken):
                    cases.append(span)
    return sorted(cases, key=lambda c: (c.turn, c.start, c.kind))
