"""Ответы сервиса детекторов (POST /v1/detect) по репликам → один DetectionLog на сессию.

Сервис видит одну строку и ничего не знает о сессии, поэтому id сущностей и кандидатов
префиксуем `<session>-t<turn>-`, а текст кандидата берём из реплики по офсетам.
"""

import tomllib
from pathlib import Path

from backend.schemas import (
    Candidate,
    DetectionLog,
    Entity,
    ModelConfig,
    RuleConfig,
    Session,
)


def load_rules(profile_path: Path) -> list[RuleConfig]:
    """Правила профиля (id, тип, регулярка, валидатор) — в трейсе сервиса есть только id."""
    with profile_path.open("rb") as handle:
        data = tomllib.load(handle)
    return [RuleConfig(**rule) for rule in data["rules"]]


def to_detection_log(
    session: Session, responses: dict[int, dict], rules: list[RuleConfig]
) -> DetectionLog:
    """responses: номер реплики → JSON-ответ /v1/detect для текста этой реплики."""
    if not responses:
        raise ValueError(f"{session.id}: нет ни одного ответа детектора")
    texts = {m.turn: m.text for m in session.messages}
    first = next(iter(responses.values()))["trace"]

    entities: list[Entity] = []
    candidates: list[Candidate] = []
    for turn, response in sorted(responses.items()):
        prefix = f"{session.id}-t{turn}-"
        text = texts[turn]
        for e in response["entities"]:
            entities.append(
                Entity(
                    id=prefix + e["id"],
                    turn=turn,
                    start=e["start"],
                    end=e["end"],
                    text=e["text"],
                    type=e["type"],
                    candidate_ids=[prefix + cid for cid in e["candidate_ids"]],
                )
            )
        for c in response["trace"]["candidates"]:
            candidates.append(
                Candidate(
                    id=prefix + c["id"],
                    turn=turn,
                    start=c["start"],
                    end=c["end"],
                    text=text[c["start"] : c["end"]],
                    type=c["type"],
                    source=c["source"],
                    decision=c["decision"],
                    reason=c["reason"],
                    entity_id=prefix + c["entity_id"] if c.get("entity_id") else None,
                    score=c.get("score"),
                    rule_id=c.get("rule_id"),
                    raw_label=c.get("raw_label"),
                )
            )

    model = first["model"]
    return DetectionLog(
        session_id=session.id,
        profile=first["profile"],
        policy_version=first["policy_version"],
        model=ModelConfig(
            id=model["id"],
            emit_threshold=model["emit_threshold"],
            accept_threshold=model["accept_threshold"],
            labels=model["labels"],
        ),
        rules=rules,
        entities=entities,
        candidates=candidates,
    )
