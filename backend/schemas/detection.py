"""Лог детектора по сессии: поля как в трейсе сервиса backend/detectors, плюс номер реплики."""

from typing import Literal

from pydantic import Field

from backend.schemas._base import StrictModel

Reason = Literal[
    "model_threshold_passed",
    "below_model_threshold",
    "rule_passed",
    "invalid_boundary",
    "invalid_format",
    "checksum_failed",
]


class Candidate(StrictModel):
    """Кандидат от модели или регулярки, принятый или отклонённый, с причиной."""

    id: str
    turn: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    type: str
    source: Literal["model", "regex"]
    decision: Literal["accepted", "rejected"]
    reason: Reason
    entity_id: str | None
    score: float | None = Field(default=None, ge=0, le=1)
    rule_id: str | None = None
    raw_label: str | None = None


class Entity(StrictModel):
    """Найденная сущность ПДн: принятые кандидаты с одинаковыми спаном и типом."""

    id: str
    turn: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    type: str
    candidate_ids: list[str]


class ModelConfig(StrictModel):
    """NER-модель и её пороги: ниже emit кандидат не выдаётся, ниже accept — отклоняется."""

    id: str
    emit_threshold: float = Field(ge=0, le=1)
    accept_threshold: float = Field(ge=0, le=1)
    labels: list[str]


class RuleConfig(StrictModel):
    """Регулярка из профиля детектора."""

    id: str
    type: str
    pattern: str
    validator: str


class DetectionLog(StrictModel):
    """Лог срабатываний по сессии и снимок профиля детектора в момент проверки."""

    session_id: str
    profile: str
    policy_version: str
    model: ModelConfig
    rules: list[RuleConfig]
    entities: list[Entity]
    candidates: list[Candidate]
