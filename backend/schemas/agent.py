from enum import StrEnum
from typing import Literal

from pydantic import Field

from backend.schemas._base import StrictModel


class Verdict(StrEnum):
    """Вердикт агента по срабатыванию или пропуску."""

    FALSE_POSITIVE = "false_positive"
    TRUE_PII = "true_pii"
    MISSED_PII = "missed_pii"
    UNCERTAIN = "uncertain"


class Citation(StrictModel):
    """Ссылка на точное место в переписке."""

    turn: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str


class Recommendation(StrictModel):
    """Что поменять в детекторе или кому передать, и почему."""

    kind: Literal["threshold", "pattern", "allowlist", "retrain", "ticket"]
    target: str
    change: str
    rationale: str


class AgentAnswer(StrictModel):
    """Ответ агента по одной сущности; entity_id = None — для пропуска."""

    entity_id: str | None
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    explanation: str
    citations: list[Citation]
    evidence: list[str]
    note: str
    recommendation: Recommendation | None = None


class SummaryIssue(StrictModel):
    """Повторяющаяся проблема детектора в нескольких сессиях."""

    title: str
    count: int = Field(ge=0)
    example_session_ids: list[str]
    example_entity_ids: list[str]


class Summary(StrictModel):
    """Сводка по N сессиям: проблемы, рекомендации и текст задачи в markdown."""

    session_ids: list[str]
    issues: list[SummaryIssue]
    recommendations: list[Recommendation]
    markdown: str
