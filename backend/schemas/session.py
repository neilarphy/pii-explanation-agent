from typing import Any, Literal

from pydantic import Field

from backend.schemas._base import StrictModel


class Message(StrictModel):
    """Одна реплика: in — пользователь → LLM, out — ответ LLM."""

    turn: int = Field(ge=0)
    direction: Literal["in", "out"]
    text: str


class Session(StrictModel):
    """Переписка целиком: реплики по порядку и метаданные."""

    id: str
    messages: list[Message]
    meta: dict[str, Any] = Field(default_factory=dict)
