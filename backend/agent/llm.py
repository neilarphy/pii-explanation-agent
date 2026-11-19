"""Клиент LLM по OpenAI-совместимому API (по умолчанию bothub.chat)."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Complete = Callable[[list[dict[str, str]]], str]

DEFAULT_BASE_URL = "https://bothub.chat/api/v2/openai/v1"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Подхватывает KEY=VALUE из .env в корне репо; уже заданные переменные не трогает."""
    if not path.exists():
        return
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _client(model: str | None):
    from openai import OpenAI

    load_env_file()

    api_key = os.environ.get("LLM_API_KEY")
    model = model or os.environ.get("LLM_MODEL")
    if not api_key or not model:
        raise RuntimeError("Задайте переменные окружения LLM_API_KEY и LLM_MODEL")
    client = OpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    return client, model


def client_from_env(model: str | None = None) -> Complete:
    """LLM_API_KEY и LLM_MODEL обязательны, LLM_BASE_URL — по желанию (из окружения или .env).

    model переопределяет LLM_MODEL — например, чтобы генерировать данные или судить другой моделью.
    """
    client, model = _client(model)

    def complete(messages: list[dict[str, str]]) -> str:
        response = client.chat.completions.create(model=model, messages=messages, temperature=0)
        return response.choices[0].message.content or ""

    return complete


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON-строка, как её прислала модель


@dataclass(frozen=True)
class Reply:
    """Ответ модели в харнессе: текст или вызовы инструментов, плюс расход токенов."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0  # USD, как его считает провайдер (usage.cost); 0 — не сообщил


Chat = Callable[[list[dict], list[dict] | None], Reply]


def chat_from_env(model: str | None = None) -> Chat:
    """Как client_from_env, но с инструментами (OpenAI function calling) и счётчиком токенов."""
    client, model = _client(model)

    def chat(messages: list[dict], tools: list[dict] | None = None) -> Reply:
        extra = {"tools": tools} if tools else {}
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0, **extra
        )
        message = response.choices[0].message
        usage = response.usage
        return Reply(
            text=message.content or "",
            tool_calls=tuple(
                ToolCall(c.id, c.function.name, c.function.arguments or "{}")
                for c in message.tool_calls or []
            ),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cost=float(getattr(usage, "cost", None) or 0),
        )

    return chat
