"""Клиент LLM по OpenAI-совместимому API (по умолчанию bothub.chat)."""

import os
from collections.abc import Callable
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


def client_from_env() -> Complete:
    """LLM_API_KEY и LLM_MODEL обязательны, LLM_BASE_URL — по желанию (из окружения или .env)."""
    from openai import OpenAI

    load_env_file()

    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not api_key or not model:
        raise RuntimeError("Задайте переменные окружения LLM_API_KEY и LLM_MODEL")
    client = OpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))

    def complete(messages: list[dict[str, str]]) -> str:
        response = client.chat.completions.create(model=model, messages=messages, temperature=0)
        return response.choices[0].message.content or ""

    return complete
