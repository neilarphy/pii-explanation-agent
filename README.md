# pii-explanation-agent

## Разработка

Нужен [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync                          # окружение и зависимости
uv run pre-commit install        # хуки перед каждым коммитом
uv run pre-commit run --all-files  # прогнать хуки вручную
uv run pytest                    # тесты
```

Что проверяется перед коммитом и в CI на каждом PR:

- `ruff check` и `ruff format` — стиль и типовые ошибки в коде и ноутбуках;
- валидность YAML, TOML и JSON, пробелы в концах строк, переводы строк LF;
- файлы больше 1 МБ — датасеты не коммитим, а скачиваем скриптом;
- `gitleaks` — ключи API и токены не должны попасть в репозиторий, держим их в `.env`.

Зависимость добавляется через `uv add <пакет>` (или `uv add --dev <пакет>` для инструментов), `uv.lock` коммитим вместе с `pyproject.toml`.
