# Issue (предложение)

Title: Data & Evaluation: контракт v1 и воспроизводимый импорт redmadrobot
Label: role:data-eval

Подготовить обзор источников, предложить единый формат примера, реализовать
строгую валидацию и pinned CSV/BIO импорт redmadrobot. Согласовать offsets,
таксономию и trace с владельцами backend. Не включать оценку агента в этот этап.

Критерии: JSON Schema, синтетические примеры, offline unit-тесты, манифест
checksum/revision/filter/counts, явное отсутствие логов до запуска детектора.

# Pull Request (предложение)

Title: Add versioned evaluation data schema and pinned redmadrobot loader
Branch: feature/data-eval-benchmarks
Base inspected: 4c0ed33

Добавлен контракт данных со строгими Unicode-спанами и ссылками на сообщения,
сущности и детекции. Импорт BIO сохраняет исходный текст, отклоняет некорректные
строки и отличает отсутствие запуска от запуска без срабатываний. Добавлены
JSONL-валидатор, JSON Schema, синтетические примеры и обзор источников/лицензий.
Pydantic — единственная новая runtime-зависимость, uv.lock обновлён.

Проверка полного pinned CSV: 2841 строка → 2840 примеров, 5612 сущностей;
1 строка с пустым токеном исключена и учтена в журнале. Реальные трассы и
результаты оценки детекторов/агента не создавались. Backend не изменён.

Запуск: uv sync --locked; uv run pytest; uv run ruff check .;
uv run ruff format --check .; команды импорта см. data/README.md.
До интеграции согласовать контракт с владельцами backend/schemas и detectors.
