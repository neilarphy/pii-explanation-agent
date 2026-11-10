# Чекпоинт 1 — минимальный системный дизайн

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 260}}}%%
flowchart LR
    FE["<b>Фронт под HiveTrace</b><br/>сессия с подсветкой<br/>кнопка «Объясни»<br/>страница сводки"]
    API["<b>Backend API</b><br/>FastAPI"]
    DB[("<b>База</b><br/>сессии<br/>лог срабатываний<br/>ответы агента")]
    DS["<b>Датасеты</b><br/>pii-bench<br/>redmadrobot<br/>сгенерированные диалоги"]
    DET["<b>Сервис детекторов</b><br/>regex + GLiNER<br/>профиль правил и порогов<br/>MVP: заранее на локальной GPU"]
    AG["<b>Агент</b><br/>вердикт + объяснение + заметка"]
    LLM["<b>LLM</b><br/>MVP: по API<br/>потом: Qwen до 27B локально"]
    SUM["<b>Сводка по N сессиям</b><br/>рекомендации: порог, паттерн<br/>.md для Claude Code"]
    EV["<b>Оценка</b><br/>метрики на размеченном наборе"]

    DS --> DET --> DB
    FE <--> API <--> DB
    API --> AG --> LLM
    AG --> DB
    AG --> SUM --> FE
    AG --> EV
    DS --> EV

    classDef models fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef data fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef agent fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef common fill:#f3f4f6,stroke:#6b7280,color:#111827
    class DET,LLM models
    class DS,EV data
    class AG,SUM agent
    class DB,API,FE common
```

Цвета: синий — Детекторы и модели, оранжевый — Данные и оценка, зелёный — Агент, серый — общее.

**Поток одного запроса**

1. Сессия загружена в базу.
2. Детекторы заранее прогоняются по сессии на локальной GPU, лог срабатываний загружается в базу.
3. Админ во фронте жмёт «Объясни».
4. Агент читает переписку и лог (включая отклонённые кандидаты), при необходимости перепроверяет регулярками и проверяет контрольную сумму.
5. Вердикт, ссылка на место в переписке и рекомендация видны во фронте.

**Эндпоинты** (без реализации): `POST /sessions`, `GET /sessions/{id}`, `GET /sessions/{id}/detections`, `POST /sessions/{id}/explain`, `POST /summary`.

**Стек и развёртывание**

- Python 3.12, FastAPI, pydantic (форматы из `backend/schemas`), SQLite на MVP.
- Фронт и backend на одном VPS без GPU через docker-compose.
- Сервис детекторов (`backend/detectors`) — отдельный контейнер с GPU. На MVP запускается локально, на сервер попадают готовые логи; перезапуск модели на сервере — после MVP, если появится режим на CPU.
- LLM на MVP по API, после MVP — локальная Qwen до 27B на GPU-квоте (рядом с инференсом HiveTrace на 40–60 ГБ).

**Чего нет в MVP**

- Правок детектора в проде: только рекомендации.
- Дообучения моделей.
- Авторизации.
