# PII detector · HTTP API v1

Изолированный сервис существующего `pii-explanation-agent`: GLiNER на одной NVIDIA GPU,
четыре независимых правила `python.re` на CPU, полный трейс кандидатов. Все новые файлы
находятся в `backend/detectors/`. Другие модули, API и зависимости корневого проекта
не изменены. В исходном backend были только пустые пакеты, общих Compose/Dockerfile
и HTTP-приложения не было; поэтому здесь отдельный Compose с одним сервисом `pii`.
Другой backend обращается к нему по HTTP. Единственное изменение вне каталога — `.github/workflows/ci.yml`: существующий
pytest job устанавливает CPU lock сервиса и запускает тесты с `uv run --no-sync`,
чтобы корневой CI проверял новый сервис без CUDA и без изменения `uv.lock`.

## Запуск

Команды ниже выполняются из `backend/detectors/`, кроме первого `cd`. Нужны Docker
Compose v2 с `--wait`, NVIDIA Container Toolkit, совместимый драйвер NVIDIA и GPU с
достаточной памятью для float32. Сборка и подготовка весов требуют сети. Runtime
читает `/models` read-only и использует только локальные файлы.

```bash
# Один раз из корня клонированного репозитория
cd backend/detectors
mkdir -p models
docker compose build pii
docker compose run --rm --no-deps -v "$PWD/models:/prepare" pii python -m scripts.prepare_models --output /prepare
docker compose up -d --wait --wait-timeout 600 pii
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8000/v1/detect -H 'Content-Type: application/json' -d '{"text":"🙂 Анна: a@example.org"}'

# Остановить сервис, если GPU не вмещает две копии модели
docker compose stop pii
docker compose run --rm --no-deps pii python -m pytest -m gpu
```

Healthcheck оставляет до 480 секунд на холодный старт. `live` отвечает, пока процесс
работает; `ready` становится успешным после загрузки и прогрева выбранной NER-схемы
и проверки повторных regex-спанов. Ошибка startup завершает Uvicorn ненулевым кодом.
CUDA OOM/ошибка GPU снимает readiness и завершает процесс с кодом 1 через секунду,
чтобы дать HTTP-ошибке отправиться; Compose перезапускает контейнер.

Если Docker сообщает `Using requested mode 'cdi'` и требует `--runtime=nvidia`,
проверьте наличие `runtime: nvidia` у сервиса `pii` в Compose. Резервирование одной
GPU в `deploy.resources.reservations.devices` при этом сохраняется. После изменения
достаточно пересоздать контейнер; пересборка образа не нужна:

```bash
docker compose run --rm --no-deps pii nvidia-smi
docker compose up -d --force-recreate --wait --wait-timeout 600 pii
```

Если появляется `unknown or invalid runtime name: nvidia`, NVIDIA runtime ещё не
зарегистрирован в Docker; настройте NVIDIA Container Toolkit для своего Docker daemon.

Если порт 8000 занят другим сервисом, задайте свободный внешний порт через `PII_PORT`.
Например, добавьте `PII_PORT=18000` в `.env` рядом с Compose и пересоздайте `pii`.
Тогда HTTP доступен по `http://127.0.0.1:18000`; внутри контейнера порт остаётся 8000.

Выбор NER-меток не требует пересборки образа:

```bash
NER_LABELS=person,address,passport docker compose up -d --force-recreate --wait --wait-timeout 600 pii
curl -fsS http://127.0.0.1:8000/v1/detect -H 'Content-Type: application/json' -d '{"text":"a@example.org"}'
```

В трейсе: `model.labels == ["person","address","passport"]`, `regex.device == "cpu"`,
принятый regex-кандидат `email_ascii_v1`. В частности, `person` означает ровно одну
метку, без автоматического добавления `first_name,last_name`. Для сохранения выбора
добавьте `NER_LABELS=person,address,passport` в локальный `.env` рядом с compose.yaml.
Для возврата к 11 меткам профиля удалите переменную из shell (`unset NER_LABELS`) и
`.env`, затем пересоздайте контейнер. Пустое значение, пустая позиция, дубликаты и
неверный синтаксис завершают запуск до загрузки весов. Пользовательская метка
`employee_id` отображается в публичный тип `EMPLOYEE_ID`. Regex от выбора не зависит.

## Контракт

`POST /v1/detect`, `Content-Type: application/json` (charset допустим), только
`{"text":"..."}`. Строгая строка, Unicode scalar values; лишние поля, null, числа,
невалидный JSON и одиночные суррогаты дают 422. Исходная строка сохраняется без trim,
нормализации и замены CRLF. `start` включён, `end` исключён, единица — кодовая точка
Python `str`; `entity.text == text[start:end]`. В JavaScript:
`Array.from(text).slice(start, end).join("")`.

Схемы ответа описаны в `app/schemas.py`. UUID сервера повторяется в `X-Request-ID`;
обнаружение и ошибки имеют `Cache-Control: no-store`. Целого документа в ответе нет.
Пустой/пробельный текст проходит readiness и admission, но даёт `skipped_blank`,
пустые окна/кандидаты/сущности и нули всех четырёх правил. Обработанный текст без
совпадений имеет статус `ok`. Отклонённые кандидаты остаются в трейсе.

Политика `1.0.0`: модельный score принимается по `>= 0.50` без округления; декодер
эмитирует от `0.20`. Кандидаты сортируются по `(start,end,type,source_order,
origin_order,raw_label,-score)`, после чего получают `c1...`. Только принятые
наблюдения с точным `(start,end,type)` объединяются в сущность `e1...`. Вложения,
пересечения и разные типы не удаляются. Каждая ссылка двусторонняя; у отклонённого
кандидата `entity_id: null`. Сводного confidence у сущности нет. Пороговые изменения
требуют новой версии профиля/политики и отдельной оценки на размеченных данных.

Окна планируются до GPU-forward по границам слов реального процессора: цель 192,
перекрытие до 32, при уменьшении окна — не более половины его символов. Размер
каждого окна проверяется через `processor.transform_record(..., max_len=None)`:
схема, структурные токены и все subwords учитываются до усечения. Ограничения
берутся из encoder/tokenizer/config; длинное слово дробится по кодовым точкам.
Первый start=0, последний end=len(text), дыр нет. Каждый вызов `extract` виден
отдельным окном, включая нули и повторные наблюдения до дедупликации. Regex всегда
видит весь документ. GLiNER2 2.1.0 сам обычно добавляет точку; адаптер отключает
`processor._normalize_text` на конкретном экземпляре, чтобы сохранить точный ввод.
Эта привязка к API покрыта отдельным тестом реального процессора в группе `gpu`.

Regex-профиль содержит исходные TOML literal strings, ID и имена валидаторов.
Правила компилируются один раз, `finditer` исполняется один раз для каждого правила.
Каждый непустой match попадает в трейс до проверок. Порядок отказов: Unicode-границы,
формат, Лун. Причины: `invalid_boundary`, `invalid_format`, `checksum_failed`.
Проверяется форма записи, без установления владельца и подлинности документов.

Лимиты: HTTP-тело 512 КиБ по фактическим ASGI-чанкам, текст 32768 кодовых точек,
4096 кандидатов суммарно до фильтрации, 512 окон включительно. Усечённый успешный
трейс запрещён. Лимит тела/текста даёт 413, трейса/окон — 422.

Ошибки содержат только `schema_version`, `request_id`, `error` с `code`,
`failed_stage`, `completed_stages`. Завершённые этапы перечисляются по порядку;
при ошибке regex, например, это `validation, admission, model`. Тексты исключений,
Pydantic `input`, `entities` и `has_pii` в ошибках отсутствуют.

| HTTP | code |
| --- | --- |
| 415 | UNSUPPORTED_MEDIA_TYPE |
| 422 | INVALID_REQUEST / TRACE_LIMIT_EXCEEDED |
| 413 | INPUT_TOO_LARGE |
| 429 | BUSY (`Retry-After: 1`) |
| 503 | NOT_READY / DETECTOR_UNAVAILABLE |
| 500 | ADAPTER_CONTRACT_ERROR / INTERNAL_ERROR |

Один worker и выделенный поток исполняют последовательно model → regex → merge.
FIFO-очередь допускает 8 ожидающих, время ожидания 5 секунд. Отмена ожидающего
удаляет его из очереди, отключение клиента во время работы не освобождает слот до
фактического завершения задачи. Health обслуживается event loop. Время монотонное;
перед завершением модельного этапа вызывается `torch.cuda.synchronize(0)`.

Технические логи содержат ID запроса, этап, код, число сущностей и время. Внешние
логгеры фильтруются; access log, телеметрия и запись запросов выключены. Кэш
токенизации очищается после запроса. Сервис рассчитан на доверенного локального
клиента, порт опубликован только на 127.0.0.1. Для внешней публикации аутентификация
и TLS возлагаются на доверенный reverse proxy.

## Артефакты и закреплённые версии

| Компонент | Закрепление |
| --- | --- |
| Базовый образ, linux/amd64 | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| Digest образа | `sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee` |
| Python в образе | 3.11.11, закреплён digest базового образа |
| PyTorch / CUDA / dtype | 2.6.0 / 12.4.1 / float32 |
| gliner2 / Transformers / tokenizers | 2.1.0 / 4.57.1 / 0.22.1 |
| Модель | `hivetrace/gliner-guard-omni@bda9b81a77421f48bff373d8339adaf225ff8764` |
| Базовый токенизатор | `microsoft/mdeberta-v3-base@02a99710df3b0711f315e921b990df0191474ef7` |
| FastAPI / Pydantic / Uvicorn | 0.141.1 / 2.13.5 / 0.52.4 |

`requirements.lock` фиксирует прямые и транзитивные Python-зависимости поверх
неизменяемого образа; torch/CUDA/triton уже поставлены в нём. Установка идёт с
`--no-deps`, затем `pip check` проверяет полноту. Корневой `uv.lock` не меняется,
так как сервис имеет отдельное окружение. CPU-разработка совместима с Python 3.12
корневого проекта; есть отдельный `requirements-cpu.lock` без torch/HF.

Подготовка скачивает pinned checkpoint, encoder_config, checkpoint tokenizer и
базовый SentencePiece. Сохраняет fast tokenizer локально с исходными ID словаря.
`manifest.json` содержит обе ревизии, фактические версии модельных библиотек и
SHA-256 каждого файла; startup проверяет файлы, отсутствие лишних артефактов и
совпадение версий. Неполная/повреждённая подготовка не даёт успешной загрузки.
Скрипт подготовки явно снимает offline-переменные только в своём процессе, до
импорта HF. В runtime остаются `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` и
изолированный `HF_HOME=/tmp/empty-hf-cache`, пользовательский кэш не монтируется.

## Тестирование и приёмка

```bash
# CPU-тесты в собственном окружении; CUDA и веса не нужны
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-cpu.lock
.venv/bin/python -m pytest -m 'not gpu'

# Regex импортируется даже без site-packages
python -S -c 'from app.config import load_profile; from app.regex_detector import RegexDetector; print(RegexDetector(load_profile()).detect("🙂 a@example.org a@example.org"))'

# В образе доступны те же CPU-тесты
docker compose run --rm --no-deps pii python -m pytest -m 'not gpu'

# Реальный forward, RU/EN, offsets, реальный процессор и последнее окно
docker compose stop pii
docker compose run --rm --no-deps pii python -m pytest -m gpu

# Холодный старт без сети и пользовательского HF-кэша
# Отсутствие host-порта в этом запуске намеренно: проверка внутри контейнера.
docker run -d --name pii-offline --runtime=nvidia --gpus 1 --network none -v "$PWD/models:/models:ro" pii-detectors:v1
# Дождаться готовности (до 600 секунд)
timeout 600 sh -c 'until docker exec pii-offline python -c '\''import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health/ready")'\''; do sleep 5; done'
docker exec pii-offline python -m pytest -m gpu
docker rm -f pii-offline

# Зафиксировать реальный HTTP-ответ для отчёта (только синтетические данные)
docker compose up -d --wait --wait-timeout 600 pii
python -m scripts.acceptance --output acceptance-output/response.json
cat acceptance-output/response.json
# Проверить отсутствие уникального синтетического маркера/значений в логах
! docker compose logs --no-color pii | rg 'acceptance-847291|example.org|Анна Иванова|John Smith'
```

`tests/fixtures/contract.json` — **контрактная фикстура**, дословно соответствующая
примеру спецификации с синтетической ревизией и заданными scores. Тест собирает и
сравнивает весь ответ. Это не ответ живой модели. GPU-smoke использует реальные
scores и проверяет корректность спанов, а не совпадение с этой фикстурой.

Статус выполнения в предоставленной среде (2026-11-11):

- CPU: Python 3.12.12, FastAPI 0.141.1, Pydantic 2.13.5, pytest 9.1.1; детерминированные
  тесты политики, regex, HTTP, лимитов, очереди, отмены и адаптера выполнены:
  **122 passed, 3 skipped** (GPU-тесты); общий запуск репозитория —
  **128 passed, 3 skipped**. После уточнения обработки disconnect отдельно повторно
  пройдены 35 тестов HTTP/очереди.
- Из-за отсутствия сети в shell пакеты взяты read-only из уже установленного
  Python 3.12-окружения, без изменения окружений других проектов. Sandbox запрещает
  `socketpair.send` (`EPERM`), поэтому для threaded/HTTP тестов использован временный
  runner с пробуждением event loop каждые 5 мс. В коде сервиса такого обхода нет.
- После исправления CDI проверен запуск контейнера с `runtime: nvidia`:
  **NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, драйвер 580.178.04**;
  Python 3.11.11, torch 2.6.0+cu124, CUDA PyTorch 12.4, gliner2 2.1.0,
  Transformers 4.57.1. `python -m pip check`: `No broken requirements found`.
- Веса и манифест подготовлены в собранном образе. Сервис прошёл startup/GPU-прогрев,
  readiness и реальный HTTP-smoke. У модели `device=cuda:0`, `dtype=float32`,
  `status=ok`; у regex `device=cpu`, `engine=python.re`, `status=ok`. Проверены срезы
  сущностей по исходной строке; email имеет оба источника. В логах этого запроса нет
  синтетического маркера, значений сущностей или тела ответа.
- На этом хосте 8000 занят `mineru-api`, поэтому локальный `.env` задаёт
  `PII_PORT=18000`, а сервис доступен на `http://127.0.0.1:18000`. Без настройки
  внешний порт по-прежнему 8000. Системная конфигурация Docker не изменялась.
- **Ещё не выполнены:** cold start с `--network none`, отдельная группа `pytest -m gpu`
  (включая проверку последнего окна) и весь CPU-набор непосредственно в образе.

Реальный ответ поднятого сервиса на синтетический текст
`🙂 Анна Иванова / John Smith: acceptance-847291@example.org` также сохранён в
[`examples/live_response.json`](examples/live_response.json). Это снимок одного
запуска: request ID, времена и scores не являются детерминированной фикстурой.

<details>
<summary>Полный реальный HTTP-ответ</summary>

```json
{
  "schema_version": "1.0",
  "request_id": "1d4e83ed-5db7-461f-8988-e7f6d980a9b9",
  "has_pii": true,
  "entities": [
    {
      "id": "e1",
      "type": "PERSON",
      "start": 2,
      "end": 6,
      "text": "Анна",
      "candidate_ids": [
        "c1"
      ]
    },
    {
      "id": "e2",
      "type": "PERSON",
      "start": 2,
      "end": 14,
      "text": "Анна Иванова",
      "candidate_ids": [
        "c2"
      ]
    },
    {
      "id": "e3",
      "type": "PERSON",
      "start": 7,
      "end": 14,
      "text": "Иванова",
      "candidate_ids": [
        "c3"
      ]
    },
    {
      "id": "e4",
      "type": "PERSON",
      "start": 17,
      "end": 21,
      "text": "John",
      "candidate_ids": [
        "c4"
      ]
    },
    {
      "id": "e5",
      "type": "PERSON",
      "start": 17,
      "end": 27,
      "text": "John Smith",
      "candidate_ids": [
        "c5"
      ]
    },
    {
      "id": "e6",
      "type": "PERSON",
      "start": 22,
      "end": 27,
      "text": "Smith",
      "candidate_ids": [
        "c6"
      ]
    },
    {
      "id": "e7",
      "type": "EMAIL",
      "start": 29,
      "end": 58,
      "text": "acceptance-847291@example.org",
      "candidate_ids": [
        "c7",
        "c8"
      ]
    }
  ],
  "trace": {
    "profile": "pii-ru-en-v1",
    "policy_version": "1.0.0",
    "offset_unit": "unicode_code_point",
    "text_length": 58,
    "model": {
      "id": "hivetrace/gliner-guard-omni",
      "revision": "bda9b81a77421f48bff373d8339adaf225ff8764",
      "device": "cuda:0",
      "dtype": "float32",
      "status": "ok",
      "emit_threshold": 0.2,
      "accept_threshold": 0.5,
      "labels": [
        "person",
        "first_name",
        "last_name",
        "address",
        "email",
        "phone",
        "passport",
        "national_id",
        "date_of_birth",
        "card_number",
        "bank_account"
      ],
      "windows": [
        {
          "id": "w1",
          "start": 0,
          "end": 58,
          "candidate_count": 7
        }
      ]
    },
    "regex": {
      "engine": "python.re",
      "device": "cpu",
      "status": "ok",
      "rules": [
        {
          "id": "email_ascii_v1",
          "matches": 1
        },
        {
          "id": "phone_plus_v1",
          "matches": 0
        },
        {
          "id": "phone_ru_v1",
          "matches": 0
        },
        {
          "id": "card_pan_v1",
          "matches": 0
        }
      ]
    },
    "candidates": [
      {
        "id": "c1",
        "type": "PERSON",
        "start": 2,
        "end": 6,
        "decision": "accepted",
        "entity_id": "e1",
        "source": "model",
        "raw_label": "first_name",
        "window_id": "w1",
        "score": 0.9982234835624695,
        "reason": "model_threshold_passed"
      },
      {
        "id": "c2",
        "type": "PERSON",
        "start": 2,
        "end": 14,
        "decision": "accepted",
        "entity_id": "e2",
        "source": "model",
        "raw_label": "person",
        "window_id": "w1",
        "score": 0.9834153652191162,
        "reason": "model_threshold_passed"
      },
      {
        "id": "c3",
        "type": "PERSON",
        "start": 7,
        "end": 14,
        "decision": "accepted",
        "entity_id": "e3",
        "source": "model",
        "raw_label": "last_name",
        "window_id": "w1",
        "score": 0.9610546827316284,
        "reason": "model_threshold_passed"
      },
      {
        "id": "c4",
        "type": "PERSON",
        "start": 17,
        "end": 21,
        "decision": "accepted",
        "entity_id": "e4",
        "source": "model",
        "raw_label": "first_name",
        "window_id": "w1",
        "score": 0.9922307133674622,
        "reason": "model_threshold_passed"
      },
      {
        "id": "c5",
        "type": "PERSON",
        "start": 17,
        "end": 27,
        "decision": "accepted",
        "entity_id": "e5",
        "source": "model",
        "raw_label": "person",
        "window_id": "w1",
        "score": 0.9110621809959412,
        "reason": "model_threshold_passed"
      },
      {
        "id": "c6",
        "type": "PERSON",
        "start": 22,
        "end": 27,
        "decision": "accepted",
        "entity_id": "e6",
        "source": "model",
        "raw_label": "last_name",
        "window_id": "w1",
        "score": 0.9968030452728271,
        "reason": "model_threshold_passed"
      },
      {
        "id": "c7",
        "type": "EMAIL",
        "start": 29,
        "end": 58,
        "decision": "accepted",
        "entity_id": "e7",
        "source": "model",
        "raw_label": "email",
        "window_id": "w1",
        "score": 0.9999179840087891,
        "reason": "model_threshold_passed"
      },
      {
        "id": "c8",
        "type": "EMAIL",
        "start": 29,
        "end": 58,
        "decision": "accepted",
        "entity_id": "e7",
        "source": "regex",
        "rule_id": "email_ascii_v1",
        "score": null,
        "reason": "rule_passed"
      }
    ],
    "timings_ms": {
      "queue": 0.005603003955911845,
      "model": 416.6748479983653,
      "regex": 0.08794699533609673,
      "merge": 0.3253430040786043,
      "total": 423.5369169982732
    }
  }
}
```

</details>

После приёмки приложите `acceptance-output/response.json`, вывод `nvidia-smi`,
`docker image inspect pii-detectors:v1`, `python --version` внутри контейнера и
результат `pytest -m gpu`. Качество RU/EN, precision/recall и latency требуют
отдельной размеченной выборки и целевого оборудования; smoke не измеряет их.

Источники: [модель и закреплённый checkpoint](https://huggingface.co/hivetrace/gliner-guard-omni/tree/bda9b81a77421f48bff373d8339adaf225ff8764),
[GLiNER2 2.1.0 processor](https://github.com/fastino-ai/GLiNER2/blob/v2.1.0/gliner2/processor.py),
[локальный extract](https://github.com/fastino-ai/GLiNER2/blob/v2.1.0/tutorial/4-combined.md),
[long context](https://github.com/fastino-ai/GLiNER2/blob/v2.1.0/tutorial/12-long_context.md),
[Compose GPU](https://docs.docker.com/compose/how-tos/gpu-support/),
[digest базового образа](https://hub.docker.com/layers/pytorch/pytorch/2.6.0-cuda12.4-cudnn9-runtime/images/sha256-77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee).
