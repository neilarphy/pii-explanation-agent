# Михаил, задача 3: документы в профиле детектора и замер «до / после»

**Зачем.** Главная польза агента для HiveTrace — не «объяснил находку», а «нашел пробел детектора →
правка → детектор стал лучше на X». Пробел уже есть: в профиле нет прав, свидетельства о рождении,
военного билета и полиса ОМС (подробно — в ревью PR #5 и [detector-profile-iter8.md](detector-profile-iter8.md)).
Нужна правка и честный замер на тех же наборах, на которых мы считаем все остальное.

**Срок:** до 26.11.

## Что сделать в детекторе (ветка `detectors`, PR #5)

1. **Метки NER** в `default_ner_labels` и `[public_types]`. Публичные типы — ровно такие, оценка
   агента уже ждет эти имена:

   | Метка GLiNER (на твой выбор) | Публичный тип |
   |---|---|
   | например `driver_license` | `DRIVER_LICENSE` |
   | например `birth_certificate` | `BIRTH_CERTIFICATE` |
   | например `military_id` | `MILITARY_ID` |
   | например `medical_policy` | `OMS` |

2. **Правила по форме** (`[[rules]]`), где форма своя:
   - свидетельство о рождении: римская серия, две буквы, 6 цифр — `III-АГ 112233`, `II - КЗ № 445566`;
   - военный билет: две буквы и 7 цифр — `МЗ 0045678`, `АА 1234567`;
   - полис ОМС (ЕНП): 16 цифр, обычно группами по 4 через пробел, дефис или точку.
   Права (`77 14 987654`) по форме совпадают с паспортом — им только метка NER.
3. **Путь к профилю из env** (`PROFILE_PATH`, по умолчанию как сейчас) — чтобы сравнивать профили без
   правки файла.
4. **Порог emit не менять** (0.20): на 0.05 мусор 83% (проверено в итерации 8).

## Как замерить

Наборы (все синтетика или открытые):
- `data/scenarios/scenarios-v1.jsonl`, `data/scenarios/hidden-v1.jsonl` — в нашем репо;
- pii-bench: `domain.parquet` с https://huggingface.co/datasets/hivetrace/pii-bench;
- redmadrobot: `test.csv` с https://huggingface.co/datasets/redmadrobot-rnd/pii_benchmark.

Для **старого** профиля (`git show origin/detectors:backend/detectors/config/profile.toml > old.toml`) и
**нового** — поднять сервис с этим профилем и из корня нашего репо (ветка `feat/agent-iter8`):

    uv run --group eval python -m scripts.run_detector --redmadrobot <путь>/test.csv \
        --profile <профиль>.toml --out runs/detector-<old|new>/redmadrobot.jsonl
    uv run --group eval python -m scripts.run_detector --data <папка с domain.parquet> \
        --profile <профиль>.toml --out runs/detector-<old|new>/pii-bench.jsonl
    uv run --group eval python -m scripts.run_detector --scenarios data/scenarios/scenarios-v1.jsonl \
        --profile <профиль>.toml --out runs/detector-<old|new>/scenarios-v1.jsonl
    uv run --group eval python -m scripts.run_detector --scenarios data/scenarios/hidden-v1.jsonl \
        --profile <профиль>.toml --out runs/detector-<old|new>/hidden-v1.jsonl

Метрики (без LLM, бесплатно), для каждого набора и обоих профилей:

    uv run python -m scripts.evaluate --logs runs/detector-<old|new>/redmadrobot.jsonl \
        --out runs/eval-detector-<old|new>-redmadrobot

Таблица «Детектор» в `report.md`: точность, полнота, F1 по типам. Сейчас на redmadrobot у четырех
документов полнота 0 (371 + 341 + 278 + 170 пропусков), PASSPORT F1 0.39, NATIONAL_ID F1 0.41.

## Что сдать

В описание PR #5 — таблица «до / после» по каждому набору:

| Тип | F1 до | F1 после | Полнота до → после | Точность до → после |
|---|---|---|---|---|
| DRIVER_LICENSE, BIRTH_CERTIFICATE, MILITARY_ID, OMS | | | | |
| PASSPORT, NATIONAL_ID, PAYMENT_CARD, BANK_ACCOUNT, PHONE | | | | |
| Всего | | | | |

Плюс 5 строк из ревью как тесты детектора. Логи `runs/detector-new/*.jsonl` пришли Анатолию (в git
они не хранятся): по ним прогоним агента и проверим, что агент тоже стал ошибаться меньше.

**Критерий успеха:** «Всего» F1 на redmadrobot растет; F1 по PASSPORT и NATIONAL_ID не падает; на
scenarios-v1 и pii-bench ничего не хуже, чем было (разница в пределах ±0.01). Если какой-то тип GLiNER
не берет — так и напиши, это тоже результат.
