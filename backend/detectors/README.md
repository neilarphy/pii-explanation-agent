# PII detector · HTTP API v1

Изолированный сервис существующего `pii-explanation-agent`: GLiNER на одной NVIDIA GPU,
16 независимых правил `python.re` на CPU, полный трейс кандидатов. Все новые файлы
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

Python-модули сервиса используют полные имена `backend.detectors.*`. Локально они
запускаются из корня репозитория; внутри образа корень пакетов — `/service`.
Compose собирает образ из корня репозитория с фильтром `Dockerfile.dockerignore`,
который оставляет только backend сервиса и исключает веса, окружения и `.env`.

```bash
# Один раз из корня клонированного репозитория
cd backend/detectors
mkdir -p models
docker compose build pii
docker compose run --rm --no-deps -v "$PWD/models:/prepare" pii python -m backend.detectors.scripts.prepare_models --output /prepare
docker compose up -d --wait --wait-timeout 600 pii
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8000/v1/detect -H 'Content-Type: application/json' -d '{"text":"🙂 Анна: a@example.org"}'

# Остановить сервис, если GPU не вмещает две копии модели
docker compose stop pii
docker compose run --rm --no-deps pii python -m pytest -m gpu
```

Healthcheck оставляет до 480 секунд на холодный старт. `live` отвечает, пока процесс
работает; `ready` становится успешным после загрузки и прогрева выбранной NER-схемы
и пробного запуска regex-правил выбранного профиля. Ошибка startup завершает Uvicorn ненулевым кодом.
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

## Выбор профиля

`PROFILE_PATH` выбирает TOML-профиль при старте процесса. Если переменная не задана,
используется встроенный `config/profile.toml`; пустое или пробельное значение — ошибка.
Относительный путь считается от рабочей директории процесса. В Python явный аргумент
`load_profile(path)` имеет приоритет над переменной окружения. `NER_LABELS` по-прежнему
переопределяет метки выбранного профиля.

Проверить выбор локально можно из корня репозитория, без весов и HTTP-зависимостей:

```bash
PROFILE_PATH=/absolute/path/experiment.toml python -c 'from backend.detectors.app.config import load_profile; print(load_profile().name)'
```

Для Compose есть дополнительный `compose.profile.yaml`. `PROFILE_FILE` указывает на
существующий файл на хосте; override подключает его только для чтения и устанавливает
контейнерный `PROFILE_PATH=/profiles/selected.toml`. Отсутствующий файл не создаётся
автоматически. Команда из `backend/detectors/`:

```bash
PROFILE_FILE=/absolute/path/experiment.toml docker compose \
  -f compose.yaml -f compose.profile.yaml \
  up -d --force-recreate --wait --wait-timeout 600 pii
```

Изменения профиля применяются после пересоздания контейнера, без пересборки образа.
Для возврата к встроенному профилю удалите `PROFILE_PATH` из shell и `.env`, затем
пересоздайте сервис с одним `-f compose.yaml`. Базовый Compose также передаёт
`PROFILE_PATH`, если вы самостоятельно подключили файл в контейнер.

Чтение файла, TOML, структура, пороги, метки и regex-правила проверяются до импорта
модельных библиотек и загрузки весов. Ошибка завершает startup с безопасной причиной,
без содержимого TOML и пути к файлу в техническом сообщении. В `trace.profile` и
`trace.policy_version` попадают идентификаторы выбранного профиля. При изменении
правил или порогов задавайте новые идентификаторы для сравнения результатов оценки.

Встроенный профиль `pii-ru-en-v4` включает `resolve_conflicts = true`. Этот флаг
включает уточнение конфликтующих типов по принятому правилу и фильтрацию карточных
номеров в явном контексте заказа, договора или реквизитов. Для старых профилей без
флага используется `false`; это позволяет сравнивать результаты с прежней политикой.
Поле принимает только TOML boolean, строки `"true"`/`"false"` отклоняются до загрузки модели.

`config/profile-with-ip.toml` содержит те же 16 правил и ещё два для IPv4/IPv6,
имя профиля — `pii-ru-en-v4-ip`. IP проверяются через стандартный `ipaddress`.
Выбрать встроенный вариант в Compose можно из `backend/detectors/`:

```bash
PROFILE_PATH=/service/backend/detectors/config/profile-with-ip.toml docker compose \
  up -d --force-recreate --wait --wait-timeout 600 pii
```

В основном профиле IP-правила выключены до решения о том, требуется ли их учитывать
как ПДн в продукте. Выбор дополнительного профиля явно включает тип `IP_ADDRESS`.

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
пустые окна/кандидаты/сущности и нули всех правил выбранного профиля. Обработанный текст без
совпадений имеет статус `ok`. Отклонённые кандидаты остаются в трейсе.

Профиль `pii-ru-en-v4`, политика `2.1.0`: модельный score принимается по `>= 0.50` без округления; декодер
эмитирует от `0.20`. Кандидаты сортируются по `(start,end,type,source_order,
origin_order,raw_label,-score)`, после чего получают `c1...`. Только принятые
наблюдения с точным `(start,end,type)` объединяются в сущность `e1...`. Политика
конфликтов ниже применяется перед объединением; остальные вложения, пересечения
и неоднозначности сохраняются. Каждая ссылка двусторонняя; у отклонённого
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
Каждый непустой match попадает в трейс до проверок. Если есть именованная группа
`entity`, используются её позиции; для телефонов словами используется группа `phone`.
Названия полей перед значением исключаются из спана. Порядок отказов: Unicode-границы,
контекст для паспортов и карт, формат, контрольное число. Причины: `invalid_boundary`,
`context_mismatch`, `invalid_format`, `checksum_failed`.
Проверяется форма записи, без установления владельца и подлинности документов.

Правило `phone_words_ru_v1` распознаёт русские телефоны словами. Примеры:

- `Телефон: восемь девять один два три четыре пять шесть семь восемь девять`;
- `Позвоните: восемь девятьсот двенадцать триста сорок пять шестьдесят семь восемьдесят девять`;
- `плюс один два ноль два пять пять пять ноль один два три`.

Regex находит запись; валидатор разбирает отдельные цифры и группы числительных до
999 и проверяет длину. С `плюс` допустимы 8–15 цифр с ненулевой первой цифрой.
Без `плюс` требуется контекст `телефон`, `тел.`, `номер телефона`, `мобильный`,
`позвони`/`позвоните` или `звони`/`звоните`; допустимы местные 7-значные,
национальные 10-значные и 11-значные номера с начальной 7 или 8. Начальный ноль
отклоняется. Это проверка формы записи, без проверки существования номера.

Поддерживаются регистр, `ноль`/`нуль`, `один`/`одна`/`одно`, `два`/`две`, пробелы,
включая неразрывные и переносы строк, запятые, дефисы и тире между словами.
Последовательности без телефонного контекста или `плюс` не классифицируются как
телефон: так проговорённые ИНН, СНИЛС и номера заказов не получают этот тип по длине.
Смешанная запись цифрами и словами, повторы вроде «три девятки» и английские
числительные этим правилом не разбираются.

Группа `(?P<phone>...)` обязательна для валидатора `phone_words_ru`: в сущность и
трейс попадают позиции самого номера, без слов `Телефон:`/`Позвоните:`. Исходная
строка сохраняется, преобразование в цифры используется только внутри валидатора.
Неверная длина даёт `invalid_format`, кандидат остаётся в трейсе целиком.

Остальные структурированные правила:

| Тип | Записи и проверки |
| --- | --- |
| `PHONE` | `+7 999 123-45-67`, `8 999 123 45 67`, `89991234567`, скобки вокруг кода, неразрывные/узкие пробелы, дефисы и тире. Для местных 7-значных, национальных 10-значных и международных номеров с разделителями есть правило с контекстом `телефон`, `тел.`, `мобильный`. |
| `INN` | После `ИНН`: 10 или 12 цифр, необязательные горизонтальные пробелы, проверка контрольных разрядов. Тип обозначает идентификатор, без вывода о том, кому он принадлежит. |
| `SNILS` | 11 цифр подряд или группами `XXX-XXX-XXX XX`, проверка контрольного числа. Старые номера, для которых контрольное число не рассчитывалось, отдельного обхода проверки не имеют. |
| `PASSPORT` | Четыре цифры серии и шесть номера, в том числе слитно. Поддерживаются `паспортные данные`, пояснение между названием документа и значением, несколько номеров и упоминание паспорта после значения. При записи `серия 75 18, номер 492137` в спан входит и обозначение серии. Отдельный шестизначный номер требует `номер`/`№` и паспортного контекста. |
| `DRIVER_LICENSE` | Такой же цифровой формат после `водительское удостоверение` или `права`, с отдельным публичным типом. |
| `BIRTH_CERTIFICATE` | Римская серия, две заглавные кириллические буквы и шесть цифр: `III-АГ 112233`; проверяется корректность римской записи. Допустимы пробелы, тире и слово `номер`. |
| `MILITARY_ID` | Две заглавные кириллические буквы и семь цифр: `МЗ 0045678`; поддерживаются поля `серия`/`номер`. |
| `OMS_POLICY` | После `полис ОМС`, `ОМС`, `ЕНП`: 16 цифр подряд или группами по четыре. Проверяется форма записи, без проверки Луна. |
| `DATE_OF_BIRTH` | После `дата рождения`, `родился`/`родилась`, `д.р.`, `DOB`, `born on`: день/месяц/год с единым разделителем либо ISO-дата. Валидируется календарь, включая високосные годы. |
| `ADDRESS` | Город с `г.` или без него, область/край/республика; тип улицы до или после названия, `пр.`, необязательные запятые и `д.`. В спан входят корпус, строение, квартира, офис, подъезд и этаж. Улица без обозначения типа требует города и дополнительных адресных компонентов либо явного адресного поля/указания доставки. |

Формат ИНН описан в [материалах ФНС](https://www.nalog.gov.ru/rn77/fl/interest/inn/),
контрольное число СНИЛС проверяется по [алгоритму ФНС](https://www.nalog.gov.ru/RN77/TAXATION/insprem/).
Все проверки выполняются локально. Новые регулярки работают с прежним набором NER-меток.

При `resolve_conflicts = true` принятая сработка с явным названием поля или `плюс`
имеет приоритет над совпадением только по форме. Для того же значения внутри её спана
конфликтующие идентификаторы и числовые `PERSON` внутри серий документов могут получить
`superseded_by_specific_rule`. Например, ОМС подавляет `PAYMENT_CARD`/`BANK_ACCOUNT`,
а права уточняют модельный `PASSPORT`/`NATIONAL_ID`. Оценка модели сохраняется.
Равноценные несовместимые правила сохраняются, отклонённое правило не отменяет принятый
кандидат другого источника. Остальные пересечения, включая обычные `PERSON`/`ADDRESS`,
обрабатываются по прежнему принципу сохранения.

Явный ближайший контекст `номер заказа`, `номер договора`, `БИК`, `КПП`, ИНН или
расчётный/корреспондентский счёт отклоняет карточную классификацию с `context_mismatch`.
Правило относится и к модельным карточным кандидатам. Число совпадений правил включает
отклонённые кандидаты; они сохраняются с `entity_id: null`. Политика `2.0.0` добавляет
значения `context_mismatch` и `superseded_by_specific_rule` в причины решений модели и
регулярок. Причина `superseded_by_specific_rule` требует принятого уточняющего правила.

Телефоны словами, шаблоны документов и адресов покрыты синтетическими проверками.
Результаты исходного HTTP-сравнения v1/v3 — в [result_v3.md](../../artifacts/result_v3.md);
проверка новых адресных и паспортных правил v4 — в [result_v4.md](../../artifacts/result_v4.md).

`passport_ru_v2` проверяет паспортное слово в пределах 140 символов перед значением
или 100 после него. Контекст не переносится через перевод строки, двойную кавычку
или точку с запятой. Явное соседнее поле ИНН, телефона, счёта или другого документа
отклоняет паспортную классификацию с `context_mismatch`; одиночные десятизначные
значения без паспортного контекста не принимаются. Это проверка контекста и формы,
без проверки подлинности документа. Адресные правила не используют словарь городов
или значения из бенчмарка и не покрывают произвольное описание местоположения.

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

Из корня репозитория `uv sync --locked` и `uv run --locked pytest` запускают тесты
без зависимостей детектора. Модули HTTP/политики и проверки HTTP-startup пропускаются,
если нужные FastAPI, HTTPX или Pydantic не установлены; regex и адаптер проверяются.
CI выполняет этот прогон, затем устанавливает CPU lock и повторяет полный набор
тестов. GPU-тесты запускаются отдельно с `-m gpu`.

Ниже команды из `backend/detectors/`. Локальный pytest использует корень репозитория
из `pytest.ini`, без изменения `sys.path` в тестовом коде.

```bash
# CPU-тесты в собственном окружении; CUDA и веса не нужны
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-cpu.lock
.venv/bin/python -m pytest -m 'not gpu'

# Regex импортируется даже без site-packages
(cd ../.. && python -S -c 'from backend.detectors.app.config import load_profile; from backend.detectors.app.regex_detector import RegexDetector; print(RegexDetector(load_profile()).detect("🙂 a@example.org a@example.org"))')

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
(cd ../.. && python -m backend.detectors.scripts.acceptance --output backend/detectors/acceptance-output/response.json)
cat acceptance-output/response.json
# Проверить отсутствие уникального синтетического маркера/значений в логах
! docker compose logs --no-color pii | rg 'acceptance-847291|example.org|Анна Иванова|John Smith'
```

Для ручного HTTP-замера есть `examples/manual_cases.json`: 36 синтетических текстов,
32 ожидаемые сущности и 9 отрицательных примеров. Ожидания фиксируются до вызова
сервиса. Совпадение проверяется по `(type, start, end)`; вложенные имя и фамилия
в полном ФИО разрешены явно. Ошибочные контрольные разряды и невозможная дата —
технические отрицательные примеры этого набора, без вывода о чувствительности
опечаток в реальных данных. Это небольшой ручной набор, не репрезентативный benchmark.

Команда из корня репозитория; укажите фактический URL и профиль работающего сервиса:

```bash
uv run --no-sync python -m backend.detectors.scripts.manual_measure \
  --url http://127.0.0.1:18000 \
  --cases backend/detectors/examples/manual_cases.json \
  --expected-profile pii-ru-en-v4 \
  --output backend/detectors/acceptance-output/manual-v4.json
```

Скрипт проверяет readiness, GPU-метаданные, версию профиля, заголовки и исходные спаны,
сохраняет полные ответы с трейсом и считает точные TP/FP/FN, precision/recall/F1,
число полностью прошедших текстов и HTTP-время. Несовпадение профиля прерывает запуск.
Начальная загрузка модели не входит во время запросов; единственный проход не служит
нагрузочным тестом. Результаты в `acceptance-output/` не попадают в git.

Для PII-Bench есть отдельный runner `scripts/pii_bench.py` с командами `collect`
и `score`. Прогон от 21.11.2026 использует весь
[`hivetrace/pii-bench@cd6a18a`](https://huggingface.co/datasets/hivetrace/pii-bench/tree/cd6a18ace16daf23e79247ccf1e2b245d4054654):
900 строк `domain` и 910 строк `entity`, без исключённых строк и ошибок разметки.
Исходные Parquet, подготовленные строки `data/items.jsonl`, SHA-256, ответы v1/v3,
CSV по типам и ошибки сохранены в `acceptance-output/pii-bench-cd6a18a-20261121/`.
Подготовка JSONL использует `pyarrow==26.0.1` в отдельном окружении оценки; он не нужен
для HTTP-прогона, подсчёта метрик или работы сервиса.

Пример повторного запуска из корня с уже подготовленным `items.jsonl`:

```bash
python -m backend.detectors.scripts.pii_bench collect \
  --items backend/detectors/acceptance-output/pii-bench-cd6a18a-20261121/data/items.jsonl \
  --url http://127.0.0.1:18000 --expected-profile pii-ru-en-v4 \
  --output backend/detectors/acceptance-output/pii-bench-new-responses.jsonl
python -m backend.detectors.scripts.pii_bench score \
  --responses backend/detectors/acceptance-output/pii-bench-new-responses.jsonl \
  --output backend/detectors/acceptance-output/pii-bench-new-metrics.json
```

В сервис передаётся точное исходное поле `text`, включая JSON-строку `L-DIALOG`,
без разбиения на реплики. Поэтому результат напрямую не сравнивается с прежним
runner из `main`, который разбирал диалоги и объединял ИНН/СНИЛС в `NATIONAL_ID`.
Здесь `INN` и `SNILS` оцениваются отдельно, ошибочный тип даёт одновременно FP и FN,
а совпадения сопоставляются один к одному. Полные метрики учитывают все 13 классов
эталона, включая неподдерживаемые `CVC`, `KPP`, `OGRN`, `OGRNIP`, `TOKEN`.
Отдельная сводка по восьми поддерживаемым классам фильтрует как эталонные, так и
предсказанные типы по этому списку.

Отчёт содержит строгие метрики на исходных entities и после удаления полностью
вложенных спанов одного типа. Постобработка не объединяет непересекающиеся части
адреса и не меняет HTTP-ответы. Дополнительно считаются типизированные пересечения
и покрытие символов независимо от типа. NAME/ADDRESS и остальные классы показаны
отдельно в CSV. При замере параметры модели, пороги и регулярки не менялись.

Для проверки только регулярок и merge есть `scripts/replay_regex.py`. Он берёт все
сохранённые модельные кандидаты, включая отклонённые, с исходными scores и окнами.
Пороги, NER-метки, отображение типов кандидатов и ревизия модели должны совпадать
с выбранным профилем. Regex и merge выполняются заново; HTTP, GPU-forward и latency
не измеряются. Снимок профиля и SHA-256 входного файла сохраняются рядом с метриками.

```bash
python -m backend.detectors.scripts.replay_regex \
  --responses backend/detectors/acceptance-output/pii-bench-cd6a18a-20261121/v3-responses.jsonl \
  --profile backend/detectors/config/profile.toml \
  --output backend/detectors/acceptance-output/regex-v4-new
```

Используйте Python-окружение с `requirements-cpu.lock`. Выходной каталог должен быть новым. Контрольный
replay исходного профиля v3 точно воспроизвёл entities и candidates для всех 1810
текстов. Правила v4 дорабатывались после просмотра ошибок PII-Bench, поэтому оценка
v4 на тех же текстах является проверкой изменений, а не независимой тестовой оценкой.

Повторный полный HTTP/GPU-прогон v4 от 21.11.2026 обработал все 1810 текстов без
HTTP-ошибок. Entities и candidates совпали с replay во всех текстах. Общий строгий
F1 после удаления вложенных дублей: 81.19% (`domain`) и 60.42% (`entity`);
нестрогий F1: 82.58% и 61.88%. Условия, время и артефакты приведены в
[result_v4.md](../../artifacts/result_v4.md).

`tests/fixtures/contract.json` — **контрактная фикстура** текущего профиля
с синтетической ревизией и заданными scores. Тест собирает и
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
запуска профиля `pii-ru-en-v1` с четырьмя правилами: request ID, времена и scores
не являются детерминированной фикстурой. Текущий профиль v4 содержит правила телефонов
словами, структурированных документов и адресов.

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
