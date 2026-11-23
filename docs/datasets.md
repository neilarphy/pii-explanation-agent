# Датасеты для оценки

| Источник | Данные и формат | Лицензия и ограничения | Использование в проекте |
|---|---|---|---|
| redmadrobot-rnd/pii_benchmark | Русский, 21 исходный тип. CSV с исходным текстом и JSON-массивами токенов и BIO-тегов. | MIT указан в карточке. Авторы сообщают о синтетической замене персональных данных в контекстах, часть которых получена из production-логов. | Внешний тестовый набор. На закреплённой ревизии проверено 2841 запись и 5614 сущностей. |
| PIIMB (pii-masking-benchmark) | Английский и европейские языки. В карточке около 25 тыс. документов и 150 тыс. предложений. JSONL с документами и предложениями. | CC BY-NC 4.0. Учитывать ограничения на коммерческое использование, атрибуцию и условия исходных материалов. | Дополнительный источник и методика. PIIMB использует символьный F2 без учёта типа, поэтому его баллы напрямую несопоставимы с метриками детектора в проекте. |
| Ai4Privacy pii-masking-200k | Английский, французский, немецкий и итальянский. В карточке около 209 тыс. примеров и 54 класса. JSONL с масками, спанами и BIO. | Собственные бесплатный и корпоративный режимы с условиями коммерческого использования. Файлы LICENSE и license.md ограничивают распространение производных от кода или вклада Ai4Privacy и по-разному формулируют критерии бесплатного использования. Для использования проверять оба файла именно этой ревизии. | Источник для сравнения. Лицензию более поздней версии OpenPII нельзя переносить на этот набор. |

Карточки закреплённых ревизий: [redmadrobot](https://huggingface.co/datasets/redmadrobot-rnd/pii_benchmark/tree/f77ea831274daf980cc45c61a93c226be9d978d6), [PIIMB](https://huggingface.co/datasets/piimb/pii-masking-benchmark/tree/72ea3393522295a680dd46f2f37caf9856d01a7c), [Ai4Privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-200k/tree/7c22e6816b8fc24b7781e9f4e22016c698842dd0). [PIIMB leaderboard](https://huggingface.co/spaces/piimb/pii-masking-benchmark-leaderboard) публикует результаты отдельно от датасета. PIIMB и Ai4Privacy целиком в рамках этой работы не скачивались.

## Воспроизведение redmadrobot

```sh
uv run python -m evaluation.redmadrobot --download
uv run python -m evaluation.redmadrobot --input data/raw/redmadrobot/test.csv
uv run python -m scripts.run_detector --redmadrobot data/raw/redmadrobot/test.csv --profile path/to/profile.toml --out runs/detector/redmadrobot.jsonl
uv run python -m scripts.evaluate --logs runs/detector/redmadrobot.jsonl
```

Загрузка и аудит сверяют SHA256 исходного CSV. [Manifest](../data/reports/redmadrobot-manifest.json)
содержит фактические счётчики закреплённого тестового набора. Аудит пишет локальный
`rejections.jsonl` с нулевым индексом записи CSV и фиксированным кодом причины.
Повторы текста учитываются без удаления записей. Каждый пример преобразуется в одну
входящую реплику, офсеты отсчитываются в Unicode code points исходного текста.
Типы сопоставляются со словарём pii-bench. Категории следуют текущему адаптеру проекта.
