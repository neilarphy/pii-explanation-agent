# Расход на LLM по итерациям

Собрано `scripts/costs.py` из сохранённых прогонов (`runs/`), без вызовов модели. Цена — по
`usage.cost` от провайдера (курс ≈ 700 000 капсов за $) или по тарифам bothub × токены,
где цены нет. В итерациях 1–4 телеметрии не было — «нет данных». Прогоны, которые не сохранили
ответов (генерация сценариев, прерванные команды), сюда не попадают — они в журнале итераций.

## Итого по итерациям

| Итерация | Прогонов | Ответов | Вызовов LLM | Токенов вход / выход | Капсов, млн | Основа |
|---|---|---|---|---|---|---|
| 1 | 1 | 50 | — | 0 / 0 | — | нет данных |
| 2 | 3 | 624 | — | 0 / 0 | — | нет данных |
| 3 | 1 | 238 | — | 0 / 0 | — | нет данных |
| 4 | 6 | 1416 | — | 0 / 0 | — | нет данных |
| 5 | 4 | 733 | 735 | 887 519 / 1 133 154 | 1.79 | тарифы × токены |
| 6 | 69 | 10003 | 10506 | 11 273 933 / 7 101 153 | 6.87 | usage.cost, смешанно, тарифы × токены |
| 7 | 20 | 3118 | 5019 | 6 342 214 / 3 904 061 | 4.92 | usage.cost, смешанно |
| 8 | 13 | 841 | 1429 | 2 155 705 / 1 111 887 | 1.05 | usage.cost |

## По шагам

### Итерация 1

| Прогон | Модель | Ответов | Капсов на ответ | Капсов, тыс. | Основа |
|---|---|---|---|---|---|
| `eval-domain` | qwen3.8-27b | 50 | — | — | нет данных |

### Итерация 2

| Прогон | Модель | Ответов | Капсов на ответ | Капсов, тыс. | Основа |
|---|---|---|---|---|---|
| `iter2-domain` | qwen3.8-27b | 226 | — | — | нет данных |
| `iter2-entity` | qwen3.8-27b | 160 | — | — | нет данных |
| `iter2-scenarios` | qwen3.8-27b | 238 | — | — | нет данных |

### Итерация 3

| Прогон | Модель | Ответов | Капсов на ответ | Капсов, тыс. | Основа |
|---|---|---|---|---|---|
| `iter3-scenarios-fixed` | qwen3.8-27b | 238 | — | — | нет данных |

### Итерация 4

| Прогон | Модель | Ответов | Капсов на ответ | Капсов, тыс. | Основа |
|---|---|---|---|---|---|
| `iter4-domain` | qwen3.8-27b | 238 | — | — | нет данных |
| `iter4-entity` | qwen3.8-27b | 217 | — | — | нет данных |
| `iter4-scenarios` | qwen3.8-27b | 253 | — | — | нет данных |
| `iter4a-domain` | qwen3.8-27b | 238 | — | — | нет данных |
| `iter4a-entity` | qwen3.8-27b | 217 | — | — | нет данных |
| `iter4a-scenarios` | qwen3.8-27b | 253 | — | — | нет данных |

### Итерация 5

| Прогон | Модель | Ответов | Капсов на ответ | Капсов, тыс. | Основа |
|---|---|---|---|---|---|
| `iter5-domain` | qwen3.8-27b | 238 | 2 243 | 534 | тарифы × токены |
| `iter5-entity` | qwen3.8-27b | 217 | 2 603 | 565 | тарифы × токены |
| `iter5-scenarios` | qwen3.8-27b | 253 | 2 425 | 614 | тарифы × токены |
| `iter5-smoke` | qwen3.8-27b | 25 | 3 183 | 80 | тарифы × токены |

### Итерация 6

| Прогон | Модель | Ответов | Капсов на ответ | Капсов, тыс. | Основа |
|---|---|---|---|---|---|
| `iter6-domain/gemma-3-12b-it__bare.jsonl` | gemma-3-12b-it | 238 | 62 | 15 | смешанно |
| `iter6-domain/gemma-3-12b-it__harness.jsonl` | gemma-3-12b-it | 238 | 76 | 18 | смешанно |
| `iter6-domain/qwen3-30b-a3b__bare.jsonl` | qwen3-30b-a3b | 238 | 310 | 74 | тарифы × токены |
| `iter6-domain/qwen3-30b-a3b__harness.jsonl` | qwen3-30b-a3b | 238 | 337 | 80 | тарифы × токены |
| `iter6-facts/gemma-3-12b-it__facts.jsonl` | gemma-3-12b-it | 253 | 61 | 16 | смешанно |
| `iter6-facts/qwen3-30b-a3b__facts.jsonl` | qwen3-30b-a3b | 253 | 326 | 82 | смешанно |
| `iter6-pilot/deepseek-v4-pro__bare.jsonl` | deepseek-v4-pro | 40 | 1 848 | 74 | тарифы × токены |
| `iter6-pilot/deepseek-v4-pro__harness.jsonl` | deepseek-v4-pro | 40 | 1 841 | 74 | тарифы × токены |
| `iter6-pilot/gemma-3-12b-it__bare.jsonl` | gemma-3-12b-it | 40 | 67 | 3 | тарифы × токены |
| `iter6-pilot/gemma-3-12b-it__harness.jsonl` | gemma-3-12b-it | 40 | 74 | 3 | тарифы × токены |
| `iter6-pilot/qwen3-30b-a3b__bare.jsonl` | qwen3-30b-a3b | 40 | 346 | 14 | тарифы × токены |
| `iter6-pilot/qwen3-30b-a3b__harness.jsonl` | qwen3-30b-a3b | 40 | 330 | 13 | тарифы × токены |
| `iter6-pilot/qwen3.5-9b__bare.jsonl` | qwen3.5-9b | 40 | 891 | 36 | тарифы × токены |
| `iter6-pilot/qwen3.5-9b__harness.jsonl` | qwen3.5-9b | 40 | 1 086 | 43 | тарифы × токены |
| `iter6-pilot/qwen3.8-27b__bare.jsonl` | qwen3.8-27b | 40 | 1 971 | 79 | тарифы × токены |
| `iter6-pilot/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 40 | 2 342 | 94 | тарифы × токены |
| `iter6-rep2/gemma-3-12b-it__bare.jsonl` | gemma-3-12b-it | 253 | 60 | 15 | смешанно |
| `iter6-rep2/gemma-3-12b-it__harness.jsonl` | gemma-3-12b-it | 253 | 80 | 20 | usage.cost |
| `iter6-rep2/qwen3-30b-a3b__bare.jsonl` | qwen3-30b-a3b | 253 | 329 | 83 | usage.cost |
| `iter6-rep2/qwen3-30b-a3b__harness.jsonl` | qwen3-30b-a3b | 253 | 432 | 109 | usage.cost |
| `iter6-rep3/gemma-3-12b-it__bare.jsonl` | gemma-3-12b-it | 253 | 60 | 15 | смешанно |
| `iter6-rep3/gemma-3-12b-it__harness.jsonl` | gemma-3-12b-it | 253 | 81 | 20 | usage.cost |
| `iter6-rep3/qwen3-30b-a3b__bare.jsonl` | qwen3-30b-a3b | 253 | 329 | 83 | usage.cost |
| `iter6-rep3/qwen3-30b-a3b__harness.jsonl` | qwen3-30b-a3b | 253 | 381 | 96 | usage.cost |
| `iter6-scenarios/deepseek-v4-pro__bare.jsonl` | deepseek-v4-pro | 253 | 2 996 | 758 | тарифы × токены |
| `iter6-scenarios/deepseek-v4-pro__harness.jsonl` | deepseek-v4-pro | 253 | 2 799 | 708 | тарифы × токены |
| `iter6-scenarios/gemma-3-12b-it__bare.jsonl` | gemma-3-12b-it | 253 | 67 | 17 | смешанно |
| `iter6-scenarios/gemma-3-12b-it__harness.jsonl` | gemma-3-12b-it | 253 | 90 | 23 | смешанно |
| `iter6-scenarios/qwen3-30b-a3b__bare.jsonl` | qwen3-30b-a3b | 253 | 323 | 82 | тарифы × токены |
| `iter6-scenarios/qwen3-30b-a3b__harness.jsonl` | qwen3-30b-a3b | 253 | 395 | 100 | тарифы × токены |
| `iter6-scenarios/qwen3.5-9b__bare.jsonl` | qwen3.5-9b | 253 | 906 | 229 | тарифы × токены |
| `iter6-scenarios/qwen3.5-9b__harness.jsonl` | qwen3.5-9b | 253 | 966 | 244 | тарифы × токены |
| `iter6-scenarios/qwen3.8-27b__bare.jsonl` | qwen3.8-27b | 253 | 2 408 | 609 | тарифы × токены |
| `iter6-scenarios/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 253 | 2 419 | 612 | тарифы × токены |
| `iter6-small/gemma-3-12b-it__bare.jsonl` | gemma-3-12b-it | 253 | 67 | 17 | смешанно |
| `iter6-small/gemma-3-12b-it__harness.jsonl` | gemma-3-12b-it | 253 | 89 | 22 | смешанно |
| `iter6-small/qwen3-30b-a3b__bare.jsonl` | qwen3-30b-a3b | 253 | 323 | 82 | тарифы × токены |
| `iter6-small/qwen3-30b-a3b__harness.jsonl` | qwen3-30b-a3b | 253 | 415 | 105 | тарифы × токены |
| `iter6-tools/qwen3-30b-a3b__tools.jsonl` | qwen3-30b-a3b | 253 | 495 | 125 | usage.cost |
| `judge-pilot/claude-haiku-5.5-gemma-3-12b-it` | судья | 30 | 756 | 23 | usage.cost |
| `judge-pilot/claude-haiku-5.5-qwen3.8-27b` | судья | 30 | 840 | 25 | usage.cost |
| `judge-pilot/gemini-3-flash-preview-gemma-3-12b-it` | судья | 30 | 7 168 | 215 | usage.cost |
| `judge-pilot/gemini-3-flash-preview-qwen3.8-27b` | судья | 30 | 2 283 | 68 | usage.cost |
| `judge-pilot/gpt-oss-120b-gemma-3-12b-it` | судья | 30 | 385 | 12 | usage.cost |
| `judge-pilot/gpt-oss-120b-qwen3.8-27b` | судья | 30 | 417 | 13 | usage.cost |
| `judge-v1/deepseek-v4-pro-bare` | судья | 60 | 827 | 50 | usage.cost |
| `judge-v1/deepseek-v4-pro-harness` | судья | 60 | 845 | 51 | usage.cost |
| `judge-v1/gemma-3-12b-it-bare` | судья | 60 | 737 | 44 | usage.cost |
| `judge-v1/gemma-3-12b-it-harness` | судья | 60 | 673 | 40 | usage.cost |
| `judge-v1/main-domain` | судья | 176 | 780 | 137 | usage.cost |
| `judge-v1/main-entity` | судья | 176 | 832 | 146 | usage.cost |
| `judge-v1/main-scenarios` | судья | 181 | 885 | 160 | usage.cost |
| `judge-v1/qwen3-30b-a3b-bare` | судья | 60 | 802 | 48 | usage.cost |
| `judge-v1/qwen3-30b-a3b-harness` | судья | 60 | 837 | 50 | usage.cost |
| `judge-v1/qwen3.5-9b-bare` | судья | 60 | 934 | 56 | usage.cost |
| `judge-v1/qwen3.5-9b-harness` | судья | 60 | 864 | 52 | usage.cost |
| `judge-v1/qwen3.8-27b-bare` | судья | 60 | 908 | 54 | usage.cost |
| `judge/deepseek-v4-pro-bare` | судья | 60 | 632 | 38 | usage.cost |
| `judge/deepseek-v4-pro-harness` | судья | 60 | 691 | 41 | смешанно |
| `judge/gemma-3-12b-it-bare` | судья | 60 | 569 | 34 | usage.cost |
| `judge/gemma-3-12b-it-harness` | судья | 60 | 569 | 34 | usage.cost |
| `judge/main-domain` | судья | 176 | 638 | 112 | usage.cost |
| `judge/main-entity` | судья | 176 | 717 | 126 | usage.cost |
| `judge/main-scenarios` | судья | 181 | 688 | 124 | usage.cost |
| `judge/qwen3-30b-a3b-bare` | судья | 60 | 750 | 45 | usage.cost |
| `judge/qwen3-30b-a3b-harness` | судья | 60 | 727 | 44 | usage.cost |
| `judge/qwen3.5-9b-bare` | судья | 60 | 767 | 46 | usage.cost |
| `judge/qwen3.5-9b-harness` | судья | 60 | 750 | 45 | usage.cost |
| `judge/qwen3.8-27b-bare` | судья | 60 | 698 | 42 | usage.cost |

### Итерация 7

| Прогон | Модель | Ответов | Капсов на ответ | Капсов, тыс. | Основа |
|---|---|---|---|---|---|
| `iter7-ambiguous-human/gemma-3-12b-it__cascade.jsonl` | gemma-3-12b-it | 21 | 173 | 4 | usage.cost |
| `iter7-ambiguous-human/qwen3-30b-a3b__cascade.jsonl` | qwen3-30b-a3b | 21 | 788 | 17 | usage.cost |
| `iter7-ambiguous/gemma-3-12b-it__cascade.jsonl` | gemma-3-12b-it | 21 | 414 | 9 | usage.cost |
| `iter7-ambiguous/qwen3-30b-a3b__cascade.jsonl` | qwen3-30b-a3b | 21 | 783 | 16 | usage.cost |
| `iter7-ambiguous/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 21 | 6 290 | 132 | usage.cost |
| `iter7-domain/qwen3-30b-a3b__cascade.jsonl` | qwen3-30b-a3b | 238 | 1 082 | 257 | usage.cost |
| `iter7-domain/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 238 | 2 514 | 598 | usage.cost |
| `iter7-redmadrobot/qwen3-30b-a3b__cascade.jsonl` | qwen3-30b-a3b | 240 | 1 681 | 403 | usage.cost |
| `iter7-redmadrobot/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 240 | 4 305 | 1 033 | usage.cost |
| `iter7-scenarios/gemma-3-12b-it__cascade.jsonl` | gemma-3-12b-it | 253 | 229 | 58 | смешанно |
| `iter7-scenarios/gemma-3-12b-it__harness.jsonl` | gemma-3-12b-it | 253 | 88 | 22 | usage.cost |
| `iter7-scenarios/qwen3-30b-a3b__cascade.jsonl` | qwen3-30b-a3b | 253 | 1 262 | 319 | usage.cost |
| `iter7-scenarios/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 253 | 2 714 | 687 | usage.cost |
| `iter7b-ambiguous/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 21 | 4 773 | 100 | usage.cost |
| `iter7b-redmadrobot-docs/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 54 | 3 430 | 185 | usage.cost |
| `iter7b-scenarios/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 253 | 2 538 | 642 | usage.cost |
| `judge-iter7/base-domain` | судья | 176 | 594 | 105 | usage.cost |
| `judge-iter7/base-scenarios` | судья | 181 | 653 | 118 | usage.cost |
| `judge-iter7/new-domain` | судья | 179 | 586 | 105 | usage.cost |
| `judge-iter7/new-scenarios` | судья | 181 | 617 | 112 | usage.cost |

### Итерация 8

| Прогон | Модель | Ответов | Капсов на ответ | Капсов, тыс. | Основа |
|---|---|---|---|---|---|
| `iter8-generate/qwen3.8-27b__generate-hidden.jsonl` | qwen3.8-27b | 35 | 1 944 | 68 | usage.cost |
| `iter8-read/hidden/qwen3-30b-a3b__read.jsonl` | qwen3-30b-a3b | 35 | 362 | 13 | usage.cost |
| `iter8-read/hidden/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 3 | 6 984 | 21 | usage.cost |
| `iter8-read/hidden/qwen3.8-27b__read.jsonl` | qwen3.8-27b | 35 | 584 | 20 | usage.cost |
| `iter8-read/redmadrobot/qwen3-30b-a3b__read.jsonl` | qwen3-30b-a3b | 45 | 411 | 19 | usage.cost |
| `iter8-read/redmadrobot/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 31 | 6 626 | 205 | usage.cost |
| `iter8-read/redmadrobot/qwen3.8-27b__read.jsonl` | qwen3.8-27b | 45 | 2 067 | 93 | usage.cost |
| `iter8-v2-ambiguous/gemma-3-12b-it__cascade.jsonl` | gemma-3-12b-it | 21 | 227 | 5 | usage.cost |
| `iter8-v2-ambiguous/qwen3-30b-a3b__cascade.jsonl` | qwen3-30b-a3b | 21 | 882 | 19 | usage.cost |
| `iter8-v2-scenarios/gemma-3-12b-it__harness.jsonl` | gemma-3-12b-it | 253 | 92 | 23 | usage.cost |
| `iter8-v2-scenarios/qwen3-30b-a3b__cascade.jsonl` | qwen3-30b-a3b | 253 | 1 237 | 313 | usage.cost |
| `iter8-wide-domain/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 4 | 3 431 | 14 | usage.cost |
| `iter8-wide-redmadrobot/qwen3.8-27b__harness.jsonl` | qwen3.8-27b | 60 | 4 018 | 241 | usage.cost |

## Итерация 8: оценка и факт по шагам

| Шаг | Оценка | Факт |
|---|---|---|
| 0. Сводка затрат | 0 | 0 |
| 1. Невидимые ПДн по политике | 0 | 0 |
| 2. Детектор с низким порогом | 0 (локальная GPU) | 0 (3 набора, ~6 мин GPU) |
| 3. Поиск кодом шире | 0 | 0 |
| 3а. Агент на находках поиска кодом (64, вне плана) | — | 0.25 млн |
| 4. Набор hidden-v1: генерация | 0.1–0.2 млн | 0.07 млн |
| 5. Чтение диалога LLM: код и тесты | 0 | 0 |
| 6. Пилот чтения: 27b против qwen3-30b | ≈ 0.3 млн | 0.37 млн (чтение 0.15, разбор 0.23) |
| 7. Полный прогон лучшего варианта | 1–2 млн | 0 — отменён: пилот ответил на вопрос |
| 8. Промпт v2 на маленьких моделях и каскаде | ≈ 0.3 млн | 0.36 млн |
| 9. Сводка, отчёт, PR | 0 | 0 |

## Баланс bothub

| Когда | Баланс |
|---|---|
| итерация 5 | пополнение на 8 млн |
| пилот итерации 6 | ≈ 6 млн — с пополнения списано ≈ 2.5 млн |
| 19.11, после итерации 6 | ≈ 0.6 млн; выгрузка bothub за 19.11 — 5.49 млн |
| 19.11, итерация 6б | пополнено до ≈ 9 млн |
| 20.11, старт итерации 7 | ≈ 6.4 млн |
| 20.11, старт итерации 8 | 9.8 млн после пополнения |

Расход по счёту больше суммы по прогонам: deepseek-v4-pro днём списывается по цене провайдера
(≈ 2.4× тарифа), а генерация сценариев и прерванные запуски ответов не сохраняют.
