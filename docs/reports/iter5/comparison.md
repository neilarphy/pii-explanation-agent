# Итерация 5: харнесс против конвейера итерации 4 по каждой находке

Те же находки, тот же эталон, та же модель (`qwen3.8-27b`, temperature 0). Итерация 4 — конвейер
(один вызов LLM, понижение по фрагменту), итерация 5 — харнесс на LangGraph с флагами по умолчанию
(факты от кода и проверка ответа включены, инструменты выключены).

| Набор | Итерация 4 | Итерация 5 | Улучшилось | Ухудшилось |
|---|---|---|---|---|
| Свои сценарии v1 | 243/253 (96.0%) | 252/253 (99.6%) | 9 | 0 |
| pii-bench domain | 232/238 (97.5%) | 231/238 (97.1%) | 1 | 2 |
| pii-bench entity | 213/217 (98.2%) | 211/217 (97.2%) | 0 | 2 |

**Перепроверка расхождений на pii-bench** (каждая находка по 3 раза, конвейер и харнесс на текущем
коде; промпт у них для этих находок одинаковый — склейки там нет):

| Находка | Конвейер | Харнесс |
|---|---|---|
| `pb-s_auto_013-t0-c1` «Московская область, г. Химки» | missed ×3 | missed ×3 |
| `pb-addr_042-t0-c1` «Тула» | missed ×3 | missed ×3 |
| `pb-st_054-t0-c3` «829461» | missed ×3 | missed ×3 |
| `pb-dialog_073-t2-e2` «Гагарина 25» | fp, true, fp | true, fp, true |
| `pb-phone_050-t0-e1` «8-812-777-88-99» | ошибка bothub, true, true | true, fp, fp |

Первые три в полном прогоне — разовый сбой модели: при перезапуске верно все 6 раз из 6. Две
последние модель решает то так, то так в обоих режимах — номер записи в клинику и «не тот» адрес
доставки действительно спорные. Регрессий от харнесса нет; точность на pii-bench ±2 находки — шум
модели при temperature 0.

## Свои сценарии v1

| Находка | Фрагмент | Эталон | Итерация 4 | Итерация 5 |
|---|---|---|---|---|
| `sc-obfuscated_email-1-t0-c3` | «точка sokolova52» | missed_pii | uncertain | missed_pii |
| `sc-obfuscated_email-2-t0-c5` | «yandex точка ru» | missed_pii | uncertain | missed_pii |
| `sc-obfuscated_email-3-t0-c2` | «gmail» | missed_pii | uncertain | missed_pii |
| `sc-obfuscated_email-4-t0-c3` | «smirnov37» | missed_pii | uncertain | missed_pii |
| `sc-obfuscated_email-5-t0-c5` | «mail точка ru» | missed_pii | uncertain | missed_pii |
| `sc-split_phone-2-t2-c1` | «855707» | missed_pii | uncertain | missed_pii |
| `sc-split_phone-3-t0-c1` | «89170» | missed_pii | uncertain | missed_pii |
| `sc-split_phone-4-t2-c2` | «953796» | missed_pii | uncertain | missed_pii |
| `sc-split_phone-5-t2-c1` | «359899» | missed_pii | uncertain | missed_pii |

## pii-bench domain

| Находка | Фрагмент | Эталон | Итерация 4 | Итерация 5 |
|---|---|---|---|---|
| `pb-dialog_073-t2-e2` | «Гагарина 25» | false_positive | true_pii | false_positive |
| `pb-s_auto_013-t0-c1` | «Московская область, г. Химки» | missed_pii | missed_pii | no_issue |
| `pb-st_054-t0-c3` | «829461» | missed_pii | missed_pii | uncertain |

## pii-bench entity

| Находка | Фрагмент | Эталон | Итерация 4 | Итерация 5 |
|---|---|---|---|---|
| `pb-addr_042-t0-c1` | «Тула» | missed_pii | missed_pii | no_issue |
| `pb-phone_050-t0-e1` | «8-812-777-88-99» | true_pii | true_pii | false_positive |
