# Итерация 4: сравнение с итерацией 3 по каждой находке

Те же находки, тот же эталон (для сценариев — с тремя дописанными спанами из
`data/scenarios/gold-fixes-v1.json`). В итерации 3 код агента не менялся, поэтому для pii-bench
взяты ответы прогона PR #4 с его пересчётом постобработки (понижение до uncertain отменено для
не-pii — так получены 219 и 155 из описания PR). Находки кода, которых не было в прошлой выборке,
здесь не сравниваются — они в отчётах наборов.

| Набор | Итерация 3 | Итерация 4 | Улучшилось | Ухудшилось |
|---|---|---|---|---|
| Свои сценарии v1 | 219/238 (92.0%) | 228/238 (95.8%) | 9 | 0 |
| pii-bench domain | 219/226 (96.9%) | 220/226 (97.3%) | 1 | 0 |
| pii-bench entity | 155/159 (97.5%) | 155/159 (97.5%) | 0 | 0 |

## Свои сценарии v1

| Находка | Фрагмент | Эталон | Итерация 3 | Итерация 4 |
|---|---|---|---|---|
| `sc-order_luhn-1-t0-e2` | «7607305997599779» | false_positive | true_pii | false_positive |
| `sc-order_luhn-1-t0-e1` | «7607305997599779» | false_positive | true_pii | false_positive |
| `sc-split_phone-2-t0-e1` | «89164» | true_pii | false_positive | true_pii |
| `sc-split_phone-4-t0-e1` | «89710» | true_pii | false_positive | true_pii |
| `sc-injection-5-t0-e1` | «7083 620142» | true_pii | false_positive | true_pii |
| `sc-obfuscated_email-1-t0-e2` | «mail точка ru» | true_pii | false_positive | true_pii |
| `sc-obfuscated_email-4-t0-e3` | «mail точка ru» | true_pii | false_positive | true_pii |
| `sc-obfuscated_email-4-t0-e2` | «точка» | true_pii | false_positive | true_pii |
| `sc-obfuscated_email-2-t0-e1` | «dima» | true_pii | false_positive | true_pii |

## pii-bench domain

| Находка | Фрагмент | Эталон | Итерация 3 | Итерация 4 |
|---|---|---|---|---|
| `pb-st_044-t0-c6` | «Екатеринбург» | missed_pii | no_issue | missed_pii |
