# Разметка людьми

- `judge-kappa-v1/` — 60 ответов агента для проверки LLM-судьи ([задача](../../docs/tasks/taisia-1-judge-kappa.md)).
  `sheet.csv` — заполнить и сохранить как `sheet-<имя>.csv`; `key.json` — чей ответ, разметчику не открывать.
- `test-v1/` — пример формата замороженного тестового набора ([задача](../../docs/tasks/taisia-2-test-set.md)).
- `operator-study-v1/` — опыт с операторами ([задача](../../docs/tasks/team-4-operator-study.md)).
  `study.html` — участникам; `key.json` — эталон, участникам не отправлять.

Собрано скриптами `scripts/review.py`, `scripts/testset.py`, `scripts/study.py` из сохраненных ответов
агента (команды — в docstring скриптов).
