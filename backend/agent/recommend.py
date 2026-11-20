"""Рекомендация агента против того, что код знает о срабатывании.

Источник срабатывания записан в логе детектора: кандидат пришёл от модели (`source=model`)
или от регулярки (`rule_id`). Правка должна адресоваться тому, кто ошибся:
- ложное срабатывание — тому, кто фрагмент принял;
- пропуск — тому, кто фрагмент отклонил, либо новый шаблон (`pattern`);
- `threshold` и `retrain` бывают только у модели, `allowlist` — у существующего правила или модели.

Проверено на 600 рекомендациях, которые оценил судья (итерация 7): объективные несоответствия
(вид правки не для этого адресата, правка пропуска не тому, кто его отклонил) судья отклонил
28 раз из 28; правку ложного срабатывания не тому, кто его принял, — 31 раз из 44.
`pattern` для модели (исключение на её выходе) и для профиля (новое правило) — допустимы.
"""

from backend.agent.cases import Case
from backend.schemas import DetectionLog

FIXES = ("false_positive", "missed_pii")  # вердикты, при которых детектор ошибся


def _source(candidate, log: DetectionLog) -> str:
    return log.model.id if candidate.source == "model" else (candidate.rule_id or "")


def recommendation_problems(
    rec: dict | None, verdict: str | None, case: Case, log: DetectionLog
) -> list[str]:
    """Пусто — рекомендация согласуется с логом детектора (или её нет)."""
    if not rec or verdict not in FIXES or rec.get("kind") == "ticket":
        return []
    kind, target = rec.get("kind"), rec.get("target")
    model, rules = log.model.id, {r.id for r in log.rules}
    found = []
    if kind in ("threshold", "retrain") and target != model:
        found.append(f"{kind} бывает только у модели {model}, а target — «{target}»")
    if kind == "allowlist" and target not in rules | {model}:
        found.append(
            f"allowlist — для существующего правила или модели ({', '.join(sorted(rules | {model}))}), "
            f"а target — «{target}»"
        )
    if verdict == "false_positive":
        accepted = {_source(c, log) for c in case.candidates if c.decision == "accepted"}
        if accepted and target not in accepted:
            found.append(
                f"фрагмент принял {', '.join(sorted(accepted))} — правка должна адресоваться ему, "
                f"а target — «{target}»"
            )
    if verdict == "missed_pii" and case.candidates and kind != "pattern":
        rejected = {_source(c, log) for c in case.candidates}
        if target not in rejected:
            found.append(
                f"фрагмент отклонил {', '.join(sorted(rejected))} — правка должна адресоваться ему "
                f"(или новый pattern), а target — «{target}»"
            )
    return found
