"""Промпт агента: системная инструкция + диалог + одна находка."""

from backend.agent.cases import Case
from backend.agent.checks import format_check
from backend.schemas import DetectionLog, Session

SYSTEM_PROMPT = """\
Ты разбираешь срабатывания детектора персональных данных (ПДн) в переписке пользователя с ботом поддержки.
Тебе дают диалог и одну находку детектора. Реши, что это:
- true_pii — детектор нашёл настоящие ПДн, маскирование верное;
- false_positive — детектор сработал, но это не ПДн (номер заказа, название компании, пример и т. п.);
- missed_pii — детектор отклонил кандидата, но это настоящие ПДн (пропуск);
- uncertain — по контексту решить нельзя, нужен человек;
- no_issue — только для отклонённого кандидата: это не ПДн, детектор прав.

Правила:
- Опирайся на контекст диалога и данные детектора: правило, score, пороги, проверку формата кодом.
- Цитаты — только дословные фрагменты реплик с номером реплики. Первая цитата — сама находка.
- Рекомендация — только если детектор ошибся. kind: threshold | pattern | allowlist | retrain | ticket;
  target — id правила или модели из профиля.
- Пиши по-русски, коротко и по делу.

Ответ — один JSON-объект, без текста вокруг:
{"verdict": "...", "confidence": 0.0-1.0, "explanation": "...",
 "citations": [{"turn": 0, "quote": "..."}], "evidence": ["..."],
 "note": "что делать человеку",
 "recommendation": null или {"kind": "...", "target": "...", "change": "...", "rationale": "..."}}
"""

_DIRECTION = {"in": "пользователь", "out": "бот"}
_CHECK = {True: "пройдена", False: "не пройдена", None: "для этого типа нет"}


def _candidate_line(c) -> str:
    parts = [f"source={c.source}"]
    if c.rule_id:
        parts.append(f"rule_id={c.rule_id}")
    if c.raw_label:
        parts.append(f"raw_label={c.raw_label}")
    if c.score is not None:
        parts.append(f"score={c.score:.2f}")
    parts.append(f"decision={c.decision}")
    parts.append(f"reason={c.reason}")
    return "  - " + " ".join(parts)


def build_messages(case: Case, session: Session, log: DetectionLog) -> list[dict[str, str]]:
    rules = ", ".join(f"{r.id} ({r.type}, проверка {r.validator})" for r in log.rules)
    dialog = "\n".join(f"[{m.turn}] {_DIRECTION[m.direction]}: {m.text}" for m in session.messages)
    status = (
        "сущность, детектор её принял"
        if case.kind == "entity"
        else "кандидат, детектор его отклонил"
    )
    user = "\n".join(
        [
            f"Профиль детектора: {log.profile}. Модель {log.model.id}: "
            f"порог выдачи {log.model.emit_threshold}, порог принятия {log.model.accept_threshold}.",
            f"Правила: {rules}.",
            "",
            "Диалог:",
            dialog,
            "",
            f"Находка (case_id: {case.id}):",
            f"- статус: {status}",
            f"- реплика {case.turn}, фрагмент «{case.text}», тип {case.type}",
            "- кандидаты детектора:",
            *(_candidate_line(c) for c in case.candidates),
            f"- проверка формата кодом: {_CHECK[format_check(case.type, case.text)]}",
        ]
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
