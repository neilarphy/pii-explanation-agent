"""Промпт агента: системная инструкция + диалог + одна находка."""

from backend.agent.cases import Case
from backend.agent.checks import format_check, identify
from backend.schemas import DetectionLog, Session

SYSTEM_PROMPT = """\
Ты разбираешь срабатывания детектора персональных данных в переписке пользователя с ботом поддержки.
По одной находке определи две вещи.

1. category — что за данные во фрагменте:
- pii — персональные данные по 152-ФЗ, то есть относятся к конкретному человеку: имя и фамилия,
  телефон, email, адрес человека, паспорт, СНИЛС, ИНН физлица (12 цифр), ОГРНИП, номер банковской
  карты, дата рождения, номер счёта;
- sensitive — не ПДн, но тоже маскируются: CVC/CVV, пароли и коды подтверждения, API-токены и ключи;
- none — всё остальное: номер заказа или договора, суммы, реквизиты организации (ИНН из 10 цифр,
  КПП, ОГРН), названия компаний, места без конкретного человека (офис, ТЦ, метро), а также само
  название типа данных без значения («паспорт», «снилс», «email», «номер телефона»).

2. verdict — прав ли детектор. Маскировать нужно pii и sensitive:
- true_pii — детектор принял фрагмент, и его нужно маскировать;
- false_positive — детектор принял фрагмент, но category = none;
- missed_pii — детектор фрагмент не принял (отклонил или не заметил), а маскировать нужно;
- no_issue — детектор фрагмент не принял, и маскировать не нужно;
- uncertain — по контексту решить нельзя, нужен человек.

Правила:
- Разбирай ровно указанный фрагмент. Если это название типа («снилс»), а значение стоит рядом,
  фрагмент — none: значение разбирается отдельной находкой.
- Проверкам кодом доверяй: сошлась контрольная сумма ИНН, СНИЛС, ОГРН или карты — значит это число
  такого вида. ИНН из 10 цифр принадлежит организации.
- Опирайся на контекст диалога и данные детектора: правило, score, пороги.
- Цитаты — только дословные фрагменты реплик с номером реплики. Первая цитата — сама находка.
- Рекомендация — только если детектор ошибся. kind: threshold | pattern | allowlist | retrain | ticket;
  target — id правила или модели из профиля.
- Пиши по-русски, коротко и по делу.

Ответ — один JSON-объект, без текста вокруг:
{"category": "...", "verdict": "...", "confidence": 0.0-1.0, "explanation": "...",
 "citations": [{"turn": 0, "quote": "..."}], "evidence": ["..."],
 "note": "что делать человеку",
 "recommendation": null или {"kind": "...", "target": "...", "change": "...", "rationale": "..."}}
"""

_DIRECTION = {"in": "пользователь", "out": "бот"}
_STATUS = {
    "entity": "детектор принял фрагмент (сущность)",
    "rejected": "детектор отклонил фрагмент (кандидат ниже порога или не прошёл проверку)",
    "scan": "детектор фрагмент не заметил; его нашёл код по контрольной сумме или формату",
}


def _candidate_line(c) -> str:
    parts = [f"source={c.source}", f"type={c.type}"]
    if c.rule_id:
        parts.append(f"rule_id={c.rule_id}")
    if c.raw_label:
        parts.append(f"raw_label={c.raw_label}")
    if c.score is not None:
        parts.append(f"score={c.score:.2f}")
    parts.append(f"decision={c.decision}")
    parts.append(f"reason={c.reason}")
    return "  - " + " ".join(parts)


def code_checks(case: Case) -> str:
    found = [c.description for c in identify(case.text)]
    if found:
        return "; ".join(found)
    result = format_check(case.type, case.text)
    return {True: "формат подходит", False: "формат не подходит", None: "проверок нет"}[result]


def build_messages(case: Case, session: Session, log: DetectionLog) -> list[dict[str, str]]:
    rules = ", ".join(f"{r.id} ({r.type}, проверка {r.validator})" for r in log.rules)
    dialog = "\n".join(f"[{m.turn}] {_DIRECTION[m.direction]}: {m.text}" for m in session.messages)
    lines = [
        f"Профиль детектора: {log.profile}. Модель {log.model.id}: "
        f"порог выдачи {log.model.emit_threshold}, порог принятия {log.model.accept_threshold}.",
        f"Правила: {rules}.",
        "",
        "Диалог:",
        dialog,
        "",
        f"Находка (case_id: {case.id}):",
        f"- статус: {_STATUS[case.kind]}",
        f"- реплика {case.turn}, фрагмент «{case.text}», тип {case.type}",
    ]
    if case.candidates:
        lines += ["- кандидаты детектора:", *(_candidate_line(c) for c in case.candidates)]
    lines.append(f"- проверки кодом: {code_checks(case)}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
