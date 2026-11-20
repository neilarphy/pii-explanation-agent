"""Промпт агента: системная инструкция + диалог + одна находка."""

from backend.agent.cases import Case
from backend.agent.checks import CodeCheck, Joined, format_check, identify, join_across_turns, scan
from backend.schemas import DetectionLog, Session

SYSTEM_PROMPT = """\
Ты разбираешь срабатывания детектора персональных данных в переписке пользователя с ботом поддержки.
По одной находке определи две вещи.

1. category — что за данные во фрагменте:
- pii — персональные данные по 152-ФЗ, то есть относятся к конкретному человеку: имя и фамилия,
  телефон, email, адрес человека, паспорт, СНИЛС, ИНН физлица (12 цифр), ОГРНИП, номер банковской
  карты, дата рождения, номер счёта, а также другие документы человека — полис ОМС, свидетельство о
  рождении, военный билет, водительское удостоверение (в том числе одна серия документа);
- sensitive — не ПДн, но тоже маскируются: CVC/CVV, пароли и коды подтверждения, API-токены и ключи;
- none — всё остальное: номер заказа или договора, суммы, реквизиты организации (ИНН из 10 цифр,
  КПП, ОГРН), названия компаний, места без конкретного человека (офис, ТЦ, метро), а также само
  название типа данных без значения («паспорт», «снилс», «email», «номер телефона»).

2. verdict — прав ли детектор. Маскировать нужно pii и sensitive:
- true_pii — детектор принял фрагмент, и его нужно маскировать;
- false_positive — детектор принял фрагмент, но category = none;
- missed_pii — детектор фрагмент не принял (отклонил или не заметил), а маскировать нужно;
- no_issue — детектор фрагмент не принял, и маскировать не нужно;
- uncertain — по контексту решить нельзя, нужен человек. Например, число по форме как номер карты,
  а что это за номер — карта, заказ, договор — в диалоге не сказано и из смысла не следует.
  Не угадывай: такой ответ честнее, чем уверенный, но случайный.

Правила:
- Разбирай ровно указанный фрагмент. Если это название типа («снилс»), а значение стоит рядом,
  фрагмент — none: значение разбирается отдельной находкой.
- Проверки кодом говорят о форме значения, а не о смысле. Сошлась контрольная сумма — число может
  быть таким номером, но решает контекст: номер заказа, договора, накладной или трек-номер с верной
  суммой Луна — не карта (none). Число, записанное как паспорт (4 цифры, пробел, 6 цифр), —
  паспорт, даже если сходится сумма ИНН.
- Реквизиты: ИНН из 10 цифр и ОГРН — организация (none). ИНН из 12 цифр и ОГРНИП — человек, в том
  числе индивидуальный предприниматель, — это pii и в деловом контексте («застройщик», «поставщик»).
- Если код показал, что фрагмент — часть целого значения (телефон, email или адрес целиком, в том
  числе продиктованный по частям в разных репликах), оценивай значение целиком: часть телефона или
  email — тоже pii, а неполные границы отметь в note.
- Опирайся на контекст диалога и данные детектора: правило, score, пороги.
- Цитаты — только дословные фрагменты реплик с номером реплики. Первая цитата — сама находка.
- Рекомендация — только если детектор ошибся (false_positive или missed_pii), иначе null. Адресуй её
  тому, кто ошибся: ложное срабатывание — правилу (rule_id) или модели, которые фрагмент приняли;
  пропуск — тому, кто его отклонил, или новый шаблон (pattern, target — id нового правила или профиля).
  kind: threshold и retrain — только модели; pattern — правилу или исключение на выходе модели;
  allowlist — существующему правилу или модели; ticket — передать людям. target — id из профиля.
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


def enclosing_value(case: Case, session: Session) -> tuple[str, CodeCheck] | None:
    """Целое значение, найденное кодом, частью которого является фрагмент («dima» в email словами)."""
    text = next((m.text for m in session.messages if m.turn == case.turn), "")
    for start, end, check in scan(text):
        if start < case.end and case.start < end and end - start > case.end - case.start:
            return text[start:end], check
    return None


def joined_value(case: Case, session: Session) -> Joined | None:
    """Значение, продиктованное по частям в разных репликах («89164» … «855707»)."""
    return join_across_turns(session, case.turn, case.start, case.end)


def code_facts(case: Case, session: Session) -> list[str]:
    """Что код знает о фрагменте: проверки, целое значение вокруг, склейка по репликам."""
    lines = [f"- проверки кодом: {code_checks(case)}"]
    if whole := enclosing_value(case, session):
        value, check = whole
        lines.append(
            f"- фрагмент — часть целого значения, найденного кодом: «{value}» ({check.description})"
        )
    if joined := joined_value(case, session):
        parts = " + ".join(f"«{text}» (реплика {turn})" for turn, text in joined.parts)
        lines.append(
            f"- фрагмент — часть значения, продиктованного по частям: {parts} = {joined.value} "
            f"({joined.check.description})"
        )
    return lines


def build_messages(
    case: Case, session: Session, log: DetectionLog, facts: bool = True
) -> list[dict[str, str]]:
    """facts=False — без фактов от кода (только диалог и данные детектора), для абляции."""
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
    if facts:
        lines += code_facts(case, session)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
