"""Проверки кодом: формат находки и дословность цитат. LLM им не доверяем на слово."""

import re

from backend.schemas import Citation, Session

_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_PHONE_EXTENSION = re.compile(r"(доб\.?|ext\.?|extension)", re.IGNORECASE)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def format_check(pii_type: str, text: str) -> bool | None:
    """True/False — формат подходит/не подходит; None — для этого типа проверки нет."""
    if pii_type == "EMAIL":
        return bool(_EMAIL.match(text.strip()))
    if pii_type == "PHONE":
        digits = re.sub(r"\D", "", _PHONE_EXTENSION.split(text)[0])
        if len(digits) == 11 and digits[0] in "78":
            return True  # российский номер: 7/8 + 10 цифр
        return 10 <= len(digits) <= 15  # международный по E.164
    if pii_type == "PAYMENT_CARD":
        digits = re.sub(r"\D", "", text)
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    return None


def resolve_citations(raw: list[dict], session: Session) -> tuple[list[Citation], list[str]]:
    """LLM даёт (реплика, цитата); офсеты считаем сами. Недословные цитаты отбрасываем."""
    texts = {m.turn: m.text for m in session.messages}
    citations: list[Citation] = []
    dropped: list[str] = []
    for item in raw:
        turn, quote = item.get("turn"), item.get("quote") or ""
        text = texts.get(turn) if isinstance(turn, int) else None
        start = text.find(quote) if text is not None and quote else -1
        if start < 0:
            dropped.append(f"реплика {turn}: «{quote}»")
            continue
        citations.append(Citation(turn=turn, start=start, end=start + len(quote), quote=quote))
    return citations, dropped
