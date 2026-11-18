"""Проверки кодом: формат и контрольные суммы, поиск номеров по тексту, дословность цитат.

LLM им не доверяем на слово: что число — ИНН или карта, решает контрольная сумма,
а не модель.
"""

import re
from dataclasses import dataclass

from backend.schemas import Category, Citation, Session

_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_PHONE_EXTENSION = re.compile(r"(доб\.?|ext\.?|extension)", re.IGNORECASE)
_PASSPORT = re.compile(r"^\d{2} ?\d{2} ?\d{6}$")
_NUMBER = re.compile(r"(?<![\w+])\+?\d(?:[\d \-()]{7,24}\d)(?!\w)")
_TOKEN = re.compile(r"(?<![\w-])(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[A-Za-z0-9_-]{20,}(?![\w-])")

# Значения, которые детектор не видит как число: адрес с улицей и домом, цифры словами, email
# словами. Шаблоны общие (маркеры улицы и дома, слова-цифры), а не под конкретный генератор.
_STREET = (
    r"(?:ул\.?|улица|пр-т|просп\.?|проспект|пер\.?|переулок|ш\.|шоссе|б-р|бульвар|наб\.?"
    r"|набережная|пл\.|площадь|мкр\.?|микрорайон|проезд|аллея|тракт)"
)
_ADDRESS = re.compile(
    r"(?:(?:г\.|город)\s*[А-ЯЁ][\w-]+(?:\s[А-ЯЁ][\w-]+)?,?\s*)?"
    rf"(?i:{_STREET})\s*[А-ЯЁ0-9][\w.-]*(?:\s[А-ЯЁ0-9][\w.-]*){{0,3}},?\s*"
    r"(?i:д\.?|дом)\s*\d+(?:/\d+)?(?:[а-яА-Я](?![а-яА-Я\d]))?"
    r"(?:,?\s*(?i:корп\.?|корпус|к\.?|стр\.?|строение)\s*\d+)?"
    r"(?:,?\s*(?i:кв\.?|квартира)\s*\d+)?"
)
_DIGIT_WORDS = {
    "ноль": "0", "нуль": "0", "один": "1", "одна": "1", "два": "2", "две": "2", "три": "3",
    "четыре": "4", "пять": "5", "шесть": "6", "семь": "7", "восемь": "8", "девять": "9",
}  # fmt: skip
_DIGIT_WORD = "(?:" + "|".join(_DIGIT_WORDS) + ")"
_PHONE_WORDS = re.compile(
    rf"(?i)(?<!\w)(?:плюс\s+)?{_DIGIT_WORD}(?:[\s,-]+{_DIGIT_WORD}){{9,14}}(?!\w)"
)
_AT_WORD = r"(?:собака|собачка|at|\(at\)|\[at\])"
_DOT_WORD = r"(?:точка|dot|\(dot\)|\[dot\])"
_WORD_EMAIL = re.compile(
    rf"(?i)(?<![\w.])[a-z0-9_-]+(?:\s*(?:{_DOT_WORD}|\.)\s*[a-z0-9_-]+)*"
    rf"\s+{_AT_WORD}\s+[a-z0-9-]+(?:\s*(?:{_DOT_WORD}|\.)\s*[a-z0-9-]+)*\s+{_DOT_WORD}\s+[a-z]{{2,6}}"
    r"(?![\w])"
)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def words_to_digits(text: str) -> str:
    """«восемь девять один ...» → «891...»; слова, которые не цифры, пропускаются."""
    return "".join(_DIGIT_WORDS.get(w, "") for w in re.findall(r"\w+", text.lower()))


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _weighted(digits: str, weights: list[int]) -> int:
    return sum(int(d) * w for d, w in zip(digits, weights, strict=False)) % 11 % 10


def inn_ok(digits: str) -> bool:
    """ИНН: 10 цифр — организация, 12 — физлицо или ИП."""
    if len(digits) == 10:
        return _weighted(digits, [2, 4, 10, 3, 5, 9, 4, 6, 8]) == int(digits[9])
    if len(digits) == 12:
        c11 = _weighted(digits, [7, 2, 4, 10, 3, 5, 9, 4, 6, 8])
        c12 = _weighted(digits, [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8])
        return c11 == int(digits[10]) and c12 == int(digits[11])
    return False


def snils_ok(digits: str) -> bool:
    if len(digits) != 11:
        return False
    total = sum(int(d) * (9 - i) for i, d in enumerate(digits[:9]))
    check = total if total < 100 else (0 if total in (100, 101) else total % 101)
    return (0 if check == 100 else check) == int(digits[9:])


def ogrn_ok(digits: str) -> bool:
    """ОГРН — 13 цифр (организация), ОГРНИП — 15 цифр (индивидуальный предприниматель)."""
    if len(digits) == 13:
        return int(digits[:12]) % 11 % 10 == int(digits[12])
    if len(digits) == 15:
        return int(digits[:14]) % 13 % 10 == int(digits[14])
    return False


@dataclass(frozen=True)
class CodeCheck:
    """Что код узнал во фрагменте: метка, публичный тип детектора и категория данных."""

    label: str
    type: str
    category: Category
    description: str


_CHECKS = {
    "CARD": CodeCheck("CARD", "PAYMENT_CARD", Category.PII, "номер карты: Луна сходится"),
    "OGRNIP": CodeCheck(
        "OGRNIP", "NATIONAL_ID", Category.PII, "ОГРНИП: контрольная сумма сходится"
    ),
    "OGRN": CodeCheck(
        "OGRN", "COMPANY_ID", Category.NONE, "ОГРН организации: контрольная сумма сходится"
    ),
    "INN_PERSON": CodeCheck(
        "INN_PERSON",
        "NATIONAL_ID",
        Category.PII,
        "ИНН физлица (12 цифр): контрольная сумма сходится",
    ),
    "SNILS": CodeCheck("SNILS", "NATIONAL_ID", Category.PII, "СНИЛС: контрольная сумма сходится"),
    "INN_COMPANY": CodeCheck(
        "INN_COMPANY",
        "COMPANY_ID",
        Category.NONE,
        "ИНН организации (10 цифр): контрольная сумма сходится",
    ),
    "PASSPORT": CodeCheck("PASSPORT", "PASSPORT", Category.PII, "формат паспорта РФ: 4 + 6 цифр"),
    "PASSPORT_GROUPED": CodeCheck(
        "PASSPORT_GROUPED",
        "PASSPORT",
        Category.PII,
        "записано как паспорт РФ: серия 4 цифры, пробел, номер 6 цифр",
    ),
    "PHONE": CodeCheck("PHONE", "PHONE", Category.PII, "формат телефона РФ: 7/8 + 10 цифр"),
    "TOKEN": CodeCheck(
        "TOKEN",
        "SECRET",
        Category.SENSITIVE,
        "похоже на токен или ключ: 20+ символов из букв и цифр",
    ),
    "ADDRESS": CodeCheck(
        "ADDRESS", "ADDRESS", Category.PII, "адрес по шаблону: улица и номер дома"
    ),
    "PHONE_WORDS": CodeCheck(
        "PHONE_WORDS", "PHONE", Category.PII, "телефон РФ, записанный цифрами-словами"
    ),
    "EMAIL_WORDS": CodeCheck(
        "EMAIL_WORDS", "EMAIL", Category.PII, "email, записанный словами («собака», «точка»)"
    ),
}


def identify(text: str) -> list[CodeCheck]:
    """Все проверки, которые фрагмент проходит; порядок — от более надёжной к менее."""
    raw, digits = text.strip(), _digits(text)
    found = []
    if 13 <= len(digits) <= 19 and luhn_ok(digits) and not raw.startswith("+"):
        found.append("CARD")
    if len(digits) == 15 and ogrn_ok(digits):
        found.append("OGRNIP")
    if len(digits) == 13 and ogrn_ok(digits):
        found.append("OGRN")
    if len(digits) == 12 and inn_ok(digits):
        found.append("INN_PERSON")
    if len(digits) == 11 and snils_ok(digits):
        found.append("SNILS")
    if len(digits) == 10 and inn_ok(digits):
        found.append("INN_COMPANY")
    if _PASSPORT.match(raw):
        # «4510 123456» записан как паспорт — это сильнее случайно сошедшейся суммы ИНН-10
        found.insert(0, "PASSPORT_GROUPED") if " " in raw else found.append("PASSPORT")
    phone = _digits(_PHONE_EXTENSION.split(raw)[0])
    if len(phone) == 11 and phone[0] in "78" and len(digits) <= 13:
        found.append("PHONE")
    if _TOKEN.fullmatch(raw):
        found.append("TOKEN")
    if _ADDRESS.fullmatch(raw):
        found.append("ADDRESS")
    if _PHONE_WORDS.fullmatch(raw) and _is_phone(words_to_digits(raw)):
        found.append("PHONE_WORDS")
    if _WORD_EMAIL.fullmatch(raw):
        found.append("EMAIL_WORDS")
    return [_CHECKS[label] for label in found]


def _is_phone(digits: str) -> bool:
    return (len(digits) == 11 and digits[0] in "78") or (len(digits) == 10 and digits[0] == "9")


def scan(text: str) -> list[tuple[int, int, CodeCheck]]:
    """Номера, токены, адреса и значения словами, которые прошли хотя бы одну проверку.

    Если длинный фрагмент не проходит ни одной проверки (два числа подряд через
    пробел), пробуем его части по пробелам.
    """
    hits: list[tuple[int, int, CodeCheck]] = []
    for m in _NUMBER.finditer(text):
        if len(_digits(m.group())) < 9:
            continue
        checks = identify(m.group())
        if checks:
            hits.append((m.start(), m.end(), checks[0]))
            continue
        for part in re.finditer(r"\S+", m.group()):
            if len(_digits(part.group())) >= 9 and (checks := identify(part.group())):
                hits.append((m.start() + part.start(), m.start() + part.end(), checks[0]))
    hits += [(m.start(), m.end(), _CHECKS["TOKEN"]) for m in _TOKEN.finditer(text)]
    hits += [(m.start(), m.end(), _CHECKS["ADDRESS"]) for m in _ADDRESS.finditer(text)]
    hits += [
        (m.start(), m.end(), _CHECKS["PHONE_WORDS"])
        for m in _PHONE_WORDS.finditer(text)
        if _is_phone(words_to_digits(m.group()))
    ]
    hits += [(m.start(), m.end(), _CHECKS["EMAIL_WORDS"]) for m in _WORD_EMAIL.finditer(text)]
    return sorted(hits)


_PART = re.compile(r"(?<![\w@.+\-])\d{2,}(?![\w@\-]|[.,]\d)")


@dataclass(frozen=True)
class Joined:
    """Значение, продиктованное по частям в нескольких репликах пользователя."""

    value: str
    parts: tuple[tuple[int, str], ...]  # (реплика, часть)
    check: CodeCheck


def join_across_turns(
    session: Session, turn: int, start: int, end: int, max_parts: int = 3
) -> Joined | None:
    """Склеивает фрагмент с соседними группами цифр из других реплик пользователя.

    «Номер начинается с 89164» … «дальше 855707»: по отдельности ни одна часть не проходит
    проверку формата, а вместе — телефон. Склеиваем только подряд идущие группы, только из
    реплик пользователя и только если части стоят в разных репликах (в одной реплике
    значение целиком находит scan). Частью считается отдельно стоящая группа цифр (не год
    в email, не кусок слова), которая сама по себе не проходит проверку: целое значение —
    не часть. Сам фрагмент должен быть ровно одной такой группой.
    """
    groups = [
        (m.turn, g.start(), g.end(), g.group())
        for m in session.messages
        if m.direction == "in"
        for g in _PART.finditer(m.text)
        if not identify(g.group())
    ]
    hit = [i for i, (t, s, e, _) in enumerate(groups) if t == turn and s < end and start < e]
    if len(hit) != 1:
        return None
    i = hit[0]
    text = next(m.text for m in session.messages if m.turn == turn)
    if text[start:end].strip(" .,:;") != groups[i][3]:
        return None
    for size in range(2, max_parts + 1):
        for first in range(max(0, i - size + 1), min(i, len(groups) - size) + 1):
            window = groups[first : first + size]
            if len({t for t, *_ in window}) < 2:
                continue
            value = "".join(digits for *_, digits in window)
            if checks := identify(value):
                return Joined(value, tuple((t, d) for t, _, _, d in window), checks[0])
    return None


def format_check(pii_type: str, text: str) -> bool | None:
    """True/False — формат подходит/не подходит; None — для этого типа проверки нет."""
    if pii_type == "EMAIL":
        return bool(_EMAIL.match(text.strip()))
    if pii_type == "PHONE":
        digits = _digits(_PHONE_EXTENSION.split(text)[0])
        if len(digits) == 11 and digits[0] in "78":
            return True  # российский номер: 7/8 + 10 цифр
        return 10 <= len(digits) <= 15  # международный по E.164
    if pii_type == "PAYMENT_CARD":
        digits = _digits(text)
        return 13 <= len(digits) <= 19 and luhn_ok(digits)
    if pii_type in ("NATIONAL_ID", "PASSPORT", "COMPANY_ID", "SECRET"):
        return bool(identify(text))
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
