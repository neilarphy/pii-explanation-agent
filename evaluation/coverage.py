"""Какие ПДн эталона агент вообще видит — без вызовов LLM.

Агент разбирает только находки: сущности детектора, отклонённые кандидаты и то, что нашёл код.
ПДн эталона, которых нет ни в одной находке, для агента невидимы — пропуск, о котором никто не
узнает. Здесь они считаются и делятся по нашей политике (`docs/pii-policy.md`): часть из них — не
ПДн (отдельный город, IP-адрес, заглушки вида «111.222.333.44»), часть — типы, которых нет в
профиле детектора, остальное — настоящие пропуски.

Способ найти невидимое (порог детектора, код, чтение диалога LLM) сравнивается по тому, сколько
невидимых он делает видимыми и сколько добавляет лишних находок.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from evaluation.pii_bench import GoldSpan

# Классы невидимых ПДн
CITY = "место без адреса"  # «Вольск», «в Лондоне» — по политике none
IP = "IP-адрес"  # по политике none, открытый вопрос к HiveTrace
PLACEHOLDER = "заглушка"  # «111.222.333.44», «123456» — примерные значения, none
OUT_OF_PROFILE = "тип вне профиля"  # права, свидетельство, военный билет, ОМС: детектор не ищет
REAL = "пропуск детектора"  # тип в профиле, значение настоящее
POLICY_NONE = (CITY, IP, PLACEHOLDER)

_STREET = re.compile(
    r"\b(ул|улиц|пр|просп|пер|переул|бул|ш|шоссе|наб|пл|площад|д|дом|кв|корп|стр|мкр|st|street|"
    r"ave|road|rd|lane|ln|drive|dr)\b",
    re.IGNORECASE,
)


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def is_placeholder(text: str) -> bool:
    """Последовательность (123456, 987654) или повторы (111222333, 001122) — примерное значение."""
    d = _digits(text)
    if len(d) < 4:
        return False
    if d in "0123456789" * 3 or d in "9876543210" * 3:
        return True
    return re.fullmatch(r"(?:(\d)\1+)+", d) is not None


def classify(g: GoldSpan, profile_types: set[str]) -> str:
    if g.type == "IP_ADDRESS":
        return IP
    if g.type == "ADDRESS" and not _digits(g.text) and not _STREET.search(g.text):
        return CITY
    if is_placeholder(g.text):
        return PLACEHOLDER
    if (g.detector_type or "").lower() not in profile_types:
        return OUT_OF_PROFILE
    return REAL


def _overlaps(turn: int, start: int, end: int, g: GoldSpan) -> bool:
    return turn == g.turn and start < g.end and g.start < end


@dataclass
class Coverage:
    total: int = 0  # ПДн в эталоне (pii и sensitive)
    visible: int = 0
    invisible: Counter = field(default_factory=Counter)  # класс → сколько
    invisible_by_type: Counter = field(default_factory=Counter)  # (класс, тип эталона) → сколько
    sessions_with_invisible: int = 0
    examples: dict = field(default_factory=dict)  # (класс, тип) → примеры


def coverage(records, spans_of, profile_types: set[str], examples: int = 5) -> Coverage:
    """records — (session, log, gold); spans_of(session, log) → [(turn, start, end)] видимых фрагментов."""
    cov = Coverage()
    for session, log, gold in records:
        spans = spans_of(session, log)
        hidden = False
        for g in gold:
            if g.category in ("none", "ambiguous"):
                continue
            cov.total += 1
            if any(_overlaps(t, s, e, g) for t, s, e in spans):
                cov.visible += 1
                continue
            hidden = True
            cls = classify(g, profile_types)
            cov.invisible[cls] += 1
            cov.invisible_by_type[cls, g.type] += 1
            ex = cov.examples.setdefault((cls, g.type), [])
            if len(ex) < examples:
                ex.append(g.text)
        cov.sessions_with_invisible += hidden
    return cov


def profile_types(log) -> set[str]:
    return {t.lower() for t in log.model.labels} | {r.type.lower() for r in log.rules}


@dataclass
class Extra:
    """Новые находки способа сверх базовых: сколько их и сколько не задевают ПДн эталона."""

    new: int = 0
    no_pii: int = 0  # не пересекаются ни с одним ПДн эталона — лишний разбор агентом
    examples: list = field(default_factory=list)


def extra_findings(records, spans_of, base_of, examples: int = 8) -> Extra:
    """records — (session, log, gold, base_log); находки spans_of, которых нет среди base_of."""
    out = Extra()
    for session, log, gold, base_log in records:
        base = base_of(session, base_log)
        pii = [g for g in gold if g.category not in ("none", "ambiguous")]
        for t, s, e in spans_of(session, log):
            if any(t == bt and s < be and bs < e for bt, bs, be in base):
                continue
            out.new += 1
            if not any(_overlaps(t, s, e, g) for g in pii):
                out.no_pii += 1
                if len(out.examples) < examples:
                    text = next(m.text for m in session.messages if m.turn == t)
                    out.examples.append(text[s:e])
    return out
