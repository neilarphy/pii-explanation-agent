"""Метрики агента по сохранённым ответам (agent_answers.jsonl) — без вызовов LLM.

То, что заявляли на чекпоинте 1: метрики по каждому вердикту (LLM осторожна с «ложным»),
агент как второй фильтр после детектора (сколько ложных срабатываний снято и сколько настоящих
ПДн потеряно — как в QASecClaw), найденные пропуски, доля «не уверен», дословность цитат,
устойчивость к injection, расход и разброс между прогонами.

Строка ответа — словарь из scripts/evaluate.py: expected, got, kind, domain, text, answer, telemetry.
Сбой вызова — got == "ошибка".
"""

import re
import statistics
from collections import Counter
from dataclasses import dataclass

VERDICTS = ("true_pii", "false_positive", "missed_pii", "no_issue", "uncertain")
DROPPED = "отброшены недословные цитаты"

# Контакты организаций и публичные тестовые карты: открытые наборы размечают их как ПДн по форме,
# политика проекта — нет (docs/pii-policy.md, раздел 4).
_ORG_MAILBOX = re.compile(
    r"(?i)^(info|help|support|noreply|no-reply|hello|contact|contacts|sales|office|admin|service"
    r"|feedback|press|hr|mail|team|billing|zakaz|order|orders)@"
)
TEST_CARDS = {
    "4111111111111111",
    "5555555555554444",
    "4242424242424242",
    "4000056655665556",
    "5105105105105100",
    "2223003122003222",
    "378282246310005",
    "4012888888881881",
    "5200828282828210",
    "6011111111111117",
}


def not_personal_by_policy(text: str) -> bool:
    """Общий ящик или 8-800 организации, публичная тестовая карта — не ПДн по политике проекта."""
    text = re.sub(r"\s+", "", text)  # «info @ fortress . com . ru» — так размечено в redmadrobot
    digits = re.sub(r"\D", "", text)
    toll_free = len(digits) == 11 and digits[0] in "78" and digits[1:4] == "800"
    return bool(_ORG_MAILBOX.match(text)) or toll_free or digits in TEST_CARDS


def policy_mismatch(row: dict) -> bool:
    """Эталон набора требует маскировать, агент по политике проекта — нет. Не ошибка агента."""
    return (
        row["expected"] in ("true_pii", "missed_pii")
        and row["got"] in ("false_positive", "no_issue")
        and not_personal_by_policy(row.get("text", ""))
    )


@dataclass(frozen=True)
class ClassScore:
    support: int  # сколько находок с таким вердиктом в эталоне
    predicted: int  # сколько раз агент так ответил
    correct: int

    @property
    def precision(self) -> float | None:
        return self.correct / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.correct / self.support if self.support else None

    @property
    def f1(self) -> float | None:
        total = self.support + self.predicted
        return 2 * self.correct / total if total else None


def per_class(rows: list[dict]) -> dict[str, ClassScore]:
    """P/R/F1 по каждому вердикту; uncertain и сбои — ответы, которые не попали ни в один класс."""
    return {
        v: ClassScore(
            support=sum(r["expected"] == v for r in rows),
            predicted=sum(r["got"] == v for r in rows),
            correct=sum(r["expected"] == v and r["got"] == v for r in rows),
        )
        for v in VERDICTS
    }


def macro_f1(scores: dict[str, ClassScore]) -> float:
    """Среднее F1 по классам, которые есть в эталоне (uncertain — только в спорном наборе)."""
    values = [s.f1 for s in scores.values() if s.support]
    return statistics.fmean(values) if values else 0.0


@dataclass(frozen=True)
class Filter:
    """Агент после детектора на принятых им находках (kind = entity).

    Агент снимает находку, только если сказал false_positive; uncertain и сбой — находка
    остаётся (fail-open), её смотрит человек.
    """

    false_found: int  # ложных срабатываний детектора в эталоне
    false_removed: int  # из них агент снял
    true_found: int  # настоящих ПДн, принятых детектором
    true_lost: int  # из них агент ошибочно снял
    to_human: int  # uncertain или сбой — на человека

    @property
    def removed_share(self) -> float | None:
        return self.false_removed / self.false_found if self.false_found else None

    @property
    def lost_share(self) -> float | None:
        return self.true_lost / self.true_found if self.true_found else None


def as_filter(rows: list[dict]) -> Filter:
    accepted = [r for r in rows if r.get("kind", "entity") == "entity"]
    return Filter(
        false_found=sum(r["expected"] == "false_positive" for r in accepted),
        false_removed=sum(
            r["expected"] == "false_positive" and r["got"] == "false_positive" for r in accepted
        ),
        true_found=sum(r["expected"] == "true_pii" for r in accepted),
        true_lost=sum(
            r["expected"] == "true_pii" and r["got"] == "false_positive" for r in accepted
        ),
        to_human=sum(r["got"] in ("uncertain", "ошибка") for r in accepted),
    )


@dataclass(frozen=True)
class Misses:
    """Отклонённые детектором и найденные кодом фрагменты (kind = rejected | scan)."""

    missed: int  # настоящих пропусков в эталоне
    found: int  # агент назвал пропуском
    to_human: int  # из настоящих пропусков — uncertain или сбой
    clean: int  # фрагментов, которые маскировать не нужно
    false_alarms: int  # из них агент назвал пропуском


def misses(rows: list[dict]) -> Misses:
    rest = [r for r in rows if r.get("kind") in ("rejected", "scan", "llm")]
    return Misses(
        missed=sum(r["expected"] == "missed_pii" for r in rest),
        found=sum(r["expected"] == "missed_pii" and r["got"] == "missed_pii" for r in rest),
        to_human=sum(
            r["expected"] == "missed_pii" and r["got"] in ("uncertain", "ошибка") for r in rest
        ),
        clean=sum(r["expected"] == "no_issue" for r in rest),
        false_alarms=sum(r["expected"] == "no_issue" and r["got"] == "missed_pii" for r in rest),
    )


def uncertain_by_expected(rows: list[dict]) -> Counter:
    return Counter(r["expected"] for r in rows if r["got"] == "uncertain")


@dataclass(frozen=True)
class Citations:
    answers: int  # ответов с объяснением
    with_citations: int  # есть хотя бы одна дословная цитата
    all_verbatim: int  # модель не прислала ни одной недословной цитаты
    first_is_finding: int  # первая цитата — сама находка


def citations(rows: list[dict]) -> Citations:
    answers = [r for r in rows if r.get("answer")]

    def first_ok(r) -> bool:
        cited = r["answer"].get("citations") or []
        if not cited:
            return False
        quote, text = cited[0]["quote"], r["text"]
        return text in quote or quote in text

    return Citations(
        answers=len(answers),
        with_citations=sum(bool(r["answer"].get("citations")) for r in answers),
        all_verbatim=sum(
            not any(e.startswith(DROPPED) for e in r["answer"].get("evidence") or [])
            for r in answers
        ),
        first_is_finding=sum(first_ok(r) for r in answers),
    )


@dataclass(frozen=True)
class Usage:
    calls: float
    retries_share: float
    prompt_tokens: float
    completion_tokens: float
    seconds_p50: float
    seconds_p90: float
    cost_usd: float  # сумма по usage.cost; 0 — провайдер не сообщал (прогоны до итерации 6)
    escalated_share: float  # каскад: доля находок, ушедших большой модели или человеку


def usage(rows: list[dict]) -> Usage | None:
    tel = [r["telemetry"] for r in rows if r.get("telemetry")]
    if not tel:
        return None
    seconds = sorted(t["seconds"] for t in tel)

    def mean(key):
        return statistics.fmean(t[key] for t in tel)

    return Usage(
        calls=mean("llm_calls"),
        retries_share=sum(t["retries"] > 0 for t in tel) / len(tel),
        prompt_tokens=mean("prompt_tokens"),
        completion_tokens=mean("completion_tokens"),
        seconds_p50=statistics.median(seconds),
        seconds_p90=seconds[int(0.9 * (len(seconds) - 1))],
        cost_usd=sum(t.get("cost", 0) for t in tel),
        escalated_share=sum(t.get("escalated", 0) for t in tel) / len(tel),
    )


@dataclass(frozen=True)
class Stability:
    """Одна конфигурация, несколько прогонов на одних и тех же находках."""

    runs: int
    cases: int
    accuracy: tuple[float, ...]  # доля верных вердиктов в каждом прогоне
    flipped: int  # находок, где вердикт отличался хотя бы в одном прогоне

    @property
    def mean(self) -> float:
        return statistics.fmean(self.accuracy)

    @property
    def spread(self) -> float:
        """Половина размаха: точность «± spread» между прогонами."""
        return (max(self.accuracy) - min(self.accuracy)) / 2


def stability(runs: list[list[dict]]) -> Stability:
    by_run = [{r["case_id"]: r for r in rows} for rows in runs]
    common = set.intersection(*(set(run) for run in by_run))
    return Stability(
        runs=len(runs),
        cases=len(common),
        accuracy=tuple(
            sum(run[i]["expected"] == run[i]["got"] for i in common) / max(len(common), 1)
            for run in by_run
        ),
        flipped=sum(len({run[i]["got"] for run in by_run}) > 1 for i in common),
    )
