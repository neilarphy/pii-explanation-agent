"""Метрики детектора и автоэталон для агента по разметке pii-bench."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from backend.agent.cases import Case
from backend.schemas import DetectionLog
from evaluation.pii_bench import GoldSpan


def _overlap(turn: int, start: int, end: int, g: GoldSpan) -> bool:
    return turn == g.turn and start < g.end and g.start < end


@dataclass
class TypeScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    exact: int = 0  # из tp: границы совпали точно

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class DetectorReport:
    by_type: dict[str, TypeScore] = field(default_factory=lambda: defaultdict(TypeScore))
    uncovered_gold: Counter = field(default_factory=Counter)  # типы pii-bench вне профиля


def score_detector(records: list[tuple[DetectionLog, tuple[GoldSpan, ...]]]) -> DetectorReport:
    """Совпадение — пересечение спанов в одной реплике при совпадающем типе.

    Пересекающиеся сущности детектора («Анна», «Анна Иванова») засчитываются одной
    находкой на эталонный спан; лишние сущности без эталона — FP.
    """
    report = DetectorReport()
    for log, gold in records:
        for g in gold:
            if g.detector_type is None:
                report.uncovered_gold[g.type] += 1
                continue
            score = report.by_type[g.detector_type]
            hits = [
                e
                for e in log.entities
                if e.type == g.detector_type and _overlap(e.turn, e.start, e.end, g)
            ]
            if hits:
                score.tp += 1
                score.exact += any(e.start == g.start and e.end == g.end for e in hits)
            else:
                score.fn += 1
        for e in log.entities:
            if not any(_overlap(e.turn, e.start, e.end, g) for g in gold):
                report.by_type[e.type].fp += 1
    return report


_RANK = {"pii": 2, "sensitive": 1, "none": 0}


def expected(case: Case, gold: tuple[GoldSpan, ...]) -> tuple[str, str]:
    """Автоэталон (вердикт, категория) по разметке pii-bench и политике 152-ФЗ.

    Категория — самая «строгая» среди пересекающихся эталонных спанов (none, если их нет).
    Маскировать нужно pii и sensitive; реквизиты организаций (КПП, ОГРН, ИНН из 10 цифр) — нет.
    """
    hits = [g.category for g in gold if _overlap(case.turn, case.start, case.end, g)]
    category = max(hits, key=_RANK.__getitem__, default="none")
    mask = category != "none"
    if case.kind == "entity":
        return ("true_pii" if mask else "false_positive"), category
    return ("missed_pii" if mask else "no_issue"), category
