"""Сводка по N сессиям: повторяющиеся ошибки детектора → задачи и .md для Claude Code.

Собирается кодом, без LLM: ответы агента, где детектор ошибся и есть рекомендация,
группируются по адресату правки (правило или модель), виду ошибки и типу данных. Из каждой
группы — предложение агента, примеры из диалогов и строки для тестов детектора. Находки,
по которым агент не уверен (каскад разошёлся, сбой LLM), — отдельным списком для человека.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

from backend.agent.cases import Case
from backend.schemas import AgentAnswer, Recommendation, Summary, SummaryIssue

EXAMPLES = 3
_ERROR = {"false_positive": "ложные срабатывания", "missed_pii": "пропуски"}


@dataclass(frozen=True)
class Finding:
    session_id: str
    case: Case
    answer: AgentAnswer | None


def _group_key(f: Finding) -> tuple[str, str, str]:
    """Одна проблема — адресат, вид ошибки и тип данных; вид правки агент выбирает по-разному."""
    return f.answer.recommendation.target, f.answer.verdict.value, f.case.type


def _test_line(f: Finding) -> str:
    if f.answer.verdict.value == "false_positive":
        return f"«{f.case.text}» не даёт {f.case.type}"
    return f"«{f.case.text}» даёт {f.case.type}"


def build_summary(findings: list[Finding]) -> Summary:
    sessions = sorted({f.session_id for f in findings})
    groups: dict[tuple, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.answer and f.answer.recommendation and f.answer.verdict.value in _ERROR:
            groups[_group_key(f)].append(f)
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ordered = [(k, v) for k, v in ranked if len(v) > 1]  # повторяющиеся — задачи
    single = [v[0] for _, v in ranked if len(v) == 1]  # разовые — коротким списком
    unsure = [f for f in findings if f.answer and f.answer.verdict.value == "uncertain"]

    issues, recommendations, sections = [], [], []
    for i, ((target, verdict, pii_type), items) in enumerate(ordered, 1):
        title = f"{target}: {_ERROR[verdict]} {pii_type}"
        kinds = Counter(f.answer.recommendation.kind for f in items)
        kind = kinds.most_common(1)[0][0]
        lead = next(f.answer.recommendation for f in items if f.answer.recommendation.kind == kind)
        issues.append(
            SummaryIssue(
                title=title,
                count=len(items),
                example_session_ids=list(dict.fromkeys(f.session_id for f in items))[:EXAMPLES],
                example_entity_ids=[f.case.entity_id for f in items if f.case.entity_id][:EXAMPLES],
            )
        )
        recommendations.append(Recommendation.model_validate(lead.model_dump()))
        variants = [
            f.answer.recommendation.change
            for f in items
            if f.answer.recommendation.change != lead.change
        ]
        sections += [
            f"## {i}. {title} ({len(items)})",
            "",
            f"- Виды правки у агента: {', '.join(f'{k} — {n}' for k, n in kinds.most_common())}",
            f"- Правка ({kind} → `{target}`): {lead.change}",
            f"- Почему: {lead.rationale}",
            *(f"- Другая формулировка агента: {v}" for v in list(dict.fromkeys(variants))[:2]),
            "- Примеры: "
            + "; ".join(f"«{f.case.text}» (сессия {f.session_id})" for f in items[:EXAMPLES]),
            "",
        ]
    tests = [_test_line(f) for _, items in ordered for f in items[:2]]
    markdown = [
        "# Задача для Claude Code: поправить профиль детектора",
        "",
        f"По {len(sessions)} сессиям агент нашёл {len(ordered)} повторяющихся проблем детектора "
        f"({sum(len(v) for _, v in ordered)} находок) и {len(single)} разовых. Проблема — адресат "
        "правки (правило профиля или модель), вид ошибки и тип данных.",
        "",
        *sections,
        *(
            [
                f"## Разовые ({len(single)})",
                "",
                *(
                    f"- `{f.answer.recommendation.target}`: «{f.case.text}» — "
                    f"{f.answer.recommendation.change}"
                    for f in single[:15]
                ),
                "",
            ]
            if single
            else []
        ),
        "## Как проверить",
        "",
        *(f"- Тест детектора: {t}." for t in tests),
        "- Прогнать тесты детектора и оценку на pii-bench: полнота по затронутым типам не упала.",
        "",
    ]
    if unsure:
        markdown += [
            f"## Нужен человек ({len(unsure)})",
            "",
            "Агент не уверен: ответы модели разошлись или агент не ответил.",
            "",
            *(
                f"- «{f.case.text}» (сессия {f.session_id}, {f.case.type}): {f.answer.note}"
                for f in unsure[:10]
            ),
            "",
        ]
    return Summary(
        session_ids=sessions,
        issues=issues,
        recommendations=recommendations,
        markdown="\n".join(markdown),
    )
