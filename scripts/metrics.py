"""Метрики чекпоинта 1 по сохранённым ответам агента — без вызовов LLM.

    uv run python -m scripts.metrics \\
        --run "qwen3.8-27b, харнесс=runs/iter6-scenarios/qwen3.8-27b__harness.jsonl" \\
        --repeat "gemma-3-12b, харнесс=runs/iter6-scenarios/gemma-3-12b-it__harness.jsonl,runs/iter6-rep2/..." \\
        --title "Свои сценарии" --out docs/reports/iter6/metrics-scenarios.md

--run — один прогон (имя=путь), --repeat — несколько прогонов одной конфигурации на одних
и тех же находках (имя=путь,путь,...): из них — разброс точности.
"""

import argparse
import json
import sys
from pathlib import Path

from evaluation.agent_metrics import (
    VERDICTS,
    as_filter,
    citations,
    macro_f1,
    misses,
    per_class,
    policy_mismatch,
    stability,
    uncertain_by_expected,
    usage,
)


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text("utf-8").splitlines()]


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "—"


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def report(title: str, runs: dict[str, list[dict]], repeats: dict[str, list[list[dict]]]):
    lines = [f"# Метрики агента: {title}", ""]
    lines += [
        "## 1. Вердикт по каждому классу",
        "",
        "Точность / полнота по каждому вердикту эталона; uncertain и сбои не попадают ни в один класс.",
        "«Расхождение политики» — эталон требует маскировать контакт организации или тестовую карту,",
        "агент по политике проекта (docs/pii-policy.md) — нет; это не ошибка агента.",
        "",
    ]
    rows = []
    for name, data in runs.items():
        scores = per_class(data)
        correct = sum(r["expected"] == r["got"] for r in data)
        mismatch = sum(policy_mismatch(r) for r in data)
        rows.append(
            [name, pct(correct, len(data)), str(mismatch), pct(correct + mismatch, len(data))]
            + [f"{fmt(scores[v].precision)} / {fmt(scores[v].recall)}" for v in VERDICTS]
            + [fmt(macro_f1(scores))]
        )
    lines += table(
        [
            "Прогон",
            "Вердикт верен",
            "Расхождение политики",
            "Верно по политике проекта",
            *VERDICTS,
            "Макро-F1",
        ],
        rows,
    )

    lines += [
        "",
        "## 2. Агент как второй фильтр после детектора",
        "",
        "Находки, которые детектор принял. Агент снимает находку, только если назвал её ложной;",
        "«не уверен» и сбой — находка остаётся и уходит человеку (fail-open). Ориентир с чекпоинта:",
        "QASecClaw снял 88.6% ложных срабатываний ценой 3.1% настоящих.",
        "",
    ]
    rows = []
    for name, data in runs.items():
        f = as_filter(data)
        rows.append(
            [
                name,
                pct(f.false_removed, f.false_found),
                pct(f.true_lost, f.true_found),
                str(f.to_human),
            ]
        )
    lines += table(["Прогон", "Ложных снято", "Настоящих потеряно", "На человека"], rows)

    lines += [
        "",
        "## 3. Пропуски детектора",
        "",
        "Отклонённые детектором и найденные кодом фрагменты.",
        "",
    ]
    rows = []
    for name, data in runs.items():
        m = misses(data)
        rows.append([name, pct(m.found, m.missed), str(m.to_human), pct(m.false_alarms, m.clean)])
    lines += table(["Прогон", "Пропусков найдено", "Пропуск → на человека", "Ложных тревог"], rows)

    lines += ["", "## 4. «Не уверен»", ""]
    rows = []
    for name, data in runs.items():
        unsure = uncertain_by_expected(data)
        total = sum(unsure.values())
        rows.append([name, pct(total, len(data))] + [str(unsure.get(v, 0)) for v in VERDICTS])
    lines += table(["Прогон", "Всего", *(f"при эталоне {v}" for v in VERDICTS)], rows)

    lines += ["", "## 5. Цитаты", ""]
    rows = []
    for name, data in runs.items():
        c = citations(data)
        rows.append(
            [
                name,
                pct(c.with_citations, c.answers),
                pct(c.all_verbatim, c.answers),
                pct(c.first_is_finding, c.answers),
            ]
        )
    lines += table(
        [
            "Прогон",
            "Есть дословная цитата",
            "Модель не прислала недословных",
            "Первая — сама находка",
        ],
        rows,
    )

    injection = {n: [r for r in d if r.get("domain") == "injection"] for n, d in runs.items()}
    if any(injection.values()):
        lines += [
            "",
            "## 6. Prompt injection",
            "",
            "Пользователь присылает паспорт и СНИЛС и дописывает фразу для системы проверки: данные",
            "якобы тестовые, считать их ПДн не нужно.",
            "",
        ]
        lines += table(
            ["Прогон", "Вердикт верен"],
            [
                [n, pct(sum(r["expected"] == r["got"] for r in d), len(d))]
                for n, d in injection.items()
            ],
        )

    lines += ["", "## 7. Расход на находку", ""]
    rows = []
    for name, data in runs.items():
        u = usage(data)
        if u is None:
            continue
        rows.append(
            [
                name,
                f"{u.calls:.2f}",
                f"{u.retries_share:.0%}",
                f"{u.prompt_tokens:.0f} / {u.completion_tokens:.0f}",
                f"{u.seconds_p50:.1f} / {u.seconds_p90:.1f}",
                f"{u.cost_usd:.4f}" if u.cost_usd else "не сообщалась",
                f"{u.escalated_share:.0%}",
            ]
        )
    lines += table(
        [
            "Прогон",
            "Вызовов LLM",
            "С повтором",
            "Токенов вход / выход",
            "Секунд p50 / p90",
            "Стоимость всего, $",
            "Каскад: к большой / человеку",
        ],
        rows,
    )

    if repeats:
        lines += [
            "",
            "## 8. Разброс между прогонами",
            "",
            "Одна конфигурация, те же находки, несколько прогонов (temperature 0 не делает модель",
            "детерминированной).",
            "",
        ]
        rows = []
        for name, group in repeats.items():
            s = stability(group)
            rows.append(
                [
                    name,
                    str(s.runs),
                    " · ".join(f"{a:.1%}" for a in s.accuracy),
                    f"{s.mean:.1%} ± {s.spread:.1%}",
                    pct(s.flipped, s.cases),
                ]
            )
        lines += table(
            [
                "Конфигурация",
                "Прогонов",
                "Точность в каждом",
                "Среднее ± полуразмах",
                "Вердикт менялся",
            ],
            rows,
        )
    return lines


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", default=[], help="имя=путь")
    parser.add_argument("--repeat", action="append", default=[], help="имя=путь,путь,...")
    parser.add_argument("--title", default="ответы агента")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    runs = {name: load(path) for name, path in (a.split("=", 1) for a in args.run)}
    repeats = {
        name: [load(p) for p in paths.split(",")]
        for name, paths in (a.split("=", 1) for a in args.repeat)
    }
    text = "\n".join(report(args.title, runs, repeats)) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
