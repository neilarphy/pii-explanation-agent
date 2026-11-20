"""Невидимые для агента ПДн и способы их найти — без вызовов LLM.

    uv run python -m scripts.coverage \\
        --set redmadrobot=runs/detector/redmadrobot.jsonl,runs/detector-emit005/redmadrobot.jsonl \\
        --out docs/reports/iter8/invisible.md

--set — имя=логи детектора[,логи того же набора с низким порогом emit]. Для каждого набора:
невидимые ПДн по классам и сравнение способов — сколько невидимых они делают видимыми и сколько
добавляют находок, которые не задевают ПДн эталона (лишний разбор агентом).
"""

import argparse
from functools import partial
from pathlib import Path

from backend.agent.cases import build_cases
from evaluation.coverage import (
    OUT_OF_PROFILE,
    POLICY_NONE,
    REAL,
    coverage,
    extra_findings,
    profile_types,
)
from scripts.evaluate import load_records


def case_spans(session, log, wide: bool = False):
    return [(c.turn, c.start, c.end) for c in build_cases(log, session, wide=wide)]


def detail(cov, sessions: int) -> list[str]:
    hidden = sum(cov.invisible.values())
    policy = sum(cov.invisible[c] for c in POLICY_NONE)
    lines = [
        f"ПДн в эталоне — {cov.total}, агент видит {cov.visible}, не видит **{hidden}** "
        f"({hidden / max(cov.total, 1):.1%}) в {cov.sessions_with_invisible} сессиях из {sessions}. "
        f"Из невидимых по нашей политике не ПДн — {policy}.",
        "",
        "| Класс | Тип в эталоне | Сколько | Примеры |",
        "|---|---|---|---|",
    ]
    for (cls, typ), n in sorted(cov.invisible_by_type.items(), key=lambda kv: (kv[0][0], -kv[1])):
        ex = "; ".join(f"«{t}»" for t in cov.examples[cls, typ][:4])
        lines.append(f"| {cls} | {typ} | {n} | {ex} |")
    lines += ["", "| Класс | Итого |", "|---|---|"]
    lines += [f"| {cls} | {n} |" for cls, n in cov.invisible.most_common()]
    return [*lines, ""]


def section(name: str, base_path: Path, low_path: Path | None) -> list[str]:
    base = {s.id: (s, log, gold) for _, s, log, gold in load_records(base_path)}
    low = {s.id: log for _, s, log, _ in load_records(low_path)} if low_path else {}
    types = profile_types(next(iter(base.values()))[1])
    methods = [("До итерации 8: детектор и код", False, False), ("+ поиск кодом шире", True, False)]
    if low:
        methods += [
            ("+ порог детектора 0.05", False, True),
            ("+ порог 0.05 и код шире", True, True),
        ]

    lines = [f"## {name}", ""]
    rows = []
    for title, wide, use_low in methods:
        records = [(s, low[sid] if use_low else log, gold) for sid, (s, log, gold) in base.items()]
        cov = coverage(records, partial(case_spans, wide=wide), types)
        if not rows:
            lines += detail(cov, len(records))
        extra = extra_findings(
            [(s, lg, gold, base[s.id][1]) for s, lg, gold in records],
            partial(case_spans, wide=wide),
            case_spans,
        )
        rows.append((title, cov, extra))

    real0 = rows[0][1].invisible[REAL] + rows[0][1].invisible[OUT_OF_PROFILE]
    lines += [
        "### Способы найти невидимое",
        "",
        f"Настоящих невидимых ПДн (пропуски детектора и типы вне профиля) — {real0}. "
        "«Лишние» — новые находки, которые не задевают ПДн эталона: агент разберёт их зря "
        "(или там не-ПДн по нашей политике).",
        "",
        "| Способ | Настоящих невидимых осталось | Найдено из них | Не ПДн по политике осталось "
        "| Новых находок | Из них лишних |",
        "|---|---|---|---|---|---|",
    ]
    for title, cov, extra in rows:
        real = cov.invisible[REAL] + cov.invisible[OUT_OF_PROFILE]
        policy = sum(cov.invisible[c] for c in POLICY_NONE)
        share = f" ({extra.no_pii / extra.new:.0%})" if extra.new else ""
        lines.append(
            f"| {title} | {real} | {real0 - real} ({(real0 - real) / max(real0, 1):.0%}) | {policy} "
            f"| {extra.new} | {extra.no_pii}{share} |"
        )
    for title, _, extra in rows[1:]:
        if extra.examples:
            lines += [
                "",
                f"Лишние находки, «{title}»: " + "; ".join(f"«{t}»" for t in extra.examples),
            ]
    return [*lines, ""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--set", action="append", required=True, help="имя=логи[,логи с низким порогом]"
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    lines = ["# ПДн, которых агент не видит, и как их найти", ""]
    for item in args.set:
        name, paths = item.split("=", 1)
        base, _, low = paths.partition(",")
        lines += section(name, Path(base), Path(low) if low else None)
    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
