"""Оценка по логам детектора на pii-bench: метрики детектора и прогон агента.

    uv run python -m scripts.evaluate                     # только метрики детектора
    uv run python -m scripts.evaluate --per-class 60      # + агент, до 60 находок на вердикт

Агент сверяется с автоэталоном из разметки pii-bench по политике 152-ФЗ
(evaluation.metrics.expected): вердикт и категория. Выборка стратифицирована по ожидаемому
вердикту, чтобы были и ложные срабатывания, и пропуски, а не одни верные находки.
"""

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.agent.agent import explain_case
from backend.agent.cases import build_cases
from backend.agent.llm import client_from_env
from backend.schemas import DetectionLog, Session
from evaluation.metrics import expected, score_detector
from evaluation.pii_bench import gold_from_json

VERDICTS = ["false_positive", "true_pii", "missed_pii", "no_issue"]
COLUMNS = [*VERDICTS, "uncertain", "ошибка"]


def load_records(path: Path):
    for line in path.read_text("utf-8").splitlines():
        r = json.loads(line)
        yield (
            r["domain"],
            Session.model_validate(r["session"]),
            DetectionLog.model_validate(r["log"]),
            gold_from_json(r["gold"]),
        )


def detector_table(records) -> list[str]:
    report = score_detector([(log, gold) for _, _, log, gold in records])
    lines = [
        "| Тип | TP | FP | FN | Точность | Полнота | F1 | Точные границы |",
        "|---|---|---|---|---|---|---|---|",
    ]
    total = Counter()
    for name, s in sorted(report.by_type.items()):
        lines.append(
            f"| {name} | {s.tp} | {s.fp} | {s.fn} | {s.precision:.2f} | {s.recall:.2f} "
            f"| {s.f1:.2f} | {s.exact}/{s.tp} |"
        )
        total.update(tp=s.tp, fp=s.fp, fn=s.fn)
    p = total["tp"] / max(total["tp"] + total["fp"], 1)
    r = total["tp"] / max(total["tp"] + total["fn"], 1)
    f1 = 2 * p * r / (p + r) if p + r else 0
    lines.append(
        f"| **Всего** | {total['tp']} | {total['fp']} | {total['fn']} | {p:.2f} | {r:.2f} | {f1:.2f} | |"
    )
    if report.uncovered_gold:
        lines.append("")
        lines.append(
            "Типы pii-bench, которых нет в профиле детектора (в метрики не входят): "
            + ", ".join(f"{t} — {n}" for t, n in report.uncovered_gold.most_common())
        )
    return lines


def sample_cases(records, per_class: int, seed: int):
    """До per_class находок на каждый ожидаемый вердикт и размеры классов."""
    pool: dict[str, list] = {v: [] for v in VERDICTS}
    for domain, session, log, gold in records:
        for case in build_cases(log, session):
            verdict, category = expected(case, gold)
            pool[verdict].append((verdict, category, domain, session, log, case))
    rng = random.Random(seed)
    picked = []
    for items in pool.values():
        rng.shuffle(items)
        picked += items[:per_class]
    return picked, {v: len(items) for v, items in pool.items()}


def run_agent(picked, workers: int):
    complete = client_from_env()

    def one(item):
        want, want_cat, domain, session, log, case = item
        try:
            answer = explain_case(case, session, log, complete)
        except Exception as exc:  # одна сломанная находка не должна валить прогон
            return {"case_id": case.id, "expected": want, "got": "ошибка", "error": repr(exc)}
        got = answer.verdict.value if answer else "no_issue"
        return {
            "case_id": case.id,
            "domain": domain,
            "kind": case.kind,
            "type": case.type,
            "text": case.text,
            "expected": want,
            "expected_category": want_cat,
            "got": got,
            "got_category": answer.category.value if answer else None,
            "answer": answer.model_dump(mode="json") if answer else None,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(one, picked), 1):
            rows.append(row)
            print(
                f"[{i}/{len(picked)}] {row['case_id']} эталон {row['expected']} → агент {row['got']}"
            )
    return rows


def agent_report(rows, sizes) -> list[str]:
    confusion = Counter((r["expected"], r["got"]) for r in rows)
    answered = [r for r in rows if r["got"] != "ошибка"]
    correct = sum(r["expected"] == r["got"] for r in answered)
    with_cat = [r for r in answered if r["got_category"] and r["expected_category"] != "none"]
    cat_ok = sum(r["got_category"] == r["expected_category"] for r in with_cat)
    by_kind = Counter((r["kind"], r["expected"] == r["got"]) for r in answered)
    lines = [
        "",
        "## Агент",
        "",
        "Находок в логах по автоэталону: " + ", ".join(f"{v} — {sizes[v]}" for v in VERDICTS),
        f"Выборка: {len(rows)}, ошибок вызова: {len(rows) - len(answered)}.",
        f"Вердикт совпал с автоэталоном: **{correct} из {len(answered)}** "
        f"({correct / max(len(answered), 1):.0%}).",
        f"Категория (pii / sensitive) верна: {cat_ok} из {len(with_cat)}.",
        "По видам находок: "
        + ", ".join(
            f"{k} — {by_kind[(k, True)]}/{by_kind[(k, True)] + by_kind[(k, False)]}"
            for k in ("entity", "rejected", "scan")
        ),
        "",
        "| Эталон \\ Агент | " + " | ".join(COLUMNS) + " |",
        "|---|" + "---|" * len(COLUMNS),
    ]
    for w in VERDICTS:
        lines.append(f"| {w} | " + " | ".join(str(confusion[(w, g)]) for g in COLUMNS) + " |")
    return lines


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=Path("runs/detector/pii-bench.jsonl"))
    parser.add_argument("--per-class", type=int, default=0, help="находок на вердикт для агента")
    parser.add_argument("--workers", type=int, default=4, help="параллельных вызовов LLM")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = parser.parse_args()

    records = list(load_records(args.logs))
    report = [f"# Оценка на pii-bench ({len(records)} сессий)", "", "## Детектор", ""]
    report += detector_table(records)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.per_class:
        picked, sizes = sample_cases(records, args.per_class, args.seed)
        rows = run_agent(picked, args.workers)
        (args.out / "agent_answers.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
        )
        report += agent_report(rows, sizes)

    (args.out / "report.md").write_text("\n".join(report) + "\n", "utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
