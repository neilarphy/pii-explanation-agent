"""Оценка по логам детектора на pii-bench: метрики детектора и прогон агента.

    uv run python -m scripts.evaluate                     # только метрики детектора
    uv run python -m scripts.evaluate --agent 50          # + агент на 50 находках

Агент сверяется с автоэталоном из разметки pii-bench (evaluation.metrics.expected_verdict).
Выборка стратифицирована по ожидаемому вердикту, чтобы в 50 случаях были и ложные
срабатывания, и пропуски, а не одни верные находки.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from backend.agent.agent import explain_case
from backend.agent.cases import build_cases
from backend.agent.llm import client_from_env
from backend.schemas import DetectionLog, Session
from evaluation.metrics import expected_verdict, score_detector
from evaluation.pii_bench import gold_from_json

VERDICTS = ["false_positive", "true_pii", "missed_pii", "no_issue"]


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


def sample_cases(records, n: int, seed: int):
    pool: dict[str, list] = {v: [] for v in VERDICTS}
    for _, session, log, gold in records:
        for case in build_cases(log):
            pool[expected_verdict(case, gold)].append((session, log, case))
    rng = random.Random(seed)
    for items in pool.values():
        rng.shuffle(items)
    sizes = {v: len(items) for v, items in pool.items()}
    picked, per = [], max(n // len(VERDICTS), 1)
    for v in VERDICTS:
        picked += [(v, *x) for x in pool[v][:per]]
    leftovers = [(v, *x) for v in VERDICTS for x in pool[v][per:]]
    rng.shuffle(leftovers)
    picked += leftovers[: n - len(picked)]
    return picked, sizes


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=Path("runs/detector/pii-bench.jsonl"))
    parser.add_argument("--agent", type=int, default=0, help="сколько находок дать агенту")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = parser.parse_args()

    records = list(load_records(args.logs))
    report = [f"# Оценка на pii-bench ({len(records)} сессий)", "", "## Детектор", ""]
    report += detector_table(records)

    if args.agent:
        picked, sizes = sample_cases(records, args.agent, args.seed)
        complete = client_from_env()
        rows, confusion, errors = [], Counter(), 0
        for i, (want, session, log, case) in enumerate(picked, 1):
            try:
                answer = explain_case(case, session, log, complete)
            except Exception as exc:  # одна сломанная находка не должна валить прогон
                errors += 1
                print(f"[{i}/{len(picked)}] {case.id}: ошибка {exc!r}")
                continue
            got = answer.verdict.value if answer else "no_issue"
            confusion[(want, got)] += 1
            rows.append(
                {
                    "case_id": case.id,
                    "kind": case.kind,
                    "type": case.type,
                    "text": case.text,
                    "expected": want,
                    "got": got,
                    "answer": answer.model_dump(mode="json") if answer else None,
                }
            )
            print(f"[{i}/{len(picked)}] {case.id} «{case.text}» эталон {want} → агент {got}")

        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "agent_answers.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
        )
        correct = sum(n for (w, g), n in confusion.items() if w == g)
        report += [
            "",
            "## Агент",
            "",
            "Находок в логах по автоэталону: " + ", ".join(f"{v} — {sizes[v]}" for v in VERDICTS),
            f"Выборка: {len(rows)} (ошибок вызова: {errors}). "
            f"Совпало с автоэталоном: **{correct} из {len(rows)}** ({correct / max(len(rows), 1):.0%}).",
            "",
            "| Эталон \\ Агент | " + " | ".join([*VERDICTS, "uncertain"]) + " |",
            "|---|" + "---|" * (len(VERDICTS) + 1),
        ]
        for w in VERDICTS:
            cells = [str(confusion[(w, g)]) for g in [*VERDICTS, "uncertain"]]
            report.append(f"| {w} | " + " | ".join(cells) + " |")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(report) + "\n", "utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
