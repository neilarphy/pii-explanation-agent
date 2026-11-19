"""Ручная проверка ответов агента и проверка самого судьи.

    # 1. выгрузить N ответов в таблицу для людей (откроется в Excel / Google Sheets)
    uv run python -m scripts.review export --answers runs/.../agent_answers.jsonl \\
        --logs runs/detector/....jsonl --n 30 --out runs/review/sheet.csv

    # 2. после заполнения: согласие человека с судьёй по каждому вопросу
    uv run python -m scripts.review agreement --sheet runs/review/sheet.csv \\
        --judge runs/.../judge.jsonl

В колонках faithful / citations_ok / recommendation_ok / verdict_ok человек ставит 1 или 0
(для рекомендации — n/a, если её нет), в comment — что не так.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from evaluation.judge import cohen_kappa
from scripts.judge import find_case, load_context

QUESTIONS = ["faithful", "citations_ok", "recommendation_ok", "verdict_ok"]
_DIRECTION = {"in": "пользователь", "out": "бот"}


def export(args) -> None:
    rows = [json.loads(line) for line in args.answers.read_text("utf-8").splitlines()]
    rows = [r for r in rows if r.get("answer")]
    if args.judged:  # только ответы, которые уже оценил судья, — чтобы было с чем сравнить
        judged = {
            json.loads(line)["case_id"] for line in args.judged.read_text("utf-8").splitlines()
        }
        rows = [r for r in rows if r["case_id"] in judged]
    random.Random(args.seed).shuffle(rows)
    context = load_context(args.logs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:  # BOM — для Excel
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "диалог",
                "находка",
                "вердикт",
                "категория",
                "объяснение",
                "цитаты",
                "рекомендация",
                *QUESTIONS,
                "comment",
            ]
        )
        for row in rows[: args.n]:
            case, session, _ = find_case(row["case_id"], context)
            a = row["answer"]
            dialog = "\n".join(
                f"[{m.turn}] {_DIRECTION[m.direction]}: {m.text}" for m in session.messages
            )
            rec = a["recommendation"]
            writer.writerow(
                [
                    row["case_id"],
                    dialog,
                    f"реплика {case.turn}: «{case.text}» ({case.kind}, {case.type})",
                    a["verdict"],
                    a["category"],
                    a["explanation"],
                    " | ".join(f"[{c['turn']}] {c['quote']}" for c in a["citations"]),
                    f"{rec['kind']} → {rec['target']}: {rec['change']}" if rec else "",
                    *[""] * len(QUESTIONS),
                    "",
                ]
            )
    print(f"Выгружено {min(args.n, len(rows))} ответов: {args.out}")


def _value(cell: str):
    cell = (cell or "").strip().lower()
    if cell in ("1", "да", "yes"):
        return 1
    if cell in ("0", "нет", "no"):
        return 0
    return None  # n/a или не заполнено


def agreement(args) -> None:
    with args.sheet.open(encoding="utf-8-sig") as handle:
        human = {r["case_id"]: r for r in csv.DictReader(handle)}
    judge = {
        r["case_id"]: r
        for r in map(json.loads, args.judge.read_text("utf-8").splitlines())
        if "error" not in r
    }
    print(f"Ответов с обеими оценками: {len(set(human) & set(judge))}")
    for q in QUESTIONS:
        pairs = [
            (h, j)
            for cid in set(human) & set(judge)
            if (h := _value(human[cid][q])) is not None and (j := judge[cid][q]) is not None
        ]
        if pairs:
            same = sum(a == b for a, b in pairs) / len(pairs)
            print(f"- {q}: совпало {same:.0%} из {len(pairs)}, kappa {cohen_kappa(pairs):.2f}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--answers", type=Path, required=True)
    e.add_argument("--logs", type=Path, required=True)
    e.add_argument("--n", type=int, default=30)
    e.add_argument("--seed", type=int, default=21)
    e.add_argument("--out", type=Path, default=Path("runs/review/sheet.csv"))
    e.add_argument("--judged", type=Path, help="judge.jsonl: выгружать только оценённые судьёй")
    a = sub.add_parser("agreement")
    a.add_argument("--sheet", type=Path, required=True)
    a.add_argument("--judge", type=Path, required=True)
    args = parser.parse_args()
    export(args) if args.command == "export" else agreement(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
