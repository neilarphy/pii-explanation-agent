"""Ручная разметка ответов агента и проверка судьи: каппа Коэна человек–судья и человек–человек.

    # 1. выгрузить ответы нескольких агентов в одну таблицу (номера пунктов не выдают агента)
    uv run python -m scripts.review export --out-dir data/labeling/judge-kappa-v1 \\
        --source main-scenarios=runs/iter7b-scenarios/qwen3.8-27b__harness.jsonl,runs/detector/scenarios-v1.jsonl \\
        --source gemma=runs/iter8-v2-scenarios/gemma-3-12b-it__harness.jsonl,runs/detector/scenarios-v1.jsonl

    # 2. судья на тех же ответах (подмножества лежат в runs/review/<имя набора>/)
    uv run python -m scripts.judge --answers runs/review/judge-kappa-v1/gemma.jsonl \\
        --logs runs/detector/scenarios-v1.jsonl --model claude-haiku-5.5 \\
        --out runs/review/judge-kappa-v1/judge-gemma.jsonl

    # 3. после разметки: согласие по каждому вопросу
    uv run python -m scripts.review agreement --dir data/labeling/judge-kappa-v1 \\
        --judge-dir runs/review/judge-kappa-v1

Человек ставит в колонках faithful / citations_ok / recommendation_ok / verdict_ok 1 или 0
(для рекомендации — n/a, если её нет), в comment — что не так. Инструкция — README.md в папке набора.
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

from backend.agent.prompt import build_messages
from evaluation.judge import cohen_kappa
from scripts.judge import find_case, load_context

QUESTIONS = ["faithful", "citations_ok", "recommendation_ok", "verdict_ok"]
_DIRECTION = {"in": "пользователь", "out": "бот"}
_COLUMNS = [
    "item",
    "диалог",
    "находка и данные детектора",
    "вердикт",
    "категория",
    "объяснение",
    "цитаты",
    "рекомендация",
    *QUESTIONS,
    "comment",
]


def _source(spec: str) -> tuple[str, Path, Path]:
    name, paths = spec.split("=", 1)
    answers, logs = paths.split(",")
    return name, Path(answers), Path(logs)


def _sheet_row(item: str, row: dict, context) -> list[str]:
    case, session, log = find_case(row["case_id"], context)
    a = row["answer"]
    # то же, что видел агент (и судья): профиль, данные детектора, проверки кодом — без диалога
    seen = build_messages(case, session, log)[1]["content"]
    profile, rest = seen.split("\n\nДиалог:", 1)
    finding = "Находка" + rest.split("\n\nНаходка", 1)[1] + "\n\n" + profile
    finding = re.sub(r" \(case_id: [^)]*\)", "", finding)  # id выдаёт тему сценария
    dialog = "\n".join(f"[{m.turn}] {_DIRECTION[m.direction]}: {m.text}" for m in session.messages)
    rec = a["recommendation"]
    return [
        item,
        dialog,
        finding,
        a["verdict"],
        a["category"],
        a["explanation"],
        " | ".join(f"[{c['turn']}] {c['quote']}" for c in a["citations"]),
        f"{rec['kind']} → {rec['target']}: {rec['change']}" if rec else "",
        *[""] * len(QUESTIONS),
        "",
    ]


def _correct(row: dict) -> bool:
    return row["expected"] == row["got"] and row.get("expected_category") == row.get("got_category")


def export(args) -> None:
    rng = random.Random(args.seed)
    picked: list[tuple[str, dict, dict]] = []  # (источник, строка ответа, контекст)
    subsets = args.subsets / args.out_dir.name
    subsets.mkdir(parents=True, exist_ok=True)
    for name, answers, logs in map(_source, args.source):
        rows = [json.loads(line) for line in answers.read_text("utf-8").splitlines()]
        rows = [r for r in rows if r.get("answer")]
        rng.shuffle(rows)
        # ошибок у сильного агента мало, а каппа на редком классе неустойчива — добираем неверные
        wrong = [r for r in rows if not _correct(r)][: round(args.n * args.wrong_share)]
        rows = wrong + [r for r in rows if _correct(r)][: args.n - len(wrong)]
        context = load_context(logs)
        (subsets / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
        )
        picked += [(name, r, context) for r in rows]
    rng.shuffle(picked)  # источники вперемешку: разметчик не знает, чей ответ
    key, lines = {}, []
    for i, (name, row, context) in enumerate(picked, 1):
        item = f"r{i:02d}"
        key[item] = {"source": name, "case_id": row["case_id"]}
        lines.append(_sheet_row(item, row, context))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)  # BOM — чтобы Excel открыл кириллицу
        writer.writerow(_COLUMNS)
        writer.writerows(lines)
    key_text = json.dumps(key, ensure_ascii=False, indent=1) + "\n"
    (args.out_dir / "key.json").write_text(key_text, "utf-8")
    print(f"Выгружено {len(picked)} ответов: {args.out_dir / 'sheet.csv'}; подмножества: {subsets}")


def _value(cell: str):
    cell = (cell or "").strip().lower()
    if cell in ("1", "да", "yes"):
        return 1
    if cell in ("0", "нет", "no"):
        return 0
    return None  # n/a или не заполнено


def read_sheet(path: Path) -> dict[str, dict[str, int | None]]:
    with path.open(encoding="utf-8-sig") as handle:
        return {r["item"]: {q: _value(r[q]) for q in QUESTIONS} for r in csv.DictReader(handle)}


def read_judge(judge_dir: Path, key: dict) -> dict[str, dict[str, int | None]]:
    by_case = {}
    for path in judge_dir.glob("judge-*.jsonl"):
        source = path.stem.removeprefix("judge-")
        for line in path.read_text("utf-8").splitlines():
            r = json.loads(line)
            if "error" not in r:
                by_case[source, r["case_id"]] = {q: r[q] for q in QUESTIONS}
    return {
        item: by_case[k["source"], k["case_id"]]
        for item, k in key.items()
        if (k["source"], k["case_id"]) in by_case
    }


def kappa_ci(
    pairs: list[tuple[int, int]], seed: int = 7, rounds: int = 2000
) -> tuple[float, float]:
    """95% бутстреп-интервал каппы: на 60 пунктах одна цифра без интервала мало что значит."""
    rng = random.Random(seed)
    values = sorted(
        cohen_kappa([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(rounds)
    )
    return values[int(0.025 * rounds)], values[int(0.975 * rounds) - 1]


def compare(a: dict, b: dict) -> list[str]:
    lines = []
    for q in QUESTIONS:
        pairs = [
            (x, y)
            for item in set(a) & set(b)
            if (x := a[item][q]) is not None and (y := b[item][q]) is not None
        ]
        if not pairs:
            continue
        same = sum(x == y for x, y in pairs) / len(pairs)
        lo, hi = kappa_ci(pairs)
        zeros = sum(x == 0 for x, _ in pairs), sum(y == 0 for _, y in pairs)
        lines.append(
            f"| {q} | {len(pairs)} | {same:.0%} | {cohen_kappa(pairs):.2f} [{lo:.2f}; {hi:.2f}] "
            f"| {zeros[0]} / {zeros[1]} |"
        )
    return lines


def agreement(args) -> None:
    key = json.loads((args.dir / "key.json").read_text("utf-8"))
    sheets = {
        p.stem.removeprefix("sheet-"): read_sheet(p) for p in sorted(args.dir.glob("sheet-*.csv"))
    }
    judge = read_judge(args.judge_dir, key)
    header = [
        "| Вопрос | Пунктов | Совпало | Каппа [95%] | Нулей: первый / второй |",
        "|---|---|---|---|---|",
    ]
    out = [
        f"# Согласие разметки: {args.dir.name}",
        "",
        f"Пунктов в наборе: {len(key)}, с оценкой судьи: {len(judge)}.",
        "",
    ]
    for name, sheet in sheets.items():
        out += [f"## {name} — судья", "", *header, *compare(sheet, judge), ""]
    names = list(sheets)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            out += [
                f"## {x} — {y} (потолок: люди между собой)",
                "",
                *header,
                *compare(sheets[x], sheets[y]),
                "",
            ]
    text = "\n".join(out)
    if args.out:
        args.out.write_text(text, "utf-8")
    print(text)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--source", action="append", required=True, help="имя=ответы.jsonl,логи.jsonl")
    e.add_argument("--n", type=int, default=15, help="ответов из каждого источника")
    e.add_argument("--wrong-share", type=float, default=0.5, help="доля неверных по эталону")
    e.add_argument("--seed", type=int, default=21)
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--subsets", type=Path, default=Path("runs/review"))
    a = sub.add_parser("agreement")
    a.add_argument(
        "--dir", type=Path, required=True, help="папка набора: key.json и sheet-<имя>.csv"
    )
    a.add_argument("--judge-dir", type=Path, required=True, help="judge-<источник>.jsonl")
    a.add_argument("--out", type=Path)
    args = parser.parse_args()
    export(args) if args.command == "export" else agreement(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
