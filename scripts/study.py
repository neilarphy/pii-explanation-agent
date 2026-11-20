"""Опыт с операторами: помогает ли объяснение агента человеку решить, маскировать ли фрагмент.

    # материалы: страница для участников и ключ для подсчёта
    uv run python -m scripts.study export --out-dir data/labeling/operator-study-v1 \\
        --right main-scenarios=runs/iter7b-scenarios/qwen3.8-27b__harness.jsonl,runs/detector/scenarios-v1.jsonl \\
        --right main-domain=runs/iter7-domain/qwen3.8-27b__harness.jsonl,runs/detector/pii-bench.jsonl \\
        --wrong gemma=runs/iter8-v2-scenarios/gemma-3-12b-it__harness.jsonl,runs/detector/scenarios-v1.jsonl \\
        --wrong qwen30b=runs/iter8-v2-scenarios/qwen3-30b-a3b__cascade.jsonl,runs/detector/scenarios-v1.jsonl

    # итог по файлам участников results-*.csv
    uv run python -m scripts.study score --dir data/labeling/operator-study-v1

Две половины по 12 находок (в каждой 3 неверных ответа агента). Вариант 1: половина A — только
решение детектора, B — с объяснением агента; вариант 2 — наоборот. Так каждый участник решает в обоих
условиях, а находка встречается в обоих условиях у разных участников.
"""

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

from scripts.judge import find_case, load_context
from scripts.review import _source

MASK = {"pii", "sensitive"}
_DIRECTION = {"in": "Пользователь", "out": "Бот"}


def _mask(category: str | None) -> bool:
    return category in MASK


def _detector_line(case) -> str:
    if case.kind == "entity":
        return f"Детектор принял фрагмент как {case.type} — будет замаскирован."
    if case.kind == "rejected":
        score = max((c.score or 0 for c in case.candidates), default=0)
        return f"Детектор отклонил кандидата {case.type} (score {score:.2f}) — маскирования нет."
    return (
        f"Детектор фрагмент не заметил; код нашёл его по формату ({case.type}) — маскирования нет."
    )


def _item(row: dict, context) -> dict:
    case, session, _ = find_case(row["case_id"], context)
    a = row["answer"]
    return {
        "dialog": [
            {"who": _DIRECTION[m.direction], "text": m.text, "turn": m.turn}
            for m in session.messages
        ],
        "mark": {"turn": case.turn, "start": case.start, "end": case.end},
        "fragment": case.text,
        "detector": _detector_line(case),
        "agent": {
            "decision": "маскировать" if _mask(a["category"]) else "не маскировать",
            "explanation": a["explanation"],
            "citations": [f"[{c['turn']}] {c['quote']}" for c in a["citations"]],
        },
    }


def _pick(specs, n: int, want_right: bool, rng, used: set) -> list[tuple[str, dict, dict]]:
    """n ответов поровну из источников: верные (решение «маскировать» совпало) или неверные."""
    out = []
    per = -(-n // len(specs))
    for name, answers, logs in map(_source, specs):
        rows = [json.loads(line) for line in answers.read_text("utf-8").splitlines()]
        rows = [
            r
            for r in rows
            if r.get("answer")
            and r.get("expected_category") != "ambiguous"
            and (_mask(r["expected_category"]) == _mask(r["got_category"])) == want_right
        ]
        rng.shuffle(rows)
        context = load_context(logs)
        taken = 0
        for r in rows:
            session = r["case_id"].rsplit("-t", 1)[0]
            if session in used or taken == per:
                continue
            used.add(session)  # одна находка на сессию: участник не видит диалог дважды
            out.append((name, r, context))
            taken += 1
    return out[:n]


def export(args) -> None:
    rng = random.Random(args.seed)
    used: set[str] = set()
    wrong = _pick(args.wrong, args.wrong_n, False, rng, used)
    right = _pick(args.right, args.n - args.wrong_n, True, rng, used)
    halves: dict[str, list] = {"A": [], "B": []}
    for i, x in enumerate(wrong):
        halves["AB"[i % 2]].append(x)
    for i, x in enumerate(right):
        halves["AB"[i % 2]].append(x)
    items, key = [], {}
    for half, xs in halves.items():
        rng.shuffle(xs)
        for j, (name, row, context) in enumerate(xs, 1):
            item_id = f"{half}{j:02d}"
            items.append({"id": item_id, "half": half, **_item(row, context)})
            key[item_id] = {
                "source": name,
                "case_id": row["case_id"],
                "gold": "маскировать" if _mask(row["expected_category"]) else "не маскировать",
                "agent_right": _mask(row["expected_category"]) == _mask(row["got_category"]),
            }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("study_page.html").read_text("utf-8")
    page = template.replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False))
    (args.out_dir / "study.html").write_text(page, "utf-8")
    key_text = json.dumps(key, ensure_ascii=False, indent=1) + "\n"
    (args.out_dir / "key.json").write_text(key_text, "utf-8")
    print(
        f"{len(items)} находок (неверных ответов агента: {len(wrong)}): {args.out_dir / 'study.html'}"
    )


def _share(xs: list[bool]) -> str:
    return f"{sum(xs)}/{len(xs)} ({sum(xs) / len(xs):.0%})" if xs else "—"


def score(args) -> None:
    key = json.loads((args.dir / "key.json").read_text("utf-8"))
    rows = []
    for path in sorted(args.dir.glob("results-*.csv")):
        with path.open(encoding="utf-8-sig") as handle:
            rows += list(csv.DictReader(handle))
    people = sorted({r["participant"] for r in rows})
    out = [
        f"# Опыт с операторами: {args.dir.name}",
        "",
        f"Участников: {len(people)}, ответов: {len(rows)}. «Не уверен» не считается ни верным, ни"
        " неверным — показан отдельно.",
        "",
        "| Условие | Верно | Не уверен | Медиана секунд на находку |"
        " Ошибся там, где агент неверен | Ошибся там, где агент верен |",
        "|---|---|---|---|---|---|",
    ]
    for cond, title in (("detector", "Только детектор"), ("agent", "Детектор и объяснение агента")):
        rs = [r for r in rows if r["condition"] == cond]
        decided = [r for r in rs if r["decision"] != "не уверен"]
        correct = [r["decision"] == key[r["item"]]["gold"] for r in decided]
        unsure = [r["decision"] == "не уверен" for r in rs]
        seconds = [float(r["seconds"]) for r in rs]
        # одни и те же находки в обоих условиях: где агент неверен, а где верен
        on_wrong = [
            r["decision"] != key[r["item"]]["gold"]
            for r in decided
            if not key[r["item"]]["agent_right"]
        ]
        on_right = [
            r["decision"] != key[r["item"]]["gold"]
            for r in decided
            if key[r["item"]]["agent_right"]
        ]
        median = f"{statistics.median(seconds):.1f}" if seconds else "—"
        out.append(
            f"| {title} | {_share(correct)} | {_share(unsure)} | {median} | {_share(on_wrong)} | "
            f"{_share(on_right)} |"
        )
    text = "\n".join(out) + "\n"
    if args.out:
        args.out.write_text(text, "utf-8")
    print(text)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--right", action="append", required=True, help="имя=ответы,логи: верные")
    e.add_argument("--wrong", action="append", required=True, help="имя=ответы,логи: неверные")
    e.add_argument("--n", type=int, default=24)
    e.add_argument("--wrong-n", type=int, default=6)
    e.add_argument("--seed", type=int, default=5)
    e.add_argument("--out-dir", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", type=Path, required=True)
    s.add_argument("--out", type=Path)
    args = parser.parse_args()
    export(args) if args.command == "export" else score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
