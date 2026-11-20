"""Пилот чтения диалога LLM (итерация 8): находит ли модель ПДн, которых не видят детектор и код.

Этап 1 — чтение (один вызов на сессию):
    uv run python -m scripts.read_dialogs read --logs runs/detector/redmadrobot.jsonl \\
        --invisible 30 --clean 15 --model qwen3.8-27b --out runs/iter8-read/redmadrobot

Этап 2 — разбор новых находок харнессом (объединение находок всех читателей в --out):
    uv run python -m scripts.read_dialogs explain --logs ... --out runs/iter8-read/redmadrobot

Этап 3 — отчёт без вызовов LLM:
    uv run python -m scripts.read_dialogs report --logs ... --out runs/iter8-read/redmadrobot

База — находки детектора и поиска кодом шире (`build_cases(wide=True)`): LLM нужна только для
того, что не нашёл код. Сессии: `--invisible N` с невидимыми настоящими ПДн, `--clean N` без
невидимых вовсе (проверка ложных тревог), или `--all`.
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from backend.agent.cases import Case, build_cases
from backend.agent.harness import Harness
from backend.agent.llm import chat_from_env
from backend.agent.reader import read_session
from evaluation.coverage import OUT_OF_PROFILE, POLICY_NONE, REAL, classify, profile_types
from evaluation.metrics import expected
from scripts.evaluate import load_records

CAPS = 700_000
ATTEMPTS = 3


def _n(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _overlap(turn, start, end, g) -> bool:
    return turn == g.turn and start < g.end and g.start < end


def sessions(args):
    records = [(s, log, gold) for _, s, log, gold in load_records(args.logs)]
    if args.all:
        return records
    types = profile_types(records[0][1])
    invisible, clean = [], []
    for s, log, gold in records:
        cases = build_cases(log, s, wide=True)
        hidden = [
            g
            for g in gold
            if g.category not in ("none", "ambiguous")
            and not any(_overlap(c.turn, c.start, c.end, g) for c in cases)
        ]
        if any(classify(g, types) in (REAL, OUT_OF_PROFILE) for g in hidden):
            invisible.append((s, log, gold))
        elif not hidden:
            clean.append((s, log, gold))
    rng = random.Random(args.seed)
    return rng.sample(invisible, min(args.invisible, len(invisible))) + rng.sample(
        clean, min(args.clean, len(clean))
    )


def case_to_json(c: Case) -> dict:
    return {k: v for k, v in asdict(c).items() if k != "candidates"}


def cmd_read(args):
    chat = chat_from_env(args.model)
    picked = sessions(args)

    def one(item):
        s, log, _ = item
        result = read_session(chat, s, build_cases(log, s, wide=True))
        r = result.reply
        return {
            "session_id": s.id,
            "found": [case_to_json(c) for c in result.cases],
            "dropped": result.dropped,
            "error": result.error,
            "telemetry": {
                "llm_calls": 1,
                "prompt_tokens": r.prompt_tokens if r else 0,
                "completion_tokens": r.completion_tokens if r else 0,
                "cost": r.cost if r else 0.0,
            },
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, picked))
    out = args.out / f"{args.model}__read.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    usd = sum(r["telemetry"]["cost"] for r in rows)
    n = sum(len(r["found"]) for r in rows)
    print(
        f"{len(rows)} сессий, новых находок {n}, сбоев {sum(bool(r['error']) for r in rows)}; "
        f"${usd:.4f} ≈ {usd * CAPS:,.0f} капсов ({usd * CAPS / max(len(rows), 1):,.0f} на сессию)"
    )
    print(f"→ {out}")


def read_files(out: Path):
    return {p.name.split("__")[0]: p for p in sorted(out.glob("*__read.jsonl"))}


def load_reads(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def union_cases(out: Path) -> dict[str, list[Case]]:
    """Находки всех читателей по сессиям; совпадающие по месту — одна находка."""
    by_session: dict[str, list[Case]] = {}
    for path in read_files(out).values():
        for row in load_reads(path):
            have = by_session.setdefault(row["session_id"], [])
            for c in row["found"]:
                case = Case(**c, candidates=())
                if not any(
                    h.turn == case.turn and h.start == case.start and h.end == case.end
                    for h in have
                ):
                    have.append(case)
    for sid, cases in by_session.items():  # одинаковые id у разных читателей — перенумеровать
        for i, c in enumerate(cases, 1):
            cases[i - 1] = Case(**{**case_to_json(c), "id": f"{sid}-t{c.turn}-l{i}"}, candidates=())
    return by_session


def cmd_explain(args):
    records = {s.id: (s, log, gold) for _, s, log, gold in load_records(args.logs)}
    harness = Harness(chat_from_env(args.explain_model))
    items = [
        (case, *records[sid]) for sid, cases in union_cases(args.out).items() for case in cases
    ]

    def one(item):
        case, s, log, gold = item
        want, want_cat = expected(case, gold)
        for attempt in range(1, ATTEMPTS + 1):
            try:
                answer, telemetry = harness.explain(case, s, log)
                break
            except Exception as exc:
                if attempt == ATTEMPTS:
                    return {
                        "case_id": case.id,
                        "expected": want,
                        "got": "ошибка",
                        "error": repr(exc),
                    }
        return {
            "case_id": case.id,
            "session_id": s.id,
            "kind": case.kind,
            "type": case.type,
            "text": case.text,
            "turn": case.turn,
            "start": case.start,
            "end": case.end,
            "expected": want,
            "expected_category": want_cat,
            "got": answer.verdict.value if answer else "no_issue",
            "got_category": answer.category.value if answer else None,
            "answer": answer.model_dump(mode="json") if answer else None,
            "telemetry": asdict(telemetry),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, items))
    out = args.out / f"{args.explain_model}__harness.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    usd = sum(r.get("telemetry", {}).get("cost", 0) for r in rows)
    print(f"{len(rows)} находок; ${usd:.4f} ≈ {usd * CAPS:,.0f} капсов → {out}")


def cmd_report(args):
    records = {s.id: (s, log, gold) for _, s, log, gold in load_records(args.logs)}
    types = profile_types(next(iter(records.values()))[1])
    answers_path = args.out / f"{args.explain_model}__harness.jsonl"
    answers = {}
    if answers_path.exists():
        for r in load_reads(answers_path):
            answers[(r["session_id"], r["turn"], r["start"], r["end"])] = r
    lines = [f"# Чтение диалога LLM: {args.title}", ""]
    lines += [
        "| Читатель | Сессий | Настоящих невидимых | Модель назвала из них | Новых находок "
        "| Из них не ПДн | После разбора агентом: пропусков подтверждено | Ложных тревог "
        "| Капсов на сессию (чтение) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for reader, path in read_files(args.out).items():
        rows = load_reads(path)
        real_total = real_found = new = not_pii = confirmed = false_alarm = 0
        for row in rows:
            s, log, gold = records[row["session_id"]]
            cases = build_cases(log, s, wide=True)
            found = [Case(**c, candidates=()) for c in row["found"]]
            for g in gold:
                if g.category in ("none", "ambiguous") or any(
                    _overlap(c.turn, c.start, c.end, g) for c in cases
                ):
                    continue
                if classify(g, types) in (REAL, OUT_OF_PROFILE):
                    real_total += 1
                    real_found += any(_overlap(c.turn, c.start, c.end, g) for c in found)
            for c in found:
                new += 1
                hits = [
                    g
                    for g in gold
                    if g.category not in ("none", "ambiguous")
                    and _overlap(c.turn, c.start, c.end, g)
                ]
                pii = any(classify(g, types) not in POLICY_NONE for g in hits)
                not_pii += not pii
                a = answers.get((row["session_id"], c.turn, c.start, c.end))
                if a and a["got"] == "missed_pii":
                    confirmed += pii
                    false_alarm += not pii
        usd = sum(r["telemetry"]["cost"] for r in rows)
        lines.append(
            f"| {reader} | {len(rows)} | {real_total} | {real_found} | {new} | {not_pii} "
            f"| {confirmed if answers else '—'} | {false_alarm if answers else '—'} "
            f"| {_n(usd * CAPS / max(len(rows), 1))} |"
        )
    if answers:
        usd = sum(r.get("telemetry", {}).get("cost", 0) for r in answers.values())
        lines += [
            "",
            f"Разбор агентом ({args.explain_model}): {len(answers)} находок (объединение "
            f"читателей по точному месту), {_n(usd * CAPS / max(len(answers), 1))} капсов на "
            f"находку, всего {_n(usd * CAPS)}.",
        ]
    text = "\n".join(lines) + "\n"
    (args.out / "report.md").write_text(text, "utf-8")
    print(text)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["read", "explain", "report"])
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="qwen3.8-27b", help="модель-читатель")
    ap.add_argument("--explain-model", default="qwen3.8-27b", help="модель разбора находок")
    ap.add_argument("--invisible", type=int, default=30)
    ap.add_argument("--clean", type=int, default=15)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    {"read": cmd_read, "explain": cmd_explain, "report": cmd_report}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
