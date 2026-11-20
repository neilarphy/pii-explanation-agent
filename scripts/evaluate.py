"""Оценка по логам детектора на pii-bench: метрики детектора и прогон агента.

    uv run python -m scripts.evaluate                     # только метрики детектора
    uv run python -m scripts.evaluate --per-class 60      # + агент, до 60 находок на вердикт
    uv run python -m scripts.evaluate --agent harness --same-as runs/iter4-domain/agent_answers.jsonl
    uv run python -m scripts.evaluate --agent bare --model qwen3.5-9b --per-class 10

Агент сверяется с автоэталоном из разметки pii-bench по политике 152-ФЗ
(evaluation.metrics.expected): вердикт и категория. Выборка стратифицирована по ожидаемому
вердикту, чтобы были и ложные срабатывания, и пропуски, а не одни верные находки.
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from backend.agent.agent import explain_case
from backend.agent.cases import build_cases
from backend.agent.harness import MODES, Cascade, Harness, Telemetry
from backend.agent.llm import chat_from_env
from backend.schemas import DetectionLog, Session
from evaluation.metrics import expected, score_detector
from evaluation.pii_bench import gold_from_json

VERDICTS = ["false_positive", "true_pii", "missed_pii", "no_issue", "uncertain"]
COLUMNS = [*VERDICTS, "ошибка"]
ATTEMPTS = 3


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


def same_cases(records, previous: Path):
    """Те же находки, что в прошлом прогоне, плюс находки кода, которых тогда не было."""
    ids = {json.loads(line)["case_id"] for line in previous.read_text("utf-8").splitlines()}
    pool: dict[str, int] = dict.fromkeys(VERDICTS, 0)
    picked = []
    for domain, session, log, gold in records:
        for case in build_cases(log, session):
            verdict, category = expected(case, gold)
            pool[verdict] += 1
            if case.id in ids or case.kind == "scan":
                picked.append((verdict, category, domain, session, log, case))
    return picked, pool, ids


def pipeline(chat):
    """Конвейер итерации 4 (один вызов LLM) с тем же учётом расхода, что у харнесса."""

    def explain(case, session, log):
        replies = []

        def complete(messages):
            replies.append(chat(messages, None))
            return replies[-1].text

        started = time.perf_counter()
        answer = explain_case(case, session, log, complete)
        telemetry = Telemetry(
            llm_calls=len(replies),
            prompt_tokens=sum(r.prompt_tokens for r in replies),
            completion_tokens=sum(r.completion_tokens for r in replies),
            cost=sum(r.cost for r in replies),
            seconds=round(time.perf_counter() - started, 2),
        )
        return answer, telemetry

    return explain


AGENTS = ["pipeline", *MODES, "cascade"]


def make_agent(agent: str, model: str | None, big_model: str | None = None, votes: int = 2):
    """Агент по имени: конвейер итерации 4, режим харнесса (bare, facts, harness, tools) или
    каскад (маленькая модель votes раз, при расхождении — big_model или «не уверен»)."""
    chat = chat_from_env(model)
    if agent == "pipeline":
        return pipeline(chat)
    if agent == "cascade":
        big = Harness(chat_from_env(big_model)) if big_model else None
        return Cascade(Harness(chat), big, votes).explain
    return Harness(chat, MODES[agent]).explain


def run_agent(picked, workers: int, explain):
    def one(item):
        want, want_cat, domain, session, log, case = item
        for attempt in range(1, ATTEMPTS + 1):  # bothub временами отдаёт 502 или пустой ответ
            try:
                answer, telemetry = explain(case, session, log)
                break
            except Exception as exc:  # одна сломанная находка не должна валить прогон
                if attempt == ATTEMPTS:
                    return {
                        "case_id": case.id,
                        "expected": want,
                        "got": "ошибка",
                        "error": repr(exc),
                    }
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
            "telemetry": asdict(telemetry),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(one, picked), 1):
            rows.append(row)
            print(
                f"[{i}/{len(picked)}] {row['case_id']} эталон {row['expected']} → агент {row['got']}"
            )
    return rows


def agent_report(rows, sizes, previous_ids: set[str] | None = None) -> list[str]:
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
    if previous_ids is not None:
        old = [r for r in answered if r["case_id"] in previous_ids]
        new = [r for r in answered if r["case_id"] not in previous_ids]
        lines += [
            "",
            f"Те же находки, что в прошлом прогоне: {sum(r['expected'] == r['got'] for r in old)} "
            f"из {len(old)}. Находки кода вне прошлой выборки: {sum(r['expected'] == r['got'] for r in new)} "
            f"из {len(new)}.",
        ]
    lines += usage_lines(answered)
    by_domain = Counter((r["domain"], r["expected"] == r["got"]) for r in answered)
    domains = sorted({d for d, _ in by_domain})
    if len(domains) > 1:
        lines += ["", "| Домен / сценарий | Совпало |", "|---|---|"]
        for d in domains:
            ok, total = by_domain[(d, True)], by_domain[(d, True)] + by_domain[(d, False)]
            lines.append(f"| {d} | {ok}/{total} |")
    return lines


def usage_lines(rows) -> list[str]:
    """Расход на находку: вызовы, повторы после проверки, инструменты, токены, время."""
    usage = [r["telemetry"] for r in rows if r.get("telemetry")]
    if not usage:
        return []
    seconds = sorted(u["seconds"] for u in usage)

    def mean(key):
        return statistics.fmean(u[key] for u in usage)

    return [
        "",
        f"Расход на находку (в среднем): вызовов LLM {mean('llm_calls'):.2f}, "
        f"токенов {mean('prompt_tokens'):.0f} на вход и {mean('completion_tokens'):.0f} на выход, "
        f"{statistics.fmean(seconds):.1f} с (90-й перцентиль {seconds[int(0.9 * (len(seconds) - 1))]:.1f} с). "
        f"Повтор после проверки — у {sum(u['retries'] > 0 for u in usage)} находок, "
        f"вызовов инструментов — {sum(u['tool_calls'] for u in usage)}. "
        f"Стоимость по данным провайдера: ${sum(u.get('cost', 0) for u in usage):.4f}.",
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=Path("runs/detector/pii-bench.jsonl"))
    parser.add_argument("--per-class", type=int, default=0, help="находок на вердикт для агента")
    parser.add_argument("--workers", type=int, default=4, help="параллельных вызовов LLM")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--same-as", type=Path, help="agent_answers.jsonl прошлого прогона: те же находки + новые"
    )
    parser.add_argument(
        "--agent",
        choices=AGENTS,
        default="pipeline",
        help="конвейер итерации 4 или режим харнесса: bare, facts, harness, tools",
    )
    parser.add_argument("--model", help="модель вместо LLM_MODEL")
    parser.add_argument("--big-model", help="каскад: большая модель для расхождений")
    parser.add_argument("--votes", type=int, default=2, help="каскад: ответов маленькой модели")
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = parser.parse_args()

    records = list(load_records(args.logs))
    report = [f"# Оценка: {args.logs.stem} ({len(records)} сессий)", "", "## Детектор", ""]
    report += detector_table(records)

    args.out.mkdir(parents=True, exist_ok=True)
    previous_ids = None
    if args.same_as:
        picked, sizes, previous_ids = same_cases(records, args.same_as)
    elif args.per_class:
        picked, sizes = sample_cases(records, args.per_class, args.seed)
    if args.same_as or args.per_class:
        explain = make_agent(args.agent, args.model, args.big_model, args.votes)
        report[1:1] = ["", f"Агент: {args.agent}, модель {args.model or 'из LLM_MODEL'}."]
        rows = run_agent(picked, args.workers, explain)
        (args.out / "agent_answers.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
        )
        report += agent_report(rows, sizes, previous_ids)

    (args.out / "report.md").write_text("\n".join(report) + "\n", "utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
