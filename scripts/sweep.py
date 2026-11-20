"""Перебор «модель × режим харнесса» на одной выборке находок: точность, сбои, расход.

    uv run python -m scripts.sweep --logs runs/detector/scenarios-v1.jsonl --per-class 10 \\
        --models qwen3.5-9b,gemma-3-12b-it --modes bare,harness --out runs/iter6-pilot

Ответы каждой пары — в <out>/<модель>__<режим>.jsonl; если файл уже есть, пара не
перезапускается (прогон можно продолжить после сбоя или пополнения баланса). Сводка — <out>/report.md.
"""

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.evaluate import AGENTS, load_records, make_agent, run_agent, sample_cases


def run_pair(
    picked, model: str, agent: str, out: Path, workers: int, big: str | None = None, votes: int = 2
) -> list[dict]:
    path = out / f"{model}__{agent}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows = run_agent(picked, workers, make_agent(agent, model, big, votes))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return rows


def summary_row(model: str, agent: str, rows: list[dict]) -> str:
    answered = [r for r in rows if r["got"] != "ошибка"]
    correct = sum(r["expected"] == r["got"] for r in answered)
    usage = [r["telemetry"] for r in answered]

    def mean(key):
        return statistics.fmean(u[key] for u in usage) if usage else 0

    return (
        f"| {model} | {agent} | **{correct}/{len(rows)}** ({correct / max(len(rows), 1):.0%}) "
        f"| {len(rows) - len(answered)} | {sum(u['retries'] > 0 for u in usage)} "
        f"| {mean('prompt_tokens'):.0f} | {mean('completion_tokens'):.0f} | {mean('seconds'):.1f} "
        f"| {sum(u.get('cost', 0) for u in usage):.4f} |"
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=Path("runs/detector/scenarios-v1.jsonl"))
    parser.add_argument("--per-class", type=int, default=10, help="находок на ожидаемый вердикт")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--models", required=True, help="через запятую")
    parser.add_argument("--modes", default="bare,harness", help=f"через запятую из {AGENTS}")
    parser.add_argument("--big-model", help="каскад: большая модель для расхождений")
    parser.add_argument("--votes", type=int, default=2, help="каскад: ответов маленькой модели")
    parser.add_argument("--case-ids", type=Path, help="файл с case_id: прогнать только эти находки")
    parser.add_argument("--workers", type=int, default=4, help="параллельных вызовов на пару")
    parser.add_argument("--pairs", type=int, default=4, help="пар одновременно")
    parser.add_argument("--out", type=Path, default=Path("runs/sweep"))
    args = parser.parse_args()

    models, modes = args.models.split(","), args.modes.split(",")
    unknown = set(modes) - set(AGENTS)
    if unknown:
        parser.error(f"нет режимов {sorted(unknown)}; есть {AGENTS}")
    picked, _ = sample_cases(list(load_records(args.logs)), args.per_class, args.seed)
    if args.case_ids:
        wanted = set(args.case_ids.read_text("utf-8").split())
        picked = [p for p in picked if p[5].id in wanted]
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = [(m, a) for m in models for a in modes]
    with ThreadPoolExecutor(max_workers=args.pairs) as pool:
        results = list(
            pool.map(
                lambda p: run_pair(picked, *p, args.out, args.workers, args.big_model, args.votes),
                pairs,
            )
        )

    report = [
        f"# Модели × режимы: {args.logs.stem}, {len(picked)} находок "
        f"(до {args.per_class} на вердикт, seed {args.seed})",
        "",
        "| Модель | Режим | Вердикт верен | Сбоев | Повторов | Токенов на вход | на выход | Секунд | Стоимость, $ |",
        "|---|---|---|---|---|---|---|---|---|",
        *(summary_row(m, a, rows) for (m, a), rows in zip(pairs, results, strict=True)),
        "",
        "Сбой — ответ не разобран и после трёх попыток (для голой модели — без возврата ошибки). "
        "Расход — среднее на находку без сбоев.",
    ]
    (args.out / "report.md").write_text("\n".join(report) + "\n", "utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
