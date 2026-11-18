"""LLM-судья по ответам агента → judge.jsonl и сводка.

    uv run python -m scripts.judge --answers runs/iter2-domain/agent_answers.jsonl \\
        --logs runs/detector/pii-bench.jsonl --model deepseek-v4-pro
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.agent.agent import parse_json
from backend.agent.cases import build_cases
from backend.agent.llm import client_from_env
from backend.schemas import AgentAnswer, DetectionLog, Session
from evaluation.judge import judge_messages, parse_verdicts

_CASE_ID = re.compile(r"^(?P<session>.+)-t\d+-[a-z]\d+$")


def load_context(logs: Path) -> dict[str, tuple[Session, DetectionLog]]:
    context = {}
    for line in logs.read_text("utf-8").splitlines():
        r = json.loads(line)
        session = Session.model_validate(r["session"])
        context[session.id] = (session, DetectionLog.model_validate(r["log"]))
    return context


def find_case(case_id: str, context):
    session, log = context[_CASE_ID.match(case_id)["session"]]
    case = next(c for c in build_cases(log, session) if c.id == case_id)
    return case, session, log


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--model", required=True, help="модель-судья (не та же, что у агента)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, help="по умолчанию judge.jsonl рядом с ответами")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.answers.read_text("utf-8").splitlines()]
    rows = [r for r in rows if r.get("answer")]
    context = load_context(args.logs)
    complete = client_from_env(args.model)

    def one(row):
        case, session, log = find_case(row["case_id"], context)
        answer = AgentAnswer.model_validate(row["answer"])
        try:
            verdicts = parse_verdicts(
                parse_json(complete(judge_messages(case, session, log, answer)))
            )
        except Exception as exc:  # судья сломался на одном ответе — фиксируем и идём дальше
            return {"case_id": row["case_id"], "error": repr(exc)}
        return {"case_id": row["case_id"], **verdicts.__dict__}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, rows))

    out = args.out or args.answers.with_name("judge.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), "utf-8")

    ok = [r for r in results if "error" not in r]
    lines = [f"# Судья {args.model}: {len(ok)} ответов (ошибок: {len(results) - len(ok)})", ""]
    for key, title in [
        ("faithful", "Объяснение опирается на факты"),
        ("citations_ok", "Цитаты по делу"),
        ("verdict_ok", "Судья согласен с вердиктом и категорией"),
        ("recommendation_ok", "Рекомендация применима"),
    ]:
        scored = [r[key] for r in ok if r[key] is not None]
        share = sum(scored) / len(scored) if scored else 0
        lines.append(f"- {title}: {sum(scored)} из {len(scored)} ({share:.0%})")
    bad = [r for r in ok if not r["faithful"]][:10]
    if bad:
        lines += ["", "## Неподтверждённые утверждения (до 10)", ""]
        lines += [f"- `{r['case_id']}`: {'; '.join(r['unsupported_claims'])}" for r in bad]
    report = out.with_name("judge-report.md")
    report.write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
