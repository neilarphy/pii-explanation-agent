"""Сводка по N сессиям из ответов агента → summary.json и задача summary.md для Claude Code.

    uv run python -m scripts.summary --answers runs/iter7-scenarios/agent_answers.jsonl \
        --logs runs/detector/scenarios-v1.jsonl --out runs/iter7-scenarios
"""

import argparse
import json
import sys
from pathlib import Path

from backend.agent.summary import Finding, build_summary
from backend.schemas import AgentAnswer
from scripts.judge import _CASE_ID, find_case, load_context


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    context = load_context(args.logs)
    findings = []
    for line in args.answers.read_text("utf-8").splitlines():
        row = json.loads(line)
        if row["got"] == "ошибка":
            continue
        case, _, _ = find_case(row["case_id"], context)
        answer = AgentAnswer.model_validate(row["answer"]) if row.get("answer") else None
        findings.append(Finding(_CASE_ID.match(row["case_id"])["session"], case, answer))

    summary = build_summary(findings)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n", "utf-8")
    (args.out / "summary.md").write_text(summary.markdown + "\n", "utf-8")
    print(summary.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
