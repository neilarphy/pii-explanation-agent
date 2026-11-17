"""Прогон агента на папке с session.json и detection_log.json.

    uv run python -m backend.agent docs/examples --out runs/s-001.json

Если в папке есть agent_answers.json, вердикты сравниваются с ним как с эталоном.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from backend.agent.agent import explain_session
from backend.agent.llm import client_from_env
from backend.schemas import AgentAnswer, DetectionLog, Session


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Прогон агента на одной сессии")
    parser.add_argument("folder", type=Path, help="папка с session.json и detection_log.json")
    parser.add_argument("--out", type=Path, help="куда сохранить ответы агента (JSON)")
    args = parser.parse_args()

    session = Session.model_validate_json((args.folder / "session.json").read_text("utf-8"))
    log = DetectionLog.model_validate_json((args.folder / "detection_log.json").read_text("utf-8"))
    results = explain_session(session, log, client_from_env())
    answers = [a for _, a in results if a is not None]

    reference_path = args.folder / "agent_answers.json"
    expected: dict[str | None, str] = {}
    if reference_path.exists():
        reference = TypeAdapter(list[AgentAnswer]).validate_json(reference_path.read_text("utf-8"))
        expected = {a.entity_id: a.verdict.value for a in reference}

    matched = 0
    for case, answer in results:
        got = answer.verdict.value if answer else "no_issue"
        want = expected.get(case.entity_id, "—")
        matched += got == want
        print(f"{case.id:<14} «{case.text}»  агент: {got:<15} эталон: {want}")
    if expected:
        print(f"\nСовпало с эталоном: {matched} из {len(results)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = [a.model_dump(mode="json") for a in answers]
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(f"Ответы сохранены: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
