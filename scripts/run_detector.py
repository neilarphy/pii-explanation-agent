"""Прогон pii-bench через сервис детекторов → runs/detector/pii-bench.jsonl.

    uv run --group eval python -m scripts.run_detector --data N:/pii-runtime/data \\
        --profile N:/pii-runtime/detectors/backend/detectors/config/profile.toml

Каждая строка выхода: {"domain", "session", "log", "gold"} — сессия, DetectionLog
в общем формате и эталонные спаны pii-bench.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from backend.agent.detector_adapter import load_rules, to_detection_log
from backend.schemas import Session
from evaluation import redmadrobot
from evaluation.pii_bench import BenchItem, gold_from_json, gold_to_json, load


def detect(url: str, text: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, help="папка с domain.parquet (pii-bench)")
    parser.add_argument("--scenarios", type=Path, help="jsonl из scripts.generate_scenarios")
    parser.add_argument(
        "--redmadrobot", type=Path, help="test.csv из redmadrobot-rnd/pii_benchmark"
    )
    parser.add_argument("--profile", type=Path, required=True, help="profile.toml детектора")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/detect")
    parser.add_argument("--split", default="domain", choices=["domain", "entity"])
    parser.add_argument("--out", type=Path, default=Path("runs/detector/pii-bench.jsonl"))
    parser.add_argument("--limit", type=int, help="первые N примеров (для проверки)")
    args = parser.parse_args()

    if args.scenarios:
        items = [
            BenchItem(r["domain"], Session.model_validate(r["session"]), gold_from_json(r["gold"]))
            for r in map(json.loads, args.scenarios.read_text("utf-8").splitlines())
        ]
    elif args.redmadrobot:
        items = redmadrobot.load(args.redmadrobot)
    elif args.data:
        items = load(args.data, args.split)
    else:
        parser.error("нужен --data, --scenarios или --redmadrobot")
    items = items[: args.limit]
    rules = load_rules(args.profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.out.open("w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            responses = {
                m.turn: detect(args.url, m.text) for m in item.session.messages if m.text.strip()
            }
            log = to_detection_log(item.session, responses, rules)
            record = {
                "domain": item.domain,
                "session": item.session.model_dump(mode="json"),
                "log": log.model_dump(mode="json"),
                "gold": gold_to_json(item.gold),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 50 == 0 or i == len(items):
                print(f"{i}/{len(items)} сессий, {time.monotonic() - started:.0f} с")
    print(f"Готово: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
