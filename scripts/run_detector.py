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
from evaluation.pii_bench import gold_to_json, load


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
    parser.add_argument("--data", type=Path, required=True, help="папка с domain.parquet")
    parser.add_argument("--profile", type=Path, required=True, help="profile.toml детектора")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/detect")
    parser.add_argument("--split", default="domain", choices=["domain", "entity"])
    parser.add_argument("--out", type=Path, default=Path("runs/detector/pii-bench.jsonl"))
    parser.add_argument("--limit", type=int, help="первые N примеров (для проверки)")
    args = parser.parse_args()

    items = load(args.data, args.split)[: args.limit]
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
