"""Генерация сложных сценариев → data/scenarios/<name>.jsonl.

    uv run python -m scripts.generate_scenarios --per-scenario 5 --model qwen3.8-27b

Строка выхода: {"domain", "session", "gold"} — как у pii-bench, только без лога детектора;
лог добавляет scripts.run_detector --scenarios.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from backend.agent.llm import client_from_env
from evaluation.pii_bench import gold_to_json
from evaluation.scenarios import DOMAINS, SCENARIOS, draw_values, fill, generator_messages


def parse_messages(raw: str) -> list[dict]:
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"нет JSON-массива: {raw[:200]!r}")
    messages = json.loads(raw[start : end + 1])
    if not messages or any(m.get("role") not in ("user", "assistant") for m in messages):
        raise ValueError("неверные роли")
    return messages


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-scenario", type=int, default=5)
    parser.add_argument("--model", help="модель-генератор (по умолчанию LLM_MODEL)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("data/scenarios/scenarios-v1.jsonl"))
    args = parser.parse_args()

    complete = client_from_env(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written, failed = 0, []
    with args.out.open("w", encoding="utf-8", newline="\n") as out:
        for scenario in SCENARIOS:
            for i in range(args.per_scenario):
                item_id = f"sc-{scenario.id}-{i + 1}"
                rng = random.Random(f"{args.seed}-{item_id}")
                values = draw_values(scenario, rng)
                domain = DOMAINS[i % len(DOMAINS)]
                for attempt in range(1, args.retries + 1):
                    try:
                        raw = complete(generator_messages(scenario, list(values), domain))
                        item = fill(scenario, item_id, parse_messages(raw), values)
                        break
                    except (ValueError, KeyError, json.JSONDecodeError) as exc:
                        print(f"{item_id}: попытка {attempt} — {exc}")
                else:
                    failed.append(item_id)
                    continue
                record = {
                    "domain": scenario.id,
                    "session": item.session.model_dump(mode="json"),
                    "gold": gold_to_json(item.gold),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                print(
                    f"{item_id}: {len(item.session.messages)} реплик, {len(item.gold)} эталонных спанов"
                )
    print(f"Готово: {written} диалогов в {args.out}; не удалось: {failed or 'нет'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
