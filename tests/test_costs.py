"""Сводка затрат по сохранённым прогонам."""

import json

from scripts.costs import CAPS_PER_USD, collect, report


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")


def test_exact_cost_estimate_and_unknown(tmp_path):
    runs = tmp_path / "runs"
    tel = {"llm_calls": 1, "prompt_tokens": 1000, "completion_tokens": 1000}
    write(runs / "iter7-x" / "qwen3.8-27b__harness.jsonl", [{"telemetry": {**tel, "cost": 0.002}}])
    write(runs / "iter5-x" / "agent_answers.jsonl", [{"telemetry": tel}])
    write(runs / "iter2-x" / "agent_answers.jsonl", [{"answer": {}}])
    write(runs / "judge-iter7" / "a" / "judge.jsonl", [{"cost": 0.001}])
    by = {r.step: r for r in collect(runs)}

    assert by["iter7-x/qwen3.8-27b__harness.jsonl"].caps == 0.002 * CAPS_PER_USD
    assert by["iter5-x"].caps == 1000 * 0.04 + 1000 * 1.55  # тарифы × токены
    assert by["iter5-x"].basis == "тарифы × токены"
    assert by["iter2-x"].basis == "нет данных"
    assert by["judge-iter7/a"].iteration == "7"
    assert "| 2 | 1 | 1 | — |" in report(list(by.values()))
