import csv
import json
from types import SimpleNamespace

from scripts.study import score


def test_score_compares_conditions_on_agent_errors(tmp_path, capsys):
    key = {
        "A01": {"gold": "маскировать", "agent_right": False},
        "B01": {"gold": "не маскировать", "agent_right": True},
    }
    (tmp_path / "key.json").write_text(json.dumps(key, ensure_ascii=False), "utf-8")
    with (tmp_path / "results-p-v1.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.writer(h)
        w.writerow(
            [
                "participant",
                "variant",
                "item",
                "half",
                "condition",
                "decision",
                "confidence",
                "seconds",
            ]
        )
        w.writerow(["p", "1", "A01", "A", "detector", "маскировать", "4", "12"])
        w.writerow(["p", "1", "B01", "B", "agent", "маскировать", "5", "8"])
    score(SimpleNamespace(dir=tmp_path, out=None))
    lines = capsys.readouterr().out.splitlines()
    assert "| Только детектор | 1/1 (100%) | 0/1 (0%) | 12.0 | 0/1 (0%) | — |" in lines
    assert "| Детектор и объяснение агента | 0/1 (0%) | 0/1 (0%) | 8.0 | — | 1/1 (100%) |" in lines
