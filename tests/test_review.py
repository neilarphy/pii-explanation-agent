import csv
import json

from scripts.review import QUESTIONS, compare, kappa_ci, read_judge, read_sheet


def test_kappa_ci_brackets_point_estimate():
    pairs = [(1, 1)] * 40 + [(0, 0)] * 15 + [(1, 0)] * 3 + [(0, 1)] * 2
    lo, hi = kappa_ci(pairs)
    assert 0 < lo < 0.8 < hi <= 1


def test_sheet_and_judge_join_by_opaque_item(tmp_path):
    key = {"r01": {"source": "gemma", "case_id": "sc-a-1-t0-e1"}}
    (tmp_path / "judge-gemma.jsonl").write_text(
        json.dumps({"case_id": "sc-a-1-t0-e1", **dict.fromkeys(QUESTIONS, 1)}) + "\n", "utf-8"
    )
    sheet = tmp_path / "sheet-taisia.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["item", *QUESTIONS, "comment"])
        w.writerow(["r01", "1", "да", "n/a", "0", ""])
    human, judge = read_sheet(sheet), read_judge(tmp_path, key)
    assert human["r01"] == {
        "faithful": 1,
        "citations_ok": 1,
        "recommendation_ok": None,
        "verdict_ok": 0,
    }
    rows = compare(human, judge)
    assert len(rows) == 3  # recommendation_ok без пары — не считается
    assert rows[-1].startswith("| verdict_ok | 1 | 0% |")
