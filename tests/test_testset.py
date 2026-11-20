from scripts.testset import build_row


def _raw(gold):
    return {
        "domain": "t",
        "session": {
            "id": "ts-t-1",
            "messages": [{"turn": 0, "direction": "in", "text": "Анна, позвоните Анне: 8 900 1"}],
        },
        "gold": gold,
    }


def test_offsets_are_computed_with_occurrence():
    row, errors = build_row(
        _raw([{"turn": 0, "text": "Ан", "type": "NAME", "cat": "pii", "occurrence": 2}]), 1
    )
    assert errors == []
    assert (row["gold"][0]["start"], row["gold"][0]["end"]) == (16, 18)


def test_errors_name_what_is_wrong():
    _, errors = build_row(
        _raw(
            [
                {"turn": 0, "text": "Ольга", "type": "NAME", "cat": "pii"},
                {"turn": 0, "text": "Анна", "type": "FIO", "cat": "maybe"},
            ]
        ),
        3,
    )
    assert len(errors) == 3
    assert "нет в реплике 0 дословно" in errors[0]
