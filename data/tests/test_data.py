import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from data.loader import convert_row, read_csv, read_jsonl
from data.schema import Example


def row(text="Почта: demo@example.invalid", tokens=None, tags=None):
    return {
        "text": text,
        "tokens": json.dumps(tokens or ["Почта", ":", "demo@example.invalid"]),
        "ner_tags": json.dumps(tags or ["O", "O", "B-EMAIL"]),
    }


def test_offsets_and_pending_run():
    example = convert_row(row(), 0)
    entity = example.gold_entities[0]
    assert (entity.start, entity.end, entity.text) == (7, 27, "demo@example.invalid")
    assert example.detections is None
    assert Example.model_validate_json(example.model_dump_json()) == example


def test_repeated_unicode_tokens_and_bio_merge():
    example = convert_row(
        row("Иван Иван\tИван", ["Иван", "Иван", "Иван"], ["O", "B-FIRST_NAME", "I-FIRST_NAME"]), 0
    )
    assert example.gold_entities[0].text == "Иван\tИван"
    assert example.gold_entities[0].start == 5


def test_case_alignment_preserves_text():
    example = convert_row(
        row(
            "www.EXAMPLE.invalid",
            ["WWW", ".", "EXAMPLE", ".", "INVALID"],
            ["B-URL", "I-URL", "I-URL", "I-URL", "I-URL"],
        ),
        0,
    )
    assert example.gold_entities[0].text == "www.EXAMPLE.invalid"


@pytest.mark.parametrize(
    "payload",
    [
        row("x y", ["y"], ["O"]),
        row("x y", ["x"], ["O"]),
        row("x", ["x"], ["I-EMAIL"]),
        row("x y", ["x", "y"], ["B-EMAIL", "I-PHONE"]),
        row("x", ["x"], ["B-UNKNOWN"]),
        row("x", ["x"], ["bad"]),
        {"text": "x", "tokens": "[]", "ner_tags": "[]"},
        {"text": "x", "tokens": '["x"]', "ner_tags": "[]"},
        {"text": "x", "tokens": "null", "ner_tags": "null"},
        {"text": "x", "tokens": "broken", "ner_tags": "[]"},
    ],
)
def test_invalid_rows(payload):
    with pytest.raises(ValueError):
        convert_row(payload, 0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["gold_entities"][0].update(start=-1),
        lambda x: x["gold_entities"][0].update(end=100),
        lambda x: x["gold_entities"][0].update(text="wrong"),
        lambda x: x["gold_entities"][0].update(message_id="missing"),
        lambda x: x["gold_entities"][0].update(start=True),
        lambda x: x["messages"].append(x["messages"][0]),
        lambda x: x.update(detections=[]),
        lambda x: x.update(gold_verdicts=[{"label": "FN", "gold_entity_id": "e0"}]),
        lambda x: x.update(unknown=True),
    ],
)
def test_invalid_contract(mutation):
    payload = convert_row(row(), 0).model_dump()
    mutation(payload)
    with pytest.raises(ValidationError):
        Example.model_validate(payload)


def actual_payload():
    payload = convert_row(row(), 0).model_dump()
    payload.update(detections=[], trace_origin="actual", detector_run_id="run-1")
    return payload


def test_actual_miss_requires_real_run():
    payload = actual_payload()
    payload["gold_verdicts"] = [{"label": "FN", "gold_entity_id": "e0"}]
    assert Example.model_validate(payload).gold_verdicts[0].label == "FN"


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence(confidence):
    payload = actual_payload()
    payload["detections"] = [
        dict(
            payload["gold_entities"][0],
            detector="fixture",
            detector_version="1",
            confidence=confidence,
        )
    ]
    payload["trace_origin"] = "unit_test"
    with pytest.raises(ValidationError):
        Example.model_validate(payload)


def test_tn_not_allowed_with_gold():
    payload = actual_payload()
    payload["gold_verdicts"] = [{"label": "TN"}]
    with pytest.raises(ValidationError):
        Example.model_validate(payload)


def test_unknown_reference():
    payload = actual_payload()
    payload["gold_verdicts"] = [{"label": "FN", "gold_entity_id": "missing"}]
    with pytest.raises(ValidationError):
        Example.model_validate(payload)


def test_jsonl_duplicates_and_bad_line(tmp_path):
    path = tmp_path / "examples.jsonl"
    text = convert_row(row(), 0).model_dump_json()
    path.write_text(text + "\n" + text + "\n")
    with pytest.raises(ValueError, match="line 2"):
        list(read_jsonl(path))
    path.write_text("\n")
    with pytest.raises(ValueError, match="line 1"):
        list(read_jsonl(path))


def test_bad_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("text,tokens\n")
    with pytest.raises(ValueError, match="header"):
        list(read_csv(path))


def test_committed_examples():
    assert len(list(read_jsonl(Path("data/examples/synthetic.jsonl")))) == 2


def test_empty_token_rejected():
    with pytest.raises(ValueError, match="invalid token"):
        convert_row(row("x", [""], ["O"]), 0)


def test_cli_offline_pipeline(tmp_path, monkeypatch):
    import csv
    import sys

    from data.__main__ import main

    csv_path = tmp_path / "input.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "tokens", "ner_tags"])
        writer.writeheader()
        writer.writerows([row(), row(), row("x", ["x"], ["I-EMAIL"])])
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        ["data", "load-redmadrobot", "--input", str(csv_path), "--output-dir", str(output)],
    )
    main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"] == {
        "rows": 3,
        "accepted": 1,
        "rejected": 1,
        "duplicates": 1,
        "entities": 1,
    }
    assert manifest["input_mode"] == "local_unverified"
    assert len(list(read_jsonl(output / "examples.jsonl"))) == 1
    assert json.loads((output / "rejections.jsonl").read_text())["source_id"] == "2"
    with pytest.raises(FileExistsError):
        main()


def test_schema_matches_model():
    assert json.loads(Path("data/example.schema.json").read_text()) == Example.model_json_schema()
