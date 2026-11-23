"""redmadrobot: BIO-разметка по токенам → спаны с офсетами в исходном тексте."""

import csv
import hashlib
import io
import json
import sys
from types import SimpleNamespace

import pytest

from evaluation import redmadrobot
from evaluation.metrics import score_detector
from evaluation.redmadrobot import load, to_item

TEXT = "Иван Петров живёт в России, тел. +7 916 555-33-22."
TOKENS = ["иван", "петров", "живёт", "в", "России", ",", "тел", ".", "+7 916 555-33-22", "."]
TAGS = ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "B-COUNTRY", "O", "O", "O", "B-PHONE", "O"]


def test_tokens_are_found_case_insensitively_and_mapped_to_bench_types():
    item = to_item(0, TEXT, TOKENS, TAGS)
    assert item is not None
    spans = [(g.type, g.text, g.category) for g in item.gold]
    assert spans == [
        ("NAME", "Иван", "pii"),
        ("NAME", "Петров", "pii"),
        ("COUNTRY", "России", "none"),
        ("PHONE_NUMBER", "+7 916 555-33-22", "pii"),
    ]
    assert item.session.messages[0].text == TEXT


def test_inside_tags_extend_the_previous_span():
    item = to_item(
        1,
        "ул. Ленина 5",
        ["ул", ".", "Ленина", "5"],
        ["B-STREET", "I-STREET", "I-STREET", "B-HOUSE"],
    )
    assert [(g.text, g.type) for g in item.gold] == [("ул. Ленина", "ADDRESS"), ("5", "ADDRESS")]


def test_item_is_skipped_when_token_is_missing_or_lengths_differ():
    assert to_item(2, "привет", ["пока"], ["O"]) is None
    assert to_item(3, "привет", ["привет"], ["O", "O"]) is None


def test_non_pii_gold_is_not_a_positive_for_the_detector():
    item = to_item(0, TEXT, TOKENS, TAGS)
    report = score_detector([(SimpleNamespace(entities=[]), item.gold)])  # детектор ничего не нашёл
    assert "COUNTRY" not in report.uncovered_gold
    assert sum(s.fn for s in report.by_type.values()) == 3


@pytest.mark.parametrize(
    ("text", "tokens", "tags"),
    [
        ("Anna", ["Anna"], ["I-FIRST_NAME"]),
        ("Anna Lee", ["Anna", "Lee"], ["B-FIRST_NAME", "I-LAST_NAME"]),
        ("Anna x Lee", ["Anna", "x", "Lee"], ["B-FIRST_NAME", "O", "I-FIRST_NAME"]),
        ("Anna Lee", ["Anna", "", "Lee"], ["B-FIRST_NAME", "O", "I-FIRST_NAME"]),
        ("Anna", ["", "Anna"], ["B-FIRST_NAME", "O"]),
        ("Anna", ["Anna", ""], ["B-FIRST_NAME", "I-FIRST_NAME"]),
        ("Anna", ["Anna"], ["B-UNKNOWN"]),
        ("Anna", ["Anna"], ["X-FIRST_NAME"]),
        ("Anna", ["Anna"], [7]),
        ("Anna", [7], ["O"]),
        ("Anna", [], []),
        ("prefix Anna", ["Anna"], ["B-FIRST_NAME"]),
        ("Anna tail", ["Anna"], ["B-FIRST_NAME"]),
    ],
)
def test_invalid_rows_are_rejected(text, tokens, tags):
    assert to_item(0, text, tokens, tags) is None


def test_empty_outside_token_preserves_country_and_city():
    item = to_item(
        1431,
        "Россия, Москва",
        ["Россия", "", ",", "Москва"],
        ["B-COUNTRY", "O", "O", "B-CITY"],
    )
    assert item is not None
    assert [(g.type, g.text) for g in item.gold] == [
        ("COUNTRY", "Россия"),
        ("ADDRESS", "Москва"),
    ]
    assert item.session.id == "rmr-1431"


def test_repeated_tokens_and_unicode_keep_original_code_point_offsets():
    text = "İ 🧪 Анна Анна"
    item = to_item(0, text, ["İ", "🧪", "анна", "анна"], ["O", "O", "B-FIRST_NAME", "B-FIRST_NAME"])
    assert item is not None
    assert [(g.start, g.end, g.text) for g in item.gold] == [(4, 8, "Анна"), (9, 13, "Анна")]
    assert all(text[g.start : g.end] == g.text for g in item.gold)


def test_bad_rows_use_only_safe_reason_codes_and_keep_source_ids(tmp_path, caplog):
    path = tmp_path / "rows.csv"
    marker = "PRIVATE_INPUT_MARKER"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "tokens", "ner_tags"])
        writer.writerow([marker, marker, "[]"])
        writer.writerow([marker, json.dumps([marker]), json.dumps(["B-" + marker])])
        writer.writerow(["Anna", '["Anna"]', '["B-FIRST_NAME"]'])
        writer.writerow(["Anna", '["Anna"]', '["B-FIRST_NAME"]'])
    assert [item.session.id for item in load(path)] == ["rmr-2", "rmr-3"]
    assert [record.getMessage() for record in caplog.records] == [
        "redmadrobot row 0: invalid_json",
        "redmadrobot row 1: invalid_bio_tag",
    ]
    assert marker not in caplog.text


def test_bad_header_and_encoding_raise_safe_errors(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("PRIVATE_INPUT_MARKER", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^invalid_csv_header$"):
        load(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match=r"^invalid_csv_encoding_or_structure$") as error:
        load(path)
    assert error.value.__context__ is None


def test_verified_download_is_atomic_and_audit_has_no_source_text(tmp_path, monkeypatch, capsys):
    path = tmp_path / "test.csv"
    path.write_bytes(b"original")
    data = b"text,tokens,ner_tags\nPRIVATE_INPUT_MARKER,invalid,[]\n"
    monkeypatch.setattr(redmadrobot.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(data))
    with pytest.raises(ValueError, match=r"^sha256_mismatch$"):
        redmadrobot.download(path)
    assert path.read_bytes() == b"original"

    monkeypatch.setattr(redmadrobot, "SHA256", hashlib.sha256(data).hexdigest())
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        ["redmadrobot", "--download", "--input", str(path), "--output-dir", str(output)],
    )
    assert redmadrobot.main() == 1
    assert path.read_bytes() == data
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["revision"] == redmadrobot.REVISION
    assert manifest["sha256"] == redmadrobot.SHA256
    assert manifest["counts"]["rejected"] == 1
    journal = (output / "rejections.jsonl").read_text()
    assert json.loads(journal) == {"source_index": 0, "reason": "invalid_json"}
    captured = capsys.readouterr()
    assert "PRIVATE_INPUT_MARKER" not in journal + captured.out + captured.err


def test_cli_checksum_failure_does_not_emit_audit(tmp_path, monkeypatch, capsys):
    path = tmp_path / "test.csv"
    path.write_bytes(b"PRIVATE_INPUT_MARKER")
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys, "argv", ["redmadrobot", "--input", str(path), "--output-dir", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        redmadrobot.main()
    assert error.value.code == 1
    assert capsys.readouterr().err == "redmadrobot: sha256_mismatch\n"
    assert not output.exists()


def test_blank_csv_text_is_rejected_without_rejecting_empty_o_token(tmp_path, caplog):
    path = tmp_path / "rows.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "tokens", "ner_tags"])
        writer.writerow(["", '[""]', '["O"]'])
        writer.writerow([" \t\u00a0", '[""]', '["O"]'])
        writer.writerow(["Anna", '["", "Anna"]', '["O", "B-FIRST_NAME"]'])
    (item,) = load(path)
    assert item.session.id == "rmr-2"
    assert [span.text for span in item.gold] == ["Anna"]
    assert [record.getMessage() for record in caplog.records] == [
        "redmadrobot row 0: empty_text",
        "redmadrobot row 1: empty_text",
    ]


def test_audit_parses_checksum_verified_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "test.csv"
    data = b'text,tokens,ner_tags\nAnna,"[""Anna""]","[""B-FIRST_NAME""]"\n'
    path.write_bytes(data)
    output = tmp_path / "audit"
    monkeypatch.setattr(redmadrobot, "SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(
        sys, "argv", ["redmadrobot", "--input", str(path), "--output-dir", str(output)]
    )
    scan = redmadrobot._scan

    def change_input_after_verification(handle, reject):
        path.write_bytes(b"replacement")
        return scan(handle, reject)

    monkeypatch.setattr(redmadrobot, "_scan", change_input_after_verification)
    assert redmadrobot.main() == 0
    assert path.read_bytes() == b"replacement"
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    assert manifest["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["counts"]["accepted"] == manifest["counts"]["entities"] == 1
    assert (output / "rejections.jsonl").read_bytes() == b""
