"""redmadrobot: BIO-разметка по токенам → спаны с офсетами в исходном тексте."""

from types import SimpleNamespace

from evaluation.metrics import score_detector
from evaluation.redmadrobot import to_item

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
