"""Spoken phones retain source spans and require phone context or an explicit plus."""

from dataclasses import replace

import pytest

from backend.detectors.app.errors import ConfigurationError
from backend.detectors.app.regex_detector import RegexDetector

_DIGITS = "восемь девять один два три четыре пять шесть семь восемь девять"
_GROUPS = "восемь девятьсот двенадцать триста сорок пять шестьдесят семь восемьдесят девять"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"Телефон: {_DIGITS}.", _DIGITS),
        (f"Позвоните: {_GROUPS}.", _GROUPS),
        (f"плюс семь {_DIGITS.split(' ', 1)[1]}", f"плюс семь {_DIGITS.split(' ', 1)[1]}"),
        (f"плюс семь {_GROUPS.split(' ', 1)[1]}", f"плюс семь {_GROUPS.split(' ', 1)[1]}"),
        (f"ТЕЛ.: {_DIGITS.upper()}!", _DIGITS.upper()),
        (f"Мобильный: {_DIGITS}", _DIGITS),
        (f"Телефон:{_DIGITS}", _DIGITS),
        (f"🙂 e\u0301\r\nТелефон:\t{_DIGITS}", _DIGITS),
        (f"Телефон: {_DIGITS.replace(' ', chr(160))}", _DIGITS.replace(" ", chr(160))),
        (f"Телефон: {_DIGITS.replace(' ', '-')}", _DIGITS.replace(" ", "-")),
        (f"Телефон: {_DIGITS.replace(' ', ', ')}", _DIGITS.replace(" ", ", ")),
        (f"Телефон: {_DIGITS.replace(' ', chr(13) + chr(10))}", _DIGITS.replace(" ", "\r\n")),
        (
            "Номер телефона: девятьсот двенадцать триста сорок пять шестьдесят семь восемьдесят девять",
            "девятьсот двенадцать триста сорок пять шестьдесят семь восемьдесят девять",
        ),
        (
            "Телефон: пятьсот пятьдесят пять двенадцать тридцать четыре",
            "пятьсот пятьдесят пять двенадцать тридцать четыре",
        ),
        ("Телефон: пять пять пять ноль один нуль два", "пять пять пять ноль один нуль два"),
        ("Телефон: пять пять пять ноль одна нуль две", "пять пять пять ноль одна нуль две"),
        (
            "плюс один два ноль два пять пять пять ноль один два три",
            "плюс один два ноль два пять пять пять ноль один два три",
        ),
    ],
)
def test_spoken_phone_source_spans(profile, text, expected):
    run = RegexDetector(profile).detect(text)
    candidates = [o for o in run.observations if o.rule_id == "phone_words_ru_v1"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.reason == "rule_passed" and candidate.type == "PHONE"
    assert (candidate.start, candidate.end) == (
        text.index(expected),
        text.index(expected) + len(expected),
    )
    assert text[candidate.start : candidate.end] == expected
    assert next(rule.matches for rule in run.metadata.rules if rule.id == candidate.rule_id) == 1


@pytest.mark.parametrize(
    "text",
    [
        _DIGITS,
        f"СНИЛС: {_DIGITS}",
        f"ИНН: {_DIGITS}",
        f"Номер заказа: {_DIGITS}",
        f"микротелефон: {_DIGITS}",
        "В магазине один два три четыре пять шесть семь восемь девять десять товаров.",
        "Телефон: один два три четыре пять шесть",  # Too short.
        "Телефон: один два три четыре пять шесть семь восемь",  # Not a local/national length.
        f"Телефон: {_DIGITS} ноль",  # Do not accept a shorter prefix.
        "Телефон: ноль один два три четыре пять шесть семь восемь девять",
        "Телефон: один два три четыре пять шесть семь восемь девять ноль один",
        "плюс ноль один два три четыре пять шесть семь восемь девять",
        "плюс один два три четыре пять шесть",
        "плюс " + "один " * 16,
        "Телефон: восемьдесятый девятый первый второй третий",
        "Телефон: минус один два три четыре пять шесть семь восемь девять ноль",
        "Телефон: восемь девять один два три четыре пять 6 семь восемь девять",
        f"Телефон: {_DIGITS} 0",
        f"Телефон: {_DIGITS}—0",
        f"Телефон: {_DIGITS} ٠",
        "Телефон: 8 девять один два три четыре пять шесть семь восемь девять",
    ],
)
def test_non_phone_and_invalid_spoken_numbers(profile, text):
    run = RegexDetector(profile).detect(text)
    assert not any(
        o.rule_id == "phone_words_ru_v1" and o.reason == "rule_passed" for o in run.observations
    )


def test_rejected_spoken_candidate_remains_in_trace(profile):
    text = f"Телефон: {_DIGITS} ноль"
    run = RegexDetector(profile).detect(text)
    candidates = [o for o in run.observations if o.rule_id == "phone_words_ru_v1"]
    assert len(candidates) == 1 and candidates[0].reason == "invalid_format"
    assert text[candidates[0].start : candidates[0].end] == f"{_DIGITS} ноль"


def test_multiple_spoken_phones(profile):
    text = f"Телефон: {_DIGITS}; позвоните: {_GROUPS}"
    found = RegexDetector(profile).detect(text).observations
    assert [text[o.start : o.end] for o in found if o.reason == "rule_passed"] == [_DIGITS, _GROUPS]


def test_long_spoken_sequence_is_not_truncated_into_a_phone(profile):
    text = "плюс " + "один " * 5000
    found = RegexDetector(profile).detect(text).observations
    assert len(found) == 1 and found[0].reason == "invalid_format"
    assert text[found[0].start : found[0].end] == text.rstrip()


def test_spoken_rule_requires_phone_capture(profile):
    rule = next(rule for rule in profile.rules if rule.id == "phone_words_ru_v1")
    broken = replace(rule, pattern="один два три")
    with pytest.raises(ConfigurationError, match="missing phone group"):
        RegexDetector(replace(profile, rules=(broken,)))
