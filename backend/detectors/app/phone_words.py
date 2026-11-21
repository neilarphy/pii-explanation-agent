"""Validate Russian spoken phone numbers without changing source text or offsets."""

import re

_VALUES = {
    word: value
    for value, words in enumerate(
        (
            "ноль нуль",
            "один одна одно",
            "два две",
            "три",
            "четыре",
            "пять",
            "шесть",
            "семь",
            "восемь",
            "девять",
        )
    )
    for word in words.split()
}
_VALUES.update(
    zip(
        [
            "десять",
            "одиннадцать",
            "двенадцать",
            "тринадцать",
            "четырнадцать",
            "пятнадцать",
            "шестнадцать",
            "семнадцать",
            "восемнадцать",
            "девятнадцать",
        ],
        range(10, 20),
        strict=True,
    )
)
_VALUES.update(
    zip(
        [
            "двадцать",
            "тридцать",
            "сорок",
            "пятьдесят",
            "шестьдесят",
            "семьдесят",
            "восемьдесят",
            "девяносто",
        ],
        range(20, 100, 10),
        strict=True,
    )
)
_VALUES.update(
    zip(
        [
            "сто",
            "двести",
            "триста",
            "четыреста",
            "пятьсот",
            "шестьсот",
            "семьсот",
            "восемьсот",
            "девятьсот",
        ],
        range(100, 1000, 100),
        strict=True,
    )
)


def valid_phone_words(value: str) -> bool:
    """Accept individual digits and conventional groups up to nine hundred ninety-nine."""
    words = re.split(r"[\s,–—-]+", value.casefold())
    international = words[0] == "плюс"
    if international:
        words = words[1:]
    if not words or any(word not in _VALUES for word in words):
        return False
    values = [_VALUES[word] for word in words]
    chunks = []
    index = 0
    while index < len(values):
        number = values[index]
        index += 1
        if number >= 100 and index < len(values) and 0 < values[index] < 100:
            lower = values[index]
            number += lower
            index += 1
            if lower >= 20 and index < len(values) and 0 < values[index] < 10:
                number += values[index]
                index += 1
        elif 20 <= number < 100 and index < len(values) and 0 < values[index] < 10:
            number += values[index]
            index += 1
        chunks.append(str(number))
    digits = "".join(chunks)
    if digits.startswith("0"):
        return False
    if international:
        return 8 <= len(digits) <= 15
    return len(digits) in (7, 10) or (len(digits) == 11 and digits[0] in "78")
