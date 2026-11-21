"""CPU-only standard-library regex adapter; safe to import without CUDA or Pydantic."""

import re

from .config import MAX_CANDIDATES, Profile
from .errors import ConfigurationError, TraceLimitError
from .observations import RegexMetadata, RegexObservation, RegexRun, RuleCount


def word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def valid_boundary(text: str, start: int, end: int, validator: str) -> bool:
    left = text[start - 1] if start else ""
    right = text[end] if end < len(text) else ""
    if validator == "email_ascii":
        if left and (left.isalnum() or left in "._%+@-"):
            return False
        if right and (right.isalnum() or right in "_%+@-"):
            return False
        if right == "." and end + 1 < len(text):
            after = text[end + 1]
            if after.isalnum() or after in "-.":
                return False
        return True
    if (left and word_char(left)) or (right and word_char(right)):
        return False
    if validator == "card_pan":
        # A greedy 19-digit match must not pass as a prefix of a longer PAN,
        # including the same ASCII separators permitted by the rule.
        before = start - 1
        after = end
        if before >= 0 and text[before] in " -":
            before -= 1
        if after < len(text) and text[after] in " -":
            after += 1
        if (before >= 0 and text[before].isdigit()) or (
            after < len(text) and text[after].isdigit()
        ):
            return False
    return True


def valid_email(value: str) -> bool:
    local, domain = value.split("@")
    parts = domain.split(".")
    return (
        value.isascii()
        and len(local) <= 64
        and len(value) <= 254
        and all(
            1 <= len(part) <= 63 and not part.startswith("-") and not part.endswith("-")
            for part in parts
        )
        and re.fullmatch("[A-Za-z]{2,63}", parts[-1]) is not None
    )


def luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def validate(text: str, start: int, end: int, validator: str) -> str:
    if not valid_boundary(text, start, end, validator):
        return "invalid_boundary"
    value = text[start:end]
    digits = value.replace(" ", "").replace("-", "")
    if validator == "email_ascii":
        good = valid_email(value)
    elif validator == "phone_plus":
        good = re.fullmatch(r"\+[1-9][0-9]{7,14}", value) is not None
    elif validator == "phone_ru":
        good = (
            re.fullmatch(
                r"(?:\+7|8)[ -]?\([0-9]{3}\)[ -]?[0-9]{3}[ -]?[0-9]{2}[ -]?[0-9]{2}", value
            )
            is not None
        )
    else:
        good = (
            re.fullmatch(r"[0-9](?:[ -]?[0-9]){12,18}", value) is not None
            and 13 <= len(digits) <= 19
            and len(set(digits)) > 1
        )
    if not good:
        return "invalid_format"
    if validator == "card_pan" and not luhn(digits):
        return "checksum_failed"
    return "rule_passed"


class RegexDetector:
    engine = "python.re"
    device = "cpu"

    def __init__(self, profile: Profile):
        self.rules = profile.rules
        self.compiled = []
        for rule in self.rules:
            if rule.validator not in {"email_ascii", "phone_plus", "phone_ru", "card_pan"}:
                raise ConfigurationError(f"regex rule {rule.id}: unknown validator")
            try:
                self.compiled.append(re.compile(rule.pattern))
            except re.error:
                raise ConfigurationError(f"regex rule {rule.id}: invalid pattern") from None

    def detect(self, text: str) -> RegexRun:
        observations = []
        counts = []
        blank = not text or text.isspace()
        for order, (rule, pattern) in enumerate(zip(self.rules, self.compiled, strict=True)):
            count = 0
            if not blank:
                for match in pattern.finditer(text):
                    start, end = match.span(0)
                    if start == end:
                        continue
                    count += 1
                    if len(observations) >= MAX_CANDIDATES:
                        raise TraceLimitError("regex")
                    observations.append(
                        RegexObservation(
                            rule.id,
                            rule.type,
                            order,
                            start,
                            end,
                            validate(text, start, end, rule.validator),
                        )
                    )
            counts.append(RuleCount(rule.id, count))
        return RegexRun(
            tuple(observations),
            RegexMetadata(
                "skipped_blank" if blank else "ok", tuple(counts), self.engine, self.device
            ),
        )
