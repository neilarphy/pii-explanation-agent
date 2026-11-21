import dataclasses
import json
import os
import subprocess
import sys

import pytest
from app.errors import ConfigurationError, TraceLimitError
from app.regex_detector import RegexDetector


@pytest.mark.parametrize(
    ("text", "spans"),
    [
        ("a@example.org a@example.org", [(0, 13), (14, 27)]),
        ("a@example.org.", [(0, 13)]),
        ("🙂a@example.org", [(1, 14)]),
        ("я a@example.org", [(2, 15)]),
        ("🙂 Анна: a@example.org", [(8, 21)]),
        ("x..a@example.org", []),
        ("x@a@example.org", []),
        ("a@-example.org", []),
        ("a@example-.org", []),
        ("a@example.org.x", []),
        ("a@example.org..", []),
        ("a@example.org.-", []),
        ("a@example.org@b", []),
        ("яa@example.org", []),
        ("a@example.orgя", []),
        ("a" * 65 + "@example.org", []),
        ("a@" + "x" * 64 + ".org", []),
        ("a@example.123", []),
        ("a@example." + "x" * 64, []),
    ],
)
def test_email(profile, text, spans):
    run = RegexDetector(profile).detect(text)
    accepted = [
        o for o in run.observations if o.rule_id == "email_ascii_v1" and o.reason == "rule_passed"
    ]
    assert [(o.start, o.end) for o in accepted] == spans
    assert sum(rule.matches for rule in run.metadata.rules) == len(run.observations)
    assert len(run.metadata.rules) == 4


@pytest.mark.parametrize(
    ("text", "rule", "reason"),
    [
        ("+12025550123", "phone_plus_v1", "rule_passed"),
        ("+7 (999) 000-00-00", "phone_ru_v1", "rule_passed"),
        ("8 (999) 000-00-00", "phone_ru_v1", "rule_passed"),
        ("4111 1111 1111 1111", "card_pan_v1", "rule_passed"),
        ("4111 1111 1111 1112", "card_pan_v1", "checksum_failed"),
        ("0000000000000", "card_pan_v1", "invalid_format"),
        ("я+12025550123", "phone_plus_v1", "invalid_boundary"),
        ("+12025550123я", "phone_plus_v1", "invalid_boundary"),
        ("_+12025550123", "phone_plus_v1", "invalid_boundary"),
        ("411111111111111111111", "card_pan_v1", "invalid_boundary"),
        ("4111 1111 1111 1111 1111", "card_pan_v1", "invalid_boundary"),
        ("+12025550123456789", "phone_plus_v1", "invalid_boundary"),
    ],
)
def test_numbers(profile, text, rule, reason):
    observations = [
        o for o in RegexDetector(profile).detect(text).observations if o.rule_id == rule
    ]
    assert observations and observations[0].reason == reason
    if reason in ("rule_passed", "checksum_failed", "invalid_format"):
        assert (observations[0].start, observations[0].end) == (0, len(text))


def test_rules_independent_of_labels(profile):
    one = RegexDetector(dataclasses.replace(profile, labels=("person",))).detect("a@example.org")
    all_labels = RegexDetector(profile).detect("a@example.org")
    assert one == all_labels
    assert one.metadata.device == "cpu"


def test_invalid_pattern(profile):
    rule = dataclasses.replace(profile.rules[0], pattern="(")
    with pytest.raises(ConfigurationError, match="email_ascii_v1"):
        RegexDetector(dataclasses.replace(profile, rules=(rule,)))


def test_limit_inclusive(profile):
    adapter = RegexDetector(profile)
    assert len(adapter.detect("a@b.co " * 4096).observations) == 4096
    with pytest.raises(TraceLimitError):
        adapter.detect("a@b.co " * 4097)


def test_standalone_standard_library():
    script = """
import json, sys
from app.config import load_profile
from app.regex_detector import RegexDetector
r=RegexDetector(load_profile()).detect('🙂 a@example.org a@example.org')
assert 'torch' not in sys.modules and 'pydantic' not in sys.modules
print(json.dumps([(o.start,o.end) for o in r.observations]))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == [[2, 15], [16, 29]]
