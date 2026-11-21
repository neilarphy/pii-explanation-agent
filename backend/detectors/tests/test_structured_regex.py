"""Structured PII extraction, checksums, context and exact source spans."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from backend.detectors.app.config import ROOT, load_profile
from backend.detectors.app.regex_detector import RegexDetector
from backend.detectors.app.structured import inn_checksum, snils_checksum


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        ("Телефон: +7 999 123-45-67", "PHONE", "+7 999 123-45-67"),
        ("Телефон: 8 999 123 45 67", "PHONE", "8 999 123 45 67"),
        ("89991234567", "PHONE", "89991234567"),
        ("Телефон: +7\u00a0999\u202f123–45—67", "PHONE", "+7\u00a0999\u202f123–45—67"),
        ("тел.: 999 123 45 67", "PHONE", "999 123 45 67"),
        ("Мобильный: 123-45-67", "PHONE", "123-45-67"),
        ("ИНН: 1234567894", "INN", "1234567894"),
        ("ИНН: 123456789047", "INN", "123456789047"),
        ("ИНН: 1234\u00a056789047", "INN", "1234\u00a056789047"),
        ("СНИЛС: 112-233-445 95", "SNILS", "112-233-445 95"),
        ("11223344595", "SNILS", "11223344595"),
        ("112 233 445 95", "SNILS", "112 233 445 95"),
        ("паспорт: 7518 492137", "PASSPORT", "7518 492137"),
        ("паспорт серия 75 18, номер 492137", "PASSPORT", "серия 75 18, номер 492137"),
        ("водительское удостоверение: 77 14 987654", "DRIVER_LICENSE", "77 14 987654"),
        (
            "Водительское удостоверение серии 77 14 номер 987654",
            "DRIVER_LICENSE",
            "77 14 номер 987654",
        ),
        ("Права: 7714987654", "DRIVER_LICENSE", "7714987654"),
        (
            "свидетельство о рождении, серия III - АГ, номер 112233",
            "BIRTH_CERTIFICATE",
            "III - АГ, номер 112233",
        ),
        ("III-АГ 112233", "BIRTH_CERTIFICATE", "III-АГ 112233"),
        ("военный билет, серия МЗ, номер 0045678", "MILITARY_ID", "МЗ, номер 0045678"),
        ("ВОЕННЫЙ БИЛЕТ, СЕРИЯ МЗ, НОМЕР 0045678", "MILITARY_ID", "МЗ, НОМЕР 0045678"),
        ("У меня АА 1234567", "MILITARY_ID", "АА 1234567"),
        ("полис ОМС: 7702 - 1234 - 5678 - 9012", "OMS_POLICY", "7702 - 1234 - 5678 - 9012"),
        ("ОМС: 7702123456789012", "OMS_POLICY", "7702123456789012"),
        (
            "ЕНП: 7702\u00a01234\u00a05678\u00a09012",
            "OMS_POLICY",
            "7702\u00a01234\u00a05678\u00a09012",
        ),
        ("Дата рождения: 29.02.2000", "DATE_OF_BIRTH", "29.02.2000"),
        ("родилась 1/2/1990", "DATE_OF_BIRTH", "1/2/1990"),
        ("DOB: 1990-03-12", "DATE_OF_BIRTH", "1990-03-12"),
        ("г. Казань, пр-т Мира, д. 43, кв. 154", "ADDRESS", "г. Казань, пр-т Мира, д. 43, кв. 154"),
        (
            "Адрес: ул. Пушкина, д. 10, корп. 2, кв. 5",
            "ADDRESS",
            "ул. Пушкина, д. 10, корп. 2, кв. 5",
        ),
    ],
)
def test_structured_entities(profile, text, kind, expected):
    text = "🙂 e\u0301\r\n" + text + "; конец."
    run = RegexDetector(profile).detect(text)
    found = [o for o in run.observations if o.type == kind and o.reason == "rule_passed"]
    assert found
    assert any(text[o.start : o.end] == expected for o in found)
    assert all(0 <= o.start < o.end <= len(text) for o in run.observations)
    assert sum(rule.matches for rule in run.metadata.rules) == len(run.observations)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Номер заказа: 7518 492137", "PASSPORT"),
        ("Номер заказа: 77 14 987654", "DRIVER_LICENSE"),
        ("1234567894", "INN"),
        ("ИНН: 1234567895", "INN"),
        ("ИНН: 123456789048", "INN"),
        ("ИНН: 0000000000", "INN"),
        ("ИНН: 1234567894000", "INN"),
        ("СНИЛС: 112-233-445 94", "SNILS"),
        ("СНИЛС: 000-000-000 00", "SNILS"),
        ("СНИЛС: 112233445951", "SNILS"),
        ("Паспорт: 7518 492137 9", "PASSPORT"),
        ("Паспорт: 0000 000000", "PASSPORT"),
        ("IIII-АГ 112233", "BIRTH_CERTIFICATE"),
        ("III-АГ 1122334", "BIRTH_CERTIFICATE"),
        ("III-АГ 112233 4", "BIRTH_CERTIFICATE"),
        ("на 1234567", "MILITARY_ID"),
        ("АА 12345678", "MILITARY_ID"),
        ("ОМС: 0000 0000 0000 0000", "OMS_POLICY"),
        ("ОМС: 7702 1234 5678 9012 3", "OMS_POLICY"),
        ("Дата рождения: 29.02.1900", "DATE_OF_BIRTH"),
        ("Дата рождения: 31.04.1990", "DATE_OF_BIRTH"),
        ("Дата рождения: 01.02/1990", "DATE_OF_BIRTH"),
        ("Дата заказа: 12.03.1990", "DATE_OF_BIRTH"),
        ("Дата рождения", "DATE_OF_BIRTH"),
        ("Адрес: улица Пушкина", "ADDRESS"),
        ("Улица, дом, квартира", "ADDRESS"),
        ("899912345678", "PHONE"),
        ("+7 (999 123-45-67", "PHONE"),
        ("+7 999) 123-45-67", "PHONE"),
        ("+7 999 123-45-67 0", "PHONE"),
    ],
)
def test_wrong_formats_and_unrelated_numbers(profile, text, kind):
    found = RegexDetector(profile).detect(text).observations
    assert not any(o.type == kind and o.reason == "rule_passed" for o in found)


@pytest.mark.parametrize("text", ["ИНН: 1234567895", "СНИЛС: 112-233-445 94"])
def test_bad_checksum_is_traced(profile, text):
    found = RegexDetector(profile).detect(text).observations
    assert any(o.reason == "checksum_failed" for o in found)


def test_context_flag_comes_from_actual_prefix(profile):
    detector = RegexDetector(profile)
    labeled = detector.detect("военный билет, серия МЗ, номер 0045678").observations
    bare = detector.detect("МЗ 0045678").observations
    assert labeled[0].contextual and not bare[0].contextual


def test_old_profiles_keep_card_acceptance_without_conflict_policy(profile):
    text = "Номер заказа: 4111 1111 1111 1111"
    old = RegexDetector(replace(profile, resolve_conflicts=False)).detect(text)
    new = RegexDetector(profile).detect(text)
    assert next(o.reason for o in old.observations if o.type == "PAYMENT_CARD") == "rule_passed"
    assert (
        next(o.reason for o in new.observations if o.type == "PAYMENT_CARD") == "context_mismatch"
    )


@pytest.mark.parametrize("digits", ["", "123", "000", "x" * 12])
def test_checksum_helpers_reject_wrong_input(digits):
    assert not inn_checksum(digits) and not snils_checksum(digits)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IP: 192.0.2.1.", "192.0.2.1"),
        ("http://192.0.2.1:8080", "192.0.2.1"),
        ("2001:db8::1", "2001:db8::1"),
        ("[2001:db8::1]:443", "2001:db8::1"),
        ("::1", "::1"),
        ("::ffff:192.0.2.1", "::ffff:192.0.2.1"),
    ],
)
def test_ip_profile_is_optional_and_uses_ipaddress(profile, text, expected):
    optional = load_profile(ROOT / "config/profile-with-ip.toml")
    assert optional.rules[:-2] == profile.rules and optional.resolve_conflicts
    assert not any(o.type == "IP_ADDRESS" for o in RegexDetector(profile).detect(text).observations)
    found = RegexDetector(optional).detect(text).observations
    assert [
        text[o.start : o.end] for o in found if o.type == "IP_ADDRESS" and o.reason == "rule_passed"
    ] == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "999.0.2.1",
        "019.0.2.1",
        "1.192.0.2.1",
        "192.0.2.1234",
        "12:34:56",
        "2001:db8:::1",
        "2001:db8::xyz",
    ],
)
def test_invalid_ip_addresses(text):
    optional = load_profile(ROOT / "config/profile-with-ip.toml")
    found = RegexDetector(optional).detect(text).observations
    assert not any(o.type == "IP_ADDRESS" and o.reason == "rule_passed" for o in found)


def test_large_incomplete_fields_do_not_backtrack():
    script = """
from backend.detectors.app.config import DEFAULT_PROFILE_PATH, load_profile
from backend.detectors.app.regex_detector import RegexDetector
detector = RegexDetector(load_profile(DEFAULT_PROFILE_PATH))
for prefix in ('Паспорт серия', 'Паспорт: 7518', '+7', 'Военный билет серия', 'СНИЛС: 112', 'ОМС: 1234'):
    text = prefix + ' ' * 32000 + 'x'
    assert not detector.detect(text).observations
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
