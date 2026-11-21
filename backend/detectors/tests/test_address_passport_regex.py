"""Address boundaries and contextual passports, including unrelated numeric fields."""

import subprocess
import sys
from pathlib import Path

import pytest

from backend.detectors.app.regex_detector import RegexDetector


def accepted(profile, text, kind):
    run = RegexDetector(profile).detect(text)
    assert sum(r.matches for r in run.metadata.rules) == len(run.observations)
    return {
        text[o.start : o.end]
        for o in run.observations
        if o.type == kind and o.reason == "rule_passed"
    }


@pytest.mark.parametrize(
    "value",
    [
        "Самара, ул. Садовая 42, кв 15",
        "г. Самара ул. Садовая 42 кв. 15",
        "г.Самара, ул. Садовая, д. 42",
        "Самарская область, г. Самара, ул. Садовая 42",
        "Самарская область г. Самара ул. Садовая 42",
        "Республика Алтай, г. Горно-Алтайск, ул. Садовая 42",
        "Нижний Новгород, улица Садовая, 42, подъезд 3, этаж 5, кв. 15",
        "Самара, Садовая 42, квартира 15",
        "Санкт-Петербург, Садовый проспект, дом 42, корпус 2, квартира 15",
        "ул. Садовая, 42, кв. 15",
        "ул. Садовая 42 корпус 2 строение 1 кв 15 подьезд 3",
        "пр. Садовый 42, офис 15Б",
        "Садовый проспект 42, 3 этаж",
        "ул. 8 Марта 42",
        "улица 50 лет Победы 42/2А",
        "Самара, ул. Садовая, д. 42, кв. 15, 3 под",
        "ул. Садовая\u00a042,\u202fкв.\u00a015",
    ],
)
def test_address_complete_source_span(profile, value):
    text = f"🙂 e\u0301\r\nДоставка: {value}; позвоните заранее."
    assert accepted(profile, text, "ADDRESS") == {value}


@pytest.mark.parametrize(
    ("prefix", "value"),
    [
        ("Адрес регистрации: ", "Самара, Садовая 42"),
        ("По адресу ", "самара, садовая 42, кв 15"),
        ("Место жительства: ", "Самара, Садовая 42, офис 15"),
        ("Доставка на ", "Самара, Садовая 42"),
    ],
)
def test_unmarked_street_requires_locality_and_address_context(profile, prefix, value):
    assert accepted(profile, prefix + value + "; спасибо", "ADDRESS") == {value}


@pytest.mark.parametrize(
    "prefix",
    [
        "Андрей Смирнов жду на ",
        "Привезите на ",
        "Меня зовут Андрей Смирнов, жду заказ на ",
        "Мой телефон 8 915 333 44 55 жду на ",
    ],
)
def test_address_does_not_swallow_name_phone_or_instructions(profile, prefix):
    assert accepted(profile, prefix + "Садовый проспект 42", "ADDRESS") == {"Садовый проспект 42"}


@pytest.mark.parametrize(
    "text",
    [
        "Адрес: улица Садовая",
        "Петров, Иванов 42",
        "Заказ: Самара, позиция 42",
        "42 товара, 15 заказов",
        "ул. Садовая, д. 123456",
        'Адрес: "Самара", "улица Садовая", "42"',
        "г. Самара, ул. Садовая\n42",
    ],
)
def test_non_addresses_are_not_accepted(profile, text):
    assert not accepted(profile, text, "ADDRESS")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Паспортные данные: 7518492137", {"7518492137"}),
        ("Для открытия счёта паспортные данные: 7518492137", {"7518492137"}),
        ("Для выдачи заказа нужен паспорт. Данные: 7518492137", {"7518492137"}),
        ("Мой паспорт не принимают: 7518 492137", {"7518 492137"}),
        ("Старый паспорт был 7518 492137, новый 7519 382146", {"7518 492137", "7519 382146"}),
        ("Исправьте 7518 492137 в паспортных данных", {"7518 492137"}),
        ("Паспорт серия 75 18, номер 492137", {"серия 75 18, номер 492137"}),
        ("Паспортные данные: серия 7518 номер 492137", {"серия 7518 номер 492137"}),
        ("Паспорт потерян. Помню только номер 492137", {"номер 492137"}),
        ("Паспорт: № 492137, серию не помню", {"№ 492137"}),
        ("Паспортные данные: 75\u00a018\u202f492137", {"75\u00a018\u202f492137"}),
    ],
)
def test_passport_context_and_full_series_value(profile, text, expected):
    assert accepted(profile, "🙂 e\u0301\r\n" + text, "PASSPORT") == expected


@pytest.mark.parametrize(
    "text",
    [
        "7518 492137",
        "Данные для идентификации: 7518492137",
        "Номер заказа: 7518 492137, паспорт предъявлю позже",
        "Паспорт уже показал. ИНН: 7518492137",
        "Паспорт уже показал. Телефон: 7518492137",
        "Паспорт уже показал. Тел.: 7518492137",
        "Паспорт уже показал. Счёт: 7518492137",
        "Паспорт уже показал. Номер заявки: 7518492137",
        "Паспорт уже показал. Номер договора: 7518492137",
        "Паспорт уже показал. Права: 7518492137",
        "Паспорт: 7518 492137 9",
        "Паспорт: 0000 000000",
        "Паспорт: номер 4921379",
        '"Паспорт предъявлен", "Номер: 7518492137"',
        "Паспорт предъявлен\n7518 492137",
        "7518 492137\nПаспорт предъявлен",
        "Загранпаспорт: 7518492137",
        "Паспорт предъявлен, звоните +7518492137",
        "Паспорт предъявлен, почта 7518492137@example.org",
    ],
)
def test_unrelated_and_malformed_passport_numbers_are_rejected(profile, text):
    assert not accepted(profile, text, "PASSPORT")


def test_passport_context_mismatch_remains_explained(profile):
    text = "Номер заказа: 7518 492137"
    candidates = RegexDetector(profile).detect(text).observations
    passport = next(o for o in candidates if o.type == "PASSPORT")
    assert passport.reason == "context_mismatch"
    assert text[passport.start : passport.end] == "7518 492137"


def test_long_incomplete_addresses_and_passports_are_bounded():
    script = """
from backend.detectors.app.config import load_profile
from backend.detectors.app.regex_detector import RegexDetector
d = RegexDetector(load_profile())
for prefix in ('Адрес:', 'г.', 'ул.', 'Паспорт: серия 7518', 'Паспорт: номер'):
    d.detect(prefix + ' ' * 32000 + 'x')
d.detect('Адрес: ' + 'Садовая ' * 4000)
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        timeout=10,
    )
