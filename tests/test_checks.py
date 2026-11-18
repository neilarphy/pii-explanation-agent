"""Проверки кодом (контрольные суммы, поиск по тексту), склейка находок, согласованность вердикта."""

import json

import pytest

from backend.agent.agent import explain_case
from backend.agent.cases import Case, build_cases
from backend.agent.checks import identify, inn_ok, ogrn_ok, scan, snils_ok
from backend.agent.prompt import enclosing_value
from backend.schemas import (
    Candidate,
    Category,
    DetectionLog,
    Entity,
    Message,
    ModelConfig,
    Session,
    Verdict,
)


@pytest.mark.parametrize(
    ("check", "value", "ok"),
    [
        (inn_ok, "7707083893", True),  # ИНН организации
        (inn_ok, "7707083894", False),
        (inn_ok, "500100732259", True),  # ИНН физлица
        (inn_ok, "500100732258", False),
        (snils_ok, "11223344595", True),
        (snils_ok, "11223344596", False),
        (ogrn_ok, "1027700132195", True),  # ОГРН
        (ogrn_ok, "1027700132196", False),
        (ogrn_ok, "304500116000157", True),  # ОГРНИП
    ],
)
def test_checksums(check, value, ok):
    assert check(value) is ok


@pytest.mark.parametrize(
    ("text", "label", "category"),
    [
        ("4276 1234 5678 9014", "CARD", Category.PII),
        ("500100732259", "INN_PERSON", Category.PII),
        ("7707083893", "INN_COMPANY", Category.NONE),
        ("112-233-445 95", "SNILS", Category.PII),
        ("1027700132195", "OGRN", Category.NONE),
        ("4510 298713", "PASSPORT_GROUPED", Category.PII),
        ("4510298713", "PASSPORT", Category.PII),
        ("7083 620142", "PASSPORT_GROUPED", Category.PII),  # сумма ИНН-10 сходится случайно
        ("7083620142", "INN_COMPANY", Category.NONE),
        ("г. Казань, пр-т Мира, д. 43, кв. 154", "ADDRESS", Category.PII),
        ("восемь девять один шесть пять пять пять три три два два", "PHONE_WORDS", Category.PII),
        ("dima точка petrova12 собака mail точка ru", "EMAIL_WORDS", Category.PII),
        ("ab12cd34ef56gh78ij90kl", "TOKEN", Category.SENSITIVE),
    ],
)
def test_identify(text, label, category):
    first = identify(text)[0]
    assert (first.label, first.category) == (label, category)


def test_scan_finds_numbers_that_pass_checks_only():
    text = "СНИЛС 112-233-445 95, заказ 123456789, сумма 25000, ИНН 500100732259"
    found = [(text[s:e], c.label) for s, e, c in scan(text)]
    assert found == [("112-233-445 95", "SNILS"), ("500100732259", "INN_PERSON")]


def test_scan_finds_values_the_detector_cannot_see():
    text = (
        "живу на ул. Красной Армии, дом 5к2, кв 7; офис на ул. Ленина; "
        "тел. плюс семь девять один шесть, пять пять пять, три три, два два; "
        "почта anna собака yandex точка ru; код один два три"
    )
    found = [(text[s:e], c.label) for s, e, c in scan(text)]
    assert found == [
        ("ул. Красной Армии, дом 5к2, кв 7", "ADDRESS"),
        ("плюс семь девять один шесть, пять пять пять, три три, два два", "PHONE_WORDS"),
        ("anna собака yandex точка ru", "EMAIL_WORDS"),
    ]


def test_fragment_gets_the_whole_value_found_by_code():
    session = Session(
        id="s",
        messages=[
            Message(turn=0, direction="in", text="пишите dima точка petrova собака mail точка ru")
        ],
    )
    case = Case("c1", "entity", 0, 7, 11, "dima", "PERSON", "c1", ())
    value, check = enclosing_value(case, session)
    assert (value, check.label) == ("dima точка petrova собака mail точка ru", "EMAIL_WORDS")
    assert (
        enclosing_value(Case("c2", "entity", 0, 0, 6, "пишите", "PERSON", "c2", ()), session)
        is None
    )


def _log(entities, candidates) -> DetectionLog:
    return DetectionLog(
        session_id="s",
        profile="pii-ru-en-v1",
        policy_version="1.0.0",
        model=ModelConfig(id="m", emit_threshold=0.2, accept_threshold=0.5, labels=["person"]),
        rules=[],
        entities=entities,
        candidates=candidates,
    )


def _cand(cid, start, end, text, eid, decision="accepted", score=0.9, label="person"):
    return Candidate(
        id=cid,
        turn=0,
        start=start,
        end=end,
        text=text,
        type="PERSON",
        source="model",
        decision=decision,
        entity_id=eid,
        score=score,
        raw_label=label,
        reason="model_threshold_passed" if decision == "accepted" else "below_model_threshold",
    )


def test_overlapping_entities_merge_and_scan_adds_unseen_numbers():
    text = "Анна Иванова, мой СНИЛС 112-233-445 95"
    session = Session(id="s", messages=[Message(turn=0, direction="in", text=text)])
    log = _log(
        entities=[
            Entity(
                id="s-t0-e1",
                turn=0,
                start=0,
                end=4,
                text="Анна",
                type="PERSON",
                candidate_ids=["s-t0-c1"],
            ),
            Entity(
                id="s-t0-e2",
                turn=0,
                start=0,
                end=12,
                text="Анна Иванова",
                type="PERSON",
                candidate_ids=["s-t0-c2"],
            ),
            Entity(
                id="s-t0-e3",
                turn=0,
                start=5,
                end=12,
                text="Иванова",
                type="PERSON",
                candidate_ids=["s-t0-c3"],
            ),
        ],
        candidates=[
            _cand("s-t0-c1", 0, 4, "Анна", "s-t0-e1", label="first_name"),
            _cand("s-t0-c2", 0, 12, "Анна Иванова", "s-t0-e2"),
            _cand("s-t0-c3", 5, 12, "Иванова", "s-t0-e3", label="last_name"),
            _cand("s-t0-c4", 18, 23, "СНИЛС", None, decision="rejected", score=0.3),
        ],
    )
    cases = build_cases(log, session)
    assert [(c.kind, c.text) for c in cases] == [
        ("entity", "Анна Иванова"),
        ("rejected", "СНИЛС"),
        ("scan", "112-233-445 95"),
    ]
    assert len(cases[0].candidates) == 3
    assert cases[2].type == "NATIONAL_ID"


def _answer(verdict, category):
    def complete(_messages):
        return json.dumps(
            {
                "verdict": verdict,
                "category": category,
                "confidence": 0.9,
                "explanation": "x",
                "citations": [],
                "evidence": [],
                "note": "",
                "recommendation": None,
            }
        )

    return complete


def test_verdict_follows_category():
    text = "ИНН организации 7707083893"
    session = Session(id="s", messages=[Message(turn=0, direction="in", text=text)])
    log = _log([], [])
    (case,) = build_cases(log, session)
    assert case.kind == "scan"
    # LLM сказала «пропуск», но категория none — значит маскировать не нужно
    assert explain_case(case, session, log, _answer("missed_pii", "none")) is None
    answer = explain_case(case, session, log, _answer("no_issue", "sensitive"))
    assert (answer.verdict, answer.category) == (Verdict.MISSED_PII, Category.SENSITIVE)


def test_sensitive_miss_is_not_downgraded_by_wrong_detector_type():
    text = "код с обратной стороны карты 673"
    session = Session(id="s", messages=[Message(turn=0, direction="in", text=text)])
    cand = Candidate(
        id="s-t0-c1",
        turn=0,
        start=29,
        end=32,
        text="673",
        type="PASSPORT",
        source="model",
        decision="rejected",
        reason="below_model_threshold",
        entity_id=None,
        score=0.3,
        raw_label="passport",
    )
    log = _log([], [cand])
    (case,) = build_cases(log, session)
    answer = explain_case(case, session, log, _answer("missed_pii", "sensitive"))
    assert answer.verdict == Verdict.MISSED_PII


def _dialog(*texts):
    from backend.schemas import Message, Session

    return Session(
        id="s",
        messages=[
            Message(turn=i, direction="in" if i % 2 == 0 else "out", text=t)
            for i, t in enumerate(texts)
        ],
    )


def _join(session, turn, fragment):
    from backend.agent.checks import join_across_turns

    start = session.messages[turn].text.index(fragment)
    return join_across_turns(session, turn, start, start + len(fragment))


def test_value_dictated_in_parts_is_joined_across_turns():
    s = _dialog("Номер начинается с 89164.", "Дальше?", "Да, дальше 855707.")
    for turn, part in ((0, "89164"), (2, "855707")):
        joined = _join(s, turn, part)
        assert joined.value == "89164855707" and joined.check.label == "PHONE"
        assert joined.parts == ((0, "89164"), (2, "855707"))


@pytest.mark.parametrize(
    ("texts", "turn", "fragment"),
    [
        # год в email — не часть номера
        (("Почта anna.1992@mail.ru", "Ок", "Паспорт 123456"), 2, "123456"),
        # фрагмент со словами — не группа цифр
        (("Серия 620142", "Ок", "Выдан 12 июля"), 2, "12 июля"),
        # целое значение — не часть
        (("Код 56", "Ок", "Телефон 89367832445"), 2, "89367832445"),
        # в одной реплике склейка не нужна: значение целиком находит scan
        (("Номер 89164 855707",), 0, "855707"),
        # реплики бота не склеиваются
        (("Начало 89164", "Конец 855707"), 0, "89164"),
    ],
)
def test_no_join_where_parts_are_not_one_value(texts, turn, fragment):
    assert _join(_dialog(*texts), turn, fragment) is None
