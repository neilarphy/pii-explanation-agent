"""Генератор сценариев: значения валидны, офсеты считает код, ошибки LLM ловятся."""

import random
import re

import pytest

from backend.agent.checks import identify
from evaluation.scenarios import (
    AMBIGUOUS,
    SCENARIOS,
    apply_fixes,
    draw_values,
    fill,
    generator_messages,
)

BY_ID = {s.id: s for s in SCENARIOS}


def _fake_dialog(keys: list[str]) -> list[dict]:
    """Диалог, в котором каждая заглушка стоит ровно один раз, через реплику."""
    messages = []
    for i, key in enumerate(keys):
        messages.append({"role": "user", "content": f"вот {{{{{key}}}}}, проверьте"})
        messages.append({"role": "assistant", "content": f"Спасибо, принял {i}."})
    return messages


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_fill_puts_values_at_exact_offsets(scenario):
    values = draw_values(scenario, random.Random(1))
    item = fill(scenario, "x", _fake_dialog(list(values)), values)
    texts = {m.turn: m.text for m in item.session.messages}
    assert len(item.gold) == len(values)
    for g in item.gold:
        assert texts[g.turn][g.start : g.end] == g.text
        assert g.category in ("pii", "sensitive", "none")


def test_generated_numbers_pass_or_fail_checks_as_intended():
    values = draw_values(BY_ID["ip_vs_company"], random.Random(3))
    assert identify(values["INN_IP"])[0].label == "INN_PERSON"
    assert identify(values["OGRNIP"])[0].label == "OGRNIP"
    assert identify(values["INN_ORG"])[0].label == "INN_COMPANY"
    assert identify(values["OGRN"])[0].label == "OGRN"
    order = draw_values(BY_ID["order_luhn"], random.Random(3))["ORDER"]
    assert identify(order)[0].label == "CARD"  # ловушка: Луна сходится, но это заказ


def test_split_phone_is_cut_into_two_parts():
    values = draw_values(BY_ID["split_phone"], random.Random(5))
    assert set(values) == {"PHONE_A", "PHONE_B"}
    assert re.fullmatch(r"89\d{9}", values["PHONE_A"] + values["PHONE_B"])


def test_bot_echo_repeats_the_same_address():
    values = draw_values(BY_ID["bot_echo"], random.Random(5))
    assert values["ADDRESS"] == values["ADDRESS_ECHO"]


def test_unknown_or_missing_placeholder_is_an_error():
    scenario = BY_ID["third_party"]
    values = draw_values(scenario, random.Random(1))
    with pytest.raises(ValueError, match="не использованы"):
        fill(scenario, "x", [{"role": "user", "content": "только {{NAME}}"}], values)
    with pytest.raises(ValueError, match="неизвестная"):
        fill(scenario, "x", [{"role": "user", "content": "{{NAME}} {{PHONE}} {{CARD}}"}], values)


def test_generator_prompt_lists_placeholders():
    messages = generator_messages(BY_ID["secrets"], ["CVC", "TOKEN"], "банк")
    assert "{{CVC}}, {{TOKEN}}" in messages[1]["content"]


def test_gold_fixes_are_added_once_with_offsets_from_text():
    record = {
        "session": {"id": "s", "messages": [{"turn": 0, "text": "Меня зовут Алексей, спасибо."}]},
        "gold": [],
    }
    fix = {"session": "s", "turn": 0, "text": "Алексей", "type": "NAME", "cat": "pii"}
    assert apply_fixes(record, [fix]) == 1
    assert apply_fixes(record, [fix]) == 0
    assert record["gold"] == [
        {"turn": 0, "start": 11, "end": 18, "type": "NAME", "text": "Алексей", "cat": "pii"}
    ]
    with pytest.raises(ValueError):
        apply_fixes(record, [{**fix, "text": "Борис"}])


@pytest.mark.parametrize("scenario", AMBIGUOUS, ids=lambda s: s.id)
def test_ambiguous_cases_expect_uncertain(scenario):
    """Спорный случай: эталон ambiguous, верный ответ агента — «не уверен»."""
    from backend.agent.cases import Case
    from evaluation.metrics import expected

    values = draw_values(scenario, random.Random(3))
    item = fill(scenario, "x", _fake_dialog(list(values)), values)
    (span,) = item.gold
    assert span.category == "ambiguous"
    case = Case(
        id="x-t0-e1",
        kind="entity",
        turn=span.turn,
        start=span.start,
        end=span.end,
        text=span.text,
        type="PAYMENT_CARD",
        entity_id="e1",
        candidates=(),
    )
    assert expected(case, item.gold) == ("uncertain", "ambiguous")


def test_bare_number_looks_like_a_card():
    from evaluation.scenarios import bare_luhn_number

    assert identify(bare_luhn_number(random.Random(5)))[0].label == "CARD"
