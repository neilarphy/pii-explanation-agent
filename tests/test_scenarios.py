"""Генератор сценариев: значения валидны, офсеты считает код, ошибки LLM ловятся."""

import json
import random
import re
import runpy
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agent.checks import identify
from backend.schemas import Session
from evaluation.metrics import expected
from evaluation.pii_bench import gold_from_json, gold_to_json
from evaluation.scenarios import (
    SCENARIOS,
    SPLITS,
    apply_fixes,
    draw_values,
    fill,
    generator_messages,
    group_split,
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


def test_variants_share_parent_split_regardless_of_order():
    scenario = BY_ID["test_card"]
    values = draw_values(scenario, random.Random(1))
    groups = [f"parent-{i}" for i in range(100)]

    def assignments(order):
        result = {split: set() for split in SPLITS}
        for group in order:
            for variant in ("a", "b"):
                messages = [{"role": "user", "content": variant + " {{TEST_CARD}}"}]
                item = fill(scenario, f"{group}-{variant}", messages, values, group_id=group)
                assert item.session.meta["group_id"] == group
                assert item.session.meta["split"] == group_split(group)
                result[item.session.meta["split"]].add(group)
        return result

    by_split = assignments(groups)
    assert by_split == assignments(reversed(groups))
    for split in SPLITS:
        assert by_split[split]
        assert not by_split[split].intersection(
            set.union(*(by_split[other] for other in SPLITS if other != split))
        )


def test_emoji_offsets_round_trip_to_utf16_and_test_card_stays_none():
    scenario = BY_ID["test_card"]
    values = draw_values(scenario, random.Random(1))
    messages = [{"role": "user", "content": "🧪 Карта {{TEST_CARD}} 😀"}]
    item = fill(scenario, "emoji", messages, values)
    session = Session.model_validate_json(item.session.model_dump_json())
    gold = gold_from_json(json.loads(json.dumps(gold_to_json(item.gold))))
    (span,) = gold
    text = session.messages[span.turn].text
    assert text[span.start : span.end] == values["TEST_CARD"]
    start = len(text[: span.start].encode("utf-16-le")) // 2
    end = len(text[: span.end].encode("utf-16-le")) // 2
    assert start == span.start + 1
    assert text.encode("utf-16-le")[start * 2 : end * 2].decode("utf-16-le") == span.text
    assert span.cat == "none"
    case = SimpleNamespace(kind="entity", turn=span.turn, start=span.start, end=span.end)
    assert expected(case, gold) == ("false_positive", "none")


def test_v2_cli_ids_are_disjoint_from_committed_v1_and_stable(tmp_path, monkeypatch):
    script = runpy.run_path("scripts/generate_scenarios.py")
    main = script["main"]

    def complete(messages):
        keys = re.findall(r"\{\{[A-Z_]+\}\}", messages[-1]["content"])
        return json.dumps([{"role": "user", "content": " ".join(keys)}])

    monkeypatch.setitem(main.__globals__, "client_from_env", lambda model: complete)

    def generate(filename, *options):
        path = tmp_path / filename
        monkeypatch.setattr(sys, "argv", ["generate_scenarios", "--out", str(path), *options])
        with redirect_stdout(StringIO()) as output:
            output.reconfigure = lambda **_: None
            assert main() == 0
        return {
            record["session"]["id"]: record
            for record in map(json.loads, path.read_text("utf-8").splitlines())
        }

    v1_ids = {
        json.loads(line)["session"]["id"]
        for line in Path("data/scenarios/scenarios-v1.jsonl").read_text("utf-8").splitlines()
    }
    v2 = generate("v2.jsonl")
    assert len(v2) == len(v1_ids) == 65
    assert v1_ids.isdisjoint(v2)
    assert v2 == generate("repeat.jsonl")
    for split in SPLITS:
        subset = generate(f"{split}.jsonl", "--split", split)
        assert subset
        assert subset == {
            sid: record for sid, record in v2.items() if record["session"]["meta"]["split"] == split
        }
    for sid, record in v2.items():
        meta = record["session"]["meta"]
        assert meta["group_id"] in v1_ids
        assert sid == meta["group_id"].replace("sc-", "sc-v2-", 1)
        assert meta["split"] == group_split(meta["group_id"])
