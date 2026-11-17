import hashlib
import json
import random
import re
import subprocess
import sys
from collections import defaultdict

import pytest

from data.loader import LABELS, read_jsonl
from data.schema import Example
from data.synthetic import generate, write_dataset


@pytest.mark.parametrize("mode", ["educational", "realistic"])
def test_contract_offsets_roles_and_clean_dialogues(mode):
    examples = generate(mode=mode)
    assert len(examples) == 80
    assert len({e.session_id for e in examples}) == 80
    assert {s.entity_type for e in examples for s in e.gold_entities} <= LABELS
    assert len({s.entity_type for e in examples for s in e.gold_entities}) == 10
    assert {m.role for e in examples for m in e.messages} == {"system", "user", "assistant", "tool"}
    for e in examples:
        assert Example.model_validate_json(e.model_dump_json()) == e
        assert len(e.messages) >= 4
        assert e.detections is None and e.detector_run_id is None
        assert e.trace_origin == "not_run" and not e.gold_verdicts
        assert bool(e.gold_entities) != e.metadata["clean"]
        messages = {m.id: m.text for m in e.messages}
        for span in e.gold_entities:
            assert messages[span.message_id][span.start : span.end] == span.text
        if e.gold_entities:
            # Emoji precedes the first entity: distinguishes code points from UTF-16/bytes.
            span = e.gold_entities[0]
            prefix = messages[span.message_id][: span.start]
            assert len(prefix.encode("utf-16-le")) // 2 > span.start
            emails = [s for s in e.gold_entities if s.entity_type == "EMAIL"]
            assert len(emails) == 2 and emails[0].text == emails[1].text
            assert emails[0].message_id != emails[1].message_id


@pytest.mark.parametrize("mode", ["educational", "realistic"])
def test_reproducibility_and_group_isolation(mode):
    state = random.getstate()
    first = generate(seed=17, groups=80, mode=mode)
    assert random.getstate() == state
    assert first == generate(seed=17, groups=80, mode=mode)
    assert first != generate(seed=18, groups=80, mode=mode)
    assert first[:16] == generate(seed=17, groups=8, mode=mode)
    assert {e.source.split for e in first} == {"train", "validation", "calibration", "test"}
    by_parent = defaultdict(list)
    by_dialogue = defaultdict(set)
    for e in first:
        by_parent[e.metadata["parent_id"]].append(e)
        by_dialogue[tuple((m.role, m.text) for m in e.messages)].add(e.source.split)
    for variants in by_parent.values():
        assert len(variants) == 2
        assert len({e.source.group_id for e in variants}) == 1
        assert len({e.source.split for e in variants}) == 1
    assert all(len(splits) == 1 for splits in by_dialogue.values())
    more_variants = generate(seed=17, groups=80, variants=3, mode=mode)
    assert {e.source.group_id: e.source.split for e in first} == {
        e.source.group_id: e.source.split for e in more_variants
    }
    assert all(e.source.dataset == "synthetic-dialogues" for e in first)


@pytest.mark.parametrize("mode", ["educational", "realistic"])
def test_files_are_byte_reproducible_and_manifest_matches(tmp_path, mode):
    left, right = tmp_path / "left", tmp_path / "right"
    manifest = write_dataset(left, mode=mode)
    assert manifest == write_dataset(right, mode=mode)
    for path in left.iterdir():
        assert path.read_bytes() == (right / path.name).read_bytes()
    assert json.loads((left / "manifest.json").read_text(encoding="utf8")) == manifest
    for name, info in manifest["files"].items():
        assert hashlib.sha256((left / name).read_bytes()).hexdigest() == info["sha256"]
        records = list(read_jsonl(left / name))
        assert len(records) == info["examples"]
        assert all(e.source.split == name.removesuffix(".jsonl") for e in records)
    before = (left / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_dataset(left)
    assert (left / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "kwargs", [{"groups": 3}, {"variants": 0}, {"seed": True}, {"mode": "bad"}]
)
def test_invalid_arguments_do_not_create_output(tmp_path, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        write_dataset(output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["educational", "realistic"])
def test_cli(tmp_path, mode):
    output = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "data.synthetic",
            "--mode",
            mode,
            "--groups",
            "8",
            "--seed",
            "7",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sum(info["examples"] for info in json.loads(result.stdout)["files"].values()) == 16


def test_realistic_formats_checksums_and_cross_mode_groups():
    educational = generate(groups=100)
    realistic = generate(groups=100, mode="realistic")
    assert generate() == generate(mode="educational")
    assert {e.session_id for e in educational}.isdisjoint(e.session_id for e in realistic)
    cards, layouts = set(), set()
    for old, new in zip(educational, realistic, strict=True):
        assert old.source.group_id == new.source.group_id
        assert old.source.split == new.source.split
        assert old.metadata["parent_id"] == new.metadata["parent_id"]
        assert new.metadata["mode"] == "realistic"
        for entity in new.gold_entities:
            if entity.entity_type == "PHONE":
                assert re.fullmatch(r"\+44 7700 900[0-9]{3}", entity.text)
            elif entity.entity_type == "PASSPORT":
                assert re.fullmatch(r"[0-9]{2} [0-9]{2} [0-9]{6}", entity.text)
            elif entity.entity_type == "INN":
                assert re.fullmatch(r"[0-9]{12}", entity.text)
                digits = list(map(int, entity.text))
                assert (
                    sum(
                        a * b
                        for a, b in zip(digits[:10], [7, 2, 4, 10, 3, 5, 9, 4, 6, 8], strict=True)
                    )
                    % 11
                    % 10
                    == digits[10]
                )
                assert (
                    sum(
                        a * b
                        for a, b in zip(
                            digits[:11], [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8], strict=True
                        )
                    )
                    % 11
                    % 10
                    == digits[11]
                )
            elif entity.entity_type == "CREDIT_CARD":
                number = entity.text.replace(" ", "").replace("-", "")
                assert number in {"4242424242424242", "5555555555554444"}
                # Independent Luhn validation, including parity for 16-digit PANs.
                total = 0
                for i, digit in enumerate(reversed(number)):
                    value = int(digit) * (2 if i % 2 else 1)
                    total += value - 9 if value > 9 else value
                assert total % 10 == 0
                cards.add(number)
                layouts.add(
                    "space" if " " in entity.text else "dash" if "-" in entity.text else "plain"
                )
    assert len(cards) == 2
    assert layouts == {"plain", "space", "dash"}
