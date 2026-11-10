import itertools
import re
from types import SimpleNamespace

import pytest
from app.config import load_profile, parse_labels
from app.errors import AdapterContractError, ConfigurationError, TraceLimitError
from app.gliner_detector import ProcessorBudget, decode_window
from app.observations import Window
from app.windowing import plan_windows


@pytest.mark.parametrize(
    "value",
    [
        "",
        "person,person",
        "person,",
        ",person",
        "first name",
        "Person",
        " person, ,passport",
        "1person",
    ],
)
def test_invalid_labels(value):
    with pytest.raises(ConfigurationError, match="NER_LABELS"):
        parse_labels(value, ("person",))


def test_labels_default_order(profile, monkeypatch):
    assert profile.labels == (
        "person",
        "first_name",
        "last_name",
        "address",
        "email",
        "phone",
        "passport",
        "national_id",
        "date_of_birth",
        "card_number",
        "bank_account",
    )
    monkeypatch.setenv("NER_LABELS", "person, passport")
    assert load_profile().labels == ("person", "passport")
    monkeypatch.setenv("NER_LABELS", "employee_id")
    assert load_profile().labels == ("employee_id",)


@pytest.mark.parametrize(
    "change",
    [
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": True},
        {"confidence": "0.5"},
        {"start": -1},
        {"end": 99},
        {"end": 2},
        {"start": 2.0},
        {"text": "wrong"},
    ],
)
def test_adapter_rejects_corrupt_offsets_and_scores(profile, change):
    value = dict(text="Анна", start=2, end=6, confidence=0.9) | change
    with pytest.raises(AdapterContractError):
        decode_window({"entities": {"first_name": [value]}}, "🙂 Анна", Window("w1", 0, 6), profile)


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"entities": []},
        {"entities": {"unknown": []}},
        {"entities": {"person": ["name"]}},
        {"entities": {"person": {}}},
    ],
)
def test_adapter_shape(profile, result):
    with pytest.raises(AdapterContractError):
        decode_window(result, "a", Window("w1", 0, 1), profile)


def test_global_offsets(profile):
    observations = decode_window(
        {"entities": {"first_name": [{"text": "Анна", "start": 2, "end": 6, "confidence": 0.5}]}},
        "prefix🙂 Анна",
        Window("w2", 6, 12),
        profile,
    )
    assert (observations[0].start, observations[0].end) == (8, 12)
    assert observations[0].window_id == "w2"


@pytest.mark.parametrize(
    "text",
    [
        "word " * 1000 + "last",
        "я🙂\r\ne\u0301\ue000 " * 500,
        "x" * 10000,
        "🙂" * 10000,
        "   text    ",
        "x" + " " * 1000,
    ],
)
def test_window_coverage(text):
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def fits(part):
        return len(part) <= 200

    windows = plan_windows(text, spans, fits)
    assert windows[0].start == 0 and windows[-1].end == len(text)
    assert all(fits(text[w.start : w.end]) for w in windows)
    assert all(a.start < b.start <= a.end and b.end > a.end for a, b in itertools.pairwise(windows))


def test_window_limit_inclusive():
    # A one-character budget must advance by one; exactly 512 windows are allowed.
    def fits(part):
        return len(part) <= 1

    assert len(plan_windows("x" * 512, [(0, 512)], fits)) == 512
    with pytest.raises(TraceLimitError):
        plan_windows("x" * 513, [(0, 513)], fits)


def test_budget_uses_processor_not_char_estimate():
    class Processor:
        tokenizer = SimpleNamespace(model_max_length=8)

        def transform_record(self, text, schema, max_len):
            assert max_len is None
            return SimpleNamespace(
                input_ids=[0] * (len(text) + len(schema)), text_tokens=list(text)
            )

    processor = Processor()
    budget = ProcessorBudget(
        SimpleNamespace(
            processor=processor,
            encoder=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=10)),
            config=SimpleNamespace(max_len=4),
        ),
        ["person", "email"],
    )
    assert budget.fits("abcd") and not budget.fits("abcde")
    assert processor._normalize_text("🙂\r\n") == "🙂\r\n"
