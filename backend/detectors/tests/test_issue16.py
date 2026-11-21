"""Issue #16: document labels, public names and the five review examples."""

import pytest

from backend.detectors.app.gliner_detector import decode_window
from backend.detectors.app.observations import Window
from backend.detectors.app.regex_detector import RegexDetector

DOCUMENTS = {
    "driver_license": "DRIVER_LICENSE",
    "birth_certificate": "BIRTH_CERTIFICATE",
    "military_id": "MILITARY_ID",
    "medical_policy": "OMS",
}


def test_document_labels_are_enabled_with_original_thresholds(profile):
    assert set(DOCUMENTS) <= set(profile.labels)
    assert {label: profile.public_type(label) for label in DOCUMENTS} == DOCUMENTS
    assert profile.emit_threshold == 0.20 and profile.accept_threshold == 0.50
    assert profile.public_type("oms_policy") == "OMS"


@pytest.mark.parametrize(("label", "kind"), DOCUMENTS.items())
def test_model_adapter_accepts_each_new_label(profile, label, kind):
    text = "🙂 1234567"
    output = {"entities": {label: [{"text": "1234567", "start": 2, "end": 9, "confidence": 0.8}]}}
    found = decode_window(output, text, Window("w1", 0, len(text)), profile)
    assert len(found) == 1 and found[0].type == kind and found[0].raw_label == label


@pytest.mark.parametrize(
    ("text", "kind", "value"),
    [
        (
            "Водительское удостоверение серии 77 14 номер 987654",
            "DRIVER_LICENSE",
            "77 14 номер 987654",
        ),
        (
            "свидетельство о рождении, серия III - АГ, номер 112233",
            "BIRTH_CERTIFICATE",
            "III - АГ, номер 112233",
        ),
        ("военный билет, серия МЗ, номер 0045678", "MILITARY_ID", "МЗ, номер 0045678"),
        ("У меня АА 1234567", "MILITARY_ID", "АА 1234567"),
        ("Запиши полис ОМС: 7702 - 1234 - 5678 - 9012", "OMS", "7702 - 1234 - 5678 - 9012"),
        ("паспорт серия 7518, номер 492137", "PASSPORT", "серия 7518, номер 492137"),
        ("II - КЗ № 445566", "BIRTH_CERTIFICATE", "II - КЗ № 445566"),
    ],
)
def test_review_regressions(profile, text, kind, value):
    observations = RegexDetector(profile).detect(text).observations
    assert any(
        o.type == kind and text[o.start : o.end] == value and o.reason == "rule_passed"
        for o in observations
    )


@pytest.mark.parametrize(
    "value",
    ["7702123456789012", "7702 1234 5678 9012", "7702-1234-5678-9012", "7702.1234.5678.9012"],
)
def test_oms_separators(profile, value):
    text = "Полис ОМС: " + value
    observations = RegexDetector(profile).detect(text).observations
    assert any(
        o.type == "OMS" and text[o.start : o.end] == value and o.reason == "rule_passed"
        for o in observations
    )


@pytest.mark.parametrize(
    "value", ["7702..1234.5678.9012", "7702.1234.5678.9012.3", "0000.0000.0000.0000"]
)
def test_oms_invalid_values_are_not_accepted(profile, value):
    observations = RegexDetector(profile).detect("Полис ОМС: " + value).observations
    assert not any(o.type == "OMS" and o.reason == "rule_passed" for o in observations)
