"""pii-bench → сессии, ответ детектора → DetectionLog, метрики и автоэталон."""

import json

from backend.agent.cases import build_cases
from backend.agent.detector_adapter import to_detection_log
from backend.schemas import RuleConfig
from evaluation.metrics import expected, score_detector
from evaluation.pii_bench import to_item

DIALOG = json.dumps(
    [
        {"role": "system", "content": "Ты помощник поддержки банка."},
        {"role": "user", "content": "списали деньги с карты 2200 3845 7712 9085"},
        {"role": "assistant", "content": "Проверяю операцию по карте."},
    ],
    ensure_ascii=False,
    indent=0,
)
CARD = "2200 3845 7712 9085"

RULES = [RuleConfig(id="card_pan_v1", type="PAYMENT_CARD", pattern="x", validator="card_pan")]


def response(entities, candidates):
    """Минимальный ответ /v1/detect в форме сервиса Михаила."""
    return {
        "entities": entities,
        "trace": {
            "profile": "pii-ru-en-v1",
            "policy_version": "1.0.0",
            "model": {
                "id": "hivetrace/gliner-guard-omni",
                "emit_threshold": 0.2,
                "accept_threshold": 0.5,
                "labels": ["person", "card_number"],
            },
            "candidates": candidates,
        },
    }


def test_dialog_is_split_into_turns_with_gold_offsets():
    start = DIALOG.index(CARD)
    row = {
        "id": "d1",
        "domain": "L-DIALOG",
        "text": DIALOG,
        "entities": [
            {"type": "BANK_CARD_NUMBER", "start": start, "end": start + len(CARD), "text": CARD}
        ],
    }
    item = to_item(row)
    assert [(m.turn, m.direction) for m in item.session.messages] == [(0, "in"), (1, "out")]
    (g,) = item.gold
    assert g.turn == 0
    assert item.session.messages[0].text[g.start : g.end] == CARD
    assert g.detector_type == "PAYMENT_CARD"


def test_plain_example_is_single_turn():
    row = {
        "id": "x",
        "domain": "S-BANK",
        "text": "карта " + CARD,
        "entities": [{"type": "BANK_CARD_NUMBER", "start": 6, "end": 25, "text": CARD}],
    }
    item = to_item(row)
    assert item.session.id == "pb-x"
    assert len(item.session.messages) == 1
    assert item.gold[0].start == 6


def _bench_item():
    text = "Анна, карта " + CARD + " и заказ 123456"
    row = {
        "id": "a",
        "domain": "S-BANK",
        "text": text,
        "entities": [
            {"type": "NAME", "start": 0, "end": 4, "text": "Анна"},
            {"type": "BANK_CARD_NUMBER", "start": 12, "end": 31, "text": CARD},
        ],
    }
    return to_item(row), text


def _log(item, text):
    order = text.index("123456")
    resp = response(
        entities=[
            {
                "id": "e1",
                "type": "PAYMENT_CARD",
                "start": 12,
                "end": 31,
                "text": CARD,
                "candidate_ids": ["c1"],
            },
            {
                "id": "e2",
                "type": "PERSON",
                "start": order,
                "end": order + 6,
                "text": "123456",
                "candidate_ids": ["c2"],
            },
        ],
        candidates=[
            {
                "id": "c1",
                "type": "PAYMENT_CARD",
                "start": 12,
                "end": 31,
                "decision": "accepted",
                "entity_id": "e1",
                "source": "regex",
                "rule_id": "card_pan_v1",
                "score": None,
                "reason": "rule_passed",
            },
            {
                "id": "c2",
                "type": "PERSON",
                "start": order,
                "end": order + 6,
                "decision": "accepted",
                "entity_id": "e2",
                "source": "model",
                "raw_label": "person",
                "score": 0.6,
                "reason": "model_threshold_passed",
            },
            {
                "id": "c3",
                "type": "PERSON",
                "start": 0,
                "end": 4,
                "decision": "rejected",
                "entity_id": None,
                "source": "model",
                "raw_label": "first_name",
                "score": 0.3,
                "reason": "below_model_threshold",
            },
        ],
    )
    return to_detection_log(item.session, {0: resp}, RULES)


def test_adapter_prefixes_ids_and_fills_candidate_text():
    item, text = _bench_item()
    log = _log(item, text)
    assert [e.id for e in log.entities] == ["pb-a-t0-e1", "pb-a-t0-e2"]
    assert log.entities[0].candidate_ids == ["pb-a-t0-c1"]
    assert log.candidates[2].text == "Анна"
    assert log.candidates[2].entity_id is None
    assert log.rules == RULES


def test_detector_metrics_and_auto_labels():
    item, text = _bench_item()
    log = _log(item, text)

    report = score_detector([(log, item.gold)])
    card, person = report.by_type["PAYMENT_CARD"], report.by_type["PERSON"]
    assert (card.tp, card.fp, card.fn, card.exact) == (1, 0, 0, 1)
    assert (person.tp, person.fp, person.fn) == (0, 1, 1)

    verdicts = {c.id: expected(c, item.gold) for c in build_cases(log)}
    assert verdicts == {
        "pb-a-t0-e1": ("true_pii", "pii"),
        "pb-a-t0-e2": ("false_positive", "none"),
        "pb-a-t0-c3": ("missed_pii", "pii"),
    }


def test_company_requisites_are_not_masked():
    text = "ИНН компании 7707083893, КПП 773601001, токен ab12cd34ef56gh78ij90kl"
    row = {
        "id": "c",
        "domain": "S-BANK",
        "text": text,
        "entities": [
            {"type": "INN", "start": 13, "end": 23, "text": "7707083893"},
            {"type": "KPP", "start": 29, "end": 38, "text": "773601001"},
            {"type": "TOKEN", "start": 46, "end": 68, "text": "ab12cd34ef56gh78ij90kl"},
        ],
    }
    item = to_item(row)
    assert [g.category for g in item.gold] == ["none", "none", "sensitive"]
