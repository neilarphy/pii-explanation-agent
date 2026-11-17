from pathlib import Path

import pytest
from pydantic import TypeAdapter

from backend.schemas import AgentAnswer, DetectionLog, Session, Summary, Verdict

EXAMPLES = Path(__file__).parent.parent / "docs" / "examples"
MODELS = {
    "session.json": TypeAdapter(Session),
    "detection_log.json": TypeAdapter(DetectionLog),
    "agent_answers.json": TypeAdapter(list[AgentAnswer]),
    "summary.json": TypeAdapter(Summary),
}


def load(name: str):
    return MODELS[name].validate_json((EXAMPLES / name).read_text(encoding="utf-8"))


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def test_every_example_has_a_model():
    assert {p.name for p in EXAMPLES.glob("*.json")} == set(MODELS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_example_validates(name):
    load(name)


def test_spans_match_text():
    session, log = load("session.json"), load("detection_log.json")
    assert log.session_id == session.id
    for item in [*log.entities, *log.candidates]:
        assert session.messages[item.turn].text[item.start : item.end] == item.text, item.id


def test_citation_spans_match_quote():
    session = load("session.json")
    for answer in load("agent_answers.json"):
        for c in answer.citations:
            assert session.messages[c.turn].text[c.start : c.end] == c.quote, c


def test_entities_and_candidates_are_linked():
    log = load("detection_log.json")
    candidates = {c.id: c for c in log.candidates}
    entity_ids = {e.id for e in log.entities}
    for c in log.candidates:
        if c.decision == "accepted":
            assert c.entity_id in entity_ids, c.id
        else:
            assert c.entity_id is None, c.id
    for e in log.entities:
        assert e.candidate_ids, e.id
        for cid in e.candidate_ids:
            c = candidates[cid]
            assert (c.entity_id, c.turn, c.start, c.end, c.type) == (
                e.id,
                e.turn,
                e.start,
                e.end,
                e.type,
            )


def test_candidates_follow_profile():
    log = load("detection_log.json")
    rules = {r.id for r in log.rules}
    for c in log.candidates:
        if c.source == "regex":
            assert c.rule_id in rules and c.score is None, c.id
        else:
            assert c.raw_label in log.model.labels, c.id
            assert c.score >= log.model.emit_threshold, c.id
            accepted = c.score >= log.model.accept_threshold
            assert c.decision == ("accepted" if accepted else "rejected"), c.id


def test_card_checksum_matches_luhn():
    cards = [
        c for c in load("detection_log.json").candidates
        if c.source == "regex" and c.type == "PAYMENT_CARD"
    ]  # fmt: skip
    assert cards
    for c in cards:
        if c.reason in ("rule_passed", "checksum_failed"):
            assert (c.reason == "rule_passed") == luhn_ok(c.text), c.id


def test_answers_reference_entities():
    ids = {e.id for e in load("detection_log.json").entities}
    for answer in load("agent_answers.json"):
        if answer.verdict == Verdict.MISSED_PII:
            assert answer.entity_id is None
        else:
            assert answer.entity_id in ids


def test_example_covers_every_verdict():
    assert {a.verdict for a in load("agent_answers.json")} == set(Verdict)
