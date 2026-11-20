"""Адресат рекомендации против лога детектора."""

import pytest

from backend.agent.cases import build_cases
from backend.agent.recommend import recommendation_problems
from backend.schemas import (
    Candidate,
    DetectionLog,
    Entity,
    Message,
    ModelConfig,
    RuleConfig,
    Session,
)

SESSION = Session(id="s", messages=[Message(turn=0, direction="in", text="Заказ 4276123456789014")])
RULE = RuleConfig(id="card_pan_v1", type="PAYMENT_CARD", pattern=r"\d{16}", validator="luhn")
MODEL = ModelConfig(id="ner", emit_threshold=0.3, accept_threshold=0.5, labels=["PAYMENT_CARD"])


def log_with(source: str, decision: str) -> DetectionLog:
    accepted = decision == "accepted"
    candidate = Candidate(
        id="c1",
        turn=0,
        start=6,
        end=22,
        text="4276123456789014",
        type="PAYMENT_CARD",
        source=source,
        decision=decision,
        reason="model_threshold_passed" if accepted else "below_model_threshold",
        entity_id="e1" if accepted else None,
        score=0.8 if accepted else 0.4,
        rule_id="card_pan_v1" if source == "regex" else None,
    )
    entities = [
        Entity(
            id="e1",
            turn=0,
            start=6,
            end=22,
            text="4276123456789014",
            type="PAYMENT_CARD",
            candidate_ids=["c1"],
        )
    ]
    return DetectionLog(
        session_id="s",
        profile="pii-ru-en-v1",
        policy_version="1",
        model=MODEL,
        rules=[RULE],
        entities=entities if accepted else [],
        candidates=[candidate],
    )


def check(rec: dict, verdict: str, source: str, decision: str) -> list[str]:
    log = log_with(source, decision)
    (case,) = build_cases(log, SESSION)
    return recommendation_problems(rec, verdict, case, log)


def rec(kind: str, target: str) -> dict:
    return {"kind": kind, "target": target, "change": "…", "rationale": "…"}


@pytest.mark.parametrize(
    ("recommendation", "source", "ok"),
    [
        (rec("threshold", "ner"), "model", True),  # приняла модель — порог модели
        (rec("pattern", "ner"), "model", True),  # исключение на выходе модели
        (rec("allowlist", "card_pan_v1"), "model", False),  # приняла модель, а правят правило
        (rec("allowlist", "card_pan_v1"), "regex", True),
        (rec("threshold", "card_pan_v1"), "regex", False),  # порога у правила нет
        (rec("ticket", "команда детектора"), "regex", True),
    ],
)
def test_false_positive_fix_goes_to_whoever_accepted(recommendation, source, ok):
    assert (check(recommendation, "false_positive", source, "accepted") == []) is ok


@pytest.mark.parametrize(
    ("recommendation", "ok"),
    [
        (rec("threshold", "ner"), True),  # модель отклонила ниже порога
        (rec("pattern", "pii-ru-en-v1"), True),  # новое правило в профиле
        (rec("allowlist", "card_pan_v1"), False),  # правило фрагмент не отклоняло
        (rec("retrain", "card_pan_v1"), False),
    ],
)
def test_miss_fix_goes_to_whoever_rejected_or_is_a_new_pattern(recommendation, ok):
    assert (check(recommendation, "missed_pii", "model", "rejected") == []) is ok


def test_no_recommendation_or_detector_right_is_fine():
    assert check(None, "missed_pii", "model", "rejected") == []
    assert check(rec("threshold", "card_pan_v1"), "true_pii", "regex", "accepted") == []
