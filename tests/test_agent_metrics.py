"""Метрики чекпоинта 1 на маленьком наборе, где каждую цифру видно глазами."""

import pytest

from evaluation.agent_metrics import (
    DROPPED,
    as_filter,
    citations,
    macro_f1,
    misses,
    per_class,
    stability,
    uncertain_by_expected,
    usage,
)


def row(case_id, kind, expected, got, text="855707", quote=None, evidence=(), seconds=1.0):
    answer = None
    if got not in ("no_issue", "ошибка"):
        cited = [{"turn": 0, "quote": quote}] if quote else []
        answer = {"citations": cited, "evidence": list(evidence)}
    return {
        "case_id": case_id,
        "kind": kind,
        "expected": expected,
        "got": got,
        "text": text,
        "answer": answer,
        "telemetry": {
            "llm_calls": 1,
            "retries": 0,
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "seconds": seconds,
            "cost": 0.001,
        },
    }


ROWS = [
    # принятые детектором: 2 ложных (одно снято), 2 настоящих (одно потеряно), 1 на человека
    row("a", "entity", "false_positive", "false_positive", quote="855707"),
    row("b", "entity", "false_positive", "true_pii", quote="855707"),
    row("c", "entity", "true_pii", "true_pii", quote="855707"),
    row("d", "entity", "true_pii", "false_positive", quote="дальше 855707"),
    row("e", "entity", "true_pii", "uncertain", quote="855", evidence=(f"{DROPPED}: x",)),
    # отклонённые: 2 пропуска (один найден), 2 чистых (одна ложная тревога)
    row("f", "rejected", "missed_pii", "missed_pii", quote="855707"),
    row("g", "scan", "missed_pii", "ошибка"),
    row("h", "rejected", "no_issue", "no_issue"),
    row("i", "rejected", "no_issue", "missed_pii", quote="855707"),
]


def test_per_class_precision_recall_f1():
    scores = per_class(ROWS)
    fp = scores["false_positive"]
    assert (fp.support, fp.predicted, fp.correct) == (2, 2, 1)
    assert fp.precision == fp.recall == fp.f1 == 0.5
    tp = scores["true_pii"]  # c верно, b — ложное назвали настоящим; d и e — не попали
    assert (tp.precision, tp.recall) == (0.5, pytest.approx(1 / 3))
    assert 0 < macro_f1(scores) < 1


def test_class_never_predicted_has_zero_f1():
    scores = per_class([row("x", "entity", "true_pii", "uncertain")])
    assert scores["true_pii"].f1 == 0.0 and scores["true_pii"].precision is None
    assert scores["no_issue"].f1 is None  # класса нет ни в эталоне, ни в ответах


def test_second_filter_keeps_uncertain_for_human():
    f = as_filter(ROWS)
    assert (f.false_found, f.false_removed, f.true_found, f.true_lost, f.to_human) == (
        2,
        1,
        3,
        1,
        1,
    )
    assert f.removed_share == 0.5 and f.lost_share == pytest.approx(1 / 3)


def test_misses_and_false_alarms():
    m = misses(ROWS)
    assert (m.missed, m.found, m.to_human, m.clean, m.false_alarms) == (2, 1, 1, 2, 1)


def test_uncertain_and_citations():
    assert uncertain_by_expected(ROWS) == {"true_pii": 1}
    c = citations(ROWS)
    # ответы есть у 7 строк (кроме no_issue и сбоя); у e модель прислала недословную цитату
    assert (c.answers, c.with_citations, c.all_verbatim) == (7, 7, 6)
    # «дальше 855707» содержит находку, «855» входит в неё — обе считаются самой находкой
    assert c.first_is_finding == 7


def test_usage():
    u = usage(ROWS)
    assert u.calls == 1 and u.prompt_tokens == 100 and u.cost_usd == pytest.approx(0.009)


def test_stability_counts_flips_between_runs():
    second = [dict(r, got="true_pii") if r["case_id"] == "b" else r for r in ROWS]
    third = [dict(r, got="false_positive") if r["case_id"] == "b" else r for r in ROWS]
    s = stability([ROWS, second, third])
    assert (s.runs, s.cases, s.flipped) == (3, 9, 1)
    assert s.accuracy == (4 / 9, 4 / 9, 5 / 9)
    assert s.spread == pytest.approx(1 / 18)


def test_policy_mismatch_is_not_an_agent_error():
    from evaluation.agent_metrics import policy_mismatch

    org = row("x", "entity", "true_pii", "false_positive", text="support@megafon.ru")
    toll = row("y", "entity", "true_pii", "false_positive", text="8-800-555-35-35")
    card = row("z", "rejected", "missed_pii", "no_issue", text="4111 1111 1111 1111")
    person = row("w", "entity", "true_pii", "false_positive", text="ivanov@company.ru")
    agreed = row("v", "entity", "true_pii", "true_pii", text="support@megafon.ru")
    assert [policy_mismatch(r) for r in (org, toll, card, person, agreed)] == [
        True,
        True,
        True,
        False,
        False,
    ]


def test_spaced_org_mailbox_is_a_policy_mismatch():
    from evaluation.agent_metrics import policy_mismatch

    spaced = row("x", "entity", "true_pii", "false_positive", text="info @ fortress . com . ru")
    assert policy_mismatch(spaced)


def test_ten_digit_inn_is_an_organisation_even_with_explicit_category():
    from evaluation.pii_bench import GoldSpan

    assert GoldSpan(0, 0, 10, "INN", "7707083893", "pii").category == "none"
    assert GoldSpan(0, 0, 12, "INN", "500100732259", "pii").category == "pii"
