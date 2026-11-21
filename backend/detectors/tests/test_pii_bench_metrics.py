"""Protect benchmark scoring from label, duplicate and many-to-one matching errors."""

from collections import Counter

from backend.detectors.scripts.pii_bench import (
    counts_for,
    matched_pairs,
    maximal_spans,
    metrics,
    summarize,
)


def total(counts):
    result = Counter()
    for count in counts.values():
        result.update(count)
    return result


def test_wrong_type_is_both_a_false_positive_and_a_miss():
    counts = total(counts_for({("INN", 0, 12)}, {("PHONE_NUMBER", 0, 12)}, overlap=True))
    assert counts == {"fp": 1, "fn": 1}


def test_partial_span_only_counts_in_overlap_mode():
    gold, predicted = {("NAME", 0, 11)}, {("NAME", 0, 4)}
    assert total(counts_for(gold, predicted)) == {"fp": 1, "fn": 1}
    assert total(counts_for(gold, predicted, overlap=True)) == {"tp": 1}


def test_one_prediction_cannot_match_two_gold_entities():
    gold = {("NAME", 0, 4), ("NAME", 6, 10)}
    predicted = {("NAME", 0, 10)}
    assert total(counts_for(gold, predicted, overlap=True)) == {"tp": 1, "fn": 1}


def test_matching_can_reassign_greedy_pairs_to_reach_maximum():
    gold = {("NAME", 0, 4), ("NAME", 3, 8)}
    predicted = {("NAME", 0, 8), ("NAME", 1, 2)}
    assert len(matched_pairs(gold, predicted, overlap=True)) == 2


def test_containment_removal_does_not_cross_types_or_merge_disjoint_spans():
    spans = {("NAME", 0, 11), ("NAME", 0, 4), ("NAME", 5, 11), ("ADDRESS", 0, 11), ("NAME", 20, 24)}
    assert maximal_spans(spans) == {("NAME", 0, 11), ("ADDRESS", 0, 11), ("NAME", 20, 24)}


def test_unsupported_gold_is_not_silently_dropped():
    assert metrics(total(counts_for({("TOKEN", 0, 10)}, set()))) == {
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }


def test_http_failure_does_not_pass_a_negative_row():
    report = summarize(
        [{"id": "negative", "domain": "L-CHAT", "text": "hello", "gold": [], "response": None}]
    )
    assert report["http_errors"] == 1 and report["rows_without_exact_errors"] == 0
    assert report["scores"]["maximal_exact_all_types"]["micro"]["tp"] == 0
