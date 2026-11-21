"""Collect live PII-Bench responses and score raw/maximal typed spans reproducibly."""

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

TYPE_MAP = {
    "PERSON": "NAME",
    "PHONE": "PHONE_NUMBER",
    "EMAIL": "EMAIL",
    "ADDRESS": "ADDRESS",
    "PAYMENT_CARD": "BANK_CARD_NUMBER",
    "PASSPORT": "PASSPORT_NUMBER",
    "INN": "INN",
    "SNILS": "SNILS",
}
SUPPORTED = frozenset(TYPE_MAP.values())


def maximal_spans(spans):
    """Remove same-type contained duplicates; keep disjoint and cross-type entities."""
    spans = set(spans)
    return {
        s
        for s in spans
        if not any(
            s != other and s[0] == other[0] and other[1] <= s[1] and s[2] <= other[2]
            for other in spans
        )
    }


def matched_pairs(gold, predicted, overlap=False):
    """Maximum one-to-one matching, requiring equal labels for both scoring modes."""
    gold, predicted = sorted(set(gold)), sorted(set(predicted))
    adjacency = [
        [
            j
            for j, p in enumerate(predicted)
            if g[0] == p[0] and ((g[1] < p[2] and p[1] < g[2]) if overlap else g[1:] == p[1:])
        ]
        for g in gold
    ]
    assigned = {}

    def augment(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(gold)):
        augment(i, set())
    return {(gold[i], predicted[j]) for j, i in assigned.items()}


def counts_for(gold, predicted, overlap=False):
    gold, predicted = set(gold), set(predicted)
    pairs = matched_pairs(gold, predicted, overlap)
    found_gold = {g for g, _ in pairs}
    found_pred = {p for _, p in pairs}
    counts = defaultdict(Counter)
    for g, _ in pairs:
        counts[g[0]]["tp"] += 1
    for g in gold - found_gold:
        counts[g[0]]["fn"] += 1
    for p in predicted - found_pred:
        counts[p[0]]["fp"] += 1
    return counts


def metrics(counts):
    tp, fp, fn = (counts.get(key, 0) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / (precision + recall), 6)
        if precision + recall
        else 0.0,
    }


def summarize(records):
    totals = defaultdict(lambda: defaultdict(Counter))
    by_domain = defaultdict(lambda: defaultdict(Counter))
    coverage = Counter()
    unsupported = Counter()
    latency = []
    provenance = None
    error_count = 0
    mistakes = []
    for record in records:
        gold = {(e["type"], e["start"], e["end"]) for e in record["gold"]}
        unsupported.update(g[0] for g in gold if g[0] not in SUPPORTED)
        body = record.get("response")
        predicted = set()
        if body is not None:
            trace = body["trace"]
            provenance = {
                "profile": trace["profile"],
                "policy_version": trace["policy_version"],
                "model": {k: v for k, v in trace["model"].items() if k != "windows"},
                "regex_rule_ids": [r["id"] for r in trace["regex"]["rules"]],
            }
            predicted = {
                (TYPE_MAP.get(e["type"], e["type"]), e["start"], e["end"]) for e in body["entities"]
            }
            if record.get("client_ms") is not None:
                latency.append(record["client_ms"])
        else:
            error_count += 1
        maximal = maximal_spans(predicted)
        coarse_gold = {("NATIONAL_ID" if k in {"INN", "SNILS"} else k, s, e) for k, s, e in gold}
        coarse_pred = {("NATIONAL_ID" if k in {"INN", "SNILS"} else k, s, e) for k, s, e in maximal}
        modes = {
            "raw_exact_all_types": (gold, predicted, False),
            "maximal_exact_all_types": (gold, maximal, False),
            "maximal_overlap_all_types": (gold, maximal, True),
            "maximal_exact_supported_8": (
                {g for g in gold if g[0] in SUPPORTED},
                {p for p in maximal if p[0] in SUPPORTED},
                False,
            ),
            "maximal_overlap_coarse_national_id": (coarse_gold, coarse_pred, True),
        }
        for mode, (truth, output, overlap) in modes.items():
            totals.setdefault(mode, defaultdict(Counter))
            by_domain[record["domain"]].setdefault(mode, Counter())
            counts = counts_for(truth, output, overlap)
            for kind, count in counts.items():
                totals[mode][kind].update(count)
                by_domain[record["domain"]][mode].update(count)
        gold_chars = {i for _, start, end in gold for i in range(start, end)}
        pred_chars = {i for _, start, end in predicted for i in range(start, end)}
        coverage.update(
            tp=len(gold_chars & pred_chars),
            fp=len(pred_chars - gold_chars),
            fn=len(gold_chars - pred_chars),
        )
        pairs = matched_pairs(gold, maximal)
        missing = gold - {g for g, _ in pairs}
        unexpected = maximal - {p for _, p in pairs}
        if missing or unexpected or body is None:

            def describe(spans, text=record["text"]):
                return [
                    {"type": k, "start": s, "end": e, "text": text[s:e]}
                    for k, s, e in sorted(spans)
                ]

            mistakes.append(
                {
                    "id": record["id"],
                    "domain": record["domain"],
                    "text": record["text"],
                    "missing": describe(missing),
                    "unexpected": describe(unexpected),
                }
            )
    result = {}
    for mode, types in totals.items():
        total = Counter()
        for count in types.values():
            total.update(count)
        result[mode] = {
            "micro": metrics(total),
            "by_type": {k: metrics(v) for k, v in sorted(types.items())},
        }
    return {
        "rows": len(records),
        "gold_entities": sum(len(r["gold"]) for r in records),
        "http_errors": error_count,
        "provenance": provenance,
        "scores": result,
        "unsupported_gold": dict(unsupported),
        "masking_character_coverage": metrics(coverage),
        "rows_without_exact_errors": len(records) - len(mistakes),
        "latency_ms": {
            "median": round(statistics.median(latency), 3),
            "p95": round(sorted(latency)[min(len(latency) - 1, int(len(latency) * 0.95))], 3),
        }
        if latency
        else None,
        "by_domain": {
            d: {mode: metrics(c) for mode, c in modes.items()} for d, modes in by_domain.items()
        },
        "mistakes": mistakes,
    }


def collect(args):
    items = [json.loads(line) for line in args.items.read_text().splitlines() if line.strip()]
    with urllib.request.urlopen(args.url + "/health/ready", timeout=30) as response:
        assert response.status == 200
    start_time = perf_counter()
    errors = 0
    with nullcontext(sys.stdout) if args.output == "-" else Path(args.output).open("x") as out:
        for i, item in enumerate(items, 1):
            text = item["text"]
            for entity in item["entities"]:
                assert 0 <= entity["start"] < entity["end"] <= len(text)
                assert text[entity["start"] : entity["end"]] == entity["text"]
            record = {key: item[key] for key in ("id", "domain", "split", "text")}
            record["gold"] = item["entities"]
            request = urllib.request.Request(
                args.url + "/v1/detect",
                data=json.dumps({"text": text}, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"},
            )
            started = perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.load(response)
                    assert body["request_id"] == response.headers["X-Request-ID"]
                    assert response.headers["Cache-Control"] == "no-store"
                assert body["trace"]["profile"] == args.expected_profile
                assert body["trace"]["model"]["device"] == "cuda:0"
                assert body["trace"]["model"]["status"] == "ok"
                for entity in body["entities"]:
                    assert text[entity["start"] : entity["end"]] == entity["text"]
                record["response"] = body
                record["client_ms"] = round((perf_counter() - started) * 1000, 3)
            except urllib.error.URLError as exc:
                record["response"] = None
                record["error"] = str(exc)
                errors += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if i % 100 == 0 or i == len(items):
                print(
                    f"{i}/{len(items)} rows, {perf_counter() - start_time:.1f}s, HTTP errors={errors}",
                    file=sys.stderr,
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--items", type=Path, required=True)
    run.add_argument("--url", default="http://127.0.0.1:8000")
    run.add_argument("--expected-profile", required=True)
    run.add_argument("--output", required=True)
    score = commands.add_parser("score")
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        args.url = args.url.rstrip("/")
        collect(args)
    else:
        records = [
            json.loads(line) for line in args.responses.read_text().splitlines() if line.strip()
        ]
        report = {
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "methodology": {
                "input": "Exact dataset text, including raw L-DIALOG JSON; no turn extraction or skipped rows",
                "types": TYPE_MAP,
                "unsupported": "All gold types count towards full metrics, including types absent from the detector",
                "maximal": "Remove same-type fully contained predictions; retain disjoint and cross-type spans",
                "matching": "Maximum one-to-one, typed exact or typed overlap; wrong labels incur FP and FN",
                "raw": "Exact entities from the supplied response records, without span postprocessing",
            },
            "splits": {
                s: summarize([r for r in records if r["split"] == s])
                for s in sorted({r["split"] for r in records})
            },
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        for split, summary in report["splits"].items():
            print(split, {mode: values["micro"] for mode, values in summary["scores"].items()})


if __name__ == "__main__":
    main()
