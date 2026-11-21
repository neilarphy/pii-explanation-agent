"""Run fixed synthetic cases against a live HTTP detector and retain complete traces."""

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


def expected_spans(text, entries):
    spans = set()
    for entry in entries:
        value = entry["text"]
        if text.count(value) != 1:
            raise ValueError("Each annotated value must occur exactly once in its case")
        start = text.index(value)
        spans.add((entry["type"], start, start + len(value)))
    return spans


def describe(spans, text):
    return [
        {"type": kind, "start": start, "end": end, "text": text[start:end]}
        for kind, start, end in sorted(spans)
    ]


def run_suite(base_url, suite, expected_profile, timeout):
    with urllib.request.urlopen(base_url + "/health/ready", timeout=timeout) as response:
        assert response.status == 200
    results = []
    counts = {"tp": 0, "fp": 0, "fn": 0, "allowed_name_parts": 0}
    latencies = []
    model_candidates = 0
    provenance = None
    for case in suite["cases"]:
        text = case["text"]
        expected = expected_spans(text, case["expected"])
        allowed = expected_spans(text, case.get("allowed", []))
        request = urllib.request.Request(
            base_url + "/v1/detect",
            data=json.dumps({"text": text}, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
                headers = dict(response.headers)
                assert body["request_id"] == response.headers["X-Request-ID"]
                assert response.headers["Cache-Control"] == "no-store"
            elapsed = (perf_counter() - started) * 1000
            trace = body["trace"]
            if trace["profile"] != expected_profile:
                raise ValueError("Live profile differs from the requested measurement profile")
            assert trace["model"]["device"] == "cuda:0" and trace["model"]["status"] == "ok"
            assert trace["regex"]["device"] == "cpu" and trace["regex"]["status"] == "ok"
            provenance = {
                "profile": trace["profile"],
                "policy_version": trace["policy_version"],
                "model": {key: value for key, value in trace["model"].items() if key != "windows"},
                "regex_rule_ids": [rule["id"] for rule in trace["regex"]["rules"]],
            }
            model_candidates += sum(c["source"] == "model" for c in trace["candidates"])
            actual = set()
            for entity in body["entities"]:
                assert 0 <= entity["start"] < entity["end"] <= len(text)
                assert text[entity["start"] : entity["end"]] == entity["text"]
                actual.add((entity["type"], entity["start"], entity["end"]))
            tp, fp, fn = actual & expected, actual - expected - allowed, expected - actual
            ignored = (actual & allowed) - expected
            for key, spans in (("tp", tp), ("fp", fp), ("fn", fn), ("allowed_name_parts", ignored)):
                counts[key] += len(spans)
            passed = not fp and not fn and body["has_pii"] == bool(expected)
            results.append(
                {
                    "id": case["id"],
                    "text": text,
                    "passed": passed,
                    "expected": describe(expected, text),
                    "allowed": describe(allowed, text),
                    "missing": describe(fn, text),
                    "unexpected": describe(fp, text),
                    "client_ms": round(elapsed, 3),
                    "response_headers": headers,
                    "response": body,
                }
            )
            latencies.append(elapsed)
            print(
                f"{case['id']}: {'PASS' if passed else 'FAIL'} TP={len(tp)} FP={len(fp)} FN={len(fn)} {elapsed:.0f}ms",
                file=sys.stderr,
                flush=True,
            )
        except urllib.error.URLError as exc:
            counts["fn"] += len(expected)
            results.append({"id": case["id"], "text": text, "passed": False, "error": str(exc)})
            print(f"{case['id']}: HTTP ERROR", file=sys.stderr, flush=True)
    if not model_candidates:
        raise RuntimeError("No live model candidates were observed")
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0
    ordered = sorted(latencies)
    summary = {
        "cases": len(results),
        "passed": sum(case["passed"] for case in results),
        "http_errors": sum("error" in case for case in results),
        **counts,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0,
        "model_candidate_count": model_candidates,
        "latency_ms": {
            "median": round(statistics.median(latencies), 3),
            "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
            "mean": round(statistics.mean(latencies), 3),
        },
    }
    return {
        "suite": suite["suite"],
        "notes": suite["notes"],
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance,
        "summary": summary,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument("--output", required=True, help="JSON file or - for stdout")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    suite = json.loads(args.cases.read_text())
    report = run_suite(args.url.rstrip("/"), suite, args.expected_profile, args.timeout)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(encoded)
    else:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
