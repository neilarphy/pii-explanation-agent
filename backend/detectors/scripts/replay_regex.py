"""Evaluate regex changes on saved model candidates, without another model forward."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from backend.detectors.app.artifacts import MODEL_ID, MODEL_REVISION
from backend.detectors.app.config import DEFAULT_PROFILE_PATH, load_profile
from backend.detectors.app.observations import ModelMetadata, ModelObservation, ModelRun, Window
from backend.detectors.app.pipeline import build_response
from backend.detectors.app.regex_detector import RegexDetector
from backend.detectors.app.schemas import Timings
from backend.detectors.scripts.pii_bench import summarize


def replay_record(record, profile, detector):
    """Restore all model candidates, including rejected ones, then rerun regex/merge."""
    original = record.get("response")
    if original is None:
        raise ValueError("Cannot replay a failed request")
    trace = original["trace"]
    metadata = trace["model"]
    if (
        metadata["id"] != MODEL_ID
        or metadata["revision"] != MODEL_REVISION
        or tuple(metadata["labels"]) != profile.labels
        or metadata["emit_threshold"] != profile.emit_threshold
        or metadata["accept_threshold"] != profile.accept_threshold
        or tuple(tuple(g) for g in metadata.get("label_groups", [metadata["labels"]]))
        != profile.model_groups
        or tuple(
            tuple(g)
            for g in metadata.get(
                "schema_labels", metadata.get("label_groups", [metadata["labels"]])
            )
        )
        != profile.model_schemas
    ):
        raise ValueError("Cached model configuration differs from the selected profile")
    text = record["text"]
    if trace["text_length"] != len(text):
        raise ValueError("Cached text length mismatch")
    for entity in original["entities"]:
        if text[entity["start"] : entity["end"]] != entity["text"]:
            raise ValueError("Cached entity text mismatch")
    windows = tuple(Window(**w) for w in metadata["windows"])
    window_orders = {w.id: i for i, w in enumerate(windows)}
    observations = []
    for candidate in trace["candidates"]:
        if candidate["source"] != "model":
            continue
        if profile.public_type(candidate["raw_label"]) != candidate["type"]:
            raise ValueError("Cached model label mapping differs from the selected profile")
        observations.append(
            ModelObservation(
                candidate["raw_label"],
                candidate["type"],
                candidate["window_id"],
                window_orders[candidate["window_id"]],
                candidate["start"],
                candidate["end"],
                candidate["score"],
            )
        )
    model = ModelRun(
        tuple(observations),
        ModelMetadata(
            **{
                k: v
                for k, v in metadata.items()
                if k not in {"labels", "windows", "label_groups", "schema_labels"}
            },
            labels=tuple(metadata["labels"]),
            windows=windows,
            label_groups=profile.model_groups,
            schema_labels=profile.model_schemas,
        ),
    )
    response = build_response(
        text,
        model,
        detector.detect(text),
        profile,
        original["request_id"],
        Timings(queue=0, model=0, regex=0, merge=0, total=0),
    )
    return {
        **{k: record[k] for k in ("id", "domain", "split", "text", "gold")},
        "evaluation_mode": "cached_model_regex_replay",
        "response": response.model_dump(mode="json"),
        "client_ms": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile = load_profile(args.profile)
    detector = RegexDetector(profile)
    (args.output / "profile.toml").write_bytes(args.profile.read_bytes())
    source = args.responses.read_bytes()
    records = [
        replay_record(json.loads(line), profile, detector)
        for line in source.decode().splitlines()
        if line.strip()
    ]
    with (args.output / "responses.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_mode": "cached_model_regex_replay",
        "methodology": "Saved model candidates, scores and windows; fresh regex and merge. No HTTP/GPU run or latency measurement. Changes developed using this benchmark; not a held-out quality estimate.",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "splits": {
            split: summarize([r for r in records if r["split"] == split])
            for split in sorted({r["split"] for r in records})
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    for split, summary in report["splits"].items():
        print(split, {mode: values["micro"] for mode, values in summary["scores"].items()})


if __name__ == "__main__":
    main()
